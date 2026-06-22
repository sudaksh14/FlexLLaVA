"""lmms-eval model wrapper for FlexLLaVA elastic checkpoints.

Extends the existing M3-patched Llava wrapper with:
  1. attach_elastic_engine after load_pretrained_model
  2. Load elastic weights (resampler / projector / LoRA) from safetensors shards
  3. Accept tok_level (0-3 index into tok_levels) instead of a raw token count

Usage:
  accelerate launch --num_processes=1 -m lmms_eval \\
      --model llava_elastic \\
      --model_args pretrained=/path/to/elastic-pretrain,tok_level=0 \\
      --tasks mme,pope,mmbench_en_dev,scienceqa_img,textvqa_val,gqa \\
      --batch_size 1 \\
      --log_samples \\
      --output_path ./logs/

tok_level values (must match training config tok_levels=[256,144,64,16]):
  0 → 256 tokens  (teacher / full quality)
  1 → 144 tokens
  2 →  64 tokens
  3 →  16 tokens  (most compressed)
"""

import json
import logging
import os
from typing import List, Optional, Union

import torch
from lmms_eval.api.registry import register_model

from lmms_eval.models.llava import Llava

eval_logger = logging.getLogger("lmms-eval")

try:
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path
except ImportError:
    eval_logger.error("LLaVA is not installed.")


def _attach_elastic(model, ckpt_dir: str):
    """Read elastic_config.json, attach engine, load weights from shards."""
    cfg_path = os.path.join(ckpt_dir, "elastic_config.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(
            f"elastic_config.json not found at {cfg_path}. "
            "Re-run training or write it manually (see eval_elastic_job.sh)."
        )

    import dataclasses
    from llava.model.elastic import ElasticConfig, attach_elastic_engine

    with open(cfg_path) as f:
        raw = json.load(f)
    valid = {fd.name for fd in dataclasses.fields(ElasticConfig)}
    cfg = ElasticConfig(**{k: v for k, v in raw.items() if k in valid})

    attach_elastic_engine(model, cfg)

    # Load elastic weights that from_pretrained silently skips (not in base class)
    import glob as _glob
    try:
        from safetensors.torch import load_file as _sfload
        shards = sorted(_glob.glob(os.path.join(ckpt_dir, "model*.safetensors")))
        _loader = _sfload
    except ImportError:
        shards = sorted(_glob.glob(os.path.join(ckpt_dir, "pytorch_model*.bin")))
        _loader = lambda f: torch.load(f, map_location="cpu")

    elastic_sd = {}
    for f in shards:
        sd = _loader(f)
        for k, v in sd.items():
            if any(t in k for t in
                   ("elastic_resampler.", "elastic_projector.", ".lora_A", ".lora_B")):
                elastic_sd[k] = v

    if elastic_sd:
        missing, unexpected = model.load_state_dict(elastic_sd, strict=False)
        loaded = len(elastic_sd) - len(unexpected)
        eval_logger.info(f"[elastic] loaded {loaded}/{len(elastic_sd)} elastic keys "
                         f"(unexpected={len(unexpected)})")
    else:
        eval_logger.warning(f"[elastic] No elastic keys found in {ckpt_dir}")

    return cfg


@register_model("llava_elastic")
class LlavaElastic(Llava):
    """FlexLLaVA elastic eval model for lmms-eval.

    Adds `tok_level` arg (index 0-3 into tok_levels). Everything else
    (generate_until, loglikelihood, etc.) is inherited unchanged from Llava.
    """

    def __init__(
        self,
        pretrained: str,
        tok_level: int = 0,
        # inherited args — keep defaults matching Llava
        truncation: Optional[bool] = True,
        device: Optional[str] = "cuda",
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        model_name=None,
        attn_implementation=None,
        device_map="cuda:0",
        conv_template="vicuna_v1",
        use_cache=True,
        truncate_context=False,
        **kwargs,
    ) -> None:
        # Call grandparent (lmms base) directly — skip Llava.__init__ because
        # we need to intercept after load_pretrained_model to attach the engine.
        from lmms_eval.api.model import lmms as _lmms_base
        from accelerate import Accelerator, InitProcessGroupKwargs
        from datetime import timedelta
        import warnings
        warnings.filterwarnings("ignore")

        _lmms_base.__init__(self)
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        self.accelerator = accelerator

        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map

        model_name = model_name or get_model_name_from_path(pretrained)
        llava_model_args = {}
        if attn_implementation:
            llava_model_args["attn_implementation"] = attn_implementation

        try:
            self._tokenizer, self._model, self._image_processor, self._max_length = \
                load_pretrained_model(pretrained, None, model_name,
                                      device_map=self.device_map, **llava_model_args)
        except TypeError:
            self._tokenizer, self._model, self._image_processor, self._max_length = \
                load_pretrained_model(pretrained, None, model_name,
                                      device_map=self.device_map)

        # Attach elastic engine and load weights
        elastic_cfg = _attach_elastic(self._model, pretrained)

        # attach_elastic_engine creates new modules on CPU in float32, even when
        # the base model is on CUDA in float16. Cast device AND dtype together.
        _ref = next(self._model.model.embed_tokens.parameters())
        _dev, _dtype = _ref.device, _ref.dtype
        for _name in ("elastic_projector", "elastic_resampler"):
            _mod = getattr(self._model, _name, None)
            if _mod is not None:
                _mod.to(device=_dev, dtype=_dtype)
        # NestedLoRALinear: lora_A/lora_B are float32 CPU; base linear is float16 CUDA.
        _vt = self._model.get_vision_tower()
        if hasattr(_vt, "_lora_wrappers"):
            for _w in _vt._lora_wrappers:
                _w.to(device=_dev, dtype=_dtype)

        # Store tok_level; pass as matryoshka_vis_token_scale during generate
        self.tok_level = int(tok_level)
        self._model.config.matryoshka_vis_token_scale = self.tok_level
        n_tok = elastic_cfg.tok_levels[self.tok_level]
        eval_logger.info(f"[elastic] tok_level={self.tok_level} → {n_tok} visual tokens")

        self._config = self._model.config
        self.model.eval()
        self.model.tie_weights()

        self.truncation = truncation
        self.batch_size_per_gpu = int(batch_size)
        self.conv_template = conv_template
        self.use_cache = use_cache
        self.truncate_context = truncate_context

        if accelerator.num_processes > 1:
            from accelerate import DistributedType
            from accelerate.state import AcceleratorState
            try:
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(
                    must_match=True)
            except Exception:
                pass
            if accelerator.distributed_type in (
                DistributedType.FSDP, DistributedType.DEEPSPEED
            ):
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(
                    self.model, evaluation_mode=True)
            self._rank = accelerator.local_process_index
            self._world_size = accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1
