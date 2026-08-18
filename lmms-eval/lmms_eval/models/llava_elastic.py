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

import contextlib
import json
import logging
import os
from typing import List, Optional, Union

import torch
from lmms_eval.api.instance import Instance
from lmms_eval.api.registry import register_model

from lmms_eval.models.llava import Llava

eval_logger = logging.getLogger("lmms-eval")

try:
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path
    from llava.constants import IMAGE_TOKEN_INDEX
except ImportError:
    eval_logger.error("LLaVA is not installed.")


def _attach_elastic(model, ckpt_dir: str):
    """Read elastic_config.json, attach engine, load weights from shards."""
    cfg_path = os.path.join(ckpt_dir, "elastic_config.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(
            f"elastic_config.json not found at {cfg_path}. "
            "train.py writes it at the end of every elastic run; if this is an "
            "older checkpoint, copy the file from a newer run with the same "
            "tok_levels/lora_ranks."
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
    # Pick shards by what the checkpoint ACTUALLY contains, in HF's own
    # precedence (safetensors, then .bin) -- not by which library imports.
    # Selecting on `import safetensors` succeeding made every .bin checkpoint
    # glob an empty list and load zero elastic weights: the LoRA save path in
    # train.py writes pytorch_model.bin, so all llava-elastic-finetune-v3 evals
    # ran with a randomly initialised projector and no resampler at all.
    shards = sorted(_glob.glob(os.path.join(ckpt_dir, "model*.safetensors")))
    if not shards:
        shards = sorted(_glob.glob(os.path.join(ckpt_dir, "pytorch_model*.bin")))

    def _load_shard(path):
        if path.endswith(".safetensors"):
            from safetensors.torch import load_file
            return load_file(path)
        return torch.load(path, map_location="cpu")

    elastic_sd = {}
    for f in shards:
        sd = _load_shard(f)
        for k, v in sd.items():
            if any(t in k for t in
                   ("elastic_resampler.", "elastic_projector.", ".lora_A", ".lora_B")):
                elastic_sd[k] = v

    if not elastic_sd:
        # A checkpoint with elastic_config.json but no elastic tensors cannot be
        # evaluated -- the resampler/projector would stay at random init and every
        # score is noise. Fail loudly instead of producing a plausible-looking 0.
        raise RuntimeError(
            f"[elastic] No elastic weights found in {ckpt_dir} "
            f"(scanned {len(shards)} shard(s): {[os.path.basename(s) for s in shards] or 'none'}). "
            "elastic_config.json is present, so this checkpoint should contain "
            "elastic_resampler.*/elastic_projector.* tensors. Evaluating without "
            "them yields a randomly initialised projector, not a real score."
        )

    missing, unexpected = model.load_state_dict(elastic_sd, strict=False)
    loaded = len(elastic_sd) - len(unexpected)
    eval_logger.info(f"[elastic] loaded {loaded}/{len(elastic_sd)} elastic keys "
                     f"(unexpected={len(unexpected)})")
    if loaded == 0:
        raise RuntimeError(
            f"[elastic] {len(elastic_sd)} elastic tensors were found in {ckpt_dir} "
            "but none matched a parameter in the attached engine -- the key names "
            "have drifted from what attach_elastic_engine() builds."
        )

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
        eff_hardware: str = "jetson_orin_nano_8gb",
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

        # ---- analytic efficiency model -------------------------------------
        # n_visual_tokens is what our elasticity axis actually changes, and it
        # lands directly in the LLM's prompt length, so the cost model needs it
        # to convert an input_ids length into a real sequence length.
        self.n_visual_tokens = int(n_tok)
        self._eff_records = {}   # task_name -> list of per-sample metric dicts
        self._analyzer = None
        try:
            from llava.eval.efficiency import ElasticAnalyzer
            self._analyzer = ElasticAnalyzer(self._model.config, eff_hardware)
            self.eff_hardware = eff_hardware
            eval_logger.info(
                f"[elastic] efficiency model on {eff_hardware}: analytic roofline, "
                f"LLM only (frozen vision tower is a constant offset across levels)")
        except Exception as e:
            # Efficiency metrics are strictly additive -- never let them break
            # an accuracy run.
            eval_logger.warning(f"[elastic] efficiency model disabled: {e}")

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

    # ------------------------------------------------------------------
    # Analytic efficiency instrumentation
    # ------------------------------------------------------------------
    def _seq_lengths(self, input_ids, attention_mask):
        """True post-expansion LLM sequence length for each row of a batch.

        input_ids still carries IMAGE_TOKEN_INDEX as a *single* placeholder
        per image; prepare_inputs_labels_for_multimodal later expands each one
        into n_visual_tokens embeddings. The cost model needs the expanded
        length, since that -- not the tokenizer output -- is what the LLM
        actually attends over, and it is precisely what our elasticity axis
        changes.
        """
        lengths = []
        for i in range(input_ids.shape[0]):
            row = input_ids[i]
            if attention_mask is not None:
                true_len = int(attention_mask[i].sum().item())
                row = row[attention_mask[i].bool()]
            else:
                true_len = int(row.numel())
            n_img = int((row == IMAGE_TOKEN_INDEX).sum().item())
            lengths.append(true_len - n_img + n_img * self.n_visual_tokens)
        return lengths

    @contextlib.contextmanager
    def _record_generate(self, sink):
        """Temporarily wrap model.generate to log (prompt_len, gen_len).

        Patch the object the parent actually calls: `self.model` unwraps
        through accelerate, so it is not necessarily `self._model`. Patching
        the wrong one silently records nothing.
        """
        target = self.model
        original = target.generate

        def wrapped(*args, **kwargs):
            out = original(*args, **kwargs)
            try:
                input_ids = args[0] if args else kwargs.get("input_ids")
                attn = kwargs.get("attention_mask")
                pad_id = kwargs.get("pad_token_id")
                prompt_lens = self._seq_lengths(input_ids, attn)
                cont = out
                for i, plen in enumerate(prompt_lens):
                    if isinstance(cont, torch.Tensor) and i < cont.shape[0]:
                        row = cont[i]
                        gen_len = int((row != pad_id).sum().item()) if pad_id is not None else int(row.numel())
                    else:
                        gen_len = 1
                    sink.append((plen, max(gen_len, 1)))
            except Exception as e:  # never break generation for a metric
                eval_logger.warning(f"[elastic] efficiency capture skipped: {e}")
            return out

        target.generate = wrapped
        try:
            yield
        finally:
            target.generate = original

    def _cost(self, prompt_len, gen_len):
        """Memoised roofline cost. The analyzer walks every layer for every
        decode step, so re-running it per sample would cost more CPU than the
        eval itself; (prompt_len, gen_len) fully determines the result."""
        key = (prompt_len, gen_len)
        cache = getattr(self, "_cost_cache", None)
        if cache is None:
            cache = self._cost_cache = {}
        if key not in cache:
            cache[key] = self._analyzer.analyze_generate_task(
                prompt_len=prompt_len, gen_len=gen_len, batchsize=1,
                w_bit=16, a_bit=16, kv_bit=16, use_flashattention=False)
        return cache[key]

    def generate_until(self, requests: List[Instance]) -> List[str]:
        if self._analyzer is None:
            return super().generate_until(requests)

        # Group by task so each parent call covers exactly one task, which is
        # what lets us attribute per-sample cost to the right benchmark. The
        # parent returns results positionally, so re-scattering to the
        # original indices keeps the contract identical to an ungrouped call.
        groups = {}
        for pos, req in enumerate(requests):
            groups.setdefault(req.args[4], []).append((pos, req))

        out: List[Optional[str]] = [None] * len(requests)
        for task_name, items in groups.items():
            captured = []
            with self._record_generate(captured):
                sub_res = super().generate_until([r for _, r in items])
            for (pos, _), text in zip(items, sub_res):
                out[pos] = text
            records = self._eff_records.setdefault(task_name, [])
            for prompt_len, gen_len in captured:
                cost = self._cost(prompt_len, gen_len)
                records.append({
                    "prompt_len": prompt_len,
                    "gen_len": gen_len,
                    **{k: cost[k] for k in (
                        "flops", "avg_flops", "prefill_flops", "prefill_time",
                        "total_time", "memory_consumption",
                        "prefill_memory_consumption")},
                })
        return out

    def efficiency_summary(self):
        """Mean efficiency metrics per task, plus deployment feasibility."""
        from llava.eval.efficiency import EFFICIENCY_METRICS

        summary = {}
        for task_name, records in self._eff_records.items():
            if not records:
                continue
            agg = {m: sum(r[m] for r in records) / len(records)
                   for m in EFFICIENCY_METRICS if m in records[0]}
            agg["total_time"] = sum(r["total_time"] for r in records) / len(records)
            agg["mean_prompt_len"] = sum(r["prompt_len"] for r in records) / len(records)
            agg["samples"] = len(records)
            # Peak, not mean, is what decides whether the board OOMs.
            peak_mem = max(r["memory_consumption"] for r in records)
            agg["peak_memory_consumption"] = peak_mem
            agg["fits_on_target"] = self._analyzer.fits_in_memory(peak_mem)
            agg["n_visual_tokens"] = self.n_visual_tokens
            agg["hardware"] = self.eff_hardware
            summary[task_name] = agg
        return summary
