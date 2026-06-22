"""Elastic-aware VQA eval loader.

Drop-in replacement for model_vqa_loader.py that:
  1. Loads the checkpoint with load_pretrained_model
  2. Reads elastic_config.json from the checkpoint dir to rebuild ElasticConfig
  3. Calls attach_elastic_engine so the forward path activates
  4. Loads elastic_resampler / elastic_projector / LoRA weights from the
     safetensors shards (they are skipped by from_pretrained because they are
     not part of the base LlavaLlamaForCausalLM class definition)
  5. Passes --tok-level (index 0-3 into tok_levels) as matryoshka_vis_token_scale

Token level index mapping (must match training config):
  0 → tok_levels[0]  (e.g. 256 tokens – teacher / highest quality)
  1 → tok_levels[1]  (e.g. 144 tokens)
  2 → tok_levels[2]  (e.g.  64 tokens)
  3 → tok_levels[3]  (e.g.  16 tokens – most compressed)
"""

import argparse
import json
import os

import shortuuid
import torch
from tqdm import tqdm

from llava.constants import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                              DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)
from llava.conversation import conv_templates
from llava.eval.model_vqa_loader import (CustomDataset, create_data_loader,
                                         get_chunk)
from llava.mm_utils import (get_model_name_from_path, process_images,
                             tokenizer_image_token)
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init


def _load_elastic_weights(model, ckpt_dir: str):
    """Warm-start elastic_resampler, elastic_projector, and LoRA A/B from the
    saved safetensors shards.  HF from_pretrained silently skips these keys."""
    import glob as _glob
    from safetensors.torch import load_file

    shards = sorted(_glob.glob(os.path.join(ckpt_dir, "model*.safetensors")))
    if not shards:
        shards = sorted(_glob.glob(os.path.join(ckpt_dir, "pytorch_model*.bin")))
        _loader = lambda f: torch.load(f, map_location="cpu")
    else:
        _loader = load_file

    elastic_sd = {}
    for f in shards:
        sd = _loader(f)
        for k, v in sd.items():
            if any(t in k for t in
                   ("elastic_resampler.", "elastic_projector.", ".lora_A", ".lora_B")):
                elastic_sd[k] = v

    if not elastic_sd:
        print(f"[elastic-eval] WARNING: no elastic keys found in {ckpt_dir}")
        return

    missing, unexpected = model.load_state_dict(elastic_sd, strict=False)
    loaded = len(elastic_sd) - len(unexpected)
    elastic_missing = [k for k in missing if any(
        t in k for t in ("elastic_resampler", "elastic_projector", "lora_A", "lora_B"))]
    print(f"[elastic-eval] loaded {loaded}/{len(elastic_sd)} elastic keys; "
          f"{len(elastic_missing)} missing, {len(unexpected)} unexpected")


def load_elastic_model(model_path: str, model_base=None, device="cuda"):
    """Load a FlexLLaVA elastic checkpoint for inference."""
    disable_torch_init()
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, model_base, model_name,
        device_map=device,
    )

    # Read the elastic config saved alongside the checkpoint.
    cfg_path = os.path.join(model_path, "elastic_config.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(
            f"elastic_config.json not found at {cfg_path}. "
            "Re-run training (it is now saved automatically at epoch end) "
            "or write it manually."
        )
    with open(cfg_path) as f:
        cfg_dict = json.load(f)

    from llava.model.elastic import ElasticConfig, attach_elastic_engine
    import dataclasses
    # Filter to valid fields only (future-proof against extra keys in json)
    valid = {f.name for f in dataclasses.fields(ElasticConfig)}
    elastic_cfg = ElasticConfig(**{k: v for k, v in cfg_dict.items() if k in valid})

    attach_elastic_engine(model, elastic_cfg)
    _load_elastic_weights(model, model_path)

    model.eval()
    return tokenizer, model, image_processor, context_len


def eval_model(args):
    tokenizer, model, image_processor, context_len = load_elastic_model(
        args.model_path, getattr(args, "model_base", None)
    )

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file))]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    data_loader = create_data_loader(
        questions, args.image_folder, tokenizer, image_processor, model.config
    )

    tok_level = args.tok_level  # index 0-3 into tok_levels

    for (input_ids, image_tensor, image_sizes), line in tqdm(
        zip(data_loader, questions), total=len(questions)
    ):
        idx = line["question_id"]
        cur_prompt = line["text"]

        input_ids = input_ids.to(device="cuda", non_blocking=True)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.to(dtype=torch.float16, device="cuda",
                                        non_blocking=True),
                image_sizes=image_sizes,
                do_sample=False,
                temperature=0,
                top_p=None,
                num_beams=1,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                matryoshka_vis_token_scale=tok_level,
            )

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        ans_file.write(json.dumps({
            "question_id": idx,
            "prompt": cur_prompt,
            "text": outputs,
            "answer_id": shortuuid.uuid(),
            "model_id": f"elastic-tok{model.elastic_engine.cfg.tok_levels[tok_level]}",
            "metadata": {"tok_level": tok_level,
                         "n_tokens": model.elastic_engine.cfg.tok_levels[tok_level]},
        }) + "\n")

    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to elastic checkpoint dir (contains elastic_config.json)")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--answers-file", type=str, required=True)
    parser.add_argument("--tok-level", type=int, default=0,
                        help="Token level index (0=256 tok, 1=144, 2=64, 3=16)")
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    eval_model(args)
