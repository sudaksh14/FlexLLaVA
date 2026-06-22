"""Launch elastic / adaptive-matryoshka training.

This is a thin wrapper over M3's existing `llava.train.train.train()`. It does
exactly one extra thing: it sets the module-level `ELASTIC_CONFIG`, which the
attach hook inside `train()` reads to:
  1. build an ElasticEngine and attach it to the model (`attach_elastic_engine`),
     injecting rank-nested LoRA into the (frozen) vision tower if use_lora;
  2. mark ONLY the elastic modules trainable (resampler, projector, LoRA factors)
     so the CLIP/SigLIP backbone and the LLM stay frozen;
  3. let the elastic forward branch run the L_tok grid each step with the
     LM loss + prefix-KL (coarse->fine) + CORAL (latent stability) terms.

Everything else -- data module, tokenizer, DeepSpeed, checkpointing -- is M3's
unchanged machinery, so all the usual LLaVA training flags still apply.

Elastic-specific args (parsed here, stripped before HfArgumentParser sees argv):
  --tok_levels        INT [INT ...]  visual-token budgets per level, descending
                                     e.g. --tok_levels 256 144 64 16
  --lora_ranks        INT [INT ...]  LoRA rank per tok level (must match len)
                                     e.g. --lora_ranks 8 16 32 64
  --prefix_kl_weight  FLOAT          weight on the prefix-KL distillation term (default 1.0)
  --coral_weight      FLOAT          weight on the CORAL token-alignment term (default 0.1)

All other flags are standard LLaVA training args.

Usage example:
    deepspeed llava/train/train_elastic.py \\
        --tok_levels 256 144 64 16 \\
        --lora_ranks 8 16 32 64 \\
        --model_name_or_path lmsys/vicuna-7b-v1.5 \\
        --version plain \\
        --data_path ./playground/data/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \\
        --image_folder ./playground/data/LLaVA-Pretrain/images \\
        --vision_tower openai/clip-vit-large-patch14-336 \\
        --mm_projector_type mlp2x_gelu \\
        --freeze_backbone True \\
        --output_dir ./checkpoints/llava-elastic-pretrain \\
        --bf16 True --num_train_epochs 1 \\
        --per_device_train_batch_size 32 --gradient_accumulation_steps 1 \\
        --learning_rate 1e-3 --tf32 True --model_max_length 2048
"""

import argparse
import sys

import llava.train.train as m3train
from llava.model.elastic import ElasticConfig


# ---- defaults (used when args are not passed on the CLI) -----------------
_DEFAULT_TOK_LEVELS      = [256, 144, 64, 16]
_DEFAULT_LORA_RANKS      = [8, 16, 32, 64]
_DEFAULT_KL_WEIGHT       = 1.0
_DEFAULT_CORAL_WEIGHT    = 0.01


def _parse_elastic_args():
    """Pre-parse only the elastic-specific flags, leaving everything else in
    sys.argv for HfArgumentParser inside m3train.train()."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--tok_levels", type=int, nargs="+", default=_DEFAULT_TOK_LEVELS,
                   help="Visual-token budgets per level, descending (e.g. 256 144 64 16).")
    p.add_argument("--lora_ranks", type=int, nargs="+", default=_DEFAULT_LORA_RANKS,
                   help="LoRA rank per tok level — must have same length as tok_levels.")
    p.add_argument("--prefix_kl_weight", type=float, default=_DEFAULT_KL_WEIGHT,
                   help="Weight on the prefix-KL self-distillation loss term.")
    p.add_argument("--coral_weight", type=float, default=_DEFAULT_CORAL_WEIGHT,
                   help="Weight on the CORAL token-alignment loss term.")
    elastic_args, remaining = p.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining  # hide elastic flags from HfArgumentParser
    return elastic_args


def main():
    elastic_args = _parse_elastic_args()

    tok_levels   = elastic_args.tok_levels
    lora_ranks   = elastic_args.lora_ranks

    if len(lora_ranks) != len(tok_levels):
        raise ValueError(
            f"--lora_ranks length ({len(lora_ranks)}) must equal "
            f"--tok_levels length ({len(tok_levels)})"
        )

    m3train.ELASTIC_CONFIG = ElasticConfig(
        token_reduction="nested_query",
        tok_levels=tok_levels,
        num_query_tokens=tok_levels[0],   # full query bank = largest level
        use_lora=True,
        lora_specialize_tok=True,
        lora_ranks=lora_ranks,
        lora_alpha=1.0,
        use_prefix_kl=True,  prefix_kl_weight=elastic_args.prefix_kl_weight,
        use_coral_align=True, coral_weight=elastic_args.coral_weight,
        use_nested_dropout=True,
        kl_teacher_tok_level=0,           # largest tok level is teacher
        log_adapter_every=50,
    )

    m3train.train(attn_implementation="flash_attention_2")


if __name__ == "__main__":
    main()
