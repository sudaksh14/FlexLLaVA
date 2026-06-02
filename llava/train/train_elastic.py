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

Usage (single GPU example; use your normal deepspeed/torchrun launcher for real
runs, passing the same args as M3's scripts/v1_5/*.sh):

    python -m llava.train.train_elastic \
        --model_name_or_path lmsys/vicuna-7b-v1.5 \
        --version v1 \
        --data_path ./playground/data/llava_v1_5_mix665k.json \
        --image_folder ./playground/data \
        --vision_tower openai/clip-vit-large-patch14-336 \
        --mm_projector_type mlp2x_gelu \
        --output_dir ./checkpoints/llava-elastic \
        --bf16 True --num_train_epochs 1 \
        --per_device_train_batch_size 8 --gradient_accumulation_steps 2 \
        --learning_rate 1e-4 --tf32 True --model_max_length 2048

Edit CONFIG below (or import and set llava.train.train.ELASTIC_CONFIG yourself)
to switch method / levels / losses.
"""

import llava.train.train as m3train
from llava.model.elastic import ElasticConfig


# ---- the experiment knob -------------------------------------------------
CONFIG = ElasticConfig(
    token_reduction="nested_query",      # "pooling" for the original M3 baseline
    tok_levels=[256, 144, 64, 16],       # visual-token budgets (L_tok)
    num_query_tokens=256,
    use_lora=True,                       # on by default for this approach
    lora_specialize_tok=True,            # adapter tuned per token budget
    lora_ranks=[8, 16, 32, 64],          # one rank per tok level (coarse->fine)
    use_prefix_kl=True, prefix_kl_weight=1.0,
    use_coral_align=True, coral_weight=0.1,
    use_nested_dropout=True,
    kl_teacher_tok_level=0,              # index of the full-token (teacher) level
    log_adapter_every=50,               # log LoRA adapter divergence every 50 steps
)


def main():
    m3train.ELASTIC_CONFIG = CONFIG
    m3train.train()


if __name__ == "__main__":
    main()
