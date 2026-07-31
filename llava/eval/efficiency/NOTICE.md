# Third-party attribution & methodology notes

## Provenance

This package is a lean port of two MIT-licensed sources:

| Upstream | Files derived from it |
|---|---|
| [LLM-Viewer](https://github.com/hahnyuan/LLM-Viewer) — MIT, © 2024 Zhihang Yuan | `roofline.py`, `llama_config.py`, `hardware.py`, the roofline plumbing in `analyzer.py` |
| [AdaLLaVA](https://github.com/zhuoyan-xu/AdaLLaVA) (`src/adallava/eval/ada_analyzer.py`) — MIT | the per-layer OPs accounting and `analyze_*` methods in `analyzer.py` |

We ported rather than vendored wholesale: upstream LLM-Viewer ships a Vue
frontend, a web backend, and configs for architectures we do not train, and
its modules use bare top-level imports (`from hardwares.hardware_params
import ...`) that require `sys.path` manipulation to work. The ~700 lines we
actually need are reproduced here with package-relative imports.

Using AdaLLaVA's cost model is deliberate: it is the model behind the FLOPs /
prefill-time / memory numbers in their paper, so our tables can be placed
next to theirs on equal footing.

## Deliberate deviations from upstream

1. **Memory is a high-water mark, not a sum.** AdaLLaVA's
   `analyze_generate_task` does `memory_consumption += result["decode"]["memory_consumption"]`
   inside the decode loop, so the reported footprint grows linearly with the
   number of generated tokens. Memory is not additive over time. We take
   `max()` instead. This matters because we use the number to check whether a
   configuration fits in the Jetson Orin Nano's 8GB.

2. **Zero-traffic guard in the roofline.** Upstream divides by
   `memory_access` unguarded; we return a compute-bound result when a kernel
   moves no bytes instead of raising `ZeroDivisionError`.

3. **Config is passed in, not re-fetched.** Upstream's `ModelAnalyzer.__init__`
   calls `AutoConfig.from_pretrained(model_id)`, which re-resolves the
   checkpoint (and can hit the network). We take an already-loaded config
   object, so the analyzer works offline and on our `llava_llama` checkpoints.

4. **Added hardware profiles**: `jetson_orin_nano_8gb` (our deployment
   target) and `nvidia_A10` (what the cluster usually schedules us on);
   neither exists upstream.

## What the numbers mean — and don't

- **LLM only.** Like AdaLLaVA's, this models the language model. The frozen
  CLIP ViT-L/336 vision tower runs at full width at *every* token level, so
  it is a constant offset that cancels when comparing levels. For an absolute
  end-to-end figure add `vision_tower_gflops()` (~162 GFLOPs/image).

- **Analytic roofline, not wall clock.** These are lower bounds assuming
  perfect compute/memory overlap and no kernel-launch, host-copy, or thermal
  effects. On a real Jetson Orin Nano — especially in the 15W mode, sharing
  8GB LPDDR5 between CPU and GPU — measured latency will be higher. Use these
  to compare token levels and to screen for configurations that obviously
  cannot fit; measure on-device before quoting a deployment latency.

- **Dense throughput.** NVIDIA rates the Orin Nano 8GB at "40 TOPS", which is
  INT8 *with 2:4 sparsity*. `hardware.py` stores dense figures (20 TOPS INT8 /
  10 TFLOPS FP16), because a dense LLaMA forward gets no sparsity benefit.
