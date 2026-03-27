# TurboQuant Session Handoff

Date: 2026-03-27
Workspace: `/Users/yashrajpandey/Desktop/TurboQuant`

This folder captures the current project context and a concrete restart plan based on:

* `notebooks/experiment-log.md`
* top-level analysis/report docs
* benchmark scripts under `benchmarks/`
* docs and git history in `turboquant_plus/`
* git history in `llama-cpp-turboquant/`

## Project structure

* `turboquant_plus/`
  * Python reference implementation of TurboQuant
  * Algorithm validation, tests, docs, benchmark helpers
* `llama-cpp-turboquant/`
  * `llama.cpp` fork with TurboQuant KV cache integration
  * Metal/GGML implementation work
* `benchmarks/`
  * Local experiment scripts used in Phases 2-4
* `notebooks/experiment-log.md`
  * Main session log; this is the effective notebook/history file in this repo

## Confirmed state so far

### Phase 1: algorithm validation

* `turboquant_plus` installed and tested successfully on M1 Pro / Python 3.12.
* All 144 tests passed.
* Synthetic validation showed the expected Gaussianization effect after rotation.
* The core algorithm is not the blocker.

Primary references:

* `notebooks/experiment-log.md`
* `turboquant_plus/PLAN.md`

### Phase 2: MLX integration

* `mlx-optiq==0.0.1` was installed and used with `mlx-community/Qwen2.5-3B-Instruct-4bit`.
* Working config:
  * 4-bit
  * `use_qjl=False`
* Broken configs:
  * 3-bit MSE-only: degenerate output
  * 2-bit MSE-only: degenerate output
  * 4-bit with QJL: looping / collapse

Interpretation from the log:

* The QJL implementation in `mlx-optiq v0.0.1` appears mathematically unbiased in expectation but too noisy in reconstructed-vector form for real KV activations at `head_dim=128`.
* The library likely applies QJL as residual reconstruction instead of correcting the attention score path directly, which is where the paper gets its benefit.

Primary references:

* `notebooks/experiment-log.md`
* `benchmarks/phase2_inference_compare.py`

### Phase 3: long-context testing

* Benchmark runner compared:
  * Ollama baseline
  * MLX baseline
  * MLX TurboQuant 4-bit MSE-only
* Needle-in-a-haystack retrieval results:
  * Ollama baseline: passed
  * MLX baseline: passed
  * TurboQuant 4-bit MSE-only: failed at all tested context lengths

Interpretation:

* MSE-only compression is not stable enough for multi-thousand-token attention in the tested MLX path.
* This is the clearest quality blocker in the local experiments.

Primary references:

* `notebooks/experiment-log.md`
* `benchmarks/phase3_long_context.py`
* `benchmarks/phase3_results.json`

### Phase 4: llama.cpp fork

* A real bug was found in `llama-cpp-turboquant`:
  * GGML context metadata sizing did not account for the two shared TurboQuant rotation tensors.
* That bug was fixed locally during investigation and documented in the experiment log.
* After the fix:
  * `turbo3` initialized and ran on M1 Pro
  * output quality was still poor / garbled
  * `turbo4` was blocked by missing non-vec Metal prefill kernel coverage

Additional practical blocker:

* Metal JIT recompilation on M1 Pro made iteration very slow after each rebuild.

Primary references:

* `notebooks/experiment-log.md`
* `benchmarks/phase4_llama_cpp.py`

## Important distinction

The local experiment log reflects the state of the M1 Pro investigation.

The upstream repos have continued beyond that point:

* `turboquant_plus/docs/context-scaling-deep-dive.md`
* `turboquant_plus/docs/decode-speed-status.md`

Those docs show later progress on speed/context behavior, especially around turbo3 dequant optimization, but they do not erase the local blockers recorded here:

* broken MLX QJL path
* poor quality in the tested local TurboQuant inference paths
* M1 Pro iteration cost in the Metal fork

## Where the last session actually got stuck

The last session was blocked in practice by inference quality and iteration speed, not by lack of understanding.

1. MLX path:
   * `use_qjl=True` was unusable.
   * `use_qjl=False` was only tolerable at short prompts and failed on long-context retrieval.

2. llama.cpp fork:
   * startup crash root cause was fixed
   * `turbo3` still produced poor output quality
   * `turbo4` prefill path was missing kernel coverage
   * Metal JIT rebuild cycles were too slow to keep iterating productively on M1 Pro

## Recommended direction

The best next step is to treat the MLX QJL issue as the main technical blocker and use the `llama.cpp` fork only as a secondary path.

Why:

* The Python/MLX path is easier to iterate on.
* The core failure mode is already isolated.
* The fork has additional Metal-specific engineering drag on M1 Pro.

Read `resume-plan.md` in this folder for the exact restart sequence.
