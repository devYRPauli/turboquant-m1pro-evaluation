# TurboQuant Resume Plan

Date: 2026-03-27

This plan is based on the current local workspace state, not on idealized upstream claims.

## Goal

Resume from the last meaningful blocker with the fewest moving parts and the highest chance of producing a real fix.

## Best path to resume

Primary path: debug and validate the MLX QJL implementation.

Secondary path: use the `llama.cpp` fork only after the MLX side has a stable quality story or when a specific Metal bug needs validation.

## Why this is the best path

1. The algorithm itself is already validated in `turboquant_plus`.
2. The strongest local failure signal is the MLX long-context collapse.
3. The `llama.cpp` fork adds heavy Metal/JIT overhead on M1 Pro.
4. If QJL is conceptually wrong in the runtime path, fixing kernels first will not solve the main quality issue.

## Immediate restart checklist

### 1. Reconfirm the local baselines

Run these first so the current environment is verified before any new changes:

```bash
cd /Users/yashrajpandey/Desktop/TurboQuant
source .venv-mlx/bin/activate
python3 benchmarks/phase2_inference_compare.py
python3 benchmarks/phase3_long_context.py
```

Expected outcome from prior log:

* baseline MLX: coherent
* TurboQuant 4-bit MSE-only: degraded but usable on short prompt
* TurboQuant 4-bit with QJL: degenerate
* long-context TurboQuant: fails needle retrieval

### 2. Inspect the exact MLX QJL implementation

Focus on the installed `optiq` package implementation that performs:

* residual computation
* QJL projection
* dequantized residual reconstruction
* cache return into the attention path

The question to answer is:

* Is `mlx-optiq` reconstructing a noisy residual vector and feeding it back into standard SDPA, instead of performing the paper-style correction on `Q·K^T`?

This is the central hypothesis from the experiment log.

### 3. Build a minimal diagnostic around a single layer/head

Create a very small local experiment that compares:

* original KV vector
* MSE-only reconstruction
* MSE+QJL reconstruction
* resulting attention score error against a real query vector

Measure:

* vector cosine similarity
* residual norm
* attention-score error
* variance across repeated seeds

The point is to determine whether:

* vector reconstruction looks acceptable but attention still breaks, or
* QJL reconstruction itself is numerically too noisy for this model/head size

### 4. Test the most likely mitigation ideas in MLX space

In order of priority:

1. Apply damping to the QJL correction term.
2. Evaluate score-space correction instead of full residual-vector reconstruction.
3. Restrict QJL use to K only, not V, if the package currently applies it symmetrically.
4. Try per-layer or per-head gating to disable QJL where residual norms are unstable.

Do not start with large kernel or C++ changes before these are falsified.

## Concrete file targets

### High-priority reading

* `/Users/yashrajpandey/Desktop/TurboQuant/notebooks/experiment-log.md`
* `/Users/yashrajpandey/Desktop/TurboQuant/benchmarks/phase2_inference_compare.py`
* `/Users/yashrajpandey/Desktop/TurboQuant/benchmarks/phase3_long_context.py`
* `/Users/yashrajpandey/Desktop/TurboQuant/turboquant_plus/docs/context-scaling-deep-dive.md`
* `/Users/yashrajpandey/Desktop/TurboQuant/turboquant_plus/docs/decode-speed-status.md`

### Secondary reading

* `/Users/yashrajpandey/Desktop/TurboQuant/benchmarks/phase4_llama_cpp.py`
* `/Users/yashrajpandey/Desktop/TurboQuant/turboquant_plus/docs/pre-rotate-queries-investigation.md`

## What not to do first

1. Do not start by rebuilding `llama-cpp-turboquant` again on M1 Pro unless the task is specifically Metal-kernel validation.
2. Do not assume upstream speed improvements imply the local M1 Pro quality problem is already solved.
3. Do not spend time on memory wins alone until long-context quality is stable.

## Decision tree

### If MLX QJL can be stabilized

Then:

* rerun `benchmarks/phase2_inference_compare.py`
* rerun `benchmarks/phase3_long_context.py`
* verify needle retrieval at 2K, 4K, 8K, 16K
* only then revisit speed and memory tradeoffs

### If MLX QJL cannot be stabilized quickly

Then:

* pivot to documenting a clean issue report against the `optiq` implementation
* preserve the minimal reproducer
* optionally evaluate whether the `llama.cpp` fork’s later turbo3 improvements are enough for your use case on a different machine class

### If you must use the llama.cpp fork next

Start from validation, not new feature work:

1. confirm current branch and working tree
2. verify whether the GGML context-size fix is already present in code
3. warm the Metal JIT once and keep the binary stable
4. rerun only the shortest meaningful turbo3 benchmark first

## Practical next command sequence

```bash
cd /Users/yashrajpandey/Desktop/TurboQuant
source .venv-mlx/bin/activate
python3 benchmarks/phase2_inference_compare.py
python3 benchmarks/phase3_long_context.py
```

If those reproduce the known failures, the next work item should be:

* inspect and patch the local `optiq` QJL path, or
* create a local wrapper/fork that changes the correction strategy

## Expected deliverable for the next session

The next productive session should aim to produce one of these:

1. a minimal reproducible QJL failure case with quantitative evidence
2. a patched MLX QJL path that improves long-context retrieval
3. a clear stop/go decision that MLX is blocked and effort should move elsewhere
