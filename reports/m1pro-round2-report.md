# TurboQuant Round 2 Report On M1 Pro

## 1. Purpose

This report records what was implemented, what failed, how each issue was debugged, how each issue was fixed, and how close the final result is to the claims in the TurboQuant paper and related community reports.

The target machine was an Apple M1 Pro with 16 GB unified memory on macOS Sequoia with Python 3.12 available.

## 2. Starting Point

The project started from a fresh repository at `/Users/yashrajpandey/Projects/turboquant-m1pro`.

The main implementation target was Aaryan Kapoor’s `llama.cpp` fork on branch `turboquant_tq3_0`.

The secondary target was TheTom’s `turboquant_plus` Python prototype for comparison and validation.

## 3. What Was Set Up

1. A new git repository was created for the evaluation run.
2. A README was added describing the project as a TurboQuant evaluation on M1 Pro 16 GB.
3. An experiment log was created and updated throughout the run with timestamps.
4. Ollama was checked and `qwen2.5:3b` was confirmed locally.
5. Aaryan’s fork was cloned, checked out on the TurboQuant branch, and built with Metal enabled.
6. The local Ollama GGUF for `qwen2.5:3b` was linked into the project.
7. TheTom’s Python prototype was cloned, installed, tested, and benchmarked.

## 4. First Major Result

The fresh setup worked.

The `q8_0` baseline on `llama-cli` worked immediately.

The first `tq3_0` sanity run did not work.

This separated the task into two parts.

1. Make TurboQuant actually run on M1 Pro.
2. Make TurboQuant produce correct output once it runs.

## 5. Errors We Faced

### 5.1 Metal `SET_ROWS` backend failure

The original requested TurboQuant run with Metal model offload and TurboQuant KV cache crashed before inference.

The concrete runtime error was a backend scheduling failure around `SET_ROWS` on `MTL0`.

This meant the Metal backend could not write TurboQuant KV rows into the cache buffer.

### 5.2 Missing Metal Flash Attention kernels

After isolating KV offload behavior, another failure appeared.

The Metal library did not contain `kernel_flash_attn_ext_vec_tq3_0_*`.

This meant even if the cache write path was bypassed, the Flash Attention path still lacked TurboQuant support.

### 5.3 CPU and Metal K cache correctness failure

When the run was forced through CPU only diagnostics, `tq3_0` still produced degraded text.

The important pattern was this:

1. `K = tq3_0, V = f16` produced bad output.
2. `K = f16, V = tq3_0` stayed coherent.

This showed the main correctness problem was in the K path, not the V path.

The same pattern remained after Metal support was added.

### 5.4 Zero norm blocks decoded as noise

The TurboQuant quantizer forced near zero blocks to use a scale of `1.0`.

That caused zero energy input blocks to decode into structured nonzero values.

This is especially harmful for key vectors because attention scores are highly sensitive to scale and direction noise.

### 5.5 Missing norm correction

The quantizer stored raw RMS in `block_tq3_0.d`.

After Lloyd Max quantization and inverse Walsh Hadamard reconstruction, the decoded block norm no longer matched the original block norm.

That distorted key vector magnitude and damaged query key dot products.

The implementation guide explicitly noted that norm correction had emerged as an important practical technique.

## 6. How Each Error Was Fixed

### 6.1 Metal support for TurboQuant was implemented

The following support was added in the Metal backend:

1. `tq3_0` quantize helper
2. `tq3_0` dequantize helper
3. `tq3_0` `SET_ROWS` kernels
4. `tq3_0` `FLASH_ATTN_EXT` kernel instantiations
5. `tq3_0` `FLASH_ATTN_EXT_VEC` kernel instantiations
6. `tq3_0` `GET_ROWS` and copy support
7. Metal allowlist entries so the backend would actually accept `GGML_TYPE_TQ3_0`

These changes were made mainly in:

`/Users/yashrajpandey/Projects/turboquant-m1pro/aaryan-llama-cpp/ggml/src/ggml-metal/ggml-metal.metal`

`/Users/yashrajpandey/Projects/turboquant-m1pro/aaryan-llama-cpp/ggml/src/ggml-metal/ggml-metal-device.m`

### 6.2 Temporary fail fast guard was removed

Before the Metal patch, a temporary fail fast guard had been added in `llama-context.cpp` so the binary would stop with a clear message rather than crashing.

After the Metal implementation was added, that guard was removed so the real code path could execute again.

### 6.3 Norm correction was added

The TurboQuant quantizer was changed so that it no longer stored raw RMS directly.

Instead it now:

1. Measures the original block norm
2. Quantizes and reconstructs the block
3. Measures the reconstructed norm
4. Stores `original_norm / reconstruction_norm` as the scale factor

This was implemented in both:

`/Users/yashrajpandey/Projects/turboquant-m1pro/aaryan-llama-cpp/ggml/src/ggml-quants.c`

`/Users/yashrajpandey/Projects/turboquant-m1pro/aaryan-llama-cpp/ggml/src/ggml-metal/ggml-metal.metal`

### 6.4 True zero block handling was added

Near zero blocks now produce:

1. `d = 0`
2. packed quant data set to zero

This prevents structured garbage from appearing when the source block should decode to zero.

## 7. What Worked After The Fixes

### 7.1 Phase 3 sanity prompt

The simple prompt asking for the capital of France became coherent in all relevant paths.

Metal `q8_0` remained correct.

Metal `tq3_0` became correct.

CPU only `tq3_0` also became correct.

This means both the execution blocker and the K path correctness blocker were fixed for the sanity test.

### 7.2 2K needle test

The ~2K needle in a haystack prompt passed for both `q8_0` and `tq3_0`.

Both outputs included the two required facts:

`FROSTBLOCK_7`

`VcMYB4`

Operationally, `tq3_0` was much slower than `q8_0`, but the retrieval result was correct.

### 7.3 4K needle test

The ~4K needle prompt also passed for both `q8_0` and `tq3_0`.

Again, both required facts were retrieved.

The `tq3_0` response was very slow, but it completed and answered correctly.

## 8. Comparison Against The Paper And Community Claims

### 8.1 What we successfully matched

1. We got TurboQuant running on Apple M1 Pro.
2. We got coherent `tq3_0` output on `llama.cpp`.
3. We got successful long context retrieval at around 2K and 4K prompt scales on the M1 Pro test model.
4. We verified that the core implementation path can be made operational on this machine.

### 8.2 What we did not fully match

We did not fully reproduce the stronger paper style or community style claims.

Specifically, we did not yet demonstrate:

1. paper style perplexity parity
2. broad benchmark parity across tasks
3. performance parity or near zero speed penalty
4. 8K retrieval success on this machine
5. full model scale validation beyond the local `qwen2.5:3b` setup
6. extensive temperature controlled output equivalence testing

### 8.3 Honest conclusion

The answer is partial success, not full replication.

We successfully implemented and debugged a working TurboQuant `tq3_0` path on M1 Pro.

We successfully tested it on sanity prompts and needle retrieval at about 2K and 4K.

We did not yet validate the full set of paper style claims.

Most importantly, speed behavior on this M1 Pro does not currently match the strongest public claims.

## 9. Performance Reality On This M1 Pro

Correctness is now much better than before.

Performance is still the weak area.

At 2K and especially 4K, `tq3_0` prompt processing was much slower than `q8_0`.

So the current state is:

1. correctness for the tested prompts is working
2. performance is not yet competitive on this host

This matters because the paper and later community reports are not only about compression and correctness. They also emphasize practical inference behavior.

## 10. Secondary Prototype Result

TheTom’s `turboquant_plus` prototype was also tested.

The Python test suite result was:

`532 passed`

`6 skipped`

`538 collected`

The demo completed successfully.

The real model validation script also completed.

However, that prototype path did not by itself solve the M1 Pro `llama.cpp` execution issue. The main successful implementation work happened in the Aaryan fork.

## 11. Final Status

The project is no longer blocked on M1 Pro compatibility.

TurboQuant `tq3_0` now runs and answers correctly on the tested setup.

Needle retrieval works at around 2K and 4K.

The main remaining gap is performance characterization and higher context stress testing, especially around 8K and beyond.

## 12. Recommended Next Steps

1. Run the 8K needle test as a pure stress and throughput check.
2. Measure perplexity against `q8_0` on a fixed validation corpus.
3. Compare prefill speed and generation speed across `f16`, `q8_0`, and `tq3_0`.
4. Profile why `tq3_0` prompt throughput is so much lower than `q8_0` on this machine.
5. Validate whether a more optimized TurboQuant key dot path is needed for Apple Silicon.

## 13. Files To Review

Primary report inputs:

`/Users/yashrajpandey/Projects/turboquant-m1pro/notebooks/experiment-log.md`

Sanity logs:

`/Users/yashrajpandey/Projects/turboquant-m1pro/notebooks/phase3-q8_0-single-turn.log`

`/Users/yashrajpandey/Projects/turboquant-m1pro/notebooks/phase3-tq3_0-single-turn.log`

`/Users/yashrajpandey/Projects/turboquant-m1pro/notebooks/phase3-tq3_0-ngl0-single-turn.log`

Needle logs:

`/Users/yashrajpandey/Projects/turboquant-m1pro/notebooks/needle-2k-q8_0.log`

`/Users/yashrajpandey/Projects/turboquant-m1pro/notebooks/needle-2k-tq3_0.log`

`/Users/yashrajpandey/Projects/turboquant-m1pro/notebooks/needle-4k-q8_0.log`

`/Users/yashrajpandey/Projects/turboquant-m1pro/notebooks/needle-4k-tq3_0.log`

Primary code changes:

`/Users/yashrajpandey/Projects/turboquant-m1pro/aaryan-llama-cpp/ggml/src/ggml-quants.c`

`/Users/yashrajpandey/Projects/turboquant-m1pro/aaryan-llama-cpp/ggml/src/ggml-metal/ggml-metal.metal`

`/Users/yashrajpandey/Projects/turboquant-m1pro/aaryan-llama-cpp/ggml/src/ggml-metal/ggml-metal-device.m`
