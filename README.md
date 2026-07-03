# TurboQuant Evaluation on Apple M1 Pro 16GB

This repository documents a two-round implementation and evaluation of TurboQuant, a KV cache compression algorithm for large language models, running on an Apple M1 Pro MacBook Pro with 16GB unified memory. It contains all experiment logs, benchmark scripts, debug logs, reports, and the specific code fixes that resolved a complete failure of long-context retrieval.

## What TurboQuant Is

TurboQuant (arXiv 2504.19874, ICLR 2026) compresses the key-value cache that transformer models maintain during inference. It does not touch model weights. Its purpose is to reduce memory consumption at runtime so that longer contexts fit within a fixed memory budget.

The algorithm works in two stages:

1. PolarQuant: A random orthogonal rotation transforms the KV vectors so that their coordinate distribution becomes approximately Gaussian. Scalar quantization is then near-optimal on each coordinate independently, using precomputed Lloyd-Max centroids. No per-block normalization constants are required.

2. QJL (Quantized Johnson-Lindenstrauss): A 1-bit sign projection of the quantization residual provides an unbiased inner-product correction. This eliminates the systematic bias that accumulates in the attention scores when using compressed keys.

The paper claims 3.5-bit quantization produces quality neutral results while compressing the KV cache by at least 4.5x, requiring no training or calibration.

The paper authors are Amir Zandieh (Google Research), Majid Daliri (New York University), Majid Hadian (Google DeepMind), and Vahab Mirrokni (Google Research). The paper reference is: TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate, arXiv 2504.19874.

## What This Repository Contains

```
reports/                        All experiment logs and reports
    round1-experiment-log.md    Round 1 full log (Phases 1 through 4, QJL fix)
    round2-experiment-log.md    Round 2 full log (Aaryan fork, Metal patches)
    m1pro-round2-report.md      Round 2 structured summary report
    final-validation-report.md  Final validation with 100% at 16K result
    round1-post-mortem-report.md   Root cause analysis and QJL fix description
    round1-comprehensive-analysis.md   Phase-by-phase technical analysis
    round1-executive-report.md  High-level summary for Round 1
    round2-benchmark-results.md Upstream turboquant_plus benchmark results (M5 Max reference)
    round2-project-readme.md    Round 2 project scope and execution rules
    round1-fresh-run-status.md  Session handoff status notes
    round1-session-handoff-readme.md  Session handoff context document
logs/                           All llama-cli debug and test run logs, needle prompts
    needle-2k-prompt.txt        Exact 2K context needle prompt used in Round 2 tests
    needle-4k-prompt.txt        Exact 4K context needle prompt used in Round 2 tests
    needle-8k-prompt.txt        Exact 8K context needle prompt used in Round 2 tests
    needle-template.txt         Needle prompt template
    niah-results/               NIAH result files from upstream turboquant_plus M5 Max reference runs (Qwen3.5-35B-A3B), not local M1 Pro results
benchmarks/                     All benchmark scripts
    build_prompt.py             Prompt generator with embedded needle
    phase2_inference_compare.py MLX inference comparison script
    phase3_long_context.py      Long-context needle-in-haystack benchmark
    phase4_llama_cpp.py         llama.cpp fork benchmarking script
    stable_long_context_benchmark.py   Stable rerun benchmark
    test_hybrid_needle.py       Hybrid K5/V4 needle test (the script that reached 100%)
    phase3_results.json         Raw Phase 3 result data
patches/
    key-fixes.md                Prose description of all five concrete code fixes
    round2-norm-correction-ggml-quants.patch      Actual diff: norm correction and zero block fix
    round2-metal-tq3-kernels.patch                Actual diff: Metal tq3_0 kernel additions
    round2-metal-device-allowlist.patch           Actual diff: Metal device allowlist for tq3_0
    round1-ggml-context-sizing-llama-kv-cache.patch  Actual diff: GGML context sizing fix
    round1-metal-modifications.patch              Vestigial whitespace-only diff (Round 1 kernel work was reverted)
guides/
    implementation-guide.md     Round 1 full implementation guide
    round2-guide.md             Round 2 guide (Aaryan fork, fresh start)
    round1-execution-checklist.md  Step-by-step rerun checklist from Round 1 handoff
    round1-resume-plan.md       Resume plan and decision tree from Round 1 handoff
```

One log file, `logs/phase3-q8_0.log` (137 MB of raw multi-turn llama-cli output), exceeds GitHub's file size limit and is excluded from the published repository via .gitignore. All other logs are included as captured.

## Hardware and Environment

* Machine: MacBook Pro, Apple M1 Pro chip, 16GB unified memory, 512GB SSD
* OS: macOS 26.3.1 and 26.4 across the two rounds (the version 26 line is macOS Tahoe; the source logs label it Sequoia, which is the version 15 line, and that label is preserved in the copied reports)
* Python: 3.12.13 (via Homebrew)
* Model tested: Qwen2.5-3B-Instruct (Q4\_K\_M GGUF via Ollama, and 4-bit MLX via Hugging Face)
* Round 1 implementations: mlx-optiq v0.0.1 (MLX inference), TheTom turboquant\_plus Python prototype, TheTom llama-cpp-turboquant fork
* Round 2 implementation: Aaryan Kapoor llama.cpp fork, branch turboquant-tq3\_0, TheTom turboquant\_plus (updated)

## Key Findings Summary

Stock TurboQuant implementations achieved 0% needle retrieval at all tested context lengths. After identifying and applying five fixes, the Hybrid K5/V4 configuration achieved 100% retrieval accuracy at 16K tokens using the MLX path, and the Aaryan Kapoor llama.cpp fork achieved correct output on sanity prompts and at 2K and 4K needle tests.

The five fixes are:

1. QJL orthogonal projection: the paper defines the QJL projection as a random Gaussian matrix (Definition 1 in the paper), and the stock implementations followed it faithfully. That construction is unbiased, but empirically it destroyed generation at head dimension 128 (immediate word-loop degeneration). Replacing the Gaussian matrix with a random orthogonal matrix from QR decomposition, a variance-reducing modification of the paper's design, eliminated the degeneration.

2. QJL dequantization scale factor: the scale must match the projection matrix. The paper's `sqrt(pi/2) / d` is correct for a Gaussian matrix, whose rows have norm near `sqrt(d)`. Once the matrix is orthogonal (change 1), the matching scale becomes `sqrt(pi/2) / sqrt(d)`; keeping the Gaussian scale with an orthogonal matrix would make the correction about 11x too small at d=128. The two changes are one coupled substitution, not two independent bug fixes.

3. Hybrid K5/V4 configuration: even with correct QJL math, keys are more sensitive to quantization noise than values because attention scores depend on precise key-query inner products. Assigning 5 bits to keys (4-bit MSE plus 1-bit QJL) and 4 bits to values (4-bit MSE only) provided the necessary precision.

4. GGML context sizing bug: the TheTom llama-cpp-turboquant fork crashed on initialization due to a metadata context allocation formula that did not count the two shared rotation matrix tensors. Adding two extra `ggml_tensor_overhead()` slots fixed the crash. This was diagnosed as a software bug, not a hardware incompatibility.

5. Norm correction and zero block handling: the tq3\_0 quantizer in Aaryan's fork stored raw RMS as the scale factor, but the correct value is `original_norm / reconstruction_norm` to account for norm change during Lloyd-Max quantization. Near-zero blocks also decoded as structured noise because the guard set scale to 1.0 rather than emitting a true zero block.

In addition to these fixes, the validated MLX configuration applies a damping factor of 0.7 to the QJL correction term (see `benchmarks/test_hybrid_needle.py`). This is close to the MMSE-optimal shrinkage `2/pi` (approximately 0.6366) that was later formalized during upstream review of pull request 93. A reproduction that omits the damping factor is not running the validated configuration. The projection, scale, and damping changes were applied together and were not ablated individually.

## Results Table

| Configuration | Context | Needle Retrieval |
|---|---|---|
| Ollama baseline (q8\_0) | 2K, 4K, 8K, 16K | 100% all lengths |
| MLX baseline (FP16 KV) | 2K, 4K, 8K, 16K | 100% all lengths |
| MLX TurboQuant MSE-only 4-bit (stock) | 2K, 4K, 8K, 16K | 0% all lengths |
| MLX TurboQuant with QJL (stock, paper-faithful Gaussian) | any | Degenerate (word loops) |
| MLX Hybrid K5/V4 (orthogonal QJL, matched scale, 0.7 damping) | 4K, 8K, 16K | 100% |
| MLX Hybrid K5/V4 (orthogonal QJL, matched scale, 0.7 damping) | 2K | 50% in the recorded run (see note) |
| Aaryan fork tq3\_0 CPU, before norm fix | any | Degenerate (repetitive) |
| Aaryan fork tq3\_0 after all fixes | 2K, 4K sanity and needle | Correct (both facts retrieved) |

Note on the 2K Hybrid entry: the recorded raw data (`benchmarks/phase3_results.json`) shows a needle score of 0.5 at 2K for the Hybrid configuration: the model retrieved VcMYB4 but wrote "FROSTst7" instead of "FROSTBLOCK-7". The round 1 post-mortem report asserts 100% at 2K in its summary but lists only 4K, 8K, and 16K in its detailed fix section, and no raw output supporting a 100% run at 2K exists in this repository. The table therefore reports the raw data.

## Speed and Memory Reference

At 16K tokens with qwen2.5:3b on M1 Pro 16GB:

* Ollama q8\_0: approximately 36.5 tokens per second
* MLX baseline FP16: approximately 2.0 tokens per second
* MLX Hybrid K5/V4 TurboQuant: approximately 1.1 tokens per second

KV memory savings at 16K tokens:

* FP16 baseline: 562 MB
* TurboQuant 4-bit: 140 MB (4.0x compression confirmed)

The speed penalty in the MLX path is due to unoptimized Python-level dequantization including full 128x128 matrix multiplications. The memory savings are real and theoretically 4x or greater.

## Upstream Contributions

The five fixes documented above have been published back upstream where appropriate:

* QJL orthogonal projection and `sqrt(d)` scale factor: merged into TheTom turboquant\_plus main on 2026-05-28 as commit 0cb20bca via pull request https://github.com/TheTom/turboquant_plus/pull/93. The maintainer review surfaced a cleaner closed form (`E[||x_hat||^2] = (pi/2) * ||x||^2` and MMSE-optimal shrinkage `2/pi`) which the merged version documents in the docstring. Note that the merged commit message describes the scale change as fixing an 11x error; as explained under the five fixes above, that factor only arises relative to the orthogonal matrix introduced in the same change, since the stock Gaussian projection with the `sqrt(pi/2)/d` scale was the paper's own unbiased construction.
* tq3\_0 norm correction, zero block handling, and full Metal GPU support: pull request to Aaryan Kapoor llama.cpp at https://github.com/Aaryan-Kapoor/llama.cpp/pull/1
* GGML context sizing for shared rotation tensors: independently fixed upstream in TheTom llama-cpp-turboquant by wxtry in commit 70e45b7e on 2026-03-29, so no separate pull request was needed

A discussion thread tracking community work on TurboQuant in llama.cpp is at https://github.com/ggml-org/llama.cpp/discussions/20969

## How to Reproduce

### Round 1 MLX path (Hybrid K5/V4, achieves 100% at 16K)

See `guides/implementation-guide.md` for full environment setup. The key script is `benchmarks/test_hybrid_needle.py`. It requires mlx-optiq, mlx-lm, and the mlx-community/Qwen2.5-3B-Instruct-4bit model.

The four changes required in the installed optiq package are:
1. Replace the Gaussian QJL projection matrix with an orthogonal matrix (QR decomposition)
2. Change the dequantization scale from `sqrt(pi/2) / d` to the matching `sqrt(pi/2) / sqrt(d)`
3. Apply a damping factor of 0.7 to the QJL correction term (approximately the MMSE-optimal shrinkage `2/pi`)
4. Use Hybrid K5/V4 bit allocation (K uses 4-bit MSE plus 1-bit QJL, V uses 4-bit MSE only)

The test\_hybrid\_needle.py script applies all four changes inline as local class overrides.

### Round 2 llama.cpp path (Aaryan fork, achieves correct output and 2K/4K needle)

See `guides/round2-guide.md` for the build steps. The fork is Aaryan Kapoor's llama.cpp branch `turboquant-tq3_0`. The Metal support gaps and norm correction bugs documented in `patches/key-fixes.md` must be applied to run tq3\_0 with Metal enabled on M1 Pro.

Full details of each code change are in `patches/key-fixes.md`.

## Credits

* Paper authors: [Amir Zandieh](https://github.com/amirzandieh), Majid Daliri, Majid Hadian, [Vahab Mirrokni](https://research.google/people/mirrokni/) (Google Research)
* [Aaryan Kapoor](https://github.com/Aaryan-Kapoor): llama.cpp fork with tq3\_0 implementation (branch turboquant-tq3\_0)
* [Tom Turney](https://github.com/TheTom) (TheTom): turboquant\_plus Python prototype and llama-cpp-turboquant fork
* [Prince Canuma](https://github.com/Blaizzy): MLX implementation reference and community benchmarks

## Paper Reference

Zandieh, A., Daliri, M., Hadian, M., and Mirrokni, V. TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate. arXiv 2504.19874. Presented at ICLR 2026 (poster).
