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
    reproduction-and-ablation-2026-07-03.md   Follow-up: 2K reproduction and QJL ablation results
logs/                           All llama-cli debug and test run logs, needle prompts, MLX raw outputs
    needle-control-k5v4-mse.json        Pure MSE K5/V4 control run, 2K to 16K, no QJL
    needle-repro-2026-10-05.json        Needle runs, all MLX configs, 2K to 16K, full responses
    kv-memory-16k-2026-10-05.json       Stored KV bytes and MLX peak memory at 16K, 5 rounds
    hybrid-reproduction-2026-07-03.json Hybrid K5/V4 rerun with the original modified optiq files
    qjl-ablation-2026-07-03.json        QJL ablation, July run
    qjl-ablation-2026-10-05.json        QJL ablation, October rerun
    phase2-rerun-2026-10-05.log         Phase 2 short-prompt comparison, October rerun
    test-hybrid-needle-2026-10-05.log   test_hybrid_needle.py output, October run
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
    test_hybrid_needle.py       Early K4/V4 needle probe, 800 tokens (fails, see the log in logs/)
    phase3_results.json         Raw Phase 3 result data
    tq_patched.py               Hybrid K5/V4 cache rebuilt as subclasses of stock mlx-optiq 0.0.1
    needle_repro.py             Needle runs for baseline, stock, and patched caches (2K to 16K)
    kv_memory.py                Stored KV bytes and MLX peak memory at 16K
    qjl_ablation.py             QJL ablation probe (projection, scale, damping) from the 2026-07-03 pass
    hybrid_reproduction.py      Hybrid K5/V4 rerun script from the 2026-07-03 pass (now uses tq_patched.py)
requirements.txt                Pinned Python environment for the MLX benchmarks
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

The files in `reports/` and `guides/` are kept as written at the time. Some of their claims were later corrected. Where they disagree with this README or `FINDINGS.md`, the README and `FINDINGS.md` take precedence, because they cite a raw file for each number.

## Hardware and Environment

* Machine: MacBook Pro, Apple M1 Pro chip, 16GB unified memory, 512GB SSD
* OS: macOS 26.3.1 and 26.4 across the two rounds (the version 26 line is macOS Tahoe; the source logs label it Sequoia, which is the version 15 line, and that label is preserved in the copied reports)
* Python: 3.12.13 (via Homebrew)
* Model tested: Qwen2.5-3B-Instruct (Q4\_K\_M GGUF via Ollama, and 4-bit MLX via Hugging Face)
* Round 1 implementations: mlx-optiq v0.0.1 (MLX inference), TheTom turboquant\_plus Python prototype, TheTom llama-cpp-turboquant fork
* Round 2 implementation: Aaryan Kapoor llama.cpp fork, branch turboquant-tq3\_0, TheTom turboquant\_plus (updated)
* 2026-10-05 MLX rerun: macOS 27.0.1, Python 3.12.14, packages pinned in `requirements.txt` (mlx 0.31.1, mlx-lm 0.31.1, mlx-optiq 0.0.1, transformers 5.3.0, numpy 2.4.3), model revision 4f83f8f146fdf28b512a06562b671d7af4fab457. Each raw JSON file from this pass records its environment.

## Key Findings Summary

Stock mlx-optiq 0.0.1 caches scored 0% needle retrieval at 2K, 4K, 8K, and 16K tokens. After five fixes, the Hybrid K5/V4 configuration on the MLX path scored 100% at 4K, 8K, and 16K in the recorded run. A rebuild of that configuration from pinned packages on 2026-10-05 scored 100% at 16K only, and 50% at the shorter lengths. The Aaryan Kapoor llama.cpp fork gave correct output on sanity prompts and in the 2K and 4K needle tests. The results table below lists the raw file behind each number.

The five fixes are:

1. QJL orthogonal projection: the paper defines the QJL projection as a random Gaussian matrix (Definition 1 in the paper), and the stock implementations followed it faithfully. That construction is unbiased, but empirically it destroyed generation at head dimension 128 (immediate word-loop degeneration). Replacing the Gaussian matrix with a random orthogonal matrix from QR decomposition, a variance-reducing modification of the paper's design, eliminated the degeneration.

2. QJL dequantization scale factor: the scale must match the projection matrix. The paper's `sqrt(pi/2) / d` is correct for a Gaussian matrix, whose rows have norm near `sqrt(d)`. Once the matrix is orthogonal (change 1), the matching scale becomes `sqrt(pi/2) / sqrt(d)`; keeping the Gaussian scale with an orthogonal matrix would make the correction about 11x too small at d=128. The two changes are one coupled substitution, not two independent bug fixes.

3. Hybrid K5/V4 configuration: even with correct QJL math, keys are more sensitive to quantization noise than values because attention scores depend on precise key-query inner products. Assigning 5 bits to keys (4-bit MSE plus 1-bit QJL) and 4 bits to values (4-bit MSE only) provided the necessary precision.

4. GGML context sizing bug: the TheTom llama-cpp-turboquant fork crashed on initialization due to a metadata context allocation formula that did not count the two shared rotation matrix tensors. Adding two extra `ggml_tensor_overhead()` slots fixed the crash. This was diagnosed as a software bug, not a hardware incompatibility.

5. Norm correction and zero block handling: the tq3\_0 quantizer in Aaryan's fork stored raw RMS as the scale factor, but the correct value is `original_norm / reconstruction_norm` to account for norm change during Lloyd-Max quantization. Near-zero blocks also decoded as structured noise because the guard set scale to 1.0 rather than emitting a true zero block.

In addition to these fixes, the validated MLX configuration applies a damping factor of 0.7 to the QJL correction term (see `benchmarks/tq_patched.py`). This is close to the MMSE-optimal shrinkage `2/pi` (approximately 0.6366) that was later formalized during upstream review of pull request 93. A reproduction that omits the damping factor is not running the validated configuration.

A follow-up ablation on 2026-07-03 (`reports/reproduction-and-ablation-2026-07-03.md`, raw output `logs/qjl-ablation-2026-07-03.json`) is consistent with the orthogonal projection and the matched `sqrt(d)` scale being one coupled substitution. Changing only the scale, or only the matrix while keeping the mismatched scale, still degenerates. The matched pair escapes word loops, and the 0.7 damping factor gives slightly cleaner text. Damping applied to the paper-faithful Gaussian configuration has no visible effect. The ablation is a qualitative probe with limits. It makes one greedy run per configuration on a synthetic 2K prompt with no chat template. Its keys use 4 bits (3-bit MSE plus 1-bit QJL), not the 5 bits of the Hybrid configuration. No configuration in it answers the question. The October rerun (`logs/qjl-ablation-2026-10-05.json`) shows the same ordering, but no response text matches the July run.

## Results Table

The needle score counts two facts, FROSTBLOCK-7 and VcMYB4, matched without regard to case by `benchmarks/build_prompt.py`. A score of 50% means the response has one of the two facts. "Recorded" is the original run. "2026-10-05" is the rerun from `requirements.txt` on macOS 27.0.1 with `benchmarks/needle_repro.py`.

| Configuration | Context | Recorded | 2026-10-05 | Raw output |
|---|---|---|---|---|
| Ollama baseline (Q4\_K\_M weights, FP16 KV) | 2K, 4K, 8K, 16K | 100% all lengths | not rerun | `benchmarks/phase3_results.json` |
| MLX baseline (FP16 KV) | 2K, 4K, 8K, 16K | 100% all lengths | 100% all lengths | `benchmarks/phase3_results.json`, `logs/needle-repro-2026-10-05.json` |
| MLX stock MSE-only 4-bit | 2K, 4K, 8K, 16K | 0% all lengths | 0% all lengths | `logs/needle-repro-2026-10-05.json` (the round 1 run kept no raw file) |
| MLX stock QJL, 4-bit keys (paper-faithful Gaussian) | 36 tokens, 2K | Degenerate (word loops) | Degenerate (word loops) | `logs/phase2-rerun-2026-10-05.log`, `logs/qjl-ablation-2026-07-03.json`, `logs/qjl-ablation-2026-10-05.json` |
| MLX K5/V4 bits with stock Gaussian QJL | 2K, 4K, 8K, 16K | not run | 0% all lengths, degenerate text | `logs/needle-repro-2026-10-05.json` |
| MLX Hybrid K5/V4 (orthogonal QJL, matched scale, 0.7 damping) | 2K | 50% | 50% | `benchmarks/phase3_results.json`, `logs/needle-repro-2026-10-05.json` |
| MLX Hybrid K5/V4 (orthogonal QJL, matched scale, 0.7 damping) | 4K | 100% | 50% | same |
| MLX Hybrid K5/V4 (orthogonal QJL, matched scale, 0.7 damping) | 8K | 100% | 50% | same |
| MLX Hybrid K5/V4 (orthogonal QJL, matched scale, 0.7 damping) | 16K | 100% | 100% | same |
| MLX pure MSE K5/V4 (5-bit MSE keys, 4-bit MSE values, no QJL) | 2K | not run | 100% | `logs/needle-control-k5v4-mse.json` |
| MLX pure MSE K5/V4 (5-bit MSE keys, 4-bit MSE values, no QJL) | 4K | not run | 50% | same |
| MLX pure MSE K5/V4 (5-bit MSE keys, 4-bit MSE values, no QJL) | 8K | not run | 50% | same |
| MLX pure MSE K5/V4 (5-bit MSE keys, 4-bit MSE values, no QJL) | 16K | not run | 100% | same |
| Aaryan fork tq3\_0 CPU, before norm fix | sanity prompt | Degenerate (repetitive) | not rerun | `logs/debug-k-tq3_0-v-f16-ngl0.log` |
| Aaryan fork tq3\_0 after all fixes | 2K, 4K needle | Both facts retrieved | not rerun | `logs/needle-2k-tq3_0.log`, `logs/needle-4k-tq3_0.log` |

Notes on the Hybrid and control rows:

* At 2K the recorded run retrieved VcMYB4 but wrote "FROSTst7" instead of "FROSTBLOCK-7". The 2026-07-03 rerun with the original modified package files gave the same text, the same scores, and the same MLX peak memory (within 0.1 MB) at all four lengths (`logs/hybrid-reproduction-2026-07-03.json`). The round 1 post-mortem report asserts 100% at 2K in its summary, but no raw output supports it.
* The recorded 4K pass depends on the case-insensitive match. The response wrote "FROstblock-7".
* The 2026-10-05 rerun uses `benchmarks/tq_patched.py`, a rebuild of the lost modified files. It scored 50% at 2K, 4K, and 8K: at 2K and 4K it wrote a corrupted allele name, and at 8K it gave only the locus. At 16K it wrote both facts exactly. See "Reproduction on 2026-10-05" in `FINDINGS.md` for the evidence that the rebuild is not op-for-op identical to the original.
* The pure MSE K5/V4 control row (`logs/needle-control-k5v4-mse.json`) uses 5-bit Lloyd-Max MSE on keys and 4-bit MSE on values with no QJL stage (`use_qjl=False` in `benchmarks/tq_patched.py`). Under case-insensitive scoring it scored 100% at 2K, 50% at 4K, 50% at 8K, and 100% at 16K, matching or exceeding the 2026-10-05 rebuild of the patched QJL hybrid across all lengths (under exact-case scoring it scored 0% at 2K and 50% at 16K because it wrote "VcMYb4"). This demonstrates that allocating the fifth bit to scalar quantization carries the 16K retrieval result without requiring QJL. At 16K it stored 290.4 MB of cache (33 percent less than the 435.6 MB Hybrid QJL cache) and peaked at 2988.0 MB (88.8 MB below the FP16 baseline on this rebuild).

## Speed and Memory Reference

Speed at 16K tokens (15947 prompt tokens, up to 80 generated) with Qwen2.5-3B on the M1 Pro, from `benchmarks/phase3_results.json`:

| Runner | Reported tokens per second | Wall time for the request |
|---|---|---|
| Ollama (FP16 KV) | 37.5 (decode only) | 49.3 s |
| MLX baseline FP16 | 2.0 (prefill included) | 39.9 s |
| MLX Hybrid K5/V4 | 1.1 (prefill included) | 71.4 s |

The phase3 request to Ollama sets no KV cache type, so Ollama used its default, f16, unless `OLLAMA_KV_CACHE_TYPE` was set on the server. No raw output records the server settings. The two tokens-per-second figures measure different things. Ollama reports its own decode rate, which excludes the prompt. The MLX figure divides the generated tokens by the total time, which includes the 16K prefill. Wall time is the closer comparison, but the Ollama wall time also covers the HTTP call and any model load. The Hybrid cache is slower than FP16 because it dequantizes the full cache at every step, with 128x128 matrix multiplies, in unfused MLX operations. In the 2026-10-05 rerun, the rebuild of Hybrid K5/V4 took 65.1 s (1.2 tok/s) and pure MSE K5/V4 took 65.2 s (1.2 tok/s, `logs/needle-control-k5v4-mse.json`), showing no wall-time difference between them.

KV memory at 16K tokens, from `logs/kv-memory-16k-2026-10-05.json` (except the pure MSE control row, which is a single run from `logs/needle-control-k5v4-mse.json`). Each configuration in the kv-memory harness ran 5 times in alternating order, and all 5 runs gave identical values. MB is 2^20 bytes. The model weights take 1655.8 MB.

| Cache | Stored K/V | Smaller than FP16 by | MLX peak | Peak vs FP16 |
|---|---|---|---|---|
| FP16 baseline | 563.4 MB | 1.00x | 3076.8 MB | 0 MB |
| Pure MSE K5/V4 (`needle-control-k5v4-mse.json`) | 290.4 MB | 1.94x | 2988.0 MB | -88.8 MB |
| Stock MSE-only 4-bit | 290.4 MB | 1.94x | 3190.6 MB | +113.8 MB |
| Hybrid K5/V4 (`tq_patched.py`) | 435.6 MB | 1.29x | 3245.4 MB | +168.6 MB |
| Hybrid K5/V4, K/V cast to FP16 before attention (diagnostic) | 429.2 MB | 1.31x | 2867.6 MB | -209.2 MB |

The stored cache is smaller than the bit counts suggest. mlx-optiq 0.0.1 stores one byte per element: uint8 codebook indices and int8 QJL signs, not packed bits. Per token and layer, FP16 K/V takes 1024 bytes, MSE-only 4-bit takes 528 bytes, and Hybrid K5/V4 takes 792 bytes. A bit-packed Hybrid layout would take about 300 bytes, 3.4x smaller than FP16. The phase3 harness printed this formula figure as `kv_theoretical_mb` (157.7 MB at 16K). An earlier version of this README gave "562 MB to 140 MB, 4.0x compression confirmed". Those were formula figures for a packed 4-bit cache, not measurements.

Peak memory is higher with the stock 4-bit MSE and Hybrid QJL caches, not lower. Those caches return the dequantized K/V in float32. Attention and the residual stream then run in float32 for every layer after the first. The peak occurs during prefill, where the float32 buffers cost more than the smaller cache saves. The diagnostic row casts the K/V back to FP16 before attention. Its peak drops by 378 MB, to 209 MB below the FP16 baseline. The pure MSE K5/V4 control configuration also achieved a lower whole-run peak (2988.0 MB, 88.8 MB below the FP16 baseline on this rebuild). The phase3 Hybrid peak was 3303.7 MB, 58 MB above the rebuild (see "Reproduction on 2026-10-05" in `FINDINGS.md`).

For Qwen2.5-3B at 16K on 16 GB, the cache size does not matter in practice: the FP16 run peaks at 3.1 GB.

## Upstream Contributions

The five fixes documented above have been published back upstream where appropriate:

* QJL orthogonal projection and `sqrt(d)` scale factor: merged into TheTom turboquant\_plus main on 2026-05-28 as commit 0cb20bca via pull request https://github.com/TheTom/turboquant_plus/pull/93. The maintainer review surfaced a cleaner closed form (`E[||x_hat||^2] = (pi/2) * ||x||^2` and MMSE-optimal shrinkage `2/pi`) which the merged version documents in the docstring. Note that the merged commit message describes the scale change as fixing an 11x error; as explained under the five fixes above, that factor only arises relative to the orthogonal matrix introduced in the same change, since the stock Gaussian projection with the `sqrt(pi/2)/d` scale was the paper's own unbiased construction.
* tq3\_0 norm correction, zero block handling, and full Metal GPU support: pull request to Aaryan Kapoor llama.cpp at https://github.com/Aaryan-Kapoor/llama.cpp/pull/1
* GGML context sizing for shared rotation tensors: independently fixed upstream in TheTom llama-cpp-turboquant by wxtry in commit 70e45b7e on 2026-03-29, so no separate pull request was needed

A discussion thread tracking community work on TurboQuant in llama.cpp is at https://github.com/ggml-org/llama.cpp/discussions/20969

## How to Reproduce

### Round 1 MLX path (Hybrid K5/V4)

The original Hybrid runs used a hand-modified copy of the installed mlx-optiq 0.0.1 package. `benchmarks/tq_patched.py` rebuilds the four changes as subclasses of the stock package:
1. Replace the Gaussian QJL projection matrix with an orthogonal matrix (QR decomposition)
2. Change the dequantization scale from `sqrt(pi/2) / d` to the matching `sqrt(pi/2) / sqrt(d)`
3. Apply a damping factor of 0.7 to the QJL correction term (approximately the MMSE-optimal shrinkage `2/pi`)
4. Use Hybrid K5/V4 bit allocation (K uses 4-bit MSE plus 1-bit QJL, V uses 4-bit MSE only)

Build the environment and run the two benchmarks:

```
uv venv --python 3.12 venv
uv pip install --python venv/bin/python -r requirements.txt
venv/bin/python benchmarks/needle_repro.py --cooldown 60 --out logs/needle-repro.json
venv/bin/python benchmarks/kv_memory.py --cooldown 60 --out logs/kv-memory-16k.json
```

`needle_repro.py` uses the phase3 prompt builder and scorer unchanged. It runs the FP16 baseline, stock MSE-only 4-bit, stock K5/V4 (Gaussian QJL), and patched K5/V4 at 2K, 4K, 8K, and 16K, and writes each full response with exact-case and case-insensitive scores. `kv_memory.py` runs 5 alternating rounds at 16K and writes stored cache bytes and MLX peak memory. The model download is about 1.7 GB. MLX peak memory stays under 3.5 GB at 16K.

Greedy decoding through the quantized cache is sensitive to the last bit of every floating point operation. Response text and scores can change on a different macOS or Metal version. On macOS 27.0.1 the Hybrid configuration scored 50% at 4K and 8K, where the recorded run scored 100% (see "Reproduction on 2026-10-05" in `FINDINGS.md`).

`benchmarks/test_hybrid_needle.py` is an early probe with 4-bit keys (3-bit MSE plus 1-bit QJL) on an 800-token prompt. It is not the Hybrid K5/V4 configuration, and it does not retrieve the needle (`logs/test-hybrid-needle-2026-10-05.log`).

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

## License

MIT.
