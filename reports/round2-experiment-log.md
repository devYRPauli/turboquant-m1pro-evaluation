# TurboQuant Experiment Log

## Hardware and Environment

* Hardware: Apple M1 Pro
* Memory: 16GB unified memory
* Target OS: macOS Sequoia
* Target Python: 3.12
* Observed OS version from `sw_vers -productVersion`: 26.4
* Observed default `python3`: 3.14.3
* Verified `python3.12`: 3.12.13

## Log

* 2026-03-27 13:31:59 EDT: Project initialized for a fresh TurboQuant evaluation run on M1 Pro 16GB at `/Users/yashrajpandey/Projects/turboquant-m1pro`.
* 2026-03-27 13:31:59 EDT: Confirmed host reports OS version `26.4`; treating macOS Sequoia as the requested target label in project docs.
* 2026-03-27 13:31:59 EDT: Confirmed `python3 --version` = `Python 3.14.3`.
* 2026-03-27 13:31:59 EDT: Confirmed `python3.12 --version` = `Python 3.12.13`.
* 2026-03-27 13:31:59 EDT: Confirmed `ollama list` succeeded and `qwen2.5:3b` is available locally.
* 2026-03-27 13:34:00 EDT: Ran `pkill -9 -f llama-cli` before setup work; no stale `llama-cli` process was present.
* 2026-03-27 13:34:21 EDT: Cloned `https://github.com/Aaryan-Kapoor/llama.cpp.git` into `aaryan-llama-cpp` and checked out branch `turboquant-tq3_0`.
* 2026-03-27 13:36:06 EDT: Built Aaryan Kapoor's fork successfully with `GGML_METAL=ON`, `GGML_METAL_EMBED_LIBRARY=ON`, and `Release` mode. Build completed with repeated compiler warnings from `common/jinja/*` about possible `noreturn` annotations, but no build errors.
* 2026-03-27 13:36:20 EDT: Verified `./build/bin/llama-cli --help` exposes `--cache-type-k` and `--cache-type-v` value `tq3_0`.
* 2026-03-27 13:36:35 EDT: Parsed the Ollama manifest for `qwen2.5:3b`; model blob digest is `sha256:5ee4f07cdb9beadbbb293e85803c569b01bd37ed059d2715faa7bb405f31caa6`.
* 2026-03-27 13:36:35 EDT: Created symlink `aaryan-llama-cpp/models/qwen2.5-3b-q4_k_m.gguf` to the local Ollama blob.
* 2026-03-27 13:37:44 EDT: First `q8_0` baseline run answered correctly: `The capital of France is Paris.` Reported speed: `Prompt 133.4 t/s`, `Generation 57.4 t/s`.
* 2026-03-27 13:37:44 EDT: The initial baseline command did not terminate cleanly after producing the answer because `llama-cli` entered conversation mode and kept printing prompt markers. Forced termination with `pkill -9 -f llama-cli` and preserved the raw output in `notebooks/phase3-q8_0.log`.
* 2026-03-27 13:37:44 EDT: Adjustment for reruns: add `--single-turn` so each test remains foreground-only but exits immediately after one response.
* 2026-03-27 13:39:00 EDT: Clean `q8_0` rerun with `--single-turn` succeeded. Output: `The capital of France is Paris.` Reported speed: `Prompt 306.6 t/s`, `Generation 57.9 t/s`. Full log saved to `notebooks/phase3-q8_0-single-turn.log`.
* 2026-03-27 13:39:00 EDT: `tq3_0` run with the same settings and `-ngl 99` failed before inference. Crash site: `ggml/src/ggml-backend.cpp:809`; error: `pre-allocated tensor (cache_k_l0 (view)) in a buffer (MTL0) that cannot run the operation (SET_ROWS)`. Full backtrace saved to `notebooks/phase3-tq3_0-single-turn.log`.
* 2026-03-27 13:39:00 EDT: Since the exact requested TurboQuant sanity command did not produce coherent output, Phase 4 needle testing is blocked unless a CPU-only workaround succeeds.
* 2026-03-27 13:39:58 EDT: Diagnostic `tq3_0` run with `-ngl 0` completed without a backend crash, but output quality was poor: `France is a country a country country?` Reported speed: `Prompt 147.2 t/s`, `Generation 47.1 t/s`. Full log saved to `notebooks/phase3-tq3_0-ngl0-single-turn.log`.
* 2026-03-27 13:39:58 EDT: Matching diagnostic `q8_0` run with `-ngl 0` remained coherent: `The capital of France is Paris.` Reported speed: `Prompt 159.4 t/s`, `Generation 47.9 t/s`. Full log saved to `notebooks/phase3-q8_0-ngl0-single-turn.log`.
* 2026-03-27 13:39:58 EDT: Decision: skip Phase 4 needle-in-a-haystack testing. TurboQuant does not meet the prerequisite on this host: `-ngl 99` crashes before inference and `-ngl 0` produces degraded output on a trivial prompt.
* 2026-03-27 13:53:11 EDT: Cloned `https://github.com/TheTom/turboquant_plus.git` into `turboquant_plus`.
* 2026-03-27 13:53:11 EDT: Created a Python 3.12 virtual environment in `turboquant_plus/.venv`, upgraded `pip`, and installed the editable package with dev dependencies. Logs saved to `notebooks/phase5-install.log`.
* 2026-03-27 13:53:11 EDT: Installed optional real-model validation dependencies `torch`, `transformers`, and `accelerate`. Logs saved to `notebooks/phase5-validate-install.log`.
* 2026-03-27 13:53:11 EDT: Ran `python -m pytest tests/ -v` inside the Python 3.12 venv. Result: `538` collected, `532 passed`, `6 skipped` in `13.34s`. This is substantially beyond Round 1's 144-test baseline and also above the README's `511+` claim. Full log saved to `notebooks/phase5-pytest.log`.
* 2026-03-27 13:53:11 EDT: Ran `python benchmarks/demo.py`. The demo completed successfully and reported expected compression/quality tradeoffs, including single-vector 3-bit compression ratio `4.9×` and KV-cache demo ratios from `1.9×` to `2.6×`. Full log saved to `notebooks/phase5-demo.log`.
* 2026-03-27 13:53:11 EDT: Ran `python benchmarks/validate_real_model.py` against `Qwen/Qwen3-1.7B`. The process initially went quiet for more than 5 minutes during model download/load, and a kill attempt was started but interrupted by the user; the script continued and completed successfully. Full log saved to `notebooks/phase5-validate-real-model.log`.
* 2026-03-27 13:53:11 EDT: Real-model validation summary: KV shape `(28, 8, 37, 128)`. Compression results included `Uniform 3-bit` K cosine `0.921966`, ratio `2.6×`; `Outlier 3.5-bit` K cosine `0.945181`, ratio `8.0×`; `Uniform 4-bit` K cosine `0.975231`, ratio `1.9×`.
* 2026-03-27 13:53:11 EDT: Real-model attention-quality summary: average attention cosine remained limited (`0.640913` for uniform 3-bit, `0.724851` for outlier 3.5-bit, `0.818580` for uniform 4-bit).
* 2026-03-27 13:53:11 EDT: The prototype's built-in real-model NIAH check at `920` tokens did not retrieve the needle. Response did not contain `TURBOQUANT42`.
* 2026-03-27 13:53:11 EDT: Round 1 comparison: the Python prototype advanced from `144/144` tests in Round 1 to `532 passed / 6 skipped / 538 collected` here, but the local llama.cpp fork results on this M1 Pro still do not support proceeding to long-context TurboQuant retrieval tests.

## M1 Pro TurboQuant Debug Findings

* 2026-03-27 14:03:13 EDT: Root cause 1 for the requested `-ngl 99` run is a Metal backend capability gap, not an arbitrary runtime hang. Source inspection shows `ggml_backend_sched_backend_id_from_cur()` aborts when a preallocated tensor buffer cannot run the current op (`ggml/src/ggml-backend.cpp`, around line 809), and `ggml_metal_device_supports_op()` only allows `GGML_OP_SET_ROWS` for destination types `f32`, `f16`, `bf16`, `q8_0`, `q4_0`, `q4_1`, `q5_0`, `q5_1`, and `iq4_nl` when `op->src[0]->type == f32` (`ggml/src/ggml-metal/ggml-metal-device.m`, around lines 1233-1248). `tq3_0` is absent from this allowlist.
* 2026-03-27 14:03:13 EDT: Evidence for root cause 1: the exact requested TurboQuant run with KV offload enabled failed with `pre-allocated tensor (cache_k_l0 (view)) in a buffer (MTL0) that cannot run the operation (SET_ROWS)`.
* 2026-03-27 14:03:13 EDT: Root cause 2 for Metal-backed execution is missing Flash Attention kernels for `tq3_0`. A diagnostic run with model offload enabled but KV offload disabled (`-ngl 99 -nkvo -fa on --cache-type-k tq3_0 --cache-type-v tq3_0`) failed with `Function kernel_flash_attn_ext_vec_tq3_0_dk128_dv128 was not found in the library`.
* 2026-03-27 14:03:13 EDT: Source inspection confirms root cause 2: `ggml/src/ggml-metal/ggml-metal.metal` instantiates `kernel_flash_attn_ext_vec_*` templates for `f32`, `f16`, `bf16`, `q4_0`, `q4_1`, `q5_0`, `q5_1`, and `q8_0`, but not for `tq3_0`.
* 2026-03-27 14:03:13 EDT: Conclusion on Metal/M1 Pro path: this fork cannot currently run `tq3_0` with model layers on Metal on this machine because the implementation is incomplete in two separate places: `SET_ROWS` support and Flash Attention kernel instantiation.
* 2026-03-27 14:03:13 EDT: CPU-only diagnostics show a separate correctness/quality issue independent of Metal. `K=tq3_0, V=f16, -ngl 0, -fa on` produced degraded output: `France is a country country located in Europe, known for its stunning cuisine, history, culture, and culture.`
* 2026-03-27 14:03:13 EDT: The same `K=tq3_0, V=f16` diagnostic with Flash Attention disabled (`-fa off`) remained badly degraded: `France is a country a country country country a country country country country a country country a country country a country a...`
* 2026-03-27 14:03:13 EDT: A complementary CPU-only diagnostic with `K=f16, V=tq3_0, -ngl 0, -fa on` remained coherent: `The capital of France is Paris.`
* 2026-03-27 14:03:13 EDT: Conclusion on CPU correctness: the primary quality failure is associated with `tq3_0` on the K cache path, not the V cache path. This points to a defect or severe quality regression in the K-side `tq3_0` implementation or its dot-product usage, not just a missing Metal feature.
* 2026-03-27 14:03:13 EDT: Added a fail-fast guard in `src/llama-context.cpp` so unsupported `tq3_0` + Metal combinations now error out early with a concrete explanation instead of crashing inside Metal backend scheduling or pipeline compilation.
* 2026-03-27 14:03:13 EDT: Verified new behavior for the original requested run (`-ngl 99 -fa on` with KV offload enabled): it now stops immediately with `tq3_0 KV cache is not supported with Metal KV offload on this build: Metal SET_ROWS does not support tq3_0 buffers`.
* 2026-03-27 14:03:13 EDT: Verified new behavior for `-ngl 99 -nkvo -fa on`: it now stops immediately with `tq3_0 KV cache is not supported with Metal Flash Attention on this build: kernel_flash_attn_ext_vec_tq3_0_* is missing from the Metal library`.
* 2026-03-27 14:20:03 EDT: Replaced the fail-fast-only state with an actual Metal implementation patch for `tq3_0` in `ggml-metal.metal` and `ggml-metal-device.m`. Added `tq3_0` quantize/dequantize helpers, `SET_ROWS` kernels, `FLASH_ATTN_EXT` and `FLASH_ATTN_EXT_VEC` kernel instantiations, and Metal allowlist entries. Removed the temporary blanket rejection in `src/llama-context.cpp`.
* 2026-03-27 14:20:03 EDT: Rebuilt `llama-cli` successfully after the Metal patch. The embedded Metal library regenerated and linked cleanly, which confirms the new `tq3_0` kernels compile on this M1 Pro toolchain.
* 2026-03-27 14:20:03 EDT: Re-ran the original requested TurboQuant sanity command on Metal: `-ngl 99 -fa on --cache-type-k tq3_0 --cache-type-v tq3_0`. Result: the run now completes instead of crashing, but output quality is still bad: `France is a country a country country country country a country country country`. Reported speed: `Prompt 66.4 t/s`, `Generation 11.6 t/s`.
* 2026-03-27 14:20:03 EDT: Interpretation of the rerun: the original M1 Pro failure was not a hardware incompatibility. It was a concrete implementation gap in Metal support for `tq3_0`, and that gap was sufficient to prevent execution until patched.
* 2026-03-27 14:20:03 EDT: Metal split-cache diagnostic with `K=f16, V=tq3_0, -ngl 99, -fa on` is coherent: `The capital of France is Paris.` Reported speed: `Prompt 257.7 t/s`, `Generation 33.0 t/s`. Full log saved to `notebooks/debug-k-f16-v-tq3_0-ngl99.log`.
* 2026-03-27 14:20:03 EDT: Metal split-cache diagnostic with `K=tq3_0, V=f16, -ngl 99, -fa on` remains degraded: `France is a country a country country country country country country country France.` Reported speed: `Prompt 242.8 t/s`, `Generation 26.2 t/s`. Full log saved to `notebooks/debug-k-tq3_0-v-f16-ngl99.log`.
* 2026-03-27 14:20:03 EDT: Updated conclusion after patching Metal: execution on M1 Pro is now possible with `tq3_0`, so the previous crash explanation is resolved. The remaining blocker is a correctness bug localized to the `K=tq3_0` path, which reproduces on both CPU-only and Metal-backed runs while `V=tq3_0` remains coherent.
* 2026-03-27 14:47:15 EDT: Identified a concrete K-path scale bug in `tq3_0` quantization. The quantizer stored raw RMS in `block_tq3_0.d`, which leaves reconstructed block norms mismatched after Lloyd-Max quantization and WHT inversion; zero-energy blocks also decoded as structured noise because the old path forced `rms = 1`. This especially corrupts attention scores on the K path.
* 2026-03-27 14:47:15 EDT: Patched both the CPU reference quantizer (`ggml/src/ggml-quants.c`) and the Metal quantizer (`ggml/src/ggml-metal/ggml-metal.metal`) to: (1) emit true zero blocks when the input norm is effectively zero, and (2) store norm-correction scale `original_norm / reconstruction_norm` instead of raw RMS.
* 2026-03-27 14:47:15 EDT: Rebuilt `llama-cli` successfully after the norm-correction fix.
* 2026-03-27 14:47:15 EDT: Re-ran Metal split-cache diagnostic with `K=tq3_0, V=f16, -ngl 99, -fa on`. Result is now coherent: `The capital of France is Paris.` Reported speed: `Prompt 213.0 t/s`, `Generation 29.8 t/s`. Full log saved to `notebooks/debug-k-tq3_0-v-f16-ngl99.log`.
* 2026-03-27 14:47:15 EDT: Re-ran the original full Metal TurboQuant sanity command with `K=tq3_0, V=tq3_0, -ngl 99, -fa on`. Result is now coherent: `The capital of France is Paris.` Reported speed: `Prompt 82.1 t/s`, `Generation 12.0 t/s`. Full log saved to `notebooks/phase3-tq3_0-single-turn.log`.
* 2026-03-27 14:47:15 EDT: Re-ran CPU-only `tq3_0` sanity check with `-ngl 0`. Result is now coherent: `The capital of France is Paris.` Reported speed: `Prompt 129.6 t/s`, `Generation 40.0 t/s`. Full log saved to `notebooks/phase3-tq3_0-ngl0-single-turn.log`.
* 2026-03-27 14:47:15 EDT: Updated conclusion: both the Metal execution blocker and the K-path correctness blocker are fixed on this M1 Pro for the Phase 3 sanity prompt. Phase 4 needle testing is now unblocked.

## Phase 4 Needle Test

* 2026-03-27 14:59:50 EDT: Generated reproducible prompt files for the retrieval test at `notebooks/needle-2k-prompt.txt` (1567 words), `notebooks/needle-4k-prompt.txt` (3055 words), and `notebooks/needle-8k-prompt.txt` (6055 words). The midpoint needle is: `IMPORTANT FACT: The experimental blueberry accession designated UF-B4291 carries a novel disease-resistance allele called FROSTBLOCK-7 in the VcMYB4 locus.`
* 2026-03-27 14:59:50 EDT: 2K baseline with `q8_0` passed. Response contained both required facts: `FROSTBLOCK-7` and `VcMYB4`. Reported speed: `Prompt 456.6 t/s`, `Generation 44.6 t/s`. Full log saved to `notebooks/needle-2k-q8_0.log`.
* 2026-03-27 14:59:50 EDT: 2K run with full `tq3_0` also passed. Response contained both required facts: `FROSTBLOCK-7` and `VcMYB4`. Reported speed: `Prompt 23.1 t/s`, `Generation 1.3 t/s`. Full log saved to `notebooks/needle-2k-tq3_0.log`.
* 2026-03-27 14:59:50 EDT: 4K baseline with `q8_0` passed. Response contained both required facts: `FROSTBLOCK-7` and `VcMYB4`. Reported speed: `Prompt 408.8 t/s`, `Generation 32.8 t/s`. Full log saved to `notebooks/needle-4k-q8_0.log`.
* 2026-03-27 14:59:50 EDT: 4K run with full `tq3_0` also passed. Response contained both required facts: `FROSTBLOCK-7` and `VcMYB4`, although the wording briefly glitched (`FROROSTBLOCK-7`) before resolving to `FROSTBLOCK-7`. Reported speed: `Prompt 12.4 t/s`, `Generation 0.7 t/s`. Full log saved to `notebooks/needle-4k-tq3_0.log`.
* 2026-03-27 14:59:50 EDT: Phase 4 status update: correctness now holds at both ~2K and ~4K prompt sizes on this M1 Pro for `tq3_0`, but prompt and generation speed degrade sharply versus `q8_0`. 8K was not started yet to avoid launching another very long run before recording the 2K/4K results.
