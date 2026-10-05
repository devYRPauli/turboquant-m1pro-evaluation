# Technical Findings: TurboQuant on Apple M1 Pro 16GB

This document provides a detailed technical writeup of the three main discoveries from this evaluation. All data and observations come from the experiment logs, debug logs, and reports in this repository.

---

## Finding 1: The Paper's QJL Construction Fails in Practice at Head Dimension 128; an Orthogonal Variant Fixes It

### What QJL Is Supposed to Do

The TurboQuant paper's two-stage design works as follows. Stage 1 (PolarQuant) applies a random orthogonal rotation to each KV vector, making the coordinates approximately Gaussian, then quantizes each coordinate independently using Lloyd-Max centroids. Stage 2 (QJL) computes the sign of a linear projection of the quantization residual (the difference between the original vector and the PolarQuant reconstruction). This 1-bit sign vector, combined with a scale factor, provides an unbiased correction to the inner product errors introduced by stage 1.

The key mathematical claim is that for a random projection matrix S, the estimator `scale * (sign(S * residual)) @ S` approximates the residual vector with zero bias. This is derived from Johnson-Lindenstrauss lemma results. When computing attention scores (query dot key), you can add this correction to get a better estimate of the true inner product.

The paper (Definition 1 and Algorithm 2) defines S as a d x d matrix with i.i.d. standard Gaussian entries and the dequantization scale as `sqrt(pi/2) / d`. That pair is self-consistent and unbiased: Gaussian rows have norm near `sqrt(d)`, so the projection output is about `sqrt(d)` larger than it would be with orthonormal rows, and the `1/d` scale absorbs exactly that factor. For an orthogonal projection matrix (orthonormal rows), the matching unbiased scale is `sqrt(pi/2) / sqrt(d)`. The two conventions differ by `sqrt(d)` and are interchangeable in expectation; what differs between them is variance.

### What the Implementations Actually Did

Both stock implementations were faithful to the paper. The pristine mlx-optiq v0.0.1 wheel from PyPI and the pre-fix state of TheTom turboquant\_plus (visible in the before side of pull request 93) both used a random Gaussian matrix for S with the scale `sqrt(pi/2) / d`, exactly as the paper specifies. Neither contained a transcription error in the QJL math. The installed package inside the round 1 virtual environment differs from the PyPI wheel because the fixes described below were applied to it in place during the project. Those two edited files (`optiq/core/turbo_quant.py` and `optiq/core/turbo_kv_cache.py`) were used again for the 2026-07-03 reproduction and were not kept after it. `benchmarks/tq_patched.py` rebuilds the same changes as subclasses of the stock 0.0.1 package, so the Hybrid configuration runs from a clean install of `requirements.txt`.

**What went wrong in practice.** The paper-faithful construction is unbiased, but its variance proved fatal on real LLM KV vectors at head dimension 128. Every token that passes through the quantizer accumulates a small error in the reconstructed attention key. Over thousands of tokens in a long context, these errors compound. The attention score distribution distorts progressively, eventually causing the softmax to collapse toward a single token or distribute nonsensically. The observed symptom was immediate word-loop degeneration the moment QJL was enabled at any context length, even 36 tokens.

**The fix: a coupled substitution, not two bug fixes.** Replacing the Gaussian matrix with an orthogonal matrix from QR decomposition eliminated the degeneration. Orthogonal matrices preserve norms and inner products exactly, so the projection step introduces no distortion of its own. Because the matrix changed, the scale had to change with it: `sqrt(pi/2) / d` is the correct unbiased scale for a Gaussian matrix, and `sqrt(pi/2) / sqrt(d)` is the correct scale for an orthogonal one. Keeping the Gaussian scale with the orthogonal matrix would make the correction approximately 11x too small at d=128 (`sqrt(128) = 11.31`). Earlier versions of this write-up, and the commit message merged upstream, described the scale itself as an 11x bug in the stock code; that framing is wrong, because in the stock code the `1/d` scale was paired with the Gaussian matrix it belongs to. In expectation the stock correction term had the same magnitude as the fixed one. The difference the fix makes is in variance, not bias.

The validated configuration additionally applies a damping factor of 0.7 to the QJL correction term. This is close to the MMSE-optimal shrinkage `2/pi` (approximately 0.6366): the unbiased estimator inflates reconstruction energy (`E[||x_hat||^2] = (pi/2) * ||x||^2`), and shrinking it toward zero trades a small bias for a lower mean squared error.

A follow-up ablation on 2026-07-03 (`reports/reproduction-and-ablation-2026-07-03.md`, raw output `logs/qjl-ablation-2026-07-03.json`) isolated the three changes. The paper-faithful Gaussian projection with the 1/d scale degenerates into word loops. Changing only the scale, or only the projection matrix while keeping the mismatched scale, still degenerates. This is consistent with the orthogonal projection and the matched `sqrt(d)` scale being one coupled substitution rather than two independent fixes. The matched pair escapes word loops into semi-coherent repetition of the filler text, and the 0.7 damping factor makes that text slightly cleaner. Damping applied to the paper-faithful Gaussian configuration has no visible effect.

The ablation has limits. It makes one greedy run per configuration on a synthetic 2K prompt with no chat template, so no configuration answers the question and the evidence is qualitative. Its keys use 4 bits (3-bit MSE plus 1-bit QJL), not the 5 bits of the Hybrid configuration. The October rerun (`logs/qjl-ablation-2026-10-05.json`) shows the same ordering, but no response text matches the July run (see "Reproduction on 2026-10-05" below).

### Why Every Implementation Dropped QJL

Both primary implementations tested (mlx-optiq and TheTom turboquant\_plus) independently arrived at the conclusion that MSE-only quantization worked better in practice than the two-stage design. The experiment log notes that both Aaryan Kapoor and TheTom independently dropped QJL in favor of MSE-only with all bits going to Lloyd-Max centroids.

This is consistent with the finding above. A paper-faithful Gaussian QJL really does make output worse at head dimension 128, so the natural engineering response was to disable the stage. The conclusion of this evaluation is that the two-stage design itself is sound: it works once the projection is orthogonal, the scale matches, and the correction is damped.

The post-mortem report records that with the fixes applied, the QJL correction reduces MSE from 0.00023 to 0.000129 (a 44 percent reduction) and improves cosine similarity to 99.7 percent on real model activations. The 44 percent figure matches the theoretical `(pi/2 - 1)`, approximately 43 percent, for the undamped estimator. The theoretical maximum with MMSE-optimal shrinkage `2/pi` is `1 - 2/pi`, approximately 64 percent. The raw output behind the post-mortem MSE and cosine figures was not kept, so they are not reproducible from this repository.

### What the Fix Required

The fix as rebuilt in `benchmarks/tq_patched.py`:

1. Generate the QJL projection matrix S using QR decomposition of a Gaussian random draw, then enforce a consistent sign convention by multiplying columns by the signs of the diagonal of R. This is the construction the package already uses for its rotation matrix.

2. Change the dequantization scale from `math.sqrt(math.pi / 2.0) / self.d` to the matching `math.sqrt(math.pi / 2.0) / math.sqrt(self.d)`.

3. Multiply the QJL correction term by a damping factor of 0.7 (approximately the MMSE-optimal shrinkage `2/pi`).

`benchmarks/test_hybrid_needle.py` holds an earlier version of the same three changes with 4-bit keys. It fails its 800-token needle test (`logs/test-hybrid-needle-2026-10-05.log`), and no raw output shows it passing.

These changes together, combined with the Hybrid K5/V4 configuration described in Finding 2, moved needle retrieval from 0% to 100% at 4K, 8K, and 16K tokens in the recorded run (`benchmarks/phase3_results.json`). At 2K the recorded run scored 50%: the model retrieved the locus VcMYB4 but corrupted the allele name FROSTBLOCK-7 into "FROSTst7". The 4K pass depends on the case-insensitive match, because the response wrote "FROstblock-7". The 2026-07-03 rerun with the original modified files reproduced the text of all four responses (`logs/hybrid-reproduction-2026-07-03.json`). The 2026-10-05 rerun with the rebuild scored 100% only at 16K (see "Reproduction on 2026-10-05" below).

### Upstream Status

The QJL orthogonal projection and `sqrt(d)` scale factor changes were merged into the TheTom turboquant\_plus reference Python implementation on 2026-05-28 as commit 0cb20bca via pull request 93: https://github.com/TheTom/turboquant_plus/pull/93. The merged version adds a `shrinkage` parameter on the dequantize path. Default is 1.0 (classical unbiased estimator). The MMSE-optimal value is `2/np.pi`, approximately 0.6366, derived from the closed forms `E[||x_hat||^2] = (pi/2) * ||x||^2` and `E[<x_hat, x>] = ||x||^2`; this derivation is documented in the dequantize docstring. The PR also adds an orthogonality contract assertion in `__init__` and a corresponding test at d in {64, 128, 256, 512} with tolerance `1e-12`. One caveat: the merged commit message describes the scale change as fixing a formula that was "11x too small"; as explained above, that factor only arises relative to the orthogonal matrix introduced in the same change, since the stock Gaussian projection with the `1/d` scale was the paper's own unbiased construction.

---

## Finding 2: Keys Are More Sensitive Than Values

### The Asymmetry

Even with the working QJL variant (orthogonal projection, matched scale factor, damping), 4-bit keys were not enough for reliable fact retrieval. With keys at 4 bits (3-bit MSE plus 1-bit QJL) and values at 4-bit MSE, `benchmarks/test_hybrid_needle.py` does not retrieve the needle from an 800-token prompt (`logs/test-hybrid-needle-2026-10-05.log`).

The diagnostic evidence for this asymmetry appeared early and clearly. In both the CPU-only path and the Metal path in Round 2:

* Running with `K=tq3_0, V=f16` produced degenerate output (repetitive text).
* Running with `K=f16, V=tq3_0` produced correct output ("The capital of France is Paris.").

This pattern held consistently across multiple configuration variants. The value cache could be heavily quantized without observable quality loss on short prompts. The key cache could not.

### Why Keys Are More Sensitive

Attention scores are computed as softmax(Q K^T / sqrt(d)). Each attention score depends on the inner product of a query vector with every key vector in the context. Key quantization noise affects all attention scores for a given position. Value quantization noise affects only the output for that position after the attention distribution is already determined.

In a needle-in-a-haystack task, the model must precisely locate one specific fact buried among thousands of tokens of filler. This requires the attention weights for the needle position to be substantially higher than for filler positions. If key quantization adds noise to the Q-K dot products, the needle's signal can be washed out by filler noise. The model then retrieves something from the filler or hallucinates.

At short context (36 tokens), there are few filler positions to compete with and the absolute noise is small. At 2K tokens or more, there are hundreds or thousands of filler positions and the accumulated noise overwhelms the needle's signal.

Values, by contrast, only determine what information is returned from a position once attention has already focused there. If the model correctly attends to the needle position (because keys are accurate), value quantization only slightly distorts the retrieved content. The retrieval still succeeds.

### The Hybrid K5/V4 Configuration

The fix is to allocate bits asymmetrically. Assign 5 bits to keys (4-bit MSE base plus 1-bit QJL correction) and 4 bits to values (4-bit MSE only, no QJL). This gives keys more precision where the attention mechanism is sensitive, while values remain at 4-bit which is sufficient for their role.

The average is 4.5 bits per cache element. A bit-packed layout would make the cache about 3.4x to 3.6x smaller than FP16, depending on how the norms are stored. mlx-optiq 0.0.1 does not pack bits: it stores one byte per codebook index and one byte per QJL sign. The measured Hybrid cache at 16K is 435.6 MB against 563.4 MB for FP16, 1.29x smaller (`logs/kv-memory-16k-2026-10-05.json`).

In the recorded run this configuration scored 100% at 4K, 8K, and 16K tokens and 50% at 2K (`benchmarks/phase3_results.json`). Stock MSE-only 4-bit scores 0% at all four lengths. The same K5/V4 bits with the stock Gaussian QJL also score 0%, with degenerate text (`logs/needle-repro-2026-10-05.json`). So the bit allocation alone does not fix retrieval: the orthogonal QJL change is also needed. The 2026-10-05 rebuild scored 50% at 2K, 4K, and 8K and 100% at 16K; see the notes under the results table in the README.

The asymmetry insight is consistent with the broader literature on attention quantization. Keys encode positional and semantic identity for retrieval. Values encode content. These have different precision requirements, and hardware implementations that ignore this distinction leave accuracy on the table.

---

## Finding 3: Implementation Bugs Found and Fixed

### Bug A: GGML Context Sizing (TheTom llama-cpp-turboquant fork)

**Location:** `src/llama-kv-cache.cpp`, line 54 in the TheTom fork.

**Symptom:** Both `--cache-type-k turbo3` and `--cache-type-k turbo4` crashed immediately with `GGML_ASSERT(obj_new) failed` in `ggml_new_tensor_impl`. The crash occurred even with `-ngl 0` (all computation on CPU, no GPU involvement).

**Initial misdiagnosis:** The crash output included a log line reporting that the Metal Tensor API was disabled for Apple7 GPU family (which is M1 Pro's GPU family). This led to an initial hypothesis that M1 Pro lacked hardware support for the operation. This was wrong.

**Root cause:** The GGML metadata context is pre-allocated with a fixed size formula before any tensors are created inside it. The formula was `2 * (1 + n_stream) * n_layer_kv * ggml_tensor_overhead()`, which accounts for one K tensor and one V tensor per layer. However, the KV cache constructor for TurboQuant types allocates two additional shared tensors after the layer loop: `turbo_rotation` and `turbo_rotation_inv`, each a 128x128 float32 matrix. These two tensors are shared across all layers but require their own descriptor entries in the metadata context. Since the formula did not count them, the context was full by the time the rotation matrices tried to allocate, and the assertion failed.

**Fix:** Add `+ 2 * ggml_tensor_overhead()` to the formula. This is a one-expression change that reserves space for exactly the two shared tensors.

**Significance:** This bug was a genuine software defect, not a hardware limitation. M1 Pro can run TurboQuant in this fork after the fix. The Metal Tensor API log line is printed during Metal device initialization regardless of what KV cache type is in use, and its presence does not indicate that the crash was Metal-related.

**Upstream status:** The same bug was independently diagnosed and fixed upstream by wxtry on 2026-03-29 in commit 70e45b7e on the TheTom llama-cpp-turboquant `feature/turboquant-kv-cache` branch. The upstream version reserves three extra slots rather than two, accounting for a third shared tensor (`turbo_innerq_scale_inv`) added in a later commit. No separate pull request from this evaluation was required.

### Bug B: Missing Metal Support for tq3\_0 (Aaryan Kapoor fork)

**Location:** `ggml/src/ggml-metal/ggml-metal.metal` and `ggml/src/ggml-metal/ggml-metal-device.m`

**Symptom:** Running with Metal model offload (`-ngl 99`) and tq3\_0 KV cache crashed with `pre-allocated tensor (cache_k_l0 (view)) in a buffer (MTL0) that cannot run the operation (SET_ROWS)`. Running with model offload but KV offload disabled failed with `Function kernel_flash_attn_ext_vec_tq3_0_dk128_dv128 was not found in the library`.

**Root cause:** Two separate gaps in Metal support. First, the Metal backend's `ggml_metal_device_supports_op` function had a per-type allowlist for SET\_ROWS that included f32, f16, bf16, q8\_0, q4\_0, q4\_1, q5\_0, q5\_1, and iq4\_nl, but not tq3\_0. Second, the Metal shader file instantiated Flash Attention kernels for f32, f16, bf16, q4\_0, q4\_1, q5\_0, q5\_1, and q8\_0, but not tq3\_0.

**Fix:** Add tq3\_0 support to both gaps. This required:
* A tq3\_0 quantize helper in the Metal shader
* A tq3\_0 dequantize helper in the Metal shader
* SET\_ROWS kernel instantiations for tq3\_0
* FLASH\_ATTN\_EXT and FLASH\_ATTN\_EXT\_VEC kernel instantiations for tq3\_0
* GET\_ROWS and copy support for tq3\_0
* Adding tq3\_0 to the Metal device allowlist

After these additions, the binary rebuilt cleanly and the run completed instead of crashing. Output quality was still wrong at that point (K-path correctness issue, addressed separately in Bug C).

### Bug C: Norm Correction and Zero Block Handling (Aaryan Kapoor fork)

**Location:** `ggml/src/ggml-quants.c` and `ggml/src/ggml-metal/ggml-metal.metal`

**Symptom:** After Metal support was added and tq3\_0 runs could complete, the output remained degenerate specifically on the K=tq3\_0 path. On any prompt, the model would produce repetitive or incoherent text when K used tq3\_0. V=tq3\_0 with K=f16 remained coherent. This localized the bug to the K-path quantization, not the attention computation or Metal kernels.

**Root cause 1 (norm storage):** The quantizer stored the raw RMS of the input block in `block_tq3_0.d`. After Lloyd-Max quantization and inverse Walsh-Hadamard reconstruction, the decoded block has a different norm than the original. The scale factor stored as raw RMS does not correct for this norm change. On decode, multiplying by raw RMS produces a block with the wrong magnitude. Key vectors with wrong magnitude produce wrong Q-K dot products. This is especially damaging for attention because the scores are softmax-normalized: a systematic scale error shifts the softmax distribution in ways that compound over long contexts.

**Root cause 2 (zero blocks):** When an input block had near-zero energy, the code used `rms = 1.0` as a guard against division by zero. This caused zero-energy input blocks to decode as structured nonzero values. Because the block should be zero, this decoded noise is all error. For key vectors, this type of garbage corrupts the attention scores for those positions.

**Fix 1 (norm correction):** Quantize and reconstruct the block inside the quantizer, measure the norm of the reconstruction, then store `original_norm / reconstruction_norm` as the scale factor. On decode, multiply by this factor. This ensures the decoded block has the same norm as the input regardless of how Lloyd-Max quantization changes the norm.

**Fix 2 (zero blocks):** When `original_norm < 1e-9`, set `d = 0` and zero all packed quantization data. On decode, a zero scale produces a zero output with no noise.

After both fixes, K=tq3\_0 paths produced coherent output on Metal and CPU. The sanity prompt ("What is the capital of France?") answered correctly in all configurations. Needle retrieval at 2K and 4K both passed, with both required facts (FROSTBLOCK-7 and VcMYB4) present in the responses.

### Upstream Status for Bugs B and C

The Metal GPU support additions for tq3\_0 (Bug B) and the norm correction plus zero block handling (Bug C) have been submitted upstream to the Aaryan Kapoor llama.cpp fork on the `turboquant-tq3_0` branch as pull request 1: https://github.com/Aaryan-Kapoor/llama.cpp/pull/1. The pull request is structured as two logical commits, one per fix, so each change can be reviewed independently.

---

## Secondary Observations

### Speed Reality on M1 Pro

The Aaryan fork's tq3\_0 is substantially slower than q8\_0 on M1 Pro at the tested context lengths. At 2K tokens, q8\_0 prefill ran at approximately 456 tokens per second while tq3\_0 prefill ran at approximately 23 tokens per second. At 4K, q8\_0 ran at 408 tokens per second while tq3\_0 ran at 12 tokens per second (`logs/needle-2k-q8_0.log`, `logs/needle-2k-tq3_0.log`, `logs/needle-4k-q8_0.log`, `logs/needle-4k-tq3_0.log`).

This is expected for an early implementation without optimized dequantization kernels. The dequantize path performs a full 128x128 matrix-vector multiply (the inverse Walsh-Hadamard rotation) for each block decoded. This is O(d^2) per block. An optimized implementation would use a fast Walsh-Hadamard transform at O(d log d). The TheTom benchmark results from an M5 Max system show 13 to 35 times slower generation with turbo3 compared to q8\_0 even on faster hardware, suggesting the bottleneck is structural in the current algorithm implementation, not specific to M1 Pro.

### TheTom turboquant\_plus Prototype Updates

Between Round 1 and Round 2, TheTom's Python prototype updated substantially. Round 1 found 144 tests. Round 2 found 538 tests collected, with 532 passing and 6 skipped. The test suite expanded significantly. The prototype's real-model validation on Qwen3-1.7B showed cosine similarity of 0.92 for uniform 3-bit and 0.97 for uniform 4-bit compression on real KV tensors, which is consistent with the paper's quality claims for those configurations.

### Memory: Smaller Cache, Higher Peak

On the MLX path the stored cache is smaller with either quantized configuration, but the MLX peak memory is higher (`logs/kv-memory-16k-2026-10-05.json`). At 16K tokens:

* FP16 stores 563.4 MB and peaks at 3076.8 MB.
* Stock MSE-only 4-bit stores 290.4 MB (1.94x smaller) and peaks at 3190.6 MB.
* Hybrid K5/V4 stores 435.6 MB (1.29x smaller) and peaks at 3245.4 MB.

Two properties of mlx-optiq 0.0.1 explain this. First, it stores one byte per element (uint8 codebook indices and int8 QJL signs), so a 4-bit cache takes about half the FP16 bytes, not a quarter. Second, the cache returns the dequantized K/V in float32. Attention and the residual stream then run in float32 for every layer after the first. The peak occurs during prefill, where the float32 buffers cost more than the smaller cache saves. A diagnostic run that casts the K/V back to FP16 before attention peaks at 2867.6 MB, 209 MB below the FP16 baseline. With packed bits and FP16 output, the Hybrid cache would be about 3.4x smaller than FP16.

The round 1 log reported 562 MB for FP16 and 140 MB for 4-bit TurboQuant at 16K, a 4.0x ratio. Those were formula figures for a packed 4-bit cache, not measurements. The round 1 projections for a 7B model at 64K and 128K context are formula figures too, and no run in this repository tests them.

For qwen2.5:3b on 16GB hardware the cache size is not operationally critical: the FP16 run peaks at 3.1 GB at 16K. A smaller cache would matter for larger models at longer contexts, but on the MLX path only with a packed layout and FP16 output.

---

## Reproduction on 2026-10-05

This pass reran the MLX headline numbers from committed code, with raw output. The modified mlx-optiq files behind the phase3 Hybrid runs were not kept, so `benchmarks/tq_patched.py` rebuilds the changes over the stock 0.0.1 package. `requirements.txt` pins the environment, with the same mlx, mlx-lm, mlx-optiq, transformers, and numpy versions as the 2026-07-03 pass. The machine is the same M1 Pro, now on macOS 27.0.1. `benchmarks/needle_repro.py` uses the phase3 prompt builder and scorer unchanged, and it writes each full response with exact-case and case-insensitive scores (`logs/needle-repro-2026-10-05.json`).

What reproduced:

* MLX FP16 baseline: 100% at all four lengths, with the same response text and the same MLX peak memory as phase3.
* Stock MSE-only 4-bit: 0% at all four lengths. Its peaks (2713.8, 2761.7, 2885.7, and 3190.6 MB) match the round 1 log figures (2714, 2762, 2886, and 3191 MB).
* Hybrid K5/V4: 50% at 2K and 100% at 16K, the same scores as the recorded run. At 16K the response has both facts in exact case.
* The QJL ablation keeps the same ordering of configurations (`logs/qjl-ablation-2026-10-05.json`).

What did not reproduce:

* Hybrid K5/V4 at 4K and 8K scored 50% instead of 100%. At 4K the response wrote "FRstblock-7". At 8K it named only the locus.
* No Hybrid response text matches phase3.
* The rebuild's Hybrid peak memory differs from phase3 by -48.7, +4.0, -170.4, and -58.2 MB at 2K, 4K, 8K, and 16K. The FP16 and stock peaks match exactly on the same machine, so the rebuild runs a different operation graph from the lost files. The cause was not found.
* No QJL ablation response text matches the July run, although the script and package versions are the same.
* The phase 2 short-prompt rerun (`logs/phase2-rerun-2026-10-05.log`) matches the round 1 FP16 text, but the MSE-only 4-bit text diverges after a few words.
* `benchmarks/test_hybrid_needle.py` fails (`logs/test-hybrid-needle-2026-10-05.log`). It tests 4-bit keys, not the Hybrid configuration.

MLX peak memory is a stable fingerprint of the operation graph: the FP16 and stock peaks match runs from earlier macOS versions. Response text through a quantized cache is not stable. The stock MSE-only and ablation texts changed with the OS update while the code and package versions stayed the same. Greedy decoding through a quantized cache turns small floating point differences into different tokens. The 4K and 8K Hybrid differences therefore have two possible causes: the rebuild differs from the lost files, and the OS changed. These runs cannot separate the two. The 2026-07-03 rerun with the original files is the evidence for the recorded 4K and 8K scores. The 16K pass is the only Hybrid pass that reproduces from committed code.
