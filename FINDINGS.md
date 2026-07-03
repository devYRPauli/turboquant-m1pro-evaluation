# Technical Findings: TurboQuant on Apple M1 Pro 16GB

This document provides a detailed technical writeup of the three main discoveries from this evaluation. All data and observations come from the experiment logs, debug logs, and reports in this repository.

---

## Finding 1: The Paper's QJL Construction Fails in Practice at Head Dimension 128; an Orthogonal Variant Fixes It

### What QJL Is Supposed to Do

The TurboQuant paper's two-stage design works as follows. Stage 1 (PolarQuant) applies a random orthogonal rotation to each KV vector, making the coordinates approximately Gaussian, then quantizes each coordinate independently using Lloyd-Max centroids. Stage 2 (QJL) computes the sign of a linear projection of the quantization residual (the difference between the original vector and the PolarQuant reconstruction). This 1-bit sign vector, combined with a scale factor, provides an unbiased correction to the inner product errors introduced by stage 1.

The key mathematical claim is that for a random projection matrix S, the estimator `scale * (sign(S * residual)) @ S` approximates the residual vector with zero bias. This is derived from Johnson-Lindenstrauss lemma results. When computing attention scores (query dot key), you can add this correction to get a better estimate of the true inner product.

The paper (Definition 1 and Algorithm 2) defines S as a d x d matrix with i.i.d. standard Gaussian entries and the dequantization scale as `sqrt(pi/2) / d`. That pair is self-consistent and unbiased: Gaussian rows have norm near `sqrt(d)`, so the projection output is about `sqrt(d)` larger than it would be with orthonormal rows, and the `1/d` scale absorbs exactly that factor. For an orthogonal projection matrix (orthonormal rows), the matching unbiased scale is `sqrt(pi/2) / sqrt(d)`. The two conventions differ by `sqrt(d)` and are interchangeable in expectation; what differs between them is variance.

### What the Implementations Actually Did

Both stock implementations were faithful to the paper. The pristine mlx-optiq v0.0.1 wheel from PyPI and the pre-fix state of TheTom turboquant\_plus (visible in the before side of pull request 93) both used a random Gaussian matrix for S with the scale `sqrt(pi/2) / d`, exactly as the paper specifies. Neither contained a transcription error in the QJL math. The installed package inside the round 1 virtual environment differs from the PyPI wheel because the fixes described below were applied to it in place during the project.

**What went wrong in practice.** The paper-faithful construction is unbiased, but its variance proved fatal on real LLM KV vectors at head dimension 128. Every token that passes through the quantizer accumulates a small error in the reconstructed attention key. Over thousands of tokens in a long context, these errors compound. The attention score distribution distorts progressively, eventually causing the softmax to collapse toward a single token or distribute nonsensically. The observed symptom was immediate word-loop degeneration the moment QJL was enabled at any context length, even 36 tokens.

**The fix: a coupled substitution, not two bug fixes.** Replacing the Gaussian matrix with an orthogonal matrix from QR decomposition eliminated the degeneration. Orthogonal matrices preserve norms and inner products exactly, so the projection step introduces no distortion of its own. Because the matrix changed, the scale had to change with it: `sqrt(pi/2) / d` is the correct unbiased scale for a Gaussian matrix, and `sqrt(pi/2) / sqrt(d)` is the correct scale for an orthogonal one. Keeping the Gaussian scale with the orthogonal matrix would make the correction approximately 11x too small at d=128 (`sqrt(128) = 11.31`). Earlier versions of this write-up, and the commit message merged upstream, described the scale itself as an 11x bug in the stock code; that framing is wrong, because in the stock code the `1/d` scale was paired with the Gaussian matrix it belongs to. In expectation the stock correction term had the same magnitude as the fixed one. The difference the fix makes is in variance, not bias.

The validated configuration additionally applies a damping factor of 0.7 to the QJL correction term. This is close to the MMSE-optimal shrinkage `2/pi` (approximately 0.6366): the unbiased estimator inflates reconstruction energy (`E[||x_hat||^2] = (pi/2) * ||x||^2`), and shrinking it toward zero trades a small bias for a lower mean squared error. The projection, scale, and damping changes were applied together and were not ablated individually, so their individual contributions to the 0% to 100% result are not isolated in this data.

### Why Every Implementation Dropped QJL

Both primary implementations tested (mlx-optiq and TheTom turboquant\_plus) independently arrived at the conclusion that MSE-only quantization worked better in practice than the two-stage design. The experiment log notes that both Aaryan Kapoor and TheTom independently dropped QJL in favor of MSE-only with all bits going to Lloyd-Max centroids.

This is consistent with the finding above. A paper-faithful Gaussian QJL really does make output worse at head dimension 128, so the natural engineering response was to disable the stage. The conclusion of this evaluation is that the two-stage design itself is sound: it works once the projection is orthogonal, the scale matches, and the correction is damped.

The post-mortem report records that with the fixes applied, the QJL correction reduces MSE from 0.00023 to 0.000129 (a 44 percent reduction) and improves cosine similarity to 99.7 percent on real model activations. The 44 percent figure matches the theoretical `(pi/2 - 1)`, approximately 43 percent, for the undamped estimator. The theoretical maximum with MMSE-optimal shrinkage `2/pi` is `1 - 2/pi`, approximately 64 percent.

### What the Fix Required

The fix as implemented in `benchmarks/test_hybrid_needle.py`:

1. Generate the QJL projection matrix S using QR decomposition of a Gaussian random draw, then enforce a consistent sign convention by multiplying columns by the signs of the diagonal of R.

2. Change the dequantization scale from `math.sqrt(math.pi / 2.0) / self.d` to the matching `math.sqrt(math.pi / 2.0) / math.sqrt(self.d)`.

3. Multiply the QJL correction term by a damping factor of 0.7 (approximately the MMSE-optimal shrinkage `2/pi`).

These changes together, combined with the Hybrid K5/V4 configuration described in Finding 2, moved needle retrieval from 0% to 100% at 4K, 8K, and 16K tokens. At 2K the recorded run scored 50% (see the note under the results table in the README): the model retrieved one of the two needle facts.

### Upstream Status

The QJL orthogonal projection and `sqrt(d)` scale factor changes were merged into the TheTom turboquant\_plus reference Python implementation on 2026-05-28 as commit 0cb20bca via pull request 93: https://github.com/TheTom/turboquant_plus/pull/93. The merged version adds a `shrinkage` parameter on the dequantize path. Default is 1.0 (classical unbiased estimator). The MMSE-optimal value is `2/np.pi`, approximately 0.6366, derived from the closed forms `E[||x_hat||^2] = (pi/2) * ||x||^2` and `E[<x_hat, x>] = ||x||^2`; this derivation is documented in the dequantize docstring. The PR also adds an orthogonality contract assertion in `__init__` and a corresponding test at d in {64, 128, 256, 512} with tolerance `1e-12`. One caveat: the merged commit message describes the scale change as fixing a formula that was "11x too small"; as explained above, that factor only arises relative to the orthogonal matrix introduced in the same change, since the stock Gaussian projection with the `1/d` scale was the paper's own unbiased construction.

---

## Finding 2: Keys Are More Sensitive Than Values

### The Asymmetry

Even with the working QJL variant (orthogonal projection, matched scale factor, damping), a symmetric 4-bit allocation failed at 2K context. Assigning 4 bits to both K and V, with K using 4-bit MSE plus 1-bit QJL and V using the same, was not enough for reliable fact retrieval.

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

The average bit rate is 4.5 bits per cache element, which is slightly above the 4-bit symmetric case. The memory savings compared to FP16 are still approximately 3.6x.

This configuration achieved 100% needle retrieval at 4K, 8K, and 16K tokens, where the stock 4-bit symmetric configuration achieved 0% at all lengths. At 2K the recorded run scored 50% (one of the two needle facts retrieved); see the note under the results table in the README.

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

The Aaryan fork's tq3\_0 is substantially slower than q8\_0 on M1 Pro at the tested context lengths. At 2K tokens, q8\_0 prefill ran at approximately 456 tokens per second while tq3\_0 prefill ran at approximately 23 tokens per second. At 4K, q8\_0 ran at 408 tokens per second while tq3\_0 ran at 12 tokens per second.

This is expected for an early implementation without optimized dequantization kernels. The dequantize path performs a full 128x128 matrix-vector multiply (the inverse Walsh-Hadamard rotation) for each block decoded. This is O(d^2) per block. An optimized implementation would use a fast Walsh-Hadamard transform at O(d log d). The TheTom benchmark results from an M5 Max system show 13 to 35 times slower generation with turbo3 compared to q8\_0 even on faster hardware, suggesting the bottleneck is structural in the current algorithm implementation, not specific to M1 Pro.

### TheTom turboquant\_plus Prototype Updates

Between Round 1 and Round 2, TheTom's Python prototype updated substantially. Round 1 found 144 tests. Round 2 found 538 tests collected, with 532 passing and 6 skipped. The test suite expanded significantly. The prototype's real-model validation on Qwen3-1.7B showed cosine similarity of 0.92 for uniform 3-bit and 0.97 for uniform 4-bit compression on real KV tensors, which is consistent with the paper's quality claims for those configurations.

### Memory Savings Are Real

The theoretical 4x compression ratio was confirmed at all tested context lengths. At 16K tokens, FP16 KV cache uses 562 MB and 4-bit TurboQuant uses 140 MB. For qwen2.5:3b on 16GB hardware the savings are not operationally critical because the total memory budget is comfortable even without compression. The value proposition grows with larger models at longer contexts. A 7B model at 128K context would use approximately 9.2 GB for the KV cache in FP16 versus approximately 2.3 GB with 4-bit TurboQuant, which is the difference between OOM risk and a comfortable fit on 16GB hardware (at 64K the corresponding figures are approximately 4.6 GB versus 1.2 GB).
