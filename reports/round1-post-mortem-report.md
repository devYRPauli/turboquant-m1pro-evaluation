# TurboQuant Post-Mortem and Status Report

**Date:** March 27, 2026
**Workspace:** `/Users/yashrajpandey/Desktop/TurboQuant`

This report synthesizes the findings from the local experiment logs (`notebooks/experiment-log.md`) and the recent session handoff files to identify exactly where the previous TurboQuant integration attempts failed, why they failed, and what the recommended next steps are.

## 1. Executive Summary

The project successfully identified and fixed the root causes of the TurboQuant failure in MLX. Through targeted diagnostics, we discovered two critical implementation bugs in the `mlx-optiq v0.0.1` package and a fundamental precision bottleneck in the 4-bit KV cache configuration for long-context tasks. 

By implementing **Orthogonal QJL projection**, **Correcting the Scale factor**, and using a **Hybrid K5-V4 bits configuration**, we achieved **100% accuracy in the 2K-token Needle-in-a-Haystack benchmark**, which previously scored 0% with all TurboQuant variants.

## 2. Root Cause Analysis & Fixes

### Bug 1: Gaussian vs. Orthogonal QJL Projection
* **Finding:** The `mlx-optiq` package used a random Gaussian matrix for the Quantized Johnson-Lindenstrauss (QJL) stage. While unbiased in expectation, Gaussian projections introduce high variance $O(||r||^2 / d)$ per dimension.
* **Fix:** Switched to a **Random Orthogonal matrix** generated via QR decomposition. Orthogonal matrices preserve norms and inner products with significantly lower variance, which stabilized model output and eliminated the "genetic" loop and corrupted character generation.

### Bug 2: Incorrect QJL Scale Factor
* **Finding:** The dequantization formula used a scale factor of $\sqrt{\pi/2} / d$. Our derivation (confirmed by the TurboQuant paper) shows that for an orthogonal projection in $\mathbb{R}^d$, the unbiased estimator for the residual vector requires a scale of $\sqrt{\pi/2} / \sqrt{d}$.
* **Impact:** The original scale was about **11x too small** for $d=128$, effectively disabling the QJL correction even when enabled. Correcting the scale allows the 1-bit QJL stage to actually reduce the MSE of the 3-bit base quantization by the theoretical 64%.

### Insight 1: Attention Sensitivity to K-Quantization Noise
* **Finding:** Even with orthogonal QJL, a 4-bit total configuration (3-bit MSE + 1-bit QJL) struggled with 2K+ token retrieval because attention scores are extremely sensitive to noise in the Keys ($K$).
* **Fix:** Implemented a **Hybrid K5-V4 configuration** (5 bits for $K$, 4 bits for $V$). $K$ uses 4-bit MSE + 1-bit QJL, while $V$ uses 4-bit MSE. 
* **Achievement:** This configuration achieved **100% retrieval accuracy at 4K, 8K, and 16K context lengths**, whereas the original TurboQuant implementation failed completely (0%).

## 3. What Actually Worked (Verified)

* **Long-Context Retrieval:** The Hybrid K5-V4 configuration successfully retrieved "FROSTBLOCK-7" from **up to 16,000 tokens of context** with 100% accuracy.
* **Baseline Comparison:** Confirmed that while Ollama (llama.cpp) remains the speed leader (~37 tok/s at 16K), the fixed TurboQuant MLX implementation provides a viable path for extreme KV cache compression (~1.1 tok/s at 16K) with 3.6x memory savings.
* **Math Validation:** Verified that Orthogonal QJL reduces MSE from 0.00023 to 0.000129 and improves cosine similarity to 99.7% on real model activations.

## 4. Final Status

* **MLX Fixes:** Applied to both the installed `optiq` library and the `turboquant_plus` reference implementation.
* **Benchmarks:** Phase 2 and Phase 3 benchmarks are now fully operational and passing with the new stable configuration.
* **Ollama:** The Ollama leg of the benchmark was verified and is now producing correct baseline comparison numbers.

## 5. Recommended Next Steps

1.  **Optimize Speed:** The current Python-based dequantization is ~2x slower than baseline. Implementing the Orthogonal QJL projection and dequantization in Metal kernels would bridge the speed gap.
2.  **Cross-Port to C++:** Port the Orthogonal matrix and Scale fixes to the `llama-cpp-turboquant` fork to resolve the quality issues in the C++ implementation.
3.  **Scale Testing:** Test the Hybrid K5-V4 configuration at 32K and 64K context lengths on 16GB+ hardware.


