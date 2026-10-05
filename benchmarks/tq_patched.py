"""Hybrid K5/V4 TurboQuant KV cache, rebuilt over stock mlx-optiq 0.0.1.

The phase3 Hybrid results were produced with two files of the installed
mlx-optiq 0.0.1 package edited in place (optiq/core/turbo_quant.py and
optiq/core/turbo_kv_cache.py). Those edited files were not kept. This module
recreates the documented changes as subclasses of the stock classes, so the
configuration can be rerun from a clean install of the pinned requirements.
It is not op-for-op identical to the lost files: its MLX peak memory differs
from the phase3 Hybrid runs by up to 170 MB, while the FP16 and stock MSE
peaks reproduce exactly (logs/needle-repro-2026-10-05.json). The changes:

  1. QJL projection: random orthogonal matrix (QR of a Gaussian draw, the same
     construction the package already uses for its rotation) instead of the
     stock Gaussian matrix. Same seed offset (seed + 1000), stored as float16
     like the stock matrix.
  2. QJL dequant scale: sqrt(pi/2) / sqrt(d), which matches the orthogonal
     matrix, instead of the stock sqrt(pi/2) / d for the Gaussian matrix.
  3. Damping: the key QJL correction is multiplied by k_damping (0.7).
  4. Hybrid bits: keys use TurboQuantProd with 5 bits (4-bit MSE plus 1-bit
     QJL), values use TurboQuantMSE with 4 bits (no QJL).

make_turbo_kv_caches() keeps the call signature that phase3_long_context.py
and hybrid_reproduction.py used: bits=(5, 4), use_qjl=True, seed=42.
"""

import math

import mlx.core as mx
from optiq.core.turbo_kv_cache import TurboQuantKVCache
from optiq.core.turbo_quant import (
    TurboQuantMSE,
    TurboQuantProd,
    generate_rotation_matrix,
)


class OrthogonalQJLProd(TurboQuantProd):
    """TurboQuantProd with orthogonal QJL projection, matched scale, damping."""

    def __init__(self, d: int, bits: int, seed: int = 42, damping: float = 0.7):
        super().__init__(d, bits, seed)
        self.qjl = generate_rotation_matrix(d, seed + 1000).astype(mx.float16)
        self.damping = damping

    def dequantize(self, mse_indices, qjl_signs, residual_norms, norms):
        x_mse = self.mse.dequantize(mse_indices, mx.ones_like(norms))
        scale = math.sqrt(math.pi / 2) / math.sqrt(self.d)
        qjl_correction = (
            scale
            * residual_norms
            * (qjl_signs.astype(mx.float16) @ self.qjl)
        )
        return (x_mse + qjl_correction * self.damping) * norms


class HybridTurboKVCache(TurboQuantKVCache):
    """K = TurboQuantProd(k_bits), V = TurboQuantMSE(v_bits).

    patched=True uses OrthogonalQJLProd for keys (the phase3 configuration).
    patched=False uses the stock Gaussian TurboQuantProd for keys, which is the
    same bit allocation with the paper-faithful QJL and no damping.
    """

    def __init__(
        self,
        head_dim: int = 128,
        bits: tuple[int, int] = (5, 4),
        seed: int = 42,
        k_damping: float = 0.7,
        patched: bool = True,
    ):
        k_bits, v_bits = bits
        super().__init__(head_dim, v_bits, use_qjl=False, seed=seed)
        if patched:
            self.k_quantizer = OrthogonalQJLProd(head_dim, k_bits, seed, damping=k_damping)
        else:
            self.k_quantizer = TurboQuantProd(head_dim, k_bits, seed)
        self.v_quantizer = TurboQuantMSE(head_dim, v_bits, seed + 500)


def make_turbo_kv_caches(
    n_layers: int,
    head_dim: int = 128,
    bits: tuple[int, int] = (5, 4),
    use_qjl: bool = True,
    seed: int = 42,
    k_damping: float = 0.7,
    patched: bool = True,
) -> list[HybridTurboKVCache]:
    """One HybridTurboKVCache per layer, seeded seed + layer index like stock."""
    if not use_qjl:
        raise ValueError("Hybrid K5/V4 needs use_qjl=True; use the stock "
                         "optiq make_turbo_kv_caches for MSE-only caches")
    return [
        HybridTurboKVCache(head_dim, bits, seed + i, k_damping=k_damping, patched=patched)
        for i in range(n_layers)
    ]
