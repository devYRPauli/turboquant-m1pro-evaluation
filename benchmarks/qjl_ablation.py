"""QJL ablation for TurboQuant on Apple M1 Pro.

Isolates the contribution of each change that separates the stock (paper-faithful)
QJL configuration from the validated Hybrid K5/V4 configuration:

  1. projection matrix:  gaussian  vs  orthogonal (QR)
  2. dequant scale:      sqrt(pi/2)/d  vs  sqrt(pi/2)/sqrt(d)
  3. damping:            1.0 (undamped)  vs  0.7 (~ MMSE shrinkage 2/pi)

Scope: this script uses a lightweight synthetic prompt (a single repeated filler
sentence with the needle at the midpoint) and NO instruct chat template. It is a
qualitative probe of generation stability, that is, whether a configuration
degenerates into word loops, produces semi-coherent fragments, or produces
coherent text. Because there is no chat framing, even a stable configuration
tends to continue the passage rather than answer, so the needle_score here is not
a retrieval measurement. For actual needle retrieval numbers under the validated
configuration, use hybrid_reproduction.py, which reuses the full phase3 harness
(build_prompt with instruct framing plus make_turbo_kv_caches).

Run with a Python environment that has the modified optiq package installed
(orthogonal QJL, sqrt(d) scale). Fixed seeds; greedy decoding (temp 0).
"""

import json
import math
import sys
import time

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate_step
from mlx_lm.sample_utils import make_sampler

from optiq.core.turbo_kv_cache import TurboQuantKVCache
from optiq.core.turbo_quant import TurboQuantProd, TurboQuantMSE

MODEL_ID = "mlx-community/Qwen2.5-3B-Instruct-4bit"
HEAD_DIM = 128


def orthogonal_matrix(d, seed):
    key = mx.random.key(seed)
    G = mx.random.normal(shape=(d, d), key=key)
    Q, R = mx.linalg.qr(G, stream=mx.cpu)
    mx.eval(Q, R)
    Q = Q * mx.sign(mx.diag(R))[None, :]
    mx.eval(Q)
    return Q.astype(mx.float16)


class ConfigurableProd(TurboQuantProd):
    """QJL prod quantizer with switchable projection, scale, and damping."""

    def __init__(self, d, bits, seed=42, projection="gaussian", scale="d", damping=1.0):
        super().__init__(d, bits, seed)
        self._proj = projection
        self._scale = scale
        self._damping = damping
        if projection == "orthogonal":
            self.qjl = orthogonal_matrix(d, seed + 1000)
        # gaussian: keep the stock self.qjl from the parent (Gaussian draw)

    def dequantize(self, mse_indices, qjl_signs, residual_norms, norms):
        x_mse = self.mse.dequantize(mse_indices, mx.ones_like(norms))
        if self._scale == "d":
            scale = math.sqrt(math.pi / 2.0) / self.d
        else:
            scale = math.sqrt(math.pi / 2.0) / math.sqrt(self.d)
        qjl_correction = scale * residual_norms * (qjl_signs.astype(mx.float16) @ self.qjl)
        return (x_mse + qjl_correction * self._damping) * norms


class AblationCache(TurboQuantKVCache):
    """Hybrid K5/V4 cache: K = configurable prod quantizer, V = MSE-only 4-bit."""

    def __init__(self, head_dim, seed=42, projection="gaussian", scale="d", damping=1.0):
        super().__init__(head_dim, bits=4, use_qjl=True, seed=seed)
        self.k_quantizer = ConfigurableProd(
            head_dim, 4, seed, projection=projection, scale=scale, damping=damping
        )
        self.v_quantizer = TurboQuantMSE(head_dim, 4, seed + 500)


def build_needle_prompt(n_tokens, tokenizer):
    needle = ("IMPORTANT FACT: The experimental blueberry accession designated UF-B4291 "
              "carries a novel disease-resistance allele called FROSTBLOCK-7 in the VcMYB4 locus.")
    question = ("What is the name of the disease-resistance allele found in blueberry accession "
                "UF-B4291, and in which locus was it identified?")
    filler = "Blueberry breeding is a complex process involving genomic selection and phenotyping. " * 1000
    filler_tokens = tokenizer.encode(filler)
    target = n_tokens - len(tokenizer.encode(needle)) - len(tokenizer.encode(question)) - 50
    filler_tokens = filler_tokens[:target]
    mid = len(filler_tokens) // 2
    toks = filler_tokens[:mid] + tokenizer.encode(needle) + filler_tokens[mid:] + tokenizer.encode(question)
    return mx.array(toks), len(toks)


def score(text):
    a = "FROSTBLOCK-7" in text
    b = "VcMYB4" in text
    return (1.0 if (a and b) else 0.5 if (a or b) else 0.0), a, b


def run(model, tokenizer, n_ctx, projection, scale, damping, max_tokens=80):
    n_layers = len(model.layers)
    caches = [AblationCache(HEAD_DIM, seed=42 + i, projection=projection, scale=scale, damping=damping)
              for i in range(n_layers)]
    prompt, ntok = build_needle_prompt(n_ctx, tokenizer)
    sampler = make_sampler(temp=0.0)
    text = ""
    t0 = time.time()
    for token, _ in generate_step(prompt, model, max_tokens=max_tokens, sampler=sampler, prompt_cache=caches):
        text += tokenizer.decode([int(token)])
        if "\n\n" in text:
            break
    dt = time.time() - t0
    sc, a, b = score(text)
    return {
        "n_ctx": n_ctx, "actual_tokens": ntok, "projection": projection, "scale": scale,
        "damping": damping, "needle_score": sc, "found_FROSTBLOCK-7": a, "found_VcMYB4": b,
        "gen_seconds": round(dt, 2), "response": text.strip()[:280],
    }


def main():
    print("Loading", MODEL_ID, flush=True)
    model, tokenizer = load(MODEL_ID)
    results = []

    # Ablation matrix at 2K (the disputed length).
    matrix = [
        ("gaussian",   "d",    1.0),  # stock / paper-faithful
        ("gaussian",   "sqrt", 1.0),  # mismatched scale on Gaussian
        ("orthogonal", "d",    1.0),  # mismatched scale on orthogonal
        ("orthogonal", "sqrt", 1.0),  # matched, undamped
        ("orthogonal", "sqrt", 0.7),  # matched, damped = validated config
        ("gaussian",   "d",    0.7),  # stock + damping only
    ]
    for proj, sc, dmp in matrix:
        r = run(model, tokenizer, 2000, proj, sc, dmp)
        print(json.dumps(r), flush=True)
        results.append(r)

    import os
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "logs", "qjl-ablation.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print("WROTE", out, flush=True)


if __name__ == "__main__":
    main()
