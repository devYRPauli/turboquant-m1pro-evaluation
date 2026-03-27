"""
Phase 2 Benchmark: Baseline vs TurboQuant KV Cache
====================================================
Loads mlx-community/Qwen2.5-3B-Instruct-4bit and runs:
  1. Baseline: standard mlx-lm KVCache (no compression)
  2. TurboQuant-MSE bits=4, no QJL (working, ~3.8× compression)
  3. TurboQuant-MSE bits=3, no QJL (working, ~4.9× compression)
  4. TurboQuant-MSE bits=2, no QJL (aggressive, ~7.1× compression)
  5. TurboQuant-MSE+QJL bits=4 (v0.0.1 has QJL noise issue — documented)

NOTE on mlx-optiq v0.0.1 QJL bug:
  use_qjl=True produces degenerate looping output on real model activations.
  use_qjl=False (MSE-only) works correctly.
  Root cause: QJL correction formula adds high-variance noise to real LLM KV
  vectors. Mathematically unbiased in expectation, but variance dominates for
  d=128 head_dim with real (non-i.i.d.) activations. Documented as a known
  limitation of mlx-optiq v0.0.1.

Usage:
    source .venv-mlx/bin/activate
    python3 benchmarks/phase2_inference_compare.py
"""

import time
import math
import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate_step
from mlx_lm.sample_utils import make_sampler
from mlx_lm.models.cache import KVCache

from optiq.core.turbo_kv_cache import TurboQuantKVCache, make_turbo_kv_caches

MODEL_ID = "mlx-community/Qwen2.5-3B-Instruct-4bit"
MAX_TOKENS = 150
TEMPERATURE = 0.0

PROMPT = (
    "What is the role of anthocyanins in blueberry fruit development? "
    "Provide a concise but detailed explanation covering biosynthesis, "
    "genetic regulation, and why they matter for fruit quality."
)


def get_model_dims(model):
    n_layers = len(model.layers)
    layer = model.layers[0]
    attn = layer.self_attn
    n_kv_heads = attn.n_kv_heads
    head_dim = getattr(attn, "head_dim", None)
    if head_dim is None:
        head_dim = round(1.0 / (attn.scale ** 2))
    return n_layers, n_kv_heads, head_dim


def estimate_kv_bytes(n_layers, n_kv_heads, head_dim, seq_len, bits_per_elem):
    bytes_per_elem = bits_per_elem / 8.0
    return 2 * n_layers * seq_len * n_kv_heads * head_dim * bytes_per_elem


def get_active_memory_mb():
    try:
        return mx.get_active_memory() / (1024 * 1024)
    except Exception:
        try:
            return mx.metal.get_active_memory() / (1024 * 1024)
        except Exception:
            return 0.0


def run_inference(model, tokenizer, prompt, cache_list, label):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    tokens = tokenizer.encode(prompt)
    prompt_tokens = mx.array(tokens)
    sampler = make_sampler(temp=TEMPERATURE)

    generated_tokens = []
    token_times = []

    mem_before = get_active_memory_mb()
    tic = time.perf_counter()
    first_token_time = None

    for token, _logprobs in generate_step(
        prompt_tokens,
        model,
        max_tokens=MAX_TOKENS,
        sampler=sampler,
        prompt_cache=cache_list,
    ):
        now = time.perf_counter()
        if first_token_time is None:
            first_token_time = now - tic
        token_times.append(now - tic)
        tic = now
        tok_int = tok_int = token.item() if hasattr(token, "item") else int(token)
        generated_tokens.append(tok_int)

    mem_after = get_active_memory_mb()
    response_text = tokenizer.decode(generated_tokens)
    n_gen = len(generated_tokens)
    total_time = sum(token_times)
    tok_per_s = n_gen / total_time if total_time > 0 else 0.0

    print(f"  Generated: {n_gen} tokens in {total_time:.2f}s  →  {tok_per_s:.1f} tok/s")
    print(f"  First tok: {(first_token_time or 0)*1000:.0f} ms")
    print(f"  Memory:    {mem_before:.0f} → {mem_after:.0f} MB  (Δ {mem_after-mem_before:+.0f} MB)")
    print(f"\n  Response:\n  {response_text[:320]}{'...' if len(response_text) > 320 else ''}")

    return {
        "label": label,
        "n_prompt_tokens": len(tokens),
        "n_generated_tokens": n_gen,
        "total_time_s": round(total_time, 3),
        "tok_per_s": round(tok_per_s, 1),
        "first_token_ms": round((first_token_time or 0) * 1000, 0),
        "mem_before_mb": round(mem_before, 1),
        "mem_after_mb": round(mem_after, 1),
        "mem_delta_mb": round(mem_after - mem_before, 1),
        "response_text": response_text,
    }


if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"  TurboQuant Phase 2 — Inference Benchmark")
    print(f"  Model: {MODEL_ID}")
    print(f"  Max tokens: {MAX_TOKENS}")
    print(f"{'='*60}\n")

    print(f"Loading {MODEL_ID}...")
    model, tokenizer = load(MODEL_ID)
    mx.eval(model.parameters())

    n_layers, n_kv_heads, head_dim = get_model_dims(model)
    print(f"\nModel: {n_layers} layers, {n_kv_heads} KV heads, head_dim={head_dim}")

    seq_512 = 512
    fp16_bytes = estimate_kv_bytes(n_layers, n_kv_heads, head_dim, seq_512, 16)
    print(f"\nTheoretical KV memory at {seq_512} tokens:")
    for bits, tag in [(16, "FP16 baseline"), (4, "4-bit TQ-MSE"), (3, "3-bit TQ-MSE"), (2, "2-bit TQ-MSE")]:
        kv_mb = estimate_kv_bytes(n_layers, n_kv_heads, head_dim, seq_512, bits) / (1024*1024)
        ratio = fp16_bytes / estimate_kv_bytes(n_layers, n_kv_heads, head_dim, seq_512, bits)
        print(f"  {tag:<20}: {kv_mb:.1f} MB  ({ratio:.1f}× vs FP16)")

    results = []

    # ── 1. Baseline ──────────────────────────────────────────────────────────
    r = run_inference(model, tokenizer, PROMPT,
                      [KVCache() for _ in range(n_layers)],
                      "Baseline — no compression (FP16 KV)")
    r["kv_bits"] = 16
    results.append(r)

    # ── 2. TurboQuant-MSE bits=4, no QJL ────────────────────────────────────
    r = run_inference(model, tokenizer, PROMPT,
                      make_turbo_kv_caches(n_layers, head_dim, bits=4, use_qjl=False, seed=42),
                      "TurboQuant-MSE bits=4, no QJL  (3.8× compression)")
    r["kv_bits"] = 4
    results.append(r)

    # ── 3. TurboQuant-MSE bits=3, no QJL ────────────────────────────────────
    r = run_inference(model, tokenizer, PROMPT,
                      make_turbo_kv_caches(n_layers, head_dim, bits=3, use_qjl=False, seed=42),
                      "TurboQuant-MSE bits=3, no QJL  (4.9× compression)")
    r["kv_bits"] = 3
    results.append(r)

    # ── 4. TurboQuant-MSE bits=2, no QJL ────────────────────────────────────
    r = run_inference(model, tokenizer, PROMPT,
                      make_turbo_kv_caches(n_layers, head_dim, bits=2, use_qjl=False, seed=42),
                      "TurboQuant-MSE bits=2, no QJL  (7.1× compression)")
    r["kv_bits"] = 2
    results.append(r)

    # ── 5. TurboQuant bits=4 WITH QJL — expected to degrade (bug doc) ────────
    r = run_inference(model, tokenizer, PROMPT,
                      make_turbo_kv_caches(n_layers, head_dim, bits=4, use_qjl=True, seed=42),
                      "TurboQuant bits=4, WITH QJL  [v0.0.1 bug — expect degradation]")
    r["kv_bits"] = "4+QJL"
    results.append(r)

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Config':<42} {'tok/s':>6} {'1st ms':>7} {'MemΔ':>7}  KV bits")
    print(f"  {'-'*42} {'-'*6} {'-'*7} {'-'*7}  -------")
    for r in results:
        print(
            f"  {r['label']:<42} "
            f"{r['tok_per_s']:>6.1f} "
            f"{r['first_token_ms']:>7.0f} "
            f"{r['mem_delta_mb']:>+6.0f}MB "
            f"  {r['kv_bits']}"
        )

    print(f"\n\n{'='*60}")
    print("  RESPONSE QUALITY (first 250 chars)")
    print(f"{'='*60}")
    for r in results:
        print(f"\n  [{r['label']}]")
        print(f"  {r['response_text'][:250]}")

    print("\n\nDone.")
