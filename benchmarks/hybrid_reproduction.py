"""Faithful reproduction of the phase3 MLX TurboQuant-Hybrid path.

Uses the exact original harness:
  - build_prompt / score_answer from benchmarks/build_prompt.py
  - make_turbo_kv_caches(bits=(5,4), use_qjl=True, seed=42) from the modified optiq
    package (orthogonal QJL projection, sqrt(d) scale, k_damping=0.7 default)

Goal: settle the 2K needle score and re-confirm 4K/8K/16K for the validated
Hybrid K5/V4 configuration, comparable to benchmarks/phase3_results.json.
"""

import json
import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate_step
from mlx_lm.sample_utils import make_sampler

import os
BENCH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BENCH)
from build_prompt import build_prompt, score_answer  # noqa: E402
from optiq.core.turbo_kv_cache import make_turbo_kv_caches  # noqa: E402

MODEL_ID = "mlx-community/Qwen2.5-3B-Instruct-4bit"
MAX_NEW_TOKENS = 80
TARGETS = [2000, 4000, 8000, 16000]


def run(model, tokenizer, target):
    n_layers = len(model.layers)
    prompt, actual_tokens, needle_pos = build_prompt(target, tokenizer)
    tokens = mx.array(tokenizer.encode(prompt))
    cache = make_turbo_kv_caches(n_layers, 128, bits=(5, 4), use_qjl=True, seed=42)
    sampler = make_sampler(temp=0.0)
    mx.reset_peak_memory()
    gen = []
    t0 = time.perf_counter()
    for tok, _ in generate_step(tokens, model, max_tokens=MAX_NEW_TOKENS, sampler=sampler, prompt_cache=cache):
        gen.append(int(tok))
    dt = time.perf_counter() - t0
    resp = tokenizer.decode(gen)
    scored = score_answer(resp)
    return {
        "config": "Hybrid K5/V4 (orthogonal QJL, sqrt(d) scale, 0.7 damping)",
        "target_tokens": target,
        "actual_tokens": actual_tokens,
        "needle_pos_tokens": needle_pos,
        "needle_score": scored["score"],
        "found_keywords": scored["found_keywords"],
        "missing_keywords": scored["missing_keywords"],
        "tokens_per_s": round(len(gen) / dt, 2) if dt > 0 else 0.0,
        "mlx_peak_mb": round(mx.get_peak_memory() / (1024 ** 2), 1),
        "gen_seconds": round(dt, 2),
        "response_preview": scored["response_preview"],
    }


def main():
    print("Loading", MODEL_ID, flush=True)
    model, tokenizer = load(MODEL_ID)
    mx.eval(model.parameters())
    results = []
    for t in TARGETS:
        r = run(model, tokenizer, t)
        print(json.dumps(r), flush=True)
        results.append(r)
    out = Path(BENCH).parent / "logs" / "hybrid-reproduction.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print("WROTE", out, flush=True)


if __name__ == "__main__":
    main()
