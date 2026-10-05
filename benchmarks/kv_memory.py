"""KV cache memory at 16K: stored cache bytes and MLX peak memory.

Runs the needle_repro.py configs on the 16K needle prompt in alternating
rounds and records, per run (MB = 2**20 bytes):

  weights_mb          MLX active memory before the run (model weights)
  prefill_peak_mb     MLX peak from the start of the run to the first token
  decode_peak_mb      MLX peak over the remaining decode steps
  peak_mb             max of the two; the same quantity phase3 reports
  active_prefill_mb   MLX active memory after the first token
  active_end_mb       MLX active memory after the last token
  freed_by_cache_mb   drop in active memory when the cache objects are deleted
  cache_allocated_mb  sum of nbytes of the arrays the cache objects hold
  cache_filled_mb     the same, counting only positions up to the offset
  kv_out_dtype        dtype of the K/V arrays the cache returns to attention

The extra config patched_k5v4_f16out is a diagnostic: the patched cache with
its dequantized K/V cast back to the input dtype (float16) before attention.
Stock and patched optiq caches return float32, so attention runs on float32
K/V over the full sequence; this config isolates that cost.

Usage:
  python benchmarks/kv_memory.py --out logs/kv-memory-16k-2026-10-05.json
"""

import argparse
import gc
import json
import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm.generate import generate_step
from mlx_lm.sample_utils import make_sampler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_prompt import build_prompt  # noqa: E402
from needle_repro import (  # noqa: E402
    CONFIGS, HEAD_DIM, MAX_NEW_TOKENS, MB, cache_bytes, environment, load_model)
from tq_patched import HybridTurboKVCache  # noqa: E402

TARGET = 16000
FIELDS = ["weights_mb", "prefill_peak_mb", "decode_peak_mb", "peak_mb",
          "active_prefill_mb", "active_end_mb", "freed_by_cache_mb",
          "cache_allocated_mb", "cache_filled_mb"]


class F16OutCache(HybridTurboKVCache):
    def update_and_fetch(self, keys, values):
        k, v = super().update_and_fetch(keys, values)
        return k.astype(keys.dtype), v.astype(values.dtype)


CONFIGS = CONFIGS | {
    "patched_k5v4_f16out": lambda n: [F16OutCache(HEAD_DIM, (5, 4), 42 + i) for i in range(n)],
}


def kv_out_dtype(make_cache):
    c = make_cache(1)[0]
    x = mx.zeros((1, 2, 4, HEAD_DIM), dtype=mx.float16)
    k, _ = c.update_and_fetch(x, x)
    return str(k.dtype)


def run(model, tokens, config):
    gc.collect()
    mx.clear_cache()
    weights = mx.get_active_memory()
    cache = CONFIGS[config](len(model.layers))
    mx.reset_peak_memory()
    gen = generate_step(tokens, model, max_tokens=MAX_NEW_TOKENS,
                        sampler=make_sampler(temp=0.0), prompt_cache=cache)
    next(gen)
    prefill_peak = mx.get_peak_memory()
    active_prefill = mx.get_active_memory()
    mx.reset_peak_memory()
    n = 1 + sum(1 for _ in gen)
    decode_peak = mx.get_peak_memory()
    active_end = mx.get_active_memory()
    allocated, filled = cache_bytes(cache)
    del gen
    gc.collect()
    before_del = mx.get_active_memory()
    del cache
    gc.collect()
    freed = before_del - mx.get_active_memory()
    r = {
        "config": config,
        "generated_tokens": n,
        "weights_mb": weights / MB,
        "prefill_peak_mb": prefill_peak / MB,
        "decode_peak_mb": decode_peak / MB,
        "peak_mb": max(prefill_peak, decode_peak) / MB,
        "active_prefill_mb": active_prefill / MB,
        "active_end_mb": active_end / MB,
        "freed_by_cache_mb": freed / MB,
        "cache_allocated_mb": allocated / MB,
        "cache_filled_mb": filled / MB,
    }
    return {k: round(v, 1) if isinstance(v, float) else v for k, v in r.items()}


def summarize(runs, configs):
    summary = {}
    for config in configs:
        rows = [r for r in runs if r["config"] == config]
        if not rows:
            continue
        summary[config] = {"n": len(rows)} | {
            f: {"median": round(statistics.median(r[f] for r in rows), 1),
                "min": min(r[f] for r in rows),
                "max": max(r[f] for r in rows)}
            for f in FIELDS}
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--configs", nargs="+", choices=list(CONFIGS),
                        default=["baseline", "stock_mse4", "patched_k5v4", "patched_k5v4_f16out"])
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--target", type=int, default=TARGET)
    parser.add_argument("--cooldown", type=float, default=0.0,
                        help="seconds to idle between runs")
    args = parser.parse_args()

    model, tokenizer = load_model()
    prompt, actual_tokens, _ = build_prompt(args.target, tokenizer)
    tokens = mx.array(tokenizer.encode(prompt))
    output = {
        "environment": environment(),
        "target_tokens": args.target,
        "actual_tokens": actual_tokens,
        "max_new_tokens": MAX_NEW_TOKENS,
        "kv_out_dtype": {c: kv_out_dtype(CONFIGS[c]) for c in args.configs},
        "runs": [],
    }
    first = True
    for i in range(args.rounds):
        for config in args.configs:
            if not first:
                time.sleep(args.cooldown)
            first = False
            r = {"round": i} | run(model, tokens, config)
            output["runs"].append(r)
            output["summary"] = summarize(output["runs"], args.configs)
            args.out.write_text(json.dumps(output, indent=2) + "\n")
            print(json.dumps(r), flush=True)
    print("WROTE", args.out, flush=True)


if __name__ == "__main__":
    main()
