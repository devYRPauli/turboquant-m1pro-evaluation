"""Needle-in-a-haystack runs for the stock and patched mlx-optiq KV caches.

Uses the phase3 harness unchanged: build_prompt() builds the prompt (needle at
the midpoint, no chat template) and score_answer() gives the case-insensitive
score that phase3_results.json reports. Each run also records an exact-case
score, the full response text, and the generated token ids.

Configs:
  baseline      mlx-lm KVCache (FP16), the phase3 mlx_baseline runner
  stock_mse4    stock optiq make_turbo_kv_caches(bits=4, use_qjl=False,
                seed=42), the TurboQuant-MSE 4-bit runner of the round 1
                phase3 table (reports/round1-experiment-log.md)
  stock_k5v4    keys: stock TurboQuantProd 5 bits (Gaussian QJL,
                sqrt(pi/2)/d scale, no damping); values: TurboQuantMSE 4 bits
  patched_k5v4  tq_patched.make_turbo_kv_caches(bits=(5, 4), use_qjl=True,
                seed=42), the phase3 mlx_turbo runner

Usage:
  python benchmarks/needle_repro.py --out logs/needle-repro-2026-10-05.json
"""

import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import date
from importlib.metadata import version
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate_step
from mlx_lm.models.cache import KVCache
from mlx_lm.sample_utils import make_sampler
from optiq.core.turbo_kv_cache import make_turbo_kv_caches as make_stock_caches

BENCH = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH))
from build_prompt import ANSWER_KEYWORDS, build_prompt, score_answer  # noqa: E402
import tq_patched  # noqa: E402

MODEL_ID = "mlx-community/Qwen2.5-3B-Instruct-4bit"
MODEL_REVISION = "4f83f8f146fdf28b512a06562b671d7af4fab457"
HEAD_DIM = 128
MAX_NEW_TOKENS = 80
TARGETS = [2000, 4000, 8000, 16000]
MB = 1024 ** 2

CONFIGS = {
    "baseline": lambda n: [KVCache() for _ in range(n)],
    "stock_mse4": lambda n: make_stock_caches(n, HEAD_DIM, bits=4, use_qjl=False, seed=42),
    "stock_k5v4": lambda n: tq_patched.make_turbo_kv_caches(
        n, HEAD_DIM, bits=(5, 4), use_qjl=True, seed=42, patched=False),
    "patched_k5v4": lambda n: tq_patched.make_turbo_kv_caches(
        n, HEAD_DIM, bits=(5, 4), use_qjl=True, seed=42),
}

# phase3_results.json runner that each config re-runs
PHASE3_RUNNER = {"baseline": "mlx_baseline", "patched_k5v4": "mlx_turbo"}


def environment():
    def sysctl(key):
        return subprocess.run(["sysctl", "-n", key], capture_output=True, text=True).stdout.strip()

    packages = ["mlx", "mlx-metal", "mlx-lm", "mlx-optiq", "transformers",
                "tokenizers", "numpy", "scipy"]
    return {
        "date": date.today().isoformat(),
        "chip": sysctl("machdep.cpu.brand_string"),
        "memory_gb": int(sysctl("hw.memsize")) // 1024 ** 3,
        "macos": platform.mac_ver()[0],
        "python": platform.python_version(),
        "packages": {p: version(p) for p in packages},
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }


def load_model():
    model, tokenizer = load(MODEL_ID, revision=MODEL_REVISION)
    mx.eval(model.parameters())
    return model, tokenizer


def cache_bytes(caches):
    """(allocated, filled) bytes of the arrays held by the layer caches.

    allocated counts every array the cache objects hold, including the unused
    tail of the last 256-step block; filled counts positions up to the offset.
    """
    allocated = filled = 0
    for c in caches:
        if isinstance(c, KVCache):
            arrays = [c.keys, c.values]
        else:
            arrays = [*c._k_store.values(), *c._v_store.values()]
        for a in arrays:
            allocated += a.nbytes
            filled += a.nbytes * c.offset // a.shape[2]
    return allocated, filled


def run(model, tokenizer, config, target, phase3):
    prompt, actual_tokens, needle_pos = build_prompt(target, tokenizer)
    tokens = mx.array(tokenizer.encode(prompt))
    cache = CONFIGS[config](len(model.layers))
    sampler = make_sampler(temp=0.0)
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    generated = [int(t) for t, _ in generate_step(
        tokens, model, max_tokens=MAX_NEW_TOKENS, sampler=sampler, prompt_cache=cache)]
    seconds = time.perf_counter() - t0
    peak = mx.get_peak_memory()
    response = tokenizer.decode(generated)
    scored = score_answer(response)
    exact = [kw for kw in ANSWER_KEYWORDS if kw in response]
    allocated, filled = cache_bytes(cache)
    result = {
        "config": config,
        "target_tokens": target,
        "actual_tokens": actual_tokens,
        "needle_pos_tokens": needle_pos,
        "score_case_insensitive": scored["score"],
        "found_case_insensitive": scored["found_keywords"],
        "score_exact_case": len(exact) / len(ANSWER_KEYWORDS),
        "found_exact_case": exact,
        "response": response,
        "generated_token_ids": generated,
        "seconds": round(seconds, 2),
        "tokens_per_s": round(len(generated) / seconds, 2),
        "mlx_peak_mb": round(peak / MB, 1),
        "cache_allocated_mb": round(allocated / MB, 1),
        "cache_filled_mb": round(filled / MB, 1),
    }
    ref = phase3.get((PHASE3_RUNNER.get(config), target))
    if ref is not None:
        result["phase3_score"] = ref["needle_score"]
        result["phase3_preview_match"] = scored["response_preview"] == ref["response_preview"]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    parser.add_argument("--targets", nargs="+", type=int, default=TARGETS)
    parser.add_argument("--cooldown", type=float, default=0.0,
                        help="seconds to idle between runs")
    args = parser.parse_args()

    phase3 = {(r["runner"], r["target_tokens"]): r
              for r in json.loads((BENCH / "phase3_results.json").read_text())}
    output = {"environment": environment(), "runs": []}
    model, tokenizer = load_model()
    first = True
    for target in args.targets:
        for config in args.configs:
            if not first:
                time.sleep(args.cooldown)
            first = False
            r = run(model, tokenizer, config, target, phase3)
            output["runs"].append(r)
            args.out.write_text(json.dumps(output, indent=2) + "\n")
            print(json.dumps({k: r[k] for k in (
                "config", "target_tokens", "score_case_insensitive", "score_exact_case",
                "mlx_peak_mb", "seconds")} | {"phase3_preview_match": r.get("phase3_preview_match")}),
                flush=True)
    print("WROTE", args.out, flush=True)


if __name__ == "__main__":
    main()
