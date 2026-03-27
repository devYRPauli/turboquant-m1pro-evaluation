"""
Phase 4 Benchmark: llama.cpp TurboQuant fork — Needle-in-a-Haystack
====================================================================
Tests three KV cache types across 4 context lengths using the
TurboQuant llama.cpp fork (feature/turboquant-kv-cache branch).

Cache types:
  q8_0   — 8-bit quantized KV (llama.cpp default for quality)
  turbo3 — TurboQuant 3.25-bit (PolarQuant 2-bit + QJL 1.25-bit, 4.9× vs FP16)
  turbo4 — TurboQuant 4.25-bit (PolarQuant 3-bit + QJL 1.25-bit, 3.8× vs FP16)

Context lengths tested: 2K, 4K, 8K, 16K tokens

Usage:
    python3 benchmarks/phase4_llama_cpp.py

Requires:
  - llama-cpp-turboquant built: ./llama-cpp-turboquant/build/bin/llama-cli
  - GGUF symlink: ./llama-cpp-turboquant/models/qwen2.5-3b-q4_k_m.gguf
  - .venv-mlx active (for Qwen tokenizer to build prompts)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
LLAMA_DIR    = PROJECT_ROOT / "llama-cpp-turboquant"
LLAMA_CLI    = LLAMA_DIR / "build" / "bin" / "llama-cli"
MODEL_GGUF   = LLAMA_DIR / "models" / "qwen2.5-3b-q4_k_m.gguf"

# Add benchmarks dir to path for build_prompt import
sys.path.insert(0, str(PROJECT_ROOT / "benchmarks"))
from build_prompt import build_prompt, score_answer

# ── Config ───────────────────────────────────────────────────────────────────
CONTEXT_LENGTHS = [2000, 4000, 8000, 16000]
CACHE_TYPES     = ["q8_0", "turbo3", "turbo4"]
MAX_NEW_TOKENS  = 80    # answer is short: ~10-15 tokens

# Theoretical KV compression vs FP16 (from paper / turbo_quant.py)
# turbo3 = 3.25 bits (2-bit PolarQuant + 1-bit QJL sign + 0.25-bit overhead)
# turbo4 = 4.25 bits (3-bit PolarQuant + 1-bit QJL sign + 0.25-bit overhead)
CACHE_BITS = {
    "q8_0":   8.0,
    "turbo3": 3.25,
    "turbo4": 4.25,
}

# Qwen2.5-3B architecture constants (same as Phase 3)
N_LAYERS   = 36
N_KV_HEADS = 2
HEAD_DIM   = 128


def kv_theoretical_mb(seq_len: int, bits: float) -> float:
    """Theoretical KV cache size in MB for Qwen2.5-3B."""
    return 2 * N_LAYERS * N_KV_HEADS * HEAD_DIM * seq_len * (bits / 8) / (1024 ** 2)


def run_llama_cli(
    prompt: str,
    cache_type_k: str,
    cache_type_v: str,
    num_ctx: int,
    max_new_tokens: int,
) -> dict:
    """Run llama-cli with the given cache type and return timing + output."""

    # Write prompt to temp file (avoids shell quoting issues with long prompts)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(prompt)
        prompt_file = f.name

    cmd = [
        str(LLAMA_CLI),
        "-m", str(MODEL_GGUF),
        "--file", prompt_file,
        "-n", str(max_new_tokens),
        "--ctx-size", str(num_ctx + max_new_tokens + 64),  # context window
        "--cache-type-k", cache_type_k,
        "--cache-type-v", cache_type_v,
        "--temp", "0.0",            # greedy
        "--no-display-prompt",      # only show generated tokens
        "--log-disable",            # suppress llama.cpp log noise
        "-t", "8",                  # CPU threads
    ]

    t0 = time.perf_counter()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )
        elapsed = time.perf_counter() - t0
        output = result.stdout.strip()
        stderr = result.stderr.strip()
        returncode = result.returncode
    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - t0
        output = ""
        stderr = "TIMEOUT"
        returncode = -1
    finally:
        os.unlink(prompt_file)

    # Extract tok/s from stderr (llama.cpp prints "eval time = ... ms / N tokens ... N.N t/s")
    tok_per_s = None
    for line in stderr.splitlines():
        if "eval time" in line and "t/s" in line:
            try:
                tok_per_s = float(line.split("t/s")[0].split()[-1])
            except (ValueError, IndexError):
                pass

    return {
        "output": output,
        "stderr_tail": stderr[-800:] if len(stderr) > 800 else stderr,
        "elapsed_s": round(elapsed, 2),
        "tok_per_s": tok_per_s,
        "returncode": returncode,
    }


def main():
    # Validate binaries
    if not LLAMA_CLI.exists():
        print(f"ERROR: llama-cli not found at {LLAMA_CLI}")
        print("  Run: cmake --build llama-cpp-turboquant/build -j8")
        sys.exit(1)
    if not MODEL_GGUF.exists():
        print(f"ERROR: model GGUF not found at {MODEL_GGUF}")
        sys.exit(1)

    print(f"\n{'='*65}")
    print("  TurboQuant Phase 4 — llama.cpp Metal Build, Needle-in-Haystack")
    print(f"  Binary : {LLAMA_CLI}")
    print(f"  Model  : {MODEL_GGUF.name}  ({MODEL_GGUF.stat().st_size / 1e9:.2f} GB)")
    print(f"  Caches : {CACHE_TYPES}")
    print(f"  Lengths: {CONTEXT_LENGTHS} tokens")
    print(f"{'='*65}\n")

    # Load tokenizer only (no model weights needed — just for token counting)
    print("Loading Qwen tokenizer for prompt generation...")
    from transformers import AutoTokenizer

    class _TokenizerWrapper:
        """Thin wrapper so build_prompt's tokenizer.encode() works."""
        def __init__(self, hf_tok):
            self._tok = hf_tok
        def encode(self, text):
            return self._tok.encode(text, add_special_tokens=False)

    # Use local cache path to avoid any network calls
    _tok_cache = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1"
    hf_tok = AutoTokenizer.from_pretrained(
        str(_tok_cache),
        local_files_only=True,
    )
    tokenizer = _TokenizerWrapper(hf_tok)
    print("Tokenizer ready.\n")

    all_results = []

    for ctx_len in CONTEXT_LENGTHS:
        prompt, actual_tokens, needle_pos = build_prompt(ctx_len, tokenizer)
        needle_pct = round(100 * needle_pos / actual_tokens, 1)

        print(f"\n{'─'*65}")
        print(f"  Context: ~{ctx_len}  (actual {actual_tokens} tokens, needle at {needle_pct}%)")
        print(f"{'─'*65}")

        for cache_type in CACHE_TYPES:
            kv_mb = kv_theoretical_mb(actual_tokens, CACHE_BITS[cache_type])
            kv_mb_fp16 = kv_theoretical_mb(actual_tokens, 16.0)
            ratio = kv_mb_fp16 / kv_mb

            label = f"{cache_type:<8} @ {ctx_len:>5}tok"
            print(f"\n  [{label}]  KV ~{kv_mb:.1f} MB  ({ratio:.1f}× vs FP16)")

            run = run_llama_cli(
                prompt=prompt,
                cache_type_k=cache_type,
                cache_type_v=cache_type,
                num_ctx=actual_tokens,
                max_new_tokens=MAX_NEW_TOKENS,
            )

            score = score_answer(run["output"])
            status = "PASS" if score["score"] == 1.0 else (
                "PARTIAL" if score["score"] > 0 else "FAIL"
            )

            tok_s_str = f"{run['tok_per_s']:.1f} t/s" if run["tok_per_s"] else "n/a"
            print(f"  Status : {status}  |  {tok_s_str}  |  {run['elapsed_s']}s wall")
            print(f"  Found  : {score['found_keywords'] or '(none)'}")
            print(f"  Output : {run['output'][:200]}")

            all_results.append({
                "ctx_len": ctx_len,
                "actual_tokens": actual_tokens,
                "needle_pct": needle_pct,
                "cache_type": cache_type,
                "kv_bits": CACHE_BITS[cache_type],
                "kv_theoretical_mb": round(kv_mb, 2),
                "kv_ratio_vs_fp16": round(ratio, 2),
                "score": score["score"],
                "found_keywords": score["found_keywords"],
                "missing_keywords": score["missing_keywords"],
                "tok_per_s": run["tok_per_s"],
                "elapsed_s": run["elapsed_s"],
                "returncode": run["returncode"],
                "response_preview": run["output"][:300],
            })

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n\n{'='*65}")
    print("  PHASE 4 SUMMARY")
    print(f"{'='*65}")
    print(f"  {'Config':<22} {'Len':>6} {'Score':>6} {'KV MB':>7} {'Ratio':>6} {'tok/s':>7}")
    print(f"  {'─'*22} {'─'*6} {'─'*6} {'─'*7} {'─'*6} {'─'*7}")
    for r in all_results:
        tps = f"{r['tok_per_s']:.1f}" if r["tok_per_s"] else " n/a"
        print(
            f"  {r['cache_type']:<22} {r['ctx_len']:>6} "
            f"  {r['score']:.2f}  {r['kv_theoretical_mb']:>7.1f}  "
            f"{r['kv_ratio_vs_fp16']:>5.1f}×  {tps:>7}"
        )

    # Save JSON
    out_path = PROJECT_ROOT / "benchmarks" / "phase4_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to {out_path}")
    print("\nDone.")

    return all_results


if __name__ == "__main__":
    main()
