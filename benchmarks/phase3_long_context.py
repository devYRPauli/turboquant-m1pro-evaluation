"""
Phase 3 — Long-Context Memory Stress Test
==========================================
Runs needle-in-a-haystack at 2K / 4K / 8K / 16K tokens through:
  A) Ollama qwen2.5:3b  (GGUF Q4_K_M, FP16 KV cache via llama.cpp)
  B) MLX + TurboQuant-MSE bits=4, no QJL  (4× compressed KV cache)
  C) MLX baseline  (standard KVCache, FP16 equivalent)

Memory measurement strategy:
  - System available memory polled every 0.25s in a background thread
  - Peak memory used = baseline_available - min(available_during_run)
  - MLX Metal: mx.reset_peak_memory() + mx.get_peak_memory()
  - Ollama: system-level available memory delta (Metal memory doesn't appear in RSS)

Usage:
    source .venv-mlx/bin/activate
    python3 benchmarks/phase3_long_context.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import mlx.core as mx
import psutil
import requests

sys.path.insert(0, str(Path(__file__).parent))
from build_prompt import build_prompt, score_answer, ANSWER_KEYWORDS

# ── Config ────────────────────────────────────────────────────────────────────
OLLAMA_URL = "http://localhost:11434"
MODEL_OLLAMA = "qwen2.5:3b"
MODEL_MLX = "mlx-community/Qwen2.5-3B-Instruct-4bit"
TARGET_LENGTHS = [2000, 4000, 8000, 16000]
MAX_NEW_TOKENS = 80   # enough to answer the needle question
TEMPERATURE = 0.0
POLL_INTERVAL = 0.25  # seconds between memory polls


# ── Memory monitoring ─────────────────────────────────────────────────────────

class MemoryMonitor:
    """Polls system available memory in a background thread."""

    def __init__(self):
        self._running = False
        self._thread: threading.Thread | None = None
        self.samples: list[int] = []   # available bytes at each sample
        self.baseline_available: int = 0

    def start(self):
        vm = psutil.virtual_memory()
        self.baseline_available = vm.available
        self.samples = [vm.available]
        self._running = True
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def _poll(self):
        while self._running:
            self.samples.append(psutil.virtual_memory().available)
            time.sleep(POLL_INTERVAL)

    def stop(self) -> dict:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        min_avail = min(self.samples) if self.samples else self.baseline_available
        peak_used_mb = (self.baseline_available - min_avail) / (1024 ** 2)
        final_avail_mb = self.samples[-1] / (1024 ** 2) if self.samples else 0
        return {
            "baseline_available_mb": round(self.baseline_available / (1024 ** 2), 1),
            "min_available_mb": round(min_avail / (1024 ** 2), 1),
            "peak_delta_mb": round(peak_used_mb, 1),
            "final_available_mb": round(final_avail_mb, 1),
            "n_samples": len(self.samples),
        }


def vm_stat_pages() -> dict[str, int]:
    """Parse vm_stat and return page counts."""
    result = subprocess.run(["vm_stat"], capture_output=True, text=True)
    pages = {}
    for line in result.stdout.split("\n"):
        if ":" in line:
            key, _, val = line.partition(":")
            try:
                pages[key.strip()] = int(val.strip().rstrip("."))
            except ValueError:
                pass
    return pages


def vm_stat_used_mb() -> float:
    """Approximate used memory in MB from vm_stat (wired + active pages)."""
    pages = vm_stat_pages()
    page_size = 16384  # 16 KB on Apple Silicon
    used_pages = pages.get("Pages wired down", 0) + pages.get("Pages active", 0)
    return used_pages * page_size / (1024 ** 2)


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class RunResult:
    runner: str          # "ollama" | "mlx_baseline" | "mlx_turbo"
    target_tokens: int
    actual_tokens: int
    needle_pos_tokens: int
    elapsed_s: float
    tokens_per_s: float
    peak_sys_delta_mb: float      # system available memory drop (psutil)
    mlx_peak_mb: float            # mx.get_peak_memory() (0 for Ollama)
    needle_score: float           # 0.0 or 0.5 or 1.0
    found_keywords: list[str]
    response_preview: str
    error: str = ""
    kv_theoretical_mb: float = 0.0


# ── Ollama runner ─────────────────────────────────────────────────────────────

def run_ollama(prompt: str, num_ctx: int) -> tuple[str, float, float]:
    """Run prompt through Ollama API. Returns (response, elapsed_s, tok_per_s)."""
    payload = {
        "model": MODEL_OLLAMA,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": TEMPERATURE,
            "num_ctx": num_ctx,
            "num_predict": MAX_NEW_TOKENS,
        },
    }
    t0 = time.perf_counter()
    resp = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=300)
    elapsed = time.perf_counter() - t0
    resp.raise_for_status()
    data = resp.json()
    response_text = data.get("response", "")
    # Ollama reports eval_count (generated tokens) and eval_duration (ns)
    eval_count = data.get("eval_count", 0)
    eval_duration_ns = data.get("eval_duration", 1)
    tok_per_s = eval_count / (eval_duration_ns / 1e9) if eval_duration_ns > 0 else 0.0
    return response_text, elapsed, round(tok_per_s, 1)


# ── MLX runner ────────────────────────────────────────────────────────────────

_mlx_model = None
_mlx_tokenizer = None

def _load_mlx_model():
    global _mlx_model, _mlx_tokenizer
    if _mlx_model is None:
        from mlx_lm import load
        print(f"  [MLX] Loading {MODEL_MLX}...")
        _mlx_model, _mlx_tokenizer = load(MODEL_MLX)
        mx.eval(_mlx_model.parameters())
        print(f"  [MLX] Model loaded.")
    return _mlx_model, _mlx_tokenizer


def run_mlx(prompt: str, use_turbo: bool) -> tuple[str, float, float, float]:
    """Run prompt through MLX. Returns (response, elapsed_s, tok_per_s, peak_mlx_mb)."""
    from mlx_lm.generate import generate_step
    from mlx_lm.sample_utils import make_sampler
    from mlx_lm.models.cache import KVCache
    from optiq.core.turbo_kv_cache import make_turbo_kv_caches

    model, tokenizer = _load_mlx_model()
    n_layers = len(model.layers)
    head_dim = 128

    tokens = tokenizer.encode(prompt)
    prompt_arr = mx.array(tokens)
    sampler = make_sampler(temp=TEMPERATURE)

    if use_turbo:
        # Use our new Gold Standard configuration: Hybrid K5-V4
        # K uses 5 bits (4-bit MSE + 1-bit Orthogonal QJL)
        # V uses 4 bits (4-bit MSE)
        cache = make_turbo_kv_caches(
            n_layers, head_dim, bits=(5, 4), use_qjl=True, seed=42
        )
        label = "TurboQuant-Hybrid K5-V4"
    else:
        cache = [KVCache() for _ in range(n_layers)]
        label = "MLX baseline"

    mx.reset_peak_memory()

    generated = []
    times = []
    t0 = time.perf_counter()
    first_tok = None

    for tok, _ in generate_step(
        prompt_arr, model,
        max_tokens=MAX_NEW_TOKENS,
        sampler=sampler,
        prompt_cache=cache,
    ):
        now = time.perf_counter()
        if first_tok is None:
            first_tok = now - t0
        times.append(now - t0)
        t0 = now
        generated.append(tok.item() if hasattr(tok, "item") else int(tok))

    peak_mlx_mb = mx.get_peak_memory() / (1024 ** 2)
    elapsed = sum(times)
    tok_per_s = len(generated) / elapsed if elapsed > 0 else 0.0
    response = tokenizer.decode(generated)
    return response, round(elapsed, 2), round(tok_per_s, 1), round(peak_mlx_mb, 1)


# ── KV cache theoretical size ─────────────────────────────────────────────────

def kv_theoretical_mb(seq_len: int, bits: float = 16.0) -> float:
    """FP16 KV cache size in MB for Qwen2.5-3B at given sequence length."""
    # 2 (K+V) × 36 layers × 2 KV-heads × 128 head-dim × seq_len × bytes
    return 2 * 36 * 2 * 128 * seq_len * (bits / 8) / (1024 ** 2)


# ── Main ──────────────────────────────────────────────────────────────────────

def run_all():
    print(f"\n{'='*65}")
    print("  Phase 3 — Long-Context Memory Stress Test")
    print(f"  Context lengths: {TARGET_LENGTHS}")
    print(f"  Runners: Ollama (qwen2.5:3b), MLX baseline, MLX TurboQuant-Hybrid K5-V4")
    print(f"{'='*65}\n")

    # Pre-load MLX model once (outside timed loops)
    _load_mlx_model()

    # Load tokenizer for prompt building (reuse MLX tokenizer)
    tokenizer = _mlx_tokenizer

    results: list[RunResult] = []

    for target in TARGET_LENGTHS:
        print(f"\n{'─'*65}")
        print(f"  Context length: ~{target//1000}K tokens")
        print(f"{'─'*65}")

        prompt, actual_toks, needle_pos = build_prompt(target, tokenizer)
        print(f"  Prompt: {actual_toks} tokens  |  Needle at token ~{needle_pos}")
        print(f"  Theoretical KV (FP16): {kv_theoretical_mb(actual_toks):.0f} MB  |"
              f"  TQ-Hybrid K5-V4: {kv_theoretical_mb(actual_toks, bits=4.5):.0f} MB  |"
              f"  Ratio: {kv_theoretical_mb(actual_toks) / kv_theoretical_mb(actual_toks, 4.5):.1f}×")

        # ── A) Ollama ─────────────────────────────────────────────────────────
        print(f"\n  [A] Ollama qwen2.5:3b  (num_ctx={actual_toks + MAX_NEW_TOKENS + 64})")
        mon = MemoryMonitor()
        mon.start()
        vm_before = vm_stat_used_mb()
        try:
            response, elapsed, tok_per_s = run_ollama(
                prompt,
                num_ctx=actual_toks + MAX_NEW_TOKENS + 64,
            )
            error = ""
        except Exception as e:
            response, elapsed, tok_per_s, error = "", 0.0, 0.0, str(e)
        vm_after = vm_stat_used_mb()
        mem = mon.stop()
        scored = score_answer(response)

        print(f"     tok/s: {tok_per_s:.1f}  |  {elapsed:.1f}s  |"
              f"  sys_delta: {mem['peak_delta_mb']:+.0f} MB  |"
              f"  vm_stat Δ: {vm_after - vm_before:+.0f} MB")
        print(f"     Needle: {scored['score']*100:.0f}%  found={scored['found_keywords']}  missing={scored['missing_keywords']}")
        print(f"     Response: {scored['response_preview'][:120]}")

        results.append(RunResult(
            runner="ollama",
            target_tokens=target, actual_tokens=actual_toks, needle_pos_tokens=needle_pos,
            elapsed_s=elapsed, tokens_per_s=tok_per_s,
            peak_sys_delta_mb=mem["peak_delta_mb"],
            mlx_peak_mb=0.0,
            needle_score=scored["score"],
            found_keywords=scored["found_keywords"],
            response_preview=scored["response_preview"],
            error=error,
            kv_theoretical_mb=kv_theoretical_mb(actual_toks),
        ))

        # ── B) MLX baseline ───────────────────────────────────────────────────
        print(f"\n  [B] MLX baseline  (FP16 KV)")
        mon = MemoryMonitor()
        mon.start()
        try:
            response, elapsed, tok_per_s, peak_mlx = run_mlx(prompt, use_turbo=False)
            error = ""
        except Exception as e:
            response, elapsed, tok_per_s, peak_mlx, error = "", 0.0, 0.0, 0.0, str(e)
            print(f"     ERROR: {e}")
        mem = mon.stop()
        scored = score_answer(response)

        print(f"     tok/s: {tok_per_s:.1f}  |  {elapsed:.1f}s  |"
              f"  sys_delta: {mem['peak_delta_mb']:+.0f} MB  |"
              f"  mlx_peak: {peak_mlx:.0f} MB")
        print(f"     Needle: {scored['score']*100:.0f}%  found={scored['found_keywords']}")
        print(f"     Response: {scored['response_preview'][:120]}")

        results.append(RunResult(
            runner="mlx_baseline",
            target_tokens=target, actual_tokens=actual_toks, needle_pos_tokens=needle_pos,
            elapsed_s=elapsed, tokens_per_s=tok_per_s,
            peak_sys_delta_mb=mem["peak_delta_mb"],
            mlx_peak_mb=peak_mlx,
            needle_score=scored["score"],
            found_keywords=scored["found_keywords"],
            response_preview=scored["response_preview"],
            error=error,
            kv_theoretical_mb=kv_theoretical_mb(actual_toks),
        ))

        # ── C) MLX + TurboQuant ───────────────────────────────────────────────
        print(f"\n  [C] MLX + TurboQuant-MSE bits=4  (~4× compressed KV)")
        mon = MemoryMonitor()
        mon.start()
        try:
            response, elapsed, tok_per_s, peak_mlx = run_mlx(prompt, use_turbo=True)
            error = ""
        except Exception as e:
            response, elapsed, tok_per_s, peak_mlx, error = "", 0.0, 0.0, 0.0, str(e)
            print(f"     ERROR: {e}")
        mem = mon.stop()
        scored = score_answer(response)

        print(f"     tok/s: {tok_per_s:.1f}  |  {elapsed:.1f}s  |"
              f"  sys_delta: {mem['peak_delta_mb']:+.0f} MB  |"
              f"  mlx_peak: {peak_mlx:.0f} MB")
        print(f"     Needle: {scored['score']*100:.0f}%  found={scored['found_keywords']}")
        print(f"     Response: {scored['response_preview'][:120]}")

        results.append(RunResult(
            runner="mlx_turbo",
            target_tokens=target, actual_tokens=actual_toks, needle_pos_tokens=needle_pos,
            elapsed_s=elapsed, tokens_per_s=tok_per_s,
            peak_sys_delta_mb=mem["peak_delta_mb"],
            mlx_peak_mb=peak_mlx,
            needle_score=scored["score"],
            found_keywords=scored["found_keywords"],
            response_preview=scored["response_preview"],
            error=error,
            kv_theoretical_mb=kv_theoretical_mb(actual_toks, bits=4.5),
        ))

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n\n{'='*65}")
    print("  PHASE 3 SUMMARY — Memory & Quality vs Context Length")
    print(f"{'='*65}")

    print(f"\n  {'Ctx':>5}  {'Runner':<22}  {'tok/s':>6}  {'SysDeltaMB':>11}  {'MLXPeakMB':>10}  {'KV_theory':>10}  {'Needle':>6}")
    print(f"  {'─'*5}  {'─'*22}  {'─'*6}  {'─'*11}  {'─'*10}  {'─'*10}  {'─'*6}")

    for r in results:
        needle_pct = f"{r.needle_score*100:.0f}%"
        print(
            f"  {r.target_tokens//1000:>4}K  "
            f"{r.runner:<22}  "
            f"{r.tokens_per_s:>6.1f}  "
            f"{r.peak_sys_delta_mb:>+10.0f}M  "
            f"{r.mlx_peak_mb:>10.0f}M  "
            f"{r.kv_theoretical_mb:>10.0f}M  "
            f"{needle_pct:>6}"
        )

    # KV memory comparison table
    print(f"\n  Theoretical KV cache size (Qwen2.5-3B, 36 layers, 2 KV heads, dim=128):")
    print(f"  {'Context':>8}  {'FP16 KV':>10}  {'TQ-4bit KV':>12}  {'Savings':>8}  {'Compression':>12}")
    print(f"  {'─'*8}  {'─'*10}  {'─'*12}  {'─'*8}  {'─'*12}")
    for toks in TARGET_LENGTHS + [32000]:
        fp16 = kv_theoretical_mb(toks)
        tq4 = kv_theoretical_mb(toks, bits=4)
        print(f"  {toks//1000:>7}K  {fp16:>9.0f}M  {tq4:>11.0f}M  {fp16-tq4:>7.0f}M  {fp16/tq4:>11.1f}×")

    # Save JSON results
    out_path = Path(__file__).parent.parent / "benchmarks" / "phase3_results.json"
    with open(out_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\n  Results saved to {out_path}")

    print("\n\nPhase 3 complete.")
    return results


if __name__ == "__main__":
    run_all()
