
import json
import time
import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate_step
from mlx_lm.sample_utils import make_sampler
from mlx_lm.models.cache import KVCache
from optiq.core.turbo_kv_cache import make_turbo_kv_caches
import sys
from pathlib import Path

# Insert current directory into path to find build_prompt
sys.path.insert(0, str(Path(__file__).parent.parent / "benchmarks"))
from build_prompt import build_prompt, score_answer

MODEL_ID = "mlx-community/Qwen2.5-3B-Instruct-4bit"
TARGET_LENGTHS = [2000, 4000, 8000] # Start with these
MAX_NEW_TOKENS = 100
TEMPERATURE = 0.0

def run_benchmark(model, tokenizer, target_len, config_name, bits, use_qjl):
    print(f"\n  [{config_name}] Target={target_len}")
    prompt, actual_toks, needle_pos = build_prompt(target_len, tokenizer)
    
    n_layers = len(model.layers)
    head_dim = 128
    
    if bits == 16:
        cache = [KVCache() for _ in range(n_layers)]
    else:
        cache = make_turbo_kv_caches(n_layers, head_dim, bits=bits, use_qjl=use_qjl, seed=42)
    
    sampler = make_sampler(temp=TEMPERATURE)
    prompt_tokens = mx.array(tokenizer.encode(prompt))
    
    generated_tokens = []
    t0 = time.perf_counter()
    
    for token, _ in generate_step(prompt_tokens, model, max_tokens=MAX_NEW_TOKENS, sampler=sampler, prompt_cache=cache):
        generated_tokens.append(token.item() if hasattr(token, "item") else int(token))
        if len(generated_tokens) == 1:
            first_tok_time = time.perf_counter() - t0
    
    total_time = time.perf_counter() - t0
    response = tokenizer.decode(generated_tokens)
    scored = score_answer(response)
    
    tok_per_s = len(generated_tokens) / total_time
    
    print(f"     tok/s: {tok_per_s:.1f}  |  Needle: {scored['score']*100:.0f}%")
    print(f"     Response: {response[:150].strip()}...")
    
    return {
        "target": target_len,
        "actual": actual_toks,
        "tok_per_s": tok_per_s,
        "score": scored['score'],
        "response": response
    }

if __name__ == "__main__":
    print("Loading model...")
    model, tokenizer = load(MODEL_ID)
    
    configs = [
        ("Baseline (FP16)", 16, False),
        ("Stable Turbo (K5-V4)", (5, 4), True),
        ("Turbo (4-bit MSE)", 4, False),
    ]
    
    all_results = {}
    
    for name, bits, qjl in configs:
        all_results[name] = []
        for target in TARGET_LENGTHS:
            res = run_benchmark(model, tokenizer, target, name, bits, qjl)
            all_results[name].append(res)
            
    # Final Table
    print("\n\n" + "="*80)
    print(f"{'Runner':<25} | {'2K Needle':<10} | {'4K Needle':<10} | {'8K Needle':<10}")
    print("-" * 80)
    for name in all_results:
        scores = [f"{r['score']*100:>3.0f}%" for r in all_results[name]]
        print(f"{name:<25} | {' | '.join(scores)}")
    print("="*80)
