# CLAUDE.md, TurboQuant Round 2 (Fresh Start)

> **Owner**: Yash
> **Hardware**: MacBook Pro M1 Pro, 16GB Unified Memory, 512GB SSD
> **Python**: 3.12
> **Date**: March 27, 2026
> **Previous Work**: ~/Desktop/TurboQuant/ (Phases 1 through 4, keep for reference)

---

## What Changed Since Round 1

The ecosystem moved fast in 48 hours. Three critical developments:

### 1. Aaryan Kapoor's CPU-only llama.cpp fork (THE NEW PATH)

Branch: `https://github.com/Aaryan-Kapoor/llama.cpp/tree/turboquant-tq3_0`

This is a completely different approach from TheTom's fork that crashed on our M1 Pro:
* **CPU-only with vec_dot support** (enables flash attention without Metal kernel dependencies)
* **Block size 32** (not 128), 14 bytes per block = 3.5 bits per value
* **Uses Walsh-Hadamard Transform** (fast O(d log d) rotation, not dense O(d²) QR)
* **Deterministic sign pattern** from golden ratio hash (reproducible, no random seed issues)
* **Reported result: "Output identical to f16 baseline on 35B model at temperature 0"**
* **No QJL**, both developers (Aaryan and TheTom) independently found MSE-only with all bits to Lloyd-Max centroids works better in practice than the paper's two-stage approach

This should work on M1 Pro without any Metal kernel issues.

### 2. TheTom's turboquant_plus major update

* Now 511 tests (up from 144)
* Zero speed penalty vs q8_0 on Apple Silicon (2747 vs 2694 tok/s prefill)
* 4.6x compression confirmed
* Perplexity matches baseline (PPL 6.20 vs 6.19)
* Fixed the context-scaling issues we hit. Full investigation documented at:
  `https://github.com/TheTom/turboquant_plus/blob/main/docs/context-scaling-deep-dive.md`

### 3. Multiple community validations

* TheTom confirmed PPL matches baseline
* Aaryan confirmed output identical to f16 at temp 0
* Both independently dropped QJL (paper's Stage 2) in favor of MSE-only
* spiritbuun's CUDA fork hit 98.8% of q8_0 prefill speed
* Norm correction technique emerged: store original_norm / ||reconstruction|| instead of raw norm

---

## Round 2 Plan

### Phase 5: Aaryan Kapoor's CPU-only TQ3_0 Fork

This is the primary target. It should work on M1 Pro without the Metal kernel crashes we hit in Phase 4.

```bash
# 1. Clean up from Round 1 (keep old work for reference)
cd ~/Desktop/TurboQuant
mkdir -p round1-archive
# Don't delete anything, just organize

# 2. Clone Aaryan's fork
git clone https://github.com/Aaryan-Kapoor/llama.cpp.git aaryan-llama-cpp
cd aaryan-llama-cpp
git checkout turboquant-tq3_0

# 3. Build with Metal for model weights (KV cache is CPU-computed via vec_dot)
cmake -B build \
  -DGGML_METAL=ON \
  -DGGML_METAL_EMBED_LIBRARY=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(sysctl -n hw.logicalcpu)

# 4. Symlink the Ollama GGUF model (same as before)
mkdir -p models
# Find the blob: cat ~/.ollama/models/manifests/registry.ollama.ai/library/qwen2.5/3b
# Then symlink it:
# ln -sf ~/.ollama/models/blobs/sha256-XXXXX models/qwen2.5-3b-q4_k_m.gguf

# 5. Quick sanity test (should NOT crash, should NOT hang on Metal JIT)
./build/bin/llama-cli \
  -m models/qwen2.5-3b-q4_k_m.gguf \
  -ngl 99 -c 512 -fa on \
  --cache-type-k tq3_0 --cache-type-v tq3_0 \
  -n 20 -p "What is the capital of France?" \
  --no-display-prompt < /dev/null 2>&1

# NOTE: The cache-type flag might be "tq3_0" or "tq3" or "turbo3"
# Check: ./build/bin/llama-cli --help 2>&1 | grep -i "tq3\|turbo"
```

### Phase 5 Success Criteria

* [ ] Builds without errors on M1 Pro
* [ ] Does NOT crash on KV cache init (no GGML_ASSERT failures)
* [ ] Does NOT hang on Metal JIT compilation
* [ ] Produces coherent output with tq3_0 cache type
* [ ] Speed is reasonable (>10 tok/s generation)

### Phase 6: Needle in a Haystack (if Phase 5 passes)

Run the same test from Phase 3 but with Aaryan's fork:

```bash
# Embed the FROSTBLOCK-7 / VcMYB4 needle in a 2K token prompt
# Run with tq3_0 cache and q8_0 baseline
# Compare: does TQ3_0 find the needle?

# IMPORTANT: Run ONE test at a time. No background processes.
# Use < /dev/null to prevent interactive mode hanging.
# Kill any stale llama-cli before starting: pkill -9 -f llama-cli
```

### Phase 7: Updated turboquant_plus (if time permits)

The Python prototype has been massively updated (511 tests now). Worth re-running to see the improvements.

```bash
cd ~/Desktop/TurboQuant/turboquant_plus
git pull origin main
source .venv/bin/activate
pip install -e ".[dev]"
python3 -m pytest tests/ -v
python3 benchmarks/demo.py
```

---

## Lessons from Round 1 (DO NOT REPEAT)

1. **NEVER run llama-cli in the background.** Always foreground with < /dev/null
2. **NEVER run multiple llama-cli processes.** One at a time. Kill before starting next.
3. **Always `pkill -9 -f llama-cli` before starting a new test**
4. **TheTom's fork crashes on M1 Pro** due to Metal kernel requirements (turbo4 non-vec kernel missing, turbo3 non-vec kernel causes 30+ min JIT compilation). We found and fixed one bug (context sizing) but hit deeper Metal compatibility issues.
5. **mlx-optiq v0.0.1 QJL is broken.** MSE-only at 4-bit was the only working config, but quality degraded at long context (0% needle retrieval at 2K-16K).
6. **Metal JIT cache invalidation:** Every rebuild of llama.cpp invalidates the shader cache, causing 20-30 min recompilation. Avoid unnecessary rebuilds.

---

## Key Findings from Round 1 (for reference)

| Phase | What | Result |
|---|---|---|
| 1 | Algorithm math (turboquant_plus) | 144/144 tests pass, kurtosis 51 to near 0 |
| 2 | mlx-optiq inference | 4-bit MSE works (35 tok/s), QJL broken, 3-bit degenerate |
| 3 | Long context needle test | 0% retrieval with TurboQuant at all context lengths |
| 4 | TheTom's llama.cpp fork | Found GGML context sizing bug (one-line fix). turbo3 runs at 29.7 tok/s but garbled. turbo4 crashes (missing Metal kernel). Metal JIT takes 30+ min after rebuild. |

---

## Why Aaryan's Fork Should Work

The critical differences from TheTom's fork:

| Issue We Hit | TheTom's Fork | Aaryan's Fork |
|---|---|---|
| Metal kernel crash | Requires M5+ for turbo4 non-vec | CPU vec_dot, no custom Metal kernels needed |
| Metal JIT hang (30+ min) | Compiles all flash attn pipelines including turbo | Standard flash attn uses existing vec_dot path |
| GGML context sizing | Extra rotation tensors not allocated | Block size 32 approach, different tensor layout |
| QJL broken | Not used (MSE-only in C port) | Not used (MSE-only) |
| Output quality | Garbled at 2-bit (turbo3) | "Identical to f16" at 3.5 bpw on 35B model |

The key insight both developers converged on: **block size 32 with 3.5 bits per value works better than block size 128 with 2-3 bits.** More bits per value = better quality. Smaller block size = better flash attention parallelism.

---

## Files and Locations

```
~/Desktop/TurboQuant/
├── CLAUDE.md                          # THIS FILE (Round 2 guide)
├── notebooks/experiment-log.md        # All findings from Round 1
├── turboquant_plus/                   # Phase 1 Python prototype
├── llama-cpp-turboquant/              # Phase 4 TheTom's fork (crashes on M1)
├── .venv-mlx/                         # MLX venv from Phase 2-3
├── aaryan-llama-cpp/                  # NEW: Phase 5 target
└── benchmarks/                        # Benchmark scripts
```

---

## Links

* Aaryan's fork: https://github.com/Aaryan-Kapoor/llama.cpp/tree/turboquant-tq3_0
* TheTom's updated repo: https://github.com/TheTom/turboquant_plus
* llama.cpp Discussion #20969: https://github.com/ggml-org/llama.cpp/discussions/20969
* Context scaling deep dive: https://github.com/TheTom/turboquant_plus/blob/main/docs/context-scaling-deep-dive.md
* Paper: https://arxiv.org/abs/2504.19874
