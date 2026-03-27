# TurboQuant Implementation Guide, M1 Pro MacBook Pro (16GB / 512GB SSD)

> **Author**: Yash (AI Agents Architect, UF Blueberry Breeding & Genomics Lab)
> **Date**: March 26, 2026
> **Hardware**: MacBook Pro M1 Pro, 16GB Unified Memory, 512GB SSD
> **Current Local AI Stack**: Ollama + Osaurus
> **Goal**: Implement and evaluate Google's TurboQuant KV cache compression for local LLM inference

---

## Table of Contents

1. [What is TurboQuant & Why It Matters for Your Setup](#1-what-is-turboquant--why-it-matters-for-your-setup)
2. [Landscape of Available Implementations](#2-landscape-of-available-implementations)
3. [Implementation Path A: Python Prototype (turboquant_plus)](#3-implementation-path-a-python-prototype-turboquant_plus)
4. [Implementation Path B: MLX Native (mlx-optiq)](#4-implementation-path-b-mlx-native-mlx-optiq)
5. [Implementation Path C: llama.cpp Fork with TurboQuant KV Cache](#5-implementation-path-c-llamacpp-fork-with-turboquant-kv-cache)
6. [Implementation Path D: Build Your Own from the Paper (Claude Code Task)](#6-implementation-path-d-build-your-own-from-the-paper-claude-code-task)
7. [Integration with Ollama (Future)](#7-integration-with-ollama-future)
8. [Benchmarking & Evaluation Plan](#8-benchmarking--evaluation-plan)
9. [Memory Budget Analysis for M1 Pro 16GB](#9-memory-budget-analysis-for-m1-pro-16gb)
10. [Documentation & Lab Relevance](#10-documentation--lab-relevance)
11. [Key References](#11-key-references)

---

## 1. What is TurboQuant & Why It Matters for Your Setup

### The Problem: KV Cache Is Your Memory Bottleneck

When you run a 7B model via Ollama on your M1 Pro 16GB, the model weights (say Q4_K_M) take ~4GB. But the **KV cache**, the running memory of your conversation, grows linearly with context length. At 8K context on a 32-layer model, that's easily 2-4GB in FP16. At 32K context, it can exceed your remaining memory entirely.

**TurboQuant compresses the KV cache, not the model weights.** It's complementary to GGUF quantization.

### How It Works (Two-Stage Pipeline)

```
Input: KV cache vector x ∈ R^d (one attention head)
    │
    ├── Extract norm: γ = ||x||, x̂ = x/γ
    │
    ├── Stage 1: PolarQuant (b-1 bits)
    │   - Random rotation Π → coordinates ~ N(0, 1/d)
    │   - Optimal scalar quantization per coordinate
    │   - No per-block normalization constants needed
    │
    ├── Stage 2: QJL (1 bit)
    │   - sign(S · residual) → unbiased inner product correction
    │   - Eliminates quantization bias
    │
    └── Output: CompressedVector(indices, signs, norms)
        Total: b bits per coordinate (e.g., 3.5 bits = 3.8× compression)
```

### What This Means for 16GB M1 Pro

| Scenario | Without TurboQuant | With TurboQuant (3.5-bit) |
|---|---|---|
| 7B Q4 model weights | ~4 GB | ~4 GB (unchanged) |
| KV cache at 8K context | ~2 GB (FP16) | ~0.5 GB |
| KV cache at 32K context | ~8 GB (FP16) | ~2 GB |
| Remaining for OS + apps | 2-6 GB | 7.5-9.5 GB |
| **Feasible context length** | **~8-16K** | **~32-64K** |

This is the difference between "it works but swaps to disk" and "it runs smoothly with room to spare."

### Key Claims from the Paper (arXiv 2504.19874)

* 3.5 bits per channel → **zero accuracy loss** (quality-neutral)
* 2.5 bits per channel → marginal degradation only
* 6× KV memory reduction minimum
* No training, no fine-tuning, no calibration data required ("data-oblivious")
* Works as a drop-in on any transformer model

---

## 2. Landscape of Available Implementations

> **Important**: Google has NOT released official code. Everything below is community-built from the paper's math.

| Implementation | Language | Apple Silicon | Status | Best For |
|---|---|---|---|---|
| **turboquant_plus** (TheTom) | Python + C (llama.cpp fork) |  Metal | 141 tests passing, v1 complete | Understanding the algorithm + llama.cpp integration |
| **mlx-optiq** (PyPI) | Python (MLX) |  Native MLX | Published on PyPI | Easiest Apple Silicon path, clean API |
| **turboquant-pytorch** (tonbistudio) | PyTorch |  CPU only on Mac | Validated on RTX 3060 | If you want PyTorch-native code to study |
| **mudler/llama.cpp feat/turbo-quant** | C/C++ |  Metal | Experimental branch, "builds/quantizes correctly" | Closest to eventual Ollama integration |
| **Prince_Canuma's MLX impl** | MLX |  Native | Tweeted results, code location TBD | Reference for MLX approach |
| **Ollama native** | Go |  Not yet | Issue #15051 filed, no implementation | Wait for upstream |

### Recommendation for Your Setup

**Start with Path A (turboquant_plus Python prototype)** to understand the algorithm and validate on your hardware. Then move to **Path B (mlx-optiq)** for actual inference with real models. Path C (llama.cpp fork) is for when you want to serve models with TurboQuant KV cache via an OpenAI-compatible API.

---

## 3. Implementation Path A: Python Prototype (turboquant_plus)

### Purpose

Understand the algorithm, run compression benchmarks, validate on real model KV tensors. This is the "study and verify" step.

### Steps

```bash
# 1. Create a project directory
mkdir -p ~/Projects/turboquant-experiment
cd ~/Projects/turboquant-experiment

# 2. Clone the repo
git clone https://github.com/TheTom/turboquant_plus.git
cd turboquant_plus

# 3. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 4. Install in dev mode
pip install -e ".[dev]"

# 5. Run tests (should see "141 passed")
python3 -m pytest tests/ -v

# 6. Run the quick compression demo (no model download needed)
python3 benchmarks/demo.py

# 7. (Optional) Validate on real Qwen3-1.7B KV tensors (~4GB download)
pip install transformers torch accelerate
python3 benchmarks/validate_real_model.py
```

### What to Look For

* **Cosine similarity** at different bit widths (3-bit, 3.5-bit, 4-bit)
* **Kurtosis** before/after rotation (should go from high → ~3.0, proving Gaussianization works)
* **Compression ratios** reported

### Disk Space Note (512GB SSD)

The Qwen3-1.7B download is ~4GB. If disk is tight, skip the real model validation and just run the synthetic demo.

---

## 4. Implementation Path B: MLX Native (mlx-optiq)

### Purpose

Run actual inference with TurboQuant KV cache compression on your M1 Pro using Apple's native MLX framework. This is the "actually use it" step.

### Prerequisites

```bash
# MLX requires macOS 13.5+ and Apple Silicon
# Verify you're on Apple Silicon
uname -m  # should show "arm64"
```

### Steps

```bash
# 1. Create a dedicated environment
cd ~/Projects/turboquant-experiment
python3 -m venv .venv-mlx
source .venv-mlx/bin/activate

# 2. Install mlx-optiq
pip install mlx-optiq

# 3. Install mlx-lm for model loading
pip install mlx-lm

# 4. Basic usage, TurboQuant KV cache with a small model
python3 << 'EOF'
from mlx_lm import load
from optiq.core.turbo_kv_cache import TurboQuantKVCache

# Load a small MLX-compatible model
model, tokenizer = load("mlx-community/Qwen2.5-1.5B-Instruct-4bit")

# Replace self-attention KV caches with TurboQuant
cache = model.make_cache()
for i, layer in enumerate(model.layers):
    if hasattr(layer, "self_attn"):
        cache[i] = TurboQuantKVCache(
            head_dim=layer.self_attn.head_dim,
            bits=4,      # try 4, 3.5, 3, 2.5
            seed=42 + i
        )

# Now run inference, TurboQuant KV is transparent to mlx-lm
import mlx.core as mx
prompt = "Explain what blueberry genomics research involves in 3 sentences."
inputs = tokenizer(prompt, return_tensors="np")
input_ids = mx.array(inputs["input_ids"])

logits = model(input_ids, cache=cache)
print("Inference with TurboQuant KV cache completed successfully!")
print(f"Output logits shape: {logits.shape}")
EOF
```

### Recommended Models for 16GB M1 Pro

| Model | Size on Disk | RAM Usage (approx) | Notes |
|---|---|---|---|
| Qwen2.5-1.5B-Instruct-4bit | ~1 GB | ~2 GB | Good for testing |
| Qwen2.5-3B-Instruct-4bit | ~2 GB | ~4 GB | Sweet spot for your RAM |
| Gemma-2-2B-4bit | ~1.5 GB | ~3 GB | Paper used Gemma for benchmarks |
| Mistral-7B-Instruct-v0.3-4bit | ~4 GB | ~6 GB | Paper used Mistral for benchmarks |

### Benchmark Script

```bash
# Create a benchmark script
cat > ~/Projects/turboquant-experiment/benchmark_turboquant_mlx.py << 'PYEOF'
"""
TurboQuant MLX Benchmark, M1 Pro 16GB
Tests KV cache compression at different bit widths.
"""
import time
import mlx.core as mx
from mlx_lm import load, generate

MODEL_ID = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
BIT_WIDTHS = [4, 3.5, 3, 2.5]
PROMPT = "What is the role of anthocyanins in blueberry fruit development? Provide a detailed explanation."

def benchmark_bitwidth(model, tokenizer, bits):
    from optiq.core.turbo_kv_cache import TurboQuantKVCache

    cache = model.make_cache()
    for i, layer in enumerate(model.layers):
        if hasattr(layer, "self_attn"):
            cache[i] = TurboQuantKVCache(
                head_dim=layer.self_attn.head_dim,
                bits=bits,
                seed=42 + i
            )

    start = time.perf_counter()
    response = generate(
        model, tokenizer,
        prompt=PROMPT,
        max_tokens=200,
        verbose=False
    )
    elapsed = time.perf_counter() - start

    return {
        "bits": bits,
        "time_s": round(elapsed, 2),
        "tokens": len(tokenizer.encode(response)),
        "tok_per_s": round(len(tokenizer.encode(response)) / elapsed, 1),
        "response_preview": response[:100] + "..."
    }

if __name__ == "__main__":
    print(f"Loading model: {MODEL_ID}")
    model, tokenizer = load(MODEL_ID)

    print(f"\n{'='*60}")
    print(f"TurboQuant KV Cache Benchmark, M1 Pro 16GB")
    print(f"Model: {MODEL_ID}")
    print(f"Prompt: {PROMPT[:60]}...")
    print(f"{'='*60}\n")

    # Baseline without TurboQuant
    print("Running baseline (no TurboQuant)...")
    start = time.perf_counter()
    baseline_response = generate(model, tokenizer, prompt=PROMPT, max_tokens=200, verbose=False)
    baseline_time = time.perf_counter() - start
    baseline_tokens = len(tokenizer.encode(baseline_response))
    print(f"  Baseline: {round(baseline_tokens/baseline_time, 1)} tok/s in {round(baseline_time, 2)}s")
    print(f"  Response: {baseline_response[:100]}...\n")

    # TurboQuant at each bit width
    for bits in BIT_WIDTHS:
        print(f"Running TurboQuant at {bits}-bit...")
        try:
            result = benchmark_bitwidth(model, tokenizer, bits)
            print(f"  {bits}-bit: {result['tok_per_s']} tok/s in {result['time_s']}s")
            print(f"  Response: {result['response_preview']}\n")
        except Exception as e:
            print(f"  {bits}-bit FAILED: {e}\n")

    print("Benchmark complete!")
PYEOF

# Run it
python3 ~/Projects/turboquant-experiment/benchmark_turboquant_mlx.py
```

---

## 5. Implementation Path C: llama.cpp Fork with TurboQuant KV Cache

### Purpose

Build a llama.cpp binary with TurboQuant KV cache support. This gives you an OpenAI-compatible server that could eventually plug into Ollama or tools like Osaurus.

### Steps

```bash
# 1. Clone the llama.cpp fork with TurboQuant
cd ~/Projects/turboquant-experiment
git clone https://github.com/TheTom/llama-cpp-turboquant.git
cd llama-cpp-turboquant
git checkout feature/turboquant-kv-cache

# 2. Build with Metal (Apple Silicon)
cmake -B build \
  -DGGML_METAL=ON \
  -DGGML_METAL_EMBED_LIBRARY=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(sysctl -n hw.ncpu)

# 3. Verify turbo types are available
./build/bin/llama-server --help | grep turbo
# Should show: turbo3, turbo4

# 4. Download a test model (if you don't have one already)
# You can use any GGUF model you already have from Ollama
# Ollama stores models in ~/.ollama/models/blobs/
# Or download one directly:
# huggingface-cli download TheBloke/Mistral-7B-Instruct-v0.2-GGUF \
#   mistral-7b-instruct-v0.2.Q4_K_M.gguf --local-dir models/

# 5. Run with TurboQuant KV cache
./build/bin/llama-cli \
  -m /path/to/your/model.gguf \
  -ngl 99 -c 4096 -fa on \
  --cache-type-k turbo3 --cache-type-v turbo3 \
  -n 100 -p "Hello, I am testing TurboQuant KV cache compression." --jinja

# 6. Or run as a server (OpenAI-compatible API)
./build/bin/llama-server \
  -m /path/to/your/model.gguf \
  --alias "model-turbo" \
  --jinja -ngl 99 -c 8192 -fa on \
  --cache-type-k turbo3 --cache-type-v turbo3 \
  -np 1 --metrics --host 127.0.0.1 --port 8080
```

### Cache Type Options

| Flag | Bits/val | Compression vs FP16 | Description |
|---|---|---|---|
| `turbo3` | 3.25 | **4.9×** | 2-bit PolarQuant + 1-bit QJL. Max compression. |
| `turbo4` | 4.25 | **3.8×** | 3-bit PolarQuant + 1-bit QJL. Better quality. |
| `q8_0` | 8 | 2.0× | llama.cpp default quantized cache. |
| `q4_0` | 4 | 4.0× | llama.cpp 4-bit cache (may have quality issues). |

###  Known Limitations (as of March 26, 2026)

* **Speed regression**: turbo3/turbo4 are currently 3-8× slower than q8_0 on generation due to the Walsh-Hadamard rotation overhead. Optimization is ongoing.
* **Not production-ready**: This is a research prototype. Use for evaluation and learning.
* **M5 Max benchmarks, not M1 Pro**: The reported benchmarks were on M5 Max 128GB. Expect proportionally lower throughput on M1 Pro.

---

## 6. Implementation Path D: Build Your Own from the Paper (Claude Code Task)

### Purpose

This is the approach from the screenshot, using an AI coding agent (GPT 5.4 Codex in the tweet, but you can use Claude Code) to implement TurboQuant from the paper directly. This is the best path for **deep understanding** and for potentially building something tailored to your lab's needs.

### Claude Code Task Prompt

Feed this to Claude Code (or use it as CLAUDE.md context):

```markdown
## Task: Implement TurboQuant KV Cache Compression in MLX

### Context
I want to implement Google's TurboQuant algorithm (arXiv 2504.19874, ICLR 2026) for
KV cache compression, targeting Apple Silicon (M1 Pro, 16GB RAM) using the MLX framework.

### Paper Reference
- Paper: https://arxiv.org/abs/2504.19874
- Blog: https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/

### Algorithm Summary
TurboQuant is a two-stage vector quantizer for KV cache compression:

**Stage 1, PolarQuant (b-1 bits):**
1. Extract norm γ = ||x||, normalize x̂ = x/γ
2. Apply random orthogonal rotation Π (via QR decomposition of Gaussian matrix)
3. After rotation, coordinates ≈ N(0, 1/d) due to concentration of measure
4. Apply optimal scalar quantizer (Lloyd-Max) per coordinate independently
5. Store quantized indices + norm

**Stage 2, QJL (1 bit):**
1. Compute residual r = Πx̂ - dequant(quant(Πx̂))
2. Generate random sign matrix S ∈ {±1}^{m×d}
3. Store sign(S · r), just 1 bit per entry
4. This provides unbiased inner product estimation

**Key Math:**
- Rotation matrix: G = randn(d,d), Π, _ = QR(G)
- For efficiency: use Walsh-Hadamard transform (O(d log d) vs O(d²))
- Lloyd-Max centroids can be precomputed for Beta distribution
- Inner product estimate: ⟨q,k⟩ ≈ γ_q · γ_k · (⟨q̂_quant, k̂_quant⟩ + QJL_correction)

### Requirements
1. Pure MLX implementation (no PyTorch dependency)
2. Support bit widths: 2, 2.5, 3, 3.5, 4 bits per coordinate
3. Implement as a drop-in KV cache replacement
4. Include benchmarks: compression ratio, cosine similarity, tok/s
5. Test with Qwen2.5-1.5B-Instruct (small enough for 16GB)

### File Structure
```
turboquant_mlx/
├── rotation.py          # Random rotation (QR + Walsh-Hadamard)
├── codebook.py          # Lloyd-Max centroid computation
├── polar_quant.py       # Stage 1: PolarQuant
├── qjl.py               # Stage 2: QJL 1-bit correction
├── turboquant.py        # Combined two-stage quantizer
├── kv_cache.py          # MLX KV cache integration
├── benchmark.py         # Compression + inference benchmarks
├── test_turboquant.py   # Unit tests
└── README.md            # Documentation
```

### Validation Criteria
- Cosine similarity ≥ 0.95 at 3.5-bit
- Cosine similarity ≥ 0.91 at 3-bit
- Rotation should Gaussianize coordinates (kurtosis → 3.0)
- Inner product bias should be near-zero (QJL stage)
```

### Tips from the Screenshot Implementation

The person in the tweet gave GPT 5.4 Codex:
1. The model weights (so it could inspect tensor shapes)
2. The PDF of the paper (arXiv 2504.19874)
3. ~25 minutes of agent time

For Claude Code, you'd do:
```bash
# Download the paper
curl -o turboquant_paper.pdf https://arxiv.org/pdf/2504.19874

# Start Claude Code with context
claude --context turboquant_paper.pdf
# Then paste the task prompt above
```

---

## 7. Integration with Ollama (Future)

### Current Status (March 26, 2026)

* **Ollama Issue #15051** is filed as a feature request, just a link to the paper, no implementation
* Ollama uses its own Go-based inference engine (not llama.cpp anymore for newer versions)
* TurboQuant would need to be implemented in Ollama's native engine

### What Needs to Happen

1. llama.cpp mainline merges TurboQuant KV cache support
2. OR Ollama's Go engine implements it natively
3. Ollama exposes `--cache-type` flags (or similar) in Modelfile / CLI

### Estimated Timeline

* llama.cpp mainline: Q2-Q3 2026 (community PRs are already in progress)
* Ollama integration: Q3-Q4 2026 (depends on llama.cpp or native Go impl)

### Workaround for Now

If you build the llama.cpp TurboQuant fork (Path C), you can run it as an OpenAI-compatible server on port 8080 and point Osaurus or any OpenAI-compatible client at it, bypassing Ollama entirely for TurboQuant-enabled inference.

---

## 8. Benchmarking & Evaluation Plan

### Tests to Run on M1 Pro 16GB

```markdown
## Benchmark Matrix

### Hardware
- Apple M1 Pro, 16GB unified memory, 512GB SSD
- macOS version: [fill in]
- MLX version: [fill in]

### Models to Test
1. Qwen2.5-1.5B-Instruct-4bit (baseline, small)
2. Qwen2.5-3B-Instruct-4bit (realistic workload)
3. Mistral-7B-Instruct-v0.3-4bit (paper's reference model)

### Metrics to Capture
For each model × bit_width combination:
- [ ] Compression ratio (vs FP16 KV cache)
- [ ] Cosine similarity of attention scores
- [ ] Tokens per second (prompt processing)
- [ ] Tokens per second (generation)
- [ ] Peak memory usage (Activity Monitor or `memory_profiler`)
- [ ] Response quality (manual spot check)
- [ ] Needle-in-a-haystack accuracy (if time permits)

### Bit Widths to Test
- FP16 baseline (no compression)
- 4-bit TurboQuant
- 3.5-bit TurboQuant
- 3-bit TurboQuant
- 2.5-bit TurboQuant

### Context Lengths to Test
- 2K tokens
- 4K tokens
- 8K tokens
- 16K tokens (if memory permits)
- 32K tokens (with TurboQuant only, won't fit without it)
```

### Quick Validation Script

```python
"""
Quick needle-in-a-haystack test for TurboQuant on M1 Pro.
Embeds a fact in a long context and checks if the model can retrieve it.
"""
NEEDLE = "The secret password for the blueberry database is ANTHO-2026."
HAYSTACK_PADDING = "This is filler text about general agriculture. " * 500
PROMPT_TEMPLATE = f"""
{HAYSTACK_PADDING[:len(HAYSTACK_PADDING)//2]}
{NEEDLE}
{HAYSTACK_PADDING[len(HAYSTACK_PADDING)//2:]}

Question: What is the secret password for the blueberry database?
Answer:"""

# Run this with and without TurboQuant and compare outputs
```

---

## 9. Memory Budget Analysis for M1 Pro 16GB

```
Total Unified Memory:           16,384 MB
├── macOS + background apps:    ~3,000 MB
├── Available for ML:           ~13,384 MB
│
├── Model weights (Q4_K_M 7B):  ~4,200 MB
├── Model overhead / buffers:     ~500 MB
│
├── KV Cache Budget:             ~8,684 MB remaining
│   ├── FP16 KV at 8K ctx:      ~2,048 MB → OK
│   ├── FP16 KV at 32K ctx:     ~8,192 MB → TIGHT (swapping)
│   ├── FP16 KV at 64K ctx:    ~16,384 MB → IMPOSSIBLE
│   │
│   ├── TurboQuant 3.5-bit at 8K:   ~537 MB → VERY OK
│   ├── TurboQuant 3.5-bit at 32K: ~2,148 MB → OK
│   ├── TurboQuant 3.5-bit at 64K: ~4,296 MB → FEASIBLE!
│   │
│   ├── TurboQuant 3-bit at 8K:     ~418 MB → VERY OK
│   ├── TurboQuant 3-bit at 32K:  ~1,672 MB → EASY
│   └── TurboQuant 3-bit at 64K:  ~3,345 MB → FEASIBLE!
```

**Key insight**: TurboQuant could unlock **64K context on your 16GB M1 Pro** with a 7B model, something that's currently impossible without it.

---

## 10. Documentation & Lab Relevance

### Why This Matters for the Blueberry Breeding & Genomics Lab

TurboQuant is directly relevant to your lab's local AI infrastructure plans:

1. **Longer context windows**: Genomics papers, protocols, and datasets are long. Being able to feed 32-64K tokens of context to a local model (without sending data to the cloud) is a significant capability upgrade.

2. **Data privacy preserved**: TurboQuant is a runtime optimization, it doesn't change the model or require sending data anywhere. Your air-gapped deployment architecture is fully compatible.

3. **Hardware efficiency**: The Mac Studio M4 Max 64GB you're procuring for the lab would benefit even more, TurboQuant on 64GB could mean 200K+ context windows locally.

4. **Workshop material**: This could be a great topic for a future AI Workshop session, "How to run bigger models on smaller hardware."

### Suggested Documentation Structure

```
lab-docs/
├── turboquant-evaluation/
│   ├── README.md              # This file
│   ├── benchmark-results/
│   │   ├── m1-pro-16gb.md     # Your MacBook results
│   │   └── m4-max-64gb.md     # Lab Mac Studio results (future)
│   ├── implementation-notes/
│   │   ├── mlx-optiq-setup.md
│   │   └── llamacpp-fork-setup.md
│   └── workshop-materials/
│       └── session-N-turboquant.md
```

---

## 11. Key References

### Papers
* **TurboQuant**: [arXiv 2504.19874](https://arxiv.org/abs/2504.19874) (ICLR 2026)
* **PolarQuant**: [arXiv 2502.02617](https://arxiv.org/abs/2502.02617) (AISTATS 2026)
* **QJL**: [arXiv 2406.03482](https://arxiv.org/abs/2406.03482) (AAAI 2025)

### Blog & Media
* [Google Research Blog Post](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/) (March 24, 2026)

### Community Implementations
* [turboquant_plus](https://github.com/TheTom/turboquant_plus), Python + llama.cpp C port, 141 tests
* [mlx-optiq](https://pypi.org/project/mlx-optiq/), MLX native, pip installable
* [turboquant-pytorch](https://github.com/tonbistudio/turboquant-pytorch), PyTorch, validated on RTX 3060

### Tracking Issues
* [Ollama Issue #15051](https://github.com/ollama/ollama/issues/15051), Feature request
* [llama.cpp Discussion #20969](https://github.com/ggml-org/llama.cpp/discussions/20969), Integration discussion
* [llama.cpp Issue #20977](https://github.com/ggml-org/llama.cpp/issues/20977), Feature request with early branch
* [ik_llama.cpp Issue #1509](https://github.com/ikawrakow/ik_llama.cpp/issues/1509), Working implementation PR

### Community Results
* Prince Canuma (MLX): 6/6 needle-in-haystack at every quant level on Qwen3.5-35B
* tonbistudio (PyTorch): 99.5% attention cosine similarity at 3-bit on Qwen2.5-3B
* TheTom (llama.cpp): 4.9× compression working end-to-end on Apple Silicon

---

## Quick Start Checklist

* [ ] Clone turboquant_plus and run tests (Path A)
* [ ] Install mlx-optiq and run with Qwen2.5-1.5B (Path B)
* [ ] Run benchmark script at multiple bit widths
* [ ] Record memory usage at each configuration
* [ ] Test needle-in-a-haystack at different context lengths
* [ ] Document results in lab-docs/turboquant-evaluation/
* [ ] (Optional) Build llama.cpp TurboQuant fork (Path C)
* [ ] (Optional) Implement from paper using Claude Code (Path D)
* [ ] Share findings with Dr. Patricio and lab team

---

*Generated March 26, 2026. TurboQuant ecosystem is evolving rapidly, verify links and versions before use.*
