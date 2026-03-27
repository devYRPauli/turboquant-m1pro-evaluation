# Fresh Run Status

Date: 2026-03-27

This file records what happened when the restart flow was actually re-executed from the current workspace state.

## What had to be fixed before the flow would run

### 1. MLX import in the default sandbox failed

Observed:

* importing `mlx` inside the default sandbox crashed with an `NSRangeException`
* rerunning the same import outside the sandbox succeeded

Interpretation:

* this was an environment/tooling issue around Metal access, not a repo code regression

### 2. The local MLX model cache was incomplete

Observed:

* Phase 2 initially hung on Hugging Face resolution
* offline rerun then failed with:
  * `FileNotFoundError: No safetensors found`
* inspection showed tokenizer/config files were present, but the actual model safetensor had not been fully downloaded

Action taken:

* repaired the cache by downloading the missing model files for:
  * `mlx-community/Qwen2.5-3B-Instruct-4bit`

Result:

* Phase 2 and Phase 3 could then run successfully in offline mode

## Phase 2 fresh rerun

Command used:

```bash
export HF_HUB_OFFLINE=1
source .venv-mlx/bin/activate
python3 benchmarks/phase2_inference_compare.py
```

Result:

* reproduced the earlier recorded behavior almost exactly

Key outputs:

* baseline:
  * `73.8 tok/s`
  * coherent response
* TurboQuant 4-bit MSE-only:
  * `35.4 tok/s`
  * degraded but still coherent
* TurboQuant 3-bit MSE-only:
  * degenerate repetition
* TurboQuant 2-bit MSE-only:
  * severe degeneration
* TurboQuant 4-bit with QJL:
  * `genetic genetic genetic...`
  * same collapse pattern as the notebook log

Conclusion:

* the original Phase 2 diagnosis still holds
* the QJL path remains broken for this setup

## Phase 3 fresh rerun

Command used:

```bash
export HF_HUB_OFFLINE=1
source .venv-mlx/bin/activate
python3 benchmarks/phase3_long_context.py
```

Result:

* MLX baseline reproduced as expected
* MLX TurboQuant 4-bit MSE-only reproduced as expected
* Ollama leg did not reproduce correctly in the current flow

### MLX baseline

* 2K: `17.1 tok/s`, needle `100%`
* 4K: `9.4 tok/s`, needle `100%`
* 8K: `4.6 tok/s`, needle `100%`
* 16K: `2.0 tok/s`, needle `100%`

### MLX TurboQuant 4-bit MSE-only

* 2K: `10.9 tok/s`, needle `0%`
* 4K: `6.1 tok/s`, needle `0%`
* 8K: `3.1 tok/s`, needle `0%`
* 16K: `1.3 tok/s`, needle `0%`

Conclusion:

* the long-context failure still reproduces
* the current MLX TurboQuant path is still not viable for the tested retrieval task

## New discrepancy vs prior log

### Ollama leg is currently broken in the benchmark flow

Observed in fresh Phase 3 run:

* Ollama rows returned `tok/s: 0.0`
* empty response text
* needle score `0%`

Notes:

* the Ollama server is reachable
* `qwen2.5:3b` is installed
* this means the problem is now in the benchmark execution path or request/response handling, not basic service availability

Implication:

* the historical statement "Ollama baseline passes in this flow" is no longer safe to assume without revalidating the Ollama runner logic

## Current truth after the fresh restart

### Reproduced

* Phase 2 MLX baseline behavior
* Phase 2 TurboQuant quality collapse patterns
* Phase 3 MLX baseline long-context success
* Phase 3 MLX TurboQuant long-context failure

### Newly uncovered issues

* sandboxed MLX import is not reliable
* local MLX model cache needed repair
* current Phase 3 Ollama benchmark leg is not producing usable output

## Updated next step

The next technically correct step is:

1. keep the main focus on the MLX QJL failure, because that behavior reproduced cleanly
2. separately repair the Ollama runner in `benchmarks/phase3_long_context.py` if Ollama-vs-MLX comparison is still needed

Priority remains with the MLX QJL path because that is the central quality blocker and it still reproduces after a fresh restart.
