# TurboQuant Fresh Restart Checklist

Date: 2026-03-27

Use this checklist to rerun the current local flow from a clean starting point and verify which parts still work.

## Objective

Re-execute the previously documented local workflow in order:

1. validate environment assumptions
2. rerun Phase 2
3. rerun Phase 3
4. compare fresh results against the existing experiment log
5. decide whether the current flow is still valid

## Preconditions

* Workspace root:
  * `/Users/yashrajpandey/Desktop/TurboQuant`
* MLX virtual environment exists:
  * `.venv-mlx`
* Local model assets are already cached
* Ollama is installed if Phase 3 is run with the Ollama baseline

## Execution steps

### Step 1: enter workspace and activate environment

```bash
cd /Users/yashrajpandey/Desktop/TurboQuant
source .venv-mlx/bin/activate
python3 --version
```

Expected:

* Python 3.12.x

### Step 2: verify core imports

```bash
python3 - <<'PY'
import mlx
import mlx_lm
import optiq
print("imports-ok")
PY
```

Expected:

* `imports-ok`

### Step 3: rerun Phase 2 benchmark

```bash
python3 benchmarks/phase2_inference_compare.py
```

Expected shape of result from prior session:

* baseline: coherent answer
* TurboQuant 4-bit MSE-only: degraded but coherent
* TurboQuant 3-bit and 2-bit: degenerate
* TurboQuant 4-bit with QJL: degenerate loop

Decision:

* If this no longer reproduces, stop and compare environment drift first.

### Step 4: rerun Phase 3 long-context benchmark

```bash
python3 benchmarks/phase3_long_context.py
```

Expected shape of result from prior session:

* Ollama baseline: pass
* MLX baseline: pass
* MLX TurboQuant 4-bit MSE-only: fail needle retrieval at all tested lengths

Decision:

* If Ollama is unavailable, note that separately from MLX behavior.
* If the MLX TurboQuant result changes materially, diff the package versions and cached model path.

### Step 5: compare fresh run against recorded log

Reference file:

* `/Users/yashrajpandey/Desktop/TurboQuant/notebooks/experiment-log.md`

Check:

* same broad quality pattern
* same failure mode for QJL
* same long-context collapse for MSE-only TurboQuant

### Step 6: choose next branch

If the old failure reproduces:

* proceed to targeted MLX QJL debugging

If the old failure does not reproduce:

* capture exact environment drift
* update the handoff and experiment log before continuing

## Minimum output expected from this restart

At the end of this flow, produce:

1. current command status
2. whether Phase 2 reproduced
3. whether Phase 3 reproduced
4. whether the old handoff is still accurate
5. the next recommended work item
