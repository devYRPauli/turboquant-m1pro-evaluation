# TurboQuant Evaluation on M1 Pro 16GB

This repository tracks a fresh TurboQuant evaluation setup on a MacBook Pro with Apple M1 Pro and 16GB unified memory.

Scope:
* Build and test Aaryan Kapoor's `llama.cpp` TurboQuant fork on Apple Silicon.
* Run baseline versus TurboQuant KV-cache sanity checks.
* Run needle-in-a-haystack retrieval tests if sanity checks pass.
* Validate the updated `turboquant_plus` Python prototype and compare against prior findings.

Target environment:
* Hardware: Apple M1 Pro, 16GB unified memory
* OS: macOS Sequoia
* Python: 3.12

Observed host at initialization:
* OS version reported by `sw_vers -productVersion`: `26.4`
* Default `python3`: `3.14.3`
* Verified `python3.12`: `3.12.13`

Execution rules:
* Run `llama-cli` only in the foreground with `< /dev/null`.
* Kill stale `llama-cli` processes before every test run.
* Run one inference test at a time.
* Document every step and failure in `notebooks/experiment-log.md`.
