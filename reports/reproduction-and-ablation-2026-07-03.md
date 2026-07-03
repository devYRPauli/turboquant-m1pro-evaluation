# QJL Reproduction and Ablation (2026-07-03)

This report documents a follow-up verification pass run after the paper review.
It has two goals:

1. Settle the disputed 2K needle score for the validated Hybrid K5/V4
   configuration by reproducing the original phase3 result from a clean,
   independently rebuilt environment.
2. Isolate which of the three QJL changes (orthogonal projection, matched scale
   factor, damping) is responsible for escaping the degeneration seen with the
   paper-faithful configuration.

All runs used Qwen2.5-3B-Instruct-4bit on the same Apple M1 Pro 16GB machine as
the rest of the study, greedy decoding (temperature 0), and fixed seeds.

## Environment

The round 1 virtual environment interpreter no longer resolved (it pointed at a
moved path), so a clean environment was rebuilt with the exact pinned versions
recorded in the original environment: mlx 0.31.1, mlx-lm 0.31.1, mlx-optiq 0.0.1,
transformers 5.3.0, numpy 2.4.3, Python 3.12.

For the reproduction, the two modified optiq source files from the original round
1 environment (`turbo_quant.py` with orthogonal QJL and the sqrt(d) scale, and
`turbo_kv_cache.py` with the k_damping default of 0.7) were installed over the
stock package, recreating the exact modified package that produced the original
results. The stock 0.0.1 wheel from PyPI was also retained for the ablation, which
constructs the projection, scale, and damping variants directly via subclasses.

## Part 1: Reproduction of the phase3 Hybrid result

Harness: the original `build_prompt` (with instruct question framing) and
`make_turbo_kv_caches(bits=(5,4), use_qjl=True, seed=42)`, scored with the
original `score_answer`. This is the same code path that produced
`benchmarks/phase3_results.json`. Script: `benchmarks/hybrid_reproduction.py`.
Raw output: `logs/hybrid-reproduction-2026-07-03.json`.

| Context | Reproduced score | Original phase3_results.json | Match |
|---|---|---|---|
| 2K | 0.5 (found VcMYB4, wrote "FROSTst7") | 0.5 (same, "FROSTst7") | exact |
| 4K | 1.0 (both facts) | 1.0 | exact |
| 8K | 1.0 (both facts) | 1.0 | exact |
| 16K | 1.0 (both facts) | 1.0 | exact |

The 2K result reproduced exactly, including the specific failure mode: the model
retrieves the locus name VcMYB4 but corrupts the allele name FROSTBLOCK-7 into
"FROSTst7". This confirms that the 2K score is genuinely 0.5 and is deterministic,
not a one-time logging artifact. The claim of 100 percent at 2K that appeared in
the round 1 post-mortem summary is not reproducible and is not supported by any
raw output in this repository. The 4K, 8K, and 16K results reproduced at 100
percent.

MLX Metal peak memory during the 16K run was approximately 3.3 GB, well within the
16 GB budget.

## Part 2: QJL ablation at 2K

Harness: a lightweight synthetic prompt (a single repeated filler sentence with
the needle at the midpoint) and no instruct chat template. This is a qualitative
probe of generation stability, not a retrieval measurement: without chat framing,
even a stable configuration continues the passage rather than answering, so all
needle scores here are 0. What varies, and what the ablation measures, is whether
generation degenerates. Script: `benchmarks/qjl_ablation.py`. Raw output:
`logs/qjl-ablation-2026-07-03.json`.

| Projection | Scale | Damping | Observed generation behavior |
|---|---|---|---|
| gaussian (paper) | sqrt(pi/2)/d (paper) | 1.0 | Word-loop degeneration ("gen gen gen ...") |
| gaussian | sqrt(pi/2)/sqrt(d) | 1.0 | Collapse ("of which in") |
| orthogonal | sqrt(pi/2)/d | 1.0 | Word-loop degeneration ("complex complex process ...") |
| orthogonal | sqrt(pi/2)/sqrt(d) | 1.0 | Semi-coherent, attends to needle region ("... locus ... blueberry breeding ...") |
| orthogonal | sqrt(pi/2)/sqrt(d) | 0.7 | Coherent text |
| gaussian (paper) | sqrt(pi/2)/d (paper) | 0.7 | Word-loop degeneration |

Reading the table:

1. The paper-faithful configuration (row 1, Gaussian projection with the 1/d
   scale) degenerates into word loops, reproducing the core failure at head
   dimension 128.
2. Changing only the scale on the Gaussian matrix (row 2) or only the matrix while
   keeping the mismatched scale (row 3) does not help. Both still fail. This is the
   direct evidence that the projection and scale are one coupled substitution: the
   orthogonal matrix requires the sqrt(d) scale, and neither change works alone.
3. The matched pair (orthogonal projection with the sqrt(d) scale, row 4) is what
   pulls generation out of degeneration into semi-coherent text that visibly
   attends to the needle region.
4. Adding the 0.7 damping factor (row 5) further stabilizes generation into fully
   coherent text. This is the validated configuration.
5. Applying damping to the paper-faithful Gaussian configuration (row 6) does
   nothing. Damping is only useful once the projection and scale are fixed.

The ablation confirms the framing used in the README and FINDINGS: the fix is a
coupled substitution (orthogonal projection plus matched scale) that escapes
degeneration, with damping as a stabilizing refinement, not two independent bug
fixes to a broken formula.

## Reproducing this pass

Both scripts require a Python environment with the modified optiq package
installed (orthogonal QJL projection and sqrt(d) scale). See the Environment
section above. Then:

```
python benchmarks/hybrid_reproduction.py   # writes logs/hybrid-reproduction.json
python benchmarks/qjl_ablation.py          # writes logs/qjl-ablation.json
```

The committed JSON files carry the date suffix 2026-07-03 to mark them as the
outputs from this pass.
