# Post-processing run 1 (network12, 100 generations, gpt-6-luna, 4 islands) - held-out test

Training: pools network12_seed0-2 (partitioned QAOA on Qoro cloud, frozen Hamiltonian = run 3 gen 28),
LP budget 2000 calls per run. Best program: gen 49, combined score 0.9901 (baseline 0.9671).
LLM cost $0.34.

Held-out pools network12_seed3 and seed4 (never seen during evolution), `analysis/heldout.py`.
Gap to best known 276474. "random starts" = same program from random valid networks (same random
set for both pools, seed 0).

| Program | LP budget | QAOA pool 3 | QAOA pool 4 | Random starts |
|---|---:|---:|---:|---:|
| baseline (initial.py) | 2000 | 286177 (+3.51%) | 290334 (+5.01%) | 300527 (+8.70%) |
| baseline (initial.py) | 10000 | 282239 (+2.09%) | 287200 (+3.88%) | 283847 (+2.67%) |
| evolved (gen 49) | 2000 | 281154 (+1.69%) | 283167 (+2.42%) | 278786 (+0.84%) |
| evolved (gen 49) | 10000 | 277572 (+0.40%) | 277572 (+0.40%) | 277000 (+0.19%) |

Findings:
- The evolved post-processing generalises to unseen pools: about half the baseline's gap at 2000 LP
  calls, 0.4% at 10000 (baseline 2-4%).
- With the evolved post-processing, random starts are as good as or better than QAOA starts on the
  held-out pools; the QAOA advantage seen on the training pools (0.990 vs 0.973) does not hold up.
