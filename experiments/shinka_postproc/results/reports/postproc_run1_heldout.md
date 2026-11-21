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

## Correction (analysis/quantum_value.py, 5 random start sets instead of 1)

The "random starts are as good" finding above came from a single random start set (set 0), which
turned out to be the luckiest of five. With 5 random sets, QAOA starts are better on average at
every LP budget. Held-out QAOA pools only (seeds 3-4) vs 5 random sets, evolved program (gen 49):

| LP budget | QAOA (held-out pools) mean gap | random starts mean gap (min..max) |
|---:|---:|---:|
| 50 | +10.60% | +18.76% (+14.94..+24.28%) |
| 100 | +9.02% | +15.29% (+11.14..+22.33%) |
| 200 | +7.38% | +9.07% (+6.43..+11.63%) |
| 500 | +3.15% | +5.70% (+3.37..+7.79%) |
| 1000 | +2.72% | +4.11% (+0.84..+6.24%) |
| 2000 | +2.06% | +2.70% (+0.84..+4.10%) |

Before any search, the repaired QAOA candidates are clearly better than random networks (best
+8.9..17.4% vs +18.2..23.9%, median +22..34% vs +44..51%) but much less diverse (58-64% mean link
overlap vs 18% for random networks). The advantage is a head start that shrinks as the LP budget
grows; ranges overlap from 200 calls on. Sample sizes are small (2 held-out pools, 5 random sets).
Full output: quantum_value_run1.txt (all 5 pools) and quantum_value_run1_heldout_only.txt.
