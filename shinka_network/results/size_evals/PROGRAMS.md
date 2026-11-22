# Surrogate programs on network12-16

Updated 2026-11-22 01:01 by `eval_sizes.py` from `evaluations.jsonl` (8 evaluations); do not edit by hand. The best topologies also go to the repository leaderboard, [../../../leaderboard/LEADERBOARD.md](../../../leaderboard/LEADERBOARD.md).

One row per program, sampler and circuit settings (non-default settings in brackets), highest mean score first. Per size: gap of the best topology over all runs to QOBLIB's best known value, with the number of runs. Mean score: 0.5 BKV/best + 0.5 BKV/median per run (1.0 = every sampled topology optimal), averaged per size, then over the sizes of each band (sorted by the first band).

| # | Program | Sampler | n12 | n13 | n14 | n15 | n16 | n17 | n18 | n19 | n20 | n21 | n22 | n23 | n24 | Mean score n5-16 | Mean score n17-24 | Last run | Code |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 | run 2 best (gen 181) | qaoa_partitioned | +11.49% (2) | +13.27% (2) | +16.31% (2) | +15.76% (2) | +19.15% (2) | – | – | – | – | – | – | – | – | 0.8113 | – | 2026-11-22 00:57 | [9925d610403d.py](programs/9925d610403d.py) |
| 2 | baseline (qoblib_tools surrogate) | sa | +15.42% (2) | +13.97% (2) | +15.28% (2) | +12.55% (2) | +14.18% (2) | +12.83% (2) | +13.14% (2) | +14.44% (2) | +12.54% (2) | +13.88% (2) | +14.07% (2) | +10.80% (2) | +15.43% (2) | 0.7958 | 0.8179 | 2026-11-22 00:29 | [6c5c46dd684f.py](programs/6c5c46dd684f.py) |
| 3 | net12mix_qaoa_cloud_run4 gen 1 | qaoa_partitioned | +15.08% (2) | +18.12% (2) | +23.41% (2) | +16.02% (2) | +17.82% (2) | – | – | – | – | – | – | – | – | 0.7798 | – | 2026-11-22 00:59 | [7dfbabcf2139.py](programs/7dfbabcf2139.py) |
| 4 | run 3 recommended (gen 128) | sa | – | – | – | – | – | +10.54% (2) | +10.79% (2) | +9.83% (2) | +10.20% (2) | +10.77% (2) | +7.52% (2) | +7.42% (2) | +13.98% (2) | – | 0.8448 | 2026-11-22 00:30 | [7e360acb6d47.py](programs/7e360acb6d47.py) |
| 5 | run 2 best (gen 181) | sa | – | – | – | – | – | +10.30% (2) | +11.15% (2) | +10.28% (2) | +7.57% (2) | +11.31% (2) | +11.28% (2) | +8.50% (2) | +14.36% (2) | – | 0.8421 | 2026-11-22 00:29 | [9925d610403d.py](programs/9925d610403d.py) |
| 6 | run 3 in-run best (gen 109) | sa | – | – | – | – | – | +11.88% (2) | +16.62% (2) | +13.65% (2) | +10.53% (2) | +12.45% (2) | +11.19% (2) | +10.36% (2) | +16.51% (2) | – | 0.8229 | 2026-11-22 00:29 | [2796423eae62.py](programs/2796423eae62.py) |
| 7 | run 3 recommended (gen 128) | qaoa_partitioned | – | – | – | – | – | +24.59% (1) | – | – | – | – | – | – | – | – | 0.7738 | 2026-11-22 00:54 | [7e360acb6d47.py](programs/7e360acb6d47.py) |
