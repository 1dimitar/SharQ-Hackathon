# Leaderboard: QOBLIB 08 network design (whole repository)

Updated 2026-11-22 07:42 from `submissions.jsonl` (162 records logged). Built by `leaderboard.py` (`show`, `scan`, `submit`); do not edit by hand.

Score: the largest link load of the topology under optimal routing (the routing LP, lower is better), recomputed for every submission. Gap: score / BKV - 1, where BKV is QOBLIB's best known value (network05-10 proven optimal). `=` marks a tie with the BKV, ★ a score below it.

![Gap of our best network per instance to QOBLIB's best known value](gap_to_best_known.png)

## Best found per instance (any method)

| Instance | BKV | Best found | Gap | Method | Found by | Found | Solution |
|---|---:|---:|---:|---|---|---|---|
| network10 | 210800 | 210800 | +0.00% = | lp_local_search | analysis/lp_local_search.py | 2026-11-21 18:15 | [network10_210800_2fccf6.sol](solutions/network10_210800_2fccf6.sol) |
| network11 | 238334 | 238334 | +0.00% = | postproc (QAOA pool + evolved LP search) | init_run1_gen49, 10000 LP calls, pool network11_r3g128_c16p2loc_seed0 | 2026-11-22 03:26 | [network11_238334_851e52.sol](solutions/network11_238334_851e52.sol) |
| network12 | 276474 | 276474 | +0.00% = | lp_local_search | analysis/lp_local_search.py | 2026-11-21 18:41 | [network12_276474_df8fd3.sol](solutions/network12_276474_df8fd3.sol) |
| network13 | 300620 | 299715 | -0.30% ★ | postproc (QAOA pool + evolved LP search) | run 1 gen 49, 20000 LP calls, pool network13_r3g128_seed0 (shinka_postproc/results/reports/network13_gen49_20k_r3g128.json) | 2026-11-22 00:33 | [network13_299715_a17639.sol](solutions/network13_299715_a17639.sol) |
| network14 | 343091 | 341375 | -0.50% ★ | postproc (QAOA pool + evolved LP search) | harvest_A_gen20_20k (network14_r3g128_seed1) | 2026-11-22 05:52 | [network14_341375_f449b6.sol](solutions/network14_341375_f449b6.sol) |
| network15 | 370870 | 369720 | -0.31% ★ | postproc (QAOA pool + evolved LP search) | harvest_A_gen20_10k (network15_r3g128_seed0) | 2026-11-22 04:04 | [network15_369720_d9b257.sol](solutions/network15_369720_d9b257.sol) |
| network16 | 397908 | 395566 | -0.59% ★ | postproc (QAOA pool + evolved LP search) | harvest_A_gen20_20k (network16_r2g181_seed1) | 2026-11-22 05:27 | [network16_395566_4901bc.sol](solutions/network16_395566_4901bc.sol) |
| network17 | 437000 | 431938 | -1.16% ★ | postproc (QAOA pool + evolved LP search) | harvest_A_gen33_25k (network17_r2g181_seed1) | 2026-11-22 07:27 | [network17_431938_a77fdf.sol](solutions/network17_431938_a77fdf.sol) |
| network18 | 473375 | 466000 | -1.56% ★ | postproc (QAOA pool + evolved LP search) | harvest_A_gen20_10k (network18_r3g128_seed1) | 2026-11-22 04:42 | [network18_466000_95f2f2.sol](solutions/network18_466000_95f2f2.sol) |
| network19 | 492924 | 480000 | -2.62% ★ | postproc (QAOA pool + evolved LP search) | harvest_A_gen33_25k (network19_r2g181_seed1) | 2026-11-22 07:40 | [network19_480000_62cc4b.sol](solutions/network19_480000_62cc4b.sol) |
| network20 | 528528 | 511769 | -3.17% ★ | postproc (QAOA pool + evolved LP search) | harvest_A_gen20_20k_big (network20_r3g128_seed1) | 2026-11-22 07:15 | [network20_511769_335311.sol](solutions/network20_511769_335311.sol) |
| network21 | 569273 | 557053 | -2.15% ★ | postproc (QAOA pool + evolved LP search) | init_run1_gen49, 10000 LP calls, pool network21_r3g128_c16p2loc_seed0 | 2026-11-22 03:42 | [network21_557053_d75f44.sol](solutions/network21_557053_d75f44.sol) |
| network22 | 618135 | 597524 | -3.33% ★ | postproc (QAOA pool + evolved LP search) | harvest_A_gen20_20k_big (network22_r3g128_seed0) | 2026-11-22 06:19 | [network22_597524_1d1bd1.sol](solutions/network22_597524_1d1bd1.sol) |
| network23 | 659369 | 636848 | -3.42% ★ | postproc (QAOA pool + evolved LP search) | harvest_A_gen20_20k_big (network23_r2g181_seed0) | 2026-11-22 06:15 | [network23_636848_b474fe.sol](solutions/network23_636848_b474fe.sol) |
| network24 | 663688 | 659539 | -0.63% ★ | postproc (QAOA pool + evolved LP search) | harvest_A_gen20_net24_25k (network24_r3g128_seed0) | 2026-11-22 05:09 | [network24_659539_e7668a.sol](solutions/network24_659539_e7668a.sol) |

## Best found with quantum sampling

Topologies that came out of a quantum sampler (QAOA, PCE), possibly repaired or improved classically.

| Instance | BKV | Best found | Gap | Method | Found by | Found | Solution |
|---|---:|---:|---:|---|---|---|---|
| network11 | 238334 | 238334 | +0.00% = | postproc (QAOA pool + evolved LP search) | init_run1_gen49, 10000 LP calls, pool network11_r3g128_c16p2loc_seed0 | 2026-11-22 03:26 | [network11_238334_851e52.sol](solutions/network11_238334_851e52.sol) |
| network12 | 276474 | 277000 | +0.19% | postproc (QAOA pool + evolved LP search) | init_run1_gen49, 10000 LP calls, pool network12_r3g128_c20p2_seed0 | 2026-11-22 05:04 | [network12_277000_20212f.sol](solutions/network12_277000_20212f.sol) |
| network13 | 300620 | 299715 | -0.30% ★ | postproc (QAOA pool + evolved LP search) | run 1 gen 49, 20000 LP calls, pool network13_r3g128_seed0 (shinka_postproc/results/reports/network13_gen49_20k_r3g128.json) | 2026-11-22 00:33 | [network13_299715_a17639.sol](solutions/network13_299715_a17639.sol) |
| network14 | 343091 | 341375 | -0.50% ★ | postproc (QAOA pool + evolved LP search) | harvest_A_gen20_20k (network14_r3g128_seed1) | 2026-11-22 05:52 | [network14_341375_f449b6.sol](solutions/network14_341375_f449b6.sol) |
| network15 | 370870 | 369720 | -0.31% ★ | postproc (QAOA pool + evolved LP search) | harvest_A_gen20_10k (network15_r3g128_seed0) | 2026-11-22 04:04 | [network15_369720_d9b257.sol](solutions/network15_369720_d9b257.sol) |
| network16 | 397908 | 395566 | -0.59% ★ | postproc (QAOA pool + evolved LP search) | harvest_A_gen20_20k (network16_r2g181_seed1) | 2026-11-22 05:27 | [network16_395566_4901bc.sol](solutions/network16_395566_4901bc.sol) |
| network17 | 437000 | 431938 | -1.16% ★ | postproc (QAOA pool + evolved LP search) | harvest_A_gen33_25k (network17_r2g181_seed1) | 2026-11-22 07:27 | [network17_431938_a77fdf.sol](solutions/network17_431938_a77fdf.sol) |
| network18 | 473375 | 466000 | -1.56% ★ | postproc (QAOA pool + evolved LP search) | harvest_A_gen20_10k (network18_r3g128_seed1) | 2026-11-22 04:42 | [network18_466000_95f2f2.sol](solutions/network18_466000_95f2f2.sol) |
| network19 | 492924 | 480000 | -2.62% ★ | postproc (QAOA pool + evolved LP search) | harvest_A_gen33_25k (network19_r2g181_seed1) | 2026-11-22 07:40 | [network19_480000_62cc4b.sol](solutions/network19_480000_62cc4b.sol) |
| network20 | 528528 | 511769 | -3.17% ★ | postproc (QAOA pool + evolved LP search) | harvest_A_gen20_20k_big (network20_r3g128_seed1) | 2026-11-22 07:15 | [network20_511769_335311.sol](solutions/network20_511769_335311.sol) |
| network21 | 569273 | 557053 | -2.15% ★ | postproc (QAOA pool + evolved LP search) | init_run1_gen49, 10000 LP calls, pool network21_r3g128_c16p2loc_seed0 | 2026-11-22 03:42 | [network21_557053_d75f44.sol](solutions/network21_557053_d75f44.sol) |
| network22 | 618135 | 597524 | -3.33% ★ | postproc (QAOA pool + evolved LP search) | harvest_A_gen20_20k_big (network22_r3g128_seed0) | 2026-11-22 06:19 | [network22_597524_1d1bd1.sol](solutions/network22_597524_1d1bd1.sol) |
| network23 | 659369 | 636848 | -3.42% ★ | postproc (QAOA pool + evolved LP search) | harvest_A_gen20_20k_big (network23_r2g181_seed0) | 2026-11-22 06:15 | [network23_636848_b474fe.sol](solutions/network23_636848_b474fe.sol) |
| network24 | 663688 | 659539 | -0.63% ★ | postproc (QAOA pool + evolved LP search) | harvest_A_gen20_net24_25k (network24_r3g128_seed0) | 2026-11-22 05:09 | [network24_659539_e7668a.sol](solutions/network24_659539_e7668a.sol) |

## Best gap per method

| Method | network10 | network11 | network12 | network13 | network14 | network15 | network16 | network17 | network18 | network19 | network20 | network21 | network22 | network23 | network24 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| postproc (QAOA pool + evolved LP search) | – | +0.00% = | +0.19% | -0.30% ★ | -0.50% ★ | -0.31% ★ | -0.59% ★ | -1.16% ★ | -1.56% ★ | -2.62% ★ | -3.17% ★ | -2.15% ★ | -3.33% ★ | -3.42% ★ | -0.63% ★ |
| lp_local_search | +0.00% = | – | +0.00% = | – | – | – | – | – | – | – | – | – | – | – | – |
| postproc (QAOA pool + baseline LP search) | – | – | – | +2.20% | – | – | – | – | – | – | – | – | – | – | – |
| sa | +4.01% | – | +7.88% | +10.21% | +6.50% | +9.36% | +9.64% | +10.30% | +10.79% | +9.83% | +7.57% | +10.77% | +7.52% | +7.42% | +13.98% |
| qaoa_partitioned | – | +10.01% | +4.94% | +8.15% | +9.11% | +12.76% | +10.28% | +11.41% | +12.80% | +16.00% | +10.50% | +14.99% | +31.14% | +18.03% | +22.72% |

## See also

- Surrogate programs compared on network12-16 (mean score per program, several runs): [shinka_network/results/size_evals/PROGRAMS.md](../shinka_network/results/size_evals/PROGRAMS.md)
