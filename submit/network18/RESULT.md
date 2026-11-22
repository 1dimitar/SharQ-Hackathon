# network18: 466000

| | |
|---|---|
| Instance | `network18` (18 nodes) |
| Binary variables | 306 |
| Largest quantum program | 12 qubits (partitioned QAOA, blocks of at most 12 variables) |
| Objective (referee) | 466000 |
| QOBLIB best known | 473375 |
| Gap | -1.56% (beats the best known value) |
| Method | Hybrid: partitioned QAOA samples candidate topologies, then an evolved classical LP local search improves them |

## Settings

1. **QUBO.** Degree constraints as penalties plus the surrogate Hamiltonian `r3g128`: shinka_network/results/net12_qaoa_cloud_run3/gen_128/main.py (QUBO surrogate evolved with ShinkaEvolve, scored by partitioned QAOA on Qoro cloud, network12). Copy: `code/hamiltonians/r3g128.py`.
2. **Quantum sampling.** partitioned QAOA (blocks <= 12 vars, p=2, MC 50x10, 10000 shots, beam 200, top 200); seed 1; 18036 circuits on the local Maestro simulator, 109 s. The 200 sampled topologies: `code/pools/network18_r3g128_seed1.json`.
3. **Post-processing.** `ckpt_A_gen20.py`, generation 20 of post-processing run 3 (shinka_postproc/results/postproc_small4_run3), as evolved (140 s time guard); seed 0 (the default of `analysis/big_budget.py`; the launch command was not logged); 5593 routing-LP calls (budget 10000; the program's 140 s time guard stopped it at 5593 calls); 140 s. Copy: `code/postproc/ckpt_A_gen20.py`.
4. **Routing and file.** Integer flows from `qoblib_tools.route_integral`, written by `qoblib_tools.write_network_solution`.

## Reproduce

From `shinka_postproc/` in the repository (the pool file must be in `shinka_postproc/pools/`):

```
uv run python analysis/big_budget.py --program ckpt_A_gen20.py --budget 10000 --pools network18_r3g128_seed1.json --seed 0 --out-dir <dir>
```

The 140 s guard makes this run depend on machine speed; on another machine the score can differ.

## Check

`qoblib_tools.check_network_solution("network18.sol", 18, load_network_demand(18))` returns `(True, 466000, 'valid')`.
