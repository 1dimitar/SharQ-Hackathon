# network17: 431938

| | |
|---|---|
| Instance | `network17` (17 nodes) |
| Binary variables | 272 |
| Largest quantum program | 12 qubits (partitioned QAOA, blocks of at most 12 variables) |
| Objective (referee) | 431938 |
| QOBLIB best known | 437000 |
| Gap | -1.16% (beats the best known value) |
| Method | Hybrid: partitioned QAOA samples candidate topologies, then an evolved classical LP local search improves them |

## Settings

1. **QUBO.** Degree constraints as penalties plus the surrogate Hamiltonian `r2g181`: shinka_network/results/net10_luna_run2_200gen/gen_181/main.py (QUBO surrogate evolved with ShinkaEvolve, scored by simulated annealing, network10). Copy: `code/hamiltonians/r2g181.py`.
2. **Quantum sampling.** partitioned QAOA (blocks <= 12 vars, p=2, MC 50x10, 10000 shots, beam 200, top 200); seed 1; 15030 circuits on the local Maestro simulator, 94 s. The 200 sampled topologies: `code/pools/network17_r2g181_seed1.json`.
3. **Post-processing.** `harvest_A_gen33.py`, generation 33 of post-processing run 3 (shinka_postproc/results/postproc_small4_run3), 140 s time guard lifted; seed 0 (the default of `analysis/big_budget.py`; the launch command was not logged); 25000 routing-LP calls; 727 s. Copy: `code/postproc/harvest_A_gen33.py`.
4. **Routing and file.** Integer flows from `qoblib_tools.route_integral`, written by `qoblib_tools.write_network_solution`.

## Reproduce

From `shinka_postproc/` in the repository (the pool file must be in `shinka_postproc/pools/`):

```
uv run python analysis/big_budget.py --program harvest_A_gen33.py --budget 25000 --pools network17_r2g181_seed1.json --seed 0 --out-dir <dir>
```

## Check

`qoblib_tools.check_network_solution("network17.sol", 17, load_network_demand(17))` returns `(True, 431938, 'valid')`.
