# network16: 395566

| | |
|---|---|
| Instance | `network16` (16 nodes) |
| Binary variables | 240 |
| Largest quantum program | 12 qubits (partitioned QAOA, blocks of at most 12 variables) |
| Objective (referee) | 395566 |
| QOBLIB best known | 397908 |
| Gap | -0.59% (beats the best known value) |
| Method | Hybrid: partitioned QAOA samples candidate topologies, then an evolved classical LP local search improves them |

## Settings

1. **QUBO.** Degree constraints as penalties plus the surrogate Hamiltonian `r2g181`: shinka_network/results/net10_luna_run2_200gen/gen_181/main.py (QUBO surrogate evolved with ShinkaEvolve, scored by simulated annealing, network10). Copy: `code/hamiltonians/r2g181.py`.
2. **Quantum sampling.** partitioned QAOA (blocks <= 12 vars, p=2, MC 50x10, 10000 shots, beam 200, top 200); seed 1; 16032 circuits on the local Maestro simulator, 32 s. The 200 sampled topologies: `code/pools/network16_r2g181_seed1.json`.
3. **Post-processing.** `harvest_A_gen20.py`, generation 20 of post-processing run 3 (shinka_postproc/results/postproc_small4_run3), 140 s time guard lifted; seed 0 (the default of `analysis/big_budget.py`; the launch command was not logged); 20000 routing-LP calls; 501 s. Copy: `code/postproc/harvest_A_gen20.py`.
4. **Routing and file.** Integer flows from `qoblib_tools.route_integral`, written by `qoblib_tools.write_network_solution`.

## Reproduce

From `shinka_postproc/` in the repository (the pool file must be in `shinka_postproc/pools/`):

```
uv run python analysis/big_budget.py --program harvest_A_gen20.py --budget 20000 --pools network16_r2g181_seed1.json --seed 0 --out-dir <dir>
```

## Check

`qoblib_tools.check_network_solution("network16.sol", 16, load_network_demand(16))` returns `(True, 395566, 'valid')`.
