# network22: 597524

| | |
|---|---|
| Instance | `network22` (22 nodes) |
| Binary variables | 462 |
| Largest quantum program | 12 qubits (partitioned QAOA, blocks of at most 12 variables) |
| Objective (referee) | 597524 |
| QOBLIB best known | 618135 |
| Gap | -3.33% (beats the best known value) |
| Method | Hybrid: partitioned QAOA samples candidate topologies, then an evolved classical LP local search improves them |

## Settings

1. **QUBO.** Degree constraints as penalties plus the surrogate Hamiltonian `r3g128`: shinka_network/results/net12_qaoa_cloud_run3/gen_128/main.py (QUBO surrogate evolved with ShinkaEvolve, scored by partitioned QAOA on Qoro cloud, network12). Copy: `code/hamiltonians/r3g128.py`.
2. **Quantum sampling.** partitioned QAOA (blocks <= 12 vars, p=2, MC 50x10, 10000 shots, beam 200, top 200); seed 0; 24048 circuits on the local Maestro simulator, 95 s. The 200 sampled topologies: `code/pools/network22_r3g128_seed0.json`.
3. **Post-processing.** `harvest_A_gen20.py`, generation 20 of post-processing run 3 (shinka_postproc/results/postproc_small4_run3), 140 s time guard lifted; seed 0 (the default of `analysis/big_budget.py`; the launch command was not logged); 20000 routing-LP calls; 779 s. Copy: `code/postproc/harvest_A_gen20.py`.
4. **Routing and file.** Integer flows from `qoblib_tools.route_integral`, written by `qoblib_tools.write_network_solution`.

## Reproduce

From `shinka_postproc/` in the repository (the pool file must be in `shinka_postproc/pools/`):

```
uv run python analysis/big_budget.py --program harvest_A_gen20.py --budget 20000 --pools network22_r3g128_seed0.json --seed 0 --out-dir <dir>
```

## Check

`qoblib_tools.check_network_solution("network22.sol", 22, load_network_demand(22))` returns `(True, 597524, 'valid')`.
