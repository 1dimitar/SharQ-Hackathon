# network12_quantum: 277000

| | |
|---|---|
| Instance | `network12` (12 nodes) |
| Binary variables | 132 |
| Largest quantum program | 20 qubits (partitioned QAOA, blocks of at most 20 variables) |
| Objective (referee) | 277000 |
| QOBLIB best known | 276474 |
| Gap | +0.19% |
| Method | Hybrid: partitioned QAOA samples candidate topologies, then an evolved classical LP local search improves them |

## Settings

1. **QUBO.** Degree constraints as penalties plus the surrogate Hamiltonian `r3g128`: shinka_network/results/net12_qaoa_cloud_run3/gen_128/main.py (QUBO surrogate evolved with ShinkaEvolve, scored by partitioned QAOA on Qoro cloud, network12). Copy: `code/hamiltonians/r3g128.py`.
2. **Quantum sampling.** partitioned QAOA (blocks <= 20 vars, p=2, MC 50x10, 10000 shots, beam 200, top 200, qoro_gpu); seed 0; 6012 circuits on the Qoro cloud, cluster qoro_gpu, 1389 s. The 200 sampled topologies: `code/pools/network12_r3g128_c20p2_seed0.json`.
3. **Post-processing.** `init_run1_gen49.py`, generation 49 of post-processing run 1 (shinka_postproc/results/postproc_net12_run1); seed 0; 10000 routing-LP calls; 75 s. Copy: `code/postproc/init_run1_gen49.py`.
4. **Routing and file.** Integer flows from `qoblib_tools.route_integral`, written by `qoblib_tools.write_network_solution`.

## Reproduce

From `shinka_postproc/` in the repository (the pool file must be in `shinka_postproc/pools/`):

```
uv run python analysis/big_budget.py --program init_run1_gen49.py --budget 10000 --pools network12_r3g128_c20p2_seed0.json --seed 0 --out-dir <dir>
```

## Check

`qoblib_tools.check_network_solution("network12.sol", 12, load_network_demand(12))` returns `(True, 277000, 'valid')`.
