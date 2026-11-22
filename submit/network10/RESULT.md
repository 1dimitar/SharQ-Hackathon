# network10: 210800

| | |
|---|---|
| Instance | `network10` (10 nodes) |
| Binary variables | 90 |
| Largest quantum program | none (classical result) |
| Objective (referee) | 210800 |
| QOBLIB best known | 210800 |
| Gap | +0.00% (ties the best known value) |
| Method | Classical LP local search (degree-preserving arc exchanges scored by the routing LP) |

## Reproduce

Script `shinka_network/analysis/lp_local_search.py`; its run report is `shinka_network/results/reports/network10_lp_local_search.json`.

## Check

`qoblib_tools.check_network_solution("network10.sol", 10, load_network_demand(10))` returns `(True, 210800, 'valid')`.
