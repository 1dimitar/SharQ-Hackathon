# network12: 276474

| | |
|---|---|
| Instance | `network12` (12 nodes) |
| Binary variables | 132 |
| Largest quantum program | none (classical result) |
| Objective (referee) | 276474 |
| QOBLIB best known | 276474 |
| Gap | +0.00% (ties the best known value) |
| Method | Classical LP local search (degree-preserving arc exchanges scored by the routing LP) |

## Reproduce

Script `shinka_network/analysis/lp_local_search.py`; its run report is `shinka_network/results/reports/network12_lp_local_search.json`.

## Check

`qoblib_tools.check_network_solution("network12.sol", 12, load_network_demand(12))` returns `(True, 276474, 'valid')`.
