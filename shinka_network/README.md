# Evolving the network-design QUBO surrogate with ShinkaEvolve

QOBLIB 08 (notebook `02_network_design.ipynb`): a QUBO proposes network topologies and the routing
LP scores them. `qoblib_tools.build_network_qubo` rewards each link by its own demand,
`-t_ij / max t`. Here ShinkaEvolve evolves that term: the function `surrogate_terms(demand)` in
`initial.py` returns linear terms, quadratic terms and a penalty weight. The degree penalty,
sampling, repair and LP scoring stay fixed.

## Files

| File | Role |
|---|---|
| `initial.py` | Starting program (the qoblib_tools surrogate). Only the EVOLVE block changes. |
| `problem.py` | Instance suites, QUBO assembly, repair + LP scoring, rules for evolved code. |
| `evaluate.py` | Evaluator with local simulated annealing (runs in this folder's environment). |
| `evaluate_qoro.py` | Evaluator with Divi PCE or partitioned QAOA, on Qoro's cloud or the local Maestro simulator. Runs in the repository's root `.venv`. |
| `run_evo.py` | Launches ShinkaEvolve: system message, models, islands, evaluator choice. |
| `Dockerfile`, `run_docker.sh` | Portable runs with the annealing evaluator. |
| `analysis/` | Comparisons of an evolved program against the qoblib_tools surrogate. |
| `results/` | Every run: programs, scores, logs, ShinkaEvolve database. `results/reports/` has the analyses. |

Score per run: `0.5 * ref/best + 0.5 * ref/median` over the sampled topologies (1.0 = optimal).

## Environments

ShinkaEvolve needs `rich<15` and qoro-divi needs `rich>=15`, so there are two environments:
this folder's (`uv sync` here: ShinkaEvolve, annealing evaluator) and the repository root's
(`uv sync` at the root: notebook, Divi, `evaluate_qoro.py`). `run_evo.py` calls the right one.

Keys go in `shinka_network/.env` (git-ignored): `OPENAI_API_KEY=...` and `QORO_API_KEY=...`.

## Commands

```bash
cd shinka_network
# annealing evaluator, in Docker
./run_docker.sh --suite net10 --generations 200 --islands 4
# quantum evaluator on Qoro's cloud (ShinkaEvolve runs on this machine)
uv run python run_evo.py --suite net12 --sampler qaoa_partitioned --backend cloud \
    --init results/net10_luna_run2_200gen/best/main.py
uv run python analysis/compare_network10.py --program results/<run>/best/main.py
uv run python analysis/compare_sizes.py --program results/<run>/best/main.py
```

## Runs

| Run | Suite | Generations | Model | Best score | LLM cost |
|---|---|---:|---|---:|---:|
| `net10_luna_run1` | network10, 4 SA seeds | 20 | gpt-6-luna | 0.820 (baseline 0.800) | $0.04 |
| `net10_luna_run2_200gen` | network10, 4 SA seeds | 200 | gpt-6-luna | 0.850 | $0.41 |
