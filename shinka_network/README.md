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
| `eval_sizes.py` | Evaluates programs (one file, or every new best of a running evolution) on network12-16 and sends the topologies to the repository leaderboard (`../leaderboard.py`). Program comparison in `results/size_evals/PROGRAMS.md`. |
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
# several instances per evaluation, sampled in parallel (run 4): network12 + 5 random 12-node
# subsets of the 24-node matrix. Compute the subset references once, before the run.
uv run python analysis/subset_refs.py --suite net12mix
uv run python run_evo.py --suite net12mix --sampler qaoa_partitioned --backend cloud \
    --max-runs 6 --parallel-instances 6 --eval-jobs 16 --proposal-jobs 8 --parent-lambda 2 \
    --islands 8 --generations 200 --models gpt-6-luna --results-dir results/net12mix_qaoa_cloud_run4
uv run python analysis/compare_network10.py --program results/<run>/best/main.py
uv run python analysis/compare_sizes.py --program results/<run>/best/main.py
```

## Programs on network12-16, and the repository leaderboard

`eval_sizes.py` runs a program through the evolution pipeline (QUBO, sampler, repair, routing LP) on
the QOBLIB instances network12 to network16. The best topology of every instance run goes to the
repository leaderboard (`../leaderboard.py`, `../leaderboard/LEADERBOARD.md`) as soon as it is found.
`results/size_evals/PROGRAMS.md` compares the programs (best gap per size, mean score). Runs under
the root `.venv`.

```bash
cd shinka_network
# every new best program of a running evolution (stops when results/<run>.pid exits)
nohup ../.venv/bin/python -u eval_sizes.py watch results/net12mix_qaoa_cloud_run4 \
    > results/size_evals/watch_net12mix_qaoa_cloud_run4.log 2>&1 &
# one program (any idea worth comparing); --sampler sa is free and takes seconds
../.venv/bin/python eval_sizes.py eval results/<run>/gen_<g>/main.py --label "run 3 gen 128"
../.venv/bin/python eval_sizes.py eval initial.py --label baseline --sampler sa --repeats 4
../.venv/bin/python eval_sizes.py show      # rebuild and print results/size_evals/PROGRAMS.md
```

Defaults: partitioned QAOA on Qoro's cloud, sizes 12-16 (`--sizes 12-20` or `12,16,20` for others),
2 runs per size (`--repeats`), 4 instance runs at a time (`--parallel`). Keep `--parallel` low
while an evolution runs on the cloud: on 2026-11-21 about 140 parallel instance runs made Qoro
calls fail (connection aborted, write timeouts, jobs cancelled by the scheduler), and ~26 still
caused occasional upload timeouts (Divi gives an upload 100 s).

Circuit settings can be changed per evaluation (`--cluster-size` = qubits per QAOA block,
`--layers`, `--mc-population`, `--mc-iterations`, `--shots`, `--top-n`) and the simulator cluster
with `--cluster qoro_gpu|qoro_cpu|qoro_maestro`. Non-default settings are added to the label
(e.g. `[cluster_size=20]`) and get their own row in PROGRAMS.md. With the default 12-variable
blocks every circuit has at most 12 qubits whatever the network size; a GPU only pays off for
larger blocks.

## Runs

| Run | Suite | Generations | Model | Best score | LLM cost |
|---|---|---:|---|---:|---:|
| `net10_luna_run1` | network10, 4 SA seeds | 20 | gpt-6-luna | 0.820 (baseline 0.800) | $0.04 |
| `net10_luna_run2_200gen` | network10, 4 SA seeds | 200 | gpt-6-luna | 0.850 | $0.41 |
