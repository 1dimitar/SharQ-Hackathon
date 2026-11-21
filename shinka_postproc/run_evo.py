#!/usr/bin/env python3
"""Evolve the classical post-processing of QAOA output (QOBLIB 08) with ShinkaEvolve.

    cd shinka_postproc
    uv run python run_evo.py --generations 100 --islands 4          # on this machine
    ./run_docker.sh --generations 100 --islands 4                   # in Docker

Needs OPENAI_API_KEY in shinka_postproc/.env (or the environment). No Qoro access is needed:
the QAOA candidate pools were sampled once by make_pools.py.
"""

import argparse
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.chdir(HERE)

from shinka.core import EvolutionConfig, ShinkaEvolveRunner  # noqa: E402
from shinka.database import DatabaseConfig  # noqa: E402
from shinka.launch import LocalJobConfig  # noqa: E402

TASK_SYS_MSG = """You are an expert in combinatorial optimisation, local search and network design. You are improving one Python function, `improve(demand, candidates, tools, rng)`: the classical post-processing that turns raw quantum (QAOA) samples into the best possible network.

## The problem (QOBLIB 08, network design)
Given an n x n traffic demand matrix (demand[i, j] = traffic from node i to node j), choose a directed network on n nodes in which every node has exactly 2 outgoing and 2 incoming links (no self-loops). All traffic is then routed, flows may split over several paths, and the score of the network is the largest total flow on any single link. Lower is better. The exact score of a network comes from a routing linear program (LP). For the instance used here (network12) the best known score is 276474.

## The pipeline
1. A fixed QUBO Hamiltonian (found earlier) predicts network quality.
2. Partitioned QAOA sampled it; its 100 best raw solutions are your `candidates`, best QUBO energy first. They are link lists and most break the degree rules.
3. YOUR FUNCTION corrects and improves them and returns the best valid network it can find.

## Tools (the only way to reach the LP)
- tools.lp(arcs) -> score. Costs 1 LP call per NEW valid network (repeats are free and cached; invalid networks return inf for free).
- tools.lp_detail(arcs) -> {"score", "loads": {link: flow}, "prices": {link: dual price}} for the same cost. Links with price > 0 are the bottlenecks that set the score; loads show how close the others are.
- tools.repair(arcs) -> closest valid network keeping as many of the given links as possible (free).
- tools.is_valid(arcs), tools.is_cached(arcs) (free).
- tools.energy(arcs) -> the fixed Hamiltonian's predicted quality (free, lower = predicted better, only loosely correlated with the LP score).
- tools.lp_calls_left, tools.n.
There is a fixed budget of LP calls per run. When it is spent, tools.lp raises BudgetExhausted; that is fine, the evaluator then takes the best network the LP has scored during the run. Every LP call should therefore count.

## Scoring
Each evaluation runs your function on several QAOA candidate pools. Score per pool = 276474 / best LP score found; the combined score is the mean (1.0 = best known reached). The same function is also run from random networks instead of the QAOA candidates; that is reported as a control, not scored.

## Rules
- Reach the LP only through `tools`: do not import scipy or any solver and do not write your own LP or flow solver. Do not access attributes of `tools` starting with "_" or tamper with the object; no file, network, multiprocessing or threading use. numpy, math, itertools, heapq, collections are fine.
- Use `rng` for all randomness. Keep the function name, arguments and return value.
- Each run must finish within 150 s of wall time (the LP calls themselves take about 4 ms each).

## What is known
- The baseline repairs every candidate, does first-improvement 2-exchange descent ((a->b),(c->d) => (a->d),(c->b)) from the 5 best, then iterated local search with random 2-4 swap kicks. With 2000 LP calls it ends 3-7% above the best known.
- With about 200 000 LP calls, plain iterated local search does reach 276474, so the challenge is to get there with far fewer LP calls.

## Directions worth exploring
- Use lp_detail: only bottleneck links (price > 0) can lower the score when moved, so focus swaps on them instead of trying all pairs.
- Rank candidate moves for free with tools.energy, link loads or demand before spending LP calls; avoid re-scoring networks (tools.is_cached).
- Better acceptance rules (simulated annealing, tabu lists, late acceptance), restarts from diverse candidates, path relinking between good networks, 3-link rotations.
- Better correction of the raw QAOA candidates than tools.repair (e.g. demand- or load-aware), and smarter choice of which candidates deserve LP calls at all.

The feedback after each evaluation lists, per pool, the best score and gap to the best known, LP calls used, run time, and the same program's result from random starts."""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--islands", type=int, default=4)
    parser.add_argument("--budget", type=float, default=5.0, help="max LLM spend in USD")
    parser.add_argument("--models", default="gpt-6-luna")
    parser.add_argument("--helper-model", default="gpt-6-luna")
    parser.add_argument("--eval-jobs", type=int, default=4)
    parser.add_argument("--proposal-jobs", type=int, default=4)
    parser.add_argument("--lp-budget", type=int, default=2000, help="LP calls per run")
    parser.add_argument("--pools", default="network12_seed0.json,network12_seed1.json,network12_seed2.json")
    parser.add_argument("--init", default="initial.py")
    parser.add_argument("--results-dir", default=None)
    args = parser.parse_args()

    missing = [p for p in args.pools.split(",") if not (HERE / "pools" / p).exists()]
    if missing:
        sys.exit(f"missing candidate pools {missing}: run make_pools.py first")
    results_dir = args.results_dir or f"results/postproc_{time.strftime('%m%d_%H%M%S')}"
    evo_config = EvolutionConfig(
        task_sys_msg=TASK_SYS_MSG,
        patch_types=["diff", "full", "cross"],
        patch_type_probs=[0.6, 0.3, 0.1],
        num_generations=args.generations,
        max_api_costs=args.budget,
        max_patch_resamples=3,
        max_patch_attempts=1,
        job_type="local",
        language="python",
        llm_models=args.models.split(","),
        llm_kwargs={"temperatures": [0.0, 0.5, 1.0], "reasoning_efforts": ["low", "medium"], "max_tokens": 32768},
        llm_dynamic_selection="ucb1",
        llm_dynamic_selection_kwargs={"exploration_coef": 1.0, "cost_aware_coef": 0.5},
        meta_rec_interval=5,
        meta_llm_models=[args.helper_model],
        meta_llm_kwargs={"temperatures": [0.0], "max_tokens": 16384},
        embedding_model="text-embedding-3-small",
        code_embed_sim_threshold=0.995,
        novelty_llm_models=[args.helper_model],
        novelty_llm_kwargs={"temperatures": [0.0]},
        max_novelty_attempts=2,
        use_text_feedback=True,
        init_program_path=args.init,
        results_dir=results_dir,
    )
    db_config = DatabaseConfig(
        num_islands=args.islands, archive_size=40, elite_selection_ratio=0.3,
        num_archive_inspirations=2, num_top_k_inspirations=2, migration_interval=10,
        migration_rate=0.1, island_elitism=True, parent_selection_strategy="weighted",
        parent_selection_lambda=10.0,
    )
    job_config = LocalJobConfig(
        eval_program_path="evaluate.py",
        extra_cmd_args={"pools": args.pools, "lp_budget": args.lp_budget},
        time="00:15:00",
        python_executable=sys.executable,
        eval_verbose=False,
    )
    runner = ShinkaEvolveRunner(
        evo_config=evo_config, job_config=job_config, db_config=db_config,
        max_evaluation_jobs=args.eval_jobs, max_proposal_jobs=args.proposal_jobs,
        max_db_workers=2, verbose=True,
    )
    print(f"pools: {args.pools} | LP budget {args.lp_budget} | results: {HERE / results_dir}")
    runner.run()


if __name__ == "__main__":
    main()
