#!/usr/bin/env python3
"""Evolve the surrogate term of the network-design QUBO with ShinkaEvolve.

    cd shinka_network
    uv run python run_evo.py --suite net10 --generations 20 --islands 4     # local
    ./run_docker.sh --suite net10 --generations 20 --islands 4             # in Docker

Needs OPENAI_API_KEY in the environment or in shinka_network/.env.
"""

import argparse
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.chdir(HERE)  # evaluate.py, initial.py and .env are looked up relative to this folder

from shinka.core import EvolutionConfig, ShinkaEvolveRunner  # noqa: E402
from shinka.database import DatabaseConfig  # noqa: E402
from shinka.launch import LocalJobConfig  # noqa: E402

TASK_SYS_MSG = """You are an expert in combinatorial optimisation, network design and QUBO modelling. You are improving one Python function, `surrogate_terms(demand)`, that builds the cost part of a QUBO.

## The problem (QOBLIB 08, network design)
Given an n x n traffic demand matrix t (t[i, j] = traffic from node i to node j, zero diagonal), build a directed network on n nodes in which every node has exactly 2 outgoing and 2 incoming links (no self-loops). All traffic is then routed over the links, flows may split over several paths, and the score of the network is the largest total flow on any single link. Lower is better. The exact score of a topology comes from a routing linear program; it is not a quadratic function of the link variables, so the QUBO can only approximate it.

## What you write
One binary variable x_ij per possible directed link (i != j). `surrogate_terms(demand)` returns `(linear, quadratic, penalty_weight)`:
- `linear`: dict {(i, j): weight}, the weight on x_ij.
- `quadratic`: dict {((i, j), (k, l)): weight}, the weight on the product x_ij * x_kl.
- `penalty_weight`: float in [0.05, 100].
Lower energy must mean a better network.

## How it is used and scored
1. Your terms are rescaled so the largest |weight| is 1.
2. The evaluator adds penalty_weight * [ sum_i (outdeg_i - 2)^2 + sum_j (indeg_j - 2)^2 ]. Do not add degree penalties yourself.
3. Simulated annealing draws a few dozen low-energy samples from this QUBO. Each sample is repaired to the nearest valid topology and scored by the routing LP.
4. Per annealing run: score = 0.5 * ref / best + 0.5 * ref / median, where best and median are over the sampled topologies and ref is the best score known for that demand matrix (the proven optimum where one exists). The combined score is the mean over several demand matrices and/or annealing seeds; 1.0 would mean every sample is optimal.
So the low-energy region of your QUBO must consist of topologies with a low maximum link load: the best sample matters, and so does the typical sample.

## Rules
- Use only the demand matrix. Pure Python and numpy (math, itertools are fine). No LP or MILP solvers, no optimisation or sampling libraries, no file or network access, no randomness that changes between calls.
- It must work for any n from 5 to 24 and run in a few seconds for n = 24 (hard limit 20 s per matrix).
- At most 4 * n^3 quadratic terms. Fewer terms are better when the score is equal: the QUBO is also run on quantum hardware simulators.
- Keys must be links (i, j) with i != j and integer node indices.
- Keep the function name, arguments and return format.

## What is known
- The baseline rewards each link by its own demand: linear[(i, j)] = -t[i, j] / max(t), no quadratic terms. It ignores that most traffic travels over several hops and that the score is a maximum, not a sum.
- Simply adding a reward -0.5 * t[i, k] on x_ij * x_jk for every two-hop path i -> j -> k scored worse than the baseline, so naive two-hop rewards are not enough.

## Directions worth exploring
- Each node has only 2 outgoing and 2 incoming links, so they must carry all of its own traffic plus transit traffic. Think about which pairs deserve a direct link and which can be served in two or three hops.
- Load balance: penalise pairs of links whose combination would overload one link or one node, for example heavy demands that would have to share a link.
- Estimate link loads under shortest-path style routing and build quadratic terms from a second-order expansion of a load-balancing objective such as the sum of squared loads.
- Node-level statistics (row sums, column sums, in/out imbalance) to decide where transit capacity is needed.
- Structure: reciprocal links (i -> j and j -> i), short cycles, hubs, and connectivity affect how far traffic must travel.
- The relative scale of linear and quadratic terms and the penalty weight matter a lot.

The feedback after each evaluation lists, per demand matrix, the gap of the best and of the median sampled topology to the reference, how many samples were valid before repair, and the number of quadratic terms. Use it to see what fails."""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="net10", help="instance suite, see problem.SUITES")
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--budget", type=float, default=5.0, help="max LLM spend in USD")
    parser.add_argument("--islands", type=int, default=4)
    parser.add_argument("--models", default="gpt-6-luna", help="comma-separated proposal models")
    parser.add_argument("--helper-model", default="gpt-6-luna", help="model for novelty checks and meta-recommendations")
    parser.add_argument("--eval-jobs", type=int, default=4)
    parser.add_argument("--proposal-jobs", type=int, default=4)
    parser.add_argument("--init", default="initial.py", help="starting program")
    parser.add_argument("--results-dir", default=None)
    args = parser.parse_args()

    results_dir = args.results_dir or f"results/{args.suite}_{time.strftime('%m%d_%H%M%S')}"
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
        num_islands=args.islands,
        archive_size=40,
        elite_selection_ratio=0.3,
        num_archive_inspirations=2,
        num_top_k_inspirations=2,
        migration_interval=10,
        migration_rate=0.1,
        island_elitism=True,
        parent_selection_strategy="weighted",
        parent_selection_lambda=10.0,
    )
    job_config = LocalJobConfig(
        eval_program_path="evaluate.py",
        extra_cmd_args={"suite": args.suite},
        time="00:10:00",
        python_executable=sys.executable,
        eval_verbose=False,
    )
    runner = ShinkaEvolveRunner(
        evo_config=evo_config,
        job_config=job_config,
        db_config=db_config,
        max_evaluation_jobs=args.eval_jobs,
        max_proposal_jobs=args.proposal_jobs,
        max_db_workers=2,
        verbose=True,
    )
    print(f"Suite: {args.suite} | results: {HERE / results_dir}")
    runner.run()


if __name__ == "__main__":
    main()
