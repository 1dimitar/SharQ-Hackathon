#!/usr/bin/env python3
"""Evolve the surrogate term of the network-design QUBO with ShinkaEvolve.

    cd shinka_network
    uv run python run_evo.py --suite net10 --generations 20 --islands 4     # local
    ./run_docker.sh --suite net10 --generations 20 --islands 4             # in Docker

Quantum evaluations on Qoro's cloud (ShinkaEvolve runs here, every circuit goes to Qoro):
    uv run python run_evo.py --suite net12 --sampler qaoa_partitioned --backend cloud \
        --init results/<run>/best/main.py
--sampler pce|qaoa_partitioned runs evaluate_qoro.py under the repository's root .venv (it needs
qoro-divi, which cannot share an environment with ShinkaEvolve), so these runs are host-only.

Needs OPENAI_API_KEY (and QORO_API_KEY for --backend cloud) in shinka_network/.env.
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
3. {sampler_text} Each sample is repaired to the nearest valid topology and scored by the routing LP.
{scoring_text}

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

SCORING_PLAIN = """4. Per run: score = 0.5 * ref / best + 0.5 * ref / median, where best and median are over the sampled topologies and ref is the best score known for that demand matrix (the proven optimum where one exists). The combined score is the mean over several demand matrices and/or seeds; 1.0 would mean every sample is optimal.
So the low-energy region of your QUBO must consist of topologies with a low maximum link load: the best sample matters, and so does the typical sample."""
SCORING_POST = """4. Per run the score has two halves (ref = best score known for that demand matrix):
   (a) sample quality = 0.5 * ref / best + 0.5 * ref / median, where best and median are over the sampled topologies;
   (b) head start = ref / best_after, where best_after is the best score that a FIXED classical post-processor reaches with {post} LP calls when it starts from your samples. It repairs the samples, LP-scores about 30 of them chosen by YOUR energy (your linear + quadratic terms, lowest first, skipping near-duplicates), then runs a local search (2-link exchanges (a->b),(c->d) => (a->d),(c->b), mostly around the most loaded links) in which it also ranks the candidate moves by YOUR energy before it spends LP calls on them.
   score = 0.5 * (a) + 0.5 * (b). The combined score is the mean over network10, 12, 13, 14, 15 and 16 (QOBLIB instances), three annealing seeds each; 1.0 would mean every sample is optimal. The evaluation is deterministic.
So your terms are used twice: to generate the samples, and as a free predictor that must rank valid topologies, and single 2-link exchanges between them, in the order of their true score. The typical sample, the best sample and the ranking all count. Every size counts equally, and after the run the best programs are checked on network17 to network24, so the terms must scale sensibly with n (number of couplings, relative weight of linear and quadratic terms, penalty weight)."""
# Measured 2026-11-22 with the final evaluator (suite small6, 500 LP calls for the fixed post-processor).
KNOWN_POST = """- With this evaluator the baseline scores 0.8692 (sample quality 0.797, head start 0.942). Its samples: best +11..+20% above the reference, median +33..+45%; after the fixed post-processor +2.3..+9.6%.
- The starting program (evolved earlier on network10 and network12 only) scores 0.8919 (0.826 / 0.958). It rewards a direct link by 0.75 t + 0.25 sqrt(t) (t = demand divided by the largest demand) and, for every demand i -> k, puts negative couplings on x_ij * x_jk for its 3-5 best two-hop paths, discounted at transit nodes j that are already busy. It has 212 couplings at n = 10 and 534 at n = 16, and 86-95% of its samples are valid before repair. Its samples: best +9..+17%, median +29..+37%; after the post-processor +1.5..+7.2%.
- Per size the starting program's sample quality is 0.82-0.83 everywhere, but its head start drops at the largest size: 0.964, 0.957, 0.956, 0.972, 0.963 for n = 10, 12, 13, 14, 15 and only 0.939 at n = 16 (the baseline has 0.931 there). Its energy stops being a useful guide as n grows; this is the clearest weakness.
- An older program with 2-4 times as many couplings of the same kind (480 at n = 10, 2128 at n = 16) scores the same, 0.8925 (0.829 / 0.956), but only 36-78% of its samples are valid before repair. More two-hop couplings did not help.
- In every program so far the MEDIAN sample is +26..+45% above the reference while the best is around +10%: the low-energy region is still full of poor networks. Raising the typical sample is worth as much as raising the best one.
- On network17 to network24 the starting program keeps its lead over the baseline in sample quality (0.845 against 0.818), but its best sample on network24 is still +14% above the best known.
- In the final pipeline the sampler is partitioned QAOA (the QUBO is split into communities of at most 12 variables by community detection on its coupling graph, each part is solved by QAOA, the parts are recombined), on instances up to n = 24. Programs evolved with simulated annealing kept their advantage there (network12: baseline 0.74, evolved 0.83-0.84), and compact, well-structured couplings keep more samples valid."""

SAMPLER_TEXT = {
    "sa": "Simulated annealing draws 64 low-energy samples from this QUBO.",
    "pce": (
        "Pauli Correlation Encoding (PCE), a variational quantum algorithm that encodes all variables on a few "
        "qubits (16 for n = 12), is optimised on this QUBO on Qoro's quantum simulator, and its 100 lowest-energy "
        "solutions are taken. PCE solutions rarely satisfy the degree rules on their own, so they lean on the repair step."
    ),
    "qaoa_partitioned": (
        "The QUBO is split into communities of at most 12 variables by community detection on its interaction "
        "graph; each part is solved by QAOA on Qoro's quantum simulator, the parts are recombined by beam search "
        "and polished by local search, and the 100 best combined solutions are taken. How your couplings group "
        "variables into communities therefore matters."
    ),
}


class CloudBoundRunner(ShinkaEvolveRunner):
    """ShinkaEvolveRunner without the one-evaluation-per-CPU-core cap.

    ShinkaEvolve assumes local evaluations are CPU-bound and clamps max_evaluation_jobs to the
    number of cores. Quantum evaluations on Qoro's cloud spend most of their time waiting on the
    cloud, so for those the requested number of evaluation jobs is granted as is.
    """

    def _validate_concurrency_settings(self, max_evaluation_jobs, max_proposal_jobs, max_db_workers, cpu_count):
        _, max_proposal_jobs, max_db_workers = super()._validate_concurrency_settings(
            1, max_proposal_jobs, max_db_workers, cpu_count
        )
        return max_evaluation_jobs, max_proposal_jobs, max_db_workers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="net10", help="instance suite, see problem.SUITES")
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--budget", type=float, default=5.0, help="max LLM spend in USD")
    parser.add_argument("--islands", type=int, default=4)
    parser.add_argument("--models", default="gpt-6-luna", help="comma-separated proposal models")
    parser.add_argument("--model-probs", default=None,
                        help="comma-separated fixed sampling probabilities for --models (default: adaptive UCB)")
    parser.add_argument("--helper-model", default="gpt-6-luna", help="model for novelty checks and meta-recommendations")
    parser.add_argument("--eval-jobs", type=int, default=4)
    parser.add_argument("--proposal-jobs", type=int, default=4)
    parser.add_argument("--init", default="initial.py", help="starting program")
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--sampler", choices=["sa", "pce", "qaoa_partitioned"], default="sa",
                        help="sa: local simulated annealing (evaluate.py); pce / qaoa_partitioned: Divi (evaluate_qoro.py)")
    parser.add_argument("--backend", choices=["cloud", "local"], default="cloud",
                        help="quantum samplers only: Qoro cloud or the local Maestro simulator")
    parser.add_argument("--max-runs", type=int, default=2,
                        help="quantum samplers only: instances of the suite per evaluation (the first k)")
    parser.add_argument("--parallel-instances", type=int, default=1,
                        help="quantum samplers only: instances sampled at once inside one evaluation")
    parser.add_argument("--parent-lambda", type=float, default=10.0,
                        help="sharpness of parent selection; lower is softer, for noisy scores")
    parser.add_argument("--eval-python", default=None, help="Python for evaluate_qoro.py (default: ../.venv)")
    parser.add_argument("--eval-time", default=None, help="time limit per evaluation, HH:MM:SS")
    parser.add_argument("--print-prompt", action="store_true", help="print the task message and exit")
    parser.add_argument("--post-budget", type=int, default=0,
                        help="sa only: LP calls of the frozen post-processor per instance (0: sample quality only)")
    parser.add_argument("--efforts", default="low,medium", help="reasoning efforts sampled per proposal")
    parser.add_argument("--max-tokens", type=int, default=32768, help="output token limit per proposal (incl. reasoning)")
    parser.add_argument("--inspirations", default="2,2", help="archive,top-k inspiration programs per prompt")
    parser.add_argument("--patch-probs", default="0.6,0.3,0.1", help="probabilities of diff,full,cross patches")
    args = parser.parse_args()

    if args.sampler == "sa":
        eval_program, eval_python = "evaluate.py", sys.executable
        eval_args = {"suite": args.suite, "post_budget": args.post_budget}
        eval_time = args.eval_time or "00:10:00"
    else:
        eval_program = "evaluate_qoro.py"
        eval_python = args.eval_python or str(HERE.parent / ".venv" / "bin" / "python")
        if not Path(eval_python).exists():
            sys.exit(f"{eval_python} not found: run `uv sync` in the repository root (the notebook environment).")
        eval_args = {"suite": args.suite, "routine": args.sampler, "backend": args.backend, "max_runs": args.max_runs,
                     "parallel": args.parallel_instances}
        eval_time = args.eval_time or "01:00:00"

    models = args.models.split(",")
    if args.model_probs:
        probs = [float(p) for p in args.model_probs.split(",")]
        if len(probs) != len(models) or abs(sum(probs) - 1.0) > 1e-9:
            sys.exit("--model-probs needs one probability per model, summing to 1")
        llm_selection, llm_selection_kwargs = "fixed", {"prior_probs": probs}
    else:
        llm_selection, llm_selection_kwargs = "ucb1", {"exploration_coef": 1.0, "cost_aware_coef": 0.5}

    results_dir = args.results_dir or f"results/{args.suite}_{time.strftime('%m%d_%H%M%S')}"
    post = args.post_budget if args.sampler == "sa" else 0
    task_sys_msg = TASK_SYS_MSG.replace("{sampler_text}", SAMPLER_TEXT[args.sampler]).replace(
        "{scoring_text}", SCORING_POST.replace("{post}", str(post)) if post else SCORING_PLAIN)
    if post:
        task_sys_msg = task_sys_msg.replace("\n\n## Directions worth exploring", "\n" + KNOWN_POST + "\n\n## Directions worth exploring").replace(
            "per demand matrix, the gap of the best and of the median sampled topology to the reference,",
            "per demand matrix, the gap of the best and of the median sampled topology to the reference, the gap after the fixed post-processor,").replace(
            "Use it to see what fails.", "It ends with both halves of the score per size. Use it to see what fails, and at which size.").replace(
            "- The relative scale of linear and quadratic terms and the penalty weight matter a lot.",
            "- The relative scale of linear and quadratic terms and the penalty weight matter a lot.\n"
            "- For the head-start half only energy DIFFERENCES between valid networks matter (all have 2n links). A 2-link exchange "
            "removes two links and adds two: your terms should make that energy difference follow the change of the largest link "
            "load, for example by penalising pairs of links that would force two heavy flows onto the same link or through the same node.")
    if args.print_prompt:
        print(task_sys_msg)
        return
    n_archive, n_topk = (int(x) for x in args.inspirations.split(","))
    evo_config = EvolutionConfig(
        task_sys_msg=task_sys_msg,
        patch_types=["diff", "full", "cross"],
        patch_type_probs=[float(x) for x in args.patch_probs.split(",")],
        num_generations=args.generations,
        max_api_costs=args.budget,
        max_patch_resamples=3,
        max_patch_attempts=1,
        job_type="local",
        language="python",
        llm_models=models,
        llm_kwargs={"temperatures": [0.0, 0.5, 1.0], "reasoning_efforts": args.efforts.split(","),
                    "max_tokens": args.max_tokens},
        llm_dynamic_selection=llm_selection,
        llm_dynamic_selection_kwargs=llm_selection_kwargs,
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
        num_archive_inspirations=n_archive,
        num_top_k_inspirations=n_topk,
        migration_interval=10,
        migration_rate=0.1,
        island_elitism=True,
        parent_selection_strategy="weighted",
        parent_selection_lambda=args.parent_lambda,
    )
    job_config = LocalJobConfig(
        eval_program_path=eval_program,
        extra_cmd_args=eval_args,
        time=eval_time,
        python_executable=eval_python,
        eval_verbose=False,
    )
    runner_cls = ShinkaEvolveRunner if args.sampler == "sa" else CloudBoundRunner
    runner = runner_cls(
        evo_config=evo_config,
        job_config=job_config,
        db_config=db_config,
        max_evaluation_jobs=args.eval_jobs,
        max_proposal_jobs=args.proposal_jobs,
        max_db_workers=2,
        verbose=True,
    )
    print(f"Suite: {args.suite} | sampler: {args.sampler} ({eval_program}, {eval_args}) | results: {HERE / results_dir}")
    runner.run()


if __name__ == "__main__":
    main()
