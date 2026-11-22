"""ShinkaEvolve evaluator: scores a program's ``surrogate_terms`` on a suite of demand matrices.

    python evaluate.py --program_path initial.py --results_dir results/manual --suite toy

For every instance the program's terms become a QUBO (plus the fixed degree penalty),
simulated annealing proposes topologies, they are repaired and LP-scored.
combined_score = mean over instances of  0.5*ref/best + 0.5*ref/median   (1.0 = perfect).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from shinka.core import run_shinka_eval

import problem


def main(program_path: str, results_dir: str, suite: str, post_budget: int = 0) -> None:
    os.makedirs(results_dir, exist_ok=True)
    instances = problem.load_suite(suite)
    static_error = problem.forbidden_use(program_path)

    def get_kwargs(run_index: int) -> dict:
        return {"demand": instances[run_index].demand}

    def validate(result) -> tuple[bool, str | None]:
        error = static_error or problem.check_result(result)
        return (error is None), error

    def aggregate(results: list) -> dict:
        errors = [msg for ok, msg in map(validate, results) if not ok]
        if errors or len(results) != len(instances):
            reason = errors[0] if errors else "surrogate_terms crashed on at least one instance"
            return {"combined_score": 0.0, "public": {}, "private": {}, "text_feedback": f"INVALID: {reason}"}
        rows = [
            problem.score_terms(inst, r["linear"], r["quadratic"], r["penalty_weight"], post_budget)
            for inst, r in zip(instances, results)
        ]
        lines = []
        for row, r in zip(rows, results):
            lines.append(
                f"{row['name']}: best {row['best']:.0f} vs reference {row['ref']:.0f} "
                f"(gap {row['gap_best_pct']:+.1f}%), median gap {row['gap_median_pct']:+.1f}%, "
                f"{row['feasible_rate']:.0%} of samples valid before repair, "
                f"{row['distinct']} distinct topologies, {row['num_quadratic']} quadratic terms, "
                f"built in {r['seconds']:.2f}s"
                + (f" | after the fixed post-processor ({row['post_lp_calls']} LP calls): {row['post_best']:.0f} "
                   f"(gap {row['gap_post_pct']:+.2f}%)" if post_budget > 0 else "")
            )
        worst = max(rows, key=lambda row: row["gap_best_pct"])
        lines.append(f"Largest gap: {worst['name']} ({worst['gap_best_pct']:+.1f}%).")
        mean = lambda key: float(sum(row[key] for row in rows) / len(rows))  # noqa: E731
        post_public = {}
        if post_budget > 0:
            sizes = sorted({row["n"] for row in rows})
            per_size = lambda key, n: float(np.mean([row[key] for row in rows if row["n"] == n]))  # noqa: E731
            lines.append("Per size (sample-quality part / head-start part): " + ", ".join(
                f"n={n} {per_size('score_raw', n):.3f} / {per_size('score_post', n):.3f}" for n in sizes))
            lines.append(f"Sample-quality part {mean('score_raw'):.4f}, head-start part {mean('score_post'):.4f}; "
                         f"score = their mean = {mean('score'):.4f}.")
            post_public = {"mean_score_raw": mean("score_raw"), "mean_score_post": mean("score_post"),
                           "mean_gap_post_pct": mean("gap_post_pct"),
                           "mean_post_seconds": mean("post_seconds")}
        (Path(results_dir) / "best_topologies.json").write_text(
            json.dumps({row["name"]: {"score": row["best"], "arcs": row["best_arcs"]} for row in rows})
        )
        return {
            "combined_score": mean("score"),
            "public": {
                "mean_gap_best_pct": mean("gap_best_pct"),
                "mean_gap_median_pct": mean("gap_median_pct"),
                "mean_feasible_rate": mean("feasible_rate"),
                "mean_quadratic_terms": mean("num_quadratic"),
                **post_public,
            },
            "private": {row["name"]: {k: row[k] for k in ("best", "median", "ref", "post_best", "post_arcs") if k in row}
                        for row in rows},
            "text_feedback": "\n".join(lines),
        }

    metrics, correct, error = run_shinka_eval(
        program_path=program_path,
        results_dir=results_dir,
        experiment_fn_name="run_surrogate",
        num_runs=len(instances),
        get_experiment_kwargs=get_kwargs,
        validate_fn=validate,
        aggregate_metrics_fn=aggregate,
    )
    print("correct:", correct, "| error:", error)
    print("combined_score:", metrics.get("combined_score"))
    if metrics.get("text_feedback"):
        print(metrics["text_feedback"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--program_path", type=str, default="initial.py")
    parser.add_argument("--results_dir", type=str, default="results/manual")
    parser.add_argument("--suite", type=str, default="toy")
    parser.add_argument("--post_budget", type=int, default=0,
                        help="LP calls of the frozen post-processor per instance (0: sample quality only)")
    args = parser.parse_args()
    main(args.program_path, args.results_dir, args.suite, args.post_budget)
