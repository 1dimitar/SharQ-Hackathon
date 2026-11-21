"""ShinkaEvolve evaluator for the post-processing step.

    python evaluate.py --program_path initial.py --results_dir results/manual

Each evaluation runs the program's `improve` on every training pool (raw QAOA output of the frozen
Hamiltonian) with a fixed LP budget, and once more per pool with random valid networks in place
of the QAOA candidates (control, reported but not scored).
combined_score = mean over QAOA pools of best_known / best_found  (1.0 = best known reached).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from shinka.core import run_shinka_eval

HERE = Path(__file__).resolve().parent
BEST_KNOWN = {10: 210800, 12: 276474, 13: 300620, 14: 343091, 15: 370870, 16: 397908}
SECONDS_PER_RUN = 150.0

# The program must reach the LP only through `tools`, and must not tamper with the oracle.
FORBIDDEN = [
    r"scipy", r"linprog", r"\bmilp\b", r"highs", r"qoblib", r"\boracle\b", r"tools\._", r"__dict__",
    r"getattr", r"setattr", r"globals\(", r"locals\(", r"\binspect\b", r"\bgc\b", r"\bsys\b", r"\bos\b",
    r"\bopen\(", r"subprocess", r"importlib", r"__import__", r"\beval\(", r"\bexec\(", r"pickle",
    r"socket", r"requests", r"urllib", r"multiprocessing", r"concurrent", r"threading", r"\bjson\b",
]  # fmt: skip


def forbidden_use(program_path: str) -> str | None:
    src = Path(program_path).read_text()
    m = re.search(r"EVOLVE-BLOCK-START(.*?)EVOLVE-BLOCK-END", src, flags=re.S)
    code = m.group(1) if m else src
    code = re.sub(r'""".*?"""', "", code, flags=re.S)
    code = re.sub(r"#.*", "", code)
    for pattern in FORBIDDEN:
        if re.search(pattern, code):
            return f"forbidden construct in the evolved code: {pattern}"
    return None


def main(program_path: str, results_dir: str, pools: str, lp_budget: int) -> None:
    pool_paths = [str(HERE / "pools" / p) for p in pools.split(",")]
    runs = [(p, False) for p in pool_paths] + [(p, True) for p in pool_paths]
    static_error = forbidden_use(program_path)

    def get_kwargs(i: int) -> dict:
        return {"pool_path": runs[i][0], "lp_budget": lp_budget, "seed": i % len(pool_paths), "control": runs[i][1]}

    def validate(r) -> tuple[bool, str | None]:
        if static_error:
            return False, static_error
        if not isinstance(r, dict) or "best" not in r:
            return False, "run_postproc must return the dict built by the fixed wrapper"
        if r["error"]:
            return False, f"improve raised {r['error']}"
        if r["seconds"] > SECONDS_PER_RUN:
            return False, f"one run took {r['seconds']:.0f}s (limit {SECONDS_PER_RUN:.0f}s)"
        if r["arcs"] is None or not r["best"] < float("inf"):
            return False, "no valid network was scored"
        return True, None

    def aggregate(results: list) -> dict:
        errors = [msg for ok, msg in map(validate, results) if not ok]
        if errors or len(results) != len(runs):
            reason = errors[0] if errors else "a run crashed"
            return {"combined_score": 0.0, "public": {}, "private": {}, "text_feedback": f"INVALID: {reason}"}
        qaoa = [r for r in results if not r["control"]]
        ctrl = [r for r in results if r["control"]]
        ratio = lambda r: BEST_KNOWN[r["n"]] / r["best"]  # noqa: E731
        lines = []
        for q, c in zip(qaoa, ctrl):
            ref = BEST_KNOWN[q["n"]]
            lines.append(
                f"{Path(q['pool']).stem}: best {q['best']:.0f} vs best known {ref} (gap {100 * (q['best'] / ref - 1):+.2f}%), "
                f"{q['lp_calls']} LP calls, {q['seconds']:.1f}s | same program from random starts: {c['best']:.0f} "
                f"(gap {100 * (c['best'] / ref - 1):+.2f}%)"
            )
        mean = lambda xs: float(sum(xs) / len(xs))  # noqa: E731
        q_score, c_score = mean([ratio(r) for r in qaoa]), mean([ratio(r) for r in ctrl])
        lines.append(f"Mean best_known/best: QAOA starts {q_score:.4f}, random starts {c_score:.4f}.")
        best_run = min(qaoa, key=lambda r: r["best"])
        (Path(results_dir) / "best_network.json").write_text(json.dumps(best_run))
        return {
            "combined_score": q_score,
            "public": {
                "mean_gap_pct": mean([100 * (r["best"] / BEST_KNOWN[r["n"]] - 1) for r in qaoa]),
                "control_score": c_score,
                "qaoa_minus_control": q_score - c_score,
                "mean_lp_calls": mean([r["lp_calls"] for r in qaoa]),
                "mean_seconds": mean([r["seconds"] for r in qaoa]),
            },
            "private": {Path(r["pool"]).stem + ("_control" if r["control"] else ""): r["best"] for r in results},
            "text_feedback": "\n".join(lines),
        }

    metrics, correct, error = run_shinka_eval(
        program_path=program_path,
        results_dir=results_dir,
        experiment_fn_name="run_postproc",
        num_runs=len(runs),
        get_experiment_kwargs=get_kwargs,
        validate_fn=validate,
        aggregate_metrics_fn=aggregate,
        run_workers=3,
    )
    print("correct:", correct, "| error:", error)
    print("combined_score:", metrics.get("combined_score"))
    if metrics.get("text_feedback"):
        print(metrics["text_feedback"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--program_path", default="initial.py")
    parser.add_argument("--results_dir", default="results/manual")
    parser.add_argument("--pools", default="network12_seed0.json,network12_seed1.json,network12_seed2.json")
    parser.add_argument("--lp_budget", type=int, default=2000)
    args = parser.parse_args()
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    main(args.program_path, args.results_dir, args.pools, args.lp_budget)
