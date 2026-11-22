"""ShinkaEvolve evaluator for the post-processing step.

    python evaluate.py --program_path initial.py --results_dir results/manual

Each evaluation runs the program's `improve` on every training pool (raw QAOA output of a frozen
Hamiltonian) with a fixed LP budget, and once more on the first --controls pools with random valid
networks in place of the QAOA candidates (control, reported but not scored; default every pool).
--score mean (default, run 1): combined_score = mean over pools of best_known / best_found.
--score best_per_hamiltonian (run 2): for each Hamiltonian the best result over its pools
(independent QAOA runs), then the mean of best_known / best over Hamiltonians.
1.0 = best known reached; above 1.0 = best known beaten.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from shinka.core import run_shinka_eval
from shinka.core.wrap_eval import save_json_results

HERE = Path(__file__).resolve().parent
BEST_KNOWN = {10: 210800, 12: 276474, 13: 300620, 14: 343091, 15: 370870, 16: 397908, 17: 437000, 18: 473375,
              19: 492924, 20: 528528, 21: 569273, 22: 618135, 23: 659369, 24: 663688}
SECONDS_PER_RUN = 150.0

# The program must reach the LP only through `tools`, and must not tamper with the oracle.
FORBIDDEN = [
    r"scipy", r"linprog", r"\bmilp\b", r"highs", r"qoblib", r"\boracle\b", r"tools\._", r"__dict__",
    r"getattr", r"setattr", r"globals\(", r"locals\(", r"\binspect\b", r"\bgc\b", r"\bsys\b", r"\bos\b",
    r"\bopen\(", r"subprocess", r"importlib", r"__import__", r"\beval\(", r"\bexec\(", r"pickle",
    r"socket", r"requests", r"urllib", r"multiprocessing", r"concurrent", r"threading", r"\bjson\b",
    r"\bsignal\b", r"\bvars\(", r"__class__", r"__setattr__", r"__getattribute__", r"\btype\(tools", r"\bobject\.",
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


def main(program_path: str, results_dir: str, pools: str, lp_budget: int, controls: int = -1,
         run_workers: int = 3, score: str = "mean") -> None:
    pool_paths = [str(HERE / "pools" / p) for p in pools.split(",")]
    # Hamiltonian that sampled each pool (pools from make_pools.py --tag); older pools share hamiltonian.py
    tags = {p: json.loads(Path(p).read_text()).get("hamiltonian_tag", "hamiltonian.py") for p in pool_paths}
    n_ctrl = len(pool_paths) if controls < 0 else min(controls, len(pool_paths))
    runs = [(p, False) for p in pool_paths] + [(p, True) for p in pool_paths[:n_ctrl]]
    static_error = forbidden_use(program_path)
    if static_error:  # never execute a program that breaks the rules
        save_json_results(results_dir, {"combined_score": 0.0, "public": {}, "private": {},
                                        "text_feedback": f"INVALID: {static_error}"}, False, static_error)
        print("correct: False | error:", static_error)
        return

    def get_kwargs(i: int) -> dict:
        return {"pool_path": runs[i][0], "lp_budget": lp_budget, "seed": i % len(pool_paths), "control": runs[i][1],
                "time_limit": SECONDS_PER_RUN}

    def validate(r) -> tuple[bool, str | None]:
        if static_error:
            return False, static_error
        if not isinstance(r, dict) or "best" not in r:
            return False, "run_postproc must return the dict built by the fixed wrapper"
        if r["error"]:
            return False, f"improve raised {r['error']}"
        if r.get("timed_out") or r["seconds"] > SECONDS_PER_RUN + 5:
            return False, f"a run hit the {SECONDS_PER_RUN:.0f}s wall-time limit (endless loop or too slow?)"
        if r["arcs"] is None or not r["best"] < float("inf"):
            return False, "no valid network was scored"
        return True, None

    def aggregate(results: list) -> dict:
        errors = [msg for ok, msg in map(validate, results) if not ok]
        if errors or len(results) != len(runs):
            reason = errors[0] if errors else "a run crashed"
            return {"combined_score": 0.0, "public": {}, "private": {}, "text_feedback": f"INVALID: {reason}"}
        qaoa = [r for r in results if not r["control"]]
        ctrl = {r["pool"]: r for r in results if r["control"]}
        ratio = lambda r: BEST_KNOWN[r["n"]] / r["best"]  # noqa: E731
        gap = lambda r: 100 * (r["best"] / BEST_KNOWN[r["n"]] - 1)  # noqa: E731
        mean = lambda xs: float(sum(xs) / len(xs))  # noqa: E731
        lines = []
        for q in qaoa:
            line = (f"{Path(q['pool']).stem}: best {q['best']:.0f} vs best known {BEST_KNOWN[q['n']]} "
                    f"(gap {gap(q):+.2f}%), {q['lp_calls']} LP calls, {q['seconds']:.1f}s")
            if q["pool"] in ctrl:
                c = ctrl[q["pool"]]
                line += f" | same program from random starts: {c['best']:.0f} (gap {gap(c):+.2f}%)"
            lines.append(line)
        groups: dict[str, list] = {}
        for q in qaoa:
            groups.setdefault(tags[q["pool"]], []).append(q)
        best_of = {t: min(rs, key=lambda r: r["best"]) for t, rs in groups.items()}
        group_score, pool_score = mean([ratio(b) for b in best_of.values()]), mean([ratio(r) for r in qaoa])
        combined = group_score if score == "best_per_hamiltonian" else pool_score
        sizes = sorted({q["n"] for q in qaoa})
        if len(sizes) > 1:  # multi-size runs: where the gap is
            lines.append("Mean gap per instance size: " + ", ".join(
                f"n={n} {mean([gap(q) for q in qaoa if q['n'] == n]):+.2f}%" for n in sizes))
        lines.append("Best per Hamiltonian (over its pools): "
                     + ", ".join(f"{t} {b['best']:.0f} ({gap(b):+.2f}%)" for t, b in best_of.items()))
        lines.append(f"Mean over Hamiltonians of best_known/best(best of its pools): {group_score:.4f}"
                     f"{' (this is the score)' if score == 'best_per_hamiltonian' else ''}; "
                     f"mean over all pools: {pool_score:.4f}{' (this is the score)' if score == 'mean' else ''}.")
        public = {
            "mean_gap_pct": mean([gap(r) for r in qaoa]),
            "best_gap_pct": min(gap(r) for r in qaoa),
            "best_per_hamiltonian_score": group_score,
            "mean_pool_score": pool_score,
            "mean_lp_calls": mean([r["lp_calls"] for r in qaoa]),
            "mean_seconds": mean([r["seconds"] for r in qaoa]),
        }
        public.update({f"mean_gap_pct_n{n}": mean([gap(q) for q in qaoa if q["n"] == n]) for n in sizes})
        if ctrl:
            c_score = mean([ratio(c) for c in ctrl.values()])
            matched = mean([ratio(q) for q in qaoa if q["pool"] in ctrl])
            lines.append(f"Random starts on {len(ctrl)} pools: {c_score:.4f} (same pools from QAOA starts: {matched:.4f}).")
            public.update({"control_score": c_score, "qaoa_minus_control": matched - c_score})
        best_run = min(qaoa, key=lambda r: r["best"])
        (Path(results_dir) / "best_network.json").write_text(json.dumps(best_run))
        return {
            "combined_score": combined,
            "public": public,
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
        run_workers=run_workers,
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
    parser.add_argument("--controls", type=int, default=-1,
                        help="random-start control runs on the first N pools (-1: every pool, 0: none)")
    parser.add_argument("--run_workers", type=int, default=3, help="runs in parallel within one evaluation")
    parser.add_argument("--score", choices=["mean", "best_per_hamiltonian"], default="mean",
                        help="mean over pools, or per Hamiltonian the best of its pools, then mean")
    args = parser.parse_args()
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    main(args.program_path, args.results_dir, args.pools, args.lp_budget, args.controls, args.run_workers, args.score)
