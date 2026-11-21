"""Run a post-processing program with a large LP budget on chosen pools and save the networks.

    cd shinka_postproc
    uv run python analysis/big_budget.py --program init_run1_gen49.py --budget 20000 \
        --pools network13_r3g128_seed0.json --out-dir results/net13_gen49_20k

Each finished run is written at once to <out-dir>/<pool>/best_network.json (the format of the
evolution's results, so `leaderboard.py scan/watch <out-dir>` imports it) and to <out-dir>/summary.json.
"""
import argparse
import importlib.util
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
BEST_KNOWN = {12: 276474, 13: 300620, 14: 343091, 15: 370870, 16: 397908}


def one(job):
    program, pool, budget, seed = job
    sys.path.insert(0, str(HERE))
    spec = importlib.util.spec_from_file_location("prog", HERE / program)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.run_postproc(str(HERE / "pools" / pool), budget, seed=seed, time_limit=3600)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--program", required=True)
    parser.add_argument("--pools", required=True)
    parser.add_argument("--budget", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(args.program, p, args.budget, args.seed) for p in args.pools.split(",")]
    results = []
    with ProcessPoolExecutor(min(args.workers, len(jobs))) as ex:
        for fut in as_completed([ex.submit(one, j) for j in jobs]):
            r = fut.result()
            r["program"], r["gap_pct"] = args.program, 100 * (r["best"] / BEST_KNOWN[r["n"]] - 1)
            print(f"{Path(r['pool']).stem}: {r['best']:.0f} ({r['gap_pct']:+.2f}%), {r['lp_calls']} LP calls, "
                  f"{r['seconds']:.0f}s, error={r['error']}", flush=True)
            if r["arcs"]:
                (out / Path(r["pool"]).stem).mkdir(exist_ok=True)
                (out / Path(r["pool"]).stem / "best_network.json").write_text(json.dumps(r))
            results.append(r)
            (out / "summary.json").write_text(json.dumps(sorted(results, key=lambda r: r["best"]), indent=1))


if __name__ == "__main__":
    main()
