"""Test post-processing programs on held-out pools at several LP budgets.

    cd shinka_postproc
    uv run python analysis/heldout.py --programs initial.py,results/postproc_net12_run1/best/main.py \
        --pools network12_seed3.json,network12_seed4.json --budgets 2000,10000
"""
import argparse
import importlib.util
import itertools
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
BEST_KNOWN = {12: 276474, 13: 300620, 14: 343091, 15: 370870, 16: 397908}


def one(job):
    program, pool, budget, control = job
    spec = importlib.util.spec_from_file_location("prog", HERE / program)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    r = module.run_postproc(str(HERE / "pools" / pool), budget, seed=0, control=control, time_limit=3600)
    return job, r


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--programs", required=True)
    parser.add_argument("--pools", required=True)
    parser.add_argument("--budgets", default="2000,10000")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    jobs = list(itertools.product(args.programs.split(","), args.pools.split(","),
                                  [int(b) for b in args.budgets.split(",")], [False, True]))
    with ProcessPoolExecutor(args.workers) as ex:
        results = list(ex.map(one, jobs))
    print(f"{'program':48s} {'pool':20s} {'LP budget':>9s} {'QAOA starts':>18s} {'random starts':>18s}")
    by = {job: r for job, r in results}
    for program, pool, budget in itertools.product(args.programs.split(","), args.pools.split(","),
                                                   [int(b) for b in args.budgets.split(",")]):
        q, c = by[(program, pool, budget, False)], by[(program, pool, budget, True)]
        ref = BEST_KNOWN[q["n"]]
        fmt = lambda r: f"{r['best']:.0f} ({100 * (r['best'] / ref - 1):+.2f}%)" + ("!" if r["error"] else "")  # noqa: E731
        print(f"{program:48s} {pool:20s} {budget:9d} {fmt(q):>18s} {fmt(c):>18s}")


if __name__ == "__main__":
    main()
