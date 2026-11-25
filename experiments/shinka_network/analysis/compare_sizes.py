"""Evolved surrogate vs qoblib_tools on several instances, with the notebook's pipeline.

    cd shinka_network
    uv run python analysis/compare_sizes.py --program results/net10_luna_run1/best/main.py

Per instance and seed: simulated annealing (300 reads) -> repair -> routing LP, as in notebook
section 1.3. With seed 1 the qoblib_tools baseline reproduces the notebook's 309750 on network12.
"""

import argparse

import numpy as np
from common import baseline_terms, load_surrogate, problem, qt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--program", required=True)
    parser.add_argument("--sizes", default="10,12,16,20,24")
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--reads", type=int, default=300)
    args = parser.parse_args()

    methods = {"qoblib_tools": baseline_terms, "evolved": load_surrogate(args.program)}
    seeds = [int(s) for s in args.seeds.split(",")]
    print(f"{'instance':9s} {'BKV':>7s} | {'method':12s} {'best per seed':>28s} {'mean best gap':>13s} {'median gap':>10s}")
    for n in (int(x) for x in args.sizes.split(",")):
        demand = qt.load_network_demand(n)
        bkv = problem.BEST_KNOWN[n]
        for name, terms_fn in methods.items():
            linear, quadratic, pw = terms_fn(demand)
            rows = [
                problem.score_terms(problem.Instance(f"network{n:02d}", demand, float(bkv), args.reads, s),
                                    linear, quadratic, pw)
                for s in seeds
            ]
            bests = [r["best"] for r in rows]
            print(f"network{n:02d} {bkv:7d} | {name:12s} {', '.join(f'{b:.0f}' for b in bests):>28s} "
                  f"{100 * (np.mean(bests) / bkv - 1):12.1f}% {np.mean([r['gap_median_pct'] for r in rows]):9.1f}%",
                  flush=True)


if __name__ == "__main__":
    main()
