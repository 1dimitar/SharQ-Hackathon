"""How much do QAOA candidates help the post-processing, compared with random starts?

    cd shinka_postproc
    uv run python analysis/quantum_value.py --program results/postproc_net12_run1/best/main.py

1. Starting quality: LP scores of the repaired QAOA candidates vs random valid networks.
2. Diversity: distinct networks and mean pairwise link overlap.
3. Best score found at each LP budget, QAOA pools vs several random start sets.
"""

import argparse
import importlib.util
import itertools
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))  # repository root (qoblib_tools)
REF = 276474


def load(program):
    spec = importlib.util.spec_from_file_location("prog", HERE / program)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(job):
    program, pool, budget, control, seed = job
    r = load(program).run_postproc(str(HERE / "pools" / pool), budget, seed=seed, control=control, time_limit=3600)
    return job, r["best"]


def start_stats(pools):
    import oracle
    import qoblib_tools as qt

    demand = qt.load_network_demand(12)
    tools = oracle.Tools(demand, 10**9, ({}, {}))
    rows = []
    for name, nets in pools.items():
        repaired = [tuple(sorted(tools.repair(c))) for c in nets]
        distinct = sorted(set(repaired))
        scores = np.array([tools.lp(t) for t in distinct])
        sets = [set(t) for t in distinct]
        overlap = np.mean([len(a & b) / 24 for a, b in itertools.combinations(sets, 2)]) if len(sets) > 1 else 1.0
        rows.append((name, len(distinct), scores.min(), np.median(scores), overlap))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--program", required=True)
    parser.add_argument("--pools", default="network12_seed0.json,network12_seed1.json,network12_seed2.json,"
                                          "network12_seed3.json,network12_seed4.json")
    parser.add_argument("--random-sets", type=int, default=5)
    parser.add_argument("--budgets", default="50,100,200,500,1000,2000")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    pools = args.pools.split(",")
    budgets = [int(b) for b in args.budgets.split(",")]

    import qoblib_tools as qt

    starts = {p: [[tuple(a) for a in c["arcs"]] for c in json.load(open(HERE / "pools" / p))["candidates"]] for p in pools}
    for s in range(args.random_sets):
        rng = np.random.default_rng(10_000 + s)  # identical to run_postproc(control=True, seed=s)
        starts[f"random_set{s}"] = [qt.random_topology(12, rng) for _ in range(100)]
    print("== 1-2. Starting points after repair (before any search)")
    print(f"{'set':22s} {'distinct':>8s} {'best':>16s} {'median':>16s} {'mean link overlap':>18s}")
    for name, k, best, med, ov in start_stats(starts):
        print(f"{name:22s} {k:8d} {best:9.0f} ({100 * (best / REF - 1):+5.1f}%) {med:9.0f} ({100 * (med / REF - 1):+5.1f}%) {ov:17.0%}")

    jobs = [(args.program, p, b, False, 0) for p in pools for b in budgets]
    jobs += [(args.program, pools[0], b, True, s) for s in range(args.random_sets) for b in budgets]
    with ProcessPoolExecutor(args.workers) as ex:
        res = dict(ex.map(run, jobs))
    print(f"\n== 3. Best score vs LP budget ({args.program})")
    print(f"{'LP budget':>9s} | {'QAOA pools: mean gap (min..max)':>34s} | {'random sets: mean gap (min..max)':>34s}")
    for b in budgets:
        q = np.array([res[(args.program, p, b, False, 0)] for p in pools]) / REF - 1
        r = np.array([res[(args.program, pools[0], b, True, s)] for s in range(args.random_sets)]) / REF - 1
        print(f"{b:9d} | {100 * q.mean():+8.2f}% ({100 * q.min():+.2f}..{100 * q.max():+.2f}%)      "
              f"| {100 * r.mean():+8.2f}% ({100 * r.min():+.2f}..{100 * r.max():+.2f}%)")


if __name__ == "__main__":
    main()
