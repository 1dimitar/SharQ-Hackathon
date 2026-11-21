"""Strong reference values for the subset instances of a suite (run once before evolving on it).

    cd shinka_network && uv run python analysis/subset_refs.py --suite net12mix --ils-seconds 90

Per subset instance: 2-exchange descent from random valid topologies, then iterated local search
from the best local optima, all scored by the routing LP. The best value goes into
cache/toy_refs.json under the instance name (kept if it is lower than what is cached), which is
where problem.load_suite reads the reference from. Freeze the references before a run starts:
changing them later changes every stored score.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from common import problem, qt
from lp_local_search import descend, iterated


def reference(name: str, demand: np.ndarray, starts: int, workers: int, ils_seconds: float):
    n = len(demand)
    rng = np.random.default_rng(12345)
    random_starts = [tuple(sorted(qt.random_topology(n, rng))) for _ in range(starts)]
    with ProcessPoolExecutor(workers) as ex:
        optima = list(ex.map(descend, [demand] * starts, random_starts, range(starts)))
    optima.sort(key=lambda r: r[0])
    distinct = list(dict((t, s) for s, t in optima).items())[:workers]  # best distinct local optima
    seeds = (distinct * workers)[:workers]
    with ProcessPoolExecutor(workers) as ex:
        ils = list(ex.map(iterated, [demand] * workers, [t for t, _ in seeds], [s for _, s in seeds],
                          range(100, 100 + workers), [ils_seconds] * workers,
                          [2 + k % 3 for k in range(workers)]))
    best_score, best = min(ils + optima, key=lambda r: r[0])
    assert qt.is_valid_topology(n, best) and abs(problem.lp_score(demand, best) - best_score) < 1e-6
    return optima[0][0], best_score, best


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--suite", default="net12mix")
    p.add_argument("--starts", type=int, default=16, help="random starts for the first descent")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--ils-seconds", type=float, default=90.0, help="iterated local search time per worker")
    args = p.parse_args()

    path = problem.CACHE_DIR / "toy_refs.json"
    problem.CACHE_DIR.mkdir(exist_ok=True)
    out = {}
    for kind, n, seed, _ in problem.SUITES[args.suite]:
        if kind != "sub":
            continue
        demand, nodes = problem.subset_demand(n, seed)
        name = f"sub{n}_s{seed}"
        t0 = time.time()
        descent, best_score, best = reference(name, demand, args.starts, args.workers, args.ils_seconds)
        refs = json.loads(path.read_text()) if path.exists() else {}
        old = refs.get(name)
        refs[name] = min(best_score, old) if old is not None else best_score
        path.write_text(json.dumps(refs, indent=1, sort_keys=True))
        out[name] = {"nodes": nodes, "ref": refs[name], "best_descent": descent, "arcs": [list(a) for a in best]}
        print(f"{name} nodes {nodes}: best descent {descent:.0f}, iterated LS {best_score:.0f} "
              f"({100 * (descent / best_score - 1):+.2f}% gained), cached ref {refs[name]:.0f} "
              f"[{time.time() - t0:.0f}s]", flush=True)
    (problem.CACHE_DIR / f"{args.suite}_refs_detail.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
