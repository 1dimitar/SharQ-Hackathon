"""Refine the best QUBO-sampled topologies with an LP-scored local search.

    cd shinka_network
    uv run python analysis/lp_local_search.py --n 10 \
        --program results/net10_luna_run2_200gen/best/main.py \
        --run results/net10_luna_run2_200gen

Starting pool: the best topologies recorded during an evolution run, plus fresh annealing samples
of an evolved program. Each start gets a first-improvement 2-exchange descent,
(a->b),(c->d) => (a->d),(c->b), scored by the routing LP. Then iterated local search: random
2-exchange kicks followed by descent, keeping the result whenever it improves.
"""

import argparse
import glob
import json
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from common import load_surrogate, problem, qt


def descend(demand, start, seed):
    rng = np.random.default_rng(seed)
    score, arcs = problem.local_search(demand, start, rng, max_sweeps=1000)
    return score, tuple(sorted(arcs))


def kick(arcs, rng, moves):
    """Apply `moves` random valid 2-exchanges (degree-preserving)."""
    cur = set(arcs)
    for _ in range(moves):
        for _ in range(100):
            (a, b), (c, d) = rng.choice(sorted(cur), size=2, replace=False)
            a, b, c, d = int(a), int(b), int(c), int(d)
            if a != d and c != b and (a, d) not in cur and (c, b) not in cur:
                cur = (cur - {(a, b), (c, d)}) | {(a, d), (c, b)}
                break
    return tuple(sorted(cur))


def iterated(demand, start, start_score, seed, seconds, moves):
    rng = np.random.default_rng(seed)
    best_score, best = start_score, start
    end = time.time() + seconds
    while time.time() < end:
        score, arcs = problem.local_search(demand, kick(best, rng, moves), rng, max_sweeps=1000)
        if score < best_score - 1e-9:
            best_score, best = score, tuple(sorted(arcs))
    return best_score, best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--program", required=True, help="evolved program whose annealing samples seed the search")
    parser.add_argument("--run", default=None, help="evolution run whose recorded best topologies seed the search")
    parser.add_argument("--sa-seeds", type=int, default=30)
    parser.add_argument("--reads", type=int, default=300)
    parser.add_argument("--starts", type=int, default=40, help="number of distinct starting topologies")
    parser.add_argument("--ils-seconds", type=float, default=120.0, help="iterated local search time per worker")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", default=None, help="JSON file for the best topology")
    args = parser.parse_args()

    n = args.n
    demand = qt.load_network_demand(n)
    ref = problem.BEST_KNOWN[n]
    pool = {}

    if args.run:
        for f in glob.glob(f"{args.run}/gen_*/results/best_topologies.json"):
            for v in json.load(open(f)).values():
                pool[tuple(sorted(map(tuple, v["arcs"])))] = v["score"]
        print(f"{len(pool)} distinct topologies recorded during {args.run}")

    linear, quadratic, pw = load_surrogate(args.program)(demand)
    bqm, _, arcs = problem.build_bqm(demand, linear, quadratic, pw)
    from dwave.samplers import SimulatedAnnealingSampler

    for seed in range(args.sa_seeds):
        for sample in SimulatedAnnealingSampler().sample(bqm, num_reads=args.reads, seed=seed).samples():
            topo = tuple(sorted(qt.repair_topology(n, qt.decode_topology(sample, arcs), demand)))
            if topo not in pool:
                pool[topo] = problem.lp_score(demand, topo)
    ranked = sorted(pool.items(), key=lambda kv: kv[1])
    print(f"pool: {len(pool)} distinct topologies, best {ranked[0][1]:.0f} ({100 * (ranked[0][1] / ref - 1):+.2f}%)")

    starts = ranked[: args.starts]
    t0 = time.time()
    with ProcessPoolExecutor(args.workers) as ex:
        results = list(ex.map(descend, [demand] * len(starts), [t for t, _ in starts], range(len(starts))))
    print(f"\n2-exchange descent from the {len(starts)} best starts ({time.time() - t0:.0f}s):")
    for (_, s_start), (s_end, _) in sorted(zip(starts, results), key=lambda p: p[1][0])[:5]:
        print(f"   start {s_start:.0f} -> local optimum {s_end:.0f}")
    results.sort(key=lambda r: r[0])
    print(f"   best local optimum: {results[0][0]:.0f} ({100 * (results[0][0] / ref - 1):+.2f}%), "
          f"{len({r[1] for r in results})} distinct local optima")

    best_score, best = results[0]
    if best_score > ref + 1e-9:
        distinct = []
        for s, t in results:
            if t not in {d[1] for d in distinct}:
                distinct.append((s, t))
        seeds = (distinct * args.workers)[: args.workers]
        t0 = time.time()
        with ProcessPoolExecutor(args.workers) as ex:
            ils = list(ex.map(iterated, [demand] * len(seeds), [t for _, t in seeds], [s for s, _ in seeds],
                              range(100, 100 + len(seeds)), [args.ils_seconds] * len(seeds),
                              [2 + k % 3 for k in range(len(seeds))]))
        ils.sort(key=lambda r: r[0])
        print(f"\nIterated local search, {len(seeds)} workers x {args.ils_seconds:.0f}s ({time.time() - t0:.0f}s): "
              f"best {ils[0][0]:.0f} ({100 * (ils[0][0] / ref - 1):+.2f}%)")
        if ils[0][0] < best_score:
            best_score, best = ils[0]

    assert qt.is_valid_topology(n, best) and abs(problem.lp_score(demand, best) - best_score) < 1e-6
    print(f"\nBEST network{n:02d}: {best_score:.0f} vs best known {ref} ({100 * (best_score / ref - 1):+.2f}%)")
    print("arcs (1-indexed):", [(i + 1, j + 1) for i, j in best])
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"instance": f"network{n:02d}", "score": best_score, "best_known": ref,
                       "arcs": [list(a) for a in best]}, f, indent=1)
        print("saved to", args.out)


if __name__ == "__main__":
    main()
