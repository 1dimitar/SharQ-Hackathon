"""Deep comparison on network10: evolved surrogate vs qoblib_tools.

    cd shinka_network
    uv run python analysis/compare_network10.py --program results/net10_luna_run1/best/main.py

1. Coefficients each method produces.
2. Does lower surrogate energy mean a lower LP score? Spearman correlation on random valid
   topologies, and where QOBLIB's proven optimum ranks.
3. Annealing performance over many seeds (notebook settings) with a Mann-Whitney test.
4. Is the bottleneck the surrogate or the sampler? How many annealed topologies have lower
   energy than the optimum.
"""

import argparse

import numpy as np
from common import baseline_terms, energy, load_surrogate, problem, published_optimum, qt
from dwave.samplers import SimulatedAnnealingSampler
from scipy.stats import mannwhitneyu, spearmanr

N, OPT = 10, 210800.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--program", required=True)
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--reads", type=int, default=300)
    parser.add_argument("--random-topologies", type=int, default=3000)
    args = parser.parse_args()

    demand = qt.load_network_demand(N)
    opt_arcs = published_optimum(N)
    methods = {"qoblib_tools": baseline_terms(demand), "evolved": load_surrogate(args.program)(demand)}

    print("== 1. Coefficients")
    for name, (lin, quad, pw) in methods.items():
        lw = np.array([w for w in lin.values() if w != 0]) if lin else np.zeros(1)
        qw = np.array(list(quad.values()) or [0.0])
        print(f"{name:12s} linear: {len(lw)} nonzero in [{lw.min():.3f}, {lw.max():.3f}] | quadratic: {len(quad)} "
              f"terms in [{qw.min():.3f}, {qw.max():.3f}] | penalty weight {pw}")
    _, quad, _ = methods["evolved"]
    if quad:
        print("strongest evolved quadratic terms:")
        for (a, b), w in sorted(quad.items(), key=lambda kv: -abs(kv[1]))[:5]:
            print(f"   x{a[0] + 1},{a[1] + 1} * x{b[0] + 1},{b[1] + 1}: {w:+.3f}")

    print(f"\n== 2. Surrogate energy vs LP score ({args.random_topologies} random valid topologies)")
    rng = np.random.default_rng(0)
    rand = [tuple(sorted(qt.random_topology(N, rng))) for _ in range(args.random_topologies)]
    rand_lp = np.array([problem.lp_score(demand, t) for t in rand])
    good = rand_lp <= np.quantile(rand_lp, 0.10)
    for name, terms in methods.items():
        e = np.array([energy(terms, t) for t in rand])
        print(f"{name:12s} Spearman all {spearmanr(e, rand_lp)[0]:.3f} | best 10% {spearmanr(e[good], rand_lp[good])[0]:.3f}"
              f" | optimum has lower energy than {100 * np.mean(e > energy(terms, opt_arcs)):.1f}% of them")

    print(f"\n== 3. Annealing, {args.seeds} seeds x {args.reads} reads -> repair -> LP")
    bests = {}
    for name, (lin, quad, pw) in methods.items():
        rows = [problem.score_terms(problem.Instance("network10", demand, OPT, args.reads, s), lin, quad, pw)
                for s in range(args.seeds)]
        b = bests[name] = np.array([r["best"] for r in rows])
        med = np.array([r["median"] for r in rows])
        print(f"{name:12s} best per seed: mean {b.mean():.0f} ({100 * (b.mean() / OPT - 1):+.1f}%), min {b.min():.0f}, "
              f"max {b.max():.0f} | median-sample gap {100 * (med.mean() / OPT - 1):.1f}%")
    p = mannwhitneyu(bests["evolved"], bests["qoblib_tools"], alternative="less").pvalue
    print(f"Mann-Whitney, evolved best < qoblib_tools best: p = {p:.2g}")

    print("\n== 4. Sampler or surrogate? (1000 annealing reads)")
    for name, terms in methods.items():
        bqm, _, arcs = problem.build_bqm(demand, *terms)
        index = {a: v for v, a in enumerate(arcs)}
        x_opt = {v: 0 for v in range(len(arcs))} | {index[a]: 1 for a in opt_arcs}
        e_opt = bqm.energy(x_opt)
        ss = SimulatedAnnealingSampler().sample(bqm, num_reads=1000, seed=0)
        seen = {}
        for sample, e in zip(ss.samples(), ss.record.energy):
            arcs_s = qt.decode_topology(sample, arcs)
            if qt.is_valid_topology(N, arcs_s):
                seen[tuple(sorted(arcs_s))] = e
        below = [t for t, e in seen.items() if e < e_opt - 1e-9]
        lp = [problem.lp_score(demand, t) for t in below]
        print(f"{name:12s} {len(below)} of {len(seen)} distinct valid samples have lower energy than the optimum"
              + (f" (LP scores {min(lp):.0f}-{max(lp):.0f})" if lp else ""))


if __name__ == "__main__":
    main()
