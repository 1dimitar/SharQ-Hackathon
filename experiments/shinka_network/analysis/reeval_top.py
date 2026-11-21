"""Re-evaluate Hamiltonian programs with many partitioned-QAOA runs (local Maestro simulator, free).

    cd shinka_network
    ../.venv/bin/python analysis/reeval_top.py --runs 6 --programs \
        results/net12_qaoa_cloud_run3/gen_0/main.py,results/net12_qaoa_cloud_run3/gen_109/main.py

Self-contained (does not import problem.py / evaluate_qoro.py) but reproduces run 3's scoring:
QUBO = surrogate rescaled to max |coefficient| 1 + penalty_weight * degree penalty, variables in link
order, normalised; partitioned QAOA (blocks <= 12 variables, p = 2, Monte Carlo 50 x 10, 10 000
shots, beam search to the 100 best); repair; routing LP; score = 0.5*ref/best + 0.5*ref/median.
Runs in the repository's root .venv (qoro-divi).
"""

import argparse
import importlib.util
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE.parent))
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402

N, REF = 12, 276474.0


def build_qubo(program: str):
    import dimod
    import qoblib_tools as qt

    spec = importlib.util.spec_from_file_location("prog", HERE / program)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    demand = qt.load_network_demand(N)
    linear, quadratic, pw = module.surrogate_terms(np.array(demand, dtype=float))
    arcs = qt.all_arcs(N)
    index = {a: v for v, a in enumerate(arcs)}
    lin = np.zeros(len(arcs))
    quad = {}
    for a, w in linear.items():
        lin[index[(int(a[0]), int(a[1]))]] += float(w)
    for (a, b), w in quadratic.items():
        u, v = index[(int(a[0]), int(a[1]))], index[(int(b[0]), int(b[1]))]
        if u == v:
            lin[u] += float(w)
        else:
            key = (min(u, v), max(u, v))
            quad[key] = quad.get(key, 0.0) + float(w)
    quad = {k: w for k, w in quad.items() if w != 0.0}
    scale = max(np.abs(lin).max(), max((abs(w) for w in quad.values()), default=0.0), 1e-12)
    bqm = dimod.BinaryQuadraticModel({v: lin[v] / scale for v in range(len(arcs))}, {}, 0.0, dimod.BINARY)
    bqm.add_quadratic_from((u, v, w / scale) for (u, v), w in quad.items())
    _, penalty, _ = qt.build_network_qubo(demand)
    pen = penalty.copy()
    pen.scale(float(pw))
    bqm.update(pen)
    bqm.normalize()
    assert list(bqm.variables) == list(range(len(arcs)))
    return bqm, arcs, demand, len(quad)


def one_run(job):
    program, run = job
    import hybrid
    import qoblib_tools as qt
    from divi.backends import MaestroSimulator
    from divi.qprog.ensembles import BeamSearchStrategy, PartitioningProgramEnsemble
    from divi.qprog.optimizers import MonteCarloOptimizer
    from divi.qprog.problems import BinaryOptimizationProblem, CommunityDecomposer, QUBOPartitioningConfig

    np.random.seed(1000 + run)
    bqm, arcs, demand, n_quad = build_qubo(program)
    problem = BinaryOptimizationProblem(
        bqm, decomposer=CommunityDecomposer(QUBOPartitioningConfig(max_n_variables_per_cluster=12)),
        composer=hybrid.SplatComposer(), local_search=True,
    )
    ensemble = PartitioningProgramEnsemble(
        problem=problem, quantum_routine="qaoa", n_layers=2,
        optimizer=MonteCarloOptimizer(population_size=50), max_iterations=10,
        backend=MaestroSimulator(shots=10_000, force_sampling=False),
    )
    ensemble.run()
    tops = ensemble.get_top_solutions(n=100, strategy=BeamSearchStrategy(beam_width=100, n_partition_candidates=10))
    cache, scores = {}, []
    for sol, _ in tops:
        raw = qt.decode_topology(dict(zip(bqm.variables, sol)), arcs)
        topo = tuple(sorted(qt.repair_topology(N, raw, demand)))
        if topo not in cache:
            cache[topo] = qt.route_min_congestion(demand, topo).objective
        scores.append(cache[topo])
    best, median = min(scores), float(np.median(scores))
    return program, run, 0.5 * REF / best + 0.5 * REF / median, best, median, n_quad


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--programs", required=True)
    parser.add_argument("--runs", type=int, default=6)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    programs = args.programs.split(",")
    jobs = [(p, r) for p in programs for r in range(args.runs)]
    with ProcessPoolExecutor(args.workers) as ex:
        results = list(ex.map(one_run, jobs))
    print(f"{'program':48s} {'quad':>5s} {'score mean ± SD':>17s} {'(min..max)':>16s} {'best topo':>10s} {'median gap':>10s}")
    for p in programs:
        rows = [r for r in results if r[0] == p]
        s = np.array([r[2] for r in rows])
        print(f"{p:48s} {rows[0][5]:5d} {s.mean():9.4f} ± {s.std(ddof=1):.4f} ({s.min():.4f}..{s.max():.4f}) "
              f"{min(r[3] for r in rows):10.0f} {100 * (np.mean([r[4] for r in rows]) / REF - 1):9.1f}%")


if __name__ == "__main__":
    main()
