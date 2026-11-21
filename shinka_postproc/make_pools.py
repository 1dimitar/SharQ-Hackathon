"""Sample QAOA candidate pools from the frozen Hamiltonian (run once, before the evolution).

    cd shinka_postproc
    ../.venv/bin/python make_pools.py --n 12 --seeds 0,1,2 --backend cloud

Runs in the repository's root .venv (it needs qoro-divi). For each seed: QUBO = frozen surrogate
(rescaled to max |coefficient| 1) + penalty_weight * degree penalty, then partitioned QAOA exactly
as in run 3 (blocks of <= 12 variables, p = 2, Monte Carlo optimiser 50 x 10, 10 000 shots, beam
search to the 100 best solutions). The raw solutions are saved BEFORE any repair, as link lists
that may break the degree rules: turning them into valid networks is the evolved part's job.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
warnings.filterwarnings("ignore")

import dimod  # noqa: E402
import numpy as np  # noqa: E402
import qoblib_tools as qt  # noqa: E402


def frozen_surrogate():
    spec = importlib.util.spec_from_file_location("hamiltonian", HERE / "hamiltonian.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.surrogate_terms


def build_qubo(demand: np.ndarray):
    """Frozen surrogate + degree penalty, variables in link order 0..n(n-1)-1, normalised."""
    linear, quadratic, penalty_weight = frozen_surrogate()(np.array(demand, dtype=float))
    arcs = qt.all_arcs(len(demand))
    index = {a: v for v, a in enumerate(arcs)}
    lin = np.zeros(len(arcs))
    quad: dict[tuple[int, int], float] = {}
    for a, w in linear.items():
        lin[index[tuple(map(int, a))]] += float(w)
    for (a, b), w in quadratic.items():
        u, v = index[tuple(map(int, a))], index[tuple(map(int, b))]
        if u == v:
            lin[u] += float(w)
        else:
            key = (min(u, v), max(u, v))
            quad[key] = quad.get(key, 0.0) + float(w)
    scale = max(np.abs(lin).max(), max((abs(w) for w in quad.values()), default=0.0), 1e-12)
    bqm = dimod.BinaryQuadraticModel({v: lin[v] / scale for v in range(len(arcs))}, {}, 0.0, dimod.BINARY)
    bqm.add_quadratic_from((u, v, w / scale) for (u, v), w in quad.items())
    _, penalty, _ = qt.build_network_qubo(demand)
    pen = penalty.copy()
    pen.scale(float(penalty_weight))
    bqm.update(pen)
    bqm.normalize()
    assert list(bqm.variables) == list(range(len(arcs)))
    return bqm, arcs


def sample_partitioned_qaoa(bqm, backend_kind: str):
    import hybrid
    from divi.backends import JobConfig, MaestroSimulator, QoroService
    from divi.qprog.ensembles import BeamSearchStrategy, PartitioningProgramEnsemble
    from divi.qprog.optimizers import MonteCarloOptimizer
    from divi.qprog.problems import BinaryOptimizationProblem, CommunityDecomposer, QUBOPartitioningConfig

    if backend_kind == "cloud":
        backend = QoroService(job_config=JobConfig(shots=10_000, simulator_cluster="qoro_maestro", force_sampling=False))
    else:
        backend = MaestroSimulator(shots=10_000, force_sampling=False)
    problem = BinaryOptimizationProblem(
        bqm,
        decomposer=CommunityDecomposer(QUBOPartitioningConfig(max_n_variables_per_cluster=12)),
        composer=hybrid.SplatComposer(),
        local_search=True,
    )
    ensemble = PartitioningProgramEnsemble(
        problem=problem, quantum_routine="qaoa", n_layers=2,
        optimizer=MonteCarloOptimizer(population_size=50), max_iterations=10, backend=backend,
    )
    ensemble.run()
    tops = ensemble.get_top_solutions(n=100, strategy=BeamSearchStrategy(beam_width=100, n_partition_candidates=10))
    return [(dict(zip(bqm.variables, sol)), float(energy)) for sol, energy in tops], ensemble.total_circuit_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=12)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--backend", choices=["cloud", "local"], default="cloud")
    args = parser.parse_args()

    demand = qt.load_network_demand(args.n)
    bqm, arcs = build_qubo(demand)
    for seed in (int(s) for s in args.seeds.split(",")):
        out = HERE / "pools" / f"network{args.n:02d}_seed{seed}.json"
        if out.exists():
            print(f"{out.name} exists, skipping")
            continue
        np.random.seed(seed)
        start = time.time()
        for attempt in range(4):
            try:
                solutions, circuits = sample_partitioned_qaoa(bqm, args.backend)
                break
            except Exception as exc:  # noqa: BLE001 - network / cloud hiccups
                if attempt == 3:
                    raise
                print(f"seed {seed}: attempt {attempt + 1} failed ({type(exc).__name__}), retrying in 60 s", flush=True)
                time.sleep(60)
        candidates = [
            {"arcs": [list(arcs[v]) for v in range(len(arcs)) if int(x[v]) == 1], "qubo_energy": e}
            for x, e in solutions
        ]
        valid = sum(qt.is_valid_topology(args.n, [tuple(a) for a in c["arcs"]]) for c in candidates)
        out.write_text(json.dumps({
            "instance": f"network{args.n:02d}", "seed": seed, "backend": args.backend,
            "sampler": "partitioned QAOA (blocks <= 12 vars, p=2, MC 50x10, 10000 shots, beam 100)",
            "hamiltonian": "hamiltonian.py (run 3 gen 28)", "circuits": int(circuits),
            "seconds": round(time.time() - start, 1), "candidates": candidates,
        }))
        print(f"{out.name}: {len(candidates)} candidates, {valid} valid before repair, "
              f"{circuits} circuits, {time.time() - start:.0f} s", flush=True)


if __name__ == "__main__":
    main()
