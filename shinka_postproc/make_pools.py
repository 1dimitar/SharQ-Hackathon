"""Sample QAOA candidate pools from the frozen Hamiltonian (run once, before the evolution).

    cd shinka_postproc
    ../.venv/bin/python make_pools.py --n 12 --seeds 0,1,2 --backend cloud
    ../.venv/bin/python make_pools.py --n 13 --seeds 0-4 --backend local --states 200 \
        --hamiltonian ../shinka_network/results/net12_qaoa_cloud_run3/gen_128/main.py --tag r3g128

Runs in the repository's root .venv (it needs qoro-divi). For each seed: QUBO = frozen surrogate
(rescaled to max |coefficient| 1) + penalty_weight * degree penalty, then partitioned QAOA exactly
as in run 3 (blocks of <= 12 variables, p = 2, Monte Carlo optimiser 50 x 10, 10 000 shots, beam
search to the --states best solutions, default 100). The raw solutions are saved BEFORE any repair,
as link lists that may break the degree rules: turning them into valid networks is the evolved part's
job.

With --tag, the Hamiltonian file is copied to pools/hamiltonians/<tag>.py, the pools are named
network<NN>_<tag>_seed<k>.json and record "hamiltonian_file", so run_postproc scores tools.energy
with the Hamiltonian that produced each pool. Without --tag: hamiltonian.py, old file names.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
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


def frozen_surrogate(path: Path = HERE / "hamiltonian.py"):
    spec = importlib.util.spec_from_file_location("hamiltonian", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.surrogate_terms


def build_qubo(demand: np.ndarray, hamiltonian: Path = HERE / "hamiltonian.py"):
    """Frozen surrogate + degree penalty, variables in link order 0..n(n-1)-1, normalised."""
    linear, quadratic, penalty_weight = frozen_surrogate(hamiltonian)(np.array(demand, dtype=float))
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


def sample_partitioned_qaoa(bqm, backend_kind: str, states: int = 100):
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
    beam = max(100, states)
    tops = ensemble.get_top_solutions(n=states, strategy=BeamSearchStrategy(beam_width=beam, n_partition_candidates=10))
    return [(dict(zip(bqm.variables, sol)), float(energy)) for sol, energy in tops], ensemble.total_circuit_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=12)
    parser.add_argument("--seeds", default="0,1,2", help="comma list and/or ranges, e.g. 0-4,7")
    parser.add_argument("--backend", choices=["cloud", "local"], default="cloud")
    parser.add_argument("--states", type=int, default=100, help="solutions kept per pool (beam search)")
    parser.add_argument("--hamiltonian", default=None, help="program file with surrogate_terms (needs --tag)")
    parser.add_argument("--tag", default=None, help="short name of the Hamiltonian, used in file names")
    args = parser.parse_args()
    if bool(args.hamiltonian) != bool(args.tag):
        sys.exit("--hamiltonian and --tag go together")

    seeds = []
    for part in args.seeds.split(","):
        lo, _, hi = part.partition("-")
        seeds.extend(range(int(lo), int(hi or lo) + 1))
    hamiltonian_file, hamiltonian_label = HERE / "hamiltonian.py", "hamiltonian.py (run 3 gen 28)"
    if args.tag:
        hamiltonian_file = HERE / "pools" / "hamiltonians" / f"{args.tag}.py"
        hamiltonian_file.parent.mkdir(parents=True, exist_ok=True)
        source = Path(args.hamiltonian).resolve()
        if hamiltonian_file.exists() and hamiltonian_file.read_text() != source.read_text():
            sys.exit(f"{hamiltonian_file} exists with different content: pick another --tag")
        shutil.copyfile(source, hamiltonian_file)
        hamiltonian_label = f"{args.tag} = {source.relative_to(HERE.parent)}"

    demand = qt.load_network_demand(args.n)
    bqm, arcs = build_qubo(demand, hamiltonian_file)
    for seed in seeds:
        name = f"network{args.n:02d}_{args.tag}_seed{seed}.json" if args.tag else f"network{args.n:02d}_seed{seed}.json"
        out = HERE / "pools" / name
        if out.exists():
            print(f"{out.name} exists, skipping")
            continue
        np.random.seed(seed)
        start = time.time()
        for attempt in range(4):
            try:
                solutions, circuits = sample_partitioned_qaoa(bqm, args.backend, args.states)
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
        record = {
            "instance": f"network{args.n:02d}", "seed": seed, "backend": args.backend,
            "sampler": f"partitioned QAOA (blocks <= 12 vars, p=2, MC 50x10, 10000 shots, "
                       f"beam {max(100, args.states)}, top {args.states})",
            "hamiltonian": hamiltonian_label, "circuits": int(circuits),
            "seconds": round(time.time() - start, 1), "candidates": candidates,
        }
        if args.tag:
            record["hamiltonian_tag"] = args.tag
            record["hamiltonian_file"] = f"hamiltonians/{args.tag}.py"  # relative to the pool's folder
        out.write_text(json.dumps(record))
        print(f"{out.name}: {len(candidates)} candidates, {valid} valid before repair, "
              f"{circuits} circuits, {time.time() - start:.0f} s", flush=True)


if __name__ == "__main__":
    main()
