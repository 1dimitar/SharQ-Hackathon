"""Quantum evaluator: samples the evolved QUBO with Divi (PCE or partitioned QAOA) on Qoro.

    ../.venv/bin/python evaluate_qoro.py --program_path initial.py --results_dir results/q \
        --suite net12 --routine pce --backend cloud

Runs under the notebook environment (the repository's root .venv, which has qoro-divi) and does
not import ShinkaEvolve: the two need incompatible versions of `rich`. It writes metrics.json and
then correct.json, the same contract as shinka.core.run_shinka_eval.

Per instance: evolved terms -> QUBO (+ fixed degree penalty) -> normalise -> PCE / partitioned
QAOA / simulated annealing -> top solutions -> repair -> routing LP. Scored exactly like
evaluate.py: combined_score = mean of 0.5*ref/best + 0.5*ref/median (1.0 = perfect).
--backend cloud sends every circuit to Qoro's qoro_maestro simulator (uses credits); --backend
local runs the same circuits on the local Maestro simulator (free, for testing).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import signal
import sys
import time
import traceback
from pathlib import Path

import problem  # puts the repository root on sys.path


class SurrogateTimeout(Exception):
    pass


def _alarm(signum, frame):
    raise SurrogateTimeout()


def load_program(path: str):
    spec = importlib.util.spec_from_file_location("program", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["program"] = module
    spec.loader.exec_module(module)
    return module


def in_arc_order(bqm, n_vars: int):
    """Copy of bqm whose variable order is 0..n_vars-1. PCE returns bitstrings in variable order,
    and dimod puts variables that first appear in quadratic terms ahead of the rest."""
    import dimod

    ordered = dimod.BinaryQuadraticModel(
        {v: bqm.get_linear(v) for v in range(n_vars)}, {}, bqm.offset, dimod.BINARY
    )
    ordered.add_quadratic_from((u, v, w) for (u, v), w in bqm.quadratic.items())
    return ordered


def make_backend(kind: str, shots: int):
    from divi.backends import JobConfig, MaestroSimulator, QoroService

    if kind == "cloud":
        return QoroService(job_config=JobConfig(shots=shots, simulator_cluster="qoro_maestro", force_sampling=False))
    return MaestroSimulator(shots=shots, force_sampling=False)


def sample_pce(bqm, backend, seed: int, args) -> tuple[list, int]:
    """PCE on the whole QUBO (notebook section 1.3): 132 variables on 16 qubits for network12."""
    from divi.qprog.algorithms import PCE, PolyEncoding, SoftObjective
    from divi.qprog.optimizers import MonteCarloOptimizer
    from divi.qprog.problems import BinaryOptimizationProblem

    pce = PCE(
        BinaryOptimizationProblem(bqm),
        encoding=PolyEncoding(),
        objective=SoftObjective(alpha=2.0),
        n_layers=args.layers,
        optimizer=MonteCarloOptimizer(population_size=args.mc_population),
        max_iterations=args.mc_iterations,
        backend=backend,
        seed=seed,
    )
    pce.run()
    tops = pce.get_top_solutions(n=args.top_n, min_prob=0.0, sort_by="energy")
    return [s.bitstring for s in tops], pce.total_circuit_count


def sample_qaoa_partitioned(bqm, backend, seed: int, args) -> tuple[list, int]:
    """Partitioned QAOA (notebook's solve_partitioned + beam search)."""
    import hybrid
    from divi.qprog.ensembles import BeamSearchStrategy, PartitioningProgramEnsemble
    from divi.qprog.optimizers import MonteCarloOptimizer
    from divi.qprog.problems import BinaryOptimizationProblem, CommunityDecomposer, QUBOPartitioningConfig

    bop = BinaryOptimizationProblem(
        bqm,
        decomposer=CommunityDecomposer(QUBOPartitioningConfig(max_n_variables_per_cluster=args.cluster_size)),
        composer=hybrid.SplatComposer(),
        local_search=True,
    )
    ensemble = PartitioningProgramEnsemble(
        problem=bop,
        quantum_routine="qaoa",
        n_layers=args.layers,
        optimizer=MonteCarloOptimizer(population_size=args.mc_population),
        max_iterations=args.mc_iterations,
        backend=backend,
    )
    ensemble.run()
    tops = ensemble.get_top_solutions(
        n=args.top_n, strategy=BeamSearchStrategy(beam_width=args.top_n, n_partition_candidates=10)
    )
    return [dict(zip(bqm.variables, solution)) for solution, _ in tops], ensemble.total_circuit_count


def evaluate(args) -> tuple[dict, bool, str | None]:
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    error = problem.forbidden_use(args.program_path)
    if error:
        return {}, False, error
    try:
        module = load_program(args.program_path)
    except Exception as exc:  # noqa: BLE001 - any failure of the evolved code is a failed candidate
        return {}, False, f"program failed to load: {type(exc).__name__}: {exc}"

    instances = problem.load_suite(args.suite)[: args.max_runs]
    backend = None if args.routine == "sa" else make_backend(args.backend, args.shots)
    sampler = {"pce": sample_pce, "qaoa_partitioned": sample_qaoa_partitioned}.get(args.routine)
    rows, built_seconds = [], []
    signal.signal(signal.SIGALRM, _alarm)
    for inst in instances:
        signal.alarm(int(problem.SECONDS_PER_INSTANCE) + 5)
        try:
            result = module.run_surrogate(inst.demand)
        except SurrogateTimeout:
            return {}, False, f"surrogate_terms exceeded {problem.SECONDS_PER_INSTANCE:.0f}s for n={len(inst.demand)}"
        except Exception as exc:  # noqa: BLE001
            return {}, False, f"surrogate_terms raised {type(exc).__name__}: {exc}"
        finally:
            signal.alarm(0)
        error = problem.check_result(result)
        if error:
            return {}, False, error
        built_seconds.append(result["seconds"])

        if args.routine == "sa":
            row = problem.score_terms(inst, result["linear"], result["quadratic"], result["penalty_weight"])
            row.update(circuits=0, quantum_seconds=0.0)
        else:
            bqm, surrogate, arcs = problem.build_bqm(
                inst.demand, result["linear"], result["quadratic"], result["penalty_weight"]
            )
            bqm = in_arc_order(bqm, len(arcs))
            bqm.normalize()
            assert list(bqm.variables) == list(range(len(arcs))), "variable order must match arcs"
            start = time.perf_counter()
            for attempt in range(2):  # one retry for transient cloud failures
                try:
                    samples, circuits = sampler(bqm, backend, inst.seed, args)
                    break
                except Exception as exc:  # noqa: BLE001
                    if attempt == 1:
                        return {}, False, f"quantum sampling failed ({args.backend}): {type(exc).__name__}: {exc}"
                    time.sleep(10)
            row = problem.score_samples(inst, samples, arcs, surrogate.num_interactions)
            row.update(circuits=int(circuits), quantum_seconds=time.perf_counter() - start)
        rows.append(row)

    lines = [
        f"{r['name']}: best {r['best']:.0f} vs reference {r['ref']:.0f} (gap {r['gap_best_pct']:+.1f}%), "
        f"median gap {r['gap_median_pct']:+.1f}%, {r['feasible_rate']:.0%} of {r['num_samples']} "
        f"{args.routine} solutions valid before repair, {r['distinct']} distinct topologies, "
        f"{r['num_quadratic']} quadratic terms, {r['circuits']} circuits"
        for r in rows
    ]
    worst = max(rows, key=lambda r: r["gap_best_pct"])
    lines.append(f"Largest gap: {worst['name']} ({worst['gap_best_pct']:+.1f}%).")
    mean = lambda key: float(sum(r[key] for r in rows) / len(rows))  # noqa: E731
    (Path(args.results_dir) / "best_topologies.json").write_text(
        json.dumps({r["name"]: {"score": r["best"], "arcs": r["best_arcs"]} for r in rows})
    )
    metrics = {
        "combined_score": mean("score"),
        "public": {
            "mean_gap_best_pct": mean("gap_best_pct"),
            "mean_gap_median_pct": mean("gap_median_pct"),
            "mean_feasible_rate": mean("feasible_rate"),
            "mean_quadratic_terms": mean("num_quadratic"),
            "routine": args.routine,
            "backend": args.backend,
        },
        "private": {
            "circuits": int(sum(r["circuits"] for r in rows)),
            "quantum_seconds": float(sum(r["quantum_seconds"] for r in rows)),
            "build_seconds": float(sum(built_seconds)),
            **{r["name"]: {k: r[k] for k in ("best", "median", "ref")} for r in rows},
        },
        "text_feedback": "\n".join(lines),
    }
    return metrics, True, None


def write_results(results_dir: str, metrics: dict, correct: bool, error: str | None) -> None:
    """Same files as shinka.core.wrap_eval.save_json_results: metrics first, correct.json last."""
    out = Path(results_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "correct.json").unlink(missing_ok=True)
    if not correct:
        metrics = {"combined_score": 0.0, "public": {}, "private": {}, "text_feedback": f"INVALID: {error}"}
    for name, payload in (("metrics.json", metrics), ("correct.json", {"correct": correct, "error": error})):
        tmp = out / f".{name}.tmp"
        tmp.write_text(json.dumps(payload, indent=4, default=str))
        os.replace(tmp, out / name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--program_path", default="initial.py")
    parser.add_argument("--results_dir", default="results/manual_qoro")
    parser.add_argument("--suite", default="net12")
    parser.add_argument("--max_runs", type=int, default=1, help="use the first k seeds of the suite")
    parser.add_argument("--routine", choices=["pce", "qaoa_partitioned", "sa"], default="pce")
    parser.add_argument("--backend", choices=["cloud", "local"], default="cloud")
    parser.add_argument("--shots", type=int, default=10_000)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--mc_population", type=int, default=50)
    parser.add_argument("--mc_iterations", type=int, default=10)
    parser.add_argument("--top_n", type=int, default=100)
    parser.add_argument("--cluster_size", type=int, default=12)
    args = parser.parse_args()

    start = time.perf_counter()
    try:
        metrics, correct, error = evaluate(args)
    except Exception as exc:  # noqa: BLE001
        metrics, correct, error = {}, False, f"evaluator crashed: {type(exc).__name__}: {exc}"
        traceback.print_exc()
    write_results(args.results_dir, metrics, correct, error)
    print(f"correct: {correct} | error: {error} | {time.perf_counter() - start:.0f}s")
    print("combined_score:", metrics.get("combined_score", 0.0))
    if metrics.get("text_feedback"):
        print(metrics["text_feedback"])


if __name__ == "__main__":
    main()
