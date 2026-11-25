"""Benchmark Qoro's cloud simulator clusters (qoro_gpu vs qoro_cpu vs qoro_maestro) on network12.

    cd shinka_network && ../.venv/bin/python analysis/benchmark_gpu.py \
        --clusters qoro_gpu qoro_cpu --routines pce qaoa_partitioned --repeats 1

Every cluster gets the same QUBO, seed and settings, so the circuits are identical. Per cell this
records wall-clock time and, from the job IDs the run submitted, the billed compute seconds and
credits (credit transactions are itemised per job as "<s>s @ <rate> credits/s"). Matching job IDs
keeps the numbers clean while other runs spend credits on the same account.
Results go to analysis/benchmark_gpu.json.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import evaluate_qoro as eq  # noqa: E402
import problem  # noqa: E402
from divi.backends import JobConfig, QoroService  # noqa: E402


class TrackingQoroService(QoroService):
    """QoroService that remembers the job IDs it submitted."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.job_ids: list[str] = []

    def submit_circuits(self, *a, **kw):
        result = super().submit_circuits(*a, **kw)
        if result.job_id:
            self.job_ids.append(result.job_id)
        return result


RATE = re.compile(r"([\d.]+)s @ ([\d.]+) credits/s")


def charges_for(service: QoroService, job_ids: list[str], timeout: float = 180.0) -> dict:
    """Sum the JOB_USAGE transactions of job_ids (waits for billing to catch up)."""
    wanted, found = set(job_ids), {}
    deadline = time.time() + timeout
    while time.time() < deadline:
        page = 1
        while page <= 40 and len(found) < len(wanted):  # newest first; ours are recent
            data = service.get_credit_transactions(page=page, page_size=100)
            for t in data["results"]:
                if t.get("job_id") in wanted and t["id"] not in found:
                    found[t["id"]] = t
            if not data.get("next"):
                break
            page += 1
        if {t["job_id"] for t in found.values()} >= wanted:
            break
        time.sleep(10)
    seconds, rates = 0.0, set()
    for t in found.values():
        m = RATE.search(t["description"] or "")
        if m:
            seconds += float(m.group(1))
            rates.add(float(m.group(2)))
    return {
        "jobs": len(wanted),
        "jobs_billed": len({t["job_id"] for t in found.values()}),
        "credits": round(-sum(float(t["amount"]) for t in found.values()), 4),
        "compute_seconds": round(seconds, 3),
        "rate_credits_per_s": sorted(rates),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--program_path", default="results/net12_qaoa_cloud_run3/best/main.py")
    p.add_argument("--clusters", nargs="+", default=["qoro_gpu", "qoro_cpu", "qoro_maestro"])
    p.add_argument("--routines", nargs="+", default=["pce", "qaoa_partitioned"])
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--seed_index", type=int, default=0, help="which net12 instance")
    p.add_argument("--shots", type=int, default=10_000)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--mc_population", type=int, default=50)
    p.add_argument("--mc_iterations", type=int, default=10)
    p.add_argument("--top_n", type=int, default=100)
    p.add_argument("--cluster_size", type=int, default=12)
    p.add_argument("--out", default=str(HERE / "benchmark_gpu.json"))
    args = p.parse_args()

    inst = problem.load_suite("net12")[args.seed_index]
    module = eq.load_program(args.program_path)
    result = module.run_surrogate(inst.demand)
    bqm, surrogate, arcs = problem.build_bqm(
        inst.demand, result["linear"], result["quadratic"], result["penalty_weight"]
    )
    bqm = eq.in_arc_order(bqm, len(arcs))
    bqm.normalize()
    samplers = {"pce": eq.sample_pce, "qaoa_partitioned": eq.sample_qaoa_partitioned}

    out = Path(args.out)
    rows = json.loads(out.read_text()) if out.exists() else []
    for rep in range(args.repeats):
        for routine in args.routines:
            for cluster in args.clusters:  # interleave clusters so load drift hits all of them
                service = TrackingQoroService(
                    job_config=JobConfig(shots=args.shots, simulator_cluster=cluster, force_sampling=False),
                    use_saved_defaults=False,
                )
                start = time.perf_counter()
                try:
                    samples, circuits = samplers[routine](bqm, service, inst.seed, args)
                    error = None
                except Exception as exc:  # noqa: BLE001
                    samples, circuits, error = [], 0, f"{type(exc).__name__}: {exc}"
                wall = time.perf_counter() - start
                row = {"cluster": cluster, "routine": routine, "repeat": rep, "instance": inst.name,
                       "wall_seconds": round(wall, 2), "circuits": int(circuits), "error": error}
                if samples:
                    s = problem.score_samples(inst, samples, arcs, surrogate.num_interactions)
                    row.update(best=s["best"], ref=s["ref"], gap_best_pct=round(s["gap_best_pct"], 2),
                               score=round(s["score"], 4))
                row.update(charges_for(service, service.job_ids))
                rows.append(row)
                out.write_text(json.dumps(rows, indent=1))
                print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
