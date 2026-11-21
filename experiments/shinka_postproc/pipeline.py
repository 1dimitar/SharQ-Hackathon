#!/usr/bin/env python3
"""Hamiltonian -> partitioned QAOA (one cloud job at a time) -> evolved post-processor -> leaderboard.

    cd shinka_postproc
    ../.venv/bin/python pipeline.py --sizes 11-24 --hamiltonian-tag r3g128 --cluster-size 20 \
        --cluster qoro_gpu --postproc init_run1_gen49.py --budget 10000

Runs in the repository's root .venv (qoro-divi). Jobs are the instances of --sizes x --seeds. Before
each job the instance with the largest current gap to QOBLIB's best known value (read from the
repository leaderboard) goes first. A job:
  1. samples a QAOA pool with the given setup, unless the pool file exists:
     pools/network<NN>_<tag>_<setup>_seed<k>.json, the make_pools.py format (raw solutions before
     repair, plus the Hamiltonian tag), so the post-processing evolution can use it too;
  2. submits the pool's best repaired candidate to the leaderboard (method qaoa_partitioned);
  3. hands the pool to a local post-processing worker: the evolved `improve` (--postproc) with
     --budget LP calls; its network goes to results/pipeline/<pool>__<program>_<budget>/ and to the
     leaderboard (method "postproc (QAOA pool + evolved LP search)").
Sampling is strictly sequential (one Qoro job at a time); post-processing runs on --pp-workers local
processes while the next pool is sampled. The Hamiltonian is pools/hamiltonians/<tag>.py; with
--hamiltonian FILE it is copied there first (make_pools.py's convention).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import shutil
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for p in (str(HERE), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

import leaderboard  # noqa: E402
import qoblib_tools as qt  # noqa: E402
from make_pools import build_qubo  # noqa: E402

POSTPROC_METHOD = "postproc (QAOA pool + evolved LP search)"  # same name leaderboard.scan uses


def log(msg: str) -> None:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def setup_name(a) -> str:
    """Short pool-name tag of the QAOA setup, e.g. c20p2 (blocks <= 20 variables, p = 2)."""
    s = f"c{a.cluster_size}p{a.layers}"
    if (a.mc_population, a.mc_iterations) != (50, 10):
        s += f"m{a.mc_population}x{a.mc_iterations}"
    return s + ("loc" if a.backend == "local" else "")


# --------------------------------------------------------------------------- #
# 1. QAOA pool (the only step that uses Qoro)
# --------------------------------------------------------------------------- #
def qoro_key() -> str | None:
    """QORO_API_KEY from the environment or a .env file (this folder, the repository, shinka_network)."""
    if os.environ.get("QORO_API_KEY"):
        return os.environ["QORO_API_KEY"]
    from dotenv import dotenv_values

    for env in (HERE / ".env", REPO / ".env", REPO / "shinka_network" / ".env"):
        if env.exists() and dotenv_values(env).get("QORO_API_KEY"):
            return dotenv_values(env)["QORO_API_KEY"]
    return None


def sample_pool(bqm, a, seed: int):
    import hybrid
    from divi.backends import JobConfig, MaestroSimulator, QoroService
    from divi.qprog.ensembles import BeamSearchStrategy, PartitioningProgramEnsemble
    from divi.qprog.optimizers import MonteCarloOptimizer
    from divi.qprog.problems import BinaryOptimizationProblem, CommunityDecomposer, QUBOPartitioningConfig

    if a.backend == "cloud":
        from divi.backends.runners import _qoro

        # Divi uploads up to 3000 circuits per request with a 100 s timeout; one stalled request fails
        # the whole ensemble. Smaller requests get through a busy uplink (read at call time).
        _qoro._MAX_CIRCUITS_PER_REQUEST = a.upload_chunk
        backend = QoroService(auth_token=qoro_key(),
                              job_config=JobConfig(shots=a.shots, simulator_cluster=a.cluster, force_sampling=False))
    else:
        backend = MaestroSimulator(shots=a.shots, force_sampling=False)
    problem = BinaryOptimizationProblem(
        bqm,
        decomposer=CommunityDecomposer(QUBOPartitioningConfig(max_n_variables_per_cluster=a.cluster_size)),
        composer=hybrid.SplatComposer(),
        local_search=True,
    )
    ensemble = PartitioningProgramEnsemble(
        problem=problem, quantum_routine="qaoa", n_layers=a.layers,
        optimizer=MonteCarloOptimizer(population_size=a.mc_population), max_iterations=a.mc_iterations,
        backend=backend, seed=seed,
    )
    ensemble.run()
    tops = ensemble.get_top_solutions(
        n=a.states, strategy=BeamSearchStrategy(beam_width=max(100, a.states), n_partition_candidates=10))
    return [(dict(zip(bqm.variables, sol)), float(e)) for sol, e in tops], ensemble.total_circuit_count


def make_pool(n: int, seed: int, a) -> Path | None:
    """Sample (or reuse) the pool for one instance; submit its best repaired candidate. None on failure."""
    out = HERE / "pools" / f"network{n:02d}_{a.hamiltonian_tag}_{setup_name(a)}_seed{seed}.json"
    if out.exists():
        log(f"{out.name} exists: reusing it")
        return out
    demand = qt.load_network_demand(n)
    bqm, arcs = build_qubo(demand, HERE / "pools" / "hamiltonians" / f"{a.hamiltonian_tag}.py")
    np.random.seed(seed)
    where = f"{a.backend}/{a.cluster}" if a.backend == "cloud" else "local Maestro"
    log(f"network{n:02d} seed {seed}: sampling partitioned QAOA ({setup_name(a)}, {where}), {len(arcs)} variables")
    start = time.time()
    waits = [60, 120, 240, 480]
    for attempt in range(len(waits) + 1):
        try:
            solutions, circuits = sample_pool(bqm, a, seed)
            break
        except Exception as exc:  # noqa: BLE001 - network / cloud trouble
            if attempt == len(waits):
                log(f"network{n:02d} seed {seed}: sampling failed {attempt + 1} times, skipping ({type(exc).__name__}: {exc})")
                return None
            log(f"network{n:02d} seed {seed}: attempt {attempt + 1} failed ({type(exc).__name__}: {str(exc)[:120]}); "
                f"retrying in {waits[attempt]}s")
            time.sleep(waits[attempt])
    candidates = [{"arcs": [list(arcs[v]) for v in range(len(arcs)) if int(x[v]) == 1], "qubo_energy": e}
                  for x, e in solutions]
    valid = sum(qt.is_valid_topology(n, [tuple(c) for c in cand["arcs"]]) for cand in candidates)
    record = {
        "instance": f"network{n:02d}", "seed": seed, "backend": a.backend,
        "sampler": f"partitioned QAOA (blocks <= {a.cluster_size} vars, p={a.layers}, MC {a.mc_population}x"
                   f"{a.mc_iterations}, {a.shots} shots, beam {max(100, a.states)}, top {a.states}"
                   f"{', ' + a.cluster if a.backend == 'cloud' else ''})",
        "setup": {k: getattr(a, k) for k in ("cluster_size", "layers", "mc_population", "mc_iterations", "shots",
                                             "states", "backend", "cluster")},
        "hamiltonian": f"{a.hamiltonian_tag} (pools/hamiltonians/{a.hamiltonian_tag}.py)",
        "hamiltonian_tag": a.hamiltonian_tag, "hamiltonian_file": f"hamiltonians/{a.hamiltonian_tag}.py",
        "circuits": int(circuits), "seconds": round(time.time() - start, 1), "candidates": candidates,
    }
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(record))
    tmp.replace(out)
    repaired = {tuple(sorted(qt.repair_topology(n, [tuple(c) for c in cand["arcs"]], demand))) for cand in candidates}
    best = min(repaired, key=lambda t: leaderboard.lp_score(n, t))
    log(f"{out.name}: {len(candidates)} candidates ({valid} valid before repair), {circuits} circuits, "
        f"{time.time() - start:.0f}s; best repaired {leaderboard.lp_score(n, best):.0f} "
        f"({100 * (leaderboard.lp_score(n, best) / leaderboard.BEST_KNOWN[n] - 1):+.2f}%)")
    for line in leaderboard.submit(n, best, "qaoa_partitioned", label=f"QAOA pool {out.stem}",
                                   source=f"shinka_postproc/pools/{out.name}"):
        log(line)
    return out


# --------------------------------------------------------------------------- #
# 2. Post-processing (local CPU, in worker processes)
# --------------------------------------------------------------------------- #
def postprocess(pool_name: str, program: str, budget: int, seed: int) -> dict:
    for p in (str(HERE), str(REPO)):
        if p not in sys.path:
            sys.path.insert(0, p)
    spec = importlib.util.spec_from_file_location("postproc_program", HERE / program)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    stem = Path(pool_name).stem
    log(f"post-processing {stem} with {program}, {budget} LP calls")
    r = module.run_postproc(str(HERE / "pools" / pool_name), budget, seed=seed, time_limit=3600)
    r["program"], r["budget"] = program, budget
    news = []
    if r["arcs"]:
        out = HERE / "results" / "pipeline" / f"{stem}__{Path(program).stem}_{budget}"
        out.mkdir(parents=True, exist_ok=True)
        (out / "best_network.json").write_text(json.dumps(r))
        news = leaderboard.submit(r["n"], r["arcs"], POSTPROC_METHOD,
                                  label=f"{Path(program).stem}, {budget} LP calls, pool {stem}",
                                  source=f"shinka_postproc/results/pipeline/{out.name}",
                                  details={"lp_calls": r["lp_calls"], "seconds": round(r["seconds"], 1)})
    gap = 100 * (r["best"] / leaderboard.BEST_KNOWN[r["n"]] - 1) if math.isfinite(r["best"]) else float("nan")
    log(f"post-processed {stem}: {r['best']:.0f} ({gap:+.2f}% vs best known), {r['lp_calls']} LP calls, "
        f"{r['seconds']:.0f}s, error={r['error']}")
    for line in news:
        log(line)
    return r


# --------------------------------------------------------------------------- #
def current_gaps() -> dict[int, float]:
    try:
        records = json.loads((leaderboard.ROOT / "records.json").read_text())["overall"]
        return {r["n"]: r["gap_pct"] for r in records.values()}
    except (OSError, ValueError, KeyError):
        return {}


def parse_list(text: str) -> list[int]:
    out = []
    for part in text.split(","):
        lo, _, hi = part.partition("-")
        out += list(range(int(lo), int(hi or lo) + 1))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sizes", default="11-24")
    p.add_argument("--seeds", default="0")
    p.add_argument("--hamiltonian-tag", default="r3g128", help="pools/hamiltonians/<tag>.py")
    p.add_argument("--hamiltonian", default=None, help="copy this program to pools/hamiltonians/<tag>.py first")
    p.add_argument("--cluster-size", type=int, default=20, help="QAOA block size = qubits per circuit")
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--mc-population", type=int, default=50)
    p.add_argument("--mc-iterations", type=int, default=10)
    p.add_argument("--shots", type=int, default=10_000)
    p.add_argument("--states", type=int, default=200, help="solutions kept per pool (beam search)")
    p.add_argument("--backend", choices=["cloud", "local"], default="cloud")
    p.add_argument("--cluster", choices=["qoro_gpu", "qoro_cpu", "qoro_maestro"], default="qoro_gpu")
    p.add_argument("--upload-chunk", type=int, default=200, help="circuits per upload request (Divi: 3000)")
    p.add_argument("--postproc", default="init_run1_gen49.py", help="post-processing program (in this folder)")
    p.add_argument("--budget", type=int, default=10_000, help="LP calls per post-processing run")
    p.add_argument("--pp-workers", type=int, default=1, help="local post-processing processes")
    a = p.parse_args()

    if a.hamiltonian:
        dest = HERE / "pools" / "hamiltonians" / f"{a.hamiltonian_tag}.py"
        src = Path(a.hamiltonian).resolve()
        if dest.exists() and dest.read_text() != src.read_text():
            sys.exit(f"{dest} exists with different content: pick another --hamiltonian-tag")
        if not dest.exists():
            shutil.copyfile(src, dest)
    if not (HERE / "pools" / "hamiltonians" / f"{a.hamiltonian_tag}.py").exists():
        sys.exit(f"no pools/hamiltonians/{a.hamiltonian_tag}.py (use --hamiltonian FILE)")

    todo = [(n, s) for n in parse_list(a.sizes) for s in parse_list(a.seeds)]
    log(f"pipeline: {len(todo)} jobs | Hamiltonian {a.hamiltonian_tag} | QAOA {setup_name(a)} on "
        f"{a.backend}{'/' + a.cluster if a.backend == 'cloud' else ''}, {a.states} states | post-processor "
        f"{a.postproc}, {a.budget} LP calls, {a.pp_workers} worker(s)")
    futures = []
    with ProcessPoolExecutor(a.pp_workers) as workers:
        while todo:
            gaps = current_gaps()
            todo.sort(key=lambda job: (-gaps.get(job[0], math.inf), job[1]))  # furthest from best known first
            n, seed = todo.pop(0)
            log(f"next: network{n:02d} seed {seed} (current gap {gaps.get(n, math.inf):+.2f}%); {len(todo)} left after it")
            try:
                pool = make_pool(n, seed, a)
            except Exception:  # noqa: BLE001 - one broken job must not stop the queue
                traceback.print_exc()
                pool = None
            if pool is not None:
                futures.append(workers.submit(postprocess, pool.name, a.postproc, a.budget, seed))
            for f in [f for f in futures if f.done()]:
                futures.remove(f)
                if f.exception():
                    log(f"post-processing failed: {f.exception()!r}")
        log("all pools sampled; waiting for post-processing")
        for f in futures:
            if f.exception():
                log(f"post-processing failed: {f.exception()!r}")
    log("pipeline done")


if __name__ == "__main__":
    main()
