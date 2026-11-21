#!/usr/bin/env python3
"""Evaluate surrogate programs on network12-16 and feed the repository leaderboard.

    cd shinka_network
    ../.venv/bin/python eval_sizes.py watch results/net12mix_qaoa_cloud_run4     # each new best of a run
    ../.venv/bin/python eval_sizes.py eval results/<run>/best/main.py --label "run 3 best"
    ../.venv/bin/python eval_sizes.py eval initial.py --label baseline --sampler sa --repeats 4
    ../.venv/bin/python eval_sizes.py show

Runs under the repository's root .venv (the quantum samplers need qoro-divi). An evaluation runs the
pipeline of evaluate_qoro.py (QUBO, sampler, repair, routing LP) on the QOBLIB instances networkNN
for every size in --sizes, --repeats times each. The best topology of every instance run goes to
the repository leaderboard (../leaderboard.py -> ../leaderboard/LEADERBOARD.md) as soon as that run
finishes, so an evaluation that fails half-way or is stopped keeps what it found.
results/size_evals/ compares the programs:

  evaluations.jsonl  one line per evaluation: best and median LP score of every instance run
  PROGRAMS.md        one row per program and sampler: best gap per size, mean score
  programs/          every evaluated program, named by the hash of its code
  runs/<eval>/       raw metrics.json of every evaluation

`watch` polls a ShinkaEvolve run's programs.sqlite. It evaluates the run's best program each time a
new one takes the lead (a leader overtaken while an evaluation runs is skipped) and imports the
topologies the run itself finds into the leaderboard. It stops once the run's process
(results/<run>.pid) has exited and the last leader is evaluated.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sqlite3
import sys
import time
import traceback
from argparse import Namespace
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import problem  # noqa: E402  (puts the repository root on sys.path)

import leaderboard  # noqa: E402  (../leaderboard.py, the repository leaderboard)

OUT = HERE / "results" / "size_evals"
# Sampler settings of the evolution runs (the defaults of evaluate_qoro.py's command line).
QUANTUM_DEFAULTS = {"shots": 10_000, "layers": 2, "mc_population": 50, "mc_iterations": 10, "top_n": 100,
                    "cluster_size": 12}
RETRYABLE = "not caused by the program"  # evaluate_qoro's wording for cloud or network failures
BANDS = [(5, 16), (17, 24)]  # PROGRAMS.md averages the score per size band
MAX_RETRIES = 2  # watch: evaluations of one program that may fail on cloud trouble before it is skipped


def log(msg: str) -> None:
    print(f"[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def code_sha(code: str) -> str:
    return hashlib.sha256(code.encode()).hexdigest()[:12]


def load_log() -> list[dict]:
    path = OUT / "evaluations.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def evaluate_program(code: str, label: str, source: str, opts) -> dict:
    """Evaluate one program on networkNN for every size; log it and submit every topology found."""
    import evaluate_qoro  # imports divi lazily, only when a quantum sampler runs

    sha = code_sha(code)
    variant = settings_variant(opts)
    if variant:
        label = f"{label} [{variant}]"
    program = OUT / "programs" / f"{sha}.py"
    program.parent.mkdir(parents=True, exist_ok=True)
    program.write_text(code)
    eval_id = f"{dt.datetime.now():%Y%m%d-%H%M%S}-{sha}-{opts.sampler}"
    runs = [(n, r) for n in opts.sizes for r in range(opts.repeats)]
    suite = f"_sizes_{eval_id}"  # registered in this process only; problem.py is not edited
    problem.SUITES[suite] = [("real", n, r, opts.reads) for n, r in runs]
    run_dir = OUT / "runs" / eval_id
    args = Namespace(program_path=str(program), results_dir=str(run_dir), suite=suite, max_runs=len(runs),
                     parallel=opts.parallel, routine=opts.sampler, backend=opts.backend,
                     **{k: getattr(opts, k) for k in QUANTUM_DEFAULTS})

    # evaluate_qoro.evaluate discards every instance once one fails; wrap its per-instance sampler so
    # each finished instance is kept and submitted to the leaderboard right away.
    finished: list[dict] = []
    sample_instance = evaluate_qoro.sample_instance

    def sample_and_submit(job, args, t0):
        row = sample_instance(job, args, t0)
        finished.append(row)
        try:
            for line in leaderboard.submit(row["n"], row["best_arcs"], method=opts.sampler, label=label,
                                           source=f"shinka_network/results/size_evals/runs/{eval_id}",
                                           details={"program": f"programs/{sha}.py", "backend": args.backend,
                                                    "cluster": opts.cluster, "instance_run": row["name"]}):
                log(line)
        except Exception as exc:  # noqa: BLE001 - the topology is still in evaluations.jsonl
            log(f"could not submit {row['name']} to the leaderboard: {type(exc).__name__}: {exc}")
        return row

    make_backend = evaluate_qoro.make_backend

    def make_backend_on_cluster(kind, shots):  # evaluate_qoro always asks for qoro_maestro
        if kind != "cloud" or opts.cluster == "qoro_maestro":
            return make_backend(kind, shots)
        from divi.backends import JobConfig, QoroService

        return QoroService(job_config=JobConfig(shots=shots, simulator_cluster=opts.cluster, force_sampling=False))

    t0 = time.perf_counter()
    evaluate_qoro.sample_instance = sample_and_submit  # evaluate() looks these names up at call time
    evaluate_qoro.make_backend = make_backend_on_cluster
    try:
        metrics, ok, error = evaluate_qoro.evaluate(args)
    except Exception as exc:  # noqa: BLE001 - an evaluator crash is logged as a failed evaluation
        metrics, ok, error = {}, False, f"evaluator crashed: {type(exc).__name__}: {exc}"
        traceback.print_exc()
    finally:
        evaluate_qoro.sample_instance = sample_instance
        evaluate_qoro.make_backend = make_backend
        del problem.SUITES[suite]
    evaluate_qoro.write_results(str(run_dir), metrics, ok, error)

    order = {f"network{n:02d}_s{r}": i for i, (n, r) in enumerate(runs)}
    instances = [{
        "instance": f"network{row['n']:02d}", "n": row["n"], "run": int(row["name"].rsplit("_s", 1)[1]),
        "bkv": row["ref"], "best": row["best"], "median": row["median"], "score": row["score"],
        "gap_best_pct": row["gap_best_pct"], "gap_median_pct": row["gap_median_pct"],
        "feasible_rate": row["feasible_rate"], "distinct": row["distinct"], "num_quadratic": row["num_quadratic"],
        "circuits": row["circuits"], "seconds": row["quantum_seconds"], "arcs": row["best_arcs"],
    } for row in sorted(finished, key=lambda row: order[row["name"]])]  # fmt: skip
    entry = {
        "eval_id": eval_id,
        "time": dt.datetime.now().isoformat(timespec="seconds"),
        "label": label,
        "source": source,
        "program_sha": sha,
        "program": f"programs/{sha}.py",
        "sampler": opts.sampler,
        "variant": variant,
        "backend": None if opts.sampler == "sa" else opts.backend,
        "config": {"reads": opts.reads} if opts.sampler == "sa" else
                  {**{k: getattr(opts, k) for k in QUANTUM_DEFAULTS}, "cluster": opts.cluster},
        "sizes": opts.sizes,
        "repeats": opts.repeats,
        "ok": ok,
        "error": error,
        "seconds": round(time.perf_counter() - t0, 1),
        "circuits": int(sum(r["circuits"] for r in instances)),
        "instances": instances,
    }
    with leaderboard.locked():  # one writer at a time for this folder too
        with open(OUT / "evaluations.jsonl", "a") as fh:
            fh.write(json.dumps(entry) + "\n")
        write_programs_md(load_log())
    return entry


def settings_variant(opts) -> str:
    """Circuit settings that differ from the evolution runs', e.g. "cluster_size=20" ("" if none)."""
    if opts.sampler == "sa":
        return "" if opts.reads == 64 else f"reads={opts.reads}"
    return ", ".join(f"{k}={getattr(opts, k)}" for k, v in QUANTUM_DEFAULTS.items() if getattr(opts, k) != v)


def report(entry: dict) -> None:
    status = "evaluated" if entry["ok"] else f"evaluation FAILED ({entry['error']}); finished runs kept:"
    log(f"{status} '{entry['label']}' with {entry['sampler']}, {entry['seconds']:.0f}s, {entry['circuits']} circuits")
    for r in entry["instances"]:
        print(f"    {r['instance']} run {r['run']}: best {r['best']:.0f} ({r['gap_best_pct']:+.2f}% vs BKV "
              f"{r['bkv']:.0f}), median {r['gap_median_pct']:+.1f}%, score {r['score']:.3f}", flush=True)


# --------------------------------------------------------------------------- #
# PROGRAMS.md
# --------------------------------------------------------------------------- #
def write_programs_md(evals: list[dict]) -> None:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for e in evals:
        if e["instances"]:
            groups[(e["program_sha"], e["sampler"], e.get("variant", ""))].append(e)
    sizes = sorted({r["n"] for es in groups.values() for e in es for r in e["instances"]})
    rows = []
    for (_, sampler, _), es in groups.items():
        by_n = defaultdict(list)
        for e in es:
            for r in e["instances"]:
                by_n[r["n"]].append(r)
        per_size = {n: sum(r["score"] for r in rs) / len(rs) for n, rs in by_n.items()}
        means = [sum(v for n, v in per_size.items() if lo <= n <= hi) / sum(lo <= n <= hi for n in per_size)
                 if any(lo <= n <= hi for n in per_size) else None for lo, hi in BANDS]
        gaps = {n: min(r["gap_best_pct"] for r in rs) for n, rs in by_n.items()}
        rows.append((means, es[0]["label"], sampler, {n: len(rs) for n, rs in by_n.items()}, gaps,
                     es[-1]["time"], es[0]["program"]))
    rows.sort(key=lambda row: [-(m if m is not None else -1.0) for m in row[0]])
    out = [
        "# Surrogate programs on network12-16",
        "",
        f"Updated {dt.datetime.now():%Y-%m-%d %H:%M} by `eval_sizes.py` from `evaluations.jsonl` "
        f"({len(evals)} evaluations); do not edit by hand. The best topologies also go to the repository "
        "leaderboard, [../../../leaderboard/LEADERBOARD.md](../../../leaderboard/LEADERBOARD.md).",
        "",
        "One row per program, sampler and circuit settings (non-default settings in brackets), highest mean score first. Per size: gap of the best topology over "
        "all runs to QOBLIB's best known value, with the number of runs. Mean score: 0.5 BKV/best + "
        "0.5 BKV/median per run (1.0 = every sampled topology optimal), averaged per size, then over the sizes "
        "of each band (sorted by the first band).",
        "",
        "| # | Program | Sampler | " + " | ".join(f"n{n}" for n in sizes) + " | "
        + " | ".join(f"Mean score n{lo}-{hi}" for lo, hi in BANDS) + " | Last run | Code |",
        "|---:|---|---|" + "---:|" * (len(sizes) + len(BANDS)) + "---|---|",
    ]
    for i, (means, label, sampler, counts, gaps, last, program) in enumerate(rows, 1):
        cells = " | ".join(f"{gaps[n]:+.2f}% ({counts[n]})" if n in gaps else "–" for n in sizes)
        mean_cells = " | ".join(f"{m:.4f}" if m is not None else "–" for m in means)
        out.append(f"| {i} | {label} | {sampler} | {cells} | {mean_cells} | {last[:16].replace('T', ' ')} "
                   f"| [{Path(program).name}]({program}) |")
    failed = [e for e in evals if not e["ok"]]
    if failed:
        out += ["", "## Incomplete evaluations (latest 10)", "", "| Time | Program | Sampler | Runs kept | Error |",
                "|---|---|---|---:|---|"]
        for e in failed[-10:]:
            error = (e["error"] or "").replace("|", "/").replace("\n", " ")[:160]
            out.append(f"| {e['time'][:16].replace('T', ' ')} | {e['label']} | {e['sampler']} | "
                       f"{len(e['instances'])}/{len(e['sizes']) * e['repeats']} | {error} |")
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / ".PROGRAMS.md.tmp"
    tmp.write_text("\n".join(out) + "\n")
    os.replace(tmp, OUT / "PROGRAMS.md")


# --------------------------------------------------------------------------- #
# Watching a ShinkaEvolve run
# --------------------------------------------------------------------------- #
def run_leader(db: Path):
    """(id, generation, combined_score, code) of the run's best correct program, or None."""
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=30)
    try:
        return con.execute(
            "SELECT id, generation, combined_score, code FROM programs WHERE correct = 1 "
            "ORDER BY combined_score DESC, timestamp ASC LIMIT 1"
        ).fetchone()
    finally:
        con.close()


def needs_evaluation(sha: str, opts) -> bool:
    """False once the program has a complete (or a non-retryable failed) evaluation with this sampler."""
    retries = 0
    for e in load_log():
        if e["program_sha"] != sha or e["sampler"] != opts.sampler or e.get("variant", "") != settings_variant(opts):
            continue
        if e["ok"] or RETRYABLE not in (e["error"] or ""):
            return False
        retries += 1
    return retries < MAX_RETRIES


def process_alive(pid_file: Path) -> bool:
    try:
        os.kill(int(pid_file.read_text().split()[0]), 0)
    except ProcessLookupError:
        return False
    except (OSError, ValueError, IndexError):
        return True  # no permission or unreadable pid file: keep watching
    return True


def import_run(run_dir: Path) -> None:
    """Topologies the run's own evaluations found (network instances only) -> leaderboard."""
    try:
        for line in leaderboard.scan([run_dir]):
            log(line)
    except Exception as exc:  # noqa: BLE001
        log(f"could not import {run_dir.name} into the leaderboard: {type(exc).__name__}: {exc}")


def watch(opts) -> None:
    run_dir = opts.run_dir
    db = run_dir / "programs.sqlite"
    pid_file = run_dir.parent / f"{run_dir.name}.pid"
    log(f"watching {db} every {opts.interval:.0f}s | sampler {opts.sampler}"
        f"{'' if opts.sampler == 'sa' else '/' + opts.backend + '/' + opts.cluster} | sizes {opts.sizes} x {opts.repeats} runs, "
        f"{opts.parallel} at once | stops when {'the pid in ' + pid_file.name + ' exits' if pid_file.exists() else 'killed'}")
    last_seen, final_pass = None, False
    while True:
        import_run(run_dir)
        try:
            leader = run_leader(db) if db.exists() else None
        except sqlite3.Error as exc:
            log(f"could not read the database ({exc}); retrying")
            leader = None
        if leader is not None:
            pid, gen, score, code = leader
            sha = code_sha(code)
            if needs_evaluation(sha, opts):
                label = f"{run_dir.name} gen {gen}"
                source = f"{run_dir.name}: program {pid}, generation {gen}, run score {score:.4f}"
                log(f"new leader {label} (run score {score:.4f}, code {sha}): evaluating")
                report(evaluate_program(code, label, source, opts))
                continue  # check right away: a better program may have appeared meanwhile
            if pid != last_seen:
                log(f"leader {run_dir.name} gen {gen} (run score {score:.4f}) is already evaluated")
                last_seen = pid
        if opts.once:
            return
        if pid_file.exists() and not process_alive(pid_file):
            if final_pass:
                log("the run has exited and its last leader is evaluated; stopping")
                return
            final_pass = True  # one more look, for a leader written just before the run exited
            continue
        time.sleep(opts.interval)


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #
def parse_sizes(text: str) -> list[int]:
    sizes = []
    for part in text.split(","):
        lo, _, hi = part.partition("-")
        sizes += list(range(int(lo), int(hi or lo) + 1))
    if not all(n in problem.BEST_KNOWN for n in sizes):
        raise argparse.ArgumentTypeError(f"sizes must be within {min(problem.BEST_KNOWN)}-{max(problem.BEST_KNOWN)}")
    return sizes


def main() -> None:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--sizes", type=parse_sizes, default=parse_sizes("12-16"), help="e.g. 12-16 or 12,14,16")
    common.add_argument("--repeats", type=int, default=2, help="runs per size (QAOA is noisy: use >= 2)")
    common.add_argument("--sampler", choices=["qaoa_partitioned", "pce", "sa"], default="qaoa_partitioned")
    common.add_argument("--backend", choices=["cloud", "local"], default="cloud", help="quantum samplers only")
    common.add_argument("--cluster", choices=["qoro_maestro", "qoro_cpu", "qoro_gpu"], default="qoro_maestro",
                        help="Qoro simulator cluster for --backend cloud (the evolution runs use qoro_maestro)")
    common.add_argument("--parallel", type=int, default=4,
                        help="instance runs sampled at once (keep low on the cloud: evolution runs share it)")
    common.add_argument("--reads", type=int, default=64, help="simulated annealing reads (sa only)")
    for key, value in QUANTUM_DEFAULTS.items():
        common.add_argument(f"--{key.replace('_', '-')}", type=int, default=value, help="quantum samplers only")

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p_eval = sub.add_parser("eval", parents=[common], help="evaluate one program file")
    p_eval.add_argument("program", type=Path)
    p_eval.add_argument("--label", default=None, help="name in PROGRAMS.md and the leaderboard (default: path)")
    p_watch = sub.add_parser("watch", parents=[common], help="evaluate every new best program of a run")
    p_watch.add_argument("run_dir", type=Path, help="ShinkaEvolve results folder (has programs.sqlite)")
    p_watch.add_argument("--interval", type=float, default=60.0, help="seconds between database polls")
    p_watch.add_argument("--once", action="store_true", help="evaluate the current leader if needed, then exit")
    sub.add_parser("show", help="rebuild PROGRAMS.md and print it")
    opts = parser.parse_args()

    for attr in ("program", "run_dir"):  # resolve user paths before moving to this folder
        if getattr(opts, attr, None) is not None:
            setattr(opts, attr, getattr(opts, attr).resolve())
    os.chdir(HERE)  # Divi reads QORO_API_KEY from ./.env

    if opts.command == "eval":
        code = opts.program.read_text()
        try:
            source = str(opts.program.relative_to(HERE))
        except ValueError:
            source = str(opts.program)
        label = opts.label or source
        where = "" if opts.sampler == "sa" else f" ({opts.backend}, {opts.cluster})"
        log(f"evaluating '{label}' (code {code_sha(code)}) with {opts.sampler}{where} on sizes {opts.sizes} "
            f"x {opts.repeats} runs, {opts.parallel} at once")
        report(evaluate_program(code, label, source, opts))
    elif opts.command == "watch":
        watch(opts)
    else:
        write_programs_md(load_log())
        print((OUT / "PROGRAMS.md").read_text())


if __name__ == "__main__":
    main()
