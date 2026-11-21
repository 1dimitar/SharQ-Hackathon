#!/usr/bin/env python3
"""Repo-wide leaderboard for QOBLIB 08 network design: the best topology found so far for every
instance, by any method.

    uv run python leaderboard.py show       # rebuild and print leaderboard/LEADERBOARD.md
    uv run python leaderboard.py scan       # import every known result file in the repository
    uv run python leaderboard.py submit FILE --method lp_local_search --label "my idea"
    uv run python leaderboard.py watch shinka_postproc/results/postproc_net13_run2   # rescan every 60 s

FILE is a QOBLIB .sol file or a JSON file with "arcs" (and "instance" or "n", or pass --instance).
From Python, in any environment that can import qoblib_tools:

    import leaderboard
    leaderboard.submit("network12", arcs, method="lp_local_search", label="ILS from QAOA pool")

No claimed score is trusted: every topology is checked (each node has 2 links out and 2 in) and
scored with qoblib_tools.route_min_congestion, the routing LP used by the notebook and QOBLIB.
`quantum` marks topologies that came out of a quantum sampler, possibly repaired or improved
classically afterwards; it defaults to True for methods named after QAOA, PCE or post-processing
of QAOA pools.

A topology is logged when it beats the best so far for its (instance, method), so the log is the
history of every method's records. Files in leaderboard/:
  submissions.jsonl  the log (append-only): score, method, label, source, date found, arcs
  LEADERBOARD.md     best per instance (any method), best with quantum sampling, best per method
  gap_to_best_known.png  chart of our best per instance against QOBLIB's best known values
  records.json       the record topologies
  solutions/         a QOBLIB .sol file (integer flows, checked) for each record
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import functools
import hashlib
import json
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
ROOT = REPO / "leaderboard"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import qoblib_tools as qt  # noqa: E402

# QOBLIB best-known values (08-network/solutions/README.md; 05-10 are proven optimal). Same table
# as shinka_network/problem.py.
BEST_KNOWN = {
    5: 65500, 6: 101000, 7: 142400, 8: 170231, 9: 196750, 10: 210800,
    11: 238334, 12: 276474, 13: 300620, 14: 343091, 15: 370870, 16: 397908,
    17: 437000, 18: 473375, 19: 492924, 20: 528528, 21: 569273, 22: 618135,
    23: 659369, 24: 663688,
}  # fmt: skip
QUANTUM_METHODS = re.compile(r"qaoa|pce|postproc", re.I)


def instance_n(instance) -> int:
    """12, "12" or "network12" -> 12."""
    m = re.fullmatch(r"(?:network)?0*(\d+)", str(instance).strip())
    if not m or int(m.group(1)) not in BEST_KNOWN:
        raise ValueError(f"unknown instance {instance!r}: use network05 ... network24")
    return int(m.group(1))


def topology_key(arcs) -> str:
    return hashlib.sha1(json.dumps(sorted(map(list, arcs))).encode()).hexdigest()[:12]


@functools.lru_cache(maxsize=None)
def demand(n: int):
    return qt.load_network_demand(n)


@functools.lru_cache(maxsize=4096)
def lp_score(n: int, arcs: tuple) -> float:
    return float(qt.route_min_congestion(demand(n), list(arcs)).objective)


@contextlib.contextmanager
def locked():
    """One writer at a time (several evaluators may submit at once, also from threads)."""
    ROOT.mkdir(parents=True, exist_ok=True)
    with open(ROOT / ".lock", "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        yield


def load() -> list[dict]:
    path = ROOT / "submissions.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


# --------------------------------------------------------------------------- #
# Submitting
# --------------------------------------------------------------------------- #
def make_entry(instance, arcs, method: str, label: str = "", source: str = "", quantum: bool | None = None,
               found: str | None = None, details: dict | None = None) -> dict:
    """Check and LP-score a topology. Raises ValueError if it is not a valid network for the instance."""
    n = instance_n(instance)
    arcs = tuple(sorted({(int(i), int(j)) for i, j in arcs}))
    if not qt.is_valid_topology(n, arcs):
        raise ValueError(f"not a valid network{n:02d} topology (every node needs 2 links out and 2 in)")
    score = lp_score(n, arcs)
    return {
        "instance": f"network{n:02d}", "n": n, "score": score, "bkv": BEST_KNOWN[n],
        "gap_pct": round(100.0 * (score / BEST_KNOWN[n] - 1.0), 4), "method": method, "label": label or method,
        "quantum": bool(QUANTUM_METHODS.search(method)) if quantum is None else bool(quantum),
        "source": source, "found": found or _now(), "logged": _now(), "key": topology_key(arcs),
        "details": details or {}, "arcs": [list(a) for a in arcs],
    }  # fmt: skip


def submit_entries(entries: list[dict]) -> list[str]:
    """Log the entries that beat the best so far for their (instance, method); rebuild. Returns news lines."""
    if not entries:
        return []
    with locked():
        best = {}
        for e in load():
            k = (e["instance"], e["method"])
            best[k] = min(best.get(k, float("inf")), e["score"])
        new = []
        for e in sorted(entries, key=lambda e: e["found"]):  # on ties the earliest find is kept
            k = (e["instance"], e["method"])
            if e["score"] < best.get(k, float("inf")):
                best[k] = e["score"]
                new.append(e)
        if new:
            with open(ROOT / "submissions.jsonl", "a") as fh:
                fh.writelines(json.dumps(e) + "\n" for e in new)
        return rebuild()


def submit(instance, arcs, method: str, label: str = "", source: str = "", quantum: bool | None = None,
           found: str | None = None, details: dict | None = None) -> list[str]:
    """Check, LP-score and log one topology (if it is a record for its method). Returns news lines,
    e.g. a new record for the instance."""
    return submit_entries([make_entry(instance, arcs, method, label, source, quantum, found, details)])


# --------------------------------------------------------------------------- #
# Records, solution files and LEADERBOARD.md (all derived from submissions.jsonl)
# --------------------------------------------------------------------------- #
def _better(table: dict, key, e: dict) -> None:
    cur = table.get(key)
    if cur is None or (e["score"], e["found"]) < (cur["score"], cur["found"]):
        table[key] = e


def write_solution(rec: dict) -> None:
    """QOBLIB .sol file for a record (integer flows from the routing MILP), checked by QOBLIB's rules."""
    path = ROOT / "solutions" / f"{rec['instance']}_{int(rec['score'])}_{rec['key'][:6]}.sol"
    rec["solution"] = f"solutions/{path.name}"
    try:
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            routing = qt.route_integral(demand(rec["n"]), [tuple(a) for a in rec["arcs"]])
            qt.write_network_solution(path, rec["n"], routing)
        valid, max_flow, msg = qt.check_network_solution(path, rec["n"], demand(rec["n"]))
        rec["solution_check"] = msg if valid and max_flow == rec["score"] else f"{msg}, max flow {max_flow}"
    except Exception as exc:  # noqa: BLE001 - the arcs are still in records.json
        rec["solution_check"] = f"no .sol file: {type(exc).__name__}: {exc}"


def rebuild() -> list[str]:
    """Rewrite records.json, solutions/ and LEADERBOARD.md (call under locked()). Returns news lines."""
    subs = load()
    overall, quantum, per_method = {}, {}, {}
    for e in subs:
        _better(overall, e["instance"], e)
        if e["quantum"]:
            _better(quantum, e["instance"], e)
        _better(per_method, (e["method"], e["instance"]), e)
    old_path = ROOT / "records.json"
    old = json.loads(old_path.read_text()) if old_path.exists() else {}
    news = []
    for kind, table in (("overall", overall), ("quantum", quantum)):
        for inst, rec in sorted(table.items()):
            write_solution(rec)
            prev = old.get(kind, {}).get(inst)
            if prev is None or rec["score"] < prev["score"]:
                was = f", was {prev['score']:.0f}" if prev else ""
                beat = "  *** BELOW QOBLIB's BEST KNOWN VALUE ***" if rec["score"] < rec["bkv"] else ""
                news.append(f"new {'record' if kind == 'overall' else 'quantum record'} {inst}: {rec['score']:.0f} "
                            f"({rec['gap_pct']:+.2f}% vs BKV {rec['bkv']}{was}) by {rec['method']} "
                            f"'{rec['label']}' -> leaderboard/{rec['solution']}{beat}")
    records = {"overall": overall, "quantum": quantum}
    current = {Path(r["solution"]).name for table in records.values() for r in table.values()}
    for path in (ROOT / "solutions").glob("*.sol"):  # superseded records (their arcs stay in the log)
        if path.name not in current:
            path.unlink()
    tmp = ROOT / ".records.json.tmp"
    tmp.write_text(json.dumps(records, indent=1))
    os.replace(tmp, old_path)
    tmp = ROOT / ".LEADERBOARD.md.tmp"
    tmp.write_text(render(subs, overall, quantum, per_method))
    os.replace(tmp, ROOT / "LEADERBOARD.md")
    try:
        plot_gaps(overall, quantum)
    except Exception as exc:  # noqa: BLE001 - a missing matplotlib must not block a submission
        print(f"leaderboard: chart not updated ({type(exc).__name__}: {exc})", file=sys.stderr)
    return news


# Chart colours: reference palette (light), categorical slots 1-2 + chrome.
INK, INK_2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
# (name, colour, marker, size, z): the quantum diamond is drawn larger and underneath, so where both series
# share a point the blue circle sits inside the orange diamond and both stay visible.
SERIES = [("Best by any method", "#2a78d6", "o", 44, 5), ("Best with quantum sampling", "#eb6834", "D", 90, 4)]


def plot_gaps(overall: dict, quantum: dict) -> None:
    """leaderboard/gap_to_best_known.png: our best per instance against QOBLIB's best known value."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure  # no pyplot: no global state, safe from evaluator threads

    instances = sorted(overall, key=lambda i: overall[i]["n"])
    if not instances:
        return
    xs = {inst: k for k, inst in enumerate(instances)}
    fig = Figure(figsize=(10, 4.8), dpi=150, facecolor=SURFACE)
    FigureCanvasAgg(fig)
    ax = fig.add_axes((0.07, 0.14, 0.77, 0.62), facecolor=SURFACE)
    gaps = [r["gap_pct"] for table in (overall, quantum) for r in table.values()]
    top = max(gaps + [1.0]) * 1.12
    bottom = min(min(gaps) * 1.3, -0.04 * top)
    ax.axhline(0, color=INK_2, lw=1.2, zorder=2)  # the main leaderboard: QOBLIB's best known values
    ax.text(len(instances) - 0.5, 0, "  QOBLIB best known", color=INK_2, fontsize=9, va="center", ha="left",
            clip_on=False)
    for (name, color, marker, size, z), table in zip(SERIES, (overall, quantum)):
        pts = sorted((xs[i], r["gap_pct"]) for i, r in table.items() if i in xs)
        if not pts:
            continue
        runs, cur = [], [pts[0]]  # connect consecutive sizes only; a missing size breaks the line
        for p in pts[1:]:
            if p[0] == cur[-1][0] + 1:
                cur.append(p)
            else:
                runs.append(cur)
                cur = [p]
        runs.append(cur)
        for run in runs:
            ax.plot([p[0] for p in run], [p[1] for p in run], color=color, lw=2, solid_capstyle="round",
                    solid_joinstyle="round", zorder=z - 2)
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=size, color=color, marker=marker,
                   edgecolors=SURFACE, linewidths=2, zorder=z, label=name)
        x_last, y_last = pts[-1]
        ax.annotate(name.replace("Best ", "best "), (x_last, y_last), xytext=(8, 0), textcoords="offset points",
                    color=INK_2, fontsize=8.5, va="center", annotation_clip=False)
    for inst, r in overall.items():  # call out instances where we beat the best known value
        if r["score"] < r["bkv"]:
            ax.annotate(f"★ {r['gap_pct']:+.2f}%",
                        (xs[inst], r["gap_pct"]), xytext=(0, 10), textcoords="offset points", ha="center",
                        color=INK, fontsize=8, zorder=6)
    ax.set_xticks(range(len(instances)), [i.replace("network", "") for i in instances], color=INK_2, fontsize=9)
    ax.set_xlabel("QOBLIB instance (network size n)", color=INK_2, fontsize=9)
    ax.set_ylabel("Gap to best known value (%)", color=INK_2, fontsize=9)
    ax.set_xlim(-0.6, len(instances) - 0.4)
    ax.set_ylim(bottom, top)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:+.0f}%" if v else "0%")
    ax.tick_params(axis="y", colors=INK_2, labelsize=9, length=0)
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="y", color=GRID, lw=1)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    fig.text(0.07, 0.94, "How close we are to QOBLIB's best known values", color=INK, fontsize=13, weight="bold")
    fig.text(0.07, 0.885, "Max link load of our best network per instance, % above the best known value "
             "(lower is better, 0 = matches it)", color=INK_2, fontsize=9.5)  # y below
    fig.legend(loc="upper left", bbox_to_anchor=(0.062, 0.86), ncol=2, frameon=False, fontsize=9,
               labelcolor=INK_2, handletextpad=0.3, columnspacing=1.6)
    fig.text(0.07, 0.025, f"Updated {dt.datetime.now():%Y-%m-%d %H:%M} by leaderboard.py. Values and "
             "solution files: LEADERBOARD.md.", color=MUTED, fontsize=8)
    tmp = ROOT / ".gap_to_best_known.png.tmp"
    fig.savefig(tmp, format="png", facecolor=SURFACE)
    os.replace(tmp, ROOT / "gap_to_best_known.png")


def _gap(rec: dict) -> str:
    mark = " ★" if rec["score"] < rec["bkv"] else (" =" if rec["score"] == rec["bkv"] else "")
    return f"{rec['gap_pct']:+.2f}%{mark}"


def _record_table(table: dict) -> list[str]:
    out = ["| Instance | BKV | Best found | Gap | Method | Found by | Found | Solution |",
           "|---|---:|---:|---:|---|---|---|---|"]
    for inst, rec in sorted(table.items()):
        check = "" if rec["solution_check"] == "valid" else f" ({rec['solution_check']})"
        out.append(f"| {inst} | {rec['bkv']} | {rec['score']:.0f} | {_gap(rec)} | {rec['method']} | {rec['label']} "
                   f"| {rec['found'][:16].replace('T', ' ')} | [{Path(rec['solution']).name}]({rec['solution']}){check} |")
    return out


def render(subs: list[dict], overall: dict, quantum: dict, per_method: dict) -> str:
    out = [
        "# Leaderboard: QOBLIB 08 network design (whole repository)",
        "",
        f"Updated {dt.datetime.now():%Y-%m-%d %H:%M} from `submissions.jsonl` ({len(subs)} records logged). "
        "Built by `leaderboard.py` (`show`, `scan`, `submit`); do not edit by hand.",
        "",
        "Score: the largest link load of the topology under optimal routing (the routing LP, lower is better), "
        "recomputed for every submission. Gap: score / BKV - 1, where BKV is QOBLIB's best known value "
        "(network05-10 proven optimal). `=` marks a tie with the BKV, ★ a score below it.",
        "",
        "![Gap of our best network per instance to QOBLIB's best known value](gap_to_best_known.png)",
        "",
        "## Best found per instance (any method)",
        "",
        *_record_table(overall),
        "",
        "## Best found with quantum sampling",
        "",
        "Topologies that came out of a quantum sampler (QAOA, PCE), possibly repaired or improved classically.",
        "",
        *(_record_table(quantum) if quantum else ["None yet."]),
        "",
        "## Best gap per method",
        "",
    ]
    instances = sorted(overall)
    methods = sorted({m for m, _ in per_method}, key=lambda m: min(
        (per_method[(m, i)]["gap_pct"] for i in instances if (m, i) in per_method)))
    out += ["| Method | " + " | ".join(instances) + " |", "|---|" + "---:|" * len(instances)]
    for m in methods:
        cells = " | ".join(_gap(per_method[(m, i)]) if (m, i) in per_method else "–" for i in instances)
        out.append(f"| {m} | {cells} |")
    out += ["", "## See also", "",
            "- Surrogate programs compared on network12-16 (mean score per program, several runs): "
            "[shinka_network/results/size_evals/PROGRAMS.md](../shinka_network/results/size_evals/PROGRAMS.md)"]
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------- #
# Importing the result files the repository's tools write
# --------------------------------------------------------------------------- #
def _mtime(path: Path) -> str:
    return dt.datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO))
    except ValueError:
        return str(path)


def _from_shinka_eval(path: Path) -> list[dict]:
    """shinka_network/results/<run>/gen_<g>/results/best_topologies.json (evaluate.py, evaluate_qoro.py)."""
    parts = path.resolve().relative_to(REPO).parts
    gen = next((p for p in parts if re.fullmatch(r"gen_\d+", p)), None)
    if gen is None or "best" in parts or ".seed-results-attempts" in parts:
        return []  # best/ and seed attempts duplicate a generation
    metrics_path = path.parent / "metrics.json"
    public = json.loads(metrics_path.read_text()).get("public", {}) if metrics_path.exists() else {}
    routine = public.get("routine") or "sa"
    out = []
    for name, v in json.loads(path.read_text()).items():
        m = re.fullmatch(r"network(\d+)_s\d+", name)  # real QOBLIB instances only (not subsets or toys)
        if m:
            out.append(make_entry(int(m.group(1)), v["arcs"], routine, f"{parts[2]} {gen.replace('_', ' ')}",
                                  _rel(path), found=_mtime(path), details={"backend": public.get("backend")}))
    return out


def _from_postproc(path: Path) -> list[dict]:
    """shinka_postproc/results/<run>/gen_<g>/results/best_network.json (QAOA pool + evolved LP search)."""
    d = json.loads(path.read_text())
    if not d.get("arcs") or d.get("control"):
        return []
    gen = next((p for p in path.parts if re.fullmatch(r"gen_\d+", p)), None)
    run = path.resolve().relative_to(REPO).parts[2]
    where = f"{run} {gen.replace('_', ' ')}" if gen else run  # no gen_<g> folder: analysis/big_budget.py
    return [make_entry(d["n"], d["arcs"], "postproc (QAOA pool + evolved LP search)",
                       f"{where} ({Path(d.get('pool', '')).stem})", _rel(path),
                       found=_mtime(path), details={"lp_calls": d.get("lp_calls")})]


def _from_pool(path: Path) -> list[dict]:
    """shinka_postproc/pools/**/*.json: raw QAOA samples; the best one per pool after the pipeline's repair."""
    d = json.loads(path.read_text())
    if not isinstance(d, dict) or "candidates" not in d or not d["candidates"]:
        return []
    n = instance_n(d["instance"])
    repaired = {tuple(sorted(qt.repair_topology(n, [tuple(a) for a in c["arcs"]], demand(n)))) for c in d["candidates"]}
    arcs = min(repaired, key=lambda a: lp_score(n, a))
    return [make_entry(n, arcs, "qaoa_partitioned", f"QAOA pool {path.stem} ({d.get('backend')}, "
                       f"{d.get('hamiltonian', '')})", _rel(path), found=_mtime(path))]


def _from_lp_json(path: Path) -> list[dict]:
    """shinka_network/results/reports/networkNN_lp_local_search.json (analysis/lp_local_search.py)."""
    d = json.loads(path.read_text())
    return [make_entry(d["instance"], d["arcs"], "lp_local_search", "analysis/lp_local_search.py",
                       _rel(path), found=_mtime(path))]


def _from_sol(path: Path, method: str, label: str) -> list[dict]:
    m = re.search(r"network(\d+)", path.name)
    _, arcs, _ = qt.read_network_solution(path)
    return [make_entry(int(m.group(1)), arcs, method, label, _rel(path), found=_mtime(path))] if m else []


# (folder, glob, reader). Add a line here when a new tool writes topologies in a new place.
SOURCES = [
    ("shinka_network/results", "**/best_topologies.json", _from_shinka_eval),
    ("shinka_postproc/results", "**/best_network.json", _from_postproc),
    ("shinka_postproc/pools", "**/*.json", _from_pool),
    ("shinka_network/results/reports", "*_lp_local_search.json", _from_lp_json),
    ("shinka_network/results/reports", "network*.sol",
     lambda p: _from_sol(p, "lp_local_search", "analysis/lp_local_search.py")),
    (".", "network*.sol", lambda p: _from_sol(p, "sa", "notebook 02_network_design (qoblib_tools QUBO)")),
]


def scan(only: list[Path] | None = None) -> list[str]:
    """Import every result file of SOURCES (only those under `only`, if given) that changed since the
    last scan. Returns news lines."""
    cache_path = ROOT / ".scan_cache.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    only = [p.resolve() for p in only] if only else None
    entries, seen = [], {}
    for folder, pattern, reader in SOURCES:
        for path in sorted((REPO / folder).glob(pattern)):
            if only and not any(path.resolve().is_relative_to(o) for o in only):
                continue
            stamp = path.stat().st_mtime
            if cache.get(_rel(path)) == stamp:
                continue
            try:
                entries += reader(path)
            except Exception as exc:  # noqa: BLE001 - one unreadable file must not stop the scan
                print(f"skipped {_rel(path)}: {type(exc).__name__}: {exc}", file=sys.stderr)
                continue  # not cached: tried again on the next scan
            seen[_rel(path)] = stamp
    news = submit_entries(entries)
    with locked():
        cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
        cache.update(seen)
        tmp = ROOT / ".scan_cache.json.tmp"
        tmp.write_text(json.dumps(cache))
        os.replace(tmp, cache_path)
    return news


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("show", help="rebuild leaderboard/LEADERBOARD.md and print it")
    p_scan = sub.add_parser("scan", help="import the result files the repository's tools write")
    p_scan.add_argument("paths", nargs="*", type=Path, help="only files under these folders")
    p_watch = sub.add_parser("watch", help="scan the given folders again and again (e.g. a running evolution)")
    p_watch.add_argument("paths", nargs="+", type=Path, help="folders to watch")
    p_watch.add_argument("--every", type=float, default=60.0, help="seconds between scans")
    p_sub = sub.add_parser("submit", help="submit a topology from a .sol or .json file")
    p_sub.add_argument("file", type=Path)
    p_sub.add_argument("--method", required=True, help="e.g. qaoa_partitioned, pce, sa, lp_local_search")
    p_sub.add_argument("--label", default="", help="what produced it (run, program, idea)")
    p_sub.add_argument("--instance", default=None, help="network12 (default: from the file)")
    p_sub.add_argument("--quantum", choices=["yes", "no"], default=None,
                       help="came out of a quantum sampler (default: guessed from --method)")
    args = parser.parse_args()

    if args.command == "show":
        with locked():
            news = rebuild()
        print((ROOT / "LEADERBOARD.md").read_text())
    elif args.command == "scan":
        news = scan(args.paths or None)
        print(f"scan done: {sum(1 for _ in load())} records logged in leaderboard/submissions.jsonl")
    elif args.command == "watch":
        import time

        print(f"watching {', '.join(map(str, args.paths))} every {args.every:.0f} s (Ctrl-C to stop)", flush=True)
        while True:
            try:
                for line in scan(args.paths):
                    print(f"[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] {line}", flush=True)
            except Exception as exc:  # noqa: BLE001 - keep watching through a bad file or a busy lock
                print(f"[{dt.datetime.now():%H:%M:%S}] scan failed: {type(exc).__name__}: {exc}", flush=True)
            time.sleep(args.every)
    else:
        if args.file.suffix == ".sol":
            _, arcs, _ = qt.read_network_solution(args.file)
            instance = args.instance or re.search(r"network\d+", args.file.name).group(0)
        else:
            d = json.loads(args.file.read_text())
            arcs, instance = d["arcs"], args.instance or d.get("instance") or d["n"]
        quantum = None if args.quantum is None else args.quantum == "yes"
        entry = make_entry(instance, arcs, args.method, args.label, _rel(args.file), quantum)
        news = submit_entries([entry])
        print(f"{entry['instance']}: score {entry['score']:.0f} ({entry['gap_pct']:+.2f}% vs BKV {entry['bkv']})")
    for line in news:
        print(line)


if __name__ == "__main__":
    main()
