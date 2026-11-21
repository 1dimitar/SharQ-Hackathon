"""Throughput and progress of a running (or finished) evolution run with the quantum evaluator.

    python3 analysis/throughput.py results/net12mix_qaoa_cloud_run4 --target 200

Reads gen_*/results/metrics.json: evaluations finished per hour, wall-clock time per evaluation and
per instance, projected time to --target generations, best combined score and best score per
instance (with the generation it came from). Standard library only.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics as st


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("--target", type=int, default=200)
    p.add_argument("--window", type=float, default=1800.0, help="seconds for the recent-rate estimate")
    args = p.parse_args()

    rows = []
    for path in glob.glob(os.path.join(args.run, "gen_*", "results", "metrics.json")):
        m = json.load(open(path))
        gen = int(path.split("gen_")[1].split(os.sep)[0])
        rows.append((os.path.getmtime(path), gen, m))
    if not rows:
        print("no finished evaluations yet")
        return
    rows.sort()
    start = min(os.path.getmtime(d) for d in glob.glob(os.path.join(args.run, "gen_*")))
    now = max(r[0] for r in rows)
    valid = [r for r in rows if r[2]["combined_score"] > 0]
    hours = max(now - start, 1.0) / 3600
    recent = [r for r in rows if r[0] >= now - args.window]
    rate = len(rows) / hours
    recent_rate = len(recent) / (args.window / 3600) if now - start > args.window else rate
    left = max(args.target - len(rows), 0)
    print(f"{len(rows)} evaluations finished ({len(valid)} valid) in {hours:.2f} h: {rate:.1f}/h overall, "
          f"{recent_rate:.1f}/h over the last {args.window / 60:.0f} min")
    print(f"projected time to {args.target}: {left / max(recent_rate, 1e-9):.1f} h more")

    walls = [r[2]["private"].get("sampling_wall_seconds") for r in valid]
    walls = [w for w in walls if w]
    if walls:
        print(f"sampling wall-clock per evaluation: median {st.median(walls):.0f}s, max {max(walls):.0f}s")
    inst_secs = [v["quantum_seconds"] for r in valid for v in r[2]["private"].values()
                 if isinstance(v, dict) and "quantum_seconds" in v]
    if inst_secs:
        print(f"per instance: median {st.median(inst_secs):.0f}s, max {max(inst_secs):.0f}s "
              f"({len(inst_secs)} instance runs)")

    if valid:
        best = max(valid, key=lambda r: r[2]["combined_score"])
        print(f"best combined score {best[2]['combined_score']:.4f} (gen {best[1]}); "
              f"seed program: {next((r[2]['combined_score'] for r in rows if r[1] == 0), float('nan')):.4f}")
        per = {}
        for _, gen, m in valid:
            for name, v in m["private"].items():
                if isinstance(v, dict) and "best" in v:
                    if name not in per or v["best"] < per[name][0]:
                        per[name] = (v["best"], v["ref"], gen)
        for name, (b, ref, gen) in sorted(per.items()):
            print(f"   best {name}: {b:.0f} vs ref {ref:.0f} ({100 * (b / ref - 1):+.2f}%), gen {gen}")


if __name__ == "__main__":
    main()
