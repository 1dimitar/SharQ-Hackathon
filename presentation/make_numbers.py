#!/usr/bin/env python3
"""Write the results table and headline numbers of the pitch from the repository leaderboard.

    cd presentation && python3 make_numbers.py && latexmk -pdf pitch.tex

Reads ../leaderboard/records.json (rebuilt by `leaderboard.py show` / the watchers) and writes
results_table.tex (macro \\ResultsRows, one table row per instance), numbers.tex (macros used on the slides) and title_graph.tex (title-slide art).
Re-run right before the talk: harvest jobs may still be adding records.
"""

import json
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
records = json.loads((HERE.parent / "leaderboard" / "records.json").read_text())["overall"]
quantum = json.loads((HERE.parent / "leaderboard" / "records.json").read_text()).get("quantum", {})

rows = sorted(records.values(), key=lambda r: r["n"])
rows = [r for r in rows if r["n"] >= 10]
max_abs_gap = max(abs(r["gap_pct"]) for r in rows) or 1.0
BAR_CM = 1.2  # length of the longest gap bar


def signed(x: float, digits: int) -> str:
    return f"{x:+.{digits}f}".replace("-", "$-$")


def num(x: float) -> str:
    return f"{int(round(x)):,}".replace(",", "\\,")


lines = []
for r in rows:
    n, gap = r["n"], r["gap_pct"]
    if gap < -1e-9:
        bar = (f"\\tikz[baseline=-0.6ex]\\fill[good] (0,-0.09) rectangle "
               f"({abs(gap) / max_abs_gap * BAR_CM:.3f},0.09);")
        gap_txt = f"\\textcolor{{good}}{{\\textbf{{{signed(gap, 2)}\\,\\%}}}}"
        ours = f"\\textbf{{{num(r['score'])}}}"
    elif abs(gap) <= 1e-9:
        bar = "\\tikz[baseline=-0.6ex]\\fill[tie] (0,0) circle (0.07);"
        gap_txt = "\\textcolor{tie}{$\\pm$0.00\\,\\%}"
        ours = num(r["score"])
    else:
        bar = ""
        gap_txt = f"{signed(gap, 2)}\\,\\%"
        ours = num(r["score"])
    lines.append(f"{n} & {n * (n - 1)} & {num(r['bkv'])} & {ours} & {gap_txt} & {bar} \\\\")
(HERE / "results_table.tex").write_text("\\newcommand{\\ResultsRows}{%\n" + "\n".join(lines) + "}\n")

records_below = [r for r in rows if r["gap_pct"] < -1e-9]
ties = [r for r in rows if abs(r["gap_pct"]) <= 1e-9]
best = min(rows, key=lambda r: r["gap_pct"])
big = max(rows, key=lambda r: r["n"])
below_from_quantum = [r for r in records_below
                      if quantum.get(r["instance"], {}).get("score") == r["score"]]
macros = {
    "NumRecords": str(len(records_below)),
    "NumRecordsQuantum": str(len(below_from_quantum)),
    "NumTies": str(len(ties)),
    "NumInstances": str(len(rows)),
    "BestGap": f"{signed(best['gap_pct'], 1)}\\,\\%",
    "BestGapN": str(best["n"]),
    "BigN": str(big["n"]),
    "BigVars": str(big["n"] * (big["n"] - 1)),
    "BigBefore": num(big["bkv"]),
    "BigAfter": num(big["score"]),
    "BigGap": f"{signed(big['gap_pct'], 2)}\\,\\%",
    "RecordsFrom": f"network{min(r['n'] for r in records_below):02d}" if records_below else "--",
    "RecordsTo": f"network{max(r['n'] for r in records_below):02d}" if records_below else "--",
    "Updated": datetime.now().strftime("%d %b %Y, %H:%M"),
}
(HERE / "numbers.tex").write_text(
    "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in macros.items()))

# Title-slide art: our record network on the largest instance, nodes on a circle.
n = big["n"]
draw = [f"\\coordinate (t{i}) at ({90 - 360 * i / n:.2f}:2.7);" for i in range(n)]
draw += [f"\\draw[titlearc] (t{i}) -- (t{j});" for i, j in big["arcs"]]
draw += [f"\\fill[titlenode] (t{i}) circle (0.075);" for i in range(n)]
(HERE / "title_graph.tex").write_text("\n".join(draw) + "\n")
print(f"{len(rows)} instances, {len(records_below)} below best known "
      f"({len(below_from_quantum)} from QAOA-seeded runs), {len(ties)} ties; "
      f"best {best['gap_pct']:+.2f}% (n={best['n']}); n={big['n']}: {big['bkv']} -> {big['score']:.0f}")
