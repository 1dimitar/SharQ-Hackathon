"""Build one half-page PDF per ShinkaEvolve-evolved program in submit/code/.

    uv run --no-project --with reportlab python docs/code_descriptions/build_pdfs.py

Edit the texts below and re-run; the PDFs are written next to this script.
"""
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

OUT = Path(__file__).resolve().parent
FONTS = Path("/System/Library/Fonts/Supplemental")
pdfmetrics.registerFont(TTFont("Arial", str(FONTS / "Arial.ttf")))
pdfmetrics.registerFont(TTFont("Arial-Bold", str(FONTS / "Arial Bold.ttf")))
pdfmetrics.registerFont(TTFont("Arial-Italic", str(FONTS / "Arial Italic.ttf")))
pdfmetrics.registerFontFamily("Arial", normal="Arial", bold="Arial-Bold", italic="Arial-Italic")

INK, MUTED, RULE, PANEL = colors.HexColor("#1f2933"), colors.HexColor("#5f6b7a"), colors.HexColor("#d5dbe3"), colors.HexColor("#f3f5f8")
KICKER = ParagraphStyle("kicker", fontName="Arial-Bold", fontSize=8, leading=10, textColor=MUTED, spaceAfter=3)
TITLE = ParagraphStyle("title", fontName="Arial-Bold", fontSize=15, leading=19, textColor=INK, spaceAfter=8)
BODY = ParagraphStyle("body", fontName="Arial", fontSize=9.6, leading=13, textColor=INK, alignment=TA_LEFT)
HEAD = ParagraphStyle("head", fontName="Arial-Bold", fontSize=10.5, leading=14, textColor=INK, spaceBefore=7, spaceAfter=3)
META_K = ParagraphStyle("metak", fontName="Arial-Bold", fontSize=8.3, leading=11, textColor=MUTED)
META_V = ParagraphStyle("metav", fontName="Arial", fontSize=8.3, leading=11, textColor=INK)


def code(s: str) -> str:
    return f'<font name="Courier">{s}</font>'


def build(name: str, title: str, meta: list[tuple[str, str]], intro: str, points: list[str], evidence: str,
          note: str | None = None) -> Path:
    path = OUT / f"{name}.pdf"
    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm,
                            topMargin=18 * mm, bottomMargin=18 * mm, title=f"{name}: {title}",
                            subject="QOBLIB 08 network design, ShinkaEvolve code")
    story = [Paragraph("QOBLIB 08 NETWORK DESIGN  ·  CODE EVOLVED WITH SHINKAEVOLVE", KICKER),
             Paragraph(f"{code(name)} — {title}", TITLE)]
    rows = [[Paragraph(k, META_K), Paragraph(v, META_V)] for k, v in meta]
    table = Table(rows, colWidths=[24 * mm, doc.width - 24 * mm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PANEL), ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ("LINEBELOW", (0, 0), (-1, -2), 0.4, RULE)]))
    story += [table, Spacer(1, 6)]
    if note:
        story += [Paragraph(f"<i>{note}</i>", BODY), Spacer(1, 4)]
    story += [Paragraph(intro, BODY), Paragraph("Main idea", HEAD),
              ListFlowable([ListItem(Paragraph(p, BODY), leftIndent=12) for p in points], bulletType="1",
                           bulletFontName="Arial-Bold", bulletFontSize=9.6, leftIndent=12, spaceBefore=0),
              Paragraph("Evidence", HEAD), Paragraph(evidence, BODY)]
    doc.build(story)
    return path


SCORE_SURR = "score = ½·BKV/best + ½·BKV/median per run, 1 = every sample optimal"

# --------------------------------------------------------------------------- Hamiltonians
build(
    "r2g181", "surrogate Hamiltonian with two-hop relay terms",
    [("File", f"{code('code/hamiltonians/r2g181.py')}, function {code('surrogate_terms')}"),
     ("Origin", "Generation 181 of ShinkaEvolve run 2 (200 generations), evolved from the starter surrogate in "
                f"{code('qoblib_tools')}"),
     ("Scored on", "network10: each candidate QUBO sampled by simulated annealing, samples repaired to valid "
                   "networks and scored with the exact routing LP"),
     ("Used for", "QAOA pools behind network16, 17, 19 and 23; starting point of r3g128")],
    "The topology QUBO has one binary x<sub>ij</sub> per directed link. The evaluator adds the degree rules "
    "(2 links out, 2 in per node) as a penalty; this program supplies the rest, an energy that should be low for "
    "networks with a low maximum link load. The starter surrogate only rewards each link by the demand it carries "
    "directly (−d<sub>ij</sub>). With just two out-links per node most traffic must be relayed, and the starter "
    "cannot see that.",
    ["<b>Direct links</b> (linear terms). Link i→j gets −(0.75·d<sub>ij</sub> + 0.25·√d<sub>ij</sub>), with "
     "d<sub>ij</sub> = t<sub>ij</sub> / max t the normalised demand. The square root keeps medium demands in play "
     "instead of only the heaviest pairs.",
     "<b>Two-hop relays</b> (couplings). For every demand pair i→k and every possible relay j, building both i→j "
     "and j→k is rewarded with −0.46·s<sub>ik</sub> / (p<sub>j</sub>·(1 + 0.5·e<sub>ijk</sub>)), where "
     "s<sub>ik</sub> = 0.7·d<sub>ik</sub> + 0.3·√d<sub>ik</sub> is the traffic the path could carry.",
     "<b>Congestion discounts.</b> p<sub>j</sub> = max(0.65, relative traffic of node j, the larger of its in- and "
     "out-totals) makes busy nodes weaker relays; e<sub>ijk</sub> = d<sub>ij</sub> + d<sub>jk</sub> does the same "
     "for links already loaded with their own direct traffic. Degree-penalty weight 1.0.",
     "<b>Dense.</b> One coupling per demand pair and relay: 880 on network12, 7700 on network24."],
    f"On network12 with partitioned QAOA (local simulator, 4 seeds; {SCORE_SURR}) the score rose from 0.742 with "
    "the starter surrogate to 0.838. Although evolved on network10 with annealing, it still beats the starter on "
    "network17–24 with annealing (mean score 0.842 vs 0.818).")

build(
    "r3g128", "sparse surrogate Hamiltonian",
    [("File", f"{code('code/hamiltonians/r3g128.py')}, function {code('surrogate_terms')}"),
     ("Origin", "Generation 128 of ShinkaEvolve run 3, which started from r2g181"),
     ("Scored on", "network12: each candidate QUBO sampled by partitioned QAOA on Qoro's cloud (2 runs per "
                   "candidate), samples repaired and scored with the exact routing LP"),
     ("Used for", "QAOA pools behind network11–15, 18, 20–22, 24 and network12_quantum")],
    "Same building blocks as r2g181: the topology QUBO has one binary per directed link, direct links are "
    "rewarded by −(0.75·d<sub>ij</sub> + 0.25·√d<sub>ij</sub>), and two-hop relays i→j→k are rewarded in "
    "proportion to the demand i→k, discounted when node j (p<sub>j</sub>) or links i→j, j→k (e<sub>ijk</sub>) are "
    "already busy. The change: each demand pair keeps only a few relays instead of all n−2.",
    ["<b>Rank the relays.</b> For each demand pair i→k, every relay j gets an affinity "
     "(0.25 + e<sub>ijk</sub>) / p<sub>j</sub><super>0.75</super>: relays whose links already serve related "
     "traffic and that are not congested come first.",
     "<b>Keep the best few.</b> The top 4 relays for the heaviest quarter of demand pairs, the top 2 for the rest. "
     "Each kept relay gets r2g181's coupling weight times a small rank bonus, 1 + 0.08·(slots − rank).",
     "<b>One fallback relay.</b> The least congested of the remaining relays is added with a weaker weight "
     "(0.30 instead of 0.46), so a pair is never tied to busy relays only.",
     "<b>Stronger degree penalty:</b> weight 1.25 instead of 1.0."],
    "308 couplings instead of 880 on network12, 1234 instead of 7700 on network24: each coupling is a two-qubit "
    "interaction in every QAOA cost layer, so the circuits are much lighter. Re-evaluated with 6 partitioned-QAOA "
    f"runs on network12 (local simulator; {SCORE_SURR}): 0.850 ± 0.014 against 0.841 ± 0.018 for r2g181, the same "
    "quality within noise with about a third of the couplings. With annealing on network17–24: mean score 0.845 "
    "(r2g181 0.842, starter 0.818).")

# --------------------------------------------------------------------------- post-processing
POOL_INTRO = ("A QAOA pool is 200 sampled topologies, most of which break the degree rules. The post-processor must "
              "return the best valid network it can find with a fixed budget of routing-LP calls; each call scores "
              "one network exactly and also returns its link loads and the LP's dual prices. Repairing a sample to "
              "the nearest valid network and computing its QUBO energy are free.")

build(
    "init_run1_gen49", "post-processing: bottleneck-focused archive search",
    [("File", f"{code('code/postproc/init_run1_gen49.py')}, function {code('improve')}"),
     ("Origin", "Generation 49 of ShinkaEvolve post-processing run 1, evolved from a baseline that repairs every "
                "sample, then runs swap descent with random kicks"),
     ("Scored on", "three QAOA pools of network12, 2000 LP calls per run; score = mean over pools of "
                   "best known / best found (1 = best known matched)"),
     ("Used for", "network11, 13, 21 and network12_quantum; starting point of run 3 (harvest_A_gen20, gen33)")],
    POOL_INTRO,
    ["<b>Diverse, low-energy starts.</b> Repair all 200 samples, then LP-score the 30 with the lowest QUBO energy, "
     "skipping any within 4 links of one already chosen.",
     "<b>An archive of parents.</b> Keep up to 32 of the best networks seen, 24 of them at least 4 links apart. "
     "Each round picks a parent: 48% of the time one of the top 6, otherwise any archive member.",
     "<b>Moves aimed at the bottleneck.</b> The dual prices mark the links that set the maximum load. The program "
     "tries every degree-preserving switch (a→b, c→d) → (a→d, c→b) that touches such a link, plus 70 random "
     "switches, three-link rotations (more when a parent keeps failing) and multi-switch kicks.",
     "<b>Free screening before paying for an LP.</b> New neighbours are ranked by QUBO energy change; a batch of 9 "
     "goes to the LP: 6 low-energy moves that differ from each other, the next best by energy, and a few random "
     "ones."],
    "On the network12 pools at 2000 LP calls the score rose from 0.967 (baseline) to 0.990, a mean gap of 1.0% to "
    "the best known value. Given random valid networks instead of the QAOA samples, the same program scored 0.973, "
    "so the QAOA samples are the better starting point. With 10000–20000 LP calls it matched the best known value "
    "on network11 and beat it on network13 (−0.30%) and network21 (−2.15%).")

GEN20_META_SCORED = ("16 QAOA pools of network13–16 (Hamiltonians r2g181 and r3g128, 2 seeds each), 2000 LP calls "
                     "per run; score = mean over pools of best known / best found")
GEN20_INTRO = ("Keeps init_run1_gen49's archive and bottleneck moves, but spends LP calls more carefully: before "
               "paying for an LP it estimates which neighbours are worth one, with a cheap routing model and a "
               "predictor it learns during the run.")
GEN20_POINTS = [
    "<b>Cheap routing, no LP.</b> For a whole batch of candidate networks at once: all-pairs hop distances "
    "(Floyd–Warshall), then every demand is sent along shortest paths, split equally at each node between "
    "out-links that stay on a shortest path. This gives approximate link loads and nine features: QUBO energy, "
    "traffic-weighted hop count, largest load, mean of the top 4 loads, RMS load, worst source and destination "
    "hop work, share of demand on direct links, unreachable pairs.",
    "<b>A predictor learned during the run.</b> Once max(3n, 36) networks have been LP-scored, a ridge regression "
    "maps the features to the LP score, with an uncertainty from each candidate's leverage. Nothing is carried "
    "over between instances.",
    "<b>Batches from several rankings.</b> Each batch (max(8, n/2) networks, at least 3 links apart) mixes the "
    "best predicted scores, the lowest QUBO energies, the best hop-count proxy, one network far from the archive "
    "and one optimistic pick (prediction − 0.65 × uncertainty).",
    "<b>Wider moves.</b> The full switch neighbourhood (moves touching the bottleneck sampled twice as often), "
    "three-link rotations, and short walks towards another archive network. After a round that improves on its "
    "parent, the next round continues from the improvement 62% of the time. Up to 2n + 4 starts, alternating QUBO "
    "energy with the routing proxy."]

build(
    "harvest_A_gen20", "post-processing with cheap routing and a learned LP predictor",
    [("File", f"{code('code/postproc/harvest_A_gen20.py')}, function {code('improve')}"),
     ("Origin", "Generation 20 of ShinkaEvolve post-processing run 3, which started from init_run1_gen49. This copy "
                "raises the evolved 140 s time guard to 3300 s so large LP budgets can finish."),
     ("Scored on", GEN20_META_SCORED),
     ("Used for", "network14, 16, 20, 22, 23, 24 (network15 and 18 with ckpt_A_gen20, the same program)")],
    GEN20_INTRO, GEN20_POINTS,
    "On the 16 training pools at 2000 LP calls the mean gap to the best known value fell from 2.32% (its starting "
    "program, init_run1_gen49) to 1.04%. With 20000–25000 LP calls it set six records, among them network23 "
    "(−3.42%) and network24 (−0.63%, 25000 calls in 28 minutes).")

build(
    "ckpt_A_gen20", "post-processing with cheap routing and a learned LP predictor (as evolved)",
    [("File", f"{code('code/postproc/ckpt_A_gen20.py')}, function {code('improve')}"),
     ("Origin", "Generation 20 of ShinkaEvolve post-processing run 3, exactly as evolved, which started from "
                "init_run1_gen49"),
     ("Scored on", GEN20_META_SCORED),
     ("Used for", "network15 (−0.31%) and network18 (−1.56%)")],
    GEN20_INTRO, GEN20_POINTS,
    "On the 16 training pools at 2000 LP calls the mean gap to the best known value fell from 2.32% (its starting "
    "program, init_run1_gen49) to 1.04%. With a 10000-call budget it beat the best known value on network15 "
    "(−0.31%, all calls used in 140 s) and network18 (−1.56%, stopped by the guard after 5593 calls).",
    note="This is the same algorithm as harvest_A_gen20. The only difference: this version keeps the evolved "
         "140 s time guard, which ends the search after 140 s even if LP calls remain; harvest_A_gen20 raises it "
         "to 3300 s.")

build(
    "harvest_A_gen33", "post-processing with uncertainty-driven exploration",
    [("File", f"{code('code/postproc/harvest_A_gen33.py')}, function {code('improve')}"),
     ("Origin", "Generation 33 of ShinkaEvolve post-processing run 3; its code is harvest_A_gen20 plus the changes "
                "below. The evolved 140 s time guard is raised to 3300 s."),
     ("Scored on", GEN20_META_SCORED),
     ("Used for", "network17 and network19")],
    "Same design as harvest_A_gen20: cheap shortest-path routing features, a ridge-regression predictor of the "
    "LP score learned during the run, and an archive search with bottleneck-focused moves. Generation 33 changes "
    "how the search explores.",
    ["<b>Better-calibrated uncertainty.</b> The residual variance is corrected for the number of fitted "
     "coefficients, the intercept's uncertainty is included, and leverage is capped, so an extrapolated "
     "candidate can never look more than one residual standard deviation better than its prediction.",
     "<b>A reserved exploration slot.</b> Every batch keeps room for one uncertainty-driven pick (lowest "
     "prediction − uncertainty among moves predicted better than the median) and one random pick, on top of the "
     "quotas for the best predicted and lowest-energy moves.",
     "<b>Rotations anchored on the bottleneck.</b> At least max(2n, 24) three-link rotations per round include a "
     "bottleneck link, instead of hitting one only by chance.",
     "<b>Escaping stalls.</b> Once a parent has failed twice, the walks head for the archive network farthest "
     "from it, and they grow longer and more numerous the longer the parent stalls."],
    "At 2000 LP calls its mean gap on the training pools (1.33%) was slightly behind generation 20 (1.04%). With "
    "25000 LP calls it found our best networks for network17 (−1.16%) and network19 (−2.62%).")

print("\n".join(sorted(str(p.name) for p in OUT.glob("*.pdf"))))
