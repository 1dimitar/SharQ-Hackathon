"""Instances, references and the scoring pipeline for evolving the network surrogate.

Fixed code: ShinkaEvolve never edits this file. It turns the terms returned by the
evolved ``surrogate_terms(demand)`` into a QUBO, samples it, repairs the samples to
valid topologies and scores them with QOBLIB's routing LP (the referee).
"""

from __future__ import annotations

import itertools
import json
import math
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

warnings.filterwarnings("ignore", category=SyntaxWarning)

import dimod  # noqa: E402
import qoblib_tools as qt  # noqa: E402
from dwave.samplers import SimulatedAnnealingSampler  # noqa: E402

CACHE_DIR = Path(__file__).resolve().parent / "cache"

# QOBLIB best-known values (08-network/solutions/README.md, fetched 2026-11-21).
BEST_KNOWN = {
    5: 65500, 6: 101000, 7: 142400, 8: 170231, 9: 196750, 10: 210800,
    11: 238334, 12: 276474, 13: 300620, 14: 343091, 15: 370870, 16: 397908,
    17: 437000, 18: 473375, 19: 492924, 20: 528528, 21: 569273, 22: 618135,
    23: 659369, 24: 663688,
}  # fmt: skip


@dataclass
class Instance:
    name: str
    demand: np.ndarray
    ref: float  # best LP score we know for this instance (lower is better)
    num_reads: int  # annealing reads = number of topologies proposed
    seed: int


# --------------------------------------------------------------------------- #
# Instances
# --------------------------------------------------------------------------- #
def toy_demand(n: int, seed: int) -> np.ndarray:
    """Random demand matrix in the style of QOBLIB's: integers, ~30% zeros, zero diagonal."""
    rng = np.random.default_rng(1000 * n + seed)
    d = rng.integers(5, 51, size=(n, n))
    d[rng.random((n, n)) < 0.3] = 0
    np.fill_diagonal(d, 0)
    return d.astype(np.int64)


def subset_demand(n: int, seed: int) -> tuple[np.ndarray, list[int]]:
    """Demand among a random n-node subset of QOBLIB's 24 nodes (a new instance per seed)."""
    nodes = sorted(np.random.default_rng(seed).choice(24, size=n, replace=False).tolist())
    full = qt.load_network_demand(24)
    return full[np.ix_(nodes, nodes)], nodes


# suite name -> list of (kind, n, seed, num_reads)
#   kind "real": QOBLIB networkNN (seed only changes the annealing seed)
#   kind "toy":  random demand matrix
#   kind "sub":  random n-node subset of the 24-node QOBLIB matrix
SUITES = {
    # network10 only: 4 annealing seeds x 64 reads, so the score does not hinge on one lucky run.
    "net10": [("real", 10, s, 64) for s in range(4)],
    # network12 (the notebook's instance). evaluate_qoro.py uses the first --max_runs seeds.
    "net12": [("real", 12, s, 64) for s in range(4)],
    # Future: new instances from node subsets of the QOBLIB matrix.
    "sub10": [("sub", 10, s, 48) for s in range(4)],
    # Pipeline test: small random matrices.
    "toy": [("toy", 6, 0, 32), ("toy", 6, 1, 32), ("toy", 7, 0, 32), ("toy", 7, 1, 32),
            ("toy", 8, 0, 32), ("toy", 8, 1, 32)],
    # Larger random matrices.
    "toy_mid": [("toy", 8, 0, 48), ("toy", 10, 0, 48), ("toy", 10, 1, 48), ("toy", 12, 0, 48)],
    # The real QOBLIB instances.
    "real_small": [("real", 8, 0, 48), ("real", 9, 0, 48), ("real", 10, 0, 48), ("real", 12, 0, 64)],
    "real": [("real", 10, 0, 64), ("real", 12, 0, 64), ("real", 16, 0, 64), ("real", 20, 0, 64),
             ("real", 24, 0, 64)],
}  # fmt: skip


def load_suite(suite: str) -> list[Instance]:
    if suite not in SUITES:
        raise ValueError(f"unknown suite {suite!r}; choose from {sorted(SUITES)}")
    instances = []
    for kind, n, seed, reads in SUITES[suite]:
        if kind == "real":
            demand = qt.load_network_demand(n)
            name, ref = f"network{n:02d}_s{seed}", float(BEST_KNOWN[n])
        elif kind == "sub":
            demand, nodes = subset_demand(n, seed)
            name = f"sub{n}_s{seed}"
            ref = toy_reference(name, demand)
        else:
            demand = toy_demand(n, seed)
            name = f"toy_n{n}_s{seed}"
            ref = toy_reference(name, demand)
        instances.append(Instance(name, demand, ref, reads, seed=17 + seed))
    return instances


# --------------------------------------------------------------------------- #
# References for toy instances (exact for n <= 6, LP local search above)
# --------------------------------------------------------------------------- #
def lp_score(demand: np.ndarray, arcs) -> float:
    return float(qt.route_min_congestion(demand, list(arcs)).objective)


def enumerate_topologies(n: int):
    rows = [list(itertools.combinations([j for j in range(n) if j != i], 2)) for i in range(n)]
    for choice in itertools.product(*rows):
        indeg = [0] * n
        for pair in choice:
            for j in pair:
                indeg[j] += 1
        if all(x == 2 for x in indeg):
            yield [(i, j) for i, pair in enumerate(choice) for j in pair]


def local_search(demand: np.ndarray, start, rng, max_sweeps: int = 50):
    """First-improvement 2-exchange, scored by the LP: (a→b),(c→d) ⇒ (a→d),(c→b)."""
    cur = set(map(tuple, start))
    cur_score = lp_score(demand, cur)
    for _ in range(max_sweeps):
        improved = False
        arcs = list(cur)
        pairs = [(p, q) for p in range(len(arcs)) for q in range(len(arcs)) if p < q]
        rng.shuffle(pairs)
        for p, q in pairs:
            (a, b), (c, d) = arcs[p], arcs[q]
            if a == d or c == b or (a, d) in cur or (c, b) in cur:
                continue
            if (a, b) not in cur or (c, d) not in cur:
                continue
            cand = (cur - {(a, b), (c, d)}) | {(a, d), (c, b)}
            s = lp_score(demand, cand)
            if s < cur_score - 1e-9:
                cur, cur_score, improved = cand, s, True
        if not improved:
            break
    return cur_score, sorted(cur)


def best_by_local_search(demand: np.ndarray, restarts: int = 12, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = len(demand)
    best = (math.inf, None)
    for _ in range(restarts):
        res = local_search(demand, qt.random_topology(n, rng), rng)
        if res[0] < best[0]:
            best = res
    return best


def toy_reference(name: str, demand: np.ndarray) -> float:
    CACHE_DIR.mkdir(exist_ok=True)
    path = CACHE_DIR / "toy_refs.json"
    refs = json.loads(path.read_text()) if path.exists() else {}
    if name not in refs:
        n = len(demand)
        if n <= 6:
            refs[name] = min(lp_score(demand, t) for t in enumerate_topologies(n))
        else:
            refs[name] = best_by_local_search(demand)[0]
        path.write_text(json.dumps(refs, indent=1, sort_keys=True))
    return float(refs[name])


# --------------------------------------------------------------------------- #
# Evolved terms -> QUBO -> samples -> LP scores
# --------------------------------------------------------------------------- #
def max_quadratic_terms(n: int) -> int:
    return 4 * n**3


def validate_terms(n: int, linear, quadratic, penalty_weight) -> str | None:
    """Return an error message, or None if the terms are usable."""
    if not isinstance(linear, dict) or not isinstance(quadratic, dict):
        return "linear and quadratic must both be dicts"
    try:
        pw = float(penalty_weight)
    except (TypeError, ValueError):
        return "penalty_weight must be a number"
    if not math.isfinite(pw) or not (0.05 <= pw <= 100.0):
        return f"penalty_weight must be in [0.05, 100], got {pw}"

    def bad_arc(a) -> bool:
        return not (
            isinstance(a, tuple) and len(a) == 2
            and all(isinstance(x, (int, np.integer)) for x in a)
            and 0 <= a[0] < n and 0 <= a[1] < n and a[0] != a[1]
        )  # fmt: skip

    for a, w in linear.items():
        if bad_arc(a):
            return f"linear key {a!r} is not a link (i, j) with i != j and 0 <= i, j < {n}"
        if not math.isfinite(float(w)):
            return f"linear[{a!r}] is not finite"
    if len(quadratic) > max_quadratic_terms(n):
        return f"too many quadratic terms: {len(quadratic)} > 4*n^3 = {max_quadratic_terms(n)}"
    for key, w in quadratic.items():
        if not (isinstance(key, tuple) and len(key) == 2) or bad_arc(key[0]) or bad_arc(key[1]):
            return f"quadratic key {key!r} is not a pair of links ((i, j), (k, l))"
        if not math.isfinite(float(w)):
            return f"quadratic[{key!r}] is not finite"
    if not linear and not quadratic:
        return "no terms returned"
    return None


def build_bqm(demand: np.ndarray, linear, quadratic, penalty_weight):
    """Surrogate (rescaled to max |coefficient| = 1) + penalty_weight * degree penalty."""
    n = len(demand)
    arcs = qt.all_arcs(n)
    index = {a: v for v, a in enumerate(arcs)}
    lin = {v: 0.0 for v in range(len(arcs))}
    quad: dict[tuple[int, int], float] = {}
    for a, w in linear.items():
        lin[index[(int(a[0]), int(a[1]))]] += float(w)
    for (a, b), w in quadratic.items():
        u, v = index[(int(a[0]), int(a[1]))], index[(int(b[0]), int(b[1]))]
        if u == v:  # x*x = x
            lin[u] += float(w)
        else:
            key = (min(u, v), max(u, v))
            quad[key] = quad.get(key, 0.0) + float(w)
    quad = {k: w for k, w in quad.items() if w != 0.0}
    scale = max([abs(w) for w in lin.values()] + [abs(w) for w in quad.values()] + [1e-12])
    surrogate = dimod.BinaryQuadraticModel(
        {v: w / scale for v, w in lin.items()}, {k: w / scale for k, w in quad.items()}, 0.0, dimod.BINARY
    )
    _, penalty, _ = qt.build_network_qubo(demand)
    bqm = surrogate.copy()
    pen = penalty.copy()
    pen.scale(float(penalty_weight))
    bqm.update(pen)
    return bqm, surrogate, arcs


def score_samples(inst: Instance, samples, arcs, num_quadratic: int) -> dict:
    """Repair every sample to a valid topology and LP-score it. Shared by all samplers.

    samples: iterable of bit assignments in variable order (dict, array or bitstring).
    """
    demand, n = inst.demand, len(inst.demand)
    scores, feasible, cache = [], 0, {}
    best_score, best_arcs = math.inf, None
    for sample in samples:
        raw = qt.decode_topology(sample, arcs)
        feasible += qt.is_valid_topology(n, raw)
        topo = tuple(sorted(qt.repair_topology(n, raw, demand)))
        if topo not in cache:
            cache[topo] = lp_score(demand, topo)
        scores.append(cache[topo])
        if cache[topo] < best_score:
            best_score, best_arcs = cache[topo], topo
    if not scores:
        raise RuntimeError("the sampler returned no samples")
    scores = np.array(scores)
    best, median = float(scores.min()), float(np.median(scores))
    return {
        "name": inst.name,
        "n": n,
        "ref": inst.ref,
        "best": best,
        "median": median,
        "gap_best_pct": 100.0 * (best / inst.ref - 1.0),
        "gap_median_pct": 100.0 * (median / inst.ref - 1.0),
        "score": 0.5 * inst.ref / best + 0.5 * inst.ref / median,
        "feasible_rate": feasible / len(scores),
        "num_samples": len(scores),
        "distinct": len(cache),
        "num_quadratic": num_quadratic,
        "best_arcs": [list(map(int, a)) for a in best_arcs],
    }


def score_terms(inst: Instance, linear, quadratic, penalty_weight) -> dict:
    """Simulated annealing on the QUBO, then repair and LP-score (the local evaluator)."""
    bqm, surrogate, arcs = build_bqm(inst.demand, linear, quadratic, penalty_weight)
    sampleset = SimulatedAnnealingSampler().sample(bqm, num_reads=inst.num_reads, seed=inst.seed)
    return score_samples(inst, sampleset.samples(), arcs, surrogate.num_interactions)


# --------------------------------------------------------------------------- #
# Rules for the evolved code (shared by evaluate.py and evaluate_qoro.py)
# --------------------------------------------------------------------------- #
# The surrogate must be built from the demand matrix alone: no LP, no solvers, no file access.
FORBIDDEN = [
    r"qoblib", r"\bproblem\b", r"scipy\.optimize", r"linprog", r"\bmilp\b", r"dimod", r"dwave",
    r"\bneal\b", r"subprocess", r"os\.system", r"\bopen\(", r"\beval\(", r"\bexec\(", r"__import__",
    r"importlib", r"\bsocket\b", r"requests", r"urllib", r"\bdivi\b",
]  # fmt: skip
SECONDS_PER_INSTANCE = 20.0


def evolve_block(source: str) -> str:
    m = re.search(r"EVOLVE-BLOCK-START(.*?)EVOLVE-BLOCK-END", source, flags=re.S)
    return m.group(1) if m else source


def forbidden_use(program_path: str) -> str | None:
    code = evolve_block(Path(program_path).read_text())
    code = re.sub(r'""".*?"""', "", code, flags=re.S)  # ignore docstrings
    code = re.sub(r"#.*", "", code)
    for pattern in FORBIDDEN:
        if re.search(pattern, code):
            return f"forbidden construct in the evolved code: {pattern}"
    return None


def check_result(result) -> str | None:
    """Validate what run_surrogate returned. None if usable, else an error message."""
    if not isinstance(result, dict) or not {"n", "linear", "quadratic", "penalty_weight"} <= set(result):
        return "run_surrogate must return the dict built by the fixed wrapper"
    if result["seconds"] > SECONDS_PER_INSTANCE:
        return (
            f"surrogate_terms took {result['seconds']:.1f}s for n={result['n']} "
            f"(limit {SECONDS_PER_INSTANCE:.0f}s)"
        )
    return validate_terms(result["n"], result["linear"], result["quadratic"], result["penalty_weight"])
