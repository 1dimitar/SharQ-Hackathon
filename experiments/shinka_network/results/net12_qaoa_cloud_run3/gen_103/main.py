"""Surrogate QUBO terms for QOBLIB 08 network design. ShinkaEvolve edits the EVOLVE block."""

import time

# EVOLVE-BLOCK-START
import numpy as np


def surrogate_terms(demand):
    """Build direct-link rewards and sparse endpoint congestion couplings."""
    traffic = np.asarray(demand, dtype=float)
    traffic = np.nan_to_num(traffic, nan=0.0, posinf=0.0, neginf=0.0)
    traffic = np.maximum(traffic, 0.0).copy()
    n = traffic.shape[0]
    np.fill_diagonal(traffic, 0.0)

    scale = float(np.max(traffic)) if traffic.size else 0.0
    if scale > 0.0:
        traffic /= scale

    # A concave reward gives moderately useful direct links some influence.
    linear = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                value = float(traffic[i, j])
                linear[(i, j)] = -(0.72 * value + 0.28 * np.sqrt(value))

    quadratic = {}
    # A sparse set of candidate links per endpoint limits interaction growth.
    # Keeping more candidates for small instances makes the balance estimate
    # less brittle; the cap keeps the graph compact as n increases.
    keep = min(n - 1, max(4, min(8, n // 2 + 2)))

    def add_pair(a, b, weight):
        if a == b:
            return
        key = (a, b) if a < b else (b, a)
        quadratic[key] = quadratic.get(key, 0.0) + float(weight)

    # Penalize jointly selecting high-demand links leaving the same source.
    # The coupling is a second-order approximation to a convex congestion cost.
    for i in range(n):
        links = [(i, j) for j in range(n) if j != i]
        links.sort(key=lambda e: (-traffic[e[0], e[1]], e[1]))
        links = links[:keep]
        for p in range(len(links)):
            a = links[p]
            va = float(traffic[a[0], a[1]])
            for q in range(p + 1, len(links)):
                b = links[q]
                vb = float(traffic[b[0], b[1]])
                # Product emphasizes pairs that are jointly capable of
                # creating a large direct-load concentration.
                add_pair(a, b, 0.34 * np.sqrt(va * vb))

    # Likewise discourage concentrating direct traffic on the same receiver.
    for j in range(n):
        links = [(i, j) for i in range(n) if i != j]
        links.sort(key=lambda e: (-traffic[e[0], e[1]], e[0]))
        links = links[:keep]
        for p in range(len(links)):
            a = links[p]
            va = float(traffic[a[0], a[1]])
            for q in range(p + 1, len(links)):
                b = links[q]
                vb = float(traffic[b[0], b[1]])
                add_pair(a, b, 0.34 * np.sqrt(va * vb))

    # The endpoint penalties are the main structural signal; this smaller
    # reward retains preference for links that directly serve substantial OD.
    penalty_weight = 1.5
    return linear, quadratic, penalty_weight


# EVOLVE-BLOCK-END


def run_surrogate(demand):
    """Fixed entry point called by the evaluator, once per demand matrix."""
    demand = np.array(demand, dtype=float)
    start = time.perf_counter()
    linear, quadratic, penalty_weight = surrogate_terms(demand.copy())
    return {
        "n": len(demand),
        "linear": linear,
        "quadratic": quadratic,
        "penalty_weight": penalty_weight,
        "seconds": time.perf_counter() - start,
    }
