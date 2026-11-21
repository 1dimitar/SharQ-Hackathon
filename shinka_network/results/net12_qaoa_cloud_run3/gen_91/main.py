"""Surrogate QUBO terms for QOBLIB 08 network design. ShinkaEvolve edits the EVOLVE block."""

import time

# EVOLVE-BLOCK-START
import numpy as np


def surrogate_terms(demand):
    """Return direct-service rewards and endpoint congestion couplings."""
    traffic = np.asarray(demand, dtype=float)
    n = len(traffic)

    traffic = np.maximum(traffic, 0.0).copy()
    np.fill_diagonal(traffic, 0.0)

    scale = float(np.max(traffic)) if traffic.size else 0.0
    if scale <= 0.0:
        scale = 1.0
    d = traffic / scale

    # A direct link is most valuable for large demands, with a concave
    # component retaining some incentive for moderate-demand pairs.
    utility = np.zeros((n, n), dtype=float)
    linear = {}
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            value = float(d[i, j])
            utility[i, j] = 0.72 * value + 0.28 * np.sqrt(value)
            linear[(i, j)] = -float(utility[i, j])

    # Estimate how much traffic is associated with each endpoint. Pairwise
    # costs are stronger at busy endpoints, where concentrating attractive
    # links is more likely to create a bottleneck.
    row_totals = np.sum(d, axis=1)
    col_totals = np.sum(d, axis=0)
    max_row = max(float(np.max(row_totals)) if n else 0.0, 1.0)
    max_col = max(float(np.max(col_totals)) if n else 0.0, 1.0)
    row_pressure = row_totals / max_row
    col_pressure = col_totals / max_col

    quadratic = {}
    coupling_scale = 0.16

    # Two selected links leaving the same source can concentrate that
    # source's direct traffic on a small number of network links.
    for i in range(n):
        links = [(i, j) for j in range(n) if j != i]
        pressure = 0.35 + 0.65 * float(row_pressure[i])
        for a in range(len(links)):
            e1 = links[a]
            u1 = float(utility[e1])
            for b in range(a + 1, len(links)):
                e2 = links[b]
                u2 = float(utility[e2])
                weight = coupling_scale * pressure * np.sqrt(u1 * u2)
                if weight > 1e-12:
                    quadratic[(e1, e2)] = float(weight)

    # Similarly discourage choosing multiple high-demand links into a
    # heavily loaded destination. These terms complement, rather than
    # duplicate, the source-side interactions above.
    for j in range(n):
        links = [(i, j) for i in range(n) if i != j]
        pressure = 0.35 + 0.65 * float(col_pressure[j])
        for a in range(len(links)):
            e1 = links[a]
            u1 = float(utility[e1])
            for b in range(a + 1, len(links)):
                e2 = links[b]
                u2 = float(utility[e2])
                weight = coupling_scale * pressure * np.sqrt(u1 * u2)
                if weight > 1e-12:
                    # No pair can share both a source and destination unless
                    # it is the same variable, so this key is unique here.
                    quadratic[(e1, e2)] = float(weight)

    # Degree constraints are added by the evaluator.
    penalty_weight = 5.0
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
