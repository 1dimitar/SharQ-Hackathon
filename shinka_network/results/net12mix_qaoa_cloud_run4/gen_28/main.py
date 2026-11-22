"""Surrogate QUBO terms for QOBLIB 08 network design. ShinkaEvolve edits the EVOLVE block."""

import time

# EVOLVE-BLOCK-START
import numpy as np


def surrogate_terms(demand):
    """Build linear and quadratic surrogate cost terms for a directed topology."""
    t = np.asarray(demand, dtype=float)
    n = len(t)
    if n < 2:
        return {}, {}, 2.0

    # Keep the surrogate well-defined for unusual input values.
    t = np.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
    t = np.maximum(t, 0.0).copy()
    np.fill_diagonal(t, 0.0)

    scale = float(t.max())
    links = [(i, j) for i in range(n) for j in range(n) if i != j]
    if scale <= 0.0:
        return {link: 0.0 for link in links}, {}, 2.0

    d = t / scale

    # The concave reward reduces the dominance of the largest OD demands,
    # while still favoring links that carry substantial direct traffic.
    linear = {
        (i, j): -float(np.sqrt(d[i, j]))
        for i, j in links
    }

    # Retain endpoint exposure penalties: selecting two outgoing links from
    # a busy node toward busy receivers, or two incoming links from busy
    # origins to a busy node, is treated as a potential congestion conflict.
    row = d.sum(axis=1)
    col = d.sum(axis=0)
    row_pressure = row / (float(row.max()) + 1e-12)
    col_pressure = col / (float(col.max()) + 1e-12)

    quadratic = {}
    pair_scale = 0.12

    for i in range(n):
        outgoing = [(i, j) for j in range(n) if j != i]
        for a in range(len(outgoing)):
            u = outgoing[a]
            for b in range(a + 1, len(outgoing)):
                v = outgoing[b]
                w = pair_scale * row_pressure[i] * np.sqrt(
                    col_pressure[u[1]] * col_pressure[v[1]]
                )
                if w > 1e-10:
                    quadratic[(u, v)] = float(w)

    for j in range(n):
        incoming = [(i, j) for i in range(n) if i != j]
        for a in range(len(incoming)):
            u = incoming[a]
            for b in range(a + 1, len(incoming)):
                v = incoming[b]
                w = pair_scale * col_pressure[j] * np.sqrt(
                    row_pressure[u[0]] * row_pressure[v[0]]
                )
                if w > 1e-10:
                    key = (u, v) if u < v else (v, u)
                    quadratic[key] = quadratic.get(key, 0.0) + float(w)

    penalty_weight = 2.0
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
