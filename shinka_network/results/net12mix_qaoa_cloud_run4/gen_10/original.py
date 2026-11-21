"""Surrogate QUBO terms for QOBLIB 08 network design. ShinkaEvolve edits the EVOLVE block."""

import time

# EVOLVE-BLOCK-START
import numpy as np


def surrogate_terms(demand):
    """Construct demand-based topology costs and congestion couplings."""
    d = np.asarray(demand, dtype=float)
    n = len(d)
    if n < 2:
        return {}, {}, 1.0

    # Scale demand to make coefficients stable across instances.
    scale = float(np.max(np.abs(d))) or 1.0
    d = np.maximum(d, 0.0) / scale
    row = d.sum(axis=1)
    col = d.sum(axis=0)
    total = float(d.sum()) or 1.0

    links = [(i, j) for i in range(n) for j in range(n) if i != j]
    linear = {}

    # Direct traffic is attractive, but links attached to high-traffic endpoints
    # are costly because those nodes also need to carry transit traffic.
    for i, j in links:
        direct = d[i, j]
        endpoint_load = (row[i] + col[j]) / total
        linear[(i, j)] = -0.72 * direct + 0.10 * endpoint_load

    quadratic = {}

    def add(a, b, value):
        if a == b or value <= 0.0:
            return
        key = (a, b) if a < b else (b, a)
        quadratic[key] = quadratic.get(key, 0.0) + float(value)

    # Competing links at a node are more concerning when that node has large
    # traffic in the corresponding direction. Degree penalties are supplied
    # separately by the evaluator; these couplings express congestion instead.
    for i in range(n):
        outgoing = [(i, j) for j in range(n) if j != i]
        incoming = [(j, i) for j in range(n) if j != i]
        out_factor = row[i] / total
        in_factor = col[i] / total
        for p in range(len(outgoing)):
            for q in range(p + 1, len(outgoing)):
                add(outgoing[p], outgoing[q], 0.10 * out_factor / max(1, n - 2))
        for p in range(len(incoming)):
            for q in range(p + 1, len(incoming)):
                add(incoming[p], incoming[q], 0.10 * in_factor / max(1, n - 2))

    # A two-hop route i->j->k places load on both links. A positive coupling
    # discourages relying heavily on such routes, especially for large OD flows;
    # direct-link rewards remain the primary preference for serving that flow.
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            a = (i, j)
            for k in range(n):
                if k == i or k == j:
                    continue
                b = (j, k)
                flow = d[i, k]
                if flow > 0:
                    add(a, b, 0.16 * flow / max(1.0, row[i] + col[k]))

    # Bound the total coefficient mass so congestion terms do not overwhelm
    # direct-service information after the evaluator's global rescaling.
    max_abs = max(
        [abs(v) for v in linear.values()] +
        [abs(v) for v in quadratic.values()] +
        [1.0]
    )
    linear = {key: value / max_abs for key, value in linear.items()}
    quadratic = {key: value / max_abs for key, value in quadratic.items()}

    return linear, quadratic, 2.5


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
