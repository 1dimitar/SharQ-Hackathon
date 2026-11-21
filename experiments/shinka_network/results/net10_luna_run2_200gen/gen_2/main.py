"""Surrogate QUBO terms for QOBLIB 08 network design. ShinkaEvolve edits the EVOLVE block."""

import time

# EVOLVE-BLOCK-START
import numpy as np


def surrogate_terms(demand):
    """Build demand-based direct-link and short-route surrogate terms."""
    n = len(demand)
    scale = max(float(np.max(np.abs(demand))), 1e-12)

    # Preserve the useful baseline signal: a direct link is valuable in
    # proportion to the traffic it carries.
    linear = {
        (i, j): -float(demand[i, j]) / scale
        for i in range(n) for j in range(n) if i != j
    }
    quadratic = {}

    # Give a small bonus to a selected two-link route for an OD pair. Keep
    # this much weaker than the direct-link reward to avoid overvaluing
    # path motifs simply because they exist.
    path_weight = 0.06
    for i in range(n):
        for j in range(n):
            if j == i:
                continue
            for k in range(n):
                if k == i or k == j:
                    continue
                value = float(demand[i, k])
                if value > 0.0:
                    e1, e2 = (i, j), (j, k)
                    key = tuple(sorted((e1, e2)))
                    quadratic[key] = quadratic.get(key, 0.0) - path_weight * value / scale

    # A reciprocal pair is especially useful when substantial traffic runs
    # in both directions; use the weaker direction so one-sided demand does
    # not create an artificial reciprocity preference.
    reciprocal_weight = 0.18
    for i in range(n):
        for j in range(i + 1, n):
            value = min(float(demand[i, j]), float(demand[j, i]))
            if value > 0.0:
                key = tuple(sorted(((i, j), (j, i))))
                quadratic[key] = quadratic.get(key, 0.0) - reciprocal_weight * value / scale

    penalty_weight = 1.0
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
