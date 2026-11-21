"""Surrogate QUBO terms for QOBLIB 08 network design. ShinkaEvolve edits the EVOLVE block."""

import time

# EVOLVE-BLOCK-START
import numpy as np


def _demand_statistics(demand):
    """Return a finite, nonnegative demand matrix and its endpoint statistics."""
    d = np.asarray(demand, dtype=float)
    # Demand data is expected to be nonnegative; clipping also prevents malformed
    # values from creating rewards that reverse the intended energy direction.
    d = np.maximum(d, 0.0)
    n = d.shape[0]
    row = d.sum(axis=1)
    col = d.sum(axis=0)
    scale = float(d.max()) if d.size else 0.0
    if scale <= 0.0:
        scale = 1.0
    return d, row, col, scale, n


def _link_costs(d, row, col, scale, n):
    """Construct unary link rewards from direct demand and endpoint pressure."""
    linear = {}
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            direct = float(d[i, j]) / scale
            # Favor links that connect nodes with substantial traffic to send or
            # receive, while keeping this broad signal weaker than direct service.
            endpoint = (float(row[i]) + float(col[j])) / ((n - 1) * scale)
            linear[(i, j)] = -direct - 0.08 * endpoint
    return linear


def _two_hop_couplings(d, row, col, scale, n):
    """Reward useful consecutive links, discounted by endpoint traffic pressure."""
    quadratic = {}
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            for k in range(n):
                if k == i or k == j:
                    continue
                traffic = float(d[i, k])
                if traffic <= 0.0:
                    continue

                # A two-hop route is less attractive than a direct link. The
                # denominator attenuates paths through endpoints already under
                # heavy traffic pressure.
                pressure = max(
                    scale,
                    0.5 * (float(row[i]) + float(col[k]))
                )
                benefit = 0.12 * (traffic / pressure)
                if benefit > 0.0:
                    quadratic[((i, j), (j, k))] = -benefit
    return quadratic


def surrogate_terms(demand):
    """Return linear terms, quadratic terms, and the degree penalty weight."""
    d, row, col, scale, n = _demand_statistics(demand)
    linear = _link_costs(d, row, col, scale, n)
    quadratic = _two_hop_couplings(d, row, col, scale, n)

    # Keep the evaluator's degree penalty strong relative to the normalized
    # objective, without adding degree terms here.
    penalty_weight = 4.0
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
