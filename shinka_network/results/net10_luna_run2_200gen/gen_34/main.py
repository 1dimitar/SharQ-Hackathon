"""Surrogate QUBO terms for QOBLIB 08 network design. ShinkaEvolve edits the EVOLVE block."""

import time

# EVOLVE-BLOCK-START
import numpy as np


def surrogate_terms(demand):
    """Return linear, quadratic, and degree-penalty terms for the QUBO."""
    t = np.asarray(demand, dtype=float)
    n = len(t)
    if n == 0:
        return {}, {}, 1.0

    t = np.maximum(t, 0.0).copy()
    np.fill_diagonal(t, 0.0)

    scale = float(np.max(t))
    if scale <= 0.0:
        scale = 1.0

    normalized = t / scale
    concave = np.sqrt(normalized)

    row = np.sum(t, axis=1)
    col = np.sum(t, axis=0)
    endpoint = np.add.outer(row, col)
    endpoint_scale = float(np.max(endpoint))
    if endpoint_scale > 0.0:
        endpoint /= endpoint_scale
    else:
        endpoint.fill(0.0)

    linear = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                linear[(i, j)] = -(
                    0.78 * float(normalized[i, j])
                    + 0.17 * float(concave[i, j])
                    + 0.05 * float(endpoint[i, j])
                )

    # Estimate the relative attractiveness of each node as a transit point.
    node_traffic = row + col
    positive = node_traffic[node_traffic > 0.0]
    mean_traffic = float(np.mean(positive)) if positive.size else 1.0
    transit_factor = 1.0 / (
        1.0 + node_traffic / max(mean_traffic, 1.0e-12)
    )

    # Reward plausible two-hop routes, sharing each OD pair's incentive among
    # its possible intermediates. The coefficient balances the milder
    # inspiration version against the stronger current implementation.
    quadratic = {}
    indirect_fraction = 0.25
    for i in range(n):
        for k in range(n):
            if i == k or t[i, k] <= 0.0:
                continue

            share = 1.0 / (n - 2)
            demand_fraction = float(normalized[i, k])
            for j in range(n):
                if j == i or j == k:
                    continue
                transit_discount = 0.65 + 0.70 * float(transit_factor[j])
                weight = (
                    -indirect_fraction
                    * demand_fraction
                    * share
                    * transit_discount
                )
                quadratic[((i, j), (j, k))] = weight

    return linear, quadratic, 1.1


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
