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

    t = np.maximum(t, 0.0)
    np.fill_diagonal(t, 0.0)
    scale = float(np.max(t))
    if scale <= 0.0:
        scale = 1.0

    # Direct links are useful, but the concave component gives moderate
    # demands some influence as well.
    normalized = t / scale
    concave = np.sqrt(normalized)
    row = np.sum(t, axis=1)
    col = np.sum(t, axis=0)
    endpoint = np.add.outer(row, col)
    endpoint_scale = float(np.max(endpoint))
    if endpoint_scale > 0.0:
        endpoint = endpoint / endpoint_scale
    else:
        endpoint = np.zeros_like(endpoint)

    linear = {}
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = float(normalized[i, j])
            linear[(i, j)] = -(
                0.78 * d
                + 0.17 * float(concave[i, j])
                + 0.05 * float(endpoint[i, j])
            )

    # An OD pair's indirect-path reward is shared between its possible
    # intermediates. Busy intermediates receive less reward, reducing the
    # tendency to make a small set of nodes universal transit hubs.
    node_traffic = row + col
    pressure_scale = max(float(np.max(node_traffic)), 1.0e-12)
    transit_factor = np.maximum(
        0.75, 1.0 / (1.0 + node_traffic / pressure_scale)
    )

    quadratic = {}
    for i in range(n):
        for k in range(n):
            if i == k or t[i, k] <= 0.0:
                continue

            # Share the incentive over candidate intermediates, with more
            # share available when few plausible intermediates exist.
            candidates = [j for j in range(n) if j != i and j != k]
            if not candidates:
                continue
            demand_fraction = float(t[i, k]) / scale
            share = 1.0 / len(candidates)

            for j in candidates:
                weight = (
                    -0.48
                    * demand_fraction
                    * share
                    * (0.65 + 0.70 * float(transit_factor[j]))
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
