"""Surrogate QUBO terms for QOBLIB 08 network design. ShinkaEvolve edits the EVOLVE block."""

import time

# EVOLVE-BLOCK-START
import numpy as np


def surrogate_terms(demand):
    """Build direct-link rewards and selective endpoint-exposure penalties."""
    t = np.asarray(demand, dtype=float)
    n = t.shape[0]

    # Ignore negative entries and self-demand.
    t = np.maximum(t, 0.0).copy()
    np.fill_diagonal(t, 0.0)

    row = t.sum(axis=1)
    col = t.sum(axis=0)
    total = float(t.sum())
    max_demand = float(t.max()) if t.size else 0.0
    if max_demand <= 0.0:
        max_demand = 1.0

    # Normalize endpoint exposure to [0, 1] so pair weights have a stable
    # scale across demand matrices.
    max_row = float(row.max()) if n else 0.0
    max_col = float(col.max()) if n else 0.0
    row_exposure = row / max_row if max_row > 0.0 else np.zeros(n)
    col_exposure = col / max_col if max_col > 0.0 else np.zeros(n)

    if total > 0.0:
        node_importance = (row + col) / total
        # Adapt the exposure strength modestly to how unevenly traffic is
        # distributed across nodes.
        mean_imp = float(node_importance.mean())
        if mean_imp > 0.0:
            heterogeneity = float(node_importance.std()) / mean_imp
            heterogeneity = min(1.0, max(0.0, heterogeneity / 1.5))
        else:
            heterogeneity = 0.0
    else:
        node_importance = np.zeros(n)
        heterogeneity = 0.0

    linear = {}
    endpoint_scale = 0.08
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            direct = float(t[i, j]) / max_demand
            endpoint = 0.5 * (float(node_importance[i]) +
                              float(node_importance[j]))
            # Negative coefficients reward useful direct links.
            linear[(i, j)] = -(direct + endpoint_scale * endpoint)

    # Positive couplings discourage choosing link pairs that concentrate
    # exposure on already heavily used endpoints.
    exposure_scale = 0.12 + 0.04 * heterogeneity
    min_pair_exposure = 0.02
    quadratic = {}

    # Same sender: two selected outgoing links expose traffic to both
    # destinations, so penalize pairs of high-inbound-exposure destinations.
    for i in range(n):
        destinations = [j for j in range(n) if j != i]
        for a in range(len(destinations)):
            j = destinations[a]
            ej = float(col_exposure[j])
            for b in range(a + 1, len(destinations)):
                k = destinations[b]
                exposure = (ej * float(col_exposure[k])) ** 0.5
                if exposure >= min_pair_exposure:
                    quadratic[((i, j), (i, k))] = exposure_scale * exposure

    # Same receiver: two selected incoming links expose traffic from both
    # origins, so penalize pairs of high-outbound-exposure origins.
    for j in range(n):
        origins = [i for i in range(n) if i != j]
        for a in range(len(origins)):
            i = origins[a]
            ei = float(row_exposure[i])
            for b in range(a + 1, len(origins)):
                k = origins[b]
                exposure = (ei * float(row_exposure[k])) ** 0.5
                if exposure >= min_pair_exposure:
                    quadratic[((i, j), (k, j))] = exposure_scale * exposure

    penalty_weight = 3.5
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
