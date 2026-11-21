"""Surrogate QUBO terms for QOBLIB 08 network design. ShinkaEvolve edits the EVOLVE block."""

import time

# EVOLVE-BLOCK-START
import numpy as np


def surrogate_terms(demand):
    """Build demand-based direct-link and congestion-aware two-hop terms."""
    demand = np.asarray(demand, dtype=float)
    n = len(demand)

    # Scale using possible links only, so a diagonal entry cannot suppress
    # all useful weights. Protect the all-zero case.
    off_diagonal = demand.copy()
    np.fill_diagonal(off_diagonal, 0.0)
    scale = max(float(np.max(np.abs(off_diagonal))), 1e-12)

    # Preserve the baseline signal: directly serving high-demand OD pairs is
    # valuable.
    linear = {
        (i, j): -float(demand[i, j]) / scale
        for i in range(n)
        for j in range(n)
        if i != j
    }

    # A two-hop route is useful when its endpoint pair has significant demand.
    # Discount routes through busy nodes and links with high direct demand.
    incident = np.sum(np.abs(demand), axis=0) + np.sum(np.abs(demand), axis=1)
    mean_incident = max(float(np.mean(incident)), 1e-12)
    node_pressure = np.maximum(0.75, incident / mean_incident)

    quadratic = {}
    path_strength = 0.16

    for i in range(n):
        for k in range(n):
            if i == k:
                continue
            through_demand = float(demand[i, k])
            if through_demand <= 0.0:
                continue

            normalized_demand = through_demand / scale
            for j in range(n):
                if j == i or j == k:
                    continue

                edge_pressure = (
                    abs(float(demand[i, j])) + abs(float(demand[j, k]))
                ) / scale
                weight = (
                    -path_strength * normalized_demand
                    / (node_pressure[j] * (1.0 + 0.5 * edge_pressure))
                )

                # Canonical key ordering ensures a stable representation of
                # the product of the two binary link variables.
                e1, e2 = (i, j), (j, k)
                key = (e1, e2) if e1 < e2 else (e2, e1)
                quadratic[key] = quadratic.get(key, 0.0) + weight

    # Degree penalties are added externally by the evaluator.
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
