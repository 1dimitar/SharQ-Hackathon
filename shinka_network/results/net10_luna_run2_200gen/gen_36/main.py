"""Surrogate QUBO terms for QOBLIB 08 network design. ShinkaEvolve edits the EVOLVE block."""

import time

# EVOLVE-BLOCK-START
import numpy as np


def surrogate_terms(demand):
    """Build direct-link rewards and pressure-adjusted two-hop rewards."""
    matrix = np.asarray(demand, dtype=float)
    n = len(matrix)

    # Traffic demands are nonnegative. Use off-diagonal traffic for scaling,
    # and protect the all-zero case.
    traffic = np.maximum(matrix, 0.0).copy()
    np.fill_diagonal(traffic, 0.0)
    scale = float(np.max(traffic)) if traffic.size else 0.0
    if scale <= 0.0:
        scale = 1.0

    # Preserve the direct-demand signal: a link serving a large OD demand
    # receives a stronger reward.
    linear = {
        (i, j): -float(traffic[i, j]) / scale
        for i in range(n)
        for j in range(n)
        if i != j
    }

    # Intermediate nodes with high total incident demand are less attractive
    # as transit points. The floor keeps this discount bounded.
    incident = np.sum(traffic, axis=0) + np.sum(traffic, axis=1)
    mean_incident = float(np.mean(incident)) if incident.size else 0.0
    if mean_incident <= 0.0:
        mean_incident = 1.0
    node_pressure = np.maximum(0.75, incident / mean_incident)

    # Reward useful two-hop routes, discounted when either constituent link
    # already serves substantial direct demand.
    quadratic = {}
    path_strength = 0.20
    for i in range(n):
        for j in range(n):
            if j == i:
                continue
            for k in range(n):
                if k == i or k == j:
                    continue

                through_demand = float(traffic[i, k])
                if through_demand <= 0.0:
                    continue

                edge_pressure = (
                    float(traffic[i, j]) + float(traffic[j, k])
                ) / scale
                quadratic[((i, j), (j, k))] = (
                    -path_strength
                    * (through_demand / scale)
                    / (float(node_pressure[j]) * (1.0 + 0.5 * edge_pressure))
                )

    # The evaluator adds the degree penalties separately.
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
