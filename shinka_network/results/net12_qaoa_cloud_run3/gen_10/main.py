"""Surrogate QUBO terms for QOBLIB 08 network design. ShinkaEvolve edits the EVOLVE block."""

import time

# EVOLVE-BLOCK-START
import numpy as np


def surrogate_terms(demand):
    """Build direct-link and pressure-discounted two-hop cost terms."""
    matrix = np.asarray(demand, dtype=float)
    n = len(matrix)

    # Use nonnegative off-diagonal traffic and handle the zero-demand case.
    traffic = np.maximum(matrix, 0.0).copy()
    np.fill_diagonal(traffic, 0.0)
    scale = float(np.max(traffic)) if traffic.size else 0.0
    if scale <= 0.0:
        scale = 1.0

    # A concave component makes moderate-demand direct links meaningful,
    # without abandoning the preference for directly serving heavy OD pairs.
    linear = {}
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            value = float(traffic[i, j]) / scale
            linear[(i, j)] = -(0.75 * value + 0.25 * np.sqrt(value))

    # Incident demand is a rough measure of how costly it is to use a node
    # as transit; discount paths through nodes with greater pressure.
    incoming = np.sum(traffic, axis=0)
    outgoing = np.sum(traffic, axis=1)
    # A relay needs capacity in both directions; its more heavily loaded
    # direction is the more informative bottleneck estimate.
    incident = np.maximum(incoming, outgoing)
    max_incident = float(np.max(incident)) if incident.size else 0.0
    if max_incident <= 0.0:
        max_incident = 1.0
    node_pressure = np.maximum(0.65, incident / max_incident)

    # Reward two-hop paths according to the OD traffic they could carry.
    # The concave path support preserves some influence for smaller demands.
    quadratic = {}
    path_strength = 0.46
    for i in range(n):
        for k in range(n):
            if i == k:
                continue
            through_demand = float(traffic[i, k])
            if through_demand <= 0.0:
                continue

            normalized_demand = through_demand / scale
            path_support = (
                0.70 * normalized_demand
                + 0.30 * np.sqrt(normalized_demand)
            )

            for j in range(n):
                if j == i or j == k:
                    continue

                edge_pressure = (
                    float(traffic[i, j]) + float(traffic[j, k])
                ) / scale
                weight = (
                    -path_strength * path_support
                    / (float(node_pressure[j]) * (1.0 + 0.5 * edge_pressure))
                )

                first, second = (i, j), (j, k)
                key = (first, second) if first < second else (second, first)
                quadratic[key] = weight

    # Degree penalties are supplied by the evaluator. Increase their strength
    # with network size to counter the growing number of two-hop interactions.
    penalty_weight = min(30.0, max(8.0, 0.8 * n))
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
