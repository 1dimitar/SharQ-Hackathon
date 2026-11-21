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

    # Reward a small, demand-adaptive set of two-hop relays per OD pair.
    # This concentrates interactions around plausible paths while retaining
    # extra alternatives for the heaviest traffic.
    quadratic = {}
    path_strength = 0.46
    positive_demands = traffic[traffic > 0.0]
    demand_cutoff = float(np.quantile(positive_demands, 0.75)) if positive_demands.size else 0.0
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

            candidates = []
            for j in range(n):
                if j == i or j == k:
                    continue
                edge_pressure = (
                    float(traffic[i, j]) + float(traffic[j, k])
                ) / scale
                # Prefer relays with useful incident demand but avoid
                # overloading nodes already under the greatest pressure.
                affinity = (
                    float(traffic[i, j]) + float(traffic[j, k])
                ) / scale
                rank = (0.25 + affinity) / (float(node_pressure[j]) ** 0.75)
                candidates.append((rank, j, edge_pressure))

            candidates.sort(reverse=True)
            keep = 4 if through_demand >= demand_cutoff else 2
            for rank, j, edge_pressure in candidates[:keep]:
                first, second = (i, j), (j, k)
                key = (first, second) if first < second else (second, first)
                # Slightly differentiate the retained relays so the best
                # ranked routes remain the most attractive.
                rank_factor = 1.0 + 0.08 * (keep - candidates.index((rank, j, edge_pressure)))
                weight = (
                    -path_strength * path_support * rank_factor
                    / (float(node_pressure[j]) * (1.0 + 0.5 * edge_pressure))
                )
                quadratic[key] = weight

    # Degree penalties are supplied by the evaluator.
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
