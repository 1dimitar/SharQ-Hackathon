"""Surrogate QUBO terms for QOBLIB 08 network design. ShinkaEvolve edits the EVOLVE block."""

import time

# EVOLVE-BLOCK-START
import numpy as np


def surrogate_terms(demand):
    """Build direct-link rewards and sparse, pressure-aware two-hop rewards."""
    matrix = np.asarray(demand, dtype=float)
    n = len(matrix)

    traffic = np.maximum(matrix, 0.0).copy()
    np.fill_diagonal(traffic, 0.0)

    scale = float(np.max(traffic)) if traffic.size else 0.0
    if scale <= 0.0:
        scale = 1.0
    normalized = traffic / scale

    # Favor high-demand direct links while retaining some influence for
    # moderate demands, as in the concave direct-link model.
    linear = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                value = float(normalized[i, j])
                linear[(i, j)] = -(0.75 * value + 0.25 * np.sqrt(value))

    # Estimate transit pressure from the busier of each node's incoming and
    # outgoing demand. The floor keeps low-demand nodes eligible as relays.
    incoming = np.sum(normalized, axis=0)
    outgoing = np.sum(normalized, axis=1)
    incident = np.maximum(incoming, outgoing)
    max_incident = float(np.max(incident)) if incident.size else 0.0
    if max_incident <= 0.0:
        max_incident = 1.0
    node_pressure = np.maximum(0.65, incident / max_incident)

    quadratic = {}
    relay_budget = min(n - 2, max(2, n // 4))
    path_strength = 0.38

    for i in range(n):
        for k in range(n):
            if i == k:
                continue

            od = float(normalized[i, k])
            if od <= 0.0:
                continue

            path_support = 0.75 * od + 0.25 * np.sqrt(od)
            candidates = []

            for j in range(n):
                if j == i or j == k:
                    continue

                first_leg = float(normalized[i, j])
                second_leg = float(normalized[j, k])

                # Preserve the bottleneck-sensitive affinity of the current
                # implementation, with a small geometric component.
                affinity = (
                    0.75 * (min(first_leg, second_leg)
                            + 0.25 * max(first_leg, second_leg))
                    + 0.25 * np.sqrt(first_leg * second_leg)
                )
                rank = affinity / (0.75 + float(node_pressure[j]))
                edge_pressure = first_leg + second_leg
                candidates.append((rank, j, edge_pressure))

            candidates.sort(key=lambda item: (-item[0], item[1]))

            for rank, j, edge_pressure in candidates[:relay_budget]:
                rank_factor = 0.8 + 0.2 * rank / (1.0 + rank)
                pressure = float(node_pressure[j])

                # Blend relay congestion and pressure on the two chosen
                # edges, avoiding excessive rewards for arbitrary relays.
                relay_discount = np.sqrt(
                    (0.75 + pressure)
                    * pressure
                    * (1.0 + 0.5 * edge_pressure)
                )
                weight = (
                    -path_strength
                    * path_support
                    * rank_factor
                    / relay_discount
                )

                first, second = (i, j), (j, k)
                key = (first, second) if first < second else (second, first)
                quadratic[key] = quadratic.get(key, 0.0) + float(weight)

    # Degree penalties are added by the evaluator.
    penalty_weight = 7.0
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
