"""Surrogate QUBO terms for QOBLIB 08 network design. ShinkaEvolve edits the EVOLVE block."""

import time

# EVOLVE-BLOCK-START
import numpy as np


def surrogate_terms(demand):
    """Build direct-link rewards and sparse, pressure-aware relay rewards."""
    matrix = np.asarray(demand, dtype=float)
    n = len(matrix)

    traffic = np.maximum(np.nan_to_num(matrix, nan=0.0), 0.0).copy()
    np.fill_diagonal(traffic, 0.0)

    scale = float(np.max(traffic)) if traffic.size else 0.0
    if scale <= 0.0:
        scale = 1.0
    traffic /= scale

    # Favor direct service for larger demands while retaining some reward for
    # moderate demands.
    linear = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                value = float(traffic[i, j])
                linear[(i, j)] = -(0.75 * value + 0.25 * np.sqrt(value))

    # A relay with high total ingress or egress is a less attractive transit
    # node. Normalize pressure and keep a floor so no node is excluded.
    incoming = np.sum(traffic, axis=0)
    outgoing = np.sum(traffic, axis=1)
    incident = np.maximum(incoming, outgoing)
    max_incident = float(np.max(incident)) if incident.size else 0.0
    if max_incident <= 0.0:
        max_incident = 1.0
    node_pressure = np.maximum(0.65, incident / max_incident)

    quadratic = {}
    relay_budget = min(n - 2, max(1, n // 5))
    path_strength = 0.46

    for i in range(n):
        for k in range(n):
            if i == k:
                continue

            od = float(traffic[i, k])
            if od <= 0.0:
                continue

            support = 0.70 * od + 0.30 * np.sqrt(od)
            candidates = []

            for j in range(n):
                if j == i or j == k:
                    continue

                first_leg = float(traffic[i, j])
                second_leg = float(traffic[j, k])

                # Prefer relays whose two legs are plausible, while
                # discounting nodes already under substantial demand.
                affinity = np.sqrt(
                    (0.1 + first_leg) * (0.1 + second_leg)
                )
                rank = affinity / (0.75 + float(node_pressure[j]))
                candidates.append((rank, j))

            candidates.sort(key=lambda item: (-item[0], item[1]))

            for rank, j in candidates[:relay_budget]:
                rank_factor = 0.8 + 0.2 * rank / (1.0 + rank)
                weight = (
                    -path_strength
                    * support
                    * rank_factor
                    / (0.75 + float(node_pressure[j]))
                )

                first, second = (i, j), (j, k)
                key = (first, second) if first < second else (second, first)
                quadratic[key] = quadratic.get(key, 0.0) + weight

    # The evaluator supplies the degree penalties; use a stronger value than
    # the current program to reduce the number of invalid samples.
    penalty_weight = 5.0
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
