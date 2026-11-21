"""Surrogate QUBO terms for QOBLIB 08 network design. ShinkaEvolve edits the EVOLVE block."""

import time

# EVOLVE-BLOCK-START
import numpy as np


def surrogate_terms(demand):
    """Build direct-link and selectively ranked two-hop surrogate costs."""
    matrix = np.asarray(demand, dtype=float)
    n = len(matrix)

    traffic = np.maximum(matrix, 0.0).copy()
    np.fill_diagonal(traffic, 0.0)
    scale = float(np.max(traffic)) if traffic.size else 0.0
    if scale <= 0.0:
        scale = 1.0
    traffic /= scale

    # Concavity retains useful signal for moderate OD demands.
    linear = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                value = float(traffic[i, j])
                linear[(i, j)] = -(0.75 * value + 0.25 * np.sqrt(value))

    incoming = np.sum(traffic, axis=0)
    outgoing = np.sum(traffic, axis=1)
    incident = np.maximum(incoming, outgoing)
    max_incident = float(np.max(incident)) if incident.size else 0.0
    if max_incident <= 0.0:
        max_incident = 1.0
    pressure = np.maximum(0.65, incident / max_incident)

    quadratic = {}
    relay_budget = min(n - 2, max(1, n // 5))
    relay_strength = 0.35

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

                first = float(traffic[i, j])
                second = float(traffic[j, k])

                # Favor relays with two credible legs; a weak leg limits the
                # usefulness of the entire path.
                affinity = min(first, second) + 0.25 * max(first, second)
                rank = affinity / (0.75 + float(pressure[j]))
                candidates.append((rank, j, first + second))

            candidates.sort(key=lambda item: (-item[0], item[1]))
            for rank, j, legs in candidates[:relay_budget]:
                rank_factor = 0.80 + 0.20 * rank / (1.0 + rank)
                weight = (
                    -relay_strength * support * rank_factor
                    / ((0.75 + float(pressure[j])) *
                       (1.0 + 0.20 * legs))
                )
                first_link, second_link = (i, j), (j, k)
                key = ((first_link, second_link)
                       if first_link < second_link
                       else (second_link, first_link))
                quadratic[key] = quadratic.get(key, 0.0) + weight

    # Degree penalties are applied by the evaluator, not included in these terms.
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
