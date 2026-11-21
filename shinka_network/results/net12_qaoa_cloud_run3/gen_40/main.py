"""Surrogate QUBO terms for QOBLIB 08 network design. ShinkaEvolve edits the EVOLVE block."""

import time

# EVOLVE-BLOCK-START
import numpy as np


def surrogate_terms(demand):
    """Build sparse route credits and endpoint-crowding costs."""
    traffic = np.asarray(demand, dtype=float)
    n = len(traffic)

    traffic = np.nan_to_num(
        traffic, nan=0.0, posinf=0.0, neginf=0.0
    )
    traffic = np.maximum(traffic, 0.0).copy()
    np.fill_diagonal(traffic, 0.0)

    scale = float(np.max(traffic)) if traffic.size else 0.0
    if scale <= 0.0:
        scale = 1.0
    q = traffic / scale

    # Direct service is valuable, but the concave component gives moderate
    # OD demands some influence when links are selected.
    linear = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                d = float(q[i, j])
                linear[(i, j)] = -(0.68 * d + 0.32 * np.sqrt(d))

    # Estimate relay pressure from each node's larger incoming/outgoing
    # demand total. A pressure exponent below one avoids over-disqualifying
    # busy nodes when they are still useful relays.
    incident = np.maximum(np.sum(q, axis=1), np.sum(q, axis=0))
    largest = float(np.max(incident)) if incident.size else 0.0
    if largest > 0.0:
        pressure = incident / largest
    else:
        pressure = np.zeros(n, dtype=float)

    quadratic = {}

    # Each OD pair has one bounded pool of indirect-route credit. Rank
    # relay options by availability, then split that pool between at most
    # two alternatives rather than rewarding every possible relay equally.
    path_budget = 0.24
    relay_count = min(2, max(0, n - 2))

    for i in range(n):
        for k in range(n):
            if i == k or q[i, k] <= 0.0 or relay_count == 0:
                continue

            d = float(q[i, k])
            support = 0.72 * d + 0.28 * np.sqrt(d)
            candidates = []

            for j in range(n):
                if j == i or j == k:
                    continue

                relay_cost = (0.75 + float(pressure[j])) ** 0.75
                edge_exposure = float(q[i, j] + q[j, k])
                quality = 1.0 / (
                    relay_cost * (1.0 + 0.22 * edge_exposure)
                )
                candidates.append((quality, j))

            candidates.sort(key=lambda item: (-item[0], item[1]))
            chosen = candidates[:relay_count]
            total_quality = sum(item[0] for item in chosen)
            if total_quality <= 0.0:
                continue

            for quality, j in chosen:
                first, second = (i, j), (j, k)
                key = (first, second) if first < second else (second, first)
                quadratic[key] = quadratic.get(key, 0.0) - (
                    path_budget * support * quality / total_quality
                )

    # Weak local crowding costs approximate endpoint capacity pressure:
    # choosing two high-demand links with the same tail or head is mildly
    # discouraged. Limiting this to each node's strongest few incident
    # demands keeps the interaction graph compact.
    crowding_strength = 0.035
    keep_per_endpoint = min(4, max(0, n - 1))

    def add_crowding_pairs(links):
        if len(links) < 2:
            return
        ranked = sorted(
            links,
            key=lambda edge: (-float(q[edge[0], edge[1]]), edge)
        )[:keep_per_endpoint]
        for a in range(len(ranked)):
            e1 = ranked[a]
            for b in range(a + 1, len(ranked)):
                e2 = ranked[b]
                value = crowding_strength * np.sqrt(
                    float(q[e1[0], e1[1]]) * float(q[e2[0], e2[1]])
                )
                if value <= 0.0:
                    continue
                key = (e1, e2) if e1 < e2 else (e2, e1)
                quadratic[key] = quadratic.get(key, 0.0) + float(value)

    for node in range(n):
        add_crowding_pairs([(node, j) for j in range(n) if j != node])
        add_crowding_pairs([(i, node) for i in range(n) if i != node])

    # Degree penalties are supplied by the evaluator.
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
