"""Surrogate QUBO terms for QOBLIB 08 network design. ShinkaEvolve edits the EVOLVE block."""

import time

# EVOLVE-BLOCK-START
import numpy as np


def surrogate_terms(demand):
    """Build direct-link rewards and focused two-hop relay rewards."""
    traffic = np.asarray(demand, dtype=float)
    n = len(traffic)

    traffic = np.maximum(traffic, 0.0).copy()
    np.fill_diagonal(traffic, 0.0)

    scale = float(np.max(traffic)) if traffic.size else 0.0
    if scale <= 0.0:
        scale = 1.0
    normalized = traffic / scale

    # Preserve a preference for direct service, while retaining a modest
    # concave contribution from medium-sized demands.
    linear = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                d = float(normalized[i, j])
                linear[(i, j)] = -(0.80 * d + 0.20 * np.sqrt(d))

    incoming = np.sum(traffic, axis=0)
    outgoing = np.sum(traffic, axis=1)
    incident = np.maximum(incoming, outgoing)
    max_incident = float(np.max(incident)) if incident.size else 0.0
    if max_incident <= 0.0:
        max_incident = 1.0
    pressure = incident / max_incident

    # Restrict added relay support to the upper demand decile.
    od_values = normalized[~np.eye(n, dtype=bool)]
    positive = od_values[od_values > 0.0]
    cutoff = float(np.percentile(positive, 90.0)) if positive.size else 0.0

    quadratic = {}
    path_strength = 0.30
    relay_cap = 3

    for i in range(n):
        for k in range(n):
            if i == k:
                continue
            d = float(normalized[i, k])
            if d <= 0.0 or d < cutoff:
                continue

            path_support = 0.72 * d + 0.28 * np.sqrt(d)
            candidates = []
            for j in range(n):
                if j == i or j == k:
                    continue
                first = float(normalized[i, j])
                second = float(normalized[j, k])
                edge_pressure = first + second
                # Favor relays with useful connection affinity, but discount
                # nodes and legs likely to attract substantial transit load.
                affinity = np.sqrt((0.10 + first) * (0.10 + second))
                relay_discount = (0.75 + float(pressure[j]) / 1.75) ** 0.75
                quality = affinity / (
                    relay_discount * (1.0 + 0.35 * edge_pressure)
                )
                candidates.append((quality, j))

            candidates.sort(key=lambda item: (-item[0], item[1]))
            chosen = candidates[:relay_cap]
            total_quality = sum(q for q, _ in chosen)
            if total_quality <= 0.0:
                continue

            for quality, j in chosen:
                weight = -path_strength * path_support * quality / total_quality
                a, b = (i, j), (j, k)
                key = (a, b) if a < b else (b, a)
                quadratic[key] = quadratic.get(key, 0.0) + weight

    # Degree penalties are supplied by the evaluator.
    penalty_weight = 3.0
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
