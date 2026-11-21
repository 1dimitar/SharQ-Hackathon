"""Surrogate QUBO terms for QOBLIB 08 network design. ShinkaEvolve edits the EVOLVE block."""

import time

# EVOLVE-BLOCK-START
import math
import numpy as np


def surrogate_terms(demand):
    """Build direct-link rewards and sparse, pressure-aware two-hop rewards."""
    traffic = np.maximum(np.asarray(demand, dtype=float), 0.0).copy()
    n = len(traffic)
    np.fill_diagonal(traffic, 0.0)

    scale = float(np.max(traffic)) if traffic.size else 0.0
    if scale <= 0.0:
        scale = 1.0
    normalized = traffic / scale

    linear = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                value = float(normalized[i, j])
                linear[(i, j)] = -(0.85 * value + 0.15 * math.sqrt(value))

    incoming = np.sum(traffic, axis=0)
    outgoing = np.sum(traffic, axis=1)
    incident = np.maximum(incoming, outgoing)
    max_incident = float(np.max(incident)) if incident.size else 0.0
    if max_incident <= 0.0:
        max_incident = 1.0
    node_pressure = np.maximum(0.65, incident / max_incident)

    # Give ordinary OD pairs a small relay budget, reserving extra choices
    # for the largest demands.
    positive_demands = normalized[normalized > 0.0]
    high_cutoff = (
        float(np.quantile(positive_demands, 0.80))
        if positive_demands.size else float("inf")
    )
    base_relays = min(n - 2, max(1, n // 5))
    extra_relays = min(2, max(0, n - 2 - base_relays))

    quadratic = {}
    for i in range(n):
        for k in range(n):
            if i == k:
                continue
            value = float(normalized[i, k])
            if value <= 0.0:
                continue

            support = 0.70 * value + 0.30 * math.sqrt(value)
            candidates = []
            for j in range(n):
                if j == i or j == k:
                    continue

                a = float(normalized[i, j])
                b = float(normalized[j, k])
                pressure = float(node_pressure[j]) * (1.0 + 0.5 * (a + b))

                # A weak leg should not be compensated fully by a strong one.
                affinity = min(a, b) + 0.15 * max(a, b) + 0.01
                candidates.append((affinity / pressure, j, pressure))

            candidates.sort(key=lambda item: (-item[0], item[1]))
            relay_budget = base_relays
            if value >= high_cutoff:
                relay_budget += extra_relays
            for _, j, pressure in candidates[:relay_budget]:
                first, second = (i, j), (j, k)
                key = (first, second) if first < second else (second, first)
                quadratic[key] = -0.30 * support / pressure

    penalty_weight = 1.5
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
