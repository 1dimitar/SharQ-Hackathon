"""Surrogate QUBO terms for QOBLIB 08 network design. ShinkaEvolve edits the EVOLVE block."""

import time

# EVOLVE-BLOCK-START
import math
import numpy as np


def surrogate_terms(demand):
    """Build direct-link rewards and sparse, pressure-aware relay rewards."""
    traffic = np.maximum(np.asarray(demand, dtype=float), 0.0).copy()
    n = len(traffic)
    np.fill_diagonal(traffic, 0.0)

    scale = float(np.max(traffic)) if traffic.size else 0.0
    if scale > 0.0:
        traffic /= scale

    linear = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                value = float(traffic[i, j])
                linear[(i, j)] = -(0.75 * value + 0.25 * math.sqrt(value))

    incident = np.maximum(traffic.sum(axis=0), traffic.sum(axis=1))
    max_incident = float(np.max(incident)) if incident.size else 0.0
    if max_incident > 0.0:
        pressure = np.maximum(0.65, incident / max_incident)
    else:
        pressure = np.ones(n, dtype=float)

    positive = traffic[traffic > 0.0]
    cutoff = float(np.quantile(positive, 0.80)) if positive.size else float("inf")
    relay_count = min(n - 2, max(2, n // 4))
    extra_relays = min(2, max(0, n - 2 - relay_count))

    quadratic = {}
    for i in range(n):
        for k in range(n):
            if i == k:
                continue
            value = float(traffic[i, k])
            if value <= 0.0:
                continue

            root = math.sqrt(value)
            support = 0.725 * value + 0.275 * root
            candidates = []
            for j in range(n):
                if j == i or j == k:
                    continue

                a = float(traffic[i, j])
                b = float(traffic[j, k])
                p = float(pressure[j])
                geometric_rank = math.sqrt((0.1 + a) * (0.1 + b)) / (0.75 + p)
                leg_pressure = p * (1.0 + 0.5 * (a + b))
                balanced_rank = (min(a, b) + 0.25 * max(a, b) + 0.01) / leg_pressure

                # Both scores favor two useful legs; the second more strongly
                # discounts links already carrying substantial local demand.
                rank = 0.5 * geometric_rank + 0.5 * balanced_rank
                candidates.append((rank, j, geometric_rank, leg_pressure))

            candidates.sort(key=lambda item: (-item[0], item[1]))
            count = relay_count + (extra_relays if value >= cutoff else 0)
            for _, j, geometric_rank, leg_pressure in candidates[:count]:
                rank_factor = 0.8 + 0.2 * geometric_rank / (1.0 + geometric_rank)
                original_weight = 0.28 * rank_factor / (0.75 + float(pressure[j]))
                crossover_weight = 0.30 / leg_pressure
                weight = -support * (0.5 * original_weight + 0.5 * crossover_weight)

                first, second = (i, j), (j, k)
                key = (first, second) if first < second else (second, first)
                quadratic[key] = weight

    return linear, quadratic, 1.5


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
