"""Surrogate QUBO terms for QOBLIB 08 network design. ShinkaEvolve edits the EVOLVE block."""

import time

# EVOLVE-BLOCK-START
import numpy as np


def surrogate_terms(demand):
    """Build direct-link and pressure-discounted two-hop cost terms."""
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

    # Direct links retain a strong preference for large OD demands, with
    # a modest concave component to give smaller demands some influence.
    linear = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                d = float(q[i, j])
                linear[(i, j)] = -(0.80 * d + 0.20 * np.sqrt(d))

    # Estimate how costly it is to use each node as a relay.
    incident = np.maximum(np.sum(q, axis=0), np.sum(q, axis=1))
    largest = float(np.max(incident)) if incident.size else 0.0
    if largest > 0.0:
        pressure = np.maximum(0.60, incident / largest)
    else:
        pressure = np.full(n, 0.60, dtype=float)

    quadratic = {}
    path_strength = 0.34
    relay_budget = min(n - 2, max(2, int(np.sqrt(n)) + 1))
    pressure_alpha = 0.75

    for i in range(n):
        for k in range(n):
            if i == k:
                continue

            d = float(q[i, k])
            if d <= 0.0:
                continue
            support = 0.75 * d + 0.25 * np.sqrt(d)

            candidates = []
            for j in range(n):
                if j == i or j == k:
                    continue

                a = float(q[i, j])
                b = float(q[j, k])

                # Reward relays with useful capacity on both legs, while
                # discounting nodes already under substantial traffic pressure.
                affinity = min(a, b) + 0.30 * max(a, b)
                edge_discount = 1.0 + 0.20 * (a + b)
                quality = affinity / (
                    (0.70 + float(pressure[j])) ** pressure_alpha
                    * edge_discount
                )
                candidates.append((quality, j))

            candidates.sort(key=lambda item: (-item[0], item[1]))
            for quality, j in candidates[:relay_budget]:
                if quality <= 0.0:
                    continue

                weight = -path_strength * support * quality
                first, second = (i, j), (j, k)
                key = (first, second) if first < second else (second, first)
                quadratic[key] = weight

    # Evaluator supplies the actual degree penalties.
    penalty_weight = 2.0
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
