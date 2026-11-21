"""Surrogate QUBO terms for QOBLIB 08 network design. ShinkaEvolve edits the EVOLVE block."""

import time

# EVOLVE-BLOCK-START
import numpy as np


def surrogate_terms(demand):
    """Build direct-link rewards and pressure-discounted two-hop rewards."""
    traffic = np.asarray(demand, dtype=float)
    n = len(traffic)

    # Negative entries and self-demands do not represent useful network traffic.
    traffic = np.maximum(traffic, 0.0).copy()
    np.fill_diagonal(traffic, 0.0)

    scale = float(np.max(traffic)) if traffic.size else 0.0
    if scale <= 0.0:
        scale = 1.0
    normalized = traffic / scale

    # Direct service is the most reliable way to reduce a link's routed load.
    # The concave component also gives moderate demands a useful, but smaller,
    # influence when selecting the limited two outgoing/incoming links.
    linear = {}
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = float(normalized[i, j])
            linear[(i, j)] = -(0.80 * d + 0.20 * np.sqrt(d))

    # A relay with large incoming or outgoing demand is a less attractive
    # place to route additional traffic. Use the larger direction as a simple
    # estimate of its transit pressure.
    incoming = np.sum(traffic, axis=0)
    outgoing = np.sum(traffic, axis=1)
    incident = np.maximum(incoming, outgoing)
    max_incident = float(np.max(incident)) if incident.size else 0.0
    if max_incident <= 0.0:
        max_incident = 1.0
    node_pressure = incident / max_incident

    # A two-hop route is useful as an alternate to a direct link, but should
    # not be rewarded as strongly as direct service. The pressure discount
    # reduces the reward for routes through busy relays or using already
    # heavily demanded edges.
    quadratic = {}
    path_strength = 0.30
    relay_budget = min(n - 2, max(2, int(np.ceil(np.sqrt(n)))))

    for i in range(n):
        for k in range(n):
            if i == k:
                continue

            d = float(normalized[i, k])
            if d <= 0.0:
                continue

            path_support = 0.72 * d + 0.28 * np.sqrt(d)
            candidates = []

            for j in range(n):
                if j == i or j == k:
                    continue

                a = float(normalized[i, j])
                b = float(normalized[j, k])
                edge_pressure = a + b
                relay_discount = 0.60 + 0.80 * float(node_pressure[j])
                edge_discount = 1.0 + 0.35 * edge_pressure

                weight = (
                    -path_strength
                    * path_support
                    / (relay_discount * edge_discount)
                )

                # Rank by support for both legs, with a floor so that
                # low-demand legs can still serve as transit links.
                affinity = 0.20 + 0.80 * (2.0 * a * b / (a + b + 1.0e-12))
                rank = affinity / (relay_discount * edge_discount)
                candidates.append((rank, j, weight))

            candidates.sort(key=lambda item: (-item[0], item[1]))
            for _, j, weight in candidates[:relay_budget]:
                first, second = (i, j), (j, k)
                key = (first, second) if first < second else (second, first)
                quadratic[key] = weight

    # The evaluator supplies the degree penalties and repairs its samples.
    # Keep the penalty moderate so the sparse routing objective remains visible.
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
