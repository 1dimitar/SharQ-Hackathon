"""Surrogate QUBO terms for QOBLIB 08 network design. ShinkaEvolve edits the EVOLVE block."""

import time

# EVOLVE-BLOCK-START
import numpy as np


def _prepare_demand(demand):
    """Return nonnegative off-diagonal traffic, scaled to [0, 1]."""
    traffic = np.maximum(np.asarray(demand, dtype=float), 0.0).copy()
    np.fill_diagonal(traffic, 0.0)

    scale = float(np.max(traffic)) if traffic.size else 0.0
    if scale > 0.0:
        traffic /= scale
    return traffic


def _direct_link_terms(traffic):
    """Reward directly connecting OD pairs with substantial demand."""
    n = traffic.shape[0]
    linear = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                value = float(traffic[i, j])
                linear[(i, j)] = -(0.75 * value + 0.25 * np.sqrt(value))
    return linear


def _relay_pressures(traffic):
    """Estimate how costly each node may be as a transit relay."""
    incoming = np.sum(traffic, axis=0)
    outgoing = np.sum(traffic, axis=1)
    incident = np.maximum(incoming, outgoing)

    maximum = float(np.max(incident)) if incident.size else 0.0
    if maximum <= 0.0:
        maximum = 1.0
    return np.maximum(0.65, incident / maximum)


def _relay_terms(traffic, pressure):
    """Construct sparse rewards for selected two-link routes."""
    n = traffic.shape[0]
    # Keep the support compact as n grows, while retaining multiple relay
    # options even for small instances.
    relay_budget = min(n - 2, max(1, n // 5))
    quadratic = {}

    for source in range(n):
        for target in range(n):
            if source == target:
                continue

            od = float(traffic[source, target])
            if od <= 0.0:
                continue

            support = 0.70 * od + 0.30 * np.sqrt(od)
            candidates = []

            for relay in range(n):
                if relay == source or relay == target:
                    continue

                left = float(traffic[source, relay])
                right = float(traffic[relay, target])

                # Geometric leg affinity favors relays whose two legs are
                # both plausible, rather than a relay with just one strong leg.
                affinity = np.sqrt((0.08 + left) * (0.08 + right))
                affinity *= 0.75 + 0.25 * (left + right) / (2.0 + left + right)
                rank = affinity / (0.80 + float(pressure[relay]))
                candidates.append((rank, relay))

            candidates.sort(key=lambda item: (-item[0], item[1]))

            for rank, relay in candidates[:relay_budget]:
                # A modest rank adjustment differentiates alternatives without
                # allowing one relay to dominate the entire surrogate.
                rank_factor = 0.85 + 0.15 * rank / (1.0 + rank)
                weight = -0.34 * support * rank_factor / (
                    0.80 + float(pressure[relay])
                )

                first = (source, relay)
                second = (relay, target)
                key = (first, second) if first < second else (second, first)
                quadratic[key] = quadratic.get(key, 0.0) + weight

    return quadratic


def surrogate_terms(demand):
    """Build direct-link and sparse relay-path cost terms."""
    traffic = _prepare_demand(demand)
    pressure = _relay_pressures(traffic)

    linear = _direct_link_terms(traffic)
    quadratic = _relay_terms(traffic, pressure)

    # Degree penalties are supplied separately by the evaluator.
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
