"""Surrogate QUBO terms for QOBLIB 08 network design. ShinkaEvolve edits the EVOLVE block."""

import time

# EVOLVE-BLOCK-START
import numpy as np


def _prepare_traffic(demand):
    """Return clipped traffic, its maximum scale, and normalized traffic."""
    traffic = np.maximum(np.asarray(demand, dtype=float), 0.0).copy()
    np.fill_diagonal(traffic, 0.0)

    n = traffic.shape[0]
    scale = float(np.max(traffic)) if traffic.size else 0.0
    if scale <= 0.0:
        scale = 1.0

    return traffic, scale, traffic / scale, n


def _direct_link_terms(normalized, n):
    """Score direct service of each OD pair."""
    linear = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                value = float(normalized[i, j])
                linear[(i, j)] = -(0.75 * value + 0.25 * np.sqrt(value))
    return linear


def _relay_pressures(traffic):
    """Estimate relay congestion from the larger directional incident load."""
    incoming = np.sum(traffic, axis=0)
    outgoing = np.sum(traffic, axis=1)
    incident = np.maximum(incoming, outgoing)

    peak = float(np.max(incident)) if incident.size else 0.0
    if peak <= 0.0:
        peak = 1.0

    return np.maximum(0.65, incident / peak)


def _relay_terms(traffic, normalized, scale, n):
    """Accumulate sparse, pressure-discounted two-hop path rewards."""
    pressure = _relay_pressures(traffic)
    quadratic = {}

    # Keep the interaction graph sparse while allowing a few alternatives
    # for each OD pair; tie-breaking by node index is deterministic.
    relay_budget = min(n - 2, max(1, n // 5))
    path_strength = 0.46

    for source in range(n):
        for target in range(n):
            if source == target:
                continue

            demand_value = float(normalized[source, target])
            if demand_value <= 0.0:
                continue

            support = 0.70 * demand_value + 0.30 * np.sqrt(demand_value)
            candidates = []

            for relay in range(n):
                if relay == source or relay == target:
                    continue

                edge_pressure = (
                    float(traffic[source, relay])
                    + float(traffic[relay, target])
                ) / scale
                reward = (
                    path_strength * support
                    / (float(pressure[relay]) * (1.0 + 0.5 * edge_pressure))
                )
                candidates.append((reward, relay))

            candidates.sort(key=lambda item: (-item[0], item[1]))

            for reward, relay in candidates[:relay_budget]:
                first = (source, relay)
                second = (relay, target)
                key = (first, second) if first < second else (second, first)
                quadratic[key] = quadratic.get(key, 0.0) - reward

    return quadratic


def surrogate_terms(demand):
    """Build direct-link and aggregated, pressure-discounted relay terms."""
    traffic, scale, normalized, n = _prepare_traffic(demand)

    linear = _direct_link_terms(normalized, n)
    quadratic = _relay_terms(traffic, normalized, scale, n)

    # Degree penalties are supplied by the evaluator.
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
