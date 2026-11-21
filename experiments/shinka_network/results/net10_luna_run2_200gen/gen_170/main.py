"""Surrogate QUBO terms for QOBLIB 08 network design. ShinkaEvolve edits the EVOLVE block."""

import time

# EVOLVE-BLOCK-START
import numpy as np


def _prepare_demand(demand):
    """Return a finite, nonnegative, zero-diagonal traffic matrix and scale."""
    traffic = np.asarray(demand, dtype=float)
    n = len(traffic)

    traffic = np.nan_to_num(traffic, nan=0.0, posinf=0.0, neginf=0.0)
    traffic = np.maximum(traffic, 0.0).copy()
    np.fill_diagonal(traffic, 0.0)

    scale = float(np.max(traffic)) if traffic.size else 0.0
    if scale <= 0.0:
        scale = 1.0

    return traffic, n, scale


def _direct_link_terms(normalized):
    """Score direct service while retaining signal for moderate demands."""
    n = len(normalized)
    linear = {}

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            demand_level = float(normalized[i, j])
            utility = 0.75 * demand_level + 0.25 * np.sqrt(demand_level)
            linear[(i, j)] = -float(utility)

    return linear


def _relay_pressures(traffic):
    """Estimate how costly each node may be as a transit point."""
    incident = np.sum(traffic, axis=0) + np.sum(traffic, axis=1)
    mean_incident = float(np.mean(incident)) if incident.size else 0.0

    if mean_incident <= 0.0:
        return np.ones(len(traffic), dtype=float)

    return np.maximum(0.65, incident / mean_incident)


def _two_hop_terms(normalized, relay_pressure):
    """Reward useful two-link paths, discounted for relay and edge pressure."""
    n = len(normalized)
    quadratic = {}
    path_strength = 0.50

    for source in range(n):
        for target in range(n):
            if source == target:
                continue

            demand_level = float(normalized[source, target])
            if demand_level <= 0.0:
                continue

            # Concave support keeps smaller OD flows relevant without making
            # path rewards proportional only to the largest demands.
            path_support = (
                0.70 * demand_level
                + 0.30 * np.sqrt(demand_level)
            )

            for relay in range(n):
                if relay == source or relay == target:
                    continue

                edge_pressure = float(
                    normalized[source, relay] + normalized[relay, target]
                )
                weight = (
                    -path_strength
                    * path_support
                    / (
                        float(relay_pressure[relay])
                        * (1.0 + 0.85 * edge_pressure)
                    )
                )

                first = (source, relay)
                second = (relay, target)
                key = (first, second) if first < second else (second, first)
                quadratic[key] = float(weight)

    return quadratic


def surrogate_terms(demand):
    """Build linear, quadratic, and degree-penalty terms for the topology QUBO."""
    traffic, n, scale = _prepare_demand(demand)
    if n == 0:
        return {}, {}, 1.0

    normalized = traffic / scale
    linear = _direct_link_terms(normalized)
    relay_pressure = _relay_pressures(traffic)
    quadratic = _two_hop_terms(normalized, relay_pressure)

    # The evaluator adds the in- and out-degree penalties.
    return linear, quadratic, 1.0


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
