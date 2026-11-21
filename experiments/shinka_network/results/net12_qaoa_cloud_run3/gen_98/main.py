"""Surrogate QUBO terms for QOBLIB 08 network design. ShinkaEvolve edits the EVOLVE block."""

import time

# EVOLVE-BLOCK-START
import numpy as np


def _prepare_traffic(demand):
    """Return nonnegative, finite off-diagonal traffic and its scale."""
    traffic = np.asarray(demand, dtype=float)
    if traffic.ndim != 2 or traffic.shape[0] != traffic.shape[1]:
        raise ValueError("demand must be a square matrix")

    traffic = np.nan_to_num(traffic, nan=0.0, posinf=0.0, neginf=0.0)
    traffic = np.maximum(traffic, 0.0).copy()
    np.fill_diagonal(traffic, 0.0)

    scale = float(np.max(traffic)) if traffic.size else 0.0
    if scale <= 0.0:
        scale = 1.0
    return traffic, traffic / scale, scale


def _direct_link_terms(normalized):
    """Build concave direct-service rewards for every possible link."""
    n = normalized.shape[0]
    linear = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                value = float(normalized[i, j])
                linear[(i, j)] = -(0.75 * value + 0.25 * np.sqrt(value))
    return linear


def _relay_context(traffic):
    """Estimate node transit pressure and the high-demand relay cutoff."""
    n = traffic.shape[0]
    incoming = np.sum(traffic, axis=0)
    outgoing = np.sum(traffic, axis=1)
    incident = np.maximum(incoming, outgoing)

    largest = float(np.max(incident)) if incident.size else 0.0
    if largest > 0.0:
        pressure = incident / largest
    else:
        pressure = np.zeros(n, dtype=float)

    positive = traffic[traffic > 0.0]
    cutoff = float(np.percentile(positive, 80)) if positive.size else float("inf")
    return pressure, cutoff


def _relay_candidates(i, k, normalized, pressure, limit=2):
    """Rank feasible intermediate nodes for the OD pair i -> k."""
    n = normalized.shape[0]
    candidates = []

    for j in range(n):
        if j == i or j == k:
            continue

        # Favor relays with lower overall pressure. Mildly prefer relays whose
        # two legs already have demand, without making that a hard requirement.
        leg_a = float(normalized[i, j])
        leg_b = float(normalized[j, k])
        leg_affinity = np.sqrt((0.05 + leg_a) * (0.05 + leg_b))
        quality = (0.75 + 0.25 * leg_affinity) / (0.7 + float(pressure[j]))
        candidates.append((quality, j))

    candidates.sort(key=lambda item: (-item[0], item[1]))
    return candidates[:limit]


def _relay_terms(traffic, normalized, pressure, cutoff):
    """Assemble sparse, demand-weighted two-hop interactions."""
    n = traffic.shape[0]
    quadratic = {}
    path_strength = 0.34

    for i in range(n):
        for k in range(n):
            if i == k:
                continue

            demand_ik = float(traffic[i, k])
            if demand_ik <= 0.0:
                continue

            value = float(normalized[i, k])
            support = 0.70 * value + 0.30 * np.sqrt(value)
            demand_factor = 1.1 if demand_ik >= cutoff else 0.9
            chosen = _relay_candidates(i, k, normalized, pressure, limit=2)

            quality_sum = sum(q for q, _ in chosen)
            if quality_sum <= 0.0:
                continue

            for quality, j in chosen:
                first, second = (i, j), (j, k)
                key = (first, second) if first < second else (second, first)
                weight = (
                    -path_strength
                    * support
                    * demand_factor
                    * quality
                    / quality_sum
                )
                quadratic[key] = quadratic.get(key, 0.0) + weight

    return quadratic


def surrogate_terms(demand):
    """Return direct-link and sparse relay QUBO terms plus degree penalty."""
    traffic, normalized, _ = _prepare_traffic(demand)
    pressure, cutoff = _relay_context(traffic)

    linear = _direct_link_terms(normalized)
    quadratic = _relay_terms(traffic, normalized, pressure, cutoff)

    # Degree penalties are applied by the evaluator, not included in these terms.
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
