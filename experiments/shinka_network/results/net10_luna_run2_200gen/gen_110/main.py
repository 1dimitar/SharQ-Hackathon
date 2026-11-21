"""Surrogate QUBO terms for QOBLIB 08 network design. ShinkaEvolve edits the EVOLVE block."""

import time

# EVOLVE-BLOCK-START
import numpy as np


def _capped_allocation(weights, target=2.0):
    """Allocate target units proportionally, with each entry capped at one."""
    weights = np.asarray(weights, dtype=float)
    result = np.zeros_like(weights)
    active = weights > 0.0
    if not np.any(active):
        return result

    w = weights[active]
    target = min(float(target), float(len(w)))
    lo, hi = 0.0, 1.0
    while float(np.minimum(1.0, hi * w).sum()) < target:
        hi *= 2.0

    for _ in range(48):
        mid = (lo + hi) * 0.5
        if float(np.minimum(1.0, mid * w).sum()) < target:
            lo = mid
        else:
            hi = mid

    result[active] = np.minimum(1.0, hi * w)
    return result


def surrogate_terms(demand):
    """Build direct-link rewards and endpoint-saturation-weighted path rewards."""
    traffic = np.maximum(np.asarray(demand, dtype=float), 0.0).copy()
    n = traffic.shape[0]
    np.fill_diagonal(traffic, 0.0)

    scale = float(np.max(traffic)) if traffic.size else 0.0
    if scale <= 0.0:
        scale = 1.0
    normalized = traffic / scale

    # Estimate direct-link selection propensity. Each row and column has two
    # units of fractional capacity, with no individual link assigned above 1.
    affinity = np.sqrt(normalized) + 0.03
    np.fill_diagonal(affinity, 0.0)
    row_propensity = np.zeros((n, n), dtype=float)
    col_propensity = np.zeros((n, n), dtype=float)
    for i in range(n):
        row_propensity[i] = _capped_allocation(affinity[i])
    for j in range(n):
        col_propensity[:, j] = _capped_allocation(affinity[:, j])

    # A link is plausible as a direct choice only when it ranks well at both
    # endpoints. The geometric combination avoids favoring edges that are
    # attractive to only their source or only their destination.
    direct_propensity = np.sqrt(row_propensity * col_propensity)

    linear = {
        (i, j): -float(normalized[i, j])
        for i in range(n)
        for j in range(n)
        if i != j
    }

    incident = traffic.sum(axis=0) + traffic.sum(axis=1)
    mean_incident = float(np.mean(incident))
    if mean_incident > 0.0:
        node_pressure = np.sqrt(np.maximum(incident, 0.0) / mean_incident)
    else:
        node_pressure = np.ones(n, dtype=float)
    node_pressure = np.maximum(node_pressure, 0.7)

    quadratic = {}
    path_strength = 0.50
    for i in range(n):
        for k in range(n):
            if i == k or traffic[i, k] <= 0.0:
                continue

            # Discount OD demand likely to receive a direct link; retain some
            # value because endpoint competition makes that estimate uncertain.
            residual = float(normalized[i, k]) * (
                1.0 - 0.85 * float(direct_propensity[i, k])
            )
            if residual <= 0.0:
                continue

            for j in range(n):
                if j == i or j == k:
                    continue

                edge_load = float(normalized[i, j] + normalized[j, k])
                weight = (
                    -path_strength * residual
                    / (float(node_pressure[j]) * (1.0 + 0.4 * edge_load))
                )

                e1, e2 = (i, j), (j, k)
                key = (e1, e2) if e1 < e2 else (e2, e1)
                quadratic[key] = weight

    # Degree penalties are added by the evaluator, not encoded in these terms.
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
