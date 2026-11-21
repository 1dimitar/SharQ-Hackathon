"""Surrogate QUBO terms for QOBLIB 08 network design. ShinkaEvolve edits the EVOLVE block."""

import time

# EVOLVE-BLOCK-START
import numpy as np


def _prepare_demand(demand):
    """Return nonnegative, off-diagonal demand normalized to [0, 1]."""
    d = np.asarray(demand, dtype=float).copy()
    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("demand must be a square matrix")
    d = np.maximum(d, 0.0)
    np.fill_diagonal(d, 0.0)

    scale = float(np.max(d)) if d.size else 0.0
    if scale > 0.0:
        d /= scale
    return d


def _direct_terms(d):
    """Score direct links and retain a small endpoint-throughput signal."""
    n = d.shape[0]
    row = d.sum(axis=1)
    col = d.sum(axis=0)
    node_scale = max(float(np.max(row)), float(np.max(col)), 1.0)

    linear = {}
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            endpoint_pressure = (
                float(row[i]) + float(col[j])
            ) / (2.0 * node_scale)
            linear[(i, j)] = -(
                0.90 * float(d[i, j]) + 0.10 * endpoint_pressure
            )
    return linear


def _path_terms(d, linear):
    """Create selective two-hop rewards, capped per OD pair."""
    n = d.shape[0]
    if n < 3:
        return {}

    node_load = d.sum(axis=0) + d.sum(axis=1)
    mean_load = float(np.mean(node_load))
    if mean_load > 0.0:
        transit = 2.0 * mean_load / (mean_load + node_load)
    else:
        transit = np.ones(n, dtype=float)

    candidate_count = min(4, n - 2)
    quadratic = {}

    for i in range(n):
        for k in range(n):
            traffic = float(d[i, k])
            if i == k or traffic <= 0.0:
                continue

            candidates = []
            for j in range(n):
                if j == i or j == k:
                    continue

                # Prefer spare transit nodes and avoid excessively favoring
                # paths whose constituent links already have direct traffic.
                edge_load = float(d[i, j]) + float(d[j, k])
                direct_support = edge_load
                quality = (
                    float(transit[j])
                    * (0.90 + 0.10 * direct_support)
                    / np.sqrt(1.0 + 0.50 * edge_load)
                )
                candidates.append((quality, j))

            candidates.sort(reverse=True)
            chosen = candidates[:candidate_count]
            quality_sum = sum(q for q, _ in chosen)
            if quality_sum <= 0.0:
                continue

            # Limit aggregate indirect support relative to both the OD's
            # direct demand and its direct-link surrogate reward.
            direct_reward = abs(float(linear[(i, k)]))
            total_reward = min(
                0.45 * traffic,
                0.40 * direct_reward,
            )
            if total_reward <= 0.0:
                continue

            for quality, j in chosen:
                weight = -total_reward * quality / quality_sum
                quadratic[((i, j), (j, k))] = float(weight)

    return quadratic


def surrogate_terms(demand):
    """Return linear terms, quadratic terms, and evaluator penalty weight."""
    d = _prepare_demand(demand)
    linear = _direct_terms(d)
    quadratic = _path_terms(d, linear)
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
