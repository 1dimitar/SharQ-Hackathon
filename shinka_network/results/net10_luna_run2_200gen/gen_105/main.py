"""Surrogate QUBO terms for QOBLIB 08 network design. ShinkaEvolve edits the EVOLVE block."""

import time

# EVOLVE-BLOCK-START
import numpy as np


def _normalise_demand(demand):
    """Return nonnegative, off-diagonal demand scaled to [0, 1]."""
    d = np.asarray(demand, dtype=float)
    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("demand must be a square matrix")

    d = np.maximum(d, 0.0).copy()
    np.fill_diagonal(d, 0.0)
    scale = float(np.max(d)) if d.size else 0.0
    if scale > 0.0:
        d /= scale
    return d


def _direct_link_terms(d):
    """Reward direct service, with a small endpoint-traffic signal."""
    n = d.shape[0]
    row = np.sum(d, axis=1)
    col = np.sum(d, axis=0)
    endpoint_scale = max(float(np.max(row)), float(np.max(col)), 1.0)

    linear = {}
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            endpoint = (float(row[i]) + float(col[j])) / (2.0 * endpoint_scale)
            value = float(d[i, j])
            linear[(i, j)] = -(0.90 * value + 0.10 * endpoint)
    return linear


def _indirect_route_terms(d):
    """Reward useful two-hop paths, discounting busy edges and transit nodes."""
    n = d.shape[0]
    node_load = np.sum(d, axis=0) + np.sum(d, axis=1)
    mean_load = float(np.mean(node_load)) if n else 0.0

    if mean_load > 0.0:
        transit_capacity = 2.0 * mean_load / (mean_load + node_load)
    else:
        transit_capacity = np.ones(n, dtype=float)

    quadratic = {}
    edge_discount = 0.85
    indirect_cap = 0.55

    for i in range(n):
        for k in range(n):
            traffic = float(d[i, k])
            if i == k or traffic <= 0.0:
                continue

            candidates = []
            total_quality = 0.0
            for j in range(n):
                if j == i or j == k:
                    continue

                # Penalise routes whose component edges already have high
                # direct demand; also avoid concentrating transit at busy nodes.
                edge_load = float(d[i, j]) + float(d[j, k])
                quality = float(transit_capacity[j]) / (
                    1.0 + edge_discount * edge_load
                )
                if quality > 0.0:
                    candidates.append((j, quality))
                    total_quality += quality

            if total_quality <= 0.0:
                continue

            # The aggregate reward is capped per OD pair, so adding more
            # candidate intermediates does not overwhelm direct-link terms.
            aggregate = min(0.30 * traffic, indirect_cap * traffic)
            for j, quality in candidates:
                weight = -aggregate * quality / total_quality
                quadratic[((i, j), (j, k))] = float(weight)

    return quadratic


def surrogate_terms(demand):
    """Build linear and quadratic QUBO cost terms."""
    d = _normalise_demand(demand)
    linear = _direct_link_terms(d)
    quadratic = _indirect_route_terms(d)
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
