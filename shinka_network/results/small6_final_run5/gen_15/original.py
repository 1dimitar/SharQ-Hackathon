"""Surrogate QUBO terms for QOBLIB 08 network design. ShinkaEvolve edits the EVOLVE block."""

import time

# EVOLVE-BLOCK-START
import numpy as np


def surrogate_terms(demand):
    """Build demand-aware path rewards with a sparse backtracking penalty."""
    matrix = np.asarray(demand, dtype=float)
    n = len(matrix)

    traffic = np.maximum(matrix, 0.0).copy()
    np.fill_diagonal(traffic, 0.0)
    scale = float(np.max(traffic)) if traffic.size else 0.0
    if scale <= 0.0:
        scale = 1.0

    normalized = traffic / scale
    roots = np.sqrt(normalized)
    linear = {
        (i, j): -float(0.75 * normalized[i, j] + 0.25 * roots[i, j])
        for i in range(n)
        for j in range(n)
        if i != j
    }

    incident = np.maximum(traffic.sum(axis=0), traffic.sum(axis=1))
    max_incident = float(np.max(incident)) if incident.size else 0.0
    if max_incident <= 0.0:
        max_incident = 1.0
    node_pressure = np.maximum(0.65, incident / max_incident)

    positive = traffic[traffic > 0.0]
    cutoff = float(np.quantile(positive, 0.75)) if positive.size else 0.0
    quadratic = {}

    for i in range(n):
        for k in range(n):
            if i == k or traffic[i, k] <= 0.0:
                continue

            support = 0.70 * normalized[i, k] + 0.30 * roots[i, k]
            candidates = []
            for j in range(n):
                if j == i or j == k:
                    continue
                edge_pressure = normalized[i, j] + normalized[j, k]
                pressure = float(node_pressure[j])
                affinity = 0.25 + edge_pressure
                rank = affinity / (pressure ** 0.75)
                discount = pressure * (1.0 + 0.5 * edge_pressure)
                candidates.append((float(rank), j, float(discount)))

            candidates.sort(reverse=True)
            keep = min(len(candidates), 4 if traffic[i, k] >= cutoff else 2)
            selected = set()

            for position, (_, j, discount) in enumerate(candidates[:keep]):
                selected.add(j)
                rank_factor = 1.0 + 0.08 * (keep - position)
                weight = -0.46 * float(support) * rank_factor / discount
                first, second = (i, j), (j, k)
                key = (first, second) if first < second else (second, first)
                quadratic[key] = weight

            # Retain one alternative selected by the full-path model's
            # pressure discount, unless it was already selected by affinity.
            remaining = [candidate for candidate in candidates if candidate[1] not in selected]
            if remaining:
                _, j, discount = min(remaining, key=lambda candidate: (candidate[2], candidate[1]))
                first, second = (i, j), (j, k)
                key = (first, second) if first < second else (second, first)
                quadratic[key] = -0.30 * float(support) / discount

    # Reciprocal links consume scarce two-hop expansion: i->j->i carries
    # no source-destination traffic. Unlike another path reward, this term
    # discourages a structural source of poor typical-sample connectivity.
    mean_direct = -float(sum(linear.values())) / max(1, n * (n - 1))
    depth_scale = max(
        0.75,
        float(np.log2(max(n, 2))) / float(np.log2(10.0)),
    )
    backtrack_weight = 0.65 * mean_direct * depth_scale

    if backtrack_weight > 0.0:
        for i in range(n):
            for j in range(i + 1, n):
                # Preserve reciprocal links when substantial traffic in
                # BOTH directions makes them genuinely useful shortcuts.
                mutual = min(
                    float(normalized[i, j]),
                    float(normalized[j, i]),
                )
                quadratic[((i, j), (j, i))] = float(
                    backtrack_weight * (1.0 - 0.45 * mutual)
                )

    # Keep constraint enforcement modest, but slightly stronger as the
    # number of variables and incident route interactions grows.
    penalty_weight = min(1.45, 1.30 + 0.008 * max(0, n - 10))
    return linear, quadratic, float(penalty_weight)


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
