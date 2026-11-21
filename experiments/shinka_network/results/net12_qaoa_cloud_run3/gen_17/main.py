"""Surrogate QUBO terms for QOBLIB 08 network design. ShinkaEvolve edits the EVOLVE block."""

import time

# EVOLVE-BLOCK-START
import numpy as np


def surrogate_terms(demand):
    """Build direct-link, selected-relay, and route-diversity QUBO terms."""
    traffic = np.asarray(demand, dtype=float).copy()
    n = traffic.shape[0]
    traffic = np.maximum(traffic, 0.0)
    np.fill_diagonal(traffic, 0.0)

    max_traffic = float(np.max(traffic)) if traffic.size else 0.0
    if max_traffic <= 0.0:
        max_traffic = 1.0
    q = traffic / max_traffic

    # Direct service remains valuable, with a concave component retaining
    # influence for moderate demands.
    linear = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                value = float(q[i, j])
                linear[(i, j)] = -(0.75 * value + 0.25 * np.sqrt(value))

    row_sum = np.sum(q, axis=1)
    col_sum = np.sum(q, axis=0)
    row_scale = max(float(np.max(row_sum)), 1.0)
    col_scale = max(float(np.max(col_sum)), 1.0)
    row_profile = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    col_profile = q / np.maximum(np.linalg.norm(q, axis=0, keepdims=True), 1e-12)

    quadratic = {}

    def add_pair(e1, e2, weight):
        if e1 == e2 or weight == 0.0:
            return
        key = (e1, e2) if e1 < e2 else (e2, e1)
        quadratic[key] = quadratic.get(key, 0.0) + float(weight)

    # Keep only the two best relay choices per OD pair. The score favors
    # routes with useful first/last legs and discounts highly loaded relays.
    relay_pressure = np.maximum(row_sum / row_scale, col_sum / col_scale)
    for i in range(n):
        for k in range(n):
            if i == k or q[i, k] <= 0.0:
                continue

            candidates = []
            for j in range(n):
                if j == i or j == k:
                    continue
                first_affinity = q[i, j] + 0.12 * row_sum[i] / max(n - 1, 1)
                second_affinity = q[j, k] + 0.12 * col_sum[k] / max(n - 1, 1)
                leg_quality = np.sqrt(max(first_affinity, 0.0) *
                                      max(second_affinity, 0.0))
                score = leg_quality / (0.75 + float(relay_pressure[j]))
                candidates.append((score, j))

            candidates.sort(reverse=True)
            if not candidates:
                continue
            best_score = candidates[0][0]
            if best_score <= 0.0:
                continue

            for score, j in candidates[:2]:
                # Normalize relay strength within this OD pair so that the
                # selected alternatives share a bounded amount of influence.
                relative = score / best_score
                weight = -0.34 * float(q[i, k]) * (0.65 + 0.35 * relative)
                add_pair((i, j), (j, k), weight)

    # Route-diversity synergy: links from the same source (or into the same
    # destination) are complementary when their neighboring nodes have
    # similar traffic profiles. This encourages alternative ways to serve
    # related OD traffic rather than rewarding isolated high-demand links.
    diversity_strength = 0.045
    for i in range(n):
        links = [(i, j) for j in range(n) if j != i]
        for a in range(len(links)):
            j = links[a][1]
            for b in range(a + 1, len(links)):
                k = links[b][1]
                overlap = float(np.dot(row_profile[j], row_profile[k]))
                add_pair(links[a], links[b], -diversity_strength * overlap)

    for j in range(n):
        links = [(i, j) for i in range(n) if i != j]
        for a in range(len(links)):
            i = links[a][0]
            for b in range(a + 1, len(links)):
                k = links[b][0]
                overlap = float(np.dot(col_profile[:, i], col_profile[:, k]))
                add_pair(links[a], links[b], -diversity_strength * overlap)

    # Degree penalties are added by the evaluator.
    penalty_weight = 1.0
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
