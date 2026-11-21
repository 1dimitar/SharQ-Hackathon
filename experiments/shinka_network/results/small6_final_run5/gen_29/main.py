"""Surrogate QUBO terms for QOBLIB 08 network design. ShinkaEvolve edits the EVOLVE block."""

import time

# EVOLVE-BLOCK-START
import numpy as np


def surrogate_terms(demand):
    """Build direct-link and selectively pressure-discounted two-hop terms."""
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

    # Estimate transit demand on plausible two-hop routes, using incident
    # load as an initial congestion estimate. Keep only the two best routes
    # per OD pair to avoid spreading load over unsupported intermediates.
    initial_pressure = np.maximum(0.65, incident / max_incident)
    transit_load = np.zeros(n, dtype=float)
    for i in range(n):
        for k in range(n):
            if i == k or traffic[i, k] <= 0.0:
                continue
            routes = []
            for j in range(n):
                if j == i or j == k:
                    continue
                evidence = float(normalized[i, j] + normalized[j, k])
                if evidence <= 0.02:
                    continue
                rank = (0.25 + evidence) / (float(initial_pressure[j]) ** 0.75)
                routes.append((rank, j, evidence))
            routes.sort(reverse=True)
            selected = routes[:2]
            if not selected:
                continue
            strengths = [
                rank * rank * (0.20 + evidence)
                for rank, _, evidence in selected
            ]
            total_strength = sum(strengths)
            routed_fraction = 0.35 + 0.65 * (1.0 - float(normalized[i, k]))
            for (_, j, _), strength in zip(selected, strengths):
                transit_load[j] += (
                    float(traffic[i, k]) * routed_fraction * strength / total_strength
                )

    projected_pressure = incident / max_incident + 0.35 * transit_load / max_incident
    node_pressure = np.maximum(0.65, np.minimum(1.8, projected_pressure))

    positive = traffic[traffic > 0.0]
    cutoff = float(np.quantile(positive, 0.75)) if positive.size else 0.0
    high_cutoff = float(np.quantile(positive, 0.90)) if positive.size else 0.0
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

            # Three-hop Taylor terms are omitted: pairwise fragments can
            # reward incomplete paths and misrank valid two-link exchanges.

    # Penalize a small number of demand-weighted redundant branch pairs.
    # Outbound profiles describe destinations; transposed profiles describe
    # origins for the analogous inbound correction.
    for inbound in (False, True):
        od = normalized.T if inbound else normalized
        profiles = np.sqrt(od).copy()
        np.fill_diagonal(profiles, 1.0)
        for center in range(n):
            weights = od[center]
            if float(weights.sum()) <= 0.0:
                continue
            gram = (profiles * weights[None, :]) @ profiles.T
            branches = [j for j in range(n) if j != center]
            normalizer = max(float(gram[j, j]) for j in branches)
            if normalizer <= 1e-12:
                continue

            redundant = []
            for position, j in enumerate(branches):
                for k in branches[position + 1:]:
                    overlap = float(gram[j, k]) / normalizer
                    if overlap > 0.15:
                        redundant.append((overlap, j, k))
            redundant.sort(reverse=True)

            for overlap, j, k in redundant[:3]:
                first = (j, center) if inbound else (center, j)
                second = (k, center) if inbound else (center, k)
                key = (first, second) if first < second else (second, first)
                quadratic[key] = quadratic.get(key, 0.0) + 0.24 * overlap

    return linear, quadratic, 1.25


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
