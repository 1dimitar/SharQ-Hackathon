"""Surrogate QUBO terms for QOBLIB 08 network design. ShinkaEvolve edits the EVOLVE block."""

import time

# EVOLVE-BLOCK-START
import numpy as np


def surrogate_terms(demand):
    """Build sparse route rewards and demand-aware expansion corrections."""
    matrix = np.asarray(demand, dtype=float)
    n = len(matrix)

    traffic = np.maximum(matrix, 0.0).copy()
    np.fill_diagonal(traffic, 0.0)
    scale = float(np.max(traffic)) if traffic.size else 0.0
    if scale <= 0.0:
        scale = 1.0

    normalized = traffic / scale
    roots = np.sqrt(normalized)
    direct = 0.75 * normalized + 0.25 * roots
    linear = {
        (i, j): -float(direct[i, j])
        for i in range(n)
        for j in range(n)
        if i != j
    }
    quadratic = {}

    if not np.any(traffic):
        return linear, quadratic, 1.2

    incident = np.maximum(traffic.sum(axis=0), traffic.sum(axis=1))
    max_incident = max(float(np.max(incident)), 1e-12)
    initial_pressure = np.maximum(0.65, incident / max_incident)

    # Infer likely transit pressure from a small set of plausible routes.
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
                rank = (
                    (0.25 + evidence)
                    / float(initial_pressure[j]) ** 0.75
                )
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
            routed_fraction = 1.0 - 0.65 * float(normalized[i, k])

            for (_, j, _), strength in zip(selected, strengths):
                transit_load[j] += (
                    float(traffic[i, k])
                    * routed_fraction
                    * strength
                    / total_strength
                )

    projected_pressure = (
        incident + 0.35 * transit_load
    ) / max_incident
    node_pressure = np.clip(projected_pressure, 0.65, 1.8)

    positive = traffic[traffic > 0.0]
    cutoff = float(np.quantile(positive, 0.75))

    # Weak scaling compensates for decreasing route-completion probability
    # without allowing route fragments to dominate direct-link preferences.
    route_scale = float(np.clip((n / 12.0) ** 0.12, 0.95, 1.09))

    for i in range(n):
        for k in range(n):
            if i == k or traffic[i, k] <= 0.0:
                continue

            support = float(
                0.70 * normalized[i, k] + 0.30 * roots[i, k]
            )
            candidates = []
            for j in range(n):
                if j == i or j == k:
                    continue
                edge_pressure = float(
                    normalized[i, j] + normalized[j, k]
                )
                pressure = float(node_pressure[j])
                rank = (0.25 + edge_pressure) / pressure ** 0.75
                discount = pressure * (1.0 + 0.5 * edge_pressure)
                candidates.append((rank, j, discount))

            candidates.sort(reverse=True)
            keep = min(
                len(candidates),
                4 if traffic[i, k] >= cutoff else 2,
            )

            if keep < len(candidates):
                frontier = candidates[keep - 1][0]
                next_rank, next_j, _ = candidates[keep]
                next_evidence = float(
                    normalized[i, next_j] + normalized[next_j, k]
                )
                best_rank = max(candidates[0][0], 1e-12)
                if (
                    next_rank / max(frontier, 1e-12) >= 0.92
                    and next_rank / best_rank >= 0.65
                    and next_evidence >= 0.05
                ):
                    keep += 1

            selected = set()
            for position, (_, j, discount) in enumerate(candidates[:keep]):
                selected.add(j)
                rank_factor = 1.0 + 0.08 * (keep - position)
                weight = (
                    -0.46 * route_scale * support * rank_factor / discount
                )
                first, second = (i, j), (j, k)
                key = (
                    (first, second)
                    if first < second
                    else (second, first)
                )
                quadratic[key] = float(weight)

            # Keep one pressure-favoured alternative to the affinity choices.
            remaining = [
                candidate
                for candidate in candidates
                if candidate[1] not in selected
            ]
            if remaining:
                _, j, discount = min(
                    remaining, key=lambda candidate: (candidate[2], candidate[1])
                )
                first, second = (i, j), (j, k)
                key = (
                    (first, second)
                    if first < second
                    else (second, first)
                )
                quadratic[key] = float(
                    -0.30 * route_scale * support / discount
                )

    # Estimate background attraction under degree-balanced soft marginals.
    if quadratic:
        propensity = np.exp(2.0 * direct)
        np.fill_diagonal(propensity, 0.0)
        for _ in range(20):
            propensity *= (
                2.0 / np.maximum(propensity.sum(axis=1), 1e-12)
            )[:, None]
            propensity *= (
                2.0 / np.maximum(propensity.sum(axis=0), 1e-12)
            )[None, :]
        np.minimum(propensity, 1.0, out=propensity)

        exposure = np.zeros((n, n), dtype=float)
        for (first, second), weight in quadratic.items():
            exposure[first] -= weight * float(propensity[second])
            exposure[second] -= weight * float(propensity[first])

        active_exposure = exposure[exposure > 0.0]
        if active_exposure.size:
            threshold = float(np.quantile(active_exposure, 0.80))
            for link in linear:
                excess = max(0.0, float(exposure[link]) - threshold)
                linear[link] += min(0.18, 0.12 * excess)

    # Retain a compact correction for branches serving similar OD profiles.
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
                key = (
                    (first, second)
                    if first < second
                    else (second, first)
                )
                quadratic[key] = (
                    quadratic.get(key, 0.0) + 0.24 * overlap
                )

    # A reciprocal pair consumes two links but contributes no new
    # non-backtracking two-hop reach. Charge a modest structural cost,
    # discounted when both directions have substantial direct demand.
    # This adds only n*(n-1)/2 couplings, with no three-hop fragments.
    expansion_scale = float(np.clip((n / 12.0) ** 0.15, 0.90, 1.11))
    for i in range(n):
        for j in range(i + 1, n):
            mutual_support = float(
                np.sqrt(normalized[i, j] * normalized[j, i])
            )
            weight = (
                0.16 * expansion_scale * (1.0 - 0.75 * mutual_support)
            )
            key = ((i, j), (j, i))
            quadratic[key] = quadratic.get(key, 0.0) + float(weight)

    pressure_cv = float(
        np.std(incident) / max(float(np.mean(incident)), 1e-12)
    )
    penalty_weight = 1.15 + 0.20 * pressure_cv / (1.0 + pressure_cv)
    return linear, quadratic, float(
        min(1.35, max(1.15, penalty_weight))
    )


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
