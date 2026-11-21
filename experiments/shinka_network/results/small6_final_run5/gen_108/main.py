"""Surrogate QUBO terms for QOBLIB 08 network design. ShinkaEvolve edits the EVOLVE block."""

import time

# EVOLVE-BLOCK-START
import numpy as np


def surrogate_terms(demand):
    """Return direct costs, adaptive path couplings, and degree penalty."""
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
        (i, j): -float(
            0.75 * normalized[i, j] + 0.25 * roots[i, j]
        )
        for i in range(n)
        for j in range(n)
        if i != j
    }

    row_demand = traffic.sum(axis=1)
    column_demand = traffic.sum(axis=0)
    incident = np.maximum(column_demand, row_demand)
    max_incident = float(np.max(incident)) if incident.size else 0.0
    node_pressure = np.maximum(
        0.65, incident / max(max_incident, 1e-15)
    )

    positive = traffic[traffic > 0.0]
    cutoff = float(np.quantile(positive, 0.75)) if positive.size else 0.0
    quadratic = {}

    # Selection parameters:
    # - discard trailing routes below 70% of the leading affinity;
    # - add one route only if it is within 8% of the boundary score
    #   and retains at least 78% of the leading score;
    # - keep reward growth sublinear when adding alternatives.
    weak_ratio = 0.70
    boundary_ratio = 0.92
    leading_ratio = 0.78
    reward_exponent = 0.35

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
                leg_a = float(normalized[i, j])
                leg_b = float(normalized[j, k])
                share_a = min(
                    1.0,
                    n * float(traffic[i, j]) / max(float(row_demand[i]), 1e-12),
                )
                share_b = min(
                    1.0,
                    n * float(traffic[j, k]) / max(float(column_demand[k]), 1e-12),
                )
                leg_a = 0.75 * leg_a + 0.25 * share_a
                leg_b = 0.75 * leg_b + 0.25 * share_b
                edge_pressure = leg_a + leg_b
                balance = (
                    2.0 * leg_a * leg_b
                    / (edge_pressure + 1e-12)
                )
                pressure = float(node_pressure[j])
                affinity = (0.25 + edge_pressure) * (0.70 + 0.30 * balance)
                rank = affinity / (pressure ** 0.75)
                discount = pressure * (1.0 + 0.5 * edge_pressure)
                candidates.append((float(rank), j, float(discount)))

            candidates.sort(reverse=True)
            if not candidates:
                continue

            base_keep = min(
                len(candidates),
                4 if traffic[i, k] >= cutoff else 2,
            )
            keep = base_keep
            best_rank = candidates[0][0]

            # Clear score gaps indicate that broadening route coverage
            # would mostly reward unlikely combinations of links.
            while (
                keep > 1
                and candidates[keep - 1][0] < weak_ratio * best_rank
            ):
                keep -= 1

            # Broaden only an unresolved boundary, not an already-pruned
            # candidate list. The extra route is strictly capped at one.
            if keep == base_keep and keep < len(candidates):
                next_rank = candidates[keep][0]
                if (
                    next_rank >= boundary_ratio * candidates[keep - 1][0]
                    and next_rank >= leading_ratio * best_rank
                ):
                    keep += 1

            # Adding a route should improve coverage, not simply make an
            # OD pair much more influential. Do not amplify pruned lists.
            reward_scale = min(
                1.0, (float(base_keep) / keep) ** reward_exponent
            )
            selected = set()

            for position, (_, j, discount) in enumerate(candidates[:keep]):
                selected.add(j)
                # Preserve existing rewards for the original ranks.
                rank_factor = 1.0 + 0.08 * max(
                    1, base_keep - position
                )
                weight = (
                    -0.46
                    * support
                    * rank_factor
                    * reward_scale
                    / discount
                )
                first, second = (i, j), (j, k)
                key = (
                    (first, second)
                    if first < second
                    else (second, first)
                )
                quadratic[key] = float(weight)

            # Keep a distinct low-pressure fallback. It serves a different
            # purpose from the affinity-boundary ambiguity candidate.
            remaining = [
                candidate
                for candidate in candidates
                if candidate[1] not in selected
            ]
            if remaining:
                _, j, discount = min(
                    remaining, key=lambda candidate: (
                        candidate[2], candidate[1]
                    )
                )
                first, second = (i, j), (j, k)
                key = (
                    (first, second)
                    if first < second
                    else (second, first)
                )
                quadratic[key] = float(-0.30 * support / discount)

    # Slightly discourage links that participate in unusually many distinct
    # route interactions, limiting the risk of concentrating transit load.
    interaction_count = {}
    for first, second in quadratic:
        interaction_count[first] = interaction_count.get(first, 0) + 1
        interaction_count[second] = interaction_count.get(second, 0) + 1

    if interaction_count:
        counts = np.asarray(list(interaction_count.values()), dtype=float)
        cutoff_count = float(np.quantile(counts, 0.75))
        max_count = float(counts.max())
        mean_direct = -float(sum(linear.values())) / max(1, n * (n - 1))
        count_range = max_count - cutoff_count
        if count_range > 0.0 and mean_direct > 0.0:
            for link, count in interaction_count.items():
                if count > cutoff_count:
                    excess = (count - cutoff_count) / count_range
                    linear[link] += 0.025 * mean_direct * float(excess)

    # Preserve the successful sparse structural penalty.
    mean_direct = -float(sum(linear.values())) / max(1, n * (n - 1))
    depth_scale = max(
        0.75, float(np.log2(max(n, 2))) / float(np.log2(10.0))
    )
    backtrack_weight = 0.65 * mean_direct * depth_scale

    if backtrack_weight > 0.0:
        for i in range(n):
            for j in range(i + 1, n):
                mutual = min(
                    float(normalized[i, j]),
                    float(normalized[j, i]),
                )
                quadratic[((i, j), (j, i))] = float(
                    backtrack_weight * (1.0 - 0.45 * mutual)
                )

    # Two useful neighbors need not provide two distinct useful routes.
    # Penalize the strongest demand-weighted coverage overlaps, rather
    # than rewarding every supported two-hop path independently.
    diversity_scale = 0.45 * mean_direct * depth_scale
    pair_budget = max(1, n // 2)

    if diversity_scale > 0.0:
        for reverse in (False, True):
            # Outgoing: neighbor a can forward traffic to destination k.
            # Incoming: predecessor a can collect traffic from source k.
            profiles = normalized.T if reverse else normalized
            for v in range(n):
                weights = normalized[:, v] if reverse else normalized[v, :]
                total = float(weights.sum())
                if total <= 0.0:
                    continue
                weights = weights / total
                coverage = profiles @ weights
                neighbors = [a for a in range(n) if a != v]
                best_coverage = max(float(coverage[a]) for a in neighbors)
                if best_coverage <= 1e-15:
                    continue

                overlaps = []
                for position, a in enumerate(neighbors):
                    for b in neighbors[position + 1:]:
                        overlap = float(
                            np.dot(np.minimum(profiles[a], profiles[b]), weights)
                        ) / best_coverage
                        if overlap > 0.35:
                            overlaps.append((overlap, a, b))

                overlaps.sort(reverse=True)
                for overlap, a, b in overlaps[:pair_budget]:
                    first = (a, v) if reverse else (v, a)
                    second = (b, v) if reverse else (v, b)
                    key = (
                        (first, second)
                        if first < second
                        else (second, first)
                    )
                    quadratic[key] = quadratic.get(key, 0.0) + float(
                        diversity_scale * overlap
                    )

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
