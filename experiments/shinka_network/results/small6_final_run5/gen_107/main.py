"""Surrogate QUBO terms for QOBLIB 08 network design. ShinkaEvolve edits the EVOLVE block."""

import time

# EVOLVE-BLOCK-START
import numpy as np


def surrogate_terms(demand):
    """Return direct costs, size-calibrated route terms, and degree penalty."""
    matrix = np.asarray(demand, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("demand must be a square matrix")

    n = matrix.shape[0]
    traffic = np.nan_to_num(
        matrix, copy=True, nan=0.0, posinf=0.0, neginf=0.0
    )
    np.maximum(traffic, 0.0, out=traffic)
    np.fill_diagonal(traffic, 0.0)

    scale = float(traffic.max()) if traffic.size else 0.0
    normalized = traffic / max(scale, 1e-15)
    roots = np.sqrt(normalized)

    linear = {
        (i, j): -float(0.75 * normalized[i, j] + 0.25 * roots[i, j])
        for i in range(n)
        for j in range(n)
        if i != j
    }
    quadratic = {}

    def add_pair(first, second, weight):
        """Accumulate a coefficient using a canonical link-pair key."""
        key = (first, second) if first < second else (second, first)
        quadratic[key] = quadratic.get(key, 0.0) + float(weight)

    if scale <= 0.0:
        penalty_weight = min(1.45, 1.30 + 0.008 * max(0, n - 10))
        return linear, quadratic, float(penalty_weight)

    outgoing = normalized.sum(axis=1)
    incoming = normalized.sum(axis=0)
    incident = np.maximum(outgoing, incoming)
    peak_incident = float(incident.max()) if incident.size else 0.0
    node_pressure = np.maximum(
        0.65, incident / max(peak_incident, 1e-15)
    )

    # The comparison below also uses normalized traffic. Keeping both
    # quantities in the same units makes the model demand-scale invariant.
    positive = normalized[normalized > 0.0]
    cutoff = float(np.quantile(positive, 0.75)) if positive.size else 0.0

    # A fixed number of candidate routes covers a decreasing fraction of
    # possible two-hop paths as n grows. Apply a modest correction, bounded
    # to avoid overwhelming direct rewards on the larger held-out sizes.
    route_size_factor = float(
        np.clip((max(n, 1) / 10.0) ** 0.40, 0.85, 1.40)
    )

    weak_ratio = 0.70
    boundary_ratio = 0.92
    leading_ratio = 0.78

    # Route-list width should mainly change coverage, not the total
    # attraction assigned to an OD pair. Use the two-route case as the
    # reference, including when a heavy OD retains four or five routes.
    width_exponent = 0.65

    for i in range(n):
        for k in range(n):
            if i == k or normalized[i, k] <= 0.0:
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
                edge_pressure = leg_a + leg_b
                balance = (
                    2.0 * np.sqrt(leg_a * leg_b)
                    / (edge_pressure + 1e-12)
                )
                pressure = float(node_pressure[j])
                affinity = (0.25 + edge_pressure) * (
                    0.70 + 0.30 * balance
                )
                rank = affinity / pressure ** 0.75
                discount = pressure * (1.0 + 0.5 * edge_pressure)
                candidates.append((float(rank), j, float(discount)))

            candidates.sort(reverse=True)
            if not candidates:
                continue

            base_keep = min(
                len(candidates),
                4 if normalized[i, k] >= cutoff else 2,
            )
            keep = base_keep
            best_rank = candidates[0][0]

            # Do not retain clearly unsupported trailing routes.
            while (
                keep > 1
                and candidates[keep - 1][0] < weak_ratio * best_rank
            ):
                keep -= 1

            # Resolve an ambiguous boundary by admitting at most one
            # additional route rather than broadening every OD uniformly.
            if keep == base_keep and keep < len(candidates):
                next_rank = candidates[keep][0]
                if (
                    next_rank >= boundary_ratio * candidates[keep - 1][0]
                    and next_rank >= leading_ratio * best_rank
                ):
                    keep += 1

            width_scale = min(1.0, (2.0 / keep) ** width_exponent)
            route_gain = route_size_factor * width_scale
            selected = set()

            for position, (_, j, discount) in enumerate(candidates[:keep]):
                selected.add(j)
                rank_factor = 1.0 + 0.08 * max(
                    1, base_keep - position
                )
                add_pair(
                    (i, j),
                    (j, k),
                    -0.46
                    * support
                    * rank_factor
                    * route_gain
                    / discount,
                )

            # Keep one distinct low-pressure alternative. It shares the
            # same OD-width correction as the affinity-selected routes.
            remaining = [
                candidate
                for candidate in candidates
                if candidate[1] not in selected
            ]
            if remaining:
                _, j, discount = min(
                    remaining,
                    key=lambda candidate: (candidate[2], candidate[1]),
                )
                add_pair(
                    (i, j),
                    (j, k),
                    -0.30 * support * route_gain / discount,
                )

    # Preserve the small correction for links represented unusually often
    # in the route catalog.
    interaction_count = {}
    for first, second in quadratic:
        interaction_count[first] = interaction_count.get(first, 0) + 1
        interaction_count[second] = interaction_count.get(second, 0) + 1

    if interaction_count:
        counts = np.asarray(list(interaction_count.values()), dtype=float)
        count_cutoff = float(np.quantile(counts, 0.75))
        count_range = float(counts.max()) - count_cutoff
        mean_direct = -float(sum(linear.values())) / max(1, n * (n - 1))

        if count_range > 0.0 and mean_direct > 0.0:
            for link, count in interaction_count.items():
                if count > count_cutoff:
                    excess = (count - count_cutoff) / count_range
                    linear[link] += 0.025 * mean_direct * float(excess)

    mean_direct = -float(sum(linear.values())) / max(1, n * (n - 1))
    depth_scale = max(
        0.75,
        float(np.log2(max(n, 2))) / float(np.log2(10.0)),
    )
    backtrack_weight = 0.65 * mean_direct * depth_scale

    # Reciprocal links consume scarce branches without expanding the
    # reachable neighborhood. Discount the penalty for strong mutual demand.
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

    # Two-node cut congestion signal. Remove endpoint-additive components
    # from the linear correction: they are constant on valid topologies.
    if backtrack_weight > 0.0 and n > 2:
        external = np.maximum(
            outgoing[:, None] + outgoing[None, :],
            incoming[:, None] + incoming[None, :],
        ) - normalized - normalized.T

        singleton_bound = 0.5 * np.maximum(
            incident[:, None], incident[None, :]
        )
        typical_share = (n - 2.0) / (n - 1.0)
        local_scale = np.clip(
            external
            / np.maximum(
                typical_share * (incident[:, None] + incident[None, :]),
                1e-15,
            ),
            0.75,
            1.25,
        )

        one_internal = np.maximum(
            0.0, external / 3.0 - singleton_bound
        )
        two_internal = np.maximum(
            0.0, external / 2.0 - singleton_bound
        )
        cut_linear = local_scale * one_internal
        cut_pair = local_scale * np.maximum(
            0.0, two_internal - 2.0 * one_internal
        )
        np.fill_diagonal(cut_linear, 0.0)
        np.fill_diagonal(cut_pair, 0.0)

        mean_cut = float(cut_pair.sum()) / max(1, n * (n - 1))
        if mean_cut > 1e-15:
            pair_count = n * (n - 1) // 2
            old_mean = sum(
                quadratic[((i, j), (j, i))]
                for i in range(n)
                for j in range(i + 1, n)
            ) / max(1, pair_count)

            cut_mix = 0.35
            cut_gain = cut_mix * old_mean / mean_cut

            row_sum = cut_linear.sum(axis=1)
            offset_total = float(row_sum.sum()) / (2.0 * (n - 1))
            offsets = (row_sum - offset_total) / (n - 2)

            for i in range(n):
                for j in range(i + 1, n):
                    key = ((i, j), (j, i))
                    quadratic[key] = float(
                        (1.0 - cut_mix) * quadratic[key]
                        + cut_gain * cut_pair[i, j]
                    )
                    correction = cut_gain * (
                        cut_linear[i, j] - offsets[i] - offsets[j]
                    )
                    linear[(i, j)] += float(correction)
                    linear[(j, i)] += float(correction)

    # Preserve sparse demand-weighted branch-diversity terms.
    diversity_scale = 0.45 * mean_direct * depth_scale
    pair_budget = max(1, n // 2)

    if diversity_scale > 0.0:
        for reverse in (False, True):
            profiles = normalized.T if reverse else normalized

            for v in range(n):
                weights = (
                    normalized[:, v] if reverse else normalized[v, :]
                )
                total = float(weights.sum())
                if total <= 0.0:
                    continue
                weights = weights / total

                neighbors = [a for a in range(n) if a != v]
                best_coverage = max(
                    (
                        float(np.dot(profiles[a], weights))
                        for a in neighbors
                    ),
                    default=0.0,
                )
                if best_coverage <= 1e-15:
                    continue

                overlaps = []
                for pos, a in enumerate(neighbors):
                    for b in neighbors[pos + 1:]:
                        overlap = float(
                            np.dot(
                                np.minimum(profiles[a], profiles[b]),
                                weights,
                            )
                        ) / best_coverage
                        if overlap > 0.35:
                            overlaps.append((overlap, a, b))

                overlaps.sort(reverse=True)
                for overlap, a, b in overlaps[:pair_budget]:
                    first = (a, v) if reverse else (v, a)
                    second = (b, v) if reverse else (v, b)
                    add_pair(
                        first, second, diversity_scale * overlap
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
