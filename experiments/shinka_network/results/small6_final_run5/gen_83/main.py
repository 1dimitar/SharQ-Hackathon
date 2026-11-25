"""Surrogate QUBO terms for QOBLIB 08 network design. ShinkaEvolve edits the EVOLVE block."""

import time

# EVOLVE-BLOCK-START
import numpy as np


def surrogate_terms(demand):
    """Build direct rewards and sparse, transit-pressure-aware path rewards."""
    traffic = np.maximum(np.asarray(demand, dtype=float), 0.0).copy()
    n = len(traffic)
    np.fill_diagonal(traffic, 0.0)

    scale = float(np.max(traffic)) if traffic.size else 0.0
    if scale <= 0.0:
        scale = 1.0

    z = traffic / scale
    root = np.sqrt(z)

    # Preserve broad support for moderate demands while emphasizing large ones.
    direct_score = 0.68 * z + 0.32 * root
    linear = {
        (i, j): -float(direct_score[i, j])
        for i in range(n)
        for j in range(n)
        if i != j
    }

    incident = np.maximum(traffic.sum(axis=0), traffic.sum(axis=1))
    incident_scale = float(np.max(incident)) if incident.size else 0.0
    if incident_scale <= 0.0:
        incident_scale = 1.0
    base_pressure = np.maximum(0.60, incident / incident_scale)

    # Candidate paths are kept for the pressure iterations and final sparse QUBO.
    # Their initial affinity rewards paths whose constituent links are plausible.
    paths = {}
    for i in range(n):
        for k in range(n):
            if i == k or traffic[i, k] <= 0.0:
                continue
            candidates = []
            for j in range(n):
                if j == i or j == k:
                    continue
                edge_affinity = z[i, j] + z[j, k]
                affinity = 0.22 + edge_affinity + 0.12 * np.sqrt(
                    max(0.0, z[i, j] * z[j, k])
                )
                candidates.append((j, float(affinity), float(edge_affinity)))
            candidates.sort(key=lambda item: (-item[1], item[0]))
            paths[(i, k)] = candidates

    # Softly estimate which nodes would become transit points. Three damped
    # updates prevent one early route choice from dominating the estimate.
    transit = np.zeros(n, dtype=float)
    temperature = 0.42
    for iteration in range(3):
        updated = np.zeros(n, dtype=float)
        for (i, k), candidates in paths.items():
            if not candidates:
                continue

            pressure = np.maximum(
                1.0,
                0.72 * (base_pressure / 0.60)
                + 0.28 * (1.0 + transit / (incident + scale)),
            )
            utilities = []
            for j, affinity, edge_affinity in candidates:
                # Route-specific transit pressure; modest edge-pressure
                # discount avoids repeatedly selecting already attractive links.
                congestion = pressure[j] ** (0.72 + 0.08 * iteration)
                edge_factor = 1.0 + 0.18 * edge_affinity
                utilities.append(np.log(max(1e-12, affinity / (congestion * edge_factor))))

            # Include a direct option so the estimate does not assume every
            # OD pair must use a transit node.
            direct_utility = np.log(
                max(1e-12, 0.30 + 1.15 * z[i, k])
            )
            all_utilities = np.asarray([direct_utility] + utilities, dtype=float)
            all_utilities = (all_utilities - np.max(all_utilities)) / temperature
            probabilities = np.exp(np.clip(all_utilities, -30.0, 0.0))
            probabilities /= max(1e-12, float(probabilities.sum()))

            amount = float(traffic[i, k])
            for idx, (j, _, _) in enumerate(candidates, start=1):
                updated[j] += amount * probabilities[idx]

        transit = 0.45 * transit + 0.55 * updated

    # Final node pressure combines observed incident demand and route-estimated
    # transit load. The floor prevents low-demand nodes from being overfavoured.
    transit_reference = max(
        scale,
        float(np.mean(incident)) if incident.size else scale,
        1e-12,
    )
    route_pressure = 1.0 + transit / transit_reference
    node_pressure = np.maximum(
        0.60,
        0.62 * (incident / incident_scale) + 0.38 * route_pressure / max(1.0, float(np.max(route_pressure))),
    )

    positive = traffic[traffic > 0.0]
    cutoff = float(np.quantile(positive, 0.72)) if positive.size else 0.0
    quadratic = {}

    # Sparse two-hop rewards: typically two intermediates per OD, three for
    # high-demand pairs. Ranking incorporates both estimated transit pressure
    # and route-specific edge affinity.
    for (i, k), candidates in paths.items():
        demand_support = 0.66 * z[i, k] + 0.34 * np.sqrt(z[i, k])
        ranked = []
        for j, affinity, edge_affinity in candidates:
            pressure = float(node_pressure[j])
            rank = affinity / (pressure ** 0.82) / (1.0 + 0.10 * edge_affinity)
            ranked.append((rank, j, pressure, edge_affinity))
        ranked.sort(key=lambda item: (-item[0], item[1]))

        base_keep = min(len(ranked), 3 if traffic[i, k] >= cutoff else 2)
        keep = base_keep
        if keep < len(ranked):
            boundary_rank = ranked[keep - 1][0]
            next_rank = ranked[keep][0]
            relative_gap = (boundary_rank - next_rank) / max(boundary_rank, 1e-12)
            if relative_gap < 0.12:
                keep += 1

        for position, (_, j, pressure, edge_affinity) in enumerate(ranked[:keep]):
            if position < base_keep:
                rank_boost = 1.0 + 0.055 * (base_keep - position - 1)
            else:
                rank_boost = 0.72
            weight = (
                -0.39
                * float(demand_support)
                * rank_boost
                / (pressure ** 0.55)
                / (1.0 + 0.08 * edge_affinity)
            )
            first, second = (i, j), (j, k)
            key = (first, second) if first < second else (second, first)
            quadratic[key] = float(weight)

    # Reciprocal links keep both arcs inside a two-node cut. Increase their
    # penalty when that pair has substantial external traffic relative to its
    # own incident load; normalize the local factors to preserve the global
    # reciprocal-penalty scale.
    mean_direct = (
        float(sum(-value for value in linear.values())) / max(1, n * (n - 1))
    )
    size_scale = max(0.80, float(np.log2(max(n, 2))) / float(np.log2(12.0)))
    reciprocal_scale = 0.43 * mean_direct * size_scale

    pair_data = []
    local_factors = []
    for i in range(n):
        for j in range(i + 1, n):
            external = (
                float(traffic[i, :].sum() + traffic[:, i].sum()
                      + traffic[j, :].sum() + traffic[:, j].sum())
                - 2.0 * float(traffic[i, j] + traffic[j, i])
            )
            local_load = float(incident[i] + incident[j])
            external_share = min(1.0, max(0.0, external / max(local_load, 1e-12)))
            factor = 0.75 + 0.50 * external_share
            pair_data.append((i, j, factor))
            local_factors.append(factor)

    factor_mean = float(np.mean(local_factors)) if local_factors else 1.0
    for i, j, factor in pair_data:
        mutual = min(float(z[i, j]), float(z[j, i]))
        localized_scale = reciprocal_scale * factor / max(factor_mean, 1e-12)
        quadratic[((i, j), (j, i))] = float(
            localized_scale * (1.0 - 0.62 * mutual)
        )

    # Debias links according to the strength, not merely the number, of
    # two-hop route rewards in which they participate. This approximates
    # accumulated route load while avoiding extra couplings.
    route_support = np.zeros((n, n), dtype=float)
    for (first, second), weight in quadratic.items():
        if weight < 0.0:
            contribution = -float(weight)
            route_support[first[0], first[1]] += contribution
            route_support[second[0], second[1]] += contribution

    off_diagonal = ~np.eye(n, dtype=bool)
    active_support = route_support[off_diagonal]
    active_support = active_support[active_support > 0.0]
    if active_support.size:
        support_cutoff = float(np.quantile(active_support, 0.80))
        support_max = float(np.max(active_support))
        if support_max > support_cutoff:
            for i in range(n):
                for j in range(n):
                    support = route_support[i, j]
                    if i != j and support > support_cutoff:
                        excess = (support - support_cutoff) / (
                            support_max - support_cutoff
                        )
                        linear[(i, j)] += float(0.12 * mean_direct * excess)

    # The evaluator adds the degree penalty separately.
    penalty_weight = min(1.50, 1.27 + 0.010 * max(0, n - 10))
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
