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

    # Reciprocal links can be useful for mutual traffic, but otherwise consume
    # scarce degree and routing capacity. Keep this structural penalty modest.
    mean_direct = (
        float(sum(-value for value in linear.values())) / max(1, n * (n - 1))
    )
    size_scale = max(0.80, float(np.log2(max(n, 2))) / float(np.log2(12.0)))
    reciprocal_scale = 0.43 * mean_direct * size_scale
    for i in range(n):
        for j in range(i + 1, n):
            mutual = min(float(z[i, j]), float(z[j, i]))
            quadratic[((i, j), (j, i))] = float(
                reciprocal_scale * (1.0 - 0.62 * mutual)
            )

    # Approximate congestion across each two-node cut. The cut has
    # 4 - x_ij - x_ji boundary links in each direction on a valid topology;
    # the quadratic expansion of its inverse-square load has coefficients
    # 7/144 for each arc and 13/144 for the reciprocal product.
    row_sum = traffic.sum(axis=1)
    column_sum = traffic.sum(axis=0)
    total_incident = row_sum + column_sum
    cut_linear = np.zeros((n, n), dtype=float)
    cut_strength = 0.80 / 144.0
    cut_data = []
    vulnerability_factors = []

    for i in range(n):
        for j in range(i + 1, n):
            internal = float(traffic[i, j] + traffic[j, i])
            external_cut = max(
                0.0,
                max(
                    float(row_sum[i] + row_sum[j]),
                    float(column_sum[i] + column_sum[j]),
                ) - internal,
            )
            squared_load = (external_cut / incident_scale) ** 2

            # Evaluate each endpoint's external exposure by direction:
            # outgoing and incoming traffic consume distinct capacities.
            out_i = max(0.0, float(row_sum[i]) - float(traffic[i, j]))
            in_i = max(0.0, float(column_sum[i]) - float(traffic[j, i]))
            out_j = max(0.0, float(row_sum[j]) - float(traffic[j, i]))
            in_j = max(0.0, float(column_sum[j]) - float(traffic[i, j]))
            vulnerability_i = max(
                out_i / max(float(row_sum[i]), 1e-12),
                in_i / max(float(column_sum[i]), 1e-12),
            )
            vulnerability_j = max(
                out_j / max(float(row_sum[j]), 1e-12),
                in_j / max(float(column_sum[j]), 1e-12),
            )
            vulnerability = max(vulnerability_i, vulnerability_j)
            factor = 0.80 + 0.40 * min(1.0, max(0.0, vulnerability))
            cut_data.append((i, j, squared_load, factor))
            vulnerability_factors.append(factor)

    factor_mean = float(np.mean(vulnerability_factors)) if vulnerability_factors else 1.0
    for i, j, squared_load, factor in cut_data:
        if squared_load <= 0.0:
            continue
        coefficient = cut_strength * squared_load * factor / max(factor_mean, 1e-12)
        cut_linear[i, j] = cut_linear[j, i] = 7.0 * coefficient
        reciprocal_key = ((i, j), (j, i))
        quadratic[reciprocal_key] = quadratic.get(reciprocal_key, 0.0) + 13.0 * coefficient

    # Remove node potentials and a common edge offset: these are constant
    # over valid degree-two topologies, so the cut correction guides rankings
    # without introducing a spurious degree preference.
    node_offset = cut_linear.sum(axis=1) / max(1, n - 1)
    edge_offset = float(cut_linear.sum()) / max(1, n * (n - 1))
    for i, j in linear:
        linear[(i, j)] += float(
            cut_linear[i, j] - node_offset[i] - node_offset[j] + edge_offset
        )

    # Debias only links that occur unusually often in route-reward couplings.
    # This is a demand-based link cost, not an additional degree penalty.
    route_frequency = np.zeros((n, n), dtype=float)
    for (first, second), weight in quadratic.items():
        if weight < 0.0:
            route_frequency[first[0], first[1]] += 1.0
            route_frequency[second[0], second[1]] += 1.0

    off_diagonal = ~np.eye(n, dtype=bool)
    active_frequency = route_frequency[off_diagonal]
    active_frequency = active_frequency[active_frequency > 0.0]
    if active_frequency.size:
        frequency_cutoff = float(np.quantile(active_frequency, 0.80))
        frequency_max = float(np.max(active_frequency))
        if frequency_max > frequency_cutoff:
            for i in range(n):
                for j in range(n):
                    count = route_frequency[i, j]
                    if i != j and count > frequency_cutoff:
                        excess = (count - frequency_cutoff) / (
                            frequency_max - frequency_cutoff
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
