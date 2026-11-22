"""Surrogate QUBO terms for QOBLIB 08 network design. ShinkaEvolve edits the EVOLVE block."""

import time

# EVOLVE-BLOCK-START
import numpy as np


def surrogate_terms(demand):
    """Build sparse direct, route-exposure, and cut-congestion costs."""
    traffic = np.asarray(demand, dtype=float)
    n = int(traffic.shape[0])

    traffic = np.maximum(
        np.nan_to_num(traffic, nan=0.0, posinf=0.0, neginf=0.0), 0.0
    ).copy()
    np.fill_diagonal(traffic, 0.0)

    off_diagonal = ~np.eye(n, dtype=bool)
    penalty_weight = float(min(1.50, 1.27 + 0.010 * max(0, n - 10)))

    scale = float(traffic.max()) if traffic.size else 0.0
    if scale <= 0.0:
        linear = {
            (i, j): 0.0
            for i in range(n)
            for j in range(n)
            if i != j
        }
        return linear, {}, penalty_weight

    z = traffic / scale
    root = np.sqrt(z)

    # Direct links retain broad support for moderate demands.
    direct_score = 0.68 * z + 0.32 * root
    costs = -direct_score.copy()
    mean_direct = float(direct_score[off_diagonal].mean())

    outgoing = z.sum(axis=1)
    incoming = z.sum(axis=0)
    incident = np.maximum(outgoing, incoming)
    incident_scale = max(float(incident.max()), 1e-12)
    base_pressure = np.maximum(0.60, incident / incident_scale)

    # Candidate two-hop paths, indexed as (source, transit, destination).
    edge_affinity = z[:, :, None] + z[None, :, :]
    affinity = (
        0.22
        + edge_affinity
        + 0.12 * np.sqrt(z[:, :, None] * z[None, :, :])
    )
    valid_paths = (
        off_diagonal[:, :, None]
        & off_diagonal[None, :, :]
        & off_diagonal[:, None, :]
    )

    # Estimate transit load with damped, soft route assignments. The direct
    # option prevents this estimate from forcing every demand onto a path.
    log_affinity = np.log(affinity)
    log_edge_discount = np.log1p(0.18 * edge_affinity)
    direct_utility = np.log(0.30 + 1.15 * z) / 0.42
    transit = np.zeros(n, dtype=float)

    for iteration in range(3):
        pressure = np.maximum(
            1.0,
            0.72 * base_pressure / 0.60
            + 0.28 * (1.0 + transit / (incident + 1.0)),
        )
        utilities = (
            log_affinity
            - (0.72 + 0.08 * iteration) * np.log(pressure)[None, :, None]
            - log_edge_discount
        ) / 0.42
        utilities = np.where(valid_paths, utilities, -np.inf)

        maximum = np.maximum(utilities.max(axis=1), direct_utility)
        route_mass = np.exp(
            np.clip(utilities - maximum[:, None, :], -30.0, 0.0)
        )
        route_mass = np.where(valid_paths, route_mass, 0.0)
        direct_mass = np.exp(
            np.clip(direct_utility - maximum, -30.0, 0.0)
        )
        denominator = route_mass.sum(axis=1) + direct_mass
        probabilities = route_mass / np.maximum(denominator[:, None, :], 1e-15)
        updated = (probabilities * z[:, None, :]).sum(axis=(0, 2))
        transit = 0.45 * transit + 0.55 * updated

    transit_reference = max(1.0, float(incident.mean()))
    route_pressure = 1.0 + transit / transit_reference
    node_pressure = np.maximum(
        0.60,
        0.62 * incident / incident_scale
        + 0.38 * route_pressure / max(1.0, float(route_pressure.max())),
    )

    ranks = (
        affinity
        / node_pressure[None, :, None] ** 0.82
        / (1.0 + 0.10 * edge_affinity)
    )
    ranks = np.where(valid_paths, ranks, -np.inf)
    ordering = np.argsort(-ranks, axis=1, kind="stable")

    positive = z[z > 0.0]
    heavy_cutoff = float(np.quantile(positive, 0.72)) if positive.size else 0.0
    demand_support = 0.66 * z + 0.34 * root

    # Keep route ranking and selection fixed, but taper the aggregate route
    # reward more strongly at larger sizes where route alternatives multiply.
    if n >= 15:
        route_size_factor = float(np.clip((12.0 / n) ** 0.30, 0.82, 0.93))
    else:
        route_size_factor = float(np.clip((12.0 / max(n, 1)) ** 0.15, 0.88, 1.0))
    route_strength = 0.39 * route_size_factor
    pressure_discount = node_pressure ** 0.55

    quadratic = {}
    route_frequency = np.zeros((n, n), dtype=float)
    route_support = np.zeros((n, n), dtype=float)

    for i in range(n):
        for k in range(n):
            if i == k or z[i, k] <= 0.0:
                continue

            candidates = ordering[i, :, k]
            base_keep = min(n - 2, 3 if z[i, k] >= heavy_cutoff else 2)
            keep = base_keep

            # Preserve one extra route only when the selection boundary is
            # ambiguous, keeping the coupling graph compact.
            if keep < n - 2:
                boundary = float(ranks[i, int(candidates[keep - 1]), k])
                following = float(ranks[i, int(candidates[keep]), k])
                relative_gap = (boundary - following) / max(boundary, 1e-12)
                if relative_gap < 0.12:
                    keep += 1

            support = float(demand_support[i, k])
            for position in range(keep):
                j = int(candidates[position])
                rank_boost = (
                    1.0 + 0.055 * (base_keep - position - 1)
                    if position < base_keep else 0.72
                )
                magnitude = float(
                    route_strength
                    * support
                    * rank_boost
                    / pressure_discount[j]
                    / (1.0 + 0.08 * edge_affinity[i, j, k])
                )

                first, second = (i, j), (j, k)
                key = (first, second) if first < second else (second, first)
                quadratic[key] = -magnitude

                route_frequency[i, j] += 1.0
                route_frequency[j, k] += 1.0
                route_support[i, j] += magnitude
                route_support[j, k] += magnitude

    # Approximate two-node cut congestion. The reciprocal term captures the
    # extra loss from retaining both directed links within a pair.
    internal = z + z.T
    external_cut = np.maximum(
        0.0,
        np.maximum(
            outgoing[:, None] + outgoing[None, :],
            incoming[:, None] + incoming[None, :],
        ) - internal,
    )

    total_incident = outgoing + incoming
    vulnerability = 0.5 * (
        np.maximum(total_incident[:, None] - internal, 0.0)
        / np.maximum(total_incident[:, None], 1e-12)
        + np.maximum(total_incident[None, :] - internal, 0.0)
        / np.maximum(total_incident[None, :], 1e-12)
    )
    cut_factor = 0.80 + 0.40 * np.clip(vulnerability, 0.0, 1.0)
    factor_mean = max(float(cut_factor[off_diagonal].mean()), 1e-12)
    cut_coefficient = (
        (0.80 / 144.0)
        * (external_cut / incident_scale) ** 2
        * cut_factor
        / factor_mean
    )
    np.fill_diagonal(cut_coefficient, 0.0)
    cut_linear = 7.0 * cut_coefficient

    # Remove node potentials and a common edge offset: they are constant
    # across valid degree-two topologies.
    node_offset = cut_linear.sum(axis=1) / max(1, n - 1)
    edge_offset = float(cut_linear[off_diagonal].mean())
    costs += (
        cut_linear
        - node_offset[:, None]
        - node_offset[None, :]
        + edge_offset
    )

    size_scale = max(0.80, float(np.log2(max(n, 2)) / np.log2(12.0)))
    reciprocal_scale = 0.43 * mean_direct * size_scale

    for i in range(n):
        for j in range(i + 1, n):
            mutual = min(float(z[i, j]), float(z[j, i]))
            quadratic[((i, j), (j, i))] = float(
                reciprocal_scale * (1.0 - 0.62 * mutual)
                + 13.0 * cut_coefficient[i, j]
            )

    # Bounded upper-tail exposure correction. Frequency discourages repeatedly
    # reused transit links; weighted support distinguishes paths carrying
    # stronger route-reward evidence.
    for exposure, fraction in (
        (route_frequency, 0.75),
        (route_support, 0.25),
    ):
        active = exposure[off_diagonal & (exposure > 0.0)]
        if not active.size:
            continue

        threshold = float(np.quantile(active, 0.80))
        maximum = float(active.max())
        if maximum > threshold:
            excess = np.clip(
                (exposure - threshold) / (maximum - threshold),
                0.0,
                1.0,
            )
            costs += 0.12 * mean_direct * fraction * excess

    linear = {
        (i, j): float(costs[i, j])
        for i in range(n)
        for j in range(n)
        if i != j
    }

    # Degree penalties are supplied by the evaluator.
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
