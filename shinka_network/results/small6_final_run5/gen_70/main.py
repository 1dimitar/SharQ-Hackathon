"""Surrogate QUBO terms for QOBLIB 08 network design. ShinkaEvolve edits the EVOLVE block."""

import time

# EVOLVE-BLOCK-START
import numpy as np


def surrogate_terms(demand):
    """Return linear costs, sparse quadratic costs, and degree-penalty weight."""
    traffic = np.maximum(np.asarray(demand, dtype=float), 0.0).copy()
    n = len(traffic)
    np.fill_diagonal(traffic, 0.0)

    scale = float(traffic.max()) if traffic.size else 0.0
    z = traffic / (scale if scale > 0.0 else 1.0)
    root = np.sqrt(z)
    mask = ~np.eye(n, dtype=bool)

    # Direct-link support retains substantial weight on moderate demands.
    direct = 0.70 * z + 0.30 * root
    linear = {
        (i, j): -float(direct[i, j])
        for i in range(n)
        for j in range(n)
        if i != j
    }
    quadratic = {}

    if scale <= 0.0:
        return linear, quadratic, 1.25

    outgoing = z.sum(axis=1)
    incoming = z.sum(axis=0)
    incident = np.maximum(outgoing, incoming)
    peak_incident = max(float(incident.max()), 1e-12)
    base_pressure = np.maximum(0.60, incident / peak_incident)

    # Arrays have axes (source, intermediate, destination).
    edge_affinity = z[:, :, None] + z[None, :, :]
    affinity = (
        0.22
        + edge_affinity
        + 0.12 * np.sqrt(z[:, :, None] * z[None, :, :])
    )
    valid = (
        mask[:, :, None]
        & mask[None, :, :]
        & mask[:, None, :]
    )

    # Four damped mean-field transit estimates; no topology optimization.
    # Include a direct option, as otherwise all demand is counted as transit.
    temperature = 0.46
    damping = 0.55
    transit = np.zeros(n, dtype=float)
    log_affinity = np.log(affinity)
    log_edge_discount = np.log1p(0.18 * edge_affinity)
    direct_utility = np.log(0.30 + 1.15 * z) / temperature

    for iteration in range(4):
        pressure = np.maximum(
            1.0,
            0.72 * base_pressure / 0.60
            + 0.28 * (1.0 + transit / (incident + 1.0)),
        )
        utilities = (
            log_affinity
            - (0.72 + 0.06 * iteration)
            * np.log(pressure)[None, :, None]
            - log_edge_discount
        ) / temperature
        utilities = np.where(valid, utilities, -np.inf)

        maximum = np.maximum(
            utilities.max(axis=1), direct_utility
        )
        route_mass = np.exp(utilities - maximum[:, None, :])
        direct_mass = np.exp(direct_utility - maximum)
        denominator = route_mass.sum(axis=1) + direct_mass
        probabilities = route_mass / denominator[:, None, :]
        updated = (probabilities * z[:, None, :]).sum(axis=(0, 2))
        transit = (1.0 - damping) * transit + damping * updated

    transit_reference = max(1.0, float(incident.mean()))
    route_pressure = 1.0 + transit / transit_reference
    node_pressure = np.maximum(
        0.60,
        0.62 * incident / peak_incident
        + 0.38 * route_pressure / float(route_pressure.max()),
    )

    positive = z[z > 0.0]
    heavy_cutoff = float(np.quantile(positive, 0.73))
    support = 0.68 * z + 0.32 * root

    # Keep expected route credit from growing with the larger candidate pool.
    # The bounded taper avoids overwhelming direct-link evidence at large n.
    size_factor = float(np.clip((12.0 / n) ** 0.20, 0.86, 1.05))
    route_strength = 0.42 * size_factor
    overlap_fraction = 0.18

    exposure = np.zeros((n, n), dtype=float)

    def add_pair(first, second, weight):
        if weight == 0.0:
            return
        key = (
            (first, second)
            if first < second
            else (second, first)
        )
        quadratic[key] = quadratic.get(key, 0.0) + float(weight)

    for i in range(n):
        for k in range(n):
            if i == k or z[i, k] <= 0.0:
                continue

            heavy = z[i, k] >= heavy_cutoff
            ranked = []
            for j in range(n):
                if j == i or j == k:
                    continue
                edge = float(edge_affinity[i, j, k])
                pressure = float(node_pressure[j])
                rank = (
                    float(affinity[i, j, k])
                    / pressure ** 0.82
                    / (1.0 + 0.10 * edge)
                )
                ranked.append((rank, j, pressure, edge))
            ranked.sort(key=lambda item: (-item[0], item[1]))

            # Fixed compact budget: no near-tie expansion.
            keep = min(len(ranked), 3 if heavy else 2)
            for position, (_, j, pressure, edge) in enumerate(ranked[:keep]):
                rank_boost = 1.0 + 0.055 * (keep - position - 1)
                magnitude = (
                    route_strength
                    * float(support[i, k])
                    * rank_boost
                    / pressure ** 0.55
                    / (1.0 + 0.12 * edge)
                )
                first, second = (i, j), (j, k)
                add_pair(first, second, -magnitude)
                exposure[i, j] += magnitude
                exposure[j, k] += magnitude

                # A quadratic approximation to diminishing route value:
                # an existing direct link reduces the need for either
                # constituent of a rewarded alternative.
                if heavy:
                    correction = overlap_fraction * magnitude
                    add_pair((i, k), first, correction)
                    add_pair((i, k), second, correction)

    mean_direct = float(direct[mask].mean())

    # Discourage degree-consuming backtracking, but retain mutual-demand links.
    reciprocal_strength = (
        0.32
        * mean_direct
        * max(0.85, float(np.log2(n) / np.log2(12.0)))
    )
    for i in range(n):
        for j in range(i + 1, n):
            mutual = min(float(z[i, j]), float(z[j, i]))
            add_pair(
                (i, j),
                (j, i),
                reciprocal_strength * (1.0 - 0.65 * mutual),
            )

    # Weighted exposure is less sensitive than route counts to weak OD pairs.
    # Correct only the upper tail, preserving the direct-demand ranking.
    active = exposure[mask]
    active = active[active > 0.0]
    concentration = 0.0
    if active.size:
        threshold = float(np.quantile(active, 0.80))
        maximum = float(active.max())
        mean_exposure = float(active.mean())
        concentration = float(np.clip(
            float(active.std()) / max(mean_exposure, 1e-12),
            0.0,
            1.0,
        ))
        if maximum > threshold:
            for i in range(n):
                for j in range(n):
                    if i != j and exposure[i, j] > threshold:
                        excess = (
                            (float(exposure[i, j]) - threshold)
                            / (maximum - threshold)
                        )
                        linear[(i, j)] += 0.10 * mean_direct * excess

    # The evaluator alone supplies degree constraints.
    pressure_cv = float(incident.std()) / max(
        float(incident.mean()), 1e-12
    )
    penalty_weight = float(np.clip(
        1.24
        + 0.14 * min(pressure_cv, 1.0)
        + 0.06 * concentration,
        1.24,
        1.44,
    ))

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
