"""Surrogate QUBO terms for QOBLIB 08 network design. ShinkaEvolve edits the EVOLVE block."""

import time

# EVOLVE-BLOCK-START
import numpy as np


def surrogate_terms(demand):
    """Build direct-link rewards and sparse, demand-aware route couplings."""
    traffic = np.asarray(demand, dtype=float)
    if traffic.ndim != 2 or traffic.shape[0] != traffic.shape[1]:
        raise ValueError("demand must be a square matrix")

    n = int(traffic.shape[0])
    traffic = np.nan_to_num(
        traffic, copy=True, nan=0.0, posinf=0.0, neginf=0.0
    )
    np.maximum(traffic, 0.0, out=traffic)
    np.fill_diagonal(traffic, 0.0)

    maximum = float(traffic.max()) if traffic.size else 0.0
    if maximum > 0.0:
        z = traffic / maximum
    else:
        z = traffic

    root = np.sqrt(z)
    row_totals = z.sum(axis=1)
    col_totals = z.sum(axis=0)
    row_share = z / np.maximum(row_totals[:, None], 1e-15)
    col_share = z / np.maximum(col_totals[None, :], 1e-15)

    incident = np.maximum(row_totals, col_totals)
    peak = float(incident.max()) if incident.size else 0.0
    if peak > 0.0:
        pressure = 0.50 + 0.70 * incident / peak
    else:
        pressure = np.ones(n, dtype=float)

    # The direct term remains the strongest simple signal for a link.
    direct_score = 0.75 * z + 0.25 * root
    linear = {
        (i, j): -float(direct_score[i, j])
        for i in range(n)
        for j in range(n)
        if i != j
    }

    quadratic = {}

    def add_pair(first, second, value):
        if first == second or abs(value) < 1e-14:
            return
        key = (first, second) if first < second else (second, first)
        quadratic[key] = quadratic.get(key, 0.0) + float(value)

    if maximum <= 0.0 or n < 3:
        return linear, quadratic, 1.15

    # A route should have support on both legs as well as affinity to the
    # source and destination. These scores are reused across all demands.
    out_affinity = 0.58 * np.sqrt(np.maximum(row_share, 0.0)) + 0.42 * root
    in_affinity = 0.58 * np.sqrt(np.maximum(col_share, 0.0)) + 0.42 * root

    positive = z[z > 0.0]
    third_route_cutoff = (
        float(np.quantile(positive, 0.64)) if positive.size else 0.0
    )

    # A modest size adjustment keeps route terms useful relative to the
    # linear terms as the number of selected links grows.
    route_size_factor = float(np.clip(np.sqrt(n / 12.0), 0.88, 1.30))
    route_position_weight = (1.0, 0.66, 0.40)

    for source in range(n):
        for target in range(n):
            amount = float(z[source, target])
            if source == target or amount <= 0.0:
                continue

            # Strong direct demands already receive a linear reward, so
            # somewhat reduce their additional indirect-route incentive.
            support = (
                0.68 * amount ** 0.74 + 0.32 * root[source, target]
            )
            support *= 1.0 - 0.28 * root[source, target]

            candidates = []
            for middle in range(n):
                if middle == source or middle == target:
                    continue

                affinity = float(np.sqrt(max(
                    out_affinity[source, middle]
                    * in_affinity[middle, target],
                    0.0,
                )))
                leg_evidence = 0.18 * (
                    root[source, middle] + root[middle, target]
                )
                rank = (
                    0.16 + affinity + leg_evidence
                ) / (pressure[middle] ** 0.82)
                candidates.append((float(rank), middle, affinity))

            candidates.sort(key=lambda item: (-item[0], item[1]))
            keep = 3 if amount >= third_route_cutoff else 2
            keep = min(keep, len(candidates))

            # Retain a third alternative only when the demand merits it;
            # ties are deterministic and no random choices are used.
            for position, (_, middle, affinity) in enumerate(
                candidates[:keep]
            ):
                first = (source, middle)
                second = (middle, target)

                # Affinity and transit capacity moderate the route reward.
                route_quality = (
                    0.48 + 0.72 * affinity
                ) / (pressure[middle] ** 0.45)
                reward = (
                    0.39
                    * route_size_factor
                    * support
                    * route_position_weight[position]
                    * route_quality
                )
                add_pair(first, second, -reward)

                # A small positive coupling limits double-counting a strong
                # direct link and one of its indirect substitutes.
                redundancy = (
                    0.050
                    * support
                    * (0.35 + 0.65 * root[source, target])
                    * (0.40 + 0.60 * affinity)
                )
                add_pair((source, target), first, redundancy)
                add_pair((source, target), second, redundancy)

    return linear, quadratic, 1.15


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
