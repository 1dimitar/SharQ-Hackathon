"""Surrogate QUBO terms for QOBLIB 08 network design. ShinkaEvolve edits the EVOLVE block."""

import time

# EVOLVE-BLOCK-START
import numpy as np


def surrogate_terms(demand):
    """Build demand-aware direct-link and sparse two-hop QUBO terms."""
    traffic = np.asarray(demand, dtype=float)
    n = int(traffic.shape[0])

    traffic = np.maximum(np.nan_to_num(traffic, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    traffic = traffic.copy()
    np.fill_diagonal(traffic, 0.0)

    scale = float(np.max(traffic)) if traffic.size else 0.0
    if scale <= 0.0:
        scale = 1.0
    z = traffic / scale

    row_sum = traffic.sum(axis=1)
    col_sum = traffic.sum(axis=0)
    row_share = traffic / np.maximum(row_sum[:, None], 1e-12)
    col_share = traffic / np.maximum(col_sum[None, :], 1e-12)

    # Transit pressure is high for nodes that already generate or receive much
    # traffic. A floor avoids making any node categorically unusable.
    incident = np.maximum(row_sum, col_sum)
    max_incident = float(np.max(incident)) if incident.size else 0.0
    if max_incident > 0.0:
        pressure = 0.48 + 0.72 * incident / max_incident
    else:
        pressure = np.ones(n, dtype=float)

    # Direct-link score: soften the largest demands while giving a modest
    # bonus to links that account for a substantial share of both endpoints.
    share_bonus = np.sqrt(np.maximum(row_share * col_share, 0.0))
    direct_score = 0.62 * np.power(z, 0.78) + 0.24 * np.sqrt(z) + 0.14 * share_bonus

    linear = {
        (i, j): -float(direct_score[i, j])
        for i in range(n)
        for j in range(n)
        if i != j
    }

    positive = traffic[traffic > 0.0]
    if positive.size:
        cutoff = float(np.quantile(positive, 0.68))
    else:
        cutoff = 0.0

    quadratic = {}

    def add_pair(a, b, value):
        # Products are symmetric, so store one canonical key and aggregate
        # contributions should any terms coincide.
        key = (a, b) if a < b else (b, a)
        quadratic[key] = quadratic.get(key, 0.0) + float(value)

    for i in range(n):
        for k in range(n):
            if i == k or traffic[i, k] <= 0.0:
                continue

            demand_score = 0.68 * z[i, k] ** 0.74 + 0.32 * np.sqrt(z[i, k])
            # A strong direct demand is already represented in the linear
            # score, so reduce (but do not remove) its indirect-path reward.
            indirect_factor = 1.0 - 0.32 * np.sqrt(z[i, k])
            support = float(demand_score * indirect_factor)

            candidates = []
            for j in range(n):
                if j == i or j == k:
                    continue

                # Prefer a transit node for which both legs have endpoint
                # affinity, but discount nodes with high aggregate pressure.
                first_affinity = 0.56 * np.sqrt(row_share[i, j]) + 0.44 * np.sqrt(z[i, j])
                second_affinity = 0.56 * np.sqrt(col_share[j, k]) + 0.44 * np.sqrt(z[j, k])
                balance = np.sqrt(max(first_affinity * second_affinity, 0.0))
                edge_evidence = 0.20 * (np.sqrt(z[i, j]) + np.sqrt(z[j, k]))
                rank = (0.18 + balance + edge_evidence) / (pressure[j] ** 0.88)
                candidates.append((float(rank), j, float(balance)))

            candidates.sort(key=lambda item: (-item[0], item[1]))
            # Spend couplings on more alternatives for larger demands, but
            # keep the graph compact for the numerous small demands.
            keep = min(len(candidates), 3 if traffic[i, k] >= cutoff else 2)

            for position, (rank, j, path_affinity) in enumerate(candidates[:keep]):
                rank_factor = 1.12 - 0.10 * position
                affinity_factor = 0.45 + 0.85 * path_affinity
                weak_leg_support = min(z[i, j], z[j, k])
                leg_support_factor = 0.45 + 0.55 * np.sqrt(weak_leg_support)
                route_scale = (
                    support * rank_factor * affinity_factor * leg_support_factor
                    / (pressure[j] ** 0.35)
                )
                first = (i, j)
                second = (j, k)

                # Reward routes more when both legs have endpoint-demand
                # evidence, while retaining a floor for plausible alternatives.
                route_reward = -0.37 * route_scale
                add_pair(first, second, route_reward)

                # Make direct-versus-indirect redundancy specific to the
                # affinity of this two-leg route.
                direct_redundancy = (
                    0.085 * support * (0.45 + 0.55 * np.sqrt(z[i, k]))
                    * (0.25 + 0.75 * path_affinity)
                )
                add_pair((i, k), first, direct_redundancy)
                add_pair((i, k), second, direct_redundancy)

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
