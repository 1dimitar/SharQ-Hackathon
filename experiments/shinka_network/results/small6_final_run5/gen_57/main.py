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
        hard_cutoff = float(np.quantile(positive, 0.82))
    else:
        cutoff = 0.0
        hard_cutoff = 0.0

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

            for position, (rank, j, balance) in enumerate(candidates[:keep]):
                rank_factor = 1.12 - 0.10 * position
                leg_gate = 0.55 + 0.45 * np.sqrt(z[i, j] * z[j, k])
                route_scale = (
                    support * rank_factor * leg_gate
                    / (pressure[j] ** 0.35)
                )
                first = (i, j)
                second = (j, k)

                # Negative coupling rewards completion of a plausible
                # two-hop route.
                route_reward = -0.37 * route_scale
                add_pair(first, second, route_reward)

                # A weak positive interaction models redundancy: when the
                # direct link is present, both legs of this same alternative
                # are less valuable specifically for this demand.
                direct_redundancy = 0.085 * support * (0.45 + 0.55 * np.sqrt(z[i, k]))
                add_pair((i, k), first, direct_redundancy)
                add_pair((i, k), second, direct_redundancy)

            # Add at most one three-hop option for a difficult, high-demand
            # pair: its best two-hop candidate has weak balanced leg support.
            if (
                traffic[i, k] > 0.0
                and traffic[i, k] >= hard_cutoff
                and max(candidate[2] for candidate in candidates) < 0.30
            ):
                three_hop = []
                for a in range(n):
                    if a == i or a == k:
                        continue
                    for b in range(n):
                        if b == i or b == k or b == a:
                            continue

                        leg0 = 0.56 * np.sqrt(row_share[i, a]) + 0.44 * np.sqrt(z[i, a])
                        leg1 = 0.56 * np.sqrt(row_share[a, b]) + 0.44 * np.sqrt(z[a, b])
                        leg2 = 0.56 * np.sqrt(row_share[b, k]) + 0.44 * np.sqrt(z[b, k])
                        weakest_ratio = min(leg0, leg1, leg2) / max(
                            leg0, leg1, leg2, 1e-12
                        )
                        quality = (
                            (leg0 * leg1 * leg2) ** (1.0 / 3.0)
                            * (0.55 + 0.45 * weakest_ratio)
                            / ((pressure[a] * pressure[b]) ** 0.30)
                        )
                        three_hop.append((float(quality), a, b))

                if three_hop:
                    quality, a, b = max(
                        three_hop, key=lambda item: (item[0], -item[1], -item[2])
                    )
                    reward = 0.045 * support * quality
                    add_pair((i, a), (a, b), -reward)
                    add_pair((a, b), (b, k), -reward)

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
