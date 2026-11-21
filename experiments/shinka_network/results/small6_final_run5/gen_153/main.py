"""Surrogate QUBO terms for QOBLIB 08 network design. ShinkaEvolve edits the EVOLVE block."""

import time

# EVOLVE-BLOCK-START
import numpy as np


def surrogate_terms(demand):
    """Build a direct-service, sparse-routing, and branch-diversity surrogate."""
    traffic = np.asarray(demand, dtype=float)
    n = int(traffic.shape[0])

    traffic = np.maximum(
        np.nan_to_num(traffic, nan=0.0, posinf=0.0, neginf=0.0), 0.0
    ).copy()
    np.fill_diagonal(traffic, 0.0)

    maximum = float(np.max(traffic)) if traffic.size else 0.0
    if maximum <= 0.0:
        maximum = 1.0
    z = traffic / maximum

    row_sum = traffic.sum(axis=1)
    col_sum = traffic.sum(axis=0)
    row_share = traffic / np.maximum(row_sum[:, None], 1e-12)
    col_share = traffic / np.maximum(col_sum[None, :], 1e-12)

    # Reward direct service, with softened demand and endpoint-share evidence.
    direct = (
        0.72 * np.power(z, 0.78)
        + 0.20 * np.sqrt(z)
        + 0.08 * np.sqrt(np.maximum(row_share * col_share, 0.0))
    )
    linear = {
        (i, j): -float(direct[i, j])
        for i in range(n)
        for j in range(n)
        if i != j
    }
    quadratic = {}

    def add_pair(a, b, value):
        if a == b or value == 0.0:
            return
        key = (a, b) if a < b else (b, a)
        quadratic[key] = quadratic.get(key, 0.0) + float(value)

    # Aggregate endpoint activity is a modest prior against routing all
    # plausible detours through the same busy transit node.
    incident = np.maximum(row_sum, col_sum)
    max_incident = max(float(np.max(incident)), 1e-12)
    pressure = 0.72 + 0.56 * incident / max_incident

    positive = traffic[traffic > 0.0]
    cutoff = float(np.quantile(positive, 0.72)) if positive.size else 0.0

    # Select only a few routes per OD pair. Their coupling strengths are
    # normalized over the selected alternatives rather than treating every
    # candidate route as if it carried the full demand.
    for i in range(n):
        for k in range(n):
            if i == k or traffic[i, k] <= 0.0:
                continue

            d = float(z[i, k])
            support = 0.68 * d ** 0.76 + 0.32 * np.sqrt(d)
            support *= 1.0 - 0.22 * np.sqrt(d)

            candidates = []
            for j in range(n):
                if j == i or j == k:
                    continue
                first = (
                    0.58 * np.sqrt(max(float(row_share[i, j]), 0.0))
                    + 0.42 * np.sqrt(float(z[i, j]))
                )
                second = (
                    0.58 * np.sqrt(max(float(col_share[j, k]), 0.0))
                    + 0.42 * np.sqrt(float(z[j, k]))
                )
                affinity = np.sqrt(max(first * second, 0.0))
                rank = (0.10 + affinity) / float(pressure[j]) ** 0.72
                candidates.append((rank, j, affinity))

            candidates.sort(key=lambda item: (-item[0], item[1]))
            keep = min(len(candidates), 3 if traffic[i, k] >= cutoff else 2)
            chosen = candidates[:keep]
            if not chosen:
                continue

            # A bounded soft allocation makes the strongest route matter most,
            # while leaving some support for distinct alternatives.
            ranks = np.asarray([max(c[0], 1e-12) for c in chosen], dtype=float)
            best_rank = max(float(ranks[0]), 1e-12)
            allocation = np.sqrt(ranks / best_rank)
            allocation /= max(float(allocation.sum()), 1e-12)

            for (_, j, affinity), fraction in zip(chosen, allocation):
                first_link = (i, j)
                second_link = (j, k)
                route_strength = (
                    support
                    * float(fraction)
                    * (0.55 + 0.75 * float(affinity))
                )
                add_pair(first_link, second_link, -0.34 * route_strength)

                # A direct link and a detour are partially substitutable for
                # the same OD pair; this avoids over-rewarding both together.
                redundancy = (
                    0.055
                    * support
                    * (0.35 + 0.65 * np.sqrt(d))
                    * (0.40 + 0.60 * float(affinity))
                )
                add_pair((i, k), first_link, redundancy)
                add_pair((i, k), second_link, redundancy)

    # Compact branch-diversity couplings. Neighbor nodes with similar traffic
    # profiles are more likely to offer redundant onward (or upstream) service.
    # Retain only a few strongest overlaps per center to limit graph density.
    row_norm = np.sqrt(np.sum(z * z, axis=1))
    col_norm = np.sqrt(np.sum(z * z, axis=0))
    row_profile = z / np.maximum(row_norm[:, None], 1e-12)
    col_profile = z / np.maximum(col_norm[None, :], 1e-12)

    for center in range(n):
        neighbors = [v for v in range(n) if v != center]

        outgoing_overlaps = []
        for pos, a in enumerate(neighbors):
            for b in neighbors[pos + 1:]:
                # Similar onward demand profiles make two outgoing branches
                # less complementary.
                overlap = float(np.dot(row_profile[a], row_profile[b]))
                if overlap > 0.18:
                    outgoing_overlaps.append((overlap, a, b))

        outgoing_overlaps.sort(key=lambda item: (-item[0], item[1], item[2]))
        activity = float(row_sum[center] / max(float(np.max(row_sum)), 1e-12))
        for overlap, a, b in outgoing_overlaps[:3]:
            add_pair(
                (center, a),
                (center, b),
                0.10 * overlap * (0.45 + 0.55 * activity),
            )

        incoming_overlaps = []
        for pos, a in enumerate(neighbors):
            for b in neighbors[pos + 1:]:
                # Similar source profiles can make two incoming branches
                # competing ways to bring comparable traffic to this node.
                overlap = float(np.dot(col_profile[:, a], col_profile[:, b]))
                if overlap > 0.18:
                    incoming_overlaps.append((overlap, a, b))

        incoming_overlaps.sort(key=lambda item: (-item[0], item[1], item[2]))
        activity = float(col_sum[center] / max(float(np.max(col_sum)), 1e-12))
        for overlap, a, b in incoming_overlaps[:3]:
            add_pair(
                (a, center),
                (b, center),
                0.09 * overlap * (0.45 + 0.55 * activity),
            )

    # Degree penalties are supplied by the evaluator. This value keeps their
    # scale useful relative to the demand-dependent objective.
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
