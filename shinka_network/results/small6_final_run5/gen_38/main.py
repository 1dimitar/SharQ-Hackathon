"""Surrogate QUBO terms for QOBLIB 08 network design. ShinkaEvolve edits the EVOLVE block."""

import time

# EVOLVE-BLOCK-START
import numpy as np


def surrogate_terms(demand):
    traffic = np.asarray(demand, dtype=float)
    n = int(traffic.shape[0])

    traffic = np.maximum(
        np.nan_to_num(traffic, nan=0.0, posinf=0.0, neginf=0.0), 0.0
    ).copy()
    np.fill_diagonal(traffic, 0.0)

    largest = float(np.max(traffic)) if traffic.size else 0.0
    if largest <= 0.0:
        largest = 1.0
    z = traffic / largest

    rows = traffic.sum(axis=1)
    cols = traffic.sum(axis=0)
    row_share = traffic / np.maximum(rows[:, None], 1e-15)
    col_share = traffic / np.maximum(cols[None, :], 1e-15)

    node_load = np.maximum(rows, cols)
    load_scale = float(np.max(node_load)) if n else 0.0
    if load_scale > 0.0:
        pressure = 0.55 + 0.75 * node_load / load_scale
    else:
        pressure = np.ones(n, dtype=float)

    # A softened direct-link value, with a small endpoint-share contribution.
    share_value = np.sqrt(np.maximum(row_share * col_share, 0.0))
    direct = 0.68 * np.power(z, 0.76) + 0.22 * np.sqrt(z) + 0.10 * share_value
    linear = {
        (i, j): -float(direct[i, j])
        for i in range(n)
        for j in range(n)
        if i != j
    }

    quadratic = {}

    def add_pair(a, b, value):
        if a == b:
            return
        key = (a, b) if a < b else (b, a)
        quadratic[key] = quadratic.get(key, 0.0) + float(value)

    # Select a small, adaptive set of two-hop alternatives per positive OD pair.
    positive = traffic[traffic > 0.0]
    cutoff = float(np.quantile(positive, 0.70)) if positive.size else 0.0

    for i in range(n):
        for k in range(n):
            if i == k or traffic[i, k] <= 0.0:
                continue

            demand_value = 0.70 * z[i, k] ** 0.72 + 0.30 * np.sqrt(z[i, k])
            # Strong direct traffic receives less indirect-route credit.
            route_support = demand_value * (1.0 - 0.28 * np.sqrt(z[i, k]))

            candidates = []
            for j in range(n):
                if j == i or j == k:
                    continue

                # A route should have some evidence on both legs, but a floor
                # retains viable routes whose constituent demands are small.
                leg_a = 0.58 * np.sqrt(row_share[i, j]) + 0.42 * np.sqrt(z[i, j])
                leg_b = 0.58 * np.sqrt(col_share[j, k]) + 0.42 * np.sqrt(z[j, k])
                balance = np.sqrt(max(leg_a * leg_b, 0.0))
                rank = (0.16 + balance) / (pressure[j] ** 0.82)
                candidates.append((float(rank), j, float(balance)))

            candidates.sort(key=lambda item: (-item[0], item[1]))
            if not candidates:
                continue

            keep = 3 if traffic[i, k] >= cutoff else 2
            keep = min(keep, len(candidates))
            # Add one more route only when it is a credible alternative to
            # the weakest currently selected candidate.
            if keep < len(candidates):
                boundary = candidates[keep - 1][0]
                if candidates[keep][0] >= 0.86 * boundary:
                    keep += 1

            for pos, (_, j, balance) in enumerate(candidates[:keep]):
                rank_factor = 1.08 - 0.09 * pos
                affinity_factor = 0.55 + 0.75 * balance
                reward = (
                    0.34 * route_support * rank_factor * affinity_factor
                    / (pressure[j] ** 0.30)
                )

                first = (i, j)
                second = (j, k)
                add_pair(first, second, -reward)

                # Avoid over-crediting an indirect route when its OD pair has
                # a direct link: this helps rank link exchanges consistently.
                redundancy = (
                    0.070 * route_support
                    * (0.35 + 0.65 * np.sqrt(z[i, k]))
                    * (0.35 + 0.65 * balance)
                )
                add_pair((i, k), first, redundancy)
                add_pair((i, k), second, redundancy)

    # Mild local congestion proxy. For each node, consider only a handful of
    # its strongest direct-demand incident links; the resulting sparse pairs
    # discourage concentrating high direct loads at the same endpoint.
    for v in range(n):
        outgoing = [(float(direct[v, j]), (v, j)) for j in range(n) if j != v]
        incoming = [(float(direct[i, v]), (i, v)) for i in range(n) if i != v]
        outgoing.sort(key=lambda item: (-item[0], item[1]))
        incoming.sort(key=lambda item: (-item[0], item[1]))

        for group, node_factor in ((outgoing[:6], 0.85), (incoming[:6], 1.0)):
            for a in range(len(group)):
                va, ea = group[a]
                for b in range(a + 1, len(group)):
                    vb, eb = group[b]
                    # Geometric scaling avoids making a single large demand
                    # dominate the interaction field.
                    congestion = (
                        0.045 * node_factor * np.sqrt(max(va * vb, 0.0))
                        * (0.70 + 0.30 * pressure[v] / 1.30)
                    )
                    add_pair(ea, eb, congestion)

    return linear, quadratic, 1.10


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
