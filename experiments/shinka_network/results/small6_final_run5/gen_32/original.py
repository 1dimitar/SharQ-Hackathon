"""Surrogate QUBO terms for QOBLIB 08 network design. ShinkaEvolve edits the EVOLVE block."""

import time

# EVOLVE-BLOCK-START
import numpy as np


def surrogate_terms(demand):
    """Return demand-aware link rewards and sparse route/congestion couplings."""
    traffic = np.asarray(demand, dtype=float)
    n = len(traffic)

    traffic = np.maximum(traffic, 0.0).copy()
    np.fill_diagonal(traffic, 0.0)
    maximum = float(traffic.max()) if traffic.size else 0.0
    if maximum <= 0.0:
        linear = {
            (i, j): 0.0
            for i in range(n) for j in range(n) if i != j
        }
        return linear, {}, 1.30

    t = traffic / maximum
    root = np.sqrt(t)
    row = traffic.sum(axis=1)
    col = traffic.sum(axis=0)
    node_load = np.maximum(row, col)
    load_max = max(float(node_load.max()), 1e-12)
    pressure = np.maximum(0.55, node_load / load_max)

    # Direct links retain a strong service reward, with a mild adjustment
    # for endpoints that have unusually high traffic.
    linear = {}
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            endpoint_need = 0.5 * (
                row[i] / max(float(row.max()), 1e-12)
                + col[j] / max(float(col.max()), 1e-12)
            )
            value = 0.73 * t[i, j] + 0.27 * root[i, j]
            linear[(i, j)] = -float(value * (0.92 + 0.08 * endpoint_need))

    quadratic = {}

    def add_q(a, b, weight):
        if a == b or abs(weight) < 1e-14:
            return
        key = (a, b) if a < b else (b, a)
        quadratic[key] = quadratic.get(key, 0.0) + float(weight)

    # Sparse transit-capacity proxy: multiple selected links entering or
    # leaving a heavily loaded node tend to concentrate transit there.
    row_scale = np.maximum(row / max(float(row.max()), 1e-12), 0.10)
    col_scale = np.maximum(col / max(float(col.max()), 1e-12), 0.10)
    for v in range(n):
        in_links = [(u, v) for u in range(n) if u != v]
        out_links = [(v, w) for w in range(n) if w != v]
        node_factor = float(pressure[v]) ** 1.2

        for a in range(len(in_links)):
            i, _ = in_links[a]
            for b in range(a + 1, len(in_links)):
                k, _ = in_links[b]
                strength = 0.075 * node_factor * np.sqrt(row_scale[i] * row_scale[k])
                add_q(in_links[a], in_links[b], strength)

        for a in range(len(out_links)):
            _, j = out_links[a]
            for b in range(a + 1, len(out_links)):
                _, k = out_links[b]
                strength = 0.075 * node_factor * np.sqrt(col_scale[j] * col_scale[k])
                add_q(out_links[a], out_links[b], strength)

    positive = t[t > 0.0]
    cutoff = float(np.quantile(positive, 0.75)) if positive.size else 0.0

    # Expansion point: the exact mean link occupancy of a valid topology.
    route_density = 2.0 / (n - 1)

    # For each OD pair, rank possible two-hop intermediates by balanced
    # leg evidence relative to transit pressure. Keep extra choices only
    # when the rank margin indicates real ambiguity.
    for i in range(n):
        for k in range(n):
            if i == k or t[i, k] <= 0.0:
                continue

            support = 0.72 * t[i, k] + 0.28 * root[i, k]
            candidates = []
            for j in range(n):
                if j == i or j == k:
                    continue
                a = float(t[i, j])
                b = float(t[j, k])
                geometric = np.sqrt(a * b)
                leg_evidence = 0.58 * geometric + 0.21 * (a + b)
                rank = (0.12 + leg_evidence) / (float(pressure[j]) ** 0.72)
                balance = 2.0 * geometric / (0.15 + a + b)
                candidates.append((float(rank), j, float(balance)))

            candidates.sort(key=lambda item: (-item[0], item[1]))
            if not candidates:
                continue

            keep = min(len(candidates), 4 if t[i, k] >= cutoff else 2)
            base_keep = keep
            best_rank = max(candidates[0][0], 1e-12)

            # Increase route coverage only if the next-ranked candidate is
            # close to the current frontier and also competitive with the best.
            while keep < min(len(candidates), base_keep + 2):
                next_rank = candidates[keep][0]
                frontier = candidates[keep - 1][0]
                if next_rank / max(frontier, 1e-12) < 0.82:
                    break
                if next_rank / best_rank < 0.58:
                    break
                keep += 1

            for rank, j, balance in candidates[:keep]:
                relative = rank / best_rank
                # Avoid making one route dominate just because its rank is
                # marginally higher; leg balance still matters.
                route_factor = (0.72 + 0.28 * relative) * (0.82 + 0.36 * balance)
                pressure_discount = float(pressure[j]) ** 0.30
                reward = 0.34 * float(support) * route_factor / pressure_discount

                # Reward service not already supplied by the direct shortcut:
                # -reward * (1 - direct) * first * second.
                # Expand the cubic about occupancy p and discard only the
                # third-order residual. Constants can also be discarded.
                first, second, direct = (i, j), (j, k), (i, k)
                p = route_density
                add_q(first, second, -reward * (1.0 - p))
                add_q(direct, first, reward * p)
                add_q(direct, second, reward * p)
                correction = reward * p * p
                linear[first] -= correction
                linear[second] -= correction
                linear[direct] -= correction

    return linear, quadratic, 1.30


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
