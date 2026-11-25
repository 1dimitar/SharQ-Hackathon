"""Surrogate QUBO terms for QOBLIB 08 network design. ShinkaEvolve edits the EVOLVE block."""

import time

# EVOLVE-BLOCK-START
import numpy as np


def _prepare_demand(demand):
    """Return finite nonnegative traffic, normalized traffic, and size."""
    traffic = np.asarray(demand, dtype=float)
    n = len(traffic)
    traffic = np.nan_to_num(traffic, nan=0.0, posinf=0.0, neginf=0.0)
    traffic = np.maximum(traffic, 0.0).copy()
    np.fill_diagonal(traffic, 0.0)

    scale = float(traffic.max()) if traffic.size else 0.0
    if scale > 0.0:
        normalized = traffic / scale
    else:
        normalized = traffic.copy()
    return traffic, normalized, n


def _add_quadratic(quadratic, first, second, weight):
    """Accumulate a symmetric QUBO interaction using canonical link keys."""
    if first == second or abs(weight) < 1e-14:
        return
    key = (first, second) if first < second else (second, first)
    quadratic[key] = quadratic.get(key, 0.0) + float(weight)


def _direct_terms(normalized, traffic):
    """Reward direct service, mildly accounting for endpoint traffic."""
    n = len(normalized)
    row = traffic.sum(axis=1)
    col = traffic.sum(axis=0)
    row_max = max(float(row.max()), 1e-12)
    col_max = max(float(col.max()), 1e-12)

    linear = {}
    roots = np.sqrt(normalized)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            endpoint_need = 0.5 * (row[i] / row_max + col[j] / col_max)
            service = 0.75 * normalized[i, j] + 0.25 * roots[i, j]
            linear[(i, j)] = -float(service * (0.94 + 0.06 * endpoint_need))
    return linear, row, col


def _node_capacity_terms(quadratic, row, col):
    """Add modest pairwise penalties for concentrated incident-link choices."""
    n = len(row)
    row_scale = np.maximum(row / max(float(row.max()), 1e-12), 0.08)
    col_scale = np.maximum(col / max(float(col.max()), 1e-12), 0.08)
    node_load = np.maximum(row, col)
    pressure = np.maximum(
        0.55, node_load / max(float(node_load.max()), 1e-12)
    )

    for v in range(n):
        factor = float(pressure[v]) ** 0.8

        # Two selected links entering the same node can concentrate transit.
        incoming = [(u, v) for u in range(n) if u != v]
        for p in range(len(incoming)):
            for q in range(p + 1, len(incoming)):
                u, w = incoming[p][0], incoming[q][0]
                weight = 0.025 * factor * np.sqrt(row_scale[u] * row_scale[w])
                _add_quadratic(quadratic, incoming[p], incoming[q], weight)

        # Likewise, account for competing choices of outgoing links.
        outgoing = [(v, u) for u in range(n) if u != v]
        for p in range(len(outgoing)):
            for q in range(p + 1, len(outgoing)):
                a, b = outgoing[p][1], outgoing[q][1]
                weight = 0.025 * factor * np.sqrt(col_scale[a] * col_scale[b])
                _add_quadratic(quadratic, outgoing[p], outgoing[q], weight)

    return pressure


def _route_terms(quadratic, normalized, pressure):
    """Reward a small set of two-hop routes supported by both of their legs."""
    n = len(normalized)
    positive = normalized[normalized > 0.0]
    cutoff = float(np.quantile(positive, 0.72)) if positive.size else 0.0

    for i in range(n):
        for k in range(n):
            if i == k:
                continue
            direct = float(normalized[i, k])
            if direct <= 0.0:
                continue

            # Stronger OD demand receives more route attention. The direct
            # term in the denominator prevents routes from overwhelming an
            # already attractive direct link.
            support = 0.70 * direct + 0.30 * np.sqrt(direct)
            candidates = []

            for j in range(n):
                if j == i or j == k:
                    continue
                first_leg = float(normalized[i, j])
                second_leg = float(normalized[j, k])
                if first_leg <= 0.0 or second_leg <= 0.0:
                    continue

                geometric = np.sqrt(first_leg * second_leg)
                balance = 2.0 * geometric / (0.12 + first_leg + second_leg)
                leg_evidence = geometric * (0.55 + 0.45 * balance)

                # Prefer routes whose two legs have independent demand
                # evidence; penalize busy transit nodes and strong direct OD.
                rank = leg_evidence / (float(pressure[j]) ** 0.45)
                direct_discount = 0.30 + 0.70 * np.sqrt(direct)
                candidates.append((float(rank / direct_discount), j, balance))

            if not candidates:
                continue
            candidates.sort(key=lambda item: (-item[0], item[1]))

            # Usually use two alternatives. Keep a third only when it is
            # competitive and the OD demand is above the common range.
            keep = min(2, len(candidates))
            if (
                len(candidates) > 2
                and direct >= cutoff
                and candidates[2][0] >= 0.76 * candidates[0][0]
            ):
                keep = 3

            best_rank = max(candidates[0][0], 1e-12)
            for position, (rank, j, balance) in enumerate(candidates[:keep]):
                relative = rank / best_rank
                route_quality = 0.75 + 0.25 * min(1.0, balance)
                reward = (
                    0.27
                    * float(support)
                    * route_quality
                    * (0.78 + 0.22 * relative)
                    / (float(pressure[j]) ** 0.22)
                )
                _add_quadratic(
                    quadratic, (i, j), (j, k), -reward
                )


def surrogate_terms(demand):
    """Build direct-service, route-evidence, and capacity QUBO terms."""
    traffic, normalized, n = _prepare_demand(demand)

    if not traffic.size or float(traffic.max()) <= 0.0:
        linear = {
            (i, j): 0.0
            for i in range(n)
            for j in range(n)
            if i != j
        }
        return linear, {}, 1.25

    linear, row, col = _direct_terms(normalized, traffic)
    quadratic = {}
    pressure = _node_capacity_terms(quadratic, row, col)
    _route_terms(quadratic, normalized, pressure)

    return linear, quadratic, 1.25


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
