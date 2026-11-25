"""Surrogate QUBO terms for QOBLIB 08 network design. ShinkaEvolve edits the EVOLVE block."""

import time

# EVOLVE-BLOCK-START
import numpy as np


def surrogate_terms(demand):
    """Build a sparse QUBO from direct demand, transit potential, and route diversity."""
    matrix = np.asarray(demand, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("demand must be a square matrix")

    n = matrix.shape[0]
    traffic = np.nan_to_num(matrix, copy=True, nan=0.0, posinf=0.0, neginf=0.0)
    np.maximum(traffic, 0.0, out=traffic)
    np.fill_diagonal(traffic, 0.0)

    scale = float(traffic.max()) if traffic.size else 0.0
    if scale > 0.0:
        d = traffic / scale
    else:
        d = np.zeros_like(traffic)

    root = np.sqrt(d)
    outgoing = d.sum(axis=1)
    incoming = d.sum(axis=0)
    max_out = max(float(outgoing.max()), 1e-12)
    max_in = max(float(incoming.max()), 1e-12)

    # Direct demand is valuable, while a modest bridge term favors links
    # joining nodes with substantial incoming and outgoing traffic.
    direct_value = 0.64 * d + 0.36 * root
    bridge = np.sqrt(
        (incoming / max_in)[:, None] * (outgoing / max_out)[None, :]
    )
    linear = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                linear[(i, j)] = -float(direct_value[i, j] + 0.10 * bridge[i, j])

    # Form plausible two-hop routes using evidence on both legs. An initial
    # transit-load estimate discounts routes through already busy nodes.
    incident = np.maximum(outgoing, incoming)
    base_pressure = incident / max(float(incident.max()), 1e-12)
    base_pressure = np.maximum(base_pressure, 0.55)

    routes_by_od = {}
    transit_load = np.zeros(n, dtype=float)

    for i in range(n):
        for k in range(n):
            if i == k or d[i, k] <= 0.0:
                continue

            candidates = []
            for j in range(n):
                if j == i or j == k:
                    continue
                a = float(d[i, j])
                b = float(d[j, k])
                if a <= 0.0 or b <= 0.0:
                    continue

                geometric = np.sqrt(a * b)
                balance = 2.0 * geometric / (a + b + 1e-12)
                quality = geometric * (0.65 + 0.35 * balance)
                if quality >= 0.012:
                    rank = quality / (float(base_pressure[j]) ** 0.60)
                    candidates.append((float(rank), j, float(quality)))

            candidates.sort(key=lambda item: (-item[0], item[1]))
            routes_by_od[(i, k)] = candidates

            # A bounded fractional assignment estimates which transit nodes
            # are likely to attract traffic, without giving every route equal
            # weight.
            for rank, j, _ in candidates[:2]:
                transit_load[j] += float(d[i, k]) * rank

    if float(transit_load.max()) > 0.0:
        pressure = base_pressure + 0.30 * transit_load / float(transit_load.max())
    else:
        pressure = base_pressure
    pressure = np.clip(pressure, 0.55, 1.8)

    positive = d[d > 0.0]
    heavy_cut = float(np.quantile(positive, 0.80)) if positive.size else 1.0
    quadratic = {}

    def add_pair(first, second, weight):
        if first == second or weight == 0.0:
            return
        key = (first, second) if first < second else (second, first)
        quadratic[key] = quadratic.get(key, 0.0) + float(weight)

    # Allocate a bounded route reward across a small set of supported paths.
    # This avoids multiplying an OD pair's reward simply because it has many
    # plausible intermediates.
    for (i, k), candidates in routes_by_od.items():
        if not candidates:
            continue

        reranked = [
            (quality / (float(pressure[j]) ** 0.72), j, quality)
            for _, j, quality in candidates
        ]
        reranked.sort(key=lambda item: (-item[0], item[1]))

        best_rank = max(reranked[0][0], 1e-12)
        reranked = [
            item for item in reranked
            if item[0] >= 0.28 * best_rank and item[2] >= 0.025
        ]
        if not reranked:
            continue

        demand_level = float(d[i, k])
        limit = 3 if demand_level >= heavy_cut else 2
        keep = min(limit, len(reranked))

        # Admit one extra near-tied route for ordinary demands, but only when
        # its support is competitive with the leading path.
        if (
            limit == 2
            and keep == 2
            and len(reranked) > 2
            and reranked[2][0] >= 0.90 * reranked[1][0]
            and reranked[2][0] >= 0.58 * best_rank
        ):
            keep = 3

        selected = reranked[:keep]
        strengths = np.asarray(
            [max(item[0], 1e-12) ** 1.45 for item in selected],
            dtype=float,
        )
        probabilities = strengths / float(strengths.sum())

        support = 0.68 * demand_level + 0.32 * float(root[i, k])
        indirect_fraction = 0.25 + 0.48 * (1.0 - float(root[i, k]))
        route_budget = 0.32 * support * indirect_fraction

        for (_, j, _), probability in zip(selected, probabilities):
            add_pair(
                (i, j),
                (j, k),
                -route_budget * float(probability),
            )

    # Penalize unusually redundant choices of two outgoing or two incoming
    # branches. Row/column profile overlap measures whether those branches
    # offer similar onward reachability or origins.
    mean_direct = float(direct_value.sum()) / max(1, n * (n - 1))
    row_profiles = np.zeros_like(d)
    col_profiles = np.zeros_like(d)
    row_sums = d.sum(axis=1)
    col_sums = d.sum(axis=0)

    for a in range(n):
        if row_sums[a] > 0.0:
            row_profiles[a] = np.sqrt(d[a] / row_sums[a])
        if col_sums[a] > 0.0:
            col_profiles[:, a] = np.sqrt(d[:, a] / col_sums[a])

    for center in range(n):
        branches = [v for v in range(n) if v != center]

        for profiles, inbound in ((row_profiles, False), (col_profiles, True)):
            pairs = []
            for pos, a in enumerate(branches):
                for b in branches[pos + 1:]:
                    overlap = float(np.dot(profiles[a], profiles[b]))
                    pairs.append((overlap, a, b))

            if not pairs:
                continue

            average_overlap = sum(item[0] for item in pairs) / len(pairs)
            pairs.sort(key=lambda item: (-item[0], item[1], item[2]))
            center_pressure = float(
                np.sqrt(max(incident[center], 0.0) / max(float(incident.max()), 1e-12))
            )

            added = 0
            for overlap, a, b in pairs:
                excess = overlap - average_overlap - 0.035
                if excess <= 0.0:
                    continue
                first = (a, center) if inbound else (center, a)
                second = (b, center) if inbound else (center, b)
                add_pair(
                    first,
                    second,
                    0.65 * mean_direct * center_pressure * excess,
                )
                added += 1
                if added == 2:
                    break

    # Reciprocal links can consume branch capacity without expanding reach.
    # Reduce this penalty where mutual direct demand makes reciprocity useful.
    reciprocity_scale = 0.045 * mean_direct
    for i in range(n):
        for j in range(i + 1, n):
            mutual = float(np.sqrt(d[i, j] * d[j, i]))
            add_pair(
                (i, j),
                (j, i),
                reciprocity_scale * (1.0 - 0.70 * mutual),
            )

    penalty_weight = 1.25 + 0.004 * max(0, n - 10)
    return linear, quadratic, float(min(100.0, max(0.05, penalty_weight)))


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
