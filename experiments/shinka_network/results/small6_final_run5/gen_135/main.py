"""Surrogate QUBO terms for QOBLIB 08 network design. ShinkaEvolve edits the EVOLVE block."""

import time

# EVOLVE-BLOCK-START
import numpy as np


def surrogate_terms(demand):
    """Build direct-link rewards, service-portfolio couplings, and route terms."""
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
    z = traffic / maximum if maximum > 0.0 else traffic.copy()
    root = np.sqrt(z)

    row_total = z.sum(axis=1)
    col_total = z.sum(axis=0)
    row_share = z / np.maximum(row_total[:, None], 1e-12)
    col_share = z / np.maximum(col_total[None, :], 1e-12)

    # Direct service is useful, but sublinear terms retain information about
    # smaller demands and about the importance of an edge to its endpoints.
    direct = (
        0.58 * np.power(z, 0.78)
        + 0.27 * root
        + 0.15 * np.sqrt(np.maximum(row_share * col_share, 0.0))
    )
    linear_array = -direct.copy()
    quadratic = {}

    def add_pair(first, second, value):
        value = float(value)
        if abs(value) < 1e-14 or first == second:
            return
        key = (first, second) if first < second else (second, first)
        quadratic[key] = quadratic.get(key, 0.0) + value

    mean_direct = float(direct.sum()) / max(1, n * (n - 1))
    depth_scale = max(
        0.8, float(np.log2(max(n, 2)) / np.log2(10.0))
    )

    def make_service_profile(matrix):
        """Return a damped Markov service profile for candidate neighbors."""
        total_out = matrix.sum(axis=1)
        total_in = matrix.sum(axis=0)
        rs = matrix / np.maximum(total_out[:, None], 1e-12)
        cs = matrix / np.maximum(total_in[None, :], 1e-12)
        mz = float(matrix.max()) if matrix.size else 0.0
        scaled = matrix / mz if mz > 0.0 else matrix.copy()

        affinity = (
            0.46 * np.sqrt(np.maximum(scaled, 0.0))
            + 0.30 * np.sqrt(np.maximum(rs, 0.0))
            + 0.24 * np.sqrt(np.maximum(cs, 0.0))
        )
        np.fill_diagonal(affinity, 0.0)
        transition = affinity / np.maximum(
            affinity.sum(axis=1, keepdims=True), 1e-15
        )
        # Candidate neighbor itself, followed by likely one- and two-step
        # continuation from that neighbor.
        return (
            0.62 * np.eye(n)
            + 0.92 * transition
            + 0.38 * (transition @ transition)
        )

    def add_portfolio_terms(matrix, reverse=False):
        """Score pairs by complementary service to local traffic."""
        if mean_direct <= 1e-15:
            return

        service = make_service_profile(matrix)
        totals = matrix.sum(axis=1)
        mean_total = float(totals.mean())
        if mean_total <= 1e-15:
            return

        # A bounded number of strong pair preferences per node keeps the
        # interaction graph structured while retaining good and poor pairs.
        pair_budget = max(8, n)

        for node in range(n):
            total = float(totals[node])
            if total <= 1e-15:
                continue

            neighbors = [v for v in range(n) if v != node]
            m = len(neighbors)
            weights = matrix[node] / total
            records = []

            for a_pos in range(m):
                a = neighbors[a_pos]
                sa = service[a]
                for b_pos in range(a_pos + 1, m):
                    b = neighbors[b_pos]
                    sb = service[b]

                    combined = np.maximum(sa + sb, 1e-10)
                    coverage = float(np.dot(weights, np.log(combined)))
                    overlap = float(np.dot(
                        weights, np.minimum(sa, sb) / combined
                    ))
                    imbalance = float(np.dot(
                        weights, np.abs(sa - sb) / combined
                    ))

                    # Higher utility means better coverage with less overlap
                    # and less dependence on just one of the two neighbors.
                    utility = coverage - 0.28 * overlap - 0.12 * imbalance
                    records.append((a, b, utility))

            if not records:
                continue

            values = np.asarray([r[2] for r in records], dtype=float)
            center = float(values.mean())
            spread = float(np.sqrt(np.mean((values - center) ** 2)))
            if spread <= 1e-12:
                continue

            importance = float(np.clip(total / mean_total, 0.60, 1.65))
            strength = (
                0.19 * mean_direct * depth_scale * importance
            )

            ranked = sorted(
                records,
                key=lambda item: (
                    -abs((item[2] - center) / spread),
                    item[0],
                    item[1],
                ),
            )
            kept = 0
            for a, b, utility in ranked:
                quality = (utility - center) / spread
                if abs(quality) < 0.45:
                    break
                first = (a, node) if reverse else (node, a)
                second = (b, node) if reverse else (node, b)
                add_pair(first, second, -strength * float(np.clip(
                    quality, -2.5, 2.5
                )))
                kept += 1
                if kept >= pair_budget:
                    break

    # Outgoing portfolios serve each node's own destinations; the transposed
    # pass models incoming links as service choices for traffic ending there.
    add_portfolio_terms(z, reverse=False)
    add_portfolio_terms(z.T, reverse=True)

    # Soft bottleneck route support. Rather than rewarding every two-hop
    # path, consider only the best plausible intermediates for each demand.
    incident = np.maximum(row_total, col_total)
    peak_incident = float(incident.max()) if incident.size else 0.0
    pressure = (
        incident / peak_incident
        if peak_incident > 1e-15
        else np.zeros(n, dtype=float)
    )

    edge_quality = (
        0.45 * root
        + 0.30 * np.sqrt(np.maximum(row_share, 0.0))
        + 0.25 * np.sqrt(np.maximum(col_share, 0.0))
    )
    support_matrix = (
        0.66 * np.power(z, 0.72) + 0.34 * root
    ) * (1.0 - 0.24 * root)

    for source in range(n):
        for target in range(n):
            amount = float(z[source, target])
            if source == target or amount <= 0.0:
                continue

            support = float(support_matrix[source, target])
            candidates = []
            for middle in range(n):
                if middle == source or middle == target:
                    continue

                first_quality = float(edge_quality[source, middle])
                second_quality = float(edge_quality[middle, target])
                # A weak leg limits the usefulness of the whole route.
                harmonic = (
                    2.0 * first_quality * second_quality
                    / max(first_quality + second_quality, 1e-12)
                )
                rank = harmonic / (0.72 + 0.55 * float(pressure[middle]))
                candidates.append((float(rank), middle, harmonic))

            candidates.sort(key=lambda item: (-item[0], item[1]))
            if not candidates:
                continue

            keep = 1
            if len(candidates) > 1 and candidates[1][0] >= 0.68 * candidates[0][0]:
                keep = 2
            chosen = candidates[:keep]

            # A small soft allocation across alternatives avoids a brittle
            # winner-take-all route signal when several paths look alike.
            top_rank = chosen[0][0]
            route_weights = np.asarray([
                np.exp((item[0] - top_rank) / 0.20) for item in chosen
            ], dtype=float)
            route_weights /= max(float(route_weights.sum()), 1e-15)

            for (rank, middle, harmonic), share in zip(chosen, route_weights):
                route_quality = float(np.clip(harmonic, 0.0, 1.0))
                route_reward = (
                    0.19 * support * float(share)
                    * (0.55 + 0.45 * route_quality)
                )
                first = (source, middle)
                second = (middle, target)
                add_pair(first, second, -route_reward)

                # Reduce modestly the double-counting of a commodity when
                # its direct link and an indirect route are both selected.
                redundancy = 0.025 * support * float(share)
                add_pair((source, target), first, redundancy)
                add_pair((source, target), second, redundancy)

    # A mild reciprocal bonus is demand-conditioned: reverse edges are
    # useful when they carry traffic in both directions, not merely because
    # they form a two-cycle.
    reciprocal_scale = 0.10 * mean_direct * depth_scale
    if reciprocal_scale > 0.0:
        for i in range(n):
            for j in range(i + 1, n):
                mutual = float(np.sqrt(z[i, j] * z[j, i]))
                if mutual > 0.0:
                    add_pair(
                        (i, j), (j, i),
                        -reciprocal_scale * mutual,
                    )

    linear = {
        (i, j): float(linear_array[i, j])
        for i in range(n)
        for j in range(n)
        if i != j
    }
    quadratic = {
        key: float(value)
        for key, value in quadratic.items()
        if abs(value) > 1e-12
    }

    penalty_weight = float(min(1.45, 1.25 + 0.008 * max(0, n - 10)))
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
