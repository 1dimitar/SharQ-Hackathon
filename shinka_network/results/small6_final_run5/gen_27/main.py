"""Surrogate QUBO terms for QOBLIB 08 network design. ShinkaEvolve edits the EVOLVE block."""

import time

# EVOLVE-BLOCK-START
import numpy as np


def surrogate_terms(demand):
    """Build direct-service, pressure-aware route, and capacity terms."""
    traffic = np.asarray(demand, dtype=float)
    n = len(traffic)

    traffic = np.nan_to_num(
        traffic, nan=0.0, posinf=0.0, neginf=0.0
    )
    traffic = np.maximum(traffic, 0.0).copy()
    np.fill_diagonal(traffic, 0.0)

    maximum = float(traffic.max()) if traffic.size else 0.0
    if maximum <= 0.0:
        linear = {
            (i, j): 0.0
            for i in range(n)
            for j in range(n)
            if i != j
        }
        return linear, {}, 1.25

    t = traffic / maximum
    root = np.sqrt(t)
    row = traffic.sum(axis=1)
    col = traffic.sum(axis=0)
    row_max = max(float(row.max()), 1e-12)
    col_max = max(float(col.max()), 1e-12)

    incident = np.maximum(row, col)
    incident_max = max(float(incident.max()), 1e-12)
    base_pressure = np.maximum(0.55, incident / incident_max)

    # Direct links receive most of their reward from the traffic they serve.
    # The small endpoint adjustment gives extra weight to high-demand sources
    # and destinations without making that adjustment dominate link value.
    linear = {}
    for i in range(n):
        source_need = float(row[i] / row_max)
        for j in range(n):
            if i == j:
                continue
            destination_need = float(col[j] / col_max)
            endpoint_factor = 0.94 + 0.06 * (
                0.5 * source_need + 0.5 * destination_need
            )
            direct_value = 0.74 * float(t[i, j]) + 0.26 * float(root[i, j])
            linear[(i, j)] = -float(direct_value * endpoint_factor)

    # Store plausible two-hop routes. Requiring positive evidence on both
    # legs avoids rewarding arbitrary paths whose links have no demand signal.
    routes_by_od = {}
    transit_load = np.zeros(n, dtype=float)

    for i in range(n):
        for k in range(n):
            if i == k or t[i, k] <= 0.0:
                continue

            candidates = []
            for j in range(n):
                if j == i or j == k:
                    continue
                a = float(t[i, j])
                b = float(t[j, k])
                geometric = float(np.sqrt(a * b))
                if geometric <= 1e-8:
                    continue

                balance = 2.0 * geometric / (a + b + 0.05)
                rank = (
                    geometric
                    * (0.65 + 0.35 * balance)
                    / (float(base_pressure[j]) ** 0.60)
                )
                candidates.append((rank, j, balance))

            if not candidates:
                continue

            candidates.sort(key=lambda item: (-item[0], item[1]))
            routes_by_od[(i, k)] = candidates[:3]

            # A preliminary, soft split of traffic over the best routes
            # estimates which intermediate nodes are likely transit hubs.
            chosen = candidates[:min(3, len(candidates))]
            strengths = [max(item[0], 1e-12) ** 1.5 for item in chosen]
            strength_sum = sum(strengths)
            indirect_fraction = 0.30 + 0.55 * (1.0 - float(t[i, k]))
            for (_, j, _), strength in zip(chosen, strengths):
                transit_load[j] += (
                    float(traffic[i, k])
                    * indirect_fraction
                    * strength
                    / strength_sum
                )

    # Blend estimated transit with endpoint demand to avoid overusing a node
    # that appears attractive for many otherwise plausible routes.
    projected = incident + 0.20 * transit_load
    pressure_scale = max(float(projected.max()), 1e-12)
    pressure = np.maximum(
        0.55, np.minimum(1.8, projected / pressure_scale)
    )

    quadratic = {}

    def add_q(first, second, weight):
        if first == second or abs(weight) < 1e-13:
            return
        key = (first, second) if first < second else (second, first)
        quadratic[key] = quadratic.get(key, 0.0) + float(weight)

    # Pairwise capacity proxies: discourage concentrating links around nodes
    # with high traffic, while keeping these terms modest relative to direct
    # service rewards.
    row_scale = np.maximum(row / row_max, 0.10)
    col_scale = np.maximum(col / col_max, 0.10)

    for v in range(n):
        incoming = [(u, v) for u in range(n) if u != v]
        outgoing = [(v, w) for w in range(n) if w != v]
        node_factor = float(pressure[v]) ** 0.85

        for a in range(len(incoming)):
            source_a = incoming[a][0]
            for b in range(a + 1, len(incoming)):
                source_b = incoming[b][0]
                weight = (
                    0.035
                    * node_factor
                    * float(np.sqrt(row_scale[source_a] * row_scale[source_b]))
                )
                add_q(incoming[a], incoming[b], weight)

        for a in range(len(outgoing)):
            destination_a = outgoing[a][1]
            for b in range(a + 1, len(outgoing)):
                destination_b = outgoing[b][1]
                weight = (
                    0.035
                    * node_factor
                    * float(np.sqrt(col_scale[destination_a] * col_scale[destination_b]))
                )
                add_q(outgoing[a], outgoing[b], weight)

    # Approximate the value of a two-hop path only when its direct alternative
    # is absent: -r * (1 - x_ik) * x_ij * x_jk. Expanding the cubic about the
    # valid-topology link density and dropping third-order terms provides
    # pairwise terms usable by the QUBO while retaining direct/path competition.
    occupancy = 2.0 / max(1, n - 1)

    for (i, k), candidates in routes_by_od.items():
        ranked = []
        for base_rank, j, balance in candidates:
            adjusted_rank = base_rank / (float(pressure[j]) ** 0.35)
            ranked.append((adjusted_rank, j, balance))

        ranked.sort(key=lambda item: (-item[0], item[1]))
        if not ranked:
            continue

        best_rank = max(float(ranked[0][0]), 1e-12)
        keep = min(2, len(ranked))
        # Include a third route only when it is genuinely competitive.
        if (
            len(ranked) > 2
            and ranked[2][0] >= 0.74 * best_rank
            and ranked[2][0] >= 0.80 * ranked[1][0]
        ):
            keep = 3

        direct = float(t[i, k])
        support = 0.68 * direct + 0.32 * float(root[i, k])

        for rank, j, balance in ranked[:keep]:
            relative = float(rank) / best_rank
            route_quality = (0.76 + 0.24 * relative) * (0.82 + 0.36 * balance)
            reward = (
                0.27
                * support
                * route_quality
                / (float(pressure[j]) ** 0.25)
            )

            first = (i, j)
            second = (j, k)
            direct_link = (i, k)

            # The base quadratic route reward and the pair terms from the
            # second-order expansion of the direct/path interaction.
            add_q(first, second, -reward * (1.0 - occupancy))
            add_q(direct_link, first, reward * occupancy)
            add_q(direct_link, second, reward * occupancy)

            correction = reward * occupancy * occupancy
            linear[first] = linear.get(first, 0.0) - correction
            linear[second] = linear.get(second, 0.0) - correction
            linear[direct_link] = linear.get(direct_link, 0.0) - correction

    # Remove numerical cancellations so the returned graph stays compact.
    quadratic = {
        key: value
        for key, value in quadratic.items()
        if abs(value) >= 1e-12
    }

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
