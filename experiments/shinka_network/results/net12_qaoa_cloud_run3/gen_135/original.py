"""Surrogate QUBO terms for QOBLIB 08 network design. ShinkaEvolve edits the EVOLVE block."""

import time

# EVOLVE-BLOCK-START
import numpy as np


def surrogate_terms(demand):
    """Build direct-link rewards with sparse relay and congestion terms."""
    traffic = np.asarray(demand, dtype=float).copy()
    n = len(traffic)

    traffic = np.maximum(traffic, 0.0)
    np.fill_diagonal(traffic, 0.0)

    scale = float(np.max(traffic)) if traffic.size else 0.0
    if scale <= 0.0:
        scale = 1.0
    t = traffic / scale

    # Direct service remains the main signal, with a concave component to
    # retain some value for moderate demands.
    linear = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                value = float(t[i, j])
                linear[(i, j)] = -(0.72 * value + 0.28 * np.sqrt(value))

    if n < 3 or not np.any(t):
        return linear, {}, 1.5

    incoming = np.sum(t, axis=0)
    outgoing = np.sum(t, axis=1)
    incident = np.maximum(incoming, outgoing)
    pressure_scale = float(np.max(incident))
    if pressure_scale <= 0.0:
        pressure_scale = 1.0
    pressure = np.maximum(0.45, incident / pressure_scale)

    quadratic = {}
    estimated_load = t.copy()

    positive = t[t > 0.0]
    important_cutoff = float(np.quantile(positive, 0.80)) if positive.size else 0.0

    # Give each OD pair a bounded amount of relay reward, concentrated on a
    # small number of relays with lower pressure and less burdened legs.
    path_reward_total = 0.40
    transit_fraction = 0.24

    for i in range(n):
        for k in range(n):
            value = float(t[i, k])
            if i == k or value <= 0.0:
                continue

            support = 0.75 * value + 0.25 * np.sqrt(value)
            candidates = []

            for j in range(n):
                if j == i or j == k:
                    continue

                leg_a = float(t[i, j])
                leg_b = float(t[j, k])

                # Favor relays with available capacity, and mildly discourage
                # routes whose legs already carry substantial direct demand.
                rank = 1.0 / (
                    (0.40 + float(pressure[j]))
                    * (1.0 + 0.35 * (leg_a + leg_b))
                )

                # A small affinity preference breaks ties toward routes whose
                # two links have some supporting traffic, without excluding
                # otherwise useful paths with a zero-demand leg.
                affinity = np.sqrt((min(leg_a, leg_b) + 0.02) *
                                   (max(leg_a, leg_b) + 0.02))
                rank *= 0.85 + 0.15 * affinity
                candidates.append((rank, j))

            candidates.sort(key=lambda item: (-item[0], item[1]))
            keep = 3 if value >= important_cutoff else 2
            selected = candidates[:keep]
            total_rank = sum(item[0] for item in selected)
            if total_rank <= 0.0:
                continue

            for rank, j in selected:
                share = rank / total_rank
                first, second = (i, j), (j, k)
                key = (first, second) if first < second else (second, first)

                quadratic[key] = quadratic.get(key, 0.0) - (
                    path_reward_total * support * share
                )

                transit = transit_fraction * value * share
                estimated_load[i, j] += transit
                estimated_load[j, k] += transit

    peak_load = float(np.max(estimated_load))
    if peak_load > 0.0:
        estimated_load /= peak_load

    # Approximate a sum-of-squared-loads penalty, but restrict pairings to
    # the busiest incident links to avoid creating an unnecessarily dense
    # interaction graph.
    congestion_strength = 0.12
    min_burden = 0.025
    keep_per_incidence = min(5, max(2, n // 3))

    def add_congestion_pair(edge_a, edge_b):
        a = float(estimated_load[edge_a[0], edge_a[1]])
        b = float(estimated_load[edge_b[0], edge_b[1]])
        if a < min_burden or b < min_burden:
            return
        key = (edge_a, edge_b) if edge_a < edge_b else (edge_b, edge_a)
        quadratic[key] = quadratic.get(key, 0.0) + congestion_strength * a * b

    for node in range(n):
        outgoing_links = [(node, j) for j in range(n) if j != node]
        outgoing_links.sort(
            key=lambda edge: (-float(estimated_load[edge[0], edge[1]]), edge[1])
        )
        selected = outgoing_links[:keep_per_incidence]
        for a in range(len(selected)):
            for b in range(a + 1, len(selected)):
                add_congestion_pair(selected[a], selected[b])

        incoming_links = [(i, node) for i in range(n) if i != node]
        incoming_links.sort(
            key=lambda edge: (-float(estimated_load[edge[0], edge[1]]), edge[0])
        )
        selected = incoming_links[:keep_per_incidence]
        for a in range(len(selected)):
            for b in range(a + 1, len(selected)):
                add_congestion_pair(selected[a], selected[b])

    # Keep coefficient magnitudes stable. The evaluator adds degree penalties
    # separately.
    max_weight = 0.0
    for value in linear.values():
        max_weight = max(max_weight, abs(float(value)))
    for value in quadratic.values():
        max_weight = max(max_weight, abs(float(value)))

    if max_weight > 0.0:
        linear = {key: float(value) / max_weight for key, value in linear.items()}
        quadratic = {
            key: float(value) / max_weight
            for key, value in quadratic.items()
            if value != 0.0
        }
    else:
        linear = {key: 0.0 for key in linear}
        quadratic = {}

    return linear, quadratic, 2.0


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
