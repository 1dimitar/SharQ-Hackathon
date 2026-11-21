"""Surrogate QUBO terms for QOBLIB 08 network design. ShinkaEvolve edits the EVOLVE block."""

import time

# EVOLVE-BLOCK-START
import numpy as np


def surrogate_terms(demand):
    """Return a demand-diffusion approximation as linear and quadratic terms."""
    raw = np.asarray(demand, dtype=float)
    n = len(raw)

    traffic = np.maximum(raw, 0.0).copy()
    np.fill_diagonal(traffic, 0.0)
    scale = float(np.max(traffic)) if traffic.size else 0.0
    if scale <= 0.0:
        scale = 1.0
    d = traffic / scale

    if n == 0:
        return {}, {}, 1.0

    # Node pressure measures the amount of traffic that a relay must
    # accommodate, but uses the average of its incoming and outgoing totals.
    incident = np.sum(d, axis=0) + np.sum(d, axis=1)
    mean_incident = float(np.mean(incident))
    if mean_incident <= 1e-12:
        pressure = np.ones(n, dtype=float)
    else:
        pressure = np.maximum(0.75, 0.5 * incident / mean_incident)

    # A weak diffusion of OD demand supplies a reachability-aware estimate:
    # traffic between i and k can be useful to i->j when j's outgoing demand
    # profile is compatible with k, and vice versa.
    row = np.sum(d, axis=1)
    col = np.sum(d, axis=0)
    row_scale = np.maximum(row, 1e-12)
    col_scale = np.maximum(col, 1e-12)

    linear = {}
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # Direct traffic remains valuable; the additional affinity term
            # favors links that lead toward destinations with substantial
            # demand from this source.
            affinity = 0.0
            if row[i] > 0.0:
                affinity = float(np.dot(d[i], d[j]) / row_scale[i])
            val = float(d[i, j])
            benefit = 0.78 * val + 0.22 * min(1.0, affinity)
            linear[(i, j)] = -benefit

    quadratic = {}

    # Wedge benefits: estimate how much OD demand can be served by a relay,
    # discounted by relay pressure and by the demand already incident to the
    # two selected links. Unlike a per-path max-load estimate, this is a
    # smooth diffusion-style proxy over all candidate relays.
    for i in range(n):
        for k in range(n):
            if i == k:
                continue
            od = float(d[i, k])
            if od <= 0.0:
                continue
            for j in range(n):
                if j == i or j == k:
                    continue

                # The profile terms make relays useful when they connect
                # sources/destinations with aligned demand patterns.
                source_fit = float(d[i, j]) + 0.20 * float(
                    np.dot(d[i], d[j]) / row_scale[i]
                )
                destination_fit = float(d[j, k]) + 0.20 * float(
                    np.dot(d[:, j], d[:, k]) / col_scale[k]
                )
                support = 0.55 + 0.45 * min(
                    1.0, 0.5 * (source_fit + destination_fit)
                )
                congestion = 1.0 + 0.35 * (
                    float(d[i, j]) + float(d[j, k])
                )
                weight = -0.40 * od * support / (
                    float(pressure[j]) * congestion
                )

                a, b = (i, j), (j, k)
                key = (a, b) if a < b else (b, a)
                quadratic[key] = quadratic.get(key, 0.0) + weight

    # Penalize pairs of incoming links to a receiver when its incident demand
    # is high. This discourages concentrating transit at overloaded nodes.
    # The coefficient is deliberately modest so it does not overwhelm useful
    # demand-serving wedges.
    for j in range(n):
        inbound = [(i, j) for i in range(n) if i != j]
        pressure_excess = max(0.0, float(pressure[j]) - 1.0)
        if pressure_excess <= 0.0:
            continue
        for a_idx in range(len(inbound)):
            a = inbound[a_idx]
            for b_idx in range(a_idx + 1, len(inbound)):
                b = inbound[b_idx]
                # Links carrying substantial demand already are less
                # attractive as a jointly selected pair.
                load = float(d[a[0], j] + d[b[0], j])
                key = (a, b) if a < b else (b, a)
                quadratic[key] = quadratic.get(key, 0.0) + (
                    0.025 * pressure_excess * (0.25 + 0.75 * load)
                )

    return linear, quadratic, 1.0


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
