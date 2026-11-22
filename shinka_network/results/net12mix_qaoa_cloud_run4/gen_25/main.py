"""Surrogate QUBO terms for QOBLIB 08 network design. ShinkaEvolve edits the EVOLVE block."""

import time

# EVOLVE-BLOCK-START
import numpy as np


def surrogate_terms(demand):
    """Build sparse demand-sensitive costs and endpoint load-balancing couplings."""
    t = np.asarray(demand, dtype=float)
    n = t.shape[0]
    if n == 0:
        return {}, {}, 1.0

    # Remove invalid/negative traffic and ignore diagonal entries.
    t = np.maximum(t, 0.0).copy()
    np.fill_diagonal(t, 0.0)
    maximum = float(t.max())
    if not np.isfinite(maximum) or maximum <= 0.0:
        maximum = 1.0

    d = t / maximum
    row = t.sum(axis=1)
    col = t.sum(axis=0)
    total = float(t.sum())

    if total > 0.0:
        node = (row + col) / total
    else:
        node = np.zeros(n, dtype=float)

    # Direct demand remains the principal link-selection signal; endpoint
    # importance provides a small correction for transit-relevant nodes.
    linear = {}
    exposure = np.zeros(n, dtype=float)
    if total > 0.0:
        exposure = (row + col) / total
        emax = float(exposure.max())
        if emax > 0.0:
            exposure = np.sqrt(exposure / emax)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            endpoint = 0.5 * (float(node[i]) + float(node[j]))
            linear[(i, j)] = -(float(d[i, j]) + 0.10 * endpoint)

    quadratic = {}

    # Sparse two-hop rewards: credit only the best relay for each OD pair.
    # Favor relays with useful endpoint exposure, while retaining a small
    # preference for less-loaded transit nodes.
    for i in range(n):
        for k in range(n):
            amount = float(d[i, k])
            if i == k or amount <= 0.0 or n < 3:
                continue

            best_score = -1.0
            best_relay = -1
            for j in range(n):
                if j == i or j == k:
                    continue
                usefulness = 0.55 + 0.45 * float(exposure[j])
                load_discount = 1.0 / (1.0 + 1.5 * float(node[j]))
                score = usefulness * load_discount
                if score > best_score:
                    best_score = score
                    best_relay = j

            if best_relay >= 0:
                a = (i, best_relay)
                b = (best_relay, k)
                key = (a, b) if a <= b else (b, a)
                quadratic[key] = quadratic.get(key, 0.0) - 0.14 * amount

    # Positive pair costs discourage selecting links that expose two busy
    # destinations from one source, or two busy sources into one receiver.
    # The square-root product gives a bounded, smooth estimate of shared load.
    pair_scale = 0.085
    cutoff = 0.025

    for i in range(n):
        links = [(i, j) for j in range(n) if j != i]
        for a in range(len(links)):
            j = links[a][1]
            ej = float(exposure[j])
            for b in range(a + 1, len(links)):
                k = links[b][1]
                value = (ej * float(exposure[k])) ** 0.5
                if value >= cutoff:
                    quadratic[(links[a], links[b])] = (
                        quadratic.get((links[a], links[b]), 0.0) + pair_scale * value
                    )

    for j in range(n):
        links = [(i, j) for i in range(n) if i != j]
        for a in range(len(links)):
            i = links[a][0]
            ei = float(exposure[i])
            for b in range(a + 1, len(links)):
                k = links[b][0]
                value = (ei * float(exposure[k])) ** 0.5
                if value >= cutoff:
                    quadratic[(links[a], links[b])] = (
                        quadratic.get((links[a], links[b]), 0.0) + pair_scale * value
                    )

    # A 24-node instance stays well below the allowed O(n^3) term budget.
    penalty_weight = 3.5
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
