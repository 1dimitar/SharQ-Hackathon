"""Surrogate QUBO terms for QOBLIB 08 network design. ShinkaEvolve edits the EVOLVE block."""

import time

# EVOLVE-BLOCK-START
import numpy as np


def surrogate_terms(demand):
    """Build linear and quadratic surrogate cost terms for a directed topology."""
    t = np.asarray(demand, dtype=float)
    n = len(t)
    if n < 2:
        return {}, {}, 1.5

    t = np.maximum(t, 0.0).copy()
    np.fill_diagonal(t, 0.0)
    scale = float(t.max())
    if scale <= 0.0:
        return ({(i, j): 0.0 for i in range(n) for j in range(n) if i != j},
                {}, 1.5)

    d = t / scale
    links = [(i, j) for i in range(n) for j in range(n) if i != j]
    linear = {(i, j): -float(d[i, j]) for i, j in links}
    quadratic = {}

    # A two-hop path is useful when its first and second arcs are plausible
    # for the corresponding origin and destination. Normalize per OD pair so
    # that relay rewards do not simply duplicate large raw demands.
    relay_for_link = np.zeros((n, n), dtype=float)
    for i in range(n):
        for k in range(n):
            if i == k or d[i, k] <= 0:
                continue
            candidates = []
            for j in range(n):
                if j != i and j != k:
                    # A mild preference for relays with useful outgoing traffic.
                    plausibility = 0.20 + 0.80 * np.sqrt(
                        max(d[j, k], 0.0) / (float(d[j].max()) + 1e-12)
                    )
                    candidates.append((j, plausibility))
            total = sum(v for _, v in candidates)
            if total <= 0:
                continue
            for j, plausibility in candidates:
                share = float(d[i, k]) * plausibility / total
                relay_for_link[i, j] += share
                relay_for_link[j, k] += share

                # Reward completing a two-hop route. The small coefficient
                # keeps this interaction subordinate to direct-demand terms.
                a, b = (i, j), (j, k)
                key = (a, b) if a < b else (b, a)
                quadratic[key] = quadratic.get(key, 0.0) - 0.16 * share

    # Small linear relay adjustment: a link can be valuable even when its own
    # endpoint demand is modest, if it participates in many useful paths.
    relay_scale = float(relay_for_link.max())
    if relay_scale > 0:
        for i, j in links:
            linear[i, j] -= 0.12 * float(relay_for_link[i, j] / relay_scale)

    # Keep the interaction graph sparse: only two-hop path pairs are emitted.
    # Drop negligible numerical interactions to avoid needless couplings.
    if quadratic:
        cutoff = 1e-8
        quadratic = {key: value for key, value in quadratic.items()
                     if abs(value) > cutoff}

    # Degree constraints are supplied externally; a stronger weight helps
    # counter the many infeasible low-energy assignments from an unconstrained
    # collection of link rewards.
    penalty_weight = 2.0
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
