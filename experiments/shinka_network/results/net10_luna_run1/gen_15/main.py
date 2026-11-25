"""Surrogate QUBO terms for QOBLIB 08 network design. ShinkaEvolve edits the EVOLVE block."""

import time

# EVOLVE-BLOCK-START
import numpy as np


def surrogate_terms(demand):
    """Build linear and quadratic QUBO cost terms from a demand matrix."""
    t = np.asarray(demand, dtype=float)
    n = len(t)
    if n == 0:
        return {}, {}, 1.0

    # Ignore negative or non-finite inputs and normalize for scale invariance.
    t = np.where(np.isfinite(t), np.maximum(t, 0.0), 0.0)
    np.fill_diagonal(t, 0.0)
    scale = float(np.max(t))
    d = t / scale if scale > 0.0 else np.zeros_like(t)

    row = d.sum(axis=1)
    linear = {
        (i, j): -float(d[i, j])
        for i in range(n)
        for j in range(n)
        if i != j
    }

    quadratic = {}
    path_weight = 0.08
    relay_floor = 0.30
    relay_strength = 0.70
    eps = 1e-12

    # Reward demand-supported paths i -> j -> k. Conditional demand shares
    # give greater weight to relays whose links are plausible for both legs.
    for i in range(n):
        if row[i] <= eps:
            continue
        for k in range(n):
            if i == k or d[i, k] <= 0.0:
                continue
            for j in range(n):
                if j == i or j == k or row[j] <= eps:
                    continue

                inbound_share = float(d[i, j] / row[i])
                outbound_share = float(d[j, k] / row[j])
                relay_fit = np.sqrt(max(0.0, inbound_share * outbound_share))
                factor = relay_floor + relay_strength * min(1.0, relay_fit)

                quadratic[((i, j), (j, k))] = (
                    -path_weight * float(d[i, k]) * factor
                )

    # Degree penalties are added by the evaluator.
    penalty_weight = 1.0
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
