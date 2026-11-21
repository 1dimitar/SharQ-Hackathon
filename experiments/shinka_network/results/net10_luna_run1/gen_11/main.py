"""Surrogate QUBO terms for QOBLIB 08 network design. ShinkaEvolve edits the EVOLVE block."""

import time

# EVOLVE-BLOCK-START
import numpy as np


def surrogate_terms(demand):
    """Build direct-link rewards and demand-conditioned two-hop path rewards."""
    t = np.maximum(np.asarray(demand, dtype=float), 0.0).copy()
    n = t.shape[0]
    np.fill_diagonal(t, 0.0)

    scale = float(t.max()) if t.size else 0.0
    if scale > 0.0:
        t /= scale

    linear = {
        (i, j): -float(t[i, j])
        for i in range(n)
        for j in range(n)
        if i != j
    }

    # A path through j is more credible when both its incoming and outgoing
    # legs are supported by traffic at j. Normalize each leg by the total
    # traffic into/out of j, then rescale shares so uniform profiles are O(1).
    rows = t.sum(axis=1)
    cols = t.sum(axis=0)
    inbound = np.divide(
        t, cols[None, :], out=np.zeros_like(t), where=cols[None, :] > 0.0
    )
    outbound = np.divide(
        t, rows[:, None], out=np.zeros_like(t), where=rows[:, None] > 0.0
    )

    quadratic = {}
    for i in range(n):
        for j in range(n):
            if j == i:
                continue
            in_support = float(inbound[i, j])
            if in_support <= 0.0:
                continue
            for k in range(n):
                if k == i or k == j or t[i, k] <= 0.0:
                    continue
                out_support = float(outbound[j, k])
                if out_support <= 0.0:
                    continue
                profile = min(1.0, n * (in_support * out_support) ** 0.5)
                value = 0.12 * float(t[i, k]) * profile
                if value > 1e-14:
                    quadratic[((i, j), (j, k))] = -value

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
