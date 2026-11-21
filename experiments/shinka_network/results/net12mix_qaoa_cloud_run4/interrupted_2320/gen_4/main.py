"""Surrogate QUBO terms for QOBLIB 08 network design. ShinkaEvolve edits the EVOLVE block."""

import time

# EVOLVE-BLOCK-START
import numpy as np


def surrogate_terms(demand):
    """Build direct-link rewards and congestion-aware two-hop path couplings."""
    d = np.maximum(np.asarray(demand, dtype=float), 0.0)
    n = len(d)
    scale = float(d.max()) or 1.0
    q = d / scale

    # Preserve the direct-traffic signal from the baseline.
    linear = {
        (i, j): -float(q[i, j])
        for i in range(n)
        for j in range(n)
        if i != j and q[i, j] != 0.0
    }

    # A two-hop route is useful, but using already busy links to carry it
    # increases congestion. Penalize that interaction through the links'
    # direct-demand estimates rather than rewarding every chain equally.
    quadratic = {}
    for a in range(n):
        for b in range(n):
            if b == a:
                continue
            for c in range(n):
                if c == a or c == b:
                    continue
                path_demand = float(q[a, c])
                if path_demand == 0.0:
                    continue
                edge_load = 0.5 * (float(q[a, b]) + float(q[b, c]))
                weight = path_demand * (-0.30 + 0.55 * edge_load)
                if abs(weight) > 1e-12:
                    e1, e2 = (a, b), (b, c)
                    key = (e1, e2) if e1 < e2 else (e2, e1)
                    quadratic[key] = weight

    penalty_weight = 3.0
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
