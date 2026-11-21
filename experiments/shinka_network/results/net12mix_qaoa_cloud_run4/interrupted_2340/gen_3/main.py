"""Surrogate QUBO terms for QOBLIB 08 network design. ShinkaEvolve edits the EVOLVE block."""

import time

# EVOLVE-BLOCK-START
import numpy as np


def surrogate_terms(demand):
    """Build direct-link rewards and modest congestion-discounted path rewards."""
    n = len(demand)
    scale = float(demand.max()) or 1.0
    linear = {
        (i, j): -float(demand[i, j]) / scale
        for i in range(n) for j in range(n) if i != j
    }

    # A node with substantial own traffic is a less attractive transit point:
    # its two incoming and two outgoing links must already serve that traffic.
    node_load = np.asarray(demand, dtype=float).sum(axis=0) + np.asarray(
        demand, dtype=float
    ).sum(axis=1)
    mean_load = float(node_load.mean()) or 1.0

    # Reward a selected i->j->k path in proportion to the traffic it can serve.
    # Distribute the reward over possible intermediates and discount congested hubs.
    quadratic = {}
    path_scale = 0.18 / np.sqrt(max(1, n - 2))
    for i in range(n):
        for j in range(n):
            if j == i:
                continue
            congestion_discount = 1.0 / np.sqrt(
                1.0 + 0.5 * float(node_load[j]) / mean_load
            )
            for k in range(n):
                if k == i or k == j:
                    continue
                a, b = (i, j), (j, k)
                key = (a, b) if a < b else (b, a)
                quadratic[key] = (
                    -path_scale
                    * (float(demand[i, k]) / scale)
                    * congestion_discount
                )

    # The evaluator applies this after normalizing the cost coefficients.
    penalty_weight = 4.0
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
