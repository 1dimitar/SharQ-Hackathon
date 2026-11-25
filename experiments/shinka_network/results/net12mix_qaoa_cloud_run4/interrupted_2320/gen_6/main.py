"""Surrogate QUBO terms for QOBLIB 08 network design. ShinkaEvolve edits the EVOLVE block."""

import time

# EVOLVE-BLOCK-START
import numpy as np


def surrogate_terms(demand):
    """Build the surrogate cost terms of the topology QUBO from the demand matrix.

    Args:
        demand: (n, n) float array; demand[i, j] is the traffic from node i to node j
            (zero diagonal).

    Returns:
        linear: dict {(i, j): weight}. Weight on the binary variable x_ij
            (x_ij = 1 if the directed link i -> j is built), i != j.
        quadratic: dict {((i, j), (k, l)): weight}. Weight on the product x_ij * x_kl.
        penalty_weight: float. Strength of the degree penalty that the evaluator adds
            after rescaling these terms so the largest |weight| is 1.

    Lower energy must mean a better topology. Baseline: reward each link by the demand
    it carries directly.
    """
    demand = np.asarray(demand, dtype=float)
    n = len(demand)
    scale = float(demand.max()) or 1.0

    # Keep direct service valuable, while allowing a pair of links to earn a
    # smaller joint reward when it provides a two-hop route for a commodity.
    linear = {
        (i, j): -float(demand[i, j]) / scale
        for i in range(n) for j in range(n) if i != j
    }

    row_load = demand.sum(axis=1)
    col_load = demand.sum(axis=0)
    node_load = row_load + col_load
    mean_load = float(node_load.mean()) or 1.0

    quadratic = {}
    for i in range(n):
        for j in range(n):
            if j == i:
                continue
            first = (i, j)
            for k in range(n):
                if k == i or k == j:
                    continue
                traffic = float(demand[i, k])
                if traffic <= 0.0:
                    continue

                # Downweight relays whose own traffic suggests they may be
                # heavily used; the square root keeps this attenuation mild.
                relay_factor = (mean_load / (mean_load + float(node_load[j]))) ** 0.5
                weight = -0.28 * (traffic / scale) * relay_factor
                second = (j, k)
                key = (first, second) if first < second else (second, first)
                quadratic[key] = quadratic.get(key, 0.0) + weight

    penalty_weight = 2.5
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
