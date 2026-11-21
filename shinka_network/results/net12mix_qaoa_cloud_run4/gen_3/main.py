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
    n = len(demand)
    scale = float(demand.max()) or 1.0

    # Keep direct traffic as the primary signal.
    linear = {
        (i, j): -float(demand[i, j]) / scale
        for i in range(n) for j in range(n) if i != j
    }

    # Reward two-hop paths for demanded OD pairs, with a discount for
    # intermediates that already have high aggregate incoming/outgoing demand.
    demand_array = np.asarray(demand, dtype=float)
    node_load = demand_array.sum(axis=0) + demand_array.sum(axis=1)
    mean_load = float(node_load.mean()) or 1.0
    quadratic = {}
    for i in range(n):
        for j in range(n):
            if j == i:
                continue
            for k in range(n):
                if k == j or k == i:
                    continue
                od_demand = float(demand_array[i, k])
                if od_demand <= 0.0:
                    continue
                hub_factor = 1.0 + float(node_load[j]) / mean_load
                quadratic[((i, j), (j, k))] = (
                    -0.25 * od_demand / (scale * hub_factor)
                )

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
