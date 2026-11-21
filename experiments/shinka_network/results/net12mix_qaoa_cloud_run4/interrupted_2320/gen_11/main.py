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
    linear = {
        (i, j): -float(demand[i, j]) / scale
        for i in range(n) for j in range(n) if i != j
    }

    # Encourage useful two-hop alternatives, but discount intermediates with
    # high traffic intensity so the surrogate does not over-favor transit hubs.
    node_load = demand.sum(axis=0) + demand.sum(axis=1)
    mean_load = float(node_load.mean()) or 1.0
    congestion = 1.0 + node_load / mean_load
    quadratic = {}
    path_strength = 0.18
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
                second = (j, k)
                # The two link variables jointly provide a route i -> j -> k.
                key = (first, second)
                quadratic[key] = (
                    -path_strength * (traffic / scale) / float(congestion[j])
                )

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
