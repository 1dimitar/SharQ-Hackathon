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
    scale = float(np.max(demand)) or 1.0
    node_load = np.sum(demand, axis=0) + np.sum(demand, axis=1)
    endpoint = np.add.outer(np.sum(demand, axis=1), np.sum(demand, axis=0))
    endpoint_scale = float(np.max(endpoint)) or 1.0
    linear = {
        (i, j): -(
            0.9 * (
                0.8 * float(demand[i, j]) / scale
                + 0.2 * np.sqrt(max(0.0, float(demand[i, j]) / scale))
            )
            + 0.1 * float(endpoint[i, j]) / endpoint_scale
        )
        for i in range(n) for j in range(n) if i != j
    }

    # A two-hop route can use any selected pair i -> j -> k.  With
    # indegree and outdegree two, at most four such routes exist per OD pair.
    # Share the indirect reward across those possibilities to avoid the
    # overcounting from assigning a large reward to every candidate path.
    max_load = float(np.max(node_load)) or 1.0
    quadratic = {}
    indirect_fraction = 0.6
    for i in range(n):
        for j in range(n):
            if j == i:
                continue
            # High-demand nodes are less desirable transit points.
            transit_factor = max(
                0.75, 1.0 - 0.25 * float(node_load[j]) / max_load
            )
            for k in range(n):
                if k == i or k == j or demand[i, k] == 0:
                    continue
                component_support = np.sqrt(
                    max(0.0, float(demand[i, j]) / scale)
                    * max(0.0, float(demand[j, k]) / scale)
                )
                plausibility = 0.25 + 0.75 * min(1.0, component_support)
                weight = (
                    -indirect_fraction
                    * float(demand[i, k]) / scale
                    * transit_factor
                    * plausibility / 4.0
                )
                quadratic[((i, j), (j, k))] = weight

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
