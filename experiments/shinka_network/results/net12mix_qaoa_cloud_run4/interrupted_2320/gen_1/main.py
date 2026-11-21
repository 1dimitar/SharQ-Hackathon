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
    demand_scale = float(demand.max()) or 1.0
    row_sums = np.asarray(demand, dtype=float).sum(axis=1)
    col_sums = np.asarray(demand, dtype=float).sum(axis=0)
    row_scale = float(row_sums.max()) or 1.0
    col_scale = float(col_sums.max()) or 1.0

    # A selected link is a potential carrier of both origin and destination
    # traffic. Penalize pairings that place high-pressure endpoints together.
    linear = {}
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            direct = float(demand[i, j]) / demand_scale
            pressure = (float(row_sums[i]) / row_scale) * (float(col_sums[j]) / col_scale)
            linear[(i, j)] = -direct + 0.30 * pressure

    # A modest synergy favors direct links for substantial two-way demand.
    quadratic = {}
    for i in range(n):
        for j in range(i + 1, n):
            reciprocal = min(float(demand[i, j]), float(demand[j, i])) / demand_scale
            if reciprocal > 0.0:
                quadratic[((i, j), (j, i))] = -0.12 * reciprocal

    penalty_weight = 6.0
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
