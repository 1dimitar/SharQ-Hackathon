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
    d = np.maximum(np.asarray(demand, dtype=float), 0.0)
    scale = float(d.max()) or 1.0
    d = d / scale

    # Direct links are useful, but their marginal value should not grow
    # linearly without limit when one pair has exceptionally large demand.
    linear = {
        (i, j): -float(d[i, j]) + 0.12 * float(d[i, j] ** 2)
        for i in range(n) for j in range(n) if i != j
    }
    quadratic = {}

    def add_pair(a, b, weight):
        key = (a, b) if a < b else (b, a)
        quadratic[key] = quadratic.get(key, 0.0) + float(weight)

    # A composed path i -> j -> k can serve i-to-k traffic. Discount its
    # benefit when either link already has substantial direct demand, and
    # add a modest congestion cost for combining two heavily demanded links.
    for i in range(n):
        for j in range(n):
            if j == i:
                continue
            for k in range(n):
                if k == i or k == j:
                    continue
                left = float(d[i, j])
                right = float(d[j, k])
                path_demand = float(d[i, k])
                suitability = 1.0 / (1.0 + 0.5 * (left + right))
                benefit = 0.22 * path_demand * suitability
                congestion = 0.08 * (left * right) ** 0.5
                add_pair((i, j), (j, k), congestion - benefit)

    # Reciprocal links are worthwhile for mutually heavy traffic, but are
    # less attractive when both directions carry little demand.
    for i in range(n):
        for j in range(i + 1, n):
            mutual = (float(d[i, j]) * float(d[j, i])) ** 0.5
            add_pair((i, j), (j, i), 0.06 - 0.18 * mutual)

    # Strengthen the evaluator's degree penalty to encourage valid samples.
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
