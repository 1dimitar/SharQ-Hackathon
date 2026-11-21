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
    t = np.maximum(np.asarray(demand, dtype=float), 0.0).copy()
    n = t.shape[0]
    np.fill_diagonal(t, 0.0)

    positive = t[t > 0.0]
    if positive.size:
        scale = float(np.percentile(positive, 90.0))
        if scale <= 0.0:
            scale = float(positive.max())
        t = np.minimum(t, scale) / scale

    linear = {(i, j): -float(t[i, j]) for i in range(n) for j in range(n) if i != j}
    quadratic = {}
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
