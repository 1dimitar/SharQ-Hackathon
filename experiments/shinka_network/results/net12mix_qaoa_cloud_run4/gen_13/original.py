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
    linear = {(i, j): -float(demand[i, j]) / scale for i in range(n) for j in range(n) if i != j}

    # Favor relays with substantial, balanced traffic entering and leaving them.
    incoming = np.sum(demand, axis=0)
    outgoing = np.sum(demand, axis=1)
    relay = np.sqrt(np.maximum(incoming, 0.0) * np.maximum(outgoing, 0.0))
    relay_max = float(relay.max()) or 1.0
    relay = (relay / relay_max) ** 1.5

    # A selected two-link path can carry traffic that has no direct link.
    # Keep this reward modest so it complements rather than overwhelms direct demand.
    quadratic = {}
    for i in range(n):
        for j in range(n):
            if j == i:
                continue
            for k in range(n):
                if k == i or k == j:
                    continue
                weight = 0.15 * (float(demand[i, k]) / scale) * float(relay[j])
                if weight:
                    quadratic[((i, j), (j, k))] = -weight

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
