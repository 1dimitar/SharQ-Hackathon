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

    # Similar endpoint demand profiles imply redundant routing options.
    # Normalize rows/columns so the interaction measures profile similarity,
    # not simply the overall traffic volume at a node.
    row_norm = np.linalg.norm(demand, axis=1)
    col_norm = np.linalg.norm(demand, axis=0)
    rows = np.divide(
        demand, row_norm[:, None],
        out=np.zeros_like(demand, dtype=float),
        where=row_norm[:, None] > 0,
    )
    cols = np.divide(
        demand, col_norm[None, :],
        out=np.zeros_like(demand, dtype=float),
        where=col_norm[None, :] > 0,
    )

    quadratic = {}
    diversity_weight = 0.15

    for i in range(n):
        neighbors = [j for j in range(n) if j != i]

        # Outgoing links from i should reach nodes with distinct downstream
        # demand profiles.
        for a_pos, j in enumerate(neighbors):
            for k in neighbors[a_pos + 1:]:
                similarity = float(np.dot(rows[j], rows[k]))
                if similarity > 1e-12:
                    quadratic[((i, j), (i, k))] = diversity_weight * similarity

        # Incoming links to i should come from nodes with distinct demand
        # origins, avoiding redundant access to the same traffic sources.
        for a_pos, j in enumerate(neighbors):
            for k in neighbors[a_pos + 1:]:
                similarity = float(np.dot(cols[j], cols[k]))
                if similarity > 1e-12:
                    quadratic[((j, i), (k, i))] = diversity_weight * similarity

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
