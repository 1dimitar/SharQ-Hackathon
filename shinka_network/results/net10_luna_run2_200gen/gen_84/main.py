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
    linear = {
        (i, j): -float(demand[i, j]) / scale
        for i in range(n) for j in range(n) if i != j
    }

    # A busy transit node has only two outgoing links over which to spread
    # through-traffic, so discount paths through nodes with large incident demand.
    incident = np.sum(demand, axis=0) + np.sum(demand, axis=1)
    mean_incident = float(np.mean(incident)) or 1.0
    node_pressure = np.maximum(0.75, incident / mean_incident)

    # Allocate a capped two-hop reward per OD pair. Candidate paths with
    # stronger component-link support receive a larger share, while pressure
    # discourages concentrating transit through already busy nodes.
    quadratic = {}
    indirect_cap = 0.45
    for i in range(n):
        for k in range(n):
            if i == k:
                continue
            through_fraction = float(demand[i, k]) / scale
            if through_fraction <= 0.0:
                continue
            through_signal = (
                0.8 * through_fraction + 0.2 * np.sqrt(through_fraction)
            )
            mids = []
            evidence = []
            for j in range(n):
                if j == i or j == k:
                    continue
                edge_pressure = (
                    float(demand[i, j]) + float(demand[j, k])
                ) / scale
                component_support = np.sqrt(
                    max(0.0, float(demand[i, j]) / scale)
                    * max(0.0, float(demand[j, k]) / scale)
                )
                path_evidence = (
                    (0.20 + 0.80 * component_support)
                    / (node_pressure[j] * (1.0 + 0.75 * edge_pressure))
                )
                mids.append(j)
                evidence.append(path_evidence)

            total_evidence = float(sum(evidence))
            if total_evidence <= 0.0:
                continue
            budget = indirect_cap * through_signal
            for j, path_evidence in zip(mids, evidence):
                weight = -budget * path_evidence / total_evidence
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
