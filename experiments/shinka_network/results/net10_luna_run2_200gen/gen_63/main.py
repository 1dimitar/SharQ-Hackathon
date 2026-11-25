"""Surrogate QUBO terms for QOBLIB 08 network design. ShinkaEvolve edits the EVOLVE block."""

import time

# EVOLVE-BLOCK-START
import numpy as np


def surrogate_terms(demand):
    """Build surrogate cost terms for the directed network-design QUBO."""
    raw = np.asarray(demand, dtype=float)
    n = int(raw.shape[0])

    # Ignore negative demand and self-traffic; use one common scale so all
    # terms remain bounded and well-defined for an all-zero matrix.
    d = np.maximum(raw, 0.0).copy()
    np.fill_diagonal(d, 0.0)
    scale = float(np.max(d))
    if scale <= 0.0:
        scale = 1.0
    d /= scale

    # Direct service remains the main signal. The concave component gives
    # moderate-demand links some influence without making them equal to the
    # largest demands.
    linear = {}
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            value = float(d[i, j])
            linear[(i, j)] = -(0.90 * value + 0.10 * np.sqrt(value))

    # Approximate indirect service by rewarding candidate two-hop paths.
    # Splitting the reward by four reflects the maximum number of combinations
    # of two outgoing/incoming links available to a node pair. Busy transit
    # nodes receive a smaller share.
    node_load = np.maximum(d.sum(axis=0), d.sum(axis=1))
    max_node_load = float(np.max(node_load))
    if max_node_load <= 0.0:
        max_node_load = 1.0

    quadratic = {}
    indirect_fraction = 0.60
    pressure_floor = 0.75
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            pressure = float(node_load[j]) / max_node_load
            transit_factor = 1.0 - (1.0 - pressure_floor) * pressure
            for k in range(n):
                if k == i or k == j:
                    continue
                traffic = float(d[i, k])
                if traffic <= 0.0:
                    continue
                weight = (
                    -indirect_fraction * traffic * transit_factor / 4.0
                )
                quadratic[((i, j), (j, k))] = weight

    # Degree penalties are added by the evaluator.
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
