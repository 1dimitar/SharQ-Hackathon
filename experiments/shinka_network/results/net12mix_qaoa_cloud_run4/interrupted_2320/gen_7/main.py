"""Surrogate QUBO terms for QOBLIB 08 network design. ShinkaEvolve edits the EVOLVE block."""

import time

# EVOLVE-BLOCK-START
import numpy as np


def _normalized_demand(demand):
    """Return a nonnegative, max-normalized demand array."""
    t = np.asarray(demand, dtype=float)
    t = np.maximum(t, 0.0)
    scale = float(t.max()) if t.size else 0.0
    if scale > 0.0:
        t = t / scale
    return t


def _direct_link_terms(t):
    """Reward links that directly serve substantial demand."""
    n = t.shape[0]
    return {
        (i, j): -float(t[i, j])
        for i in range(n)
        for j in range(n)
        if i != j
    }


def _two_hop_terms(t):
    """Build moderated rewards for useful two-link routes.

    For each OD pair, its indirect-route reward is split across candidate
    intermediaries. Nodes with larger incident demand receive less of that
    reward, as they are less attractive transit points.
    """
    n = t.shape[0]
    incident = t.sum(axis=0) + t.sum(axis=1)
    transit_weight = 1.0 / np.sqrt(1.0 + incident)

    quadratic = {}
    for source in range(n):
        for destination in range(n):
            amount = float(t[source, destination])
            if source == destination or amount <= 0.0:
                continue

            candidates = [
                mid for mid in range(n)
                if mid != source and mid != destination
            ]
            total_weight = sum(float(transit_weight[mid]) for mid in candidates)
            if total_weight <= 0.0:
                continue

            # Keep indirect routing informative without letting it dominate
            # the direct-link signal.
            for mid in candidates:
                share = float(transit_weight[mid]) / total_weight
                first = (source, mid)
                second = (mid, destination)
                key = (first, second)
                quadratic[key] = quadratic.get(key, 0.0) - 0.30 * amount * share

    return quadratic


def surrogate_terms(demand):
    """Build cost terms for the directed degree-constrained network QUBO."""
    t = _normalized_demand(demand)
    linear = _direct_link_terms(t)
    quadratic = _two_hop_terms(t)

    # Normalize all objective coefficients together so relative scales are
    # preserved. The evaluator adds the degree penalties separately.
    largest = max(
        max((abs(v) for v in linear.values()), default=0.0),
        max((abs(v) for v in quadratic.values()), default=0.0),
    )
    if largest > 0.0:
        linear = {key: value / largest for key, value in linear.items()}
        quadratic = {key: value / largest for key, value in quadratic.items()}

    penalty_weight = 2.0
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
