"""Surrogate QUBO terms for QOBLIB 08 network design. ShinkaEvolve edits the EVOLVE block."""

import time

# EVOLVE-BLOCK-START
import numpy as np


def surrogate_terms(demand):
    """Build direct rewards, endpoint-congestion interactions, and sparse path rewards."""
    traffic = np.asarray(demand, dtype=float)
    n = len(traffic)

    # Sanitize the input and ensure self-links never contribute.
    traffic = np.maximum(traffic, 0.0).copy()
    np.fill_diagonal(traffic, 0.0)
    scale = float(np.max(traffic)) if traffic.size else 0.0
    if scale <= 0.0:
        scale = 1.0
    d = traffic / scale

    linear = {
        (i, j): -float(d[i, j])
        for i in range(n) for j in range(n) if i != j
    }

    # Incident demand approximates how costly it is for a node to act as a
    # transit endpoint. A square-root transform prevents a single hub from
    # suppressing nearly all paths.
    incident = traffic.sum(axis=0) + traffic.sum(axis=1)
    mean_incident = float(np.mean(incident)) if n else 0.0
    if mean_incident <= 0.0:
        mean_incident = 1.0
    pressure = np.sqrt(np.maximum(0.25, incident / mean_incident))

    quadratic = {}

    # Congestion proxy: two selected links sharing a tail compete for that
    # node's outgoing capacity; links sharing a head compete for its incoming
    # capacity. Penalize pairs carrying substantial direct traffic, especially
    # at already busy endpoints. These interactions are O(n^3).
    congestion_strength = 0.16
    for a in range(n):
        outgoing = [b for b in range(n) if b != a]
        incoming = [b for b in range(n) if b != a]

        for p in range(len(outgoing)):
            b = outgoing[p]
            for q in range(p + 1, len(outgoing)):
                c = outgoing[q]
                load_pair = np.sqrt(d[a, b] * d[a, c])
                factor = 0.5 + 0.5 * pressure[a] / (1.0 + pressure[a])
                e1, e2 = (a, b), (a, c)
                key = (e1, e2) if e1 < e2 else (e2, e1)
                quadratic[key] = congestion_strength * load_pair * factor

        for p in range(len(incoming)):
            b = incoming[p]
            for q in range(p + 1, len(incoming)):
                c = incoming[q]
                load_pair = np.sqrt(d[b, a] * d[c, a])
                factor = 0.5 + 0.5 * pressure[a] / (1.0 + pressure[a])
                e1, e2 = (b, a), (c, a)
                key = (e1, e2) if e1 < e2 else (e2, e1)
                # An edge pair can only share one endpoint in this directed
                # graph unless it is a reciprocal pair, which is handled once.
                quadratic[key] = congestion_strength * load_pair * factor

    # Sparse path signal: for each OD pair, select only the best few
    # intermediate nodes by demand support and transit pressure. This avoids
    # accumulating weak rewards for many implausible paths.
    path_strength = 0.12
    keep = 2
    for i in range(n):
        for k in range(n):
            if i == k or d[i, k] <= 0.0:
                continue
            candidates = []
            for j in range(n):
                if j == i or j == k:
                    continue
                support = np.sqrt(d[i, j] * d[j, k])
                # Traffic on the two component links is evidence that the
                # route is useful, while busy intermediate nodes are discounted.
                merit = support / (0.5 + pressure[j])
                candidates.append((merit, j))
            candidates.sort(reverse=True)
            for merit, j in candidates[:keep]:
                if merit <= 0.0:
                    continue
                e1, e2 = (i, j), (j, k)
                key = (e1, e2) if e1 < e2 else (e2, e1)
                reward = -path_strength * d[i, k] * merit
                # Preserve the congestion term if an identical key is present.
                quadratic[key] = quadratic.get(key, 0.0) + reward

    # The evaluator supplies the degree constraints and rescales all terms.
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
