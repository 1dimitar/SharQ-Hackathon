"""Surrogate QUBO terms for QOBLIB 08 network design. ShinkaEvolve edits the EVOLVE block."""

import time

# EVOLVE-BLOCK-START
import numpy as np


def surrogate_terms(demand):
    """Build direct-link and sparse, demand-aware two-hop route terms."""
    traffic = np.asarray(demand, dtype=float)
    if traffic.ndim != 2 or traffic.shape[0] != traffic.shape[1]:
        raise ValueError("demand must be a square matrix")

    n = traffic.shape[0]
    traffic = np.nan_to_num(traffic, nan=0.0, posinf=0.0, neginf=0.0)
    traffic = np.maximum(traffic, 0.0).copy()
    np.fill_diagonal(traffic, 0.0)

    scale = float(np.max(traffic)) if traffic.size else 0.0
    if scale <= 0.0:
        scale = 1.0
    normalized = traffic / scale

    # Direct-link utility combines the demand signal with a modest concave
    # component, so medium-sized OD pairs are not completely ignored.
    linear = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                d = float(normalized[i, j])
                linear[(i, j)] = -(0.82 * d + 0.18 * np.sqrt(d))

    # Estimate relay congestion from both the traffic a node emits and
    # the traffic it receives. A geometric mean avoids treating a node with
    # only high incoming or only high outgoing demand as a perfect relay.
    row_load = np.sum(normalized, axis=1)
    col_load = np.sum(normalized, axis=0)
    mean_row = float(np.mean(row_load)) if n else 0.0
    mean_col = float(np.mean(col_load)) if n else 0.0

    if mean_row > 0.0 and mean_col > 0.0:
        relay_pressure = np.sqrt(
            np.maximum(row_load / mean_row, 0.0)
            * np.maximum(col_load / mean_col, 0.0)
        )
    else:
        relay_pressure = np.ones(n, dtype=float)

    # Bound pressure estimates so one noisy node statistic cannot eliminate
    # all paths through a node.
    relay_pressure = np.clip(relay_pressure, 0.55, 2.5)

    quadratic = {}
    route_strength = 0.40
    max_relays = 2

    for i in range(n):
        for k in range(n):
            if i == k:
                continue

            d = float(normalized[i, k])
            if d <= 0.0 or n < 3:
                continue

            demand_support = 0.72 * d + 0.28 * np.sqrt(d)
            candidates = []

            for j in range(n):
                if j == i or j == k:
                    continue

                # A relay is less attractive if it is already busy, or if
                # either link in the path has substantial direct traffic.
                edge_load = float(normalized[i, j] + normalized[j, k])
                quality = 1.0 / (
                    float(relay_pressure[j]) * (1.0 + 0.65 * edge_load)
                )
                candidates.append((quality, j))

            # Keep only the strongest few routes for this OD pair. The small
            # rank discount lets a second good route remain useful without
            # giving it the same influence as the best candidate.
            candidates.sort(reverse=True)
            for rank, (quality, j) in enumerate(candidates[:max_relays]):
                rank_factor = 1.0 if rank == 0 else 0.72
                weight = -route_strength * demand_support * quality * rank_factor

                first = (i, j)
                second = (j, k)
                key = (first, second) if first < second else (second, first)
                quadratic[key] = float(weight)

    # The evaluator adds the degree penalties separately.
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
