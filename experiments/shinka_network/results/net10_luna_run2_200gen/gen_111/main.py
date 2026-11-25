"""Surrogate QUBO terms for QOBLIB 08 network design. ShinkaEvolve edits the EVOLVE block."""

import time

# EVOLVE-BLOCK-START
import numpy as np


def _prepare_demand(demand):
    """Return finite, nonnegative, off-diagonal demand."""
    d = np.asarray(demand, dtype=float).copy()
    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("demand must be a square matrix")
    d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
    np.maximum(d, 0.0, out=d)
    np.fill_diagonal(d, 0.0)

    scale = float(np.max(d)) if d.size else 0.0
    if scale > 0.0:
        d /= scale
    return d


def surrogate_terms(demand):
    """Build direct-link rewards and congestion-aware two-hop terms."""
    d = _prepare_demand(demand)
    n = d.shape[0]

    row = d.sum(axis=1)
    col = d.sum(axis=0)
    node_scale = max(float(np.max(row)), float(np.max(col)), 1.0)

    # Direct service remains the main signal. The small concave component
    # preserves some influence for moderate-demand links.
    linear = {}
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            value = float(d[i, j])
            endpoint_pressure = (float(row[i]) + float(col[j])) / (
                2.0 * node_scale
            )
            linear[(i, j)] = -(
                0.86 * value
                + 0.08 * np.sqrt(value)
                + 0.06 * endpoint_pressure
            )

    # Estimate relay availability using two different normalizations. The
    # mean-based term reflects typical load; the max-based term responds to
    # unusually busy nodes. The floor avoids eliminating any relay entirely.
    node_load = row + col
    mean_load = float(np.mean(node_load))
    max_load = float(np.max(node_load)) if node_load.size else 0.0
    eps = 1.0e-12

    relay_mean = mean_load / (mean_load + node_load + eps)
    if max_load > 0.0:
        relay_max = 1.0 - 0.45 * node_load / (max_load + eps)
    else:
        relay_max = np.ones(n, dtype=float)

    relay = np.maximum(0.68, 0.5 * relay_mean + 0.5 * relay_max)

    # Each OD pair contributes only a bounded reward, distributed among a few
    # plausible relays. Component links with high direct demand are treated as
    # more congested, so indirect traffic is steered toward lighter links.
    quadratic = {}
    candidate_count = min(4, max(0, n - 2))

    for i in range(n):
        for k in range(n):
            traffic = float(d[i, k])
            if i == k or traffic <= 0.0 or candidate_count == 0:
                continue

            candidates = []
            for j in range(n):
                if j == i or j == k:
                    continue

                component_load = float(d[i, j]) + float(d[j, k])
                spare_factor = 1.0 / (1.0 + 0.70 * component_load)
                score = float(relay[j]) * spare_factor
                candidates.append((score, j))

            candidates.sort(reverse=True)
            chosen = candidates[:candidate_count]
            score_sum = sum(score for score, _ in chosen)
            if score_sum <= 0.0:
                continue

            # Cap the total path signal relative to the OD traffic so that
            # many alternative relays cannot overwhelm direct-link terms.
            total_reward = min(0.28, 0.48 * traffic)
            for score, j in chosen:
                weight = -total_reward * score / score_sum
                quadratic[((i, j), (j, k))] = float(weight)

    return linear, quadratic, 1.0


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
