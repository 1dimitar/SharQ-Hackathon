"""Surrogate QUBO terms for QOBLIB 08 network design. ShinkaEvolve edits the EVOLVE block."""

import time

# EVOLVE-BLOCK-START
import numpy as np


def surrogate_terms(demand):
    """Estimate link congestion and penalize concentrated endpoint capacity."""
    matrix = np.asarray(demand, dtype=float)
    n = len(matrix)

    traffic = np.maximum(matrix, 0.0).copy()
    np.fill_diagonal(traffic, 0.0)
    scale = float(np.max(traffic)) if traffic.size else 0.0
    if scale <= 0.0:
        scale = 1.0
    flow = traffic / scale

    # Distribute each OD pair's traffic over the direct edge and plausible
    # two-hop alternatives. The direct route gets a fixed preference;
    # alternatives through busy nodes receive less of the fractional flow.
    incident = traffic.sum(axis=0) + traffic.sum(axis=1)
    mean_incident = float(np.mean(incident)) if n else 0.0
    if mean_incident <= 0.0:
        mean_incident = 1.0
    pressure = np.maximum(0.65, incident / mean_incident)

    estimated_load = np.zeros((n, n), dtype=float)
    for i in range(n):
        for k in range(n):
            amount = float(flow[i, k])
            if i == k or amount <= 0.0:
                continue

            # Split flow between direct service and alternatives. This avoids
            # treating every possible path as if it carried the full demand.
            estimated_load[i, k] += 0.62 * amount
            intermediates = []
            weights = []
            for j in range(n):
                if j == i or j == k:
                    continue
                w = 1.0 / (
                    float(pressure[j])
                    * (1.0 + 0.5 * (float(flow[i, j]) + float(flow[j, k])))
                )
                intermediates.append(j)
                weights.append(w)

            total = float(sum(weights))
            if total > 0.0:
                for j, w in zip(intermediates, weights):
                    share = 0.38 * amount * w / total
                    estimated_load[i, j] += share
                    estimated_load[j, k] += share

    # Prefer links with significant direct demand, but make the estimated
    # aggregate load the main signal; sqrt keeps moderate links influential.
    linear = {}
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            load = float(estimated_load[i, j])
            direct = float(flow[i, j])
            linear[(i, j)] = 0.22 * direct - 0.78 * load

    # Endpoint crowding: selected links sharing a tail or head consume the
    # same limited two-link capacity. Penalize pairs when both are predicted
    # to carry substantial load, with a mild discount for similar roles.
    quadratic = {}
    for i in range(n):
        outgoing = [(i, j) for j in range(n) if j != i]
        incoming = [(j, i) for j in range(n) if j != i]

        for group in (outgoing, incoming):
            for a in range(len(group)):
                e1 = group[a]
                l1 = float(estimated_load[e1])
                for b in range(a + 1, len(group)):
                    e2 = group[b]
                    l2 = float(estimated_load[e2])
                    # Convex overload cost: jointly selecting two busy links
                    # is costlier than selecting either one alone.
                    value = 0.18 * min(l1, l2) + 0.06 * np.sqrt(l1 * l2)
                    if value > 0.0:
                        key = (e1, e2) if e1 < e2 else (e2, e1)
                        quadratic[key] = float(value)

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
