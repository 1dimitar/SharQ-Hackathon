"""Surrogate QUBO terms for QOBLIB 08 network design. ShinkaEvolve edits the EVOLVE block."""

import time

# EVOLVE-BLOCK-START
import numpy as np


def surrogate_terms(demand):
    """Build linear and quadratic cost terms for a directed degree-2 network."""
    d = np.asarray(demand, dtype=float)
    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("demand must be a square matrix")

    n = int(d.shape[0])
    d = np.maximum(d, 0.0).copy()
    np.fill_diagonal(d, 0.0)

    dmax = float(np.max(d)) if d.size else 0.0
    if dmax > 0.0:
        d /= dmax

    row = d.sum(axis=1)
    col = d.sum(axis=0)
    node_load = row + col
    node_max = float(np.max(node_load)) if n else 0.0
    if node_max > 0.0:
        node_pressure = node_load / node_max
    else:
        node_pressure = np.zeros(n, dtype=float)

    # Direct demand remains the primary signal. Endpoint pressure adds a
    # modest preference for links useful to busy origins and destinations.
    linear = {}
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            endpoint = 0.5 * (node_pressure[i] + node_pressure[j])
            linear[(i, j)] = -float(0.90 * d[i, j] + 0.10 * endpoint)

    # Approximate the congestion of a possible edge using both its direct
    # demand and the traffic pressure at its endpoints.
    edge_pressure = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i != j:
                endpoint = 0.5 * (node_pressure[i] + node_pressure[j])
                edge_pressure[i, j] = 0.65 * d[i, j] + 0.35 * endpoint

    # Reward a limited portfolio of promising two-hop routes per OD pair.
    # Component-edge load is discounted more strongly than in a purely
    # demand-based route score, and high-pressure transit nodes are avoided.
    quadratic = {}
    candidate_count = min(2, max(0, n - 2))

    if candidate_count:
        for i in range(n):
            for k in range(n):
                traffic = float(d[i, k])
                if i == k or traffic <= 0.0:
                    continue

                ranked = []
                for j in range(n):
                    if j == i or j == k:
                        continue

                    e1 = float(edge_pressure[i, j])
                    e2 = float(edge_pressure[j, k])
                    transit = float(node_pressure[j])

                    # Each edge has an independent congestion discount;
                    # transit pressure supplies an additional mild penalty.
                    quality = (
                        1.0
                        / ((1.0 + 0.85 * e1) ** 1.5
                           * (1.0 + 0.85 * e2) ** 1.5)
                    ) / (1.0 + 0.35 * transit)
                    ranked.append((quality, j))

                ranked.sort(reverse=True)
                chosen = ranked[:candidate_count]
                quality_sum = sum(q for q, _ in chosen)
                if quality_sum <= 0.0:
                    continue

                # Keep indirect value subordinate to a direct-link reward,
                # while distributing it according to route quality.
                total_reward = 0.18 * traffic
                for quality, j in chosen:
                    weight = -total_reward * quality / quality_sum
                    quadratic[((i, j), (j, k))] = float(weight)

    # Pairwise load-balancing terms: jointly choosing links that connect
    # highly pressured nodes incurs a small cost. These terms are sparse
    # (links sharing a tail or a head) and fit within the quadratic budget.
    balance_strength = 0.045
    for i in range(n):
        others = [j for j in range(n) if j != i]
        for a in range(len(others)):
            j = others[a]
            for b in range(a + 1, len(others)):
                k = others[b]
                quadratic[((i, j), (i, k))] = (
                    balance_strength
                    * float(node_pressure[j])
                    * float(node_pressure[k])
                )

    for j in range(n):
        others = [i for i in range(n) if i != j]
        for a in range(len(others)):
            i = others[a]
            for b in range(a + 1, len(others)):
                k = others[b]
                key = ((i, j), (k, j))
                value = (
                    balance_strength
                    * float(node_pressure[i])
                    * float(node_pressure[k])
                )
                # No path term has this key; addition also safely handles
                # any coincident contribution should the construction change.
                quadratic[key] = quadratic.get(key, 0.0) + value

    # The evaluator applies degree penalties separately.
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
