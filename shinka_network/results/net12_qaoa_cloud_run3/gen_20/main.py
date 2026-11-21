"""Surrogate QUBO terms for QOBLIB 08 network design. ShinkaEvolve edits the EVOLVE block."""

import time

# EVOLVE-BLOCK-START
import numpy as np


def surrogate_terms(demand):
    """Build direct-link rewards and sparse bottleneck-aware relay terms."""
    matrix = np.asarray(demand, dtype=float)
    n = len(matrix)

    traffic = np.maximum(matrix, 0.0).copy()
    np.fill_diagonal(traffic, 0.0)

    scale = float(np.max(traffic)) if traffic.size else 0.0
    if scale <= 0.0:
        scale = 1.0
    normalized = traffic / scale

    # Reward direct service, while retaining some influence for moderate
    # demands. All available links receive a weight for consistent keys.
    linear = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                value = float(normalized[i, j])
                linear[(i, j)] = -(0.78 * value + 0.22 * np.sqrt(value))

    if n < 3 or not np.any(traffic):
        return linear, {}, 1.5

    incoming = np.sum(traffic, axis=0)
    outgoing = np.sum(traffic, axis=1)
    incident = np.maximum(incoming, outgoing)
    pressure_scale = float(np.max(incident))
    if pressure_scale <= 0.0:
        pressure_scale = 1.0
    pressure = np.maximum(0.60, incident / pressure_scale)

    quadratic = {}
    # Two relays per OD pair retain alternatives without creating a dense
    # interaction graph. A third is reserved for especially important OD.
    positive = traffic[traffic > 0.0]
    important_cutoff = float(np.quantile(positive, 0.80)) if positive.size else 0.0
    eps = 1.0e-4

    for i in range(n):
        for k in range(n):
            demand_ik = float(normalized[i, k])
            if i == k or demand_ik <= 0.0:
                continue

            # Demand support is deliberately concave: small but nonzero
            # traffic remains relevant, without equalling the largest flows.
            demand_support = 0.72 * demand_ik + 0.28 * np.sqrt(demand_ik)
            candidates = []

            for j in range(n):
                if j == i or j == k:
                    continue

                a = float(normalized[i, j])
                b = float(normalized[j, k])

                # The minimum leg is the bottleneck; the geometric factor
                # gives limited credit when both legs have useful support.
                low = min(a, b)
                high = max(a, b)
                affinity = np.sqrt(low * (high + eps))

                # Prefer viable relays but discount nodes likely to be
                # heavily occupied by their own incident traffic.
                rank = affinity / (float(pressure[j]) ** 0.70)
                candidates.append((rank, j, affinity))

            candidates.sort(key=lambda item: (-item[0], item[1]))
            keep = 3 if traffic[i, k] >= important_cutoff else 2

            for rank, j, affinity in candidates[:keep]:
                if affinity <= 0.0:
                    continue
                # Mildly reduce route rewards for highly pressured relays.
                weight = (
                    -0.48 * demand_support * affinity
                    / (float(pressure[j]) ** 0.45)
                )
                first, second = (i, j), (j, k)
                key = (first, second) if first < second else (second, first)
                quadratic[key] = quadratic.get(key, 0.0) + weight

    # The evaluator supplies the degree penalties separately.
    return linear, quadratic, 1.5


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
