"""Surrogate QUBO terms for QOBLIB 08 network design. ShinkaEvolve edits the EVOLVE block."""

import time

# EVOLVE-BLOCK-START
import numpy as np


def surrogate_terms(demand):
    """Build direct-link and pressure/affinity-discounted two-hop terms."""
    traffic = np.asarray(demand, dtype=float)
    n = len(traffic)

    # Sanitize input and exclude self-links and self-traffic.
    traffic = np.nan_to_num(
        traffic, nan=0.0, posinf=0.0, neginf=0.0
    )
    traffic = np.maximum(traffic, 0.0).copy()
    np.fill_diagonal(traffic, 0.0)

    scale = float(np.max(traffic)) if traffic.size else 0.0
    if scale <= 0.0:
        scale = 1.0
    q = traffic / scale

    # Strongly favor direct service of heavy OD pairs, while allowing
    # moderate demands to contribute.
    linear = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                d = float(q[i, j])
                linear[(i, j)] = -(0.82 * d + 0.18 * np.sqrt(d))

    # Approximate relay pressure using the heavier of each node's incoming
    # and outgoing aggregate demands.
    incident = np.maximum(np.sum(q, axis=0), np.sum(q, axis=1))
    largest_incident = float(np.max(incident)) if incident.size else 0.0
    if largest_incident > 0.0:
        pressure = np.maximum(0.65, incident / largest_incident)
    else:
        pressure = np.full(n, 0.65, dtype=float)

    quadratic = {}
    path_strength = 0.44

    # Keep coverage across possible relays, but give modest preference to
    # relays whose two legs have useful demand and avoid especially busy
    # relay nodes or already-demanded legs.
    for i in range(n):
        for k in range(n):
            if i == k:
                continue

            d = float(q[i, k])
            if d <= 0.0:
                continue
            support = 0.72 * d + 0.28 * np.sqrt(d)

            for j in range(n):
                if j == i or j == k:
                    continue

                a = float(q[i, j])
                b = float(q[j, k])
                leg_pressure = a + b

                affinity = min(a, b) + 0.30 * max(a, b)
                affinity_factor = 0.80 + 0.20 * (affinity / 1.30)
                relay_factor = float(pressure[j])
                edge_factor = 1.0 + 0.25 * leg_pressure

                weight = (
                    -path_strength
                    * support
                    * affinity_factor
                    / (relay_factor * edge_factor)
                )

                first, second = (i, j), (j, k)
                key = (first, second) if first < second else (second, first)
                quadratic[key] = weight

    # The evaluator adds the outgoing/incoming degree penalties.
    penalty_weight = 1.5
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
