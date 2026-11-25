"""Surrogate QUBO terms for QOBLIB 08 network design. ShinkaEvolve edits the EVOLVE block."""

import time

# EVOLVE-BLOCK-START
import numpy as np


def surrogate_terms(demand):
    """Build direct-link rewards and a sparse set of pressure-aware relay terms."""
    matrix = np.asarray(demand, dtype=float)
    n = len(matrix)

    traffic = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    traffic = np.maximum(traffic, 0.0).copy()
    np.fill_diagonal(traffic, 0.0)

    scale = float(np.max(traffic)) if traffic.size else 0.0
    if scale <= 0.0:
        scale = 1.0
    traffic /= scale

    # Reward directly serving high-demand OD pairs, while retaining a
    # concave component so moderate demands also influence link selection.
    roots = np.sqrt(traffic)
    linear = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                value = float(traffic[i, j])
                linear[(i, j)] = -(0.75 * value + 0.25 * float(roots[i, j]))

    incoming = np.sum(traffic, axis=0)
    outgoing = np.sum(traffic, axis=1)
    incident = np.maximum(incoming, outgoing)
    max_incident = float(np.max(incident)) if incident.size else 0.0
    if max_incident > 0.0:
        pressure = np.maximum(0.65, incident / max_incident)
    else:
        pressure = np.ones(n, dtype=float)

    positive = traffic[traffic > 0.0]
    cutoff = float(np.quantile(positive, 0.75)) if positive.size else 0.0

    quadratic = {}
    for source in range(n):
        for target in range(n):
            if source == target:
                continue

            od = float(traffic[source, target])
            if od <= 0.0:
                continue

            support = 0.70 * od + 0.30 * float(roots[source, target])
            keep = min(n - 2, 4 if od >= cutoff else 2)
            candidates = []

            for relay in range(n):
                if relay == source or relay == target:
                    continue

                left = float(traffic[source, relay])
                right = float(traffic[relay, target])
                relay_pressure = float(pressure[relay])

                # Geometric leg affinity favors routes where both legs are
                # plausible; the additive component preserves useful routes
                # when one leg carries little direct demand.
                geometric = np.sqrt((0.05 + left) * (0.05 + right))
                additive = left + right
                affinity = 0.70 * geometric + 0.30 * additive
                rank = affinity / (0.75 + relay_pressure)
                candidates.append((rank, relay, affinity, relay_pressure, left + right))

            # Stable tie-breaking makes output deterministic.
            candidates.sort(key=lambda item: (-item[0], item[1]))

            if keep <= 0:
                continue

            selected = candidates[:keep]
            # Reserve one selected slot for a low-pressure relay when it is
            # not already among the strongest-affinity candidates.
            least_pressure = min(
                candidates, key=lambda item: (item[3], -item[2], item[1])
            )
            if keep >= 2 and least_pressure not in selected:
                selected[-1] = least_pressure

            for position, (_, relay, affinity, relay_pressure, leg_sum) in enumerate(selected):
                first = (source, relay)
                second = (relay, target)
                key = (first, second) if first < second else (second, first)

                rank_factor = 1.0 - 0.05 * position
                leg_factor = 0.85 + 0.15 * affinity / (0.10 + affinity)
                pressure_discount = (0.80 + relay_pressure) * (1.0 + 0.25 * leg_sum)

                weight = (
                    -0.40
                    * support
                    * rank_factor
                    * leg_factor
                    / pressure_discount
                )
                quadratic[key] = quadratic.get(key, 0.0) + weight

    # Degree penalties are added by the evaluator, not encoded here.
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
