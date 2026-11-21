"""Surrogate QUBO terms for QOBLIB 08 network design. ShinkaEvolve edits the EVOLVE block."""

import time

# EVOLVE-BLOCK-START
import numpy as np


def _prepare_demand(demand):
    """Return nonnegative, off-diagonal demand normalized to [0, 1]."""
    traffic = np.maximum(np.asarray(demand, dtype=float), 0.0).copy()
    np.fill_diagonal(traffic, 0.0)
    scale = float(np.max(traffic)) if traffic.size else 0.0
    if scale > 0.0:
        traffic /= scale
    return traffic


def _node_pressure(traffic):
    """Estimate transit pressure from each node's larger incident demand."""
    incoming = np.sum(traffic, axis=0)
    outgoing = np.sum(traffic, axis=1)
    incident = np.maximum(incoming, outgoing)
    largest = float(np.max(incident)) if incident.size else 0.0
    if largest <= 0.0:
        return np.ones(len(traffic), dtype=float)
    return np.maximum(0.65, incident / largest)


def _relay_candidates(traffic, pressure, source, target):
    """Yield eligible relays ordered by OD-leg affinity and relay pressure."""
    n = len(traffic)
    candidates = []
    for relay in range(n):
        if relay == source or relay == target:
            continue

        first_leg = float(traffic[source, relay])
        second_leg = float(traffic[relay, target])

        # The small floor allows useful relays even when one leg has little
        # direct demand; the geometric mean favors relays supporting both legs.
        affinity = np.sqrt((0.1 + first_leg) * (0.1 + second_leg))
        quality = affinity / (0.75 + float(pressure[relay]))
        candidates.append((quality, relay))

    candidates.sort(key=lambda item: (-item[0], item[1]))
    return candidates


def surrogate_terms(demand):
    """Build direct-link rewards and a sparse set of two-hop relay rewards."""
    traffic = _prepare_demand(demand)
    n = len(traffic)
    pressure = _node_pressure(traffic)

    # Direct-link blend preserves substantial influence for moderate demands.
    linear = {}
    positive = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            value = float(traffic[i, j])
            linear[(i, j)] = -(0.65 * value + 0.35 * np.sqrt(value))
            if value > 0.0:
                positive.append(value)

    if positive:
        high_cutoff = float(np.quantile(np.asarray(positive), 0.80))
    else:
        high_cutoff = float("inf")

    base_relays = min(n - 2, max(2, n // 4))
    extra_relays = min(2, max(0, n - 2 - base_relays))
    path_strength = 0.28
    quadratic = {}

    for source in range(n):
        for target in range(n):
            if source == target:
                continue

            od = float(traffic[source, target])
            if od <= 0.0:
                continue

            support = 0.75 * od + 0.25 * np.sqrt(od)
            count = base_relays + (extra_relays if od >= high_cutoff else 0)

            for quality, relay in _relay_candidates(
                traffic, pressure, source, target
            )[:count]:
                # Bounded quality adjustment avoids making one relay dominate.
                quality_factor = 0.8 + 0.2 * quality / (1.0 + quality)
                weight = (
                    -path_strength * support * quality_factor
                    / (0.75 + float(pressure[relay]))
                )

                first = (source, relay)
                second = (relay, target)
                key = (first, second) if first < second else (second, first)
                quadratic[key] = quadratic.get(key, 0.0) + weight

    # Degree penalties are added by the evaluator.
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
