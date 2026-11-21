"""Surrogate QUBO terms for QOBLIB 08 network design. ShinkaEvolve edits the EVOLVE block."""

import time

# EVOLVE-BLOCK-START
import numpy as np


def _prepare_demand(demand):
    """Normalize traffic and calculate relay-node pressure."""
    traffic = np.maximum(np.asarray(demand, dtype=float), 0.0).copy()
    np.fill_diagonal(traffic, 0.0)

    scale = float(np.max(traffic)) if traffic.size else 0.0
    if scale <= 0.0:
        scale = 1.0

    normalized = traffic / scale
    incident = np.maximum(traffic.sum(axis=0), traffic.sum(axis=1))
    max_incident = float(np.max(incident)) if incident.size else 0.0
    if max_incident <= 0.0:
        max_incident = 1.0
    pressure = np.maximum(0.65, incident / max_incident)

    positive = traffic[traffic > 0.0]
    cutoff = float(np.quantile(positive, 0.90)) if positive.size else 0.0
    return traffic, normalized, pressure, cutoff


def _direct_terms(normalized):
    """Reward direct service, including moderate-demand links."""
    n = len(normalized)
    roots = np.sqrt(normalized)
    linear = {
        (i, j): -float(0.75 * normalized[i, j] + 0.25 * roots[i, j])
        for i in range(n)
        for j in range(n)
        if i != j
    }
    return linear, roots


def _relay_terms(traffic, normalized, roots, pressure, cutoff):
    """Select two relays per OD pair, plus two for exceptional demands."""
    n = len(normalized)
    quadratic = {}

    for origin in range(n):
        for destination in range(n):
            demand = float(traffic[origin, destination])
            if origin == destination or demand <= 0.0:
                continue

            support = float(
                0.70 * normalized[origin, destination]
                + 0.30 * roots[origin, destination]
            )
            candidates = []
            for relay in range(n):
                if relay == origin or relay == destination:
                    continue
                edge_pressure = float(
                    normalized[origin, relay]
                    + normalized[relay, destination]
                )
                rank = (0.25 + edge_pressure) / float(pressure[relay])
                candidates.append((rank, relay, edge_pressure))

            # The node index provides deterministic tie-breaking.
            candidates.sort(reverse=True)
            keep = 4 if demand >= cutoff else 2

            for position, (_, relay, edge_pressure) in enumerate(candidates[:keep]):
                first = (origin, relay)
                second = (relay, destination)
                key = (first, second) if first < second else (second, first)
                rank_factor = 1.0 + 0.08 * (keep - position)
                quadratic[key] = (
                    -0.46 * support * rank_factor
                    / (float(pressure[relay]) * (1.0 + 0.5 * edge_pressure))
                )

    return quadratic


def surrogate_terms(demand):
    """Return direct-link costs, selected relay rewards, and penalty weight."""
    traffic, normalized, pressure, cutoff = _prepare_demand(demand)
    linear, roots = _direct_terms(normalized)
    quadratic = _relay_terms(traffic, normalized, roots, pressure, cutoff)
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
