"""Surrogate QUBO terms for QOBLIB 08 network design. ShinkaEvolve edits the EVOLVE block."""

import time

# EVOLVE-BLOCK-START
import numpy as np


def _prepare_demand(demand):
    """Sanitize demand and produce globally normalized traffic features."""
    traffic = np.asarray(demand, dtype=float)
    n = int(traffic.shape[0])

    traffic = np.nan_to_num(
        traffic, nan=0.0, posinf=0.0, neginf=0.0
    )
    traffic = np.maximum(traffic, 0.0).copy()
    np.fill_diagonal(traffic, 0.0)

    scale = float(np.max(traffic)) if traffic.size else 0.0
    if scale <= 0.0:
        scale = 1.0

    normalized = traffic / scale
    roots = np.sqrt(normalized)

    incident = np.maximum(traffic.sum(axis=0), traffic.sum(axis=1))
    largest_incident = float(np.max(incident)) if incident.size else 0.0
    if largest_incident > 0.0:
        pressure = np.maximum(0.65, incident / largest_incident)
    else:
        pressure = np.ones(n, dtype=float)

    return traffic, normalized, roots, pressure


def _direct_link_terms(normalized, roots):
    """Give direct links a concave, demand-aware reward."""
    n = int(normalized.shape[0])
    return {
        (i, j): -float(0.75 * normalized[i, j] + 0.25 * roots[i, j])
        for i in range(n)
        for j in range(n)
        if i != j
    }


def _canonical_pair(first, second):
    return (first, second) if first < second else (second, first)


def _route_terms(traffic, normalized, pressure):
    """Build a sparse set of constituent-leg-supported two-hop rewards."""
    n = int(traffic.shape[0])
    positive = traffic[traffic > 0.0]
    cutoff = float(np.quantile(positive, 0.75)) if positive.size else 0.0
    quadratic = {}

    for source in range(n):
        for destination in range(n):
            if source == destination or traffic[source, destination] <= 0.0:
                continue

            endpoint_support = (
                0.70 * normalized[source, destination]
                + 0.30 * np.sqrt(normalized[source, destination])
            )

            candidates = []
            for transit in range(n):
                if transit == source or transit == destination:
                    continue

                first_leg = float(normalized[source, transit])
                second_leg = float(normalized[transit, destination])
                edge_pressure = first_leg + second_leg

                # Bounded evidence multiplier: both constituent legs matter,
                # without eliminating routes whose leg demands are modest.
                leg_support = float(np.sqrt(first_leg * second_leg))
                leg_factor = 0.58 + 0.42 * leg_support

                node_pressure = float(pressure[transit])
                affinity = (0.25 + edge_pressure) * leg_factor
                rank = affinity / (node_pressure ** 0.75)
                discount = node_pressure * (1.0 + 0.5 * edge_pressure)

                candidates.append(
                    (float(rank), transit, float(discount), leg_factor)
                )

            candidates.sort(key=lambda item: (-item[0], item[1]))
            if not candidates:
                continue

            keep = min(
                len(candidates),
                4 if traffic[source, destination] >= cutoff else 2,
            )
            selected = set()

            for position, (_, transit, discount, leg_factor) in enumerate(
                candidates[:keep]
            ):
                selected.add(transit)
                rank_factor = 1.0 + 0.08 * (keep - position)
                weight = (
                    -0.46
                    * float(endpoint_support)
                    * leg_factor
                    * rank_factor
                    / discount
                )

                first = (source, transit)
                second = (transit, destination)
                key = _canonical_pair(first, second)
                quadratic[key] = quadratic.get(key, 0.0) + float(weight)

            # Keep a distinct low-pressure alternative when it was not
            # already selected by the route-affinity ranking.
            remaining = [
                candidate
                for candidate in candidates
                if candidate[1] not in selected
            ]
            if remaining:
                _, transit, discount, leg_factor = min(
                    remaining,
                    key=lambda item: (item[2], item[1]),
                )
                first = (source, transit)
                second = (transit, destination)
                key = _canonical_pair(first, second)
                fallback = (
                    -0.30
                    * float(endpoint_support)
                    * leg_factor
                    / discount
                )
                quadratic[key] = quadratic.get(key, 0.0) + float(fallback)

    return quadratic


def _add_structural_terms(quadratic, linear, normalized, n):
    """Discourage wasteful reciprocal links while preserving useful ones."""
    mean_direct = -float(sum(linear.values())) / max(1, n * (n - 1))
    depth_scale = max(
        0.75,
        float(np.log2(max(n, 2))) / float(np.log2(10.0)),
    )
    backtrack_weight = 0.65 * mean_direct * depth_scale

    if backtrack_weight <= 0.0:
        return

    for i in range(n):
        for j in range(i + 1, n):
            mutual_demand = min(
                float(normalized[i, j]),
                float(normalized[j, i]),
            )
            key = ((i, j), (j, i))
            value = backtrack_weight * (1.0 - 0.45 * mutual_demand)
            quadratic[key] = quadratic.get(key, 0.0) + float(value)


def surrogate_terms(demand):
    """Return linear terms, sparse quadratic terms, and degree penalty."""
    traffic, normalized, roots, pressure = _prepare_demand(demand)
    n = int(traffic.shape[0])

    linear = _direct_link_terms(normalized, roots)
    quadratic = _route_terms(traffic, normalized, pressure)
    _add_structural_terms(quadratic, linear, normalized, n)

    penalty_weight = min(1.45, 1.30 + 0.008 * max(0, n - 10))
    return linear, quadratic, float(penalty_weight)


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
