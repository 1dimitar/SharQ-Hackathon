"""Surrogate QUBO terms for QOBLIB 08 network design. ShinkaEvolve edits the EVOLVE block."""

import time

# EVOLVE-BLOCK-START
import numpy as np


def _canonical_pair(first, second):
    """Return a stable key for an unordered pair of directed links."""
    return (first, second) if first < second else (second, first)


def _add_quadratic(terms, first, second, weight):
    """Accumulate a quadratic coefficient, ignoring numerical zeros."""
    if abs(weight) <= 1e-15:
        return
    key = _canonical_pair(first, second)
    terms[key] = terms.get(key, 0.0) + float(weight)


def _clean_demand(demand):
    """Validate, sanitize, and normalize the demand matrix."""
    matrix = np.asarray(demand, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("demand must be a square matrix")

    traffic = np.nan_to_num(
        matrix, copy=True, nan=0.0, posinf=0.0, neginf=0.0
    )
    np.maximum(traffic, 0.0, out=traffic)
    np.fill_diagonal(traffic, 0.0)

    scale = float(traffic.max()) if traffic.size else 0.0
    if scale > 0.0:
        normalized = traffic / scale
    else:
        normalized = np.zeros_like(traffic)
    return traffic, normalized


def _build_route_catalog(normalized, incident):
    """Cache useful route features and iterate a congestion estimate."""
    n = normalized.shape[0]
    max_incident = max(float(incident.max()) if incident.size else 0.0, 1e-12)
    node_base = incident / max_incident
    pressure = np.maximum(0.55, node_base)

    # Each record stores an OD pair, its demand support, and all possible
    # intermediary candidates as (node, base route affinity).
    catalog = []
    for i in range(n):
        for k in range(n):
            demand_ik = float(normalized[i, k])
            if i == k or demand_ik <= 0.0:
                continue

            candidates = []
            for j in range(n):
                if j == i or j == k:
                    continue
                first_leg = float(normalized[i, j])
                second_leg = float(normalized[j, k])
                evidence = first_leg + second_leg
                balance = (
                    2.0 * np.sqrt(first_leg * second_leg)
                    / (evidence + 1e-12)
                )
                affinity = (0.22 + 0.58 * evidence + 0.20 * np.sqrt(
                    first_leg * second_leg
                )) * (0.72 + 0.28 * balance)
                candidates.append((j, float(affinity)))

            support = 0.70 * demand_ik + 0.30 * np.sqrt(demand_ik)
            # Larger demands are more likely to use their direct link; weak
            # demands therefore contribute relatively more transit pressure.
            routed_mass = demand_ik * (
                0.30 + 0.70 * np.sqrt(max(0.0, 1.0 - demand_ik))
            )
            catalog.append((i, k, float(support), float(routed_mass), candidates))

    # A few damped reassignment rounds approximate the competition for
    # transit capacity without requiring a routing solver.
    for _ in range(3):
        transit = np.zeros(n, dtype=float)
        for _, _, _, routed_mass, candidates in catalog:
            ranked = sorted(
                (
                    (affinity / (pressure[j] ** 0.80), j)
                    for j, affinity in candidates
                ),
                reverse=True,
            )[:3]
            if not ranked:
                continue
            strengths = [max(score, 1e-12) ** 1.6 for score, _ in ranked]
            total = sum(strengths)
            for strength, (_, j) in zip(strengths, ranked):
                transit[j] += routed_mass * strength / total

        projected = node_base + 0.45 * transit / max_incident
        projected = np.clip(projected, 0.55, 2.2)
        pressure = 0.50 * pressure + 0.50 * projected

    return catalog, pressure


def _add_route_terms(quadratic, catalog, pressure):
    """Reward completion of a few congestion-adjusted two-hop routes."""
    for i, k, support, _, candidates in catalog:
        ranked = sorted(
            (
                (affinity / (pressure[j] ** 0.80), j)
                for j, affinity in candidates
            ),
            reverse=True,
        )
        if not ranked:
            continue

        selected = ranked[:2]
        if len(ranked) >= 3:
            best = max(ranked[0][0], 1e-12)
            second = max(ranked[1][0], 1e-12)
            third = ranked[2][0]
            # Expand only when the boundary alternatives are genuinely close.
            if third >= 0.88 * second and second >= 0.62 * best:
                selected.append(ranked[2])

        strengths = [max(score, 1e-12) ** 1.6 for score, _ in selected]
        total = sum(strengths)
        if total <= 0.0:
            continue

        shares = [strength / total for strength in strengths]
        pressure_adjustment = sum(
            share / (pressure[j] ** 0.45)
            for share, (_, j) in zip(shares, selected)
        )
        route_budget = 0.92 * support * float(
            np.clip(pressure_adjustment, 0.65, 1.20)
        )

        for share, (_, j) in zip(shares, selected):
            _add_quadratic(
                quadratic,
                (i, j),
                (j, k),
                -route_budget * share,
            )


def _add_branch_diversity_terms(quadratic, normalized):
    """Penalize a few highly redundant pairs of outgoing/incoming branches."""
    n = normalized.shape[0]
    if n < 3:
        return

    mean_direct = float(np.mean(
        0.70 * normalized + 0.30 * np.sqrt(normalized)
    ))
    if mean_direct <= 0.0:
        return

    for inbound in (False, True):
        profiles = np.sqrt(normalized.T if inbound else normalized)
        norms = np.sqrt(np.sum(profiles * profiles, axis=1))
        profiles = profiles / np.maximum(norms[:, None], 1e-12)
        similarity = profiles @ profiles.T

        for center in range(n):
            branches = [node for node in range(n) if node != center]
            pairs = []
            for pos, a in enumerate(branches):
                for b in branches[pos + 1:]:
                    value = float(similarity[a, b])
                    if value >= 0.45:
                        pairs.append((value, a, b))
            pairs.sort(reverse=True)

            # Keep this correction deliberately sparse so it shapes branch
            # choices without overwhelming the route and direct-demand terms.
            for value, a, b in pairs[:2]:
                if inbound:
                    first, second = (a, center), (b, center)
                else:
                    first, second = (center, a), (center, b)
                _add_quadratic(
                    quadratic,
                    first,
                    second,
                    0.35 * mean_direct * value,
                )


def surrogate_terms(demand):
    """Return linear terms, sparse route/diversity couplings, and penalty."""
    traffic, normalized = _clean_demand(demand)
    n = normalized.shape[0]
    roots = np.sqrt(normalized)

    linear = {
        (i, j): -float(0.75 * normalized[i, j] + 0.25 * roots[i, j])
        for i in range(n)
        for j in range(n)
        if i != j
    }
    quadratic = {}

    if not np.any(traffic):
        return linear, quadratic, 1.25

    outgoing = normalized.sum(axis=1)
    incoming = normalized.sum(axis=0)
    incident = np.maximum(outgoing, incoming)

    catalog, pressure = _build_route_catalog(normalized, incident)
    _add_route_terms(quadratic, catalog, pressure)
    _add_branch_diversity_terms(quadratic, normalized)

    pressure_cv = float(
        np.std(incident) / max(float(np.mean(incident)), 1e-12)
    )
    penalty_weight = 1.18 + 0.18 * pressure_cv / (1.0 + pressure_cv)
    penalty_weight = float(np.clip(penalty_weight, 1.18, 1.36))
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
