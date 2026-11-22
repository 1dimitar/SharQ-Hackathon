"""Surrogate QUBO terms for QOBLIB 08 network design. ShinkaEvolve edits the EVOLVE block."""

import time

# EVOLVE-BLOCK-START
import numpy as np


def _prepare_traffic(demand):
    """Normalize traffic and collect statistics shared by all model stages."""
    traffic = np.maximum(np.asarray(demand, dtype=float), 0.0).copy()
    np.fill_diagonal(traffic, 0.0)

    scale = float(traffic.max()) if traffic.size else 0.0
    if scale > 0.0:
        traffic /= scale

    roots = np.sqrt(traffic)
    incident = np.maximum(traffic.sum(axis=0), traffic.sum(axis=1))
    incident_scale = float(incident.max()) if incident.size else 0.0
    incident /= max(incident_scale, 1e-12)

    positive = traffic[traffic > 0.0]
    cutoff = float(np.quantile(positive, 0.75)) if positive.size else 0.0
    pairs = [
        (i, k)
        for i in range(len(traffic))
        for k in range(len(traffic))
        if i != k and traffic[i, k] > 0.0
    ]
    return traffic, roots, incident, cutoff, pairs, incident_scale


def _route_catalog(traffic, pairs):
    """Cache intermediary nodes and constituent-leg evidence once per OD."""
    n = len(traffic)
    catalog = {}
    for i, k in pairs:
        nodes = np.array(
            [j for j in range(n) if j != i and j != k],
            dtype=int,
        )
        first = traffic[i, nodes]
        second = traffic[nodes, k]
        evidence = first + second

        # Bounded in [0.85, 1.15]. Both legs need demand support to
        # obtain the strongest reward; endpoint demand alone is insufficient.
        gate = 0.85 + 0.30 * np.sqrt(first * second)
        catalog[i, k] = (nodes, evidence, gate)
    return catalog


def _estimate_pressure(traffic, incident, incident_scale, catalog):
    """Estimate transit pressure from the two strongest plausible routes."""
    initial = np.maximum(0.65, incident)
    transit = np.zeros(len(traffic), dtype=float)

    for (i, k), (nodes, evidence, _) in catalog.items():
        candidates = []
        for position, node in enumerate(nodes):
            j = int(node)
            ev = float(evidence[position])
            if ev > 0.02:
                rank = (0.25 + ev) / float(initial[j]) ** 0.75
                candidates.append((rank, j, ev))

        candidates.sort(reverse=True)
        selected = candidates[:2]
        strengths = [
            rank * rank * (0.20 + ev)
            for rank, _, ev in selected
        ]
        total = sum(strengths)
        if total <= 0.0:
            continue

        amount = float(traffic[i, k]) * (
            0.35 + 0.65 * (1.0 - float(traffic[i, k]))
        )
        for (_, j, _), strength in zip(selected, strengths):
            transit[j] += amount * strength / total

    projected = incident + 0.35 * transit / max(incident_scale, 1e-12)
    return np.clip(projected, 0.65, 1.8)


def _add_pair(quadratic, first, second, weight):
    """Accumulate coefficients using one canonical orientation."""
    if second < first:
        first, second = second, first
    key = (first, second)
    quadratic[key] = quadratic.get(key, 0.0) + float(weight)


def _add_route_terms(traffic, roots, cutoff, pressure, catalog, quadratic):
    """Select a compact set of supported routes and a low-pressure fallback."""
    for (i, k), (nodes, evidence, gates) in catalog.items():
        support = float(0.70 * traffic[i, k] + 0.30 * roots[i, k])
        candidates = []

        for position, node in enumerate(nodes):
            j = int(node)
            ev = float(evidence[position])
            p = float(pressure[j])
            rank = (0.25 + ev) / p ** 0.75
            discount = p * (1.0 + 0.5 * ev)
            candidates.append(
                (rank, j, discount, float(gates[position]))
            )

        candidates.sort(reverse=True)
        keep = min(
            len(candidates),
            4 if traffic[i, k] >= cutoff else 2,
        )

        for position, (_, j, discount, gate) in enumerate(candidates[:keep]):
            rank_factor = 1.0 + 0.08 * (keep - position)
            reward = 0.46 * support * rank_factor * gate / discount
            _add_pair(quadratic, (i, j), (j, k), -reward)

        remaining = candidates[keep:]
        if remaining:
            _, j, discount, gate = min(
                remaining, key=lambda item: (item[2], item[1])
            )
            reward = 0.30 * support * gate / discount
            _add_pair(quadratic, (i, j), (j, k), -reward)


def _add_three_hop_terms(traffic, roots, pressure, catalog, quadratic):
    """Add one lightly weighted three-hop option for difficult OD pairs."""
    n = len(traffic)

    for (i, k), (nodes, evidence, _) in catalog.items():
        two_hop = []
        for position, node in enumerate(nodes):
            j = int(node)
            ev = float(evidence[position])
            rank = (0.25 + ev) / float(pressure[j]) ** 0.75
            two_hop.append((rank, ev))

        if not two_hop:
            continue
        two_hop.sort(reverse=True)
        weak = two_hop[0][1] < 0.45
        close = (
            len(two_hop) > 1
            and two_hop[1][0] >= 0.88 * two_hop[0][0]
        )
        if not (weak or close):
            continue

        best_path = None
        for j in range(n):
            if j == i or j == k:
                continue
            first_leg = float(traffic[i, j])
            if first_leg <= 0.0:
                continue

            for h in range(n):
                if h == i or h == k or h == j:
                    continue
                middle_leg = float(traffic[j, h])
                last_leg = float(traffic[h, k])
                if middle_leg <= 0.0 or last_leg <= 0.0:
                    continue

                confidence = (first_leg * middle_leg * last_leg) ** (1.0 / 3.0)
                discount = (
                    float(pressure[j]) * float(pressure[h])
                ) ** 0.5
                rank = confidence / discount
                if best_path is None or rank > best_path[0]:
                    best_path = (rank, j, h, confidence, discount)

        if best_path is None:
            continue

        _, j, h, confidence, discount = best_path
        support = float(0.70 * traffic[i, k] + 0.30 * roots[i, k])
        reward = 0.16 * support * confidence / discount
        _add_pair(quadratic, (i, j), (j, h), -0.5 * reward)
        _add_pair(quadratic, (j, h), (h, k), -0.5 * reward)


def _correct_route_exposure(direct, linear, quadratic):
    """Partially remove unusually large background route attraction."""
    if not quadratic:
        return

    # Approximate degree-balanced link marginals without discrete search.
    propensity = np.exp(2.0 * direct)
    np.fill_diagonal(propensity, 0.0)
    for _ in range(20):
        propensity *= (
            2.0 / np.maximum(propensity.sum(axis=1), 1e-12)
        )[:, None]
        propensity *= (
            2.0 / np.maximum(propensity.sum(axis=0), 1e-12)
        )[None, :]
    np.minimum(propensity, 1.0, out=propensity)

    exposure = np.zeros_like(direct)
    for (first, second), weight in quadratic.items():
        if weight < 0.0:
            exposure[first] -= weight * float(propensity[second])
            exposure[second] -= weight * float(propensity[first])

    positive = exposure[exposure > 0.0]
    if not positive.size:
        return

    threshold = float(np.quantile(positive, 0.75))
    for link in linear:
        excess = max(0.0, float(exposure[link]) - threshold)
        linear[link] += min(0.18, 0.25 * excess)


def _add_branch_diversity(traffic, quadratic):
    """Discourage a few strongly redundant incoming/outgoing branch pairs."""
    n = len(traffic)
    for inbound in (False, True):
        od = traffic.T if inbound else traffic
        profiles = np.sqrt(od)
        np.fill_diagonal(profiles, 1.0)

        for center in range(n):
            weights = od[center]
            if float(weights.sum()) <= 0.0:
                continue

            gram = (profiles * weights[None, :]) @ profiles.T
            branches = [j for j in range(n) if j != center]
            normalizer = max(float(gram[j, j]) for j in branches)
            if normalizer <= 1e-12:
                continue

            redundant = []
            for position, j in enumerate(branches):
                for k in branches[position + 1:]:
                    overlap = float(gram[j, k]) / normalizer
                    if overlap > 0.15:
                        redundant.append((overlap, j, k))
            redundant.sort(reverse=True)

            for overlap, j, k in redundant[:3]:
                first = (j, center) if inbound else (center, j)
                second = (k, center) if inbound else (center, k)
                _add_pair(quadratic, first, second, 0.24 * overlap)


def surrogate_terms(demand):
    """Return sparse cost coefficients and an external degree-penalty weight."""
    traffic, roots, incident, cutoff, pairs, incident_scale = (
        _prepare_traffic(demand)
    )
    n = len(traffic)
    direct = 0.75 * traffic + 0.25 * roots
    linear = {
        (i, j): -float(direct[i, j])
        for i in range(n)
        for j in range(n)
        if i != j
    }
    quadratic = {}

    if not pairs:
        return linear, quadratic, 1.25

    catalog = _route_catalog(traffic, pairs)
    pressure = _estimate_pressure(
        traffic, incident, incident_scale, catalog
    )
    _add_route_terms(
        traffic, roots, cutoff, pressure, catalog, quadratic
    )
    _add_three_hop_terms(traffic, roots, pressure, catalog, quadratic)
    _correct_route_exposure(direct, linear, quadratic)
    _add_branch_diversity(traffic, quadratic)

    return linear, quadratic, 1.25


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
