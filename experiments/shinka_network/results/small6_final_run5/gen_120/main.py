"""Surrogate QUBO terms for QOBLIB 08 network design. ShinkaEvolve edits the EVOLVE block."""

import time

# EVOLVE-BLOCK-START
import numpy as np


def surrogate_terms(demand):
    """Build direct-link and selectively pressure-discounted two-hop terms."""
    matrix = np.asarray(demand, dtype=float)
    n = len(matrix)

    traffic = np.maximum(matrix, 0.0).copy()
    np.fill_diagonal(traffic, 0.0)
    scale = float(np.max(traffic)) if traffic.size else 0.0
    if scale <= 0.0:
        scale = 1.0

    normalized = traffic / scale
    roots = np.sqrt(normalized)
    linear = {
        (i, j): -float(0.75 * normalized[i, j] + 0.25 * roots[i, j])
        for i in range(n)
        for j in range(n)
        if i != j
    }

    incident = np.maximum(traffic.sum(axis=0), traffic.sum(axis=1))
    max_incident = float(np.max(incident)) if incident.size else 0.0
    if max_incident <= 0.0:
        max_incident = 1.0

    # Estimate transit demand on plausible two-hop routes, using incident
    # load as an initial congestion estimate. Keep only the two best routes
    # per OD pair to avoid spreading load over unsupported intermediates.
    initial_pressure = np.maximum(0.65, incident / max_incident)
    transit_load = np.zeros(n, dtype=float)
    for i in range(n):
        for k in range(n):
            if i == k or traffic[i, k] <= 0.0:
                continue
            routes = []
            for j in range(n):
                if j == i or j == k:
                    continue
                evidence = float(normalized[i, j] + normalized[j, k])
                if evidence <= 0.02:
                    continue
                rank = (0.25 + evidence) / (float(initial_pressure[j]) ** 0.75)
                routes.append((rank, j, evidence))
            routes.sort(reverse=True)
            selected = routes[:2]
            if not selected:
                continue
            strengths = [
                rank * rank * (0.20 + evidence)
                for rank, _, evidence in selected
            ]
            total_strength = sum(strengths)
            routed_fraction = 0.35 + 0.65 * (1.0 - float(normalized[i, k]))
            for (_, j, _), strength in zip(selected, strengths):
                transit_load[j] += (
                    float(traffic[i, k]) * routed_fraction * strength / total_strength
                )

    projected_pressure = incident / max_incident + 0.35 * transit_load / max_incident
    node_pressure = np.maximum(0.65, np.minimum(1.8, projected_pressure))

    positive = traffic[traffic > 0.0]
    cutoff = float(np.quantile(positive, 0.75)) if positive.size else 0.0
    high_cutoff = float(np.quantile(positive, 0.90)) if positive.size else 0.0
    quadratic = {}

    for i in range(n):
        for k in range(n):
            if i == k or traffic[i, k] <= 0.0:
                continue

            support = 0.70 * normalized[i, k] + 0.30 * roots[i, k]
            candidates = []
            for j in range(n):
                if j == i or j == k:
                    continue
                edge_pressure = normalized[i, j] + normalized[j, k]
                pressure = float(node_pressure[j])
                affinity = 0.25 + edge_pressure
                rank = affinity / (pressure ** 0.75)
                discount = pressure * (1.0 + 0.5 * edge_pressure)
                candidates.append((float(rank), j, float(discount)))

            candidates.sort(reverse=True)
            keep = min(len(candidates), 4 if traffic[i, k] >= cutoff else 2)
            # Add one alternative only when the route ranking is genuinely
            # ambiguous and the candidate has evidence on both of its legs.
            if keep < len(candidates):
                frontier = candidates[keep - 1][0]
                next_rank, next_j, _ = candidates[keep]
                next_evidence = float(normalized[i, next_j] + normalized[next_j, k])
                best_rank = max(candidates[0][0], 1e-12)
                if (next_rank / max(frontier, 1e-12) >= 0.92
                        and next_rank / best_rank >= 0.65
                        and next_evidence >= 0.05):
                    keep += 1
            selected = set()

            for position, (_, j, discount) in enumerate(candidates[:keep]):
                selected.add(j)
                rank_factor = 1.0 + 0.08 * (keep - position)
                weight = -0.46 * float(support) * rank_factor / discount
                first, second = (i, j), (j, k)
                key = (first, second) if first < second else (second, first)
                quadratic[key] = weight

            # Retain one alternative selected by the full-path model's
            # pressure discount, unless it was already selected by affinity.
            remaining = [candidate for candidate in candidates if candidate[1] not in selected]
            if remaining:
                _, j, discount = min(remaining, key=lambda candidate: (candidate[2], candidate[1]))
                first, second = (i, j), (j, k)
                key = (first, second) if first < second else (second, first)
                quadratic[key] = -0.30 * float(support) / discount

            # Three-hop Taylor terms are omitted: pairwise fragments can
            # reward incomplete paths and misrank valid two-link exchanges.

    # Correct excessive mean-field attraction from overlapping route rewards.
    # Use soft degree-balanced link marginals rather than assuming that every
    # potential partner is equally likely to appear in a sampled topology.
    if quadratic:
        propensity = np.exp(
            2.0 * (0.75 * normalized + 0.25 * roots)
        )
        np.fill_diagonal(propensity, 0.0)
        for _ in range(20):
            propensity *= (
                2.0 / np.maximum(propensity.sum(axis=1), 1e-12)
            )[:, None]
            propensity *= (
                2.0 / np.maximum(propensity.sum(axis=0), 1e-12)
            )[None, :]
        # Fractional scaling need not respect individual binary upper bounds.
        np.minimum(propensity, 1.0, out=propensity)

        exposure = np.zeros((n, n), dtype=float)
        for (first, second), weight in quadratic.items():
            if weight < 0.0:
                exposure[first] -= weight * float(propensity[second])
                exposure[second] -= weight * float(propensity[first])

        active_exposure = exposure[exposure > 0.0]
        if active_exposure.size:
            threshold = float(np.quantile(active_exposure, 0.80))
            for link in linear:
                excess = max(0.0, float(exposure[link]) - threshold)
                # Apply a smaller correction only to the upper tail, preserving
                # more of the route-completion signal in ordinary cases.
                linear[link] += min(0.18, 0.12 * excess)

    # Score pairs by coverage and estimated first-link load balance.
    # Compress each complete pair table into unary effects plus a sparse
    # residual, exploiting the fact that exactly two branches are selected.
    base_strength = 0.24
    pair_budget = max(2, n // 3)

    for inbound in (False, True):
        od = normalized.T if inbound else normalized
        profiles = np.sqrt(od).copy()
        np.fill_diagonal(profiles, 1.0)

        for center in range(n):
            weights = od[center]
            total = float(weights.sum())
            if total <= 0.0:
                continue
            weights = weights / total

            branches = [j for j in range(n) if j != center]
            m = len(branches)
            service = profiles[branches]
            coverage = service @ weights
            gram = (service * weights[None, :]) @ service.T
            normalizer = max(0.20, float(np.max(np.diag(gram))))

            # Independent-service proxy: neither branch serves a target.
            pair_cost = (
                1.0 - coverage[:, None] - coverage[None, :] + gram
            )

            # Soft routing estimates the demand assigned to each branch.
            # Unsupported destinations split equally; direct destinations
            # use their direct branch rather than an indirect alternative.
            imbalance = np.zeros((m, m), dtype=float)
            branch_rows = np.arange(m)
            for a in range(m):
                shares = (service[a][None, :] + 1e-8) / (
                    service[a][None, :] + service + 2e-8
                )
                shares[:, branches[a]] = 1.0
                shares[branch_rows, branches] = 0.0
                assigned = shares @ weights
                imbalance[a] = (2.0 * assigned - 1.0) ** 2

            pair_cost += 0.15 * (imbalance + imbalance.T) * 0.5
            np.fill_diagonal(pair_cost, 0.0)

            # Fit C_ab = u_a + u_b + R_ab with zero row sums in R.
            # With two selected branches, the additive part is exactly
            # sum_a u_a*x_a, so it needs no quadratic couplings.
            row_totals = pair_cost.sum(axis=1)
            offset_total = float(row_totals.sum()) / (2.0 * (m - 1))
            unary = (row_totals - offset_total) / (m - 2)
            residual = pair_cost - unary[:, None] - unary[None, :]
            np.fill_diagonal(residual, 0.0)

            # This shift is constant on valid topologies and avoids adding
            # a large uniform link bias to the sampling objective.
            unary = unary - float(unary.mean())
            gain = base_strength / normalizer

            for a, node in enumerate(branches):
                link = (node, center) if inbound else (center, node)
                linear[link] += float(gain * unary[a])

            interactions = []
            for a in range(m):
                for b in range(a + 1, m):
                    value = float(residual[a, b])
                    if abs(value) > 1e-12:
                        interactions.append((abs(value), a, b, value))
            interactions.sort(reverse=True)

            for _, a, b, value in interactions[:pair_budget]:
                j, k = branches[a], branches[b]
                first = (j, center) if inbound else (center, j)
                second = (k, center) if inbound else (center, k)
                key = (first, second) if first < second else (second, first)
                quadratic[key] = quadratic.get(key, 0.0) + gain * value

    pressure_cv = float(np.std(incident) / max(float(np.mean(incident)), 1e-12))
    penalty_weight = 1.15 + 0.20 * pressure_cv / (1.0 + pressure_cv)
    return linear, quadratic, float(min(1.35, max(1.15, penalty_weight)))


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
