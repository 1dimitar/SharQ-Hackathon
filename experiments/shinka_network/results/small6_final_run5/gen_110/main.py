"""Surrogate QUBO terms for QOBLIB 08 network design. ShinkaEvolve edits the EVOLVE block."""

import time

# EVOLVE-BLOCK-START
import numpy as np


def surrogate_terms(demand):
    """Build direct-link rewards, sparse route terms, and congestion cues."""
    matrix = np.asarray(demand, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("demand must be a square matrix")

    n = matrix.shape[0]
    traffic = np.nan_to_num(
        matrix, copy=True, nan=0.0, posinf=0.0, neginf=0.0
    )
    np.maximum(traffic, 0.0, out=traffic)
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
    quadratic = {}

    def add_pair(first, second, weight):
        key = (first, second) if first < second else (second, first)
        quadratic[key] = quadratic.get(key, 0.0) + float(weight)

    incident = np.maximum(traffic.sum(axis=0), traffic.sum(axis=1))
    max_incident = float(np.max(incident)) if incident.size else 0.0
    if max_incident <= 0.0:
        max_incident = 1.0

    # Estimate transit pressure by distributing non-direct traffic over a
    # small number of plausible two-hop routes.
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
                rank = (0.25 + evidence) / float(initial_pressure[j] ** 0.75)
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
            if total_strength <= 0.0:
                continue

            routed_fraction = 0.35 + 0.65 * (1.0 - float(normalized[i, k]))
            for (_, j, _), strength in zip(selected, strengths):
                transit_load[j] += (
                    float(traffic[i, k])
                    * routed_fraction
                    * strength
                    / total_strength
                )

    projected_pressure = (
        incident / max_incident + 0.35 * transit_load / max_incident
    )
    node_pressure = np.maximum(0.65, np.minimum(1.8, projected_pressure))

    positive = traffic[traffic > 0.0]
    cutoff = float(np.quantile(positive, 0.75)) if positive.size else 0.0

    # Reward a compact set of promising two-hop paths. Larger OD demands
    # receive a few more route alternatives; transit nodes are discounted
    # according to their estimated pressure.
    for i in range(n):
        for k in range(n):
            if i == k or traffic[i, k] <= 0.0:
                continue

            support = float(
                0.70 * normalized[i, k] + 0.30 * roots[i, k]
            )
            candidates = []
            for j in range(n):
                if j == i or j == k:
                    continue
                edge_pressure = float(normalized[i, j] + normalized[j, k])
                pressure = float(node_pressure[j])
                rank = (0.25 + edge_pressure) / (pressure ** 0.75)
                discount = pressure * (1.0 + 0.5 * edge_pressure)
                candidates.append((float(rank), j, float(discount)))

            candidates.sort(reverse=True)
            keep = min(len(candidates), 4 if traffic[i, k] >= cutoff else 2)

            # Include one more route only when the selection boundary is
            # ambiguous and the candidate has some evidence on both legs.
            if keep > 0 and keep < len(candidates):
                frontier = candidates[keep - 1][0]
                next_rank, next_j, _ = candidates[keep]
                next_evidence = float(
                    normalized[i, next_j] + normalized[next_j, k]
                )
                best_rank = max(candidates[0][0], 1e-12)
                if (
                    next_rank / max(frontier, 1e-12) >= 0.92
                    and next_rank / best_rank >= 0.65
                    and next_evidence >= 0.05
                ):
                    keep += 1

            selected_nodes = set()
            for position, (_, j, discount) in enumerate(candidates[:keep]):
                selected_nodes.add(j)
                rank_factor = 1.0 + 0.08 * (keep - position)
                first, second = (i, j), (j, k)
                key = (first, second) if first < second else (second, first)
                # A two-link path is uniquely identified by its two directed
                # links, so assignment avoids accidental duplicate rewards.
                quadratic[key] = (
                    -0.46 * support * rank_factor / max(discount, 1e-12)
                )

            # Keep a distinct low-pressure fallback route where possible.
            remaining = [
                route for route in candidates
                if route[1] not in selected_nodes
            ]
            if remaining:
                _, j, discount = min(
                    remaining, key=lambda route: (route[2], route[1])
                )
                first, second = (i, j), (j, k)
                key = (first, second) if first < second else (second, first)
                quadratic[key] = -0.30 * support / max(discount, 1e-12)

    # Reduce excessive mean-field attraction from links that participate in
    # many negative route interactions.
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

        np.minimum(propensity, 1.0, out=propensity)
        exposure = np.zeros((n, n), dtype=float)

        for (first, second), weight in quadratic.items():
            if weight < 0.0:
                exposure[first] -= weight * float(propensity[second])
                exposure[second] -= weight * float(propensity[first])

        active_exposure = exposure[exposure > 0.0]
        if active_exposure.size:
            threshold = float(np.quantile(active_exposure, 0.75))
            for link in linear:
                excess = max(0.0, float(exposure[link]) - threshold)
                linear[link] += min(0.18, 0.25 * excess)

    # Penalize a few pairs of same-direction branches with highly overlapping
    # demand profiles, for outbound and inbound traffic respectively.
    for inbound in (False, True):
        od = normalized.T if inbound else normalized
        profiles = np.sqrt(od).copy()
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
            for pos, j in enumerate(branches):
                for k in branches[pos + 1:]:
                    overlap = float(gram[j, k]) / normalizer
                    if overlap > 0.15:
                        redundant.append((overlap, j, k))

            redundant.sort(reverse=True)
            for overlap, j, k in redundant[:3]:
                first = (j, center) if inbound else (center, j)
                second = (k, center) if inbound else (center, k)
                add_pair(first, second, 0.24 * overlap)

    # Add a modest two-node cut congestion cue. The endpoint corrections
    # keep this signal from acting only as an unconditional reciprocal-link
    # penalty on degree-balanced topologies.
    if n > 2:
        outgoing = normalized.sum(axis=1)
        incoming = normalized.sum(axis=0)
        incident_load = np.maximum(outgoing, incoming)

        external = np.maximum(
            outgoing[:, None] + outgoing[None, :],
            incoming[:, None] + incoming[None, :],
        ) - normalized - normalized.T
        external = np.maximum(0.0, external)

        incident_pair = incident_load[:, None] + incident_load[None, :]
        vulnerability = external / np.maximum(incident_pair, 1e-12)
        off_diagonal = ~np.eye(n, dtype=bool)
        active_vulnerability = vulnerability[
            off_diagonal & (external > 0.0)
        ]

        if active_vulnerability.size:
            vulnerability /= max(
                float(np.mean(active_vulnerability)), 1e-12
            )
            vulnerability = np.clip(vulnerability, 0.75, 1.25)

            singleton_bound = 0.5 * np.maximum(
                incident_load[:, None], incident_load[None, :]
            )
            one_internal = np.maximum(
                0.0, external / 3.0 - singleton_bound
            )
            two_internal = np.maximum(
                0.0, external / 2.0 - singleton_bound
            )
            cut_linear = vulnerability * one_internal
            cut_pair = vulnerability * np.maximum(
                0.0, two_internal - 2.0 * one_internal
            )
            np.fill_diagonal(cut_linear, 0.0)
            np.fill_diagonal(cut_pair, 0.0)

            mean_cut = float(cut_pair.sum()) / max(1, n * (n - 1))
            mean_direct = -float(sum(linear.values())) / max(
                1, n * (n - 1)
            )

            # Weaker than the original cut signal so path rewards remain
            # primary and the added reciprocal couplings stay well-scaled.
            if mean_cut > 1e-12 and mean_direct > 0.0:
                cut_gain = 0.10 * mean_direct / mean_cut
                row_sum = cut_linear.sum(axis=1)
                offset_total = float(row_sum.sum()) / (2.0 * (n - 1))
                offsets = (row_sum - offset_total) / (n - 2)

                for i in range(n):
                    for j in range(i + 1, n):
                        add_pair(
                            (i, j),
                            (j, i),
                            cut_gain * float(cut_pair[i, j]),
                        )
                        correction = cut_gain * (
                            float(cut_linear[i, j])
                            - float(offsets[i])
                            - float(offsets[j])
                        )
                        linear[(i, j)] += float(correction)
                        linear[(j, i)] += float(correction)

    pressure_cv = float(
        np.std(incident) / max(float(np.mean(incident)), 1e-12)
    )
    penalty_weight = 1.15 + 0.20 * pressure_cv / (1.0 + pressure_cv)
    penalty_weight = float(min(1.35, max(1.15, penalty_weight)))
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
