"""Surrogate QUBO terms for QOBLIB 08 network design. ShinkaEvolve edits the EVOLVE block."""

import time

# EVOLVE-BLOCK-START
import numpy as np


def _diffusion_distances(traffic):
    """Estimate continuation distances at several walk lengths."""
    n = len(traffic)

    affinity = np.exp(2.4 * traffic)
    np.fill_diagonal(affinity, 0.0)

    # Fractional degree-balanced reference graph. This is not a topology
    # optimisation: it supplies a deterministic multihop similarity metric.
    for _ in range(40):
        affinity /= np.maximum(affinity.sum(axis=1), 1e-15)[:, None]
        affinity /= np.maximum(affinity.sum(axis=0), 1e-15)[None, :]

    transition = affinity / np.maximum(
        affinity.sum(axis=1), 1e-15
    )[:, None]

    horizon = max(3, int(np.ceil(np.log2(max(n, 2)))) + 1)
    walk = transition.copy()
    reach = np.zeros_like(traffic)
    weight = 1.0
    weight_sum = 0.0

    for _ in range(horizon):
        reach += weight * walk
        weight_sum += weight
        walk = walk @ transition
        weight *= 0.48

    reach /= weight_sum

    off_diagonal = ~np.eye(n, dtype=bool)
    reference = max(float(reach[off_diagonal].mean()), 1e-15)

    # A branch is typically several hops from an arbitrary destination.
    # Larger networks therefore have a larger continuation-cost baseline.
    baseline = max(1.35, np.log2(float(n)) - 0.65)
    distance = baseline - 0.80 * np.log(
        np.maximum(reach / reference, 1e-12)
    )
    distance = np.clip(distance, 0.70, baseline + 2.0)
    np.fill_diagonal(distance, 0.0)
    return distance, baseline


def _balanced_pair_costs(weights, distance, left, right, baseline):
    """Evaluate every pair of facilities using a balanced soft assignment."""
    total = float(weights.sum())
    if total <= 1e-15:
        return np.zeros(len(left), dtype=float)

    first_cost = distance[left]
    second_cost = distance[right]
    difference = first_cost - second_cost

    temperature = 0.14 * baseline
    balance_strength = 0.22 * baseline

    # For p_d routed through the first branch, the objective is
    #
    # sum_d w_d [p_d*c1_d + (1-p_d)*c2_d + temperature*entropy(p_d)]
    #       + balance_strength * (load1-load2)^2 / total.
    #
    # The optimal assignments are logistic functions of one common offset.
    # Solve that scalar consistency condition for all pairs simultaneously.
    lower = np.full(len(left), -4.0 * balance_strength)
    upper = np.full(len(left), 4.0 * balance_strength)

    for _ in range(15):
        offset = 0.5 * (lower + upper)
        logits = np.clip(
            (difference + offset[:, None]) / temperature,
            -45.0,
            45.0,
        )
        assignment = 1.0 / (1.0 + np.exp(logits))
        imbalance = (2.0 * assignment - 1.0) @ weights
        residual = offset - 4.0 * balance_strength * imbalance / total

        above = residual > 0.0
        upper = np.where(above, offset, upper)
        lower = np.where(above, lower, offset)

    offset = 0.5 * (lower + upper)
    logits = np.clip(
        (difference + offset[:, None]) / temperature,
        -45.0,
        45.0,
    )
    assignment = 1.0 / (1.0 + np.exp(logits))
    complement = 1.0 - assignment

    entropy = (
        assignment * np.log(np.maximum(assignment, 1e-15))
        + complement * np.log(np.maximum(complement, 1e-15))
    )
    routing_cost = (
        assignment * first_cost
        + complement * second_cost
        + temperature * entropy
    ) @ weights

    imbalance = (2.0 * assignment - 1.0) @ weights
    routing_cost += balance_strength * imbalance * imbalance / total
    return routing_cost / baseline


def _add_quadratic(quadratic, first, second, weight):
    if second < first:
        first, second = second, first
    key = (first, second)
    quadratic[key] = quadratic.get(key, 0.0) + float(weight)


def _facility_orientation(traffic, distance, baseline, inbound,
                          linear, quadratic):
    """Build complementary branch costs for one traffic orientation."""
    n = len(traffic)
    totals = traffic.sum(axis=1)
    mean_total = max(float(totals.mean()), 1e-15)

    for center in range(n):
        if totals[center] <= 1e-15:
            continue

        branches = np.array(
            [j for j in range(n) if j != center], dtype=int
        )
        m = len(branches)
        a, b = np.triu_indices(m, 1)

        costs = _balanced_pair_costs(
            traffic[center],
            distance,
            branches[a],
            branches[b],
            baseline,
        )

        # Both orientations together recover approximately one direct-link
        # reward, with additional multihop and branch-complementarity effects.
        pressure = float(np.clip(
            (totals[center] / mean_total) ** 0.30, 0.70, 1.50
        ))
        costs *= 0.5 * pressure

        pair_matrix = np.zeros((m, m), dtype=float)
        pair_matrix[a, b] = costs
        pair_matrix[b, a] = costs

        # Exact decomposition on degree-two choices:
        # cost(j,k) = main(j) + main(k) + interaction(j,k).
        # The full interaction matrix has zero off-diagonal row sums.
        row_sums = pair_matrix.sum(axis=1)
        sum_main = float(costs.sum()) / (m - 1)
        main = (row_sums - sum_main) / (m - 2)
        interaction = pair_matrix - main[:, None] - main[None, :]
        np.fill_diagonal(interaction, 0.0)

        # A common main-effect shift is constant on valid degree-two rows.
        main -= float(main.mean())

        # Keep a compact, symmetric graph of the strongest pair effects.
        # This gives O(n^2) retained terms for a fixed neighbour budget.
        keep = min(4, m - 1)
        selected = np.zeros((m, m), dtype=bool)
        for position in range(m):
            ranking = np.argsort(
                -np.abs(interaction[position]), kind="stable"
            )
            accepted = 0
            for other in ranking:
                other = int(other)
                if other == position:
                    continue
                selected[position, other] = True
                accepted += 1
                if accepted == keep:
                    break

        selected |= selected.T
        np.fill_diagonal(selected, False)
        retained = np.where(selected, interaction, 0.0)

        # Full interactions have zero background field. Restore that property
        # approximately after dropping weak interactions.
        marginal = 2.0 / m
        main -= marginal * retained.sum(axis=1)

        links = []
        for position, node in enumerate(branches):
            node = int(node)
            link = (node, center) if inbound else (center, node)
            links.append(link)
            linear[link] += float(main[position])

        for position in range(m):
            for other in range(position + 1, m):
                weight = float(retained[position, other])
                if abs(weight) > 1e-12:
                    _add_quadratic(
                        quadratic, links[position], links[other], weight
                    )


def surrogate_terms(demand):
    """Return cost-only QUBO coefficients and an external degree penalty."""
    traffic = np.asarray(demand, dtype=float).copy()
    traffic = np.nan_to_num(
        traffic, nan=0.0, posinf=0.0, neginf=0.0
    )
    traffic = np.maximum(traffic, 0.0)
    np.fill_diagonal(traffic, 0.0)
    n = len(traffic)

    linear = {
        (i, j): 0.0
        for i in range(n)
        for j in range(n)
        if i != j
    }
    quadratic = {}

    maximum = float(traffic.max()) if traffic.size else 0.0
    if maximum <= 0.0:
        return linear, quadratic, 1.45

    traffic /= maximum

    # Retain sensitivity to medium-sized demands without allowing tiny
    # entries to dominate aggregate endpoint traffic.
    effective = 0.85 * traffic + 0.15 * np.sqrt(traffic)
    np.fill_diagonal(effective, 0.0)

    distance, baseline = _diffusion_distances(traffic)
    reverse_distance, reverse_baseline = _diffusion_distances(traffic.T)

    _facility_orientation(
        effective, distance, baseline, False, linear, quadratic
    )
    _facility_orientation(
        effective.T, reverse_distance, reverse_baseline,
        True, linear, quadratic
    )

    # Reciprocal arcs consume two ports while returning to an already
    # reachable endpoint. Keep them possible when both direct demands
    # are substantial, but mildly discourage gratuitous two-cycles.
    reciprocal_strength = 0.13 + 0.025 * np.log2(n / 5.0)
    for i in range(n):
        for j in range(i + 1, n):
            importance = 0.5 * (traffic[i, j] + traffic[j, i])
            weight = reciprocal_strength * (1.0 - 0.70 * importance)
            _add_quadratic(quadratic, (i, j), (j, i), weight)

    # Explicit normalization makes the intended penalty scale unambiguous;
    # the evaluator's subsequent normalization is then effectively a no-op.
    largest = max(
        max((abs(v) for v in linear.values()), default=0.0),
        max((abs(v) for v in quadratic.values()), default=0.0),
    )
    if largest > 1e-15:
        inverse = 1.0 / largest
        linear = {link: float(v * inverse) for link, v in linear.items()}
        quadratic = {
            pair: float(v * inverse) for pair, v in quadratic.items()
        }

    # No degree penalties are included in the returned coefficients.
    return linear, quadratic, 1.45


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
