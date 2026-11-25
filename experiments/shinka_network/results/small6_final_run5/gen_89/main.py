"""Surrogate QUBO terms for QOBLIB 08 network design. ShinkaEvolve edits the EVOLVE block."""

import time

# EVOLVE-BLOCK-START
import numpy as np


class _DemandModel:
    """Shared demand features for independent surrogate-factor stages."""

    def __init__(self, demand):
        traffic = np.asarray(demand, dtype=float)
        if traffic.ndim != 2 or traffic.shape[0] != traffic.shape[1]:
            raise ValueError("demand must be a square matrix")

        traffic = np.nan_to_num(
            traffic, copy=True, nan=0.0, posinf=0.0, neginf=0.0
        )
        np.maximum(traffic, 0.0, out=traffic)
        np.fill_diagonal(traffic, 0.0)

        self.n = traffic.shape[0]
        maximum = float(traffic.max()) if traffic.size else 0.0
        self.t = traffic / maximum if maximum > 0.0 else traffic

        roots = np.sqrt(self.t)
        self.direct = 0.75 * self.t + 0.25 * roots
        self.support = 0.70 * self.t + 0.30 * roots

        self.outgoing = self.t.sum(axis=1)
        self.incoming = self.t.sum(axis=0)
        incident = np.maximum(self.outgoing, self.incoming)
        peak = float(incident.max()) if incident.size else 0.0
        self.pressure = np.maximum(
            0.65, incident / max(peak, 1e-15)
        )

        positive = self.t[self.t > 0.0]
        self.heavy_cutoff = (
            float(np.quantile(positive, 0.68))
            if positive.size else 0.0
        )
        self.mean_direct = float(self.direct.sum()) / max(
            1, self.n * (self.n - 1)
        )
        self.depth_scale = max(
            0.75, float(np.log2(max(self.n, 2)) / np.log2(10.0))
        )


class _QuboBuilder:
    """Accumulate all factors using canonical, additive coupling keys."""

    def __init__(self, model):
        self.n = model.n
        self.linear = -model.direct.copy()
        self.quadratic = {}

    def add_linear(self, link, weight):
        self.linear[link] += float(weight)

    def add_pair(self, first, second, weight):
        weight = float(weight)
        if abs(weight) < 1e-14:
            return
        if first == second:
            self.add_linear(first, weight)
            return
        key = (
            (first, second) if first < second else (second, first)
        )
        self.quadratic[key] = self.quadratic.get(key, 0.0) + weight

    def export(self, penalty_weight):
        linear = {
            (i, j): float(self.linear[i, j])
            for i in range(self.n)
            for j in range(self.n)
            if i != j
        }
        quadratic = {
            key: float(value)
            for key, value in self.quadratic.items()
            if abs(value) > 1e-12
        }
        return linear, quadratic, float(penalty_weight)


def _add_route_factors(model, builder):
    """Reward a bounded collection of useful two-link routes."""
    n = model.n
    t = model.t
    pressure = model.pressure
    pressure_rank = pressure ** 0.75
    route_size_factor = float(np.clip(np.sqrt(n / 12.0), 0.90, 1.15))

    for source in range(n):
        for target in range(n):
            if source == target or t[source, target] <= 0.0:
                continue

            support = float(model.support[source, target])
            candidates = []

            for middle in range(n):
                if middle == source or middle == target:
                    continue

                leg_traffic = float(
                    t[source, middle] + t[middle, target]
                )
                rank = (
                    (0.25 + leg_traffic) / pressure_rank[middle]
                )
                discount = float(
                    pressure[middle] * (1.0 + 0.5 * leg_traffic)
                )
                candidates.append((float(rank), middle, discount))

            candidates.sort(reverse=True)
            keep = min(2, len(candidates))
            # Reserve a third route for heavy demands with near-tied options.
            if (
                t[source, target] >= model.heavy_cutoff
                and len(candidates) > 2
                and candidates[2][0] >= 0.92 * candidates[1][0]
            ):
                keep = 3

            for position, (_, middle, discount) in enumerate(
                candidates[:keep]
            ):
                rank_factor = 1.0 + 0.08 * (keep - position)
                builder.add_pair(
                    (source, middle),
                    (middle, target),
                    -0.46 * route_size_factor * support * rank_factor / discount,
                )


def _service_profiles(model):
    """Demand-conditioned background service, without selecting a topology."""
    affinity = model.direct + 0.08
    np.fill_diagonal(affinity, 0.0)

    # Balance background affinity in both directions. This prevents a
    # high-demand node from being treated as an unlimited-degree hub.
    transition = affinity.copy()
    for _ in range(24):
        transition /= np.maximum(
            transition.sum(axis=1, keepdims=True), 1e-15
        )
        transition /= np.maximum(
            transition.sum(axis=0, keepdims=True), 1e-15
        )

    # Once a neighbor is chosen, service can finish there or continue
    # for one or two more links: up to three hops from the original node.
    return (
        np.eye(model.n)
        + 2.0 * transition
        + 1.4 * (transition @ transition)
    )


def _add_neighbor_factors(model, builder):
    """Compile sparse coverage and load-partition factors at each node."""
    if model.mean_direct <= 0.0 or model.n < 4:
        return

    n = model.n
    profiles = _service_profiles(model)
    pair_budget = max(2, n // 3)
    base_strength = (
        0.24 * model.mean_direct * model.depth_scale
    )

    for reverse in (False, True):
        service = profiles.T if reverse else profiles
        traffic = model.t.T if reverse else model.t
        totals = traffic.sum(axis=1)
        mean_total = float(totals.mean())

        if mean_total <= 1e-15:
            continue

        for node in range(n):
            total = float(totals[node])
            if total <= 1e-15:
                continue

            neighbors = [a for a in range(n) if a != node]
            m = len(neighbors)
            left, right = np.triu_indices(m, 1)
            weights = traffic[node] / total

            neighbor_service = service[neighbors, :]
            first_service = neighbor_service[left]
            second_service = neighbor_service[right]
            combined = np.maximum(
                first_service + second_service, 1e-12
            )

            # Concave coverage discourages spending both links on the
            # same easy destinations. Relative service estimates how
            # unevenly this node's traffic would use the two links.
            coverage_cost = -np.log(combined) @ weights
            load_difference = np.abs(
                ((first_service - second_service) / combined)
                @ weights
            )
            costs = coverage_cost + 0.55 * load_difference

            grand_mean = float(costs.mean())
            spread = float(
                np.sqrt(np.mean((costs - grand_mean) ** 2))
            )
            if spread <= 1e-12:
                continue

            dense_cost = np.zeros((m, m), dtype=float)
            dense_cost[left, right] = costs
            dense_cost[right, left] = costs

            # On a degree-two topology exactly one neighbor pair is
            # active. Decompose its cost into a constant, endpoint main
            # effects, and a residual with zero dense row means.
            row_mean = dense_cost.sum(axis=1) / (m - 1)
            main_effect = (
                (m - 1.0) / (m - 2.0)
            ) * (row_mean - grand_mean)
            residual = (
                costs
                - grand_mean
                - main_effect[left]
                - main_effect[right]
            )

            node_importance = float(np.clip(
                total / mean_total, 0.65, 1.50
            ))
            strength = (
                base_strength
                * node_importance
                / max(0.06, spread)
            )

            def link(neighbor):
                return (
                    (neighbor, node) if reverse
                    else (node, neighbor)
                )

            # Retain all additive preferences; they require no couplings.
            for position, neighbor in enumerate(neighbors):
                builder.add_linear(
                    link(neighbor),
                    strength * float(main_effect[position]),
                )

            residual_rms = float(
                np.sqrt(np.mean(residual * residual))
            )
            if residual_rms <= 1e-12:
                continue

            # Keep only informative positive or negative interactions.
            # Removing the constant avoids introducing a degree penalty.
            ranked = sorted(
                range(len(residual)),
                key=lambda q: (
                    -abs(float(residual[q])),
                    int(left[q]),
                    int(right[q]),
                ),
            )
            for q in ranked[:pair_budget]:
                value = float(residual[q])
                if abs(value) < residual_rms:
                    break
                a = neighbors[int(left[q])]
                b = neighbors[int(right[q])]
                builder.add_pair(
                    link(a), link(b), strength * value
                )


def _add_reciprocal_factors(model, builder):
    """Discourage unproductive immediate backtracking."""
    strength = (
        0.65 * model.mean_direct * model.depth_scale
    )
    if strength <= 0.0:
        return

    for i in range(model.n):
        for j in range(i + 1, model.n):
            mutual = min(
                float(model.t[i, j]), float(model.t[j, i])
            )
            builder.add_pair(
                (i, j),
                (j, i),
                strength * (1.0 - 0.45 * mutual),
            )


def surrogate_terms(demand):
    """Return linear costs, sparse quadratic costs, and penalty strength."""
    model = _DemandModel(demand)
    builder = _QuboBuilder(model)

    _add_route_factors(model, builder)
    _add_neighbor_factors(model, builder)
    _add_reciprocal_factors(model, builder)

    # Degree constraints themselves are supplied exclusively by evaluator.
    penalty_weight = min(
        1.45, 1.30 + 0.008 * max(0, model.n - 10)
    )
    return builder.export(penalty_weight)


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
