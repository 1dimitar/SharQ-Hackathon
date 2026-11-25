"""Surrogate QUBO terms for QOBLIB 08 network design. ShinkaEvolve edits the EVOLVE block."""

import time

# EVOLVE-BLOCK-START
import numpy as np


def _prepare_demand(demand):
    """Sanitize and scale traffic, then compute reusable node statistics."""
    traffic = np.asarray(demand, dtype=float)
    if traffic.ndim != 2 or traffic.shape[0] != traffic.shape[1]:
        raise ValueError("demand must be a square matrix")

    traffic = np.nan_to_num(
        traffic, copy=True, nan=0.0, posinf=0.0, neginf=0.0
    )
    np.maximum(traffic, 0.0, out=traffic)
    np.fill_diagonal(traffic, 0.0)

    n = int(traffic.shape[0])
    maximum = float(traffic.max()) if traffic.size else 0.0
    z = traffic / (maximum if maximum > 0.0 else 1.0)

    row_sum = z.sum(axis=1)
    col_sum = z.sum(axis=0)
    row_share = z / np.maximum(row_sum[:, None], 1e-12)
    col_share = z / np.maximum(col_sum[None, :], 1e-12)

    incident = np.maximum(row_sum, col_sum)
    peak = float(incident.max()) if incident.size else 0.0
    pressure = (
        0.48 + 0.72 * incident / peak
        if peak > 0.0
        else np.ones(n, dtype=float)
    )

    return {
        "n": n,
        "maximum": maximum,
        "z": z,
        "root": np.sqrt(z),
        "row_sum": row_sum,
        "col_sum": col_sum,
        "row_share": row_share,
        "col_share": col_share,
        "pressure": pressure,
        "peak_incident": peak,
    }


def _direct_costs(model):
    """Create direct-link rewards, blending volume and endpoint shares."""
    z = model["z"]
    root = model["root"]
    row_share = model["row_share"]
    col_share = model["col_share"]
    n = model["n"]

    share = np.sqrt(np.maximum(row_share * col_share, 0.0))
    score = (
        0.62 * np.power(z, 0.78)
        + 0.24 * root
        + 0.14 * share
    )

    # A dense array simplifies later cost adjustments; only off-diagonal
    # entries are exported as link variables.
    return -score


class _TermAccumulator:
    """Combine linear and quadratic contributions with canonical keys."""

    def __init__(self, linear_array):
        self.n = int(linear_array.shape[0])
        self.linear = linear_array.copy()
        self.quadratic = {}

    def add_pair(self, first, second, value):
        value = float(value)
        if abs(value) < 1e-14:
            return
        if first == second:
            self.linear[first] += value
            return

        key = (first, second) if first < second else (second, first)
        self.quadratic[key] = self.quadratic.get(key, 0.0) + value

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


def _compile_routes(model, terms):
    """Add sparse two-hop rewards supported by both route legs."""
    n = model["n"]
    z = model["z"]
    root = model["root"]
    row_share = model["row_share"]
    col_share = model["col_share"]
    pressure = model["pressure"]

    positive = z[z > 0.0]
    if positive.size == 0:
        return
    demand_cutoff = float(np.quantile(positive, 0.70))

    # Endpoint-aware affinities are precomputed once and reused for every
    # source-target pair.
    first_affinity = 0.58 * np.sqrt(row_share) + 0.42 * root
    second_affinity = 0.58 * np.sqrt(col_share) + 0.42 * root

    rank_pressure = pressure ** 0.88
    reward_pressure = pressure ** 0.35
    size_factor = float(np.clip(np.sqrt(n / 12.0), 0.85, 1.42))

    for source in range(n):
        for target in range(n):
            amount = float(z[source, target])
            if source == target or amount <= 0.0:
                continue

            demand_support = (
                0.68 * amount ** 0.74 + 0.32 * float(root[source, target])
            )
            demand_support *= 1.0 - 0.30 * float(root[source, target])

            candidates = []
            for middle in range(n):
                if middle == source or middle == target:
                    continue

                a = float(np.sqrt(
                    0.58 * z[source, middle]
                    + 0.42 * row_share[source, middle]
                ))
                b = float(np.sqrt(
                    0.58 * z[middle, target]
                    + 0.42 * col_share[middle, target]
                ))

                # The harmonic mean ensures that a poorly supported leg
                # cannot be hidden by a strong one.
                bilateral = 2.0 * a * b / max(a + b, 1e-12)
                affinity = float(np.sqrt(
                    max(
                        first_affinity[source, middle]
                        * second_affinity[middle, target],
                        0.0,
                    )
                ))

                rank = (
                    0.16 + affinity + 0.22 * bilateral
                ) / float(rank_pressure[middle])
                candidates.append(
                    (float(rank), middle, affinity, bilateral)
                )

            candidates.sort(key=lambda item: (-item[0], item[1]))
            keep = min(2, len(candidates))

            # Keep a third alternative only for a substantial demand whose
            # route ranking is close; this avoids spreading weak rewards.
            if (
                len(candidates) >= 3
                and amount >= demand_cutoff
                and candidates[2][0] >= 0.94 * candidates[1][0]
            ):
                keep = 3

            for position, (_, middle, affinity, bilateral) in enumerate(
                candidates[:keep]
            ):
                rank_factor = 1.12 - 0.10 * position
                route_weight = (
                    0.37
                    * size_factor
                    * demand_support
                    * rank_factor
                    * (0.45 + 0.85 * affinity)
                    * (0.72 + 0.56 * bilateral)
                    / float(reward_pressure[middle])
                )

                first = (source, middle)
                second = (middle, target)
                terms.add_pair(first, second, -route_weight)

                # A modest interaction reduces double-counting when the
                # direct link and the best indirect route are both selected.
                if position == 0:
                    redundancy = (
                        0.045
                        * size_factor
                        * demand_support
                        * (0.45 + 0.55 * float(root[source, target]))
                        * (0.25 + 0.75 * affinity)
                    )
                    terms.add_pair((source, target), first, redundancy)


def _add_two_node_cut_costs(model, terms):
    """Penalize reciprocal links that leave a heavily loaded pair isolated."""
    n = model["n"]
    z = model["z"]
    row_sum = model["row_sum"]
    col_sum = model["col_sum"]
    incident_scale = max(model["peak_incident"], 1e-12)

    cut_linear = np.zeros((n, n), dtype=float)
    cut_strength = 0.80 / 144.0

    for i in range(n):
        for j in range(i + 1, n):
            internal = float(z[i, j] + z[j, i])
            external = max(
                0.0,
                max(
                    float(row_sum[i] + row_sum[j]),
                    float(col_sum[i] + col_sum[j]),
                ) - internal,
            )
            load = external / incident_scale
            coefficient = cut_strength * load * load
            if coefficient <= 0.0:
                continue

            cut_linear[i, j] = cut_linear[j, i] = 7.0 * coefficient
            terms.add_pair((i, j), (j, i), 13.0 * coefficient)

    # Remove components constant over valid degree-two topologies, avoiding
    # unintended competition with the evaluator's degree penalties.
    if n > 1:
        node_offset = cut_linear.sum(axis=1) / (n - 1)
        edge_offset = float(cut_linear.sum()) / (n * (n - 1))
        for i in range(n):
            for j in range(n):
                if i != j:
                    terms.linear[i, j] += (
                        cut_linear[i, j]
                        - node_offset[i]
                        - node_offset[j]
                        + edge_offset
                    )


def surrogate_terms(demand):
    """Return linear costs, sparse quadratic costs, and degree penalty."""
    model = _prepare_demand(demand)
    terms = _TermAccumulator(_direct_costs(model))

    if model["maximum"] > 0.0:
        _compile_routes(model, terms)
        _add_two_node_cut_costs(model, terms)

    penalty_weight = min(
        1.40, 1.25 + 0.006 * max(0, model["n"] - 10)
    )
    return terms.export(penalty_weight)


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
