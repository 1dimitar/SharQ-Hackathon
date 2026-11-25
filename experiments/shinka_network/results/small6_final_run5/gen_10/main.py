"""Surrogate QUBO terms for QOBLIB 08 network design. ShinkaEvolve edits the EVOLVE block."""

import time

# EVOLVE-BLOCK-START
import numpy as np


def surrogate_terms(demand):
    """Return direct-link costs, sparse coverage interactions, and penalty."""
    traffic = np.asarray(demand, dtype=float).copy()
    traffic = np.maximum(traffic, 0.0)
    np.fill_diagonal(traffic, 0.0)
    n = len(traffic)

    maximum = float(traffic.max()) if traffic.size else 0.0
    if maximum <= 0.0:
        linear = {
            (i, j): 0.0
            for i in range(n)
            for j in range(n)
            if i != j
        }
        return linear, {}, 1.45

    t = traffic / maximum
    root = np.sqrt(t)

    # Main parameters:
    #   direct blend:                 0.78 demand + 0.22 sqrt(demand)
    #   strong-demand quantile:       0.75
    #   selected affinity paths:      4 strong / 2 ordinary
    #   additional spare paths:       1
    #   main / spare path strengths:  0.44 / 0.27
    #   cubic-redundancy damping:     0.65
    #   degree-penalty weight:        1.45
    direct = 0.78 * t + 0.22 * root
    linear = {
        (i, j): -float(direct[i, j])
        for i in range(n)
        for j in range(n)
        if i != j
    }
    quadratic = {}

    incident = np.maximum(t.sum(axis=0), t.sum(axis=1))
    pressure = np.maximum(
        0.65, incident / max(float(incident.max()), 1e-12)
    )

    positive = t[t > 0.0]
    cutoff = float(np.quantile(positive, 0.75))

    # Fractional reference link occupancies. These are only used to expand
    # the direct/path redundancy term, not to impose degree constraints.
    attractiveness = 0.18 + direct
    np.fill_diagonal(attractiveness, 0.0)
    row_mass = attractiveness.sum(axis=1)
    col_mass = attractiveness.sum(axis=0)
    denominator = np.sqrt(row_mass[:, None] * col_mass[None, :])
    occupancy = np.minimum(
        0.80,
        2.0 * attractiveness / np.maximum(denominator, 1e-12),
    )
    np.fill_diagonal(occupancy, 0.0)

    # Keep indirect rewards from becoming increasingly dominant as n grows.
    size_factor = (10.0 / max(n, 5)) ** 0.15
    redundancy = 0.65

    def add_quadratic(a, b, weight):
        if abs(weight) < 1e-14:
            return
        key = (a, b) if a < b else (b, a)
        quadratic[key] = quadratic.get(key, 0.0) + float(weight)

    def add_path(i, j, k, reward):
        """Approximate -reward * A*B*(1-D) through second order.

        A = x_ij, B = x_jk, D = x_ik. The cubic correction is
        damped to avoid treating fractional occupancies as exact routing.
        """
        first = (i, j)
        second = (j, k)
        direct_link = (i, k)

        pa = float(occupancy[i, j])
        pb = float(occupancy[j, k])
        pd = float(occupancy[i, k])
        correction = redundancy * reward

        add_quadratic(first, second, -reward + correction * pd)
        add_quadratic(direct_link, first, correction * pb)
        add_quadratic(direct_link, second, correction * pa)

        # Linear terms belonging to the same second-order expansion.
        linear[direct_link] -= correction * pa * pb
        linear[first] -= correction * pd * pb
        linear[second] -= correction * pd * pa

    for i in range(n):
        for k in range(n):
            if i == k or t[i, k] <= 0.0:
                continue

            support = float(0.73 * t[i, k] + 0.27 * root[i, k])
            candidates = []

            for j in range(n):
                if j == i or j == k:
                    continue

                a = float(t[i, j])
                b = float(t[j, k])
                total = a + b
                geometric = float(np.sqrt(a * b))

                # Favor intermediates supported by both legs, rather than
                # a single high-demand leg and an otherwise arbitrary edge.
                balance = 2.0 * geometric / (0.20 + total)
                affinity = 0.18 + 0.65 * total + 0.70 * geometric
                node_load = float(pressure[j])
                rank = affinity / (node_load ** 0.80)
                discount = node_load * (1.0 + 0.50 * total)

                candidates.append(
                    (rank, j, discount, balance)
                )

            candidates.sort(key=lambda item: (-item[0], item[1]))
            keep = min(
                len(candidates),
                4 if t[i, k] >= cutoff else 2,
            )

            for position, (_, j, discount, balance) in enumerate(
                candidates[:keep]
            ):
                evidence = 0.78 + 0.44 * balance
                rank_factor = 1.0 + 0.07 * (keep - position)
                reward = (
                    0.44
                    * size_factor
                    * support
                    * evidence
                    * rank_factor
                    / discount
                )
                add_path(i, j, k, reward)

            # Retain a capacity-oriented alternative without expanding the
            # number of selected routes as the instance grows.
            remaining = candidates[keep:]
            if remaining:
                _, j, discount, balance = min(
                    remaining,
                    key=lambda item: (
                        item[2] / (0.85 + 0.15 * item[3]),
                        item[1],
                    ),
                )
                reward = (
                    0.27
                    * size_factor
                    * support
                    * (0.85 + 0.30 * balance)
                    / discount
                )
                add_path(i, j, k, reward)

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
