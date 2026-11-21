"""Surrogate QUBO terms for QOBLIB 08 network design. ShinkaEvolve edits the EVOLVE block."""

import time

# EVOLVE-BLOCK-START
import numpy as np


def surrogate_terms(demand):
    """Approximate routing benefit and transit congestion with sparse QUBO terms."""
    traffic = np.asarray(demand, dtype=float)
    n = len(traffic)
    traffic = np.nan_to_num(traffic, nan=0.0, posinf=0.0, neginf=0.0)
    traffic = np.maximum(traffic, 0.0).copy()
    np.fill_diagonal(traffic, 0.0)

    scale = float(np.max(traffic)) if traffic.size else 0.0
    if scale <= 0.0:
        scale = 1.0
    z = traffic / scale
    root = np.sqrt(z)

    # Direct service benefit. The concave component keeps moderate demands
    # relevant without letting the largest entries dominate all choices.
    direct = 0.68 * z + 0.32 * root
    linear = {
        (i, j): -float(direct[i, j])
        for i in range(n)
        for j in range(n)
        if i != j
    }

    # A node's baseline transit pressure combines traffic entering and leaving.
    incident = np.maximum(traffic.sum(axis=0), traffic.sum(axis=1))
    inc_scale = float(np.max(incident)) if incident.size else 0.0
    if inc_scale <= 0.0:
        inc_scale = 1.0
    base_pressure = np.maximum(0.12, incident / inc_scale)

    # For each OD pair, calculate a soft distribution across a direct option
    # and all possible two-hop options. This is used only to estimate where
    # transit load is likely to concentrate.
    route_data = {}
    transit = np.zeros(n, dtype=float)
    for i in range(n):
        for k in range(n):
            if i == k or z[i, k] <= 0.0:
                continue

            candidates = []
            for j in range(n):
                if j == i or j == k:
                    continue
                a = float(z[i, j])
                b = float(z[j, k])
                leg_support = 0.12 + np.sqrt(a * b) + 0.12 * (a + b)
                utility = np.log(max(1e-12, leg_support))
                utility -= 0.55 * np.log(0.25 + float(base_pressure[j]))
                candidates.append((j, a, b, leg_support, utility))

            # Direct service competes with two-hop routing in the soft estimate.
            direct_utility = np.log(0.16 + 1.25 * float(z[i, k]))
            logits = np.asarray(
                [direct_utility] + [item[4] for item in candidates],
                dtype=float,
            )
            logits = (logits - float(np.max(logits))) / 0.72
            probs = np.exp(np.clip(logits, -30.0, 0.0))
            probs /= max(1e-12, float(probs.sum()))

            amount = float(z[i, k])
            for idx, (j, _, _, _, _) in enumerate(candidates, start=1):
                transit[j] += amount * float(probs[idx])
            route_data[(i, k)] = candidates

    # Update pressure with the soft routing estimate. Normalize transit load
    # by overall incident traffic so the scale behaves consistently with n.
    transit_scale = max(
        1e-12,
        float(np.mean(incident)) if incident.size else 0.0,
        scale,
    )
    transit_ratio = transit / transit_scale
    max_ratio = max(1.0, float(np.max(transit_ratio)) if transit_ratio.size else 1.0)
    pressure = np.maximum(
        0.12,
        0.58 * (incident / inc_scale) + 0.42 * (transit_ratio / max_ratio),
    )

    # Start with a modest congestion tax for every possible two-link transit
    # combination. Its cubic number of terms remains below the permitted cap.
    quadratic = {}
    for j in range(n):
        hub_tax = 0.055 * (0.25 + 0.75 * float(pressure[j]))
        for i in range(n):
            if i == j:
                continue
            first = (i, j)
            for k in range(n):
                if k == i or k == j:
                    continue
                second = (j, k)
                key = (first, second) if first < second else (second, first)
                quadratic[key] = hub_tax

    # Subtract route-specific benefits from the congestion tax. Candidate
    # routes are ranked by constituent-leg support adjusted for transit load.
    positive = traffic[traffic > 0.0]
    cutoff = float(np.quantile(positive, 0.72)) / scale if positive.size else 0.0

    for (i, k), candidates in route_data.items():
        support = 0.66 * float(z[i, k]) + 0.34 * float(root[i, k])
        ranked = []
        for j, a, b, leg_support, _ in candidates:
            p = float(pressure[j])
            gate = 0.25 + 0.75 * np.sqrt(max(0.0, a * b))
            rank = leg_support * gate / (0.35 + p) ** 0.65
            ranked.append((float(rank), j, float(gate), p))

        ranked.sort(key=lambda item: (-item[0], item[1]))
        keep = min(len(ranked), 3 if z[i, k] >= cutoff else 2)

        for position, (_, j, gate, p) in enumerate(ranked[:keep]):
            # Route rewards are stronger for leading alternatives, while the
            # leg gate avoids rewarding paths with unsupported constituent links.
            rank_boost = 1.0 + 0.06 * (keep - position - 1)
            reward = (
                0.34
                * support
                * gate
                * rank_boost
                / (0.55 + p) ** 0.40
            )
            first, second = (i, j), (j, k)
            key = (first, second) if first < second else (second, first)
            quadratic[key] = quadratic.get(key, 0.0) - float(reward)

    # The evaluator supplies the degree penalty separately.
    penalty_weight = min(1.50, max(0.05, 1.27 + 0.010 * max(0, n - 10)))
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
