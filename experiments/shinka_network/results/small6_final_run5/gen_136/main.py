"""Surrogate QUBO terms for QOBLIB 08 network design. ShinkaEvolve edits the EVOLVE block."""

import time

# EVOLVE-BLOCK-START
import numpy as np


def surrogate_terms(demand):
    """Score links by direct demand and profile compatibility.

    Pairwise terms encourage diversity among a node's two outgoing neighbors
    and two incoming neighbors. Unlike path-based surrogates, this models
    each node's neighbors as a small portfolio of complementary connections.
    """
    traffic = np.asarray(demand, dtype=float)
    if traffic.ndim != 2 or traffic.shape[0] != traffic.shape[1]:
        raise ValueError("demand must be a square matrix")

    n = int(traffic.shape[0])
    traffic = np.maximum(
        np.nan_to_num(traffic, nan=0.0, posinf=0.0, neginf=0.0), 0.0
    ).copy()
    np.fill_diagonal(traffic, 0.0)

    penalty_weight = float(min(1.50, 1.27 + 0.010 * max(0, n - 10)))
    off_diagonal = ~np.eye(n, dtype=bool)
    scale = float(traffic.max()) if traffic.size else 0.0

    if scale <= 0.0:
        linear = {
            (i, j): 0.0
            for i in range(n)
            for j in range(n)
            if i != j
        }
        return linear, {}, penalty_weight

    z = traffic / scale
    root = np.sqrt(z)
    direct = 0.70 * z + 0.30 * root

    # Normalize each demand row and column into a profile. Bhattacharyya
    # overlap is in [0, 1] and remains meaningful for sparse demand matrices.
    row_sum = z.sum(axis=1)
    col_sum = z.sum(axis=0)
    row_profile = np.sqrt(
        z / np.maximum(row_sum[:, None], 1e-12)
    )
    col_profile = np.sqrt(
        z / np.maximum(col_sum[None, :], 1e-12)
    )
    row_overlap = np.clip(row_profile @ row_profile.T, 0.0, 1.0)
    col_overlap = np.clip(col_profile.T @ col_profile, 0.0, 1.0)

    # A link can be useful beyond its direct traffic if its head's demand
    # destinations overlap those of its tail. The analogous column overlap
    # captures compatibility for traffic arriving at the link's head.
    mean_row = max(float(row_sum.mean()), 1e-12)
    mean_col = max(float(col_sum.mean()), 1e-12)
    row_pressure = np.clip(row_sum / mean_row, 0.35, 2.5)
    col_pressure = np.clip(col_sum / mean_col, 0.35, 2.5)

    compatibility = (
        0.13 * row_pressure[:, None] * row_overlap
        + 0.09 * col_pressure[None, :] * col_overlap
    )

    # Avoid making high-total-demand nodes disproportionately attractive as
    # transit destinations. The direct-demand reward remains dominant.
    node_load = row_sum + col_sum
    load_scale = max(float(node_load.mean()), 1e-12)
    destination_load_cost = 0.045 * np.clip(node_load / load_scale, 0.0, 3.0)
    costs = -direct - compatibility + destination_load_cost[None, :]

    linear = {
        (i, j): float(costs[i, j])
        for i in range(n)
        for j in range(n)
        if i != j
    }

    quadratic = {}

    # With exactly two outgoing links per node, highly similar destination
    # profiles make the pair redundant. The same idea applies to incoming
    # links: similar source profiles provide less diverse access. These
    # interactions use O(n^3) terms and do not add degree penalties.
    for center in range(n):
        others = [v for v in range(n) if v != center]

        out_scale = 0.18 * float(np.clip(row_pressure[center], 0.5, 2.0))
        in_scale = 0.18 * float(np.clip(col_pressure[center], 0.5, 2.0))

        for a_idx in range(len(others)):
            a = others[a_idx]
            for b in others[a_idx + 1:]:
                # Two links with the same tail: penalize redundant heads.
                e1, e2 = (center, a), (center, b)
                key = (e1, e2) if e1 < e2 else (e2, e1)
                quadratic[key] = float(out_scale * row_overlap[a, b])

                # Two links with the same head: penalize redundant sources.
                e1, e2 = (a, center), (b, center)
                key = (e1, e2) if e1 < e2 else (e2, e1)
                quadratic[key] = float(in_scale * col_overlap[a, b])

    # Reciprocal links consume two endpoint slots. Keep a modest cost for
    # reciprocation unless there is substantial traffic in both directions.
    mean_direct = float(direct[off_diagonal].mean())
    reciprocal_scale = 0.10 * mean_direct
    for i in range(n):
        for j in range(i + 1, n):
            mutual = min(float(z[i, j]), float(z[j, i]))
            quadratic[((i, j), (j, i))] = float(
                reciprocal_scale * (1.0 - 0.65 * mutual)
            )

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
