"""Surrogate QUBO terms for QOBLIB 08 network design. ShinkaEvolve edits the EVOLVE block."""

import time

# EVOLVE-BLOCK-START
import numpy as np


def _prepare_demand(demand):
    """Return sanitized demand and normalized node-level statistics."""
    d = np.asarray(demand, dtype=float)
    d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
    d = np.maximum(d, 0.0)

    n = d.shape[0]
    scale = float(np.max(d)) if d.size else 0.0
    if scale <= 0.0:
        scale = 1.0

    rows = np.sum(d, axis=1)
    cols = np.sum(d, axis=0)

    row_norm = np.linalg.norm(d, axis=1)
    col_norm = np.linalg.norm(d, axis=0)
    row_profiles = np.divide(
        d,
        row_norm[:, None],
        out=np.zeros_like(d),
        where=row_norm[:, None] > 0.0,
    )
    col_profiles = np.divide(
        d,
        col_norm[None, :],
        out=np.zeros_like(d),
        where=col_norm[None, :] > 0.0,
    )

    # A node with high total incident demand is a less attractive transit point.
    pressure = rows + cols
    max_pressure = float(np.max(pressure)) if pressure.size else 0.0
    if max_pressure > 0.0:
        pressure = pressure / max_pressure
    else:
        pressure = np.zeros(n, dtype=float)

    return d, scale, row_profiles, col_profiles, pressure


def _build_linear(demand, scale):
    """Reward links that carry substantial traffic directly."""
    n = demand.shape[0]
    return {
        (i, j): -float(demand[i, j]) / scale
        for i in range(n)
        for j in range(n)
        if i != j
    }


def _add_endpoint_diversity(quadratic, row_profiles, col_profiles):
    """Discourage choosing two endpoint links with nearly redundant profiles."""
    n = row_profiles.shape[0]
    weight = 0.10

    for node in range(n):
        neighbors = [v for v in range(n) if v != node]

        # Outgoing neighbors are more useful as a pair when they offer
        # different downstream traffic profiles.
        for a in range(len(neighbors)):
            j = neighbors[a]
            for b in range(a + 1, len(neighbors)):
                k = neighbors[b]
                similarity = float(np.dot(row_profiles[j], row_profiles[k]))
                if similarity > 1e-12:
                    quadratic[((node, j), (node, k))] = weight * similarity

        # Incoming neighbors are more useful as a pair when they provide
        # access to different traffic origins.
        for a in range(len(neighbors)):
            j = neighbors[a]
            for b in range(a + 1, len(neighbors)):
                k = neighbors[b]
                similarity = float(np.dot(col_profiles[j], col_profiles[k]))
                if similarity > 1e-12:
                    quadratic[((j, node), (k, node))] = weight * similarity


def _add_two_hop_rewards(quadratic, demand, scale, pressure):
    """Reward link pairs that provide a discounted two-hop route for demand."""
    n = demand.shape[0]
    indirect_fraction = 0.60
    demand_cutoff = 0.01 * scale
    gate_scale = 0.05 * scale

    for source in range(n):
        for middle in range(n):
            if middle == source:
                continue
            first_link = (source, middle)

            # The destination must differ from both ends of the first link.
            for target in range(n):
                if target == source or target == middle:
                    continue

                traffic = float(demand[source, target])
                if traffic <= demand_cutoff:
                    continue

                # Smoothly suppress marginal demands near the cutoff.
                gate = min(1.0, traffic / gate_scale) if gate_scale > 0.0 else 1.0
                transit_discount = 1.0 / (1.0 + float(pressure[middle]))

                # Division by four keeps the accumulated path reward moderate
                # relative to the direct-link term.
                reward = (
                    -indirect_fraction
                    * (traffic / scale)
                    * gate
                    * transit_discount
                    / 4.0
                )
                if reward:
                    quadratic[(first_link, (middle, target))] = reward


def surrogate_terms(demand):
    """Build linear and quadratic QUBO cost terms from a demand matrix."""
    d, scale, row_profiles, col_profiles, pressure = _prepare_demand(demand)

    linear = _build_linear(d, scale)
    quadratic = {}

    _add_endpoint_diversity(quadratic, row_profiles, col_profiles)
    _add_two_hop_rewards(quadratic, d, scale, pressure)

    # The evaluator applies the degree constraints separately.
    penalty_weight = 1.0
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
