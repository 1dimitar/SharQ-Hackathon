"""Surrogate QUBO terms for QOBLIB 08 network design. ShinkaEvolve edits the EVOLVE block."""

import time

# EVOLVE-BLOCK-START
import numpy as np


def surrogate_terms(demand):
    """Build a demand-based surrogate for the network-design QUBO."""
    d = np.asarray(demand, dtype=float)
    n = len(d)

    # Traffic demands are nonnegative in the problem definition. Clipping also
    # gives well-defined behavior for an all-zero matrix.
    d = np.maximum(d, 0.0)
    scale = float(np.max(d)) if d.size else 0.0
    if scale <= 0.0:
        scale = 1.0
    dn = d / scale

    linear = {
        (i, j): -float(dn[i, j])
        for i in range(n) for j in range(n) if i != j
    }

    rows = np.sum(d, axis=1)
    cols = np.sum(d, axis=0)
    max_row = max(float(np.max(rows)) if rows.size else 0.0, 1.0e-12)
    max_col = max(float(np.max(cols)) if cols.size else 0.0, 1.0e-12)

    # Bounded endpoint pressure: retain a baseline conflict penalty, but
    # increase it for senders/receivers carrying a larger share of total load.
    row_pressure = np.sqrt(np.clip(rows / max_row, 0.0, 1.0))
    col_pressure = np.sqrt(np.clip(cols / max_col, 0.0, 1.0))
    row_factor = 0.25 + 0.75 * row_pressure
    col_factor = 0.25 + 0.75 * col_pressure

    quadratic = {}

    # Indirect two-hop service. Allocate each OD pair's small reward across
    # plausible relays instead of rewarding every possible path equally.
    relay = np.sqrt(np.maximum(rows, 0.0) * np.maximum(cols, 0.0))
    relay_max = max(float(np.max(relay)) if relay.size else 0.0, 1.0e-12)
    relay = (relay / relay_max) ** 1.5

    for i in range(n):
        for k in range(n):
            if i == k or dn[i, k] <= 0.0:
                continue

            candidates = []
            total = 0.0
            for j in range(n):
                if j == i or j == k:
                    continue
                endpoint = np.sqrt(dn[i, j] * dn[j, k])
                endpoint_factor = 0.25 + 0.75 * endpoint / (endpoint + 1.0)
                allocation = float(relay[j]) * endpoint_factor
                candidates.append((j, allocation))
                total += allocation

            if total > 0.0:
                for j, allocation in candidates:
                    weight = 0.10 * float(dn[i, k]) * allocation / total
                    if weight > 0.0:
                        quadratic[((i, j), (j, k))] = -weight

    # Competing links at a sender share its outgoing traffic; links entering
    # the same receiver share its incoming traffic. The exposure product
    # focuses these penalties on pairs that could plausibly carry substantial
    # traffic. Each endpoint factor is bounded in [0.25, 1].
    conflict_scale = 0.12
    for i in range(n):
        outgoing_links = [(i, j) for j in range(n) if j != i]
        factor = conflict_scale * float(row_factor[i])
        for a in range(len(outgoing_links)):
            u = outgoing_links[a]
            eu = float(dn[u[0], u[1]])
            if eu <= 0.0:
                continue
            for b in range(a + 1, len(outgoing_links)):
                v = outgoing_links[b]
                ev = float(dn[v[0], v[1]])
                weight = factor * eu * ev
                if weight > 0.0:
                    quadratic[(u, v)] = quadratic.get((u, v), 0.0) + weight

    for j in range(n):
        incoming_links = [(i, j) for i in range(n) if i != j]
        factor = conflict_scale * float(col_factor[j])
        for a in range(len(incoming_links)):
            u = incoming_links[a]
            eu = float(dn[u[0], u[1]])
            if eu <= 0.0:
                continue
            for b in range(a + 1, len(incoming_links)):
                v = incoming_links[b]
                ev = float(dn[v[0], v[1]])
                weight = factor * eu * ev
                if weight > 0.0:
                    quadratic[(u, v)] = quadratic.get((u, v), 0.0) + weight

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
