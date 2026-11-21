"""Surrogate QUBO terms for QOBLIB 08 network design. ShinkaEvolve edits the EVOLVE block."""

import time

# EVOLVE-BLOCK-START
import numpy as np


def surrogate_terms(demand):
    """Return direct-link, relay-path, and load-concentration QUBO terms."""
    matrix = np.asarray(demand, dtype=float)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    n = len(matrix)

    traffic = np.maximum(matrix, 0.0).copy()
    np.fill_diagonal(traffic, 0.0)

    scale = float(np.max(traffic)) if traffic.size else 0.0
    if scale <= 0.0:
        scale = 1.0
    normalized = traffic / scale

    # Direct service remains the primary signal, with modest concavity to retain
    # influence for medium-demand links.
    linear = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                value = float(normalized[i, j])
                linear[(i, j)] = -(0.80 * value + 0.20 * np.sqrt(value))

    # Estimate relay pressure from both originating and terminating traffic.
    outgoing = np.sum(traffic, axis=1)
    incoming = np.sum(traffic, axis=0)
    incident = outgoing + incoming
    mean_incident = float(np.mean(incident)) if n else 0.0
    if mean_incident <= 0.0:
        mean_incident = 1.0
    relay_pressure = np.maximum(0.5, incident / mean_incident)

    quadratic = {}

    def add_term(edge_a, edge_b, value):
        """Accumulate a symmetric QUBO coefficient under a canonical key."""
        if edge_a == edge_b:
            return
        key = (edge_a, edge_b) if edge_a < edge_b else (edge_b, edge_a)
        quadratic[key] = quadratic.get(key, 0.0) + float(value)

    # Give each OD pair a fixed total two-hop support budget. Candidate relays
    # divide that budget according to estimated relay availability and leg load.
    path_strength = 0.65
    for i in range(n):
        for k in range(n):
            if i == k:
                continue
            od_value = float(normalized[i, k])
            if od_value <= 0.0:
                continue

            candidates = []
            total_quality = 0.0
            for j in range(n):
                if j == i or j == k:
                    continue

                leg_load = float(normalized[i, j] + normalized[j, k])
                quality = (
                    1.0
                    / (0.65 + float(relay_pressure[j]))
                    / np.sqrt((1.0 + float(normalized[i, j]))
                              * (1.0 + float(normalized[j, k])))
                )
                candidates.append((j, quality))
                total_quality += quality

            if total_quality <= 0.0:
                continue

            for j, quality in candidates:
                share = quality / total_quality
                add_term(
                    (i, j),
                    (j, k),
                    -path_strength * od_value * share,
                )

    # Approximate direct-load concentration: selecting two high-traffic links
    # with the same tail or head incurs a positive interaction.
    concentration_strength = 0.30
    for center in range(n):
        neighbors = [v for v in range(n) if v != center]

        # Two outgoing links from the same source.
        for a in range(len(neighbors)):
            j = neighbors[a]
            for b in range(a + 1, len(neighbors)):
                k = neighbors[b]
                value = float(normalized[center, j] * normalized[center, k])
                if value > 0.0:
                    add_term(
                        (center, j),
                        (center, k),
                        concentration_strength * value,
                    )

        # Two incoming links to the same destination.
        for a in range(len(neighbors)):
            i = neighbors[a]
            for b in range(a + 1, len(neighbors)):
                k = neighbors[b]
                value = float(normalized[i, center] * normalized[k, center])
                if value > 0.0:
                    add_term(
                        (i, center),
                        (k, center),
                        concentration_strength * value,
                    )

    # Degree penalties are supplied by the evaluator.
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
