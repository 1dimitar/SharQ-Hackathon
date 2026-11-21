"""Surrogate QUBO terms for QOBLIB 08 network design. ShinkaEvolve edits the EVOLVE block."""

import time

# EVOLVE-BLOCK-START
import numpy as np


def surrogate_terms(demand):
    """Build a direct-service and routing-diversity QUBO surrogate."""
    traffic = np.maximum(np.asarray(demand, dtype=float), 0.0).copy()
    n = traffic.shape[0]
    np.fill_diagonal(traffic, 0.0)

    scale = float(np.max(traffic)) if traffic.size else 0.0
    if scale <= 0.0:
        scale = 1.0
    a = traffic / scale

    # Keep direct service attractive, but give moderate demands a voice too.
    linear = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                v = float(a[i, j])
                linear[(i, j)] = -(0.82 * v + 0.18 * np.sqrt(v))

    # Compare nodes by the traffic profiles they send and receive. Cosine
    # similarity measures whether two candidate neighbors offer redundant
    # connectivity options.
    outgoing = a.copy()
    incoming = a.T.copy()

    def cosine_similarity_rows(profiles):
        norms = np.sqrt(np.sum(profiles * profiles, axis=1))
        denom = np.outer(norms, norms)
        sim = np.zeros((n, n), dtype=float)
        valid = denom > 1e-15
        sim[valid] = (profiles @ profiles.T)[valid] / denom[valid]
        return np.clip(sim, 0.0, 1.0)

    out_sim = cosine_similarity_rows(outgoing)
    in_sim = cosine_similarity_rows(incoming)

    # Penalize selecting two links to near-equivalent neighbors. Scale the
    # penalty by their incident traffic so the effect is strongest where
    # transit capacity is most consequential.
    row_load = np.sum(a, axis=1)
    col_load = np.sum(a, axis=0)
    mean_load = float(np.mean(row_load + col_load))
    if mean_load <= 1e-12:
        mean_load = 1.0
    row_pressure = np.sqrt(np.maximum(row_load / mean_load, 0.0))
    col_pressure = np.sqrt(np.maximum(col_load / mean_load, 0.0))

    quadratic = {}
    strength = 0.30

    # Same-tail pairs: prefer two outgoing links with complementary
    # destination profiles.
    for i in range(n):
        for j in range(n):
            if j == i:
                continue
            for k in range(j + 1, n):
                if k == i:
                    continue
                redundancy = float(out_sim[j, k])
                pressure = 0.5 * (row_pressure[i] + col_pressure[j] +
                                  row_pressure[i] + col_pressure[k])
                weight = strength * redundancy * min(2.0, pressure)
                if weight > 1e-12:
                    quadratic[((i, j), (i, k))] = weight

    # Same-head pairs: avoid spending both incoming slots on sources whose
    # outgoing traffic profiles are alike.
    for j in range(n):
        for i in range(n):
            if i == j:
                continue
            for k in range(i + 1, n):
                if k == j:
                    continue
                redundancy = float(in_sim[i, k])
                pressure = 0.5 * (col_pressure[j] + row_pressure[i] +
                                  col_pressure[j] + row_pressure[k])
                weight = strength * redundancy * min(2.0, pressure)
                if weight > 1e-12:
                    quadratic[((i, j), (k, j))] = weight

    return linear, quadratic, 1.0


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
