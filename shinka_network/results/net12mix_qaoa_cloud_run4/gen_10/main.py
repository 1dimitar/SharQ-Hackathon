"""Surrogate QUBO terms for QOBLIB 08 network design. ShinkaEvolve edits the EVOLVE block."""

import time

# EVOLVE-BLOCK-START
import numpy as np


def _prepare_demand(demand):
    """Return a nonnegative, normalized demand matrix and node traffic totals."""
    d = np.asarray(demand, dtype=float)
    n = len(d)
    if n == 0:
        return d, np.zeros(0), np.zeros(0)

    d = np.maximum(d, 0.0).copy()
    np.fill_diagonal(d, 0.0)
    scale = float(np.max(d))
    if scale > 0.0:
        d /= scale
    return d, d.sum(axis=1), d.sum(axis=0)


def _relay_candidates(d, row, col, source, target, limit):
    """Rank plausible intermediate nodes for one OD pair."""
    n = len(d)
    candidates = []
    for relay in range(n):
        if relay == source or relay == target:
            continue

        # A relay needs both receiving and sending capacity. The small
        # uniform component avoids excluding useful relays on sparse data.
        balance = np.sqrt(max(0.0, row[relay] * col[relay]))
        balance_scale = max(1.0, float(np.max(row)) * float(np.max(col)))
        endpoint_fit = 0.25 + 0.75 * min(
            1.0, float(d[source, relay] + d[relay, target])
        )
        score = (0.25 + balance / np.sqrt(balance_scale)) * endpoint_fit

        # Prefer routes whose constituent links have observed demand, while
        # preserving a positive baseline score for otherwise plausible paths.
        score *= 0.5 + np.sqrt(max(0.0, float(d[source, relay] * d[relay, target])))
        candidates.append((score, relay))

    # Stable tie-breaking makes results repeatable across calls.
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return candidates[:limit]


def surrogate_terms(demand):
    """Construct direct-link rewards and sparse two-hop route rewards."""
    d, row, col = _prepare_demand(demand)
    n = len(d)
    if n < 2:
        return {}, {}, 1.0

    links = [(i, j) for i in range(n) for j in range(n) if i != j]
    linear = {link: -float(d[link]) for link in links}
    quadratic = {}

    # Restrict each OD pair to a few likely relays. This preserves path
    # incentives without creating every possible two-hop interaction.
    relay_limit = min(4, max(1, n - 2))
    indirect_fraction = 0.34
    component_fraction = 0.10

    for source in range(n):
        for target in range(n):
            flow = float(d[source, target])
            if source == target or flow <= 0.0:
                continue

            candidates = _relay_candidates(
                d, row, col, source, target, relay_limit
            )
            if not candidates:
                continue

            total_score = sum(score for score, _ in candidates)
            if total_score <= 0.0:
                continue

            for score, relay in candidates:
                share = flow * score / total_score
                first = (source, relay)
                second = (relay, target)
                key = (first, second) if first < second else (second, first)

                # A modest component reward keeps useful links attractive,
                # while the larger joint reward applies only to a full route.
                linear[first] -= component_fraction * share
                linear[second] -= component_fraction * share
                quadratic[key] = quadratic.get(key, 0.0) - (
                    indirect_fraction * share
                )

    # Normalize internally for numerical stability; the evaluator may apply
    # its own common rescaling to the returned coefficients.
    largest = max(
        max((abs(value) for value in linear.values()), default=0.0),
        max((abs(value) for value in quadratic.values()), default=0.0),
        1e-12,
    )
    inverse = 1.0 / largest
    linear = {key: float(value * inverse) for key, value in linear.items()}
    quadratic = {key: float(value * inverse) for key, value in quadratic.items()}

    return linear, quadratic, 2.5


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
