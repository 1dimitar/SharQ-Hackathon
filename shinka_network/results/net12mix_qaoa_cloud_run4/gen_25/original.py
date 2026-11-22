"""Surrogate QUBO terms for QOBLIB 08 network design. ShinkaEvolve edits the EVOLVE block."""

import time

# EVOLVE-BLOCK-START
import numpy as np


def _prepare_demand(demand):
    """Return nonnegative demand and traffic-based node importance."""
    t = np.asarray(demand, dtype=float)
    n = t.shape[0]
    t = np.maximum(t, 0.0).copy()
    np.fill_diagonal(t, 0.0)

    row = t.sum(axis=1)
    col = t.sum(axis=0)
    total = float(t.sum())
    if total > 0:
        # A modest node term rewards links touching nodes with substantial
        # traffic, without overwhelming the destination-specific demand.
        node = (row + col) / total
    else:
        node = np.zeros(n, dtype=float)
    return t, node


def _make_link_rewards(t, node):
    """Construct normalized direct-link utilities."""
    n = len(t)
    maximum = float(t.max()) if t.size else 0.0
    if maximum <= 0:
        maximum = 1.0

    rewards = {}
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            direct = float(t[i, j]) / maximum
            # Endpoint importance is deliberately a small correction: it
            # helps accommodate traffic concentration while preserving the
            # strong signal to directly connect high-demand pairs.
            endpoint = 0.5 * (float(node[i]) + float(node[j]))
            rewards[(i, j)] = direct + 0.12 * endpoint
    return rewards


def _make_path_couplings(t, rewards):
    """Give bounded two-hop credit to the strongest relay options per pair."""
    n = len(t)
    maximum = float(t.max()) if t.size else 0.0
    if maximum <= 0:
        return {}

    row = t.sum(axis=1)
    col = t.sum(axis=0)
    total = float(t.sum())
    node_exposure = (row + col) / total if total > 0 else np.zeros(n)
    quadratic = {}

    for i in range(n):
        for k in range(n):
            if i == k or t[i, k] <= 0:
                continue

            candidates = []
            for j in range(n):
                if j != i and j != k:
                    # Prefer relays whose two links are already attractive;
                    # modestly favor less traffic-loaded transit nodes.
                    link_hint = 1.0 + 0.25 * (
                        rewards[(i, j)] + rewards[(j, k)]
                    )
                    relay_hint = link_hint / (1.0 + 2.0 * float(node_exposure[j]))
                    candidates.append((relay_hint, j))

            candidates.sort(reverse=True)
            selected = candidates[:2]
            denominator = sum(score for score, _ in selected)
            if denominator <= 0:
                continue

            demand_strength = float(t[i, k]) / maximum
            for score, j in selected:
                key = ((i, j), (j, k))
                value = -0.24 * demand_strength * score / denominator
                quadratic[key] = quadratic.get(key, 0.0) + value

    return quadratic


def surrogate_terms(demand):
    """Build linear and quadratic QUBO cost terms from a demand matrix."""
    t, node = _prepare_demand(demand)
    linear = {link: -value for link, value in _make_link_rewards(t, node).items()}
    quadratic = _make_path_couplings(t, {link: -value for link, value in linear.items()})

    # The evaluator applies this separately to both incoming and outgoing
    # degree deviations. Keep it strong enough to discourage repair-heavy
    # samples after the QUBO terms are rescaled.
    penalty_weight = 4.0
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
