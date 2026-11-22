"""Starting program of post-processing run 2 (network13): run 1's best program (gen 49, combined
score 0.9901 on network12), copied from results/postproc_net12_run1/best/main.py. Only the fixed
wrapper below the EVOLVE block changed (tools.energy uses each pool's own Hamiltonian).
"""

"""Classical post-processing of QAOA output for QOBLIB 08. ShinkaEvolve edits the EVOLVE block."""

import json
import time

import numpy as np

# EVOLVE-BLOCK-START
import math


def improve(demand, candidates, tools, rng):
    n = tools.n
    seen = {}
    best_net = None
    best_score = float("inf")

    def canon(arcs):
        return tuple(sorted((int(a), int(b)) for a, b in arcs))

    def edge_distance(a, b):
        return len(set(a) ^ set(b))

    def energy(net):
        try:
            return float(tools.energy(net))
        except Exception:
            return 0.0

    def evaluate(net):
        nonlocal best_net, best_score
        net = canon(net)
        if net not in seen:
            seen[net] = tools.lp(net)
        score = seen[net]
        if score < best_score:
            best_score, best_net = score, net
        return score

    # Repairing samples is free. Keep a low-energy, diverse subset as initial
    # LP evaluations and as a persistent prior for the sampling model.
    repaired = set()
    for candidate in candidates:
        try:
            net = canon(tools.repair(candidate))
            if tools.is_valid(net):
                repaired.add(net)
        except Exception:
            continue

    if not repaired:
        try:
            net = canon(tools.repair([]))
            if tools.is_valid(net):
                repaired.add(net)
        except Exception:
            return []

    ordered = sorted(repaired, key=energy)
    starts = []
    for net in ordered:
        if all(edge_distance(net, old) >= 4 for old in starts):
            starts.append(net)
        if len(starts) >= 24:
            break
    for net in ordered:
        if len(starts) >= min(40, len(ordered)):
            break
        if net not in starts:
            starts.append(net)

    # Build a prior edge-frequency model from promising repaired samples.
    prior = [[2.0 / max(1, n - 1) for _ in range(n)] for _ in range(n)]
    prior_samples = ordered[:min(40, len(ordered))]
    if prior_samples:
        counts = [[0.0] * n for _ in range(n)]
        for net in prior_samples:
            for a, b in net:
                counts[a][b] += 1.0
        for a in range(n):
            for b in range(n):
                if a != b:
                    prior[a][b] = (counts[a][b] + 1.5) / (len(prior_samples) + 1.5)

    for net in starts:
        if tools.lp_calls_left <= 0:
            break
        try:
            evaluate(net)
        except Exception as exc:
            if type(exc).__name__ == "BudgetExhausted":
                break

    if best_net is None:
        return list(starts[0]) if starts else []

    # Generate a weighted perfect matching using randomized depth-first
    # assignment. The second call excludes the first matching's edges.
    def weighted_matching(logits, excluded, temperature):
        assignment = [-1] * n
        used_targets = set()
        visits = [0]

        def rec(done):
            if done == n:
                return True
            visits[0] += 1
            if visits[0] > 5000:
                return False

            # Minimum-remaining-values source choice reduces backtracking.
            chosen_source = -1
            chosen_options = None
            for source in range(n):
                if assignment[source] >= 0:
                    continue
                options = [
                    target for target in range(n)
                    if target != source
                    and target not in used_targets
                    and (source, target) not in excluded
                ]
                if not options:
                    return False
                if chosen_options is None or len(options) < len(chosen_options):
                    chosen_source, chosen_options = source, options
                    if len(options) == 1:
                        break

            vals = [logits[chosen_source][t] for t in chosen_options]
            noise = rng.gumbel(size=len(chosen_options))
            order = sorted(
                range(len(chosen_options)),
                key=lambda i: vals[i] / max(temperature, 1e-6) + float(noise[i]),
                reverse=True,
            )
            for idx in order:
                target = chosen_options[idx]
                assignment[chosen_source] = target
                used_targets.add(target)
                if rec(done + 1):
                    return True
                used_targets.remove(target)
                assignment[chosen_source] = -1
            return False

        if rec(0):
            return {(a, assignment[a]) for a in range(n)}
        return None

    def sample_network(model, temperature):
        # Log probabilities act as edge utilities for both matchings.
        logits = [[0.0] * n for _ in range(n)]
        for a in range(n):
            for b in range(n):
                if a != b:
                    logits[a][b] = math.log(max(1e-5, model[a][b]))
        first = weighted_matching(logits, set(), temperature)
        if first is None:
            return None
        second = weighted_matching(logits, first, temperature)
        if second is None:
            return None
        net = tuple(sorted(first | second))
        if len(net) != 2 * n or not tools.is_valid(net):
            return None
        return net

    def elite_model():
        # Use good but not exclusively identical networks, so the model keeps
        # information about multiple strong regions of the search space.
        ranked = sorted(seen, key=seen.get)
        elite = []
        for net in ranked:
            if all(edge_distance(net, old) >= 2 for old in elite):
                elite.append(net)
            if len(elite) >= 12:
                break
        for net in ranked:
            if len(elite) >= 24:
                break
            if net not in elite:
                elite.append(net)

        model_counts = [[0.0] * n for _ in range(n)]
        total = 0.0
        for rank, net in enumerate(elite):
            weight = math.exp(-rank / 9.0)
            total += weight
            for a, b in net:
                model_counts[a][b] += weight

        updated = [[0.0] * n for _ in range(n)]
        for a in range(n):
            for b in range(n):
                if a == b:
                    continue
                estimate = (model_counts[a][b] + 0.12) / (total + 0.12)
                # Smooth updates to avoid collapsing onto one topology too soon.
                updated[a][b] = 0.68 * model[a][b] + 0.32 * estimate
        return updated

    model = [row[:] for row in prior]
    last_update = len(seen)
    iterations = 0
    attempts_without_new = 0
    best_at_update = best_score

    while tools.lp_calls_left > 0:
        iterations += 1

        # Periodic model updates are based only on true LP scores.
        if len(seen) - last_update >= 18:
            model = elite_model()
            last_update = len(seen)

        stagnation = max(0, iterations - 1)
        # Mildly raise temperature after prolonged stagnation, then cool
        # briefly when a better incumbent is found.
        temperature = 1.05 + min(1.15, stagnation / 700.0)
        if best_score < best_at_update:
            temperature = max(0.85, temperature * 0.8)
            best_at_update = best_score

        # Occasionally mix the learned model with the QAOA prior to retain
        # independent structural information from the original samples.
        if rng.random() < 0.22:
            sampling_model = [
                [0.65 * model[a][b] + 0.35 * prior[a][b] for b in range(n)]
                for a in range(n)
            ]
        else:
            sampling_model = model

        net = sample_network(sampling_model, temperature)
        if net is None or net in seen or tools.is_cached(net):
            attempts_without_new += 1
            if attempts_without_new > 300:
                # Break out of a duplicate-heavy region by sampling nearly
                # uniformly; the matching construction still guarantees validity.
                sampling_model = [
                    [1.0 if a != b else 0.0 for b in range(n)]
                    for a in range(n)
                ]
                net = sample_network(sampling_model, 2.8)
                if net is None or net in seen or tools.is_cached(net):
                    continue
            else:
                continue

        attempts_without_new = 0
        try:
            evaluate(net)
        except Exception as exc:
            if type(exc).__name__ == "BudgetExhausted":
                break
            continue

    return list(best_net) if best_net is not None else []
# EVOLVE-BLOCK-END


def run_postproc(pool_path, lp_budget, seed, control=False, time_limit=150):
    """Fixed entry point called by the evaluator: one run on one candidate pool."""
    import importlib.util
    import signal
    from pathlib import Path

    import oracle

    pool = json.load(open(pool_path))
    import qoblib_tools as qt

    if "hamiltonian_file" in pool:  # the Hamiltonian that sampled this pool (make_pools.py --tag)
        spec = importlib.util.spec_from_file_location(
            f"hamiltonian_{pool['hamiltonian_tag']}", Path(pool_path).resolve().parent / pool["hamiltonian_file"])
        ham = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ham)
        surrogate_terms = ham.surrogate_terms
    else:  # older pools: the frozen run 3 gen 28 Hamiltonian
        from hamiltonian import surrogate_terms

    n = int(pool["instance"][-2:])
    demand = qt.load_network_demand(n)
    linear, quadratic, _ = surrogate_terms(np.array(demand, dtype=float))
    tools = oracle.Tools(demand, lp_budget, (
        {tuple(map(int, a)): float(w) for a, w in linear.items()},
        {(tuple(map(int, a)), tuple(map(int, b))): float(w) for (a, b), w in quadratic.items()},
    ))
    candidates = [[tuple(map(int, a)) for a in c["arcs"]] for c in pool["candidates"]]
    if control:  # same number of random valid networks instead of QAOA output
        crng = np.random.default_rng(10_000 + seed)
        candidates = [qt.random_topology(n, crng) for _ in candidates]
    def _timeout(signum, frame):
        raise oracle.RunTimeout()

    start = time.perf_counter()
    returned, error, timed_out = None, None, False
    signal.signal(signal.SIGALRM, _timeout)
    signal.alarm(int(time_limit))
    try:
        returned = improve(demand.copy(), candidates, tools, np.random.default_rng(seed))
    except oracle.BudgetExhausted:
        pass
    except oracle.RunTimeout:
        timed_out = True
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
    finally:
        signal.alarm(0)
    best_score, best_arcs = tools.certified_best()
    if returned is not None:
        try:
            ret = [tuple(map(int, a)) for a in returned]
            if qt.is_valid_topology(n, ret):
                s = qt.route_min_congestion(demand, ret).objective
                if s < best_score:
                    best_score, best_arcs = s, tuple(sorted(ret))
        except Exception:  # noqa: BLE001
            pass
    return {
        "pool": pool_path, "control": control, "n": n, "error": error,
        "best": float(best_score), "arcs": [list(a) for a in best_arcs] if best_arcs else None,
        "lp_calls": tools.lp_calls_used, "seconds": time.perf_counter() - start, "timed_out": timed_out,
    }
