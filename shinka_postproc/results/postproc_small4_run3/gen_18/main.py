"""Starting program of post-processing run 2 (network13): run 1's best program (gen 49, combined
score 0.9901 on network12), copied from results/postproc_net12_run1/best/main.py. Only the fixed
wrapper below the EVOLVE block changed (tools.energy uses each pool's own Hamiltonian).
"""

"""Classical post-processing of QAOA output for QOBLIB 08. ShinkaEvolve edits the EVOLVE block."""

import json
import time

import numpy as np

# EVOLVE-BLOCK-START
import numpy as np


def improve(demand, candidates, tools, rng):
    """Cross-entropy search over degree-valid networks built from matchings."""
    n = int(tools.n)
    baseline = 2.0 / n
    seen = {}
    energy_cache = {}
    best_net = None
    best_score = float("inf")

    def canon(arcs):
        return tuple(sorted((int(a), int(b)) for a, b in arcs))

    def energy(net):
        if net not in energy_cache:
            try:
                energy_cache[net] = float(tools.energy(net))
            except Exception:
                energy_cache[net] = 0.0
        return energy_cache[net]

    def evaluate(net):
        nonlocal best_net, best_score
        net = canon(net)
        if net not in seen:
            seen[net] = float(tools.lp(net))
        score = seen[net]
        if score < best_score:
            best_net, best_score = net, score
        return score

    def distance(a, b):
        return len(set(a) ^ set(b))

    # Repair the QAOA pool, then use its better and more varied members as seeds.
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

    # Initialize edge preferences from the strongest portion of the repaired pool.
    prior_count = min(len(ordered), max(24, 3 * n))
    freq = np.zeros((n, n), dtype=float)
    if prior_count:
        for net in ordered[:prior_count]:
            for u, v in net:
                freq[u, v] += 1.0 / prior_count
    model = np.log((freq + 0.04) / (baseline + 0.04))
    model = np.clip(model, -1.5, 1.5)
    np.fill_diagonal(model, 0.0)

    # Remove row/column offsets, which do not affect a perfect matching.
    def center(mat):
        out = mat.copy()
        for _ in range(2):
            out -= out.mean(axis=1, keepdims=True)
            out -= out.mean(axis=0, keepdims=True)
        np.fill_diagonal(out, 0.0)
        return out

    model = center(model)

    # Score a compact, topology-diverse seed set before generating new networks.
    starts = []
    for net in ordered:
        if all(distance(net, old) >= 4 for old in starts):
            starts.append(net)
        if len(starts) >= min(len(ordered), max(2 * n, 20)):
            break
    for net in ordered:
        if len(starts) >= min(len(ordered), max(2 * n, 20)):
            break
        if net not in starts:
            starts.append(net)

    for net in starts:
        if tools.lp_calls_left <= 0:
            break
        try:
            evaluate(net)
        except Exception as exc:
            if type(exc).__name__ == "BudgetExhausted":
                break

    if not seen:
        return list(starts[0]) if starts else []

    def matching(scores, forbidden):
        """Randomized augmenting-path matching using scores as edge preferences."""
        preferences = []
        for u in range(n):
            row = []
            for v in range(n):
                if u == v or (u, v) in forbidden:
                    continue
                row.append(v)
            row.sort(key=lambda v: scores[u, v], reverse=True)
            preferences.append(row)

        rows = list(range(n))
        rng.shuffle(rows)
        match_col = [-1] * n

        def augment(u, visited):
            for v in preferences[u]:
                if v in visited:
                    continue
                visited.add(v)
                old = match_col[v]
                if old < 0 or augment(old, visited):
                    match_col[v] = u
                    return True
            return False

        for u in rows:
            if not augment(u, set()):
                return None

        result = [None] * n
        for v, u in enumerate(match_col):
            if u < 0:
                return None
            result[u] = v
        return result

    elite_nets = []

    def refresh_model():
        """Update edge logits from LP elites, retaining structural variety."""
        nonlocal model, elite_nets
        ranked = sorted(seen, key=seen.get)
        elite_nets = []
        for net in ranked:
            if all(distance(net, old) >= 4 for old in elite_nets):
                elite_nets.append(net)
            if len(elite_nets) >= max(12, min(40, 3 * n)):
                break
        for net in ranked:
            if len(elite_nets) >= max(16, min(48, 4 * n)):
                break
            if net not in elite_nets:
                elite_nets.append(net)

        if not elite_nets:
            return

        counts = np.zeros((n, n), dtype=float)
        total = 0.0
        for rank, net in enumerate(elite_nets):
            weight = np.exp(-rank / max(5.0, len(elite_nets) * 0.35))
            total += weight
            for u, v in net:
                counts[u, v] += weight
        frequencies = counts / max(total, 1e-12)
        target = np.log((frequencies + 0.04) / (baseline + 0.04))
        target = center(np.clip(target, -2.0, 2.0))
        model = center(0.58 * model + 0.42 * target)

    def sample_network(stall_count):
        """Sample a 2-factor as two edge-disjoint perfect matchings."""
        mode = float(rng.random())
        if mode < 0.10:
            model_scale = 0.0                 # broad uniform exploration
            donor = None
        elif mode < 0.32 and elite_nets:
            model_scale = 0.65
            donor = elite_nets[int(rng.integers(len(elite_nets)))]
        else:
            model_scale = 1.0
            donor = None

        noise_scale = 0.75 + (0.45 if stall_count > 80 else 0.0)
        base = model_scale * model
        if donor is not None:
            for u, v in donor:
                base[u, v] += 0.8

        # A few attempts protect against a failed second matching.
        for _ in range(12):
            first_scores = base + rng.normal(0.0, noise_scale, size=(n, n))
            np.fill_diagonal(first_scores, -1e9)
            first = matching(first_scores, set())
            if first is None:
                continue
            first_edges = {(u, first[u]) for u in range(n)}

            second_scores = base + rng.normal(0.0, noise_scale, size=(n, n))
            np.fill_diagonal(second_scores, -1e9)
            second = matching(second_scores, first_edges)
            if second is None:
                continue

            net = tuple(sorted(first_edges | {(u, second[u]) for u in range(n)}))
            if len(net) == 2 * n and tools.is_valid(net):
                return net
        return None

    refresh_model()
    calls_at_update = len(seen)
    stall_count = 0
    duplicate_streak = 0

    while tools.lp_calls_left > 0:
        net = sample_network(stall_count)
        if net is None:
            # A random valid repaired topology is a safe fallback if matching
            # construction happens to fail repeatedly.
            try:
                net = canon(tools.repair([]))
            except Exception:
                break
            if not tools.is_valid(net):
                break

        if net in seen:
            duplicate_streak += 1
            if duplicate_streak > 100:
                # Increase exploration rather than spinning on a concentrated model.
                model = center(0.7 * model)
                duplicate_streak = 0
            continue

        duplicate_streak = 0
        old_best = best_score
        try:
            evaluate(net)
        except Exception as exc:
            if type(exc).__name__ == "BudgetExhausted":
                break
            continue

        if best_score < old_best:
            stall_count = 0
        else:
            stall_count += 1

        # Frequent updates let the learned distribution follow improvements, while
        # amortizing sorting and edge-frequency calculations over several LP calls.
        if len(seen) - calls_at_update >= max(12, n):
            refresh_model()
            calls_at_update = len(seen)

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
