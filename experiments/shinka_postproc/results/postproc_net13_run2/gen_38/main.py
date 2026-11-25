"""Starting program of post-processing run 2 (network13): run 1's best program (gen 49, combined
score 0.9901 on network12), copied from results/postproc_net12_run1/best/main.py. Only the fixed
wrapper below the EVOLVE block changed (tools.energy uses each pool's own Hamiltonian).
"""

"""Classical post-processing of QAOA output for QOBLIB 08. ShinkaEvolve edits the EVOLVE block."""

import json
import time

import numpy as np

# EVOLVE-BLOCK-START
def improve(demand, candidates, tools, rng):
    n = int(tools.n)
    best_net = None
    best_score = float("inf")
    scores = {}
    universe = [(i, j) for i in range(n) for j in range(n) if i != j]
    weight = {(i, j): 1.0 for i, j in universe}

    def canon(arcs):
        return tuple(sorted((int(a), int(b)) for a, b in arcs))

    def evaluate(net):
        nonlocal best_net, best_score
        net = canon(net)
        if net not in scores:
            scores[net] = float(tools.lp(net))
        if scores[net] < best_score:
            best_net, best_score = net, scores[net]
        return scores[net]

    # Repaired QAOA samples provide both starting topologies and a useful prior.
    repaired = set()
    for cand in candidates:
        try:
            net = canon(tools.repair(cand))
            if tools.is_valid(net):
                repaired.add(net)
        except Exception:
            pass

    if not repaired:
        try:
            net = canon(tools.repair([]))
            if tools.is_valid(net):
                repaired.add(net)
        except Exception:
            return []

    ranked = sorted(repaired, key=lambda x: float(tools.energy(x)))
    initial = ranked[:min(28, len(ranked))]
    freq = {a: 0.0 for a in universe}
    for rank, net in enumerate(ranked[:80]):
        amount = 1.0 / (1.0 + rank * 0.035)
        for a in net:
            freq[a] += amount
    for a in universe:
        weight[a] = 0.35 + freq[a]

    for net in initial:
        if tools.lp_calls_left <= 0:
            break
        try:
            evaluate(net)
        except Exception as exc:
            if type(exc).__name__ == "BudgetExhausted":
                break
            continue

    def random_matching(first, temperature):
        """Sample a perfect matching with weighted randomized row ordering."""
        rows = list(range(n))
        rng.shuffle(rows)
        used = set()
        result = []
        for i in rows:
            choices = [j for j in range(n) if j != i and j not in used
                       and (first is None or (i, j) not in first)]
            if not choices:
                return None
            vals = [max(1e-8, weight[(i, j)]) ** temperature for j in choices]
            total = sum(vals)
            pick = float(rng.random()) * total
            chosen = choices[-1]
            for j, v in zip(choices, vals):
                pick -= v
                if pick <= 0:
                    chosen = j
                    break
            used.add(chosen)
            result.append((i, chosen))
        return set(result)

    def sample():
        # Retries handle rare greedy dead ends; the final fallback is repair.
        for _ in range(24):
            first = random_matching(None, 0.7 + 0.8 * float(rng.random()))
            if first is None:
                continue
            second = random_matching(first, 0.7 + 0.8 * float(rng.random()))
            if second is not None:
                net = tuple(sorted(first | second))
                if tools.is_valid(net):
                    return net
        try:
            return canon(tools.repair([]))
        except Exception:
            return None

    generation = 0
    while tools.lp_calls_left > 0:
        generation += 1
        # Keep sampling batches large enough to use the LP budget, but adapt
        # the model only after a batch has supplied fresh elite information.
        batch_size = min(32, max(1, tools.lp_calls_left))
        batch = []
        attempts = 0
        while len(batch) < batch_size and attempts < batch_size * 12:
            attempts += 1
            net = sample()
            if net is not None and net not in scores and not tools.is_cached(net):
                batch.append(net)

        if not batch:
            # Inject repaired samples not previously evaluated before stopping.
            extras = [x for x in ranked if x not in scores and not tools.is_cached(x)]
            if not extras:
                break
            batch = extras[:batch_size]

        for net in batch:
            if tools.lp_calls_left <= 0:
                break
            try:
                evaluate(net)
            except Exception as exc:
                if type(exc).__name__ == "BudgetExhausted":
                    break
                continue

        # Cross-entropy update: learn from the best-scoring fraction so far,
        # with smoothing and a persistent uniform component to avoid collapse.
        elite_count = max(4, min(24, len(scores) // 8))
        elites = sorted(scores, key=scores.get)[:elite_count]
        if elites:
            target = {a: 0.0 for a in universe}
            for rank, net in enumerate(elites):
                contribution = 1.0 / (1.0 + 0.08 * rank)
                for a in net:
                    target[a] += contribution
            peak = max(target.values()) if target else 1.0
            if peak <= 0:
                peak = 1.0
            rate = 0.32 if generation < 12 else 0.22
            for a in universe:
                learned = 0.25 + 3.0 * target[a] / peak
                weight[a] = max(0.12, (1.0 - rate) * weight[a] + rate * learned)

    return list(best_net) if best_net is not None else list(initial[0])
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
