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
    n = tools.n
    scored = {}
    energy_cache = {}
    best_net = None
    best_score = float("inf")

    def canon(arcs):
        return tuple(sorted(tuple(map(int, a)) for a in arcs))

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
        if net not in scored:
            scored[net] = float(tools.lp(net))
        value = scored[net]
        if value < best_score:
            best_net, best_score = net, value
        return value

    def dist(a, b):
        return len(set(a) ^ set(b))

    # Repair is free. Retain surrogate-leading starts with structural variety.
    repaired = set()
    for raw in candidates:
        try:
            net = canon(tools.repair(raw))
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

    ranked = sorted(repaired, key=energy)
    starts = []
    for net in ranked:
        if all(dist(net, other) >= 4 for other in starts):
            starts.append(net)
        if len(starts) >= 18:
            break
    for net in ranked:
        if len(starts) >= 28:
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
    if not scored:
        return list(starts[0])

    archive = sorted(scored, key=scored.get)[:]
    current = best_net
    current_score = scored[current]
    stall = 0
    iteration = 0

    def switch(cur, p, q):
        arcs = list(cur)
        a, b = arcs[p]
        c, d = arcs[q]
        if a == d or c == b:
            return None
        edges = set(cur)
        if (a, d) in edges or (c, b) in edges:
            return None
        edges.remove((a, b))
        edges.remove((c, d))
        edges.add((a, d))
        edges.add((c, b))
        return tuple(sorted(edges))

    def mutate(parent, detail, strength):
        arcs = list(parent)
        edges = set(parent)
        prices = detail.get("prices", {}) if detail else {}
        loads = detail.get("loads", {}) if detail else {}
        hot = [tuple(map(int, a)) for a, p in prices.items() if float(p) > 0]
        if not hot and loads:
            hot = [tuple(map(int, a)) for a, _ in sorted(
                loads.items(), key=lambda item: float(item[1]), reverse=True
            )[:6]]

        # A perturbation is a short sequence of legal switches. Prefer to
        # include a bottleneck edge, but occasionally perturb without guidance.
        steps = int(rng.integers(1, strength + 1))
        for _ in range(steps):
            now = list(edges)
            if len(now) < 2:
                break
            p = int(rng.integers(len(now)))
            if hot and rng.random() < 0.72:
                h = hot[int(rng.integers(len(hot)))]
                if h in edges:
                    p = now.index(h)
            q = int(rng.integers(len(now) - 1))
            if q >= p:
                q += 1
            a, b = now[p]
            c, d = now[q]
            if a == d or c == b or (a, d) in edges or (c, b) in edges:
                continue
            edges.remove((a, b))
            edges.remove((c, d))
            edges.add((a, d))
            edges.add((c, b))

        # Occasionally apply a three-edge rotation, a larger move not
        # equivalent to a single destination swap.
        if len(arcs) >= 3 and rng.random() < 0.30:
            now = list(edges)
            ids = rng.choice(len(now), size=3, replace=False)
            (a, b), (c, d), (e, f) = [now[int(i)] for i in ids]
            old = {(a, b), (c, d), (e, f)}
            for new in ({(a, d), (c, f), (e, b)},
                        {(a, f), (c, b), (e, d)}):
                if len(new) != 3 or any(x == y for x, y in new):
                    continue
                if new & (edges - old):
                    continue
                trial = (edges - old) | new
                if tools.is_valid(tuple(sorted(trial))):
                    edges = trial
                    break

        result = tuple(sorted(edges))
        return result if result != parent and tools.is_valid(result) else None

    while tools.lp_calls_left > 0:
        iteration += 1

        # Refresh a small elite pool; periodically restart from a good but
        # non-identical region to avoid spending all calls on one basin.
        if iteration % 45 == 1:
            ordered = sorted(scored, key=scored.get)
            archive = []
            for net in ordered:
                if all(dist(net, old) >= 4 for old in archive):
                    archive.append(net)
                if len(archive) >= 10:
                    break
            for net in ordered:
                if len(archive) >= 22:
                    break
                if net not in archive:
                    archive.append(net)
            if stall > 10 and len(archive) > 1 and rng.random() < 0.55:
                current = archive[int(rng.integers(min(10, len(archive))))]
                current_score = scored[current]
                stall = 0

        try:
            detail = tools.lp_detail(current)
        except Exception:
            detail = None

        strength = min(7, 1 + stall // 5)
        trials = []
        for _ in range(16):
            cand = mutate(current, detail, strength)
            if (cand is not None and cand not in scored
                    and not tools.is_cached(cand)):
                trials.append(cand)
        if not trials:
            # Vary a different elite parent if the present neighborhood is spent.
            parent = archive[int(rng.integers(len(archive)))] if archive else best_net
            if parent != current:
                current = parent
                current_score = scored[current]
            continue

        base_e = energy(current)
        trials = sorted(set(trials), key=lambda x: energy(x) - base_e)
        # Mostly evaluate surrogate-favored moves, with one randomized
        # candidate to hedge against Hamiltonian bias.
        batch = trials[:2]
        if len(trials) > 2 and rng.random() < 0.65:
            batch.append(trials[int(rng.integers(2, len(trials)))])
        previous_best = best_score
        for cand in batch:
            if tools.lp_calls_left <= 0:
                break
            try:
                value = evaluate(cand)
            except Exception as exc:
                if type(exc).__name__ == "BudgetExhausted":
                    return list(best_net) if best_net is not None else []
                continue

            # Annealing accepts occasional worsening moves, with a shrinking
            # temperature as the run progresses.
            if value < current_score:
                current, current_score = cand, value
                stall = 0
            else:
                scale = max(500.0, current_score * 0.0015)
                temp = scale * max(0.08, 1.0 - iteration / 2200.0)
                delta = value - current_score
                if temp > 0 and rng.random() < float(__import__("math").exp(-min(40.0, delta / temp))):
                    current, current_score = cand, value
                stall += 1

        if best_score < previous_best:
            stall = 0

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
