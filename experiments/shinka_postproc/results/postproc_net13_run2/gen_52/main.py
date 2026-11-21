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
    seen = {}
    best_net, best_score = None, float("inf")
    current = None
    current_score = float("inf")
    temperature = 0.012
    steps = 0

    def canon(arcs):
        return tuple(sorted(tuple(map(int, e)) for e in arcs))

    def evaluate(net):
        nonlocal best_net, best_score
        net = canon(net)
        if net not in seen:
            seen[net] = tools.lp(net)
        score = seen[net]
        if score < best_score:
            best_net, best_score = net, score
        return score

    def energy(net):
        try:
            return float(tools.energy(net))
        except Exception:
            return 0.0

    def dist(a, b):
        return len(set(a) ^ set(b))

    # Seed from repaired samples, balancing predicted quality and structural variety.
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

    ranked = sorted(repaired, key=energy)
    starts = []
    for net in ranked:
        if all(dist(net, old) >= 4 for old in starts):
            starts.append(net)
        if len(starts) >= min(28, len(ranked)):
            break
    for net in ranked:
        if len(starts) >= 36:
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
    if best_net is None:
        return list(starts[0])

    current, current_score = best_net, best_score

    def switch(cur, edge1, edge2):
        a, b = edge1
        c, d = edge2
        if a == d or c == b:
            return None
        old = set(cur)
        if (a, d) in old or (c, b) in old:
            return None
        old.remove(edge1)
        old.remove(edge2)
        old.add((a, d))
        old.add((c, b))
        return tuple(sorted(old))

    def random_kick(net, hot):
        cur = set(net)
        count = 1 + int(rng.integers(4))
        for _ in range(count):
            arcs = list(cur)
            if len(arcs) < 2:
                break
            # Bias toward a bottleneck edge sometimes, but retain unrestricted moves.
            if hot and rng.random() < 0.65:
                e1 = hot[int(rng.integers(len(hot)))]
                if e1 not in cur:
                    e1 = arcs[int(rng.integers(len(arcs)))]
            else:
                e1 = arcs[int(rng.integers(len(arcs)))]
            e2 = arcs[int(rng.integers(len(arcs)))]
            if e1 == e2:
                continue
            nxt = switch(tuple(sorted(cur)), e1, e2)
            if nxt is not None:
                cur = set(nxt)
        ans = tuple(sorted(cur))
        return ans if ans != net else None

    def relink_step(net, target):
        cur, goal = set(net), set(target)
        arcs = list(cur)
        overlap = len(cur & goal)
        options = []
        # Replace two current arcs so at least one missing target arc is inserted.
        for a, b in goal - cur:
            outgoing = [e for e in arcs if e[0] == a]
            incoming = [e for e in arcs if e[1] == b]
            for e1 in outgoing:
                for e2 in incoming:
                    if e1 == e2:
                        continue
                    nxt = switch(tuple(sorted(cur)), e1, e2)
                    if nxt is not None:
                        gain = len(set(nxt) & goal) - overlap
                        if gain > 0:
                            options.append((gain, nxt))
        if not options:
            return None
        maxgain = max(x[0] for x in options)
        choices = [x[1] for x in options if x[0] == maxgain]
        return choices[int(rng.integers(len(choices)))]

    while tools.lp_calls_left > 0:
        steps += 1
        # Draw a target from the best scored networks, often distinct from current.
        archive = sorted(seen, key=seen.get)[:min(24, len(seen))]
        targets = [x for x in archive if x != current and dist(x, current) >= 4]
        target = targets[int(rng.integers(len(targets)))] if targets else None

        try:
            detail = tools.lp_detail(current)
            prices = detail.get("prices", {})
            hot = [tuple(map(int, e)) for e, p in prices.items() if float(p) > 0]
            if not hot:
                loads = detail.get("loads", {})
                hot = [tuple(map(int, e)) for e, _ in
                       sorted(loads.items(), key=lambda z: float(z[1]), reverse=True)[:6]]
        except Exception:
            hot = []

        proposals = set()
        # A short directed segment toward a target, or a perturbative walk otherwise.
        path = current
        if target is not None and rng.random() < 0.55:
            for _ in range(1 + int(rng.integers(4))):
                path = relink_step(path, target)
                if path is None:
                    break
                proposals.add(path)

        for _ in range(24):
            base = current
            if proposals and rng.random() < 0.35:
                base = list(proposals)[int(rng.integers(len(proposals)))]
            cand = random_kick(base, hot)
            if cand is not None:
                proposals.add(cand)

        fresh = [x for x in proposals
                 if x not in seen and not tools.is_cached(x) and tools.is_valid(x)]
        if not fresh:
            # Escape exhausted neighborhoods by restarting from a different scored basin.
            if archive:
                current = archive[int(rng.integers(len(archive)))]
                current_score = seen[current]
            continue

        base_e = energy(current)
        # Use surrogate for ordering, but reserve a few trials for exploration.
        fresh.sort(key=lambda x: energy(x) - base_e)
        chosen = fresh[:2]
        tail = fresh[2:]
        if tail and len(chosen) < 3:
            chosen.append(tail[int(rng.integers(len(tail)))])
        for cand in chosen:
            if tools.lp_calls_left <= 0:
                break
            try:
                score = evaluate(cand)
            except Exception as exc:
                if type(exc).__name__ == "BudgetExhausted":
                    return list(best_net) if best_net is not None else []
                continue
            # Annealed acceptance for the walk; best-so-far is never discarded.
            if score < current_score:
                current, current_score = cand, score
            else:
                scale = max(1.0, current_score)
                if rng.random() < math.exp(-max(0.0, score - current_score) /
                                           max(1.0, temperature * scale)):
                    current, current_score = cand, score

        # Cool slowly, with periodic reheating to allow a new basin transition.
        temperature *= 0.9992
        if steps % 180 == 0:
            temperature = max(temperature, 0.009)

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
