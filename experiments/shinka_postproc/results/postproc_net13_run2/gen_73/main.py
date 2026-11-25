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
    """Search repaired networks with bottleneck moves and archive path relinking."""
    n = tools.n
    scores = {}
    visits = {}
    stalls = {}
    best_net = None
    best_score = float("inf")
    rounds = 0

    def canon(arcs):
        return tuple(sorted(tuple(map(int, a)) for a in arcs))

    def dist(a, b):
        return len(set(a) ^ set(b))

    def energy(net):
        try:
            return float(tools.energy(net))
        except Exception:
            return 0.0

    def evaluate(net):
        nonlocal best_net, best_score
        net = canon(net)
        if net not in scores:
            scores[net] = tools.lp(net)
        if scores[net] < best_score:
            best_net, best_score = net, scores[net]
        return scores[net]

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

    ordered = sorted(repaired, key=energy)
    seeds = []
    for net in ordered:
        if all(dist(net, old) >= 4 for old in seeds):
            seeds.append(net)
        if len(seeds) >= 18:
            break
    for net in ordered:
        if len(seeds) >= 24:
            break
        if net not in seeds:
            seeds.append(net)

    for net in seeds:
        if tools.lp_calls_left <= 0:
            break
        try:
            evaluate(net)
        except Exception as exc:
            if type(exc).__name__ == "BudgetExhausted":
                break
    if not scores:
        return list(seeds[0]) if seeds else []

    def archive():
        ranked = sorted(scores, key=scores.get)
        out = []
        for x in ranked:
            if all(dist(x, y) >= 4 for y in out):
                out.append(x)
            if len(out) >= 18:
                break
        for x in ranked:
            if len(out) >= 32:
                break
            if x not in out:
                out.append(x)
        return out

    def choose_parent(pop):
        # Rank-biased selection with a strong under-visited bonus.
        weights = []
        for rank, net in enumerate(pop):
            weights.append((1.0 / (1 + rank) ** 0.65) /
                           (1 + visits.get(net, 0)) ** 0.55)
        pick = float(rng.random()) * sum(weights)
        for net, w in zip(pop, weights):
            pick -= w
            if pick <= 0:
                return net
        return pop[0]

    def switch(net, p, q):
        arcs = list(net)
        a, b = arcs[p]
        c, d = arcs[q]
        if a == d or c == b:
            return None
        cur = set(net)
        x, y = (a, d), (c, b)
        if x in cur or y in cur:
            return None
        cur.remove((a, b))
        cur.remove((c, d))
        cur.add(x)
        cur.add(y)
        return tuple(sorted(cur))

    def path_moves(parent, target, cap=5):
        """Greedily add target arcs through valid degree-preserving switches."""
        target = set(target)
        path = parent
        produced = []
        for _ in range(cap):
            cur = set(path)
            overlap = len(cur & target)
            arcs = list(path)
            options = []
            for p in range(len(arcs)):
                for q in range(p + 1, len(arcs)):
                    cand = switch(path, p, q)
                    if cand is None:
                        continue
                    gain = len(set(cand) & target) - overlap
                    if gain > 0:
                        options.append((gain, cand))
            if not options:
                break
            top = max(g for g, _ in options)
            choices = [x for g, x in options if g == top]
            path = choices[int(rng.integers(len(choices)))]
            produced.append(path)
        return produced

    def propose(parent, detail, pop):
        arcs = list(parent)
        cur = set(parent)
        moves = set()
        prices = detail.get("prices", {}) if detail else {}
        hot = {tuple(map(int, a)) for a, p in prices.items() if float(p) > 0}
        loads = detail.get("loads", {}) if detail else {}
        if loads:
            ranked = sorted(((float(v), tuple(map(int, a))) for a, v in loads.items()),
                            reverse=True)
            hot.update(a for _, a in ranked[:5])

        for p in range(len(arcs)):
            for q in range(p + 1, len(arcs)):
                if hot and arcs[p] not in hot and arcs[q] not in hot:
                    continue
                cand = switch(parent, p, q)
                if cand is not None:
                    moves.add(cand)

        for _ in range(100):
            p = int(rng.integers(len(arcs)))
            q = int(rng.integers(len(arcs) - 1))
            if q >= p:
                q += 1
            cand = switch(parent, p, q)
            if cand is not None:
                moves.add(cand)

        # Random three-arc rotations, including some not incident to a hot link.
        for _ in range(75):
            ids = rng.choice(len(arcs), size=3, replace=False)
            (a, b), (c, d), (e, f) = (arcs[int(i)] for i in ids)
            old = {(a, b), (c, d), (e, f)}
            if hot and not (old & hot) and rng.random() < 0.75:
                continue
            for new in ({(a, d), (c, f), (e, b)},
                        {(a, f), (c, b), (e, d)}):
                if len(new) == 3 and all(x != y for x, y in new) and not (new & (cur - old)):
                    moves.add(tuple(sorted((cur - old) | new)))

        # Relink periodically, or more often when local progress stalls.
        if len(pop) > 1 and (rounds % 3 == 0 or stalls.get(parent, 0) >= 2):
            targets = [x for x in pop if x != parent and dist(x, parent) >= 6]
            if targets:
                targets.sort(key=lambda x: scores.get(x, float("inf")))
                target = targets[int(rng.integers(min(8, len(targets))))]
                moves.update(path_moves(parent, target, 5))

        return [x for x in moves if x != parent and x not in scores
                and not tools.is_cached(x) and tools.is_valid(x)]

    def select_batch(moves, parent, limit=3):
        if not moves:
            return []
        base = energy(parent)
        ranked = sorted(moves, key=lambda x: energy(x) - base)
        selected = []
        # Prefer surrogate improvement while avoiding near-duplicate trials.
        for x in ranked:
            if all(dist(x, y) >= 2 for y in selected):
                selected.append(x)
            if len(selected) >= 2:
                break
        # One exploratory slot, chosen from a wider ranked portion.
        remaining = [x for x in ranked if x not in selected]
        head = remaining[:max(20, len(remaining) // 3)]
        if head and len(selected) < limit:
            selected.append(head[int(rng.integers(len(head)))])
        for x in ranked:
            if len(selected) >= limit:
                break
            if x not in selected:
                selected.append(x)
        return selected[:limit]

    while tools.lp_calls_left > 0:
        rounds += 1
        pop = archive()
        parent = choose_parent(pop)
        visits[parent] = visits.get(parent, 0) + 1
        try:
            detail = tools.lp_detail(parent)
        except Exception:
            detail = None
        batch = select_batch(propose(parent, detail, pop), parent)

        if not batch:
            # Try lightly visited archive members before giving up.
            for alt in sorted((x for x in pop if x != parent),
                              key=lambda x: visits.get(x, 0))[:10]:
                try:
                    det = tools.lp_detail(alt)
                except Exception:
                    det = None
                batch = select_batch(propose(alt, det, pop), alt)
                if batch:
                    parent = alt
                    break
        if not batch:
            break

        old_best = best_score
        parent_score = scores.get(parent, float("inf"))
        for net in batch:
            if tools.lp_calls_left <= 0:
                break
            try:
                evaluate(net)
            except Exception as exc:
                if type(exc).__name__ == "BudgetExhausted":
                    return list(best_net) if best_net is not None else []
        child_best = min((scores.get(x, float("inf")) for x in batch),
                         default=float("inf"))
        stalls[parent] = 0 if child_best < parent_score else stalls.get(parent, 0) + 1
        if best_score < old_best:
            stalls[parent] = 0

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
