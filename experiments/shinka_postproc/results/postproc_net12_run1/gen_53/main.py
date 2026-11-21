"""Classical post-processing of QAOA output for QOBLIB 08. ShinkaEvolve edits the EVOLVE block."""

import json
import time

import numpy as np

# EVOLVE-BLOCK-START
def improve(demand, candidates, tools, rng):
    n = tools.n
    scores = {}
    best_net = None
    best_score = float("inf")
    archive = []
    max_archive = 18
    batch_size = 6

    def canon(arcs):
        return tuple(sorted(tuple(map(int, a)) for a in arcs))

    def record(net):
        nonlocal best_net, best_score
        net = canon(net)
        if net in scores:
            return scores[net]
        if not tools.is_valid(net) or tools.lp_calls_left <= 0:
            return float("inf")
        value = float(tools.lp(net))
        scores[net] = value
        if value < best_score:
            best_net, best_score = net, value
        return value

    # Repair all samples cheaply, then spend initial evaluations on a mixture
    # of low-energy samples and structurally distinct alternatives.
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
            repaired.add(canon(tools.repair([])))
        except Exception:
            return []

    def energy(net):
        try:
            return float(tools.energy(net))
        except Exception:
            return 0.0

    def dist(a, b):
        return len(set(a) ^ set(b))

    ordered = sorted(repaired, key=energy)
    seeds = []
    for net in ordered:
        if all(dist(net, old) >= 4 for old in seeds):
            seeds.append(net)
        if len(seeds) >= min(14, len(ordered)):
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
            record(net)
        except Exception as exc:
            if type(exc).__name__ == "BudgetExhausted":
                break
            raise
    if best_net is None:
        return []

    def refresh():
        nonlocal archive
        ranked = sorted(scores, key=scores.get)
        selected = ranked[:min(5, len(ranked))]
        rest = ranked[len(selected):]
        while rest and len(selected) < max_archive:
            # Max-min selection, mildly penalizing inferior scores.
            pick = max(
                rest,
                key=lambda x: min(dist(x, y) for y in selected)
                - 0.15 * (scores[x] - scores[ranked[0]])
                  / max(1.0, abs(scores[ranked[0]]))
            )
            selected.append(pick)
            rest.remove(pick)
        archive = selected

    def switch(cur, e1, e2):
        a, b = e1
        c, d = e2
        x, y = (a, d), (c, b)
        if a == d or c == b or x in cur or y in cur:
            return None
        return frozenset((cur - {e1, e2}) | {x, y})

    def relink_moves(parent, target):
        """Generate switches that replace parent-only links with target links."""
        cur = set(parent)
        wanted = set(target) - cur
        missing = list(wanted)
        rng.shuffle(missing)
        out = set()
        # A target edge a->d can be introduced by switching out a->b and
        # c->d; the reciprocal edge preserves both in- and out-degrees.
        for x, d in missing:
            outgoing = [e for e in cur if e[0] == x]
            incoming = [e for e in cur if e[1] == d]
            rng.shuffle(outgoing)
            rng.shuffle(incoming)
            for e1 in outgoing:
                for e2 in incoming:
                    if e1 == e2:
                        continue
                    nxt = switch(cur, e1, e2)
                    if nxt is not None:
                        net = tuple(sorted(nxt))
                        if net != parent and net not in scores and tools.is_valid(net):
                            out.add(net)
        return out

    def random_walk(net, steps):
        cur = set(net)
        for _ in range(steps):
            arcs = list(cur)
            if len(arcs) < 2:
                break
            for _ in range(18):
                i, j = rng.choice(len(arcs), size=2, replace=False)
                nxt = switch(cur, arcs[int(i)], arcs[int(j)])
                if nxt is not None:
                    cur = set(nxt)
                    break
        return tuple(sorted(cur))

    refresh()
    it = 0
    while tools.lp_calls_left > 0 and it < 10000:
        it += 1
        refresh()
        # Alternate strong exploitation and archive-to-archive relinking.
        if len(archive) > 1 and rng.random() < 0.72:
            parent = archive[int(rng.integers(min(7, len(archive))))]
            choices = [x for x in archive if x != parent]
            target = choices[int(rng.integers(len(choices)))]
        else:
            parent = archive[int(rng.integers(len(archive)))]
            target = archive[int(rng.integers(len(archive)))]

        try:
            detail = tools.lp_detail(parent)
        except Exception:
            detail = {}
        prices = {tuple(k): float(v) for k, v in detail.get("prices", {}).items()}
        loads = {tuple(k): float(v) for k, v in detail.get("loads", {}).items()}

        proposals = relink_moves(parent, target) if target != parent else set()
        cur = set(parent)
        arcs = list(parent)

        # Dual-guided switches are a bias, not a restriction.
        hot = sorted(arcs, key=lambda e: (prices.get(e, 0.0), loads.get(e, 0.0)),
                     reverse=True)[:max(4, n // 2)]
        pairs = [(i, j) for i in range(len(arcs)) for j in range(i + 1, len(arcs))]
        rng.shuffle(pairs)
        for i, j in pairs[:min(100, len(pairs))]:
            nxt = switch(cur, arcs[i], arcs[j])
            if nxt is not None:
                net = tuple(sorted(nxt))
                if net not in scores and tools.is_valid(net):
                    proposals.add(net)
        # Random multi-switch proposals occasionally cross local barriers.
        for _ in range(12):
            net = random_walk(parent, 2 + int(rng.integers(4)))
            if net != parent and net not in scores and tools.is_valid(net):
                proposals.add(net)

        proposals = [x for x in proposals
                     if x not in scores and not tools.is_cached(x)]
        if not proposals:
            continue

        pset = set(parent)
        # Rank by complementary free signals: dual savings, surrogate energy,
        # and a mild topological-distance term to retain relinking diversity.
        def rank(net):
            ns = set(net)
            removed, added = pset - ns, ns - pset
            dual = sum(prices.get(e, 0.0) for e in removed) - sum(
                prices.get(e, 0.0) for e in added)
            return (energy(net), -dual, -dist(net, parent), float(rng.random()))

        proposals.sort(key=rank)
        chosen = []
        # Take leading proposals, but diversify the batch by edge distance.
        for net in proposals:
            if all(dist(net, old) >= 2 for old in chosen):
                chosen.append(net)
            if len(chosen) >= batch_size - 1:
                break
        # Reserve one slot for a less predictable alternative.
        remaining = [x for x in proposals if x not in chosen]
        if remaining:
            chosen.append(remaining[int(rng.integers(len(remaining)))])
        for net in chosen:
            if tools.lp_calls_left <= 0:
                break
            try:
                record(net)
            except Exception as exc:
                if type(exc).__name__ == "BudgetExhausted":
                    return list(best_net) if best_net is not None else []
                raise

    return list(best_net) if best_net is not None else []
# EVOLVE-BLOCK-END


def run_postproc(pool_path, lp_budget, seed, control=False, time_limit=150):
    """Fixed entry point called by the evaluator: one run on one candidate pool."""
    import signal

    import oracle
    from hamiltonian import surrogate_terms

    pool = json.load(open(pool_path))
    import qoblib_tools as qt

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
