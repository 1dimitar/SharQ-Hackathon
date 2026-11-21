"""Classical post-processing of QAOA output for QOBLIB 08. ShinkaEvolve edits the EVOLVE block."""

import json
import time

import numpy as np

# EVOLVE-BLOCK-START
def improve(demand, candidates, tools, rng):
    n = tools.n
    k = 2 * n
    seen = set()
    scored = {}
    details = {}

    def canon(arcs):
        return tuple(sorted(arcs))

    def evaluate(net):
        net = canon(net)
        if net not in scored:
            if tools.lp_calls_left <= 0:
                return float("inf")
            try:
                scored[net] = tools.lp(net)
            except Exception as exc:
                if type(exc).__name__ == "BudgetExhausted":
                    raise
                scored[net] = float("inf")
        return scored[net]

    # Repairs are free: rank them cheaply and score a diversified set.
    repaired = set()
    for cand in candidates:
        try:
            net = canon(tools.repair(cand))
            if tools.is_valid(net):
                repaired.add(net)
        except Exception:
            pass
    if not repaired:
        net = canon(tools.repair([]))
        if tools.is_valid(net):
            repaired.add(net)

    ordered = sorted(repaired, key=lambda x: tools.energy(x))
    starts = []
    for net in ordered:
        if all(len(set(net) & set(old)) < k - 2 for old in starts):
            starts.append(net)
        if len(starts) >= 20:
            break
    for net in ordered:
        if len(starts) >= 32:
            break
        if net not in starts:
            starts.append(net)
    if not starts:
        return []

    for net in starts:
        if tools.lp_calls_left <= 0:
            break
        try:
            evaluate(net)
        except Exception as exc:
            if type(exc).__name__ == "BudgetExhausted":
                break
    if not scored:
        return starts[0]

    def best_items():
        return sorted(scored, key=scored.get)

    best = best_items()[0]
    elite = best_items()[:8]

    def get_hot(net):
        if net not in details:
            try:
                details[net] = tools.lp_detail(net)
            except Exception as exc:
                if type(exc).__name__ == "BudgetExhausted":
                    raise
                details[net] = {}
        d = details[net]
        prices = d.get("prices", {})
        hot = {tuple(a) for a, p in prices.items() if p > 0}
        if not hot:
            loads = d.get("loads", {})
            vals = sorted(loads.values())
            if vals:
                cutoff = vals[int(0.65 * (len(vals) - 1))]
                hot = {tuple(a) for a, v in loads.items() if v >= cutoff}
        return hot

    def make_moves(net, hot, limit=100):
        arcs = list(net)
        cur = set(net)
        moves = set()

        # Complete two-arc exchange neighborhood, prioritizing changes to hot links.
        pairlist = []
        for p in range(len(arcs)):
            a, b = arcs[p]
            for q in range(p + 1, len(arcs)):
                c, d = arcs[q]
                if (arcs[p] not in hot and arcs[q] not in hot) and hot:
                    continue
                if a == d or c == b or (a, d) in cur or (c, b) in cur:
                    continue
                pairlist.append(canon((cur - {(a, b), (c, d)}) | {(a, d), (c, b)}))
        moves.update(pairlist)

        # Targeted cyclic destination reassignment, including paths through non-hot arcs.
        if len(arcs) >= 3:
            for _ in range(100):
                ix = rng.choice(len(arcs), size=3, replace=False)
                (a, b), (c, d), (e, f) = (arcs[int(i)] for i in ix)
                old = {(a, b), (c, d), (e, f)}
                if hot and not (old & hot):
                    continue
                new = {(a, d), (c, f), (e, b)}
                if any(x == y for x, y in new) or len(new) != 3:
                    continue
                if new & (cur - old):
                    continue
                moves.add(canon((cur - old) | new))
        return [m for m in moves if tools.is_valid(m)]

    iteration = 0
    while tools.lp_calls_left > 0:
        iteration += 1
        elite = best_items()[:8]
        if iteration % 5 == 0:
            base = best
        else:
            base = elite[int(rng.integers(len(elite)))]

        try:
            hot = get_hot(base)
        except Exception as exc:
            if type(exc).__name__ == "BudgetExhausted":
                break
            hot = set()

        moves = make_moves(base, hot)
        fresh = [m for m in moves if m not in scored and not tools.is_cached(m)]
        if not fresh and hot:
            moves = make_moves(base, set())
            fresh = [m for m in moves if m not in scored and not tools.is_cached(m)]
        if fresh:
            # Build a competitive shortlist, then avoid spending the LP batch on
            # near-duplicate topologies. Energy rank still matters, but diversity
            # helps explore distinct network structures.
            fresh.sort(key=lambda m: tools.energy(m))
            shortlist = fresh[:min(48, len(fresh))]
            batch = []
            ranks = {m: i for i, m in enumerate(shortlist)}
            while shortlist and len(batch) < 8:
                if not batch:
                    chosen = shortlist[0]
                else:
                    chosen = max(
                        shortlist,
                        key=lambda m: (
                            min(len(set(m).symmetric_difference(set(old)))
                                for old in batch) - 0.20 * ranks[m]
                        ),
                    )
                batch.append(chosen)
                shortlist.remove(chosen)

            selected = set(batch)
            tail = [m for m in fresh if m not in selected]
            if tail:
                count = min(3, len(tail))
                batch.extend(tail[int(i)] for i in rng.choice(len(tail), size=count, replace=False))
            for cand in batch:
                if tools.lp_calls_left <= 0:
                    break
                try:
                    evaluate(cand)
                except Exception as exc:
                    if type(exc).__name__ == "BudgetExhausted":
                        break
                if scored[cand] < scored[best]:
                    best = cand

        # Kick the incumbent with a short sequence of valid exchanges, biased toward bottlenecks.
        if iteration % 3 == 0 and tools.lp_calls_left > 0:
            kick = set(best)
            try:
                khot = get_hot(best)
            except Exception:
                khot = set()
            for _ in range(2 + int(rng.integers(4))):
                arcs = list(kick)
                if len(arcs) < 2:
                    break
                pairs = [(p, q) for p in range(len(arcs)) for q in range(p + 1, len(arcs))]
                if khot:
                    focused = [(p, q) for p, q in pairs
                               if arcs[p] in khot or arcs[q] in khot]
                    if focused:
                        pairs = focused
                p, q = pairs[int(rng.integers(len(pairs)))]
                a, b = arcs[p]
                c, d = arcs[q]
                if a != d and c != b and (a, d) not in kick and (c, b) not in kick:
                    kick.remove((a, b))
                    kick.remove((c, d))
                    kick.update(((a, d), (c, b)))
            kn = canon(kick)
            if tools.is_valid(kn) and kn not in scored and not tools.is_cached(kn):
                try:
                    evaluate(kn)
                    if scored[kn] < scored[best]:
                        best = kn
                except Exception as exc:
                    if type(exc).__name__ == "BudgetExhausted":
                        break

    return best


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
