"""Classical post-processing of QAOA output for QOBLIB 08. ShinkaEvolve edits the EVOLVE block."""

import json
import time

import numpy as np

# EVOLVE-BLOCK-START
import math


def improve(demand, candidates, tools, rng):
    """Improve repaired QAOA networks with bottleneck-guided annealed search."""
    n = tools.n
    pool = {}
    best_net = None
    best_score = float("inf")

    def record(net, score):
        nonlocal best_net, best_score
        net = tuple(sorted(net))
        if score < best_score:
            best_net, best_score = net, score
        pool[net] = min(score, pool.get(net, float("inf")))

    # Seed with repaired samples, but avoid spending LP calls on duplicate repairs.
    for cand in candidates:
        net = tuple(sorted(tools.repair(cand)))
        if net in pool:
            continue
        if not tools.is_valid(net):
            continue
        try:
            score = tools.lp(net)
        except Exception:
            break
        record(net, score)

    if not pool:
        return []

    ranked = sorted(pool, key=pool.get)
    # A few diverse starts are more useful than many near-identical elite starts.
    starts = []
    for net in ranked:
        if all(len(set(net) ^ set(other)) >= 4 for other in starts):
            starts.append(net)
        if len(starts) >= 8:
            break
    if not starts:
        starts = [ranked[0]]

    def propose(net, detail, count=36):
        arcs = list(net)
        aset = set(net)
        priced = set()
        if detail:
            prices = detail.get("prices", {})
            priced = {tuple(a) for a, p in prices.items() if p > 0}

        out = set()
        # Generate degree-preserving two-switches; bias the sampled pairs toward
        # links whose LP dual price says they are congestion-critical.
        for _ in range(count * 3):
            if len(out) >= count:
                break
            if priced and rng.random() < 0.72:
                eligible = [i for i, a in enumerate(arcs) if a in priced]
                p = int(rng.choice(eligible)) if eligible else int(rng.integers(len(arcs)))
            else:
                p = int(rng.integers(len(arcs)))
            q = int(rng.integers(len(arcs) - 1))
            if q >= p:
                q += 1
            (a, b), (c, d) = arcs[p], arcs[q]
            if a == d or c == b or (a, d) in aset or (c, b) in aset:
                continue
            cand = tuple(sorted((aset - {(a, b), (c, d)}) | {(a, d), (c, b)}))
            if cand != net:
                out.add(cand)

        # Occasional three-cycle rotations enlarge the neighborhood without
        # disturbing any node's in- or out-degree.
        for _ in range(max(2, count // 5)):
            ids = rng.choice(len(arcs), size=3, replace=False)
            (a, b), (c, d), (e, f) = [arcs[int(i)] for i in ids]
            variants = ((a, d), (c, f), (e, b))
            if len({x for x, _ in variants}) < 3 or len({y for _, y in variants}) < 3:
                continue
            if any(x == y for x, y in variants):
                continue
            remain = aset - {(a, b), (c, d), (e, f)}
            if any(x in remain for x in variants):
                continue
            cand = tuple(sorted(remain | set(variants)))
            if cand != net:
                out.add(cand)

        # Free surrogate ranking; retain variety so the LP does not see only
        # tiny energy changes around a single topology.
        scored = []
        base_e = tools.energy(net)
        for cand in out:
            try:
                e = tools.energy(cand)
            except Exception:
                e = base_e
            scored.append((e - base_e, cand))
        scored.sort(key=lambda x: x[0])
        keep = max(8, min(len(scored), count // 2))
        head = [x[1] for x in scored[:keep]]
        tail = [x[1] for x in scored[keep:]]
        if tail:
            rng.shuffle(tail)
            head.extend(tail[:max(0, count - len(head))])
        return head

    # Each walk cools gradually, but reheats after stagnation. Walks start from
    # different elite networks and occasionally from the global best.
    walk = 0
    while tools.lp_calls_left > 0:
        if walk < len(starts):
            cur = starts[walk]
        elif rng.random() < 0.55:
            cur = best_net
        else:
            cur = starts[int(rng.integers(len(starts)))]
        cur_score = pool.get(cur, best_score)
        try:
            detail = tools.lp_detail(cur)
        except Exception:
            detail = None
        temperature = max(150.0, cur_score * (0.0015 if walk % 3 else 0.004))
        stale = 0

        for step in range(100):
            if tools.lp_calls_left <= 0:
                break
            moves = propose(cur, detail, 34)
            if not moves:
                break

            # Evaluate energy-favored moves first, with a small randomized
            # ordering to prevent deterministic lock-in.
            rng.shuffle(moves)
            evaluated = []
            for cand in moves:
                if cand in pool or tools.is_cached(cand):
                    continue
                if not tools.is_valid(cand):
                    continue
                try:
                    s = tools.lp(cand)
                except Exception:
                    break
                record(cand, s)
                evaluated.append((s, cand))
                if tools.lp_calls_left <= 0:
                    break
            if not evaluated:
                stale += 1
                if stale > 4:
                    break
                temperature *= 1.35
                continue

            evaluated.sort(key=lambda x: x[0])
            # Usually take the best neighbor, but occasionally select another
            # and accept uphill moves according to the annealing temperature.
            chosen = None
            for s, cand in evaluated[:min(5, len(evaluated))]:
                delta = s - cur_score
                if delta <= 0 or rng.random() < math.exp(-min(40.0, delta / max(1.0, temperature))):
                    chosen = (s, cand)
                    break
            if chosen is None:
                stale += 1
                temperature *= 0.88
                if stale > 8:
                    break
            else:
                cur_score, cur = chosen
                stale = 0
                temperature *= 0.94
                if cur_score < best_score:
                    best_net, best_score = cur, cur_score
                    try:
                        detail = tools.lp_detail(cur)
                    except Exception:
                        detail = None
        walk += 1

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
