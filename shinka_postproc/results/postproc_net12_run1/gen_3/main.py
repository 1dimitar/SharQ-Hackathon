"""Classical post-processing of QAOA output for QOBLIB 08. ShinkaEvolve edits the EVOLVE block."""

import json
import time

import numpy as np

# EVOLVE-BLOCK-START
import math


def improve(demand, candidates, tools, rng):
    n = int(tools.n)
    scored = {}
    best_net, best_score = None, float("inf")

    def canonical(arcs):
        return tuple(sorted(set(tuple(a) for a in arcs)))

    def evaluate(net):
        nonlocal best_net, best_score
        net = canonical(net)
        if not tools.is_valid(net):
            return float("inf")
        if net not in scored:
            try:
                scored[net] = tools.lp(net)
            except Exception as exc:
                if exc.__class__.__name__ == "BudgetExhausted":
                    return float("inf")
                raise
        score = scored[net]
        if score < best_score:
            best_net, best_score = net, score
        return score

    # Seed the search with repaired samples, preserving distinct topologies.
    starts = []
    for cand in candidates:
        net = canonical(tools.repair(cand))
        if net in starts or not tools.is_valid(net):
            continue
        starts.append(net)
        evaluate(net)

    if not starts:
        return []

    starts.sort(key=lambda x: scored.get(x, float("inf")))
    elite = starts[:min(8, len(starts))]
    states = [(x, scored.get(x, float("inf"))) for x in elite]
    visits = {}

    while tools.lp_calls_left > 0:
        # Rotate among elite states, occasionally restarting from a less-used one.
        if rng.random() < 0.18 or not states:
            base, base_score = elite[int(rng.integers(len(elite)))], None
            base_score = scored.get(base, float("inf"))
        else:
            idx = int(rng.integers(len(states)))
            base, base_score = states[idx]

        arcs = list(base)
        current = set(base)
        try:
            detail = tools.lp_detail(base)
            prices = detail.get("prices", {})
        except Exception:
            prices = {}

        # Dual prices identify links most likely to benefit from replacement.
        hot = set()
        for link, price in prices.items():
            try:
                if float(price) > 0:
                    hot.add(tuple(link))
            except (TypeError, ValueError):
                pass

        # Build degree-preserving 2-exchanges. Keep a broad fallback so a
        # state can still move when dual information is unavailable.
        proposals = set()
        pairs = [(i, j) for i in range(len(arcs)) for j in range(i + 1, len(arcs))]
        rng.shuffle(pairs)
        for i, j in pairs:
            a, b = arcs[i]
            c, d = arcs[j]
            if a == c or b == d:
                continue
            if hot and (a, b) not in hot and (c, d) not in hot:
                continue
            if a == d or c == b or (a, d) in current or (c, b) in current:
                continue
            cand = canonical((current - {(a, b), (c, d)}) | {(a, d), (c, b)})
            if cand != base:
                proposals.add(cand)

        if len(proposals) < 20:
            for i, j in pairs:
                a, b = arcs[i]
                c, d = arcs[j]
                if a == c or b == d:
                    continue
                if a == d or c == b or (a, d) in current or (c, b) in current:
                    continue
                cand = canonical((current - {(a, b), (c, d)}) | {(a, d), (c, b)})
                if cand != base:
                    proposals.add(cand)

        # Free surrogate evaluation orders the candidate batch before LP calls.
        ranked_moves = []
        for cand in proposals:
            if cand in scored or tools.is_cached(cand):
                continue
            try:
                energy = float(tools.energy(cand))
            except Exception:
                energy = 0.0
            ranked_moves.append((energy, cand))
        ranked_moves.sort(key=lambda x: x[0])

        # A mixture of low-energy and randomly selected moves limits surrogate bias.
        limit = min(18, len(ranked_moves))
        chosen = [x[1] for x in ranked_moves[:max(1, limit * 2 // 3)]]
        tail = ranked_moves[max(1, limit * 2 // 3):]
        if tail:
            take = min(limit - len(chosen), len(tail))
            if take > 0:
                ids = rng.choice(len(tail), size=take, replace=False)
                chosen.extend(tail[int(k)][1] for k in ids)

        best_move, best_move_score = None, float("inf")
        for cand in chosen:
            s = evaluate(cand)
            if s < best_move_score:
                best_move, best_move_score = cand, s
            if tools.lp_calls_left <= 0:
                break

        if best_move is not None and best_move_score < float("inf"):
            visits[best_move] = visits.get(best_move, 0) + 1
            # Annealed acceptance explores plateaus and occasional uphill moves.
            temperature = max(150.0, 1800.0 * (0.985 ** sum(visits.values())))
            delta = best_move_score - base_score
            if delta <= 0 or rng.random() < math.exp(-min(40.0, delta / temperature)):
                states.append((best_move, best_move_score))
                states.sort(key=lambda x: x[1])
                states = states[:8]

        # Refresh a diverse elite archive using every topology scored so far.
        if len(scored) % 40 == 0:
            ranked = sorted(scored, key=scored.get)
            fresh = []
            for net in ranked:
                if all(len(set(net) ^ set(other)) >= 4 for other in fresh):
                    fresh.append(net)
                if len(fresh) >= 8:
                    break
            if fresh:
                elite = fresh
                for net in fresh:
                    if all(net != state[0] for state in states):
                        states.append((net, scored[net]))
                states.sort(key=lambda x: x[1])
                states = states[:8]

        # If the current neighborhood has no new moves, perturb by selecting
        # another archive member on the next iteration.
        if not chosen and len(states) <= 1:
            break

    return list(best_net) if best_net is not None else list(starts[0])
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
