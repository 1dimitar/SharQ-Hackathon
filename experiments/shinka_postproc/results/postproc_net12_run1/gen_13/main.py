"""Classical post-processing of QAOA output for QOBLIB 08. ShinkaEvolve edits the EVOLVE block."""

import json
import time

import numpy as np

# EVOLVE-BLOCK-START
import math


def improve(demand, candidates, tools, rng):
    """Repair and search QAOA samples for a low-congestion valid network."""
    n = int(tools.n)
    demand = np.asarray(demand, dtype=float)
    scores = {}
    elite = []

    def evaluate(net):
        """Score a network once locally; LP results are cached by tools as well."""
        net = tuple(sorted(net))
        if net in scores:
            return scores[net]
        if not tools.is_valid(net):
            scores[net] = float("inf")
            return scores[net]
        try:
            value = float(tools.lp(net))
        except Exception:
            # In particular, stop cleanly if the LP budget is exhausted.
            raise
        scores[net] = value
        return value

    def remember(net, value):
        nonlocal best_net, best_score
        if not np.isfinite(value):
            return
        if value < best_score:
            best_net, best_score = net, value
        if net not in elite:
            elite.append(net)
        elite.sort(key=lambda x: scores.get(x, float("inf")))
        del elite[14:]

    # Repair first, removing duplicate repaired topologies before spending LP calls.
    repaired = []
    for cand in candidates:
        net = tuple(sorted(tools.repair(cand)))
        if net not in repaired and tools.is_valid(net):
            repaired.append(net)

    if not repaired:
        net = tuple(sorted(tools.repair([])))
        repaired = [net]

    best_net = repaired[0]
    best_score = float("inf")
    for net in repaired:
        try:
            value = evaluate(net)
        except Exception:
            break
        remember(net, value)

    if not np.isfinite(best_score):
        return list(best_net)

    # Keep several good repaired samples available as diverse restart points.
    elite.sort(key=lambda x: scores.get(x, float("inf")))
    if not elite:
        elite = [best_net]

    def moves_for(net):
        """Return a ranked shortlist of feasible 2-exchange moves."""
        arcs = list(net)
        cur = set(net)
        detail = tools.lp_detail(net)
        prices = {tuple(k): float(v) for k, v in detail.get("prices", {}).items()}
        loads = {tuple(k): float(v) for k, v in detail.get("loads", {}).items()}
        positive = [v for v in prices.values() if v > 0]
        mean_price = float(np.mean(positive)) if positive else 1.0
        base_energy = float(tools.energy(net))

        moves = []
        for p in range(len(arcs)):
            a, b = arcs[p]
            for q in range(p + 1, len(arcs)):
                c, d = arcs[q]
                # Swap destinations; this preserves all in/out degrees.
                if a == d or c == b or (a, d) in cur or (c, b) in cur:
                    continue
                nxt = tuple(sorted((cur - {(a, b), (c, d)}) | {(a, d), (c, b)}))
                if nxt == net:
                    continue

                removed = ((a, b), (c, d))
                added = ((a, d), (c, b))
                relief = sum(max(0.0, prices.get(x, 0.0)) *
                             max(0.0, loads.get(x, 0.0)) for x in removed)
                added_proxy = mean_price * (
                    max(0.0, demand[a, d]) + max(0.0, demand[c, b])
                )
                hot = sum(1 for x in removed if prices.get(x, 0.0) > 0)
                energy_delta = float(tools.energy(nxt)) - base_energy
                demand_delta = (
                    demand[a, d] + demand[c, b] - demand[a, b] - demand[c, d]
                )
                moves.append((nxt, hot, relief - added_proxy,
                              energy_delta, float(demand_delta)))

        if not moves:
            return []

        # Use rank aggregation rather than combining quantities with incompatible
        # scales. Random tie-breaking ensures that repeated visits explore varied moves.
        def rank_order(index, reverse=False):
            order = sorted(range(len(moves)),
                           key=lambda i: moves[i][index],
                           reverse=reverse)
            ranks = [0] * len(moves)
            for r, i in enumerate(order):
                ranks[i] = r
            return ranks

        hot_rank = rank_order(1, reverse=True)
        relief_rank = rank_order(2, reverse=True)
        energy_rank = rank_order(3)
        demand_rank = rank_order(4)
        jitter = rng.random(len(moves))
        order = sorted(
            range(len(moves)),
            key=lambda i: (
                0.38 * hot_rank[i] +
                0.32 * relief_rank[i] +
                0.20 * energy_rank[i] +
                0.10 * demand_rank[i] +
                0.15 * jitter[i]
            )
        )

        # Evaluate a useful ranked core and a small exploratory sample.
        limit = min(14, len(order))
        selected = order[:limit]
        if len(order) > limit:
            extra = min(4, len(order) - limit)
            selected.extend(
                rng.choice(order[limit:], size=extra, replace=False).tolist()
            )
        return [moves[i][0] for i in selected]

    # Bounded iterated local search: exploitation from good networks, with
    # occasional kicked restarts from other elite samples.
    temperature = 1800.0
    iteration = 0
    while iteration < 5000:
        iteration += 1
        try:
            left = int(tools.lp_calls_left)
        except Exception:
            left = 1
        if left <= 0:
            break

        # Restart when stalled, or periodically to retain diversity.
        if iteration % 11 == 0 or (iteration % 5 == 0 and rng.random() < 0.35):
            source = elite[int(rng.integers(len(elite)))]
            current = source
            kicks = 1 + int(rng.integers(4))
            for _ in range(kicks):
                options = moves_for(current)
                if not options:
                    break
                current = options[int(rng.integers(len(options)))]
            try:
                current_score = evaluate(current)
            except Exception:
                break
            remember(current, current_score)
        else:
            current = best_net if iteration % 3 == 0 else elite[
                int(rng.integers(min(5, len(elite))))
            ]
            current_score = scores.get(current, best_score)

        options = moves_for(current)
        if not options:
            temperature *= 0.97
            continue

        evaluated = []
        for nxt in options:
            try:
                value = evaluate(nxt)
            except Exception:
                return list(best_net)
            evaluated.append((value, nxt))
            remember(nxt, value)
            try:
                if int(tools.lp_calls_left) <= 0:
                    break
            except Exception:
                pass

        if not evaluated:
            continue
        value, chosen = min(evaluated, key=lambda x: x[0])
        if value < current_score:
            current = chosen
        else:
            # Annealed acceptance allows non-improving transitions out of a basin.
            delta = value - current_score
            if rng.random() < math.exp(-min(delta, 1e6) / max(temperature, 1.0)):
                current = chosen

        temperature = max(180.0, temperature * 0.992)

    return list(best_net)


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
