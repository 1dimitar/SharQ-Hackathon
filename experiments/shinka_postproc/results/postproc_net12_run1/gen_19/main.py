"""Classical post-processing of QAOA output for QOBLIB 08. ShinkaEvolve edits the EVOLVE block."""

import json
import time

import numpy as np

# EVOLVE-BLOCK-START
import math


def improve(demand, candidates, tools, rng):
    """Search repaired starts with dual-ranked moves and archive-based restarts."""
    n = tools.n
    edge_count = 2 * n
    scores = {}
    details = {}
    archive = []
    best_net = None
    best_score = float("inf")
    current = None
    current_score = float("inf")

    def canon(arcs):
        return tuple(sorted((int(a), int(b)) for a, b in arcs))

    def edge_distance(x, y):
        return len(set(x).symmetric_difference(y))

    def remember(net, score):
        nonlocal best_net, best_score
        if score < best_score:
            best_net, best_score = net, score

        for old, _ in archive:
            if net == old:
                return

        # Keep good but meaningfully different states for restarts.
        if archive:
            near = min(edge_distance(net, old) for old, _ in archive)
            if near < 4 and score > min(s for _, s in archive) * 1.001:
                return

        archive.append((net, score))
        archive.sort(key=lambda item: item[1])
        del archive[12:]

    def evaluate(net):
        net = canon(net)
        if net in scores:
            return scores[net]
        if not tools.is_valid(net):
            return float("inf")
        value = float(tools.lp(net))
        scores[net] = value
        remember(net, value)
        return value

    # Build a start set that mixes low-energy repairs and structurally diverse repairs.
    repaired = set()
    for cand in candidates:
        try:
            net = canon(tools.repair(cand))
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
            pass
    if not repaired:
        return []

    by_energy = sorted(repaired, key=tools.energy)
    starts = []
    # First take a diverse energy-ranked prefix.
    for net in by_energy:
        if all(edge_distance(net, old) >= 4 for old in starts):
            starts.append(net)
        if len(starts) >= 14:
            break
    # Then fill from the remaining energy-ranked repairs.
    for net in by_energy:
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
            raise

    if not scores:
        return starts[0]

    current = best_net
    current_score = best_score

    def get_detail(net):
        if net not in details:
            try:
                details[net] = tools.lp_detail(net)
            except Exception:
                details[net] = {}
        return details[net]

    def link_data(net):
        detail = get_detail(net)
        prices = {}
        loads = {}
        for edge, value in detail.get("prices", {}).items():
            try:
                prices[tuple(edge)] = float(value)
            except Exception:
                pass
        for edge, value in detail.get("loads", {}).items():
            try:
                loads[tuple(edge)] = float(value)
            except Exception:
                pass
        hot = {edge for edge, price in prices.items() if price > 1e-10}
        if not hot and loads:
            peak = max(loads.values())
            hot = {edge for edge, load in loads.items() if load >= 0.94 * peak}
        return prices, loads, hot

    def generate_moves(net, prices, hot):
        """Enumerate degree-preserving two-switches and rank by dual relief."""
        arcs = list(net)
        edge_set = set(net)
        old_energy = tools.energy(net)
        moves = {}
        for i, (a, b) in enumerate(arcs):
            for c, d in arcs[i + 1:]:
                # Swapping destinations preserves both in- and out-degrees.
                x, y = (a, d), (c, b)
                if a == d or c == b or x == y:
                    continue
                if x in edge_set or y in edge_set:
                    continue
                candidate = canon((edge_set - {(a, b), (c, d)}) | {x, y})
                if candidate in moves:
                    continue

                removed = ((a, b), (c, d))
                added = (x, y)
                dual_gain = sum(prices.get(e, 0.0) for e in removed)
                dual_gain -= sum(prices.get(e, 0.0) for e in added)

                # Added traffic and surrogate change are inexpensive tie-breakers.
                add_demand = float(demand[x[0], x[1]]) + float(demand[y[0], y[1]])
                energy_delta = float(tools.energy(candidate)) - float(old_energy)
                touches_hot = bool(set(removed) & hot)
                moves[candidate] = (
                    dual_gain, touches_hot, energy_delta, add_demand
                )
        return moves

    def kick(net, count, hot):
        """Apply a short random sequence of valid two-switches."""
        for _ in range(count):
            arcs = list(net)
            edge_set = set(net)
            possible = []
            for i, (a, b) in enumerate(arcs):
                for c, d in arcs[i + 1:]:
                    if hot and (a, b) not in hot and (c, d) not in hot:
                        continue
                    x, y = (a, d), (c, b)
                    if a != d and c != b and x != y and x not in edge_set and y not in edge_set:
                        possible.append((a, b, c, d, x, y))
            if not possible:
                continue
            a, b, c, d, x, y = possible[int(rng.integers(len(possible)))]
            net = canon((edge_set - {(a, b), (c, d)}) | {x, y})
        return net

    iteration = 0
    stale = 0
    while tools.lp_calls_left > 0:
        iteration += 1
        previous_best = best_score

        # Draw mostly from the current trajectory, with periodic archive diversification.
        if archive and (stale >= 5 or rng.random() < 0.24):
            pick = int(rng.integers(min(len(archive), 6 if stale < 10 else len(archive))))
            source, source_score = archive[pick]
        else:
            source, source_score = current, current_score

        prices, _, hot = link_data(source)
        moves = generate_moves(source, prices, hot)
        fresh = [
            (net, rank) for net, rank in moves.items()
            if net not in scores and not tools.is_cached(net)
        ]

        # If the focused neighborhood is exhausted, broaden the search with another archive state.
        if not fresh and archive:
            source, source_score = archive[int(rng.integers(len(archive)))]
            prices, _, hot = link_data(source)
            moves = generate_moves(source, prices, hot)
            fresh = [
                (net, rank) for net, rank in moves.items()
                if net not in scores and not tools.is_cached(net)
            ]
        if not fresh:
            if archive:
                source, _ = archive[int(rng.integers(len(archive)))]
                _, _, hot = link_data(source)
                current = kick(source, 2 + int(rng.integers(4)), hot)
                current_score = scores.get(current, source_score)
            continue

        # Directional dual benefit is primary; energy and traffic break ties.
        fresh.sort(key=lambda item: (
            -item[1][0],
            not item[1][1],
            item[1][2],
            item[1][3],
        ))

        # Evaluate a narrow promising beam plus a little randomized exploration.
        batch = [net for net, _ in fresh[:5]]
        tail = fresh[5:]
        if tail and rng.random() < 0.75:
            batch.append(tail[int(rng.integers(len(tail)))][0])

        evaluated = []
        for net in batch:
            if tools.lp_calls_left <= 0:
                break
            try:
                value = evaluate(net)
                evaluated.append((net, value))
            except Exception as exc:
                if type(exc).__name__ == "BudgetExhausted":
                    break
                raise

        if evaluated:
            evaluated.sort(key=lambda item: item[1])
            winner, winner_score = evaluated[0]
            if winner_score <= current_score:
                current, current_score = winner, winner_score
            else:
                # Annealing permits movement across shallow score barriers.
                temp = max(500.0, min(3500.0, 0.010 * current_score))
                if rng.random() < math.exp(-min(40.0, (winner_score - current_score) / temp)):
                    current, current_score = winner, winner_score

        if best_score < previous_best:
            stale = 0
        else:
            stale += 1

        # Periodic kick starts a new trajectory without discarding the elite archive.
        if stale and stale % 4 == 0:
            source = best_net if stale < 12 or rng.random() < 0.5 else archive[
                int(rng.integers(len(archive)))
            ][0]
            _, _, hot = link_data(source)
            current = kick(source, min(8, 2 + stale // 4), hot)
            current_score = scores.get(current, float("inf"))
            if current_score == float("inf") and tools.lp_calls_left > 0:
                try:
                    current_score = evaluate(current)
                except Exception as exc:
                    if type(exc).__name__ == "BudgetExhausted":
                        break
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
