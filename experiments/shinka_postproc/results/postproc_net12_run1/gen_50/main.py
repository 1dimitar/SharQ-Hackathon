"""Classical post-processing of QAOA output for QOBLIB 08. ShinkaEvolve edits the EVOLVE block."""

import json
import time

import numpy as np

# EVOLVE-BLOCK-START
import math


def improve(demand, candidates, tools, rng):
    """Search repaired networks with dual-guided moves and occasional kicks."""
    scores = {}
    details = {}
    archive = []
    best_net = None
    best_score = float("inf")

    def norm(arcs):
        return tuple(sorted(tuple(map(int, a)) for a in arcs))

    def edge_set(net):
        return set(net)

    def distance(a, b):
        return len(edge_set(a).symmetric_difference(edge_set(b)))

    def remember(net, score):
        nonlocal best_net, best_score
        if score < best_score:
            best_net, best_score = net, score

        if any(net == old for old, _ in archive):
            return
        if archive:
            nearest = min(distance(net, old) for old, _ in archive)
            if nearest < 3 and score > min(s for _, s in archive) * 1.002:
                return
        archive.append((net, score))
        archive.sort(key=lambda item: item[1])
        del archive[14:]

    def evaluate(net):
        net = norm(net)
        if net in scores:
            return scores[net]
        if not tools.is_valid(net):
            return float("inf")
        value = float(tools.lp(net))
        scores[net] = value
        remember(net, value)
        return value

    # Repair all samples first; score them in surrogate-energy order.
    repaired = set()
    for cand in candidates:
        try:
            net = norm(tools.repair(cand))
            if tools.is_valid(net):
                repaired.add(net)
        except Exception:
            continue

    if not repaired:
        try:
            net = norm(tools.repair([]))
            if tools.is_valid(net):
                repaired.add(net)
        except Exception:
            pass
    if not repaired:
        return []

    ordered = sorted(repaired, key=tools.energy)
    # Favor variety among the first starts, then use remaining repaired samples.
    starts = []
    for net in ordered:
        if all(distance(net, old) >= 4 for old in starts):
            starts.append(net)
        if len(starts) >= 20:
            break
    for net in ordered:
        if len(starts) >= 32:
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

    def detail(net):
        if net not in details:
            try:
                details[net] = tools.lp_detail(net)
            except Exception:
                details[net] = {}
        return details[net]

    def bottleneck_info(net):
        d = detail(net)
        prices = {}
        for link, value in d.get("prices", {}).items():
            try:
                prices[tuple(map(int, link))] = float(value)
            except Exception:
                pass

        hot = {link for link, price in prices.items() if price > 1e-10}
        if not hot:
            loads = []
            for link, value in d.get("loads", {}).items():
                try:
                    loads.append((tuple(map(int, link)), float(value)))
                except Exception:
                    pass
            if loads:
                peak = max(value for _, value in loads)
                hot = {link for link, value in loads if value >= 0.94 * peak}
        return prices, hot or set(net)

    def propose(net):
        """Generate degree-preserving two-switches and sampled three-cycles."""
        arcs = list(net)
        arc_set = set(net)
        moves = set()

        for p, (a, b) in enumerate(arcs):
            for c, d in arcs[p + 1:]:
                x, y = (a, d), (c, b)
                if a == c or x[0] == x[1] or y[0] == y[1]:
                    continue
                if x == y or x in arc_set or y in arc_set:
                    continue
                moves.add(norm((arc_set - {(a, b), (c, d)}) | {x, y}))

        # Rotations broaden the neighborhood beyond single two-switches.
        trials = min(180, max(60, len(arcs) * 8))
        for _ in range(trials):
            i, j, k = map(int, rng.choice(len(arcs), size=3, replace=False))
            (a, b), (c, d), (e, f) = arcs[i], arcs[j], arcs[k]
            if len({a, c, e}) < 3:
                continue
            if rng.random() < 0.5:
                added = ((a, d), (c, f), (e, b))
            else:
                added = ((a, f), (c, b), (e, d))
            if any(u == v for u, v in added) or len(set(added)) != 3:
                continue
            removed = {(a, b), (c, d), (e, f)}
            if any(link in arc_set - removed for link in added):
                continue
            moves.add(norm((arc_set - removed) | set(added)))
        return moves

    def kick(net, hot, steps):
        """Apply a few valid, preferably bottleneck-involving, two-switches."""
        cur = set(net)
        for _ in range(steps):
            arcs = list(cur)
            pairs = []
            for i in range(len(arcs)):
                for j in range(i + 1, len(arcs)):
                    if hot and arcs[i] not in hot and arcs[j] not in hot:
                        continue
                    a, b = arcs[i]
                    c, d = arcs[j]
                    x, y = (a, d), (c, b)
                    if (a != c and a != d and c != b and x != y
                            and x not in cur and y not in cur):
                        pairs.append((i, j, x, y))
            if not pairs and hot:
                hot = set()
                continue
            if not pairs:
                break
            i, j, x, y = pairs[int(rng.integers(len(pairs)))]
            cur.remove(arcs[i])
            cur.remove(arcs[j])
            cur.add(x)
            cur.add(y)
        result = norm(cur)
        return result if tools.is_valid(result) else net

    current, current_score = best_net, best_score
    iteration = 0
    stagnant = 0

    while tools.lp_calls_left > 0:
        iteration += 1
        if not archive:
            break

        # Mix incumbent exploitation with diverse archive branches.
        if stagnant >= 18 and rng.random() < 0.65:
            pool = archive[2:] or archive
            source, _ = pool[int(rng.integers(len(pool)))]
            _, hot = bottleneck_info(source)
            source = kick(source, hot, min(7, 2 + stagnant // 18))
            stagnant = 0
        elif rng.random() < 0.48:
            source, _ = archive[int(rng.integers(min(6, len(archive))))]
        elif rng.random() < 0.58:
            source = current
        else:
            source, _ = archive[int(rng.integers(len(archive)))]

        prices, hot = bottleneck_info(source)
        old_links = set(source)
        neighborhood = propose(source)
        fresh = [
            net for net in neighborhood
            if net not in scores and not tools.is_cached(net)
        ]

        # If a neighborhood is exhausted, branch from another good basin.
        if not fresh and len(archive) > 1:
            source, _ = archive[int(rng.integers(len(archive)))]
            prices, hot = bottleneck_info(source)
            old_links = set(source)
            fresh = [
                net for net in propose(source)
                if net not in scores and not tools.is_cached(net)
            ]

        if not fresh:
            stagnant += 1
            continue

        def price_benefit(net):
            new_links = set(net)
            removed = old_links - new_links
            added = new_links - old_links
            return (sum(prices.get(e, 0.0) for e in removed)
                    - sum(prices.get(e, 0.0) for e in added))

        # Rank cheaply using the surrogate and dual benefit, then make the
        # expensive LP batch structurally diverse before filling by rank.
        fresh.sort(key=lambda net: (tools.energy(net), -price_benefit(net)))
        shortlist = fresh[:min(64, len(fresh))]
        selected = []
        pool = list(shortlist)

        # Greedily retain ranked moves that differ substantially from those
        # already selected. The rank term breaks ties in favor of better moves.
        while pool and len(selected) < 7:
            if not selected:
                chosen = pool[0]
            else:
                ranked = {net: i for i, net in enumerate(shortlist)}
                diverse = [
                    net for net in pool
                    if min(distance(net, old) for old in selected) >= 6
                ]
                if not diverse:
                    break
                chosen = max(
                    diverse,
                    key=lambda net: (
                        min(distance(net, old) for old in selected)
                        - 0.15 * ranked[net]
                    ),
                )
            selected.append(chosen)
            pool.remove(chosen)

        # Fill the main batch with the best unselected proposals.
        for net in fresh:
            if len(selected) >= 8:
                break
            if net not in selected:
                selected.append(net)

        remaining = [net for net in fresh if net not in set(selected)]
        if remaining:
            count = min(3, len(remaining))
            ids = rng.choice(len(remaining), size=count, replace=False)
            selected.extend(remaining[int(i)] for i in ids)

        old_best = best_score
        evaluated = []
        exhausted = False
        for net in selected:
            if tools.lp_calls_left <= 0:
                exhausted = True
                break
            try:
                evaluated.append((net, evaluate(net)))
            except Exception as exc:
                if type(exc).__name__ == "BudgetExhausted":
                    exhausted = True
                    break
                raise

        if best_score < old_best:
            stagnant = 0
        else:
            stagnant += 1

        if evaluated:
            evaluated.sort(key=lambda item: item[1])
            winner, winner_score = evaluated[0]
            if winner_score <= current_score:
                current, current_score = winner, winner_score
            else:
                scale = max(400.0, min(5000.0, 0.012 * max(1.0, current_score)))
                temperature = scale * (0.82 ** (iteration % 12))
                if rng.random() < math.exp(
                    -min(50.0, (winner_score - current_score) / temperature)
                ):
                    current, current_score = winner, winner_score

            # Periodically reset to an archive elite to avoid a single-basin loop.
            if iteration % 11 == 0 and archive:
                current, current_score = archive[
                    int(rng.integers(min(6, len(archive))))
                ]

        if exhausted:
            break

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
