"""Classical post-processing of QAOA output for QOBLIB 08. ShinkaEvolve edits the EVOLVE block."""

import json
import time

import numpy as np

# EVOLVE-BLOCK-START
import math


def improve(demand, candidates, tools, rng):
    """Search repaired networks using dual-guided, screened neighborhood moves."""
    scores = {}
    archive = []
    best_net = None
    best_score = float("inf")

    def norm(arcs):
        return tuple(sorted(tuple(map(int, a)) for a in arcs))

    def distance(a, b):
        return len(set(a).symmetric_difference(b))

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
        del archive[12:]

    def evaluate(net):
        if net in scores:
            return scores[net]
        if not tools.is_valid(net):
            return float("inf")
        value = float(tools.lp(net))
        scores[net] = value
        remember(net, value)
        return value

    # Repair each sample, deduplicate, then score the repaired starts.
    repaired = set()
    for cand in candidates:
        try:
            net = norm(tools.repair(cand))
            if tools.is_valid(net):
                repaired.add(net)
        except Exception:
            continue

    # Reserve a small, explicit initialization quota for random valid networks.
    random_starts = set()
    n = tools.n
    for _ in range(8):
        first = None
        for _attempt in range(100):
            perm = rng.permutation(n)
            if all(int(perm[i]) != i for i in range(n)):
                first = perm
                break
        if first is None:
            continue

        second = None
        for _attempt in range(100):
            perm = rng.permutation(n)
            if all(int(perm[i]) != i and int(perm[i]) != int(first[i])
                   for i in range(n)):
                second = perm
                break
        if second is None:
            continue

        net = norm(
            [(i, int(first[i])) for i in range(n)]
            + [(i, int(second[i])) for i in range(n)]
        )
        if tools.is_valid(net):
            random_starts.add(net)

    # Start with diverse, energy-ranked QAOA repairs; backfill from the ranking
    # when the pool does not contain enough distinct topologies.
    ordered = sorted(repaired, key=lambda x: tools.energy(x))
    starts = []
    for net in ordered:
        if all(len(set(net).symmetric_difference(set(old))) >= 4
               for old in starts):
            starts.append(net)
        if len(starts) >= 24:
            break
    for net in ordered:
        if len(starts) >= 32:
            break
        if net not in starts:
            starts.append(net)

    # Score the random quota first so it cannot be displaced by QAOA repairs.
    for net in random_starts:
        if tools.lp_calls_left <= 0:
            break
        try:
            evaluate(net)
        except Exception as exc:
            if type(exc).__name__ == "BudgetExhausted":
                break
            raise

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
        return []
    for net, value in sorted(scores.items(), key=lambda item: item[1]):
        remember(net, value)

    current, current_score = best_net, best_score

    def bottleneck_links(net):
        try:
            detail = tools.lp_detail(net)
        except Exception:
            return set(net)

        active = set()
        for link, price in detail.get("prices", {}).items():
            try:
                if float(price) > 1e-10:
                    active.add(tuple(map(int, link)))
            except Exception:
                pass
        if active:
            return active

        loads = detail.get("loads", {})
        try:
            max_load = max(float(x) for x in loads.values())
            active = {
                tuple(map(int, link))
                for link, load in loads.items()
                if float(load) >= 0.94 * max_load
            }
        except Exception:
            active = set()
        return active if active else set(net)

    def propose(net):
        """Generate valid two-switches and sampled three-link rotations."""
        arcs = list(net)
        arc_set = set(net)
        result = set()

        # Two-switches preserve both incoming and outgoing degrees.
        for p in range(len(arcs)):
            a, b = arcs[p]
            for q in range(p + 1, len(arcs)):
                c, d = arcs[q]
                if a == c:
                    continue
                x, y = (a, d), (c, b)
                if x[0] == x[1] or y[0] == y[1] or x == y:
                    continue
                if x in arc_set or y in arc_set:
                    continue
                result.add(tuple(sorted((arc_set - {(a, b), (c, d)}) | {x, y})))

        # Three-cycles add moves not reachable by one two-switch.
        trials = min(180, max(60, len(arcs) * 8))
        for _ in range(trials):
            i, j, k = map(int, rng.choice(len(arcs), size=3, replace=False))
            (a, b), (c, d), (e, f) = arcs[i], arcs[j], arcs[k]
            if len({a, c, e}) < 3:
                continue
            if rng.random() < 0.5:
                repl = ((a, d), (c, f), (e, b))
            else:
                repl = ((a, f), (c, b), (e, d))
            if any(u == v for u, v in repl) or len(set(repl)) != 3:
                continue
            if any(link in arc_set for link in repl):
                continue
            removed = {(a, b), (c, d), (e, f)}
            result.add(tuple(sorted((arc_set - removed) | set(repl)))
                       )
        return result

    iteration = 0
    stagnation = 0
    parent_uses = {}
    while tools.lp_calls_left > 0:
        iteration += 1

        # After a stall, kick from a least-used non-elite archive parent.
        # Otherwise retain the usual mixture of annealing and archive branches.
        if stagnation >= 8 and len(archive) > 2 and rng.random() < 0.70:
            tail = archive[2:]
            least_used = min(parent_uses.get(net, 0) for net, _ in tail)
            choices = [item for item in tail
                       if parent_uses.get(item[0], 0) == least_used]
            source, source_score = choices[int(rng.integers(len(choices)))]
        elif len(archive) > 1 and rng.random() < 0.30:
            source, source_score = archive[int(rng.integers(len(archive)))]
        elif rng.random() < 0.65:
            source, source_score = current, current_score
        else:
            source, source_score = archive[int(rng.integers(min(5, len(archive))))]

        parent_uses[source] = parent_uses.get(source, 0) + 1
        focus = bottleneck_links(source)
        neighborhood = propose(source)
        fresh = [
            net for net in neighborhood
            if net not in scores and not tools.is_cached(net)
        ]

        if not fresh and len(archive) > 1:
            source, source_score = archive[int(rng.integers(len(archive)))]
            focus = bottleneck_links(source)
            neighborhood = propose(source)
            fresh = [
                net for net in neighborhood
                if net not in scores and not tools.is_cached(net)
            ]
        if not fresh:
            continue

        # Keep the energy-ranked batch, but reserve slots for moves whose
        # removed links have greater dual price than their replacement links.
        old_links = set(source)
        fresh.sort(key=lambda net: (
            tools.energy(net),
            not bool((old_links - set(net)) & focus),
        ))
        selected = fresh[:9]

        prices = {}
        try:
            detail = tools.lp_detail(source)
            prices = {
                tuple(map(int, link)): float(price)
                for link, price in detail.get("prices", {}).items()
            }
        except Exception:
            pass

        def price_benefit(net):
            removed = old_links - set(net)
            added = set(net) - old_links
            return sum(prices.get(link, 0.0) for link in removed) - sum(
                prices.get(link, 0.0) for link in added
            )

        # Reserve three batch places for the strongest positive dual signals,
        # even when those moves rank poorly under the QUBO surrogate.
        selected_set = set(selected)
        dual_ranked = sorted(
            (net for net in fresh if net not in selected_set),
            key=lambda net: price_benefit(net),
            reverse=True,
        )
        selected.extend(
            net for net in dual_ranked[:3] if price_benefit(net) > 1e-10
        )

        # Preserve a little exploration even when the surrogate ranking is poor.
        remaining = [net for net in fresh if net not in set(selected)]
        if remaining:
            take = min(3, len(remaining))
            ids = rng.choice(len(remaining), size=take, replace=False)
            selected.extend(remaining[int(i)] for i in ids)

        iteration_best = best_score
        evaluated = []
        exhausted = False
        for net in selected:
            if tools.lp_calls_left <= 0:
                exhausted = True
                break
            try:
                value = evaluate(net)
                evaluated.append((net, value))
            except Exception as exc:
                if type(exc).__name__ == "BudgetExhausted":
                    exhausted = True
                    break
                raise

        if best_score < iteration_best:
            stagnation = 0
        else:
            stagnation += 1

        if evaluated:
            evaluated.sort(key=lambda item: item[1])
            winner, winner_score = evaluated[0]
            if winner_score <= current_score:
                current, current_score = winner, winner_score
            else:
                scale = max(400.0, min(5000.0, 0.012 * max(1.0, current_score)))
                temperature = scale * (0.82 ** (iteration % 12))
                probability = math.exp(
                    -min(50.0, (winner_score - current_score) / temperature)
                )
                if rng.random() < probability:
                    current, current_score = winner, winner_score

            if iteration % 9 == 0 and archive:
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
