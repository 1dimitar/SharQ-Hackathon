"""Starting program of post-processing run 2 (network13): run 1's best program (gen 49, combined
score 0.9901 on network12), copied from results/postproc_net12_run1/best/main.py. Only the fixed
wrapper below the EVOLVE block changed (tools.energy uses each pool's own Hamiltonian).
"""

"""Classical post-processing of QAOA output for QOBLIB 08. ShinkaEvolve edits the EVOLVE block."""

import json
import time

import numpy as np

# EVOLVE-BLOCK-START
import math


def improve(demand, candidates, tools, rng):
    """Repair QAOA samples, then search a diverse archive with mixed move batches."""
    n = tools.n
    link_count = 2 * n
    seen = {}
    best_net = None
    best_score = float("inf")
    stalls = {}

    def canon(arcs):
        return tuple(sorted(tuple(map(int, a)) for a in arcs))

    def distance(a, b):
        return len(set(a).symmetric_difference(b))

    energy_cache = {}

    def energy(net):
        if net not in energy_cache:
            try:
                energy_cache[net] = float(tools.energy(net))
            except Exception:
                energy_cache[net] = 0.0
        return energy_cache[net]

    def evaluate(net):
        nonlocal best_net, best_score
        net = canon(net)
        if net not in seen:
            seen[net] = tools.lp(net)
        score = seen[net]
        if score < best_score:
            best_score, best_net = score, net
        return score

    def repaired_samples():
        result = set()
        for cand in candidates:
            try:
                net = canon(tools.repair(cand))
                if tools.is_valid(net):
                    result.add(net)
            except Exception:
                continue
        if not result:
            try:
                net = canon(tools.repair([]))
                if tools.is_valid(net):
                    result.add(net)
            except Exception:
                pass
        return result

    def diverse_order(items, limit, min_dist=4):
        ordered = sorted(items, key=energy)
        chosen = []
        for net in ordered:
            if all(distance(net, old) >= min_dist for old in chosen):
                chosen.append(net)
                if len(chosen) >= limit:
                    return chosen
        for net in ordered:
            if net not in chosen:
                chosen.append(net)
                if len(chosen) >= limit:
                    break
        return chosen

    def archive():
        ranked = sorted(
            (net for net in seen if math.isfinite(seen[net])),
            key=seen.get,
        )
        if not ranked:
            return [best_net] if best_net is not None else []

        # Allocate places to certified elites, good energy-ranked alternatives,
        # and structurally novel networks. All limits scale with instance size.
        limit = min(48, max(24, 2 * n))
        elite_quota = max(1, limit // 2)
        energy_quota = max(1, limit // 4)
        out = ranked[:elite_quota]
        out_set = set(out)

        energy_pool = [
            net for net in ranked
            if net not in out_set and seen[net] <= best_score * 1.08
        ]
        energy_pool.sort(key=energy)
        energy_added = 0
        for net in energy_pool:
            if all(distance(net, old) >= 2 for old in out):
                out.append(net)
                out_set.add(net)
                energy_added += 1
                if energy_added >= energy_quota:
                    break

        # Greedily retain candidates far from the archive already selected.
        # The quality band prevents novelty alone from admitting very weak nets.
        novel_pool = [
            net for net in ranked
            if net not in out_set and seen[net] <= best_score * 1.08
        ]
        while novel_pool and len(out) < limit:
            net = max(
                novel_pool,
                key=lambda x: (
                    min(distance(x, old) for old in out),
                    -seen[x],
                ),
            )
            out.append(net)
            out_set.add(net)
            novel_pool.remove(net)

        for net in ranked:
            if len(out) >= limit:
                break
            if net not in out_set:
                out.append(net)
                out_set.add(net)
        return out

    def select_parent(pop):
        if not pop:
            return best_net
        # Mostly exploit, but regularly visit the full diverse archive.
        if rng.random() < 0.52:
            return pop[int(rng.integers(min(6, len(pop))))]
        return pop[int(rng.integers(len(pop)))]

    def two_switch(net, p, q):
        arcs = list(net)
        (a, b), (c, d) = arcs[p], arcs[q]
        if a == d or c == b:
            return None
        current = set(net)
        if (a, d) in current or (c, b) in current:
            return None
        current.remove((a, b))
        current.remove((c, d))
        current.add((a, d))
        current.add((c, b))
        return tuple(sorted(current))

    def add_rotation_moves(net, hot, moves, trials):
        arcs = list(net)
        current = set(net)
        if len(arcs) < 3:
            return
        for _ in range(trials):
            ids = rng.choice(len(arcs), size=3, replace=False)
            (a, b), (c, d), (e, f) = (arcs[int(i)] for i in ids)
            old = {(a, b), (c, d), (e, f)}
            if hot and not (old & hot) and rng.random() < 0.75:
                continue
            for new in (
                {(a, d), (c, f), (e, b)},
                {(a, f), (c, b), (e, d)},
            ):
                if len(new) != 3 or any(x == y for x, y in new):
                    continue
                if new & (current - old):
                    continue
                moves.add(tuple(sorted((current - old) | new)))

    def add_kick_moves(net, moves, count, strength):
        arcs = list(net)
        if len(arcs) < 2:
            return
        for _ in range(count):
            cur = set(net)
            for _ in range(strength):
                now = list(cur)
                p, q = rng.choice(len(now), size=2, replace=False)
                (a, b), (c, d) = now[int(p)], now[int(q)]
                if a == d or c == b or (a, d) in cur or (c, b) in cur:
                    continue
                cur.remove((a, b))
                cur.remove((c, d))
                cur.add((a, d))
                cur.add((c, b))
            if cur != set(net):
                moves.add(tuple(sorted(cur)))

    def generate_moves(parent, detail, stall):
        arcs = list(parent)
        current = set(parent)
        moves = set()

        prices = detail.get("prices", {}) if detail else {}
        hot = {
            tuple(map(int, arc))
            for arc, price in prices.items()
            if float(price) > 0
        }
        if not hot and detail:
            loads = detail.get("loads", {})
            top = sorted(
                ((float(load), tuple(map(int, arc))) for arc, load in loads.items()),
                reverse=True,
            )
            hot = {arc for _, arc in top[:max(4, n // 2)]}

        # Build the full focused 2-switch neighborhood. Random non-focused
        # switches are added separately so degeneracy in the dual is harmless.
        for p in range(len(arcs)):
            for q in range(p + 1, len(arcs)):
                if hot and arcs[p] not in hot and arcs[q] not in hot:
                    continue
                move = two_switch(parent, p, q)
                if move is not None:
                    moves.add(move)

        random_pairs = max(60, 5 * n)
        for _ in range(random_pairs):
            p = int(rng.integers(len(arcs)))
            q = int(rng.integers(len(arcs) - 1))
            if q >= p:
                q += 1
            move = two_switch(parent, p, q)
            if move is not None:
                moves.add(move)

        # Increase the nonlocal neighborhood only when this parent is stale.
        rotation_trials = min(260, 35 + 18 * (stall // 2))
        add_rotation_moves(parent, hot, moves, rotation_trials)
        kick_count = min(18, 6 + stall // 2)
        kick_strength = max(2, min(7, 2 + stall // 3))
        add_kick_moves(parent, moves, kick_count, kick_strength)

        return [m for m in moves if m != parent and tools.is_valid(m)], hot

    def select_batch(moves, parent, hot, batch_size):
        fresh = [
            net for net in moves
            if net not in seen and not tools.is_cached(net)
        ]
        if not fresh:
            return []

        base_energy = energy(parent)
        ranked = sorted(fresh, key=lambda net: energy(net) - base_energy)
        hot_ranked = sorted(
            fresh,
            key=lambda net: (
                -len((set(parent) ^ set(net)) & hot) if hot else 0,
                energy(net) - base_energy,
            ),
        )

        selected = []

        def take_from(order, target, min_distance=2):
            for net in order:
                if net in selected:
                    continue
                if all(distance(net, old) >= min_distance for old in selected):
                    selected.append(net)
                if len(selected) >= target:
                    break

        # Separate batch capacity between surrogate exploitation, dual guidance,
        # and uncommitted exploration to avoid a single ranking dominating.
        take_from(ranked, max(1, batch_size // 3))
        take_from(hot_ranked, max(1, 2 * batch_size // 3))
        if len(selected) < batch_size:
            tail = [net for net in fresh if net not in selected]
            rng.shuffle(tail)
            take_from(tail, batch_size, min_distance=1)
        if len(selected) < batch_size:
            for net in ranked:
                if net not in selected:
                    selected.append(net)
                if len(selected) >= batch_size:
                    break
        return selected[:batch_size]

    repaired = repaired_samples()
    if not repaired:
        return []

    starts = diverse_order(repaired, min(30, len(repaired)))
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

    # Smaller batches at larger n reduce candidate-generation overhead and
    # make each screening cycle more responsive to newly found incumbents.
    batch_size = max(5, min(10, 12 - n // 4))
    rounds_without_progress = 0
    rounds = 0

    while tools.lp_calls_left > 0:
        rounds += 1
        pop = archive()
        parent = select_parent(pop)
        if parent is None:
            break

        try:
            detail = tools.lp_detail(parent)
        except Exception:
            detail = None

        stall = stalls.get(parent, 0)
        moves, hot = generate_moves(parent, detail, stall)
        batch = select_batch(moves, parent, hot, batch_size)

        # If the chosen neighborhood is exhausted, switch regions immediately.
        if not batch:
            alternatives = [net for net in pop if net != parent]
            if alternatives:
                parent = alternatives[int(rng.integers(len(alternatives)))]
                try:
                    detail = tools.lp_detail(parent)
                except Exception:
                    detail = None
                moves, hot = generate_moves(
                    parent, detail, stalls.get(parent, 0)
                )
                batch = select_batch(moves, parent, hot, batch_size)

        if not batch:
            # A few direct kicks provide a fallback when the local neighborhood
            # is entirely cached.
            fallback = set()
            add_kick_moves(parent, fallback, 20, max(3, min(8, n // 2)))
            batch = [
                net for net in fallback
                if net != parent and net not in seen
                and not tools.is_cached(net) and tools.is_valid(net)
            ][:1]

        if not batch:
            if rounds > 10000:
                break
            continue

        parent_score = seen.get(parent, float("inf"))
        old_best = best_score
        for net in batch:
            if tools.lp_calls_left <= 0:
                break
            try:
                evaluate(net)
            except Exception as exc:
                if type(exc).__name__ == "BudgetExhausted":
                    return list(best_net) if best_net is not None else []
                continue

        batch_best = min((seen.get(net, float("inf")) for net in batch), default=float("inf"))
        if batch_best < parent_score:
            stalls[parent] = 0
        else:
            stalls[parent] = stalls.get(parent, 0) + 1

        if best_score < old_best:
            rounds_without_progress = 0
        else:
            rounds_without_progress += 1

        # Periodically favor less-visited archive regions after a long plateau.
        if rounds_without_progress > max(8, n):
            for net in pop:
                stalls[net] = stalls.get(net, 0) + 1
            rounds_without_progress = 0

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
