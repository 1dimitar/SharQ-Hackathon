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
    """Search repaired QAOA networks using a surrogate-screened diverse archive."""
    n = tools.n
    seen = {}
    best_net = None
    best_score = float("inf")
    parent_stalls = {}

    def canon(arcs):
        return tuple(sorted(tuple(map(int, a)) for a in arcs))

    def evaluate(net):
        """Score once locally; budget exhaustion is allowed to reach the caller."""
        nonlocal best_net, best_score
        net = canon(net)
        if net not in seen:
            seen[net] = tools.lp(net)
        s = seen[net]
        if s < best_score:
            best_net, best_score = net, s
        return s

    def energy(net):
        try:
            return float(tools.energy(net))
        except Exception:
            return 0.0

    def distance(a, b):
        return len(set(a) ^ set(b))

    def choose_starts(repaired, limit):
        """Prefer low surrogate energy, while retaining topological variety."""
        ordered = sorted(repaired, key=energy)
        chosen = []
        for net in ordered:
            if all(distance(net, old) >= 4 for old in chosen):
                chosen.append(net)
            if len(chosen) >= limit:
                break
        for net in ordered:
            if len(chosen) >= limit:
                break
            if net not in chosen:
                chosen.append(net)
        return chosen

    # Repair all raw samples for free, then seed the search with a compact,
    # energy-ranked but diversity-aware subset.
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
            return []

    initial = choose_starts(repaired, min(30, len(repaired)))
    for net in initial:
        if tools.lp_calls_left <= 0:
            break
        try:
            evaluate(net)
        except Exception as exc:
            if type(exc).__name__ == "BudgetExhausted":
                break
            continue

    if not seen:
        return list(initial[0])

    def archive():
        """Keep excellent networks and the best representatives of distinct regions."""
        ranked = sorted(seen, key=seen.get)
        out = []
        for net in ranked:
            if all(distance(net, old) >= 4 for old in out):
                out.append(net)
            if len(out) >= 24:
                break
        for net in ranked:
            if len(out) >= 32:
                break
            if net not in out:
                out.append(net)
        return out

    def select_parent(pop):
        """Mix elite exploitation with sampling from diverse archived networks."""
        if not pop:
            return best_net
        if rng.random() < 0.48:
            return pop[int(rng.integers(min(6, len(pop))))]
        return pop[int(rng.integers(len(pop)))]

    def switch(net, p, q):
        """Swap the destinations of two links, preserving all degrees."""
        arcs = list(net)
        (a, b), (c, d) = arcs[p], arcs[q]
        if a == d or c == b:
            return None
        cur = set(net)
        if (a, d) in cur or (c, b) in cur:
            return None
        cur.remove((a, b))
        cur.remove((c, d))
        cur.add((a, d))
        cur.add((c, b))
        return tuple(sorted(cur))

    def make_moves(net, detail):
        """Build a focused two-switch neighborhood plus sampled rotations and kicks."""
        arcs = list(net)
        current = set(net)
        moves = set()

        prices = detail.get("prices", {}) if detail else {}
        hot = {tuple(map(int, a)) for a, p in prices.items() if p > 0}
        if not hot and detail:
            loads = detail.get("loads", {})
            ranked_loads = sorted(
                ((float(v), tuple(map(int, a))) for a, v in loads.items()),
                reverse=True,
            )
            hot = {a for _, a in ranked_loads[:6]}

        # Exhaustively explore swaps involving at least one critical link.
        for p in range(len(arcs)):
            for q in range(p + 1, len(arcs)):
                if hot and arcs[p] not in hot and arcs[q] not in hot:
                    continue
                cand = switch(net, p, q)
                if cand is not None:
                    moves.add(cand)

        # Add non-focused swaps so that dual degeneracy cannot trap the search.
        for _ in range(70):
            if len(arcs) < 2:
                break
            p = int(rng.integers(len(arcs)))
            q = int(rng.integers(len(arcs) - 1))
            if q >= p:
                q += 1
            cand = switch(net, p, q)
            if cand is not None:
                moves.add(cand)

        # Generate rotations that deliberately remove high-pressure links.
        # Retain random rotations as a complementary source of nonlocal moves.
        if len(arcs) >= 3:
            stalls = parent_stalls.get(net, 0)

            def add_rotation(ids):
                (a, b), (c, d), (e, f) = (arcs[int(i)] for i in ids)
                old = {(a, b), (c, d), (e, f)}
                for new in (
                    {(a, d), (c, f), (e, b)},
                    {(a, f), (c, b), (e, d)},
                ):
                    if len(new) != 3 or any(x == y for x, y in new):
                        continue
                    if new & (current - old):
                        continue
                    moves.add(tuple(sorted((current - old) | new)))

            if hot:
                loads = detail.get("loads", {}) if detail else {}
                prices = detail.get("prices", {}) if detail else {}
                hot_order = sorted(
                    hot,
                    key=lambda a: (
                        float(prices.get(a, 0.0)),
                        float(loads.get(a, 0.0)),
                    ),
                    reverse=True,
                )
                arc_index = {a: i for i, a in enumerate(arcs)}
                target_trials = min(180, max(24, 6 * len(hot_order)))
                for t in range(target_trials):
                    critical = hot_order[t % len(hot_order)]
                    p = arc_index[critical]
                    others = [i for i in range(len(arcs)) if i != p]
                    chosen = rng.choice(len(others), size=2, replace=False)
                    add_rotation([p, others[int(chosen[0])], others[int(chosen[1])]])

            rotation_trials = min(350, 70 * (1 + stalls // 3))
            for _ in range(rotation_trials):
                ids = rng.choice(len(arcs), size=3, replace=False)
                if hot and not any(arcs[int(i)] in hot for i in ids):
                    continue
                add_rotation(ids)

        # Stalled parents get more and longer multi-switch kicks to cross barriers.
        stalls = parent_stalls.get(net, 0)
        kick_trials = min(3 * n, 16 + 2 * stalls)
        max_kick_length = max(2, 2 + n // 2)
        for _ in range(kick_trials):
            cur = set(net)
            kick_length = min(max_kick_length, 2 + int(rng.integers(4)) + stalls // 4)
            for _ in range(kick_length):
                arcs_now = list(cur)
                if len(arcs_now) < 2:
                    break
                p, q = rng.choice(len(arcs_now), size=2, replace=False)
                a, b = arcs_now[int(p)]
                c, d = arcs_now[int(q)]
                if a == d or c == b or (a, d) in cur or (c, b) in cur:
                    continue
                cur.remove((a, b))
                cur.remove((c, d))
                cur.add((a, d))
                cur.add((c, b))
            cand = tuple(sorted(cur))
            if cand != net:
                moves.add(cand)

        return [m for m in moves if m != net and tools.is_valid(m)]

    def screen_batch(moves, parent, detail=None, batch_size=9):
        """Adapt energy, pressure, and archive-novelty slots to search progress."""
        fresh = [
            m for m in moves
            if m not in seen and not tools.is_cached(m)
        ]
        if not fresh:
            return []

        energy_order = sorted(fresh, key=energy)
        stalls = parent_stalls.get(parent, 0)
        explore_quota = min(
            max(1, batch_size // 2),
            1 + stalls // 2,
            max(0, batch_size - 2),
        )
        pressure_quota = min(
            2, max(0, batch_size - explore_quota - 2)
        ) if detail else 0
        energy_quota = max(0, batch_size - explore_quota - pressure_quota)

        selected = []
        for net in energy_order:
            if all(distance(net, old) >= 2 for old in selected):
                selected.append(net)
            if len(selected) >= energy_quota:
                break

        # Use dual prices and loads as a complementary, not dominant, move signal.
        if detail and pressure_quota:
            loads = {
                tuple(map(int, a)): float(v)
                for a, v in detail.get("loads", {}).items()
            }
            prices = {
                tuple(map(int, a)): float(v)
                for a, v in detail.get("prices", {}).items()
            }
            max_load = max(loads.values(), default=1.0) or 1.0
            max_price = max(prices.values(), default=1.0) or 1.0
            max_demand = float(np.max(demand)) if np.size(demand) else 1.0
            if max_demand <= 0:
                max_demand = 1.0

            parent_set = set(parent)
            pressure_ranked = []
            for net in fresh:
                removed = parent_set - set(net)
                added = set(net) - parent_set
                proxy = sum(
                    0.65 * loads.get(a, 0.0) / max_load
                    + 0.35 * prices.get(a, 0.0) / max_price
                    for a in removed
                )
                proxy -= sum(
                    0.45 * float(demand[a[0], a[1]]) / max_demand
                    for a in added
                )
                pressure_ranked.append((proxy, net))
            pressure_ranked.sort(key=lambda item: item[0], reverse=True)

            added_pressure = 0
            for _, net in pressure_ranked:
                if net not in selected and all(
                    distance(net, old) >= 2 for old in selected
                ):
                    selected.append(net)
                    added_pressure += 1
                if added_pressure >= pressure_quota:
                    break

        # Explicitly spend exploration slots on networks far from the scored archive.
        if explore_quota:
            references = archive()
            if not references:
                references = [parent]
            novelty_pool = [net for net in fresh if net not in selected]
            rng.shuffle(novelty_pool)
            novelty_pool.sort(
                key=lambda net: min(distance(net, ref) for ref in references),
                reverse=True,
            )
            target = min(batch_size, len(selected) + explore_quota)
            for net in novelty_pool:
                if all(distance(net, old) >= 2 for old in selected):
                    selected.append(net)
                if len(selected) >= target:
                    break

        # Fill any unused quota with the best remaining energy-ranked moves.
        for net in energy_order:
            if len(selected) >= batch_size:
                break
            if net not in selected:
                selected.append(net)
        return selected[:batch_size]

    # A bounded archive drives a sequence of short, diverse neighborhood searches.
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

        moves = make_moves(parent, detail)
        batch = screen_batch(moves, parent, detail=detail)

        # If a neighborhood has already been exhausted, perturb another archive
        # member rather than repeatedly revisiting the same parent.
        if not batch:
            alternatives = [x for x in pop if x != parent]
            if alternatives:
                parent = alternatives[int(rng.integers(len(alternatives)))]
                try:
                    detail = tools.lp_detail(parent)
                except Exception:
                    detail = None
                batch = screen_batch(make_moves(parent, detail), parent, detail=detail)

        if not batch:
            # A final random degree-preserving kick guarantees progress when possible.
            for _ in range(40):
                cur = set(parent)
                arcs = list(cur)
                for _ in range(3 + int(rng.integers(4))):
                    p, q = rng.choice(len(arcs), size=2, replace=False)
                    a, b = arcs[int(p)]
                    c, d = arcs[int(q)]
                    if a == d or c == b or (a, d) in cur or (c, b) in cur:
                        continue
                    cur.remove((a, b))
                    cur.remove((c, d))
                    cur.add((a, d))
                    cur.add((c, b))
                    arcs = list(cur)
                kicked = tuple(sorted(cur))
                if kicked != parent and kicked not in seen and not tools.is_cached(kicked):
                    batch = [kicked]
                    break

        if not batch:
            # All available moves may be cached; avoid an unbounded spin.
            if rounds > 10000:
                break
            continue

        parent_score = seen.get(parent, float("inf"))
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
            parent_stalls[parent] = 0
        else:
            parent_stalls[parent] = parent_stalls.get(parent, 0) + 1

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
