"""Starting program of post-processing run 2 (network13): run 1's best program (gen 49, combined
score 0.9901 on network12), copied from results/postproc_net12_run1/best/main.py. Only the fixed
wrapper below the EVOLVE block changed (tools.energy uses each pool's own Hamiltonian).
"""

"""Classical post-processing of QAOA output for QOBLIB 08. ShinkaEvolve edits the EVOLVE block."""

import json
import time

import numpy as np

# EVOLVE-BLOCK-START
def improve(demand, candidates, tools, rng):
    """Search repaired networks with a diverse archive and mixed move guidance."""
    n = tools.n
    seen = {}
    set_cache = {}
    best_net = None
    best_score = float("inf")
    parent_stalls = {}
    parent_visits = {}

    def canon(arcs):
        return tuple(sorted(tuple(map(int, a)) for a in arcs))

    def edge_set(net):
        result = set_cache.get(net)
        if result is None:
            result = set(net)
            set_cache[net] = result
        return result

    def distance(a, b):
        return len(edge_set(a) ^ edge_set(b))

    def evaluate(net):
        nonlocal best_net, best_score
        net = canon(net)
        if net not in seen:
            seen[net] = tools.lp(net)
        score = seen[net]
        if score < best_score:
            best_net, best_score = net, score
        return score

    def energy(net):
        try:
            return float(tools.energy(net))
        except Exception:
            return 0.0

    def choose_starts(repaired, limit):
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

    if not seen:
        return list(initial[0])

    def archive():
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
        if not pop:
            return best_net
        if rng.random() < 0.48:
            parent = pop[int(rng.integers(min(6, len(pop))))]
        else:
            weights = np.asarray(
                [
                    1.0 / (np.sqrt(1.0 + i) *
                           np.sqrt(1.0 + parent_visits.get(net, 0)))
                    for i, net in enumerate(pop)
                ],
                dtype=float,
            )
            weights /= weights.sum()
            parent = pop[int(rng.choice(len(pop), p=weights))]
        parent_visits[parent] = parent_visits.get(parent, 0) + 1
        return parent

    def switch(arcs, current, p, q):
        (a, b), (c, d) = arcs[p], arcs[q]
        if a == d or c == b or (a, d) in current or (c, b) in current:
            return None
        cur = current.copy()
        cur.remove((a, b))
        cur.remove((c, d))
        cur.add((a, d))
        cur.add((c, b))
        return tuple(sorted(cur))

    def make_moves(net, detail):
        arcs = list(net)
        current = set(net)
        moves = set()

        prices = detail.get("prices", {}) if detail else {}
        loads = detail.get("loads", {}) if detail else {}
        hot = {tuple(map(int, a)) for a, price in prices.items() if price > 0}
        if not hot and detail:
            load_rank = sorted(
                ((float(v), tuple(map(int, a))) for a, v in loads.items()),
                reverse=True,
            )
            hot = {a for _, a in load_rank[:6]}

        # Focus ordinary 2-switches on critical links, while retaining a random
        # sample of the full neighborhood to avoid over-reliance on dual prices.
        for p in range(len(arcs)):
            for q in range(p + 1, len(arcs)):
                if hot and arcs[p] not in hot and arcs[q] not in hot:
                    continue
                cand = switch(arcs, current, p, q)
                if cand is not None:
                    moves.add(cand)

        random_trials = max(70, 3 * len(arcs))
        for _ in range(random_trials):
            if len(arcs) < 2:
                break
            p = int(rng.integers(len(arcs)))
            q = int(rng.integers(len(arcs) - 1))
            if q >= p:
                q += 1
            cand = switch(arcs, current, p, q)
            if cand is not None:
                moves.add(cand)

        stalls = parent_stalls.get(net, 0)

        # Degree-preserving three-link rotations provide nonlocal alternatives.
        if len(arcs) >= 3:
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
                    p = arc_index.get(critical)
                    if p is None:
                        continue
                    others = [i for i in range(len(arcs)) if i != p]
                    chosen = rng.choice(len(others), size=2, replace=False)
                    add_rotation(
                        [p, others[int(chosen[0])], others[int(chosen[1])]]
                    )

            rotation_trials = min(
                max(350, 12 * len(arcs)),
                max(70, 4 * len(arcs)) * (1 + stalls // 3),
            )
            for _ in range(rotation_trials):
                ids = rng.choice(len(arcs), size=3, replace=False)
                if hot and not any(arcs[int(i)] in hot for i in ids):
                    continue
                add_rotation(ids)

        # Adaptive multi-switch kicks help escape stalled local regions.
        kick_count = min(64, 24 + 8 * stalls)
        max_strength = max(2, min(n // 2, 2 + stalls // 2))
        for _ in range(kick_count):
            cur = set(net)
            strength = min(
                max_strength, 2 + int(rng.integers(4)) + stalls // 3
            )
            for _ in range(strength):
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

        # Relink toward the incumbent using switches that increase overlap.
        target = best_net
        if target is not None and target != net:
            target_set = set(target)
            cur = set(net)
            for _ in range(2 * n):
                out_by_source = {}
                in_by_dest = {}
                for a, b in cur:
                    out_by_source.setdefault(a, []).append(b)
                    in_by_dest.setdefault(b, []).append(a)

                best_step = None
                best_gain = 0
                for a, b in target_set - cur:
                    for d in out_by_source.get(a, ()):
                        for c in in_by_dest.get(b, ()):
                            replacement = (c, d)
                            if c == d or replacement in cur:
                                continue
                            removed = ((a, d), (c, b))
                            gain = (
                                1 + int(replacement in target_set)
                                - int(removed[0] in target_set)
                                - int(removed[1] in target_set)
                            )
                            if gain > best_gain:
                                best_gain = gain
                                best_step = (removed, ((a, b), replacement))
                if best_step is None:
                    break
                removed, added = best_step
                cur.difference_update(removed)
                cur.update(added)
                moves.add(tuple(sorted(cur)))

        return [
            m for m in moves
            if m != net and tools.is_valid(m)
        ]

    def screen_batch(moves, parent, detail=None, batch_size=9):
        fresh = [
            m for m in moves
            if m not in seen and not tools.is_cached(m)
        ]
        if not fresh:
            return []

        base_e = energy(parent)
        ranked = [(energy(net) - base_e, net) for net in fresh]
        ranked.sort(key=lambda item: item[0])

        selected = []
        energy_quota = min(4 if detail else 6, batch_size)
        for _, net in ranked:
            if all(distance(net, old) >= 2 for old in selected):
                selected.append(net)
            if len(selected) >= energy_quota:
                break

        # Independent pressure ranking combines LP loads/dual prices and direct
        # demand on newly introduced links.
        if detail and len(selected) < batch_size:
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
            max_demand = max(max_demand, 1.0)
            parent_set = set(parent)

            pressure_ranked = []
            for net in fresh:
                net_set = set(net)
                removed = parent_set - net_set
                added = net_set - parent_set
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

            pressure_quota = min(3, batch_size - len(selected))
            added_pressure = 0
            for _, net in pressure_ranked:
                if all(distance(net, old) >= 2 for old in selected):
                    selected.append(net)
                    added_pressure += 1
                if added_pressure >= pressure_quota:
                    break

        # Mostly exploit the surrogate ranking, but reserve one slot for a
        # randomly chosen fresh move to preserve exploration.
        for _, net in ranked:
            if len(selected) >= batch_size - 1:
                break
            if net not in selected:
                selected.append(net)

        tail = [net for _, net in ranked if net not in selected]
        if tail and len(selected) < batch_size:
            selected.append(tail[int(rng.integers(len(tail)))])
        return selected[:batch_size]

    rounds = 0
    cached_archive = None
    while tools.lp_calls_left > 0:
        rounds += 1
        # Refresh the expensive score-ranked archive every other iteration.
        if cached_archive is None or rounds % 2 == 1:
            cached_archive = archive()
        pop = cached_archive
        parent = select_parent(pop)
        if parent is None:
            break

        try:
            detail = tools.lp_detail(parent)
        except Exception:
            detail = None

        batch_size = max(9, min(12, 5 + n // 3))
        batch = screen_batch(
            make_moves(parent, detail), parent, detail=detail,
            batch_size=batch_size,
        )

        if not batch:
            alternatives = [x for x in pop if x != parent]
            if alternatives:
                parent = alternatives[int(rng.integers(len(alternatives)))]
                try:
                    detail = tools.lp_detail(parent)
                except Exception:
                    detail = None
                batch = screen_batch(
                    make_moves(parent, detail), parent, detail=detail,
                    batch_size=batch_size,
                )

        if not batch:
            # Last-resort random multi-switch perturbations.
            for _ in range(40):
                cur = set(parent)
                for _ in range(3 + int(rng.integers(4))):
                    arcs_now = list(cur)
                    p, q = rng.choice(len(arcs_now), size=2, replace=False)
                    a, b = arcs_now[int(p)]
                    c, d = arcs_now[int(q)]
                    if a == d or c == b or (a, d) in cur or (c, b) in cur:
                        continue
                    cur.remove((a, b))
                    cur.remove((c, d))
                    cur.add((a, d))
                    cur.add((c, b))
                kicked = tuple(sorted(cur))
                if (
                    kicked != parent and kicked not in seen
                    and not tools.is_cached(kicked)
                ):
                    batch = [kicked]
                    break

        if not batch:
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

        batch_best = min(
            (seen.get(net, float("inf")) for net in batch),
            default=float("inf"),
        )
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
