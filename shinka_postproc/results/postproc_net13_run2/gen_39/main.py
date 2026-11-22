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
    """Improve repaired candidate networks with dual-guided, diverse local search."""
    n = tools.n
    seen = {}
    best_net = None
    best_score = float("inf")
    parent_stalls = {}
    energy_cache = {}

    def canon(arcs):
        return tuple(sorted(tuple(map(int, a)) for a in arcs))

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
            best_net, best_score = net, score
        return score

    def distance(a, b):
        return len(set(a) ^ set(b))

    def choose_starts(repaired, limit):
        ordered = sorted(repaired, key=energy)
        chosen = []
        for net in ordered:
            if all(distance(net, old) >= 4 for old in chosen):
                chosen.append(net)
                if len(chosen) >= limit:
                    break
        if len(chosen) < limit:
            for net in ordered:
                if net not in chosen:
                    chosen.append(net)
                    if len(chosen) >= limit:
                        break
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
        return list(initial[0]) if initial else []

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
        # Bias toward elite networks, but continue sampling diverse regions.
        if rng.random() < 0.50:
            return pop[int(rng.integers(min(7, len(pop))))]
        return pop[int(rng.integers(len(pop)))]

    def switch(net, p, q):
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

        # Thoroughly inspect the two-switches that can remove a bottleneck arc.
        for p in range(len(arcs)):
            for q in range(p + 1, len(arcs)):
                if hot and arcs[p] not in hot and arcs[q] not in hot:
                    continue
                cand = switch(net, p, q)
                if cand is not None:
                    moves.add(cand)

        # Keep some non-bottleneck moves to avoid overfitting to dual prices.
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

        # Three-link rotations provide moves that are not a single two-switch.
        if len(arcs) >= 3:
            stalls = parent_stalls.get(net, 0)
            trials = min(280, 80 + 35 * (stalls // 3))
            for _ in range(trials):
                ids = rng.choice(len(arcs), size=3, replace=False)
                (a, b), (c, d), (e, f) = (arcs[int(i)] for i in ids)
                old = {(a, b), (c, d), (e, f)}
                if hot and not (old & hot):
                    continue
                remaining = current - old
                for new in (
                    {(a, d), (c, f), (e, b)},
                    {(a, f), (c, b), (e, d)},
                ):
                    if len(new) != 3 or any(x == y for x, y in new):
                        continue
                    if new & remaining:
                        continue
                    moves.add(tuple(sorted(remaining | new)))

        # A few compound kicks sample farther from the local neighborhood.
        for _ in range(16):
            cur = set(net)
            for _ in range(2 + int(rng.integers(4))):
                cur_arcs = list(cur)
                if len(cur_arcs) < 2:
                    break
                ids = rng.choice(len(cur_arcs), size=2, replace=False)
                (a, b), (c, d) = (cur_arcs[int(ids[0])], cur_arcs[int(ids[1])])
                if a == d or c == b or (a, d) in cur or (c, b) in cur:
                    continue
                cur.remove((a, b))
                cur.remove((c, d))
                cur.add((a, d))
                cur.add((c, b))
            cand = tuple(sorted(cur))
            if cand != net:
                moves.add(cand)

        # Relink to a strong but topologically distant member of the archive.
        pop = archive()
        targets = [x for x in pop if x != net and distance(x, net) >= 8]
        if targets:
            targets.sort(key=lambda x: seen.get(x, float("inf")))
            target = targets[int(rng.integers(min(6, len(targets))))]
            target_set = set(target)
            walk = net
            for _ in range(5):
                walk_arcs = list(walk)
                walk_set = set(walk)
                old_overlap = len(walk_set & target_set)
                best_overlap = old_overlap
                advancing = []
                for p in range(len(walk_arcs)):
                    for q in range(p + 1, len(walk_arcs)):
                        cand = switch(walk, p, q)
                        if cand is None:
                            continue
                        overlap = len(set(cand) & target_set)
                        if overlap > best_overlap:
                            best_overlap = overlap
                            advancing = [cand]
                        elif overlap == best_overlap and overlap > old_overlap:
                            advancing.append(cand)
                if not advancing:
                    break
                walk = advancing[int(rng.integers(len(advancing)))]
                moves.add(walk)

        return [m for m in moves if m != net and tools.is_valid(m)]

    def screen_batch(moves, parent, detail, batch_size=9):
        fresh = [
            m for m in moves
            if m not in seen and not tools.is_cached(m)
        ]
        if not fresh:
            return []

        prices = detail.get("prices", {}) if detail else {}
        hot = {tuple(map(int, a)) for a, p in prices.items() if p > 0}
        parent_set = set(parent)
        base_energy = energy(parent)

        # First rank by bottleneck removal, then by the Hamiltonian's surrogate.
        ranked = []
        for net in fresh:
            hot_delta = len(set(net) & hot) - len(parent_set & hot)
            ranked.append((hot_delta, energy(net) - base_energy, net))
        ranked.sort(key=lambda x: (x[0], x[1]))

        selected = []
        quality_slots = max(1, batch_size - 2)
        for _, _, net in ranked:
            if all(distance(net, old) >= 2 for old in selected):
                selected.append(net)
                if len(selected) >= quality_slots:
                    break

        # Reserve two calls for proposals that explore a new region.
        remaining = [net for _, _, net in ranked if net not in selected]
        refs = archive()
        while remaining and len(selected) < batch_size and len(selected) < quality_slots + 2:
            novelty = {
                net: min((distance(net, old) for old in refs), default=0)
                for net in remaining
            }
            chosen = max(
                remaining,
                key=lambda x: (novelty[x], -energy(x)),
            )
            selected.append(chosen)
            remaining.remove(chosen)

        # Fill unused slots with the best combined bottleneck/surrogate ranking.
        for _, _, net in ranked:
            if len(selected) >= batch_size:
                break
            if net not in selected:
                selected.append(net)
        return selected[:batch_size]

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

        batch = screen_batch(make_moves(parent, detail), parent, detail)

        if not batch:
            alternatives = [x for x in pop if x != parent]
            if alternatives:
                parent = alternatives[int(rng.integers(len(alternatives)))]
                try:
                    detail = tools.lp_detail(parent)
                except Exception:
                    detail = None
                batch = screen_batch(make_moves(parent, detail), parent, detail)

        if not batch:
            # If the local region is exhausted, try fresh compound perturbations.
            for _ in range(40):
                cur = set(parent)
                arcs = list(cur)
                for _ in range(3 + int(rng.integers(4))):
                    ids = rng.choice(len(arcs), size=2, replace=False)
                    (a, b), (c, d) = (arcs[int(ids[0])], arcs[int(ids[1])])
                    if a == d or c == b or (a, d) in cur or (c, b) in cur:
                        continue
                    cur.remove((a, b))
                    cur.remove((c, d))
                    cur.add((a, d))
                    cur.add((c, b))
                kicked = tuple(sorted(cur))
                if kicked != parent and kicked not in seen and not tools.is_cached(kicked):
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
