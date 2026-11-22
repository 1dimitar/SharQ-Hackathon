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
    """Search repaired candidates with a diverse archive and screened neighborhoods."""
    seen = {}
    energy_cache = {}
    uses = {}
    stalls = {}
    best_net = None
    best_score = float("inf")
    n_arcs = 2 * int(tools.n)

    def canon(arcs):
        return tuple(sorted((int(a), int(b)) for a, b in arcs))

    def distance(a, b):
        return len(set(a) ^ set(b))

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

    def switch(net, p, q):
        """Two-switch preserving in/out degrees."""
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

    # Stage 1: repair the entire pool for free, then score a compact,
    # energy-ranked subset that retains topology diversity.
    repaired = set()
    for candidate in candidates:
        try:
            net = canon(tools.repair(candidate))
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

    ordered = sorted(repaired, key=energy)
    starts = []
    for net in ordered:
        if all(distance(net, old) >= 4 for old in starts):
            starts.append(net)
        if len(starts) >= min(26, len(ordered)):
            break
    for net in ordered:
        if len(starts) >= min(42, len(ordered)):
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

    if not seen:
        return list(starts[0])

    # Stage 2: maintain an elite set plus structurally distinct representatives.
    def archive():
        ranked = sorted(seen, key=seen.get)
        chosen = []
        for net in ranked:
            if all(distance(net, old) >= 4 for old in chosen):
                chosen.append(net)
            if len(chosen) >= 18:
                break
        for net in ranked:
            if len(chosen) >= 42:
                break
            if net not in chosen:
                chosen.append(net)
        return chosen

    def choose_parent(pop):
        if not pop:
            return best_net
        weights = []
        for rank, net in enumerate(pop):
            nearest = min(
                (distance(net, other) for other in pop if other != net),
                default=26,
            )
            novelty = min(1.0, nearest / 26.0)
            count = uses.get(net, 0)
            quality = 1.0 / (1.0 + 0.055 * rank)
            explore = 1.0 + 0.45 / math.sqrt(count + 1.0)
            weights.append(quality * (0.65 + novelty) * explore)

        pick = float(rng.random()) * sum(weights)
        for net, weight in zip(pop, weights):
            pick -= weight
            if pick <= 0:
                return net
        return pop[-1]

    def generate_moves(parent, detail, pop):
        arcs = list(parent)
        current = set(parent)
        moves = set()

        prices = detail.get("prices", {}) if detail else {}
        hot = {tuple(map(int, arc)) for arc, price in prices.items() if price > 0}
        if not hot and detail:
            loads = detail.get("loads", {})
            ranked_loads = sorted(
                ((float(value), tuple(map(int, arc))) for arc, value in loads.items()),
                reverse=True,
            )
            hot = {arc for _, arc in ranked_loads[:7]}

        # Focus the exhaustive neighborhood on links identified by LP dual prices.
        for p in range(len(arcs)):
            for q in range(p + 1, len(arcs)):
                if hot and arcs[p] not in hot and arcs[q] not in hot:
                    continue
                candidate = switch(parent, p, q)
                if candidate is not None:
                    moves.add(candidate)

        # Keep a broad sample of non-focused switches to counter dual degeneracy.
        for _ in range(100):
            if len(arcs) < 2:
                break
            p = int(rng.integers(len(arcs)))
            q = int(rng.integers(len(arcs) - 1))
            if q >= p:
                q += 1
            candidate = switch(parent, p, q)
            if candidate is not None:
                moves.add(candidate)

        # Degree-preserving three-link rotations provide larger neighborhood steps.
        trials = min(240, 80 + 30 * stalls.get(parent, 0))
        for _ in range(trials):
            if len(arcs) < 3:
                break
            ids = rng.choice(len(arcs), size=3, replace=False)
            (a, b), (c, d), (e, f) = (arcs[int(i)] for i in ids)
            old = {(a, b), (c, d), (e, f)}
            if hot and not old.intersection(hot):
                continue
            for new in (
                {(a, d), (c, f), (e, b)},
                {(a, f), (c, b), (e, d)},
            ):
                if len(new) != 3 or any(x == y for x, y in new):
                    continue
                if new.intersection(current - old):
                    continue
                moves.add(tuple(sorted((current - old) | new)))

        # Bounded relinking: walk several valid switches toward a strong,
        # structurally distant archive member and retain the intermediate nets.
        targets = [
            net for net in pop
            if net != parent and distance(net, parent) >= 8
        ]
        if targets:
            targets.sort(key=lambda net: seen.get(net, float("inf")))
            target = targets[int(rng.integers(min(10, len(targets))))]
            current_net = parent
            target_set = set(target)
            for _ in range(5):
                best_dist = distance(current_net, target)
                improving = []
                arcs_now = list(current_net)
                for p in range(len(arcs_now)):
                    for q in range(p + 1, len(arcs_now)):
                        candidate = switch(current_net, p, q)
                        if candidate is None:
                            continue
                        d = distance(candidate, target)
                        if d < best_dist:
                            best_dist = d
                            improving = [candidate]
                        elif d == best_dist:
                            improving.append(candidate)
                if not improving:
                    break
                current_net = improving[int(rng.integers(len(improving)))]
                moves.add(current_net)
                if set(current_net) == target_set:
                    break

        # Short random kicks provide escape moves not present in the local shell.
        for _ in range(18):
            cur = set(parent)
            for _ in range(2 + int(rng.integers(4))):
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
            if len(cur) == n_arcs:
                moves.add(tuple(sorted(cur)))

        return [
            net for net in moves
            if net != parent and tools.is_valid(net)
        ]

    # Stage 3: rank fresh moves by the pool-specific surrogate, while reserving
    # batch slots for diverse alternatives and occasional tail exploration.
    def screen(moves, parent, batch_size=8):
        fresh = [
            net for net in moves
            if net not in seen and not tools.is_cached(net)
        ]
        if not fresh:
            return []

        base_energy = energy(parent)
        ranked = sorted(fresh, key=lambda net: energy(net) - base_energy)
        stalls_now = stalls.get(parent, 0)
        if stalls_now >= 5:
            batch_size = 11
        if stalls_now >= 10:
            batch_size = 14
        batch_size = min(batch_size, len(ranked))

        selected = []
        # Greedily retain diversity among the surrogate-favored proposals.
        for net in ranked:
            if all(distance(net, old) >= 2 for old in selected):
                selected.append(net)
            if len(selected) >= min(5, batch_size):
                break

        for net in ranked:
            if len(selected) >= batch_size:
                break
            if net not in selected:
                selected.append(net)

        # Randomize a few remaining candidates rather than always taking only
        # the most optimistic surrogate tail.
        if len(selected) < batch_size:
            tail = [net for net in fresh if net not in selected]
            rng.shuffle(tail)
            selected.extend(tail[:batch_size - len(selected)])
        return selected[:batch_size]

    # Stage 4: repeated archive-guided neighborhood batches consume the LP budget.
    rounds = 0
    while tools.lp_calls_left > 0:
        rounds += 1
        pop = archive()
        parent = choose_parent(pop)
        if parent is None:
            break

        try:
            detail = tools.lp_detail(parent)
        except Exception:
            detail = None

        batch = screen(generate_moves(parent, detail, pop), parent)

        # If this region is exhausted, try another archive member before stopping.
        if not batch:
            alternatives = [net for net in pop if net != parent]
            rng.shuffle(alternatives)
            for alternative in alternatives[:8]:
                try:
                    alt_detail = tools.lp_detail(alternative)
                except Exception:
                    alt_detail = None
                batch = screen(
                    generate_moves(alternative, alt_detail, pop),
                    alternative,
                )
                if batch:
                    parent = alternative
                    break

        if not batch:
            # All generated neighborhoods may be cached; try fresh random kicks.
            for _ in range(50):
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
                candidate = tuple(sorted(cur))
                if (candidate != parent and candidate not in seen
                        and not tools.is_cached(candidate)
                        and tools.is_valid(candidate)):
                    batch = [candidate]
                    break

        if not batch:
            if rounds > 2000:
                break
            continue

        parent_score = seen.get(parent, float("inf"))
        batch_best = float("inf")
        for net in batch:
            if tools.lp_calls_left <= 0:
                break
            try:
                batch_best = min(batch_best, evaluate(net))
            except Exception as exc:
                if type(exc).__name__ == "BudgetExhausted":
                    return list(best_net) if best_net is not None else []
                continue

        uses[parent] = uses.get(parent, 0) + 1
        if batch_best < parent_score:
            stalls[parent] = 0
        else:
            stalls[parent] = stalls.get(parent, 0) + 1

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
