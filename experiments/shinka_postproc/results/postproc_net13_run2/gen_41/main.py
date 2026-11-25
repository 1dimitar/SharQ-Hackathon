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
    n = tools.n
    seen = {}
    best_net = None
    best_score = float("inf")
    parent_stalls = {}
    prior = None

    def canon(arcs):
        return tuple(sorted((int(a), int(b)) for a, b in arcs))

    def energy(net):
        try:
            return float(tools.energy(net))
        except Exception:
            return 0.0

    def distance(a, b):
        return len(set(a) ^ set(b))

    def evaluate(net):
        nonlocal best_net, best_score
        net = canon(net)
        if net not in seen:
            seen[net] = tools.lp(net)
        score = seen[net]
        if score < best_score:
            best_score, best_net = score, net
        return score

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
    prior_samples = ordered[:min(40, len(ordered))]
    prior = [[0.0] * n for _ in range(n)]
    counts = [[0.0] * n for _ in range(n)]
    for net in prior_samples:
        for a, b in net:
            counts[a][b] += 1.0
    denom = len(prior_samples) + 1.5
    for a in range(n):
        for b in range(n):
            if a != b:
                prior[a][b] = (counts[a][b] + 1.5) / denom

    initial = []
    for net in ordered:
        if all(distance(net, old) >= 4 for old in initial):
            initial.append(net)
        if len(initial) >= min(24, len(ordered)):
            break
    for net in ordered:
        if len(initial) >= min(40, len(ordered)):
            break
        if net not in initial:
            initial.append(net)

    for net in initial:
        if tools.lp_calls_left <= 0:
            break
        try:
            evaluate(net)
        except Exception as exc:
            if type(exc).__name__ == "BudgetExhausted":
                break
            continue

    if best_net is None:
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
        if rng.random() < 0.50:
            return pop[int(rng.integers(min(8, len(pop))))]
        return pop[int(rng.integers(len(pop)))]

    def switch(net, p, q):
        arcs = list(net)
        (a, b), (c, d) = arcs[p], arcs[q]
        cur = set(net)
        if a == d or c == b or (a, d) in cur or (c, b) in cur:
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
            hot = {
                a for _, a in sorted(
                    ((float(v), tuple(map(int, a))) for a, v in loads.items()),
                    reverse=True
                )[:6]
            }

        for p in range(len(arcs)):
            for q in range(p + 1, len(arcs)):
                if hot and arcs[p] not in hot and arcs[q] not in hot:
                    continue
                cand = switch(net, p, q)
                if cand is not None:
                    moves.add(cand)

        for _ in range(60):
            p = int(rng.integers(len(arcs)))
            q = int(rng.integers(len(arcs) - 1))
            if q >= p:
                q += 1
            cand = switch(net, p, q)
            if cand is not None:
                moves.add(cand)

        if len(arcs) >= 3:
            trials = min(240, 60 * (1 + parent_stalls.get(net, 0) // 4))
            for _ in range(trials):
                ids = rng.choice(len(arcs), size=3, replace=False)
                (a, b), (c, d), (e, f) = (arcs[int(i)] for i in ids)
                old = {(a, b), (c, d), (e, f)}
                if hot and not (old & hot):
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

        stalls = parent_stalls.get(net, 0)
        for _ in range(min(36, 16 + 3 * stalls)):
            cur = set(net)
            for _ in range(2 + int(rng.integers(min(8, 3 + stalls // 2)))):
                now = list(cur)
                i, j = rng.choice(len(now), size=2, replace=False)
                a, b = now[int(i)]
                c, d = now[int(j)]
                if a == d or c == b or (a, d) in cur or (c, b) in cur:
                    continue
                cur.remove((a, b))
                cur.remove((c, d))
                cur.add((a, d))
                cur.add((c, b))
            moves.add(tuple(sorted(cur)))

        elites = sorted(seen, key=seen.get)[:10]
        targets = [t for t in elites if t != net and distance(t, net) >= 8][:5]
        for target in targets:
            target_arcs = set(target)
            walk = set(net)
            for _ in range(4):
                options = set()
                for a, b in target_arcs - walk:
                    outgoing = [x for x in walk if x[0] == a]
                    incoming = [x for x in walk if x[1] == b]
                    for _, x in outgoing:
                        for y, _ in incoming:
                            removed = {(a, x), (y, b)}
                            if y == a or x == b or (y, x) in walk - removed:
                                continue
                            options.add(tuple(sorted((walk - removed) | {(a, b), (y, x)})))
                if not options:
                    break
                max_overlap = max(len(set(x) & target_arcs) for x in options)
                options = [x for x in options if len(set(x) & target_arcs) == max_overlap]
                walk = set(options[int(rng.integers(len(options)))])
                moves.add(tuple(sorted(walk)))

        return [m for m in moves if m != net and tools.is_valid(m)]

    def screen(moves, parent, size=9):
        fresh = [m for m in moves if m not in seen and not tools.is_cached(m)]
        if not fresh:
            return []
        base = energy(parent)
        try:
            detail = tools.lp_detail(parent)
        except Exception:
            detail = None

        prices = detail.get("prices", {}) if detail else {}
        hot = {
            tuple(map(int, arc)): max(0.0, float(price))
            for arc, price in prices.items()
            if float(price) > 0
        }
        if not hot and detail:
            loads = detail.get("loads", {})
            ranked_loads = sorted(
                ((float(value), tuple(map(int, arc))) for arc, value in loads.items()),
                reverse=True,
            )
            hot = {arc: value for value, arc in ranked_loads[:6]}

        ranked = [(energy(net) - base, net) for net in fresh]
        ranked.sort(key=lambda item: item[0])

        # Prioritize proposals that remove costly current links, without letting
        # dual guidance consume the entire batch.
        relief = []
        parent_arcs = set(parent)
        for delta, net in ranked:
            benefit = sum(hot.get(arc, 0.0) for arc in parent_arcs - set(net))
            if benefit > 0:
                relief.append((-benefit, delta, net))
        relief.sort()

        chosen = []
        focused_limit = min(size // 2, len(relief))
        for _, _, net in relief:
            if all(distance(net, old) >= 2 for old in chosen):
                chosen.append(net)
            if len(chosen) >= focused_limit:
                break

        for _, net in ranked:
            if len(chosen) >= min(6, size):
                break
            if net not in chosen and all(distance(net, old) >= 2 for old in chosen):
                chosen.append(net)
        for _, net in ranked:
            if len(chosen) >= size:
                break
            if net not in chosen:
                chosen.append(net)
        return chosen

    def weighted_matching(prob, excluded, temperature):
        assignment = [-1] * n
        used = set()
        visits = [0]

        def rec(done):
            if done == n:
                return True
            visits[0] += 1
            if visits[0] > 3500:
                return False
            source, options = -1, None
            for a in range(n):
                if assignment[a] >= 0:
                    continue
                avail = [b for b in range(n)
                         if b != a and b not in used and (a, b) not in excluded]
                if not avail:
                    return False
                if options is None or len(avail) < len(options):
                    source, options = a, avail
            logits = [math.log(max(1e-5, prob[source][b])) for b in options]
            noise = rng.gumbel(size=len(options))
            order = sorted(
                range(len(options)),
                key=lambda i: logits[i] / max(temperature, 1e-6) + float(noise[i]),
                reverse=True
            )
            for idx in order:
                b = options[idx]
                assignment[source] = b
                used.add(b)
                if rec(done + 1):
                    return True
                used.remove(b)
                assignment[source] = -1
            return False

        return {(a, assignment[a]) for a in range(n)} if rec(0) else None

    def sample_network(model, temperature):
        first = weighted_matching(model, set(), temperature)
        if first is None:
            return None
        second = weighted_matching(model, first, temperature)
        if second is None:
            return None
        net = tuple(sorted(first | second))
        return net if len(net) == 2 * n and tools.is_valid(net) else None

    def update_model():
        ranked = sorted(seen, key=seen.get)
        elite = []
        for net in ranked:
            if all(distance(net, old) >= 2 for old in elite):
                elite.append(net)
            if len(elite) >= 12:
                break
        for net in ranked:
            if len(elite) >= 24:
                break
            if net not in elite:
                elite.append(net)
        counts = [[0.0] * n for _ in range(n)]
        total = 0.0
        for rank, net in enumerate(elite):
            w = math.exp(-rank / 9.0)
            total += w
            for a, b in net:
                counts[a][b] += w
        for a in range(n):
            for b in range(n):
                if a != b:
                    estimate = (counts[a][b] + 0.12) / (total + 0.12)
                    model[a][b] = 0.72 * model[a][b] + 0.28 * estimate

    model = [row[:] for row in prior]
    last_update = len(seen)
    rounds = 0
    while tools.lp_calls_left > 0:
        rounds += 1
        if len(seen) - last_update >= 20:
            update_model()
            last_update = len(seen)

        # Most calls go to local neighborhoods; matching proposals help escape
        # basins without displacing the better-performing exchange search.
        if rng.random() < 0.12:
            temperature = 1.25 + min(1.0, rounds / 1200.0)
            if rng.random() < 0.25:
                sampling = [
                    [0.65 * model[a][b] + 0.35 * prior[a][b] for b in range(n)]
                    for a in range(n)
                ]
            else:
                sampling = model
            net = sample_network(sampling, temperature)
            if net is not None and net not in seen and not tools.is_cached(net):
                try:
                    evaluate(net)
                except Exception as exc:
                    if type(exc).__name__ == "BudgetExhausted":
                        break
            continue

        pop = archive()
        parent = select_parent(pop)
        if parent is None:
            break
        try:
            detail = tools.lp_detail(parent)
        except Exception:
            detail = None
        batch = screen(make_moves(parent, detail), parent)
        if not batch:
            alternatives = [x for x in pop if x != parent]
            if alternatives:
                parent = alternatives[int(rng.integers(len(alternatives)))]
                try:
                    detail = tools.lp_detail(parent)
                except Exception:
                    detail = None
                batch = screen(make_moves(parent, detail), parent)
        if not batch:
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
        batch_best = min((seen.get(net, float("inf")) for net in batch), default=float("inf"))
        parent_stalls[parent] = 0 if batch_best < parent_score else parent_stalls.get(parent, 0) + 1

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
