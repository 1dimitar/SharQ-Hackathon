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
    """Search repaired candidates with a diverse archive and guided local moves."""
    n = tools.n
    scored = {}
    best_net = None
    best_score = float("inf")
    visits = {}
    stalls = {}

    def canon(arcs):
        return tuple(sorted(tuple(map(int, a)) for a in arcs))

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
        if net not in scored:
            scored[net] = tools.lp(net)
        value = scored[net]
        if value < best_score:
            best_net, best_score = net, value
        return value

    # Repair all raw samples for free, removing duplicates and invalid results.
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

    # Start with energy-ranked seeds while ensuring that early LP calls cover
    # different topologies rather than near-identical repaired samples.
    ordered = sorted(repaired, key=energy)
    initial = []
    for net in ordered:
        if all(distance(net, old) >= 4 for old in initial):
            initial.append(net)
        if len(initial) >= 22:
            break
    for net in ordered:
        if len(initial) >= min(34, len(ordered)):
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

    if not scored:
        return list(initial[0])

    def archive():
        """Return a small quality- and diversity-balanced set of scored nets."""
        ranked = sorted(scored, key=scored.get)
        out = []
        for net in ranked:
            if all(distance(net, old) >= 4 for old in out):
                out.append(net)
            if len(out) >= 16:
                break
        for net in ranked:
            if len(out) >= 28:
                break
            if net not in out:
                out.append(net)
        return out

    def choose_parent(pop):
        if not pop:
            return best_net
        weights = []
        for rank, net in enumerate(pop):
            quality = 1.0 / (1.0 + 0.13 * rank)
            exploration = 1.0 / (1.0 + 0.22 * visits.get(net, 0))
            stale = 1.0 + min(1.5, 0.12 * stalls.get(net, 0))
            weights.append(quality * exploration * stale)
        if rng.random() < 0.18:
            return pop[int(rng.integers(len(pop)))]
        return pop[int(rng.choice(len(pop), p=np.asarray(weights) / sum(weights)))]

    def switch(net, i, j):
        arcs = list(net)
        (a, b), (c, d) = arcs[i], arcs[j]
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
        hot = {
            tuple(map(int, link))
            for link, price in prices.items()
            if float(price) > 0
        }
        loads = detail.get("loads", {}) if detail else {}
        if not hot and loads:
            high = sorted(
                ((float(value), tuple(map(int, link))) for link, value in loads.items()),
                reverse=True,
            )
            hot = {link for _, link in high[:6]}

        # Bottleneck-focused two-switches, plus a bounded unrestricted sample.
        for i in range(len(arcs)):
            for j in range(i + 1, len(arcs)):
                if hot and arcs[i] not in hot and arcs[j] not in hot:
                    continue
                move = switch(net, i, j)
                if move is not None:
                    moves.add(move)

        for _ in range(38):
            i = int(rng.integers(len(arcs)))
            j = int(rng.integers(len(arcs) - 1))
            if j >= i:
                j += 1
            move = switch(net, i, j)
            if move is not None:
                moves.add(move)

        # Sample degree-preserving three-arc rotations. A subset touch a
        # bottleneck, while the remainder maintain broader exploration.
        if len(arcs) >= 3:
            trials = 58 + min(60, 12 * stalls.get(net, 0))
            for t in range(trials):
                ids = rng.choice(len(arcs), size=3, replace=False)
                (a, b), (c, d), (e, f) = (arcs[int(k)] for k in ids)
                old = {(a, b), (c, d), (e, f)}
                if t < trials * 0.65 and hot and not (old & hot):
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

        # A few short kicks can jump out of a two-switch local basin.
        for _ in range(9):
            cur = set(net)
            for _ in range(2 + int(rng.integers(4))):
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
            if cur != current:
                moves.add(tuple(sorted(cur)))

        # Add a small number of archive-guided switches: insert a target arc
        # while preserving both in- and out-degrees.
        guides = sorted(scored, key=scored.get)[:6]
        for target in guides:
            if target == net:
                continue
            missing = list(set(target) - current)
            if len(missing) > 12:
                missing = [missing[int(k)] for k in rng.choice(len(missing), 12, replace=False)]
            for a, b in missing:
                outgoing = [edge for edge in arcs if edge[0] == a]
                incoming = [edge for edge in arcs if edge[1] == b]
                for _, x in outgoing:
                    for y, _ in incoming:
                        old = {(a, x), (y, b)}
                        if len(old) != 2 or y == a or x == b:
                            continue
                        if (y, x) in current - old:
                            continue
                        moves.add(tuple(sorted((current - old) | {(a, b), (y, x)})))

        return [
            move for move in moves
            if move != net and move not in scored and not tools.is_cached(move)
            and tools.is_valid(move)
        ]

    def select_batch(moves, parent, size=6):
        if not moves:
            return []

        base = energy(parent)
        ranked = sorted(
            ((energy(net) - base, net) for net in moves),
            key=lambda item: item[0],
        )
        if not ranked:
            return []

        # Keep strong surrogate proposals, but include diverse and randomly
        # chosen alternatives so the surrogate is not an absolute filter.
        shortlist = ranked[:min(len(ranked), 36)]
        selected = [shortlist[0][1]]

        while len(selected) < min(3, size):
            options = [net for _, net in shortlist if net not in selected]
            if not options:
                break
            chosen = max(
                options,
                key=lambda net: (
                    min(distance(net, old) for old in selected),
                    -next(delta for delta, item in ranked if item == net),
                ),
            )
            selected.append(chosen)

        for _, net in ranked:
            if len(selected) >= min(4, size):
                break
            if net not in selected:
                selected.append(net)

        tail = [net for _, net in ranked[36:] if net not in selected]
        if tail and len(selected) < size:
            selected.append(tail[int(rng.integers(len(tail)))])

        for _, net in ranked:
            if len(selected) >= size:
                break
            if net not in selected:
                selected.append(net)
        return selected[:size]

    rounds_without_progress = 0
    while tools.lp_calls_left > 0:
        pop = archive()
        parent = choose_parent(pop)
        if parent is None:
            break

        visits[parent] = visits.get(parent, 0) + 1
        try:
            detail = tools.lp_detail(parent)
        except Exception:
            detail = None

        batch = select_batch(make_moves(parent, detail), parent)

        # If the selected neighborhood is exhausted, try another archive
        # member before generating a random kick.
        if not batch:
            alternatives = [net for net in pop if net != parent]
            alternatives.sort(key=lambda net: visits.get(net, 0))
            for alternative in alternatives[:5]:
                try:
                    alt_detail = tools.lp_detail(alternative)
                except Exception:
                    alt_detail = None
                batch = select_batch(make_moves(alternative, alt_detail), alternative)
                if batch:
                    parent = alternative
                    visits[parent] = visits.get(parent, 0) + 1
                    break

        if not batch:
            # Last-resort fresh multi-switch perturbations.
            for _ in range(30):
                cur = set(parent)
                for _ in range(3 + int(rng.integers(4))):
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
                kicked = tuple(sorted(cur))
                if (
                    kicked != parent and kicked not in scored
                    and not tools.is_cached(kicked) and tools.is_valid(kicked)
                ):
                    batch = [kicked]
                    break

        if not batch:
            rounds_without_progress += 1
            if rounds_without_progress > 100:
                break
            continue

        parent_score = scored.get(parent, float("inf"))
        previous_best = best_score
        for net in batch:
            if tools.lp_calls_left <= 0:
                break
            try:
                evaluate(net)
            except Exception as exc:
                if type(exc).__name__ == "BudgetExhausted":
                    return list(best_net) if best_net is not None else []
                continue

        batch_best = min((scored.get(net, float("inf")) for net in batch), default=float("inf"))
        if batch_best < parent_score:
            stalls[parent] = 0
        else:
            stalls[parent] = stalls.get(parent, 0) + 1

        if best_score < previous_best:
            rounds_without_progress = 0
        else:
            rounds_without_progress += 1

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
