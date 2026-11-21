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
    """Search valid networks using an adaptive, diverse archive."""
    n = tools.n
    scored = {}
    best_net = None
    best_score = float("inf")
    visits = {}
    gains = {}
    stalls = {}
    archive_limit = 36

    def canon(arcs):
        return tuple(sorted((int(a), int(b)) for a, b in arcs))

    def distance(a, b):
        return len(set(a) ^ set(b))

    def energy(net):
        try:
            return float(tools.energy(net))
        except Exception:
            return 0.0

    def evaluate(net):
        nonlocal best_net, best_score
        net = canon(net)
        if net not in scored:
            scored[net] = tools.lp(net)
        value = scored[net]
        if value < best_score:
            best_net, best_score = net, value
        return value

    def rebuild_archive():
        """Retain strong networks while reserving space for distinct regions."""
        ranked = sorted(scored, key=scored.get)
        out = []
        for net in ranked:
            if all(distance(net, old) >= 4 for old in out):
                out.append(net)
            if len(out) >= 22:
                break
        for net in ranked:
            if len(out) >= archive_limit:
                break
            if net not in out:
                out.append(net)
        return out

    # Repair is free. Seed from a surrogate-ranked, topologically varied set.
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
        if len(starts) >= 28:
            break
    for net in ordered:
        if len(starts) >= min(44, len(ordered)):
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

    if best_net is None:
        return list(starts[0])

    def select_parent(pop):
        """Rank-based quality plus novelty and individual search progress."""
        if not pop:
            return best_net

        # Mix a quality-biased pool with the full diverse archive.
        if rng.random() < 0.68:
            pool = pop[:min(12, len(pop))]
        else:
            pool = pop

        weights = []
        for rank, net in enumerate(pool):
            v = visits.get(net, 0)
            g = gains.get(net, 0)
            s = stalls.get(net, 0)
            quality = 1.0 / (1.0 + 0.10 * rank)
            productivity = 1.0 + min(2.0, 0.45 * g)
            exploration = 1.0 / ((1.0 + v) ** 0.42)
            weights.append(quality * productivity * exploration / (1.0 + 0.08 * s))

        total = sum(weights)
        probs = [w / total for w in weights]
        parent = pool[int(rng.choice(len(pool), p=probs))]
        visits[parent] = visits.get(parent, 0) + 1
        return parent

    def two_switch(net, p, q):
        arcs = list(net)
        a, b = arcs[p]
        c, d = arcs[q]
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

    def generate_moves(net, detail, round_no):
        arcs = list(net)
        current = set(net)
        moves = set()

        prices = detail.get("prices", {}) if detail else {}
        hot = {tuple(map(int, a)) for a, price in prices.items() if price > 0}
        if not hot and detail:
            loads = detail.get("loads", {})
            hot = {
                arc for _, arc in sorted(
                    ((float(load), tuple(map(int, arc)))
                     for arc, load in loads.items()),
                    reverse=True,
                )[:7]
            }

        # Thoroughly explore exchanges touching an LP bottleneck. Every few
        # iterations, include the entire two-switch neighborhood.
        broad = (round_no % 6 == 0) or not hot
        for p in range(len(arcs)):
            for q in range(p + 1, len(arcs)):
                if not broad and arcs[p] not in hot and arcs[q] not in hot:
                    continue
                move = two_switch(net, p, q)
                if move is not None:
                    moves.add(move)

        # Sample non-bottleneck exchanges to escape dual-degeneracy traps.
        for _ in range(85):
            p = int(rng.integers(len(arcs)))
            q = int(rng.integers(len(arcs) - 1))
            if q >= p:
                q += 1
            move = two_switch(net, p, q)
            if move is not None:
                moves.add(move)

        # Three-edge cyclic rotations preserve every node's in/out degree.
        trials = 100 + min(220, 15 * stalls.get(net, 0))
        for _ in range(trials):
            ids = rng.choice(len(arcs), size=3, replace=False)
            (a, b), (c, d), (e, f) = (arcs[int(i)] for i in ids)
            old = {(a, b), (c, d), (e, f)}
            if hot and round_no % 3 and not old.intersection(hot):
                continue
            for new in ({(a, d), (c, f), (e, b)},
                        {(a, f), (c, b), (e, d)}):
                if len(new) != 3 or any(x == y for x, y in new):
                    continue
                if new & (current - old):
                    continue
                move = tuple(sorted((current - old) | new))
                if move != net:
                    moves.add(move)

        # Relink a small number of promising parents toward a distant archive
        # member, adding intermediate topologies as additional proposals.
        if len(scored) > 1 and round_no % 2 == 0:
            ranked = sorted(scored, key=scored.get)
            targets = [
                target for target in ranked[:12]
                if target != net and distance(target, net) >= 6
            ]
            if targets:
                target = targets[int(rng.integers(len(targets)))]
                target_edges = set(target)
                path = net
                for _ in range(5):
                    path_arcs = list(path)
                    path_set = set(path)
                    overlap = len(path_set & target_edges)
                    improving = []
                    for p in range(len(path_arcs)):
                        for q in range(p + 1, len(path_arcs)):
                            move = two_switch(path, p, q)
                            if move is not None:
                                gain = len(set(move) & target_edges) - overlap
                                if gain > 0:
                                    improving.append((gain, move))
                    if not improving:
                        break
                    best_gain = max(g for g, _ in improving)
                    choices = [m for g, m in improving if g == best_gain]
                    path = choices[int(rng.integers(len(choices)))]
                    moves.add(path)

        return [move for move in moves
                if move != net and tools.is_valid(move)]

    def screen(moves, parent, width):
        """Spend LP calls on good surrogate proposals without over-clustering."""
        fresh = [
            move for move in moves
            if move not in scored and not tools.is_cached(move)
        ]
        if not fresh:
            return []

        base_energy = energy(parent)
        ranked = sorted(fresh, key=lambda move: energy(move) - base_energy)
        chosen = []

        # Favor leading surrogate candidates, but require some structural spread.
        for move in ranked:
            if all(distance(move, old) >= 2 for old in chosen):
                chosen.append(move)
            if len(chosen) >= max(2, width - 1):
                break

        # Preserve one slot for a less predictable proposal.
        tail = [move for move in fresh if move not in chosen]
        if tail and len(chosen) < width:
            rng.shuffle(tail)
            chosen.append(tail[0])

        for move in ranked:
            if len(chosen) >= width:
                break
            if move not in chosen:
                chosen.append(move)
        return chosen[:width]

    round_no = 0
    while tools.lp_calls_left > 0:
        round_no += 1
        pop = rebuild_archive()
        parent = select_parent(pop)

        try:
            detail = tools.lp_detail(parent)
        except Exception:
            detail = None

        width = 4 if round_no % 5 else 6
        batch = screen(generate_moves(parent, detail, round_no), parent, width)

        if not batch:
            alternatives = [net for net in pop if net != parent]
            if alternatives:
                parent = alternatives[int(rng.integers(len(alternatives)))]
                try:
                    detail = tools.lp_detail(parent)
                except Exception:
                    detail = None
                batch = screen(generate_moves(parent, detail, round_no), parent, width)

        # If the local neighborhood is exhausted, create a multi-switch kick.
        if not batch:
            for _ in range(50):
                current = set(parent)
                for _ in range(2 + int(rng.integers(4))):
                    arcs_now = list(current)
                    p, q = rng.choice(len(arcs_now), size=2, replace=False)
                    a, b = arcs_now[int(p)]
                    c, d = arcs_now[int(q)]
                    if a == d or c == b or (a, d) in current or (c, b) in current:
                        continue
                    current.remove((a, b))
                    current.remove((c, d))
                    current.add((a, d))
                    current.add((c, b))
                kicked = tuple(sorted(current))
                if (kicked != parent and kicked not in scored
                        and not tools.is_cached(kicked) and tools.is_valid(kicked)):
                    batch = [kicked]
                    break

        if not batch:
            continue

        parent_score = scored.get(parent, float("inf"))
        batch_best = float("inf")
        try:
            for move in batch:
                if tools.lp_calls_left <= 0:
                    break
                value = evaluate(move)
                batch_best = min(batch_best, value)
        except Exception as exc:
            if type(exc).__name__ == "BudgetExhausted":
                break

        if batch_best < parent_score:
            stalls[parent] = 0
            gains[parent] = gains.get(parent, 0) + 1
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
