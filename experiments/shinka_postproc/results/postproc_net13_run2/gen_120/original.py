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
    parent_visits = {}

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
        """Favor strong parents while allocating search to underexplored members."""
        if not pop:
            return best_net
        weights = []
        for rank, net in enumerate(pop):
            quality = 1.0 / ((1.0 + rank) ** 0.65)
            underexplored = 1.0 / ((1.0 + parent_visits.get(net, 0)) ** 0.7)
            stale = 1.0 + 0.12 * min(5, parent_stalls.get(net, 0))
            weights.append(quality * underexplored * stale)

        pick = float(rng.random()) * sum(weights)
        for net, weight in zip(pop, weights):
            pick -= weight
            if pick <= 0:
                return net
        return pop[0]

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

        # Use a small number of switches guided by arcs shared across strong,
        # structurally diverse archive members. Sampling among the leaders
        # avoids collapsing every neighborhood onto the same consensus topology.
        refs = archive()[:16]
        if refs:
            arc_frequency = {}
            for ref in refs:
                for arc in ref:
                    arc_frequency[arc] = arc_frequency.get(arc, 0) + 1
            guided = []
            for p in range(len(arcs)):
                for q in range(p + 1, len(arcs)):
                    cand = switch(net, p, q)
                    if cand is None:
                        continue
                    added = set(cand) - current
                    removed = current - set(cand)
                    gain = sum(arc_frequency.get(a, 0) for a in added)
                    gain -= sum(arc_frequency.get(a, 0) for a in removed)
                    if gain > 0:
                        guided.append((gain, cand))
            guided.sort(key=lambda item: item[0], reverse=True)
            shortlist = guided[:min(16, len(guided))]
            if shortlist:
                order = rng.permutation(len(shortlist))
                for idx in order[:min(8, len(shortlist))]:
                    moves.add(shortlist[int(idx)][1])

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

        # Escalate rotation sampling for parents whose neighborhoods have stalled.
        # Rotations preserve degrees and can cross barriers that two-switches miss.
        if len(arcs) >= 3:
            stalls = parent_stalls.get(net, 0)
            rotation_trials = min(350, 70 * (1 + stalls // 3))
            for _ in range(rotation_trials):
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
                    cand = tuple(sorted((current - old) | new))
                    moves.add(cand)

        # Occasional multi-switch kicks broaden the neighborhood beyond one step.
        for _ in range(24):
            cur = set(net)
            for _ in range(2 + int(rng.integers(4))):
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

        # Relink toward a strong, distant archive member. Each step is a legal
        # two-switch that increases overlap with the target, so intermediates
        # provide a directed bridge between otherwise separate search regions.
        targets = [
            x for x in archive()
            if x != net and distance(x, net) >= 8
        ]
        if targets:
            targets.sort(key=lambda x: seen.get(x, float("inf")))
            target = targets[int(rng.integers(min(6, len(targets))))]
            target_set = set(target)
            walk = net
            for _ in range(5):
                walk_arcs = list(walk)
                old_overlap = len(set(walk) & target_set)
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

    def screen_batch(moves, parent, batch_size=9):
        """Favor good surrogate moves but reserve one slot for archive novelty."""
        fresh = [
            m for m in moves
            if m not in seen and not tools.is_cached(m)
        ]
        if not fresh:
            return []

        base_e = energy(parent)
        ranked = sorted(
            ((energy(net) - base_e, net) for net in fresh),
            key=lambda item: item[0],
        )

        # Preserve most of the batch for promising moves, spreading the
        # initial picks across distinct topologies.
        quality_limit = max(0, batch_size - 1)
        selected = []
        for _, net in ranked:
            if all(distance(net, old) >= 2 for old in selected):
                selected.append(net)
            if len(selected) >= min(6, quality_limit):
                break

        for _, net in ranked:
            if len(selected) >= quality_limit:
                break
            if net not in selected:
                selected.append(net)

        # Spend the reserved call on a candidate far from every scored
        # archive member; use surrogate energy only to break novelty ties.
        remaining = [net for _, net in ranked if net not in selected]
        if remaining and len(selected) < batch_size:
            refs = archive()
            def novelty(net):
                return min((distance(net, old) for old in refs), default=len(net))
            novel = max(remaining, key=lambda net: (novelty(net), -energy(net)))
            selected.append(novel)

        return selected[:batch_size]

    # A bounded archive drives a sequence of short, diverse neighborhood searches.
    rounds = 0
    while tools.lp_calls_left > 0:
        rounds += 1
        pop = archive()
        parent = select_parent(pop)
        if parent is None:
            break
        parent_visits[parent] = parent_visits.get(parent, 0) + 1

        try:
            detail = tools.lp_detail(parent)
        except Exception:
            detail = None

        moves = make_moves(parent, detail)
        batch = screen_batch(moves, parent)

        # If a neighborhood has already been exhausted, perturb another archive
        # member rather than repeatedly revisiting the same parent.
        if not batch:
            alternatives = [x for x in pop if x != parent]
            if alternatives:
                parent = alternatives[int(rng.integers(len(alternatives)))]
                parent_visits[parent] = parent_visits.get(parent, 0) + 1
                try:
                    detail = tools.lp_detail(parent)
                except Exception:
                    detail = None
                batch = screen_batch(make_moves(parent, detail), parent)

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
