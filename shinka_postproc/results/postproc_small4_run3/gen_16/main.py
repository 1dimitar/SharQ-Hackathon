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
    """Improve repaired QAOA networks with adaptive, diverse LP-guided search."""
    n = int(tools.n)
    seen = {}
    best_net = None
    best_score = float("inf")
    parent_stalls = {}
    parent_visits = {}
    global_stall = 0

    def canon(arcs):
        return tuple(sorted(tuple(map(int, a)) for a in arcs))

    def distance(a, b):
        return len(set(a) ^ set(b))

    def evaluate(net):
        nonlocal best_net, best_score, global_stall
        net = canon(net)
        if net not in seen:
            score = tools.lp(net)
            seen[net] = score
        score = seen[net]
        if score < best_score:
            best_net, best_score = net, score
            global_stall = 0
        else:
            global_stall += 1
        return score

    def energy(net):
        try:
            return float(tools.energy(net))
        except Exception:
            return 0.0

    # Repair each sample for free, then score a surrogate-ranked but diverse
    # subset. Diversity is measured by symmetric arc difference.
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

    ordered = sorted(repaired, key=energy)
    initial = []
    for net in ordered:
        if all(distance(net, old) >= 4 for old in initial):
            initial.append(net)
        if len(initial) >= min(30, len(ordered)):
            break
    for net in ordered:
        if len(initial) >= min(30, len(ordered)):
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

    if not seen:
        return list(initial[0]) if initial else []

    def archive():
        """Rank good networks while retaining representatives of distinct regions."""
        ranked = sorted(seen, key=seen.get)
        out = []
        for net in ranked:
            if all(distance(net, old) >= 4 for old in out):
                out.append(net)
            if len(out) >= 24:
                break
        for net in ranked:
            if len(out) >= 36:
                break
            if net not in out:
                out.append(net)
        return out

    def select_parent(pop):
        """Select among elite and less-visited archive regions."""
        if not pop:
            return best_net
        if rng.random() < 0.42:
            idx = int(rng.integers(min(6, len(pop))))
        else:
            weights = np.asarray([
                1.0 / (math.sqrt(i + 1.0) *
                       math.sqrt(1.0 + parent_visits.get(net, 0)))
                for i, net in enumerate(pop)
            ], dtype=float)
            weights /= weights.sum()
            idx = int(rng.choice(len(pop), p=weights))
        parent = pop[idx]
        parent_visits[parent] = parent_visits.get(parent, 0) + 1
        return parent

    def switch_from(arcs, current, p, q):
        """A degree-preserving 2-switch on two directed links."""
        (a, b), (c, d) = arcs[p], arcs[q]
        if a == d or c == b:
            return None
        if (a, d) in current or (c, b) in current:
            return None
        nxt = current.copy()
        nxt.remove((a, b))
        nxt.remove((c, d))
        nxt.add((a, d))
        nxt.add((c, b))
        return tuple(sorted(nxt))

    def random_switch(cur):
        arcs = list(cur)
        if len(arcs) < 2:
            return None
        p = int(rng.integers(len(arcs)))
        q = int(rng.integers(len(arcs) - 1))
        if q >= p:
            q += 1
        return switch_from(arcs, cur, p, q)

    def make_moves(net, detail):
        """Combine bottleneck switches, random exchanges, rotations, and kicks."""
        arcs = list(net)
        current = set(net)
        moves = set()

        prices = detail.get("prices", {}) if detail else {}
        loads = detail.get("loads", {}) if detail else {}
        hot = {
            tuple(map(int, arc))
            for arc, price in prices.items()
            if float(price) > 0.0
        }
        if not hot and loads:
            hottest = sorted(
                ((float(load), tuple(map(int, arc))) for arc, load in loads.items()),
                reverse=True,
            )
            hot = {arc for _, arc in hottest[:max(4, n // 2)]}

        # Explore all 2-switches touching a positive-price link, when available.
        for p in range(len(arcs)):
            for q in range(p + 1, len(arcs)):
                if hot and arcs[p] not in hot and arcs[q] not in hot:
                    continue
                cand = switch_from(arcs, current, p, q)
                if cand is not None:
                    moves.add(cand)

        # Unfocused exchanges preserve exploration when dual prices are degenerate.
        random_trials = 45 + min(35, global_stall // 35)
        for _ in range(random_trials):
            cand = random_switch(current)
            if cand is not None:
                moves.add(cand)

        # Three-link rotations allow transitions inaccessible to a single 2-switch.
        if len(arcs) >= 3:
            stalls = parent_stalls.get(net, 0)
            trials = min(150, 30 + 10 * min(12, stalls // 2 + global_stall // 100))
            for _ in range(trials):
                ids = rng.choice(len(arcs), size=3, replace=False)
                (a, b), (c, d), (e, f) = (arcs[int(i)] for i in ids)
                old = {(a, b), (c, d), (e, f)}
                if hot and not (old & hot) and rng.random() < 0.8:
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

        # Larger degree-preserving perturbations increase with local/global stalls.
        stalls = parent_stalls.get(net, 0)
        escalation = min(6, stalls // 3 + global_stall // 120)
        kick_count = 8 + 2 * escalation
        max_strength = max(2, min(max(3, n // 2), 3 + escalation))
        for _ in range(kick_count):
            cur = set(net)
            strength = min(max_strength, 2 + int(rng.integers(3)) + escalation // 2)
            for _ in range(strength):
                nxt = random_switch(cur)
                if nxt is not None:
                    cur = set(nxt)
            if cur != current:
                moves.add(tuple(sorted(cur)))

        return [m for m in moves if m != net and tools.is_valid(m)]

    def relink_moves(parent, pop):
        """Greedily follow improving 2-switch steps toward a distant archive member."""
        if len(pop) < 2:
            return []
        ranked = sorted(pop, key=lambda x: seen.get(x, float("inf")))
        pool = ranked[:min(20, len(ranked))]
        targets = [t for t in pool if t != parent and distance(t, parent) >= 8]
        if not targets:
            return []
        target = targets[int(rng.integers(len(targets)))]
        target_set = set(target)
        cur = set(parent)
        found = []
        steps = min(6, max(3, n // 3))

        for _ in range(steps):
            arcs = list(cur)
            current_overlap = len(cur & target_set)
            best_delta = 0
            best_states = []
            # Choose switches that move toward the target without losing target arcs.
            for p in range(len(arcs)):
                for q in range(p + 1, len(arcs)):
                    nxt = switch_from(arcs, cur, p, q)
                    if nxt is None:
                        continue
                    delta = len(set(nxt) & target_set) - current_overlap
                    if delta > best_delta:
                        best_delta = delta
                        best_states = [nxt]
                    elif delta == best_delta and delta > 0:
                        best_states.append(nxt)
            if not best_states:
                break
            cur = set(best_states[int(rng.integers(len(best_states)))])
            found.append(tuple(sorted(cur)))
        return found

    def screen_batch(moves, parent, detail, pop, batch_size):
        """Select a batch mixing energy, link pressure, and structural novelty."""
        fresh = [
            m for m in moves
            if m not in seen and not tools.is_cached(m)
        ]
        if not fresh:
            return []

        base_energy = energy(parent)
        energy_rank = sorted(
            ((energy(net) - base_energy, net) for net in fresh),
            key=lambda item: item[0],
        )

        loads = {
            tuple(map(int, arc)): float(value)
            for arc, value in (detail.get("loads", {}) if detail else {}).items()
        }
        prices = {
            tuple(map(int, arc)): float(value)
            for arc, value in (detail.get("prices", {}) if detail else {}).items()
        }
        max_load = max(loads.values(), default=1.0) or 1.0
        max_price = max(prices.values(), default=1.0) or 1.0
        max_demand = float(np.max(demand)) if np.size(demand) else 1.0
        if max_demand <= 0.0:
            max_demand = 1.0

        parent_set = set(parent)
        pressure_rank = []
        for net in fresh:
            net_set = set(net)
            removed = parent_set - net_set
            added = net_set - parent_set
            pressure = sum(
                0.60 * loads.get(a, 0.0) / max_load
                + 0.40 * prices.get(a, 0.0) / max_price
                for a in removed
            )
            pressure -= sum(
                0.45 * float(demand[a[0], a[1]]) / max_demand
                for a in added
            )
            pressure_rank.append((pressure, net))
        pressure_rank.sort(key=lambda item: item[0], reverse=True)

        selected = []

        def add_diverse(net, threshold=2):
            if net in selected:
                return False
            if all(distance(net, old) >= threshold for old in selected):
                selected.append(net)
                return True
            return False

        # Reserve some calls for both surrogate and bottleneck-guided proposals.
        energy_quota = min(3, batch_size)
        for _, net in energy_rank:
            add_diverse(net)
            if len(selected) >= energy_quota:
                break

        pressure_quota = min(3, max(0, batch_size - len(selected)))
        added_pressure = 0
        for _, net in pressure_rank:
            if add_diverse(net):
                added_pressure += 1
            if added_pressure >= pressure_quota:
                break

        # At least one candidate is chosen for distance from the current archive.
        if len(selected) < batch_size and fresh:
            anchors = pop[:min(10, len(pop))]
            remaining = [net for net in fresh if net not in selected]
            if remaining:
                novelty = []
                for net in remaining:
                    near = min((distance(net, old) for old in anchors), default=0)
                    novelty.append((near, net))
                novelty.sort(key=lambda item: item[0], reverse=True)
                for _, net in novelty:
                    if add_diverse(net):
                        break

        # Fill by energy, then use a random tail proposal to retain exploration.
        for _, net in energy_rank:
            if len(selected) >= batch_size:
                break
            if net not in selected:
                selected.append(net)

        if len(selected) < batch_size:
            tail = [net for net in fresh if net not in selected]
            if tail:
                rng.shuffle(tail)
                selected.extend(tail[:batch_size - len(selected)])

        return selected[:batch_size]

    idle_rounds = 0
    while tools.lp_calls_left > 0:
        pop = archive()
        if not pop:
            break
        parent = select_parent(pop)

        try:
            detail = tools.lp_detail(parent)
        except Exception:
            detail = None

        moves = make_moves(parent, detail)

        # Stalled parents occasionally get a path-relinking batch from another region.
        if global_stall > 20 and rng.random() < min(0.55, 0.15 + global_stall / 500.0):
            moves.extend(relink_moves(parent, pop))

        batch_size = 9
        batch = screen_batch(moves, parent, detail, pop, batch_size)

        # If this parent has exhausted its neighborhood, try a different archive region.
        if not batch and len(pop) > 1:
            alternatives = [net for net in pop if net != parent]
            if alternatives:
                parent = alternatives[int(rng.integers(len(alternatives)))]
                try:
                    detail = tools.lp_detail(parent)
                except Exception:
                    detail = None
                moves = make_moves(parent, detail)
                batch = screen_batch(moves, parent, detail, pop, batch_size)

        if not batch:
            idle_rounds += 1
            if idle_rounds > 100:
                break
            continue
        idle_rounds = 0

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
