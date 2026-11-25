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
    """Search repaired candidates using an adaptive operator portfolio."""
    n = tools.n
    scored = {}
    energy_cache = {}
    best_net = None
    best_score = float("inf")
    visits = {}
    stalls = {}

    # Smoothed operator statistics: focused switches, random switches,
    # rotations, multi-switch kicks, and consensus-guided switches.
    wins = [1.0] * 5
    trials = [3.0] * 5
    operator_names = range(5)

    def canon(arcs):
        return tuple(sorted((int(a), int(b)) for a, b in arcs))

    def edge_set(net):
        return set(net)

    def distance(a, b):
        return len(edge_set(a) ^ edge_set(b))

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
        if net not in scored:
            scored[net] = tools.lp(net)
        value = scored[net]
        if value < best_score:
            best_net, best_score = net, value
        return value

    def ranked_scored():
        return sorted(scored, key=scored.get)

    # Repair and deduplicate the pool before spending any LP budget.
    repaired = set()
    for raw in candidates:
        try:
            net = canon(tools.repair(raw))
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

    energy_ranked = sorted(repaired, key=energy)
    starts = []
    for net in energy_ranked:
        if all(distance(net, old) >= 4 for old in starts):
            starts.append(net)
        if len(starts) >= min(18, len(energy_ranked)):
            break
    for net in energy_ranked:
        if len(starts) >= min(26, len(energy_ranked)):
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
            continue

    if not scored:
        return list(starts[0]) if starts else []

    def archive():
        """Return strong solutions plus representatives from distinct regions."""
        ranked = ranked_scored()
        out = ranked[:8]
        for net in ranked[8:]:
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

    def choose_parent(pop):
        # Weighted sampling over a bounded archive balances quality and exploration.
        pool = pop[:min(24, len(pop))]
        weights = []
        for rank, net in enumerate(pool):
            quality = 1.0 / (1.0 + 0.18 * rank)
            explore = 1.0 + 1.3 / math.sqrt(1.0 + visits.get(net, 0))
            stale = 1.0 + min(1.0, 0.06 * stalls.get(net, 0))
            weights.append(quality * explore * stale)
        total = sum(weights)
        pick = float(rng.random()) * total
        for net, weight in zip(pool, weights):
            pick -= weight
            if pick <= 0:
                visits[net] = visits.get(net, 0) + 1
                return net
        net = pool[-1]
        visits[net] = visits.get(net, 0) + 1
        return net

    def switch(net, p, q):
        arcs = net
        a, b = arcs[p]
        c, d = arcs[q]
        current = set(net)
        if a == d or c == b or (a, d) in current or (c, b) in current:
            return None
        current.remove((a, b))
        current.remove((c, d))
        current.add((a, d))
        current.add((c, b))
        return tuple(sorted(current))

    def extract_hot(detail):
        hot = set()
        prices = {}
        loads = {}
        if detail:
            try:
                prices = {
                    tuple(map(int, edge)): float(value)
                    for edge, value in detail.get("prices", {}).items()
                }
                hot = {edge for edge, price in prices.items() if price > 0}
                loads = {
                    tuple(map(int, edge)): float(value)
                    for edge, value in detail.get("loads", {}).items()
                }
            except Exception:
                prices, loads, hot = {}, {}, set()
        if not hot and loads:
            hot = set(sorted(loads, key=loads.get, reverse=True)[:6])
        return hot, prices

    def consensus_edges():
        representatives = []
        for net in ranked_scored():
            if all(distance(net, old) >= 4 for old in representatives):
                representatives.append(net)
            if len(representatives) >= 12:
                break
        if not representatives:
            return set()
        counts = {}
        for net in representatives:
            for edge in net:
                counts[edge] = counts.get(edge, 0) + 1
        threshold = max(2, (len(representatives) + 3) // 4)
        return {edge for edge, count in counts.items() if count >= threshold}

    def propose(parent, detail):
        """Produce separate, deduplicated proposal lists for each operator."""
        arcs = list(parent)
        current = set(parent)
        groups = [set() for _ in operator_names]
        hot, _ = extract_hot(detail)

        # Operator 0: exhaustive two-switches touching an active bottleneck.
        for p in range(len(arcs)):
            for q in range(p + 1, len(arcs)):
                if hot and arcs[p] not in hot and arcs[q] not in hot:
                    continue
                move = switch(parent, p, q)
                if move is not None and move != parent:
                    groups[0].add(move)

        # Operator 1: non-focused switches retain exploration when dual prices
        # are sparse, degenerate, or misleading.
        for _ in range(70):
            if len(arcs) < 2:
                break
            p = int(rng.integers(len(arcs)))
            q = int(rng.integers(len(arcs) - 1))
            if q >= p:
                q += 1
            move = switch(parent, p, q)
            if move is not None and move != parent:
                groups[1].add(move)

        # Operator 2: degree-preserving three-link rotations.
        rotation_trials = min(220, 65 + 30 * (stalls.get(parent, 0) // 3))
        for _ in range(rotation_trials):
            if len(arcs) < 3:
                break
            ids = rng.choice(len(arcs), size=3, replace=False)
            (a, b), (c, d), (e, f) = (arcs[int(i)] for i in ids)
            old = {(a, b), (c, d), (e, f)}
            if hot and not (old & hot):
                continue
            for new in (
                {(a, d), (c, f), (e, b)},
                {(a, f), (c, b), (e, d)},
            ):
                if any(x == y for x, y in new) or new & (current - old):
                    continue
                move = tuple(sorted((current - old) | new))
                if move != parent:
                    groups[2].add(move)

        # Operator 3: short random walks of valid two-switches.
        for _ in range(14):
            cur = set(parent)
            for _ in range(2 + int(rng.integers(4))):
                now = list(cur)
                p, q = rng.choice(len(now), size=2, replace=False)
                a, b = now[int(p)]
                c, d = now[int(q)]
                if a == d or c == b or (a, d) in cur or (c, b) in cur:
                    continue
                cur.remove((a, b))
                cur.remove((c, d))
                cur.add((a, d))
                cur.add((c, b))
            move = tuple(sorted(cur))
            if move != parent:
                groups[3].add(move)

        # Operator 4: one-switch proposals that introduce consensus edges
        # seen in good, structurally distinct archive members.
        guide = consensus_edges()
        if guide:
            targets = list(guide - current)
            if targets:
                rng.shuffle(targets)
                for a, b in targets[:10]:
                    outgoing = [edge for edge in parent if edge[0] == a]
                    incoming = [edge for edge in parent if edge[1] == b]
                    for _, x in outgoing:
                        for y, _ in incoming:
                            old = {(a, x), (y, b)}
                            if y == a or x == b or (y, x) in current - old:
                                continue
                            move = tuple(sorted((current - old) | {(a, b), (y, x)}))
                            if move != parent:
                                groups[4].add(move)

        # Filter once here, rather than repeatedly during selection/evaluation.
        for i in operator_names:
            groups[i] = {
                move for move in groups[i]
                if move not in scored and not tools.is_cached(move) and tools.is_valid(move)
            }
        return groups

    def operator_probabilities(groups):
        rates = [
            (wins[i] + 1.0) / (trials[i] + 2.0)
            for i in operator_names
        ]
        # A soft floor prevents early noisy results from disabling an operator.
        raw = [0.35 + rate for rate in rates]
        for i, group in enumerate(groups):
            if not group:
                raw[i] = 0.0
        total = sum(raw)
        if total <= 0:
            return [0.0] * len(raw)
        return [value / total for value in raw]

    def select_batch(groups, parent, detail, batch_size=4):
        """Allocate a small LP batch by operator success, then rank within groups."""
        hot, prices = extract_hot(detail)
        parent_edges = set(parent)
        base_energy = energy(parent)
        records = [[] for _ in operator_names]

        for op in operator_names:
            for move in groups[op]:
                added = set(move) - parent_edges
                removed = parent_edges - set(move)
                # Energy is the main inexpensive predictor. Dual-weighted change
                # provides a tie-breaker among moves with similar predicted energy.
                dual_gain = sum(prices.get(edge, 0.0) for edge in removed)
                dual_gain -= sum(prices.get(edge, 0.0) for edge in added)
                changed_hot = len((removed | added) & hot)
                key = (energy(move) - base_energy, -changed_hot, -dual_gain)
                records[op].append((key, move))
            records[op].sort(key=lambda item: item[0])

        probabilities = operator_probabilities(groups)
        selected = []
        selected_ops = []
        # Weighted operator choice, with duplicate proposals skipped. At most
        # one initial slot per operator where possible, then adaptive allocation.
        available = [i for i in operator_names if records[i]]
        rng.shuffle(available)
        for op in available[:batch_size]:
            move = next((m for _, m in records[op]
                         if all(distance(m, old) >= 2 for old in selected)), None)
            if move is not None:
                selected.append(move)
                selected_ops.append(op)

        while len(selected) < batch_size:
            probs = operator_probabilities(groups)
            available = [i for i in operator_names
                         if records[i] and any(m != x for _, m in records[i]
                                               for x in selected)]
            if not available:
                break
            weights = [probs[i] for i in available]
            total = sum(weights)
            if total <= 0:
                op = available[int(rng.integers(len(available)))]
            else:
                pick = float(rng.random()) * total
                op = available[-1]
                for i, weight in zip(available, weights):
                    pick -= weight
                    if pick <= 0:
                        op = i
                        break
            options = [m for _, m in records[op] if m not in selected]
            if not options:
                continue
            # Mostly take the best predicted move, occasionally sample from a
            # short ranked tail to maintain diversity.
            window = options[:min(8, len(options))]
            if len(window) > 1 and rng.random() < 0.22:
                move = window[int(rng.integers(len(window)))]
            else:
                move = window[0]
            selected.append(move)
            selected_ops.append(op)

        return selected, selected_ops

    while tools.lp_calls_left > 0:
        pop = archive()
        parent = choose_parent(pop)
        if parent is None:
            break

        try:
            detail = tools.lp_detail(parent)
        except Exception:
            detail = None

        groups = propose(parent, detail)
        batch, batch_ops = select_batch(groups, parent, detail)

        if not batch:
            # Rotate to another archive member if the current neighborhood is
            # exhausted; do not repeatedly regenerate an empty neighborhood.
            alternatives = [net for net in pop if net != parent]
            if alternatives:
                parent = alternatives[int(rng.integers(len(alternatives)))]
                try:
                    detail = tools.lp_detail(parent)
                except Exception:
                    detail = None
                groups = propose(parent, detail)
                batch, batch_ops = select_batch(groups, parent, detail)

        if not batch:
            # A final kick is a cheap fallback for a locally exhausted archive.
            for _ in range(35):
                cur = set(parent)
                for _ in range(3 + int(rng.integers(4))):
                    now = list(cur)
                    p, q = rng.choice(len(now), size=2, replace=False)
                    a, b = now[int(p)]
                    c, d = now[int(q)]
                    if a == d or c == b or (a, d) in cur or (c, b) in cur:
                        continue
                    cur.remove((a, b))
                    cur.remove((c, d))
                    cur.add((a, d))
                    cur.add((c, b))
                move = tuple(sorted(cur))
                if move != parent and move not in scored and not tools.is_cached(move):
                    if tools.is_valid(move):
                        batch, batch_ops = [move], [3]
                        break

        if not batch:
            break

        parent_score = scored.get(parent, float("inf"))
        batch_best = float("inf")
        try:
            for move, op in zip(batch, batch_ops):
                if tools.lp_calls_left <= 0:
                    break
                try:
                    value = evaluate(move)
                except Exception as exc:
                    if type(exc).__name__ == "BudgetExhausted":
                        return list(best_net) if best_net is not None else []
                    continue
                batch_best = min(batch_best, value)
                trials[op] += 1.0
                if value < parent_score:
                    wins[op] += 1.0
        except Exception as exc:
            if type(exc).__name__ == "BudgetExhausted":
                return list(best_net) if best_net is not None else []

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
