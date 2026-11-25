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
    proxy_features = {}

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
        """Reserve archive places for LP, energy, and routing-proxy leaders."""
        ranked = sorted(
            (net for net in seen if math.isfinite(seen[net])),
            key=seen.get,
        )
        out = []

        def add_group(group, limit):
            added = 0
            for net in group:
                if net in out:
                    continue
                if all(distance(net, old) >= 4 for old in out):
                    out.append(net)
                    added += 1
                    if added >= limit:
                        break

        elite_count = max(4, n // 2)
        add_group(ranked, elite_count)
        energy_ranked = sorted(ranked, key=energy)
        add_group(energy_ranked, max(2, n // 3))
        proxy_ranked = sorted(
            (net for net in ranked if net in proxy_features),
            key=lambda net: proxy_features[net][0] + 0.25 * proxy_features[net][1],
        )
        add_group(proxy_ranked, max(2, n // 3))

        # Fill remaining capacity with LP-ranked alternatives, without requiring
        # every archive member to satisfy the stronger group-separation test.
        for net in ranked:
            if len(out) >= 4 * n:
                break
            if net not in out:
                out.append(net)
        return out or ([best_net] if best_net is not None else [])

    def select_parent(pop):
        """Sample explicitly across LP, energy, routing, and broad-search groups."""
        if not pop:
            return best_net

        elite = pop[:min(max(4, n // 2), len(pop))]
        energy_group = sorted(pop, key=energy)[:min(n, len(pop))]
        proxy_group = sorted(
            (net for net in pop if net in proxy_features),
            key=lambda net: proxy_features[net][0] + 0.25 * proxy_features[net][1],
        )[:min(n, len(pop))]
        groups = [elite, energy_group, proxy_group, pop]
        probabilities = np.asarray([0.35, 0.20, 0.30, 0.15], dtype=float)
        available = np.asarray([bool(group) for group in groups])
        probabilities *= available
        probabilities /= probabilities.sum()
        group_id = int(rng.choice(len(groups), p=probabilities))
        group = groups[group_id]
        return group[int(rng.integers(len(group)))]

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

        return [m for m in moves if m != net and tools.is_valid(m)]

    def routing_peaks(net):
        """Return tie-robust and equal-split shortest-path congestion proxies."""
        demand_array = np.asarray(demand, dtype=float)
        adjacency = [[] for _ in range(n)]
        reverse = [[] for _ in range(n)]
        for u, v in net:
            adjacency[u].append(v)
            reverse[v].append(u)

        tie_loads = np.zeros((4, n, n), dtype=float)
        split_loads = np.zeros((n, n), dtype=float)
        unserved = 0.0

        # One reverse BFS per destination supports both routing estimates.
        for target in range(n):
            dist = [n + 1] * n
            dist[target] = 0
            queue = [target]
            head = 0
            while head < len(queue):
                v = queue[head]
                head += 1
                for u in reverse[v]:
                    if dist[u] == n + 1:
                        dist[u] = dist[v] + 1
                        queue.append(u)

            choices_by_node = [
                [v for v in adjacency[u] if dist[v] == dist[u] - 1]
                if dist[u] <= n else []
                for u in range(n)
            ]

            # Fractionally split each destination's traffic among shortest
            # outgoing choices, propagating flow from distant nodes to target.
            flow = demand_array[:, target].copy()
            for u in sorted(range(n), key=lambda x: dist[x], reverse=True):
                if u == target or dist[u] > n or flow[u] == 0.0:
                    continue
                choices = choices_by_node[u]
                if not choices:
                    continue
                share = flow[u] / len(choices)
                for v in choices:
                    split_loads[u, v] += share
                    flow[v] += share

            # Also route all traffic on several deterministic shortest-path
            # tie-breakings to measure sensitivity to path choices.
            for source in range(n):
                if source == target:
                    continue
                amount = float(demand_array[source, target])
                if dist[source] > n:
                    unserved += amount
                    continue
                for mode in range(4):
                    u = source
                    while u != target:
                        choices = choices_by_node[u]
                        if not choices:
                            break
                        offset = (
                            mode * (source + 1) + (mode + 1) * target
                        ) % n
                        v = min(choices, key=lambda x: (x - offset) % n)
                        tie_loads[mode, u, v] += amount
                        u = v

        tie_peaks = [
            max((tie_loads[mode, u, v] for u, v in net), default=0.0)
            + unserved
            for mode in range(4)
        ]
        split_peak = max(
            (split_loads[u, v] for u, v in net), default=0.0
        ) + unserved
        return float(np.percentile(tie_peaks, 75.0)), float(split_peak)

    def screen_batch(moves, parent, batch_size=9):
        """Mix energy and tie-robust routing ranks, with a reserved exploration slot."""
        fresh = [
            m for m in moves
            if m not in seen and not tools.is_cached(m)
        ]
        if not fresh:
            return []

        ranked = sorted(fresh, key=energy)
        energy_rank = {net: i for i, net in enumerate(ranked)}

        # Evaluate routing features only on an energy shortlist plus a random sample.
        shortlist_size = min(3 * batch_size, len(ranked))
        proxy_pool = list(ranked[:shortlist_size])
        rest = ranked[shortlist_size:]
        if rest:
            sample_size = min(n, len(rest))
            ids = rng.choice(len(rest), size=sample_size, replace=False)
            proxy_pool.extend(rest[int(i)] for i in ids)

        routing_scores = {net: routing_peaks(net) for net in proxy_pool}
        for net, scores in routing_scores.items():
            proxy_features[net] = scores
        proxy_order = sorted(
            proxy_pool, key=lambda net: routing_scores[net][0]
        )
        proxy_rank = {net: i for i, net in enumerate(proxy_order)}
        split_order = sorted(
            proxy_pool, key=lambda net: routing_scores[net][1]
        )
        split_rank = {net: i for i, net in enumerate(split_order)}
        combined_order = sorted(
            proxy_pool,
            key=lambda net: (
                energy_rank[net] / max(1, len(ranked))
                + proxy_rank[net] / max(1, len(proxy_order))
                + split_rank[net] / max(1, len(split_order))
            ),
        )

        selected = []

        def take(order, number):
            added = 0
            for net in order:
                if added >= number:
                    break
                if net in selected:
                    continue
                if all(distance(net, old) >= 2 for old in selected):
                    selected.append(net)
                    added += 1

        # Keep energy prominent while giving both routing estimates a voice.
        take(ranked, max(2, batch_size // 3))
        take(proxy_order, max(1, batch_size // 4))
        take(split_order, 1)
        take(combined_order, max(0, batch_size - 2 - len(selected)))

        # Preserve genuine exploration rather than filling every slot by rank.
        remaining = [net for net in fresh if net not in selected]
        if remaining and len(selected) < batch_size:
            selected.append(remaining[int(rng.integers(len(remaining)))])

        # Relax diversity only if the ranked and exploratory choices did not fill the batch.
        for net in ranked:
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
        batch = screen_batch(moves, parent)

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
