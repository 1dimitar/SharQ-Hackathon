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
    """Breed valid networks by alternating-component crossover and mutation."""
    n = tools.n
    demand = np.asarray(demand, dtype=float)

    def canon(arcs):
        return tuple(sorted(tuple(map(int, a)) for a in arcs))

    def distance(a, b):
        return len(set(a) ^ set(b))

    scored = {}
    energies = {}
    proxies = {}
    detail_cache = {}
    best_net = None
    best_score = float("inf")

    def energy(net):
        if net not in energies:
            try:
                value = float(tools.energy(net))
                energies[net] = value if math.isfinite(value) else float("inf")
            except Exception:
                energies[net] = float("inf")
        return energies[net]

    def shortest_path_proxy(net):
        """Estimate link loads by splitting each OD demand over shortest paths."""
        if net in proxies:
            return proxies[net]

        adj = [[] for _ in range(n)]
        rev = [[] for _ in range(n)]
        for u, v in net:
            adj[u].append(v)
            rev[v].append(u)

        # Forward shortest distances and path counts from every source.
        d_from = []
        ways_from = []
        for s in range(n):
            dist = [-1] * n
            ways = [0] * n
            dist[s], ways[s] = 0, 1
            queue = [s]
            for u in queue:
                for v in adj[u]:
                    if dist[v] < 0:
                        dist[v] = dist[u] + 1
                        ways[v] = ways[u]
                        queue.append(v)
                    elif dist[v] == dist[u] + 1:
                        ways[v] += ways[u]
            d_from.append(dist)
            ways_from.append(ways)

        # Number of shortest paths from each node to each destination.
        d_to = []
        ways_to = []
        for t in range(n):
            dist = [-1] * n
            ways = [0] * n
            dist[t], ways[t] = 0, 1
            queue = [t]
            for v in queue:
                for u in rev[v]:
                    if dist[u] < 0:
                        dist[u] = dist[v] + 1
                        queue.append(u)
            for u in sorted(range(n), key=lambda x: dist[x] if dist[x] >= 0 else n + 1):
                if u == t or dist[u] <= 0:
                    continue
                ways[u] = sum(ways[v] for v in adj[u] if dist[v] == dist[u] - 1)
            d_to.append(dist)
            ways_to.append(ways)

        loads = {edge: 0.0 for edge in net}
        for s in range(n):
            ds = d_from[s]
            fs = ways_from[s]
            for t in range(n):
                total = fs[t]
                if t == s or ds[t] < 0 or total <= 0:
                    continue
                dt = d_to[t]
                bt = ways_to[t]
                amount = float(demand[s, t])
                if amount == 0:
                    continue
                length = ds[t]
                for u, v in net:
                    if ds[u] >= 0 and dt[v] >= 0 and ds[u] + 1 + dt[v] == length:
                        paths = fs[u] * bt[v]
                        if paths:
                            loads[(u, v)] += amount * (paths / total)

        value = max(loads.values(), default=0.0)
        proxies[net] = float(value)
        return float(value)

    def evaluate(net):
        nonlocal best_net, best_score
        net = canon(net)
        if net not in scored:
            score = tools.lp(net)
            scored[net] = float(score)
        score = scored[net]
        if score < best_score:
            best_net, best_score = net, score
        return score

    def get_detail(net):
        if net not in detail_cache:
            try:
                detail_cache[net] = tools.lp_detail(net)
            except Exception:
                detail_cache[net] = None
        return detail_cache[net]

    # The symmetric difference of two degree-2 bipartite graphs has balanced
    # degrees at every vertex. Swapping any connected component preserves all
    # row and column degrees, giving an exact degree-preserving crossover.
    def crossover(parent, donor):
        pset, dset = set(parent), set(donor)
        red, blue = pset - dset, dset - pset
        if not red or not blue:
            return parent

        incident = {}
        for edge in red | blue:
            u, v = edge
            x, y = u, n + v
            incident.setdefault(x, []).append(edge)
            incident.setdefault(y, []).append(edge)

        components = []
        visited_nodes = set()
        for start in incident:
            if start in visited_nodes:
                continue
            stack = [start]
            visited_nodes.add(start)
            component_edges = set()
            while stack:
                node = stack.pop()
                for edge in incident[node]:
                    component_edges.add(edge)
                    u, v = edge
                    for other in (u, n + v):
                        if other not in visited_nodes:
                            visited_nodes.add(other)
                            stack.append(other)
            components.append(component_edges)

        child = set(parent)
        changed = False
        for component in components:
            if rng.random() < 0.5:
                changed = True
                child.difference_update(component & red)
                child.update(component & blue)
        if not changed:
            component = components[int(rng.integers(len(components)))]
            child.difference_update(component & red)
            child.update(component & blue)

        result = canon(child)
        return result if result != parent and result != donor else parent

    def switch_once(cur, hot=()):
        arcs = list(cur)
        if len(arcs) < 2:
            return cur
        hot_arcs = [i for i, edge in enumerate(arcs) if edge in hot]
        if hot_arcs and rng.random() < 0.55:
            p = hot_arcs[int(rng.integers(len(hot_arcs)))]
            q = int(rng.integers(len(arcs) - 1))
            if q >= p:
                q += 1
        else:
            p = int(rng.integers(len(arcs)))
            q = int(rng.integers(len(arcs) - 1))
            if q >= p:
                q += 1

        a, b = arcs[p]
        c, d = arcs[q]
        if a == d or c == b or (a, d) in cur or (c, b) in cur:
            return cur
        nxt = set(cur)
        nxt.remove((a, b))
        nxt.remove((c, d))
        nxt.add((a, d))
        nxt.add((c, b))
        return frozenset(nxt)

    def rotate_once(cur, hot=()):
        """A three-link cyclic destination rotation, preserving degrees."""
        arcs = list(cur)
        if len(arcs) < 3:
            return cur
        hot_arcs = [i for i, edge in enumerate(arcs) if edge in hot]
        if hot_arcs and rng.random() < 0.55:
            ids = [hot_arcs[int(rng.integers(len(hot_arcs)))]]
            while len(ids) < 3:
                j = int(rng.integers(len(arcs)))
                if j not in ids:
                    ids.append(j)
        else:
            ids = [int(i) for i in rng.choice(len(arcs), size=3, replace=False)]

        old = [arcs[i] for i in ids]
        base = set(cur)
        base.difference_update(old)
        (a, b), (c, d), (e, f) = old
        if rng.random() < 0.5:
            new = [(a, d), (c, f), (e, b)]
        else:
            new = [(a, f), (c, b), (e, d)]
        if any(u == v for u, v in new) or len(set(new)) != 3:
            return cur
        if any(edge in base for edge in new):
            return cur
        base.update(new)
        return frozenset(base)

    def hot_links(net):
        detail = get_detail(net)
        if not detail:
            return set()
        prices = detail.get("prices", {})
        hot = {
            tuple(map(int, edge))
            for edge, price in prices.items()
            if float(price) > 0
        }
        if not hot:
            loads = detail.get("loads", {})
            ordered = sorted(
                ((float(value), tuple(map(int, edge))) for edge, value in loads.items()),
                reverse=True,
            )
            hot = {edge for _, edge in ordered[:max(3, n // 2)]}
        return hot

    def mutate(net, strength=None):
        cur = frozenset(net)
        hot = hot_links(net) if rng.random() < 0.7 else set()
        if strength is None:
            strength = 1 + int(rng.integers(4))
        for _ in range(strength):
            if rng.random() < 0.16:
                cur = rotate_once(cur, hot)
            else:
                cur = switch_once(cur, hot)
        result = canon(cur)
        if result != net and tools.is_valid(result):
            return result
        return net

    # Repair the QAOA samples, then seed the population with low-energy,
    # structurally distinct networks.
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
    start_limit = min(len(ordered), max(20, min(40, 2 * n + 4)))
    starts = []
    for net in ordered:
        if all(distance(net, old) >= 4 for old in starts):
            starts.append(net)
        if len(starts) >= start_limit:
            break
    for net in ordered:
        if len(starts) >= start_limit:
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

    if not scored:
        return list(starts[0])

    def population():
        ranked = sorted(scored, key=scored.get)
        keep = []
        separation = max(4, n // 3)
        for net in ranked:
            if all(distance(net, old) >= separation for old in keep):
                keep.append(net)
            if len(keep) >= max(36, 3 * n):
                break
        for net in ranked:
            if len(keep) >= max(36, 3 * n):
                break
            if net not in keep:
                keep.append(net)
        return keep

    def choose_parent(pop):
        if len(pop) == 1:
            return pop[0]
        if rng.random() < 0.68:
            return pop[int(rng.integers(min(len(pop), max(8, n))))]
        return pop[int(rng.integers(len(pop)))]

    def propose(pop, count):
        proposals = set()
        attempts = 0
        while len(proposals) < count and attempts < count * 12:
            attempts += 1
            parent = choose_parent(pop)
            if len(pop) > 1 and rng.random() < 0.72:
                # Prefer a reasonably good but topologically different donor.
                pool = pop[:min(len(pop), max(16, 2 * n))]
                distant = [x for x in pool if x != parent and distance(x, parent) >= 4]
                donor = distant[int(rng.integers(len(distant)))] if distant else pool[
                    int(rng.integers(len(pool)))
                ]
                child = crossover(parent, donor)
                if rng.random() < 0.8:
                    child = mutate(child, 1 + int(rng.integers(4)))
            else:
                child = mutate(parent, 1 + int(rng.integers(6)))

            if child != parent and child not in scored and tools.is_valid(child):
                proposals.add(child)

        return list(proposals)

    def select_batch(proposals, batch_size):
        if not proposals:
            return []
        # Rank the two inexpensive predictors independently, so neither scale
        # nor a weakly calibrated predictor can dominate selection.
        e_order = sorted(proposals, key=energy)
        p_order = sorted(proposals, key=shortest_path_proxy)
        e_rank = {net: i for i, net in enumerate(e_order)}
        p_rank = {net: i for i, net in enumerate(p_order)}
        denom = max(1, len(proposals) - 1)
        ranked = sorted(
            proposals,
            key=lambda net: (
                0.48 * e_rank[net] / denom
                + 0.52 * p_rank[net] / denom
                + 0.025 * float(rng.random())
            ),
        )

        explore = max(1, batch_size // 4)
        selected = []
        exploitation_limit = batch_size - explore
        for net in ranked:
            if all(distance(net, old) >= 2 for old in selected):
                selected.append(net)
            if len(selected) >= exploitation_limit:
                break
        for net in ranked:
            if len(selected) >= exploitation_limit:
                break
            if net not in selected:
                selected.append(net)

        remaining = [net for net in ranked if net not in selected]
        if remaining:
            # Keep a small explicit exploration reserve, while avoiding
            # spending calls exclusively on poor-proxy tail candidates.
            top = remaining[:max(1, (3 * len(remaining)) // 4)]
            rng.shuffle(top)
            selected.extend(top[:explore])
        return selected[:batch_size]

    batch_size = max(6, min(10, 160 // max(1, n)))
    proposal_count = max(36, 4 * batch_size)

    while tools.lp_calls_left > 0:
        pop = population()
        proposals = propose(pop, proposal_count)
        batch = select_batch(proposals, batch_size)

        if not batch:
            # Fall back to stronger kicks if crossover/mutation has run out of
            # unseen immediate offspring.
            for _ in range(80):
                parent = choose_parent(pop)
                child = mutate(parent, 3 + int(rng.integers(7)))
                if child != parent and child not in scored:
                    batch.append(child)
                    if len(batch) >= batch_size:
                        break

        if not batch:
            break

        for net in batch:
            if tools.lp_calls_left <= 0:
                break
            try:
                evaluate(net)
            except Exception as exc:
                if type(exc).__name__ == "BudgetExhausted":
                    return list(best_net) if best_net is not None else []
                # Ignore malformed or otherwise unevaluable proposals.

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
