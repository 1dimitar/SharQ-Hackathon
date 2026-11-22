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
        """Mix elite exploitation with sampling from diverse archived networks."""
        if not pop:
            return best_net
        if rng.random() < 0.48:
            return pop[int(rng.integers(min(6, len(pop))))]
        return pop[int(rng.integers(len(pop)))]

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

        # Generate rotations that deliberately remove high-pressure links.
        # Retain random rotations as a complementary source of nonlocal moves.
        if len(arcs) >= 3:
            stalls = parent_stalls.get(net, 0)

            def add_rotation(ids):
                (a, b), (c, d), (e, f) = (arcs[int(i)] for i in ids)
                old = {(a, b), (c, d), (e, f)}
                for new in (
                    {(a, d), (c, f), (e, b)},
                    {(a, f), (c, b), (e, d)},
                ):
                    if len(new) != 3 or any(x == y for x, y in new):
                        continue
                    if new & (current - old):
                        continue
                    moves.add(tuple(sorted((current - old) | new)))

            if hot:
                loads = detail.get("loads", {}) if detail else {}
                prices = detail.get("prices", {}) if detail else {}
                hot_order = sorted(
                    hot,
                    key=lambda a: (
                        float(prices.get(a, 0.0)),
                        float(loads.get(a, 0.0)),
                    ),
                    reverse=True,
                )
                arc_index = {a: i for i, a in enumerate(arcs)}
                target_trials = min(180, max(24, 6 * len(hot_order)))
                for t in range(target_trials):
                    critical = hot_order[t % len(hot_order)]
                    p = arc_index[critical]
                    others = [i for i in range(len(arcs)) if i != p]
                    chosen = rng.choice(len(others), size=2, replace=False)
                    add_rotation([p, others[int(chosen[0])], others[int(chosen[1])]])

            rotation_trials = min(350, 70 * (1 + stalls // 3))
            for _ in range(rotation_trials):
                ids = rng.choice(len(arcs), size=3, replace=False)
                if hot and not any(arcs[int(i)] in hot for i in ids):
                    continue
                add_rotation(ids)

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

        # Relink the parent toward the incumbent through improving-overlap switches.
        target = best_net
        if target is not None and target != net:
            target_set = set(target)
            cur = set(net)
            for _ in range(2 * n):
                out_by_source = {}
                in_by_dest = {}
                for a, b in cur:
                    out_by_source.setdefault(a, []).append(b)
                    in_by_dest.setdefault(b, []).append(a)

                best_step = None
                best_gain = 0
                for a, b in target_set - cur:
                    for d in out_by_source.get(a, ()):
                        for c in in_by_dest.get(b, ()):
                            replacement = (c, d)
                            if c == d or replacement in cur:
                                continue
                            removed = ((a, d), (c, b))
                            gain = (
                                1 + int(replacement in target_set)
                                - int(removed[0] in target_set)
                                - int(removed[1] in target_set)
                            )
                            if gain > best_gain:
                                best_gain = gain
                                best_step = (removed, ((a, b), replacement))

                if best_step is None:
                    break
                removed, added = best_step
                cur.difference_update(removed)
                cur.update(added)
                moves.add(tuple(sorted(cur)))

        return [m for m in moves if m != net and tools.is_valid(m)]

    def screen_batch(moves, parent, detail=None, batch_size=9):
        """Mix Hamiltonian energy with cheap shortest-path routing estimates."""
        fresh = [
            m for m in moves
            if m not in seen and not tools.is_cached(m)
        ]
        if not fresh:
            return []

        base_e = energy(parent)
        ranked = [(energy(net) - base_e, net) for net in fresh]
        ranked.sort(key=lambda item: item[0])

        selected = []
        energy_quota = min(5, batch_size)
        for _, net in ranked:
            if all(distance(net, old) >= 2 for old in selected):
                selected.append(net)
            if len(selected) >= energy_quota:
                break

        # Minimum traffic-hop count / number of links is a congestion lower
        # bound. Compute it for every move, then route only a small shortlist.
        if len(selected) < batch_size:
            count = len(fresh)
            edge_array = np.asarray(fresh, dtype=int)
            rows = np.arange(count)[:, None]
            dist = np.full((count, n, n), n + 1, dtype=np.int16)
            diagonal = np.arange(n)
            dist[:, diagonal, diagonal] = 0
            dist[rows, edge_array[:, :, 0], edge_array[:, :, 1]] = 1
            for k in range(n):
                np.minimum(
                    dist,
                    dist[:, :, k, None] + dist[:, None, k, :],
                    out=dist,
                )

            traffic = np.maximum(np.asarray(demand, dtype=float), 0.0).copy()
            np.fill_diagonal(traffic, 0.0)
            hop_bound = np.sum(dist * traffic[None, :, :], axis=(1, 2)) / (2 * n)
            unreachable = np.any(
                (dist > n) & (traffic[None, :, :] > 0), axis=(1, 2)
            )
            hop_bound[unreachable] = np.inf

            index = {net: i for i, net in enumerate(fresh)}
            shortlist = set(int(i) for i in np.argsort(hop_bound)[:2 * n])
            shortlist.update(index[net] for _, net in ranked[:2 * n])
            shortlist.update(
                int(i) for i in rng.choice(count, size=min(n, count), replace=False)
            )
            ids = np.asarray(sorted(shortlist), dtype=int)
            sd = dist[ids]
            # Canonical edge order groups the two outgoing arcs of each node.
            neighbors = edge_array[ids, :, 1].reshape(len(ids), n, 2)
            batch_rows = np.arange(len(ids))[:, None, None]
            next_dist = sd[batch_rows, neighbors, :]
            eligible = next_dist + 1 == sd[:, :, None, :]
            fractions = eligible / np.maximum(
                1, np.sum(eligible, axis=2, keepdims=True)
            )

            # Propagate destination-specific traffic in descending distance
            # layers. Each unit splits equally over shortest-path next hops.
            mass = np.broadcast_to(traffic, (len(ids), n, n)).copy()
            loads = np.zeros((len(ids), n, 2), dtype=float)
            destination = np.arange(n)[None, None, None, :]
            scatter_rows = np.arange(len(ids))[:, None, None, None]
            for level in range(min(n - 1, int(sd.max())), 0, -1):
                departing = np.where(sd == level, mass, 0.0)
                mass -= departing
                sent = departing[:, :, None, :] * fractions
                loads += np.sum(sent, axis=3)
                np.add.at(
                    mass,
                    (scatter_rows, neighbors[:, :, :, None], destination),
                    sent,
                )

            # ECMP is not LP-optimal; use it for just two batch slots.
            # A small hop term breaks ties in favor of routing efficiency.
            routing_proxy = np.max(loads, axis=(1, 2)) + 0.10 * hop_bound[ids]
            routing_proxy[unreachable[ids]] = np.inf
            routing_quota = min(2, batch_size - len(selected))
            added_routing = 0
            for pos in np.argsort(routing_proxy):
                net = fresh[int(ids[int(pos)])]
                if net not in selected:
                    selected.append(net)
                    added_routing += 1
                if added_routing >= routing_quota:
                    break

        # Fill by surrogate rank, reserving one slot for exploration if possible.
        tail = [net for _, net in ranked if net not in selected]
        fill_limit = batch_size - 1 if tail else batch_size
        for _, net in ranked:
            if len(selected) >= fill_limit:
                break
            if net not in selected:
                selected.append(net)

        # A random fresh move complements the energy- and routing-guided picks.
        tail = [net for _, net in ranked if net not in selected]
        if tail and len(selected) < batch_size:
            selected.append(tail[int(rng.integers(len(tail)))])
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
        batch = screen_batch(moves, parent, detail=detail)

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
                batch = screen_batch(make_moves(parent, detail), parent, detail=detail)

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
