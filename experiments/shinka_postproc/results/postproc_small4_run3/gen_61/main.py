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
    """Search valid networks using diverse starts, free routing proxies, and LP feedback."""
    n = int(tools.n)
    m = 2 * n
    started = time.perf_counter()

    D = np.asarray(demand, dtype=float).copy()
    np.fill_diagonal(D, 0.0)
    traffic = max(float(D.sum()), 1.0)
    scale = traffic / max(m, 1)

    seen = {}
    details = {}
    energy_cache = {}
    stalls = {}
    visits = {}
    training_x = []
    training_y = []
    best_net = None
    best_score = float("inf")
    continuation = None

    def canon(arcs):
        return tuple(sorted((int(a), int(b)) for a, b in arcs))

    def energy(net):
        if net not in energy_cache:
            try:
                value = float(tools.energy(net))
                energy_cache[net] = value if math.isfinite(value) else 0.0
            except Exception:
                energy_cache[net] = 0.0
        return energy_cache[net]

    def routing_features(nets):
        """Compute shortest-path and equal-shortest-path-splitting proxy features."""
        batch = len(nets)
        if not batch:
            return np.empty((0, 9), dtype=float)

        arcs = np.asarray(nets, dtype=np.int64)
        heads = arcs[:, :, 1].reshape(batch, n, 2)
        rows = np.arange(n)
        bi = np.arange(batch)[:, None]

        dist = np.full((batch, n, n), n + 1, dtype=np.int16)
        dist[:, rows, rows] = 0
        dist[bi, arcs[:, :, 0], arcs[:, :, 1]] = 1
        for k in range(n):
            np.minimum(
                dist,
                dist[:, :, k, None] + dist[:, None, k, :],
                out=dist,
            )

        unreachable = (dist > n).sum(axis=(1, 2)).astype(float)
        d0 = dist[bi, heads[:, :, 0], :]
        d1 = dist[bi, heads[:, :, 1], :]
        ok0 = (dist == d0 + 1) & (dist <= n)
        ok1 = (dist == d1 + 1) & (dist <= n)
        denominator = np.maximum(ok0.astype(float) + ok1, 1.0)
        fraction0 = ok0 / denominator
        fraction1 = ok1 / denominator

        # Gather incoming flow to propagate traffic toward the source.
        incoming = np.argsort(heads.reshape(batch, m), axis=1, kind="stable")
        flow = np.broadcast_to(D, (batch, n, n)).copy()
        loads = np.zeros((batch, n, 2), dtype=float)
        diameter = min(n - 1, int(np.max(np.minimum(dist, n - 1))))
        for level in range(diameter, 0, -1):
            active = flow * (dist == level)
            f0 = active * fraction0
            f1 = active * fraction1
            loads[:, :, 0] += f0.sum(axis=2)
            loads[:, :, 1] += f1.sum(axis=2)
            edgeflow = np.stack((f0, f1), axis=2).reshape(batch, m, n)
            incoming_flow = np.take_along_axis(
                edgeflow, incoming[:, :, None], axis=1
            ).reshape(batch, n, 2, n).sum(axis=2)
            flow += incoming_flow

        linkloads = loads.reshape(batch, m) / scale
        sorted_loads = np.sort(linkloads, axis=1)
        work_by_source = (dist * D[None, :, :]).sum(axis=2) / scale
        work_by_target = (dist * D[None, :, :]).sum(axis=1) / scale
        direct = D[arcs[:, :, 0], arcs[:, :, 1]].sum(axis=1) / traffic

        return np.column_stack((
            np.asarray([energy(net) for net in nets]),
            (dist * D[None, :, :]).sum(axis=(1, 2)) / traffic,
            sorted_loads[:, -1],
            sorted_loads[:, -min(4, m):].mean(axis=1),
            np.sqrt(np.mean(linkloads * linkloads, axis=1)),
            work_by_source.max(axis=1),
            work_by_target.max(axis=1),
            direct,
            unreachable,
        ))

    def evaluate(net, feat=None):
        nonlocal best_net, best_score
        if net not in seen:
            detail = tools.lp_detail(net)
            score = float(detail["score"])
            seen[net] = score
            details[net] = detail
            if feat is None:
                feat = routing_features([net])[0]
            feat = np.asarray(feat, dtype=float)
            if math.isfinite(score):
                training_x.append(feat)
                training_y.append(score / scale)
        score = seen[net]
        if score < best_score:
            best_net, best_score = net, score
        return score

    def predictions(X):
        """Fit a small ridge model and return mean and uncertainty estimates."""
        count = len(training_y)
        if count < max(3 * n, 36):
            return None, None

        A = np.asarray(training_x, dtype=float)
        y = np.asarray(training_y, dtype=float)
        mean = A.mean(axis=0)
        sd = np.maximum(A.std(axis=0), 1e-8)
        Z = np.clip((A - mean) / sd, -12.0, 12.0)
        target_mean = float(y.mean())
        identity = np.eye(A.shape[1])
        gram = Z.T @ Z + 8.0 * identity
        try:
            solution = np.linalg.solve(
                gram,
                np.column_stack((Z.T @ (y - target_mean), identity)),
            )
        except np.linalg.LinAlgError:
            return None, None

        coef = solution[:, 0]
        inverse = solution[:, 1:]
        W = np.clip((X - mean) / sd, -12.0, 12.0)
        predicted = target_mean + W @ coef
        residual = y - target_mean - Z @ coef
        variance = float(residual @ residual) / max(
            1, count - A.shape[1] - 1
        )
        leverage = np.einsum("ij,jk,ik->i", W, inverse, W)
        leverage = np.clip(leverage + 1.0 / count, 0.0, 1.0)
        uncertainty = np.sqrt(max(0.0, variance) * leverage)
        return predicted, uncertainty

    # Repair the full candidate pool. Repair is free, so retain unique valid
    # networks for ranking before spending any LP calls.
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
            if not tools.is_valid(net):
                return []
            repaired.add(net)
        except Exception:
            return []

    initial_pool = sorted(repaired)
    initial_features = routing_features(initial_pool)
    energy_order = np.argsort(initial_features[:, 0], kind="stable")
    proxy = (
        initial_features[:, 1]
        + 0.10 * initial_features[:, 3]
        + initial_features[:, 8]
    )
    proxy_order = np.argsort(proxy, kind="stable")

    # Interleave rankings, first preferring structurally distinct starts and
    # then filling any remaining places with the best available candidates.
    sequence = []
    sequence_seen = set()
    for k in range(max(len(initial_pool), 1)):
        for order in (energy_order, proxy_order):
            if k < len(order):
                idx = int(order[k])
                if idx not in sequence_seen:
                    sequence_seen.add(idx)
                    sequence.append(idx)

    start_limit = min(len(initial_pool), 2 * n + 4)
    initial_order = []
    start_sets = []
    for idx in sequence:
        edge_set = set(initial_pool[idx])
        if all(m - len(edge_set & old) >= 3 for old in start_sets):
            initial_order.append(idx)
            start_sets.append(edge_set)
            if len(initial_order) >= start_limit:
                break
    if len(initial_order) < start_limit:
        for idx in sequence:
            if idx not in initial_order:
                initial_order.append(idx)
                if len(initial_order) >= start_limit:
                    break
    if not initial_order:
        return []

    for idx in initial_order:
        if tools.lp_calls_left <= 0:
            break
        try:
            evaluate(initial_pool[idx], initial_features[idx])
        except Exception as exc:
            if type(exc).__name__ == "BudgetExhausted":
                break

    if best_net is None:
        return list(initial_pool[initial_order[0]])

    def edge_distance(a, b):
        return m - len(set(a).intersection(b))

    def archive():
        ranked = sorted(
            (net for net in seen if math.isfinite(seen[net])),
            key=seen.get,
        )
        if not ranked:
            return [best_net]

        out = ranked[:max(4, n // 2)]
        edge_sets = [set(net) for net in out]
        min_separation = max(3, n // 3)
        cap = 3 * n
        for net in ranked:
            if len(out) >= cap:
                break
            if net in out or seen[net] > best_score * 1.09:
                continue
            ns = set(net)
            if all(m - len(ns & old) >= min_separation for old in edge_sets):
                out.append(net)
                edge_sets.append(ns)
        return out

    def switch(net, p, q):
        """Directed 2-switch preserving every in- and out-degree."""
        if p == q:
            return None
        (a, b), (c, d) = net[p], net[q]
        if a == c or b == d or a == d or c == b:
            return None
        current = set(net)
        if (a, d) in current or (c, b) in current:
            return None
        current.remove((a, b))
        current.remove((c, d))
        current.add((a, d))
        current.add((c, b))
        return tuple(sorted(current))

    def add_move(moves, net, exposure=0):
        if net is None:
            return
        old = moves.get(net, -1)
        if exposure > old:
            moves[net] = exposure

    def generate(parent, pop):
        current = set(parent)
        detail = details.get(parent) or {}
        prices = detail.get("prices", {})
        hot = {
            tuple(map(int, link))
            for link, price in prices.items()
            if float(price) > 1e-10
        }
        if not hot:
            loads = detail.get("loads", {})
            if loads:
                hot = set(sorted(loads, key=loads.get, reverse=True)[:max(2, n // 4)])

        moves = {}

        # Exhaustive one-step switches around the parent; bottleneck-touching
        # moves are retained as a distinct selection signal.
        for p in range(m):
            for q in range(p + 1, m):
                child = switch(parent, p, q)
                if child is not None:
                    exposure = int(parent[p] in hot) + int(parent[q] in hot)
                    add_move(moves, child, exposure)

        # Three-link rotations add transitions that cannot be reached in one
        # 2-switch. Random rotations are complemented with hot-edge-anchored ones.
        rotation_trials = 4 * n + min(stalls.get(parent, 0), 8) * n
        for _ in range(rotation_trials):
            ids = rng.choice(m, size=3, replace=False)
            old = [parent[int(i)] for i in ids]
            tails = [a for a, _ in old]
            heads = [b for _, b in old]
            if len(set(tails)) < 3 or len(set(heads)) < 3:
                continue
            rest = current.difference(old)
            shift = 1 + int(rng.integers(2))
            new = {(tails[k], heads[(k + shift) % 3]) for k in range(3)}
            if any(a == b for a, b in new) or new & rest:
                continue
            exposure = sum(edge in hot for edge in old)
            add_move(moves, tuple(sorted(rest | new)), exposure)

        hot_ids = [i for i, edge in enumerate(parent) if edge in hot]
        for _ in range(max(2 * n, 16) if hot_ids else 0):
            p = hot_ids[int(rng.integers(len(hot_ids)))]
            q, r = (int(i) for i in rng.choice(
                [i for i in range(m) if i != p], size=2, replace=False
            ))
            old = [parent[p], parent[q], parent[r]]
            tails = [a for a, _ in old]
            heads = [b for _, b in old]
            if len(set(tails)) < 3 or len(set(heads)) < 3:
                continue
            rest = current.difference(old)
            for shift in (1, 2):
                new = {(tails[k], heads[(k + shift) % 3]) for k in range(3)}
                if any(a == b for a, b in new) or new & rest:
                    continue
                add_move(moves, tuple(sorted(rest | new)), 1)

        # Pick a target archive network. When a parent has stalled, favor a
        # distant target; also try direct 2-switches that install target edges.
        target = pop[int(rng.integers(len(pop)))]
        stalls_here = min(stalls.get(parent, 0), 5)
        if stalls_here >= 2 and len(pop) > 1:
            alternatives = [net for net in pop if net != parent]
            if alternatives:
                distances = [edge_distance(parent, net) for net in alternatives]
                farthest = max(distances)
                choices = [
                    net for net, distance in zip(alternatives, distances)
                    if distance == farthest
                ]
                target = choices[int(rng.integers(len(choices)))]

        target_edges = set(target) - current
        if target_edges:
            by_tail = {}
            by_head = {}
            for i, (a, b) in enumerate(parent):
                by_tail.setdefault(a, []).append(i)
                by_head.setdefault(b, []).append(i)
            for a, d in target_edges:
                for p in by_tail.get(a, ()):
                    for q in by_head.get(d, ()):
                        child = switch(parent, p, q)
                        if child is not None and (a, d) in child:
                            add_move(moves, child, 1)

        # Short random walks are a complementary larger-kick operator. Half
        # move toward the target, while the rest retain random diversification.
        target_set = set(target)
        walk_trials = 2 * n + stalls_here * max(1, n // 3)
        for trial in range(walk_trials):
            state = parent
            steps = 2 + int(rng.integers(max(2, n // 3))) + stalls_here // 2
            for _ in range(steps):
                options = []
                for _ in range(7):
                    p, q = (int(i) for i in rng.choice(m, size=2, replace=False))
                    child = switch(state, p, q)
                    if child is not None:
                        options.append(child)
                if not options:
                    continue
                if trial < walk_trials // 2:
                    state = max(options, key=lambda x: len(target_set.intersection(x)))
                else:
                    state = options[int(rng.integers(len(options)))]
            if state != parent:
                add_move(moves, state, 0)

        return {net: exposure for net, exposure in moves.items() if net not in seen}

    def choose_parent(pop):
        if len(pop) == 1:
            return pop[0]
        u = rng.random()
        if u < 0.45:
            choices = pop[:min(4, len(pop))]
        elif u < 0.80:
            choices = pop[:min(n, len(pop))]
        else:
            choices = pop
        weights = np.asarray(
            [1.0 / math.sqrt(1.0 + visits.get(net, 0)) for net in choices],
            dtype=float,
        )
        weights /= weights.sum()
        return choices[int(rng.choice(len(choices), p=weights))]

    def select_batch(shortlist, exposure, X, pop):
        count = len(shortlist)
        if not count:
            return []

        learned, uncertainty = predictions(X)
        route_proxy = X[:, 1] + 0.10 * X[:, 3] + X[:, 8]
        eorder = np.argsort(X[:, 0], kind="stable")
        porder = np.argsort(route_proxy, kind="stable")

        if learned is None:
            rank_e = np.empty(count, dtype=float)
            rank_p = np.empty(count, dtype=float)
            rank_e[eorder] = np.arange(count)
            rank_p[porder] = np.arange(count)
            lorder = np.argsort(rank_e + rank_p, kind="stable")
        else:
            lorder = np.argsort(learned, kind="stable")

        batch_size = max(8, min(12, n // 2 + 2))
        quality_quota = max(2, batch_size // 3)
        batch_ids = []
        batch_sets = []

        def take(order, number, diverse=True):
            taken = 0
            for raw in order:
                idx = int(raw)
                if idx in batch_ids:
                    continue
                ns = set(shortlist[idx])
                if diverse and any(m - len(ns & old) < 3 for old in batch_sets):
                    continue
                batch_ids.append(idx)
                batch_sets.append(ns)
                taken += 1
                if taken >= number or len(batch_ids) >= batch_size:
                    break

        # Reserve a mixture of model quality, QUBO energy, and routing proxy.
        take(lorder, quality_quota)
        take(eorder, quality_quota)
        take(porder, 1)

        hot_order = sorted(
            range(count), key=lambda i: (exposure.get(shortlist[i], 0), -i),
            reverse=True,
        )
        take(hot_order, 1)

        # Reserve a novel trial among promising candidates rather than simply
        # filling all available LP calls with the same ranking.
        reference_sets = [set(net) for net in pop[:2 * n]]
        promising = list(dict.fromkeys(
            [int(i) for i in lorder[:max(n, count // 2)]]
            + [int(i) for i in porder[:n]]
        ))
        if promising and len(batch_ids) < batch_size - 1:
            def novelty(idx):
                ns = set(shortlist[idx])
                return min(
                    (m - len(ns & ref) for ref in reference_sets),
                    default=m,
                )
            take(sorted(promising, key=novelty, reverse=True), 1)

        if len(batch_ids) < batch_size:
            take(rng.permutation(count), 1)

        if uncertainty is not None and len(batch_ids) < batch_size:
            acquisition = learned - uncertainty
            competitive = learned <= np.median(learned)
            order = [
                int(i) for i in np.argsort(acquisition, kind="stable")
                if competitive[int(i)]
            ]
            take(order, 1)

        if len(batch_ids) < batch_size:
            take(lorder, batch_size - len(batch_ids), diverse=False)
        return [shortlist[i] for i in batch_ids[:batch_size]]

    rounds = 0
    while tools.lp_calls_left > 0 and time.perf_counter() - started < 140:
        rounds += 1
        pop = archive()
        if not pop:
            break

        if continuation is not None and rng.random() < 0.62:
            parent = continuation
        else:
            parent = choose_parent(pop)
        visits[parent] = visits.get(parent, 0) + 1
        continuation = None

        moves = generate(parent, pop)
        if not moves:
            stalls[parent] = stalls.get(parent, 0) + 1
            if rounds > 10 * max(1, len(seen)):
                break
            continue

        ordered = sorted(moves, key=energy)
        # Energy-ranked candidates and a weighted sample retain both promising
        # and bottleneck-exposed moves without proxy-evaluating every proposal.
        shortlist = ordered[:min(3 * n, len(ordered))]
        selected = set(shortlist)
        rest = [net for net in ordered if net not in selected]
        if rest:
            sample_size = min(3 * n, len(rest))
            weights = np.asarray(
                [2.0 if moves[net] > 0 else 1.0 for net in rest],
                dtype=float,
            )
            weights /= weights.sum()
            ids = rng.choice(len(rest), size=sample_size, replace=False, p=weights)
            shortlist.extend(rest[int(i)] for i in ids)

        X = routing_features(shortlist)
        batch = select_batch(shortlist, moves, X, pop)

        parent_score = seen.get(parent, float("inf"))
        local_best = None
        local_score = parent_score
        for net in batch:
            if tools.lp_calls_left <= 0:
                break
            idx = shortlist.index(net)
            try:
                score = evaluate(net, X[idx])
            except Exception as exc:
                if type(exc).__name__ == "BudgetExhausted":
                    break
                raise
            if score < local_score:
                local_best, local_score = net, score

        if local_best is not None:
            stalls[parent] = 0
            continuation = local_best
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
