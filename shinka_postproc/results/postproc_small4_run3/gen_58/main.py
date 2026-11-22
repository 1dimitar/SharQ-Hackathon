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
    """Adaptive, multi-signal search over repaired degree-regular networks."""
    n = int(tools.n)
    m = 2 * n
    started = time.perf_counter()

    D = np.asarray(demand, dtype=float).copy()
    np.fill_diagonal(D, 0.0)
    traffic = max(float(D.sum()), 1.0)
    scale = traffic / max(m, 1)
    dimension = 9

    seen = {}
    details = {}
    masks = {}
    visits = {}
    stalls = {}
    best_net = None
    best_score = float("inf")
    continuation = None

    # Online ridge-model sufficient statistics, for a modest prediction signal.
    count = 0
    sx = np.zeros(dimension)
    sxx = np.zeros((dimension, dimension))
    sxy = np.zeros(dimension)
    sy = 0.0
    syy = 0.0

    sources = ("energy", "routing", "dual", "prediction", "novelty", "random")
    source_quality = {name: 0.08 for name in sources}

    def canon(arcs):
        return tuple(sorted((int(a), int(b)) for a, b in arcs))

    def topology_mask(net):
        value = 0
        for a, b in net:
            value |= 1 << (a * n + b)
        return value

    def energy(net):
        try:
            value = float(tools.energy(net))
            return value if math.isfinite(value) else 0.0
        except Exception:
            return 0.0

    def routing_features(nets, energies=None):
        """Equal-split shortest-path loads and demand-distance features."""
        batch = len(nets)
        if batch == 0:
            return np.empty((0, dimension), dtype=float)

        arcs = np.asarray(nets, dtype=np.intp)
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

        # Valid networks have exactly two incoming arcs per vertex.
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
            flow += np.take_along_axis(
                edgeflow, incoming[:, :, None], axis=1
            ).reshape(batch, n, 2, n).sum(axis=2)

        linkloads = loads.reshape(batch, m) / scale
        ordered_loads = np.sort(linkloads, axis=1)
        weighted_distance = dist * D[None, :, :]
        if energies is None:
            energies = np.asarray([energy(net) for net in nets], dtype=float)

        return np.column_stack((
            energies,
            weighted_distance.sum(axis=(1, 2)) / traffic,
            ordered_loads[:, -1],
            ordered_loads[:, -min(4, m):].mean(axis=1),
            np.sqrt(np.mean(linkloads * linkloads, axis=1)),
            weighted_distance.sum(axis=2).max(axis=1) / scale,
            weighted_distance.sum(axis=1).max(axis=1) / scale,
            D[arcs[:, :, 0], arcs[:, :, 1]].sum(axis=1) / traffic,
            unreachable,
        ))

    def evaluate(net, feature=None):
        nonlocal best_net, best_score, count, sy, syy
        if net not in seen:
            detail = tools.lp_detail(net)
            score = float(detail["score"])
            seen[net] = score
            details[net] = detail
            masks[net] = topology_mask(net)

            if math.isfinite(score):
                if feature is None:
                    feature = routing_features([net])[0]
                feature = np.asarray(feature, dtype=float)
                y = score / scale
                count += 1
                sx[:] += feature
                sxx[:] += np.outer(feature, feature)
                sxy[:] += feature * y
                sy += y
                syy += y * y

        score = seen[net]
        if score < best_score:
            best_net, best_score = net, score
        return score

    def predict(X):
        """Regularized standardized linear prediction and a small uncertainty bonus."""
        if count < max(2 * n, 30):
            return None

        mean = sx / count
        target = sy / count
        centered = sxx - np.outer(sx, sx) / count
        covariance = centered / count
        sd = np.sqrt(np.maximum(np.diag(covariance), 1e-12))
        cov_z = covariance / np.outer(sd, sd)
        cross_z = (sxy - sx * target) / (count * sd)

        ridge = 7.0
        system = cov_z + ridge * np.eye(dimension)
        try:
            solved = np.linalg.solve(
                system, np.column_stack((cross_z, np.eye(dimension)))
            )
        except np.linalg.LinAlgError:
            return None

        coefficient = solved[:, 0]
        inverse = solved[:, 1:]
        Z = np.clip((X - mean) / sd, -12.0, 12.0)
        prediction = target + Z @ coefficient

        variance_y = max(0.0, syy / count - target * target)
        residual = (
            variance_y
            - 2.0 * coefficient.dot(cross_z)
            + coefficient.dot(cov_z @ coefficient)
        )
        sigma = max(math.sqrt(max(0.0, residual)), 0.01)
        leverage = np.maximum(
            np.einsum("ij,jk,ik->i", Z, inverse, Z), 0.0
        )
        uncertainty = sigma * np.sqrt(leverage)
        return prediction, uncertainty

    # Repair every raw sample for free and rank the resulting topologies.
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
    initial_X = routing_features(initial_pool)
    eorder = list(map(int, np.argsort(initial_X[:, 0], kind="stable")))
    initial_proxy = (
        initial_X[:, 2] + 0.20 * initial_X[:, 3]
        + 0.08 * initial_X[:, 4] + 0.02 * initial_X[:, 1]
        + 0.10 * initial_X[:, 8]
    )
    porder = list(map(int, np.argsort(initial_proxy, kind="stable")))
    initial_masks = [topology_mask(net) for net in initial_pool]

    # A smaller start set reserves more of the LP budget for topology search.
    start_limit = min(n, len(initial_pool))
    initial_ids = []
    used_ids = set()
    cursors = [0, 0]
    for turn in range(max(1, 3 * len(initial_pool))):
        if len(initial_ids) >= start_limit:
            break
        which = (0, 0, 1)[turn % 3]
        order = eorder if which == 0 else porder
        cursor = cursors[which]
        while cursor < len(order) and order[cursor] in used_ids:
            cursor += 1
        cursors[which] = cursor + 1
        if cursor >= len(order):
            continue
        i = order[cursor]
        if all(
            (initial_masks[i] ^ initial_masks[j]).bit_count() >= 6
            for j in initial_ids
        ):
            initial_ids.append(i)
            used_ids.add(i)

    if len(initial_ids) < start_limit:
        for i in eorder:
            if i not in used_ids:
                initial_ids.append(i)
                used_ids.add(i)
                if len(initial_ids) >= start_limit:
                    break

    if not initial_ids:
        initial_ids = [eorder[0]]

    for i in initial_ids:
        if tools.lp_calls_left <= 0:
            break
        try:
            evaluate(initial_pool[i], initial_X[i])
        except Exception as exc:
            if type(exc).__name__ == "BudgetExhausted":
                break
            continue

    if best_net is None:
        return list(initial_pool[initial_ids[0]])

    def archive():
        ranked = sorted(
            (net for net in seen if math.isfinite(seen[net])),
            key=seen.get,
        )
        if not ranked:
            return [best_net]

        elite_count = min(len(ranked), max(4, n // 2))
        out = ranked[:elite_count]
        out_masks = [masks[net] for net in out]
        separation = max(6, n // 2)

        for net in ranked[elite_count:]:
            if len(out) >= 3 * n:
                break
            if seen[net] > best_score * 1.08:
                break
            value = masks[net]
            if all((value ^ old).bit_count() >= separation for old in out_masks):
                out.append(net)
                out_masks.append(value)
        return out or [best_net]

    def switch(net, p, q, current=None):
        a, b = net[p]
        c, d = net[q]
        if a == c or b == d or a == d or c == b:
            return None
        if current is None:
            current = set(net)
        if (a, d) in current or (c, b) in current:
            return None
        return tuple(sorted((current - {(a, b), (c, d)}) | {(a, d), (c, b)}))

    def make_rotation(current, arcs, ids):
        old = [arcs[int(i)] for i in ids]
        tails = [a for a, _ in old]
        heads = [b for _, b in old]
        if len(set(tails)) != 3 or len(set(heads)) != 3:
            return None
        alternatives = (
            {(tails[0], heads[1]), (tails[1], heads[2]), (tails[2], heads[0])},
            {(tails[0], heads[2]), (tails[1], heads[0]), (tails[2], heads[1])},
        )
        old_set = set(old)
        rest = current.difference(old_set)
        for new in alternatives:
            if any(a == b for a, b in new) or new & rest:
                continue
            result = tuple(sorted(rest | new))
            if result != tuple(sorted(current)):
                return result
        # Try the other orientation if the first was invalid.
        for new in alternatives:
            if any(a == b for a, b in new) or new & rest:
                continue
            return tuple(sorted(rest | new))
        return None

    def generate(parent, pop):
        current = set(parent)
        arcs = list(parent)
        detail = details.get(parent, {})
        prices = detail.get("prices", {}) if detail else {}
        hot = {
            tuple(map(int, arc))
            for arc, price in prices.items()
            if float(price) > 1e-10
        }
        if not hot:
            loads = detail.get("loads", {}) if detail else {}
            if loads:
                hot = set(
                    tuple(map(int, arc))
                    for arc in sorted(loads, key=loads.get, reverse=True)[:max(2, n // 4)]
                )

        moves = {}

        # Full 2-switch neighborhood; flag moves that alter an LP-critical link.
        for p in range(m):
            for q in range(p + 1, m):
                child = switch(parent, p, q, current)
                if child is not None:
                    moves[child] = bool(parent[p] in hot or parent[q] in hot)

        stalled = stalls.get(parent, 0)
        rotation_trials = 4 * n + min(stalled, 8) * n
        for _ in range(rotation_trials):
            ids = rng.choice(m, size=3, replace=False)
            child = make_rotation(current, arcs, ids)
            if child is not None:
                old = {arcs[int(i)] for i in ids}
                moves[child] = bool(old & hot)

        # Explicitly ensure a reasonable number of rotations touch hot links.
        hot_ids = [i for i, arc in enumerate(arcs) if arc in hot]
        for _ in range(2 * n):
            if not hot_ids:
                break
            target = int(rng.choice(hot_ids))
            others = rng.choice(m - 1, size=2, replace=False)
            ids = [target] + [
                int(i) if int(i) < target else int(i) + 1 for i in others
            ]
            child = make_rotation(current, arcs, ids)
            if child is not None:
                moves[child] = True

        # Random multi-switch kicks escape one-move local minima.
        for _ in range(max(8, n)):
            state = set(parent)
            for _step in range(2 + int(rng.integers(max(2, n // 4)))):
                now = tuple(sorted(state))
                if len(now) < 2:
                    break
                p, q = rng.choice(m, size=2, replace=False)
                child = switch(now, int(p), int(q), state)
                if child is not None:
                    state = set(child)
            if state != current:
                moves[tuple(sorted(state))] = False

        # Short, sampled path-relinking walks toward other archive topologies.
        if len(pop) > 1:
            for _ in range(2):
                target = pop[int(rng.integers(len(pop)))]
                if target == parent:
                    continue
                target_set = set(target)
                state = tuple(parent)
                for step in range(max(2, n // 4)):
                    state_set = set(state)
                    state_arcs = list(state)
                    options = []
                    for _attempt in range(18):
                        p, q = rng.choice(m, size=2, replace=False)
                        child = switch(state, int(p), int(q), state_set)
                        if child is not None:
                            options.append(child)
                    if not options:
                        break
                    # Greedy early steps make progress toward the target;
                    # later steps retain some stochastic diversification.
                    if step < max(1, n // 8):
                        state = max(
                            options,
                            key=lambda x: len(target_set.intersection(x)),
                        )
                    else:
                        state = options[int(rng.integers(len(options)))]
                    moves[state] = False

        return {
            net: focused for net, focused in moves.items()
            if net != parent and net not in seen
        }

    def choose_parent(pop):
        if continuation is not None and rng.random() < 0.52:
            return continuation

        u = rng.random()
        if u < 0.42:
            choices = pop[:min(5, len(pop))]
        elif u < 0.78:
            choices = pop[:min(n, len(pop))]
        else:
            choices = pop

        weights = np.asarray([
            1.0 / math.sqrt(1.0 + visits.get(net, 0))
            for net in choices
        ], dtype=float)
        weights /= weights.sum()
        return choices[int(rng.choice(len(choices), p=weights))]

    rounds = 0
    empty_rounds = 0
    while tools.lp_calls_left > 0 and time.perf_counter() - started < 140:
        rounds += 1
        pop = archive()
        parent = choose_parent(pop)
        continuation = None
        visits[parent] = visits.get(parent, 0) + 1

        moves = generate(parent, pop)
        if not moves:
            stalls[parent] = stalls.get(parent, 0) + 1
            empty_rounds += 1
            if empty_rounds > 8 * n:
                break
            continue
        empty_rounds = 0

        move_list = list(moves)
        move_energy = np.asarray([energy(net) for net in move_list])
        energy_order = np.argsort(move_energy, kind="stable")

        # Form a bounded screening set: strong energy moves, hot-link moves,
        # plus random candidates so the proxy and model see other regions.
        selected = set()
        for raw in energy_order[:min(2 * n, len(move_list))]:
            selected.add(int(raw))

        focused_order = sorted(
            (i for i, net in enumerate(move_list) if moves[net]),
            key=lambda i: move_energy[i],
        )
        for i in focused_order[:min(2 * n, len(focused_order))]:
            selected.add(i)

        remaining = [i for i in range(len(move_list)) if i not in selected]
        if remaining:
            take = min(2 * n, len(remaining))
            chosen = rng.choice(len(remaining), size=take, replace=False)
            for raw in chosen:
                selected.add(remaining[int(raw)])

        shortlist_ids = sorted(selected)
        shortlist = [move_list[i] for i in shortlist_ids]
        shortlist_energy = move_energy[shortlist_ids]
        focused = [moves[net] for net in shortlist]
        X = routing_features(shortlist, shortlist_energy)

        proxy = (
            X[:, 2] + 0.20 * X[:, 3] + 0.08 * X[:, 4]
            + 0.02 * X[:, 1] + 0.10 * X[:, 8]
        )
        eorder = list(map(int, np.argsort(X[:, 0], kind="stable")))
        rorder = list(map(int, np.argsort(proxy, kind="stable")))
        model = predict(X)

        if model is None:
            prediction_order = sorted(
                range(len(shortlist)),
                key=lambda i: (
                    eorder.index(i) / max(1, len(shortlist))
                    + rorder.index(i) / max(1, len(shortlist))
                ),
            )
            prediction = uncertainty = None
        else:
            prediction, uncertainty = model
            acquisition = prediction - 0.35 * uncertainty
            prediction_order = list(map(int, np.argsort(acquisition, kind="stable")))

        # Novelty is measured against the evaluated archive and prioritizes
        # otherwise-promising candidates.
        history_masks = list(masks.values())
        promising = set(eorder[:min(3 * n, len(eorder))])
        promising.update(rorder[:min(3 * n, len(rorder))])
        promising.update(prediction_order[:min(2 * n, len(prediction_order))])

        candidate_masks = [topology_mask(net) for net in shortlist]
        novelty_order = sorted(
            range(len(shortlist)),
            key=lambda i: (
                i in promising,
                min(
                    ((candidate_masks[i] ^ old).bit_count() for old in history_masks),
                    default=2 * m,
                ),
                -float(proxy[i]),
            ),
            reverse=True,
        )
        dual_order = sorted(
            range(len(shortlist)),
            key=lambda i: (focused[i], -float(proxy[i]), -float(X[i, 0])),
            reverse=True,
        )
        random_order = list(map(int, rng.permutation(len(shortlist))))

        orders = {
            "energy": eorder,
            "routing": rorder,
            "dual": dual_order,
            "prediction": prediction_order,
            "novelty": novelty_order,
            "random": random_order,
        }

        batch_size = max(8, n // 2)
        quotas = {name: 1 for name in sources}
        weights = np.asarray([
            0.18 + math.sqrt(max(0.0, source_quality[name]))
            for name in sources
        ], dtype=float)
        weights /= weights.sum()
        for _ in range(max(0, batch_size - len(sources))):
            chosen_source = sources[int(rng.choice(len(sources), p=weights))]
            quotas[chosen_source] += 1

        batch = []
        batch_source = {}
        batch_set = set()

        def add_from(source, number, require_diversity=True):
            added = 0
            for i in orders[source]:
                if i in batch_set:
                    continue
                if require_diversity and any(
                    (candidate_masks[i] ^ candidate_masks[j]).bit_count() < 4
                    for j in batch
                ):
                    continue
                batch.append(i)
                batch_set.add(i)
                batch_source[i] = source
                added += 1
                if added >= number:
                    break
            return added

        # Each signal gets a protected first opportunity; later slots adapt
        # to its empirical rate of improving the parent or global best.
        for source in sources:
            add_from(source, quotas[source], require_diversity=True)

        if len(batch) < batch_size:
            for source in sources:
                add_from(source, batch_size - len(batch), require_diversity=False)
                if len(batch) >= batch_size:
                    break

        if len(batch) < batch_size:
            for i in random_order:
                if i not in batch_set:
                    batch.append(i)
                    batch_set.add(i)
                    batch_source[i] = "random"
                    if len(batch) >= batch_size:
                        break

        parent_score = seen[parent]
        local_best = None
        local_score = parent_score

        for i in batch:
            if tools.lp_calls_left <= 0 or time.perf_counter() - started >= 140:
                break

            net = shortlist[i]
            old_global = best_score
            try:
                score = evaluate(net, X[i])
            except Exception as exc:
                if type(exc).__name__ == "BudgetExhausted":
                    return list(best_net) if best_net is not None else []
                continue

            if score < local_score:
                local_best, local_score = net, score

            source = batch_source.get(i, "random")
            reward = 0.0
            if score < parent_score:
                relative_gain = (parent_score - score) / max(abs(parent_score), 1.0)
                reward = min(1.0, 0.55 + 8.0 * relative_gain)
            if score < old_global:
                reward = max(reward, 1.0)
            source_quality[source] = (
                0.92 * source_quality[source] + 0.08 * reward
            )

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
