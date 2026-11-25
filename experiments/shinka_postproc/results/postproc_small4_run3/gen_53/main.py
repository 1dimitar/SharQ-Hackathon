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
    """Diverse search with energy, inexpensive routing, and online LP prediction."""
    n = int(tools.n)
    m = 2 * n
    started = time.perf_counter()
    D = np.asarray(demand, dtype=float).copy()
    np.fill_diagonal(D, 0.0)
    traffic = max(float(D.sum()), 1.0)
    scale = traffic / max(m, 1)

    seen = {}
    details = {}
    features = {}
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
        v = float(tools.energy(net))
        return v if math.isfinite(v) else 0.0

    def evaluate(net, feat=None):
        nonlocal best_net, best_score
        if net not in seen:
            detail = tools.lp_detail(net)
            score = float(detail["score"])
            seen[net] = score
            details[net] = detail
            if feat is None:
                feat = routing_features([net])[0]
            features[net] = np.asarray(feat, dtype=float)
            if math.isfinite(score):
                training_x.append(features[net])
                training_y.append(score / scale)
        score = seen[net]
        if score < best_score:
            best_net, best_score = net, score
        return score

    def routing_features(nets):
        """Batched APSP and destination-based equal shortest-path splitting.

        This constructs a cheap feasible routing, not an LP approximation solver.
        Work, load concentration, and source/destination distance imbalance provide
        complementary signals to the QUBO energy.
        """
        b = len(nets)
        if not b:
            return np.empty((0, 9))
        arcs = np.asarray(nets, dtype=np.int64)
        heads = arcs[:, :, 1].reshape(b, n, 2)
        rows = np.arange(n)
        bi = np.arange(b)[:, None]

        dist = np.full((b, n, n), n + 1, dtype=np.int16)
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

        # Each vertex has exactly two incoming arcs. Sorting destinations gives
        # an inexpensive gather operation for accumulating incoming traffic.
        incoming = np.argsort(heads.reshape(b, m), axis=1, kind="stable")
        flow = np.broadcast_to(D, (b, n, n)).copy()
        loads = np.zeros((b, n, 2), dtype=float)
        diameter = min(n - 1, int(np.max(np.minimum(dist, n - 1))))
        for level in range(diameter, 0, -1):
            active = flow * (dist == level)
            f0 = active * fraction0
            f1 = active * fraction1
            loads[:, :, 0] += f0.sum(axis=2)
            loads[:, :, 1] += f1.sum(axis=2)
            edgeflow = np.stack((f0, f1), axis=2).reshape(b, m, n)
            incoming_flow = np.take_along_axis(
                edgeflow, incoming[:, :, None], axis=1
            ).reshape(b, n, 2, n).sum(axis=2)
            flow += incoming_flow

        linkloads = loads.reshape(b, m) / scale
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

    def predictions(X):
        """Return calibrated means and bounded epistemic uncertainties."""
        if len(training_y) < max(3 * n, 36):
            return None, None
        A = np.asarray(training_x)
        y = np.asarray(training_y)
        mean = A.mean(axis=0)
        sd = np.maximum(A.std(axis=0), 1e-8)
        Z = np.clip((A - mean) / sd, -12.0, 12.0)
        target_mean = float(y.mean())
        identity = np.eye(A.shape[1])
        gram = Z.T @ Z + 8.0 * identity
        try:
            # One small solve supplies both the original ridge coefficients
            # and the covariance geometry needed for uncertainty estimates.
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
            1, len(y) - A.shape[1] - 1
        )
        leverage = np.einsum("ij,jk,ik->i", W, inverse, W)
        # Include uncertainty in the intercept, but do not let extrapolation
        # claim more than one residual standard deviation of optimism.
        leverage = np.clip(leverage + 1.0 / len(y), 0.0, 1.0)
        uncertainty = np.sqrt(max(0.0, variance) * leverage)
        return predicted, uncertainty

    repaired = set()
    for cand in candidates:
        try:
            net = canon(tools.repair(cand))
            if tools.is_valid(net):
                repaired.add(net)
        except Exception:
            continue
    if not repaired:
        net = canon(tools.repair([]))
        if not tools.is_valid(net):
            return []
        repaired.add(net)

    initial_pool = sorted(repaired)
    initial_features = routing_features(initial_pool)
    initial_energy = np.argsort(initial_features[:, 0], kind="stable")
    initial_proxy = np.argsort(
        initial_features[:, 1] + 0.10 * initial_features[:, 3]
        + initial_features[:, 8],
        kind="stable",
    )
    initial_order = []
    used = set()
    # Interleave the original surrogate with an independent demand-based view.
    for k in range(len(initial_pool)):
        for order in (initial_energy, initial_energy, initial_proxy):
            if order is initial_energy:
                pos = next((int(i) for i in order if int(i) not in used), None)
            else:
                pos = next((int(i) for i in order if int(i) not in used), None)
            if pos is None:
                continue
            netset = set(initial_pool[pos])
            if initial_order and any(
                m - len(netset.intersection(initial_pool[j])) < 3
                for j in initial_order
            ):
                # Avoid spending all the starts on near-identical repairs.
                used.add(pos)
                continue
            initial_order.append(pos)
            used.add(pos)
            if len(initial_order) >= min(2 * n + 4, len(initial_pool)):
                break
        if len(initial_order) >= min(2 * n + 4, len(initial_pool)):
            break
        if len(used) == len(initial_pool):
            break

    if not initial_order:
        initial_order = [int(initial_energy[0])]
    for i in initial_order:
        if tools.lp_calls_left <= 0:
            break
        evaluate(initial_pool[i], initial_features[i])
    if best_net is None:
        return list(initial_pool[initial_order[0]])

    def archive():
        ranked = sorted(
            (net for net in seen if math.isfinite(seen[net])),
            key=seen.get,
        )
        if not ranked:
            return [best_net]

        # Keep LP elites, but reserve places for the independent free feature
        # rankings so that the archive does not collapse onto one criterion.
        out = ranked[:max(4, n // 2)]
        group_size = max(2, n // 3)
        energy_leaders = sorted(ranked, key=lambda net: features[net][0])
        proxy_leaders = sorted(
            ranked,
            key=lambda net: features[net][1] + 0.10 * features[net][3]
            + features[net][8],
        )
        for group in (energy_leaders, proxy_leaders):
            added = 0
            for net in group:
                if net not in out:
                    out.append(net)
                    added += 1
                    if added >= group_size:
                        break

        sets = [set(x) for x in out]
        separation = max(3, n // 3)
        for net in ranked:
            if len(out) >= 3 * n:
                break
            if net in out or seen[net] > best_score * 1.09:
                continue
            ns = set(net)
            if all(m - len(ns & old) >= separation for old in sets):
                out.append(net)
                sets.append(ns)
        return out[:3 * n]

    def switch(net, p, q):
        a, b = net[p]
        c, d = net[q]
        if a == c or b == d or a == d or c == b:
            return None
        cur = set(net)
        if (a, d) in cur or (c, b) in cur:
            return None
        cur.remove((a, b))
        cur.remove((c, d))
        cur.add((a, d))
        cur.add((c, b))
        return tuple(sorted(cur))

    def generate(parent, pop):
        cur = set(parent)
        detail = details.get(parent, {})
        hot = {
            tuple(a) for a, price in detail.get("prices", {}).items()
            if float(price) > 1e-10
        }
        if not hot:
            loads = detail.get("loads", {})
            hot = set(sorted(loads, key=loads.get, reverse=True)[:max(2, n // 4)])

        # Values encode whether a move touches a bottleneck.
        moves = {}
        for p in range(m):
            for q in range(p + 1, m):
                candidate = switch(parent, p, q)
                if candidate is not None:
                    moves[candidate] = int(parent[p] in hot or parent[q] in hot)

        stalls_here = min(stalls.get(parent, 0), 4)
        rotation_trials = 5 * n + min(stalls.get(parent, 0), 8) * n
        for _ in range(rotation_trials):
            ids = rng.choice(m, size=3, replace=False)
            old = [parent[int(i)] for i in ids]
            tails = [a for a, b in old]
            heads = [b for a, b in old]
            if len(set(tails)) < 3 or len(set(heads)) < 3:
                continue
            shift = 1 + int(rng.integers(2))
            new = {
                (tails[k], heads[(k + shift) % 3]) for k in range(3)
            }
            rest = cur.difference(old)
            if any(a == b for a, b in new) or new & rest:
                continue
            candidate = tuple(sorted(rest | new))
            moves[candidate] = int(bool(set(old) & hot))

        # Enumerate compatible rotations anchored on each hot arc. Each such
        # rotation removes that arc; keep low-direct-demand alternatives for
        # further ranking by the energy and routing screens.
        hot_ids = [i for i, edge in enumerate(parent) if edge in hot]
        keep_per_hot = max(2, n // 4)
        for p in hot_ids:
            ranked_rotations = []
            for q in range(m):
                if q == p:
                    continue
                for r in range(q + 1, m):
                    if r == p:
                        continue
                    old = [parent[p], parent[q], parent[r]]
                    tails = [a for a, b in old]
                    heads = [b for a, b in old]
                    if len(set(tails)) < 3 or len(set(heads)) < 3:
                        continue
                    rest = cur.difference(old)
                    for shift in (1, 2):
                        new = {
                            (tails[k], heads[(k + shift) % 3]) for k in range(3)
                        }
                        if any(a == b for a, b in new) or new & rest:
                            continue
                        candidate = tuple(sorted(rest | new))
                        pressure = sum(float(D[a, b]) for a, b in new)
                        ranked_rotations.append((pressure, candidate))
            ranked_rotations.sort(key=lambda item: item[0])
            for _, candidate in ranked_rotations[:keep_per_hot]:
                moves[candidate] = 1

        # Larger perturbations and short steps toward another archive topology.
        target = pop[int(rng.integers(len(pop)))]
        if stalls_here >= 2 and len(pop) > 1:
            alternatives = [x for x in pop if x != parent]
            parent_set = set(parent)
            distances = [m - len(parent_set.intersection(x)) for x in alternatives]
            farthest = max(distances)
            distant = [x for x, d in zip(alternatives, distances) if d == farthest]
            target = distant[int(rng.integers(len(distant)))]
        target_set = set(target)
        walk_trials = 2 * n + stalls_here * max(1, n // 3)
        for trial in range(walk_trials):
            state = parent
            count = 2 + int(rng.integers(max(2, n // 3))) + stalls_here // 2
            for step in range(count):
                options = []
                for _ in range(8):
                    p, q = rng.choice(m, size=2, replace=False)
                    child = switch(state, int(p), int(q))
                    if child is not None:
                        options.append(child)
                if not options:
                    continue
                if trial < walk_trials // 2:
                    state = max(options, key=lambda x: len(target_set.intersection(x)))
                else:
                    state = options[int(rng.integers(len(options)))]
            if state != parent:
                moves[state] = 0

        return {
            net: flag for net, flag in moves.items()
            if net not in seen and not tools.is_cached(net)
        }

    rounds = 0
    while tools.lp_calls_left > 0 and time.perf_counter() - started < 140:
        rounds += 1
        pop = archive()
        if continuation is not None and rng.random() < 0.62:
            parent = continuation
        else:
            elite = pop[:max(4, n // 2)]
            energy_group = sorted(pop, key=lambda net: features[net][0])[:max(4, n // 3)]
            proxy_group = sorted(
                pop,
                key=lambda net: features[net][1] + 0.10 * features[net][3]
                + features[net][8],
            )[:max(4, n // 3)]
            diverse_group = pop[max(4, n // 2):] or pop
            groups = (elite, energy_group, proxy_group, diverse_group)
            group_id = int(rng.choice(4, p=[0.35, 0.25, 0.25, 0.15]))
            choices = groups[group_id]
            weights = np.asarray([
                1.0 / math.sqrt(1.0 + visits.get(x, 0))
                for x in choices
            ])
            weights /= weights.sum()
            parent = choices[int(rng.choice(len(choices), p=weights))]
        visits[parent] = visits.get(parent, 0) + 1
        continuation = None

        moves = generate(parent, pop)
        if not moves:
            stalls[parent] = stalls.get(parent, 0) + 1
            if rounds > 10 * max(1, len(seen)):
                break
            continue

        ordered = sorted(moves, key=energy)
        # Bound routing-screening overhead. Retain low-energy moves and a
        # bottleneck-biased random sample of the rest, including large moves.
        shortlist = ordered[:min(3 * n, len(ordered))]
        selected_set = set(shortlist)
        rest = [x for x in ordered if x not in selected_set]
        if rest:
            take = min(4 * n, len(rest))
            weights = np.asarray([2.0 if moves[x] else 1.0 for x in rest])
            weights /= weights.sum()
            ids = rng.choice(len(rest), size=take, replace=False, p=weights)
            shortlist.extend(rest[int(i)] for i in ids)

        X = routing_features(shortlist)
        learned, uncertainty = predictions(X)
        proxy = X[:, 1] + 0.10 * X[:, 3] + X[:, 8]
        eorder = np.argsort(X[:, 0], kind="stable")
        porder = np.argsort(proxy, kind="stable")
        if learned is None:
            rank_e = np.empty(len(shortlist))
            rank_p = np.empty(len(shortlist))
            rank_e[eorder] = np.arange(len(shortlist))
            rank_p[porder] = np.arange(len(shortlist))
            lorder = np.argsort(rank_e + rank_p, kind="stable")
        else:
            lorder = np.argsort(learned, kind="stable")

        batch_size = max(8, n // 2)
        quality_quota = max(2, batch_size // 3)
        if uncertainty is not None:
            # Preserve both quality quotas plus routing, novelty, and random
            # trials while guaranteeing room for one uncertainty acquisition.
            batch_size = max(batch_size, 2 * quality_quota + 4)
        batch_ids = []
        batch_sets = []

        def take_from(order, number, diverse=True):
            taken = 0
            for raw in order:
                i = int(raw)
                if i in batch_ids:
                    continue
                ns = set(shortlist[i])
                if diverse and any(m - len(ns & old) < 3 for old in batch_sets):
                    continue
                batch_ids.append(i)
                batch_sets.append(ns)
                taken += 1
                if taken >= number or len(batch_ids) >= batch_size:
                    break

        take_from(lorder, quality_quota)
        take_from(eorder, quality_quota)
        take_from(porder, 1)

        # Genuine exploration capacity: unlike filling the batch first, these
        # reserved positions cannot disappear into energy-ranked selections.
        reference_sets = [set(x) for x in pop[:2 * n]]
        promising = list(dict.fromkeys(
            [int(i) for i in lorder[:max(n, len(shortlist) // 2)]]
            + [int(i) for i in porder[:n]]
        ))
        novelty = sorted(
            promising,
            key=lambda i: min(
                m - len(set(shortlist[i]) & ref) for ref in reference_sets
            ),
            reverse=True,
        )
        if len(batch_ids) < batch_size - 1:
            take_from(novelty, 1)
        if len(batch_ids) < batch_size:
            take_from(rng.permutation(len(shortlist)), 1)

        if uncertainty is not None and len(batch_ids) < batch_size:
            # Explore plausible improvements, not merely extreme feature
            # outliers. The other portfolio allocations remain untouched.
            competitive = learned <= np.median(learned)
            acquisition = learned - uncertainty
            uncertainty_order = [
                int(i)
                for i in np.argsort(acquisition, kind="stable")
                if competitive[int(i)]
            ]
            take_from(uncertainty_order, 1)

        if len(batch_ids) < batch_size:
            take_from(lorder, batch_size - len(batch_ids), diverse=False)

        parent_score = seen[parent]
        local_best = None
        local_score = parent_score
        for i in batch_ids:
            if tools.lp_calls_left <= 0:
                break
            net = shortlist[i]
            score = evaluate(net, X[i])
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
