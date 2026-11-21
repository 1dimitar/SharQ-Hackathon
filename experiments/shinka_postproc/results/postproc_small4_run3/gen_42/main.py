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
    """Return the best LP-certified network found within the available budget."""
    started = time.perf_counter()
    n = int(tools.n)
    m = 2 * n
    inf = float("inf")

    def running():
        return tools.lp_calls_left > 0 and time.perf_counter() - started < 142.0

    def canon(arcs):
        return tuple(sorted((int(u), int(v)) for u, v in arcs))

    def mask(net):
        value = 0
        for u, v in net:
            value |= 1 << (u * n + v)
        return value

    def separation(a, b):
        return (a ^ b).bit_count()

    def energy(net):
        value = float(tools.energy(net))
        return value if math.isfinite(value) else 1.0e100

    def ranks(values):
        order = np.argsort(values, kind="stable")
        result = np.empty(len(order), dtype=float)
        result[order] = np.arange(len(order), dtype=float)
        return result / max(1, len(order) - 1)

    traffic = np.maximum(np.asarray(demand, dtype=float), 0.0).copy()
    np.fill_diagonal(traffic, 0.0)
    traffic /= max(float(traffic.sum()) / max(m, 1), 1.0)

    class RoutingEstimator:
        """Batched feasible shortest-path routings, used only as cheap features."""

        def __init__(self, center, spread):
            self.center = center
            self.spread = spread

        def loads(self, successors, lengths):
            count = len(successors)
            bb = np.arange(count)[:, None]
            nodes = np.arange(n)[None, :]
            destinations = np.arange(n)[None, :]
            s0 = successors[:, :, 0]
            s1 = successors[:, :, 1]

            dist = np.full((count, n, n), np.inf)
            diagonal = np.arange(n)
            dist[:, diagonal, diagonal] = 0.0
            dist[bb, nodes, s0] = lengths[:, :, 0]
            dist[bb, nodes, s1] = lengths[:, :, 1]
            for k in range(n):
                np.minimum(
                    dist, dist[:, :, k, None] + dist[:, None, k, :],
                    out=dist,
                )

            connected = np.isfinite(dist).all(axis=(1, 2))
            d0 = dist[bb, s0, :] + lengths[:, :, 0, None]
            d1 = dist[bb, s1, :] + lengths[:, :, 1, None]
            good0 = np.isfinite(d0) & (d0 <= d1 + 1.0e-9)
            good1 = np.isfinite(d1) & (d1 <= d0 + 1.0e-9)
            denominator = np.maximum(good0.astype(float) + good1, 1.0)
            f0 = good0 / denominator
            f1 = good1 / denominator
            f0[:, diagonal, diagonal] = 0.0
            f1[:, diagonal, diagonal] = 0.0

            order = np.argsort(dist, axis=1)
            flow = np.broadcast_to(traffic, (count, n, n)).copy()
            for rank in range(n - 1, 0, -1):
                source = order[:, rank, :]
                amount = flow[bb, source, destinations].copy()
                v0 = s0[bb, source]
                v1 = s1[bb, source]
                flow[bb, v0, destinations] += (
                    amount * f0[bb, source, destinations]
                )
                flow[bb, v1, destinations] += (
                    amount * f1[bb, source, destinations]
                )

            loads = np.stack(
                ((flow * f0).sum(axis=2), (flow * f1).sum(axis=2)),
                axis=2,
            ).reshape(count, m)
            return loads, connected

        def features(self, nets, energies):
            # Chunking limits temporary storage on larger instances.
            if len(nets) > 192:
                blocks = [
                    self.features(nets[i:i + 192], energies[i:i + 192])
                    for i in range(0, len(nets), 192)
                ]
                return tuple(np.concatenate([b[j] for b in blocks])
                             for j in range(3))

            count = len(nets)
            arcs = np.asarray(nets, dtype=np.int64)
            successors = arcs[:, :, 1].reshape(count, n, 2)
            first, connected = self.loads(
                successors, np.ones((count, n, 2))
            )
            relative = first / np.maximum(first.mean(axis=1), 1.0e-9)[:, None]
            lengths = 1.0 + 1.5 * np.minimum(relative, 6.0) ** 2
            second, connected2 = self.loads(
                successors, lengths.reshape(count, n, 2)
            )
            connected &= connected2

            blend = first.copy()
            objective = first.max(axis=1) + 0.10 * first.mean(axis=1)
            for fraction in (0.2, 0.4, 0.6, 0.8, 1.0):
                trial = (1.0 - fraction) * first + fraction * second
                trial_objective = trial.max(axis=1) + 0.10 * trial.mean(axis=1)
                better = trial_objective < objective
                blend[better] = trial[better]
                objective[better] = trial_objective[better]

            top_count = max(2, n // 2)

            def summary(load):
                top = np.partition(
                    load, m - top_count, axis=1
                )[:, -top_count:]
                return np.column_stack((
                    load.mean(axis=1),
                    load.max(axis=1),
                    top.mean(axis=1),
                    np.sqrt(np.mean(load * load, axis=1)),
                ))

            values = np.column_stack((summary(first), summary(blend)))
            features = np.column_stack((
                np.log(np.maximum(values, 1.0e-9)),
                (np.asarray(energies) - self.center) / self.spread,
            ))
            proxy = 0.85 * blend.max(axis=1) + 0.15 * blend.mean(axis=1)
            proxy[~connected] = inf
            return features, proxy, connected

    class MoveModel:
        """Regularized regression on measured parent-to-child LP changes."""

        def __init__(self, initial):
            dimension = initial.shape[1]
            self.scale = np.maximum(np.std(initial, axis=0), 0.04)
            self.gram = np.eye(dimension) * 5.0
            self.rhs = np.zeros(dimension)
            self.count = 0

        def update(self, parent_features, features, parent_score, score):
            if not (0 < score < inf and 0 < parent_score < inf):
                return
            change = np.clip(
                (features - parent_features) / self.scale, -6.0, 6.0
            )
            target = float(np.clip(math.log(score / parent_score), -0.3, 0.3))
            weight = 1.0 / (1.0 + 3.0 * abs(target))
            self.gram += weight * np.outer(change, change)
            self.rhs += weight * change * target
            self.count += 1

        def rank(self, features, parent_features, proxies, energies):
            if self.count < 4 * n:
                return 0.80 * ranks(proxies) + 0.20 * ranks(energies)
            coefficients = np.linalg.solve(self.gram, self.rhs)
            changes = np.clip(
                (features - parent_features) / self.scale, -6.0, 6.0
            )
            return 0.80 * ranks(changes @ coefficients) + 0.20 * ranks(proxies)

    class Ledger:
        """Only this component sends newly proposed networks to the LP."""

        def __init__(self):
            self.records = {}
            self.details = {}
            self.best = None
            self.best_score = inf
            self.since_best = 0
            self.model = None

        def certify(self, net, en, features, proxy, parent=None):
            if net in self.records:
                return self.records[net][0]
            score = float(tools.lp(net))
            self.records[net] = (
                score, float(en), features.copy(), mask(net), float(proxy)
            )
            if parent is not None:
                old = self.records[parent]
                self.model.update(old[2], features, old[0], score)
            self.since_best += 1
            if score < self.best_score:
                self.best, self.best_score = net, score
                self.since_best = 0
            return score

        def detail(self, net):
            if net not in self.details:
                self.details[net] = tools.lp_detail(net)
            return self.details[net]

        def fresh(self, net):
            return net not in self.records and not tools.is_cached(net)

    ledger = Ledger()
    visits = {}
    stalls = {}

    # Stage 1: repair and describe the entire raw pool before allocating LP calls.
    repaired = set()
    for candidate in candidates:
        if time.perf_counter() - started >= 140.0:
            break
        try:
            net = canon(tools.repair(candidate))
            if tools.is_valid(net):
                repaired.add(net)
        except Exception as exc:
            if type(exc).__name__ in ("BudgetExhausted", "RunTimeout"):
                raise

    if not repaired:
        net = canon(tools.repair([]))
        if tools.is_valid(net):
            repaired.add(net)
    if not repaired:
        return []

    initial = sorted(repaired)
    initial_energy = np.asarray([energy(net) for net in initial])
    estimator = RoutingEstimator(
        float(np.median(initial_energy)),
        max(float(np.std(initial_energy)), 1.0e-9),
    )
    features, proxies, connected = estimator.features(initial, initial_energy)
    eligible = np.flatnonzero(connected).tolist()

    if not eligible and n >= 3:
        order = rng.permutation(n)
        initial = [canon(
            (int(order[i]), int(order[(i + step) % n]))
            for i in range(n) for step in (1, 2)
        )]
        initial_energy = np.asarray([energy(initial[0])])
        features, proxies, connected = estimator.features(initial, initial_energy)
        eligible = np.flatnonzero(connected).tolist()

    if not eligible:
        return list(initial[0])

    ledger.model = MoveModel(features[eligible])
    initial_masks = {i: mask(initial[i]) for i in eligible}
    initial_limit = min(len(eligible), max(12, 2 * n))
    chosen = []
    chosen_set = set()

    def add_start(i, require_distance=True):
        if i in chosen_set:
            return
        if require_distance and any(
            separation(initial_masks[i], initial_masks[j]) < max(6, n // 2 + 2)
            for j in chosen
        ):
            return
        chosen.append(i)
        chosen_set.add(i)

    for i in sorted(eligible, key=lambda j: (initial_energy[j], j)):
        add_start(i)
        if len(chosen) >= max(1, int(0.65 * initial_limit)):
            break
    for i in sorted(eligible, key=lambda j: (proxies[j], j)):
        if len(chosen) >= int(0.90 * initial_limit):
            break
        add_start(i)
    while len(chosen) < initial_limit:
        remaining = [i for i in eligible if i not in chosen_set]
        if not remaining:
            break
        if len(chosen) < int(0.90 * initial_limit):
            add_start(remaining[0], False)
        else:
            i = max(remaining, key=lambda j: min(
                separation(initial_masks[j], initial_masks[k]) for k in chosen
            ))
            add_start(i, False)

    for i in chosen:
        if not running():
            break
        ledger.certify(initial[i], initial_energy[i], features[i], proxies[i])

    if ledger.best is None:
        return list(initial[chosen[0]])

    # Stage 2: maintain multiple archive views over one common certification log.
    def archive():
        records = ledger.records
        ranked = sorted(
            (net for net in records if math.isfinite(records[net][0])),
            key=lambda net: records[net][0],
        )
        gap = max(6, n // 2)
        quota = max(3, n // 3)

        def separated(ordered, limit):
            result = []
            for net in ordered:
                if all(separation(records[net][3], records[x][3]) >= gap
                       for x in result):
                    result.append(net)
                    if len(result) >= limit:
                        break
            return result

        elite = ranked[:max(2, n // 4)]
        for net in ranked:
            if len(elite) >= n:
                break
            if net not in elite and all(
                separation(records[net][3], records[x][3]) >= gap for x in elite
            ):
                elite.append(net)

        near = [x for x in ranked
                if records[x][0] <= ledger.best_score * 1.045]
        energy_group = separated(sorted(near, key=lambda x: records[x][1]), quota)
        proxy_group = separated(sorted(near, key=lambda x: records[x][4]), quota)
        anchors = list(dict.fromkeys(elite + energy_group + proxy_group))

        available = [x for x in ranked
                     if records[x][0] <= ledger.best_score * 1.06
                     and x not in anchors]
        if len(available) > 6 * n:
            available = [available[int(i)] for i in np.linspace(
                0, len(available) - 1, 6 * n, dtype=int
            )]
        novel = []
        while available and len(novel) < quota:
            references = anchors + novel
            net = max(available, key=lambda x: (
                min(separation(records[x][3], records[y][3]) for y in references)
                / max(1, 4 * n)
                - 2.0 * (records[x][0] / ledger.best_score - 1.0)
            ))
            novel.append(net)
            available.remove(net)

        groups = [elite, energy_group or elite, proxy_group or elite, novel or elite]
        population = list(dict.fromkeys(x for group in groups for x in group))
        return groups, population

    def select_parent(groups):
        group_id = int(rng.choice(4, p=[0.58, 0.14, 0.14, 0.14]))
        group = groups[group_id]
        if group_id == 0 and rng.random() < 0.45:
            group = group[:min(4, len(group))]
        weights = np.asarray([
            1.0 / (math.sqrt(i + 1) *
                   (1.0 + 0.16 * visits.get(net, 0) + 0.45 * stalls.get(net, 0)))
            for i, net in enumerate(group)
        ])
        return group[int(rng.choice(len(group), p=weights / weights.sum()))]

    def switched(arcs, current, p, q):
        a, b = arcs[p]
        c, d = arcs[q]
        if a == c or b == d or a == d or c == b:
            return None
        if (a, d) in current or (c, b) in current:
            return None
        return tuple(sorted(
            (current - {(a, b), (c, d)}) | {(a, d), (c, b)}
        ))

    def random_switch(current):
        arcs = sorted(current)
        p = int(rng.integers(m))
        q = int(rng.integers(m - 1))
        if q >= p:
            q += 1
        return switched(arcs, current, p, q)

    def relink(parent, population, moves):
        records = ledger.records
        targets = [net for net in population
                   if separation(records[parent][3], records[net][3]) >= max(8, n)]
        if not targets:
            return
        target = set(targets[int(rng.integers(len(targets)))])
        cur = set(parent)
        for _ in range(max(3, n // 2)):
            outgoing = [[] for _ in range(n)]
            incoming = [[] for _ in range(n)]
            for a, b in cur:
                outgoing[a].append(b)
                incoming[b].append(a)
            options = []
            best_gain = 0
            for a, b in sorted(target - cur):
                for d in sorted(outgoing[a]):
                    for c in sorted(incoming[b]):
                        if c == d or (c, d) in cur:
                            continue
                        removed = {(a, d), (c, b)}
                        added = {(a, b), (c, d)}
                        gain = len(added & target) - len(removed & target)
                        if gain > best_gain:
                            best_gain = gain
                            options = [(removed, added)]
                        elif gain == best_gain and gain > 0:
                            options.append((removed, added))
            if not options:
                break
            removed, added = options[int(rng.integers(len(options)))]
            cur.difference_update(removed)
            cur.update(added)
            net = tuple(sorted(cur))
            if net != parent:
                moves.setdefault(net, 3)

    # Stage 3: independent proposal operators feed one common screening pipeline.
    def propose(parent, detail, population, round_id):
        arcs = list(parent)
        current = set(parent)
        moves = {}
        prices = detail.get("prices", {})
        loads = detail.get("loads", {})
        max_price = max((float(x) for x in prices.values()), default=0.0)
        hot = {tuple(map(int, arc)) for arc, value in prices.items()
               if float(value) > max(1.0e-12, max_price * 1.0e-8)}
        if not hot and loads:
            hot = {tuple(map(int, arc)) for arc in
                   sorted(loads, key=loads.get, reverse=True)[:max(2, n // 3)]}

        for p in range(m):
            for q in range(p + 1, m):
                if hot and arcs[p] not in hot and arcs[q] not in hot:
                    if rng.random() > min(0.5, 3.0 / n):
                        continue
                net = switched(arcs, current, p, q)
                if net is not None:
                    moves.setdefault(net, 0)

        hot_indices = [i for i, arc in enumerate(arcs) if arc in hot]
        trials = 6 * n + min(6 * n, 2 * n * stalls.get(parent, 0))
        for trial in range(trials):
            if hot_indices and trial % 2 == 0:
                p = hot_indices[int(rng.integers(len(hot_indices)))]
                others = rng.choice(m - 1, size=2, replace=False)
                q, r = (int(i) + int(int(i) >= p) for i in others)
                ids = (p, q, r)
            else:
                ids = rng.choice(m, size=3, replace=False)
            (a, b), (c, d), (e, f) = (arcs[int(i)] for i in ids)
            old = {(a, b), (c, d), (e, f)}
            rest = current - old
            for new in (
                {(a, d), (c, f), (e, b)},
                {(a, f), (c, b), (e, d)},
            ):
                if len(new) != 3 or any(u == v for u, v in new) or new & rest:
                    continue
                net = tuple(sorted(rest | new))
                if net != parent:
                    moves.setdefault(net, 1)

        escalation = min(
            max(1, n // 3),
            stalls.get(parent, 0) // 2 + ledger.since_best // max(1, 12 * n),
        )
        for _ in range(n + escalation):
            cur = current.copy()
            strength = 2 + int(rng.integers(3)) + escalation // 2
            successes = 0
            for _ in range(3 * strength):
                net = random_switch(cur)
                if net is not None:
                    cur = set(net)
                    successes += 1
                    if successes >= strength:
                        break
            net = tuple(sorted(cur))
            if net != parent:
                moves.setdefault(net, 2)

        if round_id % 3 == 0 or stalls.get(parent, 0) >= 3:
            relink(parent, population, moves)
        return moves

    batch_size = max(7, min(13, n // 2 + 1))

    def screen(parent, population, generated):
        fresh = [net for net in generated if ledger.fresh(net)]
        if not fresh:
            return None
        energies = np.asarray([energy(net) for net in fresh])
        feats, proxy, connected = estimator.features(fresh, energies)
        good = np.flatnonzero(connected)
        if not len(good):
            return None

        fresh = [fresh[int(i)] for i in good]
        energies, feats, proxy = energies[good], feats[good], proxy[good]
        kinds = np.asarray([generated[net] for net in fresh])
        combined = ledger.model.rank(
            feats, ledger.records[parent][2], proxy, energies
        )
        e_order = np.argsort(energies, kind="stable")
        p_order = np.argsort(proxy, kind="stable")
        model_order = np.argsort(combined, kind="stable")
        selected = []
        selected_set = set()

        def take(i):
            i = int(i)
            if i in selected_set or len(selected) >= batch_size:
                return False
            selected.append(i)
            selected_set.add(i)
            return True

        energy_quota = max(3, int(round(0.45 * batch_size)))
        for i in e_order:
            if kinds[i] == 0:
                take(i)
                if len(selected) >= min(2, energy_quota):
                    break
        for i in e_order:
            if len(selected) >= energy_quota:
                break
            take(i)
        for i in model_order:
            if len(selected) >= max(energy_quota, batch_size - 3):
                break
            take(i)
        for i in p_order:
            if take(i):
                break

        shortlist = model_order[:min(len(fresh), max(4 * batch_size, len(fresh) // 3))]
        available = [int(i) for i in shortlist if int(i) not in selected_set]
        if available and len(selected) < batch_size:
            anchors = [ledger.records[x][3] for x in population[:2 * n]]
            anchors.extend(mask(fresh[i]) for i in selected)
            novel = max(available, key=lambda i: (
                min(separation(mask(fresh[i]), old) for old in anchors),
                -combined[i],
            ))
            take(novel)

        if len(selected) < batch_size:
            tail = [int(i) for i in model_order[:max(batch_size, len(fresh) // 2)]
                    if int(i) not in selected_set and kinds[i] != 0]
            if tail:
                take(tail[int(rng.integers(len(tail)))])
        for i in model_order:
            if len(selected) >= batch_size:
                break
            take(i)
        return fresh, energies, feats, proxy, selected

    # Stage 4: short adaptive searches, rather than isolated archive mutations.
    follow = None
    idle = 0
    round_id = 0
    while running():
        round_id += 1
        groups, population = archive()
        if follow is not None and rng.random() < 0.60:
            parent = follow
        else:
            parent = select_parent(groups)
        follow = None
        visits[parent] = visits.get(parent, 0) + 1

        detail = ledger.detail(parent)
        generated = propose(parent, detail, population, round_id)
        batch = screen(parent, population, generated)
        if batch is None:
            stalls[parent] = stalls.get(parent, 0) + 1
            idle += 1
            if idle > 10 * n:
                break
            continue

        idle = 0
        fresh, energies, feats, proxy, selected = batch
        winner = None
        winner_score = ledger.records[parent][0]
        for i in selected:
            if not running():
                break
            score = ledger.certify(
                fresh[i], energies[i], feats[i], proxy[i], parent
            )
            if score < winner_score:
                winner, winner_score = fresh[i], score

        if winner is not None:
            stalls[parent] = 0
            if winner_score <= ledger.best_score * 1.035:
                follow = winner
        else:
            stalls[parent] = stalls.get(parent, 0) + 1

    return list(ledger.best) if ledger.best is not None else []
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
