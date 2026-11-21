"""Classical post-processing of QAOA output for QOBLIB 08. ShinkaEvolve edits the EVOLVE block."""

import json
import time

import numpy as np

# EVOLVE-BLOCK-START
import math


def improve(demand, candidates, tools, rng):
    n = int(tools.n)
    evaluated = {}
    energy_cache = {}
    best_net = None
    best_score = float("inf")

    def canon(arcs):
        return tuple(sorted(tuple(map(int, a)) for a in arcs))

    def edge_set(net):
        return set(net)

    def energy(net):
        net = canon(net)
        if net not in energy_cache:
            try:
                energy_cache[net] = float(tools.energy(net))
            except Exception:
                energy_cache[net] = float("inf")
        return energy_cache[net]

    def evaluate(net):
        nonlocal best_net, best_score
        net = canon(net)
        if net in evaluated or not tools.is_valid(net):
            return evaluated.get(net, float("inf"))
        if tools.lp_calls_left <= 0:
            return float("inf")
        try:
            value = float(tools.lp(net))
        except Exception:
            return float("inf")
        evaluated[net] = value
        if math.isfinite(value) and value < best_score:
            best_net, best_score = net, value
        return value

    # Repair first, then LP-score energy-favored seeds while keeping topology
    # spread; repair and energy are free, so duplicates cost nothing.
    repaired = set()
    for cand in candidates:
        try:
            net = canon(tools.repair(cand))
            if tools.is_valid(net):
                repaired.add(net)
        except Exception:
            pass
    if not repaired:
        try:
            net = canon(tools.repair([]))
            if tools.is_valid(net):
                repaired.add(net)
        except Exception:
            return []

    ordered = sorted(repaired, key=energy)
    seeds = []
    for net in ordered:
        if all(len(edge_set(net) ^ edge_set(old)) >= 6 for old in seeds):
            seeds.append(net)
        if len(seeds) >= 18:
            break
    for net in ordered:
        if len(seeds) >= 36:
            break
        if net not in seeds:
            seeds.append(net)

    for net in seeds:
        if tools.lp_calls_left <= 0:
            break
        evaluate(net)

    if best_net is None:
        return list(seeds[0]) if seeds else []

    def archive():
        """Build a compact score-ranked but structurally diverse model set."""
        ranked = sorted(
            (net for net, val in evaluated.items() if math.isfinite(val)),
            key=evaluated.get
        )
        chosen = ranked[:min(5, len(ranked))]
        for net in ranked[len(chosen):]:
            if all(len(edge_set(net) ^ edge_set(old)) >= 6 for old in chosen):
                chosen.append(net)
            if len(chosen) >= 24:
                break
        for net in ranked:
            if len(chosen) >= 32:
                break
            if net not in chosen:
                chosen.append(net)
        return chosen

    def sample_model(parents, count):
        """Sample topologies from smoothed elite edge marginals, then repair."""
        if not parents:
            return set()
        # Favor good solutions without allowing one network to define the model.
        elite = parents[:min(18, len(parents))]
        weights = np.array([1.0 / (1.0 + 0.12 * i) for i in range(len(elite))])
        weights /= weights.sum()

        marginal = np.zeros((n, n), dtype=float)
        for w, net in zip(weights, elite):
            for a, b in net:
                marginal[a, b] += w

        prior = 2.0 / max(1, n - 1)
        # The uniform component prevents premature collapse to a narrow basin.
        prob = 0.72 * marginal + 0.28 * prior
        np.fill_diagonal(prob, 0.0)

        output = set()
        for _ in range(count):
            raw = []
            # Sampling two outgoing links per source preserves half the degree
            # constraints before repair; incoming degrees are corrected by it.
            for a in range(n):
                dest = [b for b in range(n) if b != a]
                p = prob[a, dest].copy()
                if float(p.sum()) <= 0:
                    p[:] = 1.0
                p /= p.sum()
                try:
                    chosen = rng.choice(len(dest), size=2, replace=False, p=p)
                except Exception:
                    chosen = rng.choice(len(dest), size=2, replace=False)
                raw.extend((a, dest[int(k)]) for k in chosen)

            # Occasionally splice a few links from another elite member into
            # the sampled blueprint before closest-valid repair.
            if len(elite) > 1 and rng.random() < 0.35:
                donor = elite[int(rng.integers(len(elite)))]
                if donor:
                    keep = max(1, int(rng.integers(2, min(8, len(donor)) + 1)))
                    ids = rng.choice(len(donor), size=keep, replace=False)
                    raw.extend(donor[int(i)] for i in ids)
            try:
                net = canon(tools.repair(raw))
                if tools.is_valid(net):
                    output.add(net)
            except Exception:
                pass
        return output

    def random_switch(net):
        arcs = list(net)
        current = set(net)
        if len(arcs) < 2:
            return net
        for _ in range(20):
            p, q = map(int, rng.choice(len(arcs), size=2, replace=False))
            a, b = arcs[p]
            c, d = arcs[q]
            x, y = (a, d), (c, b)
            if a != d and c != b and x not in current and y not in current:
                return canon((current - {(a, b), (c, d)}) | {x, y})
        return net

    def kick(net, steps):
        cur = net
        for _ in range(steps):
            nxt = random_switch(cur)
            if nxt != cur:
                cur = nxt
        return cur

    def dual_neighbors(parent, detail):
        """Enumerate 2-switches and rank by estimated bottleneck relief."""
        arcs = list(parent)
        current = set(parent)
        prices = {}
        loads = {}
        try:
            prices = {tuple(map(int, k)): float(v)
                      for k, v in detail.get("prices", {}).items()}
            loads = {tuple(map(int, k)): float(v)
                     for k, v in detail.get("loads", {}).items()}
        except Exception:
            pass

        moves = []
        for p in range(len(arcs)):
            a, b = arcs[p]
            for q in range(p + 1, len(arcs)):
                c, d = arcs[q]
                if a == d or c == b:
                    continue
                x, y = (a, d), (c, b)
                if x in current or y in current:
                    continue
                child = canon((current - {(a, b), (c, d)}) | {x, y})
                if child == parent:
                    continue
                # Positive prices identify links whose removal may reduce
                # congestion. Loads provide a tie-break when prices are sparse.
                relief = prices.get((a, b), 0.0) + prices.get((c, d), 0.0)
                relief -= prices.get(x, 0.0) + prices.get(y, 0.0)
                relief += 1e-8 * (loads.get((a, b), 0.0) + loads.get((c, d), 0.0)
                                  - loads.get(x, 0.0) - loads.get(y, 0.0))
                moves.append((relief, child))
        moves.sort(key=lambda item: (-item[0], energy(item[1])))
        return [child for _, child in moves]

    iteration = 0
    while tools.lp_calls_left > 0:
        parents = archive()
        if not parents:
            break

        # Rotate between the incumbent basin and alternate elite basins.
        if iteration % 4 == 0 or len(parents) == 1:
            parent = best_net
        elif rng.random() < 0.55:
            parent = parents[int(rng.integers(min(8, len(parents))))]
        else:
            parent = parents[int(rng.integers(len(parents)))]

        try:
            detail = tools.lp_detail(parent)
        except Exception:
            detail = {"prices": {}, "loads": {}}

        proposals = set()

        # A few explicitly dual-ranked neighbors provide local exploitation.
        for net in dual_neighbors(parent, detail)[:10]:
            proposals.add(net)

        # The learned edge model creates qualitatively different basins.
        # Increase its sample count after stagnation, but cap generation work.
        stale = min(8, iteration // 12)
        proposals.update(sample_model(parents, 34 + 8 * stale))

        # Short random walks perturb the chosen parent, preserving an avenue
        # for exploration when the elite model becomes too concentrated.
        for _ in range(8 + 2 * stale):
            child = kick(parent, 2 + int(rng.integers(5)))
            if child != parent:
                proposals.add(child)

        fresh = [
            net for net in proposals
            if net not in evaluated
            and tools.is_valid(net)
            and not tools.is_cached(net)
        ]
        if not fresh:
            # Fresh uniform-ish repaired samples are a fallback when a basin
            # has been exhausted or repairs collapse to previously seen nets.
            fresh = list(sample_model(parents, 100))
            fresh = [
                net for net in fresh
                if net not in evaluated and tools.is_valid(net)
                and not tools.is_cached(net)
            ]
        if not fresh:
            iteration += 1
            continue

        # Energy is a cheap filter; a distance-aware pass avoids spending a
        # whole batch on nearly identical repaired samples.
        fresh.sort(key=energy)
        batch = []
        for net in fresh:
            if all(len(edge_set(net) ^ edge_set(old)) >= 4 for old in batch):
                batch.append(net)
            if len(batch) >= min(8, int(tools.lp_calls_left)):
                break
        if len(batch) < min(8, int(tools.lp_calls_left)):
            selected = set(batch)
            for net in fresh:
                if net not in selected:
                    batch.append(net)
                    selected.add(net)
                if len(batch) >= min(8, int(tools.lp_calls_left)):
                    break

        for net in batch:
            if tools.lp_calls_left <= 0:
                break
            evaluate(net)
        iteration += 1

    return list(best_net) if best_net is not None else []
# EVOLVE-BLOCK-END


def run_postproc(pool_path, lp_budget, seed, control=False, time_limit=150):
    """Fixed entry point called by the evaluator: one run on one candidate pool."""
    import signal

    import oracle
    from hamiltonian import surrogate_terms

    pool = json.load(open(pool_path))
    import qoblib_tools as qt

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
