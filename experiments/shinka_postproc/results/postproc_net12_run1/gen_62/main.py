"""Classical post-processing of QAOA output for QOBLIB 08. ShinkaEvolve edits the EVOLVE block."""

import json
import time

import numpy as np

# EVOLVE-BLOCK-START
import math


def improve(demand, candidates, tools, rng):
    """Search repaired topologies using dual-guided switches and diverse restarts."""
    n = tools.n

    def canon(arcs):
        return tuple(sorted(tuple(a) for a in arcs))

    scores = {}
    archive = []
    best_net = None
    best_score = float("inf")

    def evaluate(net):
        nonlocal best_net, best_score
        net = canon(net)
        if net not in scores:
            if tools.lp_calls_left <= 0:
                return float("inf")
            try:
                scores[net] = tools.lp(net)
            except Exception as exc:
                if type(exc).__name__ == "BudgetExhausted":
                    return float("inf")
                scores[net] = float("inf")
        value = scores[net]
        if value < best_score:
            best_net, best_score = net, value
        return value

    def remember(net):
        net = canon(net)
        value = scores.get(net, float("inf"))
        if not math.isfinite(value) or net in archive:
            return
        archive.append(net)
        archive.sort(key=lambda x: scores[x])
        # Preserve the elite while keeping topology-distinct alternatives.
        kept = archive[:5]
        for other in archive[5:]:
            if len(kept) >= 16:
                break
            if all(len(set(other) ^ set(x)) >= 6 for x in kept):
                kept.append(other)
        for other in archive:
            if len(kept) >= 16:
                break
            if other not in kept:
                kept.append(other)
        archive[:] = kept

    # Repairs are free; score a mixed set of low-energy and structurally
    # distinct repairs instead of consuming calls on many near-duplicates.
    repairs = set()
    for candidate in candidates:
        try:
            net = canon(tools.repair(candidate))
            if tools.is_valid(net):
                repairs.add(net)
        except Exception:
            pass
    if not repairs:
        try:
            repairs.add(canon(tools.repair([])))
        except Exception:
            return []

    ordered = sorted(repairs, key=tools.energy)
    seeds = []
    for net in ordered:
        if all(len(set(net) ^ set(old)) >= 6 for old in seeds):
            seeds.append(net)
        if len(seeds) >= 12:
            break
    for net in ordered:
        if len(seeds) >= 24:
            break
        if net not in seeds:
            seeds.append(net)

    # Hedge the QAOA-derived starts with two independently generated valid
    # topologies. A pair of distinct derangements gives every node degree 2
    # in both directions and excludes self-loops by construction.
    random_seeds = set()
    attempts = 0
    while len(random_seeds) < 2 and attempts < 100:
        attempts += 1
        permutations = []
        for _ in range(2):
            for _ in range(100):
                perm = [int(x) for x in rng.permutation(n)]
                if all(perm[i] != i for i in range(n)):
                    permutations.append(perm)
                    break
        if len(permutations) == 2:
            arcs = {(i, perm[i]) for perm in permutations for i in range(n)}
            net = canon(arcs)
            if len(arcs) == 2 * n and tools.is_valid(net) and net not in repairs:
                random_seeds.add(net)

    for net in seeds + list(random_seeds):
        if tools.lp_calls_left <= 0:
            break
        evaluate(net)
        remember(net)
    if best_net is None:
        return list(seeds[0]) if seeds else []

    details = {}

    def detail(net):
        if net not in details:
            try:
                details[net] = tools.lp_detail(net)
            except Exception:
                details[net] = {"prices": {}, "loads": {}}
        return details[net]

    def proposals(parent, prices, kick):
        """Generate valid 2-switches, plus a small sample of 3-rotations."""
        arcs = list(parent)
        current = set(parent)
        result = set()
        hot_ids = [i for i, a in enumerate(arcs) if prices.get(a, 0.0) > 0]

        # Enumerate all switches for n=12; bias generation is unnecessary as
        # ranking below uses the directional dual benefit.
        for p in range(len(arcs)):
            a, b = arcs[p]
            for q in range(p + 1, len(arcs)):
                c, d = arcs[q]
                if a == d or c == b or (a, d) in current or (c, b) in current:
                    continue
                result.add(canon((current - {(a, b), (c, d)}) |
                                 {(a, d), (c, b)}))

        # A few rotations provide jumps between basins not reachable by a
        # single switch. Increase their number as the search becomes stagnant.
        tries = min(100, 12 + 5 * kick)
        for _ in range(tries):
            if len(arcs) < 3:
                break
            ids = rng.choice(len(arcs), size=3, replace=False)
            (a, b), (c, d), (e, f) = (arcs[int(i)] for i in ids)
            old = {(a, b), (c, d), (e, f)}
            new = {(a, d), (c, f), (e, b)}
            if len(new) != 3 or any(x == y for x, y in new):
                continue
            if new & (current - old):
                continue
            result.add(canon((current - old) | new))

        result.discard(parent)
        return [x for x in result if tools.is_valid(x)]

    stale = 0
    kick = 1
    while tools.lp_calls_left > 0:
        # Usually explore the incumbent neighborhood; occasionally restart
        # from another high-quality archived basin.
        if len(archive) > 1 and rng.random() < min(0.65, 0.20 + stale * 0.025):
            choices = archive[:min(8, len(archive))]
            weights = [1.0 / (1.0 + i) for i in range(len(choices))]
            parent = choices[int(rng.choice(len(choices), p=[w / sum(weights) for w in weights]))]
        else:
            parent = best_net

        info = detail(parent)
        raw_prices = info.get("prices", {})
        prices = {tuple(a): float(v) for a, v in raw_prices.items()}
        if not any(v > 0 for v in prices.values()):
            loads = info.get("loads", {})
            if loads:
                threshold = sorted(loads.values())[int(0.70 * (len(loads) - 1))]
                prices = {tuple(a): float(v >= threshold) for a, v in loads.items()}

        moves = proposals(parent, prices, kick)
        fresh = [m for m in moves if m not in scores and not tools.is_cached(m)]
        if not fresh:
            stale += 1
            kick = min(14, 1 + stale // 2)
            continue

        edges = set(parent)

        def priority(net):
            changed_out = edges - set(net)
            changed_in = set(net) - edges
            gain = sum(prices.get(a, 0.0) for a in changed_out)
            gain -= sum(prices.get(a, 0.0) for a in changed_in)
            return gain

        # Reserve batch positions for dual-best moves; fill with surrogate
        # rankings and random alternatives to avoid trusting either heuristic.
        by_price = sorted(fresh, key=priority, reverse=True)
        by_energy = sorted(fresh, key=tools.energy)
        batch = []
        batch_limit = min(7, len(fresh))
        for m in by_price[:2]:
            batch.append(m)
        for m in by_energy:
            if len(batch) >= batch_limit:
                break
            if m not in batch and all(len(set(m) ^ set(x)) >= 4 for x in batch):
                batch.append(m)
        for m in by_energy:
            if len(batch) >= batch_limit:
                break
            if m not in batch:
                batch.append(m)
        if len(fresh) > len(batch) and rng.random() < 0.35:
            remaining = [m for m in fresh if m not in batch]
            batch.append(remaining[int(rng.integers(len(remaining)))])

        previous_best = best_score
        parent_score = scores.get(parent, best_score)
        for net in batch:
            if tools.lp_calls_left <= 0:
                break
            value = evaluate(net)
            if math.isfinite(value):
                if value < previous_best:
                    previous_best = value
                if value <= parent_score * 1.02 or value == best_score:
                    remember(net)

        if best_score < (scores.get(parent, float("inf")) - 1e-9):
            stale = 0
            kick = 1
        else:
            stale += 1
            if stale % 3 == 0:
                kick = min(14, kick + 1)

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
