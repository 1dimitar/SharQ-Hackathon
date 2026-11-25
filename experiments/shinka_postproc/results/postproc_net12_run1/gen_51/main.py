"""Classical post-processing of QAOA output for QOBLIB 08. ShinkaEvolve edits the EVOLVE block."""

import json
import time

import numpy as np

# EVOLVE-BLOCK-START
import math


def improve(demand, candidates, tools, rng):
    """Repair candidates, then search from a diverse archive of scored networks."""
    n = tools.n
    budget_end = lambda: tools.lp_calls_left <= 0

    def canon(arcs):
        return tuple(sorted(tuple(a) for a in arcs))

    scores = {}
    details = {}
    best_net = None
    best_score = float("inf")
    archive = []

    def evaluate(net):
        nonlocal best_net, best_score
        net = canon(net)
        if net not in scores:
            if budget_end():
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
        """Keep high-quality networks while retaining topological variety."""
        net = canon(net)
        if net not in scores or not math.isfinite(scores[net]):
            return
        if net in archive:
            return
        # Reserve places for diverse candidates rather than making the archive
        # a collection of nearly identical low-score networks.
        if len(archive) < 18:
            archive.append(net)
            return
        ranked = sorted(archive, key=lambda x: scores[x])
        quality_cut = ranked[min(7, len(ranked) - 1)]
        if scores[net] < scores[quality_cut]:
            archive.append(net)
        else:
            # Replace a similar, worse member only when the newcomer is better.
            similar = [
                old for old in archive
                if len(set(old) ^ set(net)) < 8
            ]
            if similar:
                victim = max(similar, key=lambda x: scores[x])
                if scores[net] < scores[victim]:
                    archive.remove(victim)
                    archive.append(net)
                else:
                    return
            else:
                archive.append(net)

        archive.sort(key=lambda x: scores[x])
        # Keep the elite plus diverse alternatives.
        kept = archive[:5]
        for net0 in archive[5:]:
            if len(kept) >= 18:
                break
            if all(len(set(net0) ^ set(x)) >= 6 for x in kept):
                kept.append(net0)
        for net0 in archive:
            if len(kept) >= 18:
                break
            if net0 not in kept:
                kept.append(net0)
        archive[:] = kept

    # Seed from repaired candidates ordered by the free surrogate. Diverse
    # repairs are considered first; all remaining budget goes to search.
    repairs = set()
    for cand in candidates:
        try:
            net = canon(tools.repair(cand))
            if tools.is_valid(net):
                repairs.add(net)
        except Exception:
            continue
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
        if len(seeds) >= 14:
            break
    for net in ordered:
        if len(seeds) >= 24:
            break
        if net not in seeds:
            seeds.append(net)

    for net in seeds:
        if budget_end():
            break
        value = evaluate(net)
        if math.isfinite(value):
            remember(net)

    if best_net is None:
        return list(seeds[0]) if seeds else []

    def get_detail(net):
        net = canon(net)
        if net not in details:
            try:
                details[net] = tools.lp_detail(net)
            except Exception:
                details[net] = {"prices": {}, "loads": {}}
        return details[net]

    def make_moves(net, hot, kick_size):
        """Construct valid degree-preserving switches and destination rotations."""
        arcs = list(net)
        current = set(net)
        moves = set()

        # Sample exchanges, biasing one selected edge toward a priced link.
        attempts = max(100, 35 * kick_size)
        for _ in range(attempts):
            if not arcs:
                break
            if hot and rng.random() < 0.72:
                hot_ids = [i for i, arc in enumerate(arcs) if arc in hot]
                p = int(rng.choice(hot_ids)) if hot_ids else int(rng.integers(len(arcs)))
            else:
                p = int(rng.integers(len(arcs)))
            q = int(rng.integers(len(arcs) - 1))
            if q >= p:
                q += 1
            a, b = arcs[p]
            c, d = arcs[q]
            # Swap destinations: out- and in-degrees are preserved.
            if a == d or c == b or (a, d) in current or (c, b) in current:
                continue
            moves.add(canon((current - {(a, b), (c, d)}) | {(a, d), (c, b)}))

        # Three-edge rotations allow moves that cannot be obtained in one switch.
        rotations = max(10, 8 * kick_size)
        for _ in range(rotations):
            if len(arcs) < 3:
                break
            ids = rng.choice(len(arcs), size=3, replace=False)
            (a, b), (c, d), (e, f) = (arcs[int(i)] for i in ids)
            old = {(a, b), (c, d), (e, f)}
            new = {(a, d), (c, f), (e, b)}
            if hot and not (old & hot):
                continue
            if len(new) != 3 or any(x == y for x, y in new):
                continue
            if new & (current - old):
                continue
            moves.add(canon((current - old) | new))

        moves.discard(net)
        return [m for m in moves if tools.is_valid(m)]

    def choose_parent(stale):
        # Favor good networks, but use a diversity-based archive member often
        # enough to avoid repeatedly searching only around the incumbent.
        if len(archive) == 1 or rng.random() < 0.38:
            return best_net
        top = archive[:min(6, len(archive))]
        weights = np.array([1.0 / (1.0 + i * 0.7) for i in range(len(top))])
        if stale > 8 and len(archive) > len(top):
            extra = archive[len(top):]
            if rng.random() < 0.5:
                return extra[int(rng.integers(len(extra)))]
        return top[int(rng.choice(len(top), p=weights / weights.sum()))]

    stale = 0
    kick_size = 2
    parent_failures = {}
    iteration = 0
    while not budget_end():
        iteration += 1
        parent = choose_parent(stale)
        parent_score = scores.get(parent, best_score)
        detail = get_detail(parent)
        prices = detail.get("prices", {})
        hot = {tuple(a) for a, price in prices.items() if price > 0}

        if not hot:
            loads = detail.get("loads", {})
            if loads:
                values = sorted(loads.values())
                threshold = values[int(0.70 * (len(values) - 1))]
                hot = {tuple(a) for a, value in loads.items() if value >= threshold}

        # Escalate the rotation effort only for parents whose own neighborhoods
        # have repeatedly failed, rather than penalizing every basin globally.
        parent_kick = min(20, kick_size + min(12, parent_failures.get(parent, 0) // 2))
        moves = set(make_moves(parent, hot, parent_kick))

        # Path-relink toward other strong archive members by installing one
        # of their missing links with a degree-preserving two-switch.
        parent_edges = set(parent)
        targets = [net for net in archive if net != parent]
        if targets:
            rng.shuffle(targets)
            for target in targets[:4]:
                missing = [edge for edge in target if edge not in parent_edges]
                rng.shuffle(missing)
                for a, b in missing[:12]:
                    for x in range(n):
                        if (a, x) not in parent_edges:
                            continue
                        for y in range(n):
                            if (y, b) not in parent_edges:
                                continue
                            old = {(a, x), (y, b)}
                            new = {(a, b), (y, x)}
                            if len(old) != 2 or len(new) != 2:
                                continue
                            if any(u == v for u, v in new):
                                continue
                            if new & (parent_edges - old):
                                continue
                            proposal = canon((parent_edges - old) | new)
                            if proposal != parent:
                                moves.add(proposal)
        moves = [m for m in moves if tools.is_valid(m)]

        fresh = [
            m for m in moves
            if m not in scores and not tools.is_cached(m)
        ]
        if not fresh and hot:
            fresh = [
                m for m in make_moves(parent, set(), parent_kick)
                if m not in scores and not tools.is_cached(m)
            ]

        if not fresh:
            stale += 1
            kick_size = min(12, 2 + stale // 2)
            continue

        # A positive dual price marks a link whose load contributes to the
        # current congestion objective. Prefer moves that remove priced links.
        price_map = {tuple(a): float(p) for a, p in prices.items()}
        parent_edges = set(parent)

        def move_priority(net):
            removed = parent_edges - set(net)
            added = set(net) - parent_edges
            price_gain = sum(price_map.get(a, 0.0) for a in removed)
            price_gain -= sum(price_map.get(a, 0.0) for a in added)
            return price_gain, float(tools.energy(net))

        batch_size = min(9, len(fresh))
        ranked_energy = sorted(fresh, key=tools.energy)
        ranked_price = sorted(fresh, key=lambda net: move_priority(net)[0], reverse=True)

        # Reserve some calls for the strongest directional dual signals, then
        # diversify the remainder while keeping the surrogate as a useful guide.
        batch = []
        price_slots = min(3, batch_size)
        for net in ranked_price:
            if net not in batch:
                batch.append(net)
            if len(batch) >= price_slots:
                break
        for net in ranked_energy:
            if len(batch) >= batch_size:
                break
            if all(len(set(net) ^ set(old)) >= 4 for old in batch):
                batch.append(net)
        for net in ranked_energy:
            if len(batch) >= batch_size:
                break
            if net not in batch:
                batch.append(net)

        # Compare against the incumbent from before this batch: evaluate()
        # updates best_score itself, so checking afterward would miss progress.
        previous_best = best_score
        improved = False
        parent_improved = False
        for net in batch:
            if budget_end():
                break
            value = evaluate(net)
            if math.isfinite(value):
                remember(net)
                if value < parent_score:
                    parent_improved = True
                if value < previous_best:
                    improved = True
                    previous_best = value
                # Also retain promising basin members even when not global best.
                if value <= parent_score * 1.015:
                    remember(net)

        if parent_improved:
            parent_failures[parent] = 0
        else:
            parent_failures[parent] = parent_failures.get(parent, 0) + 1

        if improved:
            stale = 0
            kick_size = 2
        else:
            stale += 1
            if stale % 3 == 0:
                kick_size = min(12, kick_size + 1)

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
