"""Classical post-processing of QAOA output for QOBLIB 08. ShinkaEvolve edits the EVOLVE block."""

import json
import time

import numpy as np

# EVOLVE-BLOCK-START
import math


def improve(demand, candidates, tools, rng):
    """Repair candidates, then search with dual-guided switches and diverse restarts."""
    n = tools.n
    scores = {}
    details = {}
    archive = []
    best_net = None
    best_score = float("inf")

    def canon(arcs):
        return tuple(sorted(tuple(map(int, a)) for a in arcs))

    def distance(a, b):
        return len(set(a).symmetric_difference(b))

    def evaluate(net):
        nonlocal best_net, best_score
        net = canon(net)
        if net not in scores:
            if tools.lp_calls_left <= 0:
                return float("inf")
            try:
                scores[net] = float(tools.lp(net))
            except Exception as exc:
                if type(exc).__name__ == "BudgetExhausted":
                    return float("inf")
                raise
        value = scores[net]
        if value < best_score:
            best_net, best_score = net, value
        return value

    def remember(net):
        """Maintain a ranked elite plus topologically diverse alternatives."""
        net = canon(net)
        if net not in scores or not math.isfinite(scores[net]) or net in archive:
            return
        archive.append(net)
        archive.sort(key=lambda x: scores[x])
        if len(archive) <= 16:
            return

        kept = archive[:5]
        for other in archive[5:]:
            if len(kept) >= 16:
                break
            if all(distance(other, old) >= 6 for old in kept):
                kept.append(other)
        for other in archive:
            if len(kept) >= 16:
                break
            if other not in kept:
                kept.append(other)
        archive[:] = kept

    # Repair every raw sample for free, then score a mix of good-energy and
    # structurally distinct valid networks.
    repairs = set()
    for candidate in candidates:
        try:
            net = canon(tools.repair(candidate))
            if tools.is_valid(net):
                repairs.add(net)
        except Exception:
            continue

    if not repairs:
        try:
            net = canon(tools.repair([]))
            if tools.is_valid(net):
                repairs.add(net)
        except Exception:
            return []

    ordered = sorted(repairs, key=tools.energy)
    seeds = []
    for net in ordered:
        if all(distance(net, old) >= 6 for old in seeds):
            seeds.append(net)
        if len(seeds) >= 12:
            break
    for net in ordered:
        if len(seeds) >= 24:
            break
        if net not in seeds:
            seeds.append(net)

    for net in seeds:
        if tools.lp_calls_left <= 0:
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

    def make_moves(net, prices, kick):
        """Enumerate valid 2-switches and sample longer 3-edge rotations."""
        arcs = list(net)
        current = set(net)
        moves = set()

        # Exhaustive 2-switch neighborhood: each operation preserves all
        # in- and out-degrees, with the tests excluding loops and duplicate arcs.
        for i, (a, b) in enumerate(arcs):
            for c, d in arcs[i + 1:]:
                if a == c or a == d or c == b:
                    continue
                x, y = (a, d), (c, b)
                if x == y or x in current or y in current:
                    continue
                moves.add(canon((current - {(a, b), (c, d)}) | {x, y}))

        # Rotations extend the neighborhood beyond a single 2-switch.
        attempts = min(100, 12 + 5 * kick)
        hot_ids = [i for i, edge in enumerate(arcs) if prices.get(edge, 0.0) > 0]
        for _ in range(attempts):
            if len(arcs) < 3:
                break
            if hot_ids and rng.random() < 0.65:
                p = int(rng.choice(hot_ids))
                rest = [i for i in range(len(arcs)) if i != p]
                q, r = (int(i) for i in rng.choice(rest, size=2, replace=False))
                ids = (p, q, r)
            else:
                ids = tuple(int(i) for i in rng.choice(len(arcs), size=3, replace=False))

            (a, b), (c, d), (e, f) = (arcs[i] for i in ids)
            old = {(a, b), (c, d), (e, f)}
            new = {(a, d), (c, f), (e, b)}
            if len(new) != 3 or any(u == v for u, v in new):
                continue
            if new & (current - old):
                continue
            moves.add(canon((current - old) | new))

        moves.discard(net)
        return moves

    def add_path_relink_moves(parent, moves):
        """Add degree-preserving moves that install links from other elites."""
        parent_edges = set(parent)
        targets = [net for net in archive if net != parent]
        if not targets:
            return
        rng.shuffle(targets)
        for target in targets[:4]:
            missing = [edge for edge in target if edge not in parent_edges]
            rng.shuffle(missing)
            for a, b in missing[:10]:
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

    def choose_parent(stale):
        if len(archive) <= 1:
            return best_net
        if stale > 7 and rng.random() < 0.35:
            return archive[int(rng.integers(len(archive)))]
        top = archive[:min(8, len(archive))]
        weights = np.array([1.0 / (1.0 + 0.7 * i) for i in range(len(top))])
        # Periodically sample the broader archive to preserve basin diversity.
        if stale > 10 and len(archive) > len(top) and rng.random() < 0.35:
            return archive[int(rng.integers(len(top), len(archive)))]
        return top[int(rng.choice(len(top), p=weights / weights.sum()))]

    stale = 0
    kick = 1

    while tools.lp_calls_left > 0:
        parent = choose_parent(stale)
        parent_score = scores.get(parent, best_score)
        info = get_detail(parent)

        prices = {}
        for edge, value in info.get("prices", {}).items():
            try:
                prices[tuple(map(int, edge))] = float(value)
            except Exception:
                continue

        if not any(value > 0 for value in prices.values()):
            loads = {}
            for edge, value in info.get("loads", {}).items():
                try:
                    loads[tuple(map(int, edge))] = float(value)
                except Exception:
                    continue
            if loads:
                values = sorted(loads.values())
                threshold = values[int(0.70 * (len(values) - 1))]
                prices = {edge: float(value >= threshold) for edge, value in loads.items()}

        moves = make_moves(parent, prices, kick)
        add_path_relink_moves(parent, moves)
        fresh = [
            net for net in moves
            if net not in scores and not tools.is_cached(net)
        ]

        if not fresh:
            stale += 1
            kick = min(14, 1 + stale // 2)
            continue

        parent_edges = set(parent)

        def dual_gain(net):
            removed = parent_edges - set(net)
            added = set(net) - parent_edges
            return (sum(prices.get(edge, 0.0) for edge in removed)
                    - sum(prices.get(edge, 0.0) for edge in added))

        by_price = sorted(fresh, key=dual_gain, reverse=True)
        by_energy = sorted(fresh, key=tools.energy)

        # Spend each batch across dual signal, surrogate quality, and diversity.
        batch_limit = min(8, len(fresh))
        batch = []
        for net in by_price[:min(2, batch_limit)]:
            batch.append(net)

        for net in by_energy:
            if len(batch) >= batch_limit:
                break
            if net not in batch and all(distance(net, old) >= 4 for old in batch):
                batch.append(net)

        for net in by_energy:
            if len(batch) >= batch_limit:
                break
            if net not in batch:
                batch.append(net)

        # Occasionally spend one call on a move outside the deterministic rank.
        if len(fresh) > len(batch) and rng.random() < 0.25:
            remaining = [net for net in fresh if net not in batch]
            batch.append(remaining[int(rng.integers(len(remaining)))])

        previous_best = best_score
        parent_improved = False
        for net in batch:
            if tools.lp_calls_left <= 0:
                break
            value = evaluate(net)
            if not math.isfinite(value):
                continue
            remember(net)
            if value < parent_score:
                parent_improved = True
            if value <= parent_score * 1.02:
                remember(net)
            if value < previous_best:
                previous_best = value

        if parent_improved:
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
