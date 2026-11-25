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
    """Search repaired starts with adaptive, bottleneck-aware local neighborhoods."""
    n = tools.n
    scored = {}
    best_net = None
    best_score = float("inf")
    stalls = {}
    operator_stats = {
        op: {"trials": 1, "wins": 0} for op in ("switch", "rotation", "kick")
    }

    def canon(arcs):
        return tuple(sorted(tuple(map(int, edge)) for edge in arcs))

    def edge_distance(a, b):
        return len(set(a) ^ set(b))

    def energy(net):
        try:
            return float(tools.energy(net))
        except Exception:
            return 0.0

    def evaluate(net):
        nonlocal best_net, best_score
        net = canon(net)
        if net not in scored:
            scored[net] = tools.lp(net)
        value = scored[net]
        if value < best_score:
            best_net, best_score = net, value
        return value

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

    # Rank by the pool's surrogate, but do not spend all initial LP calls on
    # near-identical repaired samples.
    ordered = sorted(repaired, key=energy)
    seeds = []
    for net in ordered:
        if all(edge_distance(net, old) >= 4 for old in seeds):
            seeds.append(net)
        if len(seeds) >= 24:
            break
    for net in ordered:
        if len(seeds) >= 32:
            break
        if net not in seeds:
            seeds.append(net)

    for net in seeds:
        if tools.lp_calls_left <= 0:
            break
        try:
            evaluate(net)
        except Exception as exc:
            if type(exc).__name__ == "BudgetExhausted":
                break
    if not scored:
        return list(seeds[0])

    def archive():
        ranked = sorted(scored, key=scored.get)
        out = []
        for net in ranked:
            if all(edge_distance(net, old) >= 4 for old in out):
                out.append(net)
            if len(out) >= 20:
                break
        for net in ranked:
            if len(out) >= 32:
                break
            if net not in out:
                out.append(net)
        return out

    def select_parent(pop):
        # Bias toward good networks, with under-explored parents receiving weight.
        top = pop[:min(8, len(pop))]
        if rng.random() < 0.58:
            choices = top
        else:
            choices = pop
        weights = [1.0 / (1.0 + 0.35 * stalls.get(x, 0)) for x in choices]
        point = float(rng.random()) * sum(weights)
        for net, weight in zip(choices, weights):
            point -= weight
            if point <= 0:
                return net
        return choices[-1]

    def two_switch(net, p, q):
        arcs = list(net)
        a, b = arcs[p]
        c, d = arcs[q]
        if a == d or c == b:
            return None
        current = set(net)
        if (a, d) in current or (c, b) in current:
            return None
        current.remove((a, b))
        current.remove((c, d))
        current.add((a, d))
        current.add((c, b))
        return tuple(sorted(current))

    def neighborhood(parent, detail):
        arcs = list(parent)
        current = set(parent)
        moves = {"switch": set(), "rotation": set(), "kick": set()}

        prices = detail.get("prices", {}) if detail else {}
        hot = {
            tuple(map(int, link))
            for link, price in prices.items()
            if float(price) > 0
        }
        if not hot and detail:
            loads = detail.get("loads", {})
            ranked_loads = sorted(
                ((float(value), tuple(map(int, link)))
                 for link, value in loads.items()),
                reverse=True,
            )
            hot = {link for _, link in ranked_loads[:6]}

        # Exhaustive critical-link switches are cheap at this problem size.
        for p in range(len(arcs)):
            for q in range(p + 1, len(arcs)):
                if hot and arcs[p] not in hot and arcs[q] not in hot:
                    continue
                move = two_switch(parent, p, q)
                if move is not None:
                    moves["switch"].add(move)

        # Independent stall histories control the effort spent on each operator.
        stall = stalls.get(parent, {})
        for op in ("switch", "rotation", "kick"):
            stall.setdefault(op, 0)

        def effort(op, base, cap):
            # Increase effort for operators that have repeatedly failed on this
            # parent, while preventing one neighborhood from dominating runtime.
            local = stall[op]
            global_stats = operator_stats[op]
            rate = (global_stats["wins"] + 1.0) / (global_stats["trials"] + 5.0)
            multiplier = 0.8 + min(1.0, 0.5 * local) + min(0.35, rate)
            return min(cap, max(base, int(base * multiplier)))

        for _ in range(effort("switch", 50, 120)):
            p = int(rng.integers(len(arcs)))
            q = int(rng.integers(len(arcs) - 1))
            if q >= p:
                q += 1
            move = two_switch(parent, p, q)
            if move is not None:
                moves["switch"].add(move)

        rotation_trials = effort("rotation", 40, 180)
        for _ in range(rotation_trials):
            ids = rng.choice(len(arcs), size=3, replace=False)
            (a, b), (c, d), (e, f) = (arcs[int(i)] for i in ids)
            old = {(a, b), (c, d), (e, f)}
            if hot and not (old & hot):
                continue
            for new in (
                {(a, d), (c, f), (e, b)},
                {(a, f), (c, b), (e, d)},
            ):
                if len(new) != 3 or any(x == y for x, y in new):
                    continue
                if new & (current - old):
                    continue
                moves["rotation"].add(tuple(sorted((current - old) | new)))

        kick_trials = effort("kick", 10, 48)
        for _ in range(kick_trials):
            cur = set(parent)
            for _ in range(2 + int(rng.integers(5))):
                now = list(cur)
                p, q = rng.choice(len(now), size=2, replace=False)
                a, b = now[int(p)]
                c, d = now[int(q)]
                if a == d or c == b or (a, d) in cur or (c, b) in cur:
                    continue
                cur.remove((a, b))
                cur.remove((c, d))
                cur.add((a, d))
                cur.add((c, b))
            if cur != current:
                moves["kick"].add(tuple(sorted(cur)))

        valid = {}
        for op, group in moves.items():
            valid[op] = [
                net for net in group
                if net != parent and net not in scored
                and not tools.is_cached(net) and tools.is_valid(net)
            ]
        return valid, hot

    def screen(groups, parent, hot, batch_size=8):
        fresh = [(op, net) for op, group in groups.items() for net in group]
        if not fresh:
            return []

        base_energy = energy(parent)
        ranked = []
        parent_edges = set(parent)
        for op, net in fresh:
            delta = energy(net) - base_energy
            changed_hot = len((parent_edges ^ set(net)) & hot) if hot else 0
            ranked.append((delta, -changed_hot, op, net))
        ranked.sort(key=lambda item: (item[0], item[1]))

        # Guarantee some attention to bottleneck-changing moves and spread
        # evaluations across different topologies and move operators.
        chosen = []
        for delta, neg_hot, op, net in ranked:
            if neg_hot < 0 and all(edge_distance(net, old) >= 2 for old in chosen):
                chosen.append(net)
            if len(chosen) >= min(3, batch_size):
                break

        for _, _, op, net in ranked:
            if len(chosen) >= min(6, batch_size):
                break
            if net not in chosen and all(edge_distance(net, old) >= 2 for old in chosen):
                chosen.append(net)

        for _, _, op, net in ranked:
            if len(chosen) >= batch_size:
                break
            if net not in chosen:
                chosen.append(net)
        return chosen

    rounds = 0
    while tools.lp_calls_left > 0:
        rounds += 1
        pop = archive()
        if not pop:
            break
        parent = select_parent(pop)

        try:
            detail = tools.lp_detail(parent)
        except Exception:
            detail = None
        groups, hot = neighborhood(parent, detail)
        batch = screen(groups, parent, hot)

        if not batch:
            alternatives = [x for x in pop if x != parent]
            if alternatives:
                parent = alternatives[int(rng.integers(len(alternatives)))]
                try:
                    detail = tools.lp_detail(parent)
                except Exception:
                    detail = None
                groups, hot = neighborhood(parent, detail)
                batch = screen(groups, parent, hot)

        if not batch:
            if rounds > 5000:
                break
            continue

        parent_score = scored.get(parent, float("inf"))
        best_batch = float("inf")
        success_by_op = set()
        for net in batch:
            if tools.lp_calls_left <= 0:
                break
            try:
                value = evaluate(net)
                best_batch = min(best_batch, value)
                if value < parent_score:
                    # Attribute improvement to the operator by symmetric
                    # difference size; direct switches change four arcs.
                    changed = edge_distance(net, parent)
                    op = "switch" if changed <= 4 else ("rotation" if changed <= 6 else "kick")
                    success_by_op.add(op)
            except Exception as exc:
                if type(exc).__name__ == "BudgetExhausted":
                    return list(best_net) if best_net is not None else []

        # Update per-parent and global operator statistics. A proposal batch
        # with no improvement increases only the relevant operator's stall count.
        local = stalls.setdefault(parent, {"switch": 0, "rotation": 0, "kick": 0})
        for op, group in groups.items():
            trials = min(len(group), len(batch))
            operator_stats[op]["trials"] += trials
            if op in success_by_op:
                operator_stats[op]["wins"] += 1
                local[op] = 0
            elif trials:
                local[op] = min(12, local.get(op, 0) + 1)

        if best_batch < parent_score:
            stalls[parent] = local
        else:
            stalls[parent] = local

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
