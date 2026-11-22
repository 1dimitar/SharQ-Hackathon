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
    """Search a diverse frontier, adapting neighborhood effort to observed progress."""
    n = tools.n
    scored = {}
    best_net = None
    best_score = float("inf")
    stalls = {}
    op_trials = {"switch": 1, "rotation": 1, "kick": 1}
    op_wins = {"switch": 0, "rotation": 0, "kick": 0}

    def canon(arcs):
        return tuple(sorted(tuple(map(int, edge)) for edge in arcs))

    def dist(a, b):
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
            value = tools.lp(net)
            scored[net] = value
        value = scored[net]
        if value < best_score:
            best_net, best_score = net, value
        return value

    # Repair is free; score a small, diverse set of promising seeds first.
    repaired = set()
    for candidate in candidates:
        try:
            net = canon(tools.repair(candidate))
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
        if all(dist(net, other) >= 4 for other in seeds):
            seeds.append(net)
        if len(seeds) >= min(18, len(ordered)):
            break
    for net in ordered:
        if len(seeds) >= min(24, len(ordered)):
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

    def frontier():
        ranked = sorted(scored, key=scored.get)
        out = []
        for net in ranked:
            if all(dist(net, old) >= 4 for old in out):
                out.append(net)
            if len(out) >= 16:
                break
        for net in ranked:
            if len(out) >= 28:
                break
            if net not in out:
                out.append(net)
        return out

    def choose_parent(pop):
        # Favor quality but occasionally revisit underexplored frontier regions.
        weights = []
        for net in pop:
            quality = 1.0 / (1.0 + max(0.0, scored[net] - best_score) / max(1.0, best_score))
            exploration = 1.0 / (1.0 + stalls.get(net, 0))
            weights.append(quality * (0.35 + 0.65 * exploration))
        total = sum(weights)
        r = float(rng.random()) * total
        for net, weight in zip(pop, weights):
            r -= weight
            if r <= 0:
                return net
        return pop[-1]

    def do_switch(current, e1, e2):
        a, b = e1
        c, d = e2
        if a == d or c == b:
            return None
        if (a, d) in current or (c, b) in current:
            return None
        return (current - {e1, e2}) | {(a, d), (c, b)}

    def make_proposals(parent, detail):
        current = set(parent)
        arcs = list(parent)
        hot = {
            tuple(map(int, edge))
            for edge, price in (detail.get("prices", {}) if detail else {}).items()
            if price > 0
        }
        if not hot and detail:
            loads = detail.get("loads", {})
            hot = {
                edge for _, edge in sorted(
                    ((float(v), tuple(map(int, e))) for e, v in loads.items()),
                    reverse=True,
                )[:6]
            }

        # Operator allocation adapts gradually, retaining a baseline for every type.
        rates = {
            op: (op_wins[op] + 1.0) / (op_trials[op] + 5.0)
            for op in op_trials
        }
        avg = sum(rates.values()) / 3.0
        factor = {
            op: min(1.7, max(0.65, rates[op] / max(avg, 1e-9)))
            for op in rates
        }

        proposals = {}
        # Enumerate all valid switches that touch an active bottleneck.
        for i in range(len(arcs)):
            for j in range(i + 1, len(arcs)):
                if hot and arcs[i] not in hot and arcs[j] not in hot:
                    continue
                changed = do_switch(current, arcs[i], arcs[j])
                if changed:
                    net = tuple(sorted(changed))
                    if net != parent:
                        proposals[net] = "switch"

        # Sample unrestricted switches to escape misleading or degenerate duals.
        for _ in range(int(55 * factor["switch"])):
            i = int(rng.integers(len(arcs)))
            j = int(rng.integers(len(arcs) - 1))
            if j >= i:
                j += 1
            changed = do_switch(current, arcs[i], arcs[j])
            if changed:
                net = tuple(sorted(changed))
                if net != parent:
                    proposals[net] = "switch"

        # Three-edge rotations preserve degree while changing a larger structure.
        stalls_here = stalls.get(parent, 0)
        rotation_count = min(280, int((55 + 18 * min(stalls_here, 8)) * factor["rotation"]))
        for _ in range(rotation_count):
            ids = rng.choice(len(arcs), size=3, replace=False)
            e1, e2, e3 = (arcs[int(i)] for i in ids)
            if hot and not ({e1, e2, e3} & hot):
                continue
            a, b = e1
            c, d = e2
            e, f = e3
            for additions in (
                {(a, d), (c, f), (e, b)},
                {(a, f), (c, b), (e, d)},
            ):
                old = {e1, e2, e3}
                if any(x == y for x, y in additions):
                    continue
                if len(additions) != 3 or additions & (current - old):
                    continue
                net = tuple(sorted((current - old) | additions))
                if net != parent:
                    proposals[net] = "rotation"

        # Short random walks act as basin escapes; only their endpoint is proposed.
        kick_count = int((10 + min(stalls_here, 8)) * factor["kick"])
        for _ in range(kick_count):
            walk = set(parent)
            for _ in range(2 + int(rng.integers(4))):
                now = list(walk)
                i, j = rng.choice(len(now), size=2, replace=False)
                changed = do_switch(walk, now[int(i)], now[int(j)])
                if changed:
                    walk = changed
            net = tuple(sorted(walk))
            if net != parent:
                proposals[net] = "kick"

        return proposals

    def screen(proposals, parent, detail, batch_size):
        fresh = [
            net for net in proposals
            if net not in scored and not tools.is_cached(net) and tools.is_valid(net)
        ]
        if not fresh:
            return []

        base_energy = energy(parent)
        hot = {
            tuple(map(int, edge))
            for edge, price in (detail.get("prices", {}) if detail else {}).items()
            if price > 0
        }
        ranked = sorted(
            ((energy(net) - base_energy, net) for net in fresh),
            key=lambda item: item[0],
        )
        selected = []
        focused = []
        if hot:
            pset = set(parent)
            focused = sorted(
                (
                    (delta, -len((pset ^ set(net)) & hot), net)
                    for delta, net in ranked
                    if (pset ^ set(net)) & hot
                ),
                key=lambda item: (item[0], item[1]),
            )
        for _, _, net in focused:
            if all(dist(net, old) >= 2 for old in selected):
                selected.append(net)
            if len(selected) >= max(1, batch_size // 3):
                break

        # Prefer energy-ranked proposals but diversify the first part of the batch.
        for _, net in ranked:
            if len(selected) >= min(5, batch_size):
                break
            if net not in selected and all(dist(net, old) >= 2 for old in selected):
                selected.append(net)
        for _, net in ranked:
            if len(selected) >= batch_size:
                break
            if net not in selected:
                selected.append(net)
        return selected

    rounds = 0
    while tools.lp_calls_left > 0:
        rounds += 1
        pop = frontier()
        if not pop:
            break
        parent = choose_parent(pop)
        try:
            detail = tools.lp_detail(parent)
        except Exception:
            detail = None

        proposals = make_proposals(parent, detail)
        stall = stalls.get(parent, 0)
        batch_size = min(12, 5 + stall // 3)
        batch = screen(proposals, parent, detail, batch_size)

        # When the selected basin is exhausted, try other frontier regions.
        if not batch:
            alternatives = [net for net in pop if net != parent]
            rng.shuffle(alternatives)
            for alternative in alternatives[:5]:
                try:
                    alt_detail = tools.lp_detail(alternative)
                except Exception:
                    alt_detail = None
                alt_props = make_proposals(alternative, alt_detail)
                batch = screen(alt_props, alternative, alt_detail, batch_size)
                if batch:
                    parent, detail = alternative, alt_detail
                    break

        if not batch:
            if rounds > 10000:
                break
            continue

        parent_score = scored.get(parent, float("inf"))
        batch_best = float("inf")
        for net in batch:
            if tools.lp_calls_left <= 0:
                break
            op = proposals.get(net, "kick")
            try:
                value = evaluate(net)
            except Exception as exc:
                if type(exc).__name__ == "BudgetExhausted":
                    return list(best_net) if best_net is not None else []
                continue
            batch_best = min(batch_best, value)
            changed = dist(net, parent)
            kind = "switch" if changed <= 4 else ("rotation" if changed <= 6 else op)
            op_trials[kind] += 1
            if value < parent_score:
                op_wins[kind] += 1

        if batch_best < parent_score:
            stalls[parent] = 0
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
