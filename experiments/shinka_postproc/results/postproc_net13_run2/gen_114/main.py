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
    n = tools.n
    scored = {}
    best_net = None
    best_score = float("inf")

    def canon(arcs):
        return tuple(sorted(tuple(map(int, a)) for a in arcs))

    def energy(net):
        try:
            return float(tools.energy(net))
        except Exception:
            return 0.0

    def dist(a, b):
        return len(set(a) ^ set(b))

    def evaluate(net):
        nonlocal best_net, best_score
        net = canon(net)
        if net not in scored:
            scored[net] = tools.lp(net)
        if scored[net] < best_score:
            best_net, best_score = net, scored[net]
        return scored[net]

    # Repairing and ranking samples is free; choose starting points spread
    # across topology space instead of spending the initial budget on clones.
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
    starts = []
    for net in ordered:
        if all(dist(net, other) >= 4 for other in starts):
            starts.append(net)
        if len(starts) >= 24:
            break
    for net in ordered:
        if len(starts) >= 32:
            break
        if net not in starts:
            starts.append(net)

    for net in starts:
        if tools.lp_calls_left <= 0:
            break
        try:
            evaluate(net)
        except Exception as exc:
            if type(exc).__name__ == "BudgetExhausted":
                break
    if best_net is None:
        return list(starts[0]) if starts else []

    # Each chain retains its own basin and temperature.
    chain_count = min(6, len(scored))
    ranked = sorted(scored, key=scored.get)
    states = [ranked[i % len(ranked)] for i in range(chain_count)]
    temperatures = [0.0, 250.0, 700.0, 1600.0, 3500.0, 7000.0][:chain_count]
    steps = [0] * chain_count
    rejects = [0] * chain_count

    def switch(net, p, q):
        arcs = list(net)
        (a, b), (c, d) = arcs[p], arcs[q]
        if a == d or c == b:
            return None
        cur = set(net)
        x, y = (a, d), (c, b)
        if x in cur or y in cur:
            return None
        cur.remove((a, b))
        cur.remove((c, d))
        cur.add(x)
        cur.add(y)
        return tuple(sorted(cur))

    def propose(parent, detail, chain):
        arcs = list(parent)
        cur = set(parent)
        prices = detail.get("prices", {}) if detail else {}
        hot = {tuple(map(int, a)) for a, p in prices.items() if float(p) > 0}
        if not hot and detail:
            loads = detail.get("loads", {})
            hot = {
                a for _, a in sorted(
                    ((float(v), tuple(map(int, a))) for a, v in loads.items()),
                    reverse=True
                )[:6]
            }

        moves = set()
        # Hot-link switches are especially useful in cool, local chains.
        for p in range(len(arcs)):
            for q in range(p + 1, len(arcs)):
                if hot and arcs[p] not in hot and arcs[q] not in hot:
                    continue
                x = switch(parent, p, q)
                if x is not None:
                    moves.add(x)

        # Random switches ensure the chain can leave a region with misleading
        # or degenerate dual prices.
        trials = 18 + 10 * chain
        for _ in range(trials):
            p = int(rng.integers(len(arcs)))
            q = int(rng.integers(len(arcs) - 1))
            if q >= p:
                q += 1
            x = switch(parent, p, q)
            if x is not None:
                moves.add(x)

        # A rotation changes three destinations at once and can bridge barriers
        # that cannot be crossed by a single two-switch.
        for _ in range(12 + 5 * chain):
            ids = rng.choice(len(arcs), size=3, replace=False)
            (a, b), (c, d), (e, f) = (arcs[int(i)] for i in ids)
            old = {(a, b), (c, d), (e, f)}
            if hot and not (old & hot) and rng.random() < 0.8:
                continue
            for new in (
                {(a, d), (c, f), (e, b)},
                {(a, f), (c, b), (e, d)}
            ):
                if len(new) != 3 or any(u == v for u, v in new):
                    continue
                if new & (cur - old):
                    continue
                moves.add(tuple(sorted((cur - old) | new)))

        return [x for x in moves if x != parent and x not in scored
                and not tools.is_cached(x) and tools.is_valid(x)]

    round_no = 0
    while tools.lp_calls_left > 0:
        round_no += 1
        # Occasionally exchange states: a useful basin found by a hot chain
        # then becomes available to a cooler optimizer.
        if chain_count > 1 and round_no % 11 == 0:
            i = int(rng.integers(chain_count))
            j = int(rng.integers(chain_count - 1))
            if j >= i:
                j += 1
            si, sj = states[i], states[j]
            ti, tj = max(1.0, temperatures[i]), max(1.0, temperatures[j])
            exponent = (scored[si] - scored[sj]) * (1.0 / ti - 1.0 / tj)
            if exponent >= 0 or rng.random() < math.exp(max(-40.0, exponent)):
                states[i], states[j] = sj, si

        # Cycle through chains, with a mild preference for chains that have
        # recently found improvements.
        ci = int(rng.integers(chain_count))
        parent = states[ci]
        try:
            detail = tools.lp_detail(parent)
        except Exception:
            detail = None
        moves = propose(parent, detail, ci)
        if not moves:
            rejects[ci] += 1
            continue

        # The surrogate is used only to order proposals, not as a substitute
        # for the exact LP objective. Include a randomized choice from a broad
        # leading portion to guard against surrogate misspecification.
        base_e = energy(parent)
        moves.sort(key=lambda x: energy(x) - base_e)
        head = moves[:min(18, len(moves))]
        if rng.random() < 0.72:
            trial = head[int(rng.integers(len(head)))]
        else:
            trial = moves[int(rng.integers(len(moves)))]

        try:
            value = evaluate(trial)
        except Exception as exc:
            if type(exc).__name__ == "BudgetExhausted":
                break
            rejects[ci] += 1
            continue

        old_value = scored[parent]
        steps[ci] += 1
        # Slowly cool, but never eliminate uphill exploration in hot chains.
        temp = temperatures[ci] * (0.985 ** (steps[ci] // 25))
        if value <= old_value:
            states[ci] = trial
            rejects[ci] = 0
        elif temp > 0 and rng.random() < math.exp(
            max(-40.0, -(value - old_value) / temp)
        ):
            states[ci] = trial
            rejects[ci] = 0
        else:
            rejects[ci] += 1

        # A persistently stagnant chain is restarted from a good but distinct
        # archive member, rather than repeatedly sampling the same neighborhood.
        if rejects[ci] >= 35:
            archive = sorted(scored, key=scored.get)
            pool = archive[:min(20, len(archive))]
            if pool:
                states[ci] = pool[int(rng.integers(len(pool)))]
            rejects[ci] = 0

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
