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
    seen = {}
    best_net = None
    best_score = float("inf")
    archive = []

    def canon(arcs):
        return tuple(sorted(tuple(map(int, a)) for a in arcs))

    def distance(a, b):
        return len(set(a) ^ set(b))

    def energy(net):
        try:
            return float(tools.energy(net))
        except Exception:
            return 0.0

    def evaluate(net):
        nonlocal best_net, best_score
        net = canon(net)
        if net not in seen:
            seen[net] = tools.lp(net)
        score = seen[net]
        if score < best_score:
            best_net, best_score = net, score
        return score

    def update_archive(net):
        if net not in archive:
            archive.append(net)
        archive.sort(key=lambda x: seen.get(x, float("inf")))
        # Keep good solutions as well as representatives of distinct regions.
        keep = []
        for x in archive:
            if len(keep) < 12 and all(distance(x, y) >= 4 for y in keep):
                keep.append(x)
        for x in archive:
            if len(keep) >= 24:
                break
            if x not in keep:
                keep.append(x)
        archive[:] = keep

    # Repair all samples for free. Score a diverse set of promising repairs,
    # but reserve most of the LP budget for trajectory exploration.
    repaired = set()
    for cand in candidates:
        try:
            net = canon(tools.repair(cand))
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

    ranked = sorted(repaired, key=energy)
    starts = []
    for net in ranked:
        if all(distance(net, old) >= 4 for old in starts):
            starts.append(net)
        if len(starts) >= min(18, len(ranked)):
            break
    for net in ranked:
        if len(starts) >= min(28, len(ranked)):
            break
        if net not in starts:
            starts.append(net)

    for net in starts:
        if tools.lp_calls_left <= 0:
            break
        try:
            evaluate(net)
            update_archive(net)
        except Exception as exc:
            if type(exc).__name__ == "BudgetExhausted":
                break

    if best_net is None:
        return list(starts[0]) if starts else []

    def switch(cur, p, q):
        arcs = list(cur)
        (a, b), (c, d) = arcs[p], arcs[q]
        if a == d or c == b:
            return None
        s = set(cur)
        x, y = (a, d), (c, b)
        if x in s or y in s:
            return None
        s.remove((a, b))
        s.remove((c, d))
        s.add(x)
        s.add(y)
        return tuple(sorted(s))

    def propose(parent, hot, loads):
        arcs = list(parent)
        if len(arcs) < 2:
            return None

        # Select at least one edge with probability biased toward dual-positive
        # or heavily loaded links; retain a substantial uniform component.
        weights = []
        for a in arcs:
            w = 1.0
            if a in hot:
                w += 5.0
            if a in loads:
                w += 4.0 * max(0.0, loads[a]) / max(1.0, max_load)
            weights.append(w)

        if rng.random() < 0.72:
            p = int(rng.choice(len(arcs), p=[w / sum(weights) for w in weights]))
        else:
            p = int(rng.integers(len(arcs)))
        q = int(rng.integers(len(arcs) - 1))
        if q >= p:
            q += 1
        result = switch(parent, p, q)
        if result is not None:
            return result

        # Try a few uniformly selected switches if the focused one is blocked.
        for _ in range(8):
            p, q = rng.choice(len(arcs), size=2, replace=False)
            result = switch(parent, int(p), int(q))
            if result is not None:
                return result
        return None

    def recombine(a, b):
        # Keep shared edges and fill the remaining directed degree deficits
        # by repairing a shuffled union. repair is free and preserves as many
        # of the supplied links as possible.
        common = set(a) & set(b)
        union = list(set(a) | set(b))
        rng.shuffle(union)
        seed = list(common) + [x for x in union if x not in common]
        try:
            child = canon(tools.repair(seed))
            if tools.is_valid(child) and child != a and child != b:
                return child
        except Exception:
            pass
        return None

    current = best_net
    current_score = best_score
    step = 0
    max_load = 1.0

    while tools.lp_calls_left > 0:
        step += 1

        # Restart from a strong but not invariably best archived member.
        if step % 31 == 1 and archive:
            pool = archive[:min(10, len(archive))]
            current = pool[int(rng.integers(len(pool)))]
            current_score = seen[current]

        # Occasionally recombine distinct elite solutions to move to a new basin.
        proposal = None
        if len(archive) >= 2 and rng.random() < 0.10:
            i = int(rng.integers(len(archive)))
            j = int(rng.integers(len(archive) - 1))
            if j >= i:
                j += 1
            proposal = recombine(archive[i], archive[j])

        hot, loads = set(), {}
        try:
            detail = tools.lp_detail(current)
            prices = detail.get("prices", {})
            hot = {
                tuple(map(int, edge))
                for edge, price in prices.items()
                if float(price) > 0
            }
            loads = {
                tuple(map(int, edge)): float(value)
                for edge, value in detail.get("loads", {}).items()
            }
            max_load = max([1.0] + list(loads.values()))
        except Exception:
            pass

        if proposal is None:
            proposal = propose(current, hot, loads)
        if proposal is None or proposal == current or not tools.is_valid(proposal):
            continue
        if proposal in seen or tools.is_cached(proposal):
            continue

        try:
            score = evaluate(proposal)
        except Exception as exc:
            if type(exc).__name__ == "BudgetExhausted":
                break
            continue

        update_archive(proposal)

        # Cyclic simulated annealing: frequently exploit, periodically permit
        # uphill moves to cross barriers. Temperature scales with score size.
        phase = step % 180
        temp = max(35.0, best_score * (0.00005 + 0.00075 * phase / 180.0))
        delta = score - current_score
        if delta <= 0 or rng.random() < math.exp(-min(50.0, delta / temp)):
            current, current_score = proposal, score

        # Re-anchor occasionally to the global best if the walk has drifted far.
        if step % 97 == 0 and best_net is not None and rng.random() < 0.65:
            current, current_score = best_net, best_score

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
