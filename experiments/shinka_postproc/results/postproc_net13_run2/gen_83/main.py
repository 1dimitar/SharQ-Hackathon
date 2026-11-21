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
    best_net, best_score = None, float("inf")
    scored = {}
    uses = {}
    stalls = {}

    def canon(arcs):
        return tuple(sorted(tuple(map(int, a)) for a in arcs))

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
            scored[net] = float(tools.lp(net))
        if scored[net] < best_score:
            best_net, best_score = net, scored[net]
        return scored[net]

    # Build starting states from repaired samples; include diverse, not merely
    # lowest-energy, repairs since the QUBO ranking is only a weak proxy.
    pool = set()
    for cand in candidates:
        try:
            x = canon(tools.repair(cand))
            if tools.is_valid(x):
                pool.add(x)
        except Exception:
            pass
    if not pool:
        try:
            x = canon(tools.repair([]))
            if tools.is_valid(x):
                pool.add(x)
        except Exception:
            return []

    ordered = sorted(pool, key=energy)
    starts = []
    for x in ordered:
        if all(dist(x, y) >= 4 for y in starts):
            starts.append(x)
        if len(starts) >= 24:
            break
    for x in ordered:
        if len(starts) >= min(40, len(pool)):
            break
        if x not in starts:
            starts.append(x)

    for x in starts:
        if tools.lp_calls_left <= 0:
            break
        try:
            evaluate(x)
        except Exception as exc:
            if type(exc).__name__ == "BudgetExhausted":
                break
    if not scored:
        return list(starts[0])

    def two_switch(net, p, q):
        arcs = list(net)
        (a, b), (c, d) = arcs[p], arcs[q]
        if a == d or c == b:
            return None
        cur = set(net)
        if (a, d) in cur or (c, b) in cur:
            return None
        cur.remove((a, b))
        cur.remove((c, d))
        cur.add((a, d))
        cur.add((c, b))
        return tuple(sorted(cur))

    def propose(net, hot, temperature):
        arcs = list(net)
        current = set(net)
        # Mostly alter a bottleneck link, but keep some unbiased moves to
        # escape dual-degenerate neighborhoods.
        for _ in range(30):
            if rng.random() < 0.72 and hot:
                choices = [i for i, a in enumerate(arcs) if a in hot]
                p = int(choices[int(rng.integers(len(choices)))]) if choices else int(rng.integers(len(arcs)))
            else:
                p = int(rng.integers(len(arcs)))
            q = int(rng.integers(len(arcs) - 1))
            if q >= p:
                q += 1
            out = two_switch(net, p, q)
            if out is not None:
                return out

        # A three-edge cyclic reassignment provides a larger move when the
        # ordinary switch neighborhood is blocked.
        if len(arcs) >= 3:
            ids = rng.choice(len(arcs), size=3, replace=False)
            (a, b), (c, d), (e, f) = [arcs[int(i)] for i in ids]
            old = {(a, b), (c, d), (e, f)}
            if not hot or old & hot:
                new = {(a, d), (c, f), (e, b)}
                if (len(new) == 3 and all(x != y for x, y in new)
                        and not (new & (current - old))):
                    return tuple(sorted((current - old) | new))
        return net

    def choose_state():
        # Cost-aware archive sampling: rank remains important, while novelty
        # and low visitation counts keep less explored regions in play.
        ranked = sorted(scored, key=scored.get)[:100]
        if not ranked:
            return best_net
        weights = []
        for rank, x in enumerate(ranked):
            novelty = min((dist(x, y) for y in ranked if y != x), default=26) / 26.0
            rarity = 1.0 / math.sqrt(1 + uses.get(x, 0))
            weights.append((1.0 / (1.0 + 0.035 * rank)) * (0.45 + novelty) * rarity)
        r = float(rng.random()) * sum(weights)
        for x, w in zip(ranked, weights):
            r -= w
            if r <= 0:
                return x
        return ranked[-1]

    # Repeated annealing trajectories. Each accepted/rejected proposal consumes
    # one LP call, avoiding large batches of decisions made from stale parents.
    iteration = 0
    while tools.lp_calls_left > 0:
        iteration += 1
        if iteration % 55 == 1:
            current = choose_state()
            current_score = scored[current]
            temperature = max(250.0, current_score * (0.006 if iteration % 220 else 0.014))
            age = 0
        uses[current] = uses.get(current, 0) + 1

        try:
            detail = tools.lp_detail(current)
            prices = detail.get("prices", {})
            hot = {tuple(map(int, a)) for a, p in prices.items() if float(p) > 0}
            if not hot:
                loads = detail.get("loads", {})
                hot = {tuple(map(int, a)) for a, _ in sorted(
                    ((float(v), a) for a, v in loads.items()), reverse=True
                )[:5]}
        except Exception:
            hot = set()

        trial = propose(current, hot, temperature)
        if trial == current or not tools.is_valid(trial):
            age += 1
            continue
        if trial in scored or tools.is_cached(trial):
            # Cached results are useful too; avoid spending calls to rediscover them.
            if trial in scored:
                value = scored[trial]
            else:
                age += 1
                continue
        else:
            try:
                value = evaluate(trial)
            except Exception as exc:
                if type(exc).__name__ == "BudgetExhausted":
                    break
                age += 1
                continue

        delta = value - current_score
        temp = max(100.0, temperature * (0.985 ** age))
        if delta <= 0 or rng.random() < math.exp(-min(50.0, delta / temp)):
            current, current_score = trial, value
            age = 0
        else:
            age += 1

        # Occasionally restart from an archive region rather than allowing a
        # long unproductive trajectory to consume the remaining budget.
        if age > 18:
            current = choose_state()
            current_score = scored[current]
            age = 0

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
