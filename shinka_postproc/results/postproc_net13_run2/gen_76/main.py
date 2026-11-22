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

    def canon(arcs):
        return tuple(sorted(tuple(map(int, e)) for e in arcs))

    def energy(net):
        try:
            return float(tools.energy(net))
        except Exception:
            return 0.0

    def dist(a, b):
        return len(set(a) ^ set(b))

    def score(net):
        nonlocal best_net, best_score
        net = canon(net)
        if net not in seen:
            seen[net] = tools.lp(net)
        if seen[net] < best_score:
            best_net, best_score = net, seen[net]
        return seen[net]

    # Repair and select starts by a combination of surrogate rank and diversity.
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
    starts = []
    for net in ordered:
        if all(dist(net, x) >= 4 for x in starts):
            starts.append(net)
        if len(starts) >= 24:
            break
    for net in ordered:
        if len(starts) >= 30:
            break
        if net not in starts:
            starts.append(net)

    for net in starts:
        if tools.lp_calls_left <= 0:
            break
        try:
            score(net)
        except Exception as exc:
            if type(exc).__name__ == "BudgetExhausted":
                break

    if best_net is None:
        return list(starts[0])

    def switch(cur, e1, e2):
        a, b = e1
        c, d = e2
        if a == d or c == b or (a, d) in cur or (c, b) in cur:
            return None
        out = set(cur)
        out.remove(e1)
        out.remove(e2)
        out.add((a, d))
        out.add((c, b))
        ans = tuple(sorted(out))
        return ans if tools.is_valid(ans) else None

    def archive():
        ranked = sorted(seen, key=seen.get)
        out = []
        for net in ranked:
            if all(dist(net, x) >= 4 for x in out):
                out.append(net)
            if len(out) >= 18:
                break
        for net in ranked:
            if len(out) >= 30:
                break
            if net not in out:
                out.append(net)
        return out

    def propose(parent, guide, detail):
        cur = set(parent)
        guide_edges = set(guide)
        hot = set()
        if detail:
            try:
                hot = {tuple(map(int, e)) for e, p in detail.get("prices", {}).items()
                       if float(p) > 0}
            except Exception:
                hot = set()

        # Prefer replacing a parent edge absent from the guide with a guide edge.
        proposals = set()
        missing = list(guide_edges - cur)
        rng.shuffle(missing)
        for add in missing[:16]:
            a, b = add
            # Outgoing and incoming degrees are preserved by a directed 2-switch.
            for old in list(cur):
                c, d = old
                if old in guide_edges:
                    continue
                if (a, d) in cur or (c, b) in cur or a == d or c == b:
                    continue
                new = set(cur)
                new.remove(old)
                new.remove((a, b)) if (a, b) in new else None
                # Pair add-edge's source/destination with an existing edge.
                # Reconnect (a,b),(c,d) -> (a,d),(c,b).
                if (a, b) not in cur:
                    continue
                new = set(cur)
                new.remove((a, b))
                new.remove(old)
                new.add((a, d))
                new.add((c, b))
                cand = tuple(sorted(new))
                if cand != parent and tools.is_valid(cand):
                    proposals.add(cand)

        # Bottleneck-oriented switches plus a random tail for diversification.
        arcs = list(cur)
        indices = [i for i, e in enumerate(arcs) if not hot or e in hot]
        if not indices:
            indices = list(range(len(arcs)))
        for _ in range(90):
            p = int(rng.choice(indices))
            q = int(rng.integers(len(arcs)))
            if p == q:
                continue
            cand = switch(cur, arcs[p], arcs[q])
            if cand is not None and cand != parent:
                proposals.add(cand)

        # A few longer jumps avoid restricting exploration to local exchanges.
        for _ in range(18):
            tmp = set(cur)
            for _ in range(2 + int(rng.integers(4))):
                aa = list(tmp)
                i, j = rng.choice(len(aa), size=2, replace=False)
                x, y = aa[int(i)]
                z, w = aa[int(j)]
                if x == w or z == y or (x, w) in tmp or (z, y) in tmp:
                    continue
                tmp.remove((x, y))
                tmp.remove((z, w))
                tmp.add((x, w))
                tmp.add((z, y))
            cand = tuple(sorted(tmp))
            if cand != parent and tools.is_valid(cand):
                proposals.add(cand)
        return proposals

    iteration = 0
    temperature = 0.025
    while tools.lp_calls_left > 0:
        pop = archive()
        if not pop:
            break
        # Alternate elite and exploratory parent selection.
        if rng.random() < 0.55:
            parent = pop[int(rng.integers(min(6, len(pop))))]
        else:
            parent = pop[int(rng.integers(len(pop)))]
        guides = [x for x in pop if x != parent]
        guide = guides[int(rng.integers(len(guides)))] if guides else parent

        try:
            detail = tools.lp_detail(parent)
        except Exception:
            detail = None

        moves = [m for m in propose(parent, guide, detail)
                 if m not in seen and not tools.is_cached(m)]
        if not moves:
            continue

        base_e = energy(parent)
        moves.sort(key=lambda m: energy(m) - base_e)
        # Mostly take promising surrogate moves, but retain diversity and a novel slot.
        shortlist = moves[:min(len(moves), 45)]
        selected = []
        if shortlist:
            selected.append(shortlist[0])
        while len(selected) < min(4, len(shortlist)):
            eligible = [m for m in shortlist if m not in selected]
            if not eligible:
                break
            pick = max(eligible, key=lambda m: (
                min(dist(m, s) for s in selected),
                -0.00001 * (energy(m) - base_e)))
            selected.append(pick)
        tail = [m for m in moves if m not in selected]
        if tail and len(selected) < 5:
            selected.append(tail[int(rng.integers(len(tail)))])

        old_best = best_score
        for net in selected:
            if tools.lp_calls_left <= 0:
                break
            try:
                score(net)
            except Exception as exc:
                if type(exc).__name__ == "BudgetExhausted":
                    return list(best_net) if best_net is not None else []
                continue

        # Annealing controls whether the next relinking walk remains near its parent.
        if best_score < old_best:
            temperature = max(0.004, temperature * 0.97)
        else:
            temperature = min(0.08, temperature * 1.003)
        iteration += 1

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
