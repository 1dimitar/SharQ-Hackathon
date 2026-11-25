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
    n = tools.n
    seen = {}
    best_net = None
    best_score = float("inf")
    stalls = {}
    # Smoothed success rates for two-switch, rotation, and kick proposals.
    reward = [1.0, 1.0, 1.0]
    attempts = [1.0, 1.0, 1.0]

    def canon(arcs):
        return tuple(sorted(tuple(map(int, a)) for a in arcs))

    def evaluate(net):
        nonlocal best_net, best_score
        net = canon(net)
        if net not in seen:
            seen[net] = tools.lp(net)
        score = seen[net]
        if score < best_score:
            best_score, best_net = score, net
        return score

    def energy(net):
        try:
            return float(tools.energy(net))
        except Exception:
            return 0.0

    def dist(a, b):
        return len(set(a) ^ set(b))

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
        if all(dist(net, old) >= 4 for old in starts):
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
            evaluate(net)
        except Exception as exc:
            if type(exc).__name__ == "BudgetExhausted":
                break
    if not seen:
        return list(starts[0])

    def archive():
        ranked = sorted(seen, key=seen.get)
        out = []
        for net in ranked:
            if all(dist(net, old) >= 4 for old in out):
                out.append(net)
            if len(out) >= 18:
                break
        for net in ranked:
            if len(out) >= 30:
                break
            if net not in out:
                out.append(net)
        return out

    def select_parent(pop):
        if rng.random() < 0.65:
            return pop[int(rng.integers(min(5, len(pop))))]
        return pop[int(rng.integers(len(pop)))]

    def two_switch(net, rng):
        arcs = list(net)
        if len(arcs) < 2:
            return None
        p, q = rng.choice(len(arcs), size=2, replace=False)
        (a, b), (c, d) = arcs[int(p)], arcs[int(q)]
        cur = set(net)
        if a == d or c == b or (a, d) in cur or (c, b) in cur:
            return None
        cur.remove((a, b)); cur.remove((c, d))
        cur.add((a, d)); cur.add((c, b))
        return tuple(sorted(cur))

    def proposal(net, kind, hot):
        arcs = list(net)
        cur = set(net)
        if kind == 0:
            # Enumerate a random subset of swaps, favoring links with positive dual prices.
            pairs = []
            for i in range(len(arcs)):
                for j in range(i + 1, len(arcs)):
                    if hot and arcs[i] not in hot and arcs[j] not in hot:
                        continue
                    pairs.append((i, j))
            if not pairs:
                return two_switch(net, rng)
            rng.shuffle(pairs)
            for i, j in pairs[:12]:
                (a, b), (c, d) = arcs[i], arcs[j]
                if a == d or c == b or (a, d) in cur or (c, b) in cur:
                    continue
                nxt = cur.copy()
                nxt.remove((a, b)); nxt.remove((c, d))
                nxt.add((a, d)); nxt.add((c, b))
                return tuple(sorted(nxt))
            return None
        if kind == 1:
            if len(arcs) < 3:
                return None
            for _ in range(12):
                ids = rng.choice(len(arcs), size=3, replace=False)
                (a, b), (c, d), (e, f) = (arcs[int(i)] for i in ids)
                old = {(a, b), (c, d), (e, f)}
                if hot and not (old & hot):
                    continue
                for new in (
                    {(a, d), (c, f), (e, b)},
                    {(a, f), (c, b), (e, d)},
                ):
                    if any(x == y for x, y in new) or new & (cur - old):
                        continue
                    return tuple(sorted((cur - old) | new))
            return None
        # Multi-switch kick: preserve all degrees and make several valid exchanges.
        for _ in range(2 + int(rng.integers(5))):
            nxt = two_switch(tuple(sorted(cur)), rng)
            if nxt is not None:
                cur = set(nxt)
        return tuple(sorted(cur)) if cur != set(net) else None

    def matching_candidate():
        """Build two edge-disjoint randomized matchings biased by strong archive links."""
        strong = sorted(seen, key=seen.get)[:16]
        if not strong:
            return None
        frequency = {}
        for member in strong:
            for edge in member:
                frequency[edge] = frequency.get(edge, 0) + 1

        edges = set()

        def assign_matching():
            assigned = {}
            used_destinations = set()

            def search():
                if len(assigned) == n:
                    return True

                choices_by_source = []
                for source in range(n):
                    if source in assigned:
                        continue
                    options = [
                        dest for dest in range(n)
                        if dest != source
                        and dest not in used_destinations
                        and (source, dest) not in edges
                    ]
                    if not options:
                        return False
                    choices_by_source.append((len(options), source, options))

                min_count = min(item[0] for item in choices_by_source)
                tied = [item for item in choices_by_source if item[0] == min_count]
                _, source, options = tied[int(rng.integers(len(tied)))]

                # Exponential-race ordering favors links occurring often in the archive.
                options.sort(
                    key=lambda dest: rng.random() / (1.0 + 3.0 * frequency.get((source, dest), 0))
                )
                for dest in options:
                    assigned[source] = dest
                    used_destinations.add(dest)
                    edges.add((source, dest))
                    if search():
                        return True
                    edges.remove((source, dest))
                    used_destinations.remove(dest)
                    del assigned[source]
                return False

            return assigned if search() else None

        for _ in range(2):
            if assign_matching() is None:
                return None
        result = tuple(sorted(edges))
        return result if len(result) == 2 * n else None

    rounds = 0
    while tools.lp_calls_left > 0:
        rounds += 1
        pop = archive()
        parent = select_parent(pop)
        try:
            detail = tools.lp_detail(parent)
            prices = detail.get("prices", {})
            hot = {tuple(map(int, a)) for a, p in prices.items() if float(p) > 0}
            if not hot:
                loads = detail.get("loads", {})
                hot = {tuple(map(int, a)) for a, _ in
                       sorted(loads.items(), key=lambda x: float(x[1]), reverse=True)[:7]}
        except Exception:
            hot = set()

        # Adaptive operator probabilities, with a floor for exploration.
        rates = [reward[i] / attempts[i] for i in range(3)]
        probs = np.asarray(rates, dtype=float) + 0.25
        probs /= probs.sum()
        moves = {}
        kinds = {}
        # Generate a compact batch from a mixture; LP feedback updates its mix.
        for _ in range(36):
            kind = int(rng.choice(3, p=probs))
            net = proposal(parent, kind, hot)
            if net is None or net == parent or net in seen:
                continue
            try:
                if not tools.is_valid(net) or tools.is_cached(net):
                    continue
            except Exception:
                continue
            moves[net] = kind
            kinds[net] = kind

        # Reserve a small share of trials for archive-guided matching exploration.
        if rounds % 6 == 0:
            net = matching_candidate()
            if net is not None and net != parent and net not in seen:
                try:
                    if tools.is_valid(net) and not tools.is_cached(net):
                        moves[net] = 2
                except Exception:
                    pass

        if not moves:
            # Sample a few more unrestricted perturbations if focused moves stall.
            for _ in range(30):
                net = proposal(parent, int(rng.integers(3)), set())
                if net is not None and net != parent and net not in seen:
                    try:
                        if tools.is_valid(net) and not tools.is_cached(net):
                            moves[net] = 2
                    except Exception:
                        pass

        if not moves:
            if rounds > 5000:
                break
            continue

        base_e = energy(parent)
        pmap = {}
        try:
            pmap = {tuple(map(int, a)): float(v)
                    for a, v in detail.get("prices", {}).items()}
        except Exception:
            pass
        parent_edges = set(parent)
        ranked = []
        for net, kind in moves.items():
            added, removed = set(net) - parent_edges, parent_edges - set(net)
            dual_gain = sum(pmap.get(a, 0.0) for a in removed) - sum(
                pmap.get(a, 0.0) for a in added)
            ranked.append((energy(net) - base_e - 0.02 * dual_gain, net, kind))
        ranked.sort(key=lambda x: x[0])

        batch = []
        # Diverse high-ranked choices plus one deliberately exploratory choice.
        for _, net, _ in ranked:
            if all(dist(net, old) >= 2 for old in batch):
                batch.append(net)
            if len(batch) >= 3:
                break
        if len(ranked) > 3 and rng.random() < 0.35:
            tail = [x for x in ranked[3:] if x[1] not in batch]
            if tail:
                batch.append(tail[int(rng.integers(len(tail)))][1])

        before = best_score
        parent_score = seen.get(parent, float("inf"))
        for net in batch:
            if tools.lp_calls_left <= 0:
                break
            kind = moves.get(net, 2)
            try:
                score = evaluate(net)
                attempts[kind] += 1.0
                if score < parent_score:
                    reward[kind] += 1.0
            except Exception as exc:
                if type(exc).__name__ == "BudgetExhausted":
                    return list(best_net) if best_net is not None else []
        if best_score < before:
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
