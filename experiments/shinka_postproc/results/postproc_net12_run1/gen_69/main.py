"""Classical post-processing of QAOA output for QOBLIB 08. ShinkaEvolve edits the EVOLVE block."""

import json
import time

import numpy as np

# EVOLVE-BLOCK-START
import math


def improve(demand, candidates, tools, rng):
    n = int(tools.n)
    edge_count = 2 * n
    D = np.asarray(demand, dtype=float)
    population = {}
    energy_cache = {}
    best_net = None
    best_score = float("inf")

    def canon(arcs):
        return tuple(sorted(tuple(map(int, a)) for a in arcs))

    def energy(net):
        net = canon(net)
        if net not in energy_cache:
            try:
                energy_cache[net] = float(tools.energy(net))
            except Exception:
                energy_cache[net] = float("inf")
        return energy_cache[net]

    def evaluate(net):
        nonlocal best_net, best_score
        net = canon(net)
        if net in population or not tools.is_valid(net):
            return
        score = float(tools.lp(net))
        if not math.isfinite(score):
            return
        population[net] = (score, energy(net))
        if score < best_score:
            best_net, best_score = net, score

    def trim_population(limit=32):
        nonlocal population
        if len(population) <= limit:
            return
        ordered = sorted(population, key=lambda x: population[x][0])
        keep = ordered[:8]
        for net in ordered[8:]:
            if len(keep) >= limit:
                break
            if all(len(set(net) ^ set(old)) >= 4 for old in keep):
                keep.append(net)
        for net in ordered:
            if len(keep) >= limit:
                break
            if net not in keep:
                keep.append(net)
        population = {net: population[net] for net in keep}

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

    seeds = sorted(repaired, key=energy)
    # Include diverse low-energy repairs, then fill with the remaining best.
    chosen = []
    for net in seeds:
        if all(len(set(net) ^ set(old)) >= 4 for old in chosen):
            chosen.append(net)
        if len(chosen) >= 20:
            break
    for net in seeds:
        if len(chosen) >= 36:
            break
        if net not in chosen:
            chosen.append(net)

    for net in chosen:
        if tools.lp_calls_left <= 0:
            break
        evaluate(net)
    if not population:
        return list(chosen[0]) if chosen else []

    def switch(net, hot=None):
        arcs = list(net)
        cur = set(net)
        out = []
        for p in range(len(arcs)):
            a, b = arcs[p]
            for q in range(p + 1, len(arcs)):
                c, d = arcs[q]
                if hot is not None and arcs[p] not in hot and arcs[q] not in hot:
                    continue
                if a == d or c == b:
                    continue
                x, y = (a, d), (c, b)
                if x in cur or y in cur:
                    continue
                child = canon((cur - {arcs[p], arcs[q]}) | {x, y})
                if child != net:
                    out.append(child)
        return out

    def random_switch(net):
        arcs = list(net)
        cur = set(net)
        for _ in range(24):
            p, q = map(int, rng.choice(len(arcs), size=2, replace=False))
            a, b = arcs[p]
            c, d = arcs[q]
            x, y = (a, d), (c, b)
            if a != d and c != b and x not in cur and y not in cur:
                return canon((cur - {(a, b), (c, d)}) | {x, y})
        return net

    def mutate(net, count):
        cur = net
        for _ in range(count):
            nxt = random_switch(cur)
            if nxt != cur:
                cur = nxt
        return cur

    def crossover(a, b):
        sa, sb = set(a), set(b)
        common = sa & sb
        differing = list((sa | sb) - common)
        if not differing:
            return None
        rng.shuffle(differing)
        take = int(rng.integers(max(1, len(differing) // 3),
                                max(2, 2 * len(differing) // 3 + 1)))
        try:
            child = canon(tools.repair(list(common) + differing[:take]))
            return child if tools.is_valid(child) else None
        except Exception:
            return None

    def proxy(net):
        # Direct chosen-link demand is a lightweight secondary screening signal.
        return sum(float(D[a, b]) for a, b in net)

    iteration = 0
    while tools.lp_calls_left > 0 and population:
        trim_population()
        ranked = sorted(population, key=lambda x: population[x][0])
        if iteration % 4 == 3 and len(ranked) > 2:
            parent = ranked[int(rng.integers(min(12, len(ranked))))]
        else:
            parent = ranked[iteration % min(8, len(ranked))]

        try:
            detail = tools.lp_detail(parent)
            hot = {tuple(map(int, link)) for link, price
                   in detail.get("prices", {}).items() if float(price) > 0}
        except Exception:
            hot = set()

        proposals = set(switch(parent, hot if hot else None))
        # Unrestricted switches ensure that non-bottleneck links can still
        # participate in moves when dual information is sparse or misleading.
        if iteration % 2 == 0:
            proposals.update(switch(parent))

        if len(ranked) > 1 and iteration % 3 == 0:
            other = ranked[int(rng.integers(len(ranked)))]
            child = crossover(parent, other)
            if child is not None:
                proposals.add(child)
                proposals.add(mutate(child, 1 + int(rng.integers(4))))

        # Mutation jumps add diversity without the large offspring-generation
        # overhead of a full genetic-algorithm generation.
        for _ in range(12):
            child = mutate(parent, 2 + int(rng.integers(5)))
            if child != parent:
                proposals.add(child)
        if iteration % 4 == 1 and len(ranked) > 4:
            other = ranked[int(rng.integers(4, len(ranked)))]
            for _ in range(8):
                child = mutate(other, 3 + int(rng.integers(6)))
                if child != other:
                    proposals.add(child)

        fresh = [net for net in proposals
                 if net not in population and tools.is_valid(net)
                 and not tools.is_cached(net)]
        if not fresh:
            iteration += 1
            continue

        # Blend the Hamiltonian signal and a cheap demand proxy, then reserve a
        # few evaluations for exploration from lower-ranked proposals.
        fresh.sort(key=lambda net: (energy(net), proxy(net)))
        batch_size = min(8, int(tools.lp_calls_left), len(fresh))
        batch = fresh[:batch_size]
        if len(fresh) > batch_size and batch_size >= 3:
            reserve = min(2, batch_size // 3)
            tail = fresh[batch_size:]
            picks = rng.choice(len(tail), size=min(reserve, len(tail)),
                               replace=False)
            batch = fresh[:batch_size - len(picks)]
            batch.extend(tail[int(i)] for i in picks)

        for net in batch:
            if tools.lp_calls_left <= 0:
                break
            evaluate(net)
        iteration += 1

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
