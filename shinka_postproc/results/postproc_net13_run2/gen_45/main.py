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
    valid_pool = set()

    def canon(arcs):
        return tuple(sorted((int(a), int(b)) for a, b in arcs))

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

    # Repair the full sample pool, retaining its useful structural diversity.
    for candidate in candidates:
        try:
            net = canon(tools.repair(candidate))
            if tools.is_valid(net):
                valid_pool.add(net)
        except Exception:
            pass
    if not valid_pool:
        try:
            net = canon(tools.repair([]))
            if tools.is_valid(net):
                valid_pool.add(net)
        except Exception:
            return []

    # Score an energy-ordered, spread-out subset rather than spending calls on
    # many nearly identical repairs.
    ordered = sorted(valid_pool, key=energy)
    seeds = []
    for net in ordered:
        if all(dist(net, old) >= 4 for old in seeds):
            seeds.append(net)
        if len(seeds) >= min(36, max(1, tools.lp_calls_left)):
            break
    for net in ordered:
        if len(seeds) >= min(36, max(1, tools.lp_calls_left)):
            break
        if net not in seeds:
            seeds.append(net)

    for net in seeds:
        if tools.lp_calls_left <= 0:
            break
        try:
            evaluate(net)
        except Exception:
            break
    if best_net is None:
        return list(seeds[0])

    def population():
        ranked = sorted(seen, key=seen.get)
        # Mix quality with diversity: retain distinct topologies before filling
        # remaining slots with the next-best scores.
        result = []
        for net in ranked:
            if all(dist(net, old) >= 4 for old in result):
                result.append(net)
            if len(result) >= 22:
                break
        for net in ranked:
            if len(result) >= 40:
                break
            if net not in result:
                result.append(net)
        return result

    def switch(net):
        arcs = list(net)
        if len(arcs) < 2:
            return net
        i, j = map(int, rng.choice(len(arcs), size=2, replace=False))
        a, b = arcs[i]
        c, d = arcs[j]
        if a == d or c == b:
            return net
        cur = set(net)
        if (a, d) in cur or (c, b) in cur:
            return net
        cur.remove((a, b))
        cur.remove((c, d))
        cur.add((a, d))
        cur.add((c, b))
        out = tuple(sorted(cur))
        return out if tools.is_valid(out) else net

    def model_sample(pop, probabilities):
        """Sample outgoing links using elite edge frequencies, then repair."""
        chosen = set()
        # Sample two distinct destinations per source. A small uniform component
        # prevents the edge model from collapsing around early, imperfect elites.
        for a in range(n):
            weights = []
            nodes = []
            for b in range(n):
                if a == b:
                    continue
                nodes.append(b)
                weights.append(0.12 + probabilities.get((a, b), 0.0))
            weights = __import__("numpy").asarray(weights, dtype=float)
            weights /= weights.sum()
            try:
                dests = rng.choice(nodes, size=2, replace=False, p=weights)
            except Exception:
                dests = rng.choice(nodes, size=2, replace=False)
            for b in dests:
                chosen.add((a, int(b)))
        try:
            net = canon(tools.repair(chosen))
            if tools.is_valid(net):
                return net
        except Exception:
            pass
        return None

    def edge_model(pop):
        # Fitness-weighted elite frequencies: the best networks dominate, but
        # several diverse members contribute to recombination.
        ranked = sorted(pop, key=lambda x: seen.get(x, float("inf")))
        elite = ranked[:min(12, len(ranked))]
        counts = {}
        for rank, net in enumerate(elite):
            weight = 1.0 / (1.0 + 0.18 * rank)
            for edge in net:
                counts[edge] = counts.get(edge, 0.0) + weight
        denom = sum(1.0 / (1.0 + 0.18 * rank) for rank in range(len(elite)))
        if denom:
            for edge in counts:
                counts[edge] /= denom
        return counts

    # Evolutionary rounds alternate edge-model recombination and mutation.
    # Allocate calls across a proposal batch; promising surrogate candidates
    # are preferred, with explicit novelty slots for exploration.
    rounds = 0
    while tools.lp_calls_left > 0:
        rounds += 1
        pop = population()
        if not pop:
            break
        probs = edge_model(pop)
        proposals = set()

        # Generate a broad collection cheaply. Keep some descendants of strong
        # individuals, plus model samples that can make non-local combinations.
        for _ in range(70):
            parent = pop[int(rng.integers(min(len(pop), 12)))]
            child = parent
            steps = 1 + int(rng.integers(6))
            for _ in range(steps):
                child = switch(child)
            if child != parent and tools.is_valid(child):
                proposals.add(child)

        for _ in range(45):
            child = model_sample(pop, probs)
            if child is not None:
                # Brief local randomization helps turn a repaired model sample
                # into a distinct feasible topology.
                for _ in range(int(rng.integers(4))):
                    child = switch(child)
                if tools.is_valid(child):
                    proposals.add(child)

        # Crossover-like recombination: combine the edge sets of two elites and
        # let the free repair operator resolve degree conflicts.
        for _ in range(25):
            p = pop[int(rng.integers(len(pop)))]
            q = pop[int(rng.integers(len(pop)))]
            if p == q:
                continue
            mix = set()
            for edge in set(p) | set(q):
                if edge in p and edge in q:
                    mix.add(edge)
                elif rng.random() < 0.5:
                    mix.add(edge)
            try:
                child = canon(tools.repair(mix))
                if tools.is_valid(child):
                    proposals.add(child)
            except Exception:
                pass

        fresh = [
            net for net in proposals
            if net not in seen and not tools.is_cached(net)
        ]
        if not fresh:
            # Escape a saturated model with a fresh repaired sample and a
            # multi-switch perturbation.
            base = pop[int(rng.integers(len(pop)))]
            for _ in range(30):
                child = base
                for _ in range(3 + int(rng.integers(10))):
                    child = switch(child)
                if child not in seen and tools.is_valid(child):
                    fresh.append(child)
                    break
        if not fresh:
            break

        ranked = sorted(fresh, key=energy)
        batch_size = min(8, tools.lp_calls_left)
        batch = []
        # Reserve part of each batch for candidates with good surrogate energy.
        quality_slots = max(1, batch_size - 2)
        for net in ranked:
            if all(dist(net, old) >= 2 for old in batch):
                batch.append(net)
            if len(batch) >= quality_slots:
                break

        # Spend remaining calls on topological novelty relative to the current
        # population, without discarding the surrogate-ranked proposals.
        remaining = [x for x in ranked if x not in batch]
        while remaining and len(batch) < batch_size:
            chosen = max(
                remaining,
                key=lambda x: (
                    min((dist(x, old) for old in pop), default=0),
                    -energy(x),
                ),
            )
            batch.append(chosen)
            remaining.remove(chosen)

        if not batch:
            break
        for net in batch:
            if tools.lp_calls_left <= 0:
                break
            try:
                evaluate(net)
            except Exception as exc:
                if type(exc).__name__ == "BudgetExhausted":
                    return list(best_net) if best_net is not None else []
                break

        if rounds > 10000:
            break

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
