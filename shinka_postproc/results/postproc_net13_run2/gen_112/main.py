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
    """Search a diverse, adaptive frontier of valid degree-preserving networks."""
    n = tools.n
    scores = {}
    visits = {}
    stalls = {}
    operator_stalls = {}
    last_sources = {}
    best_net = None
    best_score = float("inf")
    nonimproving_rounds = 0

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
        if net not in scores:
            scores[net] = tools.lp(net)
        value = scores[net]
        if value < best_score:
            best_net, best_score = net, value
        return value

    # Build a seed set from all repaired candidates, then score a
    # surrogate-ranked but topologically varied subset.
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

    ranked_seeds = sorted(repaired, key=energy)
    seeds = []
    for net in ranked_seeds:
        if all(distance(net, other) >= 4 for other in seeds):
            seeds.append(net)
        if len(seeds) >= 24:
            break
    for net in ranked_seeds:
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
            continue

    if not scores:
        return list(seeds[0])

    def frontier():
        """Keep high-quality representatives from distinct regions."""
        ordered = sorted(scores, key=scores.get)
        chosen = []
        for net in ordered:
            if all(distance(net, other) >= 4 for other in chosen):
                chosen.append(net)
            if len(chosen) >= 20:
                break
        for net in ordered:
            if len(chosen) >= 36:
                break
            if net not in chosen:
                chosen.append(net)
        return chosen

    def choose_parent(pop):
        """Balance elite exploitation with under-visited archive members."""
        if len(pop) == 1:
            return pop[0]

        # Reduce elite-only selection as the search experiences longer stalls.
        elite_probability = max(0.18, 0.62 - 0.08 * (nonimproving_rounds // 8))
        if rng.random() < elite_probability:
            return pop[int(rng.integers(min(6, len(pop))))]

        weights = []
        for net in pop:
            quality_rank = pop.index(net)
            quality = 1.0 / (1.0 + 0.08 * quality_rank)
            novelty = 1.0 / ((1.0 + visits.get(net, 0)) ** 0.8)
            stale_bonus = 1.0 + min(3.0, stalls.get(net, 0) * 0.15)
            weights.append(quality * novelty * stale_bonus)

        pick = float(rng.random()) * sum(weights)
        for net, weight in zip(pop, weights):
            pick -= weight
            if pick <= 0:
                return net
        return pop[-1]

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

    def propose(parent, detail):
        """Generate adaptive bottleneck-focused and exploratory proposals."""
        nonlocal last_sources
        arcs = list(parent)
        current = set(parent)
        moves = set()
        source_by_move = {}

        prices = detail.get("prices", {}) if detail else {}
        hot = {
            tuple(map(int, link))
            for link, price in prices.items()
            if float(price) > 0
        }
        if not hot and detail:
            loads = detail.get("loads", {})
            by_load = sorted(
                ((float(value), tuple(map(int, link)))
                 for link, value in loads.items()),
                reverse=True,
            )
            hot = {link for _, link in by_load[:6]}
        elif detail and detail.get("loads"):
            # If price-guided moves stall, progressively include more heavily
            # loaded links in the focused neighborhood.
            by_load = sorted(
                ((float(value), tuple(map(int, link)))
                 for link, value in detail["loads"].items()),
                reverse=True,
            )
            extra = min(12, 2 + 2 * operator_stalls.get((parent, "focused"), 0))
            hot.update(link for _, link in by_load[:extra])

        # Exhaustive two-switches touching at least one priced/load-heavy arc.
        for p in range(len(arcs)):
            for q in range(p + 1, len(arcs)):
                if hot and arcs[p] not in hot and arcs[q] not in hot:
                    continue
                move = two_switch(parent, p, q)
                if move is not None:
                    moves.add(move)
                    source_by_move.setdefault(move, "focused")

        # Expand unrestricted exploration only when this operator has stalled.
        random_stall = operator_stalls.get((parent, "random"), 0)
        for _ in range(55 + 20 * min(8, random_stall)):
            p = int(rng.integers(len(arcs)))
            q = int(rng.integers(len(arcs) - 1))
            if q >= p:
                q += 1
            move = two_switch(parent, p, q)
            if move is not None:
                moves.add(move)
                source_by_move.setdefault(move, "random")

        # Destination rotations preserve all in/out degrees and can cross
        # barriers that single two-switches do not.
        if len(arcs) >= 3:
            rotation_stall = operator_stalls.get((parent, "rotation"), 0)
            for _ in range(65 + 20 * min(8, rotation_stall)):
                ids = rng.choice(len(arcs), size=3, replace=False)
                (a, b), (c, d), (e, f) = (
                    arcs[int(i)] for i in ids
                )
                old = {(a, b), (c, d), (e, f)}
                if hot and not (old & hot):
                    continue
                for new in (
                    {(a, d), (c, f), (e, b)},
                    {(a, f), (c, b), (e, d)},
                ):
                    if len(new) != 3 or any(u == v for u, v in new):
                        continue
                    if new & (current - old):
                        continue
                    move = tuple(sorted((current - old) | new))
                    moves.add(move)
                    source_by_move.setdefault(move, "rotation")

        # Bounded matching proposals: permuting destinations across distinct
        # sources preserves the destination multiset and all degree counts.
        # Archive edge frequencies bias proposals toward recurring good structure.
        edge_frequency = {}
        elite = sorted(scores, key=scores.get)[:24]
        for rank, net in enumerate(elite):
            weight = 1.0 / (1.0 + 0.05 * rank)
            for edge in net:
                edge_frequency[edge] = edge_frequency.get(edge, 0.0) + weight

        relink_stall = operator_stalls.get((parent, "relink"), 0)
        for _ in range(12 + 2 * min(8, relink_stall)):
            k = 3 + int(rng.integers(4))
            sources = rng.choice(n, size=k, replace=False)
            selected = []
            for source in sources:
                outgoing = [edge for edge in arcs if edge[0] == int(source)]
                selected.append(outgoing[int(rng.integers(len(outgoing)))])
            old = set(selected)
            destinations = [b for _, b in selected]
            remainder = current - old

            # Sample a few assignments and keep the valid one most supported
            # by the elite archive, with random tie-breaking.
            best_assignment = None
            best_weight = -1.0
            for _ in range(8):
                permuted = list(destinations)
                rng.shuffle(permuted)
                new = {(selected[i][0], permuted[i]) for i in range(k)}
                if (len(new) != k
                        or any(a == b for a, b in new)
                        or new & remainder):
                    continue
                weight = sum(edge_frequency.get(edge, 0.0) for edge in new)
                weight += float(rng.random()) * 1e-6
                if weight > best_weight:
                    best_weight = weight
                    best_assignment = new

            if best_assignment is not None:
                move = tuple(sorted(remainder | best_assignment))
                moves.add(move)
                source_by_move.setdefault(move, "relink")

        valid_moves = [
            net for net in moves
            if net != parent
            and net not in scores
            and tools.is_valid(net)
        ]
        last_sources = {net: source_by_move.get(net, "focused") for net in valid_moves}
        return valid_moves

    def select_batch(moves, parent, limit=5):
        """Blend surrogate quality with novelty relative to the scored archive."""
        if not moves:
            return []

        base_energy = energy(parent)
        ranked = sorted(
            moves,
            key=lambda net: energy(net) - base_energy,
        )

        # Use a compact set of strong, mutually distinct archive members as
        # reference points. Distance from these rewards unexplored topology
        # regions without making novelty the only selection criterion.
        archive = sorted(scores, key=scores.get)[:24]
        if not archive:
            archive = [parent]
        novelty = {
            net: min(distance(net, ref) for ref in archive)
            for net in moves
        }

        selected = []
        # Preserve exploitation: take the strongest surrogate proposals first.
        for net in ranked:
            if all(distance(net, old) >= 3 for old in selected):
                selected.append(net)
            if len(selected) >= min(2, limit):
                break

        # Search for underrepresented regions, but only within a surrogate
        # shortlist so wildly poor-energy proposals do not dominate.
        shortlist = ranked[:max(12, limit * 5)]
        while len(selected) < min(4, limit):
            available = [net for net in shortlist if net not in selected]
            if not available:
                break
            pick = max(
                available,
                key=lambda net: (novelty[net], -ranked.index(net)),
            )
            selected.append(pick)

        # Spend any remaining batch capacity on the best-ranked proposals.
        for net in ranked:
            if len(selected) >= limit:
                break
            if net not in selected:
                selected.append(net)
        return selected

    while tools.lp_calls_left > 0:
        pop = frontier()
        if not pop:
            break

        parent = choose_parent(pop)
        visits[parent] = visits.get(parent, 0) + 1
        try:
            detail = tools.lp_detail(parent)
        except Exception:
            detail = None

        batch = select_batch(propose(parent, detail), parent)

        # Try a few other frontier members before resorting to a random kick.
        if not batch:
            alternatives = [net for net in pop if net != parent]
            alternatives.sort(key=lambda net: visits.get(net, 0))
            for alternative in alternatives[:8]:
                try:
                    alt_detail = tools.lp_detail(alternative)
                except Exception:
                    alt_detail = None
                batch = select_batch(propose(alternative, alt_detail), alternative)
                if batch:
                    parent = alternative
                    visits[parent] = visits.get(parent, 0) + 1
                    break

        # Escaping an exhausted neighborhood with a short sequence of switches.
        if not batch:
            for _ in range(50):
                current = set(parent)
                for _ in range(2 + int(rng.integers(4))):
                    now = list(current)
                    p, q = rng.choice(len(now), size=2, replace=False)
                    a, b = now[int(p)]
                    c, d = now[int(q)]
                    if a == d or c == b or (a, d) in current or (c, b) in current:
                        continue
                    current.remove((a, b))
                    current.remove((c, d))
                    current.add((a, d))
                    current.add((c, b))
                kicked = tuple(sorted(current))
                if (
                    kicked != parent
                    and kicked not in scores
                    and tools.is_valid(kicked)
                ):
                    batch = [kicked]
                    break

        if not batch:
            break

        parent_score = scores.get(parent, float("inf"))
        previous_best = best_score
        for net in batch:
            if tools.lp_calls_left <= 0:
                break
            try:
                evaluate(net)
            except Exception as exc:
                if type(exc).__name__ == "BudgetExhausted":
                    return list(best_net) if best_net is not None else []
                continue

        batch_best = min(
            (scores.get(net, float("inf")) for net in batch),
            default=float("inf"),
        )
        if batch_best < parent_score:
            stalls[parent] = 0
        else:
            stalls[parent] = stalls.get(parent, 0) + 1

        # Adapt each proposal family from its own selected/evaluated outcomes.
        tested = {}
        for net in batch:
            operator = last_sources.get(net)
            if operator is not None:
                tested.setdefault(operator, []).append(net)
        for operator, tested_nets in tested.items():
            key = (parent, operator)
            if any(scores.get(net, float("inf")) < parent_score for net in tested_nets):
                operator_stalls[key] = 0
            else:
                operator_stalls[key] = min(12, operator_stalls.get(key, 0) + 1)

        if best_score < previous_best:
            nonimproving_rounds = 0
        else:
            nonimproving_rounds += 1

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
