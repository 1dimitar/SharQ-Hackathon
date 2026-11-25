"""Classical post-processing of QAOA output for QOBLIB 08. ShinkaEvolve edits the EVOLVE block."""

import json
import time

import numpy as np

# EVOLVE-BLOCK-START
import math


def improve(demand, candidates, tools, rng):
    """Search with a diverse archive, recombination, and batched mutations."""
    n = int(tools.n)
    m = 2 * n
    archive = {}  # net -> (LP score, energy)
    best_net = None
    best_score = float("inf")
    seen = set()

    def canon(arcs):
        return tuple(sorted(tuple(map(int, a)) for a in arcs))

    def evaluate(net):
        nonlocal best_net, best_score
        net = canon(net)
        if net in seen:
            return archive.get(net, (float("inf"), 0.0))[0]
        seen.add(net)
        if not tools.is_valid(net):
            return float("inf")
        try:
            score = float(tools.lp(net))
        except Exception:
            raise
        energy = float(tools.energy(net))
        archive[net] = (score, energy)
        if score < best_score:
            best_net, best_score = net, score
        return score

    # Seed from repaired QAOA samples. Energy-ordering avoids spending the whole
    # initial budget on near-identical repairs.
    repaired = {}
    for cand in candidates:
        net = canon(tools.repair(cand))
        if tools.is_valid(net):
            repaired[net] = float(tools.energy(net))
    seeds = sorted(repaired, key=repaired.get)

    # Keep a structurally diverse subset of seeds while retaining energy quality.
    chosen = []
    for net in seeds:
        if all(len(set(net) & set(old)) <= m - 3 for old in chosen):
            chosen.append(net)
        if len(chosen) >= 36:
            break
    for net in seeds:
        if len(chosen) >= 55:
            break
        if net not in chosen:
            chosen.append(net)
    if not chosen:
        chosen = [canon(tools.repair([]))]

    for net in chosen:
        try:
            evaluate(net)
        except Exception:
            return list(best_net) if best_net is not None else list(net)

    if best_net is None:
        return list(canon(tools.repair([])))

    def ordered_archive():
        return sorted(archive, key=lambda x: archive[x][0])

    def trim_archive(limit=36):
        """Retain strong but edge-diverse parents."""
        nonlocal archive
        order = ordered_archive()
        keep = []
        for net in order:
            if all(len(set(net) & set(old)) <= m - 3 for old in keep):
                keep.append(net)
            if len(keep) >= limit:
                break
        for net in order:
            if len(keep) >= limit:
                break
            if net not in keep:
                keep.append(net)
        archive = {net: archive[net] for net in keep}

    def exchange_moves(net, hot=None):
        arcs = list(net)
        cur = set(net)
        moves = []
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
                nxt = tuple(sorted((cur - {arcs[p], arcs[q]}) | {x, y}))
                if nxt != net:
                    moves.append(nxt)
        return moves

    def mutate(net, count):
        cur = net
        for _ in range(count):
            options = exchange_moves(cur)
            if not options:
                break
            cur = options[int(rng.integers(len(options)))]
        return cur

    def recombine(x, y):
        # Repair their edge union; the repair operator supplies a valid child
        # while preserving as much parental structure as possible.
        child = canon(tools.repair(list(set(x) | set(y))))
        return child if tools.is_valid(child) else None

    def diversity_order(items, reference):
        """Greedy selection: good energy, but reward edge-set novelty."""
        remaining = list(items)
        selected = []
        while remaining:
            def key(net):
                e = float(tools.energy(net))
                novelty = min(
                    (m - len(set(net) & set(old)) for old in reference + selected),
                    default=m
                )
                # Energy dominates; novelty breaks ties and prevents cloned batches.
                return (e - 0.002 * novelty * max(1.0, abs(e)), rng.random())
            net = min(remaining, key=key)
            remaining.remove(net)
            selected.append(net)
        return selected

    iteration = 0
    while True:
        try:
            left = int(tools.lp_calls_left)
        except Exception:
            left = 0
        if left <= 0:
            break

        trim_archive()
        ranked = ordered_archive()
        if not ranked:
            break

        parents = ranked[:min(12, len(ranked))]
        proposals = set()

        # Recombine several pairs from the good, diverse part of the archive.
        for _ in range(16):
            if len(parents) >= 2:
                i, j = rng.choice(len(parents), size=2, replace=False)
                child = recombine(parents[int(i)], parents[int(j)])
                if child is not None:
                    proposals.add(child)

        # Mutate a rotating set of parents. Some mutations target LP bottlenecks;
        # others are unconstrained to permit moves away from the current basin.
        for parent in parents[:8]:
            if rng.random() < 0.55:
                try:
                    detail = tools.lp_detail(parent)
                    hot = {tuple(k) for k, v in detail.get("prices", {}).items()
                           if float(v) > 0}
                except Exception:
                    hot = set()
            else:
                hot = set()
            options = exchange_moves(parent, hot if hot else None)
            if options:
                rng.shuffle(options)
                proposals.update(options[:8])
            proposals.add(mutate(parent, 2 + int(rng.integers(4))))

        # Occasionally combine a good parent with a randomly chosen seed, preserving
        # a route to less competitive but potentially useful QAOA structures.
        if seeds and iteration % 3 == 0:
            parent = parents[int(rng.integers(len(parents)))]
            other = seeds[int(rng.integers(len(seeds)))]
            child = recombine(parent, other)
            if child is not None:
                proposals.add(child)

        proposals = [
            net for net in proposals
            if net not in seen and tools.is_valid(net) and not tools.is_cached(net)
        ]
        if not proposals:
            # Ensure progress when recombination and local neighborhoods are exhausted.
            parent = parents[int(rng.integers(len(parents)))]
            proposals = [mutate(parent, 1 + int(rng.integers(6)))]
            proposals = [x for x in proposals
                         if x not in seen and tools.is_valid(x) and not tools.is_cached(x)]
            if not proposals:
                iteration += 1
                continue

        # Rank proposals cheaply, then fill the LP batch with structurally distinct
        # candidates rather than spending calls on near-duplicates.
        proposals = diversity_order(proposals, parents[:6])
        batch_size = min(12, left, len(proposals))
        batch = proposals[:batch_size]
        if len(proposals) > batch_size and batch_size < left:
            # A small exploratory tail protects against poor energy correlation.
            tail = proposals[batch_size:]
            extra = min(2, left - batch_size, len(tail))
            if extra:
                batch.extend(rng.choice(len(tail), size=extra, replace=False)
                             .tolist())
                batch[-extra:] = [tail[int(i)] for i in batch[-extra:]]

        for net in batch:
            if net in seen or tools.is_cached(net):
                continue
            try:
                evaluate(net)
            except Exception:
                return list(best_net)

        iteration += 1

    return list(best_net)
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
