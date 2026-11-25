"""Classical post-processing of QAOA output for QOBLIB 08. ShinkaEvolve edits the EVOLVE block."""

import json
import time

import numpy as np

# EVOLVE-BLOCK-START
def improve(demand, candidates, tools, rng):
    """Search repaired candidates and their neighborhoods for a low-congestion network."""
    n = tools.n
    arcs_per_net = 2 * n
    demand = np.asarray(demand, dtype=float)

    # Main search parameters.
    archive_limit = 14
    batch_size = 8
    rotation_triples = 90
    max_iterations = 10000

    scores = {}
    archive = []
    # Per-parent search feedback: evaluated candidates, incumbent improvements,
    # and cumulative improvement magnitude.
    parent_stats = {}
    best_net = None
    best_score = float("inf")

    def canonical(net):
        return tuple(sorted(tuple(map(int, a)) for a in net))

    def record(net, score=None):
        """LP-score a new valid network once and update the incumbent."""
        nonlocal best_net, best_score
        net = canonical(net)
        if net in scores:
            return scores[net]
        if not tools.is_valid(net):
            return float("inf")
        if score is None:
            if tools.lp_calls_left <= 0:
                return float("inf")
            score = tools.lp(net)
        score = float(score)
        scores[net] = score
        if score < best_score:
            best_net, best_score = net, score
        return score

    # Seed the search with repaired QAOA samples, preserving their LP evaluations.
    repaired = set()
    for cand in candidates:
        net = canonical(tools.repair(cand))
        if net not in repaired and tools.is_valid(net):
            repaired.add(net)
            if tools.lp_calls_left > 0:
                record(net)
            else:
                break

    if best_net is None:
        fallback = canonical(tools.repair([]))
        if tools.is_valid(fallback) and tools.lp_calls_left > 0:
            record(fallback)

    if best_net is None:
        return []

    def distance(x, y):
        return len(set(x) ^ set(y))

    def refresh_archive():
        """Retain strong members while reserving room for structurally different nets."""
        nonlocal archive
        viable = [net for net in scores if np.isfinite(scores[net])]
        viable.sort(key=lambda net: scores[net])
        if len(viable) <= archive_limit:
            archive = viable
            return

        chosen = viable[:min(6, archive_limit)]
        remaining = viable[len(chosen):]
        while remaining and len(chosen) < archive_limit:
            # Prefer networks dissimilar to the selected set, but do not keep
            # extremely poor networks merely for novelty.
            eligible = remaining[:min(len(remaining), 40)]
            pick = max(
                eligible,
                key=lambda net: (
                    min(distance(net, old) for old in chosen)
                    - 0.15 * (scores[net] - best_score) / max(1.0, best_score)
                    + 1e-9 * float(rng.random())
                ),
            )
            chosen.append(pick)
            remaining.remove(pick)
        archive = chosen

    def make_switches(net):
        """Enumerate legal two-switches, including moves not touching hot links."""
        arcs = list(net)
        cur = set(net)
        proposals = set()
        for p in range(len(arcs)):
            a, b = arcs[p]
            for q in range(p + 1, len(arcs)):
                c, d = arcs[q]
                if a == d or c == b:
                    continue
                x, y = (a, d), (c, b)
                if x in cur or y in cur:
                    continue
                nxt = canonical((cur - {arcs[p], arcs[q]}) | {x, y})
                if nxt != net:
                    proposals.add(nxt)
        return proposals

    def make_rotations(net, hot):
        """Sample targeted three-link target rotations to broaden the neighborhood."""
        arcs = list(net)
        cur = set(net)
        triples = []
        for p in range(len(arcs)):
            for q in range(p + 1, len(arcs)):
                for r in range(q + 1, len(arcs)):
                    if arcs[p] in hot or arcs[q] in hot or arcs[r] in hot:
                        triples.append((p, q, r))
        if not triples:
            return set()
        rng.shuffle(triples)
        proposals = set()
        for p, q, r in triples[:rotation_triples]:
            (a, b), (c, d), (e, f) = arcs[p], arcs[q], arcs[r]
            removed = {arcs[p], arcs[q], arcs[r]}
            outside = cur - removed
            for added in (
                ((a, d), (c, f), (e, b)),
                ((a, f), (c, b), (e, d)),
            ):
                if len(set(added)) != 3 or any(x == y for x, y in added):
                    continue
                if any(edge in outside for edge in added):
                    continue
                nxt = canonical(outside | set(added))
                if nxt != net:
                    proposals.add(nxt)
        return proposals

    def random_kick(net, count):
        """Apply a few random legal switches to produce a distant proposal."""
        cur = net
        for _ in range(count):
            arcs = list(cur)
            pairs = [(p, q) for p in range(len(arcs)) for q in range(p + 1, len(arcs))]
            if not pairs:
                break
            rng.shuffle(pairs)
            cur_set = set(cur)
            for p, q in pairs:
                (a, b), (c, d) = arcs[p], arcs[q]
                x, y = (a, d), (c, b)
                if a != d and c != b and x not in cur_set and y not in cur_set:
                    cur = canonical((cur_set - {arcs[p], arcs[q]}) | {x, y})
                    break
        return cur

    def choose_parent():
        refresh_archive()
        if len(archive) == 1:
            return archive[0]
        ranked = sorted(archive, key=lambda net: scores[net])
        # Bias toward strong members without making the best member deterministic.
        weights = []
        for i, net in enumerate(ranked):
            trials, wins, gain = parent_stats.get(net, (0, 0, 0.0))
            # Favor parents whose neighborhoods have paid off, but preserve a
            # substantial rank-based and uncertainty-driven exploration term.
            productivity = (1.0 + 2.0 * wins + 4.0 * gain) / (1.0 + trials) ** 0.5
            exploration = 1.0 + 1.5 / (1.0 + trials) ** 0.5
            weights.append((1.0 / ((i + 1) ** 0.72)) *
                           (0.35 + productivity + exploration))
        weights = np.asarray(weights, dtype=float)
        return ranked[int(rng.choice(len(ranked), p=weights / weights.sum()))]

    def propose_batch(parent):
        detail = tools.lp_detail(parent)
        prices = {tuple(k): float(v) for k, v in detail.get("prices", {}).items()}
        loads = {tuple(k): float(v) for k, v in detail.get("loads", {}).items()}
        arcs = list(parent)
        hot_order = sorted(
            arcs,
            key=lambda a: (prices.get(a, 0.0), loads.get(a, 0.0)),
            reverse=True,
        )
        hot = set(hot_order[:max(5, n // 2)])

        proposals = make_switches(parent)
        proposals.update(make_rotations(parent, hot))

        # Add a few larger perturbations so the search can leave a local basin.
        for _ in range(5):
            kicked = random_kick(parent, 2 + int(rng.integers(3)))
            if kicked != parent:
                proposals.add(kicked)

        proposals = {
            net for net in proposals
            if net not in scores and not tools.is_cached(net) and tools.is_valid(net)
        }
        if not proposals:
            return []

        parent_set = set(parent)
        features = []
        for net in proposals:
            net_set = set(net)
            removed = parent_set - net_set
            added = net_set - parent_set
            dual_benefit = (
                sum(prices.get(a, 0.0) for a in removed)
                - sum(prices.get(a, 0.0) for a in added)
            )
            demand_proxy = sum(float(demand[a, b]) for a, b in net)
            energy = float(tools.energy(net))
            features.append((net, dual_benefit, energy, demand_proxy))

        dual_order = sorted(features, key=lambda x: x[1], reverse=True)
        energy_order = sorted(features, key=lambda x: x[2])
        demand_order = sorted(features, key=lambda x: x[3])
        dual_rank = {x[0]: i for i, x in enumerate(dual_order)}
        energy_rank = {x[0]: i for i, x in enumerate(energy_order)}
        demand_rank = {x[0]: i for i, x in enumerate(demand_order)}

        blended = sorted(
            proposals,
            key=lambda net: (
                0.45 * dual_rank[net]
                + 0.40 * energy_rank[net]
                + 0.15 * demand_rank[net]
                + 0.10 * float(rng.random())
            ),
        )

        # Reserve slots for each signal, then fill remaining slots with blended
        # choices. This avoids relying on a single noisy scale or ranking.
        chosen = []
        chosen_set = set()

        def take(order, count):
            for net in order:
                if len(chosen) >= batch_size or count <= 0:
                    break
                if net not in chosen_set:
                    chosen.append(net)
                    chosen_set.add(net)
                    count -= 1

        take([x[0] for x in dual_order], 2)
        take([x[0] for x in energy_order], 2)
        take(blended, batch_size - len(chosen) - 1)

        leftovers = [net for net in proposals if net not in chosen_set]
        if leftovers and len(chosen) < batch_size:
            chosen.append(leftovers[int(rng.integers(len(leftovers)))])
        return chosen

    refresh_archive()
    iteration = 0
    while tools.lp_calls_left > 0 and iteration < max_iterations:
        parent = choose_parent()
        batch = propose_batch(parent)
        if not batch:
            # A random archive member or kick can expose moves not present in
            # the current parent's neighborhood.
            parent = archive[int(rng.integers(len(archive)))]
            kicked = random_kick(parent, 3 + int(rng.integers(3)))
            batch = [kicked] if kicked not in scores and not tools.is_cached(kicked) else []
        if not batch:
            iteration += 1
            continue

        before = best_score
        evaluated = 0
        for net in batch:
            if tools.lp_calls_left <= 0:
                break
            if net in scores or tools.is_cached(net):
                continue
            record(net)
            evaluated += 1
        if evaluated:
            trials, wins, gain = parent_stats.get(parent, (0, 0, 0.0))
            improvement = max(0.0, before - best_score)
            parent_stats[parent] = (
                trials + evaluated,
                wins + int(improvement > 0.0),
                gain + improvement / max(1.0, before),
            )
        refresh_archive()
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
