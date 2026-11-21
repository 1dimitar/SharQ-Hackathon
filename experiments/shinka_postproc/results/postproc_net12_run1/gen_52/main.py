"""Classical post-processing of QAOA output for QOBLIB 08. ShinkaEvolve edits the EVOLVE block."""

import json
import time

import numpy as np

# EVOLVE-BLOCK-START
import math


def improve(demand, candidates, tools, rng):
    n = tools.n
    edge_count = 2 * n
    archive = {}
    best_net = None
    best_score = float("inf")

    def add(net):
        nonlocal best_net, best_score
        net = tuple(sorted(tuple(map(int, e)) for e in net))
        if net in archive or not tools.is_valid(net):
            return False
        score = float(tools.lp(net))
        archive[net] = score
        if score < best_score:
            best_net, best_score = net, score
        return True

    # Seed from repaired samples, with good-energy samples considered first.
    seeds = set()
    for cand in candidates:
        try:
            seeds.add(tuple(sorted(tuple(map(int, e)) for e in tools.repair(cand))))
        except Exception:
            pass
    for net in sorted(seeds, key=lambda x: tools.energy(x)):
        if tools.lp_calls_left <= 0:
            break
        add(net)

    if not archive and tools.lp_calls_left > 0:
        add(tools.repair([]))

    def trim_archive(limit=28):
        """Keep strong members, but avoid a collection of near-identical networks."""
        if len(archive) <= limit:
            return
        ordered = sorted(archive, key=archive.get)
        keep = []
        for net in ordered:
            if all(len(set(net) & set(old)) < edge_count - 2 for old in keep):
                keep.append(net)
                if len(keep) >= limit:
                    break
        for net in ordered:
            if len(keep) >= limit:
                break
            if net not in keep:
                keep.append(net)
        for net in list(archive):
            if net not in keep:
                del archive[net]

    def proposals(net):
        arcs = list(net)
        current = set(net)
        detail = tools.lp_detail(net)
        prices = {tuple(k): float(v) for k, v in detail.get("prices", {}).items()}
        loads = {tuple(k): float(v) for k, v in detail.get("loads", {}).items()}
        positive = [v for v in prices.values() if v > 0]
        scale = (sum(positive) / len(positive)) if positive else 1.0

        moves = []
        for p in range(len(arcs)):
            a, b = arcs[p]
            for q in range(p + 1, len(arcs)):
                c, d = arcs[q]
                if a == d or c == b:
                    continue
                x, y = (a, d), (c, b)
                if x in current or y in current:
                    continue
                nxt = tuple(sorted((current - {arcs[p], arcs[q]}) | {x, y}))
                if nxt == net:
                    continue

                # Approximate benefit of removing priced, loaded links and
                # inserting links with high direct demand.
                relief = 0.0
                for e in (arcs[p], arcs[q]):
                    relief += max(0.0, prices.get(e, 0.0)) * (
                        1.0 + max(0.0, loads.get(e, 0.0)) /
                        max(1.0, float(tools.lp_detail(net).get("score", 1.0)))
                    )
                added = max(0.0, float(demand[a, d])) + max(
                    0.0, float(demand[c, b])
                )
                directional = relief - scale * added
                moves.append((nxt, directional))

        # Rank both by the explicit price/demand signal and by the fixed
        # Hamiltonian surrogate; they are complementary noisy predictors.
        if not moves:
            return []
        en = [(net_, float(tools.energy(net_)), direction)
              for net_, direction in moves]
        e_sorted = sorted(en, key=lambda z: z[1])
        d_sorted = sorted(en, key=lambda z: -z[2])
        erank = {z[0]: i for i, z in enumerate(e_sorted)}
        drank = {z[0]: i for i, z in enumerate(d_sorted)}
        ranked = sorted(en, key=lambda z: (
            0.55 * erank[z[0]] + 0.45 * drank[z[0]]
        ))
        # Keep only unique proposals; downstream batch selection reserves
        # exploration from the weaker half as well.
        seen, out = set(), []
        for item in ranked:
            if item[0] not in seen:
                seen.add(item[0])
                out.append(item[0])
        return out

    iteration = 0
    stagnant = 0
    while tools.lp_calls_left > 0:
        trim_archive()
        members = list(archive)
        if not members:
            break
        ordered = sorted(members, key=archive.get)

        # Cycle among good archive members, occasionally favoring diversity.
        if iteration % 4 == 3 and len(ordered) > 2:
            parent = ordered[int(rng.integers(min(12, len(ordered))))]
        else:
            parent = ordered[iteration % min(8, len(ordered))]

        previous_best = best_score
        generated = proposals(parent)

        # When progress stalls, kick a non-elite archive member by a few
        # valid degree-preserving switches, preserving the established elites.
        if stagnant >= 8 and len(ordered) > 4 and iteration % 3 == 0:
            base = ordered[int(rng.integers(4, len(ordered)))]
            arcs = list(base)
            current = set(base)
            kick_steps = min(7, 2 + stagnant // 8)
            for _ in range(kick_steps):
                moved = False
                for _ in range(40):
                    p, q = map(int, rng.choice(len(arcs), size=2, replace=False))
                    a, b = arcs[p]
                    c, d = arcs[q]
                    x, y = (a, d), (c, b)
                    if a == d or c == b or x in current or y in current:
                        continue
                    current.remove((a, b))
                    current.remove((c, d))
                    current.add(x)
                    current.add(y)
                    arcs[p], arcs[q] = x, y
                    moved = True
                    break
                if not moved:
                    break
            kicked = tuple(sorted(current))
            if kicked != base and tools.is_valid(kicked):
                generated.append(kicked)

        # Perturb a random archive member if the one-switch neighborhood is
        # exhausted or repeatedly visited.
        fresh = [x for x in generated
                 if x not in archive and not tools.is_cached(x)]
        if len(fresh) < 4:
            base = parent
            for _ in range(2 + int(rng.integers(4))):
                neigh = proposals(base)
                options = [x for x in neigh if x != base]
                if not options:
                    break
                base = options[int(rng.integers(min(len(options), 12)))]
            if base not in archive and tools.is_valid(base):
                fresh.append(base)

        fresh = list(dict.fromkeys(
            x for x in fresh if tools.is_valid(x) and not tools.is_cached(x)
        ))
        if not fresh:
            iteration += 1
            continue

        # Mostly spend calls on ranked moves, but reserve a few for
        # independently selected proposals to hedge against proxy error.
        batch_size = min(8, tools.lp_calls_left, len(fresh))
        ranked_batch = fresh[:batch_size]
        if len(fresh) > batch_size and batch_size >= 3:
            reserve = min(2, batch_size // 3)
            tail = fresh[batch_size:]
            if tail:
                picks = rng.choice(len(tail), size=min(reserve, len(tail)),
                                   replace=False)
                ranked_batch = fresh[:batch_size - len(picks)]
                ranked_batch.extend(tail[int(i)] for i in picks)

        for net in ranked_batch:
            if tools.lp_calls_left <= 0:
                break
            add(net)

        stagnant = stagnant + 1 if best_score >= previous_best else 0
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
