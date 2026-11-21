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
    archive = {}
    best_net = None
    best_score = float("inf")

    def canon(arcs):
        return tuple(sorted(tuple(map(int, e)) for e in arcs))

    def add(net):
        nonlocal best_net, best_score
        net = canon(net)
        if net in archive or not tools.is_valid(net):
            return False
        score = float(tools.lp(net))
        energy = float(tools.energy(net))
        archive[net] = (score, energy)
        if score < best_score:
            best_net, best_score = net, score
        return True

    # Repair samples first; evaluate low-energy seeds while retaining distinct
    # topologies so the initial archive spans more than one QAOA basin.
    repaired = {}
    for cand in candidates:
        try:
            net = canon(tools.repair(cand))
            if tools.is_valid(net):
                repaired[net] = float(tools.energy(net))
        except Exception:
            continue

    energy_order = sorted(repaired, key=repaired.get)
    seeds = []
    for net in energy_order:
        if all(len(set(net) & set(old)) < edge_count - 2 for old in seeds):
            seeds.append(net)
        if len(seeds) >= 28:
            break
    for net in energy_order:
        if len(seeds) >= 48:
            break
        if net not in seeds:
            seeds.append(net)

    for net in seeds:
        if tools.lp_calls_left <= 0:
            break
        add(net)

    if not archive and tools.lp_calls_left > 0:
        try:
            add(tools.repair([]))
        except Exception:
            pass
    if not archive:
        return []

    def trim_archive(limit=30):
        """Keep strong solutions and preserve edge-set diversity."""
        if len(archive) <= limit:
            return
        ordered = sorted(archive, key=lambda x: archive[x][0])
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
        keep_set = set(keep)
        for net in list(archive):
            if net not in keep_set:
                del archive[net]

    def proposals(net):
        arcs = list(net)
        current = set(net)
        detail = tools.lp_detail(net)
        prices = {tuple(k): float(v) for k, v in detail.get("prices", {}).items()}
        loads = {tuple(k): float(v) for k, v in detail.get("loads", {}).items()}
        positive = [v for v in prices.values() if v > 0.0]
        price_scale = float(np.mean(positive)) if positive else 1.0
        score_scale = max(1.0, float(detail.get("score", 1.0)))

        directional = {}
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

                relief = 0.0
                for edge in (arcs[p], arcs[q]):
                    load = max(0.0, loads.get(edge, 0.0))
                    relief += max(0.0, prices.get(edge, 0.0)) * (
                        1.0 + load / score_scale
                    )
                added_demand = max(0.0, float(D[a, d])) + max(
                    0.0, float(D[c, b])
                )
                value = relief - price_scale * added_demand
                if value > directional.get(nxt, -float("inf")):
                    directional[nxt] = value

        if not directional:
            return []

        nets = list(directional)
        energies = {x: float(tools.energy(x)) for x in nets}
        energy_rank = {
            x: i for i, x in enumerate(sorted(nets, key=energies.get))
        }
        dual_rank = {
            x: i for i, x in enumerate(
                sorted(nets, key=directional.get, reverse=True)
            )
        }
        return sorted(
            nets,
            key=lambda x: 0.55 * energy_rank[x] + 0.45 * dual_rank[x]
        )

    def crossover(x, y):
        # Repair the parental union to combine useful complementary links.
        try:
            child = canon(tools.repair(list(set(x) | set(y))))
        except Exception:
            return None
        return child if child != x and tools.is_valid(child) else None

    def mutate(net, count):
        """Apply a short sequence of valid degree-preserving 2-exchanges."""
        cur = net
        for _ in range(count):
            arcs = list(cur)
            if len(arcs) < 2:
                break
            p, q = rng.choice(len(arcs), size=2, replace=False)
            a, b = arcs[int(p)]
            c, d = arcs[int(q)]
            old = {(a, b), (c, d)}
            new = {(a, d), (c, b)}
            if a == d or c == b or new & (set(cur) - old):
                continue
            cur = canon((set(cur) - old) | new)
        return cur if tools.is_valid(cur) else net

    iteration = 0
    while tools.lp_calls_left > 0:
        trim_archive()
        ranked = sorted(archive, key=lambda x: archive[x][0])
        if not ranked:
            break

        # Rotate through strong members, with occasional sampling from the
        # broader archive to avoid repeatedly searching one local basin.
        if iteration % 5 == 4 and len(ranked) > 2:
            parent = ranked[int(rng.integers(min(14, len(ranked))))]
        else:
            parent = ranked[iteration % min(9, len(ranked))]

        generated = proposals(parent)

        # Recombination and short kicks complement the focused 2-exchange
        # neighborhood, particularly when the local neighborhood is exhausted.
        if len(ranked) > 1 and iteration % 3 == 0:
            other = ranked[int(rng.integers(len(ranked)))]
            if other != parent:
                child = crossover(parent, other)
                if child is not None:
                    generated.append(child)
        if iteration % 2 == 0:
            child = mutate(parent, 2 + int(rng.integers(5)))
            if child != parent:
                generated.append(child)

        seen = set()
        fresh = []
        for net in generated:
            if (net not in seen and net not in archive
                    and tools.is_valid(net) and not tools.is_cached(net)):
                seen.add(net)
                fresh.append(net)

        # If the focused neighborhood is stale, kick from a randomly selected
        # elite and try a longer sequence of exchanges.
        if len(fresh) < 3:
            base = ranked[int(rng.integers(min(len(ranked), 12)))]
            child = mutate(base, 4 + int(rng.integers(7)))
            if (child != base and child not in archive
                    and child not in seen and tools.is_valid(child)
                    and not tools.is_cached(child)):
                fresh.append(child)

        if not fresh:
            iteration += 1
            continue

        batch_size = min(8, int(tools.lp_calls_left), len(fresh))
        batch = fresh[:batch_size]
        # Reserve part of the LP batch for less highly ranked moves, since both
        # the dual estimate and fixed surrogate can mis-rank promising networks.
        if len(fresh) > batch_size and batch_size >= 3:
            reserve = min(2, batch_size // 3, len(fresh) - batch_size)
            if reserve:
                picks = rng.choice(
                    len(fresh) - batch_size, size=reserve, replace=False
                )
                batch = fresh[:batch_size - reserve] + [
                    fresh[batch_size + int(i)] for i in picks
                ]

        for net in batch:
            if tools.lp_calls_left <= 0:
                break
            add(net)

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
