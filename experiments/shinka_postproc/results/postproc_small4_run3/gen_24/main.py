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
import numpy as np


def improve(demand, candidates, tools, rng):
    n = tools.n
    demand = np.asarray(demand, dtype=float)
    seen = {}
    feature_cache = {}
    best_net = None
    best_score = float("inf")
    parent_stalls = {}
    model = {"ready": False, "mean": None, "scale": None, "coef": None, "inv": None}
    rounds = 0

    def canon(arcs):
        return tuple(sorted(tuple(map(int, a)) for a in arcs))

    def energy(net):
        try:
            return float(tools.energy(net))
        except Exception:
            return 0.0

    def evaluate(net):
        nonlocal best_net, best_score
        net = canon(net)
        if net not in seen:
            seen[net] = float(tools.lp(net))
        score = seen[net]
        if score < best_score:
            best_net, best_score = net, score
        return score

    def distance(a, b):
        return len(set(a) ^ set(b))

    def features(net):
        """Cheap, fixed-size surrogate features for ranking and ridge fitting."""
        if net in feature_cache:
            return feature_cache[net]
        vals = np.asarray([demand[a, b] for a, b in net], dtype=float)
        if vals.size == 0:
            vals = np.zeros(1, dtype=float)
        row_totals = np.zeros(n, dtype=float)
        col_totals = np.zeros(n, dtype=float)
        for a, b in net:
            row_totals[a] += demand[a, b]
            col_totals[b] += demand[a, b]
        e = energy(net)
        x = np.asarray([
            e,
            float(np.sum(vals)),
            float(np.max(vals)),
            float(np.std(vals)),
            float(np.mean(vals)),
            float(np.sum(vals * vals)),
            float(np.max(row_totals)),
            float(np.std(row_totals)),
            float(np.max(col_totals)),
            float(np.std(col_totals)),
        ], dtype=float)
        feature_cache[net] = x
        return x

    def fit_model():
        """Ridge fit on a bounded, recent sample of certified LP outcomes."""
        if len(seen) < 18:
            model["ready"] = False
            return
        rows = list(seen.items())[-700:]
        X = np.vstack([features(net) for net, _ in rows])
        y = np.asarray([score for _, score in rows], dtype=float)
        mu = X.mean(axis=0)
        sd = X.std(axis=0)
        sd[sd < 1e-9] = 1.0
        Z = (X - mu) / sd
        ymu = float(y.mean())
        ysd = float(y.std())
        if ysd < 1e-9:
            ysd = max(abs(ymu) * 1e-5, 1.0)
        yz = (y - ymu) / ysd
        # Mild regularization; the intercept is handled separately.
        gram = Z.T @ Z + 2.5 * np.eye(Z.shape[1])
        try:
            inv = np.linalg.inv(gram)
            coef = inv @ (Z.T @ yz)
        except Exception:
            model["ready"] = False
            return
        model.update({
            "ready": True, "mean": mu, "scale": sd, "coef": coef,
            "inv": inv, "ymean": ymu, "yscale": ysd,
        })

    def model_values(nets):
        X = np.vstack([features(net) for net in nets])
        Z = (X - model["mean"]) / model["scale"]
        pred = model["ymean"] + (Z @ model["coef"]) * model["yscale"]
        # Posterior leverage is a useful, inexpensive uncertainty proxy.
        leverage = np.einsum("ij,jk,ik->i", Z, model["inv"], Z)
        unc = np.sqrt(np.maximum(leverage, 0.0)) * model["yscale"]
        return pred, unc

    def choose_starts(repaired, limit):
        ordered = sorted(repaired, key=energy)
        chosen = []
        separation = max(4, n // 2)
        for net in ordered:
            if all(distance(net, old) >= separation for old in chosen):
                chosen.append(net)
            if len(chosen) >= limit:
                break
        for net in ordered:
            if len(chosen) >= limit:
                break
            if net not in chosen:
                chosen.append(net)
        return chosen

    repaired = set()
    for cand in candidates:
        try:
            net = canon(tools.repair(cand))
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

    start_limit = min(len(repaired), max(24, 2 * n))
    initial = choose_starts(repaired, start_limit)
    for net in initial:
        if tools.lp_calls_left <= 0:
            break
        try:
            evaluate(net)
        except Exception as exc:
            if type(exc).__name__ == "BudgetExhausted":
                break
    if not seen:
        return list(initial[0])

    def archive():
        ranked = sorted(seen, key=seen.get)
        out = []
        separation = max(4, n // 2)
        diverse_limit = max(2 * n, 24)
        total_limit = max(2 * n + 8, 32)
        for net in ranked:
            if all(distance(net, old) >= separation for old in out):
                out.append(net)
            if len(out) >= diverse_limit:
                break
        for net in ranked:
            if len(out) >= total_limit:
                break
            if net not in out:
                out.append(net)
        return out

    def select_parent(pop):
        if not pop:
            return best_net
        elite = max(4, min(len(pop), n // 2))
        if rng.random() < 0.55:
            return pop[int(rng.integers(elite))]
        return pop[int(rng.integers(len(pop)))]

    def switch(net, p, q):
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

    def add_rotation(moves, current, arcs, ids):
        (a, b), (c, d), (e, f) = (arcs[int(i)] for i in ids)
        old = {(a, b), (c, d), (e, f)}
        for new in (
            {(a, d), (c, f), (e, b)},
            {(a, f), (c, b), (e, d)},
        ):
            if len(new) != 3 or any(x == y for x, y in new):
                continue
            if new & (current - old):
                continue
            moves.add(tuple(sorted((current - old) | new)))

    def make_moves(net, detail):
        arcs = list(net)
        current = set(net)
        moves = set()

        prices = detail.get("prices", {}) if detail else {}
        loads = detail.get("loads", {}) if detail else {}
        hot = {tuple(map(int, a)) for a, p in prices.items() if p > 0}
        if not hot and loads:
            ranked_loads = sorted(
                ((float(v), tuple(map(int, a))) for a, v in loads.items()),
                reverse=True,
            )
            hot = {a for _, a in ranked_loads[:max(4, n // 2)]}

        # Focused 2-switches plus a small unguided sample.
        for p in range(len(arcs)):
            for q in range(p + 1, len(arcs)):
                if hot and arcs[p] not in hot and arcs[q] not in hot:
                    continue
                cand = switch(net, p, q)
                if cand is not None:
                    moves.add(cand)

        for _ in range(5 * n):
            if len(arcs) < 2:
                break
            p = int(rng.integers(len(arcs)))
            q = int(rng.integers(len(arcs) - 1))
            if q >= p:
                q += 1
            cand = switch(net, p, q)
            if cand is not None:
                moves.add(cand)

        # Target some rotations at priced links, then sample broadly.
        if len(arcs) >= 3:
            arc_index = {a: i for i, a in enumerate(arcs)}
            if hot:
                hot_order = [a for a in hot if a in arc_index]
                hot_order.sort(
                    key=lambda a: (float(prices.get(a, 0.0)), float(loads.get(a, 0.0))),
                    reverse=True,
                )
                for t in range(min(3 * n, 120)):
                    if not hot_order:
                        break
                    p = arc_index[hot_order[t % len(hot_order)]]
                    others = [i for i in range(len(arcs)) if i != p]
                    chosen = rng.choice(len(others), size=2, replace=False)
                    add_rotation(
                        moves, current, arcs,
                        [p, others[int(chosen[0])], others[int(chosen[1])]],
                    )
            for _ in range(min(5 * n, 120)):
                ids = rng.choice(len(arcs), size=3, replace=False)
                add_rotation(moves, current, arcs, ids)

        # Multi-switch kicks provide occasional wider jumps.
        for _ in range(max(10, n)):
            cur = set(net)
            for _ in range(2 + int(rng.integers(4))):
                now = list(cur)
                p, q = rng.choice(len(now), size=2, replace=False)
                a, b = now[int(p)]
                c, d = now[int(q)]
                if a == d or c == b or (a, d) in cur or (c, b) in cur:
                    continue
                cur.remove((a, b))
                cur.remove((c, d))
                cur.add((a, d))
                cur.add((c, b))
            cand = tuple(sorted(cur))
            if cand != net:
                moves.add(cand)

        return [m for m in moves if m != net and tools.is_valid(m)]

    def screen_batch(moves, parent, detail):
        batch_size = max(8, n // 2)
        fresh = [m for m in moves if m not in seen and not tools.is_cached(m)]
        if not fresh:
            return []

        base_e = energy(parent)
        energy_ranked = sorted(fresh, key=lambda m: energy(m) - base_e)
        selected = []

        def add_diverse(net, min_dist=2):
            if net not in selected and all(distance(net, old) >= min_dist for old in selected):
                selected.append(net)
                return True
            return False

        # Preserve a strong energy-guided component.
        for net in energy_ranked:
            add_diverse(net)
            if len(selected) >= min(3, batch_size):
                break

        # Dual/load pressure provides an independent move ranking.
        if detail and len(selected) < batch_size:
            loads = {
                tuple(map(int, a)): float(v)
                for a, v in detail.get("loads", {}).items()
            }
            prices = {
                tuple(map(int, a)): float(v)
                for a, v in detail.get("prices", {}).items()
            }
            max_load = max(loads.values(), default=1.0) or 1.0
            max_price = max(prices.values(), default=1.0) or 1.0
            max_dem = max(float(np.max(demand)), 1.0)
            parent_set = set(parent)
            pressure = []
            for net in fresh:
                removed = parent_set - set(net)
                added = set(net) - parent_set
                value = sum(
                    0.60 * loads.get(a, 0.0) / max_load
                    + 0.40 * prices.get(a, 0.0) / max_price
                    for a in removed
                )
                value -= sum(0.35 * float(demand[a[0], a[1]]) / max_dem for a in added)
                pressure.append((value, net))
            pressure.sort(reverse=True, key=lambda x: x[0])
            for _, net in pressure:
                add_diverse(net)
                if len(selected) >= min(batch_size, 5):
                    break

        # The model supplements, rather than replacing, the existing screening.
        if len(seen) >= 18:
            if rounds % 4 == 1:
                fit_model()
            if model["ready"] and len(selected) < batch_size:
                pred, unc = model_values(fresh)
                best_pred = float(np.min(pred))
                # Limit uncertainty exploration to candidates with competitive
                # predicted quality; then prefer high leverage / uncertainty.
                eligible = [
                    i for i in range(len(fresh))
                    if pred[i] <= best_pred + 0.9 * max(model["yscale"], 1.0)
                    and fresh[i] not in selected
                ]
                if eligible:
                    idx = max(
                        eligible,
                        key=lambda i: (unc[i] - 0.12 * (pred[i] - best_pred)),
                    )
                    add_diverse(fresh[idx], min_dist=1)

        # Novelty slot relative to all scored networks.
        if len(selected) < batch_size and seen:
            history = list(seen)
            probe_count = min(len(fresh), max(20, 2 * n))
            if len(fresh) > probe_count:
                ids = rng.choice(len(fresh), size=probe_count, replace=False)
                probes = [fresh[int(i)] for i in ids]
            else:
                probes = fresh
            remaining = [m for m in probes if m not in selected]
            if remaining:
                novelty_net = max(
                    remaining,
                    key=lambda m: min((distance(m, old) for old in history), default=4 * n),
                )
                add_diverse(novelty_net, min_dist=1)

        # Fill remaining slots with energy-ranked and random-tail moves.
        for net in energy_ranked:
            if len(selected) >= batch_size:
                break
            if net not in selected:
                selected.append(net)
        if len(selected) < batch_size:
            tail = [m for m in fresh if m not in selected]
            rng.shuffle(tail)
            selected.extend(tail[:batch_size - len(selected)])
        return selected[:batch_size]

    while tools.lp_calls_left > 0:
        rounds += 1
        pop = archive()
        parent = select_parent(pop)
        if parent is None:
            break
        try:
            detail = tools.lp_detail(parent)
        except Exception:
            detail = None

        batch = screen_batch(make_moves(parent, detail), parent, detail)
        if not batch:
            alternatives = [x for x in pop if x != parent]
            if alternatives:
                parent = alternatives[int(rng.integers(len(alternatives)))]
                try:
                    detail = tools.lp_detail(parent)
                except Exception:
                    detail = None
                batch = screen_batch(make_moves(parent, detail), parent, detail)

        if not batch:
            # Avoid spinning if this parent has no unscored neighborhood.
            for _ in range(30):
                cur = set(parent)
                for _ in range(3 + int(rng.integers(4))):
                    now = list(cur)
                    p, q = rng.choice(len(now), size=2, replace=False)
                    a, b = now[int(p)]
                    c, d = now[int(q)]
                    if a == d or c == b or (a, d) in cur or (c, b) in cur:
                        continue
                    cur.remove((a, b))
                    cur.remove((c, d))
                    cur.add((a, d))
                    cur.add((c, b))
                kicked = tuple(sorted(cur))
                if kicked != parent and kicked not in seen and not tools.is_cached(kicked):
                    batch = [kicked]
                    break
        if not batch:
            if rounds > 10000:
                break
            continue

        parent_score = seen.get(parent, float("inf"))
        for net in batch:
            if tools.lp_calls_left <= 0:
                break
            try:
                evaluate(net)
            except Exception as exc:
                if type(exc).__name__ == "BudgetExhausted":
                    return list(best_net) if best_net is not None else []
        batch_best = min((seen.get(net, float("inf")) for net in batch), default=float("inf"))
        if batch_best < parent_score:
            parent_stalls[parent] = 0
        else:
            parent_stalls[parent] = parent_stalls.get(parent, 0) + 1

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
