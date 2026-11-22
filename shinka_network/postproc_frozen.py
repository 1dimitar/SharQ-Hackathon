"""Frozen classical post-processor used INSIDE the surrogate evaluation (never evolved here).

Copy of the evolved block of shinka_postproc/init_run1_gen49.py (post-processing run 1, generation 49):
repairs the candidates, LP-scores about 30 energy-ranked, mutually different starts, then runs a
population search whose 2-exchange moves are ranked with tools.energy before LP calls are spent.
"""

import json  # noqa: F401
import time  # noqa: F401

import numpy as np  # noqa: F401

import math


def improve(demand, candidates, tools, rng):
    """Search repaired QAOA networks using a surrogate-screened diverse archive."""
    n = tools.n
    seen = {}
    best_net = None
    best_score = float("inf")
    parent_stalls = {}

    def canon(arcs):
        return tuple(sorted(tuple(map(int, a)) for a in arcs))

    def evaluate(net):
        """Score once locally; budget exhaustion is allowed to reach the caller."""
        nonlocal best_net, best_score
        net = canon(net)
        if net not in seen:
            seen[net] = tools.lp(net)
        s = seen[net]
        if s < best_score:
            best_net, best_score = net, s
        return s

    def energy(net):
        try:
            return float(tools.energy(net))
        except Exception:
            return 0.0

    def distance(a, b):
        return len(set(a) ^ set(b))

    def choose_starts(repaired, limit):
        """Prefer low surrogate energy, while retaining topological variety."""
        ordered = sorted(repaired, key=energy)
        chosen = []
        for net in ordered:
            if all(distance(net, old) >= 4 for old in chosen):
                chosen.append(net)
            if len(chosen) >= limit:
                break
        for net in ordered:
            if len(chosen) >= limit:
                break
            if net not in chosen:
                chosen.append(net)
        return chosen

    # Repair all raw samples for free, then seed the search with a compact,
    # energy-ranked but diversity-aware subset.
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

    initial = choose_starts(repaired, min(30, len(repaired)))
    for net in initial:
        if tools.lp_calls_left <= 0:
            break
        try:
            evaluate(net)
        except Exception as exc:
            if type(exc).__name__ == "BudgetExhausted":
                break
            continue

    if not seen:
        return list(initial[0])

    def archive():
        """Keep excellent networks and the best representatives of distinct regions."""
        ranked = sorted(seen, key=seen.get)
        out = []
        for net in ranked:
            if all(distance(net, old) >= 4 for old in out):
                out.append(net)
            if len(out) >= 24:
                break
        for net in ranked:
            if len(out) >= 32:
                break
            if net not in out:
                out.append(net)
        return out

    def select_parent(pop):
        """Mix elite exploitation with sampling from diverse archived networks."""
        if not pop:
            return best_net
        if rng.random() < 0.48:
            return pop[int(rng.integers(min(6, len(pop))))]
        return pop[int(rng.integers(len(pop)))]

    def switch(net, p, q):
        """Swap the destinations of two links, preserving all degrees."""
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

    def make_moves(net, detail):
        """Build a focused two-switch neighborhood plus sampled rotations and kicks."""
        arcs = list(net)
        current = set(net)
        moves = set()

        prices = detail.get("prices", {}) if detail else {}
        hot = {tuple(map(int, a)) for a, p in prices.items() if p > 0}
        if not hot and detail:
            loads = detail.get("loads", {})
            ranked_loads = sorted(
                ((float(v), tuple(map(int, a))) for a, v in loads.items()),
                reverse=True,
            )
            hot = {a for _, a in ranked_loads[:6]}

        # Exhaustively explore swaps involving at least one critical link.
        for p in range(len(arcs)):
            for q in range(p + 1, len(arcs)):
                if hot and arcs[p] not in hot and arcs[q] not in hot:
                    continue
                cand = switch(net, p, q)
                if cand is not None:
                    moves.add(cand)

        # Add non-focused swaps so that dual degeneracy cannot trap the search.
        for _ in range(70):
            if len(arcs) < 2:
                break
            p = int(rng.integers(len(arcs)))
            q = int(rng.integers(len(arcs) - 1))
            if q >= p:
                q += 1
            cand = switch(net, p, q)
            if cand is not None:
                moves.add(cand)

        # Escalate rotation sampling for parents whose neighborhoods have stalled.
        # Rotations preserve degrees and can cross barriers that two-switches miss.
        if len(arcs) >= 3:
            stalls = parent_stalls.get(net, 0)
            rotation_trials = min(350, 70 * (1 + stalls // 3))
            for _ in range(rotation_trials):
                ids = rng.choice(len(arcs), size=3, replace=False)
                (a, b), (c, d), (e, f) = (arcs[int(i)] for i in ids)
                old = {(a, b), (c, d), (e, f)}
                if hot and not (old & hot):
                    continue
                for new in (
                    {(a, d), (c, f), (e, b)},
                    {(a, f), (c, b), (e, d)},
                ):
                    if len(new) != 3 or any(x == y for x, y in new):
                        continue
                    if new & (current - old):
                        continue
                    cand = tuple(sorted((current - old) | new))
                    moves.add(cand)

        # Occasional multi-switch kicks broaden the neighborhood beyond one step.
        for _ in range(24):
            cur = set(net)
            for _ in range(2 + int(rng.integers(4))):
                arcs_now = list(cur)
                if len(arcs_now) < 2:
                    break
                p, q = rng.choice(len(arcs_now), size=2, replace=False)
                a, b = arcs_now[int(p)]
                c, d = arcs_now[int(q)]
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

    def screen_batch(moves, parent, batch_size=9):
        """Use surrogate ranking and topology distance to select an LP batch."""
        fresh = [
            m for m in moves
            if m not in seen and not tools.is_cached(m)
        ]
        if not fresh:
            return []

        base_e = energy(parent)
        ranked = []
        for net in fresh:
            ranked.append((energy(net) - base_e, net))
        ranked.sort(key=lambda item: item[0])

        selected = []
        # First take good-energy moves that are not all tiny variants of one another.
        for _, net in ranked:
            if all(distance(net, old) >= 2 for old in selected):
                selected.append(net)
            if len(selected) >= min(6, batch_size):
                break

        # Fill remaining slots with the best surrogate-ranked moves.
        for _, net in ranked:
            if len(selected) >= batch_size:
                break
            if net not in selected:
                selected.append(net)

        # Reserve a little batch capacity for tail exploration.
        tail = [net for _, net in ranked if net not in selected]
        if tail and len(selected) >= 4 and len(selected) < batch_size:
            rng.shuffle(tail)
            selected.extend(tail[:batch_size - len(selected)])
        return selected[:batch_size]

    # A bounded archive drives a sequence of short, diverse neighborhood searches.
    rounds = 0
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

        moves = make_moves(parent, detail)
        batch = screen_batch(moves, parent)

        # If a neighborhood has already been exhausted, perturb another archive
        # member rather than repeatedly revisiting the same parent.
        if not batch:
            alternatives = [x for x in pop if x != parent]
            if alternatives:
                parent = alternatives[int(rng.integers(len(alternatives)))]
                try:
                    detail = tools.lp_detail(parent)
                except Exception:
                    detail = None
                batch = screen_batch(make_moves(parent, detail), parent)

        if not batch:
            # A final random degree-preserving kick guarantees progress when possible.
            for _ in range(40):
                cur = set(parent)
                arcs = list(cur)
                for _ in range(3 + int(rng.integers(4))):
                    p, q = rng.choice(len(arcs), size=2, replace=False)
                    a, b = arcs[int(p)]
                    c, d = arcs[int(q)]
                    if a == d or c == b or (a, d) in cur or (c, b) in cur:
                        continue
                    cur.remove((a, b))
                    cur.remove((c, d))
                    cur.add((a, d))
                    cur.add((c, b))
                    arcs = list(cur)
                kicked = tuple(sorted(cur))
                if kicked != parent and kicked not in seen and not tools.is_cached(kicked):
                    batch = [kicked]
                    break

        if not batch:
            # All available moves may be cached; avoid an unbounded spin.
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
                continue

        batch_best = min((seen.get(net, float("inf")) for net in batch), default=float("inf"))
        if batch_best < parent_score:
            parent_stalls[parent] = 0
        else:
            parent_stalls[parent] = parent_stalls.get(parent, 0) + 1

    return list(best_net) if best_net is not None else []
