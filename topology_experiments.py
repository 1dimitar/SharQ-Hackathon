"""Demand and routing surrogates, and degree-preserving local search for QOBLIB."""
import math
from itertools import permutations
from random import Random

import numpy as np
from scipy.optimize import linprog

import qoblib_tools as qt


def topology_model(demand, two_hop=False, concentration=False, feedback_layout=None):
    reward, penalty, arcs = qt.build_network_qubo(demand)
    model = reward + 2.0 * penalty
    bit = {arc: v for v, arc in enumerate(arcs)}
    traffic = demand / max(demand.max(), 1)
    if two_hop or concentration:
        for i, j, k in permutations(range(len(demand)), 3):
            w = traffic[i, k]
            coefficient = -0.5 * w if two_hop else 0.0
            if concentration:
                # Proxy for the increase in squared load on both legs if w
                # joins their direct traffic. The routing LP handles actual splitting.
                coefficient += 0.2 * (2 * w * (traffic[i, j] + traffic[j, k]) + 2 * w**2)
            if coefficient:
                model.add_quadratic(bit[i, j], bit[j, k], coefficient)

    if feedback_layout is not None:
        layout = sorted(feedback_layout)
        _, cost, capacity, balance, rhs = qt._routing_lp(demand, layout)
        lp = linprog(cost, A_ub=capacity, b_ub=np.zeros(len(layout)),
                     A_eq=balance, b_eq=rhs, bounds=(0, None), method="highs")
        if not lp.success:
            raise ValueError(f"Feedback routing LP failed: {lp.message}")
        # A large shadow price identifies a link where spare capacity is valuable.
        prices = dict(zip(layout, -lp.ineqlin.marginals))
        scale = max(prices.values(), default=0.0)
        if scale > 0:
            for i, k in arcs:
                if (i, k) in prices:
                    continue
                bypass = max((prices[i, j] + prices[j, k] for j in range(len(demand))
                              if (i, j) in prices and (j, k) in prices), default=0.0)
                model.add_linear(bit[i, k], -0.5 * traffic[i, k] * bypass / scale)
    return model, arcs


def improve_layout(demand, layout, score, seed, trials=200, anneal=False, three_swap=False):
    rng = Random(seed)
    edges = set(layout)
    best_edges, best_score = edges, score
    # Starting temperature is 1% of peak load, cooling geometrically to 0.01%.
    initial_temperature = max(1.0, score * 0.01)
    for step in range(trials):
        while True:
            count = 3 if three_swap and rng.random() < 0.4 else 2
            selected = rng.sample(sorted(edges), count)
            sources, targets = zip(*selected)
            if len(set(sources)) != count or len(set(targets)) != count:
                continue
            new = set(zip(sources, targets[1:] + targets[:1]))
            if not new & edges and all(i != j for i, j in new):
                break
        trial = edges.difference(selected) | new
        candidate_score = qt.route_min_congestion(demand, trial).objective
        delta = candidate_score - score
        temperature = initial_temperature * 0.01 ** (step / max(1, trials - 1))
        if delta < 0 or (anneal and rng.random() < math.exp(-delta / temperature)):
            edges, score = trial, candidate_score
            if score < best_score:
                best_edges, best_score = edges, score
    return best_edges, best_score
