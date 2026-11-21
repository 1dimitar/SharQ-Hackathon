"""Fixed tools handed to the evolved post-processor (ShinkaEvolve never edits this file).

The evolved `improve(demand, candidates, tools, rng)` may use only these tools to talk to the
routing LP. Every distinct network scored costs one LP call from a fixed budget; repeated
networks are free (cached). The oracle itself records the best network it has scored, so the
result of a run is whatever the LP actually certified, not what the program claims.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repository root (qoblib_tools)
import qoblib_tools as qt  # noqa: E402
from scipy.optimize import linprog  # noqa: E402


class BudgetExhausted(BaseException):
    """Raised by tools.lp / tools.lp_detail once the LP budget is spent.

    A BaseException, so a generic `except Exception` in evolved code cannot swallow it."""


class RunTimeout(BaseException):
    """Raised by the evaluator's alarm when a run exceeds its wall-time limit."""


def _key(arcs) -> tuple:
    return tuple(sorted((int(i), int(j)) for i, j in arcs))


class Tools:
    def __init__(self, demand: np.ndarray, lp_budget: int, energy_terms=None):
        self.n = int(len(demand))
        self._demand = np.array(demand, dtype=np.int64)
        self._budget = int(lp_budget)
        self._used = 0
        self._cache: dict[tuple, dict] = {}
        self._best_score = math.inf
        self._best_arcs: tuple | None = None
        self._energy_terms = energy_terms  # (linear dict, quadratic dict) of the frozen Hamiltonian

    # ------------------------------------------------------------------ budget
    @property
    def lp_calls_used(self) -> int:
        return self._used

    @property
    def lp_calls_left(self) -> int:
        return self._budget - self._used

    # ------------------------------------------------------------------ free helpers
    def is_valid(self, arcs) -> bool:
        """Every node has exactly 2 outgoing and 2 incoming links, no self-loops."""
        return qt.is_valid_topology(self.n, _key(arcs))

    def repair(self, arcs) -> list[tuple[int, int]]:
        """Closest valid network: keeps as many of `arcs` as possible (exact matching, free)."""
        return qt.repair_topology(self.n, _key(arcs), self._demand)

    def energy(self, arcs) -> float:
        """Frozen Hamiltonian's surrogate energy of a link set (free, lower = predicted better)."""
        lin, quad = self._energy_terms
        s = set(_key(arcs))
        return float(sum(lin.get(a, 0.0) for a in s) + sum(w for (a, b), w in quad.items() if a in s and b in s))

    def is_cached(self, arcs) -> bool:
        """True if this network was already scored (scoring it again is free)."""
        return _key(arcs) in self._cache

    # ------------------------------------------------------------------ LP (budgeted)
    def lp_detail(self, arcs) -> dict:
        """Route all traffic on a valid network. One LP call unless cached.

        Returns {"score": QOBLIB objective (max link load, lower is better),
                 "loads": {link: total flow on it},
                 "prices": {link: dual price of its capacity, > 0 only on bottleneck links}}.
        Invalid networks return score = inf at no cost.
        """
        key = _key(arcs)
        if key in self._cache:
            return self._copy(self._cache[key])
        if not qt.is_valid_topology(self.n, key):
            return {"score": math.inf, "loads": {}, "prices": {}}
        if self._used >= self._budget:
            raise BudgetExhausted(f"LP budget of {self._budget} calls spent")
        self._used += 1
        var, c, A_ub, A_eq, b_eq = qt._routing_lp(self._demand, list(key))
        res = linprog(c, A_ub=A_ub, b_ub=np.zeros(A_ub.shape[0]), A_eq=A_eq, b_eq=b_eq,
                      bounds=(0, None), method="highs")
        if res.status != 0:
            out = {"score": math.inf, "loads": {}, "prices": {}}
        else:
            flows = res.x
            loads = {a: 0.0 for a in key}
            for (k, idx), col in var.items():
                loads[key[idx]] += float(flows[col])
            prices = {a: float(-res.ineqlin.marginals[idx]) for idx, a in enumerate(key)}
            out = {"score": float(math.ceil(res.fun - 1e-6)), "loads": loads, "prices": prices}
        self._cache[key] = out
        if out["score"] < self._best_score:
            self._best_score, self._best_arcs = out["score"], key
        return self._copy(out)

    @staticmethod
    def _copy(out: dict) -> dict:
        return {"score": out["score"], "loads": dict(out["loads"]), "prices": dict(out["prices"])}

    def lp(self, arcs) -> float:
        """Score of a valid network (max link load). One LP call unless cached; inf if invalid."""
        return self.lp_detail(arcs)["score"]

    # ------------------------------------------------------------------ evaluator side
    def certified_best(self) -> tuple[float, tuple | None]:
        return self._best_score, self._best_arcs
