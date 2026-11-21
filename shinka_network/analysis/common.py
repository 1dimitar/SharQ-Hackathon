"""Shared helpers for the analysis scripts: load an evolved program and the qoblib_tools baseline."""

from __future__ import annotations

import importlib.util
import sys
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # shinka_network/ (problem.py)
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402

import problem  # noqa: E402  (also puts the repository root on sys.path)
import qoblib_tools as qt  # noqa: E402


def load_surrogate(program_path: str):
    """surrogate_terms(demand) from an evolved program (e.g. results/<run>/best/main.py)."""
    spec = importlib.util.spec_from_file_location("evolved_program", program_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return lambda demand: module.surrogate_terms(np.array(demand, dtype=float))


def baseline_terms(demand):
    """The surrogate of qoblib_tools.build_network_qubo, in surrogate_terms format."""
    objective, _, arcs = qt.build_network_qubo(demand)
    return {arcs[v]: w for v, w in objective.linear.items()}, {}, 1.0


def published_optimum(n: int) -> list[tuple[int, int]]:
    """Arcs of QOBLIB's published optimal topology (networks 05-10 are proven optimal)."""
    path = qt.fetch_qoblib_file(f"08-network/solutions/network{n:02d}.opt.sol")
    _, arcs, _ = qt.read_network_solution(path)
    return sorted(map(tuple, arcs))


def energy(terms, arcs) -> float:
    """Surrogate energy of a topology (the degree penalty is 0 on valid topologies)."""
    linear, quadratic, _ = terms
    chosen = set(arcs)
    return sum(linear.get(a, 0.0) for a in chosen) + sum(
        w for (a, b), w in quadratic.items() if a in chosen and b in chosen
    )
