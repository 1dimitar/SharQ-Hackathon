"""Run the recorded QAOA and classical-search configurations used by main.ipynb."""

import json
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import numpy as np
import qoblib_tools as qt

from .shinka_postproc import ckpt_A_gen20, harvest_A_gen20, harvest_A_gen33, init_run1_gen49
from .shinka_postproc.make_pools import build_qubo, frozen_surrogate, sample_partitioned_qaoa
from .shinka_postproc.oracle import BudgetExhausted, Tools


NETWORK_RUNS = Path(__file__).parent / "shinka_network" / "results"
BUILDERS = {
    "r2g181": NETWORK_RUNS / "net10_luna_run2_200gen/gen_181/main.py",
    "r3g128": NETWORK_RUNS / "net12_qaoa_cloud_run3/gen_128/main.py",
}


@dataclass(frozen=True)
class Recipe:
    pool: str
    search: ModuleType
    budget: int

    @property
    def builder(self):
        return self.pool.split("_")[1]

    @property
    def search_name(self):
        return self.search.__name__.split(".")[-1]


# Pool, program and nominal LP budget from the winning QAOA-seeded runs.
RECIPES = {
    recipe.pool.split("_")[0]: recipe for recipe in (
        Recipe("network11_r3g128_c16p2loc_seed0", init_run1_gen49, 10_000),
        Recipe("network12_r3g128_c20p2_seed0", init_run1_gen49, 10_000),
        Recipe("network13_r3g128_seed0", init_run1_gen49, 20_000),
        Recipe("network14_r3g128_seed1", harvest_A_gen20, 20_000),
        Recipe("network15_r3g128_seed0", ckpt_A_gen20, 10_000),
        Recipe("network16_r2g181_seed1", harvest_A_gen20, 20_000),
        Recipe("network17_r2g181_seed1", harvest_A_gen33, 25_000),
        Recipe("network18_r3g128_seed1", ckpt_A_gen20, 10_000),
        Recipe("network19_r2g181_seed1", harvest_A_gen33, 25_000),
        Recipe("network20_r3g128_seed1", harvest_A_gen20, 20_000),
        Recipe("network21_r3g128_c16p2loc_seed0", init_run1_gen49, 10_000),
        Recipe("network22_r3g128_seed0", harvest_A_gen20, 20_000),
        Recipe("network23_r2g181_seed0", harvest_A_gen20, 20_000),
        Recipe("network24_r3g128_seed0", harvest_A_gen20, 25_000),
    )
}


def candidate_pool(demand, recipe, pool_dir, fresh, backend):
    pool = json.loads((pool_dir / f"{recipe.pool}.json").read_text())
    if pool["instance"] != f"network{len(demand):02d}" or pool["hamiltonian_tag"] != recipe.builder:
        raise ValueError(f"Pool {recipe.pool} does not match the selected model and demand")
    if not fresh:
        candidates = [[tuple(arc) for arc in sample["arcs"]] for sample in pool["candidates"]]
        return candidates, pool["circuits"]

    bqm, arcs = build_qubo(demand, BUILDERS[recipe.builder])
    setup = pool.get("setup", {})
    seed = pool["seed"]
    np.random.seed(seed)
    samples, circuits = sample_partitioned_qaoa(
        bqm, backend, states=setup.get("states", len(pool["candidates"])), seed=seed,
        cluster_size=setup.get("cluster_size", 12), layers=setup.get("layers", 2),
        shots=setup.get("shots", 10_000), population=setup.get("mc_population", 50),
        iterations=setup.get("mc_iterations", 10),
    )
    return [qt.decode_topology(bits, arcs) for bits, _ in samples], circuits


def run_search(demand, candidates, recipe):
    linear, quadratic, _ = frozen_surrogate(BUILDERS[recipe.builder])(demand)
    tools = Tools(demand, recipe.budget, (linear, quadratic))
    returned = None
    try:
        # The historical classical searches used seed 0, independently of the QAOA seed.
        returned = recipe.search.improve(demand.copy(), candidates, tools, np.random.default_rng(0))
    except BudgetExhausted:
        pass
    score, arcs = tools.certified_best()
    if returned is not None and qt.is_valid_topology(len(demand), returned):
        returned_score = qt.route_min_congestion(demand, returned).objective
        if returned_score < score:
            score, arcs = returned_score, tuple(sorted(returned))
    if arcs is None:
        raise RuntimeError(f"{recipe.search_name} did not find a feasible network")
    return score, arcs, tools.lp_calls_used
