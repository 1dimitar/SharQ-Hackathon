"""Shared functions for the SharQ network notebooks."""
import hybrid
from pathlib import Path
import qoblib_tools as qt
from divi.qprog.ensembles import BeamSearchStrategy, PartitioningProgramEnsemble
from divi.qprog.optimizers import MonteCarloOptimizer
from divi.qprog.problems import BinaryOptimizationProblem, CommunityDecomposer, QUBOPartitioningConfig
import math
from random import Random
from tempfile import TemporaryDirectory


def sample_layouts(qubo, backend, cluster_size, layers, seed, *,
                   population_size=8, iterations=3, beam_width=24, partition_candidates=8):
    problem = BinaryOptimizationProblem(
        qubo,
        decomposer=CommunityDecomposer(QUBOPartitioningConfig(max_n_variables_per_cluster=cluster_size)),
        composer=hybrid.SplatComposer(),
    )
    ensemble = PartitioningProgramEnsemble(
        problem=problem,
        quantum_routine="qaoa",
        n_layers=layers,
        optimizer=MonteCarloOptimizer(population_size=population_size),
        max_iterations=iterations,
        backend=backend,
        reporting_level="off",
        seed=seed,
    )
    ensemble.run()
    return ensemble.get_top_solutions(
        n=beam_width, strategy=BeamSearchStrategy(beam_width=beam_width, n_partition_candidates=partition_candidates)
    )


def evaluate_layouts(candidates, qubo, arcs, demand):
    n_nodes = len(demand)
    layout_scores = {}
    best_raw, best_arcs, best_bits = set(), [], ""
    best_score = math.inf
    for bits, _ in candidates:
        sample = dict(zip(qubo.variables, bits, strict=True))
        raw_arcs = qt.decode_topology(sample, arcs)
        if qt.is_valid_topology(n_nodes, raw_arcs):
            topology = raw_arcs
        else:
            topology = qt.repair_topology(n_nodes, raw_arcs, demand)

        layout = tuple(sorted(topology))
        if layout in layout_scores:
            continue
        score = qt.route_min_congestion(demand, layout).objective
        layout_scores[layout] = score
        if score < best_score:
            best_score = score
            best_raw = set(raw_arcs)
            best_arcs = sorted(topology)
            best_bits = "".join(str(int(sample[v])) for v in range(len(arcs)))
    return layout_scores, best_score, best_raw, best_arcs, best_bits


def refine_layout(layout, start_score, rng, demand, swaps_per_start):
    edges = set(layout)
    score = start_score
    tried = accepted = 0
    while tried < swaps_per_start:
        first, second = rng.sample(sorted(edges), 2)
        (a, b), (c, d) = first, second
        replacements = {(a, d), (c, b)}
        if a == d or c == b or replacements & edges:
            continue

        trial = (edges - {first, second}) | replacements
        trial_score = qt.route_min_congestion(demand, trial).objective
        tried += 1
        if trial_score < score:
            edges, score = trial, trial_score
            accepted += 1
    return sorted(edges), accepted


def refine_starts(layout_scores, demand, n_starts, swaps_per_start, seed):
    n_nodes = len(demand)
    instance = f"network{n_nodes:02d}"
    starts = sorted((score, layout) for layout, score in layout_scores.items() if math.isfinite(score))
    best_checked_score, best_routing = math.inf, None
    print("Start   Initial LP   Accepted   Checked score")
    with TemporaryDirectory() as scratch:
        trial_file = Path(scratch) / f"{instance}.sol"
        for rank, (start_score, layout) in enumerate(starts[:n_starts], start=1):
            refined, accepted = refine_layout(layout, start_score, Random(seed + rank - 1), demand, swaps_per_start)
            candidate_routing = qt.route_integral(demand, refined)
            qt.write_network_solution(trial_file, n_nodes, candidate_routing)
            valid, score, message = qt.check_network_solution(trial_file, n_nodes, demand)
            if not valid:
                raise ValueError(f"Start {rank} failed the solution checker: {message}")
            print(f"{rank:>5}   {start_score:>10,.0f}   {accepted:>8}   {score:>13,}")
            if score < best_checked_score:
                best_checked_score, best_routing = score, candidate_routing
    return best_checked_score, best_routing


def draw_best_saved(instance, demand):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import networkx as nx

    n_nodes = len(demand)
    # Compare only saved solutions for the active instance, in the same score units.
    paths = sorted(Path("starter_results").rglob(f"{instance}.sol"))
    project_solution = Path(f"{instance}.sol")
    if project_solution.is_file():
        paths.append(project_solution)
    verified = []
    for path in paths:
        valid, score, _ = qt.check_network_solution(path, n_nodes, demand)
        if valid:
            verified.append((score, path))
    if not verified:
        raise ValueError(f"No checked {instance}.sol found. Run a solution-writing cell first.")

    peak, best_path = min(verified)
    _, links, flows = qt.read_network_solution(best_path)
    graph = nx.DiGraph()
    graph.add_nodes_from(range(n_nodes))
    graph.add_edges_from(links, load=0)
    for (source, i, j), amount in flows.items():
        if j != source:
            graph[i][j]["load"] += amount
    assert max(load for _, _, load in graph.edges.data("load")) == peak

    positions = nx.kamada_kawai_layout(graph.to_undirected())
    fig, ax = plt.subplots(figsize=(12, 9))
    for i, j, load in graph.edges.data("load"):
        weight = load / qt.INTSCALE
        graph[i][j]["weight"] = weight
        curve = "arc3,rad=0.16" if graph.has_edge(j, i) else "arc3,rad=0"
        nx.draw_networkx_edges(
            graph, positions, edgelist=[(i, j)], ax=ax, arrows=True,
            arrowstyle="-|>", arrowsize=17, node_size=760,
            width=0.9 + 2.6 * load / max(1, peak),
            edge_color="#bb431c" if load == peak else "#557991",
            connectionstyle=curve,
        )
        label_artists = nx.draw_networkx_edge_labels(
            graph, positions, edge_labels={(i, j): f"{weight:.3f}"}, ax=ax,
            label_pos=0.28, connectionstyle=curve, node_size=760, font_size=8,
            font_color="#943314" if load == peak else "#243f50",
            bbox=dict(boxstyle="round,pad=0.14", facecolor="white", edgecolor="none", alpha=0.95),
        )
        label_artists[i, j].set_zorder(4)
    nx.draw_networkx_nodes(graph, positions, ax=ax, node_size=760,
                           node_color="#f4f8fb", edgecolors="#29475b", linewidths=1.5)
    nx.draw_networkx_labels(graph, positions, ax=ax, font_size=11, font_weight="bold", font_color="#17374b")
    ax.set_title(f"{instance}   /   lowest saved score {peak:,}", loc="left", fontsize=16, pad=16)
    ax.legend(handles=[
        Line2D([], [], color="#bb431c", lw=3.5, label=f"Busiest link ({peak / qt.INTSCALE:g} units)"),
        Line2D([], [], color="#557991", lw=2, label="Edge labels show load in traffic units"),
    ], loc="upper center", bbox_to_anchor=(0.5, -0.025), ncol=2, frameon=False)
    ax.margins(0.14)
    ax.set_aspect("equal")
    ax.set_axis_off()
    print("Checked solution:", best_path)
    print("Busiest directed links:", [(i, j) for i, j, v in graph.edges.data("load") if v == peak])
    image_path = Path("starter_results") / f"{instance}_best_network.png"
    fig.savefig(image_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.show()
