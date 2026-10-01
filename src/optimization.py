"""Defense resource optimization.

Searches over combinations of nodes to defend (within a fixed budget)
and picks the combination that minimizes the probability of
physical_equipment reaching FAILED, evaluated via Monte Carlo
simulation.
"""

import copy
import itertools

from src.ctmc import build_ctmc_nodes
from src.graph import build_smart_grid
from src.simulation import TARGET_NODE, run_monte_carlo


def apply_defense(ctmc_nodes, defended_nodes, reduction=0.5):
    """Return a copy of ctmc_nodes with compromise_rate reduced for defended_nodes.

    Each defended node's compromise_rate is scaled by (1 - reduction);
    its Q matrix is rebuilt to reflect the new rate. Nodes not in
    defended_nodes are left untouched.
    """
    defended = copy.deepcopy(ctmc_nodes)

    for name in defended_nodes:
        node = defended[name]
        node.compromise_rate *= (1 - reduction)
        node.Q = node._build_q_matrix()

    return defended


def optimize_defense(graph, ctmc_nodes, budget=2, t_max=24, n_runs=500):
    """Find the combination of up to `budget` defended nodes that
    minimizes the probability of physical_equipment failure.

    Returns a dict with 'best_nodes' (tuple of defended node names),
    'best_probability' (its failure probability), and 'results' (a
    list of (node_combination, failure_probability) for every
    combination evaluated).
    """
    node_names = list(graph.nodes())
    results = []

    for combo in itertools.combinations(node_names, budget):
        defended_ctmc_nodes = apply_defense(ctmc_nodes, combo)
        mc_results = run_monte_carlo(graph, defended_ctmc_nodes, n_runs=n_runs, t_max=t_max)
        failure_probability = mc_results["probability_of_failure"]
        results.append((combo, failure_probability))

    best_nodes, best_probability = min(results, key=lambda item: item[1])

    return {
        "best_nodes": best_nodes,
        "best_probability": best_probability,
        "results": results,
    }


if __name__ == "__main__":
    smart_grid = build_smart_grid()
    ctmc_nodes = build_ctmc_nodes(smart_grid)

    baseline = run_monte_carlo(smart_grid, ctmc_nodes, n_runs=500, t_max=24)
    baseline_probability = baseline["probability_of_failure"]

    optimization_result = optimize_defense(smart_grid, ctmc_nodes, budget=2, t_max=24, n_runs=500)
    best_nodes = optimization_result["best_nodes"]
    best_probability = optimization_result["best_probability"]

    improvement_pct = (
        (baseline_probability - best_probability) / baseline_probability * 100
        if baseline_probability > 0
        else 0.0
    )

    print("Defense optimization results:")
    print(f"  Baseline P({TARGET_NODE} fails), no defense = {baseline_probability:.4f}")
    print(f"  Best defended nodes = {best_nodes}")
    print(f"  P({TARGET_NODE} fails) with optimal defense = {best_probability:.4f}")
    print(f"  Improvement = {improvement_pct:.2f}%")
