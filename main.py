"""Entry point for the CPS cyberattack propagation simulation.

Loads a system config (nodes, edges, entry points, target), runs a
Monte Carlo attack simulation and defense resource optimization,
prints a summary of the results, and generates all visualizations
under results/.
"""

import argparse

from src.ctmc import build_ctmc_nodes
from src.graph import load_system
from src.optimization import optimize_defense, print_ranking
from src.visualization import (
    RESULTS_DIR,
    plot_attack_propagation,
    plot_defense_comparison,
    plot_failure_distribution,
    plot_graph,
)
from src.simulation import run_attack_simulation, run_monte_carlo


def print_header(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", default="configs/substation.json", help="Path to a system config JSON file")
    parser.add_argument("--budget", type=int, default=2, help="Defense budget (number of nodes to defend)")
    parser.add_argument("--reduction", type=float, default=0.5, help="Fractional compromise_rate reduction for defended nodes")
    parser.add_argument("--n-runs", type=int, default=5000, help="Number of Monte Carlo runs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()
    n_runs = args.n_runs
    seed = args.seed
    budget = args.budget
    reduction = args.reduction

    print_header(f"1. Building graph from {args.system}")
    graph = load_system(args.system)
    target = graph.graph["target"]
    print(f"  Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}, Target: {target}")

    print_header("2. Building CTMC nodes")
    ctmc_nodes = build_ctmc_nodes(graph)
    print(f"  Built CTMC models for {len(ctmc_nodes)} nodes")

    print_header(f"3. Running Monte Carlo simulation ({n_runs} runs, seed={seed})")
    monte_carlo_result = run_monte_carlo(
        graph, ctmc_nodes, n_runs=n_runs, t_max=24, seed=seed
    )
    baseline_probability = monte_carlo_result["probability_of_failure"]
    mean_time_to_failure = monte_carlo_result["mean_time_to_failure"]
    print(f"  P({target} fails)       = {baseline_probability:.4f}")
    if mean_time_to_failure is not None:
        print(f"  Mean time to failure         = {mean_time_to_failure:.2f}h")
    else:
        print("  Mean time to failure         = N/A (no failures observed)")
    print(f"  Failed runs                  = {len(monte_carlo_result['failure_times'])} / {n_runs}")

    print_header(f"4. Running defense optimization (budget={budget}, {target} excluded - recommended)")
    optimization_result = optimize_defense(
        graph, ctmc_nodes, budget=budget, t_max=24,
        n_runs=n_runs, seed=seed, excluded_nodes=[target], reduction=reduction,
    )
    best_nodes = optimization_result["best_nodes"]
    best_probability = optimization_result["best_probability"]
    improvement_pct = (
        (baseline_probability - best_probability) / baseline_probability * 100
        if baseline_probability > 0
        else 0.0
    )
    print(f"  Combinations evaluated       = {len(optimization_result['results'])}")
    print(f"  Best defended nodes          = {best_nodes}")
    print(f"  P({target} fails) w/ defense = {best_probability:.4f}")
    print(f"  Improvement over baseline    = {improvement_pct:.2f}%")
    print_ranking(
        optimization_result["results_detailed"],
        f"  Top 5 (budget={budget}, {target} excluded):",
    )

    print_header(f"4b. Naive ranking (budget={budget}, {target} DEFENDABLE - trivial, for comparison only)")
    naive_result = optimize_defense(
        graph, ctmc_nodes, budget=budget, t_max=24,
        n_runs=n_runs, seed=seed, excluded_nodes=None, reduction=reduction,
    )
    print_ranking(
        naive_result["results_detailed"],
        f"  Top 5 (budget={budget}, {target} defendable - NAIVE, includes trivial target-defense):",
    )

    print_header(f"4c. Defense optimization (budget={budget + 1}, {target} excluded)")
    budget_plus_result = optimize_defense(
        graph, ctmc_nodes, budget=budget + 1, t_max=24,
        n_runs=n_runs, seed=seed, excluded_nodes=[target], reduction=reduction,
    )
    print_ranking(
        budget_plus_result["results_detailed"],
        f"  Top 5 (budget={budget + 1}, {target} excluded):",
    )

    print_header("5. Summary")
    print(f"  Target node                  = {target}")
    print(f"  Baseline failure probability = {baseline_probability:.4f}")
    print(f"  Optimized failure probability= {best_probability:.4f}")
    print(f"  Best defense combination     = {best_nodes}")
    print(f"  Improvement                  = {improvement_pct:.2f}%")

    print_header("6. Generating plots")
    simulation_result = run_attack_simulation(graph, ctmc_nodes, t_max=24, t_steps=100)
    plot_graph(graph)
    print("  Saved graph.png")
    plot_attack_propagation(simulation_result)
    print("  Saved attack_propagation.png")
    plot_failure_distribution(monte_carlo_result, target_node=target)
    print("  Saved failure_distribution.png")
    plot_defense_comparison(
        baseline_probability, optimization_result, target_node=target,
        title=f"Defense Resource Allocation Comparison (budget={budget}, {target} excluded - recommended)",
        filename="defense_comparison.png",
    )
    print("  Saved defense_comparison.png")
    plot_defense_comparison(
        baseline_probability, naive_result, target_node=target,
        title=f"Defense Resource Allocation Comparison (budget={budget}, {target} defendable - NAIVE)",
        filename="defense_comparison_naive.png",
    )
    print("  Saved defense_comparison_naive.png")

    print(f"\nDone. Results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
