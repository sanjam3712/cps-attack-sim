"""Entry point for the CPS cyberattack propagation simulation.

Builds the smart grid graph, runs a Monte Carlo attack simulation and
defense resource optimization, prints a summary of the results, and
generates all visualizations under results/.
"""

from src.ctmc import build_ctmc_nodes
from src.graph import build_smart_grid
from src.optimization import optimize_defense
from src.simulation import TARGET_NODE, run_monte_carlo
from src.visualization import (
    RESULTS_DIR,
    plot_attack_propagation,
    plot_defense_comparison,
    plot_failure_distribution,
    plot_graph,
)
from src.simulation import run_attack_simulation


def print_header(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def main():
    print_header("1. Building smart grid graph")
    graph = build_smart_grid()
    print(f"  Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}")

    print_header("2. Building CTMC nodes")
    ctmc_nodes = build_ctmc_nodes(graph)
    print(f"  Built CTMC models for {len(ctmc_nodes)} nodes")

    print_header("3. Running Monte Carlo simulation (1000 runs)")
    monte_carlo_result = run_monte_carlo(graph, ctmc_nodes, n_runs=1000, t_max=24)
    baseline_probability = monte_carlo_result["probability_of_failure"]
    mean_time_to_failure = monte_carlo_result["mean_time_to_failure"]
    print(f"  P({TARGET_NODE} fails)       = {baseline_probability:.4f}")
    if mean_time_to_failure is not None:
        print(f"  Mean time to failure         = {mean_time_to_failure:.2f}h")
    else:
        print("  Mean time to failure         = N/A (no failures observed)")
    print(f"  Failed runs                  = {len(monte_carlo_result['failure_times'])} / 1000")

    print_header("4. Running defense optimization (budget=2)")
    optimization_result = optimize_defense(graph, ctmc_nodes, budget=2, t_max=24, n_runs=500)
    best_nodes = optimization_result["best_nodes"]
    best_probability = optimization_result["best_probability"]
    improvement_pct = (
        (baseline_probability - best_probability) / baseline_probability * 100
        if baseline_probability > 0
        else 0.0
    )
    print(f"  Combinations evaluated       = {len(optimization_result['results'])}")
    print(f"  Best defended nodes          = {best_nodes}")
    print(f"  P({TARGET_NODE} fails) w/ defense = {best_probability:.4f}")
    print(f"  Improvement over baseline    = {improvement_pct:.2f}%")

    print_header("5. Summary")
    print(f"  Target node                  = {TARGET_NODE}")
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
    plot_failure_distribution(monte_carlo_result)
    print("  Saved failure_distribution.png")
    plot_defense_comparison(baseline_probability, optimization_result)
    print("  Saved defense_comparison.png")

    print(f"\nDone. Results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
