"""Plotting and visualization.

Plotting utilities (networkx/matplotlib) for visualizing the smart
grid graph, attack propagation over time, Monte Carlo failure-time
distribution, and the effects of defense resource optimization. Every
plot function saves its figure as a PNG under results/.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from src.ctmc import SECURE, COMPROMISED, FAILED, STATE_NAMES, build_ctmc_nodes
from src.graph import build_smart_grid
from src.optimization import DEFAULT_N_RUNS, DEFAULT_SEED, optimize_defense
from src.simulation import TARGET_NODE, run_attack_simulation, run_monte_carlo

RESULTS_DIR = "results"


def _abbreviate(node_name):
    """Shorten a node name to its word initials, e.g. 'scada_server' -> 'SS'."""
    return "".join(word[0].upper() for word in node_name.split("_"))


def plot_graph(graph):
    """Draw the smart grid topology, coloring nodes by criticality.

    Low criticality is green, high criticality is red. Node labels are
    shown. Saves to results/graph.png.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    criticalities = [graph.nodes[n]["criticality"] for n in graph.nodes()]
    cmap = plt.cm.RdYlGn_r

    fig, ax = plt.subplots(figsize=(11, 8))
    pos = nx.spring_layout(graph, seed=7, k=1.8, iterations=200)

    nodes = nx.draw_networkx_nodes(
        graph, pos, ax=ax,
        node_color=criticalities, cmap=cmap, vmin=0, vmax=1,
        node_size=2200, edgecolors="black",
    )
    nx.draw_networkx_edges(
        graph, pos, ax=ax, arrowstyle="-|>", arrowsize=20,
        edge_color="gray", connectionstyle="arc3,rad=0.05",
    )
    nx.draw_networkx_labels(graph, pos, ax=ax, font_size=8)

    fig.colorbar(nodes, ax=ax, label="Criticality")
    ax.set_title("Smart Grid Attack Graph (colored by criticality)")
    ax.margins(0.15)
    ax.axis("off")

    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "graph.png"), dpi=150)
    plt.close(fig)


def plot_attack_propagation(simulation_result):
    """Plot the state of every node over time for one attack simulation.

    0=SECURE, 1=COMPROMISED, 2=FAILED. Saves to
    results/attack_propagation.png.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    time = simulation_result["time"]
    node_states = simulation_result["node_states"]

    fig, ax = plt.subplots(figsize=(11, 6))
    cmap = plt.cm.tab10
    for i, (name, states) in enumerate(node_states.items()):
        ax.plot(
            time, states, label=name,
            color=cmap(i % 10), linewidth=2, alpha=0.85,
            marker="", drawstyle="steps-post",
        )

    ax.set_yticks([SECURE, COMPROMISED, FAILED])
    ax.set_yticklabels(STATE_NAMES)
    ax.set_ylim(-0.3, 2.3)
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("State")
    ax.set_title("Attack Propagation Over Time")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "attack_propagation.png"), dpi=150)
    plt.close(fig)


def plot_failure_distribution(monte_carlo_result):
    """Plot a histogram of Monte Carlo failure times with the mean marked.

    Saves to results/failure_distribution.png.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    failure_times = monte_carlo_result["failure_times"]
    mean_time = monte_carlo_result["mean_time_to_failure"]

    fig, ax = plt.subplots(figsize=(9, 6))

    if failure_times:
        ax.hist(failure_times, bins=30, color="steelblue", edgecolor="black", alpha=0.8)
        ax.axvline(
            mean_time, color="red", linestyle="--", linewidth=2,
            label=f"Mean time to failure = {mean_time:.2f}h",
        )
        ax.legend()
    else:
        ax.text(
            0.5, 0.5, "No failures observed", ha="center", va="center",
            transform=ax.transAxes, fontsize=14,
        )

    ax.set_xlabel("Time to failure (hours)")
    ax.set_ylabel("Number of runs")
    ax.set_title(
        f"Distribution of {TARGET_NODE} Failure Times "
        f"(P(failure)={monte_carlo_result['probability_of_failure']:.3f})"
    )
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "failure_distribution.png"), dpi=150)
    plt.close(fig)


def plot_defense_comparison(
    baseline_probability,
    optimization_result,
    title="Defense Resource Allocation Comparison",
    filename="defense_comparison.png",
):
    """Bar chart comparing failure probability across every defended-node
    combination tested during optimization, plus the undefended baseline.

    Error bars show each combination's 95% Wilson CI (from
    optimization_result["results_detailed"], when present) so the plot
    reflects whether combinations are actually statistically
    distinguishable, not just a bare point-estimate ranking.

    The best combination is highlighted green, the baseline red. Saves
    to results/<filename>.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    best_nodes = optimization_result["best_nodes"]
    results = sorted(optimization_result["results"], key=lambda item: item[1])
    ci_by_combo = {
        d["combo"]: (d["ci_low"], d["ci_high"])
        for d in optimization_result.get("results_detailed", [])
    }

    labels = ["Baseline\n(none)"]
    values = [baseline_probability]
    colors = ["red"]
    yerr_low = [0.0]
    yerr_high = [0.0]

    for combo, probability in results:
        labels.append("+".join(_abbreviate(n) for n in combo))
        values.append(probability)
        colors.append("green" if combo == best_nodes else "steelblue")
        ci_low, ci_high = ci_by_combo.get(combo, (probability, probability))
        yerr_low.append(probability - ci_low)
        yerr_high.append(ci_high - probability)

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.5), 6))
    ax.bar(
        range(len(values)), values, color=colors, edgecolor="black",
        yerr=[yerr_low, yerr_high], capsize=3, ecolor="black",
    )
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylabel(f"P({TARGET_NODE} fails)")
    ax.set_title(f"{title}\n(error bars = 95% Wilson CI)")
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, filename), dpi=150)
    plt.close(fig)


def run_all_plots():
    """Run the full simulation pipeline and generate all four plots."""
    graph = build_smart_grid()
    ctmc_nodes = build_ctmc_nodes(graph)

    simulation_result = run_attack_simulation(graph, ctmc_nodes, t_max=24, t_steps=100)
    monte_carlo_result = run_monte_carlo(
        graph, ctmc_nodes, n_runs=DEFAULT_N_RUNS, t_max=24, seed=DEFAULT_SEED
    )
    baseline_probability = monte_carlo_result["probability_of_failure"]
    optimization_result = optimize_defense(
        graph, ctmc_nodes, budget=2, t_max=24,
        n_runs=DEFAULT_N_RUNS, seed=DEFAULT_SEED, excluded_nodes=[TARGET_NODE],
    )

    plot_graph(graph)
    plot_attack_propagation(simulation_result)
    plot_failure_distribution(monte_carlo_result)
    plot_defense_comparison(baseline_probability, optimization_result)

    print(f"Saved 4 plots to {RESULTS_DIR}/")


if __name__ == "__main__":
    run_all_plots()
