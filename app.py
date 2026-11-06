"""Streamlit UI for the CPS cyberattack propagation simulation.

Lets a user pick a system config (or upload their own), tune the
Monte Carlo / defense-optimization parameters, and inspect the
results interactively. Does not change any simulation logic - it just
drives the existing src.* functions and renders their output.
"""

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from src.ctmc import build_ctmc_nodes
from src.graph import build_graph_from_config, validate_system_config
from src.optimization import optimize_defense
from src.simulation import run_attack_simulation, run_monte_carlo
from src.stats import intervals_overlap
from src.visualization import (
    plot_attack_propagation,
    plot_defense_comparison,
    plot_failure_distribution,
    plot_graph,
)

CONFIGS_DIR = Path(__file__).resolve().parent / "configs"
MAX_COMBOS_BEFORE_WARNING = 300

st.set_page_config(page_title="CPS Attack Simulation", layout="wide")


def _ranking_metric_for_horizon(horizon):
    """optimize_defense's "p_fail_24h" metric needs a 24h or exact-t_max
    entry in p_fail_by_t (see src/optimization.py), which only exists
    for horizon>=24 or horizon in FAILURE_TIME_THRESHOLDS_HOURS (3, 6,
    12, 24). Use it when safe (this matches the CLI's default ranking
    at the default 24h horizon); otherwise fall back to "p_fail_6h",
    always valid since the horizon slider's minimum is 6h."""
    if horizon >= 24 or horizon in (3, 6, 12):
        return "p_fail_24h"
    return "p_fail_6h"


@st.cache_data(show_spinner=False)
def run_pipeline(config_json, system_name, budget, reduction, n_runs, horizon, seed, include_target):
    """Run the full simulation + optimization pipeline once; cached on
    every input that affects the result, so re-running with identical
    settings is instant."""
    config = json.loads(config_json)
    graph = build_graph_from_config(config, system_name=system_name)
    ctmc_nodes = build_ctmc_nodes(graph)
    target = graph.graph["target"]

    monte_carlo_result = run_monte_carlo(graph, ctmc_nodes, n_runs=n_runs, t_max=horizon, seed=seed)

    excluded_nodes = None if include_target else [target]
    optimization_result = optimize_defense(
        graph, ctmc_nodes, budget=budget, t_max=horizon,
        n_runs=n_runs, seed=seed, excluded_nodes=excluded_nodes,
        metric=_ranking_metric_for_horizon(horizon), reduction=reduction,
    )

    simulation_result = run_attack_simulation(
        graph, ctmc_nodes, t_max=horizon, t_steps=100, rng=np.random.default_rng(seed)
    )

    return graph, monte_carlo_result, optimization_result, simulation_result


def is_illustrative(config):
    meta_text = json.dumps(config.get("_meta", {})).lower()
    return "illustrative" in meta_text


st.title("CPS Attack Simulation")
st.caption("Model cyberattack propagation through a cyber-physical system and find the best defense allocation.")

st.sidebar.header("System")

config_files = sorted(CONFIGS_DIR.glob("*.json"))
config_names = [f.name for f in config_files]
selected_name = st.sidebar.selectbox("Config file", config_names, index=0 if config_names else None)

uploaded = st.sidebar.file_uploader("Upload your own config (optional)", type=["json"])

config = None
system_name = None
config_error = None

if uploaded is not None:
    try:
        config = json.loads(uploaded.getvalue())
        system_name = Path(uploaded.name).stem
    except json.JSONDecodeError as exc:
        config_error = f"Uploaded file is not valid JSON: {exc}"
elif selected_name:
    path = CONFIGS_DIR / selected_name
    with open(path, encoding="utf-8") as f:
        config = json.load(f)
    system_name = path.stem
else:
    config_error = "No config files found in configs/ and nothing uploaded."

if config is not None and config_error is None:
    try:
        validate_system_config(config)
    except ValueError as exc:
        config_error = f"Invalid config: {exc}"

if config_error:
    st.sidebar.error(config_error)
    st.stop()

if is_illustrative(config):
    st.info("Illustrative values, not real vulnerability data")

node_names = list(config["nodes"].keys())
target_name = config["target"]

st.sidebar.header("Parameters")
budget = st.sidebar.slider("Defense budget (nodes defended)", 1, 4, 2)
include_target = st.sidebar.checkbox(
    "Include the target node as defendable (naive)", value=False,
    help="Defending the target node directly is a trivial result, not a propagation defense.",
)

eligible_count = len(node_names) if include_target else len(node_names) - 1
n_combos = math.comb(eligible_count, budget) if eligible_count >= budget >= 0 else 0

suggested_default_runs = 1000
if n_combos > MAX_COMBOS_BEFORE_WARNING:
    st.sidebar.warning(
        f"Budget={budget} on {eligible_count} defendable nodes gives {n_combos} combinations - "
        "this will be slow. Lowering the default run count."
    )
    suggested_default_runs = 500

reduction = st.sidebar.slider("Defense strength (compromise-rate reduction)", 0.1, 0.9, 0.5, step=0.1)
n_runs = st.sidebar.slider("Monte Carlo runs", 500, 5000, suggested_default_runs, step=100)
horizon = st.sidebar.slider("Time horizon (hours)", 6, 48, 24, step=1)
seed = st.sidebar.number_input("Random seed", value=42, step=1)

run_clicked = st.sidebar.button("Run simulation", type="primary")

if run_clicked:
    if budget > eligible_count:
        st.error(
            f"Budget ({budget}) exceeds the number of defendable nodes ({eligible_count}) in this system."
        )
    else:
        with st.spinner(f"Running {n_runs} Monte Carlo runs x {n_combos + 1} combinations..."):
            # sort_keys would silently reorder the "nodes" dict, changing
            # node iteration order and therefore the RNG draw sequence in
            # the simulation - must NOT sort, to stay reproducible.
            result = run_pipeline(
                json.dumps(config), system_name,
                budget, reduction, n_runs, horizon, int(seed), include_target,
            )
        st.session_state["result"] = result
        st.session_state["result_params"] = dict(
            system_name=system_name, budget=budget, reduction=reduction,
            n_runs=n_runs, horizon=horizon, seed=int(seed), include_target=include_target,
        )

if "result" not in st.session_state:
    st.info("Set parameters in the sidebar and click **Run simulation**.")
    st.stop()

graph, monte_carlo_result, optimization_result, simulation_result = st.session_state["result"]
params = st.session_state["result_params"]
target = graph.graph["target"]

baseline_probability = monte_carlo_result["probability_of_failure"]
mean_ttf = monte_carlo_result["mean_time_to_failure"]
best_nodes = optimization_result["best_nodes"]
best_probability = optimization_result["best_probability"]
baseline_p6h = monte_carlo_result["p_fail_by_t"].get(6, {}).get("p")
best_entry = optimization_result["results_detailed"][0]
best_p6h = best_entry["p_fail_by_t"].get(6, {}).get("p")

st.subheader(f"Results - {params['system_name']} (target: {target})")

cols = st.columns(5)
cols[0].metric(f"Baseline P(fail within {params['horizon']}h)", f"{baseline_probability:.4f}")
cols[1].metric(
    "Mean time to failure",
    f"{mean_ttf:.2f}h" if mean_ttf is not None else "N/A",
)
cols[2].metric("Best defended nodes", ", ".join(best_nodes) if best_nodes else "(none)")
cols[3].metric(f"P(fail) with best defense", f"{best_probability:.4f}")
cols[4].metric(
    "P(fail by 6h): baseline -> best",
    f"{baseline_p6h:.4f} -> {best_p6h:.4f}" if baseline_p6h is not None and best_p6h is not None else "N/A",
)

tab_graph, tab_propagation, tab_distribution, tab_comparison = st.tabs(
    ["System graph", "Attack propagation", "Failure-time distribution", "Defense comparison"]
)

with tab_graph:
    fig = plot_graph(graph)
    st.pyplot(fig)
    plt.close(fig)

with tab_propagation:
    fig = plot_attack_propagation(simulation_result)
    st.pyplot(fig)
    plt.close(fig)

with tab_distribution:
    fig = plot_failure_distribution(monte_carlo_result, target_node=target)
    st.pyplot(fig)
    plt.close(fig)

with tab_comparison:
    fig = plot_defense_comparison(
        baseline_probability, optimization_result, target_node=target,
        title=f"Defense Resource Allocation Comparison (budget={params['budget']}, target "
        f"{'defendable (naive)' if params['include_target'] else 'excluded'})",
        filename="defense_comparison_app.png",
    )
    st.pyplot(fig)
    plt.close(fig)

st.subheader(f"Top 5 defense combinations (P(fail within {params['horizon']}h), 95% CI)")

top5 = optimization_result["results_detailed"][:5]
table_rows = [
    {
        "Rank": i + 1,
        "Defended nodes": " + ".join(entry["combo"]),
        f"P(fail within {params['horizon']}h)": round(entry["probability_of_failure"], 4),
        "95% CI low": round(entry["ci_low"], 4),
        "95% CI high": round(entry["ci_high"], 4),
    }
    for i, entry in enumerate(top5)
]
st.table(table_rows)

if len(top5) >= 2:
    ci_1 = (top5[0]["ci_low"], top5[0]["ci_high"])
    ci_2 = (top5[1]["ci_low"], top5[1]["ci_high"])
    if intervals_overlap(ci_1, ci_2):
        st.warning("Top two combinations' 95% confidence intervals overlap - not statistically distinguishable.")
