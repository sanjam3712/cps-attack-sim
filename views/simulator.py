"""Streamlit UI for the CPS cyberattack propagation simulation.

Lets a user pick a system config (or upload their own), tune the
Monte Carlo / defense-optimization parameters, and inspect the
results interactively. Does not change any simulation logic - it just
drives the existing src.* functions and renders their output.
"""

import csv
import io
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
from src.stats import intervals_overlap, wilson_confidence_interval
from src.visualization import plot_attack_propagation, plot_failure_distribution
from views.charts import plotly_defense_comparison, plotly_failure_over_time, plotly_graph
from views.common import CONFIGS_DIR, build_insight, is_illustrative, page_header, terms_expander

MAX_COMBOS_BEFORE_WARNING = 300


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


page_header(
    "CPS Attack Simulation",
    "Model cyberattack propagation through a cyber-physical system and find the best defense allocation.",
)

st.sidebar.header("System")

config_files = sorted(CONFIGS_DIR.glob("*.json"))
config_names = [f.name for f in config_files]

if "preselect_config" in st.session_state:
    pending = st.session_state.pop("preselect_config")
    if pending in config_names:
        st.session_state["config_select"] = pending

selected_name = st.sidebar.selectbox(
    "Config file", config_names, index=0 if config_names else None, key="config_select",
    help="Pick a system topology from configs/. Each one defines its own nodes, edges, entry points and target.",
)

uploaded = st.sidebar.file_uploader(
    "Upload your own config (optional)", type=["json"],
    help="Upload a JSON file in the same {nodes, edges, entry_points, target} format as the bundled configs "
    "(see the README for the schema). Overrides the config file picked above.",
)

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
budget = st.sidebar.slider(
    "Defense budget (nodes defended)", 1, 4, 2,
    help="How many different nodes the defender can harden at once. Every combination of this "
    "many nodes is evaluated and ranked.",
)
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

reduction = st.sidebar.slider(
    "Defense strength (compromise-rate reduction)", 0.1, 0.9, 0.5, step=0.1,
    help="How much hardening cuts a defended node's compromise rate. 0.5 means a defended "
    "node is compromised about half as fast as it would be undefended.",
)
n_runs = st.sidebar.slider(
    "Monte Carlo runs", 500, 5000, suggested_default_runs, step=100,
    help="Number of simulated attacks per combination. More runs give tighter confidence "
    "intervals but take longer to compute.",
)
horizon = st.sidebar.slider(
    "Time horizon (hours)", 6, 48, 24, step=1,
    help="How many hours into the future the simulation looks. A short horizon asks "
    "\"could this fail soon?\"; a long horizon asks \"will this eventually fail?\".",
)
seed = st.sidebar.number_input(
    "Random seed", value=42, step=1,
    help="Random-number seed. The same seed and settings always reproduce the same result; "
    "change it to see results under a different random draw.",
)

terms_expander(st.sidebar)

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
baseline_ci_low, baseline_ci_high = wilson_confidence_interval(baseline_probability, params["n_runs"])
mean_ttf = monte_carlo_result["mean_time_to_failure"]
best_nodes = optimization_result["best_nodes"]
best_probability = optimization_result["best_probability"]
best_entry = optimization_result["results_detailed"][0]
best_mean_ttf = best_entry["mean_time_to_failure"]

thresholds_available = sorted(monte_carlo_result["p_fail_by_t"].keys())
long_t = 24 if 24 in monte_carlo_result["p_fail_by_t"] else thresholds_available[-1]
short_t = 6 if 6 in monte_carlo_result["p_fail_by_t"] else thresholds_available[0]

baseline_long = monte_carlo_result["p_fail_by_t"][long_t]["p"]
best_long = best_entry["p_fail_by_t"][long_t]["p"]
baseline_short = monte_carlo_result["p_fail_by_t"][short_t]["p"]
best_short = best_entry["p_fail_by_t"][short_t]["p"]

top5 = optimization_result["results_detailed"][:5]
ci_overlap_top2 = False
if len(top5) >= 2:
    ci_overlap_top2 = intervals_overlap(
        (top5[0]["ci_low"], top5[0]["ci_high"]), (top5[1]["ci_low"], top5[1]["ci_high"])
    )

st.subheader(f"Results - {params['system_name']} (target: {target})")

st.info(
    build_insight(
        defended_nodes=best_nodes,
        baseline_long=baseline_long, best_long=best_long,
        baseline_short=baseline_short, best_short=best_short,
        long_label=f"{long_t}h failure", short_label=f"{short_t}h failure",
        ci_overlap=ci_overlap_top2,
    )
)

id_cols = st.columns(3)
id_cols[0].metric("Best defended nodes", ", ".join(best_nodes) if best_nodes else "(none)")
id_cols[1].metric("Mean time to failure (baseline)", f"{mean_ttf:.2f}h" if mean_ttf is not None else "N/A")
id_cols[2].metric(
    "Mean time to failure (best defense)",
    f"{best_mean_ttf:.2f}h" if best_mean_ttf is not None else "N/A",
)

delta_cols = st.columns(4)
delta_cols[0].metric(f"Baseline P(fail by {long_t}h)", f"{baseline_long:.4f}")
delta_cols[1].metric(
    f"Best defense P(fail by {long_t}h)", f"{best_long:.4f}",
    delta=f"{best_long - baseline_long:+.4f}", delta_color="inverse",
)
delta_cols[2].metric(f"Baseline P(fail by {short_t}h)", f"{baseline_short:.4f}")
delta_cols[3].metric(
    f"Best defense P(fail by {short_t}h)", f"{best_short:.4f}",
    delta=f"{best_short - baseline_short:+.4f}", delta_color="inverse",
)

tab_graph, tab_propagation, tab_distribution, tab_comparison, tab_over_time = st.tabs(
    ["System graph", "Attack propagation", "Failure-time distribution", "Defense comparison", "Failure over time"]
)

with tab_graph:
    fig = plotly_graph(graph, entry_points=graph.graph.get("entry_points"), target=target)
    st.plotly_chart(fig, width='stretch')

with tab_propagation:
    fig = plot_attack_propagation(simulation_result)
    st.pyplot(fig)
    plt.close(fig)

with tab_distribution:
    fig = plot_failure_distribution(monte_carlo_result, target_node=target)
    st.pyplot(fig)
    plt.close(fig)

with tab_comparison:
    results_sorted = sorted(optimization_result["results"], key=lambda item: item[1])
    ci_by_combo = {
        d["combo"]: (d["ci_low"], d["ci_high"]) for d in optimization_result.get("results_detailed", [])
    }
    comparison_entries = [
        dict(label="Baseline\n(none)", probability=baseline_probability,
             ci_low=baseline_ci_low, ci_high=baseline_ci_high, kind="baseline")
    ]
    for combo, probability in results_sorted:
        ci_low, ci_high = ci_by_combo.get(combo, (probability, probability))
        comparison_entries.append(dict(
            label=" + ".join(combo), probability=probability, ci_low=ci_low, ci_high=ci_high,
            kind="best" if combo == best_nodes else "other",
        ))

    comparison_title = (
        f"Defense Resource Allocation Comparison (budget={params['budget']}, target "
        f"{'defendable (naive)' if params['include_target'] else 'excluded'})"
    )
    comparison_fig = plotly_defense_comparison(comparison_entries, target_node=target, title=comparison_title)
    st.plotly_chart(comparison_fig, width='stretch')

    dl_cols = st.columns(2)

    csv_buffer = io.StringIO()
    writer = csv.writer(csv_buffer)
    writer.writerow(["defended_nodes", f"probability_of_failure_by_{params['horizon']}h", "ci_low", "ci_high"])
    writer.writerow(["(baseline - none)", baseline_probability, baseline_ci_low, baseline_ci_high])
    for combo, probability in results_sorted:
        ci_low, ci_high = ci_by_combo.get(combo, (probability, probability))
        writer.writerow([" + ".join(combo), probability, ci_low, ci_high])
    dl_cols[0].download_button(
        "Download all combinations (CSV)", data=csv_buffer.getvalue().encode("utf-8"),
        file_name=f"{params['system_name']}_defense_combinations.csv", mime="text/csv",
    )

    png_bytes = comparison_fig.to_image(format="png", scale=2)
    dl_cols[1].download_button(
        "Download this chart (PNG)", data=png_bytes,
        file_name=f"{params['system_name']}_defense_comparison.png", mime="image/png",
    )

with tab_over_time:
    over_time_fig = plotly_failure_over_time(
        [
            dict(
                label="Baseline (no defense)", color="#DC2626", band_color="rgba(220,38,38,0.15)",
                thresholds=thresholds_available,
                p=[monte_carlo_result["p_fail_by_t"][t]["p"] for t in thresholds_available],
                ci_low=[monte_carlo_result["p_fail_by_t"][t]["ci_low"] for t in thresholds_available],
                ci_high=[monte_carlo_result["p_fail_by_t"][t]["ci_high"] for t in thresholds_available],
            ),
            dict(
                label=f"Best defense ({' + '.join(best_nodes)})" if best_nodes else "Best defense",
                color="#16A34A", band_color="rgba(22,163,74,0.15)",
                thresholds=thresholds_available,
                p=[best_entry["p_fail_by_t"][t]["p"] for t in thresholds_available],
                ci_low=[best_entry["p_fail_by_t"][t]["ci_low"] for t in thresholds_available],
                ci_high=[best_entry["p_fail_by_t"][t]["ci_high"] for t in thresholds_available],
            ),
        ],
        target_node=target,
        title="Baseline vs best defense: failure probability over time",
    )
    st.plotly_chart(over_time_fig, width='stretch')
    st.caption(
        "Defenses mainly delay failure rather than preventing it outright: both curves tend to "
        "converge toward similar failure probability at the longest horizon."
    )

st.subheader(f"Top 5 defense combinations (P(fail within {params['horizon']}h), 95% CI)")

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

if ci_overlap_top2:
    st.warning("Top two combinations' 95% confidence intervals overlap - not statistically distinguishable.")
