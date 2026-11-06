"""Guided demo: a 4-step substation walkthrough.

Loads precomputed Monte Carlo / optimization results from
results/demo_substation.json (see scripts/make_demo.py) instead of
re-running thousands of simulations on every page view. The graph
drawing and single example attack trace are cheap enough to compute
live. All failure-probability numbers shown come from that JSON, not
hardcoded.
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from src.ctmc import build_ctmc_nodes
from src.graph import build_graph_from_config
from src.simulation import run_attack_simulation
from src.stats import intervals_overlap
from src.visualization import plot_attack_propagation, plot_graph
from views.common import CONFIGS_DIR, RESULTS_DIR, load_config, preselect_config

DEMO_RESULTS_PATH = RESULTS_DIR / "demo_substation.json"
THRESHOLDS = (3, 6, 12, 24)

st.title("Guided Demo: Smart-Grid Substation")

if not DEMO_RESULTS_PATH.exists():
    st.error(
        f"Precomputed demo results not found at `{DEMO_RESULTS_PATH}`. "
        "Run `python scripts/make_demo.py` first."
    )
    st.stop()

with open(DEMO_RESULTS_PATH, encoding="utf-8") as f:
    demo = json.load(f)

config = load_config(CONFIGS_DIR / "substation.json")
graph = build_graph_from_config(config, system_name="substation")
ctmc_nodes = build_ctmc_nodes(graph)
target = graph.graph["target"]

st.caption(
    "A 4-step walkthrough of the same simulation the Simulator page runs, using precomputed "
    f"results (seed={demo['seed']}, {demo['n_runs']} Monte Carlo runs, budget={demo['budget']}) "
    "for the smart-grid substation, so this page loads instantly."
)

step1, step2, step3, step4 = st.tabs(
    ["1. System graph", "2. Example attack", "3. Baseline vs best defense", "4. Top 5 combinations"]
)

with step1:
    st.write(
        "The substation's attack graph: corporate IT down to the physical breaker, nodes "
        "colored by criticality."
    )
    fig = plot_graph(graph)
    st.pyplot(fig)
    plt.close(fig)

with step2:
    st.write("One example simulated attack run, showing how compromise spreads node by node over time.")
    sim_result = run_attack_simulation(
        graph, ctmc_nodes, t_max=demo["horizon"], t_steps=100,
        rng=np.random.default_rng(demo["seed"]),
    )
    fig = plot_attack_propagation(sim_result)
    st.pyplot(fig)
    plt.close(fig)

with step3:
    st.write("Baseline failure probability vs. the best defense at two defense strengths, over time.")

    baseline_series = [demo["baseline"]["p_fail_by_t"][str(t)]["p"] for t in THRESHOLDS]
    defense_05 = demo["defenses"]["0.5"]
    defense_09 = demo["defenses"]["0.9"]
    series_05 = [defense_05["p_fail_by_t"][str(t)]["p"] for t in THRESHOLDS]
    series_09 = [defense_09["p_fail_by_t"][str(t)]["p"] for t in THRESHOLDS]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(THRESHOLDS, baseline_series, marker="o", color="red", label="Baseline (no defense)")
    ax.plot(
        THRESHOLDS, series_05, marker="o", color="steelblue",
        label=f"Best defense, reduction=0.5 ({' + '.join(defense_05['best_nodes'])})",
    )
    ax.plot(
        THRESHOLDS, series_09, marker="o", color="green",
        label=f"Best defense, reduction=0.9 ({' + '.join(defense_09['best_nodes'])})",
    )
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel(f"P({target} fails by t)")
    ax.set_title("Baseline vs best defense: failure probability over time")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "Defenses mainly delay failure rather than preventing it outright: all three curves "
        "converge toward similar failure probability by the longest horizon."
    )

with step4:
    st.write(
        "Top 5 defended-node combinations (reduction=0.5), ranked by "
        "P(fail within 24h), with 95% confidence intervals."
    )
    top5 = demo["defenses"]["0.5"]["top5"]
    table_rows = [
        {
            "Rank": i + 1,
            "Defended nodes": " + ".join(entry["combo"]),
            "P(fail within 24h)": round(entry["probability_of_failure"], 4),
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
            st.info(
                "The top two combinations' 95% confidence intervals overlap - they are not "
                "statistically distinguishable."
            )

st.divider()
if st.button("Try it yourself in the Simulator ->", type="primary"):
    preselect_config("substation.json")
