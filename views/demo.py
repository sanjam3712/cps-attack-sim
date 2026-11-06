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
from src.visualization import plot_attack_propagation
from views.charts import plotly_defense_comparison, plotly_failure_over_time, plotly_graph
from views.common import CONFIGS_DIR, RESULTS_DIR, build_insight, load_config, page_header, preselect_config

DEMO_RESULTS_PATH = RESULTS_DIR / "demo_substation.json"
THRESHOLDS = (3, 6, 12, 24)

page_header("Guided Demo: Smart-Grid Substation")

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

_defense_05 = demo["defenses"]["0.5"]
_top5_05 = _defense_05["top5"]
_ci_overlap = len(_top5_05) >= 2 and intervals_overlap(
    (_top5_05[0]["ci_low"], _top5_05[0]["ci_high"]), (_top5_05[1]["ci_low"], _top5_05[1]["ci_high"])
)
st.info(
    build_insight(
        defended_nodes=_defense_05["best_nodes"],
        baseline_long=demo["baseline"]["p_fail_by_t"]["24"]["p"],
        best_long=_defense_05["p_fail_by_t"]["24"]["p"],
        baseline_short=demo["baseline"]["p_fail_by_t"]["6"]["p"],
        best_short=_defense_05["p_fail_by_t"]["6"]["p"],
        long_label="24h failure",
        short_label="6h failure",
        ci_overlap=_ci_overlap,
    )
)

step1, step2, step3, step4 = st.tabs(
    ["1. System graph", "2. Example attack", "3. Baseline vs best defense", "4. Top 5 combinations"]
)

with step1:
    st.write(
        "The substation's attack graph: corporate IT down to the physical breaker, nodes "
        "colored by criticality. Hover a node for its vulnerability/criticality; the entry "
        "point and target are marked with distinct shapes."
    )
    fig = plotly_graph(graph, entry_points=graph.graph.get("entry_points"), target=target)
    st.plotly_chart(fig, width='stretch')

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

    def _series(entry):
        return (
            [entry["p_fail_by_t"][str(t)]["p"] for t in THRESHOLDS],
            [entry["p_fail_by_t"][str(t)]["ci_low"] for t in THRESHOLDS],
            [entry["p_fail_by_t"][str(t)]["ci_high"] for t in THRESHOLDS],
        )

    defense_05 = demo["defenses"]["0.5"]
    defense_09 = demo["defenses"]["0.9"]
    baseline_p, baseline_lo, baseline_hi = _series(demo["baseline"])
    p_05, lo_05, hi_05 = _series(defense_05)
    p_09, lo_09, hi_09 = _series(defense_09)

    fig = plotly_failure_over_time(
        [
            dict(
                label="Baseline (no defense)", color="#DC2626", band_color="rgba(220,38,38,0.15)",
                thresholds=list(THRESHOLDS), p=baseline_p, ci_low=baseline_lo, ci_high=baseline_hi,
            ),
            dict(
                label=f"Best defense, reduction=0.5 ({' + '.join(defense_05['best_nodes'])})",
                color="#2563EB", band_color="rgba(37,99,235,0.15)",
                thresholds=list(THRESHOLDS), p=p_05, ci_low=lo_05, ci_high=hi_05,
            ),
            dict(
                label=f"Best defense, reduction=0.9 ({' + '.join(defense_09['best_nodes'])})",
                color="#16A34A", band_color="rgba(22,163,74,0.15)",
                thresholds=list(THRESHOLDS), p=p_09, ci_low=lo_09, ci_high=hi_09,
            ),
        ],
        target_node=target,
        title="Baseline vs best defense: failure probability over time",
    )
    st.plotly_chart(fig, width='stretch')

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

    bar_entries = [
        dict(
            label="Baseline\n(none)", probability=demo["baseline"]["probability_of_failure"],
            ci_low=demo["baseline"]["p_fail_by_t"]["24"]["ci_low"],
            ci_high=demo["baseline"]["p_fail_by_t"]["24"]["ci_high"],
            kind="baseline",
        )
    ]
    for i, entry in enumerate(top5):
        bar_entries.append(dict(
            label=" + ".join(entry["combo"]), probability=entry["probability_of_failure"],
            ci_low=entry["ci_low"], ci_high=entry["ci_high"],
            kind="best" if i == 0 else "other",
        ))
    fig = plotly_defense_comparison(
        bar_entries, target_node=target,
        title="Baseline vs top 5 defended-node combinations (reduction=0.5)",
    )
    st.plotly_chart(fig, width='stretch')

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
