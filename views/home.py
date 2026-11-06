"""Home page: project overview, how it works, key findings, and a
gallery of the available system configs."""

import streamlit as st

from views.common import config_summary, list_configs, preselect_config

st.title("CPS Attack Simulation")

st.write(
    "This tool models how a cyberattack spreads through a cyber-physical system - from an "
    "IT entry point, down through SCADA/control layers, to a physical target - and estimates "
    "the probability and timing of that target failing. It then searches over a limited "
    "defense budget to find which nodes are most worth hardening, and reports how confident "
    "that ranking actually is. Use the **Guided Demo** for an instant walkthrough, or the "
    "**Simulator** to run your own scenarios."
)

st.subheader("How it works")
steps = st.columns(4)
step_content = [
    ("1. Build the graph", "Each system is a directed graph: nodes are IT/OT components with a "
     "vulnerability and criticality score, edges are how an attack can spread between them."),
    ("2. Model each node", "Every node gets its own continuous-time Markov chain (secure -> "
     "compromised -> failed/recovered), driven by its vulnerability and criticality."),
    ("3. Simulate attacks", "Thousands of Monte Carlo runs trace an attack from the entry "
     "point, recording whether and when the target fails."),
    ("4. Compare defenses", "Every affordable combination of defended nodes is re-simulated "
     "with the same random draws, so defenses can be ranked with confidence intervals, not "
     "just point estimates."),
]
for col, (title, text) in zip(steps, step_content):
    with col:
        st.markdown(f"**{title}**")
        st.caption(text)

st.subheader("Key findings")
f1, f2, f3 = st.columns(3)
with f1:
    st.markdown("**Defenses buy time, not certainty**")
    st.caption(
        "Hardening nodes mostly delays failure rather than preventing it - failure "
        "probabilities across defenses converge at long horizons."
    )
with f2:
    st.markdown("**\"Best\" depends on the metric**")
    st.caption(
        "Which combination ranks first changes with the chosen metric and horizon, and the "
        "top options are often statistically tied (overlapping 95% CIs)."
    )
with f3:
    st.markdown("**Matches graph theory exactly**")
    st.caption(
        "When a node is removed entirely (not just weakened), the simulation's failure "
        "probability matches a graph min-cut analysis exactly."
    )

st.subheader("Systems")
configs = list_configs()
if not configs:
    st.info("No configs found in configs/.")
else:
    cols = st.columns(2)
    for i, path in enumerate(configs):
        summary = config_summary(path)
        with cols[i % 2]:
            with st.container(border=True):
                title = summary["name"].replace("_", " ").title()
                if summary["illustrative"]:
                    title += "  :orange[ILLUSTRATIVE]"
                st.markdown(f"**{title}**")
                st.caption(
                    f"{summary['n_nodes']} nodes &middot; target: `{summary['target']}` &middot; "
                    f"entry point: `{summary['entry_point']}`"
                )
                if st.button("Open in simulator", key=f"open_{summary['file_name']}"):
                    preselect_config(summary["file_name"])

with st.expander("Limitations"):
    st.markdown(
        "- The topology for each system is hand-drawn, not derived from a real network scan.\n"
        "- Vulnerability scores come from CVSS exploitability sub-scores scaled into "
        "compromise rates - a proxy, not a measured attack rate.\n"
        "- Defenses are modeled as a static, one-time compromise-rate reduction, not an "
        "adaptive or time-varying response.\n"
        "- There is no power-flow / physics model; node \"failure\" is a CTMC state, not a "
        "simulated physical consequence.\n"
        "- Systems marked ILLUSTRATIVE use placeholder vulnerability/criticality values, not "
        "real vulnerability data - don't draw real-world conclusions from them."
    )
