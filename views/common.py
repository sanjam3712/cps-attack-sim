"""Small app-level helpers shared across views/ pages.

Pure Streamlit/IO plumbing (listing configs, reading JSON, navigating
between pages) - no simulation logic lives here.
"""

import json
from pathlib import Path

import streamlit as st

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

SIMULATOR_PAGE = "views/simulator.py"


def list_configs():
    return sorted(CONFIGS_DIR.glob("*.json"))


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def is_illustrative(config):
    meta_text = json.dumps(config.get("_meta", {})).lower()
    return "illustrative" in meta_text


def config_summary(path):
    """Return a small dict describing a config for the home-page gallery."""
    config = load_config(path)
    entry_points = config.get("entry_points", {})
    entry_point = next(iter(entry_points), "?")
    return {
        "file_name": path.name,
        "name": path.stem,
        "n_nodes": len(config.get("nodes", {})),
        "target": config.get("target", "?"),
        "entry_point": entry_point,
        "illustrative": is_illustrative(config),
    }


def preselect_config(file_name):
    """Stage `file_name` (e.g. 'substation.json') as the simulator's
    selected config next time it loads, then jump to the simulator page."""
    st.session_state["preselect_config"] = file_name
    st.switch_page(SIMULATOR_PAGE)


_HEADER_CSS = """
<style>
.app-header {
    border-bottom: 3px solid #2563EB;
    padding-bottom: 0.4rem;
    margin-bottom: 0.8rem;
}
.app-header h1 {
    margin-bottom: 0.1rem;
    font-size: 2rem;
}
</style>
"""


def page_header(title, caption=None):
    """Render the shared page header (title + accent underline + caption)
    used by every page, so navigation between pages feels consistent."""
    st.markdown(_HEADER_CSS, unsafe_allow_html=True)
    st.markdown(f'<div class="app-header"><h1>{title}</h1></div>', unsafe_allow_html=True)
    if caption:
        st.caption(caption)


TERMS = [
    ("Vulnerability", "How easily this component can be compromised once an attacker can "
     "reach it (0 = very hard, 1 = very easy). Where available, this comes from real CVE "
     "data for a representative product of that component type."),
    ("Criticality", "How likely the component is to fail outright once it's compromised "
     "(0 = unlikely to fail even if compromised, 1 = almost certain to fail)."),
    ("Compromise rate", "How fast an attacker can take over a node once they can reach it "
     "(i.e. once an upstream neighbor is already compromised). Derived from the node's "
     "vulnerability - higher vulnerability means a faster compromise rate."),
    ("Defense strength (compromise-rate reduction)", "How much hardening a defended node "
     "cuts its compromise rate. 0.5 means a defended node is compromised about half as "
     "fast as it would be undefended."),
    ("Confidence interval (CI)", "A range that very likely contains the true failure "
     "probability, accounting for sampling noise from a finite number of simulation runs. "
     "A 95% CI means we're 95% confident the true value falls in that range. When two "
     "options' CIs overlap, the simulation can't reliably tell which is actually better."),
    ("Time horizon", "How many hours into the future the simulation looks. A short horizon "
     "asks \"could this fail soon?\"; a long horizon asks \"will this eventually fail?\"."),
]


def terms_expander(container=st):
    """Render the shared 'What do these terms mean?' expander. Pass
    `container=st.sidebar` to render it in the sidebar instead of the
    main page body."""
    with container.expander("What do these terms mean?"):
        for term, description in TERMS:
            st.markdown(f"**{term}** — {description}")


def build_insight(
    defended_nodes, baseline_long, best_long, baseline_short, best_short,
    long_label, short_label, ci_overlap,
):
    """Build a plain-English sentence summarizing the defense's effect,
    from the actual computed numbers (no hardcoded figures)."""
    nodes_text = " and ".join(defended_nodes) if defended_nodes else "no additional nodes"
    distinguish_text = (
        "The top options are not statistically distinguishable (their 95% confidence "
        "intervals overlap)."
        if ci_overlap else
        "The top option is statistically distinguishable from the runner-up (their 95% "
        "confidence intervals do not overlap)."
    )
    return (
        f"Defending **{nodes_text}** cuts **{long_label}** from {baseline_long:.1%} to "
        f"{best_long:.1%}, and **{short_label}** from {baseline_short:.1%} to {best_short:.1%}. "
        f"{distinguish_text}"
    )
