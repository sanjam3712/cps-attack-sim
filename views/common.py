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
