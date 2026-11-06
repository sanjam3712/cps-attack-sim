"""Precompute the guided-demo results for the substation config.

views/demo.py reads results/demo_substation.json instead of re-running
Monte Carlo simulations on every page view. Re-run this script whenever
the substation config or simulation logic changes:

    python scripts/make_demo.py
"""

import json
from pathlib import Path

from src.ctmc import build_ctmc_nodes
from src.graph import build_graph_from_config
from src.optimization import optimize_defense
from src.simulation import run_monte_carlo

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "substation.json"
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "results" / "demo_substation.json"

SEED = 42
N_RUNS = 5000
BUDGET = 2
HORIZON = 24
REDUCTIONS = (0.5, 0.9)


def _p_fail_by_t(p_fail_by_t):
    return {str(t): entry for t, entry in p_fail_by_t.items()}


def main():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        config = json.load(f)

    graph = build_graph_from_config(config, system_name="substation")
    ctmc_nodes = build_ctmc_nodes(graph)
    target = graph.graph["target"]

    baseline = run_monte_carlo(graph, ctmc_nodes, n_runs=N_RUNS, t_max=HORIZON, seed=SEED)

    defenses = {}
    for reduction in REDUCTIONS:
        result = optimize_defense(
            graph, ctmc_nodes, budget=BUDGET, t_max=HORIZON,
            n_runs=N_RUNS, seed=SEED, excluded_nodes=[target],
            metric="p_fail_24h", reduction=reduction,
        )
        best = result["results_detailed"][0]
        top5 = result["results_detailed"][:5]
        defenses[str(reduction)] = {
            "reduction": reduction,
            "best_nodes": list(best["combo"]),
            "probability_of_failure": best["probability_of_failure"],
            "p_fail_by_t": _p_fail_by_t(best["p_fail_by_t"]),
            "top5": [
                {
                    "combo": list(entry["combo"]),
                    "probability_of_failure": entry["probability_of_failure"],
                    "ci_low": entry["ci_low"],
                    "ci_high": entry["ci_high"],
                }
                for entry in top5
            ],
        }

    output = {
        "system_name": "substation",
        "target": target,
        "seed": SEED,
        "n_runs": N_RUNS,
        "budget": BUDGET,
        "horizon": HORIZON,
        "target_excluded": True,
        "baseline": {
            "probability_of_failure": baseline["probability_of_failure"],
            "mean_time_to_failure": baseline["mean_time_to_failure"],
            "p_fail_by_t": _p_fail_by_t(baseline["p_fail_by_t"]),
        },
        "defenses": defenses,
    }

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
