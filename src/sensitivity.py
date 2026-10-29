"""Sensitivity sweep over defense reduction, budget, and ranking metric.

A single (reduction=0.5, metric=p_fail_24h) choice can hide how much
the "best" defense combo depends on that choice. This sweeps defense
reduction x budget x ranking metric, evaluates every combo once per
(reduction, budget) pair (shared across metrics via common random
numbers, so the comparison isn't re-noised by independent sampling),
and reports whether the best combo is stable across metrics.

Running this module directly writes results/sensitivity.csv (one row
per reduction x budget x metric) and results/sensitivity_heatmap.png
(best-combo P(fail by 6h) vs budget and reduction), and prints a
per-reduction stability report for budget=2.
"""

import csv
import os

from src.ctmc import build_ctmc_nodes
from src.graph import build_smart_grid
from src.optimization import (
    DEFAULT_N_RUNS,
    DEFAULT_SEED,
    METRICS,
    evaluate_combinations,
    rank_by_metric,
)
from src.simulation import TARGET_NODE
from src.visualization import RESULTS_DIR, plot_sensitivity_heatmap

REDUCTIONS = (0.25, 0.5, 0.75, 0.9)
BUDGETS = (1, 2, 3)


def run_sweep(
    graph,
    ctmc_nodes,
    reductions=REDUCTIONS,
    budgets=BUDGETS,
    metrics=METRICS,
    t_max=24,
    n_runs=DEFAULT_N_RUNS,
    seed=DEFAULT_SEED,
    excluded_nodes=(TARGET_NODE,),
):
    """Evaluate every (reduction, budget) combo once, then rank by every
    metric. Returns (rows, cache):
      rows: list of dicts, one per (reduction, budget, metric), with the
        best combo under that metric and its stats - this is exactly
        what gets written to sensitivity.csv.
      cache: {(reduction, budget): detailed_list} from evaluate_combinations,
        reused by plot_sensitivity_heatmap so the heatmap doesn't re-run
        any Monte Carlo simulation.
    """
    rows = []
    cache = {}

    for reduction in reductions:
        for budget in budgets:
            detailed = evaluate_combinations(
                graph, ctmc_nodes, budget=budget, t_max=t_max,
                n_runs=n_runs, seed=seed, excluded_nodes=excluded_nodes,
                reduction=reduction,
            )
            cache[(reduction, budget)] = detailed

            for metric in metrics:
                best = rank_by_metric(detailed, metric, t_max=t_max)[0]
                rows.append(
                    {
                        "reduction": reduction,
                        "budget": budget,
                        "metric": metric,
                        "best_combo": "+".join(best["combo"]),
                        "metric_value": best["metric_value"],
                        "p_fail_6h": best["p_fail_by_t"][6]["p"],
                        "p_fail_24h": best["probability_of_failure"],
                        "mean_time_to_failure": best["mean_time_to_failure"],
                        "n_combos_evaluated": len(detailed),
                        "n_runs": n_runs,
                        "seed": seed,
                    }
                )

    return rows, cache


def write_csv(rows, path=os.path.join(RESULTS_DIR, "sensitivity.csv")):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fieldnames = [
        "reduction", "budget", "metric", "best_combo", "metric_value",
        "p_fail_6h", "p_fail_24h", "mean_time_to_failure",
        "n_combos_evaluated", "n_runs", "seed",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def print_stability_report(rows, budget=2, metrics=METRICS, reductions=REDUCTIONS):
    """For each reduction, print the best combo under every metric at
    `budget`, and state whether all metrics agree (stable) or not."""
    print(f"\nStability of best combo across metrics, budget={budget}:")
    by_reduction = {r: {} for r in reductions}
    for row in rows:
        if row["budget"] != budget:
            continue
        by_reduction[row["reduction"]][row["metric"]] = row["best_combo"]

    any_unstable = False
    for reduction in reductions:
        combos = by_reduction[reduction]
        print(f"\n  reduction={reduction}:")
        for metric in metrics:
            print(f"    {metric:12s} -> {combos.get(metric, 'N/A')}")
        distinct = set(combos.values())
        if len(distinct) == 1:
            print("    STABLE - all 3 metrics pick the same combo.")
        else:
            any_unstable = True
            print(f"    NOT STABLE - {len(distinct)} different combos across metrics.")

    print(
        "\nOverall: "
        + ("ranking CHANGES depending on which metric is used." if any_unstable
           else "ranking is STABLE across all tested metrics at this budget.")
    )


if __name__ == "__main__":
    smart_grid = build_smart_grid()
    ctmc_nodes = build_ctmc_nodes(smart_grid)

    print(
        f"Running sensitivity sweep: reductions={REDUCTIONS} x budgets={BUDGETS} "
        f"x metrics={METRICS}, {TARGET_NODE} excluded, n_runs={DEFAULT_N_RUNS}, seed={DEFAULT_SEED}"
    )
    rows, cache = run_sweep(smart_grid, ctmc_nodes)

    csv_path = write_csv(rows)
    print(f"\nWrote {len(rows)} rows to {csv_path}")

    print_stability_report(rows, budget=2)

    plot_sensitivity_heatmap(cache, REDUCTIONS, BUDGETS, metric="p_fail_6h")
    print(f"\nSaved {os.path.join(RESULTS_DIR, 'sensitivity_heatmap.png')}")
