"""Topology robustness check.

Tests whether the SCADA-chokepoint defense finding is an artifact of the
hand-drawn baseline topology, by sweeping 4 topology variants (baseline,
redundant, flat, multi_entry - see src.graph.build_smart_grid) x budget
(2, 3) x defense reduction (0.5, 0.9) x ranking metric (p_fail_6h,
p_fail_24h, mean_ttf). physical_equipment (the attack target) is excluded
from the defendable set in every cell. Each (variant, budget, reduction)
cell's Monte Carlo evaluation is run once and reused across metrics
(common random numbers), as in src.sensitivity.

Running this module directly writes results/topology_robustness.csv (one
row per variant x budget x reduction x metric) and prints, per variant,
whether a SCADA node (scada_server, or scada_backup in the "redundant"
variant) remains in the top-3 ranked combos across every tested
budget x reduction x metric cell.
"""

import csv
import os

from src.ctmc import build_ctmc_nodes
from src.graph import VARIANTS, build_smart_grid
from src.optimization import DEFAULT_N_RUNS, DEFAULT_SEED, METRICS, evaluate_combinations, rank_by_metric
from src.simulation import TARGET_NODE
from src.visualization import RESULTS_DIR

BUDGETS = (2, 3)
REDUCTIONS = (0.5, 0.9)


def _scada_nodes(graph):
    return {n for n in graph.nodes() if "scada" in n}


def _top3_has_scada(ranked, scada_nodes, top_n=3):
    """True if the top `top_n` ranked combos each contain at least one
    SCADA node (fewer than top_n combos available counts as False)."""
    top = ranked[:top_n]
    return len(top) == top_n and all(scada_nodes & set(entry["combo"]) for entry in top)


def run_sweep(
    variants=VARIANTS,
    budgets=BUDGETS,
    reductions=REDUCTIONS,
    metrics=METRICS,
    t_max=24,
    n_runs=DEFAULT_N_RUNS,
    seed=DEFAULT_SEED,
):
    """Evaluate every variant x budget x reduction cell once, then rank by
    every metric. Returns rows: a list of dicts, one per
    (variant, budget, reduction, metric) - exactly what gets written to
    topology_robustness.csv.
    """
    rows = []

    for variant in variants:
        graph = build_smart_grid(variant=variant)
        ctmc_nodes = build_ctmc_nodes(graph)
        scada_nodes = _scada_nodes(graph)
        excluded_nodes = [TARGET_NODE]

        for budget in budgets:
            for reduction in reductions:
                detailed = evaluate_combinations(
                    graph, ctmc_nodes, budget=budget, t_max=t_max,
                    n_runs=n_runs, seed=seed, excluded_nodes=excluded_nodes,
                    reduction=reduction,
                )

                for metric in metrics:
                    ranked = rank_by_metric(detailed, metric, t_max=t_max)
                    best = ranked[0]
                    rows.append(
                        {
                            "variant": variant,
                            "budget": budget,
                            "reduction": reduction,
                            "metric": metric,
                            "best_combo": "+".join(best["combo"]),
                            "metric_value": best["metric_value"],
                            "p_fail_6h": best["p_fail_by_t"][6]["p"],
                            "p_fail_24h": best["probability_of_failure"],
                            "mean_time_to_failure": best["mean_time_to_failure"],
                            "scada_in_top3": _top3_has_scada(ranked, scada_nodes),
                            "n_combos_evaluated": len(detailed),
                            "n_runs": n_runs,
                            "seed": seed,
                        }
                    )

    return rows


def write_csv(rows, path=os.path.join(RESULTS_DIR, "topology_robustness.csv")):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fieldnames = [
        "variant", "budget", "reduction", "metric", "best_combo", "metric_value",
        "p_fail_6h", "p_fail_24h", "mean_time_to_failure", "scada_in_top3",
        "n_combos_evaluated", "n_runs", "seed",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def print_summary(rows, variants=VARIANTS, budgets=BUDGETS, reductions=REDUCTIONS, metrics=METRICS):
    print("\nVariant x metric -> best combo (every budget x reduction cell):")
    overall_by_variant = {}

    for variant in variants:
        print(f"\n  variant={variant}:")
        variant_rows = [r for r in rows if r["variant"] == variant]

        for budget in budgets:
            for reduction in reductions:
                cell_rows = [
                    r for r in variant_rows
                    if r["budget"] == budget and r["reduction"] == reduction
                ]
                if not cell_rows:
                    continue
                print(f"    budget={budget}, reduction={reduction}:")
                for metric in metrics:
                    r = next((x for x in cell_rows if x["metric"] == metric), None)
                    if r is None:
                        continue
                    print(
                        f"      {metric:10s} -> best={r['best_combo']:<45s} "
                        f"scada_in_top3={r['scada_in_top3']}"
                    )

        always_scada = all(r["scada_in_top3"] for r in variant_rows)
        overall_by_variant[variant] = always_scada
        print(
            f"  => SCADA node in top-3 across ALL {len(variant_rows)} "
            f"(budget x reduction x metric) cells for '{variant}': {always_scada}"
        )

    print("\nOverall robustness verdict:")
    for variant, always_scada in overall_by_variant.items():
        verdict = "chokepoint HOLDS" if always_scada else "chokepoint DOES NOT consistently hold"
        print(f"  {variant:12s} -> {verdict}")


if __name__ == "__main__":
    print(
        f"Running topology robustness sweep: variants={VARIANTS} x budgets={BUDGETS} "
        f"x reductions={REDUCTIONS} x metrics={METRICS}, {TARGET_NODE} excluded, "
        f"n_runs={DEFAULT_N_RUNS}, seed={DEFAULT_SEED}"
    )
    rows = run_sweep()

    csv_path = write_csv(rows)
    print(f"\nWrote {len(rows)} rows to {csv_path}")

    print_summary(rows)
