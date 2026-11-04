"""Null model and larger-topology stress test for the min-cut hypothesis.

src.mincut_vs_simulation found that, on the small hand-drawn topologies,
the simulation's best defense combo was ALWAYS a full structural vertex
cut (48/48 cells) and almost always overlapped a true minimum cut
(47/48). That's weak evidence on its own: these graphs are so sparse
that most budget-2/3 node combinations are already vertex cuts, so a
high hit rate could be mostly chance. This module adds:

1. A null model: for each of the 4 small topology variants and each
   budget, the fraction of ALL candidate combinations (entry point(s)
   and physical_equipment excluded) that are valid vertex cuts, and how
   many of the 48 cells in the original sweep that fraction would
   predict "match" by pure chance.
2. A harder test of the hypothesis: among combos that ARE vertex cuts of
   a given size, does the simulation's ranking actually correlate with
   how close the cut is to the entry point (mean hop distance)? If the
   min-cut story is right beyond "it's a cut", defenses closer to the
   entry (cheaper to reach, same blocking effect) might behave
   differently from ones deep in the graph - this is tested with a
   Spearman correlation per variant/budget/reduction/metric between
   each cut combo's simulated metric value and its mean hop distance
   from the entry point(s).
3. Two synthetic ~15-node layered topologies ("large_layered",
   "large_random", defined in src/graph.py) on which the true minimum
   vertex cut has size 3 (the field-layer bottleneck), so most budget-2
   combos can't be cuts and most budget-3 combos aren't either - a much
   harder test than the small graphs where budget >= min cut size nearly
   always.
4. A full budget(2,3,4) x reduction(0.5,0.9) x metric(p_fail_6h,
   mean_ttf) resimulation on those two large topologies, reporting how
   often the simulation's winner is a minimum cut, a superset of one,
   overlaps one partially, or doesn't touch one at all.

Running this module directly writes results/mincut_null_and_large.csv
(one CSV, "section" column distinguishes the three kinds of rows) and
prints a summary. Takes several minutes (the large-topology sweep runs
~1080 Monte Carlo evaluations per variant x reduction cell).
"""

import csv
import itertools
import os
import random

import networkx as nx
from scipy.stats import spearmanr

from src.ctmc import build_ctmc_nodes
from src.graph import build_smart_grid
from src.mincut_vs_simulation import candidate_nodes, classify_match, disconnects, min_vertex_cuts, weighted_pick
from src.optimization import apply_defense, evaluate_combinations, rank_by_metric
from src.simulation import TARGET_NODE, run_monte_carlo
from src.stats import wilson_confidence_interval
from src.visualization import RESULTS_DIR

OUT_CSV = os.path.join(RESULTS_DIR, "mincut_null_and_large.csv")

ORIGINAL_VARIANTS = ("baseline", "redundant", "flat", "multi_entry")
ORIGINAL_BUDGETS = (2, 3)
ORIGINAL_REDUCTIONS = (0.5, 0.9)
ORIGINAL_METRICS = ("p_fail_6h", "p_fail_24h", "mean_ttf")
N_CELLS_ORIGINAL_SWEEP = len(ORIGINAL_VARIANTS) * len(ORIGINAL_BUDGETS) * len(ORIGINAL_REDUCTIONS) * len(ORIGINAL_METRICS)

SPEARMAN_N_RUNS = 1000
SPEARMAN_SEED = 42
SPEARMAN_T_MAX = 24

LARGE_VARIANTS = ("large_layered", "large_random")
LARGE_BUDGETS = (2, 3, 4)
LARGE_REDUCTIONS = (0.5, 0.9)
LARGE_METRICS = ("p_fail_6h", "mean_ttf")
LARGE_N_RUNS = 200
LARGE_T_STEPS = 80
LARGE_T_MAX = 24
LARGE_SEED = 42

FIELDNAMES = [
    "section", "variant", "budget", "reduction", "metric",
    "n_candidates", "n_combos_total", "n_combos_cut", "fraction_cut", "expected_chance_matches",
    "n_cut_combos", "spearman_corr", "spearman_pvalue",
    "min_cut_size", "predicted_min_cuts", "sim_best_combo", "mean_hops_of_winner",
    "is_cut", "match_status", "notes",
]


def mean_hops(graph, entries, combo):
    """Mean hop distance (shortest path, directed) from any entry point
    to any node in combo, over (entry, node) pairs where the node is
    actually reachable from that entry (for "multi_entry", not every
    combo node is reachable from every entry - e.g. historian_server is
    not downstream of engineering_workstation - so unreachable pairs are
    skipped rather than erroring)."""
    total, count = 0.0, 0
    for entry in entries:
        for node in combo:
            if nx.has_path(graph, entry, node):
                total += nx.shortest_path_length(graph, entry, node)
                count += 1
    return total / count if count else None


# ---------------------------------------------------------------------------
# Part 1: null model
# ---------------------------------------------------------------------------

def null_model_rows():
    rows = []
    for variant in ORIGINAL_VARIANTS:
        graph = build_smart_grid(variant)
        entries = list(graph.graph["entry_points"].keys())
        candidates = candidate_nodes(graph, entries)

        for budget in ORIGINAL_BUDGETS:
            combos = list(itertools.combinations(candidates, budget))
            n_cut = sum(1 for c in combos if disconnects(graph, entries, c))
            fraction = n_cut / len(combos) if combos else 0.0
            # this (variant, budget) group covers len(REDUCTIONS)*len(METRICS) cells
            n_cells_in_group = len(ORIGINAL_REDUCTIONS) * len(ORIGINAL_METRICS)
            rows.append({
                "section": "null_model",
                "variant": variant,
                "budget": budget,
                "n_candidates": len(candidates),
                "n_combos_total": len(combos),
                "n_combos_cut": n_cut,
                "fraction_cut": round(fraction, 4),
                "expected_chance_matches": round(fraction * n_cells_in_group, 3),
                "notes": f"covers {n_cells_in_group} of the original sweep's {N_CELLS_ORIGINAL_SWEEP} cells",
            })
    return rows


# ---------------------------------------------------------------------------
# Part 2: Spearman correlation (metric value vs. hop distance, cut combos only)
# ---------------------------------------------------------------------------

def spearman_rows():
    rows = []
    for variant in ORIGINAL_VARIANTS:
        graph = build_smart_grid(variant)
        ctmc_nodes = build_ctmc_nodes(graph)
        entries = list(graph.graph["entry_points"].keys())
        excluded = entries + [TARGET_NODE]

        for budget in ORIGINAL_BUDGETS:
            for reduction in ORIGINAL_REDUCTIONS:
                detailed = evaluate_combinations(
                    graph, ctmc_nodes, budget=budget, t_max=SPEARMAN_T_MAX,
                    n_runs=SPEARMAN_N_RUNS, seed=SPEARMAN_SEED,
                    excluded_nodes=excluded, reduction=reduction,
                )
                cut_combos = [d for d in detailed if disconnects(graph, entries, d["combo"])]

                for metric in ORIGINAL_METRICS:
                    ranked = rank_by_metric(cut_combos, metric, t_max=SPEARMAN_T_MAX)
                    values = [d["metric_value"] for d in ranked]
                    hops = [mean_hops(graph, entries, d["combo"]) for d in ranked]

                    n = len(values)
                    if n >= 3 and len(set(hops)) > 1:
                        corr, pvalue = spearmanr(values, hops)
                    else:
                        corr, pvalue = None, None

                    rows.append({
                        "section": "spearman",
                        "variant": variant,
                        "budget": budget,
                        "reduction": reduction,
                        "metric": metric,
                        "n_cut_combos": n,
                        "n_combos_total": len(detailed),
                        "spearman_corr": round(corr, 4) if corr is not None else None,
                        "spearman_pvalue": round(pvalue, 4) if pvalue is not None else None,
                        "notes": "corr between metric_value and mean hop-distance of cut nodes from entry; n/a if <3 cut combos or all same hop distance",
                    })
    return rows


# ---------------------------------------------------------------------------
# Part 4: large-topology resweep
# ---------------------------------------------------------------------------

def evaluate_combos_custom(graph, ctmc_nodes, combos, t_max, t_steps, n_runs, seed, reduction):
    detailed = []
    for combo in combos:
        defended = apply_defense(ctmc_nodes, combo, reduction=reduction)
        mc = run_monte_carlo(graph, defended, n_runs=n_runs, t_max=t_max, t_steps=t_steps, seed=seed)
        p = mc["probability_of_failure"]
        ci_low, ci_high = wilson_confidence_interval(p, n_runs)
        detailed.append({
            "combo": combo,
            "probability_of_failure": p,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "mean_time_to_failure": mc["mean_time_to_failure"],
            "p_fail_by_t": mc["p_fail_by_t"],
            "n_runs": n_runs,
            "seed": seed,
            "reduction": reduction,
        })
    return detailed


def fmt_cuts(cuts):
    return " | ".join("{" + ",".join(sorted(c)) + "}" for c in cuts)


def large_sweep_rows():
    rows = []
    for variant in LARGE_VARIANTS:
        graph = build_smart_grid(variant)
        ctmc_nodes = build_ctmc_nodes(graph)
        entries = list(graph.graph["entry_points"].keys())
        candidates = candidate_nodes(graph, entries)

        min_k, min_cuts = min_vertex_cuts(graph, entries)
        print(f"  [{variant}] n_candidates={len(candidates)}  min_cut_size={min_k}  "
              f"n_min_cuts={len(min_cuts)}  example={fmt_cuts(min_cuts[:3])}")

        for budget in LARGE_BUDGETS:
            combos = list(itertools.combinations(candidates, budget))
            n_cut = sum(1 for c in combos if disconnects(graph, entries, c))
            fraction = n_cut / len(combos) if combos else 0.0
            print(f"    budget={budget}: {n_cut}/{len(combos)} combos are cuts ({fraction:.3f})")

            rows.append({
                "section": "null_model_large",
                "variant": variant,
                "budget": budget,
                "n_candidates": len(candidates),
                "n_combos_total": len(combos),
                "n_combos_cut": n_cut,
                "fraction_cut": round(fraction, 4),
                "min_cut_size": min_k,
                "predicted_min_cuts": fmt_cuts(min_cuts),
            })

            for reduction in LARGE_REDUCTIONS:
                detailed = evaluate_combos_custom(
                    graph, ctmc_nodes, combos, t_max=LARGE_T_MAX, t_steps=LARGE_T_STEPS,
                    n_runs=LARGE_N_RUNS, seed=LARGE_SEED, reduction=reduction,
                )

                for metric in LARGE_METRICS:
                    ranked = rank_by_metric(detailed, metric, t_max=LARGE_T_MAX)
                    best = ranked[0]
                    combo = set(best["combo"])
                    is_cut = disconnects(graph, entries, combo)
                    status = classify_match(combo, min_cuts)

                    rows.append({
                        "section": "large_sweep",
                        "variant": variant,
                        "budget": budget,
                        "reduction": reduction,
                        "metric": metric,
                        "min_cut_size": min_k,
                        "predicted_min_cuts": fmt_cuts(min_cuts),
                        "sim_best_combo": "+".join(sorted(combo)),
                        "mean_hops_of_winner": round(mean_hops(graph, entries, combo), 3),
                        "is_cut": is_cut,
                        "match_status": status,
                    })
    return rows


def write_csv(rows, path=OUT_CSV):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in FIELDNAMES})
    return path


def print_summary(null_rows, spear_rows, large_rows):
    print("\n=== Part 1: null model (original 4 variants) ===")
    total_expected = 0.0
    for r in null_rows:
        print(
            f"  {r['variant']:12s} budget={r['budget']}  "
            f"{r['n_combos_cut']}/{r['n_combos_total']} combos are cuts "
            f"(fraction={r['fraction_cut']:.3f})  -> expected chance-matches "
            f"in this group's {len(ORIGINAL_REDUCTIONS)*len(ORIGINAL_METRICS)} cells: {r['expected_chance_matches']:.2f}"
        )
        total_expected += r["expected_chance_matches"]
    print(f"\n  Total expected chance-matches across all {N_CELLS_ORIGINAL_SWEEP} cells: {total_expected:.2f}"
          f"  (observed in src.mincut_vs_simulation: 47/48 contained a min-cut node, 48/48 were full cuts)")

    print("\n=== Part 2: Spearman(metric_value, mean hop-distance) among cut combos ===")
    by_variant_metric = {}
    for r in spear_rows:
        key = (r["variant"], r["metric"])
        by_variant_metric.setdefault(key, []).append(r)
        corr_str = f"{r['spearman_corr']:+.3f} (p={r['spearman_pvalue']:.3f})" if r["spearman_corr"] is not None else "n/a"
        print(
            f"  {r['variant']:12s} budget={r['budget']} reduction={r['reduction']} metric={r['metric']:10s} "
            f"n_cut_combos={r['n_cut_combos']:2d}  corr={corr_str}"
        )

    print("\n  Per-variant x metric average (across budget x reduction, ignoring n/a):")
    for (variant, metric), group in by_variant_metric.items():
        corrs = [r["spearman_corr"] for r in group if r["spearman_corr"] is not None]
        if corrs:
            avg = sum(corrs) / len(corrs)
            print(f"    {variant:12s} {metric:10s} -> mean corr={avg:+.3f}  (n_groups={len(corrs)}/{len(group)})")
        else:
            print(f"    {variant:12s} {metric:10s} -> n/a (no group had enough cut combos)")

    print("\n=== Part 3/4: large synthetic topologies ===")
    null_large = [r for r in large_rows if r["section"] == "null_model_large"]
    sweep = [r for r in large_rows if r["section"] == "large_sweep"]

    for r in null_large:
        print(
            f"  {r['variant']:14s} budget={r['budget']}  min_cut_size={r['min_cut_size']}  "
            f"{r['n_combos_cut']}/{r['n_combos_total']} combos are cuts (fraction={r['fraction_cut']:.4f})"
        )

    print("\n  Simulation winner vs. minimum cut, per cell:")
    counts = {}
    for r in sweep:
        counts[r["match_status"]] = counts.get(r["match_status"], 0) + 1
        print(
            f"    {r['variant']:14s} budget={r['budget']} reduction={r['reduction']} metric={r['metric']:10s} "
            f"winner={r['sim_best_combo']:40s} is_cut={str(r['is_cut']):5s} status={r['match_status']}"
        )

    n_total = len(sweep)
    n_is_cut = sum(1 for r in sweep if r["is_cut"])
    print(f"\n  Across {n_total} large-topology cells:")
    print(f"    winner IS a full vertex cut:      {n_is_cut}/{n_total}")
    print(f"    winner is NOT a full cut:         {n_total - n_is_cut}/{n_total}")
    for status, count in counts.items():
        print(f"    match_status={status:16s}: {count}/{n_total}")


if __name__ == "__main__":
    print("Running null model on original variants...")
    null_rows = null_model_rows()

    print("Running Spearman correlation analysis (this resimulates every cut combo)...")
    spear_rows = spearman_rows()

    print("Running large-topology sweep (budget 2/3/4 x reduction x metric)...")
    large_rows = large_sweep_rows()

    all_rows = null_rows + spear_rows + large_rows
    csv_path = write_csv(all_rows)
    print(f"\nWrote {len(all_rows)} rows to {csv_path}")

    print_summary(null_rows, spear_rows, large_rows)
