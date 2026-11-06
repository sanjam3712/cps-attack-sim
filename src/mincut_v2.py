"""Min-cut hypothesis, round 2: sanity-checked, denoised, and re-predicted.

Addresses four weaknesses in the prior analyses
(src.mincut_vs_simulation, src.mincut_null_and_large):

1. Full-removal sanity check: with defense reduction=1.0 (compromise_rate
   forced to 0 on defended nodes), a node can never transition out of
   SECURE, so a defended node is - for propagation purposes - exactly
   equivalent to deleting it from the graph. This makes "P(fail by 24h)
   == 0 iff the combo is a vertex cut" a deterministic prediction, not a
   statistical one. Checking it across all 6 variants is a sanity check
   on the simulation code itself, independent of any Monte Carlo noise
   question.

2. The large-topology sweep in src.mincut_null_and_large used n_runs=200
   per combo to stay tractable across ~1080 combos per cell - too noisy
   to trust a single "winner". This screens at n_runs=200 as before,
   then re-runs only the top 30 (per metric, deduplicated across the two
   metrics sharing a screen) at n_runs=5000 with common random numbers,
   and reports the winner's Wilson CI and whether it overlaps the
   runner-up's.

3. On the large topologies the true minimum cut has size 3, so every
   budget=2 cell is structurally incapable of producing a cut winner.
   Lumping those into "winner is/isn't a cut" statistics would bias the
   apparent failure rate of the hypothesis for a trivial reason. Those
   cells are counted and reported separately.

4. Hop-distance-from-entry (used in src.mincut_null_and_large) was a
   weak, somewhat arbitrary proxy. This replaces it with two structural
   predictors per combo: (a) the number of distinct descendants
   (downstream-reachable nodes) of the defended nodes, and (b) the sum
   of the defended nodes' vulnerability. Both are correlated (Spearman)
   against each metric among same-size cut combos, plus the partial
   Spearman correlation of descendants controlling for vulnerability,
   each with a bootstrap 95% CI.

Running this module directly writes results/mincut_v2.csv (one CSV,
"section" distinguishes the four kinds of rows) and prints a summary.
This is a long-running script (the budget=4 large-topology rerun alone
evaluates hundreds of combos at n_runs=5000) - expect tens of minutes.
"""

import csv
import itertools
import os

import networkx as nx
import numpy as np
from scipy.stats import spearmanr

from src.ctmc import build_ctmc_nodes
from src.graph import VARIANTS, build_smart_grid
from src.mincut_null_and_large import evaluate_combos_custom, fmt_cuts
from src.mincut_vs_simulation import candidate_nodes, classify_match, disconnects, min_vertex_cuts
from src.optimization import apply_defense, evaluate_combinations, rank_by_metric
from src.simulation import TARGET_NODE, run_monte_carlo
from src.stats import intervals_overlap
from src.visualization import RESULTS_DIR

OUT_CSV = os.path.join(RESULTS_DIR, "mincut_v2.csv")

ORIGINAL_VARIANTS = ("baseline", "redundant", "flat", "multi_entry")
LARGE_VARIANTS = ("large_layered", "large_random")

# Part 1: full-removal sanity check
FULL_REMOVAL_BUDGETS = (2, 3, 4)
FULL_REMOVAL_REDUCTION = 1.0
FULL_REMOVAL_N_RUNS = 200
FULL_REMOVAL_SEED = 42
FULL_REMOVAL_T_MAX = 24
FULL_REMOVAL_T_STEPS = 100

# Part 2: large-topology screen + top-30 CRN rerun
LARGE_BUDGETS = (2, 3, 4)
LARGE_REDUCTIONS = (0.5, 0.9)
LARGE_METRICS = ("p_fail_6h", "mean_ttf")
SCREEN_N_RUNS = 200
SCREEN_T_STEPS = 80
RERUN_N_RUNS = 5000
RERUN_T_STEPS = 80
LARGE_T_MAX = 24
LARGE_SEED = 42
TOP_N_RERUN = 30

# Part 4: predictor correlations (original variants use a fresh resim;
# large variants reuse the Part-2 screening pass)
ORIGINAL_BUDGETS = (2, 3)
ORIGINAL_REDUCTIONS = (0.5, 0.9)
PREDICTOR_METRICS = ("p_fail_6h", "p_fail_24h", "mean_ttf")
PREDICTOR_N_RUNS = 1000
PREDICTOR_SEED = 42
PREDICTOR_T_MAX = 24
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 123

FIELDNAMES = [
    "section", "variant", "budget", "reduction", "metric",
    "combo", "is_cut", "p_fail_24h", "violation",
    "min_cut_size", "budget_below_mincut", "winner_combo", "winner_metric_value",
    "winner_ci_low", "winner_ci_high", "runner_up_combo", "runner_up_metric_value",
    "ci_overlaps_runner_up", "match_status", "n_screened", "n_rerun_union",
    "n_cut_combos",
    "corr_descendants", "corr_descendants_ci_low", "corr_descendants_ci_high", "excludes_zero_descendants",
    "corr_vuln", "corr_vuln_ci_low", "corr_vuln_ci_high", "excludes_zero_vuln",
    "partial_corr_descendants_given_vuln", "partial_ci_low", "partial_ci_high", "excludes_zero_partial",
    "notes",
]


# ---------------------------------------------------------------------------
# Part 1: full-removal sanity check
# ---------------------------------------------------------------------------

def full_removal_rows():
    rows = []
    for variant in VARIANTS:
        graph = build_smart_grid(variant)
        ctmc_nodes = build_ctmc_nodes(graph)
        entries = list(graph.graph["entry_points"].keys())
        candidates = candidate_nodes(graph, entries)

        for budget in FULL_REMOVAL_BUDGETS:
            if budget > len(candidates):
                continue
            for combo in itertools.combinations(candidates, budget):
                is_cut = disconnects(graph, entries, combo)
                defended = apply_defense(ctmc_nodes, combo, reduction=FULL_REMOVAL_REDUCTION)
                mc = run_monte_carlo(
                    graph, defended, n_runs=FULL_REMOVAL_N_RUNS, t_max=FULL_REMOVAL_T_MAX,
                    t_steps=FULL_REMOVAL_T_STEPS, seed=FULL_REMOVAL_SEED,
                )
                p24 = mc["probability_of_failure"]
                violation = (is_cut and p24 != 0.0) or ((not is_cut) and p24 == 0.0)
                rows.append({
                    "section": "full_removal",
                    "variant": variant,
                    "budget": budget,
                    "combo": "+".join(sorted(combo)),
                    "is_cut": is_cut,
                    "p_fail_24h": p24,
                    "violation": violation,
                })
    return rows


# ---------------------------------------------------------------------------
# Part 4 predictors
# ---------------------------------------------------------------------------

def descendants_count(graph, combo):
    nodes = set()
    for n in combo:
        nodes |= nx.descendants(graph, n)
    return len(nodes)


def vulnerability_sum(graph, combo):
    return sum(graph.nodes[n]["vulnerability"] for n in combo)


def partial_spearman(x, y, z):
    """Partial Spearman correlation of x,y controlling for z: the
    standard partial-correlation formula applied to pairwise Spearman
    rhos."""
    rxy = spearmanr(x, y).correlation
    rxz = spearmanr(x, z).correlation
    ryz = spearmanr(y, z).correlation
    denom = np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
    if denom == 0 or np.isnan(denom):
        return float("nan")
    return (rxy - rxz * ryz) / denom


def bootstrap_ci(arrays, stat_fn, n_boot=N_BOOTSTRAP, seed=BOOTSTRAP_SEED):
    rng = np.random.default_rng(seed)
    n = len(arrays[0])
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        resampled = [np.asarray(a)[idx] for a in arrays]
        try:
            s = stat_fn(*resampled)
        except Exception:
            s = float("nan")
        if not np.isnan(s):
            stats.append(s)
    if not stats:
        return (float("nan"), float("nan"))
    return (float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5)))


def excludes_zero(ci_low, ci_high):
    if ci_low is None or ci_high is None or np.isnan(ci_low) or np.isnan(ci_high):
        return None
    return not (ci_low <= 0 <= ci_high)


def predictor_correlation_row(graph, cut_detailed, metric, variant, budget, reduction, t_max):
    ranked = rank_by_metric(cut_detailed, metric, t_max=t_max)
    n = len(ranked)

    row = {
        "section": "predictor_corr",
        "variant": variant,
        "budget": budget,
        "reduction": reduction,
        "metric": metric,
        "n_cut_combos": n,
    }

    metric_vals = np.array([d["metric_value"] for d in ranked], dtype=float)
    desc_vals = np.array([descendants_count(graph, d["combo"]) for d in ranked], dtype=float)
    vuln_vals = np.array([vulnerability_sum(graph, d["combo"]) for d in ranked], dtype=float)

    if n < 4 or len(set(desc_vals)) < 2 or len(set(vuln_vals)) < 2:
        row["notes"] = "insufficient sample/variation (need n>=4 and both predictors non-constant)"
        return row

    corr_desc = spearmanr(metric_vals, desc_vals).correlation
    corr_vuln = spearmanr(metric_vals, vuln_vals).correlation
    partial = partial_spearman(metric_vals, desc_vals, vuln_vals)

    ci_desc = bootstrap_ci((metric_vals, desc_vals), lambda a, b: spearmanr(a, b).correlation)
    ci_vuln = bootstrap_ci((metric_vals, vuln_vals), lambda a, b: spearmanr(a, b).correlation)
    ci_partial = bootstrap_ci((metric_vals, desc_vals, vuln_vals), partial_spearman)

    row.update({
        "corr_descendants": round(corr_desc, 4) if not np.isnan(corr_desc) else None,
        "corr_descendants_ci_low": round(ci_desc[0], 4) if not np.isnan(ci_desc[0]) else None,
        "corr_descendants_ci_high": round(ci_desc[1], 4) if not np.isnan(ci_desc[1]) else None,
        "excludes_zero_descendants": excludes_zero(*ci_desc),
        "corr_vuln": round(corr_vuln, 4) if not np.isnan(corr_vuln) else None,
        "corr_vuln_ci_low": round(ci_vuln[0], 4) if not np.isnan(ci_vuln[0]) else None,
        "corr_vuln_ci_high": round(ci_vuln[1], 4) if not np.isnan(ci_vuln[1]) else None,
        "excludes_zero_vuln": excludes_zero(*ci_vuln),
        "partial_corr_descendants_given_vuln": round(partial, 4) if not np.isnan(partial) else None,
        "partial_ci_low": round(ci_partial[0], 4) if not np.isnan(ci_partial[0]) else None,
        "partial_ci_high": round(ci_partial[1], 4) if not np.isnan(ci_partial[1]) else None,
        "excludes_zero_partial": excludes_zero(*ci_partial),
        "notes": "",
    })
    return row


def original_predictor_rows():
    rows = []
    for variant in ORIGINAL_VARIANTS:
        graph = build_smart_grid(variant)
        ctmc_nodes = build_ctmc_nodes(graph)
        entries = list(graph.graph["entry_points"].keys())
        excluded = entries + [TARGET_NODE]

        for budget in ORIGINAL_BUDGETS:
            for reduction in ORIGINAL_REDUCTIONS:
                detailed = evaluate_combinations(
                    graph, ctmc_nodes, budget=budget, t_max=PREDICTOR_T_MAX,
                    n_runs=PREDICTOR_N_RUNS, seed=PREDICTOR_SEED,
                    excluded_nodes=excluded, reduction=reduction,
                )
                cut_combos = [d for d in detailed if disconnects(graph, entries, d["combo"])]

                for metric in PREDICTOR_METRICS:
                    rows.append(
                        predictor_correlation_row(graph, cut_combos, metric, variant, budget, reduction, PREDICTOR_T_MAX)
                    )
    return rows


# ---------------------------------------------------------------------------
# Parts 2+3+4(large): shared screening pass per (variant, budget, reduction)
# ---------------------------------------------------------------------------

def large_topology_analysis():
    part2_rows = []
    part4_rows = []

    for variant in LARGE_VARIANTS:
        graph = build_smart_grid(variant)
        ctmc_nodes = build_ctmc_nodes(graph)
        entries = list(graph.graph["entry_points"].keys())
        candidates = candidate_nodes(graph, entries)
        min_k, min_cuts = min_vertex_cuts(graph, entries)
        print(f"  [{variant}] min_cut_size={min_k}  n_candidates={len(candidates)}")

        for budget in LARGE_BUDGETS:
            combos = list(itertools.combinations(candidates, budget))
            budget_below_mincut = budget < min_k

            for reduction in LARGE_REDUCTIONS:
                print(f"    screening budget={budget} reduction={reduction}: {len(combos)} combos @ n_runs={SCREEN_N_RUNS}")
                detailed_screen = evaluate_combos_custom(
                    graph, ctmc_nodes, combos, t_max=LARGE_T_MAX, t_steps=SCREEN_T_STEPS,
                    n_runs=SCREEN_N_RUNS, seed=LARGE_SEED, reduction=reduction,
                )

                cut_detailed = [d for d in detailed_screen if disconnects(graph, entries, d["combo"])]
                for metric in LARGE_METRICS:
                    part4_rows.append(
                        predictor_correlation_row(graph, cut_detailed, metric, variant, budget, reduction, LARGE_T_MAX)
                    )

                top_combo_sets = {}
                for metric in LARGE_METRICS:
                    ranked = rank_by_metric(detailed_screen, metric, t_max=LARGE_T_MAX)
                    top_combo_sets[metric] = [d["combo"] for d in ranked[:TOP_N_RERUN]]

                union_combos = list({c for combo_list in top_combo_sets.values() for c in combo_list})
                print(f"    reruning top-30 union ({len(union_combos)} unique combos) @ n_runs={RERUN_N_RUNS}")
                rerun_detailed = evaluate_combos_custom(
                    graph, ctmc_nodes, union_combos, t_max=LARGE_T_MAX, t_steps=RERUN_T_STEPS,
                    n_runs=RERUN_N_RUNS, seed=LARGE_SEED, reduction=reduction,
                )
                rerun_by_combo = {d["combo"]: d for d in rerun_detailed}

                for metric in LARGE_METRICS:
                    top30_detailed = [rerun_by_combo[c] for c in top_combo_sets[metric]]
                    ranked_rerun = rank_by_metric(top30_detailed, metric, t_max=LARGE_T_MAX)
                    winner = ranked_rerun[0]
                    runner_up = ranked_rerun[1] if len(ranked_rerun) > 1 else None

                    has_ci = metric != "mean_ttf"
                    overlap = None
                    if has_ci and runner_up is not None:
                        overlap = intervals_overlap(
                            (winner["metric_ci_low"], winner["metric_ci_high"]),
                            (runner_up["metric_ci_low"], runner_up["metric_ci_high"]),
                        )

                    winner_combo = set(winner["combo"])
                    is_cut = disconnects(graph, entries, winner_combo)
                    status = classify_match(winner_combo, min_cuts)

                    part2_rows.append({
                        "section": "large_rerun",
                        "variant": variant,
                        "budget": budget,
                        "reduction": reduction,
                        "metric": metric,
                        "min_cut_size": min_k,
                        "budget_below_mincut": budget_below_mincut,
                        "winner_combo": "+".join(sorted(winner["combo"])),
                        "winner_metric_value": winner["metric_value"],
                        "winner_ci_low": winner.get("metric_ci_low"),
                        "winner_ci_high": winner.get("metric_ci_high"),
                        "runner_up_combo": "+".join(sorted(runner_up["combo"])) if runner_up else "",
                        "runner_up_metric_value": runner_up["metric_value"] if runner_up else "",
                        "ci_overlaps_runner_up": overlap,
                        "is_cut": is_cut,
                        "match_status": status,
                        "n_screened": len(combos),
                        "n_rerun_union": len(union_combos),
                    })

    return part2_rows, part4_rows


def write_csv(rows, path=OUT_CSV):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in FIELDNAMES})
    return path


def print_summary(full_removal, part2_rows, part4_large, part4_original):
    print("\n=== Part 1: full-removal sanity check ===")
    violations = [r for r in full_removal if r["violation"]]
    print(f"  Checked {len(full_removal)} combos across all 6 variants (budgets 2,3,4; reduction=1.0).")
    print(f"  Violations: {len(violations)}")
    for r in violations[:20]:
        print(f"    VIOLATION: {r['variant']} combo={r['combo']} is_cut={r['is_cut']} p_fail_24h={r['p_fail_24h']}")
    n_cuts = sum(1 for r in full_removal if r["is_cut"])
    n_noncuts = len(full_removal) - n_cuts
    print(f"  {n_cuts} combos were cuts (all should show p_fail_24h==0), {n_noncuts} were not (all should show >0).")

    print("\n=== Part 2/3: large-topology top-30 CRN rerun (n_runs=5000) ===")
    for r in part2_rows:
        below = " [budget<mincut, EXCLUDED from is_cut stats]" if r["budget_below_mincut"] else ""
        ci_str = (
            f"CI=[{r['winner_ci_low']:.4f},{r['winner_ci_high']:.4f}]"
            if r["winner_ci_low"] is not None else "no CI (mean_ttf)"
        )
        overlap_str = (
            "overlaps runner-up" if r["ci_overlaps_runner_up"] else "distinguishable from runner-up"
        ) if r["ci_overlaps_runner_up"] is not None else "n/a"
        print(
            f"  {r['variant']:14s} budget={r['budget']} reduction={r['reduction']} metric={r['metric']:10s}{below}\n"
            f"    winner={r['winner_combo']:40s} value={r['winner_metric_value']:.4f} {ci_str}  ({overlap_str})\n"
            f"    runner_up={r['runner_up_combo']:40s} value={r['runner_up_metric_value']}\n"
            f"    is_cut={r['is_cut']} match_status={r['match_status']}"
        )

    included = [r for r in part2_rows if not r["budget_below_mincut"]]
    excluded = [r for r in part2_rows if r["budget_below_mincut"]]
    n_included_cut = sum(1 for r in included if r["is_cut"])
    print(f"\n  Cells with budget < min_cut_size (structurally excluded): {len(excluded)}/{len(part2_rows)} "
          f"(all necessarily is_cut=False)")
    print(f"  Cells with budget >= min_cut_size: {len(included)}/{len(part2_rows)} "
          f"-> winner is a cut in {n_included_cut}/{len(included)}")

    print("\n=== Part 4: predictor correlations (descendants, vulnerability sum) ===")
    print("\n  -- Large topologies (screened at n_runs=200, cut combos only) --")
    for r in part4_large:
        if r.get("corr_descendants") is None:
            print(f"  {r['variant']:14s} budget={r['budget']} reduction={r['reduction']} metric={r['metric']:10s} "
                  f"n={r['n_cut_combos']:3d}  {r.get('notes', '')}")
            continue
        print(
            f"  {r['variant']:14s} budget={r['budget']} reduction={r['reduction']} metric={r['metric']:10s} n={r['n_cut_combos']:3d}\n"
            f"    descendants: corr={r['corr_descendants']:+.3f} CI=[{r['corr_descendants_ci_low']:+.3f},{r['corr_descendants_ci_high']:+.3f}] excl0={r['excludes_zero_descendants']}\n"
            f"    vuln_sum:    corr={r['corr_vuln']:+.3f} CI=[{r['corr_vuln_ci_low']:+.3f},{r['corr_vuln_ci_high']:+.3f}] excl0={r['excludes_zero_vuln']}\n"
            f"    partial(desc|vuln): corr={r['partial_corr_descendants_given_vuln']} "
            f"CI=[{r['partial_ci_low']},{r['partial_ci_high']}] excl0={r['excludes_zero_partial']}"
        )

    print("\n  -- Original 4 variants (resimulated at n_runs=1000, cut combos only) --")
    for r in part4_original:
        if r.get("corr_descendants") is None:
            print(f"  {r['variant']:12s} budget={r['budget']} reduction={r['reduction']} metric={r['metric']:10s} "
                  f"n={r['n_cut_combos']:3d}  {r.get('notes', '')}")
            continue
        print(
            f"  {r['variant']:12s} budget={r['budget']} reduction={r['reduction']} metric={r['metric']:10s} n={r['n_cut_combos']:3d}\n"
            f"    descendants: corr={r['corr_descendants']:+.3f} CI=[{r['corr_descendants_ci_low']:+.3f},{r['corr_descendants_ci_high']:+.3f}] excl0={r['excludes_zero_descendants']}\n"
            f"    vuln_sum:    corr={r['corr_vuln']:+.3f} CI=[{r['corr_vuln_ci_low']:+.3f},{r['corr_vuln_ci_high']:+.3f}] excl0={r['excludes_zero_vuln']}\n"
            f"    partial(desc|vuln): corr={r['partial_corr_descendants_given_vuln']} "
            f"CI=[{r['partial_ci_low']},{r['partial_ci_high']}] excl0={r['excludes_zero_partial']}"
        )


if __name__ == "__main__":
    print("Part 1: full-removal sanity check (all 6 variants, budgets 2/3/4, reduction=1.0)...")
    full_removal = full_removal_rows()

    print("\nPart 4a: predictor correlations on original 4 variants (resimulating at n_runs=1000)...")
    part4_original = original_predictor_rows()

    print("\nParts 2/3/4b: large-topology screen + top-30 CRN rerun + predictor correlations...")
    part2_rows, part4_large = large_topology_analysis()

    all_rows = full_removal + part2_rows + part4_large + part4_original
    csv_path = write_csv(all_rows)
    print(f"\nWrote {len(all_rows)} rows to {csv_path}")

    print_summary(full_removal, part2_rows, part4_large, part4_original)
