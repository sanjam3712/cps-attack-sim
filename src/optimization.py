"""Defense resource optimization.

Searches over combinations of nodes to defend (within a fixed budget)
and picks the combination that performs best on a chosen metric,
evaluated via Monte Carlo simulation.

Combinations are compared using common random numbers (the same RNG
seed drives every combination's Monte Carlo run) and failure
probabilities are reported with a 95% Wilson confidence interval, so
the ranking's statistical reliability can be assessed instead of
trusting a bare point estimate.

A single P(fail by t_max) point estimate can saturate (every combo
looks about equally "bad" at a long horizon) even when combos differ a
lot in how much time they buy before failure. To avoid picking a
"best" combo off a saturated metric, `optimize_defense` accepts a
`metric` parameter to rank by P(fail by 6h), P(fail by 24h), or mean
time to failure (maximized) instead of always using the longest
horizon.
"""

import copy
import itertools

from src.ctmc import build_ctmc_nodes
from src.graph import build_smart_grid
from src.simulation import TARGET_NODE, run_monte_carlo
from src.stats import intervals_overlap, wilson_confidence_interval

DEFAULT_SEED = 42
DEFAULT_N_RUNS = 5000
DEFAULT_REDUCTION = 0.5

METRICS = ("p_fail_6h", "p_fail_24h", "mean_ttf")


def apply_defense(ctmc_nodes, defended_nodes, reduction=DEFAULT_REDUCTION):
    """Return a copy of ctmc_nodes with compromise_rate reduced for defended_nodes.

    Each defended node's compromise_rate is scaled by (1 - reduction);
    its Q matrix is rebuilt to reflect the new rate. Nodes not in
    defended_nodes are left untouched.
    """
    defended = copy.deepcopy(ctmc_nodes)

    for name in defended_nodes:
        node = defended[name]
        node.compromise_rate *= (1 - reduction)
        node.Q = node._build_q_matrix()

    return defended


def evaluate_combinations(
    graph,
    ctmc_nodes,
    budget=2,
    t_max=24,
    n_runs=DEFAULT_N_RUNS,
    seed=DEFAULT_SEED,
    excluded_nodes=None,
    reduction=DEFAULT_REDUCTION,
):
    """Run the Monte Carlo simulation for every combination of up to
    `budget` defended nodes and return their full results, independent
    of any ranking metric.

    excluded_nodes: node names that are not allowed to be defended
    (e.g. pass [TARGET_NODE] to exclude physical_equipment, since
    "defend the target itself" is a trivial result, not a propagation
    defense). Defaults to no exclusions for backward compatibility.

    reduction: fraction by which a defended node's compromise_rate is
    cut (passed to apply_defense).

    Every combination's Monte Carlo run uses the same `seed` (common
    random numbers), so combinations are compared against the same
    underlying random draws rather than independent sampling noise.

    Returns a list of dicts, one per combination, each with 'combo',
    'probability_of_failure' and 'ci_low'/'ci_high' (P(fail by t_max)
    and its 95% Wilson CI), 'mean_time_to_failure', 'p_fail_by_t' (per-
    horizon probabilities with CIs, from run_monte_carlo), 'n_runs',
    'seed', and 'reduction'. Not sorted by any particular metric.
    """
    excluded_nodes = set(excluded_nodes or ())
    node_names = [n for n in graph.nodes() if n not in excluded_nodes]

    detailed = []
    for combo in itertools.combinations(node_names, budget):
        defended_ctmc_nodes = apply_defense(ctmc_nodes, combo, reduction=reduction)
        mc_results = run_monte_carlo(
            graph, defended_ctmc_nodes, n_runs=n_runs, t_max=t_max, seed=seed
        )
        p = mc_results["probability_of_failure"]
        ci_low, ci_high = wilson_confidence_interval(p, n_runs)
        detailed.append(
            {
                "combo": combo,
                "probability_of_failure": p,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "mean_time_to_failure": mc_results["mean_time_to_failure"],
                "p_fail_by_t": mc_results["p_fail_by_t"],
                "n_runs": n_runs,
                "seed": seed,
                "reduction": reduction,
            }
        )

    return detailed


def _metric_value_and_ci(entry, metric, t_max):
    """Return (value, (ci_low, ci_high) or None) for `metric` on one
    evaluate_combinations() entry. ci is None for mean_ttf (no CI is
    computed for a mean, only for the Wilson proportion CIs)."""
    if metric == "p_fail_24h":
        pt = entry["p_fail_by_t"].get(24) or entry["p_fail_by_t"].get(t_max)
        if pt is None:
            raise ValueError("p_fail_24h requires t_max >= 24 (or a 24h threshold)")
        return pt["p"], (pt["ci_low"], pt["ci_high"])
    elif metric == "p_fail_6h":
        pt = entry["p_fail_by_t"].get(6)
        if pt is None:
            raise ValueError("p_fail_6h requires t_max >= 6")
        return pt["p"], (pt["ci_low"], pt["ci_high"])
    elif metric == "mean_ttf":
        mttf = entry["mean_time_to_failure"]
        # No failures observed in n_runs runs: the true mean TTF is
        # unknown but at least t_max, so treat it as t_max for ranking
        # (a conservative lower bound) rather than None/inf.
        value = mttf if mttf is not None else t_max
        return value, None
    else:
        raise ValueError(f"Unknown metric '{metric}'; expected one of {METRICS}")


def rank_by_metric(detailed, metric, t_max=24):
    """Return a new list of entries (shallow copies of `detailed`) sorted
    best-first by `metric`, each annotated with 'metric', 'metric_value',
    and 'metric_ci_low'/'metric_ci_high' (None for mean_ttf).

    p_fail_* metrics are minimized (lower failure probability is
    better); mean_ttf is maximized (surviving longer is better).
    """
    ranked = []
    for entry in detailed:
        value, ci = _metric_value_and_ci(entry, metric, t_max)
        annotated = dict(entry)
        annotated["metric"] = metric
        annotated["metric_value"] = value
        annotated["metric_ci_low"], annotated["metric_ci_high"] = ci if ci else (None, None)
        ranked.append(annotated)

    maximize = metric == "mean_ttf"
    ranked.sort(key=lambda e: e["metric_value"], reverse=maximize)
    return ranked


def optimize_defense(
    graph,
    ctmc_nodes,
    budget=2,
    t_max=24,
    n_runs=DEFAULT_N_RUNS,
    seed=DEFAULT_SEED,
    excluded_nodes=None,
    metric="p_fail_24h",
    reduction=DEFAULT_REDUCTION,
):
    """Find the combination of up to `budget` defended nodes that
    performs best on `metric` ("p_fail_24h", "p_fail_6h", or "mean_ttf",
    the last one maximized).

    See evaluate_combinations for excluded_nodes/reduction/CRN details.

    Returns a dict with 'best_nodes' (tuple of defended node names),
    'best_probability' (its P(fail by t_max), kept for backward
    compatibility regardless of `metric`), 'best_metric_value' (its
    value under the selected metric), 'metric', 'results' (a list of
    (node_combination, probability_of_failure) tuples, kept for
    backward compatibility with existing callers - NOT reordered by
    `metric`, always sorted by P(fail by t_max)), and 'results_detailed'
    (the full per-combo records, ranked best-first by `metric`).
    """
    detailed = evaluate_combinations(
        graph, ctmc_nodes, budget=budget, t_max=t_max, n_runs=n_runs,
        seed=seed, excluded_nodes=excluded_nodes, reduction=reduction,
    )
    ranked = rank_by_metric(detailed, metric, t_max=t_max)
    best = ranked[0]

    results_by_p_fail = sorted(detailed, key=lambda d: d["probability_of_failure"])

    return {
        "best_nodes": best["combo"],
        "best_probability": best["probability_of_failure"],
        "best_metric_value": best["metric_value"],
        "metric": metric,
        "results": [(d["combo"], d["probability_of_failure"]) for d in results_by_p_fail],
        "results_detailed": ranked,
    }


_METRIC_LABELS = {
    "p_fail_24h": "P(fail by 24h)",
    "p_fail_6h": "P(fail by 6h)",
    "mean_ttf": "mean_ttf",
}


def print_ranking(results_detailed, title, top_n=5):
    """Print the top `top_n` combinations with their metric value (and
    95% CI, for probability metrics), and state whether the best
    combo's CI overlaps the 2nd/3rd best - i.e. whether the ranking is
    statistically distinguishable. For mean_ttf (no CI available), the
    overlap check is skipped."""
    print(f"\n{title}")
    top = results_detailed[:top_n]
    if not top:
        print("  (no combinations)")
        return

    metric = top[0].get("metric", "p_fail_24h")
    label = _METRIC_LABELS.get(metric, metric)
    has_ci = metric != "mean_ttf"

    for i, r in enumerate(top, 1):
        if has_ci:
            print(
                f"  #{i}: {r['combo']} -> {label}={r['metric_value']:.4f}  "
                f"95% CI=[{r['metric_ci_low']:.4f}, {r['metric_ci_high']:.4f}]  (n={r['n_runs']})"
            )
        else:
            print(
                f"  #{i}: {r['combo']} -> {label}={r['metric_value']:.4f}h  "
                f"(no CI; maximized)  [P(fail by 24h)={r['probability_of_failure']:.4f}]"
            )

    if not has_ci:
        print("  CI overlap check skipped - no CI is computed for mean_ttf.")
        return

    if len(top) >= 2:
        best_ci = (top[0]["metric_ci_low"], top[0]["metric_ci_high"])
        second_ci = (top[1]["metric_ci_low"], top[1]["metric_ci_high"])
        overlaps = intervals_overlap(best_ci, second_ci)
        verdict = "OVERLAPS - ranking vs #2 is NOT statistically distinguishable" if overlaps else "does not overlap - #1 is distinguishable from #2"
        print(f"  Best vs #2 CI: {verdict}")
    if len(top) >= 3:
        best_ci = (top[0]["metric_ci_low"], top[0]["metric_ci_high"])
        third_ci = (top[2]["metric_ci_low"], top[2]["metric_ci_high"])
        overlaps = intervals_overlap(best_ci, third_ci)
        verdict = "OVERLAPS - ranking vs #3 is NOT statistically distinguishable" if overlaps else "does not overlap - #1 is distinguishable from #3"
        print(f"  Best vs #3 CI: {verdict}")


if __name__ == "__main__":
    smart_grid = build_smart_grid()
    ctmc_nodes = build_ctmc_nodes(smart_grid)

    baseline = run_monte_carlo(
        smart_grid, ctmc_nodes, n_runs=DEFAULT_N_RUNS, t_max=24, seed=DEFAULT_SEED
    )
    baseline_probability = baseline["probability_of_failure"]
    print(
        f"Baseline P({TARGET_NODE} fails), no defense = {baseline_probability:.4f} "
        f"(n_runs={DEFAULT_N_RUNS}, seed={DEFAULT_SEED})"
    )
    for threshold, stats in baseline["p_fail_by_t"].items():
        print(f"  P(fail by {threshold}h) = {stats['p']:.4f}  95% CI=[{stats['ci_low']:.4f}, {stats['ci_high']:.4f}]")

    # Budget=2, physical_equipment DEFENDABLE (naive ranking - includes the
    # trivial "defend the target itself" result).
    result_incl = optimize_defense(
        smart_grid, ctmc_nodes, budget=2, t_max=24,
        n_runs=DEFAULT_N_RUNS, seed=DEFAULT_SEED, excluded_nodes=None,
    )
    print_ranking(
        result_incl["results_detailed"],
        f"Budget=2, {TARGET_NODE} DEFENDABLE (naive ranking, includes trivial target-defense):",
    )

    # Budget=2, physical_equipment EXCLUDED (recommended - only propagation
    # defenses are compared).
    result_excl = optimize_defense(
        smart_grid, ctmc_nodes, budget=2, t_max=24,
        n_runs=DEFAULT_N_RUNS, seed=DEFAULT_SEED, excluded_nodes=[TARGET_NODE],
    )
    print_ranking(
        result_excl["results_detailed"],
        f"Budget=2, {TARGET_NODE} EXCLUDED (recommended):",
    )

    # Budget=3, physical_equipment EXCLUDED.
    result_b3 = optimize_defense(
        smart_grid, ctmc_nodes, budget=3, t_max=24,
        n_runs=DEFAULT_N_RUNS, seed=DEFAULT_SEED, excluded_nodes=[TARGET_NODE],
    )
    print_ranking(
        result_b3["results_detailed"],
        f"Budget=3, {TARGET_NODE} EXCLUDED:",
    )

    for label, result in (
        (f"Budget=2, {TARGET_NODE} DEFENDABLE", result_incl),
        (f"Budget=2, {TARGET_NODE} EXCLUDED", result_excl),
        (f"Budget=3, {TARGET_NODE} EXCLUDED", result_b3),
    ):
        best_probability = result["best_probability"]
        improvement_pct = (
            (baseline_probability - best_probability) / baseline_probability * 100
            if baseline_probability > 0
            else 0.0
        )
        print(
            f"\n{label}: best={result['best_nodes']}, "
            f"P(fail)={best_probability:.4f}, improvement={improvement_pct:.2f}%"
        )
