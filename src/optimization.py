"""Defense resource optimization.

Searches over combinations of nodes to defend (within a fixed budget)
and picks the combination that minimizes the probability of
physical_equipment reaching FAILED, evaluated via Monte Carlo
simulation.

Combinations are compared using common random numbers (the same RNG
seed drives every combination's Monte Carlo run) and failure
probabilities are reported with a 95% Wilson confidence interval, so
the ranking's statistical reliability can be assessed instead of
trusting a bare point estimate.
"""

import copy
import itertools
import math

from src.ctmc import build_ctmc_nodes
from src.graph import build_smart_grid
from src.simulation import TARGET_NODE, run_monte_carlo

DEFAULT_SEED = 42
DEFAULT_N_RUNS = 5000
Z_95 = 1.959963984540054  # two-sided 95% normal quantile


def apply_defense(ctmc_nodes, defended_nodes, reduction=0.5):
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


def wilson_confidence_interval(p_hat, n, z=Z_95):
    """Return (low, high), the 95% Wilson score interval for a Bernoulli
    proportion p_hat estimated from n trials. More reliable than the
    normal (Wald) approximation when p_hat is close to 0 or 1, or n is
    small - both of which happen here."""
    if n <= 0:
        return (0.0, 1.0)
    denom = 1 + z ** 2 / n
    center = (p_hat + z ** 2 / (2 * n)) / denom
    half_width = (z / denom) * math.sqrt(p_hat * (1 - p_hat) / n + z ** 2 / (4 * n ** 2))
    return (max(0.0, center - half_width), min(1.0, center + half_width))


def intervals_overlap(a, b):
    """Return True if closed intervals a=(low, high) and b=(low, high) overlap."""
    return a[0] <= b[1] and b[0] <= a[1]


def optimize_defense(
    graph,
    ctmc_nodes,
    budget=2,
    t_max=24,
    n_runs=DEFAULT_N_RUNS,
    seed=DEFAULT_SEED,
    excluded_nodes=None,
):
    """Find the combination of up to `budget` defended nodes that
    minimizes the probability of physical_equipment failure.

    Every combination's Monte Carlo run uses the same `seed` (common
    random numbers), so combinations are compared against the same
    underlying random draws rather than independent sampling noise.

    excluded_nodes: node names that are not allowed to be defended
    (e.g. pass [TARGET_NODE] to exclude physical_equipment, since
    "defend the target itself" is a trivial result, not a propagation
    defense). Defaults to no exclusions for backward compatibility.

    Returns a dict with 'best_nodes' (tuple of defended node names),
    'best_probability' (its failure probability), 'results' (a list of
    (node_combination, failure_probability) tuples, kept for backward
    compatibility with existing callers), and 'results_detailed' (the
    same combinations as a list of dicts sorted best-first, each with
    'combo', 'probability_of_failure', 'ci_low', 'ci_high', 'n_runs',
    'seed').
    """
    excluded_nodes = set(excluded_nodes or ())
    node_names = [n for n in graph.nodes() if n not in excluded_nodes]

    detailed = []
    for combo in itertools.combinations(node_names, budget):
        defended_ctmc_nodes = apply_defense(ctmc_nodes, combo)
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
                "n_runs": n_runs,
                "seed": seed,
            }
        )

    detailed.sort(key=lambda item: item["probability_of_failure"])
    best = detailed[0]

    return {
        "best_nodes": best["combo"],
        "best_probability": best["probability_of_failure"],
        "results": [(d["combo"], d["probability_of_failure"]) for d in detailed],
        "results_detailed": detailed,
    }


def print_ranking(results_detailed, title, top_n=5):
    """Print the top `top_n` combinations with their 95% CIs, and state
    whether the best combo's CI overlaps the 2nd/3rd best - i.e. whether
    the ranking is statistically distinguishable."""
    print(f"\n{title}")
    top = results_detailed[:top_n]
    for i, r in enumerate(top, 1):
        print(
            f"  #{i}: {r['combo']} -> P(fail)={r['probability_of_failure']:.4f}  "
            f"95% CI=[{r['ci_low']:.4f}, {r['ci_high']:.4f}]  (n={r['n_runs']})"
        )

    if len(top) >= 2:
        best_ci = (top[0]["ci_low"], top[0]["ci_high"])
        second_ci = (top[1]["ci_low"], top[1]["ci_high"])
        overlaps = intervals_overlap(best_ci, second_ci)
        verdict = "OVERLAPS - ranking vs #2 is NOT statistically distinguishable" if overlaps else "does not overlap - #1 is distinguishable from #2"
        print(f"  Best vs #2 CI: {verdict}")
    if len(top) >= 3:
        best_ci = (top[0]["ci_low"], top[0]["ci_high"])
        third_ci = (top[2]["ci_low"], top[2]["ci_high"])
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
