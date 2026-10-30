"""Test the hypothesis: the best simulated defense equals the minimum
vertex cut of the attack graph.

For each topology variant (src.graph.VARIANTS), computes every minimum
node cut separating the attacker's entry point(s) from physical_equipment,
excluding the entry points and physical_equipment themselves from the set
of nodes that may appear in a cut (defending the entry point itself does
nothing: the entry starts COMPROMISED at t=0, before any defense can act).
A weighted variant breaks ties among same-size minimum cuts in favor of
higher total node vulnerability (nodes more "worth defending").

For "multi_entry" (two possible entry points, chosen 50/50 per run), the
cut used for comparison is the *joint* one: the smallest node set whose
removal disconnects BOTH possible entries from the target, since the
defender must commit a combo before knowing which entry is used.

Compares each (variant, budget, reduction, metric) row of
results/topology_robustness.csv (the simulation's best defense combo)
against the predicted minimum-cut set(s), and classifies the match as:
  exact_match     - combo equals a minimum cut exactly (only possible
                     when budget == min cut size)
  superset_match   - combo fully contains a minimum cut plus extra nodes
                     (still a complete block, using surplus budget)
  partial_overlap  - combo shares some but not all nodes of every min cut
                     (does not fully block propagation)
  no_match         - combo shares no nodes with any minimum cut

Also checks, independent of the above, whether the combo actually
disconnects the entry point(s) from physical_equipment at all (is_cut),
and if not, reports what alternate path still lets the attack through.

Running this module directly writes results/mincut_vs_simulation.csv and
prints a variant x budget summary table plus the overall match counts.
"""

import csv
import itertools
import os

import networkx as nx

from src.graph import VARIANTS, build_smart_grid
from src.simulation import TARGET_NODE
from src.visualization import RESULTS_DIR

SIM_CSV = os.path.join(RESULTS_DIR, "topology_robustness.csv")
OUT_CSV = os.path.join(RESULTS_DIR, "mincut_vs_simulation.csv")


def candidate_nodes(graph, entries):
    """Nodes eligible to appear in a cut: everything except the entry
    point(s) and the target."""
    excluded = set(entries) | {TARGET_NODE}
    return [n for n in graph.nodes() if n not in excluded]


def disconnects(graph, entries, removed):
    """True if removing `removed` nodes disconnects every node in
    `entries` from TARGET_NODE."""
    sub = graph.copy()
    sub.remove_nodes_from(removed)
    for entry in entries:
        if entry not in sub:
            continue
        if nx.has_path(sub, entry, TARGET_NODE):
            return False
    return True


def min_vertex_cuts(graph, entries):
    """Brute-force search (candidate sets here are <= 6 nodes, so this is
    cheap) for the smallest r and ALL size-r subsets of candidate_nodes
    whose removal disconnects every entry in `entries` from TARGET_NODE.

    Returns (r, [set, ...]).
    """
    candidates = candidate_nodes(graph, entries)
    for r in range(0, len(candidates) + 1):
        found = [
            set(combo)
            for combo in itertools.combinations(candidates, r)
            if disconnects(graph, entries, combo)
        ]
        if found:
            return r, found
    return None, []


def weighted_pick(graph, cuts):
    """Among tied minimum cuts, the one(s) with the highest total node
    vulnerability (ties broken toward nodes "more worth defending")."""
    if not cuts:
        return []
    scored = [(sum(graph.nodes[n]["vulnerability"] for n in c), c) for c in cuts]
    best_score = max(score for score, _ in scored)
    return [c for score, c in scored if score == best_score]


def nx_cross_check(graph, entry, target):
    """Cross-check the brute-force result for a single entry against
    networkx's own min node-cut algorithm (max-flow based)."""
    try:
        cut = nx.minimum_node_cut(graph, s=entry, t=target)
        return len(cut)
    except nx.NetworkXError:
        return None


def classify_match(combo_for_cut, min_cuts):
    """Classify combo_for_cut against the list of minimum-cut sets."""
    if not min_cuts:
        return "no_match"
    if combo_for_cut in min_cuts:
        return "exact_match"
    if any(mc <= combo_for_cut for mc in min_cuts):
        return "superset_match"
    if any(combo_for_cut & mc for mc in min_cuts):
        return "partial_overlap"
    return "no_match"


def explain_not_a_cut(graph, entries, combo_for_cut):
    """For a combo that does NOT fully disconnect entries from the
    target, name a surviving path as evidence of what's different."""
    sub = graph.copy()
    sub.remove_nodes_from(combo_for_cut)
    explanations = []
    for entry in entries:
        if entry not in sub or not nx.has_path(sub, entry, TARGET_NODE):
            continue
        path = nx.shortest_path(sub, entry, TARGET_NODE)
        explanations.append(f"{entry} still reaches target via {'->'.join(path)}")
    if not explanations:
        return "combo removed an entry point itself; remaining entries (if any) already blocked"
    return "; ".join(explanations)


def analyze_variant(variant):
    graph = build_smart_grid(variant=variant)
    entries = list(graph.graph.get("entry_points", {}).keys())

    per_entry = {}
    for entry in entries:
        k, cuts = min_vertex_cuts(graph, [entry])
        nx_k = nx_cross_check(graph, entry, TARGET_NODE)
        per_entry[entry] = {"k": k, "cuts": cuts, "nx_k": nx_k}

    joint_k, joint_cuts = min_vertex_cuts(graph, entries)
    joint_weighted = weighted_pick(graph, joint_cuts)

    return {
        "graph": graph,
        "entries": entries,
        "per_entry": per_entry,
        "joint_k": joint_k,
        "joint_cuts": joint_cuts,
        "joint_weighted": joint_weighted,
    }


def load_sim_rows(path=SIM_CSV):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fmt_cuts(cuts):
    return " | ".join("{" + ",".join(sorted(c)) + "}" for c in cuts)


def run_analysis():
    variant_info = {v: analyze_variant(v) for v in VARIANTS}
    sim_rows = load_sim_rows()

    out_rows = []
    for row in sim_rows:
        variant = row["variant"]
        info = variant_info[variant]
        graph = info["graph"]
        entries = set(info["entries"])

        combo = set(row["best_combo"].split("+"))
        combo_for_cut = combo - entries  # defending an entry point is a no-op structurally

        min_cuts = info["joint_cuts"]
        weighted_cuts = info["joint_weighted"]

        is_cut = disconnects(graph, info["entries"], combo_for_cut)
        match_status = classify_match(combo_for_cut, min_cuts)
        matches_weighted = combo_for_cut in weighted_cuts
        contains_mincut_node = match_status != "no_match"
        explanation = "" if is_cut else explain_not_a_cut(graph, info["entries"], combo_for_cut)

        out_rows.append(
            {
                "variant": variant,
                "budget": row["budget"],
                "reduction": row["reduction"],
                "metric": row["metric"],
                "min_cut_size": info["joint_k"],
                "min_cuts": fmt_cuts(min_cuts),
                "weighted_pick": fmt_cuts(weighted_cuts),
                "sim_best_combo": row["best_combo"],
                "is_cut": is_cut,
                "match_status": match_status,
                "matches_weighted_pick": matches_weighted,
                "contains_mincut_node": contains_mincut_node,
                "not_a_cut_explanation": explanation,
            }
        )

    return variant_info, out_rows


def write_csv(rows, path=OUT_CSV):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fieldnames = [
        "variant", "budget", "reduction", "metric", "min_cut_size", "min_cuts",
        "weighted_pick", "sim_best_combo", "is_cut", "match_status",
        "matches_weighted_pick", "contains_mincut_node", "not_a_cut_explanation",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def print_summary(variant_info, out_rows):
    print("\nPer-entry minimum node cuts (networkx cross-checked):")
    for variant in VARIANTS:
        info = variant_info[variant]
        print(f"\n  variant={variant}  entries={info['entries']}")
        for entry, e in info["per_entry"].items():
            agree = "OK" if e["nx_k"] == e["k"] else f"MISMATCH (nx={e['nx_k']})"
            print(f"    entry={entry}: k={e['k']} cuts={fmt_cuts(e['cuts'])}  [networkx check: {agree}]")
        print(f"    JOINT (blocks every entry at once): k={info['joint_k']} cuts={fmt_cuts(info['joint_cuts'])}")
        print(f"    weighted pick (max total vulnerability among ties): {fmt_cuts(info['joint_weighted'])}")

    print("\nVariant x budget -> predicted cut vs. simulation winner(s):")
    for variant in VARIANTS:
        info = variant_info[variant]
        for budget in sorted(set(r["budget"] for r in out_rows if r["variant"] == variant)):
            cell_rows = [r for r in out_rows if r["variant"] == variant and r["budget"] == budget]
            seen = {}
            for r in cell_rows:
                key = (r["sim_best_combo"], r["match_status"], r["is_cut"])
                seen[key] = seen.get(key, 0) + 1
            print(f"\n  variant={variant:12s} budget={budget}  predicted min-cut(s) (k={info['joint_k']}): {fmt_cuts(info['joint_cuts'])}")
            for (combo, status, is_cut), count in seen.items():
                print(f"    winner={combo:55s} is_cut={is_cut!s:5s} status={status:16s} ({count}/{len(cell_rows)} reduction x metric cells)")

    n_contains = sum(1 for r in out_rows if r["contains_mincut_node"])
    n_exact = sum(1 for r in out_rows if r["match_status"] == "exact_match")
    n_superset = sum(1 for r in out_rows if r["match_status"] == "superset_match")
    n_partial = sum(1 for r in out_rows if r["match_status"] == "partial_overlap")
    n_no_match = sum(1 for r in out_rows if r["match_status"] == "no_match")
    n_not_cut = sum(1 for r in out_rows if not r["is_cut"])

    print(f"\nOverall across {len(out_rows)} cells:")
    print(f"  exact_match:      {n_exact}")
    print(f"  superset_match:   {n_superset}")
    print(f"  partial_overlap:  {n_partial}")
    print(f"  no_match:         {n_no_match}")
    print(f"  -> winner contains >=1 min-cut node in {n_contains}/{len(out_rows)} cells")
    print(f"  -> winner is NOT a full cut (attack still gets through) in {n_not_cut}/{len(out_rows)} cells")

    if n_not_cut:
        print("\nCells where the winner does NOT fully block the attack:")
        for r in out_rows:
            if not r["is_cut"]:
                print(
                    f"  {r['variant']:12s} budget={r['budget']} reduction={r['reduction']} metric={r['metric']:10s} "
                    f"winner={r['sim_best_combo']}\n    -> {r['not_a_cut_explanation']}"
                )


if __name__ == "__main__":
    variant_info, out_rows = run_analysis()
    csv_path = write_csv(out_rows)
    print(f"Wrote {len(out_rows)} rows to {csv_path}")
    print_summary(variant_info, out_rows)
