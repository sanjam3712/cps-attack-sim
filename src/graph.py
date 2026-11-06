"""Smart grid graph definition.

Defines the topology of the cyber-physical smart grid (corporate IT
through SCADA down to physical equipment) as a directed networkx
graph, along with node attributes (vulnerability, criticality) and
edge attributes (propagation_probability) used by the attack
propagation simulation.
"""

import json
import random
from pathlib import Path

import networkx as nx

VULNERABILITY_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "vulnerabilities.json"

# The physical/process asset every topology in this module attacks toward.
TARGET_NODE = "physical_equipment"

# Criticality stays hardcoded; vulnerability is loaded from VULNERABILITY_DATA_PATH.
CRITICALITY = {
    "corporate_network": 0.2,
    "historian_server": 0.4,
    "engineering_workstation": 0.5,
    "scada_server": 0.8,
    "remote_terminal_unit": 0.7,
    "intelligent_electronic_device": 0.8,
    "physical_equipment": 1.0,
}

# Fallback used only for nodes with no CVE-derived score (see data/vulnerabilities.json).
FALLBACK_VULNERABILITY = {
    "physical_equipment": 0.2,
}


def load_node_vulnerabilities(path=VULNERABILITY_DATA_PATH):
    """Load per-node vulnerability scores derived from real CVE data.

    Falls back to FALLBACK_VULNERABILITY for any node whose entry has no
    derived_vulnerability (e.g. physical_equipment, which has no CVEs).
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    vulnerabilities = {}
    for node, entry in data["nodes"].items():
        value = entry.get("derived_vulnerability")
        vulnerabilities[node] = value if value is not None else FALLBACK_VULNERABILITY[node]

    return vulnerabilities


NODES = {
    node: {"vulnerability": vulnerability, "criticality": CRITICALITY[node]}
    for node, vulnerability in load_node_vulnerabilities().items()
}

EDGES = [
    ("corporate_network", "historian_server"),
    ("corporate_network", "engineering_workstation"),
    ("historian_server", "scada_server"),
    ("engineering_workstation", "scada_server"),
    ("scada_server", "remote_terminal_unit"),
    ("scada_server", "intelligent_electronic_device"),
    ("remote_terminal_unit", "physical_equipment"),
    ("intelligent_electronic_device", "physical_equipment"),
]


VARIANTS = ("baseline", "redundant", "flat", "multi_entry", "large_layered", "large_random")

DEFAULT_ENTRY_POINTS = {"corporate_network": 1.0}

# ~15-node synthetic topologies (IT -> DMZ -> control -> field -> target),
# used to test whether the min-cut/simulation agreement seen on the small
# hand-drawn graphs survives a graph large enough that most budget-3
# combinations are NOT vertex cuts (see src/mincut_null_and_large.py).
LARGE_LAYER_SIZES = (4, 3, 4, 3)
LARGE_LAYER_PREFIXES = ("it", "dmz", "ctrl", "field")
LARGE_LAYER_CRITICALITY = {"it": 0.2, "dmz": 0.35, "ctrl": 0.75, "field": 0.8}
LARGE_LAYERED_SEED = 123
LARGE_RANDOM_SEED = 7


def _sample_vulnerabilities(rng, n):
    """Draw n vulnerability values (with replacement) from the real,
    CVE-derived per-node values in data/vulnerabilities.json (excluding
    physical_equipment, which has no CVE-derived value)."""
    pool = [v for node, v in load_node_vulnerabilities().items() if node != "physical_equipment"]
    return rng.choices(pool, k=n)


def _generate_layered_edges(rng, layers, fanout_range):
    """Connect layer i to layer i+1: the entry (layers[0][0]) fans out to
    the rest of its own layer (lateral movement within the IT zone), then
    each node in layer i links to a random fanout_range-sized sample of
    layer i+1, topped up so every layer i+1 node has >=1 incoming edge
    (no orphaned nodes). Returns (edges, entry_node_name)."""
    edges = []
    entry = layers[0][0]
    for other in layers[0][1:]:
        edges.append((entry, other))

    for i in range(len(layers) - 1):
        upper, lower = layers[i], layers[i + 1]
        incoming = {node: 0 for node in lower}
        for u in upper:
            k = min(rng.randint(*fanout_range), len(lower))
            for t in rng.sample(lower, k):
                edges.append((u, t))
                incoming[t] += 1
        for node, count in incoming.items():
            if count == 0:
                edges.append((rng.choice(upper), node))

    return edges, entry


def _build_large_topology(seed, fanout_range, bypass_layer_pairs=()):
    """Build a layered ~15-node topology (node_attrs, edges, entry).

    bypass_layer_pairs adds extra edges that skip directly from layers[i]
    to layers[j] (i<j, non-adjacent), weakening any single-layer
    chokepoint - used by "large_random" to make the graph messier than
    the purely layer-by-layer "large_layered" variant.
    """
    rng = random.Random(seed)
    layers = [
        [f"{prefix}_{i + 1}" for i in range(size)]
        for prefix, size in zip(LARGE_LAYER_PREFIXES, LARGE_LAYER_SIZES)
    ]
    edges, entry = _generate_layered_edges(rng, layers, fanout_range)

    for i, j in bypass_layer_pairs:
        for _ in range(rng.randint(1, 3)):
            edges.append((rng.choice(layers[i]), rng.choice(layers[j])))

    field_layer = layers[-1]
    for node in field_layer:
        edges.append((node, "physical_equipment"))
    edges = list(dict.fromkeys(edges))

    all_nodes = [node for layer in layers for node in layer]
    vulnerabilities = _sample_vulnerabilities(rng, len(all_nodes))

    node_attrs = {}
    for layer, prefix in zip(layers, LARGE_LAYER_PREFIXES):
        criticality = LARGE_LAYER_CRITICALITY[prefix]
        for node in layer:
            node_attrs[node] = {
                "vulnerability": vulnerabilities[all_nodes.index(node)],
                "criticality": criticality,
            }
    node_attrs["physical_equipment"] = dict(NODES["physical_equipment"])

    return node_attrs, edges, entry


def build_smart_grid(variant="baseline"):
    """Build and return the directed smart grid attack graph.

    variant selects the topology, to test whether findings (e.g. the
    SCADA chokepoint) are artifacts of the hand-drawn baseline:
      "baseline" (default): the original topology.
      "redundant": adds a second SCADA server (scada_backup), fed by
        historian_server and engineering_workstation and feeding RTU and
        IED - a redundant control path alongside scada_server, copying
        its vulnerability/criticality.
      "flat": baseline plus direct engineering_workstation -> RTU and
        engineering_workstation -> IED edges, modeling a common real
        misconfiguration that bypasses SCADA.
      "multi_entry": baseline topology, but the attacker's entry point is
        corporate_network or engineering_workstation (a remote vendor
        VPN), chosen 50/50 per simulation run instead of always
        corporate_network.
      "large_layered": a synthetic ~15-node topology (IT -> DMZ ->
        control -> field -> physical_equipment, 3-4 nodes per layer,
        single entry node) with random inter-layer edges (fixed seed) and
        vulnerabilities sampled from the real per-node CVE distributions.
      "large_random": like "large_layered", but with a wider per-node
        fanout and extra edges that skip directly from one layer to a
        non-adjacent one, weakening any single-layer chokepoint.

    Each node carries `vulnerability` and `criticality` attributes. Each
    edge carries a `propagation_probability` attribute, set to the
    vulnerability of the downstream (target) node, reflecting how easily
    an attacker can compromise that target once it is reachable.

    graph.graph["variant"] records which variant was built, and
    graph.graph["entry_points"] maps possible attacker start nodes to
    their selection probability (consumed by
    src.simulation.run_attack_simulation).
    """
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant '{variant}'; expected one of {VARIANTS}")

    graph = nx.DiGraph()

    if variant in ("large_layered", "large_random"):
        if variant == "large_layered":
            node_attrs, edges, entry = _build_large_topology(LARGE_LAYERED_SEED, fanout_range=(1, 2))
        else:
            node_attrs, edges, entry = _build_large_topology(
                LARGE_RANDOM_SEED, fanout_range=(1, 3), bypass_layer_pairs=[(0, 2), (1, 3)],
            )

        for node, attrs in node_attrs.items():
            graph.add_node(node, **attrs)
        for source, target in edges:
            propagation_probability = graph.nodes[target]["vulnerability"]
            graph.add_edge(source, target, propagation_probability=propagation_probability)

        graph.graph["variant"] = variant
        graph.graph["entry_points"] = {entry: 1.0}
        graph.graph["target"] = TARGET_NODE
        return graph

    node_attrs = dict(NODES)
    if variant == "redundant":
        node_attrs["scada_backup"] = dict(NODES["scada_server"])

    for node, attrs in node_attrs.items():
        graph.add_node(node, **attrs)

    edges = list(EDGES)
    if variant == "redundant":
        edges += [
            ("historian_server", "scada_backup"),
            ("engineering_workstation", "scada_backup"),
            ("scada_backup", "remote_terminal_unit"),
            ("scada_backup", "intelligent_electronic_device"),
        ]
    elif variant == "flat":
        edges += [
            ("engineering_workstation", "remote_terminal_unit"),
            ("engineering_workstation", "intelligent_electronic_device"),
        ]

    for source, target in edges:
        propagation_probability = graph.nodes[target]["vulnerability"]
        graph.add_edge(source, target, propagation_probability=propagation_probability)

    graph.graph["variant"] = variant
    if variant == "multi_entry":
        graph.graph["entry_points"] = {
            "corporate_network": 0.5,
            "engineering_workstation": 0.5,
        }
    else:
        graph.graph["entry_points"] = dict(DEFAULT_ENTRY_POINTS)

    graph.graph["target"] = TARGET_NODE
    return graph


def validate_system_config(config):
    """Validate a system config dict (see load_system for the format).

    Raises ValueError with a human-readable message on the first problem
    found. Returns None on success.
    """
    if not isinstance(config, dict):
        raise ValueError("Config must be a JSON object")

    missing = [k for k in ("nodes", "edges", "entry_points", "target") if k not in config]
    if missing:
        raise ValueError(f"Config is missing required key(s): {', '.join(missing)}")

    nodes = config["nodes"]
    if not isinstance(nodes, dict) or not nodes:
        raise ValueError("'nodes' must be a non-empty object of {name: {vulnerability, criticality}}")

    for name, attrs in nodes.items():
        if not isinstance(attrs, dict):
            raise ValueError(f"Node '{name}' must be an object with 'vulnerability' and 'criticality'")
        for field in ("vulnerability", "criticality"):
            if field not in attrs:
                raise ValueError(f"Node '{name}' is missing '{field}'")
            value = attrs[field]
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not (0 <= value <= 1):
                raise ValueError(f"Node '{name}' has {field}={value!r}; expected a number in [0, 1]")

    edges = config["edges"]
    if not isinstance(edges, list) or not edges:
        raise ValueError("'edges' must be a non-empty list of [source, target] pairs")
    for edge in edges:
        if not isinstance(edge, (list, tuple)) or len(edge) != 2:
            raise ValueError(f"Edge {edge!r} must be a [source, target] pair")
        source, target = edge
        for node in (source, target):
            if node not in nodes:
                raise ValueError(f"Edge {edge!r} references unknown node '{node}'")

    entry_points = config["entry_points"]
    if not isinstance(entry_points, dict) or not entry_points:
        raise ValueError("'entry_points' must be a non-empty object of {node_name: probability}")
    for node, prob in entry_points.items():
        if node not in nodes:
            raise ValueError(f"entry_points references unknown node '{node}'")
        if not isinstance(prob, (int, float)) or isinstance(prob, bool) or not (0 <= prob <= 1):
            raise ValueError(f"entry_points['{node}']={prob!r}; expected a probability in [0, 1]")
    total_prob = sum(entry_points.values())
    if abs(total_prob - 1.0) > 1e-6:
        raise ValueError(f"entry_points probabilities must sum to 1.0 (got {total_prob})")

    target = config["target"]
    if target not in nodes:
        raise ValueError(f"target '{target}' is not in 'nodes'")


def build_graph_from_config(config, system_name=None):
    """Build and return a directed attack graph from a parsed config dict.

    See load_system for the config format. Raises ValueError (via
    validate_system_config) with a clear message if the config is
    invalid. graph.graph["meta"] is set to config.get("_meta", {}), so
    callers can check e.g. for an "ILLUSTRATIVE" marker.
    """
    validate_system_config(config)

    graph = nx.DiGraph()
    for node, attrs in config["nodes"].items():
        graph.add_node(
            node,
            vulnerability=attrs["vulnerability"],
            criticality=attrs["criticality"],
        )

    for source, target in config["edges"]:
        propagation_probability = graph.nodes[target]["vulnerability"]
        graph.add_edge(source, target, propagation_probability=propagation_probability)

    graph.graph["system"] = system_name or "config"
    graph.graph["entry_points"] = dict(config["entry_points"])
    graph.graph["target"] = config["target"]
    graph.graph["meta"] = config.get("_meta", {})

    return graph


def load_system(path):
    """Build and return a directed attack graph from a system config file.

    The config is a JSON object with:
      "nodes": {node_name: {"vulnerability": float, "criticality": float}},
      "edges": [[source, target], ...],
      "entry_points": {node_name: probability},
      "target": node_name (the node whose failure the simulation tracks).

    Each edge's `propagation_probability` is set to the vulnerability of
    its downstream (target) node, same convention as build_smart_grid.
    Sets graph.graph["target"] and graph.graph["entry_points"] from the
    config, and graph.graph["system"] to the config file's stem.
    """
    with open(path, encoding="utf-8") as f:
        config = json.load(f)

    return build_graph_from_config(config, system_name=Path(path).stem)


def print_graph_info(graph):
    """Print all nodes with their attributes and all edges with theirs."""
    print("Nodes:")
    for node, attrs in graph.nodes(data=True):
        print(f"  {node}: {attrs}")

    print("\nEdges:")
    for source, target, attrs in graph.edges(data=True):
        print(f"  {source} -> {target}: {attrs}")


if __name__ == "__main__":
    smart_grid = build_smart_grid()
    print_graph_info(smart_grid)
