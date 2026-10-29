"""Smart grid graph definition.

Defines the topology of the cyber-physical smart grid (corporate IT
through SCADA down to physical equipment) as a directed networkx
graph, along with node attributes (vulnerability, criticality) and
edge attributes (propagation_probability) used by the attack
propagation simulation.
"""

import json
from pathlib import Path

import networkx as nx

VULNERABILITY_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "vulnerabilities.json"

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


def build_smart_grid():
    """Build and return the directed smart grid attack graph.

    Each node carries `vulnerability` and `criticality` attributes.
    Each edge carries a `propagation_probability` attribute, set to
    the vulnerability of the downstream (target) node, reflecting how
    easily an attacker can compromise that target once it is reachable.
    """
    graph = nx.DiGraph()

    for node, attrs in NODES.items():
        graph.add_node(node, **attrs)

    for source, target in EDGES:
        propagation_probability = graph.nodes[target]["vulnerability"]
        graph.add_edge(source, target, propagation_probability=propagation_probability)

    return graph


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
