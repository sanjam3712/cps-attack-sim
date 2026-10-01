"""Smart grid graph definition.

Defines the topology of the cyber-physical smart grid (corporate IT
through SCADA down to physical equipment) as a directed networkx
graph, along with node attributes (vulnerability, criticality) and
edge attributes (propagation_probability) used by the attack
propagation simulation.
"""

import networkx as nx

NODES = {
    "corporate_network": {"vulnerability": 0.9, "criticality": 0.2},
    "historian_server": {"vulnerability": 0.7, "criticality": 0.4},
    "engineering_workstation": {"vulnerability": 0.8, "criticality": 0.5},
    "scada_server": {"vulnerability": 0.6, "criticality": 0.8},
    "remote_terminal_unit": {"vulnerability": 0.5, "criticality": 0.7},
    "intelligent_electronic_device": {"vulnerability": 0.4, "criticality": 0.8},
    "physical_equipment": {"vulnerability": 0.2, "criticality": 1.0},
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
