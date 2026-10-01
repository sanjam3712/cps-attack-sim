"""Continuous time Markov chain (CTMC) model.

Defines the CTMC representing each node's security state (SECURE,
COMPROMISED, FAILED) and the 3x3 transition rate (Q) matrix governing
attack and recovery dynamics for that node.
"""

import numpy as np
from scipy.linalg import expm

from src.graph import build_smart_grid

SECURE, COMPROMISED, FAILED = 0, 1, 2
STATE_NAMES = ["SECURE", "COMPROMISED", "FAILED"]


class CTMCNode:
    """CTMC model of a single node's security state over time.

    States: SECURE -> COMPROMISED -> FAILED, with a recovery
    transition from COMPROMISED back to SECURE.
    """

    def __init__(self, name, vulnerability, criticality, recovery_rate=0.1):
        self.name = name
        self.vulnerability = vulnerability
        self.criticality = criticality

        self.compromise_rate = vulnerability
        self.failure_rate = criticality
        self.recovery_rate = recovery_rate

        self.Q = self._build_q_matrix()

    def _build_q_matrix(self):
        """Build the 3x3 transition rate matrix for this node."""
        Q = np.zeros((3, 3))

        Q[SECURE][COMPROMISED] = self.compromise_rate
        Q[COMPROMISED][FAILED] = self.failure_rate
        Q[COMPROMISED][SECURE] = self.recovery_rate

        for i in range(3):
            Q[i][i] = -np.sum(Q[i])

        return Q

    def get_state_probabilities(self, t):
        """Return [P(SECURE), P(COMPROMISED), P(FAILED)] at time t.

        Assumes the node starts in the SECURE state at t=0.
        """
        initial_state = np.array([1.0, 0.0, 0.0])
        transition_matrix = expm(self.Q * t)
        return initial_state @ transition_matrix


def build_ctmc_nodes(graph):
    """Build a CTMCNode for every node in the smart grid graph."""
    nodes = {}
    for name, attrs in graph.nodes(data=True):
        nodes[name] = CTMCNode(
            name,
            vulnerability=attrs["vulnerability"],
            criticality=attrs["criticality"],
        )
    return nodes


if __name__ == "__main__":
    smart_grid = build_smart_grid()
    ctmc_nodes = build_ctmc_nodes(smart_grid)

    scada_node = ctmc_nodes["scada_server"]
    print(f"State probabilities for '{scada_node.name}' "
          f"(compromise_rate={scada_node.compromise_rate}, "
          f"failure_rate={scada_node.failure_rate}, "
          f"recovery_rate={scada_node.recovery_rate}):\n")

    for t in [1, 5, 10, 24]:
        probs = scada_node.get_state_probabilities(t)
        formatted = ", ".join(
            f"{name}={p:.4f}" for name, p in zip(STATE_NAMES, probs)
        )
        print(f"  t={t:>2}h -> {formatted}")
