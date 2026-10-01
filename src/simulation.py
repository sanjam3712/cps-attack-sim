"""Attack propagation simulation.

Runs stochastic simulations of attack propagation across the smart
grid graph. A node becomes COMPROMISED when an upstream neighbor is
already COMPROMISED (or FAILED) and its propagation_probability
"fires"; once COMPROMISED, the node's own CTMC (failure_rate,
recovery_rate) governs its further evolution to FAILED or back to
SECURE.
"""

import numpy as np

from src.ctmc import SECURE, COMPROMISED, FAILED, build_ctmc_nodes
from src.graph import build_smart_grid

ENTRY_POINT = "corporate_network"
TARGET_NODE = "physical_equipment"


def run_attack_simulation(graph, ctmc_nodes, t_max=24, t_steps=100, rng=None):
    """Simulate attack propagation across the graph from t=0 to t_max.

    Returns a dict with 'time', 'node_states' (per-node state arrays,
    0=SECURE, 1=COMPROMISED, 2=FAILED), and
    'physical_equipment_failure_time' (first time physical_equipment
    reaches FAILED, or None if it never does).
    """
    if rng is None:
        rng = np.random.default_rng()

    dt = t_max / t_steps
    time = np.linspace(0, t_max, t_steps + 1)
    node_names = list(graph.nodes())

    current_states = {name: SECURE for name in node_names}
    current_states[ENTRY_POINT] = COMPROMISED

    node_states = {name: np.zeros(t_steps + 1, dtype=int) for name in node_names}
    for name in node_names:
        node_states[name][0] = current_states[name]

    physical_equipment_failure_time = (
        0.0 if current_states[TARGET_NODE] == FAILED else None
    )

    for step in range(1, t_steps + 1):
        t = time[step]
        new_states = dict(current_states)

        for name in node_names:
            state = current_states[name]

            if state == SECURE:
                compromised_preds = [
                    pred
                    for pred in graph.predecessors(name)
                    if current_states[pred] in (COMPROMISED, FAILED)
                ]
                if compromised_preds:
                    # Susceptibility is the node's own (possibly defended)
                    # compromise_rate, gated on having a compromised
                    # upstream neighbor to propagate the attack.
                    p_spread = min(ctmc_nodes[name].compromise_rate * dt, 1.0)
                    if rng.random() < p_spread:
                        new_states[name] = COMPROMISED

            elif state == COMPROMISED:
                ctmc_node = ctmc_nodes[name]
                p_fail = min(ctmc_node.failure_rate * dt, 1.0)
                p_recover = min(ctmc_node.recovery_rate * dt, 1.0)
                r = rng.random()
                if r < p_fail:
                    new_states[name] = FAILED
                elif r < p_fail + p_recover:
                    new_states[name] = SECURE

            # FAILED is absorbing; no transition.

        current_states = new_states
        for name in node_names:
            node_states[name][step] = current_states[name]

        if (
            physical_equipment_failure_time is None
            and current_states[TARGET_NODE] == FAILED
        ):
            physical_equipment_failure_time = t

    return {
        "time": time,
        "node_states": node_states,
        "physical_equipment_failure_time": physical_equipment_failure_time,
    }


def run_monte_carlo(graph, ctmc_nodes, n_runs=1000, t_max=24, t_steps=100, seed=None):
    """Run run_attack_simulation n_runs times and aggregate failure stats.

    Returns a dict with 'probability_of_failure' (fraction of runs
    where physical_equipment reached FAILED), 'mean_time_to_failure'
    (mean over runs that failed, or None if none did), and
    'failure_times' (list of failure times from runs that failed).
    """
    rng = np.random.default_rng(seed)
    failure_times = []

    for _ in range(n_runs):
        result = run_attack_simulation(graph, ctmc_nodes, t_max=t_max, t_steps=t_steps, rng=rng)
        failure_time = result["physical_equipment_failure_time"]
        if failure_time is not None:
            failure_times.append(failure_time)

    probability_of_failure = len(failure_times) / n_runs
    mean_time_to_failure = float(np.mean(failure_times)) if failure_times else None

    return {
        "probability_of_failure": probability_of_failure,
        "mean_time_to_failure": mean_time_to_failure,
        "failure_times": failure_times,
    }


if __name__ == "__main__":
    smart_grid = build_smart_grid()
    ctmc_nodes = build_ctmc_nodes(smart_grid)

    mc_results = run_monte_carlo(smart_grid, ctmc_nodes, n_runs=1000, t_max=24)

    print("Monte Carlo simulation results (1000 runs, t_max=24h):")
    print(f"  P(physical_equipment fails) = {mc_results['probability_of_failure']:.4f}")
    if mc_results["mean_time_to_failure"] is not None:
        print(f"  Mean time to failure = {mc_results['mean_time_to_failure']:.4f} h")
    else:
        print("  Mean time to failure = N/A (no failures observed)")
    print(f"  Number of failed runs = {len(mc_results['failure_times'])} / 1000")
