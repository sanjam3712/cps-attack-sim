# CPS Attack Simulation

A simulation framework for modeling cyberattack propagation in cyber-physical smart grid systems.
It represents the grid as a graph and models compromise/recovery dynamics using a continuous time Markov chain (CTMC).
Stochastic simulations trace how an attack spreads through the network over time.
An optimization module explores how to best allocate limited defense resources to reduce propagation risk.
Visualization tools help interpret the grid topology, attack spread, and defense effectiveness.

## --data-source: where the node vulnerability scores come from

Each node's `vulnerability` score in `src/graph.py` is derived from real CVE data in
`data/vulnerabilities.json` — specifically, from an **unbiased population pull**, not
hand-picked CVEs. For each node, a representative real-world product of that
component type was chosen, and *every* CVE published against that product's NVD CPE
(or, where no clean CPE exists, an NVD keyword search) in the last **10 years** (the
same window for every node, for consistency) was pulled from the NVD CVE API
(`https://services.nvd.nist.gov/rest/json/cves/2.0`), filtered to entries that carry a
CVSS v3.x score. The node's vulnerability was then computed as:

```
derived_vulnerability = median(exploitability_score over ALL matching CVEs) / 3.9
                         clipped to [0.05, 0.95] only if the raw value falls outside it
```

3.9 is the maximum possible CVSS v3.x exploitability subscore (AV:N/AC:L/PR:N/UI:N).
`data/vulnerabilities.json` also stores, per node: the exact query used, the retrieval
date, `n_cves`, and the `min`/`max`/`iqr` of the exploitability-score distribution, so
the result is reproducible. The full unfiltered API response per node is saved under
`data/raw/<node>.json` (pruned of unused bulk fields — configurations/CPE-match arrays,
non-English text — but every CVE ID, date, description and CVSS v3.x entry is kept
verbatim); the exact set of CVE IDs kept after filtering is in `data/raw/_summary.json`.

| Node | Representative component | Query | n (last 10y, has CVSSv3) | median exploitability | derived vulnerability |
|---|---|---|---|---|---|
| `corporate_network` | Fortinet FortiGate / FortiOS SSL-VPN | CPE `fortinet:fortios` | 258 | 2.8 | 0.7179 |
| `historian_server` | AVEVA/OSIsoft PI Server, PI Data Archive, PI System | CPE `osisoft:pi_server` + `aveva:pi_server` + `osisoft:pi_data_archive` ∪ keyword `"PI System"` (exact match) | 22 | 2.2 | 0.5641 |
| `engineering_workstation` | Windows-based engineering workstation (Windows 10 OS) | CPE `microsoft:windows_10` | 3660 | 1.8 | 0.4615 |
| `scada_server` | Siemens SIMATIC WinCC | CPE `siemens:simatic_wincc` | 25 | 2.8 | 0.7179 |
| `remote_terminal_unit` | Schneider Electric Modicon (Modbus) | keyword `Modicon` | 136 | 3.9 | 0.95 (capped) |
| `intelligent_electronic_device` | Siemens SIPROTEC protection relay | keyword `SIPROTEC` | 23 (small sample) | 3.9 | 0.95 (capped) |
| `physical_equipment` | *(none — see below)* | *(none)* | 0 | — | 0.2 (hardcoded fallback) |

All six scored nodes now use the **same 10-year window** (previously 5 years), for
methodological consistency across nodes.

**`historian_server` query was widened** after the 5-year pull returned only 2 CVEs
(too few to trust). A bare `keywordSearch="PI System"` was tried first and rejected —
of 1022 total hits, a 10-item manual sample was **0% relevant** (Windows 95/98, Cisco
PIX, Apache `rotatelogs`, etc. — loose substring matching on common words). Adding
`keywordExactMatch=true` to the same keyword fixed this: 15 all-time hits, all
genuinely AVEVA/OSIsoft PI System CVEs on manual sampling. The final query is the
**union** of the existing CPE matches and this exact-match keyword search, which raised
the 10-year sample from 2 to 22.

**Caveats (see `_meta.data_quality_flags` in the JSON for the full text):**
- `historian_server` (n=22) is no longer a 2-CVE anecdote, but is still the smallest
  sample among the networked nodes; treat with somewhat more caution than
  `corporate_network` or `engineering_workstation`.
- `scada_server` (n=25) and `intelligent_electronic_device` (n=23) are reasonable but
  not large samples; treat with light to moderate caution.
- `remote_terminal_unit`'s and `intelligent_electronic_device`'s medians genuinely sit
  at the CVSS ceiling (3.9) and are clipped to 0.95. `remote_terminal_unit` (n=136) is
  well-supported: a large majority of matched Modicon/Modbus CVEs over 10 years are
  unauthenticated and network-exploitable. `intelligent_electronic_device` (n=23) shows
  the same pattern with a smaller sample.
- `engineering_workstation` is scoped to generic Windows 10 (the OS layer), a much
  broader and noisier population (n=3660) than the ICS-vendor-specific products used for
  the other nodes; most of those CVEs have nothing to do with ICS/OT specifically.

**`physical_equipment` is the exception:** it represents the end physical process asset
(e.g. a circuit breaker or transformer), not networked software or firmware, so it has
no CPE entry and no CVEs in NVD. No score could be verified for it, so its
`vulnerability` is left at its original hardcoded value (0.2) in `src/graph.py` rather
than being derived from CVE data.

The previous hand-picked CVEs (3 "famous" CVEs per node) are kept in
`data/vulnerabilities.json` under each node's `legacy_famous_cves` key purely for
comparison — they are no longer used to compute `derived_vulnerability`.

`criticality` for every node remains hand-set in `src/graph.py` and is not derived from
CVE data.

## Run the app

An interactive Streamlit UI (`app.py`) wraps the same simulation and optimization code
with sliders and plots, so results can be explored without editing code or re-running
`main.py` for every parameter change:

```
pip install -r requirements.txt
streamlit run app.py
```

In the sidebar: pick a config from `configs/` (or upload your own JSON config in the
same format - it's validated on upload, with a clear error if something's missing or
malformed), set the defense budget, defense strength, Monte Carlo run count, time
horizon and seed, then click **Run simulation**. Results are cached, so re-running with
identical settings is instant. The page shows summary metric cards, the system graph,
an example attack-propagation trace, the failure-time distribution, a defense
comparison bar chart with 95% CI error bars, and a table of the top 5 defense
combinations (flagging when the top two are not statistically distinguishable).

## Run on your own system

The simulation is system-agnostic: any CPS topology can be modeled by writing a JSON
config instead of editing code. Run it with:

```
python main.py --system configs/your_system.json --budget 2 --reduction 0.5 --n-runs 5000 --seed 42
```

- `--system`: path to the config (default `configs/substation.json`)
- `--budget`: number of nodes the defender can harden at once
- `--reduction`: fractional cut to a defended node's compromise rate (0.5 = halved)
- `--n-runs` / `--seed`: Monte Carlo run count and RNG seed

### Config format

```json
{
  "nodes": {
    "node_name": {"vulnerability": 0.0, "criticality": 0.0}
  },
  "edges": [
    ["source_node", "target_node"]
  ],
  "entry_points": {
    "node_name": 1.0
  },
  "target": "node_name"
}
```

- `nodes`: every node in the graph, with `vulnerability` (compromise likelihood, 0-1)
  and `criticality` (failure likelihood once compromised, 0-1).
- `edges`: directed attack-propagation edges `[source, target]`. Each edge's
  propagation probability is set to the target node's `vulnerability`.
- `entry_points`: node(s) the attacker starts from, with selection probability per
  Monte Carlo run (probabilities should sum to 1).
- `target`: the node whose failure probability is tracked and reported.

Two example configs are included:

- `configs/substation.json` — the original hand-built smart-grid substation topology,
  with CVE-derived vulnerability scores (see the data source section above).
- `configs/water_plant.json` — an **illustrative** water-treatment-plant example (8
  nodes: remote access, engineering workstation, HMI, historian, PLC, pump controller,
  chemical doser, and the physical tank). Its `vulnerability`/`criticality` values are
  placeholder estimates, not CVE-derived — it exists to demonstrate the config format on
  a second system, not to support real conclusions.
