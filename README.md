# CPS Attack Simulation

A simulation framework for modeling cyberattack propagation in cyber-physical smart grid systems.
It represents the grid as a graph and models compromise/recovery dynamics using a continuous time Markov chain (CTMC).
Stochastic simulations trace how an attack spreads through the network over time.
An optimization module explores how to best allocate limited defense resources to reduce propagation risk.
Visualization tools help interpret the grid topology, attack spread, and defense effectiveness.

## --data-source: where the node vulnerability scores come from

Each node's `vulnerability` score in `src/graph.py` is derived from real CVE data in
`data/vulnerabilities.json`, not hand-picked. For each node, a representative
real-world product of that component type was chosen, 2-3 real CVEs against that
product were looked up on the NVD CVE API
(`https://services.nvd.nist.gov/rest/json/cves/2.0?cveId=<ID>`, the data source behind
`https://nvd.nist.gov/vuln/detail/<ID>`), and the node's vulnerability was computed as:

```
vulnerability = mean(exploitability_score for each CVE) / 3.9   # 3.9 = max CVSS v3.x exploitability subscore
                clipped to [0.05, 0.95]
```

CVSS `base_score` and `exploitability_score` values were taken verbatim from the
`nvd@nist.gov`-sourced CVSS v3.x entry for each CVE (retrieved 2026-10-29) — none were
invented or estimated. `source_url` in the JSON file links to each CVE's NVD page.

| Node | Representative component | CVEs used |
|---|---|---|
| `corporate_network` | Fortinet FortiGate SSL-VPN | CVE-2022-42475, CVE-2023-27997, CVE-2018-13379 |
| `historian_server` | AVEVA/OSIsoft PI Server (PI Data Archive / PI AF Server) | CVE-2023-34348, CVE-2016-4518, CVE-2025-36539 |
| `engineering_workstation` | Windows-based engineering workstation (OS-level RCE/privesc) | CVE-2017-0144 (EternalBlue), CVE-2021-34527 (PrintNightmare), CVE-2020-1472 (Zerologon) |
| `scada_server` | Siemens SIMATIC WinCC / PCS 7 | CVE-2019-10922, CVE-2016-5743, CVE-2021-40358 |
| `remote_terminal_unit` | Schneider Electric Modicon (Modbus protocol) | CVE-2017-6034, CVE-2017-6032, CVE-2022-45789 |
| `intelligent_electronic_device` | Siemens SIPROTEC 5 protection relay | CVE-2021-33719, CVE-2021-33720, CVE-2021-37206 |
| `physical_equipment` | *(none — see below)* | *(none)* |

**`physical_equipment` is the exception:** it represents the end physical process asset
(e.g. a circuit breaker or transformer), not networked software or firmware, so it has
no CPE entry and no CVEs in NVD. No score could be verified for it, so its
`vulnerability` is left at its original hardcoded value (0.2) in `src/graph.py` rather
than being derived from CVE data.

`criticality` for every node remains hand-set in `src/graph.py` and is not derived from
CVE data.
