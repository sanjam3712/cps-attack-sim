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
(or, where no clean CPE exists, an NVD keyword search) in the last 5 years was pulled
from the NVD CVE API
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

| Node | Representative component | Query | n (last 5y, has CVSSv3) | median exploitability | derived vulnerability |
|---|---|---|---|---|---|
| `corporate_network` | Fortinet FortiGate / FortiOS SSL-VPN | CPE `fortinet:fortios` | 203 | 2.8 | 0.7179 |
| `historian_server` | AVEVA/OSIsoft PI Server, PI Data Archive | CPE `osisoft:pi_server` + `aveva:pi_server` + `osisoft:pi_data_archive` | **2 ⚠️ low confidence** | 3.9 | 0.95 (capped) |
| `engineering_workstation` | Windows-based engineering workstation (Windows 10 OS) | CPE `microsoft:windows_10` | 664 | 1.8 | 0.4615 |
| `scada_server` | Siemens SIMATIC WinCC | CPE `siemens:simatic_wincc` | 9 (small sample) | 2.8 | 0.7179 |
| `remote_terminal_unit` | Schneider Electric Modicon (Modbus) | keyword `Modicon` | 21 | 3.9 | 0.95 (capped) |
| `intelligent_electronic_device` | Siemens SIPROTEC protection relay | keyword `SIPROTEC` | 13 (small sample) | 2.8 | 0.7179 |
| `physical_equipment` | *(none — see below)* | *(none)* | 0 | — | 0.2 (hardcoded fallback) |

**Caveats (see `_meta.data_quality_flags` in the JSON for the full text):**
- `historian_server` matched only **2** CVEs in the last 5 years — too few to trust as a
  population median; it also happens to land exactly at the 0.95 cap. Treat as anecdotal.
- `scada_server` (n=9) and `intelligent_electronic_device` (n=13) are small samples;
  treat with moderate caution.
- `remote_terminal_unit`'s median genuinely sits at the CVSS ceiling (3.9) with a
  reasonable sample size (n=21) — more than half of matched Modicon/Modbus CVEs in the
  last 5 years are unauthenticated and network-exploitable, so the 0.95 cap here
  reflects a real population result, not an outlier artifact.
- `engineering_workstation` is scoped to generic Windows 10 (the OS layer), a much
  broader and noisier population (n=664) than the ICS-vendor-specific products used for
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
