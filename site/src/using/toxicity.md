# Toxicity screening

Structural-alert and predicted-liability readouts for a molecule, from the
command line. This is a repository tool (`tools/tox`), not part of the wheel.
Run it from a git clone of ferric with RDKit installed.

```
python -m tools.tox --offline "CC(=O)Oc1ccccc1C(=O)O"
```

The screen combines a **local RDKit pass** with two optional **web
providers**. The local pass checks several hundred compiled SMARTS patterns
from six published alert catalogs (Brenk, PAINS, NIH, and the Glaxo, Dundee and
BMS sets via ChEMBL), plus Lipinski and Veber rules, and needs no network.

**What the web providers return (probed 2026-09-23):**

- **ADMETlab 3.0**'s documented `POST /api/admet` returns HTTP 404. The client
  uses the live but undocumented `POST /api/single/admet` instead, one molecule
  per request. The host rate-limits hard, so expect HTTP 429, reported as an
  unavailable provider (exit 3), on larger batches.
- **ProTox-3.0** has no documented JSON API, and by design the provider never
  scrapes the HTML results page. It never contributes endpoints: every online
  run lists it as `unsupported`, and it never affects the exit status.

For a batch, `--offline` is still the safer default. The local screen needs no
network, and it is the part whose output does not depend on a third-party
service being up.

## Usage

```
python -m tools.tox [--offline | --require-online] [--fail-on-alerts]
                    [--timeout SECONDS] [--json] SMILES|FILE [SMILES|FILE ...]
```

| flag | effect |
|---|---|
| `--offline` | local RDKit screen only; makes no network call |
| `--require-online` | an unavailable online provider is a hard failure (exit 2), not a degraded run (exit 3). Cannot be combined with `--offline` |
| `--fail-on-alerts` | exit 4 when any molecule has a structural alert |
| `--timeout SECONDS` | wall-clock limit per web request (default 20). A provider that does not answer is not retried for the rest of the run |
| `--json` | machine-readable output on stdout, including each provider's `provider_status` |

An input is read as a file when it exists and ends in `.smi`, `.smiles` or
`.txt`; otherwise it is treated as a literal SMILES. A file holds one
`<smiles> [label]` per line, and `#` starts a comment.

```
# candidates.smi
CC(=O)Oc1ccccc1C(=O)O    aspirin
CN1C=NC2=C1C(=O)N(C)C(=O)N2C   caffeine
```

```
python -m tools.tox --offline candidates.smi
```

## Exit status

| code | meaning |
|---|---|
| 0 | clean: every molecule assessed by every provider that was asked to |
| 1 | usage or input error: a bad flag, an unparseable SMILES, a duplicate label, or nothing to assess |
| 2 | a required check did not run: the local screen failed, or `--require-online` was given and an online provider was unavailable |
| 3 | online checks unavailable: the local screen ran and is reported in full; each unavailable provider is named with its reason |
| 4 | structural alerts found (only with `--fail-on-alerts`) |

When several apply, the precedence is 1 > 2 > 4 > 3 > 0.

These are distinct on purpose. "No alerts found" and "the alert screen did not
run" produce similar-looking output, and they mean opposite things — so a
provider failure is never folded into success. A web-service outage is neither
a usage error nor a verdict, so it has its own status: 3 means the local
results are usable and the online predictions are missing. If a pipeline needs
the online predictions, pass `--require-online`. Without `--fail-on-alerts`,
alerts are reported but do not change the exit status.

Each provider's outcome is in the JSON as `provider_status`: `ok`,
`unavailable` (outage, HTTP error, rate limit, timeout, unparseable response),
`unsupported` (ProTox), `no_result`, `error` or `contract_violation`. The human
output shows the same tag in brackets on its `!!` lines.

Before 2026-09-23, every run without `--offline` exited 2, even when the local
screen succeeded: ADMETlab's documented path returned 404, and ProTox's
by-design note was counted as a failure.

## Reading the output

Abridged, for aspirin with `--offline` (each line is followed by a one-line
explanation, omitted here):

```
  alert_brenk                            0.3333 probability  [higher=worse] rdkit-alerts
  alert_pains                                 0 probability  [higher=worse] rdkit-alerts
  alert_total_count                           2 count        [higher=worse] rdkit-alerts
  desc_clogp                               1.31 log10        [higher=worse] rdkit-alerts
  desc_mw                                 180.2 Da           [higher=worse] rdkit-alerts
  lipinski_violation_fraction                 0 probability  [higher=worse] rdkit-alerts
```

Every line states its **polarity**, and the JSON carries the same flag as
`higher_is_worse`. An aggregator that guesses the direction will invert a
safety ranking, so read the flag rather than assuming one. ADMETlab's
F20%/F30% columns are an example of how easy the guess is to get wrong: they
are the probability of *low* oral bioavailability (below 20% or 30%), so they
are reported as `low_bioavailability_20pct` and `low_bioavailability_30pct`,
higher is worse. They were previously mapped as bioavailability, higher is
better, which inverted them.

A value of `None` means *unknown*, never zero. For a probability-valued
endpoint, `0.0` means "confidently predicted negative", which is the opposite
of "no information".

### What the alert scores are not

The `alert_*` endpoints are scaled hit counts (n/3, capped at 1.0). They are a
**rank-only liability density**, not a probability of toxicity. A molecule with
zero alerts is not thereby safe: danuglipron screens clean across all six
catalogs and was discontinued for a liver signal. Structural alerts catch known
problem substructures; they say nothing about dose, exposure or on-target
pharmacology.

## From Python

```python
from tools.tox.assess import assess_smiles, assess_many

a = assess_smiles("CCO", include_web=False)
for e in a.endpoints:
    if e.known:
        print(e.name, e.value, e.units, "higher_is_worse" if e.higher_is_worse else "")

# A batch reuses ONE provider list: compiling the SMARTS catalogs dominates
# the runtime of a per-molecule loop.
results = assess_many({"aspirin": "CC(=O)Oc1ccccc1C(=O)O"}, include_web=False)
```

`assess_many` returns results in input order and does not rank them — ranking
depends on what else is being traded off, so it is the caller's job.
