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

**Use `--offline` today.** Neither web provider currently returns endpoints:

- **ADMETlab 3.0**'s documented API returned HTTP 404 on every path tried when
  it was last probed (2026-08-29). The client degrades cleanly and will start
  contributing again if the service comes back.
- **ProTox-3.0** has no documented JSON API. The provider only checks that the
  site is reachable, and by design it never scrapes the HTML results page.

Both record their reason as a provider error. So a run without `--offline`
currently exits with status **2** even when the local screen succeeded
(MEASURED 2026-09-23).

## Usage

```
python -m tools.tox [--offline] [--json] SMILES|FILE [SMILES|FILE ...]
```

| flag | effect |
|---|---|
| `--offline` | local RDKit screen only; makes no network call |
| `--json` | machine-readable output on stdout |

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
| 0 | every molecule assessed, and no provider reported an error |
| 1 | a SMILES could not be parsed, or there was nothing to assess |
| 2 | a provider failed or returned nothing (currently every online run; see above) |

These are distinct on purpose. "No alerts found" and "the alert screen did not
run" produce similar-looking output, and they mean opposite things — so a
provider failure is never folded into success. In a pipeline, treat a non-zero
status as *no result*, not as a clean molecule.

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
`higher_is_worse`. The local endpoints are all "higher is worse". Some web
endpoints are not: ADMETlab's oral-bioavailability columns, for example, are
"higher is better". An aggregator that guesses the direction will invert a
safety ranking. Read the flag, and don't assume a direction.

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
