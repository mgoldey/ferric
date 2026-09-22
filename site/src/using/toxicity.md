# Toxicity screening

Structural-alert and predicted-liability readouts for a molecule, from the
command line.

```
python -m tools.tox "CC(=O)Oc1ccccc1C(=O)O"
```

The screen combines a **local RDKit pass** — several hundred compiled SMARTS
patterns across six published alert catalogs, plus Lipinski and Veber rules —
with two optional **web predictors** (ADMETlab, ProTox). The local pass needs
no network.

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
| 0 | every molecule assessed |
| 1 | a SMILES could not be parsed |
| 2 | a provider failed |

These are distinct on purpose. "No alerts found" and "the alert screen did not
run" produce similar-looking output, and they mean opposite things — so a
provider failure is never folded into success. In a pipeline, treat a non-zero
status as *no result*, not as a clean molecule.

## Reading the output

```
  alert_total_count                    2 count        [higher=worse] rdkit-alerts
  desc_clogp                        1.31 log10        [higher=worse] rdkit-alerts
  desc_mw                          180.2 Da           [higher=worse] rdkit-alerts
  lipinski_violation_fraction          0 probability  [higher=worse] rdkit-alerts
```

Every line states its **polarity**. Roughly half of these endpoints are
"higher is worse" and half are not, so a bare number invites the wrong reading
and an aggregator that guesses the direction will invert a safety ranking. The
same flag is carried as `higher_is_worse` in the JSON.

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
