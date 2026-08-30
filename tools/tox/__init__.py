"""External toxicity-liability assessment for candidate ligands.

**ferric does not predict toxicity.** ferric is an electronic-structure engine;
DILI, hERG and Ames liability are empirical/statistical endpoints with no route
from a converged SCF. Everything in this package therefore delegates to an
EXTERNAL source and is explicit about which one, so no toxicity number in this
repo can ever be mistaken for a ferric result.

Providers, in the order the driver prefers them:

- `alerts.RdkitAlertsProvider` — published structural-alert catalogs (Brenk,
  PAINS, NIH, ChEMBL/Glaxo/Dundee/BMS) as shipped in RDKit's `FilterCatalog`,
  plus Lipinski/Veber physicochemical rules. Offline, deterministic, always
  available. This is the BASELINE that always runs.
- `web.AdmetlabProvider` — ADMETlab 3.0 REST (`/api/admet`). 119 endpoints
  including DILI/hERG/Ames/H-HT. **Endpoint was returning 404 on 2026-08-29**;
  kept because it is the right primary source when the service returns, and it
  degrades to `None` rather than fabricating.
- `web.ProToxProvider` — ProTox-3.0 organ-specific toxicity / LD50.

## The one invariant that matters

A provider that cannot answer returns `None`, **never** `0.0`. A fabricated
zero would rank a compound as maximally safe — the single most dangerous
possible failure mode for this package, and the reason `ToxAssessment.
liability_score` is `float | None` and every consumer must branch on it.
`tests/test_provider_contract.py` asserts this for every provider, including
against a deliberately unreachable host.
"""

from .model import ToxAssessment, ToxEndpoint, ToxProvider
from .alerts import RdkitAlertsProvider
from .web import AdmetlabProvider, ProToxProvider
from .assess import assess_smiles, assess_many

__all__ = [
    "ToxAssessment",
    "ToxEndpoint",
    "ToxProvider",
    "RdkitAlertsProvider",
    "AdmetlabProvider",
    "ProToxProvider",
    "assess_smiles",
    "assess_many",
]
