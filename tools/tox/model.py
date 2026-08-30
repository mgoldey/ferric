"""Data model shared by every toxicity provider.

The types here exist to make one specific mistake impossible: silently
treating "the service did not answer" as "the compound is safe". Both
`ToxEndpoint.value` and `ToxAssessment.liability_score` are therefore
`Optional`, and there is no default that means "fine".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class ToxEndpoint:
    """One named toxicity/liability readout from one external source.

    `value` is `None` when the source could not produce this endpoint. It is
    NEVER 0.0-as-unknown: for a probability-valued endpoint 0.0 means
    "confidently predicted negative", which is the opposite of "unknown".

    `higher_is_worse` records the polarity so a caller can aggregate endpoints
    from different sources without hardcoding per-endpoint knowledge. It has no
    default: a wrong polarity silently inverts a safety ranking, so the
    provider must state it.
    """
    name: str
    value: float | None
    higher_is_worse: bool
    source: str
    units: str = "probability"
    note: str = ""

    @property
    def known(self) -> bool:
        return self.value is not None


@dataclass
class ToxAssessment:
    """All endpoints gathered for one molecule, plus a single scalar for ranking.

    `liability_score` is a *rank-only* aggregate — dimensionless, higher = more
    liability. It is `None` when no provider produced a single usable endpoint,
    which is the signal that this molecule cannot be ranked at all rather than
    that it ranked well.

    `provider_errors` keeps the reason each failing provider failed. A run where
    every web provider 404s but the offline alert baseline succeeded is a
    perfectly usable run, but the caller must be able to SEE that the web
    endpoints are missing rather than infer safety from their absence — so the
    errors travel with the result instead of being logged and dropped.
    """
    smiles: str
    endpoints: list[ToxEndpoint] = field(default_factory=list)
    provider_errors: dict[str, str] = field(default_factory=dict)
    label: str = ""

    @property
    def known_endpoints(self) -> list[ToxEndpoint]:
        return [e for e in self.endpoints if e.known]

    @property
    def sources(self) -> list[str]:
        seen: list[str] = []
        for e in self.known_endpoints:
            if e.source not in seen:
                seen.append(e.source)
        return seen

    def endpoint(self, name: str) -> ToxEndpoint | None:
        for e in self.endpoints:
            if e.name == name:
                return e
        return None

    @property
    def liability_score(self) -> float | None:
        """Mean of known endpoints, polarity-corrected. `None` if none known.

        Deliberately a plain unweighted mean over whatever was actually
        measured, not a tuned composite: any weighting would be an invented
        toxicology model, which is exactly what this package refuses to do.
        Endpoints on a non-probability scale are excluded — averaging an LD50 in
        mg/kg against a probability is meaningless — and are left for callers
        who want them to read individually via `endpoint()`.
        """
        vals = [
            (e.value if e.higher_is_worse else 1.0 - e.value)
            for e in self.known_endpoints
            if e.units == "probability" and e.value is not None
        ]
        if not vals:
            return None
        return sum(vals) / len(vals)


class ToxProvider(Protocol):
    """A source of external toxicity endpoints for a SMILES string.

    Contract, asserted by `tests/test_provider_contract.py` for every
    implementation:

    - `name` is a stable, human-readable source identifier that appears in
      every `ToxEndpoint.source` the provider emits.
    - `fetch` returns a list of endpoints, possibly empty. It MUST NOT raise
      for an unreachable service, a rate limit, or an unparseable response —
      those are normal and must surface as an empty list (the driver records
      the reason separately). It MAY raise for programmer error (e.g. a
      non-string argument).
    - An endpoint it cannot determine is omitted, or included with
      `value=None`. Never `value=0.0`.
    """

    name: str

    def fetch(self, smiles: str) -> list[ToxEndpoint]:
        ...
