"""Driver: run every configured provider over a molecule (or a set) and collect.

`assess_smiles` is the single entry point the danuglipron experiment calls. It
exists mainly to enforce two things the providers themselves cannot:

1.  **Provider failures are recorded, not swallowed.** Each provider's
    `last_error` (or the exception it raised, which is a contract violation and
    therefore reported loudly) lands in `ToxAssessment.provider_errors`. A run
    with three dead web services and one live offline baseline is usable, but
    only if the caller can see that.

2.  **A molecule nothing could score gets `liability_score is None`.** Not 0.0,
    not omitted from the output — present, unranked, and visibly so.
"""
from __future__ import annotations

from .alerts import RdkitAlertsProvider
from .model import ToxAssessment, ToxProvider


def default_providers(include_web: bool = True) -> list[ToxProvider]:
    """Offline baseline first, then web sources.

    Order matters only for reporting (endpoints appear in provider order); the
    aggregate is order-independent. The offline provider comes first so that a
    truncated/interrupted run still has the baseline in hand.
    """
    providers: list[ToxProvider] = [RdkitAlertsProvider()]
    if include_web:
        from .web import AdmetlabProvider, ProToxProvider

        providers.append(AdmetlabProvider())
        providers.append(ProToxProvider())
    return providers


def assess_smiles(
    smiles: str,
    providers: list[ToxProvider] | None = None,
    label: str = "",
    include_web: bool = True,
) -> ToxAssessment:
    """Gather every provider's endpoints for one SMILES."""
    if providers is None:
        providers = default_providers(include_web=include_web)

    assessment = ToxAssessment(smiles=smiles, label=label)
    for p in providers:
        pname = getattr(p, "name", type(p).__name__)
        try:
            endpoints = p.fetch(smiles)
        except Exception as e:  # noqa: BLE001
            # A provider raising for a network/parse failure violates the
            # contract in model.ToxProvider. Record it as a provider error
            # rather than aborting the batch, but make the wording say clearly
            # that this is a bug in the provider, not just an absent service.
            assessment.provider_errors[pname] = (
                f"provider raised {type(e).__name__} (contract violation -- "
                f"providers must return [] for service failures): {e}"
            )
            continue
        assessment.endpoints.extend(endpoints)
        err = getattr(p, "last_error", None)
        if err:
            assessment.provider_errors[pname] = err
        elif not endpoints:
            assessment.provider_errors[pname] = (
                "returned no endpoints and reported no error"
            )
    return assessment


def assess_many(
    smiles_by_label: dict[str, str],
    providers: list[ToxProvider] | None = None,
    include_web: bool = True,
) -> list[ToxAssessment]:
    """Assess a labelled set, reusing ONE provider list across all molecules.

    Reuse is the point: `RdkitAlertsProvider.__init__` compiles several hundred
    SMARTS patterns, so constructing it per molecule dominates a batch's
    runtime. Returns results in input order (not ranked) -- ranking is the
    caller's job, since it depends on what else is being traded off.
    """
    if providers is None:
        providers = default_providers(include_web=include_web)
    return [
        assess_smiles(smi, providers=providers, label=label)
        for label, smi in smiles_by_label.items()
    ]
