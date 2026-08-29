"""Arms A/B: active-site fit — pocket electrostatic complementarity of a pose.

## What this measures, stated narrowly on purpose

`pocket_interaction_kcal` is the interaction of the ligand's electron density
with the pocket's fixed classical point-charge field:

    E_int = E(ligand, in pocket field) - E(ligand, vacuum)     [same geometry]

It is **one term of a binding free energy** — the electrostatic/polarization
term. It is NOT a binding affinity and NOT a potency prediction. Missing:
desolvation of both partners, the pocket's own reorganization, dispersion beyond
what the Hamiltonian carries, entropy, and any protein flexibility. A ranking on
this number is a ranking on electrostatic complementarity alone.

That narrowness is the point. Electrostatic complementarity is the term the
danuglipron pharmacophore is built around — the carboxylate anchor and the
electron-poor aryl near Trp33 — so it is the term most likely to discriminate a
modification that keeps those contacts from one that breaks them. And it is
cheap enough to run over a whole analogue set.

## Why the negative controls matter more than the candidates

The stated artifact hypothesis (`scripts/danuglipron/PLAN.md`): if this metric
works, `NC1-methyl-ester` (acid anchor deleted) and `NC2-decyano` (Trp33
terminus deleted) must score CLEARLY WORSE than the parent. `rank.
fit_discriminates_controls` performs that check, and it is a gate on the whole
campaign, not a diagnostic: a metric that cannot separate a known-inactive
control licenses no candidate ranking.

WHAT THE CONTROLS ACTUALLY FOUND (measured 2026-08-29, and it is a mixed
verdict worth knowing before reusing this module):

- **NC1 is discriminated decisively**: +103.7 kcal/mol worse than the parent
  (n=100). But see below -- NC1 is the only NEUTRAL species in the set, so this
  separation is charge detection, not pharmacophore recognition.
- **NC2 scores significantly BETTER than the parent**: -18.5 kcal/mol against a
  13.5 kcal/mol 2-sigma bar at n=100. A pharmacophore-deleted inactive
  outranking the parent REFUTES the metric.

At n=40 NC2 was merely unresolved (gap 17.8, bar 22.0) and this docstring said
the metric "cannot resolve" it. That was a precision statement and it is now
retracted: raising to n=100 dropped the SEM as 1/sqrt(n) exactly as predicted
(sd flat at ~46, SEM 7.6 -> 4.9) and the pair resolved -- on the WRONG SIDE.
More sampling made the refutation stronger, not weaker.

WHAT THE METRIC ACTUALLY MEASURES (2026-08-29, n=100):

  anion (q=-1) vs neutral (q=0)        -109.5 kcal/mol
  full spread among the 10 anions        41.4 kcal/mol
  r(MW, fit) among anions only            +0.490

It is dominated by formal charge. And note the correction to an earlier claim in
this file: "size is ruled out (r = +0.132)" was measured on the MIXED-charge set,
where a ~110 kcal/mol charge term swamped everything else. Controlling for charge
reverses it -- among the ten anions the size correlation is +0.490. Both numbers
are right about their own set; the mixed-charge one is the misleading one.

So this metric is REFUTED for ranking these analogues, not merely imprecise, and
the fix is not more poses. See scripts/danuglipron/RESULTS.md M5.

## Reference-state discipline

Vacuum and in-field energies must be at the SAME geometry, or the difference
picks up a relaxation energy as well. `pose_fit` enforces that by construction:
it computes both from one coordinate array.
"""
from __future__ import annotations

from dataclasses import dataclass

from .xtb_engine import HARTREE_TO_KCAL_MOL, singlepoint

# Sourced from tools.active_site.pocket_charges rather than redeclared, so the
# ligand->Bohr conversion here cannot drift from the one the pocket charges were
# generated with. A mismatch would put ligand and pocket in different unit
# systems, which shows up as "no pocket charges near this pose" -- an
# UNEVALUATED result rather than a wrong number, but a confusing one to debug.
from tools.active_site.pocket_charges import ANGSTROM_TO_BOHR

# Pocket charges beyond this distance from every ligand atom contribute
# negligibly (a 1 e charge at 25 Bohr is ~0.025 Ha*e, and the pocket is
# near-neutral so contributions cancel further). Trimming them keeps the xtb
# pcharge file small; the cut is on the SAME criterion for every pose so it
# cannot bias a comparison.
DEFAULT_FIELD_CUTOFF_BOHR = 30.0


@dataclass
class FitResult:
    """Electrostatic fit of one pose in one pocket."""
    label: str
    e_vacuum: float | None            # Hartree
    e_in_field: float | None          # Hartree
    interaction_kcal: float | None    # negative = favorable
    n_pocket_charges: int
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.interaction_kcal is not None


def _trim_charges(
    point_charges,
    ligand_coords_bohr,
    cutoff_bohr: float,
):
    """Drop charges far from every ligand atom. Same rule for every pose."""
    if cutoff_bohr is None:
        return list(point_charges)
    keep = []
    c2 = cutoff_bohr * cutoff_bohr
    for q, x, y, z in point_charges:
        for lx, ly, lz in ligand_coords_bohr:
            dx, dy, dz = x - lx, y - ly, z - lz
            if dx * dx + dy * dy + dz * dz <= c2:
                keep.append((q, x, y, z))
                break
    return keep


def pose_fit(
    symbols: list[str],
    coords_angstrom: list[tuple[float, float, float]],
    point_charges,
    label: str = "pose",
    charge: int = 0,
    field_cutoff_bohr: float | None = DEFAULT_FIELD_CUTOFF_BOHR,
) -> FitResult:
    """Interaction energy of one pose with a fixed point-charge pocket.

    `point_charges` is a list of `(q, x, y, z)` with coordinates in BOHR — the
    convention `tools.active_site.pocket_charges` produces and xtb's `pcharge`
    file expects. Both energies come from the same `coords_angstrom`, so the
    difference contains no relaxation energy.
    """
    ligand_bohr = [
        (x * ANGSTROM_TO_BOHR, y * ANGSTROM_TO_BOHR, z * ANGSTROM_TO_BOHR)
        for x, y, z in coords_angstrom
    ]
    charges = _trim_charges(point_charges, ligand_bohr, field_cutoff_bohr)

    if not charges:
        return FitResult(
            label, None, None, None, 0,
            error=(
                "no pocket charges within the field cutoff of this pose -- the "
                "pose is outside the pocket, or the coordinate frames of ligand "
                "and pocket disagree. Reporting UNEVALUATED rather than an "
                "interaction of 0, which would rank as perfectly neutral."
            ),
        )

    vac = singlepoint(symbols, coords_angstrom, charge=charge)
    if not vac.ok:
        return FitResult(label, None, None, None, len(charges),
                         error=f"vacuum single point failed: {vac.error}")

    fld = singlepoint(symbols, coords_angstrom, charge=charge, point_charges=charges)
    if not fld.ok:
        return FitResult(label, vac.energy, None, None, len(charges),
                         error=f"in-field single point failed: {fld.error}")

    return FitResult(
        label=label,
        e_vacuum=vac.energy,
        e_in_field=fld.energy,
        interaction_kcal=(fld.energy - vac.energy) * HARTREE_TO_KCAL_MOL,
        n_pocket_charges=len(charges),
    )


def best_pose_fit(
    symbols_per_conformer: list[list[str]],
    conformers: list[list[tuple[float, float, float]]],
    point_charges,
    labels: list[str] | None = None,
    charge: int = 0,
    field_cutoff_bohr: float | None = DEFAULT_FIELD_CUTOFF_BOHR,
) -> tuple[FitResult | None, list[FitResult]]:
    """Score every conformer, return (best, all).

    "Best" is the most negative interaction energy. Note this scores each
    conformer AT ITS OWN COORDINATES: for a docked ensemble those share the
    pocket frame, but for a freely-generated analogue ensemble they do NOT --
    such conformers need aligning into the pocket first (see
    `align.py`/`tools.active_site`), or every one of them will fall outside the
    field cutoff and be reported UNEVALUATED. That is the intended, loud
    failure rather than a silent zero.
    """
    labels = labels or [f"conf_{i:02d}" for i in range(len(conformers))]
    results = [
        pose_fit(syms, coords, point_charges, label=lbl, charge=charge,
                 field_cutoff_bohr=field_cutoff_bohr)
        for syms, coords, lbl in zip(symbols_per_conformer, conformers, labels)
    ]
    good = [r for r in results if r.ok]
    best = min(good, key=lambda r: r.interaction_kcal) if good else None
    return best, results
