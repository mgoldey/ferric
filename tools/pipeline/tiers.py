"""Thin adapters presenting each existing tier with one uniform signature.

Deliberately thin: the chemistry lives in `tools/docking`, `tools/morph` and
`tools/campaign` and is NOT reimplemented here. This module exists so the funnel
can call four very different methods -- an empirical docking score, a force
field, a semiempirical Hamiltonian and a DFT SCF -- without knowing anything
about any of them.

MEASURED costs (2026-09-19, through the tier functions; tier 1 from RESULTS.md M11):

    tier 1  Vina           26.4 s/ligand at exhaustiveness 4, cpu=0 (12 cores)
                           109.0 s at cpu=1; ~2 min at the old ex=32
    tier 2  MMFF94         2.2 ms @ 9 atoms, 8.2 @ 19, 21.6 @ 34
                           (~73 ms projected @ 71; embed + OPTIMIZE, not a
                           single point -- the old "~1 ms/pose" here was
                           never measured and the golden path CITED THIS
                           LINE as its source)
    tier 3  GFN2-xTB       0.152 s @ 9 atoms, 0.050 @ 19 (via tier3_gfn2;
                           the old "~0.5 s single point" was the right
                           order but was never measured)
    tier 4  ferric DFT     0.66 s @ 9, 8.7 @ 19 (STO-3G, via tier4_dft);
                           96.1 s @ 32 at def2-SVP; 612 s @ 71 at STO-3G.
                           Scales ~N^2.3 in ATOM COUNT, not N^3-N^4:
                           measured three ways (a PBE/STO-3G sweep, an
                           RHF/STO-3G sweep at 2.58, and the 32->71 pair),
                           all agreeing on 2.3-2.6. The old N^3-N^4 was the
                           textbook basis-function scaling, which is not
                           what varies when a MOLECULE grows at fixed basis.
            + D3(BJ)       microseconds -- a pairwise sum over atoms, free
                           next to the SCF. Its analytic GRADIENT is the same
                           order, so dispersion-corrected OPTIMIZATION costs
                           no more than uncorrected.

Tier 4's cost is the reason the funnel must narrow to a handful before reaching
it. See `tools/campaign/hierarchy.py` for the rules.
"""

from __future__ import annotations

import math

from dataclasses import dataclass, field
from typing import Any

from tools.isomers.model import Isomer


@dataclass
class TierResult:
    """One tier's verdict on one candidate.

    `value` is `None` for ANY failure -- never 0.0, which in an energy ranking
    reads as the best possible score and would promote a broken candidate to
    the top of the funnel.
    """

    candidate_id: str
    value: float | None
    error: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    #: Smallest difference in `value` this tier can actually resolve, in the
    #: same units. `None` means UNCHARACTERISED -- not "infinitely precise".
    #:
    #: This exists because `value` alone invites a comparison the tier cannot
    #: support. MEASURED on the danuglipron campaign, the best available ddE
    #: noise over a pose ensemble is 4.07 kcal/mol against substituent effects
    #: of 1-2 (RESULTS.md M4-M13), so two candidates 0.5 kcal/mol apart are
    #: indistinguishable no matter how many digits the tier prints.
    resolution: float | None = None

    def __post_init__(self) -> None:
        """Reject a resolution that cannot mean what the field promises.

        `resolves` squares this, so a NEGATIVE value behaves exactly as its
        absolute value -- a caller passing -4.0 gets the comparisons of +4.0
        and no indication. A NaN is worse: every `>` against it is False, so
        `resolves` reports "indistinguishable" for ANY gap. MEASURED before the
        fix, two results 990 units apart with `resolution=nan` came back
        `False`, i.e. do-not-rank.

        That inverts the field's whole purpose. `resolution` exists to stop a
        ranking the tier cannot support; a NaN instead suppresses every real
        distinction, and it does so in the direction that looks CAUTIOUS.

        Zero is allowed: a tier that genuinely resolves exact ties is a
        coherent claim, and the quadrature handles it.
        """
        if self.resolution is None:
            return
        r = self.resolution
        if not isinstance(r, (int, float)) or isinstance(r, bool):
            raise TypeError(f"resolution must be a real number or None, got {r!r}")
        # COERCE to float HERE, before the range checks. `10**400` is a finite,
        # positive Python int: it passes every check below and then raises
        # OverflowError inside `resolves`. A field that validates and then
        # throws downstream is worse than one that never validated -- the
        # caller has been told it is safe.
        try:
            r = float(r)
        except (OverflowError, ValueError) as exc:
            raise ValueError(
                f"resolution {self.resolution!r} cannot be represented as a "
                f"float ({exc}). A resolution beyond the float range is not a "
                "measurement."
            ) from exc
        object.__setattr__(self, "resolution", r)
        if r != r:  # NaN
            raise ValueError(
                f"resolution is NaN for {self.candidate_id!r}. Every comparison "
                "against NaN is False, so `resolves` would report EVERY gap as "
                "indistinguishable -- the opposite of what an unknown "
                "resolution means. Pass None for UNCHARACTERISED."
            )
        if r < 0.0:
            raise ValueError(
                f"resolution must be >= 0, got {r} for {self.candidate_id!r}. "
                "`resolves` squares it, so a negative value silently behaves as "
                "its absolute value."
            )
        if r == float("inf"):
            raise ValueError(
                f"resolution is infinite for {self.candidate_id!r}, which makes "
                "every gap unresolvable. If the tier cannot resolve anything, "
                "that is a statement worth making explicitly rather than "
                "through an arithmetic edge case."
            )

    @property
    def ok(self) -> bool:
        return self.error is None and self.value is not None

    def resolves(self, other: "TierResult") -> bool | None:
        """Is the gap to `other` larger than what this tier can resolve?

        `True`  -- the difference is real at this tier's stated resolution.
        `False` -- the two are indistinguishable; do NOT rank them.
        `None`  -- UNKNOWN, because a resolution was never characterised. It is
                   deliberately not `True`: an uncharacterised tier has not
                   earned the benefit of the doubt, and returning `True` here
                   would make every unlabelled tier look infinitely precise.

        Raises if either result failed -- comparing to a non-answer is a
        caller bug, not a `False`.
        """
        if not self.ok or not other.ok:
            raise ValueError(
                f"cannot compare {self.candidate_id!r} to {other.candidate_id!r}: "
                "one of them has no value. Check .ok first; a failed tier is not "
                "a tie."
            )
        # Both resolutions must be known. Using only one side's would silently
        # assume the other is at least as good.
        if self.resolution is None or other.resolution is None:
            return None
        # Quadrature: a difference carries both results' noise.
        # `math.hypot` rather than `(a**2 + b**2) ** 0.5`: squaring overflows
        # for any resolution above ~1.3e154, so `1e200` raised OverflowError
        # from a function whose whole job is to answer a yes/no question.
        # hypot computes the same value without the intermediate square.
        combined = math.hypot(self.resolution, other.resolution)
        gap = abs(self.value - other.value)
        # BOTH sides can saturate to inf near the float maximum, and `inf > inf`
        # is False -- so two values 3.4e308 apart with 1.7e308 noise each come
        # back "indistinguishable" when exact arithmetic says otherwise. ONE
        # wrong verdict, found by comparing against `fractions.Fraction`.
        #
        # Unreachable with real inputs: `resolution` is a noise figure in the
        # value's own units, and the largest MEASURED in this repo is 4.07
        # kcal/mol. Fixed anyway because the repair is two lines -- halve both
        # sides, which cannot change an inequality and moves everything back
        # inside the representable range.
        if gap == float("inf") or combined == float("inf"):
            half_gap = abs(self.value / 2.0 - other.value / 2.0)
            half_combined = math.hypot(self.resolution / 2.0, other.resolution / 2.0)
            return half_gap > half_combined
        return gap > combined


def tier2_forcefield(iso: Isomer, context: dict) -> TierResult:
    """MMFF94 embed + optimize. Cheap declash; NOT a ranking method.

    Measured: GFN2 moves an MMFF geometry by 12-14 kcal/mol and 0.13-0.39 A, so
    MMFF energies are adequate to reject a clashing structure and inadequate to
    order two reasonable ones.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(iso.canonical)
    if mol is None:
        return TierResult(iso.canonical, None, "unparseable SMILES")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = context.get("seed", 0xF00D)
    params.useSmallRingTorsions = True
    try:
        if AllChem.EmbedMolecule(mol, params) != 0:
            return TierResult(iso.canonical, None, "ETKDG could not embed")
        res = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=2000)
        energy = float(res[0][1])
    except Exception as e:  # noqa: BLE001 - RDKit raises RuntimeError on cages
        return TierResult(iso.canonical, None, f"MMFF failed: {type(e).__name__}: {e}")
    conf = mol.GetConformer()
    coords = [tuple(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())]
    return TierResult(
        iso.canonical,
        energy,
        payload={"coords": coords, "symbols": [a.GetSymbol() for a in mol.GetAtoms()]},
    )


def _embedded(iso: Isomer, context: dict):
    """Shared 3D geometry for tiers 3 and 4: reuse a cached one if present.

    Without the cache each tier would re-embed, and tiers 3 and 4 would then be
    scoring DIFFERENT geometries of the same candidate -- which makes their
    energies incomparable for no benefit.
    """
    cached = context.get("geometry", {}).get(iso.canonical)
    if cached:
        return cached["symbols"], cached["coords"]
    r = tier2_forcefield(iso, context)
    if not r.ok:
        return None, None
    return r.payload["symbols"], r.payload["coords"]


def tier1_dock(iso: Isomer, context: dict) -> TierResult:
    """AutoDock Vina pose search. `value` is the best Vina score (lower better).

    The score is an empirical ranking heuristic, NOT a binding free energy, and
    is used here only to order poses for the tiers above. Validated on this
    target by redocking 7LCJ to 0.95 A.

    **Spend the budget on SEEDS, not on exhaustiveness.** Measured on this
    target (RESULTS.md M11): across an 8x range of `exhaustiveness` the mean
    redock RMSD moved 0.097 A, which is SMALLER than the 0.131 A between-seed
    SEM -- and no seed improved monotonically with effort. What did move the
    number was the starting conformer. So `n_seeds` docks the ligand from
    several independent ETKDG embeddings and keeps the best-scoring pose, which
    buys real spread coverage where extra search effort bought noise.

    `n_seeds=1` reproduces the old single-embedding behaviour exactly.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    from tools.docking import dock_ligand

    mol0 = Chem.MolFromSmiles(iso.canonical)
    if mol0 is None:
        return TierResult(iso.canonical, None, "unparseable SMILES")
    # Meeko requires a single connected molecule. A transform can split one --
    # e.g. a ring contraction that severs the ring rather than shrinking it --
    # and the resulting salt/fragment pair is not a dockable ligand. Rejected
    # here with a readable reason rather than 300 lines of Meeko traceback.
    if len(Chem.GetMolFrags(mol0)) > 1:
        return TierResult(
            iso.canonical,
            None,
            f"not a single connected molecule "
            f"({len(Chem.GetMolFrags(mol0))} fragments)",
        )

    base_seed = context.get("seed", 0xF00D)
    n_seeds = max(1, int(context.get("n_seeds", 1)))
    best_overall = None
    prep_errors: list[str] = []

    for k in range(n_seeds):
        seed = base_seed + k
        mol = Chem.AddHs(Chem.Mol(mol0))
        params = AllChem.ETKDGv3()
        params.randomSeed = seed
        params.useSmallRingTorsions = True
        try:
            if AllChem.EmbedMolecule(mol, params) != 0:
                prep_errors.append(f"seed {seed}: ETKDG could not embed")
                continue
            AllChem.MMFFOptimizeMolecule(mol)
        except Exception as e:  # noqa: BLE001
            prep_errors.append(f"seed {seed}: {type(e).__name__}: {e}")
            continue

        try:
            res = dock_ligand(
                mol,
                context["receptor_pdbqt"],
                context["box_center"],
                context.get("box_size", (24.0, 24.0, 24.0)),
                # DEFAULT 4, not 16 or 32. MEASURED (RESULTS.md M11): across
                # an 8x range of exhaustiveness the mean redock RMSD moved
                # 0.097 A -- SMALLER than the 0.131 A between-seed SEM -- and
                # ex=32 had the WORST mean of the four levels tried. So effort
                # above 4 costs 6.8x for no accuracy. Spend it on `n_seeds`
                # instead, which is what actually moved the number.
                exhaustiveness=context.get("exhaustiveness", 4),
                n_poses=context.get("n_poses", 10),
                seed=seed,
                # Default 1, NOT Vina's 0: this tier runs inside a
                # funnel that fans out across ligands, and two
                # levels of parallelism oversubscribe the box.
                # See dock_ligand's `cpu` docs.
                cpu=context.get("vina_cpu", 1),
            )
        except ImportError as e:
            # vina/meeko are an optional extra (`pip install ferric[docking]`),
            # because they are not installable on every Python the wheel
            # targets. A tier that cannot run must say WHY it cannot run --
            # reporting this as a docking failure would send the reader hunting
            # for a receptor or a bad ligand when the real answer is an
            # uninstalled package.
            return TierResult(
                iso.canonical,
                None,
                f"docking unavailable ({e}); "
                f"install the 'docking' extra to enable tier 1",
            )
        if not res.ok:
            prep_errors.append(f"seed {seed}: {res.error}")
            continue
        cand = res.best
        if best_overall is None or cand.vina_score < best_overall[0].vina_score:
            best_overall = (cand, len(res.poses), seed)

    if best_overall is None:
        return TierResult(
            iso.canonical, None, "; ".join(prep_errors) or "docking produced no pose"
        )
    best, n_poses, winning_seed = best_overall
    return TierResult(
        iso.canonical,
        best.vina_score,
        payload={
            "symbols": best.symbols,
            "coords": best.coords_angstrom,
            "n_poses": n_poses,
            "n_seeds": n_seeds,
            "winning_seed": winning_seed,
        },
    )


def tier3_gfn2(iso: Isomer, context: dict) -> TierResult:
    """GFN2-xTB single point, optionally in a pocket point-charge field."""
    from tools.campaign.xtb_engine import singlepoint

    symbols, coords = _embedded(iso, context)
    if symbols is None:
        return TierResult(iso.canonical, None, "no geometry for GFN2")
    run = singlepoint(
        symbols,
        coords,
        charge=iso.net_charge,
        point_charges=context.get("point_charges"),
    )
    if not run.ok:
        return TierResult(iso.canonical, None, run.error)
    return TierResult(
        iso.canonical, run.energy, payload={"symbols": symbols, "coords": coords}
    )


def tier4_dft(iso: Isomer, context: dict) -> TierResult:
    """ferric Kohn-Sham DFT -- the most expensive tier.

    Measured 96.1 s at 32 atoms (def2-SVP/PBE; re-measured 99.0 s on
    2026-09-02), so this must only ever see the handful the tiers above left.

    Cost here is driven by ATOM COUNT, not basis size: ferric's KS grid is
    75x110 per atom, so a smaller basis does NOT make a big molecule cheap.
    See `tools/pipeline/cost.py`.

    **Dispersion.** This tier is labelled "DFT + dispersion" in
    `tools/campaign/hierarchy.py`, and as of the `ferric-d3` crate that label is
    true: `dispersion="d3bj"` is the DEFAULT here, adding Grimme's D3(BJ)
    two-body correction to the SCF energy. Dispersion is the dominant attractive
    term in ligand binding, so a bare semilocal DFT energy is not comparable
    between conformers or substituents.

    Pass `context["dispersion"] = None` to get the uncorrected SCF energy back.
    The result's payload carries `e_scf` and `e_dispersion` separately;
    `e_dispersion is None` means UNEVALUATED, never "zero dispersion".

    Two scope limits that this does NOT fix:
      - The Axilrod-Teller-Muto three-body term is not implemented (measured at
        0.1% of the two-body energy for benzene, and RISING with system size --
        see `ferric-d3`'s crate docs).
      - QM/MM dispersion is NOT covered. D3 is a QM-atom-pairwise correction, so
        dispersion between the QM region and MM point charges is still absent;
        that needs Lennard-Jones terms across the boundary, not this.

    `mem_budget_gb` is forwarded to `FERRIC_MEM_BUDGET_GB` because ferric's
    default is `0.8 x` *live* MemAvailable. That makes the internal
    Full-vs-Batched AO-cache decision depend on whatever else happens to be
    running on the box -- the same candidate can take the fast path or the
    batching path between two runs of the SAME pipeline. Pinning it keeps the
    tier reproducible; leaving it None preserves ferric's auto-detect.
    """
    import os

    import ferric

    budget = context.get("mem_budget_gb")
    prior = os.environ.get("FERRIC_MEM_BUDGET_GB")
    if budget is not None:
        os.environ["FERRIC_MEM_BUDGET_GB"] = str(budget)
    try:
        return _tier4_dft_inner(iso, context, ferric)
    finally:
        if budget is not None:
            if prior is None:
                os.environ.pop("FERRIC_MEM_BUDGET_GB", None)
            else:
                os.environ["FERRIC_MEM_BUDGET_GB"] = prior


def _tier4_dft_inner(iso: Isomer, context: dict, ferric) -> TierResult:
    symbols, coords = _embedded(iso, context)
    if symbols is None:
        return TierResult(iso.canonical, None, "no geometry for DFT")
    xyz = [str(len(symbols)), "tier4"]
    for s, (x, y, z) in zip(symbols, coords):
        xyz.append(f"{s} {x:.8f} {y:.8f} {z:.8f}")
    try:
        mol = ferric.Molecule.from_xyz_string("\n".join(xyz) + "\n", iso.net_charge, 1)
        bs = ferric.BasisSet.bundled(context.get("basis", "def2-svp"))
        res = ferric.run_dft(
            mol,
            bs,
            functional=context.get("functional", "PBE"),
            point_charges=context.get("point_charges"),
            dispersion=context.get("dispersion", "d3bj"),
        )
    except Exception as e:  # noqa: BLE001
        return TierResult(iso.canonical, None, f"DFT failed: {type(e).__name__}: {e}")
    if not res.converged:
        return TierResult(iso.canonical, None, "DFT did not converge")
    return TierResult(
        iso.canonical,
        res.total_energy,
        payload={
            "converged": True,
            "symbols": symbols,
            "coords": coords,
            # Recorded separately so a downstream consumer can tell a
            # dispersion-corrected energy from a bare SCF one. `None` here
            # means dispersion was NOT computed (the caller passed
            # dispersion=None), not that it was computed and found to be zero.
            "e_scf": res.e_scf,
            "e_dispersion": res.e_dispersion,
            # D3(BJ) IS QM-ATOM-PAIRWISE. It sums over the atoms in the
            # molecule; MM point charges are not atoms and contribute nothing.
            #
            # So with `point_charges` set, the dispersion above covers the QM
            # region INTERNALLY and the QM-to-MM interaction carries NONE. That
            # is a partially corrected energy, and the missing part is exactly
            # the one a binding or pocket question cares about -- dispersion is
            # the dominant attractive term across a ligand/pocket boundary.
            #
            # Flagged rather than refused: the QM-internal correction is still
            # correct and still worth having for conformer comparisons within
            # one ligand. What must not happen is a caller reading this as a
            # fully dispersion-corrected embedded energy.
            #
            # WHY THE GAP IS OPEN -- corrected 2026-09-19. This comment
            # previously said "closing the gap needs LJ terms on the MM sites
            # (`ferric-mm`)", declaring them absent. THAT WAS FALSE:
            # `ferric_mm::qm_mm_lj_energy_gradient` has existed since
            # 2026-08-27 (4a930a0f), does the full N x M 12-6 sum with analytic
            # gradients under Lorentz-Berthelot mixing, and is already called
            # by `ferric_scf::qmmm::qmmm_mm_terms`. The comment was written
            # three weeks AFTER the code it declared missing, and it generated
            # a ticket to rebuild what the repo already had.
            #
            # The real reason is an UNUSED CODE PATH, not absent arithmetic:
            # this tier calls `ferric.run_dft(..., point_charges=...)`, and
            # `run_dft` takes no topology or LJ argument -- the LJ machinery
            # lives on the separate `run_qmmm`/`QmmmSystem` path.
            #
            # And closing it end to end needs a third thing neither supplies:
            # `ferric-mm` ASSIGNS NO PARAMETERS. It is arithmetic only,
            # caller-supplies-everything. Tier 4's context carries point
            # charges but no sigma/epsilon, and nothing in this pipeline types
            # atoms against a force field. That PARAMETER-ASSIGNMENT problem,
            # not the 12-6 sum, is the actual remaining work.
            "dispersion_covers_qm_only": (
                res.e_dispersion is not None and bool(context.get("point_charges"))
            ),
        },
    )
