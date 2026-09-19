"""Paired ddE: hold the scaffold pose FIXED and swap only the substituent.

## The problem this addresses

Five routes to a usable substituent ranking are closed (RESULTS.md M4-M14).
Every one of them computed

    ddE = mean(E_A over ensemble_A) - mean(E_B over ensemble_B)

where `ensemble_A` and `ensemble_B` were embedded INDEPENDENTLY. The per-pose
scatter is ~28.75 kcal/mol, so that difference carries `sd*sqrt(2) = 40.66`
kcal/mol, and averaging 100 poses only gets it to **4.07** -- against
substituent effects of 1-2 kcal/mol. Hence `noise_floor` greys out every cell
of the heatmap.

**That is an UNPAIRED design.** The dominant variance is pose-conformational:
it is a property of the scaffold sitting in the pocket, and a substitution
changes a handful of atoms while ~68 others stay where they were. Variance
common to both molecules cancels in a paired difference:

    var(ddE_paired) = 2*sd^2*(1 - rho)       vs      2*sd^2 unpaired

For rho = 0.9 that is a 3.2x reduction in sd; for rho = 0.99, 10x. This module
constructs the pairing so rho can be large: pose k of the analogue is BUILT FROM
pose k of the parent, sharing the scaffold coordinates exactly.

## What "paired" means here, precisely

Not "the same random seed". ETKDG with a shared seed on two different molecular
graphs produces UNCORRELATED conformers -- the seed indexes a random stream, not
a geometry, and the graphs differ. Pairing has to be geometric: take the parent
pose, keep the MCS scaffold atoms at exactly their parent coordinates, and place
only the substituent atoms.

## THE GUARD THAT MATTERS

Pairing changes the VARIANCE of an estimator, never its EXPECTATION. If the
paired mean differs from the unpaired mean by more than sampling error, the
construction is wrong -- it is not a variance reduction, it is a different
quantity. `paired_ddE` reports both, and `PairedResult.mean_shift_is_suspicious`
flags it.

## Scope

This module does the PAIRING and the statistics. It does not score: the caller
supplies an energy function. So it is testable without xtb, DFT, or a pocket,
and its exactness anchor runs in milliseconds.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Callable, Sequence

__all__ = [
    "PairedPose",
    "PairedResult",
    "pair_poses_by_scaffold",
    "paired_ddE",
    "relax_substituent",
]

Coords = Sequence[tuple[float, float, float]]


@dataclass
class PairedPose:
    """One pose of A and the corresponding pose of B, sharing a scaffold."""

    index: int
    symbols_a: list[str]
    coords_a: list[tuple[float, float, float]]
    symbols_b: list[str]
    coords_b: list[tuple[float, float, float]]
    #: Indices into A and B of the atoms held at identical coordinates.
    scaffold_map: list[tuple[int, int]]
    #: Largest deviation over the scaffold pairs, in Angstrom. This is the
    #: measurement that says whether the pairing is REAL. It must be ~0.
    scaffold_max_dev: float = 0.0
    error: str | None = None

    @property
    def usable(self) -> bool:
        return self.error is None


@dataclass
class PairedResult:
    """The paired and unpaired estimates side by side, so they can disagree."""

    n_pairs: int
    #: The paired per-pose differences. Their sd is the number the whole
    #: exercise is about.
    differences: list[float] = field(default_factory=list)
    ddE_paired: float = float("nan")
    sd_paired: float = float("nan")
    sem_paired: float = float("nan")
    #: What the SAME data give when the pairing is ignored -- the status quo.
    ddE_unpaired: float = float("nan")
    sd_unpaired: float = float("nan")
    sem_unpaired: float = float("nan")
    #: Pearson correlation between the paired energies. The mechanism.
    rho: float = float("nan")
    notes: list[str] = field(default_factory=list)

    @property
    def variance_reduction(self) -> float:
        """How many times smaller the paired SEM is. 1.0 = pairing bought nothing.

        `inf` when the paired SEM is exactly zero: the pairing removed ALL the
        variance, which is a result rather than a failure to compute one.
        """
        if not math.isfinite(self.sem_unpaired):
            return float("nan")
        if self.sem_paired == 0.0:
            # The paired differences are IDENTICAL, so the pairing removed all
            # of the variance. That is a real, and the best possible, outcome --
            # the self-anchor hits it exactly. Returning NaN would report it as
            # "could not be computed" and a caller filtering on isfinite would
            # silently drop the strongest result in the set.
            return float("inf") if self.sem_unpaired > 0 else float("nan")
        if self.sem_paired < 0 or not math.isfinite(self.sem_paired):
            return float("nan")
        return self.sem_unpaired / self.sem_paired

    #: ddE of the parent paired with ITSELF through the same construction, when
    #: the caller measured it. Not None is what makes `reembedding_bias` real.
    self_anchor_ddE: float | None = None

    @property
    def reembedding_bias(self) -> float | None:
        """The self-anchor offset: what this construction charges for NOTHING.

        THE GUARD THAT ACTUALLY CATCHES THIS CONSTRUCTION'S FAILURE. Pairing the
        parent with itself must give ddE == 0: same molecule, same scaffold, no
        substitution. It does NOT, because `pair_poses_by_scaffold` re-embeds
        the B side, and a constrained re-embedding lands above the relaxed
        geometry it came from. MEASURED at +13.8 kcal/mol on paracetamol-like
        with MMFF (probe of 2026-09-19).

        Subtracting it is NOT a fix: the penalty is substituent-DEPENDENT
        (F +9.5, Cl +17.2, N-methyl +31.1 above the self value), so it does not
        cancel, and it is an order of magnitude above the 1-2 kcal/mol effect.
        """
        if self.self_anchor_ddE is None or not math.isfinite(self.ddE_paired):
            return None
        return self.ddE_paired - self.self_anchor_ddE

    @property
    def mean_shift_is_suspicious(self) -> bool:
        """True when the self-anchor says this construction charges for nothing.

        AN EARLIER VERSION COMPARED `ddE_paired` AGAINST `ddE_unpaired` AND WAS
        INERT: both are `mean(E_B) - mean(E_A)` over the same data, so they are
        algebraically equal and the difference is always 0 (MEASURED: identical
        to 3 decimals in all four rows of the probe). Pairing changes the
        VARIANCE of the estimator, never its value -- so a mean shift between
        them is not merely unlikely, it is impossible, and the guard could never
        fire. The real bias is against ZERO via the self-anchor.
        """
        bias = self.reembedding_bias
        if bias is None:
            return False
        if not math.isfinite(self.sem_paired) or self.sem_paired <= 0:
            return abs(self.self_anchor_ddE or 0.0) > 0.0
        # The anchor itself carries sampling error; flag only a bias larger
        # than 2 sem of the estimate it would contaminate.
        return abs(self.self_anchor_ddE or 0.0) > 2.0 * self.sem_paired


def pair_poses_by_scaffold(
    symbols_a: Sequence[str],
    poses_a: Sequence[Coords],
    smiles_b: str,
    *,
    random_seed: int = 0xF00D,
    scaffold_tolerance: float = 0.5,
    relax: bool = True,
) -> list[PairedPose]:
    """Build one B pose per A pose, holding the shared scaffold fixed.

    `poses_a` are the parent's poses (Angstrom). For each, the MCS between A and
    B is computed and B is embedded with those atoms CONSTRAINED to the parent's
    coordinates, so the scaffold is shared by construction rather than by
    alignment.

    `scaffold_tolerance` is the drift a returned pose may carry, in Angstrom,
    judged AFTER relaxation. The measured relaxed range is 0.011-0.128 A, so
    the 0.5 default passes those comfortably while rejecting a pose whose
    scaffold has genuinely moved.

    `relax` defaults to TRUE and should stay that way. With it off the scaffold
    is pinned hard, which MEASURABLY fails the self-anchor by +13.8 kcal/mol
    (see `relax_substituent`). It is exposed only so that comparison can be
    reproduced.

    Returns one `PairedPose` per input pose, in order. A pose whose embedding
    fails comes back with `error` set rather than being dropped, so the caller
    sees which ones were lost -- a silently shorter list would bias the mean.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdFMCS

    out: list[PairedPose] = []

    mol_b0 = Chem.MolFromSmiles(smiles_b)
    if mol_b0 is None:
        return [
            PairedPose(
                i,
                list(symbols_a),
                [tuple(map(float, c)) for c in p],
                [],
                [],
                [],
                error=f"unparseable SMILES {smiles_b!r}",
            )
            for i, p in enumerate(poses_a)
        ]
    mol_b = Chem.AddHs(mol_b0)

    for i, pose in enumerate(poses_a):
        ca = [tuple(float(v) for v in c) for c in pose]
        if len(ca) != len(symbols_a):
            out.append(
                PairedPose(
                    i,
                    list(symbols_a),
                    ca,
                    [],
                    [],
                    [],
                    error=f"pose {i} has {len(ca)} coordinates for "
                    f"{len(symbols_a)} symbols",
                )
            )
            continue

        # Rebuild A as an RDKit molecule from symbols+coords so the MCS is
        # computed on real connectivity rather than on a SMILES we would have
        # to trust matches these coordinates.
        mol_a = _mol_from_symbols_coords(symbols_a, ca)
        if mol_a is None:
            out.append(
                PairedPose(
                    i,
                    list(symbols_a),
                    ca,
                    [],
                    [],
                    [],
                    error="could not perceive connectivity for pose A",
                )
            )
            continue

        # STAMP THE ORIGINAL INDEX ON EVERY ATOM BEFORE RemoveHs.
        #
        # `RemoveHs(sanitize=False)` RETAINS degree-zero, isotopic and hydride
        # hydrogens -- RDKit even warns "not removing hydrogen atom without
        # neighbors". `mol_a` is perceived from raw XYZ, so a stray atom easily
        # ends up degree zero and survives. The obvious mapping ("the nth heavy
        # atom of the stripped molecule is the nth non-H of the original") is
        # then WRONG for every atom after the retained hydrogen, and an
        # in-range shifted index pairs the wrong atoms SILENTLY.
        #
        # `scaffold_max_dev` CANNOT CATCH THAT: it measures the same `pairs`
        # used to build `coord_map`, so a wrong correspondence is pinned to the
        # parent's coordinates and then measures as ZERO drift. The check and
        # the construction share the error -- an anchor cannot see a defect it
        # is downstream of.
        for m_ in (mol_a, mol_b):
            for at in m_.GetAtoms():
                at.SetIntProp("_pairIdx", at.GetIdx())
        a_heavy = Chem.RemoveHs(mol_a, sanitize=False)
        b_heavy = Chem.RemoveHs(mol_b, sanitize=False)
        mcs = rdFMCS.FindMCS(
            [a_heavy, b_heavy],
            timeout=10,
            atomCompare=rdFMCS.AtomCompare.CompareElements,
            bondCompare=rdFMCS.BondCompare.CompareAny,
            ringMatchesRingOnly=False,
            completeRingsOnly=False,
            matchValences=False,
        )
        if mcs.canceled or mcs.numAtoms < 3:
            out.append(
                PairedPose(
                    i,
                    list(symbols_a),
                    ca,
                    [],
                    [],
                    [],
                    error=f"MCS found only {mcs.numAtoms} common atoms",
                )
            )
            continue

        patt = Chem.MolFromSmarts(mcs.smartsString)
        if patt is None:
            out.append(
                PairedPose(
                    i,
                    list(symbols_a),
                    ca,
                    [],
                    [],
                    [],
                    error="MCS produced unusable SMARTS",
                )
            )
            continue
        a_match = a_heavy.GetSubstructMatch(patt)
        b_match = b_heavy.GetSubstructMatch(patt)
        if not a_match or not b_match or len(a_match) != len(b_match):
            out.append(
                PairedPose(
                    i,
                    list(symbols_a),
                    ca,
                    [],
                    [],
                    [],
                    error="MCS did not map consistently onto both",
                )
            )
            continue

        # Stripped index -> ORIGINAL index, read off the stamp rather than
        # inferred from position. Correct whether or not RemoveHs kept an H.
        try:
            pairs = [
                (
                    a_heavy.GetAtomWithIdx(x).GetIntProp("_pairIdx"),
                    b_heavy.GetAtomWithIdx(y).GetIntProp("_pairIdx"),
                )
                for x, y in zip(a_match, b_match)
            ]
        except (KeyError, RuntimeError, IndexError):
            out.append(
                PairedPose(
                    i,
                    list(symbols_a),
                    ca,
                    [],
                    [],
                    [],
                    error="heavy-atom index map inconsistent with MCS",
                )
            )
            continue

        # INDEPENDENT of the index arithmetic above: a pair whose two atoms are
        # different ELEMENTS means the map is wrong, whatever the MCS thought,
        # because the pattern matched on element identity. This check does not
        # share the failure mode it guards, which is the point -- see
        # `scaffold_max_dev`, which does.
        sb_all = [a.GetSymbol() for a in mol_b.GetAtoms()]
        mism = [(ai, bj) for ai, bj in pairs if symbols_a[ai] != sb_all[bj]]
        if mism:
            out.append(
                PairedPose(
                    i,
                    list(symbols_a),
                    ca,
                    [],
                    [],
                    [],
                    error=(
                        f"scaffold map pairs different elements at {mism[:3]}; "
                        "the heavy-atom index map is wrong"
                    ),
                )
            )
            continue

        # THE PAIRING: constrain B's scaffold atoms to A's coordinates.
        from rdkit.Geometry import Point3D

        coord_map = {bj: Point3D(*ca[ai]) for ai, bj in pairs}
        mb = Chem.Mol(mol_b)
        params = AllChem.ETKDGv3()
        params.randomSeed = random_seed + i
        try:
            cid = AllChem.EmbedMolecule(
                mb, coordMap=coord_map, useRandomCoords=True, randomSeed=random_seed + i
            )
        except Exception as exc:  # noqa: BLE001 -- one failure must not abort
            out.append(
                PairedPose(
                    i,
                    list(symbols_a),
                    ca,
                    [],
                    [],
                    [],
                    error=f"constrained embed raised {type(exc).__name__}: {exc}",
                )
            )
            continue
        if cid < 0:
            out.append(
                PairedPose(
                    i,
                    list(symbols_a),
                    ca,
                    [],
                    [],
                    [],
                    error="constrained embedding failed",
                )
            )
            continue

        conf = mb.GetConformer(cid)
        cb = [
            tuple(float(v) for v in conf.GetAtomPosition(j))
            for j in range(mb.GetNumAtoms())
        ]
        sb = [a.GetSymbol() for a in mb.GetAtoms()]

        # MEASURE the pairing rather than assuming it. RDKit treats coordMap as
        # a restraint, not a hard constraint, so the scaffold CAN drift -- and a
        # drifted scaffold is exactly the fictitious pairing this module warns
        # about. Report the deviation; let the caller decide.
        dev = max((math.dist(ca[ai], cb[bj]) for ai, bj in pairs), default=0.0)
        pp = PairedPose(i, list(symbols_a), ca, sb, cb, pairs, scaffold_max_dev=dev)
        out.append(pp)

    # THE DRIFT GUARD RUNS AFTER RELAXATION, NOT BEFORE.
    #
    # It used to read `dev > scaffold_tolerance and dev > 0.5`, which is just
    # `dev > max(scaffold_tolerance, 0.5)`: with the old 1e-6 default the
    # parameter could not lower the threshold and so did nothing at all. Worse,
    # it ran BEFORE `relax_substituent`, where drift is zero by construction
    # (coordMap pins the scaffold) -- the guard was checking the one stage that
    # cannot fail and skipping the one that can. The default is now the real
    # threshold, and drift is judged on the poses actually returned.
    out = relax_substituent(out) if relax else out
    for pp in out:
        if pp.usable and pp.scaffold_max_dev > scaffold_tolerance:
            pp.error = (
                f"scaffold drifted {pp.scaffold_max_dev:.3f} A from the parent "
                f"pose (tolerance {scaffold_tolerance:g}); the pairing is not "
                "real, and a paired difference over it is a different "
                "quantity rather than a quieter one"
            )
    return out


def _mol_from_symbols_coords(symbols: Sequence[str], coords: Coords):
    """An RDKit molecule with perceived connectivity from raw symbols+coords."""
    from rdkit import Chem

    xyz = f"{len(symbols)}\n\n" + "".join(
        f"{s} {c[0]:.8f} {c[1]:.8f} {c[2]:.8f}\n" for s, c in zip(symbols, coords)
    )
    try:
        mol = Chem.MolFromXYZBlock(xyz)
        if mol is None:
            return None
        from rdkit.Chem import rdDetermineBonds

        rdDetermineBonds.DetermineConnectivity(mol)
        return mol
    except Exception:  # noqa: BLE001 -- perception is best-effort
        return None


def relax_substituent(
    pairs: Sequence[PairedPose],
    *,
    restraint_force: float = 100.0,
    max_iterations: int = 500,
) -> list[PairedPose]:
    """MMFF-minimise the B side with the scaffold spring-restrained.

    **This is not optional polish -- it is what makes the construction pass its
    own exactness anchor.** A hard `coordMap` pin forces the substituent into
    whatever room the parent pose left, and MEASURED on paracetamol-like/MMFF
    that charges +13.841 kcal/mol for pairing the parent WITH ITSELF. After
    restrained relaxation the self-anchor is +0.004 while the scaffold still
    holds to 0.011-0.128 A. See wiki/paired-ddE-substitution-2026-09-19.md.

    The relaxation does NOT collapse the ensemble (the failure mode that would
    make sd fall for the wrong reason): pairwise heavy-atom RMSD is retained at
    100.0-100.3%, so the poses stay as distinct as they started. Only the
    embedding strain is removed.

    Poses that cannot be perceived or typed come back UNCHANGED rather than
    dropped, so the caller still sees them and the ensemble stays the same size.
    """
    import copy

    from rdkit import Chem
    from rdkit.Chem import AllChem, rdDetermineBonds

    out: list[PairedPose] = []
    for p in pairs:
        if not p.usable or not p.symbols_b:
            out.append(p)
            continue
        xyz = f"{len(p.symbols_b)}\n\n" + "".join(
            f"{s} {c[0]:.8f} {c[1]:.8f} {c[2]:.8f}\n"
            for s, c in zip(p.symbols_b, p.coords_b)
        )
        mol = Chem.MolFromXYZBlock(xyz)
        if mol is None:
            out.append(p)
            continue
        try:
            rdDetermineBonds.DetermineBonds(mol, charge=0)
            props = AllChem.MMFFGetMoleculeProperties(mol)
            if props is None:
                out.append(p)
                continue
            ff = AllChem.MMFFGetMoleculeForceField(mol, props)
            for _, bj in p.scaffold_map:
                ff.MMFFAddPositionConstraint(bj, 0.0, restraint_force)
            ff.Minimize(maxIts=max_iterations)
        except Exception:  # noqa: BLE001 -- an untypeable analogue must not abort
            out.append(p)
            continue
        conf = mol.GetConformer()
        q = copy.deepcopy(p)
        q.coords_b = [
            tuple(float(v) for v in conf.GetAtomPosition(j))
            for j in range(mol.GetNumAtoms())
        ]
        q.scaffold_max_dev = max(
            (math.dist(p.coords_a[ai], q.coords_b[bj]) for ai, bj in p.scaffold_map),
            default=0.0,
        )
        out.append(q)
    return out


def paired_ddE(
    pairs: Sequence[PairedPose],
    energy: Callable[[Sequence[str], Coords], float | None],
    *,
    self_anchor_ddE: float | None = None,
) -> PairedResult:
    """Compute ddE both ways over the same poses, so they can be compared.

    `energy` returns a number per (symbols, coords), or None for a pose it could
    not score. A pair is used only when BOTH sides score -- dropping one side
    would silently unbalance the means.
    """
    usable = [p for p in pairs if p.usable]
    ea: list[float] = []
    eb: list[float] = []
    dropped = 0
    for p in usable:
        va = energy(p.symbols_a, p.coords_a)
        vb = energy(p.symbols_b, p.coords_b)
        if va is None or vb is None or not math.isfinite(va) or not math.isfinite(vb):
            dropped += 1
            continue
        ea.append(float(va))
        eb.append(float(vb))

    res = PairedResult(n_pairs=len(ea))
    if dropped:
        res.notes.append(f"{dropped} pair(s) dropped: at least one side did not score")
    failed = len(pairs) - len(usable)
    if failed:
        res.notes.append(f"{failed} pose(s) could not be paired (see PairedPose.error)")

    if len(ea) < 2:
        res.notes.append(
            f"only {len(ea)} usable pair(s); a variance needs at least 2. "
            "No estimate is reported rather than a zero-width one."
        )
        return res

    res.differences = [b - a for a, b in zip(ea, eb)]
    res.ddE_paired = statistics.fmean(res.differences)
    res.sd_paired = statistics.stdev(res.differences)
    res.sem_paired = res.sd_paired / math.sqrt(len(res.differences))

    # The UNPAIRED estimate from the SAME numbers: difference of means, with the
    # variance the independent-ensembles protocol would carry.
    ma, mb_ = statistics.fmean(ea), statistics.fmean(eb)
    sa, sb = statistics.stdev(ea), statistics.stdev(eb)
    res.ddE_unpaired = mb_ - ma
    res.sd_unpaired = math.hypot(sa, sb)
    res.sem_unpaired = res.sd_unpaired / math.sqrt(len(ea))

    # rho is the MECHANISM: the variance reduction is 1/sqrt(1-rho) when the two
    # sds are equal, so reporting it says WHY the pairing did or did not help.
    if sa > 0 and sb > 0:
        cov = (
            statistics.fmean((a - ma) * (b - mb_) for a, b in zip(ea, eb))
            * len(ea)
            / (len(ea) - 1)
        )
        res.rho = max(-1.0, min(1.0, cov / (sa * sb)))
    else:
        res.notes.append(
            "one side has zero variance across poses, so rho is undefined "
            "(this is the self-pairing anchor, or a constant energy function)"
        )

    if self_anchor_ddE is not None:
        res.self_anchor_ddE = float(self_anchor_ddE)
        if res.mean_shift_is_suspicious:
            res.notes.append(
                f"SELF-ANCHOR IS NONZERO ({self_anchor_ddE:.3f}): pairing the "
                "parent with ITSELF through this construction charges an energy "
                "for no substitution at all. The construction re-embeds the B "
                "side, and a constrained re-embedding sits above the relaxed "
                "geometry. Subtracting it does not fix this -- the penalty is "
                "substituent-dependent, so it does not cancel in a difference."
            )
    else:
        res.notes.append(
            "no self-anchor supplied, so the re-embedding bias is UNMEASURED. "
            "Pass self_anchor_ddE=paired_ddE(pair_poses_by_scaffold(.., parent "
            "SMILES), energy).ddE_paired -- on the one system measured it was "
            "+13.8 kcal/mol, which is larger than any substituent effect."
        )
    return res
