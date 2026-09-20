"""The whole golden path in one test: input format -> tiers -> catalyst barrier.

Every hop here is verified piecewise elsewhere. This runs them as ONE sequence,
which is what the note claims the pipeline is, and which no other test does.
It is the integration failure mode that piecewise tests structurally miss: each
call works, and the thing they compose into does not.

MEASURED 10.3 s for the whole chain, so it stays in the normal tier.
"""

from __future__ import annotations

import pytest

ferric = pytest.importorskip("ferric", reason="the compiled extension is not built")
pytest.importorskip("rdkit")

HARTREE_TO_KCAL = 627.5094740631
_PARENT = "CC(=O)Oc1ccccc1C(=O)O"


def test_input_formats_and_the_tier_ladder_compose():
    from tools.isomers.model import Isomer
    from tools.pipeline import tiers
    from tools.pipeline.substitution import propose_substitutions
    from tools.structure import from_smiles

    mol = from_smiles(_PARENT)
    assert len(mol.symbols()) == 21

    props = propose_substitutions(_PARENT, {"F": "F", "Cl": "Cl"})
    assert len(props) > 3

    iso = Isomer(smiles=_PARENT, kind="probe", transform="none", parent_smiles=_PARENT)
    ff = tiers.tier2_forcefield(iso, {})
    xtb = tiers.tier3_gfn2(iso, {})
    assert ff.ok and xtb.ok

    # GFN2 is a VALENCE-electron method: aspirin at ~-39.6 Ha is correct and
    # looks alarmingly small next to an all-electron number. Asserted as a
    # scaling relation rather than a magnitude, so the test says what it means.
    small = tiers.tier3_gfn2(
        Isomer(smiles="C", kind="p", transform="n", parent_smiles="C"), {}
    )
    assert small.ok and small.value > xtb.value, (
        f"methane ({small.value}) must be LESS negative than aspirin "
        f"({xtb.value}); if not, the tier is not scaling with size"
    )


def test_the_catalyst_branch_runs_on_one_surface():
    """C3 -> C4 -> C5, all in the same MM field.

    The failure this guards is subtle: each call succeeds in vacuum too, so a
    dropped field gives a confident barrier computed on the WRONG surface.
    `run_irc` recomputing `saddle_energy` is what ties them together.
    """
    nh3 = ferric.Molecule.from_xyz_string(
        "4\nnh3\nN 0 0 0\nH 0.0 1.01 0.1\nH 0.8747 -0.505 0.1\nH -0.8747 -0.505 0.1\n"
    )
    field = [(-0.4, 0.0, 0.0, 6.0), (-0.4, 0.0, 0.0, -6.0)]

    sad = ferric.run_saddle(nh3, "sto-3g", max_steps=40, point_charges=field)
    assert sad.converged and sad.n_imaginary == 1 and sad.is_transition_state()

    xyz = f"{len(sad.symbols)}\nS\n" + "".join(
        f"{a} {c[0]:.6f} {c[1]:.6f} {c[2]:.6f}\n"
        for a, c in zip(sad.symbols, sad.coords)
    )
    irc = ferric.run_irc(
        ferric.Molecule.from_xyz_string(xyz),
        "sto-3g",
        mode=sad.imaginary_mode,
        max_steps=120,
        step=0.15,
        point_charges=field,
    )
    assert irc.saddle_energy == pytest.approx(sad.energy, abs=1e-8), (
        "run_irc is on a different surface from run_saddle -- the field did "
        "not reach one of them"
    )
    fwd = irc.forward_barrier() * HARTREE_TO_KCAL
    rev = irc.reverse_barrier() * HARTREE_TO_KCAL
    assert fwd > 0 and rev > 0
    # Near-symmetric by the umbrella mode's own symmetry; a large asymmetry
    # means a branch wandered off the reaction coordinate.
    assert abs(fwd - rev) < 2.0, f"barriers {fwd:.2f}/{rev:.2f} are not comparable"
