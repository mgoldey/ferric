"""`FrequencyResult.normal_modes` — the mode VECTORS, not just the wavenumbers.

Why this exists: `n_imaginary() == 1` is NECESSARY but not SUFFICIENT for a
transition state. A methyl rotor also gives exactly one imaginary frequency. To
confirm a saddle is the one you want, you have to look at what the imaginary
mode DISPLACES and check it lies along the reaction coordinate.

Before this binding, Python could count imaginary modes but not inspect them,
so a Python-driven catalyst workflow could not complete TS verification at all.
The vectors existed in Rust (`FrequencyResult::normal_modes`, an `Array2<f64>`)
and simply were not exposed.
"""

from __future__ import annotations

import pytest

ferric = pytest.importorskip("ferric", reason="needs the compiled extension")


def _h2():
    return ferric.Molecule.from_xyz("testdata/molecules/h2.xyz", 0, 1)


def _water():
    return ferric.Molecule.from_xyz("testdata/molecules/water.xyz", 0, 1)


def test_normal_modes_shape_matches_frequencies_and_atoms():
    """One row per frequency, 3N entries per row.

    Pinning BOTH dimensions is deliberate: a transposed array would still have
    the right element count, and a workflow indexing `modes[i]` for mode `i`
    would then silently read a coordinate instead.
    """
    mol = _water()
    r = ferric.run_frequencies(mol, "sto-3g")
    modes = r.normal_modes
    assert len(modes) == len(r.frequencies), (
        f"{len(modes)} mode vectors for {len(r.frequencies)} frequencies"
    )
    for i, row in enumerate(modes):
        assert len(row) == 3 * mol.natoms(), (
            f"mode {i} has {len(row)} entries, expected 3N = {3 * mol.natoms()}"
        )


def test_h2_stretch_is_along_the_bond():
    """H2 has ONE vibration and it must displace the atoms along z.

    H2 in this fixture lies on the z axis, so the single stretch has to move
    both atoms in z and neither in x or y. That is a real physical assertion,
    not a shape check -- it fails if the vectors are garbage, transposed, or
    left mass-weighted.
    """
    r = ferric.run_frequencies(_h2(), "sto-3g")
    assert len(r.frequencies) == 1, (
        f"H2 should have 1 vibration, got {len(r.frequencies)}"
    )
    mode = r.normal_modes[0]
    ax, ay, az = mode[0], mode[1], mode[2]
    bx, by, bz = mode[3], mode[4], mode[5]
    assert abs(az) > 1e-3 and abs(bz) > 1e-3, f"stretch does not move z: {mode}"
    for lateral in (ax, ay, bx, by):
        assert abs(lateral) < 1e-6 * max(abs(az), abs(bz)) + 1e-9, (
            f"a z-axis stretch must not displace x or y; got {mode}"
        )
    # Opposite directions: the two atoms move toward or away from each other.
    assert az * bz < 0, f"a stretch must move the two atoms oppositely: {az}, {bz}"


def test_modes_are_finite():
    """A NaN here would propagate silently into any downstream projection."""
    r = ferric.run_frequencies(_water(), "sto-3g")
    for i, row in enumerate(r.normal_modes):
        for j, v in enumerate(row):
            assert v == v and abs(v) != float("inf"), (
                f"mode {i}[{j}] is not finite: {v}"
            )


def test_modes_are_not_mass_weighted():
    """The doc says Cartesian displacements, i.e. the mass-weighted
    eigenvectors divided back through by sqrt(m).

    The discriminating factor is sqrt(m_O/m_H) = 3.98, NOT the mass ratio of
    16 -- the back-transform divides by sqrt(m), not by m. That distinction is
    why the threshold below is 6.0 and not 2.0.

    MEASURED on water/STO-3G, max H/O amplitude ratio over all modes:
        mass-weighted (mutant):  2.86
        Cartesian (correct):    11.37
    A threshold of 2.0 -- which is what this test shipped with first -- does
    NOT separate them: the mutation that removed `* inv_sqrt_m[i]` from
    frequencies.rs:465 SURVIVED it. 6.0 sits cleanly between the two measured
    values and kills that mutant.
    """
    mol = _water()
    r = ferric.run_frequencies(mol, "sto-3g")
    syms = mol.symbols()
    o = syms.index("O")
    h = syms.index("H")
    ratios = []
    for row in r.normal_modes:
        o_amp = max(abs(row[3 * o + k]) for k in range(3))
        h_amp = max(abs(row[3 * h + k]) for k in range(3))
        if o_amp > 1e-12:
            ratios.append(h_amp / o_amp)
    assert ratios, "no usable modes"
    assert max(ratios) > 6.0, (
        "in Cartesian displacements the light H must out-move the heavy O in at "
        f"least one mode; max H/O amplitude ratio was {max(ratios):.2f} "
        f"(Cartesian measures ~11.4, mass-weighted ~2.9), which "
        "suggests the vectors are still mass-weighted"
    )
