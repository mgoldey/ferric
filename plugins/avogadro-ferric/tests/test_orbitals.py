"""Orbital export: what Avogadro would draw equals what ferric computed.

``orbitals.avogadro_mo_values`` is a line-by-line Python port of avogadrolibs
2.1.0's GaussianSet normalisation + GaussianSetTools point evaluation. The MO
values it produces from the exported cjson must equal
``ferric.eval_mo_on_points`` at points the fit never saw, for cartesian d
(6-31G*), spherical d/f (cc-pVTZ) and a Kohn-Sham result (def2-SVP / PBE).
"""

from __future__ import annotations

import numpy as np
import pytest
from avoferric_testlib import WATER, make_cjson, menu_input, run_plugin

import ferric

pytestmark = pytest.mark.skipif(
    not hasattr(ferric.BasisSet, "shells")
    or not hasattr(ferric.DftResult, "mo_coefficients"),
    reason="needs BasisSet.shells / DftResult.mo_coefficients (ferric 0.1.0rc11 or newer)",
)

XYZ = "3\n\n" + "".join(f"{s} {a[1]} {a[2]} {a[3]}\n" for s, a in zip("OHH", WATER))


@pytest.mark.parametrize(
    "basis,method", [("6-31g*", "HF"), ("cc-pvtz", "HF"), ("def2-svp", "PBE")]
)
def test_avogadro_evaluation_of_exported_orbitals_matches_ferric(basis, method):
    from avogadro_ferric import orbitals

    mol = ferric.Molecule.from_xyz_string(XYZ)
    bs = ferric.BasisSet.bundled(basis)
    res = (
        ferric.run_rhf(mol, bs)
        if method == "HF"
        else ferric.run_dft(mol, bs, functional=method)
    )
    data = orbitals.orbitals_cjson(ferric, mol, bs, res, [8, 1, 1], 10)
    cj = make_cjson(WATER)
    cj.update(data)
    rng = np.random.default_rng(7)
    pts = rng.normal(size=(300, 3)) * 1.6  # Bohr, around the molecule
    ours = orbitals.avogadro_mo_values(cj, pts)
    nmo = ours.shape[1]
    ref = np.asarray(ferric.eval_mo_on_points(mol, bs, res, list(range(nmo)), pts))
    assert np.abs(ours - ref).max() < 1e-10 * np.abs(ref).max()
    occ = data["orbitals"]["occupations"]
    assert sum(occ) == 10 and len(occ) == nmo
    assert len(data["orbitals"]["moCoefficients"]) == nmo * nmo
    # energies in eV, ascending, HOMO negative
    eps = np.asarray(data["orbitals"]["energies"])
    assert np.all(np.diff(eps) >= -1e-9) and eps[4] < 0
    assert eps[0] == pytest.approx(res.orbital_energies()[0] * 27.21138505, rel=1e-12)


def test_measured_maps_are_the_known_conventions():
    """Pin what the per-shell fit finds, so a change on either side is visible.

    libint2 spherical order m = -l..l vs Avogadro's 0,+1,-1,+2,-2(,+3,-3): a
    signed permutation, all signs +. libint2 cartesian d (xx,xy,xz,yy,yz,zz,
    every component with the x^2 normalisation) vs Avogadro's unit-normalised
    xx,yy,zz,xy,xz,yz: a permutation with 1/sqrt(3) on the mixed components.
    """
    from avogadro_ferric import orbitals

    mol = ferric.Molecule.from_xyz_string(XYZ)
    found = {}
    for basis in ["6-31g*", "cc-pvtz"]:
        bs = ferric.BasisSet.bundled(basis)
        basis_dict, maps = orbitals.basis_and_maps(ferric, mol, bs, [8, 1, 1])
        for st, (_, m) in zip(basis_dict["shellTypes"], maps):
            found.setdefault(st, m)
    s3 = 1.0 / np.sqrt(3.0)
    expected_d6 = np.zeros((6, 6))
    # rows: Avogadro xx,yy,zz,xy,xz,yz ; cols: ferric xx,xy,xz,yy,yz,zz
    for avo_row, ferric_col, w in [
        (0, 0, 1),
        (1, 3, 1),
        (2, 5, 1),
        (3, 1, s3),
        (4, 2, s3),
        (5, 4, s3),
    ]:
        expected_d6[avo_row, ferric_col] = w
    np.testing.assert_allclose(found[2], expected_d6, atol=1e-9)
    # d5: Avogadro (0,+1,-1,+2,-2) <- ferric (-2,-1,0,+1,+2)
    perm5 = [2, 3, 1, 4, 0]
    np.testing.assert_allclose(found[-2], np.eye(5)[perm5], atol=1e-9)
    # f7: Avogadro (0,+1,-1,+2,-2,+3,-3) <- ferric (-3..+3)
    perm7 = [3, 4, 2, 5, 1, 6, 0]
    np.testing.assert_allclose(found[-3], np.eye(7)[perm7], atol=1e-9)
    np.testing.assert_allclose(found[1], np.eye(3), atol=1e-9)
    np.testing.assert_allclose(found[0], np.eye(1), atol=1e-9)


def test_a_wrong_evaluator_is_refused(monkeypatch):
    """If Avogadro's functions cannot represent ferric's, export must fail."""
    from avogadro_ferric import orbitals

    real = orbitals._angular

    def broken(shell_type, d):
        a = real(shell_type, d)
        if shell_type == -2:  # pre-2.1 Avogadro's d0 = z^2 - r^2
            a = a.copy()
            a[:, 0] = d[:, 2] ** 2 - np.einsum("ij,ij->i", d, d)
        return a

    monkeypatch.setattr(orbitals, "_angular", broken)
    mol = ferric.Molecule.from_xyz_string(XYZ)
    with pytest.raises(RuntimeError, match="does not map"):
        orbitals.basis_and_maps(
            ferric, mol, ferric.BasisSet.bundled("cc-pvdz"), [8, 1, 1]
        )


def test_orbitals_menu_command(plugin_env):
    env = plugin_env(method="HF", basis="6-31g*")
    out, _ = run_plugin(env, ["orbitals"], menu_input(WATER))
    cj = out["cjson"]
    assert cj["basisSet"]["shellTypes"].count(2) == 1  # one cartesian d on O
    assert cj["orbitals"]["electronCount"] == 10
    assert "HOMO" in out["message"]
