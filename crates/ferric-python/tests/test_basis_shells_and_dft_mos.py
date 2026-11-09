"""`BasisSet.shells(z)` and `DftResult.mo_coefficients()` / `orbital_energies()`.

Both exist so an external consumer (the Avogadro plugin in
plugins/avogadro-ferric) can write ferric's basis and orbitals in another
program's format. The tests pin that the data is the data the AO basis is
built from: shell order and sizes against `shell_info`, s-shell values
against `eval_ao_on_points`, and the KS orbitals against `eval_mo_on_points`
and the density.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

ferric = pytest.importorskip("ferric", reason="needs the compiled extension")

WATER = "3\n\nO 0 0 0.117\nH 0 0.757 -0.469\nH 0 -0.757 -0.469\n"


def _mol():
    return ferric.Molecule.from_xyz_string(WATER)


@pytest.mark.parametrize("name", ["sto-3g", "6-31g*", "cc-pvtz"])
def test_shells_follow_the_ao_basis_order(name):
    mol = _mol()
    bs = ferric.BasisSet.bundled(name)
    centers, offsets, dims = ferric.shell_info(mol, bs)
    listed = bs.shells(8) + bs.shells(1) + bs.shells(1)
    assert len(listed) == len(dims)
    for (ang, pure, exps, coefs), dim in zip(listed, dims):
        nfun = 2 * ang + 1 if pure else (ang + 1) * (ang + 2) // 2
        assert nfun == dim
        assert len(exps) == len(coefs) > 0
        assert all(e > 0 for e in exps)


def test_s_shell_values_are_normalised_primitive_contractions():
    """chi_s(r) = sum_k c_k (2 a_k / pi)^(3/4) exp(-a_k r^2): coefficients of
    NORMALISED primitives, as the docstring states."""
    mol = _mol()
    bs = ferric.BasisSet.bundled("cc-pvdz")
    centers, offsets, _ = ferric.shell_info(mol, bs)
    rng = np.random.default_rng(3)
    pts = centers[0] + rng.normal(size=(40, 3)) * 0.8
    ao = np.asarray(ferric.eval_ao_on_points(mol, bs, pts))
    r2 = ((pts - centers[0]) ** 2).sum(axis=1)
    checked = 0
    for si, (ang, pure, exps, coefs) in enumerate(bs.shells(8)):
        if ang != 0:
            continue
        ref = sum(
            c * (2 * a / math.pi) ** 0.75 * np.exp(-a * r2) for a, c in zip(exps, coefs)
        )
        np.testing.assert_allclose(ao[:, offsets[si]], ref, rtol=1e-10, atol=1e-12)
        checked += 1
    assert checked >= 3


def test_shells_of_an_absent_element_raise():
    with pytest.raises(ValueError):
        ferric.BasisSet.bundled("sto-3g").shells(118)


def test_dft_orbitals_reproduce_density_and_mo_values():
    mol = _mol()
    bs = ferric.BasisSet.bundled("def2-svp")
    res = ferric.run_dft(mol, bs, functional="PBE")
    c = np.asarray(res.mo_coefficients())
    eps = np.asarray(res.orbital_energies())
    nbf = c.shape[0]
    assert c.shape[1] == eps.shape[0] and c.shape[1] <= nbf
    assert np.all(np.diff(eps) >= -1e-10)
    occ = c[:, :5]
    np.testing.assert_allclose(2.0 * occ @ occ.T, res.density(), atol=1e-8)
    pts = np.random.default_rng(5).normal(size=(30, 3))
    vals = np.asarray(ferric.eval_ao_on_points(mol, bs, pts)) @ c[:, :7]
    ref = np.asarray(ferric.eval_mo_on_points(mol, bs, res, list(range(7)), pts))
    np.testing.assert_allclose(vals, ref, atol=1e-10)
