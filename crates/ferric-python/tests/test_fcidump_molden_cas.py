"""`ferric.fcidump` and Molden natural orbitals of a CASSCF result.

Anchors (bars derived here, so no number is tuned to a failure):

* Round trip (no PySCF): ``fcidump.read(write(x))`` returns the 8-fold
  expansion of the file's unique elements BITWISE: the file holds Python's
  shortest round-trip form of each double and ``read`` parses it with
  ``float``, so any difference is a bug. The unique element of each class
  is what the AO transform produced, so the expansion equals the transform's
  own tensor to its symmetry error (bar 1e-12: the transform is a chain of
  four dgemms on symmetric data).
* Energy from the file's integrals (no PySCF): the converged CASSCF RDMs
  contracted with the READ integrals reproduce ``e_total``,
  ``ecore + sum(h dm1) + 1/2 sum(eri dm2)``. That is one identity in which a
  wrong ECORE (off by the core energy, tens of Ha) or a transposed (pq|rs)
  index (the dm2 contraction pairs indices) cannot both survive. Bar 1e-9:
  the RDMs are those of a CI solved to residual 1e-10, so the energy error
  is second order in it; measured numbers are in the test output.
* Cross-code (needs PySCF): ``pyscf.tools.fcidump.read`` plus
  ``fci.direct_spin1`` on the file solves the same determinant space; its
  lowest root equals ferric's ``run_casci`` / ``run_casscf`` ``e_total``.
  Bar 1e-8 Ha: ferric's CI residual (1e-9 / 1e-10) bounds its energy error
  at ~1e-18; PySCF is run at tol 1e-12; the rest is the two codes' CASCI
  reference orbitals (independent SCFs converged to density 1e-10).
* Molden natural orbitals: the occupations sum to the electron count (core
  2 each plus ``tr(dm1)`` active), are descending within the active block,
  and the density PySCF evaluates from the file equals
  ``chi_ferric D chi_ferric^T`` with D the CASSCF density
  ``2 C_c C_c^T + C_a dm1 C_a^T``; bar 1e-10 of the peak, as in
  ``test_molden.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

import ferric
from ferric import fcidump, molden

H2O = (
    "3\n\nO 0.000000 0.000000 0.117790\n"
    "H 0.000000 0.755453 -0.471161\nH 0.000000 -0.755453 -0.471161\n"
)
N2 = "2\n\nN 0 0 0\nN 0 0 1.1\n"
FCI_BAR = 1e-8
RDM_BAR = 1e-9
SYM_BAR = 1e-12
DENS_BAR = 1e-10

CASES = {
    "water": (H2O, "6-31g", 4, (2, 2)),
    "n2": (N2, "6-31g", 6, (3, 3)),
}


def _mol(xyz):
    return ferric.Molecule.from_xyz_string(xyz)


@pytest.fixture(scope="module", params=sorted(CASES))
def cas(request):
    xyz, basis, nact, nel = CASES[request.param]
    mol, bs = _mol(xyz), ferric.BasisSet.bundled(basis)
    res = ferric.run_casscf(mol, bs, nact, nel, canonicalize=True)
    assert res.converged
    return mol, bs, res, nact, nel


def _expand(h, eri, n):
    """The 8-fold expansion of the unique (p>=q, r>=s, pq>=rs) elements."""
    out = np.zeros_like(eri)
    pairs = [(p, q) for p in range(n) for q in range(p + 1)]
    for ij, (i, j) in enumerate(pairs):
        for k, ll in pairs[: ij + 1]:
            v = eri[i, j, k, ll]
            for a, b in ((i, j), (j, i)):
                for c, d in ((k, ll), (ll, k)):
                    out[a, b, c, d] = out[c, d, a, b] = v
    h_out = np.tril(h) + np.tril(h, -1).T
    return h_out, out


def _write(tmp_path, cas):
    mol, bs, res, nact, nel = cas
    path = tmp_path / "cas.fcidump"
    fcidump.write(
        path, mol, bs, res.mo_coeff, res.n_core, nact, sum(nel), nel[0] - nel[1]
    )
    return path


def test_round_trip_is_bitwise(cas, tmp_path):
    mol, bs, res, nact, nel = cas
    h, eri, ecore = fcidump.integrals(mol, bs, res.mo_coeff, res.n_core, nact)
    path = _write(tmp_path, cas)
    got = fcidump.read(path)
    assert (got["norb"], got["nelec"], got["ms2"]) == (nact, sum(nel), 0)
    h_exp, eri_exp = _expand(h, eri, nact)
    np.testing.assert_array_equal(got["h"], h_exp)
    np.testing.assert_array_equal(got["eri"], eri_exp)
    assert got["ecore"] == ecore
    assert np.max(np.abs(eri - eri_exp)) < SYM_BAR
    assert np.max(np.abs(h - h_exp)) < SYM_BAR


def test_rdm_energy_from_the_file_equals_e_total(cas, tmp_path):
    res = cas[2]
    got = fcidump.read(_write(tmp_path, cas))
    e = got["ecore"] + np.sum(got["h"] * res.dm1) + 0.5 * np.sum(got["eri"] * res.dm2)
    print(f"RDM energy minus e_total: {e - res.e_total:.3e}")
    assert abs(e - res.e_total) < RDM_BAR
    assert got["ecore"] == pytest.approx(
        fcidump.integrals(cas[0], cas[1], res.mo_coeff, res.n_core, cas[3])[2]
    )


def _pyscf_fci(path):
    pytest.importorskip("pyscf")
    from pyscf import fci
    from pyscf.tools import fcidump as pfcidump

    d = pfcidump.read(str(path))
    n = d["NORB"]
    na = (d["NELEC"] + d["MS2"]) // 2
    nb = d["NELEC"] - na
    solver = fci.direct_spin1.FCI()
    solver.conv_tol = 1e-12
    solver.max_cycle = 500
    e, _ = solver.kernel(d["H1"], d["H2"], n, (na, nb), ecore=d["ECORE"])
    return e


def test_pyscf_fci_on_the_file_reproduces_casscf(cas, tmp_path):
    e = _pyscf_fci(_write(tmp_path, cas))
    print(f"PySCF FCI minus run_casscf e_total: {e - cas[2].e_total:.3e}")
    assert abs(e - cas[2].e_total) < FCI_BAR


@pytest.mark.parametrize("name", sorted(CASES))
def test_pyscf_fci_on_the_file_reproduces_casci(name, tmp_path):
    xyz, basis, nact, nel = CASES[name]
    mol, bs = _mol(xyz), ferric.BasisSet.bundled(basis)
    ci = ferric.run_casci(mol, bs, nact, nel, compute_rdms=False)
    rhf = ferric.run_rhf(mol, bs)
    n_core = ci.active_start
    path = tmp_path / "casci.fcidump"
    fcidump.write(path, mol, bs, rhf.mo_coefficients(), n_core, nact, sum(nel), 0)
    assert fcidump.read(path)["ecore"] == pytest.approx(ci.e_core, abs=1e-8)
    e = _pyscf_fci(path)
    print(f"PySCF FCI minus run_casci e_total: {e - ci.e_total:.3e}")
    assert abs(e - ci.e_total) < FCI_BAR


def test_open_shell_sector_header(tmp_path):
    mol, bs = _mol(N2), ferric.BasisSet.bundled("6-31g")
    res = ferric.run_casscf(mol, bs, 6, (4, 2), canonicalize=True)
    path = tmp_path / "triplet.fcidump"
    fcidump.write(path, mol, bs, res.mo_coeff, res.n_core, 6, 6, 2)
    got = fcidump.read(path)
    assert (got["nelec"], got["ms2"]) == (6, 2)
    e = _pyscf_fci(path)
    assert abs(e - res.e_total) < FCI_BAR


def test_bad_arguments_raise(cas, tmp_path):
    mol, bs, res, nact, nel = cas
    with pytest.raises(ValueError):
        fcidump.integrals(mol, bs, res.mo_coeff, res.n_core, res.mo_coeff.shape[1])
    with pytest.raises(ValueError, match="spin sector"):
        fcidump.write(tmp_path / "x", mol, bs, res.mo_coeff, res.n_core, nact, 3, 0)


# ── Molden natural orbitals ──────────────────────────────────────────────


def _density(res):
    c = np.asarray(res.mo_coeff)
    nc, na = res.n_core, res.dm1.shape[0]
    cc, ca = c[:, :nc], c[:, nc : nc + na]
    return 2.0 * cc @ cc.T + ca @ np.asarray(res.dm1) @ ca.T


def test_molden_natural_occupations(cas, tmp_path):
    mol, bs, res, nact, nel = cas
    occ = molden.orbitals(mol, bs, res)[0]["occupations"]
    nc = res.n_core
    assert occ.sum() == pytest.approx(2 * nc + sum(nel), abs=1e-10)
    assert occ[nc : nc + nact].sum() == pytest.approx(sum(nel), abs=1e-10)
    assert np.all(occ[:nc] == 2.0) and np.all(occ[nc + nact :] == 0.0)
    act = occ[nc : nc + nact]
    assert np.all(np.diff(act) <= 1e-12) and 0.0 <= act.min() and act.max() <= 2.0


def test_molden_density_matches_the_casscf_density(cas, tmp_path):
    pytest.importorskip("pyscf")
    from pyscf.tools import molden as pmolden

    mol, bs, res, nact, nel = cas
    path = tmp_path / "cas.molden"
    molden.write(path, mol, bs, res)
    pmol, _, c, occ, _, _ = pmolden.load(str(path))
    occ_ref = molden.orbitals(mol, bs, res)[0]["occupations"]
    np.testing.assert_array_equal(occ, occ_ref)
    rng = np.random.default_rng(7)
    points = np.asarray(mol.coords_bohr())[rng.integers(0, 2, 40)] + rng.normal(
        0, 1.2, (40, 3)
    )
    chi_p = pmol.eval_gto("GTOval_cart" if pmol.cart else "GTOval_sph", points)
    rho_file = np.einsum("pk,k,pk->p", chi_p @ c, occ, chi_p @ c)
    chi_f = np.asarray(ferric.eval_ao_on_points(mol, bs, points))
    rho_cas = np.einsum("pm,mn,pn->p", chi_f, _density(res), chi_f)
    rel = np.max(np.abs(rho_file - rho_cas)) / np.max(np.abs(rho_cas))
    print(f"molden density vs CASSCF density: {rel:.3e}")
    assert rel < DENS_BAR
    s = np.asarray(ferric.AoIntegrals(mol, bs).overlap())
    assert np.sum(_density(res) * s) == pytest.approx(
        2 * res.n_core + sum(nel), abs=1e-9
    )
