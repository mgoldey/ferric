"""`AoIntegrals` MO transforms (`ao2mo`, `df_mo`) and gradient pieces.

Anchors are independent constructions:

* `ao2mo` equals the dense `eri()` tensor (a separate serial quartet loop)
  contracted with numpy, for four different non-square coefficient blocks;
* `df_mo` equals a numpy fit from `eri3()` + `eri2()`, and an RI-MP2 energy
  assembled in numpy from it reproduces `run_rimp2`;
* SCF gradients assembled in numpy from `grad_*` equal central finite
  differences of `run_rhf` / `run_uhf` energies (an ECP case included), and
  every piece is translation-invariant on its own (sums to zero over atoms).
"""

from __future__ import annotations

import numpy as np
import pytest

import ferric

WATER = "3\n\nO 0 0 0.117\nH 0 0.757 -0.467\nH 0 -0.757 -0.467\n"
NH2 = "3\n\nN 0 0 0\nH 0 0.802 0.597\nH 0 -0.802 0.597\n"
TIGHT = {"energy_conv": 1e-11, "density_conv": 1e-9}
_RNG = np.random.default_rng(11)


def _ints(xyz: str, basis: str, multiplicity: int = 1):
    mol = ferric.Molecule.from_xyz_string(xyz, multiplicity=multiplicity)
    bs = ferric.BasisSet.bundled(basis)
    return mol, bs, ferric.AoIntegrals(mol, bs)


def _canonical(ints, f):
    """Orbital energies and coefficients of F C = S C e."""
    s = ints.overlap()
    w, v = np.linalg.eigh(s)
    x = v / np.sqrt(w)
    e, c = np.linalg.eigh(x.T @ f @ x)
    return e, x @ c


# ── ao2mo ──


@pytest.mark.parametrize(
    "operator, omega", [("coulomb", None), ("erfc", 0.4)], ids=["coulomb", "erfc"]
)
def test_ao2mo_matches_the_dense_eri_contraction(operator, omega):
    _, _, ints = _ints(WATER, "cc-pvdz")
    n = ints.nao
    c1, c2, c3, c4 = (_RNG.standard_normal((n, k)) for k in (3, 5, 2, 4))
    ref = np.einsum(
        "ijkl,ip,jq,kr,ls->pqrs",
        ints.eri(operator, omega=omega),
        c1,
        c2,
        c3,
        c4,
        optimize=True,
    )
    got = ints.ao2mo(c1, c2, c3, c4, operator, omega=omega)
    assert got.shape == (3, 5, 2, 4)
    np.testing.assert_allclose(got, ref, atol=1e-11)


def test_ao2mo_refuses_above_the_memory_cap_and_bad_coefficients():
    _, _, ints = _ints(WATER, "sto-3g")
    c = np.eye(ints.nao)
    with pytest.raises(ValueError, match="max_gb"):
        ints.ao2mo(c, c, c, c, max_gb=1e-9)
    with pytest.raises(ValueError, match="c3 has shape"):
        ints.ao2mo(c, c, c[:-1], c)
    with pytest.raises(ValueError, match="NaN or inf"):
        ints.ao2mo(c, c, c, np.full_like(c, np.nan))


# ── df_mo ──


def test_df_mo_matches_a_numpy_fit():
    _, _, ints = _ints(WATER, "cc-pvdz")
    aux = ferric.BasisSet.bundled("cc-pvdz-ri")
    n = ints.nao
    c1, c2 = _RNG.standard_normal((n, 4)), _RNG.standard_normal((n, 6))
    b = ints.df_mo(aux, c1, c2)
    raw = np.einsum("pij,ia,jb->pab", ints.eri3(aux), c1, c2)
    ref = np.einsum("pab,pq,qcd->abcd", raw, np.linalg.inv(ints.eri2(aux)), raw)
    np.testing.assert_allclose(np.einsum("pab,pcd->abcd", b, b), ref, atol=1e-10)


def test_numpy_ri_mp2_on_df_mo_reproduces_run_rimp2():
    mol, bs, ints = _ints(WATER, "cc-pvdz")
    aux = ferric.BasisSet.bundled("cc-pvdz-ri")
    r = ferric.run_rhf(mol, bs, **TIGHT)
    j, k = ints.jk(r.density())
    e, c = _canonical(ints, ints.hcore() + j - 0.5 * k)
    nocc = ints.nelec // 2
    b = ints.df_mo(aux, c[:, :nocc], c[:, nocc:])
    iajb = np.einsum("pia,pjb->iajb", b, b)
    denom = (
        e[:nocc, None, None, None]
        - e[None, nocc:, None, None]
        + e[None, None, :nocc, None]
        - e[None, None, None, nocc:]
    )
    e_corr = np.sum(iajb * (2 * iajb - iajb.transpose(0, 3, 2, 1)) / denom)
    ref = ferric.run_rimp2(mol, bs, aux, frozen_core=0)
    assert abs(e_corr - ref.mp2_corr) < 1e-7


# ── gradient pieces ──


def _displaced(mol, xyz, atom, axis, h_bohr, multiplicity):
    """`xyz` with one coordinate moved by h_bohr, rebuilt via Angstrom."""
    bohr_per_ang = next(
        b / a
        for cb, ca in zip(mol.coords_bohr(), mol.coords())
        for b, a in zip(cb, ca)
        if abs(a) > 0.1
    )
    lines = xyz.strip("\n").split("\n")
    head, atoms = lines[:2], [ln.split() for ln in lines[2:]]
    atoms[atom][1 + axis] = repr(float(atoms[atom][1 + axis]) + h_bohr / bohr_per_ang)
    body = "\n".join(" ".join(a) for a in atoms)
    return ferric.Molecule.from_xyz_string(
        "\n".join(head) + "\n" + body + "\n", multiplicity=multiplicity
    )


def _fd(xyz, bs, run, coords, multiplicity=1, h=1e-3):
    mol = ferric.Molecule.from_xyz_string(xyz, multiplicity=multiplicity)
    out = {}
    for atom, axis in coords:
        ep = run(_displaced(mol, xyz, atom, axis, h, multiplicity), bs).energy
        em = run(_displaced(mol, xyz, atom, axis, -h, multiplicity), bs).energy
        out[(atom, axis)] = (ep - em) / (2 * h)
    return out


def _rhf_gradient(ints, d):
    """Closed-shell HF gradient assembled from the pieces."""
    j, k = ints.jk(d)
    e, c = _canonical(ints, ints.hcore() + j - 0.5 * k)
    nocc = ints.nelec // 2
    w = 2.0 * (c[:, :nocc] * e[:nocc]) @ c[:, :nocc].T
    gj, gk = ints.grad_jk(d)
    return (
        ints.grad_nuclear_repulsion()
        + ints.grad_hcore(d)
        - ints.grad_overlap(w)
        + gj
        - 0.5 * gk
    )


def _rhf(mol, bs):
    return ferric.run_rhf(mol, bs, **TIGHT)


@pytest.mark.parametrize(
    "xyz, basis, coords",
    [
        (WATER, "sto-3g", [(0, 2), (1, 1), (1, 2), (2, 1)]),
        ("2\n\nH 0 0 0\nI 0 0 1.609\n", "def2-svp", [(0, 2), (1, 2)]),
    ],
    ids=["water", "HI-ecp"],
)
def test_rhf_gradient_from_pieces_matches_finite_difference(xyz, basis, coords):
    mol, bs, ints = _ints(xyz, basis)
    g = _rhf_gradient(ints, _rhf(mol, bs).density())
    for (atom, axis), fd in _fd(xyz, bs, _rhf, coords).items():
        assert abs(g[atom, axis] - fd) < 2e-6, (atom, axis, g[atom, axis], fd)


def _uhf(mol, bs):
    return ferric.run_uhf(mol, bs, **TIGHT)


def test_uhf_gradient_from_pieces_matches_finite_difference():
    mol, bs, ints = _ints(NH2, "sto-3g", multiplicity=2)
    r = _uhf(mol, bs)
    da, db = r.density_alpha(), r.density_beta()
    h = ints.hcore()
    j, _ = ints.jk(da + db)
    w_tot = np.zeros_like(h)
    for dsig, nsig in ((da, 5), (db, 4)):
        _, ksig = ints.jk(dsig)
        e, c = _canonical(ints, h + j - ksig)
        w_tot += (c[:, :nsig] * e[:nsig]) @ c[:, :nsig].T
    gj, _ = ints.grad_jk(da + db)
    g = (
        ints.grad_nuclear_repulsion()
        + ints.grad_hcore(da + db)
        - ints.grad_overlap(w_tot)
        + gj
        - ints.grad_jk(da)[1]
        - ints.grad_jk(db)[1]
    )
    coords = [(0, 2), (1, 1), (1, 2)]
    for (atom, axis), fd in _fd(NH2, bs, _uhf, coords, multiplicity=2).items():
        assert abs(g[atom, axis] - fd) < 2e-6, (atom, axis, g[atom, axis], fd)


def test_every_gradient_piece_is_translation_invariant():
    mol, bs, ints = _ints(WATER, "cc-pvdz")
    d = _rhf(mol, bs).density()
    w = d @ ints.overlap() @ d  # any symmetric W
    gj, gk = ints.grad_jk(d)
    for name, g in [
        ("nuc", ints.grad_nuclear_repulsion()),
        ("hcore", ints.grad_hcore(d)),
        ("overlap", ints.grad_overlap(w)),
        ("J", gj),
        ("K", gk),
    ]:
        assert g.shape == (3, 3)
        np.testing.assert_allclose(g.sum(axis=0), 0.0, atol=1e-9, err_msg=name)
        assert np.abs(g).max() > 1e-3, name


def test_grad_jk_matches_finite_difference_of_the_jk_energies():
    # Integral derivatives at FIXED D: move one atom (its basis functions
    # move with it) and difference ½Tr[D J(D)] and ½Tr[D K(D)].
    xyz = WATER
    mol, bs, ints = _ints(xyz, "sto-3g")
    d = _rhf(mol, bs).density()
    gj, gk = ints.grad_jk(d)
    h = 1e-4
    for atom, axis in [(0, 2), (1, 1)]:
        ej, ek = [], []
        for step in (h, -h):
            moved = ferric.AoIntegrals(_displaced(mol, xyz, atom, axis, step, 1), bs)
            j, k = moved.jk(d)
            ej.append(0.5 * np.sum(d * j))
            ek.append(0.5 * np.sum(d * k))
        assert abs(gj[atom, axis] - (ej[0] - ej[1]) / (2 * h)) < 1e-6
        assert abs(gk[atom, axis] - (ek[0] - ek[1]) / (2 * h)) < 1e-6


def test_gradient_pieces_reject_bad_matrices():
    _, _, ints = _ints(WATER, "sto-3g")
    n = ints.nao
    with pytest.raises(ValueError, match="shape"):
        ints.grad_hcore(np.zeros((n, n + 1)))
    asym = np.zeros((n, n))
    asym[0, 1] = 1.0
    with pytest.raises(ValueError, match="not symmetric"):
        ints.grad_jk(asym)
    with pytest.raises(ValueError, match="NaN or inf"):
        ints.grad_overlap(np.full((n, n), np.nan))
    ints.grad_overlap(asym)  # W may be asymmetric: used as given


# ── density-fitted gradient pieces ──

AUX = "cc-pvdz-ri"


@pytest.mark.parametrize(
    "operator, omega", [("coulomb", None), ("erfc", 0.4)], ids=["coulomb", "erfc"]
)
def test_grad_df_jk_matches_finite_difference_of_the_fitted_energies(operator, omega):
    # Fixed D; moving an atom moves its orbital AND auxiliary functions, so
    # this differences exactly what grad_df_jk differentiates.
    xyz = WATER
    mol, bs, ints = _ints(xyz, "sto-3g")
    aux = ferric.BasisSet.bundled(AUX)
    d = _rhf(mol, bs).density()
    gj, gk = ints.grad_df_jk(aux, d, operator, omega=omega)
    h = 1e-4
    for atom, axis in [(0, 2), (1, 1), (2, 2)]:
        ej, ek = [], []
        for step in (h, -h):
            moved = ferric.AoIntegrals(_displaced(mol, xyz, atom, axis, step, 1), bs)
            j, k = moved.df_jk(aux, operator, omega=omega).jk(d)
            ej.append(0.5 * np.sum(d * j))
            ek.append(0.5 * np.sum(d * k))
        assert abs(gj[atom, axis] - (ej[0] - ej[1]) / (2 * h)) < 1e-6
        assert abs(gk[atom, axis] - (ek[0] - ek[1]) / (2 * h)) < 1e-6


def _ri_rhf(mol, bs):
    return ferric.run_rhf(mol, bs, df_j_aux=AUX, df_k_aux=AUX, **TIGHT)


def test_ri_jk_rhf_gradient_from_pieces_matches_finite_difference():
    mol, bs, ints = _ints(WATER, "sto-3g")
    aux = ferric.BasisSet.bundled(AUX)
    d = _ri_rhf(mol, bs).density()
    j, k = ints.df_jk(aux).jk(d)
    e, c = _canonical(ints, ints.hcore() + j - 0.5 * k)
    nocc = ints.nelec // 2
    w = 2.0 * (c[:, :nocc] * e[:nocc]) @ c[:, :nocc].T
    gj, gk = ints.grad_df_jk(aux, d)
    g = (
        ints.grad_nuclear_repulsion()
        + ints.grad_hcore(d)
        - ints.grad_overlap(w)
        + gj
        - 0.5 * gk
    )
    coords = [(0, 2), (1, 1), (1, 2), (2, 1)]
    for (atom, axis), fd in _fd(WATER, bs, _ri_rhf, coords).items():
        assert abs(g[atom, axis] - fd) < 2e-6, (atom, axis, g[atom, axis], fd)
    # The fit is not the exact energy: the exact pieces miss it by far more.
    gj_x, gk_x = ints.grad_jk(d)
    assert np.abs((gj - 0.5 * gk) - (gj_x - 0.5 * gk_x)).max() > 1e-5


def _ri_uhf(mol, bs):
    return ferric.run_uhf(mol, bs, df_j_aux=AUX, df_k_aux=AUX, **TIGHT)


def test_ri_jk_uhf_gradient_from_pieces_matches_finite_difference():
    mol, bs, ints = _ints(NH2, "sto-3g", multiplicity=2)
    aux = ferric.BasisSet.bundled(AUX)
    df = ints.df_jk(aux)
    r = _ri_uhf(mol, bs)
    da, db = r.density_alpha(), r.density_beta()
    h = ints.hcore()
    j = df.j(da + db)
    w_tot = np.zeros_like(h)
    for dsig, nsig in ((da, 5), (db, 4)):
        e, c = _canonical(ints, h + j - df.k(dsig))
        w_tot += (c[:, :nsig] * e[:nsig]) @ c[:, :nsig].T
    g = (
        ints.grad_nuclear_repulsion()
        + ints.grad_hcore(da + db)
        - ints.grad_overlap(w_tot)
        + ints.grad_df_jk(aux, da + db)[0]
        - ints.grad_df_jk(aux, da)[1]
        - ints.grad_df_jk(aux, db)[1]
    )
    coords = [(0, 2), (1, 1), (1, 2)]
    for (atom, axis), fd in _fd(NH2, bs, _ri_uhf, coords, multiplicity=2).items():
        assert abs(g[atom, axis] - fd) < 2e-6, (atom, axis, g[atom, axis], fd)


def test_df_gradient_pieces_are_translation_invariant_and_refuse_indefinite_d():
    mol, bs, ints = _ints(WATER, "cc-pvdz")
    aux = ferric.BasisSet.bundled(AUX)
    d = _rhf(mol, bs).density()
    for name, g in zip("JK", ints.grad_df_jk(aux, d)):
        np.testing.assert_allclose(g.sum(axis=0), 0.0, atol=1e-9, err_msg=name)
        assert np.abs(g).max() > 1e-3, name
    with pytest.raises(RuntimeError, match="positive semidefinite"):
        ints.grad_df_jk(aux, d - 2.0 * np.eye(ints.nao))
