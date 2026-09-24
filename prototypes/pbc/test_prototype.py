"""Fast regression tests for the Gamma-point PBC prototype (pbc_gamma.py).

Run:  OPENBLAS_NUM_THREADS=1 scripts/ferric-limited -- python3 -m pytest reference/pbc/test_prototype.py -q
Default set is ~60 s incl. the RS-GDF tests (the pure-AFT H2 build is ~22 s and shared by a module fixture).
PBC_SLOW=1 also runs the live PySCF AFTDF oracle (~2 min) and the a=16 Makov-Payne check.

Pinned references (PySCF 2.13 AFTDF, mesh 61^3, cell.precision 1e-12, conv_tol 1e-12,
measured 2026-09-23 by run_h2_exxdiv.py):
  exxdiv=None  : E = -0.949002691179, eps = [-0.13811129, 1.23177667]
  exxdiv=ewald : E = -1.658327061049, eps = [-0.84743566, 1.23177667]
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pbc_gamma import Cell, build_integrals, ewald_nn, madelung, pair_ft, rhf  # noqa: E402

from pyscf import gto  # noqa: E402
from pyscf.pbc import gto as pgto  # noqa: E402

SLOW = bool(os.environ.get("PBC_SLOW"))

H2_A = np.eye(3) * 4.0
H2_ATOMS = [("H", (0.3, 0.2, 0.1)), ("H", (0.3, 0.2, 1.5))]
TRI_A = np.array([[4.6, 0.0, 0.0], [0.9, 4.3, 0.0], [0.5, 0.7, 4.8]])
TRI_ATOMS = [
    ("H", (0.1, 0.2, 0.3)),
    ("H", (0.1, 0.2, 1.7)),
    ("H", (2.4, 2.5, 2.2)),
    ("H", (3.6, 2.9, 2.6)),
]
SP_BASIS = {
    "H": gto.parse(
        """
H S
  3.42525091  0.15432897
  0.62391373  0.53532814
  0.16885540  0.44463454
H P
  0.8         1.0
"""
    )
}
E_REF = {"none": -0.949002691179, "ewald": -1.658327061049}
EPS_REF = {"none": [-0.13811129, 1.23177667], "ewald": [-0.84743566, 1.23177667]}


def pyscf_cell(a, atoms, basis):
    pc = pgto.Cell(a=a, atom=atoms, basis=basis, unit="B", cart=True, verbose=0)
    pc.precision = 1e-12
    return pc.build()


# ------------------------------------------------------------------- Ewald / Madelung
@pytest.mark.parametrize("a,atoms", [(H2_A, H2_ATOMS), (TRI_A, TRI_ATOMS)])
def test_ewald_enn_matches_pyscf_and_is_w_independent(a, atoms):
    ref = pyscf_cell(a, atoms, "sto-3g").energy_nuc()
    for w in (0.8, 1.5, 3.0):
        assert abs(ewald_nn(Cell(a, atoms, "sto-3g"), w) - ref) < 1e-12


def test_madelung_simple_cubic_literature_constant():
    # Simple-cubic Madelung constant with neutralising background: v_M * a = 2.8372974794806...
    for edge in (4.0, 10.0):
        assert (
            abs(
                madelung(Cell(np.eye(3) * edge, H2_ATOMS, "sto-3g")) * edge
                - 2.8372974794806
            )
            < 1e-11
        )


@pytest.mark.parametrize("a", [H2_A, TRI_A])
def test_madelung_matches_pyscf_cross_check(a):
    from pyscf.pbc import tools

    cell = Cell(a, H2_ATOMS, "sto-3g")
    ref = tools.pbc.madelung(pyscf_cell(a, H2_ATOMS, "sto-3g"), np.zeros((1, 3)))
    for w in (
        0.5,
        1.0,
        2.0,
    ):  # own Ewald split parameter; the constant must not depend on it
        assert abs(madelung(cell, w) - ref) < 1e-13


# ------------------------------------------------------------------------- pair FT
def _calibrated_pair_ft(cell, G, S):
    P0 = pair_ft(cell, np.zeros((1, 3)))[..., 0].real
    nrm = np.sqrt(np.diag(S) / np.diag(P0))
    return P0 * np.outer(nrm, nrm), pair_ft(cell, G) * nrm[:, None, None] * nrm[
        None, :, None
    ]


def test_pair_ft_at_G0_is_lattice_overlap_all_elements():
    """Diagonal is calibrated; the check is every element, s+p basis, triclinic."""
    cell = Cell(TRI_A, TRI_ATOMS, SP_BASIS)
    S = pyscf_cell(TRI_A, TRI_ATOMS, SP_BASIS).pbc_intor("int1e_ovlp")
    P0, _ = _calibrated_pair_ft(cell, np.zeros((0, 3)), S)
    assert abs(P0 - S).max() < 1e-11


def test_pair_ft_at_nonzero_G_matches_pyscf_ft_aopair():
    """G != 0 with p functions: the only test that reaches the (-iG)^t, t>0 branch."""
    from pyscf.pbc.df.ft_ao import ft_aopair

    pc = pyscf_cell(TRI_A, TRI_ATOMS, SP_BASIS)
    cell = Cell(TRI_A, TRI_ATOMS, SP_BASIS)
    rng = np.random.default_rng(7)
    G = rng.integers(-3, 4, size=(12, 3)) @ cell.b
    S = pc.pbc_intor("int1e_ovlp")
    _, P = _calibrated_pair_ft(cell, G, S)
    ref = ft_aopair(pc, G)  # (ng, nao, nao), same e^{-iG.r} sign convention
    assert abs(P - ref.transpose(1, 2, 0)).max() < 1e-10


# ------------------------------------------------------------------ H2 energies
@pytest.fixture(scope="module")
def h2_ints():
    return build_integrals(
        Cell(H2_A, H2_ATOMS, "sto-3g"), None, exxdiv="ewald", verbose=False
    )


@pytest.mark.parametrize("exxdiv", ["none", "ewald"])
def test_pure_aft_h2_matches_pinned_pyscf_aftdf(h2_ints, exxdiv):
    ks = h2_ints["madelung"] if exxdiv == "ewald" else 0.0
    e, eps, _ = rhf(
        h2_ints["S"],
        h2_ints["h"],
        h2_ints["I"],
        h2_ints["enn"],
        2,
        conv=1e-12,
        kshift=ks,
    )
    assert abs(e - E_REF[exxdiv]) < 1e-10
    assert abs(eps - EPS_REF[exxdiv]).max() < 1e-7  # pinned at 8 decimals


def test_ewald_shift_is_minus_vM_per_electron_pair(h2_ints):
    """At Gamma the Madelung term shifts occupied levels by -v_M and E by -v_M*N/2
    without changing D (it is v_M S D S, which acts as -v_M on the occupied space)."""
    vm = h2_ints["madelung"]
    e0, eps0, _ = rhf(
        h2_ints["S"], h2_ints["h"], h2_ints["I"], h2_ints["enn"], 2, conv=1e-12
    )
    e1, eps1, _ = rhf(
        h2_ints["S"],
        h2_ints["h"],
        h2_ints["I"],
        h2_ints["enn"],
        2,
        conv=1e-12,
        kshift=vm,
    )
    assert abs((e1 - e0) + vm) < 1e-10
    assert abs((eps1[0] - eps0[0]) + vm) < 1e-10 and abs(eps1[1] - eps0[1]) < 1e-10


def test_exxdiv_rejects_unknown_value():
    with pytest.raises(ValueError):
        build_integrals(
            Cell(H2_A, H2_ATOMS, "sto-3g"), None, exxdiv="vcut_sph", verbose=False
        )


# ------------------------------------------------------------------------- slow
@pytest.mark.skipif(not SLOW, reason="set PBC_SLOW=1 (~2 min live PySCF AFTDF)")
@pytest.mark.parametrize("exxdiv", [None, "ewald"])
def test_live_pyscf_aftdf_oracle(exxdiv):
    from pyscf.pbc import df as pdf
    from pyscf.pbc import scf as pscf

    mf = pscf.RHF(pyscf_cell(H2_A, H2_ATOMS, "sto-3g"), exxdiv=exxdiv)
    mf.with_df = pdf.AFTDF(mf.cell)
    mf.with_df.mesh = [61] * 3
    mf.conv_tol = 1e-12
    assert abs(mf.kernel() - E_REF["none" if exxdiv is None else exxdiv]) < 1e-10


@pytest.mark.skipif(not SLOW, reason="set PBC_SLOW=1 (~25 s)")
def test_molecular_limit_residual_is_makov_payne_exchange_term():
    """Independent of PySCF pbc: with exxdiv='ewald' the box residual E_pbc - E_mol must be
    the exchange Makov-Payne term -(4pi/3) sigma^2 / a^3 (sigma^2 = second central moment of
    the occupied orbital density). Measured tail coefficient -10.028 vs predicted -10.0282."""
    from pyscf import scf

    mol = gto.M(atom=H2_ATOMS, basis="sto-3g", unit="B", cart=True, verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    e_mol = mf.kernel()
    c = mf.mo_coeff[:, 0]
    cen = np.einsum("xmn,m,n->x", mol.intor("int1e_r"), c, c)
    s2 = (
        np.einsum("xxmn,m,n->", mol.intor("int1e_rr").reshape(3, 3, 2, 2), c, c)
        - cen @ cen
    )
    edge = 16.0
    ints = build_integrals(
        Cell(np.eye(3) * edge, H2_ATOMS, "sto-3g"), None, exxdiv="ewald", verbose=False
    )
    e = rhf(
        ints["S"],
        ints["h"],
        ints["I"],
        ints["enn"],
        2,
        conv=1e-12,
        kshift=ints["madelung"],
    )[0]
    pred = -4 * np.pi / 3 * s2 / edge**3
    assert abs((e - e_mol) - pred) < 0.01 * abs(pred)  # O(a^-5) remainder ~0.2% at a=16


# ============================================================ RS-GDF (pbc_gdf.py, Iteration 2)
import pbc_gdf  # noqa: E402
from pbc_gdf import build_gdf, eri_from_B, ferric_basis, jk_from_B  # noqa: E402

ANCHOR_ALPHA = 0.5


def _anchor_cell_and_aux(classes=range(8)):
    """Exactness anchor: one primitive s per H (exponent al).  Every lattice pair product
    phi_m(r) phi_n(r-L) is ONE s Gaussian (exponent 2 al) at (A_m+A_n+L)/2, and modulo the
    lattice those centres fall in 8 half-lattice classes per (m,n) type.  With those 24 aux
    functions the periodic pair densities lie exactly in the aux span => the fit is exact."""
    cell = Cell(H2_A, H2_ATOMS, {"H": [[0, [ANCHOR_ALPHA, 1.0]]]})
    cen = [
        0.5 * (cell.R[i] + cell.R[j] + np.array(h) @ cell.a)
        for i, j in [(0, 0), (1, 1), (0, 1)]
        for k, h in enumerate(np.ndindex(2, 2, 2))
        if k in classes
    ]
    aux = gto.M(
        atom=[("X", c) for c in cen],
        basis={"X": [[0, [2 * ANCHOR_ALPHA, 1.0]]]},
        unit="B",
        cart=True,
        verbose=0,
    )
    return cell, aux


@pytest.fixture(scope="module")
def anchor_ref():
    cell, _ = _anchor_cell_and_aux()
    return build_integrals(cell, None, exxdiv="ewald", verbose=False)


def _dE(ref, B, nelec=2, kshift=0.0):
    e0 = rhf(
        ref["S"], ref["h"], ref["I"], ref["enn"], nelec, conv=1e-12, kshift=kshift
    )[0]
    e1 = rhf(
        ref["S"],
        ref["h"],
        None,
        ref["enn"],
        nelec,
        conv=1e-12,
        kshift=kshift,
        jk=jk_from_B(B),
    )[0]
    return e1 - e0


def test_gdf_matches_exact_in_the_trivial_limit(anchor_ref):
    """Aux = all periodic pair products: B B^T must reproduce the pure-AFT ERI (measured 1.5e-12)."""
    cell, aux = _anchor_cell_and_aux()
    B = build_gdf(cell, None, auxmol=aux)["B"]
    assert abs(eri_from_B(B) - anchor_ref["I"]).max() < 1e-10
    for ks in (0.0, anchor_ref["madelung"]):
        assert abs(_dE(anchor_ref, B, kshift=ks)) < 1e-10


@pytest.mark.parametrize(
    "mutant",
    [
        lambda J2, J3, S, q, c0: (
            J2 - c0 * np.outer(q, q),
            J3,
        ),  # G=0 removed from metric only: dI 5.3
        lambda J2, J3, S, q, c0: (
            J2,
            J3,
        ),  # kept in both = a different (w-dependent) kernel: dI 6e-2
    ],
)
def test_gdf_anchor_fails_when_g0_is_mishandled(anchor_ref, monkeypatch, mutant):
    cell, aux = _anchor_cell_and_aux()
    monkeypatch.setattr(pbc_gdf, "_subtract_g0", mutant)
    B = build_gdf(cell, None, auxmol=aux)["B"]
    assert abs(eri_from_B(B) - anchor_ref["I"]).max() > 1e-3


def test_gdf_anchor_detects_missing_aux_images(anchor_ref):
    """SR lattice sum over aux images truncated to T=0: dI 3e-2."""
    cell, aux = _anchor_cell_and_aux()
    B = build_gdf(cell, None, auxmol=aux, rcut_aux3=0.1, rcut_aux2=0.1)["B"]
    assert abs(eri_from_B(B) - anchor_ref["I"]).max() > 1e-3


def test_gdf_is_independent_of_ewald_split_parameter():
    """J2', J3' are w-independent quantities; the split only changes how they are evaluated.
    Uses a real (non-anchor) aux basis with p and d shells."""
    cell = Cell(H2_A, H2_ATOMS, "sto-3g")
    aux = ferric_basis("cc-pvdz-ri", ["H"])
    I = [eri_from_B(build_gdf(cell, aux, w=w)["B"]) for w in (0.7, 1.4)]
    assert abs(I[0] - I[1]).max() < 1e-9


def test_aux_ft_matches_pyscf_ft_ao_with_d_shells():
    """One-centre aux FT (incl. Cartesian d with nonzero charge) vs PySCF ft_ao (oracle)."""
    from pyscf.pbc.df.ft_ao import ft_ao

    bas = ferric_basis("def2-universal-jkfit", ["H"])
    pc = pgto.Cell(
        a=TRI_A, atom=TRI_ATOMS[:2], basis=bas, unit="B", cart=True, verbose=0
    )
    pc.build()
    mol = gto.M(atom=TRI_ATOMS[:2], basis=bas, unit="B", cart=True, verbose=0)
    G = (
        np.random.default_rng(3).integers(-3, 4, size=(15, 3))
        @ Cell(TRI_A, TRI_ATOMS, "sto-3g").b
    )
    G = np.vstack([np.zeros(3), G])
    assert abs(pbc_gdf.aux_ft(mol, G) - ft_ao(pc, G).T).max() < 1e-12


def test_gdf_h2_sto3g_cc_pvdz_ri_fitting_error(h2_ints):
    """Production-like aux (ferric's bundled cc-pvdz-ri): measured dE = -1.97e-6 for both
    exxdiv conventions (the Madelung shift uses S, not the fit, so it adds no fitting error)."""
    B = build_gdf(Cell(H2_A, H2_ATOMS, "sto-3g"), ferric_basis("cc-pvdz-ri", ["H"]))[
        "B"
    ]
    d_none = _dE(h2_ints, B)
    d_ewald = _dE(h2_ints, B, kshift=h2_ints["madelung"])
    assert 1e-7 < abs(d_none) < 5e-6  # a real fit: nonzero, and at the µHa level
    assert abs(d_none - d_ewald) < 1e-10


# ============================================================ Gamma MP2 (pbc_mp2.py, Iteration 3)
from pbc_mp2 import (
    denominators,
    dipole_prediction_h2_minimal,
    gamma_mp2,
    mp2_energy,
    ovov_from_B,
)  # noqa: E402

# PySCF 2.13 pbc.mp.RMP2 on AFTDF (mesh 61^3), H2/STO-3G a=4, measured 2026-09-23 by run_mp2_oracle.py:
# exxdiv=None -> unshifted denominators, exxdiv='ewald' -> shifted (both agree with ours to 4e-16).
MP2_REF = {"unshifted": -5.891222456423e-03, "shifted": -3.881428851328e-03}


def _tri_anchor(classes=range(8)):
    """Trivial-aux limit with nocc = nvir = 2 (H2's nocc = nvir = 1 cannot see the exchange term):
    one s primitive per H on the triclinic 4H cell; aux = all 10 x 8 periodic pair products."""
    cell = Cell(TRI_A, TRI_ATOMS, {"H": [[0, [ANCHOR_ALPHA, 1.0]]]})
    n = len(TRI_ATOMS)
    cen = [
        0.5 * (cell.R[i] + cell.R[j] + np.array(h) @ cell.a)
        for i in range(n)
        for j in range(i, n)
        for k, h in enumerate(np.ndindex(2, 2, 2))
        if k in classes
    ]
    aux = gto.M(
        atom=[("X", c) for c in cen],
        basis={"X": [[0, [2 * ANCHOR_ALPHA, 1.0]]]},
        unit="B",
        cart=True,
        verbose=0,
    )
    return cell, aux


@pytest.fixture(scope="module")
def tri_anchor():
    cell, aux = _tri_anchor()
    ref = build_integrals(cell, None, exxdiv="ewald", verbose=False)
    e, eps, _, C = rhf(
        ref["S"], ref["h"], ref["I"], ref["enn"], 4, conv=1e-12, return_mo=True
    )
    return dict(
        cell=cell,
        aux=aux,
        ref=ref,
        eps=eps,
        C=C,
        B=build_gdf(cell, None, auxmol=aux)["B"],
    )


@pytest.mark.parametrize("convention", ["shifted", "unshifted"])
def test_mp2_from_B_matches_dense_aft_in_the_trivial_aux_limit(tri_anchor, convention):
    """Measured |dE_MP2| 1e-13 (same orbitals) / 7e-14..2e-13 (orbitals from the B-tensor SCF)."""
    t = tri_anchor
    e = np.concatenate(denominators(t["eps"], 2, t["ref"]["madelung"], convention))
    exact = gamma_mp2(t["C"], e, 2, eri=t["ref"]["I"])[0]
    assert abs(gamma_mp2(t["C"], e, 2, B=t["B"])[0] - exact) < 1e-11
    _, eps1, _, C1 = rhf(
        t["ref"]["S"],
        t["ref"]["h"],
        None,
        t["ref"]["enn"],
        4,
        conv=1e-12,
        jk=jk_from_B(t["B"]),
        return_mo=True,
    )
    e1 = np.concatenate(denominators(eps1, 2, t["ref"]["madelung"], convention))
    assert abs(gamma_mp2(C1, e1, 2, B=t["B"])[0] - exact) < 1e-11


def test_mp2_anchor_detects_dropped_exchange_and_incomplete_aux(tri_anchor):
    """Mutations: exchange term dropped (measured +1.4e-6) and one aux class removed (+1.1e-8).
    Both are INVISIBLE on H2 (nocc = nvir = 1: (ib|ja) == (ia|jb); 7/8 classes: 4e-13)."""
    t = tri_anchor
    Co, Cv, eo, ev = t["C"][:, :2], t["C"][:, 2:], t["eps"][:2], t["eps"][2:]
    exact = gamma_mp2(t["C"], t["eps"], 2, eri=t["ref"]["I"])[0]
    _, e_os, _ = mp2_energy(ovov_from_B(t["B"], Co, Cv), eo, ev)
    assert abs(e_os - exact) > 1e-7
    cell, aux7 = _tri_anchor(classes=range(7))
    B7 = build_gdf(cell, None, auxmol=aux7)["B"]
    assert abs(mp2_energy(ovov_from_B(B7, Co, Cv), eo, ev)[0] - exact) > 1e-9


def test_mp2_is_structurally_blind_to_the_J3_g0_term(tri_anchor, monkeypatch):
    """Anchor blind spot, pinned so nobody reads the MP2 anchor as a G=0 check: J3's G=0 term is
    c0 S_mn q_P, and C_o^T S C_v = 0, so it never reaches B_ia (the HF ERI anchor catches it)."""
    t = tri_anchor
    monkeypatch.setattr(
        pbc_gdf, "_subtract_g0", lambda J2, J3, S, q, c0: (J2 - c0 * np.outer(q, q), J3)
    )
    Bm = build_gdf(t["cell"], None, auxmol=t["aux"])["B"]
    assert abs(eri_from_B(Bm) - t["ref"]["I"]).max() > 1.0  # the ERI is badly wrong ...
    exact = gamma_mp2(t["C"], t["eps"], 2, eri=t["ref"]["I"])[0]
    assert (
        abs(gamma_mp2(t["C"], t["eps"], 2, B=Bm)[0] - exact) < 1e-11
    )  # ... and ov-MP2 cannot see it


@pytest.mark.parametrize("convention", ["shifted", "unshifted"])
def test_mp2_matches_pinned_pyscf_rmp2(h2_ints, convention):
    e, eps, _, C = rhf(
        h2_ints["S"],
        h2_ints["h"],
        h2_ints["I"],
        h2_ints["enn"],
        2,
        conv=1e-12,
        return_mo=True,
    )
    eo, ev = denominators(eps, 1, h2_ints["madelung"], convention)
    assert (
        abs(
            gamma_mp2(C, np.concatenate([eo, ev]), 1, eri=h2_ints["I"])[0]
            - MP2_REF[convention]
        )
        < 1e-11
    )


def test_shifted_denominators_are_the_ewald_scf_eigenvalues(h2_ints):
    vm = h2_ints["madelung"]
    eps_n = rhf(
        h2_ints["S"], h2_ints["h"], h2_ints["I"], h2_ints["enn"], 2, conv=1e-12
    )[1]
    eps_e = rhf(
        h2_ints["S"],
        h2_ints["h"],
        h2_ints["I"],
        h2_ints["enn"],
        2,
        conv=1e-12,
        kshift=vm,
    )[1]
    assert (
        abs(np.concatenate(denominators(eps_n, 1, vm, "shifted")) - eps_e).max() < 1e-10
    )
    with pytest.raises(ValueError):
        denominators(eps_n, 1, vm, "madelung")


def test_mp2_box_limit_shifted_is_a3_with_predicted_coefficient_unshifted_is_1_over_a():
    """Independent of PySCF pbc: H2/STO-3G at a=20, 24 vs molecular MP2.
    shifted  : residual = c3/a^3, c3 predicted from molecular moments = 0.67109 (fit 0.67136)
    unshifted: residual - shifted residual = sum N/(D + 2 v_M) - sum N/D (molecular N, D),
               i.e. ~ -2 v_M sum N/D^2 = -0.0299/a.  Measured a=6..40 in FINDINGS.md Iteration 3."""
    from pyscf import mp, scf

    mol = gto.M(atom=H2_ATOMS, basis="sto-3g", unit="B", cart=True, verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-13
    mf.kernel()
    e_mol = mp.MP2(mf).kernel()[0]
    iaia = mol.ao2mo(mf.mo_coeff, compact=False).reshape(2, 2, 2, 2)[0, 1, 0, 1]
    N, D = iaia**2, 2 * (mf.mo_energy[0] - mf.mo_energy[1])
    res = {}
    for edge in (20.0, 24.0):
        w = 8.0 / edge
        ints = build_integrals(
            Cell(np.eye(3) * edge, H2_ATOMS, "sto-3g"),
            w,
            rcut_bra=18.0,
            rcut_2e=18.0 + 6.0 / w,
            exxdiv="ewald",
            verbose=False,
        )
        vm = ints["madelung"]
        _, eps, _, C = rhf(
            ints["S"], ints["h"], ints["I"], ints["enn"], 2, conv=1e-13, return_mo=True
        )
        r = {
            c: gamma_mp2(
                C, np.concatenate(denominators(eps, 1, vm, c)), 1, eri=ints["I"]
            )[0]
            - e_mol
            for c in ("shifted", "unshifted")
        }
        c3 = dipole_prediction_h2_minimal(mol, mf.mo_coeff, mf.mo_energy, edge)[0]
        assert abs(r["shifted"] - c3) < 0.01 * abs(
            c3
        )  # measured 0.5% at a=20, 0.3% at a=24
        shift_fn = N / (D + 2 * vm) - N / D
        assert abs((r["unshifted"] - r["shifted"]) - shift_fn) < 0.01 * abs(shift_fn)
        res[edge] = r
    p = np.log(res[20.0]["shifted"] / res[24.0]["shifted"]) / np.log(24.0 / 20.0)
    assert abs(p - 3.0) < 0.05


def test_mp2_formula_matches_molecular_pyscf_with_several_occupied():
    """The anchor compares B vs dense through the SAME mp2_energy, so it cannot see a formula
    bug; neither can H2 (nocc = nvir = 1). Pin the formula on H2O/6-31G (nocc 5, nvir 8)."""
    from pyscf import mp, scf

    mol = gto.M(
        atom="O 0 0 0.2; H 0 1.4 -0.9; H 0.1 -1.5 -0.8",
        basis="6-31g",
        unit="B",
        cart=True,
        verbose=0,
    )
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    ref = mp.MP2(mf).kernel()[0]
    I = mol.intor("int2e_cart")
    nocc = mol.nelectron // 2
    assert abs(gamma_mp2(mf.mo_coeff, mf.mo_energy, nocc, eri=I)[0] - ref) < 1e-11
    ref_fc = mp.MP2(mf, frozen=1).kernel()[0]
    assert (
        abs(gamma_mp2(mf.mo_coeff, mf.mo_energy, nocc, eri=I, frozen=1)[0] - ref_fc)
        < 1e-11
    )


# ============================================================ Gamma dRPA (pbc_rpa.py, Iteration 4)
from pbc_rpa import (  # noqa: E402
    direct_mp2,
    drpa_moment_prediction_h2_minimal,
    drpa_plasmon,
    drpa_quad,
    drpa_riccati,
    drpa_second_order_quad,
    gamma_drpa,
    ov_factor,
)
from pbc_mp2 import bia_from_B, ovov_from_eri  # noqa: E402

# Pinned (2026-09-24, run_rpa_anchor.py / run_rpa_oracle.py): H2/STO-3G a=4, pure-AFT, plasmon.
DRPA_REF = {"shifted": -6.9381306144e-03, "unshifted": -1.0000520140e-02}


@pytest.fixture(scope="module")
def h2o_mol():
    from pyscf import scf

    mol = gto.M(
        atom="O 0 0 0.2; H 0 1.4 -0.9; H 0.1 -1.5 -0.8",
        basis="6-31g",
        unit="B",
        cart=True,
        verbose=0,
    )
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    nocc = mol.nelectron // 2
    Co, Cv = mf.mo_coeff[:, :nocc], mf.mo_coeff[:, nocc:]
    ov = mol.ao2mo((Co, Cv, Co, Cv), compact=False).reshape(nocc, -1, nocc, Cv.shape[1])
    return dict(
        mol=mol, mf=mf, nocc=nocc, ov=ov, eo=mf.mo_energy[:nocc], ev=mf.mo_energy[nocc:]
    )


def test_drpa_three_constructions_agree(h2o_mol):
    """Plasmon (eigenvalues), ring-CCD Riccati (amplitudes), converged frequency quadrature on the
    factorised ov block: measured agreement 3e-14 / 1e-14 on H2O/6-31G (nocc 5, nvir 8)."""
    ov, eo, ev = h2o_mol["ov"], h2o_mol["eo"], h2o_mol["ev"]
    ep = drpa_plasmon(ov, eo, ev)
    assert abs(drpa_riccati(ov, eo, ev) - ep) < 1e-11
    assert abs(drpa_quad(ov_factor(ov), eo, ev) - ep) < 1e-11


def test_drpa_quadrature_matches_pyscf_molecular_rpa_on_its_own_df(h2o_mol):
    """Convention oracle: pyscf.gw.rpa.RPA (nw=40, x0=0.5) vs our GL quadrature, same nw/x0, on
    PySCF's own DF tensor.  Measured 6e-13.  (Spin factor 4, 1/2pi half-line, sign of Pi.)"""
    from pyscf import df, lib
    from pyscf.gw import rpa as prpa

    r = prpa.RPA(h2o_mol["mf"])
    r.with_df = df.DF(h2o_mol["mol"], auxbasis="cc-pvdz-ri")
    r.verbose = 0
    e_py = r.kernel(nw=40, x0=0.5)
    nocc, C = h2o_mol["nocc"], h2o_mol["mf"].mo_coeff
    Bia = np.einsum(
        "Pmn,mi,na->Pia",
        lib.unpack_tril(np.asarray(r.with_df._cderi)),
        C[:, :nocc],
        C[:, nocc:],
    )
    assert (
        abs(drpa_quad(Bia, h2o_mol["eo"], h2o_mol["ev"], n=40, x0=0.5) - e_py) < 1e-10
    )


def test_drpa_second_order_term_is_direct_mp2(h2o_mol):
    """-(1/2pi) int_0^inf tr Pi^2/2 == 2 sum (ia|jb)^2/Delta (measured 4e-15).  Pins the 4 and the
    1/2pi on the half line independently of PySCF; a 1/pi normalisation would fail by 100%."""
    ov, eo, ev = h2o_mol["ov"], h2o_mol["eo"], h2o_mol["ev"]
    dm = direct_mp2(ov, eo, ev)
    assert abs(drpa_second_order_quad(ov_factor(ov), eo, ev, n=400) - dm) < 1e-12
    lam = 1e-3  # and the full dRPA is dMP2 to O(V^2): E(lam V)/lam^2 -> dMP2, remainder O(lam)
    assert abs(drpa_plasmon(lam * ov, eo, ev) / lam**2 - dm) < 2e-3 * abs(dm)


@pytest.mark.parametrize("convention", ["shifted", "unshifted"])
def test_drpa_from_B_matches_dense_plasmon_in_the_trivial_aux_limit(
    tri_anchor, convention
):
    """Stage-7 anchor: B (RS-GDF, aux = every periodic pair product) + frequency quadrature vs the
    dense pure-AFT (ia|jb) + plasmon formula.  Measured 1.7e-13 / 3.3e-13 (same C), 1.3e-13 / 3e-13
    (orbitals from the B-tensor SCF)."""
    t = tri_anchor
    e = np.concatenate(denominators(t["eps"], 2, t["ref"]["madelung"], convention))
    exact = gamma_drpa(t["C"], e, 2, eri=t["ref"]["I"], method="plasmon")
    assert abs(gamma_drpa(t["C"], e, 2, B=t["B"]) - exact) < 1e-11
    _, eps1, _, C1 = rhf(
        t["ref"]["S"],
        t["ref"]["h"],
        None,
        t["ref"]["enn"],
        4,
        conv=1e-12,
        jk=jk_from_B(t["B"]),
        return_mo=True,
    )
    e1 = np.concatenate(denominators(eps1, 2, t["ref"]["madelung"], convention))
    assert abs(gamma_drpa(C1, e1, 2, B=t["B"]) - exact) < 1e-11


def test_drpa_anchor_detects_mutations(tri_anchor):
    """Measured on tri 4H: spin factor 4->2 +1.5e-2, exchange-type K -1.5e-2-level, Cv/Cv -6.3e-3,
    one aux class removed +2.1e-8 (invisible on H2: 6e-13)."""
    t = tri_anchor
    Co, Cv, eo, ev = t["C"][:, :2], t["C"][:, 2:], t["eps"][:2], t["eps"][2:]
    ov = ovov_from_eri(t["ref"]["I"], Co, Cv)
    exact = drpa_plasmon(ov, eo, ev)
    Bia = bia_from_B(t["B"], Co, Cv)
    assert abs(drpa_quad(Bia / np.sqrt(2), eo, ev) - exact) > 1e-3
    assert abs(drpa_plasmon(ov - 0.5 * ov.transpose(0, 3, 2, 1), eo, ev) - exact) > 1e-3
    assert (
        abs(drpa_quad(np.einsum("Pmn,mi,na->Pia", t["B"], Cv, Cv), eo, ev) - exact)
        > 1e-3
    )
    cell, aux7 = _tri_anchor(classes=range(7))
    B7 = build_gdf(cell, None, auxmol=aux7)["B"]
    assert abs(drpa_quad(bia_from_B(B7, Co, Cv), eo, ev) - exact) > 1e-9


@pytest.mark.parametrize("convention", ["shifted", "unshifted"])
def test_drpa_matches_pinned_h2_value(h2_ints, convention):
    """Pinned our value; PySCF AFTDF (ia|jb) + plasmon agrees (run_rpa_oracle.py, FINDINGS Iteration 4)."""
    e, eps, _, C = rhf(
        h2_ints["S"],
        h2_ints["h"],
        h2_ints["I"],
        h2_ints["enn"],
        2,
        conv=1e-12,
        return_mo=True,
    )
    ed = np.concatenate(denominators(eps, 1, h2_ints["madelung"], convention))
    assert (
        abs(
            gamma_drpa(C, ed, 1, eri=h2_ints["I"], method="plasmon")
            - DRPA_REF[convention]
        )
        < 1e-11
    )
    assert abs(gamma_drpa(C, ed, 1, eri=h2_ints["I"]) - DRPA_REF[convention]) < 1e-11


def test_drpa_box_limit_shifted_is_a3_with_predicted_coefficient_unshifted_is_shift_function():
    """Independent of PySCF pbc: H2/STO-3G at a=24, 32 vs molecular dRPA (exact ERI, plasmon).
    shifted  : residual -> c3/a^3, c3 = 0.92274 predicted from molecular moments (measured 0.923
               at a=40; within 0.4% at 24 and 32);
    unshifted: residual - shifted = E_mol(e_ia - v_M) - E_mol(e_ia) (measured ratio 0.9963 / 0.9985)."""
    from pyscf import scf

    mol = gto.M(atom=H2_ATOMS, basis="sto-3g", unit="B", cart=True, verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-13
    mf.kernel()
    ov = mol.ao2mo(mf.mo_coeff, compact=False).reshape(2, 2, 2, 2)[:1, 1:, :1, 1:]
    eo, ev = mf.mo_energy[:1], mf.mo_energy[1:]
    e_mol = drpa_plasmon(ov, eo, ev)
    c3 = drpa_moment_prediction_h2_minimal(mol, mf.mo_coeff, mf.mo_energy, 1.0)[0]
    res = {}
    for edge in (24.0, 32.0):
        w = 8.0 / edge
        ints = build_integrals(
            Cell(np.eye(3) * edge, H2_ATOMS, "sto-3g"),
            w,
            rcut_bra=18.0,
            rcut_2e=18.0 + 6.0 / w,
            exxdiv="ewald",
            verbose=False,
        )
        vm = ints["madelung"]
        _, eps, _, C = rhf(
            ints["S"], ints["h"], ints["I"], ints["enn"], 2, conv=1e-13, return_mo=True
        )
        r = {
            c: gamma_drpa(
                C,
                np.concatenate(denominators(eps, 1, vm, c)),
                1,
                eri=ints["I"],
                method="plasmon",
            )
            - e_mol
            for c in ("shifted", "unshifted")
        }
        assert abs(r["shifted"] * edge**3 - c3) < 0.006 * c3
        shift_fn = drpa_plasmon(ov, eo + vm, ev) - e_mol
        assert abs((r["unshifted"] - r["shifted"]) - shift_fn) < 0.005 * abs(shift_fn)
        res[edge] = r
    p = np.log(res[24.0]["shifted"] / res[32.0]["shifted"]) / np.log(32.0 / 24.0)
    assert abs(p - 3.0) < 0.05


def test_r2_kernel_c3_reproduces_both_closed_form_moment_models():
    """pbc_rpa.r2_kernel_c3 (molecular RHF/MP2/dRPA with the O(a^-3) lattice kernel (k/2)|r-r'|^2
    added to every Coulomb interaction, d/dk) must equal the closed-form frozen-orbital models where
    those apply (H2/STO-3G: orbitals fixed by symmetry): HF -(4pi/3) sigma^2 = -10.028245 (Iteration 1),
    MP2 0.671094 and dRPA 0.922741.  Box sweeps then measured 0.671358 / 0.923006 (c3+c5 tail fits)."""
    from pyscf import scf

    from pbc_mp2 import dipole_prediction_h2_minimal
    from pbc_rpa import r2_kernel_c3

    mol = gto.M(atom=H2_ATOMS, basis="sto-3g", unit="B", cart=True, verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-13
    mf.kernel()
    c = r2_kernel_c3(mol)
    s2 = dipole_prediction_h2_minimal(mol, mf.mo_coeff, mf.mo_energy, 1.0)[1]["sigma2"]
    assert abs(c["hf"] + 4 * np.pi / 3 * s2) < 1e-6
    assert (
        abs(
            c["mp2"]
            - dipole_prediction_h2_minimal(mol, mf.mo_coeff, mf.mo_energy, 1.0)[0]
        )
        < 1e-6
    )
    assert (
        abs(
            c["drpa"]
            - drpa_moment_prediction_h2_minimal(mol, mf.mo_coeff, mf.mo_energy, 1.0)[0]
        )
        < 1e-6
    )


# ============================================================ Gamma LMP2 (pbc_lmp2.py, Iteration 5)
import pbc_lmp2 as LM  # noqa: E402
from pbc_supercell import Supercell, build_supercell  # noqa: E402

LMP2_A0 = np.eye(3) * 7.0
_lu = np.array([0.5, 0.6, 1.2])
_lu = 1.4 * _lu / np.linalg.norm(_lu)
LMP2_ATOMS = [("H", (1.0, 1.2, 1.1)), ("H", tuple(np.array([1.0, 1.2, 1.1]) + _lu))]


def _wrap_last(nat):
    """Move the last atom by -a_sc(z): that molecule straddles the supercell boundary (Gamma-neutral)."""
    return lambda i, r: (0, 0, -1) if i == nat - 1 else (0, 0, 0)


def _lmp2_system(n, wrapped=False):
    sc = Supercell(
        LMP2_A0,
        LMP2_ATOMS,
        "6-31g",
        n,
        wrap=_wrap_last(2 * int(np.prod(n))) if wrapped else None,
    )
    d = build_supercell(sc, ferric_basis("cc-pvdz-ri", ["H"]), w=0.5)
    return sc, d, LM.gamma_scf(d, 2 * sc.R)


@pytest.fixture(scope="module")
def needle4():
    sc, d, scf = _lmp2_system((1, 1, 4), wrapped=True)
    return dict(
        sc=sc, d=d, scf=scf, ec=LM.canonical_mp2(scf, d["B"]), p=LM.prepare(sc, d, scf)
    )


def test_supercell_fold_matches_explicit_supercell_build_gdf_and_pyscf():
    """Folding primitive lattice sums by residue mod n == pbc_gdf.build_gdf on the explicit supercell
    (measured J2 1.8e-14, J3 1.2e-13 at (1,1,2)); S, T, E_nn vs PySCF pbc (1e-15).  n=3, not 2: at n=2
    c'-c == c-c' (mod 2), so a flipped residue sign was INVISIBLE here (mutation run 2026-09-24)."""
    sc = Supercell(LMP2_A0, LMP2_ATOMS, "6-31g", (1, 1, 3))
    aux = ferric_basis("cc-pvdz-ri", ["H"])
    f = build_supercell(sc, aux, w=0.5)
    g = build_gdf(sc.sc, aux, w=0.5)
    assert abs(f["J2"] - g["J2"]).max() < 1e-11 and abs(f["J3"] - g["J3"]).max() < 1e-11
    pc = pyscf_cell(sc.sc.a, sc.atoms, "6-31g")
    assert abs(f["S"] - pc.pbc_intor("int1e_ovlp")).max() < 1e-12
    assert abs(f["T"] - pc.pbc_intor("int1e_kin")).max() < 1e-12
    assert abs(f["enn"] - pc.energy_nuc()) < 1e-11


def test_resta_matrices_fold_matches_direct_pair_ft_on_a_wrapped_supercell():
    """Z_k = <mu|exp(i b_k.r)|nu> (lattice summed) from the residue fold vs pbc_gamma.pair_ft on the
    explicit supercell whose last atom is wrapped across the boundary (measured 1.4e-13)."""
    sc = Supercell(LMP2_A0, LMP2_ATOMS, "6-31g", (1, 1, 3), wrap=_wrap_last(6))
    d = build_supercell(sc, ferric_basis("cc-pvdz-ri", ["H"]), w=0.5)
    P0 = pair_ft(sc.sc, np.zeros((1, 3)))[..., 0].real
    nrm = np.sqrt(np.diag(d["S"]) / np.diag(P0))
    P = pair_ft(sc.sc, d["Gk"]) * nrm[:, None, None] * nrm[None, :, None]
    assert max(abs(np.conj(P[..., k]) - d["Zk"][k]).max() for k in range(3)) < 1e-11


def test_lmp2_eps0_matches_canonical_gamma_mp2_and_detects_a_dropped_hard_virtual(
    needle4,
):
    """Exactness anchor: Berghold LMOs + periodic VV-HV + masked CG at eps=0 == canonical shifted Gamma
    MP2 on the same B (measured 1.8e-16).  Mutation: drop one hard virtual -> +3.4e-3."""
    t = needle4
    assert t["p"]["dev_orth"] < 1e-10 and t["p"]["dev_span"] < 1e-10
    assert abs(LM.solve(t["p"], 0.0)["e"] - t["ec"]) < 1e-10
    pm = LM.prepare(t["sc"], t["d"], t["scf"], drop_hv=1)
    assert abs(LM.solve(pm, 0.0)["e"] - t["ec"]) > 1e-4


def test_pair_cutoff_trivial_limit_requires_minimum_image(needle4):
    """R* = max minimum-image centroid distance is the trivial radius: exact with min-image distances,
    wrong with raw distances (measured +1.6e-4, 14/16 pairs).  Blind spot pinned below: with n=2 along
    every axis each raw distance already IS a minimum image, so the mutation cannot fail there."""
    p, ec = needle4["p"], needle4["ec"]
    R = LM.pair_distances(p).max() + 1e-6
    assert abs(LM.solve(p, 0.0, pair_cut=R)["e"] - ec) < 1e-10
    assert abs(LM.solve(p, 0.0, pair_cut=R, raw=True)["e"] - ec) > 1e-5
    sc, d, scf = _lmp2_system((2, 2, 1))
    p2 = LM.prepare(sc, d, scf)
    R2 = LM.pair_distances(p2).max() + 1e-6
    assert (
        abs(
            LM.solve(p2, 0.0, pair_cut=R2, raw=True)["e"]
            - LM.canonical_mp2(scf, d["B"])
        )
        < 1e-10
    )


def test_periodic_domain_fit_trivial_limit_is_the_global_fit(needle4):
    """Per-pair local fit in the periodic metric (J2, J3) at the trivial aux radius == global B.B
    (measured 3e-16); raw distances drop aux across the boundary (5.8e-5)."""
    p, d = needle4["p"], needle4["d"]
    R = (
        LM.min_image(p["cen"][:, None, :] - d["aux_xyz"][None, :, :], p["a_sc"]).max()
        + 1e-6
    )
    assert abs(LM.domain_fit_J(p, d, R)[0] - p["J"]).max() < 1e-12
    assert abs(LM.domain_fit_J(p, d, R, raw=True)[0] - p["J"]).max() > 1e-6


def test_translation_equivalence_berghold_exact_molecular_boys_broken(needle4):
    """Equivalent molecules (one primitive translation apart, one of them straddling the boundary):
    Berghold LMOs/virtuals/pair energies equal to 1e-15, start-independent; the non-periodic
    molecular-Boys operator breaks it (measured LMO 3.2e-4, pair energies 1.2e-5) while the eps=0
    anchor cannot see it (unitary invariance)."""
    t = needle4
    r = LM.solve(t["p"], 1e-4)
    te = LM.translation_equivalence(t["sc"], t["d"], t["p"], r["epair"])
    assert (
        te["bijective"]
        and te["occ_dev"] < 1e-10
        and te["vir_dev"] < 1e-10
        and te["pair_dev"] < 1e-12
    )
    ps = LM.prepare(t["sc"], t["d"], t["scf"], seed=3)
    assert (
        abs(ps["loc"]["f"] - t["p"]["loc"]["f"]) < 1e-8
        and abs(LM.solve(ps, 1e-4)["e"] - r["e"]) < 1e-12
    )
    pb = LM.prepare(t["sc"], t["d"], t["scf"], loc="boys-molecular")
    rb = LM.solve(pb, 1e-4)
    tb = LM.translation_equivalence(t["sc"], t["d"], pb, rb["epair"])
    assert tb["occ_dev"] > 1e-6 and tb["pair_dev"] > 1e-8
    assert abs(LM.solve(pb, 0.0)["e"] - t["ec"]) < 1e-10  # the blind spot


def test_ragged_solver_matches_dense_on_the_same_mask(needle4):
    p = needle4["p"]
    for eps, rc in ((1e-3, None), (0.0, 10.5)):
        assert (
            abs(
                LM.solve(p, eps, pair_cut=rc)["e"]
                - LM.solve(p, eps, pair_cut=rc, solver="ragged")["e"]
            )
            < 1e-12
        )


def test_far_pairs_couple_through_the_uniform_field_and_set_the_eps_onset():
    """Gamma P2: far pairs carry (ia|jb) = -(4pi/Omega_sc) mu_ia,z mu_jb,z (needle, G_par=0 dipole sheets);
    measured rel. deviation 1.0e-2 at N=8 (4.6e-3 at 12, 6.5e-4 at 32).  So the eps=3e-3 mask keeps every
    pair until N*=4pi mu*^2/(Omega_p eps)=7.7: all 36 pairs at N=6, 3 partners/molecule at N=8."""
    kept = {}
    for N in (6, 8):
        sc, d, scf = _lmp2_system((1, 1, N))
        p = LM.prepare(sc, d, scf)
        mu = LM.transition_dipoles_z(p, d)
        nstar = 4 * np.pi * abs(mu).max() ** 2 / (343.0 * 3e-3)
        assert 7.0 < nstar < 8.5
        r = LM.solve(p, 3e-3, solver="ragged")
        assert (
            r["partners"].min() == r["partners"].max()
        )  # translation-equivalent partner counts
        kept[N] = int(r["partners"][0])
        if N == 8:
            dist = LM.pair_distances(p)
            i, j = np.unravel_index(np.argmax(dist), dist.shape)
            pred = -(4 * np.pi / sc.sc.vol) * np.outer(mu[i], mu[j])
            assert (
                abs(p["J"][i, :, j, :] - pred).max()
                < 0.02 * abs(p["J"][i, :, j, :]).max()
            )
    assert kept == {6: 6, 8: 3}


def test_resta_weights_refuse_non_orthorhombic_cells():
    with pytest.raises(NotImplementedError):
        LM.resta_weights(TRI_A)


# ============================================================ Gamma UHF (pbc_uhf.py, Iteration 6)
import pbc_uhf  # noqa: E402
from pbc_uhf import uhf, uhf_c3_closed_form, uhf_r2_kernel_c3  # noqa: E402

# PySCF 2.13 pbc.scf.UHF on AFTDF (mesh 61^3), measured 2026-09-24 by run_uhf_oracle.py (agreement <= 1.3e-12).
UHF_REF = {
    "H atom a=4": {"none": -0.402177788224, "ewald": -0.756839973159},  # <S2> 0.75
    "H2 a=4 triplet": {"none": 0.322229103842, "ewald": -0.387095266028},  # <S2> 2
}
H_ATOM = [("H", (0.3, 0.2, 0.1))]
O2_ATOMS = [("O", (0.3, 0.2, 0.1)), ("O", (0.3, 0.2, 2.382))]


def _k_total(jk, Da, Db):  # MUTANT: exchange built from the total density
    J, K = jk(Da + Db)
    return J, K, K


def _madelung_half(
    S, Ds, vm
):  # MUTANT: RHF's D/2 bookkeeping applied to a spin density
    return 0.5 * vm * S @ Ds @ S


@pytest.mark.parametrize("exxdiv", ["none", "ewald"])
def test_uhf_closed_shell_equals_rhf(h2_ints, tri_anchor, exxdiv):
    """Anchor (a): na == nb through UHF == pbc_gamma.rhf (measured dE 0, 1e-15)."""
    for I, nel in ((h2_ints, 2), (tri_anchor["ref"], 4)):
        vm = I["madelung"] if exxdiv == "ewald" else 0.0
        er = rhf(I["S"], I["h"], I["I"], I["enn"], nel, conv=1e-12, kshift=vm)[0]
        u = uhf(
            I["S"], I["h"], I["I"], I["enn"], nel // 2, nel // 2, conv=1e-12, kshift=vm
        )
        assert abs(u["e"] - er) < 1e-11 and abs(u["s2"]) < 1e-10


def test_uhf_closed_shell_anchor_fails_for_k_from_total_density(h2_ints, monkeypatch):
    """Measured -4.5e-2 (H2 a=4)."""
    I = h2_ints
    er = rhf(I["S"], I["h"], I["I"], I["enn"], 2, conv=1e-12)[0]
    monkeypatch.setattr(pbc_uhf, "_jk_spin", _k_total)
    assert abs(uhf(I["S"], I["h"], I["I"], I["enn"], 1, 1, conv=1e-12)["e"] - er) > 1e-2


@pytest.mark.parametrize("exxdiv", ["none", "ewald"])
def test_uhf_open_shell_dense_equals_rsgdf_in_the_trivial_aux_limit(
    tri_anchor, exxdiv, monkeypatch
):
    """Anchor (b): triplet (na 3, nb 1) on the one-s tri cell, B with aux = every periodic pair product vs
    the dense pure-AFT ERI (measured -1.1e-11 both conventions).  K[D_total] through B: -2.4e-1.
    Blind spot: a wrong per-spin Madelung factor is applied identically on both sides -> invisible here."""
    t = tri_anchor
    I = t["ref"]
    vm = I["madelung"] if exxdiv == "ewald" else 0.0
    ud = uhf(I["S"], I["h"], I["I"], I["enn"], 3, 1, conv=1e-12, kshift=vm)
    ub = uhf(
        I["S"],
        I["h"],
        None,
        I["enn"],
        3,
        1,
        conv=1e-12,
        kshift=vm,
        jk=jk_from_B(t["B"]),
    )
    assert (
        abs(ub["e"] - ud["e"]) < 1e-10
        and abs(ub["s2"] - ud["s2"]) < 1e-9
        and ud["s2"] > 2.0001
    )
    monkeypatch.setattr(pbc_uhf, "_jk_spin", _k_total)
    assert (
        abs(
            uhf(
                I["S"],
                I["h"],
                None,
                I["enn"],
                3,
                1,
                conv=1e-12,
                kshift=vm,
                jk=jk_from_B(t["B"]),
            )["e"]
            - ud["e"]
        )
        > 1e-2
    )


@pytest.fixture(scope="module")
def h_atom_ints():
    return build_integrals(
        Cell(H2_A, H_ATOM, "sto-3g"), None, exxdiv="ewald", verbose=False
    )


@pytest.mark.parametrize("exxdiv", ["none", "ewald"])
def test_uhf_matches_pinned_pyscf_aftdf(h_atom_ints, h2_ints, exxdiv):
    for key, I, na, nb, s2 in (
        ("H atom a=4", h_atom_ints, 1, 0, 0.75),
        ("H2 a=4 triplet", h2_ints, 2, 0, 2.0),
    ):
        vm = I["madelung"] if exxdiv == "ewald" else 0.0
        u = uhf(I["S"], I["h"], I["I"], I["enn"], na, nb, conv=1e-12, kshift=vm)
        assert abs(u["e"] - UHF_REF[key][exxdiv]) < 1e-10 and abs(u["s2"] - s2) < 1e-10


def test_uhf_madelung_is_vM_per_spin_density(h_atom_ints, h2_ints, monkeypatch):
    """PySCF adds v_M S D_s S to EACH spin's K (df_jk._ewald_exxdiv_for_G0 loops over dms): ewald - none
    = -v_M (Na+Nb)/2 exactly at Gamma.  The half-factor mutant misses the pinned ewald E by v_M N/4."""
    for key, I, na, nb in (
        ("H atom a=4", h_atom_ints, 1, 0),
        ("H2 a=4 triplet", h2_ints, 2, 0),
    ):
        vm = I["madelung"]
        e = {
            x: uhf(I["S"], I["h"], I["I"], I["enn"], na, nb, conv=1e-12, kshift=k)["e"]
            for x, k in (("n", 0.0), ("e", vm))
        }
        assert abs(e["e"] - e["n"] + vm * (na + nb) / 2) < 1e-11
    monkeypatch.setattr(pbc_uhf, "_madelung_term", _madelung_half)
    I = h2_ints
    assert (
        abs(
            uhf(
                I["S"], I["h"], I["I"], I["enn"], 2, 0, conv=1e-12, kshift=I["madelung"]
            )["e"]
            - UHF_REF["H2 a=4 triplet"]["ewald"]
        )
        > 0.1
    )


def _box_residual(atoms, na, nb, edge, guess):
    w = min(1.0, 8.0 / edge)
    I = build_integrals(
        Cell(np.eye(3) * edge, atoms, "sto-3g"),
        w,
        rcut_bra=18.0,
        rcut_2e=18.0 + 6.0 / w,
        exxdiv="ewald",
        verbose=False,
    )
    return {
        x: uhf(
            I["S"], I["h"], I["I"], I["enn"], na, nb, conv=1e-12, kshift=k, guess=guess
        )
        for x, k in (("none", 0.0), ("ewald", I["madelung"]))
    }, I["madelung"]


def _mol_uhf(atoms, spin):
    from pyscf import scf

    mol = gto.M(atom=atoms, basis="sto-3g", unit="B", cart=True, verbose=0, spin=spin)
    mf = scf.UHF(mol)
    mf.conv_tol = 1e-13
    mf.kernel()
    return mol, mf


def test_uhf_box_limit_h_atom_is_exactly_c3_over_a3():
    """Independent of PySCF pbc.  H atom (one electron: Hartree == self-exchange for ANY kernel; spherical
    density sees no l=4 cubic term) => residual is c3/a^3 up to exponentially small image overlap.
    c3 = -(2pi/3) Omega_a = -4.081081 predicted; measured dE*a^3 -4.081081 at a=20..40 (-4.344 at a=10)."""
    mol, mf = _mol_uhf(H_ATOM, 1)
    c3 = uhf_c3_closed_form(mol, *mf.mo_coeff, 1, 0)[0]
    r, vm = _box_residual(H_ATOM, 1, 0, 20.0, mf.make_rdm1())
    assert abs((r["ewald"]["e"] - mf.e_tot) * 20.0**3 - c3) < 1e-5 * abs(c3)
    assert (
        abs(r["none"]["e"] - r["ewald"]["e"] - vm / 2) < 1e-12
    )  # none: +v_M N/2 = +1.4186/a


def test_uhf_box_limit_o2_triplet_ewald_is_a3_with_predicted_coefficient():
    """O2/STO-3G triplet (na 9, nb 7).  c3 = -(2pi/3)(|d|^2 + Omega_a + Omega_b) = -30.6224 predicted
    (closed form == relaxed r2-kernel FD); measured dE*a^3 -30.6886 / -30.6595 at a=24 / 32 (c5 tail),
    tail fit (24,32,40) c3 = -30.6221, exponent 3.0033 between 24 and 32.  A per-spin factor-1/2 Madelung
    would add +v_M N/4 (exponent 1)."""
    mol, mf = _mol_uhf(O2_ATOMS, 2)
    c3 = uhf_c3_closed_form(mol, *mf.mo_coeff, 9, 7)[0]
    assert abs(uhf_r2_kernel_c3(mol, 9, 7, guess=mf.make_rdm1()) - c3) < 1e-5 * abs(c3)
    res = {}
    for edge in (24.0, 32.0):
        r, vm = _box_residual(O2_ATOMS, 9, 7, edge, mf.make_rdm1())
        res[edge] = r["ewald"]["e"] - mf.e_tot
        assert abs(res[edge] * edge**3 - c3) < 0.005 * abs(c3)
        assert abs(r["none"]["e"] - r["ewald"]["e"] - vm * 8) < 1e-11
        assert abs(r["ewald"]["s2"] - mf.spin_square()[0]) < 1e-4
    p = np.log(res[24.0] / res[32.0]) / np.log(32.0 / 24.0)
    assert abs(p - 3.0) < 0.02


@pytest.mark.skipif(not SLOW, reason="set PBC_SLOW=1 (~70 s pure-AFT tri s+p build)")
def test_uhf_ewald_core_guess_traps_in_a_non_aufbau_state_none_first_does_not():
    """tri 4H s+p triplet.  Ewald and None have identical stationary points (v_M S D S = v_M x occupied
    projector), but ewald lowers occupied levels by v_M, so a None-non-aufbau state becomes ewald-stable:
    core guess -> E -1.812958714837 (alpha gap 0.617 < v_M 0.622); None first then ewald -> -1.827999723359
    == PySCF default guess (1.5e-2 lower)."""
    I = build_integrals(
        Cell(TRI_A, TRI_ATOMS, SP_BASIS), None, exxdiv="ewald", verbose=False
    )
    vm, args = I["madelung"], (I["S"], I["h"], I["I"], I["enn"])
    trapped = uhf(*args, 3, 1, conv=1e-12, kshift=vm)
    assert (
        abs(trapped["e"] - -1.812958714837) < 1e-9
        and trapped["eps_a"][3] - trapped["eps_a"][2] < vm
    )
    un = uhf(*args, 3, 1, conv=1e-12)
    good = uhf(*args, 3, 1, conv=1e-12, kshift=vm, guess=(un["Da"], un["Db"]))
    assert (
        abs(good["e"] - -1.827999723359) < 1e-10
        and abs(good["e"] - un["e"] + 2 * vm) < 1e-11
    )
    assert good["eps_a"][3] - good["eps_a"][2] > vm


# ================================================= Gamma LMP2 uniform (q=0 head) coupling (Iteration 5b)
@pytest.fixture(scope="module")
def needle_unif():
    """(1,1,4) wrapped (needle4) and (1,1,8) straight, with Richardson mu, J_unif and the head-restored refs."""
    out = {}
    for N in (4, 8):
        if N == 4:
            sc, d, scf = _lmp2_system((1, 1, 4), wrapped=True)
        else:
            sc, d, scf = _lmp2_system((1, 1, N))
        p = LM.prepare(sc, d, scf)
        Z2 = LM.resta_z_at(sc, d, 2, 2)
        mu, mi = LM.transition_dipoles_richardson(sc, d, p, Z2=Z2)
        Ju = LM.uniform_coupling(p, mu)
        Foo2, Fvv2, _ = LM.fock_head_correction(sc, d, p, mu, Z2=Z2)
        out[N] = dict(
            sc=sc,
            d=d,
            scf=scf,
            p=p,
            Z2=Z2,
            mu=mu,
            f1=mi["f1"],
            Ju=Ju,
            eu=LM.uniform_pair_energies(p, Ju),
            ec=LM.canonical_mp2(scf, d["B"]),
            eh=LM.canonical_mp2_from_local(p, scf, d["S"], p["J"] - Ju),
            ehf=LM.mp2_local_closed_form(p["J"] - Ju, Foo2, Fvv2),
        )
    return out


def test_uniform_coupling_richardson_dipoles_and_eps0_anchors(needle_unif):
    """resta_z_at(m=1) == build_supercell's Zk; the (b, 2b) Richardson dipole makes the farthest pair's
    J - J_unif small (measured N=8: 4.4e-4 rel; the single-b Resta estimate 1.0e-2).  eps=0 anchors: the
    J-unif gate + add-back == canonical Gamma MP2 (nothing dropped, add-back 0), CG on J' == the closed-form
    canonical MP2 of J' rotated to canonical orbitals (independent construction), and the closed-form local
    rotation of J itself == canonical_mp2."""
    t = needle_unif[8]
    p, d, sc = t["p"], t["d"], t["sc"]
    assert abs(LM.resta_z_at(sc, d, 2, 1) - d["Zk"][2]).max() < 1e-12
    dist = LM.pair_distances(p)
    i, j = np.unravel_index(np.argmax(dist), dist.shape)
    Jf = p["J"][i, :, j, :]
    rel = abs(Jf - t["Ju"][i, :, j, :]).max() / abs(Jf).max()
    rel1 = abs(Jf - LM.uniform_coupling(p, t["f1"])[i, :, j, :]).max() / abs(Jf).max()
    assert rel < 2e-3 < 5e-3 < rel1
    r0 = LM.solve(p, 0.0, gate="J-unif", J_unif=t["Ju"], addback=t["eu"])
    assert abs(r0["e"] - t["ec"]) < 1e-10 and r0["e_addback"] == 0.0
    assert abs(LM.solve(p, 0.0, J=p["J"] - t["Ju"])["e"] - t["eh"]) < 1e-10
    assert abs(LM.mp2_local_closed_form(p["J"], p["Foo"], p["Fvv"]) - t["ec"]) < 1e-12


def test_head_restoration_shrinks_the_gamma_finite_size_term(needle_unif):
    """E = e N + b from N = 4, 8 (b = 2 E4 - E8).  Canonical Gamma MP2 b +4.26e-3; restoring the q=0 head in
    the ERIs (J' = J - J_unif) b +1.07e-3; also in the Fock exchange (Fvv, diag Foo) b +1.6e-4.  The uniform
    coupling is the q=0 quadrature hole, a finite-size term, not correlation that survives the TDL.
    Mutation guard: a flipped Fock-head sign or J' = J + J_unif makes b GROW."""
    b = {k: 2 * needle_unif[4][k] - needle_unif[8][k] for k in ("ec", "eh", "ehf")}
    assert b["ec"] > 3e-3 and 0.5e-3 < b["eh"] < 1.6e-3 and 0 < b["ehf"] < 0.1 * b["ec"]
    # e_inf equality is NOT asserted from two small N: canonical carries c/N = +3.1e-4/N (tail fit), which
    # moves a 4/8 two-point slope by 8e-6; the 16/24/32 fits agree to 3e-9 (FINDINGS Iteration 5b).


def test_uniform_gate_removes_the_onset_and_the_addback_is_needed(needle_unif):
    """N=8.  gate on J at eps 1e-4 keeps all 64 pairs (below N* = 232); gated on J - J_unif it keeps 3
    partners/molecule (the R_c 10.5 set).  Pair-level screen on J' (energy estimate 1e-7) + analytic add-back
    reproduces canonical Gamma MP2 to 2.4e-6; WITHOUT the add-back +6.8e-4 (the dropped uniform energy).
    Solving on J' (A-drop) reproduces its own reference E_head to 2.3e-6 with no add-back."""
    t = needle_unif[8]
    p, Ju, eu, ec = t["p"], t["Ju"], t["eu"], t["ec"]
    assert LM.solve(p, 1e-4, solver="ragged")["pairs_kept"] == 64
    ra = LM.solve(p, 1e-4, gate="J-unif", J_unif=Ju, addback=eu, solver="ragged")
    assert ra["partners"].min() == ra["partners"].max() == 3
    rb = LM.solve(p, 0.0, gate="J-unif", J_unif=Ju, pair_ecut=1e-7, addback=eu)
    assert (
        rb["pairs_kept"] == 24
        and abs(rb["e"] - ec) < 1e-5
        and rb["e_solve"] - ec > 5e-4
    )
    rc = LM.solve(p, 0.0, pair_cut=10.5, addback=eu)
    assert abs(rc["e"] - rb["e"]) < 1e-12  # same pair set, same energy
    rd = LM.solve(p, 1e-4, J=p["J"] - Ju, solver="ragged")
    assert rd["partners"].max() == 3 and 0 < rd["e"] - t["eh"] < 1e-5
    with pytest.raises(ValueError):
        LM.solve(p, 1e-4, gate="J-unif")


# ================================================================ Gamma UMP2 / URPA (Iteration 7)
import pbc_ump2 as UM  # noqa: E402
from pbc_rpa import drpa_quad as _drpa_quad, gamma_drpa as _gamma_drpa  # noqa: E402

PENTA_ATOMS = TRI_ATOMS + [("H", (1.3, 3.1, 4.0))]
# PySCF 2.13 pbc.mp.UMP2 on pbc.scf.UHF + AFTDF (mesh 61^3), measured 2026-09-24 by run_ump2_oracle.py.
# 'urpa' = PySCF's own periodic AFTDF (ia|jb) blocks + our spin-orbital plasmon (PySCF has no periodic RPA).
UMP2_REF = {
    "penta one-s doublet": {
        "unshifted": (-2.167566950256e-02, -3.602910454467e-02),
        "shifted": (-1.179101239432e-02, -2.150209066693e-02),
    },  # ours - PySCF <= 1.2e-12
    "H2 6-31g triplet": {
        "unshifted": (-8.878036761172e-04, -3.225957845588e-02),
        "shifted": (-5.712132378726e-04, -1.133223577905e-02),
    },  # <= 2.3e-14 (URPA 9e-12)
    "tri 4H s+p triplet": {
        "unshifted": (-3.332802589058e-02, -6.935114524772e-02),
        "shifted": (-2.278123432187e-02, -4.834534740428e-02),
    },  # <= 4.5e-10 (SCF orbitals)
}


def _penta_anchor():
    """5 one-s H in the tri cell, doublet (na 3, nb 2): aa, bb AND ab UMP2 blocks all nonzero (the tri triplet has a
    single alpha virtual, so its aa block vanishes).  Aux = all 15 x 8 periodic pair products (exact span)."""
    cell = Cell(TRI_A, PENTA_ATOMS, {"H": [[0, [ANCHOR_ALPHA, 1.0]]]})
    n = len(PENTA_ATOMS)
    cen = [
        0.5 * (cell.R[i] + cell.R[j] + np.array(h) @ cell.a)
        for i in range(n)
        for j in range(i, n)
        for h in np.ndindex(2, 2, 2)
    ]
    aux = gto.M(
        atom=[("X", c) for c in cen],
        basis={"X": [[0, [2 * ANCHOR_ALPHA, 1.0]]]},
        unit="B",
        cart=True,
        verbose=0,
    )
    return cell, aux


@pytest.fixture(scope="module")
def penta():
    cell, aux = _penta_anchor()
    I = build_integrals(cell, None, exxdiv="ewald", verbose=False)
    u = uhf(I["S"], I["h"], I["I"], I["enn"], 3, 2, conv=1e-12)
    u = uhf(
        I["S"],
        I["h"],
        I["I"],
        I["enn"],
        3,
        2,
        conv=1e-16,
        maxiter=400,
        guess=(u["Da"], u["Db"]),
    )
    # prec 1e-15: at the default 1e-13 the SR truncation leaves max|dI| 3.5e-10 -> dUMP2 1e-11
    return dict(I=I, u=u, B=build_gdf(cell, None, auxmol=aux, prec=1e-15)["B"])


def _swap_bia(
    B, Ca, Cb, na, nb, frozen=0
):  # MUTANT: alpha orbitals used for the beta B_ia
    return UM._bia(B, Ca, na, frozen), UM._bia(B, Ca, nb, frozen)


@pytest.mark.parametrize("convention", ["shifted", "unshifted"])
def test_ump2_urpa_closed_shell_equal_restricted(
    h2_ints, tri_anchor, convention, monkeypatch
):
    """Anchor (a): na == nb through the U path == pbc_mp2 / pbc_rpa (measured <= 2.2e-16 same C; <= 1e-11 with the UHF's
    own orbitals).  Per-spin factor 4 (the closed-shell factor on each spin channel): -1.8e-2 (H2) / -5.3e-2 (tri)."""
    for I, n, B in ((h2_ints, 1, None), (tri_anchor["ref"], 2, tri_anchor["B"])):
        _, eps, _, C = rhf(
            I["S"], I["h"], I["I"], I["enn"], 2 * n, conv=1e-12, return_mo=True
        )
        e = np.concatenate(denominators(eps, n, I["madelung"], convention))
        den = UM.u_denominators(eps, eps, n, n, I["madelung"], convention)
        rm = gamma_mp2(C, e, n, eri=I["I"])[0]
        rr = _gamma_drpa(C, e, n, eri=I["I"], method="plasmon")
        assert abs(UM.gamma_ump2(C, C, den, n, n, eri=I["I"])[0] - rm) < 1e-12
        for m in ("plasmon", "quad"):
            assert (
                abs(UM.gamma_urpa(C, C, den, n, n, eri=I["I"], method=m) - rr) < 1e-12
            )
        if B is not None:
            rbq = _drpa_quad(bia_from_B(B, C[:, :n], C[:, n:]), e[:n], e[n:])
            assert abs(UM.gamma_urpa(C, C, den, n, n, B=B) - rbq) < 1e-12
        monkeypatch.setattr(UM, "SPIN_FACTOR", 4.0)
        assert (
            abs(UM.gamma_urpa(C, C, den, n, n, eri=I["I"], method="quad") - rr) > 1e-2
        )
        monkeypatch.setattr(UM, "SPIN_FACTOR", 2.0)


@pytest.mark.parametrize("convention", ["shifted", "unshifted"])
def test_ump2_urpa_open_shell_B_matches_dense_in_the_trivial_aux_limit(
    tri_anchor, penta, convention, monkeypatch
):
    """Anchor (b): tri one-s TRIPLET (na 3, nb 1; ab only) and penta DOUBLET (na 3, nb 2; aa, bb, ab), trivial-aux B vs
    dense pure-AFT ERI, same C: measured UMP2 <= 1.2e-14, URPA B-quad vs dense plasmon <= 2.1e-13.  Mutants: alpha C
    for the beta B_ia (UMP2 6e-4..1.7e-3, URPA 1e-4..2e-3); per-spin factor 4 (URPA 2.7e-2..8.5e-2)."""
    I3 = tri_anchor["ref"]
    u3 = uhf(I3["S"], I3["h"], I3["I"], I3["enn"], 3, 1, conv=1e-12)
    for I, u, B, na, nb in (
        (I3, u3, tri_anchor["B"], 3, 1),
        (penta["I"], penta["u"], penta["B"], 3, 2),
    ):
        Ca, Cb = u["Ca"], u["Cb"]
        den = UM.u_denominators(
            u["eps_a"], u["eps_b"], na, nb, I["madelung"], convention
        )
        em = UM.gamma_ump2(Ca, Cb, den, na, nb, eri=I["I"])[0]
        er = UM.gamma_urpa(Ca, Cb, den, na, nb, eri=I["I"], method="plasmon")
        assert abs(UM.gamma_ump2(Ca, Cb, den, na, nb, B=B)[0] - em) < 1e-12
        assert abs(UM.gamma_urpa(Ca, Cb, den, na, nb, B=B) - er) < 1e-11
        with monkeypatch.context() as mp_:
            mp_.setattr(UM, "u_bia", _swap_bia)
            assert abs(UM.gamma_ump2(Ca, Cb, den, na, nb, B=B)[0] - em) > 1e-4
            assert abs(UM.gamma_urpa(Ca, Cb, den, na, nb, B=B) - er) > 1e-5
        with monkeypatch.context() as mp_:
            mp_.setattr(UM, "SPIN_FACTOR", 4.0)
            assert abs(UM.gamma_urpa(Ca, Cb, den, na, nb, B=B) - er) > 1e-2
    # the penta blocks: all three nonzero, so a dropped same-spin 1/2 or dropped exchange is visible
    _, eaa, ebb, eab = UM.gamma_ump2(Ca, Cb, den, na, nb, eri=I["I"])
    assert eaa < -1e-7 and ebb < -1e-6 and eab < -1e-3
    assert abs(UM.direct_ump2(UM.u_ovov(Ca, Cb, na, nb, eri=I["I"]), *den) - em) > 1e-3


def test_urpa_second_order_term_is_direct_ump2(penta, monkeypatch):
    """Anchor (c): -(1/2pi) int tr Pi^2/2 == 1/2 sum_aa (ia|jb)^2/D + 1/2 sum_bb + sum_ab (measured 1.2e-14 / 2.4e-14),
    pinning the per-spin factor 2 and the half-line normalisation without PySCF.  Factor-4 mutant: -7.4e-2."""
    I, u = penta["I"], penta["u"]
    den = UM.u_denominators(u["eps_a"], u["eps_b"], 3, 2, I["madelung"], "shifted")
    Ba, Bb = UM.u_bia(penta["B"], u["Ca"], u["Cb"], 3, 2)
    dm = UM.direct_ump2(UM.u_ovov(u["Ca"], u["Cb"], 3, 2, eri=I["I"]), *den)
    assert abs(UM.urpa_second_order_quad(Ba, Bb, *den) - dm) < 1e-12
    monkeypatch.setattr(UM, "SPIN_FACTOR", 4.0)
    assert abs(UM.urpa_second_order_quad(Ba, Bb, *den) - dm) > 1e-2


def _pin_ump2(I, na, nb, key):
    un = uhf(I["S"], I["h"], I["I"], I["enn"], na, nb, conv=1e-12)
    un = uhf(
        I["S"],
        I["h"],
        I["I"],
        I["enn"],
        na,
        nb,
        conv=1e-16,
        maxiter=400,
        guess=(un["Da"], un["Db"]),
    )
    out = {}
    for c in ("unshifted", "shifted"):
        den = UM.u_denominators(un["eps_a"], un["eps_b"], na, nb, I["madelung"], c)
        out[c] = (
            UM.gamma_ump2(un["Ca"], un["Cb"], den, na, nb, eri=I["I"])[0],
            UM.gamma_urpa(
                un["Ca"], un["Cb"], den, na, nb, eri=I["I"], method="plasmon"
            ),
        )
    return out


def test_ump2_matches_pinned_pyscf_pbc_ump2_penta(penta):
    """exxdiv=None -> unshifted, exxdiv='ewald' -> shifted (PySCF pbc.mp.UMP2 takes mf.mo_energy; pbc UCCSD's MP2 is
    ALWAYS shifted, per spin).  The shifted/unshifted pair differ by 1e-2, so a convention swap cannot pass."""
    got = _pin_ump2(penta["I"], 3, 2, "penta one-s doublet")
    for c, (m, r) in UMP2_REF["penta one-s doublet"].items():
        assert abs(got[c][0] - m) < 1e-11 and abs(got[c][1] - r) < 1e-11


@pytest.mark.skipif(
    not SLOW, reason="set PBC_SLOW=1 (~70 s H2/6-31G + ~45 s tri s+p pure-AFT builds)"
)
def test_ump2_matches_pinned_pyscf_pbc_ump2_h2_triplet_and_tri_sp_triplet():
    """H2/6-31G a=4 triplet (aa only; STO-3G would be identically 0) and tri 4H s+p triplet (aa + ab)."""
    for key, a, atoms, basis, na, nb, tol in (
        ("H2 6-31g triplet", H2_A, H2_ATOMS, "6-31g", 2, 0, 1e-11),
        ("tri 4H s+p triplet", TRI_A, TRI_ATOMS, SP_BASIS, 3, 1, 1e-9),
    ):
        got = _pin_ump2(
            build_integrals(Cell(a, atoms, basis), None, exxdiv="ewald", verbose=False),
            na,
            nb,
            key,
        )
        for c, (m, r) in UMP2_REF[key].items():
            assert abs(got[c][0] - m) < tol and abs(got[c][1] - r) < tol


NH_ATOMS = [("N", (0.3, 0.2, 0.1)), ("H", (0.3, 0.2, 0.1 + 1.95))]


def _ump2_box(atoms, na, nb, edge, dm):
    w = min(1.0, 8.0 / edge)
    I = build_integrals(
        Cell(np.eye(3) * edge, atoms, "6-31g"),
        w,
        rcut_bra=18.0,
        rcut_2e=18.0 + 6.0 / w,
        exxdiv="ewald",
        verbose=False,
    )
    u = uhf(I["S"], I["h"], I["I"], I["enn"], na, nb, conv=1e-12, guess=dm)
    return u, I


def test_ump2_urpa_box_limit_nh_triplet_is_a3_with_predicted_c3_and_per_spin_shift():
    """NH/6-31G triplet (5,3), shifted = per-spin eps_occ,s - v_M.  Predicted (molecular harmonic kernel, UHF relaxed):
    c3 UMP2 0.761014, URPA 1.502486.  Measured (32,40) c3+c5 fit: 0.761792 (+1.0e-3) / 1.503251 (+5.1e-4); local exponent
    3.019 / 3.013 at 32->40.  MUTANT denominators leave 1/a: v_M/2 per spin dE*a -0.0545 / -0.063; alpha-only
    -0.0439 / -0.062 (at a=40 these are 30-60x the shifted residual)."""
    from run_ump2_box_limit import molecular

    mol, mf, e_mp2, e_rpa, _ = molecular(NH_ATOMS, 5, 3)
    c3 = UM.u_r2_kernel_c3(mol, 5, 3, guess=mf.make_rdm1())
    res = {}
    for edge in (32.0, 40.0):
        u, I = _ump2_box(NH_ATOMS, 5, 3, edge, mf.make_rdm1())
        vm, (ea, eb), (Ca, Cb) = (
            I["madelung"],
            (u["eps_a"], u["eps_b"]),
            (u["Ca"], u["Cb"]),
        )
        den = UM.u_denominators(ea, eb, 5, 3, vm, "shifted")
        res[edge] = (
            UM.gamma_ump2(Ca, Cb, den, 5, 3, eri=I["I"])[0] - e_mp2,
            UM.gamma_urpa(Ca, Cb, den, 5, 3, eri=I["I"], method="plasmon") - e_rpa,
        )
        for sa, sb in ((vm / 2, vm / 2), (vm, 0.0)):
            bad = (
                UM.gamma_ump2(
                    Ca, Cb, (ea[:5] - sa, ea[5:], eb[:3] - sb, eb[3:]), 5, 3, eri=I["I"]
                )[0]
                - e_mp2
            )
            assert abs(bad * edge) > 0.03  # 1/a plateau, vs shifted dE*a ~5e-4
    A = np.array([[32.0**-3, 32.0**-5], [40.0**-3, 40.0**-5]])
    for k, key in enumerate(("mp2", "rpa")):
        fit = np.linalg.solve(A, [res[32.0][k], res[40.0][k]])[0]
        assert abs(fit - c3[key]) < 3e-3 * abs(c3[key])
        p = np.log(res[32.0][k] / res[40.0][k]) / np.log(40.0 / 32.0)
        assert abs(p - 3.0) < 0.05


def test_ump2_urpa_box_limit_h_atom_ump2_zero_urpa_c3_plus_second_order_c6():
    """H/6-31G doublet: UMP2 == 0 at every a (one electron).  URPA (dRPA self-correlation) residual == c3/a^3 + c6/a^6,
    BOTH predicted from the molecule: c3 = dE/dk (0.046533), c6 = 1/2 E''(k) (4pi/3)^2 (0.79013; orbital relaxation in
    the harmonic en field, possible at 6-31G, impossible at STO-3G).  Spherical -> no c5.  Measured at a=20: residual
    after both terms 4.5e-6 of c3; with c6 omitted it would be 2.1e-3 of c3."""
    from run_ump2_box_limit import molecular

    atoms = [("H", (0.3, 0.2, 0.1))]
    mol, mf, e_mp2, e_rpa, _ = molecular(atoms, 1, 0)
    c = UM.u_r2_kernel_c3(mol, 1, 0, h=1e-3, guess=mf.make_rdm1(), second=True)
    u, I = _ump2_box(atoms, 1, 0, 20.0, mf.make_rdm1())
    den = UM.u_denominators(u["eps_a"], u["eps_b"], 1, 0, I["madelung"], "shifted")
    assert (
        e_mp2 == 0.0
        and UM.gamma_ump2(u["Ca"], u["Cb"], den, 1, 0, eri=I["I"])[0] == 0.0
    )
    d = UM.gamma_urpa(u["Ca"], u["Cb"], den, 1, 0, eri=I["I"], method="plasmon") - e_rpa
    assert abs(d * 20.0**3 - c["rpa"] - c["rpa6"] / 20.0**3) < 2e-5 * abs(c["rpa"])
    assert abs(d * 20.0**3 - c["rpa"]) > 1e-3 * abs(
        c["rpa"]
    )  # c6 is needed: the predictor's second order is real


# ============================================================ Gamma KS-DFT (pbc_dft.py, Iteration 8)
import pbc_dft as PD  # noqa: E402

# PySCF 2.13 pbc.dft.RKS, AFTDF mesh 61^3, exxdiv='ewald', BeckeGrids atom_grid=(50,146) prune=None treutler,
# small_rho_cutoff=0, conv_tol 1e-12 (run_dft_oracle.py h2)
H2_PYSCF_A1_50x146 = {
    "LDA,VWN": -1.522286692902,
    "PBE": -1.527930953892,
    "PBE0": -1.577629684839,
}
# same, UniformGrids 40^3 == 60^3 to 1e-12 (spectrally converged for all-Gaussian H density)
H2_PYSCF_UNIFORM = {"LDA,VWN": -1.521492168321, "PBE0": -1.576992652865}


def _h2_rks(I, grid, xc, kshift=None):
    ks = I["madelung"] if kshift is None else kshift
    return PD.rks(
        I["S"],
        I["h"],
        PD.dense_jk(I["I"]),
        I["enn"],
        2,
        grid,
        xc,
        kshift=ks,
        conv=1e-12,
    )[0]


def test_periodic_ssf_grid_reproduces_the_molecular_grid_in_a_huge_box():
    """Exactness anchor for the image-atom Becke construction (A2): LiH (size adjustment active) in a 30 Bohr box.
    SSF has compact support, so within 0.08a of the molecule the crystal partition must BE the molecular one
    (PySCF molecular grid, stratmann + becke_atomic_radii_adjust): measured 1e-16.  Control: dropping the
    size adjustment on the periodic side moves the same weights by 0.3."""
    from scipy.spatial import cKDTree

    atoms = [("Li", (0.3, 0.2, 0.1)), ("H", (0.3, 0.2, 3.1))]
    mol = gto.M(atom=atoms, basis="sto-3g", unit="B", cart=True, verbose=0)
    cm, wm = PD.molecular_grid_reference(mol, 30, 86, scheme="ssf")
    cen = mol.atom_coords().mean(0)
    dw = {}
    for adjust in (True, False):
        g = PD.PeriodicGrid(
            Cell(np.eye(3) * 30.0, atoms, "sto-3g"),
            30,
            86,
            D=27.0,
            scheme="ssf",
            adjust=adjust,
            deriv=None,
        )
        d, i = cKDTree(cm).query(g.coords)
        near = (d < 1e-9) & (np.linalg.norm(g.coords - cen, axis=1) < 2.4)
        assert near.sum() > 500
        dw[adjust] = np.abs(g.weights[near] - wm[i[near]]).max()
    assert dw[True] < 1e-13
    assert dw[False] > 1e-2


def test_uniform_grid_rks_matches_pinned_pyscf_pbc_uniform(h2_ints):
    """Independent numint path: our lattice-summed AOs + libxc on a 40^3 uniform grid vs PySCF pbc RKS UniformGrids."""
    g = PD.uniform_grid(Cell(H2_A, H2_ATOMS, "sto-3g"), 40)
    for xc, ref in H2_PYSCF_UNIFORM.items():
        assert abs(_h2_rks(h2_ints, g, xc) - ref) < 1e-9


def test_rks_on_pyscf_becke_points_matches_pinned_pyscf_pbc_rks(h2_ints):
    """Same points as PySCF's pbc BeckeGrids -> our numint + SCF + Madelung-on-hyb*K reproduce pbc.dft.RKS
    (measured 2.5e-14).  Mutation: Madelung dropped from the exact-exchange part moves PBE0 by ~hyb*v_M."""
    g = PD.pyscf_a1_grid(Cell(H2_A, H2_ATOMS, "sto-3g"), 50, 146)
    for xc, ref in H2_PYSCF_A1_50x146.items():
        assert abs(_h2_rks(h2_ints, g, xc) - ref) < 1e-10
    assert (
        abs(_h2_rks(h2_ints, g, "PBE0", kshift=0.0) - H2_PYSCF_A1_50x146["PBE0"]) > 0.1
    )


def test_h2_lattice_grid_error_is_the_becke_partition_exp_partition_pinned(h2_ints):
    """H2 a=4 at 75x302 vs the spectrally converged uniform reference: Becke (PySCF A1 points) is off by
    -4.4e-4 Ha, the Hirshfeld-like exp(-2r) partition (A2, same atomic grids) by 2.4e-6.  Pinned measurement
    of Iteration 8, not a bound: if the Becke number drops the grid code changed, if exp rises it broke.
    DOES NOT TRANSFER: on the triclinic 4H s+p cell at 75x302 exp is +5.3e-5 vs Becke -2.9e-5 (FINDINGS It. 8)."""
    cell = Cell(H2_A, H2_ATOMS, "sto-3g")
    e_b = (
        _h2_rks(h2_ints, PD.pyscf_a1_grid(cell, 75, 302), "LDA,VWN")
        - H2_PYSCF_UNIFORM["LDA,VWN"]
    )
    e_x = (
        _h2_rks(
            h2_ints, PD.PeriodicGrid(cell, 75, 302, D=10.0, scheme="exp"), "LDA,VWN"
        )
        - H2_PYSCF_UNIFORM["LDA,VWN"]
    )
    assert 3e-4 < abs(e_b) < 6e-4
    assert abs(e_x) < 1e-5


@pytest.mark.skipif(not SLOW, reason="set PBC_SLOW=1 (~40 s)")
def test_ks_box_limit_semilocal_has_no_a3_term_hybrid_has_hyb_times_the_hf_one():
    """Independent of PySCF pbc: H2 in a 16 Bohr box vs molecular RKS on the identical grid (75x302).  LDA: no
    exact exchange, zero dipole -> residual a^-5 (measured 8.7e-7, exponents 4.92/5.11).  PBE0: residual is
    hyb * the HF exchange Makov-Payne term, c3 = -0.25 (4pi/3) sigma^2 = -2.50706 (measured dE a^3 -2.50645
    at a=16; fitted c3 -2.50715 from a=20,24).  Madelung on the full K would give 4x."""
    atoms, edge = H2_ATOMS, 16.0
    mol = gto.M(atom=atoms, basis="sto-3g", unit="B", cart=True, verbose=0)
    I = build_integrals(
        Cell(np.eye(3) * edge, atoms, "sto-3g"), None, exxdiv="ewald", verbose=False
    )
    g = PD.PeriodicGrid(Cell(np.eye(3) * edge, atoms, "sto-3g"), 75, 302, D=14.4)
    mf = PD.molecular_rks(mol, "PBE0", 75, 302)
    c3 = -0.25 * 4 * np.pi / 3 * PD.ov_moment_sigma2(mol, mf.mo_coeff[:, :1])
    d_hyb = _h2_rks(I, g, "PBE0") - mf.e_tot
    assert abs(d_hyb * edge**3 - c3) < 1e-3 * abs(c3)
    d_lda = _h2_rks(I, g, "LDA,VWN") - PD.molecular_rks(mol, "LDA,VWN", 75, 302).e_tot
    assert abs(d_lda) < 2e-6 and abs(d_lda) < 1e-2 * abs(c3) / edge**3


# ------------------------------------------------------------ Iteration 9: k-point RHF (pbc_kpts, pure AFT)
import pbc_kpts as PK  # noqa: E402

# PySCF 2.13 KRHF + AFTDF mesh 61^3, cell.precision 1e-12, started from our dm, conv_tol 1e-11 (run_kpts_oracle.py,
# 2026-09-24); ours at the default pure-AFT cutoff agreed to 2e-14 / 3e-14, max|d eps| 1.1e-13.
KPT_REF_H2_112 = {"none": -0.902683427348, "ewald": -1.354143879961}


def test_kpts_1x1x1_mesh_is_the_gamma_code(h2_ints):
    kb = PK.build_k(Cell(H2_A, H2_ATOMS, "sto-3g"), (1, 1, 1), verbose=False)
    assert abs(kb["madelung"] - h2_ints["madelung"]) < 1e-14
    assert abs(kb["S"].imag).max() == 0.0
    for vm in (0.0, h2_ints["madelung"]):
        e_g, eps_g, _ = rhf(
            h2_ints["S"],
            h2_ints["h"],
            h2_ints["I"],
            h2_ints["enn"],
            2,
            conv=1e-12,
            kshift=vm,
        )
        e_k, eps_k, _ = PK.krhf(kb, 2, conv=1e-12, kshift=vm)
        assert abs(e_k - e_g) < 1e-12 and abs(eps_k[0] - eps_g).max() < 1e-11


@pytest.fixture(scope="module")
def kmesh_113():
    """H2/STO-3G a=4, 1x1x3 mesh vs the explicit 1x1x3 supercell at Gamma (pbc_gamma.pair_ft, no residue fold).
    The K sphere is the same on both sides ({G+q} == supercell G lattice), so the anchor is exact at ANY gcut:
    a loose one (prec 1e-4, pair thresh 1e-8) keeps this at ~30 s.  n = 3, not 2: at n = 2 e^{ik.L} = e^{-ik.L}."""
    cell = Cell(H2_A, H2_ATOMS, "sto-3g")
    gcut, th = PK.aft_gcut(cell, 1e-4), 1e-8
    sc = PK.supercell_cell(cell, (1, 1, 3))
    g = PK.gamma_aft(sc, gcut=gcut, thresh=th)
    vm_sc = madelung(sc)
    ref = {
        ex: rhf(
            g["S"],
            g["h"],
            g["I"],
            g["enn"],
            6,
            conv=1e-12,
            kshift=vm_sc if ex == "ewald" else 0.0,
        )[0]
        / 3
        for ex in ("none", "ewald")
    }
    kb = PK.build_k(cell, (1, 1, 3), gcut=gcut, thresh=th, verbose=False)
    return dict(cell=cell, gcut=gcut, th=th, ref=ref, vm_sc=vm_sc, kb=kb)


def _kmesh_e(kb, ex):
    return PK.krhf(kb, 2, conv=1e-12, kshift=kb["madelung"] if ex == "ewald" else 0.0)[
        0
    ]


def test_kmesh_equals_gamma_supercell_per_cell_and_madelungs_coincide(kmesh_113):
    """Measured 2026-09-24: 3.8e-14 (both exxdiv); also 2x2x2 H2 4e-15 (run_kpts_anchor.py b).  The k-mesh v_M IS the
    supercell v_M (PySCF tools.pbc.madelung scales the lattice by the mesh), so exxdiv=ewald needs no separate
    k-mesh constant."""
    kb = kmesh_113["kb"]
    assert abs(kb["madelung"] - kmesh_113["vm_sc"]) < 1e-14
    for ex in ("none", "ewald"):
        assert abs(_kmesh_e(kb, ex) - kmesh_113["ref"][ex]) < 1e-11


def test_kmesh_none_minus_ewald_is_nocc_vM_exactly(kmesh_113):
    """Identity (not physics): v_M S dm S shifts every occupied level by -v_M at fixed orbitals, so exxdiv=none
    converges as nocc * 2.8373/(n a) = N_k^(-1/3) with a coefficient known in advance."""
    kb = kmesh_113["kb"]
    assert abs(_kmesh_e(kb, "none") - _kmesh_e(kb, "ewald") - kb["madelung"]) < 1e-12
    # cubic n x n x n mesh: v_M = 2.8372974795 / (n a), the simple-cubic constant of the supercell
    assert (
        abs(PK.kmesh_madelung(kmesh_113["cell"], (3, 3, 3)) * 3 * 4.0 - 2.8372974794806)
        < 1e-10
    )


@pytest.mark.parametrize(
    "mutant,exx,min_err",
    [
        ("phase_P", "none", 0.1),
        ("kernel_no_q", "none", 0.1),
        ("madelung_prim", "ewald", 0.1),
    ],
)
def test_kmesh_supercell_anchor_catches_mutants(
    kmesh_113, monkeypatch, mutant, exx, min_err
):
    """Measured: phase sign in the pair FT only -0.219; v(G) instead of v(G+q) +0.377; primitive v_M -0.520 (ewald
    only; the none row is untouched, as it must be).  Blind spot: flipping the phase EVERYWHERE is k -> -k, a
    relabelling by time reversal, and no energy test can see it."""
    monkeypatch.setattr(PK, "_MUTANT", mutant)
    kb = PK.build_k(
        kmesh_113["cell"],
        (1, 1, 3),
        gcut=kmesh_113["gcut"],
        thresh=kmesh_113["th"],
        verbose=False,
    )
    assert abs(_kmesh_e(kb, exx) - kmesh_113["ref"][exx]) > min_err


def test_kmesh_hermiticity_time_reversal_and_pyscf_bloch_convention(kmesh_113):
    kb = kmesh_113["kb"]
    S, V, n, ints = kb["S"], kb["V"], kb["n"], kb["ints"]
    for k in range(kb["Nk"]):
        mk = PK.mesh_index(n, -ints[k])
        assert abs(S[k] - S[k].conj().T).max() < 1e-14
        assert (
            abs(S[mk] - S[k].conj()).max() < 1e-14
            and abs(V[mk] - V[k].conj()).max() < 1e-13
        )
    assert abs(S[1].imag).max() > 1e-2  # the test can see a phase
    # Kker time-reversal fill == brute force over all q
    saved = PK._MUTANT
    try:
        PK._MUTANT = "no_time_reversal"
        kb2 = PK.build_k(
            kmesh_113["cell"],
            (1, 1, 3),
            gcut=kmesh_113["gcut"],
            thresh=kmesh_113["th"],
            verbose=False,
        )
    finally:
        PK._MUTANT = saved
    assert abs(kb2["Kker"] - kb["Kker"]).max() < 1e-13
    pc = pyscf_cell(H2_A, H2_ATOMS, "sto-3g")
    kp = pc.make_kpts([1, 1, 3])
    assert abs(kp - kb["kpts"]).max() < 1e-14
    s_py = np.asarray(pc.pbc_intor("int1e_ovlp", hermi=1, kpts=kp))
    assert (
        abs(s_py - S).max() < 1e-12
    )  # same Bloch sign convention as PySCF (conj would differ by ~0.8)
    from pyscf.pbc import tools as ptools

    assert abs(ptools.pbc.madelung(pc, kp) - kb["madelung"]) < 1e-13


def test_kpts_h2_112_matches_pinned_pyscf_krhf_aftdf():
    kb = PK.build_k(Cell(H2_A, H2_ATOMS, "sto-3g"), (1, 1, 2), verbose=False)
    for ex in ("none", "ewald"):
        assert abs(_kmesh_e(kb, ex) - KPT_REF_H2_112[ex]) < 1e-10


# ------------------------------------------------------------ Iteration 10: Gamma UKS / ROKS (pbc_uks)
import pbc_uks as UK  # noqa: E402

# ours; PySCF 2.13 pbc.dft.UKS/ROKS (AFTDF 61^3, exxdiv='ewald', BeckeGrids (50,146) prune None treutler,
# small_rho_cutoff 0, conv 1e-12) agrees to <= 7.5e-14 (H, H2) / 3.5e-12 (tri) from its OWN default guess
# (run_uks_oracle.py, 2026-09-24).  Ours runs on PySCF's A1 points with the pure-AFT dense I.
UKS_PIN = {
    "H": {"LDA,VWN": -0.668127813328, "PBE": -0.677787718838, "PBE0": -0.700202408672},
    "H2 triplet": {
        "LDA,VWN": -0.261697156130,
        "PBE": -0.307603027052,
        "PBE0": -0.332094904636,
    },
    "tri triplet": {
        "LDA,VWN": (-1.694673507916, 2.0003319270, -1.694361887534),
        "PBE": (-1.731772052782, 2.0005413886, -1.731308266751),
        "PBE0": (-1.777429192569, 2.0006805370, -1.776801906487),
    },  # (UKS E, UKS <S2>, ROKS E)
}
_XCS = ("LDA,VWN", "PBE", "PBE0")


@pytest.fixture(scope="module")
def h2_a1_50():
    return PD.pyscf_a1_grid(Cell(H2_A, H2_ATOMS, "sto-3g"), 50, 146)


def _uks(I, na, nb, grid, xc, ex="ewald", **kw):
    return UK.uks(
        I["S"],
        I["h"],
        PD.dense_jk(I["I"]),
        I["enn"],
        na,
        nb,
        grid,
        xc,
        kshift=I["madelung"] if ex == "ewald" else 0.0,
        conv=1e-12,
        **kw,
    )


def test_uks_closed_shell_equals_rks_and_catches_bookkeeping_mutants(
    h2_ints, h2_a1_50, monkeypatch
):
    """Exactness anchor: na == nb UKS == pbc_dft.rks on the same grid (measured 0..4e-16 H2, <=3e-15 tri).
    Mutants (measured H2): hyb/2 per spin (RKS's 1/2 carried into the per-spin K) PBE0 +9.4e-2 (ewald);
    Madelung on the full K instead of hyb*K is EXACTLY -(1-hyb) v_M N/2 (LDA -0.709, PBE0 -0.532).
    Blind spot: the unpolarized-XC mutant is invisible here (rho_a = rho_b) -- see the open-shell pins."""
    vm = h2_ints["madelung"]
    for xc in _XCS:
        for ex in ("none", "ewald"):
            er = PD.rks(
                h2_ints["S"],
                h2_ints["h"],
                PD.dense_jk(h2_ints["I"]),
                h2_ints["enn"],
                2,
                h2_a1_50,
                xc,
                kshift=vm if ex == "ewald" else 0.0,
                conv=1e-12,
            )[0]
            assert abs(_uks(h2_ints, 1, 1, h2_a1_50, xc, ex)["e"] - er) < 1e-11
            assert (
                abs(
                    UK.roks(
                        h2_ints["S"],
                        h2_ints["h"],
                        PD.dense_jk(h2_ints["I"]),
                        h2_ints["enn"],
                        1,
                        1,
                        h2_a1_50,
                        xc,
                        kshift=vm if ex == "ewald" else 0.0,
                        conv=1e-12,
                    )["e"]
                    - er
                )
                < 1e-11
            )
    e0 = {xc: _uks(h2_ints, 1, 1, h2_a1_50, xc)["e"] for xc in ("LDA,VWN", "PBE0")}
    monkeypatch.setattr(UK, "_MUTANT", "madelung_full_k")
    for xc in ("LDA,VWN", "PBE0"):
        shift = -(1 - UK.hybrid_fraction(xc)) * vm * 2 / 2
        assert abs(_uks(h2_ints, 1, 1, h2_a1_50, xc)["e"] - e0[xc] - shift) < 1e-10
    monkeypatch.setattr(UK, "_MUTANT", "hyb_half_per_spin")
    assert abs(_uks(h2_ints, 1, 1, h2_a1_50, "PBE0")["e"] - e0["PBE0"]) > 1e-2
    monkeypatch.setattr(UK, "_MUTANT", "unpolarized")
    assert (
        abs(_uks(h2_ints, 1, 1, h2_a1_50, "PBE0")["e"] - e0["PBE0"]) < 1e-11
    )  # the blind spot, asserted


def test_uks_hf_is_the_gamma_uhf_open_shell(h2_ints):
    """xc='HF' through the UKS loop == pbc_uhf.uhf (independent loop) on the H2 triplet: measured 1e-16/2e-16
    (tri triplet 1e-14/3e-14, staged)."""
    for ex in ("none", "ewald"):
        ks = h2_ints["madelung"] if ex == "ewald" else 0.0
        ref = uhf(
            h2_ints["S"],
            h2_ints["h"],
            h2_ints["I"],
            h2_ints["enn"],
            2,
            0,
            conv=1e-12,
            kshift=ks,
        )["e"]
        assert abs(_uks(h2_ints, 2, 0, None, "HF", ex)["e"] - ref) < 1e-11
        assert abs(ref - UHF_REF["H2 a=4 triplet"][ex]) < 1e-10


def test_uks_matches_pinned_pyscf_pbc_uks_h_atom_and_h2_triplet(
    h2_ints, h2_a1_50, monkeypatch
):
    """Spin-polarized numint + per-spin Madelung-on-hyb*K vs PySCF pbc.dft.UKS (pinned, PySCF agrees 7e-14).
    Also: ewald - none == -hyb v_M N/2 at the same density (identity).  Mutation: the unpolarized kernel on
    the triplet moves LDA/PBE/PBE0 by +0.120/+0.129/+0.090."""
    for xc in _XCS:
        u = _uks(h2_ints, 2, 0, h2_a1_50, xc)
        assert (
            abs(u["e"] - UKS_PIN["H2 triplet"][xc]) < 1e-10
            and abs(u["s2"] - 2.0) < 1e-12
        )
        un = _uks(h2_ints, 2, 0, h2_a1_50, xc, "none", guess=(u["Da"], u["Db"]))
        assert (
            abs(u["e"] - un["e"] + UK.hybrid_fraction(xc) * h2_ints["madelung"]) < 1e-11
        )
    cell = Cell(H2_A, [("H", (0.3, 0.2, 0.1))], "sto-3g")
    Ih = build_integrals(cell, None, exxdiv="ewald", verbose=False)
    gh = PD.pyscf_a1_grid(cell, 50, 146)
    for xc in _XCS:
        assert abs(_uks(Ih, 1, 0, gh, xc)["e"] - UKS_PIN["H"][xc]) < 1e-10
    monkeypatch.setattr(UK, "_MUTANT", "unpolarized")
    assert (
        abs(
            _uks(h2_ints, 2, 0, h2_a1_50, "LDA,VWN")["e"]
            - UKS_PIN["H2 triplet"]["LDA,VWN"]
        )
        > 0.05
    )


def test_uks_polarized_vxc_is_the_derivative_of_exc_per_spin(h2_ints, h2_a1_50):
    """Energy pins are blind to potential errors that only move the (variational) energy at second order, or that act
    on an empty beta space (H, H2 triplet): dropping vsigma_ab or feeding beta the alpha vrho survived every energy
    test (source mutation, 2026-09-24).  So: tr(V_s dD) == central FD of E_xc along dD, per spin, at a density with
    BOTH spins populated and rho_a != rho_b (measured 5e-9 relative on tri, run_uks_anchor.py (c))."""
    u = _uks(h2_ints, 2, 0, h2_a1_50, "PBE")
    Da, Db = (
        u["Da"],
        0.3 * u["Ca"][:, :1] @ u["Ca"][:, :1].T
        + 0.2 * u["Ca"][:, 1:2] @ u["Ca"][:, 1:2].T,
    )
    dD = np.array([[0.3, -0.2], [-0.2, 0.5]]) * 1e-3
    for xc in ("LDA,VWN", "PBE"):
        _, Va, Vb = UK.eval_vxc_uks(h2_a1_50, Da, Db, xc)
        for s, V in ((0, Va), (1, Vb)):
            E = [
                UK.eval_vxc_uks(
                    h2_a1_50, Da + (s == 0) * e * dD, Db + (s == 1) * e * dD, xc
                )[0]
                for e in (1e-3, -1e-3)
            ]
            fd = (E[0] - E[1]) / 2e-3
            assert abs(np.sum(V * dD) - fd) < 1e-6 * abs(fd)


def _h_box(basis, edge, xcs):
    """Gamma UKS H atom in a cubic box vs molecular UKS on the identical (SSF, 75x302) grid."""
    atoms = [("H", (0.3, 0.2, 0.1))]
    mol = gto.M(atom=atoms, basis=basis, unit="B", cart=True, verbose=0, spin=1)
    mg = UK.MolGrid(mol, *PD.molecular_grid_reference(mol, 75, 302, scheme="ssf"))
    Sm, hm = (
        mol.intor("int1e_ovlp_cart"),
        mol.intor("int1e_kin_cart") + mol.intor("int1e_nuc_cart"),
    )
    w = min(1.0, 8.0 / edge)
    cell = Cell(np.eye(3) * edge, atoms, basis)
    I = build_integrals(
        cell, w, rcut_bra=18.0, rcut_2e=18.0 + 6.0 / w, exxdiv="ewald", verbose=False
    )
    g = PD.PeriodicGrid(cell, 75, 302, D=0.9 * edge, scheme="ssf")
    out = {}
    for xc in xcs:
        m = UK.uks(
            Sm, hm, PD.dense_jk(mol.intor("int2e_cart")), 0.0, 1, 0, mg, xc, conv=1e-13
        )
        p = {
            ex: _uks(I, 1, 0, g, xc, ex, guess=(m["Da"], m["Db"]))["e"] - m["e"]
            for ex in ("none", "ewald")
        }
        out[xc] = (p, m, mol, mg, I["madelung"])
    return out


def test_uks_box_limit_h_sto3g_pbe0_c3_is_hyb_times_the_uhf_one():
    """Prediction (stated first): c3 = -(2pi/3) hyb Omega_a = 0.25 x -4.081081 = -1.020270 (one AO -> no
    relaxation, spherical -> no a^-5).  Measured dE a^3: -1.036738 (12), -1.019298 (16), -1.020201 (20),
    -1.020268 (24), -1.020270 (32).  LDA/PBE have no a^-3 (1.8e-10 / 2.4e-10 at 24).  none - ewald = hyb v_M/2."""
    r = _h_box("sto-3g", 24.0, ("LDA,VWN", "PBE0"))
    p, m, mol, _, vm = r["PBE0"]
    c3, _ = UK.uks_c3_closed_form(mol, m["Ca"], m["Cb"], 1, 0, 0.25)
    assert abs(c3 + 1.020270) < 1e-6
    assert abs(p["ewald"] * 24.0**3 - c3) < 2e-5
    assert abs(p["none"] - p["ewald"] - 0.25 * vm / 2) < 1e-12
    assert abs(r["LDA,VWN"][0]["ewald"]) < 1e-8


def test_uks_box_limit_h_631g_pbe0_needs_the_predicted_relaxation_a6_term():
    """6-31G can relax: predicted c3 -1.495980 (PBE0 orbital, == relaxed r2-kernel FD) AND c6 = (1/2) E''(k) (4pi/3)^2
    = -2.0020 from the same molecular r2-kernel construction.  Measured dE a^3 at 28/32/40: -1.496071/-1.496041/
    -1.496011 == c3 + c6/a^3 to 1e-6; c3 alone is off by 9e-5 at a=28 (UHF-H showed no such term: Hartree ==
    self-exchange for one electron; the hybrid leaves (1-hyb) of it)."""
    p, m, mol, mg, _ = _h_box("6-31g", 28.0, ("PBE0",))["PBE0"]
    c3, _ = UK.uks_c3_closed_form(mol, m["Ca"], m["Cb"], 1, 0, 0.25)
    c3_r2, c6 = UK.uks_r2_kernel_c3(
        mol, 1, 0, mg, "PBE0", h=1e-3, guess=(m["Da"], m["Db"]), second=True
    )
    assert abs(c3 - c3_r2) < 1e-5 and abs(c6 + 2.002) < 2e-3
    x = p["ewald"] * 28.0**3
    assert abs(x - (c3 + c6 / 28.0**3)) < 1e-5
    assert abs(x - c3) > 5e-5


@pytest.mark.skipif(
    not SLOW,
    reason="set PBC_SLOW=1 (~5 min: tri s+p pure-AFT build + 3 xc x UKS/ROKS + trap scan)",
)
def test_uks_roks_tri_triplet_pins_and_the_ewald_trap_is_hf_only():
    """tri 4H s+p triplet vs pinned PySCF pbc.dft.UKS/ROKS (3.5e-12).  Ewald trap (run_uks_trap.py): the UHF hole
    state (none-gap -0.0057) traps pure HF from the core guess (+1.5e-2 above the staged state, flag True), but it
    is not a stationary point of alpha*HF+(1-alpha)*PBE even at alpha 0.95 (exchange-only) or with PBE correlation
    at alpha 1: every start converges to the staged state and the gap flag stays False."""
    cell = Cell(TRI_A, TRI_ATOMS, SP_BASIS)
    I = build_integrals(cell, None, exxdiv="ewald", verbose=False)
    g = PD.pyscf_a1_grid(cell, 50, 146)
    for xc in _XCS:
        e, s2, er = UKS_PIN["tri triplet"][xc]
        u = _uks(I, 3, 1, g, xc, staged=True)
        assert abs(u["e"] - e) < 1e-10 and abs(u["s2"] - s2) < 1e-8 and not u["trap"]
        r = UK.roks(
            I["S"],
            I["h"],
            PD.dense_jk(I["I"]),
            I["enn"],
            3,
            1,
            g,
            xc,
            kshift=I["madelung"],
            conv=1e-12,
            guess=(u["Da"], u["Db"]),
        )
        assert abs(r["e"] - er) < 1e-10
    trap = _uks(I, 3, 1, None, "HF")
    assert trap["trap"] and abs(trap["e"] - (-1.812958714837)) < 1e-9
    for xc in ("0.95*HF + 0.05*PBE,", "1.0*HF + 0.0*PBE, PBE"):
        st = _uks(I, 3, 1, g, xc, staged=True)
        tr = UK.uks(
            I["S"],
            I["h"],
            PD.dense_jk(I["I"]),
            I["enn"],
            3,
            1,
            g,
            xc,
            kshift=I["madelung"],
            conv=1e-11,
            guess=(trap["Da"], trap["Db"]),
            maxiter=4000,
            level_shift=0.5,
            diis_start=10**6,
        )
        assert abs(tr["e"] - st["e"]) < 1e-9 and not tr["trap"]
