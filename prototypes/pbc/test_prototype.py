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
