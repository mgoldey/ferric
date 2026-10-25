"""`DftResult.vxc()` / `DftResult.e_xc`: the XC potential of the converged density.

`vxc()` returned an all-zero placeholder from the day the binding was written
(found by the #294 symmetry probe). It now evaluates V_xc[D] of the converged
density on the SCF's own XC grid, with the SCF's functional. These tests pin:

  1. An EXACT anchor that needs no other code: for pure Slater exchange
     (LDA_X) E_x = -C sum_g w_g rho_g^(4/3) and v_x = (4/3)(E_x density), so
     tr(V_x D) = (4/3) E_x on the same grid, to rounding. Zeros give 0 != (4/3)E.
  2. An INDEPENDENT construction: PySCF's `ni.nr_rks` fed ferric's converged
     density gives the same E_xc and V_xc matrix (LDA, PBE, and the DFT part
     of PBE0). The molecule is H4 in 6-31G, s functions only, so the AO
     ordering is identical in both codes with no permutation. The grids
     differ (ferric Becke/Lebedev vs PySCF's), so both use a fine grid and
     the bar is loose; the measured gaps are recorded next to the bars.
  3. Shape, symmetry, caching, and coverage of the meta-GGA (tau) and VV10
     paths on water.
"""

from pathlib import Path

import numpy as np
import pytest

ferric = pytest.importorskip(
    "ferric",
    reason="the compiled ferric extension is not importable; build it with "
    "`cargo build --release -p ferric-python` (see CLAUDE.md).",
)

REPO = Path(__file__).resolve().parents[3]
WATER = REPO / "testdata" / "molecules" / "water.xyz"

# Rectangular H4 (two stretched H2 units), Angstrom. Closed shell, s-only basis.
H4_XYZ = """4
H4 rectangle
H 0.0 0.0 0.0
H 0.0 0.0 0.74
H 1.8 0.0 0.0
H 1.8 0.0 0.74
"""
H4_ATOMS = "H 0 0 0; H 0 0 0.74; H 1.8 0 0; H 1.8 0 0.74"


def _water_dft(functional, **kw):
    mol = ferric.Molecule.from_xyz(str(WATER))
    bs = ferric.BasisSet.bundled("sto-3g")
    return ferric.run_dft(mol, bs, functional=functional, **kw)


@pytest.mark.parametrize("functional", ["LDA", "PBE", "PBE0"])
def test_vxc_is_a_symmetric_nonzero_matrix_of_the_right_shape(functional):
    r = _water_dft(functional)
    d = r.density()
    v = r.vxc()
    assert v.shape == d.shape == (7, 7)
    assert np.all(np.isfinite(v))
    np.testing.assert_allclose(v, v.T, atol=1e-12, rtol=0)
    # The placeholder was exactly zero. V_xc is negative-definite-ish on the
    # occupied space: tr(V D) = int v_xc rho < 0 for every functional here.
    assert np.max(np.abs(v)) > 1e-2
    assert np.sum(v * d) < -1.0
    assert r.e_xc < -1.0
    # Cached: a second call returns the same matrix bit for bit.
    np.testing.assert_array_equal(r.vxc(), v)


def test_slater_exchange_homogeneity_is_exact():
    """tr(V_x D) = (4/3) E_x for LDA_X on the same grid (degree-4/3 homogeneity)."""
    r = _water_dft("LDA_X")
    e_x = r.e_xc
    tr_vd = float(np.sum(r.vxc() * r.density()))
    assert e_x < -1.0
    # MEASURED |tr(VD) - 4/3 E_x| / |E_x| = 1.4e-12 (water/STO-3G); the bar
    # leaves room for the rho floor and screening. The same ratio for "LDA"
    # (Slater + VWN, NOT homogeneous) is 1.6e-2, so the check discriminates.
    assert abs(tr_vd - 4.0 / 3.0 * e_x) <= 1e-10 * abs(e_x), (tr_vd, e_x)


@pytest.mark.parametrize("functional", ["SCAN", "wB97X-V"])
def test_vxc_covers_meta_gga_and_vv10(functional):
    r = _water_dft(functional)
    v = r.vxc()
    assert np.all(np.isfinite(v))
    np.testing.assert_allclose(v, v.T, atol=1e-12, rtol=0)
    assert np.sum(v * r.density()) < -1.0


# ferric name -> PySCF xc string with the same libxc components.
_PYSCF_XC = {
    "LDA": "lda_x,lda_c_vwn",
    "PBE": "gga_x_pbe,gga_c_pbe",
    "PBE0": "hyb_gga_xc_pbeh",
}


@pytest.mark.parametrize("functional", ["LDA", "PBE", "PBE0"])
def test_vxc_matches_pyscf_nr_rks_on_ferrics_density(functional):
    pyscf = pytest.importorskip("pyscf")
    from pyscf import dft, gto

    mol = ferric.Molecule.from_xyz_string(H4_XYZ)
    bs = ferric.BasisSet.bundled("6-31g")
    r = ferric.run_dft(
        mol,
        bs,
        functional=functional,
        grid_radial=200,
        grid_angular=590,
        df_j_aux="",
        df_k_aux="",
    )
    d = r.density()
    v = r.vxc()

    pmol = gto.M(atom=H4_ATOMS, basis="6-31g", unit="Angstrom", verbose=0)
    assert pmol.nao == d.shape[0] == 8
    # Same AO set in the same order: the overlap traces of D must agree.
    s = pmol.intor("int1e_ovlp")
    # MEASURED 2.0e-7 (bundled-vs-PySCF 6-31G digits and/or SCF tolerance;
    # not separated, and far below anything that would mask a wrong V).
    assert abs(float(np.sum(d * s)) - 4.0) < 1e-6
    grids = dft.gen_grid.Grids(pmol)
    grids.atom_grid = (200, 590)
    grids.build()
    ni = dft.numint.NumInt()
    _, e_ref, v_ref = ni.nr_rks(pmol, grids, _PYSCF_XC[functional], d)
    assert pyscf is not None
    # MEASURED (200, 590) on both sides, LDA / PBE / PBE0:
    #   |dE_xc| = 1.4e-7 / 1.4e-7 / 1.1e-7 Ha,
    #   max|dV| = 4.8e-8 / 4.9e-8 / 3.8e-8, against max|V| = 0.52 / 0.55 / 0.42.
    # The residual is two different grids plus the 2e-7 basis-digit gap above;
    # the bars sit ~7-20x above it and ~5 orders below |V| (zeros fail).
    assert abs(r.e_xc - e_ref) < 1e-6, (r.e_xc, e_ref)
    np.testing.assert_allclose(v, v_ref, atol=1e-6, rtol=0)
