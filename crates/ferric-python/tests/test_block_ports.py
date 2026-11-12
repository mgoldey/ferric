"""The Phase 1 exit test: three published methods ported onto ferric's blocks.

The ports live in examples/python/ and use only `AoIntegrals`, `XcFunctional`
and the `fock_hook` argument of the SCF drivers:

* `drpa_from_blocks.drpa_corr`: RI direct RPA by imaginary-frequency
  quadrature (df_mo);
* `adc2_from_blocks.adc2_singlets`: singlet ADC(2) excitation energies
  (ao2mo);
* `embedding_from_blocks.embed`: projection-based embedding (Manby et al.
  2012) with `veff`, `xc_functional` and `run_rhf` / `run_dft(fock_hook=)`.

Each port has an exactness anchor against an independent construction in
ferric (run_drpa's Riccati solve, the CIS port, the full KS calculation) and a
comparison with PySCF numbers from scripts/gen_pyscf_block_ports_refs.py
(testdata/reference/block_ports_pyscf.json). The PySCF comparisons read that
file, so they run without PySCF; they skip if the file is absent.

Tolerances (derivations in each test):

* dRPA vs run_drpa: the 80-node quadrature residual (<= 1e-11 Ha on both
  molecules) plus the first-order effect of the SCF density threshold 1e-9 on
  a non-variational E_c (|E_c| x 1e-9 ~ 2e-10 Ha): bar 1e-9 Ha.
* dRPA vs PySCF on the same 40 nodes: both SCFs converged to 1e-9 in the
  density, the same aux basis and fit: bar 1e-8 Ha, the bar the RI-MP2 port
  holds against PySCF DFMP2.
* ADC(1) vs CIS: the same matrix in two index layouts; only round-off
  separates them: bar 1e-10 Ha.
* ADC(2) vs PySCF: the strict ADC(2) matrix is symmetric, so a Ritz value
  with residual norm r lies within r of an eigenvalue; PySCF's Davidson
  stops at tol_residual 1e-8 and the two RHF energies agree to 1e-13: bar
  1e-8 Ha (measured <= 2.7e-9).
* Embedding anchor (PBE-in-PBE = full PBE as mu -> inf): the energy is
  E_inf + c / mu + c2 / mu^2; the two-point extrapolation from mu = 1e5 and
  1e6 leaves c2 / 1e11, measured at <= 1.2e-9 Ha on both molecules, plus the
  SCF energy threshold 1e-11: bar 1e-8 Ha.
* HF-in-PBE vs PySCF: the two codes integrate XC on different grids. The bar
  is the sum of each code's grid change between its two grids plus the
  cross-code full-KS difference at the comparison grid: 2.9e-7 Ha (water)
  and 2.8e-7 Ha (NH3), against measured differences of 3.5e-8 and 1.8e-9.

HOW THESE FAIL IF BROKEN (each seen to fail by mutation):

* dRPA: 2 instead of 4 in Q moves E_c by 0.17 Ha; scaling Delta^2 in the
  denominator by 1.0001 moves it by 3e-5 Ha; 20 nodes instead of 80 moves
  it by 3e-6 Ha against run_drpa.
* ADC: replacing - <aj||bi> with + <aj||bi> breaks the ADC(1) anchor by
  0.7 Ha; flipping the sign of any of the three second-order singles terms
  or of either antisymmetrizer in the coupling moves the ADC(2) roots by
  4.5e-3 to 0.12 Ha against PySCF.
* Embedding: omitting the - tr[gamma_A v_emb] correction (17 Ha) or the
  projector (mu = 0, 1.6 Ha) breaks the PBE-in-PBE anchor; dropping
  - g_low[gamma_A] from v_emb leaves an SCF that does not converge; adding
  1e-6 Ha to the HF-in-PBE energy breaks the PySCF comparison.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ferric = pytest.importorskip("ferric")

ROOT = Path(__file__).resolve().parents[3]
EXAMPLES = ROOT / "examples" / "python"
MOLECULES = ROOT / "testdata" / "molecules"
PYSCF_REF = ROOT / "testdata" / "reference" / "block_ports_pyscf.json"
NAMES = [
    "rhf_from_blocks",
    "cis_from_blocks",
    "drpa_from_blocks",
    "adc2_from_blocks",
    "embedding_from_blocks",
]


def _load(name: str):
    sys.path.insert(0, str(EXAMPLES))
    try:
        spec = importlib.util.spec_from_file_location(name, EXAMPLES / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(EXAMPLES))


rhf_mod, cis_mod, drpa_mod, adc_mod, emb_mod = (_load(n) for n in NAMES)

needs_ref = pytest.mark.skipif(
    not PYSCF_REF.exists(), reason=f"{PYSCF_REF.name} not generated"
)


def _pyscf(name: str) -> dict:
    return json.loads(PYSCF_REF.read_text())["molecules"][name]


_CACHE: dict = {}


def _scf(name: str, basis: str):
    """(mol, basis set, AoIntegrals, RHF dict) from rhf_from_blocks, cached."""
    if (name, basis) not in _CACHE:
        mol = ferric.Molecule.from_xyz(str(MOLECULES / f"{name}.xyz"))
        bs = ferric.BasisSet.bundled(basis)
        ints = ferric.AoIntegrals(mol, bs)
        _CACHE[name, basis] = (mol, bs, ints, rhf_mod.rhf(ints))
    return _CACHE[name, basis]


# ── RI direct RPA ──


@pytest.mark.parametrize("name", ["water", "nh3"])
def test_drpa_port_equals_exact_run_drpa(name):
    mol, bs, ints, mf = _scf(name, "cc-pvdz")
    aux = ferric.BasisSet.bundled("cc-pvdz-ri")
    e = drpa_mod.drpa_corr(ints, aux, mf, n_freq=80)
    ref = ferric.run_drpa(mol, bs, aux, scf_energy_conv=1e-11, scf_density_conv=1e-9)
    assert ref["converged"]
    assert abs(e - ref["e_corr"]) < 1e-9


@needs_ref
@pytest.mark.parametrize("name", ["water", "nh3"])
def test_drpa_port_matches_pyscf_rpa(name):
    ref = _pyscf(name)["drpa"]
    _, _, ints, mf = _scf(name, ref["basis"])
    aux = ferric.BasisSet.bundled(ref["auxbasis"])
    assert abs(mf["e_tot"] - ref["rhf_energy"]) < 1e-8
    e = drpa_mod.drpa_corr(ints, aux, mf, n_freq=ref["nw"], x0=ref["x0"])
    assert abs(e - ref["e_corr"]) < 1e-8


# ── ADC(2) ──


@pytest.mark.parametrize("name", ["water", "nh3"])
def test_adc1_equals_the_cis_port(name):
    _, _, ints, mf = _scf(name, "6-31g")
    adc1 = adc_mod.adc2_singlets(ints, mf, nroots=3, order=1)
    cis = cis_mod.cis_singlets(ints, mf, 3)
    assert np.abs(adc1 - cis).max() < 1e-10


@needs_ref
@pytest.mark.parametrize("name", ["water", "nh3"])
def test_adc2_port_matches_pyscf_radc(name):
    ref = _pyscf(name)["adc2"]
    _, _, ints, mf = _scf(name, ref["basis"])
    assert abs(mf["e_tot"] - ref["rhf_energy"]) < 1e-8
    want = np.asarray(ref["excitation_energies"])
    got = adc_mod.adc2_singlets(ints, mf, nroots=len(want))
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-8)


@pytest.mark.parametrize("order", [0, 3, -1, 1.5])
def test_adc2_rejects_an_order_other_than_1_or_2(order):
    _, _, ints, mf = _scf("water", "6-31g")
    with pytest.raises(ValueError, match="order must be 1 or 2"):
        adc_mod.adc2_singlets(ints, mf, order=order)


@pytest.mark.parametrize("x0", [0.0, -0.5, float("nan")])
def test_drpa_rejects_a_nonpositive_frequency_scale(x0):
    with pytest.raises(ValueError, match="x0 must be positive"):
        drpa_mod.legendre_frequencies(10, x0)


# ── Projection-based embedding ──

GRID = {"grid_radial": 99, "grid_angular": 590, "grid_prune": "none"}
# (ferric's 99 x 590 -> 150 x 590 change in E, ferric - PySCF full-KS energy at
# 99 x 590), Hartree, measured values rounded up; see the HF-in-DFT test.
FERRIC_GRID_BUDGET = {"water": (5e-8, 3e-8), "nh3": (1e-8, 5e-9)}


@pytest.mark.parametrize("name", ["water", "nh3"])
def test_dft_in_dft_embedding_equals_the_full_ks_energy(name):
    xyz = MOLECULES / f"{name}.xyz"
    runs = {
        mu: emb_mod.embed(xyz, "cc-pvdz", [0], high="PBE", mu=mu) for mu in (1e5, 1e6)
    }
    full = runs[1e6]["e_low_full"]
    nocc = ferric.Molecule.from_xyz(str(xyz)).nelec() // 2
    assert 0 < runs[1e6]["n_a"] < nocc
    # Finite mu leaves E low by c / mu (c = 6.96 Ha water, 5.44 Ha NH3): the
    # leak is real and shrinks tenfold per decade of mu.
    leak = [runs[mu]["e_emb"] - full for mu in (1e5, 1e6)]
    assert leak[0] < 0 and leak[1] < 0
    assert leak[0] / leak[1] == pytest.approx(10.0, rel=1e-3)
    # The two-point extrapolation leaves -c2 / (1e5 * 1e6); the measured
    # residual is 1.2e-9 (water) and 7e-10 (NH3).
    limit = emb_mod.mu_extrapolated(runs[1e5]["e_emb"], runs[1e6]["e_emb"])
    assert abs(limit - full) < 1e-8


@needs_ref
@pytest.mark.parametrize("name", ["water", "nh3"])
def test_pyscf_reference_satisfies_the_embedding_anchor(name):
    coarse, _, mu_1e5 = _pyscf(name)["embedding"]
    assert (coarse["mu"], mu_1e5["mu"]) == (1e6, 1e5)
    limit = emb_mod.mu_extrapolated(mu_1e5["low_in_low"], coarse["low_in_low"])
    assert abs(limit - coarse["e_low_full"]) < 1e-8


@needs_ref
@pytest.mark.parametrize("name", ["water", "nh3"])
def test_hf_in_dft_embedding_matches_pyscf(name):
    coarse, fine, _ = _pyscf(name)["embedding"]
    assert coarse["grid"] == [GRID["grid_radial"], GRID["grid_angular"]]
    xyz = MOLECULES / f"{name}.xyz"
    ours = emb_mod.embed(xyz, "cc-pvdz", [0], high=None, grid=GRID)
    assert ours["n_a"] == coarse["n_a"]
    np.testing.assert_allclose(ours["sv"], coarse["spade_sv"], atol=1e-6)
    # Fixed error budget per molecule, independent of the value under test.
    # PySCF's own XC quadrature error is its change on the finer 150 x 974
    # grid (read from the reference). Ferric's two terms are constants
    # measured once on the fine grid and rounded up: ferric's change between
    # 99 x 590 and 150 x 590 (4.2e-8 water, 5.0e-9 NH3) and the full-KS
    # difference at the shared 99 x 590 grid (2.1e-8 water, 3.3e-9 NH3),
    # which is what the two radial schemes and Becke partitions still
    # disagree on. The measured difference is 3.5e-8 (water) and 1.8e-9 (NH3);
    # a 1e-3 Ha error in either code's embedding energy is far outside.
    ferric_grid, ferric_full_ks = FERRIC_GRID_BUDGET[name]
    bar = abs(fine["hf_in_low"] - coarse["hf_in_low"]) + ferric_grid + ferric_full_ks
    assert abs(ours["e_emb"] - coarse["hf_in_low"]) < bar
