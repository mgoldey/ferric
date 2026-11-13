"""The Phase 1 exit test, second set: three more published methods on the blocks.

The first set (test_block_ports.py) covers df_mo, ao2mo, veff, xc_functional
and run_rhf / run_dft(fock_hook=). These three ports use the blocks it does not:

* `delta_scf_from_blocks.core_ionization`: core-hole Delta-SCF (Bagus 1965)
  held by a virtual-space level shift (STEP, Carter-Fenk and Herbert 2020):
  run_uhf(fock_hook=, dm0=) and veff_polarized;
* `cphf_polarizability_from_blocks.polarizability`: the CPHF / CPKS static
  polarizability: dipole and cphf;
* `fno_mp2_from_blocks.fno_mp2`: DF-MP2 frozen natural orbitals (Taube and
  Bartlett 2005): df_mo and ferric.einsum on run_rhf's result.

Each port has an exactness anchor against an independent construction in
ferric and a comparison with PySCF's own implementation, read from
testdata/reference/block_ports_set2_pyscf.json (written by
scripts/gen_pyscf_block_ports_set2_refs.py; those tests skip without it).

Tolerances. Every SCF here runs at energy_conv 1e-11 and density_conv 1e-9
(RMS), exact J/K; PySCF at conv_tol 1e-12 and conv_tol_grad 1e-9.

* Delta-SCF hook vs the numpy MOM-UHF (`mom_uhf`, the independent SCF):
  both are the same stationary point of the UHF energy. Each energy is within energy_conv of its limit plus a term
  quadratic in the density error, span * (n * density_conv)^2 <= 1e-13 Ha
  (span ~ 30 Ha with the core level, n = 24 and 29): bar 2 x 1e-11 + 1e-13,
  rounded to 3e-11 Ha.
* Delta-SCF rebuilt energy: the same functional on the same density through
  another code path (veff_polarized): round-off only, 1e-11 Ha. Plain UHF
  orbital gradient: the driver stops at a density change of 1e-9 RMS
  (1e-8 max), so FDS - SDF at the last density is of the order of the last
  step times the orbital-energy span, ~1e-8 x 30; bar 1e-6.
* Delta-SCF vs PySCF: the binding energy is a difference of two SCF energies
  per code; the two codes' neutral RHF energies differ by integral
  round-off (measured below 1e-11 Ha here). Bar 1e-9 Ha.
* Polarizability vs finite field: Richardson on steps h = 0.005 and 2h
  leaves the F^6 term (alpha changes by < 1e-8 between h and 2h after the
  cancellation) and an SCF energy error eps per energy amplified by
  (16 + 1) / 3 / h^2 = 2.3e5 for the diagonal and twice that for an
  off-diagonal element (three curvatures). With eps at most energy_conv,
  that is 4.5e-6: bar 1e-5 a.u.
* Polarizability vs PySCF: alpha is first order in the orbitals, and the two
  codes' RHF densities differ by 1.3e-9 (max element, water), so with
  |alpha| < 10 a.u. the budget is 1.3e-8; the CPHF residuals (1e-10 here, a
  dense solve in the reference's ``alpha_dense``) add nothing at that scale:
  bar 1e-8 a.u. against ``alpha_dense``. PySCF's Krylov answer (``alpha``,
  what ``prop.polarizability.rhf`` returns) stops short of its own dense
  solution by up to 2.5e-7 a.u.; that distance, read from the file, plus
  1e-8 is the bar against it.
* FNO with every virtual vs run_rimp2: E_corr is first order in the orbitals
  (|E_c| x 1e-9 ~ 2e-10 Ha for two separately converged SCFs): bar 1e-9 Ha.
* FNO vs PySCF: the same first-order budget for the energies (1e-9 Ha) and
  for the occupations (max occupation 2e-2 x 1e-9, so 1e-10).

HOW THESE FAIL IF BROKEN (each mutation applied to the port and seen to fail):
see the "Mutations" table in site/src/using/python.md, "Porting published
methods".
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
PYSCF_REF = ROOT / "testdata" / "reference" / "block_ports_set2_pyscf.json"
NAMES = [
    "delta_scf_from_blocks",
    "cphf_polarizability_from_blocks",
    "fno_mp2_from_blocks",
]
EXACT_JK = {"df_j_aux": "exact", "df_k_aux": "exact"}
TIGHT = {"energy_conv": 1e-11, "density_conv": 1e-9}


def _load(name: str):
    sys.path.insert(0, str(EXAMPLES))
    try:
        spec = importlib.util.spec_from_file_location(name, EXAMPLES / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(EXAMPLES))


dscf_mod, polar_mod, fno_mod = (_load(n) for n in NAMES)

needs_ref = pytest.mark.skipif(
    not PYSCF_REF.exists(), reason=f"{PYSCF_REF.name} not generated"
)


def _pyscf(name: str) -> dict:
    return json.loads(PYSCF_REF.read_text())["molecules"][name]


_CACHE: dict = {}


def _cached(key, build):
    if key not in _CACHE:
        _CACHE[key] = build()
    return _CACHE[key]


def _rhf(name: str, basis: str):
    """(mol, basis set, AoIntegrals, run_rhf result), exact J/K, cached."""

    def build():
        mol = ferric.Molecule.from_xyz(str(MOLECULES / f"{name}.xyz"))
        bs = ferric.BasisSet.bundled(basis)
        return (
            mol,
            bs,
            ferric.AoIntegrals(mol, bs),
            ferric.run_rhf(mol, bs, **EXACT_JK, **TIGHT),
        )

    return _cached(("rhf", name, basis), build)


def _dscf(name: str, mom: bool = False) -> dict:
    xyz = MOLECULES / f"{name}.xyz"
    return _cached(
        ("dscf", name, mom), lambda: dscf_mod.core_ionization(xyz, "cc-pvdz", mom=mom)
    )


# ── Core-hole Delta-SCF ──


@pytest.mark.parametrize("name", ["water", "nh3"])
def test_level_shift_hook_reaches_the_numpy_mom_state(name):
    hook, mom = _dscf(name), _dscf(name, mom=True)
    assert abs(hook["e_neutral"] - mom["e_neutral"]) < 1e-12
    assert abs(hook["ip"] - mom["ip"]) < 3e-11


@pytest.mark.parametrize("name", ["water", "nh3"])
def test_hooked_state_is_a_stationary_point_of_plain_uhf(name):
    r = _dscf(name)
    assert abs(r["e_check"] - r["e_cation"]) < 1e-11
    assert r["grad"] < 1e-6


@needs_ref
@pytest.mark.parametrize("name", ["water", "nh3"])
def test_delta_scf_port_matches_pyscf_mom(name):
    ref, r = _pyscf(name)["delta_scf"], _dscf(name)
    assert abs(r["e_neutral"] - ref["e_neutral"]) < 1e-9
    assert abs(r["ip"] - ref["ip"]) < 1e-9


# ── CPHF / CPKS polarizability ──


def _alpha(name: str):
    _, _, ints, scf = _rhf(name, "cc-pvdz")
    return polar_mod.polarizability(
        ints, scf.mo_coefficients(), scf.orbital_energies(), ints.nelec // 2
    )


def test_cphf_polarizability_equals_the_finite_field():
    mol, bs, _, _ = _rhf("water", "cc-pvdz")
    ff = polar_mod.finite_field(mol, bs, None)
    assert np.abs(_alpha("water") - ff).max() < 1e-5


def test_cpks_polarizability_equals_the_finite_field():
    mol, bs, ints, _ = _rhf("water", "cc-pvdz")
    ks = ferric.run_dft(mol, bs, functional="PBE0", **EXACT_JK, **TIGHT)
    alpha = polar_mod.polarizability(
        ints,
        ks.mo_coefficients(),
        ks.orbital_energies(),
        ints.nelec // 2,
        xc=ints.xc_functional("PBE0"),
    )
    ff = polar_mod.finite_field(mol, bs, "PBE0")
    assert np.abs(alpha - ff).max() < 1e-5


@needs_ref
@pytest.mark.parametrize("name", ["water", "nh3"])
def test_cphf_polarizability_matches_pyscf(name):
    ref = _pyscf(name)["polarizability"]
    assert abs(_rhf(name, ref["basis"])[3].energy - ref["rhf_energy"]) < 1e-9
    alpha, dense = _alpha(name), np.asarray(ref["alpha_dense"])
    np.testing.assert_allclose(alpha, dense, rtol=0, atol=1e-8)
    # PySCF's own Krylov answer, within its distance from its dense solve.
    krylov = np.asarray(ref["alpha"])
    np.testing.assert_allclose(
        alpha, krylov, rtol=0, atol=np.abs(krylov - dense).max() + 1e-8
    )


# ── DF-MP2 frozen natural orbitals ──


def _fno(name: str, keep: int | None):
    def build():
        _, _, ints, scf = _rhf(name, "cc-pvdz")
        aux = ferric.BasisSet.bundled("cc-pvdz-ri")
        c, e = scf.mo_coefficients(), scf.orbital_energies()
        return fno_mod.fno_mp2(ints, aux, c, e, ints.nelec // 2, nvir_keep=keep)

    return _cached(("fno", name, keep), build)


@pytest.mark.parametrize("name", ["water", "nh3"])
def test_fno_with_every_virtual_equals_run_rimp2(name):
    mol, bs, _, _ = _rhf(name, "cc-pvdz")
    aux = ferric.BasisSet.bundled("cc-pvdz-ri")
    ref = ferric.run_rimp2(
        mol, bs, aux, frozen_core=0, scf_energy_conv=1e-11, scf_density_conv=1e-9
    ).mp2_corr
    assert abs(_fno(name, None)[2] - ref) < 1e-9


@needs_ref
@pytest.mark.parametrize("name", ["water", "nh3"])
def test_fno_port_matches_pyscf_make_fno(name):
    ref = _pyscf(name)["fno_mp2"]
    occ = _fno(name, None)[0]
    np.testing.assert_allclose(occ, np.asarray(ref["occupations"]), rtol=0, atol=1e-10)
    for keep, e_ref in ref["e_corr_keep"].items():
        assert abs(_fno(name, int(keep))[2] - e_ref) < 1e-9


def test_fno_coefficients_are_orthonormal_and_keep_the_occupied_block():
    _, _, ints, scf = _rhf("water", "cc-pvdz")
    nocc = ints.nelec // 2
    occ, c, _ = _fno("water", 10)
    assert c.shape[1] == nocc + 10
    np.testing.assert_allclose(c.T @ ints.overlap() @ c, np.eye(c.shape[1]), atol=1e-10)
    np.testing.assert_array_equal(c[:, :nocc], scf.mo_coefficients()[:, :nocc])
    assert np.all(np.diff(occ) <= 0.0) and occ.min() > -1e-12
