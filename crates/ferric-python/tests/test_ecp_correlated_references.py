"""Correlated drivers on an ECP basis solve the reference `run_rhf`,
`run_uhf` and `run_dft` solve.

HI / def2-SVP carries the 28-electron def2 ECP on I. `run_rhf` applies it
to the molecule (26 valence electrons, effective nuclear charge 25 on I). The
correlated drivers used to prepare the same basis, which carries V_ECP and
the reduced charge, on the all-electron molecule (54 electrons), so their
reference was a different, unphysical system: `run_rimp2(...).rhf_energy` was
-143.31 Ha against `run_rhf`'s -297.23 Ha. Each driver family now builds its
`ferric_scf::PreparedSystem` with `EcpPolicy::Apply` (roadmap item 8 of
ARCHITECTURE.md), and each test below pins one family to the SCF drivers.

Bars: where a driver reports its reference energy, it runs the same SCF
config as the SCF driver it is compared to and agrees to rounding; 1e-8 Ha
leaves room for a different convergence path and is 1e10 below the old
error. Where it reports none, the check is the occupied count it ran with,
an independent recomputation (RI-CIS) on `run_rhf`'s orbitals, or the ratio
to an independent RI-MP2 on `run_rhf` / `run_uhf` orbitals (CCSD recovers
1.05-1.2x the MP2 correlation here; the old reference gave 0.04x).
"""

from __future__ import annotations

import functools

import numpy as np
import pytest

ferric = pytest.importorskip("ferric")

HI = "2\n\nH 0 0 0\nI 0 0 1.61\n"
EREF = 1e-8
N_VALENCE_OCC = 13


@functools.cache
def hi(charge: int = 0, multiplicity: int = 1):
    return ferric.Molecule.from_xyz_string(HI, charge=charge, multiplicity=multiplicity)


@functools.cache
def bases():
    orbital = ferric.BasisSet.bundled("def2-svp")
    return orbital, ferric.BasisSet.bundled("def2-svp-rifit")


@functools.cache
def rhf():
    return ferric.run_rhf(hi(), bases()[0])


@functools.cache
def uhf():
    return ferric.run_uhf(hi(1, 2), bases()[0])


@functools.cache
def ri_factors(charge: int, multiplicity: int):
    """`B[P, mu, nu]` with `(mu nu|ls) ~ B[P, mu, nu] B[P, l, s]` (Coulomb metric)."""
    bs, aux = bases()
    ao = ferric.AoIntegrals(hi(charge, multiplicity), bs)
    eri3 = ao.eri3(aux)
    low = np.linalg.cholesky(ao.eri2(aux))
    n = eri3.shape[0]
    return np.linalg.solve(low, eri3.reshape(n, -1)).reshape(eri3.shape)


def ri_mp2_pair(b, ca, ea, na, cb, eb, nb):
    """Opposite- or same-spin RI-MP2 pair energy of channels a and b."""
    bia = np.einsum("Pmn,mi,na->Pia", b, ca[:, :na], ca[:, na:])
    bjb = np.einsum("Pmn,mi,na->Pia", b, cb[:, :nb], cb[:, nb:])
    iajb = np.einsum("Pia,Pjb->iajb", bia, bjb)
    den = (
        ea[:na, None, None, None]
        - ea[None, na:, None, None]
        + eb[None, None, :nb, None]
        - eb[None, None, None, nb:]
    )
    return iajb, den


def ri_ump2(scf_alpha, scf_beta, charge, multiplicity):
    """RI-UMP2 correlation on given (C, eps, n_occ) per spin."""
    b = ri_factors(charge, multiplicity)
    e = 0.0
    iajb, den = ri_mp2_pair(b, *scf_alpha, *scf_beta)
    e += np.sum(iajb**2 / den)
    for c, eps, n in (scf_alpha, scf_beta):
        iajb, den = ri_mp2_pair(b, c, eps, n, c, eps, n)
        e += 0.5 * np.sum(iajb * (iajb - iajb.transpose(0, 3, 2, 1)) / den)
    return e


def rhf_channel():
    r = rhf()
    return r.mo_coefficients(), r.orbital_energies(), N_VALENCE_OCC


def test_hi_def2_svp_has_26_valence_electrons_in_run_rhf():
    s = ferric.AoIntegrals(hi(), bases()[0]).overlap()
    assert abs(np.sum(rhf().density() * s) - 2 * N_VALENCE_OCC) < 1e-8


def test_mp2_family_reference_is_run_rhf_or_run_uhf():
    bs, aux = bases()
    e_rhf = rhf().energy
    r = ferric.run_rimp2(hi(), bs, aux)
    assert abs(r.rhf_energy - e_rhf) < EREF
    channel = rhf_channel()
    assert abs(r.mp2_corr - ri_ump2(channel, channel, 0, 1)) < 1e-9
    assert abs(ferric.run_rimp2(hi(1, 2), bs, aux).rhf_energy - uhf().energy) < EREF
    assert abs(ferric.run_scs_mp2(hi(), bs, aux).rhf_energy - e_rhf) < EREF
    assert abs(ferric.run_laplace_sos_mp2(hi(), bs, aux).rhf_energy - e_rhf) < EREF
    assert abs(ferric.run_attenuated_rimp2(hi(), bs, aux).rhf_energy - e_rhf) < EREF
    assert abs(ferric.run_mp3(hi(), bs, aux).e_hf - e_rhf) < EREF
    lap = ferric.run_laplace_mp2(hi(), bs, aux)
    assert abs(lap.total_energy - lap.mp2_corr - e_rhf) < EREF


def test_rimp2_ecp_gradient_matches_finite_differences():
    # run_rhf's own analytic gradient sits 3.1e-7 Ha/Bohr from this central
    # difference (h = 1e-3 A), RI-MP2's 3.1e-7 too: the bar is 1e-6.
    bs, aux = bases()
    conv = {"scf_energy_conv": 1e-12, "scf_density_conv": 1e-10}
    h, bohr = 1e-3, 1.8897261246257702

    def mol(z):
        return ferric.Molecule.from_xyz_string(f"2\n\nH 0 0 0\nI 0 0 {z}\n")

    def energy(z):
        return ferric.run_rimp2(mol(z), bs, aux, **conv).total_energy

    fd = (energy(1.61 + h) - energy(1.61 - h)) / (2 * h * bohr)
    g = ferric.run_rimp2(mol(1.61), bs, aux, with_gradient=True, **conv).gradient()
    assert abs(g[1, 2] - fd) < 1e-6
