"""Correlated drivers on an ECP basis solve the reference `run_rhf`,
`run_uhf` and `run_dft` solve.

HI / def2-SVP carries the 28-electron def2 ECP on I. `run_rhf` applies it
to the molecule (26 valence electrons, effective nuclear charge 25 on I). The
correlated drivers used to prepare the same basis, which carries V_ECP and
the reduced charge, on the all-electron molecule (54 electrons), so their
reference was a different, unphysical system: `run_rimp2(...).rhf_energy` was
-143.31 Ha against `run_rhf`'s -297.23 Ha. Every driver now builds a
`ferric_scf::PreparedSystem`, which applies the ECP (roadmap item 8 of
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


def test_cc_reference_is_run_rhf_or_run_uhf():
    bs, aux = bases()
    eom = ferric.run_eom_ccsd(hi(), bs, aux, n_roots=1)
    assert abs(eom.rhf_energy - rhf().energy) < EREF
    ccsd = ferric.run_ccsd(hi(), bs, aux)
    # run_eom_ccsd converges its CCSD to 1e-10 instead of 1e-8.
    assert abs(ccsd.correlation_energy - eom.ccsd_correlation_energy) < 1e-6
    channel = rhf_channel()
    assert 1.0 < ccsd.correlation_energy / ri_ump2(channel, channel, 0, 1) < 1.3
    u = uhf()
    alpha = (u.mo_coefficients_alpha(), u.orbital_energies_alpha(), N_VALENCE_OCC)
    beta = (u.mo_coefficients_beta(), u.orbital_energies_beta(), N_VALENCE_OCC - 1)
    uccsd = ferric.run_ccsd(hi(1, 2), bs, aux)
    assert uccsd.reference == "uhf"
    assert 1.0 < uccsd.correlation_energy / ri_ump2(alpha, beta, 1, 2) < 1.3


def test_gw_family_reference_is_run_rhf_run_uhf_or_run_dft():
    bs, aux = bases()
    assert abs(ferric.run_gw(hi(), bs, aux).ref_energy - rhf().energy) < EREF
    pbe = ferric.run_dft(hi(), bs, "PBE").energy
    assert abs(ferric.run_gw(hi(), bs, aux, xc="PBE").ref_energy - pbe) < EREF
    assert abs(ferric.run_u_gw(hi(1, 2), bs, aux).ref_energy - uhf().energy) < EREF
    assert ferric.run_bse_tda(hi(), bs, aux).nocc == N_VALENCE_OCC
    pol = ferric.run_tdhf_static_polarizability(hi(), bs, aux, "PBE0")
    assert pol.nocc == N_VALENCE_OCC


def test_rpa_family_reference_is_run_rhf():
    bs, aux = bases()
    e_rhf = rhf().energy
    assert abs(ferric.run_pdep_rpa(hi(), bs, aux).rhf_energy - e_rhf) < EREF
    rs = ferric.run_rs_mp2_rpa(hi(), bs, aux, scf_jk="exact")
    assert abs(rs.rhf_energy - e_rhf) < EREF


def test_tddft_reference_is_run_rhf():
    # HF / TDA singlets are RI-CIS: A[ia, jb] = d_ij d_ab (e_a - e_i)
    # + 2 (ia|jb) - (ij|ab) on run_rhf's orbitals, RI on the same auxbasis.
    bs, aux = bases()
    c, eps, n = rhf_channel()
    b = ri_factors(0, 1)
    bia = np.einsum("Pmn,mi,na->Pia", b, c[:, :n], c[:, n:])
    bij = np.einsum("Pmn,mi,nj->Pij", b, c[:, :n], c[:, :n])
    bab = np.einsum("Pmn,ma,nb->Pab", b, c[:, n:], c[:, n:])
    nv = c.shape[1] - n
    a = 2 * np.einsum("Pia,Pjb->iajb", bia, bia) - np.einsum("Pij,Pab->iajb", bij, bab)
    a = a.reshape(n * nv, n * nv) + np.diag((eps[None, n:] - eps[:n, None]).reshape(-1))
    cis = np.linalg.eigvalsh(a)[:3]
    r = ferric.run_tddft(hi(), bs, aux, n_roots=3, method="tda")
    assert np.allclose(r.excitation_energies, cis, atol=1e-6)


def test_double_hybrid_reference_is_run_dft_or_run_uks():
    bs, aux = bases()
    dh = ferric.run_double_hybrid(hi(), bs, aux)
    assert abs(dh.e_ks - ferric.run_dft(hi(), bs, "B2PLYP").energy) < EREF
    udh = ferric.run_double_hybrid(hi(1, 2), bs, aux)
    assert abs(udh.e_ks - ferric.run_uks(hi(1, 2), bs, "B2PLYP").energy) < EREF


def test_linlccd_reference_is_run_rhf_and_drpa_refuses_the_ecp():
    bs, aux = bases()
    r = ferric.run_linlccd(hi(), bs, aux)
    assert abs(r["rhf_energy"] - rhf().energy) < EREF
    # The dRPA amplitude solver counts valence virtuals in a localized
    # basis, which ECP atoms break: it refuses rather than mis-count.
    with pytest.raises(RuntimeError, match="ECP atoms"):
        ferric.run_drpa(hi(), bs, aux)


def test_tune_omega_references_are_run_dft_and_run_uks():
    # A bracket of width 2e-4 around wB97X-V's own omega = 0.3 puts every
    # evaluation within 1e-4 of it, where both energies sit within 2.1e-5 Ha
    # of run_dft's / run_uks's (measured): the bar is 1e-4 Ha, against the
    # old errors of 152.6 Ha (neutral) and 139.5 Ha (cation).
    bs = bases()[0]
    r = ferric.tune_omega(
        hi(),
        bs,
        "wB97X-V",
        omega_lo=0.2999,
        omega_hi=0.3001,
        max_evals=2,
        check_cation_stability=False,
    )
    neutral = ferric.run_dft(hi(), bs, "wB97X-V").energy
    cation = ferric.run_uks(hi(1, 2), bs, "wB97X-V").energy
    for e in r["evals"]:
        assert abs(e["e_neutral"] - neutral) < 1e-4
        assert abs(e["e_cation"] - cation) < 1e-4


def test_opmp3_reference_is_run_rhf_or_run_rohf():
    bs = bases()[0]
    assert abs(ferric.run_opmp3(hi(), bs).e_scf - rhf().energy) < EREF
    rohf = ferric.run_rohf(hi(1, 2), bs).energy
    assert abs(ferric.run_opmp3(hi(1, 2), bs).e_scf - rohf) < EREF


def test_response_modes_occupied_space_is_run_rhfs():
    # alpha_bare(0) = 4 sum_ia mu_ia^2 / (e_a - e_i) over run_rhf's occupied
    # and virtual orbitals: an occupied count read from the all-electron
    # molecule (27 instead of 13) changes it.
    bs, aux = bases()
    c, eps, n = rhf_channel()
    dip = ferric.AoIntegrals(hi(), bs).dipole()
    mu = np.einsum("dmn,mi,na->dia", dip, c[:, :n], c[:, n:])
    alpha = 4 * np.einsum("dia,eia->de", mu / (eps[None, n:] - eps[:n, None]), mu)
    modes = ferric.response_modes(hi(), bs, aux, rhf(), n_modes=1)
    assert np.allclose(modes.alpha_bare, alpha, rtol=1e-8, atol=1e-10)


def test_a_molecule_reused_from_an_ecp_run_is_all_electron_in_sto3g():
    # run_aimd returns its molecule with def2-SVP's ECP counts on I; run in
    # STO-3G (no ECP) it must be the all-electron HI a fresh parse gives.
    moved = ferric.run_aimd(hi(), "def2-svp", 1).mol()
    sto = ferric.BasisSet.bundled("sto-3g")
    fresh = ferric.Molecule.from_xyz_string(moved.to_xyz_string())
    e_fresh = ferric.run_rhf(fresh, sto).energy
    assert abs(ferric.run_rhf(moved, sto).energy - e_fresh) < 1e-8
