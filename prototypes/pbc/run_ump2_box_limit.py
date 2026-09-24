"""Stage-8 box limit (independent of PySCF pbc): Gamma UMP2 / URPA in a cubic box a vs the MOLECULE.

usage: python3 run_ump2_box_limit.py {H|NH} a1 a2 ...     (no a: print only the predictions)
Systems (6-31G, cart): H atom doublet (1,0); NH triplet (5,3), R = 1.95 Bohr -- all three UMP2 blocks nonzero.
Ewald split w = min(1, 8/a), bra 18 Bohr, ket 18+6/w (Iterations 3/4/6); UHF started from the molecular UHF density.

PREDICTIONS (written BEFORE the sweep; molecular quantities only):
  physics, 'shifted' (per-spin eps_occ,s - v_M, = the exxdiv='ewald' UHF eigenvalues):
      dE = E_corr(pbc) - E_corr(mol) = c3/a^3 + O(a^-5), c3 from pbc_ump2.u_r2_kernel_c3 (harmonic kernel
      (k/2)|r-r'|^2, k = 4pi/3a^3, on every molecular interaction, UHF relaxed, d/dk).
  physics, 'unshifted': dE_uns - dE_sh = -(v_M a) * slope / a + O(a^-2), slope = dE_mol/ds for eps_occ -> eps_occ - s
      on BOTH spins (pbc_ump2.uniform_occ_shift_slope) -> 1/a convergence, c1 = -2.8373 * slope.
  H atom: UMP2 == 0 identically (one electron; no pair) at every a -> anchor, not a sweep.  URPA != 0 (dRPA
      self-correlation).  All H densities are spherical about ONE centre, so -- as for the UHF H atom (Iteration 6) --
      the cubic kernel beyond r^2 (l >= 4 harmonics) cannot act: expect dE*a^3 FLAT at c3 once image overlap dies.
  artifacts: per-spin chi0 factor 4 (closed-shell factor on a spin channel) -> O(1) plateau (URPA != mol at a -> inf);
      alpha orbitals for beta B -> O(1) plateau; Madelung v_M/2 per spin -> exponent 1 with c1/2 in 'shifted';
      missing images / normalisation -> plateau.  Distinguishable: exponent 3 with predicted c3 vs 1 vs 0.
  ADDED AFTER the first sweep (2026-09-24): the H residual is NOT flat at c3 (unlike the minimal-basis UHF H of
      Iteration 6): 6-31G lets the orbital relax in the harmonic en field, a SECOND-order effect, c6/a^6 with
      c6 = (1/2) E''(k) (4pi/3)^2 (u_r2_kernel_c3(second=True)).  Spherical => still no c5.
"""

import sys
import time

import numpy as np
from pyscf import gto, mp, scf

sys.path.insert(0, ".")
import pbc_ump2 as U  # noqa: E402
from pbc_gamma import Cell, build_integrals  # noqa: E402
from pbc_uhf import uhf  # noqa: E402

SYS = {
    "H": ([("H", (0.3, 0.2, 0.1))], 1, 0),
    "NH": ([("N", (0.3, 0.2, 0.1)), ("H", (0.3, 0.2, 0.1 + 1.95))], 5, 3),
}
BASIS = "6-31g"
VMA = 2.8372974794806  # v_M * a, simple cubic


def molecular(atoms, na, nb):
    mol = gto.M(atom=atoms, basis=BASIS, unit="B", cart=True, verbose=0, spin=na - nb)
    mf = scf.UHF(mol)
    mf.conv_tol = 1e-13
    mf.kernel()
    I = mol.intor("int2e")
    # NOT mf.mo_energy: PySCF 2.13 scf.UHF(mol) with ONE electron dispatches to uhf.HF1e (scf/__init__.py:157),
    # whose Fock is hcore -> virtual eps = h_aa (H/6-31G: 0.4609 vs standard UHF h+J-K 0.9162).  Occupied eps and
    # E are identical (J-K on itself = 0), so only virtual-dependent quantities (URPA, e_ia) are wrong.  Use the same
    # UHF construction as the box (pbc_uhf.uhf on the molecular S, h, I, E_nn).
    u = uhf(
        mol.intor("int1e_ovlp"),
        mol.intor("int1e_kin") + mol.intor("int1e_nuc"),
        I,
        mol.energy_nuc(),
        na,
        nb,
        conv=1e-13,
        guess=mf.make_rdm1(),
    )
    assert abs(u["e"] - mf.e_tot) < 1e-10, (u["e"], mf.e_tot)
    (ea, eb), (Ca, Cb) = (u["eps_a"], u["eps_b"]), (u["Ca"], u["Cb"])
    den = U.u_denominators(ea, eb, na, nb, 0.0, "unshifted")
    e_mp2 = U.gamma_ump2(Ca, Cb, den, na, nb, eri=I)[0]
    pm = (
        mp.UMP2(mf).run().e_corr
    )  # UMP2 uses only e_i+e_j-e_a-e_b; for 1 electron it is 0 either way
    assert abs(pm - e_mp2) < 1e-9, (
        pm,
        e_mp2,
    )  # measured 1.2e-10 (NH): SCF orbital convergence, ours vs PySCF
    e_rpa = U.gamma_urpa(Ca, Cb, den, na, nb, eri=I, method="plasmon")
    return mol, mf, e_mp2, e_rpa, U.uniform_occ_shift_slope(Ca, Cb, ea, eb, na, nb, I)


if __name__ == "__main__":
    name = sys.argv[1]
    edges = [float(x) for x in sys.argv[2:]]
    atoms, na, nb = SYS[name]
    t = time.time()
    mol, mf, e_mp2, e_rpa, slope = molecular(atoms, na, nb)
    c3 = U.u_r2_kernel_c3(mol, na, nb, guess=mf.make_rdm1())
    c3b = U.u_r2_kernel_c3(mol, na, nb, h=1e-3, guess=mf.make_rdm1())
    print(
        f"{name}/{BASIS}: E_mol UHF {mf.e_tot:.12f} <S2> {mf.spin_square()[0]:.8f}; E_mol UMP2 {e_mp2:+.12e} "
        f"(== pyscf.mp.UMP2); URPA {e_rpa:+.12e}  ({time.time() - t:.0f}s)"
    )
    print(
        f"  PREDICTED shifted c3 (h=1e-4 / 1e-3): HF {c3['hf']:.6f} / {c3b['hf']:.6f}  UMP2 {c3['mp2']:.6f} / "
        f"{c3b['mp2']:.6f}  URPA {c3['rpa']:.6f} / {c3b['rpa']:.6f}"
    )
    print(
        f"  PREDICTED unshifted-shifted c1 = -(v_M a) dE/ds: UMP2 {-VMA * slope[0]:.6f}  URPA {-VMA * slope[1]:.6f}",
        flush=True,
    )
    dm = mf.make_rdm1()
    for edge in edges:
        t = time.time()
        w = min(1.0, 8.0 / edge)
        I = build_integrals(
            Cell(np.eye(3) * edge, atoms, BASIS),
            w,
            rcut_bra=18.0,
            rcut_2e=18.0 + 6.0 / w,
            exxdiv="ewald",
            verbose=False,
        )
        vm = I["madelung"]
        u = uhf(I["S"], I["h"], I["I"], I["enn"], na, nb, conv=1e-12, guess=dm)
        r = {}
        for c in ("shifted", "unshifted"):
            den = U.u_denominators(u["eps_a"], u["eps_b"], na, nb, vm, c)
            m = U.gamma_ump2(u["Ca"], u["Cb"], den, na, nb, eri=I["I"])[0]
            p = U.gamma_urpa(
                u["Ca"], u["Cb"], den, na, nb, eri=I["I"], method="plasmon"
            )
            q = U.gamma_urpa(u["Ca"], u["Cb"], den, na, nb, eri=I["I"], method="quad")
            r[c] = (m - e_mp2, p - e_rpa, q - p)
        # MUTANT denominators (artifact hypotheses): v_M/2 per spin, and v_M on the alpha occupied only.  Both must
        # leave a 1/a residual (dE*a -> c1/2 resp. the alpha share of c1), not the a^-3 of the per-spin shift.
        ea, eb = u["eps_a"], u["eps_b"]
        mut = {}
        for tag, (sa, sb) in (("half", (vm / 2, vm / 2)), ("alpha-only", (vm, 0.0))):
            den = (ea[:na] - sa, ea[na:], eb[:nb] - sb, eb[nb:])
            mut[tag] = (
                U.gamma_ump2(u["Ca"], u["Cb"], den, na, nb, eri=I["I"])[0] - e_mp2,
                U.gamma_urpa(
                    u["Ca"], u["Cb"], den, na, nb, eri=I["I"], method="plasmon"
                )
                - e_rpa,
            )
        dhf = u["e"] - vm * (na + nb) / 2 - mf.e_tot
        print(
            f"  a={edge:5.1f}  dHF_ewald {dhf:+.6e} (*a^3 {dhf * edge**3:+.5f})  "
            f"UMP2 sh {r['shifted'][0]:+.10e} (*a^3 {r['shifted'][0] * edge**3:+.6f}) uns {r['unshifted'][0]:+.6e}  "
            f"URPA sh {r['shifted'][1]:+.10e} (*a^3 {r['shifted'][1] * edge**3:+.6f}) uns {r['unshifted'][1]:+.6e}  "
            f"quad-plasmon {max(abs(r[c][2]) for c in r):.0e}  <S2> {u['s2']:.6f}  ({time.time() - t:.0f}s)",
            flush=True,
        )
        print(
            "         MUTANT dE*a: "
            + "  ".join(
                f"{k} UMP2 {v[0] * edge:+.6f} URPA {v[1] * edge:+.6f}"
                for k, v in mut.items()
            ),
            flush=True,
        )
