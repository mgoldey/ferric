"""Stage 9 mesh convergence: H2/STO-3G cubic a=6 (the flat-band cell of Iteration 9, gap ~1.1), per-cell KMP2 and
k-dRPA vs n x n x n, shifted vs unshifted denominators, and with the q = 0 ERI head restored (pbc_kcorr head=True).
Pure AFT, gcut from prec 1e-10 (same K sphere at every n, as run_kpts_convergence.py).

Predictions written BEFORE the run (molecular quantities only; H2 at the same geometry, isolated):
  P1 (near-identity): at fixed V the two conventions differ only by 2 v_M(n) in every denominator, so
     E_unshifted(n) - E_shifted(n) = -2 v_M(n) S2 + O(v_M^2),  v_M(n) = 2.8373/(n a)  =>  N_k^(-1/3) with coefficient
     (flat band: S2 ~ molecular) c1/(n a): c1 = -0.029903 (MP2), -0.037343 (dRPA) (Iterations 3/4, box-limit c1).
     So the unshifted mesh error is ~ -5e-3/n (MP2) -- slow, and it is the SAME physics as Iteration 3's 1/a.
  P2 (physics, shifted): E(n) - E_inf = c/n^3 (N_k^-1): the missing q = 0 head (weight 1/N_k) + the q^2 Fock heads
     (Iteration 9's -(4pi/3) Omega_I for HF).  Flat-band, self-term estimate c ~ c3_mol / a^3 with the Iteration 3/4
     molecular c3 = 0.671 (MP2), 0.923 (dRPA) => c ~ +3.1e-3 (MP2), +4.3e-3 (dRPA), positive (underbinding).
     Cross-molecule terms of the uniform head (Iteration 5b: 2 <h> V_reg(q=0) + <h^2>) are NOT in this estimate,
     so only the sign, the power 3 and the order of magnitude are predicted, not the coefficient.
  P3 (head restored): adding the cubic-averaged q = 0 ERI head removes the ERI-head part of c; what remains is the
     Fock-head part (frozen-orbital split of c3_mol printed below) + the anisotropy variance <h^2> - <h>^2 (not zero:
     <cos^4> = 1/5 vs <cos^2>^2 = 1/9).  E_inf must be the SAME with and without the head (it is an O(1/N_k) term).
  Artifacts: missing Madelung on occupied => the shifted column behaves like unshifted (1/n) -- tested as a mutant in
     run_kcorr_anchor.py; a q-sign/conj error breaks the supercell anchor, not the power law.
Band-sampling caveat (Iteration 9, a=4): a dispersive band gives even/odd oscillations that hide any power law; a=6
was in the n^-3 regime for HF from n = 4.  Fit the TAIL only.
Usage: python3 run_kcorr_convergence.py nmax [nmin] [a]"""

import sys
import time

import numpy as np

sys.path.insert(0, ".")
import pbc_kcorr as KC  # noqa: E402
import pbc_kpts as PK  # noqa: E402
from pbc_gamma import Cell, rhf  # noqa: E402
from pbc_mp2 import dipole_prediction_h2_minimal  # noqa: E402
from pbc_rpa import drpa_moment_prediction_h2_minimal  # noqa: E402
from pyscf import gto  # noqa: E402
from run_kpts_anchor import H2_ATOMS  # noqa: E402


def molecular_split(a):
    mol = gto.M(atom=H2_ATOMS, basis="sto-3g", unit="B", cart=True, verbose=0)
    S, h, I = (
        mol.intor("int1e_ovlp"),
        mol.intor("int1e_kin") + mol.intor("int1e_nuc"),
        mol.intor("int2e"),
    )
    e, eps, _, C = rhf(S, h, I, mol.energy_nuc(), 2, conv=1e-13, return_mo=True)
    dm, x = dipole_prediction_h2_minimal(mol, C, eps, a)
    k = 4 * np.pi / (3 * a**3)
    d2 = x["d_ia"] @ x["d_ia"]
    mp2_eri = 2 * x["iaia"] * (-k * d2) / x["D"]
    mp2_fock = dm - mp2_eri
    dr, y = drpa_moment_prediction_h2_minimal(mol, C, eps, a)
    rpa_eri = y["dEdK"] * (-k * d2)
    return dict(
        mp2=(dm * a**3, mp2_eri * a**3, mp2_fock * a**3),
        drpa=(dr * a**3, rpa_eri * a**3, (dr - rpa_eri) * a**3),
    )


if __name__ == "__main__":
    nmax = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    nmin = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    a = float(sys.argv[3]) if len(sys.argv) > 3 else 6.0
    sp = molecular_split(a)
    for m in ("mp2", "drpa"):
        print(
            f"molecular c3 {m}: total {sp[m][0]:.6f} = ERI-head part {sp[m][1]:.6f} + Fock-head part {sp[m][2]:.6f}; "
            f"/a^3 = {sp[m][0] / a**3:.4e} (ERI {sp[m][1] / a**3:.4e}, Fock {sp[m][2] / a**3:.4e})",
            flush=True,
        )
    cell = Cell(np.eye(3) * a, H2_ATOMS, "sto-3g")
    gcut = PK.aft_gcut(cell, 1e-10)
    rows = []
    for m in range(nmin, nmax + 1):
        t = time.time()
        kb = PK.build_k(cell, (m, m, m), exxdiv="ewald", gcut=gcut, verbose=False)
        e, eps, it, C, occ = PK.krhf(kb, 2, conv=1e-12, kshift=0.0, return_mo=True)
        assert all(int(o.sum()) == 1 and o[0] for o in occ)
        vm = kb["madelung"]
        st = KC.aft_ov(cell, (m, m, m), C, 1, gcut=gcut)
        sth = KC.aft_ov(cell, (m, m, m), C, 1, gcut=gcut, head=True) if m > 0 else None
        r = dict(n=m, vm=vm, gap=min(x[1] for x in eps) - max(x[0] for x in eps))
        for cv in ("shifted", "unshifted"):
            eo, ev = KC.k_denominators(eps, 1, vm, cv)
            r[cv] = (KC.kmp2(st, eo, ev)[0], KC.kdrpa_plasmon(st, eo, ev))
            if cv == "shifted":
                r["head"] = (KC.kmp2(sth, eo, ev)[0], KC.kdrpa_plasmon(sth, eo, ev))
        rows.append(r)
        print(
            f"n={m}: gap {r['gap']:.4f} v_M {vm:.8f} | MP2 sh {r['shifted'][0]:.12e} unsh {r['unshifted'][0]:.12e} "
            f"head {r['head'][0]:.12e} | dRPA sh {r['shifted'][1]:.12e} unsh {r['unshifted'][1]:.12e} head "
            f"{r['head'][1]:.12e} | (unsh-sh)*n*a MP2 {(r['unshifted'][0] - r['shifted'][0]) * m * a:.6f} dRPA "
            f"{(r['unshifted'][1] - r['shifted'][1]) * m * a:.6f} ({time.time() - t:.0f}s)",
            flush=True,
        )
    print(
        "ROWS", repr([(r["n"], r["shifted"], r["unshifted"], r["head"]) for r in rows])
    )
