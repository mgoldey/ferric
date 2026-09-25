"""Iteration 20 oracle: huge-box limit of the Gamma ROHF force -> PySCF MOLECULAR ROHF analytic gradient
Usage: python3 run_grad_ro_oracle.py [mut]   (mut: W mutants at a = 12, 20)
(pyscf.grad.rohf; PySCF 2.13 has no pbc ROHF gradient at all: pbc/grad has only rhf/uhf/rks/uks/k*).

System: H3 doublet STO-3G (the h3 anchor geometry shifted into the box), cubic a, exxdiv=ewald, pure AFT, gcut 20
(checked against the default gcut at a = 8).

PREDICTION (written before running): E_box - E_mol = c3/a^3 + O(a^-5), c3 = -(2 pi/3)(|d|^2 + Omega_a + Omega_b)
(Iteration 6's UHF closed form; first order in the kernel perturbation at FIXED orbitals, so it holds for any
variational energy of the UHF functional, ROHF included, evaluated with the ROHF MOs: Omega_a over closed+open,
Omega_b over closed).  Then the force residual is
    g_box - g_mol = (d c3 / dR) / a^3 + O(a^-5),
with d c3/dR from a central FD of c3 over molecular ROHF solutions (h = 1e-4).  a^3 (g_box - g_mol) -> dc3/dR per
component, and a (c3 + c5/a^2) fit over adjacent pairs should hit it to ~1e-4 relative by a ~ 18-20.
ARTIFACT: a missing or wrong term (e.g. W from F_eff) leaves an a-INDEPENDENT offset. a^3 x residual then grows as a^3
instead of converging.  A wrong but a-dependent term (e.g. Madelung in the RHF form) shows up as 1/a.
"""

import time

import numpy as np
from pyscf import gto, scf

import pbc_grad_open as O
import pbc_grad_ro as R
from pbc_gamma import Cell
from pbc_uhf import uhf_c3_closed_form

ATOMS = [("H", (0.3, 0.2, 0.1)), ("H", (0.35, 0.12, 1.5)), ("H", (1.6, 0.9, 0.7))]
OFF = np.array([0.4, 0.3, 0.2])


def mol_rohf(atoms):
    m = gto.M(atom=atoms, basis="sto-3g", unit="B", cart=True, spin=1, verbose=0)
    mf = scf.ROHF(m)
    mf.conv_tol = 1e-13
    mf.conv_tol_grad = 1e-9
    mf.kernel()
    c3, p = uhf_c3_closed_form(m, mf.mo_coeff, mf.mo_coeff, 2, 1)
    return mf, m, c3


def main(avals=(8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0), muts=()):
    mf, m, c3 = mol_rohf(ATOMS)
    g_mol = mf.nuc_grad_method().kernel()
    h = 1e-4
    dc3 = np.zeros((3, 3))
    fdmol = np.zeros((3, 3))
    for A in range(3):
        for x in range(3):
            vals, es = [], []
            for s in (1, -1):
                at = [(sy, np.array(r, float)) for sy, r in ATOMS]
                at[A] = (at[A][0], at[A][1] + s * h * np.eye(3)[x])
                mfd, _, c = mol_rohf(at)
                vals.append(c)
                es.append(mfd.e_tot)
            dc3[A, x] = (vals[0] - vals[1]) / (2 * h)
            fdmol[A, x] = (es[0] - es[1]) / (2 * h)
    print(f"molecular ROHF: E {mf.e_tot:.12f}, c3 {c3:.6f}; PySCF analytic - own FD max {abs(g_mol - fdmol).max():.1e}")
    print("predicted dc3/dR:\n", np.array2string(dc3, precision=5))
    rows = {}
    for a in avals:
        t = time.time()
        cell = Cell(np.eye(3) * a, [(s, np.asarray(r) + OFF) for s, r in ATOMS], "sto-3g")
        ints = O.integrals(cell, None, "ewald", gcut=20.0)
        r = R.run_case(cell, 1, 1, "HF", ints)
        d = r["grad"] - g_mol
        rows[a] = (r["e"] - mf.e_tot, d)
        extra = ""
        if a == 8.0:
            ints0 = O.integrals(cell, None, "ewald")  # default gcut
            r0 = R.run_case(cell, 1, 1, "HF", ints0)
            extra = f"  [default-gcut: dE {r0['e'] - r['e']:+.1e}, max|dF| {abs(r0['grad'] - r['grad']).max():.1e}]"
            intsn = dict(ints, madelung=0.0)
            rn = R.run_case(cell, 1, 1, "HF", intsn)
            extra += f"  [|F_ewald - F_none| {abs(rn['grad'] - r['grad']).max():.1e}]"
        print(f"a={a:4.0f}  a^3 dE {rows[a][0] * a**3:+.5f} (pred c3 {c3:+.5f})  max|a^3 dF - dc3/dR| "
              f"{abs(d * a**3 - dc3).max():.4e}  a^3 dF[2,1] {d[2, 1] * a**3:+.5f} (pred {dc3[2, 1]:+.5f})  "
              f"|sumF| {abs(r['grad'].sum(0)).max():.0e} ({time.time() - t:.0f}s){extra}", flush=True)
        for mname in muts:  # a wrong W must leave an a-INDEPENDENT offset (a^3 x it grows), an identity must not
            R._MUTANT = mname
            try:
                gm = R.gamma_ro_grad(cell, ints, r["scf"])[0]
            finally:
                R._MUTANT = None
            dm = gm - g_mol
            print(f"      mutant {mname:11s} max|g_box - g_mol| {abs(dm).max():.3e}  max|a^3 dF - dc3/dR| "
                  f"{abs(dm * a**3 - dc3).max():.4e}", flush=True)
    a_s = sorted(rows)
    if len(a_s) < 2:
        return
    print("\nc3 + c5/a^2 fits of a^3 dF over adjacent pairs, max over components of |fit - dc3/dR| (and the energy):")
    for a1, a2 in zip(a_s[:-1], a_s[1:]):
        y1, y2 = rows[a1][1] * a1**3, rows[a2][1] * a2**3
        c3f = (y2 * a2**2 - y1 * a1**2) / (a2**2 - a1**2)
        e1, e2 = rows[a1][0] * a1**3, rows[a2][0] * a2**3
        ce = (e2 * a2**2 - e1 * a1**2) / (a2**2 - a1**2)
        rel = abs(c3f - dc3).max() / abs(dc3).max()
        print(f"  ({a1:.0f},{a2:.0f}): max|c3fit - pred| {abs(c3f - dc3).max():.2e} (rel {rel:.1e}); "
              f"energy c3fit {ce:+.6f} vs {c3:+.6f}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "mut":  # mutants at two box sizes only
        main((12.0, 20.0), ("w_uhf", "w_feff_eps", "w_no_co"))
    else:
        main()
