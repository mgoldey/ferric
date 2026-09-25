"""Stage 7 formula checks on MOLECULES (no periodicity), before any cell is touched:
1. three independent dRPA constructions agree on dense (ia|jb): plasmon, ring-CCD Riccati,
   frequency quadrature on the eigen-factorised ov block;
2. our quadrature on PySCF's own DF tensor == pyscf.gw.rpa.RPA (same nw/x0) -- convention oracle;
3. O(V^2): -(1/2pi) int tr Pi^2/2 == direct MP2, and E(lambda V)/lambda^2 -> direct MP2;
4. quadrature convergence of ferric's default grid (GL n=20, u0 0.5 / minimax table 0.5)."""

import sys
import numpy as np

sys.path.insert(0, ".")
from pyscf import gto, scf, df
from pyscf.gw import rpa as prpa
from pbc_rpa import (
    drpa_quad,
    drpa_plasmon,
    drpa_riccati,
    direct_mp2,
    ov_factor,
    drpa_second_order_quad,
)
from pbc_mp2 import mp2_energy

for label, atom, basis in [
    ("H2O/6-31G", "O 0 0 0.2; H 0 1.4 -0.9; H 0.1 -1.5 -0.8", "6-31g"),
    ("H2/STO-3G", "H 0.3 0.2 0.1; H 0.3 0.2 1.5", "sto-3g"),
]:
    mol = gto.M(atom=atom, basis=basis, unit="B", cart=True, verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    nocc = mol.nelectron // 2
    Co, Cv = mf.mo_coeff[:, :nocc], mf.mo_coeff[:, nocc:]
    eo, ev = mf.mo_energy[:nocc], mf.mo_energy[nocc:]
    ovov = mol.ao2mo((Co, Cv, Co, Cv), compact=False).reshape(
        nocc, -1, nocc, Cv.shape[1]
    )
    ep, er = drpa_plasmon(ovov, eo, ev), drpa_riccati(ovov, eo, ev)
    eq, nq = drpa_quad(ov_factor(ovov), eo, ev, return_n=True)
    print(
        f"{label} exact ERI: plasmon {ep:.14e}  riccati-plasmon {er - ep:+.1e}  quad(n={nq})-plasmon {eq - ep:+.1e}"
    )
    for n in (10, 20, 40, 80):
        print(
            f"   GL n={n:3d} x0=0.5: quad - plasmon {drpa_quad(ov_factor(ovov), eo, ev, n=n) - ep:+.2e}"
        )
    dmp2 = direct_mp2(ovov, eo, ev)
    e_os = mp2_energy(ovov, eo, ev)[1]
    e2 = drpa_second_order_quad(ov_factor(ovov), eo, ev, n=400)
    print(
        f"   direct MP2 {dmp2:.14e} (= 2 e_os: {dmp2 - 2 * e_os:+.1e}); -(1/2pi)int trPi^2/2 - dMP2 {e2 - dmp2:+.1e}"
    )
    for lam in (1e-2, 1e-3):
        el = drpa_plasmon(lam * ovov, eo, ev) / lam**2
        print(
            f"   lambda {lam:.0e}: E(lam V)/lam^2 - dMP2 {el - dmp2:+.3e}  (/lam {(el - dmp2) / lam:+.4e})"
        )
    # PySCF DF RPA vs ours on PySCF's own cderi
    for aux in ("cc-pvdz-ri", "def2-universal-jkfit"):
        r = prpa.RPA(mf)
        r.with_df = df.DF(mol, auxbasis=aux)
        e_py = r.kernel(nw=40, x0=0.5)
        Lpq = r.with_df._cderi if isinstance(r.with_df._cderi, np.ndarray) else None
        from pyscf import lib

        cderi = lib.unpack_tril(np.asarray(r.with_df._cderi))
        Bia = np.einsum("Pmn,mi,na->Pia", cderi, Co, Cv)
        e_ours = drpa_quad(Bia, eo, ev, n=40, x0=0.5)
        e_ours_conv = drpa_quad(Bia, eo, ev)
        e_pl = drpa_plasmon(np.einsum("Pia,Pjb->iajb", Bia, Bia), eo, ev)
        print(
            f"   PySCF RPA DF {aux}: {e_py:.14e}; ours(same n=40) {e_ours - e_py:+.1e}; "
            f"plasmon(DF) - PySCF {e_pl - e_py:+.1e}; converged-quad - plasmon(DF) {e_ours_conv - e_pl:+.1e}; "
            f"DF fit error (plasmon DF - exact) {e_pl - ep:+.3e} (rel {(e_pl - ep) / ep:+.1e})"
        )
