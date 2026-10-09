"""Molecular formula pins for pbc_ump2 (no periodicity): NH triplet / OH doublet, cart basis.
- ours UMP2 on exact ERIs vs pyscf.mp.UMP2 (exact)
- ours URPA quadrature on PySCF's OWN DF tensors (pyscf.gw.urpa.URPA eris) vs pyscf URPA, same nw
- ours URPA plasmon (exact) vs ours URPA quadrature (exact, joint ov factor), and tr Pi^2 == direct UMP2"""

import sys

import numpy as np
from pyscf import gto, mp, scf
from pyscf.gw import urpa as purpa

sys.path.insert(0, ".")
import pbc_ump2 as U  # noqa: E402

SYS = {
    "NH triplet": ("N 0 0 0; H 0 0 1.95", 2),
    "OH doublet": ("O 0 0 0; H 0 0 1.83", 1),
}
for name, (atom, spin) in SYS.items():
    for basis in ("sto-3g", "6-31g"):
        mol = gto.M(atom=atom, basis=basis, unit="B", cart=True, spin=spin, verbose=0)
        mf = scf.UHF(mol)
        mf.conv_tol = 1e-12
        mf.kernel()
        na, nb = mol.nelec
        (ea, eb), (Ca, Cb) = mf.mo_energy, mf.mo_coeff
        den = U.u_denominators(ea, eb, na, nb, 0.0, "unshifted")
        I = mol.intor("int2e")
        e, eaa, ebb, eab = U.gamma_ump2(Ca, Cb, den, na, nb, eri=I)
        pm = mp.UMP2(mf).run()
        ov = U.u_ovov(Ca, Cb, na, nb, eri=I)
        rp = U.urpa_plasmon(ov, *den)
        Ba, Bb = U.u_ov_factor(ov)
        rq = U.urpa_quad(Ba, Bb, *den)
        so = U.urpa_second_order_quad(Ba, Bb, *den)
        dm = U.direct_ump2(ov, *den)
        # PySCF URPA on its own DF (default aux), nw=40, vs our quadrature on the same tensors
        r = purpa.URPA(mf)
        eris = r.ao2mo()
        ecp = r.kernel(eris=eris, nw=40)
        Lov = [
            np.vstack([eris.get_ov_blk(s, 0, eris.nocc[s] * eris.nvir[s])]).T.reshape(
                -1, eris.nocc[s], eris.nvir[s]
            )
            for s in (0, 1)
        ]
        ours_df = U.urpa_quad(Lov[0], Lov[1], *den, n=40)
        print(
            f"{name}/{basis}: nelec {na},{nb}  UMP2 {e:+.12e} (aa {eaa:+.3e} bb {ebb:+.3e} ab {eab:+.3e}) "
            f"- pyscf {e - pm.e_corr:+.1e} (ss {eaa + ebb - pm.e_corr_ss:+.1e}, os {eab - pm.e_corr_os:+.1e})"
        )
        print(
            f"    URPA exact plasmon {rp:+.12e}; quad - plasmon {rq - rp:+.1e}; trPi^2 - direct UMP2 {so - dm:+.1e};"
            f"  ours quad(n=40) on PySCF DF - pyscf.gw.urpa {ours_df - ecp:+.1e} (pyscf {ecp:+.10e})"
        )
