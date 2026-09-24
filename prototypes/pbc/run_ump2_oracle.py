"""Stage-8 PySCF oracles (PySCF 2.13), Gamma UMP2 / URPA.

UMP2: pyscf.pbc.mp.UMP2 on pbc.scf.UHF + AFTDF (mesh 61^3).  PySCF takes mf.mo_energy, so exxdiv=None ->
      'unshifted', exxdiv='ewald' -> 'shifted' (per-spin -v_M on occupied).  Also pyscf.pbc.cc.UCCSD's
      init_amps MP2 (ALWAYS per-spin _adjust_occ(-madelung), pbc/cc/ccsd.py:90) -> 'shifted' for both HFs.
URPA: PySCF has no periodic RPA.  (i) PySCF's periodic AFTDF (ia|jb) blocks + our spin-orbital plasmon vs ours
      (pure-AFT ERI); (ii) PySCF's periodic RSGDF 3-index tensor (cart cc-pvdz-ri, same aux as ours) + PySCF's
      MOLECULAR URPA arithmetic (pyscf.gw.urpa.make_dielectric_matrix, rpa.kernel's loop) vs our RS-GDF B + quad.
Ours: pure-AFT dense ERI, own UHF (None first; ewald started from the None density: Iteration 6 trap);
PySCF UHF is started from OUR density (a basin difference would otherwise be indistinguishable).
Also prints the per-spin denominator check: eps(ewald) - eps(None) = -v_M on each spin's occupied, 0 on virtuals."""

import sys
import time

import numpy as np
from pyscf.gw import rpa as prpa
from pyscf.gw import urpa as purpa
from pyscf.pbc import cc as pcc
from pyscf.pbc import df as pdf
from pyscf.pbc import gto as pgto
from pyscf.pbc import mp as pmp
from pyscf.pbc import scf as pscf

sys.path.insert(0, ".")
import pbc_ump2 as U  # noqa: E402
from pbc_gamma import Cell, build_integrals  # noqa: E402
from pbc_gdf import build_gdf, ferric_basis  # noqa: E402
from pbc_uhf import uhf  # noqa: E402
from test_prototype import ANCHOR_ALPHA, SP_BASIS, TRI_A, TRI_ATOMS  # noqa: E402

PENTA_ATOMS = TRI_ATOMS + [("H", (1.3, 3.1, 4.0))]
ONE_S = {"H": [[0, [ANCHOR_ALPHA, 1.0]]]}
H2_ATOMS = [("H", (0.3, 0.2, 0.1)), ("H", (0.3, 0.2, 1.5))]
SYS = {
    # H2/STO-3G triplet has no alpha virtual and no beta electron: UMP2 == URPA == 0 identically (not a pin).
    # 6-31G gives alpha 2 occ / 2 vir -> a nonzero same-spin (aa) pin; bb and ab are empty.
    "H2 6-31g triplet": (np.eye(3) * 4.0, H2_ATOMS, "6-31g", 2, 0),
    "penta one-s doublet": (TRI_A, PENTA_ATOMS, ONE_S, 3, 2),
    "tri 4H s+p triplet": (TRI_A, TRI_ATOMS, SP_BASIS, 3, 1),
}


class Eris:  # attributes pyscf.gw.urpa.make_dielectric_matrix touches
    def __init__(self, ovL):
        self.ovL = ovL
        self.nocc = [x.shape[0] for x in ovL]
        self.nvir = [x.shape[1] for x in ovL]
        self.naux = ovL[0].shape[2]
        self.dtype = np.float64

    def get_ov_blk(self, s, p0, p1):
        return self.ovL[s].reshape(-1, self.naux)[p0:p1]


def pyscf_urpa_loop(ovL, e_ov, f_ov, nw, x0=0.5):
    er = Eris(ovL)
    e = 0.0
    for w, wt in zip(*prpa._get_scaled_legendre_roots(nw, x0)):
        diel = purpa.make_dielectric_matrix(w, e_ov, f_ov, er, blksize=10**6)
        e += (
            wt
            / (2 * np.pi)
            * (np.log(np.linalg.det(np.eye(er.naux) - diel)) + np.trace(diel))
        )
    return e.real


for name in sys.argv[1:] or list(SYS):
    a, atoms, basis, na, nb = SYS[name]
    t = time.time()
    cell = Cell(a, atoms, basis)
    I = build_integrals(cell, None, exxdiv="ewald", verbose=False)
    vm = I["madelung"]
    un = uhf(I["S"], I["h"], I["I"], I["enn"], na, nb, conv=1e-12)
    # pbc_uhf stops at |[F,D]| < 1e-7: E is then good to 1e-12 but the ORBITALS only to ~1e-7, which moved tri s+p
    # UMP2/URPA by 2-9e-8 vs PySCF.  Correlation needs the density converged: continue to conv=1e-16 (and PySCF
    # conv_tol_grad 1e-10 below).  Measured: ours-formula on PySCF's orbitals == pbc.mp.UMP2 to 2.7e-14.
    un = uhf(
        I["S"],
        I["h"],
        I["I"],
        I["enn"],
        na,
        nb,
        conv=1e-16,
        maxiter=400,
        guess=(un["Da"], un["Db"]),
    )
    ue = uhf(
        I["S"],
        I["h"],
        I["I"],
        I["enn"],
        na,
        nb,
        conv=1e-16,
        maxiter=400,
        kshift=vm,
        guess=(un["Da"], un["Db"]),
    )
    dva = ue["eps_a"] - un["eps_a"]
    dvb = ue["eps_b"] - un["eps_b"]
    print(
        f"{name}: v_M {vm:.10f}  E_none {un['e']:.12f}  E_ewald {ue['e']:.12f}  <S2> {un['s2']:.9f} ({time.time() - t:.0f}s)"
    )
    print(
        f"   eps(ewald)-eps(none)+v_M on occ: alpha {abs(dva[:na] + vm).max(initial=0.0):.1e} beta {abs(dvb[:nb] + vm).max(initial=0.0):.1e};"
        f"  on vir: {max(abs(dva[na:]).max(), abs(dvb[nb:]).max(initial=0.0)):.1e};  |Ca(e)-Ca(n)| (occ proj) "
        f"{abs(ue['Da'] - un['Da']).max():.1e}"
    )
    Ca, Cb = un["Ca"], un["Cb"]
    ours = {}
    for c in ("unshifted", "shifted"):
        den = U.u_denominators(un["eps_a"], un["eps_b"], na, nb, vm, c)
        ours[c] = (
            U.gamma_ump2(Ca, Cb, den, na, nb, eri=I["I"]),
            U.gamma_urpa(Ca, Cb, den, na, nb, eri=I["I"], method="plasmon"),
        )
        # own-convention check: the ewald SCF eigenvalues ARE the shifted denominators
        if c == "shifted":
            den_e = (
                ue["eps_a"][:na],
                ue["eps_a"][na:],
                ue["eps_b"][:nb],
                ue["eps_b"][nb:],
            )
            d_e = (
                U.gamma_ump2(ue["Ca"], ue["Cb"], den_e, na, nb, eri=I["I"])[0]
                - ours[c][0][0]
            )
            print(f"   UMP2 from ewald-SCF eigenvalues - shifted(None eps) {d_e:+.1e}")
        m = ours[c][0]
        print(
            f"   ours {c:9s} UMP2 {m[0]:+.12e} (aa {m[1]:+.4e} bb {m[2]:+.4e} ab {m[3]:+.4e})  URPA {ours[c][1]:+.12e}"
        )
    auxc = ferric_basis("cc-pvdz-ri", ["H"])
    Bc = build_gdf(cell, auxc, spherical=False)["B"]
    ours_gdf = {
        c: U.urpa_quad(
            *U.u_bia(Bc, Ca, Cb, na, nb),
            *U.u_denominators(un["eps_a"], un["eps_b"], na, nb, vm, c),
            n=40,
        )
        for c in ("unshifted", "shifted")
    }

    pc = pgto.Cell(
        a=a, atom=atoms, basis=basis, unit="B", cart=True, verbose=0, spin=na - nb
    )
    pc.precision = 1e-12
    pc.build()
    for ex, c in ((None, "unshifted"), ("ewald", "shifted")):
        t = time.time()
        mf = pscf.UHF(pc, exxdiv=ex)
        mf.with_df = pdf.AFTDF(pc)
        mf.with_df.mesh = [61] * 3
        mf.conv_tol = 1e-14
        mf.conv_tol_grad = 1e-10
        src = un if ex is None else ue
        mf.kernel(dm0=np.array([src["Da"], src["Db"]]))
        pm = pmp.UMP2(mf).run()
        ecc = pcc.UCCSD(mf)
        emp2_cc = ecc.init_amps(ecc.ao2mo())[0]
        # (i) PySCF AFTDF ov blocks + our plasmon
        (Pa, Pb), (ea, eb) = mf.mo_coeff, mf.mo_energy

        def f(C1, n1, C2, n2):
            shape = (n1, C1.shape[1] - n1, n2, C2.shape[1] - n2)
            if (
                0 in shape
            ):  # empty spin block (H2 triplet: no beta electron); PySCF ao2mo cannot take it
                return np.zeros(shape)
            return mf.with_df.ao2mo(
                (C1[:, :n1], C1[:, n1:], C2[:, :n2], C2[:, n2:]), compact=False
            ).real.reshape(shape)

        ov = dict(aa=f(Pa, na, Pa, na), bb=f(Pb, nb, Pb, nb), ab=f(Pa, na, Pb, nb))
        den_p = (ea[:na], ea[na:], eb[:nb], eb[nb:])
        e_i = U.urpa_plasmon(ov, *den_p)
        # (ii) PySCF RSGDF (same cart aux) + pyscf.gw.urpa arithmetic
        rs = pdf.RSGDF(pc)
        rs.auxbasis = auxc
        rs.build()
        Lpq = np.vstack(
            [
                np.asarray(LR).reshape(-1, pc.nao, pc.nao)
                for LR, LI, s in rs.sr_loop(compact=False)
            ]
        )
        ovL = [
            np.einsum("Pmn,mi,na->iaP", Lpq, P[:, :n], P[:, n:])
            for P, n in ((Pa, na), (Pb, nb))
        ]
        e_ov = [
            (e[:n][:, None] - e[n:][None, :]).ravel() for e, n in ((ea, na), (eb, nb))
        ]
        f_ov = [np.ones(x.size) for x in e_ov]
        e_ii = pyscf_urpa_loop(
            ovL, e_ov, f_ov, 40
        )  # an empty spin channel is an empty prange there
        print(
            f"   PySCF exxdiv={str(ex):5s} ({c}): dE_HF {mf.e_tot - src['e']:+.1e}  pbc.mp.UMP2 {pm.e_corr:+.12e} "
            f"(d {pm.e_corr - ours[c][0][0]:+.1e}; ss {pm.e_corr_ss - ours[c][0][1] - ours[c][0][2]:+.1e} "
            f"os {pm.e_corr_os - ours[c][0][3]:+.1e});  pbc UCCSD-eris MP2 {emp2_cc:+.12e} "
            f"(d vs ours shifted {emp2_cc - ours['shifted'][0][0]:+.1e})"
        )
        print(
            f"      URPA (i) PySCF AFTDF ov + our plasmon {e_i:+.12e} (d {e_i - ours[c][1]:+.1e});  (ii) PySCF RSGDF + "
            f"pyscf.gw.urpa loop nw=40 {e_ii:+.12e} (d vs ours RS-GDF n=40 {e_ii - ours_gdf[c]:+.1e}; RS-GDF fit "
            f"error {ours_gdf[c] - ours[c][1]:+.1e})  ({time.time() - t:.0f}s)",
            flush=True,
        )
