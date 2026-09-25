"""Stage 7 PySCF oracles.  PySCF 2.13 has NO periodic RPA (pyscf.pbc has gw/krgw_ac, kgw_slow, tdscf, but
no rpa module and no RPA correlation energy anywhere under pyscf/pbc).  The closest oracles are therefore:
 (i)  PySCF's periodic integrals + orbitals: pbc.scf.RHF on AFTDF (mesh 61^3), exxdiv None / 'ewald',
      (ia|jb) from mf.with_df.ao2mo, fed to OUR plasmon formula  vs  ours (pure-AFT ERI, own SCF);
 (ii) PySCF's MOLECULAR RPA arithmetic (pyscf.gw.rpa make_dielectric_matrix + _get_scaled_legendre_roots,
      i.e. rpa.kernel's loop) applied to PySCF's periodic RSGDF 3-index tensor (cart aux, same aux as ours)
      and PySCF's periodic orbitals  vs  ours from our RS-GDF B.
The molecular pyscf.gw.rpa.RPA itself is pinned in run_rpa_formula.py (1e-13)."""

import sys
import time
import numpy as np

sys.path.insert(0, ".")
from pyscf.pbc import gto as pgto, scf as pscf, df as pdf
from pyscf.gw import rpa as prpa
from pbc_gamma import Cell, build_integrals, rhf
from pbc_gdf import build_gdf, ferric_basis
from pbc_mp2 import denominators, bia_from_B
from pbc_rpa import gamma_drpa, drpa_plasmon, drpa_quad
from run_mp2_fit_error import (
    SYS,
)  # run_mp2_oracle has no __main__ guard: importing it runs it


class Eris:  # the three attributes + one method pyscf.gw.rpa.make_dielectric_matrix touches
    def __init__(self, ovL, nocc, nvir):
        self.ovL = ovL
        self.nocc, self.nvir, self.naux = nocc, nvir, ovL.shape[1]
        self.dtype = ovL.dtype

    def get_ov_blk(self, p0, p1):
        return self.ovL[p0:p1]


def pyscf_rpa_loop(ovL, e_ov, f_ov, nw, nocc, nvir, x0=0.5):
    e = 0.0
    er = Eris(ovL, nocc, nvir)
    for w, wt in zip(*prpa._get_scaled_legendre_roots(nw, x0)):
        diel = prpa.make_dielectric_matrix(w, e_ov, f_ov, er, blksize=ovL.shape[0])
        e += (
            wt
            / (2 * np.pi)
            * (np.log(np.linalg.det(np.eye(er.naux) - diel)) + np.trace(diel))
        )
    return e.real


for name in sys.argv[1:] or list(SYS):
    a, atoms, basis = SYS[name]
    nel = len(atoms)
    nocc = nel // 2
    t = time.time()
    cell = Cell(a, atoms, basis)
    ints = build_integrals(cell, None, exxdiv="ewald", verbose=False)
    vm = ints["madelung"]
    e_n, eps_n, _, C = rhf(
        ints["S"], ints["h"], ints["I"], ints["enn"], nel, conv=1e-12, return_mo=True
    )
    ours = {
        c: gamma_drpa(
            C,
            np.concatenate(denominators(eps_n, nocc, vm, c)),
            nocc,
            eri=ints["I"],
            method="plasmon",
        )
        for c in ("unshifted", "shifted")
    }
    auxc = ferric_basis("cc-pvdz-ri", ["H"])
    Bc = build_gdf(cell, auxc, spherical=False)["B"]
    ours_gdf = {
        c: gamma_drpa(C, np.concatenate(denominators(eps_n, nocc, vm, c)), nocc, B=Bc)
        for c in ("unshifted", "shifted")
    }
    print(
        f"{name}: ours ({time.time() - t:.0f}s) dRPA exact-AFT unshifted {ours['unshifted']:.12e} shifted {ours['shifted']:.12e}; "
        f"RS-GDF cart cc-pvdz-ri unshifted {ours_gdf['unshifted']:.12e} shifted {ours_gdf['shifted']:.12e}",
        flush=True,
    )
    pc = pgto.Cell(a=a, atom=atoms, basis=basis, unit="B", cart=True, verbose=0)
    pc.precision = 1e-12
    pc.build()
    for ex, conv in ((None, "unshifted"), ("ewald", "shifted")):
        t = time.time()
        mf = pscf.RHF(pc, exxdiv=ex)
        mf.with_df = pdf.AFTDF(pc)
        mf.with_df.mesh = [61] * 3
        mf.conv_tol = 1e-12
        mf.kernel()
        Co, Cv = mf.mo_coeff[:, :nocc], mf.mo_coeff[:, nocc:]
        eo, ev = mf.mo_energy[:nocc], mf.mo_energy[nocc:]
        nv = Cv.shape[1]
        ovov = mf.with_df.ao2mo((Co, Cv, Co, Cv), compact=False).real.reshape(
            nocc, nv, nocc, nv
        )
        e_i = drpa_plasmon(ovov, eo, ev)
        # (ii) PySCF RSGDF with the same (cart) aux, PySCF RPA arithmetic
        rs = pdf.RSGDF(pc)
        rs.auxbasis = auxc
        rs.build()
        Lpq = np.vstack(
            [
                np.asarray(LR).reshape(-1, pc.nao, pc.nao) * (1 if s > 0 else 1j)
                for LR, LI, s in rs.sr_loop(compact=False)
            ]
        )
        assert np.isrealobj(Lpq)
        ovL = np.einsum("Pmn,mi,na->iaP", Lpq, Co, Cv).reshape(nocc * nv, -1)
        e_ov = (eo[:, None] - ev[None, :]).ravel()
        f_ov = np.full(e_ov.size, 2.0)
        e_ii40 = pyscf_rpa_loop(ovL, e_ov, f_ov, 40, nocc, nv)
        e_ii = pyscf_rpa_loop(ovL, e_ov, f_ov, 200, nocc, nv)
        ours_same_n = drpa_quad(
            bia_from_B(Bc, C[:, :nocc], C[:, nocc:]),
            *denominators(eps_n, nocc, vm, conv),
            n=40,
        )
        print(
            f"   PySCF exxdiv={ex} ({conv}): (i) PySCF AFTDF (ia|jb)+plasmon {e_i:.12e} (d vs ours {e_i - ours[conv]:+.1e});  "
            f"(ii) PySCF RSGDF + pyscf.gw.rpa loop nw=200 {e_ii:.12e} (d vs ours RS-GDF {e_ii - ours_gdf[conv]:+.1e}); "
            f"nw=40 vs ours n=40 {e_ii40 - ours_same_n:+.1e}  ({time.time() - t:.0f}s)",
            flush=True,
        )
