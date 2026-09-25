"""Iteration 21 PySCF oracle for the k-point RHF forces (pbc_kgrad.py).  Usage: python3 run_kgrad_oracle.py {h2|tri2} [fd]

PySCF 2.13 pbc.grad.krhf needs a pseudopotential cell for get_hcore (Iterations 16-18: NotImplementedError for all-electron
cells).  (0) records whether that still holds at k.  The oracle is assembled from what PySCF CAN do, at our DEFAULT
(converged, prec 1e-14) AFT cutoff so both sides are converged in the plane-wave sum:
  (1) cell.pbc_intor('int1e_ipovlp' / 'int1e_ipkin', kpts)   vs  sum_L e^{ik.L} <grad m_0|n_L>  (phase convention of
      the derivative blocks, matrix level, per k)
  (2) FFTDF(mesh) get_jk_e1(dm_kpts, kpts, exxdiv=None), contracted exactly as pyscf/pbc/grad/krhf.grad_elec does
      (2 Re sum_k tr[vhf_x(k)[A rows] D(k)[.., A rows]] / Nk), vs our parts J + K at the SAME D(k)
  (3) [fd] total force vs central FD (h = 1e-4) of PySCF's own KRHF + AFTDF energy (independent of every line of
      pbc_kgrad), one component, exxdiv None and 'ewald' (PySCF started from our D at each displacement).
Predictions: (1) <= 1e-13 (same molecular integrals, same phase convention as S(k) in Iteration 9); (2) <= 1e-12 when
the FFTDF mesh is converged (Iteration 16 Gamma: 9e-16 at mesh 61); (3) at the h^2 FD floor (~1e-9..3e-9 as
Iteration 16's -2.42e-9), same for both exxdiv.  Artifact: a phase-convention slip gives O(0.1) in (1) at k != 0 only.
"""
import os
import sys
import time

import numpy as np
from pyscf.pbc import df as pdf
from pyscf.pbc import gto as pgto
from pyscf.pbc import scf as pscf

import pbc_grad as PGd
import pbc_kgrad as KG
import pbc_kpts as PK
from pbc_gamma import Cell
from run_kgrad_anchor import H2, TRI2

MESH = int(os.environ.get("PYSCF_MESH", "61"))
SYS = {"h2": (H2, (1, 1, 3), 2, [(0, 2)]), "tri2": (TRI2, (1, 1, 3), 2, [(1, 1)])}


def pcell(s, atoms=None):
    pc = pgto.Cell(a=s["a"], atom=atoms or s["atoms"], basis=s["basis"], unit="B", cart=True, verbose=0)
    pc.precision = 1e-12
    pc.max_memory = int(os.environ.get("PYSCF_MAXMEM", "1500"))
    pc.mesh = [MESH] * 3
    return pc.build()


def pyscf_e(s, atoms, n, ex, dm0):
    pc = pcell(s, atoms)
    kp = pc.make_kpts(list(n))
    mf = pscf.KRHF(pc, kp, exxdiv=ex)
    mf.with_df = pdf.AFTDF(pc, kp)
    mf.with_df.mesh = [MESH] * 3
    mf.conv_tol = 1e-13
    mf.conv_tol_grad = 1e-8
    return mf.kernel(dm0=dm0), mf.make_rdm1()


def main(name, flags):
    s, n, nelec, comps = SYS[name]
    cell = Cell(s["a"], s["atoms"], s["basis"])
    pc = pcell(s)
    kp = pc.make_kpts(list(n))
    t = time.time()
    kb = PK.build_k(cell, n, exxdiv="ewald")  # default converged gcut
    kbn = dict(kb, madelung=0.0)
    print(f"[{name}] mesh {n}, gcut {kb['gcut']:.2f}, build {time.time() - t:.0f}s", flush=True)
    assert np.allclose(kp, kb["kpts"])
    res = {}
    for ex, k in (("none", kbn), ("ewald", kb)):
        guess = None if ex == "none" else (res["none"]["r"]["Da"], res["none"]["r"]["Db"])
        r = KG.kscf(k, nelec // 2, nelec // 2, restricted=True, guess=guess)
        g, parts = KG.kgrad(cell, k, r["Da"], r["Db"], r["Fa"], r["Fb"], restricted=True)
        res[ex] = dict(r=r, g=g, parts=parts)
        print(f"  ours {ex:5s}: E {r['e']:.12f} err {r['err']:.1e} ({time.time() - t:.0f}s)\n"
              + np.array2string(g, precision=12), flush=True)
    D = res["none"]["r"]["Da"] * 2

    # (0) PySCF's own k-point gradient on an all-electron cell
    mf = pscf.KRHF(pc, kp, exxdiv=None)
    mf.with_df = pdf.AFTDF(pc, kp)
    mf.mo_coeff, mf.mo_occ, mf.mo_energy = None, None, None
    try:
        from pyscf.pbc.grad import krhf as pkgrad

        gobj = pkgrad.Gradients(mf)
        gobj.get_hcore(pc, kp)
        print("  (0) pbc.grad.krhf get_hcore on an all-electron cell: ACCEPTED (unexpected)")
    except NotImplementedError as e:
        print(f"  (0) pbc.grad.krhf get_hcore on an all-electron cell: NotImplementedError {e!r}")
    except Exception as e:  # noqa: BLE001
        print(f"  (0) pbc.grad.krhf: {type(e).__name__}: {e}")

    # (1) derivative-block phase convention
    for o in ("int1e_ipovlp", "int1e_ipkin"):
        ref = np.asarray(pc.pbc_intor(o, comp=3, hermi=0, kpts=kp))  # (Nk, 3, nao, nao)
        L1, X = KG.image_ip(cell, o + "_cart")
        ours = np.einsum("kL,xmLn->kxmn", np.exp(1j * kb["kpts"] @ L1.T), X)
        print(f"  (1) {o:13s} max|ours - PySCF| over k = {abs(ours - ref).max():.1e} (max|Im| {abs(ref.imag).max():.1e})")

    # (2) FFTDF 2e derivative contracted with OUR D(k), exxdiv=None
    t = time.time()
    fft = pdf.FFTDF(pc, kp)
    fft.mesh = [MESH] * 3
    vj, vk = fft.get_jk_e1(D, kp, exxdiv=None)
    vhf = np.asarray(vj).reshape(3, len(kp), *D.shape[1:]) - 0.5 * np.asarray(vk).reshape(3, len(kp), *D.shape[1:])
    aos = pc.aoslice_by_atom()
    g2 = np.zeros((pc.natm, 3))
    for A in range(pc.natm):
        p0, p1 = aos[A, 2:]
        g2[A] = 2 * np.einsum("xkij,kji->x", vhf[:, :, p0:p1], D[:, :, p0:p1]).real / len(kp)
    ours2 = res["none"]["parts"]["J"] + res["none"]["parts"]["K"]
    print(f"  (2) FFTDF(mesh {MESH}) get_jk_e1 vs our J + K: max diff {abs(g2 - ours2).max():.1e} "
          f"(|J+K| {abs(ours2).max():.2e}; {time.time() - t:.0f}s)", flush=True)

    # (3) FD of PySCF's own energy
    if "fd" in flags:
        for ex in ("none", "ewald"):
            r = res[ex]["r"]
            dm = r["Da"] * 2
            for A, x in comps:
                t = time.time()
                ep, _ = pyscf_e(s, PGd.displaced(cell, A, x, 1e-4).atoms, n, None if ex == "none" else ex, dm)
                em, _ = pyscf_e(s, PGd.displaced(cell, A, x, -1e-4).atoms, n, None if ex == "none" else ex, dm)
                fd = (ep - em) / 2e-4
                a = res[ex]["g"][A, x]
                print(f"  (3) exxdiv={ex:5s} dE/dR[{A},{x}]: PySCF-AFTDF FD {fd:+.12f} ours {a:+.12f} diff {a - fd:+.2e} "
                      f"({time.time() - t:.0f}s)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], set(sys.argv[2:]))
