"""AO-level k-point ERI tensor V_AO[ki, kj, ka][m, n, l, s] = sum_{K = G + q} P^{ki,ka}_{mn}(K) conj(P^{kb,kj}_{ls}(K))
(sqrt(v(K)/Omega) inside each P; kb = ki + kj - ka; K != 0), built in q-class chunks so every run stays short.

It contains everything the relaxation test needs at nao = 2 (H2/STO-3G):
  Jker[k, j]  = V_AO[k, j, k]        (q = 0; pbc_kpts.build_k's Jker, same K set)
  Kker[k, k'] = V_AO[k, k', k']      (build_k's Kker, index order [m, l, n, s])
  MP2 V_iajb for ANY orbitals: C_i(ki)^* C_a(ka) C_b(kb) C_j(kj)^* contracted (pbc_kcorr._accumulate's algebra)
The q = 0 chunk also stores V_ne(k) (build_k's G-sum).  Same gcut (prec 1e-10), thresh, normalisation as
run_kcorr_head_anomaly.run_mesh, so the saved mesh C reproduce its rows (anchor in relax.py).

usage: vao_build.py a m iq0 iq1 [orient]     -> vao_data/vao_a{a}_n{m}{suf}_q{iq0}-{iq1}.npz
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, "/home/matt/qc/ferric-pbc/reference/pbc")
import pbc_kcorr as KC  # noqa: E402
import pbc_kpts as PK  # noqa: E402
from pbc_gamma import Cell  # noqa: E402
from pbc_kpts import mesh_index, mp_mesh, shifted_gvectors  # noqa: E402
from pbc_supercell import Supercell, pair_ft_residues  # noqa: E402
from run_kpts_anchor import H2_ATOMS  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def h2_atoms(orient):
    if orient == "111":
        c, u = np.array([0.3, 0.2, 0.8]), np.ones(3) / np.sqrt(3)
        return [("H", tuple(c - 0.7 * u)), ("H", tuple(c + 0.7 * u))]
    return H2_ATOMS


def main():
    a, m, iq0, iq1 = (
        float(sys.argv[1]),
        int(sys.argv[2]),
        int(sys.argv[3]),
        int(sys.argv[4]),
    )
    orient = sys.argv[5] if len(sys.argv) > 5 else "z"
    suf = "" if orient == "z" else f"_{orient}"
    cell = Cell(np.eye(3) * a, h2_atoms(orient), "sto-3g")
    gcut = PK.aft_gcut(cell, 1e-10)
    n = (m, m, m)
    ints, kpts = mp_mesh(cell, n)
    Nk, nao = len(kpts), cell.mol.nao
    nn = KC._pair_norm(cell)
    scell = Supercell(cell.a, cell.atoms, cell.basis, n)
    ph_r = np.exp(1j * kpts @ scell.t.T)
    V = np.zeros((Nk, Nk, Nk, nao, nao, nao, nao), complex)
    Vk = np.zeros((Nk, nao, nao), complex)
    t0 = time.time()
    for iq in range(iq0, min(iq1, Nk)):
        q = kpts[iq]
        kof = [mesh_index(n, ints[j] - ints[iq]) for j in range(Nk)]
        K = shifted_gvectors(cell, q, gcut)
        for c0 in range(0, len(K), 4000):
            Kc = K[c0 : c0 + 4000]
            K2 = np.einsum("gi,gi->g", Kc, Kc)
            v = 4 * np.pi / K2 / cell.vol
            Q = pair_ft_residues(scell, Kc, 1e-14) * nn[None, :, :, None]
            P = np.einsum(
                "jr,rmng->jmng", ph_r, Q
            )  # P[j] = P^{kof[j], j}(K), raw (no sqrt v)
            T = np.einsum("xmng,ylsg->xymnls", P * v, P.conj(), optimize=True)
            for x in range(Nk):
                V[kof[x], :, x] += T[x]  # ki = kof[x], ka = x, kj = y, kb = kof[y]
            if iq == 0:
                SG = np.exp(-1j * Kc @ cell.R.T) @ cell.Z
                Vk -= np.einsum("kmng,g->kmn", P, 4 * np.pi / K2 * SG.conj()) / cell.vol
        print(f"  iq {iq} nK {len(K)} ({time.time() - t0:.0f}s)", flush=True)
    os.makedirs(f"{HERE}/vao_data", exist_ok=True)
    np.savez(
        f"{HERE}/vao_data/vao_a{a:g}_n{m}{suf}_q{iq0}-{iq1}.npz",
        V=V,
        Vk=Vk,
        iq0=iq0,
        iq1=iq1,
        gcut=gcut,
    )


if __name__ == "__main__":
    main()
