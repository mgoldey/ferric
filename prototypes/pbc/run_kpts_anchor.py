"""Stage 3 exactness anchors for pbc_kpts (k-point RHF, pure AFT).

(a) 1x1x1 mesh == pbc_gamma (build_integrals(cell, None) + rhf), default gcut, both exxdiv.
(b) N1xN2xN3 k-mesh E/cell == Gamma RHF of the explicit supercell / N (pbc_kpts.gamma_aft, which uses
    pbc_gamma.pair_ft directly, no residue fold).  Same |K| <= gcut sphere on both sides, so the two sum
    IDENTICAL terms: the anchor is exact at any gcut (a looser gcut is used to fit the supercell in memory).
(c) S(k) Hermitian, S(-k) = S(k)^*, F(k) Hermitian; Madelung: kmesh_madelung == PySCF tools.pbc.madelung(cell, kpts).
Mutations (pbc_kpts._MUTANT) must break (b).

Usage: python3 run_kpts_anchor.py [a|b|c|mut] ...
"""

import sys
import time

import numpy as np

sys.path.insert(0, ".")
import pbc_kpts as PK  # noqa: E402
from pbc_gamma import Cell, build_integrals, madelung, rhf  # noqa: E402
from pbc_kpts import build_k, gamma_aft, krhf, supercell_cell  # noqa: E402
from pyscf import gto  # noqa: E402

H2_A = np.eye(3) * 4.0
H2_ATOMS = [("H", (0.3, 0.2, 0.1)), ("H", (0.3, 0.2, 1.5))]
TRI_A = np.array([[4.6, 0.0, 0.0], [0.9, 4.3, 0.0], [0.5, 0.7, 4.8]])
TRI_ATOMS = [
    ("H", (0.1, 0.2, 0.3)),
    ("H", (0.1, 0.2, 1.7)),
    ("H", (2.4, 2.5, 2.2)),
    ("H", (3.6, 2.9, 2.6)),
]
SP = {
    "H": gto.parse("""
H S
  3.42525091  0.15432897
  0.62391373  0.53532814
  0.16885540  0.44463454
H P
  0.8         1.0
""")
}


def anchor_a():
    cell = Cell(H2_A, H2_ATOMS, "sto-3g")
    g = build_integrals(cell, None, exxdiv="ewald", verbose=False)
    kb = build_k(cell, (1, 1, 1), exxdiv="ewald")
    for ex in ("none", "ewald"):
        vm = g["madelung"] if ex == "ewald" else 0.0
        e_g, eps_g, _ = rhf(g["S"], g["h"], g["I"], g["enn"], 2, conv=1e-12, kshift=vm)
        e_k, eps_k, _ = krhf(kb, 2, conv=1e-12, kshift=vm)
        print(
            f"(a) H2 1x1x1 {ex}: E_k {e_k:.12f} E_gamma {e_g:.12f} dE {e_k - e_g:.1e} "
            f"deps {abs(eps_k[0] - eps_g).max():.1e} max|Im S| {abs(kb['S'].imag).max():.1e}"
        )


def anchor_b(name, cell, n, nelec, prec):
    gcut = PK.aft_gcut(cell, prec)
    t = time.time()
    kb = build_k(cell, n, exxdiv="ewald", gcut=gcut)
    sc = supercell_cell(cell, n)
    t1 = time.time()
    g = gamma_aft(sc, gcut=gcut)
    Nk = int(np.prod(n))
    vm_sc = madelung(sc)
    print(
        f"  supercell build {time.time() - t1:.1f}s nG_sc={g['nG']}; v_M kmesh {kb['madelung']:.12f} "
        f"supercell {vm_sc:.12f} d {kb['madelung'] - vm_sc:.1e}"
    )
    out = {}
    for ex in ("none", "ewald"):
        vm = kb["madelung"] if ex == "ewald" else 0.0
        e_k, eps_k, it = krhf(kb, nelec, conv=1e-12, kshift=vm)
        e_s, eps_s, _ = rhf(
            g["S"],
            g["h"],
            g["I"],
            g["enn"],
            nelec * Nk,
            conv=1e-12,
            kshift=(vm_sc if vm else 0.0),
        )
        ek = np.sort(np.concatenate(eps_k))
        out[ex] = e_k
        print(
            f"(b) {name} {n} {ex}: E_k/cell {e_k:.12f} E_sc/N {e_s / Nk:.12f} dE {e_k - e_s / Nk:.1e} "
            f"d(all eps) {abs(ek - np.sort(eps_s)).max():.1e} it {it} ({time.time() - t:.0f}s)",
            flush=True,
        )
    return kb, out


def anchor_c(cell, n):
    kb = build_k(cell, n, exxdiv="ewald", gcut=PK.aft_gcut(cell, 1e-8))
    S, ints = kb["S"], kb["ints"]
    herm = max(abs(S[k] - S[k].conj().T).max() for k in range(kb["Nk"]))
    tr = max(
        abs(S[PK.mesh_index(n, -ints[k])] - S[k].conj()).max() for k in range(kb["Nk"])
    )
    V = kb["V"]
    vtr = max(
        abs(V[PK.mesh_index(n, -ints[k])] - V[k].conj()).max() for k in range(kb["Nk"])
    )
    dm = np.array(
        [
            np.eye(S.shape[1]) * 0.3
            + 0.05j * (np.triu(np.ones_like(S[0]), 1) - np.tril(np.ones_like(S[0]), -1))
            for _ in range(kb["Nk"])
        ]
    )
    J, K = PK.jk_k(kb, dm)
    hj = max(abs(J[k] - J[k].conj().T).max() for k in range(kb["Nk"]))
    hk = max(abs(K[k] - K[k].conj().T).max() for k in range(kb["Nk"]))
    from pyscf.pbc import gto as pgto
    from pyscf.pbc import tools as ptools

    pc = pgto.Cell(
        a=cell.a, atom=cell.atoms, basis=cell.basis, unit="B", cart=True, verbose=0
    )
    pc.precision = 1e-12
    pc.build()
    kp = pc.make_kpts(list(n))
    s_py = np.asarray(pc.pbc_intor("int1e_ovlp", hermi=1, kpts=kp))
    vm_py = ptools.pbc.madelung(pc, kp)
    print(
        f"(c) {n}: |S-S^H| {herm:.1e}, |S(-k)-S(k)*| {tr:.1e}, |V(-k)-V(k)*| {vtr:.1e}, "
        f"|J-J^H| {hj:.1e}, |K-K^H| {hk:.1e} (Hermitian test dm), |kpts - pyscf| {abs(kp - kb['kpts']).max():.1e}, "
        f"|S - pyscf S(k)| {abs(s_py - S).max():.1e}, |S - conj pyscf| {abs(s_py.conj() - S).max():.1e}, "
        f"v_M {kb['madelung']:.12f} pyscf {vm_py:.12f} d {kb['madelung'] - vm_py:.1e}"
    )


if __name__ == "__main__":
    what = sys.argv[1:] or ["a", "b", "c"]
    h2 = Cell(H2_A, H2_ATOMS, "sto-3g")
    tri = Cell(TRI_A, TRI_ATOMS, SP)
    if "a" in what:
        anchor_a()
    if "c" in what:
        anchor_c(h2, (2, 2, 3))
        anchor_c(tri, (1, 2, 3))
    if "b" in what:
        anchor_b("H2", h2, (1, 1, 3), 2, 1e-8)
        anchor_b("H2", h2, (2, 2, 2), 2, 1e-6)
        anchor_b("tri", tri, (1, 1, 3), 4, 1e-6)
    if "mut" in what:
        for m in ("phase_P", "kernel_no_q", "madelung_prim"):
            PK._MUTANT = m
            print("MUTANT", m)
            anchor_b("H2", h2, (1, 1, 3), 2, 1e-8)
        PK._MUTANT = None
