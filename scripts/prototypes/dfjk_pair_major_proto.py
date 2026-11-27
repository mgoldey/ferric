#!/usr/bin/env python3
"""Prototype: pair-major, Schwarz-sparse storage for RI-JK (J and K from one dressed tensor).

ferric's RI-JK today (crates/ferric-scf/src/df_k.rs, df_j.rs) holds two tensors:
the raw `(P|mu nu)` for J and the dressed `B = L^-1 (P|mu nu)` for K, both
aux-major (`[P][mu nu]`, packed triangle), every AO pair stored. K unpacks one
`B_P` at a time and runs DSYMM against C_occ, then one DSYRK per aux chunk.

The candidate keeps ONE tensor, AO-major:

    B_mu[P, nu]   for nu in sig(mu) = {nu : shell pair (mu, nu) survives Schwarz}

  * integrals are generated pair-outer (not prototyped here: PySCF supplies them),
  * dressing is an in-place TRSM per mu on the columns nu <= mu, then a mirror,
  * J is two GEMVs per mu on the dressed tensor (no raw tensor, no V^-1 solve),
  * K half-transforms with one GEMM per mu against the gathered rows C[sig(mu)],
    then one SYRK over the whole (n, naux*nocc) intermediate.

This script checks the candidate against PySCF's DF J/K on the same auxiliary
basis and times the BLAS shapes of both formulations at one thread. Both run in
this process under the same BLAS, so the RATIOS are evidence about shapes and
FLOPs; the absolute numbers say nothing about the Rust wall time.

    python3 scripts/prototypes/dfjk_pair_major_proto.py [ncarbon=12] [basis=cc-pvdz] [thresh=1e-12]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.linalg.blas as blas

sys.path.insert(0, str(Path(__file__).resolve().parent))
from protokit import Stages, alkane, bench, df_inputs, jk_error, rhf_reference, threads  # noqa: E402

AUX = "def2-universal-jkfit"


# --------------------------------------------------------------------------- screening / layout


def schwarz_sig(
    mol, eri3: np.ndarray, metric: np.ndarray, thresh: float
) -> list[np.ndarray]:
    """Per-function significant partners, decided per SHELL pair as ferric's `PairScreen` does:
    keep (s1, s2) iff max sqrt((mu nu|mu nu)) * max_P sqrt((P|P)) > thresh."""
    n = eri3.shape[1]
    ao_loc = mol.ao_loc_nr()
    # (mu nu|mu nu) is a four-centre diagonal: one shell quartet per shell pair.
    diag = np.empty((n, n))
    for s1 in range(mol.nbas):
        for s2 in range(s1 + 1):
            blk = mol.intor_by_shell("int2e_sph", (s1, s2, s1, s2))
            d = np.einsum("ijij->ij", blk)
            diag[ao_loc[s1] : ao_loc[s1 + 1], ao_loc[s2] : ao_loc[s2 + 1]] = d
            diag[ao_loc[s2] : ao_loc[s2 + 1], ao_loc[s1] : ao_loc[s1 + 1]] = d.T
    q = np.sqrt(np.abs(diag))
    q_aux = np.sqrt(np.abs(metric.diagonal())).max()
    keep = np.zeros((n, n), dtype=bool)
    for s1 in range(mol.nbas):
        a = slice(ao_loc[s1], ao_loc[s1 + 1])
        for s2 in range(mol.nbas):
            b = slice(ao_loc[s2], ao_loc[s2 + 1])
            keep[a, b] = q[a, b].max() * q_aux > thresh
    return [np.flatnonzero(keep[mu]) for mu in range(n)]


class PairMajor:
    """`blocks[mu]` is the Fortran-ordered `(naux, len(sig[mu]))` block `B_mu[P, nu in sig(mu)]`."""

    def __init__(self, eri3: np.ndarray, sig: list[np.ndarray]):
        self.sig = sig
        self.naux, self.nao = eri3.shape[0], eri3.shape[1]
        self.blocks = [
            np.asfortranarray(eri3[:, mu, sig[mu]]) for mu in range(self.nao)
        ]
        # Position of nu inside sig[mu], for the mirror.
        self.pos = [dict(zip(s.tolist(), range(len(s)))) for s in sig]

    @property
    def stored(self) -> int:
        return sum(b.shape[1] for b in self.blocks)

    def dress_in_place(self, low: np.ndarray, st: Stages) -> None:
        """`B_mu[:, nu <= mu] <- L^-1 B_mu[:, nu <= mu]` (TRSM), then mirror to nu > mu."""
        low_f = np.asfortranarray(low)
        for mu, blk in enumerate(self.blocks):
            nlow = int(np.searchsorted(self.sig[mu], mu, side="right"))
            with st("dress: trsm (lower triangle)"):
                blk[:, :nlow] = blas.dtrsm(1.0, low_f, blk[:, :nlow], lower=1)
        with st("dress: mirror"):
            for mu, blk in enumerate(self.blocks):
                s = self.sig[mu]
                for col in range(int(np.searchsorted(s, mu, side="right")), len(s)):
                    nu = int(s[col])
                    blk[:, col] = self.blocks[nu][:, self.pos[nu][mu]]

    def j(self, dm: np.ndarray) -> np.ndarray:
        """`J = sum_P B_P (B_P . D)`: two GEMVs per mu."""
        d = np.zeros(self.naux)
        for mu, blk in enumerate(self.blocks):
            d += blk @ dm[mu, self.sig[mu]]
        out = np.zeros((self.nao, self.nao))
        for mu, blk in enumerate(self.blocks):
            out[mu, self.sig[mu]] = d @ blk
        return out

    def k(self, c_occ: np.ndarray, st: Stages | None = None) -> np.ndarray:
        """`K = T T^T`, `T[mu, (P, i)] = sum_{nu in sig(mu)} B_mu[P, nu] C[nu, i]`."""
        st = st or Stages()
        nocc = c_occ.shape[1]
        t = np.empty(
            (self.naux * nocc, self.nao), order="F"
        )  # column mu = T[mu] flattened
        for mu, blk in enumerate(self.blocks):
            with st("K: gather C rows"):
                c_sig = np.asfortranarray(c_occ[self.sig[mu]])
            with st("K: half-transform GEMM"):
                t[:, mu] = blas.dgemm(1.0, blk, c_sig).ravel(order="F")
        with st("K: SYRK"):
            k = blas.dsyrk(1.0, t, trans=1, lower=1)
        return np.tril(k) + np.tril(k, -1).T


# --------------------------------------------------------------------------- the aux-major baseline


def k_aux_major(
    b: np.ndarray, c_occ: np.ndarray, chunk: int, st: Stages | None = None
) -> np.ndarray:
    """ferric's occupied path: per P a DSYMM into an (n, chunk*nocc) panel, one DSYRK per chunk.
    The packed-row unpack ferric also pays is NOT charged here (the baseline is flattered)."""
    st = st or Stages()
    naux, n, _ = b.shape
    nocc = c_occ.shape[1]
    c_f = np.asfortranarray(c_occ)
    k = np.zeros((n, n))
    for q0 in range(0, naux, chunk):
        q1 = min(q0 + chunk, naux)
        panel = np.empty((n, (q1 - q0) * nocc), order="F")
        with st("K0: DSYMM per P"):
            for p in range(q0, q1):
                off = (p - q0) * nocc
                panel[:, off : off + nocc] = blas.dsymm(1.0, b[p], c_f, lower=1)
        with st("K0: DSYRK per chunk"):
            k += blas.dsyrk(1.0, panel, lower=1)
    return np.tril(k) + np.tril(k, -1).T


def dress_aux_major_flops(naux: int, n: int) -> float:
    """TRMM-shaped dressing of every lower-triangle pair with a triangular `L^-1`."""
    return naux * naux * n * (n + 1) / 2


# --------------------------------------------------------------------------- driver


def main() -> None:
    nc = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    basis = sys.argv[2] if len(sys.argv) > 2 else "cc-pvdz"
    thresh = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-12
    reps = 3

    mol = alkane(nc, basis)
    with threads(2):
        ints = df_inputs(mol, AUX)
        ref = rhf_reference(mol, AUX)
    n, naux, nocc = ints.nao, ints.naux, ref.nocc
    c_occ = ref.c_occ
    print(
        f"C{nc}H{2 * nc + 2}/{basis}  n={n} naux={naux} nocc={nocc}  Schwarz thresh={thresh:g}"
    )

    with threads(1):
        sig = schwarz_sig(mol, ints.eri3, ints.metric, thresh)
        pm = PairMajor(ints.eri3, sig)
        frac = pm.stored / (n * n)
        low_pairs = sum(
            int(np.searchsorted(s, mu, side="right")) for mu, s in enumerate(sig)
        )
        print(
            f"kept function pairs: {frac:.3f} of n^2  ({low_pairs} of {n * (n + 1) // 2} lower-triangle)"
        )
        gb = 8 / 1e9
        print(
            "resident tensors: aux-major raw+dressed packed "
            f"{2 * naux * n * (n + 1) / 2 * gb:.2f} GB  vs  pair-major dressed {naux * pm.stored * gb:.2f} GB"
        )

        low = np.linalg.cholesky(ints.metric)
        st = Stages()
        pm.dress_in_place(low, st)
        st.report("pair-major dressing (1 thread)")
        f_new, f_old = naux * naux * low_pairs, dress_aux_major_flops(naux, n)
        print(
            f"dressing FLOPs: aux-major {f_old / 1e9:.1f} G  pair-major {f_new / 1e9:.1f} G  ({f_old / f_new:.2f}x)"
        )

        # ---- correctness against PySCF's DF J/K on the same aux basis
        j_new = pm.j(ref.dm)
        k_new = 2.0 * pm.k(c_occ)
        ej, ek = jk_error(j_new, ref.j, ref.dm), jk_error(k_new, ref.k, ref.dm)
        print("J vs PySCF DF:", {k: f"{v:.2e}" for k, v in ej.items()})
        print("K vs PySCF DF:", {k: f"{v:.2e}" for k, v in ek.items()})
        e2_new = 0.5 * np.einsum("ij,ij->", ref.dm, j_new - 0.5 * k_new)
        e2_ref = 0.5 * np.einsum("ij,ij->", ref.dm, ref.j - 0.5 * ref.k)
        print(f"two-electron energy error: {abs(e2_new - e2_ref):.2e} Ha")

        # ---- K kernel shapes, one thread
        b_ref = ints.dressed()
        chunk = min(max(4096 // n, 4), 64)  # three_index_source::df_chunk_width
        k_old = 2.0 * k_aux_major(b_ref, c_occ, chunk)
        print(
            "aux-major K vs PySCF DF:",
            {k: f"{v:.2e}" for k, v in jk_error(k_old, ref.k, ref.dm).items()},
        )
        s_old, s_new = Stages(), Stages()
        t_old = bench(lambda: k_aux_major(b_ref, c_occ, chunk, s_old), reps=reps)
        t_new = bench(lambda: pm.k(c_occ, s_new), reps=reps)
        s_old.report(f"aux-major K, {reps + 1} builds")
        s_new.report(f"pair-major K, {reps + 1} builds")
        fl_old = 2 * naux * n * n * nocc + naux * nocc * n * n
        fl_new = 2 * naux * pm.stored * nocc + naux * nocc * n * n
        print(
            f"K build (1 thread): aux-major {t_old}  ({fl_old / t_old.median / 1e9:.1f} GF/s)\n"
            f"                    pair-major {t_new}  ({fl_new / t_new.median / 1e9:.1f} GF/s)\n"
            f"                    speedup {t_old.median / t_new.median:.2f}x  (FLOP ratio {fl_old / fl_new:.2f}x)"
        )

        t_j = bench(lambda: pm.j(ref.dm), reps=reps)
        print(f"pair-major J (1 thread): {t_j}")


if __name__ == "__main__":
    main()
