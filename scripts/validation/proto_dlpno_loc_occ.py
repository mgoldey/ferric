"""Prototype (numpy/PySCF) of DLPNO-MP2 with Boys-localized occupieds (#475).

Formulation under test: with occupieds rotated by U (C_loc = C_occ U), the
Fock occupied block is f = U^T diag(eps_occ) U (non-diagonal), virtuals stay
canonical, and the RI-MP2 amplitude equation is

  K_ij^ab + (e_a+e_b) T_ij^ab - sum_k f_ik T_kj^ab - sum_k f_kj T_ik^ab = 0

solved by Jacobi on the f off-diagonal with DIIS, over the retained pair set
only. Energy E = sum_ij sum_ab K_ij^ab (2 T_ij^ab - T_ij^ba) (full i,j sum).

Hypotheses (written BEFORE measuring pair retention):
  X (physics): localized occupieds give spatially compact pair energies, so
      the diagonal-f estimate |e_ij| of distant pairs falls below t_cut_pairs
      and pair retention < 1 at C6/C8 for a loose-enough t_cut_pairs.
  Y (artifact): the estimate is rotation-covariant only through the diagonal
      of f_loc; if the estimator or solve were broken, untruncated energy would
      differ from canonical (anchor) -- and retention would equal the canonical
      value regardless of U.
Usage: proto_dlpno_loc_occ.py <xyz> <basis> [aux]
"""

import sys

import numpy as np
from pyscf import df, gto, lo, scf


def run(xyz, basis, aux="cc-pvdz-ri", tcuts=(1e-4, 1e-5, 1e-6)):
    mol = gto.M(atom=xyz, basis=basis, verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-11)
    nocc = mol.nelectron // 2
    c = mf.mo_coeff
    eps = mf.mo_energy
    ncore = sum(1 for a in range(mol.natm) if mol.atom_charge(a) > 2)
    # block-diagonal Boys: core and valence separately
    u = np.zeros((nocc, nocc))
    cl = np.zeros((mol.nao, nocc))
    for sl in (slice(0, ncore), slice(ncore, nocc)):
        cb = c[:, sl]
        cb_loc = lo.Boys(mol, cb).kernel()
        u[sl, sl] = cb.T @ mf.get_ovlp() @ cb_loc
        cl[:, sl] = cb_loc
    assert np.allclose(u.T @ u, np.eye(nocc), atol=1e-8)
    auxmol = df.addons.make_auxmol(mol, aux)
    j3 = df.incore.aux_e2(mol, auxmol, "int3c2e")
    j2 = auxmol.intor("int2c2e")
    lm = np.linalg.cholesky(j2)
    nao = mol.nao
    j3 = j3.reshape(nao, nao, -1)
    l3 = np.linalg.solve(lm, j3.reshape(nao * nao, -1).T).reshape(-1, nao, nao)
    cv = c[:, nocc:]
    b_ov = np.einsum("Pmn,mi,na->Pia", l3, c[:, :nocc], cv)
    b_loc = np.einsum("Pia,ik->Pka", b_ov, u)
    ev = eps[nocc:]
    f = u.T @ np.diag(eps[:nocc]) @ u
    out = {}

    def kblock(b, i, j):
        return b[:, i, :].T @ b[:, j, :]

    # canonical reference
    e_can = 0.0
    for i in range(nocc):
        for j in range(nocc):
            k = kblock(b_ov, i, j)
            d = eps[i] + eps[j] - ev[:, None] - ev[None, :]
            t = k / d
            e_can += np.sum(k * (2 * t - t.T))
    out["e_can"] = e_can

    def solve(b, pairs, maxit=200, tol=1e-10):
        K = {p: kblock(b, *p) for p in pairs}
        T = {
            p: K[p] / (f[p[0], p[0]] + f[p[1], p[1]] - ev[:, None] - ev[None, :])
            for p in pairs
        }
        ps = set(pairs)
        for it in range(maxit):
            rmax = 0.0
            Tn = {}
            for i, j in pairs:
                acc = -K[(i, j)].copy()
                for k in range(nocc):
                    if k != i and (k, j) in ps:
                        acc += f[i, k] * T[(k, j)]
                    if k != j and (i, k) in ps:
                        acc += f[k, j] * T[(i, k)]
                d = f[i, i] + f[j, j] - ev[:, None] - ev[None, :]
                # (e_a+e_b-fii-fjj) T = -K + sum f T  ->  T = (-K+sum)/(-d)
                Tn[(i, j)] = acc / (-d)
                rmax = max(rmax, np.abs(Tn[(i, j)] - T[(i, j)]).max())
            T = Tn
            if rmax < tol:
                break
        e = sum(np.sum(K[p] * (2 * T[p] - T[p].T)) for p in pairs)
        return e, it + 1

    allp = [(i, j) for i in range(nocc) for j in range(nocc)]
    e_full, nit = solve(b_loc, allp)
    out["e_loc_full"] = e_full
    out["iters"] = nit
    # pair estimates (diagonal f), unique-pair weighted
    est = np.zeros((nocc, nocc))
    for i in range(nocc):
        for j in range(nocc):
            k = kblock(b_loc, i, j)
            d = f[i, i] + f[j, j] - ev[:, None] - ev[None, :]
            t = k / d
            est[i, j] = np.sum(k * (2 * t - t.T))
    out["ret"] = {}
    for tc in tcuts:
        # symmetric pair energy e_ij+e_ji vs unique-pair; use (i<=j) weight
        keep = [
            (i, j)
            for i in range(nocc)
            for j in range(nocc)
            if abs(est[i, j] + (est[j, i] if i != j else 0.0)) >= tc
        ]
        e_t, _ = solve(b_loc, keep)
        out["ret"][tc] = (len(keep) / nocc**2, e_t - e_can)
    return out


if __name__ == "__main__":
    xyz = sys.argv[1]
    print(run(xyz, sys.argv[2]))
