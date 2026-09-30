"""The q = 0 Fock (exchange) head with TRUE Bloch pairing, per k and band pair, incl. the ov block that rotates the
mesh orbitals (candidates (c) and (e)).

    dK(k) = (4pi/(Omega Nk)) lim_{q->0} < [N(q) - N(0)] / q^2 >_dir,   N(q) = P^{k,k+q}(q) dm(k+q) P^{k,k+q}(q)^H
(the mesh with exxdiv='ewald' already has v_M N(0) = v_M S dm S; the degree -1 part is odd and cancels in the +-q
average; the degree-0 part is a quadratic form in q^, l <= 2, so the cubic +-x,y,z average is exact).
    fixed-k (run_kcorr_head_anomaly.fock_head_shifts):  P^{kk}(q), dm(k)        (no k-derivative of dm or pairing)
    true:                                                P^{k,k+q}(q), dm(k+q)  (orbitals at k+q from a Wannier/Fourier
                                                         interpolation of the mesh Fock, S(k+q) exact)
evaluated by central differences at small delta (Richardson over two deltas).

Artifact hypotheses (before running)
  * finite-delta: fixed-k evaluated THIS way must equal the analytic A/B formula (npz A, B) to O(delta^2) — an
    error in the pair-FT-per-L or the normalisation would show as an O(1) mismatch.
  * Fourier interpolation of F: repeat with F_none and F_ewald (same eigenvectors on the mesh, different
    interpolants); a difference comparable to (true - fixed) voids the true numbers.
  * independent check of the diagonal: the k-averaged eigenvalue shifts must match the EMPIRICAL mesh drift of the
    ewald eigenvalues (FINDINGS H1 table: k-avg occ -0.0465, vir +0.0173 x n^-3 from n = 4 -> 5), not just the
    fixed-k -0.0458/+0.0179.
Physics: nonzero ov elements => the mesh orbitals are rotated by kappa = O(1/Nk) relative to the converged ones,
and the MP2 NUMERATORS carry an O(1/Nk) error that E3 (eigenvalue-only Fock head) does not correct.
"""
import itertools
import sys

import numpy as np
import scipy.linalg as sla

sys.path.insert(0, "/home/matt/qc/ferric-pbc/reference/pbc")
sys.path.insert(0, "/home/matt/qc/ferric-pbc/prototypes/pbc/kmp2_head")
import pbc_kcorr as KC  # noqa: E402
from pbc_gamma import Cell, shell_table  # noqa: E402
from pbc_kpts import mp_mesh  # noqa: E402
from pbc_supercell import pair_ft_residues  # noqa: E402
from cheap_account import fock_dK, load  # noqa: E402
from vao_build import h2_atoms  # noqa: E402


class _PerL:
    """Supercell stand-in for pair_ft_residues: one residue per translation (no folding)."""

    def __init__(self, cell, thresh=1e-14):
        self.prim = cell
        sh = shell_table(cell.mol)
        amin = min(s["exps"].min() for s in sh)
        self.Ls = cell.translations(np.sqrt(2 * np.log(1 / thresh) / amin) + 2.0)
        self.R = len(self.Ls)

    def residues(self, Ls):
        assert len(Ls) == self.R and np.allclose(Ls, self.Ls)
        return np.arange(self.R)


class Bands:
    def __init__(self, cell, z, m, conv="ewald", rcut=22.0):
        self.cell, self.m = cell, m
        ints, kpts = mp_mesh(cell, (m, m, m))
        self.kpts = kpts
        L1 = cell.translations(rcut)
        sm = cell.supermol(L1)
        nao, nb0 = cell.mol.nao, cell.mol.nbas
        self.L1 = L1
        self.S_L = sm.intor("int1e_ovlp_cart", shls_slice=(0, nb0, 0, sm.nbas)).reshape(nao, len(L1), nao)
        C, eps = z["C"], z["eps"].copy()
        if conv == "ewald":
            eps[:, 0] -= float(z["vm"])
        Sk = np.array([self.S(k) for k in kpts])
        SC = np.einsum("kmn,knp->kmp", Sk, C)
        Fk = np.einsum("kmp,kp,knp->kmn", SC, eps, SC.conj())
        rng = range(-(m // 2), m // 2 + 1)
        self.FL = []
        for mv in itertools.product(rng, repeat=3):
            w = np.prod([0.5 if (m % 2 == 0 and abs(x) == m // 2) else 1.0 for x in mv])
            L = np.array(mv, float) @ cell.a
            self.FL.append((L, w * np.einsum("k,kmn->mn", np.exp(-1j * kpts @ L), Fk) / len(kpts)))
        self.C, self.eps = C, eps
        # per-L pair FT machinery
        self.pl = _PerL(cell)
        self.nn = KC._pair_norm(cell)

    def S(self, k):
        return np.einsum("L,mLn->mn", np.exp(1j * self.L1 @ k), self.S_L)

    def F(self, k):
        return sum(np.exp(1j * L @ k) * FL for L, FL in self.FL)

    def orbitals(self, k):
        e, c = sla.eigh(self.F(k), self.S(k))
        return e, c

    def QL(self, K):
        return pair_ft_residues(self.pl, np.atleast_2d(K), 1e-14)[..., 0] * self.nn[None]

    def P(self, QL, kp):
        return np.einsum("L,Lmn->mn", np.exp(1j * self.pl.Ls @ kp), QL)


def dK_mo(b, z, deltas=(4e-3, 2e-3), true=True):
    """(Nk, 2, 2) MO-basis dK per k (mesh MO basis), Richardson over two deltas."""
    cell, Nk = b.cell, len(b.kpts)
    pref = 4 * np.pi / (cell.vol * Nk)
    res = []
    keys = [(dl, ax, sg) for dl in deltas for ax, sg in itertools.product(range(3), (1, -1))]
    Ks = np.zeros((len(keys) + 1, 3))
    for i, (dl, ax, sg) in enumerate(keys):
        Ks[i, ax] = sg * dl
    Qall = pair_ft_residues(b.pl, Ks, 1e-14) * b.nn[None, :, :, None]
    Qcache = {key: Qall[..., i] for i, key in enumerate(keys)}
    Q0 = Qall[..., -1]
    out = []
    for k in range(Nk):
        kv = b.kpts[k]
        Ck = b.C[k]
        dm0 = 2 * Ck[:, :1] @ Ck[:, :1].conj().T
        S0 = b.P(Q0, kv)
        N0 = S0 @ dm0 @ S0.conj().T
        vals = []
        for dl in deltas:
            acc = np.zeros((2, 2), complex)
            for ax, sg in itertools.product(range(3), (1, -1)):
                q = np.zeros(3)
                q[ax] = sg * dl
                if true:
                    Pq = b.P(Qcache[(dl, ax, sg)], kv + q)
                    _, c = b.orbitals(kv + q)
                    dmq = 2 * c[:, :1] @ c[:, :1].conj().T
                else:
                    Pq = b.P(Qcache[(dl, ax, sg)], kv)
                    dmq = dm0
                acc += (Pq @ dmq @ Pq.conj().T - N0) / dl**2
            vals.append(acc / 6)
        v = (4 * vals[1] - vals[0]) / 3 if len(vals) == 2 else vals[0]
        out.append(pref * Ck.conj().T @ v @ Ck)
    return np.array(out)


def main():
    a = float(sys.argv[1]) if len(sys.argv) > 1 else 6.0
    orient = sys.argv[2] if len(sys.argv) > 2 else "z"
    ms = [int(x) for x in sys.argv[3:]] or [3, 4]
    suf = "" if orient == "z" else f"_{orient}"
    cell = Cell(np.eye(3) * a, h2_atoms(orient), "sto-3g")
    for m in ms:
        z, _ = load(a, m, suf)
        Nk, n3 = m**3, m**3
        ana = fock_dK(cell, z, Nk)
        rows = {}
        for conv in ("ewald", "none"):
            b = Bands(cell, z, m, conv)
            # interpolant reproduces the mesh: orbitals at mesh k == saved C (projector)
            e0, c0 = b.orbitals(b.kpts[1])
            pr = abs(c0[:, :1] @ c0[:, :1].conj().T - z["C"][1][:, :1] @ z["C"][1][:, :1].conj().T).max()
            if conv == "ewald":
                rows["fixed(fd)"] = dK_mo(b, z, true=False)
            rows[f"true[{conv}]"] = dK_mo(b, z, true=True)
            print(f"  [{conv}] interpolant == mesh projector at k1: {pr:.1e}")
        print(f"== a={a:g} {orient} n={m}  (x n^3; eps shift = +dK_pp/2 is mesh - true)")
        print(f"   anchor fixed(fd) - analytic A/B: max {abs(rows['fixed(fd)'] - ana).max() * n3:.2e}")
        for name, d in [("fixed(A/B)", ana)] + list(rows.items()):
            print(f"   {name:12s} k-avg dK_ii/2 {d[:, 0, 0].real.mean() / 2 * n3:+.5f} dK_aa/2 {d[:, 1, 1].real.mean() / 2 * n3:+.5f} "
                  f"gap {(d[:, 1, 1] - d[:, 0, 0]).real.mean() / 2 * n3:+.5f} | |dK_ia| max {abs(d[:, 0, 1]).max() * n3:.4f} "
                  f"k-avg {abs(d[:, 0, 1]).mean() * n3:.4f} | Gamma occ {d[0, 0, 0].real / 2 * n3:+.5f}")
        np.savez(f"/home/matt/qc/ferric-pbc/prototypes/pbc/kmp2_head/dK_a{a:g}_n{m}{suf}.npz", ana=ana,
                 fixed_fd=rows["fixed(fd)"], true_ewald=rows["true[ewald]"], true_none=rows["true[none]"])


if __name__ == "__main__":
    main()
