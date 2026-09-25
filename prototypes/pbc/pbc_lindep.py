"""Linear-dependence (lindep) handling for periodic calculations with diffuse basis sets.

Stage 2 item "lindep filtering of diffuse functions at Cell build" (FINDINGS.md adversarial review, 5b).
Molecular primitives only (PySCF molecular intor as libint2 stand-in); PySCF pbc is used ONLY as an oracle.

Objects (Bohr, chi_mk = sum_L e^{ik.L} phi_m(r - L), pbc_kpts convention):

    S(k)_mn = sum_L e^{ik.L} <phi_m | phi_n(. - L)>       (Hermitian PSD; S_sc eigenvalues = U_k eig S(k))

Why S(k) goes singular -- two mechanisms (Iteration 15, measured):
  1. ZONE-BOUNDARY SCALE: for one unit-normalised s Gaussian exp(-alpha r^2) per cell, Poisson summation gives
         S(k) = (1/Omega) (2 pi/alpha)^{3/2} sum_G exp(-|k+G|^2 / (2 alpha))      (single_s_poisson, the anchor)
     i.e. the Bloch sum is the Gaussian's FT sampled on k+G: exponentially small at the zone boundary.  This is a
     SCALE problem of one Bloch function (diagonal scaling removes it), and the function still carries physics.
  2. GAMMA DEPENDENCE: near Gamma the Bloch sums of all very diffuse functions tend to the same slowly varying
     function, so different AOs become genuinely dependent there (H2/aug a <= 5: lam_min(S(Gamma)) ~ 1e-12).
  Either way S(k) is computed by cancellation of O(1) real-space terms, so its error is ABSOLUTE,
  ~ eps * sum_L |S_L| (measured 1e-15..1e-12): an eigenvalue below ~1e-11 is noise, whatever its sign.
  An absolute cut on the unnormalised S(k) (ferric kscf.rs, PySCF 2.x default scf.hf.check_linear_dependency 1e-6)
  is therefore a precision cut; a cut on the diagonal-normalised S(k) (the deprecated PySCF addon canonical_orth_)
  divides noise by a noise-level diagonal (measured normalised eigenvalues of -4 for LiH/aug) and is not k-mesh ==
  supercell consistent.  Both are provided for measurement.

Orthogonalisers (each returns X (nao, nkept) with X^H S X = 1 on the kept space):
    canonical(S, thr, normalize)     eigenvectors of S (or D^-1/2 S D^-1/2) with eig >= thr
    pivoted_cholesky(S, tol, thr)    Lehtola 2019: pivoted Cholesky on the normalised S picks a sub-basis of
                                     AO Bloch functions, then canonical on it (PySCF partial_cholesky_orth_)
    discard_basis(basis, emin)       exp_to_discard: drop primitives with exponent < emin before the cell exists
                                     (PySCF pbc Cell.exp_to_discard semantics; empty shells removed)
"""

from __future__ import annotations

import itertools

import numpy as np
from pyscf import gto

# Test-only mutation switch.  None in production.
#   'drop_largest'  : canonical keeps the SMALLEST eigenvalues' complement wrongly (drops the largest instead)
#   'no_normalize'  : pivoted_cholesky pivots on the unnormalised S (diagonal-scale-dominated pivot order)
_MUTANT = None


# ------------------------------------------------------------------------------------------------ overlap
def lattice_points(a, rcut):
    a = np.asarray(a, float)
    spacing = 1.0 / np.linalg.norm(np.linalg.inv(a), axis=0)
    nmax = np.ceil(rcut / spacing).astype(int) + 1
    n = np.array(list(itertools.product(*(range(-k, k + 1) for k in nmax))))
    L = n @ a
    L = L[np.linalg.norm(L, axis=1) <= rcut]
    return L[np.argsort(np.linalg.norm(L, axis=1), kind="stable")]


def lattice_overlap(a, atoms, basis, cart=False, prec=1e-16, chunk=400):
    """S_L[m, iL, n] = <phi_m(r) | phi_n(r - L)> for all L out to the range where the MOST diffuse pair
    exp(-amin |L|^2 / 2) (times the cell diameter slack) drops below prec.  Chunked molecular intor_cross."""
    mol = gto.M(atom=atoms, basis=basis, unit="B", cart=cart, verbose=0, spin=_parity(atoms))
    amin = min(mol.bas_exp(i).min() for i in range(mol.nbas))
    R = mol.atom_coords()
    diam = np.ptp(R, axis=0).max() if len(R) > 1 else 0.0
    rcut = np.sqrt(2 * np.log(1 / prec) / amin) + diam + 1.0
    Ls = lattice_points(a, rcut)
    nao = mol.nao
    S_L = np.zeros((nao, len(Ls), nao))
    for c0 in range(0, len(Ls), chunk):
        Lc = Ls[c0 : c0 + chunk]
        img = gto.M(atom=[(s, np.asarray(r, float) + L) for L in Lc for (s, r) in atoms], basis=basis, unit="B",
                    cart=cart, verbose=0, spin=_parity(atoms) * len(Lc) % 2)
        blk = gto.intor_cross("int1e_ovlp", mol, img).reshape(nao, len(Lc), nao)
        S_L[:, c0 : c0 + len(Lc)] = blk
    return mol, Ls, S_L


def _parity(atoms):
    return int(sum(gto.charge(s) for s, _ in atoms)) % 2


def s_of_k(Ls, S_L, kpts):
    ph = np.exp(1j * np.asarray(kpts) @ Ls.T)
    S = np.einsum("kL,mLn->kmn", ph, S_L)
    return 0.5 * (S + S.conj().transpose(0, 2, 1))


def gamma_mesh(a, n):
    b = 2 * np.pi * np.linalg.inv(np.asarray(a, float)).T
    ints = np.array(list(itertools.product(*(range(k) for k in n))))
    return (ints / np.array(n)) @ b


def single_s_poisson(alpha, a, kpts, gmax=6):
    """Closed form S(k) for one unit-normalised s Gaussian exp(-alpha r^2) per cell:
    <phi|phi(.-L)> = exp(-alpha L^2 / 2)  =>  S(k) = (1/Omega) (2 pi/alpha)^{3/2} sum_G exp(-|k+G|^2 / (2 alpha))."""
    a = np.asarray(a, float)
    b = 2 * np.pi * np.linalg.inv(a).T
    vol = abs(np.linalg.det(a))
    n = np.array(list(itertools.product(*(range(-gmax, gmax + 1),) * 3))) @ b
    kg = np.asarray(kpts)[:, None, :] + n[None]
    return (2 * np.pi / alpha) ** 1.5 / vol * np.exp(-np.einsum("kgi,kgi->kg", kg, kg) / (2 * alpha)).sum(1)


def spectrum_report(S):
    """min eig of S(k) and of the diagonal-normalised S(k), per k."""
    out = []
    for Sk in S:
        dg = np.diag(Sk).real
        w = np.linalg.eigvalsh(Sk)
        if dg.min() > 0:
            d = np.sqrt(dg)
            wn = np.linalg.eigvalsh(Sk / d[:, None] / d[None, :])
        else:  # a diagonal lost to lattice-sum cancellation: the normalised metric is undefined
            wn = np.full_like(w, np.nan)
        out.append((w[0], wn[0], dg.min(), (w < 1e-6).sum(), (w < 1e-8).sum(), (wn < 1e-6).sum()))
    return np.array(out)  # lam_min(S), lam_min(normalised S), min diag S, #eig<1e-6, #eig<1e-8, #norm-eig<1e-6


# ------------------------------------------------------------------------------------------ orthogonalisers
def canonical(S, thr=1e-6, normalize=False):
    if normalize:
        d = 1.0 / np.sqrt(np.diag(S).real)
        s, U = np.linalg.eigh(S * d[:, None] * d[None, :])
    else:
        d = None
        s, U = np.linalg.eigh(S)
    keep = s >= thr
    if _MUTANT == "drop_largest":
        keep = np.arange(len(s)) < int(keep.sum())
    X = U[:, keep] / np.sqrt(s[keep])
    return X if d is None else d[:, None] * X


def pivoted_cholesky_pivots(A, tol):
    """Pivots of a (complex Hermitian) pivoted Cholesky of A, stopping when the largest remaining diagonal < tol."""
    A = np.array(A, dtype=complex)
    n = len(A)
    d = np.diag(A).real.copy()
    L = np.zeros((n, n), complex)
    piv = []
    for j in range(n):
        p = int(np.argmax(np.where(np.isin(np.arange(n), piv), -np.inf, d)))
        if d[p] < tol:
            break
        piv.append(p)
        col = (A[:, p] - L[:, :j] @ L[p, :j].conj()) / np.sqrt(d[p])
        L[:, j] = col
        d = d - abs(col) ** 2
    return piv


def pivoted_cholesky(S, tol=1e-6, thr=1e-9):
    """Lehtola (JCP 151, 241102): pivot on the diagonal-normalised S, canonical-orthogonalise the sub-basis."""
    dn = 1.0 / np.sqrt(np.diag(S).real)
    Sn = S if _MUTANT == "no_normalize" else S * dn[:, None] * dn[None, :]
    piv = sorted(pivoted_cholesky_pivots(Sn, tol))
    Xs = canonical(S[np.ix_(piv, piv)], thr, normalize=True)
    X = np.zeros((len(S), Xs.shape[1]), complex)
    X[piv] = Xs
    return X


def orth_factory(kind, **kw):
    if kind == "canonical":
        return lambda S: canonical(S, kw.get("thr", 1e-6), kw.get("normalize", False))
    if kind == "cholesky":
        return lambda S: pivoted_cholesky(S, kw.get("tol", 1e-6), kw.get("thr", 1e-9))
    if kind == "lowdin":  # symmetric S^{-1/2}, nothing dropped (reference when S is well enough conditioned)
        def low(S):
            s, U = np.linalg.eigh(S)
            return (U / np.sqrt(s)) @ U.conj().T
        return low
    raise ValueError(kind)


# --------------------------------------------------------------------------------------- exp_to_discard
def discard_basis(basis, symbols, emin):
    """PySCF pbc Cell.exp_to_discard on a named basis: drop primitives with exponent < emin from every
    shell (contraction coefficients of the survivors kept as-is; PySCF renormalises at mol build), remove
    shells left empty.  Returns (basis dict, {symbol: n_primitives_dropped})."""
    out, dropped = {}, {}
    for el in sorted(set(symbols)):
        shells = gto.basis.load(basis, el) if isinstance(basis, str) else basis[el]
        new, nd = [], 0
        for sh in shells:
            l, prims = sh[0], [p for p in sh[1:] if p[0] >= emin]
            nd += len(sh) - 1 - len(prims)
            if prims:
                # a column left all-zero by the cut would be a null function: drop that column
                ncol = len(prims[0]) - 1
                cols = [c for c in range(ncol) if any(abs(p[1 + c]) > 0 for p in prims)]
                if cols:
                    new.append([l] + [[p[0]] + [p[1 + c] for c in cols] for p in prims])
        out[el] = new
        dropped[el] = nd
    return out, dropped


# ------------------------------------------------------------------------------------------ SCF plumbing
def gamma_kb(g):
    """A pbc_gamma / pbc_kpts.gamma_aft Gamma build as an Nk = 1 k-point bundle, so the SAME krhf (and the same
    orthogonaliser hook) runs the supercell.  rhf's K_mn = sum I[m,l,s,n] D_ls  <=>  Kker[m,l,n,s] = I[m,l,s,n]."""
    I = g["I"]
    return dict(S=g["S"][None].astype(complex), h=g["h"][None].astype(complex), Jker=I[None, None].astype(complex),
                Kker=I.transpose(0, 1, 3, 2)[None, None].astype(complex), enn=g["enn"], madelung=0.0, Nk=1)


def kept_counts(S, orth):
    return [orth(Sk).shape[1] for Sk in S]
