"""Gamma-point range-separated Gaussian density fitting (RS-GDF), built from MOLECULAR
integral primitives (PySCF molecular `intor` standing in for libint2) plus the analytic
pair FT of pbc_gamma.py.  PySCF pbc is NOT used here (oracle only, in the tests).

Target: B[P,m,n] with  I[m,n,l,s] ~= sum_P B[P,m,n] B[P,l,s], where I is the Gamma ERI of
pbc_gamma.py in its G=0 convention,

    I_mnls = (1/Omega) sum_{G!=0} 4pi/G^2  conj(P_mn(G)) P_ls(G),   P_mn(G) = FT of rho_mn.

Everything below is a quantity in the SAME G=0-dropped Coulomb kernel v'(G) = 4pi/G^2 (G!=0):

    J2[P,Q]   = (P|Q)'    = (1/Omega) sum_{G!=0} v(G) conj(X_P(G)) X_Q(G)      (aux metric)
    J3[mn,P]  = (mn|P)'   = (1/Omega) sum_{G!=0} v(G) conj(P_mn(G)) X_P(G)
    B         = s^{-1/2} U^T J3^T   with J2 = U s U^T, s > lindep            (Dunlap-robust)

Because the fitting metric IS the target kernel, the fit is variational in the kernel that
the energy uses; the density's G=0 component (charge S_mn) is simply invisible to both.
There is no charge constraint and no compensating charge: charged aux functions are fine
because the only divergent term (G=0) is removed identically from J2 and J3.  This is the
PySCF RSGDF convention (rsdf_builder.py get_2c2e g0_fac / gen_j3c_loader vbar).

Evaluation of each primed quantity by the Ewald split 1/r = erfc(wr)/r + erf(wr)/r:
  SR : real-space lattice sums of ordinary molecular erfc integrals over image shells
       J2: sum_T (P_0|Q_T)_erfc        J3: sum_{L,T} (m_0 n_L | P_T)_erfc
       These sums implicitly contain the erfc G=0 term  pi/(w^2 Omega) q_P q_Q  (resp. S_mn q_P),
       q_P = int chi_P  (nonzero for s AND for Cartesian d/f with an r^2 component).
  LR : (1/Omega) sum_{G!=0} 4pi/G^2 exp(-G^2/4w^2) conj(A(G)) X_P(G); X_P is the one-centre
       Gaussian FT (hermite_E with lb=0).
  G=0: subtract pi/(w^2 Omega) q_P q_Q from J2 and pi/(w^2 Omega) S_mn q_P from J3.
Result is w-independent (tested).  Units Bohr/Hartree; orbital basis Cartesian (as pbc_gamma),
aux basis spherical by default (as ferric), Cartesian optional.
"""

from __future__ import annotations

import json
import os

import numpy as np
from pyscf import gto
from scipy.special import erfc

from pbc_gamma import Cell, cart_comps, hermite_E, pair_ft

BUNDLED = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "crates",
    "ferric-core",
    "src",
    "basis",
    "bundled",
)
ELEMENTS = {"H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9}


# --------------------------------------------------------------------------- aux bases
def ferric_basis(name, symbols):
    """Read one of ferric's bundled BSE-JSON basis files into a PySCF basis dict."""
    d = json.load(open(os.path.join(BUNDLED, name + ".json")))
    out = {}
    for s in set(symbols):
        shells = []
        for sh in d["elements"][str(ELEMENTS[s])]["electron_shells"]:
            exps = [float(x) for x in sh["exponents"]]
            for l, coefs in zip(
                sh["angular_momentum"] * len(sh["coefficients"])
                if len(sh["angular_momentum"]) == 1
                else sh["angular_momentum"],
                sh["coefficients"],
            ):
                shells.append([l] + [[e, float(c)] for e, c in zip(exps, coefs)])
        out[s] = shells
    return out


def even_tempered(symbols, lmax, amin, beta, n):
    """n uncontracted exponents amin*beta^k for every l <= lmax (same for every element)."""
    exps = amin * beta ** np.arange(n)
    return {
        s: [[l, [e, 1.0]] for l in range(lmax + 1) for e in exps] for s in set(symbols)
    }


# ---------------------------------------------------------------------- one-centre FT
def _raw_ft(mol, G):
    """Unnormalised Cartesian FT X[P,g] of every (cart) AO of `mol` (one-centre Gaussians):
    FT[x^i y^j z^k e^{-a r^2}](G) = (pi/a)^1.5 e^{-G^2/4a - iG.A} prod_d sum_t E^{i0}_t (-iG_d)^t."""
    G = np.asarray(G, float).reshape(-1, 3)
    G2 = np.einsum("gi,gi->g", G, G)
    X = np.zeros((mol.nao, len(G)), complex)
    raw_ovlp = np.zeros(mol.nao)
    off = 0
    for ib in range(mol.nbas):
        l, A = mol.bas_angular(ib), mol.atom_coord(mol.bas_atom(ib))
        exps, coef = mol.bas_exp(ib), mol._libcint_ctr_coeff(ib)
        comps = cart_comps(l)
        powg = [
            np.stack([(-1j * G[:, d]) ** t for t in range(l + 1)]) for d in range(3)
        ]
        phase = np.exp(-1j * (G @ A))
        for ic in range(mol.bas_nctr(ib)):
            c = coef[:, ic]
            for u, lxyz in enumerate(comps):
                acc = np.zeros(len(G), complex)
                so = 0.0
                for a, ca in zip(exps, c):
                    f = ca * (np.pi / a) ** 1.5 * np.exp(-G2 / (4 * a))
                    for d in range(3):
                        E = hermite_E(lxyz[d], 0, a, 0.0, 0.0)[lxyz[d], 0]
                        f = f * (E[: l + 1] @ powg[d][: len(E[: l + 1])])
                    acc += f
                    for b, cb in zip(exps, c):  # raw self-overlap, for calibration
                        t = ca * cb * (np.pi / (a + b)) ** 1.5
                        for d in range(3):
                            t *= hermite_E(lxyz[d], lxyz[d], a, b, 0.0)[
                                lxyz[d], lxyz[d], 0
                            ]
                        so += t
                X[off + u] = acc * phase
                raw_ovlp[off + u] = so
            off += len(comps)
    return X, raw_ovlp


def aux_ft(auxmol, G, c2s=None):
    """Normalised FT of the aux functions, (naux, ng); calibrated so that the implied
    self-overlap equals PySCF's int1e_ovlp_cart diagonal (same trick as pair_ft/S)."""
    X, so = _raw_ft(auxmol, G)
    nrm = np.sqrt(np.diag(auxmol.intor("int1e_ovlp_cart")) / so)
    X = X * nrm[:, None]
    return X if c2s is None else c2s.T @ X


# ------------------------------------------------------------------- range estimates
def _rcut_erfc(theta, prec):
    """R with erfc(sqrt(theta) R)/R = prec (bisection)."""
    lo, hi = 1e-3, 500.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if erfc(np.sqrt(theta) * mid) / mid > prec else (lo, mid)
    return hi


def _subtract_g0(J2, J3, S, q, c0):
    """The erfc real-space sums contain the G=0 term c0 = pi/(w^2 Omega); remove it from BOTH
    the metric and the 3-index tensor.  (Single function so tests can mutate it.)"""
    return J2 - c0 * np.outer(q, q), J3 - c0 * S.reshape(-1)[:, None] * q[None, :]


# -------------------------------------------------------------------------- builder
def build_gdf(
    cell,
    auxbasis,
    w=1.0,
    prec=1e-13,
    rcut_pair=None,
    rcut_aux3=None,
    rcut_aux2=None,
    gcut=None,
    spherical=True,
    lindep=1e-10,
    auxmol=None,
    verbose=False,
):
    """RS-GDF three-index tensor at Gamma.  Returns dict with B (naux_kept, nao, nao) and counts.

    auxbasis: PySCF basis spec for atom-centred aux (ignored when `auxmol` is given: an explicit
    cartesian PySCF Mole of aux functions placed in cell 0, e.g. on ghost centres)."""
    mol = cell.mol
    nao, nb0 = mol.nao, mol.nbas
    if auxmol is None:
        nel = sum(
            gto.charge(s) for s, _ in cell.atoms
        )  # spin = parity only to satisfy Mole (nothing reads it)
        auxmol = gto.M(
            atom=cell.atoms,
            basis=auxbasis,
            unit="B",
            cart=True,
            verbose=0,
            spin=nel % 2,
        )
    c2s = None
    if spherical:
        sph = auxmol.copy()
        sph.cart = False
        c2s = sph.cart2sph_coeff()
    naux_c = auxmol.nao
    q_c = aux_ft(auxmol, np.zeros((1, 3)))[:, 0].real  # charges of cart aux
    amin_aux = min(auxmol.bas_exp(i).min() for i in range(auxmol.nbas))
    amin_orb = min(mol.bas_exp(i).min() for i in range(nb0))
    c0 = np.pi / (w * w * cell.vol)

    # ---- ranges (all conservative; tests check convergence w.r.t. them)
    rcut_pair = rcut_pair or np.sqrt(2 * np.log(1 / prec) / amin_orb) + 2.0
    # the SR tail of (P|Q) is ~ q_P q_Q erfc(sqrt(theta) R)/R: normalised diffuse aux carry
    # |q| >> 1 (|q|=22 at a=0.1), so the per-integral target is prec/qmax (3c) and prec/qmax^2 (2c)
    qmax = max(1.0, abs(q_c).max())
    th3 = 1.0 / (1.0 / amin_aux + 1.0 / (2 * amin_orb) + 1.0 / w**2)
    rcut_aux3 = rcut_aux3 or _rcut_erfc(th3, prec / qmax) + 2.0
    th2 = 1.0 / (2.0 / amin_aux + 1.0 / w**2)
    rcut_aux2 = rcut_aux2 or _rcut_erfc(th2, prec / qmax**2) + 2.0
    gcut = gcut or 2 * w * np.sqrt(np.log(1 / prec))

    # ---- lattice overlap (G=0 of the pair density)
    L1 = cell.translations(rcut_pair)
    sm = cell.supermol(L1)
    S = (
        sm.intor("int1e_ovlp_cart", shls_slice=(0, nb0, 0, sm.nbas))
        .reshape(nao, len(L1), nao)
        .sum(1)
    )

    # ---- SR metric: sum_T (P_0 | Q_T)_erfc
    T2 = cell.translations(rcut_aux2)
    aux_img2 = _images(auxmol, T2)
    with aux_img2.with_range_coulomb(-w):
        J2 = aux_img2.intor(
            "int2c2e_cart", shls_slice=(0, auxmol.nbas, 0, aux_img2.nbas)
        )
    J2 = J2.reshape(naux_c, len(T2), naux_c).sum(1)

    # ---- SR 3c: sum_{L,T} (m_0 n_L | P_T)_erfc, chunked over aux images
    T3 = cell.translations(rcut_aux3)
    J3 = np.zeros((nao * nao, naux_c))
    chunk = max(1, int(2e7 // (nao * nao * len(L1) * naux_c)))
    for t0 in range(0, len(T3), chunk):
        ai = _images(auxmol, T3[t0 : t0 + chunk])
        big = gto.conc_mol(sm, ai)
        with big.with_range_coulomb(-w):
            v = big.intor(
                "int3c2e_cart", shls_slice=(0, nb0, 0, sm.nbas, sm.nbas, big.nbas)
            )
        nt = len(T3[t0 : t0 + chunk])
        J3 += (
            v.reshape(nao, len(L1), nao, nt, naux_c)
            .sum(axis=(1, 3))
            .reshape(nao * nao, naux_c)
        )
    asym3 = abs(
        J3.reshape(nao, nao, -1) - J3.reshape(nao, nao, -1).transpose(1, 0, 2)
    ).max()

    # ---- LR in reciprocal space, G != 0
    G = cell.gvectors(gcut)
    G = G[np.einsum("gi,gi->g", G, G) > 1e-12]
    G2 = np.einsum("gi,gi->g", G, G)
    vlr = 4 * np.pi / G2 * np.exp(-G2 / (4 * w * w)) / cell.vol
    X = aux_ft(auxmol, G)
    P0 = pair_ft(cell, np.zeros((1, 3)))[..., 0].real
    pn = np.sqrt(np.diag(S) / np.diag(P0))
    P = (pair_ft(cell, G) * pn[:, None, None] * pn[None, :, None]).reshape(
        nao * nao, -1
    )
    J2 = J2 + ((X.conj() * vlr) @ X.T).real
    J3 = J3 + ((P.conj() * vlr) @ X.T).real

    # ---- G=0 bookkeeping, then aux -> spherical, symmetrise
    J2, J3 = _subtract_g0(J2, J3, S, q_c, c0)
    if c2s is not None:
        J2, J3 = c2s.T @ J2 @ c2s, J3 @ c2s
    asym2 = abs(J2 - J2.T).max()
    J2 = 0.5 * (J2 + J2.T)
    J3 = J3.reshape(nao, nao, -1)
    J3 = 0.5 * (J3 + J3.transpose(1, 0, 2))

    # ---- metric: eigen-decomposition with lindep filter (PySCF LINEAR_DEP_THR = 1e-10)
    s, U = np.linalg.eigh(J2)
    keep = s > lindep
    B = np.einsum("Pk,mnP->kmn", U[:, keep] / np.sqrt(s[keep]), J3)
    info = dict(
        naux=J2.shape[0],
        naux_kept=int(keep.sum()),
        metric_min=s.min(),
        metric_max=s.max(),
        n_pair_images=len(L1),
        n_aux_images_3c=len(T3),
        n_aux_images_2c=len(T2),
        nG=len(G),
        gcut=gcut,
        rcut_pair=rcut_pair,
        rcut_aux3=rcut_aux3,
        rcut_aux2=rcut_aux2,
        asym_J2=asym2,
        asym_J3=asym3,
        B_size=B.size,
        n_3c_shell_triplets=nb0 * sm.nbas * auxmol.nbas * len(T3),
    )
    if verbose:
        print(
            "  gdf:",
            {k: (f"{v:.3g}" if isinstance(v, float) else v) for k, v in info.items()},
        )
    return dict(
        B=B, J2=J2, J3=J3, S=S, q=q_c if c2s is None else c2s.T @ q_c, info=info
    )


def _images(auxmol, Ts):
    """Molecule holding copies of every aux shell translated by each T (block order: T-major)."""
    atoms = [
        (auxmol.atom_symbol(i), auxmol.atom_coord(i) + T)
        for T in Ts
        for i in range(auxmol.natm)
    ]
    nelec = sum(gto.charge(a) for a, _ in atoms)  # only to satisfy Mole's spin check
    return gto.M(
        atom=atoms, basis=auxmol._basis, unit="B", cart=True, verbose=0, spin=nelec % 2
    )


# ------------------------------------------------------------------------------ JK
def jk_from_B(B):
    """J, K from B tensors (what ferric's molecular DF code consumes)."""

    def jk(D):
        J = np.einsum("Pmn,P->mn", B, np.einsum("Pls,ls->P", B, D))
        BD = np.einsum("Pml,ls->Pms", B, D)
        K = np.einsum("Pms,Psn->mn", BD, B)
        return J, K

    return jk


def eri_from_B(B):
    return np.einsum("Pmn,Pls->mnls", B, B)


__all__ = [
    "Cell",
    "build_gdf",
    "ferric_basis",
    "even_tempered",
    "aux_ft",
    "jk_from_B",
    "eri_from_B",
]
