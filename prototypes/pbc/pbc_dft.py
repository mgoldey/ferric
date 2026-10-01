"""Gamma-point periodic KS-DFT prototype (Iteration 8): all-electron XC on a PERIODIC BECKE grid.

Only molecular primitives are used (PySCF molecular `eval_gto` stands in for ferric's
`ao_grid::eval_basis_on_points`, pyscf.dft.libxc for ferric's libxc wrapper); PySCF pbc is the
oracle only (run_dft_*.py, tests).  J and exact exchange come from pbc_gamma (dense pure-AFT or
Ewald-split I) or pbc_gdf (RS-GDF B) through a jk(D) -> (J, K) callable, exactly as in the HF SCF.

Grid construction ("A2", recommended for ferric):
  * atom-centred Treutler-Ahlrichs M4 radial x Lebedev grids for the atoms OF THE CELL only
    (identical to ferric's molecular `build_atomic_grid`: same radial formula + xi table);
  * Becke fuzzy weight of the home atom A evaluated in the INFINITE crystal: the cell functions
    P_B(r) = prod_C s(nu_BC(r)) run over IMAGE atoms B, C = atom + L.  Truncation: at point r only
    image atoms with |r - R_B| <= D take part (translation-covariant, so it is still an exact
    partition of unity at every r; D only changes the partition, not its sum);
  * AO values are lattice sums  chi^Gamma_m(r) = sum_L chi_m(r - L)  (image shells within the AO
    extent of the point);
  * an integral of a lattice-periodic f over ONE cell is  sum_{A in cell} sum_{g in grid(A)} w_g f(r_g).

Why that is exact:  let w_{A,L}(r) be the Becke weight of image A+L.  Covariance gives
w_{A,L}(r) = w_{A,0}(r - L), and sum_{A,L} w_{A,L}(r) = 1 at every r.  For periodic f,
  int_cell f = int_cell sum_{A,L} w_{A,L} f = sum_A sum_L int_{cell-L} w_{A,0} f = sum_A int_{R^3} w_{A,0} f,
i.e. the cells' atoms' FULL atomic grids (no cut at the cell boundary) with crystal Becke weights.
PySCF's pbc BeckeGrids ("A1") instead places grids on all image atoms and keeps only points inside
one parallelepiped (half weight on the faces) -- the same partition identity, but with a hard
domain cut through every atomic grid.  Both are measured below.

scheme='becke' is ferric's becke.rs (3x Becke polynomial, Becke/Bragg-Slater size adjustment);
scheme='ssf' is Stratmann-Scuseria-Frisch (compact support |mu|<0.64 -> finite exact image lists).
Units Bohr/Hartree, Cartesian AOs (as pbc_gamma)."""

from __future__ import annotations

import numpy as np
from pyscf.dft import libxc
from pyscf.dft.LebedevGrid import MakeAngularGrid

# ferric radial.rs TA_XI (== PySCF radi._treutler_ahlrichs_xi), index Z
TA_XI = [
    1.0,
    0.8,
    0.9,
    1.8,
    1.4,
    1.3,
    1.1,
    0.9,
    0.9,
    0.9,
    0.9,
    1.4,
    1.3,
    1.3,
    1.2,
    1.1,
    1.0,
    1.0,
    1.0,
]
# ferric becke.rs bragg_slater_bohr (Angstrom), Z = 1..18
BRAGG_A = [
    1.0,
    0.35,
    0.30,
    1.45,
    1.05,
    0.85,
    0.70,
    0.65,
    0.60,
    0.50,
    0.45,
    1.80,
    1.50,
    1.25,
    1.10,
    1.00,
    1.00,
    1.00,
    0.71,
]
ANG2BOHR = 1.8897259886


# ------------------------------------------------------------------------ atomic grids
def ta_m4(z, n):
    """ferric radial.rs treutler_ahlrichs_m4: (r, w) with 4 pi r^2 included, r ascending."""
    xi = TA_XI[z] if z < len(TA_XI) else 1.5
    k = np.arange(1, n + 1)
    th = np.pi * k / (n + 1)
    x = np.cos(th)
    ln2 = xi / np.log(2.0)
    lt = np.log((1 - x) / 2)
    r = -ln2 * (1 + x) ** 0.6 * lt
    dr = ln2 * (1 + x) ** 0.6 * (-0.6 / (1 + x) * lt + 1 / (1 - x))
    w = np.pi / (n + 1) * np.sin(th) * dr * 4 * np.pi * r * r
    return r[::-1], w[::-1]


def atomic_grid(z, n_rad, n_ang):
    """Unpartitioned atom-centred grid: offsets (n,3), weights (n,), radius (n,)."""
    r, wr = ta_m4(z, n_rad)
    ang = MakeAngularGrid(n_ang)  # weights sum to 1
    xyz = (r[:, None, None] * ang[None, :, :3]).reshape(-1, 3)
    w = (wr[:, None] * ang[None, :, 3]).ravel()
    return xyz, w, np.repeat(r, len(ang))


# ---------------------------------------------------------------------- lattice helpers
def lattice_points(a, rcut):
    spacing = 1.0 / np.linalg.norm(np.linalg.inv(a), axis=0)
    nmax = np.ceil(rcut / spacing).astype(int) + 1
    n = np.stack(
        np.meshgrid(*[np.arange(-k, k + 1) for k in nmax], indexing="ij"), -1
    ).reshape(-1, 3)
    L = n @ a
    L = L[np.linalg.norm(L, axis=1) <= rcut]
    return L[np.argsort(np.linalg.norm(L, axis=1), kind="stable")]


def image_atoms(a, R, Z, center, rcut):
    """All image atoms R_i + L within rcut of `center` (returns xyz, Z, index-in-cell)."""
    diam = np.ptp(R, axis=0).max() if len(R) > 1 else 0.0
    Ls = lattice_points(
        a, rcut + np.linalg.norm(R - center, axis=1).max() + diam + 1e-9
    )
    xyz = (R[None, :, :] + Ls[:, None, :]).reshape(-1, 3)
    zz = np.tile(Z, len(Ls))
    idx = np.tile(np.arange(len(R)), len(Ls))
    keep = np.linalg.norm(xyz - center, axis=1) <= rcut
    return xyz[keep], zz[keep], idx[keep]


# -------------------------------------------------------------------- Becke partition
def _f3(nu, k=3):
    for _ in range(k):
        nu = 0.5 * nu * (3 - nu * nu)
    return nu


def _ssf(nu, a=0.64):
    m = nu / a
    m2 = m * m
    g = m * (35 + m2 * (-35 + m2 * (21 - 5 * m2))) / 16
    return np.where(nu <= -a, -1.0, np.where(nu >= a, 1.0, g))


def _smooth(nu, scheme):
    """'becke' = ferric becke.rs (3 iterations); 'becke1'/'becke2' = fewer (smoother) iterations;
    'ssf' = Stratmann-Scuseria-Frisch (compact support)."""
    if scheme == "ssf":
        return _ssf(nu)
    k = {"becke": 3, "becke3": 3, "becke2": 2, "becke1": 1, "becke4": 4}[scheme]
    return _f3(nu, k)


def size_adjust(zs, adjust=True):
    """Becke size-adjustment a_BC (ferric becke.rs == PySCF becke_atomic_radii_adjust)."""
    if not adjust:
        return np.zeros((len(zs), len(zs)))
    rad = np.array([BRAGG_A[z] if z < len(BRAGG_A) else 1.0 for z in zs]) * ANG2BOHR
    chi = rad[:, None] / rad[None, :]
    return np.clip(0.25 * (1 / chi - chi), -0.5, 0.5)


def partition_weight(pts, home, nb_xyz, nb_z, D, scheme="becke", adjust=True):
    """Becke weight of image atom `home` (index into nb_xyz) at points pts, in the crystal.
    nb_* must contain every image atom within D of every point.  Returns (w, n_nb per point)."""
    d = np.linalg.norm(pts[:, None, :] - nb_xyz[None, :, :], axis=2)  # (p, B)
    mask = d <= D
    if (
        scheme == "exp"
    ):  # smooth Hirshfeld-like partition with a free-H-atom-like 1s density e^{-2r}
        P = np.exp(-2.0 * d) * mask
        tot = P.sum(axis=1)
        return np.where(tot > 0, P[:, home] / np.where(tot > 0, tot, 1), 0.0), mask.sum(
            axis=1
        )
    Rbc = np.linalg.norm(nb_xyz[:, None] - nb_xyz[None], axis=2)
    np.fill_diagonal(Rbc, 1.0)
    acorr = size_adjust(nb_z, adjust)
    mu = (d[:, :, None] - d[:, None, :]) / Rbc[None]
    nu = mu + acorr[None] * (1 - mu * mu)
    s = 0.5 * (1 - _smooth(nu, scheme))
    nb = len(nb_xyz)
    s[:, np.arange(nb), np.arange(nb)] = 1.0
    s = np.where(mask[:, None, :], s, 1.0)  # C outside D: no factor
    P = s.prod(axis=2) * mask  # B outside D: P_B = 0
    tot = P.sum(axis=1)
    w = np.where(tot > 0, P[:, home] / np.where(tot > 0, tot, 1), 0.0)
    return w, mask.sum(axis=1)


class PeriodicGrid:
    """Periodic Becke grid (construction A2) with lattice-summed AO values cached.

    ao: (npts, nao) or (4, npts, nao) for deriv=1 (value, d/dx, d/dy, d/dz)."""

    def __init__(
        self,
        cell,
        n_rad=75,
        n_ang=110,
        D=10.0,
        scheme="becke",
        adjust=True,
        deriv=1,
        ao_thresh=1e-15,
        wdrop=0.0,
        chunk=64,
        box=1.5,
    ):
        self.cell, self.D = cell, D
        a, R = cell.a, cell.R
        Z = cell.mol.atom_charges()
        coords, weights, nnb, homes = [], [], [], []
        for A in range(len(R)):
            off, w0, _ = atomic_grid(int(Z[A]), n_rad, n_ang)
            pts = R[A] + off
            # home-atom reach is D (points beyond D from A have P_A = 0 by the mask)
            keep = np.linalg.norm(off, axis=1) <= D
            pts, w0 = pts[keep], w0[keep]
            nb_xyz, nb_z, nb_idx = image_atoms(a, R, Z, R[A], 2 * D + 2 * box)
            home = int(
                np.nonzero(
                    (nb_idx == A) & (np.linalg.norm(nb_xyz - R[A], axis=1) < 1e-9)
                )[0][0]
            )
            # spatial bins of edge `box`: one candidate list per bin (atoms within D + half-diagonal)
            key = np.floor(pts / box).astype(np.int64)
            _, inv = np.unique(key, axis=0, return_inverse=True)
            inv = inv.ravel()
            for b in range(inv.max() + 1):
                ib = np.nonzero(inv == b)[0]
                p = pts[ib]
                cen = (np.floor(p[0] / box) + 0.5) * box
                near = np.linalg.norm(nb_xyz - cen, axis=1) <= D + 0.87 * box + 1e-9
                near[home] = True
                sel = np.nonzero(near)[0]
                ck = max(1, int(chunk * 4e4 / max(len(sel), 1) ** 2))
                for q0 in range(0, len(ib), ck):
                    w, nn = partition_weight(
                        p[q0 : q0 + ck],
                        int(np.nonzero(sel == home)[0][0]),
                        nb_xyz[sel],
                        nb_z[sel],
                        D,
                        scheme,
                        adjust,
                    )
                    coords.append(p[q0 : q0 + ck])
                    weights.append(w0[ib[q0 : q0 + ck]] * w)
                    nnb.append(nn)
                    homes.append(np.full(len(nn), A))
        self.coords = np.vstack(coords)
        self.weights = np.concatenate(weights)
        self.n_nb = np.concatenate(nnb)
        self.home = np.concatenate(homes)
        self.n_total = len(self.weights)
        live = np.abs(self.weights) > wdrop
        self.coords, self.weights, self.n_nb, self.home = (
            self.coords[live],
            self.weights[live],
            self.n_nb[live],
            self.home[live],
        )
        if deriv is not None:  # deriv=None: weights only (partition studies)
            self.ao, self.n_img = ao_gamma(cell, self.coords, deriv, ao_thresh)

    @property
    def size(self):
        return len(self.weights)


def ao_rcut(mol, thresh=1e-15):
    """Radius beyond which every primitive of every shell is below thresh (value, incl. r^l)."""
    rc = 0.0
    for ib in range(mol.nbas):
        l = mol.bas_angular(ib)
        for e, c in zip(mol.bas_exp(ib), np.abs(mol._libcint_ctr_coeff(ib)).max(1)):
            r = np.sqrt(np.log(max(c, 1.0) / thresh) / e) + 1.0
            for _ in range(10):
                r = np.sqrt(
                    max(np.log(max(c, 1.0) * max(r, 1.0) ** l / thresh), 1e-3) / e
                )
            rc = max(rc, r)
    return rc


def ao_gamma(cell, pts, deriv=0, thresh=1e-15, chunk=448, box=2.0):
    """chi^Gamma_m(r) = sum_L chi_m(r - L) (+ gradients) on arbitrary points: ONE molecular supermol
    of image shells covering every point's AO extent, evaluated in spatially sorted chunks with
    libcint's per-(56-point block, shell) distance screening (eval_gto cutoff=thresh).
    Returns (ao, n_live_images per chunk)."""
    mol = cell.mol
    nao = mol.nao
    rc = ao_rcut(mol, thresh)
    comp = 4 if deriv else 1
    out = np.zeros((comp, len(pts), nao))
    cen = cell.R.mean(0)
    reach = np.linalg.norm(pts - cen, axis=1).max() + rc
    diam = np.linalg.norm(cell.R - cen, axis=1).max()
    Ls = lattice_points(cell.a, reach + diam)
    Ls = Ls[np.linalg.norm(cell.R[None] + Ls[:, None] - cen, axis=2).min(1) <= reach]
    sm = cell.supermol(Ls)
    order = np.lexsort(np.floor(pts / box).T[::-1])
    nimg = []
    name = "GTOval_cart_deriv1" if deriv else "GTOval_cart"
    for p0 in range(0, len(pts), chunk):
        idx = order[p0 : p0 + chunk]
        p = pts[idx]
        v = np.asarray(sm.eval_gto(name, p, cutoff=thresh)).reshape(
            comp, len(p), len(Ls), nao
        )
        live = np.abs(v[0]).max(axis=(0, 2)) > 0
        nimg.append(int(live.sum()))
        out[:, idx] = v.sum(2)
    return (out if deriv else out[0]), np.array(nimg)


def uniform_grid(cell, n, deriv=1, thresh=1e-15):
    """Construction B (oracle): n^3 uniform points over the cell, weight Omega/n^3."""
    f = (
        np.stack(np.meshgrid(*[np.arange(n)] * 3, indexing="ij"), -1).reshape(-1, 3)
        + 0.0
    ) / n
    pts = f @ cell.a
    g = PeriodicGrid.__new__(PeriodicGrid)
    g.cell, g.coords = cell, pts
    g.weights = np.full(len(pts), cell.vol / len(pts))
    g.ao, g.n_img = ao_gamma(cell, pts, deriv, thresh)
    g.n_nb = np.zeros(len(pts), int)
    return g


def pyscf_a1_grid(cell, n_rad, n_ang, deriv=1):
    """Construction A1 (ORACLE ONLY): PySCF pbc BeckeGrids points/weights -- grids on all image atoms
    within PySCF's rcut, points kept inside the parallelepiped [-1/2, 1/2)^3 (half weight on faces),
    original Becke over those image atoms, NO size adjustment -- with our lattice-summed AOs."""
    from pyscf.dft import radi
    from pyscf.pbc import gto as pgto
    from pyscf.pbc.dft import gen_grid as pgg

    pc = pgto.Cell(
        a=cell.a,
        atom=cell.atoms,
        basis=cell.basis,
        unit="B",
        cart=True,
        verbose=0,
        spin=cell.mol.spin,
    )
    pc.precision = 1e-12
    pc.build()
    c, w = pgg.get_becke_grids(
        pc, atom_grid=(n_rad, n_ang), radi_method=radi.treutler, prune=None
    )
    g = PeriodicGrid.__new__(PeriodicGrid)
    g.cell, g.coords, g.weights = cell, c, w
    g.ao, g.n_img = ao_gamma(cell, c, deriv)
    g.n_nb = np.zeros(len(c), int)
    return g


# ----------------------------------------------------------------------------- numint
def _ao0(grid):
    return grid.ao[0] if grid.ao.ndim == 3 else grid.ao


def rho_on_grid(grid, D, gga):
    ao = grid.ao
    if not gga:
        a0 = _ao0(grid)
        return np.einsum("pi,pi->p", a0 @ D, a0)
    c = ao[0] @ D
    rho = np.empty((4, ao.shape[1]))
    rho[0] = np.einsum("pi,pi->p", c, ao[0])
    for x in range(3):
        rho[x + 1] = 2 * np.einsum("pi,pi->p", c, ao[x + 1])
    return rho


def xc_family(xc):
    return "GGA" if libxc.is_gga(xc) else ("LDA" if libxc.is_lda(xc) else "MGGA")


def eval_vxc(grid, D, xc):
    """(E_xc, V_xc) for the semilocal part of `xc` (for hybrids libxc returns only the DFT part)."""
    gga = xc_family(xc) == "GGA"
    if xc_family(xc) == "MGGA":
        raise NotImplementedError("meta-GGA not prototyped")
    rho = rho_on_grid(grid, D, gga)
    exc, vxc = libxc.eval_xc(xc, rho, spin=0, deriv=1)[:2]
    w = grid.weights
    r0 = rho[0] if gga else rho
    exc_tot = np.dot(w, r0 * exc)
    if not gga:
        a0 = _ao0(grid)
        V = a0.T @ (a0 * (w * vxc[0])[:, None])
    else:
        ao = grid.ao
        wv = np.empty((4, len(w)))
        wv[0] = 0.5 * w * vxc[0]
        wv[1:] = 2 * w * vxc[1] * rho[1:]
        aow = np.einsum("xpi,xp->pi", ao, wv)
        V = ao[0].T @ aow
        V = V + V.T
    return exc_tot, V, rho


def rks(
    S,
    h,
    jk,
    enn,
    nelec,
    grid,
    xc,
    kshift=0.0,
    conv=1e-11,
    maxiter=100,
    D0=None,
    return_all=False,
):
    """Gamma RKS on (S, h, jk, grid).  jk: D -> (J, K) (exact periodic Coulomb from pbc_gamma I or
    pbc_gdf B).  Hybrids: K_eff = hyb * (K + kshift S D S) (Madelung only on the exact-exchange part)."""
    hyb = libxc.hybrid_coeff(xc)
    if libxc.rsh_coeff(xc)[0] != 0:
        raise NotImplementedError(
            "range-separated hybrids need the attenuated periodic K"
        )
    s, U = np.linalg.eigh(S)
    X = U[:, s > 1e-8] / np.sqrt(s[s > 1e-8])
    nocc = nelec // 2
    if D0 is None:
        e0, C0 = np.linalg.eigh(X.T @ h @ X)
        C0 = X @ C0
        D = 2 * C0[:, :nocc] @ C0[:, :nocc].T
    else:
        D = D0
    from collections import deque

    focks, errs = deque(maxlen=8), deque(maxlen=8)
    e_old = 0.0
    for it in range(maxiter):
        J, K = jk(D)
        if kshift:
            K = K + kshift * S @ D @ S
        exc, V, _ = eval_vxc(grid, D, xc)
        F = h + J - 0.5 * hyb * K + V
        e = np.sum(D * h) + 0.5 * np.sum(D * J) - 0.25 * hyb * np.sum(D * K) + exc + enn
        err = X.T @ (F @ D @ S - S @ D @ F) @ X
        focks.append(F)
        errs.append(err)
        Fd = F
        if len(focks) > 1:
            n = len(focks)
            B = -np.ones((n + 1, n + 1))
            B[-1, -1] = 0
            for i in range(n):
                for j in range(n):
                    B[i, j] = np.sum(errs[i] * errs[j])
            rhs = np.zeros(n + 1)
            rhs[-1] = -1
            c = np.linalg.lstsq(B, rhs, rcond=None)[0][:n]
            Fd = sum(ci * fi for ci, fi in zip(c, focks))
        eps, C = np.linalg.eigh(X.T @ Fd @ X)
        C = X @ C
        D = 2 * C[:, :nocc] @ C[:, :nocc].T
        if abs(e - e_old) < conv and abs(err).max() < 1e-7:
            # report the energy and eigenvalues of the final (undamped) Fock
            eps_f = np.linalg.eigh(X.T @ F @ X)[0]
            out = dict(
                e=e,
                eps=eps_f,
                it=it,
                D=D,
                C=C,
                exc=exc,
                nelec_grid=np.dot(
                    grid.weights,
                    rho_on_grid(grid, D, False)
                    if grid.ao.ndim == 2
                    else rho_on_grid(grid, D, True)[0],
                ),
            )
            return out if return_all else (e, eps_f, it)
        e_old = e
    raise RuntimeError("RKS not converged")


def dense_jk(I):
    return lambda D: (np.einsum("mnls,ls->mn", I, D), np.einsum("mlsn,ls->mn", I, D))


def lattice_overlap(cell, rcut=None):
    rcut = rcut or 2 * ao_rcut(cell.mol, 1e-16)
    Ls = lattice_points(cell.a, rcut)
    sm = cell.supermol(Ls)
    nao, nb0 = cell.mol.nao, cell.mol.nbas
    return (
        sm.intor("int1e_ovlp_cart", shls_slice=(0, nb0, 0, sm.nbas))
        .reshape(nao, len(Ls), nao)
        .sum(1)
    )


def molecular_grid_reference(mol, n_rad, n_ang, adjust=True, scheme="becke"):
    """PySCF MOLECULAR grid with ferric's settings (TA-M4 + xi, Lebedev, original Becke, Becke size
    adjust, no pruning) -> (coords, weights).  The periodic grid must reproduce it in a huge box."""
    from pyscf.dft import gen_grid, radi

    g = gen_grid.Grids(mol)
    g.atom_grid = (n_rad, n_ang)
    g.prune = None
    g.radi_method = radi.treutler
    g.radii_adjust = radi.becke_atomic_radii_adjust if adjust else None
    g.atomic_radii = radi.BRAGG_RADII
    tab = gen_grid.gen_atomic_grids(mol, g.atom_grid, g.radi_method, g.level, None)
    fn = {"becke": gen_grid.original_becke, "ssf": gen_grid.stratmann}[scheme]
    return gen_grid.get_partition(mol, tab, g.radii_adjust, g.atomic_radii, fn)


def molecular_rks(mol, xc, n_rad, n_ang, conv=1e-12, dm0=None):
    """PySCF molecular RKS on exactly the grid of molecular_grid_reference (exact J/K)."""
    from pyscf import dft
    from pyscf.dft import radi

    mf = dft.RKS(mol)
    mf.xc = xc
    mf.grids.atom_grid = (n_rad, n_ang)
    mf.grids.prune = None
    mf.grids.radi_method = radi.treutler
    mf.grids.radii_adjust = radi.becke_atomic_radii_adjust
    mf.grids.atomic_radii = radi.BRAGG_RADII
    mf.grids.cutoff = 1e-100  # 0.0 breaks numint.nr_rks binning (log 0)
    mf.small_rho_cutoff = 0.0
    mf.conv_tol = conv
    mf.verbose = 0
    mf.kernel(dm0)
    return mf


def ov_moment_sigma2(mol, C_occ):
    """Second central moment of the (single) occupied orbital: sum over the occupied block of
    <r^2> - |<r>|^2 (Foster-Boys spread, invariant), used for the hybrid a^-3 prediction."""
    r = mol.intor("int1e_r_cart")
    r2 = mol.intor("int1e_r2_cart")
    rr = np.einsum("xmn,mi,nj->xij", r, C_occ, C_occ)
    return np.einsum("mn,mi,ni->", r2, C_occ, C_occ) - np.einsum("xij,xij->", rr, rr)


def probe_density(cell, S, nelec=None, kind="flat"):
    """A PSD test density with tr(D S_latt) = N exactly.  kind='flat': D = (N/nao) S^-1 (every AO
    equally, delocalised; rho > 0 everywhere).  kind='core': occupied eigenvectors of the cell-0
    MOLECULAR core Hamiltonian in the lattice-S metric (Li 1s-like; for H2 it picks the antibonding
    combination, whose sum-of-D is ~0 and so HIDES a uniform S error -- do not use it for H cells)."""
    mol = cell.mol
    nelec = nelec or mol.nelectron
    if kind == "flat":
        return nelec / mol.nao * np.linalg.inv(S)
    h = mol.intor("int1e_kin_cart") + mol.intor("int1e_nuc_cart")
    s, U = np.linalg.eigh(S)
    X = U[:, s > 1e-8] / np.sqrt(s[s > 1e-8])
    C = X @ np.linalg.eigh(X.T @ h @ X)[1]
    return 2 * C[:, : nelec // 2] @ C[:, : nelec // 2].T


def grid_anchors(grid, S, D, nelec):
    """(|int rho - N|, max|S_grid - S_latt|, E_x^LDA (Slater) on the grid)."""
    a0 = _ao0(grid)
    Sg = a0.T @ (a0 * grid.weights[:, None])
    rho = np.einsum("pi,pi->p", a0 @ D, a0)
    ex = np.dot(grid.weights, rho * libxc.eval_xc("SLATER", rho, spin=0, deriv=0)[0])
    return abs(np.dot(grid.weights, rho) - nelec), abs(Sg - S).max(), ex
