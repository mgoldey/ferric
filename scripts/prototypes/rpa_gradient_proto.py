#!/usr/bin/env python
"""Analytic RI-dRPA nuclear gradient prototype in numpy on ferric's blocks.

Correctness reference for ``crates/ferric-rpa/src/gradient_analytic.rs``.
The equations are section "Analytic nuclear gradient" of
``crates/ferric-rpa/theory.md``; this file follows its notation. The energy
differentiated is the QUADRATURE SUM ferric evaluates at full rank,

    E_c = (1/2pi) sum_k w_k [ln det(1 + Pi_k) - tr Pi_k],
    Pi_k = B g_k(Delta) B^T,   g_k(e) = 4 e / (w_k^2 + e^2),

with B^P_ia = (V^-1/2)_PQ (Q|ia) and Delta the ov orbital-energy-difference
operator (F_vv (x) 1 - 1 (x) F_oo), so E_c is invariant to occupied-occupied
and virtual-virtual rotations and its F dependence is taken by the
Daleckii-Krein divided-difference formula.

Blocks used: ``run_rhf`` (orbitals, the HF gradient), ``run_pdep_rpa``
(the energy and its quadrature nodes and weights), ``AoIntegrals``
(``eri3``/``eri2``, ``jk``, ``cphf``, ``hcore_deriv``, ``overlap_deriv``,
``jk_deriv``). The three- and two-centre RI derivative integrals have no
Python block: their contraction with the fixed RI densities is a 5-point
finite difference of ``eri3``/``eri2`` in the nuclear coordinates (h = 1e-3
Bohr; the integrals are analytic in R, truncation ~ h^4 relative, measured
below 1e-11 Ha/Bohr by halving h). Nothing about speed may be quoted from
this prototype.

Anchors (``--check`` picks one, default all):

* ``energy``: E_c from B equals ``run_pdep_rpa(trunc_thresh=0)``.
* ``mp2``: the same Lagrangian assembly fed closed-shell RI-MP2 densities
  equals ``run_rimp2(with_gradient=True)`` (an independent construction of
  the shared Z-vector / Lagrangian part).
* ``ring``: the second-order term alone (ln det(1+Pi) - tr Pi replaced by
  -tr Pi^2 / 2) at a 200-point grid: Gamma, P_oo, P_vv and the gradient
  equal the closed-form direct-ring MP2 ones (E = 2 E_OS, i.e. SOS-MP2 with
  c_os = 2) to the quadrature error.
* ``fd``: the full gradient against the step-converged 5-point FD of the
  same n-point energy (testdata/reference/validation/rpa_gradient, PySCF
  surface, h = 2e-3 vs 1e-3 agree to 3.9e-10 / 7.4e-10) at n = 40 and
  n = 6, all-electron and frozen core; and against a 5-point FD of ferric's
  own energy (all-electron, n = 40).
* ``invariance``: translational sum and rotational torque of the gradient.
* ``mutations``: the orbital-energy term dropped; a quadrature weight
  applied twice. Each must miss the FD reference by orders of magnitude.

Usage: ``python scripts/prototypes/rpa_gradient_proto.py [--check NAME]``
from the repository root with ferric importable.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import ferric

ROOT = Path(__file__).resolve().parents[2]
MOL_DIR = ROOT / "testdata/molecules/validation"
REF_DIR = ROOT / "testdata/reference/validation/rpa_gradient"
BASIS = "cc-pvdz"
AUX = "cc-pvdz-ri"
H_INT = 1e-3  # step of the RI derivative-integral FD (Bohr)
FD5 = np.array([1.0, -8.0, 8.0, -1.0]) / 12.0  # weights at -2h, -h, +h, +2h
FD5_STEPS = (-2.0, -1.0, 1.0, 2.0)


# ---------------------------------------------------------------------------
# Reference state
# ---------------------------------------------------------------------------


@dataclass
class Reference:
    """RHF reference and the RI quantities of one geometry."""

    symbols: list
    coords: np.ndarray  # Bohr
    mol: object
    ints: object
    aux: object
    rhf: object
    c: np.ndarray
    eps: np.ndarray
    nocc: int
    v_mhalf: np.ndarray  # V^-1/2
    b_full: np.ndarray  # (naux, nmo, nmo), B^P_pq = (V^-1/2 (Q|pq))

    @property
    def nvir(self) -> int:
        return self.c.shape[1] - self.nocc


BOHR_ANGSTROM = 0.52917721092  # ferric_core::units::BOHR_RADIUS_ANGSTROM


def molecule(symbols, coords_bohr):
    """Molecule at Bohr coordinates (via repr-exact Angstrom XYZ text)."""
    rows = [
        f"{s} {x * BOHR_ANGSTROM!r} {y * BOHR_ANGSTROM!r} {z * BOHR_ANGSTROM!r}"
        for s, (x, y, z) in zip(symbols, coords_bohr)
    ]
    return ferric.Molecule.from_xyz_string(f"{len(rows)}\n\n" + "\n".join(rows) + "\n")


def tight_rhf(mol, bs):
    rhf = ferric.run_rhf(mol, bs, density_conv=1e-10, with_gradient=True)
    assert rhf.converged
    return rhf


def build_reference(symbols, coords) -> Reference:
    bs = ferric.BasisSet.bundled(BASIS)
    aux = ferric.BasisSet.bundled(AUX)
    mol = molecule(symbols, coords)
    rhf = tight_rhf(mol, bs)
    ints = ferric.AoIntegrals(mol, bs)
    c = np.asarray(rhf.mo_coefficients())
    eps = np.asarray(rhf.orbital_energies())
    nocc = mol.nelec() // 2
    v = np.asarray(ints.eri2(aux))
    w, u = np.linalg.eigh(v)
    v_mhalf = (u / np.sqrt(w)) @ u.T
    eri3 = np.asarray(ints.eri3(aux))
    b_full = np.einsum("PQ,Qmn,mp,nq->Ppq", v_mhalf, eri3, c, c, optimize=True)
    return Reference(
        symbols, np.asarray(coords), mol, ints, aux, rhf, c, eps, nocc, v_mhalf, b_full
    )


def load_system(name):
    mol = ferric.Molecule.from_xyz(str(MOL_DIR / f"{name}.xyz"))
    syms = [
        line.split()[0]
        for line in (MOL_DIR / f"{name}.xyz").read_text().splitlines()[2:]
        if line.strip()
    ]
    return syms, np.asarray(mol.coords_bohr())


# ---------------------------------------------------------------------------
# Energy, Gamma and the Daleckii-Krein orbital-energy densities
# ---------------------------------------------------------------------------


def g_of(e, w):
    return 4.0 * e / (w * w + e * e)


def g_divided_difference(e1, e2, w):
    """g[1](e1, e2) = (g(e1) - g(e2)) / (e1 - e2), g'(e) on the diagonal.

    Closed form: 4 (w^2 - e1 e2) / ((w^2 + e1^2)(w^2 + e2^2)), which has no
    0/0 at e1 = e2 and equals g'(e1) there.
    """
    return 4.0 * (w * w - e1 * e2) / ((w * w + e1 * e1) * (w * w + e2 * e2))


@dataclass
class CorrDensities:
    """dE_c/dB (Gamma, naux x nocc_act x nvir), dE_c/dF_oo, dE_c/dF_vv."""

    energy: float
    gamma: np.ndarray
    p_oo: np.ndarray  # active occupied block
    p_vv: np.ndarray


def rpa_densities(
    b_ov, e_occ, e_vir, freqs, weights, order="full", weight_mutation=False
):
    """Energy, Gamma, P_oo, P_vv of the n-point quadrature sum.

    order="full": S(w) = ln det(1+Pi) - tr Pi, dS = tr[Q dPi], Q = (1+Pi)^-1 - 1.
    order="ring": S(w) = -tr Pi^2 / 2, Q = -Pi (the second-order term).
    weight_mutation: apply the first node's weight twice (a mutation).
    """
    naux, no, nv = b_ov.shape
    bm = b_ov.reshape(naux, no * nv)
    e_ia = (e_vir[None, :] - e_occ[:, None]).reshape(-1)
    energy = 0.0
    gamma = np.zeros_like(bm)
    p_oo = np.zeros((no, no))
    p_vv = np.zeros((nv, nv))
    eye = np.eye(naux)
    for k, (w, wk) in enumerate(zip(freqs, weights)):
        if weight_mutation and k == 0:
            wk = wk * wk
        g = g_of(e_ia, w)
        pi = (bm * g) @ bm.T
        if order == "full":
            sign, logdet = np.linalg.slogdet(eye + pi)
            assert sign > 0
            s = logdet - np.trace(pi)
            q = np.linalg.inv(eye + pi) - eye
        else:
            s = -0.5 * np.sum(pi * pi)
            q = -pi
        energy += wk * s / (2 * np.pi)
        qb = q @ bm
        gamma += (wk / np.pi) * g * qb
        m = (bm.T @ qb).reshape(no, nv, no, nv)
        e2 = e_ia.reshape(no, nv)
        # P_ab = (1/2pi) w_k sum_i M[ia,ib] g1(e_ia, e_ib)
        m_iab = np.einsum("iaib->iab", m)
        g1v = g_divided_difference(e2[:, :, None], e2[:, None, :], w)
        p_vv += (wk / (2 * np.pi)) * np.einsum("iab,iab->ab", m_iab, g1v)
        # P_ij = -(1/2pi) w_k sum_a M[ia,ja] g1(e_ia, e_ja)
        m_ija = np.einsum("iaja->ija", m)
        g1o = g_divided_difference(e2.T[:, :, None], e2.T[:, None, :], w)  # (a, i, j)
        p_oo -= (wk / (2 * np.pi)) * np.einsum("ija,aij->ij", m_ija, g1o)
    return CorrDensities(energy, gamma.reshape(naux, no, nv), p_oo, p_vv)


def direct_ring_mp2_densities(b_ov, e_occ, e_vir):
    """Closed-form direct-ring MP2 (E = 2 sum K^2/D, D = e_i+e_j-e_a-e_b)."""
    naux, no, nv = b_ov.shape
    k = np.einsum("Pia,Pjb->iajb", b_ov, b_ov)
    d = (
        e_occ[:, None, None, None]
        - e_vir[None, :, None, None]
        + e_occ[None, None, :, None]
        - e_vir[None, None, None, :]
    )
    t = k / d
    energy = 2.0 * np.sum(k * t)
    gamma = 8.0 * np.einsum("iajb,Pjb->Pia", t, b_ov)
    p_vv = 4.0 * np.einsum("iajc,ibjc->ab", t, t)
    p_oo = -4.0 * np.einsum("iakb,jakb->ij", t, t)
    return CorrDensities(energy, gamma, p_oo, p_vv)


def rimp2_densities(b_ov, e_occ, e_vir):
    """Closed-shell MP2 E = sum K (2K - K_x) / D and its Hylleraas densities."""
    k = np.einsum("Pia,Pjb->iajb", b_ov, b_ov)
    d = (
        e_occ[:, None, None, None]
        - e_vir[None, :, None, None]
        + e_occ[None, None, :, None]
        - e_vir[None, None, None, :]
    )
    t = k / d
    tt = 2.0 * t - t.transpose(0, 3, 2, 1)
    energy = np.sum(k * tt)
    gamma = 4.0 * np.einsum("iajb,Pjb->Pia", tt, b_ov)
    # dE/dD_iajb = -t tt; dD = de_i + de_j - de_a - de_b, so per slot:
    p_vv = 2.0 * np.einsum("iajc,ibjc->ab", t, tt)
    p_oo = -2.0 * np.einsum("iakb,jakb->ij", t, tt)
    return CorrDensities(energy, gamma, 0.5 * (p_oo + p_oo.T), 0.5 * (p_vv + p_vv.T))


# ---------------------------------------------------------------------------
# Lagrangian, Z-vector and the gradient
# ---------------------------------------------------------------------------


def g_op(ints, x):
    """G[X] = J[X] - K[X]/2 (TOTAL-density convention, X symmetric)."""
    j, kk = ints.jk(0.5 * (x + x.T), thresh=1e-14)
    return np.asarray(j) - 0.5 * np.asarray(kk)


def ri_derivative_contractions(ref: Reference, gamma_t_ao, gamma_pq):
    """sum G~^Q_mn d(Q|mn)/dx and -1/2 sum gamma_PQ dV_PQ/dx by 5-point FD."""
    bs = ferric.BasisSet.bundled(BASIS)
    natm = len(ref.symbols)
    out = np.zeros((natm, 3))
    for a in range(natm):
        for x in range(3):
            acc = 0.0
            for wgt, s in zip(FD5, FD5_STEPS):
                coords = ref.coords.copy()
                coords[a, x] += s * H_INT
                ints = ferric.AoIntegrals(molecule(ref.symbols, coords), bs)
                e3 = np.asarray(ints.eri3(ref.aux))
                e2 = np.asarray(ints.eri2(ref.aux))
                acc += wgt * (np.sum(gamma_t_ao * e3) - 0.5 * np.sum(gamma_pq * e2))
            out[a, x] = acc / H_INT
    return out


def per_coordinate(deriv, mat):
    """sum_mn deriv[x, m, n] mat[m, n] reshaped to (natoms, 3)."""
    return np.einsum("xmn,mn->x", deriv, mat).reshape(-1, 3)


@dataclass
class GradientParts:
    total: np.ndarray
    hf: np.ndarray
    density: np.ndarray  # relaxed correlation density term
    overlap: np.ndarray
    ri: np.ndarray
    z_norm: float


def correlation_gradient(
    ref: Reference, dens: CorrDensities, frozen_core=0, drop_orbital_energy=False
):
    """Total gradient E_HF + E_c from the correlation densities (theory.md)."""
    c, eps, no = ref.c, ref.eps, ref.nocc
    nmo = c.shape[1]
    fc = frozen_core
    co, cv = c[:, :no], c[:, no:]
    gamma = dens.gamma  # (naux, no - fc, nv)
    p_oo = np.zeros((no, no))
    p_oo[fc:, fc:] = dens.p_oo
    p_vv = dens.p_vv.copy()
    if drop_orbital_energy:
        p_oo[:] = 0.0
        p_vv[:] = 0.0
    bf = ref.b_full
    # Y_pq = dE/dU_pq: orbital q's coefficient change along p.
    y = np.zeros((nmo, nmo))
    y[:, fc:no] = np.einsum("Pia,Ppa->pi", gamma, bf[:, :, no:])
    y[:, no:] = np.einsum("Pia,Pip->pa", gamma, bf[:, fc:no, :])
    if fc:
        # Core-active rotations are not a symmetry of E_c: the canonical
        # condition F_Ij = 0 fixes them (theory.md, frozen core). The pair
        # term is split over P_Ij and P_jI, which sum dF over both orders.
        for i_core in range(fc):
            for j in range(fc, no):
                pij = -0.5 * (y[i_core, j] - y[j, i_core]) / (eps[i_core] - eps[j])
                p_oo[i_core, j] = p_oo[j, i_core] = pij
    y[:no, :no] += 2.0 * p_oo * eps[:no, None]
    y[no:, no:] += 2.0 * p_vv * eps[no:, None]
    p_ao = co @ p_oo @ co.T + cv @ p_vv @ cv.T
    gp = g_op(ref.ints, p_ao)
    lag = y[no:, :no] - y[:no, no:].T + 4.0 * (cv.T @ gp @ co)
    cphf = ref.ints.cphf(c, list(eps), no)
    z = np.asarray(cphf.solve(lag, tol=1e-11, max_iter=200))
    z_ao = cv @ z @ co.T
    z_sym = 0.5 * (z_ao + z_ao.T)
    d2 = p_ao - z_sym
    d_hf = np.asarray(ref.rhf.density())
    jx, kx = ref.ints.jk_deriv(d_hf)
    fx = np.asarray(ref.ints.hcore_deriv()) + np.asarray(jx) - 0.5 * np.asarray(kx)
    g_density = per_coordinate(fx, d2)
    gd2 = g_op(ref.ints, d2)
    w = np.zeros((nmo, nmo))
    w[:no, :no] = -0.5 * y[:no, :no] - 2.0 * (co.T @ gd2 @ co)
    w[no:, no:] = -0.5 * y[no:, no:]
    w[:no, no:] = -y[:no, no:]
    w[no:, :no] = z * eps[None, :no]
    g_overlap = per_coordinate(np.asarray(ref.ints.overlap_deriv()), c @ w @ c.T)
    gamma_t = np.einsum("PQ,Qia->Pia", ref.v_mhalf, gamma)
    gamma_t_ao = np.einsum("Pia,mi,na->Pmn", gamma_t, c[:, fc:no], cv, optimize=True)
    cfit = np.einsum("PQ,Qia->Pia", ref.v_mhalf, bf[:, fc:no, no:])
    gamma_pq = np.einsum("Pia,Qia->PQ", gamma_t, cfit)
    g_ri = ri_derivative_contractions(ref, gamma_t_ao, gamma_pq)
    g_hf = np.asarray(ref.rhf.gradient())
    total = g_hf + g_density + g_overlap + g_ri
    return GradientParts(
        total, g_hf, g_density, g_overlap, g_ri, float(np.linalg.norm(z))
    )


# ---------------------------------------------------------------------------
# Anchors
# ---------------------------------------------------------------------------


def quadrature(ref: Reference, n_quad, frozen_core=0):
    bs = ferric.BasisSet.bundled(BASIS)
    r = ferric.run_pdep_rpa(
        ref.mol,
        bs,
        ref.aux,
        frozen_core=frozen_core,
        n_quad=n_quad,
        u0=0.5,
        trunc_thresh=0.0,
        scf_density_conv=1e-10,
    )
    return np.asarray(r.quad_freqs), np.asarray(r.quad_weights), r


def b_ov(ref: Reference, fc=0):
    return ref.b_full[:, fc : ref.nocc, ref.nocc :]


def rpa_of(ref, n_quad, fc=0, **kw):
    f, w, r = quadrature(ref, n_quad, fc)
    d = rpa_densities(
        b_ov(ref, fc), ref.eps[fc : ref.nocc], ref.eps[ref.nocc :], f, w, **kw
    )
    return d, r


def maxabs(a):
    return float(np.max(np.abs(a)))


def check_energy(ref, name):
    for n in (40, 6):
        for fc in (0, 1):
            d, r = rpa_of(ref, n, fc)
            print(
                f"[energy] {name} n={n} fc={fc}: E_c proto {d.energy:+.14f} ferric {r.e_rpa:+.14f} |d| {abs(d.energy - r.e_rpa):.2e}"
            )


def check_mp2(ref, name):
    bs = ferric.BasisSet.bundled(BASIS)
    res = ferric.run_rimp2(
        ref.mol, bs, ref.aux, scf_density_conv=1e-10, with_gradient=True
    )
    d = rimp2_densities(b_ov(ref), ref.eps[: ref.nocc], ref.eps[ref.nocc :])
    g = correlation_gradient(ref, d)
    print(
        f"[mp2] {name}: E_MP2 proto {d.energy:+.12f} ferric {res.mp2_corr:+.12f} |d| {abs(d.energy - res.mp2_corr):.2e}"
    )
    print(
        f"[mp2] {name}: |g_proto - g_ferric RI-MP2|_max {maxabs(g.total - np.asarray(res.gradient())):.2e}"
    )


def check_ring(ref, name):
    occ, vir = ref.eps[: ref.nocc], ref.eps[ref.nocc :]
    exact = direct_ring_mp2_densities(b_ov(ref), occ, vir)
    g_exact = correlation_gradient(ref, exact)
    for n in (40, 100, 200):
        f, w = gauss_legendre(n, 0.5)
        q = rpa_densities(b_ov(ref), occ, vir, f, w, order="ring")
        g_q = correlation_gradient(ref, q)
        print(
            f"[ring] {name} n={n}: |dE| {abs(q.energy - exact.energy):.2e} |dGamma| {maxabs(q.gamma - exact.gamma):.2e} "
            f"|dP_oo| {maxabs(q.p_oo - exact.p_oo):.2e} |dP_vv| {maxabs(q.p_vv - exact.p_vv):.2e} "
            f"|dg| {maxabs(g_q.total - g_exact.total):.2e}"
        )


def gauss_legendre(n, u0):
    """ferric's GaussLegendre map: w = u0 (1+x)/(1-x)."""
    x, wx = np.polynomial.legendre.leggauss(n)
    return u0 * (1 + x) / (1 - x), wx * 2 * u0 / (1 - x) ** 2


def reference_fd(name):
    return json.loads((REF_DIR / f"{name}_{BASIS}.json").read_text())


def own_fd5(ref, n_quad, h=1e-3):
    bs = ferric.BasisSet.bundled(BASIS)
    natm = len(ref.symbols)
    g = np.zeros((natm, 3))
    for a in range(natm):
        for x in range(3):
            acc = 0.0
            for wgt, s in zip(FD5, FD5_STEPS):
                coords = ref.coords.copy()
                coords[a, x] += s * h
                r = ferric.run_pdep_rpa(
                    molecule(ref.symbols, coords),
                    bs,
                    ref.aux,
                    n_quad=n_quad,
                    u0=0.5,
                    trunc_thresh=0.0,
                    scf_density_conv=1e-10,
                )
                acc += wgt * r.total_energy
            g[a, x] = acc / h
    return g


def check_fd(ref, name, own=True):
    js = reference_fd(name)
    fd = js["gradient"]["fd"]["fd5_h0.002"]
    cases = [(40, 0, "total"), (6, 0, "total_coarse"), (40, 1, "total_fc")]
    out = {}
    for n, fc, key in cases:
        if key not in fd:
            continue
        d, _ = rpa_of(ref, n, fc)
        g = correlation_gradient(ref, d, frozen_core=fc)
        dev = maxabs(g.total - np.asarray(fd[key]))
        out[key] = g.total
        print(
            f"[fd] {name} n={n} fc={fc}: |g - FD5(PySCF)|_max {dev:.2e}  |z| {g.z_norm:.3e}"
        )
    print(
        f"[fd] {name}: n=40 vs n=6 analytic differ by {maxabs(out['total'] - out['total_coarse']):.2e}"
    )
    if own:
        g_own = own_fd5(ref, 40)
        print(
            f"[fd] {name}: |g - FD5(ferric own, h=1e-3)|_max {maxabs(out['total'] - g_own):.2e}"
        )
    return out["total"]


def check_invariance(ref, name, g):
    trans = maxabs(g.sum(axis=0))
    torque = maxabs(np.cross(ref.coords, g).sum(axis=0))
    print(
        f"[invariance] {name}: |sum_A g_A|_max {trans:.2e}  |sum_A R_A x g_A|_max {torque:.2e}"
    )


def check_mutations(ref, name):
    fd = np.asarray(reference_fd(name)["gradient"]["fd"]["fd5_h0.002"]["total"])
    d, _ = rpa_of(ref, 40)
    g_drop = correlation_gradient(ref, d, drop_orbital_energy=True)
    print(
        f"[mutation] {name}: orbital-energy term dropped: |g - FD5|_max {maxabs(g_drop.total - fd):.2e}"
    )
    d2, _ = rpa_of(ref, 40, weight_mutation=True)
    g_w = correlation_gradient(ref, d2)
    print(
        f"[mutation] {name}: first quadrature weight applied twice: |g - FD5|_max {maxabs(g_w.total - fd):.2e}"
    )


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", default="all")
    ap.add_argument("--systems", default="h2o_distorted,nh3_distorted")
    ap.add_argument("--no-own-fd", action="store_true")
    args = ap.parse_args(argv)
    print(f"ferric build {ferric.__build__}")
    for name in args.systems.split(","):
        ref = build_reference(*load_system(name))
        want = args.check
        if want in ("all", "energy"):
            check_energy(ref, name)
        if want in ("all", "mp2"):
            check_mp2(ref, name)
        if want in ("all", "ring"):
            check_ring(ref, name)
        if want in ("all", "fd", "invariance"):
            g = check_fd(ref, name, own=not args.no_own_fd)
            check_invariance(ref, name, g)
        if want in ("all", "mutations"):
            check_mutations(ref, name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
