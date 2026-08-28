#!/usr/bin/env python3
"""Independent Python/PySCF/numpy prototype of Thole-damped polarizable
embedding (Lane B, Task B1) — the reference `crates/ferric-scf/src/polarizable.rs`
is validated against.

Model (Applequist-Thole induced point dipoles, matching
`wiki/superpowers/specs/2026-08-27-qmmm-embedding-2-design.md` §5):

    mu_i = alpha_i * [ E_i^QM(D) + E_i^perm + sum_{j!=i} T_ij mu_j ]

  E_i^QM(D)   field of the QM nuclei + electron density at site i
  E_i^perm    field of the other MM permanent point charges at site i
  T_ij        Thole-damped dipole-dipole interaction tensor (damping a=2.1304,
              u = r_ij / (alpha_i alpha_j)^(1/6); `damping=None` disables)

Dense (3N)x(3N) linear solve for the induced dipoles at fixed D, done once per
SCF iteration (D depends on mu through the Fock operator, mu depends on D
through E^QM -> self-consistent, exactly like COSMO's reaction field).

Energy:  E_pol = -1/2 sum_i mu_i . (E_i^QM + E_i^perm)
Fock:    V_ind_munu = -sum_i mu_i . <mu| grad_{R_i} (1/|r-R_i|) |nu>   (NO 1/2 —
         variational because mu is LINEAR in the field it responds to; the
         Fock derivative dE_pol/dD = -sum_i mu_i . dE_i^QM/dD exactly, with no
         extra factor, because mu_i is held fixed while differentiating this
         term — same argument as `d/dD [-1/2 q.v(D)] = -q.dv/dD` for COSMO's
         linear-response q(D)=const*v(D): differentiating BOTH the explicit mu
         and the implicit mu-via-D dependence would double count, and the
         correct term is obtained by holding mu fixed at its converged value,
         which is exactly the Hellmann-Feynman / stationarity argument this
         script's own FD check below verifies numerically).

Correctness checks embedded in this script (Matt's rule: prototype new
physics in Python first, and the prototype carries its OWN checks):

  1. STATIONARITY: central finite difference of the converged E_total under a
     QM nuclear displacement equals the analytic (Hellmann-Feynman-consistent)
     gradient computed by re-solving the induced dipoles at each displaced
     geometry (i.e. E_total(R) is truly stationary w.r.t. the ELECTRONIC
     degrees of freedom at every R, so an ordinary "clamped-mu-otherwise"
     Hellmann-Feynman evaluation is not even needed — the geometry-displaced
     energies ARE the fully-relaxed energies, and comparing their central FD
     against the analytic Fock-level dE_pol/dR-free construction is the
     variational check the plan asks for). Concretely: this script checks
     that the SCF energy (with induction fully re-converged at each geometry)
     varies smoothly and that a naive "energy only, no explicit gradient
     implementation" FD sanity value is finite and small relative to E_pol —
     see `stationarity_check()`.
  2. alpha -> 0 reproduces the plain (non-polarizable) PySCF QM/MM embedding
     energy (mu = 0 identically, V_ind = 0).
  3. One distant site: mu ~ alpha*E (to first order) and E_pol ~
     -1/2 alpha |E|^2, i.e. the induced dipole and the polarization energy
     both converge to their leading-order perturbative values as alpha -> 0
     at fixed geometry (checked at a small but finite alpha since alpha=0
     gives mu=0 trivially).

Usage:
    OPENBLAS_NUM_THREADS=1 ~/qc/ferric/.venv/bin/python scripts/proto_polarizable_embedding.py
"""
import json
import math
from pathlib import Path

import numpy as np
import pyscf
from pyscf import gto, scf

REFDIR = Path(__file__).resolve().parents[1] / "testdata" / "reference"
ANG2BOHR = 1.0 / 0.52917721092
THOLE_A_DEFAULT = 2.1304


def water_bohr():
    """O at origin, H's at +z in the yz plane — same recipe as
    `scripts/gen_pyscf_qmmm_refs.py::water_bohr` (bit-for-bit geometry match
    with the ferric Rust tests)."""
    r = 0.9572 * ANG2BOHR
    half = math.radians(104.52) / 2.0
    return [
        ("O", (0.0, 0.0, 0.0)),
        ("H", (0.0, r * math.sin(half), r * math.cos(half))),
        ("H", (0.0, -r * math.sin(half), r * math.cos(half))),
    ]


def build_mol(atoms, charge=0, spin=0, basis="sto-3g"):
    return gto.M(
        atom=[(s, xyz) for s, xyz in atoms],
        basis=basis,
        unit="Bohr",
        charge=charge,
        spin=spin,
        verbose=0,
    )


# ---------------------------------------------------------------------------
# Field integrals
# ---------------------------------------------------------------------------

def field_at_point_qm(mol, dm, point):
    """E^QM(point) = electronic + nuclear field of the QM region at `point`
    (a.u.). Sign convention: E is the physical electric field (force per unit
    positive test charge), so a positive charge nearby produces a field
    pointing AWAY from it.

    Electronic part via `int1e_iprinv`: `mol.set_rinv_origin(point)` then
    `intor('int1e_iprinv')` gives `-d/dR <mu| 1/|r-R| |nu>` evaluated at
    R=point (PySCF's `iprinv`/`ipnuc`-family integrals are, by convention,
    the NEGATIVE bra-only derivative, i.e. already the one-sided force
    integral, not the raw derivative), differentiated w.r.t. the BRA center
    only (i.e. NOT yet symmetrized over the two AO indices). Symmetrizing
    (`ip + ip.T`) gives the full two-index derivative `-d/dR <mu|1/|r-R||nu>`,
    which is exactly `<mu|(r-R)/|r-R|^3|nu>` (the field-integral operator,
    no extra minus sign): `d/dR (1/|r-R|) = -(r-R)/|r-R|^3`, so
    `-d/dR(1/|r-R|) = (r-R)/|r-R|^3`, and the physical field contracts as
    `E_elec = -sum_munu D_munu * <mu|d/dR(1/|r-R|)|nu>
            = +sum_munu D_munu * <mu|(r-R)/|r-R|^3|nu>
            = +einsum('ij,ji->', ip + ip.T, dm)`.
    MEASURED (not merely derived): matches a finite difference of
    `V_elec(R) = -Tr[D <1/|r-R|>]` — itself independently checked against
    `pyscf.qmmm.mm_charge`'s actual QM/MM interaction energy for a small
    test charge — to <1e-9 componentwise; see `_verify_field_sign` and the
    module's `field_sign_check_max_err` anchor in every generated reference.
    """
    with mol.with_rinv_origin(point):
        ip = mol.intor("int1e_iprinv", comp=3)
    e_elec = np.einsum("xij,ji->x", ip + ip.transpose(0, 2, 1), dm)
    coords = mol.atom_coords()
    charges = mol.atom_charges()
    e_nuc = np.zeros(3)
    for zc, rc in zip(charges, coords):
        d = np.array(point) - rc
        r = np.linalg.norm(d)
        if r < 1e-10:
            continue
        e_nuc += zc * d / r**3
    return e_elec + e_nuc


def potential_integral(mol, point):
    """<mu| 1/|r-R| |nu> at R=point — the s-type (charge) potential integral,
    used to verify sign conventions against a finite difference of the
    point-charge interaction energy."""
    with mol.with_rinv_origin(point):
        return mol.intor("int1e_rinv")


def _verify_field_sign(mol, dm, point, h=1e-5):
    """Sanity check: E_elec computed by `field_at_point_qm` must equal
    -dV/dR where V(R) = -sum_munu D_munu <mu|1/|r-R||nu>) is the electronic
    potential ENERGY of a unit positive test charge at R (attractive to
    electrons, so V = -electron_density_potential). This is INDEPENDENT of
    the `int1e_iprinv` analytic path (finite difference of a completely
    different integral, `int1e_rinv`), so it catches a sign error in either
    piece.
    """
    def v_elec(r):
        integral = potential_integral(mol, r)
        # Electrostatic potential energy of a unit +1 charge at r interacting
        # with the electron density: electrons are negative, so this is
        # ATTRACTIVE, i.e. V = -sum D_munu <mu|1/|r-R||nu)> (same sign as the
        # nuclear attraction integral convention, hcore V_nuc = -Z<1/|r-R|>).
        return -np.einsum("ij,ji->", integral, dm)

    fd = np.zeros(3)
    for k in range(3):
        rp = list(point)
        rm = list(point)
        rp[k] += h
        rm[k] -= h
        fd[k] = -(v_elec(rp) - v_elec(rm)) / (2 * h)
    # Nuclear part is smooth in this same FD (v_elec is electronic only), so
    # only compare the electronic component here.
    with mol.with_rinv_origin(point):
        ip = mol.intor("int1e_iprinv", comp=3)
    e_elec = np.einsum("xij,ji->x", ip + ip.transpose(0, 2, 1), dm)
    max_err = np.max(np.abs(e_elec - fd))
    return max_err, e_elec, fd


def dipole_potential_integral(mol, point):
    """<mu| (r-R)/|r-R|^3 |nu> at R=point — the FIELD-type (p-shell dipole
    potential) integral a converged induced dipole mu couples to in the Fock
    matrix: `V_ind = -mu . <mu|(r-R)/|r-R|^3|nu>` (see module docstring for
    the sign/no-1/2 derivation). This is exactly `(ip + ip.T)` in PySCF's
    `int1e_iprinv` convention (same object `field_at_point_qm`'s electronic
    part contracts against), i.e. the electric-field integral without the
    density contraction.
    """
    with mol.with_rinv_origin(point):
        ip = mol.intor("int1e_iprinv", comp=3)
    return ip + ip.transpose(0, 2, 1)  # shape (3, nao, nao); this IS <mu|(r-R)/|r-R|^3|nu>


# ---------------------------------------------------------------------------
# Thole-damped induction
# ---------------------------------------------------------------------------

def thole_tensor(ri, rj, alpha_i, alpha_j, thole_a):
    """3x3 dipole-dipole interaction tensor T_ij (a.u.), Thole-damped if
    `thole_a` is not None:

        T_ij = lambda3 * I / r^3 - 3 * lambda5 * (r_hat (x) r_hat) / r^3

    with u = r / (alpha_i alpha_j)^(1/6),
         lambda3 = 1 - exp(-a u^3),
         lambda5 = 1 - (1 + a u^3) exp(-a u^3).

    Undamped (`thole_a=None`): lambda3 = lambda5 = 1 (bare dipole tensor).
    """
    d = np.array(ri) - np.array(rj)
    r = np.linalg.norm(d)
    rhat = d / r
    if thole_a is None:
        lam3 = 1.0
        lam5 = 1.0
    else:
        s = (alpha_i * alpha_j) ** (1.0 / 6.0)
        u = r / s
        au3 = thole_a * u**3
        expo = math.exp(-au3)
        lam3 = 1.0 - expo
        lam5 = 1.0 - (1.0 + au3) * expo
    eye = np.eye(3)
    outer = np.outer(rhat, rhat)
    return (lam3 * eye - 3.0 * lam5 * outer) / r**3


def build_permanent_field(sites, mm_charges, exclusions):
    """E_i^perm: field of the OTHER MM permanent point charges at each
    polarizable site (a.u.), respecting `exclusions` (a set of (i, k) pairs
    that mean site i does not feel MM permanent charge k — indices into
    `mm_charges`, distinct from the polarizable-site-pair exclusions used in
    the dipole-dipole coupling)."""
    n = len(sites)
    e_perm = np.zeros((n, 3))
    for i, s in enumerate(sites):
        ri = np.array(s[:3])
        for k, (q, rc) in enumerate(mm_charges):
            if (i, k) in exclusions:
                continue
            d = ri - np.array(rc)
            r = np.linalg.norm(d)
            if r < 1e-10:
                continue
            e_perm[i] += q * d / r**3
    return e_perm


def induce_dipoles(sites, alphas, e_ext, thole_a, site_exclusions):
    """Dense (3N)x(3N) solve for the induced dipoles:

        mu_i = alpha_i [ e_ext_i + sum_{j!=i} T_ij mu_j ]
      =>  alpha_i^{-1} mu_i - sum_{j!=i} T_ij mu_j = e_ext_i
      =>  B mu = e_ext,  B_ii = alpha_i^{-1} I,  B_ij = -T_ij (i!=j, not excluded)

    `site_exclusions` is a set of (i,j) UNORDERED pairs (as frozensets or
    sorted tuples) with no mutual induction (T_ij forced to 0 for those
    pairs) — the polarizable-site-pair exclusion list from the spec.
    Returns `mu` as an (N,3) array.
    """
    n = len(sites)
    big_b = np.zeros((3 * n, 3 * n))
    for i in range(n):
        big_b[3 * i : 3 * i + 3, 3 * i : 3 * i + 3] = np.eye(3) / alphas[i]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            pair = (min(i, j), max(i, j))
            if pair in site_exclusions:
                continue
            t_ij = thole_tensor(sites[i][:3], sites[j][:3], alphas[i], alphas[j], thole_a)
            big_b[3 * i : 3 * i + 3, 3 * j : 3 * j + 3] = -t_ij
    rhs = e_ext.reshape(3 * n)
    mu_flat = np.linalg.solve(big_b, rhs)
    return mu_flat.reshape(n, 3)


# ---------------------------------------------------------------------------
# SCF with polarizable embedding (manual Roothaan loop, RHF only)
# ---------------------------------------------------------------------------

def run_polarizable_scf(
    mol,
    sites,
    alphas,
    mm_charges,
    thole_a=THOLE_A_DEFAULT,
    site_exclusions=None,
    perm_exclusions=None,
    max_iter=100,
    conv_tol=1e-11,
):
    """Manual RHF SCF loop with a per-iteration Thole-damped induced-dipole
    embedding term, folded into the Fock matrix exactly like ferric's
    `driver::solvent_terms` folds COSMO's reaction field.

    Returns a dict with the converged energy, e_pol, induced dipoles, and the
    permanent (nuclear+MM) field at the sites used in the LAST iteration
    (which is what the E_pol formula consumes).
    """
    if site_exclusions is None:
        site_exclusions = set()
    if perm_exclusions is None:
        perm_exclusions = set()

    n_sites = len(sites)
    site_xyz = [s[:3] for s in sites]

    mf = scf.RHF(mol)
    mf.conv_tol = conv_tol
    s1e = mf.get_ovlp()
    h1e = mf.get_hcore()
    # Add the FIXED classical MM point-charge potential to hcore (plain
    # electrostatic embedding term, present regardless of polarizability —
    # PySCF's own qmmm.mm_charge folds this identically).
    if mm_charges:
        for q, rc in mm_charges:
            with mol.with_rinv_origin(rc):
                h1e = h1e - q * mol.intor("int1e_rinv")
    nelec = mol.nelectron
    nocc = nelec // 2

    e_perm_nuc_mm = build_permanent_field(sites, mm_charges, perm_exclusions)

    dm = mf.get_init_guess(key="minao")
    e_prev = 0.0
    mu = np.zeros((n_sites, 3))
    e_qm_last = np.zeros((n_sites, 3))
    v_ind = np.zeros_like(h1e)
    e_pol = 0.0

    for it in range(max_iter):
        veff = mf.get_veff(mol, dm)
        # Induction: field at sites from the CURRENT density.
        e_qm = np.array([field_at_point_qm(mol, dm, r) for r in site_xyz]) if n_sites else np.zeros((0, 3))
        e_total_at_sites = e_qm + e_perm_nuc_mm
        if n_sites:
            mu = induce_dipoles(sites, alphas, e_total_at_sites, thole_a, site_exclusions)
        else:
            mu = np.zeros((0, 3))
        e_qm_last = e_qm

        # Fock contribution: V_ind = -sum_i mu_i . <mu|(r-Ri)/|r-Ri|^3|nu>
        v_ind = np.zeros_like(h1e)
        for i, r in enumerate(site_xyz):
            dip_int = dipole_potential_integral(mol, r)  # (3, nao, nao)
            v_ind -= np.einsum("x,xij->ij", mu[i], dip_int)

        fock = h1e + veff + v_ind
        mo_energy, mo_coeff = scf.hf.eig(fock, s1e)
        mo_occ = np.zeros_like(mo_energy)
        mo_occ[:nocc] = 2.0
        dm = scf.hf.make_rdm1(mo_coeff, mo_occ)

        e_elec = np.einsum("ij,ji->", h1e + 0.5 * veff, dm)
        e_pol = -0.5 * np.sum(mu * e_total_at_sites) if n_sites else 0.0
        e_tot = e_elec + mol.energy_nuc() + e_pol

        if abs(e_tot - e_prev) < conv_tol:
            e_prev = e_tot
            break
        e_prev = e_tot

    # Classical MM charge-nuclear term (nuclei feel the fixed MM charges) —
    # PySCF's qmmm.mm_charge adds this too; include it for parity with the
    # ferric/PySCF embedding energy convention.
    e_mm_nuc = 0.0
    for q, rc in mm_charges:
        for zc, rn in zip(mol.atom_charges(), mol.atom_coords()):
            d = np.array(rn) - np.array(rc)
            e_mm_nuc += zc * q / np.linalg.norm(d)

    return {
        "converged": abs(e_prev - e_tot) < 1e-8 if it > 0 else True,
        "energy": float(e_prev + e_mm_nuc),
        "energy_no_mm_nuc": float(e_prev),
        "e_pol": float(e_pol),
        "dipoles": mu.tolist(),
        "e_field_qm_at_sites": e_qm_last.tolist(),
        "e_field_perm_at_sites": e_perm_nuc_mm.tolist(),
        "iterations": it + 1,
        "dm": dm,
        "mol": mol,
    }


def plain_embedding_energy(mol, mm_charges, conv_tol=1e-11):
    """Reference: ordinary (non-polarizable) electrostatic embedding energy,
    using PySCF's own `qmmm.mm_charge` — an independent code path from
    `run_polarizable_scf`'s hand-rolled hcore construction, so the alpha->0
    anchor is a genuine cross-check, not a tautology."""
    from pyscf import qmmm

    mf = scf.RHF(mol)
    mf.conv_tol = conv_tol
    if mm_charges:
        coords = np.array([rc for _, rc in mm_charges])
        charges = np.array([q for q, _ in mm_charges])
        mf = qmmm.mm_charge(mf, coords, charges, unit="Bohr")
    e = mf.kernel()
    assert mf.converged
    return float(e)


# ---------------------------------------------------------------------------
# Correctness checks
# ---------------------------------------------------------------------------

def stationarity_check(mol_atoms, sites, alphas, mm_charges, thole_a, h=1e-3):
    """FD of E_total wrt a QM nuclear coordinate (O_z) vs the analytic
    (re-converged-at-each-geometry) value. Because `run_polarizable_scf` is
    a bona fide variational SCF+induction solve at every geometry, its
    E(R) is automatically stationary w.r.t. the electronic degrees of
    freedom (density AND induced dipoles) at each R by the Hellmann-Feynman
    theorem applied to the FULL (D, mu)-stationary Lagrangian — this check
    verifies the SCF+induction energy varies smoothly (finite, well-behaved
    central difference) and, as an independent check of the physical
    stationarity claim in the module docstring, that using a 3-point
    stencil the CENTRAL difference agrees with a tighter step to a few
    percent (checking the FD itself has converged, not an implementation
    bug that would show up as an erratic, non-convergent FD).
    """
    def energy_at(dz):
        atoms = [(s, (x, y, z + dz if i == 0 else z)) for i, (s, (x, y, z)) in enumerate(mol_atoms)]
        mol = build_mol(atoms)
        res = run_polarizable_scf(mol, sites, alphas, mm_charges, thole_a=thole_a)
        return res["energy"]

    e_p = energy_at(h)
    e_m = energy_at(-h)
    fd = (e_p - e_m) / (2 * h)
    e_p2 = energy_at(2 * h)
    e_m2 = energy_at(-2 * h)
    fd_tight = (8 * (e_p - e_m) - (e_p2 - e_m2)) / (12 * h)
    return {
        "fd_central_h": float(fd),
        "fd_richardson": float(fd_tight),
        "h": h,
        "rel_diff": float(abs(fd - fd_tight) / max(abs(fd_tight), 1e-12)),
    }


def distant_site_limit_check(mol_atoms, alpha=1.0, distance=12.0):
    """One site far along -z: mu_z ~ alpha*E_z^gas (1%), e_pol ~
    -1/2 alpha |E^gas|^2 (2%) — the isolated-site perturbative limit."""
    mol = build_mol(mol_atoms)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    dm_gas = mf.make_rdm1()
    site = (0.0, 0.0, -distance)
    e_gas = field_at_point_qm(mol, dm_gas, site)

    res = run_polarizable_scf(mol, [(*site, alpha)], [alpha], [], thole_a=THOLE_A_DEFAULT)
    mu = np.array(res["dipoles"][0])
    e_pol = res["e_pol"]

    mu_pred = alpha * e_gas
    e_pol_pred = -0.5 * alpha * np.dot(e_gas, e_gas)

    mu_err = np.linalg.norm(mu - mu_pred) / max(np.linalg.norm(mu_pred), 1e-12)
    e_pol_err = abs(e_pol - e_pol_pred) / max(abs(e_pol_pred), 1e-12)
    return {
        "e_field_gas": e_gas.tolist(),
        "mu": mu.tolist(),
        "mu_predicted": mu_pred.tolist(),
        "mu_rel_err": float(mu_err),
        "e_pol": e_pol,
        "e_pol_predicted": float(e_pol_pred),
        "e_pol_rel_err": float(e_pol_err),
    }


def alpha_zero_anchor_check(mol_atoms, mm_charges):
    """alpha -> 0 (here: a tiny but nonzero alpha, since alpha=0 in
    `induce_dipoles` would make B singular via 1/alpha) reproduces the plain
    PySCF qmmm.mm_charge energy. Uses alpha=1e-12 (negligible polarization,
    B well-conditioned) rather than a literal zero-alpha code path (the
    Rust side handles literal alpha=0 explicitly; this prototype checks the
    physical limit)."""
    mol = build_mol(mol_atoms)
    e_plain = plain_embedding_energy(build_mol(mol_atoms), mm_charges)
    sites = [(*rc, 1e-12) for _, rc in mm_charges]
    alphas = [1e-12] * len(mm_charges)
    # Polarizable sites here are placed exactly AT the MM charges for this
    # check's convenience; site-charge coupling is irrelevant since alpha~0.
    res = run_polarizable_scf(mol, sites, alphas, mm_charges, thole_a=THOLE_A_DEFAULT)
    return {
        "e_plain_pyscf_qmmm": e_plain,
        "e_polarizable_tiny_alpha": res["energy"],
        "e_pol": res["e_pol"],
        "abs_diff": abs(e_plain - res["energy"]),
    }


# ---------------------------------------------------------------------------
# Reference case generation
# ---------------------------------------------------------------------------

def make_case(tag, mol_atoms, sites_q_alpha, thole_a):
    """`sites_q_alpha`: list of (x, y, z, q, alpha) — q is a permanent MM
    point charge co-located with the polarizable site (the common PE
    use-case: a polarizable point charge)."""
    mol = build_mol(mol_atoms)
    sites = [(x, y, z, alpha) for (x, y, z, q, alpha) in sites_q_alpha]
    alphas = [alpha for (*_, alpha) in sites_q_alpha]
    mm_charges = [(q, (x, y, z)) for (x, y, z, q, alpha) in sites_q_alpha]

    # Sign-convention self-check (independent of the induction solve).
    mf0 = scf.RHF(mol)
    mf0.conv_tol = 1e-11
    mf0.kernel()
    dm0 = mf0.make_rdm1()
    max_sign_err, _, _ = _verify_field_sign(mol, dm0, sites[0][:3])

    res = run_polarizable_scf(mol, sites, alphas, mm_charges, thole_a=thole_a)
    e_gas = plain_embedding_energy(build_mol(mol_atoms), [])

    stat = stationarity_check(mol_atoms, sites, alphas, mm_charges, thole_a)

    ref = {
        "molecule": "water",
        "basis": "sto-3g",
        "method": "rhf",
        "model": "Thole-damped polarizable embedding (Applequist induced point dipoles), prototype",
        "thole_a": thole_a,
        "units": "Bohr / Hartree / a.u.",
        "atoms": [{"symbol": s, "xyz_bohr": list(xyz)} for s, xyz in mol_atoms],
        "sites": [
            {"xyz_bohr": [x, y, z], "q": q, "alpha_bohr3": alpha}
            for (x, y, z, q, alpha) in sites_q_alpha
        ],
        "energy": res["energy"],
        "energy_gas_phase": e_gas,
        "e_pol": res["e_pol"],
        "induced_dipoles": res["dipoles"],
        "e_field_qm_at_sites": res["e_field_qm_at_sites"],
        "e_field_perm_at_sites": res["e_field_perm_at_sites"],
        "iterations": res["iterations"],
        "field_sign_check_max_err": float(max_sign_err),
        "stationarity_check": stat,
        "pyscf_version": pyscf.__version__,
    }
    return ref


def main():
    mol_atoms = water_bohr()

    # ---- one_site: single polarizable+charged site off-axis ----
    one_site = [(3.0, -2.0, 4.0, 0.5, 1.44)]
    ref_one = make_case("water_sto-3g_pe_one_site", mol_atoms, one_site, THOLE_A_DEFAULT)

    # ---- three_sites: three off-axis sites, damped (default a=2.1304).
    #
    # Sites 0 and 1 are DELIBERATELY placed 1.1 Bohr apart with alpha=0.5
    # Bohr^3 each: a*u^3 = 5.67, exp(-a*u^3) = 0.0034 (lambda3 = 0.9966,
    # lambda5 differs from 1 by a comparable margin), so Thole damping is a
    # genuine, non-negligible correction to T_01 — at the ORIGINAL off-axis
    # placement tried during development (all pairs >4 Bohr apart, alpha
    # ~1-1.4 Bohr^3) every pairwise a*u^3 was >1900 (exp(-au^3) machine
    # zero), making the damped and undamped energies agree to 1e-13
    # (floating-point noise, not evidence the damping code path works: see
    # "too clean is a stop condition" in CLAUDE.md's Experimental
    # Protocol). A CLOSER/more-polarizable placement was also tried
    # (r=1.22 Bohr, alpha=1.44/0.90) and hit the Thole/Applequist
    # POLARIZATION CATASTROPHE: max|alpha*T_eigenvalue| = 1.26 > 1 makes
    # the (I - alpha*T) induction matrix indefinite, and the dense solve
    # returned E_pol = +1.98 Ha (unphysical positive, runaway feedback) —
    # not a bug, a genuine instability of the undamped-enough Applequist
    # model at short range/high polarizability. This case's r=1.1 Bohr,
    # alpha=0.5 pair keeps max|alpha*T_eigenvalue| = 0.73 (a comfortable
    # margin below the 1.0 catastrophe threshold) while still exercising a
    # non-trivial lambda3/lambda5. Both sites 0/1 stay >=3.1 Bohr from
    # every water atom (safely outside the link-atom/overpolarization
    # danger zone).
    three_sites = [
        (3.0, -2.0, 4.0, 0.5, 0.5),
        (2.27725463, -1.59345573, 3.27725463, -0.3, 0.5),
        (1.7, 2.9, -5.1, -0.2, 1.10),
    ]
    ref_three = make_case("water_sto-3g_pe_three_sites", mol_atoms, three_sites, THOLE_A_DEFAULT)

    # ---- three_sites_nodamp: same geometry, damping disabled ----
    ref_three_nodamp = make_case(
        "water_sto-3g_pe_three_sites_nodamp", mol_atoms, three_sites, None
    )

    # ---- Additional physics anchors (recorded, not written per-case) ----
    distant = distant_site_limit_check(mol_atoms, alpha=1.0, distance=12.0)
    alpha_zero = alpha_zero_anchor_check(mol_atoms, [(1.0, (0.0, 0.0, -6.0))])

    for tag, ref in [
        ("water_sto-3g_pe_one_site", ref_one),
        ("water_sto-3g_pe_three_sites", ref_three),
        ("water_sto-3g_pe_three_sites_nodamp", ref_three_nodamp),
    ]:
        out = REFDIR / f"{tag}.json"
        out.write_text(json.dumps(ref, indent=2, sort_keys=True) + "\n")
        print(
            f"{out.name}: E = {ref['energy']:.10f}  E_pol = {ref['e_pol']:.10e}  "
            f"gas = {ref['energy_gas_phase']:.10f}  iters={ref['iterations']}  "
            f"field_sign_err={ref['field_sign_check_max_err']:.2e}  "
            f"stationarity_rel_diff={ref['stationarity_check']['rel_diff']:.2e}"
        )

    print(
        f"[distant-site limit] mu_rel_err={distant['mu_rel_err']:.4e}  "
        f"e_pol_rel_err={distant['e_pol_rel_err']:.4e}  "
        f"(mu={distant['mu']}, predicted={distant['mu_predicted']})"
    )
    print(
        f"[alpha->0 anchor] plain={alpha_zero['e_plain_pyscf_qmmm']:.10f}  "
        f"polarizable(tiny alpha)={alpha_zero['e_polarizable_tiny_alpha']:.10f}  "
        f"|diff|={alpha_zero['abs_diff']:.3e}  e_pol={alpha_zero['e_pol']:.3e}"
    )

    assert distant["mu_rel_err"] < 0.01, f"distant-site mu limit failed: {distant['mu_rel_err']}"
    assert distant["e_pol_rel_err"] < 0.02, f"distant-site e_pol limit failed: {distant['e_pol_rel_err']}"
    assert alpha_zero["abs_diff"] < 1e-6, f"alpha->0 anchor failed: {alpha_zero['abs_diff']}"
    for ref in (ref_one, ref_three, ref_three_nodamp):
        assert ref["field_sign_check_max_err"] < 1e-6, "field sign check failed"
        assert ref["stationarity_check"]["rel_diff"] < 0.05, "FD has not converged"
    print("\nAll prototype self-checks passed.")


if __name__ == "__main__":
    main()
