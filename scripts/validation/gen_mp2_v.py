"""PySCF/numpy references for the VALIDATION row "attMP2 + VV10 (MP2-V)" (#273).

Consumer: crates/ferric-mp2/tests/validation_mp2_v.rs
Output:   testdata/reference/validation/mp2_v/<system>_<basis>.json

WHAT FERRIC COMPUTES (read from crates/ferric-mp2/src/att_vv10.rs and
crates/ferric-dft/src/vv10.rs, not assumed):

    E_MP2-V = E_HF + E_c^att(r0) + E_nl[rho_HF; b, C, r0]

* E_nl is evaluated post-HF on the converged RHF density, on ferric's NLC grid
  `default_nlc_grid()` = 50 radial (Treutler-Ahlrichs M4) x 50 Lebedev,
  unpruned, Becke partition with ferric's Bragg-Slater size adjustment.
  Points with rho < 1e-8 are dropped (`RHO_THRESH`). The energy is
  E_nl = sum_i w_i rho_i [beta + 1/2 sum_p w_p rho_p Phi_ip f(R_ip)], and the
  damping f(R) = 1 - terfc(R, r0)^2 multiplies the pair kernel only, never beta
  (paper Eq. 11). terfc(R, r0) = 1 - 1/2[erf((R-r0)/(r0 sqrt2)) +
  erf((R+r0)/(r0 sqrt2))], r0 in Bohr. The pair sum includes the self pair
  (R = 0), where f = 0 for every r0 > 0.
* E_c^att is RI-MP2 with the attenuated operator on BOTH (P|mu nu) and (P|Q).
  The published operator is terfc(r, r0)/r. The erfc control is
  erfc(omega r)/r with omega = 1/(r0 sqrt2) in Bohr^-1.

WHAT THIS SCRIPT WRITES, AND WHY EACH PIECE IS INDEPENDENT (OR NOT)

1. PySCF exact-J/K RHF; the density matrix permuted into ferric AO order, plus
   the AO overlap in ferric order (the Rust test checks the permutation against
   ferric's own overlap before using the density).
2. Ferric's NLC grid rebuilt from PySCF primitives (gen_properties.py's
   `atom_centred_grid` + `becke_partition`). Recorded: point count, sum of
   weights, and the integrated electron count of the density on it, so a grid
   mismatch is localised before any VV10 number is compared.
3. UNDAMPED E_nl, (b, C) = (11.0, 0.0089): PySCF
   `dft.numint._vv10nlc(rho, coords, rho, w, coords, (b, C))` on that grid and
   density. ferric's kernel is a port of this function, so this anchor checks
   the density/grid/threshold/summation, NOT the VV10 formula.
4. The VV10 formula is checked by a numpy kernel written from Vydrov & Van
   Voorhis, JCP 133, 244103 (2010): omega_p^2 = 4 pi rho, omega_g^2 =
   C |grad rho / rho|^4, omega_0 = sqrt(omega_g^2 + omega_p^2/3), kappa =
   b v_F^2 / omega_p with v_F = (3 pi^2 rho)^(1/3), Phi = -3 / (2 g g' (g+g')),
   g = omega_0 R^2 + kappa, beta = (1/32)(3/b^2)^(3/4). Dense pair sum, no
   cutoff. With f = 1 it must equal `_vv10nlc` (asserted before writing).
5. DAMPED E_nl from the numpy kernel at r0 = 1.00 A (b = 11.0) and at the
   Table 1 point r0 = 0.85 A (b = 8.0). PySCF has no damped VV10; this is the
   only reference for the damping.
6. The r0 -> 0 limit: the self-pair term S = sum_i w_i rho_i * 1/2 * w_i rho_i
   * Phi_ii is the only pair that stays damped as r0 -> 0 (f(0) = 0 for every
   r0). So damped(r0 << min pair distance) == undamped - S exactly. Recorded
   with the r0 at which it holds.
7. The erfc-attenuator control: numpy RI-MP2 on PySCF integrals under
   `with_range_coulomb(-omega)` on the orbital AND aux Moles (the
   gen_attenuated_mp2.py construction), omega = 1/(r0 sqrt2), frozen core as
   the paper (1 per first-row heavy atom). Assembly checked against PySCF
   DFMP2 (Coulomb, same frozen core) before writing.
8. The terfc-attenuated E_c is NOT computed here. Its independent reference is
   the row-119 terfc generator (#275), which this row reuses rather than
   duplicating; it was not on main when this script was written.

Run (inside the box-wide slot):
    scripts/validation/run_slot.sh -- \\
        /home/matt/qc/ferric/.venv/bin/python scripts/validation/gen_mp2_v.py [system ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402
import gen_properties as gp  # noqa: E402

ROW = "mp2_v"
ROW_NAME = "attMP2 + VV10 (MP2-V)"
GENERATOR = "scripts/validation/gen_mp2_v.py"

# ferric att_vv10.rs BOHR_PER_ANG (the r0 conversion ferric uses).
BOHR_PER_ANG = 1.8897259886

VV10_C = 0.0089
# (r0 / A, b): Table 1 of Goldey, Belzunces & Head-Gordon, JCTC 11, 4159 (2015).
DAMPED_POINTS = ((1.00, 11.0), (0.85, 8.0))
RHO_THRESH = 1e-8
NLC_GRID = (50, 50)

# (system, xyz path, basis, aux basis, frozen core)
CASES = (
    ("h2o", common.MOL_DIR / "h2o.xyz", "cc-pvdz", "cc-pvdz-ri", 1),
    ("h2o", common.MOL_DIR / "h2o.xyz", "aug-cc-pvdz", "aug-cc-pvdz-rifit", 1),
    ("nh3", common.MOL_DIR / "nh3.xyz", "cc-pvdz", "cc-pvdz-ri", 1),
    (
        "water_dimer",
        common.ROOT / "testdata" / "molecules" / "s22" / "water_dimer.xyz",
        "aug-cc-pvdz",
        "aug-cc-pvdz-rifit",
        2,
    ),
)

CONV_TOL = 1e-12
CONV_TOL_GRAD = 1e-9
CHUNK = 400


# ---------------------------------------------------------------------------
# VV10 from the paper
# ---------------------------------------------------------------------------


def vv10_point_quantities(rho, grad, b, c):
    """omega_0, kappa per point (VV10 paper, Eqs. 2-4 and 9-10)."""
    import numpy as np

    omega_p2 = 4.0 * np.pi * rho
    omega_g2 = c * (np.sum(grad * grad, axis=0) / rho**2) ** 2
    omega0 = np.sqrt(omega_g2 + omega_p2 / 3.0)
    v_f = (3.0 * np.pi**2 * rho) ** (1.0 / 3.0)
    kappa = b * v_f**2 / np.sqrt(omega_p2)
    return omega0, kappa


def terfc(r, r0):
    import numpy as np
    from scipy.special import erf

    s = r0 * np.sqrt(2.0)
    return 1.0 - 0.5 * (erf((r - r0) / s) + erf((r + r0) / s))


def vv10_paper_energy(pts, w, rho, grad, b, c, r0_bohr=None):
    """E_nl = sum_i w_i rho_i [beta + 1/2 sum_p w_p rho_p Phi_ip f(R_ip)].

    Dense, no cutoff, self pair included. `r0_bohr=None` is the undamped kernel
    (f = 1). Points below RHO_THRESH are dropped, as in ferric and PySCF.
    Returns (E_nl, self_pair_term) where self_pair_term is the p = i part of the
    pair sum (already multiplied by the f(0) of this call)."""
    import numpy as np

    act = rho >= RHO_THRESH
    x, wa, ra, ga = pts[act], w[act], rho[act], grad[:, act]
    omega0, kappa = vv10_point_quantities(ra, ga, b, c)
    beta = (3.0 / b**2) ** 0.75 / 32.0
    rw = ra * wa
    n = len(ra)
    pair = np.zeros(n)
    for i0 in range(0, n, CHUNK):
        i1 = min(i0 + CHUNK, n)
        d = x[i0:i1, None, :] - x[None, :, :]
        r2 = np.einsum("ijk,ijk->ij", d, d)
        gi = omega0[i0:i1, None] * r2 + kappa[i0:i1, None]
        gp_ = omega0[None, :] * r2 + kappa[None, :]
        phi = -1.5 / (gi * gp_ * (gi + gp_))
        if r0_bohr is not None:
            phi = phi * (1.0 - terfc(np.sqrt(r2), r0_bohr) ** 2)
        pair[i0:i1] = phi @ rw
    e = float(np.sum(rw * (beta + 0.5 * pair)))
    phi_self = -1.5 / (2.0 * kappa**3)
    f0 = 1.0 if r0_bohr is None else 1.0 - terfc(0.0, r0_bohr) ** 2
    self_term = float(np.sum(rw * 0.5 * rw * phi_self * f0))
    return e, self_term


def min_pair_distance(pts_active):
    import numpy as np

    best = np.inf
    n = len(pts_active)
    for i0 in range(0, n, CHUNK):
        i1 = min(i0 + CHUNK, n)
        d = pts_active[i0:i1, None, :] - pts_active[None, :, :]
        r2 = np.einsum("ijk,ijk->ij", d, d)
        idx = np.arange(i0, i1)
        r2[idx - i0, idx] = np.inf
        best = min(best, float(np.sqrt(r2.min())))
    return best


def max_pair_distance(pts_active):
    import numpy as np

    best = 0.0
    n = len(pts_active)
    for i0 in range(0, n, CHUNK):
        i1 = min(i0 + CHUNK, n)
        d = pts_active[i0:i1, None, :] - pts_active[None, :, :]
        best = max(best, float(np.sqrt(np.einsum("ijk,ijk->ij", d, d).max())))
    return best


# ---------------------------------------------------------------------------
# Attenuated RI-MP2 with frozen core (gen_attenuated_mp2.py construction)
# ---------------------------------------------------------------------------


def ints(mol, auxmol, omega):
    from pyscf import df

    if omega == 0.0:
        return (
            df.incore.aux_e2(mol, auxmol, intor="int3c2e", aosym="s1"),
            auxmol.intor("int2c2e"),
        )
    with mol.with_range_coulomb(-omega), auxmol.with_range_coulomb(-omega):
        v3 = df.incore.aux_e2(mol, auxmol, intor="int3c2e", aosym="s1")
        v2 = auxmol.intor("int2c2e")
    return v3, v2


def ri_mp2_fc(mf, v3, v2, nfc):
    import numpy as np
    import scipy.linalg

    nocc = int(np.count_nonzero(mf.mo_occ > 0))
    c = mf.mo_coeff
    e = mf.mo_energy
    co, cv = c[:, nfc:nocc], c[:, nocc:]
    no = nocc - nfc
    iaP = np.einsum("mnP,mi,na->iaP", v3, co, cv, optimize=True)
    naux = v2.shape[0]
    L = scipy.linalg.cholesky(v2, lower=True)
    B = scipy.linalg.solve_triangular(L, iaP.reshape(-1, naux).T, lower=True)
    nvir = cv.shape[1]
    B = B.reshape(naux, no, nvir)
    eo, ev = e[nfc:nocc], e[nocc:]
    e_os = 0.0
    e_ss = 0.0
    for i in range(no):
        for j in range(no):
            v = np.einsum("Pa,Pb->ab", B[:, i, :], B[:, j, :])
            d = eo[i] + eo[j] - ev[:, None] - ev[None, :]
            e_os += float(np.sum(v * v / d))
            e_ss += float(np.sum(v * (v - v.T) / d))
    return {"e_os": e_os, "e_ss": e_ss, "e_corr": e_os + e_ss}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_case(system, xyz, basis, auxbasis_name, nfc):
    import numpy as np
    import pyscf
    from pyscf import df, gto, scf
    from pyscf.dft import numint
    from pyscf.mp import dfmp2

    symbols, coords = common.read_xyz(xyz)
    mol = common.build_pyscf_mol(xyz, basis)
    basis_check = common.check_basis_like_for_like(mol, basis, symbols)
    auxbasis, auxcart = common.pyscf_basis(auxbasis_name, symbols)
    assert not auxcart
    auxmol = gto.M(
        atom=common.pyscf_atom_bohr(symbols, coords),
        unit="Bohr",
        basis=auxbasis,
        cart=False,
        verbose=0,
    )
    assert auxmol.nao_nr() == common.ferric_nao(auxbasis_name, symbols)

    mf = scf.RHF(mol)
    mf.conv_tol = CONV_TOL
    mf.conv_tol_grad = CONV_TOL_GRAD
    mf.max_cycle = 200
    mf.verbose = 0
    mf.kernel()
    assert mf.converged, f"{system}/{basis}: RHF did not converge"
    dm = mf.make_rdm1()
    perm = gp.ferric_ao_permutation(mol, basis, symbols)

    # --- grid (ferric's NLC grid) and density on it -------------------------
    zs = [int(mol.atom_charge(i)) for i in range(mol.natm)]
    xyz_b = np.asarray(mol.atom_coords(unit="Bohr"))
    home, pts, w_rl = gp.atom_centred_grid(zs, xyz_b, *NLC_GRID)
    wb = gp.becke_partition(zs, xyz_b, pts)[home, np.arange(len(pts))]
    w = w_rl * wb
    ao = numint.eval_ao(mol, pts, deriv=1)
    rho4 = numint.eval_rho(mol, ao, dm, xctype="GGA")
    rho, grad = rho4[0], rho4[1:4]
    n_elec_grid = float(np.dot(w, rho))
    act = rho >= RHO_THRESH
    n_active = int(np.count_nonzero(act))
    dmin = min_pair_distance(pts[act])
    dmax = max_pair_distance(pts[act])

    # --- undamped anchors per b -------------------------------------------
    vv = {}
    for r0_ang, b in DAMPED_POINTS:
        exc, _ = numint._vv10nlc(rho4, pts, rho4, w, pts, (b, VV10_C))
        e_pyscf = float(np.sum(w * rho * exc))
        e_np, self_undamped = vv10_paper_energy(pts, w, rho, grad, b, VV10_C)
        d_np = abs(e_np - e_pyscf)
        assert d_np < 1e-10, f"{system}: numpy VV10 vs _vv10nlc {d_np:.2e} (b={b})"
        r0_b = r0_ang * BOHR_PER_ANG
        e_damped, _ = vv10_paper_energy(pts, w, rho, grad, b, VV10_C, r0_b)
        assert abs(e_damped - e_np) > 1e-5, "damping not live"
        # r0 -> 0 limit: only the self pair stays damped.
        r0_tiny = dmin / 50.0
        e_tiny, _ = vv10_paper_energy(pts, w, rho, grad, b, VV10_C, r0_tiny)
        d_lim = abs(e_tiny - (e_np - self_undamped))
        assert d_lim < 1e-12, f"{system}: r0->0 limit off by {d_lim:.2e}"
        vv[f"b={b}"] = {
            "b": b,
            "c": VV10_C,
            "undamped_pyscf_vv10nlc": e_pyscf,
            "undamped_numpy_paper": e_np,
            "numpy_vs_pyscf_abs": d_np,
            "self_pair_term": self_undamped,
            "r0_to_zero": {
                "r0_bohr": r0_tiny,
                "e_nl_numpy_damped": e_tiny,
                "undamped_minus_self_pair": e_np - self_undamped,
                "abs_diff": d_lim,
            },
            "damped": {
                "r0_angstrom": r0_ang,
                "r0_bohr": r0_b,
                "e_nl_numpy": e_damped,
                "damped_minus_undamped": e_damped - e_np,
            },
        }
        print(
            f"{system:11s} {basis:12s} b={b:5.1f} E_nl undamped pyscf={e_pyscf:+.12f} "
            f"numpy={e_np:+.12f} |d|={d_np:.1e}  damped(r0={r0_ang} A)={e_damped:+.12f} "
            f"shift={e_damped - e_np:+.3e}  r0->0 |d|={d_lim:.1e}"
        )

    # --- erfc control ------------------------------------------------------
    r0_b = 1.00 * BOHR_PER_ANG
    omega = 1.0 / (r0_b * np.sqrt(2.0))
    v3c, v2c = ints(mol, auxmol, 0.0)
    coul = ri_mp2_fc(mf, v3c, v2c, nfc)
    pt = dfmp2.DFMP2(mf, frozen=nfc)
    pt.with_df = df.DF(mol)
    pt.with_df.auxbasis = auxbasis
    pt.kernel()
    d_assembly = abs(coul["e_corr"] - float(pt.e_corr))
    assert d_assembly < 1e-10, f"numpy DF-MP2 vs PySCF DFMP2 {d_assembly:.2e}"
    v3, v2 = ints(mol, auxmol, omega)
    erfc = ri_mp2_fc(mf, v3, v2, nfc)
    print(
        f"{system:11s} {basis:12s} E_RHF={mf.e_tot:.12f} N_grid={n_elec_grid:.10f} "
        f"npts={len(pts)} active={n_active} dmax={dmax:.2f} "
        f"E_c(erfc, w={omega:.6f}, fc={nfc})={erfc['e_corr']:.12f} "
        f"E_c(coul)={coul['e_corr']:.12f} DFMP2|d|={d_assembly:.1e}"
    )

    s_ferric = gp.to_ferric_order(mol.intor("int1e_ovlp"), perm)
    d_ferric = gp.to_ferric_order(dm, perm)
    return {
        "row": ROW_NAME,
        "system": system,
        "basis": basis,
        "auxbasis": auxbasis_name,
        "charge": 0,
        "multiplicity": 1,
        "nao": mol.nao_nr(),
        "naux": auxmol.nao_nr(),
        "frozen_core": nfc,
        "nuclear_repulsion": float(mol.energy_nuc()),
        "rhf_energy": float(mf.e_tot),
        "overlap_ferric_order": gp.mat_json(s_ferric),
        "density_ferric_order": gp.mat_json(d_ferric),
        "nlc_grid": {
            "n_radial": NLC_GRID[0],
            "n_angular": NLC_GRID[1],
            "prune": None,
            "npts": int(len(pts)),
            "sum_weights": float(np.sum(w)),
            "n_electrons_on_grid": n_elec_grid,
            "n_active": n_active,
            "rho_thresh": RHO_THRESH,
            "min_active_pair_distance_bohr": dmin,
            "max_active_pair_distance_bohr": dmax,
        },
        "vv10": vv,
        "mp2_erfc_control": {
            "r0_angstrom": 1.00,
            "omega_bohr_inv": float(omega),
            "frozen_core": nfc,
            "e_os": erfc["e_os"],
            "e_ss": erfc["e_ss"],
            "e_corr": erfc["e_corr"],
            "coulomb_e_corr": coul["e_corr"],
            "numpy_vs_pyscf_dfmp2_coulomb_abs": d_assembly,
        },
        "mp2_terfc": "not computed here: the independent terfc reference is the "
        "row-119 generator (#275), not on main when this file was written",
        "provenance": common.provenance(
            code="PySCF (RHF, AO values, _vv10nlc, integrals, DFMP2 anchor) + numpy "
            "(VV10 from Vydrov & Van Voorhis 2010, damped per MP2-V Eq. 11; RI-MP2 assembly)",
            version=pyscf.__version__,
            keywords={
                "rhf": "scf.RHF, exact Coulomb J/K",
                "nlc_grid": "TA-M4 50 radial x Lebedev 50, unpruned, ferric Becke partition",
                "vv10": f"C={VV10_C}, b per Table 1, rho >= {RHO_THRESH}, dense pair sum",
                "damping": "1 - terfc(R, r0)^2 on the pair kernel only",
                "erfc_control": "erfc(omega r)/r on int3c2e and int2c2e, omega=1/(r0 sqrt2)",
                "numpy": np.__version__,
            },
            basis_name=basis,
            xyz_path=xyz,
            coords_bohr=coords,
            symbols=symbols,
            grid={"nlc": list(NLC_GRID), "prune": None},
            aux=auxbasis_name,
            frozen_core=nfc,
            scf_conv={"conv_tol": CONV_TOL, "conv_tol_grad": CONV_TOL_GRAD},
            stability=None,
            generator=GENERATOR,
            extra={"basis_self_check": basis_check},
        ),
    }


def main() -> int:
    only = set(sys.argv[1:])
    for system, xyz, basis, aux, nfc in CASES:
        if only and system not in only:
            continue
        payload = run_case(system, xyz, basis, aux, nfc)
        path = common.write_reference(ROW, system, basis, payload)
        print(f"wrote {path.relative_to(common.ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
