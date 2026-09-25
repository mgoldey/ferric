"""PySCF references for the QM/MM validation rows (validation tier W2).

Rows (VALIDATION.md / plan §5.4):
    "QM/MM"                                 RHF point-charge embedding
    "Smeared charges"                       Gaussian-smeared MM charges
    "KS QM/MM"                              RKS (B3LYP, PBE) point-charge embedding
    "External potential in MP2 gradients"   RI-MP2 gradient in a point-charge field

Consumers:
    crates/ferric-scf/tests/validation_qmmm.rs              (first three rows)
    crates/ferric-mp2/tests/validation_qmmm_mp2_gradient.rs (MP2 row)
Output:
    testdata/reference/validation/qmmm/<case>_<basis>.json
    testdata/molecules/validation/ch3oh_tip3p_shell.json    (the 501-site shell)

MODEL (both codes): fixed MM charges folded into the one-electron Hamiltonian
plus the classical charge-nuclear energy; no MM-MM term, no van der Waals.
PySCF side is `pyscf.qmmm.mm_charge` (point, and `radii=` Gaussian).

SMEARED-CHARGE UNIT CONVENTION (read from both sources, and checked below):
  * PySCF 2.13 `qmmm/mm_mole.py::create_mm_mol`: `radii` is in the SAME unit
    as the coordinates (`unit=`), converted to Bohr when unit is Angstrom
    (`radii / param.BOHR`), then `zeta = 1 / radii**2`. The charge density is
    q (zeta/pi)^{3/2} exp(-zeta r^2) (`fakemol_for_charges`, normalized), and
    the nuclear term is q Z erf(sqrt(zeta) r)/r (`itrf.py::energy_nuc`).
    NOTE: `mm_charge`'s default unit is ANGSTROM, so a radius passed without
    `unit="Bohr"` is read in Angstrom.
  * ferric `SmearedCharge.width` / `QmmmAtom.width` is in BOHR and
    `zeta = 1 / width**2`, potential q erf(r/width)/r
    (`crates/ferric-core/src/external_potential.rs`).
  => ferric width [Bohr] == PySCF radius with unit="Bohr" == PySCF radius
     [Angstrom] * (1/0.52917721092) with unit="Angstrom". Every smeared
     reference records `unit_convention_check`: the energy with coords AND
     radii handed to PySCF in Angstrom must equal the Bohr run to 1e-10
     (refused otherwise), and the "Bohr number passed as Angstrom" trap must
     move the energy (recorded, refused if it does not).

SYSTEMS
  * water (testdata/molecules/validation/h2o.xyz) + 10 point charges: two
    TIP3P-geometry waters H-bonded to the QM water (one accepting from H1, one
    donating to the O lone pair) plus +1/-1 ions and a +-0.3 pair, net zero,
    no symmetry (every gradient component is live). cc-pVDZ and aug-cc-pVDZ.
  * methanol (ch3oh.xyz) + 501 TIP3P charges (167 waters; O -0.834,
    H +0.417) in a random shell, numpy default_rng(SHELL_SEED), rejection-
    sampled (parameters in SHELL_PARAMS). The shell is committed to
    testdata/molecules/validation/ch3oh_tip3p_shell.json; rerunning this script
    regenerates it from the seed and REFUSES if it differs from the committed
    file (catches an RNG/algorithm drift). 6-31G, RHF.
  * smeared: the water/10-charge set at cc-pVDZ with every site smeared at a
    uniform width 0.5, 1.0 and 2.0 Bohr (three references).
  * KS: methanol / def2-SVP + the 20 shell sites nearest to any QM atom,
    B3LYP and PBE, EXACT four-centre J/K on both sides (ferric
    df_j_aux = df_k_aux = Some("")), grid (75,110) unpruned, Becke partition
    with Becke-1988 radii adjustment (== ferric's default grid).
  * MP2 gradient: RHF (exact J/K) + DF-MP2 with ferric's cc-pvdz-ri aux,
    frozen=None, in the field. water / cc-pVDZ + 10 charges and methanol /
    6-31G + 20 charges. PySCF 2.13 has no analytic DF-MP2 gradient, so the
    reference is a 5-point central finite difference of the DF-MP2 TOTAL
    energy of the mm_charge-wrapped SCF, at two step sizes (h and 2h) that
    must agree to FD_AGREE (refused otherwise).

CONTROLS recorded per case (the Rust tests assert ferric MISSES what it must):
    energy_gas_phase             vacuum (the anti-vacuum-test control)
    control_scaled_101.energy    every MM charge x 1.01
    smeared: control_point_limit.energy (same sites as point charges) and
             control_widths.<w>.energy for the other two widths
    KS: the other functional's energy lives in the sibling file
    MP2: rhf_gradient (the SCF-only gradient in the field)

Run (light; the MP2 FDs are ~200 small DF-MP2 energies in total):
    scripts/validation/run_slot.sh --light -- \\
        /home/matt/qc/ferric/.venv/bin/python scripts/validation/gen_qmmm.py [group ...]
groups: point smeared ks mp2 (default all)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402

ROW = "qmmm"
GEN = "scripts/validation/gen_qmmm.py"
CONV_TOL = 1e-12
CONV_TOL_GRAD = 1e-9
KS_CONV_TOL = 1e-11
KS_CONV_TOL_GRAD = 1e-8
MAIN_GRID = (75, 110)
PYSCF_XC = {"b3lyp": "B3LYP", "pbe": "PBE,PBE"}
SMEAR_WIDTHS_BOHR = (0.5, 1.0, 2.0)
MP2_AUX = "cc-pvdz-ri"
H_FD = 2e-3  # Bohr
FD_AGREE = 5e-8  # |g(h) - g(2h)| max, Ha/Bohr
UNIT_CHECK_TOL = 1e-10
SCALE_CONTROL = 1.01

SHELL_SEED = 20260924
SHELL_PARAMS = {
    "n_waters": 167,
    "r_inner_angstrom": 3.0,
    "r_outer_angstrom": 12.0,
    "min_site_to_qm_angstrom": 2.0,
    "min_oo_angstrom": 2.8,
    "min_site_site_angstrom": 1.5,
    "tip3p_r_oh_angstrom": 0.9572,
    "tip3p_hoh_deg": 104.52,
    "q_o": -0.834,
    "q_h": 0.417,
    "center": "centroid of the methanol atoms",
    "sampling": "O uniform in shell volume, bisector and plane normal uniform on the sphere",
}
SHELL_PATH = common.MOL_DIR / "ch3oh_tip3p_shell.json"
KS_N_CHARGES = 20

A2B = common.ANGSTROM_TO_BOHR


# ---------------------------------------------------------------------------
# MM charge sets (built in Angstrom, stored in Bohr)
# ---------------------------------------------------------------------------


def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def tip3p_water(o, bisector, normal_hint):
    """O and two H positions (Angstrom) of a TIP3P-geometry water whose H-O-H
    bisector points along `bisector` and whose plane contains `bisector` and
    (the component orthogonal to it of) `normal_hint`'s cross product."""
    r = SHELL_PARAMS["tip3p_r_oh_angstrom"]
    half = np.radians(SHELL_PARAMS["tip3p_hoh_deg"]) / 2.0
    b = _unit(bisector)
    n = np.asarray(normal_hint, dtype=float)
    n = _unit(n - n.dot(b) * b)
    t = np.cross(n, b)
    o = np.asarray(o, dtype=float)
    return [
        o,
        o + r * (np.cos(half) * b + np.sin(half) * t),
        o + r * (np.cos(half) * b - np.sin(half) * t),
    ]


def water_charges_10(qm_ang):
    """The 10-site set around the QM water (Angstrom in, [(q, x, y, z)] Bohr out)."""
    o, h1, h2 = (np.asarray(p) for p in qm_ang)
    # Acceptor water: its O 1.95 A beyond H1 along O->H1, H's pointing away.
    u1 = _unit(h1 - o)
    acc = tip3p_water(h1 + 1.95 * u1, u1 + np.array([0.2, 0.0, 0.1]), [1.0, 0.3, 0.2])
    # Donor water: one H 1.95 A from O on the lone-pair side (opposite the
    # H-O-H bisector, tilted out of the molecular plane along x).
    bis = _unit(h1 + h2 - 2.0 * o)
    lp = _unit(-bis + np.array([0.45, 0.0, 0.0]))
    h_don = o + 1.95 * lp
    o_don = h_don + 0.9572 * _unit(lp + np.array([0.0, 0.25, 0.0]))
    # Place the donor O, then its second H by the TIP3P angle.
    b = _unit((h_don - o_don))  # O->H(donating) direction
    rot_axis = _unit(np.cross(b, [0.3, 1.0, 0.1]))
    ang = np.radians(SHELL_PARAMS["tip3p_hoh_deg"])
    # Rodrigues rotation of b about rot_axis by the H-O-H angle.
    b2 = (
        b * np.cos(ang)
        + np.cross(rot_axis, b) * np.sin(ang)
        + rot_axis * rot_axis.dot(b) * (1 - np.cos(ang))
    )
    don = [o_don, h_don, o_don + 0.9572 * b2]
    sites = [
        (SHELL_PARAMS["q_o"], acc[0]),
        (SHELL_PARAMS["q_h"], acc[1]),
        (SHELL_PARAMS["q_h"], acc[2]),
        (SHELL_PARAMS["q_o"], don[0]),
        (SHELL_PARAMS["q_h"], don[1]),
        (SHELL_PARAMS["q_h"], don[2]),
        (1.0, np.array([2.9, 1.3, -1.6])),
        (-1.0, np.array([-3.4, -1.1, 1.2])),
        (0.3, np.array([1.7, -4.6, 2.3])),
        (-0.3, np.array([-2.2, 3.9, -3.8])),
    ]
    return [(float(q), *(float(c) * A2B for c in p)) for q, p in sites]


def methanol_shell(qm_ang):
    """Deterministic 501-site TIP3P shell around methanol, [(q,x,y,z)] Bohr."""
    p = SHELL_PARAMS
    rng = np.random.default_rng(SHELL_SEED)
    qm = np.asarray(qm_ang)
    center = qm.mean(axis=0)
    waters = []
    attempts = 0
    while len(waters) < p["n_waters"]:
        attempts += 1
        if attempts > 200000:
            raise RuntimeError("shell rejection sampling did not finish")
        d = _unit(rng.normal(size=3))
        r3 = rng.uniform(p["r_inner_angstrom"] ** 3, p["r_outer_angstrom"] ** 3)
        o = center + np.cbrt(r3) * d
        w = tip3p_water(o, rng.normal(size=3), rng.normal(size=3))
        w = np.asarray(w)
        if (
            np.min(np.linalg.norm(w[:, None, :] - qm[None, :, :], axis=2))
            < p["min_site_to_qm_angstrom"]
        ):
            continue
        ok = True
        for other in waters:
            if np.linalg.norm(other[0] - w[0]) < p["min_oo_angstrom"]:
                ok = False
                break
            if (
                np.min(np.linalg.norm(other[:, None, :] - w[None, :, :], axis=2))
                < p["min_site_site_angstrom"]
            ):
                ok = False
                break
        if ok:
            waters.append(w)
    out = []
    for w in waters:
        for k, pos in enumerate(w):
            q = p["q_o"] if k == 0 else p["q_h"]
            out.append((float(q), *(float(c) * A2B for c in pos)))
    return out, attempts


def load_or_check_shell(qm_ang):
    sites, attempts = methanol_shell(qm_ang)
    payload = {
        "description": "TIP3P-charge solvent shell around testdata/molecules/validation/"
        "ch3oh.xyz; generated by scripts/validation/gen_qmmm.py",
        "seed": SHELL_SEED,
        "params": SHELL_PARAMS,
        "rng": f"numpy.random.default_rng (PCG64), numpy {np.__version__}",
        "attempts": attempts,
        "n_sites": len(sites),
        "net_charge": float(sum(s[0] for s in sites)),
        "units": "Bohr (ferric ANGSTROM_TO_BOHR = 1/0.52917721092)",
        "sites": [{"q": q, "xyz_bohr": [x, y, z]} for q, x, y, z in sites],
    }
    if SHELL_PATH.exists():
        old = json.loads(SHELL_PATH.read_text())
        old_sites = [(s["q"], *s["xyz_bohr"]) for s in old["sites"]]
        if (
            len(old_sites) != len(sites)
            or np.max(np.abs(np.array(old_sites) - np.array(sites))) > 1e-12
        ):
            raise RuntimeError(
                f"{SHELL_PATH}: regenerated shell differs from the committed one "
                "(RNG or algorithm drift) — refusing"
            )
        return [(s["q"], *s["xyz_bohr"]) for s in old["sites"]]
    SHELL_PATH.write_text(json.dumps(payload, indent=1) + "\n")
    return sites


def nearest_sites(sites, qm_bohr, n):
    qm = np.asarray(qm_bohr)
    d = [float(np.min(np.linalg.norm(qm - np.array(s[1:]), axis=1))) for s in sites]
    order = np.argsort(d, kind="stable")[:n]
    return [sites[i] for i in sorted(order)]


# ---------------------------------------------------------------------------
# PySCF runs
# ---------------------------------------------------------------------------


def _mm_arrays(mm):
    coords = np.array([[x, y, z] for _, x, y, z in mm])
    charges = np.array([q for q, _, _, _ in mm])
    return coords, charges


def _configure_ks(mf):
    from pyscf import dft

    mf.grids.atom_grid = MAIN_GRID
    mf.grids.prune = None
    mf.grids.radii_adjust = dft.radi.becke_atomic_radii_adjust
    mf.conv_tol = KS_CONV_TOL
    mf.conv_tol_grad = KS_CONV_TOL_GRAD
    return mf


def _mf(mol, xc=None):
    from pyscf import dft, scf

    if xc is None:
        mf = scf.RHF(mol)
        mf.conv_tol = CONV_TOL
        mf.conv_tol_grad = CONV_TOL_GRAD
    else:
        mf = _configure_ks(dft.RKS(mol, xc=PYSCF_XC[xc]))
    mf.max_cycle = 300
    mf.verbose = 0
    return mf


def embedded_energy(mol, mm, xc=None, radii=None, unit="Bohr", dm0=None):
    from pyscf import qmmm

    coords, charges = _mm_arrays(mm)
    kw = {} if radii is None else {"radii": np.asarray(radii, dtype=float)}
    mf = qmmm.mm_charge(_mf(mol, xc), coords, charges, unit=unit, **kw)
    e = mf.kernel(dm0=dm0)
    if not mf.converged:
        raise RuntimeError("embedded SCF did not converge")
    return float(e), mf


def dipole(mol, dm):
    ao = mol.intor_symmetric("int1e_r", comp=3)
    el = -np.einsum("xij,ji->x", ao, dm)
    nuc = np.einsum("i,ix->x", mol.atom_charges(), mol.atom_coords())
    return [float(v) for v in el + nuc]


def full_case(mol, mm, xc=None, radii=None):
    """Energy, gas-phase energy, dipole, QM gradient and MM gradient."""
    e, mf = embedded_energy(mol, mm, xc=xc, radii=radii)
    dm = mf.make_rdm1()
    g = mf.nuc_grad_method()
    out = {}
    if xc is not None:
        g.grid_response = False
        out["qm_gradient"] = np.asarray(g.kernel()).tolist()
        g2 = mf.nuc_grad_method()
        g2.grid_response = True
        out["qm_gradient_grid_response"] = np.asarray(g2.kernel()).tolist()
    else:
        out["qm_gradient"] = np.asarray(g.kernel()).tolist()
    out["mm_gradient"] = np.asarray(g.grad_hcore_mm(dm) + g.grad_nuc_mm()).tolist()
    gas = _mf(mol, xc)
    e0 = gas.kernel()
    if not gas.converged:
        raise RuntimeError("gas-phase SCF did not converge")
    out.update(
        {
            "energy": e,
            "energy_gas_phase": float(e0),
            "shift": e - float(e0),
            "dipole": dipole(mol, dm),
        }
    )
    return out, dm


def scaled_control(mol, mm, xc=None, radii=None, dm0=None):
    mm_s = [(q * SCALE_CONTROL, x, y, z) for q, x, y, z in mm]
    e, _ = embedded_energy(mol, mm_s, xc=xc, radii=radii, dm0=dm0)
    return {"scale": SCALE_CONTROL, "energy": e}


def mol_for(xyz, basis, coords_bohr=None):
    from pyscf import gto

    symbols, coords = common.read_xyz(xyz)
    if coords_bohr is not None:
        coords = coords_bohr
    bas, cart = common.pyscf_basis(basis, symbols)
    assert not cart, f"{basis}: Cartesian l>=2 shells"
    return gto.M(
        atom=common.pyscf_atom_bohr(symbols, coords),
        unit="Bohr",
        basis=bas,
        cart=False,
        verbose=0,
    )


def common_payload(
    row_name,
    system,
    xyz,
    basis,
    mm,
    method,
    keywords,
    extra_prov=None,
    grid=None,
    aux=None,
    scf_conv=None,
):
    import pyscf

    symbols, coords = common.read_xyz(xyz)
    mol = mol_for(xyz, basis)
    basis_check = common.check_basis_like_for_like(mol, basis, symbols)
    return mol, {
        "row": row_name,
        "system": system,
        "basis": basis,
        "method": method,
        "charge": 0,
        "multiplicity": 1,
        "nao": int(mol.nao_nr()),
        "nuclear_repulsion": float(mol.energy_nuc()),
        "atoms": [{"symbol": s, "xyz_bohr": c} for s, c in zip(symbols, coords)],
        "mm_charges": [{"q": q, "xyz_bohr": [x, y, z]} for q, x, y, z in mm],
        "units": "Bohr / Hartree / a.u.; gradients are dE/dR "
        "(ferric mm_forces returns the FORCE = -mm_gradient)",
        "provenance": common.provenance(
            code="PySCF",
            version=pyscf.__version__,
            keywords=keywords,
            basis_name=basis,
            xyz_path=xyz,
            coords_bohr=coords,
            symbols=symbols,
            grid=grid,
            aux=aux,
            frozen_core="none (all electrons correlated)"
            if method == "rimp2"
            else None,
            scf_conv=scf_conv,
            stability=None,
            generator=GEN,
            extra={"basis_check": basis_check, **(extra_prov or {})},
        ),
    }


# ---------------------------------------------------------------------------
# Groups
# ---------------------------------------------------------------------------


def _xyz(name):
    return common.MOL_DIR / name


def _qm_angstrom(xyz):
    _, coords = common.read_xyz(xyz)
    return [np.array(c) / A2B for c in coords]


def water_mm():
    return water_charges_10(_qm_angstrom(_xyz("h2o.xyz")))


def shell_mm():
    return load_or_check_shell(_qm_angstrom(_xyz("ch3oh.xyz")))


RHF_KW = (
    "scf.RHF exact 4-index J/K, conv_tol 1e-12, conv_tol_grad 1e-9; "
    "qmmm.mm_charge(mf, coords, charges, unit='Bohr'); gradient "
    "mf.nuc_grad_method().kernel(); MM gradient g.grad_hcore_mm(dm)+g.grad_nuc_mm()"
)


def group_point(written):
    cases = [
        ("h2o_q10", "h2o.xyz", "cc-pvdz", water_mm()),
        ("h2o_q10", "h2o.xyz", "aug-cc-pvdz", water_mm()),
        ("ch3oh_tip3p501", "ch3oh.xyz", "6-31g", shell_mm()),
    ]
    for system, xyzname, basis, mm in cases:
        xyz = _xyz(xyzname)
        mol, payload = common_payload(
            "QM/MM",
            system,
            xyz,
            basis,
            mm,
            "rhf",
            RHF_KW,
            scf_conv={"conv_tol": CONV_TOL, "conv_tol_grad": CONV_TOL_GRAD},
        )
        res, dm = full_case(mol, mm)
        payload.update(res)
        payload["control_scaled_101"] = scaled_control(mol, mm, dm0=dm)
        path = common.write_reference(ROW, system, basis, payload)
        written.append(path)
        print(
            f"{path.name}: E {res['energy']:.10f} shift {res['shift']:+.8f} "
            f"x1.01 dE {payload['control_scaled_101']['energy'] - res['energy']:+.3e} "
            f"max|g_mm| {np.abs(res['mm_gradient']).max():.3e}"
        )


def group_smeared(written):
    xyz = _xyz("h2o.xyz")
    mm = water_mm()
    basis = "cc-pvdz"
    energies = {}
    for w in SMEAR_WIDTHS_BOHR:
        radii = [w] * len(mm)
        system = f"h2o_q10_smeared_w{w:g}".replace(".", "p")
        kw = RHF_KW.replace("unit='Bohr'", f"radii=[{w}]*{len(mm)}, unit='Bohr'")
        mol, payload = common_payload(
            "Smeared charges",
            system,
            xyz,
            basis,
            mm,
            "rhf",
            kw,
            scf_conv={"conv_tol": CONV_TOL, "conv_tol_grad": CONV_TOL_GRAD},
        )
        res, dm = full_case(mol, mm, radii=radii)
        payload.update(res)
        payload["widths_bohr"] = radii
        payload["zeta_convention"] = "zeta = 1/width_bohr**2 (PySCF radii, unit='Bohr')"
        payload["control_scaled_101"] = scaled_control(mol, mm, radii=radii, dm0=dm)
        e_point, _ = embedded_energy(mol, mm, dm0=dm)
        payload["control_point_limit"] = {"energy": e_point}
        energies[w] = res["energy"]

        # Unit convention: coords AND radii in Angstrom must reproduce the Bohr run.
        mm_ang = [(q, x / A2B, y / A2B, z / A2B) for q, x, y, z in mm]
        e_ang, _ = embedded_energy(
            mol, mm_ang, radii=[w / A2B] * len(mm), unit="Angstrom", dm0=dm
        )
        # The trap: the Bohr NUMBER handed over as an Angstrom radius.
        e_trap, _ = embedded_energy(mol, mm_ang, radii=radii, unit="Angstrom", dm0=dm)
        d_ang = abs(e_ang - res["energy"])
        d_trap = abs(e_trap - res["energy"])
        if d_ang > UNIT_CHECK_TOL:
            raise RuntimeError(
                f"{system}: Angstrom-unit PySCF run differs by {d_ang:.2e}"
            )
        if d_trap < 1e-6:
            raise RuntimeError(
                f"{system}: unit trap does not move the energy ({d_trap:.2e})"
            )
        payload["unit_convention_check"] = {
            "energy_coords_and_radii_in_angstrom": e_ang,
            "abs_diff_vs_bohr_run": d_ang,
            "energy_bohr_radius_misread_as_angstrom": e_trap,
            "abs_diff_trap_vs_bohr_run": d_trap,
        }
        d_point = abs(e_point - res["energy"])
        if d_point < 1e-5:
            raise RuntimeError(
                f"{system}: smeared vs point differs by only {d_point:.2e}; "
                "the point-limit control would not discriminate"
            )
        path = common.write_reference(ROW, system, basis, payload)
        written.append(path)
        print(
            f"{path.name}: E {res['energy']:.10f} shift {res['shift']:+.8f} "
            f"|E-E_point| {d_point:.3e} unit-check {d_ang:.1e} trap {d_trap:.3e}"
        )
    # Cross-width controls: each file records the other widths' energies.
    for w in SMEAR_WIDTHS_BOHR:
        system = f"h2o_q10_smeared_w{w:g}".replace(".", "p")
        path = common.reference_path(ROW, system, basis)
        payload = json.loads(path.read_text())
        payload["control_widths"] = {
            f"{v:g}": {
                "energy": energies[v],
                "abs_diff": abs(energies[v] - energies[w]),
            }
            for v in SMEAR_WIDTHS_BOHR
            if v != w
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")


def group_ks(written):
    xyz = _xyz("ch3oh.xyz")
    _, qm_bohr = common.read_xyz(xyz)
    mm = nearest_sites(shell_mm(), qm_bohr, KS_N_CHARGES)
    basis = "def2-svp"
    for xc in ("b3lyp", "pbe"):
        system = f"ch3oh_q20_{xc}"
        kw = (
            f"dft.RKS(xc='{PYSCF_XC[xc]}') EXACT 4-index J/K (no density_fit), grid "
            f"{MAIN_GRID} prune=None radii_adjust=becke_atomic_radii_adjust, conv_tol "
            f"{KS_CONV_TOL} conv_tol_grad {KS_CONV_TOL_GRAD}; qmmm.mm_charge unit='Bohr'; "
            "qm_gradient grid_response=False (informational), "
            "qm_gradient_grid_response grid_response=True (the reference: ferric's KS "
            "gradient includes the grid response)"
        )
        mol, payload = common_payload(
            "KS QM/MM",
            system,
            xyz,
            basis,
            mm,
            "rks",
            kw,
            grid={
                "atom_grid": list(MAIN_GRID),
                "prune": None,
                "partition": "Becke",
                "radii_adjust": "becke_atomic_radii_adjust",
            },
            aux=None,
            scf_conv={"conv_tol": KS_CONV_TOL, "conv_tol_grad": KS_CONV_TOL_GRAD},
            extra_prov={
                "jk": 'exact four-centre J and K (ferric df_j_aux = df_k_aux = Some(""))',
                "mm_selection": f"the {KS_N_CHARGES} shell sites nearest to any QM atom",
            },
        )
        res, dm = full_case(mol, mm, xc=xc)
        payload.update(res)
        payload["xc"] = xc
        payload["control_scaled_101"] = scaled_control(mol, mm, xc=xc, dm0=dm)
        path = common.write_reference(ROW, system, basis, payload)
        written.append(path)
        gr = np.abs(
            np.array(res["qm_gradient"]) - np.array(res["qm_gradient_grid_response"])
        ).max()
        print(
            f"{path.name}: E {res['energy']:.10f} shift {res['shift']:+.8f} "
            f"|g - g_gridresp| {gr:.2e}"
        )


def _dfmp2_energy(mol, mm, dm0=None):
    """(e_total, e_rhf, e_corr, mf, naux) for exact-J/K RHF in the field + DF-MP2."""
    from pyscf import df
    from pyscf.mp import dfmp2

    e_rhf, mf = embedded_energy(mol, mm, dm0=dm0)
    symbols = [mol.atom_symbol(i) for i in range(mol.natm)]
    aux_bas, aux_cart = common.pyscf_basis(MP2_AUX, symbols)
    assert not aux_cart
    mp = dfmp2.DFMP2(mf, frozen=None)
    mp.with_df = df.DF(mol)
    mp.with_df.auxmol = df.addons.make_auxmol(mol, aux_bas)
    mp.with_df.auxmol.cart = False
    mp.with_df.auxmol.build()
    mp.with_df.auxbasis = aux_bas
    mp.kernel()
    return e_rhf + mp.e_corr, e_rhf, float(mp.e_corr), mf, mp.with_df.auxmol.nao_nr()


def _fd_gradient(xyz, basis, mm, h, dm0):
    _, c0 = common.read_xyz(xyz)
    c0 = np.array(c0)
    g = np.zeros_like(c0)
    stencil = [(-2, 1.0 / 12), (-1, -8.0 / 12), (1, 8.0 / 12), (2, -1.0 / 12)]
    for a in range(c0.shape[0]):
        for k in range(3):
            acc = 0.0
            for step, wgt in stencil:
                c = c0.copy()
                c[a, k] += step * h
                e_tot, *_ = _dfmp2_energy(mol_for(xyz, basis, c.tolist()), mm, dm0)
                acc += wgt * e_tot
            g[a, k] = acc / h
    return g


def group_mp2(written):
    xyz_w, xyz_m = _xyz("h2o.xyz"), _xyz("ch3oh.xyz")
    _, qm_m = common.read_xyz(xyz_m)
    cases = [
        ("h2o_q10_rimp2", xyz_w, "cc-pvdz", water_mm()),
        (
            "ch3oh_q20_rimp2",
            xyz_m,
            "6-31g",
            nearest_sites(shell_mm(), qm_m, KS_N_CHARGES),
        ),
    ]
    for system, xyz, basis, mm in cases:
        kw = (
            "scf.RHF exact 4-index J/K (conv_tol 1e-12, conv_tol_grad 1e-9) wrapped by "
            "qmmm.mm_charge(unit='Bohr'); mp.dfmp2.DFMP2(frozen=None) with auxmol from "
            f"ferric's {MP2_AUX} JSON; gradient = 5-point central FD of E_total at "
            f"h = {H_FD} and 2h Bohr (must agree to {FD_AGREE})"
        )
        mol, payload = common_payload(
            "External potential in MP2 gradients",
            system,
            xyz,
            basis,
            mm,
            "rimp2",
            kw,
            aux={
                "correlation": MP2_AUX,
                "correlation_json": str(
                    common.basis_json_path(MP2_AUX).relative_to(common.ROOT)
                ),
                "correlation_sha256": common.sha256_file(
                    common.basis_json_path(MP2_AUX)
                ),
                "scf": "none (exact four-centre J/K)",
            },
            scf_conv={"conv_tol": CONV_TOL, "conv_tol_grad": CONV_TOL_GRAD},
        )
        e_tot, e_rhf, e_corr, mf, naux = _dfmp2_energy(mol, mm)
        dm = mf.make_rdm1()
        g_rhf = np.asarray(mf.nuc_grad_method().kernel())
        g_h = _fd_gradient(xyz, basis, mm, H_FD, dm)
        g_2h = _fd_gradient(xyz, basis, mm, 2 * H_FD, dm)
        d_fd = float(np.abs(g_h - g_2h).max())
        if d_fd > FD_AGREE:
            raise RuntimeError(f"{system}: FD h vs 2h disagree by {d_fd:.2e}")
        e_tot_s, *_ = _dfmp2_energy(
            mol, [(q * SCALE_CONTROL, x, y, z) for q, x, y, z in mm], dm
        )
        # vacuum DF-MP2 energy, for the shift
        from pyscf import df, scf
        from pyscf.mp import dfmp2

        mf0 = scf.RHF(mol)
        mf0.conv_tol, mf0.conv_tol_grad, mf0.verbose = CONV_TOL, CONV_TOL_GRAD, 0
        mf0.kernel()
        assert mf0.converged
        symbols = [mol.atom_symbol(i) for i in range(mol.natm)]
        aux_bas, _ = common.pyscf_basis(MP2_AUX, symbols)
        mp0 = dfmp2.DFMP2(mf0, frozen=None)
        mp0.with_df = df.DF(mol)
        mp0.with_df.auxmol = df.addons.make_auxmol(mol, aux_bas)
        mp0.with_df.auxmol.cart = False
        mp0.with_df.auxmol.build()
        mp0.with_df.auxbasis = aux_bas
        mp0.kernel()
        e_gas = float(mf0.e_tot + mp0.e_corr)

        payload.update(
            {
                "aux_basis": MP2_AUX,
                "naux": int(naux),
                "e_rhf": e_rhf,
                "e_corr": e_corr,
                "energy": e_tot,
                "energy_gas_phase": e_gas,
                "shift": e_tot - e_gas,
                "gradient_fd": g_h.tolist(),
                "gradient_fd_2h": g_2h.tolist(),
                "fd": {
                    "stencil": "5-point central",
                    "step_bohr": H_FD,
                    "max_abs_diff_h_vs_2h": d_fd,
                },
                "rhf_gradient": g_rhf.tolist(),
                "control_scaled_101": {
                    "scale": SCALE_CONTROL,
                    "energy": float(e_tot_s),
                },
            }
        )
        path = common.write_reference(ROW, system, basis, payload)
        written.append(path)
        print(
            f"{path.name}: E_tot {e_tot:.10f} shift {e_tot - e_gas:+.8f} "
            f"|g_h-g_2h| {d_fd:.1e} |g_mp2-g_rhf| {np.abs(g_h - g_rhf).max():.2e}"
        )


GROUPS = {
    "point": group_point,
    "smeared": group_smeared,
    "ks": group_ks,
    "mp2": group_mp2,
}


def main() -> int:
    only = [a for a in sys.argv[1:] if not a.startswith("-")]
    for g in only:
        if g not in GROUPS:
            raise SystemExit(f"unknown group {g}; choose from {sorted(GROUPS)}")
    written = []
    for name, fn in GROUPS.items():
        if only and name not in only:
            continue
        fn(written)
    print(f"GEN_QMMM_DONE written={len(written)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
