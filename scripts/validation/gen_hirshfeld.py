"""PySCF / numpy references for the VALIDATION.md "Hirshfeld" row (design §5.3,
row 150), plus the Hirshfeld effective volumes that feed ferric's TS C6.

    Hirshfeld charges     -> hirshfeld/<system>_<basis>.json
    Hirshfeld volumes        (same file)

Consumer: crates/ferric-rpa/tests/validation_hirshfeld.rs

WHAT FERRIC COMPUTES (read from the code, not the doc comments)
---------------------------------------------------------------
`ferric_rpa::properties::hirshfeld_charges(mol, bs, D, proatom)`:

    q_A = Z_A - s * n_A,   n_A = sum_g w_g rho(r_g) w_A(r_g),
    w_A(r) = rho0_A(|r - R_A|) / (sum_B rho0_B(|r - R_B|) + 1e-12),
    s = N_e / sum_A n_A          (renormalization to the electron count)

on ferric's default XC grid (75 TA-M4 radial x 110 Lebedev, unpruned, Becke
partition of the home atom folded into w_g). Two proatom sources:

* `proatom = Some(provider)` (what ferric-cli and the Python binding
  `ferric.hirshfeld_charges` pass by default, `proatom="scf"`): a tabulated radial
  density, `RadialProatom { radii = 0.05, 0.10, ..., 30.0 Bohr, rho }`, where
  rho(r_k) is the Lebedev-110 spherical average of the free NEUTRAL atom's SCF
  density in the molecule's own basis (`spherically_averaged_proatom`). Between
  nodes `RadialProatom::at` interpolates linearly; below 0.05 Bohr it returns
  rho(0.05); at or beyond 30 Bohr it returns 0. The free-atom SCF is ferric-cli's
  recipe with an HF molecular config: RHF for singlets, else UHF with MOM after
  iteration 5 (H doublet, C and O triplets here).
* `proatom = None` (what the Python binding passes only for `proatom="slater"`):
  rho0_A(r) = Z_A xi^3 / pi * exp(-2 xi r), a single normalized Slater
  exponential with xi = 1 / R_BS(Z) and R_BS from `slater_xi_for_z`
  (ferric-scf/src/properties.rs), a table that is NOT the Becke-partition
  Bragg-Slater table (H is 0.25 A here, 0.35 A there).

`ferric_rpa::properties::atomic_effective_volumes_hirshfeld(mol, bs, D, proatom)`:

    v_A = sum_g h^3 rho(r_g) w_A(r_g) |r_g - R_A|^3

on a UNIFORM Cartesian lattice (`GridSpec::bounding_box`, spacing h = 0.20 Bohr,
6 Bohr margin, n = ceil(L / h) points per axis, origin at the box corner), same
Hirshfeld weight and 1e-12 floor, no renormalization. ferric-cli uses it for the
TS volume ratios: molecule with the SCF proatom, free-atom denominator with
`None` on a single atom.

THE DESIGN: SAME DENSITY, SAME PROATOMS, TWO CODES
--------------------------------------------------
Each file carries PySCF's converged RHF density in ferric's AO order (the
permutation and its overlap check are gen_properties.py's), and, per element,
PySCF's free-atom UHF density in ferric's AO order plus its tabulated proatom.
The Rust test feeds ferric (a) the reference density and the reference proatom
tables, which isolates the partition + quadrature code, and (b) its own SCF and
its own free-atom SCFs (the full chain).

The numpy side is an independent implementation of the definition above:
PySCF AO values (`eval_ao`) and PySCF's Lebedev and TA-M4 radial rules, with
ferric's grid rebuilt from them (gen_properties.atom_centred_grid /
becke_partition — the partition and table choices are DEFINITIONS). The
spherical average of the free-atom density uses Lebedev 302, not ferric's 110:
both are exact for the degree-4 angular content of a cc-pVDZ / def2-SVP atom,
so any difference is a construction error, not a quadrature one.

Stored alongside, as MEASUREMENTS (asserted loosely or not at all):
* the same charges on a dense (200, 590) grid: ferric's grid error;
* the same charges on PySCF's own level-9 molecular grid (PySCF's Becke
  partition and pruning): an independent quadrature of the same integral, which
  must agree with the dense grid;
* the same charges with the proatom tabulated at 0.005 Bohr instead of 0.05:
  the tabulation error of ferric's `RadialProatom`;
* the Hirshfeld volumes on a dense Becke-Lebedev grid: the lattice's error
  (spacing AND the 6 Bohr box truncation);
* per free atom, the `None`-path single-atom lattice volume against the same
  integral with weight 1: the effect of the 1e-12 floor on the TS denominator.

Run (light; a couple of minutes):
    scripts/validation/run_slot.sh --light -- \\
        uv run --no-sync python scripts/validation/gen_hirshfeld.py
    (a trailing system name restricts, e.g. `... gen_hirshfeld.py co`)
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402
import gen_properties as gp  # noqa: E402

GENERATOR = "scripts/validation/gen_hirshfeld.py"
ROW = "hirshfeld"

SYSTEMS = ("h2o", "co", "ch3oh")
BASES = ("cc-pvdz", "def2-svp")

# ferric-cli proatom_gs_mult for the elements used here.
FREE_ATOM_MULT = {1: 2, 2: 1, 6: 3, 8: 3}
FREE_ATOM_XYZ = {1: "h_atom", 2: "he_atom", 6: "c_atom", 8: "o_atom"}

# Exactness anchor: a PROMOLECULE density (block-diagonal free-atom densities)
# of spherical atoms has zero Hirshfeld charges up to the proatom tabulation
# and the grid. He (1S) and H (2S) are spherical in any basis.
ANCHOR_SYSTEM = "heh"

CONV_TOL = 1e-12
CONV_TOL_GRAD = 1e-9

# ferric-cli: `(1..=600).map(|k| k as f64 * 0.05)`.
PROATOM_STEP = 0.05
PROATOM_N = 600
FINE_STEP = 0.005  # tabulation-error measurement only
FINE_N = 6000
SPHERE_LEBEDEV = 302  # ferric uses 110; both exact here (see docstring)

EPS_FLOOR = 1e-12  # ferric: `rho_sum[g] + eps_floor`

FERRIC_GRID = gp.FERRIC_GRID  # (75, 110)
DENSE_GRID = gp.DENSE_GRID  # (200, 590)
PYSCF_GRID_LEVEL = 9  # PySCF-native grid, an independent quadrature cross-check

# ferric-scf properties.rs hirshfeld_spacing()/hirshfeld_margin() defaults.
LATTICE_SPACING = 0.20
LATTICE_MARGIN = 6.0

# ferric-scf/src/properties.rs::slater_xi_for_z, VERBATIM (Angstrom, and the
# conversion factor that function uses).
_SLATER_R_ANG = {
    1: 0.25, 2: 0.30, 3: 1.45, 4: 1.05, 5: 0.85, 6: 0.70, 7: 0.65, 8: 0.60,
    9: 0.50, 10: 0.45, 11: 1.80, 12: 1.50, 13: 1.25, 14: 1.10, 15: 1.00,
    16: 1.00, 17: 1.00, 18: 0.71,
}  # fmt: skip
_SLATER_ANG_TO_BOHR = 1.8897259886

CHUNK = 40_000


def slater_xi(z: int) -> float:
    return 1.0 / (_SLATER_R_ANG.get(z, 1.00) * _SLATER_ANG_TO_BOHR)


# ---------------------------------------------------------------------------
# Proatoms
# ---------------------------------------------------------------------------


def lebedev_unit(n: int):
    import numpy as np
    from pyscf.dft.LebedevGrid import MakeAngularGrid

    ang = np.asarray(MakeAngularGrid(n))
    if ang.shape != (n, 4):
        raise ValueError(f"Lebedev {n}: got shape {ang.shape}")
    return ang[:, :3], ang[:, 3] / ang[:, 3].sum()


def spherical_average(atom_mol, dm, radii):
    """(1/4pi) \\int rho(r Omega) dOmega at each radius, clamped at 0 (ferric)."""
    import numpy as np
    from pyscf.dft import numint

    dirs, w = lebedev_unit(SPHERE_LEBEDEV)
    out = np.empty(len(radii))
    per = max(1, CHUNK // len(dirs))
    for i0 in range(0, len(radii), per):
        rr = np.asarray(radii[i0 : i0 + per])
        pts = (rr[:, None, None] * dirs[None, :, :]).reshape(-1, 3)
        ao = numint.eval_ao(atom_mol, pts, deriv=0)
        rho = np.einsum("pi,ij,pj->p", ao, dm, ao).reshape(len(rr), len(dirs))
        out[i0 : i0 + len(rr)] = rho @ w
    return np.maximum(out, 0.0)


def table_eval(radii, rho, r):
    """ferric `RadialProatom::at`: rho[0] at r <= radii[0], 0 at r >= radii[-1],
    linear in between."""
    import numpy as np

    out = np.interp(r, radii, rho)
    out = np.where(r <= radii[0], rho[0], out)
    return np.where(r >= radii[-1], 0.0, out)


def free_atom(z: int, basis_name: str) -> dict:
    """UHF free atom (ferric-cli's HF proatom recipe) + its proatom tables."""
    import numpy as np

    xyz = common.MOL_DIR / f"{FREE_ATOM_XYZ[z]}.xyz"
    symbols, coords = common.read_xyz(xyz)
    mol = common.build_pyscf_mol(xyz, basis_name, multiplicity=FREE_ATOM_MULT[z])
    basis_check = common.check_basis_like_for_like(mol, basis_name, symbols)
    if FREE_ATOM_MULT[z] == 1:
        info, mf = gp.run_rhf(mol)  # ferric-cli: RHF for singlet free atoms
        dm = mf.make_rdm1()
    else:
        info, mf = common.run_open_shell(
            mol, "uhf", conv_tol=CONV_TOL, conv_tol_grad=CONV_TOL_GRAD, return_mf=True
        )
        dm = mf.make_rdm1()
        dm = dm[0] + dm[1]
    perm = gp.ferric_ao_permutation(mol, basis_name, symbols)
    radii = PROATOM_STEP * np.arange(1, PROATOM_N + 1)
    fine = FINE_STEP * np.arange(1, FINE_N + 1)
    rho = spherical_average(mol, dm, radii)
    rho_fine = spherical_average(mol, dm, fine)
    # Self-check: the tabulated proatom integrates to Z (trapezoid on the table
    # from r = 0 with rho(0) ~ rho(0.005), i.e. only a sanity bound).
    nel = float(np.trapz(4 * np.pi * fine**2 * rho_fine, fine))
    if abs(nel - z) > 1e-3:
        raise RuntimeError(f"Z={z} {basis_name}: proatom integrates to {nel:.6f}")
    return {
        "z": z,
        "symbol": symbols[0],
        "multiplicity": FREE_ATOM_MULT[z],
        "nao": mol.nao_nr(),
        "scf": info,
        "scf_method": "rhf" if FREE_ATOM_MULT[z] == 1 else "uhf",
        "overlap_ferric_order": gp.mat_json(
            gp.to_ferric_order(mol.intor("int1e_ovlp"), perm)
        ),
        "density_total_ferric_order": gp.mat_json(gp.to_ferric_order(dm, perm)),
        "proatom_radii_step_bohr": PROATOM_STEP,
        "proatom_n_radii": PROATOM_N,
        "proatom_rho": [float(x) for x in rho],
        "proatom_electrons_fine_trapezoid": nel,
        "_fine": (fine, rho_fine),
        "_mol": mol,
        "_dm": dm,
        "_basis_check": basis_check,
    }


# ---------------------------------------------------------------------------
# Hirshfeld on an atom-centred grid
# ---------------------------------------------------------------------------


def density_on(mol, dm, pts):
    import numpy as np
    from pyscf.dft import numint

    out = np.empty(len(pts))
    for g0 in range(0, len(pts), CHUNK):
        ao = numint.eval_ao(mol, pts[g0 : g0 + CHUNK], deriv=0)
        out[g0 : g0 + CHUNK] = np.einsum("pi,ij,pj->p", ao, dm, ao)
    return out


def becke_grid(mol, level):
    """(pts, w_g, home): ferric's flat grid with the home Becke factor folded in."""
    import numpy as np

    zs = [int(mol.atom_charge(i)) for i in range(mol.natm)]
    xyz = np.asarray(mol.atom_coords(unit="Bohr"))
    home, pts, w_rl = gp.atom_centred_grid(zs, xyz, *level)
    wb = np.empty(len(pts))
    for g0 in range(0, len(pts), CHUNK):
        g1 = min(g0 + CHUNK, len(pts))
        wb[g0:g1] = gp.becke_partition(zs, xyz, pts[g0:g1])[
            home[g0:g1], np.arange(g1 - g0)
        ]
    return pts, w_rl * wb, home


def proatoms_on(mol, pts, proatom_fns):
    """(natm, npts) rho0_A(|r - R_A|)."""
    import numpy as np

    xyz = np.asarray(mol.atom_coords(unit="Bohr"))
    out = np.empty((mol.natm, len(pts)))
    for a in range(mol.natm):
        r = np.linalg.norm(pts - xyz[a], axis=1)
        out[a] = proatom_fns[int(mol.atom_charge(a))](r)
    return out


def hirshfeld_weights(rho0):
    return rho0 / (rho0.sum(axis=0) + EPS_FLOOR)


def hirshfeld_charges(mol, rho, wg, rho0) -> tuple[list[float], list[float]]:
    import numpy as np

    n = (hirshfeld_weights(rho0) * (rho * wg)[None, :]).sum(axis=1)
    s = mol.nelectron / n.sum()
    zs = np.array([mol.atom_charge(a) for a in range(mol.natm)], dtype=float)
    return [float(x) for x in zs - s * n], [float(x) for x in n]


def lattice(mol):
    """ferric `GridSpec::bounding_box(mol, 6.0, 0.20)` points, ferric's order."""
    import numpy as np

    xyz = np.asarray(mol.atom_coords(unit="Bohr"))
    lo, hi = xyz.min(axis=0), xyz.max(axis=0)
    origin = lo - LATTICE_MARGIN
    lengths = (hi - lo) + 2.0 * LATTICE_MARGIN
    n = [int(math.ceil(float(v) / LATTICE_SPACING)) for v in lengths]
    ax = [origin[i] + np.arange(n[i]) * LATTICE_SPACING for i in range(3)]
    gx, gy, gz = np.meshgrid(*ax, indexing="ij")  # g = (ix*ny + iy)*nz + iz
    pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    return pts, n


def hirshfeld_volumes(mol, rho, wg, rho0, pts) -> list[float]:
    import numpy as np

    xyz = np.asarray(mol.atom_coords(unit="Bohr"))
    w = hirshfeld_weights(rho0)
    vol = []
    for a in range(mol.natm):
        r3 = np.linalg.norm(pts - xyz[a], axis=1) ** 3
        vol.append(float(np.sum(w[a] * rho * r3 * wg)))
    return vol


# ---------------------------------------------------------------------------
# Row
# ---------------------------------------------------------------------------


def gen(only: set[str]) -> list[Path]:
    import numpy as np
    import pyscf
    from pyscf import dft

    written = []
    for basis_name in BASES:
        atoms: dict[int, dict] = {}
        for system in SYSTEMS:
            if only and system not in only:
                continue
            t0 = time.time()
            xyz = common.MOL_DIR / f"{system}.xyz"
            symbols, coords = common.read_xyz(xyz)
            mol = common.build_pyscf_mol(xyz, basis_name)
            basis_check = common.check_basis_like_for_like(mol, basis_name, symbols)
            scf_info, mf = gp.run_rhf(mol)
            dm = mf.make_rdm1()
            perm = gp.ferric_ao_permutation(mol, basis_name, symbols)

            zset = sorted({int(mol.atom_charge(a)) for a in range(mol.natm)})
            for z in zset:
                if z not in atoms:
                    atoms[z] = free_atom(z, basis_name)
            radii = PROATOM_STEP * np.arange(1, PROATOM_N + 1)
            tab = {
                z: (
                    lambda r, t=np.asarray(atoms[z]["proatom_rho"]): table_eval(
                        radii, t, r
                    )
                )
                for z in zset
            }
            fine = {
                z: (lambda r, f=atoms[z]["_fine"]: table_eval(f[0], f[1], r))
                for z in zset
            }
            slater = {}
            for z in zset:
                xi = slater_xi(z)
                slater[z] = lambda r, z=z, xi=xi: (
                    z * xi**3 / math.pi * np.exp(-2.0 * xi * r)
                )

            # --- charges, ferric's grid and dense grid
            out = {}
            for tag, level in (
                ("ferric_grid", FERRIC_GRID),
                ("dense_grid", DENSE_GRID),
            ):
                pts, wg, _home = becke_grid(mol, level)
                rho = density_on(mol, dm, pts)
                q_scf, n_scf = hirshfeld_charges(
                    mol, rho, wg, proatoms_on(mol, pts, tab)
                )
                q_sl, n_sl = hirshfeld_charges(
                    mol, rho, wg, proatoms_on(mol, pts, slater)
                )
                out[tag] = {
                    "npts": int(len(pts)),
                    "n_electrons_grid": float(np.sum(rho * wg)),
                    "scf_proatom": {"charges": q_scf, "n_raw": n_scf},
                    "slater": {"charges": q_sl, "n_raw": n_sl},
                }
                if tag == "dense_grid":
                    q_fine, _ = hirshfeld_charges(
                        mol, rho, wg, proatoms_on(mol, pts, fine)
                    )
                    out[tag]["scf_proatom_fine_table"] = {"charges": q_fine}
                    xyz_b = np.asarray(mol.atom_coords(unit="Bohr"))
                    w = hirshfeld_weights(proatoms_on(mol, pts, tab))
                    out[tag]["hirshfeld_volumes_scf_proatom"] = [
                        float(
                            np.sum(
                                w[a]
                                * rho
                                * wg
                                * np.linalg.norm(pts - xyz_b[a], axis=1) ** 3
                            )
                        )
                        for a in range(mol.natm)
                    ]

            # --- Independent quadrature: PySCF's own molecular grid (its Becke
            # partition with Treutler radii adjustment, NWChem pruning, level 9).
            pg = dft.gen_grid.Grids(mol)
            pg.level = PYSCF_GRID_LEVEL
            pg.build()
            prho = density_on(mol, dm, pg.coords)
            q_pyscf, _ = hirshfeld_charges(
                mol, prho, pg.weights, proatoms_on(mol, pg.coords, tab)
            )
            out["pyscf_native_grid"] = {
                "level": PYSCF_GRID_LEVEL,
                "npts": int(len(pg.weights)),
                "scf_proatom": {"charges": q_pyscf},
            }

            # --- Hirshfeld volumes on ferric's lattice (SCF proatom, as ferric-cli)
            lpts, nlat = lattice(mol)
            dv = LATTICE_SPACING**3
            lrho = density_on(mol, dm, lpts)
            vol_lat = hirshfeld_volumes(
                mol, lrho, np.full(len(lpts), dv), proatoms_on(mol, lpts, tab), lpts
            )
            vol_dense = out["dense_grid"]["hirshfeld_volumes_scf_proatom"]

            q_f = out["ferric_grid"]["scf_proatom"]["charges"]
            q_d = out["dense_grid"]["scf_proatom"]["charges"]
            q_fine = out["dense_grid"]["scf_proatom_fine_table"]["charges"]
            qs_f = out["ferric_grid"]["slater"]["charges"]
            qs_d = out["dense_grid"]["slater"]["charges"]
            measurements = {
                "charges_dense_vs_pyscf_native_grid_max_abs": max(
                    abs(a - b) for a, b in zip(q_d, q_pyscf)
                ),
                "charges_ferric_grid_vs_dense_max_abs": max(
                    abs(a - b) for a, b in zip(q_f, q_d)
                ),
                "charges_slater_ferric_grid_vs_dense_max_abs": max(
                    abs(a - b) for a, b in zip(qs_f, qs_d)
                ),
                "charges_table_0p05_vs_0p005_dense_max_abs": max(
                    abs(a - b) for a, b in zip(q_d, q_fine)
                ),
                "charges_scf_vs_slater_proatom_max_abs": max(
                    abs(a - b) for a, b in zip(q_f, qs_f)
                ),
                "volumes_lattice_vs_dense_max_rel": max(
                    abs(a - b) / abs(b) for a, b in zip(vol_lat, vol_dense)
                ),
            }

            atoms_json = {}
            for z in zset:
                a = atoms[z]
                atoms_json[str(z)] = {
                    k: v for k, v in a.items() if not k.startswith("_")
                }
            payload = {
                "row": "Hirshfeld charges / Hirshfeld effective volumes",
                "system": system,
                "basis": basis_name,
                "charge": 0,
                "multiplicity": 1,
                "reference": "rhf",
                "nao": mol.nao_nr(),
                "nelectron": int(mol.nelectron),
                "nuclear_repulsion": float(mol.energy_nuc()),
                "scf": scf_info,
                "ao_permutation_ferric_to_pyscf": perm,
                "overlap_ferric_order": gp.mat_json(
                    gp.to_ferric_order(mol.intor("int1e_ovlp"), perm)
                ),
                "density_total_ferric_order": gp.mat_json(gp.to_ferric_order(dm, perm)),
                "free_atoms": atoms_json,
                "slater_xi": {str(z): slater_xi(z) for z in zset},
                "charges": {
                    "scf_proatom": q_f,
                    "slater": qs_f,
                    "scf_proatom_dense_grid": q_d,
                    "slater_dense_grid": qs_d,
                    "scf_proatom_fine_table_dense_grid": q_fine,
                    "detail": out,
                },
                "hirshfeld_volumes": {
                    "scf_proatom_lattice": vol_lat,
                    "scf_proatom_dense_grid": vol_dense,
                    "lattice": {
                        "spacing_bohr": LATTICE_SPACING,
                        "margin_bohr": LATTICE_MARGIN,
                        "n": nlat,
                        "npts": int(len(lpts)),
                        "n_electrons_lattice": float(np.sum(lrho) * dv),
                    },
                },
                "measurements": measurements,
                "definition": {
                    "weight": "rho0_A / (sum_B rho0_B + 1e-12)",
                    "charges": "Z_A - (N_e / sum_B n_B) n_A on ferric's (75,110) TA-M4 x "
                    "Lebedev Becke grid",
                    "scf_proatom": "RadialProatom table r_k = 0.05 k (k=1..600) Bohr, "
                    "linear interpolation, rho(0.05) below, 0 at/after 30 Bohr; neutral "
                    "free-atom UHF (H 2S, C 3P, O 3P) in the molecule's basis",
                    "slater": "Z xi^3/pi exp(-2 xi r), xi = 1/R_BS from slater_xi_for_z",
                    "volumes": "sum_g h^3 rho w_A |r-R_A|^3 on GridSpec::bounding_box(6.0, 0.20)",
                },
                "provenance": common.provenance(
                    code="PySCF + numpy",
                    version=pyscf.__version__,
                    keywords={
                        "scf": "scf.RHF (molecule); scf.UHF + stability loop (free atoms)",
                        "eri": "exact 4-index (no density fitting)",
                        "sphere_average": f"Lebedev {SPHERE_LEBEDEV}",
                        "numpy": np.__version__,
                    },
                    basis_name=basis_name,
                    xyz_path=xyz,
                    coords_bohr=coords,
                    symbols=symbols,
                    grid={
                        "charges": list(FERRIC_GRID),
                        "dense": list(DENSE_GRID),
                        "volumes_lattice": [LATTICE_SPACING, LATTICE_MARGIN],
                    },
                    aux=None,
                    frozen_core=None,
                    scf_conv={"conv_tol": CONV_TOL, "conv_tol_grad": CONV_TOL_GRAD},
                    stability=scf_info["stability"],
                    generator=GENERATOR,
                    extra={
                        "basis_self_check": basis_check,
                        "free_atom_basis_self_check": {
                            str(z): atoms[z]["_basis_check"] for z in zset
                        },
                    },
                ),
            }
            written.append(common.write_reference(ROW, system, basis_name, payload))
            print(
                f"{system:6s} {basis_name:9s} E={scf_info['energy']:.10f} "
                f"q_scf={['%+.5f' % x for x in q_f]} q_slater={['%+.5f' % x for x in qs_f]} "
                f"| grid {measurements['charges_ferric_grid_vs_dense_max_abs']:.1e} "
                f"pyscf-grid {measurements['charges_dense_vs_pyscf_native_grid_max_abs']:.1e} "
                f"table {measurements['charges_table_0p05_vs_0p005_dense_max_abs']:.1e} "
                f"lattice_vol_rel {measurements['volumes_lattice_vs_dense_max_rel']:.1e} "
                f"({time.time() - t0:.0f} s)",
                flush=True,
            )
        if not only or ANCHOR_SYSTEM in only:
            written.append(gen_anchor(basis_name, atoms))
        # Free-atom TS denominator: ferric-cli computes it with `None` on one atom.
        for z, a in atoms.items():
            if z not in (1, 6, 8):
                continue
            amol, adm = a["_mol"], a["_dm"]
            lpts, _ = lattice(amol)
            dv = LATTICE_SPACING**3
            rho = density_on(amol, adm, lpts)
            r = np.linalg.norm(lpts, axis=1)
            xi = slater_xi(z)
            rho0 = z * xi**3 / math.pi * np.exp(-2.0 * xi * r)
            w = rho0 / (rho0 + EPS_FLOOR)
            v_ferric = float(np.sum(w * rho * r**3) * dv)
            v_w1 = float(np.sum(rho * r**3) * dv)
            pts, wg, _ = becke_grid(amol, DENSE_GRID)
            rr = np.linalg.norm(pts, axis=1)
            v_exact = float(np.sum(density_on(amol, adm, pts) * rr**3 * wg))
            path = common.write_reference(
                ROW,
                f"{FREE_ATOM_XYZ[z]}_volume",
                basis_name,
                {
                    "row": "Hirshfeld volumes (free-atom TS denominator)",
                    "system": FREE_ATOM_XYZ[z],
                    "basis": basis_name,
                    "charge": 0,
                    "multiplicity": FREE_ATOM_MULT[z],
                    "nao": amol.nao_nr(),
                    "scf": a["scf"],
                    "overlap_ferric_order": a["overlap_ferric_order"],
                    "density_total_ferric_order": a["density_total_ferric_order"],
                    "volume_ferric_none_lattice": v_ferric,
                    "volume_weight_one_lattice": v_w1,
                    "volume_dense_grid": v_exact,
                    "measurements": {
                        "floor_loss_rel": (v_w1 - v_ferric) / v_w1,
                        "lattice_vs_dense_rel": (v_ferric - v_exact) / v_exact,
                    },
                    "provenance": common.provenance(
                        code="PySCF + numpy",
                        version=pyscf.__version__,
                        keywords={
                            "scf": "scf.UHF + stability loop",
                            "numpy": np.__version__,
                        },
                        basis_name=basis_name,
                        xyz_path=common.MOL_DIR / f"{FREE_ATOM_XYZ[z]}.xyz",
                        coords_bohr=[[0.0, 0.0, 0.0]],
                        symbols=[a["symbol"]],
                        grid={
                            "lattice": [LATTICE_SPACING, LATTICE_MARGIN],
                            "dense": list(DENSE_GRID),
                        },
                        aux=None,
                        frozen_core=None,
                        scf_conv={"conv_tol": CONV_TOL, "conv_tol_grad": CONV_TOL_GRAD},
                        stability=a["scf"]["stability"],
                        generator=GENERATOR,
                        extra={"basis_self_check": a["_basis_check"]},
                    ),
                },
            )
            written.append(path)
            print(
                f"free {a['symbol']:2s} {basis_name:9s} E={a['scf']['energy']:.10f} "
                f"v_ferric={v_ferric:.6f} v_w=1={v_w1:.6f} v_dense={v_exact:.6f}",
                flush=True,
            )
    return written


def gen_anchor(basis_name: str, atoms: dict) -> Path:
    """HeH promolecule: D = He (+) H free-atom densities, charges must be ~0."""
    import numpy as np
    import pyscf
    from scipy.linalg import block_diag

    xyz = common.MOL_DIR / f"{ANCHOR_SYSTEM}.xyz"
    symbols, coords = common.read_xyz(xyz)
    mol = common.build_pyscf_mol(xyz, basis_name, multiplicity=2)
    for z in (2, 1):
        if z not in atoms:
            atoms[z] = free_atom(z, basis_name)
    # PySCF's AO order is atom-major, so the free-atom blocks stack directly.
    dm = block_diag(atoms[2]["_dm"], atoms[1]["_dm"])
    if dm.shape != (mol.nao_nr(),) * 2:
        raise RuntimeError("promolecule block sizes do not match the HeH AO count")
    perm = gp.ferric_ao_permutation(mol, basis_name, symbols)
    radii = PROATOM_STEP * np.arange(1, PROATOM_N + 1)
    tab = {
        z: (lambda r, t=np.asarray(atoms[z]["proatom_rho"]): table_eval(radii, t, r))
        for z in (1, 2)
    }
    fine = {
        z: (lambda r, f=atoms[z]["_fine"]: table_eval(f[0], f[1], r)) for z in (1, 2)
    }
    slater = {
        z: (lambda r, z=z, xi=slater_xi(z): z * xi**3 / math.pi * np.exp(-2.0 * xi * r))
        for z in (1, 2)
    }
    res = {}
    for tag, level, fns in (
        ("ferric_grid", FERRIC_GRID, tab),
        ("ferric_grid_slater", FERRIC_GRID, slater),
        ("dense_grid", DENSE_GRID, tab),
        ("dense_grid_fine_table", DENSE_GRID, fine),
    ):
        pts, wg, _ = becke_grid(mol, level)
        rho = density_on(mol, dm, pts)
        q, n = hirshfeld_charges(mol, rho, wg, proatoms_on(mol, pts, fns))
        res[tag] = {"charges": q, "n_raw": n}
    payload = {
        "row": "Hirshfeld charges (promolecule exactness anchor)",
        "system": ANCHOR_SYSTEM,
        "basis": basis_name,
        "charge": 0,
        "multiplicity": 2,
        "nao": mol.nao_nr(),
        "nelectron": int(mol.nelectron),
        "nuclear_repulsion": float(mol.energy_nuc()),
        "overlap_ferric_order": gp.mat_json(
            gp.to_ferric_order(mol.intor("int1e_ovlp"), perm)
        ),
        "density_total_ferric_order": gp.mat_json(gp.to_ferric_order(dm, perm)),
        "free_atoms": {
            str(z): {k: v for k, v in atoms[z].items() if not k.startswith("_")}
            for z in (2, 1)
        },
        "charges": res,
        "provenance": common.provenance(
            code="PySCF + numpy",
            version=pyscf.__version__,
            keywords={
                "density": "block-diagonal free-atom densities (He RHF, H UHF); no molecular SCF",
                "numpy": np.__version__,
            },
            basis_name=basis_name,
            xyz_path=xyz,
            coords_bohr=coords,
            symbols=symbols,
            grid={"charges": list(FERRIC_GRID), "dense": list(DENSE_GRID)},
            aux=None,
            frozen_core=None,
            scf_conv={"conv_tol": CONV_TOL, "conv_tol_grad": CONV_TOL_GRAD},
            stability=None,
            generator=GENERATOR,
        ),
    }
    print(
        f"anchor {basis_name:9s} q(ferric grid)={['%+.2e' % x for x in res['ferric_grid']['charges']]} "
        f"q(dense)={['%+.2e' % x for x in res['dense_grid']['charges']]} "
        f"q(dense, fine table)={['%+.2e' % x for x in res['dense_grid_fine_table']['charges']]} "
        f"q(slater)={['%+.2e' % x for x in res['ferric_grid_slater']['charges']]}",
        flush=True,
    )
    return common.write_reference(ROW, ANCHOR_SYSTEM, basis_name, payload)


def main(argv: list[str]) -> int:
    t0 = time.time()
    only = {a.lower() for a in argv[1:]}
    paths = gen(only)
    for p in paths:
        print("wrote", p.relative_to(common.ROOT))
    print(f"total {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
