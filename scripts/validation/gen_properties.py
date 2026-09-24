"""PySCF / numpy references for the VALIDATION.md property rows (validation W1).

Rows covered (design §5.3):

    ESP at nuclei (147)          -> density_properties/<system>_<basis>.json
    Electric field at nuclei (149)  (same file)
    Becke volumes (152)             (same file; free atoms in free_atom_volumes/)
    Static alpha (148)           -> static_alpha/<system>_<basis>.json

Consumers:
    crates/ferric-scf/tests/validation_density_properties.rs  (ESP, field, volumes)
    crates/ferric-rpa/tests/validation_static_alpha.rs        (static alpha)

THE DESIGN: SAME DENSITY, TWO CODES
-----------------------------------
A property compared "ferric SCF -> property" against "PySCF SCF -> property"
mixes the SCF convergence error into the property error (the old 1e-4 a.u.
ESP bar did exactly that). So every file here carries PySCF's converged
density (or MOs) PERMUTED INTO FERRIC'S AO ORDER, and the Rust test feeds that
same density to ferric's property function. That comparison isolates the
property code (integrals / grid / contraction) and is expected at ~1e-10. The
Rust test ALSO runs ferric's own SCF and compares against the same reference
numbers at a looser bar (the full chain, ~1e-6).

AO ORDER (the one convention this design depends on)
----------------------------------------------------
ferric's AO order is atom-major, shells in the bundled JSON's order with each
general-contraction column split into its own shell (`common.ferric_shells`),
p as (x, y, z), spherical l>=2 as m = -l..+l (libint2 standard). PySCF sorts
each atom's shells by l (stable within an l), has p as (x, y, z) and spherical
l>=2 as m = -l..+l. `ferric_ao_permutation` matches the k-th ferric shell of
(atom, l) to the k-th PySCF shell of (atom, l), checks the non-zero-coefficient
exponents agree, and records `perm[i]` = PySCF AO index of ferric AO i.

The permutation is TESTED, not trusted: each file carries PySCF's AO overlap
matrix in ferric order, and the Rust test asserts it equals ferric's own
overlap elementwise (~1e-12). A wrong shell match, a swapped m-component or a
sign-convention difference in a solid harmonic moves off-diagonal elements by
O(0.1) and fails there, before any property is compared.

WHAT EACH REFERENCE IS
----------------------
* ESP at nucleus A (a.u.): sum_{B!=A} Z_B/R_AB - sum D_uv <u|1/|r-R_A||v>,
  PySCF `int1e_rinv` under `with_rinv_origin(R_A)` — ferric's convention
  (`ferric_scf::properties::esp_at_atoms`: no self term).
* Field at nucleus A: E = -grad V. Electronic part = sum D (ip + ip^T) with
  ip = PySCF `int1e_iprinv` = <grad u|1/|r-R_A||v> (derivation: translational
  invariance, d/dR = -(d/dA + d/dB)); nuclear part sum_{B!=A} Z_B (R_A-R_B)/R^3.
  The generator also central-differences its OWN ESP in the rinv origin and
  REFUSES to write if the analytic field disagrees (catches a sign/factor slip
  in this script before it becomes a reference).
* Becke volume v_A = sum_{g: home=A} w_g rho(r_g) |r_g - R_A|^3 on FERRIC'S
  grid, rebuilt here from PySCF's own primitives: Treutler-Ahlrichs M4 radial
  nodes (`pyscf.dft.radi.treutler_ahlrichs`, the table ferric copied), PySCF's
  Lebedev rule (`MakeAngularGrid`), and PySCF's AO values (`eval_ao`). The
  PARTITION is ferric's: Becke 1988 with the size adjustment of Eq. A4 on the
  Bragg-Slater radii in `ferric-dft/src/becke.rs` (chi = R_A/R_B, a = u/(u^2-1)
  clipped to +-1/2). PySCF's own `becke_atomic_radii_adjust` uses a different
  radii table and adjustment, so it is deliberately NOT used: this row checks
  the quadrature, the AO evaluation and the contraction, and the partition is a
  DEFINITION reimplemented from its documented formula, not an approximation
  with an independent reference. The same volumes on a dense (200, 590) grid
  are stored as `volumes_dense_grid` so the test can report ferric's
  default-grid quadrature error (measurement, not asserted).
* Static alpha: what `ferric_rpa::properties::pdep_polarizability_static`
  computes is the closed-shell DIRECT-RPA (time-dependent Hartree, no exchange
  kernel) static polarizability with an RI Coulomb kernel:
      alpha_xy = 4 mu_x^T (Delta + 4 K)^{-1} mu_y,
      K_ia,jb = (ia|P) V^{-1} (P|jb) over the RI aux basis, no frozen core,
      mu_ia = <i|r|a> (origin 0), Delta = eps_a - eps_i.
  ferric solves it in the aux space by Sherman-Morrison-Woodbury; this script
  solves the ov-space linear system directly (an independent algebraic route).
  Also stored, for SCOPE (not what ferric computes): the same dRPA alpha with
  exact 4-index integrals (`alpha_drpa_exact_eri`, the RI error size) and the
  coupled-perturbed HF alpha with the exchange kernel (`alpha_cphf`), which the
  Rust test asserts ferric MISSES — so the row cannot be re-read as "CPHF".
* Free atoms (Z = 1, 6, 7, 8, aug-cc-pVDZ): UKS-PBE on ferric's (75, 110)
  TA-M4 grid, exact J (ferric's `RhfConfig` default). H and N are integer-
  occupation states followed through a `stability()` loop; C and O are the
  spherically averaged fractional-occupation ensemble ferric's TS branch uses
  (`fractional_occ`), reproduced with `scf.addons.frac_occ(tol=0.05)` — ferric's
  `density_fractional` grouping tolerance. An ensemble has no stability
  analysis; that is recorded, not skipped silently.

Run (light steps; seconds to a few minutes):
    scripts/validation/run_slot.sh --light -- \\
        uv run --no-sync python scripts/validation/gen_properties.py density
    scripts/validation/run_slot.sh --light -- \\
        uv run --no-sync python scripts/validation/gen_properties.py atoms
    scripts/validation/run_slot.sh --light -- \\
        uv run --no-sync python scripts/validation/gen_properties.py alpha
    (or `all`; a trailing system name restricts, e.g. `density ho2`)
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402

GENERATOR = "scripts/validation/gen_properties.py"

ROW_DENSITY = "density_properties"
ROW_ATOMS = "free_atom_volumes"
ROW_ALPHA = "static_alpha"

# system -> (charge, multiplicity, reference)
DENSITY_SYSTEMS = {
    "h2o": (0, 1, "rhf"),
    "ch3oh": (0, 1, "rhf"),
    "ho2": (0, 2, "uhf"),
}
DENSITY_BASES = ("cc-pvdz", "def2-svp")

# symbol -> multiplicity (ferric's gs_mult for the TS free-atom branch)
FREE_ATOMS = {"h": 2, "c": 3, "n": 4, "o": 3}
FREE_ATOM_BASIS = "aug-cc-pvdz"
FREE_ATOM_XC = "PBE"
FRAC_OCC_TOL = 0.05  # ferric-scf uhf.rs density_fractional EPS_TOL

# (system, orbital basis, RI aux basis)
ALPHA_CASES = (
    ("h2o", "aug-cc-pvdz", "aug-cc-pvdz-rifit"),
    ("ch3oh", "cc-pvdz", "cc-pvdz-ri"),
)
# An aux basis the Rust test swaps in as a negative control (not a reference).
ALPHA_SWAP_AUX = "def2-universal-jkfit"

CONV_TOL = 1e-12
CONV_TOL_GRAD = 1e-9

# ferric's default property/XC grid: AtomicGridConfig::default()
FERRIC_GRID = (75, 110)
DENSE_GRID = (200, 590)

# crates/ferric-dft/src/becke.rs::bragg_slater_bohr, VERBATIM (Angstrom, and
# ferric's own Angstrom->Bohr factor in that function, which differs from
# ANGSTROM_TO_BOHR in mol.rs in the 11th digit — reproduced, not "fixed").
_BRAGG_SLATER_ANG = {
    1: 0.35, 2: 0.30, 3: 1.45, 4: 1.05, 5: 0.85, 6: 0.70, 7: 0.65, 8: 0.60,
    9: 0.50, 10: 0.45, 11: 1.80, 12: 1.50, 13: 1.25, 14: 1.10, 15: 1.00,
    16: 1.00, 17: 1.00, 18: 0.71,
}  # fmt: skip
_BECKE_RS_ANG_TO_BOHR = 1.8897259886

FD_STEP = 1e-4  # Bohr, rinv-origin central difference for the field self-check
FD_FIELD_TOL = 1e-6


# ---------------------------------------------------------------------------
# AO order: ferric <- PySCF
# ---------------------------------------------------------------------------


def ferric_ao_permutation(mol, basis_name: str, symbols: list[str]) -> list[int]:
    """`perm[i]` = PySCF AO index of ferric AO `i` (see module docstring)."""
    import numpy as np

    ao_loc = mol.ao_loc_nr()
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for ib in range(mol.nbas):
        if mol.bas_nctr(ib) != 1:
            raise ValueError(
                f"PySCF shell {ib} has nctr={mol.bas_nctr(ib)}; expected segmented shells"
            )
        groups[(mol.bas_atom(ib), mol.bas_angular(ib))].append(ib)
    used: dict[tuple[int, int], int] = defaultdict(int)
    perm: list[int] = []
    for ia, sym in enumerate(symbols):
        for sh in common.ferric_shells(basis_name, common.z_of(sym)):
            ell = sh["l"]
            key = (ia, ell)
            k = used[key]
            used[key] += 1
            if k >= len(groups[key]):
                raise ValueError(
                    f"{basis_name}: ferric has more l={ell} shells on atom {ia}"
                )
            ib = groups[key][k]
            fe = sorted(e for e, c in zip(sh["exps"], sh["coefs"]) if c != 0.0)
            pc = mol.bas_ctr_coeff(ib)[:, 0]
            pe = sorted(e for e, c in zip(mol.bas_exp(ib), pc) if c != 0.0)
            if len(fe) != len(pe) or not np.allclose(fe, pe, rtol=1e-12, atol=0.0):
                raise ValueError(
                    f"{basis_name} atom {ia} l={ell} shell #{k}: exponents differ "
                    f"(ferric {fe} vs PySCF {pe})"
                )
            nf_ferric = (
                (2 * ell + 1) if (sh["pure"] or ell < 2) else (ell + 1) * (ell + 2) // 2
            )
            nf_pyscf = int(ao_loc[ib + 1] - ao_loc[ib])
            if nf_ferric != nf_pyscf:
                raise ValueError(
                    f"{basis_name} atom {ia} l={ell}: {nf_ferric} ferric vs {nf_pyscf} PySCF functions"
                )
            perm.extend(range(int(ao_loc[ib]), int(ao_loc[ib + 1])))
    for key, lst in groups.items():
        if used[key] != len(lst):
            raise ValueError(f"PySCF shells left unmatched for (atom, l) = {key}")
    if sorted(perm) != list(range(mol.nao_nr())):
        raise ValueError("AO map is not a permutation")
    return perm


def to_ferric_order(mat, perm):
    """Symmetric AO matrix (PySCF order) -> ferric order."""
    import numpy as np

    p = np.asarray(perm)
    return mat[np.ix_(p, p)]


def mat_json(a) -> list:
    return [[float(x) for x in row] for row in a]


# ---------------------------------------------------------------------------
# SCF drivers
# ---------------------------------------------------------------------------


def run_rhf(mol) -> tuple[dict, object]:
    from pyscf import scf

    mf = scf.RHF(mol)
    mf.conv_tol = CONV_TOL
    mf.conv_tol_grad = CONV_TOL_GRAD
    mf.max_cycle = 500
    mf.verbose = 0
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("RHF did not converge")
    _mo, stable = common._stability_status(mf)
    if not stable:
        raise RuntimeError("RHF reference is internally unstable; refusing to write it")
    info = {
        "energy": float(mf.e_tot),
        "converged": True,
        "stability": {"internal_stable": True, "kind": "PySCF RHF internal (real)"},
    }
    return info, mf


def ferric_like_grids(mol, level):
    """PySCF `Grids` configured as ferric's XC grid (flat TA-M4 x Lebedev)."""
    from pyscf import dft

    g = dft.gen_grid.Grids(mol)
    g.atom_grid = level
    g.radi_method = dft.radi.treutler_ahlrichs
    g.prune = None
    g.becke_scheme = dft.gen_grid.original_becke
    return g


def run_uks_atom(mol, mult: int) -> tuple[dict, object]:
    """Free-atom UKS-PBE, integer occupation + stability loop, or the
    fractional-occupation ensemble when the frontier shell is degenerate."""
    from pyscf import dft, scf

    na, nb = mol.nelec
    # Ensemble iff a spin's valence p shell is partially filled (C: a 2/3,
    # O: b 2/3). H (1s1) and N (p3 alpha, p0 beta) are integer states.
    use_frac = mol.atom_charge(0) in (6, 8)

    def make():
        mf = dft.UKS(mol)
        mf.xc = FREE_ATOM_XC
        mf.grids = ferric_like_grids(mol, FERRIC_GRID)
        mf.conv_tol = CONV_TOL
        mf.conv_tol_grad = CONV_TOL_GRAD
        mf.max_cycle = 500
        mf.verbose = 0
        if use_frac:
            mf = scf.addons.frac_occ(mf, tol=FRAC_OCC_TOL)
        return mf

    mf = make()
    mf.kernel()
    if not mf.converged:
        raise RuntimeError(f"UKS free atom Z={mol.atom_charge(0)} did not converge")
    if use_frac:
        stability = {
            "internal_stable": None,
            "kind": "not applicable: fractional-occupation (spherical) ensemble",
            "frac_occ_tol": FRAC_OCC_TOL,
        }
    else:
        rounds = 0
        while True:
            mo_i, stable = common._stability_status(mf)
            if stable or rounds >= 10:
                break
            mf.kernel(dm0=mf.make_rdm1(mo_i, mf.mo_occ))
            rounds += 1
        if not stable:
            raise RuntimeError(f"UKS free atom Z={mol.atom_charge(0)}: no stable state")
        stability = {
            "internal_stable": True,
            "kind": "PySCF UKS internal (real)",
            "rounds": rounds,
        }
    s2, _ = mf.spin_square()
    occ_a, occ_b = mf.mo_occ
    info = {
        "energy": float(mf.e_tot),
        "converged": True,
        "xc": FREE_ATOM_XC,
        "fractional_occupation": use_frac,
        "nelec_alpha": int(na),
        "nelec_beta": int(nb),
        "s_squared": float(s2),
        "occupations_alpha": [float(x) for x in occ_a if x > 0],
        "occupations_beta": [float(x) for x in occ_b if x > 0],
        "stability": stability,
    }
    return info, mf


# ---------------------------------------------------------------------------
# ESP and field at the nuclei
# ---------------------------------------------------------------------------


def _nuc_esp(zs, xyz, a):
    import numpy as np

    v = 0.0
    for b in range(len(zs)):
        if b != a:
            v += zs[b] / np.linalg.norm(xyz[a] - xyz[b])
    return v


def _nuc_field(zs, xyz, a):
    import numpy as np

    e = np.zeros(3)
    for b in range(len(zs)):
        if b != a:
            d = xyz[a] - xyz[b]
            e += zs[b] * d / np.linalg.norm(d) ** 3
    return e


def esp_and_field(mol, dm) -> dict:
    """ESP and field at every nucleus from AO density `dm` (PySCF order)."""
    import numpy as np

    zs = [float(mol.atom_charge(i)) for i in range(mol.natm)]
    xyz = np.asarray(mol.atom_coords(unit="Bohr"))

    def v_elec_at(origin):
        with mol.with_rinv_origin(origin):
            t = mol.intor("int1e_rinv")
        return -float(np.einsum("ij,ij->", dm, t))

    esp, esp_el, field, field_el, fd_dev = [], [], [], [], 0.0
    for a in range(mol.natm):
        ve = v_elec_at(xyz[a])
        esp_el.append(ve)
        esp.append(ve + _nuc_esp(zs, xyz, a))
        with mol.with_rinv_origin(xyz[a]):
            ip = mol.intor("int1e_iprinv", comp=3)
        e_el = np.array(
            [float(np.einsum("ij,ij->", dm, ip[d] + ip[d].T)) for d in range(3)]
        )
        # Self-check: E_el = -dV_el/dR by central differences in the origin.
        e_fd = np.empty(3)
        for d in range(3):
            step = np.zeros(3)
            step[d] = FD_STEP
            e_fd[d] = -(v_elec_at(xyz[a] + step) - v_elec_at(xyz[a] - step)) / (
                2 * FD_STEP
            )
        fd_dev = max(fd_dev, float(np.max(np.abs(e_el - e_fd))))
        field_el.append(e_el.tolist())
        field.append((e_el + _nuc_field(zs, xyz, a)).tolist())
    if fd_dev > FD_FIELD_TOL:
        raise RuntimeError(
            f"analytic field (int1e_iprinv) disagrees with the FD of int1e_rinv by "
            f"{fd_dev:.2e} > {FD_FIELD_TOL:.0e}: sign/factor slip in this generator"
        )
    return {
        "esp_at_nuclei": esp,
        "esp_electronic_at_nuclei": esp_el,
        "electric_field_at_nuclei": field,
        "electric_field_electronic_at_nuclei": field_el,
        "field_vs_fd_of_esp_max_abs": fd_dev,
        "fd_step_bohr": FD_STEP,
    }


# ---------------------------------------------------------------------------
# Becke effective volumes on ferric's grid
# ---------------------------------------------------------------------------


def _bragg_slater_bohr(z: int) -> float:
    return _BRAGG_SLATER_ANG.get(z, 1.00) * _BECKE_RS_ANG_TO_BOHR


def becke_partition(zs, xyz, pts):
    """(natm, npts) Becke weights with ferric's size adjustment (becke.rs)."""
    import numpy as np

    natm = len(zs)
    if natm == 1:
        return np.ones((1, len(pts)))
    r = np.linalg.norm(pts[None, :, :] - xyz[:, None, :], axis=2)
    cell = np.ones((natm, len(pts)))
    for a in range(natm):
        ra = _bragg_slater_bohr(zs[a])
        for b in range(natm):
            if a == b:
                continue
            rab = np.linalg.norm(xyz[a] - xyz[b])
            if rab < 1e-12:
                continue
            mu = (r[a] - r[b]) / rab
            chi = ra / _bragg_slater_bohr(zs[b])
            u = (chi - 1.0) / (chi + 1.0)
            acorr = min(max(u / (u * u - 1.0), -0.5), 0.5)
            x = mu + acorr * (1.0 - mu * mu)
            for _ in range(3):
                x = 0.5 * x * (3.0 - x * x)
            cell[a] *= 0.5 * (1.0 - x)
    tot = cell.sum(axis=0)
    w = np.where(tot < 1e-30, 0.0, cell / np.where(tot < 1e-30, 1.0, tot))
    return w


def atom_centred_grid(zs, xyz, n_rad, n_ang):
    """(home, pts, w_rl) of ferric's flat grid: TA-M4 radial (weights incl.
    4 pi r^2) x Lebedev (weights normalised to sum 1)."""
    import numpy as np
    from pyscf.dft import radi
    from pyscf.dft.LebedevGrid import MakeAngularGrid

    ang = np.asarray(MakeAngularGrid(n_ang))
    if ang.shape != (n_ang, 4):
        raise ValueError(f"Lebedev {n_ang}: got shape {ang.shape}")
    w_ang = ang[:, 3] / ang[:, 3].sum()
    homes, pts, wts = [], [], []
    for a, z in enumerate(zs):
        rad, dr = radi.treutler_ahlrichs(n_rad, int(z))
        w_rad = 4.0 * np.pi * rad**2 * dr
        p = xyz[a][None, None, :] + rad[:, None, None] * ang[None, :, :3]
        pts.append(p.reshape(-1, 3))
        wts.append((w_rad[:, None] * w_ang[None, :]).ravel())
        homes.append(np.full(n_rad * n_ang, a))
    return np.concatenate(homes), np.concatenate(pts), np.concatenate(wts)


def becke_volumes(mol, dm, level, chunk=50_000) -> list[float]:
    """v_A = sum_{g: home=A} w_rl(g) w_A(g) rho(g) |g - R_A|^3 (PySCF-order dm)."""
    import numpy as np
    from pyscf.dft import numint

    zs = [int(mol.atom_charge(i)) for i in range(mol.natm)]
    xyz = np.asarray(mol.atom_coords(unit="Bohr"))
    home, pts, w_rl = atom_centred_grid(zs, xyz, *level)
    vol = np.zeros(mol.natm)
    for g0 in range(0, len(pts), chunk):
        g1 = min(g0 + chunk, len(pts))
        p = pts[g0:g1]
        h = home[g0:g1]
        ao = numint.eval_ao(mol, p, deriv=0)
        rho = np.einsum("pi,ij,pj->p", ao, dm, ao)
        wb = becke_partition(zs, xyz, p)[h, np.arange(g1 - g0)]
        r3 = np.linalg.norm(p - xyz[h], axis=1) ** 3
        np.add.at(vol, h, w_rl[g0:g1] * wb * rho * r3)
    return [float(v) for v in vol]


def volumes_block(mol, dm) -> dict:
    v = becke_volumes(mol, dm, FERRIC_GRID)
    vd = becke_volumes(mol, dm, DENSE_GRID)
    rel = max(abs(a - b) / abs(b) for a, b in zip(v, vd))
    return {
        "volumes": v,
        "volumes_dense_grid": vd,
        "ferric_grid_vs_dense_max_rel": rel,
        "grid": {
            "radial": "Treutler-Ahlrichs M4 (pyscf.dft.radi.treutler_ahlrichs, atom-specific xi)",
            "n_radial": FERRIC_GRID[0],
            "n_angular": FERRIC_GRID[1],
            "prune": "none",
            "partition": "Becke 1988, 3 smoothing iterations, Eq. A4 size adjustment on "
            "ferric-dft becke.rs Bragg-Slater radii (chi = R_A/R_B)",
            "dense_grid": list(DENSE_GRID),
        },
    }


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------


def gen_density(only: set[str]) -> list[Path]:
    import numpy
    import pyscf

    written = []
    for system, (charge, mult, ref) in DENSITY_SYSTEMS.items():
        if only and system not in only:
            continue
        xyz = common.MOL_DIR / f"{system}.xyz"
        symbols, coords = common.read_xyz(xyz)
        for basis_name in DENSITY_BASES:
            mol = common.build_pyscf_mol(
                xyz, basis_name, charge=charge, multiplicity=mult
            )
            basis_check = common.check_basis_like_for_like(mol, basis_name, symbols)
            if ref == "rhf":
                scf_info, mf = run_rhf(mol)
            else:
                scf_info, mf = common.run_open_shell(
                    mol,
                    "uhf",
                    conv_tol=CONV_TOL,
                    conv_tol_grad=CONV_TOL_GRAD,
                    return_mf=True,
                )
            dm = mf.make_rdm1()
            if dm.ndim == 3:
                dm = dm[0] + dm[1]
            perm = ferric_ao_permutation(mol, basis_name, symbols)
            s_f = to_ferric_order(mol.intor("int1e_ovlp"), perm)
            payload = {
                "row": "ESP at nuclei / Electric field at nuclei / Becke volumes",
                "system": system,
                "basis": basis_name,
                "charge": charge,
                "multiplicity": mult,
                "reference": ref,
                "nao": mol.nao_nr(),
                "nelectron": int(mol.nelectron),
                "nuclear_repulsion": float(mol.energy_nuc()),
                "scf": scf_info,
                "ao_permutation_ferric_to_pyscf": perm,
                "overlap_ferric_order": mat_json(s_f),
                "density_total_ferric_order": mat_json(to_ferric_order(dm, perm)),
                **esp_and_field(mol, dm),
                "becke": volumes_block(mol, dm),
                "provenance": common.provenance(
                    code="PySCF + numpy",
                    version=pyscf.__version__,
                    keywords={
                        "scf": "scf.RHF"
                        if ref == "rhf"
                        else "scf.UHF + stability loop",
                        "esp": "int1e_rinv with_rinv_origin(R_A), point nuclei, no self term",
                        "field": "int1e_iprinv (ip + ip^T), FD-checked against int1e_rinv",
                        "eri": "exact 4-index (no density fitting)",
                        "numpy": numpy.__version__,
                    },
                    basis_name=basis_name,
                    xyz_path=xyz,
                    coords_bohr=coords,
                    symbols=symbols,
                    grid={"volumes": list(FERRIC_GRID), "dense": list(DENSE_GRID)},
                    aux=None,
                    frozen_core=None,
                    scf_conv={"conv_tol": CONV_TOL, "conv_tol_grad": CONV_TOL_GRAD},
                    stability=scf_info["stability"],
                    generator=GENERATOR,
                    extra={"basis_self_check": basis_check},
                ),
            }
            written.append(
                common.write_reference(ROW_DENSITY, system, basis_name, payload)
            )
            print(
                f"{system:6s} {basis_name:9s} E={scf_info['energy']:.10f} "
                f"ESP[0]={payload['esp_at_nuclei'][0]:+.10f} "
                f"fd_dev={payload['field_vs_fd_of_esp_max_abs']:.1e} "
                f"vol_grid_rel={payload['becke']['ferric_grid_vs_dense_max_rel']:.1e}"
            )
    return written


def gen_atoms(only: set[str]) -> list[Path]:
    import numpy
    import pyscf

    written = []
    for sym, mult in FREE_ATOMS.items():
        system = f"{sym}_atom"
        if only and system not in only and sym not in only:
            continue
        xyz = common.MOL_DIR / f"{system}.xyz"
        symbols, coords = common.read_xyz(xyz)
        mol = common.build_pyscf_mol(xyz, FREE_ATOM_BASIS, multiplicity=mult)
        basis_check = common.check_basis_like_for_like(mol, FREE_ATOM_BASIS, symbols)
        info, mf = run_uks_atom(mol, mult)
        dm = mf.make_rdm1()
        dm = dm[0] + dm[1]
        perm = ferric_ao_permutation(mol, FREE_ATOM_BASIS, symbols)
        payload = {
            "row": "Becke volumes (free atoms)",
            "system": system,
            "basis": FREE_ATOM_BASIS,
            "charge": 0,
            "multiplicity": mult,
            "nao": mol.nao_nr(),
            "nuclear_repulsion": 0.0,
            "uks": info,
            "ao_permutation_ferric_to_pyscf": perm,
            "overlap_ferric_order": mat_json(
                to_ferric_order(mol.intor("int1e_ovlp"), perm)
            ),
            "density_total_ferric_order": mat_json(to_ferric_order(dm, perm)),
            "becke": volumes_block(mol, dm),
            "provenance": common.provenance(
                code="PySCF + numpy",
                version=pyscf.__version__,
                keywords={
                    "scf": "dft.UKS xc=PBE"
                    + (
                        f" + scf.addons.frac_occ(tol={FRAC_OCC_TOL})"
                        if info["fractional_occupation"]
                        else ""
                    ),
                    "coulomb": "exact J (no density fitting), as ferric RhfConfig default",
                    "numpy": numpy.__version__,
                },
                basis_name=FREE_ATOM_BASIS,
                xyz_path=xyz,
                coords_bohr=coords,
                symbols=symbols,
                grid={
                    "xc": f"{FERRIC_GRID} TA-M4 x Lebedev, unpruned",
                    "volumes": list(FERRIC_GRID),
                    "dense": list(DENSE_GRID),
                },
                aux=None,
                frozen_core=None,
                scf_conv={"conv_tol": CONV_TOL, "conv_tol_grad": CONV_TOL_GRAD},
                stability=info["stability"],
                generator=GENERATOR,
                extra={"basis_self_check": basis_check},
            ),
        }
        written.append(
            common.write_reference(ROW_ATOMS, system, FREE_ATOM_BASIS, payload)
        )
        print(
            f"{system:7s} E={info['energy']:.10f} frac={info['fractional_occupation']} "
            f"<S2>={info['s_squared']:.6f} v={payload['becke']['volumes'][0]:.6f} "
            f"grid_rel={payload['becke']['ferric_grid_vs_dense_max_rel']:.1e}"
        )
    return written


def _aux_mol(mol, aux_name: str, symbols, coords):
    from pyscf import gto

    basis, cart = common.pyscf_basis(aux_name, symbols)
    if cart:
        raise ValueError(
            f"{aux_name}: Cartesian l>=2 aux shells; not matched like-for-like"
        )
    auxmol = gto.M(
        atom=common.pyscf_atom_bohr(symbols, coords),
        unit="Bohr",
        basis=basis,
        cart=False,
        charge=mol.charge,
        spin=mol.spin,
        verbose=0,
    )
    return auxmol


def _alpha(mu, delta, kernel):
    """4 mu^T (Delta + kernel)^{-1} mu for mu of shape (3, nov)."""
    import numpy as np

    m = np.diag(delta) + kernel
    x = np.linalg.solve(m, mu.T)
    a = 4.0 * mu @ x
    return 0.5 * (a + a.T)


def gen_alpha(only: set[str]) -> list[Path]:
    import numpy as np
    import pyscf
    import scipy.linalg
    from pyscf import ao2mo, df

    written = []
    for system, basis_name, aux_name in ALPHA_CASES:
        if only and system not in only:
            continue
        xyz = common.MOL_DIR / f"{system}.xyz"
        symbols, coords = common.read_xyz(xyz)
        mol = common.build_pyscf_mol(xyz, basis_name)
        basis_check = common.check_basis_like_for_like(mol, basis_name, symbols)
        scf_info, mf = run_rhf(mol)
        auxmol = _aux_mol(mol, aux_name, symbols, coords)
        aux_check = common.check_basis_like_for_like(auxmol, aux_name, symbols)

        nocc = mol.nelectron // 2
        c = mf.mo_coeff
        eps = mf.mo_energy
        co, cv = c[:, :nocc], c[:, nocc:]
        nvir = cv.shape[1]
        nov = nocc * nvir
        delta = (eps[nocc:][None, :] - eps[:nocc][:, None]).ravel()
        mu_ao = mol.intor("int1e_r")  # <u|r|v>, origin (0,0,0)
        mu = np.einsum("xuv,ui,va->xia", mu_ao, co, cv).reshape(3, nov)

        # RI Coulomb kernel with the SAME aux ferric uses: K = (ia|P) V^-1 (P|jb).
        int3c = df.incore.aux_e2(mol, auxmol, intor="int3c2e", aosym="s1")
        iap = np.einsum("uvP,ui,va->iaP", int3c, co, cv).reshape(nov, -1)
        v2c = auxmol.intor("int2c2e")
        low = np.linalg.cholesky(v2c)
        b = scipy.linalg.solve_triangular(low, iap.T, lower=True)  # (naux, nov)
        k_ri = b.T @ b
        alpha_ri = _alpha(mu, delta, 4.0 * k_ri)

        # Scope: exact-ERI dRPA and CPHF (exchange kernel), for context.
        eri = ao2mo.kernel(mol, c, compact=False).reshape([c.shape[1]] * 4)
        ovov = eri[:nocc, nocc:, :nocc, nocc:].reshape(nov, nov)
        alpha_exact = _alpha(mu, delta, 4.0 * ovov)
        ibja = eri[:nocc, nocc:, :nocc, nocc:].transpose(0, 3, 2, 1).reshape(nov, nov)
        ijab = eri[:nocc, :nocc, nocc:, nocc:].transpose(0, 2, 1, 3).reshape(nov, nov)
        alpha_cphf = _alpha(mu, delta, 4.0 * ovov - ibja - ijab)

        perm = ferric_ao_permutation(mol, basis_name, symbols)
        c_f = c[np.asarray(perm), :]
        payload = {
            "row": "Static alpha",
            "system": system,
            "basis": basis_name,
            "aux_basis": aux_name,
            "swap_aux_negative_control": ALPHA_SWAP_AUX,
            "charge": 0,
            "multiplicity": 1,
            "nao": mol.nao_nr(),
            "naux": auxmol.nao_nr(),
            "nocc": nocc,
            "nuclear_repulsion": float(mol.energy_nuc()),
            "scf": scf_info,
            "ao_permutation_ferric_to_pyscf": perm,
            "overlap_ferric_order": mat_json(
                to_ferric_order(mol.intor("int1e_ovlp"), perm)
            ),
            "mo_coeff_ferric_order": mat_json(c_f),
            "mo_energy": [float(x) for x in eps],
            "alpha_drpa_ri": mat_json(alpha_ri),
            "alpha_drpa_exact_eri": mat_json(alpha_exact),
            "alpha_cphf": mat_json(alpha_cphf),
            "definition": "alpha = 4 mu^T (Delta + 4 K_RI)^-1 mu, K_RI = (ia|P)V^-1(P|jb), "
            "no frozen core, mu = <i|r|a> about the origin (direct RPA / TDH, no exchange)",
            "provenance": common.provenance(
                code="PySCF + numpy",
                version=pyscf.__version__,
                keywords={
                    "scf": "scf.RHF, exact 4-index",
                    "alpha": "dense ov-space solve (not SMW), symmetrised",
                    "ri": "df.incore.aux_e2 int3c2e + Cholesky of int2c2e (Coulomb metric)",
                    "numpy": np.__version__,
                },
                basis_name=basis_name,
                xyz_path=xyz,
                coords_bohr=coords,
                symbols=symbols,
                grid=None,
                aux=aux_name,
                frozen_core=0,
                scf_conv={"conv_tol": CONV_TOL, "conv_tol_grad": CONV_TOL_GRAD},
                stability=scf_info["stability"],
                generator=GENERATOR,
                extra={"basis_self_check": basis_check, "aux_self_check": aux_check},
            ),
        }
        written.append(common.write_reference(ROW_ALPHA, system, basis_name, payload))
        iso = lambda a: float(np.trace(a)) / 3.0  # noqa: E731
        print(
            f"{system:6s} {basis_name:12s} aux={aux_name:18s} "
            f"iso dRPA-RI {iso(alpha_ri):.8f}  exact {iso(alpha_exact):.8f}  "
            f"CPHF {iso(alpha_cphf):.8f}"
        )
    return written


def main(argv: list[str]) -> int:
    rows = {"density": gen_density, "atoms": gen_atoms, "alpha": gen_alpha}
    what = argv[1] if len(argv) > 1 else "all"
    only = set(argv[2:])
    if what not in (*rows, "all"):
        print(__doc__)
        return 2
    written = []
    for name, fn in rows.items():
        if what in (name, "all"):
            written += fn(only)
    print(f"GEN_PROPERTIES_DONE written={len(written)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
