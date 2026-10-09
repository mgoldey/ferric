"""pymbd / libmbd references for the VALIDATION.md rows "TS C6" (153) and
"MBD@TS" (156).

    ts_mbd/<system>_<basis>.json   per-system TS + MBD numbers from identical inputs
    ts_mbd/anchors_model.json      model-level exactness anchors (no molecule, no basis)

Consumer: crates/ferric-rpa/tests/validation_ts_mbd.rs

WHAT FERRIC COMPUTES (read from the loops in crates/ferric-rpa/src/dispersion.rs
and dispersion/mbd.rs, not the doc comments)
---------------------------------------------------------------------------
TS (`ts_atom_params` + `ts_dynamic_polarizability` + `casimir_polder_c6`):

    alpha_A = r_A * alpha_free[Z]      C6_A = r_A^2 * C6_free[Z]
    omega_A = (4/3) C6_A / alpha_A^2
    alpha_A(iu) = alpha_A / (1 + (u/omega_A)^2)          (times a unit-iso shape tensor)
    C6_AB = (3/pi) sum_k w_k alpha_A(iu_k) alpha_B(iu_k)  (QUADRATURE, every pair)

The TS combination rule 2 C6_A C6_B / (a_B/a_A C6_A + a_A/a_B C6_B) is the EXACT
value of that integral for single-pole oscillators, so ferric's pair C6 equals
the combination rule up to the frequency quadrature error only. ferric has NO
TS pairwise ENERGY (no sR/d Fermi-damped sum) — nothing to compare there.

MBD (`mbd_screen`, `mbd_dynamic_polarizability`, `mbd_energy`):

    sigma_A(iu) = (sqrt(2/pi) alpha_A(iu) / 3)^(1/3)
    T_AB = Gaussian-damped dipole tensor, sigma_AB = sqrt(sigma_A^2 + sigma_B^2),
           NO Fermi damping, NO beta, NO R_vdw   (pymbd `T_erf_coulomb`; libmbd 'dip,gg')
    alpha_A^scs(iu) = sum_B [ (diag(1/alpha(iu)) + T)^-1 ]_AB         (full 3x3)
    E_MBD = 1/2 sum sqrt(eig(H)) - 3/2 sum_A omega_A,
        H = diag(omega^2) + omega_A omega_B sqrt(alpha_A alpha_B) T(sigma(alpha_static))
        with the UNSCREENED TS alpha/omega.

That is the plain-SCS screening (libmbd variant='scs') and the plain MBD energy
with 'dip,gg' damping (libmbd variant='plain'). It is NOT MBD@rsSCS (Ambrosetti
2014: range-separated screening with the Fermi (beta, R_vdw) split, then a
Fermi-damped bare-dipole energy from the SCREENED alpha/C6/R_vdw), which is what
pymbd.mbd_energy / mbd_energy_species compute. For those functions the rsSCS
numbers (`mbd/scope_rsscs`) are a SCOPE control they must MISS.

MBD@rsSCS (`dispersion::mbd_rsscs::mbd_rsscs_energy`) is a separate ferric
function; its references are `mbd_rsscs` in each system file (consumer:
crates/ferric-rpa/tests/validation_mbd_rsscs.rs). Two constructions, required
to agree to < 1e-10: pymbd python (`screening` + `mbd_energy`) and libmbd
Fortran (variant='rsscs', with its alpha_0_scs/C6_scs intermediates). Stored
for beta = 0.83 (PBE) and 0.85 (PBE0/HSE06): alpha_0^rsSCS, C6^rsSCS,
R_vdw^rsSCS, omega^rsSCS, E; plus E on 30- and 60-node grids (grid
sensitivity, pymbd primitives — pymbd.mbd_energy hard-codes nfreq=15).
Each run also stores `gradient`: libmbd's analytic nuclear gradient
(`mbd_energy(..., variant='rsscs', force=True)`) as dE/dR (N x 3, Hartree/Bohr,
alpha_0/C6/R_vdw i.e. the volume ratios held FIXED). Whether libmbd's array is
dE/dR or the force -dE/dR is DECIDED HERE by a central finite difference of
libmbd's own energy (h = RSSCS_GRAD_FD_H Bohr, every component), asserted, and
recorded with its residuals in `gradient_fd_check`.
R_vdw(TS) is ferric's `ts_free_atom_r_vdw` table, parsed from
free_atom_ref.rs and required equal to pymbd's `R_vdw(TS)` for every Z=1..54.

ferric's erf is Abramowitz-Stegun 7.1.26 (|err| < 1.5e-7). Every MBD quantity is
therefore also computed here with THAT erf (`*_as_erf`), and the gap between the
two is stored as the predicted floor of a ferric-vs-reference comparison.

THREE INDEPENDENT CONSTRUCTIONS of the MBD target
--------------------------------------------------
(1) ferric (Rust); (2) numpy on pymbd's own primitives (`pymbd.pymbd.T_erf_coulomb`,
scipy erf); (3) libmbd (Fortran, through `pymbd.fortran.MBDGeom`): variant='scs'
intermediates give the static screened alpha and C6_AA on libmbd's 15-node grid,
variant='plain' damping='dip,gg' gives the energy. (2) vs (3) is recorded in
`measurements` and must be ~1e-12 or the reference is refused.

INPUTS (identical on both sides by construction)
------------------------------------------------
The Rust test reads Z, coordinates (Bohr), volume ratios, alpha_free/C6_free AND
the frequency nodes/weights from this JSON; nothing is recomputed on the ferric
side except the model. alpha_free/C6_free are ferric's table, parsed from
crates/ferric-rpa/src/dispersion/free_atom_ref.rs, and checked equal to pymbd's
`vdw_params` (TS columns) for every element used; the full Z=1..54 table diff
goes in anchors_model.json.

Volume ratios r_A = v_A / v_free,A are built the way ferric-cli builds them
(`atomic_effective_volumes_hirshfeld` on the 0.20 Bohr lattice, SCF proatom for
the molecule, `None` proatom on the isolated atom for the denominator):
  * h2o, co, ch3oh x {cc-pvdz, def2-svp}: read from the committed Hirshfeld row
    (hirshfeld/<sys>_<basis>.json `hirshfeld_volumes.scf_proatom_lattice` over
    hirshfeld/<el>_atom_volume_<basis>.json `volume_ferric_none_lattice`), the
    PySCF+numpy mirror of ferric's construction that validation_hirshfeld.rs
    checks ferric against.
  * ch4, benzene x cc-pvdz: computed here with gen_hirshfeld.py's own functions
    (same definition, same lattice, same proatom tables).
  * `--ratios-json PATH` overrides any system's ratios with numbers produced by
    ferric itself: {"<system>_<basis>": [r_0, r_1, ...]}. The source is recorded.
For this row the ratios are INPUTS: the comparison is of the dispersion model
given identical inputs, so their provenance changes realism, not the bar.

Frequency grids stored per system:
  * "ferric_default": Gauss-Legendre n=20 mapped by u = u0 (1+x)/(1-x), u0 = 0.5
    — ferric's `QuadratureConfig::default()` (MiniMax -> optimized_u0(20) = 0.5).
  * "pymbd15": pymbd `freq_grid(15)` (a u=0 node of weight 0 prepended; L=0.6) —
    libmbd's default grid, so the libmbd 'scs' C6 can be compared node-for-node.

Run (light, under a minute; ch4/benzene run a few small PySCF SCFs):
    scripts/validation/run_slot.sh --light -- \\
        uv run --no-sync python scripts/validation/gen_ts_mbd.py [--ratios-json F] [system ...]
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402

GENERATOR = "scripts/validation/gen_ts_mbd.py"
ROW = "ts_mbd"
FREE_ATOM_REF_RS = common.ROOT / "crates/ferric-rpa/src/dispersion/free_atom_ref.rs"

# (system, basis, ratio source)
SYSTEMS = (
    ("h2o", "cc-pvdz", "hirshfeld_ref"),
    ("h2o", "def2-svp", "hirshfeld_ref"),
    ("co", "cc-pvdz", "hirshfeld_ref"),
    ("co", "def2-svp", "hirshfeld_ref"),
    ("ch3oh", "cc-pvdz", "hirshfeld_ref"),
    ("ch3oh", "def2-svp", "hirshfeld_ref"),
    ("ch4", "cc-pvdz", "pyscf_lattice"),
    ("benzene", "cc-pvdz", "pyscf_lattice"),
)
XYZ = {
    "h2o": common.MOL_DIR / "h2o.xyz",
    "co": common.MOL_DIR / "co.xyz",
    "ch3oh": common.MOL_DIR / "ch3oh.xyz",
    "ch4": common.MOL_DIR / "ch4.xyz",
    "benzene": common.ROOT / "testdata/molecules/benzene.xyz",
}
ELEM_FILE = {1: "h", 6: "c", 7: "n", 8: "o"}

FERRIC_GL_N = 20
FERRIC_GL_U0 = 0.5
PYMBD_NFREQ = 15

# MBD@rsSCS scope control: the PBE value of beta (pymbd docstring, Ambrosetti 2014).
RSSCS_BETA = 0.83
# MBD@rsSCS targets: PBE 0.83, PBE0/HSE06 0.85 (Ambrosetti 2014).
RSSCS_BETAS = (0.83, 0.85)
RSSCS_GRID_SENSITIVITY_N = (30, 60)
# Central-FD step (Bohr) for the libmbd gradient sign/convention check. FD error
# ~ h^2 E'''/6 + eps*sum(omega)/h ~ 1e-11 Ha/Bohr at h = 1e-4.
RSSCS_GRAD_FD_H = 1e-4
# The FD must reproduce libmbd's gradient (in the decided convention) to this
# fraction of max|FD|, and miss the opposite convention by > 1.0 of it.
RSSCS_GRAD_FD_REL_TOL = 1e-4

# Model anchors.
ANCHOR_ELEMENTS = (1, 6, 7, 8)
DIMERS = (  # (Z_A, Z_B, R Bohr): screening + energy, analytic per direction
    (6, 6, 2.5),
    (6, 6, 4.0),
    (6, 6, 8.0),
    (6, 1, 2.0),
    (6, 1, 5.0),
    (8, 1, 1.8),
)
LONDON = (  # (Z_A, Z_B, R): E_MBD R^6 / (-C6_TS) -> 1
    (6, 6, 15.0),
    (6, 6, 20.0),
    (6, 6, 30.0),
    (6, 1, 20.0),
    (8, 1, 20.0),
)
MP_DPS = 50

SQRT_PI = math.sqrt(math.pi)


# ---------------------------------------------------------------------------
# Free-atom table: ferric's (parsed from source) vs pymbd's
# ---------------------------------------------------------------------------


def ferric_free_atom_table() -> dict[int, tuple[float, float]]:
    """`ts_free_atom` rows parsed from the Rust source (derived, not hand-listed)."""
    pat = re.compile(
        r"^\s*(\d+)\s*=>\s*\(\s*([0-9.eE+-]+)\s*,\s*([0-9.eE+-]+)\s*,\s*None\s*\)"
    )
    table = {}
    for line in FREE_ATOM_REF_RS.read_text().splitlines():
        m = pat.match(line)
        if m:
            table[int(m.group(1))] = (float(m.group(2)), float(m.group(3)))
    if sorted(table) != list(range(1, 55)):
        raise RuntimeError(
            f"parsed ferric free-atom table has Z={sorted(table)}; expected 1..54 — "
            "the regex no longer matches free_atom_ref.rs"
        )
    return table


def ferric_r_vdw_table() -> dict[int, float]:
    """`ts_free_atom_r_vdw` rows parsed from the Rust source (Bohr)."""
    text = FREE_ATOM_REF_RS.read_text()
    start = text.index("pub fn ts_free_atom_r_vdw")
    body = text[start : text.index("_ => return None", start)]
    pat = re.compile(r"^\s*(\d+)\s*=>\s*([0-9.eE+-]+)\s*,", re.M)
    table = {int(m.group(1)): float(m.group(2)) for m in pat.finditer(body)}
    if sorted(table) != list(range(1, 55)):
        raise RuntimeError(
            f"parsed ferric R_vdW table has Z={sorted(table)}; expected 1..54"
        )
    return table


def r_vdw_table_comparison(r_tab) -> dict:
    from pymbd.pymbd import vdw_params

    rows, mism = [], []
    for z in range(1, 55):
        sym = common.ELEMENTS[z]
        rp = float(vdw_params[sym]["R_vdw(TS)"])
        rows.append(
            {"z": z, "symbol": sym, "ferric_r_vdw": r_tab[z], "pymbd_r_vdw_ts": rp}
        )
        if r_tab[z] != rp:
            mism.append(z)
    if mism:
        raise RuntimeError(
            f"ferric R_vdW table differs from pymbd R_vdw(TS) at Z={mism}"
        )
    return {"rows": rows, "mismatched_z": mism}


def table_comparison(ferric_tab) -> dict:
    from pymbd.pymbd import vdw_params

    rows, mism = [], []
    for z in range(1, 55):
        sym = common.ELEMENTS[z]
        a_f, c_f = ferric_tab[z]
        p = vdw_params[sym]
        a_p, c_p = float(p["alpha_0(TS)"]), float(p["C6(TS)"])
        row = {
            "z": z,
            "symbol": sym,
            "ferric_alpha": a_f,
            "ferric_c6": c_f,
            "pymbd_alpha_ts": a_p,
            "pymbd_c6_ts": c_p,
            "alpha_rel_diff": (a_f - a_p) / a_p,
            "c6_rel_diff": (c_f - c_p) / c_p,
        }
        rows.append(row)
        if a_f != a_p or c_f != c_p:
            mism.append(z)
    return {"rows": rows, "mismatched_z": mism}


# ---------------------------------------------------------------------------
# Frequency grids
# ---------------------------------------------------------------------------


def ferric_default_grid():
    import numpy as np

    x, w = np.polynomial.legendre.leggauss(FERRIC_GL_N)
    u = FERRIC_GL_U0 * (1 + x) / (1 - x)
    wt = w * 2 * FERRIC_GL_U0 / (1 - x) ** 2
    return u, wt


def pymbd_grid():
    from pymbd.pymbd import freq_grid

    return freq_grid(PYMBD_NFREQ)


# ---------------------------------------------------------------------------
# Model, numpy construction
# ---------------------------------------------------------------------------


def erf_as(x):
    """ferric's erf: Abramowitz & Stegun 7.1.26, coefficient-for-coefficient."""
    import numpy as np

    x = np.asarray(x, dtype=float)
    t = 1.0 / (1.0 + 0.3275911 * np.abs(x))
    y = 1.0 - (
        ((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
        + 0.254829592
    ) * t * np.exp(-x * x)
    return np.where(x >= 0, y, -y)


def t_gg_own(coords, sigma, erf_fn):
    """The Gaussian-damped dipole tensor written out (for the A&S-erf variant).
    Checked equal to pymbd's T_erf_coulomb when erf_fn is scipy's."""
    import numpy as np

    n = len(coords)
    out = np.zeros((n, n, 3, 3))
    for a in range(n):
        for b in range(n):
            if a == b:
                continue
            d = coords[b] - coords[a]
            r = float(np.linalg.norm(d))
            nv = d / r
            u = r / math.hypot(sigma[a], sigma[b])
            e = math.exp(-u * u)
            zeta = float(erf_fn(u)) - 2 * u / SQRT_PI * e
            eta = 4 * u**3 / (3 * SQRT_PI) * e
            nn = np.outer(nv, nv)
            out[a, b] = zeta * (np.eye(3) - 3 * nn) / r**3 + eta * 3 * nn / r**3
    return out.transpose(0, 2, 1, 3).reshape(3 * n, 3 * n)


def t_gg_pymbd(coords, sigma):
    import numpy as np
    from pymbd.pymbd import T_erf_coulomb

    n = len(coords)
    rs = coords[:, None, :] - coords[None, :, :]
    sij = np.sqrt(sigma[:, None] ** 2 + sigma[None, :] ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = T_erf_coulomb(rs, sij)
    t[np.arange(n), np.arange(n)] = 0.0  # on-site blocks: R=0 gives nan
    return t.transpose(0, 2, 1, 3).reshape(3 * n, 3 * n)


def sigma_of(alpha):
    import numpy as np

    return (np.sqrt(2 / np.pi) * alpha / 3) ** (1 / 3)


def ts_params(z, ratios, tab):
    import numpy as np

    a_free = np.array([tab[zz][0] for zz in z])
    c_free = np.array([tab[zz][1] for zz in z])
    r = np.asarray(ratios, dtype=float)
    alpha = r * a_free
    c6 = r * r * c_free
    omega = (4.0 / 3.0) * c6 / alpha**2
    return alpha, c6, omega, a_free, c_free


def ts_dynamic(alpha, omega, u):
    return alpha[None, :] / (1.0 + (u[:, None] / omega[None, :]) ** 2)  # [k, A]


def c6_closed(alpha, c6):
    a, c = alpha[:, None], c6[:, None]
    ab, cb = alpha[None, :], c6[None, :]
    return 2 * c * cb / (ab / a * c + a / ab * cb)


def c6_cp_iso(iso, w):
    import numpy as np

    return 3 / np.pi * np.einsum("k,ka,kb->ab", w, iso, iso)


def mbd_screen_np(coords, alpha_dyn, tensor_fn):
    """[k, A, 3, 3] screened per-atom tensors (ferric's `mbd_screen`)."""
    import numpy as np

    n = alpha_dyn.shape[1]
    out = np.empty((alpha_dyn.shape[0], n, 3, 3))
    for k, al in enumerate(alpha_dyn):
        t = tensor_fn(coords, sigma_of(al))
        c = np.diag(np.repeat(1 / al, 3)) + t
        ci = np.linalg.inv(c)
        out[k] = ci.reshape(n, 3, n, 3).swapaxes(1, 2).sum(axis=1)
    return out


def mbd_c6(scr, w):
    import numpy as np

    iso = np.einsum("kaii->ka", scr) / 3
    c6_iso = 3 / np.pi * np.einsum("k,ka,kb->ab", w, iso, iso)
    c6_aniso = 3 / np.pi * np.einsum("k,kaij,kbij->abij", w, scr, scr)
    mol = scr.sum(axis=1)
    iso_mol = np.einsum("kii->k", mol) / 3
    c6_mol = 3 / np.pi * float(np.sum(w * iso_mol**2))
    return c6_iso, c6_aniso, c6_mol


def mbd_energy_np(coords, alpha, omega, tensor_fn):
    import numpy as np

    t = tensor_fn(coords, sigma_of(alpha))
    pre = np.repeat(omega * np.sqrt(alpha), 3)
    h = np.diag(np.repeat(omega**2, 3)) + np.outer(pre, pre) * t
    ev = np.linalg.eigvalsh(h)
    return float(np.sum(np.sqrt(np.maximum(ev, 0.0))) / 2 - 1.5 * np.sum(omega)), float(
        ev.min()
    )


def rsscs_np(xyz, alpha, c6, r_vdw, beta, nfreq):
    """MBD@rsSCS from pymbd's primitives on an `nfreq`-node grid (pymbd's own
    `mbd_energy` hard-codes nfreq=15). Returns (E, a0_rs, c6_rs, r_rs, om_rs, ev_min)."""
    import numpy as np
    from pymbd.pymbd import dipole_matrix, screening

    a_rs, c6_rs, r_rs = screening(xyz, alpha, c6, r_vdw, beta, nfreq=nfreq)
    om = 4 / 3 * c6_rs / a_rs**2
    pre = np.repeat(om * np.sqrt(a_rs), 3)
    h = np.diag(np.repeat(om**2, 3)) + np.outer(pre, pre) * dipole_matrix(
        xyz, "fermi,dip", R_vdw=r_rs, beta=beta
    )
    ev = np.linalg.eigvalsh(h)
    if ev.min() <= 0:
        raise RuntimeError(f"rsSCS: non-positive coupled eigenvalue {ev.min():.3e}")
    e = float(np.sum(np.sqrt(ev)) / 2 - 1.5 * np.sum(om))
    return e, a_rs, c6_rs, r_rs, om, float(ev.min())


def rel(a, b):
    import numpy as np

    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    scale = max(float(np.max(np.abs(b))), 1e-300)
    return float(np.max(np.abs(a - b)) / scale)


def tolist(x):
    import numpy as np

    return np.asarray(x, dtype=float).tolist()


# ---------------------------------------------------------------------------
# Volume ratios
# ---------------------------------------------------------------------------


def ratios_from_hirshfeld_ref(system, basis):
    ref = json.loads(
        (common.REF_DIR / "hirshfeld" / f"{system}_{basis}.json").read_text()
    )
    geom = ref["provenance"]["geometry_bohr"]
    symbols = [g[0] for g in geom]
    vols = ref["hirshfeld_volumes"]["scf_proatom_lattice"]
    vfree, files = {}, {}
    for s in sorted(set(symbols)):
        z = common.z_of(s)
        f = common.REF_DIR / "hirshfeld" / f"{ELEM_FILE[z]}_atom_volume_{basis}.json"
        vfree[s] = json.loads(f.read_text())["volume_ferric_none_lattice"]
        files[s] = str(f.relative_to(common.ROOT))
    ratios = [v / vfree[s] for v, s in zip(vols, symbols)]
    src = {
        "kind": "committed Hirshfeld row (PySCF+numpy mirror of ferric's lattice volumes)",
        "molecular_volumes": f"testdata/reference/validation/hirshfeld/{system}_{basis}.json"
        " hirshfeld_volumes.scf_proatom_lattice",
        "free_atom_volumes": {
            s: f + " volume_ferric_none_lattice" for s, f in files.items()
        },
        "v_mol": vols,
        "v_free": vfree,
    }
    return symbols, [g[1:] for g in geom], ratios, src


def ratios_from_pyscf_lattice(system, basis):
    import numpy as np

    import gen_hirshfeld as gh
    import gen_properties as gp

    xyz = XYZ[system]
    symbols, coords = common.read_xyz(xyz)
    mol = common.build_pyscf_mol(xyz, basis)
    common.check_basis_like_for_like(mol, basis, symbols)
    scf_info, mf = gp.run_rhf(mol)
    dm = mf.make_rdm1()
    zset = sorted({common.z_of(s) for s in symbols})
    radii = gh.PROATOM_STEP * np.arange(1, gh.PROATOM_N + 1)
    tab = {}
    for z in zset:
        atom = gh.free_atom(z, basis)
        tab[z] = lambda r, t=np.asarray(atom["proatom_rho"]): gh.table_eval(radii, t, r)
    lpts, nlat = gh.lattice(mol)
    lrho = gh.density_on(mol, dm, lpts)
    vols = gh.hirshfeld_volumes(
        mol,
        lrho,
        np.full(len(lpts), gh.LATTICE_SPACING**3),
        gh.proatoms_on(mol, lpts, tab),
        lpts,
    )
    vfree, files = {}, {}
    for s in sorted(set(symbols)):
        z = common.z_of(s)
        f = common.REF_DIR / "hirshfeld" / f"{ELEM_FILE[z]}_atom_volume_{basis}.json"
        vfree[s] = json.loads(f.read_text())["volume_ferric_none_lattice"]
        files[s] = str(f.relative_to(common.ROOT))
    ratios = [v / vfree[s] for v, s in zip(vols, symbols)]
    src = {
        "kind": "computed here with gen_hirshfeld.py functions (RHF density, UHF free-atom "
        "SCF proatom tables, ferric's 0.20 Bohr lattice)",
        "rhf": scf_info,
        "lattice_n": nlat,
        "free_atom_volumes": {
            s: f + " volume_ferric_none_lattice" for s, f in files.items()
        },
        "v_mol": vols,
        "v_free": vfree,
    }
    return symbols, coords, ratios, src


# ---------------------------------------------------------------------------
# Per-system reference
# ---------------------------------------------------------------------------


def libmbd_rsscs_gradient(xyz, alpha, c6, r_vdw, beta, e_ref, ctx):
    """libmbd's MBD@rsSCS nuclear gradient, returned as dE/dR (N x 3).

    The convention of libmbd's `force=True` array (dE/dR or -dE/dR) is not
    assumed: it is decided by a central FD of libmbd's own energy and the
    decision is asserted. Returns (dE/dR, check-dict)."""
    import numpy as np
    from pymbd.fortran import MBDGeom

    def energy(x):
        return float(
            MBDGeom(x, n_freq=PYMBD_NFREQ).mbd_energy(
                alpha, c6, r_vdw, beta=beta, variant="rsscs"
            )
        )

    with MBDGeom(xyz, n_freq=PYMBD_NFREQ) as g:
        e_f, arr = g.mbd_energy(
            alpha, c6, r_vdw, beta=beta, variant="rsscs", force=True
        )
    arr = np.asarray(arr, dtype=float)
    if arr.shape != xyz.shape:
        raise RuntimeError(f"{ctx}: libmbd gradient shape {arr.shape} != {xyz.shape}")
    h = RSSCS_GRAD_FD_H
    fd = np.zeros_like(xyz)
    for a in range(xyz.shape[0]):
        for x in range(3):
            xp = xyz.copy()
            xm = xyz.copy()
            xp[a, x] += h
            xm[a, x] -= h
            fd[a, x] = (energy(xp) - energy(xm)) / (2 * h)
    scale = float(np.max(np.abs(fd)))
    res_grad = float(np.max(np.abs(arr - fd)))
    res_force = float(np.max(np.abs(arr + fd)))
    if res_grad < RSSCS_GRAD_FD_REL_TOL * scale and res_force > scale:
        kind, de_dr, res, res_other = "dE/dR", arr, res_grad, res_force
    elif res_force < RSSCS_GRAD_FD_REL_TOL * scale and res_grad > scale:
        kind, de_dr, res, res_other = "force (-dE/dR)", -arr, res_force, res_grad
    else:
        raise RuntimeError(
            f"{ctx}: libmbd gradient matches neither dE/dR nor -dE/dR by FD "
            f"(max|grad-fd|={res_grad:.2e}, max|grad+fd|={res_force:.2e}, max|fd|={scale:.2e})"
        )
    # Rounding-level agreement: measured 1.8e-12 rel (4.4e-16 Ha) for
    # h2o_cc-pvdz beta=0.83, so the bar is 1e-11, not 1e-12.
    if not abs(e_f - e_ref) <= 1e-11 * abs(e_ref):
        raise RuntimeError(
            f"{ctx}: libmbd energy with force=True {e_f!r} != without {e_ref!r}"
        )
    chk = {
        "libmbd_array_is": kind,
        "fd_step_bohr": h,
        "max_abs_fd": scale,
        "max_abs_residual_vs_fd": res,
        "max_abs_residual_opposite_convention": res_other,
        "energy_with_force": float(e_f),
    }
    return de_dr, chk


def gen_system(system, basis, source, tab, r_tab, overrides) -> Path:
    import numpy as np
    import pymbd
    from pymbd.fortran import MBDGeom
    from scipy.special import erf as scipy_erf

    if source == "hirshfeld_ref":
        symbols, coords, ratios, rsrc = ratios_from_hirshfeld_ref(system, basis)
    else:
        symbols, coords, ratios, rsrc = ratios_from_pyscf_lattice(system, basis)
    key = f"{system}_{basis}"
    if key in overrides:
        if len(overrides[key]) != len(symbols):
            raise ValueError(
                f"--ratios-json {key}: {len(overrides[key])} ratios for {len(symbols)} atoms"
            )
        rsrc = {"kind": "ferric-produced ratios from --ratios-json", "replaced": rsrc}
        ratios = list(overrides[key])
    z = [common.z_of(s) for s in symbols]
    xyz = np.asarray(coords, dtype=float)
    alpha, c6, omega, a_free, c_free = ts_params(z, ratios, tab)

    # pymbd's own from_volumes must give the same alpha/C6 (table cross-check).
    a_p, c6_p, rvdw_p = pymbd.from_volumes(symbols, ratios)
    from_vol_alpha = rel(a_p, alpha)
    from_vol_c6 = rel(c6_p, c6)
    if from_vol_alpha > 1e-15 or from_vol_c6 > 1e-15:
        raise RuntimeError(
            f"{key}: pymbd.from_volumes differs from ferric's table for an element in use "
            f"(alpha {from_vol_alpha:.2e}, C6 {from_vol_c6:.2e}); pass ferric's values"
        )

    # TS
    grids = {"ferric_default": ferric_default_grid(), "pymbd15": pymbd_grid()}
    closed = c6_closed(alpha, c6)
    ts = {"c6_pair_closed_form": tolist(closed), "grids": {}}
    for g, (u, w) in grids.items():
        iso = ts_dynamic(alpha, omega, u)
        cp = c6_cp_iso(iso, w)
        mol_iso = iso.sum(axis=1)
        ts["grids"][g] = {
            "c6_pair_cp": tolist(cp),
            "c6_molecular_iso": float(3 / np.pi * np.sum(w * mol_iso**2)),
            "cp_vs_closed_form_max_rel": float(np.max(np.abs(cp - closed) / closed)),
        }

    # MBD (target: plain SCS, Gaussian-damped T, exact erf through pymbd's kernel)
    mbd = {"grids": {}}
    meas = {}
    for g, (u, w) in grids.items():
        dyn = ts_dynamic(alpha, omega, u)
        scr = mbd_screen_np(xyz, dyn, t_gg_pymbd)
        scr_as = mbd_screen_np(xyz, dyn, lambda c, s: t_gg_own(c, s, erf_as))
        scr_own = mbd_screen_np(xyz, dyn, lambda c, s: t_gg_own(c, s, scipy_erf))
        meas[f"own_tensor_vs_pymbd_kernel_alpha_scs_{g}_max_rel"] = rel(scr_own, scr)
        c6_iso, c6_aniso, c6_mol = mbd_c6(scr, w)
        c6_iso_as, c6_aniso_as, c6_mol_as = mbd_c6(scr_as, w)
        mbd["grids"][g] = {
            "alpha_scs": tolist(scr.transpose(1, 0, 2, 3)),  # [A][k][3][3] like ferric
            "c6_pair_iso": tolist(c6_iso),
            "c6_pair_aniso": tolist(c6_aniso),  # [A][B][3][3]
            "c6_molecular_iso": c6_mol,
            "c6_molecular_iso_as_erf": c6_mol_as,
            "alpha_scs_iso_as_erf": tolist(
                (np.einsum("kaii->ka", scr_as) / 3).T
            ),  # [A][k]
        }
        meas[f"as_erf_vs_exact_alpha_scs_{g}_max_rel"] = rel(scr_as, scr)
        meas[f"as_erf_vs_exact_c6_pair_iso_{g}_max_rel"] = rel(c6_iso_as, c6_iso)
        meas[f"as_erf_vs_exact_c6_molecular_{g}_rel"] = abs(c6_mol_as - c6_mol) / c6_mol
        meas[f"screened_vs_unscreened_c6_pair_iso_{g}_max_rel"] = rel(
            c6_iso, c6_cp_iso(dyn, w)
        )
    e_np, ev_min = mbd_energy_np(xyz, alpha, omega, t_gg_pymbd)
    e_as, _ = mbd_energy_np(xyz, alpha, omega, lambda c, s: t_gg_own(c, s, erf_as))
    mbd["energy"] = e_np
    mbd["energy_as_erf"] = e_as
    mbd["energy_h_min_eigenvalue"] = ev_min
    meas["as_erf_vs_exact_energy_rel"] = abs(e_as - e_np) / abs(e_np)

    # libmbd (Fortran) — third construction.
    sig = sigma_of(alpha)
    e_lib = MBDGeom(xyz).mbd_energy(
        alpha, c6, rvdw_p, sigma=sig, damping="dip,gg", variant="plain"
    )
    _e, a_scs_lib, c6_scs_lib = MBDGeom(xyz, n_freq=PYMBD_NFREQ).mbd_energy(
        alpha,
        c6,
        rvdw_p,
        beta=RSSCS_BETA,
        damping="fermi,dip",
        variant="scs",
        intermediates=True,
    )
    g15 = mbd["grids"]["pymbd15"]
    a_scs_np0 = np.array(
        [np.trace(np.array(g15["alpha_scs"][a][0])) / 3 for a in range(len(z))]
    )
    c6_diag_np = np.diag(np.array(g15["c6_pair_iso"]))
    meas["libmbd_plain_dipgg_energy_vs_numpy_rel"] = abs(e_lib - e_np) / abs(e_np)
    meas["libmbd_scs_static_alpha_vs_numpy_max_rel"] = rel(a_scs_lib, a_scs_np0)
    meas["libmbd_scs_c6_vs_numpy_pymbd15_max_rel"] = rel(c6_scs_lib, c6_diag_np)
    for k in (
        "libmbd_plain_dipgg_energy_vs_numpy_rel",
        "libmbd_scs_static_alpha_vs_numpy_max_rel",
        "libmbd_scs_c6_vs_numpy_pymbd15_max_rel",
    ):
        if not meas[k] < 1e-10:
            raise RuntimeError(
                f"{key}: numpy and libmbd constructions disagree: {k}={meas[k]:.2e}"
            )
    mbd["libmbd"] = {
        "plain_dipgg_energy": e_lib,
        "scs_static_alpha_iso": tolist(a_scs_lib),
        "scs_c6_diag_pymbd15": tolist(c6_scs_lib),
    }

    # Scope control: standard MBD@rsSCS (pymbd python + libmbd), beta=0.83 (PBE).
    a_rs, c6_rs, _r = pymbd.screening(xyz, alpha, c6, rvdw_p, RSSCS_BETA)
    e_rs = pymbd.mbd_energy(xyz, alpha, c6, rvdw_p, RSSCS_BETA)
    e_rs_lib = MBDGeom(xyz).mbd_energy(alpha, c6, rvdw_p, beta=RSSCS_BETA)
    mbd["scope_rsscs"] = {
        "beta": RSSCS_BETA,
        "r_vdw": tolist(rvdw_p),
        "static_alpha_iso": tolist(a_rs),
        "c6_diag": tolist(c6_rs),
        "energy_pymbd": float(e_rs),
        "energy_libmbd": float(e_rs_lib),
    }
    meas["rsscs_vs_target_static_alpha_max_rel"] = rel(a_rs, a_scs_np0)
    meas["rsscs_vs_target_energy_rel"] = abs(e_rs - e_np) / abs(e_np)

    # MBD@rsSCS targets (ferric `mbd_rsscs`): pymbd python vs libmbd Fortran.
    r_free = np.array([r_tab[zz] for zz in z])
    r_vdw_ts = r_free * np.asarray(ratios, dtype=float) ** (1 / 3)
    meas["pymbd_from_volumes_vs_ferric_r_vdw_max_rel"] = rel(rvdw_p, r_vdw_ts)
    if meas["pymbd_from_volumes_vs_ferric_r_vdw_max_rel"] > 1e-15:
        raise RuntimeError(f"{key}: pymbd R_vdw differs from ferric's R_vdW table")
    rsscs = {
        "r_vdw_free": tolist(r_free),
        "r_vdw_ts": tolist(r_vdw_ts),
        "fermi_a": 6.0,
        "n_freq": PYMBD_NFREQ,
        "runs": [],
    }
    for beta in RSSCS_BETAS:
        e15, a15, c15, r15, om15, evmin = rsscs_np(
            xyz, alpha, c6, r_vdw_ts, beta, PYMBD_NFREQ
        )
        e_pm = float(pymbd.mbd_energy(xyz, alpha, c6, r_vdw_ts, beta))
        e_lb, a_lb, c_lb = MBDGeom(xyz, n_freq=PYMBD_NFREQ).mbd_energy(
            alpha, c6, r_vdw_ts, beta=beta, variant="rsscs", intermediates=True
        )
        chk = {
            "own_np_vs_pymbd_energy_rel": abs(e15 - e_pm) / abs(e_pm),
            "libmbd_vs_pymbd_energy_rel": abs(e_lb - e_pm) / abs(e_pm),
            "libmbd_vs_pymbd_alpha0_max_rel": rel(a_lb, a15),
            "libmbd_vs_pymbd_c6_max_rel": rel(c_lb, c15),
        }
        for k, v in chk.items():
            if not v < 1e-10:
                raise RuntimeError(
                    f"{key} beta={beta}: rsSCS constructions disagree {k}={v:.2e}"
                )
        grad, grad_chk = libmbd_rsscs_gradient(
            xyz, alpha, c6, r_vdw_ts, beta, e_lb, f"{key} beta={beta}"
        )
        sens = {}
        for nf in RSSCS_GRID_SENSITIVITY_N:
            e_n = rsscs_np(xyz, alpha, c6, r_vdw_ts, beta, nf)[0]
            sens[str(nf)] = {"energy": e_n, "rel_vs_15": abs(e_n - e15) / abs(e15)}
        rsscs["runs"].append(
            {
                "beta": beta,
                "energy_pymbd": e_pm,
                "energy_libmbd": float(e_lb),
                "alpha_0_rsscs": tolist(a15),
                "c6_rsscs": tolist(c15),
                "r_vdw_rsscs": tolist(r15),
                "omega_rsscs": tolist(om15),
                "alpha_0_rsscs_libmbd": tolist(a_lb),
                "c6_rsscs_libmbd": tolist(c_lb),
                "h_min_eigenvalue": evmin,
                "gradient": tolist(grad),
                "gradient_fd_check": grad_chk,
                "grid_sensitivity": sens,
                "measurements": chk,
            }
        )
    mbd["rsscs_targets"] = rsscs

    payload = {
        "row": "TS C6 (153) / MBD@TS (156)",
        "system": system,
        "basis": basis,
        "inputs": {
            "symbols": symbols,
            "z": z,
            "coords_bohr": tolist(xyz),
            "volume_ratios": [float(r) for r in ratios],
            "volume_ratio_source": rsrc,
            "alpha_free": tolist(a_free),
            "c6_free": tolist(c_free),
        },
        "ts_params": {
            "alpha_eff": tolist(alpha),
            "c6_eff": tolist(c6),
            "omega": tolist(omega),
        },
        "grids": {
            g: {"freqs": tolist(u), "weights": tolist(w)} for g, (u, w) in grids.items()
        },
        "ts": ts,
        "mbd": mbd,
        "measurements": {
            "pymbd_from_volumes_vs_ferric_table_alpha_max_rel": from_vol_alpha,
            "pymbd_from_volumes_vs_ferric_table_c6_max_rel": from_vol_c6,
            **meas,
        },
        "provenance": common.provenance(
            code="pymbd (numpy primitives) + libmbd (Fortran, via pymbd.fortran)",
            version=".".join(map(str, pymbd.__version__[:3])),
            keywords={
                "target_screening": "plain SCS: (diag(1/alpha(iu)) + T_gg)^-1, T_gg = "
                "pymbd.pymbd.T_erf_coulomb, sigma=(sqrt(2/pi) alpha/3)^(1/3); libmbd variant='scs'",
                "target_energy": "plain MBD, damping 'dip,gg', static TS alpha/omega; libmbd variant='plain'",
                "scope_control": f"MBD@rsSCS beta={RSSCS_BETA} (pymbd.screening/mbd_energy, libmbd default)",
                "rsscs_targets": f"MBD@rsSCS beta in {RSSCS_BETAS}, a=6, pymbd 15-node grid; pymbd "
                "python + libmbd variant='rsscs' (alpha_0_scs/C6_scs intermediates); R_vdw = "
                "ferric ts_free_atom_r_vdw * ratio^(1/3)",
                "rsscs_gradient": "libmbd variant='rsscs' force=True, stored as dE/dR (Hartree/Bohr, "
                "TS alpha_0/C6/R_vdw fixed); convention decided by central FD of libmbd's energy "
                f"(h={RSSCS_GRAD_FD_H} Bohr), see runs[*].gradient_fd_check",
                "ts_c6": "Casimir-Polder quadrature on the stored grids + closed-form combination rule",
                "free_atom_table": "ferric ts_free_atom (parsed from free_atom_ref.rs) == pymbd vdw_params TS",
            },
            basis_name=basis,
            xyz_path=XYZ[system],
            coords_bohr=tolist(xyz),
            symbols=symbols,
            grid={g: {"n": len(u)} for g, (u, w) in grids.items()},
            generator=GENERATOR,
            extra={"basis_role": "only produced the volume ratios"},
        ),
    }
    path = common.write_reference(ROW, system, basis, payload)
    print(
        f"{key:18s} TS cp-vs-closed(ferric grid)={ts['grids']['ferric_default']['cp_vs_closed_form_max_rel']:.1e} "
        f"E_mbd={e_np:.10f} AS-erf alpha={meas['as_erf_vs_exact_alpha_scs_ferric_default_max_rel']:.1e} "
        f"E={meas['as_erf_vs_exact_energy_rel']:.1e} libmbd E={meas['libmbd_plain_dipgg_energy_vs_numpy_rel']:.1e} "
        f"rsSCS E={meas['rsscs_vs_target_energy_rel']:.2f}",
        flush=True,
    )
    for run in rsscs["runs"]:
        m = run["measurements"]
        print(
            f"{key:18s} rsSCS beta={run['beta']}: E={run['energy_pymbd']:.12e} "
            f"libmbd-vs-pymbd E={m['libmbd_vs_pymbd_energy_rel']:.1e} "
            f"a0={m['libmbd_vs_pymbd_alpha0_max_rel']:.1e} c6={m['libmbd_vs_pymbd_c6_max_rel']:.1e} "
            f"grad[{run['gradient_fd_check']['libmbd_array_is']}] "
            f"vs FD={run['gradient_fd_check']['max_abs_residual_vs_fd']:.1e} "
            "grid "
            + " ".join(
                f"n{k}:{v['rel_vs_15']:.1e}" for k, v in run["grid_sensitivity"].items()
            ),
            flush=True,
        )
    return path


# ---------------------------------------------------------------------------
# Model anchors (mpmath, closed forms)
# ---------------------------------------------------------------------------


def dimer_t(mp, r, sa, sb, erf_fn):
    """(t_perp, t_par) of the Gaussian-damped T for a dimer on the z axis."""
    u = r / mp.sqrt(sa**2 + sb**2)
    e = mp.e ** (-(u**2))
    zeta = erf_fn(u) - 2 * u / mp.sqrt(mp.pi) * e
    eta = 4 * u**3 / (3 * mp.sqrt(mp.pi)) * e
    return zeta / r**3, (-2 * zeta + 3 * eta) / r**3


def dimer_screen(mp, a, b, t):
    """Row sums of [[1/a, t], [t, 1/b]]^-1: (alpha_A^scs, alpha_B^scs) for one direction."""
    det = 1 / (a * b) - t * t
    return (1 / b - t) / det, (1 / a - t) / det


def dimer_energy(mp, a, b, wa, wb, tp, tz):
    """Exact E for two isotropic QHOs with diagonal T (x, y: tp; z: tz)."""
    e = mp.mpf(0)
    for t in (tp, tp, tz):
        c = wa * wb * mp.sqrt(a * b) * t
        m = (wa**2 + wb**2) / 2
        d = mp.sqrt(((wa**2 - wb**2) / 2) ** 2 + c * c)
        e += (mp.sqrt(m + d) + mp.sqrt(m - d) - wa - wb) / 2
    return e


def gen_anchors(tab, r_tab) -> Path:
    import mpmath as mp
    import numpy as np
    import pymbd

    mp.mp.dps = MP_DPS
    as_erf = lambda x: mp.mpf(float(erf_as(float(x))))  # noqa: E731
    u_f, w_f = ferric_default_grid()

    # (a) ratio = 1: alpha, C6 reproduce the table; closed form C6_AA = C6_free.
    ratio1 = {}
    for z in ANCHOR_ELEMENTS:
        a, c = tab[z]
        om = 4 * mp.mpf(c) / (3 * mp.mpf(a) ** 2)
        c6_closed_aa = mp.mpf(3) / 4 * mp.mpf(a) ** 2 * om
        iso = ts_dynamic(np.array([a]), np.array([float(om)]), u_f)
        cp = float(c6_cp_iso(iso, w_f)[0, 0])
        ratio1[common.ELEMENTS[z]] = {
            "z": z,
            "alpha_free": a,
            "c6_free": c,
            "omega": float(om),
            "c6_closed_form_aa": float(c6_closed_aa),
            "c6_cp_ferric_default_grid": cp,
            "cp_quadrature_rel_err": (cp - c) / c,
        }
    pairs = {}
    for i, za in enumerate(ANCHOR_ELEMENTS):
        for zb in ANCHOR_ELEMENTS[i:]:
            (aa, ca), (ab, cb) = tab[za], tab[zb]
            pairs[f"{common.ELEMENTS[za]}-{common.ELEMENTS[zb]}"] = float(
                c6_closed(np.array([aa, ab]), np.array([ca, cb]))[0, 1]
            )

    # (b) dimers: analytic screening at every node of the ferric grid + energy.
    dimers = []
    for za, zb, r in DIMERS:
        (aa, ca), (ab, cb) = tab[za], tab[zb]
        wa = 4 * mp.mpf(ca) / (3 * mp.mpf(aa) ** 2)
        wb = 4 * mp.mpf(cb) / (3 * mp.mpf(ab) ** 2)
        rr = mp.mpf(r)
        sig = lambda al: (mp.sqrt(2 / mp.pi) * al / 3) ** (mp.mpf(1) / 3)  # noqa: E731
        scr = {"exact": [], "as_erf": []}
        for u in u_f:
            ak = mp.mpf(aa) / (1 + (mp.mpf(u) / wa) ** 2)
            bk = mp.mpf(ab) / (1 + (mp.mpf(u) / wb) ** 2)
            for tag, fn in (("exact", mp.erf), ("as_erf", as_erf)):
                tp, tz = dimer_t(mp, rr, sig(ak), sig(bk), fn)
                pa, pb = dimer_screen(mp, ak, bk, tp)
                za_, zb_ = dimer_screen(mp, ak, bk, tz)
                scr[tag].append(
                    {
                        "a_perp": float(pa),
                        "a_par": float(za_),
                        "b_perp": float(pb),
                        "b_par": float(zb_),
                    }
                )
        energies = {}
        for tag, fn in (("exact", mp.erf), ("as_erf", as_erf)):
            tp, tz = dimer_t(mp, rr, sig(mp.mpf(aa)), sig(mp.mpf(ab)), fn)
            energies[tag] = float(
                dimer_energy(mp, mp.mpf(aa), mp.mpf(ab), wa, wb, tp, tz)
            )
        # Cross-check the closed form against the numpy construction used for the
        # molecules (independent algebra: 2x2 per direction vs 6x6 inverse/eigh).
        xyz2 = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, r]])
        dyn2 = ts_dynamic(np.array([aa, ab]), np.array([float(wa), float(wb)]), u_f)
        scr_np = mbd_screen_np(xyz2, dyn2, t_gg_pymbd)
        cf = np.array(
            [
                [[x["a_perp"], x["a_par"]], [x["b_perp"], x["b_par"]]]
                for x in scr["exact"]
            ]
        )
        npv = scr_np[:, :, [0, 2], [0, 2]]  # [k, A, (xx, zz)]
        e_np2, _ = mbd_energy_np(
            xyz2, np.array([aa, ab]), np.array([float(wa), float(wb)]), t_gg_pymbd
        )
        chk_a, chk_e = (
            rel(npv, cf),
            abs(e_np2 - energies["exact"]) / abs(energies["exact"]),
        )
        if chk_a > 1e-12 or chk_e > 1e-10:
            raise RuntimeError(
                f"dimer {za}-{zb}@{r}: closed form vs numpy {chk_a:.1e} / {chk_e:.1e}"
            )
        dimers.append(
            {
                "closed_form_vs_numpy_alpha_max_rel": chk_a,
                "closed_form_vs_numpy_energy_rel": chk_e,
                "z": [za, zb],
                "r_bohr": r,
                "alpha": [aa, ab],
                "c6": [ca, cb],
                "omega": [float(wa), float(wb)],
                "screened_per_node": scr,
                "energy": energies["exact"],
                "energy_as_erf": energies["as_erf"],
            }
        )

    # (c) London limit: E_MBD R^6 -> -C6_TS(closed form).
    london = []
    for za, zb, r in LONDON:
        (aa, ca), (ab, cb) = tab[za], tab[zb]
        wa = 4 * mp.mpf(ca) / (3 * mp.mpf(aa) ** 2)
        wb = 4 * mp.mpf(cb) / (3 * mp.mpf(ab) ** 2)
        sig = lambda al: (mp.sqrt(2 / mp.pi) * al / 3) ** (mp.mpf(1) / 3)  # noqa: E731
        rr = mp.mpf(r)
        tp, tz = dimer_t(mp, rr, sig(mp.mpf(aa)), sig(mp.mpf(ab)), mp.erf)
        e = dimer_energy(mp, mp.mpf(aa), mp.mpf(ab), wa, wb, tp, tz)
        c6ab = mp.mpf(3) / 2 * aa * ab * wa * wb / (wa + wb)
        london.append(
            {
                "z": [za, zb],
                "r_bohr": r,
                "energy": float(e),
                "c6_ts_closed_form": float(c6ab),
                "energy_times_r6_over_minus_c6": float(-e * rr**6 / c6ab),
            }
        )

    ferric_tab_cmp = table_comparison(tab)
    payload = {
        "row": "TS C6 (153) / MBD@TS (156) — model anchors",
        "ratio1": ratio1,
        "c6_closed_form_pairs_ratio1": pairs,
        "ferric_default_grid": {"freqs": tolist(u_f), "weights": tolist(w_f)},
        "dimers": dimers,
        "london": london,
        "free_atom_table_comparison": ferric_tab_cmp,
        "r_vdw_table_comparison": r_vdw_table_comparison(r_tab),
        "provenance": {
            "code": "mpmath closed forms (dimer: 2x2 per Cartesian direction) + pymbd vdw_params",
            "version": {
                "mpmath": mp.__version__,
                "pymbd": ".".join(map(str, pymbd.__version__[:3])),
            },
            "mp_dps": MP_DPS,
            "definitions": {
                "screening": "per direction d: [[1/a, t_d], [t_d, 1/b]]^-1 row sums; "
                "t_perp = zeta/R^3, t_par = (-2 zeta + 3 eta)/R^3, "
                "zeta = erf(u) - 2u/sqrt(pi) e^-u^2, eta = 4u^3/(3 sqrt(pi)) e^-u^2, u = R/sigma_AB",
                "energy": "sum_d [sqrt(m+D) + sqrt(m-D) - w_A - w_B]/2, m=(w_A^2+w_B^2)/2, "
                "D = sqrt(((w_A^2-w_B^2)/2)^2 + (w_A w_B sqrt(a b) t_d)^2)",
                "london": "C6_AB = 3/2 a_A a_B w_A w_B/(w_A+w_B) == TS combination rule",
            },
            "free_atom_table": str(FREE_ATOM_REF_RS.relative_to(common.ROOT)),
            "free_atom_table_sha256": common.sha256_file(FREE_ATOM_REF_RS),
            "generator": GENERATOR,
            "git_head": common.git_head(),
        },
    }
    path = common.REF_DIR / ROW / "anchors_model.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        "anchors: table mismatches at Z =",
        ferric_tab_cmp["mismatched_z"],
        "| ratio1 cp quad err:",
        {k: f"{v['cp_quadrature_rel_err']:.1e}" for k, v in ratio1.items()},
        "| london:",
        [
            f"{x['z']}@{x['r_bohr']}:{x['energy_times_r6_over_minus_c6'] - 1:.1e}"
            for x in london
        ],
        flush=True,
    )
    return path


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratios-json", type=Path, default=None)
    ap.add_argument("only", nargs="*")
    args = ap.parse_args(argv[1:])
    overrides = json.loads(args.ratios_json.read_text()) if args.ratios_json else {}
    only = {s.lower() for s in args.only}
    known = {s for s, _, _ in SYSTEMS} | {"anchors"}
    if only - known:
        ap.error(f"unknown system(s): {sorted(only - known)}")
    t0 = time.time()
    tab = ferric_free_atom_table()
    r_tab = ferric_r_vdw_table()
    paths = []
    if not only or "anchors" in only:
        paths.append(gen_anchors(tab, r_tab))
    for system, basis, source in SYSTEMS:
        if only and system not in only:
            continue
        paths.append(gen_system(system, basis, source, tab, r_tab, overrides))
    for p in paths:
        print(
            "wrote", p.relative_to(common.ROOT), f"({p.stat().st_size / 1024:.0f} KB)"
        )
    print(f"total {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
