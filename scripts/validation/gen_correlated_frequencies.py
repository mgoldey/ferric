"""References for the "Correlated harmonic frequencies" validation row.

Consumer: crates/ferric-cli/tests/validation_correlated_frequencies.rs
Output:   testdata/reference/validation/correlated_frequencies/<case>.json

WHAT IS COMPARED. `task = "frequencies"` for the correlated kinds with an
analytic nuclear gradient (`rimp2` closed and open shell, `pdep-rpa`;
`scs-mp2` and `oo-rimp2` are not covered, see CASES) differences that analytic
gradient: per Cartesian coordinate j, H[:, j] = (g(x + d e_j) - g(x - d e_j)) /
(2 d) with d = DELTA_BOHR = 5e-3 (the CLI default, `DEFAULT_DELTA`), followed by
symmetrization, mass-weighting with ferric's mass table, and projection of
translations and rotations.

THE REFERENCE. PySCF has no analytic DF-MP2 or RI-RPA gradient, so every
gradient here is a 5-point central finite difference (step H_G = 2e-3 Bohr) of
PySCF's own total energy:

  * rimp2      E_RHF + E_DF-MP2        (exact-J/K RHF, `mp.dfmp2.DFMP2`)
               E_UHF + E_DF-UMP2       (open shell, `mp.dfump2.DFUMP2`)
  * scs-mp2    E_RHF + 1.2 E_OS + 1/3 E_SS  (ferric's `ScsMp2Config` defaults)
  * pdep-rpa   E_RHF + E_c^RPA         (`gw.rpa.RPA`, 40-point scaled
               Gauss-Legendre, x0 = 0.5, the grid the RPA gradient row uses)

with ferric's own bundled orbital and auxiliary bases handed to PySCF, so the
two surfaces are the same functional of the same basis (the energy anchor in
the test checks this at the reference geometry).

The reference Hessian is then built with THE SAME STENCIL ferric applies: the
central difference of that reference gradient at +-DELTA_BOHR. Its remaining
difference to ferric's Hessian is therefore not the O(d^2) truncation of the
frequency stencil (both carry it identically) but the gradient disagreement:

    |dH_ij| <= (eps_g + eps_ref) / d          (two gradients, over 2 d)

eps_g   the committed bar of the kind's gradient validation against the same
        PySCF energy surface (`validation_rimp2_gradient` / `validation_scs_mp2_gradient` /
        `rpa_analytic_gradient`: 2e-8 Ha/Bohr; open-shell RI-MP2 has only the
        self-FD test `u_rimp2_gradient_fd`, ANALYTIC = 2e-7);
eps_ref the reference gradient's own error, 1e-9 Ha/Bohr: 5-point truncation
        and noise measured step-converged to <= 7.4e-10 by `rpa_analytic_gradient`
        (h = 2e-3 vs 1e-3) with these convergence thresholds.

The wavenumber bar per mode follows by first-order perturbation theory of the
mass-weighted Hessian. With u_c = 1/sqrt(m_c) over the 3N Cartesian
coordinates (masses in electron masses), the elementwise bound b on dH gives
||dH_mw||_2 <= b * sum_c u_c^2 (the all-b matrix weighted by u u^T has that
spectral norm), and lambda_i = (nu_i / AU_FREQ_TO_CM)^2 moves by at most that:

    |d nu_i| <= nu_i / (2 lambda_i) * b * sum_c 1/m_c .

The bar stored in each JSON (`bar_cm1`, per mode) is that bound; the test also
bounds every Cartesian Hessian element by b.

Run (light; ~300 PySCF energies per case (translation invariance drops the last atom)):
    scripts/validation/run_slot.sh --light -- \
        uv run --no-sync python scripts/validation/gen_correlated_frequencies.py [case ...]
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402
import gen_rimp2_gradient as grg  # noqa: E402

ROW = "correlated_frequencies"
ROOT = common.ROOT
OUT_DIR = common.REF_DIR / ROW

DELTA_BOHR = 5.0e-3  # `ferric_scf::frequencies::DEFAULT_DELTA`
H_G = 2.0e-3  # 5-point gradient step of the reference
EPS_REF = 1.0e-9
AU_FREQ_TO_CM = 219_474.631_363_2  # ferric_scf::frequencies::AU_FREQ_TO_CM
AMU_TO_ME = 1.0 / 5.485_799_090_65e-4  # same file, AMU_TO_ELECTRON_MASS
STENCIL5 = ((-2, 1.0 / 12), (-1, -8.0 / 12), (1, 8.0 / 12), (2, -1.0 / 12))

# Gradient bars (Ha/Bohr) of the kinds' committed gradient validations.
EPS_G = {"rimp2": 2e-8, "scs-mp2": 2e-8, "pdep-rpa": 2e-8}
EPS_G_OPEN_SHELL = 2e-7

AUX = "cc-pvdz-ri"
BASIS = "6-31g"


def rotate(coords: np.ndarray) -> np.ndarray:
    """A fixed rotation (axis (1,2,3)/sqrt(14), 0.7 rad), so no Cartesian
    component of the gradient or Hessian is zero by orientation."""
    ax = np.array([1.0, 2.0, 3.0]) / np.sqrt(14.0)
    th = 0.7
    k = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    r = np.eye(3) + np.sin(th) * k + (1 - np.cos(th)) * (k @ k)
    return coords @ r.T


def geometries() -> dict[str, tuple[list[str], np.ndarray, int]]:
    """Slightly strained, asymmetric, rotated geometries (Angstrom)."""
    d2r = np.pi / 180.0
    # H2O: r1 = 0.96, r2 = 0.97 A, HOH 104.5 deg, planar.
    a = 104.5 * d2r
    h2o = np.array(
        [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [0.97 * np.cos(a), 0.97 * np.sin(a), 0.0]]
    )
    # NH3: r = 1.02, 1.01, 1.00 A, polar angle 68 deg from +z, azimuths 0/118/243.
    th = 68.0 * d2r
    nh3 = [[0.0, 0.0, 0.0]]
    for r, ph in ((1.02, 0.0), (1.01, 118.0), (1.00, 243.0)):
        p = ph * d2r
        nh3.append(
            [r * np.sin(th) * np.cos(p), r * np.sin(th) * np.sin(p), -r * np.cos(th)]
        )
    oh = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.97]])
    return {
        "h2o": (["O", "H", "H"], rotate(h2o), 1),
        "nh3": (["N", "H", "H", "H"], rotate(np.array(nh3)), 1),
        "oh": (["O", "H"], rotate(oh), 2),
    }


# case -> (kind, system, frozen core orbitals)
CASES = {
    "rimp2_h2o": ("rimp2", "h2o", 0),
    "rimp2_oh_uhf": ("rimp2", "oh", 0),
    "pdep-rpa_h2o": ("pdep-rpa", "h2o", 0),
}
# scs-mp2 (and frozen-core / NH3 cases) were cut for box time: a case costs
# ~300 PySCF energies (1.8-5 s each at H2O/6-31G, 1 thread). scs-mp2 and
# oo-rimp2 frequencies stay ungraded.


# ---------------------------------------------------------------------------
# Energies
# ---------------------------------------------------------------------------


def _mol(symbols, coords_bohr, mult):
    from pyscf import gto

    bas, cart = common.pyscf_basis(BASIS, symbols)
    assert not cart
    return gto.M(
        atom=common.pyscf_atom_bohr(symbols, coords_bohr),
        unit="Bohr",
        basis=bas,
        cart=False,
        spin=mult - 1,
        verbose=0,
    )


def _aux_df(mol, symbols):
    from pyscf import df

    aux_bas, aux_cart = common.pyscf_basis(AUX, symbols)
    assert not aux_cart
    with_df = df.DF(mol)
    with_df.auxmol = df.addons.make_auxmol(mol, aux_bas)
    with_df.auxmol.cart = False
    with_df.auxmol.build()
    with_df.auxbasis = aux_bas
    return with_df


def total_energy(kind, symbols, coords_bohr, mult, ncore):
    """E_total of the kind's surface at `coords_bohr` (exact-J/K SCF)."""
    from pyscf import scf

    mol = _mol(symbols, coords_bohr, mult)
    mf = scf.RHF(mol) if mult == 1 else scf.UHF(mol)
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-10
    mf.max_cycle = 300
    try:
        mf.kernel()
    except np.linalg.LinAlgError:  # PySCF DIIS eigh failure: restart, atom guess
        mf = scf.RHF(mol) if mult == 1 else scf.UHF(mol)
        mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-12, 1e-10, 300
        mf.init_guess = "atom"
        mf.diis_space = 6
        mf.kernel()
    if not mf.converged:
        raise RuntimeError("PySCF SCF did not converge")
    frozen = ncore if ncore else None
    if kind == "rimp2" or kind == "scs-mp2":
        if mult == 1:
            from pyscf.mp import dfmp2

            pt = dfmp2.DFMP2(mf, frozen=frozen)
        else:
            from pyscf.mp import dfump2

            pt = dfump2.DFUMP2(mf, frozen=frozen)
        pt.with_df = _aux_df(mol, symbols)
        pt.kernel()
        if kind == "rimp2":
            return float(mf.e_tot + pt.e_corr)
        return float(mf.e_tot + 1.2 * pt.e_corr_os + (1.0 / 3.0) * pt.e_corr_ss)
    if kind == "pdep-rpa":
        import gen_rpa_gradient as grpa

        with_df = grpa.gen_urpa.make_df(mol, grpa.gen_urpa.aux_dict(AUX, symbols))
        e_c, *_ = grpa.rpa_ecorr(mf, with_df, grpa.NW, frozen=frozen)
        return float(mf.e_tot + e_c)
    raise ValueError(kind)


def fd_gradient(energy, x0: np.ndarray) -> np.ndarray:
    """5-point gradient. Translation invariance (sum over atoms of dE/dR = 0
    for a surface of the geometry alone) gives the last atom's row, so only
    the first N-1 atoms are differenced."""
    g = np.zeros_like(x0)
    for a in range(x0.shape[0] - 1):
        for c in range(3):
            acc = 0.0
            for step, w in STENCIL5:
                x = x0.copy()
                x[a, c] += step * H_G
                acc += w * energy(x)
            g[a, c] = acc / H_G
    g[-1] = -g[:-1].sum(0)
    return g


# ---------------------------------------------------------------------------
# Frequencies (the same algebra as ferric_scf::frequencies)
# ---------------------------------------------------------------------------


def ferric_masses(symbols: list[str]) -> list[float]:
    src = (ROOT / "crates/ferric-core/src/elements.rs").read_text()
    body = src.split("const ATOMIC_MASSES: &[f64] = &[")[1].split("];")[0]
    table = [
        float(v.replace("_", ""))
        for v in re.findall(
            r"[0-9][0-9_]*\.?[0-9_]*(?:e-?[0-9]+)?", re.sub(r"//.*", "", body)
        )
    ]
    return [table[common.z_of(s) - 1] for s in symbols]


def frequencies(hess: np.ndarray, coords_bohr: np.ndarray, masses_amu: list[float]):
    n = len(masses_amu)
    m = np.array(masses_amu) * AMU_TO_ME
    inv = np.repeat(1.0 / np.sqrt(m), 3)
    mw = hess * inv[:, None] * inv[None, :]
    # translations + rotations in mass-weighted coordinates
    com = (m[:, None] * coords_bohr).sum(0) / m.sum()
    r = coords_bohr - com
    vecs = []
    for k in range(3):
        v = np.zeros((n, 3))
        v[:, k] = np.sqrt(m)
        vecs.append(v.ravel())
    for k in range(3):
        e = np.zeros(3)
        e[k] = 1.0
        v = np.cross(e, r) * np.sqrt(m)[:, None]
        vecs.append(v.ravel())
    u, s, _ = np.linalg.svd(np.array(vecs).T, full_matrices=False)
    t = u[:, s > 1e-8 * s.max()]
    p = np.eye(3 * n) - t @ t.T
    lam, vec = np.linalg.eigh(p @ mw @ p)
    keep = np.argsort(np.abs(lam))[t.shape[1] :]
    keep = keep[np.argsort(lam[keep])]
    lam_v, vec_v = lam[keep], vec[:, keep]
    nu = np.sign(lam_v) * np.sqrt(np.abs(lam_v)) * AU_FREQ_TO_CM
    return nu, lam_v, float((1.0 / m).sum() * 3.0)


def generate(case: str) -> dict:
    kind, system, ncore = CASES[case]
    symbols, coords_ang, mult = geometries()[system]
    x0 = np.array(coords_ang) * common.ANGSTROM_TO_BOHR

    def energy(x):
        return total_energy(kind, symbols, x, mult, ncore)

    e0 = energy(x0)
    g0 = fd_gradient(energy, x0)
    n = len(symbols)
    hess = np.zeros((3 * n, 3 * n))
    for j in range(3 * (n - 1)):
        a, c = divmod(j, 3)
        xp, xm = x0.copy(), x0.copy()
        xp[a, c] += DELTA_BOHR
        xm[a, c] -= DELTA_BOHR
        hess[:, j] = (
            (fd_gradient(energy, xp) - fd_gradient(energy, xm)) / (2 * DELTA_BOHR)
        ).ravel()
        print(f"  {case}: column {j + 1}/{3 * (n - 1)}", flush=True)
    # Last atom's columns from translation invariance (sum_k H[:, 3k+c] = 0).
    for c in range(3):
        hess[:, 3 * (n - 1) + c] = -sum(hess[:, 3 * k + c] for k in range(n - 1))
    asym = float(np.abs(hess - hess.T).max())
    hess = 0.5 * (hess + hess.T)

    masses = ferric_masses(symbols)
    nu, lam, sum_inv_m = frequencies(hess, x0, masses)
    eps_g = EPS_G_OPEN_SHELL if mult > 1 else EPS_G[kind]
    b = (eps_g + EPS_REF) / DELTA_BOHR
    bar = [float(v / (2.0 * l) * b * sum_inv_m) for v, l in zip(nu, lam)]

    # Negative control: the same molecule's RHF/UHF frequencies from an
    # energy-only stencil are not needed; PySCF's analytic SCF Hessian gives
    # the SCF-level frequencies the correlated ones must miss by >> bar.
    from pyscf import scf
    from pyscf import hessian as _hessian  # noqa: F401

    mol = _mol(symbols, x0, mult)
    mf = scf.RHF(mol) if mult == 1 else scf.UHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    h_scf = mf.Hessian().kernel()  # (natm, natm, 3, 3)
    h_scf = h_scf.transpose(0, 2, 1, 3).reshape(3 * n, 3 * n)
    nu_scf, _, _ = frequencies(0.5 * (h_scf + h_scf.T), x0, masses)

    return {
        "case": case,
        "kind": kind,
        "system": system,
        "basis": BASIS,
        "auxbasis": AUX,
        "frozen_core": ncore,
        "multiplicity": mult,
        "symbols": symbols,
        "geometry_bohr": x0.tolist(),
        "e_total": e0,
        "gradient_fd5": g0.tolist(),
        "delta_bohr": DELTA_BOHR,
        "h_gradient_bohr": H_G,
        "reference_asymmetry": asym,
        "masses_amu": masses,
        "cartesian_hessian": hess.ravel().tolist(),
        "frequencies_cm1": nu.tolist(),
        "scf_frequencies_cm1": nu_scf.tolist(),
        "bar": {
            "eps_gradient": eps_g,
            "eps_reference": EPS_REF,
            "hessian_element": b,
            "sum_inverse_mass_au": sum_inv_m,
            "cm1_per_mode": bar,
        },
        "provenance": {
            "code": "PySCF",
            "version": __import__("pyscf").__version__,
            "keywords": f"{kind}: 5-point FD (h={H_G} Bohr) of the PySCF total energy, "
            f"then the 3-point FD of that gradient at +-{DELTA_BOHR} Bohr",
            "basis": BASIS,
            "basis_sha256": common.sha256_file(common.basis_json_path(BASIS)),
            "aux_basis": AUX,
            "aux_sha256": common.sha256_file(common.basis_json_path(AUX)),
            "frozen_core": ncore,
            "scf_convergence": {"conv_tol": 1e-12, "conv_tol_grad": 1e-10},
            "bohr_in_angstrom": common.BOHR_IN_ANGSTROM,
            "generator": "scripts/validation/gen_correlated_frequencies.py",
            "git_head": common.git_head(),
        },
    }


def main(argv: list[str]) -> int:
    cases = argv or list(CASES)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for case in cases:
        payload = generate(case)
        (OUT_DIR / f"{case}.json").write_text(json.dumps(payload, indent=1) + "\n")
        print(
            f"{case}: nu = {np.round(payload['frequencies_cm1'], 2).tolist()}  "
            f"bar = {np.round(payload['bar']['cm1_per_mode'], 3).tolist()}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
