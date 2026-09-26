"""PySCF references for the VALIDATION.md "RPA gradient" row (closed-shell
PDEP-RPA nuclear gradient).

Consumer: crates/ferric-rpa/tests/validation_rpa_gradient.rs.
Output:   testdata/reference/validation/rpa_gradient/<system>_<basis>.json

WHAT FERRIC COMPUTES (read from crates/ferric-rpa/src/gradient.rs, not from
its doc comments)
------------------------------------------------------------------------------
`total_rpa_gradient(mol, obs, aux, op, cfg, h)` returns (E_tot, g) where
  * E_tot = E_RHF + E_c^RPA at the reference geometry, and
  * g[A, k] = (E(R + h e_Ak) - E(R - h e_Ak)) / (2 h), a 3-point CENTRAL FINITE
    DIFFERENCE of the TOTAL energy E(R) = E_RHF(R) + E_c^RPA(R), each point a
    fresh `solve_rhf` (RhfConfig::default() fitting fields = EXACT four-centre
    J/K, density_conv 1e-9) followed by a full `run_pdep_rpa` with the
    caller's config (`rpa_total_energy` returns `rhf.energy + r.e_rpa`).
It is not an analytic gradient and does not hold any Ritz basis fixed across
displacements. The reference orbitals are the RHF
orbitals of the (displaced) geometry; frozen core is whatever
`cfg.frozen_core` says (threaded into `RiMp2Config` by `run_pdep_rpa`); the
quadrature is `cfg.quadrature`.

HOW THIS SCRIPT MATCHES IT
--------------------------
PySCF has no RPA gradient. The reference is therefore a finite difference of
the PySCF energy surface E(R) = E_RHF(exact integrals) + E_c(`gw.rpa.RPA`):
  * orbital basis and aux basis from ferric's bundled JSON (common.pyscf_basis),
    geometry in Bohr with ferric's constant, DF Coulomb metric Cholesky-whitened
    (gen_urpa helpers);
  * frequency grid MATCHED: `_get_scaled_legendre_roots(40, 0.5)` is ferric's
    `QuadratureScheme::GaussLegendre` with n_points 40, u0 0.5 (established by
    the U-RPA row), so the comparison carries no quadrature error;
  * full rank (ferric trunc_thresh 0), all electrons (and one frozen-core block
    with PySCF `frozen=1` = ferric `frozen_core = 1`);
  * RHF conv_tol CONV_TOL / conv_tol_grad CONV_TOL_GRAD, each displaced SCF
    started from the reference density.

Gradients written (Ha/Bohr, atom-major (natm, 3)):
  * `fd5[h]`: 5-point central FD at h in H_FD5 (two steps; their max |diff| is
    the recorded step-convergence and must be <= STEP_CONV_MAX or the reference
    is refused). The larger step is the one the test compares against.
  * `fd3_h<H_LIB>`: 3-point central FD at ferric's library step H_LIB -- the
    SAME stencil ferric runs, so ferric vs this is like-for-like (no
    truncation-error difference); fd3 vs fd5 is recorded as the truncation
    error of ferric's stencil.
  * controls: the PySCF analytic RHF gradient (the RPA gradient must MISS it);
    the 5-point FD of E_c ALONE (a correlation-only gradient; the total
    gradient must MISS it); the 5-point FD at a COARSE
    quadrature (NW_COARSE points; ferric at NW_COARSE must MATCH it and ferric
    at 40 must MISS it); frozen-core (FC systems only).
The energy at the reference geometry is also cross-checked by numpy on PySCF's
Cholesky tensors (gen_urpa.numpy_urpa); disagreement > 1e-10 refuses.

Run (light; ~a second per displaced PySCF RHF+RPA):
    OPENBLAS_NUM_THREADS=1 scripts/validation/run_slot.sh --light -- \\
        uv run --no-sync python scripts/validation/gen_rpa_gradient.py [system ...]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402
import gen_urpa  # noqa: E402  (aux_dict, make_df, check_naux, spin_channels, numpy_urpa)

ROW = "rpa_gradient"
ROW_NAME = "RPA gradient"
NW = gen_urpa.NW  # 40
X0 = gen_urpa.X0  # 0.5
NW_COARSE = 6
H_FD5 = (2e-3, 1e-3)  # Bohr; H_FD5[0] is the primary reference
H_LIB = 5e-4  # ferric's gradient.rs step used by the test (3-point stencil)
STEP_CONV_MAX = 1e-7
CONV_TOL = 1e-12
CONV_TOL_GRAD = 1e-10
NUMPY_VS_PYSCF_MAX = 1e-10

# system -> (basis, aux, frozen-core count for the FC block or None)
SYSTEMS = {
    "h2o_distorted": ("cc-pvdz", "cc-pvdz-ri", 1),
    "nh3_distorted": ("cc-pvdz", "cc-pvdz-ri", None),
}

STENCIL5 = ((-2, 1.0 / 12), (-1, -8.0 / 12), (1, 8.0 / 12), (2, -1.0 / 12))
STENCIL3 = ((-1, -0.5), (1, 0.5))


def pyscf_mol(symbols, coords_bohr, basis):
    from pyscf import gto

    bas, cart = common.pyscf_basis(basis, symbols)
    assert not cart
    return gto.M(
        atom=common.pyscf_atom_bohr(symbols, coords_bohr),
        unit="Bohr",
        basis=bas,
        cart=False,
        verbose=0,
    )


def rpa_ecorr(mf, with_df, nw, frozen=None):
    from pyscf.gw.rpa import RPA

    rpa = RPA(mf, frozen=frozen)
    rpa.with_df = with_df
    rpa.verbose = 0
    eris = rpa.ao2mo()
    rpa.kernel(eris=eris, nw=nw, x0=X0)
    return float(rpa.e_corr), rpa, eris


def energies(symbols, coords, basis, aux_name, dm0, ncore):
    """All energies one geometry contributes: E_RHF and E_c for every block."""
    from pyscf import scf

    mol = pyscf_mol(symbols, coords, basis)
    mf = scf.RHF(mol)
    mf.conv_tol = CONV_TOL
    mf.conv_tol_grad = CONV_TOL_GRAD
    mf.max_cycle = 300
    mf.kernel(dm0=dm0)
    if not mf.converged:
        raise RuntimeError("PySCF RHF did not converge")
    aux = gen_urpa.aux_dict(aux_name, symbols)
    with_df = gen_urpa.make_df(mol, aux)
    out = {"e_rhf": float(mf.e_tot)}
    out["e_c"], rpa, eris = rpa_ecorr(mf, with_df, NW)
    out["e_c_coarse"] = rpa_ecorr(mf, with_df, NW_COARSE)[0]
    if ncore:
        out["e_c_fc"] = rpa_ecorr(mf, with_df, NW, frozen=ncore)[0]
    return out, mf, with_df, rpa, eris


def fd_gradients(symbols, coords, basis, aux_name, dm0, ncore):
    """{label: (natm,3) gradient} for every stencil/step/block; runs each
    displaced geometry once and reuses it across blocks."""
    c0 = np.asarray(coords)
    natm = len(symbols)
    cache = {}

    def at(a, k, disp):
        key = (a, k, round(disp, 12))
        if key not in cache:
            c = c0.copy()
            c[a, k] += disp
            cache[key] = energies(symbols, c.tolist(), basis, aux_name, dm0, ncore)[0]
        return cache[key]

    blocks = {
        "total": lambda e: e["e_rhf"] + e["e_c"],
        "corr_only": lambda e: e["e_c"],
        "total_coarse": lambda e: e["e_rhf"] + e["e_c_coarse"],
    }
    if ncore:
        blocks["total_fc"] = lambda e: e["e_rhf"] + e["e_c_fc"]
    runs = [(f"fd5_h{h:g}", STENCIL5, h) for h in H_FD5]
    runs.append((f"fd3_h{H_LIB:g}", STENCIL3, H_LIB))
    grads = {r: {b: np.zeros((natm, 3)) for b in blocks} for r, _, _ in runs}
    for name, stencil, h in runs:
        for a in range(natm):
            for k in range(3):
                for step, w in stencil:
                    e = at(a, k, step * h)
                    for b, f in blocks.items():
                        grads[name][b][a, k] += w * f(e) / h
    return grads, len(cache)


def gen(system):
    import pyscf

    basis, aux_name, ncore = SYSTEMS[system]
    t0 = time.perf_counter()
    xyz = common.MOL_DIR / f"{system}.xyz"
    symbols, coords = common.read_xyz(xyz)
    mol0 = pyscf_mol(symbols, coords, basis)
    ll = common.check_basis_like_for_like(mol0, basis, symbols)

    e0, mf, with_df, rpa, eris = energies(symbols, coords, basis, aux_name, None, ncore)
    naux = gen_urpa.check_naux(with_df, aux_name, symbols)
    channels = gen_urpa.spin_channels(rpa, eris, unrestricted=False)
    e_np, _ = gen_urpa.numpy_urpa(channels, NW)
    d_np = abs(e_np - e0["e_c"])
    if d_np > NUMPY_VS_PYSCF_MAX:
        raise RuntimeError(f"{system}: numpy RPA vs PySCF {d_np:.2e}")
    g_rhf = np.asarray(mf.nuc_grad_method().kernel())
    dm = mf.make_rdm1()

    grads, n_points = fd_gradients(symbols, coords, basis, aux_name, dm, ncore)
    h1, h2 = (f"fd5_h{h:g}" for h in H_FD5)
    step_conv = {b: float(np.abs(grads[h1][b] - grads[h2][b]).max()) for b in grads[h1]}
    if step_conv["total"] > STEP_CONV_MAX:
        raise RuntimeError(
            f"{system}: 5-point FD not step-converged: {step_conv['total']:.2e} "
            f"> {STEP_CONV_MAX:.0e}"
        )
    g_ref = grads[h1]["total"]
    fd3 = grads[f"fd3_h{H_LIB:g}"]["total"]
    diag = {
        "fd5_step_convergence_max": step_conv,
        "fd3_lib_step_vs_fd5_max": float(np.abs(fd3 - g_ref).max()),
        "rpa_vs_rhf_gradient_max": float(np.abs(g_ref - g_rhf).max()),
        "total_vs_corr_only_gradient_max": float(
            np.abs(g_ref - grads[h1]["corr_only"]).max()
        ),
        "nw40_vs_coarse_gradient_max": float(
            np.abs(g_ref - grads[h1]["total_coarse"]).max()
        ),
        "translation_sum_max": float(np.abs(g_ref.sum(axis=0)).max()),
    }
    if ncore:
        diag["all_electron_vs_fc_gradient_max"] = float(
            np.abs(g_ref - grads[h1]["total_fc"]).max()
        )
    wall = time.perf_counter() - t0

    payload = {
        "row": ROW_NAME,
        "system": system,
        "basis": basis,
        "aux": aux_name,
        "charge": 0,
        "multiplicity": 1,
        "nao": int(mol0.nao_nr()),
        "naux": int(naux),
        "nelectron": int(mol0.nelectron),
        "nuclear_repulsion": float(mol0.energy_nuc()),
        "like_for_like": ll,
        "nw": NW,
        "x0": X0,
        "nw_coarse": NW_COARSE,
        "frozen_core_block": ncore,
        "energy": {
            "e_rhf": e0["e_rhf"],
            "e_c": e0["e_c"],
            "e_total": e0["e_rhf"] + e0["e_c"],
            "e_c_numpy": e_np,
            "numpy_vs_pyscf_abs_diff": d_np,
            "e_c_coarse": e0["e_c_coarse"],
            **({"e_c_fc": e0["e_c_fc"]} if ncore else {}),
        },
        "gradient": {
            "primary": h1,
            "fd": {
                name: {b: g.tolist() for b, g in blk.items()}
                for name, blk in grads.items()
            },
            "rhf_analytic": g_rhf.tolist(),
            "h_lib": H_LIB,
            "note": "total = E_RHF + E_c(nw=40) (what ferric's total_rpa_gradient "
            "differentiates); corr_only = E_c alone; total_coarse = E_RHF + "
            f"E_c(nw={NW_COARSE}); total_fc = E_RHF + E_c(frozen={ncore})",
        },
        "diagnostics": diag,
        "n_displaced_points": n_points,
        "generator_seconds": wall,
        "provenance": common.provenance(
            code="PySCF",
            version=pyscf.__version__,
            keywords={
                "scf": f"scf.RHF exact 4-index ERIs, conv_tol {CONV_TOL}, conv_tol_grad "
                f"{CONV_TOL_GRAD}, displaced SCFs start from the reference density",
                "rpa": "pyscf.gw.rpa.RPA (kernel = rpa.kernel), nw 40, x0 0.5; "
                "df.DF(mol, auxbasis=ferric aux JSON)",
                "gradient": f"central FD of the total energy: 5-point at h = {H_FD5} Bohr, "
                f"3-point at h = {H_LIB} Bohr",
            },
            basis_name=basis,
            xyz_path=xyz,
            coords_bohr=coords,
            symbols=symbols,
            grid=None,
            aux=aux_name,
            frozen_core=f"none (all-electron blocks); frozen={ncore} in the total_fc block"
            if ncore
            else None,
            scf_conv={"conv_tol": CONV_TOL, "conv_tol_grad": CONV_TOL_GRAD},
            stability=None,
            generator="scripts/validation/gen_rpa_gradient.py",
        ),
    }
    path = common.write_reference(ROW, system, basis, payload)
    print(
        f"{system}/{basis}: E_RHF {e0['e_rhf']:.10f} E_c {e0['e_c']:.10f} "
        f"numpy |d| {d_np:.1e}\n  step conv (5pt h={H_FD5}) "
        + ", ".join(f"{b} {v:.1e}" for b, v in step_conv.items())
        + "\n  "
        + ", ".join(
            f"{k} {v:.2e}" for k, v in diag.items() if k != "fd5_step_convergence_max"
        )
        + f"\n  {n_points} displaced points, {wall:.1f} s -> {path.relative_to(common.ROOT)}"
    )


def main(argv):
    unknown = set(argv) - set(SYSTEMS)
    if unknown:
        raise SystemExit(
            f"unknown systems: {sorted(unknown)} (known: {sorted(SYSTEMS)})"
        )
    for s in SYSTEMS:
        if not argv or s in argv:
            gen(s)


if __name__ == "__main__":
    main(sys.argv[1:])
