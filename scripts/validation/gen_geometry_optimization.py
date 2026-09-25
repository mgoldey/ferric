"""PySCF references for the VALIDATION.md rows "Geometry optimization RHF/RKS"
and "Geometry optimization UHF/ROHF" (W1).

Consumer: crates/ferric-scf/tests/validation_geometry_optimization.rs.
Output:   testdata/reference/validation/geometry_optimization/<system>_<basis>.json

WHAT IS REFERENCED. A converged minimum is a property of the potential energy
surface, not of the optimizer that found it, so the reference is PySCF's
ANALYTIC gradient driven to max|g| <= GRAD_TOL (1e-6 Ha/Bohr) from the SAME
distorted start geometry ferric starts from. geomeTRIC is not installed in the
project venv (pyscf.geomopt.geometric_solver fails to import), so the driver
is scipy BFGS on the Cartesian coordinates in Bohr, with an energy-consistent
analytic gradient (KS: `grid_response = True`, so the gradient is the exact
derivative of the grid energy, which is also what ferric's KS gradient
differentiates). The recorded final gradient is recomputed from a fresh SCF at
the final geometry and the generator REFUSES to write a point whose
max|g| > GRAD_TOL.

The Rust test compares orientation-free quantities: ALL interatomic distances
(which fix the geometry up to a reflection) and the listed bond angles, plus
the final energy. The final Cartesian geometry is also recorded, in the
reference's own frame, so the Rust side can run a single point there (energy
and gradient like-for-like, no optimizer involved).

Like-for-like (scripts/validation/common.py): ferric's own bundled basis JSON
and ferric's Bohr geometry; EXACT 4-index J/K (ferric RhfConfig
df_j_aux = df_k_aux = Some(""), PySCF without density fitting); KS grid
(75,110) unpruned, Becke partition with Becke-1988 radii (ferric's default
grid); frozen core n/a.

Systems x methods (start geometries testdata/molecules/validation/<sys>_opt_start.xyz):

    h2o, nh3, ch2o  x  RHF/6-31G, RKS-B3LYP/6-31G, RKS-PBE/cc-pVDZ
    ho2, ch3, nh2   x  UHF/6-31G, ROHF/6-31G, UKS-PBE/6-31G

Open shells: at the START geometry, three guesses each followed through an
internal stability() loop; the lowest stable state is selected. Every
optimizer evaluation starts from the previous evaluation's density and its
<S^2> (UHF/UKS) must stay within S2_DRIFT_TOL of the start value (a state
jump fails loudly). At the END geometry the state is stability-checked again;
if it is unstable the optimization is RESTARTED from the lower state (at most
MAX_RESTARTS times) and a still-unstable end point is REFUSED.

Run (light; a few minutes):
    scripts/validation/run_slot.sh --light -- \\
        uv run --no-sync python scripts/validation/gen_geometry_optimization.py
Restrict to systems by name:  ... gen_geometry_optimization.py ch3 nh2
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402

ROW = "geometry_optimization"
ROW_NAME = {
    "closed": "Geometry optimization RHF/RKS",
    "open": "Geometry optimization UHF/ROHF",
}
MAIN_GRID = (75, 110)
CONV_TOL = 1e-12
CONV_TOL_GRAD = 1e-9
GRAD_TOL = 1.0e-6  # max |dE/dx| (Ha/Bohr) the written minimum must satisfy
BFGS_GTOL = 2.0e-7  # scipy BFGS stops on max|g| (inf-norm) below this
MAX_BFGS_ROUNDS = 6  # restarts of BFGS (fresh Hessian) if it stalls above GRAD_TOL
MAX_RESTARTS = 3  # restarts after an end-point stability instability
GUESSES = ("minao", "atom", "huckel")
MAX_STAB_ROUNDS = 10
S2_DRIFT_TOL = 5e-2  # <S^2> may change with geometry; a state jump is O(1)

PYSCF_XC = {"pbe": "PBE,PBE", "b3lyp": "B3LYP"}

# system -> (charge, multiplicity, family, {basis: methods}, angle triples (i, vertex, k))
SYSTEMS = {
    "h2o": (
        0,
        1,
        "closed",
        {"6-31g": ("rhf", "rks_b3lyp"), "cc-pvdz": ("rks_pbe",)},
        [(1, 0, 2)],
    ),
    "nh3": (
        0,
        1,
        "closed",
        {"6-31g": ("rhf", "rks_b3lyp"), "cc-pvdz": ("rks_pbe",)},
        [(1, 0, 2), (1, 0, 3), (2, 0, 3)],
    ),
    "ch2o": (
        0,
        1,
        "closed",
        {"6-31g": ("rhf", "rks_b3lyp"), "cc-pvdz": ("rks_pbe",)},
        [(1, 0, 2), (1, 0, 3), (2, 0, 3)],
    ),
    "ho2": (0, 2, "open", {"6-31g": ("uhf", "rohf", "uks_pbe")}, [(2, 0, 1)]),
    "ch3": (
        0,
        2,
        "open",
        {"6-31g": ("uhf", "rohf", "uks_pbe")},
        [(1, 0, 2), (1, 0, 3), (2, 0, 3)],
    ),
    "nh2": (0, 2, "open", {"6-31g": ("uhf", "rohf", "uks_pbe")}, [(1, 0, 2)]),
}


def _is_ks(method: str) -> bool:
    return method.startswith(("rks", "uks"))


def _is_open(method: str) -> bool:
    return method in ("uhf", "rohf") or method.startswith("uks")


def _has_s2(method: str) -> bool:
    return method == "uhf" or method.startswith("uks")


def _make_mf(mol, method: str):
    from pyscf import dft, scf

    if method == "rhf":
        mf = scf.RHF(mol)
    elif method == "uhf":
        mf = scf.UHF(mol)
    elif method == "rohf":
        mf = scf.ROHF(mol)
    else:
        kind, xc = method.split("_")
        mf = {"rks": dft.RKS, "uks": dft.UKS}[kind](mol, xc=PYSCF_XC[xc])
        mf.grids.atom_grid = MAIN_GRID
        mf.grids.prune = None
        # ferric uses Becke (1988) size adjustment, not PySCF's default Treutler.
        mf.grids.radii_adjust = dft.radi.becke_atomic_radii_adjust
    mf.conv_tol = CONV_TOL
    mf.conv_tol_grad = CONV_TOL_GRAD
    mf.max_cycle = 500
    mf.verbose = 0
    return mf


def _converge(mf, dm0=None):
    mf.kernel(dm0=dm0)
    if not mf.converged:
        mf = mf.newton()
        mf.kernel(mf.mo_coeff, mf.mo_occ)
    if not mf.converged:
        raise RuntimeError("SCF did not converge")
    return mf


def _stable(mf) -> tuple[bool, object]:
    mo_i, _mo_e, stable_i, _stable_e = mf.stability(
        internal=True, external=False, return_status=True
    )
    return bool(stable_i), mo_i


def _follow_to_stable(mf):
    rounds = 0
    while True:
        stable, mo_i = _stable(mf)
        if stable or rounds >= MAX_STAB_ROUNDS:
            return mf, stable, rounds
        mf = _converge(mf, dm0=mf.make_rdm1(mo_i, mf.mo_occ))
        rounds += 1


def _gradient(mf, method: str):
    import numpy as np

    g = mf.nuc_grad_method()
    if _is_ks(method):
        g.grid_response = True
    g.verbose = 0
    return np.asarray(g.kernel())


def _at(mol, x_bohr):
    m = mol.copy()
    m.set_geom_(x_bohr.reshape(-1, 3), unit="Bohr")
    return m


def _start_state(mol, method: str):
    """Converged (open shell: lowest internally STABLE over GUESSES) SCF at the
    start geometry, with a record of how it was selected."""
    if not _is_open(method):
        return _converge(_make_mf(mol, method)), None
    best, scan = None, []
    for guess in GUESSES:
        mf = _make_mf(mol, method)
        mf.init_guess = guess
        try:
            mf = _converge(mf)
        except RuntimeError:
            scan.append({"guess": guess, "converged": False})
            continue
        mf, stable, rounds = _follow_to_stable(mf)
        scan.append(
            {
                "guess": guess,
                "energy": float(mf.e_tot),
                "internal_stable": stable,
                "stability_rounds": rounds,
            }
        )
        if stable and (best is None or mf.e_tot < best[0].e_tot):
            best = (mf, guess, rounds)
    if best is None:
        raise RuntimeError(f"{method}: no stable state at the start geometry: {scan}")
    mf, guess, rounds = best
    stable_es = [g["energy"] for g in scan if g.get("internal_stable")]
    return mf, {
        "internal_stable": True,
        "kind": f"PySCF {method.upper()} internal (real)",
        "selected_guess": guess,
        "rounds": rounds,
        "guess_scan": scan,
        "multiple_stable_minima": (max(stable_es) - min(stable_es)) > 1e-6,
    }


def _bfgs(mol, method: str, x0, dm0, s2_ref):
    """scipy BFGS on Cartesian Bohr coordinates. Returns (x, dm, n_evals,
    rounds). Each evaluation starts from the previous density."""
    import numpy as np
    from scipy.optimize import minimize

    cache = {"dm": dm0, "n": 0, "worst_s2": 0.0}

    def fg(x):
        mf = _converge(_make_mf(_at(mol, x), method), dm0=cache["dm"])
        if s2_ref is not None:
            d = abs(mf.spin_square()[0] - s2_ref)
            cache["worst_s2"] = max(cache["worst_s2"], d)
            if d > S2_DRIFT_TOL:
                raise RuntimeError(
                    f"{method}: <S^2> jumped by {d:.3f} during the optimization"
                )
        cache["dm"] = mf.make_rdm1()
        cache["n"] += 1
        return float(mf.e_tot), _gradient(mf, method).ravel()

    x = np.asarray(x0, float)
    rounds = 0
    for rounds in range(1, MAX_BFGS_ROUNDS + 1):
        res = minimize(
            fg,
            x,
            jac=True,
            method="BFGS",
            options={"gtol": BFGS_GTOL, "maxiter": 1000},
        )
        x = res.x
        if np.max(np.abs(res.jac)) <= BFGS_GTOL:
            break
    return x, cache["dm"], cache["n"], rounds, cache["worst_s2"]


def _distances(x_bohr):
    import numpy as np

    x = np.asarray(x_bohr).reshape(-1, 3)
    n = len(x)
    return [
        [i, j, float(np.linalg.norm(x[i] - x[j]))]
        for i in range(n)
        for j in range(i + 1, n)
    ]


def _angles(x_bohr, triples):
    import numpy as np

    x = np.asarray(x_bohr).reshape(-1, 3)
    out = []
    for i, v, k in triples:
        a, b = x[i] - x[v], x[k] - x[v]
        c = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        out.append([i, v, k, math.degrees(math.acos(max(-1.0, min(1.0, c))))])
    return out


def _method_block(mol, method: str, triples) -> dict:
    import numpy as np

    x_start = mol.atom_coords(unit="Bohr").ravel()
    mf0, state0 = _start_state(mol, method)
    s2_start = float(mf0.spin_square()[0]) if _has_s2(method) else None
    block = {
        "energy_start": float(mf0.e_tot),
        "stability_start": state0,
    }
    if s2_start is not None:
        block["s_squared_start"] = s2_start

    x, dm = x_start, mf0.make_rdm1()
    total_evals, restarts, bfgs_rounds, worst_s2 = 0, 0, 0, 0.0
    while True:
        x, dm, n, rounds, ws2 = _bfgs(mol, method, x, dm, s2_start)
        total_evals += n
        bfgs_rounds += rounds
        worst_s2 = max(worst_s2, ws2)
        mf = _converge(_make_mf(_at(mol, x), method), dm0=dm)
        if not _is_open(method):
            end_state = None
            break
        mf_s, stable, st_rounds = _follow_to_stable(mf)
        if not stable:
            raise RuntimeError(f"{method}: end point could not be made stable")
        if st_rounds == 0:
            end_state = {
                "internal_stable": True,
                "kind": f"PySCF {method.upper()} internal (real)",
                "rounds": 0,
                "optimization_restarts_after_instability": restarts,
            }
            break
        restarts += 1
        if restarts > MAX_RESTARTS:
            raise RuntimeError(
                f"{method}: end point still unstable after {MAX_RESTARTS} restarts"
            )
        dm = mf_s.make_rdm1()
        if s2_start is not None:
            s2_start = float(mf_s.spin_square()[0])

    g = _gradient(mf, method)
    gmax = float(np.max(np.abs(g)))
    if gmax > GRAD_TOL:
        raise RuntimeError(
            f"{method}: final max|g| {gmax:.2e} > {GRAD_TOL:.0e}; refusing"
        )
    d_start = _distances(x_start)
    d_opt = _distances(x)
    moved = max(abs(a[2] - b[2]) for a, b in zip(d_start, d_opt))
    block.update(
        {
            "energy_opt": float(mf.e_tot),
            "converged": True,
            "geometry_opt_bohr": np.asarray(x).reshape(-1, 3).tolist(),
            "nuclear_repulsion_opt": float(_at(mol, x).energy_nuc()),
            "distances_opt_bohr": d_opt,
            "angles_opt_deg": _angles(x, triples),
            "gradient_opt": g.tolist(),
            "max_abs_gradient_opt": gmax,
            "rms_gradient_opt": float(np.sqrt(np.mean(g**2))),
            "max_distance_change_from_start_bohr": moved,
            "stability_end": end_state,
            "optimizer": {
                "driver": "scipy.optimize.minimize(method='BFGS'), Cartesian Bohr",
                "gtol_inf_norm": BFGS_GTOL,
                "bfgs_rounds": bfgs_rounds,
                "energy_gradient_evaluations": total_evals,
                "final_point_refuse_above_max_abs_gradient": GRAD_TOL,
            },
        }
    )
    if _has_s2(method):
        block["s_squared_opt"] = float(mf.spin_square()[0])
        block["max_s_squared_drift_during_optimization"] = worst_s2
    if moved < 1e-2:
        raise RuntimeError(f"{method}: optimizer barely moved ({moved:.2e} Bohr)")
    return block


def _run(system: str, basis_name: str, spec) -> Path:
    import numpy
    import pyscf
    import scipy

    charge, mult, family, bases, triples = spec
    methods = bases[basis_name]
    xyz = common.MOL_DIR / f"{system}_opt_start.xyz"
    symbols, coords = common.read_xyz(xyz)
    mol = common.build_pyscf_mol(xyz, basis_name, charge=charge, multiplicity=mult)
    basis_check = common.check_basis_like_for_like(mol, basis_name, symbols)
    blocks = {}
    for method in methods:
        b = _method_block(mol, method, triples)
        blocks[method] = b
        s2 = f" <S2> {b['s_squared_opt']:.6f}" if "s_squared_opt" in b else ""
        print(
            f"{system:5s} {basis_name:8s} {method:10s} E0 {b['energy_start']:.10f} "
            f"Eopt {b['energy_opt']:.10f} |g| {b['max_abs_gradient_opt']:.1e} "
            f"moved {b['max_distance_change_from_start_bohr']:.3f} "
            f"evals {b['optimizer']['energy_gradient_evaluations']}{s2}",
            flush=True,
        )
    # Method separation inside this file (what a method swap would miss by).
    sep = {}
    for i, a in enumerate(methods):
        for c in methods[i + 1 :]:
            da, dc = blocks[a]["distances_opt_bohr"], blocks[c]["distances_opt_bohr"]
            sep[f"{a}_vs_{c}"] = max(abs(p[2] - q[2]) for p, q in zip(da, dc))
    payload = {
        "row": ROW_NAME[family],
        "system": system,
        "basis": basis_name,
        "charge": charge,
        "multiplicity": mult,
        "nao": mol.nao_nr(),
        "nuclear_repulsion": float(mol.energy_nuc()),
        "start_geometry_bohr": numpy.asarray(coords).tolist(),
        "distances_start_bohr": _distances(numpy.asarray(coords).ravel()),
        "angle_triples": [list(t) for t in triples],
        "methods": list(methods),
        "method_separation_max_distance_bohr": sep,
        "conventions": {
            "distances": "all pairs i<j, [i, j, r_ij in Bohr], atom order as in the xyz",
            "angles": "[i, vertex, k, degrees]",
            "gradient": "N x 3 Cartesian, Hartree/Bohr, in the reference's own frame",
        },
        **blocks,
        "provenance": common.provenance(
            code="PySCF + scipy BFGS",
            version=f"pyscf {pyscf.__version__}, scipy {scipy.__version__}",
            keywords={
                "methods": list(methods),
                "xc": {k: PYSCF_XC[k.split("_")[1]] for k in methods if _is_ks(k)},
                "conv_tol": CONV_TOL,
                "conv_tol_grad": CONV_TOL_GRAD,
                "max_cycle": 500,
                "eri": "exact 4-index (no density fitting)",
                "ks_gradient": "grid_response=True",
                "optimizer": "scipy BFGS, Cartesian Bohr, gtol (inf-norm) "
                f"{BFGS_GTOL}, restarted up to {MAX_BFGS_ROUNDS}x; final max|g| <= {GRAD_TOL}",
                "why_not_geometric": "geomeTRIC is not installed in the project venv",
            },
            basis_name=basis_name,
            xyz_path=xyz,
            coords_bohr=coords,
            symbols=symbols,
            grid={
                "atom_grid": list(MAIN_GRID),
                "prune": None,
                "partition": "Becke",
                "radii_adjust": "becke_atomic_radii_adjust (Becke 1988)",
            }
            if any(_is_ks(m) for m in methods)
            else None,
            aux=None,
            frozen_core=None,
            scf_conv={"conv_tol": CONV_TOL, "conv_tol_grad": CONV_TOL_GRAD},
            stability=(
                {
                    "internal_stable": True,
                    "kind": "PySCF internal stability() loop at the start AND end geometry",
                }
                if family == "open"
                else None
            ),
            generator="scripts/validation/gen_geometry_optimization.py",
            extra={"basis_like_for_like": basis_check, "numpy": numpy.__version__},
        ),
    }
    return common.write_reference(ROW, system, basis_name, payload)


def main() -> int:
    only = set(sys.argv[1:])
    for system, spec in SYSTEMS.items():
        if only and system not in only:
            continue
        for basis_name in spec[3]:
            path = _run(system, basis_name, spec)
            print(f"wrote {path.relative_to(common.ROOT)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
