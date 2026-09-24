"""PySCF references for the VALIDATION.md rows "Meta-GGA gradient, closed" and
"Meta-GGA gradient, open" (validation tier W1).

Consumer: crates/ferric-scf/tests/validation_mgga_gradients.rs.
Output:   testdata/reference/validation/mgga_gradients/<system>_<basis>.json

CASES

1. Closed shell, RKS SCAN and r2SCAN, analytic gradient with
   `grid_response = True`:

       h2o, nh3  x  6-31G, def2-SVP  x  {RI-J, exact J}

   RI-J is ferric's DEFAULT for every functional (rhf.rs `resolve_aux`: an
   unset `df_j_aux` auto-defaults to def2-universal-jkfit when `xc` is set),
   so the RI-J block is the like-for-like reference for the default path; its
   PySCF side is `density_fit(aux)` with the aux fed from ferric's JSON, and
   PySCF's DF gradient includes the auxiliary-basis response
   (`auxbasis_response = True`, the default). The exact-J block is the control:
   ferric `df_j_aux = Some("")` (explicit conventional four-centre J) against
   plain PySCF RKS.

2. Open shell, UKS SCAN and r2SCAN, RI-J, `grid_response = True`:

       ho2 (2A''), nh2 (2B1)  x  6-31G

   Each UKS state is converged from three guesses and followed to an
   internally STABLE state by common.run_open_shell (mf_factory); PySCF's UKS
   stability for a meta-GGA goes through `gen_response`, which carries the
   tau-dependent kernel in PySCF 2.13. An unstable state is refused.

   NH2 UKS additionally carries a central finite difference of PySCF's own
   UKS energy (`fd_check`) — a check that the analytic grid-response reference
   is the derivative of the energy it is paired with.

3. Open shell, ROKS SCAN and r2SCAN, RI-J: nh2 x 6-31G. The ROKS reference is
   a CENTRAL FINITE DIFFERENCE of PySCF's ROKS energy (`fd`), at three steps
   h = 2e-3, 1e-3, 5e-4 Bohr, each displaced SCF seeded from the reference
   density; the Richardson combination (4 g(h/2) - g(h)) / 3 of the two
   smallest steps is recorded as `fd.gradient`, and the step convergence
   max|g(1e-3) - g(5e-4)| and the Richardson-vs-h=5e-4 gap are recorded next to
   it. PySCF's analytic ROKS gradient (`nuc_grad_method()` — which exists and
   runs for meta-GGA in PySCF 2.13, via the UKS machinery on ROKS orbitals) is
   recorded as a SECOND, independent number, and its gap to the FD is stored;
   the Rust test asserts against the FD only.

LIKE-FOR-LIKE RECIPE

  * basis + aux: ferric's bundled JSON via common.py; aux AO count checked.
  * grid: (75,110) unpruned, Becke partition (PySCF default original_becke),
    Becke (1988) radii adjustment, PySCF's default Treutler-Ahlrichs radial
    grid — the recipe scripts/validation/gen_ks_energies.py matches ferric's
    KS energies with to ~1e-12 Ha (PBE/B3LYP). ferric's KS gradient
    (ks_gradient.rs) builds its grid from `AtomicGridConfig::default()` =
    (75,110) flat, so the SCF and gradient grids are the same on both sides.
  * geometry in Bohr via ferric's constant (common.read_xyz).
  * convergence: conv_tol 1e-11 / conv_tol_grad 1e-8 (analytic), 1e-12 / 1e-8
    for every FD energy.

Run (light; a few minutes, dominated by the ROKS FD sweep):
    scripts/validation/run_slot.sh --light -- \\
        /home/matt/qc/ferric/.venv/bin/python scripts/validation/gen_mgga_gradients.py
Restrict to systems by name:  ... gen_mgga_gradients.py nh2
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402

ROW = "mgga_gradients"
ROW_NAME = "Meta-GGA gradient, closed / open"
AUX = "def2-universal-jkfit"
MAIN_GRID = (75, 110)
CONV_TOL = 1e-11
CONV_TOL_GRAD = 1e-8
FD_CONV_TOL = 1e-12
FD_CONV_TOL_GRAD = 1e-8
FD_STEPS = (2e-3, 1e-3, 5e-4)  # Bohr
UKS_FD_STEPS = (1e-3, 5e-4)
GUESSES = ("minao", "atom", "huckel")
MAX_STAB_ROUNDS = 10

PYSCF_XC = {"scan": "SCAN", "r2scan": "R2SCAN"}
XCS = ("scan", "r2scan")

CLOSED_SHELL = {"h2o": (0, 1), "nh3": (0, 1)}
CLOSED_SHELL_BASES = ("6-31g", "def2-svp")
J_MODES = ("rij", "exact")

OPEN_SHELL = {"ho2": (0, 2), "nh2": (0, 2)}
OPEN_SHELL_BASES = ("6-31g",)
ROKS_SYSTEMS = ("nh2",)
UKS_FD_SYSTEMS = ("nh2",)

J_RECIPE = {
    "rij": "RI-J: PySCF density_fit(aux from ferric JSON), gradient with auxbasis_response; "
    "ferric default (df_j_aux unset -> def2-universal-jkfit) / explicit Some(aux) open shell",
    "exact": 'exact four-centre J: plain PySCF KS; ferric df_j_aux = Some("")',
}


def _aux_basis(symbols):
    aux, cart = common.pyscf_basis(AUX, symbols)
    if cart:
        raise ValueError(f"{AUX}: Cartesian l>=2 aux shells; not matched like-for-like")
    return aux


def _check_aux(mol, aux, symbols) -> dict:
    from pyscf.df import addons

    auxmol = addons.make_auxmol(mol, aux)
    want = common.ferric_nao(AUX, symbols)
    assert auxmol.nao_nr() == want, (
        f"{AUX}: PySCF aux nao {auxmol.nao_nr()} != ferric {want}"
    )
    return {"aux_nao": want}


def _configure(mf, conv_tol=CONV_TOL, conv_tol_grad=CONV_TOL_GRAD):
    from pyscf import dft

    mf.grids.atom_grid = MAIN_GRID
    mf.grids.prune = None
    mf.grids.radii_adjust = dft.radi.becke_atomic_radii_adjust
    mf.conv_tol = conv_tol
    mf.conv_tol_grad = conv_tol_grad
    mf.max_cycle = 500
    mf.verbose = 0
    return mf


def _make(kind: str, mol, aux, xc: str, j: str, **conv):
    from pyscf import dft

    cls = {"rks": dft.RKS, "uks": dft.UKS, "roks": dft.ROKS}[kind]
    mf = cls(mol, xc=PYSCF_XC[xc])
    if j == "rij":
        mf = mf.density_fit(auxbasis=aux)
    elif j != "exact":
        raise ValueError(j)
    return _configure(mf, **conv)


def _analytic_gradient(mf):
    g = mf.nuc_grad_method()
    g.grid_response = True
    g.verbose = 0
    return g.kernel()


def _as_rows(a) -> list[list[float]]:
    return [[float(v) for v in row] for row in a]


def _fd_gradient(mol, aux, kind, xc, j, dm_ref, e_ref, h) -> list[list[float]]:
    """Central FD of the `kind` energy, every displaced SCF seeded from dm_ref."""
    import numpy as np

    coords = mol.atom_coords(unit="Bohr")
    grad = np.zeros_like(coords)
    for a in range(coords.shape[0]):
        for c in range(3):
            e = []
            for sign in (+1.0, -1.0):
                x = coords.copy()
                x[a, c] += sign * h
                m = mol.set_geom_(x, unit="Bohr", inplace=False)
                mf = _make(
                    kind,
                    m,
                    aux,
                    xc,
                    j,
                    conv_tol=FD_CONV_TOL,
                    conv_tol_grad=FD_CONV_TOL_GRAD,
                )
                ei = mf.kernel(dm0=dm_ref)
                if not mf.converged:
                    raise RuntimeError(
                        f"FD {kind} {xc} atom={a} c={c} sign={sign} diverged"
                    )
                # Same basin: a displacement of 1e-3 Bohr moves E by << 1e-3 Ha.
                if abs(ei - e_ref) > 1e-3:
                    raise RuntimeError(
                        f"FD {kind} {xc} atom={a} c={c}: E jumped {ei - e_ref:+.3e} (basin change)"
                    )
                e.append(ei)
            grad[a, c] = (e[0] - e[1]) / (2.0 * h)
    return grad


def _prov(xyz, basis_name, symbols, coords, keywords, stability, extra):
    import numpy
    import pyscf

    keywords = dict(keywords)
    keywords["numpy"] = numpy.__version__
    return common.provenance(
        code="PySCF",
        version=pyscf.__version__,
        keywords=keywords,
        basis_name=basis_name,
        xyz_path=xyz,
        coords_bohr=coords,
        symbols=symbols,
        grid={
            "main": {
                "atom_grid": list(MAIN_GRID),
                "prune": None,
                "partition": "Becke (original_becke)",
                "radii_adjust": "becke_atomic_radii_adjust",
                "radial": "PySCF default (treutler_ahlrichs)",
            },
            "grid_response": True,
        },
        aux={
            "name": AUX,
            "json": str(common.basis_json_path(AUX).relative_to(common.ROOT)),
            "sha256": common.sha256_file(common.basis_json_path(AUX)),
            "used_by": "the *_rij blocks only; *_exact blocks are four-centre J",
        },
        frozen_core=None,
        scf_conv={
            "analytic": {"conv_tol": CONV_TOL, "conv_tol_grad": CONV_TOL_GRAD},
            "fd": {"conv_tol": FD_CONV_TOL, "conv_tol_grad": FD_CONV_TOL_GRAD},
        },
        stability=stability,
        generator="scripts/validation/gen_mgga_gradients.py",
        extra=extra,
    )


def _header(system, basis_name, charge, mult, mol):
    return {
        "row": ROW_NAME,
        "system": system,
        "basis": basis_name,
        "charge": charge,
        "multiplicity": mult,
        "nao": mol.nao_nr(),
        "nuclear_repulsion": float(mol.energy_nuc()),
        "units": {"energy": "Hartree", "gradient": "Hartree/Bohr, dE/dR, (natoms, 3)"},
    }


def _max_abs_diff(a, b) -> float:
    import numpy as np

    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def gen_closed_shell(system, charge, mult) -> list[Path]:
    xyz = common.MOL_DIR / f"{system}.xyz"
    symbols, coords = common.read_xyz(xyz)
    written = []
    for basis_name in CLOSED_SHELL_BASES:
        mol = common.build_pyscf_mol(xyz, basis_name, charge=charge, multiplicity=mult)
        basis_check = common.check_basis_like_for_like(mol, basis_name, symbols)
        aux = _aux_basis(symbols)
        basis_check.update(_check_aux(mol, aux, symbols))
        payload = _header(system, basis_name, charge, mult, mol)
        for xc in XCS:
            for j in J_MODES:
                mf = _make("rks", mol, aux, xc, j)
                e = mf.kernel()
                if not mf.converged:
                    raise RuntimeError(
                        f"{system}/{basis_name} RKS {xc} {j} did not converge"
                    )
                stable = bool(
                    mf.stability(internal=True, external=False, return_status=True)[2]
                )
                g = _analytic_gradient(mf)
                payload[f"rks_{xc}_{j}"] = {
                    "xc": PYSCF_XC[xc],
                    "j": J_RECIPE[j],
                    "energy": float(e),
                    "converged": True,
                    "internal_stable": stable,
                    "gradient": _as_rows(g),
                }
                print(
                    f"{system:4s} {basis_name:9s} RKS {xc:6s} {j:5s} E={e:.10f} "
                    f"|g|max={abs(g).max():.4e} stable={stable}"
                )
        # Separations the Rust negative controls rely on, recorded for review.
        payload["separations"] = {
            f"scan_vs_r2scan_{j}": _max_abs_diff(
                payload[f"rks_scan_{j}"]["gradient"],
                payload[f"rks_r2scan_{j}"]["gradient"],
            )
            for j in J_MODES
        } | {
            f"rij_vs_exact_{xc}": _max_abs_diff(
                payload[f"rks_{xc}_rij"]["gradient"],
                payload[f"rks_{xc}_exact"]["gradient"],
            )
            for xc in XCS
        }
        print(f"{system:4s} {basis_name:9s} separations {payload['separations']}")
        payload["provenance"] = _prov(
            xyz,
            basis_name,
            symbols,
            coords,
            {
                "method": "dft.RKS, nuc_grad_method() with grid_response=True",
                "xc": {k: PYSCF_XC[k] for k in XCS},
                "j": J_RECIPE,
                "init_guess": "minao (PySCF default)",
            },
            None,
            {"basis_self_check": basis_check},
        )
        written.append(common.write_reference(ROW, system, basis_name, payload))
    return written


def _fd_block(mol, aux, kind, xc, mf, steps) -> dict:
    dm = mf.make_rdm1()
    grads = {}
    for h in steps:
        grads[h] = _fd_gradient(mol, aux, kind, xc, "rij", dm, mf.e_tot, h)
    h_small, h_mid = steps[-1], steps[-2]
    rich = (4.0 * grads[h_small] - grads[h_mid]) / 3.0
    block = {
        "steps_bohr": list(steps),
        "gradients_by_step": {f"{h:.0e}": _as_rows(grads[h]) for h in steps},
        "richardson_from": [h_mid, h_small],
        "gradient": _as_rows(rich),
        "step_convergence": {
            f"max|g({a:.0e})-g({b:.0e})|": _max_abs_diff(grads[a], grads[b])
            for a, b in zip(steps, steps[1:])
        },
        "richardson_vs_smallest_step": _max_abs_diff(rich, grads[h_small]),
        "scf": {"conv_tol": FD_CONV_TOL, "conv_tol_grad": FD_CONV_TOL_GRAD},
        "seed": "every displaced SCF starts from the reference-geometry density",
    }
    return block


def gen_open_shell(system, charge, mult) -> list[Path]:
    xyz = common.MOL_DIR / f"{system}.xyz"
    symbols, coords = common.read_xyz(xyz)
    written = []
    for basis_name in OPEN_SHELL_BASES:
        mol = common.build_pyscf_mol(xyz, basis_name, charge=charge, multiplicity=mult)
        basis_check = common.check_basis_like_for_like(mol, basis_name, symbols)
        aux = _aux_basis(symbols)
        basis_check.update(_check_aux(mol, aux, symbols))
        payload = _header(system, basis_name, charge, mult, mol)
        stab = {}
        for xc in XCS:
            res, mf = common.run_open_shell(
                mol,
                "uks",
                conv_tol=CONV_TOL,
                conv_tol_grad=CONV_TOL_GRAD,
                max_stab_rounds=MAX_STAB_ROUNDS,
                guesses=GUESSES,
                mf_factory=lambda m, xc=xc: _make("uks", m, aux, xc, "rij"),
                return_mf=True,
            )
            g = _analytic_gradient(mf)
            res["xc"] = PYSCF_XC[xc]
            res["j"] = J_RECIPE["rij"]
            res["gradient"] = _as_rows(g)
            if system in UKS_FD_SYSTEMS:
                fd = _fd_block(mol, aux, "uks", xc, mf, UKS_FD_STEPS)
                fd["analytic_vs_fd"] = _max_abs_diff(g, fd["gradient"])
                res["fd_check"] = fd
                print(
                    f"{system:4s} {basis_name:9s} UKS {xc:6s} analytic-vs-FD "
                    f"{fd['analytic_vs_fd']:.2e} steps {fd['step_convergence']}"
                )
            payload[f"uks_{xc}_rij"] = res
            stab[f"uks_{xc}_rij"] = res["stability"]
            print(
                f"{system:4s} {basis_name:9s} UKS {xc:6s} E={res['energy']:.10f} "
                f"<S2>={res['s_squared']:.6f} lmin={res['stability']['lambda_min']:+.3e} "
                f"rounds={res['stability']['rounds']} multi={res['multiple_stable_minima']} "
                f"|g|max={abs(g).max():.4e}"
            )
        if system in ROKS_SYSTEMS:
            for xc in XCS:
                res, mf = common.run_open_shell(
                    mol,
                    "rohf",
                    conv_tol=CONV_TOL,
                    conv_tol_grad=CONV_TOL_GRAD,
                    max_stab_rounds=MAX_STAB_ROUNDS,
                    guesses=GUESSES,
                    mf_factory=lambda m, xc=xc: _make("roks", m, aux, xc, "rij"),
                    return_mf=True,
                )
                res["stability"]["kind"] = "PySCF ROKS internal (real)"
                fd = _fd_block(mol, aux, "roks", xc, mf, FD_STEPS)
                g_an = _analytic_gradient(mf)
                block = {
                    "xc": PYSCF_XC[xc],
                    "j": J_RECIPE["rij"],
                    "energy": res["energy"],
                    "converged": True,
                    "stability": res["stability"],
                    "guess_scan": res["guess_scan"],
                    "multiple_stable_minima": res["multiple_stable_minima"],
                    "fd": fd,
                    "pyscf_analytic_gradient": _as_rows(g_an),
                    "pyscf_analytic_vs_fd": _max_abs_diff(g_an, fd["gradient"]),
                    "reference": "fd.gradient (Richardson of the two smallest steps)",
                }
                payload[f"roks_{xc}_rij"] = block
                stab[f"roks_{xc}_rij"] = res["stability"]
                print(
                    f"{system:4s} {basis_name:9s} ROKS {xc:6s} E={res['energy']:.10f} "
                    f"steps {fd['step_convergence']} rich-vs-small "
                    f"{fd['richardson_vs_smallest_step']:.2e} pyscf-analytic-vs-FD "
                    f"{block['pyscf_analytic_vs_fd']:.2e}"
                )
        seps = {
            "uks_scan_vs_r2scan": _max_abs_diff(
                payload["uks_scan_rij"]["gradient"],
                payload["uks_r2scan_rij"]["gradient"],
            )
        }
        if system in ROKS_SYSTEMS:
            seps["roks_scan_vs_r2scan"] = _max_abs_diff(
                payload["roks_scan_rij"]["fd"]["gradient"],
                payload["roks_r2scan_rij"]["fd"]["gradient"],
            )
            for xc in XCS:
                seps[f"uks_vs_roks_{xc}"] = _max_abs_diff(
                    payload[f"uks_{xc}_rij"]["gradient"],
                    payload[f"roks_{xc}_rij"]["fd"]["gradient"],
                )
        payload["separations"] = seps
        print(f"{system:4s} {basis_name:9s} separations {seps}")
        payload["provenance"] = _prov(
            xyz,
            basis_name,
            symbols,
            coords,
            {
                "method": "dft.UKS (analytic, grid_response=True)"
                + (
                    " + dft.ROKS (central FD of the energy)"
                    if system in ROKS_SYSTEMS
                    else ""
                ),
                "xc": {k: PYSCF_XC[k] for k in XCS},
                "j": J_RECIPE["rij"],
                "init_guess_scan": list(GUESSES),
                "stability": "internal=True external=False, restart from the unstable "
                f"direction, max {MAX_STAB_ROUNDS} rounds (common.run_open_shell)",
                "fd_steps_bohr": {
                    "roks": list(FD_STEPS),
                    "uks_fd_check": list(UKS_FD_STEPS),
                },
            },
            stab,
            {"basis_self_check": basis_check},
        )
        written.append(common.write_reference(ROW, system, basis_name, payload))
    return written


def main() -> int:
    only = set(sys.argv[1:])
    written = []
    for system, (charge, mult) in CLOSED_SHELL.items():
        if not only or system in only:
            written += gen_closed_shell(system, charge, mult)
    for system, (charge, mult) in OPEN_SHELL.items():
        if not only or system in only:
            written += gen_open_shell(system, charge, mult)
    print(f"GEN_MGGA_GRADIENTS_DONE written={len(written)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
