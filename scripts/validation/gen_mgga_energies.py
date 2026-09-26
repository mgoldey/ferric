"""PySCF references for the VALIDATION.md row "SCAN/r2SCAN energy" (validation
tier W3).

Consumer: crates/ferric-scf/tests/validation_mgga_energies.rs.
Output:   testdata/reference/validation/mgga_energies/<system>_<basis>.json

CASES

1. Closed shell, RKS SCAN and r2SCAN:

       h2o, nh3, h2s, ch4  x  def2-SVP, def2-TZVP  x  {RI-J, exact J}

   RI-J is ferric's DEFAULT for every functional (rhf.rs `resolve_aux`: an
   unset `df_j_aux` auto-defaults to def2-universal-jkfit when `xc` is set);
   its PySCF side is `density_fit(aux)` with the aux fed from ferric's JSON.
   The exact-J block is plain PySCF RKS against ferric `df_j_aux = Some("")`.
   Each RKS state is checked internally stable with PySCF's `stability()`.
   Control `rhf_control`: exact-integral RHF, which ferric RKS must miss.

2. Open shell, UKS SCAN and r2SCAN, RI-J: nh2 (2B1)  x  def2-SVP, def2-TZVP.
   Converged from three guesses and followed to an internally STABLE state by
   common.run_open_shell (PySCF's UKS stability for a meta-GGA goes through
   `gen_response`, which carries the tau-dependent kernel in PySCF 2.13). An
   unstable state is refused. Control `uhf_control`: exact UHF.

LIKE-FOR-LIKE RECIPE (read from the ferric code)

  * basis + aux: ferric's bundled JSON via common.py; AO and aux AO counts
    checked against ferric's parser.
  * grid: (75,110) unpruned, Becke partition (PySCF default original_becke),
    Becke (1988) radii adjustment, PySCF's default Treutler-Ahlrichs radial
    grid; `small_rho_cutoff` is PySCF's RKS/UKS default 0 (no grid pruning;
    ferric prunes none either). With this recipe gen_mgga_gradients.py
    matches ferric's SCAN/r2SCAN energies to 5.7e-13 Ha at 6-31G/def2-SVP.
  * XC density floor: ferric (vxc.rs / xc_batch.rs, DENSITY_FLOOR = 1e-10)
    zeroes EVERY V_xc factor (v_rho, v_sigma, v_tau) at points where
    rho <= 1e-10 — the total rho for RKS, rho_sigma per spin for UKS — while
    E_xc sums every point unfloored. `_apply_ferric_density_floor` does exactly
    that to PySCF's `eval_xc_eff` output. Each block also records
    `density_floor_shift` = E(floored) - E(unfloored), measured by a second,
    unfloored SCF seeded from the floored density, so the size of this
    emulation is on file.
  * ferric's meta-GGA SCF adds a default 0.5 Ha virtual-block level shift that
    is ramped to 0 at convergence; it changes the path, not the converged
    energy, so PySCF runs without one.
  * geometry in Bohr via ferric's constant (common.read_xyz).
  * convergence: conv_tol 1e-11 / conv_tol_grad 1e-8.

SEPARATIONS written for the Rust negative controls (`separations`):
SCAN vs r2SCAN per J mode, RI-J vs exact J per functional. The basis-swap
control reads the other basis's file.

Run (light; single-threaded, well under 20 min):
    OPENBLAS_NUM_THREADS=1 scripts/validation/run_slot.sh --light -- \\
        uv run --no-sync python scripts/validation/gen_mgga_energies.py
Restrict to systems by name:  ... gen_mgga_energies.py h2s nh2
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402

ROW = "mgga_energies"
ROW_NAME = "SCAN/r2SCAN energy"
AUX = "def2-universal-jkfit"
MAIN_GRID = (75, 110)
CONV_TOL = 1e-11
CONV_TOL_GRAD = 1e-8
GUESSES = ("minao", "atom", "huckel")
MAX_STAB_ROUNDS = 10
# ferric's XC density floor, crates/ferric-dft/src/vxc.rs `DENSITY_FLOOR`.
FERRIC_DENSITY_FLOOR = 1e-10

PYSCF_XC = {"scan": "SCAN", "r2scan": "R2SCAN"}
XCS = ("scan", "r2scan")
BASES = ("def2-svp", "def2-tzvp")

CLOSED_SHELL = {"h2o": (0, 1), "nh3": (0, 1), "h2s": (0, 1), "ch4": (0, 1)}
J_MODES = ("rij", "exact")
OPEN_SHELL = {"nh2": (0, 2)}

J_RECIPE = {
    "rij": "RI-J: PySCF density_fit(aux from ferric JSON); ferric default "
    "(df_j_aux unset -> def2-universal-jkfit) closed shell / explicit Some(aux) open shell",
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


def _apply_ferric_density_floor(mf, floor=FERRIC_DENSITY_FLOOR):
    """ferric's XC floor on PySCF's `eval_xc_eff`: every V_xc component is
    zeroed where rho <= floor (RKS: total rho; UKS: rho_sigma, per spin);
    exc is left UNFLOORED, as ferric sums E_xc over every point. Returns a
    dict counting calls, so the caller can prove the wrapper was reached."""
    ni = mf._numint
    orig = ni.eval_xc_eff
    stats = {"calls": 0}

    def floored(
        xc_code, rho, deriv=1, omega=None, xctype=None, verbose=None, spin=None
    ):
        exc, vxc, fxc, kxc = orig(xc_code, rho, deriv, omega, xctype, verbose, spin)
        stats["calls"] += 1
        if vxc is None:
            return exc, vxc, fxc, kxc
        vxc = vxc.copy()
        polarized = spin == 1 or (hasattr(rho, "ndim") and rho.ndim == 3)
        if polarized:
            for s in range(2):
                rs = rho[s][0] if rho[s].ndim > 1 else rho[s]
                vxc[s][..., rs <= floor] = 0.0
        else:
            r0 = rho[0] if rho.ndim > 1 else rho
            vxc[..., r0 <= floor] = 0.0
        return exc, vxc, fxc, kxc

    ni.eval_xc_eff = floored
    return stats


def _make(kind: str, mol, aux, xc: str, j: str, floor: bool = True):
    from pyscf import dft

    cls = {"rks": dft.RKS, "uks": dft.UKS}[kind]
    mf = cls(mol, xc=PYSCF_XC[xc])
    if j == "rij":
        mf = mf.density_fit(auxbasis=aux)
    elif j != "exact":
        raise ValueError(j)
    mf.grids.atom_grid = MAIN_GRID
    mf.grids.prune = None
    mf.grids.radii_adjust = dft.radi.becke_atomic_radii_adjust
    mf.small_rho_cutoff = 0.0
    mf.conv_tol = CONV_TOL
    mf.conv_tol_grad = CONV_TOL_GRAD
    mf.max_cycle = 500
    mf.verbose = 0
    mf.ferric_floor_stats = _apply_ferric_density_floor(mf) if floor else None
    return mf


def _unfloored_shift(kind, mol, aux, xc, j, mf_floored) -> float:
    """E(floored) - E(unfloored): the unfloored SCF is seeded from the floored
    density, so it lands in the same basin."""
    mf = _make(kind, mol, aux, xc, j, floor=False)
    e = mf.kernel(dm0=mf_floored.make_rdm1())
    if not mf.converged:
        raise RuntimeError(f"unfloored {kind} {xc} {j} did not converge")
    return float(mf_floored.e_tot - e)


def _rhf_control(mol) -> float:
    from pyscf import scf

    mf = scf.RHF(mol)
    mf.conv_tol = CONV_TOL
    mf.verbose = 0
    e = mf.kernel()
    if not mf.converged:
        raise RuntimeError("RHF control did not converge")
    return float(e)


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
                "small_rho_cutoff": 0.0,
            },
            "xc_density_floor": {
                "value": FERRIC_DENSITY_FLOOR,
                "rule": "V_xc (v_rho, v_sigma, v_tau) zeroed where rho <= floor "
                "(RKS total rho; UKS per spin); E_xc unfloored (ferric vxc.rs/xc_batch.rs)",
            },
        },
        aux={
            "name": AUX,
            "json": str(common.basis_json_path(AUX).relative_to(common.ROOT)),
            "sha256": common.sha256_file(common.basis_json_path(AUX)),
            "used_by": "the *_rij blocks only; *_exact blocks are four-centre J",
        },
        frozen_core=None,
        scf_conv={"conv_tol": CONV_TOL, "conv_tol_grad": CONV_TOL_GRAD},
        stability=stability,
        generator="scripts/validation/gen_mgga_energies.py",
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
        "units": {"energy": "Hartree"},
    }


def gen_closed_shell(system, charge, mult) -> list[Path]:
    xyz = common.MOL_DIR / f"{system}.xyz"
    symbols, coords = common.read_xyz(xyz)
    written = []
    for basis_name in BASES:
        mol = common.build_pyscf_mol(xyz, basis_name, charge=charge, multiplicity=mult)
        basis_check = common.check_basis_like_for_like(mol, basis_name, symbols)
        aux = _aux_basis(symbols)
        basis_check.update(_check_aux(mol, aux, symbols))
        payload = _header(system, basis_name, charge, mult, mol)
        nocc = mol.nelectron // 2
        for xc in XCS:
            for j in J_MODES:
                mf = _make("rks", mol, aux, xc, j)
                e = mf.kernel()
                if not mf.converged:
                    raise RuntimeError(
                        f"{system}/{basis_name} RKS {xc} {j} did not converge"
                    )
                assert mf.ferric_floor_stats["calls"] > 0, "density floor never applied"
                stable = bool(
                    mf.stability(internal=True, external=False, return_status=True)[2]
                )
                if not stable:
                    raise RuntimeError(
                        f"{system}/{basis_name} RKS {xc} {j}: state is internally UNSTABLE"
                    )
                shift = _unfloored_shift("rks", mol, aux, xc, j, mf)
                payload[f"rks_{xc}_{j}"] = {
                    "xc": PYSCF_XC[xc],
                    "j": J_RECIPE[j],
                    "energy": float(e),
                    "converged": True,
                    "internal_stable": stable,
                    "homo": float(mf.mo_energy[nocc - 1]),
                    "lumo": float(mf.mo_energy[nocc]),
                    "density_floor_shift": shift,
                }
                print(
                    f"{system:4s} {basis_name:9s} RKS {xc:6s} {j:5s} E={e:.12f} "
                    f"stable={stable} floor_shift={shift:+.2e}",
                    flush=True,
                )
        payload["rhf_control"] = {
            "energy": _rhf_control(mol),
            "note": "exact-integral RHF; ferric RKS must MISS this (functional applied)",
        }
        payload["separations"] = {
            f"scan_vs_r2scan_{j}": abs(
                payload[f"rks_scan_{j}"]["energy"]
                - payload[f"rks_r2scan_{j}"]["energy"]
            )
            for j in J_MODES
        } | {
            f"rij_vs_exact_{xc}": abs(
                payload[f"rks_{xc}_rij"]["energy"]
                - payload[f"rks_{xc}_exact"]["energy"]
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
                "method": "dft.RKS (+ scf.RHF control)",
                "xc": {k: PYSCF_XC[k] for k in XCS},
                "j": J_RECIPE,
                "init_guess": "minao (PySCF default)",
                "stability": "RKS internal (real) stability(), must be stable",
            },
            None,
            {"basis_self_check": basis_check},
        )
        written.append(common.write_reference(ROW, system, basis_name, payload))
    return written


def gen_open_shell(system, charge, mult) -> list[Path]:
    xyz = common.MOL_DIR / f"{system}.xyz"
    symbols, coords = common.read_xyz(xyz)
    written = []
    for basis_name in BASES:
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
            assert mf.ferric_floor_stats["calls"] > 0, "density floor never applied"
            res["xc"] = PYSCF_XC[xc]
            res["j"] = J_RECIPE["rij"]
            res["density_floor_shift"] = _unfloored_shift(
                "uks", mol, aux, xc, "rij", mf
            )
            # guess_scan trajectories are bulky and not consumed; keep the verdicts.
            for g in res["guess_scan"]:
                g.pop("energy_trajectory", None)
            payload[f"uks_{xc}_rij"] = res
            stab[f"uks_{xc}_rij"] = res["stability"]
            print(
                f"{system:4s} {basis_name:9s} UKS {xc:6s} E={res['energy']:.12f} "
                f"<S2>={res['s_squared']:.8f} lmin={res['stability']['lambda_min']:+.3e} "
                f"rounds={res['stability']['rounds']} multi={res['multiple_stable_minima']} "
                f"floor_shift={res['density_floor_shift']:+.2e}",
                flush=True,
            )
        uhf = common.run_open_shell(
            mol,
            "uhf",
            conv_tol=CONV_TOL,
            conv_tol_grad=CONV_TOL_GRAD,
            max_stab_rounds=MAX_STAB_ROUNDS,
            guesses=GUESSES,
        )
        payload["uhf_control"] = {
            "energy": uhf["energy"],
            "s_squared": uhf["s_squared"],
            "note": "exact-integral UHF; ferric UKS must MISS this (functional applied)",
        }
        payload["separations"] = {
            "uks_scan_vs_r2scan": abs(
                payload["uks_scan_rij"]["energy"] - payload["uks_r2scan_rij"]["energy"]
            )
        }
        payload["provenance"] = _prov(
            xyz,
            basis_name,
            symbols,
            coords,
            {
                "method": "dft.UKS (+ scf.UHF control)",
                "xc": {k: PYSCF_XC[k] for k in XCS},
                "j": J_RECIPE["rij"],
                "init_guess_scan": list(GUESSES),
                "stability": "internal=True external=False, restart from the unstable "
                f"direction, max {MAX_STAB_ROUNDS} rounds (common.run_open_shell)",
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
    print(f"GEN_MGGA_ENERGIES_DONE written={len(written)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
