"""PySCF references for the VALIDATION.md "KS-DFT, ROKS" row.

Consumer: crates/ferric-scf/tests/validation_roks_energies.rs.
Output:   testdata/reference/validation/roks_energies/<system>_<basis>.json

WHAT IS COMPARED
----------------
ferric's ROKS (`ferric_scf::rohf::solve_rohf` with `RhfConfig::xc` set, the
path the CLI runs for `method.kind = "rohf"` + `[dft] functional`,
crates/ferric-cli/src/lib.rs `run_rohf`) against PySCF 2.13 `dft.ROKS`:

    nh2   NH2  2B1   doublet
    ch3   CH3  2A2'' doublet (planar D3h)
    ho2   HO2  2A''  doublet
    oh    OH   2Pi   doublet   <- DEGENERATE pi hole (the F6 ROKS lane)

x PBE, B3LYP x 6-31G and def2-SVP.

LIKE-FOR-LIKE RECIPE (read from ferric's code, not a doc)
---------------------------------------------------------
* basis / aux / geometry: ferric's bundled JSON and Bohr geometry via
  common.py; aux def2-universal-jkfit, AO count checked against ferric's parser.
* grid: (75,110) unpruned Becke, Becke-1988 radii adjust -- the recipe of
  gen_ks_energies.py, which ferric's UKS already matches to ~1e-12 Ha.
* J/K: ferric's solve_rohf uses `crate::fock_assembly::build_df_jk` with
  `j_aux_eff = df_j_aux` (omega == 0) and `k_aux_eff = df_k_aux` when exact
  exchange is used -- the SAME resolution as solve_uhf, and the CLI's
  `scf_xc_and_aux_defaults` puts def2-universal-jkfit in both for a
  functional. So PBE: RI-J (K unused); B3LYP: RI-J + RI-K. PySCF:
  `dft.ROKS(...).density_fit(auxbasis=aux)` == RI-JK with that aux.
* Coupling: both codes build the Guest-Saunders effective Fock (ferric
  rohf.rs `roothaan_fock` is a port of PySCF `rohf.get_roothaan_fock`). The
  converged ENERGY does not depend on the coupling choice (only the
  stationarity condition does, and that is coupling-independent), so the
  energy is the like-for-like quantity. Roothaan orbital energies ARE
  coupling-dependent and are recorded for inspection only.

STATE SELECTION
---------------
Each (system, basis, functional) is started from three guesses x two level
shifts (0.0, 0.5), each DIIS run polished with PySCF's second-order solver
(`mf.newton()`), and a start is ACCEPTED when its final max orbital gradient
is <= ACCEPT_GMAX. The lowest accepted energy is the reference; its internal
ROKS stability (PySCF `rohf_internal`, real) must be stable or the reference
is refused. Every start is recorded in `start_scan`, and `accepted_spread`
is max - min energy over accepted starts.

OH: PySCF's OWN ROKS is unstable as an SCF on this system. Measured
(2026-09-25, 6-31G, PBE): plain DIIS from minao/atom/huckel never meets
conv_tol 1e-10 in 300 cycles; the Newton polish from those points ends at
-75.6208455 .. -75.6208464 Ha, i.e. a 1e-6 Ha spread between converged-looking
points. That is the grid anisotropy of the pi-hole orientation (the Lebedev
grid is 4-fold symmetric about the OH axis, not cylindrical), the same
~1e-6 Ha spread ferric's own tests/rohf_degenerate_open_shell.rs measures.
The reference is therefore the lowest ACCEPTED point and the Rust bar for OH
covers `accepted_spread`; a state change (sigma* occupied, the 0.58 Ha F6
defect) is >= 0.1 Ha away and cannot hide under it.

CONTROLS written alongside (consumed by the Rust negative checks):
  * `uks_<xc>_control`: stability-followed UKS with the same recipe
    (gen_ks_energies._uks_factory). ROKS is a constrained UKS, so its energy
    must lie ABOVE the UKS minimum (spin contamination makes them differ).
  * `rohf_control`: exact-integral ROHF, stability-followed. ferric ROKS must
    MISS it by > 1e-2 Ha (the functional is applied).

Run (light):
    OPENBLAS_NUM_THREADS=1 scripts/validation/run_slot.sh --light -- \\
        uv run --no-sync python scripts/validation/gen_roks_energies.py [system ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402
import gen_ks_energies as ks  # noqa: E402

ROW = "roks_energies"
ROW_NAME = "KS-DFT, ROKS"
SYSTEMS = {  # system -> (charge, multiplicity)
    "nh2": (0, 2),
    "ch3": (0, 2),
    "ho2": (0, 2),
    "oh": (0, 2),
}
BASES = ("6-31g", "def2-svp")
XCS = ("pbe", "b3lyp")
GUESSES = ("minao", "atom", "huckel")
LEVEL_SHIFTS = (0.0, 0.5)
CONV_TOL = 1e-10
CONV_TOL_GRAD = 1e-7
# A start is accepted when PySCF's own max |orbital gradient| is below this.
# The energy error is second order in it (<~1e-10 Ha).
ACCEPT_GMAX = 1e-5

JK_RECIPE = {
    "pbe": "RI-J (density_fit, aux fed from ferric JSON); K unused",
    "b3lyp": "RI-J + RI-K (density_fit)",
}


def _roks_factory(aux, xc):
    from pyscf import dft

    def factory(mol):
        mf = dft.ROKS(mol, xc=ks.PYSCF_XC[xc]).density_fit(auxbasis=aux)
        return ks._configure_grids(mf, xc)

    return factory


def run_roks(mol, aux, xc) -> dict:
    """Multi-start ROKS with a Newton polish; see the module doc."""
    import numpy as np

    factory = _roks_factory(aux, xc)
    scan, best = [], None
    for guess in GUESSES:
        for ls in LEVEL_SHIFTS:
            mf = factory(mol)
            mf.init_guess = guess
            mf.level_shift = ls
            mf.max_cycle = 300
            mf.kernel()
            diis_converged = bool(mf.converged)
            e_diis = float(mf.e_tot)
            nt = mf.newton()
            nt.conv_tol = CONV_TOL
            nt.conv_tol_grad = CONV_TOL_GRAD
            nt.max_cycle = 100
            nt.verbose = 0
            nt.kernel(mf.mo_coeff, mf.mo_occ)
            gmax = float(np.abs(nt.get_grad(nt.mo_coeff, nt.mo_occ)).max())
            accepted = gmax <= ACCEPT_GMAX
            entry = {
                "guess": guess,
                "level_shift": ls,
                "diis_converged": diis_converged,
                "energy_diis": e_diis,
                "newton_converged": bool(nt.converged),
                "energy": float(nt.e_tot),
                "max_orbital_gradient": gmax,
                "accepted": accepted,
            }
            scan.append(entry)
            if accepted and (best is None or nt.e_tot < best[0].e_tot):
                best = (nt, entry)
    if best is None:
        raise RuntimeError(
            f"ROKS {xc}: no start reached max|g| <= {ACCEPT_GMAX}: {scan}"
        )
    mf, entry = best
    # Internal ROKS stability of the selected state (real rotations).
    _mo, stable = common._stability_status(mf)
    if not stable:
        raise RuntimeError(
            f"ROKS {xc}: selected state is internally UNSTABLE ({entry})"
        )
    acc = [s["energy"] for s in scan if s["accepted"]]
    na, nb = mol.nelec
    e = mf.mo_energy
    return {
        "energy": float(mf.e_tot),
        "converged": True,
        "max_orbital_gradient": entry["max_orbital_gradient"],
        "nelec_alpha": int(na),
        "nelec_beta": int(nb),
        "stability": {
            "internal_stable": True,
            "kind": "PySCF ROKS internal (real), rohf_internal",
            "selected_start": {
                "guess": entry["guess"],
                "level_shift": entry["level_shift"],
            },
        },
        "start_scan": scan,
        "n_accepted": len(acc),
        "accepted_spread": float(max(acc) - min(acc)),
        "orbital_energy_convention": (
            "PySCF ROKS Roothaan (Guest-Saunders) effective-Fock eigenvalues; "
            "recorded for inspection, NOT compared"
        ),
        "somo_roothaan": [float(x) for x in e[nb:na]],
        "homo_docc_roothaan": float(e[nb - 1]),
        "lumo_roothaan": float(e[na]),
        "xc": ks.PYSCF_XC[xc],
        "jk": JK_RECIPE[xc],
    }


def gen(system, charge, mult) -> list[Path]:
    import numpy
    import pyscf

    xyz = common.MOL_DIR / f"{system}.xyz"
    symbols, coords = common.read_xyz(xyz)
    written = []
    for basis_name in BASES:
        mol = common.build_pyscf_mol(xyz, basis_name, charge=charge, multiplicity=mult)
        basis_check = common.check_basis_like_for_like(mol, basis_name, symbols)
        aux = ks._aux_basis(symbols)
        basis_check.update(ks._check_aux(mol, aux, symbols))
        payload = {
            "row": ROW_NAME,
            "system": system,
            "basis": basis_name,
            "charge": charge,
            "multiplicity": mult,
            "nao": mol.nao_nr(),
            "nuclear_repulsion": float(mol.energy_nuc()),
        }
        stab = {}
        for xc in XCS:
            res = run_roks(mol, aux, xc)
            payload[f"roks_{xc}"] = res
            stab[f"roks_{xc}"] = res["stability"]
            print(
                f"{system:4s} {basis_name:9s} ROKS {xc:6s} {res['energy']:.10f} "
                f"accepted {res['n_accepted']}/{len(res['start_scan'])} "
                f"spread {res['accepted_spread']:.2e} gmax {res['max_orbital_gradient']:.1e}",
                flush=True,
            )
            uks = common.run_open_shell(
                mol,
                "uks",
                conv_tol=CONV_TOL,
                conv_tol_grad=CONV_TOL_GRAD,
                guesses=GUESSES,
                mf_factory=ks._uks_factory(aux, xc),
            )
            payload[f"uks_{xc}_control"] = {
                "energy": uks["energy"],
                "s_squared": uks["s_squared"],
                "stability": uks["stability"],
                "multiple_stable_minima": uks["multiple_stable_minima"],
                "note": "stability-followed UKS, same recipe; ferric ROKS must lie ABOVE it",
            }
            print(
                f"{system:4s} {basis_name:9s} UKS  {xc:6s} {uks['energy']:.10f} "
                f"<S2>={uks['s_squared']:.6f}  ROKS-UKS {res['energy'] - uks['energy']:+.3e}",
                flush=True,
            )
        rohf = common.run_open_shell(mol, "rohf", conv_tol=1e-11, conv_tol_grad=1e-8)
        payload["rohf_control"] = {
            "energy": rohf["energy"],
            "stability": rohf["stability"],
            "note": "exact-integral ROHF; ferric ROKS must MISS this (functional applied)",
        }
        payload["provenance"] = common.provenance(
            code="PySCF",
            version=pyscf.__version__,
            keywords={
                "method": "dft.ROKS (+ dft.UKS and scf.ROHF controls)",
                "xc": {k: ks.PYSCF_XC[k] for k in XCS},
                "jk": JK_RECIPE,
                "starts": {
                    "init_guess": list(GUESSES),
                    "level_shift": list(LEVEL_SHIFTS),
                },
                "polish": f"mf.newton(), conv_tol {CONV_TOL}, conv_tol_grad {CONV_TOL_GRAD}",
                "accept": f"max |orbital gradient| <= {ACCEPT_GMAX}",
                "numpy": numpy.__version__,
            },
            basis_name=basis_name,
            xyz_path=xyz,
            coords_bohr=coords,
            symbols=symbols,
            grid={
                "main": {
                    "atom_grid": list(ks.MAIN_GRID),
                    "prune": None,
                    "partition": "Becke (original_becke)",
                    "radii_adjust": "becke_atomic_radii_adjust",
                },
            },
            aux={
                "name": ks.AUX,
                "json": str(common.basis_json_path(ks.AUX).relative_to(common.ROOT)),
                "sha256": common.sha256_file(common.basis_json_path(ks.AUX)),
            },
            frozen_core=None,
            scf_conv={"conv_tol": CONV_TOL, "conv_tol_grad": CONV_TOL_GRAD},
            stability=stab,
            generator="scripts/validation/gen_roks_energies.py",
            extra={"basis_self_check": basis_check},
        )
        written.append(common.write_reference(ROW, system, basis_name, payload))
    return written


def main() -> int:
    only = set(sys.argv[1:])
    written = []
    for system, (charge, mult) in SYSTEMS.items():
        if not only or system in only:
            written += gen(system, charge, mult)
    print(f"GEN_ROKS_ENERGIES_DONE written={len(written)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
