"""PySCF references for the VALIDATION.md "KS-DFT" row (validation tier W1).

Consumer: crates/ferric-scf/tests/validation_ks_energies.rs.
Output:   testdata/reference/validation/ks_energies/<system>_<basis>.json

Two blocks of new coverage, beyond the closed-shell cc-pVDZ/def2-SVP set of
the older dft_{lda,pbe,b3lyp,wb97xv}.rs tests:

1. OPEN-SHELL UKS (not ROKS: ROKS on a degenerate shell is defect F6, fixed
   separately), x PBE, B3LYP, wB97X-V, x 6-31G and def2-SVP:

       nh2   NH2  2B1   doublet
       ch3   CH3  2A2'' doublet (planar D3h)
       ho2   HO2  2A''  doublet
       o2    O2   3Sg-  triplet

   Radicals with a DEGENERATE singly-occupied shell (OH, NO, CH: 2Pi) are
   deliberately left out. Their UKS is fine, but the symmetry-broken and the
   symmetric pi^3 solutions are near-degenerate, so which one each code lands
   on is a state-selection question (the F6/ROKS lane), and a mismatch there
   would be read as a KS-energy failure. Every system here has a
   non-degenerate SOMO (O2's two pi* SOMOs are both singly occupied, so the
   triplet is a single determinant with no pi^3 ambiguity).

   Each (system, basis, functional) is converged from three guesses and
   followed to an internally STABLE UKS state with PySCF's `stability()` loop
   (common.run_open_shell with an mf_factory); an unstable state is REFUSED.
   The record carries <S^2>, the frontier orbital energies and the lowest
   eigenvalue of PySCF's own (dense) UKS orbital Hessian.

   Controls written alongside (consumed by the Rust negative checks):
     * `uhf_control`: exact-integral UHF, stability-followed. ferric UKS must
       MISS it by > 1e-2 Ha, i.e. the functional is actually applied.
     * O2 only: <S^2> must be ~2. (UKS at multiplicity 3 fixes
       N_alpha - N_beta = 2, so it cannot become a singlet; a restricted
       singlet control was dropped -- closed-shell singlet O2 doubly occupies
       one of the degenerate pi* pair and would not converge in reasonable
       time, and it adds nothing the multiplicity does not already fix.)

2. CLOSED-SHELL second row, x PBE, B3LYP, x def2-SVP and def2-TZVP:

       h2s, hcl, sih4

   Control `rhf_control`: exact-integral RHF, which ferric RKS must miss.

LIKE-FOR-LIKE RECIPE (read the ferric code, not a doc: crates/ferric-scf/src/
rhf.rs `resolve_aux`, uhf.rs `j_aux_eff`/`k_aux_eff`, driver.rs RSH fitters):

  * basis: ferric's bundled JSON via common.py (never PySCF's built-in copy);
    the aux basis def2-universal-jkfit is ALSO fed from ferric's JSON, and its
    AO count is checked against ferric's parser.
  * grid: (75,110) unpruned, Becke partition with Becke (1988) radii
    adjustment; VV10 on (50,50) unpruned. Identical to
    scripts/gen_pyscf_dft_refs.py, whose recipe ferric already matches to
    2e-8 (PBE) / 1.6e-8 (B3LYP) / 3.1e-5 (wB97X-V) Ha on closed shells.
  * J/K, per ferric code path:
      - RKS PBE / B3LYP (ferric `solve_rhf_ladder`, df_j_aux [+ df_k_aux]):
        RI-J (+ RI-K) with the jkfit aux == PySCF `density_fit(aux)`.
      - UKS PBE (ferric `solve_uhf`, df_j_aux set): RI-J == `density_fit`.
      - UKS B3LYP (df_j_aux + df_k_aux): RI-JK == `density_fit`.
      - UKS wB97X-V: ferric's open-shell solver builds J EXACTLY for a
        range-separated functional (uhf.rs `j_aux_eff` is None when omega > 0,
        whatever df_j_aux says) and K ONLY by density fitting, as
        c_SR K^DF[erfc] + c_LR K^DF[erf], each fitted in its own attenuated
        metric. PySCF's `density_fit` would fit J too, and its UKS assembles
        K as hyb K_full + (alpha - hyb) K_LR. So here J is exact, and every K
        request is served by DF in the attenuated metric, with a full-range K
        request answered as K_SR^DF + K_LR^DF — algebraically ferric's
        hyb K_SR + alpha K_LR. See `_UksExactJDfK`.
  * convergence: conv_tol 1e-10, conv_tol_grad 1e-7.

KNOWN LIMITS of the reference state checks (recorded in provenance):
  * PySCF's KS response omits the VV10 kernel, so wB97X-V stability verdicts
    and lambda_min are for the Hessian without VV10's second derivative.
  * ferric's own UKS stability analysis SKIPS range-separated functionals
    (StabilitySkip::RangeSeparated); for wB97X-V the Rust test therefore
    relies on the energy + <S^2> match to the stability-checked reference.

Run (light-to-medium; the dense Hessians and 3-guess stability loops over
4 x 2 x 3 open-shell cases dominate — use the slot):
    scripts/validation/run_slot.sh -- \\
        uv run --no-sync python scripts/validation/gen_ks_energies.py
Restrict to systems by name:  ... gen_ks_energies.py o2 hcl
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402

ROW = "ks_energies"
ROW_NAME = "KS-DFT"
AUX = "def2-universal-jkfit"
MAIN_GRID = (75, 110)
NLC_GRID = (50, 50)
CONV_TOL = 1e-10
CONV_TOL_GRAD = 1e-7
GUESSES = ("minao", "atom", "huckel")
MAX_STAB_ROUNDS = 10

# ferric xc name (as the Rust test passes it) -> PySCF xc string
PYSCF_XC = {
    "pbe": "PBE,PBE",
    "b3lyp": "B3LYP",
    "wb97x-v": "wB97X_V",
}

OPEN_SHELL = {  # system -> (charge, multiplicity)
    "nh2": (0, 2),
    "ch3": (0, 2),
    "ho2": (0, 2),
    "o2": (0, 3),
}
OPEN_SHELL_BASES = ("6-31g", "def2-svp")
OPEN_SHELL_XC = ("pbe", "b3lyp", "wb97x-v")

CLOSED_SHELL = {"h2s": (0, 1), "hcl": (0, 1), "sih4": (0, 1)}
CLOSED_SHELL_BASES = ("def2-svp", "def2-tzvp")
CLOSED_SHELL_XC = ("pbe", "b3lyp")

# Human-readable J/K recipe per (shell, xc), recorded in provenance.
JK_RECIPE = {
    ("rks", "pbe"): "RI-J (density_fit, aux fed from ferric JSON); K unused",
    ("rks", "b3lyp"): "RI-J + RI-K (density_fit)",
    ("uks", "pbe"): "RI-J (density_fit); K unused",
    ("uks", "b3lyp"): "RI-J + RI-K (density_fit)",
    ("uks", "wb97x-v"): (
        "EXACT J (4-index) + density-fitted K in the attenuated metric; full-range "
        "K served as K_SR^DF[erfc] + K_LR^DF[erf] (== ferric's c_SR K_SR + c_LR K_LR)"
    ),
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


def _configure_grids(mf, xc: str):
    from pyscf import dft

    mf.grids.atom_grid = MAIN_GRID
    mf.grids.prune = None
    # ferric uses Becke (1988) size adjustment, not PySCF's default Treutler.
    mf.grids.radii_adjust = dft.radi.becke_atomic_radii_adjust
    if xc == "wb97x-v":
        mf.nlc = "VV10"
        mf.nlcgrids.atom_grid = NLC_GRID
        mf.nlcgrids.prune = None
    mf.conv_tol = CONV_TOL
    mf.conv_tol_grad = CONV_TOL_GRAD
    mf.max_cycle = 500
    mf.verbose = 0
    return mf


def _uks_exact_j_dfk_class():
    """UKS whose J is exact and whose K is density-fitted in the attenuated
    metric — ferric's open-shell range-separated construction (see module doc).
    """
    from pyscf import df, dft, scf

    class _UksExactJDfK(dft.uks.UKS):
        _keys = {"ferric_dfobj"}

        def get_jk(
            self, mol=None, dm=None, hermi=1, with_j=True, with_k=True, omega=None
        ):
            if mol is None:
                mol = self.mol
            if dm is None:
                dm = self.make_rdm1()
            vj = vk = None
            if with_j:
                if omega not in (None, 0, 0.0):
                    raise NotImplementedError("attenuated J is never requested by UKS")
                vj = scf.hf.get_jk(mol, dm, hermi, with_j=True, with_k=False)[0]
            if with_k:
                dfo = self.ferric_dfobj
                if omega in (None, 0, 0.0):
                    w, _, _ = self._numint.rsh_and_hybrid_coeff(self.xc, spin=mol.spin)
                    if w == 0:
                        raise ValueError(
                            "_UksExactJDfK is for range-separated functionals"
                        )
                    vk = dfo.get_jk(dm, hermi, with_j=False, with_k=True, omega=-w)[1]
                    vk = (
                        vk
                        + dfo.get_jk(dm, hermi, with_j=False, with_k=True, omega=w)[1]
                    )
                else:
                    vk = dfo.get_jk(dm, hermi, with_j=False, with_k=True, omega=omega)[
                        1
                    ]
            return vj, vk

    def make(mol, aux, xc):
        mf = _UksExactJDfK(mol, xc=PYSCF_XC[xc])
        mf.ferric_dfobj = df.DF(mol, auxbasis=aux)
        return mf

    return make


def _uks_factory(aux, xc):
    from pyscf import dft

    if xc == "wb97x-v":
        make = _uks_exact_j_dfk_class()

        def factory(mol):
            return _configure_grids(make(mol, aux, xc), xc)
    else:

        def factory(mol):
            mf = dft.UKS(mol, xc=PYSCF_XC[xc]).density_fit(auxbasis=aux)
            return _configure_grids(mf, xc)

    return factory


def _rks(mol, aux, xc) -> dict:
    from pyscf import dft

    mf = dft.RKS(mol, xc=PYSCF_XC[xc]).density_fit(auxbasis=aux)
    _configure_grids(mf, xc)
    e = mf.kernel()
    if not mf.converged:
        raise RuntimeError(f"RKS {xc} did not converge")
    nocc = mol.nelectron // 2
    return {
        "energy": float(e),
        "converged": True,
        "homo": float(mf.mo_energy[nocc - 1]),
        "lumo": float(mf.mo_energy[nocc]),
        "jk": JK_RECIPE[("rks", xc)],
    }


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
            },
            "nlc_vv10": {"atom_grid": list(NLC_GRID), "prune": None},
        },
        aux={
            "name": AUX,
            "json": str(common.basis_json_path(AUX).relative_to(common.ROOT)),
            "sha256": common.sha256_file(common.basis_json_path(AUX)),
        },
        frozen_core=None,
        scf_conv={"conv_tol": CONV_TOL, "conv_tol_grad": CONV_TOL_GRAD},
        stability=stability,
        generator="scripts/validation/gen_ks_energies.py",
        extra=extra,
    )


def gen_open_shell(system, charge, mult) -> list[Path]:
    xyz = common.MOL_DIR / f"{system}.xyz"
    symbols, coords = common.read_xyz(xyz)
    written = []
    for basis_name in OPEN_SHELL_BASES:
        mol = common.build_pyscf_mol(xyz, basis_name, charge=charge, multiplicity=mult)
        basis_check = common.check_basis_like_for_like(mol, basis_name, symbols)
        aux = _aux_basis(symbols)
        basis_check.update(_check_aux(mol, aux, symbols))
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
        for xc in OPEN_SHELL_XC:
            res = common.run_open_shell(
                mol,
                "uks",
                conv_tol=CONV_TOL,
                conv_tol_grad=CONV_TOL_GRAD,
                max_stab_rounds=MAX_STAB_ROUNDS,
                guesses=GUESSES,
                mf_factory=_uks_factory(aux, xc),
            )
            res["xc"] = PYSCF_XC[xc]
            res["jk"] = JK_RECIPE[("uks", xc)]
            if xc == "wb97x-v":
                res["stability"]["caveat"] = (
                    "PySCF KS response omits the VV10 second derivative; verdict and "
                    "lambda_min are for the Hessian without it"
                )
            payload[f"uks_{xc}"] = res
            stab[f"uks_{xc}"] = res["stability"]
            print(
                f"{system:5s} {basis_name:9s} UKS {xc:8s} {res['energy']:.10f} "
                f"<S2>={res['s_squared']:.6f} lmin={res['stability']['lambda_min']:+.3e} "
                f"rounds={res['stability']['rounds']} multi={res['multiple_stable_minima']}"
            )
        uhf = common.run_open_shell(
            mol,
            "uhf",
            conv_tol=1e-11,
            conv_tol_grad=1e-8,
            max_stab_rounds=MAX_STAB_ROUNDS,
            guesses=GUESSES,
        )
        payload["uhf_control"] = {
            "energy": uhf["energy"],
            "s_squared": uhf["s_squared"],
            "stability": uhf["stability"],
            "note": "exact-integral UHF; ferric UKS must MISS this (functional applied)",
        }
        payload["provenance"] = _prov(
            xyz,
            basis_name,
            symbols,
            coords,
            {
                "method": "dft.UKS (+ scf.UHF control)",
                "xc": {k: PYSCF_XC[k] for k in OPEN_SHELL_XC},
                "jk": {xc: JK_RECIPE[("uks", xc)] for xc in OPEN_SHELL_XC},
                "init_guess_scan": list(GUESSES),
                "stability": "internal=True external=False, restart from the unstable "
                f"direction, max {MAX_STAB_ROUNDS} rounds; lambda_min from a dense "
                "gen_g_hop_uhf Hessian (includes f_xc via gen_response)",
            },
            stab,
            {"basis_self_check": basis_check},
        )
        written.append(common.write_reference(ROW, system, basis_name, payload))
    return written


def gen_closed_shell(system, charge, mult) -> list[Path]:
    xyz = common.MOL_DIR / f"{system}.xyz"
    symbols, coords = common.read_xyz(xyz)
    written = []
    for basis_name in CLOSED_SHELL_BASES:
        mol = common.build_pyscf_mol(xyz, basis_name, charge=charge, multiplicity=mult)
        basis_check = common.check_basis_like_for_like(mol, basis_name, symbols)
        aux = _aux_basis(symbols)
        basis_check.update(_check_aux(mol, aux, symbols))
        payload = {
            "row": ROW_NAME,
            "system": system,
            "basis": basis_name,
            "charge": charge,
            "multiplicity": mult,
            "nao": mol.nao_nr(),
            "nuclear_repulsion": float(mol.energy_nuc()),
        }
        for xc in CLOSED_SHELL_XC:
            res = _rks(mol, aux, xc)
            res["xc"] = PYSCF_XC[xc]
            payload[f"rks_{xc}"] = res
            print(f"{system:5s} {basis_name:9s} RKS {xc:8s} {res['energy']:.10f}")
        payload["rhf_control"] = {
            "energy": _rhf_control(mol),
            "note": "exact-integral RHF; ferric RKS must MISS this (functional applied)",
        }
        payload["provenance"] = _prov(
            xyz,
            basis_name,
            symbols,
            coords,
            {
                "method": "dft.RKS (+ scf.RHF control)",
                "xc": {k: PYSCF_XC[k] for k in CLOSED_SHELL_XC},
                "jk": {xc: JK_RECIPE[("rks", xc)] for xc in CLOSED_SHELL_XC},
                "init_guess": "minao (PySCF default)",
            },
            None,
            {"basis_self_check": basis_check},
        )
        written.append(common.write_reference(ROW, system, basis_name, payload))
    return written


def main() -> int:
    only = set(sys.argv[1:])
    written = []
    for system, (charge, mult) in OPEN_SHELL.items():
        if not only or system in only:
            written += gen_open_shell(system, charge, mult)
    for system, (charge, mult) in CLOSED_SHELL.items():
        if not only or system in only:
            written += gen_closed_shell(system, charge, mult)
    print(f"GEN_KS_ENERGIES_DONE written={len(written)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
