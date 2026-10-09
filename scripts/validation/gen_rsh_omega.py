"""PySCF references for the VALIDATION.md "RSH ω tuning" row (plan row 102).

Consumer: crates/ferric-scf/tests/validation_rsh_omega.rs.
Output:   testdata/reference/validation/rsh_omega/<system>_def2-svp.json

WHAT FERRIC TUNES (read from crates/ferric-scf/src/omega_tuning.rs, not a doc):

    J(ω) = ε_HOMO(N; ω) + IP(ω),     IP(ω) = E(N−1; ω) − E(N; ω)

J is SIGNED and NOT squared; `tune_omega` minimizes |J| by golden-section
search over [omega_lo, omega_hi] and returns the evaluation with the smallest
|J|. There is no cation (anion/EA) term: the neutral is closed-shell RKS
(`solve_rhf`), the cation is the doublet UKS (`solve_uhf`), both at the SAME
ω via `RhfConfig::xc_omega` (libxc `_omega` override; the CAM SR/LR split
follows the override, the CAM mixing coefficients do not). So ω* is the root
of J, found here with `scipy.optimize.brentq` on the signed J.

Systems (def2-SVP, functional ωB97X-V, ω in Bohr⁻¹): H2O and NH3 — J at
ω ∈ OMEGAS plus ω* to XTOL; N2 — J at 0.2, 0.3, 0.4, 0.45, 0.5 only, NO ω*.
The cation is converged from three guesses and followed to an internally
STABLE UKS state at every point (common.run_open_shell), and
`multiple_stable_minima` is recorded.

Why N2 has no ω*: the D∞h ²Σg+ N2+ cation becomes UKS-UNSTABLE (the hole
localizes on one N) between ω = 0.53 and 0.56, and J's root on the symmetric
branch lies at ~0.56, past the onset. There the stable state is
symmetry-broken and PySCF's second-order solver does not converge it, so no
stable reference exists at ω*. The onset is re-measured on every run
(`symmetric_cation_probe`: single-guess cation λ_min and J at 0.53, 0.56).
NH3 (non-degenerate 3a1 hole, no equivalent-atom localization) replaces N2
for the ω* comparison.

LIKE-FOR-LIKE RECIPE (read from rhf.rs `resolve_aux`, uhf.rs `j_aux_eff`,
fock_assembly.rs `build_rsh_dfk_pair`):

  * basis/geometry: ferric's bundled JSON and Bohr geometry via common.py.
  * grid: (75,110) unpruned, Becke partition, Becke (1988) radii adjustment;
    VV10 on (50,50) unpruned — ferric's defaults, as gen_ks_energies.py.
  * J: EXACT (4-index) for BOTH states. ferric's open-shell solver builds J
    exactly whenever ω > 0 (uhf.rs `j_aux_eff`), whatever df_j_aux says. Its
    closed-shell solver would auto-default to RI-J, so the Rust test sets
    `df_j_aux = Some("")` (explicit conventional J) to put the neutral on the
    same footing; `tune_omega` resolves an unset `df_j_aux` to the same exact
    J. `neutral_rij_shift` records, per ω, how far an RI-J neutral would move
    (the size of the error mixed treatments would put in the IP); it is not
    compared.
  * K: density-fitted in the attenuated metric only (ferric has no exact RSH
    K), c_SR·K^DF[erfc(ω)] + c_LR·K^DF[erf(ω)], jkfit aux fed from ferric's
    JSON. PySCF's full-range K request is served as K_SR^DF + K_LR^DF.
  * ω override: `mf.omega = ω`, which reaches both libxc's `_omega` and the
    SR/LR K split (`rsh_and_hybrid_coeff`). Anchored below: at the published
    ω the override is bit-identical to no override, and at another ω it moves
    the energy.
  * XC density floor: ferric zeroes XC where ρ ≤ 1e-10; not emulated here (it
    moves occupied eigenvalues ~1e-10 Ha, see gen_gw.py), far below the
    DF-K / VV10 floor of this functional.
  * convergence: conv_tol 1e-10, conv_tol_grad 1e-7 (both states).

Reference-side floors recorded in the JSON (`floors` block): the brentq
bracket/tolerance on ω*, and the J change under a 10x tighter conv_tol at the
first ω point.

Run (light, 13 min single-threaded: H2O 4, NH3 7.5, N2 1.7 min; each point is one RKS +
a 3-guess stability-followed UKS):
    OPENBLAS_NUM_THREADS=1 scripts/validation/run_slot.sh --light -- \\
        uv run --no-sync python scripts/validation/gen_rsh_omega.py
Restrict to systems by name:  ... gen_rsh_omega.py n2
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402

ROW = "rsh_omega"
ROW_NAME = "RSH omega tuning"
BASIS = "def2-svp"
AUX = "def2-universal-jkfit"
XC_FERRIC = "wB97X-V"
XC_PYSCF = "wB97X_V"
MAIN_GRID = (75, 110)
NLC_GRID = (50, 50)
CONV_TOL = 1e-10
CONV_TOL_GRAD = 1e-7
GUESSES = ("minao", "atom", "huckel")
MAX_STAB_ROUNDS = 10
OMEGAS = (0.2, 0.3, 0.4, 0.5, 0.6)  # Bohr^-1
XTOL = 1e-7  # brentq tolerance on omega*, Bohr^-1
# system -> (charge, multiplicity, omega grid, reference omega*?)
SYSTEMS = {
    "h2o": (0, 1, OMEGAS, True),
    "nh3": (0, 1, OMEGAS, True),
    # N2: the D-inf-h 2Sg+ cation becomes UKS-UNSTABLE (hole localization onto
    # one N) between omega 0.53 and 0.56, and J's root on that branch is ~0.56.
    # Above the onset the stable solution is symmetry-broken and PySCF's own
    # second-order solver does not converge it, so no stable reference exists
    # at omega*. N2 is therefore referenced on a grid below the onset only, and
    # the onset itself is measured live (`symmetric_cation_probe`).
    "n2": (0, 1, (0.2, 0.3, 0.4, 0.45, 0.5), False),
}
N2_PROBE_OMEGAS = (0.53, 0.56)
JK_RECIPE = (
    "EXACT J (4-index) for neutral and cation + density-fitted K in the attenuated "
    "metric; full-range K served as K_SR^DF[erfc] + K_LR^DF[erf] "
    "(== ferric's c_SR K_SR + c_LR K_LR)"
)


def _aux_basis(symbols):
    aux, cart = common.pyscf_basis(AUX, symbols)
    if cart:
        raise ValueError(f"{AUX}: Cartesian l>=2 aux shells; not matched like-for-like")
    return aux


def _exact_j_dfk(base):
    """`base` (RKS or UKS) with exact J and DF-K in the attenuated metric."""
    from pyscf import scf

    class _ExactJDfK(base):
        _keys = {"ferric_dfobj", "ferric_rij_dfobj"}

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
                    raise NotImplementedError("attenuated J is never requested")
                if self.ferric_rij_dfobj is not None:
                    vj = self.ferric_rij_dfobj.get_jk(
                        dm, hermi, with_j=True, with_k=False
                    )[0]
                else:
                    vj = scf.hf.get_jk(mol, dm, hermi, with_j=True, with_k=False)[0]
            if with_k:
                dfo = self.ferric_dfobj
                if omega in (None, 0, 0.0):
                    w, _, _ = self._numint.rsh_and_hybrid_coeff(self.xc, spin=mol.spin)
                    if w == 0:
                        raise ValueError(
                            "_ExactJDfK is for range-separated functionals"
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

    return _ExactJDfK


def _make(base, mol, aux, omega, conv_tol=CONV_TOL, rij=False):
    """Configured KS object: ferric's grid, VV10 grid, ω override."""
    from pyscf import df, dft

    mf = _exact_j_dfk(base)(mol, xc=XC_PYSCF)
    mf.ferric_dfobj = df.DF(mol, auxbasis=aux)
    mf.ferric_rij_dfobj = df.DF(mol, auxbasis=aux) if rij else None
    if omega is not None:
        mf.omega = float(omega)
    mf.grids.atom_grid = MAIN_GRID
    mf.grids.prune = None
    mf.grids.radii_adjust = dft.radi.becke_atomic_radii_adjust
    mf.nlc = "VV10"
    mf.nlcgrids.atom_grid = NLC_GRID
    mf.nlcgrids.prune = None
    mf.conv_tol = conv_tol
    mf.conv_tol_grad = CONV_TOL_GRAD
    mf.max_cycle = 500
    mf.verbose = 0
    return mf


def _neutral(mol, aux, omega, conv_tol=CONV_TOL, rij=False):
    from pyscf import dft

    mf = _make(dft.rks.RKS, mol, aux, omega, conv_tol=conv_tol, rij=rij)
    e = mf.kernel()
    if not mf.converged:
        raise RuntimeError(f"neutral RKS did not converge at omega={omega}")
    nocc = mol.nelectron // 2
    return float(e), float(mf.mo_energy[nocc - 1])


def _cation(cmol, aux, omega, conv_tol=CONV_TOL):
    from pyscf import dft

    def factory(m):
        return _make(dft.uks.UKS, m, aux, omega, conv_tol=conv_tol)

    return common.run_open_shell(
        cmol,
        "uks",
        conv_tol=conv_tol,
        conv_tol_grad=CONV_TOL_GRAD,
        max_stab_rounds=MAX_STAB_ROUNDS,
        guesses=GUESSES,
        mf_factory=factory,
    )


def _point(mol, cmol, aux, omega, conv_tol=CONV_TOL, with_rij=False) -> dict:
    e0, eps = _neutral(mol, aux, omega, conv_tol=conv_tol)
    cat = _cation(cmol, aux, omega, conv_tol=conv_tol)
    ip = cat["energy"] - e0
    out = {
        "omega": float(omega),
        "eps_homo": eps,
        "e_neutral": e0,
        "e_cation": cat["energy"],
        "ip": ip,
        "j": eps + ip,
        "cation_s_squared": cat["s_squared"],
        "cation_lambda_min": cat["stability"]["lambda_min"],
        "cation_selected_guess": cat["stability"]["selected_guess"],
        "cation_multiple_stable_minima": cat["multiple_stable_minima"],
        "cation_guess_energies": [g["energy"] for g in cat["guess_scan"]],
    }
    if with_rij:
        e0_rij, _ = _neutral(mol, aux, omega, conv_tol=conv_tol, rij=True)
        out["neutral_rij_shift"] = e0_rij - e0
    return out


def _override_anchor(mol, aux) -> dict:
    """mf.omega at the published value must equal no override; another ω must
    move the energy. Refuses to write a reference if either fails."""
    from pyscf import dft

    probe = _make(dft.rks.RKS, mol, aux, None)
    w_pub, _, _ = probe._numint.rsh_and_hybrid_coeff(probe.xc)
    e_default = float(probe.kernel())
    e_pub = _neutral(mol, aux, w_pub)[0]
    e_other = _neutral(mol, aux, w_pub + 0.1)[0]
    d_pub, d_other = abs(e_pub - e_default), abs(e_other - e_default)
    if d_pub > 1e-10 or d_other < 1e-5:
        raise RuntimeError(
            f"omega override anchor failed: |E(w_pub)-E(default)|={d_pub:.2e}, "
            f"|E(w_pub+0.1)-E(default)|={d_other:.2e}"
        )
    return {
        "published_omega": float(w_pub),
        "abs_e_override_at_published_minus_default": d_pub,
        "abs_e_override_plus_0p1_minus_default": d_other,
    }


def _symmetric_cation_probe(mol, cmol, aux, omegas) -> list[dict]:
    """Single-guess (minao) cation at each omega: its lowest orbital-Hessian
    eigenvalue and J on that branch. Records where the symmetric cation stops
    being a minimum; never written as a reference point."""
    from pyscf import dft

    out = []
    for w in omegas:
        mf = _make(dft.uks.UKS, cmol, aux, w)
        mf.kernel()
        if not mf.converged:
            raise RuntimeError(f"probe cation did not converge at omega={w}")
        e0, eps = _neutral(mol, aux, w)
        out.append(
            {
                "omega": float(w),
                "cation_lambda_min": common.uhf_lambda_min(mf),
                "cation_s_squared": float(mf.spin_square()[0]),
                "j_on_this_branch": eps + float(mf.e_tot) - e0,
            }
        )
        print(f"  probe {out[-1]}", flush=True)
    return out


def gen(system, charge, mult, omegas, want_omega_star) -> Path:
    t0 = time.time()
    xyz = common.MOL_DIR / f"{system}.xyz"
    symbols, coords = common.read_xyz(xyz)
    mol = common.build_pyscf_mol(xyz, BASIS, charge=charge, multiplicity=mult)
    cmol = common.build_pyscf_mol(xyz, BASIS, charge=charge + 1, multiplicity=2)
    basis_check = common.check_basis_like_for_like(mol, BASIS, symbols)
    aux = _aux_basis(symbols)
    from pyscf.df import addons

    aux_nao = addons.make_auxmol(mol, aux).nao_nr()
    assert aux_nao == common.ferric_nao(AUX, symbols), "aux nao mismatch"
    basis_check["aux_nao"] = aux_nao

    anchor = _override_anchor(mol, aux)
    print(f"{system}: omega override anchor {anchor}", flush=True)

    points = []
    for w in omegas:
        p = _point(mol, cmol, aux, w, with_rij=True)
        points.append(p)
        print(
            f"{system} w={w:.3f} eps={p['eps_homo']:+.10f} IP={p['ip']:+.10f} "
            f"J={p['j']:+.3e} <S2>+={p['cation_s_squared']:.5f} "
            f"multi={p['cation_multiple_stable_minima']} "
            f"rij_shift={p['neutral_rij_shift']:+.2e} ({time.time() - t0:.0f}s)",
            flush=True,
        )

    payload = {
        "row": ROW_NAME,
        "system": system,
        "basis": BASIS,
        "charge": charge,
        "multiplicity": mult,
        "cation_charge": charge + 1,
        "cation_multiplicity": 2,
        "nao": mol.nao_nr(),
        "nuclear_repulsion": float(mol.energy_nuc()),
        "functional": XC_FERRIC,
        "objective": (
            "J(w) = eps_HOMO(N;w) + E(N-1;w) - E(N;w), signed, Hartree; "
            "omega* = root of J (== argmin |J|); omega in Bohr^-1"
        ),
        "points": points,
        "omega_override_anchor": anchor,
    }
    # Reference-side floor: J at the first grid point under a 10x tighter conv_tol.
    tight = _point(mol, cmol, aux, omegas[0], conv_tol=CONV_TOL / 10)
    floors = {
        "j_change_conv_tol_x0p1_at_first_omega": tight["j"] - points[0]["j"],
        "eps_homo_change_conv_tol_x0p1_at_first_omega": tight["eps_homo"]
        - points[0]["eps_homo"],
    }
    if want_omega_star:
        payload.update(_omega_star(system, mol, cmol, aux, points, floors))
    else:
        payload["omega_star"] = None
        payload["omega_star_not_referenced"] = (
            "the symmetric cation turns UKS-unstable below J's root; see "
            "symmetric_cation_probe"
        )
        payload["symmetric_cation_probe"] = _symmetric_cation_probe(
            mol, cmol, aux, N2_PROBE_OMEGAS
        )
    payload["floors"] = floors
    return _write(system, payload, xyz, symbols, coords, basis_check, t0)


def _omega_star(system, mol, cmol, aux, points, floors) -> dict:
    import numpy
    from scipy.optimize import brentq

    # omega*: root of the signed J, bracketed by the grid's sign change.
    js = [p["j"] for p in points]
    brk = None
    for a, b in zip(points, points[1:]):
        if a["j"] * b["j"] < 0:
            brk = (a["omega"], b["omega"])
            break
    if brk is None:
        raise RuntimeError(f"{system}: J does not change sign on the grid: {js}")
    trace = []

    def j_of(w):
        p = _point(mol, cmol, aux, w)
        trace.append({"omega": p["omega"], "j": p["j"]})
        print(f"{system}   brentq w={w:.8f} J={p['j']:+.3e}", flush=True)
        return p["j"]

    w_star = brentq(j_of, brk[0], brk[1], xtol=XTOL, rtol=4 * numpy.finfo(float).eps)
    at_star = _point(mol, cmol, aux, w_star)
    # Local slope of J at omega* (for converting a J error into an omega* error).
    h = 1e-3
    j_plus = _point(mol, cmol, aux, w_star + h)["j"]
    j_minus = _point(mol, cmol, aux, w_star - h)["j"]
    slope = (j_plus - j_minus) / (2 * h)
    floors.update(
        {
            "brentq_xtol": XTOL,
            "brentq_bracket": list(brk),
            "j_at_omega_star": at_star["j"],
            "dj_domega_at_omega_star": slope,
            "omega_star_uncertainty_from_j_at_root": abs(at_star["j"] / slope),
        }
    )
    print(f"{system}: omega* = {w_star:.8f} floors {floors}", flush=True)
    return {"omega_star": w_star, "at_omega_star": at_star, "brentq_trace": trace}


def _write(system, payload, xyz, symbols, coords, basis_check, t0) -> Path:
    import numpy
    import pyscf

    payload["provenance"] = common.provenance(
        code="PySCF",
        version=pyscf.__version__,
        keywords={
            "method": "dft.RKS neutral + dft.UKS doublet cation, mf.omega override",
            "xc": XC_PYSCF,
            "jk": JK_RECIPE,
            "cation_init_guess_scan": list(GUESSES),
            "stability": "cation: internal=True, restart from the unstable direction, "
            f"max {MAX_STAB_ROUNDS} rounds; PySCF KS response omits the VV10 kernel",
            "neutral_rij_shift": "E_neutral(RI-J, same aux) - E_neutral(exact J): the "
            "IP error a mixed RI-J neutral / exact-J cation would carry; tune_omega "
            "resolves unset df_j_aux to exact J (recorded, not compared)",
            "root_finder": f"scipy.optimize.brentq on signed J, xtol {XTOL}",
            "numpy": numpy.__version__,
        },
        basis_name=BASIS,
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
        stability={"cation": "stability-followed per point (see points[*])"},
        generator="scripts/validation/gen_rsh_omega.py",
        extra={
            "basis_self_check": basis_check,
            "runtime_s": round(time.time() - t0, 1),
        },
    )
    path = common.write_reference(ROW, system, BASIS, payload)
    print(f"{system}: wrote {path} ({time.time() - t0:.0f}s)", flush=True)
    return path


def main() -> int:
    only = set(sys.argv[1:])
    written = [gen(s, *cfg) for s, cfg in SYSTEMS.items() if not only or s in only]
    print(f"GEN_RSH_OMEGA_DONE written={len(written)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
