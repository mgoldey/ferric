"""PySCF references for the VALIDATION.md "SCF stability" row (validation tier W1).

Consumer: crates/ferric-scf/tests/validation_scf_stability.rs.
Output:   testdata/reference/validation/scf_stability/<system>_6-31g.json

What ferric computes (crates/ferric-scf/src/stability.rs) and what it is
matched against here:

    ferric                               PySCF construction (dense, from the matvec)
    -----------------------------------  ------------------------------------------------
    uhf_internal_stability (UHF, HF)     newton_ah.gen_g_hop_uhf h_op            factor 1
    uhf_internal_stability (UKS, fxc)    newton_ah.gen_g_hop_uhf h_op on a UKS   factor 1
                                         (gen_response carries f_xc)
    rhf_internal_stability (RHF singlet) newton_ah.gen_g_hop_rhf h_op            factor 1/2
    RHF->UHF external (triplet block of  stability._gen_hop_rhf_external         factor 1
      the UHF Hessian at the RHF point)    hop_rhf2uhf

NORMALIZATION (read from both codes, then PROVED numerically below):
  * ferric rhf_newton::hessian_matvec returns (e_a - e_i) k + [dJ - 1/2 dK]_ai on
    dD = 2 C(k + k^T)C^T, i.e. the singlet (A+B). PySCF gen_g_hop_rhf builds the
    SAME quantity and returns it `* 2` ("x2.ravel() * 2"), so its dense
    eigenvalues are 2x ferric's. (PySCF's own rhf_internal Davidson multiplies
    by 2 AGAIN, "hop(x).real * 2", so its logged eigenvalues are 4x ferric's.)
  * ferric uhf_newton::hessian_matvec and PySCF gen_g_hop_uhf both return
    (e_a - e_i) k_s + [J(dD_a + dD_b) - K(dD_s)]_ai with dD_s = C_s(k_s + k_s^T)C_s^T
    and no overall factor: equal.
  * PySCF hop_rhf2uhf returns (e_a - e_i) k + [-1/2 K(2 C(k + k^T)C^T)]_ai, the
    triplet (A+B), no overall factor. The UHF Hessian at an RHF point, restricted
    to k_b = -k_a (normalized (k, -k)/sqrt2), is exactly that operator: J cancels
    and each spin keeps -K(dD_s). Restricted to k_b = +k_a it is the singlet
    (A+B), i.e. ferric's RHF Hessian. So
        spec(UHF Hessian at the RHF point) = spec(gen_g_hop_rhf)/2  U  spec(hop_rhf2uhf)
    and this generator ASSERTS that identity to 1e-9 on every closed-shell
    system before writing, which proves the factors above inside PySCF itself.

Systems (geometries in testdata/molecules/validation/, sources in the xyz
comment lines), all 6-31G from ferric's bundled JSON:

    n2_plus          N2+  UHF  doublet, stability-followed (common.run_open_shell)
    oh               OH   UHF  doublet (2Pi). The pi hole breaks the axial symmetry,
                     so rotating it about the axis is an exact zero mode of the
                     Hessian (a Goldstone mode): lambda_min = 0 to convergence.
                     The first NONZERO eigenvalue is what carries information.
    water_eq         H2O  RHF, r(OH) = 0.9572 A: internally and externally STABLE
    water_stretched  H2O  RHF, r(OH) = 2.0 A: RHF-internal stable, RHF->UHF
                     UNSTABLE (the negative control of the row)
    nh2              NH2  UKS/PBE, EXACT J (no density fitting), (75,110) unpruned
                     Becke grid with Becke-1988 radii, stability-followed; plus an
                     exact UHF control whose lambda_min the UKS one must MISS.

RANGE-SEPARATED block (issue #292), written to a SEPARATE def2-SVP file
`n2_plus_def2-svp.json` because the row's other systems are 6-31G:

    n2_plus (RSH)    N2+ UKS/wB97X-V, def2-SVP, omega OVERRIDDEN to 0.40 and 0.60
                     via mf.omega. Exact J, DF-K in the attenuated metric
                     (K_SR^DF[erfc] + K_LR^DF[erf]) and the (75,110)/(50,50)
                     grids -- the like-for-like recipe of gen_rsh_omega.py,
                     because that is how ferric builds RSH exchange (there is no
                     exact four-centre RSH K in ferric; see driver.rs).

                     The cation is converged from the minao guess and NOT
                     stability-followed, exactly as gen_rsh_omega.py's
                     `symmetric_cation_probe` does, because the point of the
                     comparison is the SYMMETRIC D-inf-h branch: past the onset
                     (bracketed at 0.53-0.56 by gen_rsh_omega.py) that branch is
                     a SADDLE, and following it to stability would destroy the
                     negative control. The block therefore carries no
                     `stability` key -- `common.write_reference` refuses a block
                     whose recorded verdict is unstable, and omitting the key is
                     how a deliberately-unstable reference state is recorded
                     rather than smuggled past that guard.

                     omega = 0.40 must be STABLE (lowest > 0) and omega = 0.60
                     UNSTABLE (lowest < 0); the generator asserts both before
                     writing, so a reference that cannot discriminate is never
                     produced.

                     CAVEAT recorded in the block: PySCF's KS response omits the
                     VV10 second derivative, and so does ferric's f_xc kernel.
                     The comparison is like-for-like ON THAT OMISSION and is not
                     the complete second derivative of wB97X-V's energy.

Every Hessian here is built DENSE from the PySCF matvec (applied to every unit
vector, symmetrized, max asymmetry recorded) and the lowest N_SPECTRUM
eigenvalues are stored, not just the lowest one.

Run (light; seconds):
    scripts/validation/run_slot.sh --light -- \\
        env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \\
        uv run --no-sync python scripts/validation/gen_scf_stability.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402

ROW = "scf_stability"
ROW_NAME = "SCF stability"
BASIS = "6-31g"
N_SPECTRUM = 8
HF_CONV_TOL = 1e-11
HF_CONV_TOL_GRAD = 1e-10
KS_CONV_TOL = 1e-11
KS_CONV_TOL_GRAD = 1e-10
GUESSES = ("minao", "atom", "huckel")
MAX_STAB_ROUNDS = 10
MAIN_GRID = (75, 110)
UNION_TOL = 1e-9

# --- range-separated block (#292) ---
RSH_BASIS = "def2-svp"
RSH_AUX = "def2-universal-jkfit"
RSH_FUNCTIONAL = "wB97X-V"
RSH_SYSTEM = ("n2_plus", 1, 2)
# omega = 0.40 below the 0.53-0.56 onset (STABLE), 0.60 above it (UNSTABLE).
RSH_OMEGAS = (0.40, 0.60)
RSH_NLC_GRID = (50, 50)
RSH_CONV_TOL = 1e-10
RSH_CONV_TOL_GRAD = 1e-7

OPEN_SHELL_UHF = {"n2_plus": (1, 2), "oh": (0, 2)}
CLOSED_SHELL = ("water_eq", "water_stretched")
UKS_SYSTEM = ("nh2", 0, 2)


# ---------------------------------------------------------------------------
# Dense Hessians from PySCF matvecs
# ---------------------------------------------------------------------------


def dense_from_hop(hop, n: int):
    """Dense symmetric matrix from a matvec; returns (sorted eigenvalues, asym)."""
    import numpy as np

    hess = np.empty((n, n))
    e = np.zeros(n)
    for i in range(n):
        e[:] = 0.0
        e[i] = 1.0
        hess[:, i] = np.asarray(hop(e)).real.ravel()
    asym = float(np.max(np.abs(hess - hess.T)))
    evals = np.linalg.eigvalsh(0.5 * (hess + hess.T))
    return evals, asym


def uhf_spectrum(mf):
    from pyscf.soscf import newton_ah

    out = newton_ah.gen_g_hop_uhf(mf, mf.mo_coeff, mf.mo_occ)
    h_op, h_diag = out[-2], out[-1]
    evals, asym = dense_from_hop(h_op, len(h_diag))
    g = out[0]
    return evals, asym, float(abs(g).max())


# `common.run_open_shell` returns a dict, not the SCF object. The dense UHF
# spectrum needs the selected state's MOs, so capture the mf that
# run_open_shell hands to `uhf_lambda_min` for the SELECTED state (it is called
# exactly once, after selection).
_CAPTURED: dict = {}
_orig_lambda_min = common.uhf_lambda_min


def _capturing_lambda_min(mf):
    _CAPTURED["mf"] = mf
    return _orig_lambda_min(mf)


common.uhf_lambda_min = _capturing_lambda_min


def _open_shell_block(
    mol, method, mf_factory=None, conv_tol=HF_CONV_TOL, conv_tol_grad=HF_CONV_TOL_GRAD
):
    _CAPTURED.clear()
    res = common.run_open_shell(
        mol,
        method,
        conv_tol=conv_tol,
        conv_tol_grad=conv_tol_grad,
        max_stab_rounds=MAX_STAB_ROUNDS,
        guesses=GUESSES,
        mf_factory=mf_factory,
    )
    mf = _CAPTURED["mf"]
    evals, asym, gmax = uhf_spectrum(mf)
    assert abs(evals[0] - res["stability"]["lambda_min"]) < 1e-12
    res["hessian"] = {
        "construction": "dense newton_ah.gen_g_hop_uhf h_op (== ferric uhf_internal_stability, factor 1)",
        "dim": int(len(evals)),
        "lowest": [float(x) for x in evals[:N_SPECTRUM]],
        "max_asymmetry": asym,
        "max_orbital_gradient": gmax,
    }
    return res


# ---------------------------------------------------------------------------
# Closed shell: RHF internal (singlet) + RHF->UHF external (triplet)
# ---------------------------------------------------------------------------


def closed_shell_block(mol) -> dict:
    import numpy as np
    from pyscf import scf
    from pyscf.scf import stability as pstab
    from pyscf.soscf import newton_ah

    mf = scf.RHF(mol)
    mf.conv_tol = HF_CONV_TOL
    mf.conv_tol_grad = HF_CONV_TOL_GRAD
    mf.max_cycle = 500
    mf.verbose = 0
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("RHF did not converge")

    # Singlet (RHF internal): PySCF raw = 2 x ferric.
    g, h_op, h_diag = newton_ah.gen_g_hop_rhf(mf, mf.mo_coeff, mf.mo_occ)
    sing_raw, asym_s = dense_from_hop(h_op, len(h_diag))
    sing = sing_raw / 2.0
    if sing[0] < 0:
        raise RuntimeError(
            f"RHF state is internally UNSTABLE (singlet lambda_min {sing[0]:+.3e}); "
            "this row needs the lowest RHF solution"
        )

    # Triplet (RHF -> UHF external): factor 1.
    _hop1, _hd1, hop_trip, hdiag_trip = pstab._gen_hop_rhf_external(mf)
    trip, asym_t = dense_from_hop(hop_trip, len(hdiag_trip))

    # UHF Hessian at the RHF point: factor 1; spectrum must be the union.
    umf = scf.addons.convert_to_uhf(mf)
    uev, asym_u, _ = uhf_spectrum(umf)
    union = np.sort(np.concatenate([sing, trip]))
    union_dev = float(np.max(np.abs(union - uev)))
    assert union_dev < UNION_TOL, (
        f"spec(UHF at RHF point) != spec(rhf)/2 U spec(triplet): max dev {union_dev:.2e} "
        "-- the normalization stated in the module doc is wrong"
    )

    # PySCF's own verdicts (Davidson, threshold -1e-5 on its own scale).
    _mo_i, _mo_e, stable_i, stable_e = mf.stability(
        internal=True, external=True, return_status=True
    )

    nocc = mol.nelectron // 2
    block = {
        "energy": float(mf.e_tot),
        "converged": True,
        "nocc": int(nocc),
        "homo": float(mf.mo_energy[nocc - 1]),
        "lumo": float(mf.mo_energy[nocc]),
        "max_orbital_gradient": float(abs(g).max()),
        "singlet_rhf_internal": {
            "construction": "dense newton_ah.gen_g_hop_rhf h_op / 2 (ferric rhf_internal_stability convention)",
            "pyscf_raw_factor": 2.0,
            "lowest": [float(x) for x in sing[:N_SPECTRUM]],
            "lowest_pyscf_raw": [float(x) for x in sing_raw[:N_SPECTRUM]],
            "max_asymmetry": asym_s,
        },
        "triplet_rhf_to_uhf": {
            "construction": "dense stability._gen_hop_rhf_external hop_rhf2uhf (factor 1)",
            "lowest": [float(x) for x in trip[:N_SPECTRUM]],
            "max_asymmetry": asym_t,
        },
        "uhf_hessian_at_rhf_point": {
            "construction": "dense gen_g_hop_uhf on scf.addons.convert_to_uhf(rhf) (factor 1)",
            "dim": int(len(uev)),
            "lowest": [float(x) for x in uev[:N_SPECTRUM]],
            "union_identity_max_dev": union_dev,
            "max_asymmetry": asym_u,
        },
        "pyscf_verdict": {
            "internal_stable": bool(stable_i),
            "external_stable": bool(stable_e),
            "threshold": "PySCF: unstable iff Davidson eigenvalue < -1e-5 on its own scale",
        },
        "stability": {
            # write_reference refuses internal_stable == False; the EXTERNAL
            # verdict is the one this row expects to be False on water_stretched.
            "internal_stable": bool(stable_i),
            "external_stable": bool(stable_e),
            "kind": "PySCF RHF internal + RHF->UHF external (real)",
        },
    }
    if not stable_e:
        # Follow the external instability to the broken-symmetry UHF minimum:
        # informational, and it proves the negative eigenvector is downhill.
        u = scf.UHF(mol)
        u.conv_tol = HF_CONV_TOL
        u.conv_tol_grad = HF_CONV_TOL_GRAD
        u.max_cycle = 500
        u.verbose = 0
        mo_e = _mo_e
        dm = u.make_rdm1(mo_e, umf.mo_occ)
        u.kernel(dm0=dm)
        if not u.converged:
            u = u.newton()
            u.kernel(u.mo_coeff, u.mo_occ)
        for _ in range(MAX_STAB_ROUNDS):
            mo_i, _m, st_i, _s = u.stability(
                internal=True, external=False, return_status=True
            )
            if st_i:
                break
            u.kernel(dm0=u.make_rdm1(mo_i, u.mo_occ))
        s2, _ = u.spin_square()
        block["broken_symmetry_uhf"] = {
            "energy": float(u.e_tot),
            "converged": bool(u.converged),
            "s_squared": float(s2),
            "below_rhf_by": float(mf.e_tot - u.e_tot),
        }
    return block


# ---------------------------------------------------------------------------
# UKS / PBE, exact J
# ---------------------------------------------------------------------------


def _uks_pbe_factory(mol):
    from pyscf import dft

    mf = dft.UKS(mol, xc="PBE,PBE")  # no density_fit: EXACT J, like ferric's default
    mf.grids.atom_grid = MAIN_GRID
    mf.grids.prune = None
    mf.grids.radii_adjust = dft.radi.becke_atomic_radii_adjust
    mf.conv_tol = KS_CONV_TOL
    mf.conv_tol_grad = KS_CONV_TOL_GRAD
    mf.max_cycle = 500
    mf.verbose = 0
    return mf


# ---------------------------------------------------------------------------


def _payload(system, mol, charge, mult):
    return {
        "row": ROW_NAME,
        "system": system,
        "basis": BASIS,
        "charge": charge,
        "multiplicity": mult,
        "nao": mol.nao_nr(),
        "nuclear_repulsion": float(mol.energy_nuc()),
    }


def _prov(xyz, symbols, coords, keywords, stability, grid, basis_check):
    import numpy
    import pyscf

    keywords = dict(keywords)
    keywords["numpy"] = numpy.__version__
    keywords["eri"] = "exact 4-index (no density fitting)"
    keywords["normalization"] = (
        "stored 'lowest' arrays are in FERRIC's convention: gen_g_hop_uhf x1, "
        "gen_g_hop_rhf x1/2, hop_rhf2uhf x1 (see generator docstring)"
    )
    return common.provenance(
        code="PySCF",
        version=pyscf.__version__,
        keywords=keywords,
        basis_name=BASIS,
        xyz_path=xyz,
        coords_bohr=coords,
        symbols=symbols,
        grid=grid,
        aux=None,
        frozen_core=None,
        scf_conv={"conv_tol": HF_CONV_TOL, "conv_tol_grad": HF_CONV_TOL_GRAD},
        stability=stability,
        generator="scripts/validation/gen_scf_stability.py",
        extra={"basis_self_check": basis_check},
    )


def _load(system, charge, mult):
    xyz = common.MOL_DIR / f"{system}.xyz"
    symbols, coords = common.read_xyz(xyz)
    mol = common.build_pyscf_mol(xyz, BASIS, charge=charge, multiplicity=mult)
    check = common.check_basis_like_for_like(mol, BASIS, symbols)
    return xyz, symbols, coords, mol, check


def _default_guess_saddle(mol):
    """UHF from PySCF's default (minao) guess with NO stability following.

    Recorded only when that state is internally UNSTABLE (a saddle): it is a
    second, independent negative control on a real open-shell state. The key
    is `pyscf_verdict`, not `stability`, so write_reference's refusal of
    unstable blocks (meant for the REFERENCE state) does not fire on it.
    """
    from pyscf import scf

    mf = scf.UHF(mol)
    mf.conv_tol = HF_CONV_TOL
    mf.conv_tol_grad = HF_CONV_TOL_GRAD
    mf.max_cycle = 500
    mf.verbose = 0
    mf.kernel()
    if not mf.converged:
        return None
    evals, asym, gmax = uhf_spectrum(mf)
    if evals[0] > -1e-5:
        return None
    s2, _ = mf.spin_square()
    return {
        "energy": float(mf.e_tot),
        "init_guess": "minao",
        "s_squared": float(s2),
        "hessian": {
            "construction": "dense newton_ah.gen_g_hop_uhf h_op (factor 1)",
            "dim": int(len(evals)),
            "lowest": [float(x) for x in evals[:N_SPECTRUM]],
            "max_asymmetry": asym,
            "max_orbital_gradient": gmax,
        },
        "pyscf_verdict": {"internal_stable": False},
    }


def gen_uhf(system, charge, mult):
    xyz, symbols, coords, mol, check = _load(system, charge, mult)
    payload = _payload(system, mol, charge, mult)
    res = _open_shell_block(mol, "uhf")
    payload["uhf"] = res
    saddle = _default_guess_saddle(mol)
    if saddle is not None:
        payload["uhf_default_guess_saddle"] = saddle
        print(
            f"{system:16s} default-guess SADDLE {saddle['energy']:.10f} "
            f"lowest={saddle['hessian']['lowest'][:2]}"
        )
    payload["provenance"] = _prov(
        xyz,
        symbols,
        coords,
        {
            "method": "scf.UHF",
            "init_guess_scan": list(GUESSES),
            "stability": "internal loop then dense gen_g_hop_uhf",
        },
        {"uhf": res["stability"]},
        None,
        check,
    )
    print(
        f"{system:16s} UHF {res['energy']:.10f} lowest={res['hessian']['lowest'][:3]} "
        f"rounds={res['stability']['rounds']} multi={res['multiple_stable_minima']}"
    )
    return common.write_reference(ROW, system, BASIS, payload)


def gen_closed(system):
    xyz, symbols, coords, mol, check = _load(system, 0, 1)
    payload = _payload(system, mol, 0, 1)
    blk = closed_shell_block(mol)
    payload["rhf"] = blk
    payload["provenance"] = _prov(
        xyz,
        symbols,
        coords,
        {
            "method": "scf.RHF",
            "init_guess": "minao",
            "stability": "dense singlet/triplet/UHF-at-RHF Hessians + mf.stability(external=True)",
        },
        {"rhf": blk["stability"]},
        None,
        check,
    )
    print(
        f"{system:16s} RHF {blk['energy']:.10f} singlet={blk['singlet_rhf_internal']['lowest'][0]:+.10e} "
        f"triplet={blk['triplet_rhf_to_uhf']['lowest'][0]:+.10e} "
        f"verdict int={blk['pyscf_verdict']['internal_stable']} ext={blk['pyscf_verdict']['external_stable']} "
        f"union_dev={blk['uhf_hessian_at_rhf_point']['union_identity_max_dev']:.1e} "
        + (
            f"BS-UHF {blk['broken_symmetry_uhf']}"
            if "broken_symmetry_uhf" in blk
            else ""
        )
    )
    return common.write_reference(ROW, system, BASIS, payload)


def gen_uks():
    system, charge, mult = UKS_SYSTEM
    xyz, symbols, coords, mol, check = _load(system, charge, mult)
    payload = _payload(system, mol, charge, mult)
    uks = _open_shell_block(
        mol,
        "uks",
        mf_factory=_uks_pbe_factory,
        conv_tol=KS_CONV_TOL,
        conv_tol_grad=KS_CONV_TOL_GRAD,
    )
    uks["xc"] = "PBE,PBE"
    uks["jk"] = "EXACT J (4-index); no K (pure GGA)"
    uks["hessian"]["construction"] += " on dft.UKS (f_xc via gen_response)"
    uhf = _open_shell_block(mol, "uhf")
    payload["uks_pbe"] = uks
    payload["uhf_control"] = {
        "energy": uhf["energy"],
        "stability": uhf["stability"],
        "hessian": uhf["hessian"],
        "note": "exact UHF; ferric UKS lambda_min must MISS this (f_xc kernel applied)",
    }
    payload["provenance"] = _prov(
        xyz,
        symbols,
        coords,
        {
            "method": "dft.UKS PBE (+ scf.UHF control)",
            "init_guess_scan": list(GUESSES),
            "ks_conv": {"conv_tol": KS_CONV_TOL, "conv_tol_grad": KS_CONV_TOL_GRAD},
            "stability": "internal loop then dense gen_g_hop_uhf (includes f_xc)",
        },
        {"uks_pbe": uks["stability"], "uhf_control": uhf["stability"]},
        {
            "main": {
                "atom_grid": list(MAIN_GRID),
                "prune": None,
                "partition": "Becke (original_becke)",
                "radii_adjust": "becke_atomic_radii_adjust",
            }
        },
        check,
    )
    print(
        f"{system:16s} UKS/PBE {uks['energy']:.10f} lowest={uks['hessian']['lowest'][:3]} | "
        f"UHF ctl lmin={uhf['hessian']['lowest'][0]:+.6e}"
    )
    return common.write_reference(ROW, system, BASIS, payload)


# ---------------------------------------------------------------------------
# Range-separated: N2+ UKS/wB97X-V, omega overridden (#292)
# ---------------------------------------------------------------------------


def _rsh_mf(cmol, aux, omega):
    """UKS/wB97X-V at `omega`, with gen_rsh_omega.py's OWN exact-J/DF-K recipe.

    Imported rather than reimplemented: a second copy of the `_ExactJDfK`
    get_jk override is precisely the Fock/response divergence #292 is about, one
    level up. `gen_rsh_omega._make` already encodes exact J, DF-K in the
    attenuated metric served as K_SR^DF[erfc] + K_LR^DF[erf], the (75,110) main
    grid with Becke radii, the (50,50) VV10 grid, and the `mf.omega` override --
    the like-for-like recipe against ferric's uhf.rs + fock_assembly.rs.
    """
    from pyscf import dft

    import gen_rsh_omega

    return gen_rsh_omega._make(dft.uks.UKS, cmol, aux, omega, conv_tol=RSH_CONV_TOL)


def gen_rsh():
    """N2+ UKS/wB97X-V dense UKS-internal Hessian spectra at omega 0.40 / 0.60.

    The cation is converged from the DEFAULT (minao) guess and NOT
    stability-followed: the comparison is about the SYMMETRIC D-inf-h branch,
    which past the onset is a saddle. Following it to stability would converge a
    symmetry-broken state and destroy the negative control.
    """
    import gen_rsh_omega

    system, charge, mult = RSH_SYSTEM
    xyz = common.MOL_DIR / f"{system}.xyz"
    symbols, coords = common.read_xyz(xyz)
    cmol = common.build_pyscf_mol(xyz, RSH_BASIS, charge=charge, multiplicity=mult)
    basis_check = common.check_basis_like_for_like(cmol, RSH_BASIS, symbols)
    aux, cart = common.pyscf_basis(RSH_AUX, symbols)
    if cart:
        raise ValueError(f"{RSH_AUX}: Cartesian l>=2 aux shells; not like-for-like")
    from pyscf.df import addons

    aux_nao = addons.make_auxmol(cmol, aux).nao_nr()
    assert aux_nao == common.ferric_nao(RSH_AUX, symbols), "aux nao mismatch"
    basis_check["aux_nao"] = aux_nao

    # ANCHOR, before any measurement: mf.omega at wB97X-V's PUBLISHED value must
    # be a no-op, and another omega must move the energy. Without this an
    # override that never reached the functional would produce two identical
    # "spectra" and look like a clean result.
    probe = _rsh_mf(cmol, aux, None)
    w_pub, _, _ = probe._numint.rsh_and_hybrid_coeff(probe.xc, spin=cmol.spin)
    e_default = float(probe.kernel())
    e_pub = float(_rsh_mf(cmol, aux, w_pub).kernel())
    e_other = float(_rsh_mf(cmol, aux, w_pub + 0.1).kernel())
    d_pub, d_other = abs(e_pub - e_default), abs(e_other - e_default)
    if d_pub > 1e-9 or d_other < 1e-5:
        raise RuntimeError(
            f"omega override anchor failed: |E(w_pub)-E(default)|={d_pub:.2e} "
            f"(want <=1e-9), |E(w_pub+0.1)-E(default)|={d_other:.2e} (want >=1e-5)"
        )

    points = []
    for w in RSH_OMEGAS:
        mf = _rsh_mf(cmol, aux, w)
        e = float(mf.kernel())
        if not mf.converged:
            raise RuntimeError(f"N2+ UKS/wB97X-V did not converge at omega={w}")
        evals, asym, gmax = uhf_spectrum(mf)
        lam = common.uhf_lambda_min(mf)
        assert abs(evals[0] - lam) < 1e-12, (
            f"omega={w}: dense spectrum low {evals[0]:.12e} disagrees with "
            f"uhf_lambda_min {lam:.12e}"
        )
        points.append(
            {
                "omega": float(w),
                "energy": e,
                "s_squared": float(mf.spin_square()[0]),
                "selected_guess": "minao (default; NOT stability-followed)",
                "hessian": {
                    "construction": (
                        "dense newton_ah.gen_g_hop_uhf h_op on dft.UKS "
                        "(== ferric uhf_internal_stability, factor 1); "
                        "f_xc via gen_response"
                    ),
                    "dim": int(len(evals)),
                    "lowest": [float(x) for x in evals[:N_SPECTRUM]],
                    "max_asymmetry": asym,
                    "max_orbital_gradient": gmax,
                },
            }
        )
        print(
            f"{system:16s} RSH w={w:.2f} E={e:.10f} <S2>={points[-1]['s_squared']:.6f} "
            f"lowest={evals[:3]} gmax={gmax:.2e}",
            flush=True,
        )

    # The REACHABILITY assertion: a reference that does not straddle zero cannot
    # discriminate, so refuse to write one rather than let the Rust test assert
    # arithmetic. 0.40 is below the 0.53-0.56 onset, 0.60 above it.
    lo, hi = points[0], points[1]
    if not (lo["hessian"]["lowest"][0] > 0.0 > hi["hessian"]["lowest"][0]):
        raise RuntimeError(
            f"the RSH reference does not straddle zero: lambda_min "
            f"{lo['hessian']['lowest'][0]:+.6e} at omega={lo['omega']} and "
            f"{hi['hessian']['lowest'][0]:+.6e} at omega={hi['omega']}. "
            "omega=0.40 should be STABLE and 0.60 UNSTABLE (onset 0.53-0.56, "
            "gen_rsh_omega.py). Without the straddle the negative control is "
            "vacuous, so no reference is written."
        )

    payload = {
        "row": ROW_NAME,
        "system": system,
        "basis": RSH_BASIS,
        "charge": charge,
        "multiplicity": mult,
        "nao": cmol.nao_nr(),
        "nuclear_repulsion": float(cmol.energy_nuc()),
        "functional": RSH_FUNCTIONAL,
        "omega_override_anchor": {
            "published_omega": float(w_pub),
            "abs_e_override_at_published_minus_default": d_pub,
            "abs_e_override_plus_0p1_minus_default": d_other,
        },
        "rsh_uks_omega_scan": points,
        "caveat": (
            "PySCF's KS response omits the VV10 second derivative, and so does "
            "ferric's f_xc kernel; this comparison is like-for-like ON THAT "
            "OMISSION and is not the complete second derivative of wB97X-V's "
            "energy. The cation is NOT stability-followed: these are the "
            "symmetric D-inf-h branch's spectra, which is the point (past the "
            "onset that branch is a saddle)."
        ),
    }
    payload["provenance"] = common.provenance(
        code="PySCF",
        version=__import__("pyscf").__version__,
        keywords={
            "method": "dft.UKS wB97X-V with mf.omega overridden",
            "xc": gen_rsh_omega.XC_PYSCF,
            "jk": gen_rsh_omega.JK_RECIPE,
            "init_guess": "minao (default), NOT stability-followed",
            "stability": (
                "dense gen_g_hop_uhf (includes f_xc); PySCF KS response omits "
                "the VV10 kernel"
            ),
            "normalization": (
                "stored 'lowest' arrays are in FERRIC's convention: gen_g_hop_uhf x1"
            ),
            "numpy": __import__("numpy").__version__,
        },
        basis_name=RSH_BASIS,
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
            "nlc_vv10": {"atom_grid": list(RSH_NLC_GRID), "prune": None},
        },
        aux={"name": RSH_AUX},
        frozen_core=None,
        scf_conv={"conv_tol": RSH_CONV_TOL, "conv_tol_grad": RSH_CONV_TOL_GRAD},
        stability={
            "note": (
                "no stability-following; verdicts are read from the dense spectra above"
            )
        },
        generator="scripts/validation/gen_scf_stability.py",
        extra={"basis_self_check": basis_check},
    )
    return common.write_reference(ROW, system, RSH_BASIS, payload)


def main() -> int:
    only = set(sys.argv[1:])
    written = []
    for system, (charge, mult) in OPEN_SHELL_UHF.items():
        if not only or system in only:
            written.append(gen_uhf(system, charge, mult))
    for system in CLOSED_SHELL:
        if not only or system in only:
            written.append(gen_closed(system))
    if not only or UKS_SYSTEM[0] in only:
        written.append(gen_uks())
    # Selected by name "rsh" (not by system name: n2_plus already names the
    # 6-31G UHF block, and the two write different files).
    if not only or "rsh" in only:
        written.append(gen_rsh())
    print(f"GEN_SCF_STABILITY_DONE written={len(written)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
