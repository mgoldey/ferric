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

Every Hessian here is built DENSE from the PySCF matvec (applied to every unit
vector, symmetrized, max asymmetry recorded) and the lowest N_SPECTRUM
eigenvalues are stored, not just the lowest one.

Run (light; seconds):
    scripts/validation/run_slot.sh --light -- \\
        env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \\
        python scripts/validation/gen_scf_stability.py
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
    print(f"GEN_SCF_STABILITY_DONE written={len(written)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
