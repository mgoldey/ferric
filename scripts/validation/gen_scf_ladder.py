"""PySCF references for the VALIDATION.md "SCF ladder" row (validation tier W4).

Consumer: crates/ferric-scf/tests/validation_scf_ladder.rs.
Output:   testdata/reference/validation/scf_ladder/<system>_def2-svp.json

What the row asks
-----------------
ferric's default SCF convergence ladder (`ferric_scf::ladder::default_ladder`,
walked by `solve_rhf_ladder`: MINAO+DIIS -> +ADIIS -> +level shift -> +SOSCF
-> +Fermi smearing) must land on the same, internally STABLE RHF minimum that
PySCF reaches with a second-order solver (`.newton()`) followed by a
`stability()` loop. This generator produces that minimum.

Protocol per system
-------------------
1. Several initial guesses (GUESSES). From each: `scf.RHF(mol).newton()`
   (second-order, Augmented-Hessian SOSCF), then iterate `stability(internal,
   external)` — following any INTERNAL instability by re-converging from the
   rotated orbitals — until internally stable (MAX_STAB_ROUNDS).
2. The reference state is the LOWEST internally stable RHF solution over all
   guesses. Every distinct stable solution found is recorded
   (`stable_minima_found`), so a ferric result on a different (higher) stable
   RHF minimum is identifiable rather than a mystery mHa miss.
3. Internal (RHF->RHF, singlet) Hessian at the reference: lowest eigenvalues
   in FERRIC's convention (PySCF `newton_ah.gen_g_hop_rhf` / 2 — see
   gen_scf_stability.py for the proof of that factor). ALWAYS the full dense
   spectrum of the Hessian built explicitly from MO integrals
   (`mo_integral_hessians`), checked against PySCF's matvec on random vectors.
   Never a Davidson: one seeded on the lowest diagonal entries stayed inside
   one Oh block on Cr(CO)6 and reported +0.18006 for a true minimum of
   +0.07716 (see `mo_integral_hessians`).
4. External (RHF->UHF, triplet) Hessian: PySCF `stability._gen_hop_rhf_external`
   `hop_rhf2uhf` (factor 1 == ferric's triplet channel). INFORMATION ONLY: the
   row is about the RHF minimum. If negative, the instability is followed to
   a broken-symmetry UHF minimum and its energy/<S^2> is recorded.
5. As INFORMATION for the non-vacuity control: plain PySCF RHF (DIIS, MINAO,
   no newton, no stability following), its convergence flag and energy.

Basis like-for-like
-------------------
* N2: ferric's bundled `def2-svp.json`.
* CuCN, Cr(CO)6: ferric's bundled `def2-svp.json` has NO Z=21-34 (it carries
  1-20, 35, 37, 53, 54). The row therefore uses
  testdata/reference/validation/scf_ladder/basis/def2-svp_bse_c-n-o-cr-cu.json,
  a subset (C, N, O, Cr, Cu) of the Basis Set Exchange def2-SVP JSON (version
  1, Turbomole 7.3 data). Its C/N/O shells are byte-identical to the bundled
  file (checked when the subset was cut). The Rust test loads it with
  `ferric_core::basis::load_bse_json`; PySCF gets it through common.py exactly
  as a bundled basis.
* Cu and Cr in def2-SVP are ALL-ELECTRON: the def2 ECPs start at Rb (Z=37).
  The subset file has no `ecp_potentials`/`ecp_electrons` for any element,
  and the generator asserts that (`_assert_no_ecp`).
* d/f shells are spherical on both sides (BSE `gto_spherical`).

Geometries (sources in the xyz comment lines):
    n2_r1.60   N2 at r = 1.60 A (stretched; RHF->UHF unstable)
    cucn       linear Cu-C-N, rm(2) microwave structure (Grotjahn, Brewster &
               Ziurys, JACS 124, 5895 (2002)): r(Cu-C) 1.82962, r(C-N) 1.162 A
    cr_co6     Oh, r(Cr-C) 1.916, r(C-O) 1.171 A (X-ray, Whitaker & Jeffery,
               Acta Cryst. 23, 977 (1967))

All exact 4-index integrals (no density fitting), no frozen core, no grid.

Run:
    # N2 and CuCN: light (seconds to a minute)
    scripts/validation/run_slot.sh --light -- \\
        uv run --no-sync python scripts/validation/gen_scf_ladder.py n2_r1.60 cucn
    # Cr(CO)6: ~200 AOs, takes the slot
    scripts/validation/run_slot.sh -- \\
        uv run --no-sync python scripts/validation/gen_scf_ladder.py cr_co6
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402

ROW = "scf_ladder"
ROW_NAME = "SCF ladder"
BASIS_LABEL = "def2-svp"
TM_BASIS = (
    "testdata/reference/validation/scf_ladder/basis/def2-svp_bse_c-n-o-cr-cu.json"
)

# system -> (basis spec handed to common.py, charge, multiplicity)
SYSTEMS = {
    "n2_r1.60": ("def2-svp", 0, 1),
    "cucn": (TM_BASIS, 0, 1),
    "cr_co6": (TM_BASIS, 0, 1),
}

CONV_TOL = 1e-11
CONV_TOL_GRAD = 1e-7
MAX_CYCLE = 300
GUESSES = ("minao", "atom", "huckel", "1e")
MAX_STAB_ROUNDS = 10
N_SPECTRUM = 8
# MO-integral Hessian vs PySCF's matvec (random vectors), Hartree.
MATVEC_CHECK_TOL = 1e-8
# Two stable solutions closer than this are the same state.
SAME_STATE_TOL = 1e-7
# newton() vs DIIS-polished energy must agree to this (see _polish).
POLISH_TOL = 1e-8
# |lambda| at or below this is an exact zero mode, not curvature (same scale
# as ferric's StabilityConfig::noise_floor, 1e-6).
ZERO_MODE_TOL = 1e-6


def _assert_no_ecp(basis_spec: str, symbols: list[str]) -> None:
    data = json.loads(common.basis_json_path(basis_spec).read_text())
    for s in set(symbols):
        el = data["elements"][str(common.z_of(s))]
        assert "ecp_potentials" not in el and "ecp_electrons" not in el, (
            f"{s} carries an ECP in {basis_spec}; this row is all-electron"
        )


# ---------------------------------------------------------------------------
# Hessians (ferric convention)
# ---------------------------------------------------------------------------


def mo_integral_hessians(mf):
    """Dense singlet and triplet RHF orbital Hessians from MO integrals.

    Index order (a, i) row-major — PySCF's (nvir, nocc) rotation layout, which
    is also ferric's `(nvirt, nocc)` kappa block. In FERRIC's convention
    (derived from rhf_newton::hessian_matvec on dD = 2C(k + k^T)C^T):

        singlet  H[ai,bj] = f_ab d_ij - d_ab f_ji + 4(ai|bj) - (ab|ij) - (aj|bi)
        triplet  H[ai,bj] = f_ab d_ij - d_ab f_ji            - (ab|ij) - (aj|bi)

    (f = MO Fock matrix; f_ab d_ij - d_ab f_ji = d_ij d_ab (e_a - e_i) when the
    orbitals are exactly canonical)

    Built EXPLICITLY, not from a matvec on a few guess vectors: a Davidson seeded
    on the lowest diagonal entries can converge entirely inside one symmetry
    block. That happened here — the first Cr(CO)6 reference (lib.davidson1,
    8 roots seeded on the 8 smallest diagonal entries, all "converged") reported
    +0.18006 as the singlet minimum, while ferric's block Davidson found +0.07716
    on the same state. `_check_against_matvecs` ties this construction to
    PySCF's own operators on random vectors."""
    import numpy as np
    from pyscf import ao2mo

    mol = mf.mol
    occ = mf.mo_occ > 0
    co, cv = mf.mo_coeff[:, occ], mf.mo_coeff[:, ~occ]
    no, nv = co.shape[1], cv.shape[1]
    vovo = ao2mo.general(mol, (cv, co, cv, co), compact=False).reshape(nv, no, nv, no)
    vvoo = ao2mo.general(mol, (cv, cv, co, co), compact=False).reshape(nv, nv, no, no)
    exch = vvoo.transpose(0, 2, 1, 3) + vovo.transpose(0, 3, 2, 1)  # (ab|ij) + (aj|bi)
    del vvoo
    n = nv * no
    # The one-electron term uses the FULL MO Fock blocks (f_ab d_ij - d_ab f_ji),
    # as both PySCF's h_op and ferric's hessian_matvec do, not just eigenvalue
    # differences: at a converged-but-not-exact point the off-diagonal Fock
    # elements are O(gradient), and the eigenvalue-only form missed PySCF's
    # matvec by 3e-8 on CuCN.
    fock = mf.mo_coeff.T @ mf.get_fock() @ mf.mo_coeff
    foo, fvv = fock[np.ix_(occ, occ)], fock[np.ix_(~occ, ~occ)]
    h4 = -exch
    h4 += np.einsum("ab,ij->aibj", fvv, np.eye(no))
    h4 -= np.einsum("ab,ji->aibj", np.eye(nv), foo)
    h_t = h4.reshape(n, n)
    h_s = 4.0 * vovo.reshape(n, n) + h_t
    return h_s, h_t


def _check_against_matvecs(h, hop, n_vec=3, seed=7):
    """Max |H x - hop(x)| over random vectors (not unit vectors: a random
    vector touches every symmetry block)."""
    import numpy as np

    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(n_vec):
        x = rng.standard_normal(h.shape[0])
        worst = max(
            worst, float(np.max(np.abs(h @ x - np.asarray(hop(x)).real.ravel())))
        )
    return worst


def _spectrum(h):
    import numpy as np

    asym = float(np.max(np.abs(h - h.T)))
    ev = np.linalg.eigvalsh(0.5 * (h + h.T))
    return ev, asym


def rhf_hessians(mf):
    """(singlet lowest, singlet meta, triplet lowest, triplet meta, max |g|)."""
    from pyscf.scf import stability as pstab
    from pyscf.soscf import newton_ah

    h_s, h_t = mo_integral_hessians(mf)
    g, h_op, _hd = newton_ah.gen_g_hop_rhf(mf, mf.mo_coeff, mf.mo_occ)
    _h1, _d1, hop_trip, _hdt = pstab._gen_hop_rhf_external(mf)
    # PySCF gen_g_hop_rhf = 2 x ferric; hop_rhf2uhf = 1 x ferric.
    dev_s = _check_against_matvecs(h_s, lambda x: h_op(x) / 2.0)
    dev_t = _check_against_matvecs(h_t, hop_trip)
    if max(dev_s, dev_t) > MATVEC_CHECK_TOL:
        raise RuntimeError(
            f"MO-integral Hessian disagrees with PySCF's matvec: singlet {dev_s:.2e}, "
            f"triplet {dev_t:.2e} — the convention in mo_integral_hessians is wrong"
        )
    ev_s, asym_s = _spectrum(h_s)
    del h_s
    ev_t, asym_t = _spectrum(h_t)
    base = {
        "method": "dense eigvalsh of the explicit MO-integral Hessian",
        "dim": int(len(ev_s)),
    }
    meta_s = {
        **base,
        "construction": "f_ab d_ij - d_ab f_ji + 4(ai|bj) - (ab|ij) - (aj|bi) (ferric rhf_internal_stability convention)",
        "matvec_check": "max |H x - gen_g_hop_rhf(x)/2| over 3 random vectors",
        "matvec_check_max_dev": dev_s,
        "max_asymmetry": asym_s,
    }
    meta_t = {
        **base,
        "construction": "f_ab d_ij - d_ab f_ji - (ab|ij) - (aj|bi) (== stability._gen_hop_rhf_external hop_rhf2uhf, factor 1)",
        "matvec_check": "max |H x - hop_rhf2uhf(x)| over 3 random vectors",
        "matvec_check_max_dev": dev_t,
        "max_asymmetry": asym_t,
    }
    return (
        [float(x) for x in ev_s[:N_SPECTRUM]],
        meta_s,
        [float(x) for x in ev_t[:N_SPECTRUM]],
        meta_t,
        float(abs(g).max()),
    )


# ---------------------------------------------------------------------------
# SCF
# ---------------------------------------------------------------------------


def _rhf(mol, newton: bool):
    from pyscf import scf

    mf = scf.RHF(mol)
    mf.conv_tol = CONV_TOL
    mf.conv_tol_grad = CONV_TOL_GRAD
    mf.max_cycle = MAX_CYCLE
    mf.verbose = 0
    return mf.newton() if newton else mf


def _polish(mol, mf):
    """Finish a newton() run with plain DIIS from its density.

    MEASURED (N2 r=1.60, stable state): PySCF's AH solver stalls at
    |g| = 1.9e-7 for 50 macro-iterations with delta_E = 0, so it never sets
    `converged` at conv_tol_grad 1e-7 although the energy is converged to
    1e-14. DIIS started from that density converges in a few cycles to the same
    energy. The polished object is returned only if it converged and its energy
    agrees with the newton one to POLISH_TOL (else the newton object is kept and
    its `converged` flag decides)."""
    m = _rhf(mol, newton=False)
    m.kernel(dm0=mf.make_rdm1())
    if m.converged and abs(m.e_tot - mf.e_tot) < POLISH_TOL:
        return m, True
    return mf, False


def stable_rhf_from(mol, guess: str) -> dict:
    """newton() from `guess`, then follow internal instabilities to a minimum."""
    mf = _rhf(mol, newton=True)
    dm0 = mf.get_init_guess(key=guess)
    t0 = time.time()
    mf.kernel(dm0=dm0)
    mf, polished = _polish(mol, mf)
    rounds = []
    stable_i = False
    for k in range(MAX_STAB_ROUNDS):
        mo_i, _mo_e, stable_i, stable_e = mf.stability(
            internal=True, external=True, return_status=True
        )
        rounds.append(
            {
                "round": k,
                "energy": float(mf.e_tot),
                "converged": bool(mf.converged),
                "internal_stable": bool(stable_i),
                "external_stable": bool(stable_e),
            }
        )
        if stable_i:
            break
        nmf = _rhf(mol, newton=True)
        nmf.kernel(mo_coeff=mo_i, mo_occ=mf.mo_occ)
        mf, polished = _polish(mol, nmf)
    return {
        "guess": guess,
        "polished_with_diis": polished,
        "mf": mf,
        "energy": float(mf.e_tot),
        "converged": bool(mf.converged),
        "internal_stable": bool(stable_i),
        "rounds": rounds,
        "wall_s": time.time() - t0,
    }


def plain_diis(mol) -> dict:
    """PySCF's own first-order SCF, no newton, no stability following."""
    mf = _rhf(mol, newton=False)
    mf.max_cycle = 100
    t0 = time.time()
    mf.kernel()
    return {
        "construction": "scf.RHF, MINAO guess, Pulay DIIS (PySCF default), max_cycle 100, no newton, no stability following",
        "converged": bool(mf.converged),
        "energy": float(mf.e_tot),
        "wall_s": time.time() - t0,
        "mf": mf,
    }


def follow_external(mol, mf) -> dict:
    """Follow the RHF->UHF instability (information only)."""
    from pyscf import scf

    umf = scf.addons.convert_to_uhf(mf)
    _mo_i, mo_e, _si, _se = mf.stability(
        internal=False, external=True, return_status=True
    )
    u = scf.UHF(mol)
    u.conv_tol = CONV_TOL
    u.conv_tol_grad = CONV_TOL_GRAD
    u.max_cycle = MAX_CYCLE
    u.verbose = 0

    def _uhf():
        x = scf.UHF(mol)
        x.conv_tol = CONV_TOL
        x.conv_tol_grad = CONV_TOL_GRAD
        x.max_cycle = MAX_CYCLE
        x.verbose = 0
        return x

    def _upolish(n):
        # Same AH-stall as _polish (see there): finish with plain UHF DIIS.
        x = _uhf()
        x.kernel(dm0=n.make_rdm1())
        return x if x.converged and abs(x.e_tot - n.e_tot) < POLISH_TOL else n

    n = u.newton()
    n.kernel(dm0=n.make_rdm1(mo_e, umf.mo_occ))
    u = _upolish(n)
    st_i = False
    for _ in range(MAX_STAB_ROUNDS):
        mo_i, _m, st_i, _s = u.stability(
            internal=True, external=False, return_status=True
        )
        if st_i:
            break
        n = _uhf().newton()
        n.kernel(mo_coeff=mo_i, mo_occ=u.mo_occ)
        u = _upolish(n)
    s2, _ = u.spin_square()
    return {
        "energy": float(u.e_tot),
        "converged": bool(u.converged),
        "internal_stable": bool(st_i),
        "s_squared": float(s2),
        "below_rhf_by": float(mf.e_tot - u.e_tot),
        "construction": "UHF.newton() from the RHF->UHF eigenvector, then internal stability loop",
    }


def gen(system: str) -> Path:
    import numpy as np
    import pyscf

    basis_spec, charge, mult = SYSTEMS[system]
    xyz = common.MOL_DIR / f"{system}.xyz"
    symbols, coords = common.read_xyz(xyz)
    _assert_no_ecp(basis_spec, symbols)
    mol = common.build_pyscf_mol(xyz, basis_spec, charge=charge, multiplicity=mult)
    mol.max_memory = 8000
    check = common.check_basis_like_for_like(mol, basis_spec, symbols)
    print(f"{system}: nao={mol.nao_nr()} nelec={mol.nelectron}", flush=True)

    runs = []
    for g in GUESSES:
        try:
            r = stable_rhf_from(mol, g)
        except Exception as e:  # a guess that fails is recorded, not fatal
            print(f"  guess {g}: FAILED {e!r}", flush=True)
            runs.append({"guess": g, "error": repr(e)})
            continue
        print(
            f"  guess {g:6s}: E={r['energy']:.10f} conv={r['converged']} "
            f"int_stable={r['internal_stable']} rounds={len(r['rounds'])} {r['wall_s']:.0f}s",
            flush=True,
        )
        runs.append(r)

    ok = [r for r in runs if "mf" in r and r["converged"] and r["internal_stable"]]
    if not ok:
        raise RuntimeError(f"{system}: no converged, internally stable RHF solution")
    best = min(ok, key=lambda r: r["energy"])
    mf = best["mf"]

    distinct: list[float] = []
    for r in sorted(ok, key=lambda r: r["energy"]):
        if not distinct or abs(r["energy"] - distinct[-1]) > SAME_STATE_TOL:
            distinct.append(r["energy"])

    sing, sing_meta, trip, trip_meta, gmax = rhf_hessians(mf)
    if sing[0] < -ZERO_MODE_TOL:
        raise RuntimeError(
            f"{system}: selected RHF state has singlet lambda_min {sing[0]:+.3e} < 0"
        )
    # Exact zero modes (Goldstone: a continuous symmetry the state breaks, so
    # rotating along it costs nothing). MEASURED on N2 r=1.60: the stable RHF
    # minimum has ONE singlet eigenvalue +9.0e-10, then a gap to +1.109e-1.
    # ferric's verdict there must be MARGINAL (|lambda| <= noise floor), not
    # STABLE; the first NONZERO eigenvalue carries the curvature information.
    zero_modes = sum(1 for x in sing if abs(x) <= ZERO_MODE_TOL)
    first_nonzero = next((x for x in sing if x > ZERO_MODE_TOL), None)
    print(f"  singlet {sing[:3]}  triplet {trip[:3]}", flush=True)

    nocc = mol.nelectron // 2
    eps = np.asarray(mf.mo_energy)
    block = {
        "energy": float(mf.e_tot),
        "converged": True,
        "selected_from_guess": best["guess"],
        "nocc": int(nocc),
        "homo": float(eps[nocc - 1]),
        "lumo": float(eps[nocc]),
        "aufbau": bool(eps[nocc - 1] < eps[nocc]),
        "occupied_energies_top8": [float(x) for x in eps[max(0, nocc - 8) : nocc]],
        "max_orbital_gradient": gmax,
        "singlet_rhf_internal": {"lowest": sing, **sing_meta},
        "triplet_rhf_to_uhf": {"lowest": trip, **trip_meta},
        "stability": {
            "internal_stable": bool(sing[0] >= -ZERO_MODE_TOL),
            "internal_zero_modes": zero_modes,
            "internal_first_nonzero": first_nonzero,
            "external_stable": bool(trip[0] > 0.0),
            "kind": "RHF internal (dense singlet) + RHF->UHF external (triplet), real, MO-integral Hessians",
        },
    }
    payload = {
        "row": ROW_NAME,
        "system": system,
        "basis": BASIS_LABEL,
        "basis_file": str(common.basis_json_path(basis_spec).relative_to(common.ROOT)),
        "charge": charge,
        "multiplicity": mult,
        "nao": mol.nao_nr(),
        "nelectron": mol.nelectron,
        "nuclear_repulsion": float(mol.energy_nuc()),
        "rhf": block,
        "stable_minima_found": distinct,
        "guess_scan": [{k: v for k, v in r.items() if k != "mf"} for r in runs],
        "plain_diis_pyscf": plain_diis(mol),
    }
    pd = payload["plain_diis_pyscf"]
    pd_mf = pd.pop("mf")
    pd["same_state_as_reference"] = bool(
        abs(pd["energy"] - block["energy"]) <= SAME_STATE_TOL
    )
    if pd["converged"] and not pd["same_state_as_reference"]:
        # A DIFFERENT converged state from plain DIIS: record its singlet
        # spectrum so ferric's verdict on the same state can be compared
        # (the row's non-vacuity control).
        pd_sing, pd_meta, _pt, _ptm, pd_g = rhf_hessians(pd_mf)
        pd["above_reference_by"] = float(pd["energy"] - block["energy"])
        pd["singlet_rhf_internal"] = {"lowest": pd_sing, **pd_meta}
        pd["max_orbital_gradient"] = pd_g
        pd["internal_stable"] = bool(pd_sing[0] >= -ZERO_MODE_TOL)
    print(f"  plain DIIS: {payload['plain_diis_pyscf']}", flush=True)
    if not block["stability"]["external_stable"]:
        payload["broken_symmetry_uhf"] = follow_external(mol, mf)
        print(f"  BS-UHF: {payload['broken_symmetry_uhf']}", flush=True)

    payload["provenance"] = common.provenance(
        code="PySCF",
        version=pyscf.__version__,
        keywords={
            "method": "scf.RHF(mol).newton() + stability(internal, external) loop",
            "guesses": list(GUESSES),
            "numpy": np.__version__,
            "eri": "exact 4-index (no density fitting)",
            "selection": "lowest converged, internally stable RHF over all guesses",
            "normalization": "singlet = gen_g_hop_rhf / 2 (ferric convention); triplet = hop_rhf2uhf x1",
        },
        basis_name=basis_spec,
        xyz_path=xyz,
        coords_bohr=coords,
        symbols=symbols,
        grid=None,
        aux=None,
        frozen_core=None,
        scf_conv={"conv_tol": CONV_TOL, "conv_tol_grad": CONV_TOL_GRAD},
        stability={"rhf": block["stability"]},
        generator="scripts/validation/gen_scf_ladder.py",
        extra={"basis_self_check": check, "all_electron": True},
    )
    path = common.write_reference(ROW, system, BASIS_LABEL, payload)
    print(f"  wrote {path}", flush=True)
    return path


def main() -> int:
    only = sys.argv[1:] or list(SYSTEMS)
    for s in only:
        if s not in SYSTEMS:
            raise SystemExit(f"unknown system {s!r}; choose from {list(SYSTEMS)}")
    written = [gen(s) for s in only]
    print(f"GEN_SCF_LADDER_DONE written={len(written)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
