"""PySCF + numpy references for the VALIDATION.md "ωB97X-L-V components" row.

Consumer: crates/ferric-cc/tests/validation_wb97xlv.rs
Output:   testdata/reference/validation/wb97xlv/<system>_def2-svp.json

THE FUNCTIONAL (Ransford & Carter-Fenk, PCCP 28, 14428 (2026), eqn 27)
---------------------------------------------------------------------
    E = E_KS[ωB97X-L + VV10(b=10, C=0.01)] + λ·E_c,LinLCCD(hh)^{sr,ω,λ}

No absolute ωB97X-L-V energy is published (testdata/reference/
wb97x_l_v_params.json, `_no_total_energy_available`), so this row checks the
two COMPONENTS independently and their sum:

1. E_KS. PySCF 2.13 KS with libxc HYB_GGA_XC_WB97X_V (id 466) customised via
   `dft.libxc.register_custom_functional_` with ferric's 18 ext params
   (`WB97X_L_V_EXT_PARAMS`, crates/ferric-dft/src/libxc.rs). The values are
   parsed from that Rust constant AND from the paper transcription
   (testdata/reference/wb97x_l_v_params.json) and the generator refuses to run
   unless the two agree exactly and libxc's own ext-param NAMES (read through
   `xc_func_info_get_ext_params_name`) are in ferric's order. VV10 b/C are not
   ext params of 466 (libxc carries wB97X-V's b = 6.0), so the custom
   functional's `nlc_coeff` is overridden to ((10.0, 0.01), 1); the override is
   verified to change E_KS (b = 6 vs b = 10).
     water: RKS.   OH ²Π: ROKS (the paper's open-shell SCF).

2. E_c. numpy LinLCCD(hh) (the exact eigen-solve of gen_linlccd.py, reused)
   with RI integrals of the composite operator λ·erfc(ωr)/r, λ = 0.6,
   ω = 0.1 Bohr⁻¹, fitted in the SAME operator's metric (as ferric's
   `linlccd`/`u_linlccd` do: `coulomb_metric_2c(op)` + `eri3_tensor(op)`), with
   ferric's def2-SVP-RIFIT aux JSON. Under that fit every 4-index integral is
   exactly λ × the erfc-fitted one, so the energy returned is λ·E_c^{sr,ω,λ} —
   the quantity ferric reports as `e_c_scaled` — with λ in the amplitude
   equations (eqn 22), i.e. "quadratic in λ".
     water: on the RKS orbitals.
     OH:    ROKS → ONE unrestricted KS Fock build (F_α, F_β from the ROKS
            density, same XC/J/K recipe) → semicanonicalise occ-occ and
            vir-vir blocks per spin → unrestricted (spin-orbital) LinLCCD(hh).
   All electrons correlated (ferric's DoubleHybridConfig default frozen_core 0).

LIKE-FOR-LIKE RECIPE (read from ferric's code)
----------------------------------------------
* Basis: ferric's def2-SVP JSON via common.py. The reference orbitals are
  written in FERRIC's AO order (shell-by-shell match on l + exponents against
  common.ferric_shells), together with PySCF's AO overlap in that order, so the
  Rust test can (a) prove the mapping by comparing overlaps and (b) inject the
  reference orbitals into ferric's correlation solver.
* XC grid: (75,110) unpruned, Becke partition, Becke-1988 radii adjust; VV10 on
  (50,50) unpruned, ALSO Becke radii adjust (ferric builds both grids with the
  same `build_molecular_grid`; gen_ks_energies.py left PySCF's default
  Treutler adjust on the NLC grid).
* Exchange: ferric builds range-separated exchange ONLY by density fitting
  (driver.rs `build_rsh_dfk_pair`: K[erfc(ω)] and K[erf(ω)], each fitted in
  its own attenuated metric, def2-universal-jkfit; there is no conventional
  erf/erfc K path). Reproduced here exactly: every K request is served by
  PySCF DF in the attenuated metric, a full-range request answered as
  K_SR^DF + K_LR^DF, so PySCF's hyb·K + (α−hyb)·K_LR assembles to ferric's
  c_SR·K_SR + c_LR·K_LR (c_SR = 0.6, c_LR = 1.0).
* Coulomb: "ks" blocks use EXACT four-centre J (ferric: ROKS ignores df_j_aux
  for an RSH functional; RKS gets df_j_aux = Some("") in the test).
  The water-only "ks_rij" block uses RI-J with def2-universal-jkfit — the
  production path of ferric's `run_wb97x_l_v`, which sets df_j_aux to it.
* The OH semicanonical Fock follows ferric's `semicanonicalize(.., Some(XcSpec))`:
  exact J[D_α+D_β], DF-K as above per spin, V_xc^σ + VV10 on the same grids.

ANCHORS asserted HERE before anything is written
------------------------------------------------
* Ext params: Rust constant == paper JSON == libxc names/order; libxc's
  rsh_coeff of the custom functional == (ω, α, β) = (0.1, 1.0, −0.4).
* Operator plumbing: 3-centre (mn|P) under erfc(ω) + erf(ω) == Coulomb to
  1e-12 (proves PySCF's omega < 0 really is erfc and reaches aux_e2).
* λ = 0 ⇒ nothing; DriversOnly(λ) numpy == λ² · PySCF DF-(U)MP2 run on the
  same orbitals with `with_df.range_coulomb(-ω)` (an independent code path),
  to 1e-12.
* Closed shell: spin-adapted == spin-orbital LinLCCD(hh) to 1e-12.
* Dense amplitude residuals ≤ 1e-11 (inherited from gen_linlccd.spin_orbital).
* hh is resolvable: |E_hh − E_drivers| > 1e-5 Eh; and the PR-#239 "linear λ"
  construction λ·E_hh(λ=1) is recorded as a negative-control value, together
  with |λ·E_hh(λ=1) − E_hh(λ)|.

Run (light; water ~1 min, OH ~3 min on a quiet box -- its six-start ROKS dominates):
    scripts/validation/run_slot.sh --light -- \\
        uv run --no-sync python scripts/validation/gen_wb97xlv.py [h2o|oh]
"""

from __future__ import annotations

import ctypes
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402
import gen_linlccd as lin  # noqa: E402

ROW = "wb97xlv"
ROW_NAME = "ωB97X-L-V components"
BASIS = "def2-svp"
JK_AUX = "def2-universal-jkfit"
CORR_AUX = "def2-svp-rifit"
SYSTEMS = {"h2o": (0, 1), "oh": (0, 2)}

LAMBDA = 0.6
OMEGA = 0.1
VV10_B = 10.0
VV10_C = 0.01
LIBXC_ID = 466  # HYB_GGA_XC_WB97X_V
BASE_XC = "HYB_GGA_XC_WB97X_V"
# Lower case: PySCF's dispersion.parse_dft lower-cases mf.xc before the
# custom-functional lookup (an exact-key dict), so an upper-case name is lost.
CUSTOM_XC = "wb97x_l_v_ferric"

MAIN_GRID = (75, 110)
NLC_GRID = (50, 50)
CONV_TOL = 1e-11
CONV_TOL_GRAD = 1e-8
# OH ROKS multi-start (gen_roks_energies.py recipe).
GUESSES = ("minao", "atom", "huckel")
LEVEL_SHIFTS = (0.0, 0.5)
# MEASURED 2026-10-01: PySCF's Newton polish of OH ROKS/wB97X-L-V stalls at
# max|g| 3.2e-6 .. 8.4e-5 over the six starts (the response omits VV10, and the
# pi-hole orientation is a near-zero mode on the 4-fold Lebedev grid), with a
# 5.3e-7 Ha spread between starts; the selected (lowest) start has 3.2e-6. Accepting at 1e-4 and recording gmax + spread; E_c is
# compared on INJECTED orbitals, so this only enters the E_KS / own-orbital bars.
ACCEPT_GMAX = 1e-4

TOL_ANCHOR = 1e-12

LIBXC_RS = common.ROOT / "crates" / "ferric-dft" / "src" / "libxc.rs"
PAPER_JSON = common.ROOT / "testdata" / "reference" / "wb97x_l_v_params.json"


# ---------------------------------------------------------------------------
# The functional
# ---------------------------------------------------------------------------


def ferric_ext_params() -> list[tuple[str, float]]:
    """WB97X_L_V_EXT_PARAMS parsed from the Rust source, cross-checked vs the paper."""
    src = LIBXC_RS.read_text()
    m = re.search(
        r"pub const WB97X_L_V_EXT_PARAMS: \[\(&str, f64\); 18\] = \[(.*?)\];", src, re.S
    )
    if m is None:
        raise RuntimeError(f"WB97X_L_V_EXT_PARAMS not found in {LIBXC_RS}")
    pairs = [
        (n, float(v))
        for n, v in re.findall(r'\("(_\w+)",\s*([-+0-9.eE]+)\)', m.group(1))
    ]
    if len(pairs) != 18:
        raise RuntimeError(f"parsed {len(pairs)} ext params, expected 18")

    paper = json.loads(PAPER_JSON.read_text())
    fixed = paper["fixed_parameters"]
    fit = paper["least_squares_optimized_parameters_final"]
    mix = paper["exchange_mixing"]
    want = {"_cx0": fixed["cx_0"], "_css0": fixed["css_0"], "_cos0": fixed["cab_0"]}
    for i in range(1, 5):
        want[f"_cx{i}"] = fit[f"cx_{i}"]
        want[f"_css{i}"] = fit[f"css_{i}"]
        want[f"_cos{i}"] = fit[f"cab_{i}"]
    want.update(
        {"_alpha": mix["alpha"], "_beta": mix["beta"], "_omega": fixed["omega"]}
    )
    for n, v in pairs:
        if want[n] != v:
            raise RuntimeError(f"ext param {n}: ferric {v} != paper {want[n]}")
    if (fixed["lambda"], fixed["b"], fixed["C"], fixed["omega"]) != (
        LAMBDA,
        VV10_B,
        VV10_C,
        OMEGA,
    ):
        raise RuntimeError("generator lambda/b/C/omega disagree with the paper JSON")
    return pairs


def register_functional() -> dict:
    """Register ferric's ωB97X-L-V as a PySCF custom libxc functional."""
    from pyscf.dft import libxc

    params = ferric_ext_params()
    stock = libxc._get_xc(BASE_XC, 0)
    info = libxc._itrf.xc_func_get_info(stock.xc_objs[0])
    n = libxc._itrf.xc_func_info_get_n_ext_params(info)
    name_fn = libxc._itrf.xc_func_info_get_ext_params_name
    name_fn.argtypes = (ctypes.c_void_p, ctypes.c_int)
    name_fn.restype = ctypes.c_char_p
    names = [name_fn(info, i).decode() for i in range(n)]
    if names != [p[0] for p in params]:
        raise RuntimeError(f"libxc ext-param order {names} != ferric {params}")

    libxc.register_custom_functional_(
        CUSTOM_XC, BASE_XC, ext_params={LIBXC_ID: [v for _, v in params]}
    )
    nlc = (((VV10_B, VV10_C), 1),)
    for table in (libxc._CUSTOM_FUNC_R, libxc._CUSTOM_FUNC_U):
        # cached_property: an instance attribute shadows it.
        table[CUSTOM_XC].nlc_coeff = nlc
        table[CUSTOM_XC.upper()] = table[CUSTOM_XC]
    rsh = tuple(float(x) for x in libxc.rsh_coeff(CUSTOM_XC))
    if any(abs(a - b) > 1e-15 for a, b in zip(rsh, (OMEGA, 1.0, -0.4))):
        raise RuntimeError(f"custom functional rsh_coeff {rsh} != (0.1, 1.0, -0.4)")
    if libxc.nlc_coeff(CUSTOM_XC) != nlc:
        raise RuntimeError("VV10 override did not take")
    if not libxc.is_nlc(CUSTOM_XC):
        raise RuntimeError("custom functional lost its VV10 flag")
    return {
        "libxc_id": LIBXC_ID,
        "based_on": BASE_XC,
        "ext_params": {k: v for k, v in params},
        "ext_param_names_from_libxc": names,
        "rsh_coeff_omega_alpha_beta": list(rsh),
        "stock_rsh_coeff": [float(x) for x in stock.rsh_coeff],
        "vv10_b_C": [VV10_B, VV10_C],
        "stock_vv10_b_C": list(stock.nlc_coeff[0][0]),
        "pyscf_libxc_version": libxc.__version__,
    }


def set_vv10(b: float, c: float) -> None:
    from pyscf.dft import libxc

    for table in (libxc._CUSTOM_FUNC_R, libxc._CUSTOM_FUNC_U):
        table[CUSTOM_XC].nlc_coeff = (((b, c), 1),)


# ---------------------------------------------------------------------------
# KS with ferric's J/K construction
# ---------------------------------------------------------------------------


def ks_class(base, j_mode: str):
    """`base` (RKS/ROKS) with ferric's J/K: J exact or RI-J; K DF in the
    attenuated metric only (see module doc)."""
    from pyscf import scf

    assert j_mode in ("exact", "rij")

    class FerricJK(base):
        _keys = {"ferric_dfobj"}

        def get_jk(
            self, mol=None, dm=None, hermi=1, with_j=True, with_k=True, omega=None
        ):
            if mol is None:
                mol = self.mol
            if dm is None:
                dm = self.make_rdm1()
            dfo = self.ferric_dfobj
            vj = vk = None
            if with_j:
                if omega not in (None, 0, 0.0):
                    raise NotImplementedError("attenuated J is never requested")
                if j_mode == "exact":
                    vj = scf.hf.get_jk(mol, dm, hermi, with_j=True, with_k=False)[0]
                else:
                    vj = dfo.get_jk(dm, hermi, with_j=True, with_k=False)[0]
            if with_k:
                if omega in (None, 0, 0.0):
                    vk = dfo.get_jk(dm, hermi, with_j=False, with_k=True, omega=-OMEGA)[
                        1
                    ]
                    vk = (
                        vk
                        + dfo.get_jk(dm, hermi, with_j=False, with_k=True, omega=OMEGA)[
                            1
                        ]
                    )
                else:
                    if abs(abs(omega) - OMEGA) > 1e-15:
                        raise ValueError(f"unexpected omega {omega}")
                    vk = dfo.get_jk(dm, hermi, with_j=False, with_k=True, omega=omega)[
                        1
                    ]
            return vj, vk

    return FerricJK


def make_ks(mol, kind: str, j_mode: str):
    from pyscf import df, dft

    base = {"rks": dft.rks.RKS, "roks": dft.roks.ROKS}[kind]
    mf = ks_class(base, j_mode)(mol, xc=CUSTOM_XC)
    aux, cart = common.pyscf_basis(JK_AUX, sorted(set(mol.elements)))
    assert not cart
    mf.ferric_dfobj = df.DF(mol, auxbasis=aux)
    mf.grids.atom_grid = MAIN_GRID
    mf.grids.prune = None
    mf.grids.radii_adjust = dft.radi.becke_atomic_radii_adjust
    mf.nlcgrids.atom_grid = NLC_GRID
    mf.nlcgrids.prune = None
    mf.nlcgrids.radii_adjust = dft.radi.becke_atomic_radii_adjust
    mf.conv_tol = CONV_TOL
    mf.conv_tol_grad = CONV_TOL_GRAD
    mf.max_cycle = 300
    mf.verbose = 0
    return mf


def run_rks(mol, j_mode: str):
    mf = make_ks(mol, "rks", j_mode)
    mf.kernel()
    if not mf.converged:
        raise RuntimeError(f"RKS ({j_mode} J) did not converge")
    g = float(np.abs(mf.get_grad(mf.mo_coeff, mf.mo_occ)).max())
    return mf, {"energy": float(mf.e_tot), "max_orbital_gradient": g}


def run_roks(mol):
    """Multi-start ROKS + Newton polish, lowest accepted, internally stable."""
    scan, best = [], None
    for guess in GUESSES:
        for ls in LEVEL_SHIFTS:
            mf = make_ks(mol, "roks", "exact")
            mf.init_guess = guess
            mf.level_shift = ls
            mf.kernel()
            e_diis, diis_conv = float(mf.e_tot), bool(mf.converged)
            nt = mf.newton()
            nt.conv_tol = CONV_TOL
            nt.conv_tol_grad = CONV_TOL_GRAD
            nt.max_cycle = 100
            nt.verbose = 0
            nt.kernel(mf.mo_coeff, mf.mo_occ)
            gmax = float(np.abs(nt.get_grad(nt.mo_coeff, nt.mo_occ)).max())
            entry = {
                "guess": guess,
                "level_shift": ls,
                "diis_converged": diis_conv,
                "energy_diis": e_diis,
                "newton_converged": bool(nt.converged),
                "energy": float(nt.e_tot),
                "max_orbital_gradient": gmax,
                "accepted": gmax <= ACCEPT_GMAX,
            }
            scan.append(entry)
            print(
                f"  ROKS start {guess:6s} ls={ls}: {entry['energy']:.12f} g={gmax:.1e}",
                flush=True,
            )
            if entry["accepted"] and (best is None or nt.e_tot < best[0].e_tot):
                best = (nt, entry)
    if best is None:
        raise RuntimeError(f"ROKS: no start reached max|g| <= {ACCEPT_GMAX}")
    mf, entry = best
    _mo, stable = common._stability_status(mf)
    if not stable:
        raise RuntimeError(f"ROKS: selected state is internally UNSTABLE ({entry})")
    acc = [s["energy"] for s in scan if s["accepted"]]
    return mf, {
        "energy": float(mf.e_tot),
        "max_orbital_gradient": entry["max_orbital_gradient"],
        "stability": {
            "internal_stable": True,
            "kind": "PySCF ROKS internal (real); response omits the VV10 kernel",
            "selected_start": {
                "guess": entry["guess"],
                "level_shift": entry["level_shift"],
            },
        },
        "start_scan": scan,
        "n_accepted": len(acc),
        "accepted_spread": float(max(acc) - min(acc)),
    }


def somo_character(mol, c_somo) -> dict:
    """Mulliken-like weight of the SOMO on O p_x/p_y (π) vs p_z/s (σ)."""
    s = mol.intor("int1e_ovlp")
    w = c_somo * (s @ c_somo)
    labels = mol.ao_labels()
    pi = sum(w[i] for i, lab in enumerate(labels) if re.search(r"O \d?p[xy]", lab))
    return {"pi_weight": float(pi), "is_pi": bool(pi > 0.9)}


def semicanonicalize(mf):
    """One unrestricted KS Fock build on the ROKS density, then occ/vir blocks
    diagonalised per spin (ferric `semicanonicalize`)."""
    mol = mf.mol
    dm = mf.make_rdm1()  # (2, nao, nao) for ROKS
    h = mf.get_hcore()
    veff = mf.get_veff(mol, dm)
    na, nb = mol.nelec
    c = mf.mo_coeff
    out = {}
    for spin, f_ao, nocc in (("a", h + veff[0], na), ("b", h + veff[1], nb)):
        f_mo = c.T @ f_ao @ c
        f_mo = 0.5 * (f_mo + f_mo.T)
        eo, uo = np.linalg.eigh(f_mo[:nocc, :nocc])
        ev, uv = np.linalg.eigh(f_mo[nocc:, nocc:])
        cn = np.hstack([c[:, :nocc] @ uo, c[:, nocc:] @ uv])
        fn = cn.T @ f_ao @ cn
        out[spin] = (
            cn,
            np.concatenate([eo, ev]),
            float(np.abs(fn[:nocc, nocc:]).max()),
        )
    return out


# ---------------------------------------------------------------------------
# AO order: PySCF -> ferric
# ---------------------------------------------------------------------------


def ferric_ao_permutation(mol, symbols) -> np.ndarray:
    """perm such that ferric_AO[k] == pyscf_AO[perm[k]]."""
    off = 0
    per_atom = {i: [] for i in range(mol.natm)}
    for ib in range(mol.nbas):
        ell = mol.bas_angular(ib)
        assert mol.bas_nctr(ib) == 1, "ferric splits general contractions"
        n = 2 * ell + 1
        per_atom[mol.bas_atom(ib)].append((ell, mol.bas_exp(ib).copy(), off))
        off += n
    assert off == mol.nao_nr()
    perm = []
    for ia, sym in enumerate(symbols):
        avail = list(per_atom[ia])
        for sh in common.ferric_shells(BASIS, common.z_of(sym)):
            ell = sh["l"]
            assert sh["pure"] or ell < 2
            hit = next(
                k
                for k, (l2, e2, _o) in enumerate(avail)
                if l2 == ell
                and len(e2) == len(sh["exps"])
                and np.allclose(e2, sh["exps"], rtol=1e-14, atol=0)
            )
            _l, _e, o = avail.pop(hit)
            # Within a shell: s; p as (x, y, z) in both codes (libcint orders
            # spherical p as x, y, z; ferric's p is Cartesian x, y, z); pure
            # d as m = -2..2 in both. Verified in the Rust test by overlap.
            perm.extend(range(o, o + 2 * ell + 1))
        assert not avail, f"atom {ia}: unmatched PySCF shells {avail}"
    perm = np.array(perm)
    assert sorted(perm.tolist()) == list(range(mol.nao_nr()))
    return perm


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------


def attenuated_raw(mol, aux_bas, omega_signed):
    """(P|op|mn) as (naux, nao, nao) and (P|op|Q); op = erf (ω>0) / erfc (ω<0) /
    Coulomb (ω=0)."""
    from pyscf import df

    auxmol = df.addons.make_auxmol(mol, aux_bas)
    auxmol.cart = False
    auxmol.build()
    with mol.with_range_coulomb(omega_signed), auxmol.with_range_coulomb(omega_signed):
        int3 = df.incore.aux_e2(mol, auxmol, intor="int3c2e", aosym="s1")
        int2 = auxmol.intor("int2c2e")
    nao, naux = mol.nao_nr(), auxmol.nao_nr()
    return int3.reshape(nao * nao, naux).T.reshape(naux, nao, nao), int2


def attenuated_factor(mol, aux_bas, omega_signed):
    """L^P_mn = Σ_Q (L^{-1})_PQ (Q|op|mn) with the metric (P|op|Q) in the SAME
    operator (ferric's coulomb_metric_2c(op) + cholesky_inverse_sqrt)."""
    from scipy.linalg import cholesky, solve_triangular

    int3, int2 = attenuated_raw(mol, aux_bas, omega_signed)
    naux, nao, _ = int3.shape
    low = cholesky(int2, lower=True)
    lao = solve_triangular(low, int3.reshape(naux, nao * nao), lower=True)
    return lao.reshape(naux, nao, nao), naux


def operator_identity_check(mol, aux_bas) -> float:
    """erfc(ω) + erf(ω) == Coulomb, on the raw 3-centre AND 2-centre integrals.
    (The erf(0.1) metric is numerically singular, so it is never factorised.)"""
    sr3, sr2 = attenuated_raw(mol, aux_bas, -OMEGA)
    lr3, lr2 = attenuated_raw(mol, aux_bas, OMEGA)
    c3, c2 = attenuated_raw(mol, aux_bas, 0.0)
    d = max(float(np.abs(sr3 + lr3 - c3).max()), float(np.abs(sr2 + lr2 - c2).max()))
    gap = float(np.abs(sr3 - c3).max())
    if d > 1e-10 or gap < 1e-3:
        raise RuntimeError(
            f"erfc+erf != Coulomb (|d| {d:.2e}) or erfc == Coulomb ({gap:.2e})"
        )
    return d


def pyscf_sr_mp2(mf_like, aux_bas, unrestricted) -> float:
    """PySCF DF-(U)MP2 with the erfc(ω)-attenuated DF object, λ = 1."""
    from pyscf import df
    from pyscf.mp import dfmp2, dfump2

    mol = mf_like.mol
    dfobj = df.DF(mol, auxbasis=aux_bas)
    dfobj.auxmol = df.addons.make_auxmol(mol, aux_bas)
    dfobj.auxmol.cart = False
    dfobj.auxmol.build()
    with dfobj.range_coulomb(-OMEGA) as rdf:
        cls = dfump2.DFUMP2 if unrestricted else dfmp2.DFMP2
        mp = cls(mf_like, frozen=None)
        mp.with_df = rdf
        mp.verbose = 0
        mp.kernel()
    return float(mp.e_corr)


def correlation(
    mol, lao_erfc, c_a, c_b, e_a, e_b, aux_bas, mf_like, unrestricted
) -> dict:
    na, nb = mol.nelec
    lam_l = np.sqrt(LAMBDA) * lao_erfc  # composite λ·erfc: every (pq|rs) scales by λ
    e_drv, r_drv = lin.spin_orbital(lam_l, c_a, c_b, e_a, e_b, na, nb, ladder=False)
    e_hh, r_hh = lin.spin_orbital(lam_l, c_a, c_b, e_a, e_b, na, nb, ladder=True)
    e_hh_l1, _ = lin.spin_orbital(lao_erfc, c_a, c_b, e_a, e_b, na, nb, ladder=True)
    e_drv_l1, _ = lin.spin_orbital(lao_erfc, c_a, c_b, e_a, e_b, na, nb, ladder=False)
    e_pyscf_mp2 = pyscf_sr_mp2(mf_like, aux_bas, unrestricted)
    checks = {
        "drivers_vs_lambda2_pyscf_sr_mp2": abs(e_drv - LAMBDA**2 * e_pyscf_mp2),
        "drivers_lambda1_vs_pyscf_sr_mp2": abs(e_drv_l1 - e_pyscf_mp2),
        "residual_drivers_max": r_drv,
        "residual_hh_max": r_hh,
    }
    if not unrestricted:
        cocc, cvir = c_a[:, :na], c_a[:, na:]
        lov = np.einsum("Pmn,mi,na->Pia", lam_l, cocc, cvir, optimize=True)
        loo = np.einsum("Pmn,mi,nj->Pij", lam_l, cocc, cocc, optimize=True)
        e_hh_sa, _ = lin.spin_adapted(lov, loo, e_a[:na], e_a[na:], ladder=True)
        e_drv_sa, _ = lin.spin_adapted(lov, loo, e_a[:na], e_a[na:], ladder=False)
        checks["hh_spin_adapted_vs_spin_orbital"] = abs(e_hh_sa - e_hh)
        checks["drivers_spin_adapted_vs_spin_orbital"] = abs(e_drv_sa - e_drv)
    for k, v in checks.items():
        bar = 1e-11 if k.startswith("residual") else TOL_ANCHOR
        if v > bar:
            raise RuntimeError(f"anchor {k} = {v:.2e} > {bar:.0e}")
    if abs(e_hh - e_drv) < 1e-5:
        raise RuntimeError("hh ladder correction unresolvable")
    linear_bug = LAMBDA * e_hh_l1
    return {
        "lambda": LAMBDA,
        "omega": OMEGA,
        "e_c_scaled": e_hh,
        "e_c_scaled_note": "lambda * E_c^{sr,omega,lambda} = ferric DoubleHybridResult.e_c_scaled "
        "= linlccd(op = lambda*erfc(omega)).correlation_energy",
        "e_c_wft": e_hh / LAMBDA,
        "drivers_only_scaled": e_drv,
        "drivers_only_note": "LadderVariant::DriversOnly under lambda*erfc(omega) = lambda^2 * SR-MP2",
        "pyscf_sr_mp2_lambda1": e_pyscf_mp2,
        "hh_lambda1": e_hh_l1,
        "negative_control_linear_lambda": linear_bug,
        "negative_control_linear_lambda_note": "lambda * E_hh(operator erfc at lambda = 1): the "
        "pre-PR-#239 construction (lambda outside the amplitude equations)",
        "linear_lambda_minus_correct": linear_bug - e_hh,
        "hh_minus_drivers": e_hh - e_drv,
        "checks": checks,
    }


# ---------------------------------------------------------------------------


def _to_list(a):
    return np.asarray(a).tolist()


def gen(system: str, charge: int, mult: int, xc_info: dict) -> Path:
    import pyscf
    from pyscf import scf

    xyz = common.MOL_DIR / f"{system}.xyz"
    symbols, coords = common.read_xyz(xyz)
    mol = common.build_pyscf_mol(xyz, BASIS, charge=charge, multiplicity=mult)
    basis_check = common.check_basis_like_for_like(mol, BASIS, symbols)
    jk_aux, _ = common.pyscf_basis(JK_AUX, symbols)
    corr_aux, cart = common.pyscf_basis(CORR_AUX, symbols)
    assert not cart
    perm = ferric_ao_permutation(mol, symbols)
    s_ferric = mol.intor("int1e_ovlp")[np.ix_(perm, perm)]
    op_check = operator_identity_check(mol, corr_aux)
    lao_erfc, naux_corr = attenuated_factor(mol, corr_aux, -OMEGA)
    ctx = f"{system}/{BASIS}"
    unrestricted = mult != 1

    payload = {
        "row": ROW_NAME,
        "system": system,
        "basis": BASIS,
        "jk_aux_basis": JK_AUX,
        "corr_aux_basis": CORR_AUX,
        "charge": charge,
        "multiplicity": mult,
        "nao": mol.nao_nr(),
        "naux_corr": naux_corr,
        "nelec_alpha": mol.nelec[0],
        "nelec_beta": mol.nelec[1],
        "nuclear_repulsion": float(mol.energy_nuc()),
        "lambda": LAMBDA,
        "omega": OMEGA,
        "ao_overlap_ferric_order": _to_list(s_ferric),
        "ao_order_note": "every matrix below is in FERRIC's AO order (rows = AOs)",
    }
    stab = None

    if not unrestricted:
        mf, ks = run_rks(mol, "exact")
        na = mol.nelectron // 2
        c = mf.mo_coeff
        e = mf.mo_energy
        corr = correlation(mol, lao_erfc, c, c, e, e, corr_aux, mf, False)
        # VV10 override is live: b = 6 (stock) must give a different E_KS.
        set_vv10(6.0, VV10_C)
        mf6 = make_ks(mol, "rks", "exact")
        e_b6 = float(mf6.kernel(dm0=mf.make_rdm1()))
        set_vv10(VV10_B, VV10_C)
        if abs(e_b6 - ks["energy"]) < 1e-5:
            raise RuntimeError("VV10 b override has no effect")
        ks.update(
            {
                "jk": "EXACT J; K = 0.6 K^DF[erfc] + 1.0 K^DF[erf], attenuated metrics, jkfit",
                "homo": float(e[na - 1]),
                "lumo": float(e[na]),
                "control_vv10_b6_energy": e_b6,
                "mo_coeff": _to_list(c[perm, :]),
                "mo_energy": _to_list(e),
            }
        )
        payload["ks"] = ks
        payload["corr"] = corr
        payload["total"] = ks["energy"] + corr["e_c_scaled"]

        mf_rij, ks_rij = run_rks(mol, "rij")
        corr_rij = correlation(
            mol, lao_erfc, mf_rij.mo_coeff, mf_rij.mo_coeff, mf_rij.mo_energy,
            mf_rij.mo_energy, corr_aux, mf_rij, False,
        )  # fmt: skip
        ks_rij["jk"] = (
            "RI-J (jkfit) + the same attenuated DF-K: ferric run_wb97x_l_v defaults"
        )
        payload["ks_rij"] = ks_rij
        payload["corr_rij"] = {
            k: corr_rij[k] for k in ("e_c_scaled", "drivers_only_scaled", "checks")
        }
        payload["total_rij"] = ks_rij["energy"] + corr_rij["e_c_scaled"]
        payload["rij_minus_exact_j_ks"] = ks_rij["energy"] - ks["energy"]
        hf = scf.RHF(mol)
        hf.conv_tol = 1e-11
        hf.verbose = 0
        payload["rhf_control"] = {
            "energy": float(hf.kernel()),
            "note": "exact RHF; ferric KS must MISS",
        }
        print(
            f"{ctx}: E_KS {ks['energy']:.12f} (RI-J {ks_rij['energy']:.12f}) "
            f"lamE_c {corr['e_c_scaled']:+.12f} drv {corr['drivers_only_scaled']:+.12f} "
            f"linear-bug {corr['negative_control_linear_lambda']:+.12f}"
        )
    else:
        mf, ks = run_roks(mol)
        na, nb = mol.nelec
        somo = somo_character(mol, mf.mo_coeff[:, nb])
        if not somo["is_pi"]:
            raise RuntimeError(f"{ctx}: ROKS SOMO is not pi ({somo}) — wrong state")
        sc = semicanonicalize(mf)
        (ca, ea, ova), (cb, eb, ovb) = sc["a"], sc["b"]
        # A UHF-shaped holder for PySCF's DFUMP2 anchor (mo_energy diagonal in
        # the occ/vir blocks; DFUMP2 reads only mo_coeff/mo_energy/mo_occ).
        uhf = scf.UHF(mol)
        uhf.mo_coeff = (ca, cb)
        uhf.mo_energy = (ea, eb)
        occ_a = np.zeros(mol.nao_nr())
        occ_a[:na] = 1
        occ_b = np.zeros(mol.nao_nr())
        occ_b[:nb] = 1
        uhf.mo_occ = (occ_a, occ_b)
        uhf.e_tot = ks["energy"]
        uhf.converged = True
        corr = correlation(mol, lao_erfc, ca, cb, ea, eb, corr_aux, uhf, True)
        ks.update(
            {
                "jk": "EXACT J; K = 0.6 K^DF[erfc] + 1.0 K^DF[erf] per spin, attenuated metrics, jkfit",
                "somo": somo,
                "semicanonical": {
                    "fock": "one unrestricted KS Fock build on the ROKS density (same XC/J/K)",
                    "max_ov_alpha": ova,
                    "max_ov_beta": ovb,
                    "mo_coeff_alpha": _to_list(ca[perm, :]),
                    "mo_coeff_beta": _to_list(cb[perm, :]),
                    "mo_energy_alpha": _to_list(ea),
                    "mo_energy_beta": _to_list(eb),
                },
            }
        )
        stab = {"roks": ks["stability"]}
        payload["ks"] = ks
        payload["corr"] = corr
        payload["total"] = ks["energy"] + corr["e_c_scaled"]
        hf = scf.ROHF(mol)
        hf.conv_tol = 1e-11
        hf.verbose = 0
        payload["rohf_control"] = {
            "energy": float(hf.kernel()),
            "note": "exact ROHF; ferric KS must MISS",
        }
        print(
            f"{ctx}: E_ROKS {ks['energy']:.12f} spread {ks['accepted_spread']:.1e} "
            f"lamE_c {corr['e_c_scaled']:+.12f} drv {corr['drivers_only_scaled']:+.12f} "
            f"max_ov a/b {ova:.1e}/{ovb:.1e}"
        )

    payload["provenance"] = common.provenance(
        code="PySCF + numpy",
        version=pyscf.__version__,
        keywords={
            "xc": xc_info,
            "scf": "dft.RKS"
            if not unrestricted
            else "dft.ROKS multi-start + newton polish",
            "jk": "see ks.jk; K served by df.DF(...).get_jk(omega=-0.1 | +0.1) only",
            "correlation": "numpy LinLCCD(hh) (gen_linlccd.spin_orbital eigen-solve) with "
            "integrals of lambda*erfc(omega r)/r fitted in the erfc metric, all electrons",
            "numpy": np.__version__,
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
            "nlc_vv10": {
                "atom_grid": list(NLC_GRID),
                "prune": None,
                "radii_adjust": "becke_atomic_radii_adjust",
            },
        },
        aux={
            "jk": {
                "name": JK_AUX,
                "sha256": common.sha256_file(common.basis_json_path(JK_AUX)),
            },
            "correlation": {
                "name": CORR_AUX,
                "sha256": common.sha256_file(common.basis_json_path(CORR_AUX)),
            },
        },
        frozen_core=None,
        scf_conv={"conv_tol": CONV_TOL, "conv_tol_grad": CONV_TOL_GRAD},
        stability=stab,
        generator="scripts/validation/gen_wb97xlv.py",
        extra={
            "basis_self_check": basis_check,
            "operator_identity_erfc_plus_erf_minus_coulomb_3c": op_check,
        },
    )
    return common.write_reference(ROW, system, BASIS, payload)


def main() -> int:
    only = set(sys.argv[1:])
    unknown = only - set(SYSTEMS)
    if unknown:
        raise SystemExit(f"unknown systems {sorted(unknown)}; known {sorted(SYSTEMS)}")
    xc_info = register_functional()
    written = []
    for system, (charge, mult) in SYSTEMS.items():
        if only and system not in only:
            continue
        written.append(gen(system, charge, mult, xc_info))
    for p in written:
        print(f"wrote {p.relative_to(common.ROOT)} ({p.stat().st_size} B)")
    print(f"GEN_WB97XLV_DONE written={len(written)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
