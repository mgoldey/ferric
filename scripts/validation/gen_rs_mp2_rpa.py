"""PySCF/numpy references for the VALIDATION.md "RS-MP2-RPA B and T" row.

Consumer: crates/ferric-rpa/tests/validation_rs_mp2_rpa.rs.
Output:   testdata/reference/validation/rs_mp2_rpa/<system>_<basis>.json

WHAT FERRIC COMPUTES (read from crates/ferric-rpa/src/rs_mp2_rpa.rs)
--------------------------------------------------------------------
`rs_mp2_lr_rpa(mol, obs, dfbs, rhf, cfg)` builds, per operator
op in {Coulomb, erf(w), erfc(w)}, ONE `compute_rpa_intermediates(.., op, ..)`
(B^P_ia = [V_op^-1/2 (Q|op|ia)]_P: the SAME op in the 3-centre integrals AND
the 2-centre metric; V^-1/2 by regularized eigh (drop eigenvalues < 1e-10) for
erf, Cholesky for Coulomb/erfc) and feeds that one B to both

  * `spin_components_from_b_ov` -> SC[op] = {E_OS, E_SS, E_MP2 = E_OS + E_SS},
    E_OS = sum_ijab (ia|jb)^2 / D,  E_SS = sum_ijab (ia|jb)[(ia|jb)-(ib|ja)] / D,
    D = e_i + e_j - e_a - e_b  (closed shell, no frozen core here);
  * `run_pdep_rpa_from_intermediates` -> E_dRPA[op], the closed-shell dRPA
    sum_k w_k/(2 pi)[ln det(I + Pi) - tr Pi], Pi = B^T diag(4D/(w^2+D^2)) B.

The RHF is the ordinary full-Coulomb RHF. The reported fields:

  e_mp2_full  = SC[Coulomb].E_MP2                      (rs_mp2_rpa.rs:464)
  e_sr_mp2    = SC[erfc].E_MP2                         (:465)
  e_lr_mp2    = SC[erf].E_MP2                          (:466)
  e_dmp2_lr   = 2 SC[erf].E_OS                         (:461)
  B (DeltaLr):
    e_drpa_lr    = E_dRPA[erf]                         (:408-413)
    e_corr_naive = SC[erfc].E_MP2 + E_dRPA[erf]        (:414)
    e_corr       = SC[C].E_MP2 + E_dRPA[erf] - 2 SC[erf].E_OS   (:404, :410)
  T (CoupledRings):
    e_delta_drpa_full = E_dRPA[C] - 2 SC[C].E_OS       (:442)
    e_delta_drpa_sr   = E_dRPA[erfc] - 2 SC[erfc].E_OS (:443)
    e_corr            = SC[C].E_MP2 + dfull - dsr      (:445)
  total_energy = E_RHF + e_corr                        (:473)

EXACT LIMITS (of the formulas above, independent of any approximation):
  B: w -> 0   erf -> 0      => e_corr -> E_MP2[C]
     w -> inf erf -> C      => e_corr -> E_MP2[C] + (E_dRPA[C] - 2 E_OS[C])
  T: w -> 0   erfc -> C     => the two Delta terms cancel => E_MP2[C]
     w -> inf erfc -> 0     => e_corr -> E_MP2[C] + (E_dRPA[C] - 2 E_OS[C])
Both formulations share both endpoints; they differ at finite w by the mixed
SR x LR rings (T has them, B does not).

THE REFERENCE (independent assembly)
------------------------------------
numpy on PySCF integrals, reusing gen_attenuated_rpa's machinery (int3c2e AND
int2c2e under Mole.with_range_coulomb on both Moles; same metric factor as
ferric; numpy dRPA on the 40-point Gauss-Legendre grid, x0 = 0.5), plus a
numpy RI-MP2 spin decomposition from the same B. Every component is stored
separately so a failing test names the piece. Checks run before writing:

  1. erf + erfc == Coulomb on int3c2e and int2c2e (kernel sign convention);
  2. Coulomb numpy dRPA == PySCF gw.rpa.RPA (same aux, grid) to 1e-10;
  3. Coulomb numpy E_OS / E_SS / E_MP2 == PySCF dfmp2.DFMP2 (same aux) to 1e-10;
  4. the ring-coefficient check: the 2nd-order term of the numpy dRPA by the
     frequency route, sum_k w_k/(2 pi)(-1/2) tr Pi^2 on a converged 160-point
     grid, must equal 2 E_OS (an energy-denominator sum) for every kernel --
     this pins the "2" in 2 E_OS against the RPA normalization by an
     independent route. The same term on the production 40-point grid is
     recorded too: that quadrature error enters E_dRPA but not 2 E_OS, so it
     is carried by Delta dRPA (identically in ferric and here);
  5. the four exact limits above, on the numpy side (recorded under `limits`).

omega grid (Bohr^-1; ferric's Operator takes Bohr^-1, the CLI/Python take
Angstrom^-1): 0.222254 (= 0.420 A^-1, `RsMp2RpaConfig::default().omega`) and
0.42 (the default's A^-1 digits read as Bohr^-1: the unit-slip control, and
the second omega of the row).

CONTROL VALUES (for the test to assert ferric MISSES them):
  * `wrong_sign`: B with +2 E_OS[erf]; T with + dsr instead of - dsr.
  * `dmp2_coeff_1`: the ring subtraction with E_OS instead of 2 E_OS
    (B: E_MP2 + E_dRPA[erf] - E_OS[erf]; T: both Delta terms with 1 E_OS).

Run (light; under a minute per system):
    OPENBLAS_NUM_THREADS=1 scripts/validation/run_slot.sh --light -- \\
        uv run --no-sync python scripts/validation/gen_rs_mp2_rpa.py [system_basis ...]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402
import gen_attenuated_rpa as gar  # noqa: E402
import gen_urpa  # noqa: E402

ROW = "rs_mp2_rpa"
ROW_NAME = "RS-MP2-RPA B and T"
NW = gar.NW
X0 = gar.X0
BOHR_PER_ANG = gar.BOHR_PER_ANG
OMEGA_DEFAULT_BOHR = gar.OMEGA_DEFAULT_BOHR
OMEGAS = (OMEGA_DEFAULT_BOHR, 0.42)
NUMPY_VS_PYSCF_MAX = 1e-10
# Ring-coefficient check: the frequency-integral 2nd-order dRPA term on a
# converged grid vs 2 E_OS (measured ~1e-14 rel; GL-40 itself is ~3e-9 rel).
RING_NW = 160
RING_REL_MAX = 1e-12
# Limit scans (Bohr^-1). B/w->0 uses erf at small w (ill-conditioned erf
# metric, most modes dropped -- the erf pieces vanish either way).
LIMIT_SCANS = {
    "B_omega_to_0": (1e-2, 1e-3, 1e-4),
    "B_omega_to_inf": (1e3, 1e4, 1e5),
    "T_omega_to_0": (1e-3, 1e-4, 1e-5),
    "T_omega_to_inf": (1e2, 1e3, 1e4),
}
# The anchor omega the test runs for each limit (one of each scan).
ANCHORS = {
    "B_omega_to_0": 1e-2,
    "B_omega_to_inf": 1e5,
    "T_omega_to_0": 1e-5,
    "T_omega_to_inf": 1e2,
}
CASES = gar.CASES


def spin_components(B, mf):
    """E_OS, E_SS, E_MP2 from B (nov, naux), closed shell, no frozen core."""
    import numpy as np

    nocc = int(np.count_nonzero(mf.mo_occ > 0))
    e = mf.mo_energy
    eo, ev = e[:nocc], e[nocc:]
    nvir = ev.size
    B3 = B.reshape(nocc, nvir, -1)
    e_os = 0.0
    e_ss = 0.0
    for i in range(nocc):
        for j in range(nocc):
            v = B3[i] @ B3[j].T  # (a, b) = (ia|jb)
            d = eo[i] + eo[j] - ev[:, None] - ev[None, :]
            e_os += float(np.sum(v * v / d))
            e_ss += float(np.sum(v * (v - v.T) / d))
    return {"e_os": e_os, "e_ss": e_ss, "e_mp2": e_os + e_ss}


class Kernels:
    """Per-(kind, omega) cache of B, spin components and dRPA."""

    def __init__(self, mf, mol, auxmol):
        self.mf, self.mol, self.auxmol = mf, mol, auxmol
        self.d = gar.gaps(mf)
        self._cache = {}

    def B(self, kind, omega):
        v3, v2 = gar.ints(self.mol, self.auxmol, gar.pyscf_omega(kind, omega))
        m, info = gar.metric_inv_sqrt(v2, kind)
        return gar.b_ov(self.mf, v3, m), info

    def get(self, kind, omega=0.0):
        key = (kind, 0.0 if kind == "coulomb" else omega)
        if key not in self._cache:
            B, info = self.B(kind, omega)
            sc = spin_components(B, self.mf)
            sc["e_drpa"] = gar.numpy_rpa(B, self.d)
            sc["metric"] = info
            self._cache[key] = (B, sc)
        return self._cache[key][1]

    def ring_coefficient(self, kind, omega=0.0):
        """Second-order (ring) term of the numpy dRPA by the frequency route:
        E2 = sum_k w_k/(2 pi) (-1/2) tr Pi(iw_k)^2 (the lambda^2 coefficient of
        ln det(I + lambda Pi) - lambda tr Pi), on a converged RING_NW-point grid,
        vs the energy-denominator route 2 E_OS. Returns (E2, rel diff, E2 on
        the production NW grid)."""
        import numpy as np

        self.get(kind, omega)
        key = (kind, 0.0 if kind == "coulomb" else omega)
        B, sc = self._cache[key]

        def e2(nw):
            freqs, wts = gen_urpa.scaled_legendre(nw, X0)
            e = 0.0
            for w, wt in zip(freqs, wts):
                chi = 4.0 * self.d / (w**2 + self.d**2)
                pi = (B.T * chi) @ B
                e += wt / (2.0 * np.pi) * (-0.5 * float(np.sum(pi * pi)))
            return e

        e_conv = e2(RING_NW)
        rel = abs(e_conv - 2.0 * sc["e_os"]) / abs(2.0 * sc["e_os"])
        return e_conv, rel, e2(NW)


def components(K, omega):
    """Every RsMp2RpaResult field (both formulations) at omega."""
    c = K.get("coulomb")
    lr = K.get("erf", omega)
    sr = K.get("erfc", omega)
    mp2 = c["e_mp2"]
    d_full = c["e_drpa"] - 2.0 * c["e_os"]
    d_sr = sr["e_drpa"] - 2.0 * sr["e_os"]
    e_b = mp2 + lr["e_drpa"] - 2.0 * lr["e_os"]
    e_t = mp2 + d_full - d_sr
    return {
        "omega_bohr_inv": omega,
        "omega_angstrom_inv": omega * BOHR_PER_ANG,
        "pieces": {
            "coulomb": {k: c[k] for k in ("e_os", "e_ss", "e_mp2", "e_drpa")},
            "erf": {k: lr[k] for k in ("e_os", "e_ss", "e_mp2", "e_drpa")},
            "erfc": {k: sr[k] for k in ("e_os", "e_ss", "e_mp2", "e_drpa")},
        },
        "e_mp2_full": mp2,
        "e_sr_mp2": sr["e_mp2"],
        "e_lr_mp2": lr["e_mp2"],
        "e_dmp2_lr": 2.0 * lr["e_os"],
        "B": {
            "e_drpa_lr": lr["e_drpa"],
            "e_corr_naive": sr["e_mp2"] + lr["e_drpa"],
            "e_corr": e_b,
        },
        "T": {
            "e_delta_drpa_full": d_full,
            "e_delta_drpa_sr": d_sr,
            "e_corr": e_t,
        },
        "controls": {
            "B_wrong_sign": mp2 + lr["e_drpa"] + 2.0 * lr["e_os"],
            "T_wrong_sign": mp2 + d_full + d_sr,
            "B_dmp2_coeff_1": mp2 + lr["e_drpa"] - lr["e_os"],
            "T_dmp2_coeff_1": mp2
            + (c["e_drpa"] - c["e_os"])
            - (sr["e_drpa"] - sr["e_os"]),
        },
        "B_minus_T": e_b - e_t,
        "erf_metric": lr["metric"],
    }


def pyscf_dfmp2(mf, aux):
    from pyscf.mp import dfmp2

    pt = dfmp2.DFMP2(mf)
    pt.with_df = gen_urpa.make_df(mf.mol, aux)
    pt.verbose = 0
    pt.kernel()
    return {
        "e_os": float(pt.e_corr_os),
        "e_ss": float(pt.e_corr_ss),
        "e_mp2": float(pt.e_corr),
    }


def gen_case(key):
    import numpy as np
    import pyscf
    import scipy
    from pyscf import gto, scf

    system, basis_name, aux_name = CASES[key]
    t0 = time.perf_counter()
    xyz = common.MOL_DIR / f"{system}.xyz"
    symbols, coords = common.read_xyz(xyz)
    mol = common.build_pyscf_mol(xyz, basis_name)
    ll = common.check_basis_like_for_like(mol, basis_name, symbols)
    aux = gen_urpa.aux_dict(aux_name, symbols)
    auxmol = gto.M(
        atom=common.pyscf_atom_bohr(symbols, coords),
        unit="Bohr",
        basis=aux,
        cart=False,
        verbose=0,
    )
    naux = int(auxmol.nao_nr())
    assert naux == common.ferric_nao(aux_name, symbols), "aux count differs from ferric"

    mf = scf.RHF(mol)
    mf.conv_tol = gar.CONV_TOL
    mf.conv_tol_grad = gar.CONV_TOL_GRAD
    mf.max_cycle = 500
    mf.verbose = 0
    mf.kernel()
    assert mf.converged, "RHF did not converge"
    rhf_stable = common._stability_status(mf)[1]
    assert rhf_stable, "RHF not internally stable"

    K = Kernels(mf, mol, auxmol)
    c = K.get("coulomb")

    # --- check 2: Coulomb dRPA vs PySCF RPA --------------------------------
    e_rpa_pyscf = gar.pyscf_rpa_coulomb(mf, aux)
    d_rpa = abs(c["e_drpa"] - e_rpa_pyscf)
    if d_rpa > NUMPY_VS_PYSCF_MAX:
        raise RuntimeError(f"numpy dRPA {c['e_drpa']:.12f} vs PySCF {e_rpa_pyscf:.12f}")
    # --- check 3: Coulomb spin components vs PySCF DFMP2 -------------------
    mp2_pyscf = pyscf_dfmp2(mf, aux)
    d_mp2 = max(abs(c[k] - mp2_pyscf[k]) for k in ("e_os", "e_ss", "e_mp2"))
    if d_mp2 > NUMPY_VS_PYSCF_MAX:
        raise RuntimeError(f"numpy MP2 spin components {c} vs PySCF DFMP2 {mp2_pyscf}")

    worst_ident = 0.0
    blocks = []
    ring_checks = {}
    for omega in OMEGAS:
        worst_ident = max(worst_ident, gar.kernel_identity(mol, auxmol, omega))
        blk = components(K, omega)
        blocks.append(blk)
        for kind in ("erf", "erfc"):
            e2, rel, e2_nw = K.ring_coefficient(kind, omega)
            ring_checks[f"{kind}({omega:.6f})"] = {
                "e2_freq_converged": e2,
                "rel_vs_2eos": rel,
                "e2_freq_nw40_minus_2eos": e2_nw - 2.0 * K.get(kind, omega)["e_os"],
            }
        print(
            f"{key} w={omega:.6f} B {blk['B']['e_corr']:.12f} T {blk['T']['e_corr']:.12f} "
            f"B-T {blk['B_minus_T']:+.3e} MP2 {blk['e_mp2_full']:.12f} "
            f"dmp2_lr {blk['e_dmp2_lr']:.3e} drpa_lr {blk['B']['e_drpa_lr']:.3e} "
            f"dfull {blk['T']['e_delta_drpa_full']:+.3e} dsr {blk['T']['e_delta_drpa_sr']:+.3e}"
        )
    e2, rel, e2_nw = K.ring_coefficient("coulomb")
    ring_checks["coulomb"] = {
        "e2_freq_converged": e2,
        "rel_vs_2eos": rel,
        "e2_freq_nw40_minus_2eos": e2_nw - 2.0 * c["e_os"],
    }
    worst_ring = max(v["rel_vs_2eos"] for v in ring_checks.values())
    worst_nw = max(abs(v["e2_freq_nw40_minus_2eos"]) for v in ring_checks.values())
    print(
        f"{key} ring coefficient: max rel |E2(freq, nw={RING_NW}) - 2 E_OS| = {worst_ring:.2e}; "
        f"GL-{NW} ring quadrature error <= {worst_nw:.2e} Ha"
    )
    assert worst_ring < RING_REL_MAX, f"ring-coefficient check failed: {ring_checks}"

    # --- limits ---------------------------------------------------------------
    mp2 = c["e_mp2"]
    mp2_plus_dfull = mp2 + c["e_drpa"] - 2.0 * c["e_os"]
    targets = {
        "B_omega_to_0": mp2,
        "B_omega_to_inf": mp2_plus_dfull,
        "T_omega_to_0": mp2,
        "T_omega_to_inf": mp2_plus_dfull,
    }
    limits = {}
    for name, omegas in LIMIT_SCANS.items():
        form = name[0]
        limits[name] = {"target": targets[name], "scan": {}}
        for omega in omegas:
            blk = components(K, omega)
            e = blk[form]["e_corr"]
            limits[name]["scan"][f"{omega:g}"] = {
                "e_corr": e,
                "minus_target": e - targets[name],
                "e_corr_other_formulation": blk["T" if form == "B" else "B"]["e_corr"],
                "components": {
                    k: blk[k]
                    for k in ("e_mp2_full", "e_sr_mp2", "e_lr_mp2", "e_dmp2_lr")
                }
                | {f"B_{k}": v for k, v in blk["B"].items()}
                | {f"T_{k}": v for k, v in blk["T"].items()},
            }
            print(
                f"{key} limit {name} w={omega:g}: e_corr - target = {e - targets[name]:+.3e}"
            )
    anchors = {}
    for name, w in ANCHORS.items():
        ent = limits[name]["scan"][f"{w:g}"]
        anchors[name] = {
            "omega": w,
            "e_corr": ent["e_corr"],
            "target": targets[name],
            "minus_target": ent["minus_target"],
            "components": ent["components"],
        }

    payload = {
        "row": ROW_NAME,
        "system": system,
        "basis": basis_name,
        "aux": aux_name,
        "charge": 0,
        "multiplicity": 1,
        "nao": int(mol.nao_nr()),
        "naux": naux,
        "nelectron": int(mol.nelectron),
        "nuclear_repulsion": float(mol.energy_nuc()),
        "like_for_like": ll,
        "rhf": {"energy": float(mf.e_tot), "internal_stable": bool(rhf_stable)},
        "nw": NW,
        "x0": X0,
        "lindep": gar.LINDEP,
        "frozen_core": 0,
        "coulomb": {
            "e_os": c["e_os"],
            "e_ss": c["e_ss"],
            "e_mp2": c["e_mp2"],
            "e_drpa": c["e_drpa"],
            "e_drpa_pyscf_rpa": e_rpa_pyscf,
            "pyscf_dfmp2": mp2_pyscf,
        },
        "omegas": blocks,
        "limits": limits,
        "anchors": anchors,
        "ring_coefficient_checks": ring_checks,
        "checks": {
            "erf_plus_erfc_minus_coulomb_max_abs": worst_ident,
            "numpy_coulomb_drpa_vs_pyscf_rpa_abs": d_rpa,
            "numpy_coulomb_mp2_vs_pyscf_dfmp2_max_abs": d_mp2,
            "ring_coefficient_max_rel": worst_ring,
        },
        "generator_seconds": time.perf_counter() - t0,
    }
    payload["provenance"] = common.provenance(
        code="PySCF (integrals, RHF, Coulomb RPA + DFMP2 anchors) + numpy "
        "(attenuated RI-MP2 spin components, dRPA, B/T assembly)",
        version=pyscf.__version__,
        keywords={
            "rhf": "scf.RHF, exact 4-index Coulomb J/K, conv_tol 1e-11",
            "operator": "Mole.with_range_coulomb(w) on mol AND auxmol: w>0 erf, w<0 erfc",
            "metric_factor": "erf: symmetric eigh V^-1/2, drop eigenvalues < 1e-10; "
            "erfc/Coulomb: Cholesky L^-1 (ferric metric_inverse_sqrt)",
            "mp2": "E_OS = sum (ia|jb)^2/D, E_SS = sum (ia|jb)[(ia|jb)-(ib|ja)]/D, same B as RPA",
            "rpa": "closed-shell dRPA, full rank, ln det(I+Pi) - tr Pi",
            "B": "E_MP2[C] + E_dRPA[erf] - 2 E_OS[erf]",
            "T": "E_MP2[C] + (E_dRPA[C] - 2 E_OS[C]) - (E_dRPA[erfc] - 2 E_OS[erfc])",
            "nw": NW,
            "x0": X0,
            "frequency_grid": "Gauss-Legendre, w = x0 (1+x)/(1-x) (rpa._get_scaled_legendre_roots)",
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        basis_name=basis_name,
        xyz_path=xyz,
        coords_bohr=coords,
        symbols=symbols,
        grid=None,
        aux=aux_name,
        frozen_core=0,
        scf_conv={"conv_tol": gar.CONV_TOL, "conv_tol_grad": gar.CONV_TOL_GRAD},
        stability={"rhf_internal_stable": bool(rhf_stable)},
        generator="scripts/validation/gen_rs_mp2_rpa.py",
        extra={
            "aux_basis_json": str(
                common.basis_json_path(aux_name).relative_to(common.ROOT)
            ),
            "aux_basis_sha256": common.sha256_file(common.basis_json_path(aux_name)),
        },
    )
    path = common.write_reference(ROW, system, basis_name, payload)
    print(
        f"{key}: E_RHF {mf.e_tot:.10f} MP2 {mp2:.12f} (DFMP2 |d| {d_mp2:.1e}) "
        f"dRPA {c['e_drpa']:.12f} (RPA |d| {d_rpa:.1e}) identity {worst_ident:.1e} "
        f"({payload['generator_seconds']:.1f} s) -> {path.relative_to(common.ROOT)}"
    )


def main(argv):
    want = list(argv) or list(CASES)
    unknown = [k for k in want if k not in CASES]
    if unknown:
        raise SystemExit(f"unknown cases: {unknown} (known: {sorted(CASES)})")
    for k in want:
        gen_case(k)
    print(f"GEN_RS_MP2_RPA_DONE written={len(want)}")


if __name__ == "__main__":
    main(sys.argv[1:])
