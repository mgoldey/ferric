//! Range-separated exact exchange on the k-point RS-GDF J/K
//! (`ferric_pbc::rsgdf::kpoint::klr`, `KRsGdf::build_rsh`).
//!
//! # Physics hypothesis / artifact hypothesis (written before measuring)
//!
//! `K = c_sr K[1/r] + (c_lr − c_sr) K[erf(ωr)/r] + v_M^{mix} S D S`, the erf
//! piece fitted in G-space only. If the construction is right, the fitted
//! kernels equal the dense-AFT RSH kernels to the aux-fit error (zero in the
//! trivial-aux limit), the E/cell of an N-point mesh equals the Gamma energy
//! of the N-cell supercell / N, and PySCF's pinned RSH energies are
//! reproduced to the fit error. If the build is broken in one of the ways the
//! mutants model (no damping, wrong ω units, erf/erfc swapped, no or
//! primitive-cell Madelung), the kernel-level anchors move by O(1e-3..1e-1)
//! and the supercell anchor by the Madelung or kernel difference. The
//! anchors that can NOT see a defect: the `ω → 0` anchor sees only the
//! Madelung bookkeeping (the erf K set is empty, so the attenuation, the ω
//! units and the role swap are invisible there), and the equal-fraction
//! anchor sees nothing about the erf piece (its coefficient is 0).
//!
//! The kernel-level anchors compare `fitted_kernels` (`mixed_k_kernel`); the
//! production contraction (`add_k_blocks` + `mix_rsh_exchange` + Madelung) is
//! checked against the dense builder on fixed densities by the `prod` column of
//! `kernel_gap`. The Madelung constant is compared with a closed form summed in
//! this file, not with a second call of `RshParams::madelung`.
//!
//! Anchors:
//! * `kernel_matches_dense_in_the_trivial_aux_limit` — fitted `K` vs the
//!   dense RSH `K` (same layout), non-TRIM 1×1×3 mesh, HSE-like and
//!   CAM-like fractions, plus the mixed Madelung constant.
//! * `small_omega_is_the_global_hybrid_without_exxdiv` — `ω → 0`,
//!   `ExxDiv::None`: RSH ≡ global hybrid `c_sr`; with Ewald the difference is
//!   exactly `(c_lr − c_sr) v_M^{LR}(ω) ~ 2ω/√π`, not 0.
//! * `rsgdf_rsh_is_the_dense_rsh_to_the_fit_error` — SCF energies, HSE06 and
//!   CAM-B3LYP, RS-GDF vs dense on the same grid; the difference must be of
//!   the order of the plain-Coulomb (PBE0) RS-GDF-vs-dense difference.
//! * `rsgdf_rsh_k_mesh_is_the_supercell` — 1×1×3 mesh E/cell vs the Gamma
//!   energy of the 3-cell supercell / 3 (both RS-GDF).
//! * `rsgdf_rsh_matches_pinned_pyscf` — PySCF 2.13 KRKS pins of `pbc_krsh`.
//! * `rsgdf_rsh_mutations_break_the_anchors`, the derivative source checks
//!   (the force and stress themselves: `pbc_krsh_rsgdf_force.rs`,
//!   `validation_krsh_rsgdf_force.rs`).
//!
//! WEEKLY HALF: `rsgdf_rsh_is_the_dense_rsh_to_the_fit_error` (both meshes x
//! both aux sets x three functionals) and
//! `rsgdf_rsh_matches_pinned_pyscf_to_the_fit_error` (all six PySCF pins on a
//! 61^3 grid), each with CI >= 120 s, run in `validation_pbc_krsh_rsgdf.rs`;
//! the helpers are `krsh_rsgdf_support/mod.rs`. Per-commit canaries:
//! `rsgdf_rsh_is_the_dense_rsh_to_the_fit_error_canary` (1x1x1, cc-pVDZ-RI,
//! HSE06 and CAM-B3LYP) and `rsgdf_rsh_matches_one_pinned_pyscf_energy`
//! (the HSE06 1x1x1 pin).

mod common;
#[path = "krsh_rsgdf_support/mod.rs"]
mod support;

use common::*;
use support::*;

#[test]
fn kernel_matches_dense_in_the_trivial_aux_limit() {
    for (omega, f) in [
        (0.11, HSE_LIKE),
        (0.2, HSE_LIKE),
        (0.2, CAM_LIKE),
        (0.35, CAM_LIKE),
    ] {
        for exxdiv in [ExxDiv::None, ExxDiv::Ewald] {
            let (d, dv, scale, prod) = kernel_gap([1, 1, 3], rsh(omega, f), exxdiv, None);
            eprintln!(
                "trivial aux 1x1x3 w={omega} f={f:?} {exxdiv:?}: max|dK| {d:.2e} (|K| {scale:.2e}), |d v_M| {dv:.1e}"
            );
            assert!(d < KERNEL_BAR, "w={omega} {f:?} {exxdiv:?}: {d:e}");
            // v_M against the test's own closed-form sum (not RshParams::madelung).
            assert!(dv < 1e-10, "w={omega} {f:?} {exxdiv:?}: v_M {dv:e}");
            assert!(
                prod < KERNEL_BAR,
                "w={omega} {f:?} {exxdiv:?}: production K {prod:e}"
            );
            assert!(scale > 0.1, "kernel is not trivially small: {scale}");
        }
    }
}

#[test]
fn kernel_anchor_sees_every_mutant() {
    use KRshMutant::*;
    let r = rsh(0.2, CAM_LIKE);
    for m in [
        NoAttenuation,
        OmegaUnits,
        SwappedRoles,
        NoMadelung,
        PrimitiveMadelung,
    ] {
        let (d, dv, _, prod) =
            kernel_gap([1, 1, 3], r, ExxDiv::Ewald, Some(KRsGdfMutation::Rsh(m)));
        eprintln!("{m:?}: max|dK| {d:.2e}, |d v_M| {dv:.2e}");
        // Madelung mutants shift v_M (and the production K by v_M S D S);
        // kernel mutants shift the fitted kernel and the production K.
        let seen = match m {
            NoMadelung | PrimitiveMadelung => dv > MUTANT_BAR && prod > MUTANT_BAR,
            _ => d > MUTANT_BAR && prod > MUTANT_BAR,
        };
        assert!(seen, "{m:?} left the anchor green (dK {d:e}, dvM {dv:e})");
    }
}

/// `ω → 0`: no K vector lies inside the erf Gaussian cut, so the fitted RSH
/// kernel is exactly `c_sr` times the full-Coulomb fit, i.e. the global
/// hybrid; with Ewald the only difference is the constant
/// `(c_lr − c_sr) v_M^{LR}(ω)`, which tends to 0 only linearly (`2ω/√π`).
#[test]
fn small_omega_is_the_global_hybrid_without_exxdiv() {
    let cell = h2_cell(4.0);
    let prep = prep_for(&cell, &single_s_h(ANCHOR_ALPHA));
    let mesh = KPointMesh::gamma_centred(&cell, [1, 1, 3]).unwrap();
    let hk = periodic_hcore_kpts(
        &cell,
        &prep,
        &mesh,
        &PeriodicHcoreConfig::with_omega(OMEGA_HCORE),
    )
    .unwrap();
    let site = SiteBasis::new(&anchor_sites(&cell), 0).unwrap();
    let omega = 0.01;
    let r = rsh(omega, CAM_LIKE);
    for exxdiv in [ExxDiv::None, ExxDiv::Ewald] {
        let cfg = gdf_cfg(exxdiv, None);
        let plain = KRsGdf::build(&cell, &prep, &site.prep, &mesh, &hk.s, &cfg).unwrap();
        let mix = KRsGdf::build_rsh(&cell, &prep, &site.prep, &mesh, &hk.s, &cfg, r).unwrap();
        let (_, kp) = plain
            .fitted_kernels(DEFAULT_K_FITTED_KERNELS_MAX_BYTES)
            .unwrap();
        let (_, km) = mix
            .fitted_kernels(DEFAULT_K_FITTED_KERNELS_MAX_BYTES)
            .unwrap();
        let mut worst = 0.0_f64;
        for (a, b) in kp.iter().zip(&km) {
            worst = worst.max(cmax(&b.mapv(|z| z / r.c_sr), a));
        }
        let sc = Cell::new(cell.mol().clone(), mesh.supercell_lattice()).unwrap();
        let lr = ferric_pbc::rsh::madelung_lr(&sc, omega).unwrap();
        let want = match exxdiv {
            ExxDiv::None => 0.0,
            ExxDiv::Ewald => closed_form_mixed_madelung(&cell, &mesh, r),
        };
        eprintln!(
            "w={omega} {exxdiv:?}: max|K_rsh/c_sr - K_full| {worst:.2e}, v_M {:.12} (formula {want:.12}), v_LR {lr:.3e}",
            mix.madelung()
        );
        assert!(worst < 1e-13, "{exxdiv:?}: {worst:e}");
        assert!((mix.madelung() - want).abs() < 1e-10, "{exxdiv:?}");
        if exxdiv == ExxDiv::Ewald {
            // The erf self-term does not vanish at finite omega: v_M is NOT the
            // global hybrid's c_sr v_M (this is physics, not an artifact).
            assert!((lr - 2.0 * omega / std::f64::consts::PI.sqrt()).abs() < 1e-6 * lr);
            assert!((mix.madelung() - r.c_sr * plain.madelung_ewald()).abs() > 1e-3 * lr);
        }
    }
}

#[test]
fn mutants_break_the_energy() {
    let cell = h2_cell(4.0);
    let bs = pyscf_sto3g_h();
    let grid = PeriodicGrid::uniform(&cell, [GRID; 3]).unwrap();
    let mesh = KPointMesh::gamma_centred(&cell, [1, 1, 3]).unwrap();
    for f in ["HSE06", "HYB_GGA_XC_CAM_B3LYP"] {
        let d = run(&cell, &bs, None, &mesh, &grid, f, None).scf.energy;
        for m in [
            KRshMutant::NoAttenuation,
            KRshMutant::OmegaUnits,
            KRshMutant::SwappedRoles,
            KRshMutant::NoMadelung,
            KRshMutant::PrimitiveMadelung,
        ] {
            let e = run(
                &cell,
                &bs,
                Some("cc-pvdz-ri"),
                &mesh,
                &grid,
                f,
                Some(KRsGdfMutation::Rsh(m)),
            )
            .scf
            .energy;
            eprintln!("{f} {m:?}: E - E_dense {:.3e}", e - d);
            assert!(
                (e - d).abs() > 1e-3,
                "{f} {m:?} left the energy anchor green"
            );
        }
    }
}

/// The k-mesh/supercell identity is a statement about the fit construction
/// (the mesh fit IS the supercell Gamma fit), so it holds to round-off, not to
/// the fit error. Non-TRIM 1x1x3 (`-1 != 1 mod 3`) and the TRIM 1x1x2.
#[test]
fn rsgdf_rsh_k_mesh_is_the_supercell() {
    for f in ["HSE06", "HYB_GGA_XC_CAM_B3LYP"] {
        for n in [[1, 1, 2], [1, 1, 3]] {
            let d = supercell_gap(f, n, None);
            eprintln!("{f} {n:?}: E_k - E_super/N {d:.2e}");
            assert!(d.abs() < 1e-8, "{f} {n:?}: {d:e}");
        }
    }
}

#[test]
fn supercell_anchor_sees_every_mutant() {
    use KRshMutant::*;
    for m in [
        NoAttenuation,
        OmegaUnits,
        SwappedRoles,
        NoMadelung,
        PrimitiveMadelung,
    ] {
        let d = supercell_gap(
            "HYB_GGA_XC_CAM_B3LYP",
            [1, 1, 3],
            Some(KRsGdfMutation::Rsh(m)),
        );
        eprintln!("CAM-B3LYP 1x1x3 {m:?}: {d:.2e}");
        assert!(
            d.abs() > 1e-4,
            "{m:?} left the supercell anchor green: {d:e}"
        );
    }
}

/// Per-commit canary of the weekly `rsgdf_rsh_is_the_dense_rsh_to_the_fit_error`
/// (`validation_pbc_krsh_rsgdf.rs`): the RS-GDF vs dense RSH energy gap is the
/// plain-Coulomb fit error, 1x1x1, cc-pVDZ-RI, HSE06 and CAM-B3LYP.
#[test]
fn rsgdf_rsh_is_the_dense_rsh_to_the_fit_error_canary() {
    fit_error_case(
        &[[1, 1, 1]],
        &["cc-pvdz-ri"],
        &["HSE06", "HYB_GGA_XC_CAM_B3LYP"],
    );
}

/// Per-commit canary of the weekly
/// `rsgdf_rsh_matches_pinned_pyscf_to_the_fit_error`: the HSE06 1x1x1 PySCF
/// KRKS pin (the first of the six), to the cc-pVDZ-RI fit error.
#[test]
fn rsgdf_rsh_matches_one_pinned_pyscf_energy() {
    pyscf_pin_case(&PYSCF[..1]);
}

/// Forces and stress of a range-separated RS-GDF SCF: the HF entries and a KS
/// entry with another functional refuse the attenuated J/K (typed); the
/// closed-shell RKS force and stress with the functional's own range
/// separation run on them. The Python / CLI allowlist still refuses the name.
#[test]
fn rsh_rsgdf_derivatives_need_the_rks_entry_with_the_functionals_rsh() {
    use ferric_pbc::kdft::solve_krks_keep_on_grid;
    use ferric_pbc::kgrad::{kpoint_rhf_gradient, KGradConfig};
    use ferric_pbc::kstress::{kpoint_rhf_stress, KStressConfig};
    let cell = h2_cell(4.0);
    let bs = pyscf_sto3g_h();
    let prep = prep_for(&cell, &bs);
    let aux = PreparedBasis::new(cell.mol(), &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let mesh = KPointMesh::gamma_centred(&cell, [1, 1, 2]).unwrap();
    let grid = PeriodicGrid::uniform(&cell, [24; 3]).unwrap();
    let cfg = krks_cfg(&cell, "HSE06", true, None);
    let (r, kept) =
        solve_krks_keep_on_grid(&cell, &prep, Some(&aux), &mesh, &grid, &cfg).expect("hse keep");
    let g = kpoint_rhf_gradient(
        &cell,
        &prep,
        &mesh,
        &cfg.krhf.hcore,
        &kept.hk,
        kept.grad_jk(Some(&aux), &cfg.krhf.rsgdf).unwrap(),
        &r.scf,
        ExxDiv::Ewald,
        &KGradConfig::default(),
    );
    let e = g.expect_err("force on an RSH RS-GDF SCF must be refused");
    assert!(e.to_string().contains("range-separated"), "{e}");
    let s = kpoint_rhf_stress(
        &cell,
        &prep,
        &mesh,
        &cfg.krhf.hcore,
        &kept.hk,
        kept.stress_jk(Some(&aux), &cfg.krhf.rsgdf).unwrap(),
        &r.scf,
        ExxDiv::Ewald,
        &KStressConfig::default(),
    );
    let e = s.expect_err("stress on an RSH RS-GDF SCF must be refused");
    assert!(e.to_string().contains("range-separated"), "{e}");
    // The KS entry with a NON-range-separated name (so the functional guard
    // does not fire) must still be refused on the RSH-carrying J/K.
    let kgrad = ferric_pbc::kxcgrad::kpoint_rks_gradient(
        &cell,
        &prep,
        &mesh,
        &cfg.krhf.hcore,
        &kept.hk,
        kept.grad_jk(Some(&aux), &cfg.krhf.rsgdf).unwrap(),
        &r.scf,
        "PBE0",
        &ferric_pbc::kxcgrad::KRksGradConfig {
            exxdiv: Some(ExxDiv::Ewald),
            ..Default::default()
        },
    );
    let e = kgrad.expect_err("KS force on an RSH RS-GDF SCF must be refused");
    assert!(e.to_string().contains("range-separated"), "{e}");
    // The KS entry WITH the range-separated name runs on the RS-GDF RSH J/K
    // (the Fock guard passes: it is the SCF's own source); the forces sum to
    // zero. Its FD validation is validation_krsh_rsgdf_force.rs.
    let kgrad = ferric_pbc::kxcgrad::kpoint_rks_gradient(
        &cell,
        &prep,
        &mesh,
        &cfg.krhf.hcore,
        &kept.hk,
        kept.grad_jk(Some(&aux), &cfg.krhf.rsgdf).unwrap(),
        &r.scf,
        "HSE06",
        &ferric_pbc::kxcgrad::KRksGradConfig {
            exxdiv: Some(ExxDiv::Ewald),
            grid: ferric_pbc::kxcgrad::KXcGrid::Uniform([24; 3]),
            ..Default::default()
        },
    );
    let g = kgrad.expect("HSE06 KS force on its own RS-GDF RSH J/K");
    assert!(g.hf.net_force < 1e-10, "{:e}", g.hf.net_force);
    assert!(g.hf.fock_mismatch < 1e-8, "{:e}", g.hf.fock_mismatch);
    let st = ferric_pbc::kxcstress::kpoint_rks_stress(
        &cell,
        &prep,
        &mesh,
        &cfg.krhf.hcore,
        &kept.hk,
        kept.stress_jk(Some(&aux), &cfg.krhf.rsgdf).unwrap(),
        &r.scf,
        "HSE06",
        &ferric_pbc::kxcstress::KRksStressConfig {
            exxdiv: Some(ExxDiv::Ewald),
            grid: ferric_pbc::kxcgrad::KXcGrid::Uniform([24; 3]),
            ..Default::default()
        },
    );
    assert!(st.is_ok(), "{:?}", st.err());
    // Another range separation than the build's is refused.
    let e = ferric_pbc::kxcgrad::kpoint_rks_gradient(
        &cell,
        &prep,
        &mesh,
        &cfg.krhf.hcore,
        &kept.hk,
        kept.grad_jk(Some(&aux), &cfg.krhf.rsgdf).unwrap(),
        &r.scf,
        "HYB_GGA_XC_CAM_B3LYP",
        &ferric_pbc::kxcgrad::KRksGradConfig {
            exxdiv: Some(ExxDiv::Ewald),
            ..Default::default()
        },
    )
    .expect_err("CAM-B3LYP on HSE06 J/K must be refused")
    .to_string();
    assert!(e.contains("functional's"), "{e}");
    // The binding derivative gate takes the RSH force and stress on RS-GDF
    // for every SCF family (#498).
    use ferric_pbc::kxcgrad::{check_deriv_functional, DerivSpin};
    let cam = "HYB_GGA_XC_CAM_B3LYP";
    check_deriv_functional(DerivSpin::Restricted, true, "HSE06").unwrap();
    check_deriv_functional(DerivSpin::Unrestricted, true, cam).unwrap();
    check_deriv_functional(DerivSpin::RestrictedOpen, true, cam).unwrap();
}

/// k-point ROKS takes the attenuated RS-GDF K per spin as well
/// (mgoldey/ferric#354): in the closed-shell limit it must be the RKS energy
/// on the same fit.
#[test]
fn roks_rsgdf_closed_shell_rsh_is_krks() {
    use ferric_pbc::kroks::{solve_kroks_on_grid, KRoksConfig};
    let cell = h2_cell(4.0);
    let bs = pyscf_sto3g_h();
    let prep = prep_for(&cell, &bs);
    let aux = PreparedBasis::new(cell.mol(), &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let mesh = KPointMesh::gamma_centred(&cell, [1, 1, 1]).unwrap();
    let grid = PeriodicGrid::uniform(&cell, [24; 3]).unwrap();
    let k = krks_cfg(&cell, "HSE06", true, None);
    let mut roks = KRoksConfig::new(&cell, ExxDiv::Ewald, "HSE06");
    roks.rohf.hcore = k.krhf.hcore;
    roks.rohf.scf.energy_conv = 1e-12;
    roks.rohf.scf.grad_conv = 1e-9;
    roks.rohf.jk = KJkKind::RsGdf;
    roks.rohf.rsgdf = k.krhf.rsgdf;
    let ro = solve_kroks_on_grid(&cell, &prep, Some(&aux), &mesh, &grid, &roks)
        .unwrap()
        .scf
        .energy;
    let r = run(&cell, &bs, Some("cc-pvdz-ri"), &mesh, &grid, "HSE06", None)
        .scf
        .energy;
    eprintln!("closed-shell HSE06 RS-GDF: KROKS - KRKS {:.2e}", ro - r);
    assert!((ro - r).abs() < 1e-11, "{:.2e}", ro - r);
}

#[test]
fn rsgdf_hse06_matches_the_python_binding_pin() {
    let cell = h2_cell(4.0);
    let bs = pyscf_sto3g_h();
    let grid = PeriodicGrid::build(&cell, &PeriodicGridConfig::with_size(40, 110)).unwrap();
    let mesh = KPointMesh::gamma_centred(&cell, [1, 1, 2]).unwrap();
    let e = run(&cell, &bs, Some("cc-pvdz-ri"), &mesh, &grid, "HSE06", None)
        .scf
        .energy;
    eprintln!("HSE06 RS-GDF 1x1x2 atom grid: {e:.10} (pin {HSE06_RSGDF_112_ATOM_GRID})");
    assert!((e - HSE06_RSGDF_112_ATOM_GRID).abs() < 1e-7, "{e:.10}");
}

/// Regression: a q class whose K vectors lie inside the Gaussian cut but whose
/// kernel weights are all under the absolute lindep used to abort the build
/// ("every metric eigenvalue ... <= lindep"). H2 cubic a = 5.3 Bohr, Gamma,
/// HSE06 (w = 0.11 -> cut 1.2 Bohr^-1, smallest |G| = 1.185, weights ~1e-12).
/// The class contributes below the build precision, so the fit must run, agree
/// with the dense RSH to the plain-Coulomb fit scale, and give a sane HSE06 vs
/// PBE0 gap.
#[test]
fn hse06_rsgdf_runs_when_every_erf_weight_is_below_lindep() {
    let cell = h2_cell(5.3);
    let bs = pyscf_sto3g_h();
    let grid = PeriodicGrid::uniform(&cell, [GRID; 3]).unwrap();
    let mesh = KPointMesh::gamma_centred(&cell, [1, 1, 1]).unwrap();
    let aux = "cc-pvdz-ri";
    let hse_d = run(&cell, &bs, None, &mesh, &grid, "HSE06", None)
        .scf
        .energy;
    let hse_g = run(&cell, &bs, Some(aux), &mesh, &grid, "HSE06", None)
        .scf
        .energy;
    let pbe_d = run(&cell, &bs, None, &mesh, &grid, "PBE0", None).scf.energy;
    let pbe_g = run(&cell, &bs, Some(aux), &mesh, &grid, "PBE0", None)
        .scf
        .energy;
    let (scale, d) = (pbe_g - pbe_d, hse_g - hse_d);
    eprintln!(
        "a = 5.3: HSE06 gdf {hse_g:.10} dense {hse_d:.10} (dE {d:.3e}); PBE0 gdf {pbe_g:.10} dense {pbe_d:.10} (plain fit {scale:.3e})"
    );
    assert!(scale.abs() > 1e-7);
    assert!(
        (d - scale).abs() < FIT_BAR_RATIO * scale.abs(),
        "{d:e} vs {scale:e}"
    );
    // HSE06 and PBE0 differ by the short-range-screened exchange: finite, and
    // the same on both J/K up to the fit error.
    let gap = hse_g - pbe_g;
    assert!(gap.abs() > 1e-4 && gap.abs() < 0.1, "HSE06 - PBE0 = {gap}");
    assert!((gap - (hse_d - pbe_d)).abs() < 3.0 * scale.abs());
}

/// The other side of the criterion: when the kernel weights are NOT negligible
/// but the aux basis cannot represent them (one extremely tight s function,
/// |X(K)|^2 ~ 1e-12), the metric is genuinely singular and the typed error
/// stays.
#[test]
fn singular_attenuated_metric_with_real_weights_is_still_an_error() {
    let cell = h2_cell(4.0);
    let prep = prep_for(&cell, &single_s_h(ANCHOR_ALPHA));
    let mesh = KPointMesh::gamma_centred(&cell, [1, 1, 1]).unwrap();
    let hk = periodic_hcore_kpts(
        &cell,
        &prep,
        &mesh,
        &PeriodicHcoreConfig::with_omega(OMEGA_HCORE),
    )
    .unwrap();
    let site = SiteBasis::new(&[[0.3, 0.2, 0.1, 2.0e8]], 0).unwrap();
    let e = KRsGdf::build_rsh(
        &cell,
        &prep,
        &site.prep,
        &mesh,
        &hk.s,
        &gdf_cfg(ExxDiv::Ewald, None),
        rsh(0.5, HSE_LIKE),
    )
    .expect_err("a singular attenuated metric with real weights must be an error")
    .to_string();
    assert!(e.contains("not negligible"), "{e}");
}
