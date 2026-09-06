//! Exactness anchors + mutation arms for INTEGRAL-DIRECT amplitude-threshold
//! LinLCCD (`ferric_cc::linlccd_amplitude::amplitude_linlccd_direct`,
//! DriversOnly + Hh tiers — the Full/pp tier is scoped by the Phase-1 PSD
//! measurement, `scripts/queue/proto_linlccd_pp_psd.py`, and hard-errors
//! here until wired).
//!
//! Protocol order (CLAUDE.md Experimental Protocol): trivial-limit anchors
//! FIRST — the direct path at trivial maps must land on (i) the canonical
//! spin-orbital `linlccd` (independent formulation, shared RI integrals
//! only) and (ii) the existing global-B `amplitude_linlccd` (same
//! formulation, different integral/transform code — the gap is the
//! reassociation floor, TIGHT here because LinLCCD is Hylleraas-protected:
//! solver/operator error enters quadratically, unlike dRPA's first-order
//! sensitivity). The occ-occ Gram gets its own value-level anchor + arm.
//! No sweep claims are made by this file.

use ferric_cc::linlccd::{linlccd, LadderVariant};
use ferric_cc::linlccd_amplitude::{
    amplitude_linlccd, amplitude_linlccd_direct, AmplitudeLinLccdConfig,
};
use ferric_cc::CcConfig;
use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::lmp2_amplitude::{build_vvhv, localized_spaces};
use ferric_mp2::lmp2_direct::{assemble_boo_direct, DirectConfig, OoGram};
use ferric_mp2::rimp2::metric_inverse_sqrt;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

struct Setup {
    mol: Molecule,
    obs: PreparedBasis,
    obs_bs: basis::BasisSet,
    dfbs: PreparedBasis,
    rhf: ferric_scf::result::ScfResult,
}

fn setup(xyz: &str) -> Setup {
    let mol = Molecule::load_xyz(&format!(
        "{}/../../testdata/molecules/{xyz}",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap();
    let obs_bs = basis::bundled("6-31g").unwrap();
    let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let rhf = solve_rhf(
        &ferric_core::parallel::ParallelContext::default(),
        &mol,
        &obs,
        op,
        &bounds,
        &RhfConfig { energy_conv: 1e-10, ..Default::default() },
    )
    .unwrap();
    Setup { mol, obs, obs_bs, dfbs, rhf }
}

/// Trivial maps: every locality knob at its no-op limit.
fn trivial_maps() -> DirectConfig {
    DirectConfig {
        aux_radius_bohr: 1e6,
        virt_radius_bohr: None,
        ao_tail: 0.0,
        ..Default::default()
    }
}

fn canonical(su: &Setup, variant: LadderVariant, fc: usize) -> f64 {
    linlccd(
        &su.mol,
        &su.obs,
        &su.dfbs,
        Operator::coulomb(),
        &su.rhf,
        &CcConfig { frozen_core: fc, energy_conv: 1e-11, max_iter: 200, ..Default::default() },
        variant,
    )
    .unwrap()
    .correlation_energy
}

/// Build the direct-path OoGram at given knobs, water-scale helper.
fn oo_direct(su: &Setup, fc: usize, ao_tail: f64, gate: Option<f64>, eps: f64) -> (OoGram, usize) {
    let vvhv = build_vvhv(&su.mol, &su.obs, &su.obs_bs, &su.rhf).unwrap();
    let spaces = localized_spaces(&su.mol, &su.obs, &su.rhf, fc, &vvhv).unwrap();
    let dcfg = DirectConfig { ao_tail, ..trivial_maps() };
    let oo = assemble_boo_direct(
        &su.obs, &su.dfbs, Operator::coulomb(), &spaces, eps, gate, &dcfg,
    )
    .unwrap();
    (oo, spaces.no)
}

/// Global-path (ik|jl): full AO eri3_tensor + transform_3center_oo +
/// global whitening — the object the direct pass must reproduce.
fn oo_global(su: &Setup, fc: usize) -> (ndarray::Array2<f64>, usize) {
    let vvhv = build_vvhv(&su.mol, &su.obs, &su.obs_bs, &su.rhf).unwrap();
    let spaces = localized_spaces(&su.mol, &su.obs, &su.rhf, fc, &vvhv).unwrap();
    let no = spaces.no;
    let op = Operator::coulomb();
    let eri3_ao = ferric_integrals::threeindex::eri3_tensor(op, &su.obs, &su.dfbs).unwrap();
    let naux = su.dfbs.nbasis();
    let boo = ferric_mp2::mo_transform::transform_3center_oo(&eri3_ao, &spaces.c_locc)
        .into_shape_with_order((naux, no * no))
        .unwrap();
    let v2c = ferric_integrals::threeindex::coulomb_metric_2c(op, &su.dfbs).unwrap();
    let vis = metric_inverse_sqrt(&v2c, op).unwrap();
    let boo_t = vis.dot(&boo);
    (boo_t.t().dot(&boo_t), no)
}

/// VALUE-LEVEL ANCHOR: the direct occ-occ Gram at trivial knobs equals the
/// global path's (ik|jl) entry by entry (full aux rows + global whitening
/// — same object, different integral route; reassociation floor).
#[test]
fn oo_gram_trivial_matches_global_object() {
    let su = setup("water.xyz");
    let (oo, no) = oo_direct(&su, 1, 0.0, None, 0.0);
    let (g_ref, no_g) = oo_global(&su, 1);
    assert_eq!(no, no_g);
    let mut dmax = 0.0f64;
    for i in 0..no {
        for k in 0..no {
            for j in 0..no {
                for l in 0..no {
                    let d = (oo.coeff(i, k, j, l) - g_ref[(i * no + k, j * no + l)]).abs();
                    dmax = dmax.max(d);
                }
            }
        }
    }
    eprintln!("OO-GRAM ANCHOR water: max|direct - global| = {dmax:.3e} (ncols {})", oo.ncols());
    assert_eq!(oo.ncols(), no * (no + 1) / 2, "trivial gate must keep every column");
    assert!(dmax < 1e-10, "occ-occ direct Gram anchor FAILED: {dmax:.3e}");
}

/// MUTATION ARMS on the occ-occ pass — each map, gutted, must move the
/// Gram loudly (a test never seen to fail is an assumption).
#[test]
fn oo_gram_gutted_maps_are_loud() {
    let su = setup("water.xyz");
    let (base, no) = oo_direct(&su, 1, 0.0, None, 0.0);
    // (a) AO supports gutted: |C| >= 0.5 shells only
    let (mut_a, _) = oo_direct(&su, 1, 0.5, None, 0.0);
    let mut da = 0.0f64;
    // (b) pair gate gutted: absurd calibration drops every off-diagonal
    // column -> those coefficients collapse to exactly 0.0
    let (mut_g, _) = oo_direct(&su, 1, 0.0, Some(1e-30), 1e-3);
    let mut off_diag_base = 0.0f64;
    let mut off_diag_mut = 0.0f64;
    for i in 0..no {
        for k in 0..no {
            for j in 0..no {
                for l in 0..no {
                    da = da.max((mut_a.coeff(i, k, j, l) - base.coeff(i, k, j, l)).abs());
                    if i != k {
                        off_diag_base = off_diag_base.max(base.coeff(i, k, j, l).abs());
                        off_diag_mut = off_diag_mut.max(mut_g.coeff(i, k, j, l).abs());
                    }
                }
            }
        }
    }
    eprintln!(
        "OO-GRAM MUTATIONS water: ao_tail=0.5 max|dG|={da:.3e}; gate gutted: off-diag max {off_diag_base:.3e} -> {off_diag_mut:.3e} (cols {} -> {})",
        base.ncols(),
        mut_g.ncols()
    );
    assert!(da > 1e-3, "AO-support gutting not detected: {da:.3e}");
    assert!(
        off_diag_base > 1e-3 && off_diag_mut == 0.0,
        "gate gutting not detected ({off_diag_base:.3e} -> {off_diag_mut:.3e})"
    );
}

/// TRIVIAL-LIMIT ANCHOR: water ε=0, DriversOnly + Hh — the direct path vs
/// the canonical spin-orbital solver (independent formulation) AND the
/// existing global-B path (reassociation floor; Hylleraas-protected, so
/// the bar is 1e-9, tighter than dRPA's justification needed).
#[test]
fn eps_zero_trivial_maps_matches_canonical_and_global() {
    let su = setup("water.xyz");
    let cfg = AmplitudeLinLccdConfig { eps: 0.0, frozen_core: 1, ..Default::default() };
    for variant in [LadderVariant::DriversOnly, LadderVariant::Hh] {
        let r_glob = amplitude_linlccd(
            &su.mol, &su.obs, &su.obs_bs, &su.dfbs, Operator::coulomb(), &su.rhf, &cfg, variant,
        )
        .unwrap();
        let (r_dir, stats) = amplitude_linlccd_direct(
            &su.mol, &su.obs, &su.obs_bs, &su.dfbs, Operator::coulomb(), &su.rhf, &cfg,
            &trivial_maps(), variant,
        )
        .unwrap();
        let e_can = canonical(&su, variant, 1);
        let de_can = r_dir.e_corr - e_can;
        let de_glob = r_dir.e_corr - r_glob.e_corr;
        eprintln!(
            "ANCHOR water direct {variant:?}: E={:.12} |vs canonical|={:.3e} |vs global path|={:.3e} (cg {}, strips {}x{})",
            r_dir.e_corr,
            de_can.abs(),
            de_glob.abs(),
            r_dir.cg_iterations,
            stats.strip_rows_max,
            stats.strip_cols_max
        );
        assert!(r_dir.cg_converged);
        assert!(de_can.abs() < 5e-9, "{variant:?} canonical anchor FAILED: {de_can:+.3e}");
        assert!(de_glob.abs() < 1e-9, "{variant:?} global-path anchor FAILED: {de_glob:+.3e}");
        // trivial maps must actually be trivial
        assert_eq!(stats.strip_rows_max, su.dfbs.nbasis());
    }
}

/// FINITE-ε ANCHOR: alkane_4 ε=1e-3 Hh, trivial maps — direct vs global-B
/// path on the same mask (Hylleraas-protected; the measured gap is the
/// reassociation floor, printed).
#[test]
fn finite_eps_trivial_maps_matches_global_path() {
    let su = setup("alkane_4.xyz");
    let cfg = AmplitudeLinLccdConfig { eps: 1e-3, frozen_core: 4, ..Default::default() };
    let r_glob = amplitude_linlccd(
        &su.mol, &su.obs, &su.obs_bs, &su.dfbs, Operator::coulomb(), &su.rhf, &cfg,
        LadderVariant::Hh,
    )
    .unwrap();
    let (r_dir, _) = amplitude_linlccd_direct(
        &su.mol, &su.obs, &su.obs_bs, &su.dfbs, Operator::coulomb(), &su.rhf, &cfg,
        &trivial_maps(), LadderVariant::Hh,
    )
    .unwrap();
    let dd = (r_dir.e_corr - r_glob.e_corr).abs();
    eprintln!(
        "DIRECT finite-eps C4 Hh: E={:.10} global={:.10} |diff|={dd:.3e} keep {:.4} vs {:.4}",
        r_dir.e_corr, r_glob.e_corr, r_dir.keep_fraction, r_glob.keep_fraction
    );
    assert!(r_dir.cg_converged);
    assert!(dd < 1e-8, "finite-eps direct vs global FAILED: {dd:.3e}");
}

/// PRODUCTION-MAP SUB-DOMINANCE: alkane_8 Hh ε=1e-3 with the calibrated
/// production maps + pair gate — the locality-map error must stay
/// sub-dominant to the ε truncation. The ε-truncation scale is measured as
/// |E(1e-3) − E(1e-4)| at trivial maps (the C8 spin-orbital canonical and
/// the ε=0 solve are unaffordable in CI; the 1e-3→1e-4 delta is a lower
/// bound on the 1e-3 truncation, so passing this bar is conservative).
#[test]
fn production_maps_error_is_subdominant_to_eps_truncation() {
    let su = setup("alkane_8.xyz");
    let base = AmplitudeLinLccdConfig { eps: 1e-3, frozen_core: 8, ..Default::default() };
    let (r_eps, _) = amplitude_linlccd_direct(
        &su.mol, &su.obs, &su.obs_bs, &su.dfbs, Operator::coulomb(), &su.rhf, &base,
        &trivial_maps(), LadderVariant::Hh,
    )
    .unwrap();
    let (r_eps4, _) = amplitude_linlccd_direct(
        &su.mol, &su.obs, &su.obs_bs, &su.dfbs, Operator::coulomb(), &su.rhf,
        &AmplitudeLinLccdConfig { eps: 1e-4, ..base.clone() },
        &trivial_maps(), LadderVariant::Hh,
    )
    .unwrap();
    let prod = DirectConfig {
        aux_radius_bohr: 10.0,
        virt_radius_bohr: Some(12.0),
        ao_tail: 1e-3,
        ..Default::default()
    };
    let (r_prod, stats) = amplitude_linlccd_direct(
        &su.mol, &su.obs, &su.obs_bs, &su.dfbs, Operator::coulomb(), &su.rhf,
        &AmplitudeLinLccdConfig { pair_gate_cal: Some(0.7), ..base.clone() },
        &prod, LadderVariant::Hh,
    )
    .unwrap();
    let trunc_err = (r_eps.e_corr - r_eps4.e_corr).abs();
    let map_err = (r_prod.e_corr - r_eps.e_corr).abs();
    eprintln!(
        "C8 Hh production maps: eps-trunc(1e-3 vs 1e-4) {trunc_err:.3e}, map+gate err {map_err:.3e} (strips rows {:.0}/{} cols {:.0}/{}, keep {:.4})",
        stats.strip_rows_mean,
        stats.strip_rows_max,
        stats.strip_cols_mean,
        stats.strip_cols_max,
        r_prod.keep_fraction
    );
    assert!(map_err > 0.0, "maps changed nothing at production radii — vacuous?");
    assert!(
        map_err < trunc_err,
        "locality-map error ({map_err:.3e}) dominates the eps truncation ({trunc_err:.3e})"
    );
}

/// FULL-TIER TRIVIAL ANCHOR: water ε=0 — the fitted-pp direct path at the
/// trivial domain (full aux ⇒ the fit equals the licensed whitened Gram
/// up to reassociation) vs the canonical spin-orbital Full solver and the
/// global-B Full path. The finite-radius PSD/energy behavior of the fit
/// is the Phase-1 prototype's measured record (proto_linlccd_pp_psd.py),
/// not re-derived here.
#[test]
fn full_tier_trivial_maps_matches_canonical_and_global() {
    let su = setup("water.xyz");
    let cfg = AmplitudeLinLccdConfig { eps: 0.0, frozen_core: 1, ..Default::default() };
    let r_glob = amplitude_linlccd(
        &su.mol, &su.obs, &su.obs_bs, &su.dfbs, Operator::coulomb(), &su.rhf, &cfg,
        LadderVariant::Full,
    )
    .unwrap();
    let (r_dir, _) = amplitude_linlccd_direct(
        &su.mol, &su.obs, &su.obs_bs, &su.dfbs, Operator::coulomb(), &su.rhf, &cfg,
        &trivial_maps(), LadderVariant::Full,
    )
    .unwrap();
    let e_can = canonical(&su, LadderVariant::Full, 1);
    let de_can = r_dir.e_corr - e_can;
    let de_glob = r_dir.e_corr - r_glob.e_corr;
    eprintln!(
        "ANCHOR water direct Full: E={:.12} |vs canonical|={:.3e} |vs global path|={:.3e} (cg {})",
        r_dir.e_corr,
        de_can.abs(),
        de_glob.abs(),
        r_dir.cg_iterations
    );
    assert!(r_dir.cg_converged);
    assert!(de_can.abs() < 5e-9, "Full canonical anchor FAILED: {de_can:+.3e}");
    assert!(de_glob.abs() < 1e-9, "Full global-path anchor FAILED: {de_glob:+.3e}");
}

/// FULL-TIER FINITE-ε ANCHOR: alkane_4 ε=1e-3 trivial maps — fitted pp at
/// the trivial domain vs the global-B Full path on the same mask.
#[test]
fn full_tier_finite_eps_trivial_maps_matches_global_path() {
    let su = setup("alkane_4.xyz");
    let cfg = AmplitudeLinLccdConfig { eps: 1e-3, frozen_core: 4, ..Default::default() };
    let r_glob = amplitude_linlccd(
        &su.mol, &su.obs, &su.obs_bs, &su.dfbs, Operator::coulomb(), &su.rhf, &cfg,
        LadderVariant::Full,
    )
    .unwrap();
    let (r_dir, _) = amplitude_linlccd_direct(
        &su.mol, &su.obs, &su.obs_bs, &su.dfbs, Operator::coulomb(), &su.rhf, &cfg,
        &trivial_maps(), LadderVariant::Full,
    )
    .unwrap();
    let dd = (r_dir.e_corr - r_glob.e_corr).abs();
    eprintln!(
        "DIRECT finite-eps C4 Full: E={:.10} global={:.10} |diff|={dd:.3e}",
        r_dir.e_corr, r_glob.e_corr
    );
    assert!(r_dir.cg_converged);
    assert!(dd < 1e-8, "Full finite-eps direct vs global FAILED: {dd:.3e}");
}

/// pp-FIT MUTATION ARM (value level): shrinking the fit domain must change
/// the per-pair pp factors loudly — otherwise the domain map is dead code
/// on this path.
#[test]
fn pp_fitted_domain_gutted_is_loud() {
    use ferric_mp2::lmp2_direct::{assemble_pp_fitted_direct, assemble_ragged_direct_local};
    let su = setup("water.xyz");
    let vvhv = build_vvhv(&su.mol, &su.obs, &su.obs_bs, &su.rhf).unwrap();
    let spaces = localized_spaces(&su.mol, &su.obs, &su.rhf, 1, &vvhv).unwrap();
    let op = Operator::coulomb();
    let (rg, _, _) = assemble_ragged_direct_local(
        &su.mol, &su.obs, &su.dfbs, op, &spaces, 1e-3, 1.0, None, &trivial_maps(),
    )
    .unwrap();
    let base = assemble_pp_fitted_direct(
        &su.mol, &su.obs, &su.dfbs, op, &spaces, &rg, &trivial_maps(),
    )
    .unwrap();
    let gutted = assemble_pp_fitted_direct(
        &su.mol, &su.obs, &su.dfbs, op, &spaces, &rg,
        &DirectConfig { aux_radius_bohr: 1.0, ..trivial_maps() },
    )
    .unwrap();
    let mut dmax = 0.0f64;
    for (b, g) in base.iter().zip(&gutted) {
        let mb = b.a_rows.dot(&b.gb);
        let mg = g.a_rows.dot(&g.gb);
        for (x, y) in mb.iter().zip(mg.iter()) {
            dmax = dmax.max((x - y).abs());
        }
    }
    eprintln!("PP-FIT MUTATION water: aux_radius 1e6 -> 1.0 max|dm|={dmax:.3e}");
    assert!(dmax > 1e-3, "pp fit domain gutting not detected: {dmax:.3e}");
}
