//! Boys-screened χ₀ tests (C7).
//!
//! Validates that the per-orbital screened B-tile representation reproduces
//! the dense PDEP-RPA energy at `thresh = 0` (algebraic equivalence) and
//! converges to the dense answer at production thresholds with substantial
//! pair reduction.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_rpa::config::{Chi0Sparsity, Eigensolver, QuadratureConfig, QuadratureScheme};
use ferric_rpa::{run_pdep_rpa, screen, PdepRpaConfig};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::ScfResult;
use ferric_scf::screening::SchwarzBounds;

fn setup(
    xyz: &str,
    obs_name: &str,
    dfbs_name: &str,
) -> (Molecule, PreparedBasis, PreparedBasis, Operator, ScfResult) {
    let ctx = ParallelContext::default();
    let mol = Molecule::load_xyz(xyz).unwrap();
    let obs_bs = basis::bundled(obs_name).unwrap();
    let dfbs_bs = basis::bundled(dfbs_name).unwrap();
    let op = Operator::coulomb();
    let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
    let dfbs = PreparedBasis::new(&mol, &dfbs_bs).unwrap();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default()).unwrap();
    (mol, obs, dfbs, op, rhf)
}

fn base_cfg() -> PdepRpaConfig {
    PdepRpaConfig {
        quadrature: QuadratureConfig {
            scheme: QuadratureScheme::GaussLegendre,
            n_points: 40,
            u0: 0.5,
        },
        frozen_core: 0,
        trunc_thresh: 0.0,
        eigensolver_conv_thresh: 1e-10,
        ..Default::default()
    }
}

#[test]
fn h2o_cc_pvdz_screened_equivalence_thresh_zero() {
    // At thresh = 0 no aux rows are dropped; the screened tile representation
    // should match the dense path to high precision.
    let (mol, obs, dfbs, op, rhf) =
        setup("../../testdata/molecules/water.xyz", "cc-pvdz", "cc-pvdz-ri");

    let cfg_dense = base_cfg();
    let r_dense = run_pdep_rpa(&mol, &obs, &dfbs, op, &rhf, &cfg_dense).unwrap();

    let mut cfg_screen = base_cfg();
    cfg_screen.chi0_sparsity = Chi0Sparsity::BoysScreened { thresh: 0.0, dist_cutoff: f64::INFINITY };
    let r_scr = run_pdep_rpa(&mol, &obs, &dfbs, op, &rhf, &cfg_screen).unwrap();

    let diff = (r_scr.e_rpa - r_dense.e_rpa).abs();
    println!(
        "H2O/cc-pVDZ thresh=0  dense={:.10}  screened={:.10}  diff={:.2e}",
        r_dense.e_rpa, r_scr.e_rpa, diff
    );
    // At thresh=0 the screened tile representation retains every aux row, so it
    // is *algebraically* equivalent to dense — but the two paths are not
    // bit-identical: the dense solve seeds block-Lanczos from the full identity
    // block, while the screened solve seeds it from Boys-localized tile-column
    // sums (build_boys_screened_seed). Different-but-spanning seeds converge to
    // the same dielectric eigenspace, yet accumulate GEMM reductions in a
    // different order, so the final RPA energy differs at the ~1e-8 finite-
    // precision floor. This diff is fully deterministic (bit-stable run-to-run
    // and independent of BLAS thread count) — it is genuine algorithm-ordering
    // drift, not nondeterminism. The 1e-9 originally asserted here was below
    // that floor and never passed (fails identically on main). 5e-8 sits ~6×
    // above the observed 8.2e-9 floor while still catching a real screening
    // regression (which corrupts the energy at the mHa scale, not 1e-8) by many
    // orders of magnitude.
    assert!(
        diff < 5e-8,
        "screened-vs-dense diff at thresh=0 = {:.2e}; expected <5e-8",
        diff
    );
}

/// Same equivalence check as `h2o_cc_pvdz_screened_equivalence_thresh_zero`,
/// but forcing `Eigensolver::Davidson` so the parallelized
/// `sternheimer_sparse::dielectric_matrix_screened` (the projected-dielectric
/// / Davidson matvec form) is exercised directly — the default eigensolver is
/// Lanczos, which only exercises the sibling `dielectric_apply_screened`
/// (block-Lanczos matvec form). Both functions were parallelized over `i_loc`
/// via a rayon + `ferric_scf::reduce::grouped_deterministic_sum` region
/// (sternheimer-sparse-parallelize); this test is the correctness gap-closer
/// for the Davidson-specific accumulate path (`out += rhs_i @ rhs_i.T`),
/// which the existing suite never independently covered.
#[test]
fn h2o_cc_pvdz_screened_davidson_equivalence_thresh_zero() {
    let (mol, obs, dfbs, op, rhf) =
        setup("../../testdata/molecules/water.xyz", "cc-pvdz", "cc-pvdz-ri");

    let mut cfg_dense = base_cfg();
    cfg_dense.eigensolver = Eigensolver::Davidson;
    let r_dense = run_pdep_rpa(&mol, &obs, &dfbs, op, &rhf, &cfg_dense).unwrap();

    let mut cfg_screen = base_cfg();
    cfg_screen.eigensolver = Eigensolver::Davidson;
    cfg_screen.chi0_sparsity = Chi0Sparsity::BoysScreened { thresh: 0.0, dist_cutoff: f64::INFINITY };
    let r_scr = run_pdep_rpa(&mol, &obs, &dfbs, op, &rhf, &cfg_screen).unwrap();

    let diff = (r_scr.e_rpa - r_dense.e_rpa).abs();
    println!(
        "H2O/cc-pVDZ Davidson thresh=0  dense={:.10}  screened={:.10}  diff={:.2e}",
        r_dense.e_rpa, r_scr.e_rpa, diff
    );
    // Same rationale/tolerance as the Lanczos-path sibling test: algebraically
    // equivalent at thresh=0, but different seed/accumulation order versus
    // dense means the two are not bit-identical, only close to the ~1e-8
    // finite-precision floor.
    assert!(
        diff < 5e-8,
        "Davidson screened-vs-dense diff at thresh=0 = {:.2e}; expected <5e-8",
        diff
    );
}

#[test]
fn h2o_cc_pvdz_screened_production_thresh() {
    let (mol, obs, dfbs, op, rhf) =
        setup("../../testdata/molecules/water.xyz", "cc-pvdz", "cc-pvdz-ri");

    let r_dense = run_pdep_rpa(&mol, &obs, &dfbs, op, &rhf, &base_cfg()).unwrap();

    let thresh = 1e-6;
    let mut cfg = base_cfg();
    cfg.chi0_sparsity = Chi0Sparsity::BoysScreened { thresh, dist_cutoff: f64::INFINITY };
    let r_scr = run_pdep_rpa(&mol, &obs, &dfbs, op, &rhf, &cfg).unwrap();

    // Diagnostic: pair retention.
    let (sb, _) =
        screen::build_screened_bov_boys(&mol, &obs, &dfbs, op, &rhf, 0, thresh, f64::INFINITY).unwrap();
    let total_possible = sb.n_occ_loc * sb.naux;
    println!(
        "H2O/cc-pVDZ thresh={:.0e}  retained {}/{} ({:.1}%)  ΔE={:.2e}",
        thresh,
        sb.total_retained,
        total_possible,
        100.0 * sb.total_retained as f64 / total_possible as f64,
        (r_scr.e_rpa - r_dense.e_rpa).abs(),
    );

    let diff = (r_scr.e_rpa - r_dense.e_rpa).abs();
    assert!(
        diff < 1e-7,
        "screened-vs-dense diff at thresh={:.0e} = {:.2e}; expected <1e-7",
        thresh, diff
    );
}

/// G6 equivalence-at-large-radius regression test (item 4 of the brief).
///
/// A distance cutoff larger than the whole molecule can prune nothing: every
/// aux shell and OBS pair sits within `r_ref` of every Boys centroid, so the
/// `min(1, r_ref/R)` envelope is ≡ 1 and `build_screened_bov`'s output — the
/// retained aux lists AND the dressed tile values themselves — must be
/// **bit-for-bit identical** to the `dist_cutoff = ∞` build. This is the direct,
/// noise-free proof that the distance filter is a strict pre-filter that
/// composes with — never replaces — the exact metric: at a radius where it
/// cannot fire, `screen.rs`'s output is byte-identical to the pre-G6 path.
///
/// We assert on `ScreenedBov` (screen.rs's actual product), NOT on the
/// downstream RPA energy: the energy is built by an *iterative* block-Lanczos
/// eigensolve whose "best-effort" Ritz pairs are not run-to-run deterministic at
/// the 1e-13 level when it does not fully converge (see the solver-honesty
/// reliability convention — an unconverged Lanczos warns and returns best-effort
/// pairs). Byte-identical tiles feeding a nondeterministic-at-1e-9 downstream
/// solver is exactly the invariant the brief's "algebraically equivalent to
/// today's screen.rs output" demands; the downstream energy noise is not part of
/// screen.rs and must not be smuggled into this test's tolerance.
#[test]
fn h2o_cc_pvdz_dist_cutoff_large_radius_equivalence() {
    let (mol, obs, dfbs, op, rhf) =
        setup("../../testdata/molecules/water.xyz", "cc-pvdz", "cc-pvdz-ri");

    let thresh = 1e-5;
    // Water spans < 4 Bohr; 1e6 Bohr is astronomically larger than any pair.
    let r_ref = 1.0e6;

    let (sb_inf, _) =
        screen::build_screened_bov_boys(&mol, &obs, &dfbs, op, &rhf, 0, thresh, f64::INFINITY)
            .unwrap();
    let (sb_big, _) =
        screen::build_screened_bov_boys(&mol, &obs, &dfbs, op, &rhf, 0, thresh, r_ref).unwrap();

    // Retained sets must be bit-identical.
    assert_eq!(
        sb_inf.total_retained, sb_big.total_retained,
        "large-radius total_retained {} != infinite-radius {}",
        sb_big.total_retained, sb_inf.total_retained
    );
    assert_eq!(
        sb_inf.p_lists, sb_big.p_lists,
        "large-radius per-orbital aux lists differ from infinite-radius"
    );

    // The dressed tiles — the actual screen.rs output that feeds χ₀ — must be
    // element-wise bit-for-bit identical (same integrals, same contraction
    // order; the envelope only gates *which* shells are evaluated, and here it
    // gates none, so nothing about the arithmetic changes).
    assert_eq!(
        sb_inf.tiles.len(),
        sb_big.tiles.len(),
        "tile count differs between ∞ and large-radius builds"
    );
    let mut max_tile_diff = 0.0f64;
    for (i, (t_inf, t_big)) in sb_inf.tiles.iter().zip(sb_big.tiles.iter()).enumerate() {
        assert_eq!(
            t_inf.dim(),
            t_big.dim(),
            "tile {i} shape differs between ∞ and large-radius builds"
        );
        for (a, b) in t_inf.iter().zip(t_big.iter()) {
            max_tile_diff = max_tile_diff.max((a - b).abs());
        }
    }
    println!(
        "H2O/cc-pVDZ dist_cutoff ∞ vs {r_ref:.0e} Bohr @thresh={thresh:.0e}: \
         retained {} (both), max|Δtile|={max_tile_diff:.2e}",
        sb_inf.total_retained
    );
    assert_eq!(
        max_tile_diff, 0.0,
        "large-radius screened tiles differ from infinite-radius by {max_tile_diff:.2e}; \
         expected bit-for-bit identical (the envelope must be a strict no-op here)"
    );
}

/// G6 PROBE (run with `FERRIC_G6_PROBE=1 --nocapture`): quantify how tight the
/// Cauchy-Schwarz bound is on n-hexane vs the exact `p_ii` locality. Builds the
/// screened representation once and lets screen.rs dump per-aux-shell (R, exact,
/// bound). Cheap: no RPA energy solve. This is the diagnostic that decides
/// whether the distance envelope can prune at all.
#[test]
#[ignore] // diagnostic: run explicitly with FERRIC_G6_PROBE=1 --nocapture
fn n_hexane_g6_probe() {
    let (mol, obs, dfbs, op, rhf) =
        setup("../../testdata/molecules/scaling/n-hexane.xyz", "cc-pvdz", "cc-pvdz-ri");
    let thresh = 1e-4;
    let frozen = 6;
    let (sb, _) =
        screen::build_screened_bov_boys(&mol, &obs, &dfbs, op, &rhf, frozen, thresh, 10.0).unwrap();
    let poss = sb.n_occ_loc * sb.naux;
    println!(
        "n-hexane probe: total_retained={}/{} ({:.1}%)",
        sb.total_retained,
        poss,
        100.0 * sb.total_retained as f64 / poss as f64
    );
}

/// G6 probe on a LONG chain (n-hexadecane, ~43.6 Bohr) — the geometry where a
/// Boys centroid at one end is genuinely >30 Bohr from aux functions at the
/// other. This is the definitive test of whether distance screening EVER bites:
/// if the exact |p_ii| still keeps ~100% even here, the `1/r` Coulomb tail is
/// simply not local at any molecular scale we care about.
#[test]
#[ignore] // diagnostic: run explicitly with FERRIC_G6_PROBE=1 --nocapture
fn n_hexadecane_g6_probe() {
    let (mol, obs, dfbs, op, rhf) =
        setup("../../testdata/molecules/scaling/n-hexadecane.xyz", "cc-pvdz", "cc-pvdz-ri");
    let thresh = 1e-4;
    let frozen = 16;
    let (sb, _) =
        screen::build_screened_bov_boys(&mol, &obs, &dfbs, op, &rhf, frozen, thresh, 15.0).unwrap();
    let poss = sb.n_occ_loc * sb.naux;
    println!(
        "n-hexadecane probe: total_retained={}/{} ({:.1}%)",
        sb.total_retained,
        poss,
        100.0 * sb.total_retained as f64 / poss as f64
    );
}

/// G6 probe with a MINIMAL orbital basis (STO-3G) on the long chain. Matt's
/// suggestion: does a compact basis make distance screening bite? STO-3G orbital
/// products are spatially tighter, but the aux `(P|i_loc i_loc)` metric still
/// carries the `1/R` monopole tail of the unit-charge localized density, which
/// is basis-independent. This probe measures the exact-metric retention for the
/// minimal case to confirm the tail — not the orbital extent — is what governs.
#[test]
#[ignore] // diagnostic: run explicitly with FERRIC_G6_PROBE=1 --nocapture
fn n_hexadecane_sto3g_g6_probe() {
    let (mol, obs, dfbs, op, rhf) =
        setup("../../testdata/molecules/scaling/n-hexadecane.xyz", "sto-3g", "cc-pvdz-ri");
    let thresh = 1e-4;
    let frozen = 16;
    let (sb, _) =
        screen::build_screened_bov_boys(&mol, &obs, &dfbs, op, &rhf, frozen, thresh, 15.0).unwrap();
    let poss = sb.n_occ_loc * sb.naux;
    println!(
        "n-hexadecane/STO-3G probe: total_retained={}/{} ({:.1}%)",
        sb.total_retained,
        poss,
        100.0 * sb.total_retained as f64 / poss as f64
    );
}

#[test]
#[ignore] // diagnostic: run explicitly with FERRIC_G6_PROBE=1 --nocapture
fn naphthalene_g6_probe() {
    let (mol, obs, dfbs, op, rhf) =
        setup("../../testdata/molecules/scaling/naphthalene.xyz", "cc-pvdz", "cc-pvdz-ri");
    let thresh = 1e-4;
    let frozen = 10; // 10 carbon 1s cores
    let (sb, _) =
        screen::build_screened_bov_boys(&mol, &obs, &dfbs, op, &rhf, frozen, thresh, 8.0).unwrap();
    let poss = sb.n_occ_loc * sb.naux;
    println!(
        "naphthalene probe: total_retained={}/{} ({:.1}%)",
        sb.total_retained,
        poss,
        100.0 * sb.total_retained as f64 / poss as f64
    );
}

/// G6 MEASURED DEAD-END (see docs/pdep-boys-laplace-scaling.md "G6" section and
/// docs/open-work-triage-2026-07-14-open.md #38): on n-hexane the distance
/// pre-filter prunes NOTHING at any usable radius, so it cannot speed sparse-RPA
/// up. This test pins that measured fact so a future change that *did* start
/// pruning here would flip it and force a re-read of the analysis.
///
/// Root cause, measured (`FERRIC_G6_PROBE=1`): at thresh=1e-4 the *exact*
/// pass-1 metric `|p_ii[P]|` already retains 100% of aux functions
/// (`total_retained == n_occ_loc·naux`) — the diagnosis the brief started from.
/// The farthest aux shell from any Boys centroid is only ~9.3 Bohr away, and at
/// that distance the exact metric is still 2e-2…1.4 (200×…14000× above thresh):
/// the `1/r` Coulomb tail of the unit-charge `i_loc i_loc` density does NOT
/// decay across the molecule's own diameter. Because the distance envelope
/// `min(1,r_ref/R)·CS_bound` is a composable *upper* bound on that exact metric,
/// it can only ever drop a SUBSET of what the exact metric drops — and the exact
/// metric drops nothing. Hence: `retained(r_ref) == retained(∞)` for every
/// r_ref large enough to leave the genuine physics intact, and the energy is
/// unchanged. The only r_ref that would make the *bound* fire (≈2e-4 Bohr, from
/// the probe's `r_ref@thresh_via_bound` column) is far smaller than the molecule
/// and would wrongly drop shells whose exact coupling is O(1) — i.e. it would be
/// physically wrong, not a speedup. So the honest assertion here is EQUIVALENCE,
/// not pruning.
#[test]
#[ignore] // moderately slow: n-hexane/cc-pVDZ SCF + two PDEP builds
fn n_hexane_dist_cutoff_is_measured_noop_at_production_radius() {
    let (mol, obs, dfbs, op, rhf) =
        setup("../../testdata/molecules/scaling/n-hexane.xyz", "cc-pvdz", "cc-pvdz-ri");

    let thresh = 1e-4;
    let frozen = 6; // 6 carbon 1s cores
    let r_ref = 10.0; // Bohr — shorter than the ~14.5 Bohr chain, yet prunes nothing.

    let (sb_inf, _) =
        screen::build_screened_bov_boys(&mol, &obs, &dfbs, op, &rhf, frozen, thresh, f64::INFINITY)
            .unwrap();
    let (sb_cut, _) =
        screen::build_screened_bov_boys(&mol, &obs, &dfbs, op, &rhf, frozen, thresh, r_ref).unwrap();

    let poss = sb_inf.n_occ_loc * sb_inf.naux;
    println!(
        "n-hexane/cc-pVDZ @thresh={thresh:.0e}: retained ∞={}/{} ({:.1}%) cut(r={r_ref})={}",
        sb_inf.total_retained,
        poss,
        100.0 * sb_inf.total_retained as f64 / poss as f64,
        sb_cut.total_retained,
    );

    // MEASURED: the exact metric already keeps everything, so distance pruning is
    // a no-op — retained sets and tiles are bit-identical to the ∞ build.
    assert_eq!(
        sb_cut.total_retained, sb_inf.total_retained,
        "distance cutoff changed n-hexane retention — the measured dead-end (exact \
         metric keeps 100%, envelope can only prune a subset of that) would be broken; \
         re-read docs/pdep-boys-laplace-scaling.md G6"
    );
    assert_eq!(
        sb_cut.p_lists, sb_inf.p_lists,
        "distance cutoff changed n-hexane per-orbital aux lists (expected identical)"
    );

    // And the downstream RPA energy is therefore unchanged (byte-identical tiles).
    let mut cfg_inf = base_cfg();
    cfg_inf.frozen_core = frozen;
    cfg_inf.chi0_sparsity = Chi0Sparsity::BoysScreened { thresh, dist_cutoff: f64::INFINITY };
    let r_inf = run_pdep_rpa(&mol, &obs, &dfbs, op, &rhf, &cfg_inf).unwrap();

    let mut cfg_cut = base_cfg();
    cfg_cut.frozen_core = frozen;
    cfg_cut.chi0_sparsity = Chi0Sparsity::BoysScreened { thresh, dist_cutoff: r_ref };
    let r_cut = run_pdep_rpa(&mol, &obs, &dfbs, op, &rhf, &cfg_cut).unwrap();

    let diff = (r_inf.e_rpa - r_cut.e_rpa).abs();
    println!("n-hexane ΔE(cut vs ∞) = {diff:.2e}");
    assert!(
        diff < 1e-8,
        "distance cutoff changed n-hexane RPA energy by {diff:.2e}; expected ~0 \
         (it prunes nothing at r={r_ref} Bohr, so the energy must be identical)"
    );
}

#[test]
#[ignore] // slow: benzene/cc-pVDZ is the scaling demonstration
fn benzene_cc_pvdz_screened_scaling() {
    let (mol, obs, dfbs, op, rhf) =
        setup("../../testdata/molecules/benzene.xyz", "cc-pvdz", "cc-pvdz-ri");

    use std::time::Instant;

    let mut cfg_dense = base_cfg();
    cfg_dense.frozen_core = 6;
    let t0 = Instant::now();
    let r_dense = run_pdep_rpa(&mol, &obs, &dfbs, op, &rhf, &cfg_dense).unwrap();
    let dt_dense = t0.elapsed().as_secs_f64();

    // C7-tighten: exact (P|i_loc i_loc) density-pair metric. The bound is
    // genuine Cauchy-Schwarz on (P|i a). For benzene, π orbitals span all
    // atoms so retention saturates near 100% below ~5e-3; tightening below
    // this discards no shells. Anything larger discards rapidly. The dial
    // test below sweeps a broader range.
    let thresh = 5e-3;
    let mut cfg_scr = base_cfg();
    cfg_scr.frozen_core = 6;
    cfg_scr.chi0_sparsity = Chi0Sparsity::BoysScreened { thresh, dist_cutoff: f64::INFINITY };
    let t0 = Instant::now();
    let r_scr = run_pdep_rpa(&mol, &obs, &dfbs, op, &rhf, &cfg_scr).unwrap();
    let dt_scr = t0.elapsed().as_secs_f64();

    let (sb, _) =
        screen::build_screened_bov_boys(&mol, &obs, &dfbs, op, &rhf, 6, thresh, f64::INFINITY).unwrap();
    let total_possible = sb.n_occ_loc * sb.naux;
    let reduction = total_possible as f64 / sb.total_retained.max(1) as f64;

    println!(
        "Benzene/cc-pVDZ thresh={:.0e}  retained {}/{} (reduction {:.2}×)  ΔE={:.2e}  dense {:.2}s  screened {:.2}s",
        thresh,
        sb.total_retained,
        total_possible,
        reduction,
        (r_scr.e_rpa - r_dense.e_rpa).abs(),
        dt_dense,
        dt_scr,
    );

    let diff = (r_scr.e_rpa - r_dense.e_rpa).abs();
    assert!(
        diff < 1e-3,
        "screened-vs-dense diff on benzene at thresh={:.0e} = {:.2e}; expected <1e-3",
        thresh, diff
    );
    // Demonstrate non-trivial pair reduction.
    assert!(
        reduction >= 1.1,
        "pair reduction factor {:.2}× too small; expected ≥1.1×",
        reduction
    );
}

/// Accuracy/sparsity dial: sweep thresh on benzene/cc-pVDZ and print the
/// retained-pair fraction plus ΔE at each setting. Informational — checks
/// the screen builds and yields a monotone tradeoff over 3+ decades.
#[test]
#[ignore] // slow
fn benzene_cc_pvdz_thresh_sweep() {
    let (mol, obs, dfbs, op, rhf) =
        setup("../../testdata/molecules/benzene.xyz", "cc-pvdz", "cc-pvdz-ri");

    let mut cfg_dense = base_cfg();
    cfg_dense.frozen_core = 6;
    let r_dense = run_pdep_rpa(&mol, &obs, &dfbs, op, &rhf, &cfg_dense).unwrap();

    println!("Benzene/cc-pVDZ dense e_rpa = {:.10}", r_dense.e_rpa);

    for &thresh in &[1e-1, 5e-2, 1e-2, 5e-3, 1e-3] {
        use std::time::Instant;
        let t_build = Instant::now();
        let (sb, _) =
            screen::build_screened_bov_boys(&mol, &obs, &dfbs, op, &rhf, 6, thresh, f64::INFINITY)
                .unwrap();
        let dt_build = t_build.elapsed().as_secs_f64();

        let mut cfg = base_cfg();
        cfg.frozen_core = 6;
        cfg.chi0_sparsity = Chi0Sparsity::BoysScreened { thresh, dist_cutoff: f64::INFINITY };
        let t_run = Instant::now();
        let r = run_pdep_rpa(&mol, &obs, &dfbs, op, &rhf, &cfg).unwrap();
        let dt_run = t_run.elapsed().as_secs_f64();

        let total = sb.n_occ_loc * sb.naux;
        let frac = 100.0 * sb.total_retained as f64 / total as f64;
        let de = (r.e_rpa - r_dense.e_rpa).abs();
        println!(
            "  thresh={:.0e}  retained {}/{} ({:.1}%)  ΔE={:.2e}  build={:.2}s  run={:.2}s",
            thresh, sb.total_retained, total, frac, de, dt_build, dt_run,
        );
    }
}
