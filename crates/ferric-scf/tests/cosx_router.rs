//! Task D1: the COSX Hölder precision router classifies and counts; it never
//! changes a bit of K. CPU only; butane/def2-SVP (water for the SCF check);
//! the density is a converged RI-JK RHF density, the SAME density on every
//! arm. Two rayon workers throughout.
//!
//! The router's decision is `Route::F32` iff the maximum over screening groups
//! of `est_q * fmax_q` (the product that decides the screen) is below
//! `tau = fp64_multiplier * cosx_screen_thresh`; the flop weight of a unit is
//! `Md3c1e::pair_flops_per_point * points`. Every kernel evaluation is still
//! f64 in this build, so K must be bit-identical for EVERY multiplier.
use ferric_core::basis::bundled;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_dft::grid::AtomicGridConfig;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::md3c1e::{Md3c1e, PrimPairSum};
use ferric_integrals::operator::Operator;
use ferric_scf::cosx_k::{
    CosxBackend, CosxConfig, CosxK, CosxTimings, COSX_DEFAULT_FP64_MULTIPLIER,
    COSX_DEFAULT_SCREEN_THRESH, COSX_SUB_BATCH_POINTS,
};
use ferric_scf::fock::KBuilder;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ndarray::Array2;

const JK: &str = "def2-universal-jkfit";
const WATER: &str = "3\nwater\nO 0.0 0.0 0.1173\nH 0.0 0.7572 -0.4692\nH 0.0 -0.7572 -0.4692\n";

struct Setup {
    mol: Molecule,
    prep: PreparedBasis,
    d: Array2<f64>,
}

fn pool() -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .expect("pool")
}

fn setup(mol: Molecule, basis: &str) -> Setup {
    let bs = bundled(basis).expect("basis");
    let prep = PreparedBasis::new(&mol, &bs).expect("prep");
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).expect("schwarz");
    let cfg = RhfConfig {
        df_j_aux: Some(JK.to_string()),
        df_k_aux: Some(JK.to_string()),
        energy_conv: 1e-10,
        density_conv: 1e-8,
        ..Default::default()
    };
    let r = solve_rhf(
        &ParallelContext::default(),
        &mol,
        &prep,
        Operator::coulomb(),
        &bounds,
        &cfg,
    )
    .expect("rhf");
    assert!(r.converged);
    Setup {
        mol,
        prep,
        d: r.density_total,
    }
}

fn water() -> Setup {
    setup(Molecule::parse_xyz(WATER, 0, 1).unwrap(), "def2-svp")
}

fn butane() -> Setup {
    let path = format!(
        "{}/../../testdata/molecules/alkane_4.xyz",
        env!("CARGO_MANIFEST_DIR")
    );
    setup(Molecule::load_xyz(&path).expect("alkane_4"), "def2-svp")
}

fn cfg(multiplier: f64) -> CosxConfig {
    CosxConfig {
        grid: AtomicGridConfig {
            n_radial: 30,
            n_angular: 110,
            ..Default::default()
        },
        fp64_multiplier: multiplier,
        ..CosxConfig::flat_reference()
    }
}

fn build(s: &Setup, c: CosxConfig) -> (Array2<f64>, CosxTimings) {
    let n = s.prep.nbasis();
    pool().install(|| {
        let ctx = ParallelContext::default();
        let mut kb = CosxK::new(&ctx, &s.mol, &s.prep, c, usize::MAX).expect("CosxK::new");
        let mut k = Array2::zeros((n, n));
        kb.build(&s.d, &mut k).expect("build");
        (k, *kb.last_timings())
    })
}

fn bits(k: &Array2<f64>) -> Vec<u64> {
    k.iter().map(|v| v.to_bits()).collect()
}

#[test]
fn multiplier_zero_routes_nothing_and_k_is_bit_identical_to_the_unrouted_build() {
    let s = butane();
    // The library default is the router off (asserted); `k_zero` is that arm.
    assert_eq!(CosxConfig::default().fp64_multiplier, 0.0);
    let (k_zero, t_zero) = build(&s, cfg(0.0));
    assert_eq!(t_zero.route_f32_units, 0);
    assert_eq!(t_zero.route_f32_flop_share, 0.0);
    assert_eq!(t_zero.fp64_tau, 0.0);
    // The golden: the screen with the router ENABLED keeps exactly the same
    // pairs and produces the same bits (the router only classifies).
    let (k_on, t_on) = build(&s, cfg(1e5));
    assert_eq!(bits(&k_on), bits(&k_zero));
    assert_eq!(t_on.pairs_kept, t_zero.pairs_kept);
    assert_eq!(t_on.pairs_kept_geom, t_zero.pairs_kept_geom);
    assert_eq!(t_on.bound_evals, t_zero.bound_evals);
    assert_eq!(t_on.route_units, t_zero.route_units);
}

/// The same bit-identity on the PRODUCTION default grid (`CosxConfig::default`,
/// not the 30x110 test grid): router on (1e5) vs off, butane/def2-SVP.
#[test]
#[ignore = "slow tier: two production-grid butane K builds"]
fn k_is_bit_identical_with_the_router_on_at_the_production_default_grid() {
    let s = butane();
    let at = |m: f64| {
        build(
            &s,
            CosxConfig {
                fp64_multiplier: m,
                ..CosxConfig::default()
            },
        )
    };
    let (k0, t0) = at(0.0);
    let (k1, t1) = at(1e5);
    assert_eq!(bits(&k1), bits(&k0));
    assert_eq!(t1.pairs_kept, t0.pairs_kept);
    assert!(t1.route_f32_units > 0 && t1.route_f32_units < t1.route_units);
}

#[test]
#[ignore = "slow tier: 8 butane K builds (one per decade); the seed point is checked in the fast tier"]
fn routed_share_is_zero_at_multiplier_zero_one_at_infinity_and_monotone_between() {
    let s = butane();
    let (k0, t0) = build(&s, cfg(0.0));
    assert_eq!(t0.route_f32_flop_share, 0.0);
    let mut prev = (0.0_f64, 0usize);
    let mut shares = vec![0.0];
    for m in [1e2, 1e3, 1e4, 1e5, 1e6, f64::INFINITY] {
        let (k, t) = build(&s, cfg(m));
        assert_eq!(bits(&k), bits(&k0), "multiplier {m}: K bits changed");
        assert_eq!(t.fp64_tau, m * COSX_DEFAULT_SCREEN_THRESH, "tau at {m}");
        assert!(
            t.route_f32_flop_share >= prev.0 && t.route_f32_units >= prev.1,
            "multiplier {m}: share {} / units {} fell below {:?}",
            t.route_f32_flop_share,
            t.route_f32_units,
            prev
        );
        prev = (t.route_f32_flop_share, t.route_f32_units);
        shares.push(t.route_f32_flop_share);
    }
    assert_eq!(*shares.last().unwrap(), 1.0, "infinite tau routes all");
    // Measured strictly increasing on every decade here (0, .24, .44, .69,
    // .88, .99, 1.0): the router is live at every multiplier.
    assert!(shares.windows(2).all(|w| w[0] < w[1]), "{shares:?}");
    let t_inf_units = build(&s, cfg(f64::INFINITY)).1;
    assert_eq!(t_inf_units.route_f32_units, t_inf_units.route_units);
    // Non-inert at the seed: shares = [0, 1e2, 1e3, 1e4, 1e5, 1e6, inf], so
    // index 4 is the 1e5 seed (tau 1e-2; the spike measured 0.65-0.93 on
    // ethane/benzene/octane). Loose band around the measured ~0.88.
    let mid = shares[4];
    assert!(0.80 < mid && mid < 0.95, "share at 1e5: {mid}");
    eprintln!("butane/def2-SVP router shares (0, 1e2..1e6, inf): {shares:?}");
}

#[test]
fn the_router_never_routes_a_unit_the_screen_dropped() {
    let s = butane();
    let (_, t) = build(&s, cfg(1e5));
    assert!(t.route_units > 0 && t.route_f32_units <= t.route_units);
    // `pairs_kept` is the kept-unit count scaled by each sub-batch's points;
    // `route_points` sums the same point counts over the routed units, so a
    // unit is routed iff kept and the two agree exactly. `route_units` is
    // unscaled (>= 1 point per unit, <= COSX_SUB_BATCH_POINTS).
    assert_eq!(t.route_points, t.pairs_kept);
    assert!(t.route_units <= t.pairs_kept);
    assert!(t.pairs_kept <= t.route_units * COSX_SUB_BATCH_POINTS);
    assert!(t.pairs_kept < t.pairs_total, "the screen dropped something");
    // Unscreened: no screen, no routing decisions, and the multiplier is refused.
    let (_, tu) = build(
        &s,
        CosxConfig {
            screen_thresh: None,
            ..cfg(0.0)
        },
    );
    assert_eq!(tu.route_f32_units, 0);
    assert_eq!(tu.fp64_tau, 0.0);
    // The seed cannot go inert: shares over (0, 1e2..1e6, inf) measured 0,
    // .24, .44, .69, .88, .99, 1.0 on butane/def2-SVP; 1e5 is index 4.
    let share = t.route_f32_flop_share;
    assert!(0.80 < share && share < 0.95, "share at 1e5: {share}");
    // The share is flop-weighted (`route_f32_flops / route_flops`), not the
    // unit share: they differ by more than 0.01 here.
    let weighted = t.route_f32_flops as f64 / t.route_flops as f64;
    assert_eq!(weighted, share);
    let unweighted = t.route_f32_units as f64 / t.route_units as f64;
    assert!(
        (weighted - unweighted).abs() > 0.01,
        "{weighted} vs {unweighted}"
    );
}

/// The flop weight: with everything kept (screen threshold so low that almost
/// no unit is dropped) the weighted denominator is within 1% of
/// `sum(pair_flops_table) x npts`, recomputed from the table (an
/// unweighted-share mutant, `unit_flops = points`, failed this assertion by 2 orders of
/// magnitude when performed once, on its butane form). Water, so it is cheap.
#[test]
fn the_flop_weight_is_the_pair_flops_table_times_points() {
    let s = water();
    let kern = Md3c1e::new(&s.prep).expect("kernel");
    let table_sum: u64 = kern.pair_flops_table().iter().sum();
    let (_, all) = build(
        &s,
        CosxConfig {
            screen_thresh: Some(1e-30),
            ..cfg(1e30)
        },
    );
    // Not every unit need survive even at 1e-30 (AO values of tight cores
    // underflow to exactly 0 far away; butane kept 68 560 800 of 68 607 000
    // pairs), so the denominator is bracketed: at most the whole table x npts.
    let npts = pool().install(|| {
        let ctx = ParallelContext::default();
        CosxK::new(&ctx, &s.mol, &s.prep, cfg(0.0), usize::MAX)
            .expect("CosxK")
            .npts()
    }) as u64;
    assert!(all.pairs_kept as f64 > 0.99 * all.pairs_total as f64);
    let full = table_sum * npts;
    assert!(
        all.route_flops <= full && all.route_flops as f64 > 0.99 * full as f64,
        "{} vs {full}",
        all.route_flops
    );
    assert_eq!(all.route_f32_flops, all.route_flops, "tau routes all");
}

/// The `f32_route` test seam: with it set, exactly the router's F32 units are
/// computed by the f32 block (counted), K changes (so the path is live) and
/// the F64 units keep their bits (every unit routed F64 at multiplier 0 is the
/// unrouted build); with it unset nothing is computed in f32; set without a
/// multiplier it is refused.
#[test]
fn f32_route_computes_exactly_the_routed_units() {
    let s = water();
    let (k_f64, t_f64) = build(&s, cfg(1e5));
    assert_eq!((t_f64.f32_blocks, t_f64.f32_fallbacks), (0, 0));
    for sum in PrimPairSum::ALL {
        let (k, t) = build(
            &s,
            CosxConfig {
                f32_route: Some(sum),
                ..cfg(1e5)
            },
        );
        assert_eq!(t.route_f32_units, t_f64.route_f32_units, "{sum:?}");
        assert_eq!(t.f32_blocks + t.f32_fallbacks, t.route_f32_units, "{sum:?}");
        assert_eq!(
            t.f32_fallbacks, 0,
            "{sum:?}: no f32-range fallback on water/SVP"
        );
        assert!(t.f32_blocks > 0, "{sum:?}");
        assert_ne!(bits(&k), bits(&k_f64), "{sum:?}: the f32 path must be live");
    }
    let ctx = ParallelContext::default();
    let e = CosxK::new(
        &ctx,
        &s.mol,
        &s.prep,
        CosxConfig {
            f32_route: Some(PrimPairSum::F64),
            ..cfg(0.0)
        },
        usize::MAX,
    )
    .unwrap_err();
    assert!(format!("{e}").contains("f32_route needs fp64_multiplier > 0"));
}

#[test]
fn pair_flops_sum_to_the_kernel_total() {
    let s = butane();
    let kern = Md3c1e::new(&s.prep).expect("kernel");
    let total = kern.flops_per_point();
    let table = kern.pair_flops_table();
    let nsh = kern.nshells();
    assert_eq!(table.len(), nsh * (nsh + 1) / 2);
    let mut sum = 0u64;
    for s1 in 0..nsh {
        for s2 in 0..=s1 {
            let f = kern.pair_flops_per_point(s1, s2);
            assert_eq!(f, table[s1 * (s1 + 1) / 2 + s2]);
            sum += f;
        }
    }
    assert_eq!(sum as f64, total.r_tensor + total.contraction);
    // d-d is dearer than s-s (an ordering the count must respect).
    let by_l: Vec<usize> = (0..nsh).map(|i| kern.shell_l(i)).collect();
    let hi = (0..nsh).max_by_key(|&i| by_l[i]).unwrap();
    let lo = (0..nsh).min_by_key(|&i| by_l[i]).unwrap();
    assert!(kern.pair_flops_per_point(hi, hi) > kern.pair_flops_per_point(lo, lo));
}

#[test]
fn a_router_that_cannot_act_is_refused_by_the_library() {
    let s = butane();
    let ctx = ParallelContext::default();
    let refused = |c: CosxConfig, needle: &str| {
        let e = CosxK::new(&ctx, &s.mol, &s.prep, c, usize::MAX).unwrap_err();
        let m = format!("{e}");
        assert!(m.contains(needle), "{m}");
    };
    refused(
        CosxConfig {
            screen_thresh: None,
            ..cfg(1e5)
        },
        "screen_thresh > 0",
    );
    refused(
        CosxConfig {
            backend: CosxBackend::CosxA,
            screen_thresh: None,
            ..cfg(1e5)
        },
        "md3c1e backend only",
    );
    refused(cfg(-1.0), "must be >= 0");
    refused(cfg(f64::NAN), "must be >= 0");
    assert_eq!(COSX_DEFAULT_FP64_MULTIPLIER, 1e5);
}

/// Water RIJCOSX through the SCF driver: the converged energy is bit-identical
/// with the router off and on (SCF K builds and the final pass go through the
/// same code).
#[test]
fn rijcosx_scf_energy_is_bit_identical_with_the_router_on() {
    let mol = Molecule::parse_xyz(WATER, 0, 1).unwrap();
    let bs = bundled("def2-svp").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
    let run = |m: f64| {
        let c = RhfConfig {
            k_builder: Some("cosx".to_string()),
            cosx: CosxConfig {
                fp64_multiplier: m,
                ..CosxConfig::default()
            },
            df_j_aux: Some(JK.to_string()),
            df_k_aux: Some(String::new()),
            energy_conv: 1e-9,
            density_conv: 1e-8,
            ..Default::default()
        };
        pool().install(|| {
            solve_rhf(
                &ParallelContext::default(),
                &mol,
                &prep,
                Operator::coulomb(),
                &bounds,
                &c,
            )
            .expect("rhf")
        })
    };
    let (a, b) = (run(0.0), run(1e5));
    assert!(a.converged && b.converged);
    assert_eq!(a.energy.to_bits(), b.energy.to_bits());
    assert_eq!(a.iterations, b.iterations);
}

/// Water/def2-SVP with the deterministic test density used by the two tests below.
fn water_with_test_density() -> Setup {
    let s = water();
    let n = s.prep.nbasis();
    let mut d = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            d[(i, j)] = 0.1 / (1.0 + (i as f64 - j as f64).abs()) * ((i + 2 * j) as f64).cos();
        }
    }
    let d = &d + &d.t();
    Setup { d, ..s }
}

/// In-process, host independent: the seam-off K is run-to-run deterministic,
/// is not changed by the router classifying (multiplier 1e5, seam off), and the
/// comparison can see a change (negative control: the f32 seam moves the bits).
#[test]
fn seam_off_k_is_deterministic_unrouted_equal_and_the_comparison_has_teeth() {
    let s = water_with_test_density();
    let (k0, _) = build(&s, cfg(0.0));
    let (k0_again, _) = build(&s, cfg(0.0));
    assert_eq!(
        bits(&k0_again),
        bits(&k0),
        "seam-off K is not deterministic"
    );
    let (k_routed, _) = build(&s, cfg(1e5));
    assert_eq!(bits(&k_routed), bits(&k0), "classifying changed a bit of K");
    let (k_f32, _) = build(
        &s,
        CosxConfig {
            f32_route: Some(PrimPairSum::F32),
            ..cfg(1e5)
        },
    );
    assert_ne!(
        bits(&k_f32),
        bits(&k0),
        "the comparison cannot see an f32 route"
    );
}

/// Is this the host class the golden below was measured on: x86-64 with AVX2 and
/// FMA but no AVX-512? OpenBLAS picks different GEMM kernels (different summation
/// order) on AVX-512 hosts, so the hash is only comparable on this class.
fn golden_host_class() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx2")
            && std::is_x86_feature_detected!("fma")
            && !std::is_x86_feature_detected!("avx512f")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// GOLDEN: the seam-off K of water/def2-SVP (30x110 grid, two workers, the
/// deterministic density above) hashes to the value measured at the D1 commit
/// 52fe8d88, i.e. D2 changed no f64 bit of K. FNV-1a over the K bits. Measured
/// on an AVX2+FMA host without AVX-512 (the kernel's FMA path and that host's
/// BLAS kernels); on any other host the hash is not comparable, so the test
/// returns early there and the in-process test above carries the check.
#[test]
fn seam_off_k_bits_match_the_d1_golden() {
    const GOLDEN: u64 = 0xb8c5_7af4_e256_e48a;
    let s = water_with_test_density();
    if !golden_host_class() || !Md3c1e::new(&s.prep).unwrap().uses_fma() {
        eprintln!("skipping: not the AVX2+FMA, no-AVX-512 host class the golden was taken on");
        return;
    }
    let (k, _) = build(&s, cfg(0.0));
    let h = k.iter().fold(0xcbf2_9ce4_8422_2325_u64, |h, v| {
        (h ^ v.to_bits()).wrapping_mul(0x100_0000_01b3)
    });
    assert_eq!(h, GOLDEN, "K bits changed vs the D1 commit: {h:016x}");
}
