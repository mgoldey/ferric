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
use ferric_integrals::md3c1e::Md3c1e;
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
    // The default config never names the field: that is the pre-change build's
    // behaviour (multiplier 0.0 is its default, asserted).
    assert_eq!(CosxConfig::default().fp64_multiplier, 0.0);
    let (k_default, t_default) = build(
        &s,
        CosxConfig {
            fp64_multiplier: CosxConfig::default().fp64_multiplier,
            ..cfg(0.0)
        },
    );
    let (k_zero, t_zero) = build(&s, cfg(0.0));
    assert_eq!(bits(&k_zero), bits(&k_default));
    assert_eq!(t_zero.route_f32_units, 0);
    assert_eq!(t_zero.route_f32_flop_share, 0.0);
    assert_eq!(t_zero.fp64_tau, 0.0);
    assert_eq!(t_default.pairs_kept, t_zero.pairs_kept);
    // The golden: the screen with the router ENABLED keeps exactly the same
    // pairs and produces the same bits (the router only classifies).
    let (k_on, t_on) = build(&s, cfg(1e5));
    assert_eq!(bits(&k_on), bits(&k_zero));
    assert_eq!(t_on.pairs_kept, t_zero.pairs_kept);
    assert_eq!(t_on.pairs_kept_geom, t_zero.pairs_kept_geom);
    assert_eq!(t_on.bound_evals, t_zero.bound_evals);
    assert_eq!(t_on.route_units, t_zero.route_units);
}

#[test]
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
    // Non-inert in the middle: 1e5 x 1e-7 = tau 1e-2 routes a strict
    // fraction (the spike measured 0.65-0.93 on ethane/benzene/octane).
    let mid = shares[5];
    assert!(0.0 < mid && mid < 1.0, "share at 1e5: {mid}");
    eprintln!("butane/def2-SVP router shares (0, 1e2..1e6, inf): {shares:?}");
}

#[test]
fn the_router_never_routes_a_unit_the_screen_dropped() {
    let s = butane();
    let (_, t) = build(&s, cfg(1e5));
    assert!(t.route_units > 0 && t.route_f32_units <= t.route_units);
    // `pairs_kept` is the kept-unit count scaled by each sub-batch's points
    // (1..=COSX_SUB_BATCH_POINTS), `route_units` the unscaled count: a unit is
    // routed only if kept, so both bracket each other.
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
