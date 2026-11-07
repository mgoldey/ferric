//! The screened, spatially batched XC nuclear gradient
//! (`ferric_dft::gradient::screened`) against the whole-grid gradient it
//! replaced as the production path (`*_dense` / `_batched`, kept unchanged as
//! the reference).
//!
//! # Hypotheses, stated before measuring
//!
//! * Right: with the screen OFF (`screen_thresh = 0`, every shell kept) the
//!   screened path evaluates the same terms over the same points and differs
//!   only by re-association (per-batch partial sums, small GEMMs): agreement
//!   at the ~1e-13 Ha/Bohr rounding floor. With the default screen the dropped
//!   terms are each below 1e-10 in every AO factor, so the gradient moves by
//!   well under 1e-10 Ha/Bohr.
//! * Broken (a mis-gathered weight1 row, a wrong home atom, a sub-density built
//!   from the wrong rows, a lost batch, a dropped Hessian term): the miss is
//!   the size of a gradient term, 1e-4..1e-1 — eight or more orders above the
//!   floor. The two outcomes cannot be confused.
//!
//! Every path the production dispatch can take is covered: closed-shell LDA,
//! GGA (PBE), hybrid GGA (B3LYP), meta-GGA (r2SCAN), and UKS GGA / meta-GGA.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_dft::gradient::{
    xc_gradient_closed_gga_from_density_batched, xc_gradient_closed_lda_from_density_dense,
    xc_gradient_closed_mgga_from_density_dense, xc_gradient_screened_with,
    xc_gradient_uks_from_density_dense, xc_gradient_uks_mgga_from_density_dense, GradScreenConfig,
    XcGradPath,
};
use ferric_dft::grid::AtomicGridConfig;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::solve_uhf;
use ndarray::Array2;

const WATER: &str = "3\nwater\nO 0.000000 0.000000 0.117790\nH 0.000000 0.755453 -0.471161\nH 0.000000 -0.755453 -0.471161\n";
const OH: &str = "2\nOH\nO 0.0 0.0 0.0\nH 0.0 0.0 0.97\n";

/// One converged density plus everything the XC-gradient entry points need.
struct Case {
    mol: Molecule,
    bs: basis::BasisSet,
    prep: PreparedBasis,
    d_a: Array2<f64>,
    d_b: Option<Array2<f64>>,
    xc: String,
    path: XcGradPath,
    grid: AtomicGridConfig,
}

fn converge(xyz: &str, mult: usize, basis_name: &str, xc: &str, path: XcGradPath) -> Case {
    let mol = Molecule::parse_xyz(xyz, 0, mult).unwrap();
    let bs = basis::bundled(basis_name).unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let grid = AtomicGridConfig::production();
    let cfg = RhfConfig {
        xc: Some(xc.into()),
        dft_grid: Some(grid.clone()),
        energy_conv: 1e-9,
        density_conv: 1e-6,
        max_iter: 300,
        ..Default::default()
    };
    let ctx = ParallelContext::default();
    let (d_a, d_b) = if mult == 1 {
        let r = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg).unwrap();
        (r.density_r().clone(), None)
    } else {
        let r = solve_uhf(&ctx, &mol, &prep, &bounds, &cfg).unwrap();
        (
            r.density_alpha.clone(),
            Some(r.density_beta.clone().expect("UKS has a beta density")),
        )
    };
    Case {
        mol,
        bs,
        prep,
        d_a,
        d_b,
        xc: xc.into(),
        path,
        grid,
    }
}

/// The whole-grid reference (the pre-screening production code).
fn dense(c: &Case) -> Array2<f64> {
    let (s2a, off, dims) = (
        c.prep.shell_to_atom(),
        c.prep.shell_offsets(),
        c.prep.shell_dims(),
    );
    let (m, bs, d, x, g) = (&c.mol, &c.bs, &c.d_a, c.xc.as_str(), &c.grid);
    match c.path {
        XcGradPath::ClosedLda => {
            xc_gradient_closed_lda_from_density_dense(m, bs, d, x, g, s2a, off, dims)
        }
        XcGradPath::ClosedGga => {
            xc_gradient_closed_gga_from_density_batched(m, bs, d, x, g, s2a, off, dims, None)
        }
        XcGradPath::ClosedMgga => {
            xc_gradient_closed_mgga_from_density_dense(m, bs, d, x, g, s2a, off, dims)
        }
        XcGradPath::Uks => xc_gradient_uks_from_density_dense(
            m,
            bs,
            d,
            c.d_b.as_ref().unwrap(),
            x,
            g,
            s2a,
            off,
            dims,
        ),
        XcGradPath::UksMgga => xc_gradient_uks_mgga_from_density_dense(
            m,
            bs,
            d,
            c.d_b.as_ref().unwrap(),
            x,
            g,
            s2a,
            off,
            dims,
        ),
    }
    .unwrap()
}

fn screened(c: &Case, cfg: GradScreenConfig) -> Array2<f64> {
    xc_gradient_screened_with(
        &c.mol,
        &c.bs,
        &c.d_a,
        c.d_b.as_ref(),
        &c.xc,
        c.path,
        &c.grid,
        c.prep.shell_to_atom(),
        c.prep.shell_offsets(),
        c.prep.shell_dims(),
        cfg,
    )
    .unwrap()
}

fn max_abs_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    assert_eq!(a.dim(), b.dim());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

fn screen_off() -> GradScreenConfig {
    GradScreenConfig {
        screen_thresh: 0.0,
        ..GradScreenConfig::default()
    }
}

/// Every production dispatch path, on systems small enough for the per-commit
/// tier: closed shells on water / cc-pVDZ (d shells, so the pure-d AO Hessian
/// is exercised), open shells on the OH doublet.
fn cases() -> Vec<Case> {
    vec![
        converge(WATER, 1, "cc-pvdz", "LDA", XcGradPath::ClosedLda),
        converge(WATER, 1, "cc-pvdz", "PBE", XcGradPath::ClosedGga),
        converge(WATER, 1, "cc-pvdz", "B3LYP", XcGradPath::ClosedGga),
        converge(WATER, 1, "cc-pvdz", "r2SCAN", XcGradPath::ClosedMgga),
        converge(OH, 2, "cc-pvdz", "PBE", XcGradPath::Uks),
        converge(OH, 2, "cc-pvdz", "r2SCAN", XcGradPath::UksMgga),
    ]
}

/// EXACTNESS ANCHOR, then the measurement. Screen off: the re-association
/// floor. Default screen: below 1e-10 Ha/Bohr. Plus two vacuity guards: the
/// default screen must actually change the result (else the 1e-10 bar is
/// measuring nothing), and so must the batch size (else the batching is not
/// what is being compared).
#[test]
fn screened_xc_gradient_matches_the_whole_grid_gradient() {
    eprintln!(
        "\n{:<10} {:<8} {:>12} {:>14} {:>14} {:>14}",
        "path", "xc", "max|g|", "off-dense", "default-dense", "default-off"
    );
    for c in cases() {
        let reference = dense(&c);
        let off = screened(&c, screen_off());
        let def = screened(&c, GradScreenConfig::default());
        let gmax = reference.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        let (d_off, d_def, d_scr) = (
            max_abs_diff(&off, &reference),
            max_abs_diff(&def, &reference),
            max_abs_diff(&def, &off),
        );
        eprintln!(
            "{:<10} {:<8} {gmax:>12.3e} {d_off:>14.3e} {d_def:>14.3e} {d_scr:>14.3e}",
            format!("{:?}", c.path),
            c.xc
        );
        assert!(gmax > 1e-3, "{:?} {}: vacuous reference", c.path, c.xc);
        assert!(
            d_off < 1e-12,
            "{:?} {}: screen OFF misses the whole-grid gradient by {d_off:.3e} — \
             far above re-association, so a term is lost or mis-assigned",
            c.path,
            c.xc
        );
        assert!(
            d_def < 1e-10,
            "{:?} {}: the default screen moves the gradient by {d_def:.3e}",
            c.path,
            c.xc
        );
        assert!(
            d_scr > 0.0,
            "{:?} {}: the default screen dropped nothing on cc-pVDZ water/OH — the \
             1e-10 bar above is then not a measurement of screening",
            c.path,
            c.xc
        );
        let wide = screened(
            &c,
            GradScreenConfig {
                max_batch_pts: 100_000_000,
                ..screen_off()
            },
        );
        assert!(
            max_abs_diff(&wide, &off) > 0.0 && max_abs_diff(&wide, &reference) < 1e-12,
            "{:?} {}: one whole-grid batch must re-associate (≠ 128-pt batches) yet \
             stay at the floor",
            c.path,
            c.xc
        );
    }
}

/// The gradient's group partition and reduction order are functions of the grid
/// alone, so the result is bit-identical across rayon pool sizes — including a
/// one-thread pool, which takes the same group path.
#[test]
fn screened_xc_gradient_is_bit_identical_across_thread_counts() {
    for c in [
        converge(WATER, 1, "cc-pvdz", "PBE", XcGradPath::ClosedGga),
        converge(OH, 2, "cc-pvdz", "r2SCAN", XcGradPath::UksMgga),
    ] {
        let run = |threads: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| screened(&c, GradScreenConfig::default()))
        };
        let (g1, g3) = (run(1), run(3));
        for (a, b) in g1.iter().zip(g3.iter()) {
            assert_eq!(a.to_bits(), b.to_bits(), "{:?}: {a:e} vs {b:e}", c.path);
        }
    }
}

/// Timing + agreement on a production-size system. `#[ignore]`: run with
/// `FERRIC_XCGRAD_BENCH=<xyz>:<basis>:<xc>:<mult>[:skip-dense]`, e.g.
/// `../../testdata/molecules/sulfamethoxazole.xyz:def2-svp:PBE:1`.
#[test]
#[ignore = "benchmark; needs FERRIC_XCGRAD_BENCH"]
fn bench_screened_vs_dense_xc_gradient() {
    let spec = std::env::var("FERRIC_XCGRAD_BENCH").expect("FERRIC_XCGRAD_BENCH");
    let parts: Vec<&str> = spec.split(':').collect();
    let xyz = std::fs::read_to_string(parts[0]).unwrap();
    let (basis_name, xc) = (parts[1], parts[2]);
    let mult: usize = parts[3].parse().unwrap();
    let skip_dense = parts.get(4) == Some(&"skip-dense");
    let fam = ferric_dft::libxc::xc_def_from_name(xc).unwrap();
    let is_mgga = fam
        .funcs
        .iter()
        .any(|f| matches!(f.family(), ferric_dft::libxc::FunctionalFamily::MetaGga));
    let is_lda = fam
        .funcs
        .iter()
        .all(|f| matches!(f.family(), ferric_dft::libxc::FunctionalFamily::Lda));
    let path = match (mult == 1, is_mgga, is_lda) {
        (true, true, _) => XcGradPath::ClosedMgga,
        (true, false, true) => XcGradPath::ClosedLda,
        (true, false, false) => XcGradPath::ClosedGga,
        (false, true, _) => XcGradPath::UksMgga,
        (false, false, _) => XcGradPath::Uks,
    };
    let t0 = std::time::Instant::now();
    let c = converge(&xyz, mult, basis_name, xc, path);
    eprintln!(
        "BENCH scf {:.2} s  nbf {}  path {:?}",
        t0.elapsed().as_secs_f64(),
        c.prep.nbasis(),
        path
    );
    let t0 = std::time::Instant::now();
    let new = screened(&c, GradScreenConfig::default());
    eprintln!("BENCH screened {:.3} s", t0.elapsed().as_secs_f64());
    let t0 = std::time::Instant::now();
    let off = screened(&c, screen_off());
    eprintln!("BENCH screen-off {:.3} s", t0.elapsed().as_secs_f64());
    eprintln!("BENCH |default - off| {:.3e}", max_abs_diff(&new, &off));
    if !skip_dense {
        let t0 = std::time::Instant::now();
        let old = dense(&c);
        eprintln!("BENCH dense {:.3} s", t0.elapsed().as_secs_f64());
        eprintln!(
            "BENCH |screened - dense| {:.3e}  |off - dense| {:.3e}  max|g| {:.3e}",
            max_abs_diff(&new, &old),
            max_abs_diff(&off, &old),
            old.iter().fold(0.0_f64, |m, v| m.max(v.abs()))
        );
    }
}
