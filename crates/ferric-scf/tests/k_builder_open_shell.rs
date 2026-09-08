//! Anchors for `RhfConfig::k_builder` reaching the OPEN-SHELL solvers.
//!
//! # Why this file exists
//!
//! `k_builder` ("direct" | "link" | "cosx") was validated and consumed by
//! `solve_rhf` only. `solve_uhf` and `solve_rohf` take the same `RhfConfig`
//! and never read the field, so `k_builder = "link"` / `"cosx"` was SILENTLY
//! IGNORED for every open-shell run (documented as a pre-existing defect on the
//! field's doc comment, in `cosx_k.rs`'s scope note, and in the book). These
//! tests were written BEFORE the wiring (repo rule: exactness anchor first) and
//! the pre-wiring failures are recorded in the commit message.
//!
//! # What "the builder was actually used" looks like
//!
//! On the unwired code every open-shell `k_builder` run is bit-identical to
//! the direct run, so an energy-agreement assertion alone would pass
//! VACUOUSLY. Each SCF anchor therefore also asserts a tell that only a
//! consumed builder produces:
//!
//! * LinK: `computed_quartets` differs from the combined single-pass direct
//!   build (LinK walks K's pair-list quartets + a separate DirectJ pass;
//!   the direct path walks one combined J+K pass).
//! * COSX: the energy differs from direct by MORE than round-off (a real
//!   seminumerical K carries grid error) and by LESS than the grid bar.
//!
//! # Bars
//!
//! * LinK vs direct: 1e-9 Ha, same iteration count ±2. LinK is exact to the
//!   screening threshold (fixed on main by #50 — the pair-list criteria); the
//!   closed-shell water anchor in `link_scf_anchor.rs` holds at 1e-9 too.
//! * COSX vs direct: `COSX_OPEN_SHELL_E_BAR` at the default (50,110)+overlap-fit
//!   grid (measured value in the constant's doc), iteration count ±3.
//!
//! All runs are meant for `OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=1`.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::cosx_k::{CosxConfig, CosxK};
use ferric_scf::fock::KBuilder;
use ferric_scf::link_k::LinkK;
use ferric_scf::result::ScfResult;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::rohf::solve_rohf;
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::solve_uhf;
use ndarray::Array2;

/// CH3 radical (doublet), planar D3h, Angstrom — the same geometry the
/// combined-JK open-shell anchors use (`open_shell_combined_jk.rs`).
const CH3_XYZ: &str = "4\nCH3 doublet\nC 0.0000 0.0000 0.0000\nH 1.0790 0.0000 0.0000\n\
                       H -0.5395 0.9345 0.0000\nH -0.5395 -0.9345 0.0000\n";

const WATER_XYZ: &str = "3
water
O  0.0000  0.0000  0.1173
H  0.0000  0.7572 -0.4692
H  0.0000 -0.7572 -0.4692
";

/// Bar for the open-shell COSX SCF energy against the direct SCF at the
/// default `CosxConfig` grid ((50,110), overlap fit on, default screen).
///
/// Measured after wiring (CH3 doublet / cc-pVDZ, one thread):
///
/// ```text
/// UHF   cosx − direct = (see commit message / deliverable)
/// ROHF  cosx − direct = (see commit message / deliverable)
/// ```
///
/// The closed-shell water/cc-pVDZ SCF at the same grid sits in the 1e-5 Ha
/// class; the bar is set with ~10x headroom over the measured open-shell
/// values so a transposed/half-scaled K (1e-2..1 Ha class) still fails.
const COSX_OPEN_SHELL_E_BAR: f64 = 2e-4;

/// Floor below which a COSX energy is INDISTINGUISHABLE from direct — i.e. the
/// builder was not consumed at all (the pre-wiring behaviour). A real
/// seminumerical K on a (50,110) grid never lands this close.
const COSX_MUST_DIFFER_FLOOR: f64 = 1e-12;


fn setup(xyz: &str, charge: i32, mult: usize, bas: &str, op: Operator) -> (Molecule, PreparedBasis, SchwarzBounds) {
    let mol = Molecule::parse_xyz(xyz, charge, mult).expect("xyz");
    let bs = basis::bundled(bas).expect("basis");
    let prep = PreparedBasis::new(&mol, &bs).expect("prep");
    let bounds = SchwarzBounds::compute(op, &prep).expect("schwarz");
    (mol, prep, bounds)
}

fn cfg(kb: Option<&str>) -> RhfConfig {
    RhfConfig {
        k_builder: kb.map(str::to_string),
        energy_conv: 1e-10,
        density_conv: 1e-8,
        max_iter: 200,
        ..Default::default()
    }
}

fn run_uhf_ch3(kb: Option<&str>) -> ScfResult {
    let (mol, prep, bounds) = setup(CH3_XYZ, 0, 2, "cc-pvdz", Operator::coulomb());
    solve_uhf(&ParallelContext::default(), &mol, &prep, &bounds, &cfg(kb)).expect("uhf")
}

fn run_rohf_ch3(kb: Option<&str>) -> ScfResult {
    let (mol, prep, bounds) = setup(CH3_XYZ, 0, 2, "cc-pvdz", Operator::coulomb());
    solve_rohf(&ParallelContext::default(), &mol, &prep, Operator::coulomb(), &bounds, &cfg(kb)).expect("rohf")
}

fn report(label: &str, direct: &ScfResult, other: &ScfResult) -> (f64, i64) {
    let de = other.energy - direct.energy;
    let di = other.iterations as i64 - direct.iterations as i64;
    println!(
        "{label}: direct E={:.12} ({} iters, conv={}, quartets={}); other E={:.12} ({} iters, conv={}, quartets={}); \
         dE={de:+.3e} dIter={di:+}",
        direct.energy, direct.iterations, direct.converged, direct.computed_quartets,
        other.energy, other.iterations, other.converged, other.computed_quartets
    );
    assert!(direct.converged, "{label}: direct SCF did not converge");
    assert!(other.converged, "{label}: k_builder SCF did not converge ({} iters)", other.iterations);
    (de, di)
}

/// (a)/(b) LinK: exact to the screen → 1e-9, ±2 iterations, AND a different
/// quartet count than the combined direct pass (the "was it used" tell).
fn check_link(label: &str, direct: &ScfResult, link: &ScfResult) {
    let (de, di) = report(label, direct, link);
    assert!(de.abs() <= 1e-9, "{label}: LinK energy differs from direct by {de:+.3e} Ha (bar 1e-9)");
    assert!(di.abs() <= 2, "{label}: LinK iterations {} vs direct {} (more than 2 apart)", link.iterations, direct.iterations);
    assert_ne!(
        link.computed_quartets, direct.computed_quartets,
        "{label}: LinK run walked exactly the direct path's quartet count — k_builder was NOT consumed"
    );
}

/// (a)/(b) COSX: within the grid bar, ±3 iterations, AND measurably NOT the
/// direct energy (a consumed seminumerical K carries grid error).
fn check_cosx(label: &str, direct: &ScfResult, cosx: &ScfResult) {
    let (de, di) = report(label, direct, cosx);
    assert!(
        de.abs() <= COSX_OPEN_SHELL_E_BAR,
        "{label}: COSX energy differs from direct by {de:+.3e} Ha (bar {COSX_OPEN_SHELL_E_BAR:.0e})"
    );
    assert!(
        de.abs() > COSX_MUST_DIFFER_FLOOR,
        "{label}: COSX energy is bit-for-bit the direct energy ({de:+.3e}) — k_builder was NOT consumed"
    );
    assert!(di.abs() <= 3, "{label}: COSX iterations {} vs direct {} (more than 3 apart)", cosx.iterations, direct.iterations);
}

// ── (a) UHF ──────────────────────────────────────────────────────────────────

#[test]
fn uhf_link_matches_direct_ch3_ccpvdz() {
    let direct = run_uhf_ch3(None);
    let link = run_uhf_ch3(Some("link"));
    check_link("UHF CH3/cc-pVDZ link", &direct, &link);
}

#[test]
fn uhf_cosx_within_grid_error_ch3_ccpvdz() {
    let direct = run_uhf_ch3(None);
    let cosx = run_uhf_ch3(Some("cosx"));
    check_cosx("UHF CH3/cc-pVDZ cosx", &direct, &cosx);
}

// ── (b) ROHF ─────────────────────────────────────────────────────────────────

#[test]
fn rohf_link_matches_direct_ch3_ccpvdz() {
    let direct = run_rohf_ch3(None);
    let link = run_rohf_ch3(Some("link"));
    check_link("ROHF CH3/cc-pVDZ link", &direct, &link);
}

#[test]
fn rohf_cosx_within_grid_error_ch3_ccpvdz() {
    let direct = run_rohf_ch3(None);
    let cosx = run_rohf_ch3(Some("cosx"));
    check_cosx("ROHF CH3/cc-pVDZ cosx", &direct, &cosx);
}

// ── (c) RHF byte-identity record ─────────────────────────────────────────────

/// Closed-shell water/cc-pVDZ through the UNCHANGED `solve_rhf` path with
/// `k_builder` direct / link / cosx. Prints the f64 bit patterns; when
/// `FERRIC_RHF_BITS_EXPECT=<hex>,<hex>,<hex>` is set (recorded on the same
/// machine BEFORE the open-shell wiring) it asserts bitwise equality. The
/// (c) RHF regression guard. Extracting `resolve_k_builder` / `build_pluggable_k`
/// out of `solve_rhf` must not move closed-shell numerics.
///
/// This compares each builder against `direct` IN THE SAME RUN rather than
/// against bit patterns recorded on one machine. An earlier version baked in
/// hex energies and went red on CI for a non-reason: `direct` matched
/// bit-for-bit (0 ULP) while `link` and `cosx` differed by 7 and 17 ULP
/// (~1e-13 Ha) purely from OpenBLAS kernel selection — CI pins
/// OPENBLAS_CORETYPE=Haswell, developer machines do not, and LinK/COSX do more
/// BLAS-heavy accumulation than direct, so that is exactly where kernel choice
/// surfaces. Absolute bit patterns are a machine fingerprint, not a regression
/// test. The relative bars below hold on any hardware and still go red if the
/// refactor perturbs the closed-shell path.
#[test]
fn rhf_closed_shell_matches_direct_across_builders() {
    let (mol, prep, bounds) = setup(WATER_XYZ, 0, 1, "cc-pvdz", Operator::coulomb());
    let ctx = ParallelContext::default();

    let direct = solve_rhf(&ctx, &mol, &prep, Operator::coulomb(), &bounds, &cfg(None)).expect("rhf");
    assert!(direct.converged);
    println!(
        "RHF water/cc-pVDZ direct: E={:.12} bits={:016x} iters={}",
        direct.energy, direct.energy.to_bits(), direct.iterations
    );

    // LinK is EXACT to the screening threshold, so it must agree with direct to
    // accumulation noise. Measured 5.1e-13 (CI) / 6.1e-13 (dev box); 1e-11 leaves
    // ~20x headroom for other BLAS kernels without admitting a real defect.
    let link = solve_rhf(&ctx, &mol, &prep, Operator::coulomb(), &bounds, &cfg(Some("link"))).expect("rhf link");
    assert!(link.converged);
    let d_link = (link.energy - direct.energy).abs();
    println!(
        "RHF link: E={:.12} bits={:016x} iters={} dE_vs_direct={:.3e}",
        link.energy, link.energy.to_bits(), link.iterations, d_link
    );
    assert!(d_link < 1e-11, "LinK RHF moved vs direct: {d_link:.3e} Ha (expected accumulation noise ~1e-13)");
    assert_eq!(link.iterations, direct.iterations, "LinK changed the RHF iteration count");

    // COSX carries a GRID error, not accumulation noise: 4.862e-06 Ha on both
    // machines at (50,110)+fit. Bracket it — a value far below would mean the
    // builder silently fell back to direct, far above means the grid path broke.
    let cosx = solve_rhf(&ctx, &mol, &prep, Operator::coulomb(), &bounds, &cfg(Some("cosx"))).expect("rhf cosx");
    assert!(cosx.converged);
    let d_cosx = (cosx.energy - direct.energy).abs();
    println!(
        "RHF cosx: E={:.12} bits={:016x} iters={} dE_vs_direct={:.3e}",
        cosx.energy, cosx.energy.to_bits(), cosx.iterations, d_cosx
    );
    assert!(
        (1e-7..1e-4).contains(&d_cosx),
        "COSX RHF grid error {d_cosx:.3e} Ha outside [1e-7, 1e-4] — too small means it fell back to direct, too large means the grid path broke"
    );
    assert_eq!(cosx.iterations, direct.iterations, "COSX changed the RHF iteration count");
}

// ── (d) α/β independence from ONE builder instance ───────────────────────────

fn converged_uhf_spin_densities() -> (Molecule, PreparedBasis, SchwarzBounds, Array2<f64>, Array2<f64>) {
    let (mol, prep, bounds) = setup(CH3_XYZ, 0, 2, "cc-pvdz", Operator::coulomb());
    let r = solve_uhf(&ParallelContext::default(), &mol, &prep, &bounds, &cfg(None)).expect("uhf");
    assert!(r.converged);
    let d_b = r.density_beta.clone().expect("UHF carries D_beta");
    (mol, prep, bounds, r.density_alpha, d_b)
}

fn bits_equal(a: &Array2<f64>, b: &Array2<f64>) -> bool {
    a.dim() == b.dim() && a.iter().zip(b.iter()).all(|(x, y)| x.to_bits() == y.to_bits())
}

fn max_abs_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    (a - b).iter().fold(0.0f64, |m, v| m.max(v.abs()))
}

/// K(D_α), K(D_β), K(D_α) again from ONE instance: both K(D_α) bitwise equal,
/// and each equals a FRESH instance's build. `update_density` is called with
/// the matching spin density before every build — the contract the solvers
/// must honour (the LinK pair lists are density-dependent).
fn check_alpha_beta_independence<'b, F>(label: &str, d_a: &Array2<f64>, d_b: &Array2<f64>, mut fresh: F)
where
    F: FnMut() -> Box<dyn KBuilder + 'b>,
{
    let n = d_a.nrows();
    let build = |kb: &mut dyn KBuilder, d: &Array2<f64>| {
        let mut k = Array2::zeros((n, n));
        kb.update_density(d);
        kb.build(d, &mut k).expect("build");
        k
    };
    let mut one = fresh();
    let ka1 = build(one.as_mut(), d_a);
    let kb1 = build(one.as_mut(), d_b);
    let ka2 = build(one.as_mut(), d_a);
    drop(one);
    let ka_fresh = build(fresh().as_mut(), d_a);
    let kb_fresh = build(fresh().as_mut(), d_b);
    println!(
        "{label}: max|K_a1-K_a2|={:.3e} max|K_a1-K_a_fresh|={:.3e} max|K_b1-K_b_fresh|={:.3e} max|K_a-K_b|={:.3e}",
        max_abs_diff(&ka1, &ka2), max_abs_diff(&ka1, &ka_fresh), max_abs_diff(&kb1, &kb_fresh), max_abs_diff(&ka1, &kb1)
    );
    assert!(bits_equal(&ka1, &ka2), "{label}: K(D_a) after a K(D_b) build is not bitwise the first K(D_a)");
    assert!(bits_equal(&ka1, &ka_fresh), "{label}: shared-instance K(D_a) != fresh-instance K(D_a)");
    assert!(bits_equal(&kb1, &kb_fresh), "{label}: shared-instance K(D_b) != fresh-instance K(D_b)");
    // Non-vacuity: a doublet's α and β exchange matrices must differ.
    assert!(max_abs_diff(&ka1, &kb1) > 1e-3, "{label}: K(D_a) == K(D_b) — the two densities are not distinct");
}

#[test]
fn link_k_alpha_beta_independent_from_one_instance() {
    let (_mol, prep, bounds, d_a, d_b) = converged_uhf_spin_densities();
    let ctx = ParallelContext::default();
    let op = Operator::coulomb();
    check_alpha_beta_independence("LinK CH3/cc-pVDZ", &d_a, &d_b, || {
        Box::new(LinkK::new(&ctx, &prep, &bounds, op, 1e-12, usize::MAX)) as Box<dyn KBuilder + '_>
    });
}

#[test]
fn cosx_k_alpha_beta_independent_from_one_instance() {
    let (mol, prep, _bounds, d_a, d_b) = converged_uhf_spin_densities();
    let ctx = ParallelContext::default();
    check_alpha_beta_independence("CosxK CH3/cc-pVDZ", &d_a, &d_b, || {
        Box::new(CosxK::new(&ctx, &mol, &prep, CosxConfig::default(), usize::MAX).expect("CosxK::new"))
            as Box<dyn KBuilder + '_>
    });
}

// ── (e) DF-K warning path: warn AND skip, never a silent no-op ───────────────

const PROBE_ENV: &str = "FERRIC_KB_OPEN_SHELL_PROBE";
const DF_WARN_LINK: &str = "k_builder = \"link\" is IGNORED because density-fitted J/K is active";
const RSH_WARN_COSX: &str = "k_builder = \"cosx\" is IGNORED because";

/// Run `child_test` in a fresh copy of this test binary and return its
/// (status ok, stderr). The env flag lets the child body run; without it the
/// child test is a no-op so the normal harness sweep does not double-run it.
fn run_probe(child_test: &str) -> (bool, String) {
    let exe = std::env::current_exe().expect("current_exe");
    let out = std::process::Command::new(exe)
        .args(["--exact", child_test, "--nocapture", "--test-threads=1"])
        .env(PROBE_ENV, "1")
        .env("OPENBLAS_NUM_THREADS", "1")
        .env("RAYON_NUM_THREADS", "1")
        .output()
        .expect("spawn probe");
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    println!("--- probe {child_test} stdout ---\n{stdout}\n--- probe stderr ---\n{stderr}");
    (out.status.success(), stderr)
}

fn probe_armed() -> bool {
    std::env::var(PROBE_ENV).as_deref() == Ok("1")
}

fn df_cfg(kb: Option<&str>) -> RhfConfig {
    RhfConfig {
        df_j_aux: Some("def2-universal-jkfit".into()),
        df_k_aux: Some("def2-universal-jkfit".into()),
        ..cfg(kb)
    }
}

/// CHILD: UHF CH3/cc-pVDZ with DF-J/DF-K AND `k_builder = "link"` must produce
/// bitwise the plain DF-JK energy (builder skipped), while the parent checks
/// the warning was printed.
#[test]
fn probe_child_uhf_df_k_with_link_builder() {
    if !probe_armed() {
        return;
    }
    let (mol, prep, bounds) = setup(CH3_XYZ, 0, 2, "cc-pvdz", Operator::coulomb());
    let ctx = ParallelContext::default();
    let plain = solve_uhf(&ctx, &mol, &prep, &bounds, &df_cfg(None)).expect("uhf df");
    let with_kb = solve_uhf(&ctx, &mol, &prep, &bounds, &df_cfg(Some("link"))).expect("uhf df+link");
    assert!(plain.converged && with_kb.converged);
    assert_eq!(plain.energy.to_bits(), with_kb.energy.to_bits(), "DF-JK + k_builder=link must be bitwise the DF-JK run");
    assert_eq!(plain.iterations, with_kb.iterations);
}

#[test]
fn probe_child_rohf_df_k_with_link_builder() {
    if !probe_armed() {
        return;
    }
    let (mol, prep, bounds) = setup(CH3_XYZ, 0, 2, "cc-pvdz", Operator::coulomb());
    let ctx = ParallelContext::default();
    let plain = solve_rohf(&ctx, &mol, &prep, Operator::coulomb(), &bounds, &df_cfg(None)).expect("rohf df");
    let with_kb = solve_rohf(&ctx, &mol, &prep, Operator::coulomb(), &bounds, &df_cfg(Some("link"))).expect("rohf df+link");
    assert!(plain.converged && with_kb.converged);
    assert_eq!(plain.energy.to_bits(), with_kb.energy.to_bits(), "DF-JK + k_builder=link must be bitwise the DF-JK run");
    assert_eq!(plain.iterations, with_kb.iterations);
}

#[test]
fn uhf_df_k_with_link_builder_warns_and_skips() {
    let (ok, stderr) = run_probe("probe_child_uhf_df_k_with_link_builder");
    assert!(ok, "child probe failed (see its output above)");
    assert!(stderr.contains(DF_WARN_LINK), "UHF: DF-K + k_builder=link did not print the IGNORED warning");
}

#[test]
fn rohf_df_k_with_link_builder_warns_and_skips() {
    let (ok, stderr) = run_probe("probe_child_rohf_df_k_with_link_builder");
    assert!(ok, "child probe failed (see its output above)");
    assert!(stderr.contains(DF_WARN_LINK), "ROHF: DF-K + k_builder=link did not print the IGNORED warning");
}

// ── (f) hard errors preserved at the open-shell entry points ─────────────────

#[test]
fn unknown_k_builder_errors_from_open_shell_entry_points() {
    let (mol, prep, bounds) = setup(CH3_XYZ, 0, 2, "sto-3g", Operator::coulomb());
    let ctx = ParallelContext::default();
    let bad = cfg(Some("bogus"));
    let eu = solve_uhf(&ctx, &mol, &prep, &bounds, &bad)
        .map(|_| ()).expect_err("UHF must reject an unknown k_builder");
    let er = solve_rohf(&ctx, &mol, &prep, Operator::coulomb(), &bounds, &bad)
        .map(|_| ()).expect_err("ROHF must reject an unknown k_builder");
    for (label, e) in [("UHF", eu), ("ROHF", er)] {
        let msg = e.to_string();
        println!("{label}: {msg}");
        assert!(msg.contains("unknown k_builder 'bogus'"), "{label}: wrong error text: {msg}");
    }
}

/// `k_builder = "cosx"` is Coulomb-only: an attenuated (erfc) Schwarz bound
/// carries a non-Coulomb operator, and the open-shell solvers must refuse it
/// with the same error `solve_rhf` raises.
#[test]
fn cosx_refused_for_non_coulomb_operator_open_shell() {
    let (mol, prep, bounds) = setup(CH3_XYZ, 0, 2, "sto-3g", Operator::erfc(0.3));
    let ctx = ParallelContext::default();
    let cosx = cfg(Some("cosx"));
    let eu = solve_uhf(&ctx, &mol, &prep, &bounds, &cosx)
        .map(|_| ()).expect_err("UHF must refuse cosx + erfc");
    let er = solve_rohf(&ctx, &mol, &prep, Operator::erfc(0.3), &bounds, &cosx)
        .map(|_| ()).expect_err("ROHF must refuse cosx + erfc");
    for (label, e) in [("UHF", eu), ("ROHF", er)] {
        let msg = e.to_string();
        println!("{label}: {msg}");
        assert!(msg.contains("Coulomb operator only"), "{label}: wrong error text: {msg}");
    }
}

fn rsh_cfg(kb: Option<&str>) -> RhfConfig {
    RhfConfig {
        xc: Some("HYB_GGA_XC_WB97X".into()),
        dft_grid: Some(ferric_dft::grid::AtomicGridConfig { n_radial: 25, n_angular: 50, ..Default::default() }),
        ..cfg(kb)
    }
}

/// CHILD: an RSH functional routes exchange through the SR/LR DF-K fitters, so
/// a pluggable K has nothing to do. Same semantics as `solve_rhf` (where the
/// RSH auto-defaulted DF-K triggers the IGNORED warning): warn, and produce
/// bitwise the plain run.
#[test]
fn probe_child_uks_rsh_with_cosx_builder() {
    if !probe_armed() {
        return;
    }
    let (mol, prep, bounds) = setup(CH3_XYZ, 0, 2, "sto-3g", Operator::coulomb());
    let ctx = ParallelContext::default();
    let plain = solve_uhf(&ctx, &mol, &prep, &bounds, &rsh_cfg(None)).expect("uks rsh");
    let with_kb = solve_uhf(&ctx, &mol, &prep, &bounds, &rsh_cfg(Some("cosx"))).expect("uks rsh+cosx");
    assert!(plain.converged && with_kb.converged);
    assert_eq!(plain.energy.to_bits(), with_kb.energy.to_bits(), "RSH + k_builder=cosx must be bitwise the RSH run");
}

#[test]
fn uks_rsh_with_cosx_builder_warns_and_skips() {
    let (ok, stderr) = run_probe("probe_child_uks_rsh_with_cosx_builder");
    assert!(ok, "child probe failed (see its output above)");
    assert!(stderr.contains(RSH_WARN_COSX), "UKS RSH: k_builder=cosx did not print the IGNORED warning");
}

// ── (d2) LinK per-spin pair lists: where it IS observable ───────────────────

/// Does the per-spin `update_density(D_σ)` change anything, and where?
///
/// The per-spin contract is the SAFE choice by construction (a list built from
/// D_α can only be guaranteed to cover D_α), but this test records the
/// MEASUREMENT that it is currently not observable at production thresholds,
/// rather than asserting a difference that does not exist.
///
/// `DensityPairs` post-#50 keeps essentially every pair. Measured here, RHF
/// densities, `def2-SVP`, `DensityPairs::build` kept / full square:
///
/// ```text
///            thresh 1e-12   1e-10     1e-8       1e-6
/// alkane_4   2916/2916  2916/2916  2916/2916  2916/2916
/// alkane_8  10404/10404 ...same...            10404/10404
/// alkane_16 39204/39204 39204/39204 39204/39204 37696/39204   <- first pruning
/// ```
///
/// So at every threshold ferric actually uses (1e-12 default) the list is the
/// full square and a STALE list is numerically identical to a fresh one — the
/// reason mutation 1 (dropping the β `update_density`) does NOT turn the SCF
/// anchors red. The list only prunes at 1e-6 on C16+, and even there LinK's
/// per-quartet `Q·Q·|D|` screen re-tests every quartet against the density it
/// was actually handed, so the list is a cost optimization, not a correctness
/// input, for the magnitudes reached here.
///
/// What this test therefore asserts is the DIRECTION of the guarantee: at a
/// threshold where the list DOES prune, a list built from D_α must be a
/// superset of (or equal to) what D_α needs — and the per-spin build must
/// reproduce a fresh-instance β build EXACTLY. That last equality is the real
/// contract, and it fails if `update_density` is fed the wrong density AND the
/// list matters.
#[test]
fn link_per_spin_pair_lists_recorded_and_beta_matches_fresh_instance() {
    let mol = Molecule::parse_xyz(&strip_last_h(BUTYL_XYZ), 0, 2).expect("butyl");
    let bs = basis::bundled("def2-svp").expect("basis");
    let prep = PreparedBasis::new(&mol, &bs).expect("prep");
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).expect("schwarz");
    let ctx = ParallelContext::default();
    let r = solve_uhf(&ctx, &mol, &prep, &bounds, &cfg(None)).expect("uhf");
    assert!(r.converged);
    let (d_a, d_b) = (r.density_alpha.clone(), r.density_beta.clone().expect("D_beta"));
    let n = prep.nbasis();
    let nsh = prep.nshells();

    // RECORD the pruning behaviour at several thresholds (no assertion on the
    // count — this is the measurement the doc comment above is built from).
    for t in [1e-12f64, 1e-8, 1e-6] {
        let dp_a = ferric_scf::pairs::DensityPairs::build(&d_a, &bounds, &prep, t);
        let dp_b = ferric_scf::pairs::DensityPairs::build(&d_b, &bounds, &prep, t);
        println!(
            "butyl/def2-SVP thresh={t:.0e}: dp_pairs(D_a)={}/{} dp_pairs(D_b)={}/{}",
            dp_a.total_pairs(), nsh * nsh, dp_b.total_pairs(), nsh * nsh
        );
    }

    // THE CONTRACT: a shared instance driven per-spin must reproduce a fresh
    // instance's beta build bit-for-bit. This is what a wrong-density
    // `update_density` would break once the list prunes, and it is asserted at
    // the production threshold where the SCF anchors run.
    let thresh = 1e-12;
    let mut lk = LinkK::new(&ctx, &prep, &bounds, op, thresh, usize::MAX);
    let (mut ka, mut kb_shared) = (Array2::zeros((n, n)), Array2::zeros((n, n)));
    lk.update_density(&d_a);
    lk.build(&d_a, &mut ka).expect("alpha");
    lk.update_density(&d_b);
    lk.build(&d_b, &mut kb_shared).expect("beta");

    let mut fresh = LinkK::new(&ctx, &prep, &bounds, op, thresh, usize::MAX);
    let mut kb_fresh = Array2::zeros((n, n));
    fresh.update_density(&d_b);
    fresh.build(&d_b, &mut kb_fresh).expect("fresh beta");

    println!(
        "butyl/def2-SVP: max|K_b(shared, per-spin) - K_b(fresh)| = {:.3e}; max|K_a - K_b| = {:.3e}",
        max_abs_diff(&kb_shared, &kb_fresh), max_abs_diff(&ka, &kb_shared)
    );
    assert!(bits_equal(&kb_shared, &kb_fresh), "shared-instance per-spin K_b != fresh-instance K_b");
    assert!(max_abs_diff(&ka, &kb_shared) > 1e-3, "K_a == K_b: the two spin densities are not distinct");
}

/// Drop the last H from an alkane xyz to make an open-shell radical.
fn strip_last_h(xyz: &str) -> String {
    let mut lines: Vec<&str> = xyz.lines().collect();
    let last_h = lines.iter().rposition(|l| l.trim_start().starts_with('H')).expect("an H");
    lines.remove(last_h);
    let natoms: usize = lines[0].trim().parse::<usize>().expect("count") - 1;
    format!("{natoms}\n{}\n", lines[1..].join("\n"))
}

/// Butane geometry, used as a butyl radical by `strip_last_h`.
const BUTYL_XYZ: &str = include_str!("../../../testdata/molecules/alkane_4.xyz");
