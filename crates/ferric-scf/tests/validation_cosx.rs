//! VALIDATION tier — VALIDATION.md row "COSX seminumerical exchange".
//!
//! Runs only in the weekly `validation` CI job (and on demand):
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 cargo nextest run -p ferric-scf --test validation_cosx \
//!     --run-ignored only -E 'binary(/^validation_/)'
//! ```
//!
//! # What is compared
//!
//! Systems: H2O / aug-cc-pVDZ (RHF and B3LYP) and all-anti butane
//! (`testdata/molecules/alkane_4.xyz`) / def2-SVP (RHF). Two inputs that the
//! grid error responds to differently: diffuse functions on a 3-atom molecule,
//! and a 14-atom chain (more Becke cell boundaries, the system on which the
//! "−3.4e-5 floor" was investigated and resolved as the Lebedev-302 angular
//! limit — wiki/validation-campaign-bugs-2026-09.md C1).
//!
//! 1. **Exactness anchor (grid → ∞ is exact K).** For each system, the COSX
//!    SCF energy is computed live on a ladder of grids and compared with
//!    ferric's own exact-K (direct four-centre) SCF energy computed live in the
//!    same test. Both sides share J, the one-electron part and (B3LYP) the XC
//!    grid, so `E_cosx − E_exact` is the COSX grid error and nothing else.
//!    The exact-K energy is itself anchored to PySCF's exact RHF/RKS on the
//!    same basis JSON and geometry ([`exact_k_matches_pyscf`]).
//! 2. **Independent construction on the SAME grid.** PySCF's `sgx`
//!    (2.13.1, `get_jk_favorj`) with exact J, `fit_ovlp=False`, no density
//!    screening and no point removal, on a grid built exactly as ferric builds
//!    its COSX grid (flat Treutler radial × Lebedev, Becke partition with
//!    Becke size adjustment — the recipe that matches ferric's KS energies to
//!    1e-10). That is the same discrete operator `K = sym(Σ_g w_g χ_g (A^g D
//!    χ_g)^T)`, so ferric (fit off, screens off) must match it at EVERY grid,
//!    not only in the dense limit. This is stronger than the dense-limit
//!    comparison the plan asked for, and it is what can see a construction
//!    error smaller than the grid error.
//! 3. **PySCF's own fit (information only).** PySCF fits the overlap on the
//!    DENSITY side (`F = w χ^T (S_num^{-1} S D)`), ferric on the output side
//!    (`K = sym(S S_num^{-1} K̃)`). Both → exact K, but they differ at every
//!    finite grid, so fitted energies are NOT compared grid by grid: the
//!    PySCF-fit ladder (110 … 974) is printed next to ferric's, and only the
//!    dense-limit approach to `exact` is comparable.
//! 4. **Analytic COSX gradient vs central FD** of the COSX SCF energy (the
//!    gradient landed in PR #164, `cosx_gradient.rs`), at the production knobs
//!    (default (50,110) grid, default screens): RHF with the overlap fit
//!    (Z-vector response) on water/aug-cc-pVDZ and NH3/def2-SVP, and B3LYP with
//!    the fit OFF (fitted COSX + KS gradients are refused by design).
//!
//! References: `scripts/validation/gen_cosx.py` →
//! `testdata/reference/validation/cosx/{h2o_aug-cc-pvdz,butane_def2-svp}.json`.
//! The ferric-vs-ferric ladders need no reference; only tests 1-2 read JSON.
//!
//! # Physics hypothesis vs artifact hypothesis (Experimental Protocol)
//!
//! * If COSX is right: `|E_cosx − E_exact|` falls by a large measured factor
//!   from (75,110) to (75,590), roughly monotonically in magnitude along the
//!   angular ladder (sign may oscillate: PySCF on butane gives + − + across
//!   302/434/590), and lands at the ~1e-6 Ha level; the fit-off/unscreened
//!   energy equals PySCF's same-grid SGX energy to the SCF floor at every grid.
//! * If the construction is wrong by something grid-INDEPENDENT (a missing
//!   symmetrization, a weight normalised to 4π instead of 1, `w` instead of
//!   `sqrt(w)` in X, a dropped Becke partition, a wrong A sign): the error
//!   does NOT shrink with the grid — the [`MIN_REDUCTION`] ratio fails even
//!   when the dense-grid value happens to be small, and the same-grid PySCF
//!   comparison fails at O(1e-3..1) Ha.
//! * If the construction is wrong by something that ALSO vanishes with the
//!   grid (e.g. `Ktilde` not symmetrized: its antisymmetric part is a grid
//!   error), the ladder can pass; the same-grid PySCF comparison is the
//!   assertion that catches it, because both sides see the same quadrature.
//! * If the harness is wrong (basis, geometry, functional, XC grid): nuclear
//!   repulsion, AO count and the exact-K energy vs PySCF fail first.
//! * If COSX is silently not used (DF-K active, k_builder ignored): the error
//!   is exactly 0; [`NOT_EXACT_FLOOR`] fails.
//!
//! # MEASURED
//!
//! PySCF side (gen_cosx.py, PySCF 2.13.1, 2026-10-01), E − E_exact in Ha, radial 75:
//!
//! | system / method | fit  | 110       | 302       | 434       | 590       | 974       |
//! |-----------------|------|-----------|-----------|-----------|-----------|-----------|
//! | H2O/aDZ HF      | off  | +5.308e-6 | −2.986e-7 | +4.761e-8 | +1.635e-8 |           |
//! | H2O/aDZ HF      | PySCF| +6.129e-6 | −7.275e-9 | +3.874e-9 | −2.269e-9 | +2.852e-10|
//! | H2O/aDZ B3LYP   | off  | +1.111e-6 | −6.071e-8 | +9.956e-9 | +3.365e-9 |           |
//! | H2O/aDZ B3LYP   | PySCF| +1.072e-6 | −1.376e-9 | +9.684e-10| −4.694e-10| +8.711e-12|
//! | butane/SVP HF   | off  | −2.490e-4 | +5.950e-5 | −5.579e-6 | −6.708e-6 |           |
//! | butane/SVP HF   | PySCF| +1.691e-4 | −3.396e-5 | −5.863e-6 | +1.565e-6 | −6.319e-8 |
//!
//! NOTE the fit-OFF butane ladder is NOT monotone in magnitude (434 → 590 grows
//! 1.2x), which is why the ladder tests run with the fit ON and the fit-off
//! runs are only compared same-grid. The ferric fit-on butane values from #199
//! (−3.4e-5 / −5.6e-6 / +1.8e-6 at 302/434/590) sit within ~5% of PySCF's
//! own-fit column even though the two fit variants differ — not asserted.
//! ferric (this suite, release, 2026-10-01): fit-off same-grid vs PySCF sgx
//! ≤1.2e-12 at every grid; fit-on ladder tracks PySCF's own-fit column (water
//! to ~1e-11, butane within 15%); exact-K vs PySCF ≤2.1e-11; production
//! (50,110)+fit water RHF +6.11e-6, B3LYP +1.07e-6, butane +1.69e-4.
//!
//! # NEGATIVE CONTROLS / MUTATIONS (to run once against a broken build)
//!
//! * Always-on: `|E_cosx(75,110) − E_exact| ≥ NOT_EXACT_FLOOR` (COSX in use).
//! * Always-on: the exact-K gradient evaluated at the COSX density must MISS
//!   FD of the COSX energy by ≥ [`GRAD_MUST_MISS_FACTOR`] × the bar (the FD
//!   harness can see the defect PR #164 fixed).
//! * MUTATION A — `cosx_k.rs`: K_plain = Ktilde (drop the `0.5(K̃ + K̃^T)`):
//!   expected to fail [`cosx_matches_pyscf_sgx_on_the_same_grid_water`]; may
//!   PASS the ladders (grid-vanishing defect) — that is the point of test 2.
//! * MUTATION B — `cosx_k.rs`: use `w` instead of `sqrt(|w|)` in X: every
//!   ladder fails `MIN_REDUCTION` and the dense bar by O(1) Ha.
//! * MUTATION C — `cosx_gradient.rs`: drop the Becke weight-response term (c):
//!   the FD tests fail (prototype ablation: O(1e-2..1)).

use std::path::{Path, PathBuf};

use ferric_core::basis::{self, BasisSet};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_dft::grid::AtomicGridConfig;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::cosx_k::{CosxConfig, CosxHalfTransform};
use ferric_scf::gradient::{restricted_scf_gradient, rhf_gradient};
use ferric_scf::ks_gradient::ks_gradient_closed;
use ferric_scf::result::ScfResult;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ndarray::Array2;
use rayon::prelude::*;
use serde_json::Value;

const ROW_DIR: &str = "testdata/reference/validation/cosx";
const RADIAL: usize = 75;

// ---------------------------------------------------------------------------
// TOLERANCES, measured 2026-10-01 (release; water/aug-cc-pVDZ RHF and B3LYP,
// butane/def2-SVP RHF). Each bar sits between the measured agreement and the
// smallest defect the matching control or mutation produces.
// ---------------------------------------------------------------------------

/// ferric exact-K SCF energy vs PySCF exact. Measured ≤2.1e-11 (butane).
const TOL_EXACT_VS_PYSCF: f64 = 1e-9;
/// ferric COSX (fit off, screens off) vs PySCF SGX on the identical grid.
/// Measured ≤1.2e-12 at every grid: the same discrete operator in two codes.
const TOL_SAME_GRID: f64 = 1e-10;
/// `|E_cosx − E_exact|` at the densest ladder grid (75,590) with production
/// knobs. Measured: butane 1.8e-6, water HF 2.4e-9, B3LYP 4.2e-10.
const TOL_DENSE_GAP: f64 = 5e-6;
/// `|E_cosx(75,110) − E_exact| / |E_cosx(75,590) − E_exact|`. Measured: butane
/// 93x, water HF 2576x, B3LYP 2525x. A grid-independent defect gives ~1.
const MIN_REDUCTION: f64 = 30.0;
/// Each step along the angular ladder may grow `|error|` by at most this
/// factor (the sign oscillates). Measured fit-on magnitudes decrease at every
/// step except water at 302 → 434 (×0.53 HF, ×0.74 B3LYP: still decreasing).
const MONOTONE_SLACK: f64 = 1.5;
/// The production default (50,110)+fit vs exact: the plan's "2e-5 SCF" bar.
/// Measured water RHF +6.11e-6, B3LYP +1.07e-6. Butane/def2-SVP at the
/// production grid is +1.69e-4 (bug list C1: angular-grid limit), so this bar
/// is applied to water only and butane's production point is printed.
const TOL_PRODUCTION_WATER: f64 = 2e-5;
/// COSX must actually be in use, checked at the COARSEST grid only: the true
/// COSX error falls to ~1e-8 at (75,434) for B3LYP, so a floor on every grid
/// would fail a correct COSX. Measured at (75,110): ≥1.1e-6.
const NOT_EXACT_FLOOR: f64 = 1e-7;
/// Analytic COSX gradient vs central FD of the COSX energy, Ha/Bohr.
/// Measured ≤1.6e-8 (water RHF fitted, water B3LYP unfitted, NH3 fitted).
const TOL_GRAD_FD: f64 = 2e-7;
/// Negative control: the exact-K gradient at the COSX density must miss FD by
/// at least this multiple of [`TOL_GRAD_FD`]. Measured misses 2.1e-5..8.5e-5.
const GRAD_MUST_MISS_FACTOR: f64 = 5.0;
const FD_H: f64 = 1e-4;
const TOL_ENUC: f64 = 1e-9;

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

fn workspace_root() -> PathBuf {
    let looks_like_root = |p: &Path| {
        p.join("Cargo.toml").is_file() && p.join("testdata").is_dir() && p.join("crates").is_dir()
    };
    if let Ok(cwd) = std::env::current_dir() {
        let mut here: Option<&Path> = Some(cwd.as_path());
        while let Some(p) = here {
            if looks_like_root(p) {
                return p.to_path_buf();
            }
            here = p.parent();
        }
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("ferric-scf manifest dir should be <root>/crates/ferric-scf")
        .to_path_buf()
}

fn reference(system: &str, basis_name: &str) -> Value {
    let path = workspace_root()
        .join(ROW_DIR)
        .join(format!("{system}_{basis_name}.json"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "missing validation reference {} ({e}); regenerate with \
             scripts/validation/gen_cosx.py — a missing reference is a failure, never a skip",
            path.display()
        )
    });
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{}: bad JSON: {e}", path.display()))
}

fn num(v: &Value, ptr: &str, ctx: &str) -> f64 {
    v.pointer(ptr)
        .and_then(Value::as_f64)
        .unwrap_or_else(|| panic!("{ctx}: reference field {ptr} missing or not a number"))
}

/// The two systems of the row.
#[derive(Clone, Copy)]
enum Sys {
    Water,
    Butane,
}

impl Sys {
    fn name(self) -> &'static str {
        match self {
            Sys::Water => "h2o",
            Sys::Butane => "butane",
        }
    }
    fn basis(self) -> &'static str {
        match self {
            Sys::Water => "aug-cc-pvdz",
            Sys::Butane => "def2-svp",
        }
    }
    fn xyz(self) -> PathBuf {
        let root = workspace_root();
        match self {
            Sys::Water => root.join("testdata/molecules/validation/h2o.xyz"),
            Sys::Butane => root.join("testdata/molecules/alkane_4.xyz"),
        }
    }
}

struct Setup {
    label: String,
    mol: Molecule,
    bs: BasisSet,
    prep: PreparedBasis,
    bounds: SchwarzBounds,
}

fn setup_from(label: String, mol: Molecule, basis_name: &str) -> Setup {
    let bs = basis::bundled(basis_name).expect("bundled basis");
    let prep = PreparedBasis::new(&mol, &bs).expect("prep");
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).expect("schwarz");
    Setup {
        label,
        mol,
        bs,
        prep,
        bounds,
    }
}

fn setup(sys: Sys) -> Setup {
    let xyz = sys.xyz();
    let mol = Molecule::load_xyz_with_charge(xyz.to_str().unwrap(), 0, 1)
        .unwrap_or_else(|e| panic!("{}: {e:?}", xyz.display()));
    setup_from(format!("{}/{}", sys.name(), sys.basis()), mol, sys.basis())
}

/// Harness checks against the reference before any energy is read.
fn check_harness(s: &Setup, r: &Value) {
    let enuc = s.mol.nuclear_repulsion();
    let want = num(r, "/nuclear_repulsion", &s.label);
    assert!(
        (enuc - want).abs() < TOL_ENUC,
        "{}: E_nuc {enuc:.12} vs reference {want:.12}",
        s.label
    );
    assert_eq!(
        s.prep.nbasis(),
        r["nao"].as_u64().expect("nao") as usize,
        "{}: AO count differs from the reference's",
        s.label
    );
}

/// `xc = None` is RHF. J is always exact four-centre (`df_j_aux = ""` for KS,
/// which otherwise defaults to RI-J), so COSX is the only approximation.
fn scf_cfg(xc: Option<&str>, cosx: Option<CosxConfig>) -> RhfConfig {
    RhfConfig {
        k_builder: cosx.as_ref().map(|_| "cosx".to_string()),
        cosx: cosx.unwrap_or_default(),
        xc: xc.map(str::to_string),
        df_j_aux: xc.map(|_| String::new()),
        df_k_aux: xc.map(|_| String::new()),
        // The XC grid (B3LYP only) is pinned to the reference's (75,110) flat.
        dft_grid: xc.map(|_| AtomicGridConfig {
            n_radial: 75,
            n_angular: 110,
            ..Default::default()
        }),
        // density_conv is ferric's primary criterion; energy_conv is a sanity
        // bound, reachable here because nothing is density-fitted.
        energy_conv: 1e-9,
        density_conv: 1e-9,
        max_iter: 300,
        ..Default::default()
    }
}

fn seeded(cfg: RhfConfig, seed: &Array2<f64>) -> RhfConfig {
    RhfConfig {
        init_guess_density: Some(seed.clone()),
        use_sad_guess: false,
        ..cfg
    }
}

fn solve(s: &Setup, cfg: &RhfConfig, what: &str) -> ScfResult {
    let r = solve_rhf(
        &ParallelContext::default(),
        &s.mol,
        &s.prep,
        Operator::coulomb(),
        &s.bounds,
        cfg,
    )
    .unwrap_or_else(|e| panic!("{} {what}: SCF failed: {e:?}", s.label));
    assert!(
        r.converged,
        "{} {what}: SCF did not converge ({:?})",
        s.label, r.exit
    );
    r
}

fn grid(n_radial: usize, n_angular: usize) -> AtomicGridConfig {
    AtomicGridConfig {
        n_radial,
        n_angular,
        ..Default::default()
    }
}

/// Production knobs (overlap fit, default screens, sparse half transforms) on
/// a chosen grid.
fn cosx_production(g: AtomicGridConfig) -> CosxConfig {
    CosxConfig {
        grid: g,
        final_grid: None,
        ..CosxConfig::flat_reference()
    }
}

/// The operator PySCF's `sgx_matched_nofit` block computes: no fit, no pair
/// screen, dense half transforms (both screens' trivial limits).
fn cosx_unscreened_nofit(g: AtomicGridConfig) -> CosxConfig {
    CosxConfig {
        grid: g,
        overlap_fit: false,
        screen_thresh: None,
        half_transform: CosxHalfTransform::Dense,
        final_grid: None,
        ..CosxConfig::flat_reference()
    }
}

fn method_key(xc: Option<&str>) -> &'static str {
    match xc {
        None => "hf",
        Some(x) if x.eq_ignore_ascii_case("b3lyp") => "b3lyp",
        Some(x) => panic!("no reference block for xc {x}"),
    }
}

fn max_abs(a: &Array2<f64>) -> f64 {
    a.iter().fold(0.0_f64, |m, v| m.max(v.abs()))
}

// ---------------------------------------------------------------------------
// 1. Exact-K anchor vs PySCF
// ---------------------------------------------------------------------------

#[test]
#[ignore = "validation: COSX seminumerical exchange"]
fn exact_k_matches_pyscf() {
    for (sys, xcs) in [
        (Sys::Water, vec![None, Some("B3LYP")]),
        (Sys::Butane, vec![None]),
    ] {
        let s = setup(sys);
        let r = reference(sys.name(), sys.basis());
        check_harness(&s, &r);
        for xc in xcs {
            let key = method_key(xc);
            let e = solve(&s, &scf_cfg(xc, None), "exact").energy;
            let want = num(&r, &format!("/methods/{key}/exact/energy"), &s.label);
            let d = (e - want).abs();
            eprintln!(
                "{} {key}: exact-K E ferric {e:.12} PySCF {want:.12} |d| {d:.2e} (tol {TOL_EXACT_VS_PYSCF:.0e})",
                s.label
            );
            assert!(
                d < TOL_EXACT_VS_PYSCF,
                "{} {key}: exact-K energy {e:.12} vs PySCF {want:.12} (|d| {d:.2e})",
                s.label
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 2. Same grid, independent construction (PySCF sgx, fit off)
// ---------------------------------------------------------------------------

fn same_grid_case(sys: Sys, xc: Option<&str>, angulars: &[usize]) {
    let s = setup(sys);
    let r = reference(sys.name(), sys.basis());
    check_harness(&s, &r);
    let key = method_key(xc);
    let exact = solve(&s, &scf_cfg(xc, None), "exact");
    let mut worst = 0.0_f64;
    for &ang in angulars {
        let cfg = seeded(
            scf_cfg(xc, Some(cosx_unscreened_nofit(grid(RADIAL, ang)))),
            exact.density_r(),
        );
        let e = solve(&s, &cfg, &format!("cosx ({RADIAL},{ang}) fit off")).energy;
        let ptr = format!("/methods/{key}/sgx_matched_nofit/{RADIAL}_{ang}/energy");
        let want = num(&r, &ptr, &s.label);
        let d = (e - want).abs();
        worst = worst.max(d);
        eprintln!(
            "{} {key} ({RADIAL},{ang:3}) fit off, unscreened: ferric {e:.12} PySCF-sgx {want:.12} |d| {d:.2e}; \
             ferric E-E_exact {:+.3e}",
            s.label,
            e - exact.energy
        );
        assert!(
            ang != angulars[0] || (e - exact.energy).abs() >= NOT_EXACT_FLOOR,
            "{} {key} ({RADIAL},{ang}): COSX energy equals exact-K to {:.1e} — COSX not in use?",
            s.label,
            (e - exact.energy).abs()
        );
    }
    assert!(
        worst < TOL_SAME_GRID,
        "{} {key}: ferric COSX vs PySCF sgx on the same grid: max |d| {worst:.2e} >= {TOL_SAME_GRID:.0e}",
        s.label
    );
}

#[test]
#[ignore = "validation: COSX seminumerical exchange"]
fn cosx_matches_pyscf_sgx_on_the_same_grid_water() {
    same_grid_case(Sys::Water, None, &[110, 302, 434, 590]);
    same_grid_case(Sys::Water, Some("B3LYP"), &[110, 302, 434, 590]);
}

/// Butane at (75,110) and (75,302) only: unscreened COSX on the denser grids
/// costs minutes per SCF (the reference has 434/590 too, should a slot run
/// want them).
#[test]
#[ignore = "validation: COSX seminumerical exchange"]
fn cosx_matches_pyscf_sgx_on_the_same_grid_butane() {
    same_grid_case(Sys::Butane, None, &[110, 302]);
}

/// 2b. The PRUNED `sgx` grid (`PruneScheme::Sgx`, one radial count for every
/// element, ORCA/PySCF region orders) vs PySCF SGX with its own `sgx_prune`
/// on the same Treutler radial and Becke partition
/// (`scripts/validation/gen_cosx_pruned.py`): the same discrete operator, so
/// ferric (fit off, screens off) must match to the same-grid bar. Fails at
/// O(1e-4..1e-3) if a region boundary, a region order or the Bragg radius
/// differs (the pruned and flat (35,194) energies differ by that much).
#[test]
#[ignore = "validation: COSX seminumerical exchange (pruned grid)"]
fn pruned_sgx_grid_matches_pyscf_on_the_same_grid() {
    for sys in [Sys::Water, Sys::Butane] {
        let s = setup(sys);
        let path = workspace_root()
            .join("testdata/reference/validation/cosx_pruned")
            .join(format!("{}_{}.json", sys.name(), sys.basis()));
        let r: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap_or_else(|e| {
            panic!("missing {} ({e}); run gen_cosx_pruned.py", path.display())
        }))
        .expect("json");
        check_harness(&s, &r);
        let exact = solve(&s, &scf_cfg(None, None), "exact");
        for (nr, na) in [(35usize, 194usize), (50, 302)] {
            let mut g = grid(nr, na);
            g.prune = Some(ferric_dft::prune::PruneScheme::Sgx);
            let npts =
                ferric_dft::grid::atomic_grid_point_count(&s.mol, &g, g.prune).expect("count");
            let cfg = seeded(
                scf_cfg(None, Some(cosx_unscreened_nofit(g))),
                exact.density_r(),
            );
            let e = solve(&s, &cfg, &format!("sgx({nr},{na}) fit off")).energy;
            let key = format!("/sgx_pruned_matched_nofit/{nr}_{na}");
            let want = num(&r, &format!("{key}/energy"), &s.label);
            let want_n = num(&r, &format!("{key}/n_points"), &s.label) as usize;
            let d = (e - want).abs();
            eprintln!(
                "{} sgx({nr},{na}) fit off: ferric {e:.12} PySCF {want:.12} |d| {d:.2e}; npts                  ferric {npts} PySCF {want_n}; E-E_exact {:+.3e}",
                s.label,
                e - exact.energy
            );
            assert_eq!(npts, want_n, "{}: pruned grid point count differs", s.label);
            assert!(
                d < TOL_SAME_GRID,
                "{}: pruned sgx({nr},{na}) |d| {d:.2e}",
                s.label
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 3. Grid convergence to exact K (ferric vs ferric, both sides live)
// ---------------------------------------------------------------------------

/// The angular ladder at fixed radial 75 (C1: the radial axis is already
/// converged at 50 on butane; the whole error is angular).
const LADDER: [usize; 4] = [110, 302, 434, 590];

fn ladder_case(sys: Sys, xc: Option<&str>) {
    let s = setup(sys);
    let key = method_key(xc);
    // PySCF's fitted ladder, printed alongside (different fit variant: only
    // the approach to exact is comparable). Read but never asserted.
    let r = reference(sys.name(), sys.basis());
    check_harness(&s, &r);

    let exact = solve(&s, &scf_cfg(xc, None), "exact");

    // Production point: the library default (SCF grid + final pass).
    let prod_cfg = seeded(scf_cfg(xc, Some(CosxConfig::default())), exact.density_r());
    let e_prod = solve(&s, &prod_cfg, "cosx production default").energy - exact.energy;
    eprintln!(
        "{} {key}: production default  E-E_exact {e_prod:+.3e}",
        s.label
    );

    let mut errs = Vec::with_capacity(LADDER.len());
    for &ang in &LADDER {
        let cfg = seeded(
            scf_cfg(xc, Some(cosx_production(grid(RADIAL, ang)))),
            exact.density_r(),
        );
        let res = solve(&s, &cfg, &format!("cosx ({RADIAL},{ang})+fit"));
        let err = res.energy - exact.energy;
        let pyscf_fit = r
            .pointer(&format!(
                "/methods/{key}/sgx_pyscf_fit/{RADIAL}_{ang}/minus_exact"
            ))
            .and_then(Value::as_f64);
        eprintln!(
            "{} {key}: ({RADIAL},{ang:3})+fit  E-E_exact {err:+.3e}  ({} iters)  [PySCF own-fit same grid: {}]",
            s.label,
            res.iterations,
            pyscf_fit.map_or("n/a".into(), |v| format!("{v:+.3e}"))
        );
        errs.push(err);
    }
    if let Some(v) = r
        .pointer(&format!(
            "/methods/{key}/sgx_pyscf_fit/{RADIAL}_974/minus_exact"
        ))
        .and_then(Value::as_f64)
    {
        eprintln!(
            "{} {key}: [PySCF own-fit ({RADIAL},974): {v:+.3e}]",
            s.label
        );
    }

    let coarse = errs[0].abs();
    let dense = errs[errs.len() - 1].abs();
    let reduction = coarse / dense;
    eprintln!(
        "{} {key}: |err| ({RADIAL},110) {coarse:.3e} -> ({RADIAL},590) {dense:.3e}: reduction {reduction:.1}x",
        s.label
    );
    assert!(
        coarse >= NOT_EXACT_FLOOR,
        "{} {key}: COSX equals exact-K to {coarse:.1e} at ({RADIAL},110) — COSX not in use?",
        s.label
    );
    assert!(
        dense < TOL_DENSE_GAP,
        "{} {key}: COSX at ({RADIAL},590) is {dense:.3e} from exact K (bar {TOL_DENSE_GAP:.0e})",
        s.label
    );
    assert!(
        reduction >= MIN_REDUCTION,
        "{} {key}: refining ({RADIAL},110) -> ({RADIAL},590) shrank the error only {reduction:.2}x \
         (< {MIN_REDUCTION}): a grid-independent component is present",
        s.label
    );
    for w in errs.windows(2) {
        assert!(
            w[1].abs() <= MONOTONE_SLACK * w[0].abs(),
            "{} {key}: |error| grew along the ladder: {:.3e} -> {:.3e} (slack {MONOTONE_SLACK})",
            s.label,
            w[0].abs(),
            w[1].abs()
        );
    }
    if matches!(sys, Sys::Water) {
        assert!(
            e_prod.abs() < TOL_PRODUCTION_WATER,
            "{} {key}: production (50,110)+fit is {e_prod:.3e} from exact (bar {TOL_PRODUCTION_WATER:.0e})",
            s.label
        );
    }
}

#[test]
#[ignore = "validation: COSX seminumerical exchange"]
fn cosx_converges_to_exact_k_water_augccpvdz_rhf() {
    ladder_case(Sys::Water, None);
}

#[test]
#[ignore = "validation: COSX seminumerical exchange"]
fn cosx_converges_to_exact_k_water_augccpvdz_b3lyp() {
    ladder_case(Sys::Water, Some("B3LYP"));
}

/// Heavy: the (75,434)/(75,590) butane SCFs were 409 s / 489 s each in #199.
/// Run in the validation slot.
#[test]
#[ignore = "validation: COSX seminumerical exchange (heavy: ~25 min, use the slot)"]
fn cosx_converges_to_exact_k_butane_def2svp_rhf() {
    ladder_case(Sys::Butane, None);
}

// ---------------------------------------------------------------------------
// 4. Analytic COSX gradient vs FD of the COSX energy (production knobs)
// ---------------------------------------------------------------------------

fn displaced(mol: &Molecule, atom: usize, c: usize, d: f64) -> Molecule {
    let mut m = mol.clone();
    match c {
        0 => m.atoms[atom].x += d,
        1 => m.atoms[atom].y += d,
        _ => m.atoms[atom].zpos += d,
    }
    m
}

/// Central FD of the SCF energy; every displaced SCF is seeded from the
/// reference density (keeps all 6N solves in one basin).
fn fd_gradient(s: &Setup, basis_name: &str, cfg: &RhfConfig, seed: &Array2<f64>) -> Array2<f64> {
    let n = s.mol.atoms.len();
    let cfg = seeded(cfg.clone(), seed);
    let pairs: Vec<(usize, usize)> = (0..n).flat_map(|a| (0..3).map(move |c| (a, c))).collect();
    let vals: Vec<f64> = pairs
        .par_iter()
        .map(|&(a, c)| {
            let e = |d: f64| {
                let sp = setup_from(s.label.clone(), displaced(&s.mol, a, c, d), basis_name);
                solve(&sp, &cfg, "FD").energy
            };
            (e(FD_H) - e(-FD_H)) / (2.0 * FD_H)
        })
        .collect();
    let mut g = Array2::zeros((n, 3));
    for (&(a, c), v) in pairs.iter().zip(vals) {
        g[[a, c]] = v;
    }
    g
}

/// (analytic gradient through the dispatch, exact-K gradient at the same
/// density = the pre-#164 pairing).
fn grads(s: &Setup, cfg: &RhfConfig, r: &ScfResult) -> (Array2<f64>, Array2<f64>) {
    let op = Operator::coulomb();
    let g = restricted_scf_gradient(&s.mol, &s.prep, &s.bs, op, &s.bounds, cfg, r)
        .unwrap_or_else(|e| panic!("{}: gradient: {e:?}", s.label));
    let g_exact_k = match cfg.xc.as_deref() {
        Some(xc) => ks_gradient_closed(&s.mol, &s.prep, &s.bs, op, &s.bounds, xc, r, None)
            .expect("ks gradient"),
        None => rhf_gradient(&s.mol, &s.prep, op, &s.bounds, r, None).expect("rhf gradient"),
    };
    (g, g_exact_k)
}

/// One FD case. For a functional the residual is taken RELATIVE to the same
/// residual of an exact-K run (cancels what the XC gradient leaves against FD
/// and isolates exchange); for HF the absolute residual is asserted.
fn fd_case(xyz_rel: &str, basis_name: &str, xc: Option<&str>, cosx: CosxConfig) {
    let xyz = workspace_root().join(xyz_rel);
    let mol = Molecule::load_xyz_with_charge(xyz.to_str().unwrap(), 0, 1).expect("xyz");
    let label = format!(
        "{}/{basis_name}/{}",
        xyz.file_stem().unwrap().to_string_lossy(),
        xc.unwrap_or("HF")
    );
    let s = setup_from(label.clone(), mol, basis_name);

    // Gradient work runs without the (energy-only) final pass, as the
    // geometry drivers do.
    let cfg_c = scf_cfg(
        xc,
        Some(CosxConfig {
            final_grid: None,
            ..cosx
        }),
    );
    let r_c = solve(&s, &cfg_c, "cosx");
    let (g_c, g_old) = grads(&s, &cfg_c, &r_c);
    let fd_c = fd_gradient(&s, basis_name, &cfg_c, r_c.density_r());
    let res_new = &g_c - &fd_c;
    let res_old = &g_old - &fd_c;

    let (new, old) = if xc.is_some() {
        let cfg_d = scf_cfg(xc, None);
        let r_d = solve(&s, &cfg_d, "exact");
        let (g_d, _) = grads(&s, &cfg_d, &r_d);
        let res_d = &g_d - &fd_gradient(&s, basis_name, &cfg_d, r_d.density_r());
        eprintln!("{label}: exact-K run max|g - FD| = {:.3e}", max_abs(&res_d));
        (max_abs(&(&res_new - &res_d)), max_abs(&(&res_old - &res_d)))
    } else {
        (max_abs(&res_new), max_abs(&res_old))
    };
    eprintln!(
        "{label}: max|g_cosx - FD| = {new:.3e} (bar {TOL_GRAD_FD:.0e}); \
         exact-K gradient at the COSX density misses FD by {old:.3e} (must be >= {:.0e})",
        GRAD_MUST_MISS_FACTOR * TOL_GRAD_FD
    );
    assert!(
        new < TOL_GRAD_FD,
        "{label}: COSX gradient off FD of the COSX energy by {new:.3e}"
    );
    assert!(
        old >= GRAD_MUST_MISS_FACTOR * TOL_GRAD_FD,
        "{label}: NEGATIVE CONTROL blind — exact-K gradient is only {old:.3e} off FD"
    );
}

#[test]
#[ignore = "validation: COSX seminumerical exchange"]
fn cosx_gradient_matches_fd_water_augccpvdz_rhf_fitted() {
    fd_case(
        "testdata/molecules/validation/h2o_distorted.xyz",
        "aug-cc-pvdz",
        None,
        CosxConfig::default(),
    );
}

#[test]
#[ignore = "validation: COSX seminumerical exchange"]
fn cosx_gradient_matches_fd_nh3_def2svp_rhf_fitted() {
    fd_case(
        "testdata/molecules/validation/nh3_distorted.xyz",
        "def2-svp",
        None,
        CosxConfig::default(),
    );
}

/// Fitted COSX + KS gradients are refused (cosx_gradient.rs module doc), so
/// the hybrid runs with the fit OFF and the production screens.
#[test]
#[ignore = "validation: COSX seminumerical exchange"]
fn cosx_gradient_matches_fd_water_augccpvdz_b3lyp_nofit() {
    fd_case(
        "testdata/molecules/validation/h2o_distorted.xyz",
        "aug-cc-pvdz",
        Some("B3LYP"),
        CosxConfig {
            overlap_fit: false,
            ..CosxConfig::flat_reference()
        },
    );
}
