//! VALIDATION tier — VALIDATION.md rows "Geometry optimization RHF/RKS" and
//! "Geometry optimization UHF/ROHF".
//!
//! Runs only in the weekly `validation` CI job (and on demand):
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 cargo nextest run -p ferric-scf --test validation_geometry_optimization \
//!     --run-ignored only -E 'binary(/^validation_/)'
//! ```
//!
//! # What ferric computes (read from `crates/ferric-scf/src/optimize.rs`)
//!
//! `optimize_geometry` / `optimize_geometry_uhf` / `optimize_geometry_rohf`
//! run the shared Cartesian BFGS driver (`optimize_coordinates`): identity
//! initial inverse Hessian, no line search, the step clipped to
//! `trust_radius` (Bohr), a fresh SCF + analytic gradient at every point.
//! It stops when `|ΔE| < e_conv` AND `max|g| < g_max_thresh` AND
//! `rms(g) < g_rms_thresh`. The closed-shell KS gradient includes the grid
//! response; UHF/UKS go through `unrestricted_scf_gradient`, ROHF through
//! `rohf_gradient`.
//!
//! # References (`scripts/validation/gen_geometry_optimization.py`)
//!
//! A minimum is a property of the surface, not of the optimizer that finds
//! it. The reference is PySCF 2.13's analytic gradient (KS with
//! `grid_response = True`, the exact derivative of the grid energy, the same
//! quantity ferric differentiates) driven by scipy BFGS to
//! max|g| ≤ 1e-6 Ha/Bohr (recorded values 3e-9 .. 2e-7) from the SAME distorted
//! start geometry (`testdata/molecules/validation/<sys>_opt_start.xyz`).
//! geomeTRIC is not installed in the project venv, which is why scipy drives
//! it. PySCF was fed ferric's own basis JSON and Bohr geometry; J/K are EXACT
//! 4-index on both sides (`df_j_aux = df_k_aux = Some("")`; `None` would
//! auto-enable RI-J for closed-shell KS); KS grid (75,110) unpruned Becke with
//! Becke-1988 radii (ferric's default).
//!
//! * closed shell: H2O, NH3, CH2O × RHF/6-31G, B3LYP/6-31G, PBE/cc-pVDZ
//! * open shell:   HO2, CH3, NH2 × UHF, ROHF, UKS-PBE, all 6-31G. The start
//!   state is the lowest internally stable one over three guesses; every
//!   optimizer step follows it by density continuity with an ⟨S²⟩ jump guard,
//!   and the END state is stability-checked again (the generator restarts
//!   from a lower state and refuses an unstable end point).
//!
//! Compared: ALL interatomic distances (they fix the geometry up to a
//! reflection) and the listed bond angles, which are orientation-free, and
//! the optimized energy. Cartesians are NOT compared (the two optimizers can
//! end in different orientations).
//!
//! # Checks per system × method, in order
//!
//! 1. Harness: nuclear repulsion at the start geometry, AO count.
//! 2. LIKE-FOR-LIKE ANCHOR, no optimizer involved: ferric single point at the
//!    reference's optimized Cartesian geometry. Its energy must equal the
//!    reference's `energy_opt`, and ferric's own gradient there must be ~0.
//!    This proves the two codes share the minimum before any optimizer runs.
//! 3. Start-point energy (pins the state and the integrals at the start).
//! 4. ferric optimizes from the start geometry; must report converged.
//! 5. Stationarity: a FRESH single point at ferric's end geometry has
//!    max|g| below the bar (independent of the optimizer's own bookkeeping).
//! 6. Energy lower than at the start; optimized energy vs reference.
//! 7. Distances and angles vs reference.
//! 8. Open shell: ⟨S²⟩ (UHF/UKS) at the end vs reference; UHF/UKS end state
//!    STABLE by ferric's own stability analysis.
//!
//! # Physics hypothesis vs artifact hypothesis
//!
//! * If ferric is right: the anchor gradient sits at the integral/grid floor,
//!   the optimized internals agree to ~g_thresh/k ≈ 1e-5 Bohr (k ≈ 0.1–0.5
//!   Ha/Bohr² for bends/stretches), energies to the SCF floor.
//! * If the GRADIENT is wrong (e.g. the ROHF W-matrix class fixed in #163, a
//!   missing grid-response term): the anchor gradient at the reference minimum
//!   is non-zero by the size of the defect, and ferric converges to a
//!   different geometry — the anchor fails even if the optimizer is fine.
//! * If the OPTIMIZER is wrong or stops early (loose threshold, stalled BFGS):
//!   the anchor passes but the fresh-point stationarity check and the
//!   distance comparison fail.
//! * If an open-shell step lands on another STATE: ⟨S²⟩, the stability
//!   verdict, and the energy miss by orders of magnitude.
//! * If the HARNESS is broken (units, wrong file): nuclear repulsion or AO
//!   count fails first. A method-mislabelled file fails the method-swap
//!   controls below.
//! * Grid artifact (KS only): the (75,110) grid is not rotationally
//!   invariant, so two optimizers ending in slightly different orientations
//!   see slightly different surfaces. Expected at the ~1e-5 Bohr / 1e-8 Ha
//!   level; if the KS distance misses are systematically larger than HF's,
//!   this is the first suspect (check by re-running at a finer grid on both
//!   sides), not the gradient.
//!
//! # TOLERANCES
//!
//! Each bar is set from the measured maximum recorded on its const: distances
//! 2.6e-6 Bohr, angles 1.1e-4°, optimized energies 9.3e-13 Ha.
//!
//! Reference-side facts (generator run 2026-09-24):
//!
//! | quantity | value |
//! |---|---:|
//! | reference max abs gradient at its minimum | 3.2e-9 .. 1.9e-7 Ha/Bohr |
//! | start → optimum, max distance change | 0.039 (CH2O B3LYP) .. 0.269 (NH3 B3LYP) Bohr |
//! | smallest method separation (UHF vs ROHF, NH2) | 2.64e-3 Bohr |
//! | UHF vs ROHF, CH3 / HO2 | 4.34e-3 / 6.30e-3 Bohr |
//! | HF vs KS (any pair) | ≥ 4.97e-2 Bohr |
//!
//! # NEGATIVE CONTROLS (asserted inside the tests)
//!
//! * Optimizer moved: max |r_ij(end) − r_ij(start)| ≥ `MUST_MISS_FACTOR` ×
//!   `TOL_R`, for ferric AND for the reference, so the distance comparison is
//!   not satisfied by the start geometry.
//! * Method swap: ferric's optimized distances for method A must MISS the
//!   reference distances for every other method B of the same system by
//!   ≥ `MUST_MISS_FACTOR` × `TOL_R` (RHF vs B3LYP vs PBE; UHF vs ROHF vs UKS).
//!   UHF vs ROHF is the finest separation (2.6e-3 Bohr on NH2), so the
//!   distance bar demonstrably resolves two nearby, physically distinct
//!   minima.
//!
//! A missing reference JSON is a HARD failure (panic naming the path).

use std::path::{Path, PathBuf};

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::gradient::{restricted_scf_gradient, rohf_gradient, unrestricted_scf_gradient};
use ferric_scf::optimize::{
    optimize_geometry, optimize_geometry_rohf, optimize_geometry_uhf, OptimizeConfig,
    OptimizeResult,
};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::rohf::solve_rohf;
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::stability::StabilityVerdict;
use ferric_scf::uhf::solve_uhf;
use ferric_scf::{ScfResult, Spin};
use ndarray::Array2;
use serde_json::Value;

const ROW_DIR: &str = "testdata/reference/validation/geometry_optimization";
const MOL_DIR: &str = "testdata/molecules/validation";

const TOL_ENUC: f64 = 1e-9;
/// Single-point energy at a SHARED geometry (start, and the reference minimum).
/// Measured max 6.8e-13 Ha.
const TOL_E_SP: f64 = 1e-10;
/// ferric's max|g| at the REFERENCE minimum. Measured max 2.0e-7 Ha/Bohr,
/// which is the reference's own residual gradient (≤ 1.9e-7).
const TOL_G_AT_REF: f64 = 2e-6;
/// ferric's max|g| at its OWN end point, from a fresh single point. Measured
/// max 4.3e-7 Ha/Bohr (the optimizer stops at max|g| < 1e-6).
const TOL_G_END: f64 = 2e-6;
/// Optimized interatomic distances. Measured max 2.6e-6 Bohr.
const TOL_R: f64 = 2e-5;
/// Optimized bond angles. Measured max 1.1e-4 deg.
const TOL_ANG_DEG: f64 = 1e-3;
/// Optimized energy (ferric's minimum vs the reference's minimum). Measured
/// max 9.3e-13 Ha.
const TOL_E_OPT: f64 = 1e-10;
/// ⟨S²⟩ (UHF/UKS) at start and end. Measured max 4.9e-9.
const TOL_S2: f64 = 5e-8;
/// Distances that must be MISSED (moved-from-start, method swaps): this
/// factor × TOL_R. The closest pair is UHF vs ROHF on NH2 (2.64e-3 Bohr).
const MUST_MISS_FACTOR: f64 = 10.0;

// Optimizer and SCF convergence for the ferric side. ferric's BFGS took 6-8
// steps for UHF/ROHF and 18-54 for UKS-PBE; the reference needed 7-77 energy
// + gradient evaluations (scipy BFGS with a line search).
const OPT_G_MAX: f64 = 1e-6;
const OPT_G_RMS: f64 = 5e-7;
const OPT_E_CONV: f64 = 1e-10;
const OPT_MAX_STEPS: usize = 300;
const SCF_ENERGY_CONV: f64 = 1e-11;
const SCF_DENSITY_CONV: f64 = 1e-9;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    Rhf,
    Uhf,
    Rohf,
}

#[derive(Clone, Copy, Debug)]
struct Method {
    /// Block key in the reference JSON.
    key: &'static str,
    basis: &'static str,
    kind: Kind,
    xc: Option<&'static str>,
}

const CLOSED: [Method; 3] = [
    Method {
        key: "rhf",
        basis: "6-31g",
        kind: Kind::Rhf,
        xc: None,
    },
    Method {
        key: "rks_b3lyp",
        basis: "6-31g",
        kind: Kind::Rhf,
        xc: Some("B3LYP"),
    },
    Method {
        key: "rks_pbe",
        basis: "cc-pvdz",
        kind: Kind::Rhf,
        xc: Some("PBE"),
    },
];

const OPEN: [Method; 3] = [
    Method {
        key: "uhf",
        basis: "6-31g",
        kind: Kind::Uhf,
        xc: None,
    },
    Method {
        key: "rohf",
        basis: "6-31g",
        kind: Kind::Rohf,
        xc: None,
    },
    Method {
        key: "uks_pbe",
        basis: "6-31g",
        kind: Kind::Uhf,
        xc: Some("PBE"),
    },
];

/// Workspace root, found by walking up from the CWD (nextest sets the CWD to
/// the package dir); `CARGO_MANIFEST_DIR` is only a fallback because it is
/// baked in at compile time and is wrong inside a nextest archive.
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

/// Load a reference JSON. Missing or unparsable is a HARD failure.
fn reference(system: &str, basis_name: &str) -> Value {
    let path = workspace_root()
        .join(ROW_DIR)
        .join(format!("{system}_{basis_name}.json"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "missing validation reference {} ({e}); regenerate with \
             scripts/validation/gen_geometry_optimization.py — a missing reference is a \
             failure, never a skip",
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

fn arr<'a>(v: &'a Value, ptr: &str, ctx: &str) -> &'a Vec<Value> {
    v.pointer(ptr)
        .and_then(Value::as_array)
        .unwrap_or_else(|| panic!("{ctx}: reference field {ptr} missing or not an array"))
}

/// `[[i, j, r], ...]` → `Vec<(i, j, r)>`.
fn ref_distances(v: &Value, ptr: &str, ctx: &str) -> Vec<(usize, usize, f64)> {
    arr(v, ptr, ctx)
        .iter()
        .map(|e| {
            let e = e.as_array().expect("distance entry");
            (
                e[0].as_u64().unwrap() as usize,
                e[1].as_u64().unwrap() as usize,
                e[2].as_f64().unwrap(),
            )
        })
        .collect()
}

fn load_start(system: &str, r: &Value) -> Molecule {
    let charge = r["charge"].as_i64().expect("charge") as i32;
    let mult = r["multiplicity"].as_u64().expect("multiplicity") as usize;
    let xyz = workspace_root()
        .join(MOL_DIR)
        .join(format!("{system}_opt_start.xyz"));
    Molecule::load_xyz_with_charge(xyz.to_str().unwrap(), charge, mult)
        .unwrap_or_else(|e| panic!("{}: {e:?}", xyz.display()))
}

/// `template` with its Cartesian coordinates (Bohr) replaced.
fn with_coords(template: &Molecule, xyz_bohr: &[[f64; 3]]) -> Molecule {
    assert_eq!(template.atoms.len(), xyz_bohr.len(), "atom count");
    let mut m = template.clone();
    for (a, c) in m.atoms.iter_mut().zip(xyz_bohr) {
        a.x = c[0];
        a.y = c[1];
        a.zpos = c[2];
    }
    m
}

fn ref_geometry(v: &Value, ptr: &str, ctx: &str) -> Vec<[f64; 3]> {
    arr(v, ptr, ctx)
        .iter()
        .map(|row| {
            let row = row.as_array().expect("geometry row");
            [
                row[0].as_f64().unwrap(),
                row[1].as_f64().unwrap(),
                row[2].as_f64().unwrap(),
            ]
        })
        .collect()
}

fn pos(m: &Molecule, i: usize) -> [f64; 3] {
    let a = &m.atoms[i];
    [a.x, a.y, a.zpos]
}

fn dist(m: &Molecule, i: usize, j: usize) -> f64 {
    let (a, b) = (pos(m, i), pos(m, j));
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

fn angle_deg(m: &Molecule, i: usize, v: usize, k: usize) -> f64 {
    let (a, o, b) = (pos(m, i), pos(m, v), pos(m, k));
    let u = [a[0] - o[0], a[1] - o[1], a[2] - o[2]];
    let w = [b[0] - o[0], b[1] - o[1], b[2] - o[2]];
    let dot = u[0] * w[0] + u[1] * w[1] + u[2] * w[2];
    let nu = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt();
    let nw = (w[0] * w[0] + w[1] * w[1] + w[2] * w[2]).sqrt();
    (dot / (nu * nw)).clamp(-1.0, 1.0).acos().to_degrees()
}

/// Largest |r_ij(mol) − r_ij(reference list)|, index-checked.
fn max_distance_miss(m: &Molecule, want: &[(usize, usize, f64)]) -> f64 {
    let n = m.atoms.len();
    assert_eq!(want.len(), n * (n - 1) / 2, "reference lists every pair");
    want.iter()
        .map(|&(i, j, r)| (dist(m, i, j) - r).abs())
        .fold(0.0_f64, f64::max)
}

fn max_abs(g: &Array2<f64>) -> f64 {
    g.iter().map(|v| v.abs()).fold(0.0_f64, f64::max)
}

fn check_close(ctx: &str, what: &str, got: f64, want: f64, tol: f64) -> f64 {
    let d = (got - want).abs();
    eprintln!("{ctx}: {what:<20} ferric {got:+.12} ref {want:+.12} |d| {d:.2e} (tol {tol:.0e})");
    assert!(
        d < tol,
        "{ctx}: {what} {got:.12} vs reference {want:.12} (|d| {d:.2e} >= {tol:.0e})"
    );
    d
}

fn scf_config(m: &Method) -> RhfConfig {
    let unrestricted = m.kind == Kind::Uhf;
    RhfConfig {
        xc: m.xc.map(str::to_string),
        // Explicit EXACT J/K: `None` auto-enables RI-J for closed-shell KS.
        df_j_aux: Some(String::new()),
        df_k_aux: Some(String::new()),
        energy_conv: SCF_ENERGY_CONV,
        density_conv: SCF_DENSITY_CONV,
        max_iter: 500,
        // UHF/UKS end on a stable state, as the reference does. ferric's ROHF
        // skips stability by design (see the UHF/ROHF row); the reference is
        // ROHF-stable, so a ferric saddle shows as an energy/geometry miss.
        check_stability: unrestricted,
        scf_stability_descent: unrestricted,
        ..Default::default()
    }
}

fn opt_config() -> OptimizeConfig {
    OptimizeConfig {
        max_steps: OPT_MAX_STEPS,
        g_max_thresh: OPT_G_MAX,
        g_rms_thresh: OPT_G_RMS,
        e_conv: OPT_E_CONV,
        ..Default::default()
    }
}

/// One SCF + analytic gradient, through the same public entry points the
/// optimizer uses internally.
struct Point {
    energy: f64,
    grad: Array2<f64>,
    res: ScfResult,
    prep: PreparedBasis,
}

fn single_point(ctx: &str, m: &Method, mol: &Molecule) -> Point {
    let cfg = scf_config(m);
    let op = Operator::coulomb();
    let pctx = ParallelContext::default();
    let bs = basis::bundled(m.basis).unwrap();
    let prep = PreparedBasis::new(mol, &bs).unwrap();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let (res, grad) = match m.kind {
        Kind::Rhf => {
            let res = solve_rhf(&pctx, mol, &prep, op, &bounds, &cfg)
                .unwrap_or_else(|e| panic!("{ctx}: solve_rhf: {e:?}"));
            let g = restricted_scf_gradient(mol, &prep, &bs, op, &bounds, &cfg, &res)
                .unwrap_or_else(|e| panic!("{ctx}: gradient: {e:?}"));
            (res, g)
        }
        Kind::Uhf => {
            let res = solve_uhf(&pctx, mol, &prep, &bounds, &cfg)
                .unwrap_or_else(|e| panic!("{ctx}: solve_uhf: {e:?}"));
            let g = unrestricted_scf_gradient(mol, &prep, &bs, op, &bounds, &cfg, &res)
                .unwrap_or_else(|e| panic!("{ctx}: gradient: {e:?}"));
            (res, g)
        }
        Kind::Rohf => {
            assert!(m.xc.is_none(), "this row's ROHF is HF only");
            let res = solve_rohf(&pctx, mol, &prep, op, &bounds, &cfg)
                .unwrap_or_else(|e| panic!("{ctx}: solve_rohf: {e:?}"));
            assert!(matches!(res.spin, Spin::RestrictedOpen));
            let g = rohf_gradient(mol, &prep, op, &bounds, &res, None)
                .unwrap_or_else(|e| panic!("{ctx}: gradient: {e:?}"));
            (res, g)
        }
    };
    assert!(res.converged, "{ctx}: SCF not converged");
    Point {
        energy: res.energy,
        grad,
        res,
        prep,
    }
}

fn nocc_ab(mol: &Molecule) -> (usize, usize) {
    let nelec = mol.nelec() as usize;
    let two_s = mol.multiplicity - 1;
    ((nelec + two_s) / 2, (nelec - two_s) / 2)
}

/// ⟨S²⟩ of a UHF/UKS determinant from ferric's MOs and overlap.
fn s_squared(p: &Point, mol: &Molecule) -> f64 {
    let (na, nb) = nocc_ab(mol);
    let s = ferric_integrals::oneelectron::overlap(&p.prep);
    let sz = 0.5 * (na as f64 - nb as f64);
    let ca = p.res.mos_alpha.slice(ndarray::s![.., ..na]);
    let cb = p
        .res
        .mos_beta
        .as_ref()
        .unwrap()
        .slice(ndarray::s![.., ..nb]);
    let ov = ca.t().dot(&s).dot(&cb);
    sz * (sz + 1.0) + nb as f64 - ov.iter().map(|v| v * v).sum::<f64>()
}

/// UHF/UKS: ferric's own stability analysis must call the state STABLE.
fn assert_stable(ctx: &str, p: &Point) {
    let st = p
        .res
        .stability
        .as_ref()
        .unwrap_or_else(|| panic!("{ctx}: check_stability was set but no verdict came back"));
    eprintln!("{ctx}: {}", st.summary());
    assert_eq!(
        st.verdict(),
        StabilityVerdict::Stable,
        "{ctx}: ferric state is not STABLE: {}",
        st.summary()
    );
}

/// Run every check for one system × method. Returns ferric's optimized
/// geometry for the cross-method controls.
fn check_method(system: &str, m: &Method) -> Molecule {
    let r = reference(system, m.basis);
    let ctx = format!("{system}/{}/{}", m.basis, m.key);
    let blk = format!("/{}", m.key);
    assert!(
        r.pointer(&blk).is_some(),
        "{ctx}: reference file has no {} block",
        m.key
    );

    // 1. Harness.
    let start = load_start(system, &r);
    check_close(
        &ctx,
        "E_nuc (start)",
        start.nuclear_repulsion(),
        num(&r, "/nuclear_repulsion", &ctx),
        TOL_ENUC,
    );
    let bs = basis::bundled(m.basis).unwrap();
    let nao =
        ferric_integrals::oneelectron::overlap(&PreparedBasis::new(&start, &bs).unwrap()).nrows();
    assert_eq!(nao as u64, r["nao"].as_u64().unwrap(), "{ctx}: AO count");

    // 2. Like-for-like anchor at the REFERENCE minimum (no optimizer).
    let ref_min = with_coords(
        &start,
        &ref_geometry(&r, &format!("{blk}/geometry_opt_bohr"), &ctx),
    );
    check_close(
        &ctx,
        "E_nuc (ref minimum)",
        ref_min.nuclear_repulsion(),
        num(&r, &format!("{blk}/nuclear_repulsion_opt"), &ctx),
        TOL_ENUC,
    );
    let at_ref = single_point(&ctx, m, &ref_min);
    let e_opt_ref = num(&r, &format!("{blk}/energy_opt"), &ctx);
    check_close(&ctx, "E at ref minimum", at_ref.energy, e_opt_ref, TOL_E_SP);
    let g_at_ref = max_abs(&at_ref.grad);
    eprintln!(
        "{ctx}: ferric max|g| at the reference minimum {g_at_ref:.2e} (tol {TOL_G_AT_REF:.0e})"
    );
    assert!(
        g_at_ref < TOL_G_AT_REF,
        "{ctx}: the reference minimum is not stationary on ferric's surface: max|g| {g_at_ref:.2e}"
    );

    // 3. Start point: same state and integrals before the optimizer runs.
    let at_start = single_point(&ctx, m, &start);
    let e_start_ref = num(&r, &format!("{blk}/energy_start"), &ctx);
    check_close(&ctx, "E at start", at_start.energy, e_start_ref, TOL_E_SP);
    if m.kind == Kind::Uhf {
        assert_stable(&format!("{ctx} (start)"), &at_start);
        check_close(
            &ctx,
            "<S^2> start",
            s_squared(&at_start, &start),
            num(&r, &format!("{blk}/s_squared_start"), &ctx),
            TOL_S2,
        );
    }

    // 4. Optimize.
    let pctx = ParallelContext::default();
    let op = Operator::coulomb();
    let cfg = scf_config(m);
    let oc = opt_config();
    let res: OptimizeResult = match m.kind {
        Kind::Rhf => optimize_geometry(&pctx, &start, m.basis, op, &cfg, &oc),
        Kind::Uhf => optimize_geometry_uhf(&pctx, &start, m.basis, op, &cfg, &oc),
        Kind::Rohf => optimize_geometry_rohf(&pctx, &start, m.basis, op, &cfg, &oc),
    }
    .unwrap_or_else(|e| panic!("{ctx}: optimizer failed: {e:?}"));
    eprintln!(
        "{ctx}: ferric optimizer {} steps, {} evaluations, converged {} (reference: {} evaluations)",
        res.steps,
        res.energy_trace.len(),
        res.converged,
        num(
            &r,
            &format!("{blk}/optimizer/energy_gradient_evaluations"),
            &ctx
        )
    );
    assert!(
        res.converged,
        "{ctx}: optimizer did not converge in {OPT_MAX_STEPS} steps"
    );
    let end = res.mol.clone();

    // 5. Stationarity from a FRESH single point at ferric's end geometry.
    let at_end = single_point(&ctx, m, &end);
    let g_end = max_abs(&at_end.grad);
    eprintln!("{ctx}: ferric max|g| at its own end point {g_end:.2e} (tol {TOL_G_END:.0e})");
    assert!(
        g_end < TOL_G_END,
        "{ctx}: ferric's end point is not stationary: max|g| {g_end:.2e}"
    );

    // 6. Energies.
    assert!(
        res.energy < at_start.energy,
        "{ctx}: optimized energy {:.10} is not below the start energy {:.10}",
        res.energy,
        at_start.energy
    );
    check_close(&ctx, "E optimized", res.energy, e_opt_ref, TOL_E_OPT);
    check_close(
        &ctx,
        "E end (fresh SCF)",
        at_end.energy,
        res.energy,
        TOL_E_SP,
    );

    // 7. Internals.
    let d_ref = ref_distances(&r, &format!("{blk}/distances_opt_bohr"), &ctx);
    let d_miss = max_distance_miss(&end, &d_ref);
    eprintln!("{ctx}: max |dr| vs reference {d_miss:.2e} Bohr (tol {TOL_R:.0e})");
    assert!(
        d_miss < TOL_R,
        "{ctx}: optimized distances miss the reference by {d_miss:.3e} Bohr"
    );
    let mut a_miss = 0.0_f64;
    for a in arr(&r, &format!("{blk}/angles_opt_deg"), &ctx) {
        let a = a.as_array().unwrap();
        let (i, v, k) = (
            a[0].as_u64().unwrap() as usize,
            a[1].as_u64().unwrap() as usize,
            a[2].as_u64().unwrap() as usize,
        );
        let want = a[3].as_f64().unwrap();
        let got = angle_deg(&end, i, v, k);
        a_miss = a_miss.max((got - want).abs());
        assert!(
            (got - want).abs() < TOL_ANG_DEG,
            "{ctx}: angle {i}-{v}-{k} ferric {got:.6} vs reference {want:.6} deg"
        );
    }
    eprintln!("{ctx}: max |d angle| vs reference {a_miss:.2e} deg (tol {TOL_ANG_DEG:.0e})");

    // 8. Open-shell state at the end.
    if m.kind == Kind::Uhf {
        assert_stable(&format!("{ctx} (end)"), &at_end);
        check_close(
            &ctx,
            "<S^2> end",
            s_squared(&at_end, &end),
            num(&r, &format!("{blk}/s_squared_opt"), &ctx),
            TOL_S2,
        );
    }

    // NEGATIVE CONTROL: the optimizer moved, on both sides, by far more than
    // the distance bar — the comparison is not satisfied by the start.
    let d_start = ref_distances(&r, "/distances_start_bohr", &ctx);
    let must = MUST_MISS_FACTOR * TOL_R;
    let moved_ferric = max_distance_miss(&end, &d_start);
    let moved_ref = max_distance_miss(&ref_min, &d_start);
    eprintln!("{ctx}: moved from start: ferric {moved_ferric:.3e}, reference {moved_ref:.3e} Bohr");
    assert!(
        moved_ferric > must && moved_ref > must,
        "{ctx}: optimizer barely moved (ferric {moved_ferric:.2e}, reference {moved_ref:.2e} \
         Bohr <= {must:.0e}); the distance check cannot fail"
    );
    end
}

/// NEGATIVE CONTROL: ferric's geometry for method A must MISS the reference
/// geometry of every other method B of the same system.
fn assert_methods_discriminate(system: &str, methods: &[Method], ferric: &[Molecule]) {
    let must = MUST_MISS_FACTOR * TOL_R;
    for (a, mol_a) in methods.iter().zip(ferric) {
        for b in methods.iter().filter(|b| b.key != a.key) {
            let r = reference(system, b.basis);
            let ctx = format!("{system}: ferric {} vs reference {}", a.key, b.key);
            let d = max_distance_miss(
                mol_a,
                &ref_distances(&r, &format!("/{}/distances_opt_bohr", b.key), &ctx),
            );
            eprintln!("{ctx}: max |dr| {d:.3e} Bohr (must exceed {must:.0e})");
            assert!(
                d > must,
                "{ctx}: max |dr| {d:.3e} <= {must:.0e} — the distance bar cannot tell these \
                 methods apart"
            );
        }
    }
}

fn closed_row(system: &str) {
    let geoms: Vec<Molecule> = CLOSED.iter().map(|m| check_method(system, m)).collect();
    assert_methods_discriminate(system, &CLOSED, &geoms);
}

fn open_row(system: &str) {
    let geoms: Vec<Molecule> = OPEN.iter().map(|m| check_method(system, m)).collect();
    assert_methods_discriminate(system, &OPEN, &geoms);
}

#[test]
#[ignore = "validation: Geometry optimization RHF/RKS"]
fn closed_shell_h2o_optimization_vs_pyscf() {
    closed_row("h2o");
}

#[test]
#[ignore = "validation: Geometry optimization RHF/RKS"]
fn closed_shell_nh3_optimization_vs_pyscf() {
    closed_row("nh3");
}

#[test]
#[ignore = "validation: Geometry optimization RHF/RKS"]
fn closed_shell_ch2o_optimization_vs_pyscf() {
    closed_row("ch2o");
}

#[test]
#[ignore = "validation: Geometry optimization UHF/ROHF"]
fn open_shell_ho2_optimization_vs_pyscf() {
    open_row("ho2");
}

#[test]
#[ignore = "validation: Geometry optimization UHF/ROHF"]
fn open_shell_ch3_optimization_vs_pyscf() {
    open_row("ch3");
}

#[test]
#[ignore = "validation: Geometry optimization UHF/ROHF"]
fn open_shell_nh2_optimization_vs_pyscf() {
    open_row("nh2");
}
