//! VALIDATION tier — VALIDATION.md row "cDFT constrained state selection".
//!
//! Runs only in the weekly `validation` CI job (and on demand):
//!
//! ```text
//! cargo nextest run -p ferric-scf --test validation_cdft_state \
//!     --run-ignored only -E 'binary(/^validation_/)'
//! ```
//!
//! # The question
//!
//! At an over-constrained Becke target the constrained UHF problem has SEVERAL
//! solutions and the starting point picks one. Started from NWChem's converged
//! orbitals, does ferric's constrained solve STAY on NWChem's state,
//! reproducing E, λ and the population — when both codes impose the SAME
//! constraint?
//!
//! Systems (references: `scripts/validation/gen_cdft_state.py` ->
//! `testdata/reference/validation/cdft_state/<system>_def2-svp.json`, NWChem
//! inputs in `scripts/validation/nwchem/cdft_state/`):
//!
//! | system | fragment | target | NWChem states |
//! |---|---|---|---|
//! | HeNe⁺, R = 2.0 Å | He | 2.000 (integer; natural 1.954) | low (atomic guess) / high (hcore) |
//! | LiH⁺, R = 3.0 Å | Li | 2.800 (natural ~2.1) | low (atomic) / high (atomic + swap α 2↔3), 2.32 eV apart |
//!
//! # The Becke partition differs for He/Ne, and that WAS the 0.563 eV
//!
//! Both codes use Becke cells with the Bragg–Slater size adjustment, which
//! depends only on the radius ratio. The tables agree for H/Li but not for
//! the noble gases: ferric He/Ne = 0.30/0.45 Å, NWChem 0.35/0.50 Å. So at
//! "N_He = 2.000" the codes constrain DIFFERENT populations. MEASURED
//! (2026-09-30): NWChem's native LOW density has N_He = 2 − 8.754e-3 under
//! ferric's W (and 2.0000016 / 1.9912457 under the NWChem / ferric radius pair
//! in an independent PySCF-grid quadrature, generator `pyscf_checks`). At
//! λ ≈ −2.3 that offset is the whole historical "0.563 eV above NWChem LOW";
//! a first-order λ·δ correction cannot repair it (the operator SHAPE differs,
//! not just the target: the corrected residual was still 7.3e-4 Ha).
//!
//! The generator therefore also runs NWChem in FERRIC's partition (atoms
//! retagged C/Be, whose NWChem radii have ferric's ratio, nuclear charges
//! restored; relabeled unconstrained E equals native to 1e-9), seeded from its
//! own native orbitals. That variant (`ferric_partition`) is the like-for-like
//! reference for HeNe⁺; for LiH⁺ the native runs already are (`like_for_like`
//! in each state block names which). The native HeNe⁺ numbers are kept as the
//! negative control: ferric must MISS them.
//!
//! The residual W-QUADRATURE difference (NWChem home-atom points on its own
//! radial nodes vs ferric all-points) remains: δ = N_ferric[D_NWChem] − target
//! is ~1e-6, and the energy comparison is `E_NWChem + λ_NWChem·δ`
//! (dE/dN_target = −λ in both codes, first order, valid at this δ).
//!
//! # Exactness anchor (run first)
//!
//! `ao_map_and_one_shot_energy_anchor`: ferric's overlap equals PySCF's (stored
//! in ferric order); NWChem's orbitals are orthonormal in ferric's metric;
//! ferric's ONE-SHOT UHF energy of NWChem's UNCONSTRAINED orbitals equals
//! NWChem's (no SCF, no grid: integrals + AO map only), and of each constrained
//! state's orbitals equals that state's extrapolated E0. Also: ferric's W
//! applied to NWChem's densities must reproduce the INDEPENDENT PySCF-grid
//! ferric-radius population, which pins the partition story with a third code.
//! Negative controls: undoing the d(m=+1) sign map must miss (HeNe⁺ only; LiH⁺
//! has no d shells); the native-partition HeNe⁺ density must be ≥ 1e-3 off
//! ferric's target.
//!
//! # Physics hypothesis vs artifact hypothesis
//!
//! * PHYSICS (same functional + same constraint, guess decides the state):
//!   seeded at a like-for-like NWChem state (its orbitals and λ) ferric
//!   converges to E, λ matching that state at the W-quadrature floor; the two
//!   seeds give two different answers, each missing the other by the state gap;
//!   ferric's unseeded path with the descent OFF lands on NWChem high and with
//!   the descent ON on NWChem low (HeNe⁺).
//! * ARTIFACT (partition): comparing against the native HeNe⁺ references
//!   instead must MISS by ~2e-2 Ha — asserted, so the partition dependence
//!   stays visible.
//! * ARTIFACT (seeding inert): both seeds would return the same number.
//!
//! # Saddles
//!
//! NWChem high is a SADDLE of ferric's λ-augmented functional in both systems
//! (λ_min −4.0e-2 HeNe⁺, −1.28e-1 LiH⁺, measured 2026-09-30). Seeded with its
//! orbitals AND λ ferric stays on it (a fixed point); seeded with orbitals only
//! (λ from 0) LiH⁺ high slid to low (measured), because the intermediate-λ
//! inner solves leave a saddle's basin. So orbitals-only seeding is asserted
//! for LOW states and only measured for HIGH ones.
//!
//! # TOLERANCES
//!
//! Each `TOL_*` records its measured value (2026-10-01, release). The bars sit
//! orders of magnitude below the gaps between states (≥ 0.024 Ha on HeNe⁺,
//! 0.085 Ha on LiH⁺). λ on HeNe⁺ is steep, so its bar (1e-4) is set from the
//! measured 3.2e-5; LiH⁺ λ agrees to 2.9e-7. Precision note:
//! libint precision 1e-14 on this branch (PR #226 moves it to 1e-20 and changed
//! HeNe⁺ cDFT outer-loop convergence, issue #225); re-measure after #226.

use std::path::{Path, PathBuf};

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_dft::ao_grid::eval_basis_on_points;
use ferric_dft::cdft::{build_weight_matrix, population, Constraint, SpinChannel};
use ferric_dft::grid::{build_atomic_grid, AtomicGridConfig};
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::oneelectron;
use ferric_integrals::operator::Operator;
use ferric_scf::cdft_driver::{solve_cdft_uhf, solve_cdft_uhf_seeded, CdftResult, CdftSeed};
use ferric_scf::rhf::{build_jk, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::stability::{uhf_internal_stability, StabilityConfig};
use ferric_scf::uhf_newton::UhfNewtonInputs;
use ndarray::Array2;
use serde_json::Value;

const ROW_DIR: &str = "testdata/reference/validation/cdft_state";
const MOL_DIR: &str = "testdata/molecules/validation";
const BASIS: &str = "def2-svp";
const SYSTEMS: [&str; 2] = ["hene", "lih_r3"];
const STATES: [&str; 2] = ["low", "high"];
const HARTREE_EV: f64 = 27.211_386_245_988;

const TOL_ENUC: f64 = 1e-9;
/// ferric overlap vs PySCF's (ferric order). MEASURED 9.99e-16 (HeNe⁺).
const TOL_OVERLAP: f64 = 1e-10;
/// max|CᵀSC − I| of NWChem's orbitals in ferric's metric. MEASURED ≤ 2.4e-14.
const TOL_ORTHO: f64 = 1e-9;
/// One-shot UHF energy of NWChem's UNCONSTRAINED orbitals vs NWChem.
/// MEASURED 3.17e-11 (HeNe⁺, ferric); PySCF 1.4e-12 (LiH⁺).
const TOL_E_ONESHOT_UNC: f64 = 1e-8;
/// One-shot UHF energy of a constrained state's eps=1e-7 orbitals vs E0.
/// MEASURED 4.67e-10 (HeNe⁺ native low, ferric) — the largest eps second
/// difference (8.1e-10 Ha, NWChem SCF noise) is on that state too.
const TOL_E_ONESHOT_STATE: f64 = 1e-8;
/// |δ| of a LIKE-FOR-LIKE NWChem density in ferric's W (W quadrature only).
/// LiH⁺ MEASURED 2.0e-7 / 2.2e-7; HeNe⁺ ferric-partition expected ~2e-6.
const TOL_W_OFFSET: f64 = 1e-5;
/// ferric W on NWChem's density vs the independent PySCF-grid ferric-radius
/// population (two quadratures of one integral).
const TOL_POP_CROSS: f64 = 1e-5;
/// The NATIVE HeNe⁺ density must be at least this far off ferric's target
/// (measured 8.754e-3): the partition mismatch must stay visible.
const MUST_MISS_NATIVE_PARTITION_POP: f64 = 1e-3;
/// Seeded ferric E vs E_NW + λ_NW·δ (plan: 1e-6). MEASURED ≤ 7.7e-10 (HeNe⁺ high), 3.0e-11 (LiH⁺).
const TOL_E_SEEDED: f64 = 1e-8;
/// Seeded ferric λ vs NWChem λ (matched grid). Plan 1e-6. LiH⁺ MEASURED
/// 2.9e-7. HeNe⁺ is steeper: ferric −2.439011 vs NWChem −2.439043 (3.2e-5)
/// before any δ correction, and NWChem's own λ moves 1.3e-6 between its two
/// grids — so the HeNe⁺ λ floor is the W quadrature times dλ/dN (~4).
const TOL_LAMBDA: f64 = 1e-4;
/// Population vs target after a seeded solve.
const TOL_POP: f64 = 1e-8;
/// A seeded answer must miss the OTHER state (and, on HeNe⁺, the native-
/// partition reference of its own state) by at least this.
const MUST_MISS: f64 = 1000.0 * TOL_E_SEEDED;

// Solver knobs: the HeNe⁺ lane's level shift, tight convergence.
const LEVEL_SHIFT: f64 = 0.5;
const CDFT_LAMBDA_TOL: f64 = 1e-9;
const CDFT_MAX_OUTER: usize = 40;

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

/// Missing or unparsable reference is a HARD failure.
fn reference(system: &str) -> Value {
    let path = workspace_root()
        .join(ROW_DIR)
        .join(format!("{system}_{BASIS}.json"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "missing validation reference {} ({e}); regenerate with \
             scripts/validation/gen_cdft_state.py — a missing reference is a failure, never a skip",
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

fn matrix(v: &Value, ptr: &str, ctx: &str) -> Array2<f64> {
    let rows = v
        .pointer(ptr)
        .and_then(Value::as_array)
        .unwrap_or_else(|| panic!("{ctx}: reference matrix {ptr} missing"));
    let nr = rows.len();
    let nc = rows[0].as_array().expect("row").len();
    let mut m = Array2::<f64>::zeros((nr, nc));
    for (i, r) in rows.iter().enumerate() {
        let r = r.as_array().expect("row");
        assert_eq!(r.len(), nc, "{ctx}: ragged matrix {ptr}");
        for (j, x) in r.iter().enumerate() {
            m[(i, j)] = x.as_f64().expect("number");
        }
    }
    m
}

struct System {
    tag: String,
    r: Value,
    mol: Molecule,
    bs: basis::BasisSet,
    prep: PreparedBasis,
    bounds: SchwarzBounds,
    ctx: ParallelContext,
    fragment: Vec<usize>,
    target: f64,
    nocc_a: usize,
    nocc_b: usize,
    /// The reference variants stored for each state ("native" and, for
    /// HeNe⁺, "ferric_partition").
    variants: Vec<String>,
}

fn load_system(system: &str) -> System {
    let r = reference(system);
    let tag = format!("{system}/{BASIS}");
    let charge = r["charge"].as_i64().expect("charge") as i32;
    let mult = r["multiplicity"].as_u64().expect("multiplicity") as usize;
    let xyz_name = r
        .pointer("/provenance/geometry_xyz")
        .and_then(Value::as_str)
        .expect("provenance.geometry_xyz");
    let xyz = workspace_root().join(xyz_name);
    assert!(
        xyz.starts_with(workspace_root().join(MOL_DIR)),
        "{tag}: geometry {xyz_name} is not under {MOL_DIR}"
    );
    let mol = Molecule::load_xyz_with_charge(xyz.to_str().unwrap(), charge, mult)
        .unwrap_or_else(|e| panic!("{}: {e:?}", xyz.display()));
    let enuc_ref = num(&r, "/nuclear_repulsion", &tag);
    let enuc = mol.nuclear_repulsion();
    assert!(
        (enuc - enuc_ref).abs() < TOL_ENUC,
        "{tag}: nuclear repulsion {enuc:.12} vs NWChem {enuc_ref:.12} — geometry/unit mismatch"
    );
    let bs = basis::bundled(BASIS).unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    assert_eq!(
        prep.nbasis(),
        r["nao"].as_u64().expect("nao") as usize,
        "{tag}: AO count differs from the reference's"
    );
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
    let fragment: Vec<usize> = r["fragment"]
        .as_array()
        .expect("fragment")
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let target = num(&r, "/target", &tag);
    let nocc: Vec<usize> = r["nocc"]
        .as_array()
        .expect("nocc")
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let nelec = mol.nelec() as usize;
    let two_s = mol.multiplicity - 1;
    assert_eq!(
        (nocc[0], nocc[1]),
        ((nelec + two_s) / 2, (nelec - two_s) / 2),
        "{tag}: occupation counts disagree with the molecule"
    );
    let mut variants = vec!["native".to_string()];
    if r.pointer("/states/low/ferric_partition").is_some() {
        variants.push("ferric_partition".to_string());
    }
    System {
        tag,
        r,
        mol,
        bs,
        prep,
        bounds,
        ctx: ParallelContext::default(),
        fragment,
        target,
        nocc_a: nocc[0],
        nocc_b: nocc[1],
        variants,
    }
}

/// Which stored variant of `state` is the like-for-like reference.
fn like_for_like(s: &System, state: &str) -> String {
    s.r.pointer(&format!("/states/{state}/like_for_like"))
        .and_then(Value::as_str)
        .unwrap_or_else(|| panic!("{}: states/{state}/like_for_like missing", s.tag))
        .to_string()
}

/// NWChem orbitals (ferric AO order, occupied first). `base` is
/// "/unconstrained" or "/states/<state>/<variant>".
fn nwchem_mos(s: &System, base: &str) -> (Array2<f64>, Array2<f64>) {
    let ca = matrix(&s.r, &format!("{base}/alpha/coefficients"), &s.tag);
    let cb = matrix(&s.r, &format!("{base}/beta/coefficients"), &s.tag);
    let n = s.prep.nbasis();
    assert_eq!(ca.dim(), (n, n), "{}: {base} alpha MO shape", s.tag);
    assert_eq!(cb.dim(), (n, n), "{}: {base} beta MO shape", s.tag);
    (ca, cb)
}

fn state_base(state: &str, variant: &str) -> String {
    format!("/states/{state}/{variant}")
}

/// NWChem's eps→0 constrained (E, λ) for a state/variant on a grid.
fn nwchem_state(s: &System, state: &str, variant: &str, grid: &str) -> (f64, f64) {
    let b = state_base(state, variant);
    (
        num(&s.r, &format!("{b}/{grid}/energy_extrapolated"), &s.tag),
        num(&s.r, &format!("{b}/{grid}/lambda_extrapolated"), &s.tag),
    )
}

fn density(c: &Array2<f64>, nocc: usize) -> Array2<f64> {
    let occ = c.slice(ndarray::s![.., ..nocc]);
    occ.dot(&occ.t())
}

fn trace_prod(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    (a * b).sum()
}

/// ferric's UHF energy of the given orbitals, with NO SCF: one J/K build per
/// spin. `E = Σ_σ ½ Tr[D_σ (h + F_σ)] + E_nn`, `F_σ = h + J[D_α+D_β] − K[D_σ]`.
fn one_shot_uhf_energy(s: &System, ca: &Array2<f64>, cb: &Array2<f64>) -> f64 {
    let n = s.prep.nbasis();
    let thresh = RhfConfig::default().integral_thresh;
    let h = oneelectron::hcore(&s.prep);
    let da = density(ca, s.nocc_a);
    let db = density(cb, s.nocc_b);
    let dt = &da + &db;
    let jk = |d: &Array2<f64>| {
        let mut j = Array2::<f64>::zeros((n, n));
        let mut k = Array2::<f64>::zeros((n, n));
        build_jk(&s.ctx, &s.prep, &s.bounds, thresh, d, &mut j, &mut k).unwrap();
        (j, k)
    };
    let (j, _) = jk(&dt);
    let (_, ka) = jk(&da);
    let (_, kb) = jk(&db);
    let fa = &h + &j - &ka;
    let fb = &h + &j - &kb;
    0.5 * trace_prod(&da, &(&h + &fa))
        + 0.5 * trace_prod(&db, &(&h + &fb))
        + s.mol.nuclear_repulsion()
}

fn grid_cfg() -> AtomicGridConfig {
    AtomicGridConfig {
        n_radial: 99,
        n_angular: 302,
        ..Default::default()
    }
}

/// The driver's own W (same grid, same construction as `solve_cdft_uhf`).
fn weight_matrix(s: &System) -> Array2<f64> {
    let grid = build_atomic_grid(&s.mol, &grid_cfg());
    let pts: Vec<[f64; 3]> = grid.iter().map(|g| g.xyz).collect();
    let chi = eval_basis_on_points(&s.mol, &s.bs, &pts).unwrap();
    build_weight_matrix(&s.mol, &grid, &chi, &s.fragment)
}

fn ferric_population(s: &System, w: &Array2<f64>, ca: &Array2<f64>, cb: &Array2<f64>) -> f64 {
    population(
        w,
        &density(ca, s.nocc_a),
        &density(cb, s.nocc_b),
        &SpinChannel::Total,
    )
}

/// Constrained pure-UHF config for this row. Exact J/K, 99×302 W grid.
fn cdft_cfg(s: &System, descent: bool) -> RhfConfig {
    RhfConfig {
        constraints: vec![Constraint {
            fragment: s.fragment.clone(),
            target: s.target,
            spin: SpinChannel::Total,
        }],
        max_iter: 400,
        energy_conv: 1e-11,
        density_conv: 1e-9,
        level_shift: LEVEL_SHIFT,
        cdft_lambda_tol: CDFT_LAMBDA_TOL,
        cdft_max_outer: CDFT_MAX_OUTER,
        cdft_stability_descent: descent,
        use_sad_guess: false,
        dft_grid: Some(grid_cfg()),
        ..Default::default()
    }
}

/// How a seeded solve starts.
#[derive(Clone, Copy, Debug)]
enum Start {
    /// The like-for-like NWChem orbitals AND λ: a fixed-point check.
    FixedPoint,
    /// NWChem's NATIVE-partition orbitals, λ from 0: "start from what NWChem
    /// printed and see where ferric's own λ-Newton loop goes".
    NativeOrbitals,
}

fn seeded_solve(s: &System, state: &str, start: Start, descent: bool) -> CdftResult {
    let lfl = like_for_like(s, state);
    let variant = match start {
        Start::FixedPoint => lfl.as_str(),
        Start::NativeOrbitals => "native",
    };
    let (ca, cb) = nwchem_mos(s, &state_base(state, variant));
    let (_, lam_nw) = nwchem_state(s, state, &lfl, "matched");
    let lam0 = [lam_nw];
    let seed = CdftSeed {
        mos: Some((&ca, &cb)),
        lambdas: match start {
            Start::FixedPoint => Some(&lam0[..]),
            Start::NativeOrbitals => None,
        },
    };
    let cfg = cdft_cfg(s, descent);
    let r = solve_cdft_uhf_seeded(&s.ctx, &s.mol, &s.prep, &s.bs, &s.bounds, &cfg, seed)
        .unwrap_or_else(|e| panic!("{}: seeded {state} ({start:?}) cDFT failed: {e:?}", s.tag));
    assert!(
        r.scf.converged,
        "{}: seeded {state} ({start:?}) inner SCF not converged",
        s.tag
    );
    r
}

/// The like-for-like NWChem state's energy predicted at FERRIC's target:
/// `E_NW + λ_NW·δ`, δ = N_ferric[D_NW] − target. Returns (prediction, δ).
fn predicted_energy(s: &System, w: &Array2<f64>, state: &str) -> (f64, f64) {
    let lfl = like_for_like(s, state);
    let (ca, cb) = nwchem_mos(s, &state_base(state, &lfl));
    let delta = ferric_population(s, w, &ca, &cb) - s.target;
    let (e_nw, lam_nw) = nwchem_state(s, state, &lfl, "matched");
    (e_nw + lam_nw * delta, delta)
}

/// λ_min of the λ-augmented orbital Hessian at a converged constrained
/// solution (the driver's stability-descent operator: plain UHF Hessian with
/// the λ-augmented MO Focks).
fn augmented_lambda_min(s: &System, r: &CdftResult) -> (f64, String) {
    let ca = &r.scf.mos_alpha;
    let cb = r.scf.mos_beta.as_ref().expect("UHF beta MOs");
    let fa = &r.scf.fock_alpha;
    let fb = r.scf.fock_beta.as_ref().expect("UHF beta Fock");
    let fa_mo = ca.t().dot(fa).dot(ca);
    let fb_mo = cb.t().dot(fb).dot(cb);
    let inputs = UhfNewtonInputs {
        prep: &s.prep,
        bounds: &s.bounds,
        c_a: ca,
        c_b: cb,
        f_a_mo: &fa_mo,
        f_b_mo: &fb_mo,
        nocc_a: s.nocc_a,
        nocc_b: s.nocc_b,
        k_mix_sr: 1.0,
        rsh: None,
        fxc: None,
        thresh: RhfConfig::default().integral_thresh,
        ooc_budget: 0,
    };
    let res = uhf_internal_stability(&s.ctx, &inputs, &StabilityConfig::default())
        .unwrap_or_else(|e| panic!("{}: stability analysis failed: {e:?}", s.tag));
    (res.lowest_eigenvalue, format!("{:?}", res.verdict()))
}

fn check_close(ctx: &str, what: &str, got: f64, want: f64, tol: f64) {
    let d = (got - want).abs();
    eprintln!("{ctx}: {what:<30} ferric {got:+.12} ref {want:+.12} |d| {d:.2e} (tol {tol:.0e})");
    assert!(
        d < tol,
        "{ctx}: {what}: ferric {got:.12} vs reference {want:.12}, |d| {d:.3e} >= {tol:.1e}"
    );
}

/// Check a seeded result against the like-for-like reference of `state`, and
/// that it MISSES the other state (and the HeNe⁺ native-partition number).
fn assert_on_state(s: &System, ctx: &str, r: &CdftResult, state: &str, pred: &[(f64, f64)]) {
    let k = STATES.iter().position(|x| *x == state).unwrap();
    let lfl = like_for_like(s, state);
    let (e_pred, delta) = pred[k];
    let (_, lam_nw) = nwchem_state(s, state, &lfl, "matched");
    let (_, lam_conv) = nwchem_state(s, state, &lfl, "converged");
    eprintln!(
        "{ctx}: outer {} N_C {:.12} δ_NW {delta:+.3e} λ(NWChem 300x974) {lam_conv:+.10}",
        r.outer_iters, r.populations[0]
    );
    check_close(ctx, "population", r.populations[0], s.target, TOL_POP);
    check_close(
        ctx,
        "E vs E_NW + λ_NW·δ",
        r.scf.energy,
        e_pred,
        TOL_E_SEEDED,
    );
    check_close(ctx, "λ vs NWChem 99x302", r.lambdas[0], lam_nw, TOL_LAMBDA);
    let other = pred[1 - k].0;
    let miss = (r.scf.energy - other).abs();
    assert!(
        miss > MUST_MISS,
        "{ctx}: also matches the OTHER NWChem state ({miss:.2e} Ha) — not discriminating"
    );
    if lfl != "native" {
        let (e_native, _) = nwchem_state(s, state, "native", "matched");
        let miss = (r.scf.energy - e_native).abs();
        eprintln!("{ctx}: [partition control] |E − E_NWChem native partition| {miss:.3e} Ha");
        assert!(
            miss > MUST_MISS,
            "{ctx}: matches NWChem's NATIVE-partition number too ({miss:.2e} Ha) — the \
             test cannot see the partition"
        );
    }
}

// ---------------------------------------------------------------------------
// 1. Exactness anchor: AO map, one-shot energies, partition
// ---------------------------------------------------------------------------

/// THE ANCHOR. No constrained solve: if this fails, nothing below is
/// interpretable.
#[test]
#[ignore = "validation: cDFT constrained state selection"]
fn ao_map_and_one_shot_energy_anchor() {
    for system in SYSTEMS {
        let s = load_system(system);
        let ctx = s.tag.clone();

        // (a) ferric's overlap == PySCF's in ferric order (the map's base).
        let s_ferric = oneelectron::overlap(&s.prep);
        let s_ref = matrix(&s.r, "/overlap_pyscf_ferric_order", &ctx);
        let d_ovl = (&s_ferric - &s_ref)
            .iter()
            .fold(0.0_f64, |m, x| m.max(x.abs()));
        eprintln!("{ctx}: max|S_ferric - S_pyscf| {d_ovl:.2e}");
        assert!(d_ovl < TOL_OVERLAP, "{ctx}: overlap mismatch {d_ovl:.2e}");

        // (b) every imported orbital set is orthonormal in ferric's metric.
        let ortho = |c: &Array2<f64>| {
            let m = c.t().dot(&s_ferric).dot(c);
            let mut d = 0.0_f64;
            for i in 0..m.nrows() {
                for j in 0..m.ncols() {
                    let e = if i == j { 1.0 } else { 0.0 };
                    d = d.max((m[(i, j)] - e).abs());
                }
            }
            d
        };
        let mut bases = vec!["/unconstrained".to_string()];
        for st in STATES {
            for v in &s.variants {
                bases.push(state_base(st, v));
            }
        }
        for b in &bases {
            let (ca, cb) = nwchem_mos(&s, b);
            let d = ortho(&ca).max(ortho(&cb));
            eprintln!("{ctx}: {b:<32} max|CᵀSC − I| {d:.2e}");
            assert!(
                d < TOL_ORTHO,
                "{ctx}: {b} orbitals not orthonormal: {d:.2e}"
            );
        }

        // (c) one-shot UHF energy of NWChem's UNCONSTRAINED orbitals.
        let (ca_u, cb_u) = nwchem_mos(&s, "/unconstrained");
        let e_unc_ref = num(&s.r, "/unconstrained/energy", &ctx);
        check_close(
            &ctx,
            "one-shot E(unconstrained)",
            one_shot_uhf_energy(&s, &ca_u, &cb_u),
            e_unc_ref,
            TOL_E_ONESHOT_UNC,
        );

        // (d) per state and variant: one-shot E vs E0; ferric's population of
        // the NWChem density vs the independent PySCF-grid ferric-radius one.
        let w = weight_matrix(&s);
        for st in STATES {
            let lfl = like_for_like(&s, st);
            for v in &s.variants {
                let tag = format!("{st}/{v}");
                let (ca, cb) = nwchem_mos(&s, &state_base(st, v));
                let (e0, _) = nwchem_state(&s, st, v, "matched");
                check_close(
                    &ctx,
                    &format!("one-shot E({tag})"),
                    one_shot_uhf_energy(&s, &ca, &cb),
                    e0,
                    TOL_E_ONESHOT_STATE,
                );
                let n_f = ferric_population(&s, &w, &ca, &cb);
                // JSON-pointer key "low/native": the '/' is escaped as "~1".
                let n_pyscf = num(
                    &s.r,
                    &format!("/pyscf_checks/{st}~1{v}/becke_population/ferric_radii"),
                    &ctx,
                );
                check_close(
                    &ctx,
                    &format!("N_frag({tag}) vs PySCF grid"),
                    n_f,
                    n_pyscf,
                    TOL_POP_CROSS,
                );
                let delta = n_f - s.target;
                eprintln!("{ctx}: {tag:<22} δ = N_ferric[D_NWChem] − target = {delta:+.3e}");
                if *v == lfl {
                    assert!(
                        delta.abs() < TOL_W_OFFSET,
                        "{ctx}: {tag} (like-for-like) density is {delta:+.3e} off ferric's \
                         target — more than a W-quadrature difference"
                    );
                } else {
                    // NEGATIVE CONTROL: the native-partition density must sit
                    // visibly off ferric's target, else the partition story is
                    // wrong (or the two tables were made to agree).
                    assert!(
                        delta.abs() > MUST_MISS_NATIVE_PARTITION_POP,
                        "{ctx}: {tag} is only {delta:+.3e} off ferric's target — the \
                         He/Ne Becke-radius mismatch is no longer visible"
                    );
                }
            }
        }

        // (e) NEGATIVE CONTROL for the AO map: undo the d(m=+1) sign flip.
        // Reachable only where d shells exist (HeNe⁺); LiH⁺/def2-SVP has none.
        let signs: Vec<f64> = s.r["nwchem_to_ferric_ao_sign"]
            .as_array()
            .expect("nwchem_to_ferric_ao_sign")
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        if signs.iter().any(|&x| x < 0.0) {
            let unflip = |c: &Array2<f64>| {
                let mut c = c.clone();
                for (i, &sg) in signs.iter().enumerate() {
                    c.row_mut(i).mapv_inplace(|x| x * sg);
                }
                c
            };
            let e_bad = one_shot_uhf_energy(&s, &unflip(&ca_u), &unflip(&cb_u));
            let miss = (e_bad - e_unc_ref).abs();
            eprintln!("{ctx}: [negative control] unflipped d(m=+1): |dE| {miss:.2e}");
            assert!(
                miss > 10.0 * TOL_E_ONESHOT_UNC,
                "{ctx}: removing the d(m=+1) sign map changed E by only {miss:.2e} — \
                 the anchor cannot see the AO map"
            );
        } else {
            eprintln!("{ctx}: [negative control] no d(m=+1) rows; map mutation unreachable here");
        }
    }
}

// ---------------------------------------------------------------------------
// 2. Seeded solves stay on NWChem's states
// ---------------------------------------------------------------------------

/// Fixed point: like-for-like NWChem orbitals AND λ, descent OFF. Both states
/// (the saddle included) must be reproduced, each missing the other.
fn seeded_fixed_points(system: &str) {
    let s = load_system(system);
    let w = weight_matrix(&s);
    let pred: Vec<(f64, f64)> = STATES
        .iter()
        .map(|st| predicted_energy(&s, &w, st))
        .collect();
    eprintln!(
        "{}: NWChem high − low = {:.4} eV (like-for-like, at ferric's target)",
        s.tag,
        (pred[1].0 - pred[0].0) * HARTREE_EV
    );
    for state in STATES {
        let ctx = format!("{} seeded {state} (fixed point)", s.tag);
        let r = seeded_solve(&s, state, Start::FixedPoint, false);
        assert_on_state(&s, &ctx, &r, state, &pred);
    }
}

#[test]
#[ignore = "validation: cDFT constrained state selection"]
fn seeded_hene_states_are_fixed_points() {
    seeded_fixed_points("hene");
}

#[test]
#[ignore = "validation: cDFT constrained state selection"]
fn seeded_lih_cation_states_are_fixed_points() {
    seeded_fixed_points("lih_r3");
}

/// From NWChem's NATIVE orbitals with λ from 0 (ferric's own λ-Newton path),
/// descent OFF. LOW must be reached on both systems (for HeNe⁺ these orbitals
/// come from the OTHER partition, so this is also "same physical state under
/// either constraint"). HIGH is a saddle and is only MEASURED: LiH⁺ high was
/// seen to slide to low.
#[test]
#[ignore = "validation: cDFT constrained state selection"]
fn native_orbital_seed_reaches_low() {
    for system in SYSTEMS {
        let s = load_system(system);
        let w = weight_matrix(&s);
        let pred: Vec<(f64, f64)> = STATES
            .iter()
            .map(|st| predicted_energy(&s, &w, st))
            .collect();
        let ctx = format!("{} seeded low (native orbitals, λ from 0)", s.tag);
        let r = seeded_solve(&s, "low", Start::NativeOrbitals, false);
        assert_on_state(&s, &ctx, &r, "low", &pred);
        let r = seeded_solve(&s, "high", Start::NativeOrbitals, false);
        eprintln!(
            "{} seeded high (native orbitals, λ from 0) [MEASURE]: E {:.10} λ {:+.8} \
             vs low {:+.3e} Ha, high {:+.3e} Ha",
            s.tag,
            r.scf.energy,
            r.lambdas[0],
            r.scf.energy - pred[0].0,
            r.scf.energy - pred[1].0
        );
    }
}

/// MEASUREMENT: ferric's λ-augmented internal stability at each like-for-like
/// NWChem state, and what the descent does from there. Asserted: a finite
/// λ_min, the descent never RAISES the energy, and from a state the analysis
/// calls UNSTABLE the descent reaches LOW (measured 2026-09-30 on both
/// systems: HeNe⁺ high λ_min −4.0e-2 → low; LiH⁺ high −1.28e-1 → low).
#[test]
#[ignore = "validation: cDFT constrained state selection"]
fn seeded_states_stability_and_descent() {
    for system in SYSTEMS {
        let s = load_system(system);
        let w = weight_matrix(&s);
        let e_low = predicted_energy(&s, &w, "low").0;
        for state in STATES {
            let ctx = format!("{} {state}", s.tag);
            let off = seeded_solve(&s, state, Start::FixedPoint, false);
            let (lmin, verdict) = augmented_lambda_min(&s, &off);
            eprintln!("{ctx}: λ-augmented λ_min {lmin:+.6e} ({verdict})");
            assert!(lmin.is_finite(), "{ctx}: no λ_min");
            let on = seeded_solve(&s, state, Start::FixedPoint, true);
            eprintln!(
                "{ctx}: descent ON  E {:.10} λ {:+.8}  (ΔE vs OFF {:+.3e} Ha = {:+.4} eV)",
                on.scf.energy,
                on.lambdas[0],
                on.scf.energy - off.scf.energy,
                (on.scf.energy - off.scf.energy) * HARTREE_EV
            );
            assert!(
                on.scf.energy <= off.scf.energy + 1e-9,
                "{ctx}: the descent RAISED the energy ({:.10} > {:.10})",
                on.scf.energy,
                off.scf.energy
            );
            if verdict == "Unstable" {
                check_close(
                    &ctx,
                    "descended E vs NWChem low",
                    on.scf.energy,
                    e_low,
                    TOL_E_SEEDED,
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 3. The unseeded paths, against the like-for-like states
// ---------------------------------------------------------------------------

/// ferric's unseeded HeNe⁺ path from hcore: descent OFF lands on NWChem HIGH
/// (the saddle, E −130.40219), descent ON on NWChem LOW (−130.42671) — both in
/// ferric's partition. This replaces the old "0.563 eV above NWChem LOW"
/// statement, which compared across partitions. It is also the row's
/// non-vacuity control: same code and constraint, two answers, each matching a
/// different NWChem state. MINAO is measured alongside.
#[test]
#[ignore = "validation: cDFT constrained state selection"]
fn unseeded_hene_paths_land_on_nwchem_states() {
    let s = load_system("hene");
    let w = weight_matrix(&s);
    let pred: Vec<(f64, f64)> = STATES
        .iter()
        .map(|st| predicted_energy(&s, &w, st))
        .collect();
    for (label, descent, expect) in [
        ("hcore, descent OFF", false, "high"),
        ("hcore, descent ON", true, "low"),
    ] {
        let ctx = format!("{} unseeded ({label})", s.tag);
        let cfg = cdft_cfg(&s, descent);
        let r = solve_cdft_uhf(&s.ctx, &s.mol, &s.prep, &s.bs, &s.bounds, &cfg)
            .unwrap_or_else(|e| panic!("{ctx}: {e:?}"));
        assert_on_state(&s, &ctx, &r, expect, &pred);
    }
    let cfg = RhfConfig {
        use_sad_guess: true,
        ..cdft_cfg(&s, true)
    };
    match solve_cdft_uhf(&s.ctx, &s.mol, &s.prep, &s.bs, &s.bounds, &cfg) {
        Ok(r) => eprintln!(
            "{} unseeded (MINAO, descent ON) [MEASURE]: E {:.10} λ {:+.8} vs low {:+.3e} Ha, \
             high {:+.3e} Ha",
            s.tag,
            r.scf.energy,
            r.lambdas[0],
            r.scf.energy - pred[0].0,
            r.scf.energy - pred[1].0
        ),
        Err(e) => eprintln!(
            "{} unseeded (MINAO, descent ON) [MEASURE]: did not converge ({e:?}); see issue #225",
            s.tag
        ),
    }
}

/// LiH⁺: the unseeded path (hcore and MINAO, descent ON) reaches NWChem LOW
/// (measured 3.0e-11 Ha, 2026-09-30).
#[test]
#[ignore = "validation: cDFT constrained state selection"]
fn unseeded_lih_cation_paths_land_on_low() {
    let s = load_system("lih_r3");
    let w = weight_matrix(&s);
    let pred: Vec<(f64, f64)> = STATES
        .iter()
        .map(|st| predicted_energy(&s, &w, st))
        .collect();
    for (label, sad) in [("hcore, descent ON", false), ("MINAO, descent ON", true)] {
        let ctx = format!("{} unseeded ({label})", s.tag);
        let cfg = RhfConfig {
            use_sad_guess: sad,
            ..cdft_cfg(&s, true)
        };
        let r = solve_cdft_uhf(&s.ctx, &s.mol, &s.prep, &s.bs, &s.bounds, &cfg)
            .unwrap_or_else(|e| panic!("{ctx}: {e:?}"));
        assert_on_state(&s, &ctx, &r, "low", &pred);
    }
}
