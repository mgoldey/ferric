//! VALIDATION tier — VALIDATION.md row "U-RPA" (open-shell PDEP-RPA
//! correlation energy).
//!
//! Runs only in the weekly `validation` CI job (and on demand):
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 cargo nextest run -p ferric-rpa --release \
//!     --test validation_urpa --run-ignored only -E 'binary(/^validation_/)'
//! ```
//!
//! # What is compared
//!
//! ferric `run_u_pdep_rpa` (spin-summed dielectric ε = I + Π_α + Π_β, full
//! rank) against PySCF 2.13.1 `pyscf.gw.urpa.URPA`, both on the same
//! exact-integral, stability-checked UHF state, fed ferric's own orbital and
//! RI-aux JSON and ferric's Bohr geometry (`scripts/validation/common.py`):
//!
//! | test | systems / basis (aux) | reference |
//! |---|---|---|
//! | open shell | OH, CH3, NH2 (doublets), O2 (³Σg⁻) / cc-pVDZ (cc-pvdz-ri); OH / aug-cc-pVDZ (aug-cc-pvdz-rifit) | `URPA` E_corr |
//! | closed-shell limit | H2O / cc-pVDZ, through BOTH `run_pdep_rpa` (RHF) and `run_u_pdep_rpa` (singlet UHF) | `RPA` and `URPA` on the same RHF (equal in PySCF to 0) |
//!
//! References: `scripts/validation/gen_urpa.py` →
//! `testdata/reference/validation/urpa/<system>_<basis>.json`.
//!
//! # Frequency grid: MATCHED, not converged (the design decision)
//!
//! PySCF's `rpa.kernel` integrates on `_get_scaled_legendre_roots(nw, x0)`:
//! nw Gauss–Legendre nodes mapped by ω = x0(1+x)/(1−x) with weight
//! w_GL·2x0/(1−x)². ferric's `QuadratureScheme::GaussLegendre` with `u0` is
//! the SAME map and the same weight (`quadrature::gauss_legendre_nodes`), so
//! both sides run the identical 40-point grid (PySCF's default `nw = 40`,
//! x0 = u0 = 0.5) and the comparison contains no quadrature error at all.
//! PySCF's NW-convergence (20/40/80/160) is recorded in the JSON as a
//! diagnostic only: E_c(160) − E_c(40) is ≤1.6e-9 Ha on every system here
//! (O2), E_c(20) − E_c(40) is 4.7e-7…5.3e-6 Ha.
//!
//! # The like-for-like recipe (read from the code)
//!
//! * Π_s(iω) = B_s diag(2Δ/(ω²+Δ²)) B_sᵀ, Δ = ε_a − ε_i per spin
//!   (`sternheimer::dielectric_matrix_unrestricted`); PySCF's `diel` is −Π
//!   (it carries e_ov = ε_i − ε_a < 0, f_ov = 1), and its
//!   ln det(I − diel) + tr(diel) is ferric's Σ_α [ln λ_α + 1 − λ_α].
//! * B (ferric) and L (PySCF) are both Cholesky-whitened with the same
//!   Coulomb metric, so they differ by an orthogonal aux rotation and the
//!   full-rank energy is identical. `trunc_thresh = 0` with the Lanczos arm is
//!   one dense eigh of ε(0) that keeps every mode (asserted:
//!   `n_eigenpotentials == naux`).
//! * UHF: exact 4-index integrals on both sides; the test asserts E_UHF
//!   before any RPA number. No frozen core on either side.
//!
//! # Physics hypothesis vs artifact hypothesis (per the Experimental Protocol)
//!
//! * If ferric's U-RPA is right: grid, rank, orbitals and aux all matched, so
//!   only the integral engines and SCF convergence differ; E_c agrees at the
//!   ~1e-10 Ha level.
//! * If a spin channel is dropped, doubled, or fed the other spin's orbital
//!   energies: E_c moves by 0.1 Ha (the α-only / β-only values are recorded
//!   and asserted MISSED). The closed-shell limit CANNOT see an α/β mix-up
//!   (α ≡ β there) — that is why the open-shell systems carry the row.
//! * If the prefactor is wrong (4 instead of 2 per spin, the closed-shell
//!   factor): the singlet-UHF path misses `run_pdep_rpa` by a large fraction
//!   of E_c.
//! * If the truncation or projection machinery is wrong: the truncation
//!   control misses its numpy emulation.
//! * If the HARNESS is wrong (geometry, basis, aux, a different UHF state):
//!   E_nuc, the AO/aux counts, or E_UHF fail first.
//!
//! # TOLERANCES
//!
//! Each bar is 3–25× the worst |d| measured over every system (2026-09-25):
//! E_c 9.2e-11 (bar 1e-9), truncated E_c 6.4e-11 (1e-9), U vs R path 3.2e-12
//! (3e-11), SCF 4.7e-12 (5e-11). The smallest control miss is 2.8e-6 Ha
//! (20- vs 40-point grid), 280× its 1e-8 must-miss bar.
//!
//! # NEGATIVE CONTROLS (asserted in the tests)
//!
//! * Spin channel: ferric's full E_c must MISS the α-only and β-only
//!   references (numpy on PySCF's per-spin Cholesky tensors).
//! * Quadrature: ferric at 20 points must MATCH PySCF at nw = 20 and MISS the
//!   40-point reference — proves `n_points` reaches the integral and the map
//!   is the same one.
//! * Truncation: ferric at `trunc_thresh = 1e-2` must keep exactly the
//!   emulated number of modes, MATCH the numpy emulation of ferric's
//!   truncation and MISS the full-rank reference (by 5.2e-4 Ha on OH,
//!   9.6e-4 Ha on O2).
//! * Reference orbitals: ferric's U-RPA on its (PySCF-energy-checked) ROHF
//!   state of OH must MISS the UHF-based reference.
//! * Basis: ferric OH/cc-pVDZ must MISS the OH/aug-cc-pVDZ reference.
//!
//! # MUTATIONS (run 2026-09-25)
//!
//! (A) `crates/ferric-rpa/src/sternheimer.rs`, `dielectric_matrix_unrestricted`:
//!     `for chan in [chan_a, chan_b]` → `[chan_a, chan_a]` (α twice, β dropped)
//!     fails all 8 open-shell tests. The closed-shell-limit test and the ROHF
//!     must-miss control pass: α ≡ β at the closed-shell limit, so that anchor
//!     cannot see an α/β mix-up.
//! (B) The same edit in `dielectric_apply_unrestricted` (the static ε(0) matvec
//!     that only builds the eigenbasis) is an identity at full rank: it fails
//!     exactly the two `urpa_truncation_control_*` tests and nothing else. The
//!     truncation controls are the only coverage of that function.
//! (C) `crates/ferric-rpa/src/lib.rs`, `rhf.eps_b()` → `rhf.eps_a()` (β
//!     denominators from α energies) fails all 8 open-shell tests.
//!
//! A missing reference JSON is a HARD failure (panic naming the path).

use std::path::{Path, PathBuf};

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_rpa::config::{Eigensolver, PdepRpaConfig, QuadratureConfig, QuadratureScheme};
use ferric_rpa::{run_pdep_rpa, run_u_pdep_rpa, PdepRpaResult};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::rohf::solve_rohf;
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::stability::StabilityVerdict;
use ferric_scf::uhf::solve_uhf;
use ferric_scf::ScfResult;
use serde_json::Value;

const ROW_DIR: &str = "testdata/reference/validation/urpa";
const MOL_DIR: &str = "testdata/molecules/validation";
/// Frequency points on both sides; PySCF `nw` in the generator.
const N_QUAD: usize = 40;
/// The quadrature control's point count (reference `nw_convergence/20`).
const N_QUAD_CONTROL: usize = 20;
/// Gauss–Legendre map scale; PySCF `x0`.
const U0: f64 = 0.5;
/// The truncation control's threshold (reference `truncated/0.01`). The
/// nearest static mode sits 1.3e-3 (OH) and 7.4e-4 (O2) from this cut — the
/// two systems the control runs on — so the kept-mode count cannot flip
/// between codes.
const TRUNC_CONTROL: f64 = 1e-2;

// Grid matched, full rank, same aux and UHF state: worst |d| 9.2e-11 Ha
// (OH/aug-cc-pVDZ); cc-pVDZ cases 6.1e-12..6.4e-11.
const TOL_E_C: f64 = 1e-9;
// ferric truncated vs the numpy emulation of ferric's truncation on PySCF's
// tensors: 6.4e-11 Ha (O2).
const TOL_E_C_TRUNC: f64 = 1e-9;
// U-RPA on a singlet UHF vs run_pdep_rpa on the RHF (two SCF solves of the
// same state): 3.2e-12 Ha.
const TOL_U_VS_R: f64 = 3e-11;
// SCF energies vs PySCF: 4.7e-12 Ha (H2O RHF); UHF states ≤ 1.8e-12.
const TOL_E_SCF: f64 = 5e-11;
const TOL_ENUC: f64 = 1e-9;
/// A reference ferric must MISS is missed by at least this multiple of the bar.
const MUST_MISS_FACTOR: f64 = 10.0;

// ---------------------------------------------------------------------------
// Reference plumbing
// ---------------------------------------------------------------------------

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
        .expect("ferric-rpa manifest dir should be <root>/crates/ferric-rpa")
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
             scripts/validation/gen_urpa.py — a missing reference is a failure, never a skip",
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

fn check_close(ctx: &str, what: &str, got: f64, want: f64, tol: f64) -> f64 {
    let d = (got - want).abs();
    eprintln!("{ctx}: {what:<34} ferric {got:+.12} ref {want:+.12} |d| {d:.2e}");
    assert!(
        d < tol,
        "{ctx}: {what}: ferric {got:.12} vs reference {want:.12} (|d| {d:.2e} Ha) exceeds {tol:.1e} Ha"
    );
    d
}

/// |got − want| must exceed `factor × tol`: the comparison responds to the
/// thing the control changes.
fn assert_misses(ctx: &str, what: &str, got: f64, want: f64, tol: f64) {
    let d = (got - want).abs();
    eprintln!(
        "{ctx}: control {what}: |d| {d:.3e} Ha, must exceed {:.1e}",
        MUST_MISS_FACTOR * tol
    );
    assert!(
        d > MUST_MISS_FACTOR * tol,
        "{ctx}: negative control '{what}' did not miss: |d| {d:.3e} Ha <= {MUST_MISS_FACTOR} x \
         {tol:.1e} — the comparison does not respond to what the control changes"
    );
}

// ---------------------------------------------------------------------------
// System setup
// ---------------------------------------------------------------------------

struct Sys {
    label: String,
    r: Value,
    mol: Molecule,
    obs: PreparedBasis,
    dfbs: PreparedBasis,
    bounds: SchwarzBounds,
    ctx: ParallelContext,
}

fn load_system(system: &str, basis_name: &str) -> Sys {
    let r = reference(system, basis_name);
    let label = format!("{system}/{basis_name}");
    let charge = r["charge"].as_i64().expect("charge") as i32;
    let mult = r["multiplicity"].as_u64().expect("multiplicity") as usize;
    let xyz = workspace_root().join(MOL_DIR).join(format!("{system}.xyz"));
    let mol = Molecule::load_xyz_with_charge(xyz.to_str().unwrap(), charge, mult)
        .unwrap_or_else(|e| panic!("{}: {e:?}", xyz.display()));
    // Geometry like-for-like FIRST.
    check_close(
        &label,
        "E_nuc",
        mol.nuclear_repulsion(),
        num(&r, "/nuclear_repulsion", &label),
        TOL_ENUC,
    );
    assert_eq!(
        mol.nelec() as i64,
        r["nelectron"].as_i64().unwrap(),
        "{label}: electron count"
    );
    let obs_bs = basis::bundled(basis_name).unwrap();
    let aux_name = r["aux"].as_str().expect("aux").to_string();
    let aux_bs = basis::bundled(&aux_name).unwrap();
    let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
    let dfbs = PreparedBasis::new(&mol, &aux_bs).unwrap();
    assert_eq!(
        obs.nbasis() as i64,
        r["nao"].as_i64().unwrap(),
        "{label}: AO count differs from the reference's"
    );
    assert_eq!(
        dfbs.nbasis() as i64,
        r["naux"].as_i64().unwrap(),
        "{label}: aux count differs from the reference's"
    );
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &obs).unwrap();
    Sys {
        label,
        r,
        mol,
        obs,
        dfbs,
        bounds,
        ctx: ParallelContext::default(),
    }
}

fn rhf_config() -> RhfConfig {
    RhfConfig {
        max_iter: 400,
        energy_conv: 1e-11,
        density_conv: 1e-10,
        ..Default::default()
    }
}

fn uhf_config() -> RhfConfig {
    RhfConfig {
        max_iter: 400,
        energy_conv: 1e-11,
        density_conv: 1e-9,
        check_stability: true,
        scf_stability_descent: true,
        ..Default::default()
    }
}

/// UHF on the reference's stable state (energy `e_ref`). First the
/// stability-following config, then the level-shift + MOM steering that
/// NH2/cc-pVDZ is known to need; accepted only if E equals the reference.
fn uhf(sys: &Sys, e_ref: f64) -> ScfResult {
    let steered = RhfConfig {
        level_shift: 0.5,
        mom_after_iter: 5,
        ..uhf_config()
    };
    let mut tried = Vec::new();
    for (name, cfg) in [("stability-descent", uhf_config()), ("ls0.5+MOM5", steered)] {
        let res = solve_uhf(&sys.ctx, &sys.mol, &sys.obs, &sys.bounds, &cfg)
            .unwrap_or_else(|e| panic!("{}: solve_uhf failed: {e:?}", sys.label));
        let d = (res.energy - e_ref).abs();
        tried.push(format!("{name}: E {:.10} |d| {d:.2e}", res.energy));
        if res.converged && d < TOL_E_SCF {
            eprintln!("{}: UHF state reached with {name}", sys.label);
            if let Some(st) = res.stability.as_ref() {
                // OH's degenerate π hole is an exact zero mode, reported
                // MARGINAL by ferric; unstable or indeterminate still fails.
                assert!(
                    matches!(
                        st.verdict(),
                        StabilityVerdict::Stable | StabilityVerdict::Marginal
                    ),
                    "{}: ferric UHF state not stable: {}",
                    sys.label,
                    st.summary()
                );
            }
            check_close(&sys.label, "E_UHF", res.energy, e_ref, TOL_E_SCF);
            return res;
        }
    }
    panic!(
        "{}: ferric did not reach the reference UHF state {e_ref:.10}: {tried:?}",
        sys.label
    );
}

/// Full rank (every dielectric mode kept), Gauss–Legendre with PySCF's map.
fn rpa_cfg(n_quad: usize, trunc_thresh: f64) -> PdepRpaConfig {
    PdepRpaConfig {
        frozen_core: 0,
        trunc_thresh,
        eigensolver: Eigensolver::Lanczos,
        quadrature: QuadratureConfig {
            scheme: QuadratureScheme::GaussLegendre,
            n_points: n_quad,
            u0: U0,
        },
        ..Default::default()
    }
}

fn u_rpa(sys: &Sys, scf: &ScfResult, n_quad: usize, trunc_thresh: f64) -> PdepRpaResult {
    run_u_pdep_rpa(
        &sys.mol,
        &sys.obs,
        &sys.dfbs,
        Operator::coulomb(),
        scf,
        &rpa_cfg(n_quad, trunc_thresh),
    )
    .unwrap_or_else(|e| panic!("{}: run_u_pdep_rpa failed: {e:?}", sys.label))
}

fn assert_full_rank(sys: &Sys, res: &PdepRpaResult) {
    let naux = sys.dfbs.nbasis();
    assert_eq!(
        res.n_eigenpotentials, naux,
        "{}: trunc_thresh = 0 must keep all {naux} dielectric modes",
        sys.label
    );
    assert_eq!(
        res.quad_freqs.len(),
        N_QUAD,
        "{}: quadrature size",
        sys.label
    );
}

/// Open-shell system on its reference UHF state, full-rank U-RPA compared,
/// then the spin-channel control. Returns (sys, UHF, E_c) for further controls.
fn open_shell_case(system: &str, basis_name: &str) -> (Sys, ScfResult, f64) {
    let sys = load_system(system, basis_name);
    let e_ref = num(&sys.r, "/uhf/energy", &sys.label);
    let scf = uhf(&sys, e_ref);
    let res = u_rpa(&sys, &scf, N_QUAD, 0.0);
    assert_full_rank(&sys, &res);
    let ctx = &sys.label;
    check_close(
        ctx,
        "E_c (U-RPA, 40-pt GL)",
        res.e_rpa,
        num(&sys.r, "/urpa/e_corr", ctx),
        TOL_E_C,
    );
    // Spin-channel control: the reference distinguishes dropping either Π_s.
    for spin in ["alpha", "beta"] {
        assert_misses(
            ctx,
            &format!("{spin}-only Pi reference"),
            res.e_rpa,
            num(&sys.r, &format!("/urpa/e_corr_{spin}_only"), ctx),
            TOL_E_C,
        );
    }
    let e_c = res.e_rpa;
    (sys, scf, e_c)
}

/// ferric at `TRUNC_CONTROL` keeps the emulated number of modes, matches the
/// numpy emulation of its own truncation, and misses the full-rank value.
fn truncation_control(sys: &Sys, scf: &ScfResult, block: &str) {
    let ctx = &sys.label;
    let key = format!("/{block}/truncated/0.01");
    let t_ref = num(&sys.r, &format!("{key}/trunc_thresh"), ctx);
    assert_eq!(t_ref, TRUNC_CONTROL, "{ctx}: truncation control threshold");
    let res = u_rpa(sys, scf, N_QUAD, TRUNC_CONTROL);
    let n_keep_ref = sys
        .r
        .pointer(&format!("{key}/n_keep"))
        .unwrap()
        .as_u64()
        .unwrap() as usize;
    eprintln!(
        "{ctx}: truncation control kept {} modes (reference {n_keep_ref} of {})",
        res.n_eigenpotentials,
        sys.dfbs.nbasis()
    );
    assert_eq!(
        res.n_eigenpotentials, n_keep_ref,
        "{ctx}: trunc_thresh {TRUNC_CONTROL} kept a different number of modes"
    );
    check_close(
        ctx,
        "E_c truncated (emulation)",
        res.e_rpa,
        num(&sys.r, &format!("{key}/e_corr"), ctx),
        TOL_E_C_TRUNC,
    );
    assert_misses(
        ctx,
        "truncated vs full-rank reference",
        res.e_rpa,
        num(&sys.r, &format!("/{block}/e_corr"), ctx),
        TOL_E_C,
    );
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Closed-shell limit: H2O through BOTH paths. PySCF `RPA` and `URPA` on the
/// same RHF are equal (the generator refuses otherwise); ferric's
/// `run_pdep_rpa` (RHF) and `run_u_pdep_rpa` (singlet UHF) must equal them and
/// each other. Blind to α/β mix-ups (α ≡ β), not to the per-spin prefactor.
#[test]
#[ignore = "validation: U-RPA"]
fn urpa_closed_shell_limit_h2o_vs_pyscf_rpa() {
    let sys = load_system("h2o", "cc-pvdz");
    let ctx = &sys.label;
    let e_rhf_ref = num(&sys.r, "/rhf/energy", ctx);
    let rhf = solve_rhf(
        &sys.ctx,
        &sys.mol,
        &sys.obs,
        Operator::coulomb(),
        &sys.bounds,
        &rhf_config(),
    )
    .unwrap_or_else(|e| panic!("{ctx}: RHF failed: {e:?}"));
    assert!(rhf.converged, "{ctx}: RHF did not converge");
    check_close(ctx, "E_RHF", rhf.energy, e_rhf_ref, TOL_E_SCF);
    // The singlet UHF must land on the RHF state (no symmetry breaking).
    let scf_u = uhf(&sys, e_rhf_ref);

    let r = run_pdep_rpa(
        &sys.mol,
        &sys.obs,
        &sys.dfbs,
        Operator::coulomb(),
        &rhf,
        &rpa_cfg(N_QUAD, 0.0),
    )
    .unwrap_or_else(|e| panic!("{ctx}: run_pdep_rpa failed: {e:?}"));
    assert_full_rank(&sys, &r);
    let u = u_rpa(&sys, &scf_u, N_QUAD, 0.0);
    assert_full_rank(&sys, &u);

    let e_rpa_ref = num(&sys.r, "/rpa/e_corr", ctx);
    let e_urpa_ref = num(&sys.r, "/urpa_on_rhf/e_corr", ctx);
    check_close(
        ctx,
        "PySCF RPA vs URPA (same RHF)",
        e_rpa_ref,
        e_urpa_ref,
        1e-12,
    );
    check_close(ctx, "E_c run_pdep_rpa vs RPA", r.e_rpa, e_rpa_ref, TOL_E_C);
    check_close(
        ctx,
        "E_c run_u_pdep_rpa vs URPA",
        u.e_rpa,
        e_urpa_ref,
        TOL_E_C,
    );
    check_close(ctx, "E_c U path vs R path", u.e_rpa, r.e_rpa, TOL_U_VS_R);
    // The reference distinguishes a one-channel (per-spin prefactor) error.
    assert_misses(
        ctx,
        "alpha-only Pi reference (U path)",
        u.e_rpa,
        num(&sys.r, "/urpa_on_rhf/e_corr_alpha_only", ctx),
        TOL_E_C,
    );
}

#[test]
#[ignore = "validation: U-RPA"]
fn urpa_oh_vs_pyscf_urpa() {
    let _ = open_shell_case("oh", "cc-pvdz");
}

#[test]
#[ignore = "validation: U-RPA"]
fn urpa_oh_aug_cc_pvdz_vs_pyscf_urpa_and_basis_control() {
    let (sys, _scf, e_c) = open_shell_case("oh", "aug-cc-pvdz");
    // Basis control: the aug-cc-pVDZ result misses the cc-pVDZ reference.
    let other = reference("oh", "cc-pvdz");
    assert_misses(
        &sys.label,
        "cc-pVDZ reference",
        e_c,
        num(&other, "/urpa/e_corr", "oh/cc-pvdz"),
        TOL_E_C,
    );
}

#[test]
#[ignore = "validation: U-RPA"]
fn urpa_ch3_vs_pyscf_urpa() {
    let _ = open_shell_case("ch3", "cc-pvdz");
}

#[test]
#[ignore = "validation: U-RPA"]
fn urpa_nh2_vs_pyscf_urpa() {
    let _ = open_shell_case("nh2", "cc-pvdz");
}

#[test]
#[ignore = "validation: U-RPA"]
fn urpa_o2_triplet_vs_pyscf_urpa() {
    let _ = open_shell_case("o2", "cc-pvdz");
}

/// Quadrature control: ferric at 20 points matches PySCF at nw = 20 (same
/// map, so the match is again quadrature-exact) and misses the 40-point value.
#[test]
#[ignore = "validation: U-RPA"]
fn urpa_quadrature_control_oh() {
    let sys = load_system("oh", "cc-pvdz");
    let ctx = sys.label.clone();
    let scf = uhf(&sys, num(&sys.r, "/uhf/energy", &ctx));
    let res = u_rpa(&sys, &scf, N_QUAD_CONTROL, 0.0);
    assert_eq!(res.quad_freqs.len(), N_QUAD_CONTROL);
    check_close(
        &ctx,
        "E_c 20-pt vs PySCF nw=20",
        res.e_rpa,
        num(&sys.r, "/urpa/nw_convergence/20", &ctx),
        TOL_E_C,
    );
    assert_misses(
        &ctx,
        "20-pt vs the 40-pt reference",
        res.e_rpa,
        num(&sys.r, "/urpa/e_corr", &ctx),
        TOL_E_C,
    );
}

#[test]
#[ignore = "validation: U-RPA"]
fn urpa_truncation_control_oh() {
    let sys = load_system("oh", "cc-pvdz");
    let scf = uhf(&sys, num(&sys.r, "/uhf/energy", &sys.label));
    truncation_control(&sys, &scf, "urpa");
}

#[test]
#[ignore = "validation: U-RPA"]
fn urpa_truncation_control_o2_triplet() {
    let sys = load_system("o2", "cc-pvdz");
    let scf = uhf(&sys, num(&sys.r, "/uhf/energy", &sys.label));
    truncation_control(&sys, &scf, "urpa");
}

/// Reference-orbital control: U-RPA on OH's ROHF state (checked against
/// PySCF's stability-checked ROHF energy, so the control runs on a real ROHF
/// state) must miss the UHF-based reference.
#[test]
#[ignore = "validation: U-RPA"]
fn urpa_reference_orbital_control_oh_rohf() {
    let sys = load_system("oh", "cc-pvdz");
    let ctx = sys.label.clone();
    let rohf = solve_rohf(
        &sys.ctx,
        &sys.mol,
        &sys.obs,
        Operator::coulomb(),
        &sys.bounds,
        &RhfConfig {
            max_iter: 400,
            energy_conv: 1e-11,
            density_conv: 1e-9,
            ..Default::default()
        },
    )
    .unwrap_or_else(|e| panic!("{ctx}: solve_rohf failed: {e:?}"));
    assert!(rohf.converged, "{ctx}: ROHF did not converge");
    check_close(
        &ctx,
        "E_ROHF",
        rohf.energy,
        num(&sys.r, "/rohf/energy", &ctx),
        TOL_E_SCF,
    );
    let res = u_rpa(&sys, &rohf, N_QUAD, 0.0);
    assert_misses(
        &ctx,
        "ROHF-based U-RPA vs UHF reference",
        res.e_rpa,
        num(&sys.r, "/urpa/e_corr", &ctx),
        TOL_E_C,
    );
}
