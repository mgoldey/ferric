//! VALIDATION tier — VALIDATION.md row "U-COHSEX" (spin-unrestricted static
//! COHSEX, `crates/ferric-gw/src/u_cohsex.rs`).
//!
//! Runs only in the weekly `validation` CI job (and on demand):
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 cargo nextest run -p ferric-gw --release \
//!     --test validation_u_cohsex --run-ignored only -E 'binary(/^validation_/)'
//! ```
//!
//! # What is compared
//!
//! `run_u_gw` with `GwMethod::Cohsex` (lib.rs dispatch → `u_cohsex::run_u_cohsex`)
//! against a numpy U-COHSEX evaluated on PySCF density-fitted integrals with
//! ferric's aux, on the same stability-checked UHF the U-G0W0 row uses
//! (`scripts/validation/gen_u_cohsex.py`, which imports its SCF, basis, aux
//! and window helpers from `gen_gw.py`):
//!
//! | test | systems / basis (aux) | reference |
//! |---|---|---|
//! | U-COHSEX@UHF | OH, CH3, NH2 / cc-pVDZ (cc-pvdz-ri), aug-cc-pVDZ (aug-cc-pvdz-rifit); O2 (³Σg⁻), CH2 (³B1) / aug-cc-pVDZ | numpy on PySCF DF `L^σ_{P,pq}` |
//! | closed-shell anchor | H2O / cc-pVDZ, singlet UHF through `run_u_gw` AND RHF through `run_gw` | same numpy; `gen_gw.cohsex_block`; the committed `gw/h2o_cc-pvdz.json` `cohsex_hf` |
//!
//! References: `testdata/reference/validation/u_cohsex/<system>_<basis>.json`.
//!
//! # Conventions mirrored (read from the code, with line refs)
//!
//! * ONE W for both spins: `u_cohsex.rs:26` builds `w_static` once from
//!   `pdep.eigenvalues_static` and passes the same slice to both spins'
//!   `cohsex_pieces` (`:46-47`). The dielectric is the spin-summed
//!   ε̃(0) = I + Π_α(0) + Π_β(0) with prefactor 2 per spin
//!   (`ferric-rpa` `sternheimer::dielectric_apply_unrestricted`), each spin with
//!   its own occupied/virtual B tensor and orbital energies (`run_u_pdep_rpa`).
//! * Static weights w_α = 1/λ_α(0) − 1 (`w_pdep.rs:86-87`); at
//!   `trunc_thresh = 0` the PDEP basis is the full aux space, so
//!   Σ_α w_α M²_{α,mn} = L_mnᵀ(ε⁻¹ − I)L_mn, which the generator evaluates both
//!   by inverse and by eigh+weights (asserted equal to ~1e-15).
//! * Per spin (`cohsex.rs` `cohsex_pieces`): ΔΣ_SEX sums over the OCCUPIED
//!   orbitals OF THAT SPIN (`mo_b.n_occ_act` = nocc_α / nocc_β, `mo_b.rs`
//!   `build_full_b_both_spins`); Σ_COH = ½ Σ over ALL active orbitals of that
//!   spin.
//! * QP energy: ε_qp,σ = ε_mf,σ + ΔΣ_SEX,σ + Σ_COH,σ (`u_cohsex.rs:75-78`). The
//!   bare Σx is NOT replaced (HF reference, Σx − v_x = 0); it is reported as
//!   `sigma_x_*` (`cohsex.rs` `sigma_x_diag`: −Σ_{P,i occ σ} B̃^P_{mi}²) and
//!   `sigma_c_*` is ΔΣ_SEX + Σ_COH. Z = 1, no Newton solve.
//! * Frozen core 0; QP window HOMO−2 … LUMO+2 of the majority spin
//!   (`default_u_qp_range`), same MO indices for both spins.
//! * COHSEX reads only the static eigenpairs, never the frequency stack, so
//!   the W quadrature is irrelevant; this file runs a small grid
//!   (`COHSEX_N_QUAD`) and the anchor test proves the independence by running
//!   two grids and asserting identical QP energies.
//!
//! # Physics hypothesis vs artifact hypothesis
//!
//! * Right: closed form on identical integrals, so agreement at the ε_mf / Σx
//!   floor (1e-9 … 1e-8 Ha); no continuation or quadrature noise.
//! * W missing a spin channel or with the closed-shell prefactor: ≥7e-3 Ha
//!   (the `controls` blocks, measured by the generator on every system).
//! * Spins crossed (β Σ from α pair densities, α/β swapped): tenths of Ha.
//! * Harness wrong (geometry, basis, aux, state): E_nuc, AO/aux counts, E_UHF,
//!   ε_mf or Σx fails first.
//!
//! # Exactness anchors (checked before any QP energy)
//!
//! 1. E_UHF equals the stability-checked reference state; ε_mf per spin.
//! 2. Σx per spin equals −Σ_{P,i} L_{P,mi}² from PySCF's Lpq — an independent
//!    construction of each spin's B tensor.
//! 3. ε_qp − ε_mf − Σc = 0 per state (the closed-form QP equation).
//! 4. Closed-shell limit: U-COHSEX on a singlet UHF equals ferric's own
//!    closed-shell COHSEX on the RHF, state by state and spin by spin.
//!
//! # TOLERANCES
//!
//! Each bar is 3–25x the worst |d| measured over every system (2026-09-25).
//!
//! # NEGATIVE CONTROLS (asserted)
//!
//! * α-only W (β polarizability dropped): ferric misses `controls/alpha_only_w`
//!   on every system, both spins (and on the H2O singlet).
//! * Closed-shell COHSEX formula applied to the open shell (each spin's Σ
//!   with W from Π = 4× that spin's own ov block): ferric misses
//!   `controls/closed_formula_per_spin` on every open-shell system. At the H2O
//!   singlet that formula IS the exact limit, and ferric must MATCH it there.
//! * α/β swap: ferric's α QP energies miss the β reference.
//! * Basis swap: OH, CH3, NH2 at cc-pVDZ miss the aug-cc-pVDZ reference and
//!   vice versa.
//!
//! # MUTATION (run once, record the outcome here)
//!
//! `crates/ferric-gw/src/u_cohsex.rs:47`
//! `let (dsex_b, scoh_b) = cohsex_pieces(mo_b_b, &m_proj_b, &w_static);` →
//! `&m_proj_a` (β self-energy from α pair densities). Predicted: every
//! open-shell case fails its β `dSEX+COH` / `eps_qp` comparison; the H2O
//! singlet anchor still PASSES (α and β projections coincide there) — the
//! anchor alone cannot see a spin-crossing defect, which is why the open-shell
//! rows exist. Run 2026-09-25: all four open-shell tests fail; the H2O singlet
//! anchor passes.

use std::path::{Path, PathBuf};

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_gw::{run_gw, run_u_gw, GwConfig, GwMethod, GwResult, UGwResult};
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_rpa::config::{Eigensolver, PdepRpaConfig, QuadratureConfig, QuadratureScheme};
use ferric_scf::ladder::{default_ladder_from, solve_rhf_ladder};
use ferric_scf::rhf::RhfConfig;
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::stability::StabilityVerdict;
use ferric_scf::uhf::solve_uhf;
use ferric_scf::ScfResult;
use serde_json::Value;

const ROW_DIR: &str = "testdata/reference/validation/u_cohsex";
const MOL_DIR: &str = "testdata/molecules/validation";
const HA_TO_EV: f64 = 27.211_386_245_988;
/// W frequency points. COHSEX never reads the frequency stack (u_cohsex.rs:26
/// uses `eigenvalues_static` only); `ANCHOR_N_QUAD_ALT` proves it.
const COHSEX_N_QUAD: usize = 16;
const ANCHOR_N_QUAD_ALT: usize = 4;

// U-COHSEX ε_qp and ΔΣ_SEX+Σ_COH vs numpy, open shell: 3.3e-9.
const TOL_U_COHSEX: f64 = 3e-8;
// Closed-shell COHSEX: 9.4e-10.
const TOL_COHSEX: f64 = 1e-8;
// ferric U-COHSEX(singlet UHF) vs ferric COHSEX(RHF): 8.1e-10.
const TOL_ANCHOR: f64 = 1e-8;
// Same PDEP static eigensolve at two W grids: identical (0).
const TOL_QUAD_INDEPENDENCE: f64 = 1e-13;
// Σx (DF) per spin: 4.7e-9.
const TOL_SX: f64 = 3e-8;
// ε_mf per spin: 3.7e-9.
const TOL_EPS_MF: f64 = 3e-8;
// E_UHF / E_RHF: 4.9e-12.
const TOL_E_SCF: f64 = 5e-11;
/// ε_qp − ε_mf − Σc is an identity of the closed form (u_cohsex.rs:77-78).
const TOL_RESID: f64 = 1e-12;
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
        .expect("ferric-gw manifest dir should be <root>/crates/ferric-gw")
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
             scripts/validation/gen_u_cohsex.py — a missing reference is a failure, never a skip",
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

fn vec_f64(v: &Value, ptr: &str, ctx: &str) -> Vec<f64> {
    v.pointer(ptr)
        .and_then(Value::as_array)
        .unwrap_or_else(|| panic!("{ctx}: reference field {ptr} missing or not an array"))
        .iter()
        .map(|x| x.as_f64().expect("number"))
        .collect()
}

fn vec_usize(v: &Value, ptr: &str, ctx: &str) -> Vec<usize> {
    v.pointer(ptr)
        .and_then(Value::as_array)
        .unwrap_or_else(|| panic!("{ctx}: reference field {ptr} missing or not an array"))
        .iter()
        .map(|x| x.as_u64().expect("index") as usize)
        .collect()
}

/// The reference QP window as a contiguous range (ferric's `qp_mos` type).
fn window(orbs: &[usize], ctx: &str) -> std::ops::Range<usize> {
    assert!(!orbs.is_empty(), "{ctx}: empty QP window");
    for w in orbs.windows(2) {
        assert_eq!(
            w[1],
            w[0] + 1,
            "{ctx}: QP window {orbs:?} is not contiguous"
        );
    }
    orbs[0]..orbs[orbs.len() - 1] + 1
}

fn check_close(ctx: &str, what: &str, got: f64, want: f64, tol: f64) {
    let d = (got - want).abs();
    eprintln!(
        "{ctx}: {what:<28} ferric {got:+.10} ref {want:+.10} |d| {d:.2e} ({:.4} meV)",
        d * HA_TO_EV * 1e3
    );
    assert!(
        d < tol,
        "{ctx}: {what}: ferric {got:.10} vs reference {want:.10} (|d| {d:.2e} Ha) exceeds {tol:.1e} Ha"
    );
}

/// Print every state's difference FIRST (the whole window is the measurement
/// the bars are set from), then assert the worst one.
fn check_vec(ctx: &str, what: &str, orbs: &[usize], got: &[f64], want: &[f64], tol: f64) -> f64 {
    assert_eq!(got.len(), want.len(), "{ctx}: {what} length");
    assert_eq!(got.len(), orbs.len(), "{ctx}: {what} vs window length");
    let mut worst = (0.0_f64, 0usize, 0.0_f64, 0.0_f64);
    for ((&p, &g), &w) in orbs.iter().zip(got).zip(want) {
        let d = (g - w).abs();
        eprintln!(
            "{ctx}: {:<28} ferric {g:+.10} ref {w:+.10} |d| {d:.2e} ({:.4} meV)",
            format!("{what}[{p}]"),
            d * HA_TO_EV * 1e3
        );
        if d > worst.0 {
            worst = (d, p, g, w);
        }
    }
    let (d, p, g, w) = worst;
    eprintln!(
        "{ctx}: {what} WORST |d| {d:.3e} Ha ({:.4} meV) at MO {p}",
        d * HA_TO_EV * 1e3
    );
    assert!(
        d < tol,
        "{ctx}: {what}[{p}]: ferric {g:.10} vs reference {w:.10} (|d| {d:.2e} Ha) exceeds {tol:.1e} Ha"
    );
    d
}

/// max_p |got_p − want_p| must exceed `factor × tol`.
fn assert_misses(ctx: &str, what: &str, got: &[f64], want: &[f64], tol: f64, factor: f64) {
    assert_eq!(got.len(), want.len(), "{ctx}: control {what} length");
    let d = got
        .iter()
        .zip(want)
        .map(|(g, w)| (g - w).abs())
        .fold(0.0_f64, f64::max);
    eprintln!(
        "{ctx}: control {what}: max |d| {d:.3e} Ha ({:.3} meV), must exceed {:.1e}",
        d * HA_TO_EV * 1e3,
        factor * tol
    );
    assert!(
        d > factor * tol,
        "{ctx}: negative control '{what}' did not miss: max |d| {d:.3e} Ha <= {factor} x {tol:.1e} \
         — the comparison does not respond to what the control changes"
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
        "{label}: AO count"
    );
    assert_eq!(
        dfbs.nbasis() as i64,
        r["naux"].as_i64().unwrap(),
        "{label}: aux count"
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

fn rhf(sys: &Sys) -> ScfResult {
    let lr = solve_rhf_ladder(
        &sys.ctx,
        &sys.mol,
        &sys.obs,
        Operator::coulomb(),
        &sys.bounds,
        &default_ladder_from(&rhf_config()),
    )
    .unwrap_or_else(|e| panic!("{}: RHF ladder failed: {e:?}", sys.label));
    assert!(lr.converged, "{}: RHF did not converge", sys.label);
    let scf = lr.result;
    check_close(
        &sys.label,
        "E_RHF",
        scf.energy,
        num(&sys.r, "/rhf/energy", &sys.label),
        TOL_E_SCF,
    );
    scf
}

/// UHF on the reference's stable state (the U-G0W0 row's recipe: stability
/// descent first, then level shift + MOM), accepted only at the reference
/// energy.
fn uhf(sys: &Sys) -> ScfResult {
    let e_ref = num(&sys.r, "/uhf/energy", &sys.label);
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
                // OH's degenerate π hole gives an exact zero mode (MARGINAL).
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

fn pdep_cfg(n_quad: usize) -> PdepRpaConfig {
    PdepRpaConfig {
        frozen_core: 0,
        // Full rank: the dense Lanczos path keeps every dielectric mode.
        trunc_thresh: 0.0,
        eigensolver: Eigensolver::Lanczos,
        quadrature: QuadratureConfig {
            scheme: QuadratureScheme::GaussLegendre,
            n_points: n_quad,
            u0: 0.5,
        },
        need_inv_dielectric_freq: true,
        need_eigenvalues_freq: true,
        ..Default::default()
    }
}

fn gw_cfg(qp: std::ops::Range<usize>) -> GwConfig {
    GwConfig {
        method: GwMethod::Cohsex,
        qp_mos: Some(qp),
        frozen_core: 0,
        ..Default::default()
    }
}

fn run_u_cohsex(
    sys: &Sys,
    scf: &ScfResult,
    qp: std::ops::Range<usize>,
    n_quad: usize,
) -> UGwResult {
    let res = run_u_gw(
        &sys.mol,
        &sys.obs,
        &sys.dfbs,
        Operator::coulomb(),
        scf,
        &pdep_cfg(n_quad),
        &gw_cfg(qp.clone()),
    )
    .unwrap_or_else(|e| panic!("{}: run_u_gw(Cohsex) failed: {e:?}", sys.label));
    assert_eq!(
        res.mo_indices,
        qp.collect::<Vec<_>>(),
        "{}: QP window",
        sys.label
    );
    // Closed form: no Newton solve, no outer loop, Z = 1.
    assert_eq!(res.n_ev_iter, 0, "{}: U-COHSEX iterated", sys.label);
    assert!(res.outer_converged && res.qp_converged_a.iter().all(|&c| c));
    assert!(res.qp_converged_b.iter().all(|&c| c));
    assert!(
        res.z_factor_a
            .iter()
            .chain(res.z_factor_b.iter())
            .all(|&z| z == 1.0),
        "{}: static COHSEX must have Z = 1",
        sys.label
    );
    res
}

fn run_closed_cohsex(sys: &Sys, scf: &ScfResult, qp: std::ops::Range<usize>) -> GwResult {
    run_gw(
        &sys.mol,
        &sys.obs,
        &sys.dfbs,
        Operator::coulomb(),
        scf,
        &pdep_cfg(COHSEX_N_QUAD),
        &gw_cfg(qp),
        None,
    )
    .unwrap_or_else(|e| panic!("{}: run_gw(Cohsex) failed: {e:?}", sys.label))
}

/// Per-spin view of a UGwResult.
fn spin_view(res: &UGwResult, spin: &str) -> [Vec<f64>; 4] {
    if spin == "alpha" {
        [
            res.eps_mf_a.to_vec(),
            res.eps_qp_a.to_vec(),
            res.sigma_x_a.to_vec(),
            res.sigma_c_a.to_vec(),
        ]
    } else {
        [
            res.eps_mf_b.to_vec(),
            res.eps_qp_b.to_vec(),
            res.sigma_x_b.to_vec(),
            res.sigma_c_b.to_vec(),
        ]
    }
}

/// Anchors + comparison for one spin against `/u_cohsex_uhf/<spin>`.
/// Returns ferric's ε_qp for that spin.
fn check_u_spin(sys: &Sys, orbs: &[usize], res: &UGwResult, spin: &str) -> Vec<f64> {
    let ctx = format!("{} U-COHSEX {spin}", sys.label);
    let p = format!("/u_cohsex_uhf/{spin}");
    let [eps_mf, eps_qp, sx, sc] = spin_view(res, spin);
    check_vec(
        &ctx,
        "eps_mf",
        orbs,
        &eps_mf,
        &vec_f64(&sys.r, &format!("{p}/eps_mf"), &ctx),
        TOL_EPS_MF,
    );
    check_vec(
        &ctx,
        "sigma_x(DF)",
        orbs,
        &sx,
        &vec_f64(&sys.r, &format!("{p}/sigma_x_df"), &ctx),
        TOL_SX,
    );
    for (k, &mo) in orbs.iter().enumerate() {
        let resid = eps_qp[k] - eps_mf[k] - sc[k];
        assert!(
            resid.abs() < TOL_RESID,
            "{ctx}: QP residual {resid:.2e} Ha at MO {mo}"
        );
    }
    check_vec(
        &ctx,
        "dSEX+COH",
        orbs,
        &sc,
        &vec_f64(&sys.r, &format!("{p}/sigma_c"), &ctx),
        TOL_U_COHSEX,
    );
    check_vec(
        &ctx,
        "eps_qp",
        orbs,
        &eps_qp,
        &vec_f64(&sys.r, &format!("{p}/eps_qp"), &ctx),
        TOL_U_COHSEX,
    );
    eps_qp
}

// ---------------------------------------------------------------------------
// Open-shell cases
// ---------------------------------------------------------------------------

/// Returns ferric's α ε_qp (for the basis-swap control).
fn u_cohsex_case(system: &str, basis_name: &str) -> Vec<f64> {
    let sys = load_system(system, basis_name);
    let scf = uhf(&sys);
    let orbs = vec_usize(&sys.r, "/u_cohsex_uhf/orbs", &sys.label);
    let res = run_u_cohsex(&sys, &scf, window(&orbs, &sys.label), COHSEX_N_QUAD);
    let qa = check_u_spin(&sys, &orbs, &res, "alpha");
    let qb = check_u_spin(&sys, &orbs, &res, "beta");
    let ctx = &sys.label;
    for (spin, q) in [("alpha", &qa), ("beta", &qb)] {
        assert_misses(
            ctx,
            &format!("{spin} QP vs alpha-only-W reference"),
            q,
            &vec_f64(
                &sys.r,
                &format!("/controls/alpha_only_w/{spin}_eps_qp"),
                ctx,
            ),
            TOL_U_COHSEX,
            MUST_MISS_FACTOR,
        );
        assert_misses(
            ctx,
            &format!("{spin} QP vs closed-shell-formula-per-spin reference"),
            q,
            &vec_f64(
                &sys.r,
                &format!("/controls/closed_formula_per_spin/{spin}_eps_qp"),
                ctx,
            ),
            TOL_U_COHSEX,
            MUST_MISS_FACTOR,
        );
    }
    assert_misses(
        ctx,
        "alpha QP vs the beta reference (spin swap)",
        &qa,
        &vec_f64(&sys.r, "/u_cohsex_uhf/beta/eps_qp", ctx),
        TOL_U_COHSEX,
        MUST_MISS_FACTOR,
    );
    assert_misses(
        ctx,
        "beta QP vs the alpha reference (spin swap)",
        &qb,
        &vec_f64(&sys.r, "/u_cohsex_uhf/alpha/eps_qp", ctx),
        TOL_U_COHSEX,
        MUST_MISS_FACTOR,
    );
    qa
}

const TWO_BASES: [&str; 2] = ["cc-pvdz", "aug-cc-pvdz"];

fn u_cohsex_two_bases(system: &str) {
    let qa = [
        u_cohsex_case(system, TWO_BASES[0]),
        u_cohsex_case(system, TWO_BASES[1]),
    ];
    for (i, basis_name) in TWO_BASES.iter().enumerate() {
        let other = TWO_BASES[1 - i];
        assert_misses(
            &format!("{system}/{basis_name}"),
            &format!("alpha QP vs the {other} reference (basis swap)"),
            &qa[i],
            &vec_f64(
                &reference(system, other),
                "/u_cohsex_uhf/alpha/eps_qp",
                system,
            ),
            TOL_U_COHSEX,
            MUST_MISS_FACTOR,
        );
    }
}

#[test]
#[ignore = "validation: U-COHSEX"]
fn u_cohsex_oh_vs_numpy() {
    u_cohsex_two_bases("oh");
}

#[test]
#[ignore = "validation: U-COHSEX"]
fn u_cohsex_ch3_vs_numpy() {
    u_cohsex_two_bases("ch3");
}

#[test]
#[ignore = "validation: U-COHSEX"]
fn u_cohsex_nh2_vs_numpy() {
    u_cohsex_two_bases("nh2");
}

#[test]
#[ignore = "validation: U-COHSEX"]
fn u_cohsex_triplets_o2_ch2_vs_numpy() {
    u_cohsex_case("o2", "aug-cc-pvdz");
    u_cohsex_case("ch2_triplet", "aug-cc-pvdz");
}

// ---------------------------------------------------------------------------
// Closed-shell anchor: U-COHSEX(singlet UHF) == COHSEX(RHF)
// ---------------------------------------------------------------------------

#[test]
#[ignore = "validation: U-COHSEX"]
fn u_cohsex_singlet_h2o_equals_closed_shell_cohsex() {
    let sys = load_system("h2o", "cc-pvdz");
    let ctx = format!("{} anchor", sys.label);
    let scf_r = rhf(&sys);
    let scf_u = uhf(&sys);
    let orbs = vec_usize(&sys.r, "/u_cohsex_uhf/orbs", &ctx);
    assert_eq!(
        orbs,
        vec_usize(&sys.r, "/cohsex_rhf/orbs", &ctx),
        "{ctx}: U and closed windows differ"
    );
    let qp = window(&orbs, &ctx);

    // Closed-shell path vs the closed-shell numpy (gen_gw.cohsex_block).
    let closed = run_closed_cohsex(&sys, &scf_r, qp.clone());
    check_vec(
        &ctx,
        "closed dSEX+COH",
        &orbs,
        closed.sigma_c.as_slice().unwrap(),
        &vec_f64(&sys.r, "/cohsex_rhf/sigma_c", &ctx),
        TOL_COHSEX,
    );
    check_vec(
        &ctx,
        "closed eps_qp",
        &orbs,
        closed.eps_qp.as_slice().unwrap(),
        &vec_f64(&sys.r, "/cohsex_rhf/eps_qp", &ctx),
        TOL_COHSEX,
    );

    // Unrestricted path on the singlet UHF vs the U numpy.
    let res = run_u_cohsex(&sys, &scf_u, qp.clone(), COHSEX_N_QUAD);
    let qa = check_u_spin(&sys, &orbs, &res, "alpha");
    let qb = check_u_spin(&sys, &orbs, &res, "beta");

    // ferric vs ferric: both spins of U-COHSEX equal closed-shell COHSEX.
    let closed_sc = closed.sigma_c.to_vec();
    let closed_qp = closed.eps_qp.to_vec();
    for (spin, q, sc) in [
        ("alpha", &qa, res.sigma_c_a.to_vec()),
        ("beta", &qb, res.sigma_c_b.to_vec()),
    ] {
        check_vec(
            &ctx,
            &format!("U {spin} dSEX+COH vs closed"),
            &orbs,
            &sc,
            &closed_sc,
            TOL_ANCHOR,
        );
        check_vec(
            &ctx,
            &format!("U {spin} eps_qp vs closed"),
            &orbs,
            q,
            &closed_qp,
            TOL_ANCHOR,
        );
        // At a singlet the closed-shell formula IS the exact limit: match it.
        check_vec(
            &ctx,
            &format!("U {spin} vs closed-formula control"),
            &orbs,
            q,
            &vec_f64(
                &sys.r,
                &format!("/controls/closed_formula_per_spin/{spin}_eps_qp"),
                &ctx,
            ),
            TOL_U_COHSEX,
        );
        // Dropping a spin's polarizability must still show at a singlet.
        assert_misses(
            &ctx,
            &format!("U {spin} vs alpha-only-W reference"),
            q,
            &vec_f64(
                &sys.r,
                &format!("/controls/alpha_only_w/{spin}_eps_qp"),
                &ctx,
            ),
            TOL_U_COHSEX,
            MUST_MISS_FACTOR,
        );
    }

    // COHSEX reads only the static eigenpairs: another W grid, same answer.
    let alt = run_u_cohsex(&sys, &scf_u, qp, ANCHOR_N_QUAD_ALT);
    for (spin, a, b) in [
        ("alpha", &qa, alt.eps_qp_a.to_vec()),
        ("beta", &qb, alt.eps_qp_b.to_vec()),
    ] {
        check_vec(
            &ctx,
            &format!("{spin} eps_qp, n_quad {COHSEX_N_QUAD} vs {ANCHOR_N_QUAD_ALT}"),
            &orbs,
            &b,
            a,
            TOL_QUAD_INDEPENDENCE,
        );
    }
}
