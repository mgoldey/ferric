//! VALIDATION tier — validation.md Anchors row "RS-MP2-RPA at production ω"
//! (`rs_mp2_lr_rpa`, formulations `DeltaLr` (B) and `CoupledRings` (T), the
//! erf/erfc split at the default ω = 0.420 Å⁻¹ = 0.222254 Bohr⁻¹).
//!
//! Runs only in the weekly `validation` CI job (and on demand):
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 cargo nextest run -p ferric-rpa --release \
//!     --test validation_rs_mp2_rpa_pyscf --run-ignored only -E 'binary(/^validation_/)'
//! ```
//!
//! # Operators
//!
//! | field | operator | definition |
//! |---|---|---|
//! | `e_mp2_full` | Coulomb 1/r | SC[C].E_MP2 |
//! | `e_sr_mp2` | erfc(ωr)/r | SC[erfc].E_MP2 |
//! | `e_lr_mp2`, `e_dmp2_lr` | erf(ωr)/r | SC[erf].E_MP2, 2·SC[erf].E_OS |
//! | B `e_drpa_lr` | erf | E_dRPA[erf] |
//! | B `e_corr_naive` | erfc, erf | SC[erfc].E_MP2 + E_dRPA[erf] |
//! | B `e_corr` | Coulomb, erf | SC[C].E_MP2 + E_dRPA[erf] − 2·SC[erf].E_OS |
//! | T `e_delta_drpa_full` | Coulomb | E_dRPA[C] − 2·SC[C].E_OS |
//! | T `e_delta_drpa_sr` | erfc | E_dRPA[erfc] − 2·SC[erfc].E_OS |
//! | T `e_corr` | Coulomb, erfc | SC[C].E_MP2 + Δfull − Δsr |
//!
//! The operator is in the 3-centre integrals AND the RI metric; the RHF is
//! the full-Coulomb RHF. No attenuated nuclear attraction is used.
//!
//! # Reference
//!
//! PySCF 2.13.1's own code (`scripts/validation/gen_rs_mp2_rpa_pyscf.py` →
//! `testdata/reference/validation/rs_mp2_rpa_pyscf/<system>_<basis>.json`):
//! per operator one `df.DF` built inside `mol.with_range_coulomb(w)`, handed
//! to `mp.dfmp2.DFMP2` (E_OS, E_SS) and `gw.rpa.RPA` (E_dRPA, 20 and 40
//! Gauss–Legendre points, x0 = 0.5); every field assembled from those. PySCF's
//! sign convention (w > 0 erf, w < 0 erfc) is checked in the generator against
//! the closed form of a one-primitive (ss|ss) integral.
//!
//! H2O, C2H4 and the S22 water dimer at cc-pVDZ (cc-pvdz-ri) and aug-cc-pVDZ
//! (aug-cc-pvdz-rifit).
//!
//! ferric runs on PySCF's RHF orbitals and orbital energies (written in
//! ferric's AO order; the map is checked here by orthonormality in ferric's
//! own overlap), so the SCF is not part of the comparison.
//!
//! # The bar, per field
//!
//! What remains between the two codes, each term from the reference file or
//! computed here:
//!
//! 1. **RI metric factorization** (`metric_shift`). ferric factors the erf
//!    metric by eigh, dropping eigenvalues below 1e-10; PySCF's Cholesky fails
//!    on it (eigenvalues down to ~−1e-14) and it falls back to eigh dropping
//!    those at or below 1e-7. The generator evaluates every field twice with
//!    the same numpy code on the same PySCF integrals, once with each
//!    factorization; the difference is the shift. Coulomb and erfc are
//!    Cholesky in both codes (shift 0).
//! 2. **Integrals** (`integral_shift`). The generator also evaluates every
//!    field with ferric's factorization on ferric's OWN 3-centre and
//!    2-centre integrals (`ferric.AoIntegrals.eri3/eri2`, the libint2
//!    engines this path runs at precision 1e-14); the difference from the
//!    same numpy on PySCF's integrals is the shift. On Coulomb and erfc it is
//!    at rounding level. On erf it is not: the erf metric has eigenvalues just
//!    above the 1e-10 cut, and V⁻¹ amplifies a 1e-14 integral difference by
//!    up to 1/λ.
//! 3. **A mode at the cut** (`cut_flip`). Where an eigenvalue of ferric's erf
//!    metric is closer to the 1e-10 cut than the backward-error bound
//!    n·ε·λ_max of a symmetric eigensolver (the reference records the margin
//!    per kernel), ferric's own eigh may place it on either side. The
//!    generator evaluates the field with that one mode moved across the cut;
//!    the change enters the bar. 0 where every mode is clear of the cut.
//! 4. **Metric eigenvalues** (`eigenvalue_error`). Each code's eigh (or
//!    Cholesky) returns every metric eigenvalue to within n·ε·λ_max, a large
//!    RELATIVE error on the erf metric's smallest kept modes. The generator
//!    moves each kept mode with λ ≤ 1e9·n·ε·λ_max of ferric's metric by that
//!    bound alone and sums |Δfield| over the modes; twice that (one error per
//!    code) enters the bar.
//! 5. **Log-det rounding.** ferric's dRPA summand is ln det(I + Π) + tr(I −
//!    (I + Π)) from an LU (`energy::dielectric_trace_log_summand`), the
//!    reference's ln det(I + Π) − tr Π: both are O(Π²) results built from m
//!    pivots and diagonal entries of size 1, so each carries about m·ε per
//!    frequency (m = naux). The term is 2·m·ε·Σ_k w_k/2π per dRPA energy in
//!    the field, with ferric's own weights.
//! 6. **Reference reproduction** (`code_residual`): numpy on PySCF's own
//!    fitted tensor vs PySCF's DFMP2/RPA, ≤ 1e-12 by the generator's check.
//! 7. **Frequency quadrature.** Both codes use the Gauss–Legendre map
//!    ω = x0(1+x)/(1−x); ferric's nodes come from its own Newton iteration.
//!    With δ the largest relative node or weight difference, and the dRPA
//!    integrand g(ω) obeying |ω g′| ≤ 4|g| (Π ∝ 1/(ω²+Δ²) gives |ω Π′| ≤ 2Π,
//!    and f(λ) = ln(1+λ) − λ gives |λ f′| ≤ 2|f|), a field's quadrature
//!    difference is at most 5·δ·Σ|E_dRPA| over the dRPA energies in it. 0 for
//!    the MP2 fields.
//! 8. **Floating point** (`fp_floor`): the change of the field when every
//!    integral is perturbed by a relative ±64ε (the generator's
//!    `INT_REL_ERR`), with ferric's factorization: the rounding of ferric's
//!    own transform, factorization and dRPA solve, amplified by the
//!    conditioning of the metric.
//!
//! Asserted, per field and quadrature size:
//!
//! * `|ferric − PySCF| ≤ |metric_shift| + |integral_shift| + floor` —
//!   agreement with PySCF's code at the stated metric and integral
//!   differences;
//! * `|ferric − (PySCF + metric_shift + integral_shift)| ≤ floor` — ferric
//!   lands where its own factorization and integrals predict;
//!
//! with floor = cut_flip + eigenvalues + log-det + residual + quadrature + fp.
//!
//! # Physics hypothesis vs artifact hypothesis
//!
//! If ferric is right, each field sits at PySCF + metric_shift +
//! integral_shift to the floating-point floor. A ring-coefficient slip (1·E_OS for 2·E_OS in the Δ form) moves B
//! `e_corr` by E_OS[erf] (≥ 9e-5 Ha here) while `e_dmp2_lr` (recomputed after
//! the match) still passes; a sign slip, by 4·E_OS[erf]; a swap of the
//! formulations, by B − T (≥ 2e-3 Ha). The controls below are asserted
//! missed by ten times the bar. An operator-sign slip (erf for erfc) moves
//! `e_sr_mp2` by E_MP2[erfc] − E_MP2[erf], which is most of E_MP2.
//!
//! # MUTATION (the ledger patch, `testdata/mutations/rs-mp2-rpa.patch`)
//!
//! `rs_mp2_rpa.rs`, `DeltaLr` arm: `let e_dmp2_lr = 2.0 * sc_lr.e_os;` →
//! `1.0 * sc_lr.e_os` fails the B `e_corr` check of every case.
//!
//! A missing reference JSON is a HARD failure (panic naming the path).

use std::path::{Path, PathBuf};

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_rpa::config::{Eigensolver, PdepRpaConfig, QuadratureConfig, QuadratureScheme};
use ferric_rpa::quadrature::gauss_legendre_nodes;
use ferric_rpa::{rs_mp2_lr_rpa, RsMp2RpaConfig, RsMp2RpaFormulation, RsMp2RpaResult};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::ScfResult;
use ferric_testkit::reference::num_ctx;
use ndarray::Array2;
use serde_json::Value;

const ROW_DIR: &str = "testdata/reference/validation/rs_mp2_rpa_pyscf";
const MOL_DIR: &str = "testdata/molecules/validation";
/// Gauss–Legendre sizes: ferric's default `QuadratureConfig` and 40.
const N_QUADS: [usize; 2] = [20, 40];
const U0: f64 = 0.5;
/// A control is missed by at least this multiple of the field's bar.
const MUST_MISS_FACTOR: f64 = 10.0;
/// Nuclear repulsion: the same geometry to the last digits both codes print.
const TOL_ENUC: f64 = 1e-9;

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

fn reference(system: &str, basis_name: &str) -> Value {
    let path = workspace_root()
        .join(ROW_DIR)
        .join(format!("{system}_{basis_name}.json"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "missing validation reference {} ({e}); regenerate with \
             scripts/validation/gen_rs_mp2_rpa_pyscf.py — a missing reference is a failure",
            path.display()
        )
    });
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{}: bad JSON: {e}", path.display()))
}

fn vec_f64(v: &Value, ptr: &str, ctx: &str) -> Vec<f64> {
    v.pointer(ptr)
        .and_then(Value::as_array)
        .unwrap_or_else(|| panic!("{ctx}: reference array {ptr} missing"))
        .iter()
        .map(|x| x.as_f64().expect("number"))
        .collect()
}

/// PySCF's RHF orbitals (ferric AO order) and energies on ferric's own RHF
/// result as a template.
fn pyscf_orbitals(template: &ScfResult, r: &Value, nocc: usize, ctx: &str) -> ScfResult {
    let rows = r
        .pointer("/rhf/mo_coeff_ferric_order")
        .and_then(Value::as_array)
        .unwrap_or_else(|| panic!("{ctx}: no mo_coeff_ferric_order"));
    let nao = rows.len();
    let nmo = rows[0].as_array().unwrap().len();
    let c = Array2::from_shape_fn((nao, nmo), |(m, p)| rows[m][p].as_f64().unwrap());
    let co = c.slice(ndarray::s![.., ..nocc]);
    let d_alpha = co.dot(&co.t());
    // ferric's RHF stores the per-spin density as half the total.
    let half = &template.density_total * 0.5;
    let ratio = (&template.density_alpha - &half)
        .iter()
        .fold(0.0_f64, |m, x| m.max(x.abs()));
    assert!(
        ratio < 1e-12,
        "{ctx}: ferric's RHF density_alpha is not density_total / 2"
    );
    let mut out = template.clone();
    out.mos_alpha = c;
    out.eps_alpha = vec_f64(r, "/rhf/mo_energy", ctx);
    out.density_total = &d_alpha * 2.0;
    out.density_alpha = d_alpha;
    out.energy = num_ctx(r, "/rhf/energy", ctx);
    out
}

fn rpa_cfg(n_points: usize) -> PdepRpaConfig {
    PdepRpaConfig {
        frozen_core: 0,
        trunc_thresh: 0.0,
        eigensolver: Eigensolver::Lanczos,
        quadrature: QuadratureConfig {
            scheme: QuadratureScheme::GaussLegendre,
            n_points,
            u0: U0,
        },
        ..Default::default()
    }
}

struct Sys {
    label: String,
    r: Value,
    mol: Molecule,
    obs: PreparedBasis,
    dfbs: PreparedBasis,
    rhf: ScfResult,
}

fn load_system(system: &str, basis_name: &str) -> Sys {
    let r = reference(system, basis_name);
    let label = format!("{system}/{basis_name}");
    let xyz = workspace_root().join(MOL_DIR).join(format!("{system}.xyz"));
    let mol = Molecule::load_xyz_with_charge(xyz.to_str().unwrap(), 0, 1)
        .unwrap_or_else(|e| panic!("{}: {e:?}", xyz.display()));
    let enuc = mol.nuclear_repulsion();
    let enuc_ref = num_ctx(&r, "/nuclear_repulsion", &label);
    assert!(
        (enuc - enuc_ref).abs() < TOL_ENUC,
        "{label}: E_nuc {enuc} vs {enuc_ref}"
    );
    assert_eq!(mol.nelec() as i64, r["nelectron"].as_i64().unwrap());
    assert_eq!(r["frozen_core"].as_i64(), Some(0));
    let omega = num_ctx(&r, "/omega_bohr_inv", &label);
    assert!(
        (omega - RsMp2RpaConfig::default().omega).abs() < 1e-15,
        "{label}: the reference omega {omega} is not the production default"
    );
    let obs_bs = basis::bundled(basis_name).unwrap();
    let aux_name = r["aux"].as_str().expect("aux").to_string();
    let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled(&aux_name).unwrap()).unwrap();
    assert_eq!(
        obs.nbasis() as i64,
        r["nao"].as_i64().unwrap(),
        "{label}: nao"
    );
    assert_eq!(
        dfbs.nbasis() as i64,
        r["naux"].as_i64().unwrap(),
        "{label}: naux"
    );

    let bounds = SchwarzBounds::compute(Operator::coulomb(), &obs).unwrap();
    let own = solve_rhf(
        &ParallelContext::default(),
        &mol,
        &obs,
        Operator::coulomb(),
        &bounds,
        &RhfConfig::default()
            .with_max_iter(400)
            .with_energy_conv(1e-11)
            .with_density_conv(1e-10),
    )
    .unwrap_or_else(|e| panic!("{label}: RHF failed: {e:?}"));
    assert!(own.converged, "{label}: RHF did not converge");
    eprintln!(
        "{label}: ferric's own RHF {:+.12} vs PySCF {:+.12} (|d| {:.1e}; not used below)",
        own.energy,
        num_ctx(&r, "/rhf/energy", &label),
        (own.energy - num_ctx(&r, "/rhf/energy", &label)).abs()
    );
    let nocc = (mol.nelec() as usize) / 2;
    let rhf = pyscf_orbitals(&own, &r, nocc, &label);
    check_orthonormal(&label, &obs, &rhf, &r);
    Sys {
        label,
        r,
        mol,
        obs,
        dfbs,
        rhf,
    }
}

/// PySCF's orbitals are orthonormal in ferric's overlap: the AO map and the
/// solid-harmonic convention are right. Bar: the backward error of PySCF's
/// generalized eigensolve, n ε ‖S‖ ‖C‖² ≤ n² ε / λ_min(S) (normalized AOs give
/// ‖S‖ ≤ n, and ‖C‖² = 1/λ_min(S)). A wrong map moves elements by O(0.1).
fn check_orthonormal(ctx: &str, obs: &PreparedBasis, rhf: &ScfResult, r: &Value) {
    let s = ferric_integrals::oneelectron::overlap(obs);
    let c = &rhf.mos_alpha;
    let g = c.t().dot(&s).dot(c);
    let n = c.nrows() as f64;
    let mut worst = 0.0_f64;
    for ((i, j), v) in g.indexed_iter() {
        let want = if i == j { 1.0 } else { 0.0 };
        worst = worst.max((v - want).abs());
    }
    let bar = n * n * f64::EPSILON / num_ctx(r, "/overlap_min_eigenvalue", ctx);
    eprintln!("{ctx}: max |C^T S C - I| {worst:.2e}, bar {bar:.1e}");
    assert!(
        worst < bar,
        "{ctx}: PySCF orbitals are not orthonormal in ferric's overlap ({worst:.2e} > {bar:.1e}): \
         the AO permutation or solid-harmonic convention is wrong"
    );
}

/// δ: the largest relative difference of ferric's Gauss–Legendre nodes and
/// weights from PySCF's.
fn quadrature_delta(ctx: &str, q: &Value, n: usize) -> f64 {
    let (f_nodes, f_wts) = gauss_legendre_nodes(n, U0);
    let p_nodes = vec_f64(q, "/pyscf_nodes", ctx);
    let p_wts = vec_f64(q, "/pyscf_weights", ctx);
    assert_eq!(p_nodes.len(), n, "{ctx}: node count");
    let rel = |a: &[f64], b: &[f64]| {
        let mut a = a.to_vec();
        let mut b = b.to_vec();
        a.sort_by(f64::total_cmp);
        b.sort_by(f64::total_cmp);
        a.iter()
            .zip(&b)
            .map(|(x, y)| ((x - y) / y).abs())
            .fold(0.0_f64, f64::max)
    };
    // Weights sorted by their nodes, so pair them through the node order.
    let pair = |nodes: &[f64], wts: &[f64]| {
        let mut v: Vec<(f64, f64)> = nodes.iter().copied().zip(wts.iter().copied()).collect();
        v.sort_by(|a, b| a.0.total_cmp(&b.0));
        v.into_iter().map(|(_, w)| w).collect::<Vec<f64>>()
    };
    let dn = rel(&f_nodes, &p_nodes);
    let fw = pair(&f_nodes, &f_wts);
    let pw = pair(&p_nodes, &p_wts);
    let dw = fw
        .iter()
        .zip(&pw)
        .map(|(x, y)| ((x - y) / y).abs())
        .fold(0.0_f64, f64::max);
    let d = dn.max(dw);
    eprintln!("{ctx}: GL-{n} nodes vs PySCF: max rel node {dn:.1e}, weight {dw:.1e}");
    // A different map or x0 moves nodes by O(1) relative.
    assert!(
        d < 1e-10,
        "{ctx}: ferric's GL-{n} grid is not PySCF's (δ {d:.1e})"
    );
    d
}

fn run(sys: &Sys, form: RsMp2RpaFormulation, n_points: usize) -> RsMp2RpaResult {
    let cfg = RsMp2RpaConfig {
        formulation: form,
        frozen_core: 0,
        drpa: rpa_cfg(n_points),
        ..Default::default()
    };
    rs_mp2_lr_rpa(&sys.mol, &sys.obs, &sys.dfbs, &sys.rhf, &cfg)
        .unwrap_or_else(|e| panic!("{}: rs_mp2_lr_rpa({form:?}) failed: {e:?}", sys.label))
}

/// The |E_dRPA| that enter each field: (field, kernels whose dRPA it holds).
fn drpa_kernels(field: &str) -> &'static [&'static str] {
    match field {
        "B_e_drpa_lr" | "B_e_corr_naive" | "B_e_corr" => &["erf"],
        "T_e_delta_drpa_full" => &["coulomb"],
        "T_e_delta_drpa_sr" => &["erfc"],
        "T_e_corr" => &["coulomb", "erfc"],
        _ => &[],
    }
}

/// One field: both assertions of the module doc. Returns (|ferric − PySCF|,
/// bar).
fn check_field(
    ctx: &str,
    sys: &Sys,
    q: &Value,
    nq: usize,
    field: &str,
    got: f64,
    dq: f64,
) -> (f64, f64) {
    let p = num_ctx(q, &format!("/fields/{field}"), ctx);
    let metric = num_ctx(q, &format!("/metric_shift/{field}"), ctx);
    let ints = num_ctx(q, &format!("/integral_shift/{field}"), ctx);
    let shift = metric + ints;
    let resid = num_ctx(q, &format!("/code_residual/{field}"), ctx);
    let fp = num_ctx(q, &format!("/fp_floor/{field}"), ctx);
    let flip = num_ctx(q, &format!("/cut_flip/{field}"), ctx);
    let e_drpa: f64 = drpa_kernels(field)
        .iter()
        .map(|k| num_ctx(&sys.r, &format!("/kernels/{k}/pyscf/e_drpa_nw{nq}"), ctx).abs())
        .sum();
    let quad = 5.0 * dq * e_drpa;
    let eig = num_ctx(q, &format!("/eigenvalue_error/{field}"), ctx);
    // Log-det summand rounding, both codes: m·ε per frequency per dRPA kernel.
    let (_, f_wts) = gauss_legendre_nodes(nq, U0);
    let m = num_ctx(&sys.r, "/naux", ctx);
    let n_rings = drpa_kernels(field).len() as f64;
    let logdet =
        2.0 * n_rings * m * f64::EPSILON * f_wts.iter().sum::<f64>() / (2.0 * std::f64::consts::PI);
    let floor = resid + quad + fp + flip + eig + logdet;
    let bar = metric.abs() + ints.abs() + floor;
    let d = (got - p).abs();
    let d_pred = (got - (p + shift)).abs();
    eprintln!(
        "{ctx} GL-{nq}: {field:<22} ferric {got:+.12} PySCF {p:+.12} |d| {d:.2e} (bar {bar:.2e}); \
         shift {metric:+.2e} metric {ints:+.2e} integrals |d - shift| {d_pred:.2e} (floor {floor:.2e}: eig {eig:.1e} logdet {logdet:.1e} fp {fp:.1e} flip {flip:.1e})"
    );
    assert!(
        d <= bar,
        "{ctx} GL-{nq}: {field}: ferric {got:.12} vs PySCF {p:.12} (|d| {d:.2e} Ha) exceeds the \
         bar {bar:.2e} Ha (metric shift {metric:.2e}, integral shift {ints:.2e}, floor {floor:.2e})"
    );
    assert!(
        d_pred <= floor,
        "{ctx} GL-{nq}: {field}: ferric {got:.12} misses PySCF + metric and integral shifts {:.12} by \
         {d_pred:.2e} Ha > floor {floor:.2e} Ha",
        p + shift
    );
    (d, bar)
}

fn assert_misses(ctx: &str, what: &str, got: f64, want: f64, bar: f64) {
    let d = (got - want).abs();
    eprintln!(
        "{ctx}: control {what}: |d| {d:.3e} Ha, must exceed {:.1e}",
        MUST_MISS_FACTOR * bar
    );
    assert!(
        d > MUST_MISS_FACTOR * bar,
        "{ctx}: control '{what}' did not miss: |d| {d:.3e} Ha <= {MUST_MISS_FACTOR} x {bar:.1e}"
    );
}

/// The whole row for one system.
fn run_case(system: &str, basis_name: &str) {
    let sys = load_system(system, basis_name);
    let ctx = sys.label.clone();
    let mut worst = (0.0_f64, 0.0_f64);
    for nq in N_QUADS {
        let q = &sys.r["quadratures"][nq.to_string()];
        let qctx = format!("{ctx} GL-{nq}");
        let dq = quadrature_delta(&qctx, q, nq);
        let b = run(&sys, RsMp2RpaFormulation::DeltaLr, nq);
        let t = run(&sys, RsMp2RpaFormulation::CoupledRings, nq);
        let opt =
            |v: Option<f64>, name: &str| v.unwrap_or_else(|| panic!("{qctx}: {name} missing"));
        let fields = [
            ("e_mp2_full", b.e_mp2_full),
            ("e_sr_mp2", b.e_sr_mp2),
            ("e_lr_mp2", b.e_lr_mp2),
            ("e_dmp2_lr", b.e_dmp2_lr),
            ("B_e_drpa_lr", opt(b.e_drpa_lr, "e_drpa_lr")),
            ("B_e_corr_naive", opt(b.e_corr_naive, "e_corr_naive")),
            ("B_e_corr", b.e_corr),
            ("e_mp2_full", t.e_mp2_full),
            ("e_sr_mp2", t.e_sr_mp2),
            ("e_lr_mp2", t.e_lr_mp2),
            ("e_dmp2_lr", t.e_dmp2_lr),
            (
                "T_e_delta_drpa_full",
                opt(t.e_delta_drpa_full, "e_delta_drpa_full"),
            ),
            (
                "T_e_delta_drpa_sr",
                opt(t.e_delta_drpa_sr, "e_delta_drpa_sr"),
            ),
            ("T_e_corr", t.e_corr),
        ];
        let mut bars = std::collections::BTreeMap::new();
        for (field, got) in fields {
            let (d, bar) = check_field(&ctx, &sys, q, nq, field, got, dq);
            bars.insert(field, bar);
            if field.ends_with("e_corr") {
                worst.0 = worst.0.max(d);
                worst.1 = worst.1.max(bar);
            }
        }
        for (f, e) in [("B", b.e_corr), ("T", t.e_corr)] {
            let bar = bars[format!("{f}_e_corr").as_str()];
            for control in ["dmp2_coeff_1", "wrong_sign"] {
                assert_misses(
                    &qctx,
                    &format!("{f} e_corr vs {f}_control_{control}"),
                    e,
                    num_ctx(q, &format!("/fields/{f}_control_{control}"), &ctx),
                    bar,
                );
            }
        }
        assert_misses(
            &qctx,
            "B e_corr vs the T reference",
            b.e_corr,
            num_ctx(q, "/fields/T_e_corr", &ctx),
            bars["B_e_corr"],
        );
        assert_misses(
            &qctx,
            "T e_corr vs the B reference",
            t.e_corr,
            num_ctx(q, "/fields/B_e_corr", &ctx),
            bars["T_e_corr"],
        );
    }
    eprintln!(
        "{ctx}: worst B/T e_corr |ferric - PySCF| {:.2e} Ha (largest bar {:.2e})",
        worst.0, worst.1
    );
}

#[test]
#[ignore = "validation: RS-MP2-RPA vs PySCF"]
fn rs_mp2_rpa_h2o_cc_pvdz_vs_pyscf() {
    run_case("h2o", "cc-pvdz");
}

#[test]
#[ignore = "validation: RS-MP2-RPA vs PySCF"]
fn rs_mp2_rpa_h2o_aug_cc_pvdz_vs_pyscf() {
    run_case("h2o", "aug-cc-pvdz");
}

#[test]
#[ignore = "validation: RS-MP2-RPA vs PySCF"]
fn rs_mp2_rpa_c2h4_cc_pvdz_vs_pyscf() {
    run_case("c2h4", "cc-pvdz");
}

#[test]
#[ignore = "validation: RS-MP2-RPA vs PySCF"]
fn rs_mp2_rpa_c2h4_aug_cc_pvdz_vs_pyscf() {
    run_case("c2h4", "aug-cc-pvdz");
}

#[test]
#[ignore = "validation: RS-MP2-RPA vs PySCF"]
fn rs_mp2_rpa_water_dimer_cc_pvdz_vs_pyscf() {
    run_case("water_dimer", "cc-pvdz");
}

#[test]
#[ignore = "validation: RS-MP2-RPA vs PySCF"]
fn rs_mp2_rpa_water_dimer_aug_cc_pvdz_vs_pyscf() {
    run_case("water_dimer", "aug-cc-pvdz");
}

/// The auxiliary sets the reference files name (the anchor's envelope).
#[test]
#[ignore = "validation: RS-MP2-RPA vs PySCF"]
fn rs_mp2_rpa_pyscf_aux_sets_are_named() {
    for aux in ["cc-pvdz-ri", "aug-cc-pvdz-rifit"] {
        assert!(basis::bundled(aux).is_ok(), "{aux}");
    }
}
