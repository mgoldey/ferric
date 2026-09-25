//! VALIDATION tier — VALIDATION.md row "BSE-TDA".
//!
//! Runs only in the weekly `validation` CI job (and on demand):
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 cargo nextest run -p ferric-gw --release \
//!     --test validation_bse --run-ignored only -E 'binary(/^validation_/)'
//! ```
//!
//! # What is compared
//!
//! `run_bse_tda` (singlet BSE-TDA on G0W0@HF) and `run_cis_tda` against an
//! INDEPENDENT numpy BSE-TDA (`scripts/validation/gen_bse.py`) built from PySCF
//! 2.13 ingredients: exact-integral RHF, `gw_ac`'s density-fitted Lpq with
//! ferric's aux, and `gw_ac.get_sigma` for Σc. PySCF has no BSE.
//!
//! | test | systems / basis (aux) |
//! |---|---|
//! | BSE-TDA singlets Ω, oscillator strengths | H2O / cc-pVDZ (cc-pvdz-ri), aug-cc-pVDZ (aug-cc-pvdz-rifit); NH3, CH2O / cc-pVDZ |
//! | CIS-TDA (`run_cis_tda`) | the same four |
//!
//! Lowest five singlets per system, extended to close a degenerate group
//! (NH3's E pairs); oscillator strengths are compared as sums over each
//! degenerate group, which are rotation invariant.
//!
//! References: `testdata/reference/validation/bse/<system>_<basis>.json`.
//!
//! # The conventions mirrored (read from `crates/ferric-gw/src/bse.rs`)
//!
//! * Singlet TDA only (bse.rs:300-332): A_{ia,jb} = (ε^QP_a − ε^QP_i) δ +
//!   2(ia|jb) − (ab|W|ij), (ia|jb) the bare DF integral from `b_full`
//!   (bse.rs:285-291, :326-328). ferric has no triplet and no B block here.
//! * W is STATIC and comes from the same PDEP run as the G0W0 step:
//!   (pq|W|rs) = (pq|rs) + Σ_α (1/λ_α(0) − 1) M_α,pq M_α,rs (bse.rs:253-298).
//!   At `trunc_thresh = 0` the mode set is the whole RI space, so W =
//!   Lᵀ(I + Π(0))⁻¹L with Π(0) = 4 Σ_ia L_ia L_iaᵀ/(ε_a − ε_i) on the HF
//!   energies (G0W0 never rebuilds W), all occupied × all virtual.
//! * QP energies: `run_bse_tda` always runs its own G0W0@HF for EVERY MO
//!   (`qp_mos: Some(0..nmo)`, bse.rs:208-226). It takes no external QP
//!   energies or scissor. The generator re-implements ferric's QP solve
//!   (sigma.rs `solve_qp_for_mo`: textbook Thiele, linearized start with
//!   Z ∈ [0, 1.5], 4-point derivative h = 0.05, ≤ 30 Newton steps to
//!   |step| < 1e-7) on PySCF's Σc(ef + iω) at the G0W0 row's matched settings
//!   (100-point GL W integral, u₀ = 0.5, 18 Padé nodes, no cut).
//! * Frozen core 0 (with frozen_core > 0 the all-MO QP range reaches the
//!   frozen block and `run_gw` refuses it), ia = i·nvir + a (bse.rs:315-325).
//! * Oscillator strengths: length gauge, f = ⅔ Ω |√2 Σ X_ia ⟨i|r|a⟩|²
//!   (bse.rs:111-183).
//!
//! # Separating BSE from GW
//!
//! Only the diagonal of A depends on the QP energies. The reference stores
//! the QP-independent kernel K = 2(ia|jb) − (ab|W|ij) (`bse_kernel`, upper
//! triangle) and the occ–vir dipoles, and the headline comparison
//! diagonalizes K + diag(ε^QP_a − ε^QP_i) with FERRIC'S OWN QP energies
//! (`BseResult::eps_qp`). That comparison contains no GW difference: it tests
//! the kernel, the assembly and the oscillator strengths.
//!
//! Why this matters: outside HOMO−2 … LUMO+2 the G0W0 QP energies are not
//! reproducible. The Thiele continuation far from ef is ill-conditioned; the
//! generator measured (`qp.sensitivity`) that a RELATIVE 1e-10 perturbation of
//! Σc(iω) at the nodes moves core and high-virtual QP energies by up to
//! 0.23–0.29 Ha, and two runs of the generator itself differ by 1.9e-3 Ha on
//! H2O/cc-pVDZ MO 23. ferric's QP energies there are equally arbitrary, so the
//! raw Ω (on each code's own QP energies) is compared at a looser bar, and
//! each MO's QP energy at a bar scaled by its measured sensitivity.
//!
//! For NH3 that noise also splits the QP energies of degenerate orbitals, so
//! the diagonal is not invariant under a rotation inside the degenerate set
//! and Ω depends on each code's arbitrary MO rotation
//! (`degenerate_rotation_ambiguity`, 1.9e-8 to 4.9e-8 Ha across generator
//! runs; zero for the C2v systems). NH3 gets its own bar for that reason.
//!
//! # Physics hypothesis vs artifact hypothesis
//!
//! * If ferric's BSE kernel is right: on ferric's own QP energies the only
//!   differences left are the integral engines, the SCF and the eigensolver,
//!   and Ω agrees far below the spacing of the controls (≥ 0.02 Ha).
//! * Wrong singlet factor (2(ia|jb) dropped or doubled): Ω lands on or near
//!   the TRIPLET control (0.03–0.11 Ha away).
//! * W not reaching the kernel (bare exchange, or no exchange): Ω lands on the
//!   `bse_w_bare` (CIS on QP energies, 0.02–0.07 Ha away) or `bse_w_off`
//!   control (0.21–0.36 Ha away).
//! * W on the wrong orbital energies, a missing spin factor in Π, or a wrong
//!   mode weight (1/λ vs 1/λ − 1 with the bare part): Ω moves by a fraction
//!   of the screening itself (λ_max(0) = 2.2–2.6 here), tens of meV.
//! * QP energies not threaded into the diagonal (HF ε instead): the raw Ω
//!   misses by the GW gap correction (eV) and the QP comparison fails, while
//!   the kernel comparison (which uses ferric's reported `eps_qp`) catches the
//!   case where the reported and the used energies differ.
//! * HARNESS error (geometry, basis, aux, a mislabelled file, the kernel's
//!   storage layout): E_nuc, the AO and aux counts, E_RHF, ε_mf and the
//!   kernel read-back anchor fail first; the basis-swap control shows the
//!   comparison resolves the basis.
//!
//! # Exactness anchors
//!
//! 1. (generator) The numpy A builder fed EXACT MO integrals, HF energies and
//!    W → v reproduces PySCF `tdscf.rhf.get_ab` to ≤ 5.1e-14 — layout, the
//!    singlet factor and the exchange index order are PySCF's, independently.
//! 2. (here) The stored kernel, read back by THIS file and given the
//!    reference's own QP energies, reproduces the stored singlet spectrum
//!    (`TOL_READBACK`): the test reads what the generator wrote.
//! 3. (here) ferric's `run_cis_tda` — the same assembly with W → v and HF
//!    energies — equals the numpy DF-CIS; and, with a bar set by the DF
//!    fitting error of the MP2-type aux (measured ≤ 6.8e-4 Ha, H2O/cc-pVDZ),
//!    PySCF's exact-integral TDA. `tdscf.TDA`'s Davidson roots equal the dense
//!    get_ab spectrum to ≤ 2e-13 Ha.
//! 4. (generator) ferric's QP recipe on PySCF's Σc reproduces PySCF's own
//!    Padé roots, and the G0W0 row's stored QP energies, in HOMO−2 … LUMO+2
//!    to ≤ 1.6e-7 Ha (the textbook vs PySCF Padé evaluation).
//!
//! # TOLERANCES
//!
//! Each bar is 3–25× the worst case measured over all four systems
//! (2026-09-25), noted beside its const; the degenerate-orbital bar is set
//! by the generator's rotation-ambiguity measurement instead.
//!
//! # NEGATIVE CONTROLS (asserted in the tests)
//!
//! * Singlet vs triplet: ferric's Ω misses the triplet reference (A = D − W).
//! * W on vs off: ferric's Ω misses `bse_w_off` (A = D + 2(ia|jb)) and
//!   `bse_w_bare` (A = D + 2(ia|jb) − (ab|ij), CIS on QP energies).
//! * Basis swap: ferric's H2O/cc-pVDZ Ω misses the aug-cc-pVDZ reference and
//!   vice versa.
//! * BSE is not CIS: ferric's `run_bse_tda` misses the CIS reference, and
//!   `run_cis_tda` misses the BSE reference.
//!
//! # MUTATIONS (run 2026-09-25)
//!
//! * bse.rs, the singlet kernel `row[jb] = 2.0 * coul - scr;` → triplet
//!   (`-scr`): all three `bse_tda_*` tests fail; the CIS anchor (separate path)
//!   passes.
//! * bse.rs, the screened-interaction factor `.map(|&l| 1.0 / l - 1.0)` → 0
//!   (W → bare v): all three `bse_tda_*` tests fail; the CIS anchor passes.

use std::path::{Path, PathBuf};

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_gw::bse::{run_bse_tda, run_cis_tda, BseResult};
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_rpa::config::{Eigensolver, PdepRpaConfig, QuadratureConfig, QuadratureScheme};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::ScfResult;
use ndarray::Array2;
use ndarray_linalg::{Eigh, UPLO};
use serde_json::Value;

const ROW_DIR: &str = "testdata/reference/validation/bse";
const MOL_DIR: &str = "testdata/molecules/validation";
const HA_TO_EV: f64 = 27.211_386_245_988;
/// W frequency points; `gen_gw.NW` in the generator.
const N_QUAD: usize = 100;

// Ω from ferric vs the reference kernel diagonalized on FERRIC's QP energies
// (no GW difference left): integral engines, SCF and eigensolver only.
// Measured 1.8e-10 Ha (CH2O, H2O/aug-cc-pVDZ).
const TOL_OMEGA: f64 = 2e-9;

// The same, for a system with degenerate orbitals (NH3): the QP splitting of
// degenerate orbitals makes Ω depend on the MO rotation by up to ~5e-8 Ha
// (generator `degenerate_rotation_ambiguity`).
// This run measured 5.9e-11 Ha; the bar covers the generator's measured
// rotation ambiguity (up to 4.9e-8), which depends on each code's arbitrary
// rotation within the degenerate pair, not on this run's value.
const TOL_OMEGA_DEGENERATE: f64 = 5e-7;

// Oscillator strength (group sums), ferric vs the kernel on ferric's QP.
// Measured 2.6e-9 (CH2O).
const TOL_F: f64 = 3e-8;

// Raw Ω: ferric on its own QP energies vs the reference on the generator's.
// Carries the ill-conditioned core / high-virtual QP differences.
// Measured 2.2e-7 Ha (NH3): the ill-conditioned core / high-virtual QP
// energies move the lowest five excitations by at most this much.
const TOL_OMEGA_RAW: f64 = 2e-6;

// QP energies in HOMO−2 … LUMO+2 (the G0W0 row's bar there).
// Measured 8.1e-9 Ha (H2O/cc-pVDZ LUMO).
const TOL_QP_WINDOW: f64 = 1e-7;

// Every MO: |d| < max(TOL_QP_WINDOW, QP_SENS_FACTOR × sensitivity_p), the
// generator's measured per-MO conditioning (qp.sensitivity).
const QP_SENS_FACTOR: f64 = 10.0;

// Reading back the stored kernel with the reference's own QP energies.
const TOL_READBACK: f64 = 1e-10;

// run_cis_tda vs the numpy DF-CIS (same integrals, no GW).
// Measured 1.6e-9 Ha (H2O/aug-cc-pVDZ).
const TOL_CIS: f64 = 2e-8;

// Oscillator strength (group sums), run_cis_tda vs the numpy DF-CIS.
// Measured 3.6e-9 (CH2O).
const TOL_CIS_F: f64 = 3e-8;

// run_cis_tda vs PySCF exact-integral TDA: the DF fitting error of the
// MP2-type aux, measured by the generator at 6.8e-4 Ha (H2O/cc-pVDZ),
// 4.5e-4 (NH3), 1.2e-4 (CH2O), 3.2e-5 (H2O/aug-cc-pVDZ).
const TOL_CIS_VS_EXACT: f64 = 2e-3;

const TOL_E_SCF: f64 = 1e-9;
const TOL_EPS_MF: f64 = 3e-8;
const TOL_ENUC: f64 = 1e-9;
/// A reference ferric must MISS is missed by at least this multiple of the bar.
const MUST_MISS_FACTOR: f64 = 10.0;

// ---------------------------------------------------------------------------
// Reference plumbing
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
             scripts/validation/gen_bse.py — a missing reference is a failure, never a skip",
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

/// The stored kernel triangle: base64 of little-endian f64 bytes
/// (`gen_bse.py::encode_kernel`; a JSON array would exceed the repo's
/// 500 KB large-file limit for CH2O).
fn kernel_upper(r: &Value, ctx: &str) -> Vec<f64> {
    let text = r
        .pointer("/bse_kernel/upper_f64le_b64")
        .and_then(Value::as_str)
        .unwrap_or_else(|| panic!("{ctx}: reference field /bse_kernel/upper_f64le_b64 missing"));
    let bytes = base64_decode(text);
    assert_eq!(
        bytes.len() % 8,
        0,
        "{ctx}: kernel byte length {}",
        bytes.len()
    );
    bytes
        .as_chunks::<8>()
        .0
        .iter()
        .map(|c| f64::from_le_bytes(*c))
        .collect()
}

/// Standard-alphabet base64 with `=` padding (what Python's b64encode writes).
fn base64_decode(text: &str) -> Vec<u8> {
    fn val(c: u8) -> u32 {
        match c {
            b'A'..=b'Z' => (c - b'A') as u32,
            b'a'..=b'z' => (c - b'a' + 26) as u32,
            b'0'..=b'9' => (c - b'0' + 52) as u32,
            b'+' => 62,
            b'/' => 63,
            _ => panic!("base64: invalid byte {c}"),
        }
    }
    let raw = text.as_bytes();
    assert_eq!(
        raw.len() % 4,
        0,
        "base64: length {} is not a multiple of 4",
        raw.len()
    );
    let mut out = Vec::with_capacity(raw.len() / 4 * 3);
    for quad in raw.as_chunks::<4>().0 {
        let pad = quad.iter().rev().take_while(|&&c| c == b'=').count();
        let mut n = 0u32;
        for &c in &quad[..4 - pad] {
            n = (n << 6) | val(c);
        }
        n <<= 6 * pad as u32;
        let b = [(n >> 16) as u8, (n >> 8) as u8, n as u8];
        out.extend_from_slice(&b[..3 - pad]);
    }
    out
}

#[test]
#[ignore = "validation: BSE-TDA"]
fn base64_decoder_matches_python_b64encode() {
    // Python: base64.b64encode(struct.pack("<2d", 1.0, -0.25)).decode()
    let bytes = base64_decode("AAAAAAAA8D8AAAAAAADQvw==");
    let v: Vec<f64> = bytes
        .as_chunks::<8>()
        .0
        .iter()
        .map(|c| f64::from_le_bytes(*c))
        .collect();
    assert_eq!(v, vec![1.0, -0.25]);
    assert_eq!(base64_decode("TWFu"), b"Man");
    assert_eq!(base64_decode("TWE="), b"Ma");
    assert_eq!(base64_decode("TQ=="), b"M");
}

fn groups(v: &Value, ptr: &str, ctx: &str) -> Vec<Vec<usize>> {
    v.pointer(ptr)
        .and_then(Value::as_array)
        .unwrap_or_else(|| panic!("{ctx}: reference field {ptr} missing or not an array"))
        .iter()
        .map(|g| {
            g.as_array()
                .expect("group")
                .iter()
                .map(|k| k.as_u64().expect("state index") as usize)
                .collect()
        })
        .collect()
}

fn check_close(ctx: &str, what: &str, got: f64, want: f64, tol: f64) {
    let d = (got - want).abs();
    eprintln!(
        "{ctx}: {what:<34} ferric {got:+.10} ref {want:+.10} |d| {d:.2e} ({:.4} meV)",
        d * HA_TO_EV * 1e3
    );
    assert!(
        d < tol,
        "{ctx}: {what}: ferric {got:.10} vs reference {want:.10} (|d| {d:.2e} Ha) exceeds {tol:.1e}"
    );
}

/// Print every element's difference first, then assert the worst one.
fn check_vec(ctx: &str, what: &str, got: &[f64], want: &[f64], tol: f64) -> f64 {
    assert_eq!(got.len(), want.len(), "{ctx}: {what} length");
    let mut worst = (0.0_f64, 0usize);
    for (k, (&g, &w)) in got.iter().zip(want).enumerate() {
        let d = (g - w).abs();
        eprintln!(
            "{ctx}: {:<34} ferric {g:+.10} ref {w:+.10} |d| {d:.2e}",
            format!("{what}[{k}]")
        );
        if d > worst.0 {
            worst = (d, k);
        }
    }
    let (d, k) = worst;
    eprintln!(
        "{ctx}: {what} WORST |d| {d:.3e} ({:.4} meV if Ha) at {k}",
        d * HA_TO_EV * 1e3
    );
    assert!(
        d < tol,
        "{ctx}: {what}[{k}]: ferric {:.10} vs reference {:.10} (|d| {d:.2e}) exceeds {tol:.1e}",
        got[k],
        want[k]
    );
    d
}

/// max_k |got_k − want_k| over the common prefix must exceed `factor × tol`.
fn assert_misses(ctx: &str, what: &str, got: &[f64], want: &[f64], tol: f64) {
    let n = got.len().min(want.len());
    assert!(n > 0, "{ctx}: control '{what}' has no states");
    let d = got[..n]
        .iter()
        .zip(&want[..n])
        .map(|(g, w)| (g - w).abs())
        .fold(0.0_f64, f64::max);
    eprintln!(
        "{ctx}: control {what}: max |d| {d:.3e} Ha ({:.2} meV), must exceed {:.1e}",
        d * HA_TO_EV * 1e3,
        MUST_MISS_FACTOR * tol
    );
    assert!(
        d > MUST_MISS_FACTOR * tol,
        "{ctx}: negative control '{what}' did not miss: max |d| {d:.3e} <= {MUST_MISS_FACTOR} x \
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
    scf: ScfResult,
}

fn load_system(system: &str, basis_name: &str) -> Sys {
    let r = reference(system, basis_name);
    let label = format!("{system}/{basis_name}");
    let xyz = workspace_root().join(MOL_DIR).join(format!("{system}.xyz"));
    let mol = Molecule::load_xyz_with_charge(xyz.to_str().unwrap(), 0, 1)
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
    let cfg = RhfConfig {
        max_iter: 400,
        energy_conv: 1e-11,
        density_conv: 1e-10,
        ..Default::default()
    };
    let scf = solve_rhf(
        &ParallelContext::default(),
        &mol,
        &obs,
        Operator::coulomb(),
        &bounds,
        &cfg,
    )
    .unwrap_or_else(|e| panic!("{label}: RHF failed: {e:?}"));
    assert!(scf.converged, "{label}: RHF did not converge");
    check_close(
        &label,
        "E_RHF",
        scf.energy,
        num(&r, "/rhf/energy", &label),
        TOL_E_SCF,
    );
    check_vec(
        &label,
        "eps_mf",
        scf.eps_r(),
        &vec_f64(&r, "/qp/eps_mf", &label),
        TOL_EPS_MF,
    );
    Sys {
        label,
        r,
        mol,
        obs,
        dfbs,
        scf,
    }
}

/// The G0W0 row's like-for-like PDEP settings: full rank, 100-point GL.
fn pdep_cfg() -> PdepRpaConfig {
    PdepRpaConfig {
        frozen_core: 0,
        trunc_thresh: 0.0,
        eigensolver: Eigensolver::Lanczos,
        quadrature: QuadratureConfig {
            scheme: QuadratureScheme::GaussLegendre,
            n_points: N_QUAD,
            u0: 0.5,
        },
        need_inv_dielectric_freq: true,
        need_eigenvalues_freq: true,
        ..Default::default()
    }
}

fn run_bse(sys: &Sys) -> BseResult {
    run_bse_tda(
        &sys.mol,
        &sys.obs,
        &sys.dfbs,
        Operator::coulomb(),
        &sys.scf,
        &pdep_cfg(),
        0,
    )
    .unwrap_or_else(|e| panic!("{}: run_bse_tda failed: {e:?}", sys.label))
}

/// The reference kernel K (upper triangle, row-major) diagonalized with
/// `eps_qp` on the diagonal: the lowest `n` Ω and length-gauge f, with the
/// same formula as bse.rs `tda_oscillator_strengths`.
fn kernel_spectrum(sys: &Sys, eps_qp: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let ctx = &sys.label;
    let r = &sys.r;
    let nocc = r["nocc"].as_u64().unwrap() as usize;
    let nvir = r["nvir"].as_u64().unwrap() as usize;
    let nov = nocc * nvir;
    assert_eq!(
        r["bse_kernel"]["nov"].as_u64(),
        Some(nov as u64),
        "{ctx}: nov"
    );
    assert_eq!(eps_qp.len(), nocc + nvir, "{ctx}: eps_qp length");
    let upper = kernel_upper(r, ctx);
    assert_eq!(upper.len(), nov * (nov + 1) / 2, "{ctx}: kernel length");
    let mut a = Array2::<f64>::zeros((nov, nov));
    let mut k = 0;
    for p in 0..nov {
        for q in p..nov {
            a[(p, q)] = upper[k];
            a[(q, p)] = upper[k];
            k += 1;
        }
    }
    for i in 0..nocc {
        for b in 0..nvir {
            a[(i * nvir + b, i * nvir + b)] += eps_qp[nocc + b] - eps_qp[i];
        }
    }
    let (w, x) = a
        .eigh(UPLO::Upper)
        .unwrap_or_else(|e| panic!("{ctx}: eigh of the reference kernel: {e}"));
    let dip: Vec<Vec<f64>> = (0..3)
        .map(|d| vec_f64(r, &format!("/dipole_ia/values/{d}"), ctx))
        .collect();
    let f = (0..n)
        .map(|s| {
            let mu2: f64 = dip
                .iter()
                .map(|dd| {
                    let m = std::f64::consts::SQRT_2
                        * (0..nov).map(|ia| x[(ia, s)] * dd[ia]).sum::<f64>();
                    m * m
                })
                .sum();
            (2.0 / 3.0) * w[s] * mu2
        })
        .collect();
    (w.iter().take(n).copied().collect(), f)
}

fn group_sums(f: &[f64], gs: &[Vec<usize>]) -> Vec<f64> {
    gs.iter().map(|g| g.iter().map(|&k| f[k]).sum()).collect()
}

/// Every comparison and control for one system; returns ferric's raw Ω.
fn bse_case(system: &str, basis_name: &str) -> Vec<f64> {
    let sys = load_system(system, basis_name);
    let ctx = format!("{} BSE-TDA", sys.label);
    let r = &sys.r;
    let om_ref = vec_f64(r, "/bse_singlet/omega", &ctx);
    let n = om_ref.len();
    let gs = groups(r, "/bse_singlet/groups", &ctx);

    // Anchor: the stored kernel, read back here, with the reference's own QP
    // energies reproduces the stored spectrum.
    let eps_ref = vec_f64(r, "/qp/eps_qp", &ctx);
    let (om_back, f_back) = kernel_spectrum(&sys, &eps_ref, n);
    check_vec(
        &ctx,
        "kernel read-back Omega",
        &om_back,
        &om_ref,
        TOL_READBACK,
    );
    check_vec(
        &ctx,
        "kernel read-back f (groups)",
        &group_sums(&f_back, &gs),
        &vec_f64(r, "/bse_singlet/group_osc_strength", &ctx),
        TOL_READBACK,
    );

    let res = run_bse(&sys);
    assert_eq!(res.nocc as u64, r["nocc"].as_u64().unwrap(), "{ctx}: nocc");
    assert_eq!(res.nvir as u64, r["nvir"].as_u64().unwrap(), "{ctx}: nvir");
    assert!(
        res.omega.len() >= n,
        "{ctx}: ferric returned too few states"
    );

    // QP energies: the window at the G0W0 row's bar, then every MO at a bar
    // scaled by its measured Padé conditioning.
    let win: Vec<usize> = r["qp"]["window"]
        .as_array()
        .expect("window")
        .iter()
        .map(|x| x.as_u64().unwrap() as usize)
        .collect();
    let got_w: Vec<f64> = win.iter().map(|&p| res.eps_qp[p]).collect();
    let want_w: Vec<f64> = win.iter().map(|&p| eps_ref[p]).collect();
    check_vec(
        &ctx,
        "eps_qp(HOMO-2..LUMO+2)",
        &got_w,
        &want_w,
        TOL_QP_WINDOW,
    );
    let sens = vec_f64(r, "/qp/sensitivity", &ctx);
    let mut worst_ratio = (0.0_f64, 0usize);
    for (p, ((&g, &w), &s)) in res.eps_qp.iter().zip(&eps_ref).zip(&sens).enumerate() {
        let d = (g - w).abs();
        let bar = TOL_QP_WINDOW.max(QP_SENS_FACTOR * s);
        eprintln!(
            "{ctx}: eps_qp[{p:>2}] ferric {g:+.10} ref {w:+.10} |d| {d:.2e} sensitivity {s:.1e} bar {bar:.1e}"
        );
        if d / bar > worst_ratio.0 {
            worst_ratio = (d / bar, p);
        }
    }
    let (ratio, p) = worst_ratio;
    assert!(
        ratio < 1.0,
        "{ctx}: eps_qp[{p}] differs by {:.2e} Ha, {ratio:.2} x its bar max({TOL_QP_WINDOW:.0e}, \
         {QP_SENS_FACTOR} x sensitivity {:.1e})",
        (res.eps_qp[p] - eps_ref[p]).abs(),
        sens[p]
    );

    // HEADLINE: the reference kernel on FERRIC's QP energies.
    let (om_sub, f_sub) = kernel_spectrum(&sys, &res.eps_qp, n);
    let degenerate = num(r, "/degenerate_rotation_ambiguity", &ctx) > 0.0;
    let tol = if degenerate {
        TOL_OMEGA_DEGENERATE
    } else {
        TOL_OMEGA
    };
    check_vec(
        &ctx,
        "Omega vs kernel on ferric's QP",
        &res.omega[..n],
        &om_sub,
        tol,
    );
    check_vec(
        &ctx,
        "f (groups) vs kernel on ferric's QP",
        &group_sums(&res.oscillator_strength[..n], &gs),
        &group_sums(&f_sub, &gs),
        TOL_F,
    );

    // Raw: each code on its own QP energies.
    let raw = res.omega[..n].to_vec();
    check_vec(&ctx, "Omega (raw, own QP)", &raw, &om_ref, TOL_OMEGA_RAW);

    // Negative controls: singlet vs triplet, W on vs off, BSE vs CIS.
    for (what, ptr) in [
        ("triplet (A = D - W)", "/controls/bse_triplet/omega"),
        ("W off (A = D + 2v)", "/controls/bse_w_off/omega"),
        ("W -> bare v (CIS on QP)", "/controls/bse_w_bare/omega"),
        ("CIS on HF", "/cis_df/omega"),
    ] {
        assert_misses(&ctx, what, &raw, &vec_f64(r, ptr, &ctx), TOL_OMEGA_RAW);
    }
    eprintln!(
        "{ctx}: lowest singlet {:.6} eV (ref {:.6} eV)",
        res.omega[0] * HA_TO_EV,
        om_ref[0] * HA_TO_EV
    );
    raw
}

// ---------------------------------------------------------------------------
// BSE-TDA
// ---------------------------------------------------------------------------

#[test]
#[ignore = "validation: BSE-TDA"]
fn bse_tda_h2o_vs_numpy() {
    let bases = ["cc-pvdz", "aug-cc-pvdz"];
    let om: Vec<Vec<f64>> = bases.iter().map(|b| bse_case("h2o", b)).collect();
    // Basis swap: each basis's ferric Ω misses the other basis's reference.
    for (i, b) in bases.iter().enumerate() {
        let other = bases[1 - i];
        assert_misses(
            &format!("h2o/{b}"),
            &format!("vs the {other} reference"),
            &om[i],
            &vec_f64(&reference("h2o", other), "/bse_singlet/omega", "h2o"),
            TOL_OMEGA_RAW,
        );
    }
}

#[test]
#[ignore = "validation: BSE-TDA"]
fn bse_tda_nh3_vs_numpy() {
    bse_case("nh3", "cc-pvdz");
}

#[test]
#[ignore = "validation: BSE-TDA"]
fn bse_tda_ch2o_vs_numpy() {
    bse_case("ch2o", "cc-pvdz");
}

// ---------------------------------------------------------------------------
// CIS-TDA anchor (W -> v, HF energies)
// ---------------------------------------------------------------------------

fn cis_case(system: &str, basis_name: &str) {
    let sys = load_system(system, basis_name);
    let ctx = format!("{} CIS-TDA", sys.label);
    let r = &sys.r;
    let res = run_cis_tda(
        &sys.mol,
        &sys.obs,
        &sys.dfbs,
        Operator::coulomb(),
        &sys.scf,
        0,
    )
    .unwrap_or_else(|e| panic!("{ctx}: run_cis_tda failed: {e:?}"));
    let want = vec_f64(r, "/cis_df/omega", &ctx);
    let n = want.len();
    let got = &res.omega[..n];
    check_vec(&ctx, "Omega vs numpy DF-CIS", got, &want, TOL_CIS);
    let gs = groups(r, "/cis_df/groups", &ctx);
    check_vec(
        &ctx,
        "f (group sums) vs numpy DF-CIS",
        &group_sums(&res.oscillator_strength[..n], &gs),
        &vec_f64(r, "/cis_df/group_osc_strength", &ctx),
        TOL_CIS_F,
    );
    // Exact-integral PySCF TDA: differs by the DF fitting error only.
    let exact = vec_f64(r, "/cis_exact_tdscf/omega", &ctx);
    let d_exact = check_vec(
        &ctx,
        "Omega vs PySCF TDA (exact)",
        got,
        &exact[..n],
        TOL_CIS_VS_EXACT,
    );
    let recorded = num(r, "/cis_df_vs_exact_max", &ctx);
    eprintln!("{ctx}: DF error vs exact TDA {d_exact:.3e} Ha (generator: {recorded:.3e})");
    // Control: CIS is not BSE.
    assert_misses(
        &ctx,
        "vs the BSE singlet reference",
        got,
        &vec_f64(r, "/bse_singlet/omega", &ctx),
        TOL_CIS,
    );
}

#[test]
#[ignore = "validation: BSE-TDA"]
fn cis_tda_anchor_vs_numpy_and_pyscf_tda() {
    cis_case("h2o", "cc-pvdz");
    cis_case("h2o", "aug-cc-pvdz");
    cis_case("nh3", "cc-pvdz");
    cis_case("ch2o", "cc-pvdz");
}
