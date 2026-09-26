//! VALIDATION tier — VALIDATION.md row "Harmonic frequencies".
//!
//! Runs only in the weekly `validation` CI job (and on demand):
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 cargo nextest run -p ferric-scf --test validation_frequencies \
//!     --run-ignored only -E 'binary(/^validation_/)'
//! ```
//!
//! # What ferric computes (read from `crates/ferric-scf/src/frequencies.rs`)
//!
//! ferric has NO analytic second derivatives. `harmonic_frequencies` builds the
//! Cartesian Hessian by a CENTRAL FINITE DIFFERENCE of the ANALYTIC nuclear
//! gradient, step `DEFAULT_DELTA` = 5e-3 Bohr (one fresh SCF + gradient per
//! ± step on each of the 3N coordinates), symmetrizes it, mass-weights it with
//! `atom_masses` (IUPAC-2013 isotope averages), projects the mass-weighted
//! translations and rotations about the centre of mass out on both sides
//! (P H P) and diagonalizes. The FD step therefore carries an O(δ²) truncation
//! error (third derivatives): PySCF's own FD-vs-analytic gap at the same step
//! is recorded in every reference block (`pyscf_fd_vs_analytic_*`) and is
//! ~0.1 cm⁻¹ on the OH stretch.
//!
//! # References (`scripts/validation/gen_frequencies.py` → `testdata/reference/validation/frequencies/`)
//!
//! PySCF 2.13 fed ferric's own basis JSON, Bohr geometry and MASSES:
//!
//! * `hessian_fd` — central FD of PySCF's ANALYTIC gradient at the SAME step
//!   (KS with `grid_response = True`), symmetrized. Identical construction to
//!   ferric's, so the truncation error cancels and the comparison floor is
//!   SCF/grid noise. This is the TIGHT reference for every method.
//! * `hessian_analytic` — PySCF's analytic Hessian (`hessian.rhf/rks/uhf/uks`),
//!   stored symmetrized (PySCF's KS Hessian is asymmetric by up to 6.4e-7
//!   Ha/Bohr² and its `harmonic_analysis` reads one triangle; ferric
//!   symmetrizes before diagonalizing).
//!   For HF it is the exact second derivative, so ferric must match it to the
//!   FD truncation. For KS it is NOT like-for-like: PySCF's semilocal KS
//!   Hessian omits the grid-response term, while ferric's KS gradient
//!   includes grid response (`ferric_dft::gradient`,
//!   `build_atomic_grid_with_response`; the module doc there saying "no grid
//!   response" is stale — the code builds and uses `weight1`), so ferric's FD
//!   Hessian differentiates it. The KS comparison to the analytic Hessian is
//!   therefore held to a looser bar sized by PySCF's own FD-vs-analytic gap.
//! * ROHF: PySCF has no ROHF Hessian. The reference is FD of `grad.rohf` at
//!   steps 1e-2 / 5e-3 / 2.5e-3 plus the Richardson extrapolate; ferric is
//!   held tightly to the 5e-3 FD and loosely to the extrapolate.
//! * Frequencies: `hessian.thermo.harmonic_analysis(mass = ferric's masses,
//!   imaginary_freq = False)`, ascending, imaginary as negative.
//!
//! Systems: H2O, NH3 × {RHF, PBE, B3LYP} × {6-31G, cc-pVDZ}; UHF for OH (²Π),
//! planar CH3 (²A2''), HO2 (²A'') at 6-31G; UKS-PBE and ROHF for CH3 at 6-31G.
//! J/K are EXACT (4-index) on both sides: `df_j_aux = df_k_aux = Some("")`,
//! the explicit no-fit sentinel — `None` would auto-enable RI-J for closed-
//! shell KS (`rhf.rs` `resolve_aux`). Grid: ferric's default (75,110)
//! unpruned Becke with Becke-1988 radii == the generator's PySCF grid.
//!
//! # Projection at a non-stationary geometry
//!
//! Geometries are used AS GIVEN (experimental / HF-optimized, not optimized
//! at each level), so the gradient is not zero. That is harmless for this
//! comparison because both codes project the SAME subspace: PySCF
//! (`thermo._get_TR`) spans the mass-weighted translations and the rotations
//! about the centre of mass in the principal-axis frame, ferric
//! (`translation_rotation_basis`) the same rotations in the lab frame — the
//! same span. PySCF restricts H to the complement; ferric diagonalizes P H P
//! and discards the T/R eigenvectors by overlap. Identical nonzero spectra.
//! The claim is TESTED, not assumed: `anchor_postprocessing_matches_pyscf`
//! feeds PySCF's own Hessian through ferric's `frequencies_from_cartesian_hessian`
//! and must reproduce PySCF's frequencies with no SCF involved. The
//! Cartesian Hessian is additionally asserted elementwise, which does not
//! depend on any projection at all.
//!
//! # Physics hypothesis vs artifact hypothesis
//!
//! * If ferric is right: the Cartesian Hessian matches `hessian_fd` at the SCF
//!   noise floor (~1e-7..1e-6 Ha/Bohr²) and the frequencies to hundredths of a
//!   cm⁻¹; HF matches the analytic Hessian to the O(δ²) truncation.
//! * If the post-processing is wrong (masses, units, projection, a missing
//!   conversion constant): the no-SCF anchor fails while the elementwise
//!   Hessian checks still pass — the two are designed to separate these.
//! * If the gradient is wrong (a missing term, a wrong grid-response sign, an
//!   open-shell spin factor): the Hessian misses elementwise by far more than
//!   the noise floor, on the systems/methods that exercise that term.
//! * If a displaced SCF lands on a different STATE (open shell): a Hessian
//!   column jumps by orders of magnitude; the centre energy check pins the
//!   centre state, and UHF/UKS run with ferric's stability descent at every
//!   displaced point exactly as the reference follows stability there.
//! * If the HARNESS is broken (units, geometry constant, wrong file, masses
//!   silently different): nuclear repulsion, AO count, the exact mass
//!   equality and the FD-step equality assertions fail first.
//!
//! # TOLERANCES
//!
//! Each bar is 3-10× the measured ferric-vs-reference maximum, recorded on
//! its const. Same-step FD: Hessian 2.4e-7 Ha/Bohr², frequencies 7.5e-4 cm⁻¹.
//!
//! Reference-side floors (PySCF against itself, measured by the generator
//! 2026-09-24 over all 7 files):
//!
//! | quantity | measured max |
//! |---|---:|
//! | FD Hessian asymmetry at δ = 5e-3 (= O(δ²) truncation) | 1.4e-5 Ha/Bohr² |
//! | HF (RHF, UHF): max abs(H_FD − H_analytic) | 2.7e-5 Ha/Bohr² |
//! | HF: FD vs analytic frequencies | 0.10 cm⁻¹ (OH stretch) |
//! | ROHF CH3: H(5e-3) − H(2.5e-3); H(5e-3) − H_Richardson | 1.1e-5; 1.4e-5 Ha/Bohr² |
//! | KS: H_FD(grid response) − H_analytic(no grid response) | 1.3e-3 .. 3.5e-3 Ha/Bohr² |
//! | KS: FD vs analytic frequencies at (75,110) | 5.1 .. 9.4 cm⁻¹ (H2O, NH3); 34.8 cm⁻¹ (CH3 UKS umbrella) |
//! | open shell: ⟨S²⟩ drift over the displaced SCFs | 1.8e-4 (HO2) |
//! | max abs gradient at the reference geometries | 2.8e-3 .. 4.0e-2 Ha/Bohr |
//!
//! The KS gap is the grid-response term, not a reference defect: for H2O
//! PBE/6-31G it is 1.3e-3 Ha/Bohr² (6.5 cm⁻¹) on (75,110), 1.0e-5 (0.07 cm⁻¹,
//! the HF-like truncation) on (99,590) and 7.6e-6 on (150,974). At ferric's
//! default (75,110) grid, KS frequencies carry a grid error of this size in
//! BOTH codes; the degenerate E modes of NH3 and CH3 split by 0.5-22 cm⁻¹ for
//! the same reason. The tight comparison is the same-construction `hessian_fd`.
//!
//! # NEGATIVE CONTROLS (asserted inside the tests)
//!
//! * Mass swap: ferric's own Hessian re-analysed with every H replaced by D
//!   (2.014 u) must move the frequencies away from the reference by
//!   ≥ `MUST_MISS_FACTOR` × the frequency bar — the masses are consumed and
//!   the frequency comparison can fail.
//! * Basis (closed shell): ferric's Hessian in basis A must miss basis B's
//!   `hessian_fd` by ≥ `MUST_MISS_FACTOR` × the Hessian bar.
//! * Method: ferric's Hessian for method X must miss every other method's
//!   `hessian_fd` in the same file by ≥ `MUST_MISS_FACTOR` × the Hessian bar
//!   (a silently ignored `xc` or UHF-vs-ROHF mixup fails here).
//!
//! A missing reference JSON is a HARD failure (panic naming the path).

use std::path::{Path, PathBuf};

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::frequencies::{
    atom_masses, frequencies_from_cartesian_hessian, harmonic_frequencies, FrequencyConfig,
    FrequencyReference, FrequencyResult, HessianMethod, DEFAULT_DELTA,
};
use ferric_scf::rhf::RhfConfig;
use ndarray::Array2;
use serde_json::Value;

const ROW_DIR: &str = "testdata/reference/validation/frequencies";
const MOL_DIR: &str = "testdata/molecules/validation";
const CLOSED_BASES: [&str; 2] = ["6-31g", "cc-pvdz"];
const CLOSED_METHODS: [&str; 3] = ["rhf", "rks_pbe", "rks_b3lyp"];

/// Deuterium mass (u) for the mass-swap negative control.
const DEUTERIUM_MASS: f64 = 2.014;

/// Nuclear repulsion, Ha: geometry/constant like-for-like.
const TOL_ENUC: f64 = 1e-9;
/// Centre-geometry SCF energy vs reference, Ha (pins geometry, basis, state,
/// functional, exact J/K). The KS-energy row measured ≤3.1e-12 for PBE/B3LYP
/// on the same grid.
const TOL_ENERGY: f64 = 1e-8;
/// Post-processing anchor (no SCF): ferric's frequencies from PySCF's
/// (symmetrized) Hessian vs PySCF's harmonic_analysis, cm⁻¹. Measured
/// ≤2.3e-5 before symmetrizing the stored Hessians; the unit constants differ
/// at ~1e-9 relative (~4e-6 cm⁻¹ at 4000 cm⁻¹).
const TOL_FREQ_POSTPROC: f64 = 1e-4;
/// Cartesian Hessian vs the SAME-STEP PySCF FD Hessian, Ha/Bohr², elementwise.
/// The O(δ²) truncation (~1e-5) is identical on both sides and cancels.
/// Measured max 2.4e-7. A failure at ~1e-5 means the two FD constructions
/// differ (different step or a one-sided difference), not a physics defect.
const TOL_HESS_FD: f64 = 2e-6;
/// Frequencies vs the same-step PySCF FD frequencies, cm⁻¹. Measured max
/// 7.5e-4.
const TOL_FREQ_FD: f64 = 5e-3;
/// HF (RHF/UHF) Cartesian Hessian vs PySCF's ANALYTIC Hessian, Ha/Bohr². The
/// floor is ferric's O(δ²) FD truncation at δ = 5e-3: measured max 2.75e-5
/// (OH/UHF), matching PySCF's own FD-vs-analytic gap of 2.7e-5.
const TOL_HESS_ANALYTIC_HF: f64 = 1e-4;
/// HF frequencies vs analytic, cm⁻¹. Measured max 0.10 (OH/UHF).
const TOL_FREQ_ANALYTIC_HF: f64 = 0.5;
/// ROHF vs the Richardson-extrapolated PySCF FD Hessian, Ha/Bohr² (truncation
/// of ferric's 5e-3 step). Measured max 1.4e-5 (CH3).
const TOL_HESS_ROHF_RICHARDSON: f64 = 5e-5;
/// A reference ferric must MISS is missed by at least this multiple of the bar.
const MUST_MISS_FACTOR: f64 = 100.0;

/// SCF convergence for the FD Hessian. `density_conv` is the tight gate.
const SCF_DENSITY_CONV: f64 = 1e-9;
const SCF_ENERGY_CONV: f64 = 1e-10;

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
             scripts/validation/gen_frequencies.py — a missing reference is a failure, never a skip",
            path.display()
        )
    });
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{}: bad JSON: {e}", path.display()))
}

/// Every reference file in the row directory, as (system, basis, json).
fn all_references() -> Vec<(String, String, Value)> {
    let dir = workspace_root().join(ROW_DIR);
    let mut out = Vec::new();
    for entry in std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("missing reference dir {} ({e})", dir.display()))
    {
        let path = entry.unwrap().path();
        if path.extension().and_then(|s| s.to_str()) != Some("json") {
            continue;
        }
        let v: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap())
            .unwrap_or_else(|e| panic!("{}: bad JSON: {e}", path.display()));
        let system = v["system"].as_str().expect("system").to_string();
        let basis_name = v["basis"].as_str().expect("basis").to_string();
        out.push((system, basis_name, v));
    }
    out.sort_by(|a, b| (&a.0, &a.1).cmp(&(&b.0, &b.1)));
    assert!(
        out.len() >= 7,
        "expected >= 7 reference files in {}, found {}",
        dir.display(),
        out.len()
    );
    out
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

fn matrix(v: &Value, ptr: &str, ctx: &str) -> Array2<f64> {
    let rows = v
        .pointer(ptr)
        .and_then(Value::as_array)
        .unwrap_or_else(|| panic!("{ctx}: reference field {ptr} missing or not a matrix"));
    let n = rows.len();
    let mut m = Array2::<f64>::zeros((n, n));
    for (i, row) in rows.iter().enumerate() {
        let row = row.as_array().expect("matrix row");
        assert_eq!(row.len(), n, "{ctx}: {ptr} is not square");
        for (j, x) in row.iter().enumerate() {
            m[(i, j)] = x.as_f64().expect("number");
        }
    }
    m
}

fn load_mol(system: &str, reference: &Value) -> Molecule {
    let charge = reference["charge"].as_i64().expect("charge") as i32;
    let mult = reference["multiplicity"].as_u64().expect("multiplicity") as usize;
    let xyz = workspace_root().join(MOL_DIR).join(format!("{system}.xyz"));
    Molecule::load_xyz_with_charge(xyz.to_str().unwrap(), charge, mult)
        .unwrap_or_else(|e| panic!("{}: {e:?}", xyz.display()))
}

/// Harness like-for-like: geometry, AO count, masses and FD step. Runs before
/// any number is compared, so a harness slip fails as itself.
fn check_harness(ctx: &str, mol: &Molecule, basis_name: &str, r: &Value) {
    let enuc_ref = num(r, "/nuclear_repulsion", ctx);
    let enuc = mol.nuclear_repulsion();
    assert!(
        (enuc - enuc_ref).abs() < TOL_ENUC,
        "{ctx}: nuclear repulsion {enuc:.12} vs reference {enuc_ref:.12} — geometry/unit mismatch"
    );
    let bs = basis::bundled(basis_name).unwrap();
    let prep = PreparedBasis::new(mol, &bs).unwrap();
    let nao = ferric_integrals::oneelectron::overlap(&prep).nrows();
    assert_eq!(
        nao as u64,
        r["nao"].as_u64().expect("nao"),
        "{ctx}: AO count differs from the reference's"
    );
    // Masses: EXACT equality. A different mass table is a ~0.4% frequency
    // shift that would otherwise read as a physics defect.
    let masses = atom_masses(mol).unwrap();
    let ref_masses = vec_f64(r, "/masses_amu", ctx);
    assert_eq!(
        masses, ref_masses,
        "{ctx}: ferric atom_masses differ from the masses the reference used"
    );
}

fn check_hessian(ctx: &str, what: &str, got: &Array2<f64>, want: &Array2<f64>, tol: f64) -> f64 {
    assert_eq!(got.dim(), want.dim(), "{ctx}: {what} Hessian shape");
    let d = got
        .iter()
        .zip(want.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    eprintln!("{ctx}: Hessian vs {what:<11} max|d| {d:.2e} Ha/Bohr^2 (tol {tol:.0e})");
    assert!(
        d < tol,
        "{ctx}: Cartesian Hessian vs {what}: max |d| {d:.3e} >= {tol:.0e} Ha/Bohr^2"
    );
    d
}

fn max_freq_diff(got: &[f64], want: &[f64]) -> f64 {
    assert_eq!(got.len(), want.len(), "frequency count differs");
    let mut g = got.to_vec();
    let mut w = want.to_vec();
    g.sort_by(|a, b| a.partial_cmp(b).unwrap());
    w.sort_by(|a, b| a.partial_cmp(b).unwrap());
    g.iter()
        .zip(w.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max)
}

fn check_freqs(ctx: &str, what: &str, got: &[f64], want: &[f64], tol: f64) -> f64 {
    let d = max_freq_diff(got, want);
    eprintln!(
        "{ctx}: freqs vs {what:<11} max|d| {d:.3e} cm-1 (tol {tol:.0e}) ferric {:?}",
        got.iter().map(|f| format!("{f:.2}")).collect::<Vec<_>>()
    );
    assert!(
        d < tol,
        "{ctx}: frequencies vs {what}: max |d| {d:.4} >= {tol} cm-1\n  ferric {got:?}\n  ref    {want:?}"
    );
    d
}

fn is_ks(method: &str) -> bool {
    method.starts_with("rks_") || method.starts_with("uks_")
}

fn scf_config(method: &str) -> (RhfConfig, FrequencyReference) {
    let (xc, reference) = match method {
        "rhf" => (None, FrequencyReference::Rhf),
        "rks_pbe" => (Some("PBE"), FrequencyReference::Rhf),
        "rks_b3lyp" => (Some("B3LYP"), FrequencyReference::Rhf),
        "uhf" => (None, FrequencyReference::Uhf),
        "uks_pbe" => (Some("PBE"), FrequencyReference::Uhf),
        "rohf" => (None, FrequencyReference::Rohf),
        other => panic!("unknown method {other}"),
    };
    // UHF/UKS: follow instabilities at EVERY displaced SCF, as the reference
    // does. ROHF skips stability by design in ferric (see the UHF/ROHF row).
    let unrestricted = reference == FrequencyReference::Uhf;
    let cfg = RhfConfig {
        xc: xc.map(str::to_string),
        // Explicit EXACT J/K: `None` auto-enables RI-J for closed-shell KS.
        df_j_aux: Some(String::new()),
        df_k_aux: Some(String::new()),
        energy_conv: SCF_ENERGY_CONV,
        density_conv: SCF_DENSITY_CONV,
        max_iter: 500,
        check_stability: unrestricted,
        scf_stability_descent: unrestricted,
        ..Default::default()
    };
    (cfg, reference)
}

/// Run ferric's frequencies for one system × basis × method and check them
/// against the reference block. Returns the ferric result for the cross-file
/// negative controls.
fn check_method(system: &str, basis_name: &str, method: &str) -> FrequencyResult {
    let r = reference(system, basis_name);
    let ctx = format!("{system}/{basis_name}/{method}");
    let mol = load_mol(system, &r);
    check_harness(&ctx, &mol, basis_name, &r);
    let blk = format!("/{method}");
    assert!(
        (num(&r, &format!("{blk}/fd/fd_step_bohr"), &ctx) - DEFAULT_DELTA).abs() < 1e-15,
        "{ctx}: reference FD step differs from ferric's DEFAULT_DELTA — the same-step \
         comparison would no longer cancel the truncation error"
    );

    let (cfg, reference_kind) = scf_config(method);
    // This row validates the finite-difference Hessian (same-step PySCF
    // reference); the analytic RHF Hessian has its own row
    // (validation_rhf_hessian.rs).
    let freq_cfg = FrequencyConfig {
        delta: DEFAULT_DELTA,
        reference: reference_kind,
        hessian: HessianMethod::FiniteDifference,
    };
    let ctxp = ParallelContext::default();
    let res = harmonic_frequencies(
        &ctxp,
        &mol,
        basis_name,
        Operator::coulomb(),
        &cfg,
        &freq_cfg,
    )
    .unwrap_or_else(|e| panic!("{ctx}: harmonic_frequencies failed: {e:?}"));
    eprintln!(
        "{ctx}: E {:.10} asymmetry {:.2e} gradients {}",
        res.energy, res.asymmetry, res.n_gradient_evaluations
    );

    // Centre state/geometry/functional anchor.
    let e_ref = num(&r, &format!("{blk}/energy"), &ctx);
    assert!(
        (res.energy - e_ref).abs() < TOL_ENERGY,
        "{ctx}: centre energy {:.12} vs reference {e_ref:.12} (|d| {:.2e})",
        res.energy,
        (res.energy - e_ref).abs()
    );

    // Tight: same construction, same step.
    let h_fd = matrix(&r, &format!("{blk}/hessian_fd"), &ctx);
    check_hessian(&ctx, "PySCF FD", &res.cartesian_hessian, &h_fd, TOL_HESS_FD);
    let f_fd = vec_f64(&r, &format!("{blk}/freq_fd_cm"), &ctx);
    check_freqs(&ctx, "PySCF FD", &res.frequencies, &f_fd, TOL_FREQ_FD);

    // External analytic second derivative.
    if method == "rohf" {
        let h_rich = matrix(&r, &format!("{blk}/hessian_richardson"), &ctx);
        check_hessian(
            &ctx,
            "Richardson",
            &res.cartesian_hessian,
            &h_rich,
            TOL_HESS_ROHF_RICHARDSON,
        );
    } else {
        let h_an = matrix(&r, &format!("{blk}/hessian_analytic"), &ctx);
        let f_an = vec_f64(&r, &format!("{blk}/freq_analytic_cm"), &ctx);
        let gap = num(
            &r,
            &format!("{blk}/pyscf_fd_vs_analytic_freq_max_abs_cm"),
            &ctx,
        );
        if is_ks(method) {
            // No grid response in PySCF's KS Hessian: ferric may differ from it
            // by PySCF's OWN FD(grid response)-vs-analytic gap (+ the FD bar),
            // never more. That gap is the grid-response term at (75,110)
            // (module doc: it collapses to the HF-like truncation on a finer
            // grid), so this is a consistency bound, not an accuracy claim.
            eprintln!("{ctx}: PySCF's own FD(grid response) vs analytic gap {gap:.3} cm-1");
            check_freqs(
                &ctx,
                "PySCF anal.",
                &res.frequencies,
                &f_an,
                gap + TOL_FREQ_FD,
            );
        } else {
            check_hessian(
                &ctx,
                "PySCF anal.",
                &res.cartesian_hessian,
                &h_an,
                TOL_HESS_ANALYTIC_HF,
            );
            check_freqs(
                &ctx,
                "PySCF anal.",
                &res.frequencies,
                &f_an,
                TOL_FREQ_ANALYTIC_HF,
            );
        }
    }

    // NEGATIVE CONTROL — mass swap H -> D on ferric's own Hessian.
    let masses_d: Vec<f64> = mol
        .atoms
        .iter()
        .zip(atom_masses(&mol).unwrap())
        .map(|(a, m)| if a.z == 1 { DEUTERIUM_MASS } else { m })
        .collect();
    let swapped = frequencies_from_cartesian_hessian(&mol, &res.cartesian_hessian, &masses_d)
        .unwrap_or_else(|e| panic!("{ctx}: mass-swapped analysis failed: {e:?}"));
    let d_swap = max_freq_diff(&swapped.frequencies, &f_fd);
    eprintln!(
        "{ctx}: H->D mass swap moves freqs by {d_swap:.1} cm-1 (must exceed {:.1})",
        MUST_MISS_FACTOR * TOL_FREQ_FD
    );
    assert!(
        d_swap > MUST_MISS_FACTOR * TOL_FREQ_FD,
        "{ctx}: H->D mass swap moved the frequencies by only {d_swap:.3} cm-1 — the \
         frequency comparison does not respond to the masses"
    );

    // NEGATIVE CONTROL — method: must miss every other method's FD Hessian.
    let methods: Vec<String> = r
        .as_object()
        .unwrap()
        .iter()
        .filter(|(_, v)| v.get("hessian_fd").is_some())
        .map(|(k, _)| k.clone())
        .collect();
    for other in methods.iter().filter(|m| m.as_str() != method) {
        let h_other = matrix(&r, &format!("/{other}/hessian_fd"), &ctx);
        let d = res
            .cartesian_hessian
            .iter()
            .zip(h_other.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            d > MUST_MISS_FACTOR * TOL_HESS_FD,
            "{ctx}: ferric Hessian is within {:.0e} of the {other} reference ({d:.2e}) — the \
             comparison does not respond to the method",
            MUST_MISS_FACTOR * TOL_HESS_FD
        );
    }
    res
}

/// Closed shell: run both bases, then require each to MISS the other basis's
/// reference (two-input requirement).
fn check_closed(system: &str, method: &str) {
    let results: Vec<FrequencyResult> = CLOSED_BASES
        .iter()
        .map(|b| check_method(system, b, method))
        .collect();
    for (i, b_self) in CLOSED_BASES.iter().enumerate() {
        let b_other = CLOSED_BASES[1 - i];
        let r_other = reference(system, b_other);
        let h_other = matrix(&r_other, &format!("/{method}/hessian_fd"), system);
        let d = results[i]
            .cartesian_hessian
            .iter()
            .zip(h_other.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            d > MUST_MISS_FACTOR * TOL_HESS_FD,
            "{system}/{method}: ferric {b_self} Hessian is within {:.0e} of the {b_other} \
             reference ({d:.2e}) — the comparison does not respond to the basis",
            MUST_MISS_FACTOR * TOL_HESS_FD
        );
    }
}

/// EXACTNESS / LIKE-FOR-LIKE ANCHOR (no SCF): PySCF's own Hessians pushed
/// through ferric's post-processing must reproduce PySCF's harmonic_analysis
/// frequencies. Proves the masses, unit constants and T/R projection agree,
/// at these non-stationary geometries, independently of any gradient.
#[test]
#[ignore = "validation: Harmonic frequencies"]
fn anchor_postprocessing_matches_pyscf() {
    let mut worst = 0.0_f64;
    let mut n_checked = 0usize;
    for (system, basis_name, r) in all_references() {
        let ctx = format!("{system}/{basis_name}");
        let mol = load_mol(&system, &r);
        check_harness(&ctx, &mol, &basis_name, &r);
        let masses = vec_f64(&r, "/masses_amu", &ctx);
        let methods: Vec<String> = r
            .as_object()
            .unwrap()
            .iter()
            .filter(|(_, v)| v.get("hessian_fd").is_some())
            .map(|(k, _)| k.clone())
            .collect();
        for method in methods {
            let mut pairs = vec![("hessian_fd", "freq_fd_cm")];
            if r[&method]["hessian_analytic"].is_array() {
                pairs.push(("hessian_analytic", "freq_analytic_cm"));
            }
            for (hkey, fkey) in pairs {
                let c = format!("{ctx}/{method}/{hkey}");
                let h = matrix(&r, &format!("/{method}/{hkey}"), &c);
                let want = vec_f64(&r, &format!("/{method}/{fkey}"), &c);
                let got = frequencies_from_cartesian_hessian(&mol, &h, &masses)
                    .unwrap_or_else(|e| panic!("{c}: {e:?}"));
                let expected_nvib = 3 * mol.atoms.len() - if got.is_linear { 5 } else { 6 };
                assert_eq!(got.frequencies.len(), expected_nvib, "{c}: vib count");
                worst = worst.max(check_freqs(
                    &c,
                    "harm.anal.",
                    &got.frequencies,
                    &want,
                    TOL_FREQ_POSTPROC,
                ));
                n_checked += 1;
            }
            // Non-stationarity is real (so the projection claim is exercised).
            let g = max_abs_rows(&r, &format!("/{method}/gradient"));
            eprintln!("{ctx}/{method}: reference max|gradient| {g:.2e} Ha/Bohr");
        }
    }
    eprintln!("anchor: {n_checked} Hessians, worst post-processing |d| {worst:.2e} cm-1");

    // NEGATIVE CONTROL: the same anchor with H -> D masses must fail.
    let (system, basis_name, r) = all_references().into_iter().next().unwrap();
    let mol = load_mol(&system, &r);
    let masses_d: Vec<f64> = mol
        .atoms
        .iter()
        .zip(vec_f64(&r, "/masses_amu", "anchor"))
        .map(|(a, m)| if a.z == 1 { DEUTERIUM_MASS } else { m })
        .collect();
    let method = r
        .as_object()
        .unwrap()
        .iter()
        .find(|(_, v)| v.get("hessian_fd").is_some())
        .map(|(k, _)| k.clone())
        .unwrap();
    let h = matrix(&r, &format!("/{method}/hessian_fd"), "anchor");
    let got = frequencies_from_cartesian_hessian(&mol, &h, &masses_d).unwrap();
    let d = max_freq_diff(
        &got.frequencies,
        &vec_f64(&r, &format!("/{method}/freq_fd_cm"), "anchor"),
    );
    assert!(
        d > MUST_MISS_FACTOR * TOL_FREQ_FD,
        "{system}/{basis_name}/{method}: H->D masses moved the anchor by only {d:.3} cm-1"
    );
}

/// Largest |element| of a (natm × 3) gradient array in the reference.
fn max_abs_rows(r: &Value, ptr: &str) -> f64 {
    r.pointer(ptr)
        .and_then(Value::as_array)
        .map(|rows| {
            rows.iter()
                .flat_map(|row| row.as_array().unwrap().iter())
                .map(|x| x.as_f64().unwrap().abs())
                .fold(0.0_f64, f64::max)
        })
        .unwrap_or(f64::NAN)
}

#[test]
#[ignore = "validation: Harmonic frequencies"]
fn h2o_rhf() {
    check_closed("h2o", "rhf");
}

#[test]
#[ignore = "validation: Harmonic frequencies"]
fn h2o_rks_pbe() {
    check_closed("h2o", "rks_pbe");
}

#[test]
#[ignore = "validation: Harmonic frequencies"]
fn h2o_rks_b3lyp() {
    check_closed("h2o", "rks_b3lyp");
}

#[test]
#[ignore = "validation: Harmonic frequencies"]
fn nh3_rhf() {
    check_closed("nh3", "rhf");
}

#[test]
#[ignore = "validation: Harmonic frequencies"]
fn nh3_rks_pbe() {
    check_closed("nh3", "rks_pbe");
}

#[test]
#[ignore = "validation: Harmonic frequencies"]
fn nh3_rks_b3lyp() {
    check_closed("nh3", "rks_b3lyp");
}

#[test]
#[ignore = "validation: Harmonic frequencies"]
fn oh_uhf() {
    let _ = check_method("oh", "6-31g", "uhf");
}

#[test]
#[ignore = "validation: Harmonic frequencies"]
fn ch3_uhf() {
    let _ = check_method("ch3", "6-31g", "uhf");
}

#[test]
#[ignore = "validation: Harmonic frequencies"]
fn ho2_uhf() {
    let _ = check_method("ho2", "6-31g", "uhf");
}

#[test]
#[ignore = "validation: Harmonic frequencies"]
fn ch3_uks_pbe() {
    let _ = check_method("ch3", "6-31g", "uks_pbe");
}

#[test]
#[ignore = "validation: Harmonic frequencies"]
fn ch3_rohf() {
    let _ = check_method("ch3", "6-31g", "rohf");
}

/// Keeps `CLOSED_METHODS` honest: every closed-shell reference file must carry
/// exactly these method blocks (a generator change that drops one fails here
/// rather than silently shrinking the negative-control set).
#[test]
#[ignore = "validation: Harmonic frequencies"]
fn closed_shell_files_carry_every_method() {
    for system in ["h2o", "nh3"] {
        for b in CLOSED_BASES {
            let r = reference(system, b);
            for m in CLOSED_METHODS {
                assert!(
                    r[m]["hessian_fd"].is_array(),
                    "{system}/{b}: missing {m} block"
                );
            }
        }
    }
}
