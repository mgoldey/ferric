//! VALIDATION tier — VALIDATION.md row "RI-MP2 gradient".
//!
//! Runs only in the weekly `validation` CI job (and on demand):
//!
//! ```text
//! cargo nextest run -p ferric-mp2 --test validation_rimp2_gradient \
//!     --run-ignored only -E 'binary(/^validation_/)'
//! ```
//!
//! # What is compared
//!
//! ferric's analytic RI-MP2 nuclear gradient (`rimp2_gradient_analytical`,
//! closed-shell RHF reference, all electrons correlated) against two
//! independent references, on three systems whose geometries are distorted so
//! that every Cartesian gradient component is nonzero:
//!
//! | system | basis | aux (/C) |
//! |---|---|---|
//! | H2O (off C2v) | cc-pVDZ | cc-pvdz-ri |
//! | HCN (bent, stretched) | cc-pVDZ | cc-pvdz-ri |
//! | NH3 (off C3v) | def2-SVP | def2-svp-rifit |
//!
//! * **ORCA 6.1.1** analytic gradient: `! RI-MP2 NoRI NoFrozenCore ExtremeSCF
//!   EnGrad`, ferric's orbital basis as `NewGTO` and ferric's aux basis as
//!   `NewAuxCGTO`, geometry in Bohr.
//! * **PySCF 2.13.1** 5-point central finite difference (h = 2e-3 Bohr) of the
//!   DF-MP2 total energy (`pyscf.mp.dfmp2.DFMP2`, auxmol from ferric's aux
//!   JSON). PySCF has no analytic DF-MP2 gradient, so this is an independent
//!   METHOD as well as an independent code. The FD is converged in the step:
//!   h = 1e-3 and 4e-3 move it by at most 3.6e-10 Ha/Bohr (H2O, measured).
//!
//! SCF treatment is matched: ferric's `RhfConfig::default()` has no fitting in
//! the SCF (`df_j_aux = df_k_aux = None`), and its z-vector and 2e-derivative
//! blocks are exact four-centre, so both references use exact J/K (ORCA
//! `NoRI`, PySCF plain `scf.RHF`). Only the MP2 correlation is fitted, with the
//! Coulomb metric, on all three sides.
//!
//! References: `scripts/validation/gen_rimp2_gradient.py` →
//! `testdata/reference/validation/rimp2_gradient/<system>_<basis>.json`; ORCA
//! inputs under `scripts/validation/orca/rimp2_gradient/`. The generator refuses
//! to write unless ORCA and PySCF agree with EACH OTHER: RHF energy ≤ 1e-8,
//! RI-MP2 correlation energy ≤ 1e-8 (the check that ORCA used ferric's aux
//! basis), gradient ≤ 1e-6.
//!
//! Reference-vs-reference floor (measured at generation, in each JSON's
//! `cross_check`): ORCA vs PySCF |ΔE_RHF| 0.8–1.3e-10, |ΔE_corr| 3–5e-11,
//! max |Δg| 1.3e-8 (HCN), 1.7e-8 (NH3), 5.0e-8 (H2O). The PySCF FD is stable to
//! 3.6e-10 in the step, so the ≤5e-8 residual is ORCA's analytic-gradient floor;
//! PySCF-FD is therefore the TIGHT gradient reference and ORCA the looser one.
//!
//! # Exactness anchor (asserted first, per system)
//!
//! Everything except the correlation part is checked against a reference with
//! no approximation in common with the MP2 machinery: nuclear repulsion
//! (geometry/units), AO and aux-function counts (basis like-for-like), the RHF
//! energy, and ferric's exact-J/K RHF gradient against PySCF's analytic RHF
//! gradient. If the anchor fails, the MP2 comparison below it means nothing.
//!
//! # Physics hypothesis vs artifact hypothesis (per the Experimental Protocol)
//!
//! * If ferric's RI-MP2 gradient is right: it agrees with the PySCF FD to the
//!   combined SCF/integral floor (expected ~1e-8 Ha/Bohr) and with ORCA to
//!   ORCA's floor (~5e-8), on all three systems and both bases.
//! * If a Lagrangian block is wrong or missing (the defect class the internal
//!   analytic-vs-FD test cannot see when ferric's energy itself is wrong): the
//!   error is a fraction of the correlation gradient, which is
//!   |g_MP2 − g_RHF| = 1.6e-2 .. 1.2e-1 Ha/Bohr here — five or more orders
//!   above the bar.
//! * If the HARNESS is broken: a wrong geometry/constant fails the E_nuc check;
//!   a wrong orbital or aux basis fails the nao/naux counts or the energy
//!   anchors; a silently frozen core (ferric or reference) misses by the
//!   frozen-core gradient shift, 6.9e-4 .. 1.4e-3 Ha/Bohr (measured, JSON
//!   `cross_check.orca_mp2_vs_orca_fc_gradient_max`) — the negative control
//!   below asserts ferric MISSES the frozen-core reference by far more than the
//!   bar, so the bar demonstrably separates the two.
//!
//! # TOLERANCES
//!
//! Each bar is 3–25× the measured ferric-vs-reference floor, recorded on its
//! const. Against the PySCF FD the gradient agrees to 2.1e-9 Ha/Bohr; against
//! ORCA to 5.1e-8, which is ORCA's own floor.
//!
//! # NEGATIVE CONTROLS / MUTATIONS
//!
//! * Always on — FROZEN CORE: ferric's all-electron gradient must miss ORCA's
//!   frozen-core RI-MP2 gradient by > [`MUST_MISS`].
//! * Always on — HF ONLY: ferric's RHF gradient must miss the RI-MP2 reference
//!   by > [`MUST_MISS`], and ferric's RI-MP2 gradient must miss PySCF's RHF
//!   gradient by the same (the correlation contribution is what is tested).
//! * MUTATION (manual, on the branch the test reaches): drop the RI 3c/2c
//!   integral-response term (`grad += &integral_response_gradient_3c2c(..)` in
//!   `rimp2_gradient_analytical`). The energies still pass; the gradient
//!   assertion must fail on every system.
//!
//! A missing reference JSON is a HARD failure (panic naming the path).

use std::path::{Path, PathBuf};

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::gradient::rimp2_gradient_analytical;
use ferric_mp2::rimp2::{ri_mp2, RiMp2Config};
use ferric_scf::gradient::rhf_gradient_exact_jk;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ndarray::Array2;
use serde_json::Value;

const ROW_DIR: &str = "testdata/reference/validation/rimp2_gradient";
const MOL_DIR: &str = "testdata/molecules/validation";

/// Geometry check. The reference E_nuc is PySCF's, from ferric's geometry in
/// Bohr with ferric's constant, so only arithmetic separates them.
const TOL_ENUC: f64 = 1e-9;
/// RHF energy vs PySCF and ORCA (exact J/K on all three sides). Measured
/// max 3.5e-12 (PySCF) and 1.3e-10 (ORCA, whose ExtremeSCF floor dominates).
const TOL_E_RHF: f64 = 1e-9;
/// RI-MP2 correlation energy vs PySCF and ORCA. Measured max 6.0e-12 (PySCF)
/// and 5.0e-11 (ORCA).
const TOL_E_CORR: f64 = 1e-9;
/// ferric RHF gradient vs PySCF analytic RHF gradient (anchor). Measured max
/// 9.6e-10.
const TOL_G_RHF: f64 = 1e-8;
/// RI-MP2 gradient vs the PySCF central FD of the DF-MP2 energy (step-stable
/// to 3.6e-10). Measured max 2.1e-9 (HCN).
const TOL_G_FD: f64 = 2e-8;
/// RI-MP2 gradient vs ORCA analytic. Measured max 5.1e-8 (H2O), which is
/// ORCA's own distance from the FD reference (5.0e-8), not ferric's.
const TOL_G_ORCA: f64 = 2.5e-7;
/// A reference that ferric's gradient must MISS (negative controls): 40x the
/// loosest gradient bar, and still 70x below the smallest measured frozen-core
/// shift (6.9e-4).
const MUST_MISS: f64 = 1e-5;

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
        .expect("ferric-mp2 manifest dir should be <root>/crates/ferric-mp2")
        .to_path_buf()
}

fn reference(system: &str, basis_name: &str) -> Value {
    let path = workspace_root()
        .join(ROW_DIR)
        .join(format!("{system}_{basis_name}.json"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "missing validation reference {} ({e}); regenerate with \
             scripts/validation/gen_rimp2_gradient.py — a missing reference is a failure, never a skip",
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

fn grad(v: &Value, ptr: &str, natm: usize, ctx: &str) -> Array2<f64> {
    let rows = v
        .pointer(ptr)
        .and_then(Value::as_array)
        .unwrap_or_else(|| panic!("{ctx}: reference field {ptr} missing or not an array"));
    assert_eq!(rows.len(), natm, "{ctx}: {ptr} has {} rows", rows.len());
    let mut g = Array2::zeros((natm, 3));
    for (a, row) in rows.iter().enumerate() {
        for k in 0..3 {
            g[(a, k)] = row[k]
                .as_f64()
                .unwrap_or_else(|| panic!("{ctx}: {ptr}[{a}][{k}] not a number"));
        }
    }
    g
}

fn max_abs_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    assert_eq!(a.dim(), b.dim());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

fn check_close(ctx: &str, what: &str, got: f64, want: f64, tol: f64) {
    let d = (got - want).abs();
    eprintln!("{ctx}: {what:<22} ferric {got:+.12} ref {want:+.12} |d| {d:.2e} (tol {tol:.0e})");
    assert!(
        d < tol,
        "{ctx}: {what} {got:.12} vs reference {want:.12} (|d| {d:.2e} >= {tol:.0e})"
    );
}

fn check_grad(ctx: &str, what: &str, got: &Array2<f64>, want: &Array2<f64>, tol: f64) {
    let d = max_abs_diff(got, want);
    eprintln!("{ctx}: {what:<22} max|d| {d:.2e} (tol {tol:.0e})");
    assert!(
        d < tol,
        "{ctx}: {what} max |d| {d:.2e} >= {tol:.0e}\n ferric {got:?}\n ref {want:?}"
    );
}

fn check_miss(ctx: &str, what: &str, got: &Array2<f64>, want: &Array2<f64>) {
    let d = max_abs_diff(got, want);
    eprintln!("{ctx}: {what:<22} max|d| {d:.2e} (must exceed {MUST_MISS:.0e})");
    assert!(
        d > MUST_MISS,
        "{ctx}: negative control {what}: max |d| {d:.2e} <= {MUST_MISS:.0e} — the gradient \
         bar cannot tell these apart"
    );
}

fn check_system(system: &str, basis_name: &str, aux_name: &str) {
    let r = reference(system, basis_name);
    let ctx = format!("{system}/{basis_name}");
    assert_eq!(r["aux_basis"].as_str(), Some(aux_name), "{ctx}: aux basis");
    assert_eq!(r["basis"].as_str(), Some(basis_name), "{ctx}: basis");

    // ---- exactness anchor: geometry, basis, RHF ----
    let xyz = workspace_root().join(MOL_DIR).join(format!("{system}.xyz"));
    let mol = Molecule::load_xyz(xyz.to_str().unwrap())
        .unwrap_or_else(|e| panic!("{}: {e:?}", xyz.display()));
    let natm = mol.atoms.len();
    check_close(
        &ctx,
        "E_nuc",
        mol.nuclear_repulsion(),
        num(&r, "/nuclear_repulsion", &ctx),
        TOL_ENUC,
    );

    let obs_bs = basis::bundled(basis_name).unwrap();
    let aux_bs = basis::bundled(aux_name).unwrap();
    let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
    let dfbs = PreparedBasis::new(&mol, &aux_bs).unwrap();
    assert_eq!(
        obs.nbasis() as u64,
        r["nao"].as_u64().unwrap(),
        "{ctx}: AO count"
    );
    assert_eq!(
        dfbs.nbasis() as u64,
        r["naux"].as_u64().unwrap(),
        "{ctx}: aux count"
    );

    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    // Exact J/K (no df_j_aux/df_k_aux), as in `total_rimp2_gradient`. Tight ΔP:
    // the MP2 part is not variational in the orbitals.
    let cfg = RhfConfig {
        density_conv: 1e-9,
        max_iter: 200,
        ..Default::default()
    };
    assert!(cfg.df_j_aux.is_none() && cfg.df_k_aux.is_none());
    let rhf = solve_rhf(&ParallelContext::default(), &mol, &obs, op, &bounds, &cfg)
        .unwrap_or_else(|e| panic!("{ctx}: solve_rhf failed: {e:?}"));
    assert!(rhf.converged, "{ctx}: RHF not converged");
    check_close(
        &ctx,
        "E_RHF vs PySCF",
        rhf.energy,
        num(&r, "/pyscf/e_rhf", &ctx),
        TOL_E_RHF,
    );
    check_close(
        &ctx,
        "E_RHF vs ORCA",
        rhf.energy,
        num(&r, "/orca/e_rhf", &ctx),
        TOL_E_RHF,
    );
    let g_rhf = rhf_gradient_exact_jk(&mol, &obs, op, &bounds, &rhf, None).unwrap();
    let g_rhf_ref = grad(&r, "/pyscf/rhf_gradient", natm, &ctx);
    check_grad(&ctx, "RHF grad vs PySCF", &g_rhf, &g_rhf_ref, TOL_G_RHF);

    // ---- RI-MP2 energy ----
    let mp2_cfg = RiMp2Config::default();
    assert_eq!(mp2_cfg.frozen_core, 0, "{ctx}: all-electron is the default");
    let mp2 = ri_mp2(&mol, &obs, &dfbs, op, &rhf, &mp2_cfg).unwrap();
    check_close(
        &ctx,
        "E_corr vs PySCF",
        mp2.mp2_corr,
        num(&r, "/pyscf/e_corr", &ctx),
        TOL_E_CORR,
    );
    check_close(
        &ctx,
        "E_corr vs ORCA",
        mp2.mp2_corr,
        num(&r, "/orca/e_corr", &ctx),
        TOL_E_CORR,
    );
    check_close(
        &ctx,
        "E_total vs ORCA",
        mp2.total_energy,
        num(&r, "/orca/e_total", &ctx),
        TOL_E_RHF + TOL_E_CORR,
    );

    // ---- RI-MP2 gradient ----
    let g = rimp2_gradient_analytical(&mol, &obs, &dfbs, op, &bounds, &rhf, &mp2_cfg, None)
        .unwrap_or_else(|e| panic!("{ctx}: rimp2_gradient_analytical failed: {e:?}"));
    let g_fd = grad(&r, "/pyscf/gradient_fd", natm, &ctx);
    let g_orca = grad(&r, "/orca/gradient", natm, &ctx);
    check_grad(&ctx, "MP2 grad vs PySCF FD", &g, &g_fd, TOL_G_FD);
    check_grad(&ctx, "MP2 grad vs ORCA", &g, &g_orca, TOL_G_ORCA);

    // ---- negative controls ----
    let g_fc = grad(&r, "/orca_frozen_core/gradient", natm, &ctx);
    check_miss(&ctx, "MP2 vs ORCA frozen-core", &g, &g_fc);
    check_miss(&ctx, "MP2 vs PySCF RHF grad", &g, &g_rhf_ref);
    check_miss(&ctx, "ferric RHF vs MP2 ref", &g_rhf, &g_fd);
}

#[test]
#[ignore = "validation: RI-MP2 gradient"]
fn rimp2_gradient_h2o_ccpvdz_vs_orca_and_pyscf() {
    check_system("h2o_distorted", "cc-pvdz", "cc-pvdz-ri");
}

#[test]
#[ignore = "validation: RI-MP2 gradient"]
fn rimp2_gradient_hcn_ccpvdz_vs_orca_and_pyscf() {
    check_system("hcn_bent", "cc-pvdz", "cc-pvdz-ri");
}

#[test]
#[ignore = "validation: RI-MP2 gradient"]
fn rimp2_gradient_nh3_def2svp_vs_orca_and_pyscf() {
    check_system("nh3_distorted", "def2-svp", "def2-svp-rifit");
}
