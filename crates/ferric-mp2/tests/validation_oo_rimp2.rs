//! VALIDATION tier — VALIDATION.md rows "OO-RI-MP2 energy" and "OO-MP2 orbital
//! and nuclear gradients".
//!
//! Runs only in the weekly `validation` CI job (and on demand):
//!
//! ```text
//! cargo nextest run -p ferric-mp2 --test validation_oo_rimp2 \
//!     --run-ignored only -E 'binary(/^validation_/)'
//! ```
//!
//! # What is compared
//!
//! ferric's orbital-optimized RI-MP2 (`oo_ri_mp2`, closed shell; `u_oo_ri_mp2`,
//! UHF-based open shell) and its analytic nuclear gradient
//! (`oo_ri_mp2_gradient`, closed shell only — ferric has no U-OO nuclear
//! gradient) against an independent numpy OO-RI-MP2 (energies) and ORCA 6.1.1
//! (cross-check), all electrons correlated:
//!
//! | system | reference | basis | aux (/C) | checked |
//! |---|---|---|---|---|
//! | H2O (off C2v) | RHF | cc-pVDZ | cc-pvdz-ri | energy, nuclear gradient |
//! | NH3 (off C3v) | RHF | cc-pVDZ | cc-pvdz-ri | energy, nuclear gradient |
//! | CH3 (D3h, doublet) | UHF | cc-pVDZ | cc-pvdz-ri | energy |
//!
//! ferric's functional (read from `oo_rimp2.rs`): `E_HF(C) + E_MP2^RI(C)` with
//! the MP2 part in the semicanonical frame of `C`, exact four-centre J/K in
//! `E_HF` and the Fock matrix, RI (Coulomb metric) in the correlation part
//! only.
//!
//! The TIGHT energy reference is the JSON `numpy_oo` block: the same
//! functional written directly in numpy on PySCF integrals (exact J/K from
//! `int2e`, Coulomb-metric fitted B from `int3c2e`/`int2c2e` with ferric's
//! aux), RHF or UHF (independent α/β rotations) as appropriate, minimised over
//! vir–occ rotations `C = C_SCF exp(K)` with a finite-difference orbital
//! gradient to max |dE/dκ| ≤ 1e-10 (measured 5.7e-11 to 8.8e-11, stored as
//! `max_orbital_gradient`), so its energy is converged far below 1e-12 (the
//! error is second order in the gradient).
//! It shares no code with ferric or ORCA and has no analytic orbital
//! derivative. Its anchor is asserted in the generator: at κ = 0 it equals
//! PySCF SCF + DF-MP2 (`DFMP2`/`DFUMP2`, same auxmol) to 1e-12 (RHF) / 1e-10
//! (UHF), recorded as `anchor_kappa0_diff`. The reference/doubles split is
//! first order in the orbitals, so it is compared at a looser bar than the
//! total.
//!
//! The ORCA CROSS-CHECK is ORCA `! OO-RI-MP2 NoRI NoFrozenCore
//! ExtremeSCF [UHF] EnGrad` with ferric's orbital basis as `NewGTO` and ferric's
//! aux basis as `NewAuxCGTO`: `NoRI` makes ORCA's per-iteration Fock builds
//! exact, and ORCA also semicanonicalises every orbital iteration. ORCA's
//! `FINAL SINGLE POINT ENERGY` is `Total Energy(D)` (reference + doubles), the
//! same functional; its perturbative-singles `Total Energy(SD)` is not, and is
//! not compared. ORCA converges (reported ||g|| ~3e-10) to a point ABOVE the
//! minimum of this functional: its total is higher than the numpy minimum by
//! 6.6e-8 (H2O), 7.0e-8 (NH3) and 3.7e-8 Ha (CH3), and its reference/doubles
//! split differs by ~1.5e-5 (CH3 8.9e-6). ferric and the numpy construction
//! agree with each other, below ORCA. The ORCA energies are therefore compared
//! at loose bars covering that offset, and the sign of the offset is asserted
//! (`numpy_oo.e_total < orca_oo.e_total`) so a change in ORCA's behaviour is
//! noticed.
//!
//! The TIGHT gradient anchor is ferric's analytic OO gradient against a 5-point
//! central finite difference (h = 2e-3 Bohr) of ferric's own re-converged OO
//! energy (`*_self_fd` tests below); it needs no external code. The external
//! gradient references are loose cross-checks: the 5-point FD of ORCA's OO
//! energy (step-converged: H2O H1-y at h = 1e-3, 2e-3, 4e-3 agree to 7.5e-10
//! Ha/Bohr) inherits ORCA's non-stationary orbitals, an error first order in
//! the orbital offset, so it is compared at [`TOL_G_FD`]; ORCA's ANALYTIC
//! OO-RI-MP2 gradient misses that FD by up to 8.0e-6 Ha/Bohr on distorted H2O,
//! so it is compared only at [`TOL_G_ORCA_ANALYTIC`]. There is no PySCF OMP2
//! and no numpy nuclear gradient.
//!
//! References: `scripts/validation/gen_oo_rimp2.py` →
//! `testdata/reference/validation/oo_rimp2/<system>_<basis>.json`; ORCA inputs
//! under `scripts/validation/orca/oo_rimp2/`.
//!
//! # Exactness anchor (asserted first, per system)
//!
//! Everything below the orbital optimization is checked against references
//! that share no OO machinery with it: nuclear repulsion (geometry/units), AO
//! and aux-function counts, the SCF energy (ORCA and PySCF, exact J/K; the
//! PySCF UHF is stability-checked) and the plain (non-OO) RI-MP2 total energy
//! (ORCA `RI-MP2 NoRI`, and PySCF DF-MP2 for the closed shells). A wrong
//! orbital or aux basis fails here, before any OO number is read. At zero
//! orbital rotation OO-RI-MP2 IS plain RI-MP2, so this also anchors the
//! functional's starting point.
//!
//! # Physics hypothesis vs artifact hypothesis (per the Experimental Protocol)
//!
//! * If ferric's OO-RI-MP2 is the textbook functional and its stationary point
//!   is found: the total energy matches the numpy minimum to the integral floor
//!   (expected ≲1e-10 Ha), ORCA to within its measured offset (≤7e-8 Ha), and
//!   the analytic nuclear gradient matches ferric's own FD to the FD floor
//!   (~1e-8 Ha/Bohr).
//! * If ferric minimises a DIFFERENT functional (e.g. denominators read off
//!   diag(CᵀFC) in a non-semicanonical frame): the energy misses by 3e-5 to
//!   1e-4 Ha (the measured diagonal-Fock vs textbook gap for H2O and NH3
//!   cc-pVDZ), four orders above the bar.
//! * If the solver stops short of stationarity: the energy error is quadratic
//!   in the residual rotation and small, but the envelope-theorem gradient
//!   error and the reference/doubles split are linear in it — the self-FD
//!   gradient and the split vs numpy catch this even if the total does not.
//! * If the HARNESS is broken: geometry fails E_nuc, a basis mismatch fails the
//!   nao/naux counts and the SCF/RI-MP2 anchors, a frozen core fails the
//!   RI-MP2 anchor (ORCA's printed E_corr is all-electron).
//!
//! # TOLERANCES
//!
//! Each bar is 3–25x the worst |d| measured over H2O, NH3 and CH3
//! (2026-09-25):
//!
//! | quantity | measured max \|d\| | bar |
//! |---|---:|---:|
//! | OO total vs numpy | 7.5e-13 Ha | [`TOL_E_OO_NUMPY`] 1e-11 |
//! | OO reference/doubles split vs numpy | 6.4e-10 Ha (CH3) | [`TOL_E_OO_SPLIT_NUMPY`] 5e-9 |
//! | OO total vs ORCA | 7.0e-8 Ha (ORCA above the minimum) | [`TOL_E_OO_ORCA`] 2e-7 |
//! | OO split vs ORCA | 1.54e-5 Ha | [`TOL_E_OO_SPLIT_ORCA`] 5e-5 |
//! | SCF vs ORCA / PySCF | 1.05e-10 Ha | [`TOL_E_SCF`] 1e-9 |
//! | RI-MP2 vs ORCA / PySCF | 6.3e-11 Ha | [`TOL_E_RIMP2`] 1e-9 |
//! | gradient vs own FD | 8.0e-9 Ha/Bohr | [`TOL_G_SELF_FD`] 1e-7 |
//! | gradient vs FD of ORCA's energy | 2.0e-7 Ha/Bohr | [`TOL_G_FD`] 1e-6 |
//! | gradient vs ORCA analytic | 7.9e-6 Ha/Bohr | [`TOL_G_ORCA_ANALYTIC`] 3e-5 |
//!
//! # NEGATIVE CONTROLS
//!
//! * Always on — NOT ORBITAL-OPTIMIZED: ferric's plain RI-MP2 total energy must
//!   miss the numpy OO reference by > [`MUST_MISS_E`], and ORCA's own
//!   OO-minus-RI-MP2 gap must too (so the bar is reachable from both sides).
//! * Always on — closed shell: ferric's plain RI-MP2 analytic gradient must
//!   miss ORCA's OO gradient by > [`MUST_MISS_G`].
//! * MUTATION (manual, on the branch the test reaches): in
//!   `oo_rimp2_gradient.rs` drop the `vhf_s1occ` term from the energy-weighted
//!   density. The energies still pass; the self-FD check must fail.
//!
//! A missing reference JSON is a HARD failure (panic naming the path).

use std::path::{Path, PathBuf};

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::gradient::rimp2_gradient_analytical;
use ferric_mp2::oo_rimp2::{oo_ri_mp2, OoRiMp2Config, OoRiMp2Result};
use ferric_mp2::oo_rimp2_gradient::oo_ri_mp2_gradient;
use ferric_mp2::rimp2::{ri_mp2, RiMp2Config};
use ferric_mp2::u_oo_rimp2::{u_oo_ri_mp2, UOoRiMp2Config};
use ferric_mp2::u_rimp2::u_ri_mp2;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::solve_uhf;
use ferric_scf::{ScfResult, Spin};
use ndarray::Array2;
use serde_json::Value;

const ROW_DIR: &str = "testdata/reference/validation/oo_rimp2";
const MOL_DIR: &str = "testdata/molecules/validation";
const BASIS: &str = "cc-pvdz";
const AUX: &str = "cc-pvdz-ri";

/// Geometry check (reference E_nuc is PySCF's, from ferric's Bohr geometry).
const TOL_ENUC: f64 = 1e-9;
/// SCF energy vs ORCA and PySCF (exact J/K on all sides): 1.05e-10 Ha.
const TOL_E_SCF: f64 = 1e-9;
/// Plain RI-MP2 total energy vs ORCA `RI-MP2 NoRI` and PySCF DF-MP2: 6.3e-11 Ha.
const TOL_E_RIMP2: f64 = 1e-9;
/// OO-RI-MP2 TOTAL energy vs the numpy OO-RI-MP2 minimum (the tight reference;
/// its max |dE/dκ| ≤ 8.8e-11): 7.5e-13 Ha.
const TOL_E_OO_NUMPY: f64 = 1e-11;
/// OO reference/doubles split vs numpy: 6.4e-10 Ha (CH3). The split is first
/// order in the orbital difference, so it reflects ferric's grad_conv 1e-8.
const TOL_E_OO_SPLIT_NUMPY: f64 = 5e-9;
/// Max |dE/dκ| the numpy reference must have reached (generator bar).
const NUMPY_MAX_ORBITAL_GRAD: f64 = 1e-8;
/// OO-RI-MP2 TOTAL energy vs ORCA: a loose cross-check. ORCA stops ABOVE the
/// minimum of the functional: ORCA − numpy = +6.6e-8 (H2O), +7.0e-8 (NH3),
/// +3.7e-8 Ha (CH3), with ferric equal to numpy (see module doc).
const TOL_E_OO_ORCA: f64 = 2e-7;
/// OO reference/doubles split vs ORCA's 9-decimal prints. ORCA's split differs
/// from the numpy minimum by 1.54e-5 (H2O), 1.52e-5 (NH3) and 8.9e-6 Ha (CH3),
/// first order in ORCA's orbital offset.
const TOL_E_OO_SPLIT_ORCA: f64 = 5e-5;
/// OO nuclear gradient vs the 5-point FD of ORCA's OO energy: 2.0e-7 Ha/Bohr.
/// That FD is taken on ORCA's non-minimal energy surface; the tight gradient
/// anchor is [`TOL_G_SELF_FD`].
const TOL_G_FD: f64 = 1e-6;
/// OO nuclear gradient vs ORCA ANALYTIC: 7.9e-6 Ha/Bohr. ORCA's analytic OO
/// gradient misses the FD of ORCA's own energy by up to 8.0e-6 (JSON
/// `cross_check.orca_oo_analytic_vs_fd_gradient_max`), so ORCA sets this bar.
const TOL_G_ORCA_ANALYTIC: f64 = 3e-5;
/// ferric analytic OO gradient vs 5-point FD of ferric's own re-converged OO
/// energy (h = 2e-3 Bohr): 8.0e-9 Ha/Bohr.
const TOL_G_SELF_FD: f64 = 1e-7;
const H_FD: f64 = 2e-3;
/// Energy that a non-OO result must MISS the OO reference by. The measured
/// OO-minus-RI-MP2 gaps are 9.2e-4 (H2O), 8.7e-4 (NH3) and 6.1e-4 Ha (CH3).
const MUST_MISS_E: f64 = 1e-6;
/// Gradient that plain RI-MP2 must MISS the OO gradient by: measured
/// |g_RIMP2 − g_OO| is 1.48e-3 (H2O) and 8.1e-4 Ha/Bohr (NH3); 10x TOL_G_FD.
const MUST_MISS_G: f64 = 1e-5;

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

fn reference(system: &str) -> Value {
    let path = workspace_root()
        .join(ROW_DIR)
        .join(format!("{system}_{BASIS}.json"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "missing validation reference {} ({e}); regenerate with \
             scripts/validation/gen_oo_rimp2.py — a missing reference is a failure, never a skip",
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
    eprintln!("{ctx}: {what:<26} ferric {got:+.12} ref {want:+.12} |d| {d:.2e} (tol {tol:.0e})");
    assert!(
        d < tol,
        "{ctx}: {what} {got:.12} vs reference {want:.12} (|d| {d:.2e} >= {tol:.0e})"
    );
}

fn check_miss_e(ctx: &str, what: &str, got: f64, want: f64) {
    let d = (got - want).abs();
    eprintln!("{ctx}: {what:<26} |d| {d:.2e} (must exceed {MUST_MISS_E:.0e})");
    assert!(
        d > MUST_MISS_E,
        "{ctx}: negative control {what}: |d| {d:.2e} <= {MUST_MISS_E:.0e} — the energy bar \
         cannot tell OO from non-OO"
    );
}

fn check_grad(ctx: &str, what: &str, got: &Array2<f64>, want: &Array2<f64>, tol: f64) {
    let d = max_abs_diff(got, want);
    eprintln!("{ctx}: {what:<26} max|d| {d:.2e} (tol {tol:.0e})");
    assert!(
        d < tol,
        "{ctx}: {what} max |d| {d:.2e} >= {tol:.0e}\n ferric {got:?}\n ref {want:?}"
    );
}

/// The OO energy checks shared by the closed- and open-shell tests: split then
/// total, first vs the numpy minimum (tight), then vs ORCA (loose). Returns the
/// numpy total, the OO reference for the negative controls.
fn check_oo_energies(ctx: &str, r: &Value, label: &str, e_ref: f64, e_dbl: f64, e_tot: f64) -> f64 {
    // The numpy reference itself: converged and anchored (generator-asserted,
    // re-checked so a hand-edited JSON cannot slip through).
    let g_np = num(r, "/numpy_oo/max_orbital_gradient", ctx);
    assert!(
        g_np <= NUMPY_MAX_ORBITAL_GRAD,
        "{ctx}: numpy OO reference max |dE/dkappa| {g_np:.2e} > {NUMPY_MAX_ORBITAL_GRAD:.0e}"
    );
    let anchor = num(r, "/numpy_oo/anchor_kappa0_diff", ctx);
    assert!(
        anchor.abs() <= 1e-10,
        "{ctx}: numpy OO kappa=0 anchor vs PySCF SCF+DF-MP2 {anchor:.2e}"
    );
    let np_tot = num(r, "/numpy_oo/e_total", ctx);
    let orca_tot = num(r, "/orca_oo/e_total", ctx);
    // ORCA stops ABOVE the minimum of the functional (module doc). If this
    // flips, ORCA's OO solver changed and the loose ORCA bars must be revisited.
    let offset = orca_tot - np_tot;
    eprintln!("{ctx}: ORCA OO - numpy OO total {offset:+.2e} Ha (expected +3.7e-8..+7.0e-8)");
    assert!(
        offset > 0.0 && offset < TOL_E_OO_ORCA,
        "{ctx}: ORCA - numpy OO total {offset:+.2e}: expected ORCA above the numpy minimum \
         by < {TOL_E_OO_ORCA:.0e}"
    );

    check_close(
        ctx,
        &format!("{label} reference vs numpy"),
        e_ref,
        num(r, "/numpy_oo/e_reference", ctx),
        TOL_E_OO_SPLIT_NUMPY,
    );
    check_close(
        ctx,
        &format!("{label} doubles vs numpy"),
        e_dbl,
        num(r, "/numpy_oo/e_doubles", ctx),
        TOL_E_OO_SPLIT_NUMPY,
    );
    check_close(
        ctx,
        &format!("{label} reference vs ORCA"),
        e_ref,
        num(r, "/orca_oo/e_reference", ctx),
        TOL_E_OO_SPLIT_ORCA,
    );
    check_close(
        ctx,
        &format!("{label} doubles vs ORCA"),
        e_dbl,
        num(r, "/orca_oo/e_doubles", ctx),
        TOL_E_OO_SPLIT_ORCA,
    );
    check_close(
        ctx,
        &format!("{label} total vs numpy"),
        e_tot,
        np_tot,
        TOL_E_OO_NUMPY,
    );
    check_close(
        ctx,
        &format!("{label} total vs ORCA"),
        e_tot,
        orca_tot,
        TOL_E_OO_ORCA,
    );
    np_tot
}

struct Sys {
    mol: Molecule,
    obs: PreparedBasis,
    dfbs: PreparedBasis,
    bounds: SchwarzBounds,
    op: Operator,
}

impl Sys {
    fn new(mol: Molecule) -> Self {
        let obs = PreparedBasis::new(&mol, &basis::bundled(BASIS).unwrap()).unwrap();
        let dfbs = PreparedBasis::new(&mol, &basis::bundled(AUX).unwrap()).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        Self {
            mol,
            obs,
            dfbs,
            bounds,
            op,
        }
    }
}

/// Load the geometry and check the harness-level anchors (E_nuc, nao, naux).
fn load(system: &str, r: &Value) -> Sys {
    let ctx = format!("{system}/{BASIS}");
    assert_eq!(r["basis"].as_str(), Some(BASIS), "{ctx}: basis");
    assert_eq!(r["aux_basis"].as_str(), Some(AUX), "{ctx}: aux basis");
    let charge = r["charge"].as_i64().expect("charge") as i32;
    let mult = r["multiplicity"].as_u64().expect("multiplicity") as usize;
    let xyz = workspace_root().join(MOL_DIR).join(format!("{system}.xyz"));
    let mol = Molecule::load_xyz_with_charge(xyz.to_str().unwrap(), charge, mult)
        .unwrap_or_else(|e| panic!("{}: {e:?}", xyz.display()));
    check_close(
        &ctx,
        "E_nuc",
        mol.nuclear_repulsion(),
        num(r, "/nuclear_repulsion", &ctx),
        TOL_ENUC,
    );
    let s = Sys::new(mol);
    assert_eq!(
        s.obs.nbasis() as u64,
        r["nao"].as_u64().unwrap(),
        "{ctx}: AO count"
    );
    assert_eq!(
        s.dfbs.nbasis() as u64,
        r["naux"].as_u64().unwrap(),
        "{ctx}: aux count"
    );
    s
}

fn rhf_config() -> RhfConfig {
    let cfg = RhfConfig {
        energy_conv: 1e-11,
        density_conv: 1e-9,
        max_iter: 200,
        ..Default::default()
    };
    assert!(
        cfg.df_j_aux.is_none() && cfg.df_k_aux.is_none(),
        "exact J/K"
    );
    cfg
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

fn oo_config() -> OoRiMp2Config {
    let cfg = OoRiMp2Config {
        grad_conv: 1e-8,
        energy_conv: 1e-11,
        max_iter: 200,
        ..Default::default()
    };
    assert_eq!(
        cfg.frozen_core, 0,
        "all electrons correlated (ORCA NoFrozenCore)"
    );
    cfg
}

fn solve_closed(s: &Sys, ctx: &str) -> ScfResult {
    let rhf = solve_rhf(
        &ParallelContext::default(),
        &s.mol,
        &s.obs,
        s.op,
        &s.bounds,
        &rhf_config(),
    )
    .unwrap_or_else(|e| panic!("{ctx}: solve_rhf failed: {e:?}"));
    assert!(rhf.converged, "{ctx}: RHF not converged");
    rhf
}

fn solve_oo(s: &Sys, rhf: &ScfResult, ctx: &str) -> OoRiMp2Result {
    let oo = oo_ri_mp2(
        &s.mol,
        &s.obs,
        &s.dfbs,
        s.op,
        &s.bounds,
        rhf,
        &oo_config(),
        None,
    )
    .unwrap_or_else(|e| panic!("{ctx}: oo_ri_mp2 failed: {e:?}"));
    assert!(
        oo.converged,
        "{ctx}: OO-RI-MP2 not converged (|g| {:.2e})",
        oo.grad_norm
    );
    oo
}

/// Energy anchors + OO energy + OO nuclear gradient for a closed shell.
fn check_closed(system: &str) {
    let r = reference(system);
    let ctx = format!("{system}/{BASIS}");
    let s = load(system, &r);
    let natm = s.mol.atoms.len();

    // ---- exactness anchor: SCF and plain RI-MP2 ----
    let rhf = solve_closed(&s, &ctx);
    check_close(
        &ctx,
        "E_RHF vs ORCA",
        rhf.energy,
        num(&r, "/orca_rimp2/e_scf", &ctx),
        TOL_E_SCF,
    );
    check_close(
        &ctx,
        "E_RHF vs PySCF",
        rhf.energy,
        num(&r, "/pyscf/e_scf", &ctx),
        TOL_E_SCF,
    );
    let mp2_cfg = RiMp2Config::default();
    assert_eq!(mp2_cfg.frozen_core, 0);
    let mp2 = ri_mp2(&s.mol, &s.obs, &s.dfbs, s.op, &rhf, &mp2_cfg).unwrap();
    check_close(
        &ctx,
        "E_RIMP2 vs ORCA",
        mp2.total_energy,
        num(&r, "/orca_rimp2/e_total", &ctx),
        TOL_E_RIMP2,
    );
    check_close(
        &ctx,
        "E_corr(RI) vs PySCF DF-MP2",
        mp2.mp2_corr,
        num(&r, "/pyscf/e_corr_dfmp2", &ctx),
        TOL_E_RIMP2,
    );

    // ---- OO-RI-MP2 energy ----
    let oo = solve_oo(&s, &rhf, &ctx);
    let e_oo_ref = check_oo_energies(&ctx, &r, "E_OO", oo.hf_energy, oo.mp2_corr, oo.total_energy);

    // ---- negative control: non-OO misses the OO reference ----
    check_miss_e(
        &ctx,
        "ferric RI-MP2 vs numpy OO",
        mp2.total_energy,
        e_oo_ref,
    );
    check_miss_e(
        &ctx,
        "ORCA RI-MP2 vs ORCA OO",
        num(&r, "/orca_rimp2/e_total", &ctx),
        num(&r, "/orca_oo/e_total", &ctx),
    );

    // ---- OO nuclear gradient ----
    let g = oo_ri_mp2_gradient(&s.mol, &s.obs, &s.dfbs, s.op, &s.bounds, &oo, 0, None)
        .unwrap_or_else(|e| panic!("{ctx}: oo_ri_mp2_gradient failed: {e:?}"));
    let g_fd = grad(&r, "/orca_oo/gradient_fd", natm, &ctx);
    let g_orca = grad(&r, "/orca_oo/gradient", natm, &ctx);
    check_grad(&ctx, "OO grad vs ORCA FD", &g, &g_fd, TOL_G_FD);
    check_grad(
        &ctx,
        "OO grad vs ORCA analytic",
        &g,
        &g_orca,
        TOL_G_ORCA_ANALYTIC,
    );

    // ---- negative control: the plain RI-MP2 gradient misses it ----
    let g_mp2 = rimp2_gradient_analytical(
        &s.mol, &s.obs, &s.dfbs, s.op, &s.bounds, &rhf, &mp2_cfg, None,
    )
    .unwrap();
    let d = max_abs_diff(&g_mp2, &g_fd);
    eprintln!(
        "{ctx}: {:<26} max|d| {d:.2e} (must exceed {MUST_MISS_G:.0e})",
        "RI-MP2 grad vs OO ref"
    );
    assert!(
        d > MUST_MISS_G,
        "{ctx}: negative control: plain RI-MP2 gradient within {d:.2e} of the OO reference — \
         the gradient bar cannot tell OO from non-OO"
    );
}

/// Total OO-RI-MP2 energy at `mol`, fully re-converged (RHF + OO).
fn oo_energy_at(mol: Molecule, ctx: &str) -> f64 {
    let s = Sys::new(mol);
    let rhf = solve_closed(&s, ctx);
    solve_oo(&s, &rhf, ctx).total_energy
}

/// ferric analytic OO gradient vs 5-point FD of ferric's own OO energy: the
/// TIGHT gradient anchor (the ORCA FD is taken on ORCA's non-minimal surface).
fn check_self_fd(system: &str) {
    let r = reference(system);
    let ctx = format!("{system}/{BASIS} self-FD");
    let s = load(system, &r);
    let natm = s.mol.atoms.len();
    let rhf = solve_closed(&s, &ctx);
    let oo = solve_oo(&s, &rhf, &ctx);
    let g = oo_ri_mp2_gradient(&s.mol, &s.obs, &s.dfbs, s.op, &s.bounds, &oo, 0, None).unwrap();
    let stencil = [
        (-2.0, 1.0 / 12.0),
        (-1.0, -8.0 / 12.0),
        (1.0, 8.0 / 12.0),
        (2.0, -1.0 / 12.0),
    ];
    let mut g_fd = Array2::<f64>::zeros((natm, 3));
    for a in 0..natm {
        for k in 0..3 {
            let mut acc = 0.0;
            for (step, w) in stencil {
                let mut m = s.mol.clone();
                let at = &mut m.atoms[a];
                let d = step * H_FD;
                match k {
                    0 => at.x += d,
                    1 => at.y += d,
                    _ => at.zpos += d,
                }
                acc += w * oo_energy_at(m, &ctx);
            }
            g_fd[(a, k)] = acc / H_FD;
        }
    }
    check_grad(&ctx, "OO analytic vs own FD", &g, &g_fd, TOL_G_SELF_FD);
}

/// Open-shell (UHF-based) OO-RI-MP2 energy.
fn check_open(system: &str) {
    let r = reference(system);
    let ctx = format!("{system}/{BASIS}");
    let s = load(system, &r);

    // ---- exactness anchor: UHF and plain U-RI-MP2 ----
    let uhf = solve_uhf(
        &ParallelContext::default(),
        &s.mol,
        &s.obs,
        &s.bounds,
        &uhf_config(),
    )
    .unwrap_or_else(|e| panic!("{ctx}: solve_uhf failed: {e:?}"));
    assert!(uhf.converged, "{ctx}: UHF not converged");
    assert!(
        matches!(uhf.spin, Spin::Unrestricted),
        "{ctx}: not a UHF result"
    );
    check_close(
        &ctx,
        "E_UHF vs ORCA",
        uhf.energy,
        num(&r, "/orca_rimp2/e_scf", &ctx),
        TOL_E_SCF,
    );
    check_close(
        &ctx,
        "E_UHF vs PySCF (stable)",
        uhf.energy,
        num(&r, "/pyscf/e_scf", &ctx),
        TOL_E_SCF,
    );
    let mp2_cfg = RiMp2Config::default();
    assert_eq!(mp2_cfg.frozen_core, 0);
    let mp2 = u_ri_mp2(&s.mol, &s.obs, &s.dfbs, s.op, &uhf, &mp2_cfg).unwrap();
    check_close(
        &ctx,
        "E_URIMP2 vs ORCA",
        mp2.total_energy,
        num(&r, "/orca_rimp2/e_total", &ctx),
        TOL_E_RIMP2,
    );

    // ---- U-OO-RI-MP2 energy ----
    let cfg = UOoRiMp2Config {
        grad_conv: 1e-8,
        energy_conv: 1e-11,
        max_iter: 200,
        ..Default::default()
    };
    assert_eq!(cfg.frozen_core, 0);
    let oo = u_oo_ri_mp2(&s.mol, &s.obs, &s.dfbs, s.op, &s.bounds, &uhf, &cfg, None)
        .unwrap_or_else(|e| panic!("{ctx}: u_oo_ri_mp2 failed: {e:?}"));
    assert!(
        oo.converged,
        "{ctx}: U-OO-RI-MP2 not converged (|g| {:.2e})",
        oo.grad_norm
    );
    let e_oo_ref = check_oo_energies(
        &ctx,
        &r,
        "E_UOO",
        oo.hf_energy,
        oo.mp2_corr,
        oo.total_energy,
    );

    // ---- negative control ----
    check_miss_e(
        &ctx,
        "ferric U-RI-MP2 vs numpy OO",
        mp2.total_energy,
        e_oo_ref,
    );
    check_miss_e(
        &ctx,
        "ORCA U-RI-MP2 vs ORCA OO",
        num(&r, "/orca_rimp2/e_total", &ctx),
        num(&r, "/orca_oo/e_total", &ctx),
    );
}

#[test]
#[ignore = "validation: OO-RI-MP2 energy"]
fn oo_rimp2_h2o_ccpvdz_energy_and_gradient_vs_orca() {
    check_closed("h2o_distorted");
}

#[test]
#[ignore = "validation: OO-RI-MP2 energy"]
fn oo_rimp2_nh3_ccpvdz_energy_and_gradient_vs_orca() {
    check_closed("nh3_distorted");
}

#[test]
#[ignore = "validation: OO-RI-MP2 energy"]
fn u_oo_rimp2_ch3_ccpvdz_energy_vs_orca() {
    check_open("ch3");
}

#[test]
#[ignore = "validation: OO-MP2 orbital and nuclear gradients"]
fn oo_rimp2_h2o_ccpvdz_gradient_self_fd() {
    check_self_fd("h2o_distorted");
}

#[test]
#[ignore = "validation: OO-MP2 orbital and nuclear gradients"]
fn oo_rimp2_nh3_ccpvdz_gradient_self_fd() {
    check_self_fd("nh3_distorted");
}
