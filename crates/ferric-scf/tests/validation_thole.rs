//! VALIDATION tier: VALIDATION.md row "Thole polarizable embedding".
//!
//! Runs only in the weekly `validation` CI job (and on demand):
//!
//! ```text
//! cargo nextest run -p ferric-scf --test validation_thole \
//!     --run-ignored only -E 'binary(/^validation_/)'
//! ```
//!
//! # What is compared
//!
//! ferric's Thole-damped induced-dipole embedding (`ferric_scf::polarizable`:
//! `induce` inside the SCF loop via `RhfConfig.polarizable`,
//! `rhf/uhf_gradient_with_polarizable` for the QM nuclei,
//! `qmmm::full_gradient_with_polarizable` for the MM rows) against an
//! INDEPENDENT numpy implementation of the same model driven through PySCF
//! 2.13 (`scripts/validation/gen_thole.py`; no external polarizable-embedding
//! code is installed). The reference uses exact point-dipole field integrals
//! (`int1e_iprinv`), a dense induction solve, `qmmm.mm_charge` for the
//! permanent charges, and patched `get_veff`/`energy_elec` for the
//! self-consistent polarization. Its gradients are 5-point central finite
//! differences of the fully reconverged total energy (h and 2h agree to
//! ≤ 4.2e-9 Ha/Bohr), and its MM rows are cross-checked inside the generator
//! against the stationary functional
//! `W = −μ·E0 + ½ μ·B·μ` at fixed (D, μ) (≤ 3.7e-10 Ha/Bohr, refused above 2e-7).
//!
//! | test | QM / basis | MM | method |
//! |---|---|---|---|
//! | `thole_h2o_w4_excl_ccpvdz` | H2O / cc-pVDZ | 4 TIP3P waters, α O 0.837 / H 0.496 Å³, intramolecular site pairs excluded | RHF |
//! | `thole_h2o_w4_noexcl_ccpvdz` | H2O / cc-pVDZ | same waters, NO exclusions (the `run_qmmm` default), α × 0.3 | RHF |
//! | `thole_nh2_w3_uhf_ccpvdz` | NH2 (²B1) / cc-pVDZ | 3 TIP3P waters, exclusions | UHF |
//! | `anchor_far_site_classical_limit` | H2O / cc-pVDZ | 4 waters + a +1 ion as fixed charges, ONE site (10 Å³) 30 Å away | RHF |
//! | `anchor_tiny_alpha_is_point_charge_embedding` | H2O / cc-pVDZ | the 4 waters, every α × 1e-12 | RHF |
//!
//! # The model both sides implement (ferric file:line)
//!
//! `μ = B⁻¹ E0`, `B_ii = I/α_i`, `B_ij = −T_ij` (polarizable.rs:359-413);
//! `E0 = E_QM(D) + E_perm`, permanent fields undamped, a site skips its own
//! charge and the charges of excluded partner sites (polarizable.rs:285-354);
//! exponential Thole `λ3 = 1 − e^{−a u³}`, `λ5 = 1 − (1 + a u³) e^{−a u³}`,
//! `u = r/(α_i α_j)^{1/6}`, `a = 2.1304` (polarizable.rs:144, 262-271);
//! `E_pol = −½ Σ μ·E0` as a standalone energy term (polarizable.rs:556-562,
//! rhf.rs:1498, uhf.rs:940); Fock term `V = −Σ μ·⟨(r−R)/|r−R|³⟩` with no ½,
//! re-solved every iteration (polarizable.rs:436-440, driver.rs:376-392).
//!
//! THE TENSOR SIGN. The reference uses the physical dipole-field tensor
//! `T = (3λ5 r̂r̂ − λ3 I)/r³` (the field at R_i of a dipole μ_j is `T μ_j`;
//! the generator derives it from −∇ of the dipole potential and refuses to
//! write if the closed form disagrees). ferric's `thole_tensor` returns the
//! same tensor. Every reference also carries the control `negated_tensor`
//! (the opposite sign, which head-to-tail dipoles weaken instead of
//! reinforce); the energy assertion reports ferric's distance to BOTH, so a
//! sign regression names itself. The far-site and tiny-α anchors have no site
//! pairs and cannot see this sign.
//!
//! # Physics hypothesis vs artifact hypothesis
//!
//! * If ferric implements the model: energies and E_pol agree at the
//!   point-dipole / SCF floor, dipoles likewise, gradients at the FD floor.
//! * Polarization never reaches the SCF (the vacuum-test failure mode):
//!   ferric returns the plain point-charge embedding energy, which is
//!   3.5e-3 / 0.38 / 7.3e-4 Ha from the three references (asserted as a
//!   control).
//! * Mutual induction dropped (T = 0): the `no_mutual` control is 2.8e-4 /
//!   0.12 / 5.4e-6 Ha away.
//! * Tensor sign flipped: `negated_tensor` is 3.9e-4 / 0.17 / 1.0e-5 Ha away.
//! * Damping dropped: `no_damping` is 2.2e-4 Ha away in the no-exclusion
//!   case. With intramolecular exclusions the closest site pair is an
//!   intermolecular H-bond (u ≈ 2.3, damping ~e^{−25}), so the damping gap
//!   there is ≤ 2.1e-13 Ha and that control is asserted ONLY in the no-exclusion
//!   test.
//! * Naive fixed-μ site gradient (`−½ μ·dE0/dR`, no T derivative): misses
//!   the FD by 2.0e-3 (excl) / 0.10 (noexcl) Ha/Bohr; ferric must miss it too.
//! * Harness broken (geometry constant, basis file, site order): the
//!   nuclear-repulsion, AO-count and site-count assertions fail first.
//!
//! # Why the no-exclusion case uses α × 0.3
//!
//! With `exp(−a u³)` damping at `a = 2.1304`, the full α (O 0.837, H 0.496 Å³)
//! puts every intramolecular O–H site pair in the polarization catastrophe:
//! B is indefinite (min eigenvalue −0.094) and the SCF lands on a saddle with
//! an MM-only E_pol of +10 Ha. At α × 0.3, B's min eigenvalue is 0.32 and
//! the damping still moves the energy by 2.2e-4 Ha.
//!
//! # TOLERANCES
//!
//! Each bar is 3–25x the worst |d| measured over all cases (2026-09-25),
//! noted beside it.
//!
//! # MUTATIONS (run 2026-09-25; each turned the listed tests red)
//!
//! * `thole_tensor` back to `(λ3 I − 3λ5 r̂r̂)/r³` (the opposite sign): all
//!   three `thole_*` tests, each landing on the `negated_tensor` control.
//! * `(1.0 - expo, 1.0 - (1.0 + au3) * expo)` → `(1.0 - expo, 1.0 - expo)`
//!   (wrong λ5): `thole_h2o_w4_noexcl_ccpvdz`, the case where damping is
//!   resolvable.
//! * `let coeff = -mu[c] * norm_p;` → `mu[c] * norm_p` (wrong-sign Fock term):
//!   all three `thole_*` tests.
//! * `g[k] += -1.0 * d_dot_dr;` → `-0.5 * d_dot_dr` (tensor-derivative term):
//!   the MM rows of both water tests.
//!
//! A missing reference JSON is a HARD failure (panic naming the path).

use std::path::{Path, PathBuf};

use ferric_core::basis::{self, BasisSet};
use ferric_core::elements;
use ferric_core::external_potential::ExternalPotential;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_integrals::site_basis::SiteBasis;
use ferric_scf::gradient::{
    rhf_gradient, rhf_gradient_with_polarizable, uhf_gradient, uhf_gradient_with_polarizable,
};
use ferric_scf::polarizable::{induce, PolarizableSites, DEFAULT_THOLE_A};
use ferric_scf::qmmm::{
    full_gradient_with_polarizable, mm_forces, QmSelection, QmmmAtom, QmmmSystem,
};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::solve_uhf;
use ferric_scf::ScfResult;
use ndarray::Array2;
use serde_json::Value;

const ROW_DIR: &str = "testdata/reference/validation/thole";

/// Total energy and the plain-embedding control energy: measured 3.7e-12.
const TOL_E: f64 = 5e-11;
/// E_pol alone: measured 1.5e-9 (the 0.38 Ha no-exclusion case). E_pol is a
/// component, not stationary in mu, so it carries the SCF floor the total
/// energy does not.
const TOL_E_POL: f64 = 1e-8;
/// Induced dipoles, max abs component.
const TOL_MU: f64 = 1e-8; // measured 1.5e-9 a.u.
/// QM gradient and MM full-gradient rows vs the 5-point FD reference
/// (FD h-vs-2h floor ≤ 4.2e-9 Ha/Bohr).
const TOL_G: f64 = 3e-8; // measured 3.0e-9 Ha/Bohr
/// Far-site E_pol vs the classical limit −½α|E0|², relative. The numpy SCF
/// itself sits 7.7e-8 from it (the back-reaction of a 30 Å dipole).
const TOL_FAR_REL: f64 = 1e-6; // measured 7.7e-8
/// Tiny-α anchor vs PySCF `mm_charge` (validation_qmmm measured 4.8e-12 for
/// plain embedding; residual E_pol at α × 1e-12 is ~1e-13 Ha).
const TOL_ANCHOR: f64 = 3e-11; // measured 3.0e-12 Ha
const TOL_ENUC: f64 = 1e-9;
const TINY_ALPHA_SCALE: f64 = 1e-12;
/// A control must miss by this multiple of the bar it is asserted against.
const MISS_FACTOR: f64 = 10.0;

// ---------------------------------------------------------------------------
// Reference loading
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

fn reference(file_stem: &str) -> Value {
    let path = workspace_root()
        .join(ROW_DIR)
        .join(format!("{file_stem}.json"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "missing validation reference {} ({e}); regenerate with \
             scripts/validation/gen_thole.py — a missing reference is a failure, never a skip",
            path.display()
        )
    });
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{}: bad JSON: {e}", path.display()))
}

fn num(v: &Value, ptr: &str) -> f64 {
    v.pointer(ptr)
        .and_then(Value::as_f64)
        .unwrap_or_else(|| panic!("reference field {ptr} missing or not a number"))
}

fn xyz3(v: &Value) -> [f64; 3] {
    let a = v.as_array().expect("xyz array");
    [
        a[0].as_f64().unwrap(),
        a[1].as_f64().unwrap(),
        a[2].as_f64().unwrap(),
    ]
}

fn rows3(v: &Value, ptr: &str) -> Vec<[f64; 3]> {
    v.pointer(ptr)
        .and_then(Value::as_array)
        .unwrap_or_else(|| panic!("reference field {ptr} missing"))
        .iter()
        .map(xyz3)
        .collect()
}

// ---------------------------------------------------------------------------
// ferric side
// ---------------------------------------------------------------------------

/// QM atoms first (dummy charge, dropped as QM), then one `QmmmAtom` per MM
/// atom carrying BOTH its permanent charge and its polarisability (Bohr³,
/// × `alpha_scale`), exactly as `run_qmmm` builds them.
fn build_system(r: &Value, alpha_scale: f64) -> QmmmSystem {
    let mut atoms: Vec<QmmmAtom> = r["atoms"]
        .as_array()
        .unwrap()
        .iter()
        .map(|a| {
            let sym = a["symbol"].as_str().unwrap();
            let z = elements::symbol_to_z(sym).unwrap();
            let [x, y, zz] = xyz3(&a["xyz_bohr"]);
            QmmmAtom::new(sym, z, x, y, zz, 99.0)
        })
        .collect();
    let nqm = atoms.len();
    for a in r["mm_atoms"].as_array().unwrap() {
        let [x, y, z] = xyz3(&a["xyz_bohr"]);
        let q = a["q"].as_f64().unwrap();
        let alpha = a["alpha_bohr3"].as_f64().unwrap() * alpha_scale;
        atoms.push(QmmmAtom::new("X", 0, x, y, z, q).with_alpha(alpha));
    }
    let mult = r["multiplicity"].as_u64().unwrap() as usize;
    QmmmSystem::new(
        &atoms,
        QmSelection::Indices((0..nqm).collect()),
        r["charge"].as_i64().unwrap() as i32,
        mult,
    )
    .unwrap()
}

fn pol_sites(r: &Value, sys: &QmmmSystem, thole_a: Option<f64>) -> PolarizableSites {
    let exclusions = r["exclusions"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| {
            let p = p.as_array().unwrap();
            (
                p[0].as_u64().unwrap() as usize,
                p[1].as_u64().unwrap() as usize,
            )
        })
        .collect();
    PolarizableSites {
        sites: sys.to_polarizable_sites(),
        thole_a,
        exclusions,
        ..Default::default()
    }
}

struct Qm {
    mol: Molecule,
    prep: PreparedBasis,
    bounds: SchwarzBounds,
    _bs: BasisSet,
}

fn qm_part(r: &Value, sys: &QmmmSystem, ctx: &str) -> Qm {
    let mol = sys.to_qm_molecule();
    assert_eq!(
        mol.atoms.len(),
        r["atoms"].as_array().unwrap().len(),
        "{ctx}: QM atom count"
    );
    let enuc = mol.nuclear_repulsion();
    let enuc_ref = num(r, "/nuclear_repulsion");
    assert!(
        (enuc - enuc_ref).abs() < TOL_ENUC,
        "{ctx}: nuclear repulsion {enuc:.12} vs reference {enuc_ref:.12} — geometry/unit mismatch"
    );
    let bs = basis::bundled(r["basis"].as_str().unwrap()).unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let nao = ferric_integrals::oneelectron::overlap(&prep).nrows();
    assert_eq!(nao as u64, r["nao"].as_u64().unwrap(), "{ctx}: AO count");
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
    Qm {
        mol,
        prep,
        bounds,
        _bs: bs,
    }
}

fn is_uhf(r: &Value) -> bool {
    r["method"].as_str().unwrap() == "uhf"
}

fn scf(
    qm: &Qm,
    ext: Option<ExternalPotential>,
    pol: Option<PolarizableSites>,
    uhf: bool,
    ctx: &str,
) -> ScfResult {
    let cfg = RhfConfig {
        external_potential: ext,
        polarizable: pol,
        energy_conv: 1e-11,
        density_conv: 1e-10,
        max_iter: 300,
        ..Default::default()
    };
    let ctx_p = ParallelContext::default();
    let res = if uhf {
        solve_uhf(&ctx_p, &qm.mol, &qm.prep, &qm.bounds, &cfg)
    } else {
        solve_rhf(
            &ctx_p,
            &qm.mol,
            &qm.prep,
            Operator::coulomb(),
            &qm.bounds,
            &cfg,
        )
    }
    .unwrap_or_else(|e| panic!("{ctx}: SCF failed: {e:?}"));
    assert!(res.converged, "{ctx}: SCF not converged");
    res
}

/// E_pol and μ at ferric's converged density (the `run_qmmm` recipe:
/// `ScfResult` carries μ but not E_pol).
fn e_pol_of(
    qm: &Qm,
    ext: Option<&ExternalPotential>,
    pol: &PolarizableSites,
    res: &ScfResult,
) -> (f64, Array2<f64>) {
    let site_xyz: Vec<[f64; 4]> = pol
        .sites
        .iter()
        .map(|s| [s.x, s.y, s.z, pol.dipole_zeta])
        .collect();
    let basis_p = SiteBasis::new(&site_xyz, 1).unwrap();
    let ir = induce(&qm.mol, &qm.prep, ext, pol, &basis_p, res.density_total()).unwrap();
    (ir.e_pol, ir.dipoles)
}

fn qm_gradient(
    qm: &Qm,
    res: &ScfResult,
    ext: Option<&ExternalPotential>,
    pol: Option<&PolarizableSites>,
    uhf: bool,
) -> Array2<f64> {
    let op = Operator::coulomb();
    match (uhf, pol) {
        (false, Some(p)) => rhf_gradient_with_polarizable(
            &qm.mol,
            &qm.prep,
            op,
            &qm.bounds,
            res,
            ext,
            Some(p),
            res.induced_dipoles.as_ref(),
        ),
        (true, Some(p)) => uhf_gradient_with_polarizable(
            &qm.mol,
            &qm.prep,
            op,
            &qm.bounds,
            res,
            ext,
            Some(p),
            res.induced_dipoles.as_ref(),
        ),
        (false, None) => rhf_gradient(&qm.mol, &qm.prep, op, &qm.bounds, res, ext),
        (true, None) => uhf_gradient(&qm.mol, &qm.prep, op, &qm.bounds, res, ext),
    }
    .unwrap()
}

// ---------------------------------------------------------------------------
// Assertions
// ---------------------------------------------------------------------------

fn check_close(ctx: &str, what: &str, got: f64, want: f64, tol: f64) {
    let d = (got - want).abs();
    eprintln!("{ctx}: {what:<30} ferric {got:+.12} ref {want:+.12} |d| {d:.2e} (tol {tol:.0e})");
    assert!(
        d < tol,
        "{ctx}: {what} {got:.12} vs reference {want:.12} (|d| {d:.2e} >= {tol:.0e})"
    );
}

fn check_misses(ctx: &str, what: &str, d: f64, tol: f64) {
    let bar = MISS_FACTOR * tol;
    eprintln!("{ctx}: CONTROL {what:<44} |d| {d:.2e} (must exceed {bar:.0e})");
    assert!(
        d > bar,
        "{ctx}: negative control '{what}' did NOT miss (|d| {d:.2e} <= {bar:.0e}) \
         — the assertion cannot tell this arrangement from the real one"
    );
}

fn max_row_diff(a: &[[f64; 3]], b: &[[f64; 3]]) -> (f64, usize, usize) {
    assert_eq!(a.len(), b.len(), "row count");
    let mut worst = (0.0, 0, 0);
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        for k in 0..3 {
            let d = (x[k] - y[k]).abs();
            if d > worst.0 {
                worst = (d, i, k);
            }
        }
    }
    worst
}

fn rows_of(g: &Array2<f64>) -> Vec<[f64; 3]> {
    g.rows().into_iter().map(|r| [r[0], r[1], r[2]]).collect()
}

fn check_rows(ctx: &str, what: &str, got: &[[f64; 3]], want: &[[f64; 3]], tol: f64) {
    let (d, i, k) = max_row_diff(got, want);
    eprintln!(
        "{ctx}: {what:<30} max|d| {d:.2e} at [{i}][{k}] over {} rows (tol {tol:.0e})",
        got.len()
    );
    assert!(
        d < tol,
        "{ctx}: {what}[{i}][{k}] ferric {} vs reference {} (|d| {d:.2e} >= {tol:.0e})",
        got[i][k],
        want[i][k]
    );
}

/// The controls this test asserts, and the energy gap each one has from the
/// reference (read from the JSON and checked REACHABLE before use).
fn control_energy(r: &Value, ctx: &str, name: &str) -> f64 {
    let c = &r["controls"][name];
    assert!(
        c["converged"].as_bool().unwrap_or(true),
        "{ctx}: control {name} did not converge in the generator"
    );
    let gap = num(r, &format!("/controls/{name}/abs_diff_energy_vs_reference"));
    assert!(
        gap > 2.0 * MISS_FACTOR * TOL_E,
        "{ctx}: control {name} is only {gap:.2e} Ha from the reference — UNREACHABLE at \
         TOL_E {TOL_E:.0e}; it cannot discriminate and must not be asserted"
    );
    num(r, &format!("/controls/{name}/energy"))
}

// ---------------------------------------------------------------------------
// One full case
// ---------------------------------------------------------------------------

fn check_case(file_stem: &str, damping_control: bool, mm_rows: bool) {
    let r = reference(file_stem);
    let ctx = file_stem.to_string();
    let uhf = is_uhf(&r);
    let thole_a = r["thole_a"].as_f64().unwrap();
    assert_eq!(
        thole_a, DEFAULT_THOLE_A,
        "{ctx}: reference damping parameter"
    );
    let sys = build_system(&r, 1.0);
    let qm = qm_part(&r, &sys, &ctx);
    let n_mm = r["mm_atoms"].as_array().unwrap().len();
    let pol = pol_sites(&r, &sys, Some(thole_a));
    assert_eq!(
        pol.sites.len(),
        n_mm,
        "{ctx}: every MM atom is a polarizable site"
    );
    let ext = sys.to_external_potential();
    assert!(ext.is_some(), "{ctx}: MM region produced no potential");

    // --- plain point-charge embedding: harness check AND anti-vacuum control
    let plain = scf(&qm, ext.clone(), None, uhf, &format!("{ctx}/plain"));
    check_close(
        &ctx,
        "E plain embedding",
        plain.energy,
        num(&r, "/energy_plain_embedding"),
        TOL_E,
    );
    let e_ref = num(&r, "/energy");
    check_misses(
        &ctx,
        "plain embedding vs polarizable ref",
        (plain.energy - e_ref).abs(),
        TOL_E,
    );

    // --- polarizable ------------------------------------------------------
    let res = scf(&qm, ext.clone(), Some(pol.clone()), uhf, &ctx);
    let e_neg = num(&r, "/controls/negated_tensor/energy");
    let d_ref = (res.energy - e_ref).abs();
    let d_neg = (res.energy - e_neg).abs();
    eprintln!("{ctx}: DIAGNOSTIC |E - ref| {d_ref:.2e}   |E - negated_tensor control| {d_neg:.2e}");
    assert!(
        d_ref < TOL_E,
        "{ctx}: E {:.12} vs reference {e_ref:.12} (|d| {d_ref:.2e} >= {TOL_E:.0e}); ferric is \
         {d_neg:.2e} from the NEGATED-tensor control{}",
        res.energy,
        if d_neg < TOL_E {
            " — thole_tensor has the wrong sign: it must be the dipole field tensor \
             (3λ5 r̂r̂ − λ3 I)/r³"
        } else {
            ""
        }
    );
    let (e_pol, mu) = e_pol_of(&qm, ext.as_ref(), &pol, &res);
    check_close(&ctx, "E_pol", e_pol, num(&r, "/e_pol"), TOL_E_POL);
    let mu_ref = rows3(&r, "/induced_dipoles");
    let mu_rows = rows_of(&mu);
    check_rows(&ctx, "induced dipoles (induce)", &mu_rows, &mu_ref, TOL_MU);
    let mu_scf = rows_of(res.induced_dipoles.as_ref().expect("induced_dipoles"));
    check_rows(
        &ctx,
        "induced dipoles (ScfResult)",
        &mu_scf,
        &mu_ref,
        TOL_MU,
    );

    // --- controls: ferric must miss every discriminating control --------
    for name in ["no_mutual", "negated_tensor"] {
        let e_c = control_energy(&r, &ctx, name);
        check_misses(
            &ctx,
            &format!("ferric vs {name} control"),
            (res.energy - e_c).abs(),
            TOL_E,
        );
    }
    if damping_control {
        let e_c = control_energy(&r, &ctx, "no_damping");
        check_misses(
            &ctx,
            "ferric vs no_damping control",
            (res.energy - e_c).abs(),
            TOL_E,
        );
        // And ferric with damping OFF must land on the undamped reference.
        let undamped = scf(
            &qm,
            ext.clone(),
            Some(pol_sites(&r, &sys, None)),
            uhf,
            &format!("{ctx}/undamped"),
        );
        check_close(
            &ctx,
            "E damping off vs no_damping",
            undamped.energy,
            e_c,
            TOL_E,
        );
    }

    // --- QM gradient ------------------------------------------------------
    let g_ref = rows3(&r, "/qm_gradient_fd");
    let g = qm_gradient(&qm, &res, ext.as_ref(), Some(&pol), uhf);
    check_rows(&ctx, "QM gradient vs FD", &rows_of(&g), &g_ref, TOL_G);
    let g_nopol = qm_gradient(&qm, &res, ext.as_ref(), None, uhf);
    let (d_nopol, _, _) = max_row_diff(&rows_of(&g_nopol), &g_ref);
    check_misses(
        &ctx,
        "QM gradient without the polarizable term",
        d_nopol,
        TOL_G,
    );

    if !mm_rows {
        return;
    }

    // --- MM rows: charge force + site gradient + charge reaction rows -----
    let fd = rows3(&r, "/mm_gradient_fd");
    let naive = rows3(&r, "/mm_gradient_naive_fixed_mu");
    let (d_naive_fd, _, _) = max_row_diff(&naive, &fd);
    check_misses(
        &ctx,
        "reference: naive fixed-mu rows vs FD",
        d_naive_fd,
        TOL_G,
    );
    let forces = mm_forces(&sys, &qm.mol, &qm.prep, res.density_total()).unwrap();
    let full = full_gradient_with_polarizable(
        &sys,
        &g,
        &forces,
        ext.as_ref(),
        &pol,
        res.induced_dipoles.as_ref(),
        &qm.mol,
        &qm.prep,
        res.density_total(),
    )
    .unwrap();
    let nqm = qm.mol.atoms.len();
    assert_eq!(full.nrows(), nqm + n_mm, "{ctx}: full-gradient row count");
    let full_rows = rows_of(&full);
    check_rows(
        &ctx,
        "full gradient QM rows vs FD",
        &full_rows[..nqm],
        &g_ref,
        TOL_G,
    );
    check_rows(
        &ctx,
        "full gradient MM rows vs FD",
        &full_rows[nqm..],
        &fd,
        TOL_G,
    );
    let (d_ferric_naive, _, _) = max_row_diff(&full_rows[nqm..], &naive);
    check_misses(
        &ctx,
        "ferric MM rows vs naive fixed-mu rows",
        d_ferric_naive,
        TOL_G,
    );
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
#[ignore = "validation: Thole embedding"]
fn thole_h2o_w4_excl_ccpvdz() {
    check_case("h2o_w4_excl_cc-pvdz", false, true);
}

#[test]
#[ignore = "validation: Thole embedding"]
fn thole_h2o_w4_noexcl_ccpvdz() {
    check_case("h2o_w4_noexcl_cc-pvdz", true, true);
}

#[test]
#[ignore = "validation: Thole embedding"]
fn thole_nh2_w3_uhf_ccpvdz() {
    check_case("nh2_w3_uhf_cc-pvdz", false, false);
}

/// Anchor: ONE polarizable site (no charge) 30 Å from the QM region. Its
/// dipole barely acts back on the density, so E_pol must reach the
/// classical limit −½ α |E0|², E0 = the field of the NON-polarized embedded
/// system at the site (PySCF density + nuclei + MM charges). No site pairs:
/// blind to the tensor sign by construction.
#[test]
#[ignore = "validation: Thole embedding"]
fn anchor_far_site_classical_limit() {
    let r = reference("far_site_cc-pvdz");
    let ctx = "anchor/far_site";
    let sys = build_system(&r, 1.0);
    let qm = qm_part(&r, &sys, ctx);
    let pol = pol_sites(&r, &sys, Some(DEFAULT_THOLE_A));
    assert_eq!(pol.sites.len(), 1, "{ctx}: exactly one polarizable site");
    let ext = sys.to_external_potential();
    let plain = scf(&qm, ext.clone(), None, false, &format!("{ctx}/plain"));
    check_close(
        ctx,
        "E plain embedding",
        plain.energy,
        num(&r, "/energy_plain_embedding"),
        TOL_E,
    );
    let res = scf(&qm, ext.clone(), Some(pol.clone()), false, ctx);
    let (e_pol, mu) = e_pol_of(&qm, ext.as_ref(), &pol, &res);
    let e_cl = num(&r, "/far_site/e_pol_classical_limit");
    let rel = (e_pol - e_cl).abs() / e_cl.abs();
    eprintln!(
        "{ctx}: E_pol {e_pol:+.10e} classical {e_cl:+.10e} rel {rel:.2e} (tol {TOL_FAR_REL:.0e})"
    );
    assert!(
        rel < TOL_FAR_REL,
        "{ctx}: E_pol misses the classical limit (rel {rel:.2e})"
    );
    let mu_cl = xyz3(&r["far_site"]["mu_classical"]);
    let mu_norm = (mu_cl[0].powi(2) + mu_cl[1].powi(2) + mu_cl[2].powi(2)).sqrt();
    for k in 0..3 {
        let d = (mu[(0, k)] - mu_cl[k]).abs() / mu_norm;
        assert!(
            d < TOL_FAR_REL,
            "{ctx}: mu[{k}] {} vs classical {} (rel {d:.2e})",
            mu[(0, k)],
            mu_cl[k]
        );
    }
    check_close(ctx, "E_pol vs numpy SCF", e_pol, num(&r, "/e_pol"), TOL_E);
    check_close(
        ctx,
        "E(pol) - E(plain)",
        res.energy - plain.energy,
        num(&r, "/energy_difference_pol_minus_plain"),
        TOL_E,
    );
    // Non-triviality: E_pol (2.6e-6 Ha) is far above the energy bar.
    check_misses(ctx, "E_pol vs zero", e_pol.abs(), TOL_E);
}

/// Anchor: every α × 1e-12 (the α → 0 limit; α = 0 exactly is "not a
/// site"). ferric must reduce to plain point-charge embedding, i.e. to PySCF
/// `qmmm.mm_charge`.
#[test]
#[ignore = "validation: Thole embedding"]
fn anchor_tiny_alpha_is_point_charge_embedding() {
    let r = reference("h2o_w4_excl_cc-pvdz");
    let ctx = "anchor/tiny_alpha";
    let sys = build_system(&r, TINY_ALPHA_SCALE);
    let qm = qm_part(&r, &sys, ctx);
    let pol = pol_sites(&r, &sys, Some(DEFAULT_THOLE_A));
    assert_eq!(
        pol.sites.len(),
        r["mm_atoms"].as_array().unwrap().len(),
        "{ctx}: site count"
    );
    let ext = sys.to_external_potential();
    let res = scf(&qm, ext.clone(), Some(pol.clone()), false, ctx);
    check_close(
        ctx,
        "E(alpha x 1e-12) vs PySCF mm_charge",
        res.energy,
        num(&r, "/energy_plain_embedding"),
        TOL_ANCHOR,
    );
    // The same geometry at full alpha is 3.5e-3 Ha away: the anchor is not vacuous.
    check_misses(
        ctx,
        "full-alpha reference vs mm_charge",
        (num(&r, "/energy") - res.energy).abs(),
        TOL_ANCHOR,
    );
}
