//! VALIDATION tier: EE-EOM-CCSD singlets (`ferric_cc::eom_ccsd`) against
//! PySCF `eom_rccsd.EOMEESinglet` on like-for-like DF integrals.
//!
//! ```text
//! cargo nextest run -p ferric-cc --test validation_eom_ccsd \
//!     --run-ignored only -E 'binary(/^validation_/)'
//! ```
//!
//! # What is compared
//!
//! The lowest five singlet excitation energies of H2O and C2H4 / cc-pVDZ
//! (cc-pVDZ-RI), all electrons correlated, against
//! `scripts/validation/gen_eom_ccsd.py`, stored in
//! `testdata/reference/validation/eom_ccsd/<system>_cc-pvdz.json`. The
//! reference makes the same approximation as ferric: exact-integral RHF, then
//! every correlation integral from one DF factorization with ferric's bundled
//! aux basis and a diagonal canonical Fock (the `gen_cc.py` construction).
//! PySCF's CCSD and EOM product then run on those integrals; its Davidson
//! starts from spread guesses and keeps the lowest real roots, and on water
//! the whole packed operator is also diagonalized (the root-selection check).
//!
//! # Tolerances
//!
//! * E_RHF 1e-9 and E_CCSD 2e-10 (the `validation_cc.rs` bars, same systems
//!   class).
//! * ω: [`TOL_OMEGA`]. Error sources: the CCSD amplitudes (ferric converges
//!   |ΔE| < 1e-10, PySCF |ΔE| < 1e-12, |Δt| < 1e-10; ω moves at first order
//!   in δt), the two Davidson residuals (ferric 1e-8 here, PySCF ≤ 1e-8 and
//!   recorded per root; first order for a non-symmetric problem) and the
//!   integral engines (~1e-12). The measured maximum is recorded on the
//!   constant.
//!
//! # Mutation
//!
//! Flipping the sign of the doubles-singles `(ovvv·r1)_af t2` term in
//! `eom_ccsd::sigma` moves the H2O roots by 5.5e-4 to 1.3e-3 Ha and fails
//! the ω assertion.
//!
//! # Physics vs artifact
//!
//! * Correct product: ω agrees to the bar above.
//! * A wrong doubles-singles or singles-doubles term: the singly excited
//!   roots move by ~1e-4 to 1e-2 Ha, far above the bar.
//! * A Davidson that skipped a root: the k-th ferric root lands on the
//!   reference's (k+1)-th; the roots here are separated by ≥ 3e-3 Ha.
//! * A harness slip (geometry, basis, aux, exact-vs-DF): the nuclear
//!   repulsion, AO/aux counts, E_RHF and E_CCSD checks fail first.

use std::path::{Path, PathBuf};

use ferric_cc::ccsd_closed_shell::ccsd_closed_shell;
use ferric_cc::eom_ccsd::{eom_ccsd_singlets, EomCcsdConfig, EOM_CCSD_ENERGY_CONV};
use ferric_cc::CcConfig;
use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use serde_json::Value;

const ROW_DIR: &str = "testdata/reference/validation/eom_ccsd";
const TOL_ENUC: f64 = 1e-9;
const TOL_RHF: f64 = 1e-9;
const TOL_CC: f64 = 2e-10;
/// Excitation energies. Measured max 2.9e-10 (H2O), 7.4e-11 (C2H4); the
/// bar is ten times ferric's 1e-8 residual, the first-order eigenvalue error
/// with room for the eigenvector condition.
const TOL_OMEGA: f64 = 1e-7;

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
        .expect("<root>/crates/ferric-cc")
        .to_path_buf()
}

fn num(v: &Value, ptr: &str) -> f64 {
    v.pointer(ptr)
        .and_then(Value::as_f64)
        .unwrap_or_else(|| panic!("reference field {ptr} missing"))
}

fn run(system: &str) {
    let root = workspace_root();
    let path = root.join(ROW_DIR).join(format!("{system}_cc-pvdz.json"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "missing reference {} ({e}); regenerate with scripts/validation/gen_eom_ccsd.py",
            path.display()
        )
    });
    let r: Value = serde_json::from_str(&text).unwrap();
    let xyz = root.join(
        r.pointer("/provenance/geometry_xyz")
            .and_then(Value::as_str)
            .expect("geometry_xyz"),
    );
    let mol = Molecule::load_xyz_with_charge(xyz.to_str().unwrap(), 0, 1).unwrap();
    assert!((mol.nuclear_repulsion() - num(&r, "/nuclear_repulsion")).abs() < TOL_ENUC);
    let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    assert_eq!(obs.nbasis() as f64, num(&r, "/nao"));
    assert_eq!(dfbs.nbasis() as f64, num(&r, "/naux"));
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let scf = RhfConfig::default()
        .with_max_iter(200)
        .with_energy_conv(1e-11)
        .with_density_conv(1e-9);
    let rhf = solve_rhf(&ParallelContext::default(), &mol, &obs, op, &bounds, &scf).unwrap();
    assert!(rhf.converged && rhf.df_jk.is_none());
    let d_rhf = (rhf.energy - num(&r, "/rhf/energy")).abs();
    assert!(d_rhf < TOL_RHF, "{system}: E_RHF off by {d_rhf:e}");

    let t0 = std::time::Instant::now();
    let cc_cfg = CcConfig {
        max_iter: 200,
        energy_conv: EOM_CCSD_ENERGY_CONV,
        ..Default::default()
    };
    let cc = ccsd_closed_shell(&mol, &obs, &dfbs, op, &rhf, &cc_cfg).unwrap();
    let t_cc = t0.elapsed().as_secs_f64();
    let d_cc = (cc.correlation_energy - num(&r, "/ccsd/e_corr")).abs();
    assert!(d_cc < TOL_CC, "{system}: E_CCSD off by {d_cc:e}");

    let refs: Vec<f64> = r
        .pointer("/eom_ccsd_singlets/excitation_energies")
        .and_then(Value::as_array)
        .expect("excitation_energies")
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect();
    let t1 = std::time::Instant::now();
    let cfg = EomCcsdConfig {
        n_roots: refs.len(),
        conv_tol: 1e-8,
        max_iter: 200,
        ..Default::default()
    };
    let eom = eom_ccsd_singlets(&mol, &obs, &dfbs, op, &rhf, &cc, 0, &cfg).unwrap();
    let t_eom = t1.elapsed().as_secs_f64();
    println!(
        "{system}: CCSD {t_cc:.2} s, EOM {t_eom:.2} s ({} iterations, {} products, ladder {:?})",
        eom.iterations, eom.products, eom.ladder_route
    );
    let mut worst = 0.0f64;
    for (k, (&w, &w_ref)) in eom.excitation_energies.iter().zip(&refs).enumerate() {
        let d = (w - w_ref).abs();
        worst = worst.max(d);
        println!(
            "  root {k}: ferric {w:.10} PySCF {w_ref:.10} |d| = {d:.2e} (res {:.1e}, singles {:.3})",
            eom.residuals[k], eom.singles_weight[k]
        );
    }
    println!("{system}: max |d omega| = {worst:.3e}");
    assert!(worst < TOL_OMEGA, "{system}: max |d omega| = {worst:e}");
}

#[test]
#[ignore = "validation: EOM-CCSD"]
fn validation_eom_ccsd_h2o_ccpvdz() {
    run("h2o");
}

#[test]
#[ignore = "validation: EOM-CCSD"]
fn validation_eom_ccsd_c2h4_ccpvdz() {
    run("c2h4");
}
