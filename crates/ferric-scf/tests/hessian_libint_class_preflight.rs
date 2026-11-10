//! Per-integral-class libint2 capability preflight
//! (`ferric_integrals::capability`).
//!
//! A libint2 that did not generate an integral class (conda-forge win-64
//! 2.13.1: no 4-centre second derivatives) still constructs an engine for it,
//! and the first compute aborts the process on libint2's
//! `"null build function ptr"` assert. The linked library cannot be swapped
//! in a test, so the capability table ferric decides from is overridden
//! (`set_libint_max_am_for_test`); libint2 itself is untouched, so without the
//! check the analytic Hessian would simply run here (that is the mutation
//! check: drop the check and (a) fails, because the Hessian succeeds).
//!
//! Hypotheses. Right: (a) with the 4-centre deriv-2 entry removed the analytic
//! Hessian is refused before any SCF with a typed error naming the class and
//! the library; `Analytic` returns it; (b) `Auto` falls back to finite
//! differences; (c) installing the linked table as an override changes nothing
//! (bit-identical Hessian); (d) RI-JK needs no 4-centre second derivatives and
//! still runs analytically under that table; (e) every derivative engine
//! constructor refuses a class/order/AM the table lacks (3-centre ket shells
//! included). Artifact: the check fires on a full table (c fails), or not at
//! all (a, e fail), or for RI-JK too (d fails).
//!
//! Own test binary and one test function: the override is process-wide.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::capability::{
    is_libint_class_unavailable, linked_libint_am_table, set_libint_max_am_for_test, LibintAmTable,
};
use ferric_integrals::engine::Engine;
use ferric_integrals::ffi;
use ferric_integrals::operator::Operator;
use ferric_scf::frequencies::{
    harmonic_frequencies, FrequencyConfig, FrequencyResult, HessianMethod, HessianSource,
};
use ferric_scf::hessian::analytic_hessian_available;
use ferric_scf::rhf::RhfConfig;

const WATER: &str = "3\nH2O\nO 0.01 -0.02 0.12\nH 0.03 0.77 -0.46\nH -0.02 -0.74 -0.49\n";
const BASIS: &str = "sto-3g";

fn freq(mol: &Molecule, cfg: &RhfConfig, method: HessianMethod) -> Result<FrequencyResult, String> {
    harmonic_frequencies(
        &ParallelContext::default(),
        mol,
        BASIS,
        Operator::coulomb(),
        cfg,
        &FrequencyConfig {
            hessian: method,
            ..Default::default()
        },
    )
    .map_err(|e| e.to_string())
}

/// The linked table with one entry removed.
fn without(row: usize, deriv: usize) -> LibintAmTable {
    let mut t = linked_libint_am_table();
    t[row][deriv] = -1;
    t
}

fn assert_named_refusal(what: &str, msg: &str, class: &str) {
    eprintln!("{what}: {msg}");
    assert!(
        msg.contains("libint2 integral class unavailable"),
        "{what}: {msg}"
    );
    assert!(msg.contains(class), "{what} does not name {class:?}: {msg}");
    assert!(
        msg.contains(ferric_integrals::libint_version()),
        "{what}: {msg}"
    );
}

#[test]
fn missing_libint_classes_are_refused_before_libint_and_auto_falls_back() {
    let linked = linked_libint_am_table();
    eprintln!("linked libint2 capability table: {linked:?}");
    if linked[0][2] < 1 || linked[3][2] < 1 {
        eprintln!("SKIP: linked libint2 lacks second derivatives for p shells");
        return;
    }
    let mol = Molecule::parse_xyz(WATER, 0, 1).unwrap();
    let prep = PreparedBasis::new(&mol, &basis::bundled(BASIS).unwrap()).unwrap();
    let rhf = RhfConfig::default();
    let op = Operator::coulomb();

    // --- (c) baseline, then the linked table installed as an override -------
    set_libint_max_am_for_test(None);
    let base = freq(&mol, &rhf, HessianMethod::Auto).unwrap();
    assert_eq!(base.hessian_source, HessianSource::Analytic);
    set_libint_max_am_for_test(Some(linked));
    let same = freq(&mol, &rhf, HessianMethod::Auto).unwrap();
    assert_eq!(same.hessian_source, HessianSource::Analytic);
    assert_eq!(
        base.cartesian_hessian, same.cartesian_hessian,
        "a full capability table must not change the Hessian"
    );
    assert_eq!(base.frequencies, same.frequencies);

    // --- (a) no 4-centre second derivatives: refused, named -----------------
    set_libint_max_am_for_test(Some(without(0, 2)));
    let probe = analytic_hessian_available(&mol, &prep, op, &rhf).unwrap_err();
    assert!(is_libint_class_unavailable(&probe));
    assert_named_refusal(
        "probe",
        &probe.to_string(),
        "second-derivative 4-centre ERI",
    );
    let analytic = freq(&mol, &rhf, HessianMethod::Analytic).unwrap_err();
    assert_named_refusal("Analytic", &analytic, "second-derivative 4-centre ERI");

    // --- (b) Auto falls back to finite differences ---------------------------
    let auto = freq(&mol, &rhf, HessianMethod::Auto).unwrap();
    assert_eq!(auto.hessian_source, HessianSource::FiniteDifference);
    assert!(auto.n_gradient_evaluations > 0);
    let max_dev = base
        .frequencies
        .iter()
        .zip(&auto.frequencies)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    eprintln!("analytic vs finite-difference frequencies: max |diff| {max_dev:.3e} cm^-1");
    assert!(
        max_dev < 1.0,
        "FD fallback frequencies off by {max_dev} cm^-1"
    );

    // --- (d) RI-JK has no four-centre skeleton: still analytic --------------
    let rijk = RhfConfig {
        df_j_aux: Some("cc-pvdz-ri".into()),
        df_k_aux: Some("cc-pvdz-ri".into()),
        ..Default::default()
    };
    analytic_hessian_available(&mol, &prep, op, &rijk).unwrap();
    let fitted = freq(&mol, &rijk, HessianMethod::Analytic).unwrap();
    assert_eq!(fitted.hessian_source, HessianSource::Analytic);

    // --- (e) the constructors themselves (the net under every caller) -------
    let e = Engine::new_2e_deriv2(op, &prep, 1e-14)
        .unwrap_err()
        .to_string();
    assert_named_refusal("new_2e_deriv2", &e, "second-derivative 4-centre ERI");
    set_libint_max_am_for_test(Some(without(0, 1)));
    let e = Engine::new_2e_deriv(op, &prep, 1e-14)
        .unwrap_err()
        .to_string();
    assert_named_refusal("new_2e_deriv", &e, "first-derivative 4-centre ERI");
    let grad_fd = freq(&mol, &rhf, HessianMethod::FiniteDifference).unwrap_err();
    assert_named_refusal(
        "FD Hessian (gradients)",
        &grad_fd,
        "first-derivative 4-centre ERI",
    );
    set_libint_max_am_for_test(Some(without(3, 2)));
    let e = Engine::new_1e_deriv2(ffi::OP_OVERLAP, &prep, 1e-14)
        .unwrap_err()
        .to_string();
    assert_named_refusal("new_1e_deriv2", &e, "second-derivative one-body");
    let mut t = linked;
    t[3][1] = 0; // one-body deriv-1 only up to s: water/STO-3G has p shells
    set_libint_max_am_for_test(Some(t));
    let e = Engine::new_1e_deriv(ffi::OP_KINETIC, &prep, 1e-14)
        .unwrap_err()
        .to_string();
    assert_named_refusal("new_1e_deriv", &e, "up to l = 0");
    let aux = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let mut t = linked;
    t[4][1] = 0; // 3-centre ket (orbital) shells only up to s
    set_libint_max_am_for_test(Some(t));
    let e = Engine::new_3center_deriv(op, &prep, &aux, 1e-14)
        .unwrap_err()
        .to_string();
    assert_named_refusal("new_3center_deriv ket", &e, "(ket shells)");
    set_libint_max_am_for_test(Some(without(2, 1)));
    let e = Engine::new_2center_deriv(op, &aux, 1e-14)
        .unwrap_err()
        .to_string();
    assert_named_refusal("new_2center_deriv", &e, "first-derivative 2-centre ERI");

    set_libint_max_am_for_test(None);
    Engine::new_2center_deriv(op, &aux, 1e-14).unwrap();
}
