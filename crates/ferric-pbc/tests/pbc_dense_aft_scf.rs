//! Stage-1 PBC step 5: test-only `DenseAftEri` J/K (pure AFT, nao⁴, hard
//! size cap) with the Madelung `exxdiv` shift, driven through
//! `ferric_scf::rhf::solve_rhf_injected` with the step-4 `PeriodicHcore`.
//!
//! References (PySCF 2.13 AFTDF, mesh 61³, pinned in
//! `reference/pbc/test_prototype.py` and `run_triclinic_p.py`):
//! * H2/STO-3G, cubic a = 4: E = −1.658327061049 (ewald),
//!   −0.949002691179 (none); ε_ewald = [−0.84743566, 1.23177667],
//!   ε_none = [−0.13811129, 1.23177667].
//! * Triclinic 4H, STO-3G s + Cartesian p(0.8): E_none = −1.0784116041
//!   (PySCF and the prototype agree to 7.8e-13).
//!
//! What is independent of what: `h` comes from the Ewald-split `PeriodicHcore`
//! (libint2 SR + analytic LR), J/K from the pure-AFT tensor — the same
//! convention the PySCF oracle uses, but a different code path for h.
//! The triclinic cell has p functions, so it reaches the pair FT's t > 0
//! (−iG)^t branch that s-only H2 cannot see (FINDINGS "Iteration 1").
//!
//! Mutation plan:
//! * drop the Madelung term in `DenseAftK`: the ewald energies (H2, and the
//!   triclinic `E_ewald − E_none = −v_M N_e/2` identity) fail.
//! * K contracted as `I[μν,λσ]` (= J): every energy fails.
//! * drop the factor 2 of the half G sphere: the tensor-vs-prototype check
//!   and every energy fail.
//! * remove the size cap: `dense_aft_refuses_an_oversize_tensor` fails.
//! * `ExxDiv` parse defaulting instead of erroring: the strict-parse test.

mod common;

use common::*;
use ferric_pbc::dense_aft::{
    DenseAftEri, ExxDiv, DEFAULT_DENSE_AFT_MAX_BYTES, DEFAULT_DENSE_AFT_PRECISION,
};
use ferric_pbc::ewald::madelung_constant;
use ferric_pbc::hcore::{periodic_hcore, PeriodicHcoreConfig};

const H2_E_EWALD: f64 = -1.658327061049;
const H2_E_NONE: f64 = -0.949002691179;
const H2_EPS_EWALD: [f64; 2] = [-0.84743566, 1.23177667];
const H2_EPS_NONE: [f64; 2] = [-0.13811129, 1.23177667];
const TRI_E_NONE: f64 = -1.0784116041;

/// `pbc_gamma.build_integrals(..., None)["I"]` for H2/STO-3G a = 4, raveled
/// (μ, ν, λ, σ) — the same flattening as `DenseAftEri::eri()`.
const H2_ERI_REF: [f64; 16] = [
    0.27177525294826854,
    0.13372637837762433,
    0.13372637837762433,
    0.0815723334613884,
    0.13372637837762433,
    0.102823673190457,
    0.102823673190457,
    0.13372637837762424,
    0.13372637837762433,
    0.102823673190457,
    0.102823673190457,
    0.13372637837762424,
    0.0815723334613884,
    0.13372637837762424,
    0.13372637837762424,
    0.2717752529482681,
];

/// ω for the nuclear-attraction split: not 1 (a 1/ω vs 1/ω² G = 0 slip
/// would be invisible there).
const OMEGA: f64 = 0.8;

#[test]
fn dense_aft_eri_matches_prototype_h2() {
    let cell = h2_cell(4.0);
    let prep = prep_for(&cell, &pyscf_sto3g_h());
    let hc = periodic_hcore(&cell, &prep, &PeriodicHcoreConfig::with_omega(OMEGA)).unwrap();
    let eri = DenseAftEri::build(
        &cell,
        &prep,
        &hc.s,
        ExxDiv::None,
        DEFAULT_DENSE_AFT_PRECISION,
        DEFAULT_DENSE_AFT_MAX_BYTES,
    )
    .unwrap();
    let got: Vec<f64> = eri.eri().iter().copied().collect();
    let worst = got.iter().zip(&H2_ERI_REF).fold(0.0_f64, |m, (a, b)| {
        let d = (a - b).abs();
        assert!(d.is_finite(), "non-finite ERI difference: {a} vs {b}");
        m.max(d)
    });
    eprintln!(
        "H2 a=4 dense AFT ERI vs prototype: {worst:.2e} ({} half-G)",
        eri.n_g_half()
    );
    assert!(worst < 1e-12, "{worst:.3e}");
}

#[test]
fn h2_sto3g_a4_matches_pinned_pyscf_aftdf() {
    let cell = h2_cell(4.0);
    let prep = prep_for(&cell, &pyscf_sto3g_h());
    let hc = periodic_hcore(&cell, &prep, &PeriodicHcoreConfig::with_omega(OMEGA)).unwrap();
    let base = DenseAftEri::build(
        &cell,
        &prep,
        &hc.s,
        ExxDiv::None,
        DEFAULT_DENSE_AFT_PRECISION,
        DEFAULT_DENSE_AFT_MAX_BYTES,
    )
    .unwrap();
    for (exx, e_ref, eps_ref) in [
        (ExxDiv::None, H2_E_NONE, H2_EPS_NONE),
        (ExxDiv::Ewald, H2_E_EWALD, H2_EPS_EWALD),
    ] {
        let eri = base.clone().with_exxdiv(&cell, exx).unwrap();
        let r = gamma_rhf(&cell, &prep, &hc, &eri);
        let eps = r.eps_r();
        eprintln!(
            "{exx:?}: E = {:.12} (ref {e_ref:.12}, dE {:.2e}), eps = {:?}, v_M = {}",
            r.energy,
            r.energy - e_ref,
            eps,
            eri.madelung()
        );
        assert!((r.energy - e_ref).abs() < 1e-9, "{exx:?}: E {}", r.energy);
        for k in 0..2 {
            assert!(
                (eps[k] - eps_ref[k]).abs() < 1e-7,
                "{exx:?}: eps[{k}] {}",
                eps[k]
            );
        }
    }
}

#[test]
fn triclinic_4h_sp_matches_pinned_pyscf_aftdf() {
    let cell = triclinic_cell();
    let prep = prep_for(&cell, &sp_basis_h());
    let hc = periodic_hcore(&cell, &prep, &PeriodicHcoreConfig::with_omega(OMEGA)).unwrap();
    let base = DenseAftEri::build(
        &cell,
        &prep,
        &hc.s,
        ExxDiv::None,
        DEFAULT_DENSE_AFT_PRECISION,
        DEFAULT_DENSE_AFT_MAX_BYTES,
    )
    .unwrap();
    let mut energies = Vec::new();
    for exx in [ExxDiv::None, ExxDiv::Ewald] {
        let eri = base.clone().with_exxdiv(&cell, exx).unwrap();
        let r = gamma_rhf(&cell, &prep, &hc, &eri);
        eprintln!(
            "triclinic {exx:?}: E = {:.12}, {} iterations, {} half-G",
            r.energy,
            r.iterations,
            eri.n_g_half()
        );
        energies.push(r.energy);
    }
    assert!(
        (energies[0] - TRI_E_NONE).abs() < 1e-9,
        "E_none {} vs {TRI_E_NONE}",
        energies[0]
    );
    // At Gamma, v_M S D S acts as −v_M on the occupied space without
    // changing D: E_ewald − E_none = −v_M · N_e/2 (N_e = 4).
    let vm = madelung_constant(&cell).unwrap();
    let shift = energies[1] - energies[0];
    assert!(
        (shift + 2.0 * vm).abs() < 1e-9,
        "shift {shift} vs -2 v_M {}",
        -2.0 * vm
    );
}

#[test]
fn dense_aft_refuses_an_oversize_tensor() {
    let cell = h2_cell(4.0);
    let prep = prep_for(&cell, &pyscf_sto3g_h());
    let hc = periodic_hcore(&cell, &prep, &PeriodicHcoreConfig::with_omega(OMEGA)).unwrap();
    let need = 8 * prep.nbasis().pow(4);
    let err = DenseAftEri::build(&cell, &prep, &hc.s, ExxDiv::Ewald, 1e-10, need - 1)
        .expect_err("cap must be enforced");
    let msg = err.to_string();
    assert!(
        msg.contains(&format!("{need} bytes")) && msg.contains("cap"),
        "{msg}"
    );
    assert!(DenseAftEri::build(&cell, &prep, &hc.s, ExxDiv::Ewald, 1e-10, need).is_ok());
}

#[test]
fn exxdiv_parse_is_strict() {
    assert_eq!(ExxDiv::parse_config_str("ewald").unwrap(), ExxDiv::Ewald);
    assert_eq!(ExxDiv::parse_config_str("None").unwrap(), ExxDiv::None);
    for bad in ["vcut_sph", "", "ewald "] {
        assert!(ExxDiv::parse_config_str(bad).is_err(), "{bad:?} accepted");
    }
}
