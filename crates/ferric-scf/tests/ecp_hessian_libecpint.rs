//! The analytic ECP Hessian on the libecpint backend (issue #406).
//!
//! The default (and validated) ECP backend is ferric's `ecp_quad`; libecpint
//! (`FERRIC_ECP_BACKEND=libecpint`) is the cross-check. The skeleton stencil
//! differentiates whichever backend's first-derivative kernel the gradient
//! uses, and divides that backend's integral noise by the step.
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 cargo test -p ferric-scf --release --test ecp_hessian_libecpint
//! ```
//!
//! # Finding, and why the Hessian is REFUSED on this backend
//!
//! Measured 2026-10-15 (one-time), BEFORE the refusal existed, so these
//! full-Hessian numbers cannot be re-run at this commit (the entry points
//! refuse libecpint; only `ecp_skeleton_hessian` still runs on it):
//!
//! * HOI/def2-SVP, full analytic Hessian vs central FD of the libecpint
//!   analytic gradient: 2.33e-7 (quadrature 2.33e-7); libecpint vs quadrature
//!   Hessians 1.4e-8; PySCF's analytic ECP Hessian (`validation_ecp_hessian.rs`
//!   with `FERRIC_ECP_BACKEND=libecpint`): HI, CH3I and CH2I passed every bar.
//! * Ag2/aug-cc-pVDZ-PP FAILED the same validation: the ECP skeleton misses
//!   PySCF's fixed-density FD of its ECP gradient term by 2.76e-6 (bar 5e-7;
//!   quadrature 3.4e-10). Step sweep of the skeleton, libecpint: 4.2e-6,
//!   2.2e-6, 8.9e-7 at h = 1e-3, 2e-3, 4e-3 (the stencil error changes sign
//!   between steps; noise-dominated, roughly 1/h at small h, not strictly
//!   proportional: 4e-3 to 8e-3 falls 31x), floor 2.8e-8 from h >= 8e-3. The
//!   floor is a REAL second-derivative bias of libecpint (independent review:
//!   xx component noise-free with scatter 8.7e-12, steady bias of about
//!   -1.5e-8 Ha/Bohr² at every h), ~18x under the bar; the noise part
//!   (scatter 3.4e-9 about a cubic fit in zz) is what fails it.
//! * libecpint's Ag2 GRADIENT misses PySCF by 8.0e-8 Ha/Bohr (quadrature
//!   5.3e-10), inside the existing libecpint gradient floor (1.3e-7,
//!   `validation_ecp_gradient.rs`). NO committed gradient anchor covers
//!   libecpint for Ag or any aug-cc-pVDZ-PP element (only I and Sn on def2),
//!   and the finite-difference fallback under `FERRIC_ECP_BACKEND=libecpint`
//!   is NOT validated on Ag2 (estimated ~5e-7, at the bar; unmeasured).
//!
//! The artifact hypothesis (stencil truncation) predicts an error that FALLS
//! with h as h^4; a sign-changing, roughly 1/h error is first-derivative
//! integral noise, which the FD-of-the-gradient anchor cannot see (it
//! differences the same noisy gradient). Rather than loosen the bar, the
//! availability probe refuses an ECP system on this backend, so `Auto`
//! frequencies use finite differences there and `Analytic` is an error.
//! Running analytic-Hessian tests with `FERRIC_ECP_BACKEND=libecpint` set
//! globally therefore fails by design.
//!
//! ONE test function: the backend is a process-wide environment variable read
//! on every ECP call, so a second concurrent test would race on it.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::hessian_ecp::ecp_skeleton_hessian;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

const HOI: &str = "3\nHOI tilted\nH 0.30 -0.40 0.20\nO 0.00 0.10 0.95\nI 0.45 0.20 2.80\n";
/// libecpint vs quadrature skeleton on one density (Ha/Bohr²); HOI is not a
/// noise-limited case (see the module doc).
const TOL_SKELETON_BACKENDS: f64 = 1e-6;

fn probe(backend: &str) -> Result<(), ferric_core::FerricError> {
    std::env::set_var("FERRIC_ECP_BACKEND", backend);
    let bs = basis::bundled("def2-svp").unwrap();
    let mut mol = Molecule::parse_xyz(HOI, 0, 1).unwrap();
    mol.apply_ecp(&bs);
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let r = ferric_scf::hessian::analytic_hessian_available(
        &mol,
        &prep,
        Operator::coulomb(),
        &RhfConfig::default(),
    );
    std::env::remove_var("FERRIC_ECP_BACKEND");
    r
}

#[test]
fn libecpint_backend_is_refused_by_the_probe_and_its_skeleton_agrees_on_hoi() {
    // Typed refusal on libecpint, before any SCF and whatever the libint.
    let err = probe("libecpint").expect_err("libecpint must be refused by the probe");
    eprintln!("probe: {err}");
    assert!(err.to_string().contains("libecpint"), "{err}");
    if ferric_integrals::engine::libint_max_deriv_order() >= 2 {
        probe("quadrature").expect("the default backend is accepted");
    }

    // The skeleton itself still runs on the active backend (diagnostic).
    let bs = basis::bundled("def2-svp").unwrap();
    let mut mol = Molecule::parse_xyz(HOI, 0, 1).unwrap();
    mol.apply_ecp(&bs);
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let cfg = RhfConfig {
        max_iter: 300,
        energy_conv: 1e-12,
        density_conv: 1e-10,
        df_j_aux: Some(String::new()),
        df_k_aux: Some(String::new()),
        ..Default::default()
    };
    let scf = solve_rhf(&ParallelContext::default(), &mol, &prep, op, &bounds, &cfg).unwrap();
    assert!(scf.converged);
    let d = scf.density_r();

    std::env::set_var("FERRIC_ECP_BACKEND", "quadrature");
    let quad = ecp_skeleton_hessian(&mol, &bs, d).unwrap();
    std::env::set_var("FERRIC_ECP_BACKEND", "libecpint");
    let lib = ecp_skeleton_hessian(&mol, &bs, d).unwrap();
    std::env::remove_var("FERRIC_ECP_BACKEND");

    let scale = quad.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let diff = quad
        .iter()
        .zip(&lib)
        .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
    eprintln!("HOI skeleton, libecpint vs quadrature: {diff:.3e} (max element {scale:.3e})");
    assert!(scale > 1e-3, "vacuous skeleton");
    assert!(diff < TOL_SKELETON_BACKENDS, "{diff:.3e}");
}
