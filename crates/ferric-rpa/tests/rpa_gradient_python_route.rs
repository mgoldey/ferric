//! The Python route `run_pdep_rpa(with_gradient=True)` against the function the
//! CLI's pdep-rpa `task = "optimize"` calls at the starting geometry.
//!
//! The binding builds an exact four-centre RHF (`density_conv` 1e-9, default
//! everything else) and calls `rpa_analytic_gradient_on` on it; the CLI's
//! optimize step calls `total_rpa_gradient_by(.., RpaGradientMethod::Analytic,
//! ..)`, which solves the same RHF and calls the same function. This test
//! pins that the two are bit-for-bit equal, and, when
//! `FERRIC_RPA_GRAD_DUMP` names a file, writes the gradient bits as JSON (run it with
//! `FERRIC_BLAS_THREADS=1`, the setting that is bit-identical across thread counts) so
//! `crates/ferric-python/tests/test_pdep_rpa_gradient.py` can compare the
//! Python result against it (that comparison skips when the file is absent).
//!
//! Mutation: replacing the binding's `RingOrder::Full` call by one on a
//! looser SCF (default `density_conv`) breaks the equality asserted here.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_rpa::config::{QuadratureConfig, QuadratureScheme};
use ferric_rpa::gradient::{total_rpa_gradient_by, RpaGradientMethod};
use ferric_rpa::gradient_analytic::{rpa_analytic_gradient_on, RingOrder};
use ferric_rpa::PdepRpaConfig;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::warm_start::ScfWarmStart;

const XYZ: &str = "3\nstrained water\nO 0 0 0\nH 0 0.9 0.1\nH 0 -0.3 0.85\n";

#[test]
fn python_route_equals_the_cli_optimize_gradient_bit_for_bit() {
    let mol = Molecule::parse_xyz(XYZ, 0, 1).unwrap();
    let obs_bs = basis::bundled("sto-3g").unwrap();
    let aux_bs = basis::bundled("cc-pvdz-ri").unwrap();
    let op = Operator::coulomb();
    let cfg = PdepRpaConfig {
        frozen_core: 0,
        trunc_thresh: 0.0,
        quadrature: QuadratureConfig {
            scheme: QuadratureScheme::GaussLegendre,
            n_points: 4,
            u0: 0.5,
        },
        ..Default::default()
    };

    // The CLI optimize path at the starting geometry.
    let (e_cli, g_cli) = total_rpa_gradient_by(
        &mol,
        &obs_bs,
        &aux_bs,
        op,
        &cfg,
        RpaGradientMethod::Analytic,
        &mut ScfWarmStart::new(false),
    )
    .unwrap();

    // The binding: its own RHF at density_conv 1e-9, then `_on`.
    let ctx = ParallelContext::default();
    let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
    let dfbs = PreparedBasis::new(&mol, &aux_bs).unwrap();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let rhf_cfg = RhfConfig::default().with_density_conv(1e-9);
    let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &rhf_cfg).unwrap();
    assert!(rhf.converged);
    let r = rpa_analytic_gradient_on(&mol, &obs, &dfbs, op, &bounds, &rhf, &cfg, RingOrder::Full)
        .unwrap();

    assert_eq!(e_cli.to_bits(), r.energy.to_bits());
    for (a, b) in g_cli.iter().zip(r.gradient.iter()) {
        assert_eq!(a.to_bits(), b.to_bits(), "{a:e} vs {b:e}");
    }

    if let Ok(path) = std::env::var("FERRIC_RPA_GRAD_DUMP") {
        let bits: Vec<String> = g_cli
            .iter()
            .map(|x| format!("{:016x}", x.to_bits()))
            .collect();
        std::fs::write(path, format!("{{\"bits\": {bits:?}}}")).unwrap();
    }
}
