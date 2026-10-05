//! The `RpaxScreening` validation hook must not change the shipped function.
//!
//! `run_rpax_static_polarizability` is `rpax_static_polarizability_with_screening`
//! with `RpaxScreening::StaticPdep`; asking for the A±B spectrum must not
//! perturb α either. Bit-for-bit, on a seconds-scale system (water/STO-3G/PBE,
//! the guard test's reproducer, at its healthy scissor 0.36 Ha). The bare
//! mode must differ, or the hook is not switching anything.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_gw::bse::{
    rpax_static_polarizability_with_screening, run_rpax_static_polarizability, RpaxScreening,
};
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_rpa::config::{PdepRpaConfig, QuadratureConfig, QuadratureScheme};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

const WATER_XYZ: &str =
    "3\nH2O\nO 0.0 0.0 0.117790\nH 0.0 0.755453 -0.471161\nH 0.0 -0.755453 -0.471161\n";

#[test]
fn static_pdep_hook_is_bit_identical_to_the_public_function() {
    let mol = Molecule::parse_xyz(WATER_XYZ, 0, 1).unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled("sto-3g").unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let ks = solve_rhf(
        &ParallelContext::default(),
        &mol,
        &obs,
        op,
        &bounds,
        &RhfConfig {
            xc: Some("PBE".to_string()),
            ..Default::default()
        },
    )
    .unwrap();
    assert!(ks.converged);
    let cfg = PdepRpaConfig {
        quadrature: QuadratureConfig {
            scheme: QuadratureScheme::GaussLegendre,
            n_points: 8,
            u0: 0.5,
        },
        trunc_thresh: 0.0,
        frozen_core: 0,
        ..Default::default()
    };
    let scissor = 0.36;
    let public = run_rpax_static_polarizability(&mol, &obs, &dfbs, op, &ks, &cfg, 0, scissor)
        .expect("public");
    for spectrum in [false, true] {
        let hook = rpax_static_polarizability_with_screening(
            &mol,
            &obs,
            &dfbs,
            op,
            &ks,
            &cfg,
            0,
            scissor,
            RpaxScreening::StaticPdep,
            spectrum,
        )
        .expect("hook");
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(
                    public.tensor[i][j].to_bits(),
                    hook.result.tensor[i][j].to_bits(),
                    "alpha[{i}][{j}] differs (kernel_spectrum = {spectrum})"
                );
            }
        }
        assert_eq!(public.iso.to_bits(), hook.result.iso.to_bits());
        assert_eq!(hook.min_eig_apb.is_some(), spectrum);
        assert_eq!(hook.min_eig_amb.is_some(), spectrum);
    }
    let bare = rpax_static_polarizability_with_screening(
        &mol,
        &obs,
        &dfbs,
        op,
        &ks,
        &cfg,
        0,
        scissor,
        RpaxScreening::Bare,
        false,
    )
    .expect("bare");
    let d = (bare.result.iso - public.iso).abs() / public.iso;
    eprintln!(
        "screened iso {:.8} bare iso {:.8} rel {d:.2e}",
        public.iso, bare.result.iso
    );
    assert!(d > 1e-3, "Bare mode did not change alpha (rel {d:.2e})");
}
