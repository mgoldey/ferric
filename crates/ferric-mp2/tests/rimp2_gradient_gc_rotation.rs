//! The RI-MP2 analytic gradient on a reference SCF run with the contraction
//! rotation (`ferric_scf::direct_jk::gc_rotation`): its Z-vector response
//! J/K (`ResponseJk` on the SCF's bounds) and its separable two-electron
//! derivative terms (`twoelectron_gradient_bilinear`, `with_response`) run on
//! the rotated shells and agree with the unrotated gradient within 1e-10
//! Eh/bohr (water / cc-pVDZ, the measured difference is printed).

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::gradient::rimp2_gradient_analytical;
use ferric_mp2::rimp2::RiMp2Config;
use ferric_scf::direct_jk::GcRotate;
use ferric_scf::incore_jk::EriMode;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ndarray::Array2;

fn gradient(mode: GcRotate) -> Array2<f64> {
    let mol = Molecule::parse_xyz(
        "3\nwater\nO 0 0 0.1173\nH 0 0.7572 -0.4692\nH 0 -0.7572 -0.4692\n",
        0,
        1,
    )
    .unwrap();
    let prep = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let mut c = RhfConfig::default()
        .with_energy_conv(1e-11)
        .with_density_conv(1e-9)
        .with_gc_rotation(mode);
    c.eri = EriMode::Direct;
    let rhf = solve_rhf(&ParallelContext::default(), &mol, &prep, op, &bounds, &c).unwrap();
    assert!(rhf.converged);
    rimp2_gradient_analytical(
        &mol,
        &prep,
        &dfbs,
        op,
        &bounds,
        &rhf,
        &RiMp2Config::default(),
        None,
    )
    .unwrap()
}

#[test]
fn rimp2_gradient_rotated_matches_unrotated() {
    let g0 = gradient(GcRotate::Off);
    let g1 = gradient(GcRotate::Auto);
    let e = (&g1 - &g0).iter().fold(0.0f64, |m, v| {
        if v.is_nan() {
            f64::INFINITY
        } else {
            m.max(v.abs())
        }
    });
    eprintln!("water/cc-pvdz RI-MP2 gradient: max|dG| {e:.2e} Eh/bohr");
    assert!(e <= 1e-10, "{e:e}");
    assert!(
        g0.iter().zip(&g1).any(|(a, b)| a.to_bits() != b.to_bits()),
        "the gradient did not rotate"
    );
}
