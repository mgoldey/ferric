use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::gradient::{restricted_scf_gradient, unrestricted_scf_gradient};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::solve_uhf;

fn h(a: &ndarray::Array2<f64>) -> u64 {
    let mut x = 0xcbf29ce484222325u64;
    for v in a.iter() {
        for b in v.to_bits().to_le_bytes() {
            x ^= b as u64;
            x = x.wrapping_mul(0x100000001b3);
        }
    }
    x
}

#[test]
fn bitid() {
    let water = "3\n\nO 0.0 0.0 0.1173\nH 0.0 0.7572 -0.4692\nH 0.0 -0.7572 -0.4692\n";
    let nh2 = "3\n\nN 0.0 0.0 0.1289\nH 0.0 0.8033 -0.5157\nH 0.0 -0.8033 -0.5157\n";
    let bs = basis::bundled("6-31g").unwrap();
    let op = Operator::coulomb();
    let ctx = ParallelContext::default();
    for (label, xyz, mult, xc, open) in [
        ("RKS-LDA", water, 1, "LDA", false),
        ("RKS-PBE0", water, 1, "PBE0", false),
        ("UKS-PBE", nh2, 2, "PBE", true),
        ("RKS-wB97X-V", water, 1, "wB97X-V", false),
    ] {
        let mol = Molecule::parse_xyz(xyz, 0, mult).unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let cfg = RhfConfig {
            xc: Some(xc.into()),
            energy_conv: 1e-10,
            density_conv: 1e-8,
            ..Default::default()
        };
        let g = if open {
            let r = solve_uhf(&ctx, &mol, &prep, &bounds, &cfg).unwrap();
            unrestricted_scf_gradient(&mol, &prep, &bs, op, &bounds, &cfg, &r).unwrap()
        } else {
            let r = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg).unwrap();
            restricted_scf_gradient(&mol, &prep, &bs, op, &bounds, &cfg, &r).unwrap()
        };
        println!("BITID {label} {:016x}", h(&g));
    }
}
