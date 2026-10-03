//! MEASUREMENT (not a gate): the free-atom volume on the three quadratures
//! that matter — `mbd_volume_grid` (what MBD@rsSCS's denominator uses), the
//! bounding-box lattice, and the production Becke-Lebedev grid.
//!
//!   OPENBLAS_NUM_THREADS=1 cargo test --release -p ferric-rpa \
//!       --test measure_free_atom_quadratures -- --ignored --nocapture

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_rpa::properties::{
    atomic_effective_volumes_hirshfeld, atomic_effective_volumes_hirshfeld_on_grid,
    hirshfeld_volume_grid, mbd_volume_grid,
};
use ferric_scf::properties::proatom_ground_state_mult;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::solve_uhf;

#[test]
#[ignore = "measurement: free-atom volume quadratures"]
fn measure_free_atom_volume_quadratures() {
    let ctx = ParallelContext::default();
    let op = Operator::coulomb();
    for bname in ["cc-pvdz", "def2-svp"] {
        let bs = basis::bundled(bname).expect("basis");
        for z in [1_i32, 6, 8] {
            let sym = ferric_core::elements::z_to_symbol(z).unwrap();
            let mult = proatom_ground_state_mult(z);
            let mol = Molecule::parse_xyz(&format!("1\n{sym}\n{sym} 0 0 0\n"), 0, mult).unwrap();
            let prep = PreparedBasis::new(&mol, &bs).unwrap();
            let bounds = SchwarzBounds::compute(op, &prep).unwrap();
            let cfg = RhfConfig {
                max_iter: 300,
                energy_conv: 1e-11,
                density_conv: 1e-9,
                mom_after_iter: if mult > 1 { 5 } else { 0 },
                ..Default::default()
            };
            let d = if mult > 1 {
                solve_uhf(&ctx, &mol, &prep, &bounds, &cfg)
                    .unwrap()
                    .density_total()
                    .to_owned()
            } else {
                solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg)
                    .unwrap()
                    .density_total()
                    .to_owned()
            };
            let v_becke = atomic_effective_volumes_hirshfeld(&mol, &bs, &d, None).unwrap()[0];
            let v_bbox = atomic_effective_volumes_hirshfeld_on_grid(
                &mol,
                &bs,
                &d,
                None,
                &hirshfeld_volume_grid(&mol),
            )
            .unwrap()[0];
            let v_mbd = atomic_effective_volumes_hirshfeld_on_grid(
                &mol,
                &bs,
                &d,
                None,
                &mbd_volume_grid(&mol),
            )
            .unwrap()[0];
            println!(
                "{sym}/{bname}: becke {v_becke:.9} bbox {v_bbox:.9} ({:+.2e} rel) \
                 mbd_lattice {v_mbd:.9} ({:+.2e} rel vs becke, {:+.2e} rel vs bbox)",
                (v_bbox - v_becke) / v_becke,
                (v_mbd - v_becke) / v_becke,
                (v_mbd - v_bbox) / v_bbox
            );
        }
    }
}
