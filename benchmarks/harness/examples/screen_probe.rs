use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_rpa::config::{Chi0Sparsity, QuadratureConfig, QuadratureScheme};
use ferric_rpa::{run_pdep_rpa, screen, PdepRpaConfig};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use std::time::Instant;

fn main() {
    let ctx = ParallelContext::default();
    let mol = Molecule::load_xyz("testdata/molecules/benzene.xyz").unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default()).unwrap();

    let cfg = PdepRpaConfig {
        quadrature: QuadratureConfig {
            scheme: QuadratureScheme::GaussLegendre, n_points: 40, u0: 0.5,
        },
        frozen_core: 6,
        trunc_thresh: 0.0,
        eigensolver_conv_thresh: 1e-10,
        ..Default::default()
    };

    let t0 = Instant::now();
    let r_dense = run_pdep_rpa(&mol, &obs, &dfbs, op, &rhf, &cfg).unwrap();
    let dt_dense = t0.elapsed().as_secs_f64();
    println!("dense E_c={:.10}  t={:.2}s", r_dense.e_rpa, dt_dense);

    for &thresh in &[1e-3, 5e-3, 1e-2, 2e-2, 3e-2, 5e-2] {
        let (sb, _) = screen::build_screened_bov_boys(&mol, &obs, &dfbs, op, &rhf, 6, thresh).unwrap();
        let total = sb.n_occ_loc * sb.naux;
        let mut cfg_s = cfg.clone();
        cfg_s.chi0_sparsity = Chi0Sparsity::BoysScreened { thresh };
        let t0 = Instant::now();
        let r_scr = run_pdep_rpa(&mol, &obs, &dfbs, op, &rhf, &cfg_s).unwrap();
        let dt_scr = t0.elapsed().as_secs_f64();
        println!("thresh={:.0e}: retained {}/{} ({:.2}× red) E_c={:.10} ΔE={:.2e} t={:.2}s",
            thresh, sb.total_retained, total,
            total as f64 / sb.total_retained.max(1) as f64,
            r_scr.e_rpa,
            (r_scr.e_rpa - r_dense.e_rpa).abs(),
            dt_scr,
        );
    }
}
