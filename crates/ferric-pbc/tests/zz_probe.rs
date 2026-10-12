mod common;
use common::*;
use ferric_pbc::dense_aft::ExxDiv;
use ferric_pbc::dft::PeriodicGridConfig;
use ferric_pbc::hcore::PeriodicHcoreConfig;
use ferric_pbc::kdft::{solve_krks, KRksConfig};
use ferric_pbc::kpts::KPointMesh;
use ferric_pbc::lattice::Cell;

fn energy(pos: &[[f64; 3]], f: &str, nr: usize, na: usize) -> f64 {
    let cell = Cell::new(hydrogens(pos), cubic(4.0)).unwrap();
    let bs = pyscf_sto3g_h();
    let prep = prep_for(&cell, &bs);
    let mesh = KPointMesh::gamma_centred(&cell, [1, 1, 3]).unwrap();
    let mut cfg = KRksConfig::new(&cell, ExxDiv::None, f);
    cfg.krhf.hcore = PeriodicHcoreConfig::with_omega(0.8);
    cfg.krhf.scf.energy_conv = 1e-13;
    cfg.krhf.scf.grad_conv = 1e-10;
    cfg.grid = PeriodicGridConfig::with_size(nr, na);
    solve_krks(&cell, &prep, None, &mesh, &cfg).unwrap().scf.energy
}

#[test]
#[ignore]
fn probe() {
    let t = std::time::Instant::now();
    let base = H2_ATOMS.to_vec();
    for (nr, na) in [(20usize, 50usize), (40, 110)] {
        let e0 = energy(&base, "PBE", nr, na);
        eprintln!("grid ({nr},{na}) e0={e0:.12} t={:?}", t.elapsed());
        for h in [1e-5, 2.5e-5, 1e-4, 3e-4, 1e-3] {
            let mut p = base.clone();
            p[1][2] += h;
            let ep = energy(&p, "PBE", nr, na);
            p[1][2] -= 2.0 * h;
            let em = energy(&p, "PBE", nr, na);
            eprintln!(
                "  h={h:e} fd={:.10} d2={:.3e} (d2/h^2={:.4})",
                (ep - em) / (2.0 * h),
                ep + em - 2.0 * e0,
                (ep + em - 2.0 * e0) / (h * h)
            );
        }
    }
}
