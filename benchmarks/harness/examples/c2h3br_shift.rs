//! Find the level shift (if any) that converges C2H3Br RHF at aug-cc-pVDZ.
//! PySCF target: -2649.796875. Default DIIS oscillates (~err 3, E ~ -2477).
use ferric_core::basis; use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis; use ferric_integrals::operator::Operator;
use ferric_scf::rhf::{solve_rhf, RhfConfig}; use ferric_scf::screening::SchwarzBounds;
fn main() {
    let ctx = ParallelContext::default(); let op = Operator::coulomb();
    let bs = basis::bundled("aug-cc-pvdz").unwrap();
    let xyz = "6\nmol\nC 0.000000 0.000000 0.000000\nC 0.000000 0.000000 1.325600\nH -0.895976 0.000000 -0.602298\nH -0.894897 0.000000 1.927173\nH 0.908386 0.000000 -0.581003\nBr 1.357668 0.000000 2.194533\n";
    let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    for (ls, mi) in [(0.5, 200), (1.0, 200), (2.0, 300), (0.5, 500)] {
        let cfg = RhfConfig { max_iter: mi, level_shift: ls, ..Default::default() };
        match solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg) {
            Ok(r) => println!("ls={ls} maxit={mi}: conv={} iters={} E={:.6} (Δ vs PySCF {:.4})",
                r.converged, r.iterations, r.energy, r.energy - (-2649.796875)),
            Err(e) => println!("ls={ls} maxit={mi}: ERR {e:?}"),
        }
    }
}
