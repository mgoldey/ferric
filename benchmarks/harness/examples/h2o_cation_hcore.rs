//! Dump ferric's hcore eigenvalues for H2O+ to compare with PySCF.
use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::oneelectron;
use ndarray_linalg::Eigh;

fn main() {
    let xyz = "3\nH2O+\nO 0.0 0.0 0.117790\nH 0.0 0.755453 -0.471161\nH 0.0 -0.755453 -0.471161\n";
    let cation = Molecule::parse_xyz(xyz, 1, 2).unwrap();
    let bs = basis::bundled("cc-pvdz").unwrap();
    let prep = PreparedBasis::new(&cation, &bs).unwrap();
    let s = oneelectron::overlap(&prep);
    let h = oneelectron::hcore(&prep);
    let n = prep.nbasis();
    let (s_evals, s_evecs) = s.eigh(ndarray_linalg::UPLO::Upper).unwrap();
    let mut u = s_evecs.clone();
    for i in 0..n { let sc = 1.0/s_evals[i].sqrt(); for mu in 0..n { u[(mu,i)] *= sc; } }
    let s_inv_sqrt = u.dot(&s_evecs.t());
    let h_prime = s_inv_sqrt.dot(&h).dot(&s_inv_sqrt);
    let (eps, _) = h_prime.eigh(ndarray_linalg::UPLO::Upper).unwrap();
    println!("ferric hcore eigenvalues (Hartree):");
    for i in 0..10.min(eps.len()) { println!("  {}: {:+.6}", i, eps[i]); }
    println!("\nPySCF reference:");
    println!("   0: -33.056104");
    println!("   1: -8.937147");
    println!("   2: -8.710025");
    println!("   3: -8.528492   ← β HOMO");
    println!("   4: -8.519780   ← α HOMO / β hole");
    println!("\nGap between idx 3 and 4: 8 mHa (near-degenerate — small numerical");
    println!("differences could swap them and pick a different hole orbital).");
}
