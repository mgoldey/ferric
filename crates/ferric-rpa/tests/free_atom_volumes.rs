//! Compute free-atom Becke effective volumes at several basis sets.
//!
//! For a single isolated atom the Becke partition weight w^A(r) = 1 everywhere
//! (no neighbors), so:
//!   v_free = ∫ ρ_atom(r) |r - R_A|³ dr
//! This is exactly what the TS model uses as the denominator vol_free.
//! Running this with the same basis as the molecular calculation gives
//! a self-consistent volume ratio v_mol/v_free.
//!
//!   OPENBLAS_NUM_THREADS=1 cargo test -p ferric-rpa --test free_atom_volumes \
//!     --release -- --nocapture

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_rpa::properties::atomic_effective_volumes_becke;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::uhf::solve_uhf;
use ferric_scf::screening::SchwarzBounds;

struct AtomSpec {
    symbol: &'static str,
    z: usize,
    mult: usize,  // 1=singlet(RHF), >1=open-shell(UHF)
    xyz: &'static str,
}

fn vol_for_atom(spec: &AtomSpec, obs_name: &str) -> f64 {
    let mol = Molecule::parse_xyz(spec.xyz, 0, spec.mult).unwrap();
    let obs_bs = basis::bundled(obs_name).unwrap();
    let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
    let op = Operator::coulomb();
    let ctx = ParallelContext::default();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let mut cfg = RhfConfig::default();
    cfg.mom_after_iter = if spec.mult > 1 { 5 } else { 0 };

    let density = if spec.mult > 1 {
        let rhf = solve_uhf(&ctx, &mol, &obs, op, &bounds, &cfg).unwrap();
        rhf.density_total().to_owned()
    } else {
        let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &cfg).unwrap();
        rhf.density_r().to_owned()
    };

    let vols = atomic_effective_volumes_becke(&mol, &obs, &obs_bs, &density).unwrap();
    vols[0]
}

#[test]
fn free_atom_becke_volumes() {
    let atoms = vec![
        AtomSpec { symbol: "H",  z: 1,  mult: 2, xyz: "1\nH\nH 0 0 0\n" },
        AtomSpec { symbol: "He", z: 2,  mult: 1, xyz: "1\nHe\nHe 0 0 0\n" },
        AtomSpec { symbol: "C",  z: 6,  mult: 1, xyz: "1\nC\nC 0 0 0\n" },   // RHF singlet
        AtomSpec { symbol: "N",  z: 7,  mult: 4, xyz: "1\nN\nN 0 0 0\n" },   // UHF quartet
        AtomSpec { symbol: "O",  z: 8,  mult: 1, xyz: "1\nO\nO 0 0 0\n" },   // RHF singlet
        AtomSpec { symbol: "F",  z: 9,  mult: 2, xyz: "1\nF\nF 0 0 0\n" },   // UHF doublet
        AtomSpec { symbol: "Ne", z: 10, mult: 1, xyz: "1\nNe\nNe 0 0 0\n" },
    ];

    // vol_free values currently in the code (to compare against)
    let current = [9.149, 4.711, 34.054, 25.097, 19.750, 15.746, 12.443];

    let bases = ["cc-pvdz", "aug-cc-pvdz", "aug-cc-pvtz"];

    println!("\nFree-atom Becke effective volumes v_free = ∫ ρ(r) |r|³ dr (a.u.)");
    println!("(For a single atom, Becke w=1 everywhere, so this is the TS vol_free)");
    println!();

    print!("{:>4}  {:>8}", "Atom", "table");
    for b in &bases { print!("  {:>12}", b); }
    println!();
    println!("{}", "-".repeat(4 + 8 + 2 + bases.len() * 14 + 4));

    for (i, spec) in atoms.iter().enumerate() {
        print!("{:>4}  {:>8.3}", spec.symbol, current[i]);
        for &b in &bases {
            let v = vol_for_atom(spec, b);
            print!("  {:>12.3}", v);
        }
        println!();
    }

    println!();
    println!("'table' = values currently in free_atom_ref.rs (believed to be PBE free-atom)");
    println!("Ferric computes RHF/UHF free-atom volumes — compare to validate the table.");
}
