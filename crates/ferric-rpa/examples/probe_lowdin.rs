use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_rpa::properties::lowdin_charges;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

fn run(label: &str, xyz: &str, basis_name: &str) {
    let ctx = ParallelContext::new();
    let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();

    let bs = basis::bundled(basis_name).unwrap();
    let obs = PreparedBasis::new(&mol, &bs).unwrap();

    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let rhf = solve_rhf(
        &ctx,
        &mol,
        &obs,
        op,
        &bounds,
        &RhfConfig { energy_conv: 1e-12, density_conv: 1e-11, ..Default::default() },
    )
    .unwrap();

    let charges = lowdin_charges(&mol, &obs, rhf.density_r()).unwrap();

    println!("=== {label} ({basis_name}) ===");
    println!("scf_energy = {:.12}", rhf.energy);
    println!("lowdin_charges = {:?}", charges);
    println!("sum = {:.3e}", charges.iter().sum::<f64>());
}

fn main() {
    // Same geometries as scripts/gen_pyscf_lowdin_ref.py.
    run(
        "water",
        "3\nwater\nO 0.000000 0.000000 0.117790\nH 0.000000 0.755453 -0.471161\nH 0.000000 -0.755453 -0.471161\n",
        "cc-pvdz",
    );

    run(
        "methane",
        "5\nmethane\nC 0.000000 0.000000 0.000000\nH 0.629118 0.629118 0.629118\nH -0.629118 -0.629118 0.629118\nH -0.629118 0.629118 -0.629118\nH 0.629118 -0.629118 -0.629118\n",
        "cc-pvdz",
    );

    // H2 at 0.74083 A (=1.4 Bohr), matching gen_pyscf_rpa_props.py / gen_pyscf_lowdin_ref.py.
    run(
        "h2",
        "2\nh2\nH 0.000000 0.000000 0.000000\nH 0.000000 0.000000 0.740830\n",
        "cc-pvdz",
    );
}
