//! Surface ESP is a descriptor, not an element label.
use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::properties::{esp_at_atoms, esp_on_surface};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

#[test]
fn surface_esp_is_not_a_nuclear_charge_readout() {
    let xyz = "3\nwater\nO 0.000 0.000 0.117\nH 0.000 0.757 -0.469\nH 0.000 -0.757 -0.469\n";
    let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
    let bs = basis::bundled("cc-pvdz").unwrap();
    let obs = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let ctx = ParallelContext::default();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default()).unwrap();
    let d = rhf.density_r().to_owned();

    let (pts, esp) = esp_on_surface(&mol, &obs, &d, 1.4, 110).unwrap();
    assert!(
        pts.len() == esp.len() && pts.len() > 50,
        "got {} points",
        pts.len()
    );

    // At the nuclei the potential is dominated by the local nuclear cusp and
    // runs to tens of Hartree; on the surface it is the small, chemically
    // meaningful field a partner molecule would feel.
    let at_nuclei = esp_at_atoms(&mol, &obs, &d).unwrap();
    let worst_nuc = at_nuclei.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let worst_surf = esp.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    assert!(
        worst_surf < 0.2 && worst_nuc > 1.0,
        "surface |ESP| max {worst_surf:.3} should be small; nuclear |ESP| max {worst_nuc:.3} large"
    );

    // Water's surface must carry both signs: negative over the oxygen lone
    // pairs, positive over the hydrogens. A single-signed field would mean the
    // sampling is stuck on one face.
    assert!(
        esp.iter().any(|&v| v > 0.0) && esp.iter().any(|&v| v < 0.0),
        "surface ESP should change sign around water; got range {:.3}..{:.3}",
        esp.iter().cloned().fold(f64::MAX, f64::min),
        esp.iter().cloned().fold(f64::MIN, f64::max)
    );
}

fn rhf_density(xyz: &str, basis_name: &str) -> (Molecule, PreparedBasis, ndarray::Array2<f64>) {
    let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
    let bs = basis::bundled(basis_name).unwrap();
    let obs = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let ctx = ParallelContext::default();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default()).unwrap();
    let d = rhf.density_r().to_owned();
    (mol, obs, d)
}

#[test]
fn surface_counts_buried_points_for_polyatomic() {
    let xyz = "3\nwater\nO 0.000 0.000 0.117\nH 0.000 0.757 -0.469\nH 0.000 -0.757 -0.469\n";
    let (mol, obs, d) = rhf_density(xyz, "cc-pvdz");
    let (pts, esp, n_buried) =
        ferric_scf::properties::esp_on_surface_counted(&mol, &obs, &d, 1.4, 110).unwrap();
    assert!(n_buried > 0, "water must bury points");
    assert_eq!(pts.len() + n_buried, 3 * 110);
    // The counted variant must agree exactly with the plain one.
    let (pts2, esp2) = esp_on_surface(&mol, &obs, &d, 1.4, 110).unwrap();
    assert_eq!(pts, pts2);
    assert_eq!(esp, esp2);
    // Reference values pinned by the Python binding test.
    let sum: f64 = esp.iter().sum();
    let sum_abs: f64 = esp.iter().map(|v| v.abs()).sum();
    println!(
        "SURFACE_REF n={} buried={} sum={:.15e} sum_abs={:.15e} first={:.15e} last={:.15e}",
        pts.len(),
        n_buried,
        sum,
        sum_abs,
        esp[0],
        esp[esp.len() - 1]
    );
}

#[test]
fn surface_buries_nothing_for_an_isolated_atom() {
    let (mol, obs, d) = rhf_density("1\nne\nNe 0.0 0.0 0.0\n", "sto-3g");
    let (pts, _esp, n_buried) =
        ferric_scf::properties::esp_on_surface_counted(&mol, &obs, &d, 1.4, 110).unwrap();
    assert_eq!(n_buried, 0);
    assert_eq!(pts.len(), 110);
}
