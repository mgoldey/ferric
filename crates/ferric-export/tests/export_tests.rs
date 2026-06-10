use ferric_core::mol::{Molecule, Atom};
use ferric_export::cube::{export_cube, GridSpec};
use ferric_export::ml::export_npz;
use ndarray::{Array2, Array3};
use std::fs;

#[test]
fn test_export_cube() {
    let mol = Molecule {
        atoms: vec![
            Atom { symbol: "H".to_string(), z: 1, x: 0.0, y: 0.0, zpos: 0.0, ghost: false },
            Atom { symbol: "H".to_string(), z: 1, x: 1.4, y: 0.0, zpos: 0.0, ghost: false },
        ],
        charge: 0,
        multiplicity: 1,
    };

    let grid = GridSpec::bounding_box(&mol, 2.0, 0.5);
    let mut data = Array3::<f64>::zeros((grid.n_x, grid.n_y, grid.n_z));
    data[[0, 0, 0]] = 1.234;

    let path = "test_output.cube";
    export_cube(path, &mol, &grid, &data, "Test cube export").unwrap();

    let content = fs::read_to_string(path).unwrap();
    assert!(content.contains("Ferric generated cube file"));
    assert!(content.contains("Test cube export"));
    assert!(content.contains("1.23400E"));

    fs::remove_file(path).unwrap();
}

#[test]
fn test_export_npz() {
    let mo_coeffs = Array2::<f64>::eye(2);
    let orbital_energies = vec![-0.5, 0.1];
    let path = "test_output.npz";

    export_npz(
        path,
        Some(&mo_coeffs),
        Some(&orbital_energies),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ).unwrap();

    assert!(std::path::Path::new(path).exists());
    fs::remove_file(path).unwrap();
}
