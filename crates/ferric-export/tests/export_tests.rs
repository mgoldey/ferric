use ferric_core::mol::{Molecule, Atom};
use ferric_export::cube::{export_cube, GridSpec};
use ferric_export::ml::export_npz;
use ndarray::{Array1, Array2, Array3};
use ndarray_npy::NpzReader;
use std::fs;

#[test]
fn test_export_cube() {
    let mol = Molecule {
        atoms: vec![
            Atom { symbol: "H".to_string(), z: 1, x: 0.0, y: 0.0, zpos: 0.0, ghost: false, n_core_ecp: 0 },
            Atom { symbol: "H".to_string(), z: 1, x: 1.4, y: 0.0, zpos: 0.0, ghost: false, n_core_ecp: 0 },
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
        None,
    ).unwrap();

    assert!(std::path::Path::new(path).exists());
    fs::remove_file(path).unwrap();
}

/// Real round-trip check (not just "the file exists"): write non-trivial
/// mo_coeffs/orbital_energies, read the NPZ back with NpzReader, and assert
/// the values match exactly. Catches silent-None-passthrough bugs like the
/// one found 2026-07-18 (item #11 in the triage doc): the CLI's export_npz
/// call site hardcoded `None` for mo_coeffs despite the library API
/// supporting it, so mo_coeffs was never actually written by any real
/// caller even though the synthetic library-level test above always
/// "passed" (it only checked file existence, not content).
#[test]
fn test_export_npz_round_trip_values() {
    let mo_coeffs = Array2::<f64>::from_shape_vec((2, 3), vec![
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
    ]).unwrap();
    let orbital_energies = vec![-1.5, -0.75, 0.25];
    let path = "test_round_trip.npz";

    export_npz(
        path,
        Some(&mo_coeffs),
        Some(&orbital_energies),
        None, None, None, None, None, None, None, None, None, None, None,
        None, None, None, None, None, None,
    ).unwrap();

    let mut npz = NpzReader::new(fs::File::open(path).unwrap()).unwrap();
    let read_mo_coeffs: Array2<f64> = npz.by_name("mo_coeffs.npy").unwrap();
    let read_orbital_energies: Array1<f64> = npz.by_name("orbital_energies.npy").unwrap();

    assert_eq!(read_mo_coeffs, mo_coeffs, "mo_coeffs round-trip mismatch");
    assert_eq!(
        read_orbital_energies.to_vec(),
        orbital_energies,
        "orbital_energies round-trip mismatch"
    );

    fs::remove_file(path).unwrap();
}
