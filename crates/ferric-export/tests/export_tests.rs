use ferric_core::mol::{Molecule, Atom};
use ferric_export::cube::{export_cube, GridSpec};
use ferric_export::ml::{export_npz, NpzBundle};
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
        &NpzBundle {
            mo_coeffs: Some(&mo_coeffs),
            orbital_energies: Some(&orbital_energies),
            ..Default::default()
        },
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
        &NpzBundle {
            mo_coeffs: Some(&mo_coeffs),
            orbital_energies: Some(&orbital_energies),
            ..Default::default()
        },
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

/// Surface ESP round-trips through the NPZ, values and coordinates together.
///
/// `esp_surface` without `esp_points` is unusable — a bare list of potentials
/// with no idea where they were evaluated. The exporter therefore rejects each
/// half without the other rather than writing a half-specified file that only
/// fails downstream, in someone else's code, much later.
#[test]
fn npz_surface_esp_round_trips_with_its_coordinates() {
    use ferric_export::ml::PolarizabilityBundle;

    let pts = Array2::from_shape_vec(
        (3, 3),
        vec![0.0, 0.0, 10.0, 0.0, 0.0, 20.0, 1.0, 2.0, 3.0],
    )
    .unwrap();
    let vals = [-6.84e-3, -1.71e-3, 4.2e-2];

    let path = "test_esp_surface.npz";
    let bundle = NpzBundle {
        polarizability: PolarizabilityBundle {
            esp_surface: Some(&vals),
            esp_points: Some(&pts),
            ..Default::default()
        },
        ..Default::default()
    };
    export_npz(path, &bundle).unwrap();

    let mut r = NpzReader::new(fs::File::open(path).unwrap()).unwrap();
    let got_v: Array1<f64> = r.by_name("esp_surface").unwrap();
    let got_p: Array2<f64> = r.by_name("esp_points").unwrap();
    assert_eq!(got_v.len(), 3);
    assert_eq!(got_p.shape(), &[3, 3]);
    for (a, b) in got_v.iter().zip(vals.iter()) {
        assert!((a - b).abs() < 1e-15, "value round-trip: {a} vs {b}");
    }
    assert!((got_p[(1, 2)] - 20.0).abs() < 1e-15, "coordinate round-trip");
    fs::remove_file(path).ok();
}

/// TEETH for the pairing invariant: each half alone must be REJECTED.
///
/// Without these the `match` in the writer could silently take the `(None, _)`
/// arm and drop the data, which is precisely the silent-wrong outcome the
/// pairing check exists to prevent.
#[test]
fn npz_surface_esp_rejects_a_missing_half() {
    use ferric_export::ml::PolarizabilityBundle;

    let pts = Array2::from_shape_vec((2, 3), vec![0.0, 0.0, 1.0, 0.0, 0.0, 2.0]).unwrap();
    let vals = [1.0, 2.0];

    let values_only = NpzBundle {
        polarizability: PolarizabilityBundle {
            esp_surface: Some(&vals),
            ..Default::default()
        },
        ..Default::default()
    };
    let err = export_npz("test_esp_bad1.npz", &values_only).unwrap_err().to_string();
    assert!(err.contains("without esp_points"), "got: {err}");

    let points_only = NpzBundle {
        polarizability: PolarizabilityBundle {
            esp_points: Some(&pts),
            ..Default::default()
        },
        ..Default::default()
    };
    let err = export_npz("test_esp_bad2.npz", &points_only).unwrap_err().to_string();
    assert!(err.contains("without esp_surface"), "got: {err}");

    // ...and a length/shape mismatch must not be written either.
    let three = [1.0, 2.0, 3.0];
    let mismatched = NpzBundle {
        polarizability: PolarizabilityBundle {
            esp_surface: Some(&three),
            esp_points: Some(&pts),
            ..Default::default()
        },
        ..Default::default()
    };
    let err = export_npz("test_esp_bad3.npz", &mismatched).unwrap_err().to_string();
    assert!(err.contains("expected (3, 3)"), "got: {err}");

    for f in ["test_esp_bad1.npz", "test_esp_bad2.npz", "test_esp_bad3.npz"] {
        fs::remove_file(f).ok();
    }
}
