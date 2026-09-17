use ferric_core::mol::{Atom, Molecule};
use ferric_export::cube::{export_cube, GridSpec};
use ferric_export::ml::{export_npz, NpzBundle};
use ndarray::{Array1, Array2, Array3, Array4};
use ndarray_npy::NpzReader;
use std::fs;

#[test]
fn test_export_cube() {
    let mol = Molecule {
        atoms: vec![
            Atom {
                symbol: "H".to_string(),
                z: 1,
                x: 0.0,
                y: 0.0,
                zpos: 0.0,
                ghost: false,
                n_core_ecp: 0,
            },
            Atom {
                symbol: "H".to_string(),
                z: 1,
                x: 1.4,
                y: 0.0,
                zpos: 0.0,
                ghost: false,
                n_core_ecp: 0,
            },
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
    )
    .unwrap();

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
    let mo_coeffs =
        Array2::<f64>::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let orbital_energies = vec![-1.5, -0.75, 0.25];
    let path = "test_round_trip.npz";

    export_npz(
        path,
        &NpzBundle {
            mo_coeffs: Some(&mo_coeffs),
            orbital_energies: Some(&orbital_energies),
            ..Default::default()
        },
    )
    .unwrap();

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

    let pts = Array2::from_shape_vec((3, 3), vec![0.0, 0.0, 10.0, 0.0, 0.0, 20.0, 1.0, 2.0, 3.0])
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
    assert!(
        (got_p[(1, 2)] - 20.0).abs() < 1e-15,
        "coordinate round-trip"
    );
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
    let err = export_npz("test_esp_bad1.npz", &values_only)
        .unwrap_err()
        .to_string();
    assert!(err.contains("without esp_points"), "got: {err}");

    let points_only = NpzBundle {
        polarizability: PolarizabilityBundle {
            esp_points: Some(&pts),
            ..Default::default()
        },
        ..Default::default()
    };
    let err = export_npz("test_esp_bad2.npz", &points_only)
        .unwrap_err()
        .to_string();
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
    let err = export_npz("test_esp_bad3.npz", &mismatched)
        .unwrap_err()
        .to_string();
    assert!(err.contains("expected (3, 3)"), "got: {err}");

    for f in [
        "test_esp_bad1.npz",
        "test_esp_bad2.npz",
        "test_esp_bad3.npz",
    ] {
        fs::remove_file(f).ok();
    }
}

// ---------------------------------------------------------------------------
// Exhaustive NpzBundle round-trip coverage (added 2026-09-16).
//
// Motivation: the pre-existing round-trip test above covers only
// `mo_coeffs`/`orbital_energies`. The two fields the VALIDATION.md row is
// NAMED for (`pdep_eigenvectors`, `boys_coeffs`) had no value check at all,
// and neither did the other ~23 exportable fields. These tests close that
// gap for EVERY field `export_npz` can write.
//
// Design rules for the fixture values below — each exists to catch a specific
// silent-wrong failure mode, and every one has been mutation-tested:
//   * NOTHING is zero, all-ones, sequential, or symmetric. A symmetric matrix
//     survives a transpose; sequential integers survive an off-by-one shift.
//   * Every 2-D/3-D/4-D array is NON-SQUARE in its leading axes where the API
//     permits, so a transposed write fails on SHAPE, not just on values.
//   * Every field's values live in its OWN decade (1e1 for mo_coeffs, 1e2 for
//     pdep_eigenvectors, ...). Two fields swapped with each other therefore
//     fail loudly instead of both "round-tripping" plausible-looking numbers.
//   * Nothing is `None`: a silent-None passthrough at a call site (the
//     originating bug class, see the 2026-07-18 note above) shows up as a
//     MISSING KEY, which `npz_export_writes_exactly_the_expected_key_set`
//     turns into a hard failure.
// ---------------------------------------------------------------------------

/// Owned backing storage for the fully-populated bundle. `NpzBundle` borrows
/// everything, so the arrays must outlive it; keeping them in one struct lets
/// each test build the bundle with a single call.
struct FullFixture {
    mo_coeffs: Array2<f64>,
    orbital_energies: Vec<f64>,
    pdep_eigenvectors: Array2<f64>,
    boys_coeffs: Array2<f64>,
    coords: Array2<f64>,
    atomic_numbers: Vec<usize>,
    density_matrix: Array2<f64>,
    orbital_centers: Array2<f64>,
    orbital_spreads: Vec<f64>,
    density_second_moment: Array2<f64>,
    dipole: [f64; 3],
    hirshfeld: Vec<f64>,
    lowdin: Vec<f64>,
    mulliken: Vec<f64>,
    chelpg: Vec<f64>,
    resp: Vec<f64>,
    esp_atoms: Vec<f64>,
    esp_surface: Vec<f64>,
    esp_points: Array2<f64>,
    alpha_tensor: [[f64; 3]; 3],
    electric_field: Vec<[f64; 3]>,
    alpha_atomic: Vec<[[f64; 3]; 3]>,
    c6_freqs: Vec<f64>,
    c6_weights: Vec<f64>,
    alpha_atomic_dynamic: Vec<Vec<[[f64; 3]; 3]>>,
    c6_iso: Array2<f64>,
    c6_aniso: Vec<Vec<[[f64; 3]; 3]>>,
}

/// Number of atoms in the fixture. Deliberately NOT equal to 3, so a per-atom
/// (n, 3) array cannot be transposed into a valid (3, n) one.
const NAT: usize = 4;
/// Number of quadrature frequencies. Deliberately != NAT and != 3, so the
/// (NAT, NFREQ, 3, 3) dynamic-polarizability array has four distinct axis
/// lengths and any axis permutation is a shape error.
const NFREQ: usize = 5;

impl FullFixture {
    /// Builds every field with distinctive, asymmetric, per-field-banded values.
    fn new() -> Self {
        // Helper: a strictly-increasing but NON-linear, non-round generator.
        // `band` separates fields by decade; `i` never produces a repeat.
        let v = |band: f64, i: usize| band * (1.0 + (i as f64) * 0.317) + 0.0719 * (i as f64);

        // (2, 3) non-square, row-major and asymmetric.
        let mo_coeffs =
            Array2::from_shape_vec((2, 3), (0..6).map(|i| v(10.0, i)).collect()).unwrap();
        // (3, 2) — deliberately the TRANSPOSE shape of mo_coeffs, in a
        // different decade, so mo_coeffs<->pdep_eigenvectors confusion is a
        // shape error AND a value error.
        let pdep_eigenvectors =
            Array2::from_shape_vec((3, 2), (0..6).map(|i| v(100.0, i)).collect()).unwrap();
        // (4, 2) non-square.
        let boys_coeffs =
            Array2::from_shape_vec((4, 2), (0..8).map(|i| v(1000.0, i)).collect()).unwrap();
        // (NAT, 3) Cartesian, non-square since NAT != 3.
        let coords =
            Array2::from_shape_vec((NAT, 3), (0..NAT * 3).map(|i| v(0.5, i)).collect()).unwrap();
        // Distinct, non-sequential Z values.
        let atomic_numbers = vec![8usize, 1, 6, 17];
        // (3, 3) but explicitly NON-symmetric: a density matrix is symmetric in
        // physics, so a physical fixture would silently survive a transpose.
        // The exporter does no symmetrisation, so an asymmetric fixture is the
        // honest probe of what the WRITER does.
        let density_matrix =
            Array2::from_shape_vec((3, 3), (0..9).map(|i| v(0.01, i)).collect()).unwrap();
        let orbital_centers =
            Array2::from_shape_vec((NAT, 3), (0..NAT * 3).map(|i| v(2.0, i)).collect()).unwrap();
        let orbital_spreads = (0..NAT).map(|i| v(0.25, i)).collect::<Vec<_>>();
        // Again asymmetric on purpose — see density_matrix.
        let density_second_moment =
            Array2::from_shape_vec((3, 3), (0..9).map(|i| v(7.0, i)).collect()).unwrap();
        let dipole = [0.401_3, -1.205_7, 2.908_1];

        // Five charge schemes, each in its own band, so a cross-wired key
        // (e.g. mulliken written under "lowdin_charges") fails on VALUES.
        let hirshfeld = (0..NAT).map(|i| v(0.11, i)).collect::<Vec<_>>();
        let lowdin = (0..NAT).map(|i| v(0.22, i)).collect::<Vec<_>>();
        let mulliken = (0..NAT).map(|i| v(0.33, i)).collect::<Vec<_>>();
        let chelpg = (0..NAT).map(|i| v(0.44, i)).collect::<Vec<_>>();
        let resp = (0..NAT).map(|i| v(0.55, i)).collect::<Vec<_>>();

        let esp_atoms = (0..NAT).map(|i| v(0.66, i)).collect::<Vec<_>>();
        // Surface ESP: 6 points, so its length differs from NAT and from every
        // other per-atom array -- esp_surface<->esp_atoms confusion is a
        // length error.
        let esp_surface = (0..6).map(|i| v(0.077, i)).collect::<Vec<_>>();
        let esp_points =
            Array2::from_shape_vec((6, 3), (0..18).map(|i| v(3.0, i)).collect()).unwrap();

        // Asymmetric 3x3 (a real polarizability tensor is symmetric; see above).
        let mut alpha_tensor = [[0.0f64; 3]; 3];
        for (i, row) in alpha_tensor.iter_mut().enumerate() {
            for (j, cell) in row.iter_mut().enumerate() {
                *cell = v(20.0, i * 3 + j);
            }
        }
        let electric_field = (0..NAT)
            .map(|a| {
                let mut r = [0.0f64; 3];
                for (k, cell) in r.iter_mut().enumerate() {
                    *cell = v(0.008, a * 3 + k);
                }
                r
            })
            .collect::<Vec<_>>();
        let alpha_atomic = (0..NAT)
            .map(|a| {
                let mut t = [[0.0f64; 3]; 3];
                for (i, row) in t.iter_mut().enumerate() {
                    for (j, cell) in row.iter_mut().enumerate() {
                        *cell = v(5.0, a * 9 + i * 3 + j);
                    }
                }
                t
            })
            .collect::<Vec<_>>();

        let c6_freqs = (0..NFREQ).map(|i| v(0.9, i)).collect::<Vec<_>>();
        let c6_weights = (0..NFREQ).map(|i| v(0.045, i)).collect::<Vec<_>>();
        let alpha_atomic_dynamic = (0..NAT)
            .map(|a| {
                (0..NFREQ)
                    .map(|f| {
                        let mut t = [[0.0f64; 3]; 3];
                        for (i, row) in t.iter_mut().enumerate() {
                            for (j, cell) in row.iter_mut().enumerate() {
                                *cell = v(30.0, (a * NFREQ + f) * 9 + i * 3 + j);
                            }
                        }
                        t
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        // (NAT, NAT) pair matrix, asymmetric on purpose.
        let c6_iso =
            Array2::from_shape_vec((NAT, NAT), (0..NAT * NAT).map(|i| v(40.0, i)).collect())
                .unwrap();
        let c6_aniso = (0..NAT)
            .map(|a| {
                (0..NAT)
                    .map(|b| {
                        let mut t = [[0.0f64; 3]; 3];
                        for (i, row) in t.iter_mut().enumerate() {
                            for (j, cell) in row.iter_mut().enumerate() {
                                *cell = v(60.0, (a * NAT + b) * 9 + i * 3 + j);
                            }
                        }
                        t
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        Self {
            mo_coeffs,
            orbital_energies: (0..3).map(|i| v(-0.7, i)).collect(),
            pdep_eigenvectors,
            boys_coeffs,
            coords,
            atomic_numbers,
            density_matrix,
            orbital_centers,
            orbital_spreads,
            density_second_moment,
            dipole,
            hirshfeld,
            lowdin,
            mulliken,
            chelpg,
            resp,
            esp_atoms,
            esp_surface,
            esp_points,
            alpha_tensor,
            electric_field,
            alpha_atomic,
            c6_freqs,
            c6_weights,
            alpha_atomic_dynamic,
            c6_iso,
            c6_aniso,
        }
    }

    /// Every `Option` field set to `Some`. Nothing is left `None`, which is
    /// what makes the key-set test able to detect a dropped field.
    fn bundle(&self) -> NpzBundle<'_> {
        use ferric_export::ml::{ChargeSchemes, DispersionBundle, PolarizabilityBundle};
        NpzBundle {
            mo_coeffs: Some(&self.mo_coeffs),
            orbital_energies: Some(&self.orbital_energies),
            pdep_eigenvectors: Some(&self.pdep_eigenvectors),
            boys_coeffs: Some(&self.boys_coeffs),
            coords: Some(&self.coords),
            atomic_numbers: Some(&self.atomic_numbers),
            density_matrix: Some(&self.density_matrix),
            orbital_centers: Some(&self.orbital_centers),
            orbital_spreads: Some(&self.orbital_spreads),
            density_second_moment: Some(&self.density_second_moment),
            dipole: Some(&self.dipole),
            charges: ChargeSchemes {
                hirshfeld: Some(&self.hirshfeld),
                lowdin: Some(&self.lowdin),
                mulliken: Some(&self.mulliken),
                chelpg: Some(&self.chelpg),
                resp: Some(&self.resp),
            },
            polarizability: PolarizabilityBundle {
                esp_atoms: Some(&self.esp_atoms),
                esp_surface: Some(&self.esp_surface),
                esp_points: Some(&self.esp_points),
                alpha_tensor: Some(&self.alpha_tensor),
                electric_field: Some(&self.electric_field),
                alpha_atomic: Some(&self.alpha_atomic),
            },
            dispersion: DispersionBundle {
                c6_freqs: Some(&self.c6_freqs),
                c6_weights: Some(&self.c6_weights),
                alpha_atomic_dynamic: Some(&self.alpha_atomic_dynamic),
                c6_iso: Some(&self.c6_iso),
                c6_aniso: Some(&self.c6_aniso),
            },
        }
    }
}

/// The complete set of NPZ keys a fully-populated `NpzBundle` must produce.
///
/// This list is the COVERAGE LEDGER for the round-trip test below. It is
/// asserted to match the written archive EXACTLY (set equality, both
/// directions), so:
///   * adding a new `NpzBundle` field + writer branch without adding it here
///     FAILS ("unexpected key"), forcing the author to also add a value
///     assertion — the omission cannot pass silently as the struct grows;
///   * a writer branch that stops firing (a silent-None passthrough at a call
///     site, the originating bug class) FAILS ("missing key").
const EXPECTED_NPZ_KEYS: [&str; 27] = [
    "mo_coeffs",
    "orbital_energies",
    "pdep_eigenvectors",
    "boys_coeffs",
    "orbital_centers",
    "orbital_spreads",
    "density_second_moment",
    "coords",
    "atomic_numbers",
    "esp_atoms",
    "esp_surface",
    "esp_points",
    "alpha_tensor",
    "electric_field",
    "density_matrix",
    "alpha_atomic",
    "hirshfeld_charges",
    "lowdin_charges",
    "mulliken_charges",
    "chelpg_charges",
    "resp_charges",
    "c6_freqs",
    "c6_weights",
    "alpha_atomic_dynamic",
    "c6_iso",
    "c6_aniso",
    "dipole",
];

/// A fully-populated bundle writes EXACTLY the expected key set — no more, no
/// fewer. See `EXPECTED_NPZ_KEYS` for why this is the structural guard.
#[test]
fn npz_export_writes_exactly_the_expected_key_set() {
    let fx = FullFixture::new();
    let path = "test_npz_full_keyset.npz";
    export_npz(path, &fx.bundle()).unwrap();

    let mut r = NpzReader::new(fs::File::open(path).unwrap()).unwrap();
    let mut got: Vec<String> = r
        .names()
        .unwrap()
        .into_iter()
        // ndarray-npy stores each array as "<name>.npy" inside the zip.
        .map(|n| n.trim_end_matches(".npy").to_string())
        .collect();
    got.sort();

    let mut want: Vec<String> = EXPECTED_NPZ_KEYS.iter().map(|s| s.to_string()).collect();
    want.sort();

    let missing: Vec<&String> = want.iter().filter(|k| !got.contains(k)).collect();
    let unexpected: Vec<&String> = got.iter().filter(|k| !want.contains(k)).collect();
    assert!(
        missing.is_empty(),
        "NPZ is MISSING expected key(s) {missing:?} — a writer branch did not fire \
         (silent-None passthrough?). Got: {got:?}"
    );
    assert!(
        unexpected.is_empty(),
        "NPZ contains UNEXPECTED key(s) {unexpected:?} — a new NpzBundle field was \
         added without adding it to EXPECTED_NPZ_KEYS and to \
         npz_export_round_trips_every_bundle_field"
    );
    assert_eq!(got, want, "NPZ key set mismatch");
    assert_eq!(
        got.len(),
        EXPECTED_NPZ_KEYS.len(),
        "duplicate or dropped key in the archive"
    );

    fs::remove_file(path).ok();
}

/// EXACT value round-trip for EVERY field `export_npz` can write, including
/// `pdep_eigenvectors` and `boys_coeffs` (which previously had no value check
/// at all) and every nested charge/polarizability/dispersion field.
///
/// Equality is EXACT (`assert_eq!` on f64 / `Array` values, not a tolerance):
/// NPZ is a lossless binary container, so anything other than bit-identical
/// readback is a bug, not a rounding artefact. Shapes are asserted separately
/// and BEFORE the values, so a transposed or flattened array reports as a
/// shape failure rather than a confusing value diff.
#[test]
fn npz_export_round_trips_every_bundle_field() {
    let fx = FullFixture::new();
    let path = "test_npz_full_roundtrip.npz";
    export_npz(path, &fx.bundle()).unwrap();

    let mut r = NpzReader::new(fs::File::open(path).unwrap()).unwrap();

    // --- rank-2 arrays: shape FIRST (catches transpose), then exact values.
    for (key, want) in [
        ("mo_coeffs", &fx.mo_coeffs),
        ("pdep_eigenvectors", &fx.pdep_eigenvectors),
        ("boys_coeffs", &fx.boys_coeffs),
        ("coords", &fx.coords),
        ("density_matrix", &fx.density_matrix),
        ("orbital_centers", &fx.orbital_centers),
        ("density_second_moment", &fx.density_second_moment),
        ("esp_points", &fx.esp_points),
        ("c6_iso", &fx.c6_iso),
    ] {
        let got: Array2<f64> = r
            .by_name(key)
            .unwrap_or_else(|e| panic!("{key}: not readable as (n, m) f64: {e}"));
        assert_eq!(
            got.shape(),
            want.shape(),
            "{key}: shape/orientation mismatch"
        );
        assert_eq!(&got, want, "{key}: value round-trip mismatch");
    }

    // --- rank-1 f64 arrays.
    for (key, want) in [
        ("orbital_energies", &fx.orbital_energies),
        ("orbital_spreads", &fx.orbital_spreads),
        ("esp_atoms", &fx.esp_atoms),
        ("esp_surface", &fx.esp_surface),
        ("hirshfeld_charges", &fx.hirshfeld),
        ("lowdin_charges", &fx.lowdin),
        ("mulliken_charges", &fx.mulliken),
        ("chelpg_charges", &fx.chelpg),
        ("resp_charges", &fx.resp),
        ("c6_freqs", &fx.c6_freqs),
        ("c6_weights", &fx.c6_weights),
    ] {
        let got: Array1<f64> = r
            .by_name(key)
            .unwrap_or_else(|e| panic!("{key}: not readable as (n,) f64: {e}"));
        assert_eq!(got.len(), want.len(), "{key}: length mismatch");
        assert_eq!(&got.to_vec(), want, "{key}: value round-trip mismatch");
    }

    // --- dipole: fixed-length 3-vector.
    let dipole: Array1<f64> = r.by_name("dipole").unwrap();
    assert_eq!(dipole.len(), 3, "dipole: length mismatch");
    assert_eq!(
        dipole.to_vec(),
        fx.dipole.to_vec(),
        "dipole: value mismatch"
    );

    // --- atomic_numbers: written as i64, NOT f64. Reading it as i64 is itself
    // part of the contract a consumer relies on.
    let z: Array1<i64> = r
        .by_name("atomic_numbers")
        .expect("atomic_numbers must be readable as i64");
    assert_eq!(
        z.to_vec(),
        fx.atomic_numbers
            .iter()
            .map(|&x| x as i64)
            .collect::<Vec<i64>>(),
        "atomic_numbers: value round-trip mismatch"
    );

    // --- alpha_tensor: [[f64; 3]; 3] flattened to a (3, 3) array. Asserting
    // [i][j] elementwise (not a flat compare) is what catches a transpose.
    let at: Array2<f64> = r.by_name("alpha_tensor").unwrap();
    assert_eq!(at.shape(), &[3, 3], "alpha_tensor: shape mismatch");
    for i in 0..3 {
        for j in 0..3 {
            assert_eq!(
                at[(i, j)],
                fx.alpha_tensor[i][j],
                "alpha_tensor[{i}][{j}] mismatch (transposed?)"
            );
        }
    }

    // --- electric_field: (NAT, 3).
    let ef: Array2<f64> = r.by_name("electric_field").unwrap();
    assert_eq!(ef.shape(), &[NAT, 3], "electric_field: shape mismatch");
    for (a, row) in fx.electric_field.iter().enumerate() {
        for (k, &want) in row.iter().enumerate() {
            assert_eq!(ef[(a, k)], want, "electric_field[{a}][{k}] mismatch");
        }
    }

    // --- alpha_atomic: (NAT, 3, 3).
    let aa: Array3<f64> = r.by_name("alpha_atomic").unwrap();
    assert_eq!(aa.shape(), &[NAT, 3, 3], "alpha_atomic: shape mismatch");
    for (a, t) in fx.alpha_atomic.iter().enumerate() {
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(
                    aa[(a, i, j)],
                    t[i][j],
                    "alpha_atomic[{a}][{i}][{j}] mismatch"
                );
            }
        }
    }

    // --- alpha_atomic_dynamic: (NAT, NFREQ, 3, 3). Four distinct axis lengths,
    // so ANY axis permutation is a shape failure.
    let ad: Array4<f64> = r.by_name("alpha_atomic_dynamic").unwrap();
    assert_eq!(
        ad.shape(),
        &[NAT, NFREQ, 3, 3],
        "alpha_atomic_dynamic: shape mismatch"
    );
    for (a, per_atom) in fx.alpha_atomic_dynamic.iter().enumerate() {
        for (f, t) in per_atom.iter().enumerate() {
            for i in 0..3 {
                for j in 0..3 {
                    assert_eq!(
                        ad[(a, f, i, j)],
                        t[i][j],
                        "alpha_atomic_dynamic[{a}][{f}][{i}][{j}] mismatch"
                    );
                }
            }
        }
    }

    // --- c6_aniso: (NAT, NAT, 3, 3). The pair axes are asymmetric on purpose,
    // so an (A, B) <-> (B, A) swap is caught.
    let ca: Array4<f64> = r.by_name("c6_aniso").unwrap();
    assert_eq!(ca.shape(), &[NAT, NAT, 3, 3], "c6_aniso: shape mismatch");
    for (a, row) in fx.c6_aniso.iter().enumerate() {
        for (b, t) in row.iter().enumerate() {
            for i in 0..3 {
                for j in 0..3 {
                    assert_eq!(
                        ca[(a, b, i, j)],
                        t[i][j],
                        "c6_aniso[{a}][{b}][{i}][{j}] mismatch"
                    );
                }
            }
        }
    }

    fs::remove_file(path).ok();
}

/// REACHABILITY / TEETH check for the two tests above.
///
/// Per the repo's Experimental Protocol ("a test you have never seen fail is
/// an assumption", "check the pass condition is REACHABLE"), this test proves
/// the failure modes the assertions claim to catch are actually detectable by
/// the fixture's VALUES — without editing the implementation. It constructs
/// the corruptions directly and asserts the fixture distinguishes them.
#[test]
fn full_fixture_values_can_actually_distinguish_the_failure_modes() {
    let fx = FullFixture::new();

    // 1. TRANSPOSE is detectable: every rank-2 fixture array differs from its
    //    own transpose (either in shape, or in values for the square ones).
    for (name, a) in [
        ("mo_coeffs", &fx.mo_coeffs),
        ("pdep_eigenvectors", &fx.pdep_eigenvectors),
        ("boys_coeffs", &fx.boys_coeffs),
        ("coords", &fx.coords),
        ("density_matrix", &fx.density_matrix),
        ("orbital_centers", &fx.orbital_centers),
        ("density_second_moment", &fx.density_second_moment),
        ("esp_points", &fx.esp_points),
        ("c6_iso", &fx.c6_iso),
    ] {
        let t = a.t().to_owned();
        assert!(
            t.shape() != a.shape() || t != *a,
            "{name} is transpose-invariant (symmetric/square-identical): a \
             transposed write would round-trip CLEAN and the test would be inert"
        );
    }
    // alpha_tensor likewise.
    let mut tensor_asym = false;
    for i in 0..3 {
        for j in 0..3 {
            if fx.alpha_tensor[i][j] != fx.alpha_tensor[j][i] {
                tensor_asym = true;
            }
        }
    }
    assert!(
        tensor_asym,
        "alpha_tensor is symmetric — transpose undetectable"
    );

    // 2. FIELD SWAP is detectable: no two same-typed fixture arrays are equal,
    //    so writing one under the other's key changes the values.
    let rank1: [(&str, &Vec<f64>); 11] = [
        ("orbital_energies", &fx.orbital_energies),
        ("orbital_spreads", &fx.orbital_spreads),
        ("esp_atoms", &fx.esp_atoms),
        ("esp_surface", &fx.esp_surface),
        ("hirshfeld", &fx.hirshfeld),
        ("lowdin", &fx.lowdin),
        ("mulliken", &fx.mulliken),
        ("chelpg", &fx.chelpg),
        ("resp", &fx.resp),
        ("c6_freqs", &fx.c6_freqs),
        ("c6_weights", &fx.c6_weights),
    ];
    for (i, (na, a)) in rank1.iter().enumerate() {
        for (nb, b) in rank1.iter().skip(i + 1) {
            assert!(
                a != b,
                "{na} and {nb} hold identical values — swapping them would be \
                 undetectable and both assertions would be inert"
            );
        }
    }

    // 3. SILENT-NONE / EMPTY is detectable: nothing is empty and nothing is
    //    all-zero, so an absent or zeroed array cannot match.
    for (name, a) in rank1.iter() {
        assert!(!a.is_empty(), "{name} is empty");
        assert!(
            a.iter().any(|&x| x != 0.0),
            "{name} is all-zero — a zeroed/default array would round-trip clean"
        );
    }

    // 4. FLATTENING / MIS-STRIDING is detectable: each multi-axis fixture has
    //    at least two DISTINCT axis lengths, so a reshape changes the shape.
    assert_ne!(NAT, 3, "NAT == 3 would make (NAT, 3) arrays square");
    assert_ne!(
        NFREQ, 3,
        "NFREQ == 3 would make an axis permutation invisible"
    );
    assert_ne!(NAT, NFREQ, "NAT == NFREQ would hide an atom/frequency swap");
    assert_ne!(
        fx.mo_coeffs.shape(),
        fx.pdep_eigenvectors.shape(),
        "mo_coeffs and pdep_eigenvectors share a shape — a key swap would not \
         be caught on shape alone"
    );

    // 5. OFF-BY-ONE / SHIFT is detectable: values are strictly monotone and
    //    non-sequential, so a shifted read never realigns.
    for (name, a) in rank1.iter() {
        for w in a.windows(2) {
            assert_ne!(
                w[0], w[1],
                "{name} has a repeated value — a shift could hide"
            );
        }
    }
}
