//! VALIDATION tier - VALIDATION.md row "Analytic fixed-grid KS-LDA Hessian".
//!
//! Runs only in the weekly `validation` CI job (and on demand):
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 cargo nextest run -p ferric-scf \
//!     --test validation_ks_fixed_grid_hessian --run-ignored only \
//!     -E 'binary(/^validation_/)'
//! ```
//!
//! # What ferric computes
//!
//! `rks_lda_fixed_grid_hessian_parts_with` on an SCF whose XC lives on a grid
//! frozen in space (`ferric_scf::ks_fixed_grid`): exact four-centre J, pure
//! LDA (`LDA_X` + `LDA_C_VWN`), no weight response.
//!
//! # Reference (`scripts/validation/gen_ks_fixed_grid_hessian.py`)
//!
//! PySCF `dft.RKS(xc="LDA_X,LDA_C_VWN")` with ferric's own basis JSON, Bohr
//! geometry and masses, and **ferric's grid**: the 30 radial x 50 angular
//! Becke-Lebedev points and weights ferric builds for the reference geometry
//! are written to `h2o_distorted_grid.f64` (rows of `x y z w`, little-endian
//! f64, Bohr) by the test `reference_grid_matches_the_grid_ferric_builds` (run with `FERRIC_WRITE_REFERENCE_GRID=1`) and installed as `mf.grids.coords` /
//! `weights`, so both codes integrate on the same points with the same weights
//! and neither differentiates them (`Hessian.grid_response = False`, the
//! default). The like-for-like conditions are: same points and weights, same
//! basis functions, same functional (libxc), exact J, no density fitting, SCF
//! converged to 1e-12 in energy on both sides.
//!
//! Stored per system: PySCF's total energy, analytic Hessian, central FD of
//! PySCF's analytic fixed-grid gradient (the reference's own consistency, so a
//! PySCF Hessian that is not the fixed-grid surface would show), and
//! harmonic frequencies (ferric's masses).
//!
//! # Physics hypothesis vs artifact hypothesis
//!
//! * Right: ferric's Hessian matches PySCF's analytic Hessian to the SCF/CPHF
//!   noise (~1e-9), the energies to ~1e-9, on identical points.
//! * A grid mismatch (different points or weights): the ENERGIES differ at
//!   the 1e-6 level or more, before any Hessian is compared; the energy check
//!   below runs first for that reason.
//! * Weight response in PySCF's reference (it would then not be the
//!   fixed-grid surface): `hessian_fd` of PySCF's fixed-grid gradient would
//!   disagree with its own analytic Hessian.
//! * Production Hessian (moving grid) as the reference: misses by the weight
//!   response scale, measured in `ks_fixed_grid_hessian.rs`.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_dft::grid::AtomicGridConfig;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::frequencies::{atom_masses, frequencies_from_cartesian_hessian};
use ferric_scf::hessian::{rks_lda_fixed_grid_hessian_parts_with, KsFixedGridMutants};
use ferric_scf::ks_fixed_grid::{solve_rks_fixed_grid, FixedGridLda};
use ferric_scf::rhf::RhfConfig;
use ferric_scf::screening::SchwarzBounds;
use ndarray::Array2;
use serde_json::Value;

const GRID_FILE: &str =
    "testdata/reference/validation/ks_fixed_grid_hessian/h2o_distorted_grid.f64";

fn repo_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap()
}

fn molecule() -> Molecule {
    let path = repo_root().join("testdata/molecules/validation/h2o_distorted.xyz");
    Molecule::load_xyz(path.to_str().unwrap()).unwrap()
}

/// The grid file PySCF was fed must still be the grid ferric builds for the
/// reference geometry (a drift guard). With `FERRIC_WRITE_REFERENCE_GRID=1` it
/// is rewritten instead (then re-run `gen_ks_fixed_grid_hessian.py`).
#[test]
#[ignore = "validation: shared reference grid matches ferric's grid"]
fn reference_grid_matches_the_grid_ferric_builds() {
    let mol = molecule();
    let cfg = AtomicGridConfig {
        n_radial: 30,
        n_angular: 50,
        prune: None,
    };
    let grid = ferric_scf::ks_fixed_grid::FixedGridLda::new(&mol, "LDA", &cfg).unwrap();
    let (pts, w) = grid.points_and_weights();
    let mut bytes = Vec::with_capacity(32 * pts.len());
    for (p, wg) in pts.iter().zip(w) {
        for v in [p[0], p[1], p[2], *wg] {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
    }
    let path = repo_root().join(GRID_FILE);
    if std::env::var("FERRIC_WRITE_REFERENCE_GRID").is_ok_and(|v| v == "1") {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, &bytes).unwrap();
        eprintln!("wrote {} points to {}", pts.len(), path.display());
        return;
    }
    let on_disk = std::fs::read(&path).unwrap();
    assert!(
        on_disk == bytes,
        "{} differs from the grid ferric builds now; the PySCF references were made on the old \
         grid (FERRIC_WRITE_REFERENCE_GRID=1 rewrites it, then regenerate the JSON)",
        path.display()
    );
}

fn load_grid() -> (Vec<[f64; 3]>, Vec<f64>) {
    let bytes = std::fs::read(repo_root().join(GRID_FILE)).unwrap();
    assert_eq!(bytes.len() % 32, 0, "grid file is rows of x y z w (f64)");
    let val = |i: usize| f64::from_le_bytes(bytes[8 * i..8 * i + 8].try_into().unwrap());
    let n = bytes.len() / 32;
    let pts = (0..n)
        .map(|g| [val(4 * g), val(4 * g + 1), val(4 * g + 2)])
        .collect();
    let w = (0..n).map(|g| val(4 * g + 3)).collect();
    (pts, w)
}

fn matrix(v: &Value) -> Array2<f64> {
    let rows = v.as_array().unwrap();
    let n = rows.len();
    Array2::from_shape_fn((n, n), |(i, j)| rows[i][j].as_f64().unwrap())
}

fn max_abs_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()))
}

struct Case {
    basis: &'static str,
    /// Functional name (ferric's) and the reference file suffix.
    xc: &'static str,
    suffix: &'static str,
    /// Energy vs PySCF, Hartree.
    tol_energy: f64,
    /// Hessian vs PySCF's analytic Hessian, Hartree/Bohr^2.
    tol_hessian: f64,
    /// Frequencies vs PySCF's, cm^-1.
    tol_freq: f64,
}

/// Bars: ~10x the maxima measured on the first run (see the VALIDATION row).
const CASES: [Case; 3] = [
    Case {
        basis: "6-31g",
        xc: "LDA",
        suffix: "",
        tol_energy: 1e-9,
        tol_hessian: 5e-7,
        tol_freq: 1e-3,
    },
    Case {
        basis: "cc-pvdz",
        xc: "LDA",
        suffix: "",
        tol_energy: 1e-9,
        tol_hessian: 1e-5,
        tol_freq: 3e-2,
    },
    // Global-hybrid LDA (a = 0.25): the a-scaled exact exchange of the skeleton,
    // first-order Fock and CPHF against PySCF's own hybrid Hessian.
    Case {
        basis: "6-31g",
        xc: "HYB_LDA_XC_LDA0",
        suffix: "_hyb",
        tol_energy: 1e-9,
        tol_hessian: 5e-7,
        tol_freq: 1e-3,
    },
];

fn run_case(case: &Case) {
    if ferric_integrals::engine::libint_max_deriv_order() < 2 {
        eprintln!("SKIP: libint2 has no second derivatives");
        return;
    }
    let path = repo_root().join(format!(
        "testdata/reference/validation/ks_fixed_grid_hessian/h2o_distorted_{}{}.json",
        case.basis, case.suffix
    ));
    let text = std::fs::read_to_string(&path).unwrap();
    let r: Value = serde_json::from_str(&text).unwrap();
    let mol = molecule();
    let bs = basis::bundled(case.basis).unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();
    let (pts, w) = load_grid();
    assert_eq!(pts.len() as u64, r["npts"].as_u64().unwrap());
    let grid = FixedGridLda::from_points(case.xc, pts, w).unwrap();

    let enuc = (mol.nuclear_repulsion() - r["nuclear_repulsion"].as_f64().unwrap()).abs();
    assert!(
        enuc < 1e-10,
        "geometry/constants mismatch: dVnn = {enuc:.3e}"
    );

    let scf_cfg = RhfConfig {
        use_sad_guess: false,
        max_iter: 200,
        energy_conv: 1e-13,
        density_conv: 1e-11,
        ..Default::default()
    };
    let res = solve_rks_fixed_grid(&ctx, &mol, &prep, op, &bounds, &grid, &scf_cfg).unwrap();
    assert!(res.converged);
    let de = (res.energy - r["energy"].as_f64().unwrap()).abs();
    eprintln!("{}: |E_ferric - E_PySCF| = {de:.3e} Ha", case.basis);
    assert!(
        de < case.tol_energy,
        "energy {de:.3e}: grid or functional mismatch"
    );

    let hess_cfg = RhfConfig {
        xc: Some(case.xc.into()),
        df_j_aux: Some(String::new()),
        df_k_aux: Some(String::new()),
        ..Default::default()
    };
    let h = rks_lda_fixed_grid_hessian_parts_with(
        &ctx,
        &mol,
        &prep,
        op,
        &bounds,
        &res,
        &hess_cfg,
        &grid,
        &KsFixedGridMutants::default(),
    )
    .unwrap()
    .total();
    let h_an = matrix(&r["hessian_analytic"]);
    let h_fd = matrix(&r["hessian_fd"]);
    let d_an = max_abs_diff(&h, &h_an);
    let d_fd = max_abs_diff(&h, &h_fd);
    let masses = atom_masses(&mol).unwrap();
    let f = frequencies_from_cartesian_hessian(&mol, &h, &masses).unwrap();
    let f_ref: Vec<f64> = r["frequencies_analytic_cm1"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    let dnu = f
        .frequencies
        .iter()
        .zip(&f_ref)
        .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
    eprintln!(
        "{}: ferric vs PySCF analytic {d_an:.3e}, vs PySCF FD {d_fd:.3e}, PySCF analytic vs its \
         FD {:.3e}; max |dnu| = {dnu:.3e} cm^-1 (max |H| {:.3e})",
        case.basis,
        r["pyscf_analytic_vs_fd_max"].as_f64().unwrap(),
        h.iter().fold(0.0f64, |m, x| m.max(x.abs()))
    );
    assert!(d_an < case.tol_hessian, "Hessian {d_an:.3e}");
    assert!(dnu < case.tol_freq, "frequencies {dnu:.3e}");
}

#[test]
#[ignore = "validation: analytic fixed-grid KS-LDA Hessian vs PySCF"]
fn fixed_grid_lda_hessian_6_31g_matches_pyscf_on_the_same_grid() {
    run_case(&CASES[0]);
}

#[test]
#[ignore = "validation: analytic fixed-grid KS-LDA Hessian vs PySCF"]
fn fixed_grid_lda_hessian_cc_pvdz_matches_pyscf_on_the_same_grid() {
    run_case(&CASES[1]);
}

#[test]
#[ignore = "validation: analytic fixed-grid KS hybrid-LDA Hessian vs PySCF"]
fn fixed_grid_hybrid_lda_hessian_6_31g_matches_pyscf_on_the_same_grid() {
    run_case(&CASES[2]);
}
