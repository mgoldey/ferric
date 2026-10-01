//! Hirshfeld-volume nuclear derivative and the MBD@rsSCS-on-SCF gradient.
//!
//! Fast tests (run by default):
//!   * `on_grid_volumes_are_bit_identical_to_default` — the refactor anchor.
//!   * `radial_proatom_deriv_matches_fd_inside_segments` — the interpolant slope.
//!   * `hirshfeld_volume_gradient_matches_fd_*` — THE construction test:
//!     analytic ∂(Σ_A c_A v_A)/∂R vs central FD of the on-grid volumes with D and
//!     the lattice held fixed.
//!
//! Measurements (`#[ignore]`, they print; the main session reads them):
//!   * `measure_translation_sum_with_fixed_lattice`
//!   * `measure_full_pipeline_fd_rks_pbe`
//!
//!   OPENBLAS_NUM_THREADS=1 cargo test -p ferric-rpa --test mbd_scf_gradient --release
//!   OPENBLAS_NUM_THREADS=1 cargo test -p ferric-rpa --test mbd_scf_gradient --release \
//!     -- --ignored --nocapture

use ferric_core::basis::{self, BasisSet};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::ao_grid::GridSpec;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_rpa::dispersion::{mbd_rsscs_for_density, MbdFreeAtomCache, MbdRsscsConfig};
use ferric_rpa::properties::{
    atomic_effective_volumes_hirshfeld, atomic_effective_volumes_hirshfeld_on_grid,
    hirshfeld_volume_gradient, hirshfeld_volume_grid, ProatomProvider, RadialProatom,
};
use ferric_scf::properties::{scf_proatom_radii, slater_xi_for_z};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ndarray::Array2;

const H2O_XYZ: &str = "3\nwater\nO 0.000000 0.000000 0.117300\nH 0.000000 0.757200 -0.469200\nH 0.000000 -0.757200 -0.469200\n";
const NH3_XYZ: &str = "4\nammonia\nN 0.000000 0.000000 0.116489\nH 0.000000 0.939731 -0.271808\nH 0.813831 -0.469865 -0.271808\nH -0.813831 -0.469865 -0.271808\n";

/// Bar for the Slater-proatom construction test, relative to max(1, max|G|).
/// Measured 2.5e-8 abs (1.1e-8 rel) at h = 1e-4 Bohr, H2O/STO-3G; a one-sided
/// cusp derivative on the lattice point that coincides with a nucleus missed
/// by 2.3e-3.
const FD_TOL_SMOOTH_REL: f64 = 1e-6;
/// Bar for the tabulated-proatom construction test. The piecewise-linear
/// interpolant has kinks at its knots (0.05 Bohr apart), and a lattice point
/// whose distance to the moved atom crosses a knot within ±h breaks central
/// FD. Measured: h = 1e-5 → 1.2e-4 abs (a few such points), h = 1e-6 →
/// 1.3e-7 abs (7.5e-8 rel), so the test runs at h = 1e-6. A one-sided slope at
/// the knots that the lattice-coincident H atom's neighbours sit on missed by
/// 3.2e-2.
const FD_TOL_TABULATED_REL: f64 = 1e-6;

fn rhf_density(mol: &Molecule, bs: &BasisSet, cfg: &RhfConfig) -> Array2<f64> {
    let prep = PreparedBasis::new(mol, bs).expect("prepared basis");
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).expect("schwarz");
    let ctx = ParallelContext::default();
    let r = solve_rhf(&ctx, mol, &prep, op, &bounds, cfg).expect("scf");
    assert!(r.converged, "SCF did not converge");
    r.density_r().to_owned()
}

fn tight_hf() -> RhfConfig {
    RhfConfig {
        max_iter: 200,
        energy_conv: 1e-10,
        density_conv: 1e-8,
        ..Default::default()
    }
}

fn h2o_sto3g() -> (Molecule, BasisSet, Array2<f64>) {
    let mol = Molecule::parse_xyz(H2O_XYZ, 0, 1).expect("h2o");
    let bs = basis::bundled("sto-3g").expect("sto-3g");
    let d = rhf_density(&mol, &bs, &tight_hf());
    (mol, bs, d)
}

/// A tabulated proatom that is NOT the Slater fallback (different exponent),
/// on the production radii, so the provider path and `RadialProatom::deriv`
/// are exercised without a free-atom SCF.
fn synthetic_proatom(z: i32) -> RadialProatom {
    let xi = 1.3 * slater_xi_for_z(z);
    let prefac = z as f64 * xi.powi(3) / std::f64::consts::PI;
    let radii = scf_proatom_radii();
    let rho = radii
        .iter()
        .map(|&r| prefac * (-2.0 * xi * r).exp())
        .collect();
    RadialProatom { radii, rho }
}

fn synthetic_provider() -> impl Fn(i32, i32) -> Option<RadialProatom> {
    |z, q| {
        if q == 0 {
            Some(synthetic_proatom(z))
        } else {
            None
        }
    }
}

/// A lattice coarser than production so the FD loop stays fast; any fixed
/// lattice is a valid construction test.
fn test_grid(mol: &Molecule) -> GridSpec {
    GridSpec::bounding_box(mol, 5.0, 0.25)
}

fn displaced(mol: &Molecule, atom: usize, k: usize, h: f64) -> Molecule {
    let mut m = mol.clone();
    match k {
        0 => m.atoms[atom].x += h,
        1 => m.atoms[atom].y += h,
        _ => m.atoms[atom].zpos += h,
    }
    m
}

fn contracted(v: &[f64], c: &[f64]) -> f64 {
    v.iter().zip(c).map(|(a, b)| a * b).sum()
}

/// Central FD of Σ_A c_A v_A on a FIXED `grid` and FIXED `d`.
fn fd_contracted_volumes(
    mol: &Molecule,
    bs: &BasisSet,
    d: &Array2<f64>,
    proatom: Option<&ProatomProvider>,
    grid: &GridSpec,
    c: &[f64],
    h: f64,
) -> Array2<f64> {
    let n = mol.atoms.len();
    let mut out = Array2::<f64>::zeros((n, 3));
    for a in 0..n {
        for k in 0..3 {
            let vp = atomic_effective_volumes_hirshfeld_on_grid(
                &displaced(mol, a, k, h),
                bs,
                d,
                proatom,
                grid,
            )
            .expect("v+");
            let vm = atomic_effective_volumes_hirshfeld_on_grid(
                &displaced(mol, a, k, -h),
                bs,
                d,
                proatom,
                grid,
            )
            .expect("v-");
            out[(a, k)] = (contracted(&vp, c) - contracted(&vm, c)) / (2.0 * h);
        }
    }
    out
}

fn max_abs(a: &Array2<f64>) -> f64 {
    a.iter().fold(0.0_f64, |m, v| m.max(v.abs()))
}

fn compare(label: &str, analytic: &Array2<f64>, fd: &Array2<f64>, tol_rel: f64) {
    let scale = max_abs(fd).max(1.0);
    let diff = max_abs(&(analytic - fd));
    println!(
        "{label}: max|G|={:.6e} max|analytic-FD|={diff:.3e} (bar {:.1e})",
        max_abs(fd),
        tol_rel * scale
    );
    for b in 0..analytic.nrows() {
        println!(
            "  atom {b}: analytic {:?}  fd {:?}",
            analytic.row(b).to_vec(),
            fd.row(b).to_vec()
        );
    }
    assert!(
        diff <= tol_rel * scale,
        "{label}: analytic vs FD differ by {diff:.3e} > {:.3e}",
        tol_rel * scale
    );
}

/// Catches: any change to the volume arithmetic in the refactor (loop order,
/// dV formula, floor, lattice construction) — `assert_eq` on f64, so one ulp
/// fails it. Run with both the Slater fallback and a provider proatom.
#[test]
fn on_grid_volumes_are_bit_identical_to_default() {
    let (mol, bs, d) = h2o_sto3g();
    let grid = hirshfeld_volume_grid(&mol);
    let old = atomic_effective_volumes_hirshfeld(&mol, &bs, &d, None).unwrap();
    let new = atomic_effective_volumes_hirshfeld_on_grid(&mol, &bs, &d, None, &grid).unwrap();
    for (a, (o, n)) in old.iter().zip(&new).enumerate() {
        assert_eq!(
            o.to_bits(),
            n.to_bits(),
            "Slater proatom, atom {a}: {o} vs {n}"
        );
    }
    let p = synthetic_provider();
    let pref: &ProatomProvider = &p;
    let old = atomic_effective_volumes_hirshfeld(&mol, &bs, &d, Some(pref)).unwrap();
    let new = atomic_effective_volumes_hirshfeld_on_grid(&mol, &bs, &d, Some(pref), &grid).unwrap();
    for (a, (o, n)) in old.iter().zip(&new).enumerate() {
        assert_eq!(
            o.to_bits(),
            n.to_bits(),
            "provider proatom, atom {a}: {o} vs {n}"
        );
    }
}

/// Catches: the slope of the wrong segment (off-by-one bracket, e.g. the
/// left-hand segment), a sign error, dividing by the wrong knot spacing, and
/// a non-zero value where `at` is constant (below radii[0], beyond the end).
/// Points sit strictly inside segments (fraction 0.37 of the way), so central
/// FD of the linear piece is exact up to rounding.
#[test]
fn radial_proatom_deriv_matches_fd_inside_segments() {
    let p = synthetic_proatom(8);
    // Make the table non-uniform so a wrong-spacing bug is visible.
    let radii: Vec<f64> = p.radii.iter().map(|&r| r + 0.01 * r * r).collect();
    let pa = RadialProatom {
        radii: radii.clone(),
        rho: p.rho.clone(),
    };
    let mut checked = 0;
    for k in (0..radii.len() - 1).step_by(7) {
        let dr = radii[k + 1] - radii[k];
        let r = radii[k] + 0.37 * dr;
        let h = 1e-3 * dr;
        let fd = (pa.at(r + h) - pa.at(r - h)) / (2.0 * h);
        let an = pa.deriv(r);
        let tol = 1e-8 * fd.abs().max(1e-12) + 1e-14;
        assert!(
            (an - fd).abs() <= tol,
            "segment {k}: deriv {an:.12e} vs FD {fd:.12e}"
        );
        checked += 1;
    }
    assert!(checked > 50);
    assert_eq!(
        pa.deriv(0.5 * radii[0]),
        0.0,
        "below radii[0] `at` is constant"
    );
    assert_eq!(
        pa.deriv(radii[radii.len() - 1] + 1.0),
        0.0,
        "beyond the table `at` is 0"
    );
    let empty = RadialProatom {
        radii: vec![],
        rho: vec![],
    };
    assert_eq!(empty.deriv(1.0), 0.0);
}

/// THE construction test, smooth proatom (Slater fallback, `proatom = None`).
/// D is given a small ANTISYMMETRIC part: ρ = χᵀDχ does not see it, so the
/// volumes are unchanged, but a gradient that uses D instead of (D+Dᵀ)/2 in
/// ∂ρ/∂R = −2 Σ_{μ∈B} ∇χ_μ (Dχ)_μ fails.
/// Catches: a wrong sign or factor in any of terms (i) ∂ρ/∂R, (ii) ∂w/∂R
/// (including the ε placement and the −w_A cross term), (iii) ∂|r−R|³/∂R, a
/// wrong AO→atom map, a missing dV, the D-symmetrization, a wrong Slater
/// radial derivative, and any chunk-fold bug (the lattice spans many chunks).
#[test]
fn hirshfeld_volume_gradient_matches_fd_slater_proatom() {
    let (mol, bs, d) = h2o_sto3g();
    let n = d.nrows();
    let mut d_ns = d.clone();
    for i in 0..n {
        for j in 0..n {
            d_ns[(i, j)] += 0.01 * ((i as f64) - (j as f64)) / (n as f64);
        }
    }
    let grid = test_grid(&mol);
    let c = [0.7, -1.3, 0.45];
    let an = hirshfeld_volume_gradient(&mol, &bs, &d_ns, None, &grid, &c).unwrap();
    let fd = fd_contracted_volumes(&mol, &bs, &d_ns, None, &grid, &c, 1e-4);
    compare("slater proatom", &an, &fd, FD_TOL_SMOOTH_REL);
}

/// THE construction test, tabulated provider proatom.
/// Catches: everything the Slater variant does on the provider branch, plus
/// using the Slater derivative (or zero) for a provider atom and a wrong
/// `RadialProatom::deriv` wiring. Unequal c_A per atom so a per-atom
/// permutation of de_dv is visible.
#[test]
fn hirshfeld_volume_gradient_matches_fd_tabulated_proatom() {
    let (mol, bs, d) = h2o_sto3g();
    let grid = test_grid(&mol);
    let p = synthetic_provider();
    let pref: &ProatomProvider = &p;
    let c = [-0.6, 1.1, 0.25];
    let an = hirshfeld_volume_gradient(&mol, &bs, &d, Some(pref), &grid, &c).unwrap();
    let fd = fd_contracted_volumes(&mol, &bs, &d, Some(pref), &grid, &c, 1e-6);
    compare("tabulated proatom", &an, &fd, FD_TOL_TABULATED_REL);
}

/// Measurement, not a check: with the lattice held fixed the quadrature is not
/// translation invariant, so Σ_B G[B,:] ≠ 0. Prints its size relative to
/// max|G| on the production lattice. Would catch nothing by itself; it sizes
/// the lattice-fixed approximation.
#[test]
#[ignore = "measurement: prints the translation sum of the fixed-lattice volume gradient"]
fn measure_translation_sum_with_fixed_lattice() {
    let (mol, bs, d) = h2o_sto3g();
    let grid = hirshfeld_volume_grid(&mol);
    for (label, c) in [
        ("c = (1,1,1)", [1.0, 1.0, 1.0]),
        ("c = (0.7,-1.3,0.45)", [0.7, -1.3, 0.45]),
    ] {
        let g = hirshfeld_volume_gradient(&mol, &bs, &d, None, &grid, &c).unwrap();
        let s: Vec<f64> = (0..3).map(|k| g.column(k).sum()).collect();
        let m = max_abs(&g);
        println!(
            "{label}: sum_B G = {:?}, max|G| = {m:.6e}, |sum|/max|G| = {:?}",
            s,
            s.iter().map(|v| v.abs() / m).collect::<Vec<_>>()
        );
    }
}

fn rks_pbe() -> RhfConfig {
    RhfConfig {
        xc: Some("PBE".to_string()),
        max_iter: 300,
        energy_conv: 1e-11,
        density_conv: 1e-9,
        ..Default::default()
    }
}

/// Occupied MOs of a closed-shell SCF and its density.
fn rhf_occ(mol: &Molecule, bs: &BasisSet, cfg: &RhfConfig) -> (Array2<f64>, Array2<f64>) {
    let prep = PreparedBasis::new(mol, bs).expect("prepared basis");
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).expect("schwarz");
    let ctx = ParallelContext::default();
    let r = solve_rhf(&ctx, mol, &prep, op, &bounds, cfg).expect("scf");
    assert!(r.converged, "SCF did not converge");
    let nocc = mol.nelec() as usize / 2;
    let c = r.mos_r().slice(ndarray::s![.., ..nocc]).to_owned();
    (c, r.density_r().to_owned())
}

/// D = 2 C' C'ᵀ with C' = C (Cᵀ S(mol) C)^{-1/2}: the reference orbitals,
/// unrotated, Löwdin-orthonormalized in `mol`'s (displaced) basis.
fn reorthonormalized_density(mol: &Molecule, bs: &BasisSet, c: &Array2<f64>) -> Array2<f64> {
    use ndarray_linalg::{Eigh, UPLO};
    let prep = PreparedBasis::new(mol, bs).expect("prepared basis");
    let s = ferric_integrals::oneelectron::overlap(&prep);
    let m = c.t().dot(&s).dot(c);
    let (w, v) = m.eigh(UPLO::Lower).expect("eigh");
    let mut vs = v.clone();
    for (j, wj) in w.iter().enumerate() {
        let f = 1.0 / wj.sqrt();
        vs.column_mut(j).mapv_inplace(|x| x * f);
    }
    let minv = vs.dot(&v.t());
    let cp = c.dot(&minv);
    cp.dot(&cp.t()) * 2.0
}

/// Bar for the analytic MBD gradient against FD of the pipeline with the
/// reference orbitals kept and only re-orthonormalized (what the analytic
/// gradient models). Measured max 3.3e-8 (H2O/6-31G PBE), 2.9e-8
/// (NH3/6-31G PBE), Hartree/Bohr: the lattice following the molecule, which
/// the analytic gradient holds fixed. Dropping the orthonormality term misses
/// by 1.4e-6 (H2O, the size of that term).
const FD_ORTH_TOL: f64 = 3e-7;

/// THE model test, fast (HF/STO-3G water). Catches: the orthonormality term
/// dropped or sign-flipped, `de_dv` not divided by v_free, the volume term
/// dropped, any atom-order mixup between the three terms.
#[test]
fn mbd_gradient_matches_fd_with_reorthonormalized_orbitals() {
    let mol = Molecule::parse_xyz(H2O_XYZ, 0, 1).expect("h2o");
    let bs = basis::bundled("sto-3g").expect("sto-3g");
    let op = Operator::coulomb();
    let ctx = ParallelContext::default();
    let cfg = tight_hf();
    let mbd_cfg = MbdRsscsConfig::for_functional("PBE").expect("beta");
    let cache = MbdFreeAtomCache::build(&ctx, &mol, &bs, op, &cfg).expect("cache");
    let (c0, d0) = rhf_occ(&mol, &bs, &cfg);
    let r0 = mbd_rsscs_for_density(&cache, &mol, &bs, &d0, &mbd_cfg, true).expect("mbd");
    let g = r0.gradient.expect("gradient");
    let orth = r0.gradient_orthonormality.expect("orth");
    let h = 1e-3;
    let n = mol.atoms.len();
    let mut fd = Array2::<f64>::zeros((n, 3));
    for a in 0..n {
        for k in 0..3 {
            let e = |s: f64| {
                let m = displaced(&mol, a, k, s * h);
                let d = reorthonormalized_density(&m, &bs, &c0);
                mbd_rsscs_for_density(&cache, &m, &bs, &d, &mbd_cfg, false)
                    .expect("mbd")
                    .energy
            };
            fd[(a, k)] = (e(1.0) - e(-1.0)) / (2.0 * h);
        }
    }
    let diff = max_abs(&(&g - &fd));
    println!(
        "HF/STO-3G water: max|g| = {:.3e}, max|orth| = {:.3e}, max|g - FD_orth| = {diff:.3e}",
        max_abs(&g),
        max_abs(&orth)
    );
    assert!(
        diff < FD_ORTH_TOL,
        "analytic vs FD_orth: {diff:.3e} >= {FD_ORTH_TOL:.1e}"
    );
    // Reachability: the orthonormality term is resolvable at this bar.
    assert!(
        max_abs(&orth) > 3.0 * FD_ORTH_TOL,
        "orth term {:.3e} below the bar",
        max_abs(&orth)
    );
}

/// Measurement, not a validation. Full pipeline at displaced geometries
/// (lattice rebuilt by `hirshfeld_volume_grid` each time), central FD of E_MBD,
/// decomposed:
///   FD_full  — SCF re-solved: the true derivative of the pipeline.
///   FD_orth  — reference orbitals kept, only re-orthonormalized: what the
///              analytic gradient models (term 1 + fixed-D volume term +
///              orthonormality term), up to the lattice motion.
///   FD_full − FD_orth = the orbital-relaxation term (not implemented).
#[test]
#[ignore = "measurement: full-pipeline FD, slow-ish"]
fn measure_full_pipeline_fd_rks_pbe() {
    let bs = basis::bundled("6-31g").expect("6-31g");
    let op = Operator::coulomb();
    let ctx = ParallelContext::default();
    let cfg = rks_pbe();
    let mbd_cfg = MbdRsscsConfig::for_functional("PBE").expect("beta");
    let h = 1e-3;
    for (name, xyz) in [("H2O", H2O_XYZ), ("NH3", NH3_XYZ)] {
        let mol = Molecule::parse_xyz(xyz, 0, 1).expect("mol");
        let cache = MbdFreeAtomCache::build(&ctx, &mol, &bs, op, &cfg).expect("cache");
        let (c0, d0) = rhf_occ(&mol, &bs, &cfg);
        let r0 = mbd_rsscs_for_density(&cache, &mol, &bs, &d0, &mbd_cfg, true).expect("mbd");
        let g = r0.gradient.clone().expect("gradient");
        let g1 = r0.gradient_fixed_ratios.clone().expect("term 1");
        let g2 = r0.gradient_volume_fixed_d.clone().expect("term 2");
        let g3 = r0.gradient_orthonormality.clone().expect("orth");
        println!(
            "\n{name}/6-31G RKS-PBE: E_MBD = {:.10e}, ratios = {:?}",
            r0.energy, r0.volume_ratios
        );
        let n = mol.atoms.len();
        let mut fd = Array2::<f64>::zeros((n, 3));
        let mut fd_orth = Array2::<f64>::zeros((n, 3));
        for a in 0..n {
            for k in 0..3 {
                let e_full = |s: f64| {
                    let m = displaced(&mol, a, k, s * h);
                    let d = rhf_density(&m, &bs, &cfg);
                    mbd_rsscs_for_density(&cache, &m, &bs, &d, &mbd_cfg, false)
                        .expect("mbd")
                        .energy
                };
                let e_orth = |s: f64| {
                    let m = displaced(&mol, a, k, s * h);
                    let d = reorthonormalized_density(&m, &bs, &c0);
                    mbd_rsscs_for_density(&cache, &m, &bs, &d, &mbd_cfg, false)
                        .expect("mbd")
                        .energy
                };
                fd[(a, k)] = (e_full(1.0) - e_full(-1.0)) / (2.0 * h);
                fd_orth[(a, k)] = (e_orth(1.0) - e_orth(-1.0)) / (2.0 * h);
            }
        }
        println!(
            "{:>4} {:>2} {:>13} {:>13} {:>13} {:>11} {:>11} {:>11} {:>11} {:>11}",
            "atom",
            "k",
            "analytic",
            "FD_orth",
            "FD_full",
            "an-FDorth",
            "relax",
            "term1",
            "term2",
            "orth"
        );
        for a in 0..n {
            for k in 0..3 {
                println!(
                    "{a:>4} {k:>2} {:>13.6e} {:>13.6e} {:>13.6e} {:>11.3e} {:>11.3e} {:>11.3e} {:>11.3e} {:>11.3e}",
                    g[(a, k)],
                    fd_orth[(a, k)],
                    fd[(a, k)],
                    g[(a, k)] - fd_orth[(a, k)],
                    fd[(a, k)] - fd_orth[(a, k)],
                    g1[(a, k)],
                    g2[(a, k)],
                    g3[(a, k)]
                );
            }
        }
        let mfd = max_abs(&fd);
        println!(
            "{name}: max|FD_full| = {mfd:.4e}; max|analytic - FD_orth| = {:.3e}; \
             max|relaxation| = max|FD_full - FD_orth| = {:.3e}; max|analytic - FD_full| = {:.3e}",
            max_abs(&(&g - &fd_orth)),
            max_abs(&(&fd - &fd_orth)),
            max_abs(&(&g - &fd))
        );
        assert!(
            max_abs(&(&g - &fd_orth)) < FD_ORTH_TOL,
            "{name}: analytic vs FD_orth"
        );
    }
}
