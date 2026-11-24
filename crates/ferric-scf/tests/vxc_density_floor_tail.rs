//! The production closed-shell V_xc is the derivative of the production E_xc
//! along a DIFFUSE direction, so its density floor does not cut a tail that
//! libxc still evaluates.
//!
//! E_xc is summed over every point; V_xc drops points with ρ at or below the
//! functional's `DensityFloor` (`ferric-dft` vxc.rs). Where libxc's potential
//! is non-zero below that floor (B88 exchange decays slowly), Tr[V_xc δD] and
//! the finite difference of E_xc disagree along a direction that lives in the
//! tail. On water/aug-cc-pVDZ B3LYP a floor of 1e-10 moved diffuse virtual
//! orbital energies by up to 3.5e-3 eV against PySCF on the same grid.
//!
//! Hypotheses. Floor below libxc's own thresholds: both directions agree to
//! the FD truncation level. Floor cutting the tail: the HOMO direction (no
//! tail weight) still agrees, the LUMO direction misses by far more. Broken
//! test (FD too coarse, or a direction with no tail weight): both miss, or
//! both agree under either floor — ruled out by the HOMO anchor and by the
//! mutation run (semilocal floor 1e-10) recorded at TOL_REL.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_dft::ao_grid::eval_basis_and_grad_on_points;
use ferric_dft::density_on_grid::eval_density_closed;
use ferric_dft::grid::{build_atomic_grid, AtomicGridConfig};
use ferric_dft::libxc::xc_def_from_name;
use ferric_dft::vxc::semilocal_vxc_closed;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ndarray::Array2;

/// Relative bar on |Tr[V δD] − FD(E)| / |FD(E)| at the finest rung of
/// `STEPS`. The ladder runs to small h because E_xc is not analytic at ρ → 0
/// (LDA exchange ~ ρ^{4/3}): along a tail direction the FD error falls slowly
/// with h, so coarse rungs carry it. Measured at h = 1e-6 (2026-11-05): HOMO
/// 5.6e-9 under either floor; LUMO 5.74e-4 with a 1e-10 semilocal floor and
/// 5.2e-7 with the 1e-14 floor. The bar is about their geometric mean
/// (1.7e-5), ~30x from each.
const TOL_REL: f64 = 2e-5;
const STEPS: [f64; 4] = [1e-3, 1e-4, 1e-5, 1e-6];

/// (Tr[V δD], central-difference dE/dh per rung of `STEPS`) along δD = c cᵀ
/// for MO column `k`.
fn directional(k: usize) -> (f64, Vec<f64>) {
    let mol = Molecule::load_xyz(&format!(
        "{}/../../testdata/molecules/water.xyz",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap();
    let bs = basis::bundled("aug-cc-pvdz").unwrap();
    let obs = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let rhf = solve_rhf(
        &ParallelContext::default(),
        &mol,
        &obs,
        op,
        &bounds,
        &RhfConfig::default(),
    )
    .unwrap();

    let grid = build_atomic_grid(&mol, &AtomicGridConfig::default());
    let pts: Vec<[f64; 3]> = grid.iter().map(|g| g.xyz).collect();
    let (chi, dchi) = eval_basis_and_grad_on_points(&mol, &bs, &pts).unwrap();
    let xc = xc_def_from_name("B3LYP").unwrap();

    let c = rhf.mos_r().column(k).to_owned();
    let n = c.len();
    let dd = Array2::from_shape_fn((n, n), |(m, v)| c[m] * c[v]);

    let d0 = &rhf.density_total;
    let dens = eval_density_closed(d0, &chi, &dchi);
    let (_, vxc) = semilocal_vxc_closed(&grid, &chi, &dchi, &dens, None, &xc);
    let analytic = (&vxc * &dd).sum();

    let e_at = |s: f64| {
        let d = d0 + &(s * &dd);
        let dens = eval_density_closed(&d, &chi, &dchi);
        semilocal_vxc_closed(&grid, &chi, &dchi, &dens, None, &xc).0
    };
    let fd = STEPS
        .iter()
        .map(|&h| (e_at(h) - e_at(-h)) / (2.0 * h))
        .collect();
    (analytic, fd)
}

/// Relative miss at the finest rung of the ladder, printing every rung.
fn finest_rel(label: &str, k: usize) -> f64 {
    let (a, fd) = directional(k);
    let mut r = f64::NAN;
    for (h, f) in STEPS.iter().zip(&fd) {
        r = (a - f).abs() / f.abs();
        println!("{label}: h {h:.0e} Tr[V dD] = {a:.12e}, FD = {f:.12e}, rel = {r:.2e}");
    }
    r
}

/// Trivial limit: a compact occupied direction (HOMO) has no weight in the
/// sub-floor tail, so V_xc is the derivative of E_xc there under any floor.
#[test]
fn vxc_is_the_energy_derivative_along_the_homo() {
    let r = finest_rel("HOMO", 4);
    assert!(r < TOL_REL, "HOMO direction: rel {r:.2e} >= {TOL_REL:e}");
}

/// The diffuse LUMO of water/aug-cc-pVDZ lives in the low-density tail: the
/// floor must not drop the potential there.
#[test]
fn vxc_is_the_energy_derivative_along_the_diffuse_lumo() {
    let r = finest_rel("LUMO", 5);
    assert!(
        r < TOL_REL,
        "diffuse LUMO direction: V_xc is not dE_xc/dD (rel {r:.2e} >= {TOL_REL:e}) — \
         the density floor drops a tail libxc still evaluates"
    );
}
