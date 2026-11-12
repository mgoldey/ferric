//! A Kohn-Sham SCF on the Lebedev orders above 590.
//!
//! Water/6-31G PBE on flat grids with 75 radial shells, at angular orders
//! 590, 974 and 5810 (the largest rule, the angular reference). With the
//! radial grid fixed, `E(5810)` stands in for the angular limit, so a grid's
//! angular error is `|E(n) − E(5810)|`. A working 974-point rule has a
//! smaller angular error than 590; a broken one (a wrong weight or node)
//! moves the energy by the size of the breakage, far past 590's error.
//!
//! The comparison is against the reference, not `|E(974) − E(590)| <
//! err(590)`: 974 lands on the other side of the limit from 590 (measured
//! below), so its step from 590 is `err(590) + err(974)`, which exceeds
//! `err(590)` for a correct rule. The step is bounded by `2 err(590)`, which
//! any rule at least as accurate as 590 satisfies.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_dft::grid::AtomicGridConfig;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

const H2O: &str = "3\nH2O\nO 0 0 0\nH 0 0.7572 0.5868\nH 0 -0.7572 0.5868\n";

fn pbe_energy(n_angular: usize) -> f64 {
    let mol = Molecule::parse_xyz(H2O, 0, 1).unwrap();
    let bs = basis::bundled("6-31g").unwrap();
    let obs = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let cfg = RhfConfig {
        xc: Some("PBE".into()),
        energy_conv: 1e-11,
        density_conv: 1e-9,
        dft_grid: Some(AtomicGridConfig {
            n_radial: 75,
            n_angular,
            prune: None,
            weight_cut: None,
        }),
        ..Default::default()
    };
    let r = solve_rhf(&ParallelContext::default(), &mol, &obs, op, &bounds, &cfg).unwrap();
    assert!(r.converged, "{n_angular}: SCF did not converge");
    r.energy
}

/// Measured 2026-11-12: E(590) −76.298159143929, E(974) −76.298159123100,
/// E(5810) −76.298159123875 Ha; err(590) 2.0e-8, err(974) 7.8e-10 (26x
/// smaller), step 974−590 2.08e-8.
#[test]
fn a_974_point_ks_run_beats_the_590_grid_error() {
    let e590 = pbe_energy(590);
    let e974 = pbe_energy(974);
    let e_ref = pbe_energy(5810);
    let err590 = (e590 - e_ref).abs();
    let err974 = (e974 - e_ref).abs();
    let step = (e974 - e590).abs();
    println!(
        "E(590) {e590:.12}  E(974) {e974:.12}  E(5810) {e_ref:.12}  \
         err(590) {err590:.3e}  err(974) {err974:.3e}  |E974-E590| {step:.3e}"
    );
    assert!(
        err974 < err590,
        "974's angular error {err974:.3e} Ha is not below 590's {err590:.3e}"
    );
    assert!(
        step < 2.0 * err590,
        "974 moved the energy {step:.3e} Ha from 590, more than twice 590's grid error {err590:.3e}"
    );
}
