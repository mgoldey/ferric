//! End-to-end cDFT identities on LiH/def2-SVP. No external reference — all
//! checks are exact identities or self-consistency conditions.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_dft::cdft::{population, Constraint, SpinChannel};
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::cdft_driver::solve_cdft_uhf;
use ferric_scf::rhf::RhfConfig;
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::solve_uhf;

fn setup() -> (Molecule, basis::BasisSet, PreparedBasis, Operator, SchwarzBounds) {
    let mol = Molecule::parse_xyz("2\nLiH\nLi 0 0 0\nH 0 0 1.60\n", 0, 1).unwrap();
    let bs = basis::bundled("def2-svp").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    (mol, bs, prep, op, bounds)
}

/// The fragment residual c(λ) = N_C(λ) − target is monotonic in λ — the
/// correct saddle-point sign. Sample three λ values via single inner solves
/// (Total channel adds +λW to both spins) and check N_C is monotonic in λ.
#[test]
fn residual_is_monotonic_in_lambda() {
    use ferric_dft::ao_grid::eval_basis_on_points;
    use ferric_dft::cdft::build_weight_matrix;
    use ferric_dft::grid::{build_atomic_grid, AtomicGridConfig};
    use ndarray::Array2;

    let (mol, bs, prep, op, bounds) = setup();
    let ctx = ParallelContext::default();

    // W^Li once on the 302-pt angular grid (matches the driver).
    let grid = build_atomic_grid(
        &mol,
        &AtomicGridConfig {
            n_radial: 99,
            n_angular: 302,
        },
    );
    let pts: Vec<[f64; 3]> = grid.iter().map(|g| g.xyz).collect();
    let chi = eval_basis_on_points(&mol, &bs, &pts).unwrap();
    let w = build_weight_matrix(&mol, &grid, &chi, &[0]);

    // N_Li at three increasing λ. Assert strict monotonicity.
    let mut pops = Vec::new();
    for &lam in &[-0.10_f64, 0.0, 0.10] {
        let wl = &w;
        let fm = move |f_a: &mut Array2<f64>, f_b: &mut Array2<f64>| {
            let lw = lam * wl;
            *f_a += &lw;
            *f_b += &lw;
        };
        let scf = ferric_scf::uhf::solve_uhf_fockmod(
            &ctx,
            &mol,
            &prep,
            op,
            &bounds,
            &RhfConfig::default(),
            None,
            Some(&fm),
        )
        .unwrap();
        let d_a = &scf.density_alpha;
        let d_b = scf.density_beta.as_ref().unwrap_or(d_a);
        pops.push(population(&w, d_a, d_b, &SpinChannel::Total));
    }
    // Strictly monotonic (one direction; sign depends on the convention above).
    let increasing = pops[0] < pops[1] && pops[1] < pops[2];
    let decreasing = pops[0] > pops[1] && pops[1] > pops[2];
    assert!(increasing || decreasing, "N_Li not monotonic in λ: {pops:?}");
}

/// The constraint is actually satisfied at convergence.
#[test]
fn charge_constraint_is_satisfied() {
    let (mol, bs, prep, op, bounds) = setup();
    let ctx = ParallelContext::default();
    let target = 2.2; // pull a bit of charge onto Li (Z=3)
    let cfg = RhfConfig {
        constraints: vec![Constraint {
            fragment: vec![0],
            spin: SpinChannel::Total,
            target,
        }],
        cdft_lambda_tol: 1e-5,
        ..Default::default()
    };
    let res = solve_cdft_uhf(&ctx, &mol, &prep, &bs, op, &bounds, &cfg).unwrap();
    assert!(
        (res.populations[0] - target).abs() < 1e-5,
        "Li pop {} vs target {target}",
        res.populations[0]
    );
    assert!(res.scf.converged, "inner SCF not converged");
}

/// λ = 0 (no constraint added) reproduces plain UHF. Exercises the
/// solve_uhf_fockmod plumbing directly: an empty closure must not change the
/// converged energy.
///
/// Note: this is *not* a bit-for-bit comparison. Two independent `solve_uhf`
/// runs on identical inputs already differ at the ULP level (~2e-15) because
/// the J/K builds reduce float sums under rayon in a nondeterministic order.
/// The meaningful "no behavior change" criterion is therefore agreement to the
/// SCF's own energy convergence (1e-10), which the empty closure trivially
/// preserves (it is a literal no-op).
#[test]
fn lambda_zero_equals_plain_uhf() {
    let (mol, _bs, prep, op, bounds) = setup();
    let ctx = ParallelContext::default();
    let plain = solve_uhf(&ctx, &mol, &prep, op, &bounds, &RhfConfig::default()).unwrap();
    let fm = |_fa: &mut ndarray::Array2<f64>, _fb: &mut ndarray::Array2<f64>| {};
    let modded = ferric_scf::uhf::solve_uhf_fockmod(
        &ctx,
        &mol,
        &prep,
        op,
        &bounds,
        &RhfConfig::default(),
        None,
        Some(&fm),
    )
    .unwrap();
    assert!(
        (plain.energy - modded.energy).abs() < 1e-10,
        "λ=0 fock_mod changed the energy: plain {} vs modded {}",
        plain.energy,
        modded.energy
    );
}
