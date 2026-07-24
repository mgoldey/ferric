//! Validation for the opt-in NWChem-style angular grid pruning
//! (`ferric_dft::prune`, `grid::build_atomic_grid_pruned`).
//!
//! Three things are checked, in increasing strength:
//!
//! 1. **Point-count reduction** — pruning must actually remove a meaningful
//!    fraction of the grid, on molecules with a real mix of elements.
//! 2. **∫ρ dV = N_electrons** — the standard grid-quality check. A
//!    weight-normalisation bug (the main correctness risk when the Lebedev
//!    order changes between radial shells) shows up here immediately and
//!    loudly, because the flat and pruned grids would then disagree by a
//!    large factor rather than by quadrature noise.
//! 3. **E_xc on a fixed converged density** — the grid-sensitive part of the
//!    DFT total energy, evaluated on the same SCF density with the flat and
//!    pruned grids. This is the quantity that would shift the total energy
//!    away from the PySCF reference if pruning were too aggressive.
//!
//! Why E_xc rather than a full pruned-grid SCF: `build_atomic_grid_pruned` is
//! deliberately not wired into `ferric-scf`'s KS driver yet (pruning ships
//! opt-in at the grid layer; making it the SCF default is a separate decision
//! after this accuracy data is in). Re-evaluating E_xc on a fixed converged
//! density isolates the grid's contribution exactly and is a *conservative*
//! proxy: in a real pruned SCF the density would relax variationally against
//! the pruned grid, which can only lower the discrepancy.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_dft::ao_grid::eval_basis_and_grad_on_points;
use ferric_dft::density_on_grid::eval_density_closed;
use ferric_dft::grid::{build_atomic_grid, build_atomic_grid_pruned, AtomicGridConfig, GridPoint};
use ferric_dft::libxc::xc_def_from_name;
use ferric_dft::prune::PruneScheme;
use ferric_dft::vxc::semilocal_vxc_closed;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

const H2O: &str = "3\nH2O\nO 0 0 0\nH 0 0.7572 0.5868\nH 0 -0.7572 0.5868\n";
const CH4: &str = "5\nCH4\nC 0 0 0\n\
                   H 0.6276 0.6276 0.6276\n\
                   H -0.6276 -0.6276 0.6276\n\
                   H -0.6276 0.6276 -0.6276\n\
                   H 0.6276 -0.6276 -0.6276\n";
const BENZENE: &str = "12\nC6H6\n\
    C  0.0000  1.3970  0.0000\n\
    C  1.2098  0.6985  0.0000\n\
    C  1.2098 -0.6985  0.0000\n\
    C  0.0000 -1.3970  0.0000\n\
    C -1.2098 -0.6985  0.0000\n\
    C -1.2098  0.6985  0.0000\n\
    H  0.0000  2.4810  0.0000\n\
    H  2.1487  1.2405  0.0000\n\
    H  2.1487 -1.2405  0.0000\n\
    H  0.0000 -2.4810  0.0000\n\
    H -2.1487 -1.2405  0.0000\n\
    H -2.1487  1.2405  0.0000\n";

/// Grid point counts for `mol` at `cfg`, flat vs pruned.
fn point_counts(mol: &Molecule, cfg: &AtomicGridConfig) -> (usize, usize) {
    let flat = build_atomic_grid(mol, cfg);
    let pruned = build_atomic_grid_pruned(mol, cfg, Some(PruneScheme::NwchemLike)).unwrap();
    (flat.len(), pruned.len())
}

#[test]
fn point_count_reduction_is_in_the_expected_range() {
    let cfg = AtomicGridConfig::default(); // 75 x 110
    for (label, xyz) in [("H2O", H2O), ("CH4", CH4), ("C6H6", BENZENE)] {
        let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
        let (flat, pruned) = point_counts(&mol, &cfg);
        let saved = 100.0 * (1.0 - pruned as f64 / flat as f64);
        eprintln!("[{label}] 75x110: flat {flat} pts -> pruned {pruned} pts ({saved:.1}% fewer)");
        assert!(
            (20.0..30.0).contains(&saved),
            "{label}: pruning saved {saved:.1}%, outside the expected 20-30% band"
        );
    }

    // Also exercise the finer 99x302 grid, where the region table snaps to
    // [26, 110, 302, 302, 302] -- the coarse core is a bigger relative win at
    // 302, hence the larger reduction.
    let big = AtomicGridConfig { n_radial: 99, n_angular: 302, ..Default::default() };
    let mol = Molecule::parse_xyz(H2O, 0, 1).unwrap();
    let (flat, pruned) = point_counts(&mol, &big);
    let saved = 100.0 * (1.0 - pruned as f64 / flat as f64);
    eprintln!("[H2O] 99x302: flat {flat} pts -> pruned {pruned} pts ({saved:.1}% fewer)");
    assert!(saved > 20.0, "99x302 pruning saved only {saved:.1}%");
}

struct Case {
    label: &'static str,
    n_elec: f64,
    /// (flat, pruned) grids and the converged density evaluated on each.
    flat: (Vec<GridPoint>, ferric_dft::density_on_grid::DensityGrid),
    pruned: (Vec<GridPoint>, ferric_dft::density_on_grid::DensityGrid),
}

fn prepare(label: &'static str, xyz: &str, basis_name: &str, n_elec: f64) -> Case {
    let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
    let bs = basis::bundled(basis_name).unwrap();
    let obs = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let ctx = ParallelContext::default();
    let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default()).unwrap();

    let cfg = AtomicGridConfig::default();
    let mut out = Vec::new();
    for grid in [
        build_atomic_grid(&mol, &cfg),
        build_atomic_grid_pruned(&mol, &cfg, Some(PruneScheme::NwchemLike)).unwrap(),
    ] {
        let pts: Vec<[f64; 3]> = grid.iter().map(|g| g.xyz).collect();
        let (chi, dchi) = eval_basis_and_grad_on_points(&mol, &bs, &pts).unwrap();
        let dens = eval_density_closed(&rhf.density_total, &chi, &dchi);
        // Keep chi/dchi alive for the E_xc pass by recomputing there; storing
        // them here would balloon the test's peak memory on benzene.
        out.push((grid, dens));
    }
    let pruned = out.pop().unwrap();
    let flat = out.pop().unwrap();
    Case { label, n_elec, flat, pruned }
}

fn integrate_rho(grid: &[GridPoint], dens: &ferric_dft::density_on_grid::DensityGrid) -> f64 {
    grid.iter().zip(dens.rho.iter()).map(|(g, &r)| g.weight * r).sum()
}

#[test]
fn pruned_grid_integrates_rho_to_the_electron_count() {
    // THE grid-quality check. A per-shell weight-normalisation bug would show
    // up here as an O(1) error, not the ~1e-5 quadrature noise we tolerate.
    for case in [
        prepare("H2O/cc-pVDZ", H2O, "cc-pvdz", 10.0),
        prepare("CH4/cc-pVDZ", CH4, "cc-pvdz", 10.0),
    ] {
        let n_flat = integrate_rho(&case.flat.0, &case.flat.1);
        let n_pruned = integrate_rho(&case.pruned.0, &case.pruned.1);
        let e_flat = (n_flat - case.n_elec).abs();
        let e_pruned = (n_pruned - case.n_elec).abs();
        eprintln!(
            "[{}] Ne: flat {n_flat:.9} (err {e_flat:.2e}), pruned {n_pruned:.9} (err {e_pruned:.2e})",
            case.label
        );
        // Absolute bar. Note CH4's flat 75x110 grid is itself only ~4e-4
        // accurate in Ne -- that is a pre-existing property of ferric's flat
        // grid, NOT something pruning introduces (see the relative check
        // below, where pruned == flat to ~1e-12). Do not tighten this past
        // what the flat grid already delivers.
        assert!(
            e_pruned < 1e-3,
            "{}: pruned grid recovers {n_pruned} electrons, expected {} (err {e_pruned:.2e})",
            case.label,
            case.n_elec
        );
        // The real invariant: pruning must not degrade the electron count
        // relative to the flat grid it replaces. A per-shell weight bug would
        // break this by orders of magnitude even where the absolute bar above
        // is loose.
        let drift = (n_pruned - n_flat).abs();
        assert!(
            drift < 1e-6,
            "{}: pruning shifted Ne by {drift:.2e} relative to the flat grid",
            case.label
        );
    }
}

/// E_xc on a fixed converged density, flat grid vs pruned grid.
fn exc_delta(label: &str, xyz: &str, basis_name: &str, xc_name: &str) -> f64 {
    let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
    let bs = basis::bundled(basis_name).unwrap();
    let obs = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let ctx = ParallelContext::default();
    let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default()).unwrap();
    let xc = xc_def_from_name(xc_name).unwrap();
    let cfg = AtomicGridConfig::default();

    let eval = |grid: Vec<GridPoint>| -> f64 {
        let pts: Vec<[f64; 3]> = grid.iter().map(|g| g.xyz).collect();
        let (chi, dchi) = eval_basis_and_grad_on_points(&mol, &bs, &pts).unwrap();
        let dens = eval_density_closed(&rhf.density_total, &chi, &dchi);
        semilocal_vxc_closed(&grid, &chi, &dchi, &dens, None, &xc).0
    };
    let e_flat = eval(build_atomic_grid(&mol, &cfg));
    let e_pruned =
        eval(build_atomic_grid_pruned(&mol, &cfg, Some(PruneScheme::NwchemLike)).unwrap());
    let delta = e_pruned - e_flat;
    eprintln!(
        "[{label}/{xc_name}] E_xc flat = {e_flat:.10} Ha, pruned = {e_pruned:.10} Ha, \
         delta = {delta:+.3e} Ha"
    );
    delta
}

#[test]
fn pruned_exc_matches_flat_grid_to_dft_reference_tolerance() {
    // The DFT reference tests in ferric-scf gate at 2e-5 Ha (PBE) / 5e-5 Ha
    // (B3LYP) against PySCF. Pruning's own contribution must be comfortably
    // inside that so it cannot be the thing that breaks a reference test.
    //
    // The measured worst case with the shipped region table is ~8e-10 Ha, so
    // the bar is set at 1e-8 Ha -- roughly 1000x tighter than the reference
    // tests need and ~100x above the observed noise. It is deliberately NOT
    // set at the 1e-5 Ha "would be acceptable" level: an earlier region table
    // that coarsened the outermost region cost 4.7e-5 Ha on CH4/PBE, and this
    // assertion is what caught it. Widening this test is not the correct
    // response to it failing.
    const TOL: f64 = 1e-8;
    for (label, xyz, xc) in [
        ("H2O", H2O, "LDA"),
        ("H2O", H2O, "PBE"),
        ("H2O", H2O, "B3LYP"),
        ("CH4", CH4, "PBE"),
        // Benzene is the perf motivation (12 atoms, 99k flat grid points, each
        // paying the O(natoms^2) Becke cost) so it must also be the accuracy
        // check -- pruning errors that cancel on 3-atom systems need not.
        ("C6H6", BENZENE, "PBE"),
    ] {
        let d = exc_delta(label, xyz, "cc-pvdz", xc).abs();
        assert!(
            d < TOL,
            "{label}/{xc}: pruning shifts E_xc by {d:.3e} Ha, above the {TOL:.0e} Ha bar"
        );
    }
}

/// Wall-clock for the grid-construction stage that pruning actually targets:
/// Becke weights (O(natoms^2) per point) plus AO-on-grid evaluation.
///
/// Reported, not asserted — timing on a shared box is not a stable gate. Run
/// with `--nocapture` to see the numbers.
#[test]
fn report_grid_pipeline_timing_benzene() {
    use std::time::Instant;

    let mol = Molecule::parse_xyz(BENZENE, 0, 1).unwrap();
    let bs = basis::bundled("cc-pvdz").unwrap();
    let cfg = AtomicGridConfig::default();

    let time_it = |prune: Option<PruneScheme>| -> (usize, f64, f64) {
        let t0 = Instant::now();
        let grid = build_atomic_grid_pruned(&mol, &cfg, prune).unwrap();
        let t_grid = t0.elapsed().as_secs_f64();
        let pts: Vec<[f64; 3]> = grid.iter().map(|g| g.xyz).collect();
        let t1 = Instant::now();
        let _ = eval_basis_and_grad_on_points(&mol, &bs, &pts).unwrap();
        let t_ao = t1.elapsed().as_secs_f64();
        (grid.len(), t_grid, t_ao)
    };

    // Warm up so the first-touch allocator cost does not land on the flat run.
    let _ = time_it(None);

    let (n_flat, g_flat, a_flat) = time_it(None);
    let (n_pruned, g_pruned, a_pruned) = time_it(Some(PruneScheme::NwchemLike));

    eprintln!(
        "[C6H6/cc-pVDZ 75x110 timing]\n  \
         flat  : {n_flat:6} pts  grid(Becke) {g_flat:.3} s  AO+grad {a_flat:.3} s  total {:.3} s\n  \
         pruned: {n_pruned:6} pts  grid(Becke) {g_pruned:.3} s  AO+grad {a_pruned:.3} s  total {:.3} s\n  \
         speedup: grid {:.2}x, AO {:.2}x, combined {:.2}x",
        g_flat + a_flat,
        g_pruned + a_pruned,
        g_flat / g_pruned,
        a_flat / a_pruned,
        (g_flat + a_flat) / (g_pruned + a_pruned),
    );
}
