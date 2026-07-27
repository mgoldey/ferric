//! Re-measure the Boys-screening crossover after the localizer sign fix.
//!
//! The original crossover numbers (benzene 1.8-4.2x SLOWER, crossover at
//! naphthalene) were measured while `boys_localize` was minimizing the Boys
//! functional, collapsing every centroid onto a single point. Distance screening
//! cannot discriminate under that degeneracy, so those numbers are void.
//!
//! Small basis (STO-3G) throughout to keep RAM low: this measures the SCALING
//! TREND of screened-vs-dense, which is a property of the pair topology, not of
//! the basis. Absolute timings are not comparable to the cc-pVDZ originals.
//!
//! # RUN THIS ON A QUIET BOX
//!
//! It is `#[ignore]`d and reports wall clocks, so it is only meaningful with
//! nothing else competing for CPU. The 2026-07-26 run was taken at load average
//! 25-35 against 6 competing jobs; its timings were recorded as UNTRUSTWORTHY and
//! the crossover point is currently UNKNOWN. Before quoting any ratio from this
//! harness, confirm the box is idle (`uptime`, `ps aux | grep ferric`).
//!
//! The NON-timing observations from that run are still good, because counts and
//! energies do not care about load: the screen's energy error grows smoothly and
//! monotonically with size, and `dist_cutoff` prunes essentially nothing even with
//! correct Boys centroids.

use ferric_core::{basis, mol::Molecule, parallel::ParallelContext};
use ferric_integrals::{basis_bridge::PreparedBasis, operator::Operator};
use ferric_rpa::config::{Chi0Sparsity, PdepRpaConfig};
use ferric_rpa::run_pdep_rpa;
use ferric_scf::{
    rhf::{solve_rhf, RhfConfig},
    screening::SchwarzBounds,
};
use std::time::Instant;

fn sweep_one(name: &str, path: &str) {
    let ctx = ParallelContext::default();
    let mol = match Molecule::load_xyz(path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("{name:12} SKIP (load: {e:?})");
            return;
        }
    };
    let obs = PreparedBasis::new(&mol, &basis::bundled("sto-3g").unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &obs).unwrap();
    let scf = RhfConfig { density_conv: 1e-8, max_iter: 200, ..Default::default() };

    let rhf = match solve_rhf(&ctx, &mol, &obs, Operator::coulomb(), &bounds, &scf) {
        Ok(r) if r.converged => r,
        Ok(r) => {
            eprintln!("{name:12} SKIP (SCF not converged, {} iters)", r.iterations);
            return;
        }
        Err(e) => {
            eprintln!("{name:12} SKIP (SCF: {e:?})");
            return;
        }
    };

    let natom = mol.atoms.len();
    let nocc = (mol.nelec() / 2) as usize;

    let run = |sparsity: Chi0Sparsity| -> Option<(f64, f64)> {
        let cfg = PdepRpaConfig { chi0_sparsity: sparsity, ..Default::default() };
        let t = Instant::now();
        match run_pdep_rpa(&mol, &obs, &dfbs, Operator::coulomb(), &rhf, &cfg) {
            Ok(r) => Some((t.elapsed().as_secs_f64(), r.e_rpa)),
            Err(e) => {
                eprintln!("{name:12} run failed: {e:?}");
                None
            }
        }
    };

    let Some((t_dense, e_dense)) = run(Chi0Sparsity::Dense) else { return };

    // Two screens, measured separately because only ONE of them was contaminated
    // by the Boys sign bug:
    //   * `thresh` gates on the (P|i_loc i_loc) Cauchy-Schwarz bound -- an
    //     integral-MAGNITUDE test, independent of where the centroids sit.
    //   * `dist_cutoff` is the G6 centroid-DISTANCE envelope -- exactly the thing
    //     that could not possibly prune when every centroid was collapsed onto one
    //     point, which is why "dist_cutoff prunes NOTHING" needs re-testing.
    let Some((t_bound, e_bound)) =
        run(Chi0Sparsity::BoysScreened { thresh: 1e-3, dist_cutoff: f64::INFINITY })
    else {
        return;
    };
    let Some((t_dist, e_dist)) =
        run(Chi0Sparsity::BoysScreened { thresh: 1e-3, dist_cutoff: 10.0 })
    else {
        return;
    };

    let tag = |t: f64| if t < t_dense { "FASTER" } else { "slower" };
    eprintln!(
        "{name:12} nat={natom:3} nocc={nocc:3} | dense {t_dense:7.3}s | \
         bound {t_bound:7.3}s {:5.2}x {:6} dE {:+.1e} | \
         +dist10 {t_dist:7.3}s {:5.2}x {:6} dE {:+.1e}",
        t_dense / t_bound.max(1e-9),
        tag(t_bound),
        e_bound - e_dense,
        t_dense / t_dist.max(1e-9),
        tag(t_dist),
        e_dist - e_dense
    );
}

#[test]
#[ignore = "timing sweep; run explicitly with --ignored"]
fn boys_screening_crossover_sto3g() {
    eprintln!("Boys-screening crossover, STO-3G/cc-pVDZ-RI, thresh=1e-3");
    eprintln!("(re-measured after the boys_localize sign fix)");
    for n in [2usize, 4, 6, 8, 10, 12, 14, 16] {
        sweep_one(
            &format!("alkane_{n}"),
            &format!("../../testdata/molecules/alkane_{n}.xyz"),
        );
    }
    sweep_one("benzene", "../../testdata/molecules/benzene.xyz");
}
