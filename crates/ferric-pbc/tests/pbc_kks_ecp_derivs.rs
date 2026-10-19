//! k-point Kohn-Sham (RKS, UKS, ROKS) and ROHF forces and stress on cells
//! whose basis carries an ECP (issue #353): the per-commit subset. The full
//! anchors (every route, both J/K sources, every force component, all nine
//! strains, ECP mutant sweeps), the hypotheses and the measured ledger are in
//! `validation_kks_ecp_derivs.rs`; the harness is `kks_ecp_support/mod.rs`.
//!
//! Here: RKS PBE on HI and UKS PBE on the H2I doublet (dense AFT, 1x1x2), one
//! force component and one strain against central FD of ferric's own k
//! energy, and the dropped-ECP mutant (`NoEcp`) of force and stress against
//! the same FD. Bars and step: `validation_kks_ecp_derivs.rs` (floor at h =
//! 2.5e-5: force <= 3.1e-10, dE/deps <= 3.4e-9; |F_ECP| ~6e-2, |dE/deps_ECP|
//! ~1.5e-1).

mod common;
#[path = "kks_ecp_support/mod.rs"]
mod support;

use ferric_pbc::kgrad::KGradMutation;
use ferric_pbc::kstress::KStressMutation;
use support::*;

const FD_H: f64 = 2.5e-5;
const FORCE_BAR: f64 = 3e-9;
const STRESS_BAR: f64 = 3e-8;

fn quick(route: Route, pos: &[[f64; 3]]) {
    let spec = Spec::new(route, Jk::Dense, [1, 1, 2]);
    let fx = fixture(pos, &spec);
    let (g, _) = force(&fx, None);
    let (s, _) = stress(&fx, None);
    let ia = pos.len() - 1;
    let fd_f = fd_force(pos, &spec, ia, 0, FD_H);
    let fd_s = fd_strain(&fx.cell, &spec, 2, 2, FD_H);
    let (gm, _) = force(&fx, Some(KGradMutation::NoEcp));
    let (sm, _) = stress(&fx, Some(KStressMutation::NoEcp));
    let mf = (g[(ia, 0)] - fd_f).abs();
    let ms = (s[2][2] - fd_s).abs();
    let mfm = (gm[(ia, 0)] - fd_f).abs();
    let msm = (sm[2][2] - fd_s).abs();
    eprintln!(
        "{route:?} 1x1x2: F(I,x) miss {mf:.2e} (NoEcp {mfm:.2e}); dE/de_zz miss {ms:.2e} \
         (NoEcp {msm:.2e})"
    );
    assert!(mf < FORCE_BAR, "{route:?} force {mf:e}");
    assert!(ms < STRESS_BAR, "{route:?} stress {ms:e}");
    assert!(
        mfm > 1e3 * FORCE_BAR,
        "NoEcp survives the force FD: {mfm:e}"
    );
    assert!(
        msm > 1e3 * STRESS_BAR,
        "NoEcp survives the stress FD: {msm:e}"
    );
}

#[test]
fn rks_pbe_ecp_force_and_stress_match_fd() {
    quick(Route::Rks("PBE"), &HI_POS);
}

#[test]
fn uks_pbe_ecp_force_and_stress_match_fd() {
    quick(Route::Uks("PBE"), &H2I_POS);
}

/// PROBE: FD step sweep, per route, for a force and two strain components.
#[test]
#[ignore = "probe: FD step sweep, 6 routes x 3 steps x 6 SCFs; measured 345 s release"]
fn probe_fd_steps() {
    let routes: Vec<(Route, &[[f64; 3]])> = vec![
        (Route::Rks("LDA"), &HI_POS),
        (Route::Rks("PBE"), &HI_POS),
        (Route::Rks("PBE0"), &HI_POS),
        (Route::Uks("PBE"), &H2I_POS),
        (Route::Roks("PBE"), &H2I_POS),
        (Route::Rohf, &H2I_POS),
    ];
    for (route, pos) in routes {
        let spec = Spec::new(route, Jk::Dense, [1, 1, 3]);
        let t0 = std::time::Instant::now();
        let fx = fixture(pos, &spec);
        let t1 = t0.elapsed().as_secs_f64();
        let (g, ge) = force(&fx, None);
        let (s, se) = stress(&fx, None);
        let t2 = t0.elapsed().as_secs_f64();
        eprintln!(
            "{route:?}: scf {t1:.1}s, derivs {:.1}s; |g| {:.3e} |g_ecp| {:.3e} |s| {:.3e} |s_ecp| {:.3e}",
            t2 - t1,
            amax(&g),
            amax(&ge),
            mmax(&s),
            mmax(&se)
        );
        let ia = pos.len() - 1;
        for h in [1e-4, 5e-5, 2.5e-5] {
            let f = fd_force(pos, &spec, ia, 0, h);
            let z = fd_strain(&fx.cell, &spec, 2, 2, h);
            let o = fd_strain(&fx.cell, &spec, 0, 1, h);
            eprintln!(
                "  h {h:.1e}: F(I,x) {:.2e}  S(zz) {:.2e}  S(xy) {:.2e}",
                (g[(ia, 0)] - f).abs(),
                (s[2][2] - z).abs(),
                (s[0][1] - o).abs()
            );
        }
        eprintln!("  total {:.1}s", t0.elapsed().as_secs_f64());
    }
}

/// PROBE: every strain component, three steps and three neighbour cutoffs:
/// separates an FD artifact of the hard neighbour mask (non-monotone in h,
/// moves with the cutoff) from a stress defect (stable in h and cutoff).
/// `FERRIC_PROBE_ROUTE=uks` probes UKS PBE on the H2I doublet instead of RKS
/// PBE on HI.
#[test]
#[ignore = "probe: 9 strains x 3 steps x 3 cutoffs (1x1x3); measured 272 s release (RKS)"]
fn probe_all_nine_strains() {
    let uks = std::env::var("FERRIC_PROBE_ROUTE").as_deref() == Ok("uks");
    let (route, pos): (Route, &[[f64; 3]]) = if uks {
        (Route::Uks("PBE"), &H2I_POS)
    } else {
        (Route::Rks("PBE"), &HI_POS)
    };
    for cutoff in [14.0, 16.0, 20.0] {
        let mut spec = Spec::new(route, Jk::Dense, [1, 1, 3]);
        spec.grid.neighbour_cutoff = Some(cutoff);
        let fx = fixture(pos, &spec);
        let (s, _) = stress(&fx, None);
        for (i, j) in ALL9 {
            let mut row = String::new();
            for h in [1.25e-5, 2.5e-5, 5e-5] {
                let d = fd_strain(&fx.cell, &spec, i, j, h);
                row += &format!(" h={h:.2e}:{:.2e}", (s[i][j] - d).abs());
            }
            eprintln!(
                "{route:?} cutoff {cutoff} eps({i},{j}) {:+.6e}:{row}",
                s[i][j]
            );
        }
    }
}
