//! Per-commit conditioning anchor of the k-point RS-GDF fit derivative
//! (`rsgdf::kpoint::kderiv` module doc "Conditioning", #512): at FIXED
//! density a rigid translation must leave the fit force unchanged, also when
//! metric eigenvalues just above the production lindep (1e-10) are kept.
//!
//! # Hypotheses (written before measuring)
//!
//! * Physics: the force of a translation-invariant energy is translation
//!   invariant; the only allowed drift is the roundoff of the integrals
//!   themselves, amplified by the fit's conditioning (`ε/√s_min` in the
//!   energy's own `B = s^{-1/2} U^H J3`).
//! * Artifact (the #512 defect, prototyped in Python on an exact-integral
//!   s-only model, `prototypes/pbc/kgrad_conditioning.py`): forming
//!   `f(J2) = U diag(1/s) U^H` explicitly spreads `ε/s_min` into every entry
//!   of `Z` and `c`, and forming `H` in the aux basis before rotating loses
//!   `ε‖H‖` that `f_i f_j` amplifies by `1/s_min²`. Either shows here as a
//!   drift of 1e-8..1e-6 that a lindep of 1e-6 removes.
//!
//! # Fixtures
//!
//! TRI2 H2 / STO-3G, 1×1×2, at the production lindep 1e-10, fixed k-RHF
//! density, rigid translation `1e-3 (1, 0.7, −0.4)`:
//! * the ET-sp aux plus a near-duplicate s shell (`0.75 (1 + 3e-4)`): kept
//!   metric eigenvalues 1.28e-10 (q = 0) and 2.87e-10 (q = 1), nothing
//!   dropped; the plain Coulomb fit and the CAM-type attenuated fit
//!   (`ω = 0.33`) on it;
//! * the plain ET-sp aux, CAM-type attenuated fit (the #512 report's row).
//!
//! # Measured (test profile, 2026-10-23)
//!
//! | fixture | before (explicit `f(J2)`, `H` rotated late) | now | bar |
//! |---|---|---|---|
//! | near-duplicate, plain fit | 9.2e-7 | 1.3e-10 | 2e-9 |
//! | near-duplicate, attenuated | 1.05e-7 | 1.8e-10 | 2e-9 |
//! | ET-sp, attenuated | 3.7e-8 | 3.6e-11 | 5e-10 |
//!
//! #512's acceptance target (translation drift <= 1e-10 at the default
//! lindep) is MET on the ET-sp attenuated row and NOT MET on the
//! near-duplicate fixture: 1.29e-10 (plain) / 1.79e-10 (attenuated). That
//! fixture's ENERGY also moves under the same translation (dE = 5.4e-11, the
//! same before and after: the derivative code cannot touch it), and FD of the
//! plain fit grows as 1/h (6.06e-8 / 5.47e-7 / 4.92e-6 at h = 1e-3 / 1e-4 /
//! 1e-5; before: 6.20e-7 / 6.30e-7 / 4.91e-6), so FD cannot check this force
//! below ~5e-8. Reading the residual as the integral-roundoff floor of the
//! near-singular fit is PROVISIONAL for Rust: it rests on the exact-integral
//! Python model (whose floor at `s_min` 9e-11 is 7e-10) plus the moving
//! energy, not on an exact-integral Rust reference. With a better conditioned
//! duplicate (`1 + 1e-3`, kept `s_min` 1.4e-9) the drift is 9.6e-12 / 6.4e-11
//! (before: 4.7e-8 / 2.0e-8). At lindep 1e-6 (near-null modes dropped) the
//! drifts are 1e-15..8e-13 before and after.
//!
//! The #512 report's own control (ET-sp plus `0.75 (1 + 1e-4)`, smallest
//! metric eigenvalue 1.42e-11 / 3.19e-11 at q = 0 / 1, DROPPED at lindep
//! 1e-10, 2 dropped per class), measured through this file's exchange-only
//! fixed-density entry point (not a per-commit row):
//!
//! | fit | translation drift before / now | FD miss h = 1e-3 / 1e-4 / 1e-5, before | now |
//! |---|---|---|---|
//! | plain | 4.8e-14 / 8.6e-16 | 2.30e-8 / 2.29e-10 / 2.51e-11 | the same |
//! | attenuated | 2.10e-8 / 4.9e-11 | 3.29e-8 / 3.86e-8 / 3.87e-8 (flat) | 5.72e-9 / 1.90e-11 / 7.8e-11 |
//!
//! (energy translation drift 1.1e-16 both). The report's flat ~4e-6 FD miss
//! of the PLAIN fit is therefore not reproduced on this path, before or
//! after: its exchange-only force was already clean. Which path produced the
//! report's number is not recorded; this entry point zeroes the q = 0
//! Coulomb `ρρ^H` term, and that term's conditioning is not re-measured here.
//!
//! Complex q: both classes of this 1×1×2 mesh are time-reversal invariant
//! (real metric), so a dropped or misplaced conjugation is invisible here;
//! it is guarded by `pbc_kgrad.rs`'s 1×1×3 RS-GDF rows.
//!
//! MUTATION CHECKS (each run once against a deliberately broken build, then
//! reverted; rows near-duplicate plain / attenuated / ET-sp attenuated): the
//! pre-#512 code (explicit `f(J2)`, `H` rotated late) misses 9.2e-7 / 1.05e-7
//! / 3.7e-8 (the "before" column); restoring ONLY the explicit `f(J2)` in
//! `Z` misses 1.4e-6 / 1.05e-7 / 8.0e-10; restoring ONLY the late rotation
//! of `H` misses 2.8e-6 / 6.7e-9 / 9.0e-9. Every mutant fails the plain row
//! by >= 700x its bar. This fixture is exchange only (zero total density):
//! the q = 0 Coulomb `c = U (f ∘ U^H ρ)` is covered by the RS-GDF force FD
//! tests (`pbc_kgrad.rs`, `pbc_kstress.rs`), not here.

mod common;
#[path = "krsh_rsgdf_force_support/mod.rs"]
mod support;

use ferric_core::basis::BasisSet;
use ferric_pbc::dense_aft::ExxDiv;
use support::*;

/// Max change of the fixed-density fit force under a rigid translation of
/// 1e-3, for the plain fit (`att = None`) and the attenuated fit at `ω`.
fn translation_drift(aux: &BasisSet, atts: &[Option<f64>]) -> Vec<f64> {
    let t = 1e-3;
    let cfg = gdf_cfg(LINDEP_DEFAULT, ExxDiv::None);
    let s = sys_aux(tri2(&TRI2_ATOMS), aux);
    let st = sys_aux(tri2(&translated(&TRI2_ATOMS, [t, 0.7 * t, -0.4 * t])), aux);
    let plain = plain_fit(&s, &cfg);
    let d = fixed_density(&s, &plain);
    atts.iter()
        .map(|att| {
            let (f0, f1) = match *att {
                None => (
                    fit_force(&s, &plain, &cfg, &d, None).unwrap(),
                    fit_force(&st, &plain_fit(&st, &cfg), &cfg, &d, None).unwrap(),
                ),
                Some(w) => {
                    let gc = Some((w, prod_gcut(&cfg, w)));
                    (
                        fit_force(&s, &att_fit(&s, &cfg, w, None), &cfg, &d, gc).unwrap(),
                        fit_force(&st, &att_fit(&st, &cfg, w, None), &cfg, &d, gc).unwrap(),
                    )
                }
            };
            assert!(max_abs(&f0) > 1e-2, "the fit force must be live");
            diff(&f0, &f1)
        })
        .collect()
}

#[test]
fn fit_force_is_translation_invariant_with_near_singular_metric() {
    let aux = et_sp_dup_aux(3e-4);
    {
        // The fixture really keeps near-null modes at the production cut.
        let s = sys_aux(tri2(&TRI2_ATOMS), &aux);
        let plain = plain_fit(&s, &gdf_cfg(LINDEP_DEFAULT, ExxDiv::None));
        let smin = plain
            .stats()
            .per_q
            .iter()
            .map(|q| {
                assert_eq!(q.n_dropped, 0);
                q.metric_eig_min
            })
            .fold(f64::INFINITY, f64::min);
        assert!(smin > LINDEP_DEFAULT && smin < 1e-9, "kept s_min {smin:e}");
    }
    let dup = translation_drift(&aux, &[None, Some(0.33)]);
    let etsp = translation_drift(&et_sp_aux(), &[Some(0.33)]);
    eprintln!(
        "translation drift: near-duplicate plain {:.2e}, attenuated {:.2e}; ET-sp attenuated {:.2e}",
        dup[0], dup[1], etsp[0]
    );
    assert!(dup[0] < 2e-9, "near-duplicate plain fit: {:e}", dup[0]);
    assert!(dup[1] < 2e-9, "near-duplicate attenuated fit: {:e}", dup[1]);
    assert!(etsp[0] < 5e-10, "ET-sp attenuated fit: {:e}", etsp[0]);
}
