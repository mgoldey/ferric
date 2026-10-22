//! Per-commit anchors of the range-separated k-point force and stress on
//! RS-GDF J/K (`ferric_pbc::kgrad`, `ferric_pbc::kstress`,
//! `rsgdf::kpoint::kderiv` module doc "Attenuated fit"). The Python spec is
//! `reference/pbc/rsgdf_rsh_force_proto.md`.
//!
//! # Hypotheses (written before measuring)
//!
//! `E_x^RSH = c_sr E_K[plain fit] + (c_lr − c_sr) E_K[attenuated fit] −
//! (v_M^mix / 2N_k) Σ tr[D S D S]`. The attenuated fit is the LR pass alone
//! at `ω` with `K = 0` dropped and no `c0`; its derivative is the plain fit's
//! LR and metric passes with `v_LR → v_ω`.
//!
//! * Physics: at `ω → ∞` with a finite cut covering the Coulomb tail, the
//!   attenuated fit IS the plain fit with K = 0 dropped, so energy and force
//!   agree to the build precision. Artifact: a `c0` wrongly kept or wrongly
//!   needed, a missing Hermitisation, a wrong kernel or a wrong metric weight
//!   shows as O(1e-3..1e-1) here (`c0 ∝ 1/ω²` itself is invisible at
//!   `ω = 1e7`: that blind spot is covered by the FD rows and the ledger).
//! * A q class the build left empty (no K inside the cut, or all metric
//!   eigenvalues under lindep with negligible weights) contributes exactly 0:
//!   the derivative must skip it, not error, and still match FD.
//!
//! # Measured (release test profile, 2026-10-22)
//!
//! | anchor | measured | bar |
//! |---|---|---|
//! | `ω = 1e7`, cut 24 vs plain fit: energy | 1.4e-15 | 1e-11 |
//! | same, force (`max|F|` 0.21) | 6.6e-14 | 1e-11 |
//! | ω = 0.11, empty q = 0 class (TRI2, lindep 1e-10), FD h = 1e-4 | 2.2e-14 (`max|F|` 1.75e-5) | 1e-11 |
//! | ω = 0.11, negligible q = 0 class (a = 6.5, lindep 1e-6, 3 K vectors), FD h = 1e-4 | 7.3e-13 (`max|F|` 6.7e-4) | 1e-11 |
//! | zero-XC CAM-type force, `Σ_A F_A` | ≤ 1.2e-15 | 1e-12 |
//!
//! MUTATION CHECKS of these tests (each run once against a deliberately
//! broken build, then reverted): the metric weight dropped in attenuated mode
//! misses the `ω → ∞` anchor by 8.7e-2; the empty-class skip removed makes
//! the empty-class test return the typed "every metric eigenvalue at q class
//! 0 is <= lindep" error. An EQUIVALENT mutant, recorded so nobody relies on
//! it: dropping the q = 0 `Z` Hermitisation in attenuated mode changes
//! nothing (6.6e-14 on the anchor), because the derivative contracts the
//! full `{G}` set, whose `dJ3` at q = 0 is already pair-Hermitian. The
//! production-wired mutants (`KGradMutation::RshFit*`, `RshMadelung*`,
//! `KStressMutation::RshFit*`) are below and in
//! `validation_krsh_rsgdf_force.rs`.
//!
//! The FD rows of the total force and stress, the default-lindep
//! characterisation (the analytic force is roundoff-limited when metric
//! eigenvalues near 1e-11 are kept, in the plain fit too; see #512) and the
//! full mutation ledger are weekly
//! (`validation_krsh_rsgdf_force.rs`).

mod common;
#[path = "krsh_rsgdf_force_support/mod.rs"]
mod support;

use ferric_pbc::dense_aft::ExxDiv;
use ferric_pbc::kgrad::KGradMutation;
use ferric_pbc::lattice::Cell;
use support::*;

fn big(pos: &[[f64; 3]]) -> Cell {
    Cell::new(common::hydrogens(pos), common::cubic(6.5)).unwrap()
}

/// EXACTNESS ANCHOR: the attenuated fit at `ω → ∞` (cut 24) is the plain
/// Coulomb fit with K = 0 dropped (module doc).
#[test]
fn attenuated_force_is_the_plain_fit_force_as_omega_goes_to_infinity() {
    let s = sys(tri2(&TRI2_ATOMS));
    let cfg = gdf_cfg(LINDEP_DEFAULT, ExxDiv::None);
    let plain = plain_fit(&s, &cfg);
    let d = fixed_density(&s, &plain);
    let (w, gcut) = (1e7, 24.0);
    let att = att_fit(&s, &cfg, w, Some(gcut));
    let (ep, ea) = (ex_energy(&plain, &d), ex_energy(&att, &d));
    let fp = fit_force(&s, &plain, &cfg, &d, None).unwrap();
    let fa = fit_force(&s, &att, &cfg, &d, Some((w, gcut))).unwrap();
    let dd = diff(&fp, &fa);
    eprintln!(
        "w -> inf: E plain {ep:.14} att {ea:.14} (diff {:.2e}); max|F| {:.3e}, force diff {dd:.2e}",
        ea - ep,
        max_abs(&fp),
    );
    assert!(max_abs(&fp) > 1e-2);
    assert!((ea - ep).abs() < 1e-11, "{:e}", ea - ep);
    assert!(dd < 1e-11, "{dd:e}");
}

/// The fixed-density attenuated force at `ω = 0.11` against central FD of
/// the fixed-density attenuated energy (h = 1e-4, three components), after
/// asserting that the q = 0 class is the expected skipped one (`nK` K
/// vectors inside the cut, nothing kept).
fn skipped_class_row(
    tag: &str,
    cell_of: &dyn Fn(&[[f64; 3]]) -> Cell,
    pos: &[[f64; 3]],
    lindep: f64,
    nk_q0_nonzero: bool,
) {
    let w = 0.11;
    let s = sys(cell_of(pos));
    let cfg = gdf_cfg(lindep, ExxDiv::None);
    let plain = plain_fit(&s, &cfg);
    let d = fixed_density(&s, &plain);
    let att = att_fit(&s, &cfg, w, None);
    let q0 = &att.stats().per_q[0];
    eprintln!(
        "{tag}: q = 0 class: {} K vectors, {} kept; q = 1: {} K vectors, {} kept",
        q0.n_k_vectors,
        q0.naux_kept,
        att.stats().per_q[1].n_k_vectors,
        att.stats().per_q[1].naux_kept
    );
    assert_eq!(q0.naux_kept, 0);
    assert_eq!(q0.n_k_vectors > 0, nk_q0_nonzero);
    assert!(
        att.stats().per_q[1].naux_kept > 0,
        "the q != 0 class must carry the force"
    );
    let g = fit_force(&s, &att, &cfg, &d, Some((w, prod_gcut(&cfg, w)))).unwrap();
    let h = 1e-4;
    let mut worst = 0.0_f64;
    for (a, x) in [(0, 0), (0, 2), (1, 1)] {
        let ep = att_energy_at(cell_of, &moved(pos, a, x, h), &cfg, w, &d);
        let em = att_energy_at(cell_of, &moved(pos, a, x, -h), &cfg, w, &d);
        worst = worst.max(((ep - em) / (2.0 * h) - g[(a, x)]).abs());
    }
    eprintln!("{tag}: max|F| {:.3e}, FD miss {worst:.2e}", max_abs(&g));
    assert!(max_abs(&g) > 1e-7);
    assert!(worst < 1e-11, "{tag}: {worst:e}");
}

/// The empty-class skip (HSE06's `ω = 0.11`): on TRI2 the q = 0 class has no
/// K vector inside the cut (|G|min ≈ 1.3 > 1.2); on the a = 6.5 cube at
/// lindep 1e-6 it has K vectors whose kernel weights are under lindep.
#[test]
fn empty_and_negligible_attenuated_classes_contribute_nothing() {
    skipped_class_row(
        "TRI2 empty",
        &|p| tri2(p),
        &TRI2_ATOMS,
        LINDEP_DEFAULT,
        false,
    );
    skipped_class_row("a = 6.5 negligible", &big, &H2_BIG, LINDEP_FD, true);
}

/// Wiring of the RS-GDF RSH force and stress on the zero-XC CAM-type SCF
/// (TRI2, lindep 1e-6; FD-validated weekly): the attenuated piece is live,
/// the forces sum to zero, and each production-wired mutant moves the force
/// or stress far beyond the weekly FD floor (~3e-9; full ledger in the
/// validation file), the `ω_0` one is refused by the rebuilt-B check.
#[test]
fn zero_xc_rsh_force_and_stress_wiring_and_mutants() {
    use ferric_pbc::kstress::KStressMutation as SM;
    let z = zx_run(
        tri2(&TRI2_ATOMS),
        rsh_params(0.19, 0.65, 0.33),
        LINDEP_FD,
        ExxDiv::Ewald,
    );
    let g = zx_force_m(&z, None).unwrap();
    let att = max_abs(&g.hf.parts.fit_attenuated);
    eprintln!(
        "CAM-type zero-XC: max|F| {:.3e}, max|F_att| {att:.3e}, net {:.2e}",
        max_abs(&g.grad),
        g.hf.net_force
    );
    assert!(att > 1e-3);
    assert!(g.hf.net_force < 1e-12, "{:e}", g.hf.net_force);
    for m in [
        KGradMutation::RshFitKeepC0,
        KGradMutation::RshFitNoScale,
        KGradMutation::RshMadelungScaled,
    ] {
        let miss = diff(&zx_force_m(&z, Some(m)).unwrap().grad, &g.grad);
        eprintln!("  {m:?}: force miss {miss:.2e}");
        assert!(miss > 1e-4, "{m:?}: {miss:e}");
    }
    let e = zx_force_m(&z, Some(KGradMutation::RshFitOmega0))
        .expect_err("a derivative at the Ewald split must be refused")
        .to_string();
    assert!(e.contains("pass the config the build used"), "{e}");
    let st = zx_stress_m(&z, None).unwrap();
    assert!(mmax(&st.hf.parts.fit_attenuated) > 1e-3);
    for m in [SM::RshFitNoScale, SM::RshFitNoAttenuated] {
        let miss = mdiff(&zx_stress_m(&z, Some(m)).unwrap().de_deps, &st.de_deps);
        eprintln!("  {m:?}: stress miss {miss:.2e}");
        assert!(miss > 1e-3, "{m:?}: {miss:e}");
    }
}
