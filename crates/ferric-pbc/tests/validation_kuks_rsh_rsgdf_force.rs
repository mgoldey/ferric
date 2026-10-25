//! Weekly validation of the OPEN-SHELL (UKS, ROKS) range-separated k-point
//! force on RS-GDF J/K. Every test here is `#[ignore]`d; the per-commit
//! subset (singlet UKS == RKS, equal fractions, one FD row with the ledger,
//! the stress refusal) is `pbc_kuks_rsh_rsgdf_force.rs`, whose header carries
//! the derivation and the hypotheses.
//!
//! Systems: H / STO-3G, ET-sp aux, 1x1x2 mesh, RS-GDF split `ω_0 = 1`, hcore
//! `ω = 0.8`: the H3 doublet `H3D` in `TRI_A` (CAM-type `ω = 0.33`, lindep
//! 1e-6, the closed-shell RS-GDF file's conditioning; KS rows CAM-B3LYP at
//! lindep 1e-6 and HSE06 at the production 1e-10, grid (20, 50) with
//! neighbour cutoff 14, the dense UKS file's) and the same H3 in a 6.5 Bohr
//! cube (HSE-type `ω = 0.11`, lindep 1e-10: only there does the q = 0 class
//! keep K vectors inside the cut, so `RshFitKeepC0` / `RshFitNoScale` are
//! live for an HSE separation).
//!
//! MEASURED (test profile, OPENBLAS/RAYON = 1, 2026-10-25):
//!
//! * Zero-XC FD, all 9 components, h = 2e-4 / 1e-4 / 5e-5 (ratio per
//!   halving): CAM-type TRI_A UKS 1.56e-8 / 3.95e-9 / 9.67e-10 (3.94), ROKS
//!   1.62e-8 / 4.10e-9 / 1.00e-9 (3.94); HSE-type a = 6.5 UKS 9.92e-9 /
//!   2.53e-9 / 7.09e-10 (3.93), ROKS 9.95e-9 / 2.53e-9 / 7.07e-10 (3.93): the
//!   h^2 truncation. `max|F|` 6.1e-2 / 1.95e-1, `max|F_att|` 1.3e-2 / 7.0e-5,
//!   `Σ_A F_A` <= 2.8e-15. Bars 1e-8 at h = 5e-5, ratio in 3..6.
//! * Equal fractions == global hybrid (`c = 0.4`): UKS ewald 3.6e-14, ROKS
//!   ewald 1.9e-14, both under exxdiv none exactly 0; `|ΔE|` <= 2.2e-16.
//! * KS FD (3 components, h = 1e-4 / 5e-5): CAM-B3LYP UKS 3.80e-9 / 8.90e-10,
//!   ROKS 4.52e-9 / 1.10e-9; HSE06 UKS 3.78e-9 / 9.24e-10, ROKS 4.48e-9 /
//!   1.12e-9. Bar 2e-8 at h = 5e-5. HSE06's attenuated piece is nearly dead
//!   on TRI_A (`max|F_att|` 2.7e-6), so the HSE06 KS row tests the plain fit
//!   and the mixed constant; the HSE-type zero-XC rows on the 6.5 cube test
//!   the attenuated fit at `ω = 0.11`.
//! * RS-GDF − dense UKS force gap on the same cell / grid: CAM-B3LYP 1.26e-5,
//!   HSE06 1.08e-5 vs PBE0 1.08e-5 (ratios 1.17 / 1.00), the same at lindep
//!   1e-6 and 1e-10. Bar 1.5 x the PBE0 gap.
//!
//! MUTATION LEDGER (max |F_mut − F| vs the production force, Ha/Bohr; bar
//! 1e-5, FD floor ~4e-9 at h = 1e-4):
//!
//! | mutant | CAM-type UKS | CAM-type ROKS | HSE-type UKS | HSE-type ROKS |
//! |---|---|---|---|---|
//! | ExchDropBeta | 2.08e-2 | 2.15e-2 | 2.13e-2 | 2.17e-2 |
//! | ExchBetaUsesAlpha | 3.06e-3 | 3.89e-3 | 1.56e-2 | 1.59e-2 |
//! | ExchClosedShellFactor | 1.93e-2 | 1.96e-2 | 1.35e-2 | 1.38e-2 |
//! | MTermTotalDensity | 1.15e-2 | 1.15e-2 | 4.97e-3 | 4.96e-3 |
//! | PulayKSwap | 2.78e-2 | 2.83e-2 | 2.86e-2 | 2.88e-2 |
//! | RshFitNoScale | 1.54e-2 | 1.57e-2 | 3.50e-4 | 3.60e-4 |
//! | RshFitKeepC0 | 1.68e-2 | 1.69e-2 | 7.82e-3 | 7.87e-3 |
//! | RshMadelungUnmixed | 2.73e-2 | 2.73e-2 | 2.15e-3 | 2.16e-3 |
//! | RshMadelungScaled | 3.54e-2 | 3.55e-2 | 2.04e-3 | 2.05e-3 |
//! | RshPlainMadelung | 4.29e-2 | 4.29e-2 | 1.68e-2 | 1.69e-2 |
//! | RoWDropCross | - | 3.31e-2 | - | 1.80e-2 |
//!
//! SOURCE MUTANTS (production `KSpins::exch_terms` edited, per-commit file
//! run, then reverted): the closed-shell factor `(½, D_α), (½, D_β)` fails
//! the singlet anchor and FD (miss 1.93e-2) while the equal-fraction anchor
//! PASSES (both sides share the term list: blind by construction); dropping
//! the β term fails all three (FD miss 2.08e-2).

mod common;
#[path = "krsh_rsgdf_force_support/mod.rs"]
mod rs;
#[path = "kuks_rsh_rsgdf_force_support/mod.rs"]
mod support;

use ferric_pbc::dense_aft::ExxDiv;
use ferric_pbc::kgrad::KGradJk;
use ferric_pbc::rsh::RshParams;
use support::*;

fn all_comps(n: usize) -> Vec<(usize, usize)> {
    (0..n).flat_map(|a| (0..3).map(move |x| (a, x))).collect()
}

/// Worst |analytic − FD| of a zero-XC fixture over `comps` at step `h`.
fn zx_fd_worst(
    z: &OZx,
    lat: Lat,
    spin: Spin,
    rsh: RshParams,
    lindep: f64,
    comps: &[(usize, usize)],
    h: f64,
) -> f64 {
    let g = zx_open_force(z);
    comps
        .iter()
        .map(|&(a, x)| {
            let fd = zx_open_fd((lat, 2), &H3D, (a, x, h), spin, rsh, lindep, z.exxdiv);
            (fd - g.grad[(a, x)]).abs()
        })
        .fold(0.0, f64::max)
}

/// FD of the zero-XC polarized energy on EVERY component (H3 doublet), UKS
/// and ROKS, CAM-type on `TRI_A` (lindep 1e-6) and HSE-type on the 6.5 cube
/// (lindep 1e-10), with the h sweep 2e-4 / 1e-4 / 5e-5. Bar: 1e-8 at
/// h = 5e-5 (#514's) and a 3..6 ratio between h = 2e-4 and 1e-4 (the h^2
/// truncation; an analytic error would not shrink with h).
#[test]
#[ignore = "validation: ~40 fresh RS-GDF builds + SCFs per case"]
fn zero_xc_force_matches_fd_every_component_and_h_sweep() {
    let cases: [(&str, Lat, RshParams, f64); 2] = [
        ("CAM-type TRI_A", Lat::Tri, cam_type(), rs::LINDEP_FD),
        (
            "HSE-type a = 6.5",
            Lat::Cube65,
            hse_type(),
            rs::LINDEP_DEFAULT,
        ),
    ];
    for (tag, lat, rsh, lindep) in cases {
        for spin in [Spin::Uks, Spin::Roks] {
            let z = zx_open(
                cell_of(lat, 2, &H3D),
                spin,
                Some(rsh),
                1.0,
                lindep,
                ExxDiv::Ewald,
            );
            assert!(z.scf.asym() > 0.1, "polarized");
            let g = zx_open_force(&z);
            let w = |h: f64| zx_fd_worst(&z, lat, spin, rsh, lindep, &all_comps(3), h);
            let (w2, w1, w05) = (w(2e-4), w(1e-4), w(5e-5));
            eprintln!(
                "{tag} {spin:?}: FD miss {w2:.2e} / {w1:.2e} / {w05:.2e} at h = 2e-4 / 1e-4 / \
                 5e-5 (ratio {:.2}); max|F| {:.2e}, max|F_att| {:.2e}, net {:.1e}",
                w2 / w1,
                max_abs(&g.grad),
                max_abs(&g.hf.parts.fit_attenuated),
                g.hf.net_force
            );
            assert!(w05 < 1e-8, "{tag} {spin:?}: {w05:e}");
            assert!(
                (3.0..6.0).contains(&(w2 / w1)),
                "{tag} {spin:?}: ratio {}",
                w2 / w1
            );
            assert!(g.hf.net_force < 1e-12);
        }
    }
}

/// Equal fractions == global hybrid under `exxdiv = none` too, and for ROKS
/// (the per-commit file runs UKS / ewald). Bar `ANCHOR_TOL` (1e-10).
#[test]
#[ignore = "validation: four RS-GDF builds + SCFs per spin"]
fn equal_fractions_anchor_both_exxdiv_both_spins() {
    let eq = RshParams {
        omega: 0.33,
        c_sr: 0.4,
        c_lr: 0.4,
    };
    for spin in [Spin::Uks, Spin::Roks] {
        for exxdiv in [ExxDiv::Ewald, ExxDiv::None] {
            let cell = || cell_of(Lat::Tri, 2, &H3D);
            let zr = zx_open(cell(), spin, Some(eq), 1.0, rs::LINDEP_FD, exxdiv);
            let zg = zx_open(cell(), spin, None, 0.4, rs::LINDEP_FD, exxdiv);
            let de = (zr.scf.energy() - zg.scf.energy()).abs();
            let d = diff(&zx_open_force(&zr).grad, &zx_open_force(&zg).grad);
            eprintln!("{spin:?} {exxdiv:?}: equal fractions |dE| {de:.1e}, force {d:.2e}");
            assert!(
                de < 1e-11 && d < ANCHOR_TOL,
                "{spin:?} {exxdiv:?}: {de:e} {d:e}"
            );
        }
    }
}

/// KS force (CAM-B3LYP at lindep 1e-6, HSE06 at 1e-10; UKS and ROKS) on the
/// polarized H3 doublet (`TRI_A`, ewald) vs central FD of the KS energy,
/// three components, h = 1e-4 / 5e-5. Bar 2e-8 at h = 5e-5 (#514's KS bar).
#[test]
#[ignore = "validation: 12 KS SCFs per (functional, spin)"]
fn ks_force_matches_fd() {
    let comps = [(0, 2), (1, 0), (2, 1)];
    for (f, lindep) in [(CAM, rs::LINDEP_FD), ("HSE06", rs::LINDEP_DEFAULT)] {
        for spin in [Spin::Uks, Spin::Roks] {
            let ks = ks_open(
                cell_of(Lat::Tri, 2, &H3D),
                spin,
                f,
                Some(lindep),
                ExxDiv::Ewald,
            );
            assert!(ks.scf.asym() > 0.1, "polarized");
            let g = ks_open_force(&ks);
            let w = |h: f64| {
                comps
                    .iter()
                    .map(|&(a, x)| {
                        let fd = ks_open_fd(
                            (Lat::Tri, 2),
                            &H3D,
                            (a, x, h),
                            spin,
                            f,
                            Some(lindep),
                            ExxDiv::Ewald,
                        );
                        (fd - g.grad[(a, x)]).abs()
                    })
                    .fold(0.0, f64::max)
            };
            let (w1, w05) = (w(1e-4), w(5e-5));
            eprintln!(
                "{f} {spin:?}: KS FD miss {w1:.2e} / {w05:.2e} at h = 1e-4 / 5e-5; max|F| {:.2e}, \
                 max|F_att| {:.2e}",
                max_abs(&g.grad),
                max_abs(&g.hf.parts.fit_attenuated)
            );
            assert!(w05 < 2e-8, "{f} {spin:?}: {w05:e}");
        }
    }
}

/// INDEPENDENT CONSTRUCTION: the RS-GDF UKS force against the dense-AFT UKS
/// force (#505, its own FD and PySCF-pinned energy) on the same polarized
/// cell, functional and grid. The two differ by the fit error only; for the
/// plain Coulomb fit that is the PBE0 gap. Artifact hypothesis: a wrongly
/// composed attenuated piece (scale, `c0`, spin factor) would make the RSH
/// gap O(1e-3..1e-2), far above the PBE0 gap; bar: RSH gap < 1.5 x the PBE0
/// gap (#514's closed-shell bar, measured there within 5%).
#[test]
#[ignore = "validation: six KS SCFs (dense and RS-GDF)"]
fn rsgdf_minus_dense_gap_is_the_plain_fit_gap() {
    let gap = |f: &str, lindep: f64| {
        let cell = || cell_of(Lat::Tri, 2, &H3D);
        let g = ks_open(cell(), Spin::Uks, f, Some(lindep), ExxDiv::Ewald);
        let d = ks_open(cell(), Spin::Uks, f, None, ExxDiv::Ewald);
        assert!(matches!(
            d.kept.grad_jk(None, &d.rcfg).unwrap(),
            KGradJk::Dense(_)
        ));
        diff(&ks_open_force(&g).grad, &ks_open_force(&d).grad)
    };
    for lindep in [rs::LINDEP_FD, rs::LINDEP_DEFAULT] {
        let pbe0 = gap("PBE0", lindep);
        for f in [CAM, "HSE06"] {
            let r = gap(f, lindep);
            eprintln!(
                "lindep {lindep:e}: RS-GDF - dense UKS force gap {f} {r:.2e} vs PBE0 {pbe0:.2e} \
                 (ratio {:.3})",
                r / pbe0
            );
            assert!(r < 1.5 * pbe0, "{f} lindep {lindep:e}: {r:e} vs {pbe0:e}");
        }
    }
}

/// The full mutation ledger (production-wired defects, `ledger_mutants`)
/// against the production force on the CAM-type and HSE-type polarized
/// zero-XC fixtures, UKS and ROKS (plus ROKS's `RoWDropCross`). Bar: every
/// miss > 1e-5 (>= 1000 FD floors of ~4e-9), except where a mutant is blind by
/// construction (listed in the output).
#[test]
#[ignore = "validation: full mutation ledger"]
fn full_mutation_ledger() {
    use ferric_pbc::kuxcgrad::KUSpinMutation;
    let cases: [(&str, Lat, RshParams, f64); 2] = [
        ("CAM-type TRI_A", Lat::Tri, cam_type(), rs::LINDEP_FD),
        (
            "HSE-type a = 6.5",
            Lat::Cube65,
            hse_type(),
            rs::LINDEP_DEFAULT,
        ),
    ];
    for (tag, lat, rsh, lindep) in cases {
        for spin in [Spin::Uks, Spin::Roks] {
            let z = zx_open(
                cell_of(lat, 2, &H3D),
                spin,
                Some(rsh),
                1.0,
                lindep,
                ExxDiv::Ewald,
            );
            let g = zx_open_force(&z);
            let mut ledger = ledger_mutants();
            if spin == Spin::Roks {
                ledger.push(("RoWDropCross", None, Some(KUSpinMutation::RoWDropCross)));
            }
            for (n, m, sm) in ledger {
                let miss = diff(&zx_open_force_m(&z, m, sm).unwrap().grad, &g.grad);
                eprintln!("{tag} {spin:?} {n}: {miss:.2e}");
                assert!(miss > 1e-5, "{tag} {spin:?} {n}: {miss:e}");
            }
        }
    }
}
