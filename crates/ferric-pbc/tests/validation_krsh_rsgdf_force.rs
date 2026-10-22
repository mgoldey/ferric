//! Weekly validation of the range-separated k-point force and stress on
//! RS-GDF J/K (`KRsGdf::build_rsh`; `rsgdf::kpoint::kderiv` "Attenuated
//! fit"). Every test here is `#[ignore]`d; the per-commit subset (the `ω → ∞`
//! exactness anchor, the empty-class skip, a wiring smoke with mutants) is
//! `pbc_krsh_rsgdf_force.rs`, whose header carries the hypotheses.
//!
//! Systems: H2 / STO-3G, ET-sp aux, 1x1x2 mesh, RS-GDF split `ω_0 = 1`:
//! TRI2 (the prototype's triclinic cell, CAM-type `ω = 0.33`) and an
//! off-axis H2 in a 6.5 Bohr cube (HSE-type `ω = 0.11`: q = 0 carries K
//! vectors inside its cut only for a > ~5.2 Bohr, so the `c0` and scale
//! mutants are visible there and blind on TRI2). The total energy is the
//! ZERO-XC range-separated exchange energy (`kpoint_null_xc_*_rsh`): the
//! atom-centred XC grid adds FD steps on these cells that are identical on
//! dense AFT (TRI2 CAM-B3LYP FD residual 4.4e-6 / 8.8e-6 at h = 1e-4 / 5e-5,
//! growing as 1/h, on dense and RS-GDF alike; HSE06 at a = 6.5 jumps by
//! 5.6e-8 Ha at h = 1e-4 on both), so the KS FD rows run on the 4 Bohr cube
//! where the grid is smooth.
//!
//! MEASURED (release test profile, 2026-10-22):
//!
//! * Attenuated fit alone vs FD of its fixed-density energy, `ω = 0.33`,
//!   lindep 1e-6: 5.8e-9 / 5.2e-11 / 9.0e-12 at h = 1e-3 / 1e-4 / 1e-5
//!   (`max|F|` 4.6e-2). Bar 1e-9 at h = 1e-4.
//! * Default lindep (1e-10) characterisation of the same fit. The ENERGY is
//!   clean: re-chunking the build's K sum (2, 4, 10 GEMM chunks) and a rigid
//!   translation move E by <= 7e-17 at lindep 1e-10, 1e-8 and 1e-6 (the
//!   prototype's ~5e-8 summation-order noise does not occur in Rust). The
//!   analytic FORCE is roundoff-limited when metric eigenvalues near 1e-11
//!   are kept: at fixed density a rigid translation of 1e-3 moves it by
//!   3.7e-8 / 2.9e-9 / 4.6e-12 at lindep 1e-10 / 1e-8 / 1e-6 (it must not
//!   move at all), and its miss against FD of the energy is flat in h at
//!   lindep 1e-10 (1.67e-8, 1.69e-8, 1.69e-8 at h = 1e-3, 1e-4, 1e-5),
//!   2.1e-9 at 1e-8, the h^2 floor at 1e-6. So the attenuated force at the
//!   default lindep is accurate to about 4e-8 for the CAM-type fit; on the
//!   total CAM-type force this is a 3.8e-9 floor (below). The production
//!   default is unchanged.
//! * EXISTING LIMIT, not specific to the attenuated fit (independent review):
//!   the plain Coulomb RS-GDF force shows the same thing. The ET-sp aux plus
//!   a near-duplicate s shell (kept eigenvalues 1.4e-11, lindep 1e-10) gives
//!   a flat FD miss of ~4e-6 at h = 1e-3 / 1e-4 / 1e-5, dropping to the h^2
//!   floor at lindep 1e-6. The kderiv near-null-mode conditioning is tracked
//!   separately: see #512.
//! * Zero-XC total force vs central FD (fresh build + SCF per point), three
//!   components: CAM-type TRI2 (lindep 1e-6) 1.11e-8 / 2.78e-9 / 6.60e-10
//!   and HSE-type a = 6.5 (lindep 1e-10) 9.71e-9 / 2.44e-9 / 6.25e-10 at
//!   h = 2e-4 / 1e-4 / 5e-5 (ratio 4.0 per halving: the h^2 truncation).
//!   CAM-type at lindep 1e-10: 7.5e-9 / 3.8e-9 / 3.9e-9 (the bias floor).
//!   Bars: 1e-8 at h = 5e-5 (15x) and a 3..6 ratio between h = 2e-4 and
//!   1e-4. `Σ_A F_A` <= 1.2e-15 (bar 1e-12).
//! * KS (CAM-B3LYP lindep 1e-6, HSE06 lindep 1e-10), 4 Bohr cube vs FD of
//!   the KS energy, three components: 2.15e-9 / 5.77e-10 (CAM-B3LYP) and
//!   2.13e-9 / 5.77e-10 (HSE06) at h = 1e-4 / 5e-5 (dense AFT on the same
//!   cell: 2.16e-9 / 5.81e-10). Bar 2e-8 at h = 5e-5.
//! * RS-GDF vs dense AFT KS force on the same cell: TRI2 PBE0 / HSE06 /
//!   CAM-B3LYP 1.60e-5 / 1.60e-5 / 1.65e-5, a = 6.5 1.71e-5 / 1.71e-5 /
//!   1.79e-5: the RSH gap is the plain-Coulomb (PBE0) fit gap to within 5%.
//!   Bar: RSH gap < 1.5 x the PBE0 gap.
//! * Zero-XC stress vs central FD of the strained energy, all nine
//!   components: CAM-type TRI2 1.26e-8 / 3.21e-9, HSE-type a = 6.5 8.13e-9 /
//!   2.06e-9 at h = 1e-4 / 5e-5. Bar 3e-8 at h = 5e-5 plus a 3..6 ratio.
//!
//! MUTATION LEDGER (max |F_mut − F| or |dE/dε_mut − dE/dε|, Hartree/Bohr or
//! Hartree per cell; FD floors ~3e-9 at h = 1e-4; bar 1e-6):
//!
//! | mutant | CAM-type TRI2 | HSE-type a = 6.5 |
//! |---|---|---|
//! | RshFitKeepC0 (force) | 2.80e-2 | 6.14e-2 |
//! | RshFitNoScale (force) | 2.50e-2 | 8.40e-4 |
//! | RshMadelungUnmixed | 4.82e-2 | 1.26e-2 |
//! | RshMadelungFullLr | 2.22e-2 | 1.60e-2 |
//! | RshMadelungScaled | 6.26e-2 | 1.20e-2 |
//! | RshPlainMadelung | 7.58e-2 | 9.87e-2 |
//! | RshNoMadelungTerm | 7.73e-2 | 1.60e-2 |
//! | RshFitOmega0 | refused (rebuilt-B check) | refused |
//! | RshFitNoScale (stress) | 1.26e-1 | 1.60e-2 |
//! | RshFitNoAttenuated (stress) | 1.08e-1 | 3.20e-3 |
//! | RshPlainMadelungStrain | 2.48e-1 | 2.46e-1 |
//! | RshNoLrMadelungStrain | 1.64e-1 | 4.18e-3 |
//!
//! On TRI2 at lindep 1e-6 HSE06's q = 0 class is skipped (weights under the
//! cut), which is why the HSE-type rows run on the 6.5 Bohr cube at the
//! production lindep, where q = 0 keeps 5 eigenvectors and KeepC0 is live.

mod common;
#[path = "krsh_rsgdf_force_support/mod.rs"]
mod support;

use ferric_pbc::dense_aft::ExxDiv;
use ferric_pbc::kgrad::KGradMutation;
use ferric_pbc::kstress::KStressMutation;
use ferric_pbc::lattice::Cell;
use ferric_pbc::rsgdf::kpoint::{KRsGdf, KRsGdfConfig};
use ferric_pbc::rsh::RshParams;
use support::*;

fn big(pos: &[[f64; 3]]) -> Cell {
    Cell::new(common::hydrogens(pos), common::cubic(6.5)).unwrap()
}

fn cube4(pos: &[[f64; 3]]) -> Cell {
    Cell::new(common::hydrogens(pos), common::cubic(4.0)).unwrap()
}

const COMPS: [(usize, usize); 3] = [(0, 0), (0, 2), (1, 1)];

/// The two zero-XC systems: (tag, cell builder, positions, fractions, lindep).
type ZxCase = (
    &'static str,
    fn(&[[f64; 3]]) -> Cell,
    &'static [[f64; 3]; 2],
    RshParams,
    f64,
);

fn cases() -> [ZxCase; 2] {
    [
        (
            "CAM-type TRI2",
            tri2,
            &TRI2_ATOMS,
            rsh_params(0.19, 0.65, 0.33),
            LINDEP_FD,
        ),
        (
            "HSE-type a = 6.5",
            big,
            &H2_BIG,
            rsh_params(0.25, 0.0, 0.11),
            LINDEP_DEFAULT,
        ),
    ]
}

/// Worst |analytic − FD| of the attenuated fit alone (TRI2) over `COMPS`.
fn att_fd_miss(s: &Sys, cfg: &KRsGdfConfig, w: f64, h: f64) -> f64 {
    let d = fixed_density(s, &plain_fit(s, cfg));
    let att = att_fit(s, cfg, w, None);
    let g = fit_force(s, &att, cfg, &d, Some((w, prod_gcut(cfg, w)))).unwrap();
    let cell_of = |p: &[[f64; 3]]| tri2(p);
    let mut worst = 0.0_f64;
    for (a, x) in COMPS {
        let ep = att_energy_at(&cell_of, &moved(&TRI2_ATOMS, a, x, h), cfg, w, &d);
        let em = att_energy_at(&cell_of, &moved(&TRI2_ATOMS, a, x, -h), cfg, w, &d);
        worst = worst.max(((ep - em) / (2.0 * h) - g[(a, x)]).abs());
    }
    worst
}

/// The attenuated fit ALONE (`ω = 0.33`, TRI2) vs central FD of its
/// fixed-density exchange energy at the FD-clean metric cut 1e-6. Physics:
/// the h^2 truncation (falling ~100x per decade of h); artifact: a wrong
/// kernel, metric or aux term is O(1e-3..1e-2), h-independent.
#[test]
#[ignore = "validation: attenuated fit alone vs FD (w = 0.33, h sweep, 12 fit builds)"]
fn attenuated_fit_force_matches_fd() {
    let s = sys(tri2(&TRI2_ATOMS));
    let cfg = gdf_cfg(LINDEP_FD, ExxDiv::None);
    let m: Vec<f64> = [1e-3, 1e-4]
        .iter()
        .map(|&h| att_fd_miss(&s, &cfg, 0.33, h))
        .collect();
    eprintln!(
        "lindep 1e-6: FD miss h = 1e-3 {:.2e}, h = 1e-4 {:.2e}",
        m[0], m[1]
    );
    assert!(m[1] < 1e-9, "{:e}", m[1]);
    assert!(m[0] / m[1] > 10.0, "not h^2: {} / {}", m[0], m[1]);
}

/// Characterisation of the attenuated fit at the PRODUCTION metric cut and
/// two tighter ones (header). ENERGY checks: re-chunking the build's K sum
/// and a rigid translation move E at roundoff (these do not test the
/// derivative: the force's own rebuild keeps one chunking). FORCE check: a
/// rigid translation of 1e-3 must leave the fixed-density force unchanged;
/// it does to roundoff at lindep 1e-6 and moves by the analytic force's
/// conditioning error at 1e-10 / 1e-8 (asserted below the measured accuracy
/// bound, not a pass condition to tighten). The flat FD miss at lindep 1e-10
/// is that same error.
#[test]
#[ignore = "validation: default-lindep characterisation of the attenuated fit (chunking, translation, FD offset)"]
fn attenuated_fit_default_lindep_noise_and_offset() {
    let w = 0.33;
    let s = sys(tri2(&TRI2_ATOMS));
    for lindep in [LINDEP_DEFAULT, 1e-8, LINDEP_FD] {
        let cfg = gdf_cfg(lindep, ExxDiv::None);
        let d = fixed_density(&s, &plain_fit(&s, &cfg));
        let att = att_fit(&s, &cfg, w, None);
        let gc = Some((w, prod_gcut(&cfg, w)));
        let (e0, g0) = (
            ex_energy(&att, &d),
            fit_force(&s, &att, &cfg, &d, gc).unwrap(),
        );
        let resident = att.stats().resident_bytes;
        for extra in [1usize << 14, 1 << 16] {
            let mut c2 = cfg;
            c2.gdf.budget_bytes = Some(resident + extra);
            let a2 = KRsGdf::build_attenuated_only(
                &s.cell, &s.prep, &s.aux, &s.mesh, &s.hk.s, &c2, w, None,
            )
            .unwrap()
            .unwrap();
            assert!(a2.stats().n_lr_chunks > att.stats().n_lr_chunks);
            let de = ex_energy(&a2, &d) - e0;
            eprintln!(
                "lindep {lindep:e}: {} vs {} chunks: dE {de:.2e}",
                a2.stats().n_lr_chunks,
                att.stats().n_lr_chunks
            );
            assert!(de.abs() < 1e-14, "{de:e}");
        }
        for t in [1e-6, 1e-4] {
            let st = sys(tri2(&translated(&TRI2_ATOMS, [t, 0.7 * t, -0.4 * t])));
            let de = ex_energy(&att_fit(&st, &cfg, w, None), &d) - e0;
            eprintln!("lindep {lindep:e}: rigid translation {t:e}: dE {de:.2e}");
            assert!(de.abs() < 1e-14, "{de:e}");
        }
        // The derivative check: the fixed-density force is translation
        // invariant.
        let t = 1e-3;
        let st = sys(tri2(&translated(&TRI2_ATOMS, [t, 0.7 * t, -0.4 * t])));
        let at = att_fit(&st, &cfg, w, None);
        let df = diff(&g0, &fit_force(&st, &at, &cfg, &d, gc).unwrap());
        eprintln!("lindep {lindep:e}: rigid translation {t:e}: dF {df:.2e}");
        if lindep == LINDEP_FD {
            assert!(df < 1e-10, "{df:e}");
        }
        assert!(df < 3e-7, "{df:e}");
    }
    let cfg = gdf_cfg(LINDEP_DEFAULT, ExxDiv::None);
    let (m3, m5) = (
        att_fd_miss(&s, &cfg, w, 1e-3),
        att_fd_miss(&s, &cfg, w, 1e-5),
    );
    eprintln!("lindep 1e-10: analytic - FD offset {m3:.2e} (h = 1e-3), {m5:.2e} (h = 1e-5)");
    assert!(m5 < 1e-7, "{m5:e}");
    assert!(
        (m3 / m5 - 1.0).abs() < 0.25,
        "offset not h-independent: {m3:e} {m5:e}"
    );
}

/// Zero-XC total RSH force vs central FD of ferric's own energy (fresh build
/// and SCF per displacement), h sweep, both systems; forces sum to zero and
/// the attenuated piece is live. Physics: the h^2 truncation; artifact: a
/// scale, Madelung or fit defect is >= 8e-4 (ledger).
#[test]
#[ignore = "validation: zero-XC RSH force vs FD, 2 systems x 3 components x 3 steps (36 RS-GDF SCFs)"]
fn zero_xc_rsh_force_matches_fd() {
    for (tag, cell_of, pos, r, lindep) in cases() {
        let z = zx_run(cell_of(pos), r, lindep, ExxDiv::Ewald);
        let g = zx_force_m(&z, None).unwrap();
        let att = max_abs(&g.hf.parts.fit_attenuated);
        eprintln!(
            "{tag}: max|F| {:.3e}, max|F_att| {att:.3e}, net {:.2e}",
            max_abs(&g.grad),
            g.hf.net_force
        );
        assert!(att > 1e-4, "{tag}: attenuated piece not exercised");
        assert!(g.hf.net_force < 1e-12, "{tag}: {:e}", g.hf.net_force);
        let miss: Vec<f64> = [2e-4, 1e-4, 5e-5]
            .iter()
            .map(|&h| {
                COMPS.iter().fold(0.0_f64, |m, &(a, x)| {
                    let fd = zx_fd(&cell_of, pos, (a, x, h), r, lindep, ExxDiv::Ewald);
                    m.max((fd - g.grad[(a, x)]).abs())
                })
            })
            .collect();
        eprintln!("{tag}: FD miss h = 2e-4 / 1e-4 / 5e-5: {miss:?}");
        assert!(miss[2] < 1e-8, "{tag}: {:e}", miss[2]);
        let ratio = miss[0] / miss[1];
        assert!(ratio > 3.0 && ratio < 6.0, "{tag}: not h^2 ({ratio})");
    }
}

/// The production-wired mutants on both zero-XC systems (header ledger):
/// each misses the FD-validated force / stress by more than 1e-6 (>= 300x the
/// FD floor); the `ω_0` one is refused by the rebuilt-B check.
#[test]
#[ignore = "validation: RS-GDF RSH mutation ledger (2 SCFs, 13 derivative runs each)"]
fn zero_xc_rsh_mutation_ledger() {
    for (tag, cell_of, pos, r, lindep) in cases() {
        let z = zx_run(cell_of(pos), r, lindep, ExxDiv::Ewald);
        let g = zx_force_m(&z, None).unwrap();
        for m in [
            KGradMutation::RshFitKeepC0,
            KGradMutation::RshFitNoScale,
            KGradMutation::RshMadelungUnmixed,
            KGradMutation::RshMadelungFullLr,
            KGradMutation::RshMadelungScaled,
            KGradMutation::RshPlainMadelung,
            KGradMutation::RshNoMadelungTerm,
        ] {
            let miss = diff(&zx_force_m(&z, Some(m)).unwrap().grad, &g.grad);
            eprintln!("{tag} {m:?}: force miss {miss:.2e}");
            assert!(miss > 1e-6, "{tag} {m:?}: {miss:e}");
        }
        let e = zx_force_m(&z, Some(KGradMutation::RshFitOmega0))
            .expect_err("RshFitOmega0 must be refused")
            .to_string();
        assert!(e.contains("pass the config the build used"), "{e}");
        let st = zx_stress_m(&z, None).unwrap();
        for m in [
            KStressMutation::RshFitNoScale,
            KStressMutation::RshFitNoAttenuated,
            KStressMutation::RshPlainMadelungStrain,
            KStressMutation::RshNoLrMadelungStrain,
        ] {
            let miss = mdiff(&zx_stress_m(&z, Some(m)).unwrap().de_deps, &st.de_deps);
            eprintln!("{tag} {m:?}: stress miss {miss:.2e}");
            assert!(miss > 1e-6, "{tag} {m:?}: {miss:e}");
        }
    }
}

/// Zero-XC RSH stress vs central FD of the strained energy, all nine
/// components, two steps. Physics: the h^2 truncation; artifact: a missing
/// attenuated strain or Madelung strain is >= 3e-3 (ledger).
#[test]
#[ignore = "validation: zero-XC RSH stress vs FD, 2 systems x 9 components x 2 steps (72 RS-GDF SCFs)"]
fn zero_xc_rsh_stress_matches_fd() {
    for (tag, cell_of, pos, r, lindep) in cases() {
        let cell = cell_of(pos);
        let z = zx_run(cell.clone(), r, lindep, ExxDiv::Ewald);
        let st = zx_stress_m(&z, None).unwrap();
        assert!(mmax(&st.hf.parts.fit_attenuated) > 1e-4, "{tag}");
        let miss: Vec<f64> = [1e-4, 5e-5]
            .iter()
            .map(|&h| {
                let mut m = 0.0_f64;
                for i in 0..3 {
                    for j in 0..3 {
                        let fd = zx_fd_strain(&cell, (i, j, h), r, lindep, ExxDiv::Ewald);
                        m = m.max((fd - st.de_deps[i][j]).abs());
                    }
                }
                m
            })
            .collect();
        eprintln!(
            "{tag}: max|dE/de| {:.3e}; strain FD miss h = 1e-4 / 5e-5: {miss:?}",
            mmax(&st.de_deps)
        );
        assert!(miss[1] < 3e-8, "{tag}: {:e}", miss[1]);
        let ratio = miss[0] / miss[1];
        assert!(ratio > 3.0 && ratio < 6.0, "{tag}: not h^2 ({ratio})");
    }
}

/// The KS force (CAM-B3LYP at lindep 1e-6, HSE06 at the production cut) on
/// RS-GDF vs central FD of the KS energy, on the 4 Bohr cube where the atom
/// grid is FD-smooth (header). Physics: h^2 truncation ~6e-10 at h = 5e-5;
/// artifact: >= 1e-5. The CAM-B3LYP row exercises the attenuated fit
/// (asserted); the HSE06 row is WIRING ONLY: on the 4 Bohr cube its erf
/// weights are ~3e-6 and its q = 0 class is empty, so its attenuated piece is
/// negligible (max|F_att| 9.7e-8 vs 1.5e-2 for CAM-B3LYP; the HSE-type
/// attenuated derivative is validated on the 6.5 Bohr cube, zero XC).
#[test]
#[ignore = "validation: KS RSH force on RS-GDF vs FD (2 functionals x 3 components x 2 steps)"]
fn ks_rsh_force_on_rsgdf_matches_fd() {
    for (f, lindep) in [(CAM, LINDEP_FD), ("HSE06", LINDEP_DEFAULT)] {
        let ks = ks_run(cube4(&H2_BIG), f, Some(lindep), ExxDiv::Ewald);
        let g = ks_force(&ks);
        let att = max_abs(&g.hf.parts.fit_attenuated);
        eprintln!("{f} a = 4: max|F_att| {att:.2e}");
        if f == CAM {
            assert!(att > 1e-3, "{f}: attenuated piece not exercised ({att:e})");
        }
        for h in [1e-4, 5e-5] {
            let worst = COMPS.iter().fold(0.0_f64, |m, &(a, x)| {
                let fd = ks_fd(&cube4, &H2_BIG, (a, x, h), f, Some(lindep), ExxDiv::Ewald);
                m.max((fd - g.grad[(a, x)]).abs())
            });
            eprintln!("{f} a = 4 h = {h:e}: FD miss {worst:.2e}");
            if h == 5e-5 {
                assert!(worst < 2e-8, "{f}: {worst:e}");
            }
        }
    }
}

/// RS-GDF vs dense-AFT KS force on the same cell and grid: the RSH gap must
/// be the plain-Coulomb (PBE0) fit gap of the same aux and mesh (header).
/// Physics: ratio ~1; artifact: a defect adds >= 8e-4 against gaps ~2e-5.
#[test]
#[ignore = "validation: RS-GDF vs dense RSH force (2 cells x 3 functionals x 2 J/K)"]
fn rsgdf_rsh_force_is_the_dense_rsh_force_to_the_fit_error() {
    let rows: [(&str, fn(&[[f64; 3]]) -> Cell, &[[f64; 3]; 2]); 2] =
        [("TRI2", tri2, &TRI2_ATOMS), ("a = 6.5", big, &H2_BIG)];
    for (tag, cell_of, pos) in rows {
        let gap = |f: &str| {
            let g = ks_force(&ks_run(cell_of(pos), f, Some(LINDEP_FD), ExxDiv::Ewald));
            let d = ks_force(&ks_run(cell_of(pos), f, None, ExxDiv::Ewald));
            diff(&g.grad, &d.grad)
        };
        let scale = gap("PBE0");
        assert!(scale > 1e-7, "fit error not measurable");
        for f in ["HSE06", CAM] {
            let d = gap(f);
            eprintln!(
                "{tag} {f}: RS-GDF - dense force {d:.2e} (PBE0 {scale:.2e}, ratio {:.3})",
                d / scale
            );
            assert!(d < 1.5 * scale, "{tag} {f}: {d:e} vs {scale:e}");
        }
    }
}
