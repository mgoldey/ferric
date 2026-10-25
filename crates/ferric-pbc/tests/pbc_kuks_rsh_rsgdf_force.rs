//! Per-commit anchors of the OPEN-SHELL (UKS, ROKS) range-separated k-point
//! force on RS-GDF J/K (`ferric_pbc::kgrad`, module doc "Scope";
//! `ferric_pbc::kuxcgrad`, "Range-separated hybrids (UKS, ROKS)").
//!
//! # Derivation (a composition, no new derivative algebra)
//!
//! The RS-GDF RSH exchange is `K_σ = c_sr K[1/r][D_σ] + (c_lr − c_sr)
//! K[erf][D_σ] + v_M^mix S D_σ S` per spin, scale 1. The closed-shell force
//! (#514) differentiates the plain fit's exchange pieces with the exchange
//! terms `Σ_s c_s D_s X D_s` scaled by `c_sr`, the attenuated fit's LR and
//! metric passes with the same terms scaled by `c_lr − c_sr`, and the M-term
//! `−v_M^mix Σ_s D_s S D_s`. The open-shell UKS force on dense AFT (#505) and
//! the UHF force on RS-GDF use the same assembly with the UHF terms
//! `(1, D_α), (1, D_β)` (unit occupation, no ½: twice the closed-shell
//! coefficient `(½, D_total)` per spin). Both fit passes already take that
//! term list, so the open-shell RS-GDF RSH force is the closed-shell one with
//! the open-shell term list; the only code change is lifting the refusal.
//!
//! # Hypotheses (written before measuring)
//!
//! * (a) SINGLET: on a closed-shell cell the UKS path reproduces #514's RKS
//!   force (separate SCFs) to the SCF floor. Artifact: a per-spin factor
//!   slip (`½` per spin, `ExchClosedShellFactor`; a dropped β term) shows as
//!   O(1e-2); pure spin-assignment defects are invisible by construction,
//!   which is why (b)-(c) run on a polarized cell.
//! * (b) EQUAL FRACTIONS `c_sr = c_lr = c`: the attenuated piece has
//!   coefficient 0 and the mixed constant is `c v_M`, so the force is the
//!   global-hybrid RS-GDF UKS force with fraction `c` (two SCFs, polarized).
//!   Blind to the attenuated fit (coefficient 0) and to M-term spin errors
//!   (both sides share the sandwich); FD covers those.
//! * (c) FD of ferric's own polarized UKS / ROKS RS-GDF energy (zero XC, no
//!   grid steps) at h = 1e-4. Physics: the h^2 floor (#514 closed shell:
//!   2.8e-9 at h = 1e-4); artifact: every ledger mutant misses by >= 1e-5.
//!
//! No PySCF force oracle exists for this: PySCF 2.13's periodic KS gradients
//! (`pbc/grad/kuks.py`) call `with_df.get_jk_e1`, which only FFTDF
//! implements (no GDF / RS-GDF gradient), and have no grid response. The
//! independent construction is the dense-AFT UKS RSH force (#505) on the
//! same cell (weekly file: the RS-GDF − dense gap must match the plain-fit
//! (PBE0) gap).
//!
//! MEASURED and MUTATION LEDGER: see `validation_kuks_rsh_rsgdf_force.rs`.

mod common;
#[path = "krsh_rsgdf_force_support/mod.rs"]
mod rs;
#[path = "kuks_rsh_rsgdf_force_support/mod.rs"]
mod support;

use ferric_pbc::dense_aft::ExxDiv;
use ferric_pbc::rsh::RshParams;
use support::*;

/// (a) SINGLET: zero-XC CAM-type RSH on the TRI2 H2 singlet (RS-GDF, lindep
/// 1e-6): the UKS force (UKS SCF, UKS entry) equals #514's closed-shell RKS
/// force (independent RKS SCF and entry) per part, `<= 1e-10`. MEASURED:
/// energy 1.1e-16, every part and the total EXACTLY 0. Too clean, so
/// audited: the singlet UKS SCF lands on `D_σ = D/2` bitwise, and the
/// open-shell terms `1·(D/2) X (D/2)` twice vs the closed-shell `½·D X D`
/// differ only by power-of-two scalings, which are exact in floating point.
/// The anchor is not vacuous: `ExchDropBeta` / `ExchClosedShellFactor` move
/// it by 3.1e-2, and the source mutants of `KSpins::exch_terms` fail it
/// (validation file header).
#[test]
fn singlet_uks_is_the_rks_rsgdf_rsh_force() {
    let cell = rs::tri2(&rs::TRI2_ATOMS);
    let u = zx_open(
        cell.clone(),
        Spin::Uks,
        Some(cam_type()),
        1.0,
        rs::LINDEP_FD,
        ExxDiv::Ewald,
    );
    assert!(u.scf.asym() < 1e-8, "singlet");
    let r = rs::zx_run(cell, cam_type(), rs::LINDEP_FD, ExxDiv::Ewald);
    let gu = zx_open_force(&u);
    let gr = rs::zx_force_m(&r, None).unwrap();
    let (p, q) = (&gu.hf.parts, &gr.hf.parts);
    assert!(max_abs(&p.fit_attenuated) > 1e-3, "attenuated piece live");
    let rows = [
        ("energy", (u.scf.energy() - r.scf.energy).abs()),
        ("fit_attenuated", diff(&p.fit_attenuated, &q.fit_attenuated)),
        ("fit_orb_lr", diff(&p.fit_orb_lr, &q.fit_orb_lr)),
        ("fit_aux_lr", diff(&p.fit_aux_lr, &q.fit_aux_lr)),
        ("fit_metric_lr", diff(&p.fit_metric_lr, &q.fit_metric_lr)),
        ("fit_orb_sr", diff(&p.fit_orb_sr, &q.fit_orb_sr)),
        ("fit_metric_sr", diff(&p.fit_metric_sr, &q.fit_metric_sr)),
        ("overlap", diff(&p.overlap, &q.overlap)),
        ("total", diff(&gu.grad, &gr.grad)),
    ];
    for (n, d) in rows {
        eprintln!("singlet UKS vs RKS {n}: {d:.2e}");
        assert!(d < ANCHOR_TOL, "{n}: {d:e}");
    }
    for (n, m, sm) in ledger_mutants() {
        let blind = SPIN_ONLY.contains(&n);
        if !blind && n != "ExchDropBeta" && n != "ExchClosedShellFactor" {
            continue; // the rest of the ledger: weekly
        }
        let miss = diff(&zx_open_force_m(&u, m, sm).unwrap().grad, &gu.grad);
        eprintln!("singlet: {n} miss {miss:.2e}");
        if blind {
            assert!(
                miss < ANCHOR_TOL,
                "{n} should be blind on a singlet: {miss:e}"
            );
        } else {
            assert!(miss > 1e-3, "{n}: {miss:e}");
        }
    }
}

/// (b) EQUAL FRACTIONS (polarized H3 doublet, `TRI_A`, ewald; UKS here,
/// ROKS and exxdiv none weekly): `c_sr = c_lr = 0.4` on `KRsGdf::build_rsh`
/// reproduces the global-hybrid force with fraction 0.4 on the plain fit,
/// `<= 1e-10` (measured 3.6e-14). The spin defects are visible here (the
/// global side is correct); the attenuated-fit and `c_sr v_M` defects are
/// blind by construction (coefficient 0, mixed constant `c v_M`).
#[test]
fn equal_fractions_are_the_global_hybrid_on_rsgdf() {
    let c = 0.4;
    let eq = RshParams {
        omega: 0.33,
        c_sr: c,
        c_lr: c,
    };
    {
        let spin = Spin::Uks;
        let cell = || cell_of(Lat::Tri, 2, &H3D);
        let zr = zx_open(cell(), spin, Some(eq), 1.0, rs::LINDEP_FD, ExxDiv::Ewald);
        let zg = zx_open(cell(), spin, None, c, rs::LINDEP_FD, ExxDiv::Ewald);
        assert!(zr.scf.asym() > 0.1, "polarized");
        let de = (zr.scf.energy() - zg.scf.energy()).abs();
        let (gr, gg) = (zx_open_force(&zr), zx_open_force(&zg));
        let d = diff(&gr.grad, &gg.grad);
        eprintln!(
            "{spin:?} equal fractions vs global c = {c}: |dE| {de:.1e}, force {d:.2e} (max|F| {:.2e})",
            max_abs(&gg.grad)
        );
        assert!(de < 1e-11, "{de:e}");
        assert!(d < ANCHOR_TOL, "{spin:?}: {d:e}");
        for n in ["ExchDropBeta", "ExchBetaUsesAlpha", "ExchClosedShellFactor"] {
            let (_, m, sm) = ledger_mutants().into_iter().find(|t| t.0 == n).unwrap();
            let miss = diff(&zx_open_force_m(&zr, m, sm).unwrap().grad, &gg.grad);
            eprintln!("  {spin:?} equal-fraction anchor vs {n}: {miss:.2e}");
            assert!(miss > 1e-3, "{n}: {miss:e}");
        }
    }
}

/// (c) Central FD (h = 1e-4) of the polarized zero-XC CAM-type UKS energy
/// on RS-GDF (H3 doublet, `TRI_A`, lindep 1e-6, ewald), one component (every
/// component, ROKS, the h sweep, HSE-type and KS rows weekly): measured
/// 3.9e-9 (bar 2e-8), `max|F|` 6.1e-2. The spin and attenuated-scale
/// mutants miss the production force by >= 1e-5 (smallest measured 3.1e-3,
/// ExchBetaUsesAlpha); the full ledger is weekly.
#[test]
fn polarized_cam_type_uks_force_matches_fd_and_sees_the_ledger() {
    let z = zx_open(
        cell_of(Lat::Tri, 2, &H3D),
        Spin::Uks,
        Some(cam_type()),
        1.0,
        rs::LINDEP_FD,
        ExxDiv::Ewald,
    );
    assert!(z.scf.asym() > 0.1, "polarized");
    let g = zx_open_force(&z);
    assert!(max_abs(&g.hf.parts.fit_attenuated) > 1e-3);
    assert!(g.hf.net_force < 1e-12, "{:e}", g.hf.net_force);
    let (a, x, h) = (0, 2, 1e-4);
    let fd = zx_open_fd(
        (Lat::Tri, 2),
        &H3D,
        (a, x, h),
        Spin::Uks,
        cam_type(),
        rs::LINDEP_FD,
        ExxDiv::Ewald,
    );
    let miss = (fd - g.grad[(a, x)]).abs();
    eprintln!(
        "UKS CAM-type FD miss {miss:.2e} (max|F| {:.2e})",
        max_abs(&g.grad)
    );
    assert!(miss < 2e-8, "{miss:e}");
    let per_commit = [
        "ExchDropBeta",
        "ExchBetaUsesAlpha",
        "ExchClosedShellFactor",
        "MTermTotalDensity",
        "RshFitNoScale",
    ];
    for (n, m, sm) in ledger_mutants() {
        if !per_commit.contains(&n) {
            continue; // the full ledger: weekly
        }
        let mm = diff(&zx_open_force_m(&z, m, sm).unwrap().grad, &g.grad);
        eprintln!("  UKS {n}: miss {mm:.2e}");
        assert!(mm > 1e-5, "{n}: {mm:e}");
    }
}

/// The open-shell STRESS keeps refusing a range-separated RS-GDF source
/// (a later #492 item): the UKS stress on the CAM-type build.
#[test]
fn open_shell_stress_still_refuses_rsgdf_rsh() {
    use ferric_pbc::kuxcstress::{kpoint_uks_stress, KUksStressConfig};
    let z = zx_open(
        cell_of(Lat::Tri, 2, &H3D),
        Spin::Uks,
        Some(cam_type()),
        1.0,
        rs::LINDEP_FD,
        ExxDiv::Ewald,
    );
    let OScf::U(scf) = &z.scf else { unreachable!() };
    let s = &z.sys;
    let e = kpoint_uks_stress(
        &s.cell,
        &s.prep,
        &s.mesh,
        &ferric_pbc::hcore::PeriodicHcoreConfig::with_omega(rs::OMEGA_HCORE),
        &s.hk,
        ferric_pbc::kstress::KStressJk::RsGdf(ferric_pbc::kgrad::KRsGdfGradSource {
            gdf: &z.gdf,
            cfg: &z.cfg,
            aux: &s.aux,
        }),
        scf,
        CAM,
        &KUksStressConfig::default(),
    )
    .unwrap_err()
    .to_string();
    eprintln!("{e}");
    assert!(e.contains("range-separated"), "{e}");
}
