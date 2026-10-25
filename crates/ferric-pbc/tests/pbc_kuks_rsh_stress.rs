//! Per-commit anchors of the OPEN-SHELL (UKS, ROKS) range-separated k-point
//! STRESS on the dense-AFT and the RS-GDF J/K (`ferric_pbc::kuxcstress`,
//! module doc "Range-separated hybrids (UKS and ROKS)"), and of its Gamma
//! route (`ferric_pbc::gamma_rsh::gamma_uks_rsh_stress`).
//!
//! # Derivation (a composition, no new derivative algebra)
//!
//! The closed-shell RSH stress (#503 dense, #514 RS-GDF) differentiates the
//! exchange with the term list `Σ_s c_s D_s X D_s` (`(½, D)`), the M-term
//! sandwich `½ D S D`, the attenuated kernel's strain `dv/dK² = −v/K² +
//! (4π/K²) f'(K²)` (dense) or the attenuated fit's LR strain passes scaled
//! by `c_lr − c_sr` (RS-GDF), and the mixed Madelung strain `c_sr dv_M/dε +
//! (c_lr − c_sr) dv_M^LR/dε`. The kernel and Madelung strains are scalars
//! that multiply a quantity bilinear in the term list or the sandwich, so
//! they do not see the spin. The open-shell assembly already exists for the
//! global-hybrid UKS / ROKS stress with the term list `(1, D_α), (1, D_β)`
//! and the sandwich `Σ_σ D_σ S D_σ`. So the open-shell RSH stress is that
//! assembly with the range-separated J/K, exchange scale 1 and the mixed
//! `v_M` in the Fock rebuild; the code change is lifting the refusal and
//! threading `rsh` into the input check and the Madelung shift. No Python
//! prototype was needed: no strain derivative assumes closed shell.
//!
//! # Hypotheses (written before measuring)
//!
//! * (a) SINGLET: on a closed-shell cell the UKS path reproduces the RKS RSH
//!   stress (independent SCFs and entries) to `<= 1e-10` per part, dense and
//!   RS-GDF. Artifact: a per-spin factor slip (`ExchClosedShellFactor`, a
//!   dropped β term) shows as O(1e-2); spin-reassignment defects are
//!   invisible by construction, hence (b)-(c) on a polarized cell.
//! * (b) EQUAL FRACTIONS `c_sr = c_lr = c`: the RSH stress equals the
//!   global-hybrid stress with fraction `c` (polarized, both J/K, UKS and
//!   ROKS). Blind to the attenuated piece (coefficient 0) and to M-term spin
//!   errors (both sides share the sandwich); FD covers those.
//! * (c) FD of ferric's own polarized zero-XC energy under strain (no grid,
//!   so no neighbour-cutoff steps, #495): physics is an h² truncation
//!   (ratio ≈ 4 per halving) with a Richardson residue at the SCF floor;
//!   artifact: every ledger mutant misses by orders of magnitude more.
//!
//! PySCF has no oracle: its periodic stress (`pbc/grad/krks_stress.py`,
//! `kuks_stress.py`) refuses hybrids. MEASURED and the full MUTATION LEDGER:
//! `validation_kuks_rsh_stress.rs`.

mod common;
#[path = "krsh_rsgdf_force_support/mod.rs"]
mod rs;
#[path = "kuks_rsh_stress_support/mod.rs"]
mod support;

use ferric_pbc::dense_aft::ExxDiv;
use ferric_pbc::kstress::KStressMutation as M;
use ferric_pbc::rsh::RshParams;
use support::*;

/// (a) SINGLET: zero-XC CAM-type RSH on the H2 singlet in `TRI_A`: the UKS
/// stress (UKS SCF, UKS entry) equals the closed-shell RKS stress
/// (independent RKS SCF and entry) per part, `<= 1e-10`, on each J/K. The
/// factor defects move it; the spin-reassignment defects cannot.
fn singlet_case(route: Route) {
    let u = zx(
        h2s(),
        (route, Spin::Uks),
        Some(cam_type()),
        1.0,
        ExxDiv::Ewald,
    );
    let r = zx(
        h2s(),
        (route, Spin::Rks),
        Some(cam_type()),
        1.0,
        ExxDiv::Ewald,
    );
    assert!(u.scf.asym() < 1e-8, "singlet");
    let su = zx_stress(&u);
    let sr = zx_stress(&r);
    let (p, q) = (&su.hf.parts, &sr.hf.parts);
    let rows = [
        ("energy", (u.scf.energy() - r.scf.energy()).abs()),
        ("exchange", mdiff(&p.exchange, &q.exchange)),
        (
            "fit_attenuated",
            mdiff(&p.fit_attenuated, &q.fit_attenuated),
        ),
        ("madelung", mdiff(&p.madelung, &q.madelung)),
        (
            "overlap_madelung",
            mdiff(&p.overlap_madelung, &q.overlap_madelung),
        ),
        ("overlap_pulay", mdiff(&p.overlap_pulay, &q.overlap_pulay)),
        ("total", mdiff(&su.de_deps, &sr.de_deps)),
    ];
    for (n, d) in rows {
        eprintln!("{route:?} singlet UKS vs RKS {n}: {d:.2e}");
        assert!(d < ANCHOR_TOL, "{route:?} {n}: {d:e}");
    }
    let live = match route {
        Route::Dense => mmax(&p.exchange),
        Route::RsGdf => mmax(&p.fit_attenuated),
    };
    assert!(
        live > 1e-4,
        "{route:?}: the attenuated exchange is live ({live:e})"
    );
    assert!(mmax(&p.madelung) > 1e-4, "Madelung strain live");
    for (n, m) in [
        ("ExchDropBeta", M::ExchDropBeta),
        ("ExchClosedShellFactor", M::ExchClosedShellFactor),
        ("MTermTotalDensity", M::MTermTotalDensity),
    ] {
        let miss = mdiff(
            &zx_stress_m(&u, Some(m), None).unwrap().de_deps,
            &su.de_deps,
        );
        eprintln!("{route:?} singlet: {n} miss {miss:.2e}");
        if SPIN_ONLY.contains(&n) {
            assert!(miss < ANCHOR_TOL, "{n} blind on a singlet: {miss:e}");
        } else {
            assert!(miss > 1e-3, "{route:?} {n}: {miss:e}");
        }
    }
}

#[test]
fn singlet_uks_is_the_rks_rsh_stress_dense() {
    singlet_case(Route::Dense);
}

#[test]
fn singlet_uks_is_the_rks_rsh_stress_rsgdf() {
    singlet_case(Route::RsGdf);
}

/// (b) EQUAL FRACTIONS on the polarized H3 doublet (ewald): `c_sr = c_lr =
/// 0.4` reproduces the global-hybrid stress with fraction 0.4 on bare
/// kernels / the plain fit, `<= 1e-10`. A spin defect on the RSH side is
/// visible (the global side is correct; the spin mutants against this
/// anchor, dense ROKS and exxdiv none are weekly).
pub fn equal_fraction_case(route: Route, spin: Spin) {
    let c = 0.4;
    let eq = RshParams {
        omega: 0.33,
        c_sr: c,
        c_lr: c,
    };
    let zr = zx(h3d(), (route, spin), Some(eq), 1.0, ExxDiv::Ewald);
    let zg = zx(h3d(), (route, spin), None, c, ExxDiv::Ewald);
    assert!(zr.scf.asym() > 0.1, "polarized");
    let de = (zr.scf.energy() - zg.scf.energy()).abs();
    let sr = zx_stress(&zr);
    let sg = zx_stress(&zg);
    let d = mdiff(&sr.de_deps, &sg.de_deps);
    eprintln!("{route:?} {spin:?}: equal fractions vs global c={c}: |dE| {de:.1e} stress {d:.2e}");
    assert!(de < 1e-10, "{de:e}");
    assert!(d < ANCHOR_TOL, "{route:?} {spin:?}: {d:e}");
}

#[test]
fn equal_fractions_is_the_global_hybrid_stress_uks_dense() {
    equal_fraction_case(Route::Dense, Spin::Uks);
}

#[test]
fn equal_fractions_is_the_global_hybrid_stress_uks_rsgdf() {
    equal_fraction_case(Route::RsGdf, Spin::Uks);
}

#[test]
fn equal_fractions_is_the_global_hybrid_stress_roks_rsgdf() {
    equal_fraction_case(Route::RsGdf, Spin::Roks);
}

/// FD step and bar of the per-commit FD row (h²-dominated at this step; the
/// h-sweep and the Richardson check are weekly,
/// `validation_kuks_rsh_stress.rs`, where the measured numbers live).
pub const FD_H: f64 = 1e-4;
pub const FD_BAR: f64 = 1e-7;

/// (c) FD of the polarized zero-XC UKS energy, CAM-type, ewald, on the
/// diagonal strain `(1, 1)` (the largest component on this cell), plus the
/// attenuated piece's strain dropped (the dense `f'` slope / the RS-GDF
/// attenuated fit pass) missing by more than 100 bars. The spin-factor
/// mutants are per commit in (a); the rest of the ledger is weekly (each
/// stress here costs ~11-13 s on one thread).
fn fd_case(route: Route) {
    let cell = h3d();
    let rsh = cam_type();
    let z = zx(
        cell.clone(),
        (route, Spin::Uks),
        Some(rsh),
        1.0,
        ExxDiv::Ewald,
    );
    let s = zx_stress(&z);
    let (i, j) = (1, 1);
    let fd = zx_fd(&cell, (i, j, FD_H), (route, Spin::Uks), rsh, ExxDiv::Ewald);
    let miss = (fd - s.de_deps[i][j]).abs();
    eprintln!(
        "{route:?} ({i},{j}): dE/de {:.6e}  FD miss {miss:.2e}",
        s.de_deps[i][j]
    );
    assert!(miss < FD_BAR, "{route:?} ({i},{j}): {miss:e}");
    let att = match route {
        Route::Dense => M::RshNoKernelSlope,
        Route::RsGdf => M::RshFitNoAttenuated,
    };
    let mm = (zx_stress_m(&z, Some(att), None).unwrap().de_deps[i][j] - fd).abs();
    eprintln!("{route:?} {att:?}: FD miss {mm:.2e}");
    assert!(mm > 100.0 * FD_BAR, "{route:?} {att:?}: {mm:e}");
}

#[test]
fn fd_zero_xc_polarized_uks_dense() {
    fd_case(Route::Dense);
}

#[test]
fn fd_zero_xc_polarized_uks_rsgdf() {
    fd_case(Route::RsGdf);
}

/// Typed refusals of the open-shell stress: a range-separated name on bare
/// kernels, kernels of another range separation, a global-hybrid name on
/// range-separated kernels, a missing exxdiv; and the binding gate keeps
/// refusing the open-shell RSH stress (library only, #498).
#[test]
fn oshell_rsh_stress_refusals() {
    use ferric_pbc::kuxcstress::{kpoint_uks_stress, KUksStressConfig};
    let k = ks(h3d(), (Route::Dense, Spin::Uks), "HSE06", ExxDiv::Ewald);
    let Scf::U(scf) = &k.scf else { unreachable!() };
    let s = &k.sys;
    let call = |eri: &ferric_pbc::kdense_aft::KDenseAftEri, f: &str, ewald: bool| {
        let mut base = base_cfg(ExxDiv::Ewald, None);
        base.grid = ferric_pbc::kxcgrad::KXcGrid::Atom(grid_cfg());
        if !ewald {
            base.exxdiv = None;
        }
        kpoint_uks_stress(
            &s.cell,
            &s.prep,
            &s.mesh,
            &hcore(),
            &k.kept.hk,
            ferric_pbc::kstress::KStressJk::Dense(eri),
            scf,
            f,
            &KUksStressConfig {
                base,
                spin_mutation: None,
            },
        )
    };
    let build = |r| {
        ferric_pbc::kdense_aft::KDenseAftEri::build(
            &s.cell,
            &s.prep,
            &s.mesh,
            &s.hk.s,
            ExxDiv::None,
            &dense_cfg(r),
        )
        .unwrap()
    };
    let bare = build(None);
    let own = build(Some(rsh_of("HSE06")));
    let other = build(Some(rsh_of(CAM)));
    let e = call(&bare, "HSE06", true).unwrap_err().to_string();
    assert!(e.contains("bare Coulomb exchange"), "{e}");
    let e = call(&other, "HSE06", true).unwrap_err().to_string();
    assert!(e.contains("functional's is"), "{e}");
    let e = call(&own, "PBE0", true).unwrap_err().to_string();
    assert!(e.contains("does not differentiate it"), "{e}");
    let e = call(&own, "HSE06", false).unwrap_err().to_string();
    assert!(e.contains("needs KRksStressConfig::exxdiv"), "{e}");
    let ok = call(&own, "HSE06", true).unwrap_or_else(|e| panic!("own kernels: {e}"));
    assert!(ok.hf.fock_mismatch < 1e-8 && ok.de_deps.iter().flatten().all(|v| v.is_finite()));
    use ferric_pbc::kxcgrad::{check_deriv_functional, Deriv, DerivSpin};
    for f in FUNCTIONALS {
        for (spin, g) in [
            (DerivSpin::Unrestricted, false),
            (DerivSpin::Unrestricted, true),
            (DerivSpin::RestrictedOpen, false),
        ] {
            let e = check_deriv_functional(spin, Deriv::Stress, g, f)
                .unwrap_err()
                .to_string();
            assert!(e.contains("not implemented in the bindings"), "{f}: {e}");
        }
    }
}
