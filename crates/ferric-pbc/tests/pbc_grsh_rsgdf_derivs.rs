//! Gamma-point range-separated hybrid (CAM-B3LYP) nuclear forces and stress
//! on RS-GDF J/K (`ferric_pbc::gamma_rsh::gamma_{rks,uks,roks}_rsh_gradient`,
//! `gamma_rks_rsh_stress` handed a `GammaRshExchange::build_rsgdf` exchange;
//! mgoldey/ferric#492), per commit.
//!
//! The Gamma point IS the 1x1x1 mesh: the routes hand the Gamma result to the
//! 1x1x1 k-point force / stress with the `k = 0` RS-GDF fit the SCF's
//! exchange ran on (`KRsGdf::build_rsh`, its config and aux), whose plain fit
//! also rebuilds the Gamma J. No new derivative algebra.
//!
//! Oracles (artifact hypothesis in each test doc):
//! 1. (anchor) Gamma == an INDEPENDENT 1x1x1 k SCF + force / stress (its own
//!    hcore, J, grid and SCF), RKS / UKS / ROKS, exxdiv ewald (+ none, RKS);
//! 2. the routed Fock guard refuses an SCF whose Gamma J is not the
//!    exchange's fit (dense J under an RS-GDF exchange and the reverse);
//! 3. central FD of ferric's own Gamma RS-GDF energy (fresh fits + SCF per
//!    point): one force component here, the rest in
//!    `validation_grsh_rsgdf_derivs.rs`.
//!
//! The shared k-point derivative itself (attenuated fit passes, mixed
//! Madelung) is validated by `pbc_krsh_rsgdf_force.rs`,
//! `pbc_kuks_rsh_rsgdf_force.rs` and their validation files; this file checks
//! the Gamma GLUE onto it plus one FD row (two force components).

mod common;
#[path = "grsh_rsgdf_support/mod.rs"]
mod gs;
#[path = "krsh_rsgdf_force_support/mod.rs"]
mod rs;

use ferric_pbc::dense_aft::ExxDiv;
use gs::*;

/// (1) GAMMA == 1x1x1 (force). Physics: the two SCFs reach the same density
/// to the SCF floor and the Gamma RS-GDF J equals the k = 0 plain fit's J, so
/// the forces agree to ~1e-10. Artifact: a wrong source (the dense kernels,
/// the plain fit without the attenuated one, another split) either fails the
/// routed Fock guard or misses by >= 1e-3; a density / spin factor slip in
/// the Gamma -> 1x1x1 glue shows as O(1e-2). Measured 4.4e-16 (RKS), 1.9e-14
/// (UKS), 1.1e-14 (ROKS), the Fock guard at <= 5.6e-16: on these minimal
/// cells the two SCFs land on the same density to roundoff, so this is a
/// GLUE check (the kernel is shared); the FD test below (two components)
/// and the validation
/// file check the kernel. exxdiv none is in the validation file.
#[test]
fn gamma_rsgdf_rsh_rks_force_is_the_one_point_mesh_force() {
    one_point_force_anchor(Sp::Rks);
}

/// (1) as above, UKS (H3 doublet).
#[test]
fn gamma_rsgdf_rsh_uks_force_is_the_one_point_mesh_force() {
    one_point_force_anchor(Sp::Uks);
}

/// (1) as above, ROKS (H3 doublet).
#[test]
fn gamma_rsgdf_rsh_roks_force_is_the_one_point_mesh_force() {
    one_point_force_anchor(Sp::Roks);
}

/// The Gamma == 1x1x1 force anchor of one spin treatment (exxdiv ewald).
fn one_point_force_anchor(sp: Sp) {
    let x = ExxDiv::Ewald;
    let g = gamma_run(cell_at(sp.atoms()), sp, CAM, x, Jk::RsGdf);
    let gf = gamma_force(&g).unwrap_or_else(|e| panic!("{sp:?} {x:?}: {e}"));
    let k = k1_run(cell_at(sp.atoms()), sp, CAM, x);
    let kf = k1_force(&k, CAM, x);
    let d = diff(&gf.grad, &kf.grad);
    eprintln!(
        "{sp:?} {x:?}: Gamma vs 1x1x1 RS-GDF force {d:.2e} (|F| {:.2e}, Fock guard {:.1e}, \
         dE {:.1e})",
        max_abs(&gf.grad),
        gf.hf.fock_mismatch,
        g.scf.energy - kscf_energy(&k)
    );
    assert!(d < ANCHOR_TOL, "{sp:?} {x:?}: {d:e}");
    assert!(max_abs(&gf.grad) > 1e-2, "a non-trivial force");
}

fn kscf_energy(k: &K1) -> f64 {
    match &k.scf {
        KScf::R(r) => r.energy,
        KScf::U(r) => r.energy,
        KScf::Ro(r) => r.energy,
    }
}

/// (1') GAMMA == 1x1x1 (stress, closed shell). As the force, on all nine
/// components of `dE/dε`.
#[test]
fn gamma_rsgdf_rsh_stress_is_the_one_point_mesh_stress() {
    for x in [ExxDiv::Ewald, ExxDiv::None] {
        let g = gamma_run(cell_at(&H2), Sp::Rks, CAM, x, Jk::RsGdf);
        let gs = gamma_stress(&g).unwrap_or_else(|e| panic!("{x:?}: {e}"));
        let k = k1_run(cell_at(&H2), Sp::Rks, CAM, x);
        let ks = k1_stress(&k, CAM, x);
        let d = mdiff(&gs.de_deps, &ks.de_deps);
        let big = gs
            .de_deps
            .iter()
            .flatten()
            .fold(0.0_f64, |m, v| m.max(v.abs()));
        eprintln!("RKS {x:?}: Gamma vs 1x1x1 RS-GDF dE/deps {d:.2e} (max {big:.2e})");
        assert!(d < ANCHOR_TOL, "{x:?}: {d:e}");
        assert!(big > 1e-2);
    }
}

/// (2) A J THAT IS NOT THE EXCHANGE'S FIT IS REFUSED. The routed force
/// rebuilds `h + J + V_xc − K` from the exchange's source; an SCF that ran
/// its Gamma J on the other source (dense J with an RS-GDF exchange, RS-GDF
/// J with a dense one) differs by the fit error (~1e-5) and must be refused
/// by the Fock guard, not differentiated as if consistent.
#[test]
fn gamma_rsh_route_refuses_a_j_that_is_not_the_exchanges() {
    for (sp, jk) in [
        (Sp::Rks, Jk::DenseJRsGdfK),
        (Sp::Rks, Jk::RsGdfJDenseK),
        (Sp::Uks, Jk::DenseJRsGdfK),
        (Sp::Roks, Jk::DenseJRsGdfK),
    ] {
        let g = gamma_run(cell_at(sp.atoms()), sp, CAM, ExxDiv::Ewald, jk);
        let e = gamma_force(&g)
            .expect_err("a foreign J accepted")
            .to_string();
        eprintln!("{sp:?} {jk:?}: {e}");
        assert!(
            e.contains("differs from the SCF's stored"),
            "{sp:?} {jk:?}: {e}"
        );
        if sp == Sp::Rks {
            let e = gamma_stress(&g)
                .expect_err("a foreign J accepted (stress)")
                .to_string();
            assert!(e.contains("differs from the SCF's stored"), "{jk:?}: {e}");
        }
    }
}

/// (2') AN RS-GDF GAMMA J ON ANOTHER FIT than the exchange's (closed shell,
/// exchange built at the production config: metric cut 1e-6, ω₀ = 1, ET-sp
/// aux). Only the routed Fock guard (1e-8) ties the two together, so this
/// pins what it does. Measured 2026-10-26:
///
/// | Gamma J fit                 | ΔE vs consistent | outcome                  |
/// |-----------------------------|------------------|--------------------------|
/// | metric cut 1e-8             | 0 (bitwise)      | accepted, force Δ 0      |
/// | metric cut 1e-2 (drops 2)   | −1.5e-7          | refused, Fock Δ 1.26e-6  |
/// | ω₀ = 0.9                    | −6.7e-15         | accepted, force Δ 2.8e-16 |
/// | aux + extra s shell         | +4.1e-6          | refused, Fock Δ 5.44e-6  |
///
/// The plain-metric eigenvalues of this aux are all >= 5.3e-3, so a cut of
/// 1e-8 (or 1e-4) changes nothing, and the RS-GDF J does not depend on the
/// split ω₀ beyond the integral precision: those J are the SAME J, and their
/// forces equal the consistent (FD-validated) one. Every fit change that
/// moves the J is refused. Not covered: a fit change that moves the Fock by
/// less than the 1e-8 bar would pass, with a force error of that order.
#[test]
fn gamma_rsh_route_on_a_differently_configured_rsgdf_j() {
    let x = ExxDiv::Ewald;
    let base = gamma_run(cell_at(&H2), Sp::Rks, CAM, x, Jk::RsGdf);
    let fb = gamma_force(&base).unwrap();
    for (m, same) in [
        (Mis::Lindep, true),
        (Mis::LindepLoose, false),
        (Mis::Omega0, true),
        (Mis::Aux, false),
    ] {
        let g = gamma_run(cell_at(&H2), Sp::Rks, CAM, x, Jk::RsGdfJOther(m));
        let de = g.scf.energy - base.scf.energy;
        match gamma_force(&g) {
            Ok(gf) => {
                let d = diff(&gf.grad, &fb.grad);
                eprintln!(
                    "{m:?}: dE {de:.2e} accepted (Fock {:.1e}), force vs consistent {d:.2e}",
                    gf.hf.fock_mismatch
                );
                assert!(same, "{m:?}: a J that moved by {de:e} was accepted");
                assert!(d < 1e-12, "{m:?}: {d:e}");
            }
            Err(e) => {
                let e = e.to_string();
                eprintln!("{m:?}: dE {de:.2e} refused: {e}");
                assert!(!same, "{m:?}: the same J was refused: {e}");
                assert!(e.contains("differs from the SCF's stored"), "{m:?}: {e}");
            }
        }
    }
}

/// (3, per-commit row) Central FD of the Gamma RS-GDF CAM-B3LYP RKS energy
/// (fresh fits + SCF per displacement, h = 1e-4), exxdiv ewald, two
/// components. Physics: the h² floor (~2e-9, as the k-point RS-GDF KS row);
/// artifact: the k-point ledger mutants miss by >= 2e-2.
#[test]
fn gamma_rsgdf_rsh_force_matches_fd() {
    let g = gamma_run(cell_at(&H2), Sp::Rks, CAM, ExxDiv::Ewald, Jk::RsGdf);
    let gf = gamma_force(&g).unwrap();
    for (a, x) in [(0, 2), (1, 0)] {
        let d = fd_force(Sp::Rks, CAM, ExxDiv::Ewald, (a, x, 1e-4));
        let r = (gf.grad[(a, x)] - d).abs();
        eprintln!(
            "RKS ({a},{x}): analytic {:+.10e} FD {d:+.10e} |diff| {r:.2e}",
            gf.grad[(a, x)]
        );
        assert!(r < FD_BAR, "({a},{x}): {r:e}");
    }
}

/// Anchor bar (Gamma == 1x1x1).
const ANCHOR_TOL: f64 = 1e-9;
/// FD bar at h = 1e-4.
const FD_BAR: f64 = 5e-8;
