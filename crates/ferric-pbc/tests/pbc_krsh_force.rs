//! Analytic forces of the closed-shell k-point RKS energy for RANGE-SEPARATED
//! hybrids on the dense-AFT J/K (`ferric_pbc::kxcgrad`, module doc
//! "Range-separated hybrids"), and the Gamma-point force routed through the
//! 1x1x1 mesh (`ferric_pbc::gamma_rsh::gamma_rks_rsh_gradient`).
//!
//! The derivative is the global-hybrid one at `a = 1` with the exchange walks
//! weighted by `f(K) = c_sr + (c_lr - c_sr) e^{-K^2/4w^2}` (a function of |K|
//! only: no new pair-FT derivative) and the SCF's mixed Madelung constant in
//! the M-term. Oracles (each with its artifact hypothesis in the test doc):
//!
//! 1. central FD of ferric's own energy (fresh SCF per displacement): one
//!    CAM-B3LYP row here, HSE06 and CAM-B3LYP under exxdiv ewald and none
//!    plus the FD floor sweep in `validation_krsh_force.rs`;
//! 2. `c_sr == c_lr == c` reproduces the global hybrid with fraction `c` (zero
//!    XC, two separate SCFs);
//! 3. the 1x1x3 mesh force equals the force on one copy in the explicit 3-cell
//!    supercell (1-point mesh there);
//! 4. Gamma (`gamma_rks_rsh` + `gamma_rks_rsh_gradient`) equals the 1x1x1 mesh
//!    (separate SCFs, Gamma J / hcore / grid vs the k ones).
//!
//! MUTATION LEDGER (production-wired `KGradMutation::Rsh*`), measured
//! 2026-10-18 (and reproduced by an independent review); miss = max
//! |F_mutant - F_ref| in Hartree/Bohr against the FD-validated production
//! force (1x1x3, ewald; it agrees with FD to 2.9e-9), FD bar 1e-7:
//!
//! | mutant            | CAM-B3LYP | HSE06   | equal-fraction anchor (ewald / none) |
//! |-------------------|-----------|---------|--------------------------------------|
//! | RshNoKernelFactor | 1.48e-1   | 1.64e-1 | 1.23e-1 / 1.23e-1                    |
//! | RshKernelOnJ      | 1.01e-1   | 9.03e-2 | 6.91e-2 / 6.91e-2                    |
//! | RshSwapFractions  | 3.00e-2   | 5.44e-2 | blind (c_sr = c_lr)                  |
//! | RshPlainMadelung  | 3.40e-2   | 5.27e-2 | 3.14e-2 / blind (no Madelung)        |
//! | RshOmegaScaled    | 5.20e-4   | 1.45e-5 | not run                              |
//! | RshNoErfPiece     | 3.77e-2   | 1.22e-4 | not run                              |
//! | RshNoMadelungTerm | 2.79e-2   | 5.12e-3 | not run                              |
//!
//! The erf (attenuated) piece is exercised by both functionals: the largest
//! `|c_lr - c_sr| e^{-K^2/4w^2}` over the 1x1x3 exchange K set is 0.245 for
//! CAM-B3LYP and 8.7e-4 for HSE06 (w = 0.11), yet dropping HSE06's erf piece
//! from the derivative misses by 1.2e-4 (1200 FD bars) and w x 1.01 by
//! 1.45e-5 (145 bars): the small-K (q != 0) terms carry it. Bars asserted:
//! 1e-5 (CAM-B3LYP), 1e-6 (HSE06, 10 FD bars).
//!
//! Anchor numbers (measured): equal fractions vs the global hybrid 5.6e-17
//! (energy difference 0); 1x1x3 vs supercell 1.0e-14 (HSE06), 1.1e-14
//! (CAM-B3LYP); Gamma vs 1x1x1 <= 1.1e-16 with the routed Fock guard at
//! <= 6.7e-16 (the two SCFs reach the same density to roundoff on this
//! one-orbital cell; the Gamma check is a GLUE check, the kernel is shared,
//! and a different grid in the Gamma config is refused by the guard).
//! Bars: 1e-9 for the anchors (>= 1e4 x measured, >= 3e7 below the
//! smallest mutant miss), FD_BAR 1e-7 for FD (35x the 2.9e-9 h^2 floor).
//!
//! The full FD matrix (both functionals x both exxdiv) and the FD floor sweep
//! are in `validation_krsh_force.rs`; this file runs one FD row.
//!
//! NOT covered here: RS-GDF (`pbc_krsh_rsgdf_force.rs`,
//! `validation_krsh_rsgdf_force.rs`; closed-shell RKS only, the Gamma route
//! refuses it), open shell (UKS on dense AFT: `pbc_kuks_rsh_force.rs`; ROKS
//! refused), p/d
//! shells (H2/STO-3G only), shifted meshes, a PySCF force pin, Python / CLI
//! (the shared derivative gates still refuse range-separated hybrids).

mod common;
#[path = "krsh_force_support/mod.rs"]
mod support;

use common::*;
use ferric_pbc::dense_aft::{DenseAftEri, ExxDiv, DEFAULT_DENSE_AFT_MAX_BYTES};
use ferric_pbc::dft::{gamma_rks_rsh, GammaRksConfig, PeriodicGridConfig};
use ferric_pbc::gamma_rsh::{gamma_rks_rsh_gradient, GammaRshExchange};
use ferric_pbc::hcore::{periodic_hcore, PeriodicHcoreConfig};
use ferric_pbc::kdense_aft::{KDenseAftConfig, KDenseAftEri};
use ferric_pbc::kgrad::{KGradConfig, KGradJk, KGradMutation};
use ferric_pbc::kpts::KPointMesh;
use ferric_pbc::kscf::KScfResult;
use ferric_pbc::kxcgrad::{kpoint_null_xc_gradient_rsh, kpoint_rks_gradient, KRksGradient};
use ferric_pbc::lattice::Cell;
use ferric_pbc::rsh::RshParams;
use ferric_pbc::uhf::GammaUhfIntegrals;
use ndarray::Array2;
use support::*;

/// (1, per-commit row) Central FD of the CAM-B3LYP energy (fresh SCF per
/// displacement, h = 1e-4), dense AFT, 1x1x3 (non-TRIM), exxdiv ewald, two
/// components. Physics: the h^2 floor (2.9e-9 measured); artifact: every
/// ledger mutant misses by >= 3e-2.
#[test]
fn cam_b3lyp_force_matches_fd() {
    let fx = fixture(&H2, [1, 1, 3], CAM, ExxDiv::Ewald);
    let g = force(&fx);
    for (a, x) in [(0, 2), (1, 1)] {
        let d = fd(&H2, a, x, 1e-4, [1, 1, 3], CAM, ExxDiv::Ewald);
        let r = (g.grad[(a, x)] - d).abs();
        eprintln!(
            "CAM-B3LYP ({a},{x}): analytic {:+.10e} FD {d:+.10e} |diff| {r:.2e}",
            g.grad[(a, x)]
        );
        assert!(r < FD_BAR, "({a},{x}): {r:e}");
    }
    assert!(max_abs(&g.hf.parts.exchange) > 1e-4, "K walks exercised");
}

// ---------------------------------------------------- equal fractions anchor

fn null_force(
    fx: &Fx,
    scf: &KScfResult,
    eri: &KDenseAftEri,
    a: f64,
    rsh: Option<RshParams>,
    m: Option<KGradMutation>,
) -> KRksGradient {
    kpoint_null_xc_gradient_rsh(
        &fx.cell,
        &fx.prep,
        &fx.mesh,
        &PeriodicHcoreConfig::with_omega(OMEGA),
        &fx.hk,
        KGradJk::Dense(eri),
        scf,
        a,
        rsh,
        &gcfg(fx.exxdiv, m),
    )
    .unwrap()
}

/// (2) EQUAL FRACTIONS: `c_sr = c_lr = c` makes the attenuated kernel `c 4pi/K^2`
/// and the mixed Madelung `c v_M`, i.e. the global hybrid with fraction `c`
/// (mirrors `rsh_with_equal_fractions_is_the_global_hybrid` for the energy).
/// Two SCFs (RSH kernels at scale 1 vs bare kernels at scale c, zero XC) and
/// two force paths (kernel factor vs exchange-coefficient scaling) must give
/// the same force to the SCF floor. Artifact: a factor applied to J, dropped,
/// or a plain Madelung in the M-term each move it by >= 1e-3 (asserted).
#[test]
fn rsh_with_equal_fractions_is_the_global_hybrid_force() {
    let c = 0.4;
    let eq = RshParams {
        omega: 0.33,
        c_sr: c,
        c_lr: c,
    };
    for exxdiv in [ExxDiv::Ewald, ExxDiv::None] {
        let (fr, sr, er) = zero_xc(exxdiv, Some(eq), 1.0);
        let (fg, sg, eg) = zero_xc(exxdiv, None, c);
        let de = (sr.energy - sg.energy).abs();
        let gr = null_force(&fr, &sr, &er, 1.0, Some(eq), None);
        let gg = null_force(&fg, &sg, &eg, c, None, None);
        let d = diff(&gr.grad, &gg.grad);
        let dk = diff(&gr.hf.parts.exchange, &gg.hf.parts.exchange);
        eprintln!("{exxdiv:?}: equal fractions vs global c={c}: |dE| {de:.1e} force {d:.2e} K part {dk:.2e}");
        assert!(de < 1e-11, "{de:e}");
        assert!(d < 1e-9, "{exxdiv:?}: {d:e}");
        assert!(max_abs(&gg.hf.parts.exchange) > 1e-4);
        for m in [
            KGradMutation::RshNoKernelFactor,
            KGradMutation::RshKernelOnJ,
            KGradMutation::RshPlainMadelung,
        ] {
            let miss = diff(
                &null_force(&fr, &sr, &er, 1.0, Some(eq), Some(m)).grad,
                &gg.grad,
            );
            eprintln!("  {exxdiv:?} equal-fraction anchor vs {m:?}: miss {miss:.2e}");
            if m == KGradMutation::RshPlainMadelung && exxdiv == ExxDiv::None {
                assert!(miss < 1e-9, "blind by construction under exxdiv none");
            } else {
                assert!(miss > 1e-3, "{m:?} {exxdiv:?}: {miss:e}");
            }
        }
    }
}

// ------------------------------------------------------------ supercell

/// (3) SUPERCELL: the 1x1x3 mesh force on atom A is the force on ONE copy of A
/// in the explicit 3-cell supercell (1-point mesh: no q classes, the
/// supercell's own primitive Madelung constants). Physics: equal to the SCF
/// floor; artifact: a wrong q-dependent kernel argument or a mesh-vs-primitive
/// Madelung mix-up shows as an O(1e-3) gap.
#[test]
fn rsh_k_mesh_force_is_the_supercell_force() {
    for f in ["HSE06", CAM] {
        let k = force(&fixture(&H2, [1, 1, 3], f, ExxDiv::Ewald));
        let sc = force(&fixture_cell(
            supercell(&H2, [1, 1, 3]),
            [1, 1, 1],
            f,
            ExxDiv::Ewald,
        ));
        let first = sc.grad.slice(ndarray::s![0..2, ..]).to_owned();
        let d = diff(&k.grad, &first);
        let dk = diff(
            &k.hf.parts.exchange,
            &sc.hf.parts.exchange.slice(ndarray::s![0..2, ..]).to_owned(),
        );
        eprintln!("{f}: 1x1x3 vs 3-cell supercell force {d:.2e} (exchange part {dk:.2e})");
        assert!(d < 1e-9, "{f}: {d:e}");
    }
}

// ----------------------------------------------------------------- Gamma

fn gamma_force(f: &str, exxdiv: ExxDiv) -> KRksGradient {
    let cell = cell_of(&H2);
    let prep = prep_for(&cell, &pyscf_sto3g_h());
    let hcfg = PeriodicHcoreConfig::with_omega(OMEGA);
    let hc = periodic_hcore(&cell, &prep, &hcfg).unwrap();
    let eri = DenseAftEri::build(
        &cell,
        &prep,
        &hc.s,
        exxdiv,
        PRECISION,
        DEFAULT_DENSE_AFT_MAX_BYTES,
    )
    .unwrap();
    let dc = KDenseAftConfig {
        precision: PRECISION,
        budget_bytes: Some(AMPLE),
        ..Default::default()
    };
    let ex = GammaRshExchange::build_dense(&cell, &prep, &hc.s, exxdiv, &dc, rsh_of(f)).unwrap();
    let mut gc = GammaRksConfig::new(f);
    gc.grid = grid_cfg();
    gc.exxdiv = exxdiv;
    gc.scf.energy_conv = 1e-13;
    gc.scf.density_conv = 1e-10;
    gc.scf.max_iter = 200;
    let r = gamma_rks_rsh(
        &cell,
        &prep,
        &hc,
        GammaUhfIntegrals::DenseAft(&eri),
        &ex,
        &gc,
    )
    .unwrap();
    let kc = KGradConfig {
        budget_bytes: Some(AMPLE),
        ..Default::default()
    };
    // The routed Fock guard refuses a config that is not the SCF's (another
    // grid gives another V_xc): the glue cannot silently differentiate a
    // different energy.
    let mut other = gc.clone();
    other.grid = PeriodicGridConfig::with_size(30, 110);
    other.grid.neighbour_cutoff = Some(CUTOFF);
    let e = gamma_rks_rsh_gradient(&cell, &prep, &hcfg, &r.scf, &other, &ex, &kc)
        .expect_err("another grid accepted")
        .to_string();
    assert!(e.contains("differs from the SCF's stored fock"), "{e}");
    if f == "HSE06" && exxdiv == ExxDiv::Ewald {
        gamma_refusals(&cell, &prep, &hc.s, &hcfg, &r.scf, &gc, &kc, f);
    }
    gamma_rks_rsh_gradient(&cell, &prep, &hcfg, &r.scf, &gc, &ex, &kc).unwrap()
}

/// Typed refusals of the routed Gamma force: an RS-GDF exchange (the Gamma
/// route differentiates the dense one only; the k force covers RS-GDF) and an open-shell result (here the
/// restricted result relabelled `Spin::Unrestricted`, which must be refused by
/// the spin check, not differentiated as closed shell).
#[allow(clippy::too_many_arguments)]
fn gamma_refusals(
    cell: &Cell,
    prep: &ferric_integrals::basis_bridge::PreparedBasis,
    s: &Array2<f64>,
    hcfg: &PeriodicHcoreConfig,
    scf: &ferric_scf::result::ScfResult,
    gc: &GammaRksConfig,
    kc: &KGradConfig,
    f: &str,
) {
    let aux = ferric_integrals::basis_bridge::PreparedBasis::new(
        cell.mol(),
        &ferric_core::basis::bundled("cc-pvdz-ri").unwrap(),
    )
    .unwrap();
    let rcfg = ferric_pbc::rsgdf::kpoint::KRsGdfConfig {
        gdf: ferric_pbc::rsgdf::RsGdfConfig {
            omega: 1.0,
            exxdiv: gc.exxdiv,
            budget_bytes: Some(AMPLE),
            ..Default::default()
        },
        mutation: None,
    };
    let ex_gdf =
        GammaRshExchange::build_rsgdf(cell, prep, &aux, s, gc.exxdiv, &rcfg, rsh_of(f)).unwrap();
    let e = gamma_rks_rsh_gradient(cell, prep, hcfg, scf, gc, &ex_gdf, kc)
        .expect_err("RS-GDF Gamma exchange accepted")
        .to_string();
    eprintln!("Gamma RS-GDF refusal: {e}");
    assert!(e.contains("RS-GDF") && e.contains("no derivative"), "{e}");
    let dc = KDenseAftConfig {
        precision: PRECISION,
        budget_bytes: Some(AMPLE),
        ..Default::default()
    };
    let ex = GammaRshExchange::build_dense(cell, prep, s, gc.exxdiv, &dc, rsh_of(f)).unwrap();
    let mut open = scf.clone();
    open.spin = ferric_scf::result::Spin::Unrestricted;
    let e = gamma_rks_rsh_gradient(cell, prep, hcfg, &open, gc, &ex, kc)
        .expect_err("open-shell result accepted")
        .to_string();
    eprintln!("Gamma open-shell refusal: {e}");
    assert!(e.contains("closed-shell (restricted) results only"), "{e}");
}

/// (4) GAMMA == 1x1x1: the Gamma RSH force (Gamma SCF, Gamma J and hcore,
/// routed into the 1x1x1 force) against the 1x1x1 k SCF + force. Physics: the
/// SCF floor (~1e-10); artifact: a glue slip (wrong density factor, Fock,
/// k-point, Madelung) shows as O(1e-3), a grid mismatch fails the Fock guard.
#[test]
fn gamma_rsh_force_is_the_one_point_mesh_force() {
    for f in ["HSE06", CAM] {
        for exxdiv in [ExxDiv::Ewald, ExxDiv::None] {
            let g = gamma_force(f, exxdiv);
            let k = force(&fixture(&H2, [1, 1, 1], f, exxdiv));
            let d = diff(&g.grad, &k.grad);
            eprintln!(
                "{f} {exxdiv:?}: Gamma vs 1x1x1 force {d:.2e} (Fock guard {:.1e}, max|F| {:.3e})",
                g.hf.fock_mismatch,
                max_abs(&g.grad)
            );
            assert!(d < 1e-9, "{f} {exxdiv:?}: {d:e}");
            assert!(max_abs(&g.grad) > 1e-2);
        }
    }
}

// ----------------------------------------------------------------- mutants

/// The erf piece is live on the CAM-B3LYP fixture and nearly dead for HSE06
/// at this cell (measured max weight printed); every `Rsh*` mutant moves the
/// FD-validated CAM-B3LYP force by far more than the FD bar.
#[test]
fn rsh_mutants_are_caught() {
    let cell = cell_of(&H2);
    let mesh = KPointMesh::gamma_centred(&cell, [1, 1, 3]).unwrap();
    for f in ["HSE06", CAM] {
        eprintln!(
            "{f}: max erf weight |c_lr - c_sr| e^(-K^2/4w^2) on the 1x1x3 K set {:.2e}",
            max_erf_weight(&cell, &mesh, rsh_of(f))
        );
    }
    assert!(max_erf_weight(&cell, &mesh, rsh_of(CAM)) > 0.1);
    // Each functional's production force is FD-validated (FD_BAR) in
    // `validation_krsh_force.rs` (and CAM-B3LYP per commit above), so a miss
    // against it is a miss against FD to within 2.9e-9.
    for (f, bar) in [(CAM, 1e-5), ("HSE06", 10.0 * FD_BAR)] {
        let fx = fixture(&H2, [1, 1, 3], f, ExxDiv::Ewald);
        let base = force(&fx);
        for m in [
            KGradMutation::RshNoKernelFactor,
            KGradMutation::RshKernelOnJ,
            KGradMutation::RshSwapFractions,
            KGradMutation::RshPlainMadelung,
            KGradMutation::RshOmegaScaled,
            KGradMutation::RshNoErfPiece,
            KGradMutation::RshNoMadelungTerm,
        ] {
            let miss = diff(&force_m(&fx, Some(m)).unwrap().grad, &base.grad);
            eprintln!(
                "{f} 1x1x3 ewald: {m:?} miss {miss:.2e} ({:.0} FD bars)",
                miss / FD_BAR
            );
            assert!(miss > bar, "{f} {m:?}: {miss:e}");
        }
    }
}

// --------------------------------------------------------------- refusals

/// Typed refusals: kernels without the functional's range separation, with
/// other parameters, and the open-shell allowlist (the routed Gamma force's
/// RS-GDF and open-shell refusals are in `gamma_refusals`; the k-point RS-GDF
/// source checks are in `pbc_krsh_rsgdf.rs`).
#[test]
fn rsh_force_refusals() {
    let fx = fixture(&H2, [1, 1, 1], "HSE06", ExxDiv::Ewald);
    let call = |eri: &KDenseAftEri, f: &str| {
        kpoint_rks_gradient(
            &fx.cell,
            &fx.prep,
            &fx.mesh,
            &PeriodicHcoreConfig::with_omega(OMEGA),
            &fx.hk,
            KGradJk::Dense(eri),
            &fx.res.scf,
            f,
            &gcfg(ExxDiv::Ewald, None),
        )
        .map(|_| ())
        .map_err(|e| e.to_string())
    };
    let bare = eri_of(&fx, None);
    let e = call(&bare, "HSE06").unwrap_err();
    assert!(e.contains("bare Coulomb exchange"), "{e}");
    let other = eri_of(&fx, Some(rsh_of(CAM)));
    let e = call(&other, "HSE06").unwrap_err();
    assert!(e.contains("functional's is"), "{e}");
    assert!(call(&eri_of(&fx, Some(rsh_of("HSE06"))), "HSE06").is_ok());
    // The Python / CLI derivative gate: the ROKS force keeps refusing range
    // separation; the UKS force is wired on the dense-AFT J/K (the library
    // UKS force is `pbc_kuks_rsh_force.rs`, the closed-shell stress
    // `pbc_krsh_stress.rs`).
    use ferric_pbc::kxcgrad::{check_deriv_functional, Deriv, DerivSpin};
    assert!(
        check_deriv_functional(DerivSpin::RestrictedOpen, Deriv::Force, false, "HSE06").is_err()
    );
    check_deriv_functional(DerivSpin::Unrestricted, Deriv::Force, false, "HSE06").unwrap();
}
