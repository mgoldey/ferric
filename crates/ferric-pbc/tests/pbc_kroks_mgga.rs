//! k-point ROKS meta-GGA (`ferric_pbc::kroks::solve_kroks`, pure SCAN and
//! r2SCAN, ENERGY ONLY; issue #492, item "meta-GGA ROKS energies").
//!
//! What changed. The k-ROKS SCF had its own spin-resolved XC builder without a
//! tau term and refused meta-GGA. It now runs on the k-point UKS evaluator
//! `KPeriodicXcPolarized` (each spin's `τ_σ` from its own `D_σ`, each spin
//! Fock its own `V^τ_σ`), and the duplicate builder is deleted. No new
//! formula: the Roothaan combination of the spin Focks is unchanged, so this
//! is wiring, checked here against independent constructions.
//!
//! # Artifact hypothesis next to the physics hypothesis
//!
//! * Physics right: (1) a singlet k-ROKS equals the closed-shell k-RKS (a
//!   different evaluator: unpolarized kernel, total tau); (2) a FULLY
//!   polarized state (`N_beta = 0`, no closed shell) equals k-UKS, since the
//!   ROHF and UHF variational spaces coincide there; (3) a genuinely
//!   restricted-open-shell state (`N_alpha = 2`, `N_beta = 1`) matches PySCF
//!   `pbc.dft.KROKS` (whose veff is its KUKS one on the ROHF occupations).
//! * Artifact: a spin-channel tau error (`τ_β` from `D_α`) or an ROKS
//!   coupling error (`V^xc_α` / `V^xc_β` swapped between the spin Focks, the
//!   bare alpha Fock diagonalised) leaves (1) exactly green (`D_α = D_β`,
//!   `F_α = F_β`). (2) has no beta density and an empty Roothaan closed-open
//!   block: it is blind to the bare alpha Fock, sees the swapped `V^xc` (the
//!   alpha Fock then carries `V_β`), and under `τ_β` from `D_α` the SCF does
//!   not converge (a refusal, not a value). Both anchors are therefore
//!   necessary, not sufficient; every defect is required to fail the
//!   POLARIZED PySCF pin (3). Both anchors catch a tau dropped from the
//!   kernel input.
//!
//! # Measured (2026-10-27, test profile)
//!
//! * Singlet k-ROKS == k-RKS (H2, 1x1x3): |dE| <= 2.2e-16, |dE_xc| <= 1.1e-16.
//! * Fully polarized k-ROKS == k-UKS (H2 triplet, two s per H, 1x1x3): |dE|
//!   <= 8.8e-15, |dE_xc| <= 2.6e-13.
//! * PySCF 2.13.1 `KROKS` (H3 doublet): |dE| <= 5.4e-14 on all four pins
//!   (1x1x1, 1x1x3; SCAN, r2SCAN), `<S2>` = 3.75 at 1x1x3. The ROKS energy is
//!   1.4e-4 to 1.2e-3 above PySCF's KUKS on the same setup.
//!
//! # Mutation ledger (miss; SCAN / r2SCAN; from `validation_pbc_kroks_mgga.rs`)
//!
//! | defect | PySCF pins 1x1x1 / 1x1x3 | singlet anchor | fully polarized anchor |
//! |---|---|---|---|
//! | `TauBetaFromAlpha` | 4.3e-2 / 3.3e-2 (r2SCAN 4.5e-2 / 3.5e-2) | 2.2e-16 (blind) | SCF does not converge |
//! | `DropTau` | 3.1e-2 / 4.3e-2 (3.2e-2 / 4.5e-2) | 1.5e-2 / 1.6e-2 | 4.2e-2 / 4.3e-2 |
//! | `XcSpinSwap` | 1.6e-4 / 3.6e-3 (1.6e-4 / 3.6e-3) | 2.2e-16 (blind) | 1.8e-2 / 1.9e-2 |
//! | `BareAlphaFock` | 9.7e-4 / 1.4e-3 (9.9e-4 / 1.4e-3) | 0 (blind) | 0 (blind) |
//!
//! `V^xc_β := V^xc_α` was tried as the coupling defect first and dropped: with
//! no exact exchange it makes `F_β = F_α`, where the Roothaan `F_eff` collapses
//! to `F_α`, so it IS `BareAlphaFock` (identical misses to 7 digits).
//!
//! Not covered: the ROKS meta-GGA force and stress (refused, checked here on
//! a converged SCAN ROKS state), TPSS (refused, `pbc_kroks.rs`), RS-GDF J/K
//! for ROKS meta-GGA (refused by name: no ROKS RS-GDF meta-GGA pin exists)
//! and the Python / CLI bindings (refused, #498). The 1x1x1 == Gamma ROKS
//! anchor lives with the Gamma tests
//! (`pbc_gamma_open_mgga.rs::gamma_roks_doublet_is_one_point_kroks`).

mod common;
#[path = "kroks_mgga_support/mod.rs"]
mod support;

use common::*;
use ferric_pbc::dense_aft::ExxDiv;
use ferric_pbc::hcore::kpoint::periodic_hcore_kpts;
use ferric_pbc::hcore::PeriodicHcoreConfig;
use ferric_pbc::kdense_aft::KDenseAftEri;
use ferric_pbc::kgrad::{KGradConfig, KGradJk};
use ferric_pbc::kpts::KPointMesh;
use ferric_pbc::kroks::{solve_kroks, solve_kroks_on_grid, KRohfMutation};
use ferric_pbc::kscf::KJkKind;
use ferric_pbc::kstress::{KStressConfig, KStressJk};
use ferric_pbc::kuks::KUksMggaMutation;
use ferric_pbc::kuxcgrad::{kpoint_roks_gradient, KUksGradConfig};
use ferric_pbc::kuxcstress::{kpoint_roks_stress, KUksStressConfig};
use ferric_pbc::kxcgrad::{KRksGradConfig, KXcGrid};
use ferric_pbc::kxcstress::KRksStressConfig;
use support::*;

/// EXACTNESS ANCHOR 1 (written and passed before any PySCF number): a singlet
/// k-ROKS (H2, one s per H, 1x1x3, complex Bloch AOs) equals k-RKS, energy
/// and `E_xc`. Blind to every spin-channel and coupling defect (module doc);
/// it must catch a tau dropped from the kernel input.
#[test]
fn singlet_k_roks_is_k_rks() {
    let cell = cube_cell(&H2, 1);
    let bs = s05();
    let grid = uniform_grid(&cell);
    let n = [1, 1, 3];
    for f in FUNCTIONALS {
        let ro = kroks(&cell, &bs, n, f, &grid);
        let r = krks(&cell, &bs, n, f, &grid);
        let de = ro.scf.energy - r.scf.energy;
        eprintln!(
            "H2 {f} {n:?}: E_roks {:.12} E_rks {:.12} dE {de:.2e} dE_xc {:.2e}; it {} / {}",
            ro.scf.energy,
            r.scf.energy,
            ro.e_xc - r.e_xc,
            ro.scf.iterations,
            r.scf.iterations
        );
        assert!(de.abs() < 1e-10, "{f}: {de:e}");
        assert!((ro.e_xc - r.e_xc).abs() < 1e-10, "{f} E_xc");
        assert!(ro.scf.s2.abs() < 1e-8, "{f} <S2> {}", ro.scf.s2);
        let bad = try_kroks(
            &cell,
            &bs,
            n,
            f,
            &grid,
            Some(Defect::Tau(KUksMggaMutation::DropTau)),
        )
        .map_or(f64::INFINITY, |b| (b.scf.energy - r.scf.energy).abs());
        eprintln!("H2 {f} DropTau: |dE| {bad:.2e}");
        assert!(bad > 1e-6, "{f}: DropTau not seen by the singlet ({bad:e})");
    }
}

/// EXACTNESS ANCHOR 2: a fully polarized state (H2 triplet, two diffuse s per
/// H so virtuals exist, 1x1x3; `N_beta = 0`) has no closed shell, so its ROKS and
/// UKS minima coincide: k-ROKS == k-UKS. Unlike anchor 1 the XC kernel is
/// genuinely polarized (`ρ_β = τ_β = 0`).
#[test]
fn fully_polarized_k_roks_is_k_uks() {
    let cell = cube_cell(&H2, 3);
    let bs = two_s();
    let grid = uniform_grid(&cell);
    let n = [1, 1, 3];
    for f in FUNCTIONALS {
        let ro = kroks(&cell, &bs, n, f, &grid);
        let u = kuks(&cell, &bs, n, f, &grid);
        let de = ro.scf.energy - u.scf.energy;
        eprintln!(
            "H2 triplet {f} {n:?}: E_roks {:.12} E_uks {:.12} dE {de:.2e} dE_xc {:.2e}; \
             <S2> {:.6} / {:.6}; it {} / {}",
            ro.scf.energy,
            u.scf.energy,
            ro.e_xc - u.e_xc,
            ro.scf.s2,
            u.scf.s2,
            ro.scf.iterations,
            u.scf.iterations
        );
        assert!(de.abs() < 1e-9, "{f}: {de:e}");
        assert!((ro.e_xc - u.e_xc).abs() < 1e-8, "{f} E_xc");
        // The coupling defect this anchor can see (the alpha Fock carries
        // V_beta); SCAN only per commit (r2SCAN: the weekly ledger).
        if f != "SCAN" {
            continue;
        }
        let bad = try_kroks(
            &cell,
            &bs,
            n,
            f,
            &grid,
            Some(Defect::Ro(KRohfMutation::XcSpinSwap)),
        )
        .map_or(f64::INFINITY, |b| (b.scf.energy - u.scf.energy).abs());
        eprintln!("H2 triplet {f} XcSpinSwap: |dE| {bad:.2e}");
        assert!(bad > 1e-6, "{f}: XcSpinSwap not seen ({bad:e})");
    }
}

/// The polarized oracle: PySCF 2.13.1 `pbc.dft.KROKS` on the H3 doublet (one s
/// per H, FFTDF 40^3, UniformGrids 32^3; `scripts/gen_pyscf_kroks_mgga_refs.py`)
/// at 1x1x3 for both functionals, against ferric's dense-AFT J on the same
/// uniform grid. Every defect of [`DEFECTS`] must miss the SCAN pin.
#[test]
fn h3_kroks_matches_pinned_pyscf() {
    let cell = cube_cell(&H3, 2);
    let bs = s05();
    let grid = uniform_grid(&cell);
    let n = [1, 1, 3];
    for f in FUNCTIONALS {
        let pin = pyscf_pin("e_tot", &key(f, n));
        let r = kroks(&cell, &bs, n, f, &grid);
        let de = r.scf.energy - pin;
        eprintln!(
            "H3 {f} {n:?}: E {:.12} pin {pin:.12} dE {de:.2e}; <S2> {:.6}; it {}; vs KUKS pin {:.3e}",
            r.scf.energy,
            r.scf.s2,
            r.scf.iterations,
            r.scf.energy - pyscf_pin("e_tot_kuks", &key(f, n))
        );
        assert!(de.abs() < 1e-9, "{f}: {de:e}");
        // A pure spin state: S_z = (N_a - N_b) N_k / 2 = 1.5.
        assert!((r.scf.s2 - 3.75).abs() < 1e-8, "{f} <S2> {}", r.scf.s2);
    }
    let pin = pyscf_pin("e_tot", &key("SCAN", n));
    for d in DEFECTS {
        let miss = defect_miss("SCAN", n, d, pin);
        eprintln!("H3 SCAN {n:?} {d:?}: |dE| vs pin {miss:.2e}");
        assert!(miss > 1e-6, "{d:?} not detected: {miss:e}");
    }
}

/// The library ROKS meta-GGA is ENERGY only: a converged SCAN ROKS state is
/// refused by the ROKS force and stress (item 3 of #492 is later).
#[test]
fn kroks_meta_gga_force_and_stress_stay_refused() {
    let cell = cube_cell(&H3, 2);
    let bs = s05();
    let grid = uniform_grid(&cell);
    let n = [1, 1, 1];
    let r = kroks(&cell, &bs, n, "SCAN", &grid);
    let prep = prep_for(&cell, &bs);
    let mesh = KPointMesh::gamma_centred(&cell, n).unwrap();
    let hcfg = PeriodicHcoreConfig::with_omega(OMEGA_HCORE);
    let hk = periodic_hcore_kpts(&cell, &prep, &mesh, &hcfg).unwrap();
    let eri = KDenseAftEri::build(
        &cell,
        &prep,
        &mesh,
        &hk.s,
        ExxDiv::Ewald,
        &Default::default(),
    )
    .unwrap();
    let ugrid = KXcGrid::Uniform([PIN_GRID; 3]);
    let gcfg = KUksGradConfig {
        base: KRksGradConfig {
            grad: KGradConfig::default(),
            grid: ugrid.clone(),
            ..Default::default()
        },
        spin_mutation: None,
    };
    let scfg = KUksStressConfig {
        base: KRksStressConfig {
            stress: KStressConfig::default(),
            grid: ugrid,
            ..Default::default()
        },
        spin_mutation: None,
    };
    for f in FUNCTIONALS {
        let e = kpoint_roks_gradient(
            &cell,
            &prep,
            &mesh,
            &hcfg,
            &hk,
            KGradJk::Dense(&eri),
            &r.scf,
            f,
            &gcfg,
        )
        .expect_err("ROKS meta-GGA force")
        .to_string();
        assert!(
            e.contains("k-point ROKS meta-GGA force is not implemented"),
            "{f}: {e}"
        );
        let e = kpoint_roks_stress(
            &cell,
            &prep,
            &mesh,
            &hcfg,
            &hk,
            KStressJk::Dense(&eri),
            &r.scf,
            f,
            &scfg,
        )
        .expect_err("ROKS meta-GGA stress")
        .to_string();
        assert!(
            e.contains("k-point ROKS meta-GGA stress is not implemented"),
            "{f}: {e}"
        );
    }
}

/// The k-ROKS meta-GGA is validated on the dense-AFT J/K only: on the RS-GDF
/// J/K it is refused by name on both entry points, before any integral (the
/// k-UKS one IS validated on RS-GDF, `pbc_kuks_mgga_rsgdf.rs`, and its gate
/// is not touched; PBE ROKS on RS-GDF is pinned in `pbc_kroks.rs`).
#[test]
fn kroks_meta_gga_on_rsgdf_is_refused() {
    let cell = cube_cell(&H3, 2);
    let bs = s05();
    let prep = prep_for(&cell, &bs);
    let aux = prep_for(&cell, &single_s_h(1.0));
    let mesh = KPointMesh::gamma_centred(&cell, [1, 1, 1]).unwrap();
    let grid = uniform_grid(&cell);
    for f in FUNCTIONALS {
        let mut cfg = kroks_cfg(&cell, f);
        cfg.rohf.jk = KJkKind::RsGdf;
        for e in [
            solve_kroks(&cell, &prep, Some(&aux), &mesh, &cfg).map(|_| ()),
            solve_kroks_on_grid(&cell, &prep, Some(&aux), &mesh, &grid, &cfg).map(|_| ()),
        ] {
            let e = e.expect_err("ROKS meta-GGA on RS-GDF").to_string();
            assert!(
                e.contains("meta-GGA on RS-GDF")
                    && e.contains("validated on the dense-AFT J/K only"),
                "{f}: {e}"
            );
        }
    }
    // The k-UKS gate is unchanged (SCAN on RS-GDF is #579's validated route).
    ferric_pbc::kuks::resolve_kuks_functional("SCAN").unwrap();
}
