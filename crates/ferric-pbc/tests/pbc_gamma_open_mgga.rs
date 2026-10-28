//! Gamma-point open-shell meta-GGA ENERGY (`ferric_pbc::dft::gamma_uks` and
//! `ferric_pbc::rohf::gamma_roks`, pure SCAN and r2SCAN; issue #492).
//!
//! The change under test is wiring only: `PeriodicXc::eval_polarized` now
//! builds `(τ_α, τ_β)` with ferric-dft's `eval_tau_uks` (each `τ_σ` from its
//! own `D_σ`) and hands them to the unchanged `semilocal_vxc_polarized`
//! kernel, which already carries the per-spin `V^τ_σ` (the molecular UKS
//! meta-GGA path). No new formula, so no Python prototype.
//!
//! # Artifact hypothesis next to the physics hypothesis
//!
//! * Physics right: (1) a polarized Gamma UKS equals the k-point UKS on a
//!   1x1x1 mesh (an independent construction: complex Bloch AOs and the
//!   k-point `chunk_tau`, PySCF-pinned in `pbc_kuks_mgga.rs`); (2) a singlet
//!   Gamma UKS / ROKS equals Gamma RKS; (3) polarized UKS and ROKS energies
//!   match PySCF `pbc.dft.UKS` / `pbc.dft.ROKS`; (4) `V_σ = ∂E_xc/∂D_σ`.
//! * Artifact: a spin-channel tau error (τ_β from D_α, τ omitted, swapped τ,
//!   `V^τ_β` dropped) is INVISIBLE in the singlet, where `τ_α = τ_β`. So the
//!   singlet anchors only say "no gross break"; every mutant must fail a
//!   POLARIZED check (the 1x1x1 anchor, the PySCF pin or the FD of `E_xc`).
//!
//! The force and stress of the Gamma open-shell meta-GGA are NOT implemented
//! (they refuse, `open_shell_mgga_derivatives_and_other_meta_ggas_refuse`).
//! The full PySCF table (UKS / ROKS x SCAN / r2SCAN) and every tau mutant
//! against those pins are in `validation_pbc_gamma_open_mgga.rs` (weekly).

mod common;
#[path = "gamma_open_mgga_support/mod.rs"]
mod support;

use common::*;
use ferric_core::FerricError;
use ferric_pbc::dense_aft::ExxDiv;
use ferric_pbc::dft::{
    gamma_rks, gamma_uks, GammaMggaMutation, GammaRksConfig, PeriodicGrid, PeriodicXc,
};
use ferric_pbc::hcore::PeriodicHcoreConfig;
use ferric_pbc::kpts::KPointMesh;
use ferric_pbc::kuks::{solve_kuks_on_grid, KUksConfig};
use ferric_pbc::rohf::gamma_roks;
use ferric_scf::result::ScfResult;
use ndarray::Array2;
use support::*;

// ------------------------------------------------------------ the anchors

/// EXACTNESS ANCHOR (run before any PySCF number): the polarized H3 doublet
/// Gamma UKS equals the k-point UKS on a 1x1x1 mesh on the same atom-centred
/// grid. Measured correct miss in the commit message / PR (bar 1e-10). Every
/// tau mutant (driven through `gamma_uks_with_xc` on the same grid) must miss
/// by > 1e-7, required here: the anchor is the polarized pin that catches
/// them at the energy level.
#[test]
fn gamma_uks_doublet_is_one_point_kuks() {
    let su = setup(h3_doublet());
    let gcfg = grid_cfg(40, 110);
    let grid = PeriodicGrid::build(&su.cell, &gcfg).unwrap();
    let mesh = KPointMesh::gamma_centred(&su.cell, [1, 1, 1]).unwrap();
    for f in FUNCTIONALS {
        let g = gamma_uks(
            &su.cell,
            &su.prep,
            &su.hc,
            ints(&su),
            &uks_cfg(f, gcfg.clone()),
        )
        .unwrap_or_else(|e| panic!("gamma_uks {f}: {e}"));
        let mut kcfg = KUksConfig::new(&su.cell, ExxDiv::Ewald, f);
        kcfg.kuhf.hcore = PeriodicHcoreConfig::with_omega(OMEGA);
        kcfg.kuhf.scf.energy_conv = 1e-12;
        kcfg.kuhf.scf.grad_conv = 1e-9;
        let k = solve_kuks_on_grid(&su.cell, &su.prep, None, &mesh, &grid, &kcfg).unwrap();
        let de = g.scf.energy - k.scf.energy;
        let dxc = g.e_xc - k.e_xc;
        eprintln!(
            "H3 {f}: E_gamma {:.12} E_k111 {:.12} dE {de:.2e} dE_xc {dxc:.2e} <S2> {:.6} it {}",
            g.scf.energy, k.scf.energy, g.s2, g.scf.iterations
        );
        assert!(de.abs() < 1e-10, "{f}: dE {de:.2e}");
        assert!(dxc.abs() < 1e-10, "{f}: dE_xc {dxc:.2e}");
        // Not a singlet in disguise: the spin channels differ.
        assert!(g.s2 > 0.7, "{f}: <S2> {}", g.s2);
        for m in MUTANTS {
            let miss = mutant_miss(
                uks_with(&su, &mut open_xc(&su, &grid, f, Some(m)), f),
                k.scf.energy,
            );
            eprintln!("H3 {f} {m:?}: |E - E_k111| {miss:.2e}");
            assert!(miss > 1e-7, "{f} {m:?} not detected: {miss:.2e}");
        }
    }
}

/// EXACTNESS ANCHOR (ROKS): the polarized H3 doublet Gamma ROKS equals the
/// k-point ROKS on a 1x1x1 mesh on the same atom-centred grid. Independent
/// constructions of the same energy: ferric-dft's `eval_tau_uks` on real
/// lattice-summed AOs (`PeriodicXc::eval_polarized`, Gamma ROHF driver) vs
/// `chunk_tau` on complex Bloch AOs (`KPeriodicXcPolarized`, k-ROKS driver,
/// PySCF-pinned in `pbc_kroks_mgga.rs`). Bar 1e-10. Every Gamma-side tau
/// mutant, and the k-side `τ_β` from `D_α` and swapped `V^xc_σ`, must miss
/// by > 1e-7 (SCAN; r2SCAN anchor only, to keep the per-commit cost down).
#[test]
fn gamma_roks_doublet_is_one_point_kroks() {
    use ferric_pbc::kroks::{solve_kroks_on_grid, KRohfMutation, KRoksConfig};
    use ferric_pbc::kuks::KUksMggaMutation;
    let su = setup(h3_doublet());
    let gcfg = grid_cfg(40, 110);
    let grid = PeriodicGrid::build(&su.cell, &gcfg).unwrap();
    let mesh = KPointMesh::gamma_centred(&su.cell, [1, 1, 1]).unwrap();
    let kroks = |f: &str, tau: Option<KUksMggaMutation>, ro: Option<KRohfMutation>| {
        let mut kcfg = KRoksConfig::new(&su.cell, ExxDiv::Ewald, f);
        kcfg.rohf.hcore = PeriodicHcoreConfig::with_omega(OMEGA);
        kcfg.rohf.scf.energy_conv = 1e-12;
        kcfg.rohf.scf.grad_conv = 1e-9;
        kcfg.mgga_mutation = tau;
        kcfg.rohf.mutation = ro;
        solve_kroks_on_grid(&su.cell, &su.prep, None, &mesh, &grid, &kcfg)
            .map(|r| (r.scf.energy, r.e_xc))
    };
    for f in FUNCTIONALS {
        let g = gamma_roks(
            &su.cell,
            &su.prep,
            &su.hc,
            ints(&su),
            &roks_cfg(f, gcfg.clone()),
        )
        .unwrap_or_else(|e| panic!("gamma_roks {f}: {e}"));
        let (e_k, xc_k) = kroks(f, None, None).unwrap_or_else(|e| panic!("k-ROKS {f}: {e}"));
        let de = g.scf.energy - e_k;
        let dxc = g.e_xc - xc_k;
        eprintln!(
            "H3 ROKS {f}: E_gamma {:.12} E_k111 {e_k:.12} dE {de:.2e} dE_xc {dxc:.2e} <S2> {:.6}",
            g.scf.energy, g.s2
        );
        assert!(de.abs() < 1e-10, "{f}: dE {de:.2e}");
        assert!(dxc.abs() < 1e-10, "{f}: dE_xc {dxc:.2e}");
        assert!(g.s2 > 0.7, "{f}: <S2> {}", g.s2);
        if f != "SCAN" {
            continue;
        }
        for m in MUTANTS {
            let miss = mutant_miss(roks_with(&su, &mut open_xc(&su, &grid, f, Some(m)), f), e_k);
            eprintln!("H3 ROKS {f} Gamma {m:?}: |E - E_k111| {miss:.2e}");
            assert!(miss > 1e-7, "{f} Gamma {m:?} not detected: {miss:.2e}");
        }
        for (name, r) in [
            (
                "k TauBetaFromAlpha",
                kroks(f, Some(KUksMggaMutation::TauBetaFromAlpha), None),
            ),
            (
                "k XcSpinSwap",
                kroks(f, None, Some(KRohfMutation::XcSpinSwap)),
            ),
        ] {
            let miss = mutant_miss(r, g.scf.energy);
            eprintln!("H3 ROKS {f} {name}: |E_k111 - E_gamma| {miss:.2e}");
            assert!(miss > 1e-7, "{f} {name} not detected: {miss:.2e}");
        }
    }
}

/// A singlet (`N_alpha == N_beta`) Gamma UKS and ROKS equal Gamma RKS, on the
/// H2 cube and the four-H triclinic cell. Blind to every spin-channel tau
/// mutant by construction (module doc): necessary, not sufficient.
#[test]
fn singlet_gamma_uks_and_roks_are_gamma_rks() {
    for (label, cell) in [("H2", h2_cell(4.0)), ("H4 tri", triclinic_cell())] {
        let su = setup(cell);
        let gcfg = grid_cfg(30, 110);
        for f in FUNCTIONALS {
            let mut rcfg = GammaRksConfig::new(f);
            rcfg.grid = gcfg.clone();
            rcfg.scf.energy_conv = 1e-12;
            let r = gamma_rks(&su.cell, &su.prep, &su.hc, ints(&su), &rcfg).unwrap();
            let u = gamma_uks(
                &su.cell,
                &su.prep,
                &su.hc,
                ints(&su),
                &uks_cfg(f, gcfg.clone()),
            )
            .unwrap();
            let o = gamma_roks(
                &su.cell,
                &su.prep,
                &su.hc,
                ints(&su),
                &roks_cfg(f, gcfg.clone()),
            )
            .unwrap();
            let (du, dro) = (u.scf.energy - r.scf.energy, o.scf.energy - r.scf.energy);
            eprintln!(
                "{label} {f}: E_rks {:.12} dE_uks {du:.2e} dE_roks {dro:.2e} dE_xc {:.2e} / {:.2e}",
                r.scf.energy,
                u.e_xc - r.e_xc,
                o.e_xc - r.e_xc
            );
            assert!(du.abs() < 1e-10 && dro.abs() < 1e-10, "{label} {f}");
            assert!((u.e_xc - r.e_xc).abs() < 1e-10, "{label} {f} UKS E_xc");
            assert!((o.e_xc - r.e_xc).abs() < 1e-10, "{label} {f} ROKS E_xc");
            assert!(u.s2.abs() < 1e-8, "{label} {f}: <S2> {}", u.s2);
        }
    }
}

// ---------------------------------------------------- polarized V = dE/dD

/// Spin densities on the exact rotation curve `c_i(h) = cos h c_i + sin h c_a`
/// of ONE spin's HOMO `i` into its LUMO `a` (S-orthonormal MOs, so `D(h)` stays
/// idempotent and of rank `N_σ`), and `dD/dh` at `h = 0`.
fn rotated_density(c: &Array2<f64>, nocc: usize, h: f64) -> Array2<f64> {
    let mut occ = c.slice(ndarray::s![.., ..nocc]).to_owned();
    let (i, a) = (nocc - 1, nocc);
    let col = &c.column(i) * h.cos() + &c.column(a) * h.sin();
    occ.column_mut(i).assign(&col);
    occ.dot(&occ.t())
}

/// Worst relative miss, over the two spins, between a central finite
/// difference of `E_xc` along the HOMO -> LUMO rotation of ONE spin (the other
/// spin fixed) and `Tr(V_σ dD_σ/dh) = 2 c_a^T V_σ c_i` of `xc` itself.
fn fd_vs_v_worst_rel(xc: &mut PeriodicXc, scf: &ScfResult, nocc: [usize; 2], tag: &str) -> f64 {
    let cb = scf.mos_beta.as_ref().expect("UKS has beta MOs");
    let mos = [&scf.mos_alpha, cb];
    let d = |s: usize, h: f64| rotated_density(mos[s], nocc[s], h);
    let (_, va, vb) = xc.eval_polarized(&d(0, 0.0), &d(1, 0.0)).unwrap();
    let h = 2e-5;
    let mut worst = 0.0_f64;
    for spin in 0..2 {
        let mut e_at = |t: f64| {
            let (da, db) = if spin == 0 {
                (d(0, t), d(1, 0.0))
            } else {
                (d(0, 0.0), d(1, t))
            };
            xc.eval_polarized(&da, &db).unwrap().0
        };
        let fd = (e_at(h) - e_at(-h)) / (2.0 * h);
        let (v, c) = if spin == 0 {
            (&va, mos[0])
        } else {
            (&vb, mos[1])
        };
        let (ci, ca) = (c.column(nocc[spin] - 1), c.column(nocc[spin]));
        let an = 2.0 * ca.dot(&v.dot(&ci));
        let rel = (fd - an).abs() / fd.abs().max(1e-12);
        eprintln!("{tag} spin {spin}: FD {fd:.10} analytic {an:.10} rel {rel:.2e}");
        worst = worst.max(rel);
    }
    worst
}

/// `V_σ` is the derivative of `E_xc` with respect to `D_σ` at the converged
/// polarized H3 state, spin by spin, along the HOMO -> LUMO rotation of each
/// spin. This checks each spin's `V^τ_σ` separately with no SCF in the loop:
/// an energy pin is weak for a Fock-only error (dropping `V^τ_β` is second
/// order in the energy), the FD is first order. `DropBetaVtau` and `SwapTau`
/// must fail it (> 1e-5). Step h = 2e-5: the correct miss is FD truncation,
/// scaling as h² (SCAN worst 1.3e-5 at h = 1e-3, 1.3e-7 at 1e-4, 4.1e-9 at
/// 2e-5); bar 1e-7.
///
/// Why a rotation and not an arbitrary symmetric `ΔD` (as the k-point test
/// uses): H3 has ONE β orbital, so at the Gamma point `τ_β = τ_W,β` exactly
/// (SCAN's iso-orbital limit, α = 0). A generic `ΔD_β` breaks rank 1 and
/// pushes `τ_β` below `τ_W` on one side of `h = 0`, where libxc clamps; the
/// energy has a kink there and a central FD averages two slopes. Measured with
/// the generic direction: α (two orbitals) 9.3e-11 relative, β 7.95e-2. The
/// rotation keeps `D_β` idempotent and rank 1, so `τ_β = τ_W` along the whole
/// curve. (The k-point test escapes this because two k-points give the β
/// channel two Bloch orbitals.)
#[test]
fn polarized_vxc_is_the_derivative_of_exc() {
    let su = setup(h3_doublet());
    let gcfg = grid_cfg(30, 110);
    let grid = PeriodicGrid::build(&su.cell, &gcfg).unwrap();
    for f in FUNCTIONALS {
        let r = gamma_uks(
            &su.cell,
            &su.prep,
            &su.hc,
            ints(&su),
            &uks_cfg(f, gcfg.clone()),
        )
        .unwrap();
        let nocc = [r.nocc.0, r.nocc.1];
        let ok = fd_vs_v_worst_rel(&mut open_xc(&su, &grid, f, None), &r.scf, nocc, f);
        assert!(ok < 1e-7, "{f}: worst rel {ok:.2e}");
        for m in [GammaMggaMutation::DropBetaVtau, GammaMggaMutation::SwapTau] {
            let bad = fd_vs_v_worst_rel(&mut open_xc(&su, &grid, f, Some(m)), &r.scf, nocc, f);
            eprintln!("{f} {m:?}: FD miss {bad:.2e} (correct {ok:.2e})");
            assert!(bad > 1e-5, "{f} {m:?} not detected: {bad:.2e}");
        }
    }
}

// ------------------------------------------------------------- PySCF pins

/// One polarized pin per route (PySCF 2.13.1 Gamma `pbc.dft.UKS` /
/// `pbc.dft.ROKS`, AFTDF + UniformGrids 61^3,
/// `scripts/gen_pyscf_gamma_open_mgga_refs.py`), on the same 61^3 grid; bar
/// 1e-9. The r2SCAN pins and the mutants-versus-pin table are in
/// `validation_pbc_gamma_open_mgga.rs` (weekly). The ROKS pin runs on every
/// commit next to its polarized anchor `gamma_roks_doublet_is_one_point_kroks`
/// (the 1x1x1 k-ROKS, itself PySCF-pinned in `pbc_kroks_mgga.rs`).
fn pin(route: &str) {
    let su = setup(h3_doublet());
    let grid = PeriodicGrid::uniform(&su.cell, [61; 3]).unwrap();
    let e_ref = pyscf_ref(&format!("{route} SCAN"));
    let (e, _) = run_route(route, &su, &mut open_xc(&su, &grid, "SCAN", None), "SCAN").unwrap();
    eprintln!(
        "H3 {route} SCAN: {e:.12} PySCF {e_ref:.12} dE {:.2e}",
        e - e_ref
    );
    assert!((e - e_ref).abs() < 1e-9, "{route}: {e} vs {e_ref}");
}

#[test]
fn h3_gamma_uks_scan_matches_pinned_pyscf() {
    pin("UKS");
}

#[test]
fn h3_gamma_roks_scan_matches_pinned_pyscf() {
    pin("ROKS");
}

// --------------------------------------------------------------- refusals

/// Outside the validated envelope everything refuses BY NAME: TPSS (and raw
/// meta-GGA names) on both open-shell entries, and the Gamma open-shell
/// meta-GGA force and stress (UKS and ROKS), which are not implemented.
#[test]
fn open_shell_mgga_derivatives_and_other_meta_ggas_refuse() {
    use ferric_pbc::grad::{gamma_roks_gradient, gamma_uks_gradient};
    use ferric_pbc::stress::{gamma_roks_stress, gamma_uks_stress, GammaStressConfig};
    let su = setup(h3_doublet());
    let gcfg = grid_cfg(20, 50);
    for f in ["TPSS", "MGGA_X_TPSS"] {
        let e = gamma_uks(
            &su.cell,
            &su.prep,
            &su.hc,
            ints(&su),
            &uks_cfg(f, gcfg.clone()),
        )
        .expect_err(f)
        .to_string();
        assert!(e.contains("only the pure SCAN / r2SCAN"), "UKS {f}: {e}");
        let e = gamma_roks(
            &su.cell,
            &su.prep,
            &su.hc,
            ints(&su),
            &roks_cfg(f, gcfg.clone()),
        )
        .expect_err(f)
        .to_string();
        assert!(e.contains("only the pure SCAN / r2SCAN"), "ROKS {f}: {e}");
    }
    let hcfg = PeriodicHcoreConfig::with_omega(OMEGA);
    let open_shell = |e: FerricError| {
        let e = e.to_string();
        assert!(
            e.contains("meta-GGA") && e.contains("open shell (UKS / ROKS)"),
            "{e}"
        );
    };
    let ucfg = uks_cfg("SCAN", gcfg.clone());
    let u = gamma_uks(&su.cell, &su.prep, &su.hc, ints(&su), &ucfg).unwrap();
    open_shell(
        gamma_uks_gradient(&su.cell, &su.prep, &hcfg, &su.hc, &su.eri, &u.scf, &ucfg).unwrap_err(),
    );
    let sc = GammaStressConfig::default();
    open_shell(
        gamma_uks_stress(
            &su.cell, &su.prep, &hcfg, &su.hc, &su.eri, &u.scf, &ucfg, &sc,
        )
        .unwrap_err(),
    );
    let ocfg = roks_cfg("SCAN", gcfg);
    let o = gamma_roks(&su.cell, &su.prep, &su.hc, ints(&su), &ocfg).unwrap();
    open_shell(
        gamma_roks_gradient(&su.cell, &su.prep, &hcfg, &su.hc, &su.eri, &o.scf, &ocfg).unwrap_err(),
    );
    open_shell(
        gamma_roks_stress(
            &su.cell, &su.prep, &hcfg, &su.hc, &su.eri, &o.scf, &ocfg, &sc,
        )
        .unwrap_err(),
    );
}

/// The open-shell Gamma meta-GGA routes that are NOT validated refuse BY
/// NAME before any SCF: RS-GDF J/K (UKS and ROKS), and the second-order ROKS
/// solver, whose XC Hessian is a finite difference of the polarized XC along
/// an arbitrary `ΔD` (the iso-orbital kink of `polarized_vxc_is_the_derivative_of_exc`).
/// The bindings' message names an unsupported meta-GGA (TPSS) as unsupported,
/// not as a validated SCAN / r2SCAN energy.
#[test]
fn unvalidated_open_shell_mgga_routes_refuse() {
    use ferric_pbc::rohf::RohfSolver;
    use ferric_pbc::rsgdf::{RsGdf, RsGdfConfig};
    use ferric_pbc::uhf::GammaUhfIntegrals;
    let su = setup(h3_doublet());
    let aux = ferric_integrals::basis_bridge::PreparedBasis::new(
        su.cell.mol(),
        &ferric_core::basis::bundled("cc-pvdz-ri").unwrap(),
    )
    .unwrap();
    let gcfg = RsGdfConfig {
        exxdiv: ExxDiv::None,
        ..Default::default()
    };
    let gdf = RsGdf::build(&su.cell, &su.prep, &aux, &su.hc.s, &gcfg).unwrap();
    let rs = GammaUhfIntegrals::RsGdf(&gdf);
    let grid = grid_cfg(20, 50);
    for f in FUNCTIONALS {
        let e = gamma_uks(&su.cell, &su.prep, &su.hc, rs, &uks_cfg(f, grid.clone()))
            .expect_err("UKS meta-GGA on RS-GDF")
            .to_string();
        assert!(
            e.contains("validated on the dense-AFT J/K only"),
            "UKS {f}: {e}"
        );
        let e = gamma_roks(&su.cell, &su.prep, &su.hc, rs, &roks_cfg(f, grid.clone()))
            .expect_err("ROKS meta-GGA on RS-GDF")
            .to_string();
        assert!(
            e.contains("validated on the dense-AFT J/K only"),
            "ROKS {f}: {e}"
        );
        let mut cfg = roks_cfg(f, grid.clone());
        cfg.solver = RohfSolver::second_order();
        let e = gamma_roks(&su.cell, &su.prep, &su.hc, ints(&su), &cfg)
            .expect_err("ROKS meta-GGA second order")
            .to_string();
        assert!(e.contains("DIIS solver only"), "ROKS {f}: {e}");
    }
    use ferric_pbc::dft::{check_gamma_roks_request, check_gamma_uks_request};
    for (route, e) in [
        ("UKS", check_gamma_uks_request("TPSS", &[])),
        ("ROKS", check_gamma_roks_request("TPSS", &[])),
    ] {
        let e = e.expect_err("TPSS").to_string();
        assert!(
            e.contains(&format!("TPSS: unsupported at the Gamma-point {route}"))
                && !e.contains("validated in the library"),
            "{route}: {e}"
        );
    }
}
