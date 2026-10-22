//! Analytic STRESS of the closed-shell k-point RKS energy for RANGE-SEPARATED
//! hybrids on the dense-AFT J/K (`ferric_pbc::kxcstress`, module doc
//! "Range-separated hybrids"), and the Gamma-point stress routed through the
//! 1x1x1 mesh (`ferric_pbc::gamma_rsh::gamma_rks_rsh_stress`).
//!
//! The stress is the global-hybrid one at `a = 1` with the exchange walks on
//! `v(K) = 4pi f(K^2)/K^2` and its slope `dv/dK^2 = -v/K^2 + (4pi/K^2) f'(K^2)`
//! (J bare), and the Madelung strain of the mixed constant
//! `c_sr dv_M/de + (c_lr - c_sr) dv_M^LR/de` of the supercell. PySCF has no
//! hybrid stress, so every oracle is internal; each test states its artifact
//! hypothesis:
//!
//! 0. `madelung_lr_strain` / `RshParams::madelung_strain` against central FD of
//!    the constants on a triclinic lattice (prototyped in Python first);
//! 1. central FD of ferric's own CAM-B3LYP energy (fresh SCF per strained
//!    cell), three components here, all nine for both functionals and both
//!    exxdiv plus the h sweep in `validation_krsh_stress.rs`;
//! 2. `c_sr == c_lr == c` reproduces the global hybrid with fraction `c` (zero
//!    XC, two SCFs);
//! 3. the 1x1x3 mesh stress equals the explicit 3-cell supercell stress
//!    (sigma equal, dE/de / 3);
//! 4. Gamma (`gamma_rks_rsh` + `gamma_rks_rsh_stress`) equals the 1x1x1 mesh;
//! 5. ewald - none: the SCF density does not depend on the exxdiv, so
//!    `E_ewald - E_none = -n_occ v_M(eps)` at every strain and the stress
//!    difference is `-n_occ dv_M/de`, checked against FD of the mixed constant
//!    on strained supercells (independent of the analytic Madelung strain AND
//!    of the Pulay / M-term bookkeeping);
//! 6. the mutation ledger below; 7. typed refusals.
//!
//! MUTATION LEDGER (production-wired `KStressMutation::Rsh*`): see the table
//! in `rsh_stress_mutants_are_caught`.
//!
//! NOT covered: RS-GDF (`validation_krsh_rsgdf_force.rs`), open shell (refused), p/d shells (H2/STO-3G
//! only), shifted meshes, Python / CLI (they still refuse range-separated
//! derivatives), wB97X (accepted, not exercised).

mod common;
#[path = "krsh_stress_support/mod.rs"]
mod stress_support;
#[path = "krsh_force_support/mod.rs"]
mod support;

use common::*;
use ferric_pbc::dense_aft::{DenseAftEri, ExxDiv, DEFAULT_DENSE_AFT_MAX_BYTES};
use ferric_pbc::dft::{gamma_rks_rsh, GammaRksConfig};
use ferric_pbc::gamma_rsh::{gamma_rks_rsh_stress, GammaRshExchange};
use ferric_pbc::hcore::{periodic_hcore, PeriodicHcoreConfig};
use ferric_pbc::kdense_aft::KDenseAftConfig;
use ferric_pbc::kpts::KPointMesh;
use ferric_pbc::kstress::{kpoint_rhf_stress, KStressConfig, KStressJk, KStressMutation};
use ferric_pbc::kxcstress::{kpoint_rks_stress, KRksStress};
use ferric_pbc::lattice::Cell;
use ferric_pbc::rsh::{madelung_lr, madelung_lr_strain, RshParams};
use ferric_pbc::stress::Mat3;
use ferric_pbc::uhf::GammaUhfIntegrals;
use stress_support::*;
use support::*;

// ------------------------------------------------- (0) Madelung strain

fn triclinic() -> Cell {
    let lat = [[4.0, 0.3, -0.2], [0.1, 4.5, 0.25], [-0.15, 0.2, 5.0]];
    Cell::new(hydrogens(&[[0.3, 0.2, 0.1]]), lat).unwrap()
}

fn fd_mat(f: impl Fn(&Cell) -> f64, cell: &Cell, h: f64) -> Mat3 {
    let mut o = [[0.0; 3]; 3];
    for (i, j) in ALL9 {
        let p = f(&cell.strained(&eps_at(i, j, h)).unwrap());
        let m = f(&cell.strained(&eps_at(i, j, -h)).unwrap());
        o[i][j] = (p - m) / (2.0 * h);
    }
    o
}

/// (0) The strain of `v_M^LR(w)` and of the mixed constant against central FD
/// of the constants themselves, all nine components on a triclinic lattice.
/// Python prototype (scratch, same lattice, w = 0.6): 2.4e-7 / 2.4e-9 /
/// 3.4e-11 at h = 1e-3 / 1e-4 / 1e-5 (h^2). Artifact: a dropped volume term
/// or G-shape term misses at O(1e-1) (|dv/de| ~ 0.18), and the h sweep would
/// stall instead of falling 100x per decade. MEASURED: see the printout
/// (bar 1e-7 at h = 1e-4, ratio > 50 per decade).
#[test]
fn madelung_strains_match_fd_of_the_constants() {
    let cell = triclinic();
    let w = 0.6;
    let ana = madelung_lr_strain(&cell, w).unwrap();
    let mut prev = f64::INFINITY;
    for h in [1e-3, 1e-4] {
        let fd = fd_mat(|c| madelung_lr(c, w).unwrap(), &cell, h);
        let d = mdiff(&ana, &fd);
        eprintln!(
            "v_M^LR strain h={h:.0e}: |ana - FD| {d:.2e} (|ana| {:.3e})",
            mmax(&ana)
        );
        assert!(prev / d > 50.0, "not h^2: {prev:e} -> {d:e}");
        prev = d;
    }
    assert!(prev < 1e-7 && mmax(&ana) > 1e-2, "{prev:e}");
    for f in ["HSE06", CAM] {
        let r = rsh_of(f);
        let ana = r.madelung_strain(&cell).unwrap();
        let fd = fd_mat(|c| r.madelung(c).unwrap(), &cell, 1e-4);
        let d = mdiff(&ana, &fd);
        eprintln!(
            "{f}: mixed Madelung strain |ana - FD| {d:.2e} (|ana| {:.3e})",
            mmax(&ana)
        );
        assert!(d < 1e-7, "{f}: {d:e}");
        // The plain-Coulomb strain (the RshPlainMadelungStrain defect) is not it.
        let plain = ferric_pbc::ewald::madelung_strain(&cell).unwrap();
        assert!(mdiff(&plain, &fd) > 1e-3, "{f}");
    }
}

// --------------------------------------------------------------- (1) FD

/// (1, per-commit row) Central FD of the CAM-B3LYP energy (fresh SCF per
/// strained cell at fixed fractional coordinates, shared explicit grid
/// cutoff), dense AFT, 1x1x3 (non-TRIM), exxdiv ewald: a diagonal, a shear and
/// the mesh axis. Physics: the FD floor (all-nine sweep in the validation
/// file); artifact: every ledger mutant misses by orders of magnitude more.
/// MEASURED: 7.2e-10 / 1.7e-10 / 7.0e-10 (bar 3e-8).
#[test]
fn cam_b3lyp_stress_matches_fd() {
    let fx = fixture(&H2, [1, 1, 3], CAM, ExxDiv::Ewald);
    let s = stress(&fx);
    for c in [(2, 2), (0, 2), (1, 1)] {
        let d = fd_strain(&fx.cell, c, FD_H, [1, 1, 3], CAM, ExxDiv::Ewald);
        let miss = (s.de_deps[c.0][c.1] - d).abs();
        eprintln!(
            "CAM-B3LYP eps{c:?}: analytic {:+.10e} FD {d:+.10e} miss {miss:.2e}",
            s.de_deps[c.0][c.1]
        );
        assert!(miss < STRESS_FD_BAR, "{c:?}: {miss:e}");
    }
    assert!(mmax(&s.hf.parts.exchange) > 1e-4, "K walks exercised");
    assert!(
        mmax(&s.hf.parts.madelung) > 1e-4,
        "Madelung strain exercised"
    );
}

// ----------------------------------------------------- (2) equal fractions

/// (2) EQUAL FRACTIONS: `c_sr = c_lr = c` makes the kernel `c 4pi/K^2` (slope
/// `-v/K^2`, `f' = 0`) and the mixed Madelung `c v_M` (its strain `c dv_M/de`):
/// the global hybrid with fraction `c`. Two SCFs (RSH kernels at scale 1 vs
/// bare kernels at scale c, zero XC) and two stress paths (kernel factor vs
/// exchange-coefficient scaling). Artifact: a factor on J, or the plain
/// Madelung strain, moves it by >= 1e-3 (asserted; blind for the slope,
/// fraction swap and v_M^LR defects, which vanish at `c_sr = c_lr`, and for
/// the Madelung defects under exxdiv none).
#[test]
fn rsh_with_equal_fractions_is_the_global_hybrid_stress() {
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
        let gr = null_stress(&fr, &sr, &er, 1.0, Some(eq), None);
        let gg = null_stress(&fg, &sg, &eg, c, None, None);
        let d = mdiff(&gr.de_deps, &gg.de_deps);
        let dk = mdiff(&gr.hf.parts.exchange, &gg.hf.parts.exchange);
        let dm = mdiff(&gr.hf.parts.madelung, &gg.hf.parts.madelung);
        eprintln!(
            "{exxdiv:?}: equal fractions vs global c={c}: |dE| {de:.1e} stress {d:.2e} \
             K part {dk:.2e} Madelung part {dm:.2e}"
        );
        assert!(de < 1e-11, "{de:e}");
        assert!(d < 1e-9, "{exxdiv:?}: {d:e}");
        assert!(mmax(&gg.hf.parts.exchange) > 1e-4);
        for m in [
            KStressMutation::RshKernelOnJ,
            KStressMutation::RshPlainMadelungStrain,
        ] {
            let miss = mdiff(
                &null_stress(&fr, &sr, &er, 1.0, Some(eq), Some(m)).de_deps,
                &gg.de_deps,
            );
            eprintln!("  {exxdiv:?} equal-fraction anchor vs {m:?}: miss {miss:.2e}");
            if m == KStressMutation::RshPlainMadelungStrain && exxdiv == ExxDiv::None {
                assert!(miss < 1e-9, "blind by construction under exxdiv none");
            } else {
                assert!(miss > 1e-3, "{m:?} {exxdiv:?}: {miss:e}");
            }
        }
    }
}

// --------------------------------------------------------- (3) supercell

/// (3) SUPERCELL: the 1x1x3 mesh stress equals the stress of the explicit
/// 3-cell supercell on its 1-point mesh (sigma equal, dE/de / 3), whose
/// mixed Madelung constant and strain are the primitive ones of the
/// supercell lattice. Physics: the SCF floor; artifact: a q-dependent kernel
/// slip or a mesh-vs-primitive Madelung strain mix-up shows as O(1e-3).
#[test]
fn rsh_k_mesh_stress_is_the_supercell_stress() {
    {
        let f = CAM;
        let fx = fixture(&H2, [1, 1, 3], f, ExxDiv::Ewald);
        let k = stress(&fx);
        let sc = stress(&fixture_cell(
            supercell(&H2, [1, 1, 3]),
            [1, 1, 1],
            f,
            ExxDiv::Ewald,
        ));
        let ds = mdiff(&k.sigma, &sc.sigma);
        let dk = mdiff(
            &k.hf.parts.exchange,
            &scale(&sc.hf.parts.exchange, 1.0 / 3.0),
        );
        let dm = mdiff(
            &k.hf.parts.madelung,
            &scale(&sc.hf.parts.madelung, 1.0 / 3.0),
        );
        eprintln!(
            "{f}: 1x1x3 vs 3-cell supercell sigma {ds:.2e} (exchange part {dk:.2e}, \
             Madelung part {dm:.2e}; |sigma| {:.3e})",
            mmax(&k.sigma)
        );
        assert!(
            ds < 1e-10 && dk < 1e-9 && dm < 1e-9,
            "{f}: {ds:e} {dk:e} {dm:e}"
        );
        assert!(mmax(&k.hf.parts.madelung) > 1e-4);
        // Seen failing: the primitive-cell (not supercell) Madelung strain.
        let bad = stress_m(&fx, Some(KStressMutation::GammaMadelungStrain)).unwrap();
        let miss = mdiff(&bad.sigma, &sc.sigma);
        eprintln!("{f}: GammaMadelungStrain mutant vs supercell sigma {miss:.2e}");
        assert!(miss > 1e-6, "{miss:e}");
    }
}

fn scale(a: &Mat3, f: f64) -> Mat3 {
    let mut o = *a;
    o.iter_mut().flatten().for_each(|x| *x *= f);
    o
}

// ------------------------------------------------------------- (4) Gamma

fn gamma_stress(f: &str, exxdiv: ExxDiv) -> KRksStress {
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
    let kc = KStressConfig {
        budget_bytes: Some(AMPLE),
        ..Default::default()
    };
    let mut open = r.scf.clone();
    open.spin = ferric_scf::result::Spin::Unrestricted;
    let e = gamma_rks_rsh_stress(&cell, &prep, &hcfg, &open, &gc, &ex, &kc)
        .expect_err("open-shell result accepted")
        .to_string();
    assert!(e.contains("closed-shell (restricted) results only"), "{e}");
    gamma_rks_rsh_stress(&cell, &prep, &hcfg, &r.scf, &gc, &ex, &kc).unwrap()
}

/// (4) GAMMA == 1x1x1: the Gamma RSH stress (Gamma SCF, Gamma J and hcore,
/// routed into the 1x1x1 stress) against the 1x1x1 k SCF + stress. A GLUE
/// check (the kernel is shared): the SCF floor; a glue slip (density factor,
/// Fock, Madelung) is O(1e-3), a grid mismatch fails the Fock guard.
#[test]
fn gamma_rsh_stress_is_the_one_point_mesh_stress() {
    for (f, exxdiv) in [(CAM, ExxDiv::Ewald), ("HSE06", ExxDiv::None)] {
        let g = gamma_stress(f, exxdiv);
        let k = stress(&fixture(&H2, [1, 1, 1], f, exxdiv));
        let d = mdiff(&g.de_deps, &k.de_deps);
        eprintln!(
            "{f} {exxdiv:?}: Gamma vs 1x1x1 stress {d:.2e} (Fock guard {:.1e}, max|dE/de| {:.3e})",
            g.hf.fock_mismatch,
            mmax(&g.de_deps)
        );
        assert!(d < 1e-9, "{f} {exxdiv:?}: {d:e}");
        assert!(mmax(&g.de_deps) > 1e-2);
    }
}

// --------------------------------------------------------- (5) ewald - none

/// (5) EXXDIV CONSISTENCY: the Madelung shift moves only the occupied
/// levels, so both SCFs reach the same density and `E_ewald - E_none = -n_occ
/// v_M` (n_occ = 1 per cell here) at EVERY strain; hence `dE/de(ewald) -
/// dE/de(none) = -dv_M/de` of the mixed supercell constant, here from central
/// FD of `RshParams::madelung` on strained supercells (independent of the
/// analytic Madelung strain and of the M-term / Pulay cancellation).
/// Artifact: the plain or LR-less Madelung strain misses by the full
/// difference of the constants' strains (asserted > 1e-3).
#[test]
fn ewald_minus_none_stress_is_the_mixed_madelung_strain() {
    let f = CAM;
    let r = rsh_of(f);
    let fe = fixture(&H2, [1, 1, 3], f, ExxDiv::Ewald);
    let fn_ = fixture(&H2, [1, 1, 3], f, ExxDiv::None);
    let sc = |c: &Cell| {
        let m = KPointMesh::gamma_centred(c, [1, 1, 3]).unwrap();
        Cell::new(c.mol().clone(), m.supercell_lattice()).unwrap()
    };
    let vm = r.madelung(&sc(&fe.cell)).unwrap();
    let de = fe.res.scf.energy - fn_.res.scf.energy;
    eprintln!("E_ewald - E_none {de:+.12e}, -v_M {:+.12e}", -vm);
    assert!((de + vm).abs() < 1e-9, "{:e}", de + vm);
    let dvm = fd_mat(|c| r.madelung(&sc(c)).unwrap(), &fe.cell, 1e-4);
    let want = scale(&dvm, -1.0);
    let check = |m: Option<KStressMutation>| {
        let s_e = stress_m(&fe, m).unwrap();
        let s_n = stress_m(&fn_, m).unwrap();
        let mut got = s_e.de_deps;
        for i in 0..3 {
            for j in 0..3 {
                got[i][j] -= s_n.de_deps[i][j];
            }
        }
        mdiff(&got, &want)
    };
    let d = check(None);
    eprintln!(
        "ewald - none stress vs -dv_M/de (FD): {d:.2e} (|dv_M/de| {:.3e})",
        mmax(&dvm)
    );
    assert!(d < 1e-8, "{d:e}");
    for m in [
        KStressMutation::RshPlainMadelungStrain,
        KStressMutation::RshNoLrMadelungStrain,
    ] {
        let miss = check(Some(m));
        eprintln!("  {m:?}: miss {miss:.2e}");
        assert!(miss > 1e-3, "{m:?}: {miss:e}");
    }
}

// ----------------------------------------------------------- (6) mutants

/// (6) MUTATION LEDGER: every `Rsh*` stress defect against the production
/// stress (1x1x3, ewald), which is FD-validated (CAM-B3LYP here, both
/// functionals x all nine x both exxdiv in `validation_krsh_stress.rs`), so a
/// miss against it is a miss against FD to within the FD floor (7.2e-10 at
/// h = 2.5e-5). Measured 2026-10-19 (max |dE/de_mut - dE/de| per cell,
/// Hartree; STRESS_FD_BAR = 3e-8):
///
/// | mutant                 | CAM-B3LYP | HSE06   | other anchors that catch it            |
/// |------------------------|-----------|---------|----------------------------------------|
/// | RshNoKernelSlope       | 1.17e-1   | 3.45e-3 | none (equal fractions blind: f' = 0)   |
/// | RshKernelOnJ           | 5.98e-2   | 5.77e-2 | equal fractions 4.86e-2 (both exxdiv)  |
/// | RshSwapFractions       | 1.03e-1   | 1.52e-1 | none (equal fractions blind)           |
/// | RshPlainMadelungStrain | 3.44e-1   | 5.94e-1 | ewald-none 3.44e-1; equal fr. 4.71e-1  |
/// | RshNoLrMadelungStrain  | 2.92e-1   | 5.10e-3 | ewald-none 2.92e-1                     |
///
/// Even HSE06, whose erf weight on this K set is small, misses by >= 1.1e5
/// FD bars for the kernel-slope defect: the small-K (q != 0) terms carry
/// `f'`. The supercell anchor catches the primitive-cell Madelung strain
/// (`GammaMadelungStrain`, printed there).
#[test]
fn rsh_stress_mutants_are_caught() {
    for (f, bar) in [(CAM, 1e-5), ("HSE06", 10.0 * STRESS_FD_BAR)] {
        let fx = fixture(&H2, [1, 1, 3], f, ExxDiv::Ewald);
        let base = stress(&fx);
        for m in [
            KStressMutation::RshNoKernelSlope,
            KStressMutation::RshKernelOnJ,
            KStressMutation::RshSwapFractions,
            KStressMutation::RshPlainMadelungStrain,
            KStressMutation::RshNoLrMadelungStrain,
        ] {
            let miss = mdiff(&stress_m(&fx, Some(m)).unwrap().de_deps, &base.de_deps);
            eprintln!(
                "{f} 1x1x3 ewald: {m:?} miss {miss:.2e} ({:.0} FD bars)",
                miss / STRESS_FD_BAR
            );
            assert!(miss > bar, "{f} {m:?}: {miss:e}");
        }
    }
}

// ---------------------------------------------------------- (7) refusals

/// Typed refusals: bare kernels with a range-separated name, kernels with
/// another functional's parameters, the HF stress on range-separated kernels,
/// a range-separated name on RS-GDF J/K, and the open-shell allowlist.
#[test]
fn rsh_stress_refusals() {
    let fx = fixture(&H2, [1, 1, 1], "HSE06", ExxDiv::Ewald);
    let hcfg = PeriodicHcoreConfig::with_omega(OMEGA);
    let call = |jk: KStressJk<'_>, f: &str| {
        kpoint_rks_stress(
            &fx.cell,
            &fx.prep,
            &fx.mesh,
            &hcfg,
            &fx.hk,
            jk,
            &fx.res.scf,
            f,
            &scfg(ExxDiv::Ewald, None),
        )
        .map(|_| ())
        .map_err(|e| e.to_string())
    };
    let bare = eri_of(&fx, None);
    let e = call(KStressJk::Dense(&bare), "HSE06").unwrap_err();
    assert!(e.contains("bare Coulomb exchange"), "{e}");
    let other = eri_of(&fx, Some(rsh_of(CAM)));
    let e = call(KStressJk::Dense(&other), "HSE06").unwrap_err();
    assert!(e.contains("functional's is"), "{e}");
    let own = eri_of(&fx, Some(rsh_of("HSE06")));
    assert!(call(KStressJk::Dense(&own), "HSE06").is_ok());
    // The HF stress does not differentiate an attenuated K.
    let e = kpoint_rhf_stress(
        &fx.cell,
        &fx.prep,
        &fx.mesh,
        &hcfg,
        &fx.hk,
        KStressJk::Dense(&own),
        &fx.res.scf,
        ExxDiv::Ewald,
        &KStressConfig::default(),
    )
    .expect_err("HF stress on RSH kernels")
    .to_string();
    assert!(e.contains("does not differentiate it"), "{e}");
    // RS-GDF without a range separation, range-separated name: refused by the
    // source check.
    let aux = ferric_integrals::basis_bridge::PreparedBasis::new(
        fx.cell.mol(),
        &ferric_core::basis::bundled("cc-pvdz-ri").unwrap(),
    )
    .unwrap();
    let kc = ferric_pbc::rsgdf::kpoint::KRsGdfConfig {
        gdf: ferric_pbc::rsgdf::RsGdfConfig {
            omega: 1.0,
            exxdiv: ExxDiv::Ewald,
            budget_bytes: Some(AMPLE),
            ..Default::default()
        },
        mutation: None,
    };
    let gdf =
        ferric_pbc::rsgdf::kpoint::KRsGdf::build(&fx.cell, &fx.prep, &aux, &fx.mesh, &fx.hk.s, &kc)
            .unwrap()
            .with_exxdiv(ExxDiv::Ewald);
    let src = ferric_pbc::kgrad::KRsGdfGradSource {
        gdf: &gdf,
        cfg: &kc,
        aux: &aux,
    };
    let e = call(KStressJk::RsGdf(src), "HSE06").unwrap_err();
    assert!(
        e.contains("RS-GDF") && e.contains("bare Coulomb exchange"),
        "{e}"
    );
    // An RS-GDF source that carries the attenuated fit (build_rsh) is admitted
    // for the closed-shell RKS stress (validation_krsh_rsgdf_force.rs); this
    // SCF ran on the dense kernels, so the Fock guard refuses it.
    let gdf_rsh = ferric_pbc::rsgdf::kpoint::KRsGdf::build_rsh(
        &fx.cell,
        &fx.prep,
        &aux,
        &fx.mesh,
        &fx.hk.s,
        &kc,
        rsh_of("HSE06"),
    )
    .unwrap()
    .with_exxdiv(ExxDiv::Ewald);
    let src = ferric_pbc::kgrad::KRsGdfGradSource {
        gdf: &gdf_rsh,
        cfg: &kc,
        aux: &aux,
    };
    let e = call(KStressJk::RsGdf(src), "HSE06").unwrap_err();
    eprintln!("RS-GDF build_rsh on a dense SCF: {e}");
    assert!(
        e.contains("rebuilt from the density and the J/K source"),
        "{e}"
    );
    // Open shell keeps refusing range separation (the UKS / ROKS stress entry
    // with HSE06 is asserted word for word in pbc_kuxcstress.rs and
    // pbc_kuxcstress_ro.rs).
    for spin in [
        ferric_pbc::kxcgrad::DerivSpin::Unrestricted,
        ferric_pbc::kxcgrad::DerivSpin::RestrictedOpen,
    ] {
        assert!(ferric_pbc::kxcgrad::check_deriv_functional(spin, false, "HSE06").is_err());
    }
}
