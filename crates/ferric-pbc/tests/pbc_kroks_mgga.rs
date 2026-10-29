//! k-point ROKS meta-GGA (`ferric_pbc::kroks::solve_kroks`, pure SCAN and
//! r2SCAN; issue #492): the energy (item "meta-GGA ROKS energies") and the
//! force and stress (`kpoint_roks_gradient` / `kpoint_roks_stress`, item 3).
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
//! # Force and stress (item 3)
//!
//! No new derivative code: the ROKS force and stress are the k-UKS assembly
//! on the ROHF spin densities, whose meta-GGA path already carries every tau
//! term; the ROKS Pulay weight `W_RO = sym(D_α F_α D_α + D_α F_β D_β)` is
//! built from the rebuilt spin Focks, which include `V^τ_σ`. Only the
//! refusal was lifted.
//!
//! * Physics right: (1) singlet k-ROKS == k-RKS per part; (2) fully
//!   polarized (`N_β = 0`) k-ROKS == k-UKS per part (`W_RO` is then the UKS
//!   `W`); (3) on the H3 doublet the force and `dE/dε` equal the central FD
//!   of ferric's own converged k-ROKS energy.
//! * Artifact: a tau term on the wrong spin, a dropped `V^τ_β` in the Pulay
//!   Fock or a dropped `W_RO` cross term. (1) is blind to a `v_τα`/`v_τβ`
//!   swap and (2) to the cross term and both β tau terms (`D_β = 0`), so the
//!   polarized FD (3) carries the ledger (`validation_pbc_kroks_mgga.rs`).
//!   A dropped `V^τ_β` cannot reach the force at all: the ROHF
//!   orbital-gradient gate is evaluated on the rebuilt Focks and refuses it.
//!
//! No external oracle exists: PySCF has no k-point ROKS gradient and its
//! `KUKS` force refuses meta-GGA. The independent constructions are the
//! anchors above and, weekly, the FD of the Gamma-point ROKS energy (real
//! AOs, `eval_tau_uks`) against the 1x1x1 k-ROKS force and stress.
//!
//! Not covered: TPSS (refused, `pbc_kroks.rs`; its force / stress refused by
//! name, `pbc_kuxc_mgga.rs`), RS-GDF J/K for ROKS meta-GGA (the SCF refuses
//! it by name: no ROKS RS-GDF meta-GGA pin exists), the Gamma-point ROKS
//! meta-GGA force and stress (refused) and the Python / CLI bindings
//! (refused, #498). The 1x1x1 == Gamma ROKS energy anchor lives with the
//! Gamma tests (`pbc_gamma_open_mgga.rs::gamma_roks_doublet_is_one_point_kroks`).

mod common;
#[path = "kroks_mgga_support/mod.rs"]
mod support;

use common::*;
use ferric_pbc::kpts::KPointMesh;
use ferric_pbc::kroks::{solve_kroks, solve_kroks_on_grid, KRohfMutation};
use ferric_pbc::kscf::KJkKind;
use ferric_pbc::kuks::KUksMggaMutation;
use ferric_pbc::kuxcgrad::KUSpinMutation;
use ferric_pbc::kuxcstress::KUStressMutation;
use ferric_pbc::kxcgrad::KXcMutation;
use ferric_pbc::kxcstress::KXcStressMutation;
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

// ------------------------------------------- force and stress (#492, item 3)

/// Singlet k-ROKS vs k-RKS, per part, and fully polarized k-ROKS vs k-UKS.
/// Measured (2026-10-28, SCAN / r2SCAN): singlet <= 4.4e-16 / 7.8e-16, fully
/// polarized <= 1.9e-12 / 5.5e-14.
const DERIV_ANCHOR_BAR: f64 = 1e-11;
/// The FD row of [`h3_kroks_scan_force_and_stress_match_fd_row`]. Measured:
/// force 4.9e-10, `dE/dε` (1,2) 8.4e-11 at h = 5e-5 (orbital gradient
/// 5.7e-12). The diagonal strains (1,1) and (2,2) carry an h^2 stencil error
/// of ~3e-8 at this h, so the diagonal component has its own bar
/// ([`FD_DIAG_BAR`]).
const FD_H: f64 = 5e-5;
const FD_BAR: f64 = 1e-8;
const FD_FORCE_ROW: [(usize, usize); 2] = [(0, 2), (2, 1)];
const FD_STRAIN_ROW: [(usize, usize); 1] = [(1, 2)];
/// A diagonal strain (a volume-term defect shows here) at the same h, with a
/// looser bar: its raw h^2 stencil error is 3.2e-8 at h = 5e-5 (Richardson
/// would cost two more SCFs; the weekly file extrapolates). The derivative
/// mutants miss the stress by >= 3.3e-2.
const FD_DIAG_ROW: [(usize, usize); 1] = [(1, 1)];
const FD_DIAG_BAR: f64 = 1e-7;
const MUTANT_BAR: f64 = 1e-5;

/// EXACTNESS ANCHOR 1 for the derivatives: the singlet k-ROKS (H2, 1x1x3)
/// SCAN force and stress equal the k-RKS ones per part (independent SCF loop,
/// unpolarized kernel and total tau). Blind to a swap of `v_τα` and `v_τβ`
/// (equal here); the uniform grid has no grid response, so `xc_point` and
/// `xc_weight` are zero on both sides.
#[test]
fn singlet_k_roks_scan_force_and_stress_equal_k_rks_per_part() {
    let cell = cube_cell(&H2, 1);
    let bs = s05();
    let grid = uniform_grid(&cell);
    let n = [1, 1, 3];
    let fx = deriv_fx(&cell, &bs, n);
    let f = "SCAN";
    let ro = kroks(&cell, &bs, n, f, &grid);
    let r = krks(&cell, &bs, n, f, &grid);
    let rod = (
        roks_force(&fx, &ro.scf, f, &gcfg(&fx, None, None)).unwrap(),
        roks_stress(&fx, &ro.scf, f, &scfg(&fx, None, None)).unwrap(),
    );
    let rd = rks_derivs(&fx, &r.scf, f);
    assert!(rd.0.xc_ao.iter().any(|x| x.abs() > 1e-3) && mmax(&rd.1.xc_ao) > 1e-3);
    for (name, d) in part_rows(&rod, &rd) {
        eprintln!("H2 singlet {f} k-ROKS vs k-RKS: {name:<17} {d:.2e}");
        assert!(d < DERIV_ANCHOR_BAR, "{name}: {d:e}");
    }
}

/// EXACTNESS ANCHOR 2: a fully polarized state (H2 triplet, two s per H,
/// 1x1x3, `N_beta = 0`) has `D_beta = 0` and no closed shell, so `W_RO = D_α
/// F_α D_α` is the UKS `W` and the ROKS force and stress equal the k-UKS ones
/// of an independent k-UKS SCF, per part. Polarized tau terms; blind to the
/// `W_RO` cross term (it multiplies `D_β = 0`) and to a dropped beta tau term.
#[test]
fn fully_polarized_k_roks_scan_force_and_stress_equal_k_uks() {
    let cell = cube_cell(&H2, 3);
    let bs = two_s();
    let grid = uniform_grid(&cell);
    let n = [1, 1, 3];
    let fx = deriv_fx(&cell, &bs, n);
    let f = "SCAN";
    let ro = kroks(&cell, &bs, n, f, &grid);
    let u = kuks(&cell, &bs, n, f, &grid);
    let rod = (
        roks_force(&fx, &ro.scf, f, &gcfg(&fx, None, None)).unwrap(),
        roks_stress(&fx, &ro.scf, f, &scfg(&fx, None, None)).unwrap(),
    );
    let ud = uks_derivs(&fx, &u.scf, f);
    assert!(mmax(&ud.1.xc_ao) > 1e-3);
    for (name, d) in part_rows(&rod, &ud) {
        eprintln!("H2 triplet {f} k-ROKS vs k-UKS: {name:<17} {d:.2e}");
        assert!(d < DERIV_ANCHOR_BAR, "{name}: {d:e}");
    }
}

/// The POLARIZED check per commit: on the H3 doublet (N_alpha = 2, N_beta =
/// 1, a genuine closed-open-virtual ROHF state), 1x1x3 (complex Bloch AOs),
/// the SCAN force (two components) and `dE/dε` (one shear and one diagonal
/// component, the latter at a looser bar) against
/// the central FD of ferric's own converged k-ROKS energy (h = 5e-5). The
/// β `V^τ` dropped from the Pulay Fock (`TauNoVxcBeta`, Fock guard skipped)
/// is refused by the ROHF orbital-gradient gate, and `V^τ` dropped with the
/// gate off misses the same row; the full ledger is in
/// `validation_pbc_kroks_mgga.rs`.
#[test]
fn h3_kroks_scan_force_and_stress_match_fd_row() {
    let cell = cube_cell(&H3, 2);
    let n = [1, 1, 3];
    let f = "SCAN";
    let fx = deriv_fx(&cell, &s05(), n);
    let r = kroks(&cell, &s05(), n, f, &uniform_grid(&cell));
    assert!(spin_asymmetry(&r.scf) > 0.1, "not polarized");
    let g = roks_force(&fx, &r.scf, f, &gcfg(&fx, None, None)).unwrap();
    let s = roks_stress(&fx, &r.scf, f, &scfg(&fx, None, None)).unwrap();
    let fd_f = fd_roks_force(&H3, n, f, FD_H, &FD_FORCE_ROW);
    let fd_s = fd_roks_stress(&cell, n, f, FD_H, &FD_STRAIN_ROW);
    let wf = worst_force(&g.grad, &fd_f, &FD_FORCE_ROW);
    let ws = worst_stress(&s.de_deps, &fd_s, &FD_STRAIN_ROW);
    eprintln!(
        "H3 {f} 1x1x3: orbital gradient {:.2e}; force vs FD {wf:.2e}, dE/deps vs FD {ws:.2e} \
         (FD force {fd_f:?}, FD dE/deps {fd_s:?})",
        g.hf.commutator
    );
    assert!(
        fd_f.iter().chain(&fd_s).any(|x| x.abs() > 1e-2),
        "vacuous FD"
    );
    assert!(wf < FD_BAR, "force {wf:e}");
    assert!(ws < FD_BAR, "stress {ws:e}");
    let d_h = fd_roks_stress(&cell, n, f, FD_H, &FD_DIAG_ROW);
    let wd = worst_stress(&s.de_deps, &d_h, &FD_DIAG_ROW);
    eprintln!("H3 {f} dE/deps (1,1) vs FD {wd:.2e} (raw, h^2-limited)");
    assert!(wd < FD_DIAG_BAR, "diagonal stress {wd:e}");
    // The volume (grid-weight) term of the stress dropped misses the diagonal.
    let sv = roks_stress(
        &fx,
        &r.scf,
        f,
        &scfg(&fx, Some(KXcStressMutation::NoWeightDeriv), None),
    )
    .unwrap();
    let mv = worst_stress(&sv.de_deps, &d_h, &FD_DIAG_ROW);
    eprintln!("H3 {f} NoWeightDeriv: (1,1) miss {mv:.2e}");
    assert!(
        mv > MUTANT_BAR,
        "NoWeightDeriv survives the diagonal row: {mv:e}"
    );
    // β V^τ dropped from the rebuilt Fock (guard skipped): on a ROHF state
    // the orbital-gradient gate, evaluated on the rebuilt Focks, refuses it.
    let e = roks_force(
        &fx,
        &r.scf,
        f,
        &gcfg(&fx, None, Some(KUSpinMutation::TauNoVxcBeta)),
    )
    .expect_err("TauNoVxcBeta must be refused by the ROHF gate")
    .to_string();
    eprintln!("H3 {f} TauNoVxcBeta force: {e}");
    assert!(e.contains("ROHF orbital gradient"), "{e}");
    let e = roks_stress(
        &fx,
        &r.scf,
        f,
        &scfg(&fx, None, Some(KUStressMutation::TauNoVxcBeta)),
    )
    .expect_err("TauNoVxcBeta must be refused by the ROHF gate")
    .to_string();
    assert!(e.contains("ROHF orbital gradient"), "{e}");
    // With the gate off as well, the Pulay-side V^tau defect misses the FD row.
    let gm = roks_force(
        &fx,
        &r.scf,
        f,
        &gcfg(
            &fx,
            Some(KXcMutation::TauNoVxcUnguarded),
            Some(KUSpinMutation::RoGateOff),
        ),
    )
    .unwrap();
    let sm = roks_stress(
        &fx,
        &r.scf,
        f,
        &scfg(
            &fx,
            Some(KXcStressMutation::TauNoVxc),
            Some(KUStressMutation::RoGateOff),
        ),
    )
    .unwrap();
    let (mf, ms) = (
        worst_force(&gm.grad, &fd_f, &FD_FORCE_ROW),
        worst_stress(&sm.de_deps, &fd_s, &FD_STRAIN_ROW),
    );
    eprintln!("H3 {f} V^tau dropped, gate off: force miss {mf:.2e}, dE/deps miss {ms:.2e}");
    assert!(mf > MUTANT_BAR && ms > MUTANT_BAR, "{mf:e} {ms:e}");
}

/// Defence in depth: the ROKS meta-GGA force and stress refuse the RS-GDF
/// J/K by name before any integral, even for a hand-supplied state (the SCF
/// already refuses to make one; here a converged dense-AFT SCAN state is
/// handed in with an RS-GDF source).
#[test]
fn kroks_meta_gga_derivatives_refuse_rsgdf() {
    use ferric_pbc::dense_aft::ExxDiv;
    use ferric_pbc::kgrad::{KGradJk, KRsGdfGradSource};
    use ferric_pbc::kstress::KStressJk;
    use ferric_pbc::rsgdf::kpoint::{KRsGdf, KRsGdfConfig};
    use ferric_pbc::rsgdf::RsGdfConfig;
    let cell = cube_cell(&H3, 2);
    let n = [1, 1, 1];
    let fx = deriv_fx(&cell, &s05(), n);
    let r = kroks(&cell, &s05(), n, "SCAN", &uniform_grid(&cell));
    let aux = prep_for(&cell, &single_s_h(1.0));
    let kc = KRsGdfConfig {
        gdf: RsGdfConfig {
            omega: 0.8,
            exxdiv: ExxDiv::Ewald,
            budget_bytes: Some(AMPLE),
            ..Default::default()
        },
        mutation: None,
    };
    let gdf = KRsGdf::build(&cell, &fx.prep, &aux, &fx.mesh, &fx.hk.s, &kc).unwrap();
    let src = || KRsGdfGradSource {
        gdf: &gdf,
        cfg: &kc,
        aux: &aux,
    };
    for f in FUNCTIONALS {
        let e = ferric_pbc::kuxcgrad::kpoint_roks_gradient(
            &fx.cell,
            &fx.prep,
            &fx.mesh,
            &fx.hcfg,
            &fx.hk,
            KGradJk::RsGdf(src()),
            &r.scf,
            f,
            &gcfg(&fx, None, None),
        )
        .expect_err("ROKS meta-GGA force on RS-GDF")
        .to_string();
        assert!(
            e.contains("kpoint_roks_gradient") && e.contains("ROKS meta-GGA derivative on RS-GDF"),
            "{f}: {e}"
        );
        let e = ferric_pbc::kuxcstress::kpoint_roks_stress(
            &fx.cell,
            &fx.prep,
            &fx.mesh,
            &fx.hcfg,
            &fx.hk,
            KStressJk::RsGdf(src()),
            &r.scf,
            f,
            &scfg(&fx, None, None),
        )
        .expect_err("ROKS meta-GGA stress on RS-GDF")
        .to_string();
        assert!(
            e.contains("kpoint_roks_stress") && e.contains("ROKS meta-GGA derivative on RS-GDF"),
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
