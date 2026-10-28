//! Analytic stress of the k-point ROHF / ROKS energy
//! (`ferric_pbc::kuxcstress::{kpoint_rohf_stress, kpoint_roks_stress}`, module
//! doc "ROHF / ROKS"): the open-shell UKS stress assembly on the ROHF spin
//! densities with the orthonormality Lagrangian `W_RO` as the overlap weight
//! and a per-k ROHF orbital-gradient gate.
//!
//! WHICH ANCHOR COVERS WHICH DEFECT (measured in the sweeps named below; the
//! per-part residuals and bars are in the doc comment of each constant/test):
//!
//! * (1) closed-shell limit (no open orbital): ROHF == the RHF stress, ROKS ==
//!   the RKS stress, per part (floor 6.7e-16). Sees `RoWDropCross` (4.3e-1) and
//!   any overall spin factor; BLIND to `RoWUhfForm` (identity), to every
//!   alpha/beta asymmetry and to the exchange-scale defects at a = 1.
//! * (2) 1x1x1 vs the Gamma ROHF / ROKS stress per part (H3 doublet, H4
//!   triplet, floor <= 1.1e-11): `RoWDropCross` (5.5e-1), the XC spin defects
//!   (XcSpinSwap 1.6e-1, XcDropBeta, XcCrossDropped 1.6e-2, XcCrossOwn,
//!   XcSpinAveraged), the XC strain terms (NoAo, NoPointMotion, NoWeightDeriv,
//!   NoGga, NoHessian 4.6e-3 = smallest, NoStrainMoment), WrongFactor,
//!   ExchUnscaled (1.2), ExchScaledTwice (3.0e-1), MadelungUnscaled, CrossSpinK,
//!   NoPulay: all in `mutants_are_caught_by_the_gamma_roks_anchor` (ROKS PBE0).
//!   BLIND to `RoWUhfForm` (identity, 5.8e-14). `SwapSpinFock` acts only on the
//!   unrestricted assembly and is not in the RO mutant lists.
//! * the orbital-gradient gate: the defects that tamper with F (PulayFockSwap,
//!   PulayKSwap, PulayJOneSpin, PulayNoVxcBeta, PulayNoVxc, KWrongDensity) skip
//!   the Fock guard by construction and are REFUSED by the gate, not measured
//!   (asserted: the gate's message with g > 1e-3 and not the Fock guard's;
//!   g 7e-3 .. 4e-2 on the sweep state).
//! * (3) 1x1x3 vs the 3-copy Gamma supercell (ROHF, ROKS PBE0; floor 1.0e-11)
//!   anchors the correct value on the mesh. The MESH-ONLY mutants are NOT
//!   compared to the supercell but to the unmutated 1x1x3 k stress (which the
//!   supercell and the FD anchor): XC side NoInvNk, NoPhase, ConjPhase
//!   (KXcStressMutation) and HF side NoInvNk, NoPhase, ConjD, WrongQ,
//!   GammaMadelung, GammaMadelungStrain, CrossSpinK, plus RoWDropCross; the
//!   defects are in `k_dependent_mutants_are_caught_on_the_mesh` (ignored).
//! * (4) FD of the energy under symmetric strain: the stress of the energy as
//!   computed (all nine components for ROHF, ROKS PBE0, ROKS PBE; a
//!   Monkhorst-Pack mesh; the trace).
//! * (5) a = 1 null-XC through the ROKS SCF driver == the ROHF-driver stress.
//! * the gate itself: a loose state is refused (exact message, default set) and
//!   is wrong by 1.9e-4 (g 2.8e-4) with the gate OFF (ignored test, first-order slope
//!   band); the unrestricted-form `W` is no better (2.5e-4).
//!
//! WHAT THE DEFAULT SET DOES NOT PROTECT: the XC strain terms of ROKS (every
//! default test is ROHF or a refusal; ROKS is exercised in the default set only
//! by the gate/guard refusals), the a-scaling defects (a = 1 for ROHF), a
//! hybrid ROKS stress value, all FD components but one, every shear component,
//! Monkhorst-Pack meshes, the triplet, the supercell identities, the mutant
//! sweeps: all are ignored tests that must be run before trusting a change to
//! the XC strain or the exchange scaling.
//!
//! RS-GDF J/K: `pbc_kuxcstress_ro_rsgdf.rs`.
//!
//! NOT covered: p/d shells and heavy atoms / ECP (every
//! fixture is H/STO-3G), range-separated hybrids / meta-GGA / VV10 (refused),
//! the production (75, 302) grid, degenerate open shells (the gate's occupation
//! check rejects them), no external (PySCF) anchor.

mod common;

use common::*;
use ferric_pbc::dense_aft::ExxDiv;
use ferric_pbc::dense_aft::{DenseAftEri, DEFAULT_DENSE_AFT_MAX_BYTES};
use ferric_pbc::dft::PeriodicGridConfig;
use ferric_pbc::hcore::kpoint::{periodic_hcore_kpts, PeriodicHcoreK};
use ferric_pbc::hcore::periodic_hcore;
use ferric_pbc::hcore::PeriodicHcoreConfig;
use ferric_pbc::kdense_aft::KDenseAftEri;
use ferric_pbc::kdft::{solve_krks, KRksConfig};
use ferric_pbc::kpts::KPointMesh;
use ferric_pbc::kroks::{
    solve_krohf, solve_kroks, solve_kroks_injected, KRoScfResult, KRohfConfig, KRoksConfig,
    KRoksResult,
};
use ferric_pbc::kscf::{solve_krhf, KPointInjection, KPointJk, KRhfConfig};
use ferric_pbc::kstress::{kpoint_rhf_stress, KStressConfig, KStressJk, KStressMutation};
use ferric_pbc::kuxcgrad::RO_GATE_N2_ARRAYS;
use ferric_pbc::kuxcstress::{
    kpoint_null_xc_roks_stress, kpoint_rohf_stress, kpoint_roks_stress, KUStressMutation,
    KUksStress, KUksStressConfig,
};
use ferric_pbc::kxcgrad::KXcGrid;
use ferric_pbc::kxcstress::{kpoint_rks_stress, KRksStressConfig, KXcStressMutation};
use ferric_pbc::lattice::Cell;
use ferric_pbc::rohf::{gamma_rohf, gamma_roks, GammaRohfConfig, GammaRoksConfig};
use ferric_pbc::stress::{
    gamma_rohf_stress, gamma_roks_stress, GammaStress, GammaStressConfig, Mat3,
};
use ferric_pbc::uhf::GammaUhfIntegrals;
use num_complex::Complex64;

const OMEGA: f64 = 0.8;
const AMPLE: usize = 1 << 31;
/// H3 doublet (na = 2, nb = 1): closed 1, open 1, virtual 1 (n = 3).
const H3: [[f64; 3]; 3] = [[0.3, 0.2, 0.1], [0.35, 0.12, 1.5], [0.3, 1.9, 0.4]];
/// H4 triplet (na = 3, nb = 1): closed 1, open 2, virtual 1 (n = 4).
const H4: [[f64; 3]; 4] = [
    [0.3, 0.2, 0.1],
    [0.35, 0.12, 1.5],
    [0.3, 1.9, 0.4],
    [1.8, 0.5, 2.2],
];
const H2: [[f64; 3]; 2] = [[0.3, 0.2, 0.1], [0.35, 0.12, 1.5]];
const ALL9: [(usize, usize); 9] = [
    (0, 0),
    (0, 1),
    (0, 2),
    (1, 0),
    (1, 1),
    (1, 2),
    (2, 0),
    (2, 1),
    (2, 2),
];
/// Shared explicit neighbour cutoff (the default depends on the cell and puts a
/// one-step discontinuity in the energy; see `pbc_kuxcstress.rs`).
const CUTOFF: f64 = 14.0;
const ANCHOR_PRECISION: f64 = 1e-12;
/// FD step and bar. `ro_fd_window_probe` (H3 doublet, ROKS PBE0, dense Ewald,
/// 1x1x3, shared cutoff 14, all nine components): worst |FD - analytic| 5.4e-9
/// at h = 2.5e-5, 1.5e-8 at 5e-5, 5.4e-8 at 1e-4 (h^2 truncation; the
/// h = 2.5e-5 residual is the SCF-energy noise / 2h floor); all-nine at 2.5e-5:
/// ROHF Ewald 3.0e-9, ROKS PBE0 Ewald 5.4e-9, ROKS PBE none 5.4e-9; Monkhorst-
/// Pack (ROHF, PBE0) <= 7.6e-10; trace 8.2e-9. Bar 3e-8 = 5.6x the worst
/// correct residual and ~1.5e5 below the smallest mutant miss (NoHessian,
/// 4.6e-3 against the Gamma ROKS anchor).
const FD_H: f64 = 2.5e-5;
const FD_BAR: f64 = 3e-8;
const MUTANT_BAR: f64 = 1e-5;

#[derive(Clone, Copy)]
struct Mesh {
    n: [usize; 3],
    mp: bool,
}
const G111: Mesh = Mesh {
    n: [1, 1, 1],
    mp: false,
};
const G112: Mesh = Mesh {
    n: [1, 1, 2],
    mp: false,
};
const G113: Mesh = Mesh {
    n: [1, 1, 3],
    mp: false,
};

fn mesh_of(cell: &Cell, m: Mesh) -> KPointMesh {
    if m.mp {
        KPointMesh::monkhorst_pack(cell, m.n).unwrap()
    } else {
        KPointMesh::gamma_centred(cell, m.n).unwrap()
    }
}

thread_local! {
    /// (energy_conv, grad_conv) of the k-ROHF/ROKS SCF on THIS test thread.
    static SETTINGS: std::cell::Cell<(f64, f64)> = const { std::cell::Cell::new((1e-13, 1e-10)) };
}

fn grid_cfg() -> PeriodicGridConfig {
    let mut g = PeriodicGridConfig::with_size(20, 50);
    g.neighbour_cutoff = Some(CUTOFF);
    g
}

fn cell_of(pos: &[[f64; 3]]) -> Cell {
    let mut mol = hydrogens(pos);
    mol.multiplicity = match pos.len() {
        3 | 5 => 2,
        4 => 3,
        _ => 1,
    };
    Cell::new(mol, cubic(4.0)).expect("cell")
}

fn rohf_cfg(cell: &Cell, exxdiv: ExxDiv) -> KRohfConfig {
    let mut c = KRohfConfig::for_cell(cell, exxdiv);
    c.hcore = PeriodicHcoreConfig::with_omega(OMEGA);
    let (e, g) = SETTINGS.with(|s| s.get());
    c.scf.energy_conv = e;
    c.scf.grad_conv = g;
    c
}

/// A converged (unless the thread settings are loosened) ROHF/ROKS SCF.
fn scf_on(cell: &Cell, m: Mesh, functional: Option<&str>, exxdiv: ExxDiv) -> KRoksResult {
    let prep = prep_for(cell, &pyscf_sto3g_h());
    let mesh = mesh_of(cell, m);
    let rc = rohf_cfg(cell, exxdiv);
    let r = match functional {
        None => solve_krohf(cell, &prep, None, &mesh, &rc),
        Some(f) => {
            let mut c = KRoksConfig::new(cell, exxdiv, f);
            c.rohf = rc;
            c.grid = grid_cfg();
            solve_kroks(cell, &prep, None, &mesh, &c)
        }
    }
    .unwrap_or_else(|e| panic!("k-ROHF/ROKS {functional:?}: {e}"));
    assert!(r.scf.converged, "{functional:?} not converged");
    r
}

fn energy_on(cell: &Cell, m: Mesh, f: Option<&str>, exxdiv: ExxDiv) -> f64 {
    scf_on(cell, m, f, exxdiv).scf.energy
}

struct Fx {
    cell: Cell,
    prep: ferric_integrals::basis_bridge::PreparedBasis,
    mesh: KPointMesh,
    hk: PeriodicHcoreK,
    res: KRoksResult,
    exxdiv: ExxDiv,
    functional: Option<String>,
}

fn fixture_cell(cell: Cell, m: Mesh, functional: Option<&str>, exxdiv: ExxDiv) -> Fx {
    let prep = prep_for(&cell, &pyscf_sto3g_h());
    let mesh = mesh_of(&cell, m);
    let res = scf_on(&cell, m, functional, exxdiv);
    let hk =
        periodic_hcore_kpts(&cell, &prep, &mesh, &PeriodicHcoreConfig::with_omega(OMEGA)).unwrap();
    Fx {
        cell,
        prep,
        mesh,
        hk,
        res,
        exxdiv,
        functional: functional.map(|s| s.to_string()),
    }
}

fn fixture(pos: &[[f64; 3]], m: Mesh, functional: Option<&str>, exxdiv: ExxDiv) -> Fx {
    fixture_cell(cell_of(pos), m, functional, exxdiv)
}

fn scfg(
    xm: Option<KXcStressMutation>,
    sm: Option<KUStressMutation>,
    hm: Option<KStressMutation>,
) -> KUksStressConfig {
    KUksStressConfig {
        base: KRksStressConfig {
            stress: KStressConfig {
                budget_bytes: Some(AMPLE),
                mutation: hm,
                ..Default::default()
            },
            grid: KXcGrid::Atom(grid_cfg()),
            exxdiv: Some(ExxDiv::Ewald),
            mutation: xm,
            ..Default::default()
        },
        spin_mutation: sm,
    }
}

fn dense_eri(fx: &Fx) -> KDenseAftEri {
    KDenseAftEri::build(
        &fx.cell,
        &fx.prep,
        &fx.mesh,
        &fx.hk.s,
        fx.exxdiv,
        &Default::default(),
    )
    .unwrap()
}

/// The RO stress entry matching the fixture (ROHF if `functional` is `None`).
fn stress_with(
    fx: &Fx,
    scf: &KRoScfResult,
    cfg: &KUksStressConfig,
) -> Result<KUksStress, ferric_core::FerricError> {
    let hcfg = PeriodicHcoreConfig::with_omega(OMEGA);
    let eri = dense_eri(fx);
    let mut c = cfg.clone();
    c.base.exxdiv = Some(fx.exxdiv);
    match &fx.functional {
        None => kpoint_rohf_stress(
            &fx.cell,
            &fx.prep,
            &fx.mesh,
            &hcfg,
            &fx.hk,
            KStressJk::Dense(&eri),
            scf,
            fx.exxdiv,
            &c,
        ),
        Some(f) => kpoint_roks_stress(
            &fx.cell,
            &fx.prep,
            &fx.mesh,
            &hcfg,
            &fx.hk,
            KStressJk::Dense(&eri),
            scf,
            f,
            &c,
        ),
    }
}

fn stress(fx: &Fx) -> KUksStress {
    stress_with(fx, &fx.res.scf, &scfg(None, None, None)).expect("k RO stress")
}

fn stress_mut(
    fx: &Fx,
    xm: Option<KXcStressMutation>,
    sm: Option<KUStressMutation>,
    hm: Option<KStressMutation>,
) -> KUksStress {
    stress_with(fx, &fx.res.scf, &scfg(xm, sm, hm)).expect("k RO stress (mutant)")
}

// ------------------------------------------------------------ small helpers

fn mdiff(a: &Mat3, b: &Mat3) -> f64 {
    let mut m = 0.0_f64;
    for i in 0..3 {
        for j in 0..3 {
            m = m.max((a[i][j] - b[i][j]).abs());
        }
    }
    m
}

fn mmax(a: &Mat3) -> f64 {
    a.iter().flatten().fold(0.0_f64, |m, x| m.max(x.abs()))
}

fn mscale(a: &Mat3, f: f64) -> Mat3 {
    let mut o = *a;
    for r in o.iter_mut() {
        for x in r.iter_mut() {
            *x *= f;
        }
    }
    o
}

fn madd(a: &Mat3, b: &Mat3) -> Mat3 {
    let mut o = *a;
    for i in 0..3 {
        for j in 0..3 {
            o[i][j] += b[i][j];
        }
    }
    o
}

fn eps_at(i: usize, j: usize, h: f64) -> Mat3 {
    let mut e = [[0.0; 3]; 3];
    e[i][j] = h;
    e
}

fn strained(cell: &Cell, i: usize, j: usize, h: f64) -> Cell {
    cell.strained(&eps_at(i, j, h)).expect("strained cell")
}

/// Central FD of the ROHF/ROKS SCF energy for the listed components.
fn fd_components(
    cell: &Cell,
    m: Mesh,
    f: Option<&str>,
    exxdiv: ExxDiv,
    h: f64,
    comps: &[(usize, usize)],
) -> Vec<f64> {
    comps
        .iter()
        .map(|&(i, j)| {
            let ep = energy_on(&strained(cell, i, j, h), m, f, exxdiv);
            let em = energy_on(&strained(cell, i, j, -h), m, f, exxdiv);
            (ep - em) / (2.0 * h)
        })
        .collect()
}

// ------------------------------------------------------------------ (2)

fn gamma_side(cell: &Cell, f: Option<&str>) -> GammaStress {
    let prep = prep_for(cell, &pyscf_sto3g_h());
    let hcfg = PeriodicHcoreConfig::with_omega(OMEGA);
    let hc = periodic_hcore(cell, &prep, &hcfg).unwrap();
    let eri = DenseAftEri::build(
        cell,
        &prep,
        &hc.s,
        ExxDiv::Ewald,
        ANCHOR_PRECISION,
        DEFAULT_DENSE_AFT_MAX_BYTES,
    )
    .unwrap();
    let gc = GammaStressConfig {
        budget_bytes: Some(AMPLE),
        ..Default::default()
    };
    match f {
        None => {
            let mut c = GammaRohfConfig::default();
            c.exxdiv = ExxDiv::Ewald;
            c.scf.energy_conv = 1e-13;
            c.scf.density_conv = 1e-10;
            c.scf.max_iter = 600;
            let r = gamma_rohf(cell, &prep, &hc, GammaUhfIntegrals::DenseAft(&eri), &c).unwrap();
            gamma_rohf_stress(cell, &prep, &hcfg, &hc, &eri, &r.scf, ExxDiv::Ewald, &gc).unwrap()
        }
        Some(f) => {
            let mut c = GammaRoksConfig::new(f);
            c.grid = grid_cfg();
            c.exxdiv = ExxDiv::Ewald;
            c.scf.energy_conv = 1e-13;
            c.scf.density_conv = 1e-10;
            c.scf.max_iter = 600;
            let r = gamma_roks(cell, &prep, &hc, GammaUhfIntegrals::DenseAft(&eri), &c).unwrap();
            gamma_roks_stress(cell, &prep, &hcfg, &hc, &eri, &r.scf, &c, &gc).unwrap()
        }
    }
}

/// Rows of (name, |k part - Gamma part|) for `n_cells` cells against the
/// (super)cell Gamma stress (`dE/de` per cell = supercell value / n_cells).
fn gamma_rows(k: &KUksStress, g: &GammaStress, n_cells: f64) -> Vec<(&'static str, f64)> {
    let p = &g.parts;
    let kp = &k.hf.parts;
    let s = 1.0 / n_cells;
    vec![
        ("xc_ao", mdiff(&k.xc_ao, &mscale(&p.xc_ao, s))),
        ("xc_weight", mdiff(&k.xc_weight, &mscale(&p.xc_weight, s))),
        ("overlap", mdiff(&kp.overlap(), &mscale(&p.overlap, s))),
        ("kinetic", mdiff(&kp.kinetic, &mscale(&p.kinetic, s))),
        ("vsr", mdiff(&kp.vsr, &mscale(&p.vsr, s))),
        ("vlr", mdiff(&kp.vlr, &mscale(&p.vlr, s))),
        ("c0_volume", mdiff(&kp.c0_volume, &mscale(&p.c0_volume, s))),
        (
            "eri=J+K",
            mdiff(&madd(&kp.coulomb, &kp.exchange), &mscale(&p.eri, s)),
        ),
        ("madelung", mdiff(&kp.madelung, &mscale(&p.madelung, s))),
        ("sigma", mdiff(&k.sigma, &g.sigma)),
    ]
}

/// Per-part bar against the Gamma anchors. Worst correct residual 1.1e-11
/// (J+K, 1x1x1 ROKS PBE), 1.0e-11 (1x1x3 ROHF supercell), 2.1e-12 (1x1x3 ROKS
/// PBE0 supercell), 4.9e-12 (H4 triplet ROHF); smallest mutant miss 4.6e-3
/// (NoHessian). Bar 1e-9: 90x above the floor, 4.6e6 below the smallest mutant.
const BAR_GAMMA: f64 = 1e-9;

/// Per-part comparison with the Gamma stress; with `w_mutant`, the same
/// reference must reject the `W_RO` cross term dropped and accept the
/// unrestricted-form `W` (an identity at convergence).
fn one_point_vs_gamma(pos: &[[f64; 3]], f: Option<&str>, w_mutant: bool) {
    let gam = gamma_side(&cell_of(pos), f);
    let fx = fixture(pos, G111, f, ExxDiv::Ewald);
    let k = stress(&fx);
    for (n, d) in gamma_rows(&k, &gam, 1.0) {
        eprintln!("{f:?} H{} 1x1x1 vs Gamma RO: {n:<10} {d:.2e}", pos.len());
        assert!(d < BAR_GAMMA, "{f:?} {n}: {d:e}");
    }
    assert!(f == Some("PBE") || mmax(&k.hf.parts.exchange) > 1e-4);
    assert!(f.is_none() || (mmax(&k.xc_ao) > 1e-4 && mmax(&k.xc_weight) > 1e-4));
    // The reported gate value is the orbital gradient, tiny at convergence.
    assert!(k.hf.commutator < 1e-8, "{:e}", k.hf.commutator);
    if w_mutant {
        let drop = stress_mut(&fx, None, Some(KUStressMutation::RoWDropCross), None);
        let d = mdiff(&drop.de_deps, &gam.de_deps);
        eprintln!("{f:?} vs Gamma RO under RoWDropCross: miss {d:.2e}");
        assert!(
            d > MUTANT_BAR,
            "RoWDropCross survives the Gamma anchor: {d:e}"
        );
        let uhf = stress_mut(&fx, None, Some(KUStressMutation::RoWUhfForm), None);
        let d = mdiff(&uhf.de_deps, &k.de_deps);
        eprintln!("{f:?} RoWUhfForm at convergence: {d:.2e}");
        assert!(d < 1e-8, "identity at convergence: {d:e}");
    }
}

/// (2a) 1x1x1 == the Gamma ROHF stress, per part, on the H3 doublet.
#[test]
fn one_point_mesh_matches_the_gamma_rohf_stress_per_part() {
    one_point_vs_gamma(&H3, None, true);
}

#[test]
#[ignore = "H4 triplet ROHF vs Gamma ROHF; measured 64 s"]
fn one_point_mesh_matches_the_gamma_rohf_stress_per_part_triplet() {
    one_point_vs_gamma(&H4, None, true);
}

#[test]
#[ignore = "Gamma ROKS 1x1x1 anchors (PBE, PBE0); measured 58 s"]
fn one_point_mesh_matches_the_gamma_roks_stress_per_part() {
    one_point_vs_gamma(&H3, Some("PBE"), false);
    one_point_vs_gamma(&H3, Some("PBE0"), false);
}

// ------------------------------------------------------------------ (1)

/// Every labelled HF part of a `KStressParts`.
fn hf_parts(p: &ferric_pbc::kstress::KStressParts) -> Vec<(&'static str, Mat3)> {
    vec![
        ("overlap_pulay", p.overlap_pulay),
        ("overlap_c0", p.overlap_c0),
        ("overlap_madelung", p.overlap_madelung),
        ("kinetic", p.kinetic),
        ("vsr", p.vsr),
        ("ecp", p.ecp),
        ("vlr", p.vlr),
        ("c0_volume", p.c0_volume),
        ("coulomb", p.coulomb),
        ("exchange", p.exchange),
        ("madelung", p.madelung),
        ("nn_sr", p.nn_sr),
        ("nn_lr", p.nn_lr),
        ("nn_background", p.nn_background),
    ]
}

/// (1) CLOSED-SHELL LIMIT. A singlet H2 cell through the ROHF entry (no open
/// orbital: `W_RO = D_α F_α D_α + D_α F_β D_β` with `D_α = D_β`) is the k-point
/// RHF stress, per part (independent SCF loops: ROHF vs RHF). Artifact
/// hypothesis: a spin factor of 2 vs 1 or a wrong `W` normalisation breaks the
/// equality; physics predicts agreement at the SCF-density floor. The same
/// state is NOT blind to `RoWDropCross` (halves `W`): asserted visible.
#[test]
fn closed_shell_rohf_equals_the_rhf_stress_per_part() {
    let m = G112;
    let fx = fixture(&H2, m, None, ExxDiv::Ewald);
    assert_eq!(fx.res.scf.nalpha, fx.res.scf.nbeta);
    let ro = stress(&fx);
    let mut rc = KRhfConfig::for_cell(&fx.cell, ExxDiv::Ewald);
    rc.hcore = PeriodicHcoreConfig::with_omega(OMEGA);
    rc.scf.energy_conv = 1e-14;
    rc.scf.grad_conv = 1e-12;
    rc.scf.max_iter = 300;
    let r = solve_krhf(&fx.cell, &fx.prep, None, &fx.mesh, &rc).unwrap();
    let eri = dense_eri(&fx);
    let rhf = kpoint_rhf_stress(
        &fx.cell,
        &fx.prep,
        &fx.mesh,
        &PeriodicHcoreConfig::with_omega(OMEGA),
        &fx.hk,
        KStressJk::Dense(&eri),
        &r,
        ExxDiv::Ewald,
        &KStressConfig {
            budget_bytes: Some(AMPLE),
            ..Default::default()
        },
    )
    .unwrap();
    let mut rows = vec![("de_deps", mdiff(&ro.de_deps, &rhf.de_deps))];
    for ((n, a), (_, b)) in hf_parts(&ro.hf.parts).into_iter().zip(hf_parts(&rhf.parts)) {
        rows.push((n, mdiff(&a, &b)));
    }
    for (n, d) in rows {
        eprintln!("closed-shell ROHF vs RHF: {n:<18} {d:.2e}");
        assert!(d < BAR_CLOSED, "{n}: {d:e}");
    }
    assert!(mmax(&ro.hf.parts.exchange) > 1e-4);
    let drop = stress_mut(&fx, None, Some(KUStressMutation::RoWDropCross), None);
    let d = mdiff(&drop.de_deps, &rhf.de_deps);
    eprintln!("closed-shell RoWDropCross visible: {d:.2e}");
    assert!(d > MUTANT_BAR, "{d:e}");
    let uhf = stress_mut(&fx, None, Some(KUStressMutation::RoWUhfForm), None);
    let d = mdiff(&uhf.de_deps, &ro.de_deps);
    eprintln!("closed-shell RoWUhfForm (identity): {d:.2e}");
    assert!(d < BAR_CLOSED, "{d:e}");
}

/// Closed-shell residual bar: worst correct residual 6.7e-16 (ROHF vs RHF,
/// ROKS vs RKS); the smallest visible mutant miss is `RoWDropCross`, 4.3e-1.
/// Bar 1e-11: 1.5e4 above the floor, 4e10 below the mutant.
const BAR_CLOSED: f64 = 1e-11;

// ------------------------------------------------------------------ (5)

/// (5) HONEST SCOPE: this checks that two SCF drivers (ROHF vs ROKS-injected with
/// no XC) agree through the SAME stress assembly; it cannot see an a-scaling
/// defect (a = 1) nor any defect in the shared assembly.
/// a = 1 with a zero XC through the ROKS SCF driver (`solve_kroks_injected`,
/// no XC) is the ROHF stress of the ROHF driver, per part (two SCF drivers, two
/// entry points, one assembly). BLIND to every a-scaling defect (a = 1).
#[test]
#[ignore = "second SCF driver on 1x1x2; measured 60 s"]
fn null_xc_roks_route_equals_the_rohf_stress_per_part() {
    let fx = fixture(&H3, G112, None, ExxDiv::Ewald);
    let ro = stress(&fx);
    let hcfg = PeriodicHcoreConfig::with_omega(OMEGA);
    let vm = fx.mesh.madelung(&fx.cell).unwrap();
    let eri = KDenseAftEri::build(
        &fx.cell,
        &fx.prep,
        &fx.mesh,
        &fx.hk.s,
        ExxDiv::None,
        &Default::default(),
    )
    .unwrap();
    let inj = KPointInjection {
        s: fx.hk.s.clone(),
        h: fx.hk.h.clone(),
        vnn: fx.hk.enn,
        jk: Box::new(eri.jk_builder_with_madelung(vm)) as Box<dyn KPointJk + '_>,
    };
    let mut sc = ferric_pbc::kscf::KScfConfig::default();
    sc.energy_conv = 1e-14;
    sc.grad_conv = 1e-12;
    let scf = solve_kroks_injected(&fx.cell, &fx.mesh, &sc, inj, 2, 1, None, 1.0).unwrap();
    assert!(scf.converged);
    let deri = dense_eri(&fx);
    let z = kpoint_null_xc_roks_stress(
        &fx.cell,
        &fx.prep,
        &fx.mesh,
        &hcfg,
        &fx.hk,
        KStressJk::Dense(&deri),
        &scf,
        1.0,
        &scfg(None, None, None),
    )
    .unwrap();
    let mut rows = vec![("de_deps", mdiff(&z.de_deps, &ro.de_deps))];
    for ((n, a), (_, b)) in hf_parts(&z.hf.parts)
        .into_iter()
        .zip(hf_parts(&ro.hf.parts))
    {
        rows.push((n, mdiff(&a, &b)));
    }
    for (n, d) in rows {
        eprintln!("null-XC ROKS route vs ROHF: {n:<18} {d:.2e}");
        assert!(d < BAR_NULL, "{n}: {d:e}");
    }
    // ExchUnscaled / ExchScaledTwice are identities at a = 1 (asserted).
    for xm in [
        KXcStressMutation::ExchUnscaled,
        KXcStressMutation::ExchScaledTwice,
    ] {
        let d = mdiff(&stress_mut(&fx, Some(xm), None, None).de_deps, &ro.de_deps);
        assert!(d < 1e-14, "{xm:?} at a = 1 is an identity: {d:e}");
    }
}

/// Worst correct residual 8.5e-13 (the two SCF drivers at 1e-14/1e-12; 2.2e-10
/// when the injected SCF stops at 1e-13/1e-10, hence the tighter settings
/// there); mutants miss by >= 4.6e-3. Bar 1e-10: 120x above the floor.
const BAR_NULL: f64 = 1e-10;

// ------------------------------------------------------- the orbital gradient

/// (energy_conv, grad_conv) of the deliberately loose ROKS SCF of the gate
/// tests (`ro_stress_gate_probe` records the orbital gradient they give).
const LOOSE_RO: (f64, f64) = (1e-4, 1e-2);

fn loose_fixture() -> Fx {
    SETTINGS.with(|s| s.set(LOOSE_RO));
    let fx = fixture(&H3, G111, Some("PBE"), ExxDiv::None);
    SETTINGS.with(|s| s.set((1e-13, 1e-10)));
    fx
}

/// Parse the gate's exact message; returns the reported orbital gradient.
fn gate_message_value(e: &str, who: &str) -> f64 {
    let head = format!("{who}: ROHF orbital gradient ");
    let tail = " exceeds 1e-5: not a stationary ROHF state (the stress is first order in it); \
                converge the SCF tighter";
    let body = e.strip_prefix("General error: ").unwrap_or(e);
    let mid = body
        .strip_prefix(head.as_str())
        .and_then(|r| r.strip_suffix(tail))
        .unwrap_or_else(|| panic!("message differs: {e}"));
    mid.parse().unwrap_or_else(|_| panic!("number {mid}: {e}"))
}

/// A Fock-tampering defect skips the Fock guard (by construction of the
/// mutant), so what fires is the ORBITAL-GRADIENT GATE: assert exactly that
/// (the gate's message, not the Fock guard's), with g far above the tolerance.
fn assert_gate_refusal(label: &str, r: Result<KUksStress, ferric_core::FerricError>) {
    let m = r
        .err()
        .unwrap_or_else(|| panic!("{label}: a Fock-tampering defect was NOT refused"))
        .to_string();
    eprintln!("{label}: refused by the orbital-gradient gate: {m}");
    let g = gate_message_value(&m, "kpoint_roks_stress");
    assert!(g > 1e-3, "{label}: g = {g:e}");
    assert!(!m.contains("Fock matrix rebuilt"), "{label}: {m}");
}

/// The orbital-gradient gate FIRES in the default set: a deliberately loose
/// ROKS SCF is refused with the exact message and the reported value is above
/// `RO_ORBITAL_GRADIENT_TOL` (1e-5); the refusal is the gate's, not the Fock
/// guard's (the stored Fock is the density's own). Deleting the gate in
/// `uks_stress_with` makes this test fail (the stress is returned).
#[test]
fn loose_roks_scf_fires_the_orbital_gradient_gate_with_the_exact_message() {
    let loose = loose_fixture();
    let e = stress_with(&loose, &loose.res.scf, &scfg(None, None, None))
        .expect_err("loose ROKS accepted")
        .to_string();
    eprintln!("loose ROKS: {e}");
    assert!(!e.contains("Fock matrix rebuilt"), "{e}");
    let g = gate_message_value(&e, "kpoint_roks_stress");
    assert!(g > 1e-5, "{g:e}");
}

/// The same state with the gate off is wrong by far more than the FD bar, the
/// miss stays at its recorded slope (a REGRESSION PIN on one deterministic state, not a
/// first-order test: that is `gate_error_is_first_order_in_g_between_two_states`; two-sided slope band
/// from `ro_stress_gate_probe`), and the unrestricted-form `W` is no better:
/// the gate, not the form of `W`, protects off stationarity.
#[test]
#[ignore = "two ROKS SCFs + 3 stresses; measured 61 s"]
fn the_gate_matters_off_stationarity_and_the_uhf_form_is_no_remedy() {
    let tight = fixture(&H3, G111, Some("PBE"), ExxDiv::None);
    let st = stress(&tight);
    assert!(st.hf.commutator < 1e-8, "{:e}", st.hf.commutator);
    let loose = loose_fixture();
    let off = stress_mut(&loose, None, Some(KUStressMutation::RoGateOff), None);
    let g = off.hf.commutator;
    let miss = mdiff(&off.de_deps, &st.de_deps);
    let slope = miss / g;
    eprintln!(
        "loose ROKS: g = {g:.3e}, stress error with the gate off {miss:.3e}, slope {slope:.3}"
    );
    assert!(g > 1e-5, "state is not past the gate: g = {g:e}");
    assert!(
        miss > 10.0 * FD_BAR,
        "gate off is not wrong enough: {miss:e}"
    );
    assert!(
        (GATE_SLOPE_PIN.0..=GATE_SLOPE_PIN.1).contains(&slope),
        "the pinned slope of this deterministic state moved: miss {miss:e}, g {g:e}, slope {slope}"
    );
    // The UHF-form W is NOT a remedy: with the gate off it is wrong by a
    // comparable amount (and with the gate on it is refused like W_RO).
    let uhf_off = stress_mut(
        &loose,
        None,
        Some(KUStressMutation::RoWUhfFormGateOff),
        None,
    );
    let miss_uhf = mdiff(&uhf_off.de_deps, &st.de_deps);
    eprintln!("gate-off UHF-form W miss {miss_uhf:.3e} (W_RO form {miss:.3e})");
    assert!(
        miss_uhf > 10.0 * FD_BAR,
        "the UHF-form W is accurate off stationarity: {miss_uhf:e}"
    );
    let e = stress_with(
        &loose,
        &loose.res.scf,
        &scfg(None, Some(KUStressMutation::RoWUhfForm), None),
    )
    .expect_err("UHF-form W with the gate on")
    .to_string();
    assert!(e.contains("ROHF orbital gradient"), "{e}");
}

/// FIRST-ORDER check from two independent states (measured, ROKS PBE H3 1x1x1):
/// g 9.345e-7 (SCF 1e-6/1e-3) and g 7.495e-9 (SCF 1e-8/1e-4) differ by 125x in
/// g and 139x in the gate-off stress error. The ratio of the errors over the
/// ratio of the g's must be within a factor of two of 1 (measured 1.11).
#[test]
#[ignore = "three ROKS SCFs + 2 stresses; measured ~60 s"]
fn gate_error_is_first_order_in_g_between_two_states() {
    let st = stress(&fixture(&H3, G111, Some("PBE"), ExxDiv::None));
    let at = |ec: f64, gc: f64| {
        SETTINGS.with(|s| s.set((ec, gc)));
        let fx = fixture(&H3, G111, Some("PBE"), ExxDiv::None);
        SETTINGS.with(|s| s.set((1e-13, 1e-10)));
        let off = stress_mut(&fx, None, Some(KUStressMutation::RoGateOff), None);
        (off.hf.commutator, mdiff(&off.de_deps, &st.de_deps))
    };
    let (g1, e1) = at(1e-6, 1e-3);
    let (g2, e2) = at(1e-8, 1e-4);
    let r = (e1 / e2) / (g1 / g2);
    eprintln!("g {g1:.3e} -> {e1:.3e}; g {g2:.3e} -> {e2:.3e}; ratio of ratios {r:.3}");
    assert!(
        g1 / g2 > 50.0,
        "the two states are not far apart: {}",
        g1 / g2
    );
    assert!((0.5..=2.0).contains(&r), "not first order in g: {r}");
}

/// REGRESSION PIN (miss / g) of the gate-off stress error at one deterministic
/// state (the low outlier of the slopes below). `ro_stress_gate_probe`
/// (H3 doublet ROKS PBE 1x1x1, SCF stopped early; basis-independent orbital
/// gradient g -> max |dE/deps - tight| with the gate off): g 6.876e-3 ->
/// 2.397e-2 (slope 3.49); g 2.791e-4 -> 1.860e-4 (0.666); g 9.345e-7 ->
/// 3.658e-6 (3.91); g 7.495e-9 -> 2.635e-8 (3.52). The two small-g states differ
/// by 125x in g and 139x in error: first order. Slope 0.67 .. 3.9, so at
/// `RO_ORBITAL_GRADIENT_TOL` = 1e-5 the gate admits a dE/deps error of up to
/// about 4e-5 (about 1.3e3 times the FD bar; ~18 MPa on this 64 Bohr^3 cell):
/// looser than the anchors, it refuses a plainly unconverged state and does not
/// certify the stress to the anchors' accuracy. The band is a factor of two each
/// way around the slope 0.666 of the tested state (g 2.791e-4). Measured on the
/// H3 doublet, STO-3G, 1x1x1, ROKS PBE only (every block rank 1). (The superseded
/// F_eff-eigenvector form reported g 3.5e-5 for this same state.)
const GATE_SLOPE_PIN: (f64, f64) = (0.33, 1.33);

/// Probe behind the gate calibration: the orbital gradient g of ROKS SCFs
/// stopped early against the stress error with the gate off.
#[test]
#[ignore = "probe: 7 open-shell SCFs; measured 123 s"]
fn ro_stress_gate_probe() {
    let tight = fixture(&H3, G111, Some("PBE"), ExxDiv::None);
    let st = stress(&tight);
    for (ec, gc) in [
        (1e-2, 1e-1),
        (1e-3, 3e-2),
        (1e-4, 1e-2),
        (1e-5, 3e-3),
        (1e-6, 1e-3),
        (1e-8, 1e-4),
        (1e-10, 1e-6),
    ] {
        SETTINGS.with(|s| s.set((ec, gc)));
        let res = std::panic::catch_unwind(|| fixture(&H3, G111, Some("PBE"), ExxDiv::None));
        SETTINGS.with(|s| s.set((1e-13, 1e-10)));
        let loose = match res {
            Ok(f) => f,
            Err(_) => {
                eprintln!("conv ({ec:.0e},{gc:.0e}): SCF did not converge");
                continue;
            }
        };
        match stress_with(
            &loose,
            &loose.res.scf,
            &scfg(None, Some(KUStressMutation::RoGateOff), None),
        ) {
            Ok(off) => eprintln!(
                "conv ({ec:.0e},{gc:.0e}): g {:.3e}, stress error (gate off) {:.3e}",
                off.hf.commutator,
                mdiff(&off.de_deps, &st.de_deps)
            ),
            Err(e) => eprintln!("conv ({ec:.0e},{gc:.0e}): refused: {e}"),
        }
    }
}

/// H5 doublet (na = 3, nb = 2): closed 2, open 1, virtual 2, so the
/// closed-virtual block is 2 x 2 (rank up to 2) and the spectral norm is below
/// the Frobenius norm by up to sqrt(2).
const H5: [[f64; 3]; 5] = [
    [0.3, 0.2, 0.1],
    [0.35, 0.12, 1.5],
    [0.3, 1.9, 0.4],
    [1.8, 0.5, 2.2],
    [2.2, 3.0, 3.0],
];

/// PROBE: the same gate-off calibration as `ro_stress_gate_probe` on a system with
/// a rank > 1 block (H5 doublet, ROKS PBE, 1x1x1).
#[test]
#[ignore = "probe: 4 ROKS SCFs on H5; measured 415 s; run with --nocapture"]
fn ro_stress_gate_probe_rank2() {
    let st = stress(&fixture(&H5, G111, Some("PBE"), ExxDiv::None));
    for (ec, gc) in [(1e-4, 1e-2), (1e-6, 1e-3), (1e-8, 1e-4)] {
        SETTINGS.with(|s| s.set((ec, gc)));
        let res = std::panic::catch_unwind(|| fixture(&H5, G111, Some("PBE"), ExxDiv::None));
        SETTINGS.with(|s| s.set((1e-13, 1e-10)));
        let Ok(loose) = res else {
            eprintln!("RANK2 conv ({ec:.0e},{gc:.0e}): SCF did not converge");
            continue;
        };
        match stress_with(
            &loose,
            &loose.res.scf,
            &scfg(None, Some(KUStressMutation::RoGateOff), None),
        ) {
            Ok(off) => eprintln!(
                "RANK2 conv ({ec:.0e},{gc:.0e}): g {:.3e}, stress error (gate off) {:.3e}, slope {:.3}",
                off.hf.commutator,
                mdiff(&off.de_deps, &st.de_deps),
                mdiff(&off.de_deps, &st.de_deps) / off.hf.commutator
            ),
            Err(e) => eprintln!("RANK2 conv ({ec:.0e},{gc:.0e}): refused: {e}"),
        }
    }
}

// ------------------------------------------------------------- refusals

/// Typed refusals (exact wording) of the ROKS/ROHF entries: unsupported
/// functionals, a hybrid without exxdiv, the per-spin Fock guard from every
/// side, a short MO set and an unconverged SCF.
#[test]
fn refusals_and_the_fock_guard() {
    let fx = fixture(&H3, G111, Some("PBE"), ExxDiv::None);
    let ok_cfg = scfg(None, None, None);
    let eri = dense_eri(&fx);
    let run = |f: &str, s: &KRoScfResult, cfg: &KUksStressConfig| {
        kpoint_roks_stress(
            &fx.cell,
            &fx.prep,
            &fx.mesh,
            &PeriodicHcoreConfig::with_omega(OMEGA),
            &fx.hk,
            KStressJk::Dense(&eri),
            s,
            f,
            cfg,
        )
    };
    for (name, key) in [
        (
            "HSE06",
            // Differentiated since #492 on J/K that carry its range
            // separation; these bare kernels are refused by the input check.
            "but the dense kernels carry the bare Coulomb exchange; build the J/K with KDenseAftConfig::rsh set",
        ),
        (
            // SCAN / r2SCAN are differentiated since #492 (pbc_kroks_mgga.rs);
            // any other meta-GGA is refused by name before the Fock guard.
            "MGGA_X_TPSS",
            "kpoint_roks_stress: MGGA_X_TPSS: the k-point UKS / ROKS force and stress cover the pure SCAN and r2SCAN meta-GGAs only",
        ),
        ("wB97X-V", "VV10"),
        ("B2PLYP", "double hybrid"),
    ] {
        let e = run(name, &fx.res.scf, &ok_cfg)
            .err()
            .unwrap_or_else(|| panic!("{name} accepted"))
            .to_string();
        assert!(e.contains(key), "{name}: {e}");
    }
    let mut no_exx = scfg(None, None, None);
    no_exx.base.exxdiv = None;
    let e = run("PBE0", &fx.res.scf, &no_exx)
        .expect_err("no exxdiv")
        .to_string();
    assert!(
        e.contains("kpoint_roks_stress: a hybrid functional (exact exchange 0.25) needs KRksStressConfig::exxdiv"),
        "{e}"
    );
    let ok = run("PBE", &fx.res.scf, &ok_cfg).unwrap();
    assert!(ok.hf.fock_mismatch < 1e-12, "{:e}", ok.hf.fock_mismatch);
    assert!(ok.hf.commutator < 1e-8, "{:e}", ok.hf.commutator);
    let guard_msg = "kpoint_roks_stress: the Fock matrix rebuilt";
    let cases: Vec<(&str, Result<KUksStress, ferric_core::FerricError>)> = vec![(
        "PBE0 stress on a PBE SCF",
        run("PBE0", &fx.res.scf, &ok_cfg),
    )];
    for (n, r) in cases {
        let e = r
            .err()
            .unwrap_or_else(|| panic!("{n} accepted"))
            .to_string();
        assert!(e.contains(guard_msg), "{n}: {e}");
    }
    for (spin, which) in [(0, "fock_alpha"), (1, "fock_beta")] {
        let mut t = fx.res.scf.clone();
        let m = if spin == 0 {
            &mut t.fock_alpha
        } else {
            &mut t.fock_beta
        };
        m[0][(0, 1)] += Complex64::new(1e-6, 0.0);
        let e = run("PBE", &t, &ok_cfg).expect_err("tamper").to_string();
        assert!(e.contains(guard_msg) && e.contains(which), "{e}");
    }
    // A short MO set (one set per k is required).
    let mut t = fx.res.scf.clone();
    t.mos.pop();
    let e = run("PBE", &t, &ok_cfg).expect_err("short mos").to_string();
    eprintln!("short mos: {e}");
    assert!(
        e.contains("kpoint_roks_stress:") && e.contains("0 MO sets for 1 k-points"),
        "{e}"
    );
    // Unconverged SCF.
    let mut t = fx.res.scf.clone();
    t.converged = false;
    let e = run("PBE", &t, &ok_cfg)
        .expect_err("unconverged")
        .to_string();
    assert!(
        e.contains(
            "kpoint_roks_stress: the SCF did not converge; a gradient needs a stationary density"
        ),
        "{e}"
    );
}

/// The ROHF entry takes the exxdiv of the SCF: a mismatched one (None on an
/// Ewald SCF) is caught by the Fock guard with the exact message.
#[test]
fn rohf_with_a_mismatched_exxdiv_is_refused_by_the_fock_guard() {
    let fx = fixture(&H3, G111, None, ExxDiv::Ewald);
    let eri = KDenseAftEri::build(
        &fx.cell,
        &fx.prep,
        &fx.mesh,
        &fx.hk.s,
        ExxDiv::None,
        &Default::default(),
    )
    .unwrap();
    let e = kpoint_rohf_stress(
        &fx.cell,
        &fx.prep,
        &fx.mesh,
        &PeriodicHcoreConfig::with_omega(OMEGA),
        &fx.hk,
        KStressJk::Dense(&eri),
        &fx.res.scf,
        ExxDiv::None,
        &scfg(None, None, None),
    )
    .expect_err("mismatched exxdiv accepted")
    .to_string();
    assert!(
        e.contains("kpoint_rohf_stress: the Fock matrix rebuilt"),
        "{e}"
    );
}

// ------------------------------------------------------------------ memory

/// The byte count of a typed ledger error (`... (N bytes; ...`).
fn charge_of(m: &str) -> usize {
    let head = &m[..m.find(" bytes;").expect("byte count in the error")];
    head[head.rfind('(').unwrap() + 1..].parse().unwrap()
}

/// The orbital-gradient gate's scratch charge is read from the typed error the
/// PRODUCTION ROHF path raises when the budget holds the per-k arrays and the
/// Fock-guard matrices but not the gate's transients (zero XC: no chunk gate
/// stands in the way). Stage order: open_ledger, Fock-guard reservation, gate
/// scratch check, assembly. A probe error is "too small" iff it comes from
/// open_ledger or the guard, "too big" iff the run passes the gate. The charge
/// must be `RO_GATE_N2_ARRAYS` `n^2` complex arrays (the constant shared with
/// the allocator's documentation), and that count must cover the arrays
/// enumerable from the gate's code: 15 named arrays live at once plus one
/// product temporary plus the six of the spectral-norm stage = 22. A charge of
/// the old eight (or none) fails both checks.
#[test]
fn the_gate_scratch_charge_is_read_from_the_production_error() {
    let fx = fixture(&H3, G111, None, ExxDiv::Ewald);
    let at = |b: usize| {
        let mut cfg = scfg(None, None, None);
        cfg.base.stress.budget_bytes = Some(b);
        stress_with(&fx, &fx.res.scf, &cfg)
    };
    let (mut lo, mut hi) = (0usize, 1usize << 22);
    let mut found = None;
    for _ in 0..40 {
        let mid = lo + (hi - lo) / 2;
        match at(mid) {
            Err(e) => {
                let m = e.to_string();
                if m.contains("orbital-gradient gate scratch") {
                    found = Some(m);
                    break;
                } else if m.contains("k stress per-k complex")
                    || m.contains("residue-folded")
                    || m.contains("Fock-guard matrices")
                {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            Ok(_) => hi = mid,
        }
    }
    let m = found.expect("the bisection must land in the gate-scratch band");
    eprintln!("gate scratch: {m}");
    let n = fx.prep.nbasis();
    assert_eq!(charge_of(&m), RO_GATE_N2_ARRAYS * n * n * 16, "{m}");
    assert!(
        charge_of(&m) >= 22 * n * n * 16,
        "the charge covers fewer arrays than the gate holds: {m}"
    );
}

// ------------------------------------------------------------------ (4)

fn fd_case(f: Option<&str>, exxdiv: ExxDiv, m: Mesh, comps: &[(usize, usize)]) {
    let fx = fixture(&H3, m, f, exxdiv);
    let s = stress(&fx);
    let fd = fd_components(&fx.cell, m, f, exxdiv, FD_H, comps);
    let mut worst = 0.0_f64;
    for (&(i, j), d) in comps.iter().zip(&fd) {
        let miss = (s.de_deps[i][j] - d).abs();
        eprintln!(
            "{f:?} {exxdiv:?} eps({i},{j}): analytic {:.9e} FD {d:.9e} miss {miss:.2e}",
            s.de_deps[i][j]
        );
        worst = worst.max(miss);
    }
    assert!(worst < FD_BAR, "{f:?} {exxdiv:?}: {worst:e}");
}

/// (4) FD of ferric's own ROHF energy under symmetric strain at fixed
/// fractional coordinates (fresh SCF per displaced cell, mesh rebuilt on the
/// strained cell, shared cutoff 14), one component on a 1x1x2 mesh (default
/// set; the stress of a closed-open-virtual H3 doublet, non-TRIM mesh).
#[test]
fn rohf_stress_matches_fd_1x1x2() {
    fd_case(None, ExxDiv::Ewald, G112, &[(2, 2)]);
}

#[test]
#[ignore = "ROHF 1x1x3: 3 components x 2 SCFs; measured 31 s"]
fn rohf_stress_matches_fd_1x1x3() {
    fd_case(None, ExxDiv::Ewald, G113, &[(2, 2), (0, 2), (0, 0)]);
}

#[test]
#[ignore = "ROKS PBE row of the FD anchor (6 SCFs); measured 27 s"]
fn roks_pbe_stress_matches_fd_1x1x3() {
    fd_case(Some("PBE"), ExxDiv::None, G113, &[(2, 2), (0, 2), (1, 0)]);
}

#[test]
#[ignore = "ROKS PBE0 row of the FD anchor (6 SCFs); measured 37 s"]
fn roks_pbe0_stress_matches_fd_1x1x3() {
    fd_case(Some("PBE0"), ExxDiv::Ewald, G113, &[(2, 2), (0, 2)]);
}

#[test]
#[ignore = "8-point shifted mesh, ROHF + ROKS PBE0, 2 components x 2 SCFs; measured 409 s on a loaded box"]
fn ro_stress_matches_fd_on_the_monkhorst_pack_mesh() {
    let m = Mesh {
        n: [2, 2, 2],
        mp: true,
    };
    fd_case(None, ExxDiv::Ewald, m, &[(2, 2), (0, 1)]);
    fd_case(Some("PBE0"), ExxDiv::Ewald, m, &[(2, 2), (0, 1)]);
}

#[test]
#[ignore = "all-nine FD for ROHF Ewald, ROKS PBE0 Ewald and ROKS PBE none (54 SCFs); measured 145 s"]
fn ro_stress_matches_fd_all_nine() {
    for (f, exxdiv) in [
        (None, ExxDiv::Ewald),
        (Some("PBE0"), ExxDiv::Ewald),
        (Some("PBE"), ExxDiv::None),
    ] {
        let fx = fixture(&H3, G113, f, exxdiv);
        let s = stress(&fx);
        let fd = fd_components(&fx.cell, G113, f, exxdiv, FD_H, &ALL9);
        let mut worst = 0.0_f64;
        for (&(i, j), d) in ALL9.iter().zip(&fd) {
            worst = worst.max((d - s.de_deps[i][j]).abs());
        }
        eprintln!("{f:?} {exxdiv:?}: all-nine FD miss {worst:.2e}");
        assert!(worst < FD_BAR, "{f:?} {exxdiv:?}: {worst:e}");
    }
}

/// PROBE: the FD step window. All nine components of ROKS PBE0 (dense, Ewald,
/// 1x1x3 H3) at four steps; the printout is the floor and the window.
#[test]
#[ignore = "probe: 4 steps x 18 SCFs, measured 140 s; run with --nocapture"]
fn ro_fd_window_probe() {
    let fx = fixture(&H3, G113, Some("PBE0"), ExxDiv::Ewald);
    let s = stress(&fx);
    for h in [1e-5, 2.5e-5, 5e-5, 1e-4] {
        let fd = fd_components(&fx.cell, G113, Some("PBE0"), ExxDiv::Ewald, h, &ALL9);
        let mut row = String::new();
        let mut worst = 0.0_f64;
        for (&(i, j), d) in ALL9.iter().zip(&fd) {
            let m = (d - s.de_deps[i][j]).abs();
            worst = worst.max(m);
            row += &format!(" {m:.1e}");
        }
        eprintln!("h={h:.1e} worst {worst:.2e}; per component (00 01 02 10 11 12 20 21 22):{row}");
    }
}

/// Pressure from isotropic scaling: sum_a dE/de_aa equals the FD of E under
/// eps = h*I (ROKS PBE, dense, none).
#[test]
#[ignore = "ROKS PBE trace vs isotropic FD (3 SCFs); measured 21 s"]
fn ro_trace_matches_isotropic_scaling_fd() {
    let fx = fixture(&H3, G113, Some("PBE"), ExxDiv::None);
    let s = stress(&fx);
    let tr = s.de_deps[0][0] + s.de_deps[1][1] + s.de_deps[2][2];
    let iso = |h: f64| {
        let e = [[h, 0.0, 0.0], [0.0, h, 0.0], [0.0, 0.0, h]];
        let c = fx.cell.strained(&e).unwrap();
        energy_on(&c, G113, Some("PBE"), ExxDiv::None)
    };
    let fd = (iso(FD_H) - iso(-FD_H)) / (2.0 * FD_H);
    eprintln!(
        "ROKS PBE trace: analytic {tr:.9e} FD(isotropic) {fd:.9e} miss {:.2e}",
        (tr - fd).abs()
    );
    assert!((tr - fd).abs() < 3.0 * FD_BAR);
}

// ------------------------------------------------------------------ (3)

/// Explicit diag(n) supercell of `pos` (copy-major, copy 0 first).
fn supercell_pos(pos: &[[f64; 3]], a: f64, n: [usize; 3], mult: usize) -> Cell {
    let mut atoms = Vec::new();
    for m0 in 0..n[0] {
        for m1 in 0..n[1] {
            for m2 in 0..n[2] {
                for p in pos {
                    atoms.push([
                        p[0] + m0 as f64 * a,
                        p[1] + m1 as f64 * a,
                        p[2] + m2 as f64 * a,
                    ]);
                }
            }
        }
    }
    let lat = [
        [a * n[0] as f64, 0.0, 0.0],
        [0.0, a * n[1] as f64, 0.0],
        [0.0, 0.0, a * n[2] as f64],
    ];
    let mut mol = hydrogens(&atoms);
    mol.multiplicity = mult;
    Cell::new(mol, lat).unwrap()
}

/// (3) SUPERCELL ANCHOR on the non-TRIM 1x1x3 mesh: the k ROHF stress per cell
/// is the Gamma ROHF stress of the explicit 3-copy supercell (multiplicity
/// (Na - Nb) N_k + 1 = 4; sigma equal, dE/de / 3), per part. With Ewald the
/// supercell Gamma Madelung constant IS the mesh constant. The k-ROHF state is
/// the `RoWDropCross`-sensitive high-spin state; the Gamma ROHF supercell SCF
/// converging is a precondition (the force test could not converge it).
fn supercell_case(f: Option<&str>) {
    let gam = gamma_side(&supercell_pos(&H3, 4.0, [1, 1, 3], 4), f);
    let fx = fixture(&H3, G113, f, ExxDiv::Ewald);
    let k = stress(&fx);
    for (n, d) in gamma_rows(&k, &gam, 3.0) {
        eprintln!("{f:?} H3 1x1x3 vs supercell: {n:<10} {d:.2e}");
        assert!(d < BAR_GAMMA, "{f:?} {n}: {d:e}");
    }
    assert!(mmax(&k.hf.parts.madelung) > 1e-6);
}

#[test]
#[ignore = "9-atom Gamma ROHF supercell; precondition: the Gamma high-spin SCF converges; measured 54 s"]
fn k_mesh_rohf_stress_is_the_gamma_supercell_stress_per_part() {
    supercell_case(None);
}

#[test]
#[ignore = "9-atom Gamma ROKS PBE0 supercell; precondition: the Gamma high-spin SCF converges; measured 84 s"]
fn k_mesh_roks_stress_is_the_gamma_supercell_stress_per_part() {
    supercell_case(Some("PBE0"));
}

// ------------------------------------------------------- ROKS closed shell

/// (1b) ROKS closed-shell limit == the RKS stress (PBE, H2 1x1x3, per part).
#[test]
#[ignore = "closed-shell ROKS vs RKS (PBE); measured 15 s"]
fn closed_shell_roks_equals_the_rks_stress_per_part() {
    let f = "PBE";
    let fx = fixture(&H2, G113, Some(f), ExxDiv::None);
    let ro = stress(&fx);
    let mut rcfg = KRksConfig::new(&fx.cell, ExxDiv::None, f);
    rcfg.krhf.hcore = PeriodicHcoreConfig::with_omega(OMEGA);
    rcfg.krhf.scf.energy_conv = 1e-14;
    rcfg.krhf.scf.grad_conv = 1e-12;
    rcfg.krhf.scf.max_iter = 200;
    rcfg.grid = grid_cfg();
    let r = solve_krks(&fx.cell, &fx.prep, None, &fx.mesh, &rcfg).unwrap();
    let eri = dense_eri(&fx);
    let rks = kpoint_rks_stress(
        &fx.cell,
        &fx.prep,
        &fx.mesh,
        &PeriodicHcoreConfig::with_omega(OMEGA),
        &fx.hk,
        KStressJk::Dense(&eri),
        &r.scf,
        f,
        &KRksStressConfig {
            stress: KStressConfig {
                budget_bytes: Some(AMPLE),
                ..Default::default()
            },
            grid: KXcGrid::Atom(grid_cfg()),
            exxdiv: Some(ExxDiv::None),
            ..Default::default()
        },
    )
    .unwrap();
    let mut rows = vec![
        ("xc_ao", mdiff(&ro.xc_ao, &rks.xc_ao)),
        ("xc_weight", mdiff(&ro.xc_weight, &rks.xc_weight)),
        ("de_deps", mdiff(&ro.de_deps, &rks.de_deps)),
        ("sigma", mdiff(&ro.sigma, &rks.sigma)),
    ];
    for ((n, a), (_, b)) in hf_parts(&ro.hf.parts)
        .into_iter()
        .zip(hf_parts(&rks.hf.parts))
    {
        rows.push((n, mdiff(&a, &b)));
    }
    for (n, d) in rows {
        eprintln!("closed-shell ROKS vs RKS: {n:<18} {d:.2e}");
        assert!(d < BAR_CLOSED, "{n}: {d:e}");
    }
    assert!(mmax(&ro.xc_ao) > 1e-4 && mmax(&ro.xc_weight) > 1e-4);
}

// ----------------------------------------------------------------- mutants

/// Production-wired spin, XC and exchange-scale defects and the W_RO defects
/// against the Gamma ROKS stress at 1x1x1 (H3, PBE0, Ewald): the anchor that
/// sees the spin defects (Da != Db), the a-scaling defects (a = 0.25, Madelung
/// on) and the W_RO cross term. Each must miss the Gamma reference by far more
/// than the Gamma bar (the correct residual is <= BAR_GAMMA).
#[test]
#[ignore = "mutant sweep against the Gamma ROKS stress; measured 401 s"]
fn mutants_are_caught_by_the_gamma_roks_anchor() {
    use KStressMutation as S;
    use KUStressMutation as U;
    use KXcStressMutation as X;
    let f = Some("PBE0");
    let gam = gamma_side(&cell_of(&H3), f);
    let fx = fixture(&H3, G111, f, ExxDiv::Ewald);
    let sigma_gam = gam.de_deps;
    let base = stress(&fx);
    let d0 = mdiff(&base.de_deps, &sigma_gam);
    eprintln!("control: {d0:.2e}");
    assert!(d0 < BAR_GAMMA, "{d0:e}");
    let check = |label: String, s: KUksStress| {
        let d = mdiff(&s.de_deps, &sigma_gam);
        eprintln!("1x1x1 vs Gamma ROKS under {label}: miss {d:.2e}");
        assert!(d > MUTANT_BAR, "{label} survives the Gamma anchor: {d:e}");
    };
    for sm in [
        U::XcSpinSwap,
        U::XcDropBeta,
        U::XcCrossDropped,
        U::XcCrossOwn,
        U::XcSpinAveraged,
        U::RoWDropCross,
    ] {
        check(format!("{sm:?}"), stress_mut(&fx, None, Some(sm), None));
    }
    // The Fock-tampering defects skip the Fock guard; on the restricted-open
    // assembly the orbital-gradient gate then sees the tampered Focks. A defect
    // is caught if the gate refuses it OR it misses the Gamma reference.
    for sm in [
        U::PulayFockSwap,
        U::PulayKSwap,
        U::PulayJOneSpin,
        U::PulayNoVxcBeta,
    ] {
        assert_gate_refusal(
            &format!("{sm:?}"),
            stress_with(&fx, &fx.res.scf, &scfg(None, Some(sm), None)),
        );
    }
    for xm in [
        X::NoAo,
        X::NoPointMotion,
        X::NoWeightDeriv,
        X::NoGga,
        X::NoHessian,
        X::NoStrainMoment,
        X::WrongFactor,
        X::ExchUnscaled,
        X::ExchScaledTwice,
    ] {
        check(format!("{xm:?}"), stress_mut(&fx, Some(xm), None, None));
    }
    assert_gate_refusal(
        "PulayNoVxc",
        stress_with(&fx, &fx.res.scf, &scfg(Some(X::PulayNoVxc), None, None)),
    );
    for hm in [S::MadelungUnscaled, S::CrossSpinK, S::NoPulay] {
        check(format!("{hm:?}"), stress_mut(&fx, None, None, Some(hm)));
    }
    // The unrestricted-form W is an identity at convergence: reported, not counted.
    let uhf = stress_mut(&fx, None, Some(U::RoWUhfForm), None);
    let d = mdiff(&uhf.de_deps, &base.de_deps);
    eprintln!("RoWUhfForm at convergence (identity): {d:.2e}");
    assert!(d < 1e-8, "{d:e}");
}

/// The k-dependent defects need nk > 1: against the unmutated 1x1x3 ROKS PBE0
/// stress (itself anchored by the FD). Correct residuals are <= 1e-9.
#[test]
#[ignore = "11 ROKS PBE0 stresses on 1x1x3 (+1 gate refusal); measured 228 s for the first 7, ~330 s expected"]
fn k_dependent_mutants_are_caught_on_the_mesh() {
    use KStressMutation as S;
    use KXcStressMutation as X;
    let fx = fixture(&H3, G113, Some("PBE0"), ExxDiv::Ewald);
    let base = stress(&fx);
    // Compared against the UNMUTATED 1x1x3 k stress (itself anchored by the
    // supercell and the FD), not against the supercell. NoInvNk, NoPhase and
    // ConjPhase here are the XC-side defects (KXcStressMutation); the HF-side
    // ones (KStressMutation) follow.
    for (xm, sm, hm) in [
        (Some(X::NoInvNk), None, None),
        (Some(X::NoPhase), None, None),
        (Some(X::ConjPhase), None, None),
        (None, None, Some(S::NoInvNk)),
        (None, None, Some(S::NoPhase)),
        (None, None, Some(S::ConjD)),
        (None, None, Some(S::WrongQ)),
        (None, None, Some(S::GammaMadelung)),
        (None, None, Some(S::GammaMadelungStrain)),
        (None, None, Some(S::CrossSpinK)),
        (None, Some(KUStressMutation::RoWDropCross), None),
    ] {
        let d = mdiff(&stress_mut(&fx, xm, sm, hm).de_deps, &base.de_deps);
        eprintln!("1x1x3 {xm:?}/{sm:?}/{hm:?}: miss {d:.2e}");
        assert!(d > MUTANT_BAR, "{xm:?}/{sm:?}/{hm:?} survives: {d:e}");
    }
    // KWrongDensity tampers F (the Fock guard is skipped): the gate refuses it.
    assert_gate_refusal(
        "KWrongDensity",
        stress_with(&fx, &fx.res.scf, &scfg(Some(X::KWrongDensity), None, None)),
    );
}

#[test]
#[ignore = "two full ROKS PBE0 stresses at 1 and 4 threads; thread bit-identity of the shared assembly is also pinned by pbc_parallel_bitwise; measured 74 s"]
fn ro_stress_is_bit_identical_across_threads() {
    let fx = fixture(&H3, G113, Some("PBE0"), ExxDiv::Ewald);
    let run = |threads: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| stress(&fx))
    };
    let (a, b) = (run(1), run(4));
    assert_eq!(a.de_deps, b.de_deps);
    assert_eq!(a.xc_ao, b.xc_ao);
    assert_eq!(a.e_xc, b.e_xc);
}
