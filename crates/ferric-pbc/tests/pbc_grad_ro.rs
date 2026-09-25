//! Gamma-point periodic ROHF / ROKS analytic forces
//! (`ferric_pbc::grad::{gamma_rohf_gradient_with, gamma_roks_gradient_with,
//! *_rsgdf}`) and ROHF stress (`ferric_pbc::stress::gamma_rohf_stress`), the
//! Rust port of `reference/pbc/pbc_grad_ro.py` (FINDINGS "Iteration 20").
//!
//! The force is the UHF/UKS assembly on the ROHF spin densities and the
//! SPIN Focks rebuilt from them, with `W_RO = sym(D_α F_α D_α + D_α F_β D_β)`.
//!
//! What is anchored against what:
//! * EXACTNESS ANCHOR: analytic vs central FD of ferric's OWN
//!   `gamma_rohf` / `gamma_roks` energy, displaced SCFs seeded from the
//!   reference MOs (`initial_mos`) and checked to land on the SAME occupation
//!   state (closed/open subspace overlaps with the reference, `CONT_BAR`;
//!   prototype 4.2e-5 = the O(h) metric change). Prototype floors at
//!   h = 1e-4: H3 ROHF 3.5e-9, H3 ROKS LDA/PBE/PBE0 3.4e-9..3.6e-9, H3 RS-GDF
//!   2.8e-9, H4 ROHF 9.7e-9 (h² truncation). Bar 1e-7. ROKS uses h = 5e-5:
//!   on the SSF grid a displaced point can cross the hard `D` neighbour mask
//!   and the ENERGY jumps (FINDINGS h-scan: 2e-7 on H4, 1e-5 on the triclinic
//!   cell at h = 1e-4, identical for UKS — not a ROKS term).
//! * Mutants (hidden `GammaGradConfig::mutation`), each must miss FD by
//!   > `MUTANT_BAR = 1e-4` (prototype H3 smallest miss 4.8e-3 = `RoWNoCo`
//!   on ROKS LDA; the closed–open mutant scales with `|(f_α)_co|`, which is
//!   0.068 on H3 ROHF and only 0.004–0.007 on H4 — hence H3, where the test
//!   also asserts `|(f_α)_co| > 1e-2` so the mutant is not vacuous):
//!   `RoWFeffEps` (W from the Roothaan EFFECTIVE Fock, `ScfResult::fock_alpha`),
//!   `RoWNoCo`, `WSpinSum`, `ExchTotal` (α > 0), `MadelungTotal` (ewald,
//!   α > 0), `RoXcRksForm` (ROKS). `ΣF` is blind to all of them.
//! * NOT a mutant: the UHF-form `W = Σ_σ D_σ F_σ D_σ` (`RoWUhfForm`). It
//!   differs from `W_RO` by ½ × the closed–open β orbital gradient, i.e. it
//!   is an IDENTITY at the stationary point (memory "A mutation can be an
//!   identity at convergence"). Pinned instead as `|F(W_UHF) − F(W_RO)| ≤
//!   1e-10` and `|W_RO − W_UHF| ≤ 1e-8` (prototype ≤ 1.1e-12 / 2.4e-12), so a
//!   future change that breaks the identity turns red.
//! * Identities on ONE density: closed-shell ROHF(D/2, D/2) ≡ RHF(D) ≤ 1e-12,
//!   ROKS ≡ RKS ≤ 1e-11 (polarized vs unpolarized kernel); F(ewald) ≡
//!   F(none) ≤ 1e-11 (RHF-form Madelung mutant breaks it).
//! * ROHF vs UHF at the same geometry (UHF seeded from the ROHF MOs):
//!   reported; asserted only above the FD floor (prototype 8.45e-3 on H3 HF).
//! * Refusals: a UHF result, an unconverged flag, and a non-stationary ROHF
//!   state (closed/virtual rotated by 0.05 rad) are refused by name.
//!
//! Slow (`#[ignore]`): H4 ROHF triplet, H3 ROHF stress (all nine strain
//! components). Not covered: tri ROHF (does not converge, FINDINGS), ROKS
//! on RS-GDF with GGA, meta-GGA/RSH, k-points, ROKS stress FD.

mod common;

use common::*;
use ferric_core::basis::{self, BasisSet};
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_pbc::dense_aft::{
    DenseAftEri, ExxDiv, DEFAULT_DENSE_AFT_MAX_BYTES, DEFAULT_DENSE_AFT_PRECISION,
};
use ferric_pbc::dft::{gamma_rks, GammaRksConfig, PeriodicGridConfig};
use ferric_pbc::ewald::madelung_constant;
use ferric_pbc::grad::{
    gamma_rhf_gradient_with, gamma_rks_gradient_with, gamma_rohf_gradient_rsgdf,
    gamma_rohf_gradient_with, gamma_roks_gradient_with, gamma_uhf_gradient_with, rohf_lagrangian_w,
    rohf_orbital_gradient, GammaGradConfig, GammaGradient, GradMutation, RsGdfGradSource,
};
use ferric_pbc::hcore::{periodic_hcore, PeriodicHcore, PeriodicHcoreConfig};
use ferric_pbc::lattice::Cell;
use ferric_pbc::rohf::{
    gamma_rohf, gamma_roks, GammaRohfConfig, GammaRohfResult, GammaRoksConfig, GammaRoksResult,
};
use ferric_pbc::rsgdf::{RsGdf, RsGdfConfig, DEFAULT_RSGDF_LINDEP};
use ferric_pbc::stress::{gamma_rohf_stress, GammaStressConfig};
use ferric_pbc::uhf::{gamma_uhf, GammaUhfConfig, GammaUhfIntegrals};
use ferric_scf::fock::{JBuilder, KBuilder};
use ferric_scf::result::{ScfResult, Spin};
use ndarray::{s, Array2};
use std::sync::OnceLock;

/// ω of the nuclear-attraction split (as `pbc_grad.rs`).
const OMEGA: f64 = 0.8;
const HCORE_PRECISION: f64 = 1e-14;
const FD_H: f64 = 1e-4;
/// ROKS step (module doc: hard-mask energy jumps at h ≥ 1e-4).
const FD_H_KS: f64 = 5e-5;
const FD_BAR: f64 = 1e-7;
const MUTANT_BAR: f64 = 1e-4;
const NET_FORCE_BAR: f64 = 1e-10;
const IDENTITY_BAR: f64 = 1e-12;
/// ROKS(D/2, D/2) vs RKS(D): polarized vs unpolarized libxc kernel and the
/// per-spin density gates (as `pbc_grad_open.rs`).
const XC_IDENTITY_BAR: f64 = 1e-11;
const EWALD_NONE_BAR: f64 = 1e-11;
/// `|F(W_UHF-form) − F(W_RO)|` (prototype ≤ 1.1e-12 at SCF stop 1e-11;
/// ferric stops on ΔP_rms < 1e-10, so the SCF residual sets the scale).
const W_IDENTITY_FORCE_BAR: f64 = 1e-10;
/// `|W_RO − W_UHF-form|` elementwise (prototype ≤ 2.4e-12).
const W_IDENTITY_BAR: f64 = 1e-8;
/// Occupation-subspace continuity of a displaced SCF (prototype 4.2e-5).
const CONT_BAR: f64 = 1e-3;
/// ROHF ≠ UHF force: asserted only above the FD floor.
const RO_VS_U_FLOOR: f64 = 1e-5;
/// Stress FD bar for an open shell (`pbc_stress.rs` FD_BAR_OPEN).
const STRESS_FD_BAR_OPEN: f64 = 2e-7;
const STRESS_ANTISYM_BAR: f64 = 1e-10;
/// RS-GDF split and ample budget (as `pbc_grad_rsgdf.rs`).
const GDF_OMEGA: f64 = 1.0;
const AMPLE: usize = 1 << 31;

/// `run_grad_ro_anchor.py` H3 (Bohr), a = 4.5, STO-3G, doublet (nd, no) = (1, 1).
const H3_ATOMS: [[f64; 3]; 3] = [[0.3, 0.2, 0.1], [0.35, 0.12, 1.5], [1.6, 0.9, 0.7]];
const H3_A: f64 = 4.5;
const H3_COMPS: [(usize, usize); 3] = [(0, 0), (2, 1), (1, 2)];
/// `run_grad_ro_anchor.py` H4 (Bohr), a = 5, STO-3G, triplet (1, 2): two
/// stretched H2 units.
const H4_ATOMS: [[f64; 3]; 4] = [
    [0.3, 0.2, 0.1],
    [0.35, 0.12, 1.5],
    [2.6, 2.4, 2.2],
    [2.7, 2.35, 3.65],
];
const H4_A: f64 = 5.0;
const H4_COMPS: [(usize, usize); 3] = [(0, 0), (2, 1), (3, 2)];
/// Closed-shell H2 (Bohr), a = 4.
const GRAD_H2_ATOMS: [[f64; 3]; 2] = [[0.3, 0.2, 0.1], [0.35, 0.12, 1.5]];

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

type Fd = Vec<((usize, usize), Vec<f64>)>;
type Mat3 = [[f64; 3]; 3];

fn hcore_cfg() -> PeriodicHcoreConfig {
    PeriodicHcoreConfig {
        precision: HCORE_PRECISION,
        ..PeriodicHcoreConfig::with_omega(OMEGA)
    }
}

/// The prototype's anchor grid: (40, 50), SSF, D = 8.
fn grid_cfg() -> PeriodicGridConfig {
    PeriodicGridConfig {
        neighbour_cutoff: Some(8.0),
        ..PeriodicGridConfig::with_size(40, 50)
    }
}

fn cell_at(pos: &[[f64; 3]], lattice: [[f64; 3]; 3], mult: usize) -> Cell {
    let mut mol: Molecule = hydrogens(pos);
    mol.multiplicity = mult;
    Cell::new(mol, lattice).expect("cell")
}

fn moved(pos: &[[f64; 3]], a: usize, x: usize, h: f64) -> Vec<[f64; 3]> {
    let mut p = pos.to_vec();
    p[a][x] += h;
    p
}

fn h3_cell() -> Cell {
    cell_at(&H3_ATOMS, cubic(H3_A), 2)
}

struct Setup {
    cell: Cell,
    prep: PreparedBasis,
    hc: PeriodicHcore,
    eri: DenseAftEri,
}

/// hcore + the dense tensor (its own exxdiv is irrelevant: the drivers and
/// the gradients take exxdiv from their configs).
fn setup(cell: Cell, bs: &BasisSet) -> Setup {
    let prep = prep_for(&cell, bs);
    let hc = periodic_hcore(&cell, &prep, &hcore_cfg()).expect("hcore");
    let eri = DenseAftEri::build(
        &cell,
        &prep,
        &hc.s,
        ExxDiv::None,
        DEFAULT_DENSE_AFT_PRECISION,
        DEFAULT_DENSE_AFT_MAX_BYTES,
    )
    .expect("dense AFT");
    Setup {
        cell,
        prep,
        hc,
        eri,
    }
}

fn rohf_cfg(exx: ExxDiv, init: Option<Array2<f64>>) -> GammaRohfConfig {
    GammaRohfConfig {
        exxdiv: exx,
        initial_mos: init,
        ..GammaRohfConfig::default()
    }
}

/// `GammaRoksConfig::new` (hybrids: level shift 0.05, 600 iterations) on
/// the anchor grid.
fn roks_cfg(xc: &str, exx: ExxDiv, init: Option<Array2<f64>>) -> GammaRoksConfig {
    GammaRoksConfig {
        grid: grid_cfg(),
        exxdiv: exx,
        initial_mos: init,
        ..GammaRoksConfig::new(xc)
    }
}

fn rohf_on(su: &Setup, cfg: &GammaRohfConfig) -> GammaRohfResult {
    let r = gamma_rohf(
        &su.cell,
        &su.prep,
        &su.hc,
        GammaUhfIntegrals::DenseAft(&su.eri),
        cfg,
    )
    .expect("gamma_rohf");
    assert!(r.scf.converged);
    assert_eq!(r.scf.spin, Spin::RestrictedOpen);
    r
}

fn roks_on(su: &Setup, cfg: &GammaRoksConfig) -> GammaRoksResult {
    let r = gamma_roks(
        &su.cell,
        &su.prep,
        &su.hc,
        GammaUhfIntegrals::DenseAft(&su.eri),
        cfg,
    )
    .unwrap_or_else(|e| panic!("gamma_roks {}: {e}", cfg.functional));
    assert!(r.scf.converged);
    r
}

fn gcfg(mutation: Option<GradMutation>) -> GammaGradConfig {
    GammaGradConfig {
        mutation,
        ..Default::default()
    }
}

fn rohf_grad(su: &Setup, scf: &ScfResult, exx: ExxDiv, m: Option<GradMutation>) -> GammaGradient {
    gamma_rohf_gradient_with(
        &su.cell,
        &su.prep,
        &hcore_cfg(),
        &su.hc,
        &su.eri,
        scf,
        exx,
        &gcfg(m),
    )
    .expect("gamma_rohf_gradient")
}

fn roks_grad(
    su: &Setup,
    scf: &ScfResult,
    cfg: &GammaRoksConfig,
    m: Option<GradMutation>,
) -> GammaGradient {
    gamma_roks_gradient_with(
        &su.cell,
        &su.prep,
        &hcore_cfg(),
        &su.hc,
        &su.eri,
        scf,
        cfg,
        &gcfg(m),
    )
    .expect("gamma_roks_gradient")
}

fn max_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

/// Central FD `(E(+h) − E(−h))/2h` of every energy `energies` returns.
fn fd<F>(
    pos: &[[f64; 3]],
    lattice: [[f64; 3]; 3],
    mult: usize,
    comps: &[(usize, usize)],
    h: f64,
    energies: F,
) -> Fd
where
    F: Fn(&Cell) -> Vec<f64>,
{
    comps
        .iter()
        .map(|&(a, x)| {
            let ep = energies(&cell_at(&moved(pos, a, x, h), lattice, mult));
            let em = energies(&cell_at(&moved(pos, a, x, -h), lattice, mult));
            let d = ep
                .iter()
                .zip(&em)
                .map(|(p, m)| (p - m) / (2.0 * h))
                .collect();
            ((a, x), d)
        })
        .collect()
}

fn max_fd_err(g: &Array2<f64>, fdv: &Fd, k: usize) -> f64 {
    fdv.iter()
        .map(|((a, x), v)| (g[(*a, *x)] - v[k]).abs())
        .fold(0.0_f64, f64::max)
}

/// `max(|‖C_cᵀ S C_c^ref‖² − nd|, |‖C_oᵀ S C_o^ref‖² − no|)`: 0 (up to the
/// O(h) metric change) when a displaced SCF continues the reference
/// occupation pattern. ROHF MO columns are `[closed | open | virtual]`.
fn continuity(
    r: &ScfResult,
    nocc: (usize, usize),
    c_ref: &Array2<f64>,
    s_mat: &Array2<f64>,
) -> f64 {
    let (na, nb) = nocc;
    let c = &r.mos_alpha;
    let blk = |lo: usize, hi: usize| -> f64 {
        if hi == lo {
            return 0.0;
        }
        let ov = c
            .slice(s![.., lo..hi])
            .t()
            .dot(s_mat)
            .dot(&c_ref.slice(s![.., lo..hi]));
        (ov.iter().map(|v| v * v).sum::<f64>() - (hi - lo) as f64).abs()
    };
    blk(0, nb).max(blk(nb, na))
}

/// ROHF spin Focks `F_σ = h + J[D] − K[D_σ] − v_M S D_σ S` rebuilt in the
/// test from the dense tensor (independent of the gradient's own rebuild).
fn rohf_spin_focks(su: &Setup, scf: &ScfResult, exx: ExxDiv) -> (Array2<f64>, Array2<f64>) {
    let n = su.prep.nbasis();
    let vm = match exx {
        ExxDiv::None => 0.0,
        ExxDiv::Ewald => madelung_constant(&su.cell).unwrap(),
    };
    let da = &scf.density_alpha;
    let db = scf.density_beta.as_ref().unwrap();
    let mut j = Array2::<f64>::zeros((n, n));
    su.eri.j_builder().build(&(da + db), &mut j).unwrap();
    let mut ka = Array2::<f64>::zeros((n, n));
    let mut kb = Array2::<f64>::zeros((n, n));
    su.eri
        .k_builder_with_madelung(vm)
        .build(da, &mut ka)
        .unwrap();
    su.eri
        .k_builder_with_madelung(vm)
        .build(db, &mut kb)
        .unwrap();
    let base = &su.hc.h + &j;
    (&base - &ka, &base - &kb)
}

// ------------------------------------------------------------------ H3 ROHF

/// Reference H3 doublet ROHF: `(none stage, ewald)` of the staged run.
fn h3_rohf_ref(su: &Setup) -> (GammaRohfResult, ScfResult, ScfResult) {
    let r = rohf_on(su, &rohf_cfg(ExxDiv::Ewald, None));
    let none = r.none_stage.clone().expect("staged none stage");
    let ewald = r.scf.clone();
    (r, none, ewald)
}

static H3_ROHF_FD: OnceLock<Fd> = OnceLock::new();

/// FD `[none, ewald]` of the H3 ROHF energy (one staged run per geometry,
/// seeded from the reference none-stage MOs; each stage checked to continue
/// the reference occupation state).
fn h3_rohf_fd() -> &'static Fd {
    H3_ROHF_FD.get_or_init(|| {
        let bs = pyscf_sto3g_h();
        let (r0, none, ewald) = h3_rohf_ref(&setup(h3_cell(), &bs));
        let init = Some(none.mos_alpha.clone());
        fd(&H3_ATOMS, cubic(H3_A), 2, &H3_COMPS, FD_H, |c| {
            let su = setup(c.clone(), &bs);
            let r = rohf_on(&su, &rohf_cfg(ExxDiv::Ewald, init.clone()));
            let rn = r.none_stage.expect("none stage");
            let cn = continuity(&rn, r0.nocc, &none.mos_alpha, &su.hc.s);
            let ce = continuity(&r.scf, r0.nocc, &ewald.mos_alpha, &su.hc.s);
            assert!(
                cn < CONT_BAR && ce < CONT_BAR,
                "displaced H3 ROHF left the reference state: {cn:e} / {ce:e}"
            );
            vec![rn.energy, r.scf.energy]
        })
    })
}

#[test]
fn h3_rohf_force_matches_fd_of_own_energy_both_exxdiv() {
    let su = setup(h3_cell(), &pyscf_sto3g_h());
    let (_, none, ewald) = h3_rohf_ref(&su);
    let fdv = h3_rohf_fd();
    for (k, (scf, exx)) in [(&none, ExxDiv::None), (&ewald, ExxDiv::Ewald)]
        .into_iter()
        .enumerate()
    {
        let g = rohf_grad(&su, scf, exx, None);
        let err = max_fd_err(&g.grad, fdv, k);
        eprintln!(
            "H3 ROHF doublet {exx:?}: max|analytic − FD| = {err:.2e}, |ΣF| {:.1e}, \
             ROHF orbital gradient {:.1e}\n{:.10}",
            g.net_force, g.commutator, g.grad
        );
        assert!(err < FD_BAR, "{exx:?}: {err:e}");
        assert!(g.net_force < NET_FORCE_BAR, "ΣF {:e}", g.net_force);
        assert!(g.e_xc.is_none() && g.n_grid_points == 0);
        assert!((g.exact_exchange_fraction - 1.0).abs() < 1e-15);
    }
}

#[test]
fn h3_rohf_fd_anchor_catches_each_mutant() {
    let su = setup(h3_cell(), &pyscf_sto3g_h());
    let (_, none, ewald) = h3_rohf_ref(&su);
    let fdv = h3_rohf_fd();
    // The closed–open mutant is sized by |(f_α)_co| (prototype 0.068 here).
    let (fa, fb) = rohf_spin_focks(&su, &none, ExxDiv::None);
    let db = none.density_beta.as_ref().unwrap();
    let og = rohf_orbital_gradient(&none.mos_alpha, &su.hc.s, &none.density_alpha, db, &fa, &fb)
        .expect("orbital gradient");
    eprintln!("H3 ROHF: {og:?}");
    assert_eq!((og.n_closed, og.n_open), (1, 1));
    assert!(
        og.alpha_closed_open > 1e-2,
        "RoWNoCo would be near-vacuous: |(f_α)_co| = {:e}",
        og.alpha_closed_open
    );
    for (m, exx) in [
        (GradMutation::RoWFeffEps, ExxDiv::None),
        (GradMutation::RoWFeffEps, ExxDiv::Ewald),
        (GradMutation::RoWNoCo, ExxDiv::None),
        (GradMutation::RoWNoCo, ExxDiv::Ewald),
        (GradMutation::WSpinSum, ExxDiv::None),
        (GradMutation::WSpinSum, ExxDiv::Ewald),
        (GradMutation::ExchTotal, ExxDiv::None),
        (GradMutation::ExchTotal, ExxDiv::Ewald),
        (GradMutation::MadelungTotal, ExxDiv::Ewald),
    ] {
        let (scf, k) = match exx {
            ExxDiv::None => (&none, 0),
            ExxDiv::Ewald => (&ewald, 1),
        };
        let g = rohf_grad(&su, scf, exx, Some(m));
        let err = max_fd_err(&g.grad, fdv, k);
        eprintln!(
            "H3 ROHF mutant {m:?} ({exx:?}): max|analytic − FD| = {err:.2e}, |ΣF| {:.1e}",
            g.net_force
        );
        assert!(
            err > MUTANT_BAR,
            "{m:?} ({exx:?}) escaped the FD anchor: {err:e}"
        );
    }
}

/// The UHF-form W is an IDENTITY at the ROHF stationary point (module doc):
/// `W_RO − Σ_σ D_σ F_σ D_σ = ½[C_o (f_β)_oc C_cᵀ + h.c.]`, zero where the
/// closed–open β gradient vanishes. Pinned so that a change breaking it (or
/// a W_RO that stops being the Lagrangian) turns red; NOT a must-fail mutant.
#[test]
fn h3_rohf_uhf_form_w_is_an_identity_at_convergence() {
    let su = setup(h3_cell(), &pyscf_sto3g_h());
    let (_, none, ewald) = h3_rohf_ref(&su);
    for (scf, exx) in [(&none, ExxDiv::None), (&ewald, ExxDiv::Ewald)] {
        let (fa, fb) = rohf_spin_focks(&su, scf, exx);
        let da = &scf.density_alpha;
        let db = scf.density_beta.as_ref().unwrap();
        let w_ro = rohf_lagrangian_w(da, db, &fa, &fb);
        let w_u = da.dot(&fa).dot(da) + db.dot(&fb).dot(db);
        let dw = max_diff(&w_ro, &w_u);
        let og = rohf_orbital_gradient(&scf.mos_alpha, &su.hc.s, da, db, &fa, &fb).unwrap();
        let g = rohf_grad(&su, scf, exx, None);
        let gu = rohf_grad(&su, scf, exx, Some(GradMutation::RoWUhfForm));
        let df = max_diff(&g.grad, &gu.grad);
        eprintln!(
            "H3 ROHF {exx:?}: |W_RO − W_UHF| {dw:.2e}, |(f_β)_co| {:.2e}, \
             |F(W_UHF) − F(W_RO)| {df:.2e}, |W_RO| {:.3}",
            og.closed_open,
            w_ro.iter().fold(0.0_f64, |m, v| m.max(v.abs()))
        );
        assert!(dw < W_IDENTITY_BAR, "{exx:?}: |ΔW| {dw:e}");
        assert!(df < W_IDENTITY_FORCE_BAR, "{exx:?}: |ΔF| {df:e}");
        // Not vacuous: the W the port could get wrong (F_eff eigenpairs) is
        // far from both.
        let gf = rohf_grad(&su, scf, exx, Some(GradMutation::RoWFeffEps));
        assert!(max_diff(&g.grad, &gf.grad) > MUTANT_BAR);
    }
}

/// F(ewald) ≡ F(none) on ONE ROHF density; the RHF-form Madelung mutant
/// breaks it.
#[test]
fn h3_rohf_ewald_and_none_forces_agree_on_one_density() {
    let su = setup(h3_cell(), &pyscf_sto3g_h());
    let (_, _, ewald) = h3_rohf_ref(&su);
    let gn = rohf_grad(&su, &ewald, ExxDiv::None, None);
    let ge = rohf_grad(&su, &ewald, ExxDiv::Ewald, None);
    let d = max_diff(&gn.grad, &ge.grad);
    eprintln!(
        "H3 ROHF: |F_ewald − F_none| = {d:.2e} (v_M = {})",
        ge.madelung
    );
    assert!(ge.madelung > 0.1);
    assert!(d < EWALD_NONE_BAR, "{d:e}");
    let gm = rohf_grad(
        &su,
        &ewald,
        ExxDiv::Ewald,
        Some(GradMutation::MadelungTotal),
    );
    let dm = max_diff(&gn.grad, &gm.grad);
    eprintln!("H3 ROHF MadelungTotal mutant: |F_ewald − F_none| = {dm:.2e}");
    assert!(dm > MUTANT_BAR, "{dm:e}");
}

/// ROHF and UHF forces at the same geometry differ at the size of the spin
/// contamination (prototype H3 HF: ΔE 5.06e-4, max|ΔF| 8.45e-3). Reported;
/// asserted only above the FD floor, with E_RO ≥ E_U.
#[test]
fn h3_rohf_and_uhf_forces_differ() {
    let su = setup(h3_cell(), &pyscf_sto3g_h());
    let (_, none, _) = h3_rohf_ref(&su);
    let g_ro = rohf_grad(&su, &none, ExxDiv::None, None);
    let c = none.mos_alpha.clone();
    let u = gamma_uhf(
        &su.cell,
        &su.prep,
        &su.hc,
        GammaUhfIntegrals::DenseAft(&su.eri),
        &GammaUhfConfig {
            exxdiv: ExxDiv::None,
            initial_mos: Some((c.clone(), c)),
            ..GammaUhfConfig::default()
        },
    )
    .expect("gamma_uhf");
    assert!(u.scf.converged);
    let g_u = gamma_uhf_gradient_with(
        &su.cell,
        &su.prep,
        &hcore_cfg(),
        &su.hc,
        &su.eri,
        &u.scf,
        ExxDiv::None,
        &GammaGradConfig::default(),
    )
    .unwrap();
    let de = none.energy - u.scf.energy;
    let df = max_diff(&g_ro.grad, &g_u.grad);
    eprintln!(
        "H3 HF: E_RO − E_U = {de:.3e}, max|F_RO − F_U| = {df:.3e}, max|F| {:.3}",
        g_ro.grad.iter().fold(0.0_f64, |m, v| m.max(v.abs()))
    );
    assert!(de > -1e-10, "ROHF below UHF: {de:e}");
    assert!(df > RO_VS_U_FLOOR, "{df:e}");
}

// ------------------------------------------------------------------ H3 ROKS

/// Reference H3 doublet ROKS: LDA, PBE (α = 0: exxdiv does not enter),
/// PBE0 none (the none stage of the staged run) and PBE0 ewald.
fn h3_roks_refs(su: &Setup) -> Vec<(GammaRoksConfig, ScfResult)> {
    let lda = roks_on(su, &roks_cfg("LDA", ExxDiv::Ewald, None)).scf;
    let pbe = roks_on(su, &roks_cfg("PBE", ExxDiv::Ewald, None)).scf;
    let pbe0 = roks_on(su, &roks_cfg("PBE0", ExxDiv::Ewald, None));
    let pbe0_none = pbe0.none_stage.clone().expect("staged PBE0 none stage");
    vec![
        (roks_cfg("LDA", ExxDiv::Ewald, None), lda),
        (roks_cfg("PBE", ExxDiv::Ewald, None), pbe),
        (roks_cfg("PBE0", ExxDiv::None, None), pbe0_none),
        (roks_cfg("PBE0", ExxDiv::Ewald, None), pbe0.scf),
    ]
}

static H3_ROKS_FD: OnceLock<Fd> = OnceLock::new();

/// FD of the four H3 ROKS energies at h = 5e-5 (grid rebuilt per geometry;
/// seeded with the reference MOs; state continuity checked).
fn h3_roks_fd() -> &'static Fd {
    H3_ROKS_FD.get_or_init(|| {
        let bs = pyscf_sto3g_h();
        let refs = h3_roks_refs(&setup(h3_cell(), &bs));
        let nocc = (2, 1);
        fd(&H3_ATOMS, cubic(H3_A), 2, &H3_COMPS, FD_H_KS, |c| {
            let su = setup(c.clone(), &bs);
            let seed = |k: usize| Some(refs[k].1.mos_alpha.clone());
            let lda = roks_on(&su, &roks_cfg("LDA", ExxDiv::Ewald, seed(0)));
            let pbe = roks_on(&su, &roks_cfg("PBE", ExxDiv::Ewald, seed(1)));
            // The staged PBE0 run starts from the none-stage MOs.
            let pbe0 = roks_on(&su, &roks_cfg("PBE0", ExxDiv::Ewald, seed(2)));
            let pbe0_none = pbe0.none_stage.expect("none stage");
            let states = [&lda.scf, &pbe.scf, &pbe0_none, &pbe0.scf];
            for (k, r) in states.iter().enumerate() {
                let cont = continuity(r, nocc, &refs[k].1.mos_alpha, &su.hc.s);
                assert!(
                    cont < CONT_BAR,
                    "ROKS state {k} left the reference: {cont:e}"
                );
            }
            states.iter().map(|r| r.energy).collect()
        })
    })
}

#[test]
fn h3_roks_force_matches_fd_of_own_energy_with_grid_response() {
    let su = setup(h3_cell(), &pyscf_sto3g_h());
    let fdv = h3_roks_fd();
    for (k, (cfg, scf)) in h3_roks_refs(&su).iter().enumerate() {
        let g = roks_grad(&su, scf, cfg, None);
        let err = max_fd_err(&g.grad, fdv, k);
        let resp = &g.parts.xc_point + &g.parts.xc_weight;
        let resp_max = resp.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        eprintln!(
            "H3 ROKS {} {:?}: max|analytic − FD| = {err:.2e}, |ΣF| {:.1e}, orbital gradient \
             {:.1e}, npts {}, |grid response| {resp_max:.2e}\n{:.10}",
            cfg.functional, cfg.exxdiv, g.net_force, g.commutator, g.n_grid_points, g.grad
        );
        assert!(err < FD_BAR, "{} {:?}: {err:e}", cfg.functional, cfg.exxdiv);
        assert!(g.net_force < NET_FORCE_BAR, "ΣF {:e}", g.net_force);
        assert!(resp_max > 1e-4, "grid response {resp_max:e}");
    }
}

#[test]
fn h3_roks_fd_anchor_catches_each_mutant_and_uhf_form_w_is_an_identity() {
    let su = setup(h3_cell(), &pyscf_sto3g_h());
    let fdv = h3_roks_fd();
    for (k, (cfg, scf)) in h3_roks_refs(&su).iter().enumerate() {
        let hybrid = cfg.functional == "PBE0";
        let mut muts = vec![
            GradMutation::RoWFeffEps,
            GradMutation::RoWNoCo,
            GradMutation::WSpinSum,
            GradMutation::RoXcRksForm,
        ];
        if hybrid {
            muts.push(GradMutation::ExchTotal);
            if cfg.exxdiv == ExxDiv::Ewald {
                muts.push(GradMutation::MadelungTotal);
            }
        }
        for m in muts {
            let g = roks_grad(&su, scf, cfg, Some(m));
            let err = max_fd_err(&g.grad, fdv, k);
            eprintln!(
                "H3 ROKS {} {:?} mutant {m:?}: max|analytic − FD| = {err:.2e}, |ΣF| {:.1e}",
                cfg.functional, cfg.exxdiv, g.net_force
            );
            assert!(
                err > MUTANT_BAR,
                "{} {:?}: {m:?} escaped the FD anchor: {err:e}",
                cfg.functional,
                cfg.exxdiv
            );
        }
        let g = roks_grad(&su, scf, cfg, None);
        let gu = roks_grad(&su, scf, cfg, Some(GradMutation::RoWUhfForm));
        let df = max_diff(&g.grad, &gu.grad);
        eprintln!(
            "H3 ROKS {} {:?}: |F(W_UHF) − F(W_RO)| {df:.2e} (identity)",
            cfg.functional, cfg.exxdiv
        );
        assert!(df < W_IDENTITY_FORCE_BAR, "{df:e}");
    }
}

#[test]
fn h3_roks_pbe0_ewald_and_none_forces_agree_on_one_density() {
    let su = setup(h3_cell(), &pyscf_sto3g_h());
    let r = roks_on(&su, &roks_cfg("PBE0", ExxDiv::Ewald, None));
    let gn = roks_grad(&su, &r.scf, &roks_cfg("PBE0", ExxDiv::None, None), None);
    let ge = roks_grad(&su, &r.scf, &roks_cfg("PBE0", ExxDiv::Ewald, None), None);
    let d = max_diff(&gn.grad, &ge.grad);
    eprintln!("H3 ROKS PBE0: |F_ewald − F_none| = {d:.2e}");
    assert!((ge.exact_exchange_fraction - 0.25).abs() < 1e-12);
    assert!(d < EWALD_NONE_BAR, "{d:e}");
}

// ---------------------------------------------------- closed-shell identities

/// A closed-shell restricted result as a restricted-open one `(D/2, D/2)`
/// on the same MOs (`nd = N/2`, `no = 0`).
fn as_restricted_open(r: &ScfResult) -> ScfResult {
    let mut o = r.clone();
    let half = &r.density_total * 0.5;
    o.spin = Spin::RestrictedOpen;
    o.density_alpha = half.clone();
    o.density_beta = Some(half);
    o.mos_beta = None;
    o
}

/// ROHF(D/2, D/2) ≡ RHF(D) on one density, both exxdiv.
#[test]
fn closed_shell_rohf_gradient_is_the_rhf_gradient() {
    let su = setup(cell_at(&GRAD_H2_ATOMS, cubic(4.0), 1), &pyscf_sto3g_h());
    let eri = su.eri.clone().with_exxdiv(&su.cell, ExxDiv::Ewald).unwrap();
    let scf = gamma_rhf(&su.cell, &su.prep, &su.hc, &eri);
    let ro = as_restricted_open(&scf);
    for exx in [ExxDiv::None, ExxDiv::Ewald] {
        let grhf = gamma_rhf_gradient_with(
            &su.cell,
            &su.prep,
            &hcore_cfg(),
            &su.hc,
            &su.eri,
            &scf,
            exx,
            &GammaGradConfig::default(),
        )
        .unwrap();
        let grohf = rohf_grad(&su, &ro, exx, None);
        let d = max_diff(&grhf.grad, &grohf.grad);
        eprintln!("H2 {exx:?}: |F_ROHF(D/2,D/2) − F_RHF(D)| = {d:.2e}");
        assert!(d < IDENTITY_BAR, "{exx:?}: {d:e}");
    }
}

/// ROKS(D/2, D/2) ≡ RKS(D) on one density, PBE and PBE0 with ewald.
#[test]
fn closed_shell_roks_gradient_is_the_rks_gradient() {
    let su = setup(cell_at(&GRAD_H2_ATOMS, cubic(4.0), 1), &pyscf_sto3g_h());
    for xc in ["PBE", "PBE0"] {
        let rcfg = GammaRksConfig {
            grid: grid_cfg(),
            exxdiv: ExxDiv::Ewald,
            ..GammaRksConfig::new(xc)
        };
        let scf = gamma_rks(
            &su.cell,
            &su.prep,
            &su.hc,
            GammaUhfIntegrals::DenseAft(&su.eri),
            &rcfg,
        )
        .unwrap()
        .scf;
        let grks = gamma_rks_gradient_with(
            &su.cell,
            &su.prep,
            &hcore_cfg(),
            &su.hc,
            &su.eri,
            &scf,
            &rcfg,
            &GammaGradConfig::default(),
        )
        .unwrap();
        let groks = roks_grad(
            &su,
            &as_restricted_open(&scf),
            &roks_cfg(xc, ExxDiv::Ewald, None),
            None,
        );
        let d = max_diff(&grks.grad, &groks.grad);
        eprintln!("H2 {xc}: |F_ROKS(D/2,D/2) − F_RKS(D)| = {d:.2e}");
        assert!(d < XC_IDENTITY_BAR, "{xc}: {d:e}");
    }
}

// ---------------------------------------------------------------- refusals

#[test]
fn ro_gradients_refuse_wrong_kind_unconverged_and_non_stationary() {
    let su = setup(h3_cell(), &pyscf_sto3g_h());
    let (_, none, _) = h3_rohf_ref(&su);
    let call = |scf: &ScfResult| {
        gamma_rohf_gradient_with(
            &su.cell,
            &su.prep,
            &hcore_cfg(),
            &su.hc,
            &su.eri,
            scf,
            ExxDiv::None,
            &GammaGradConfig::default(),
        )
    };
    // A UHF result.
    let u = gamma_uhf(
        &su.cell,
        &su.prep,
        &su.hc,
        GammaUhfIntegrals::DenseAft(&su.eri),
        &GammaUhfConfig {
            exxdiv: ExxDiv::None,
            ..GammaUhfConfig::default()
        },
    )
    .unwrap();
    let e = call(&u.scf).expect_err("a UHF result must be refused");
    assert!(e.to_string().contains("restricted open-shell"), "{e}");
    // An unconverged flag.
    let mut bad = none.clone();
    bad.converged = false;
    let e = call(&bad).expect_err("an unconverged result must be refused");
    assert!(e.to_string().contains("did not converge"), "{e}");
    // A non-stationary ROHF state: rotate closed (0) with virtual (2) by
    // 0.05 rad (S-orthonormality kept), densities rebuilt consistently.
    let c = &none.mos_alpha;
    assert!(c.ncols() >= 3);
    let (cs, sn) = (0.05_f64.cos(), 0.05_f64.sin());
    let mut cr = c.clone();
    let c0 = c.column(0).to_owned();
    let c2 = c.column(2).to_owned();
    cr.column_mut(0).assign(&(&c0 * cs + &c2 * sn));
    cr.column_mut(2).assign(&(&c2 * cs - &c0 * sn));
    let cc = cr.slice(s![.., 0..1]).to_owned();
    let co = cr.slice(s![.., 1..2]).to_owned();
    let db = cc.dot(&cc.t());
    let da = &db + &co.dot(&co.t());
    let mut rot = none.clone();
    rot.mos_alpha = cr;
    rot.density_total = &da + &db;
    rot.density_alpha = da;
    rot.density_beta = Some(db);
    let e = call(&rot).expect_err("a non-stationary ROHF state must be refused");
    assert!(e.to_string().contains("orbital gradient"), "{e}");
    // The untouched result passes the same gate.
    let g = call(&none).expect("converged ROHF");
    assert!(g.commutator < 1e-7, "orbital gradient {:e}", g.commutator);
}

// ------------------------------------------------------------ H3 RS-GDF ROHF

fn gdf_cfg() -> RsGdfConfig {
    RsGdfConfig {
        omega: GDF_OMEGA,
        lindep: DEFAULT_RSGDF_LINDEP,
        exxdiv: ExxDiv::None,
        budget_bytes: Some(AMPLE),
        ..Default::default()
    }
}

fn cc_pvdz_ri() -> BasisSet {
    basis::bundled("cc-pvdz-ri").expect("cc-pvdz-ri")
}

fn rohf_gdf(
    cell: &Cell,
    prep: &PreparedBasis,
    hc: &PeriodicHcore,
    gdf: &RsGdf,
    init: Option<Array2<f64>>,
) -> GammaRohfResult {
    let r = gamma_rohf(
        cell,
        prep,
        hc,
        GammaUhfIntegrals::RsGdf(gdf),
        &rohf_cfg(ExxDiv::Ewald, init),
    )
    .expect("gamma_rohf RS-GDF");
    assert!(r.scf.converged);
    r
}

/// H3 ROHF with RS-GDF J/K (cc-pvdz-ri, prototype 2.8e-9), both exxdiv, and
/// the two W mutants the fit path shares with dense AFT (prototype
/// 4.1e-2 / 1.7e-1 and 1.1e-2).
#[test]
fn h3_rohf_rsgdf_force_matches_fd_and_catches_w_mutants() {
    let bs = pyscf_sto3g_h();
    let aux_bs = cc_pvdz_ri();
    let cell = h3_cell();
    let prep = prep_for(&cell, &bs);
    let hc = periodic_hcore(&cell, &prep, &hcore_cfg()).unwrap();
    let aux = PreparedBasis::new(cell.mol(), &aux_bs).unwrap();
    let gdf = RsGdf::build_for_gradient(&cell, &prep, &aux, &hc.s, &gdf_cfg())
        .expect("build_for_gradient");
    let r = rohf_gdf(&cell, &prep, &hc, &gdf, None);
    let none = r.none_stage.clone().expect("none stage");
    let init = Some(none.mos_alpha.clone());
    let fdv = fd(&H3_ATOMS, cubic(H3_A), 2, &H3_COMPS, FD_H, |c| {
        let p = prep_for(c, &bs);
        let h = periodic_hcore(c, &p, &hcore_cfg()).unwrap();
        let a = PreparedBasis::new(c.mol(), &aux_bs).unwrap();
        let g = RsGdf::build(c, &p, &a, &h.s, &gdf_cfg()).unwrap();
        let rr = rohf_gdf(c, &p, &h, &g, init.clone());
        vec![rr.none_stage.expect("none stage").energy, rr.scf.energy]
    });
    let src = RsGdfGradSource {
        gdf: &gdf,
        aux: &aux,
        aux_jac: None,
    };
    let grad = |scf: &ScfResult, exx: ExxDiv, m: Option<GradMutation>| {
        gamma_rohf_gradient_rsgdf(
            &cell,
            &prep,
            &hcore_cfg(),
            &hc,
            &src,
            scf,
            exx,
            &GammaGradConfig {
                mutation: m,
                budget_bytes: Some(AMPLE),
                ..Default::default()
            },
        )
        .expect("gamma_rohf_gradient_rsgdf")
    };
    for (k, (scf, exx)) in [(&none, ExxDiv::None), (&r.scf, ExxDiv::Ewald)]
        .into_iter()
        .enumerate()
    {
        let g = grad(scf, exx, None);
        let err = max_fd_err(&g.grad, &fdv, k);
        eprintln!(
            "H3 ROHF RS-GDF {exx:?}: max|analytic − FD| = {err:.2e}, |ΣF| {:.1e}\n{:.10}",
            g.net_force, g.grad
        );
        assert!(err < FD_BAR, "{exx:?}: {err:e}");
        assert!(g.net_force < NET_FORCE_BAR, "ΣF {:e}", g.net_force);
        for m in [GradMutation::RoWFeffEps, GradMutation::RoWNoCo] {
            let e = max_fd_err(&grad(scf, exx, Some(m)).grad, &fdv, k);
            eprintln!("  mutant {m:?}: {e:.2e}");
            assert!(e > MUTANT_BAR, "{m:?} ({exx:?}) escaped: {e:e}");
        }
        let du = max_diff(
            &g.grad,
            &grad(scf, exx, Some(GradMutation::RoWUhfForm)).grad,
        );
        assert!(du < W_IDENTITY_FORCE_BAR, "UHF-form W identity: {du:e}");
    }
}

// ------------------------------------------------------ H4 ROHF triplet (slow)

#[test]
#[ignore = "slow: H4 ROHF triplet, 6 displaced dense-AFT builds + 12 staged SCF stages; run with --release -- --ignored"]
fn h4_rohf_triplet_force_matches_fd_both_exxdiv() {
    let bs = pyscf_sto3g_h();
    let h4 = || cell_at(&H4_ATOMS, cubic(H4_A), 3);
    let su = setup(h4(), &bs);
    let r0 = rohf_on(&su, &rohf_cfg(ExxDiv::Ewald, None));
    let none = r0.none_stage.clone().expect("none stage");
    let ewald = r0.scf.clone();
    assert_eq!(r0.nocc, (3, 1));
    let init = Some(none.mos_alpha.clone());
    let fdv = fd(&H4_ATOMS, cubic(H4_A), 3, &H4_COMPS, FD_H, |c| {
        let s = setup(c.clone(), &bs);
        let r = rohf_on(&s, &rohf_cfg(ExxDiv::Ewald, init.clone()));
        let rn = r.none_stage.expect("none stage");
        let cn = continuity(&rn, r0.nocc, &none.mos_alpha, &s.hc.s);
        let ce = continuity(&r.scf, r0.nocc, &ewald.mos_alpha, &s.hc.s);
        assert!(
            cn < CONT_BAR && ce < CONT_BAR,
            "H4 left the state: {cn:e} / {ce:e}"
        );
        vec![rn.energy, r.scf.energy]
    });
    for (k, (scf, exx)) in [(&none, ExxDiv::None), (&ewald, ExxDiv::Ewald)]
        .into_iter()
        .enumerate()
    {
        let g = rohf_grad(&su, scf, exx, None);
        let err = max_fd_err(&g.grad, &fdv, k);
        eprintln!(
            "H4 ROHF triplet {exx:?}: max|analytic − FD| = {err:.2e} (prototype h² floor 9.7e-9), \
             |ΣF| {:.1e}, orbital gradient {:.1e}\n{:.10}",
            g.net_force, g.commutator, g.grad
        );
        assert!(err < FD_BAR, "{exx:?}: {err:e}");
        assert!(g.net_force < NET_FORCE_BAR, "ΣF {:e}", g.net_force);
        // |(f_α)_co| is small here (prototype 0.024 ROHF): RoWNoCo 2.5e-3.
        let e = max_fd_err(
            &rohf_grad(&su, scf, exx, Some(GradMutation::RoWNoCo)).grad,
            &fdv,
            k,
        );
        eprintln!("  mutant RoWNoCo: {e:.2e}");
        assert!(e > MUTANT_BAR, "RoWNoCo ({exx:?}) escaped: {e:e}");
    }
}

// ------------------------------------------------------ H3 ROHF stress (slow)

fn eps_at(i: usize, j: usize, h: f64) -> Mat3 {
    let mut e = [[0.0; 3]; 3];
    e[i][j] = h;
    e
}

fn max_err3(a: &Mat3, b: &Mat3) -> f64 {
    let mut m = 0.0_f64;
    for i in 0..3 {
        for j in 0..3 {
            let d = (a[i][j] - b[i][j]).abs();
            assert!(d.is_finite(), "non-finite stress difference");
            m = m.max(d);
        }
    }
    m
}

fn antisym3(a: &Mat3) -> f64 {
    let mut m = 0.0_f64;
    for i in 0..3 {
        for j in 0..3 {
            m = m.max((a[i][j] - a[j][i]).abs());
        }
    }
    m
}

/// ROHF stress (UHF-form W, which equals W_RO at convergence) vs central FD
/// (h = 1e-4) of the ROHF energy on strained cells (`Cell::strained`, index
/// sets frozen), all nine components, both exxdiv.
#[test]
#[ignore = "slow: H3 ROHF, 18 strained dense-AFT builds + 36 staged SCF stages; run with --release -- --ignored"]
fn h3_rohf_stress_matches_fd_all_nine_both_exxdiv() {
    let bs = pyscf_sto3g_h();
    let cell = h3_cell();
    let su = setup(cell.clone(), &bs);
    let (r0, none, ewald) = h3_rohf_ref(&su);
    let init = Some(none.mos_alpha.clone());
    let mut fdv = [[[0.0_f64; 3]; 3]; 2];
    for (i, j) in ALL9 {
        let mut e = [[0.0_f64; 2]; 2];
        for (si, sgn) in [1.0_f64, -1.0].into_iter().enumerate() {
            let c = cell.strained(&eps_at(i, j, sgn * FD_H)).expect("strained");
            let s = setup(c, &bs);
            let r = rohf_on(&s, &rohf_cfg(ExxDiv::Ewald, init.clone()));
            let rn = r.none_stage.expect("none stage");
            let cont = continuity(&rn, r0.nocc, &none.mos_alpha, &s.hc.s);
            assert!(cont < CONT_BAR, "strained H3 left the state: {cont:e}");
            e[si] = [rn.energy, r.scf.energy];
        }
        for k in 0..2 {
            fdv[k][i][j] = (e[0][k] - e[1][k]) / (2.0 * FD_H);
        }
    }
    for (k, (scf, exx)) in [(&none, ExxDiv::None), (&ewald, ExxDiv::Ewald)]
        .into_iter()
        .enumerate()
    {
        let st = gamma_rohf_stress(
            &su.cell,
            &su.prep,
            &hcore_cfg(),
            &su.hc,
            &su.eri,
            scf,
            exx,
            &GammaStressConfig::default(),
        )
        .expect("gamma_rohf_stress");
        let err = max_err3(&st.de_deps, &fdv[k]);
        eprintln!(
            "H3 ROHF stress {exx:?}: max|an − FD| = {err:.2e}, |an − anᵀ| {:.1e}\n{:?}",
            antisym3(&st.de_deps),
            st.de_deps
        );
        assert!(err < STRESS_FD_BAR_OPEN, "{exx:?}: {err:e}");
        assert!(antisym3(&st.de_deps) < STRESS_ANTISYM_BAR);
    }
}
