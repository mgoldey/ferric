//! Gamma-point periodic UHF / RKS / UKS analytic forces
//! (`ferric_pbc::grad::{gamma_uhf_gradient, gamma_rks_gradient,
//! gamma_uks_gradient}`), the Rust port of `reference/pbc/pbc_grad_open.py`
//! (FINDINGS "Iteration 17").
//!
//! What is anchored against what:
//! * EXACTNESS ANCHOR: analytic vs central finite difference of ferric's OWN
//!   Gamma energy (`periodic_hcore` + `DenseAftEri` + `gamma_uhf` /
//!   `gamma_rks` / `gamma_uks`), the XC grid REBUILT at each displaced
//!   geometry (points ride on their home atom, weights recomputed) — the
//!   derivative of the energy as defined, so it needs the full grid
//!   response. Grid (40, 50), SSF, D = 8 (the prototype's anchor grid).
//!   Prototype floors at h = 1e-4: H3 UHF 3.8e-9, H3 UKS LDA/PBE/PBE0
//!   3.3e-9..3.5e-9; the Rust RHF anchor adds the primitive screens and
//!   reaches 2.4e-9 on H2. Bar 1e-7 (as `pbc_grad.rs`).
//!   H2 a = 4 KS uses h = 5e-5: at h ≥ 1e-4 displaced points cross the hard
//!   `D` neighbour mask and the ENERGY jumps by 1e-11..1e-9 Ha (prototype
//!   h-scan: (1,y) +4.1e-8 at 1e-4, −2.2e-11 at 5e-5; (0,z) −5.8e-10 at
//!   5e-5, +2.05e-6 at 2e-4).
//! * Mutants (the hidden `GammaGradConfig::mutation` knob), each must miss FD
//!   by > 1e-3 (prototype, H3: smallest miss 5.7e-3 = `XcNoGga` on PBE0; the
//!   response pieces 1.4e-1..1.8e-1; spin mutants 9.2e-3..1.3e-1). The bar
//!   sits four orders above the passing residual and ~6x below the smallest
//!   measured miss.
//! * Identities on ONE density (no second SCF, so they test the formula,
//!   not SCF convergence): closed-shell UHF(D/2, D/2) ≡ RHF(D) and
//!   UKS(D/2, D/2) ≡ RKS(D) ≤ 1e-12 (prototype 0 / 1.1e-16); the three spin
//!   mutants are EXACTLY blind there (algebra; the UKS/RKS pair at 1e-11,
//!   see `XC_IDENTITY_BAR`); F(ewald) ≡ F(none) on an
//!   open-shell idempotent density ≤ 1e-11 (prototype 1.5e-12 across two
//!   SCFs), and the RHF-form Madelung mutant breaks it by > 1e-3 (prototype
//!   1.3e-1 UHF, 3.3e-2 PBE0).
//! * Translation invariance ΣF ≤ 1e-10 with the full response (exact by
//!   construction: `Σ_A (ao + point) = 0` per point and `Σ_A dW = 0`); the
//!   `NoPointMotion` and `XcNoAo` mutants break it (prototype 1e-4..3e-4) —
//!   a check on the image fold of the weight derivative, and the only test
//!   here that the sum of forces can pass or fail.
//!
//! Not covered (as in the prototype): FD with FROZEN points/weights (the
//! drivers always rebuild the grid; the response is instead pinned by the
//! two response mutants), the Becke scheme's full anchor (the weight
//! derivative is FD-checked for both schemes in ferric-dft's
//! `partition_weight_derivative_matches_fd`), meta-GGA, RSH, ROHF/ROKS,
//! RS-GDF J/K forces, k-points, stress.

mod common;

use common::*;
use ferric_core::basis::BasisSet;
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_pbc::dense_aft::{
    DenseAftEri, ExxDiv, DEFAULT_DENSE_AFT_MAX_BYTES, DEFAULT_DENSE_AFT_PRECISION,
};
use ferric_pbc::dft::{
    gamma_rks, gamma_uks, GammaRksConfig, GammaUksConfig, GammaUksResult, PeriodicGridConfig,
};
use ferric_pbc::grad::{
    gamma_rhf_gradient_with, gamma_rks_gradient_with, gamma_uhf_gradient_with,
    gamma_uks_gradient_with, GammaGradConfig, GammaGradient, GradMutation,
};
use ferric_pbc::hcore::{periodic_hcore, PeriodicHcore, PeriodicHcoreConfig};
use ferric_pbc::lattice::Cell;
use ferric_pbc::uhf::{gamma_uhf, GammaUhfConfig, GammaUhfIntegrals, GammaUhfResult};
use ferric_scf::result::{ScfResult, Spin};
use ndarray::Array2;
use std::sync::OnceLock;

/// ω of the nuclear-attraction split (as `pbc_grad.rs`).
const OMEGA: f64 = 0.8;
const HCORE_PRECISION: f64 = 1e-14;
const FD_H: f64 = 1e-4;
/// H2 a = 4 KS step (module doc: neighbour-mask jumps at h ≥ 1e-4).
const FD_H_H2_KS: f64 = 5e-5;
const FD_BAR: f64 = 1e-7;
const MUTANT_BAR: f64 = 1e-3;
const NET_FORCE_BAR: f64 = 1e-10;
const IDENTITY_BAR: f64 = 1e-12;
/// UKS(D/2, D/2) vs RKS(D): polarized vs unpolarized libxc kernels, and the
/// `ρ_σ > 1e-10` vs `ρ > 1e-10` potential gates differ in a thin tail shell
/// (the same asymmetry the energy path has; `pbc_uks.rs` asserts UKS ≡ RKS
/// energies at 1e-11).
const XC_IDENTITY_BAR: f64 = 1e-11;
const EWALD_NONE_BAR: f64 = 1e-11;

/// `run_grad_open_anchor.py` H2 (Bohr), a = 4, STO-3G.
const GRAD_H2_ATOMS: [[f64; 3]; 2] = [[0.3, 0.2, 0.1], [0.35, 0.12, 1.5]];
const H2_COMPS: [(usize, usize); 3] = [(0, 0), (0, 2), (1, 1)];
/// `run_grad_open_anchor.py` H3 (Bohr), a = 4.5, STO-3G, doublet (2, 1).
const H3_ATOMS: [[f64; 3]; 3] = [[0.3, 0.2, 0.1], [0.35, 0.12, 1.5], [1.6, 0.9, 0.7]];
const H3_A: f64 = 4.5;
const H3_COMPS: [(usize, usize); 3] = [(0, 0), (2, 1), (1, 2)];
/// `run_grad_oracle.py` TRI_MOVED (Bohr), s+p basis, triplet (3, 1).
const TRI_MOVED: [[f64; 3]; 4] = [
    [0.13, 0.25, 0.31],
    [0.02, 0.27, 1.66],
    [2.47, 2.41, 2.25],
    [3.52, 2.98, 2.71],
];
const TRI_COMPS: [(usize, usize); 2] = [(2, 1), (0, 2)];

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

type Mos = Option<(Array2<f64>, Array2<f64>)>;

fn mos_of(r: &ScfResult) -> Mos {
    Some((r.mos_alpha.clone(), r.mos_beta.clone().expect("beta MOs")))
}

fn uhf_cfg(exx: ExxDiv, init: Mos) -> GammaUhfConfig {
    GammaUhfConfig {
        exxdiv: exx,
        initial_mos: init,
        ..GammaUhfConfig::default()
    }
}

fn uks_cfg(xc: &str, exx: ExxDiv, init: Mos) -> GammaUksConfig {
    GammaUksConfig {
        grid: grid_cfg(),
        exxdiv: exx,
        initial_mos: init,
        ..GammaUksConfig::new(xc)
    }
}

fn rks_cfg(xc: &str, exx: ExxDiv) -> GammaRksConfig {
    GammaRksConfig {
        grid: grid_cfg(),
        exxdiv: exx,
        ..GammaRksConfig::new(xc)
    }
}

fn uhf(su: &Setup, cfg: &GammaUhfConfig) -> GammaUhfResult {
    let r = gamma_uhf(
        &su.cell,
        &su.prep,
        &su.hc,
        GammaUhfIntegrals::DenseAft(&su.eri),
        cfg,
    )
    .expect("gamma_uhf");
    assert!(r.scf.converged);
    r
}

fn uks(su: &Setup, cfg: &GammaUksConfig) -> GammaUksResult {
    let r = gamma_uks(
        &su.cell,
        &su.prep,
        &su.hc,
        GammaUhfIntegrals::DenseAft(&su.eri),
        cfg,
    )
    .unwrap_or_else(|e| panic!("gamma_uks {}: {e}", cfg.functional));
    assert!(r.scf.converged);
    r
}

fn rks(su: &Setup, cfg: &GammaRksConfig) -> ScfResult {
    gamma_rks(
        &su.cell,
        &su.prep,
        &su.hc,
        GammaUhfIntegrals::DenseAft(&su.eri),
        cfg,
    )
    .unwrap_or_else(|e| panic!("gamma_rks {}: {e}", cfg.functional))
    .scf
}

fn gcfg(mutation: Option<GradMutation>) -> GammaGradConfig {
    GammaGradConfig {
        mutation,
        ..Default::default()
    }
}

fn uhf_grad(su: &Setup, scf: &ScfResult, exx: ExxDiv, m: Option<GradMutation>) -> GammaGradient {
    gamma_uhf_gradient_with(
        &su.cell,
        &su.prep,
        &hcore_cfg(),
        &su.hc,
        &su.eri,
        scf,
        exx,
        &gcfg(m),
    )
    .expect("gamma_uhf_gradient")
}

fn uks_grad(
    su: &Setup,
    scf: &ScfResult,
    cfg: &GammaUksConfig,
    m: Option<GradMutation>,
) -> GammaGradient {
    gamma_uks_gradient_with(
        &su.cell,
        &su.prep,
        &hcore_cfg(),
        &su.hc,
        &su.eri,
        scf,
        cfg,
        &gcfg(m),
    )
    .expect("gamma_uks_gradient")
}

fn rks_grad(
    su: &Setup,
    scf: &ScfResult,
    cfg: &GammaRksConfig,
    m: Option<GradMutation>,
) -> GammaGradient {
    gamma_rks_gradient_with(
        &su.cell,
        &su.prep,
        &hcore_cfg(),
        &su.hc,
        &su.eri,
        scf,
        cfg,
        &gcfg(m),
    )
    .expect("gamma_rks_gradient")
}

/// The same closed-shell density as an unrestricted result `(D/2, D/2)`.
fn as_unrestricted(r: &ScfResult) -> ScfResult {
    let mut u = r.clone();
    let half = &r.density_total * 0.5;
    u.spin = Spin::Unrestricted;
    u.density_alpha = half.clone();
    u.density_beta = Some(half);
    u.mos_beta = Some(r.mos_alpha.clone());
    u.eps_beta = Some(r.eps_alpha.clone());
    u.fock_beta = Some(r.fock_alpha.clone());
    u
}

fn max_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

/// Central FD `(E(+h) − E(−h))/2h` of every energy `energies` returns, per
/// component.
fn fd<F>(
    pos: &[[f64; 3]],
    lattice: [[f64; 3]; 3],
    mult: usize,
    comps: &[(usize, usize)],
    h: f64,
    energies: F,
) -> Vec<((usize, usize), Vec<f64>)>
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

fn max_fd_err(g: &Array2<f64>, fdv: &[((usize, usize), Vec<f64>)], k: usize) -> f64 {
    fdv.iter()
        .map(|((a, x), v)| (g[(*a, *x)] - v[k]).abs())
        .fold(0.0_f64, f64::max)
}

fn h3_cell() -> Cell {
    cell_at(&H3_ATOMS, cubic(H3_A), 2)
}

// ------------------------------------------------------------------ H3 UHF

/// Reference H3 doublet UHF: the staged ewald run and its none stage.
fn h3_uhf_ref(su: &Setup) -> (ScfResult, ScfResult) {
    let r = uhf(su, &uhf_cfg(ExxDiv::Ewald, None));
    let none = r.none_stage.clone().expect("staged none stage");
    (none, r.scf)
}

static H3_UHF_FD: OnceLock<Vec<((usize, usize), Vec<f64>)>> = OnceLock::new();

/// FD `[none, ewald]` of the H3 UHF energy (one staged run per geometry,
/// seeded with the reference none-stage MOs so every geometry lands on the
/// same state).
fn h3_uhf_fd() -> &'static Vec<((usize, usize), Vec<f64>)> {
    H3_UHF_FD.get_or_init(|| {
        let bs = pyscf_sto3g_h();
        let (none, _) = h3_uhf_ref(&setup(h3_cell(), &bs));
        let init = mos_of(&none);
        fd(&H3_ATOMS, cubic(H3_A), 2, &H3_COMPS, FD_H, |c| {
            let r = uhf(
                &setup(c.clone(), &bs),
                &uhf_cfg(ExxDiv::Ewald, init.clone()),
            );
            vec![r.none_stage.expect("none stage").energy, r.scf.energy]
        })
    })
}

#[test]
fn h3_uhf_force_matches_fd_of_own_energy_both_exxdiv() {
    let su = setup(h3_cell(), &pyscf_sto3g_h());
    let (none, ewald) = h3_uhf_ref(&su);
    let fdv = h3_uhf_fd();
    for (k, (scf, exx)) in [(&none, ExxDiv::None), (&ewald, ExxDiv::Ewald)]
        .into_iter()
        .enumerate()
    {
        let g = uhf_grad(&su, scf, exx, None);
        let err = max_fd_err(&g.grad, fdv, k);
        eprintln!(
            "H3 UHF doublet {exx:?}: max|analytic − FD| = {err:.2e}, |ΣF| {:.1e}, comm {:.1e}\n{:.10}",
            g.net_force, g.commutator, g.grad
        );
        assert!(err < FD_BAR, "{exx:?}: {err:e}");
        assert!(g.net_force < NET_FORCE_BAR, "ΣF {:e}", g.net_force);
        assert!(g.e_xc.is_none() && g.n_grid_points == 0);
    }
}

#[test]
fn h3_uhf_fd_anchor_catches_each_spin_mutant() {
    let su = setup(h3_cell(), &pyscf_sto3g_h());
    let (none, ewald) = h3_uhf_ref(&su);
    let fdv = h3_uhf_fd();
    // (mutant, exxdiv): MadelungTotal is visible only with ewald (prototype
    // 1.3e-1); WSpinSum 1.3e-1 / 6.6e-2; ExchTotal 3.7e-2.
    for (m, exx) in [
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
        let g = uhf_grad(&su, scf, exx, Some(m));
        let err = max_fd_err(&g.grad, fdv, k);
        eprintln!(
            "H3 UHF mutant {m:?} ({exx:?}): max|analytic − FD| = {err:.2e}, |ΣF| {:.1e} \
             (translation invariance cannot see it)",
            g.net_force
        );
        assert!(
            err > MUTANT_BAR,
            "{m:?} ({exx:?}) escaped the FD anchor: {err:e}"
        );
    }
}

/// F(ewald) ≡ F(none) on ONE open-shell idempotent density (the per-spin
/// Madelung term cancels `−v_M D_σ` inside `W_σ` exactly), and the RHF-form
/// Madelung mutant breaks it.
#[test]
fn h3_uhf_ewald_and_none_forces_agree_on_one_density() {
    let su = setup(h3_cell(), &pyscf_sto3g_h());
    let (_, ewald) = h3_uhf_ref(&su);
    let gn = uhf_grad(&su, &ewald, ExxDiv::None, None);
    let ge = uhf_grad(&su, &ewald, ExxDiv::Ewald, None);
    let d = max_diff(&gn.grad, &ge.grad);
    eprintln!(
        "H3 UHF: |F_ewald − F_none| = {d:.2e} (v_M = {})",
        ge.madelung
    );
    assert!(ge.madelung > 0.1);
    assert!(d < EWALD_NONE_BAR, "{d:e}");
    let gm = uhf_grad(
        &su,
        &ewald,
        ExxDiv::Ewald,
        Some(GradMutation::MadelungTotal),
    );
    let dm = max_diff(&gn.grad, &gm.grad);
    eprintln!("H3 UHF MadelungTotal mutant: |F_ewald − F_none| = {dm:.2e}");
    assert!(dm > MUTANT_BAR, "{dm:e}");
}

// ------------------------------------------------------------------ H3 UKS

/// Reference H3 doublet UKS results: LDA and PBE (`α = 0`, so exxdiv does
/// not enter their energy), PBE0 none and PBE0 ewald (the none entry is the
/// none stage of the staged PBE0 ewald run).
fn h3_uks_refs(su: &Setup) -> Vec<(GammaUksConfig, ScfResult)> {
    let lda = uks(su, &uks_cfg("LDA", ExxDiv::Ewald, None)).scf;
    let pbe = uks(su, &uks_cfg("PBE", ExxDiv::Ewald, None)).scf;
    let pbe0 = uks(su, &uks_cfg("PBE0", ExxDiv::Ewald, None));
    let pbe0_none = pbe0.none_stage.clone().expect("staged PBE0 none stage");
    vec![
        (uks_cfg("LDA", ExxDiv::Ewald, None), lda),
        (uks_cfg("PBE", ExxDiv::Ewald, None), pbe),
        (uks_cfg("PBE0", ExxDiv::None, None), pbe0_none),
        (uks_cfg("PBE0", ExxDiv::Ewald, None), pbe0.scf),
    ]
}

static H3_UKS_FD: OnceLock<Vec<((usize, usize), Vec<f64>)>> = OnceLock::new();

/// FD of the four H3 UKS energies (grid rebuilt per geometry; seeded with
/// the reference MOs).
fn h3_uks_fd() -> &'static Vec<((usize, usize), Vec<f64>)> {
    H3_UKS_FD.get_or_init(|| {
        let bs = pyscf_sto3g_h();
        let refs = h3_uks_refs(&setup(h3_cell(), &bs));
        let seeds: Vec<Mos> = refs.iter().map(|(_, r)| mos_of(r)).collect();
        fd(&H3_ATOMS, cubic(H3_A), 2, &H3_COMPS, FD_H, |c| {
            let su = setup(c.clone(), &bs);
            let lda = uks(&su, &uks_cfg("LDA", ExxDiv::Ewald, seeds[0].clone()));
            let pbe = uks(&su, &uks_cfg("PBE", ExxDiv::Ewald, seeds[1].clone()));
            // The staged PBE0 run starts from the none-stage MOs.
            let pbe0 = uks(&su, &uks_cfg("PBE0", ExxDiv::Ewald, seeds[2].clone()));
            vec![
                lda.scf.energy,
                pbe.scf.energy,
                pbe0.none_stage.expect("none stage").energy,
                pbe0.scf.energy,
            ]
        })
    })
}

#[test]
fn h3_uks_force_matches_fd_of_own_energy_with_grid_response() {
    let su = setup(h3_cell(), &pyscf_sto3g_h());
    let fdv = h3_uks_fd();
    for (k, (cfg, scf)) in h3_uks_refs(&su).iter().enumerate() {
        let g = uks_grad(&su, scf, cfg, None);
        let err = max_fd_err(&g.grad, fdv, k);
        let resp = &g.parts.xc_point + &g.parts.xc_weight;
        let resp_max = resp.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        eprintln!(
            "H3 UKS {} {:?}: max|analytic − FD| = {err:.2e}, |ΣF| {:.1e}, comm {:.1e}, \
             npts {}, |grid response| {resp_max:.2e}\n{:.10}",
            cfg.functional, cfg.exxdiv, g.net_force, g.commutator, g.n_grid_points, g.grad
        );
        assert!(err < FD_BAR, "{} {:?}: {err:e}", cfg.functional, cfg.exxdiv);
        assert!(g.net_force < NET_FORCE_BAR, "ΣF {:e}", g.net_force);
        // The response is live (prototype ~2-3e-3 at (40, 50)): an anchor
        // that passed with it absent would not be testing it.
        assert!(resp_max > 1e-4, "grid response {resp_max:e}");
    }
}

#[test]
fn h3_uks_fd_anchor_catches_each_mutant() {
    let su = setup(h3_cell(), &pyscf_sto3g_h());
    let fdv = h3_uks_fd();
    let refs = h3_uks_refs(&su);
    for (k, (cfg, scf)) in refs.iter().enumerate() {
        let hybrid = cfg.functional == "PBE0";
        let gga = cfg.functional != "LDA";
        let mut muts = vec![
            GradMutation::XcNoAo,
            GradMutation::NoPointMotion,
            GradMutation::NoWeightDeriv,
            GradMutation::WSpinSum,
        ];
        if gga {
            muts.push(GradMutation::XcNoGga);
        }
        if hybrid {
            muts.push(GradMutation::ExchTotal);
            if cfg.exxdiv == ExxDiv::Ewald {
                muts.push(GradMutation::MadelungTotal);
            }
        }
        for m in muts {
            let g = uks_grad(&su, scf, cfg, Some(m));
            let err = max_fd_err(&g.grad, fdv, k);
            eprintln!(
                "H3 UKS {} {:?} mutant {m:?}: max|analytic − FD| = {err:.2e}, |ΣF| {:.1e}",
                cfg.functional, cfg.exxdiv, g.net_force
            );
            assert!(
                err > MUTANT_BAR,
                "{} {:?}: {m:?} escaped the FD anchor: {err:e}",
                cfg.functional,
                cfg.exxdiv
            );
            // Dropping either half of the AO-term bookkeeping breaks the
            // translation invariance the full response restores.
            if matches!(m, GradMutation::XcNoAo | GradMutation::NoPointMotion) {
                assert!(g.net_force > 1e-6, "{m:?}: ΣF {:e}", g.net_force);
            }
        }
    }
}

#[test]
fn h3_pbe0_ewald_and_none_forces_agree_on_one_density() {
    let su = setup(h3_cell(), &pyscf_sto3g_h());
    let r = uks(&su, &uks_cfg("PBE0", ExxDiv::Ewald, None));
    let gn = uks_grad(&su, &r.scf, &uks_cfg("PBE0", ExxDiv::None, None), None);
    let ge = uks_grad(&su, &r.scf, &uks_cfg("PBE0", ExxDiv::Ewald, None), None);
    let d = max_diff(&gn.grad, &ge.grad);
    eprintln!("H3 PBE0: |F_ewald − F_none| = {d:.2e}");
    assert!((ge.exact_exchange_fraction - 0.25).abs() < 1e-12);
    assert!(d < EWALD_NONE_BAR, "{d:e}");
    let gm = uks_grad(
        &su,
        &r.scf,
        &uks_cfg("PBE0", ExxDiv::Ewald, None),
        Some(GradMutation::MadelungTotal),
    );
    let dm = max_diff(&gn.grad, &gm.grad);
    eprintln!("H3 PBE0 MadelungTotal mutant: |F_ewald − F_none| = {dm:.2e}");
    assert!(dm > MUTANT_BAR, "{dm:e}");
}

// ---------------------------------------------------- closed-shell identities

/// UHF(D/2, D/2) ≡ RHF(D) on one density, and the three spin mutants are
/// exactly blind on it (their difference from the correct form is zero
/// algebraically for `D_α = D_β`, `F_α = F_β`).
#[test]
fn closed_shell_uhf_gradient_is_the_rhf_gradient() {
    let cell = cell_at(&GRAD_H2_ATOMS, cubic(4.0), 1);
    let su = setup(cell, &pyscf_sto3g_h());
    let eri = su.eri.clone().with_exxdiv(&su.cell, ExxDiv::Ewald).unwrap();
    let scf = gamma_rhf(&su.cell, &su.prep, &su.hc, &eri);
    let grhf = gamma_rhf_gradient_with(
        &su.cell,
        &su.prep,
        &hcore_cfg(),
        &su.hc,
        &su.eri,
        &scf,
        ExxDiv::Ewald,
        &GammaGradConfig::default(),
    )
    .unwrap();
    let u = as_unrestricted(&scf);
    let guhf = uhf_grad(&su, &u, ExxDiv::Ewald, None);
    let d = max_diff(&grhf.grad, &guhf.grad);
    eprintln!("H2: |F_UHF(D/2,D/2) − F_RHF(D)| = {d:.2e}");
    assert!(d < IDENTITY_BAR, "{d:e}");
    for m in [
        GradMutation::WSpinSum,
        GradMutation::MadelungTotal,
        GradMutation::ExchTotal,
    ] {
        let gm = uhf_grad(&su, &u, ExxDiv::Ewald, Some(m));
        let dm = max_diff(&gm.grad, &guhf.grad);
        assert!(
            dm < IDENTITY_BAR,
            "{m:?} should be blind on a closed shell: {dm:e}"
        );
    }
}

/// UKS(D/2, D/2) ≡ RKS(D) on one density (polarized vs unpolarized libxc
/// kernel and the per-spin bookkeeping), PBE and PBE0 with ewald.
#[test]
fn closed_shell_uks_gradient_is_the_rks_gradient() {
    let cell = cell_at(&GRAD_H2_ATOMS, cubic(4.0), 1);
    let su = setup(cell, &pyscf_sto3g_h());
    for xc in ["PBE", "PBE0"] {
        let rcfg = rks_cfg(xc, ExxDiv::Ewald);
        let scf = rks(&su, &rcfg);
        let grks = rks_grad(&su, &scf, &rcfg, None);
        let guks = uks_grad(
            &su,
            &as_unrestricted(&scf),
            &uks_cfg(xc, ExxDiv::Ewald, None),
            None,
        );
        let d = max_diff(&grks.grad, &guks.grad);
        let de = (grks.e_xc.unwrap() - guks.e_xc.unwrap()).abs();
        eprintln!("H2 {xc}: |F_UKS(D/2,D/2) − F_RKS(D)| = {d:.2e}, |ΔE_xc| {de:.2e}");
        assert!(d < XC_IDENTITY_BAR, "{xc}: {d:e}");
        assert!(grks.net_force < NET_FORCE_BAR, "ΣF {:e}", grks.net_force);
    }
}

// ------------------------------------------------------------ H2 RKS anchor

static H2_RKS_FD: OnceLock<Vec<((usize, usize), Vec<f64>)>> = OnceLock::new();

const H2_RKS_XCS: [&str; 2] = ["LDA", "PBE0"];

fn h2_rks_fd() -> &'static Vec<((usize, usize), Vec<f64>)> {
    H2_RKS_FD.get_or_init(|| {
        let bs = pyscf_sto3g_h();
        fd(&GRAD_H2_ATOMS, cubic(4.0), 1, &H2_COMPS, FD_H_H2_KS, |c| {
            let su = setup(c.clone(), &bs);
            H2_RKS_XCS
                .iter()
                .map(|xc| rks(&su, &rks_cfg(xc, ExxDiv::Ewald)).energy)
                .collect()
        })
    })
}

/// Closed-shell RKS on H2 a = 4 at h = 5e-5 (module doc), LDA and PBE0.
#[test]
fn h2_rks_force_matches_fd_of_own_energy() {
    let su = setup(cell_at(&GRAD_H2_ATOMS, cubic(4.0), 1), &pyscf_sto3g_h());
    let fdv = h2_rks_fd();
    for (k, xc) in H2_RKS_XCS.iter().enumerate() {
        let cfg = rks_cfg(xc, ExxDiv::Ewald);
        let scf = rks(&su, &cfg);
        let g = rks_grad(&su, &scf, &cfg, None);
        let err = max_fd_err(&g.grad, fdv, k);
        eprintln!(
            "H2 RKS {xc}: max|analytic − FD| = {err:.2e}, |ΣF| {:.1e}, comm {:.1e}\n{:.10}",
            g.net_force, g.commutator, g.grad
        );
        assert!(err < FD_BAR, "{xc}: {err:e}");
        assert!(g.net_force < NET_FORCE_BAR, "ΣF {:e}", g.net_force);
        // Response mutants on a closed shell (prototype H2: 1.1e-1 each).
        for m in [GradMutation::NoPointMotion, GradMutation::NoWeightDeriv] {
            let gm = rks_grad(&su, &scf, &cfg, Some(m));
            let e = max_fd_err(&gm.grad, fdv, k);
            eprintln!("  mutant {m:?}: {e:.2e}");
            assert!(e > MUTANT_BAR, "{xc} {m:?}: {e:e}");
        }
    }
}

#[test]
fn gradients_reject_the_wrong_spin_kind() {
    let su = setup(cell_at(&GRAD_H2_ATOMS, cubic(4.0), 1), &pyscf_sto3g_h());
    let scf = gamma_rhf(&su.cell, &su.prep, &su.hc, &su.eri);
    let err = gamma_uhf_gradient_with(
        &su.cell,
        &su.prep,
        &hcore_cfg(),
        &su.hc,
        &su.eri,
        &scf,
        ExxDiv::None,
        &GammaGradConfig::default(),
    )
    .expect_err("a restricted result must be refused by the UHF gradient");
    assert!(err.to_string().contains("unrestricted"), "{err}");
    let err = gamma_rks_gradient_with(
        &su.cell,
        &su.prep,
        &hcore_cfg(),
        &su.hc,
        &su.eri,
        &as_unrestricted(&scf),
        &rks_cfg("PBE", ExxDiv::None),
        &GammaGradConfig::default(),
    )
    .expect_err("an unrestricted result must be refused by the RKS gradient");
    assert!(err.to_string().contains("restricted"), "{err}");
}

// ---------------------------------------------------- triclinic s+p triplet

/// Richardson-extrapolated central FD from steps `h` and `h/2`: removes the
/// `h² f'''/6` truncation, which is large on this state (prototype frozen-
/// grid h-scan, PBE (0,z): −5.7e-8 / −2.27e-7 at h = 5e-5 / 1e-4).
fn richardson(
    fd_h: &[((usize, usize), Vec<f64>)],
    fd_h2: &[((usize, usize), Vec<f64>)],
) -> Vec<((usize, usize), Vec<f64>)> {
    fd_h.iter()
        .zip(fd_h2)
        .map(|((c, a), (_, b))| {
            (
                *c,
                a.iter().zip(b).map(|(x, y)| (4.0 * y - x) / 3.0).collect(),
            )
        })
        .collect()
}

#[test]
#[ignore = "slow: triclinic 4H s+p triplet, 8 displaced dense-AFT builds + grids + 16 SCFs; run with --release -- --ignored"]
fn triclinic_sp_triplet_uhf_and_uks_force_match_fd() {
    let bs = sp_basis_h();
    let su = setup(cell_at(&TRI_MOVED, TRI_A, 3), &bs);
    let r_uhf = uhf(&su, &uhf_cfg(ExxDiv::None, None)).scf;
    let pbe_cfg = uks_cfg("PBE", ExxDiv::None, None);
    let r_pbe = uks(&su, &pbe_cfg).scf;
    let seeds = [mos_of(&r_uhf), mos_of(&r_pbe)];
    let energies = |c: &Cell| {
        let s = setup(c.clone(), &bs);
        vec![
            uhf(&s, &uhf_cfg(ExxDiv::None, seeds[0].clone())).scf.energy,
            uks(&s, &uks_cfg("PBE", ExxDiv::None, seeds[1].clone()))
                .scf
                .energy,
        ]
    };
    let f1 = fd(&TRI_MOVED, TRI_A, 3, &TRI_COMPS, FD_H, &energies);
    let f2 = fd(&TRI_MOVED, TRI_A, 3, &TRI_COMPS, 0.5 * FD_H, &energies);
    let fdv = richardson(&f1, &f2);
    let g_uhf = uhf_grad(&su, &r_uhf, ExxDiv::None, None);
    let g_pbe = uks_grad(&su, &r_pbe, &pbe_cfg, None);
    for (k, (name, g)) in [("UHF", &g_uhf), ("UKS PBE", &g_pbe)]
        .into_iter()
        .enumerate()
    {
        let err = max_fd_err(&g.grad, &fdv, k);
        let err_h = max_fd_err(&g.grad, &f1, k);
        eprintln!(
            "tri s+p triplet {name}: max|analytic − FD_Richardson| = {err:.2e} \
             (plain h = 1e-4: {err_h:.2e}), |ΣF| {:.1e}\n{:.10}",
            g.net_force, g.grad
        );
        // p shells: the libint 3-centre derivative floor measured on the RHF
        // triclinic anchor is 1.85e-7 (pbc_grad.rs); bar from that floor.
        assert!(err < 5e-7, "{name}: {err:e}");
        assert!(g.net_force < NET_FORCE_BAR, "ΣF {:e}", g.net_force);
    }
}
