//! Gamma-point analytic forces with RS-GDF J/K
//! (`ferric_pbc::grad::gamma_{rhf,uhf,rks,uks}_gradient_rsgdf`), the Rust
//! port of `reference/pbc/pbc_grad_gdf.py` (FINDINGS "Iteration 18").
//!
//! What is anchored against what:
//! * EXACTNESS ANCHOR: analytic vs central FD (h = 1e-4) of ferric's OWN
//!   RS-GDF energy (`periodic_hcore` + `RsGdf::build` rebuilt at every
//!   displaced geometry with the same config + the SCF on its J/K), both
//!   exxdiv. Prototype floors (pure-AFT h): H2 ET-40 8.4e-11..2.42e-9, H3 UHF
//!   2.7e-9, H3 UKS PBE0 2.2e-9, tri RHF 2.7e-9 / UHF triplet 8.8e-10 — the
//!   dense-AFT floors of Iterations 16/17. The Rust dense anchors add the
//!   primitive screens and use bar 1e-7 (`pbc_grad.rs`); the same bar here.
//! * Mutants (`GradMutation::Fit*`), prototype max|analytic − FD|:
//!   | mutant | H2 RHF | H3 UHF | H3 UKS PBE0 | tri RHF | tri UHF |
//!   | FitNoMetric | 1.78e-2 | 1.22e-1 | 1.02e-1 | 7.04e-2 | 2.54e-1 |
//!   | FitNoAux | 1.78e-2 | 1.22e-1 | 1.02e-1 | 7.05e-2 | 2.55e-1 |
//!   | FitNoG0 | 3.50e-3 | 2.60e-2 | 3.22e-2 | 2.23e-2 | 9.90e-3 |
//!   | FitG0Dense | 7.39e-4 | 1.21e-3 | 1.58e-3 | 1.03e-2 | 4.99e-3 |
//!   | FitNoLr3 | 2.00e-2 | 1.16e-1 | 7.41e-2 | 9.07e-2 | 1.97e-1 |
//!   Bar `MUTANT_BAR = 1e-4`: ~7x below the smallest miss (H2 FitG0Dense)
//!   and three decades above the FD bar.
//! * ΣF ≤ 1e-10 (prototype ≤ 1.2e-13); blind to every fit mutant except
//!   `FitNoAux` (prototype H3 5.8e-5, H2 1.2e-13 — H2 is blind by symmetry,
//!   so the ΣF check on it is asserted on H3 only).
//! * F(ewald) ≡ F(none) on ONE open-shell density ≤ 1e-11; UHF(D/2, D/2) ≡
//!   RHF(D) ≤ 1e-12 and UKS(D/2, D/2) ≡ RKS(D) ≤ 1e-11 on one density.
//! * INDEPENDENT CONSTRUCTION: exact aux span (one s α = 0.5 per H, a = 4,
//!   24 s aux of exponent 1 on the 8 half-lattice classes of the 3 pair
//!   types, ghost `SiteBasis` sites moving with the atoms through `aux_jac`)
//!   ⇒ F_gdf ≡ F_dense-AFT (prototype 4.1e-12, bar 1e-10). Every fit mutant
//!   fails it (prototype 1.0e-2..4.3e-2) EXCEPT `FitG0Dense` (3.6e-9: blind
//!   in the span limit by construction — the reason the incomplete-aux FD
//!   tests carry it).
//! * EIGENVALUE CUT: with nothing dropped the Loewner and textbook metric
//!   terms are bitwise equal (trivial-limit anchor). With an ACTIVE cut (H2
//!   ET-40, lindep 3e-3: prototype 4 dropped, E shift −9.6e-7,
//!   max|F_dk − F_std| 2.9e-7, dk − FD −2.42e-9, std − FD −1.48e-7 / 4.2e-8)
//!   the Loewner form matches FD and the textbook form does not; the
//!   dropped count must be the same at ±h (else E(R) is discontinuous and
//!   the FD meaningless).
//!
//! h here is the Ewald-split `PeriodicHcore` (c0_h ≠ 0), not the prototype's
//! pure-AFT h; its G = 0 M-term composes additively (Iteration 16), so the
//! numbers above are the prototype's and the Rust residuals are to be
//! recorded when first run.
//!
//! NOT covered: k-points, stress, ROHF/ROKS, RSH, aux l > 2, non-H atoms,
//! PySCF RSDF fixed-D oracle (PySCF has no GDF forces; the prototype's
//! fixed-D FD of PySCF's E2 is not ported).

mod common;

use common::*;
use ferric_core::basis::{self, BasisSet, Shell};
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::site_basis::SiteBasis;
use ferric_pbc::dense_aft::{
    DenseAftEri, ExxDiv, DEFAULT_DENSE_AFT_MAX_BYTES, DEFAULT_DENSE_AFT_PRECISION,
};
use ferric_pbc::dft::{
    gamma_rks, gamma_uks, GammaRksConfig, GammaUksConfig, GammaUksResult, PeriodicGridConfig,
};
use ferric_pbc::grad::{
    gamma_rhf_gradient_rsgdf, gamma_rhf_gradient_with, gamma_rks_gradient_rsgdf,
    gamma_uhf_gradient_rsgdf, gamma_uks_gradient_rsgdf, GammaGradConfig, GammaGradient,
    GradMutation, RsGdfGradSource,
};
use ferric_pbc::hcore::{periodic_hcore, PeriodicHcore, PeriodicHcoreConfig};
use ferric_pbc::lattice::Cell;
use ferric_pbc::rsgdf::{RsGdf, RsGdfConfig, DEFAULT_RSGDF_LINDEP};
use ferric_pbc::uhf::{gamma_uhf, GammaUhfConfig, GammaUhfIntegrals, GammaUhfResult};
use ferric_scf::result::{ScfResult, Spin};
use ndarray::Array2;
use std::collections::HashMap;
use std::sync::OnceLock;

/// ω of the nuclear-attraction split (as `pbc_grad.rs`).
const OMEGA: f64 = 0.8;
const HCORE_PRECISION: f64 = 1e-14;
/// RS-GDF split (the prototype's w = 1; the span anchor uses 1.2 so a slip
/// between h's c0 and the fit's c0' is visible).
const GDF_OMEGA: f64 = 1.0;
const AMPLE: usize = 1 << 31;
const FD_H: f64 = 1e-4;
const FD_BAR: f64 = 1e-7;
const MUTANT_BAR: f64 = 1e-4;
const NET_FORCE_BAR: f64 = 1e-10;
const EWALD_NONE_BAR: f64 = 1e-11;
const IDENTITY_BAR: f64 = 1e-12;
const XC_IDENTITY_BAR: f64 = 1e-11;
const SPAN_BAR: f64 = 1e-10;
/// Active-cut bars (module doc): prototype dk − FD 2.42e-9, std − FD
/// 1.48e-7, |F_dk − F_std| 2.9e-7.
const CUT_LINDEP: f64 = 3e-3;
const CUT_DK_BAR: f64 = 3e-8;
const CUT_STD_MIN: f64 = 7e-8;
const CUT_CROSS_MIN: f64 = 1e-7;

/// `run_grad_gdf_anchor.py` H2 (Bohr), a = 4.
const H2_ATOMS_G: [[f64; 3]; 2] = [[0.3, 0.2, 0.1], [0.35, 0.12, 1.5]];
const H2_COMPS: [(usize, usize); 3] = [(0, 0), (0, 2), (1, 1)];
/// H3 (Bohr), a = 4.5, doublet.
const H3_ATOMS: [[f64; 3]; 3] = [[0.3, 0.2, 0.1], [0.35, 0.12, 1.5], [1.6, 0.9, 0.7]];
const H3_A: f64 = 4.5;
const H3_COMPS: [(usize, usize); 3] = [(0, 0), (2, 1), (1, 2)];
/// `run_grad_oracle.py` TRI_MOVED (Bohr), s+p basis.
const TRI_MOVED: [[f64; 3]; 4] = [
    [0.13, 0.25, 0.31],
    [0.02, 0.27, 1.66],
    [2.47, 2.41, 2.25],
    [3.52, 2.98, 2.71],
];
const TRI_COMPS: [(usize, usize); 2] = [(2, 1), (0, 2)];

const FIT_MUTANTS: [GradMutation; 5] = [
    GradMutation::FitNoMetric,
    GradMutation::FitNoAux,
    GradMutation::FitNoG0,
    GradMutation::FitG0Dense,
    GradMutation::FitNoLr3,
];

fn hcore_cfg() -> PeriodicHcoreConfig {
    PeriodicHcoreConfig {
        precision: HCORE_PRECISION,
        ..PeriodicHcoreConfig::with_omega(OMEGA)
    }
}

fn gdf_cfg(omega: f64, lindep: f64) -> RsGdfConfig {
    RsGdfConfig {
        omega,
        lindep,
        exxdiv: ExxDiv::None,
        budget_bytes: Some(AMPLE),
        ..Default::default()
    }
}

/// The prototype's anchor grid: (40, 50), SSF, D = 8.
fn grid_cfg() -> PeriodicGridConfig {
    PeriodicGridConfig {
        neighbour_cutoff: Some(8.0),
        ..PeriodicGridConfig::with_size(40, 50)
    }
}

/// Even-tempered s + Cartesian p aux on H: `a0 β^k`, k < n (the prototype's
/// "ET l≤1" sets; one unit-normalised primitive per shell).
fn et_aux(a0: f64, beta: f64, n: usize) -> BasisSet {
    let mut shells = Vec::new();
    for k in 0..n {
        let a = a0 * beta.powi(k as i32);
        for l in [0, 1] {
            shells.push(Shell {
                l,
                pure: false,
                exponents: vec![a],
                coefficients: vec![1.0],
            });
        }
    }
    let mut m = HashMap::new();
    m.insert(1, shells);
    BasisSet {
        name: "et-aux-H".into(),
        shells: m,
        ecps: HashMap::new(),
    }
}

/// H2 ET l≤1 a0 0.3, β 2.5, 5 exponents: 40 aux (prototype s_min 3.9e-5).
fn et40() -> BasisSet {
    et_aux(0.3, 2.5, 5)
}

fn cc_pvdz_ri() -> BasisSet {
    basis::bundled("cc-pvdz-ri").expect("cc-pvdz-ri")
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
    aux: PreparedBasis,
    /// `build_for_gradient`, exxdiv none (the drivers / K builders take the
    /// Madelung shift from their own config).
    gdf: RsGdf,
}

fn setup(cell: Cell, bs: &BasisSet, aux_bs: &BasisSet, lindep: f64) -> Setup {
    let prep = prep_for(&cell, bs);
    let hc = periodic_hcore(&cell, &prep, &hcore_cfg()).expect("hcore");
    let aux = PreparedBasis::new(cell.mol(), aux_bs).expect("aux prep");
    let gdf = RsGdf::build_for_gradient(&cell, &prep, &aux, &hc.s, &gdf_cfg(GDF_OMEGA, lindep))
        .expect("RsGdf::build_for_gradient");
    Setup {
        cell,
        prep,
        hc,
        aux,
        gdf,
    }
}

/// Energy-only build (what the FD differentiates): `RsGdf::build`.
fn energy_gdf(
    cell: &Cell,
    prep: &PreparedBasis,
    hc: &PeriodicHcore,
    aux_bs: &BasisSet,
    lindep: f64,
) -> (PreparedBasis, RsGdf) {
    let aux = PreparedBasis::new(cell.mol(), aux_bs).expect("aux prep");
    let gdf = RsGdf::build(cell, prep, &aux, &hc.s, &gdf_cfg(GDF_OMEGA, lindep)).expect("RsGdf");
    (aux, gdf)
}

fn rhf_on(
    cell: &Cell,
    prep: &PreparedBasis,
    hc: &PeriodicHcore,
    gdf: &RsGdf,
    exx: ExxDiv,
) -> ScfResult {
    let g = gdf.clone().with_exxdiv(cell, exx).expect("with_exxdiv");
    gamma_rhf_jk(
        cell,
        prep,
        hc,
        Box::new(g.j_builder()),
        Box::new(g.k_builder()),
    )
}

fn src(su: &Setup) -> RsGdfGradSource<'_> {
    RsGdfGradSource {
        gdf: &su.gdf,
        aux: &su.aux,
        aux_jac: None,
    }
}

fn gcfg(mutation: Option<GradMutation>) -> GammaGradConfig {
    GammaGradConfig {
        mutation,
        budget_bytes: Some(AMPLE),
        ..Default::default()
    }
}

fn rhf_grad(su: &Setup, scf: &ScfResult, exx: ExxDiv, m: Option<GradMutation>) -> GammaGradient {
    gamma_rhf_gradient_rsgdf(
        &su.cell,
        &su.prep,
        &hcore_cfg(),
        &su.hc,
        &src(su),
        scf,
        exx,
        &gcfg(m),
    )
    .expect("gamma_rhf_gradient_rsgdf")
}

fn uhf_grad(su: &Setup, scf: &ScfResult, exx: ExxDiv, m: Option<GradMutation>) -> GammaGradient {
    gamma_uhf_gradient_rsgdf(
        &su.cell,
        &su.prep,
        &hcore_cfg(),
        &su.hc,
        &src(su),
        scf,
        exx,
        &gcfg(m),
    )
    .expect("gamma_uhf_gradient_rsgdf")
}

fn uks_grad(
    su: &Setup,
    scf: &ScfResult,
    cfg: &GammaUksConfig,
    m: Option<GradMutation>,
) -> GammaGradient {
    gamma_uks_gradient_rsgdf(
        &su.cell,
        &su.prep,
        &hcore_cfg(),
        &su.hc,
        &src(su),
        scf,
        cfg,
        &gcfg(m),
    )
    .expect("gamma_uks_gradient_rsgdf")
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

fn uhf_on(
    cell: &Cell,
    prep: &PreparedBasis,
    hc: &PeriodicHcore,
    gdf: &RsGdf,
    cfg: &GammaUhfConfig,
) -> GammaUhfResult {
    let r = gamma_uhf(cell, prep, hc, GammaUhfIntegrals::RsGdf(gdf), cfg).expect("gamma_uhf");
    assert!(r.scf.converged);
    r
}

fn uks_on(
    cell: &Cell,
    prep: &PreparedBasis,
    hc: &PeriodicHcore,
    gdf: &RsGdf,
    cfg: &GammaUksConfig,
) -> GammaUksResult {
    let r = gamma_uks(cell, prep, hc, GammaUhfIntegrals::RsGdf(gdf), cfg)
        .unwrap_or_else(|e| panic!("gamma_uks {}: {e}", cfg.functional));
    assert!(r.scf.converged);
    r
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

type Fd = Vec<((usize, usize), Vec<f64>)>;

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

fn report(tag: &str, g: &GammaGradient, err: f64) {
    let f = g.fit.as_ref().expect("RS-GDF diagnostics");
    eprintln!(
        "{tag}: max|analytic − FD| = {err:.2e}, |ΣF| {:.1e}, comm {:.1e}; naux {} dropped {} \
         (s_kept_min {:.2e}, s_drop_max {:?}, cross {:.1e}); SR3 {} SR2 {} half-G {}\n\
         parts: orb_sr {:.3e} orb_lr {:.3e} aux_sr {:.3e} aux_lr {:.3e} met_sr {:.3e} met_lr {:.3e} g0 {:.3e}\n{:.10}",
        g.net_force,
        g.commutator,
        f.naux,
        f.n_dropped,
        f.s_kept_min,
        f.s_dropped_max,
        f.cross_norm,
        f.n_sr3_deriv,
        f.n_sr2_deriv,
        f.n_g_half,
        amax(&g.parts.fit_orb_sr),
        amax(&g.parts.fit_orb_lr),
        amax(&g.parts.fit_aux_sr),
        amax(&g.parts.fit_aux_lr),
        amax(&g.parts.fit_metric_sr),
        amax(&g.parts.fit_metric_lr),
        amax(&g.parts.fit_g0),
        g.grad
    );
}

fn amax(a: &Array2<f64>) -> f64 {
    a.iter().fold(0.0_f64, |m, v| m.max(v.abs()))
}

// ------------------------------------------------------------ build identity

/// `build_for_gradient` changes no energy: B, W and the stats are bitwise
/// those of `build`.
#[test]
fn build_for_gradient_is_bitwise_build() {
    let cell = cell_at(&H2_ATOMS_G, cubic(4.0), 1);
    let prep = prep_for(&cell, &pyscf_sto3g_h());
    let hc = periodic_hcore(&cell, &prep, &hcore_cfg()).unwrap();
    for (aux_bs, lindep) in [(cc_pvdz_ri(), DEFAULT_RSGDF_LINDEP), (et40(), CUT_LINDEP)] {
        let aux = PreparedBasis::new(cell.mol(), &aux_bs).unwrap();
        let cfg = gdf_cfg(GDF_OMEGA, lindep);
        let a = RsGdf::build(&cell, &prep, &aux, &hc.s, &cfg).unwrap();
        let b = RsGdf::build_for_gradient(&cell, &prep, &aux, &hc.s, &cfg).unwrap();
        assert!(!a.has_gradient_parts() && b.has_gradient_parts());
        assert_eq!(a.b().dim(), b.b().dim());
        for (x, y) in a.b().iter().zip(b.b().iter()) {
            assert_eq!(x.to_bits(), y.to_bits());
        }
        for (x, y) in a.metric_inv_sqrt().iter().zip(b.metric_inv_sqrt().iter()) {
            assert_eq!(x.to_bits(), y.to_bits());
        }
        assert_eq!(a.stats().n_dropped, b.stats().n_dropped);
        eprintln!(
            "{} lindep {lindep:e}: naux {} dropped {}",
            aux_bs.name,
            a.stats().naux,
            a.stats().n_dropped
        );
    }
}

// ------------------------------------------------------------------ H2 RHF

static H2_FD: OnceLock<Fd> = OnceLock::new();

/// FD `[none, ewald]` of the H2 ET-40 RS-GDF RHF energy.
fn h2_fd() -> &'static Fd {
    H2_FD.get_or_init(|| {
        let bs = pyscf_sto3g_h();
        let aux_bs = et40();
        fd(&H2_ATOMS_G, cubic(4.0), 1, &H2_COMPS, FD_H, |c| {
            let prep = prep_for(c, &bs);
            let hc = periodic_hcore(c, &prep, &hcore_cfg()).unwrap();
            let (_, gdf) = energy_gdf(c, &prep, &hc, &aux_bs, DEFAULT_RSGDF_LINDEP);
            [ExxDiv::None, ExxDiv::Ewald]
                .iter()
                .map(|&e| rhf_on(c, &prep, &hc, &gdf, e).energy)
                .collect()
        })
    })
}

fn h2_setup() -> Setup {
    setup(
        cell_at(&H2_ATOMS_G, cubic(4.0), 1),
        &pyscf_sto3g_h(),
        &et40(),
        DEFAULT_RSGDF_LINDEP,
    )
}

#[test]
fn h2_rhf_force_matches_fd_of_own_rsgdf_energy() {
    let su = h2_setup();
    let fdv = h2_fd();
    for (k, exx) in [ExxDiv::None, ExxDiv::Ewald].into_iter().enumerate() {
        let scf = rhf_on(&su.cell, &su.prep, &su.hc, &su.gdf, exx);
        let g = rhf_grad(&su, &scf, exx, None);
        let err = max_fd_err(&g.grad, fdv, k);
        report(&format!("H2 ET-40 RHF {exx:?}"), &g, err);
        assert!(err < FD_BAR, "{exx:?}: {err:e}");
        assert!(g.net_force < NET_FORCE_BAR, "ΣF {:e}", g.net_force);
        // The aux-motion and metric pieces are each O(1e-2) and cancel to
        // the fit error: an anchor that passed with them absent would not
        // test them.
        assert!(amax(&g.parts.fit_aux_sr) + amax(&g.parts.fit_aux_lr) > 1e-3);
        assert!(amax(&g.parts.fit_metric_sr) + amax(&g.parts.fit_metric_lr) > 1e-3);
        assert!(
            amax(&g.parts.eri) == 0.0,
            "dense-AFT ERI term on the fit path"
        );
        // Nothing is dropped at the default cut (prototype s_min 3.9e-5):
        // the Loewner and textbook metric terms must then be IDENTICAL.
        let f = g.fit.as_ref().unwrap();
        assert_eq!(f.n_dropped, 0, "ET-40 should drop nothing at 1e-10");
        let gt = rhf_grad(&su, &scf, exx, Some(GradMutation::FitTextbookMetric));
        assert_eq!(max_diff(&g.grad, &gt.grad), 0.0);
    }
}

#[test]
fn h2_fd_anchor_catches_each_fit_mutant() {
    let su = h2_setup();
    let fdv = h2_fd();
    let scf = rhf_on(&su.cell, &su.prep, &su.hc, &su.gdf, ExxDiv::None);
    for m in FIT_MUTANTS {
        let g = rhf_grad(&su, &scf, ExxDiv::None, Some(m));
        let err = max_fd_err(&g.grad, fdv, 0);
        eprintln!(
            "H2 RHF mutant {m:?}: max|analytic − FD| = {err:.2e}, |ΣF| {:.1e}",
            g.net_force
        );
        assert!(err > MUTANT_BAR, "{m:?} escaped the FD anchor: {err:e}");
    }
}

/// UHF(D/2, D/2) ≡ RHF(D) on one density (fit path).
#[test]
fn closed_shell_uhf_rsgdf_gradient_is_the_rhf_gradient() {
    let su = h2_setup();
    let scf = rhf_on(&su.cell, &su.prep, &su.hc, &su.gdf, ExxDiv::Ewald);
    let gr = rhf_grad(&su, &scf, ExxDiv::Ewald, None);
    let gu = uhf_grad(&su, &as_unrestricted(&scf), ExxDiv::Ewald, None);
    let d = max_diff(&gr.grad, &gu.grad);
    eprintln!("H2 RS-GDF: |F_UHF(D/2,D/2) − F_RHF(D)| = {d:.2e}");
    assert!(d < IDENTITY_BAR, "{d:e}");
}

/// UKS(D/2, D/2) ≡ RKS(D) on one density, PBE0 ewald (fit path; exercises
/// the RKS entry point).
#[test]
fn closed_shell_uks_rsgdf_gradient_is_the_rks_gradient() {
    let su = h2_setup();
    let rcfg = GammaRksConfig {
        grid: grid_cfg(),
        exxdiv: ExxDiv::Ewald,
        ..GammaRksConfig::new("PBE0")
    };
    let scf = gamma_rks(
        &su.cell,
        &su.prep,
        &su.hc,
        GammaUhfIntegrals::RsGdf(&su.gdf),
        &rcfg,
    )
    .expect("gamma_rks")
    .scf;
    let grks = gamma_rks_gradient_rsgdf(
        &su.cell,
        &su.prep,
        &hcore_cfg(),
        &su.hc,
        &src(&su),
        &scf,
        &rcfg,
        &gcfg(None),
    )
    .expect("gamma_rks_gradient_rsgdf");
    let guks = uks_grad(
        &su,
        &as_unrestricted(&scf),
        &uks_cfg("PBE0", ExxDiv::Ewald, None),
        None,
    );
    let d = max_diff(&grks.grad, &guks.grad);
    eprintln!("H2 RS-GDF PBE0: |F_UKS(D/2,D/2) − F_RKS(D)| = {d:.2e}");
    assert!(d < XC_IDENTITY_BAR, "{d:e}");
    assert!(grks.net_force < NET_FORCE_BAR, "ΣF {:e}", grks.net_force);
}

// ------------------------------------------------------------------ H3 UHF

fn h3_setup() -> Setup {
    setup(
        cell_at(&H3_ATOMS, cubic(H3_A), 2),
        &pyscf_sto3g_h(),
        &cc_pvdz_ri(),
        DEFAULT_RSGDF_LINDEP,
    )
}

/// Reference H3 doublet UHF: `(none stage, ewald)`.
fn h3_uhf_ref(su: &Setup) -> (ScfResult, ScfResult) {
    let r = uhf_on(
        &su.cell,
        &su.prep,
        &su.hc,
        &su.gdf,
        &uhf_cfg(ExxDiv::Ewald, None),
    );
    (r.none_stage.clone().expect("none stage"), r.scf)
}

static H3_UHF_FD: OnceLock<Fd> = OnceLock::new();

fn h3_uhf_fd() -> &'static Fd {
    H3_UHF_FD.get_or_init(|| {
        let bs = pyscf_sto3g_h();
        let aux_bs = cc_pvdz_ri();
        let (none, _) = h3_uhf_ref(&h3_setup());
        let init = mos_of(&none);
        fd(&H3_ATOMS, cubic(H3_A), 2, &H3_COMPS, FD_H, |c| {
            let prep = prep_for(c, &bs);
            let hc = periodic_hcore(c, &prep, &hcore_cfg()).unwrap();
            let (_, gdf) = energy_gdf(c, &prep, &hc, &aux_bs, DEFAULT_RSGDF_LINDEP);
            let r = uhf_on(c, &prep, &hc, &gdf, &uhf_cfg(ExxDiv::Ewald, init.clone()));
            vec![r.none_stage.expect("none stage").energy, r.scf.energy]
        })
    })
}

#[test]
fn h3_uhf_force_matches_fd_and_catches_each_fit_mutant() {
    let su = h3_setup();
    let (none, ewald) = h3_uhf_ref(&su);
    let fdv = h3_uhf_fd();
    for (k, (scf, exx)) in [(&none, ExxDiv::None), (&ewald, ExxDiv::Ewald)]
        .into_iter()
        .enumerate()
    {
        let g = uhf_grad(&su, scf, exx, None);
        let err = max_fd_err(&g.grad, fdv, k);
        report(&format!("H3 UHF doublet cc-pvdz-ri {exx:?}"), &g, err);
        assert!(err < FD_BAR, "{exx:?}: {err:e}");
        assert!(g.net_force < NET_FORCE_BAR, "ΣF {:e}", g.net_force);
        for m in FIT_MUTANTS {
            let gm = uhf_grad(&su, scf, exx, Some(m));
            let e = max_fd_err(&gm.grad, fdv, k);
            eprintln!(
                "  mutant {m:?} ({exx:?}): {e:.2e}, |ΣF| {:.1e}",
                gm.net_force
            );
            assert!(
                e > MUTANT_BAR,
                "{m:?} ({exx:?}) escaped the FD anchor: {e:e}"
            );
            // Only the aux-motion mutant breaks translation invariance
            // (prototype 5.8e-5); the others keep ΣF at the floor.
            if m == GradMutation::FitNoAux {
                assert!(gm.net_force > 1e-6, "{m:?}: ΣF {:e}", gm.net_force);
            } else {
                assert!(gm.net_force < NET_FORCE_BAR, "{m:?}: ΣF {:e}", gm.net_force);
            }
        }
    }
}

/// F(ewald) ≡ F(none) on ONE open-shell density (fit path).
#[test]
fn h3_uhf_rsgdf_ewald_and_none_forces_agree_on_one_density() {
    let su = h3_setup();
    let (_, ewald) = h3_uhf_ref(&su);
    let gn = uhf_grad(&su, &ewald, ExxDiv::None, None);
    let ge = uhf_grad(&su, &ewald, ExxDiv::Ewald, None);
    let d = max_diff(&gn.grad, &ge.grad);
    eprintln!(
        "H3 UHF RS-GDF: |F_ewald − F_none| = {d:.2e} (v_M = {})",
        ge.madelung
    );
    assert!(ge.madelung > 0.1);
    assert!(d < EWALD_NONE_BAR, "{d:e}");
}

// ------------------------------------------------------------ H3 UKS PBE0

static H3_UKS_FD: OnceLock<Fd> = OnceLock::new();

fn h3_pbe0_ref(su: &Setup) -> GammaUksResult {
    uks_on(
        &su.cell,
        &su.prep,
        &su.hc,
        &su.gdf,
        &uks_cfg("PBE0", ExxDiv::Ewald, None),
    )
}

/// FD `[none stage, ewald]` of the H3 UKS PBE0 energy (grid rebuilt per
/// geometry, seeded with the reference none-stage MOs).
fn h3_uks_fd() -> &'static Fd {
    H3_UKS_FD.get_or_init(|| {
        let bs = pyscf_sto3g_h();
        let aux_bs = cc_pvdz_ri();
        let r = h3_pbe0_ref(&h3_setup());
        let seed = mos_of(r.none_stage.as_ref().expect("none stage"));
        fd(&H3_ATOMS, cubic(H3_A), 2, &H3_COMPS, FD_H, |c| {
            let prep = prep_for(c, &bs);
            let hc = periodic_hcore(c, &prep, &hcore_cfg()).unwrap();
            let (_, gdf) = energy_gdf(c, &prep, &hc, &aux_bs, DEFAULT_RSGDF_LINDEP);
            let r = uks_on(
                c,
                &prep,
                &hc,
                &gdf,
                &uks_cfg("PBE0", ExxDiv::Ewald, seed.clone()),
            );
            vec![r.none_stage.expect("none stage").energy, r.scf.energy]
        })
    })
}

#[test]
fn h3_uks_pbe0_force_matches_fd_and_catches_each_fit_mutant() {
    let su = h3_setup();
    let r = h3_pbe0_ref(&su);
    let none = r.none_stage.clone().expect("none stage");
    let fdv = h3_uks_fd();
    for (k, (scf, exx)) in [(&none, ExxDiv::None), (&r.scf, ExxDiv::Ewald)]
        .into_iter()
        .enumerate()
    {
        let cfg = uks_cfg("PBE0", exx, None);
        let g = uks_grad(&su, scf, &cfg, None);
        let err = max_fd_err(&g.grad, fdv, k);
        report(&format!("H3 UKS PBE0 cc-pvdz-ri {exx:?}"), &g, err);
        assert!((g.exact_exchange_fraction - 0.25).abs() < 1e-12);
        assert!(err < FD_BAR, "{exx:?}: {err:e}");
        assert!(g.net_force < NET_FORCE_BAR, "ΣF {:e}", g.net_force);
        for m in FIT_MUTANTS {
            let gm = uks_grad(&su, scf, &cfg, Some(m));
            let e = max_fd_err(&gm.grad, fdv, k);
            eprintln!("  mutant {m:?} ({exx:?}): {e:.2e}");
            assert!(
                e > MUTANT_BAR,
                "{m:?} ({exx:?}) escaped the FD anchor: {e:e}"
            );
        }
    }
}

// ------------------------------------------------- exact aux span vs dense

/// The 24 exact-span aux sites (`pbc_rsgdf.rs::anchor_sites`, all 8
/// classes) and their Jacobian `dC_k/dR_A`: (0,0) classes ride on atom 0,
/// (1,1) on atom 1, (0,1) midpoints half on each.
fn span_sites(cell: &Cell) -> (Vec<[f64; 4]>, Array2<f64>) {
    let r = cell.positions();
    let a = cell.lattice();
    let mut out = Vec::new();
    let mut jac = Array2::<f64>::zeros((24, 2));
    let mut row = 0;
    for (i, j) in [(0usize, 0usize), (1, 1), (0, 1)] {
        for k in 0..8usize {
            let h = [(k >> 2) & 1, (k >> 1) & 1, k & 1];
            let mut c = [0.0; 3];
            for d in 0..3 {
                let ha: f64 = (0..3).map(|x| h[x] as f64 * a[x][d]).sum();
                c[d] = 0.5 * (r[i][d] + r[j][d] + ha);
            }
            out.push([c[0], c[1], c[2], 1.0]);
            jac[(row, i)] += 0.5;
            jac[(row, j)] += 0.5;
            row += 1;
        }
    }
    (out, jac)
}

#[test]
fn exact_aux_span_rsgdf_force_equals_dense_aft_force() {
    let cell = cell_at(&H2_ATOMS_G, cubic(4.0), 1);
    let prep = prep_for(&cell, &single_s_h(0.5));
    let hc = periodic_hcore(&cell, &prep, &hcore_cfg()).unwrap();
    let (sites, jac) = span_sites(&cell);
    let site = SiteBasis::new(&sites, 0).unwrap();
    assert_eq!(site.prep.nbasis(), 24);
    // GDF ω ≠ h's ω: a c0 (h) vs c0' (fit) slip is visible.
    let gdf = RsGdf::build_for_gradient(
        &cell,
        &prep,
        &site.prep,
        &hc.s,
        &gdf_cfg(1.2, DEFAULT_RSGDF_LINDEP),
    )
    .unwrap();
    let dense = DenseAftEri::build(
        &cell,
        &prep,
        &hc.s,
        ExxDiv::None,
        DEFAULT_DENSE_AFT_PRECISION,
        DEFAULT_DENSE_AFT_MAX_BYTES,
    )
    .unwrap();
    let fit = RsGdfGradSource {
        gdf: &gdf,
        aux: &site.prep,
        aux_jac: Some(&jac),
    };
    for exx in [ExxDiv::None, ExxDiv::Ewald] {
        let eri = dense.clone().with_exxdiv(&cell, exx).unwrap();
        let scf_d = gamma_rhf(&cell, &prep, &hc, &eri);
        let scf_f = rhf_on(&cell, &prep, &hc, &gdf, exx);
        let gd = gamma_rhf_gradient_with(
            &cell,
            &prep,
            &hcore_cfg(),
            &hc,
            &eri,
            &scf_d,
            exx,
            &gcfg(None),
        )
        .unwrap();
        let run = |m: Option<GradMutation>| {
            gamma_rhf_gradient_rsgdf(&cell, &prep, &hcore_cfg(), &hc, &fit, &scf_f, exx, &gcfg(m))
                .unwrap()
        };
        let gf = run(None);
        let d = max_diff(&gf.grad, &gd.grad);
        let fdiag = gf.fit.as_ref().unwrap();
        eprintln!(
            "exact span {exx:?}: E_gdf − E_dense = {:.2e}, max|F_gdf − F_dense| = {d:.2e} \
             (|F| {:.3}), dropped {}, ΣF {:.1e}",
            scf_f.energy - scf_d.energy,
            amax(&gd.grad),
            fdiag.n_dropped,
            gf.net_force
        );
        assert!(amax(&gd.grad) > 0.05, "vacuous comparison");
        assert!(d < SPAN_BAR, "{exx:?}: {d:e}");
        // Aux motion is live on the ghost sites (the Jacobian path is used).
        assert!(amax(&gf.parts.fit_aux_sr) + amax(&gf.parts.fit_aux_lr) > 1e-3);
        for m in FIT_MUTANTS {
            let dm = max_diff(&run(Some(m)).grad, &gd.grad);
            eprintln!("  mutant {m:?}: max|F − F_dense| = {dm:.2e}");
            if m == GradMutation::FitG0Dense {
                // Blind by construction in the exact-span limit (prototype
                // 3.6e-9): the fitted pair charges ARE the true ones.
                assert!(dm < 1e-7, "{m:?} should be blind here: {dm:e}");
            } else {
                assert!(dm > 1e-3, "{m:?} escaped the span anchor: {dm:e}");
            }
        }
    }
}

// ------------------------------------------------------ active eigenvalue cut

/// H2 ET-40 at lindep 3e-3 (module doc): the Loewner form matches FD of the
/// CUT energy, the textbook form does not, and the dropped count is the
/// same at every displaced geometry.
#[test]
fn active_eigenvalue_cut_needs_the_loewner_metric_term() {
    let su = setup(
        cell_at(&H2_ATOMS_G, cubic(4.0), 1),
        &pyscf_sto3g_h(),
        &et40(),
        CUT_LINDEP,
    );
    let n_drop_ref = su.gdf.stats().n_dropped;
    assert!(n_drop_ref > 0, "the cut must be active");
    let bs = pyscf_sto3g_h();
    let aux_bs = et40();
    let fdv = fd(&H2_ATOMS_G, cubic(4.0), 1, &H2_COMPS, FD_H, |c| {
        let prep = prep_for(c, &bs);
        let hc = periodic_hcore(c, &prep, &hcore_cfg()).unwrap();
        let (_, gdf) = energy_gdf(c, &prep, &hc, &aux_bs, CUT_LINDEP);
        // A count change across ±h makes E(R) discontinuous and the FD
        // meaningless (prototype: never happened at h = 1e-4).
        assert_eq!(
            gdf.stats().n_dropped,
            n_drop_ref,
            "dropped count changed at a displaced geometry"
        );
        vec![rhf_on(c, &prep, &hc, &gdf, ExxDiv::None).energy]
    });
    let scf = rhf_on(&su.cell, &su.prep, &su.hc, &su.gdf, ExxDiv::None);
    let g_dk = rhf_grad(&su, &scf, ExxDiv::None, None);
    let g_std = rhf_grad(
        &su,
        &scf,
        ExxDiv::None,
        Some(GradMutation::FitTextbookMetric),
    );
    let e_dk = max_fd_err(&g_dk.grad, &fdv, 0);
    let e_std = max_fd_err(&g_std.grad, &fdv, 0);
    let cross = max_diff(&g_dk.grad, &g_std.grad);
    report("H2 ET-40 RHF cut 3e-3 (Loewner)", &g_dk, e_dk);
    eprintln!(
        "  textbook: max|analytic − FD| = {e_std:.2e}; |F_dk − F_std| = {cross:.2e}; \
         dropped {n_drop_ref}"
    );
    assert_eq!(g_dk.fit.as_ref().unwrap().n_dropped, n_drop_ref);
    assert!(
        cross > CUT_CROSS_MIN,
        "kept–dropped term inactive: {cross:e}"
    );
    assert!(e_dk < CUT_DK_BAR, "Loewner vs FD {e_dk:e}");
    assert!(
        e_std > CUT_STD_MIN,
        "textbook form escaped the FD anchor: {e_std:e}"
    );
}

// ---------------------------------------------------- triclinic s+p (slow)

#[test]
#[ignore = "slow: triclinic 4H s+p, cc-pvdz-ri, 8 displaced RS-GDF builds + 24 SCFs; run with --release -- --ignored"]
fn triclinic_sp_rhf_and_uhf_triplet_rsgdf_forces_match_fd() {
    let bs = sp_basis_h();
    let aux_bs = cc_pvdz_ri();
    // RHF, exxdiv none.
    let su_r = setup(
        cell_at(&TRI_MOVED, TRI_A, 1),
        &bs,
        &aux_bs,
        DEFAULT_RSGDF_LINDEP,
    );
    let scf_r = rhf_on(&su_r.cell, &su_r.prep, &su_r.hc, &su_r.gdf, ExxDiv::None);
    // UHF triplet, staged ewald (none stage + ewald).
    let su_u = setup(
        cell_at(&TRI_MOVED, TRI_A, 3),
        &bs,
        &aux_bs,
        DEFAULT_RSGDF_LINDEP,
    );
    let r_u = uhf_on(
        &su_u.cell,
        &su_u.prep,
        &su_u.hc,
        &su_u.gdf,
        &uhf_cfg(ExxDiv::Ewald, None),
    );
    let none_u = r_u.none_stage.clone().expect("none stage");
    let seed = mos_of(&none_u);
    let fd_r = fd(&TRI_MOVED, TRI_A, 1, &TRI_COMPS, FD_H, |c| {
        let prep = prep_for(c, &bs);
        let hc = periodic_hcore(c, &prep, &hcore_cfg()).unwrap();
        let (_, gdf) = energy_gdf(c, &prep, &hc, &aux_bs, DEFAULT_RSGDF_LINDEP);
        vec![rhf_on(c, &prep, &hc, &gdf, ExxDiv::None).energy]
    });
    let fd_u = fd(&TRI_MOVED, TRI_A, 3, &TRI_COMPS, FD_H, |c| {
        let prep = prep_for(c, &bs);
        let hc = periodic_hcore(c, &prep, &hcore_cfg()).unwrap();
        let (_, gdf) = energy_gdf(c, &prep, &hc, &aux_bs, DEFAULT_RSGDF_LINDEP);
        let r = uhf_on(c, &prep, &hc, &gdf, &uhf_cfg(ExxDiv::Ewald, seed.clone()));
        vec![r.none_stage.expect("none stage").energy, r.scf.energy]
    });
    // p shells: the libint 3-centre derivative floor of the Gaussian-nucleus
    // SR attraction on this cell is 1.85e-7 (pbc_grad.rs); bar from it.
    let tri_bar = 5e-7;
    let g = rhf_grad(&su_r, &scf_r, ExxDiv::None, None);
    let err = max_fd_err(&g.grad, &fd_r, 0);
    report("tri s+p RHF cc-pvdz-ri", &g, err);
    assert!(err < tri_bar, "RHF: {err:e}");
    assert!(g.net_force < NET_FORCE_BAR, "ΣF {:e}", g.net_force);
    for m in FIT_MUTANTS {
        let e = max_fd_err(
            &rhf_grad(&su_r, &scf_r, ExxDiv::None, Some(m)).grad,
            &fd_r,
            0,
        );
        eprintln!("  RHF mutant {m:?}: {e:.2e}");
        assert!(e > MUTANT_BAR, "RHF {m:?}: {e:e}");
    }
    for (k, (scf, exx)) in [(&none_u, ExxDiv::None), (&r_u.scf, ExxDiv::Ewald)]
        .into_iter()
        .enumerate()
    {
        let g = uhf_grad(&su_u, scf, exx, None);
        let err = max_fd_err(&g.grad, &fd_u, k);
        report(&format!("tri s+p UHF triplet {exx:?}"), &g, err);
        assert!(err < tri_bar, "UHF {exx:?}: {err:e}");
        assert!(g.net_force < NET_FORCE_BAR, "ΣF {:e}", g.net_force);
        for m in FIT_MUTANTS {
            let e = max_fd_err(&uhf_grad(&su_u, scf, exx, Some(m)).grad, &fd_u, k);
            eprintln!("  UHF mutant {m:?} ({exx:?}): {e:.2e}");
            assert!(e > MUTANT_BAR, "UHF {m:?} ({exx:?}): {e:e}");
        }
    }
}
