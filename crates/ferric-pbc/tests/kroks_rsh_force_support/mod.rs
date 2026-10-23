//! Shared fixtures of the restricted-open-shell range-separated force tests
//! (`pbc_kroks_rsh_force.rs` per commit, `validation_kroks_rsh_force.rs`
//! weekly): the k-ROKS SCF on dense-AFT kernels with the functional's range
//! separation, its force, central FD of the converged energy, the explicit
//! supercell, the zero-XC equal-fraction SCF, the closed-shell twin and the
//! Gamma route. Cells, grids, kernels and bars are the UKS ones
//! (`kuks_rsh_force_support`, included by the test files as `uks`), so a ROKS
//! row and its UKS row are the same physical problem.
#![allow(dead_code)]

use crate::common::*;
use crate::uks::*;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_pbc::dense_aft::ExxDiv;
use ferric_pbc::hcore::kpoint::{periodic_hcore_kpts, PeriodicHcoreK};
use ferric_pbc::hcore::PeriodicHcoreConfig;
use ferric_pbc::kdense_aft::KDenseAftEri;
use ferric_pbc::kgrad::{KGradConfig, KGradJk, KGradMutation};
use ferric_pbc::kpts::KPointMesh;
use ferric_pbc::kroks::{
    solve_kroks, solve_kroks_injected, KRoScfResult, KRoksConfig, KRoksResult,
};
use ferric_pbc::kscf::KPointInjection;
use ferric_pbc::kuxcgrad::{
    kpoint_null_xc_roks_gradient_rsh, kpoint_roks_gradient, KUSpinMutation, KUksGradConfig,
    KUksGradient,
};
use ferric_pbc::kxcgrad::{KRksGradConfig, KXcGrid};
use ferric_pbc::lattice::Cell;
use ferric_pbc::rsh::RshParams;

/// FD step: the UKS file's h = 1e-4 (its sweep shows the h^2 regime there).
pub const FD_H: f64 = 1e-4;

/// The k-ROKS settings: the UKS file's convergence, plus the 0.05 Ha level
/// shift (ramped to zero at convergence) the hybrid KROKS tests use.
pub fn kroks_cfg(cell: &Cell, f: &str, exxdiv: ExxDiv, conv: (f64, f64)) -> KRoksConfig {
    let mut c = KRoksConfig::new(cell, exxdiv, f);
    c.rohf.hcore = PeriodicHcoreConfig::with_omega(OMEGA);
    c.rohf.scf.energy_conv = conv.0;
    c.rohf.scf.grad_conv = conv.1;
    c.rohf.scf.max_iter = 600;
    c.rohf.level_shift = 0.05;
    c.rohf.dense = dense_cfg(None);
    c.grid = grid_cfg();
    c
}

/// Tight convergence (the UKS file's).
pub const TIGHT: (f64, f64) = (1e-13, 1e-10);
/// Loose convergence: a state past the orbital-gradient gate (measured on the
/// CAM-B3LYP H4 triplet 1x1x1: g 3.6e-5; 1e-4 / 1e-2 gives 3.1e-6, inside it).
pub const LOOSE: (f64, f64) = (1e-3, 1e-1);

pub fn roks_scf_conv(
    cell: &Cell,
    n: [usize; 3],
    f: &str,
    exxdiv: ExxDiv,
    conv: (f64, f64),
) -> KRoksResult {
    let prep = prep_for(cell, &pyscf_sto3g_h());
    let mesh = KPointMesh::gamma_centred(cell, n).unwrap();
    let r = solve_kroks(cell, &prep, None, &mesh, &kroks_cfg(cell, f, exxdiv, conv))
        .unwrap_or_else(|e| panic!("solve_kroks {f} {n:?} {exxdiv:?}: {e}"));
    assert!(r.scf.converged, "{f} {n:?} {exxdiv:?} not converged");
    r
}

pub fn roks_scf(cell: &Cell, n: [usize; 3], f: &str, exxdiv: ExxDiv) -> KRoksResult {
    roks_scf_conv(cell, n, f, exxdiv, TIGHT)
}

/// A converged k-ROKS RSH state with the kernels its force needs.
pub struct RoFx {
    pub cell: Cell,
    pub prep: PreparedBasis,
    pub mesh: KPointMesh,
    pub hk: PeriodicHcoreK,
    pub eri: KDenseAftEri,
    pub res: KRoksResult,
    pub f: String,
    pub exxdiv: ExxDiv,
}

pub fn ro_fixture_conv(
    cell: Cell,
    n: [usize; 3],
    f: &str,
    exxdiv: ExxDiv,
    conv: (f64, f64),
) -> RoFx {
    let prep = prep_for(&cell, &pyscf_sto3g_h());
    let mesh = KPointMesh::gamma_centred(&cell, n).unwrap();
    let res = roks_scf_conv(&cell, n, f, exxdiv, conv);
    let hk =
        periodic_hcore_kpts(&cell, &prep, &mesh, &PeriodicHcoreConfig::with_omega(OMEGA)).unwrap();
    // The kernels exactly as solve_kroks built them (the functional's rsh).
    let eri = KDenseAftEri::build(
        &cell,
        &prep,
        &mesh,
        &hk.s,
        ExxDiv::None,
        &dense_cfg(Some(rsh_of(f))),
    )
    .unwrap();
    RoFx {
        cell,
        prep,
        mesh,
        hk,
        eri,
        res,
        f: f.into(),
        exxdiv,
    }
}

pub fn ro_fixture_cell(cell: Cell, n: [usize; 3], f: &str, exxdiv: ExxDiv) -> RoFx {
    ro_fixture_conv(cell, n, f, exxdiv, TIGHT)
}

pub fn ro_fixture(kind: Kind, pos: &[[f64; 3]], n: [usize; 3], f: &str, exxdiv: ExxDiv) -> RoFx {
    ro_fixture_cell(cell_of(kind, pos), n, f, exxdiv)
}

pub fn ro_try_force(
    fx: &RoFx,
    scf: &KRoScfResult,
    f: &str,
    cfg: &KUksGradConfig,
) -> Result<KUksGradient, FerricError> {
    kpoint_roks_gradient(
        &fx.cell,
        &fx.prep,
        &fx.mesh,
        &PeriodicHcoreConfig::with_omega(OMEGA),
        &fx.hk,
        KGradJk::Dense(&fx.eri),
        scf,
        f,
        cfg,
    )
}

pub fn ro_force_m(
    fx: &RoFx,
    m: Option<KGradMutation>,
    sm: Option<KUSpinMutation>,
) -> Result<KUksGradient, FerricError> {
    ro_try_force(fx, &fx.res.scf, &fx.f, &gcfg(fx.exxdiv, m, sm))
}

pub fn ro_force(fx: &RoFx) -> KUksGradient {
    ro_force_m(fx, None, None).unwrap_or_else(|e| panic!("kpoint_roks_gradient {}: {e}", fx.f))
}

/// Central FD of the converged k-ROKS energy (fresh SCF per displacement).
#[allow(clippy::too_many_arguments)]
pub fn ro_fd(
    kind: Kind,
    pos: &[[f64; 3]],
    a: usize,
    x: usize,
    h: f64,
    n: [usize; 3],
    f: &str,
    e: ExxDiv,
) -> f64 {
    let en = |p: &[[f64; 3]]| roks_scf(&cell_of(kind, p), n, f, e).scf.energy;
    (en(&moved(pos, a, x, h)) - en(&moved(pos, a, x, -h))) / (2.0 * h)
}

/// `Σ_k Σ |D_α − D_β|`: proof that a ROKS state has an open shell.
pub fn ro_spin_asymmetry(scf: &KRoScfResult) -> f64 {
    scf.density_alpha
        .iter()
        .zip(&scf.density_beta)
        .map(|(a, b)| (a - b).iter().map(|z| z.norm()).sum::<f64>())
        .sum()
}

/// The UKS ledger (`kuks_rsh_force_support::ledger_mutants`) plus the ROKS
/// energy-weighted-matrix defect.
pub fn ro_ledger_mutants() -> Vec<(&'static str, Option<KGradMutation>, Option<KUSpinMutation>)> {
    let mut v = ledger_mutants();
    v.push(("RoWDropCross", None, Some(KUSpinMutation::RoWDropCross)));
    v
}

// ------------------------------------------------ zero-XC (equal fractions)

/// A zero-XC k-ROKS SCF on dense kernels with range separation `rsh` (or
/// bare) and exchange scale `a`.
pub struct RoZx {
    pub cell: Cell,
    pub prep: PreparedBasis,
    pub mesh: KPointMesh,
    pub hk: PeriodicHcoreK,
    pub eri: KDenseAftEri,
    pub scf: KRoScfResult,
    pub exxdiv: ExxDiv,
    pub rsh: Option<RshParams>,
    pub a: f64,
}

pub fn ro_zero_xc(
    pos: &[[f64; 3]],
    mult: usize,
    n: [usize; 3],
    exxdiv: ExxDiv,
    rsh: Option<RshParams>,
    a: f64,
) -> RoZx {
    let cell = cell_of(Kind::Tri(mult), pos);
    let prep = prep_for(&cell, &pyscf_sto3g_h());
    let mesh = KPointMesh::gamma_centred(&cell, n).unwrap();
    let hk =
        periodic_hcore_kpts(&cell, &prep, &mesh, &PeriodicHcoreConfig::with_omega(OMEGA)).unwrap();
    let eri =
        KDenseAftEri::build(&cell, &prep, &mesh, &hk.s, ExxDiv::None, &dense_cfg(rsh)).unwrap();
    // The K shift the SCF applies: the mixed constant of a range-separated
    // kernel (as solve_kroks), the mesh constant of a bare one.
    let vm = match (exxdiv, rsh) {
        (ExxDiv::None, _) => 0.0,
        (ExxDiv::Ewald, Some(_)) => eri.madelung_ewald(),
        (ExxDiv::Ewald, None) => mesh.madelung(&cell).unwrap(),
    };
    let mut sc = ferric_pbc::kscf::KScfConfig::default();
    sc.energy_conv = 1e-13;
    sc.grad_conv = 1e-10;
    sc.max_iter = 600;
    let scf = {
        let inj = KPointInjection {
            s: hk.s.clone(),
            h: hk.h.clone(),
            vnn: hk.enn,
            jk: Box::new(eri.jk_builder_with_madelung(vm)),
        };
        let ne = cell.mol().nelec() as usize;
        let nb = (ne + 1 - mult) / 2;
        let na = ne - nb;
        solve_kroks_injected(&cell, &mesh, &sc, inj, na, nb, None, a).unwrap()
    };
    assert!(
        scf.converged,
        "zero-XC k-ROKS ({rsh:?}, a = {a}) not converged"
    );
    RoZx {
        cell,
        prep,
        mesh,
        hk,
        eri,
        scf,
        exxdiv,
        rsh,
        a,
    }
}

pub fn ro_null_force(z: &RoZx, m: Option<KGradMutation>) -> KUksGradient {
    kpoint_null_xc_roks_gradient_rsh(
        &z.cell,
        &z.prep,
        &z.mesh,
        &PeriodicHcoreConfig::with_omega(OMEGA),
        &z.hk,
        KGradJk::Dense(&z.eri),
        &z.scf,
        z.a,
        z.rsh,
        &gcfg(z.exxdiv, m, None),
    )
    .unwrap()
}

/// The equal-fraction anchor (`pbc_kroks_rsh_force.rs` (b)) under `exxdiv`:
/// `(|ΔE|, |ΔF|)` and each listed mutant's miss against the global hybrid.
pub fn ro_equal_fraction_case(exxdiv: ExxDiv) {
    let c = 0.4;
    let eq = RshParams {
        omega: 0.33,
        c_sr: c,
        c_lr: c,
    };
    let zr = ro_zero_xc(&H3D, 2, [1, 1, 2], exxdiv, Some(eq), 1.0);
    let zg = ro_zero_xc(&H3D, 2, [1, 1, 2], exxdiv, None, c);
    let de = (zr.scf.energy - zg.scf.energy).abs();
    let gr = ro_null_force(&zr, None);
    let gg = ro_null_force(&zg, None);
    let d = diff(&gr.grad, &gg.grad);
    eprintln!(
        "ROKS {exxdiv:?}: equal fractions vs global c={c}: |dE| {de:.1e} force {d:.2e} \
         K part {:.2e}",
        diff(&gr.hf.parts.exchange, &gg.hf.parts.exchange)
    );
    assert!(ro_spin_asymmetry(&zr.scf) > 0.1, "open shell");
    assert!(de < 1e-11, "{de:e}");
    assert!(d < ANCHOR_TOL, "{exxdiv:?}: {d:e}");
    assert!(max_abs(&gg.hf.parts.exchange) > 1e-4);
    for m in [
        KGradMutation::RshNoKernelFactor,
        KGradMutation::RshKernelOnJ,
        KGradMutation::RshPlainMadelung,
        KGradMutation::MTermTotalDensity,
        KGradMutation::ExchBetaUsesAlpha,
    ] {
        let miss = diff(&ro_null_force(&zr, Some(m)).grad, &gg.grad);
        eprintln!("  ROKS {exxdiv:?} equal-fraction anchor vs {m:?}: miss {miss:.2e}");
        let blind = exxdiv == ExxDiv::None
            && matches!(
                m,
                KGradMutation::RshPlainMadelung | KGradMutation::MTermTotalDensity
            );
        if blind {
            assert!(miss < ANCHOR_TOL, "blind by construction under exxdiv none");
        } else {
            assert!(miss > 1e-3, "{m:?} {exxdiv:?}: {miss:e}");
        }
    }
}

// ------------------------------------------------ closed-shell twin

/// The closed-shell k-RKS RSH force (#486) on the cell, mesh, grid and
/// kernels of the singlet ROKS fixture, from an independent closed-shell SCF.
pub fn ro_rks_twin(fx: &RoFx) -> ferric_pbc::kxcgrad::KRksGradient {
    let mut cfg = ferric_pbc::kdft::KRksConfig::new(&fx.cell, fx.exxdiv, &fx.f);
    cfg.krhf.hcore = PeriodicHcoreConfig::with_omega(OMEGA);
    cfg.krhf.scf.energy_conv = 1e-13;
    cfg.krhf.scf.grad_conv = 1e-10;
    cfg.krhf.scf.max_iter = 300;
    cfg.krhf.dense = dense_cfg(None);
    cfg.grid = grid_cfg();
    let r = ferric_pbc::kdft::solve_krks(&fx.cell, &fx.prep, None, &fx.mesh, &cfg).unwrap();
    assert!(r.scf.converged);
    ferric_pbc::kxcgrad::kpoint_rks_gradient(
        &fx.cell,
        &fx.prep,
        &fx.mesh,
        &PeriodicHcoreConfig::with_omega(OMEGA),
        &fx.hk,
        KGradJk::Dense(&fx.eri),
        &r.scf,
        &fx.f,
        &KRksGradConfig {
            grad: KGradConfig {
                budget_bytes: Some(AMPLE),
                ..Default::default()
            },
            grid: KXcGrid::Atom(grid_cfg()),
            exxdiv: Some(fx.exxdiv),
            ..Default::default()
        },
    )
    .unwrap()
}

/// Singlet k-ROKS vs k-RKS per part on the H2 singlet (1x1x3): `(label,
/// |ROKS - RKS|)` rows, plus the ROKS fixture and force for the blindness
/// checks.
pub fn ro_singlet_rows(f: &str) -> (Vec<(&'static str, f64)>, RoFx, KUksGradient) {
    let fx = ro_fixture(Kind::Cube, &H2, [1, 1, 3], f, ExxDiv::Ewald);
    assert!(ro_spin_asymmetry(&fx.res.scf) < 1e-8);
    let u = ro_force(&fx);
    let r = ro_rks_twin(&fx);
    let (p, q) = (&u.hf.parts, &r.hf.parts);
    assert!(max_abs(&p.exchange) > 1e-3, "exchange exercised");
    let rows = vec![
        ("exchange", diff(&p.exchange, &q.exchange)),
        ("coulomb", diff(&p.coulomb, &q.coulomb)),
        ("overlap", diff(&p.overlap, &q.overlap)),
        ("xc_ao", diff(&u.xc_ao, &r.xc_ao)),
        ("total", diff(&u.grad, &r.grad)),
    ];
    (rows, fx, u)
}

/// The defects that only reassign spins: invisible on a singlet (the UKS
/// list; `RoWDropCross` drops half of `W` there and is NOT blind).
pub const RO_SPIN_ONLY: [&str; 3] = SPIN_ONLY;

// ------------------------------------------------ supercell, Gamma

/// 1x1x2 mesh force of the H3 doublet vs one copy in the explicit 2-cell
/// supercell (6 H, high-spin ROKS triplet, one k-point).
pub fn ro_mesh_vs_supercell(f: &str) -> f64 {
    let k = ro_force(&ro_fixture(Kind::Tri(2), &H3D, [1, 1, 2], f, ExxDiv::Ewald));
    let sc = ro_force(&ro_fixture_cell(
        supercell(&H3D, [1, 1, 2], 3),
        [1, 1, 1],
        f,
        ExxDiv::Ewald,
    ));
    diff(&k.grad, &sc.grad.slice(ndarray::s![0..3, ..]).to_owned())
}

/// The Gamma ROKS RSH SCF of the H3 doublet (`gamma_roks_rsh`, dense J and
/// exchange) and everything its routed force needs.
pub struct GammaRoFx {
    pub cell: Cell,
    pub prep: PreparedBasis,
    pub hcfg: PeriodicHcoreConfig,
    pub ex: ferric_pbc::gamma_rsh::GammaRshExchange,
    pub gc: ferric_pbc::rohf::GammaRoksConfig,
    pub scf: ferric_scf::result::ScfResult,
}

pub fn ro_gamma_fixture(f: &str, exxdiv: ExxDiv) -> GammaRoFx {
    use ferric_pbc::dense_aft::{DenseAftEri, DEFAULT_DENSE_AFT_MAX_BYTES};
    let cell = cell_of(Kind::Tri(2), &H3D);
    let prep = prep_for(&cell, &pyscf_sto3g_h());
    let hcfg = PeriodicHcoreConfig::with_omega(OMEGA);
    let hc = ferric_pbc::hcore::periodic_hcore(&cell, &prep, &hcfg).unwrap();
    let eri = DenseAftEri::build(
        &cell,
        &prep,
        &hc.s,
        exxdiv,
        PRECISION,
        DEFAULT_DENSE_AFT_MAX_BYTES,
    )
    .unwrap();
    let ex = ferric_pbc::gamma_rsh::GammaRshExchange::build_dense(
        &cell,
        &prep,
        &hc.s,
        exxdiv,
        &dense_cfg(None),
        rsh_of(f),
    )
    .unwrap();
    let mut gc = ferric_pbc::rohf::GammaRoksConfig::new(f);
    gc.grid = grid_cfg();
    gc.exxdiv = exxdiv;
    gc.scf.energy_conv = 1e-13;
    gc.scf.density_conv = 1e-10;
    let r = ferric_pbc::rohf::gamma_roks_rsh(
        &cell,
        &prep,
        &hc,
        ferric_pbc::uhf::GammaUhfIntegrals::DenseAft(&eri),
        &ex,
        &gc,
    )
    .unwrap();
    assert!(r.scf.converged);
    GammaRoFx {
        cell,
        prep,
        hcfg,
        ex,
        gc,
        scf: r.scf,
    }
}

pub fn ro_gamma_force(g: &GammaRoFx) -> Result<KUksGradient, FerricError> {
    ferric_pbc::gamma_rsh::gamma_roks_rsh_gradient(
        &g.cell,
        &g.prep,
        &g.hcfg,
        &g.scf,
        &g.gc,
        &g.ex,
        &kgrad_cfg(),
    )
}

/// Gamma vs the 1x1x1 k-ROKS force on the H3 doublet: `(|diff|, routed
/// Fock guard on the stored spin Focks, max |F|)`.
pub fn ro_gamma_vs_one_point(f: &str, exxdiv: ExxDiv) -> (f64, f64, f64) {
    let g = ro_gamma_force(&ro_gamma_fixture(f, exxdiv)).unwrap();
    let k = ro_force(&ro_fixture(Kind::Tri(2), &H3D, [1, 1, 1], f, exxdiv));
    (diff(&g.grad, &k.grad), g.hf.fock_mismatch, max_abs(&g.grad))
}
