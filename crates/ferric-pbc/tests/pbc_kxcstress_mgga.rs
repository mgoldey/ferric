//! Analytic stress of the k-point closed-shell RKS energy for META-GGA
//! functionals (SCAN, r2SCAN; `ferric_pbc::kxcstress`, tau strain term).
//!
//! Derivation (the new piece). At fixed `D`, `tau = (1/2N_k) sum_k sum_i sum_{mu nu}
//! D_{mu nu} d_i phi_mu (d_i phi_nu)^*`, and `d(d_i phi)/d eps_ab = m2[3i+a, b]`
//! (the same strained-gradient moments the GGA `grad rho` strain already uses),
//! so `d tau/d eps_ab = (1/N_k) Re sum_{mu,i} m2[3i+a,b]_mu md^i_mu` (the 1/2
//! cancels the mu<->nu doubling), `md^i_mu = sum_nu D_{mu nu} (d_i phi_nu)^*`.
//! The stress gains `sum_g w_g v_tau d tau/d eps`; the weight response already
//! carries `eps_g = rho eps_xc(rho, sigma, tau)`; and `V^tau` enters the
//! Pulay Fock (guarded against the SCF's own Fock).
//!
//! Anchors (artifact hypothesis stated at each test):
//! * `null_tau_path_reproduces_the_gga_stress_exactly` -- PBE through the tau
//!   path (`v_tau = 0`) is the PBE stress;
//! * `fixed_density_xc_stress_matches_fd_of_exc` -- `xc_ao + xc_weight` against
//!   the central FD of `E_xc[D]` at FIXED `D` under cell strain (no SCF, the
//!   independent construction), SCAN and r2SCAN, atom and uniform grids, all
//!   nine components; the same comparison with each tau defect injected must fail;
//! * `meta_gga_stress_matches_fd_of_the_scf_energy` -- total dE/d eps against FD
//!   of the converged SCF energy (uniform grid, three components);
//! * `supercell_identity_per_part` -- the 1x1x3 stress equals the explicit
//!   3-cell supercell at 1x1x1 (same code; checks the k sum and phases);
//! * refusals, the Fock guard, production-wired mutants.
//!
//! NOT covered: the 1x1x1 = Gamma anchor (the Gamma periodic path has no
//! meta-GGA), hybrid meta-GGAs (none resolve), range-separated / VV10 / double
//! hybrid meta-GGAs (refused), open-shell meta-GGA stress (UKS/ROKS keep their
//! typed refusal), RS-GDF J, p/d shells, ECPs, heavy atoms, metals.

mod common;

use common::*;
use ferric_core::basis::{BasisSet, Shell};
use ferric_pbc::dense_aft::ExxDiv;
use ferric_pbc::dft::{PeriodicGrid, PeriodicGridConfig, PeriodicXcConfig};
use ferric_pbc::hcore::kpoint::{periodic_hcore_kpts, PeriodicHcoreK};
use ferric_pbc::hcore::PeriodicHcoreConfig;
use ferric_pbc::kdense_aft::KDenseAftEri;
use ferric_pbc::kdft::{solve_krks, solve_krks_on_grid, KPeriodicXc, KRksConfig, KRksResult};
use ferric_pbc::kgrad::KRsGdfGradSource;
use ferric_pbc::kpts::KPointMesh;
use ferric_pbc::kscf::{KJkKind, KPointXc, KScfResult};
use ferric_pbc::kstress::KStressConfig;
use ferric_pbc::kstress::KStressJk;
use ferric_pbc::kxcgrad::KXcGrid;
use ferric_pbc::kxcstress::{kpoint_rks_stress, KRksStress, KRksStressConfig, KXcStressMutation};
use ferric_pbc::lattice::Cell;
use ferric_pbc::rsgdf::kpoint::{KRsGdf, KRsGdfConfig};
use ferric_pbc::rsgdf::RsGdfConfig;
use ferric_pbc::stress::Mat3;
use ndarray::Array2;
use num_complex::Complex64;
use std::collections::HashMap;

const OMEGA: f64 = 0.8;
const AMPLE: usize = 1 << 31;
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
/// Neighbour cutoff shared by every atom-grid comparison that rebuilds a grid
/// (the default depends on the cell).
const CUTOFF: f64 = 10.0;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Jk {
    Dense,
    Gdf,
}

/// The prototype's ET-sp aux on H (as `pbc_kxcgrad.rs`).
fn et_sp_aux() -> BasisSet {
    let mut shells = Vec::new();
    for a in [4.7, 1.9, 0.75, 0.3] {
        shells.push(Shell {
            l: 0,
            pure: false,
            exponents: vec![a],
            coefficients: vec![1.0],
        });
    }
    for a in [1.25, 0.5] {
        shells.push(Shell {
            l: 1,
            pure: false,
            exponents: vec![a],
            coefficients: vec![1.0],
        });
    }
    let mut m = HashMap::new();
    m.insert(1, shells);
    BasisSet {
        name: "et-sp-aux-H".into(),
        shells: m,
        ecps: HashMap::new(),
    }
}

fn kgdf_cfg() -> KRsGdfConfig {
    KRsGdfConfig {
        gdf: RsGdfConfig {
            omega: 1.0,
            exxdiv: ExxDiv::None,
            budget_bytes: Some(AMPLE),
            ..Default::default()
        },
        mutation: None,
    }
}

fn atom_grid(nr: usize, na: usize, cutoff: Option<f64>) -> KXcGrid {
    let mut g = PeriodicGridConfig::with_size(nr, na);
    g.neighbour_cutoff = cutoff;
    KXcGrid::Atom(g)
}

fn cell_of(pos: &[[f64; 3]]) -> Cell {
    Cell::new(hydrogens(pos), cubic(4.0)).expect("cell")
}

fn krks_cfg(cell: &Cell, functional: &str, jk: Jk, grid: &KXcGrid, exxdiv: ExxDiv) -> KRksConfig {
    let mut cfg = KRksConfig::new(cell, exxdiv, functional);
    cfg.krhf.hcore = PeriodicHcoreConfig::with_omega(OMEGA);
    cfg.krhf.scf.energy_conv = 1e-13;
    cfg.krhf.scf.grad_conv = 1e-10;
    cfg.krhf.scf.max_iter = 200;
    if jk == Jk::Gdf {
        cfg.krhf.jk = KJkKind::RsGdf;
        cfg.krhf.rsgdf = kgdf_cfg();
    }
    if let KXcGrid::Atom(g) = grid {
        cfg.grid = g.clone();
    }
    cfg
}

/// Fresh SCF (energy) on `cell`.
fn scf_on(
    cell: &Cell,
    mesh_n: [usize; 3],
    functional: &str,
    jk: Jk,
    grid: &KXcGrid,
    exxdiv: ExxDiv,
) -> KRksResult {
    scf_bs(cell, mesh_n, functional, jk, grid, exxdiv, &pyscf_sto3g_h())
}

fn scf_bs(
    cell: &Cell,
    mesh_n: [usize; 3],
    functional: &str,
    jk: Jk,
    grid: &KXcGrid,
    exxdiv: ExxDiv,
    bs: &BasisSet,
) -> KRksResult {
    let prep = prep_for(cell, bs);
    let mesh = KPointMesh::gamma_centred(cell, mesh_n).unwrap();
    let cfg = krks_cfg(cell, functional, jk, grid, exxdiv);
    let aux_bs = et_sp_aux();
    let aux = (jk == Jk::Gdf).then(|| prep_for(cell, &aux_bs));
    match grid {
        KXcGrid::Atom(_) => solve_krks(cell, &prep, aux.as_ref(), &mesh, &cfg),
        KXcGrid::Uniform(n) => {
            let g = PeriodicGrid::uniform(cell, *n).unwrap();
            solve_krks_on_grid(cell, &prep, aux.as_ref(), &mesh, &g, &cfg)
        }
    }
    .unwrap_or_else(|e| panic!("solve_krks {functional}: {e}"))
}

fn energy_on(cell: &Cell, mesh_n: [usize; 3], f: &str, jk: Jk, grid: &KXcGrid) -> f64 {
    scf_on(cell, mesh_n, f, jk, grid, ExxDiv::None).scf.energy
}

struct Fx {
    cell: Cell,
    prep: ferric_integrals::basis_bridge::PreparedBasis,
    aux: ferric_integrals::basis_bridge::PreparedBasis,
    mesh: KPointMesh,
    hk: PeriodicHcoreK,
    res: KRksResult,
    jk: Jk,
    grid: KXcGrid,
    functional: String,
    exxdiv: ExxDiv,
}

fn fixture_cell(cell: Cell, mesh_n: [usize; 3], functional: &str, jk: Jk, grid: KXcGrid) -> Fx {
    fixture_exx(cell, mesh_n, functional, jk, grid, ExxDiv::None)
}

fn fixture_exx(
    cell: Cell,
    mesh_n: [usize; 3],
    functional: &str,
    jk: Jk,
    grid: KXcGrid,
    exxdiv: ExxDiv,
) -> Fx {
    fixture_bs(cell, mesh_n, functional, jk, grid, exxdiv, &pyscf_sto3g_h())
}

fn fixture_bs(
    cell: Cell,
    mesh_n: [usize; 3],
    functional: &str,
    jk: Jk,
    grid: KXcGrid,
    exxdiv: ExxDiv,
    bs: &BasisSet,
) -> Fx {
    let prep = prep_for(&cell, bs);
    let mesh = KPointMesh::gamma_centred(&cell, mesh_n).unwrap();
    let res = scf_bs(&cell, mesh_n, functional, jk, &grid, exxdiv, bs);
    let cfg = krks_cfg(&cell, functional, jk, &grid, exxdiv);
    let hk = periodic_hcore_kpts(&cell, &prep, &mesh, &cfg.krhf.hcore).unwrap();
    let aux = prep_for(&cell, &et_sp_aux());
    Fx {
        cell,
        prep,
        aux,
        mesh,
        hk,
        res,
        jk,
        grid,
        functional: functional.to_string(),
        exxdiv,
    }
}

fn fixture(pos: &[[f64; 3]], mesh_n: [usize; 3], functional: &str, jk: Jk, grid: KXcGrid) -> Fx {
    fixture_cell(cell_of(pos), mesh_n, functional, jk, grid)
}

fn scfg(grid: &KXcGrid, mutation: Option<KXcStressMutation>) -> KRksStressConfig {
    KRksStressConfig {
        stress: KStressConfig {
            budget_bytes: Some(AMPLE),
            ..Default::default()
        },
        grid: grid.clone(),
        mutation,
        ..Default::default()
    }
}

fn try_stress_cfg(fx: &Fx, cfg: &KRksStressConfig) -> Result<KRksStress, ferric_core::FerricError> {
    let hcfg = PeriodicHcoreConfig::with_omega(OMEGA);
    match fx.jk {
        Jk::Dense => {
            let eri = KDenseAftEri::build(
                &fx.cell,
                &fx.prep,
                &fx.mesh,
                &fx.hk.s,
                fx.exxdiv,
                &Default::default(),
            )
            .unwrap();
            kpoint_rks_stress(
                &fx.cell,
                &fx.prep,
                &fx.mesh,
                &hcfg,
                &fx.hk,
                KStressJk::Dense(&eri),
                &fx.res.scf,
                &fx.functional,
                cfg,
            )
        }
        Jk::Gdf => {
            let kc = kgdf_cfg();
            let gdf = KRsGdf::build(&fx.cell, &fx.prep, &fx.aux, &fx.mesh, &fx.hk.s, &kc)
                .unwrap()
                .with_exxdiv(fx.exxdiv);
            kpoint_rks_stress(
                &fx.cell,
                &fx.prep,
                &fx.mesh,
                &hcfg,
                &fx.hk,
                KStressJk::RsGdf(KRsGdfGradSource {
                    gdf: &gdf,
                    cfg: &kc,
                    aux: &fx.aux,
                }),
                &fx.res.scf,
                &fx.functional,
                cfg,
            )
        }
    }
}

fn stress_cfg(fx: &Fx, cfg: &KRksStressConfig) -> KRksStress {
    try_stress_cfg(fx, cfg).expect("kpoint_rks_stress")
}

fn stress(fx: &Fx, mutation: Option<KXcStressMutation>) -> KRksStress {
    stress_cfg(fx, &scfg(&fx.grid, mutation))
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

fn eps_at(i: usize, j: usize, h: f64) -> Mat3 {
    let mut e = [[0.0; 3]; 3];
    e[i][j] = h;
    e
}

fn strained(cell: &Cell, i: usize, j: usize, h: f64) -> Cell {
    cell.strained(&eps_at(i, j, h)).expect("strained cell")
}

// ----------------------------------------------------------- fixed-D XC FD

/// `E_xc[D]` at FIXED Bloch densities `d` on `cell` (a strained cell keeps the
/// fractional k mesh and the AO coefficients; the grid is rebuilt on the cell).
fn exc_fixed_d_on(
    cell: &Cell,
    mesh_n: [usize; 3],
    f: &str,
    grid: &KXcGrid,
    d: &[Array2<Complex64>],
) -> f64 {
    let bs = pyscf_sto3g_h();
    let mesh = KPointMesh::gamma_centred(cell, mesh_n).unwrap();
    let pg = match grid {
        KXcGrid::Atom(cfg) => PeriodicGrid::build(cell, cfg).unwrap(),
        KXcGrid::Uniform(n) => PeriodicGrid::uniform(cell, *n).unwrap(),
    };
    let mut xc = KPeriodicXc::new(cell, &bs, f, &pg, &mesh, &PeriodicXcConfig::default()).unwrap();
    xc.build(d).unwrap().0
}

/// Central FD of `E_xc[D_fixed]` under strain: all nine components.
fn fd_xc_all9(
    cell: &Cell,
    mesh_n: [usize; 3],
    f: &str,
    grid: &KXcGrid,
    d: &[Array2<Complex64>],
    h: f64,
) -> Mat3 {
    let mut out = [[0.0; 3]; 3];
    for (i, j) in ALL9 {
        let ep = exc_fixed_d_on(&strained(cell, i, j, h), mesh_n, f, grid, d);
        let em = exc_fixed_d_on(&strained(cell, i, j, -h), mesh_n, f, grid, d);
        out[i][j] = (ep - em) / (2.0 * h);
    }
    out
}

fn xc_part(s: &KRksStress) -> Mat3 {
    let mut o = s.xc_ao;
    for i in 0..3 {
        for j in 0..3 {
            o[i][j] += s.xc_weight[i][j];
        }
    }
    o
}

/// FD step of the atom-grid fixed-D FD (as the GGA anchor's `FD_H_ATOM`).
const FD_H_XC: f64 = 2.5e-5;
const FD_BAR_XC: f64 = 3e-8;
const MUTANT_BAR: f64 = 1e-5;

/// Null-tau limit. Artifact hypothesis: a tau path that perturbed the non-tau
/// terms (mis-sized plane, stray accumulation) would shift the PBE stress;
/// physics predicts exactly 0 because `v_tau = 0`.
#[test]
fn null_tau_path_reproduces_the_gga_stress_exactly() {
    let fx = fixture(
        &H2,
        [1, 1, 3],
        "PBE",
        Jk::Dense,
        atom_grid(20, 50, Some(CUTOFF)),
    );
    let base = stress(&fx, None);
    let hooked = stress(&fx, Some(KXcStressMutation::StressTauPath));
    let d = mdiff(&hooked.de_deps, &base.de_deps);
    eprintln!("PBE through the tau path: max diff {d:.2e}");
    assert!(d < 1e-12, "{d:e}");
}

/// The XC part (`xc_ao + xc_weight`) against central FD of `E_xc[D]` at fixed
/// `D` (the independent construction: no SCF, no HF part). Physics hypothesis:
/// residual at the FD floor. Artifact hypothesis: a missing/doubled tau strain
/// term, lost weight response or wrong prefactor misses by O(|v_tau dtau/de|),
/// far above the floor; this test also asserts each defect misses.
#[test]
fn fixed_density_xc_stress_matches_fd_of_exc() {
    let grid = atom_grid(20, 50, Some(CUTOFF));
    for f in ["SCAN", "R2SCAN"] {
        let fx = fixture(&H2, [1, 1, 3], f, Jk::Dense, grid.clone());
        let s = stress(&fx, None);
        let d0 = fx.res.scf.densities.clone();
        let fd = fd_xc_all9(&fx.cell, [1, 1, 3], f, &grid, &d0, FD_H_XC);
        let miss = mdiff(&xc_part(&s), &fd);
        eprintln!(
            "{f} atom (20,50): |xc - FD| {miss:.2e}; |xc| max {:.3e}",
            mmax(&xc_part(&s))
        );
        assert!(miss < FD_BAR_XC, "{f}: {miss:e}");
        for m in [
            KXcStressMutation::NoTauStrain,
            KXcStressMutation::TauDoubled,
            KXcStressMutation::WrongFactor,
            KXcStressMutation::NoWeightDeriv,
            KXcStressMutation::NoPointMotion,
            KXcStressMutation::NoHessian,
            KXcStressMutation::NoStrainMoment,
        ] {
            let ms = stress(&fx, Some(m));
            let mm = mdiff(&xc_part(&ms), &fd);
            eprintln!("  {f} {m:?}: |xc - FD| {mm:.2e}");
            assert!(mm > MUTANT_BAR, "{f} {m:?} survives the FD anchor: {mm:e}");
        }
    }
}

/// Same on a uniform grid (no partition, no home atom).
#[test]
fn fixed_density_xc_stress_matches_fd_on_the_uniform_grid() {
    let grid = KXcGrid::Uniform([24, 24, 24]);
    for f in ["SCAN", "R2SCAN"] {
        let fx = fixture(&H2, [1, 1, 3], f, Jk::Dense, grid.clone());
        let s = stress(&fx, None);
        let d0 = fx.res.scf.densities.clone();
        let fd = fd_xc_all9(&fx.cell, [1, 1, 3], f, &grid, &d0, 1e-4);
        let miss = mdiff(&xc_part(&s), &fd);
        eprintln!("{f} uniform 24^3: |xc - FD| {miss:.2e}");
        assert!(miss < FD_BAR_XC, "{f}: {miss:e}");
        let t = stress(&fx, Some(KXcStressMutation::NoTauStrain));
        assert!(mdiff(&xc_part(&t), &fd) > MUTANT_BAR);
    }
}

/// TPSS through its raw component names on a PBE density (there is no TPSS
/// SCF; `GuardSkippedOnly` skips only the Fock guard, which the XC parts do
/// not read).
#[test]
fn tpss_components_xc_stress_matches_fd_of_exc() {
    let grid = atom_grid(20, 50, Some(CUTOFF));
    let fx = fixture(&H2, [1, 1, 3], "PBE", Jk::Dense, grid.clone());
    let d0 = fx.res.scf.densities.clone();
    for f in ["MGGA_X_TPSS", "MGGA_C_TPSS"] {
        let mut fx2 = fixture_cell(fx.cell.clone(), [1, 1, 3], "PBE", Jk::Dense, grid.clone());
        fx2.functional = f.to_string();
        let s = stress(&fx2, Some(KXcStressMutation::GuardSkippedOnly));
        let fd = fd_xc_all9(&fx.cell, [1, 1, 3], f, &grid, &d0, FD_H_XC);
        let miss = mdiff(&xc_part(&s), &fd);
        eprintln!(
            "{f}: |xc - FD| {miss:.2e}; |xc| max {:.3e}",
            mmax(&xc_part(&s))
        );
        assert!(miss < FD_BAR_XC, "{f}: {miss:e}");
    }
}

// ------------------------------------------------------------ SCF-energy FD

/// Total dE/d eps against FD of the converged SCAN SCF energy, uniform grid.
#[test]
fn meta_gga_stress_matches_fd_of_the_scf_energy() {
    let grid = KXcGrid::Uniform([24, 24, 24]);
    let f = "SCAN";
    let fx = fixture(&H2, [1, 1, 3], f, Jk::Dense, grid.clone());
    let s = stress(&fx, None);
    let h = 1e-4;
    let mut worst = 0.0_f64;
    for (i, j) in [(0, 0), (2, 2), (0, 2)] {
        let ep = energy_on(&strained(&fx.cell, i, j, h), [1, 1, 3], f, Jk::Dense, &grid);
        let em = energy_on(
            &strained(&fx.cell, i, j, -h),
            [1, 1, 3],
            f,
            Jk::Dense,
            &grid,
        );
        let fd = (ep - em) / (2.0 * h);
        worst = worst.max((s.de_deps[i][j] - fd).abs());
    }
    eprintln!("SCAN uniform 24^3: |analytic - FD of SCF energy| {worst:.2e}");
    assert!(worst < 5e-7, "{worst:e}");
}

// ---------------------------------------------------------------- supercell

fn supercell_pos(pos: &[[f64; 3]], a: f64, n: [usize; 3]) -> Cell {
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
    Cell::new(hydrogens(&atoms), lat).unwrap()
}

/// The 1x1x3 stress per cell equals the explicit 3-cell supercell at 1x1x1 by
/// the same code (carries the Bloch phases and the 1/N_k; not the tau
/// derivative itself). Artifact hypothesis: a missing phase / 1/N_k in the tau
/// term would break it at the size of the tau term, not at 1e-10.
#[test]
fn supercell_identity_per_part() {
    let grid = atom_grid(20, 50, Some(CUTOFF));
    let f = "SCAN";
    let k = stress(&fixture(&H2, [1, 1, 3], f, Jk::Dense, grid.clone()), None);
    let sc = supercell_pos(&H2, 4.0, [1, 1, 3]);
    let g = stress(
        &fixture_cell(sc, [1, 1, 1], f, Jk::Dense, grid.clone()),
        None,
    );
    let d_ao = mdiff(&mscale(&k.xc_ao, 3.0), &g.xc_ao);
    let d_w = mdiff(&mscale(&k.xc_weight, 3.0), &g.xc_weight);
    let d_t = mdiff(&mscale(&k.de_deps, 3.0), &g.de_deps);
    eprintln!("SCAN 1x1x3 vs supercell: xc_ao {d_ao:.2e} xc_weight {d_w:.2e} total {d_t:.2e}");
    assert!(d_ao < 1e-9 && d_w < 1e-9 && d_t < 1e-9);
}

// ----------------------------------------------------- refusals, guard, mutants

#[test]
fn refusals_and_the_fock_guard() {
    let fx = fixture(
        &H2,
        [1, 1, 3],
        "SCAN",
        Jk::Dense,
        atom_grid(20, 50, Some(CUTOFF)),
    );
    let eri = KDenseAftEri::build(
        &fx.cell,
        &fx.prep,
        &fx.mesh,
        &fx.hk.s,
        ExxDiv::None,
        &Default::default(),
    )
    .unwrap();
    let hcfg = PeriodicHcoreConfig::with_omega(OMEGA);
    let cfg = scfg(&fx.grid, None);
    let run = |f: &str, scf: &KScfResult| {
        kpoint_rks_stress(
            &fx.cell,
            &fx.prep,
            &fx.mesh,
            &hcfg,
            &fx.hk,
            KStressJk::Dense(&eri),
            scf,
            f,
            &cfg,
        )
    };
    for (name, key) in [("HSE06", "range-separated"), ("wB97X-V", "VV10")] {
        let e = run(name, &fx.res.scf)
            .err()
            .unwrap_or_else(|| panic!("{name} accepted"))
            .to_string();
        assert!(e.contains(key), "{name}: {e}");
    }
    // Guard, both sides: the SCAN result is accepted as SCAN and refused as
    // PBE / r2SCAN (the missing or different V^tau is seen).
    let ok = run("SCAN", &fx.res.scf).unwrap();
    assert!(ok.hf.fock_mismatch < 1e-12, "{:e}", ok.hf.fock_mismatch);
    for wrong in ["PBE", "R2SCAN"] {
        let e = run(wrong, &fx.res.scf).expect_err(wrong).to_string();
        assert!(e.contains("differs"), "{wrong}: {e}");
    }
    let mut t = fx.res.scf.clone();
    t.fock[1][(0, 1)] += Complex64::new(1e-6, 0.0);
    assert!(run("SCAN", &t).is_err(), "1e-6 tamper must be refused");
    // A Fock built without V^tau (defect inside the stress): skipped guard.
    let pulay = stress(&fx, Some(KXcStressMutation::TauNoVxc));
    assert!(pulay.hf.fock_mismatch == 0.0);
}

/// Production-wired mutants against the FD-anchored base.
#[test]
fn mutants_are_caught() {
    let fx = fixture(
        &H2,
        [1, 1, 3],
        "SCAN",
        Jk::Dense,
        atom_grid(20, 50, Some(CUTOFF)),
    );
    let base = stress(&fx, None);
    for m in [
        KXcStressMutation::NoTauStrain,
        KXcStressMutation::TauDoubled,
        KXcStressMutation::TauNoVxc,
        KXcStressMutation::NoWeightDeriv,
        KXcStressMutation::NoPointMotion,
        KXcStressMutation::WrongFactor,
        KXcStressMutation::NoHessian,
        KXcStressMutation::NoStrainMoment,
    ] {
        let d = mdiff(&stress(&fx, Some(m)).de_deps, &base.de_deps);
        eprintln!("{m:?}: miss {d:.2e}");
        assert!(d > MUTANT_BAR, "{m:?} survives: {d:e}");
    }
}
