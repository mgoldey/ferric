//! Analytic stress of the k-point closed-shell RKS energy for META-GGA
//! functionals (SCAN, r2SCAN, TPSS; `ferric_pbc::kxcstress`, tau strain term).
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
//!   independent construction), SCAN and r2SCAN, atom grid, all nine
//!   components; the same comparison with each XC strain defect injected must
//!   fail; `fixed_density_xc_stress_matches_fd_on_the_uniform_grid` repeats it
//!   on the uniform grid with the same defects;
//! * `p_shell_triclinic_fixed_density_xc_stress_matches_fd` -- s+p basis on the
//!   triclinic cell, all nine components, and the `TauTranspose` defect (the
//!   (a,b)-transposed tau strain term) that the cubic s-only fixtures cannot see;
//! * `uniform_grid_stress_matches_pyscf_fftdf_krks_stress_scan` -- PySCF 2.13.1
//!   analytic k-point SCAN stress (JSON under testdata/reference);
//! * `meta_gga_stress_matches_fd_of_the_scf_energy` -- total dE/d eps against FD
//!   of the converged SCF energy (uniform grid, three components);
//! * `supercell_identity_per_part` -- the 1x1x3 stress equals the explicit
//!   3-cell supercell at 1x1x1 (same code; checks the k sum and phases);
//! * RS-GDF J: `rsgdf_meta_gga_stress_matches_fd_of_the_scf_energy` (FD of the
//!   SCAN RS-GDF SCF energy, tau defects on that route must miss; all nine
//!   components SCAN + r2SCAN ignored) and
//!   `rsgdf_minus_dense_stress_transfers_from_pbe_to_scan` (the route
//!   difference is the fit error, the same for SCAN as for PBE);
//! * refusals, the Fock guard, production-wired mutants;
//! * TPSS (issue #492): `uniform_grid_stress_matches_pyscf_fftdf_krks_stress_tpss`
//!   (PySCF `krks_stress` accepts TPSS) and
//!   `tpss_stress_matches_fd_of_the_scf_energy` (FD of the TPSS SCF energy,
//!   with the tau mutants); the H4 triclinic cell and the TPSS k-mesh ==
//!   supercell identity are weekly (`validation_pbc_mgga_derivs.rs`).
//!
//! NOT covered: hybrid meta-GGAs (none resolve), range-separated / VV10 /
//! double hybrid meta-GGAs (refused), open-shell meta-GGA stress (UKS/ROKS
//! keep their typed refusal; open-shell TPSS is refused), TPSS on RS-GDF, the
//! range-split RS-GDF build, d shells, ECPs, heavy atoms, metals. The
//! Gamma-point meta-GGA stress is this code on the 1x1x1 mesh
//! (`pbc_gamma_mgga_derivs.rs`).

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
    bs: &BasisSet,
    mesh_n: [usize; 3],
    f: &str,
    grid: &KXcGrid,
    d: &[Array2<Complex64>],
) -> f64 {
    let mesh = KPointMesh::gamma_centred(cell, mesh_n).unwrap();
    let pg = match grid {
        KXcGrid::Atom(cfg) => PeriodicGrid::build(cell, cfg).unwrap(),
        KXcGrid::Uniform(n) => PeriodicGrid::uniform(cell, *n).unwrap(),
    };
    let mut xc = KPeriodicXc::new(cell, bs, f, &pg, &mesh, &PeriodicXcConfig::default()).unwrap();
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
    fd_xc_all9_bs(cell, &pyscf_sto3g_h(), mesh_n, f, grid, d, h)
}

fn fd_xc_all9_bs(
    cell: &Cell,
    bs: &BasisSet,
    mesh_n: [usize; 3],
    f: &str,
    grid: &KXcGrid,
    d: &[Array2<Complex64>],
    h: f64,
) -> Mat3 {
    let mut out = [[0.0; 3]; 3];
    for (i, j) in ALL9 {
        let ep = exc_fixed_d_on(&strained(cell, i, j, h), bs, mesh_n, f, grid, d);
        let em = exc_fixed_d_on(&strained(cell, i, j, -h), bs, mesh_n, f, grid, d);
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
/// The defects of the XC strain part (the Pulay-only `TauNoVxc` is not in the
/// fixed-density XC FD).
const XC_MUTANTS: [KXcStressMutation; 7] = [
    KXcStressMutation::NoTauStrain,
    KXcStressMutation::TauDoubled,
    KXcStressMutation::WrongFactor,
    KXcStressMutation::NoWeightDeriv,
    KXcStressMutation::NoPointMotion,
    KXcStressMutation::NoHessian,
    KXcStressMutation::NoStrainMoment,
];

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
        for m in XC_MUTANTS {
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
        for m in XC_MUTANTS {
            let t = stress(&fx, Some(m));
            let mm = mdiff(&xc_part(&t), &fd);
            eprintln!("  {f} uniform {m:?}: |xc - FD| {mm:.2e}");
            // A uniform grid has no partition/home atom: `NoPointMotion` is
            // still a real defect there (points move with the lattice), and
            // `PointSelf` is not in the list (an identity on this grid).
            assert!(mm > MUTANT_BAR, "{f} uniform {m:?} survives: {mm:e}");
        }
    }
}

/// TPSS through its raw component names on a PBE density (`GuardSkippedOnly`
/// skips only the Fock guard, which the XC parts do not read).
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

// ------------------------------------------------------------- p shells

/// Fixed-density XC stress FD with s+p shells on the TRICLINIC cell (uniform
/// grid, SCAN, all nine components). Physics hypothesis: residual at the FD
/// floor (measured below). Artifact hypothesis: an (a,b)-transposed tau strain
/// term is an identity when the moments are symmetric in (a,b) (cubic cell,
/// s shells) and misses here; the cubic s-only tests above do not see it.
/// The atom grid needs `neighbour_cutoff >= 14` for a smooth FD (cutoff 10
/// puts a 3.6e-5 mask jump in the energy), so this is a uniform-grid test.
#[test]
#[ignore = "16-function s+p SCAN SCF on the 4-atom triclinic cell, a fixed-D XC FD (18 strained builds) and three stresses; measured 217 s alone on a shared box, run serially on a quiet box"]
fn p_shell_triclinic_fixed_density_xc_stress_matches_fd() {
    let bs = sp_basis_h();
    let grid = KXcGrid::Uniform([24; 3]);
    let fx = fixture_bs(
        triclinic_cell(),
        [1, 1, 2],
        "SCAN",
        Jk::Dense,
        grid.clone(),
        ExxDiv::None,
        &bs,
    );
    let s = stress(&fx, None);
    let d0 = fx.res.scf.densities.clone();
    let fd = fd_xc_all9_bs(&fx.cell, &bs, [1, 1, 2], "SCAN", &grid, &d0, 1e-4);
    let miss = mdiff(&xc_part(&s), &fd);
    eprintln!(
        "triclinic s+p SCAN uniform 24^3: |xc - FD| {miss:.2e}; |xc| max {:.3e}",
        mmax(&xc_part(&s))
    );
    assert!(miss < P_SHELL_BAR, "{miss:e}");
    for m in [
        KXcStressMutation::TauTranspose,
        KXcStressMutation::NoTauStrain,
        KXcStressMutation::NoHessian,
    ] {
        let mm = mdiff(&xc_part(&stress(&fx, Some(m))), &fd);
        eprintln!("  triclinic s+p {m:?}: |xc - FD| {mm:.2e}");
        assert!(mm > MUTANT_BAR, "{m:?} survives: {mm:e}");
    }
}

/// Both sides, measured (1x1x2, uniform 24^3, h = 1e-4): correct 3.4e-8;
/// `TauTranspose` 6.0e-4, `NoHessian` 6.6e-2, `NoTauStrain` 1.4e-1. The bar is
/// 3x above the floor and 6000x below the smallest defect.
const P_SHELL_BAR: f64 = 1e-7;

// ----------------------------------------------------------- PySCF anchor

/// PySCF 2.13.1 `pbc.grad.krks_stress` for SCAN / r2SCAN (FFTDF on
/// `UniformGrids(61^3)`, exxdiv ewald, conv_tol 1e-12, H2/STO-3G in the a = 4
/// cube, 1x1x2 gamma-centred mesh; the settings of the GGA anchor in
/// `pbc_kxcstress.rs`), written by `scripts/gen_pyscf_kks_stress_refs.py` to
/// `testdata/reference/pbc_h2_sto3g_kks_stress_mgga.json`: the full 3x3
/// `sigma = (1/Omega) dE/d eps` (Hartree/Bohr^3). PySCF's off-diagonals are
/// zero to 2.6e-11 (its own noise). Bar 1e-9: the GGA anchor's PySCF
/// analytic-vs-FD agreement is 2.3e-10 and the observed ferric agreement is
/// 7e-11 (printed); the mutant side is asserted against the same reference.
const PYSCF_MGGA_BAR: f64 = 1e-9;

fn pyscf_mgga_case(f: &str, key: &str) {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../testdata/reference/pbc_h2_sto3g_kks_stress_mgga.json"
    );
    let doc: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path).expect("reference json")).unwrap();
    assert_eq!(doc["pyscf_version"], "2.13.1");
    let rows = &doc["rows"][key]["sigma"];
    let mut want = [[0.0; 3]; 3];
    for (i, row) in want.iter_mut().enumerate() {
        for (j, x) in row.iter_mut().enumerate() {
            *x = rows[i][j].as_f64().expect("sigma entry");
        }
    }
    let fx = fixture_exx(
        cell_of(&H2_ATOMS),
        [1, 1, 2],
        f,
        Jk::Dense,
        KXcGrid::Uniform([61, 61, 61]),
        ExxDiv::Ewald,
    );
    let s = stress(&fx, None);
    let d = mdiff(&s.sigma, &want);
    eprintln!(
        "{f} 61^3 1x1x2 vs PySCF: |sigma - sigma_pyscf| {d:.2e} (xx {:.9e} vs {:.9e}, zz {:.9e} vs {:.9e})",
        s.sigma[0][0], want[0][0], s.sigma[2][2], want[2][2]
    );
    assert!(d < PYSCF_MGGA_BAR, "{f}: {d:e}");
    // Both sides: the tau defect misses the same reference.
    let m = stress(&fx, Some(KXcStressMutation::NoTauStrain));
    let dm = mdiff(&m.sigma, &want);
    eprintln!("{f} NoTauStrain vs PySCF: {dm:.2e}");
    assert!(dm > 1e4 * PYSCF_MGGA_BAR, "{f}: mutant {dm:e}");
}

#[test]
fn uniform_grid_stress_matches_pyscf_fftdf_krks_stress_scan() {
    pyscf_mgga_case("SCAN", "SCAN");
}

#[test]
#[ignore = "second 61^3 PySCF-anchor row (r2SCAN); one SCF plus two stresses on a 61^3 grid, measured about the SCAN row's time"]
fn uniform_grid_stress_matches_pyscf_fftdf_krks_stress_r2scan() {
    pyscf_mgga_case("R2SCAN", "R2SCAN");
}

/// TPSS row of the PySCF anchor (issue #492). PySCF's own TPSS analytic
/// stress agrees with the central FD of its energy to 2.3e-10 (h = 1e-4,
/// xx / yy / zz / xz, run by hand with the generator's cell); its xx and yy
/// differ by 4.1e-8 and its xz is -3.4e-9 in BOTH the analytic tensor and the
/// FD (the uniform-grid egg-box effect, larger for TPSS than for SCAN), so
/// they are part of the reference, not noise. MEASURED (2026-10-22): ferric
/// vs PySCF 7.5e-11 (bar 1e-9); `NoTauStrain` misses by 6.5e-4. The source
/// mutant this pin exists for, `TPSS` mapped to the SCAN pair in the libxc
/// name table (run and reverted), misses by 7.8e-5: FD of ferric's own
/// energy cannot see it.
#[test]
fn uniform_grid_stress_matches_pyscf_fftdf_krks_stress_tpss() {
    pyscf_mgga_case("TPSS", "TPSS");
}

/// TPSS dE/d eps against FD of the converged TPSS SCF energy (uniform 24^3,
/// two components), and the tau defects against the same FD. Physics: the
/// FD floor (SCAN's on this fixture is in `meta_gga_stress_matches_fd_of_the_scf_energy`).
/// Artifact: a dropped / doubled tau strain term or V^tau missing from the
/// Pulay Fock misses by >= 1e-4. MEASURED (2026-10-22): correct 1.1e-8 (bar
/// 5e-7, shared with the SCAN row); misses `NoTauStrain` 5.0e-2, `TauDoubled`
/// 5.0e-2, `TauNoVxc` 2.3e-2, `NoPointMotion` 5.2e-1. `StressTauPath` (take
/// the tau path) is an EQUIVALENT mutant on a meta-GGA (the path is already
/// taken): measured 0, asserted.
#[test]
fn tpss_stress_matches_fd_of_the_scf_energy() {
    let grid = KXcGrid::Uniform([24, 24, 24]);
    let f = "TPSS";
    let fx = fixture(&H2, [1, 1, 3], f, Jk::Dense, grid.clone());
    let s = stress(&fx, None);
    let h = 1e-4;
    let comps = [(0, 0), (2, 2)];
    let fd: Vec<f64> = comps
        .iter()
        .map(|&(i, j)| {
            (energy_on(&strained(&fx.cell, i, j, h), [1, 1, 3], f, Jk::Dense, &grid)
                - energy_on(
                    &strained(&fx.cell, i, j, -h),
                    [1, 1, 3],
                    f,
                    Jk::Dense,
                    &grid,
                ))
                / (2.0 * h)
        })
        .collect();
    let miss = |t: &KRksStress| {
        comps.iter().zip(&fd).fold(0.0_f64, |m, (&(i, j), v)| {
            m.max((t.de_deps[i][j] - v).abs())
        })
    };
    let worst = miss(&s);
    eprintln!("TPSS uniform 24^3: |analytic - FD of SCF energy| {worst:.2e}");
    assert!(worst < 5e-7, "{worst:e}");
    for m in [
        KXcStressMutation::NoTauStrain,
        KXcStressMutation::TauDoubled,
        KXcStressMutation::TauNoVxc,
        KXcStressMutation::NoPointMotion,
    ] {
        let d = miss(&stress(&fx, Some(m)));
        eprintln!("  TPSS {m:?}: miss vs FD {d:.2e}");
        assert!(d > MUTANT_BAR, "{m:?} survives: {d:e}");
    }
    let d = mdiff(
        &stress(&fx, Some(KXcStressMutation::StressTauPath)).de_deps,
        &s.de_deps,
    );
    eprintln!("  TPSS StressTauPath (identity on a meta-GGA): {d:.2e}");
    assert!(d < 1e-12, "{d:e}");
}

// ------------------------------------------------------------- RS-GDF J

// The tau strain term lives in the grid XC response (`GridXc`), which never
// sees the J route; RS-GDF changes only the HF assembly (the fit's Coulomb
// pieces), anchored for PBE in `pbc_kxcstress.rs`. PHYSICS hypothesis: the
// SCAN RS-GDF stress is the derivative of the SCAN RS-GDF energy (FD at the
// dense-route floor), and the route difference (RS-GDF - dense) of the
// stress is the fit error, nearly the same for SCAN as for PBE on the same
// cell/aux. ARTIFACT hypotheses: a route-dependent tau defect (e.g. V^tau
// missing from the Pulay Fock on the fit route) misses FD by the size of the
// tau term and breaks the transfer by the same amount; a "pass" because the
// dense code ran is excluded by asserting the RS-GDF stress differs from the
// dense one by more than the transfer gap.

/// Total dE/d eps of a SCAN RS-GDF SCF against FD of its energy (uniform grid,
/// the dense test's step and bar), with the tau defects on the RS-GDF route
/// missing the same FD. Measured: (2,2) 1.1e-8, (0,2) 2.0e-10 (bar 5e-7);
/// all nine (ignored) worst 1.1e-8 SCAN and r2SCAN. Mutants against the same
/// FD: NoTauStrain 6.0e-2, TauDoubled 6.0e-2, TauNoVxc 5.6e-4.
#[test]
fn rsgdf_meta_gga_stress_matches_fd_of_the_scf_energy() {
    rsgdf_scf_fd_case("SCAN", &[(2, 2), (0, 2)], true);
}

/// All nine components, SCAN and r2SCAN, on RS-GDF.
#[test]
#[ignore = "SCAN + r2SCAN RS-GDF all-nine FD: 2 x 18 strained SCFs on a 24^3 uniform grid"]
fn rsgdf_meta_gga_stress_matches_fd_all_nine() {
    for f in ["SCAN", "R2SCAN"] {
        rsgdf_scf_fd_case(f, &ALL9, false);
    }
}

/// FD step and bar of the SCF-energy FD on the uniform grid (the dense
/// `meta_gga_stress_matches_fd_of_the_scf_energy` values).
const SCF_FD_H: f64 = 1e-4;
const SCF_FD_BAR: f64 = 5e-7;

fn rsgdf_scf_fd_case(f: &str, comps: &[(usize, usize)], mutants: bool) {
    let grid = KXcGrid::Uniform([24, 24, 24]);
    let fx = fixture(&H2, [1, 1, 3], f, Jk::Gdf, grid.clone());
    let s = stress(&fx, None);
    let mut fds = Vec::new();
    let mut worst = 0.0_f64;
    for &(i, j) in comps {
        let ep = energy_on(
            &strained(&fx.cell, i, j, SCF_FD_H),
            [1, 1, 3],
            f,
            Jk::Gdf,
            &grid,
        );
        let em = energy_on(
            &strained(&fx.cell, i, j, -SCF_FD_H),
            [1, 1, 3],
            f,
            Jk::Gdf,
            &grid,
        );
        let fd = (ep - em) / (2.0 * SCF_FD_H);
        let miss = (s.de_deps[i][j] - fd).abs();
        eprintln!(
            "{f} RS-GDF uniform 24^3 eps({i},{j}): analytic {:.9e} FD {fd:.9e} miss {miss:.2e}",
            s.de_deps[i][j]
        );
        worst = worst.max(miss);
        fds.push(((i, j), fd));
    }
    eprintln!("{f} RS-GDF: |analytic - FD of SCF energy| {worst:.2e}");
    assert!(worst < SCF_FD_BAR, "{f}: {worst:e}");
    if !mutants {
        return;
    }
    for m in [
        KXcStressMutation::NoTauStrain,
        KXcStressMutation::TauDoubled,
        KXcStressMutation::TauNoVxc,
    ] {
        let t = stress(&fx, Some(m));
        let mm = fds.iter().fold(0.0_f64, |w, &((i, j), fd)| {
            w.max((t.de_deps[i][j] - fd).abs())
        });
        eprintln!("  {f} RS-GDF {m:?}: FD miss {mm:.2e}");
        assert!(mm > MUTANT_BAR, "{f} RS-GDF {m:?} survives the FD: {mm:e}");
    }
}

/// Fit-error transfer: `(sigma_RSGDF - sigma_dense)` for SCAN equals the same
/// difference for PBE on the same cell, aux and grid, up to the (small)
/// density dependence of the fit error. Bar: `TRANSFER_BAR` (measured below);
/// a tau defect on the RS-GDF route alone must break it. A tau defect common
/// to BOTH routes cancels in the difference: this test sees only
/// route-dependent defects, and the FD test above is the load-bearing anchor.
#[test]
fn rsgdf_minus_dense_stress_transfers_from_pbe_to_scan() {
    let grid = atom_grid(20, 50, Some(CUTOFF));
    let route_diff = |f: &str, m: Option<KXcStressMutation>| {
        let g = stress(&fixture(&H2, [1, 1, 3], f, Jk::Gdf, grid.clone()), m);
        let d = stress(&fixture(&H2, [1, 1, 3], f, Jk::Dense, grid.clone()), None);
        let mut o = g.de_deps;
        for i in 0..3 {
            for j in 0..3 {
                o[i][j] -= d.de_deps[i][j];
            }
        }
        o
    };
    let pbe = route_diff("PBE", None);
    let scan = route_diff("SCAN", None);
    let gap = mdiff(&scan, &pbe);
    eprintln!(
        "RS-GDF - dense: |PBE| {:.3e}, |SCAN| {:.3e}, |SCAN - PBE| {gap:.2e}",
        mmax(&pbe),
        mmax(&scan)
    );
    // The route difference is resolved (the RS-GDF route really ran) and the
    // transfer holds far below it.
    assert!(mmax(&pbe) > 10.0 * TRANSFER_BAR, "{:e}", mmax(&pbe));
    assert!(gap < TRANSFER_BAR, "{gap:e}");
    for m in [KXcStressMutation::NoTauStrain, KXcStressMutation::TauNoVxc] {
        let mm = mdiff(&route_diff("SCAN", Some(m)), &pbe);
        eprintln!("  RS-GDF-only {m:?}: |SCAN - PBE| {mm:.2e}");
        assert!(mm > MUTANT_BAR, "{m:?} survives the transfer: {mm:e}");
    }
}

/// Measured (H2/STO-3G, 1x1x3, atom grid (20, 50), et-sp aux, omega = 1):
/// route difference max |RS-GDF - dense| 7.08e-5 (PBE) / 7.10e-5 (SCAN),
/// |SCAN - PBE| 3.0e-7; RS-GDF-only mutants NoTauStrain 5.2e-2, TauNoVxc
/// 2.4e-2. Bar 10x the measured gap, 8000x below the smallest mutant.
const TRANSFER_BAR: f64 = 3e-6;
