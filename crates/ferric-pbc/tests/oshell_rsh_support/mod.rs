//! Shared setup of `pbc_oshell_rsh.rs` (always-on) and
//! `validation_oshell_rsh.rs` (weekly): cells, Gamma / k-point runs, the PySCF
//! pins (read from `testdata/reference/pbc_oshell_sto3g_rsh.json`, so a pin is
//! the generator's full-precision number) and the bars.
#![allow(dead_code)]

use crate::common::*;
use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_pbc::dense_aft::{
    DenseAftEri, ExxDiv, DEFAULT_DENSE_AFT_MAX_BYTES, DEFAULT_DENSE_AFT_PRECISION,
};
use ferric_pbc::dft::{
    gamma_rks_rsh_on_grid, gamma_uks_rsh_on_grid, GammaRksConfig, GammaUksConfig, GammaUksResult,
    PeriodicGrid,
};
use ferric_pbc::gamma_rsh::{GammaRshExchange, GammaSpinMutant};
use ferric_pbc::hcore::{periodic_hcore, PeriodicHcoreConfig};
use ferric_pbc::kdense_aft::KDenseAftConfig;
use ferric_pbc::kdft::{solve_krks_on_grid, KRksConfig};
use ferric_pbc::kpts::KPointMesh;
use ferric_pbc::kscf::KJkKind;
use ferric_pbc::kuks::{solve_kuks_on_grid, KUksConfig, KUksResult};
use ferric_pbc::kuscf::KUhfMutation;
use ferric_pbc::lattice::Cell;
use ferric_pbc::rsgdf::kpoint::KRsGdfConfig;
use ferric_pbc::rsgdf::{RsGdf, RsGdfConfig};
use ferric_pbc::rsh::resolve_periodic_functional_rsh;
use ferric_pbc::uhf::GammaUhfIntegrals;

pub const OMEGA_HCORE: f64 = 0.8;

pub const MESH: usize = 41;

pub const AMPLE: usize = 1 << 30;

pub const AUX: &str = "cc-pvdz-ri";

/// The H3 doublet / H4 triplet cells of `scripts/gen_pyscf_oshell_rsh_refs.py`
/// (TRICLINIC lattice `TRI_A`); `H2_ATOMS` is the closed-shell cell.
pub const H3D: [[f64; 3]; 3] = [[0.3, 0.2, 0.1], [0.3, 0.2, 1.5], [0.3, 0.2, 2.9]];

pub const H4T: [[f64; 3]; 4] = [
    [0.3, 0.2, 0.1],
    [0.3, 0.2, 1.5],
    [1.6, 1.1, 2.9],
    [1.6, 1.1, 4.1],
];

pub fn name_of(f: &str) -> &'static str {
    match f {
        "HSE06" => "HSE06",
        _ => "HYB_GGA_XC_CAM_B3LYP",
    }
}

pub fn cell_of(atoms: &[[f64; 3]], mult: usize) -> Cell {
    let mut mol: Molecule = hydrogens(atoms);
    mol.multiplicity = mult;
    Cell::new(mol, TRI_A).expect("cell")
}

pub fn aux_of(cell: &Cell) -> PreparedBasis {
    PreparedBasis::new(cell.mol(), &basis::bundled(AUX).unwrap()).unwrap()
}

/// How a Gamma run is set up.
#[derive(Clone, Copy, Default)]
pub struct GRun {
    /// RS-GDF (J and K fit on `AUX`) instead of dense AFT.
    pub rsgdf: bool,
    pub spin_mutant: Option<GammaSpinMutant>,
    /// Replace the range-separation parameters (a mutant; the exchange is
    /// then built `unchecked`).
    pub rsh: Option<ferric_pbc::rsh::RshParams>,
}

pub fn rsh_of(f: &str) -> ferric_pbc::rsh::RshParams {
    resolve_periodic_functional_rsh(name_of(f))
        .unwrap()
        .2
        .unwrap()
}

/// The exchange, the J source and the hcore of a Gamma run: shared by the
/// RKS and UKS drivers so a closed-shell comparison differs only in the SCF.
pub struct GSetup {
    pub prep: PreparedBasis,
    pub hc: ferric_pbc::hcore::PeriodicHcore,
    pub ex: GammaRshExchange,
    pub dense: Option<DenseAftEri>,
    pub gdf: Option<RsGdf>,
    pub grid: PeriodicGrid,
}

impl GSetup {
    pub fn new(cell: &Cell, f: &str, exxdiv: ExxDiv, run: GRun) -> Self {
        let bs = pyscf_sto3g_h();
        let prep = prep_for(cell, &bs);
        let hc =
            periodic_hcore(cell, &prep, &PeriodicHcoreConfig::with_omega(OMEGA_HCORE)).unwrap();
        let grid = PeriodicGrid::uniform(cell, [MESH; 3]).unwrap();
        let rsh = run.rsh.unwrap_or_else(|| rsh_of(f));
        let (mut ex, dense, gdf);
        if run.rsgdf {
            let aux = aux_of(cell);
            let gcfg = RsGdfConfig {
                omega: OMEGA_HCORE,
                exxdiv,
                budget_bytes: Some(AMPLE),
                ..Default::default()
            };
            let kcfg = KRsGdfConfig {
                gdf: gcfg,
                mutation: None,
            };
            ex = GammaRshExchange::build_rsgdf(cell, &prep, &aux, &hc.s, exxdiv, &kcfg, rsh)
                .unwrap();
            gdf = Some(RsGdf::build(cell, &prep, &aux, &hc.s, &gcfg).unwrap());
            dense = None;
        } else {
            let kcfg = KDenseAftConfig::default();
            ex = GammaRshExchange::build_dense(cell, &prep, &hc.s, exxdiv, &kcfg, rsh).unwrap();
            dense = Some(
                DenseAftEri::build(
                    cell,
                    &prep,
                    &hc.s,
                    exxdiv,
                    DEFAULT_DENSE_AFT_PRECISION,
                    DEFAULT_DENSE_AFT_MAX_BYTES,
                )
                .unwrap(),
            );
            gdf = None;
        }
        if let Some(m) = run.spin_mutant {
            ex = ex.with_spin_mutant_for_mutation_tests(m);
        }
        if run.rsh.is_some() {
            ex = ex.unchecked_for_mutation_tests();
        }
        Self {
            prep,
            hc,
            ex,
            dense,
            gdf,
            grid,
        }
    }

    pub fn ints(&self) -> GammaUhfIntegrals<'_> {
        match (&self.dense, &self.gdf) {
            (Some(e), _) => GammaUhfIntegrals::DenseAft(e),
            (_, Some(g)) => GammaUhfIntegrals::RsGdf(g),
            _ => unreachable!(),
        }
    }
}

pub fn gamma_uks(cell: &Cell, f: &str, exxdiv: ExxDiv, run: GRun) -> GammaUksResult {
    let s = GSetup::new(cell, f, exxdiv, run);
    let mut cfg = GammaUksConfig::new(name_of(f));
    cfg.exxdiv = exxdiv;
    cfg.scf.density_conv = 1e-11;
    cfg.scf.energy_conv = 1e-12;
    let r = gamma_uks_rsh_on_grid(cell, &s.prep, &s.hc, s.ints(), &s.ex, &s.grid, &cfg)
        .unwrap_or_else(|e| panic!("gamma_uks_rsh {f} {exxdiv:?}: {e}"));
    assert!(r.scf.converged);
    r
}

pub fn gamma_rks_energy(cell: &Cell, f: &str, exxdiv: ExxDiv, run: GRun) -> f64 {
    let s = GSetup::new(cell, f, exxdiv, run);
    let mut cfg = GammaRksConfig::new(name_of(f));
    cfg.exxdiv = exxdiv;
    cfg.scf.density_conv = 1e-11;
    cfg.scf.energy_conv = 1e-12;
    let r = gamma_rks_rsh_on_grid(cell, &s.prep, &s.hc, s.ints(), &s.ex, &s.grid, &cfg)
        .unwrap_or_else(|e| panic!("gamma_rks_rsh {f} {exxdiv:?}: {e}"));
    assert!(r.scf.converged);
    r.scf.energy
}

/// k-point runs: dense AFT, or RS-GDF on `AUX`.
pub fn kuks_run(
    cell: &Cell,
    f: &str,
    n: [usize; 3],
    exxdiv: ExxDiv,
    rsgdf: bool,
    mutation: Option<KUhfMutation>,
) -> KUksResult {
    let bs = pyscf_sto3g_h();
    let prep = prep_for(cell, &bs);
    let grid = PeriodicGrid::uniform(cell, [MESH; 3]).unwrap();
    let mesh = KPointMesh::gamma_centred(cell, n).unwrap();
    let mut cfg = KUksConfig::new(cell, exxdiv, name_of(f));
    cfg.kuhf.hcore = PeriodicHcoreConfig::with_omega(OMEGA_HCORE);
    cfg.kuhf.scf.energy_conv = 1e-12;
    cfg.kuhf.scf.grad_conv = 1e-9;
    cfg.kuhf.mutation = mutation;
    let aux = rsgdf.then(|| aux_of(cell));
    if rsgdf {
        cfg.kuhf.jk = KJkKind::RsGdf;
        cfg.kuhf.rsgdf = rsgdf_cfg(exxdiv);
    }
    solve_kuks_on_grid(cell, &prep, aux.as_ref(), &mesh, &grid, &cfg)
        .unwrap_or_else(|e| panic!("solve_kuks {f} {n:?} {exxdiv:?}: {e}"))
}

pub fn rsgdf_cfg(exxdiv: ExxDiv) -> KRsGdfConfig {
    KRsGdfConfig {
        gdf: RsGdfConfig {
            omega: OMEGA_HCORE,
            exxdiv,
            budget_bytes: Some(AMPLE),
            ..Default::default()
        },
        mutation: None,
    }
}

pub fn krks_energy(cell: &Cell, f: &str, n: [usize; 3], exxdiv: ExxDiv, rsgdf: bool) -> f64 {
    let bs = pyscf_sto3g_h();
    let prep = prep_for(cell, &bs);
    let grid = PeriodicGrid::uniform(cell, [MESH; 3]).unwrap();
    let mesh = KPointMesh::gamma_centred(cell, n).unwrap();
    let mut cfg = KRksConfig::new(cell, exxdiv, name_of(f));
    cfg.krhf.hcore = PeriodicHcoreConfig::with_omega(OMEGA_HCORE);
    cfg.krhf.scf.energy_conv = 1e-12;
    cfg.krhf.scf.grad_conv = 1e-9;
    let aux = rsgdf.then(|| aux_of(cell));
    if rsgdf {
        cfg.krhf.jk = KJkKind::RsGdf;
        cfg.krhf.rsgdf = rsgdf_cfg(exxdiv);
    }
    solve_krks_on_grid(cell, &prep, aux.as_ref(), &mesh, &grid, &cfg)
        .unwrap()
        .scf
        .energy
}

/// The pinned energies are PySCF's; this is the bar (see the ledger).
pub const PIN_TOL: f64 = 1e-11;

pub fn cell_named(name: &str) -> Cell {
    match name {
        "h3d" => cell_of(&H3D, 2),
        "h4t" => cell_of(&H4T, 3),
        _ => unreachable!(),
    }
}

/// The N-cell supercell of `cell`, `mult` the giant-determinant multiplicity.
pub fn supercell(cell: &Cell, n: [usize; 3], mult: usize) -> (Cell, Vec<[f64; 3]>) {
    let a = *cell.lattice();
    let (mut atoms, mut shifts) = (Vec::new(), Vec::new());
    for m0 in 0..n[0] {
        for m1 in 0..n[1] {
            for m2 in 0..n[2] {
                let t: Vec<f64> = (0..3)
                    .map(|d| m0 as f64 * a[0][d] + m1 as f64 * a[1][d] + m2 as f64 * a[2][d])
                    .collect();
                shifts.push([t[0], t[1], t[2]]);
                for p in cell.positions() {
                    atoms.push([p[0] + t[0], p[1] + t[1], p[2] + t[2]]);
                }
            }
        }
    }
    let mut lat = a;
    for (i, row) in lat.iter_mut().enumerate() {
        row.iter_mut().for_each(|v| *v *= n[i] as f64);
    }
    let mut mol = hydrogens(&atoms);
    mol.multiplicity = mult;
    (Cell::new(mol, lat).expect("supercell"), shifts)
}

/// Gamma UKS with the GLOBAL hybrid PBE0 on the same grid and J source as
/// the RSH runs (the plain-Coulomb reference for fit errors and for the
/// "HSE06 is not PBE0" check).
pub fn gamma_pbe0(cell: &Cell, rsgdf: bool) -> f64 {
    use ferric_pbc::dft::{gamma_uks_with_xc, PeriodicXc, PeriodicXcConfig};
    let s = GSetup::new(
        cell,
        "HSE06",
        ExxDiv::Ewald,
        GRun {
            rsgdf,
            ..Default::default()
        },
    );
    let bs = pyscf_sto3g_h();
    let mut pxc =
        PeriodicXc::new(cell, &bs, "PBE0", &s.grid, &PeriodicXcConfig::default()).unwrap();
    let mut cfg = GammaUksConfig::new("PBE0");
    cfg.scf.density_conv = 1e-11;
    cfg.scf.energy_conv = 1e-12;
    gamma_uks_with_xc(cell, &s.prep, &s.hc, s.ints(), &mut pxc, &cfg)
        .unwrap()
        .scf
        .energy
}

pub fn kuks_pbe0(cell: &Cell, n: [usize; 3], rsgdf: bool) -> f64 {
    let bs = pyscf_sto3g_h();
    let prep = prep_for(cell, &bs);
    let grid = PeriodicGrid::uniform(cell, [MESH; 3]).unwrap();
    let mesh = KPointMesh::gamma_centred(cell, n).unwrap();
    let mut cfg = KUksConfig::new(cell, ExxDiv::Ewald, "PBE0");
    cfg.kuhf.hcore = PeriodicHcoreConfig::with_omega(OMEGA_HCORE);
    cfg.kuhf.scf.energy_conv = 1e-12;
    cfg.kuhf.scf.grad_conv = 1e-9;
    let aux = rsgdf.then(|| aux_of(cell));
    if rsgdf {
        cfg.kuhf.jk = KJkKind::RsGdf;
        cfg.kuhf.rsgdf = rsgdf_cfg(ExxDiv::Ewald);
    }
    solve_kuks_on_grid(cell, &prep, aux.as_ref(), &mesh, &grid, &cfg)
        .unwrap()
        .scf
        .energy
}

pub fn closed_limit_case(f: &str, x: ExxDiv) {
    let h2 = cell_of(&H2_ATOMS, 1);
    let u = gamma_uks(&h2, f, x, GRun::default());
    let r = gamma_rks_energy(&h2, f, x, GRun::default());
    eprintln!(
        "closed Gamma {f} {x:?}: U-R {:.3e} (<S2> {:.1e})",
        u.scf.energy - r,
        u.s2
    );
    assert!((u.scf.energy - r).abs() < CLOSED_TOL, "{f} {x:?}");
    assert!(u.s2.abs() < 1e-8);
}

/// Measured 0.0 (bit-identical) on all four cells; the bar is the SCF
/// convergence scale, 1e10 below the smallest spin mutant (6.2e-2).
pub const CLOSED_TOL: f64 = 1e-11;

/// Measured shifts on the H3 doublet / HSE06 / ewald Gamma pin (see the
/// ledger in the module doc of the mutation tests): each is the unmutated
/// energy minus the mutant's. A mutant must move E by more than `MUT_MIN`.
pub const MUT_MIN: f64 = 1e-3;

pub const ONE_POINT_TOL: f64 = 1e-11;

pub const ONE_POINT_EXC_TOL: f64 = 1e-10;

pub const SUPERCELL_TOL: f64 = 1e-10;

pub const OMEGA_MUT_MIN: f64 = 1e-4;

const PIN_JSON: &str = include_str!("../../../../testdata/reference/pbc_oshell_sto3g_rsh.json");

fn pin_row(cell: &str, f: &str, n: [usize; 3], x: ExxDiv) -> serde_json::Value {
    let v: serde_json::Value = serde_json::from_str(PIN_JSON).expect("pin JSON");
    let xs = if x == ExxDiv::Ewald { "ewald" } else { "none" };
    v["rows"]
        .as_array()
        .expect("rows")
        .iter()
        .find(|r| {
            r["cell"] == cell
                && r["functional"] == f
                && r["exxdiv"] == xs
                && r["kmesh"] == serde_json::json!(n)
        })
        .unwrap_or_else(|| panic!("no pin {cell} {f} {n:?} {x:?}"))
        .clone()
}

/// PySCF 2.13.1 `pbc.dft.UKS` (Gamma) / `KUKS` (`mf.nelec = (N_a N_k, N_b N_k)`)
/// total energy per cell, AFTDF, UniformGrids 41^3, conv_tol 1e-12,
/// `cell.precision` 1e-11 (`scripts/gen_pyscf_oshell_rsh_refs.py`).
pub fn pin(cell: &str, f: &str, n: [usize; 3], x: ExxDiv) -> f64 {
    pin_row(cell, f, n, x)["e_tot"].as_f64().expect("e_tot")
}

/// PySCF PBE0 on the h3d cell, Gamma, ewald (the same run's global hybrid).
pub fn pbe0_pin() -> f64 {
    pin_row("h3d", "HSE06", [1, 1, 1], ExxDiv::Ewald)["e_pbe0"]
        .as_f64()
        .expect("e_pbe0")
}

/// Singlet H4 (the 4 H of `TRI_ATOMS`, 2 occupied orbitals NOT fixed by a
/// symmetry) in `TRI_A`: the closed-shell anchor cell with SCF freedom.
pub fn cell_h4_singlet() -> Cell {
    triclinic_cell()
}
