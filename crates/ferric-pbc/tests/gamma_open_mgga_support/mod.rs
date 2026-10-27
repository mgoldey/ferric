//! Shared fixtures of the Gamma-point open-shell meta-GGA energy tests
//! (`pbc_gamma_open_mgga.rs` per commit, `validation_pbc_gamma_open_mgga.rs`
//! weekly): the polarized H3 doublet, a dense-AFT Gamma setup, UKS / ROKS
//! runs with a caller-built (optionally mutated) `PeriodicXc`, and the PySCF
//! pins of `scripts/gen_pyscf_gamma_open_mgga_refs.py`.
#![allow(dead_code)]

use super::common::*;
use ferric_core::basis::BasisSet;
use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_pbc::dense_aft::{
    DenseAftEri, ExxDiv, DEFAULT_DENSE_AFT_MAX_BYTES, DEFAULT_DENSE_AFT_PRECISION,
};
use ferric_pbc::dft::{
    gamma_uks_with_xc, GammaMggaMutation, GammaUksConfig, PeriodicGrid, PeriodicGridConfig,
    PeriodicXc, PeriodicXcConfig,
};
use ferric_pbc::hcore::{periodic_hcore, PeriodicHcore, PeriodicHcoreConfig};
use ferric_pbc::lattice::Cell;
use ferric_pbc::rohf::{gamma_roks_with_xc, GammaRoksConfig};
use ferric_pbc::uhf::GammaUhfIntegrals;

pub const OMEGA: f64 = 0.8;
pub const FUNCTIONALS: [&str; 2] = ["SCAN", "R2SCAN"];
pub const MUTANTS: [GammaMggaMutation; 4] = [
    GammaMggaMutation::TauBetaFromAlpha,
    GammaMggaMutation::TauOmitted,
    GammaMggaMutation::SwapTau,
    GammaMggaMutation::DropBetaVtau,
];

/// Three hydrogens in the cubic a = 4 cell: doublet, N_alpha = 2, N_beta = 1
/// (the `h3` cell of `pbc_kuks_mgga.rs` and the PySCF generators).
pub const H3_ATOMS: [[f64; 3]; 3] = [[0.3, 0.2, 0.1], [0.3, 0.2, 1.5], [0.3, 1.9, 0.4]];

pub fn h3_doublet() -> Cell {
    let mut mol: Molecule = hydrogens(&H3_ATOMS);
    mol.multiplicity = 2;
    Cell::new(mol, cubic(4.0)).expect("cell")
}

/// Everything a Gamma SCF needs on one cell.
pub struct Setup {
    pub cell: Cell,
    pub bs: BasisSet,
    pub prep: PreparedBasis,
    pub hc: PeriodicHcore,
    pub eri: DenseAftEri,
}

pub fn setup(cell: Cell) -> Setup {
    let bs = pyscf_sto3g_h();
    let prep = prep_for(&cell, &bs);
    let hc = periodic_hcore(&cell, &prep, &PeriodicHcoreConfig::with_omega(OMEGA)).unwrap();
    let eri = DenseAftEri::build(
        &cell,
        &prep,
        &hc.s,
        ExxDiv::None,
        DEFAULT_DENSE_AFT_PRECISION,
        DEFAULT_DENSE_AFT_MAX_BYTES,
    )
    .unwrap();
    Setup {
        cell,
        bs,
        prep,
        hc,
        eri,
    }
}

pub fn grid_cfg(n_rad: usize, n_ang: usize) -> PeriodicGridConfig {
    PeriodicGridConfig::with_size(n_rad, n_ang)
}

pub fn uks_cfg(functional: &str, grid: PeriodicGridConfig) -> GammaUksConfig {
    let mut cfg = GammaUksConfig::new(functional);
    cfg.grid = grid;
    cfg.scf.energy_conv = 1e-12;
    cfg
}

pub fn roks_cfg(functional: &str, grid: PeriodicGridConfig) -> GammaRoksConfig {
    let mut cfg = GammaRoksConfig::new(functional);
    cfg.grid = grid;
    cfg.scf.energy_conv = 1e-12;
    cfg
}

pub fn ints(su: &Setup) -> GammaUhfIntegrals<'_> {
    GammaUhfIntegrals::DenseAft(&su.eri)
}

/// A [`PeriodicXc`] on `grid`, optionally carrying a tau mutant.
pub fn open_xc(
    su: &Setup,
    grid: &PeriodicGrid,
    f: &str,
    m: Option<GammaMggaMutation>,
) -> PeriodicXc {
    let x =
        PeriodicXc::new_open_mgga(&su.cell, &su.bs, f, grid, &PeriodicXcConfig::default()).unwrap();
    match m {
        Some(m) => x.with_mgga_mutation(m),
        None => x,
    }
}

/// `(E_tot, E_xc)` of the Gamma UKS with the caller's XC builder (the
/// config's grid is ignored: the builder carries its own).
pub fn uks_with(su: &Setup, xc: &mut PeriodicXc, f: &str) -> Result<(f64, f64), FerricError> {
    let cfg = uks_cfg(f, PeriodicGridConfig::default());
    let r = gamma_uks_with_xc(&su.cell, &su.prep, &su.hc, ints(su), xc, &cfg)?;
    Ok((r.scf.energy, r.e_xc))
}

/// `(E_tot, E_xc)` of the Gamma ROKS with the caller's XC builder.
pub fn roks_with(su: &Setup, xc: &mut PeriodicXc, f: &str) -> Result<(f64, f64), FerricError> {
    let cfg = roks_cfg(f, PeriodicGridConfig::default());
    let r = gamma_roks_with_xc(&su.cell, &su.prep, &su.hc, ints(su), xc, &cfg)?;
    Ok((r.scf.energy, r.e_xc))
}

/// Miss of a mutant run against `e_ref`; a refused or non-finite run counts
/// as detected (infinite miss).
pub fn mutant_miss(r: Result<(f64, f64), FerricError>, e_ref: f64) -> f64 {
    match r {
        Ok((e, _)) if e.is_finite() => (e - e_ref).abs(),
        _ => f64::INFINITY,
    }
}

/// Run `route` (`"UKS"` or `"ROKS"`) with the caller's XC builder.
pub fn run_route(
    route: &str,
    su: &Setup,
    xc: &mut PeriodicXc,
    f: &str,
) -> Result<(f64, f64), FerricError> {
    match route {
        "UKS" => uks_with(su, xc, f),
        _ => roks_with(su, xc, f),
    }
}

/// The PySCF Gamma `pbc.dft.UKS` / `ROKS` energy of `row` (e.g. `"UKS SCAN"`).
pub fn pyscf_ref(row: &str) -> f64 {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../testdata/reference/pbc_h3_sto3g_gamma_open_mgga.json"
    );
    let doc: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path).expect(path)).unwrap();
    doc["rows"][row]["e_tot"]
        .as_f64()
        .unwrap_or_else(|| panic!("row {row}"))
}
