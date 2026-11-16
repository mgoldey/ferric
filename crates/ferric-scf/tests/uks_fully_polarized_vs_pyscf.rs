//! UKS on fully spin-polarized densities (ρ_β = 0 everywhere, ζ = 1) against
//! PySCF 2.13.
//!
//! `validation_ks_energies.rs` covers radicals whose β density is non-zero
//! (NH2, CH3, HO2, O2). `uks.rs` has the H atom, but at a 5e-4 Ha bar on a
//! different grid. This file pins the ζ = 1 limit at the cross-code PBE floor.
//!
//! Systems: H2 triplet (r = 0.74 Å) and the H atom, STO-3G, UKS PBE. With
//! STO-3G and every electron α there is one determinant, so there is no
//! state-selection ambiguity. Both are homonuclear or atomic, so the Becke
//! atomic-size adjustment (ferric: Becke 1988, PySCF default: Treutler) does
//! not enter.
//!
//! PySCF recipe: `dft.UKS(mol); xc = "pbe"; grids.atom_grid = (75, 110);
//! grids.prune = None; conv_tol = 1e-13`, exact J (`exact` rows) or
//! `density_fit(auxbasis = "def2-universal-jkfit")` (`df` row).
//!
//! # Physics vs artifact hypothesis
//!
//! * If the ζ = 1 path is right, ferric matches PySCF at the floor the
//!   closed-shell PBE tests measure (2.1e-8 Ha). Measured here: H2 triplet
//!   exact J 6.7e-9, RI-J 6.5e-9; H atom 4e-10.
//! * If ferric mishandles ρ_β = 0 (a floor on ρ_β or σ_ββ, a wrong ζ = 1
//!   branch, libxc fed a non-zero β channel), the energies move. These
//!   systems are insensitive to small floors: feeding libxc
//!   max(ρ_β, 1e-10) moves the H2 triplet by 4.4e-8 Ha and the H atom by
//!   2.4e-8 Ha, so the bar has to sit at the cross-code floor, not above it.
//!
//! # Bar
//!
//! 2e-8 Ha: the cross-code PBE floor of the closed-shell tests (2.1e-8 Ha,
//! `validation_ks_energies.rs` module doc), 3x the largest residual measured
//! here (6.7e-9). SCF convergence (energy_conv 1e-11) contributes below
//! 1e-10. RI-J runs in f64: the f32 storage default rounds the energy by
//! 1e-7..1e-6 Ha, above the bar.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_dft::grid::AtomicGridConfig;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::df_j::RiJPrecision;
use ferric_scf::rhf::RhfConfig;
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::solve_uhf;

const BAR: f64 = 2e-8;

fn uks_pbe(xyz: &str, mult: usize, df_j_aux: Option<&str>) -> f64 {
    let mol = Molecule::parse_xyz(xyz, 0, mult).unwrap();
    // The premise of this file: no β electrons at all.
    assert_eq!(mult as i32 - 1, mol.nelec(), "{xyz}: not fully polarized");
    let bs = basis::bundled("sto-3g").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
    let cfg = RhfConfig {
        xc: Some("PBE".into()),
        df_j_aux: df_j_aux.map(str::to_string),
        ri_j_precision: Some(RiJPrecision::F64),
        energy_conv: 1e-11,
        density_conv: 1e-9,
        max_iter: 200,
        dft_grid: Some(AtomicGridConfig::default()),
        ..Default::default()
    };
    let res = solve_uhf(&ParallelContext::default(), &mol, &prep, &bounds, &cfg).unwrap();
    assert!(res.converged, "{xyz}: SCF did not converge");
    res.energy
}

fn check(ctx: &str, got: f64, want: f64) {
    let d = (got - want).abs();
    eprintln!("{ctx}: ferric {got:.10} PySCF {want:.10} |d| {d:.2e} (bar {BAR:.0e})");
    assert!(d <= BAR, "{ctx}: |E - E_pyscf| = {d:.3e} > {BAR:.0e}");
}

const H2_TRIPLET: &str = "2\nH2\nH 0 0 0\nH 0 0 0.74\n";

#[test]
fn h2_triplet_sto3g_pbe_exact_j_matches_pyscf() {
    check(
        "H2 triplet exact J",
        uks_pbe(H2_TRIPLET, 3, None),
        -0.537_002_898_0,
    );
}

#[test]
fn h2_triplet_sto3g_pbe_ri_j_matches_pyscf_density_fit() {
    check(
        "H2 triplet RI-J",
        uks_pbe(H2_TRIPLET, 3, Some("def2-universal-jkfit")),
        -0.538_097_881_8,
    );
}

#[test]
fn h_atom_sto3g_pbe_exact_j_matches_pyscf() {
    check(
        "H atom exact J",
        uks_pbe("1\nH\nH 0 0 0\n", 2, None),
        -0.464_375_663_3,
    );
}
