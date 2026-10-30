//! Shared fixtures and helpers of `pbc_kuxcstress_ro_rsgdf.rs` (per commit)
//! and `validation_pbc_kuxcstress_ro_rsgdf.rs` (weekly): the H3 / H2 k-ROHF /
//! ROKS RS-GDF SCF, stress and force drivers, the range-split switch and the
//! mutant lists.
#![allow(dead_code, unused_imports)]

use crate::common::*;
pub use ferric_core::basis::{BasisSet, Shell};
pub use ferric_pbc::dense_aft::ExxDiv;
pub use ferric_pbc::dft::PeriodicGridConfig;
pub use ferric_pbc::hcore::kpoint::periodic_hcore_kpts;
pub use ferric_pbc::hcore::PeriodicHcoreConfig;
pub use ferric_pbc::kgrad::{
    kpoint_rhf_gradient, KGradConfig, KGradJk, KGradMutation, KRsGdfGradSource,
};
pub use ferric_pbc::kpts::KPointMesh;
pub use ferric_pbc::kroks::{
    solve_krohf, solve_krohf_keep, solve_kroks, KRohfConfig, KRoksConfig, KRoksResult,
};
pub use ferric_pbc::kscf::{solve_krhf, KJkKind, KRhfConfig};
pub use ferric_pbc::kstress::{kpoint_rhf_stress, KStressConfig, KStressJk, KStressMutation};
pub use ferric_pbc::kuxcgrad::{
    kpoint_rohf_gradient, kpoint_roks_gradient, KUSpinMutation, KUksGradConfig, KUksGradient,
};
pub use ferric_pbc::kuxcstress::{
    kpoint_rohf_stress, kpoint_roks_stress, KUStressMutation, KUksStress, KUksStressConfig,
};
pub use ferric_pbc::kxcgrad::KRksGradConfig;
pub use ferric_pbc::kxcgrad::KXcGrid;
pub use ferric_pbc::kxcstress::KRksStressConfig;
pub use ferric_pbc::lattice::Cell;
pub use ferric_pbc::rsgdf::kpoint::{KRsGdf, KRsGdfConfig};
pub use ferric_pbc::rsgdf::RsGdfConfig;
pub use ferric_pbc::stress::Mat3;
pub use std::collections::HashMap;

pub const OMEGA: f64 = 0.8;

thread_local! {
    /// `true`: the orbital basis is cc-pVDZ H (s contraction sharing primitives
    /// with an s singlet, plus a p shell), so the default SR column rotation of
    /// the RS-GDF / hcore builds is ACTIVE (it does nothing on STO-3G). Set per
    /// test thread.
    pub static POLARIZED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// `Some(lambda)`: every RS-GDF build and config of this test thread (SCF,
    /// force, stress) runs the range split at `lambda`. Set per test thread.
    pub static SPLIT: std::cell::Cell<Option<f64>> = const { std::cell::Cell::new(None) };
}

pub fn orbital_basis() -> BasisSet {
    if POLARIZED.with(|p| p.get()) {
        ferric_core::basis::bundled("cc-pvdz").expect("cc-pvdz")
    } else {
        pyscf_sto3g_h()
    }
}
pub const AMPLE: usize = 1 << 31;
pub const CUTOFF: f64 = 14.0;
pub const H3: [[f64; 3]; 3] = [[0.3, 0.2, 0.1], [0.35, 0.12, 1.5], [0.3, 1.9, 0.4]];
pub const H2: [[f64; 3]; 2] = [[0.3, 0.2, 0.1], [0.35, 0.12, 1.5]];
pub const FD_H: f64 = 2.5e-5;
pub const FD_BAR: f64 = 3e-8;
pub const ALL9: [(usize, usize); 9] = [
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

pub fn mesh_of(cell: &Cell, n: [usize; 3]) -> KPointMesh {
    KPointMesh::gamma_centred(cell, n).unwrap()
}

pub fn grid_cfg() -> PeriodicGridConfig {
    let mut g = PeriodicGridConfig::with_size(20, 50);
    g.neighbour_cutoff = Some(CUTOFF);
    g
}

pub fn cell_of(pos: &[[f64; 3]]) -> Cell {
    let mut mol = hydrogens(pos);
    mol.multiplicity = if pos.len() % 2 == 1 { 2 } else { 1 };
    Cell::new(mol, cubic(4.0)).expect("cell")
}

pub fn kgdf_cfg() -> KRsGdfConfig {
    KRsGdfConfig {
        gdf: RsGdfConfig {
            omega: 1.0,
            exxdiv: ExxDiv::None,
            budget_bytes: Some(AMPLE),
            range_split: SPLIT
                .with(|s| s.get())
                .map(ferric_pbc::rsgdf::RangeSplit::new),
            ..Default::default()
        },
        mutation: None,
    }
}

pub fn et_sp_aux() -> BasisSet {
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

pub fn rohf_cfg(cell: &Cell, exxdiv: ExxDiv) -> KRohfConfig {
    let mut c = KRohfConfig::for_cell(cell, exxdiv);
    c.hcore = PeriodicHcoreConfig::with_omega(OMEGA);
    c.scf.energy_conv = 1e-13;
    c.scf.grad_conv = 1e-10;
    c.jk = KJkKind::RsGdf;
    c.rsgdf = kgdf_cfg();
    c
}

pub fn scf_on(cell: &Cell, n: [usize; 3], f: Option<&str>, exxdiv: ExxDiv) -> KRoksResult {
    let prep = prep_for(cell, &orbital_basis());
    let aux = prep_for(cell, &et_sp_aux());
    let mesh = mesh_of(cell, n);
    let rc = rohf_cfg(cell, exxdiv);
    let r = match f {
        None => solve_krohf(cell, &prep, Some(&aux), &mesh, &rc),
        Some(f) => {
            let mut c = KRoksConfig::new(cell, exxdiv, f);
            c.rohf = rc;
            c.grid = grid_cfg();
            solve_kroks(cell, &prep, Some(&aux), &mesh, &c)
        }
    }
    .unwrap_or_else(|e| panic!("RS-GDF k-ROHF/ROKS {f:?}: {e}"));
    assert!(r.scf.converged);
    r
}

pub fn stress_of(
    cell: &Cell,
    n: [usize; 3],
    f: Option<&str>,
    exxdiv: ExxDiv,
    res: &KRoksResult,
    hm: Option<KStressMutation>,
    sm: Option<KUStressMutation>,
) -> KUksStress {
    let prep = prep_for(cell, &orbital_basis());
    let aux = prep_for(cell, &et_sp_aux());
    let mesh = mesh_of(cell, n);
    let hcfg = PeriodicHcoreConfig::with_omega(OMEGA);
    let hk = periodic_hcore_kpts(cell, &prep, &mesh, &hcfg).unwrap();
    let kc = kgdf_cfg();
    let gdf = KRsGdf::build(cell, &prep, &aux, &mesh, &hk.s, &kc)
        .unwrap()
        .with_exxdiv(exxdiv);
    let cfg = KUksStressConfig {
        base: KRksStressConfig {
            stress: KStressConfig {
                budget_bytes: Some(AMPLE),
                mutation: hm,
                ..Default::default()
            },
            grid: KXcGrid::Atom(grid_cfg()),
            exxdiv: Some(exxdiv),
            ..Default::default()
        },
        spin_mutation: sm,
    };
    let jk = KStressJk::RsGdf(KRsGdfGradSource {
        gdf: &gdf,
        cfg: &kc,
        aux: &aux,
    });
    match f {
        None => kpoint_rohf_stress(cell, &prep, &mesh, &hcfg, &hk, jk, &res.scf, exxdiv, &cfg),
        Some(f) => kpoint_roks_stress(cell, &prep, &mesh, &hcfg, &hk, jk, &res.scf, f, &cfg),
    }
    .expect("RS-GDF RO stress")
}

pub fn mdiff(a: &Mat3, b: &Mat3) -> f64 {
    let mut m = 0.0_f64;
    for i in 0..3 {
        for j in 0..3 {
            m = m.max((a[i][j] - b[i][j]).abs());
        }
    }
    m
}

pub fn strained(cell: &Cell, i: usize, j: usize, h: f64) -> Cell {
    let mut e = [[0.0; 3]; 3];
    e[i][j] = h;
    cell.strained(&e).expect("strained cell")
}

pub fn fd_case(f: Option<&str>, exxdiv: ExxDiv, n: [usize; 3], comps: &[(usize, usize)]) {
    let cell = cell_of(&H3);
    let res = scf_on(&cell, n, f, exxdiv);
    let s = stress_of(&cell, n, f, exxdiv, &res, None, None);
    let mut worst = 0.0_f64;
    for &(i, j) in comps {
        let ep = scf_on(&strained(&cell, i, j, FD_H), n, f, exxdiv)
            .scf
            .energy;
        let em = scf_on(&strained(&cell, i, j, -FD_H), n, f, exxdiv)
            .scf
            .energy;
        let fd = (ep - em) / (2.0 * FD_H);
        let miss = (s.de_deps[i][j] - fd).abs();
        eprintln!(
            "RS-GDF {f:?} {exxdiv:?} eps({i},{j}): analytic {:.9e} FD {fd:.9e} miss {miss:.2e}",
            s.de_deps[i][j]
        );
        worst = worst.max(miss);
    }
    assert!(worst < FD_BAR, "{f:?} {exxdiv:?}: {worst:e}");
}

// ------------------------------------------------------------------ forces

pub fn force_of(
    cell: &Cell,
    n: [usize; 3],
    f: Option<&str>,
    exxdiv: ExxDiv,
    res: &KRoksResult,
    gm: Option<KGradMutation>,
    sm: Option<KUSpinMutation>,
) -> KUksGradient {
    let prep = prep_for(cell, &orbital_basis());
    let aux = prep_for(cell, &et_sp_aux());
    let mesh = mesh_of(cell, n);
    let hcfg = PeriodicHcoreConfig::with_omega(OMEGA);
    let hk = periodic_hcore_kpts(cell, &prep, &mesh, &hcfg).unwrap();
    let kc = kgdf_cfg();
    let gdf = KRsGdf::build(cell, &prep, &aux, &mesh, &hk.s, &kc)
        .unwrap()
        .with_exxdiv(exxdiv);
    let cfg = KUksGradConfig {
        base: KRksGradConfig {
            grad: KGradConfig {
                budget_bytes: Some(AMPLE),
                mutation: gm,
                ..Default::default()
            },
            grid: KXcGrid::Atom(grid_cfg()),
            exxdiv: Some(exxdiv),
            ..Default::default()
        },
        spin_mutation: sm,
    };
    let jk = KGradJk::RsGdf(KRsGdfGradSource {
        gdf: &gdf,
        cfg: &kc,
        aux: &aux,
    });
    match f {
        None => kpoint_rohf_gradient(cell, &prep, &mesh, &hcfg, &hk, jk, &res.scf, exxdiv, &cfg),
        Some(f) => kpoint_roks_gradient(cell, &prep, &mesh, &hcfg, &hk, jk, &res.scf, f, &cfg),
    }
    .expect("RS-GDF RO force")
}

pub fn moved(pos: &[[f64; 3]], a: usize, x: usize, h: f64) -> Vec<[f64; 3]> {
    let mut p = pos.to_vec();
    p[a][x] += h;
    p
}

pub const FD_H_F: f64 = 5e-5;
pub const FD_BAR_F: f64 = 1e-8;

pub fn force_fd_case(f: Option<&str>, exxdiv: ExxDiv, comps: &[(usize, usize)]) {
    let n = [1, 1, 3];
    let cell = cell_of(&H3);
    let res = scf_on(&cell, n, f, exxdiv);
    let g = force_of(&cell, n, f, exxdiv, &res, None, None);
    let e = |p: &[[f64; 3]]| scf_on(&cell_of(p), n, f, exxdiv).scf.energy;
    let mut worst = 0.0_f64;
    for &(a, x) in comps {
        let fd = (e(&moved(&H3, a, x, FD_H_F)) - e(&moved(&H3, a, x, -FD_H_F))) / (2.0 * FD_H_F);
        let miss = (g.grad[(a, x)] - fd).abs();
        eprintln!(
            "RS-GDF force {f:?} {exxdiv:?} ({a},{x}): analytic {:.9e} FD {fd:.9e} miss {miss:.2e}",
            g.grad[(a, x)]
        );
        worst = worst.max(miss);
    }
    assert!(worst < FD_BAR_F, "{f:?} {exxdiv:?}: {worst:e}");
}

// --------------------------------------------- rotated (p-shell) fixture

/// Step and bars of the rotating-basis test. cc-pVDZ energies carry ~1e-12 of
/// SCF/screening noise, so the FD miss is noise-dominated and FALLS with h
/// (force probe, (0,2)/(2,1): 3.3e-8/6.2e-8 at h = 2.5e-5, 1.4e-8/4.2e-8 at
/// 5e-5, 3.1e-9/7.9e-9 at 1e-4, 6.9e-9/6.0e-9 at 2e-4 where h^2 truncation
/// starts); stress (2,2) at h = 1.25e-5 .. 2e-4: 2.3e-7, 1.4e-7, 8.4e-8, 3.9e-7, 1.6e-6 (noise
/// below, h^2 truncation above; (0,2) <= 4.5e-9 from 5e-5). So force h = 1e-4 with
/// bar 5e-8 and stress h = 5e-5 with bar 3e-7, still >= 3 orders below any mutant.
pub const ROT_H: f64 = 1e-4;
pub const ROT_HS: f64 = 5e-5;
pub const ROT_BAR_F: f64 = 5e-8;
pub const ROT_BAR_S: f64 = 3e-7;

// ------------------------------------------- range-split RS-GDF (lambda = 1)
//
// The RO force and stress run the same fit pass (`kpoint_fit_gradient`,
// `kstress::assemble`) as the RHF/UKS ones, which differentiate a range-split
// build along its partition. PHYSICS hypothesis: the RO split force/stress is
// the derivative of the split energy (FD at the unsplit floor) and its
// closed-shell limit is the RHF split result to round-off. ARTIFACT
// hypotheses: (1) "split passes because the unsplit code ran" -- guarded by
// `assert_split_ran` (moved primitives on both sides) and by the split-only
// mutants, which delete only split terms and must miss FD; (2) "an RO-specific
// split term is missing" -- the closed-shell limit is blind to `W_RO`, so the
// open-shell H3 FD carries it.

/// lambda = 1 at omega = 1 moves the aux primitives s(0.75), s(0.3), p(0.5)
/// and the STO-3G orbital primitive 0.1689 (as `pbc_kstress.rs`).
pub const SPLIT_LAMBDA: f64 = 1.0;

/// Panics unless this thread's RS-GDF build runs the range split and moves
/// orbital and aux primitives.
pub fn assert_split_ran(cell: &Cell, n: [usize; 3]) {
    let prep = prep_for(cell, &orbital_basis());
    let aux = prep_for(cell, &et_sp_aux());
    let mesh = mesh_of(cell, n);
    let hk =
        periodic_hcore_kpts(cell, &prep, &mesh, &PeriodicHcoreConfig::with_omega(OMEGA)).unwrap();
    let kc = kgdf_cfg();
    assert!(kc.gdf.range_split.is_some(), "SPLIT not set on this thread");
    let gdf = KRsGdf::build(cell, &prep, &aux, &mesh, &hk.s, &kc).unwrap();
    let moved = |name: &str| {
        gdf.stats()
            .split_counters
            .iter()
            .find(|(c, _)| *c == name)
            .map_or(0, |&(_, v)| v)
    };
    let (o, a) = (
        moved("rsgdf split orbital prims smooth"),
        moved("rsgdf split aux prims smooth"),
    );
    eprintln!("split build: moved orbital {o}, aux {a}");
    assert!(o > 0 && a > 0, "split path did not move primitives");
}

/// The split-only stress terms (deleting one changes only the split build).
pub const SPLIT_STRESS_MUTANTS: [KStressMutation; 6] = [
    KStressMutation::SplitNoSrKernelStrain,
    KStressMutation::SplitNoSmoothPair,
    KStressMutation::SplitFullG0,
    KStressMutation::SplitNoSmoothOverlap,
    KStressMutation::SplitFlipSrSlope,
    KStressMutation::SplitNoSmoothVolume,
];

/// The split-only force terms.
pub const SPLIT_FORCE_MUTANTS: [KGradMutation; 3] = [
    KGradMutation::SplitGammaKernel,
    KGradMutation::SplitNoSmoothOverlap,
    KGradMutation::SplitFullG0,
];
