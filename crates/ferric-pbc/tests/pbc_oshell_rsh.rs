//! Open-shell range-separated hybrids (UKS), ENERGY only, at Gamma
//! (`gamma_uks_rsh`) and on a k-mesh (`solve_kuks` with HSE06 / CAM-B3LYP),
//! mgoldey/ferric#354.
//!
//! # Hypotheses (written before any number was measured)
//!
//! The attenuated K builder is linear in the density and carries the
//! fractions and the MIXED Madelung constant, so the open-shell step is
//! `K_sigma = K[D_sigma]`, scaled by 1.
//!
//! * Physics: a singlet UKS equals RKS to SCF precision; the 1x1x1 mesh
//!   equals the Gamma UKS; an N-point mesh equals the Gamma UKS of the
//!   supercell / N; PySCF's polarized pins agree.
//! * Artifact: a per-spin error that the closed-shell limit hides (alpha K on
//!   the beta Fock, a dropped beta term, a halved per-spin Madelung constant,
//!   a swapped spin build, the plain instead of the mixed Madelung constant)
//!   leaves every closed-shell anchor intact (D_alpha = D_beta) and moves ONLY
//!   polarized energies. The mutant ledger below is therefore measured on
//!   POLARIZED cells and every mutant must fail the PySCF pin by far more
//!   than its bar.
//!
//! # Measured ledger (2026-10-18, release, 6-core box, PySCF 2.13.1 pins)
//!
//! Agreement (E_ferric - E_PySCF): Gamma H3 doublet HSE06 1.5e-13, H4 triplet 2.7e-13 (HSE06) and 8.1e-13 (CAM-B3LYP); PBE0 on
//! the same cell 3.7e-13; k-mesh rows 1.2e-14 .. 8.1e-13 (1x1x1 and 1x1x2,
//! HSE06 / CAM-B3LYP, ewald / none). Closed-shell UKS - RKS: 0.0 (Gamma,
//! bit-identical) and <= 6.7e-15 (k). 1x1x1 - Gamma: 8.7e-15. 1x1x2 -
//! supercell/2: <= 4.7e-15. RS-GDF - dense: -3.075e-6 (HSE06) vs -3.076e-6
//! (PBE0) at 1x1x2 and -1.730e-5 vs -1.730e-5 at Gamma, i.e. the plain-Coulomb
//! fit error x 1.000. HSE06 - PBE0 on the H3 cell is -2.4e-5 (the erf piece
//! is live here: omega x0.01 moves E by -4.6e-2, unlike H2 in a = 4 Bohr).
//!
//! Mutant shifts E - pin (all on the POLARIZED cell; closed-shell shifts of
//! the spin mutants are 2.2e-16, so the closed-shell anchor cannot see them):
//! Gamma: alpha K on beta -1.39e-2, beta K dropped +6.95e-2, Madelung/2 per
//! spin +9.34e-2; omega x1.01 +4.66e-4, erf piece off -4.61e-2, c_lr := c_sr
//! -4.66e-2, c_sr x0.99 +2.58e-3. k 1x1x2: alpha K on beta -1.99e-2, spin
//! swap +6.17e-2, beta K dropped +6.36e-2, Madelung/2 +4.61e-2, plain
//! (unmixed) Madelung -4.63e-1, K from the total density -1.67e-1.
//! Bars: pins 1e-11 (17x the largest miss, 4.7e7 below the smallest mutant),
//! spin/Madelung mutants must move E by > 1e-3 (smallest measured 1.4e-2),
//! parameter mutants by > 1e-4 (smallest 4.7e-4).
//! No spin swap exists at Gamma (the SCF calls K once per spin, alpha then
//! beta, so only the beta-call mutants and the Madelung halving are
//! injectable there); the k-point code has all of them.
//!
//! The cheap ones (seconds) run in every shard: the closed-shell limit,
//! 1x1x1 == Gamma, ONE polarized PySCF pin (H3 doublet, HSE06, dense, Gamma)
//! and the Gamma spin mutants against it. Meshes, RS-GDF and the remaining
//! PySCF pins carry `#[ignore]` with their measured time.

mod common;

use common::*;
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

const OMEGA_HCORE: f64 = 0.8;
const MESH: usize = 41;
const AMPLE: usize = 1 << 30;
const AUX: &str = "cc-pvdz-ri";

/// The H3 doublet / H4 triplet cells of `scripts/gen_pyscf_oshell_rsh_refs.py`
/// (TRICLINIC lattice `TRI_A`); `H2_ATOMS` is the closed-shell cell.
const H3D: [[f64; 3]; 3] = [[0.3, 0.2, 0.1], [0.3, 0.2, 1.5], [0.3, 0.2, 2.9]];
const H4T: [[f64; 3]; 4] = [
    [0.3, 0.2, 0.1],
    [0.3, 0.2, 1.5],
    [1.6, 1.1, 2.9],
    [1.6, 1.1, 4.1],
];

fn name_of(f: &str) -> &'static str {
    match f {
        "HSE06" => "HSE06",
        _ => "HYB_GGA_XC_CAM_B3LYP",
    }
}

fn cell_of(atoms: &[[f64; 3]], mult: usize) -> Cell {
    let mut mol: Molecule = hydrogens(atoms);
    mol.multiplicity = mult;
    Cell::new(mol, TRI_A).expect("cell")
}

fn aux_of(cell: &Cell) -> PreparedBasis {
    PreparedBasis::new(cell.mol(), &basis::bundled(AUX).unwrap()).unwrap()
}

/// How a Gamma run is set up.
#[derive(Clone, Copy, Default)]
struct GRun {
    /// RS-GDF (J and K fit on `AUX`) instead of dense AFT.
    rsgdf: bool,
    spin_mutant: Option<GammaSpinMutant>,
    /// Replace the range-separation parameters (a mutant; the exchange is
    /// then built `unchecked`).
    rsh: Option<ferric_pbc::rsh::RshParams>,
}

fn rsh_of(f: &str) -> ferric_pbc::rsh::RshParams {
    resolve_periodic_functional_rsh(name_of(f))
        .unwrap()
        .2
        .unwrap()
}

/// The exchange, the J source and the hcore of a Gamma run: shared by the
/// RKS and UKS drivers so a closed-shell comparison differs only in the SCF.
struct GSetup {
    prep: PreparedBasis,
    hc: ferric_pbc::hcore::PeriodicHcore,
    ex: GammaRshExchange,
    dense: Option<DenseAftEri>,
    gdf: Option<RsGdf>,
    grid: PeriodicGrid,
}

impl GSetup {
    fn new(cell: &Cell, f: &str, exxdiv: ExxDiv, run: GRun) -> Self {
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

    fn ints(&self) -> GammaUhfIntegrals<'_> {
        match (&self.dense, &self.gdf) {
            (Some(e), _) => GammaUhfIntegrals::DenseAft(e),
            (_, Some(g)) => GammaUhfIntegrals::RsGdf(g),
            _ => unreachable!(),
        }
    }
}

fn gamma_uks(cell: &Cell, f: &str, exxdiv: ExxDiv, run: GRun) -> GammaUksResult {
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

fn gamma_rks_energy(cell: &Cell, f: &str, exxdiv: ExxDiv, run: GRun) -> f64 {
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
fn kuks_run(
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

fn rsgdf_cfg(exxdiv: ExxDiv) -> KRsGdfConfig {
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

fn krks_energy(cell: &Cell, f: &str, n: [usize; 3], exxdiv: ExxDiv, rsgdf: bool) -> f64 {
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

// ------------------------------------------------------------ pins

/// PySCF 2.13.1 `pbc.dft.UKS` (Gamma) / `KUKS` (`mf.nelec = (N_a N_k, N_b N_k)`)
/// with AFTDF, UniformGrids 41^3, conv_tol 1e-12, `cell.precision` 1e-11:
/// `testdata/reference/pbc_oshell_sto3g_rsh.json`
/// (`scripts/gen_pyscf_oshell_rsh_refs.py`). (cell, functional, mesh, exxdiv, E)
type Pin = (&'static str, &'static str, [usize; 3], ExxDiv, f64);
const PINS: [Pin; 9] = [
    ("h3d", "HSE06", [1, 1, 1], ExxDiv::Ewald, -1.542269349094),
    ("h3d", "HSE06", [1, 1, 1], ExxDiv::None, -1.355401160117),
    (
        "h3d",
        "CAM_B3LYP",
        [1, 1, 1],
        ExxDiv::Ewald,
        -1.572493318134,
    ),
    ("h3d", "HSE06", [1, 1, 2], ExxDiv::Ewald, -1.656598223289),
    (
        "h3d",
        "CAM_B3LYP",
        [1, 1, 2],
        ExxDiv::Ewald,
        -1.664930856577,
    ),
    ("h3d", "HSE06", [1, 1, 2], ExxDiv::None, -1.564476521484),
    ("h4t", "HSE06", [1, 1, 1], ExxDiv::Ewald, -1.712031415020),
    (
        "h4t",
        "CAM_B3LYP",
        [1, 1, 1],
        ExxDiv::Ewald,
        -1.747824617777,
    ),
    ("h4t", "HSE06", [1, 1, 2], ExxDiv::Ewald, -1.535524899227),
];
/// The pinned energies are PySCF's; this is the bar (see the ledger).
const PIN_TOL: f64 = 1e-11;

fn pin(cell: &str, f: &str, n: [usize; 3], x: ExxDiv) -> f64 {
    PINS.iter()
        .find(|p| p.0 == cell && p.1 == f && p.2 == n && p.3 == x)
        .unwrap_or_else(|| panic!("no pin {cell} {f} {n:?} {x:?}"))
        .4
}

fn cell_named(name: &str) -> Cell {
    match name {
        "h3d" => cell_of(&H3D, 2),
        "h4t" => cell_of(&H4T, 3),
        _ => unreachable!(),
    }
}

// ------------------------------------------- closed-shell limit (Gamma)

fn closed_limit_case(f: &str, x: ExxDiv) {
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
const CLOSED_TOL: f64 = 1e-11;

#[test]
fn gamma_closed_shell_uks_is_rks() {
    closed_limit_case("HSE06", ExxDiv::Ewald);
    closed_limit_case("CAM_B3LYP", ExxDiv::None);
}

#[test]
#[ignore = "the other two (functional, exxdiv) rows of the Gamma closed-shell anchor; ~10 s"]
fn gamma_closed_shell_uks_is_rks_other_rows() {
    closed_limit_case("HSE06", ExxDiv::None);
    closed_limit_case("CAM_B3LYP", ExxDiv::Ewald);
}

// ------------------------------------------ polarized pins (Gamma)

#[test]
fn gamma_polarized_matches_pyscf_h3_doublet() {
    let cell = cell_named("h3d");
    let r = gamma_uks(&cell, "HSE06", ExxDiv::Ewald, GRun::default());
    let d = r.scf.energy - pin("h3d", "HSE06", [1, 1, 1], ExxDiv::Ewald);
    eprintln!(
        "H3 doublet HSE06 Gamma: dE vs PySCF {d:.2e}, <S2> {:.4}",
        r.s2
    );
    assert!(d.abs() < PIN_TOL);
    assert!(
        r.s2 > 0.75 && r.s2 < 0.76,
        "doublet, not a closed-shell mix"
    );
}

#[test]
#[ignore = "Gamma H4 triplet (both spins populated, S2 = 2) vs PySCF, HSE06 and CAM-B3LYP; ~15 s"]
fn gamma_polarized_matches_pyscf_h4_triplet() {
    let cell = cell_named("h4t");
    for f in ["HSE06", "CAM_B3LYP"] {
        let r = gamma_uks(&cell, f, ExxDiv::Ewald, GRun::default());
        let d = r.scf.energy - pin("h4t", f, [1, 1, 1], ExxDiv::Ewald);
        eprintln!(
            "H4 triplet {f} Gamma: dE vs PySCF {d:.2e}, <S2> {:.4}",
            r.s2
        );
        assert!(d.abs() < PIN_TOL, "{f}: {d:.2e}");
        assert!((r.s2 - 2.0).abs() < 1e-3);
    }
}

/// Measured shifts on the H3 doublet / HSE06 / ewald Gamma pin (see the
/// ledger in the module doc of the mutation tests): each is the unmutated
/// energy minus the mutant's. A mutant must move E by more than `MUT_MIN`.
const MUT_MIN: f64 = 1e-3;

#[test]
fn gamma_spin_mutants_fail_the_polarized_pin() {
    let cell = cell_named("h3d");
    let want = pin("h3d", "HSE06", [1, 1, 1], ExxDiv::Ewald);
    for m in [
        GammaSpinMutant::AlphaOnBeta,
        GammaSpinMutant::DropBeta,
        GammaSpinMutant::MadelungHalf,
    ] {
        let run = GRun {
            spin_mutant: Some(m),
            ..Default::default()
        };
        let d = gamma_uks(&cell, "HSE06", ExxDiv::Ewald, run).scf.energy - want;
        eprintln!("Gamma mutant {m:?}: E - PySCF pin {d:+.3e}");
        assert!(d.abs() > MUT_MIN, "{m:?} not detected (shift {d:.2e})");
    }
}

// ---------------------------------------- 1x1x1 mesh == Gamma (polarized)

#[test]
fn one_point_mesh_is_gamma_uks_rsh() {
    let cell = cell_named("h3d");
    let g = gamma_uks(&cell, "HSE06", ExxDiv::Ewald, GRun::default());
    let k = kuks_run(&cell, "HSE06", [1, 1, 1], ExxDiv::Ewald, false, None);
    let d = g.scf.energy - k.scf.energy;
    eprintln!(
        "H3 doublet HSE06: Gamma - 1x1x1 {d:.2e}; E_xc {:.2e}",
        g.e_xc - k.e_xc
    );
    assert!(d.abs() < ONE_POINT_TOL);
    assert!((g.e_xc - k.e_xc).abs() < ONE_POINT_EXC_TOL);
    assert!((g.madelung - k.madelung).abs() < 1e-10, "mixed v_M");
    assert!(
        (k.exact_exchange_fraction - rsh_of("HSE06").c_sr).abs() < 1e-15,
        "reported fraction is c_sr"
    );
}
const ONE_POINT_TOL: f64 = 1e-11;
const ONE_POINT_EXC_TOL: f64 = 1e-10;

// =============================================================== heavier

fn kmesh_pin_row(cell: &str, f: &str, n: [usize; 3], x: ExxDiv) -> f64 {
    let c = cell_named(cell);
    let r = kuks_run(&c, f, n, x, false, None);
    let d = r.scf.energy - pin(cell, f, n, x);
    eprintln!(
        "{cell} {f} {n:?} {x:?}: E {:.12} dE vs PySCF {d:+.2e} ({} it, <S2> {:.4})",
        r.scf.energy, r.scf.iterations, r.scf.s2
    );
    d
}

#[test]
#[ignore = "k-mesh PySCF rows, H3 doublet and H4 triplet (8 SCFs); ~2 min"]
fn kpoint_polarized_matches_pyscf() {
    for (cell, f, n, x, _) in PINS {
        if (n, f, x) == ([1, 1, 1], "HSE06", ExxDiv::Ewald) {
            continue; // the always-on Gamma / 1x1x1 test
        }
        let d = kmesh_pin_row(cell, f, n, x);
        assert!(d.abs() < PIN_TOL, "{cell} {f} {n:?} {x:?}: {d:.2e}");
    }
}

#[test]
#[ignore = "k-point closed-shell limit, KUKS vs KRKS at 1x1x1 and 1x1x2; ~30 s"]
fn kpoint_closed_shell_uks_is_rks() {
    let h2 = cell_of(&H2_ATOMS, 1);
    for n in [[1, 1, 1], [1, 1, 2]] {
        for f in ["HSE06", "CAM_B3LYP"] {
            for x in [ExxDiv::Ewald, ExxDiv::None] {
                let u = kuks_run(&h2, f, n, x, false, None);
                let r = krks_energy(&h2, f, n, x, false);
                eprintln!("closed k {n:?} {f} {x:?}: U-R {:.2e}", u.scf.energy - r);
                assert!((u.scf.energy - r).abs() < CLOSED_TOL, "{n:?} {f} {x:?}");
            }
        }
    }
}

/// The N-cell supercell of `cell`, `mult` the giant-determinant multiplicity.
fn supercell(cell: &Cell, n: [usize; 3], mult: usize) -> (Cell, Vec<[f64; 3]>) {
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

#[test]
#[ignore = "1x1x2 mesh vs the explicit 6-atom supercell (two UKS SCFs); ~30 s"]
fn kmesh_is_the_gamma_supercell_rsh() {
    let cell = cell_named("h3d");
    let n = [1, 1, 2];
    // (N_a - N_b) N_k + 1 with (N_a, N_b) = (2, 1) per cell and N_k = 2.
    let (sc, _) = supercell(&cell, n, 3);
    for f in ["HSE06", "CAM_B3LYP"] {
        let sup = GSetup::new(&sc, f, ExxDiv::Ewald, GRun::default());
        // 41 x 41 x 82 uniform points = two copies of the cell grid.
        let grid = PeriodicGrid::uniform(&sc, [MESH, MESH, 2 * MESH]).unwrap();
        let mut cfg = GammaUksConfig::new(name_of(f));
        cfg.scf.density_conv = 1e-11;
        cfg.scf.energy_conv = 1e-12;
        let g = gamma_uks_rsh_on_grid(&sc, &sup.prep, &sup.hc, sup.ints(), &sup.ex, &grid, &cfg)
            .unwrap();
        let k = kuks_run(&cell, f, n, ExxDiv::Ewald, false, None);
        let d = k.scf.energy - g.scf.energy / 2.0;
        eprintln!(
            "{f}: E_k {:.12} E_super/2 {:.12} d {d:.2e}",
            k.scf.energy,
            g.scf.energy / 2.0
        );
        assert!(d.abs() < SUPERCELL_TOL, "{f}: {d:.2e}");
    }
}
const SUPERCELL_TOL: f64 = 1e-10;

#[test]
#[ignore = "five k-point spin/Madelung mutants on the 1x1x2 doublet pin; ~1 min"]
fn kpoint_spin_mutants_fail_the_polarized_pin() {
    let cell = cell_named("h3d");
    let want = pin("h3d", "HSE06", [1, 1, 2], ExxDiv::Ewald);
    for m in [
        KUhfMutation::KAlphaOnBeta,
        KUhfMutation::KSwapSpin,
        KUhfMutation::KDropBeta,
        KUhfMutation::MadelungHalfPerSpin,
        KUhfMutation::PlainMadelungForRsh,
        KUhfMutation::KFromTotalDensity,
    ] {
        let d = kuks_run(&cell, "HSE06", [1, 1, 2], ExxDiv::Ewald, false, Some(m))
            .scf
            .energy
            - want;
        eprintln!("k mutant {m:?}: E - PySCF pin {d:+.3e}");
        assert!(d.abs() > MUT_MIN, "{m:?} not detected ({d:.2e})");
    }
}

#[test]
#[ignore = "closed-shell k mutants: which are INVISIBLE without polarization; 4 SCFs, ~20 s"]
fn alpha_on_beta_and_swap_are_invisible_in_the_closed_shell_limit() {
    // Documents the artifact hypothesis: the spin mutants cannot be caught by
    // the closed-shell anchor, which is why they are measured on polarized
    // cells above.
    let h2 = cell_of(&H2_ATOMS, 1);
    let base = kuks_run(&h2, "HSE06", [1, 1, 2], ExxDiv::Ewald, false, None)
        .scf
        .energy;
    for m in [KUhfMutation::KAlphaOnBeta, KUhfMutation::KSwapSpin] {
        let e = kuks_run(&h2, "HSE06", [1, 1, 2], ExxDiv::Ewald, false, Some(m))
            .scf
            .energy;
        eprintln!("closed-shell {m:?}: dE {:.2e}", e - base);
        assert!((e - base).abs() < CLOSED_TOL, "{m:?}");
    }
}

#[test]
#[ignore = "omega / fraction mutants of the exchange on the polarized Gamma pin; 4 SCFs, ~20 s"]
fn gamma_rsh_parameter_mutants_fail_the_polarized_pin_and_erf_piece_is_live() {
    let cell = cell_named("h3d");
    let want = pin("h3d", "HSE06", [1, 1, 1], ExxDiv::Ewald);
    let r = rsh_of("HSE06");
    let mut shifts = Vec::new();
    for (label, p) in [
        (
            "omega x1.01",
            ferric_pbc::rsh::RshParams {
                omega: r.omega * 1.01,
                ..r
            },
        ),
        (
            "omega x0.01 (erf piece off)",
            ferric_pbc::rsh::RshParams {
                omega: r.omega * 0.01,
                ..r
            },
        ),
        (
            "c_lr := c_sr",
            ferric_pbc::rsh::RshParams { c_lr: r.c_sr, ..r },
        ),
        (
            "c_sr x0.99",
            ferric_pbc::rsh::RshParams {
                c_sr: r.c_sr * 0.99,
                ..r
            },
        ),
    ] {
        let run = GRun {
            rsh: Some(p),
            ..Default::default()
        };
        let d = gamma_uks(&cell, "HSE06", ExxDiv::Ewald, run).scf.energy - want;
        eprintln!("Gamma RSH mutant {label}: E - pin {d:+.3e}");
        shifts.push(d);
    }
    // omega x1.01 is the finest (1% in the range-separation parameter).
    assert!(shifts.iter().all(|d| d.abs() > OMEGA_MUT_MIN), "{shifts:?}");
}
const OMEGA_MUT_MIN: f64 = 1e-4;

/// Gamma UKS with the GLOBAL hybrid PBE0 on the same grid and J source as
/// the RSH runs (the plain-Coulomb reference for fit errors and for the
/// "HSE06 is not PBE0" check).
fn gamma_pbe0(cell: &Cell, rsgdf: bool) -> f64 {
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

#[test]
#[ignore = "PBE0 on the same polarized cell: HSE06 must differ from it by far more than the bar; 1 SCF, ~10 s"]
fn hse06_differs_from_pbe0_on_the_polarized_cell() {
    let cell = cell_named("h3d");
    let e_pbe0 = gamma_pbe0(&cell, false);
    let pyscf_pbe0 = -1.5422451368080061;
    let e_hse = pin("h3d", "HSE06", [1, 1, 1], ExxDiv::Ewald);
    eprintln!(
        "PBE0 {e_pbe0:.12} vs PySCF {pyscf_pbe0:.12}: {:.2e}; HSE06 - PBE0 {:.3e}",
        e_pbe0 - pyscf_pbe0,
        e_hse - e_pbe0
    );
    assert!((e_pbe0 - pyscf_pbe0).abs() < PIN_TOL, "polarized PBE0 pin");
    assert!(
        (e_hse - e_pbe0).abs() > 1e5 * PIN_TOL,
        "HSE06 must not collapse onto PBE0"
    );
}

// ------------------------------------------------------------- RS-GDF

#[test]
#[ignore = "RS-GDF vs dense, Gamma and 1x1x2, polarized (4 fitted SCFs + PBE0 fit-error references); ~2 min"]
fn rsgdf_oshell_rsh_is_dense_to_the_plain_coulomb_fit_error() {
    // PBE0 (global hybrid) RS-GDF - dense is the plain-Coulomb fit error; the
    // RSH RS-GDF - dense must be of the same order (the attenuated K adds a
    // G-space fit on top of the same full-Coulomb one).
    let cell = cell_named("h3d");
    let n = [1, 1, 2];
    let dense = kuks_run(&cell, "HSE06", n, ExxDiv::Ewald, false, None)
        .scf
        .energy;
    let fit = kuks_run(&cell, "HSE06", n, ExxDiv::Ewald, true, None)
        .scf
        .energy;
    let pbe0_d = kuks_pbe0(&cell, n, false);
    let pbe0_f = kuks_pbe0(&cell, n, true);
    let (e_rsh, e_c) = (fit - dense, pbe0_f - pbe0_d);
    eprintln!(
        "k 1x1x2: RS-GDF - dense: HSE06 {e_rsh:.3e}, PBE0 {e_c:.3e}, ratio {:.3}",
        e_rsh / e_c
    );
    assert!(e_rsh.abs() < 2.0 * e_c.abs() && e_rsh.abs() > 0.2 * e_c.abs());
    let gd = gamma_uks(&cell, "HSE06", ExxDiv::Ewald, GRun::default())
        .scf
        .energy;
    let gf = gamma_uks(
        &cell,
        "HSE06",
        ExxDiv::Ewald,
        GRun {
            rsgdf: true,
            ..Default::default()
        },
    )
    .scf
    .energy;
    let g_c = gamma_pbe0(&cell, true) - gamma_pbe0(&cell, false);
    eprintln!(
        "Gamma: RS-GDF - dense HSE06 {:.3e}, PBE0 {g_c:.3e}, ratio {:.3}",
        gf - gd,
        (gf - gd) / g_c
    );
    assert!((gf - gd).abs() < 2.0 * g_c.abs() && (gf - gd).abs() > 0.2 * g_c.abs());
}

fn kuks_pbe0(cell: &Cell, n: [usize; 3], rsgdf: bool) -> f64 {
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

// ------------------------------------------------------------ refusals

#[test]
fn derivatives_and_roks_still_refuse_range_separated_hybrids_by_name() {
    // Forces / stress: the shared k-point gate, open-shell flavour.
    let e = ferric_pbc::kxcgrad::check_kpoint_deriv_functional(true, "HSE06", "with_gradient")
        .unwrap_err();
    assert!(e.contains("range-separated"), "{e}");
    // ROKS: the Gamma entry resolves the functional with the global-hybrid
    // gate before any grid or integral is built.
    let cell = cell_named("h3d");
    let bs = pyscf_sto3g_h();
    let prep = prep_for(&cell, &bs);
    let hc = periodic_hcore(&cell, &prep, &PeriodicHcoreConfig::with_omega(OMEGA_HCORE)).unwrap();
    let eri = DenseAftEri::build(
        &cell,
        &prep,
        &hc.s,
        ExxDiv::None,
        DEFAULT_DENSE_AFT_PRECISION,
        DEFAULT_DENSE_AFT_MAX_BYTES,
    )
    .unwrap();
    let cfg = ferric_pbc::rohf::GammaRoksConfig::new("HSE06");
    let e =
        ferric_pbc::rohf::gamma_roks(&cell, &prep, &hc, GammaUhfIntegrals::DenseAft(&eri), &cfg)
            .unwrap_err()
            .to_string();
    assert!(e.contains("range-separated"), "{e}");
}
