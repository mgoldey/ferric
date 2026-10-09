//! k-point restricted open-shell HF / KS (`ferric_pbc::kroks`).
//!
//! No k-point ROHF existed (`rohf.rs` is Gamma), so the Roothaan effective
//! Fock, the PySCF `KROHF.get_occ` global occupation rule and the
//! spin-resolved Bloch-AO XC are all new here. What is independent of what:
//!
//! * `closed_shell_limit_is_krhf_and_krks` — `N_α = N_β`: k-ROHF must be
//!   `solve_krhf` and k-ROKS `solve_krks` (different SCF code, different
//!   density/XC path: the closed-shell Bloch XC vs the spin-resolved one).
//! * `one_point_mesh_is_gamma_rohf_and_roks` — 1×1×1 against `gamma_rohf` /
//!   `gamma_roks` (real Gamma path through `ferric_scf::rohf`), H3 doublet
//!   (`N_α, N_β = 2, 1`, so the Roothaan coupling is exercised).
//! * `k_mesh_is_the_gamma_supercell` — E/cell of a 1×1×2 mesh against the
//!   Gamma ROHF/ROKS energy of the explicit 2-cell supercell (triplet) / 2,
//!   on N translated copies of the cell grid ([`PeriodicGrid::replicated`]).
//! * `n_beta_zero_is_kuhf_and_pyscf` — with `N_β = 0` ROHF is UHF: k-ROHF
//!   against `solve_kuhf` and PySCF `KUHF` pins from `pbc_kuhf.rs`, plus the
//!   zchain 1×1×2 non-uniform-occupation case (global aufbau, per-k mutant).
//! * `pyscf_krohf_kroks_pins` — PySCF 2.13 `pbc.dft.KROKS` (H3 doublet 1x1x2,
//!   AFTDF mesh 45^3, `UniformGrids(45^3)`, exxdiv none/ewald,
//!   `mf.nelec = (N_α N_k, N_β N_k)`; scratchpad `ref_roks/pins.py`): LDA,
//!   PBE, PBE0 ewald and PBE0 none. There is NO PySCF `KROHF` pin for H3
//!   1x1x2: PySCF's own `get_occ` cycles there (see
//!   `alpha_energy_rule_is_pyscf_get_occ_including_its_cycle`), so k-ROHF is
//!   pinned by the supercell and Gamma anchors and, for `N_β = 0`, by the
//!   PySCF `KUHF` pins.
//!
//! # Mutation plan (each must fail the named test)
//!
//! * `KRohfMutation::BareAlphaFock` -> `one_point_mesh_is_gamma_rohf_and_roks`
//!   (and every open-shell test; invisible in the closed-shell limit).
//! * `KRohfMutation::PerKAufbau` -> `n_beta_zero_is_kuhf_and_pyscf` (zchain).
//! * `KRohfMutation::UnscaledExchange` -> `closed_shell_limit_...` (PBE0).

mod common;

use common::*;
use ferric_core::basis::BasisSet;
use ferric_core::mol::Molecule;
use ferric_pbc::dense_aft::{
    DenseAftEri, ExxDiv, DEFAULT_DENSE_AFT_MAX_BYTES, DEFAULT_DENSE_AFT_PRECISION,
};
use ferric_pbc::dft::{PeriodicGrid, PeriodicGridConfig, PeriodicXc, PeriodicXcConfig};
use ferric_pbc::hcore::{periodic_hcore, PeriodicHcoreConfig};
use ferric_pbc::kdft::{solve_krks_on_grid, KRksConfig};
use ferric_pbc::kpts::KPointMesh;
use ferric_pbc::kroks::{
    solve_krohf, solve_kroks_on_grid, KRohfConfig, KRohfMutation, KRoksConfig, KRoksResult,
    OpenOrbitalRule,
};
use ferric_pbc::kscf::{solve_krhf, KRhfConfig, KScfConfig};
use ferric_pbc::kuscf::{solve_kuhf, KUhfConfig};
use ferric_pbc::lattice::Cell;
use ferric_pbc::rohf::{gamma_rohf, gamma_roks_with_xc, GammaRohfConfig, GammaRoksConfig};
use ferric_pbc::uhf::{EwaldStart, GammaUhfIntegrals};

const OMEGA: f64 = 0.8;
const EXX: [ExxDiv; 2] = [ExxDiv::None, ExxDiv::Ewald];

const H3_ATOMS: [[f64; 3]; 3] = [[0.3, 0.2, 0.1], [0.35, 0.12, 1.5], [1.6, 0.9, 0.7]];
const H3_A: f64 = 4.5;
const H_ATOM: [[f64; 3]; 1] = [[0.3, 0.2, 0.1]];

// PySCF KUHF + AFTDF pins of the H atom (pbc_kuhf.rs `run_kuhf_oracle.py`):
// (none, ewald). N_β = 0 so ROHF == UHF.
const H1X2: (f64, f64) = (-0.399399818915, -0.625130045222);
const H1X3: (f64, f64) = (-0.528717460735, -0.623551487522);

// PySCF 2.13 KROKS (H3 doublet, cubic a = 4.5, STO-3G, 1x1x2, AFTDF mesh 45^3
// precision 1e-9, UniformGrids(45^3), `mf.nelec = (2 N_k, N_k)`, conv_tol 1e-11;
// scratchpad `ref_roks/pins.py`, 2026-10-09): (functional, mesh, exxdiv, E/cell).
const KROKS_PINS: [(&str, [usize; 3], &str, f64); 4] = [
    ("LDA", [1, 1, 2], "ewald", -1.447882440346),
    ("PBE", [1, 1, 2], "ewald", -1.474623275092),
    ("PBE0", [1, 1, 2], "ewald", -1.483324363436),
    ("PBE0", [1, 1, 2], "none", -1.332837545898),
];

fn cell_cm(atoms: &[[f64; 3]], lattice: [[f64; 3]; 3], charge: i32, mult: usize) -> Cell {
    let mut mol: Molecule = hydrogens(atoms);
    mol.charge = charge;
    mol.multiplicity = mult;
    Cell::new(mol, lattice).expect("cell")
}

fn h3_cell() -> Cell {
    cell_cm(&H3_ATOMS, cubic(H3_A), 0, 2)
}

fn kscf_cfg() -> KScfConfig {
    KScfConfig {
        energy_conv: 1e-12,
        grad_conv: 1e-9,
        ..Default::default()
    }
}

fn rohf_cfg(cell: &Cell, exx: ExxDiv, mutation: Option<KRohfMutation>) -> KRohfConfig {
    let mut c = KRohfConfig::for_cell(cell, exx);
    c.scf = kscf_cfg();
    c.hcore = PeriodicHcoreConfig::with_omega(OMEGA);
    c.mutation = mutation;
    c
}

/// Hybrid functionals get the Gamma ROKS DIIS-robustness defaults.
fn roks_cfg(
    cell: &Cell,
    exx: ExxDiv,
    functional: &str,
    mutation: Option<KRohfMutation>,
) -> KRoksConfig {
    let mut c = KRoksConfig::new(cell, exx, functional);
    c.rohf = rohf_cfg(cell, exx, mutation);
    if functional.contains("PBE0") {
        c.rohf.level_shift = 0.05;
        c.rohf.scf.max_iter = 600;
    }
    c
}

fn grid_cfg(n_rad: usize, n_ang: usize) -> PeriodicGridConfig {
    PeriodicGridConfig::with_size(n_rad, n_ang)
}

fn close(got: f64, want: f64, tol: f64, what: &str) {
    eprintln!(
        "  {what}: {got:.12} (want {want:.12}, diff {:.2e})",
        got - want
    );
    assert!(got.is_finite(), "{what}: non-finite");
    assert!(
        (got - want).abs() < tol,
        "{what}: {got:.12e} vs {want:.12e} (tol {tol:.1e})"
    );
}

fn sz_s2(na: usize, nb: usize, nk: usize) -> f64 {
    let sz = 0.5 * (na as f64 - nb as f64) * nk as f64;
    sz * (sz + 1.0)
}

fn krohf(cell: &Cell, bs: &BasisSet, n: [usize; 3], cfg: &KRohfConfig) -> KRoksResult {
    let prep = prep_for(cell, bs);
    let mesh = KPointMesh::gamma_centred(cell, n).unwrap();
    let r = solve_krohf(cell, &prep, None, &mesh, cfg).expect("k-ROHF");
    report("kROHF", n, &r);
    r
}

fn report(what: &str, n: [usize; 3], r: &KRoksResult) {
    let s = &r.scf;
    eprintln!(
        "  {what} {n:?}: E/cell {:.12} <S2> {:.10} ({} it) closed/k {:?} open/k {:?} gaps {:?}",
        s.energy, s.s2, s.iterations, s.nclosed_per_k, s.nopen_per_k, r.gaps
    );
    assert!(s.converged);
}

fn kroks_on_grid(
    cell: &Cell,
    bs: &BasisSet,
    n: [usize; 3],
    grid: &PeriodicGrid,
    cfg: &KRoksConfig,
) -> KRoksResult {
    let prep = prep_for(cell, bs);
    let mesh = KPointMesh::gamma_centred(cell, n).unwrap();
    let r = solve_kroks_on_grid(cell, &prep, None, &mesh, grid, cfg)
        .unwrap_or_else(|e| panic!("k-ROKS {}: {e}", cfg.functional));
    report(&format!("kROKS {}", cfg.functional), n, &r);
    r
}

// ---------------------------------------------------- closed-shell limit

#[test]
fn closed_shell_limit_is_krhf_and_krks() {
    let cell = h2_cell(4.0);
    let bs = pyscf_sto3g_h();
    let prep = prep_for(&cell, &bs);
    let n = [1, 1, 2];
    let mesh = KPointMesh::gamma_centred(&cell, n).unwrap();
    for exx in EXX {
        let mut rc = KRhfConfig::for_cell(&cell, exx);
        rc.scf = kscf_cfg();
        rc.hcore = PeriodicHcoreConfig::with_omega(OMEGA);
        let rhf = solve_krhf(&cell, &prep, None, &mesh, &rc).unwrap();
        let ro = krohf(&cell, &bs, n, &rohf_cfg(&cell, exx, None));
        close(
            ro.scf.energy,
            rhf.energy,
            1e-11,
            &format!("{exx:?} kROHF vs kRHF"),
        );
        close(ro.scf.s2, 0.0, 1e-10, "closed-shell <S2>");
        assert!(ro.scf.nopen_per_k.iter().all(|&o| o == 0));
    }
    let grid = PeriodicGrid::uniform(&cell, [36; 3]).unwrap();
    for functional in ["LDA", "PBE", "PBE0"] {
        let mut kc = KRksConfig::new(&cell, ExxDiv::Ewald, functional);
        kc.krhf.hcore = PeriodicHcoreConfig::with_omega(OMEGA);
        kc.krhf.scf = kscf_cfg();
        let ks = solve_krks_on_grid(&cell, &prep, None, &mesh, &grid, &kc).unwrap();
        let ro = kroks_on_grid(
            &cell,
            &bs,
            n,
            &grid,
            &roks_cfg(&cell, ExxDiv::Ewald, functional, None),
        );
        eprintln!(
            "  {functional}: kROKS e_xc {:.10} N_grid {:.8} | kRKS e_xc {:.10} N_grid {:.8}",
            ro.e_xc, ro.electrons_on_grid, ks.e_xc, ks.electrons_on_grid
        );
        close(
            ro.scf.energy,
            ks.scf.energy,
            1e-10,
            &format!("{functional} kROKS vs kRKS"),
        );
        close(ro.e_xc, ks.e_xc, 1e-10, &format!("{functional} E_xc"));
        close(
            ro.electrons_on_grid,
            ks.electrons_on_grid,
            1e-10,
            "electrons on grid",
        );
        if functional == "PBE0" {
            // Mutant: unscaled exact exchange in a hybrid.
            let m = kroks_on_grid(
                &cell,
                &bs,
                n,
                &grid,
                &roks_cfg(
                    &cell,
                    ExxDiv::Ewald,
                    functional,
                    Some(KRohfMutation::UnscaledExchange),
                ),
            );
            let d = m.scf.energy - ks.scf.energy;
            eprintln!("  MUTANT unscaled K PBE0: dE {d:+.3e}");
            assert!(d.abs() > 1e-2, "unscaled-exchange mutant not caught: {d:e}");
        }
    }
}

// ----------------------------------------------------------- 1x1x1 = Gamma

#[test]
fn one_point_mesh_is_gamma_rohf_and_roks() {
    let cell = h3_cell();
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
    let ints = || GammaUhfIntegrals::DenseAft(&eri);
    for exx in EXX {
        let g = gamma_rohf(
            &cell,
            &prep,
            &hc,
            ints(),
            &GammaRohfConfig {
                exxdiv: exx,
                ewald_start: EwaldStart::Staged,
                ..Default::default()
            },
        )
        .unwrap();
        let k = krohf(&cell, &bs, [1, 1, 1], &rohf_cfg(&cell, exx, None));
        close(
            k.scf.energy,
            g.scf.energy,
            1e-9,
            &format!("{exx:?} 1x1x1 kROHF vs gamma_rohf"),
        );
        close(k.scf.s2, g.s2, 1e-9, "<S2>");
        assert_eq!((k.scf.nclosed_per_k[0], k.scf.nopen_per_k[0]), (1, 1));
        if exx == ExxDiv::Ewald {
            let m = krohf(
                &cell,
                &bs,
                [1, 1, 1],
                &rohf_cfg(&cell, exx, Some(KRohfMutation::BareAlphaFock)),
            );
            let d = m.scf.energy - g.scf.energy;
            eprintln!("  MUTANT bare F_alpha (HF ewald): dE {d:+.3e}");
            assert!(d.abs() > 1e-5, "bare-F_alpha mutant not caught: {d:e}");
        }
    }
    let grid = PeriodicGrid::build(&cell, &grid_cfg(50, 110)).unwrap();
    for functional in ["LDA", "PBE", "PBE0"] {
        let mut gcfg = GammaRoksConfig::new(functional);
        gcfg.grid = grid_cfg(50, 110);
        let mut pxc =
            PeriodicXc::new(&cell, &bs, functional, &grid, &PeriodicXcConfig::default()).unwrap();
        let g = gamma_roks_with_xc(&cell, &prep, &hc, ints(), &mut pxc, &gcfg).unwrap();
        let k = kroks_on_grid(
            &cell,
            &bs,
            [1, 1, 1],
            &grid,
            &roks_cfg(&cell, ExxDiv::Ewald, functional, None),
        );
        close(
            k.scf.energy,
            g.scf.energy,
            1e-9,
            &format!("{functional} 1x1x1 kROKS vs gamma_roks"),
        );
        close(k.e_xc, g.e_xc, 1e-9, &format!("{functional} E_xc"));
        close(k.scf.s2, g.s2, 1e-9, "<S2>");
    }
}

// ----------------------------------------------------- k-mesh = supercell

/// Explicit diag(n) supercell of `cell` with total multiplicity `mult`, and
/// the translation vectors of its copies (m0 outer).
fn supercell(cell: &Cell, n: [usize; 3], mult: usize) -> (Cell, Vec<[f64; 3]>) {
    let a = *cell.lattice();
    let mut atoms = Vec::new();
    let mut shifts = Vec::new();
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
        for v in row.iter_mut() {
            *v *= n[i] as f64;
        }
    }
    (cell_cm(&atoms, lat, 0, mult), shifts)
}

#[test]
fn k_mesh_is_the_gamma_supercell() {
    let cell = h3_cell();
    let bs = pyscf_sto3g_h();
    let (na, nb) = (2usize, 1usize);
    for n in [[1usize, 1, 2], [2, 2, 1]] {
        let nk = n[0] * n[1] * n[2];
        let mult = (na - nb) * nk + 1;
        let (sc, shifts) = supercell(&cell, n, mult);
        let sprep = prep_for(&sc, &bs);
        let shc = periodic_hcore(&sc, &sprep, &PeriodicHcoreConfig::with_omega(OMEGA)).unwrap();
        let seri = DenseAftEri::build(
            &sc,
            &sprep,
            &shc.s,
            ExxDiv::None,
            DEFAULT_DENSE_AFT_PRECISION,
            DEFAULT_DENSE_AFT_MAX_BYTES,
        )
        .unwrap();
        for exx in EXX {
            let g = gamma_rohf(
                &sc,
                &sprep,
                &shc,
                GammaUhfIntegrals::DenseAft(&seri),
                &GammaRohfConfig {
                    exxdiv: exx,
                    ..Default::default()
                },
            )
            .unwrap();
            eprintln!(
                "  supercell {n:?} {exx:?}: E/N {:.12} ({} it)",
                g.scf.energy / nk as f64,
                g.scf.iterations
            );
            let k = krohf(&cell, &bs, n, &rohf_cfg(&cell, exx, None));
            close(
                k.scf.energy,
                g.scf.energy / nk as f64,
                1e-9,
                &format!("H3 {n:?} {exx:?} kROHF vs supercell/N"),
            );
            close(k.scf.s2, sz_s2(na, nb, nk), 1e-9, "giant <S2> = Sz(Sz+1)");
            close(g.s2, sz_s2(na, nb, nk), 1e-9, "supercell <S2>");
            assert_eq!(k.scf.nclosed_per_k, vec![1; nk]);
            assert_eq!(k.scf.nopen_per_k, vec![1; nk]);
        }
        let cell_grid = PeriodicGrid::build(&cell, &grid_cfg(40, 110)).unwrap();
        let sc_grid = cell_grid.replicated(&shifts);
        for functional in ["LDA", "PBE", "PBE0"] {
            let mut gcfg = GammaRoksConfig::new(functional);
            gcfg.scf.density_conv = 1e-11;
            let mut pxc =
                PeriodicXc::new(&sc, &bs, functional, &sc_grid, &PeriodicXcConfig::default())
                    .unwrap();
            let g = gamma_roks_with_xc(
                &sc,
                &sprep,
                &shc,
                GammaUhfIntegrals::DenseAft(&seri),
                &mut pxc,
                &gcfg,
            )
            .unwrap();
            let k = kroks_on_grid(
                &cell,
                &bs,
                n,
                &cell_grid,
                &roks_cfg(&cell, ExxDiv::Ewald, functional, None),
            );
            close(
                k.scf.energy,
                g.scf.energy / nk as f64,
                1e-8,
                &format!("H3 {n:?} {functional} kROKS vs supercell/N"),
            );
            close(k.scf.s2, sz_s2(na, nb, nk), 1e-9, "giant <S2>");
        }
    }
}

// ------------------------------------------------------------ N_beta = 0

fn kuhf_energy(cell: &Cell, bs: &BasisSet, n: [usize; 3], exx: ExxDiv) -> f64 {
    let prep = prep_for(cell, bs);
    let mesh = KPointMesh::gamma_centred(cell, n).unwrap();
    let mut c = KUhfConfig::for_cell(cell, exx);
    c.scf = kscf_cfg();
    c.hcore = PeriodicHcoreConfig::with_omega(OMEGA);
    let r = solve_kuhf(cell, &prep, None, &mesh, &c).expect("kUHF");
    assert!(r.scf.converged);
    r.scf.energy
}

fn zchain() -> Cell {
    cell_cm(
        &[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 2.0]],
        1,
        2,
    )
}

#[test]
fn n_beta_zero_is_kuhf_and_pyscf() {
    let cell = cell_cm(&H_ATOM, cubic(4.0), 0, 2);
    let bs = pyscf_sto3g_h();
    for (n, pins) in [([1usize, 1, 2], H1X2), ([1, 1, 3], H1X3)] {
        for (i, exx) in EXX.into_iter().enumerate() {
            let r = krohf(&cell, &bs, n, &rohf_cfg(&cell, exx, None));
            let u = kuhf_energy(&cell, &bs, n, exx);
            close(
                r.scf.energy,
                u,
                1e-11,
                &format!("{n:?} {exx:?} kROHF vs kUHF"),
            );
            let pin = if i == 0 { pins.0 } else { pins.1 };
            close(
                r.scf.energy,
                pin,
                1e-9,
                &format!("{n:?} {exx:?} vs PySCF KUHF"),
            );
            close(
                r.scf.s2,
                sz_s2(1, 0, n[0] * n[1] * n[2]),
                1e-9,
                "giant <S2>",
            );
        }
    }
    // zchain 1x1x2: GLOBAL aufbau puts both open orbitals at Gamma.
    let z = zchain();
    for exx in EXX {
        let r = krohf(&z, &bs, [1, 1, 2], &rohf_cfg(&z, exx, None));
        let u = kuhf_energy(&z, &bs, [1, 1, 2], exx);
        close(
            r.scf.energy,
            u,
            1e-10,
            &format!("zchain {exx:?} kROHF vs kUHF"),
        );
        assert_eq!(
            r.scf.nopen_per_k,
            vec![2, 0],
            "{exx:?}: open orbitals per k"
        );
        let m = krohf(
            &z,
            &bs,
            [1, 1, 2],
            &rohf_cfg(&z, exx, Some(KRohfMutation::PerKAufbau)),
        );
        let d = m.scf.energy - r.scf.energy;
        eprintln!("  MUTANT per-k aufbau {exx:?}: dE {d:+.4}");
        assert_eq!(m.scf.nopen_per_k, vec![1, 1]);
        assert!(d > 0.2, "per-k aufbau mutant not caught: {d:e}");
    }
}

// ------------------------------------------------------ open-orbital rule

/// PySCF `KROHF.get_occ` picks the open orbitals by `c† F_α c`
/// ([`OpenOrbitalRule::AlphaEnergy`]). On H3 1x1x2 (exxdiv none) that rule
/// has no fixed point at the ROHF minimum: PySCF 2.13 itself cycles with
/// period 2 between E = -0.771028047100 (a stationary point, |g| ~ 1e-10) and
/// -0.825316355 (scratchpad `ref_roks/hfdiag.py`, mesh 31^3, 40 cycles), the
/// same two energies this implementation's alpha-energy rule alternates
/// between. The default Roothaan-order rule converges to the supercell state
/// (`k_mesh_is_the_gamma_supercell`).
#[test]
fn alpha_energy_rule_is_pyscf_get_occ_including_its_cycle() {
    let cell = h3_cell();
    let bs = pyscf_sto3g_h();
    let prep = prep_for(&cell, &bs);
    let mesh = KPointMesh::gamma_centred(&cell, [1, 1, 2]).unwrap();
    let mut cfg = rohf_cfg(&cell, ExxDiv::None, None);
    cfg.open_rule = OpenOrbitalRule::AlphaEnergy;
    cfg.scf.max_iter = 60;
    let err = solve_krohf(&cell, &prep, None, &mesh, &cfg).expect_err("PySCF rule cycles here");
    eprintln!("  alpha-energy rule: {err}");
    assert!(format!("{err}").contains("not converged"), "{err}");
    // Where the rule has a fixed point it is the same state: 1x1x1 and N_beta = 0.
    let r1 = {
        let mesh = KPointMesh::gamma_centred(&cell, [1, 1, 1]).unwrap();
        let mut c = rohf_cfg(&cell, ExxDiv::None, None);
        c.open_rule = OpenOrbitalRule::AlphaEnergy;
        solve_krohf(&cell, &prep, None, &mesh, &c).expect("1x1x1")
    };
    let base = krohf(&cell, &bs, [1, 1, 1], &rohf_cfg(&cell, ExxDiv::None, None));
    close(
        r1.scf.energy,
        base.scf.energy,
        1e-10,
        "1x1x1 alpha-energy rule vs Roothaan order",
    );
}

// ------------------------------------------------------------- PySCF pins

#[test]
fn pyscf_krohf_kroks_pins() {
    let cell = h3_cell();
    let bs = pyscf_sto3g_h();
    let grid_n = 45;
    let grid = PeriodicGrid::uniform(&cell, [grid_n; 3]).unwrap();
    for &(functional, n, exx, e_ref) in &KROKS_PINS {
        let exx = if exx == "none" {
            ExxDiv::None
        } else {
            ExxDiv::Ewald
        };
        let r = if functional == "HF" {
            krohf(&cell, &bs, n, &rohf_cfg(&cell, exx, None))
        } else {
            kroks_on_grid(
                &cell,
                &bs,
                n,
                &grid,
                &roks_cfg(&cell, exx, functional, None),
            )
        };
        close(
            r.scf.energy,
            e_ref,
            2e-8,
            &format!("H3 {functional} {n:?} {exx:?} vs PySCF"),
        );
    }
}

/// KPeriodicXc accepts meta-GGA for the closed-shell solver; the open-shell
/// solver has no tau term, so it must refuse SCAN on BOTH entry points
/// rather than return a wrong energy.
#[test]
fn kroks_refuses_meta_gga() {
    let cell = h3_cell();
    let bs = pyscf_sto3g_h();
    let prep = prep_for(&cell, &bs);
    let mesh = KPointMesh::gamma_centred(&cell, [1, 1, 2]).unwrap();
    let cfg = roks_cfg(&cell, ExxDiv::None, "SCAN", None);
    let grid = PeriodicGrid::build(&cell, &cfg.grid).unwrap();
    let on_grid = solve_kroks_on_grid(&cell, &prep, None, &mesh, &grid, &cfg);
    assert!(on_grid.is_err(), "solve_kroks_on_grid accepted SCAN");
    let plain = ferric_pbc::kroks::solve_kroks(&cell, &prep, None, &mesh, &cfg);
    assert!(plain.is_err(), "solve_kroks accepted SCAN");
}
