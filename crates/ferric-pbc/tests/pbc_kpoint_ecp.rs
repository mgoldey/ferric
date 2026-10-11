// LANL2DZ exponents such as 0.318 are basis data, not approximations of 1/pi.
#![allow(clippy::approx_constant)]
//! k-point forces and stress with a periodic ECP (`ferric_pbc::kgrad` /
//! `ferric_pbc::kstress`, term `parts.ecp`).
//!
//! System: HI, H STO-3G + compact I s/p basis, LANL2DZ ECP on I (the
//! fixtures of `pbc_grad_ecp.rs`), cell diag(6, 6, 7), atoms off the common
//! axis. 8 electrons, 5 AOs.
//!
//! What is anchored against what:
//! (1) TERM GATE at a FIXED Hermitian `D(k)` (no SCF, before any two-electron
//!     or Pulay term is involved): the analytic ECP force and strain of the
//!     k energy `(1/N_k) Σ_k Re tr[D(k) V_ECP(k)]` against central FD of
//!     ferric's own lattice sum (`periodic_ecp_images(..).at_kpts`) on
//!     displaced / strained cells, the mesh rebuilt from the same integer
//!     numerators.
//! (2) SUPERCELL at fixed density: a Gamma-centred 1x1x3 mesh IS the explicit
//!     diag(1,1,3) supercell, so per cell the ECP term equals the Gamma ECP
//!     term of the supercell at the unfolded density (force: the same atom in
//!     any copy; strain: `dE/dε_sc / N`). 1x1x3 is NON-TRIM (ConjD and a
//!     transposed weight are identities on meshes with every `n_i <= 2`).
//! (3) SCF level (RHF/UHF, forces and stress, both exxdiv): 1x1x1 mesh vs the
//!     Gamma code, supercell/N per part and in total, FD of the k energy.
//!
//! ARTIFACT vs PHYSICS hypotheses are stated next to each test.

mod common;

use common::prep_for;
use ferric_core::basis::{BasisSet, Shell};
use ferric_core::ecp::{EcpDef, EcpShell, EcpTerm};
use ferric_core::mol::{Atom, Molecule};
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_pbc::dense_aft::{DenseAftEri, ExxDiv, DEFAULT_DENSE_AFT_MAX_BYTES};
use ferric_pbc::ecp::{
    periodic_ecp_gradient, periodic_ecp_images, periodic_ecp_strain, PeriodicEcpConfig,
};
use ferric_pbc::grad::{gamma_rhf_gradient_with, GammaGradConfig, GammaGradient};
use ferric_pbc::hcore::kpoint::{periodic_hcore_kpts, PeriodicHcoreK};
use ferric_pbc::hcore::{periodic_hcore, PeriodicHcore, PeriodicHcoreConfig};
use ferric_pbc::kdense_aft::{KDenseAftConfig, KDenseAftEri};
use ferric_pbc::kgrad::{
    kpoint_ecp_gradient_fixed_d, kpoint_rhf_gradient, kpoint_uhf_gradient, KGradConfig, KGradJk,
    KGradMutation, KGradient,
};
use ferric_pbc::kpts::KPointMesh;
use ferric_pbc::kscf::{solve_krhf_injected, KPointInjection, KPointJk, KScfConfig, KScfResult};
use ferric_pbc::kstress::{
    kpoint_hcore_stress_fixed_d, kpoint_rhf_stress, kpoint_uhf_stress, KStress, KStressConfig,
    KStressJk, KStressMutation,
};
use ferric_pbc::kuscf::{solve_kuhf_injected, KUScfResult};
use ferric_pbc::lattice::Cell;
use ferric_pbc::sr_rotation::SrColumnRotation;
use ferric_pbc::stress::{gamma_rhf_stress, GammaStress, GammaStressConfig};
use ferric_scf::result::{ScfExit, ScfResult, Spin};
use ndarray::Array2;
use num_complex::Complex64;
use std::collections::HashMap;

type Mat3 = [[f64; 3]; 3];

const OMEGA: f64 = 0.8;
const HCORE_PRECISION: f64 = 1e-14;
const AMPLE: usize = 1 << 31;
const HI_A: [[f64; 3]; 3] = [[6.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 7.0]];
const MOVED: [[f64; 3]; 2] = [[0.3, 0.2, 0.4], [0.9, -0.4, 3.3]];
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

// ------------------------------------------------------------ fixtures
// (pbc_grad_ecp.rs's compact HI; test files cannot share them.)

fn shell(l: i32, exps: &[f64], coefs: &[f64]) -> Shell {
    let lf = l as f64;
    let mut s = 0.0;
    for (a, ca) in exps.iter().zip(coefs) {
        for (b, cb) in exps.iter().zip(coefs) {
            s += ca * cb * (2.0 * (a * b).sqrt() / (a + b)).powf(lf + 1.5);
        }
    }
    Shell {
        l,
        pure: false,
        exponents: exps.to_vec(),
        coefficients: coefs.iter().map(|c| c / s.sqrt()).collect(),
    }
}

fn lanl2dz_i_ecp() -> EcpDef {
    let ch = |l: i32, t: &[(i32, f64, f64)]| EcpShell {
        angular_momentum: l,
        terms: t
            .iter()
            .map(|&(n, z, d)| EcpTerm {
                coef: d,
                r_exp: n,
                gexp: z,
            })
            .collect(),
    };
    EcpDef {
        n_core: 46,
        shells: vec![
            ch(
                3,
                &[
                    (0, 1.0715702, -0.0747621),
                    (1, 44.1936028, -30.0811224),
                    (2, 12.9367609, -75.3722721),
                    (2, 3.1956412, -22.0563758),
                    (2, 0.8589806, -1.6979585),
                ],
            ),
            ch(
                0,
                &[
                    (0, 127.9202670, 2.9380036),
                    (1, 78.6211465, 41.2471267),
                    (2, 36.5146237, 287.8680095),
                    (2, 9.9065681, 114.3758506),
                    (2, 1.9420086, 37.6547714),
                ],
            ),
            ch(
                1,
                &[
                    (0, 13.0035304, 2.2222630),
                    (1, 76.0331404, 39.4090831),
                    (2, 24.1961684, 177.4075002),
                    (2, 6.4053433, 77.9889462),
                    (2, 1.5851786, 25.7547641),
                ],
            ),
            ch(
                2,
                &[
                    (0, 40.4278108, 7.0524360),
                    (1, 28.9084375, 33.3041635),
                    (2, 15.6268936, 186.9453875),
                    (2, 4.1442856, 71.9688361),
                    (2, 0.9377235, 9.3630657),
                ],
            ),
        ],
    }
}

/// H STO-3G + compact I s+p, LANL2DZ ECP on I.
fn hi_basis() -> BasisSet {
    let mut shells = HashMap::new();
    shells.insert(
        1,
        vec![shell(
            0,
            &[3.42525091, 0.62391373, 0.1688554],
            &[0.15432897, 0.53532814, 0.44463454],
        )],
    );
    shells.insert(
        53,
        vec![shell(0, &[0.4653], &[1.0]), shell(1, &[0.318], &[1.0])],
    );
    let mut ecps = HashMap::new();
    ecps.insert(53, lanl2dz_i_ecp());
    BasisSet {
        name: "HI-compact".into(),
        shells,
        ecps,
    }
}

fn atom(symbol: &str, z: i32, r: [f64; 3]) -> Atom {
    Atom {
        symbol: symbol.into(),
        z,
        x: r[0],
        y: r[1],
        zpos: r[2],
        ghost: false,
        n_core_ecp: 0,
    }
}

/// One cell of `species` at `pos`, repeated at each of `shifts` (Bohr), through
/// `apply_ecp(bs)`, with spin multiplicity `mult`.
fn ecp_cell(
    species: &[(&str, i32)],
    pos: &[[f64; 3]],
    shifts: &[[f64; 3]],
    lattice: [[f64; 3]; 3],
    bs: &BasisSet,
    mult: usize,
) -> Cell {
    let mut atoms = Vec::new();
    for t in shifts {
        for ((sym, z), r) in species.iter().zip(pos) {
            atoms.push(atom(sym, *z, [r[0] + t[0], r[1] + t[1], r[2] + t[2]]));
        }
    }
    let mut mol = Molecule {
        atoms,
        charge: 0,
        multiplicity: mult,
    };
    mol.apply_ecp(bs);
    Cell::new(mol, lattice).expect("ECP cell")
}

/// The z translations of the diag(1, 1, n) supercell copies.
fn z_shifts(n: usize) -> Vec<[f64; 3]> {
    (0..n).map(|m| [0.0, 0.0, m as f64 * HI_A[2][2]]).collect()
}

/// The diag(1, 1, n) supercell lattice.
fn z_lattice(n: usize) -> [[f64; 3]; 3] {
    let mut lat = HI_A;
    lat[2][2] *= n as f64;
    lat
}

const HI_SPECIES: [(&str, i32); 2] = [("H", 1), ("I", 53)];

fn hi_cell(pos: &[[f64; 3]], bs: &BasisSet) -> Cell {
    ecp_cell(&HI_SPECIES, pos, &[[0.0; 3]], HI_A, bs, 1)
}

/// The explicit diag(1, 1, n) supercell of the HI cell.
fn hi_supercell(pos: &[[f64; 3]], n: usize, bs: &BasisSet) -> Cell {
    ecp_cell(&HI_SPECIES, pos, &z_shifts(n), z_lattice(n), bs, 1)
}

fn moved(pos: &[[f64; 3]], a: usize, x: usize, h: f64) -> Vec<[f64; 3]> {
    let mut p = pos.to_vec();
    p[a][x] += h;
    p
}

fn hcore_cfg() -> PeriodicHcoreConfig {
    PeriodicHcoreConfig {
        precision: HCORE_PRECISION,
        ..PeriodicHcoreConfig::with_omega(OMEGA)
    }
    .with_sr_column_rotation(SrColumnRotation::Off)
}

fn ecp_cfg() -> PeriodicEcpConfig {
    PeriodicEcpConfig::with_precision(HCORE_PRECISION)
}

fn kstress_cfg() -> KStressConfig {
    KStressConfig {
        budget_bytes: Some(AMPLE),
        ..Default::default()
    }
}

fn eps_at(i: usize, j: usize, h: f64) -> Mat3 {
    let mut e = [[0.0; 3]; 3];
    e[i][j] = h;
    e
}

fn max_mat(a: &Mat3, b: &Mat3) -> f64 {
    let mut m = 0.0_f64;
    for i in 0..3 {
        for j in 0..3 {
            m = m.max((a[i][j] - b[i][j]).abs());
        }
    }
    m
}

fn mat_max(a: &Mat3) -> f64 {
    a.iter().flatten().fold(0.0_f64, |m, x| m.max(x.abs()))
}

// ------------------------------------------------------------ densities

/// A fixed Hermitian `D(k)` with `D(−k) = D(k)*` that is NOT a one-point
/// density: `D(k) = B0 + e^{ik·a3} B1 + e^{−ik·a3} B1ᵀ` with real `B0`
/// (symmetric) and `B1`, so its real-space blocks couple neighbouring cells
/// along z. Complex at every non-TRIM k of a 1x1x3 mesh.
fn fixed_dk(mesh: &KPointMesh, n: usize) -> Vec<Array2<Complex64>> {
    let b0 = Array2::from_shape_fn((n, n), |(i, j)| {
        let (a, b) = (i as f64, j as f64);
        0.3 * (1.1 * a + 0.7 * b).cos()
            + 0.3 * (1.1 * b + 0.7 * a).cos()
            + if i == j { 0.5 } else { 0.0 }
    });
    let b1 = Array2::from_shape_fn((n, n), |(i, j)| {
        0.12 * (0.9 * i as f64 - 0.4 * j as f64 + 0.3).sin()
    });
    (0..mesh.nk())
        .map(|k| {
            let ph = mesh.phase(k, [0, 0, 1]);
            Array2::from_shape_fn((n, n), |(i, j)| {
                Complex64::new(b0[(i, j)], 0.0) + ph * b1[(i, j)] + ph.conj() * b1[(j, i)]
            })
        })
        .collect()
}

/// The k density unfolded to the explicit z-supercell:
/// `D_sc[(c,m),(c',n)] = (1/N_k) Σ_k e^{ik·(T_c − T_c')} D(k)_mn` (real).
fn unfold_z(mesh: &KPointMesh, dk: &[Array2<Complex64>]) -> Array2<f64> {
    let nc = mesh.n()[2];
    assert_eq!(mesh.n()[0] * mesh.n()[1], 1);
    let nao = dk[0].nrows();
    let nk = mesh.nk();
    let mut out = Array2::<f64>::zeros((nc * nao, nc * nao));
    let mut max_im = 0.0_f64;
    for c in 0..nc {
        for c2 in 0..nc {
            let nl = [0, 0, c as i64 - c2 as i64];
            for m in 0..nao {
                for nn in 0..nao {
                    let mut z = Complex64::new(0.0, 0.0);
                    for (k, d) in dk.iter().enumerate() {
                        z += mesh.phase(k, nl) * d[(m, nn)];
                    }
                    z /= nk as f64;
                    max_im = max_im.max(z.im.abs());
                    out[(c * nao + m, c2 * nao + nn)] = z.re;
                }
            }
        }
    }
    assert!(max_im < 1e-10, "unfolded density not real: {max_im:e}");
    out
}

// ------------------------------------------------------------ energies

/// `(1/N_k) Σ_k Re tr[D(k) V_ECP(k)]`: ferric's own k-point ECP lattice sum
/// on `cell` with the mesh rebuilt from the integer numerators `n`.
fn e_ecp_k(cell: &Cell, bs: &BasisSet, n: [usize; 3], dk: &[Array2<Complex64>]) -> f64 {
    let p = prep_for(cell, bs);
    let mesh = KPointMesh::gamma_centred(cell, n).expect("mesh");
    let v = periodic_ecp_images(cell, &p, &ecp_cfg())
        .expect("ecp images")
        .expect("ECP centre present")
        .at_kpts(cell, &mesh)
        .expect("V_ECP(k)");
    let mut e = 0.0;
    for (d, vk) in dk.iter().zip(&v) {
        for i in 0..d.nrows() {
            for j in 0..d.ncols() {
                e += (d[(i, j)] * vk[(j, i)]).re;
            }
        }
    }
    e / mesh.nk() as f64
}

struct KSys {
    cell: Cell,
    prep: PreparedBasis,
    mesh: KPointMesh,
    hk: PeriodicHcoreK,
}

fn ksys(cell: Cell, bs: &BasisSet, n: [usize; 3]) -> KSys {
    let prep = prep_for(&cell, bs);
    let mesh = KPointMesh::gamma_centred(&cell, n).expect("mesh");
    let hk = periodic_hcore_kpts(&cell, &prep, &mesh, &hcore_cfg()).expect("hcore(k)");
    assert!(hk.n_ecp_triples > 0, "no ECP triples: the cell is vacuous");
    KSys {
        cell,
        prep,
        mesh,
        hk,
    }
}

// ============================================================ (1) term gates

/// Richardson of central FDs at `h` and `h/2`.
fn richardson(f1: f64, f2: f64) -> f64 {
    (4.0 * f2 - f1) / 3.0
}

/// PHYSICS: `dE_ECP/dR_A` of `(1/N_k) Σ_k Re tr[D(k) V_ECP(k)]` equals the
/// FD of ferric's own lattice sum. ARTIFACT: a wrong image weight (conjugate,
/// transposed, missing phase) changes the analytic value but NOT the FD,
/// because the FD differentiates the same `V_ECP(k)` the energy uses;
/// `D(k)` is complex at the two non-TRIM points, so the phases are exercised.
#[test]
fn ecp_force_at_fixed_density_matches_fd_hi_1x1x3() {
    let bs = hi_basis();
    let cell = hi_cell(&MOVED, &bs);
    let n = [1, 1, 3];
    let ks = ksys(cell.clone(), &bs, n);
    let dk = fixed_dk(&ks.mesh, ks.prep.nbasis());
    let t0 = std::time::Instant::now();
    let g = kpoint_ecp_gradient_fixed_d(&ks.cell, &ks.prep, &ks.mesh, &hcore_cfg(), &dk)
        .unwrap()
        .unwrap();
    eprintln!(
        "ECP force at fixed D: {} triples, {} calls, {:.1} s",
        g.n_triples,
        g.n_calls,
        t0.elapsed().as_secs_f64()
    );
    assert_eq!(g.n_triples, ks.hk.n_ecp_triples);
    let fd = |a: usize, x: usize, h: f64| {
        let ep = e_ecp_k(&hi_cell(&moved(&MOVED, a, x, h), &bs), &bs, n, &dk);
        let em = e_ecp_k(&hi_cell(&moved(&MOVED, a, x, -h), &bs), &bs, n, &dk);
        (ep - em) / (2.0 * h)
    };
    let mut worst = 0.0_f64;
    for (a, x) in [(0usize, 2usize), (1, 0)] {
        let an = g.grad[(a, x)];
        let (f1, f2) = (fd(a, x, 1e-3), fd(a, x, 5e-4));
        let rich = richardson(f1, f2);
        let e1 = (an - f1).abs();
        let er = (an - rich).abs();
        eprintln!(
            "  atom {a} x{x}: analytic {an:.12e} bra {:.3e} ket {:.3e} centre {:.3e}; \
             |an − FD(1e-3)| {e1:.2e}, |an − Richardson| {er:.2e}",
            g.bra[(a, x)],
            g.ket[(a, x)],
            g.centre[(a, x)]
        );
        assert!(an.abs() > 1e-6, "vacuous component");
        worst = worst.max(er);
    }
    let net = (0..3)
        .map(|c| g.grad.column(c).sum().abs())
        .fold(0.0_f64, f64::max);
    eprintln!("  worst Richardson miss {worst:.2e}; ΣF {net:.1e}");
    assert!(worst < 1e-7, "{worst:e}");
    assert!(net < 1e-11, "ΣF {net:e}");
}

/// The strain analogue of the force gate: all nine components of the ECP
/// strain at fixed `D(k)` (`kpoint_hcore_stress_fixed_d(..).ecp`) vs FD on
/// `Cell::strained` cells with the mesh rebuilt from the same numerators.
#[test]
fn ecp_strain_at_fixed_density_matches_fd_hi_1x1x3() {
    let bs = hi_basis();
    let cell = hi_cell(&MOVED, &bs);
    let n = [1, 1, 3];
    let ks = ksys(cell.clone(), &bs, n);
    let dk = fixed_dk(&ks.mesh, ks.prep.nbasis());
    let s = kpoint_hcore_stress_fixed_d(
        &ks.cell,
        &ks.prep,
        &ks.mesh,
        &hcore_cfg(),
        &ks.hk,
        &dk,
        &kstress_cfg(),
    )
    .unwrap();
    assert!(mat_max(&s.ecp) > 1e-3, "vacuous strain term");
    assert_eq!(s.n_ecp_triples, ks.hk.n_ecp_triples);
    let fd = |h: f64| -> Mat3 {
        let mut o = [[0.0; 3]; 3];
        for (i, j) in ALL9 {
            let cp = cell.strained(&eps_at(i, j, h)).unwrap();
            let cm = cell.strained(&eps_at(i, j, -h)).unwrap();
            o[i][j] = (e_ecp_k(&cp, &bs, n, &dk) - e_ecp_k(&cm, &bs, n, &dk)) / (2.0 * h);
        }
        o
    };
    let (f1, f2) = (fd(1e-3), fd(5e-4));
    let mut rich = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            rich[i][j] = richardson(f1[i][j], f2[i][j]);
        }
    }
    let (e1, er) = (max_mat(&s.ecp, &f1), max_mat(&s.ecp, &rich));
    eprintln!(
        "ECP strain at fixed D (max {:.3e}): |an − FD(1e-3)| {e1:.2e}, |an − Richardson| {er:.2e}",
        mat_max(&s.ecp)
    );
    assert!(er < 1e-8, "{er:e}");
}

// ============================================================ (2) supercell

/// PHYSICS: per cell the k ECP force on atom A equals the Gamma ECP force of
/// the diag(1,1,3) supercell on ANY copy of A, at the unfolded density, and
/// the k ECP strain equals the supercell's divided by N. Both sides use the
/// same lattice-sum code (one `EcpPlan`), so this tests the image weights
/// and the Bloch phases, not libecpint.
#[test]
fn ecp_term_at_fixed_density_is_the_supercell_term_per_cell() {
    let bs = hi_basis();
    let n = [1, 1, 3];
    let nc = 3usize;
    let ks = ksys(hi_cell(&MOVED, &bs), &bs, n);
    let dk = fixed_dk(&ks.mesh, ks.prep.nbasis());
    let t0 = std::time::Instant::now();
    let gk = kpoint_ecp_gradient_fixed_d(&ks.cell, &ks.prep, &ks.mesh, &hcore_cfg(), &dk)
        .unwrap()
        .unwrap();
    let sk = kpoint_hcore_stress_fixed_d(
        &ks.cell,
        &ks.prep,
        &ks.mesh,
        &hcore_cfg(),
        &ks.hk,
        &dk,
        &kstress_cfg(),
    )
    .unwrap();
    eprintln!("k terms {:.1} s", t0.elapsed().as_secs_f64());
    let sc = hi_supercell(&MOVED, nc, &bs);
    let psc = prep_for(&sc, &bs);
    let dsc = unfold_z(&ks.mesh, &dk);
    let t0 = std::time::Instant::now();
    let gsc = periodic_ecp_gradient(&sc, &psc, &ecp_cfg(), &dsc)
        .unwrap()
        .unwrap();
    let (ssc, _) = periodic_ecp_strain(&sc, &psc, &ecp_cfg(), &dsc)
        .unwrap()
        .unwrap();
    eprintln!("supercell terms {:.1} s", t0.elapsed().as_secs_f64());
    let mut fmiss = 0.0_f64;
    for c in 0..nc {
        for a in 0..2 {
            for x in 0..3 {
                fmiss = fmiss.max((gk.grad[(a, x)] - gsc.grad[(c * 2 + a, x)]).abs());
            }
        }
    }
    let mut ssc_n = ssc;
    for row in ssc_n.iter_mut() {
        for v in row.iter_mut() {
            *v /= nc as f64;
        }
    }
    let smiss = max_mat(&sk.ecp, &ssc_n);
    eprintln!(
        "ECP term, supercell vs k: force miss {fmiss:.2e} (max |F| {:.2e}); strain miss \
         {smiss:.2e} (max {:.2e})",
        gk.grad.iter().fold(0.0_f64, |m, x| m.max(x.abs())),
        mat_max(&sk.ecp)
    );
    assert!(fmiss < 1e-11, "force supercell miss {fmiss:e}");
    assert!(smiss < 1e-11, "strain supercell miss {smiss:e}");
}

// ============================================================ (3) SCF level

const ANCHOR_PRECISION: f64 = 1e-12;
const EXX: [ExxDiv; 2] = [ExxDiv::None, ExxDiv::Ewald];
/// Supercell-anchor bar of the totals (p shells on I). The whole miss is the
/// SR attraction derivative (libint2's precision floor at `SC_NUCLEUS_EXPONENT`,
/// larger here than for H because Z_eff(I) = 7): MEASURED 2026-10-11, HI 1x1x3,
/// force `vsr_nuc` 1.5e-8 and `vsr_basis` 1.5e-9 (7.7e-8 total at the default
/// exponent 1e9), stress total 7.4e-9; every other part (overlap, kinetic,
/// V_LR, ERI, ECP) agrees at <= 1.3e-13. The bar sits three orders below the
/// smallest ECP mutant.
const SC_BAR_P: f64 = 5e-8;
/// Bar of every part other than the SR attraction.
const SC_BAR_PART: f64 = 1e-11;

fn kscf_cfg() -> KScfConfig {
    KScfConfig {
        energy_conv: 1e-13,
        grad_conv: 1e-10,
        max_iter: 400,
        ..Default::default()
    }
}

fn kdense(ks: &KSys, precision: f64) -> KDenseAftEri {
    KDenseAftEri::build(
        &ks.cell,
        &ks.prep,
        &ks.mesh,
        &ks.hk.s,
        ExxDiv::None,
        &KDenseAftConfig {
            precision,
            ..Default::default()
        },
    )
    .expect("dense k kernels")
}

fn krhf_with(
    ks: &KSys,
    hk: &PeriodicHcoreK,
    jk: Box<dyn KPointJk + '_>,
    seed: Option<&[Array2<Complex64>]>,
) -> KScfResult {
    let _ = seed;
    let inj = KPointInjection {
        s: hk.s.clone(),
        h: hk.h.clone(),
        vnn: hk.enn,
        jk,
    };
    let r = solve_krhf_injected(&ks.cell, &ks.mesh, &kscf_cfg(), inj).expect("k-RHF");
    assert!(r.converged, "k-RHF not converged ({} it)", r.iterations);
    r
}

fn krhf(ks: &KSys, jk: Box<dyn KPointJk + '_>) -> KScfResult {
    krhf_with(ks, &ks.hk, jk, None)
}

/// The k-RHF SCF of each exxdiv `[none, ewald]`.
fn scf_pair(ks: &KSys, eri: &KDenseAftEri) -> [KScfResult; 2] {
    let vm = ks.mesh.madelung(&ks.cell).unwrap();
    [
        krhf(ks, Box::new(eri.jk_builder_with_madelung(0.0))),
        krhf(ks, Box::new(eri.jk_builder_with_madelung(vm))),
    ]
}

/// Gaussian-nucleus exponent of the SR attraction DERIVATIVE on both sides of
/// the supercell anchors. With p shells the k mesh and the supercell evaluate
/// that derivative at different image offsets, so libint2's integral
/// precision floor does not cancel; it tracks the exponent (see
/// `pbc_kgrad.rs` `SC_BAR_P`: 1e-9 at 1e8, 2.5e-8 at the default 1e9). The
/// ECP part does not depend on it.
const SC_NUCLEUS_EXPONENT: f64 = 1e8;

fn gcfg(m: Option<KGradMutation>) -> KGradConfig {
    KGradConfig {
        budget_bytes: Some(AMPLE),
        mutation: m,
        nucleus_exponent: Some(SC_NUCLEUS_EXPONENT),
    }
}

fn scfg(m: Option<KStressMutation>) -> KStressConfig {
    KStressConfig {
        budget_bytes: Some(AMPLE),
        mutation: m,
        nucleus_exponent: Some(SC_NUCLEUS_EXPONENT),
    }
}

fn kforce(
    ks: &KSys,
    eri: &KDenseAftEri,
    scf: &KScfResult,
    exx: ExxDiv,
    m: Option<KGradMutation>,
) -> KGradient {
    kpoint_rhf_gradient(
        &ks.cell,
        &ks.prep,
        &ks.mesh,
        &hcore_cfg(),
        &ks.hk,
        KGradJk::Dense(eri),
        scf,
        exx,
        &gcfg(m),
    )
    .expect("kpoint_rhf_gradient")
}

fn kstress(
    ks: &KSys,
    eri: &KDenseAftEri,
    scf: &KScfResult,
    exx: ExxDiv,
    m: Option<KStressMutation>,
) -> KStress {
    kpoint_rhf_stress(
        &ks.cell,
        &ks.prep,
        &ks.mesh,
        &hcore_cfg(),
        &ks.hk,
        KStressJk::Dense(eri),
        scf,
        exx,
        &scfg(m),
    )
    .expect("kpoint_rhf_stress")
}

fn real_of(d: &Array2<Complex64>) -> Array2<f64> {
    let im = d.iter().fold(0.0_f64, |m, z| m.max(z.im.abs()));
    assert!(im < 1e-12, "Gamma density has an imaginary part {im:e}");
    d.mapv(|z| z.re)
}

fn gamma_restricted(d: Array2<f64>) -> ScfResult {
    let n = d.nrows();
    ScfResult {
        spin: Spin::Restricted,
        energy: 0.0,
        density_alpha: &d * 0.5,
        density_total: d,
        density_beta: None,
        mos_alpha: Array2::eye(n),
        mos_beta: None,
        eps_alpha: vec![0.0; n],
        eps_beta: None,
        fock_alpha: Array2::zeros((n, n)),
        fock_beta: None,
        converged: true,
        exit: ScfExit::Converged,
        iterations: 0,
        computed_quartets: 0,
        induced_dipoles: None,
        stability: None,
        stability_external: None,
        df_jk: None,
        cosx_final: None,
        cosx_schedule: None,
        rohf_spin_focks: None,
    }
}

struct GSys {
    cell: Cell,
    prep: PreparedBasis,
    hc: PeriodicHcore,
    eri: DenseAftEri,
}

fn gsys(cell: Cell, bs: &BasisSet) -> GSys {
    let prep = prep_for(&cell, bs);
    let hc = periodic_hcore(&cell, &prep, &hcore_cfg()).expect("gamma hcore");
    assert!(hc.n_ecp_triples > 0, "no ECP triples");
    let eri = DenseAftEri::build(
        &cell,
        &prep,
        &hc.s,
        ExxDiv::None,
        ANCHOR_PRECISION,
        DEFAULT_DENSE_AFT_MAX_BYTES,
    )
    .expect("gamma dense AFT");
    GSys {
        cell,
        prep,
        hc,
        eri,
    }
}

fn gamma_force(g: &GSys, scf: &ScfResult, exx: ExxDiv) -> GammaGradient {
    gamma_rhf_gradient_with(
        &g.cell,
        &g.prep,
        &hcore_cfg(),
        &g.hc,
        &g.eri,
        scf,
        exx,
        &GammaGradConfig {
            budget_bytes: Some(AMPLE),
            nucleus_exponent: Some(SC_NUCLEUS_EXPONENT),
            ..Default::default()
        },
    )
    .expect("gamma gradient")
}

fn gamma_stress(g: &GSys, scf: &ScfResult, exx: ExxDiv) -> GammaStress {
    gamma_rhf_stress(
        &g.cell,
        &g.prep,
        &hcore_cfg(),
        &g.hc,
        &g.eri,
        scf,
        exx,
        &GammaStressConfig {
            budget_bytes: Some(AMPLE),
            nucleus_exponent: Some(SC_NUCLEUS_EXPONENT),
            ..Default::default()
        },
    )
    .expect("gamma stress")
}

fn mat_scaled(m: &Mat3, f: f64) -> Mat3 {
    let mut o = *m;
    for r in o.iter_mut() {
        for v in r.iter_mut() {
            *v *= f;
        }
    }
    o
}

/// `max |F_k[A, x] − F_sc[c·natoms + A, x]|` over every copy `c`.
fn sc_miss(gk: &Array2<f64>, gsc: &Array2<f64>) -> f64 {
    let na = gk.nrows();
    let nc = gsc.nrows() / na;
    let mut w = 0.0_f64;
    for c in 0..nc {
        for a in 0..na {
            for x in 0..3 {
                w = w.max((gk[(a, x)] - gsc[(c * na + a, x)]).abs());
            }
        }
    }
    w
}

fn amax(a: &Array2<f64>) -> f64 {
    a.iter().fold(0.0_f64, |m, x| m.max(x.abs()))
}

/// 1x1x1 mesh vs the Gamma code. PHYSICS: a one-point Gamma mesh IS the
/// Gamma calculation, so forces, stress and the ECP parts agree to roundoff.
/// ARTIFACT: independent constructions (complex residue-weighted walk vs
/// the real Gamma walk), but BLIND to the k-specific defects (every phase is
/// 1 on a one-point mesh): the supercell anchor carries those.
#[test]
fn one_point_mesh_ecp_force_and_stress_are_the_gamma_ones() {
    let bs = hi_basis();
    let ks = ksys(hi_cell(&MOVED, &bs), &bs, [1, 1, 1]);
    let eri = kdense(&ks, ANCHOR_PRECISION);
    let scfs = scf_pair(&ks, &eri);
    let g = gsys(hi_cell(&MOVED, &bs), &bs);
    for (i, exx) in EXX.into_iter().enumerate() {
        let gscf = gamma_restricted(real_of(&scfs[i].densities[0]));
        let kf = kforce(&ks, &eri, &scfs[i], exx, None);
        let gf = gamma_force(&g, &gscf, exx);
        let ks_ = kstress(&ks, &eri, &scfs[i], exx, None);
        let gs = gamma_stress(&g, &gscf, exx);
        let f_ecp = (&kf.parts.ecp - &gf.parts.ecp)
            .iter()
            .fold(0.0_f64, |m, x| m.max(x.abs()));
        let f_tot = (&kf.grad - &gf.grad)
            .iter()
            .fold(0.0_f64, |m, x| m.max(x.abs()));
        let s_ecp = max_mat(&ks_.parts.ecp, &gs.parts.ecp);
        let s_tot = max_mat(&ks_.de_deps, &gs.de_deps);
        eprintln!(
            "1x1x1 {exx:?}: force ecp part {f_ecp:.2e} total {f_tot:.2e} (|F_ecp| {:.2e}); stress \
             ecp part {s_ecp:.2e} total {s_tot:.2e} (|S_ecp| {:.2e})",
            amax(&kf.parts.ecp),
            mat_max(&ks_.parts.ecp)
        );
        assert!(
            amax(&kf.parts.ecp) > 1e-3 && mat_max(&ks_.parts.ecp) > 1e-3,
            "vacuous ECP"
        );
        assert_eq!(kf.n_ecp_triples, ks.hk.n_ecp_triples);
        assert!(f_ecp < 1e-12, "force ECP part {f_ecp:e}");
        assert!(s_ecp < 1e-12, "stress ECP part {s_ecp:e}");
        assert!(f_tot < 1e-9, "force total {f_tot:e}");
        assert!(s_tot < 1e-9, "stress total {s_tot:e}");
    }
}

/// PHYSICS: the 1x1x3 Gamma-centred mesh IS the diag(1,1,3) supercell, so
/// per cell the force on atom A equals the supercell force on any copy of A
/// and `dE/dε = dE/dε_sc / 3`; the same for the ECP part alone. ARTIFACT: a
/// wrong phase / conjugate / residue in the ECP walk would not reproduce the
/// explicit supercell (non-TRIM mesh).
fn supercell_anchor() -> (
    KSys,
    KDenseAftEri,
    [KScfResult; 2],
    [GammaGradient; 2],
    [GammaStress; 2],
) {
    let bs = hi_basis();
    let n = [1, 1, 3];
    let ks = ksys(hi_cell(&MOVED, &bs), &bs, n);
    let eri = kdense(&ks, ANCHOR_PRECISION);
    let scfs = scf_pair(&ks, &eri);
    let g = gsys(hi_supercell(&MOVED, 3, &bs), &bs);
    let mut gfs = Vec::new();
    let mut gss = Vec::new();
    for (i, exx) in EXX.into_iter().enumerate() {
        let gscf = gamma_restricted(unfold_z(&ks.mesh, &scfs[i].densities));
        gfs.push(gamma_force(&g, &gscf, exx));
        gss.push(gamma_stress(&g, &gscf, exx));
    }
    let gs: [GammaStress; 2] = [gss.remove(0), gss.remove(0)];
    let gf: [GammaGradient; 2] = [gfs.remove(0), gfs.remove(0)];
    (ks, eri, scfs, gf, gs)
}

#[test]
#[ignore = "slow (~2.5 min release, loaded box): HI 1x1x3 k SCFs + the 3-cell Gamma supercell (6 atoms) hcore/dense AFT/force/stress x2 exxdiv"]
fn hi_1x1x3_force_and_stress_are_the_supercell_ones_per_cell() {
    let t0 = std::time::Instant::now();
    let (ks, eri, scfs, gf, gs) = supercell_anchor();
    eprintln!("supercell setup {:.1} s", t0.elapsed().as_secs_f64());
    for (i, exx) in EXX.into_iter().enumerate() {
        let kf = kforce(&ks, &eri, &scfs[i], exx, None);
        let ksr = kstress(&ks, &eri, &scfs[i], exx, None);
        let f_ecp = sc_miss(&kf.parts.ecp, &gf[i].parts.ecp);
        let f_tot = sc_miss(&kf.grad, &gf[i].grad);
        let s_ecp = max_mat(&ksr.parts.ecp, &mat_scaled(&gs[i].parts.ecp, 1.0 / 3.0));
        let s_tot = max_mat(&ksr.de_deps, &mat_scaled(&gs[i].de_deps, 1.0 / 3.0));
        eprintln!(
            "HI 1x1x3 {exx:?}: force ecp {f_ecp:.2e} total {f_tot:.2e}; stress ecp {s_ecp:.2e} \
             total {s_tot:.2e}; |F_ecp| {:.2e} |S_ecp| {:.2e}",
            amax(&kf.parts.ecp),
            mat_max(&ksr.parts.ecp)
        );
        let parts = [
            ("overlap", sc_miss(&kf.parts.overlap, &gf[i].parts.overlap)),
            ("kinetic", sc_miss(&kf.parts.kinetic, &gf[i].parts.kinetic)),
            (
                "vsr_basis",
                sc_miss(&kf.parts.vsr_basis, &gf[i].parts.vsr_basis),
            ),
            ("vsr_nuc", sc_miss(&kf.parts.vsr_nuc, &gf[i].parts.vsr_nuc)),
            (
                "vlr_basis",
                sc_miss(&kf.parts.vlr_basis, &gf[i].parts.vlr_basis),
            ),
            ("vlr_nuc", sc_miss(&kf.parts.vlr_nuc, &gf[i].parts.vlr_nuc)),
            (
                "eri",
                sc_miss(&(&kf.parts.coulomb + &kf.parts.exchange), &gf[i].parts.eri),
            ),
        ];
        eprintln!("  force parts vs supercell: {parts:?}");
        for (name, miss) in &parts {
            let bar = if name.starts_with("vsr") {
                SC_BAR_P
            } else {
                SC_BAR_PART
            };
            assert!(
                *miss < bar,
                "force part {name} misses the supercell by {miss:e}"
            );
        }
        assert!(
            f_ecp < SC_BAR_PART && s_ecp < SC_BAR_PART,
            "ECP part misses the supercell"
        );
        assert!(f_tot < SC_BAR_P, "force total miss {f_tot:e}");
        assert!(s_tot < SC_BAR_P, "stress total miss {s_tot:e}");
    }
}

// ============================================================ mutants

/// Smallest miss a mutant must make (the repo's `MUTANT_BAR`); the anchors
/// sit at <= 5e-9, so this leaves four orders of magnitude.
const MUTANT_BAR: f64 = 1e-4;

fn grad_miss(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    (a - b).iter().fold(0.0_f64, |m, x| m.max(x.abs()))
}

/// The ECP-term references of the unmutated 1x1x3 SCF density: the Gamma
/// supercell ECP force / strain at the unfolded density (cheap: no J/K, no
/// hcore), which is the supercell anchor of the ECP PART. A mutant of the
/// ECP walk changes only the ECP part of the total, whose other parts are
/// anchored by `hi_1x1x3_force_and_stress_are_the_supercell_ones_per_cell`,
/// so the miss of the total equals the miss of the part.
struct EcpRefs {
    force: Array2<f64>,
    strain: Mat3,
}

fn ecp_refs(ks: &KSys, scf: &KScfResult, bs: &BasisSet) -> EcpRefs {
    let sc = hi_supercell(&MOVED, 3, bs);
    let psc = prep_for(&sc, bs);
    let dsc = unfold_z(&ks.mesh, &scf.densities);
    let g = periodic_ecp_gradient(&sc, &psc, &ecp_cfg(), &dsc)
        .unwrap()
        .unwrap();
    let (s, _) = periodic_ecp_strain(&sc, &psc, &ecp_cfg(), &dsc)
        .unwrap()
        .unwrap();
    // One copy of each atom (all copies are equal, asserted by the anchor).
    EcpRefs {
        force: g.grad.slice(ndarray::s![0..2, ..]).to_owned(),
        strain: mat_scaled(&s, 1.0 / 3.0),
    }
}

/// PHYSICS: every defect of the ECP walk must be visible against the
/// supercell anchor of the ECP part (and so of the total). The k-specific
/// ones (`NoPhase`, `ConjD`, `EcpWrongResidue`) need the non-TRIM mesh.
/// Blind spots are asserted where they exist, with their measured size.
#[test]
#[ignore = "slow (~3-4 min release, loaded box): HI 1x1x3 k SCF + 9 force and 8 stress mutant evaluations"]
fn ecp_mutants_miss_the_supercell_anchor() {
    let bs = hi_basis();
    let ks = ksys(hi_cell(&MOVED, &bs), &bs, [1, 1, 3]);
    let eri = kdense(&ks, ANCHOR_PRECISION);
    let scf = krhf(&ks, Box::new(eri.jk_builder_with_madelung(0.0)));
    let refs = ecp_refs(&ks, &scf, &bs);
    let base_f = kforce(&ks, &eri, &scf, ExxDiv::None, None);
    let base_s = kstress(&ks, &eri, &scf, ExxDiv::None, None);
    let f0 = grad_miss(&base_f.parts.ecp, &refs.force);
    let s0 = max_mat(&base_s.parts.ecp, &refs.strain);
    eprintln!("unmutated ECP part vs supercell: force {f0:.2e}, strain {s0:.2e}");
    assert!(f0 < 1e-11 && s0 < 1e-11);
    for m in [
        KGradMutation::NoEcp,
        KGradMutation::EcpNoCentre,
        KGradMutation::EcpCentreSign,
        KGradMutation::EcpCentreWrongAtom,
        KGradMutation::EcpWrongResidue,
        KGradMutation::EcpL0Only,
        KGradMutation::EcpM0Only,
        KGradMutation::NoPhase,
        KGradMutation::ConjD,
    ] {
        let f = kforce(&ks, &eri, &scf, ExxDiv::None, Some(m));
        let miss = grad_miss(&f.parts.ecp, &refs.force);
        let dtot = grad_miss(&f.grad, &base_f.grad);
        eprintln!("  force mutant {m:?}: ECP part misses by {miss:.2e}; total moves by {dtot:.2e}");
        assert!(miss > MUTANT_BAR, "{m:?} escaped: {miss:e}");
    }
    for m in [
        KStressMutation::NoEcp,
        KStressMutation::EcpNoCentre,
        KStressMutation::EcpCentreSign,
        KStressMutation::EcpNoKetImage,
        KStressMutation::EcpUnitPairVector,
        KStressMutation::EcpWrongResidue,
        KStressMutation::NoPhase,
        KStressMutation::ConjD,
    ] {
        let s = kstress(&ks, &eri, &scf, ExxDiv::None, Some(m));
        let miss = max_mat(&s.parts.ecp, &refs.strain);
        let dtot = max_mat(&s.de_deps, &base_s.de_deps);
        eprintln!(
            "  stress mutant {m:?}: ECP part misses by {miss:.2e}; total moves by {dtot:.2e}"
        );
        assert!(miss > MUTANT_BAR, "{m:?} escaped: {miss:e}");
    }
}

// ============================================================ guards

/// The same hcore without its ECP: `h` stripped of `V_ECP`, `v_ecp` cleared.
fn strip_ecp(hk: &PeriodicHcoreK) -> PeriodicHcoreK {
    let mut o = hk.clone();
    for (h, v) in o.h.iter_mut().zip(hk.v_ecp.as_ref().expect("v_ecp")) {
        *h -= v;
    }
    o.v_ecp = None;
    o.n_ecp_triples = 0;
    o
}

/// Fock-consistency guard with an ECP in the rebuilt `F` (`hk.h` carries
/// `V_ECP`). BOTH SIDES of the bar: the matching SCF reproduces the stored
/// Fock to roundoff and is accepted; an SCF run WITHOUT the ECP (a stored
/// Fock lacking `V_ECP`) is refused by the stress and shows a large
/// commutator in the force; an `hk` that lost its `v_ecp` marker while the
/// basis carries an ECP is refused by both.
#[test]
fn ecp_fock_guard_refuses_an_scf_run_without_the_ecp() {
    let bs = hi_basis();
    let ks = ksys(hi_cell(&MOVED, &bs), &bs, [1, 1, 3]);
    let eri = kdense(&ks, 1e-10);
    let good = krhf(&ks, Box::new(eri.jk_builder_with_madelung(0.0)));
    let s = kstress(&ks, &eri, &good, ExxDiv::None, None);
    let f = kforce(&ks, &eri, &good, ExxDiv::None, None);
    eprintln!(
        "matching SCF: Fock mismatch {:.2e}, force commutator {:.2e}",
        s.fock_mismatch, f.commutator
    );
    assert!(s.fock_mismatch < 1e-10 && f.commutator < 1e-7);
    // SCF without the ECP.
    let hk0 = strip_ecp(&ks.hk);
    let bad = krhf_with(&ks, &hk0, Box::new(eri.jk_builder_with_madelung(0.0)), None);
    let e = kpoint_rhf_stress(
        &ks.cell,
        &ks.prep,
        &ks.mesh,
        &hcore_cfg(),
        &ks.hk,
        KStressJk::Dense(&eri),
        &bad,
        ExxDiv::None,
        &scfg(None),
    )
    .unwrap_err()
    .to_string();
    eprintln!("stress on an ECP-free SCF: {e}");
    assert!(e.contains("differs from the SCF's stored"), "{e}");
    let fb = kforce(&ks, &eri, &bad, ExxDiv::None, None);
    eprintln!("force commutator of an ECP-free SCF: {:.2e}", fb.commutator);
    assert!(fb.commutator > 1e-3, "{:e}", fb.commutator);
    // hk with the v_ecp marker cleared (h still carries the ECP).
    let mut hk_lost = ks.hk.clone();
    hk_lost.v_ecp = None;
    for call_stress in [true, false] {
        let e = if call_stress {
            kpoint_rhf_stress(
                &ks.cell,
                &ks.prep,
                &ks.mesh,
                &hcore_cfg(),
                &hk_lost,
                KStressJk::Dense(&eri),
                &good,
                ExxDiv::None,
                &scfg(None),
            )
            .unwrap_err()
            .to_string()
        } else {
            kpoint_rhf_gradient(
                &ks.cell,
                &ks.prep,
                &ks.mesh,
                &hcore_cfg(),
                &hk_lost,
                KGradJk::Dense(&eri),
                &good,
                ExxDiv::None,
                &gcfg(None),
            )
            .unwrap_err()
            .to_string()
        };
        assert!(e.contains("carries an ECP for a cell atom"), "{e}");
    }
}

// ============================================================ threads

fn in_pool<R: Send>(n: usize, f: impl FnOnce() -> R + Send) -> R {
    rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .build()
        .expect("pool")
        .install(f)
}

fn bits_equal(a: &Array2<f64>, b: &Array2<f64>) -> bool {
    a.dim() == b.dim()
        && a.iter()
            .zip(b.iter())
            .all(|(x, y)| x.to_bits() == y.to_bits())
}

/// The ordered-parallel image loop is bit-identical at 1/2/6 threads for the
/// k force and strain, and the frozen serial image loop gives the same ECP
/// force part bit for bit (`SerialDerivWalks`).
#[test]
fn ecp_k_force_and_strain_are_thread_bit_identical() {
    let bs = hi_basis();
    let ks = ksys(hi_cell(&MOVED, &bs), &bs, [1, 1, 3]);
    let dk = fixed_dk(&ks.mesh, ks.prep.nbasis());
    let run = |n: usize| {
        in_pool(n, || {
            let g = kpoint_ecp_gradient_fixed_d(&ks.cell, &ks.prep, &ks.mesh, &hcore_cfg(), &dk)
                .unwrap()
                .unwrap();
            let s = kpoint_hcore_stress_fixed_d(
                &ks.cell,
                &ks.prep,
                &ks.mesh,
                &hcore_cfg(),
                &ks.hk,
                &dk,
                &kstress_cfg(),
            )
            .unwrap();
            (g.grad, s.ecp)
        })
    };
    let (g1, s1) = run(1);
    for n in [2, 6] {
        let (g, s) = run(n);
        assert!(bits_equal(&g, &g1), "force differs at {n} threads");
        assert_eq!(s, s1, "strain differs at {n} threads");
    }
    // Serial oracle vs ordered-parallel, through the production entry.
    let eri = kdense(&ks, 1e-10);
    let scf = krhf(&ks, Box::new(eri.jk_builder_with_madelung(0.0)));
    let par = in_pool(6, || kforce(&ks, &eri, &scf, ExxDiv::None, None));
    let ser = in_pool(1, || {
        kforce(
            &ks,
            &eri,
            &scf,
            ExxDiv::None,
            Some(KGradMutation::SerialDerivWalks),
        )
    });
    assert!(
        bits_equal(&par.parts.ecp, &ser.parts.ecp),
        "serial vs parallel ECP force part differ"
    );
}

// ============================================================ FD of the k energy

/// k-RHF energies `[none, ewald]` per cell of a (displaced / strained) cell.
fn k_energies(cell: &Cell, bs: &BasisSet, n: [usize; 3]) -> [f64; 2] {
    let ks = ksys(cell.clone(), bs, n);
    let eri = kdense(&ks, ANCHOR_PRECISION);
    let sc = scf_pair(&ks, &eri);
    [sc[0].energy, sc[1].energy]
}

/// Bar of the SCF-level FD anchors (h = 1e-4). The Gamma ECP anchors measured
/// 2.5e-7 (forces) / 6.3e-8 (stress) at the same h, libecpint's value-kernel
/// jitter (`pbc_grad_ecp.rs`); the ECP mutants miss by >= 1e-2.
const SCF_FD_BAR: f64 = 5e-7;
const FD_H: f64 = 1e-4;

/// PHYSICS: analytic k force vs central FD (h = 1e-4) of ferric's own k
/// energy, both exxdiv, a fresh SCF at every displaced geometry (the analytic
/// value is at the converged density, so the density response drops out).
/// Components: H z (no ECP centre of its own) and I x (bra + ket + centre).
#[test]
#[ignore = "slow (~3 min release, loaded box): 8 displaced HI 1x1x3 k hcore (ECP lattice sum) + dense AFT + 16 k-RHF SCFs"]
fn hi_1x1x3_force_matches_fd_of_own_k_energy() {
    let bs = hi_basis();
    let n = [1, 1, 3];
    let ks = ksys(hi_cell(&MOVED, &bs), &bs, n);
    let eri = kdense(&ks, ANCHOR_PRECISION);
    let scfs = scf_pair(&ks, &eri);
    let comps = [(0usize, 2usize), (1, 0)];
    let mut fd = [[0.0; 2]; 2];
    for (ci, (a, x)) in comps.into_iter().enumerate() {
        let ep = k_energies(&hi_cell(&moved(&MOVED, a, x, FD_H), &bs), &bs, n);
        let em = k_energies(&hi_cell(&moved(&MOVED, a, x, -FD_H), &bs), &bs, n);
        for e in 0..2 {
            fd[ci][e] = (ep[e] - em[e]) / (2.0 * FD_H);
        }
    }
    for (i, exx) in EXX.into_iter().enumerate() {
        let f = kforce(&ks, &eri, &scfs[i], exx, None);
        let net = (0..3)
            .map(|c| f.grad.column(c).sum().abs())
            .fold(0.0_f64, f64::max);
        for (ci, (a, x)) in comps.into_iter().enumerate() {
            let miss = (f.grad[(a, x)] - fd[ci][i]).abs();
            eprintln!(
                "  HI 1x1x3 {exx:?} atom {a} x{x}: analytic {:.10e} FD {:.10e} miss {miss:.2e}; \
                 ECP part {:.3e}",
                f.grad[(a, x)],
                fd[ci][i],
                f.parts.ecp[(a, x)]
            );
            assert!(miss < SCF_FD_BAR, "{exx:?} ({a},{x}) FD miss {miss:e}");
        }
        eprintln!("  ΣF {net:.2e}");
        assert!(net < 1e-7, "ΣF {net:e}");
    }
}

/// The stress analogue: all nine components of `dE/dε` vs central FD (h =
/// 1e-4) of the k energy on `Cell::strained` cells, mesh rebuilt from the same
/// integer numerators, fresh SCF per point, both exxdiv.
#[test]
#[ignore = "slow (~6 min release, loaded box): 18 strained HI 1x1x3 k hcore + dense AFT builds + 36 k-RHF SCFs"]
fn hi_1x1x3_stress_matches_fd_of_own_k_energy_all_nine() {
    let bs = hi_basis();
    let n = [1, 1, 3];
    let cell = hi_cell(&MOVED, &bs);
    let ks = ksys(cell.clone(), &bs, n);
    let eri = kdense(&ks, ANCHOR_PRECISION);
    let scfs = scf_pair(&ks, &eri);
    let mut fd = [[[0.0; 3]; 3]; 2];
    for (i, j) in ALL9 {
        let ep = k_energies(&cell.strained(&eps_at(i, j, FD_H)).unwrap(), &bs, n);
        let em = k_energies(&cell.strained(&eps_at(i, j, -FD_H)).unwrap(), &bs, n);
        for e in 0..2 {
            fd[e][i][j] = (ep[e] - em[e]) / (2.0 * FD_H);
        }
    }
    for (i, exx) in EXX.into_iter().enumerate() {
        let s = kstress(&ks, &eri, &scfs[i], exx, None);
        let miss = max_mat(&s.de_deps, &fd[i]);
        eprintln!(
            "  HI 1x1x3 {exx:?}: |analytic − FD| {miss:.2e} (ECP part max {:.3e})",
            mat_max(&s.parts.ecp)
        );
        assert!(miss < SCF_FD_BAR, "{exx:?}: FD miss {miss:e}");
    }
}

// ============================================================ UHF

fn kuhf_with(ks: &KSys, eri: &KDenseAftEri, na: usize, nb: usize) -> KUScfResult {
    let inj = KPointInjection {
        s: ks.hk.s.clone(),
        h: ks.hk.h.clone(),
        vnn: ks.hk.enn,
        jk: Box::new(eri.jk_builder_with_madelung(0.0)),
    };
    let r = solve_kuhf_injected(&ks.cell, &ks.mesh, &kscf_cfg(), inj, na, nb).expect("k-UHF");
    assert!(r.converged, "k-UHF not converged ({} it)", r.iterations);
    r
}

fn kforce_u(ks: &KSys, eri: &KDenseAftEri, scf: &KUScfResult) -> KGradient {
    kpoint_uhf_gradient(
        &ks.cell,
        &ks.prep,
        &ks.mesh,
        &hcore_cfg(),
        &ks.hk,
        KGradJk::Dense(eri),
        scf,
        ExxDiv::None,
        &gcfg(None),
    )
    .expect("kpoint_uhf_gradient")
}

fn kstress_u(ks: &KSys, eri: &KDenseAftEri, scf: &KUScfResult) -> KStress {
    kpoint_uhf_stress(
        &ks.cell,
        &ks.prep,
        &ks.mesh,
        &hcore_cfg(),
        &ks.hk,
        KStressJk::Dense(eri),
        scf,
        ExxDiv::None,
        &scfg(None),
    )
    .expect("kpoint_uhf_stress")
}

/// A closed-shell UHF (N_α = N_β) is the RHF: forces and stress agree, ECP
/// part included (they share `assemble`; the ECP weight is built from
/// `D_α + D_β`).
#[test]
fn closed_shell_kuhf_ecp_force_and_stress_are_the_krhf_ones() {
    let bs = hi_basis();
    let ks = ksys(hi_cell(&MOVED, &bs), &bs, [1, 1, 3]);
    let eri = kdense(&ks, 1e-10);
    let r = krhf(&ks, Box::new(eri.jk_builder_with_madelung(0.0)));
    let u = kuhf_with(&ks, &eri, 4, 4);
    let (fr, fu) = (
        kforce(&ks, &eri, &r, ExxDiv::None, None),
        kforce_u(&ks, &eri, &u),
    );
    let (sr, su) = (
        kstress(&ks, &eri, &r, ExxDiv::None, None),
        kstress_u(&ks, &eri, &u),
    );
    let (df, ds) = (
        grad_miss(&fr.grad, &fu.grad),
        max_mat(&sr.de_deps, &su.de_deps),
    );
    let (dfe, dse) = (
        grad_miss(&fr.parts.ecp, &fu.parts.ecp),
        max_mat(&sr.parts.ecp, &su.parts.ecp),
    );
    eprintln!(
        "closed-shell UHF vs RHF: force {df:.2e} (ECP {dfe:.2e}), stress {ds:.2e} (ECP {dse:.2e})"
    );
    assert!(df < 1e-7 && ds < 1e-7 && dfe < 1e-8 && dse < 1e-8);
}

/// H2I (9 valence electrons, doublet 5α/4β): a genuinely spin-polarised
/// density. The UHF ECP force and strain parts equal the Gamma supercell ECP
/// terms at the unfolded TOTAL density (`D_α + D_β`): an α-only weight, or a
/// half-weight, would miss.
#[test]
#[ignore = "slow (~2 min release, loaded box): H2I doublet 1x1x3 k-UHF + force + stress, supercell ECP terms at fixed density"]
fn h2i_doublet_uhf_ecp_parts_are_the_supercell_ecp_terms() {
    let bs = hi_basis();
    let species: [(&str, i32); 3] = [("H", 1), ("H", 1), ("I", 53)];
    let pos = [[0.3, 0.2, 0.4], [2.0, 0.6, 1.0], [0.9, -0.4, 3.3]];
    let cell = ecp_cell(&species, &pos, &[[0.0; 3]], HI_A, &bs, 2);
    let n = [1, 1, 3];
    let ks = ksys(cell, &bs, n);
    let eri = kdense(&ks, 1e-10);
    let u = kuhf_with(&ks, &eri, 5, 4);
    let spin = (u.nalpha, u.nbeta);
    let mag: f64 = u
        .density_alpha
        .iter()
        .zip(&u.density_beta)
        .map(|(a, b)| (a - b).iter().map(|z| z.norm()).sum::<f64>())
        .sum();
    eprintln!(
        "H2I k-UHF {spin:?}, Σ|D_α − D_β| {mag:.3}, E {:.8}",
        u.energy
    );
    assert!(mag > 0.5, "not spin polarised");
    let f = kforce_u(&ks, &eri, &u);
    let s = kstress_u(&ks, &eri, &u);
    let sc = ecp_cell(&species, &pos, &z_shifts(3), z_lattice(3), &bs, 2);
    let psc = prep_for(&sc, &bs);
    let dtot: Vec<Array2<Complex64>> = u
        .density_alpha
        .iter()
        .zip(&u.density_beta)
        .map(|(a, b)| a + b)
        .collect();
    let dsc = unfold_z(&ks.mesh, &dtot);
    let g = periodic_ecp_gradient(&sc, &psc, &ecp_cfg(), &dsc)
        .unwrap()
        .unwrap();
    let (ss, _) = periodic_ecp_strain(&sc, &psc, &ecp_cfg(), &dsc)
        .unwrap()
        .unwrap();
    let na = species.len();
    let fref = g.grad.slice(ndarray::s![0..na, ..]).to_owned();
    let fmiss = grad_miss(&f.parts.ecp, &fref);
    let smiss = max_mat(&s.parts.ecp, &mat_scaled(&ss, 1.0 / 3.0));
    eprintln!(
        "H2I UHF ECP parts vs supercell: force {fmiss:.2e} (max {:.2e}), strain {smiss:.2e} (max {:.2e})",
        amax(&f.parts.ecp),
        mat_max(&s.parts.ecp)
    );
    assert!(fmiss < SC_BAR_PART && smiss < SC_BAR_PART);
    // A force on the right atoms: the net ECP force is zero.
    let net = (0..3)
        .map(|c| f.parts.ecp.column(c).sum().abs())
        .fold(0.0_f64, f64::max);
    assert!(net < 1e-11, "ΣF_ecp {net:e}");
}
