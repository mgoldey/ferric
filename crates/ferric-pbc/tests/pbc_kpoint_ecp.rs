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
//! (3) SCF level: 1x1x1 mesh vs the Gamma code; the 1x1x3 mesh vs the explicit
//!     supercell per part and in total (RHF, both exxdiv); FD of the k energy
//!     (force and all nine stress components, dense AFT; RS-GDF force and
//!     three strain components); closed-shell UHF == RHF and a spin-polarised
//!     H2I doublet's ECP parts vs the supercell.
//! (4) The Fock-consistency guard with the ECP in the rebuilt `F`, thread
//!     bit-identity (1/2/6 threads) and the serial image-loop oracle, a
//!     shifted (Monkhorst-Pack) mesh at fixed density, and 17 production-wired
//!     mutants of the ECP walk against the supercell anchor of the ECP part.
//!
//! NOT tested (safe by construction per review, but unmeasured): more than one
//! ECP atom per cell, triclinic or non-orthogonal cells, a pure-d (l = 2)
//! orbital shell on the ECP element, an ECP with a range-split RS-GDF source
//! (the ECP call in `one_electron_pass` is source-independent), a non-I ECP.
//!
//! Measured times are in each `#[ignore]` reason. The non-ignored set (about
//! 220 s in total, every test <= 67 s) is: the two fixed-density FD gates, the
//! fixed-density supercell anchor with its mutant sweep, the exact shifted-mesh
//! identity (and its mesh-count-modulus mutant), the per-part supercell
//! anchor of force and stress (exxdiv none), the Fock/`hk` guards, thread bit-
//! identity, the dense force FD and the RS-GDF force + strain FD.
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
    kpoint_ecp_gradient_fixed_d, kpoint_ecp_gradient_fixed_d_with, kpoint_rhf_gradient,
    kpoint_uhf_gradient, KGradConfig, KGradJk, KGradMutation, KGradient, KRsGdfGradSource,
};
use ferric_pbc::kpts::{KPointMesh, MeshCentring};
use ferric_pbc::kscf::{solve_krhf_injected, KPointInjection, KPointJk, KScfConfig, KScfResult};
use ferric_pbc::kstress::{
    kpoint_hcore_stress_fixed_d, kpoint_rhf_stress, kpoint_uhf_stress, KStress, KStressConfig,
    KStressJk, KStressMutation,
};
use ferric_pbc::kuscf::{solve_kuhf_injected, KUScfResult};
use ferric_pbc::lattice::Cell;
use ferric_pbc::rsgdf::kpoint::{KRsGdf, KRsGdfConfig};
use ferric_pbc::rsgdf::RsGdfConfig;
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
const CONTROL_SPECIES: [(&str, i32); 2] = [("H", 1), ("N", 7)];

/// The species of a cell of `bs`: HI for the ECP basis, H + a Z = 7 nucleus
/// ("N") for the no-ECP control.
fn species_of(bs: &BasisSet) -> [(&'static str, i32); 2] {
    if bs.ecps.is_empty() {
        CONTROL_SPECIES
    } else {
        HI_SPECIES
    }
}

/// CONTROL of `tests/pbc_grad_ecp_nucleus_scan.rs`: the same H and I shells
/// (I's moved to a Z = 7 nucleus), NO ECP. Same nuclear charges, electron
/// count and AO layout as [`hi_basis`]; the ECP operator is the only
/// difference.
fn control_basis() -> BasisSet {
    let mut b = hi_basis();
    let i_shells = b.shells.remove(&53).expect("I shells");
    b.shells.insert(7, i_shells);
    b.ecps.clear();
    b.name = "HI-control-Z7".into();
    b
}

fn hi_cell(pos: &[[f64; 3]], bs: &BasisSet) -> Cell {
    ecp_cell(&species_of(bs), pos, &[[0.0; 3]], HI_A, bs, 1)
}

/// The explicit diag(1, 1, n) supercell of the HI cell.
fn hi_supercell(pos: &[[f64; 3]], n: usize, bs: &BasisSet) -> Cell {
    ecp_cell(&species_of(bs), pos, &z_shifts(n), z_lattice(n), bs, 1)
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
    e_ecp_k_c(cell, bs, n, MeshCentring::Gamma, dk)
}

/// [`e_ecp_k`] on a mesh of any centring.
fn e_ecp_k_c(
    cell: &Cell,
    bs: &BasisSet,
    n: [usize; 3],
    centring: MeshCentring,
    dk: &[Array2<Complex64>],
) -> f64 {
    let p = prep_for(cell, bs);
    let mesh = KPointMesh::new(cell, n, centring).expect("mesh");
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
    ksys_c(cell, bs, n, MeshCentring::Gamma)
}

fn ksys_c(cell: Cell, bs: &BasisSet, n: [usize; 3], centring: MeshCentring) -> KSys {
    let prep = prep_for(&cell, bs);
    let mesh = KPointMesh::new(&cell, n, centring).expect("mesh");
    let hk = periodic_hcore_kpts(&cell, &prep, &mesh, &hcore_cfg()).expect("hcore(k)");
    assert!(
        bs.ecps.is_empty() || hk.n_ecp_triples > 0,
        "no ECP triples: the cell is vacuous"
    );
    KSys {
        cell,
        prep,
        mesh,
        hk,
    }
}

// ============================================================ (1) term gates

/// Bars of the fixed-density gates (Richardson of FD at h = 1e-3, 5e-4 against
/// the analytic term; the ECP value kernel is exact to ~1e-13, FINDINGS, so
/// the floor is the h^4 residue of the Richardson). Derived from BOTH sides:
/// MEASURED here 2.1e-12 (force, worst of two components), 6.0e-12 (strain,
/// all nine), 2.8e-12 / 3.2e-12 (shifted mesh) -> bars >= 50x above; the
/// smallest ECP mutant moves the same quantities by 2.1e-3 (force, `M0Only`)
/// and 1.1e-1 (strain, `NoKetImage`), i.e. >= 7 orders above the bars.
const FIXED_D_FORCE_BAR: f64 = 1e-10;
const FIXED_D_STRAIN_BAR: f64 = 3e-10;

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
    assert!(worst < FIXED_D_FORCE_BAR, "{worst:e}");
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
    assert!(er < FIXED_D_STRAIN_BAR, "{er:e}");
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
    assert!(fmiss < 1e-12, "force supercell miss {fmiss:e}");
    assert!(smiss < 1e-12, "strain supercell miss {smiss:e}");

    // Mutant sweep at fixed density (cheap, production-wired): each must
    // miss the supercell reference by > MUTANT_BAR (the smallest, EcpM0Only
    // at 2.1e-3 for the force and EcpNoKetImage at 1.1e-1 for the strain, is
    // measured in the SCF-level sweep; the clean value is ~1e-14).
    let fref = gsc.grad.slice(ndarray::s![0..2, ..]).to_owned();
    for m in [
        KGradMutation::EcpNoCentre,
        KGradMutation::EcpCentreSign,
        KGradMutation::EcpCentreWrongAtom,
        KGradMutation::EcpWrongResidue,
        KGradMutation::EcpL0Only,
        KGradMutation::EcpM0Only,
        KGradMutation::NoPhase,
        KGradMutation::ConjD,
    ] {
        let g = kpoint_ecp_gradient_fixed_d_with(
            &ks.cell,
            &ks.prep,
            &ks.mesh,
            &hcore_cfg(),
            &dk,
            Some(m),
        )
        .unwrap()
        .unwrap();
        let miss = grad_miss(&g.grad.slice(ndarray::s![0..2, ..]).to_owned(), &fref);
        eprintln!("  fixed-D force mutant {m:?}: misses by {miss:.2e}");
        assert!(miss > MUTANT_BAR, "{m:?} escaped: {miss:e}");
    }
    for m in [
        KStressMutation::EcpNoCentre,
        KStressMutation::EcpCentreSign,
        KStressMutation::EcpNoKetImage,
        KStressMutation::EcpUnitPairVector,
        KStressMutation::EcpWrongResidue,
        KStressMutation::NoPhase,
    ] {
        let s = kpoint_hcore_stress_fixed_d(
            &ks.cell,
            &ks.prep,
            &ks.mesh,
            &hcore_cfg(),
            &ks.hk,
            &dk,
            &KStressConfig {
                mutation: Some(m),
                ..kstress_cfg()
            },
        )
        .unwrap();
        let miss = max_mat(&s.ecp, &ssc_n);
        eprintln!("  fixed-D strain mutant {m:?}: misses by {miss:.2e}");
        assert!(miss > MUTANT_BAR, "{m:?} escaped: {miss:e}");
    }
}

/// Smallest miss a mutant must make (the repo's `MUTANT_BAR`); the clean
/// anchors sit at <= 5e-8 (SR floor) and the ECP parts at ~1e-14.
const MUTANT_BAR: f64 = 1e-4;

fn grad_miss(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    (a - b).iter().fold(0.0_f64, |m, x| m.max(x.abs()))
}

/// EXACT IDENTITY on a shifted mesh, no FD and no supercell: the Monkhorst-
/// Pack [1,1,2] mesh is {1/4, 3/4} b3, the Gamma-centred [1,1,4] is {0, 1/4,
/// 1/2, 3/4} b3 and [1,1,2] is {0, 1/2} b3, so for ANY smooth `D(k)` (here
/// the same function of k evaluated on each mesh)
/// `G_MP[1,1,2] = 2 G_Γ[1,1,4] − G_Γ[1,1,2]`, for the energy and therefore for
/// its force and strain. The MP mesh's pair-image modulus is 4 (the
/// residue bins of `W_L` have 2N = 4 entries), so a weight read at residue
/// modulo the mesh counts (2) aliases images L and L + 2 and breaks it.
#[test]
fn shifted_mesh_ecp_terms_are_a_gamma_mesh_combination_at_fixed_density() {
    let bs = hi_basis();
    let cell = hi_cell(&MOVED, &bs);
    let mp = ksys_c(cell.clone(), &bs, [1, 1, 2], MeshCentring::MonkhorstPack);
    let g4 = ksys(cell.clone(), &bs, [1, 1, 4]);
    let g2 = ksys(cell.clone(), &bs, [1, 1, 2]);
    assert_eq!(mp.mesh.residue_moduli(), [1, 1, 4]);
    let nao = mp.prep.nbasis();
    let force = |k: &KSys, m: Option<KGradMutation>| {
        let dk = fixed_dk(&k.mesh, nao);
        kpoint_ecp_gradient_fixed_d_with(&k.cell, &k.prep, &k.mesh, &hcore_cfg(), &dk, m)
            .unwrap()
            .unwrap()
            .grad
    };
    let strain = |k: &KSys, m: Option<KStressMutation>| {
        let dk = fixed_dk(&k.mesh, nao);
        kpoint_hcore_stress_fixed_d(
            &k.cell,
            &k.prep,
            &k.mesh,
            &hcore_cfg(),
            &k.hk,
            &dk,
            &KStressConfig {
                mutation: m,
                ..kstress_cfg()
            },
        )
        .unwrap()
        .ecp
    };
    let comb_f = |f4: &Array2<f64>, f2: &Array2<f64>| f4 * 2.0 - f2;
    let (f4, f2) = (force(&g4, None), force(&g2, None));
    let fmp = force(&mp, None);
    let fmiss = grad_miss(&fmp, &comb_f(&f4, &f2));
    let (s4, s2, smp) = (strain(&g4, None), strain(&g2, None), strain(&mp, None));
    let comb_s = |a: &Mat3, b: &Mat3| mat_sum(&mat_scaled(a, 2.0), &mat_scaled(b, -1.0));
    let smiss = max_mat(&smp, &comb_s(&s4, &s2));
    eprintln!("MP identity: force miss {fmiss:.2e}, strain miss {smiss:.2e}");
    assert!(fmiss < 1e-11 && smiss < 1e-11);
    assert!(amax(&fmp) > 1e-3 && mat_max(&smp) > 1e-3, "vacuous");
    // The mesh-count-modulus mutant is caught (identity breaks).
    let fm = force(&mp, Some(KGradMutation::EcpMeshCountModuli));
    let sm = strain(&mp, Some(KStressMutation::EcpMeshCountModuli));
    let (dfm, dsm) = (
        grad_miss(&fm, &comb_f(&f4, &f2)),
        max_mat(&sm, &comb_s(&s4, &s2)),
    );
    eprintln!("  mutant EcpMeshCountModuli: force misses by {dfm:.2e}, strain by {dsm:.2e}");
    assert!(dfm > MUTANT_BAR && dsm > MUTANT_BAR);
    // On the Gamma mesh the same mutant is an identity (n == moduli): blind.
    let f4m = force(&g4, Some(KGradMutation::EcpMeshCountModuli));
    assert!(grad_miss(&f4m, &f4) == 0.0);
}

// ============================================================ (3) SCF level

const ANCHOR_PRECISION: f64 = 1e-12;
const EXX: [ExxDiv; 2] = [ExxDiv::None, ExxDiv::Ewald];
/// Bars of the 1x1x3-vs-supercell anchors at `SC_NUCLEUS_EXPONENT` = 1e7.
/// MEASURED (HI, `hi_and_control_sr_floor_scan_over_the_nucleus_exponent`,
/// 2026-10-11) at zeta = 1e7 / 1e8 / 1e9: force total 4.1e-10 / 1.5e-8 /
/// 7.7e-8, force `vsr_nuc` 1.05e-9 / 1.5e-8 / 2.3e-7, stress total 2.6e-10 /
/// 7.4e-9 / 1.1e-7, while the ECP part is FLAT at 1.4e-15 (force) and 2.0e-15
/// (stress) at every zeta. The no-ECP Z = 7 control (same shells, same
/// nuclear charges) shows 2.9e-10 / 9.8e-9 / 4.9e-8 (force total) and
/// 2.5e-10 / 5.5e-9 / 7.9e-8 (stress): the miss is the libint2 SR-attraction-
/// derivative floor at the gradient-nucleus exponent (it grows ~10-30x per
/// decade and is present without any ECP code); the ECP-shaped density plus
/// Z_eff = 7 make it ~1.5x the control's. Every non-SR part is at <= 5.6e-13.
/// Bars: SR parts 5e-9 and totals 2e-9 (5-8x above the 1e7 measurement,
/// >= 5 orders below the smallest ECP mutant), other parts 3e-12.
const SC_BAR_P: f64 = 2e-9;
const SC_BAR_SR_PART: f64 = 5e-9;
/// Bar of every part other than the SR attraction and the totals.
const SC_BAR_PART: f64 = 3e-12;

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
/// the supercell anchors. The k mesh and the supercell evaluate that
/// derivative at different image offsets, so libint2's integral precision
/// floor does not cancel; it grows with the exponent (see `SC_BAR_P`). The
/// ECP part does not depend on it.
const SC_NUCLEUS_EXPONENT: f64 = 1e7;
/// Exponent of the FD anchors and of the default configs: against FD of the
/// energy (nucleus exponent 1e16) 1e8 is the better compromise (force FD
/// miss 2.3e-9 at 1e8 vs 2.1e-8 at 1e7, where the mismatch with the energy's
/// nucleus takes over), the opposite of the k-vs-supercell comparison.
const FD_NUCLEUS_EXPONENT: f64 = 1e8;

fn gcfg(m: Option<KGradMutation>) -> KGradConfig {
    gcfg_z(m, FD_NUCLEUS_EXPONENT)
}

fn gcfg_z(m: Option<KGradMutation>, zeta: f64) -> KGradConfig {
    KGradConfig {
        budget_bytes: Some(AMPLE),
        mutation: m,
        nucleus_exponent: Some(zeta),
    }
}

fn scfg(m: Option<KStressMutation>) -> KStressConfig {
    scfg_z(m, FD_NUCLEUS_EXPONENT)
}

fn scfg_z(m: Option<KStressMutation>, zeta: f64) -> KStressConfig {
    KStressConfig {
        budget_bytes: Some(AMPLE),
        mutation: m,
        nucleus_exponent: Some(zeta),
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

fn gsys(cell: Cell, bs: &BasisSet, aprec: f64) -> GSys {
    let prep = prep_for(&cell, bs);
    let hc = periodic_hcore(&cell, &prep, &hcore_cfg()).expect("gamma hcore");
    assert!(bs.ecps.is_empty() || hc.n_ecp_triples > 0, "no ECP triples");
    let eri = DenseAftEri::build(
        &cell,
        &prep,
        &hc.s,
        ExxDiv::None,
        aprec,
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

fn gamma_force(g: &GSys, scf: &ScfResult, exx: ExxDiv, zeta: f64) -> GammaGradient {
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
            nucleus_exponent: Some(zeta),
            ..Default::default()
        },
    )
    .expect("gamma gradient")
}

fn gamma_stress(g: &GSys, scf: &ScfResult, exx: ExxDiv, zeta: f64) -> GammaStress {
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
            nucleus_exponent: Some(zeta),
            ..Default::default()
        },
    )
    .expect("gamma stress")
}

fn mat_sum(a: &Mat3, b: &Mat3) -> Mat3 {
    let mut o = *a;
    for i in 0..3 {
        for j in 0..3 {
            o[i][j] += b[i][j];
        }
    }
    o
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
#[ignore = "slow (18 s release, loaded box): HI 1x1x1 k SCFs + Gamma hcore/dense AFT/force/stress, both exxdiv"]
fn one_point_mesh_ecp_force_and_stress_are_the_gamma_ones() {
    let bs = hi_basis();
    let ks = ksys(hi_cell(&MOVED, &bs), &bs, [1, 1, 1]);
    let eri = kdense(&ks, ANCHOR_PRECISION);
    let scfs = scf_pair(&ks, &eri);
    let g = gsys(hi_cell(&MOVED, &bs), &bs, ANCHOR_PRECISION);
    for (i, exx) in EXX.into_iter().enumerate() {
        let gscf = gamma_restricted(real_of(&scfs[i].densities[0]));
        let kf = kforce(&ks, &eri, &scfs[i], exx, None);
        let gf = gamma_force(&g, &gscf, exx, FD_NUCLEUS_EXPONENT);
        let ks_ = kstress(&ks, &eri, &scfs[i], exx, None);
        let gs = gamma_stress(&g, &gscf, exx, FD_NUCLEUS_EXPONENT);
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
struct Anchor {
    ks: KSys,
    eri: KDenseAftEri,
    exx: Vec<ExxDiv>,
    scfs: Vec<KScfResult>,
    g: GSys,
    gscfs: Vec<ScfResult>,
}

/// k-RHF SCFs on the 1x1x3 mesh and the explicit 3-cell Gamma supercell at the
/// unfolded densities, for each exxdiv in `exx`.
fn anchor_setup(bs: &BasisSet, exx: &[ExxDiv], aprec: f64) -> Anchor {
    let ks = ksys(hi_cell(&MOVED, bs), bs, [1, 1, 3]);
    let eri = kdense(&ks, aprec);
    let vm = ks.mesh.madelung(&ks.cell).unwrap();
    let scfs: Vec<KScfResult> = exx
        .iter()
        .map(|e| {
            let v = if *e == ExxDiv::None { 0.0 } else { vm };
            krhf(&ks, Box::new(eri.jk_builder_with_madelung(v)))
        })
        .collect();
    let g = gsys(hi_supercell(&MOVED, 3, bs), bs, aprec);
    let gscfs = scfs
        .iter()
        .map(|sc| gamma_restricted(unfold_z(&ks.mesh, &sc.densities)))
        .collect();
    Anchor {
        ks,
        eri,
        exx: exx.to_vec(),
        scfs,
        g,
        gscfs,
    }
}

type Parts = Vec<(&'static str, f64)>;

/// Per-part misses of the k force / stress against the supercell at
/// gradient-nucleus exponent `zeta` on both sides (SCFs reused).
fn anchor_parts(a: &Anchor, i: usize, zeta: f64) -> (Parts, Parts) {
    let exx = a.exx[i];
    let kf = kpoint_rhf_gradient(
        &a.ks.cell,
        &a.ks.prep,
        &a.ks.mesh,
        &hcore_cfg(),
        &a.ks.hk,
        KGradJk::Dense(&a.eri),
        &a.scfs[i],
        exx,
        &gcfg_z(None, zeta),
    )
    .expect("kpoint_rhf_gradient");
    let ksr = kpoint_rhf_stress(
        &a.ks.cell,
        &a.ks.prep,
        &a.ks.mesh,
        &hcore_cfg(),
        &a.ks.hk,
        KStressJk::Dense(&a.eri),
        &a.scfs[i],
        exx,
        &scfg_z(None, zeta),
    )
    .expect("kpoint_rhf_stress");
    let gf = gamma_force(&a.g, &a.gscfs[i], exx, zeta);
    let gs = gamma_stress(&a.g, &a.gscfs[i], exx, zeta);
    let (kp, gp) = (&kf.parts, &gf.parts);
    let eri_k = &kp.coulomb + &kp.exchange;
    let force: Parts = vec![
        ("overlap", sc_miss(&kp.overlap, &gp.overlap)),
        ("kinetic", sc_miss(&kp.kinetic, &gp.kinetic)),
        ("ecp", sc_miss(&kp.ecp, &gp.ecp)),
        ("vsr_basis", sc_miss(&kp.vsr_basis, &gp.vsr_basis)),
        ("vsr_nuc", sc_miss(&kp.vsr_nuc, &gp.vsr_nuc)),
        ("vlr_basis", sc_miss(&kp.vlr_basis, &gp.vlr_basis)),
        ("vlr_nuc", sc_miss(&kp.vlr_nuc, &gp.vlr_nuc)),
        ("eri", sc_miss(&eri_k, &gp.eri)),
        // The Ewald real/reciprocal split depends on the cell (the default
        // omega is cell-dependent), so only their SUM is comparable.
        (
            "nn",
            sc_miss(&(&kp.nn_sr + &kp.nn_lr), &(&gp.nn_sr + &gp.nn_lr)),
        ),
        ("TOTAL", sc_miss(&kf.grad, &gf.grad)),
    ];
    let third = |m: &Mat3| mat_scaled(m, 1.0 / 3.0);
    let (sk, sg) = (&ksr.parts, &gs.parts);
    let eri_s = {
        let mut t = sk.coulomb;
        for r in 0..3 {
            for c in 0..3 {
                t[r][c] += sk.exchange[r][c];
            }
        }
        t
    };
    let stress: Parts = vec![
        ("overlap", max_mat(&sk.overlap(), &third(&sg.overlap))),
        ("kinetic", max_mat(&sk.kinetic, &third(&sg.kinetic))),
        ("ecp", max_mat(&sk.ecp, &third(&sg.ecp))),
        ("vsr", max_mat(&sk.vsr, &third(&sg.vsr))),
        ("vlr", max_mat(&sk.vlr, &third(&sg.vlr))),
        ("c0_volume", max_mat(&sk.c0_volume, &third(&sg.c0_volume))),
        ("eri", max_mat(&eri_s, &third(&sg.eri))),
        ("madelung", max_mat(&sk.madelung, &third(&sg.madelung))),
        (
            "nn",
            max_mat(
                &mat_sum(&mat_sum(&sk.nn_sr, &sk.nn_lr), &sk.nn_background),
                &third(&mat_sum(&mat_sum(&sg.nn_sr, &sg.nn_lr), &sg.nn_background)),
            ),
        ),
        ("TOTAL", max_mat(&ksr.de_deps, &third(&gs.de_deps))),
    ];
    (force, stress)
}

/// Asserts the supercell anchor at one `zeta`: every part except the SR
/// attraction (`vsr*`) and the totals within `SC_BAR_PART`, the SR parts and
/// totals within `sr_bar`. Returns the parts for the scan table.
fn assert_anchor(a: &Anchor, i: usize, zeta: f64, sr_bar: f64) -> (Parts, Parts) {
    let (f, s) = anchor_parts(a, i, zeta);
    eprintln!(
        "  {:?} zeta {zeta:.0e}: force parts {}",
        a.exx[i],
        fmt_parts(&f)
    );
    eprintln!(
        "  {:?} zeta {zeta:.0e}: stress parts {}",
        a.exx[i],
        fmt_parts(&s)
    );
    for (kind, parts) in [("force", &f), ("stress", &s)] {
        for (name, miss) in parts {
            let bar = if *name == "TOTAL" {
                sr_bar
            } else if name.starts_with("vsr") {
                sr_bar.max(SC_BAR_SR_PART)
            } else {
                SC_BAR_PART
            };
            assert!(
                *miss < bar,
                "{kind} part {name} misses the supercell by {miss:e} (bar {bar:e}, zeta {zeta:e})"
            );
        }
    }
    (f, s)
}

fn fmt_parts(p: &Parts) -> String {
    p.iter()
        .map(|(n, v)| format!("{n} {v:.2e}"))
        .collect::<Vec<_>>()
        .join(", ")
}

fn part_of(p: &Parts, name: &str) -> f64 {
    p.iter().find(|x| x.0 == name).expect("part").1
}

/// PHYSICS: the 1x1x3 Gamma-centred mesh IS the diag(1,1,3) supercell, so per
/// cell the force on atom A equals the supercell force on any copy of A and
/// `dE/dε = dE/dε_sc / 3`, part by part (force: overlap, kinetic, ECP, SR
/// basis/nucleus, LR, ERI, Ewald real/reciprocal; stress: every part).
/// ARTIFACT: a wrong phase / conjugate / residue would not reproduce the
/// explicit supercell (non-TRIM mesh). The SR attraction derivative carries
/// libint2's gradient-nucleus-exponent floor (see
/// `hi_and_control_sr_floor_scan_over_the_nucleus_exponent`); everything else
/// is at roundoff.
#[test]
fn hi_1x1x3_force_and_stress_are_the_supercell_ones_per_cell() {
    let bs = hi_basis();
    let t0 = std::time::Instant::now();
    let a = anchor_setup(&bs, &[ExxDiv::None], ANCHOR_PRECISION);
    eprintln!("supercell setup {:.1} s", t0.elapsed().as_secs_f64());
    let (f, s) = assert_anchor(&a, 0, SC_NUCLEUS_EXPONENT, SC_BAR_P);
    assert!(part_of(&f, "ecp") < SC_BAR_PART && part_of(&s, "ecp") < SC_BAR_PART);
}

/// Both exxdiv at the same zeta (Ewald exercises the Madelung terms).
#[test]
#[ignore = "slow (114 s release, loaded box): HI 1x1x3 k SCFs (none, ewald) + the 3-cell Gamma supercell force/stress"]
fn hi_1x1x3_supercell_anchor_both_exxdiv() {
    let bs = hi_basis();
    let a = anchor_setup(&bs, &EXX, ANCHOR_PRECISION);
    for i in 0..2 {
        assert_anchor(&a, i, SC_NUCLEUS_EXPONENT, SC_BAR_P);
    }
}

/// DIAGNOSTIC + ASSERTION: where does the 1.5e-8 SR-part miss come from?
/// Hypotheses: (ECP) it is ECP related; (LIBINT) it is libint2's gradient-
/// nucleus-exponent floor of the SR attraction derivative, in which case the
/// `vsr_*` parts show a U shape in zeta (rising ~10-30x per decade above
/// ~1e9, rising again at 1e7 from the mismatch with the energy's nucleus)
/// while the ECP part stays flat at roundoff. The Z = 7 no-ECP control
/// (identical AO shells and nuclear charges) separates "ECP-shaped density"
/// from "Z_eff = 7": the control floor is printed next to HI's.
#[test]
#[ignore = "slow (261 s release, loaded box): HI and control 1x1x3 supercell anchors at zeta 1e7, 1e8, 1e9"]
fn hi_and_control_sr_floor_scan_over_the_nucleus_exponent() {
    for (name, bs) in [("HI+ECP", hi_basis()), ("control Z=7", control_basis())] {
        let a = anchor_setup(&bs, &[ExxDiv::None], ANCHOR_PRECISION);
        for zeta in [1e7, 1e8, 1e9] {
            let (f, s) = assert_anchor(&a, 0, zeta, 1e-6);
            assert!(part_of(&f, "ecp") < 1e-12 && part_of(&s, "ecp") < 1e-12);
            eprintln!(
                "SCAN {name} zeta {zeta:.0e}: force vsr_basis {:.2e} vsr_nuc {:.2e} ecp {:.2e} \
                 total {:.2e}; stress vsr {:.2e} ecp {:.2e} total {:.2e}",
                part_of(&f, "vsr_basis"),
                part_of(&f, "vsr_nuc"),
                part_of(&f, "ecp"),
                part_of(&f, "TOTAL"),
                part_of(&s, "vsr"),
                part_of(&s, "ecp"),
                part_of(&s, "TOTAL"),
            );
        }
    }
}

// ============================================================ mutants

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
#[ignore = "slow (189 s release, loaded box): HI 1x1x3 k SCF + 9 force and 8 stress mutant evaluations"]
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
/// `V_ECP`), force AND stress. BOTH SIDES of the bar (1e-8): the matching SCF
/// reproduces the stored Fock to 5.7e-17 and is accepted; an SCF run WITHOUT
/// the ECP (a stored Fock lacking `V_ECP`) differs by 6.9 and is refused by
/// both. Two `hk` inconsistencies are refused by both entry points: the
/// `v_ecp` marker cleared while `h` carries the ECP, and `h` stripped of the
/// ECP while the marker is kept (the SCF run on that `h` is self-consistent,
/// so only the `h = T + V + V_ECP` check sees it).
#[test]
fn ecp_fock_guard_refuses_an_scf_run_without_the_ecp() {
    let bs = hi_basis();
    let ks = ksys(hi_cell(&MOVED, &bs), &bs, [1, 1, 3]);
    let eri = kdense(&ks, 1e-10);
    let good = krhf(&ks, Box::new(eri.jk_builder_with_madelung(0.0)));
    let s = kstress(&ks, &eri, &good, ExxDiv::None, None);
    let f = kforce(&ks, &eri, &good, ExxDiv::None, None);
    eprintln!(
        "matching SCF: Fock mismatch stress {:.2e} force {:.2e}, force commutator {:.2e}",
        s.fock_mismatch, f.fock_mismatch, f.commutator
    );
    assert!(s.fock_mismatch < 1e-10 && f.fock_mismatch < 1e-10 && f.commutator < 1e-7);
    let force_err = |hk: &PeriodicHcoreK, scf: &KScfResult| {
        kpoint_rhf_gradient(
            &ks.cell,
            &ks.prep,
            &ks.mesh,
            &hcore_cfg(),
            hk,
            KGradJk::Dense(&eri),
            scf,
            ExxDiv::None,
            &gcfg(None),
        )
        .unwrap_err()
        .to_string()
    };
    let stress_err = |hk: &PeriodicHcoreK, scf: &KScfResult| {
        kpoint_rhf_stress(
            &ks.cell,
            &ks.prep,
            &ks.mesh,
            &hcore_cfg(),
            hk,
            KStressJk::Dense(&eri),
            scf,
            ExxDiv::None,
            &scfg(None),
        )
        .unwrap_err()
        .to_string()
    };
    // SCF without the ECP.
    let hk0 = strip_ecp(&ks.hk);
    let bad = krhf_with(&ks, &hk0, Box::new(eri.jk_builder_with_madelung(0.0)), None);
    let e = stress_err(&ks.hk, &bad);
    eprintln!("ECP-free SCF, stress: {e}");
    assert!(e.contains("differs from the SCF's stored"), "{e}");
    let e = force_err(&ks.hk, &bad);
    eprintln!("ECP-free SCF, force: {e}");
    assert!(e.contains("beyond a Madelung (exxdiv) shift"), "{e}");
    // hk with the v_ecp marker cleared (h still carries the ECP).
    let mut hk_lost = ks.hk.clone();
    hk_lost.v_ecp = None;
    for e in [stress_err(&hk_lost, &good), force_err(&hk_lost, &good)] {
        assert!(e.contains("carries an ECP for a cell atom"), "{e}");
    }
    // h stripped of the ECP, marker kept; the SCF on that h is self-consistent.
    let mut hk_h = ks.hk.clone();
    for (h, v) in hk_h.h.iter_mut().zip(ks.hk.v_ecp.as_ref().unwrap()) {
        *h -= v;
    }
    let scf_h = krhf_with(
        &ks,
        &hk_h,
        Box::new(eri.jk_builder_with_madelung(0.0)),
        None,
    );
    for e in [stress_err(&hk_h, &scf_h), force_err(&hk_h, &scf_h)] {
        eprintln!("stripped h: {e}");
        assert!(e.contains("differs from T + V + V_ECP"), "{e}");
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
}

/// The frozen serial image loop (`SerialDerivWalks`) is the bit-identity
/// oracle of the ordered-parallel ECP force, through the production entry.
#[test]
#[ignore = "slow (16 s release, loaded box): HI 1x1x3 k-SCF + two force evaluations"]
fn ecp_k_force_serial_image_loop_is_the_parallel_one_bitwise() {
    let bs = hi_basis();
    let ks = ksys(hi_cell(&MOVED, &bs), &bs, [1, 1, 3]);
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

/// Bars of the SCF-level FD anchors (central FD, h = 1e-4). The floor of
/// these is the h^2 truncation of the central difference (FD(h) vs FD(2h)
/// differ at that order, as in the Gamma and k-stress anchors) plus, for the
/// force, the Gaussian-nucleus floor of the SR derivative at `zeta = 1e8`;
/// the ECP value kernel itself is exact to ~1e-13 (the 2.5e-7 of the Gamma
/// force anchor at zeta = 1e9 is V_SR, not V_ECP). MEASURED here: force
/// 2.3e-9 / 1.1e-9 (dense), 1.1e-9 / 2.3e-9 (RS-GDF); stress 3.1e-8 / 5.4e-8
/// (dense, all nine), 2.3e-8 / 4.2e-9 / 2.6e-8 (RS-GDF). The smallest ECP
/// mutant misses the same anchors by >= 2e-3 (force) and >= 5e-2 (stress).
const FORCE_FD_BAR: f64 = 2e-8;
const STRESS_FD_BAR: f64 = 2e-7;
const FD_H: f64 = 1e-4;

/// k-RHF energy per cell at exxdiv none of a (displaced) cell.
fn k_energy_none(cell: &Cell, bs: &BasisSet, n: [usize; 3]) -> f64 {
    let ks = ksys(cell.clone(), bs, n);
    let eri = kdense(&ks, ANCHOR_PRECISION);
    krhf(&ks, Box::new(eri.jk_builder_with_madelung(0.0))).energy
}

/// PHYSICS: analytic k force vs central FD (h = 1e-4) of ferric's own k
/// energy (exxdiv none; the force is exxdiv-independent and the Ewald energy
/// differs by a constant per cell, both exxdiv are in the ignored stress FD),
/// a fresh SCF at every displaced geometry (the analytic value is at the
/// converged density, so the density response drops out). Components: H z
/// (no ECP centre of its own) and I x (bra + ket + centre).
#[test]
fn hi_1x1x3_force_matches_fd_of_own_k_energy() {
    let bs = hi_basis();
    let n = [1, 1, 3];
    let ks = ksys(hi_cell(&MOVED, &bs), &bs, n);
    let eri = kdense(&ks, ANCHOR_PRECISION);
    let scf = krhf(&ks, Box::new(eri.jk_builder_with_madelung(0.0)));
    let f = kforce(&ks, &eri, &scf, ExxDiv::None, None);
    let net = (0..3)
        .map(|c| f.grad.column(c).sum().abs())
        .fold(0.0_f64, f64::max);
    for (a, x) in [(0usize, 2usize), (1, 0)] {
        let ep = k_energy_none(&hi_cell(&moved(&MOVED, a, x, FD_H), &bs), &bs, n);
        let em = k_energy_none(&hi_cell(&moved(&MOVED, a, x, -FD_H), &bs), &bs, n);
        let fd = (ep - em) / (2.0 * FD_H);
        let miss = (f.grad[(a, x)] - fd).abs();
        eprintln!(
            "  HI 1x1x3 atom {a} x{x}: analytic {:.10e} FD {fd:.10e} miss {miss:.2e}; ECP part \
             {:.3e}",
            f.grad[(a, x)],
            f.parts.ecp[(a, x)]
        );
        assert!(miss < FORCE_FD_BAR, "({a},{x}) FD miss {miss:e}");
    }
    eprintln!("  ΣF {net:.2e}");
    assert!(net < 1e-7, "ΣF {net:e}");
}

/// The stress analogue: all nine components of `dE/dε` vs central FD (h =
/// 1e-4) of the k energy on `Cell::strained` cells, mesh rebuilt from the same
/// integer numerators, fresh SCF per point, both exxdiv.
#[test]
#[ignore = "slow (55 s release, loaded box): 18 strained HI 1x1x3 k hcore + dense AFT builds + 36 k-RHF SCFs"]
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
        assert!(miss < STRESS_FD_BAR, "{exx:?}: FD miss {miss:e}");
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
#[ignore = "slow (33 s release, loaded box): HI 1x1x3 k-RHF and k-UHF SCFs, 2 forces, 2 stresses"]
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
#[ignore = "slow (37 s release, loaded box): H2I doublet 1x1x3 k-UHF + force + stress, supercell ECP terms at fixed density"]
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

/// H2I doublet cell (5α/4β per cell).
fn h2i_cell(pos: &[[f64; 3]], bs: &BasisSet) -> Cell {
    ecp_cell(
        &[("H", 1), ("H", 1), ("I", 53)],
        pos,
        &[[0.0; 3]],
        HI_A,
        bs,
        2,
    )
}

fn h2i_energy(cell: &Cell, bs: &BasisSet) -> f64 {
    let ks = ksys(cell.clone(), bs, [1, 1, 3]);
    let eri = kdense(&ks, ANCHOR_PRECISION);
    kuhf_with(&ks, &eri, 5, 4).energy
}

/// SCF-level spin-polarised anchor with an ECP: the analytic k-UHF force on
/// I x and the zz strain of the H2I doublet vs central FD (h = 1e-4) of the
/// k-UHF energy, a fresh UHF SCF at every displaced / strained geometry.
/// (The closed-shell identity and the supercell ECP-part test above cannot
/// see a spin slip in the Pulay weight `W = Σ_σ D_σ F_σ D_σ`; this does.)
#[test]
#[ignore = "slow (65 s release, loaded box): H2I doublet 1x1x3, 1 UHF SCF + 4 displaced/strained UHF SCFs"]
fn h2i_doublet_uhf_force_and_stress_match_fd_of_own_k_energy() {
    let bs = hi_basis();
    let pos = [[0.3, 0.2, 0.4], [2.0, 0.6, 1.0], [0.9, -0.4, 3.3]];
    let cell = h2i_cell(&pos, &bs);
    let ks = ksys(cell.clone(), &bs, [1, 1, 3]);
    let eri = kdense(&ks, ANCHOR_PRECISION);
    let u = kuhf_with(&ks, &eri, 5, 4);
    let f = kforce_u(&ks, &eri, &u);
    let s = kstress_u(&ks, &eri, &u);
    let step = |a: usize, x: usize, h: f64| {
        let mut p = pos;
        p[a][x] += h;
        p
    };
    let fd_f = (h2i_energy(&h2i_cell(&step(2, 0, FD_H), &bs), &bs)
        - h2i_energy(&h2i_cell(&step(2, 0, -FD_H), &bs), &bs))
        / (2.0 * FD_H);
    let fd_s = (h2i_energy(&cell.strained(&eps_at(2, 2, FD_H)).unwrap(), &bs)
        - h2i_energy(&cell.strained(&eps_at(2, 2, -FD_H)).unwrap(), &bs))
        / (2.0 * FD_H);
    let (mf, ms) = (
        (f.grad[(2, 0)] - fd_f).abs(),
        (s.de_deps[2][2] - fd_s).abs(),
    );
    eprintln!(
        "H2I k-UHF: force (I,x) analytic {:.10e} FD {fd_f:.10e} miss {mf:.2e}; strain zz \
         analytic {:.10e} FD {fd_s:.10e} miss {ms:.2e}",
        f.grad[(2, 0)],
        s.de_deps[2][2]
    );
    assert!(mf < FORCE_FD_BAR && ms < STRESS_FD_BAR);
}

// ============================================================ RS-GDF

/// Even-tempered s/p aux on H and I (one unit-normalised primitive per
/// shell). The fit quality is irrelevant here: forces and stress are the
/// exact derivatives of the FITTED energy, whatever the aux.
fn hi_aux() -> BasisSet {
    let sh = |l: i32, a: f64| Shell {
        l,
        pure: false,
        exponents: vec![a],
        coefficients: vec![1.0],
    };
    let mut m = HashMap::new();
    m.insert(
        1,
        vec![
            sh(0, 4.7),
            sh(0, 1.9),
            sh(0, 0.75),
            sh(0, 0.3),
            sh(1, 1.25),
            sh(1, 0.5),
        ],
    );
    m.insert(
        53,
        vec![
            sh(0, 2.0),
            sh(0, 0.8),
            sh(0, 0.3),
            sh(1, 1.2),
            sh(1, 0.5),
            sh(1, 0.2),
        ],
    );
    BasisSet {
        name: "hi-aux".into(),
        shells: m,
        ecps: HashMap::new(),
    }
}

fn krgdf_cfg() -> KRsGdfConfig {
    KRsGdfConfig {
        gdf: RsGdfConfig {
            omega: 1.0,
            exxdiv: ExxDiv::None,
            budget_bytes: Some(AMPLE),
            sr_column_rotation: SrColumnRotation::Off,
            ..Default::default()
        },
        mutation: None,
    }
}

struct FitSys {
    cfg: KRsGdfConfig,
    aux: PreparedBasis,
    gdf: KRsGdf,
}

fn fitsys(ks: &KSys, bs_aux: &BasisSet) -> FitSys {
    let cfg = krgdf_cfg();
    let aux = PreparedBasis::new(ks.cell.mol(), bs_aux).expect("aux prep");
    let gdf = KRsGdf::build(&ks.cell, &ks.prep, &aux, &ks.mesh, &ks.hk.s, &cfg).expect("KRsGdf");
    FitSys { cfg, aux, gdf }
}

impl FitSys {
    fn src(&self) -> KRsGdfGradSource<'_> {
        KRsGdfGradSource {
            gdf: &self.gdf,
            cfg: &self.cfg,
            aux: &self.aux,
        }
    }
}

fn fit_scf(ks: &KSys, fs: &FitSys) -> KScfResult {
    krhf(ks, Box::new(fs.gdf.jk_builder_with_madelung(0.0)))
}

fn k_fit_energy(cell: &Cell, bs: &BasisSet, n: [usize; 3]) -> f64 {
    let ks = ksys(cell.clone(), bs, n);
    let fs = fitsys(&ks, &hi_aux());
    fit_scf(&ks, &fs).energy
}

/// RS-GDF (unsplit) k force and stress with an ECP. PHYSICS: the ECP part is
/// the same lattice walk as with the dense kernels; here the density comes
/// from the FITTED SCF. ANCHORS: (a) the ECP part equals the Gamma
/// supercell ECP term at the unfolded density (1e-15 level); (b) the
/// total force (I x) and the zz stress component match central FD of
/// the k energy built on the same fit (fresh `KRsGdf` per displaced / strained
/// cell). ARTIFACT: a fit-specific slip (e.g. dropping the ECP from the
/// rebuilt `F` of the Pulay weight) is invisible to (a) and caught by (b).
#[test]
fn hi_1x1x3_rsgdf_ecp_force_and_stress() {
    let bs = hi_basis();
    let n = [1, 1, 3];
    let cell = hi_cell(&MOVED, &bs);
    let ks = ksys(cell.clone(), &bs, n);
    let t0 = std::time::Instant::now();
    let fs = fitsys(&ks, &hi_aux());
    let scf = fit_scf(&ks, &fs);
    eprintln!("RS-GDF build + SCF {:.1} s", t0.elapsed().as_secs_f64());
    let f = kpoint_rhf_gradient(
        &ks.cell,
        &ks.prep,
        &ks.mesh,
        &hcore_cfg(),
        &ks.hk,
        KGradJk::RsGdf(fs.src()),
        &scf,
        ExxDiv::None,
        &gcfg(None),
    )
    .expect("fit force");
    let s = kpoint_rhf_stress(
        &ks.cell,
        &ks.prep,
        &ks.mesh,
        &hcore_cfg(),
        &ks.hk,
        KStressJk::RsGdf(fs.src()),
        &scf,
        ExxDiv::None,
        &scfg(None),
    )
    .expect("fit stress");
    // (a) supercell ECP part at fixed density.
    let sc = hi_supercell(&MOVED, 3, &bs);
    let psc = prep_for(&sc, &bs);
    let dsc = unfold_z(&ks.mesh, &scf.densities);
    let g = periodic_ecp_gradient(&sc, &psc, &ecp_cfg(), &dsc)
        .unwrap()
        .unwrap();
    let (ss, _) = periodic_ecp_strain(&sc, &psc, &ecp_cfg(), &dsc)
        .unwrap()
        .unwrap();
    let fm = grad_miss(
        &f.parts.ecp,
        &g.grad.slice(ndarray::s![0..2, ..]).to_owned(),
    );
    let sm = max_mat(&s.parts.ecp, &mat_scaled(&ss, 1.0 / 3.0));
    eprintln!("RS-GDF ECP parts vs supercell: force {fm:.2e}, strain {sm:.2e}");
    assert!(fm < SC_BAR_PART && sm < SC_BAR_PART);
    // (b) FD of the fitted k energy.
    {
        let (a, x) = (1usize, 0usize);
        let ep = k_fit_energy(&hi_cell(&moved(&MOVED, a, x, FD_H), &bs), &bs, n);
        let em = k_fit_energy(&hi_cell(&moved(&MOVED, a, x, -FD_H), &bs), &bs, n);
        let fd = (ep - em) / (2.0 * FD_H);
        let miss = (f.grad[(a, x)] - fd).abs();
        eprintln!(
            "  force ({a},{x}): analytic {:.10e} FD {fd:.10e} miss {miss:.2e}",
            f.grad[(a, x)]
        );
        assert!(miss < FORCE_FD_BAR, "force FD miss {miss:e}");
    }
    {
        let (i, j) = (2usize, 2usize);
        let ep = k_fit_energy(&cell.strained(&eps_at(i, j, FD_H)).unwrap(), &bs, n);
        let em = k_fit_energy(&cell.strained(&eps_at(i, j, -FD_H)).unwrap(), &bs, n);
        let fd = (ep - em) / (2.0 * FD_H);
        let miss = (s.de_deps[i][j] - fd).abs();
        eprintln!(
            "  strain ({i},{j}): analytic {:.10e} FD {fd:.10e} miss {miss:.2e}",
            s.de_deps[i][j]
        );
        assert!(miss < STRESS_FD_BAR, "strain FD miss {miss:e}");
    }
}

/// A SHIFTED (Monkhorst-Pack, even) mesh: a twisted supercell with no
/// explicit-supercell anchor, so only the FD anchors it. The residue moduli
/// differ from the mesh counts here (the pair-image modulus is twice the
/// mesh's), so this exercises the residue bins of `W_L` that a Gamma-centred
/// mesh cannot. Fixed Hermitian `D(k)`; force (I x) and the zz / xz strain.
#[test]
#[ignore = "slow (6 s release, loaded box): HI MP [1,1,2] fixed-density ECP force + strain FD (10 ECP lattice sums, Richardson)"]
fn ecp_terms_on_a_shifted_mesh_match_fd_at_fixed_density() {
    let bs = hi_basis();
    let n = [1, 1, 2];
    let cell = hi_cell(&MOVED, &bs);
    let ks = ksys_c(cell.clone(), &bs, n, MeshCentring::MonkhorstPack);
    let moduli = ks.mesh.residue_moduli();
    eprintln!("MP mesh {n:?}: residue moduli {moduli:?}");
    let dk = fixed_dk(&ks.mesh, ks.prep.nbasis());
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
    let e = |c: &Cell| e_ecp_k_c(c, &bs, n, MeshCentring::MonkhorstPack, &dk);
    let fd_f = |h: f64| {
        (e(&hi_cell(&moved(&MOVED, 1, 0, h), &bs)) - e(&hi_cell(&moved(&MOVED, 1, 0, -h), &bs)))
            / (2.0 * h)
    };
    let rich = richardson(fd_f(1e-3), fd_f(5e-4));
    let miss_f = (g.grad[(1, 0)] - rich).abs();
    let mut miss_s = 0.0_f64;
    for (i, j) in [(2usize, 2usize), (0, 2)] {
        let fd = |h: f64| {
            let cp = cell.strained(&eps_at(i, j, h)).unwrap();
            let cm = cell.strained(&eps_at(i, j, -h)).unwrap();
            (e(&cp) - e(&cm)) / (2.0 * h)
        };
        miss_s = miss_s.max((s.ecp[i][j] - richardson(fd(1e-3), fd(5e-4))).abs());
    }
    eprintln!("MP fixed-D ECP: force miss {miss_f:.2e}, strain miss {miss_s:.2e}");
    assert!(miss_f < FIXED_D_FORCE_BAR && miss_s < FIXED_D_STRAIN_BAR);
}
