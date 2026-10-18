// LANL2DZ exponents such as 0.318 are basis data, not approximations of 1/pi.
#![allow(clippy::approx_constant)]
//! k-point forces and stress with a periodic ECP TOGETHER WITH a rotating
//! orbital basis (issue #421; the combination of `pbc_kpoint_ecp.rs`, #386,
//! and `pbc_sr_rotation_kderiv.rs`, #396).
//!
//! System: HI in diag(6, 6, 7), H cc-pVDZ (one rotating s column per H) and an
//! I s/p set written as generally contracted columns over one exponent list
//! with a single-primitive column (so the SR walks rotate columns ON THE ECP
//! CENTRE too), LANL2DZ ECP on I. 8 valence electrons.
//!
//! ARTIFACT hypothesis (written before measuring). The rotation acts only on
//! the SR attraction and the SR 3-centre sums; the ECP images stay on the
//! parent basis with the unrotated image weights `W_L`. A defect in the
//! combination would be (i) the ECP weights forward-transformed with `T`
//! (ECP part moves by O(|T_ks|) = O(0.1..1) on the rotated columns), (ii) the
//! ECP part dropped or double counted when the hcore walk is rotated (O(1)),
//! (iii) an ECP Fock/h consistency break when `h` is assembled from the
//! rotated `V_SR` (refused by the Fock guard, bar 1e-8), (iv) a column count
//! mismatch between the energy build and the derivative walk on the ECP atom
//! (counters). If the combination is correct: the ECP parts are BITWISE the
//! unrotated ones (same code, same weights), the totals agree with the
//! unrotated pipeline to the screen, and all anchors of the unrotated ECP
//! tests hold unchanged. Independent constructions: FD of the rotated energy
//! (does not use the derivative walks at all) and supercell/N (explicit
//! supercell with rotation on both sides); the unrotated pipeline is an
//! independent walk of the same function.
//!
//! HISTORY (fixed in #437, issue #433): while this fixture was built, the
//! erfc 3-centre block of a CONTRACTED s pair at a far-field shift read
//! uninitialised memory when the engine precision is <= 1e-20 (the hcore SR
//! walk uses `f64::MIN_POSITIVE`): libint2's Boys Chebyshev table was read one
//! row past its end at x == 117.0. A derivative engine created and destroyed
//! earlier in the process moved k-RHF energies of HI with an I s contraction
//! (1.2, 0.45, 0.16) by 1.2e-5 Ha, and any FD anchor computed after an
//! analytic call by up to 2e-8 per energy; the UNROTATED `KRsGdf` fit for this
//! basis hit the same read (perturb 1: LAPACK failure; 170: "k-RHF not
//! converged (400 it)"). The fixtures were therefore pinned to I s exponents
//! (3.0, 1.0, 0.35) / p (2.0, 0.7), which are kept (see `rot_basis`).
//! Reproducer, now a normal 0.03 s test:
//! `far_field_contracted_s_blocks_do_not_read_uninitialised_memory` (section
//! 5), which re-executes a finiteness check under `MALLOC_PERTURB_` 1, 170 and
//! 2. (glibc fills fresh memory with `n ^ 0xff`, so 255 fills with 0x00 and is
//! the default state: it tests nothing extra, and neither do 0, 85 or 254.)
//! Verified on the fixed build (59114c8e + #437): the RS-GDF anchor
//! (including its unrotated comparison run), the fixed-density hcore FD anchor
//! and the 1x1x1-vs-Gamma anchor were each run under `MALLOC_PERTURB_` unset,
//! 1, 170 and 255 with identical printed residuals (255 equals unset by
//! construction). The other slow anchors in this file (1x1x3 supercell, SCF
//! FD force/stress, refusal guard) were NOT rerun after #437. The 2.7.2
//! override hunk of #437 was never compiled by the independent reviewer.
//!
//! Triclinic ECP cell (section 4, lattice rows (6,0,0), (1,6,0), (1.2,-0.8,7)):
//! 1x1x1 vs Gamma, Gamma-centred 1x1x2 vs the 2-cell supercell, and FD of the
//! rotated k energy at 1x1x1. A shifted (Monkhorst-Pack) mesh with rotation is
//! covered in `pbc_sr_rotation_kderiv.rs` (without an ECP).
//!
//! Section 4 measured errors and mutant ledger (release, quiet box):
//! 1x1x1 vs Gamma: ECP parts 0 (bitwise; EXPECTED, not evidence: both sides
//! call the same ECP strain kernel at one point, so the anchor sees the k
//! wrapper only; the independent kernel check on the skew cell is the FD
//! test), force total 9.8e-13, stress total
//! 1.0e-12, antisymmetry 2e-14, |S| 7.6e-1, off-diagonal 2.5e-1 (ECP part
//! 7.3e-2); stress mutants NoEcp 4.6e-1, EcpNoCentre 9.8e-1, EcpCentreSign
//! 1.95, EcpNoKetImage 1.19, EcpUnitPairVector 3.5e-1 (bar 1e-4, ECP 1e-12).
//! A transposed stress is NOT a mutant: the stress is symmetric (miss
//! 5.0e-12). 1x1x2 vs supercell/N: force total 4.7e-13, stress total 2.3e-12,
//! worst part force eri 4.55e-13 (bar 5e-12, 11x), stress vlr 5.9e-14; the
//! mesh is TRIM, so it checks only the k wrapper (the supercell shares
//! `ecp_strain_image`). Test mutation (a HAND EDIT of `tri_shifts`: translate
//! along z instead of a3) fails at force kinetic 1.1e-2. FD at 1x1x1: force 7.5e-9,
//! stress 3.3e-8 (zz, xy, yz, xz); RotDropT 1.96 and EcpUnitPairVector 3.5e-1.
//!
//! Skew sensitivity (MEASURED, same MOVED positions, 1x1x1): the orthorhombic
//! cell already has off-diagonal stress 3.5e-1 (ECP part 1.95e-1; |S| 8.2e-1)
//! against 2.5e-1 (7.3e-2; 7.6e-1) triclinic, so TRI_OFFDIAG_FLOOR is a
//! vacuity guard, not skew evidence. Off-diagonal miss of ECP stress mutants
//! (orthorhombic / triclinic): EcpNoKetImage 3.6e-2 / 2.3e-1, EcpCentreSign
//! 1.44 / 6.3e-1, EcpUnitPairVector 9.0e-1 / 3.5e-1. Every mutant is visible
//! on both cells, so NO skew-specific mutant exists and detection of a
//! skew-only ECP defect is NOT shown; the only skew-sensitive check is the
//! supercell translation (hand-edit mutation above).
//!
//! Covered in `validation_kstress_followups.rs` (section C, this cell and
//! basis, RS-GDF): the range-split source with an ECP (split vs unsplit, FD
//! of the split energy, split mutants), the fit-acting FORCE mutants
//! (`RotFitWrongResidue`, `RotSwapPair`, `FitAuxPhase`, ...) against an
//! FD-anchored fitted force, and a shifted (MP 1x1x2) mesh with the ECP,
//! where `EcpWrongResidue` and `EcpMeshCountModuli` act.
//!
//! NOT tested: several ECP atoms, a NON-TRIM mesh (1x1x3) or an FD of the
//! 1x1x2 mesh on the triclinic ECP cell (the orthorhombic 1x1x3 supercell
//! anchor alone is 24 min; the triclinic 1x1x2 one is 10 min), a shifted or
//! range-split ECP anchor on the dense source or on a triclinic cell,
//! skew-SPECIFIC detection (no mutant is shown to be visible only on a
//! skewed cell; see the comparison test), k-specific ECP mutants on the
//! triclinic cell (`EcpWrongResidue` is a no-op on a 1x1x2 mesh because
//! -L mod 2 == L mod 2, so it is unrunnable there; `EcpMeshCountModuli` is a
//! no-op on every Gamma-centred mesh). The dense force loop below only
//! asserts that the fit-acting force mutants are no-ops; the RS-GDF loop
//! here is stress-only (6.97e-1 and 1.12e-1).

mod common;

use common::prep_for;
use ferric_core::basis::{self, BasisSet, Shell};
use ferric_core::ecp::{EcpDef, EcpShell, EcpTerm};
use ferric_core::mol::{Atom, Molecule};
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_pbc::dense_aft::{DenseAftEri, ExxDiv, DEFAULT_DENSE_AFT_MAX_BYTES};
use ferric_pbc::grad::GRAD_NUCLEUS_EXPONENT;
use ferric_pbc::grad::{gamma_rhf_gradient_with, GammaGradConfig};
use ferric_pbc::hcore::kpoint::{periodic_hcore_kpts, PeriodicHcoreK};
use ferric_pbc::hcore::{periodic_hcore, PeriodicHcoreConfig};
use ferric_pbc::kdense_aft::{KDenseAftConfig, KDenseAftEri};
use ferric_pbc::kgrad::{
    kpoint_ecp_gradient_fixed_d, kpoint_rhf_gradient, KGradConfig, KGradJk, KGradMutation,
    KGradient, KRsGdfGradSource,
};
use ferric_pbc::kpts::KPointMesh;
use ferric_pbc::kscf::{solve_krhf_injected, KPointInjection, KPointJk, KScfConfig, KScfResult};
use ferric_pbc::kstress::{
    kpoint_hcore_stress_fixed_d, kpoint_rhf_stress, KStress, KStressConfig, KStressJk,
    KStressMutation,
};
use ferric_pbc::lattice::Cell;
use ferric_pbc::rsgdf::kpoint::{KRsGdf, KRsGdfConfig};
use ferric_pbc::rsgdf::RsGdfConfig;
use ferric_pbc::sr_rotation::SrColumnRotation;
use ferric_pbc::stress::{gamma_rhf_stress, GammaStressConfig, Mat3};
use ferric_scf::result::{ScfExit, ScfResult, Spin};
use ndarray::Array2;
use num_complex::Complex64;
use std::collections::HashMap;

const OMEGA: f64 = 0.8;
const HCORE_PRECISION: f64 = 1e-14;
const ANCHOR_PRECISION: f64 = 1e-12;
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
const FD_H: f64 = 1e-4;
// Every anchor here runs the SR derivative at the library default
// `GRAD_NUCLEUS_EXPONENT` (the energy's own 1e16 nucleus), so the derivative
// carries no model-nucleus error against the energy.

// ------------------------------------------------------------ fixtures

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

/// H cc-pVDZ (s column over 4 primitives + the single-primitive 0.122 column,
/// so one rotating column per H) and an I s/p set with the same structure
/// (s: 3 primitives, columns {contraction, last primitive}; p: 2 primitives,
/// likewise), LANL2DZ ECP on I.
fn rot_basis() -> BasisSet {
    let mut shells = HashMap::new();
    let h = basis::bundled("cc-pvdz").expect("cc-pvdz");
    shells.insert(1, h.shells.get(&1).expect("H cc-pvdz").clone());
    // Exponents pinned while the libint2 Boys table-edge defect (#433, fixed
    // in #437; see the module doc) was open; kept to leave the anchors'
    // numbers unchanged. Measured after #437: the original diffuse s set
    // (1.2, 0.45, 0.16) also passes the fixed-D hcore, 1x1x1 and RS-GDF
    // anchors under MALLOC_PERTURB_ unset/1/170 (and 255, = unset) with the same bars, but
    // costs ~30% more wall time and was not run through the two multi-hour
    // anchors, so the pinned set stays.
    let s_exps = [3.0, 1.0, 0.35];
    let p_exps = [2.0, 0.7];
    shells.insert(
        53,
        vec![
            shell(0, &s_exps, &[0.3, 0.5, 0.4]),
            shell(0, &s_exps, &[0.0, 0.0, 1.0]),
            shell(1, &p_exps, &[0.4, 0.8]),
            shell(1, &p_exps, &[0.0, 1.0]),
        ],
    );
    let mut ecps = HashMap::new();
    ecps.insert(53, lanl2dz_i_ecp());
    BasisSet {
        name: "HI-rot".into(),
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

fn ecp_cell(pos: &[[f64; 3]], shifts: &[[f64; 3]], lattice: [[f64; 3]; 3], bs: &BasisSet) -> Cell {
    let species = [("H", 1), ("I", 53)];
    let mut atoms = Vec::new();
    for t in shifts {
        for ((sym, z), r) in species.iter().zip(pos) {
            atoms.push(atom(sym, *z, [r[0] + t[0], r[1] + t[1], r[2] + t[2]]));
        }
    }
    let mut mol = Molecule {
        atoms,
        charge: 0,
        multiplicity: 1,
    };
    mol.apply_ecp(bs);
    Cell::new(mol, lattice).expect("ECP cell")
}

fn z_shifts(n: usize) -> Vec<[f64; 3]> {
    (0..n).map(|m| [0.0, 0.0, m as f64 * HI_A[2][2]]).collect()
}

fn z_lattice(n: usize) -> [[f64; 3]; 3] {
    let mut lat = HI_A;
    lat[2][2] *= n as f64;
    lat
}

fn hi_cell(pos: &[[f64; 3]], bs: &BasisSet) -> Cell {
    ecp_cell(pos, &[[0.0; 3]], HI_A, bs)
}

fn hi_supercell(pos: &[[f64; 3]], n: usize, bs: &BasisSet) -> Cell {
    ecp_cell(pos, &z_shifts(n), z_lattice(n), bs)
}

fn moved(pos: &[[f64; 3]], a: usize, x: usize, h: f64) -> Vec<[f64; 3]> {
    let mut p = pos.to_vec();
    p[a][x] += h;
    p
}

fn on() -> SrColumnRotation {
    SrColumnRotation::on()
}

fn hcore_cfg(rot: SrColumnRotation) -> PeriodicHcoreConfig {
    PeriodicHcoreConfig {
        precision: HCORE_PRECISION,
        ..PeriodicHcoreConfig::with_omega(OMEGA)
    }
    .with_sr_column_rotation(rot)
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

fn mat_scaled(m: &Mat3, f: f64) -> Mat3 {
    let mut o = *m;
    for r in o.iter_mut() {
        for v in r.iter_mut() {
            *v *= f;
        }
    }
    o
}

fn grad_miss(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    (a - b).iter().fold(0.0_f64, |m, x| m.max(x.abs()))
}

fn amax(a: &Array2<f64>) -> f64 {
    a.iter().fold(0.0_f64, |m, x| m.max(x.abs()))
}

fn bits_equal(a: &Array2<f64>, b: &Array2<f64>) -> bool {
    a.dim() == b.dim() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

fn mat_bits_equal(a: &Mat3, b: &Mat3) -> bool {
    (0..3).all(|i| (0..3).all(|j| a[i][j].to_bits() == b[i][j].to_bits()))
}

/// A fixed Hermitian `D(k)` with `D(-k) = D(k)*` coupling neighbouring cells
/// along z (complex at every non-TRIM k of a 1x1x3 mesh).
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

struct KSys {
    cell: Cell,
    prep: PreparedBasis,
    mesh: KPointMesh,
    hk: PeriodicHcoreK,
    hcfg: PeriodicHcoreConfig,
}

fn ksys(cell: Cell, bs: &BasisSet, n: [usize; 3], rot: SrColumnRotation) -> KSys {
    let prep = prep_for(&cell, bs);
    let mesh = KPointMesh::gamma_centred(&cell, n).expect("mesh");
    let hcfg = hcore_cfg(rot);
    let hk = periodic_hcore_kpts(&cell, &prep, &mesh, &hcfg).expect("hcore(k)");
    assert!(hk.n_ecp_triples > 0, "no ECP triples: the cell is vacuous");
    KSys {
        cell,
        prep,
        mesh,
        hk,
        hcfg,
    }
}

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

fn krhf(ks: &KSys, jk: Box<dyn KPointJk + '_>) -> KScfResult {
    let inj = KPointInjection {
        s: ks.hk.s.clone(),
        h: ks.hk.h.clone(),
        vnn: ks.hk.enn,
        jk,
    };
    let r = solve_krhf_injected(&ks.cell, &ks.mesh, &kscf_cfg(), inj).expect("k-RHF");
    assert!(r.converged, "k-RHF not converged ({} it)", r.iterations);
    r
}

fn gcfg(m: Option<KGradMutation>, zeta: f64) -> KGradConfig {
    KGradConfig {
        budget_bytes: Some(AMPLE),
        mutation: m,
        nucleus_exponent: Some(zeta),
    }
}

fn scfg(m: Option<KStressMutation>, zeta: f64) -> KStressConfig {
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
    m: Option<KGradMutation>,
    zeta: f64,
) -> KGradient {
    kpoint_rhf_gradient(
        &ks.cell,
        &ks.prep,
        &ks.mesh,
        &ks.hcfg,
        &ks.hk,
        KGradJk::Dense(eri),
        scf,
        ExxDiv::None,
        &gcfg(m, zeta),
    )
    .expect("kpoint_rhf_gradient")
}

fn kstress(
    ks: &KSys,
    eri: &KDenseAftEri,
    scf: &KScfResult,
    m: Option<KStressMutation>,
    zeta: f64,
) -> KStress {
    kpoint_rhf_stress(
        &ks.cell,
        &ks.prep,
        &ks.mesh,
        &ks.hcfg,
        &ks.hk,
        KStressJk::Dense(eri),
        scf,
        ExxDiv::None,
        &scfg(m, zeta),
    )
    .expect("kpoint_rhf_stress")
}

/// k-RHF energy (exxdiv none, dense AFT) of a (displaced / strained) cell.
fn k_energy(cell: &Cell, bs: &BasisSet, n: [usize; 3], rot: SrColumnRotation) -> f64 {
    let ks = ksys(cell.clone(), bs, n, rot);
    let eri = kdense(&ks, ANCHOR_PRECISION);
    krhf(&ks, Box::new(eri.jk_builder_with_madelung(0.0))).energy
}

fn real_of(d: &Array2<Complex64>) -> Array2<f64> {
    let im = d.iter().fold(0.0_f64, |m, z| m.max(z.im.abs()));
    // MEASURED 3.3e-11 here (SCF grad_conv 1e-10 leaves that much imaginary
    // density at Gamma); dropping it is second order in the Gamma anchors.
    assert!(im < 1e-9, "Gamma density has an imaginary part {im:e}");
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
        convergence: None,
        rohf_spin_focks: None,
    }
}

struct GSys {
    cell: Cell,
    prep: PreparedBasis,
    hcfg: PeriodicHcoreConfig,
    hc: ferric_pbc::hcore::PeriodicHcore,
    eri: DenseAftEri,
}

fn gsys(cell: Cell, bs: &BasisSet, rot: SrColumnRotation) -> GSys {
    let prep = prep_for(&cell, bs);
    let hcfg = hcore_cfg(rot);
    let hc = periodic_hcore(&cell, &prep, &hcfg).expect("gamma hcore");
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
        hcfg,
        hc,
        eri,
    }
}

// ============================================================ (1) fixed D

/// Bar of the fixed-density hcore FD gate (central FD, h = 1e-4, all nine
/// components). MEASURED 8.2e-8 on a strain of max |component| 4.9 (relative
/// 1.7e-8: the h^2 truncation of the central difference) at the default
/// nucleus exponent. The test runs
/// the rotation mutants at fixed density: totals MEASURED at the default move by RotDropT
/// 1.03e-1, RotUseT 5.05e-1, RotOneSided 9.06e-2, RotFlipSign 2.15e-1,
/// RotHcoreWrongResidue 1.07, each > 10x this bar; RotSwapPair and
/// RotFitWrongResidue act on the fit only and move nothing (0).
const HCORE_FD_BAR: f64 = 3e-7;
/// The rotated energy's hcore (screen 1e-14) vs the unrotated one.
const ROT_VS_OFF_H_BAR: f64 = 1e-10;

/// PHYSICS: the basis really rotates, the rotation is on the ECP centre as
/// well as on H, and the ECP operator is untouched by it. ARTIFACT: a
/// vacuous test (no rotation, or rotation that also moved V_ECP) would pass
/// every anchor below; so the counts and the BITWISE equality of S, T and
/// V_ECP(k) between the rotated and unrotated builds are asserted first.
#[test]
fn rotating_ecp_basis_rotates_and_leaves_the_ecp_untouched() {
    let bs = rot_basis();
    let cell = hi_cell(&MOVED, &bs);
    let on_ = ksys(cell.clone(), &bs, [1, 1, 3], on());
    let off = ksys(cell, &bs, [1, 1, 3], SrColumnRotation::Off);
    eprintln!(
        "rotated columns: {} (unrotated build {}); nbasis {}",
        on_.hk.sr_rotated_columns,
        off.hk.sr_rotated_columns,
        on_.prep.nbasis()
    );
    assert_eq!(off.hk.sr_rotated_columns, 0);
    // One s column on each of H (cc-pVDZ), I s and I p: 3 columns.
    assert_eq!(on_.hk.sr_rotated_columns, 3, "H s + I s + I p");
    assert_eq!(on_.hk.n_ecp_triples, off.hk.n_ecp_triples);
    let von = on_.hk.v_ecp.as_ref().unwrap();
    let voff = off.hk.v_ecp.as_ref().unwrap();
    for (a, b) in von.iter().zip(voff) {
        assert!(a
            .iter()
            .zip(b)
            .all(|(x, y)| x.re.to_bits() == y.re.to_bits() && x.im.to_bits() == y.im.to_bits()));
    }
    for (a, b) in on_.hk.s.iter().zip(&off.hk.s) {
        assert!(a
            .iter()
            .zip(b)
            .all(|(x, y)| x.re.to_bits() == y.re.to_bits()));
    }
    let mut dh = 0.0_f64;
    for (a, b) in on_.hk.h.iter().zip(&off.hk.h) {
        for (x, y) in a.iter().zip(b) {
            dh = dh.max((x - y).norm());
        }
    }
    eprintln!("rotated vs unrotated h(k): {dh:.2e}");
    assert!(dh < ROT_VS_OFF_H_BAR, "{dh:e}");
}

/// PHYSICS: at a FIXED density the rotated hcore strain (kinetic + V_SR with
/// ECP centre rotated + V_LR + ECP + Ewald) equals central FD of
/// `(1/N_k) Σ Re tr[D h(k)] + E_nn` with the rotation ON in every strained
/// build, all nine components; the ECP part is BITWISE the unrotated one and
/// the ECP force (fixed D) too. ARTIFACT: FD never touches the derivative
/// walks, so a weight slip in the rotated SR strain walk or an ECP weight
/// that picked up T shows here before any SCF is involved.
#[test]
#[ignore = "slow (~73 s release): 18 strained rotated k hcore builds at precision 1e-14 with an ECP"]
fn rotated_hcore_with_ecp_at_fixed_density_matches_fd_and_keeps_the_ecp_part() {
    let bs = rot_basis();
    let n = [1, 1, 3];
    let cell = hi_cell(&MOVED, &bs);
    let ks = ksys(cell.clone(), &bs, n, on());
    let off = ksys(cell.clone(), &bs, n, SrColumnRotation::Off);
    let dk = fixed_dk(&ks.mesh, ks.prep.nbasis());
    let kcfg = scfg(None, GRAD_NUCLEUS_EXPONENT);
    let s_on =
        kpoint_hcore_stress_fixed_d(&ks.cell, &ks.prep, &ks.mesh, &ks.hcfg, &ks.hk, &dk, &kcfg)
            .unwrap();
    let s_off = kpoint_hcore_stress_fixed_d(
        &off.cell, &off.prep, &off.mesh, &off.hcfg, &off.hk, &dk, &kcfg,
    )
    .unwrap();
    assert_eq!(s_on.hcore_rotated_columns, 3);
    assert_eq!(s_off.hcore_rotated_columns, 0);
    assert!(mat_max(&s_on.ecp) > 1e-3, "vacuous ECP strain");
    assert!(
        mat_bits_equal(&s_on.ecp, &s_off.ecp),
        "the rotation moved the ECP strain part by {:.2e}",
        max_mat(&s_on.ecp, &s_off.ecp)
    );
    let g_on = kpoint_ecp_gradient_fixed_d(&ks.cell, &ks.prep, &ks.mesh, &ks.hcfg, &dk)
        .unwrap()
        .unwrap();
    let g_off = kpoint_ecp_gradient_fixed_d(&off.cell, &off.prep, &off.mesh, &off.hcfg, &dk)
        .unwrap()
        .unwrap();
    assert!(amax(&g_on.grad) > 1e-3);
    assert!(bits_equal(&g_on.grad, &g_off.grad), "ECP force moved");
    let e = |c: &Cell| -> f64 {
        let k = ksys(c.clone(), &bs, n, on());
        let mut e = 0.0;
        for (d, h) in dk.iter().zip(&k.hk.h) {
            for m in 0..d.nrows() {
                for q in 0..d.nrows() {
                    e += (d[(m, q)] * h[(q, m)]).re;
                }
            }
        }
        e / k.mesh.nk() as f64 + k.hk.enn
    };
    let mut fd = [[0.0; 3]; 3];
    for (i, j) in ALL9 {
        let ep = e(&cell.strained(&eps_at(i, j, FD_H)).unwrap());
        let em = e(&cell.strained(&eps_at(i, j, -FD_H)).unwrap());
        fd[i][j] = (ep - em) / (2.0 * FD_H);
    }
    let tot = s_on.hcore_total();
    let miss = max_mat(&tot, &fd);
    let tot_off = s_off.hcore_total();
    eprintln!(
        "rotated hcore+ECP fixed-D strain: |analytic - FD| {miss:.2e} (max {:.2e}); \
         rotated vs unrotated analytic {:.2e}",
        mat_max(&tot),
        max_mat(&tot, &tot_off)
    );
    for a in 0..3 {
        eprintln!(
            "  [{a}] analytic {:+.8e} {:+.8e} {:+.8e} | FD {:+.8e} {:+.8e} {:+.8e}",
            tot[a][0], tot[a][1], tot[a][2], fd[a][0], fd[a][1], fd[a][2]
        );
    }
    assert!(miss < HCORE_FD_BAR, "{miss:e}");
    // Production-wired rotation mutants at fixed density.
    for m in [
        KStressMutation::RotDropT,
        KStressMutation::RotUseT,
        KStressMutation::RotOneSided,
        KStressMutation::RotFlipSign,
        KStressMutation::RotHcoreWrongResidue,
        KStressMutation::RotSwapPair,
        KStressMutation::RotFitWrongResidue,
    ] {
        let sm = kpoint_hcore_stress_fixed_d(
            &ks.cell,
            &ks.prep,
            &ks.mesh,
            &ks.hcfg,
            &ks.hk,
            &dk,
            &scfg(Some(m), GRAD_NUCLEUS_EXPONENT),
        )
        .unwrap();
        let moved = max_mat(&sm.hcore_total(), &tot);
        eprintln!("  fixed-D hcore mutant {m:?}: total moves {moved:.2e}");
        if matches!(
            m,
            KStressMutation::RotSwapPair | KStressMutation::RotFitWrongResidue
        ) {
            assert!(moved < 1e-12, "{m:?} moved an hcore-only total {moved:e}");
        } else {
            assert!(moved > 10.0 * HCORE_FD_BAR, "{m:?} escaped: {moved:e}");
        }
    }
}

// ============================================================ (2) SCF level

/// Bars of the 1x1x1 and 1x1x3 anchors at the default nucleus exponent.
/// MEASURED 1x1x3 (24 min release): force TOTAL 4.6e-13, stress TOTAL 3.0e-12,
/// vsr parts <= 1.5e-14, every other part <= 4.8e-13; 1x1x1 force total
/// 5.6e-12, stress total 1.1e-12, ECP parts 0 (expected, not evidence: both
/// sides call the same ECP kernel at one point; see the section 4 note).
const SC_BAR_TOTAL: f64 = 3e-11;
const SC_BAR_SR: f64 = 1e-12;
const SC_BAR_PART: f64 = 3e-12;
const ONE_POINT_TOTAL: f64 = 1e-10;
const ONE_POINT_ECP: f64 = 1e-12;
/// SCF-level FD bars (h = 1e-4; unrotated ECP file: force 2e-8, stress 2e-7).
const FORCE_FD_BAR: f64 = 2e-8;
const STRESS_FD_BAR: f64 = 2e-7;
/// Smallest movement a production-wired rotation mutant must make.
const MUTANT_FLOOR: f64 = 1e-6;
/// Rotated vs unrotated derivative of the SAME energy and density (the walks
/// are the same function up to the screening). Dense AFT source, hcore screen
/// 1e-14: MEASURED force 9.6e-15, stress 4.8e-13 (bar 20x above).
///
/// RS-GDF source (`ROT_VS_OFF_FIT_BAR`), the same anchor on two builds
/// (default allocator):
///
/// | build                      | force   | stress  |
/// |----------------------------|---------|---------|
/// | fixed (#437)               | 7.02e-14 | 2.59e-12 |
/// | pre-#437 (defect present)  | 3.28e-10 | 1.19e-7  |
///
/// The pre-#437 row is NOT a screening floor: it is the footprint of the #433
/// Boys table-edge read in the UNROTATED fit (the rotated one is clean), and
/// it is why this bar used to be 5e-7, which sat above the defective run and
/// could not have caught it. Measured on the fixed build the two walks agree
/// to the screen (<= 2.6e-12). The bar 5e-11 separates the rows with margin on
/// both sides: 19x above the fixed stress (2.59e-12), 712x above the fixed
/// force (7.02e-14); 6.5x below the defective force (3.28e-10), 2380x below the
/// defective stress (1.19e-7). A rotation defect moves the total by >= 5e-2.
/// This comparison is asserted LAST in the RS-GDF test (the FD anchor next to
/// it is the discriminating one for rotation defects).
const ROT_VS_OFF_BAR: f64 = 1e-11;
const ROT_VS_OFF_FIT_BAR: f64 = 5e-11;

struct Run {
    ks: KSys,
    eri: KDenseAftEri,
    scf: KScfResult,
}

fn run(cell: Cell, bs: &BasisSet, n: [usize; 3], rot: SrColumnRotation) -> Run {
    let ks = ksys(cell, bs, n, rot);
    let eri = kdense(&ks, ANCHOR_PRECISION);
    let scf = krhf(&ks, Box::new(eri.jk_builder_with_madelung(0.0)));
    Run { ks, eri, scf }
}

impl Run {
    fn force(&self, m: Option<KGradMutation>, zeta: f64) -> KGradient {
        kforce(&self.ks, &self.eri, &self.scf, m, zeta)
    }
    fn stress(&self, m: Option<KStressMutation>, zeta: f64) -> KStress {
        kstress(&self.ks, &self.eri, &self.scf, m, zeta)
    }
}

fn gamma_force(g: &GSys, scf: &ScfResult, zeta: f64) -> ferric_pbc::grad::GammaGradient {
    gamma_rhf_gradient_with(
        &g.cell,
        &g.prep,
        &g.hcfg,
        &g.hc,
        &g.eri,
        scf,
        ExxDiv::None,
        &GammaGradConfig {
            budget_bytes: Some(AMPLE),
            nucleus_exponent: Some(zeta),
            ..Default::default()
        },
    )
    .expect("gamma gradient")
}

fn gamma_stress(g: &GSys, scf: &ScfResult, zeta: f64) -> ferric_pbc::stress::GammaStress {
    gamma_rhf_stress(
        &g.cell,
        &g.prep,
        &g.hcfg,
        &g.hc,
        &g.eri,
        scf,
        ExxDiv::None,
        &GammaStressConfig {
            budget_bytes: Some(AMPLE),
            nucleus_exponent: Some(zeta),
            ..Default::default()
        },
    )
    .expect("gamma stress")
}

/// PHYSICS: a one-point Gamma mesh IS the Gamma calculation, here with the
/// rotation on in BOTH (k hcore and Gamma hcore rotate the same 3 columns per
/// cell). ARTIFACT: blind to k-specific defects (every phase is 1), so the
/// 1x1x3 anchor below carries those; what this adds is that the rotated ECP
/// cell's total and ECP parts match the independent real Gamma code.
#[test]
#[ignore = "slow (~131 s release): rotated HI 1x1x1 k SCF + Gamma hcore/dense AFT/force/stress"]
fn rotated_ecp_one_point_mesh_force_and_stress_are_the_gamma_ones() {
    let bs = rot_basis();
    let cell = hi_cell(&MOVED, &bs);
    let r = run(cell.clone(), &bs, [1, 1, 1], on());
    assert_eq!(r.ks.hk.sr_rotated_columns, 3);
    let g = gsys(cell, &bs, on());
    assert_eq!(g.hc.sr_rotated_columns, 3);
    let gscf = gamma_restricted(real_of(&r.scf.densities[0]));
    let gf = gamma_force(&g, &gscf, GRAD_NUCLEUS_EXPONENT);
    let gs = gamma_stress(&g, &gscf, GRAD_NUCLEUS_EXPONENT);
    let kf = r.force(None, GRAD_NUCLEUS_EXPONENT);
    let ks = r.stress(None, GRAD_NUCLEUS_EXPONENT);
    assert_eq!(kf.hcore_rotated_columns, 3);
    assert_eq!(ks.hcore_rotated_columns, 3);
    let (f_ecp, f_tot) = (
        grad_miss(&kf.parts.ecp, &gf.parts.ecp),
        grad_miss(&kf.grad, &gf.grad),
    );
    let (s_ecp, s_tot) = (
        max_mat(&ks.parts.ecp, &gs.parts.ecp),
        max_mat(&ks.de_deps, &gs.de_deps),
    );
    eprintln!(
        "rotated 1x1x1 vs Gamma: force ecp {f_ecp:.2e} total {f_tot:.2e} (|F_ecp| {:.2e}); \
         stress ecp {s_ecp:.2e} total {s_tot:.2e} (|S_ecp| {:.2e})",
        amax(&kf.parts.ecp),
        mat_max(&ks.parts.ecp)
    );
    assert!(amax(&kf.parts.ecp) > 1e-3 && mat_max(&ks.parts.ecp) > 1e-3);
    assert!(f_ecp < ONE_POINT_ECP && s_ecp < ONE_POINT_ECP);
    assert!(f_tot < ONE_POINT_TOTAL && s_tot < ONE_POINT_TOTAL);
}

fn third(m: &Mat3) -> Mat3 {
    mat_scaled(m, 1.0 / 3.0)
}

/// `max |F_k[A, x] - F_sc[c * natoms + A, x]|` over every copy `c`.
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

/// PHYSICS: the Gamma-centred 1x1x3 mesh IS the diag(1,1,3) supercell, so per
/// cell the force on atom A equals the supercell force on any copy of A and
/// `dE/de = dE/de_sc / 3`, with the rotation (3 columns per cell, 9 in the
/// supercell) and the ECP both on. ARTIFACT: independent construction
/// (explicit supercell, real Gamma walks, its own rotated SR derivative walk
/// on 3x the columns). A rotated-walk weight that mixed images, or an ECP
/// weight that picked up T, would not reproduce it on the NON-TRIM mesh.
#[test]
#[ignore = "slow (~24 min release, quiet box): rotated HI 1x1x3 k SCF + 3-cell Gamma supercell hcore/dense AFT/force/stress"]
fn rotated_ecp_1x1x3_force_and_stress_are_the_supercell_ones_per_cell() {
    let bs = rot_basis();
    let r = run(hi_cell(&MOVED, &bs), &bs, [1, 1, 3], on());
    assert_eq!(r.ks.hk.sr_rotated_columns, 3);
    let g = gsys(hi_supercell(&MOVED, 3, &bs), &bs, on());
    assert_eq!(g.hc.sr_rotated_columns, 9);
    let gscf = gamma_restricted(unfold_z(&r.ks.mesh, &r.scf.densities));
    let zeta = GRAD_NUCLEUS_EXPONENT;
    let gf = gamma_force(&g, &gscf, zeta);
    let gs = gamma_stress(&g, &gscf, zeta);
    let kf = r.force(None, zeta);
    let ks = r.stress(None, zeta);
    assert_eq!(kf.hcore_rotated_columns, 3);
    assert_eq!(ks.hcore_rotated_columns, 3);
    let (kp, gp) = (&kf.parts, &gf.parts);
    let eri_k = &kp.coulomb + &kp.exchange;
    let force = [
        ("kinetic", sc_miss(&kp.kinetic, &gp.kinetic), SC_BAR_PART),
        ("overlap", sc_miss(&kp.overlap, &gp.overlap), SC_BAR_PART),
        ("ecp", sc_miss(&kp.ecp, &gp.ecp), SC_BAR_PART),
        (
            "vsr_basis",
            sc_miss(&kp.vsr_basis, &gp.vsr_basis),
            SC_BAR_SR,
        ),
        ("vsr_nuc", sc_miss(&kp.vsr_nuc, &gp.vsr_nuc), SC_BAR_SR),
        (
            "vlr_basis",
            sc_miss(&kp.vlr_basis, &gp.vlr_basis),
            SC_BAR_PART,
        ),
        ("eri", sc_miss(&eri_k, &gp.eri), SC_BAR_PART),
        ("TOTAL", sc_miss(&kf.grad, &gf.grad), SC_BAR_TOTAL),
    ];
    let (sk, sg) = (&ks.parts, &gs.parts);
    let stress = [
        (
            "kinetic",
            max_mat(&sk.kinetic, &third(&sg.kinetic)),
            SC_BAR_PART,
        ),
        ("ecp", max_mat(&sk.ecp, &third(&sg.ecp)), SC_BAR_PART),
        ("vsr", max_mat(&sk.vsr, &third(&sg.vsr)), SC_BAR_SR),
        ("vlr", max_mat(&sk.vlr, &third(&sg.vlr)), SC_BAR_PART),
        (
            "TOTAL",
            max_mat(&ks.de_deps, &third(&gs.de_deps)),
            SC_BAR_TOTAL,
        ),
    ];
    for (kind, parts) in [("force", &force[..]), ("stress", &stress[..])] {
        for (name, miss, bar) in parts {
            eprintln!("  rotated+ECP 1x1x3 {kind} {name}: {miss:.2e} (bar {bar:.0e})");
            assert!(miss < bar, "{kind} {name} {miss:e} >= {bar:e}");
        }
    }
    assert!(
        amax(&kp.ecp) > 1e-3 && mat_max(&sk.ecp) > 1e-3,
        "vacuous ECP"
    );
}

const FORCE_MUTANTS: [KGradMutation; 7] = [
    KGradMutation::RotDropT,
    KGradMutation::RotUseT,
    KGradMutation::RotOneSided,
    KGradMutation::RotFlipSign,
    KGradMutation::RotSwapPair,
    KGradMutation::RotFitWrongResidue,
    KGradMutation::RotHcoreWrongResidue,
];
const STRESS_MUTANTS: [KStressMutation; 7] = [
    KStressMutation::RotDropT,
    KStressMutation::RotUseT,
    KStressMutation::RotOneSided,
    KStressMutation::RotFlipSign,
    KStressMutation::RotSwapPair,
    KStressMutation::RotFitWrongResidue,
    KStressMutation::RotHcoreWrongResidue,
];

fn eps_cell(cell: &Cell, i: usize, j: usize, h: f64) -> Cell {
    cell.strained(&eps_at(i, j, h)).expect("strained")
}

/// PHYSICS (the anchor that does not use any derivative walk): analytic force
/// on I x / H z and stress zz / xz / xx of the rotated + ECP k energy vs
/// central FD of the ROTATED energy (rotation on in every displaced build;
/// fresh hcore, dense AFT and SCF per point). Also: the rotated analytic
/// equals the UNROTATED analytic of the same energy (ROT_VS_OFF_BAR, two
/// constructions of one function), the walked-column counters equal the
/// energy's (3), the ECP part is the unrotated one, and every
/// production-wired rotation mutant moves the total (MUTANT_FLOOR).
/// `RotFitWrongResidue` acts only on the fit and `RotSwapPair` only on a
/// fit pair, so with the dense source they cannot move anything: they are
/// asserted to leave the dense total UNCHANGED (a no-op by construction).
#[test]
#[ignore = "slow (~106 min release, quiet box; each ECP stress evaluation is minutes; MEASURED force 3.3e-9, stress 2.2e-8): 1 + 10 rotated k pipelines, 2 analytic pipelines, 12 mutant evaluations"]
fn rotated_ecp_force_and_stress_match_fd_of_the_rotated_k_energy() {
    let bs = rot_basis();
    let n = [1, 1, 3];
    let cell = hi_cell(&MOVED, &bs);
    // FD first (clean process state).
    let forces = [(1usize, 0usize), (0, 2)];
    let strains = [(2usize, 2usize), (0, 2), (0, 0)];
    let mut fd_f = Vec::new();
    for &(a, x) in &forces {
        let ep = k_energy(&hi_cell(&moved(&MOVED, a, x, FD_H), &bs), &bs, n, on());
        let em = k_energy(&hi_cell(&moved(&MOVED, a, x, -FD_H), &bs), &bs, n, on());
        fd_f.push((ep - em) / (2.0 * FD_H));
    }
    let mut fd_s = Vec::new();
    for &(i, j) in &strains {
        let ep = k_energy(&eps_cell(&cell, i, j, FD_H), &bs, n, on());
        let em = k_energy(&eps_cell(&cell, i, j, -FD_H), &bs, n, on());
        fd_s.push((ep - em) / (2.0 * FD_H));
    }
    let r = run(cell.clone(), &bs, n, on());
    let off = run(cell, &bs, n, SrColumnRotation::Off);
    assert_eq!(r.ks.hk.sr_rotated_columns, 3);
    let zeta = GRAD_NUCLEUS_EXPONENT;
    let f = r.force(None, zeta);
    let s = r.stress(None, zeta);
    let fo = off.force(None, zeta);
    let so = off.stress(None, zeta);
    assert_eq!((f.hcore_rotated_columns, s.hcore_rotated_columns), (3, 3));
    assert_eq!((fo.hcore_rotated_columns, so.hcore_rotated_columns), (0, 0));
    let mut worst_f = 0.0_f64;
    for (k, &(a, x)) in forces.iter().enumerate() {
        let d = (f.grad[(a, x)] - fd_f[k]).abs();
        assert!(
            d.is_finite(),
            "force component ({a},{x}): non-finite difference"
        );
        worst_f = worst_f.max(d);
    }
    let mut worst_s = 0.0_f64;
    for (k, &(i, j)) in strains.iter().enumerate() {
        let d = (s.de_deps[i][j] - fd_s[k]).abs();
        assert!(
            d.is_finite(),
            "stress component ({i},{j}): non-finite difference"
        );
        worst_s = worst_s.max(d);
    }
    let (d_f, d_s) = (
        grad_miss(&f.grad, &fo.grad),
        max_mat(&s.de_deps, &so.de_deps),
    );
    eprintln!(
        "rotated+ECP vs FD: force {worst_f:.2e} (bar {FORCE_FD_BAR:.0e}), stress {worst_s:.2e} \
         (bar {STRESS_FD_BAR:.0e}); rotated vs unrotated analytic: force {d_f:.2e}, stress {d_s:.2e}"
    );
    assert!(worst_f < FORCE_FD_BAR && worst_s < STRESS_FD_BAR);
    assert!(d_f < ROT_VS_OFF_BAR && d_s < ROT_VS_OFF_BAR);
    let (e_f, e_s) = (
        grad_miss(&f.parts.ecp, &fo.parts.ecp),
        max_mat(&s.parts.ecp, &so.parts.ecp),
    );
    eprintln!("ECP part rotated vs unrotated run: force {e_f:.2e}, stress {e_s:.2e}");
    assert!(e_f < 1e-12 && e_s < 1e-12);
    for m in FORCE_MUTANTS {
        let g = r.force(Some(m), zeta);
        let moved_total = grad_miss(&g.grad, &f.grad);
        eprintln!("  force mutant {m:?}: total moves {moved_total:.2e}");
        let fit_only = matches!(
            m,
            KGradMutation::RotFitWrongResidue | KGradMutation::RotSwapPair
        );
        if fit_only {
            assert!(
                moved_total < 1e-12,
                "{m:?} moved a dense total {moved_total:e}"
            );
        } else {
            assert!(moved_total > MUTANT_FLOOR, "{m:?} escaped: {moved_total:e}");
        }
    }
    for m in STRESS_MUTANTS {
        // The fit-only stress mutants are no-ops with a dense source (measured
        // 0.00e0 for both the force and the stress; the force loop above
        // asserts it, and the RS-GDF test below exercises them where they
        // act); an ECP stress evaluation costs minutes, so they are skipped.
        if matches!(
            m,
            KStressMutation::RotFitWrongResidue | KStressMutation::RotSwapPair
        ) {
            continue;
        }
        let g = r.stress(Some(m), zeta);
        let moved_total = max_mat(&g.de_deps, &s.de_deps);
        eprintln!("  stress mutant {m:?}: total moves {moved_total:.2e}");
        assert!(moved_total > MUTANT_FLOOR, "{m:?} escaped: {moved_total:e}");
    }
}

// ============================================================ (3) RS-GDF

/// Even-tempered s/p aux on H and I (one unit-normalised primitive per
/// shell; the fit quality is irrelevant: forces and stress are the exact
/// derivatives of the FITTED energy, whatever the aux).
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

fn krgdf_cfg(rot: SrColumnRotation) -> KRsGdfConfig {
    KRsGdfConfig {
        gdf: RsGdfConfig {
            omega: 1.0,
            exxdiv: ExxDiv::None,
            budget_bytes: Some(AMPLE),
            sr_column_rotation: rot,
            ..Default::default()
        },
        mutation: None,
    }
}

struct FitRun {
    ks: KSys,
    cfg: KRsGdfConfig,
    aux: PreparedBasis,
    gdf: KRsGdf,
    scf: KScfResult,
}

fn fit_run(cell: Cell, bs: &BasisSet, n: [usize; 3], rot: SrColumnRotation) -> FitRun {
    let ks = ksys(cell, bs, n, rot);
    let cfg = krgdf_cfg(rot);
    let aux = PreparedBasis::new(ks.cell.mol(), &hi_aux()).expect("aux prep");
    let gdf = KRsGdf::build(&ks.cell, &ks.prep, &aux, &ks.mesh, &ks.hk.s, &cfg).expect("KRsGdf");
    let scf = krhf(&ks, Box::new(gdf.jk_builder_with_madelung(0.0)));
    FitRun {
        ks,
        cfg,
        aux,
        gdf,
        scf,
    }
}

impl FitRun {
    fn src(&self) -> KRsGdfGradSource<'_> {
        KRsGdfGradSource {
            gdf: &self.gdf,
            cfg: &self.cfg,
            aux: &self.aux,
        }
    }
}

fn k_fit_energy(cell: &Cell, bs: &BasisSet, n: [usize; 3], rot: SrColumnRotation) -> f64 {
    fit_run(cell.clone(), bs, n, rot).scf.energy
}

/// RS-GDF (unsplit) k force and stress with an ECP AND a rotating basis: the
/// fit rotates the SR 3-centre columns of H and of I, the hcore rotates the
/// SR attraction, the ECP stays unrotated. PHYSICS: analytic force (I x, H z)
/// and stress (zz, xz) vs central FD of the fitted ROTATED energy (fresh
/// hcore, fit, SCF per point). ARTIFACT: a fit-side rotated walk that
/// double-transformed the ECP centre's columns, or a hcore-side slip that
/// only appears once the fit is also rotated, is invisible to the dense
/// anchors (the dense J/K does not rotate). Also: the walked-column counters
/// of BOTH walks equal the energy's, and the rotated analytic equals the
/// unrotated RS-GDF analytic of the same energy to the screen.
///
/// HISTORY (#433, fixed in #437): before the Boys table-edge fix the UNROTATED
/// fit for this basis read uninitialised memory (under `MALLOC_PERTURB_=1`:
/// LAPACK failure; 170: "k-RHF not converged (400 it)"), so the
/// `ROT_VS_OFF_FIT_BAR` comparison was against a non-reproducible run. It is
/// allocator-independent now (see the module doc). The FD and mutant parts use
/// only the rotated fit.
#[test]
#[ignore = "slow (~168 s release, quiet box): 1 + 8 rotated RS-GDF k pipelines, HI 1x1x3, + one unrotated pipeline"]
fn rotated_ecp_rsgdf_force_and_stress_match_fd_of_the_rotated_fitted_energy() {
    let bs = rot_basis();
    let n = [1, 1, 3];
    let cell = hi_cell(&MOVED, &bs);
    let forces = [(1usize, 0usize), (0, 2)];
    let strains = [(2usize, 2usize), (0, 2)];
    let mut fd_f = Vec::new();
    for &(a, x) in &forces {
        let ep = k_fit_energy(&hi_cell(&moved(&MOVED, a, x, FD_H), &bs), &bs, n, on());
        let em = k_fit_energy(&hi_cell(&moved(&MOVED, a, x, -FD_H), &bs), &bs, n, on());
        fd_f.push((ep - em) / (2.0 * FD_H));
    }
    let mut fd_s = Vec::new();
    for &(i, j) in &strains {
        let ep = k_fit_energy(&eps_cell(&cell, i, j, FD_H), &bs, n, on());
        let em = k_fit_energy(&eps_cell(&cell, i, j, -FD_H), &bs, n, on());
        fd_s.push((ep - em) / (2.0 * FD_H));
    }
    let r = fit_run(cell.clone(), &bs, n, on());
    assert_eq!(r.ks.hk.sr_rotated_columns, 3);
    assert!(
        r.gdf.stats().sr_rotated_columns > 0,
        "the fit did not rotate"
    );
    let zeta = GRAD_NUCLEUS_EXPONENT;
    let force = |fr: &FitRun| {
        kpoint_rhf_gradient(
            &fr.ks.cell,
            &fr.ks.prep,
            &fr.ks.mesh,
            &fr.ks.hcfg,
            &fr.ks.hk,
            KGradJk::RsGdf(fr.src()),
            &fr.scf,
            ExxDiv::None,
            &gcfg(None, zeta),
        )
        .expect("fit force")
    };
    let stress = |fr: &FitRun| {
        kpoint_rhf_stress(
            &fr.ks.cell,
            &fr.ks.prep,
            &fr.ks.mesh,
            &fr.ks.hcfg,
            &fr.ks.hk,
            KStressJk::RsGdf(fr.src()),
            &fr.scf,
            ExxDiv::None,
            &scfg(None, zeta),
        )
        .expect("fit stress")
    };
    let (f, s) = (force(&r), stress(&r));
    // Both walks ran the energy's columns.
    assert_eq!(f.hcore_rotated_columns, 3);
    assert_eq!(s.hcore_rotated_columns, 3);
    let fit_cols = r.gdf.stats().sr_rotated_columns;
    assert_eq!(f.fit_checks.map(|c| c.rotated_columns), Some(fit_cols));
    assert_eq!(s.fit_checks.map(|c| c.rotated_columns), Some(fit_cols));
    let mut worst_f = 0.0_f64;
    for (k, &(a, x)) in forces.iter().enumerate() {
        let d = (f.grad[(a, x)] - fd_f[k]).abs();
        assert!(
            d.is_finite(),
            "force component ({a},{x}): non-finite difference"
        );
        worst_f = worst_f.max(d);
    }
    let mut worst_s = 0.0_f64;
    for (k, &(i, j)) in strains.iter().enumerate() {
        let d = (s.de_deps[i][j] - fd_s[k]).abs();
        assert!(
            d.is_finite(),
            "stress component ({i},{j}): non-finite difference"
        );
        worst_s = worst_s.max(d);
    }
    eprintln!(
        "rotated+ECP RS-GDF vs FD: force {worst_f:.2e}, stress {worst_s:.2e}; fit columns {fit_cols}"
    );
    assert!(worst_f < FORCE_FD_BAR && worst_s < STRESS_FD_BAR);
    // Mutants through the fit: RotFitWrongResidue and RotSwapPair act here.
    for m in [
        KStressMutation::RotFitWrongResidue,
        KStressMutation::RotSwapPair,
        KStressMutation::RotDropT,
        KStressMutation::RotUseT,
    ] {
        let g = kpoint_rhf_stress(
            &r.ks.cell,
            &r.ks.prep,
            &r.ks.mesh,
            &r.ks.hcfg,
            &r.ks.hk,
            KStressJk::RsGdf(r.src()),
            &r.scf,
            ExxDiv::None,
            &scfg(Some(m), zeta),
        )
        .expect("fit stress mutant");
        let moved_total = max_mat(&g.de_deps, &s.de_deps);
        eprintln!("  RS-GDF stress mutant {m:?}: total moves {moved_total:.2e}");
        assert!(moved_total > MUTANT_FLOOR, "{m:?} escaped: {moved_total:e}");
    }
    // The unrotated run LAST (kept from the time its KRsGdf fit read
    // uninitialised memory, #433/#437): a failure inside it must not hide the
    // rotated FD and mutant checks above, which do not depend on it.
    let off = fit_run(cell, &bs, n, SrColumnRotation::Off);
    assert_eq!(off.gdf.stats().sr_rotated_columns, 0);
    let (fo, so) = (force(&off), stress(&off));
    let (d_f, d_s) = (
        grad_miss(&f.grad, &fo.grad),
        max_mat(&s.de_deps, &so.de_deps),
    );
    eprintln!("rotated vs unrotated analytic: force {d_f:.2e}, stress {d_s:.2e}");
    assert!(d_f < ROT_VS_OFF_FIT_BAR && d_s < ROT_VS_OFF_FIT_BAR);
}

// ============================================================ (4) guards

/// A derivative call whose hcore config disagrees with the (rotated) energy
/// build is refused BY NAME also when an ECP is present, never silently run
/// on the unrotated walk (whose ECP part would be right and whose SR part
/// the screen-sized wrong one: invisible numerically at the default
/// screen). Both directions, force and stress; the consistent request runs
/// and keeps the Fock consistency guard (mismatch < 1e-8) with `V_ECP` in
/// the rebuilt `F`.
#[test]
#[ignore = "slow (~104 s release): two HI 1x1x1 k SCFs (rotated and unrotated) + 6 derivative calls"]
fn ecp_derivative_with_a_mismatched_rotation_config_is_refused() {
    let bs = rot_basis();
    let cell = hi_cell(&MOVED, &bs);
    let rot = run(cell.clone(), &bs, [1, 1, 1], on());
    let unrot = run(cell, &bs, [1, 1, 1], SrColumnRotation::Off);
    assert_eq!(rot.ks.hk.sr_rotated_columns, 3);
    let call = |r: &Run, hcfg: &PeriodicHcoreConfig| {
        let f = kpoint_rhf_gradient(
            &r.ks.cell,
            &r.ks.prep,
            &r.ks.mesh,
            hcfg,
            &r.ks.hk,
            KGradJk::Dense(&r.eri),
            &r.scf,
            ExxDiv::None,
            &gcfg(None, GRAD_NUCLEUS_EXPONENT),
        );
        let s = kpoint_rhf_stress(
            &r.ks.cell,
            &r.ks.prep,
            &r.ks.mesh,
            hcfg,
            &r.ks.hk,
            KStressJk::Dense(&r.eri),
            &r.scf,
            ExxDiv::None,
            &scfg(None, GRAD_NUCLEUS_EXPONENT),
        );
        (f, s)
    };
    let off = hcore_cfg(SrColumnRotation::Off);
    let (f, s) = call(&rot, &off);
    for (what, e) in [("force", f.err()), ("stress", s.err())] {
        let msg = e
            .unwrap_or_else(|| panic!("rotated build, config Off: {what} not refused"))
            .to_string();
        assert!(msg.contains("rotat"), "{what}: {msg}");
    }
    let (f, s) = call(&unrot, &hcore_cfg(on()));
    for (what, e) in [("force", f.err()), ("stress", s.err())] {
        let msg = e
            .unwrap_or_else(|| panic!("unrotated build, config On: {what} not refused"))
            .to_string();
        assert!(msg.contains("rotat"), "{what}: {msg}");
    }
    let (f, s) = call(&rot, &rot.ks.hcfg);
    let (f, s) = (f.expect("consistent force"), s.expect("consistent stress"));
    eprintln!(
        "consistent rotated ECP request: Fock mismatch force {:.1e}, stress {:.1e}",
        f.fock_mismatch, s.fock_mismatch
    );
    assert!(f.fock_mismatch < 1e-8 && s.fock_mismatch < 1e-8);
}

// ====================================== (5) regression: Boys table edge (#433)

/// The call pattern of the SR attraction walk on one I s,s pair at the walker's
/// image shifts, with a contracted I s shell (exponents 1.2, 0.45: min < 0.5).
/// Every block must be finite and tiny (true values <= 1e-12 for a Gaussian
/// nucleus of exponent 1e16 six or more Bohr away).
fn far_field_s_blocks_are_clean() -> Result<(), String> {
    use ferric_integrals::engine::Engine;
    use ferric_integrals::operator::Operator;
    use ferric_integrals::site_basis::SiteBasis;
    let mut bs = rot_basis();
    bs.shells
        .insert(53, vec![shell(0, &[1.2, 0.45], &[0.25, 0.55])]);
    let cell = hi_cell(&MOVED, &bs);
    let prep = prep_for(&cell, &bs);
    let ish = prep.nshells() - 1;
    let site = SiteBasis::new(
        &[
            [MOVED[0][0], MOVED[0][1], MOVED[0][2], 1e16],
            [MOVED[1][0], MOVED[1][1], MOVED[1][2], 1e16],
        ],
        0,
    )
    .map_err(|e| e.to_string())?;
    let mut eng = Engine::new_3center(
        ferric_integrals::operator::Operator::erfc(0.8),
        &prep,
        &site.prep,
        f64::MIN_POSITIVE,
    )
    .map_err(|e| e.to_string())?;
    let _ = Operator::coulomb;
    for k in 0..2 {
        for l in -1i32..=1 {
            for m in -1i32..=1 {
                for lz in -1i32..=1 {
                    let lv = [6.0 * l as f64, 0.0, 7.0 * lz as f64];
                    let mv = [6.0 * m as f64, 0.0, 7.0 * (lz - m) as f64];
                    let blk = eng
                        .compute_eri3_shifted(
                            &prep,
                            &site.prep,
                            site.site_shell[k],
                            ish,
                            ish,
                            [mv, [0.0; 3], lv],
                        )
                        .map_err(|e| e.to_string())?;
                    if let Some(b) = blk {
                        if b.iter().any(|x| !x.is_finite() || x.abs() > 1e-3) {
                            return Err(format!("k {k} l {lv:?} m {mv:?}: {:?}", b));
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

/// Helper of the reproducer below, not a protection by itself: in the default
/// allocator uninitialised memory is usually zero, and the check only catches
/// blocks that are non-finite or larger than 1e-3, so it would pass on an
/// uninitialised read unless the allocator fills memory. Run it only through
/// the perturbed re-exec.
#[test]
#[ignore = "helper of far_field_contracted_s_blocks_do_not_read_uninitialised_memory; only meaningful under MALLOC_PERTURB_ (regression guard for #433)"]
fn far_field_contracted_s_blocks_are_finite_in_the_default_allocator() {
    far_field_s_blocks_are_clean().expect("block check");
}

/// Regression test for #433 (fixed in #437: the libint2 Boys Chebyshev table
/// was read one row past its end at x == 117.0). Before the fix, with
/// `MALLOC_PERTURB_` set (glibc fills freshly allocated memory with a
/// pattern) the erfc 3-centre block of a CONTRACTED s pair at a far-field
/// shift returned ~1e288-1e301; in the default allocator the memory was
/// usually zero and the block ~1e-17, so energies were right most of the time
/// but an allocator-state change (a derivative engine created and destroyed
/// earlier in the process) shifted k-RHF energies by 1.2e-5 Ha on the HI
/// system with the s exponents (1.2, 0.45, 0.16). Measured boundary then: the
/// engine precision mattered (garbage at 2.2e-308 [the hcore walk's
/// `ERI3_ENGINE_PRECISION`], 1e-100, 1e-20; clean at 1e-14 and 1e-10).
/// The test re-executes this binary with `MALLOC_PERTURB_` 1, 170 and 2 (not
/// 255: glibc fills with `n ^ 0xff`, so 255 gives the default zero fill).
#[test]
fn far_field_contracted_s_blocks_do_not_read_uninitialised_memory() {
    let exe = std::env::current_exe().expect("current exe");
    for perturb in ["1", "170", "2"] {
        let out = std::process::Command::new(&exe)
            .args([
                "far_field_contracted_s_blocks_are_finite_in_the_default_allocator",
                "--exact",
                "--include-ignored",
                "--test-threads=1",
            ])
            .env("MALLOC_PERTURB_", perturb)
            .output()
            .expect("spawn");
        let stdout = String::from_utf8_lossy(&out.stdout);
        // A filter that matches nothing exits 0 too (a rename would pass).
        assert!(
            out.status.success() && stdout.contains("1 passed"),
            "MALLOC_PERTURB_={perturb} run failed:\n{stdout}"
        );
    }
}

// ============================================================ (4) triclinic

// (#421 A) The anchors above are orthorhombic diag(6, 6, 7). Untested
// combination before this section: a periodic ECP image sum (pair vectors,
// centre images, residue weights) and the rotated SR walks on a cell whose
// lattice rows are NOT along the Cartesian axes.
//
// ARTIFACT hypothesis (written before measuring). Correct: 1x1x1 equals the
// Gamma code to the orthorhombic bars; the 1x1x2 mesh equals the 2-cell
// supercell per cell; central FD of the rotated k energy equals the analytic
// force and ALL stress components. Defect: an ECP image sum or strain term
// that uses the lattice rows as if orthogonal (Cartesian vs lattice
// coordinates in a pair vector, a missing off-diagonal row of the strain
// map, d(a_i)/d(eps) applied to the wrong row) is invisible on diag(6, 6, 7)
// and shows on a skewed cell as an off-diagonal stress error. So the
// off-diagonal stress components must be non-negligible (a vacuity guard, not
// skew evidence: see the orthorhombic comparison test below). REFUTED part of
// this hypothesis: "a TRANSPOSED stress must miss the Gamma anchor". dE/de of a
// rotation-invariant energy is symmetric (measured miss 5e-12), so transposition
// is not a mutant; the antisymmetry check replaces it.
//
// WHAT EACH ANCHOR CAN SEE. At 1x1x1 the k side calls `periodic_ecp_strain_weighted`
// with `EcpWeights::Residue` (moduli [1,1,1]) and the Gamma side the same kernel
// with `EcpWeights::Gamma`, so "ECP parts 0 (bitwise)" is EXPECTED, not
// evidence about the kernel; the five ECP mutants act on the k side only and
// show the mutants are live, not that the anchor detects kernel defects. The
// 1x1x2 supercell test shares `ecp_strain_image` with the Gamma side and so
// independently checks only the k-specific part (residue lookup through the
// reciprocal lattice, moduli [1,1,2]). The ONLY independent check of the ECP
// strain kernel on the skewed cell is the 1x1x1 FD test.

const TRI_HI: [[f64; 3]; 3] = [[6.0, 0.0, 0.0], [1.0, 6.0, 0.0], [1.2, -0.8, 7.0]];

fn tri_lattice(n: usize) -> [[f64; 3]; 3] {
    let mut lat = TRI_HI;
    for v in lat[2].iter_mut() {
        *v *= n as f64;
    }
    lat
}

fn tri_shifts(n: usize) -> Vec<[f64; 3]> {
    (0..n).map(|m| TRI_HI[2].map(|v| v * m as f64)).collect()
}

fn tri_cell(pos: &[[f64; 3]], bs: &BasisSet) -> Cell {
    let dot = |a: &[f64; 3], b: &[f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    assert!(
        dot(&TRI_HI[0], &TRI_HI[1]).abs() > 1.0 && dot(&TRI_HI[0], &TRI_HI[2]).abs() > 1.0,
        "the triclinic fixture is not skewed"
    );
    ecp_cell(pos, &[[0.0; 3]], TRI_HI, bs)
}

fn tri_supercell(pos: &[[f64; 3]], n: usize, bs: &BasisSet) -> Cell {
    ecp_cell(pos, &tri_shifts(n), tri_lattice(n), bs)
}

fn transposed(m: &Mat3) -> Mat3 {
    let mut o = [[0.0; 3]; 3];
    for (i, row) in o.iter_mut().enumerate() {
        for (j, v) in row.iter_mut().enumerate() {
            *v = m[j][i];
        }
    }
    o
}

/// max |off-diagonal| of a stress.
fn offdiag_max(m: &Mat3) -> f64 {
    let mut w = 0.0_f64;
    for i in 0..3 {
        for j in 0..3 {
            if i != j {
                w = w.max(m[i][j].abs());
            }
        }
    }
    w
}

/// Triclinic bars: see the measured values in the test docs.
const TRI_ONE_POINT_ECP: f64 = 1e-12;
const TRI_ONE_POINT_TOTAL: f64 = 1e-10;
const TRI_SC_BAR_TOTAL: f64 = 3e-11;
/// The worst part is force `eri` (4.55e-13): bar 11x above it; the next are
/// stress vlr 5.9e-14 (85x) and force ecp 6.6e-15. `SC_BAR_SR` (1e-12, the
/// orthorhombic file's constant, unchanged) vs force vsr_basis 3.8e-15 and
/// stress vsr 1.8e-14: 260x / 56x.
const TRI_SC_BAR_PART: f64 = 5e-12;
/// The stress of the skewed cell must have off-diagonal weight at least this
/// (MEASURED total 2.5e-1, ECP part 7.3e-2), so the anchors are not diagonal-only.
const TRI_OFFDIAG_FLOOR: f64 = 1e-2;
/// Triclinic 1x1x1 FD bars. MEASURED force 7.5e-9, stress 3.3e-8 (central FD,
/// h = 1e-4, |S| up to 7.6e-1; the orthorhombic 1x1x3 file bars are 2e-8 /
/// 2e-7 on measured 3.3e-9 / 2.2e-8). 13x / 12x above; the smallest mutant
/// miss is 3.5e-1 (EcpUnitPairVector), > 1e6x the bars.
const TRI_FD_F_BAR: f64 = 1e-7;
const TRI_FD_S_BAR: f64 = 4e-7;
/// ECP-acting stress mutants must move the triclinic total by at least this.
const TRI_ECP_MUTANT_FLOOR: f64 = 1e-4;

/// Anchor (exactness, run first): 1x1x1 mesh == Gamma on the triclinic ECP
/// cell, rotation on in both; ECP-acting stress mutants and a transposed
/// stress must miss it.
#[test]
#[ignore = "slow (~7.5 min release, quiet box): triclinic rotated HI 1x1x1 k SCF + Gamma hcore/dense AFT/force/stress + 5 ECP stress mutants"]
fn triclinic_rotated_ecp_one_point_mesh_is_the_gamma_one() {
    let bs = rot_basis();
    let cell = tri_cell(&MOVED, &bs);
    let r = run(cell.clone(), &bs, [1, 1, 1], on());
    assert_eq!(r.ks.hk.sr_rotated_columns, 3);
    let g = gsys(cell, &bs, on());
    assert_eq!(g.hc.sr_rotated_columns, 3);
    let gscf = gamma_restricted(real_of(&r.scf.densities[0]));
    let zeta = GRAD_NUCLEUS_EXPONENT;
    let gf = gamma_force(&g, &gscf, zeta);
    let gs = gamma_stress(&g, &gscf, zeta);
    let kf = r.force(None, zeta);
    let ks = r.stress(None, zeta);
    let (f_ecp, f_tot) = (
        grad_miss(&kf.parts.ecp, &gf.parts.ecp),
        grad_miss(&kf.grad, &gf.grad),
    );
    let (s_ecp, s_tot) = (
        max_mat(&ks.parts.ecp, &gs.parts.ecp),
        max_mat(&ks.de_deps, &gs.de_deps),
    );
    let off = offdiag_max(&ks.de_deps);
    let off_ecp = offdiag_max(&ks.parts.ecp);
    let t_miss = max_mat(&transposed(&ks.de_deps), &gs.de_deps);
    eprintln!(
        "triclinic rotated 1x1x1 vs Gamma: force ecp {f_ecp:.2e} total {f_tot:.2e}; \
         stress ecp {s_ecp:.2e} total {s_tot:.2e}; |S| {:.2e}, offdiag {off:.2e} \
         (ECP part offdiag {off_ecp:.2e}); antisym {:.2e}; transposed-stress miss {t_miss:.2e}",
        mat_max(&ks.de_deps),
        ks.antisymmetry()
    );
    assert!(amax(&kf.parts.ecp) > 1e-3 && mat_max(&ks.parts.ecp) > 1e-3);
    assert!(f_ecp < TRI_ONE_POINT_ECP && s_ecp < TRI_ONE_POINT_ECP);
    assert!(f_tot < TRI_ONE_POINT_TOTAL && s_tot < TRI_ONE_POINT_TOTAL);
    assert!(off > TRI_OFFDIAG_FLOOR && off_ecp > TRI_OFFDIAG_FLOOR);
    // dE/de of a rotation-invariant energy is symmetric, so a transposed
    // stress is NOT a mutant (measured miss 5.0e-12, the anchor floor); the
    // symmetry itself is the physics check.
    assert!(
        ks.antisymmetry() < TRI_ONE_POINT_TOTAL,
        "{:e}",
        ks.antisymmetry()
    );
    assert!(t_miss < TRI_ONE_POINT_TOTAL, "{t_miss:e}");
    for m in [
        KStressMutation::NoEcp,
        KStressMutation::EcpNoCentre,
        KStressMutation::EcpCentreSign,
        KStressMutation::EcpNoKetImage,
        KStressMutation::EcpUnitPairVector,
    ] {
        let miss = max_mat(&r.stress(Some(m), zeta).de_deps, &gs.de_deps);
        eprintln!("  triclinic 1x1x1 stress mutant {m:?}: miss {miss:.2e}");
        assert!(miss > TRI_ECP_MUTANT_FLOOR, "{m:?} escaped: {miss:e}");
    }
}

/// PHYSICS: the Gamma-centred 1x1x2 mesh of the triclinic ECP cell IS the
/// 2-cell supercell (rows a1, a2, 2 a3) per cell, rotation on both sides.
/// 1x1x3 (non-TRIM) was not run: the orthorhombic 1x1x3 supercell anchor
/// takes 24 min.
#[test]
#[ignore = "slow (~10 min release, quiet box): triclinic rotated HI 1x1x2 k SCF + 2-cell Gamma supercell hcore/dense AFT/force/stress"]
fn triclinic_rotated_ecp_1x1x2_force_and_stress_are_the_supercell_ones_per_cell() {
    let bs = rot_basis();
    let r = run(tri_cell(&MOVED, &bs), &bs, [1, 1, 2], on());
    assert_eq!(r.ks.hk.sr_rotated_columns, 3);
    let g = gsys(tri_supercell(&MOVED, 2, &bs), &bs, on());
    assert_eq!(g.hc.sr_rotated_columns, 6);
    let gscf = gamma_restricted(unfold_z(&r.ks.mesh, &r.scf.densities));
    let zeta = GRAD_NUCLEUS_EXPONENT;
    let gf = gamma_force(&g, &gscf, zeta);
    let gs = gamma_stress(&g, &gscf, zeta);
    let kf = r.force(None, zeta);
    let ks = r.stress(None, zeta);
    let half = |m: &Mat3| mat_scaled(m, 0.5);
    let (kp, gp) = (&kf.parts, &gf.parts);
    let eri_k = &kp.coulomb + &kp.exchange;
    let force = [
        (
            "kinetic",
            sc_miss(&kp.kinetic, &gp.kinetic),
            TRI_SC_BAR_PART,
        ),
        ("ecp", sc_miss(&kp.ecp, &gp.ecp), TRI_SC_BAR_PART),
        (
            "vsr_basis",
            sc_miss(&kp.vsr_basis, &gp.vsr_basis),
            SC_BAR_SR,
        ),
        ("eri", sc_miss(&eri_k, &gp.eri), TRI_SC_BAR_PART),
        ("TOTAL", sc_miss(&kf.grad, &gf.grad), TRI_SC_BAR_TOTAL),
    ];
    let (sk, sg) = (&ks.parts, &gs.parts);
    let stress = [
        (
            "kinetic",
            max_mat(&sk.kinetic, &half(&sg.kinetic)),
            TRI_SC_BAR_PART,
        ),
        ("ecp", max_mat(&sk.ecp, &half(&sg.ecp)), TRI_SC_BAR_PART),
        ("vsr", max_mat(&sk.vsr, &half(&sg.vsr)), SC_BAR_SR),
        ("vlr", max_mat(&sk.vlr, &half(&sg.vlr)), TRI_SC_BAR_PART),
        (
            "TOTAL",
            max_mat(&ks.de_deps, &half(&gs.de_deps)),
            TRI_SC_BAR_TOTAL,
        ),
    ];
    for (kind, parts) in [("force", &force[..]), ("stress", &stress[..])] {
        for (name, miss, bar) in parts {
            eprintln!("  triclinic rotated+ECP 1x1x2 {kind} {name}: {miss:.2e} (bar {bar:.0e})");
            assert!(miss < bar, "{kind} {name} {miss:e} >= {bar:e}");
        }
    }
    let off = offdiag_max(&ks.de_deps);
    eprintln!(
        "  triclinic 1x1x2 offdiag |S| {off:.2e}; antisym {:.2e}",
        ks.antisymmetry()
    );
    assert!(off > TRI_OFFDIAG_FLOOR);
    assert!(ks.antisymmetry() < TRI_SC_BAR_TOTAL);
    assert!(
        amax(&kp.ecp) > 1e-3 && mat_max(&sk.ecp) > 1e-3,
        "vacuous ECP"
    );
}

/// PHYSICS (independent of every derivative walk and of the Gamma code):
/// analytic force (I x, H z) and stress (zz, xy, yz, xz: three off-diagonal
/// components of a skewed cell) of the triclinic rotated ECP cell at the
/// 1x1x1 mesh vs central FD of the ROTATED k energy (rotation on in every
/// displaced / strained build). Rotation and ECP stress mutants must move
/// the total.
#[test]
#[ignore = "slow (~6 min release, quiet box): 1 + 12 rotated triclinic 1x1x1 k pipelines, 1 analytic force + stress, 2 mutant stresses"]
fn triclinic_rotated_ecp_force_and_stress_match_fd_of_the_rotated_k_energy() {
    let bs = rot_basis();
    let n = [1, 1, 1];
    let cell = tri_cell(&MOVED, &bs);
    let forces = [(1usize, 0usize), (0, 2)];
    let strains = [(2usize, 2usize), (0, 1), (1, 2), (0, 2)];
    let mut fd_f = Vec::new();
    for &(a, x) in &forces {
        let ep = k_energy(&tri_cell(&moved(&MOVED, a, x, FD_H), &bs), &bs, n, on());
        let em = k_energy(&tri_cell(&moved(&MOVED, a, x, -FD_H), &bs), &bs, n, on());
        fd_f.push((ep - em) / (2.0 * FD_H));
    }
    let mut fd_s = Vec::new();
    for &(i, j) in &strains {
        let ep = k_energy(&eps_cell(&cell, i, j, FD_H), &bs, n, on());
        let em = k_energy(&eps_cell(&cell, i, j, -FD_H), &bs, n, on());
        fd_s.push((ep - em) / (2.0 * FD_H));
    }
    let r = run(cell, &bs, n, on());
    assert_eq!(r.ks.hk.sr_rotated_columns, 3);
    let zeta = GRAD_NUCLEUS_EXPONENT;
    let f = r.force(None, zeta);
    let s = r.stress(None, zeta);
    let miss_f = |g: &KGradient| {
        forces.iter().zip(&fd_f).fold(0.0_f64, |w, (&(a, x), fd)| {
            w.max((g.grad[(a, x)] - fd).abs())
        })
    };
    let miss_s = |g: &KStress| {
        strains.iter().zip(&fd_s).fold(0.0_f64, |w, (&(i, j), fd)| {
            w.max((g.de_deps[i][j] - fd).abs())
        })
    };
    let (wf, ws) = (miss_f(&f), miss_s(&s));
    eprintln!("triclinic rotated+ECP 1x1x1 vs FD: force {wf:.2e}, stress {ws:.2e}");
    for (&(i, j), fd) in strains.iter().zip(&fd_s) {
        eprintln!(
            "  stress [{i}][{j}] analytic {:+.8e} FD {fd:+.8e}",
            s.de_deps[i][j]
        );
    }
    assert!(wf.is_finite() && ws.is_finite());
    assert!(wf < TRI_FD_F_BAR && ws < TRI_FD_S_BAR);
    for m in [
        KStressMutation::RotDropT,
        KStressMutation::EcpUnitPairVector,
    ] {
        let w = miss_s(&r.stress(Some(m), zeta));
        eprintln!("  stress mutant {m:?}: FD miss {w:.2e}");
        assert!(w > MUTANT_FLOOR, "{m:?} escaped the FD: {w:e}");
    }
}

/// MEASUREMENT (no skew claim asserted beyond vacuity): the same MOVED
/// positions in the orthorhombic and the triclinic cell at 1x1x1. MOVED tilts
/// the H-I bond, so the orthorhombic cell already carries off-diagonal stress;
/// this reports its size next to the triclinic one and the off-diagonal miss
/// of three ECP stress mutants (vs the unmutated analytic stress, which the FD
/// test pins on the triclinic cell) on both. A mutant that is visible only on
/// the skewed cell would have a ~0 orthorhombic off-diagonal miss and a large
/// triclinic one; see the printed table.
#[test]
#[ignore = "slow (~10 min release, quiet box): two HI 1x1x1 k SCFs + 2 stresses + 6 ECP mutant stresses"]
fn ecp_stress_offdiagonal_size_and_mutant_misses_orthorhombic_vs_triclinic() {
    let bs = rot_basis();
    let zeta = GRAD_NUCLEUS_EXPONENT;
    for (name, cell) in [
        ("orthorhombic", hi_cell(&MOVED, &bs)),
        ("triclinic", tri_cell(&MOVED, &bs)),
    ] {
        let r = run(cell, &bs, [1, 1, 1], on());
        let s0 = r.stress(None, zeta);
        eprintln!(
            "{name}: offdiag total {:.2e}, ECP part {:.2e} (max |S| {:.2e})",
            offdiag_max(&s0.de_deps),
            offdiag_max(&s0.parts.ecp),
            mat_max(&s0.de_deps)
        );
        assert!(offdiag_max(&s0.parts.ecp) > 1e-3, "{name}: vacuous");
        for m in [
            KStressMutation::EcpNoKetImage,
            KStressMutation::EcpCentreSign,
            KStressMutation::EcpUnitPairVector,
        ] {
            let sm = r.stress(Some(m), zeta);
            let mut off = 0.0_f64;
            let mut diag = 0.0_f64;
            for i in 0..3 {
                for j in 0..3 {
                    let d = (sm.de_deps[i][j] - s0.de_deps[i][j]).abs();
                    if i == j {
                        diag = diag.max(d);
                    } else {
                        off = off.max(d);
                    }
                }
            }
            eprintln!("  {name} mutant {m:?}: offdiag miss {off:.2e}, diagonal miss {diag:.2e}");
        }
    }
}
