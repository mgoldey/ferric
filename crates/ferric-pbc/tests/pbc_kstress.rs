//! k-point RHF / UHF analytic stress (`ferric_pbc::kstress`), dense AFT.
//!
//! What is anchored against what (every anchor at ONE fixed density, so SCF
//! convergence does not enter the comparison unless stated):
//! (0) KERNEL: the residue-resolved pair-FT strain kernel, summed over the
//!     residues, is the Gamma strain kernel at the same vectors, and EACH
//!     residue's `dP_r` is the finite difference of the independent residue
//!     pair FT ([`pair_ft_residues_chunked`]) on strained cells with `K`
//!     held at fixed Miller index + fixed fractional `q`.
//! (1) HCORE GATE (before any ERI term is trusted): the one-electron + `E_nn`
//!     strain at a FIXED `D(k)` against central FD of
//!     `(1/N_k) Σ_k tr[D h(k)] + E_nn` on strained cells with the mesh rebuilt
//!     from the same integer numerators.
//! (2) 1×1×1 mesh ≡ the Gamma stress, term by term.
//!
//! ARTIFACT vs PHYSICS hypotheses are stated next to each test.

mod common;

use common::*;
use ferric_core::basis::{BasisSet, Shell};
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_pbc::dense_aft::{DenseAftEri, ExxDiv, DEFAULT_DENSE_AFT_MAX_BYTES};
use ferric_pbc::dft::PeriodicGridConfig;
use ferric_pbc::hcore::kpoint::{periodic_hcore_kpts, PeriodicHcoreK};
use ferric_pbc::hcore::{periodic_hcore, PeriodicHcore, PeriodicHcoreConfig};
use ferric_pbc::kdense_aft::{KDenseAftConfig, KDenseAftEri};
use ferric_pbc::kdft::{solve_krks, KRksConfig};
use ferric_pbc::kgrad::KRsGdfGradSource;
use ferric_pbc::kpts::{KPointMesh, MeshCentring};
use ferric_pbc::kscf::{solve_krhf_injected, KPointInjection, KPointJk, KScfConfig, KScfResult};
use ferric_pbc::kstress::{
    kpoint_hcore_stress_fixed_d, kpoint_rhf_stress, kpoint_uhf_stress, KStress, KStressConfig,
    KStressJk, KStressMutation, FOCK_CONSISTENCY_BAR,
};
use ferric_pbc::kuscf::{solve_kuhf_injected, KUScfResult};
use ferric_pbc::lattice::Cell;
use ferric_pbc::pair_ft::residues::{pair_ft_residues_chunked, pair_ft_strain_residues_chunked};
use ferric_pbc::pair_ft::{pair_ft_strain_chunked, PairFtStrainTerms};
use ferric_pbc::rsgdf::kpoint::{KRsGdf, KRsGdfConfig};
use ferric_pbc::rsgdf::RsGdfConfig;
use ferric_pbc::sr_rotation::SrColumnRotation;
use ferric_pbc::stress::{gamma_rhf_stress, gamma_uhf_stress, GammaStress, GammaStressConfig};
use ferric_scf::result::{ScfExit, ScfResult, Spin};
use ndarray::{Array2, Array3};
use num_complex::Complex64;
use std::collections::HashMap;

type Mat3 = [[f64; 3]; 3];

const OMEGA: f64 = 0.8;
const HCORE_PRECISION: f64 = 1e-14;
/// Dense-AFT precision of the fixed-density anchors (see `pbc_kgrad.rs`).
const ANCHOR_PRECISION: f64 = 1e-12;
const AMPLE: usize = 1 << 31;
const FD_H: f64 = 1e-4;
const GAMMA_BAR: f64 = 1e-12;
/// Measured h = 1e-4 floor of the hcore FD 6.7e-9 (h² truncation); the smallest
/// mutant miss is 7e-2, so the bar sits between at 5e-8.
const HCORE_FD_BAR: f64 = 5e-8;
const H2_ATOMS: [[f64; 3]; 2] = [[0.3, 0.2, 0.1], [0.35, 0.12, 1.5]];
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

/// Rotation pinned OFF so these anchors do not depend on the default: the
/// column-rotated stress is anchored in `pbc_sr_rotation_kderiv.rs`.
fn hcore_cfg() -> PeriodicHcoreConfig {
    PeriodicHcoreConfig {
        precision: HCORE_PRECISION,
        ..PeriodicHcoreConfig::with_omega(OMEGA)
    }
    .with_sr_column_rotation(SrColumnRotation::Off)
}

fn kscf_cfg() -> KScfConfig {
    KScfConfig {
        energy_conv: 1e-13,
        grad_conv: 1e-10,
        max_iter: 400,
        ..Default::default()
    }
}

fn scfg(mutation: Option<KStressMutation>) -> KStressConfig {
    KStressConfig {
        budget_bytes: Some(AMPLE),
        mutation,
        ..Default::default()
    }
}

fn cell_at(pos: &[[f64; 3]], lattice: [[f64; 3]; 3], mult: usize) -> Cell {
    let mut mol: Molecule = hydrogens(pos);
    mol.multiplicity = mult;
    Cell::new(mol, lattice).expect("cell")
}

fn h2_cell_g() -> Cell {
    cell_at(&H2_ATOMS, cubic(4.0), 1)
}

fn eps_at(i: usize, j: usize, h: f64) -> Mat3 {
    let mut e = [[0.0; 3]; 3];
    e[i][j] = h;
    e
}

fn strained(cell: &Cell, i: usize, j: usize, h: f64) -> Cell {
    cell.strained(&eps_at(i, j, h)).expect("strained cell")
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

// ---------------------------------------------------------------- setups

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
    KSys {
        cell,
        prep,
        mesh,
        hk,
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

fn real_of(d: &Array2<Complex64>) -> Array2<f64> {
    let im = d.iter().fold(0.0_f64, |m, z| m.max(z.im.abs()));
    assert!(im < 1e-12, "Gamma density has an imaginary part {im:e}");
    d.mapv(|z| z.re)
}

fn gamma_restricted(d: Array2<f64>) -> ScfResult {
    let n = d.nrows();
    ScfResult {
        guess_basis: None,
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
    hc: PeriodicHcore,
    eri: DenseAftEri,
}

fn gsys(cell: Cell, bs: &BasisSet) -> GSys {
    let prep = prep_for(&cell, bs);
    let hc = periodic_hcore(&cell, &prep, &hcore_cfg()).expect("gamma hcore");
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

// ============================================================ (0) kernel

/// `K(ε)` of the reference vectors `ks` (Bohr⁻¹) on the strained cell: the
/// same fractional coordinates in the strained reciprocal lattice.
fn k_on(cell_ref: &Cell, cell_s: &Cell, ks: &[[f64; 3]]) -> Vec<[f64; 3]> {
    let a = cell_ref.lattice();
    let bs = cell_s.reciprocal();
    let tp = 2.0 * std::f64::consts::PI;
    ks.iter()
        .map(|k| {
            let f: Vec<f64> = (0..3)
                .map(|i| (k[0] * a[i][0] + k[1] * a[i][1] + k[2] * a[i][2]) / tp)
                .collect();
            let mut o = [0.0; 3];
            for d in 0..3 {
                o[d] = f[0] * bs[0][d] + f[1] * bs[1][d] + f[2] * bs[2][d];
            }
            o
        })
        .collect()
}

/// Reference `K = G + q` vectors: Miller shells shifted by a non-trivial
/// fractional `q` (so `K` is NOT a reciprocal-lattice vector).
fn shifted_ks(cell: &Cell, q_frac: [f64; 3], take: usize) -> Vec<[f64; 3]> {
    let b = cell.reciprocal();
    let idx: Vec<[i64; 3]> = cell
        .gvector_indices(3.5)
        .unwrap()
        .into_iter()
        .filter(|n| *n != [0, 0, 0])
        .take(take)
        .collect();
    idx.iter()
        .map(|n| {
            let mut k = [0.0; 3];
            for i in 0..3 {
                let f = n[i] as f64 + q_frac[i];
                for d in 0..3 {
                    k[d] += f * b[i][d];
                }
            }
            k
        })
        .collect()
}

fn collect_strain(
    cell: &Cell,
    prep: &PreparedBasis,
    ks: &[[f64; 3]],
    moduli: [usize; 3],
    thresh: f64,
    terms: PairFtStrainTerms,
) -> (Vec<Array3<Complex64>>, Vec<[Array3<Complex64>; 9]>) {
    let mut got = None;
    pair_ft_strain_residues_chunked(
        cell,
        prep,
        ks,
        moduli,
        thresh,
        1 << 30,
        0,
        terms,
        |_, _, p, dp| {
            got = Some((p.to_vec(), dp.to_vec()));
            Ok(())
        },
    )
    .unwrap();
    got.expect("one chunk")
}

/// PHYSICS: `P^{k}(K) = Σ_r e^{ik·t_r} P_r(K)` with `P_r` a pure function of
/// `K` and the image set, and `dP_r/dε` its strain derivative at fixed Miller
/// index, so (a) `Σ_r P_r`, `Σ_r dP_r` are the Gamma kernel's `P`, `dP` at the
/// same `K` and (b) each `dP_r` is the FD of the independent residue FT.
/// ARTIFACT: a wrong bucket map (e.g. residues swapped between images) leaves
/// (a) intact (a sum over all residues) but breaks (b); a bucketed copy that
/// drops a term breaks both. (b) is therefore the sharp check.
#[test]
fn residue_strain_kernel_is_gamma_kernel_summed_and_fd_per_residue() {
    let cell = triclinic_cell();
    let bs = sp_basis_h();
    let prep = prep_for(&cell, &bs);
    let ks = shifted_ks(&cell, [0.31, 0.17, 1.0 / 3.0], 8);
    let thresh = 1e-15;
    let moduli = [1, 2, 3];
    let nr = 6;
    let (p, dp) = collect_strain(&cell, &prep, &ks, moduli, thresh, PairFtStrainTerms::ALL);
    assert_eq!(p.len(), nr);
    // (a) sum over residues = Gamma kernel.
    let mut gp = None;
    pair_ft_strain_chunked(
        &cell,
        &prep,
        &ks,
        thresh,
        1 << 30,
        0,
        PairFtStrainTerms::ALL,
        |_, _, p, dp| {
            gp = Some((p.clone(), dp.to_vec()));
            Ok(())
        },
    )
    .unwrap();
    let (gp, gdp) = gp.unwrap();
    let mut sum_p = Array3::<Complex64>::zeros(gp.dim());
    for r in &p {
        sum_p += r;
    }
    let ea = (&sum_p - &gp).iter().fold(0.0_f64, |m, z| m.max(z.norm()));
    let mut ed = 0.0_f64;
    for k in 0..9 {
        let mut s = Array3::<Complex64>::zeros(gp.dim());
        for r in &dp {
            s += &r[k];
        }
        ed = ed.max((&s - &gdp[k]).iter().fold(0.0_f64, |m, z| m.max(z.norm())));
    }
    eprintln!("residue strain kernel: |ΣP_r − P_gamma| {ea:.2e}, |ΣdP_r − dP_gamma| {ed:.2e}");
    assert!(ea < 1e-13 && ed < 1e-12, "{ea:e} {ed:e}");
    // (b) per residue vs FD of the independent residue FT.
    let (_, dp_nc) = collect_strain(
        &cell,
        &prep,
        &ks,
        moduli,
        thresh,
        PairFtStrainTerms {
            centres: false,
            g_shape: true,
        },
    );
    let (_, dp_ng) = collect_strain(
        &cell,
        &prep,
        &ks,
        moduli,
        thresh,
        PairFtStrainTerms {
            centres: true,
            g_shape: false,
        },
    );
    let h = 1e-5;
    let n = prep.nbasis();
    let (mut worst, mut fmax) = (0.0_f64, 0.0_f64);
    let (mut w_nc, mut w_ng, mut w_swap) = (0.0_f64, 0.0_f64, 0.0_f64);
    for (a, b) in ALL9 {
        let fd_p = |s: f64| {
            let c = strained(&cell, a, b, s * h);
            let pr = prep_for(&c, &bs);
            let kk = k_on(&cell, &c, &ks);
            let mut out = None;
            pair_ft_residues_chunked(&c, &pr, &kk, moduli, thresh, 1 << 30, 0, |_, _, q| {
                out = Some(q.to_vec());
                Ok(())
            })
            .unwrap();
            out.unwrap()
        };
        let (pp, pm) = (fd_p(1.0), fd_p(-1.0));
        let k = 3 * a + b;
        for r in 0..nr {
            for m in 0..n {
                for q in 0..n {
                    for g in 0..ks.len() {
                        let f = (pp[r][[m, q, g]] - pm[r][[m, q, g]]) / (2.0 * h);
                        fmax = fmax.max(f.norm());
                        worst = worst.max((dp[r][k][[m, q, g]] - f).norm());
                        w_nc = w_nc.max((dp_nc[r][k][[m, q, g]] - f).norm());
                        w_ng = w_ng.max((dp_ng[r][k][[m, q, g]] - f).norm());
                        // MUTANT: residue buckets rotated by one.
                        w_swap = w_swap.max((dp[(r + 1) % nr][k][[m, q, g]] - f).norm());
                    }
                }
            }
        }
    }
    eprintln!("residue strain kernel vs FD per residue: {worst:.2e} (|FD| max {fmax:.2e})");
    assert!(fmax > 1e-3, "the strain must change P_r");
    eprintln!(
        "  kernel mutants: no centres {w_nc:.2e}, no G shape {w_ng:.2e}, rotated buckets {w_swap:.2e}"
    );
    assert!(worst < 1e-7 * fmax.max(1.0), "{worst:e}");
    for (tag, w) in [
        ("FtNoCentres", w_nc),
        ("GUnstrained", w_ng),
        ("rotated buckets", w_swap),
    ] {
        assert!(w > 1e-3 * fmax, "{tag} escaped the per-residue FD: {w:e}");
    }
}

// ============================================================ (1) hcore gate

/// `(1/N_k) Σ_k Re tr[D h(k)] + E_nn` on a (strained) cell, `D(k)` FIXED.
fn e_hcore_fixed_d(cell: &Cell, bs: &BasisSet, n: [usize; 3], dk: &[Array2<Complex64>]) -> f64 {
    let prep = prep_for(cell, bs);
    let mesh = KPointMesh::gamma_centred(cell, n).unwrap();
    let hk = periodic_hcore_kpts(cell, &prep, &mesh, &hcore_cfg()).unwrap();
    let nk = mesh.nk();
    let nao = prep.nbasis();
    let mut e = 0.0;
    for (d, h) in dk.iter().zip(&hk.h) {
        for m in 0..nao {
            for q in 0..nao {
                e += (d[(m, q)] * h[(q, m)]).re;
            }
        }
    }
    e / nk as f64 + hk.enn
}

fn fd_hcore(cell: &Cell, bs: &BasisSet, n: [usize; 3], dk: &[Array2<Complex64>]) -> Mat3 {
    let mut out = [[0.0; 3]; 3];
    for (i, j) in ALL9 {
        let ep = e_hcore_fixed_d(&strained(cell, i, j, FD_H), bs, n, dk);
        let em = e_hcore_fixed_d(&strained(cell, i, j, -FD_H), bs, n, dk);
        out[i][j] = (ep - em) / (2.0 * FD_H);
    }
    out
}

/// PHYSICS: with `D(k)` fixed and the mesh at fixed integer numerators, the
/// strain derivative of `(1/N_k) Σ tr[D h(k)] + E_nn` is kinetic + V_SR + V_LR
/// + c0 (volume AND `c0 Z_tot Σ D dS`) + Ewald. ARTIFACT: a missing/wrong
/// phase weight (NoPhase), a missing 1/N_k, or a pair vector without its
/// image shift (NoSrImages) all leave 1×1×1 intact but move this by O(1).
#[test]
fn hcore_strain_at_fixed_density_matches_fd_h2_1x1x3() {
    let bs = pyscf_sto3g_h();
    let n = [1, 1, 3];
    let ks = ksys(h2_cell_g(), &bs, n);
    let eri = kdense(&ks, ANCHOR_PRECISION);
    let scf = krhf(&ks, Box::new(eri.jk_builder_with_madelung(0.0)));
    let an = kpoint_hcore_stress_fixed_d(
        &ks.cell,
        &ks.prep,
        &ks.mesh,
        &hcore_cfg(),
        &ks.hk,
        &scf.densities,
        &scfg(None),
    )
    .unwrap();
    let fd = fd_hcore(&ks.cell, &bs, n, &scf.densities);
    let tot = an.hcore_total();
    let miss = max_mat(&tot, &fd);
    eprintln!("H2 1x1x3 hcore fixed-D: |analytic − FD| {miss:.2e}");
    for a in 0..3 {
        eprintln!(
            "  [{a}] analytic {:+.8e} {:+.8e} {:+.8e} | FD {:+.8e} {:+.8e} {:+.8e}",
            tot[a][0], tot[a][1], tot[a][2], fd[a][0], fd[a][1], fd[a][2]
        );
    }
    assert!(miss < HCORE_FD_BAR, "hcore FD miss {miss:e}");
    for m in [
        KStressMutation::NoPhase,
        KStressMutation::NoInvNk,
        KStressMutation::NoSrImages,
        KStressMutation::NoC0Volume,
        KStressMutation::NoVolume,
        KStressMutation::NoEwaldLr,
        KStressMutation::NoEwaldBg,
        KStressMutation::FtNoCentres,
        KStressMutation::GUnstrained,
    ] {
        let a = kpoint_hcore_stress_fixed_d(
            &ks.cell,
            &ks.prep,
            &ks.mesh,
            &hcore_cfg(),
            &ks.hk,
            &scf.densities,
            &scfg(Some(m)),
        )
        .unwrap();
        let t = a.hcore_total();
        let mm = max_mat(&t, &fd);
        // Off-diagonal miss (the volume-only mutants are diagonal by construction).
        let mut off = 0.0_f64;
        for i in 0..3 {
            for j in 0..3 {
                if i != j {
                    off = off.max((t[i][j] - fd[i][j]).abs());
                }
            }
        }
        eprintln!("  hcore mutant {m:?}: miss {mm:.2e} (off-diagonal {off:.2e})");
        assert!(mm > 1e-4, "{m:?} not caught by the hcore gate ({mm:e})");
    }
}

// ============================================================ (2) 1x1x1 = Gamma

/// PHYSICS: at 1×1×1 every phase is 1, the residue is the only bucket, and
/// the k code is the Gamma code over the full sphere instead of the half
/// sphere. ARTIFACT: independent constructions (real half-sphere vs complex
/// full-sphere residue kernel; image-summed vs phase-folded weights), so
/// agreement is not an identity of one code path.
#[test]
fn one_point_mesh_hcore_parts_are_the_gamma_parts() {
    // (The triclinic s+p cell is covered by the `#[ignore]`d full 1x1x1 anchor.)
    for (name, cell, bs) in [("H2", h2_cell_g(), pyscf_sto3g_h())] {
        let ks = ksys(cell.clone(), &bs, [1, 1, 1]);
        let eri = kdense(&ks, ANCHOR_PRECISION);
        let scf = krhf(&ks, Box::new(eri.jk_builder_with_madelung(0.0)));
        let g = gsys(cell, &bs);
        let gscf = gamma_restricted(real_of(&scf.densities[0]));
        let gs = gamma_rhf_stress(
            &g.cell,
            &g.prep,
            &hcore_cfg(),
            &g.hc,
            &g.eri,
            &gscf,
            ExxDiv::None,
            &GammaStressConfig {
                budget_bytes: Some(AMPLE),
                ..Default::default()
            },
        )
        .unwrap();
        let an = kpoint_hcore_stress_fixed_d(
            &ks.cell,
            &ks.prep,
            &ks.mesh,
            &hcore_cfg(),
            &ks.hk,
            &scf.densities,
            &scfg(None),
        )
        .unwrap();
        let pairs: [(&str, &Mat3, &Mat3); 7] = [
            ("kinetic", &an.kinetic, &gs.parts.kinetic),
            ("vsr", &an.vsr, &gs.parts.vsr),
            ("vlr", &an.vlr, &gs.parts.vlr),
            ("c0_volume", &an.c0_volume, &gs.parts.c0_volume),
            ("nn_sr", &an.nn_sr, &gs.parts.nn_sr),
            ("nn_lr", &an.nn_lr, &gs.parts.nn_lr),
            ("nn_background", &an.nn_background, &gs.parts.nn_background),
        ];
        for (tag, a, b) in pairs {
            let d = max_mat(a, b);
            eprintln!("  {name} 1x1x1 {tag}: |k − Gamma| {d:.2e}");
            assert!(d < 1e-10, "{name} {tag}: {d:e}");
        }
    }
}

// ====================================================== full-stress fixtures

const H3_ATOMS: [[f64; 3]; 3] = [[0.3, 0.2, 0.1], [0.35, 0.12, 1.5], [1.6, 0.9, 0.7]];
const H3_A: f64 = 4.5;
const EXX: [ExxDiv; 2] = [ExxDiv::None, ExxDiv::Ewald];
const MUTANT_BAR: f64 = 1e-4;
const IDENTITY_BAR: f64 = 1e-12;
const ANTISYM_BAR: f64 = 1e-10;

fn h3_cell() -> Cell {
    cell_at(&H3_ATOMS, cubic(H3_A), 2)
}

/// Explicit diag(n) supercell (copy c = (m0, m1, m2), m0 outer) with
/// multiplicity `mult`.
fn supercell(cell: &Cell, n: [usize; 3], mult: usize) -> Cell {
    let a = *cell.lattice();
    let mut atoms = Vec::new();
    for m0 in 0..n[0] {
        for m1 in 0..n[1] {
            for m2 in 0..n[2] {
                let t: Vec<f64> = (0..3)
                    .map(|d| m0 as f64 * a[0][d] + m1 as f64 * a[1][d] + m2 as f64 * a[2][d])
                    .collect();
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
    cell_at(&atoms, lat, mult)
}

/// The k density unfolded to the explicit supercell:
/// `D_sc[(c,m),(c',n)] = (1/N_k) Σ_k e^{ik·(T_c − T_c')} D(k)_mn` (real).
fn unfold(mesh: &KPointMesh, dk: &[Array2<Complex64>]) -> Array2<f64> {
    let n = mesh.n();
    let nao = dk[0].nrows();
    let nk = mesh.nk();
    let copies: Vec<[i64; 3]> = (0..n[0])
        .flat_map(|m0| {
            (0..n[1]).flat_map(move |m1| (0..n[2]).map(move |m2| [m0 as i64, m1 as i64, m2 as i64]))
        })
        .collect();
    let nc = copies.len();
    let mut out = Array2::<f64>::zeros((nc * nao, nc * nao));
    let mut max_im = 0.0_f64;
    for (c, tc) in copies.iter().enumerate() {
        for (c2, tc2) in copies.iter().enumerate() {
            let nl = [tc[0] - tc2[0], tc[1] - tc2[1], tc[2] - tc2[2]];
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

fn gamma_unrestricted(da: Array2<f64>, db: Array2<f64>) -> ScfResult {
    let n = da.nrows();
    ScfResult {
        guess_basis: None,
        spin: Spin::Unrestricted,
        energy: 0.0,
        density_total: &da + &db,
        density_alpha: da,
        density_beta: Some(db),
        mos_alpha: Array2::eye(n),
        mos_beta: Some(Array2::eye(n)),
        eps_alpha: vec![0.0; n],
        eps_beta: Some(vec![0.0; n]),
        fock_alpha: Array2::zeros((n, n)),
        fock_beta: Some(Array2::zeros((n, n))),
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

fn exx_idx(e: ExxDiv) -> usize {
    match e {
        ExxDiv::None => 0,
        ExxDiv::Ewald => 1,
    }
}

/// The k-RHF SCF of each exxdiv `[none, ewald]` (the stress guards that the
/// SCF's exxdiv is the one it is asked for).
fn scf_pair(ks: &KSys, eri: &KDenseAftEri) -> [KScfResult; 2] {
    let vm = ks.mesh.madelung(&ks.cell).unwrap();
    [
        krhf(ks, Box::new(eri.jk_builder_with_madelung(0.0))),
        krhf(ks, Box::new(eri.jk_builder_with_madelung(vm))),
    ]
}

fn uscf_pair(ks: &KSys, eri: &KDenseAftEri, na: usize, nb: usize) -> [KUScfResult; 2] {
    let vm = ks.mesh.madelung(&ks.cell).unwrap();
    [
        kuhf(ks, Box::new(eri.jk_builder_with_madelung(0.0)), na, nb),
        kuhf(ks, Box::new(eri.jk_builder_with_madelung(vm)), na, nb),
    ]
}

fn kuhf(ks: &KSys, jk: Box<dyn KPointJk + '_>, na: usize, nb: usize) -> KUScfResult {
    let inj = KPointInjection {
        s: ks.hk.s.clone(),
        h: ks.hk.h.clone(),
        vnn: ks.hk.enn,
        jk,
    };
    let r = solve_kuhf_injected(&ks.cell, &ks.mesh, &kscf_cfg(), inj, na, nb).expect("k-UHF");
    assert!(r.converged, "k-UHF not converged ({} it)", r.iterations);
    r
}

fn ks_rhf(
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

fn ks_uhf(
    ks: &KSys,
    eri: &KDenseAftEri,
    scf: &KUScfResult,
    exx: ExxDiv,
    m: Option<KStressMutation>,
) -> KStress {
    kpoint_uhf_stress(
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
    .expect("kpoint_uhf_stress")
}

fn gamma_stress(g: &GSys, scf: &ScfResult, exx: ExxDiv) -> GammaStress {
    let cfg = GammaStressConfig {
        budget_bytes: Some(AMPLE),
        ..Default::default()
    };
    if scf.spin == Spin::Restricted {
        gamma_rhf_stress(
            &g.cell,
            &g.prep,
            &hcore_cfg(),
            &g.hc,
            &g.eri,
            scf,
            exx,
            &cfg,
        )
    } else {
        gamma_uhf_stress(
            &g.cell,
            &g.prep,
            &hcore_cfg(),
            &g.hc,
            &g.eri,
            scf,
            exx,
            &cfg,
        )
    }
    .expect("gamma stress")
}

fn scale3(m: &Mat3, f: f64) -> Mat3 {
    let mut o = *m;
    for r in o.iter_mut() {
        for v in r.iter_mut() {
            *v *= f;
        }
    }
    o
}

fn offdiag_miss(a: &Mat3, b: &Mat3) -> f64 {
    let mut w = 0.0_f64;
    for i in 0..3 {
        for j in 0..3 {
            if i != j {
                w = w.max((a[i][j] - b[i][j]).abs());
            }
        }
    }
    w
}

fn report(tag: &str, s: &KStress) {
    eprintln!(
        "  {tag}: |dE/dε| max {:.3e} antisym(σ) {:.1e} comm {:.1e} v_M {:.6} N {:.8} (images {}, SR {}, \
         G_lr {}, K {}, chunks {})",
        s.de_deps.iter().flatten().fold(0.0_f64, |m, v| m.max(v.abs())),
        s.antisymmetry(),
        s.commutator,
        s.madelung,
        s.electrons,
        s.n_images,
        s.n_sr_triplets,
        s.n_g_lr,
        s.n_k_eri,
        s.n_chunks
    );
}

// ====================================================== (2) 1x1x1 = Gamma, full

fn assert_one_point(name: &str, ksres: &KStress, gs: &GammaStress, exx: ExxDiv, antisym_bar: f64) {
    report(&format!("{name} 1x1x1 {exx:?}"), ksres);
    let total = max_mat(&ksres.de_deps, &gs.de_deps);
    let ov = max_mat(&ksres.parts.overlap(), &gs.parts.overlap);
    let mut two = ksres.parts.coulomb;
    for i in 0..3 {
        for j in 0..3 {
            two[i][j] += ksres.parts.exchange[i][j];
        }
    }
    let eri = max_mat(&two, &gs.parts.eri);
    let mad = max_mat(&ksres.parts.madelung, &gs.parts.madelung);
    let e2 = (ksres.e_coulomb.unwrap() + ksres.e_exchange.unwrap() - gs.e_eri).abs();
    let evlr = (ksres.e_vlr - gs.e_vlr).abs();
    eprintln!(
        "  {name} 1x1x1 {exx:?}: |dE/dε k − Gamma| {total:.2e} (overlap {ov:.2e}, J+K {eri:.2e}, \
         Madelung {mad:.2e}); |E_2e| {e2:.2e}, |E_vlr| {evlr:.2e}"
    );
    assert!(total <= GAMMA_BAR, "{name} {exx:?}: {total:e}");
    assert!(ov <= GAMMA_BAR && eri <= GAMMA_BAR && mad <= GAMMA_BAR);
    assert!(e2 <= 1e-12 && evlr <= 1e-12, "energies {e2:e} {evlr:e}");
    eprintln!(
        "  {name} 1x1x1 {exx:?}: antisym(σ) k {:.2e} Gamma {:.2e}",
        ksres.antisymmetry(),
        gs.antisymmetry()
    );
    assert!(ksres.antisymmetry() <= antisym_bar);
}

/// p-shell antisymmetry of the stress at 1x1x1. MEASURED 2026-10-16: 2.0e-13 /
/// 2.4e-13 (triclinic s+p, none / ewald), 7.8e-13 / 8.8e-13 (H2 triplet), 7.6e-13
/// / 1.3e-13 (H4 triplet); s-only cells sit at 1e-18. Before #434 this was
/// 1.1e-10 (k, triclinic) and set by the SR derivative's coordinate-dependent
/// rounding, which an earlier revision of this comment called libint2's
/// precision. Bar 5e-12 = 6x the worst measurement.
const P_ANTISYM_BAR: f64 = 5e-12;

/// PHYSICS: at 1×1×1 the k stress is the Gamma stress. ARTIFACT: independent
/// constructions (half-sphere real Gamma kernel vs full-sphere complex
/// residue kernel; image-summed vs phase-folded weights), but BLIND to the
/// phases (all 1), to 1/N_k (N_k = 1) and to the supercell Madelung.
#[test]
fn one_point_mesh_rhf_stress_is_the_gamma_stress() {
    one_point_rhf(("H2", h2_cell_g(), pyscf_sto3g_h(), ANTISYM_BAR));
}

#[test]
#[ignore = "slow (~1 min release): triclinic s+p, 2.6M SR triplets"]
fn one_point_mesh_rhf_stress_is_the_gamma_stress_triclinic_sp() {
    one_point_rhf(("tri2 s+p", triclinic_cell(), sp_basis_h(), P_ANTISYM_BAR));
}

fn one_point_rhf((name, cell, bs, asym): (&str, Cell, BasisSet, f64)) {
    {
        let ks = ksys(cell.clone(), &bs, [1, 1, 1]);
        let eri = kdense(&ks, ANCHOR_PRECISION);
        let scfs = scf_pair(&ks, &eri);
        let g = gsys(cell, &bs);
        for exx in EXX {
            let scf = &scfs[exx_idx(exx)];
            let gscf = gamma_restricted(real_of(&scf.densities[0]));
            let k = ks_rhf(&ks, &eri, scf, exx, None);
            let gs = gamma_stress(&g, &gscf, exx);
            assert_one_point(name, &k, &gs, exx, asym);
        }
    }
}

#[test]
fn one_point_mesh_uhf_stress_is_the_gamma_stress() {
    let bs = pyscf_sto3g_h();
    let ks = ksys(h3_cell(), &bs, [1, 1, 1]);
    let eri = kdense(&ks, ANCHOR_PRECISION);
    let scfs = uscf_pair(&ks, &eri, 2, 1);
    let g = gsys(h3_cell(), &bs);
    for exx in EXX {
        let scf = &scfs[exx_idx(exx)];
        let gscf = gamma_unrestricted(
            real_of(&scf.density_alpha[0]),
            real_of(&scf.density_beta[0]),
        );
        let k = ks_uhf(&ks, &eri, scf, exx, None);
        let gs = gamma_stress(&g, &gscf, exx);
        assert_one_point("H3 UHF", &k, &gs, exx, ANTISYM_BAR);
    }
}

// ====================================================== energy bookkeeping

/// The pieces the stress differentiates add up to the SCF energy: a check
/// that `e_vlr`, `e_coulomb`, `e_exchange`, `e_madelung` are the energy's
/// terms with the energy's prefactors (the strain of a mis-scaled `Φ` would
/// still be self-consistent against nothing).
#[test]
fn stress_energy_pieces_sum_to_the_scf_energy_h2_1x1x3() {
    let bs = pyscf_sto3g_h();
    let ks = ksys(h2_cell_g(), &bs, [1, 1, 3]);
    let eri = kdense(&ks, ANCHOR_PRECISION);
    let vm = ks.mesh.madelung(&ks.cell).unwrap();
    for (exx, m) in [(ExxDiv::None, 0.0), (ExxDiv::Ewald, vm)] {
        let scf = krhf(&ks, Box::new(eri.jk_builder_with_madelung(m)));
        let st = ks_rhf(&ks, &eri, &scf, exx, None);
        // (1/N_k) Σ_k tr[h D] + E_J + E_K + E_M + E_nn.
        let nk = ks.mesh.nk();
        let n = ks.prep.nbasis();
        let mut e1 = 0.0;
        for (d, h) in scf.densities.iter().zip(&ks.hk.h) {
            for a in 0..n {
                for b in 0..n {
                    e1 += (d[(a, b)] * h[(b, a)]).re;
                }
            }
        }
        e1 /= nk as f64;
        let (ej, ek) = (st.e_coulomb.unwrap(), st.e_exchange.unwrap());
        let e = e1 + ej + ek + st.e_madelung + ks.hk.enn;
        eprintln!(
            "  {exx:?}: E_pieces {e:.12} vs SCF {:.12} (diff {:.2e}); E_J {:.8} E_K {:.8} E_M {:.8}",
            scf.energy,
            e - scf.energy,
            ej,
            ek,
            st.e_madelung
        );
        assert!(
            (e - scf.energy).abs() < 1e-9,
            "{exx:?}: {:e}",
            e - scf.energy
        );
    }
}

// ====================================================== (3) supercell anchor

const SC_BAR: f64 = 1e-10;

struct ScRef {
    /// Gamma supercell `dE/dε / N`, per exxdiv.
    de: [Mat3; 2],
}

/// Supercell anchor of a closed-shell system: `dE/dε_k = dE/dε_sc / N`, σ
/// equal. Returns the references for the mutants.
fn rhf_supercell_anchor(
    name: &str,
    cell: Cell,
    bs: &BasisSet,
    n: [usize; 3],
) -> (KSys, KDenseAftEri, [KScfResult; 2], ScRef) {
    let nc = (n[0] * n[1] * n[2]) as f64;
    let ks = ksys(cell.clone(), bs, n);
    let eri = kdense(&ks, ANCHOR_PRECISION);
    let scfs = scf_pair(&ks, &eri);
    let g = gsys(supercell(&cell, n, 1), bs);
    let mut de = [ZERO; 2];
    for (i, exx) in EXX.into_iter().enumerate() {
        let gscf = gamma_restricted(unfold(&ks.mesh, &scfs[i].densities));
        let k = ks_rhf(&ks, &eri, &scfs[i], exx, None);
        let gs = gamma_stress(&g, &gscf, exx);
        let r = scale3(&gs.de_deps, 1.0 / nc);
        let miss = max_mat(&k.de_deps, &r);
        let smiss = max_mat(&k.sigma, &gs.sigma);
        report(&format!("{name} {n:?} {exx:?}"), &k);
        eprintln!(
            "  {name} {n:?} {exx:?}: |dE/dε_k − dE/dε_sc/N| {miss:.2e}; |σ_k − σ_sc| {smiss:.2e}; \
             v_M k {:.6} sc {:.6}",
            k.madelung, gs.madelung
        );
        assert!(miss <= SC_BAR, "{name} {exx:?}: supercell miss {miss:e}");
        assert!(
            smiss <= SC_BAR / k.volume,
            "{name} {exx:?}: σ miss {smiss:e}"
        );
        assert!(k.antisymmetry() <= ANTISYM_BAR, "σ not symmetric");
        de[i] = r;
    }
    (ks, eri, scfs, ScRef { de })
}

const ZERO: Mat3 = [[0.0; 3]; 3];

/// The k-specific defects (phases, q, 1/N_k, mesh Madelung): the new risk of
/// this module, run in CI.
const K_MUTANTS: [KStressMutation; 4] = [
    KStressMutation::NoPhase,
    KStressMutation::ConjD,
    KStressMutation::WrongQ,
    KStressMutation::NoInvNk,
];
const K_EWALD_MUTANTS: [KStressMutation; 2] = [
    KStressMutation::GammaMadelung,
    KStressMutation::GammaMadelungStrain,
];
/// The Gamma-style term defects. Measured against the 1x1x1 Gamma anchor
/// (`term_mutants_miss_the_one_point_gamma_anchor`) and the hcore gate; also
/// swept against the supercell in the `#[ignore]`d `h2_1x1x3_term_mutants_*`.
const TERM_MUTANTS: [KStressMutation; 9] = [
    KStressMutation::NoPulay,
    KStressMutation::NoC0Overlap,
    KStressMutation::FtNoCentres,
    KStressMutation::GUnstrained,
    KStressMutation::NoVolume,
    KStressMutation::NoC0Volume,
    KStressMutation::NoSrImages,
    KStressMutation::NoEwaldLr,
    KStressMutation::NoEwaldBg,
];
const TERM_EWALD_MUTANTS: [KStressMutation; 2] =
    [KStressMutation::NoMadelung, KStressMutation::MadelungS];
/// Mutants that only change the diagonal by construction (volume terms).
const DIAGONAL_ONLY: [KStressMutation; 3] = [
    KStressMutation::NoVolume,
    KStressMutation::NoC0Volume,
    KStressMutation::NoEwaldBg,
];

/// Every mutant must miss the supercell anchor by > MUTANT_BAR on at least one
/// component (both exxdiv; the Madelung ones under ewald only and BLIND under
/// none); the diagonal-only ones must leave the off-diagonal at the floor.
fn assert_mutants_fail(
    name: &str,
    run: &dyn Fn(ExxDiv, Option<KStressMutation>) -> KStress,
    refs: &ScRef,
    always: &[KStressMutation],
    ewald_only: &[KStressMutation],
) {
    assert_mutants_fail_floor(name, run, refs, always, ewald_only, SC_BAR);
}

/// [`assert_mutants_fail`] with an explicit anchor floor `sc_bar` for the
/// blind / diagonal-only checks (the p-shell supercell anchor sits at 4.6e-13,
/// above the s-only `SC_BAR` by a small factor; before #434 it sat at 1.7e-10
/// with the exponent pinned at 1e7).
fn assert_mutants_fail_floor(
    name: &str,
    run: &dyn Fn(ExxDiv, Option<KStressMutation>) -> KStress,
    refs: &ScRef,
    always: &[KStressMutation],
    ewald_only: &[KStressMutation],
    sc_bar: f64,
) {
    for &m in always {
        for (i, exx) in EXX.into_iter().enumerate() {
            let s = run(exx, Some(m));
            let miss = max_mat(&s.de_deps, &refs.de[i]);
            let off = offdiag_miss(&s.de_deps, &refs.de[i]);
            eprintln!("  {name} mutant {m:?} {exx:?}: miss {miss:.2e} (off-diagonal {off:.2e})");
            assert!(
                miss > MUTANT_BAR,
                "{name}: {m:?} {exx:?} not caught ({miss:e})"
            );
            if DIAGONAL_ONLY.contains(&m) {
                assert!(
                    off <= sc_bar,
                    "{name}: {m:?} must be diagonal-only ({off:e})"
                );
            }
        }
    }
    for &m in ewald_only {
        let se = run(ExxDiv::Ewald, Some(m));
        let sn = run(ExxDiv::None, Some(m));
        let (me, mn) = (
            max_mat(&se.de_deps, &refs.de[1]),
            max_mat(&sn.de_deps, &refs.de[0]),
        );
        eprintln!("  {name} mutant {m:?}: ewald miss {me:.2e}, none miss {mn:.2e} (blind)");
        assert!(me > MUTANT_BAR, "{name}: {m:?} (ewald) not caught ({me:e})");
        assert!(
            mn <= sc_bar,
            "{name}: {m:?} must be blind under none ({mn:e})"
        );
    }
}

/// PHYSICS: a Gamma-centred k mesh IS the diag(N) supercell, so per cell
/// `dE/dε = dE/dε_sc/N` and `σ = σ_sc` (Ω_sc = N Ω), with the supercell's
/// Madelung constant as the exchange-divergence constant. ARTIFACT: a wrong
/// phase / q sign / 1/N_k / Madelung lattice would NOT reproduce an
/// explicit supercell. NON-TRIM mesh (n_2 = 3): ConjD and WrongQ are
/// identities on every n_i <= 2 mesh.
#[test]
fn h2_1x1x3_stress_is_the_supercell_stress_per_cell() {
    let bs = pyscf_sto3g_h();
    let (ks, eri, scfs, refs) = rhf_supercell_anchor("H2", h2_cell_g(), &bs, [1, 1, 3]);
    assert_mutants_fail(
        "H2 1x1x3",
        &|exx, m| ks_rhf(&ks, &eri, &scfs[exx_idx(exx)], exx, m),
        &refs,
        &K_MUTANTS,
        &K_EWALD_MUTANTS,
    );
}

/// The Gamma-style term mutants against the same supercell anchor.
#[test]
#[ignore = "slow (~2 min release): 22 more stress evaluations after the supercell anchor"]
fn h2_1x1x3_term_mutants_miss_the_supercell_anchor() {
    let bs = pyscf_sto3g_h();
    let (ks, eri, scfs, refs) = rhf_supercell_anchor("H2", h2_cell_g(), &bs, [1, 1, 3]);
    assert_mutants_fail(
        "H2 1x1x3",
        &|exx, m| ks_rhf(&ks, &eri, &scfs[exx_idx(exx)], exx, m),
        &refs,
        &TERM_MUTANTS,
        &TERM_EWALD_MUTANTS,
    );
}

/// UHF H3 doublet: supercell (6, 3), multiplicity 4.
#[test]
#[ignore = "slow (~8 min release): 9-atom supercell Gamma stress x2 + 36 k stress evaluations"]
fn h3_uhf_1x1x3_stress_is_the_supercell_stress_per_cell() {
    let bs = pyscf_sto3g_h();
    let n = [1, 1, 3];
    let ks = ksys(h3_cell(), &bs, n);
    let eri = kdense(&ks, ANCHOR_PRECISION);
    let scfs = uscf_pair(&ks, &eri, 2, 1);
    let g = gsys(supercell(&h3_cell(), n, 4), &bs);
    let mut de = [ZERO; 2];
    for (i, exx) in EXX.into_iter().enumerate() {
        let gscf = gamma_unrestricted(
            unfold(&ks.mesh, &scfs[i].density_alpha),
            unfold(&ks.mesh, &scfs[i].density_beta),
        );
        let k = ks_uhf(&ks, &eri, &scfs[i], exx, None);
        let gs = gamma_stress(&g, &gscf, exx);
        let r = scale3(&gs.de_deps, 1.0 / 3.0);
        let miss = max_mat(&k.de_deps, &r);
        report(&format!("H3 UHF {n:?} {exx:?}"), &k);
        eprintln!("  H3 UHF {n:?} {exx:?}: |dE/dε_k − dE/dε_sc/N| {miss:.2e}");
        assert!(miss <= SC_BAR, "{exx:?}: {miss:e}");
        assert!(k.antisymmetry() <= ANTISYM_BAR);
        de[i] = r;
    }
    let refs = ScRef { de };
    let run = |exx, m| ks_uhf(&ks, &eri, &scfs[exx_idx(exx)], exx, m);
    assert_mutants_fail("H3 UHF 1x1x3", &run, &refs, &K_MUTANTS, &K_EWALD_MUTANTS);
    assert_mutants_fail(
        "H3 UHF 1x1x3",
        &run,
        &refs,
        &TERM_MUTANTS,
        &TERM_EWALD_MUTANTS,
    );
}

/// BLIND SPOT, asserted: on the TRIM-only 2×2×2 mesh `D(k)` is real and
/// `q ≡ −q`, so ConjD and WrongQ are IDENTITIES; NoPhase and NoInvNk are not.
#[test]
#[ignore = "slow (~2.5 min release): 8-point mesh, 10 stress evaluations"]
fn h2_2x2x2_trim_mesh_is_blind_to_phase_mutants() {
    let bs = pyscf_sto3g_h();
    let ks = ksys(h2_cell_g(), &bs, [2, 2, 2]);
    let eri = kdense(&ks, ANCHOR_PRECISION);
    let scfs = scf_pair(&ks, &eri);
    for exx in EXX {
        let scf = &scfs[exx_idx(exx)];
        let f0 = ks_rhf(&ks, &eri, scf, exx, None);
        for m in [KStressMutation::ConjD, KStressMutation::WrongQ] {
            let f = ks_rhf(&ks, &eri, scf, exx, Some(m));
            let d = max_mat(&f.de_deps, &f0.de_deps);
            eprintln!("  H2 2x2x2 {exx:?} {m:?}: |mut − ref| {d:.2e} (identity on TRIM meshes)");
            assert!(
                d <= IDENTITY_BAR,
                "{m:?} should be an identity on 2x2x2 ({d:e})"
            );
        }
        for m in [KStressMutation::NoPhase, KStressMutation::NoInvNk] {
            let f = ks_rhf(&ks, &eri, scf, exx, Some(m));
            let d = max_mat(&f.de_deps, &f0.de_deps);
            eprintln!("  H2 2x2x2 {exx:?} {m:?}: |mut − ref| {d:.2e}");
            assert!(d > MUTANT_BAR, "{m:?} not caught on 2x2x2 ({d:e})");
        }
        assert!(f0.antisymmetry() <= ANTISYM_BAR);
    }
}

/// closed-shell kUHF(D/2, D/2) ≡ kRHF(D).
#[test]
fn closed_shell_kuhf_stress_is_the_krhf_stress() {
    let bs = pyscf_sto3g_h();
    let ks = ksys(h2_cell_g(), &bs, [1, 1, 3]);
    let eri = kdense(&ks, ANCHOR_PRECISION);
    let rs = scf_pair(&ks, &eri);
    let mut us = uscf_pair(&ks, &eri, 1, 1);
    for exx in EXX {
        let (r, u) = (&rs[exx_idx(exx)], &mut us[exx_idx(exx)]);
        let half: Vec<Array2<Complex64>> =
            r.densities.iter().map(|d| d.mapv(|z| z * 0.5)).collect();
        u.density_alpha = half.clone();
        u.density_beta = half;
        let sr = ks_rhf(&ks, &eri, r, exx, None);
        let su = ks_uhf(&ks, &eri, u, exx, None);
        let d = max_mat(&sr.de_deps, &su.de_deps);
        eprintln!("  H2 1x1x3 {exx:?}: |σ_kUHF(D/2,D/2) − σ_kRHF(D)| {d:.2e}");
        assert!(d <= IDENTITY_BAR, "{exx:?}: {d:e}");
    }
}

// ====================================================== (4) FD of own k energy

/// k-RHF energies `[none, ewald]` per cell of a (strained) cell, dense AFT at
/// `ANCHOR_PRECISION`, mesh rebuilt from the same integer numerators.
fn k_energies(cell: &Cell, bs: &BasisSet, n: [usize; 3]) -> [f64; 2] {
    let ks = ksys(cell.clone(), bs, n);
    let eri = kdense(&ks, ANCHOR_PRECISION);
    let vm = ks.mesh.madelung(&ks.cell).unwrap();
    [
        krhf(&ks, Box::new(eri.jk_builder_with_madelung(0.0))).energy,
        krhf(&ks, Box::new(eri.jk_builder_with_madelung(vm))).energy,
    ]
}

/// PHYSICS: analytic `dE/dε` of the k energy vs central FD (h = 1e-4) of
/// ferric's own k energy on strained cells (lattice AND atoms strained, index
/// sets frozen, k at fixed fractional coordinates), all nine components, both
/// exxdiv, with a fresh SCF at every strained point (the analytic value is at
/// the converged density, so the density response drops out by stationarity).
#[test]
fn h2_1x1x3_stress_matches_fd_of_own_k_energy_all_nine() {
    let bs = pyscf_sto3g_h();
    let n = [1, 1, 3];
    let ks = ksys(h2_cell_g(), &bs, n);
    let eri = kdense(&ks, ANCHOR_PRECISION);
    let vm = ks.mesh.madelung(&ks.cell).unwrap();
    let scf = [
        krhf(&ks, Box::new(eri.jk_builder_with_madelung(0.0))),
        krhf(&ks, Box::new(eri.jk_builder_with_madelung(vm))),
    ];
    let mut fd = [[[0.0; 3]; 3]; 2];
    for (i, j) in ALL9 {
        let ep = k_energies(&strained(&ks.cell, i, j, FD_H), &bs, n);
        let em = k_energies(&strained(&ks.cell, i, j, -FD_H), &bs, n);
        for x in 0..2 {
            fd[x][i][j] = (ep[x] - em[x]) / (2.0 * FD_H);
        }
    }
    for (x, exx) in EXX.into_iter().enumerate() {
        let s = ks_rhf(&ks, &eri, &scf[x], exx, None);
        let miss = max_mat(&s.de_deps, &fd[x]);
        eprintln!("  H2 1x1x3 {exx:?}: |analytic − FD| {miss:.2e}");
        assert!(miss <= K_FD_BAR, "{exx:?}: FD miss {miss:e}");
    }
}

/// Measured floor 1.0e-8 (h = 1e-4: the h² f‴/6 truncation of the central FD,
/// identical on both exxdiv); the smallest mutant miss is 4.9e-2.
const K_FD_BAR: f64 = 5e-8;

// ====================================================== refusals

#[test]
fn stress_refuses_unconverged_scf_and_ecp() {
    let bs = pyscf_sto3g_h();
    let ks = ksys(h2_cell_g(), &bs, [1, 1, 3]);
    let eri = kdense(&ks, ANCHOR_PRECISION);
    let mut scf = krhf(&ks, Box::new(eri.jk_builder_with_madelung(0.0)));
    // Unconverged SCF.
    scf.converged = false;
    let e = kpoint_rhf_stress(
        &ks.cell,
        &ks.prep,
        &ks.mesh,
        &hcore_cfg(),
        &ks.hk,
        KStressJk::Dense(&eri),
        &scf,
        ExxDiv::None,
        &scfg(None),
    )
    .unwrap_err()
    .to_string();
    assert!(e.contains("did not converge"), "{e}");
    scf.converged = true;
    // An hcore carrying V_ECP for a cell whose basis has no ECP centre: the ECP walk is
    // driven by the basis and cross-checked against hk.v_ecp.
    let mut hk_ecp = ks.hk.clone();
    hk_ecp.v_ecp = Some(hk_ecp.h.clone());
    let e = kpoint_rhf_stress(
        &ks.cell,
        &ks.prep,
        &ks.mesh,
        &hcore_cfg(),
        &hk_ecp,
        KStressJk::Dense(&eri),
        &scf,
        ExxDiv::None,
        &scfg(None),
    )
    .unwrap_err()
    .to_string();
    assert!(e.contains("carries V_ECP but the basis"), "{e}");
}

/// FD `dE/dε` of `energies` (`[none, ewald]`) on `cell`, all nine components, step `h`.
fn fd_all9(cell: &Cell, h: f64, energies: &dyn Fn(&Cell) -> [f64; 2]) -> [Mat3; 2] {
    let mut fd = [[[0.0; 3]; 3]; 2];
    for (i, j) in ALL9 {
        let ep = energies(&strained(cell, i, j, h));
        let em = energies(&strained(cell, i, j, -h));
        for x in 0..2 {
            fd[x][i][j] = (ep[x] - em[x]) / (2.0 * h);
        }
    }
    fd
}

/// k-RHF energies `[none, ewald]` on `(cell, bs, n, centring)`.
fn k_energies_c(cell: &Cell, bs: &BasisSet, n: [usize; 3], centring: MeshCentring) -> [f64; 2] {
    let ks = ksys_c(cell.clone(), bs, n, centring);
    let eri = kdense(&ks, ANCHOR_PRECISION);
    let vm = ks.mesh.madelung(&ks.cell).unwrap();
    [
        krhf(&ks, Box::new(eri.jk_builder_with_madelung(0.0))).energy,
        krhf(&ks, Box::new(eri.jk_builder_with_madelung(vm))).energy,
    ]
}

/// k-UHF energy of the H3 doublet (2, 1) on a (strained) cell, exxdiv none and ewald.
fn k_energies_uhf(cell: &Cell, bs: &BasisSet, n: [usize; 3]) -> [f64; 2] {
    let ks = ksys(cell.clone(), bs, n);
    let eri = kdense(&ks, ANCHOR_PRECISION);
    let vm = ks.mesh.madelung(&ks.cell).unwrap();
    [
        kuhf(&ks, Box::new(eri.jk_builder_with_madelung(0.0)), 2, 1).energy,
        kuhf(&ks, Box::new(eri.jk_builder_with_madelung(vm)), 2, 1).energy,
    ]
}

/// The UHF twin of the FD anchor (H3 doublet 1x1x3, both exxdiv), at h and 2h:
/// the h = 1e-4 miss is the central-FD h² truncation (Richardson removes it),
/// and the two UHF-specific mutants (alpha/beta cross-paired exchange, swapped
/// spin in the Pulay weight; both identities for RHF) must miss.
#[test]
#[ignore = "slow (~90 s release): 36 strained dense-AFT builds + 72 k-UHF SCFs (h and 2h)"]
fn h3_uhf_1x1x3_stress_matches_fd_of_own_k_energy_all_nine() {
    let bs = pyscf_sto3g_h();
    let n = [1, 1, 3];
    let ks = ksys(h3_cell(), &bs, n);
    let eri = kdense(&ks, ANCHOR_PRECISION);
    let vm = ks.mesh.madelung(&ks.cell).unwrap();
    let scf = [
        kuhf(&ks, Box::new(eri.jk_builder_with_madelung(0.0)), 2, 1),
        kuhf(&ks, Box::new(eri.jk_builder_with_madelung(vm)), 2, 1),
    ];
    let e = |c: &Cell| k_energies_uhf(c, &bs, n);
    let fd1 = fd_all9(&ks.cell, FD_H, &e);
    let fd2 = fd_all9(&ks.cell, 2.0 * FD_H, &e);
    for (x, exx) in EXX.into_iter().enumerate() {
        let s = ks_uhf(&ks, &eri, &scf[x], exx, None);
        let mut rich = ZERO;
        for a in 0..3 {
            for b in 0..3 {
                rich[a][b] = (4.0 * fd1[x][a][b] - fd2[x][a][b]) / 3.0;
            }
        }
        let miss = max_mat(&s.de_deps, &fd1[x]);
        let miss2 = max_mat(&s.de_deps, &fd2[x]);
        let missr = max_mat(&s.de_deps, &rich);
        eprintln!(
            "  H3 UHF 1x1x3 {exx:?}: |analytic − FD| h {miss:.2e}, 2h {miss2:.2e} (ratio {:.2}), \
             Richardson {missr:.2e}",
            miss2 / miss
        );
        assert!(miss <= K_FD_BAR_UHF, "{exx:?}: FD miss {miss:e}");
        // h² scaling: the miss is truncation, not a formula error.
        assert!(
            (3.0..5.0).contains(&(miss2 / miss)),
            "not h²: {}",
            miss2 / miss
        );
        assert!(missr <= K_FD_BAR_UHF / 10.0, "Richardson miss {missr:e}");
        for m in [KStressMutation::CrossSpinK, KStressMutation::SwapSpinFock] {
            let sm = ks_uhf(&ks, &eri, &scf[x], exx, Some(m));
            let mm = max_mat(&sm.de_deps, &fd1[x]);
            eprintln!("  H3 UHF 1x1x3 {exx:?} mutant {m:?}: miss vs FD {mm:.2e}");
            assert!(mm > MUTANT_BAR, "{m:?} {exx:?} not caught ({mm:e})");
        }
    }
}

/// This test's own measured floor: 2.7e-8 at h = 1e-4, 1.2e-7 at 2h (ratio
/// 4.2, i.e. h² truncation), 2e-9 after Richardson. The bar is ~4x the floor
/// and 6 orders below the smallest mutant miss (CrossSpinK 7.9e-1,
/// SwapSpinFock 5.8e-1).
const K_FD_BAR_UHF: f64 = 1e-7;

// ====================================================== input consistency

fn rhf_stress_result(
    ks: &KSys,
    eri: &KDenseAftEri,
    scf: &KScfResult,
    exx: ExxDiv,
) -> Result<KStress, ferric_core::FerricError> {
    kpoint_rhf_stress(
        &ks.cell,
        &ks.prep,
        &ks.mesh,
        &hcore_cfg(),
        &ks.hk,
        KStressJk::Dense(eri),
        scf,
        exx,
        &scfg(None),
    )
}

/// PHYSICS: a stress is only meaningful for the energy whose SCF produced
/// `D`; the stored Fock tells which one that was. ARTIFACT: a wrong `exxdiv`
/// or a KS result both give a FINITE σ (nothing else fails), so the guard is
/// the stored-Fock comparison. Measured mismatches are printed below; the
/// matching case sits at the 1e-12 level (see `FOCK_CONSISTENCY_BAR`).
#[test]
fn stress_refuses_a_wrong_exxdiv_and_a_ks_result() {
    let bs = pyscf_sto3g_h();
    let n = [1, 1, 2];
    let ks = ksys(h2_cell_g(), &bs, n);
    let eri = kdense(&ks, ANCHOR_PRECISION);
    let vm = ks.mesh.madelung(&ks.cell).unwrap();
    let scf_none = krhf(&ks, Box::new(eri.jk_builder_with_madelung(0.0)));
    let scf_ewald = krhf(&ks, Box::new(eri.jk_builder_with_madelung(vm)));
    // Matching inputs pass (and the match is far below the bar).
    let a = rhf_stress_result(&ks, &eri, &scf_none, ExxDiv::None).expect("none/none");
    let b = rhf_stress_result(&ks, &eri, &scf_ewald, ExxDiv::Ewald).expect("ewald/ewald");
    eprintln!(
        "  matching inputs: Fock mismatch {:.2e} (none), {:.2e} (ewald)",
        a.fock_mismatch, b.fock_mismatch
    );
    assert!(a.fock_mismatch < 1e-11 && b.fock_mismatch < 1e-11);
    // (a) wrong exxdiv, both directions.
    for (tag, scf, exx) in [
        ("SCF none, stress ewald", &scf_none, ExxDiv::Ewald),
        ("SCF ewald, stress none", &scf_ewald, ExxDiv::None),
    ] {
        let e = rhf_stress_result(&ks, &eri, scf, exx)
            .unwrap_err()
            .to_string();
        eprintln!("  {tag}: {e}");
        assert!(e.contains("differs from the SCF's stored fock"), "{e}");
    }
    // (b) a KS-DFT (LDA) result on the same mesh: its Fock carries V_xc.
    let mut kcfg = KRksConfig::new(&ks.cell, ExxDiv::None, "LDA");
    kcfg.krhf.hcore = hcore_cfg();
    kcfg.krhf.scf.energy_conv = 1e-10;
    kcfg.krhf.scf.grad_conv = 1e-7;
    kcfg.grid = PeriodicGridConfig::with_size(30, 50);
    let kres = solve_krks(&ks.cell, &ks.prep, None, &ks.mesh, &kcfg).expect("k-LDA");
    let e = rhf_stress_result(&ks, &eri, &kres.scf, ExxDiv::None)
        .unwrap_err()
        .to_string();
    eprintln!("  KS (LDA) result: {e}");
    assert!(e.contains("differs from the SCF's stored fock"), "{e}");
}

/// A dense source built with a range-separated K kernel is refused (its
/// strain is not implemented), even though its own SCF/Fock are consistent.
#[test]
fn stress_refuses_a_range_separated_dense_source() {
    let bs = pyscf_sto3g_h();
    let ks = ksys(h2_cell_g(), &bs, [1, 1, 2]);
    let eri = KDenseAftEri::build(
        &ks.cell,
        &ks.prep,
        &ks.mesh,
        &ks.hk.s,
        ExxDiv::None,
        &KDenseAftConfig {
            precision: ANCHOR_PRECISION,
            rsh: Some(ferric_pbc::rsh::RshParams {
                omega: 0.4,
                c_sr: 1.0,
                c_lr: 0.25,
            }),
            ..Default::default()
        },
    )
    .expect("rsh dense kernels");
    let scf = krhf(&ks, Box::new(eri.jk_builder_with_madelung(0.0)));
    let e = rhf_stress_result(&ks, &eri, &scf, ExxDiv::None)
        .unwrap_err()
        .to_string();
    assert!(e.contains("range-separated"), "{e}");
}

/// The fixed-D hcore entry applies the same input checks.
#[test]
fn hcore_fixed_d_entry_checks_its_inputs() {
    let bs = pyscf_sto3g_h();
    let ks = ksys(h2_cell_g(), &bs, [1, 1, 2]);
    let eri = kdense(&ks, ANCHOR_PRECISION);
    let scf = krhf(&ks, Box::new(eri.jk_builder_with_madelung(0.0)));
    let call = |cfg: &PeriodicHcoreConfig, hk: &PeriodicHcoreK, d: &[Array2<Complex64>]| {
        kpoint_hcore_stress_fixed_d(&ks.cell, &ks.prep, &ks.mesh, cfg, hk, d, &scfg(None))
            .err()
            .map(|e| e.to_string())
    };
    assert!(call(&hcore_cfg(), &ks.hk, &scf.densities).is_none());
    let mut other = hcore_cfg();
    other.omega = 0.9;
    assert!(call(&other, &ks.hk, &scf.densities)
        .unwrap()
        .contains("omega"));
    assert!(call(&hcore_cfg(), &ks.hk, &scf.densities[..1])
        .unwrap()
        .contains("shape mismatch"));
    let mut hk_ecp = ks.hk.clone();
    hk_ecp.v_ecp = Some(hk_ecp.h.clone());
    assert!(call(&hcore_cfg(), &hk_ecp, &scf.densities)
        .unwrap()
        .contains("ECP"));
}

// ====================================================== measured mutant claims

/// The Gamma-style term mutants against the 1x1x1 Gamma anchor (H2): each
/// must miss it, and the k-specific ones (phases, q, 1/N_k, mesh Madelung)
/// are BLIND there (asserted) — which is why the 1x1x3 supercell anchor, not
/// this one, carries them.
#[test]
#[ignore = "slow-ish (~1 min release): ~30 stress evaluations on H2 1x1x1"]
fn term_mutants_miss_the_one_point_gamma_anchor() {
    let bs = pyscf_sto3g_h();
    let ks = ksys(h2_cell_g(), &bs, [1, 1, 1]);
    let eri = kdense(&ks, ANCHOR_PRECISION);
    let scfs = scf_pair(&ks, &eri);
    let g = gsys(h2_cell_g(), &bs);
    for (i, exx) in EXX.into_iter().enumerate() {
        let scf = &scfs[i];
        let gscf = gamma_restricted(real_of(&scf.densities[0]));
        let gs = gamma_stress(&g, &gscf, exx);
        let mut all: Vec<KStressMutation> = TERM_MUTANTS.to_vec();
        all.extend(TERM_EWALD_MUTANTS);
        for m in all {
            if TERM_EWALD_MUTANTS.contains(&m) && i == 0 {
                continue;
            }
            let miss = max_mat(&ks_rhf(&ks, &eri, scf, exx, Some(m)).de_deps, &gs.de_deps);
            eprintln!("  1x1x1 anchor, mutant {m:?} {exx:?}: miss {miss:.2e}");
            assert!(miss > MUTANT_BAR, "{m:?} {exx:?}: {miss:e}");
        }
        let mut blind: Vec<KStressMutation> = K_MUTANTS.to_vec();
        blind.extend(K_EWALD_MUTANTS);
        for m in blind {
            let miss = max_mat(&ks_rhf(&ks, &eri, scf, exx, Some(m)).de_deps, &gs.de_deps);
            eprintln!("  1x1x1 anchor, k-mutant {m:?} {exx:?}: miss {miss:.2e} (blind)");
            // Identities at N_k = 1 (phases 1, 1/N_k = 1, mesh = primitive v_M).
            assert!(miss <= GAMMA_BAR * 100.0, "{m:?} {exx:?}: {miss:e}");
        }
    }
}

// ====================================================== MP mesh, p shells

/// PHYSICS: a Monkhorst-Pack even mesh (k = ±b3/4 for N = 2, residue modulus
/// 4) is a twisted-boundary supercell with NO explicit-supercell anchor, so
/// only the FD of ferric's own k energy pins it. Non-TRIM (2k ∉ G).
#[test]
fn h2_mp_1x1x2_stress_matches_fd_of_own_k_energy_all_nine() {
    let bs = pyscf_sto3g_h();
    let n = [1, 1, 2];
    let c = MeshCentring::MonkhorstPack;
    let ks = ksys_c(h2_cell_g(), &bs, n, c);
    assert_eq!(ks.mesh.residue_moduli(), [1, 1, 4]);
    let eri = kdense(&ks, ANCHOR_PRECISION);
    let vm = ks.mesh.madelung(&ks.cell).unwrap();
    let scf = [
        krhf(&ks, Box::new(eri.jk_builder_with_madelung(0.0))),
        krhf(&ks, Box::new(eri.jk_builder_with_madelung(vm))),
    ];
    let e = |cell: &Cell| k_energies_c(cell, &bs, n, c);
    let fd1 = fd_all9(&ks.cell, FD_H, &e);
    let fd2 = fd_all9(&ks.cell, 2.0 * FD_H, &e);
    for (x, exx) in EXX.into_iter().enumerate() {
        let s = ks_rhf(&ks, &eri, &scf[x], exx, None);
        let (m1, m2) = (max_mat(&s.de_deps, &fd1[x]), max_mat(&s.de_deps, &fd2[x]));
        eprintln!("  H2 MP 1x1x2 {exx:?}: |analytic − FD| h {m1:.2e}, 2h {m2:.2e}");
        assert!(m1 <= K_FD_BAR, "{exx:?}: {m1:e}");
        assert!(s.antisymmetry() <= ANTISYM_BAR);
        for m in [KStressMutation::NoPhase, KStressMutation::NoInvNk] {
            let mm = max_mat(&ks_rhf(&ks, &eri, &scf[x], exx, Some(m)).de_deps, &fd1[x]);
            eprintln!("  H2 MP 1x1x2 {exx:?} mutant {m:?}: miss {mm:.2e}");
            assert!(mm > MUTANT_BAR, "{m:?} {exx:?}: {mm:e}");
        }
        for m in K_EWALD_MUTANTS {
            let mm = max_mat(&ks_rhf(&ks, &eri, &scf[x], exx, Some(m)).de_deps, &fd1[x]);
            eprintln!("  H2 MP 1x1x2 {exx:?} mutant {m:?}: miss {mm:.2e}");
            if exx == ExxDiv::Ewald {
                assert!(mm > MUTANT_BAR, "{m:?}: {mm:e}");
            }
        }
    }
}

/// l > 0 at N_k > 1: H2 with an s+p basis (nao 8), Gamma-centred 1x1x3
/// (non-TRIM), FD of the own k energy, analytic at the default gradient
/// nucleus exponent and at 1e8 (both agree with the FD at the FD-truncation
/// level; the former exponent-dependent p-shell SR-derivative noise is gone,
/// #434).
#[test]
#[ignore = "slow: s+p H2 1x1x3, 18 strained dense-AFT builds + 36 k-SCFs"]
fn h2_sp_1x1x3_stress_matches_fd_of_own_k_energy_all_nine() {
    let bs = sp_basis_h();
    let n = [1, 1, 3];
    let c = MeshCentring::Gamma;
    let ks = ksys_c(h2_cell_g(), &bs, n, c);
    let eri = kdense(&ks, ANCHOR_PRECISION);
    let vm = ks.mesh.madelung(&ks.cell).unwrap();
    let scf = [
        krhf(&ks, Box::new(eri.jk_builder_with_madelung(0.0))),
        krhf(&ks, Box::new(eri.jk_builder_with_madelung(vm))),
    ];
    let e = |cell: &Cell| k_energies_c(cell, &bs, n, c);
    let fd1 = fd_all9(&ks.cell, FD_H, &e);
    for (x, exx) in EXX.into_iter().enumerate() {
        for nuc in [None, Some(1e8)] {
            let s = kpoint_rhf_stress(
                &ks.cell,
                &ks.prep,
                &ks.mesh,
                &hcore_cfg(),
                &ks.hk,
                KStressJk::Dense(&eri),
                &scf[x],
                exx,
                &KStressConfig {
                    nucleus_exponent: nuc,
                    ..scfg(None)
                },
            )
            .unwrap();
            let m1 = max_mat(&s.de_deps, &fd1[x]);
            eprintln!(
                "  H2 s+p 1x1x3 {exx:?} nucleus exponent {nuc:?}: |analytic − FD| {m1:.2e}, \
                 antisym(σ) {:.2e}",
                s.antisymmetry()
            );
            assert!(m1 <= K_FD_BAR_P, "{exx:?} {nuc:?}: {m1:e}");
        }
        for m in [
            KStressMutation::NoPhase,
            KStressMutation::WrongQ,
            KStressMutation::ConjD,
        ] {
            let mm = max_mat(&ks_rhf(&ks, &eri, &scf[x], exx, Some(m)).de_deps, &fd1[x]);
            eprintln!("  H2 s+p 1x1x3 {exx:?} mutant {m:?}: miss {mm:.2e}");
            assert!(mm > MUTANT_BAR, "{m:?} {exx:?}: {mm:e}");
        }
    }
}

/// Measured s+p floor: 9.8e-9 at the default nucleus exponent, 9.4e-9 / 3.5e-9
/// at 1e8 (pre-#434 numbers; the h = 1e-4 central-FD truncation dominates
/// exactly as for s shells); smallest mutant miss 4.6e-2. Bar 5e-8 = the
/// s-shell bar.
const K_FD_BAR_P: f64 = 5e-8;

// ============================================================ RS-GDF (unsplit)
//
// The fitted J/K at k: SR 3-centre and metric strain walks with the force's
// per-(l, t) / per-t phase-folded weights, the per-q LR strain on the full
// {G + q} set (complex residue weights, aux-FT strain), the J3 G = 0 overlap
// and volume terms and the strain-only J2 G = 0 volume term.

use ferric_pbc::grad::RsGdfGradSource;
use ferric_pbc::rsgdf::{RangeSplit, RsGdf};

/// A deliberately COARSE aux on H: s(1.9), s(0.5) (2 functions per H). Used
/// by the Fock-guard test as "a fit on another aux basis than the SCF's" (its
/// energy differs from the ET-sp fit at the 1e-3 level).
fn coarse_aux() -> BasisSet {
    let shells = [1.9, 0.5]
        .iter()
        .map(|&a| Shell {
            l: 0,
            pure: false,
            exponents: vec![a],
            coefficients: vec![1.0],
        })
        .collect();
    let mut m = HashMap::new();
    m.insert(1, shells);
    BasisSet {
        name: "coarse-aux-H".into(),
        shells: m,
        ecps: HashMap::new(),
    }
}

/// The prototype's ET-sp aux on H: s(4.7, 1.9, 0.75, 0.3) + p(1.25, 0.5),
/// one unit-normalised primitive per shell (10 functions per H).
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

/// Rotation pinned OFF (see `hcore_cfg`).
fn gdf_cfg() -> RsGdfConfig {
    RsGdfConfig {
        omega: 1.0,
        exxdiv: ExxDiv::None,
        budget_bytes: Some(AMPLE),
        sr_column_rotation: SrColumnRotation::Off,
        ..Default::default()
    }
}

fn kgdf_cfg() -> KRsGdfConfig {
    KRsGdfConfig {
        gdf: gdf_cfg(),
        mutation: None,
    }
}

/// A k-point RS-GDF system: the build, its config and aux basis.
struct FitSys {
    cfg: KRsGdfConfig,
    aux: PreparedBasis,
    gdf: KRsGdf,
}

fn fitsys(ks: &KSys) -> FitSys {
    fitsys_cfg(ks, kgdf_cfg())
}

fn fitsys_cfg(ks: &KSys, cfg: KRsGdfConfig) -> FitSys {
    fitsys_with(ks, cfg, &et_sp_aux())
}

fn fitsys_with(ks: &KSys, cfg: KRsGdfConfig, aux_bs: &BasisSet) -> FitSys {
    let aux = PreparedBasis::new(ks.cell.mol(), aux_bs).unwrap();
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

    /// The SCF's J/K at the exxdiv.
    fn jk(&self, ks: &KSys, exx: ExxDiv) -> Box<dyn KPointJk + '_> {
        let vm = match exx {
            ExxDiv::None => 0.0,
            ExxDiv::Ewald => ks.mesh.madelung(&ks.cell).unwrap(),
        };
        Box::new(self.gdf.jk_builder_with_madelung(vm))
    }
}

fn fit_scfs(ks: &KSys, fs: &FitSys) -> [KScfResult; 2] {
    [
        krhf(ks, fs.jk(ks, ExxDiv::None)),
        krhf(ks, fs.jk(ks, ExxDiv::Ewald)),
    ]
}

fn ks_rhf_fit(
    ks: &KSys,
    fs: &FitSys,
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
        KStressJk::RsGdf(fs.src()),
        scf,
        exx,
        &scfg(m),
    )
    .expect("kpoint_rhf_stress (RS-GDF)")
}

/// Gamma RS-GDF setup of a cell (the explicit supercell, or the cell itself).
struct GFit {
    cell: Cell,
    prep: PreparedBasis,
    hc: PeriodicHcore,
    aux: PreparedBasis,
    gdf: RsGdf,
}

fn gfit(cell: Cell, bs: &BasisSet) -> GFit {
    gfit_aux(cell, bs, &et_sp_aux(), &gdf_cfg())
}

fn gfit_aux(cell: Cell, bs: &BasisSet, aux_bs: &BasisSet, cfg: &RsGdfConfig) -> GFit {
    let prep = prep_for(&cell, bs);
    let hc = periodic_hcore(&cell, &prep, &hcore_cfg()).expect("gamma hcore");
    let aux = PreparedBasis::new(cell.mol(), aux_bs).unwrap();
    let gdf = RsGdf::build_for_gradient(&cell, &prep, &aux, &hc.s, cfg).expect("RsGdf");
    GFit {
        cell,
        prep,
        hc,
        aux,
        gdf,
    }
}

fn gamma_fit_stress(g: &GFit, scf: &ScfResult, exx: ExxDiv) -> GammaStress {
    let src = RsGdfGradSource {
        gdf: &g.gdf,
        aux: &g.aux,
        aux_jac: None,
    };
    let cfg = GammaStressConfig {
        budget_bytes: Some(AMPLE),
        ..Default::default()
    };
    if scf.spin == Spin::Restricted {
        ferric_pbc::stress::gamma_rhf_stress_rsgdf(
            &g.cell,
            &g.prep,
            &hcore_cfg(),
            &g.hc,
            &src,
            scf,
            exx,
            &cfg,
        )
    } else {
        ferric_pbc::stress::gamma_uhf_stress_rsgdf(
            &g.cell,
            &g.prep,
            &hcore_cfg(),
            &g.hc,
            &src,
            scf,
            exx,
            &cfg,
        )
    }
    .expect("gamma rsgdf stress")
}

/// Bar of the fitted 1x1x1 = Gamma anchor (the same energy functional built
/// two ways). Measured floor:
/// total 6.5e-14, worst single part 1.1e-12 (H2 RHF), 2.6e-13 (H3 UHF); the
/// smallest mutant miss at 1x1x1 is printed by the test (>= 3e-2), so the bar
/// sits 10x above the floor and >= 9 orders below the mutants.
const GAMMA_FIT_BAR: f64 = 1e-11;

fn fit_parts(p: &ferric_pbc::kstress::KStressParts) -> [(&'static str, Mat3); 7] {
    [
        ("J3 SR", p.fit_j3_sr),
        ("J3 LR", p.fit_j3_lr),
        ("J2 SR", p.fit_j2_sr),
        ("J2 LR", p.fit_j2_lr),
        ("G0 overlap", p.fit_g0_overlap),
        ("J3 G0 volume", p.fit_j3_g0_volume),
        ("J2 G0 volume", p.fit_j2_g0_volume),
    ]
}

fn gamma_fit_parts(p: &ferric_pbc::stress::GammaStressParts) -> [Mat3; 7] {
    [
        p.fit_j3_sr,
        p.fit_j3_lr,
        p.fit_j2_sr,
        p.fit_j2_lr,
        p.fit_g0_overlap,
        p.fit_j3_g0_volume,
        p.fit_j2_g0_volume,
    ]
}

/// PHYSICS: at 1x1x1 the k RS-GDF stress is the Gamma RS-GDF stress, term by
/// term (SR3, SR2, LR3, LR2, G = 0 overlap, both G = 0 volumes). ARTIFACT: the
/// two are independent constructions (half-sphere real LR vs full-sphere
/// complex per-q LR; one Y vs per-(l, t) bins; `Y q` vs `Z`-based `M_g0`), but
/// the anchor is BLIND to phases, 1/N_k, q != 0 (only the q = 0 class exists)
/// and the supercell Madelung; the 1x1x3 supercell anchor covers those.
#[test]
fn one_point_mesh_rsgdf_stress_is_the_gamma_rsgdf_stress() {
    let bs = pyscf_sto3g_h();
    let ks = ksys(h2_cell_g(), &bs, [1, 1, 1]);
    let fs = fitsys(&ks);
    let scfs = fit_scfs(&ks, &fs);
    let g = gfit(h2_cell_g(), &bs);
    for exx in EXX {
        let scf = &scfs[exx_idx(exx)];
        let gscf = gamma_restricted(real_of(&scf.densities[0]));
        let k = ks_rhf_fit(&ks, &fs, scf, exx, None);
        let gs = gamma_fit_stress(&g, &gscf, exx);
        report(&format!("H2 RS-GDF 1x1x1 {exx:?}"), &k);
        let total = max_mat(&k.de_deps, &gs.de_deps);
        eprintln!("  H2 RS-GDF 1x1x1 {exx:?}: |dE/dε k − Gamma| {total:.2e}");
        for ((name, kp), gp) in fit_parts(&k.parts).iter().zip(gamma_fit_parts(&gs.parts)) {
            let d = max_mat(kp, &gp);
            eprintln!(
                "    {name}: |k − Gamma| {d:.2e} (|part| {:.2e})",
                max_mat(kp, &ZERO)
            );
            assert!(d <= GAMMA_FIT_BAR, "{exx:?} {name}: {d:e}");
        }
        assert!(total <= GAMMA_FIT_BAR, "{exx:?}: {total:e}");
        assert!(k.antisymmetry() <= ANTISYM_BAR);
        if exx == ExxDiv::None {
            for m in [
                KStressMutation::NoMetric,
                KStressMutation::NoAuxFt3,
                KStressMutation::NoJ2G0Vol,
                KStressMutation::NoJ3G0Vol,
                KStressMutation::NoG0,
            ] {
                let x = ks_rhf_fit(&ks, &fs, scf, exx, Some(m));
                let miss = max_mat(&x.de_deps, &gs.de_deps);
                eprintln!("  1x1x1 mutant {m:?}: miss vs Gamma {miss:.2e}");
                assert!(miss > MUTANT_BAR, "{m:?} not caught ({miss:e})");
            }
        }
    }
}

// ----------------------------------------------- supercell anchor (RS-GDF)

/// Bar of the fitted supercell anchor (`dE/dε` per cell). Measured 2.1e-12
/// (none) / 2.3e-12 (ewald) on H2 1x1x3; the smallest sharp mutant miss is
/// 1.3e-2.
const SC_FIT_BAR: f64 = 1e-10;

/// Fit-specific term defects (all production-wired in `kstress`).
const FIT_MUTANTS: [KStressMutation; 5] = [
    KStressMutation::FitAuxPhase,
    KStressMutation::NoMetric,
    KStressMutation::NoAuxFt3,
    KStressMutation::NoJ2G0Vol,
    KStressMutation::NoJ3G0Vol,
];
/// Mutants that change only the diagonal by construction.
const FIT_DIAGONAL_ONLY: [KStressMutation; 2] =
    [KStressMutation::NoJ2G0Vol, KStressMutation::NoJ3G0Vol];

struct FitAnchor {
    ks: KSys,
    fs: FitSys,
    scfs: [KScfResult; 2],
    /// Gamma supercell `dE/dε / N` per exxdiv.
    refs: [Mat3; 2],
    /// Its seven fitted parts (`fit_parts` order) per exxdiv, `/ N`.
    ref_parts: [[Mat3; 7]; 2],
}

/// Miss of a mutated run against the supercell: all nine components of the
/// total, the off-diagonal of the total, and the worst of the seven FITTED
/// PARTS individually (a part-level defect whose total is hidden by aux
/// stationarity is a sharp miss here).
struct Miss {
    total: f64,
    off: f64,
    part: f64,
}

/// Supercell anchor of an RS-GDF k-RHF system (H2 or any closed-shell cell):
/// `dE/dε_k = dE/dε_sc/N`, both exxdiv, term by term; returns the setup and
/// the references for the mutants.
fn fit_supercell_anchor(name: &str, cell: Cell, bs: &BasisSet, n: [usize; 3]) -> FitAnchor {
    fit_supercell_anchor_aux(name, cell, bs, &et_sp_aux(), n)
}

fn fit_supercell_anchor_aux(
    name: &str,
    cell: Cell,
    bs: &BasisSet,
    aux_bs: &BasisSet,
    n: [usize; 3],
) -> FitAnchor {
    fit_supercell_anchor_cfg(name, cell, bs, aux_bs, &gdf_cfg(), n)
}

fn fit_supercell_anchor_cfg(
    name: &str,
    cell: Cell,
    bs: &BasisSet,
    aux_bs: &BasisSet,
    gdf: &RsGdfConfig,
    n: [usize; 3],
) -> FitAnchor {
    fit_supercell_anchor_bars(name, cell, bs, aux_bs, gdf, n, SC_FIT_BAR)
}

/// [`fit_supercell_anchor_cfg`] with the bar of the PER-PART comparison
/// (`part_bar`; the total keeps `SC_FIT_BAR`).
fn fit_supercell_anchor_bars(
    name: &str,
    cell: Cell,
    bs: &BasisSet,
    aux_bs: &BasisSet,
    gdf: &RsGdfConfig,
    n: [usize; 3],
    part_bar: f64,
) -> FitAnchor {
    let nc = (n[0] * n[1] * n[2]) as f64;
    let ks = ksys(cell.clone(), bs, n);
    let kcfg = KRsGdfConfig {
        gdf: *gdf,
        mutation: None,
    };
    let fs = fitsys_with(&ks, kcfg, aux_bs);
    let scfs = fit_scfs(&ks, &fs);
    let g = gfit_aux(supercell(&cell, n, 1), bs, aux_bs, gdf);
    let mut refs = [ZERO; 2];
    let mut ref_parts = [[ZERO; 7]; 2];
    for exx in EXX {
        let i = exx_idx(exx);
        let gscf = gamma_restricted(unfold(&ks.mesh, &scfs[i].densities));
        let k = ks_rhf_fit(&ks, &fs, &scfs[i], exx, None);
        let gs = gamma_fit_stress(&g, &gscf, exx);
        refs[i] = scale3(&gs.de_deps, 1.0 / nc);
        let miss = max_mat(&k.de_deps, &refs[i]);
        report(&format!("{name} RS-GDF {n:?} {exx:?}"), &k);
        eprintln!("  {name} RS-GDF {n:?} {exx:?}: |dE/dε_k − dE/dε_sc/N| {miss:.2e}");
        // Term by term: at the same fit (Z, Wm agree to 1e-12 across the two
        // constructions) each fitted piece is individually the supercell's.
        for (pi, ((pn, kp), gp)) in fit_parts(&k.parts)
            .iter()
            .zip(gamma_fit_parts(&gs.parts))
            .enumerate()
        {
            ref_parts[i][pi] = scale3(&gp, 1.0 / nc);
            let d = max_mat(kp, &ref_parts[i][pi]);
            eprintln!("    {pn}: |k − sc/N| {d:.2e}");
            assert!(d <= part_bar, "{name} {exx:?} {pn}: {d:e}");
        }
        assert!(miss <= SC_FIT_BAR, "{name} {exx:?}: {miss:e}");
        assert!(k.antisymmetry() <= ANTISYM_BAR);
    }
    FitAnchor {
        ks,
        fs,
        scfs,
        refs,
        ref_parts,
    }
}

impl FitAnchor {
    /// [`Miss`] of a mutated run against the supercell.
    fn mutant(&self, exx: ExxDiv, m: KStressMutation) -> Miss {
        let i = exx_idx(exx);
        let s = ks_rhf_fit(&self.ks, &self.fs, &self.scfs[i], exx, Some(m));
        let part = fit_parts(&s.parts)
            .iter()
            .zip(&self.ref_parts[i])
            .map(|((_, p), r)| max_mat(p, r))
            .fold(0.0_f64, f64::max);
        Miss {
            total: max_mat(&s.de_deps, &self.refs[i]),
            off: offdiag_miss(&s.de_deps, &self.refs[i]),
            part,
        }
    }
}

/// PHYSICS: a Gamma-centred k mesh IS the diag(N) supercell, so per cell
/// `dE/dε = dE/dε_sc/N` with the RS-GDF fit built on either side. ARTIFACT: a
/// wrong phase / q sign / 1/N_k / Loewner per-q Wm / residue bin / conjugated
/// density would not reproduce an explicit supercell. NON-TRIM mesh (n_2 = 3):
/// ConjD and the q-sign defects are identities on every n_i <= 2 mesh. The k
/// defects here are the new risk of this module, so they run in CI.
#[test]
fn h2_1x1x3_rsgdf_stress_is_the_supercell_stress_per_cell() {
    let a = fit_supercell_anchor("H2", h2_cell_g(), &pyscf_sto3g_h(), [1, 1, 3]);
    let k_defects = [
        KStressMutation::NoPhase,
        KStressMutation::ConjD,
        KStressMutation::NoInvNk,
        KStressMutation::FitAuxPhase,
        KStressMutation::FitLrWrongQ,
    ];
    for m in k_defects {
        let x = a.mutant(ExxDiv::None, m);
        eprintln!(
            "  H2 RS-GDF mutant {m:?}: miss {:.2e} (off-diagonal {:.2e})",
            x.total, x.off
        );
        assert!(x.total > MUTANT_BAR, "{m:?} not caught ({:e})", x.total);
    }
}

/// The remaining fit-term defects against the same anchor, both exxdiv. The
/// volume-only mutants must leave the off-diagonal at the floor. `NoAuxFt`
/// (aux FT strain-free in J3 and J2) errs by the fit error only in the TOTAL
/// (aux stationarity: measured 5.0e-5, the Gamma prototype 7.7e-6 to 1.2e-4),
/// but each of the J3-LR and J2-LR parts misses its supercell value sharply,
/// so the per-part comparison catches it (and any other defect in the shared
/// `aux_ft_strain_shells`). `WrongQ` is an identity on the fitted path (it
/// acts on the dense exchange pairs only). Aux coverage: s and Cartesian p
/// shells here; pure-d aux is in `h2_1x1x3_rsgdf_stress_with_a_d_aux_*`.
#[test]
#[ignore = "slow (~140 s release, loaded box): 22 stress evaluations after the supercell anchor"]
fn h2_1x1x3_rsgdf_term_mutants_miss_the_supercell_anchor() {
    let a = fit_supercell_anchor("H2", h2_cell_g(), &pyscf_sto3g_h(), [1, 1, 3]);
    let terms = FIT_MUTANTS.into_iter().chain([
        KStressMutation::NoG0,
        KStressMutation::NoSrImages,
        KStressMutation::FtNoCentres,
        KStressMutation::GUnstrained,
        KStressMutation::NoVolume,
    ]);
    for m in terms {
        for exx in EXX {
            let x = a.mutant(exx, m);
            eprintln!(
                "  H2 RS-GDF mutant {m:?} {exx:?}: miss {:.2e} (off-diagonal {:.2e}, worst part {:.2e})",
                x.total, x.off, x.part
            );
            assert!(
                x.total > MUTANT_BAR,
                "{m:?} {exx:?} not caught ({:e})",
                x.total
            );
            assert!(x.part > MUTANT_BAR, "{m:?} {exx:?}: parts ({:e})", x.part);
            if FIT_DIAGONAL_ONLY.contains(&m) || m == KStressMutation::NoVolume {
                assert!(x.off <= SC_FIT_BAR, "{m:?} diagonal-only ({:e})", x.off);
            }
        }
    }
    let x = a.mutant(ExxDiv::None, KStressMutation::NoAuxFt);
    eprintln!(
        "  H2 RS-GDF mutant NoAuxFt: total miss {:.2e}, worst part {:.2e}",
        x.total, x.part
    );
    assert!(
        x.total < MUTANT_BAR,
        "total hidden by aux stationarity ({:e})",
        x.total
    );
    assert!(
        x.part > MUTANT_BAR,
        "NoAuxFt not caught per part ({:e})",
        x.part
    );
    let x = a.mutant(ExxDiv::None, KStressMutation::WrongQ);
    assert!(
        x.total <= SC_FIT_BAR,
        "WrongQ must be an identity ({:e})",
        x.total
    );
}

// ------------------------------------------------ FD of the RS-GDF k energy

/// k-RHF RS-GDF energies `[none, ewald]` per cell of a (strained) cell, mesh
/// rebuilt from the same integer numerators, plus the per-q kept metric
/// counts (the FD is only smooth while the lindep cut keeps the same count).
fn k_fit_energies(cell: &Cell, bs: &BasisSet, n: [usize; 3]) -> ([f64; 2], Vec<usize>) {
    let ks = ksys(cell.clone(), bs, n);
    let fs = fitsys(&ks);
    let kept = fs.gdf.stats().per_q.iter().map(|q| q.naux_kept).collect();
    let e = [
        krhf(&ks, fs.jk(&ks, ExxDiv::None)).energy,
        krhf(&ks, fs.jk(&ks, ExxDiv::Ewald)).energy,
    ];
    (e, kept)
}

/// PHYSICS: analytic `dE/dε` of the RS-GDF k energy vs central FD (h = 1e-4,
/// fresh SCF + fresh fit at every strained point, lattice AND atoms strained,
/// index sets frozen, k at fixed fractional coordinates), all nine
/// components, both exxdiv. ARTIFACT: the FD sees the energy's truncation
/// (frozen K sets, the energy's own screening), the supercell anchor does
/// not; a piecewise energy (a metric eigenvalue crossing `lindep`, a K at the
/// sphere edge flipping) would show as FD(h) vs FD(2h) disagreeing faster than
/// h², asserted below.
#[test]
fn h2_1x1x3_rsgdf_stress_matches_fd_of_own_k_energy_all_nine() {
    let bs = pyscf_sto3g_h();
    let n = [1, 1, 3];
    let ks = ksys(h2_cell_g(), &bs, n);
    let fs = fitsys(&ks);
    let scfs = fit_scfs(&ks, &fs);
    let kept0: Vec<usize> = fs.gdf.stats().per_q.iter().map(|q| q.naux_kept).collect();
    let e = |c: &Cell| {
        let (e, kept) = k_fit_energies(c, &bs, n);
        assert_eq!(
            kept, kept0,
            "the metric cut changed under strain (E(ε) piecewise)"
        );
        e
    };
    let fd1 = fd_all9(&ks.cell, FD_H, &e);
    let fd2 = fd_all9(&ks.cell, 2.0 * FD_H, &e);
    for exx in EXX {
        let x = exx_idx(exx);
        let s = ks_rhf_fit(&ks, &fs, &scfs[x], exx, None);
        let (m1, m2) = (max_mat(&s.de_deps, &fd1[x]), max_mat(&s.de_deps, &fd2[x]));
        eprintln!(
            "  H2 RS-GDF 1x1x3 {exx:?}: |analytic − FD| h {m1:.2e}, 2h {m2:.2e} (ratio {:.2})",
            m2 / m1
        );
        assert!(m1 <= K_FD_BAR_FIT, "{exx:?}: FD miss {m1:e}");
        assert!((3.0..5.0).contains(&(m2 / m1)), "not h²: {}", m2 / m1);
        // The FD sees the sub-1e-4 fit-error mutant the supercell anchor
        // cannot (both sides carry the same fit).
        let aux_ft = ks_rhf_fit(&ks, &fs, &scfs[x], exx, Some(KStressMutation::NoAuxFt));
        let ma = max_mat(&aux_ft.de_deps, &fd1[x]);
        eprintln!("  H2 RS-GDF 1x1x3 {exx:?} mutant NoAuxFt: |mut − FD| {ma:.2e}");
        assert!(ma > 1e-5, "NoAuxFt not caught by FD ({ma:e})");
    }
}

/// Measured floor 1.02e-8 (h = 1e-4: the h² f‴/6 truncation of the central
/// FD; 2h gives 4.31e-8, ratio 4.21, identical for both exxdiv); the smallest
/// sharp mutant miss is 1.3e-2 and the blind-on-supercell `NoAuxFt` misses
/// by 5.0e-5.
const K_FD_BAR_FIT: f64 = 5e-8;

// ----------------------------------------------------------- RS-GDF UHF

fn ks_uhf_fit(
    ks: &KSys,
    fs: &FitSys,
    scf: &KUScfResult,
    exx: ExxDiv,
    m: Option<KStressMutation>,
) -> KStress {
    kpoint_uhf_stress(
        &ks.cell,
        &ks.prep,
        &ks.mesh,
        &hcore_cfg(),
        &ks.hk,
        KStressJk::RsGdf(fs.src()),
        scf,
        exx,
        &scfg(m),
    )
    .expect("kpoint_uhf_stress (RS-GDF)")
}

fn fit_uscfs(ks: &KSys, fs: &FitSys, na: usize, nb: usize) -> [KUScfResult; 2] {
    [
        kuhf(ks, fs.jk(ks, ExxDiv::None), na, nb),
        kuhf(ks, fs.jk(ks, ExxDiv::Ewald), na, nb),
    ]
}

/// PHYSICS/ARTIFACT as the RHF 1x1x1 anchor, on the H3 doublet (per-spin
/// exchange pairs `D_s Z^H D_s`, spin-summed Coulomb).
#[test]
fn one_point_mesh_rsgdf_uhf_stress_is_the_gamma_rsgdf_stress() {
    let bs = pyscf_sto3g_h();
    let ks = ksys(h3_cell(), &bs, [1, 1, 1]);
    let fs = fitsys(&ks);
    let scfs = fit_uscfs(&ks, &fs, 2, 1);
    let g = gfit(h3_cell(), &bs);
    for exx in EXX {
        let scf = &scfs[exx_idx(exx)];
        let gscf = gamma_unrestricted(
            real_of(&scf.density_alpha[0]),
            real_of(&scf.density_beta[0]),
        );
        let k = ks_uhf_fit(&ks, &fs, scf, exx, None);
        let gs = gamma_fit_stress(&g, &gscf, exx);
        let total = max_mat(&k.de_deps, &gs.de_deps);
        report(&format!("H3 UHF RS-GDF 1x1x1 {exx:?}"), &k);
        eprintln!("  H3 UHF RS-GDF 1x1x1 {exx:?}: |dE/dε k − Gamma| {total:.2e}");
        for ((name, kp), gp) in fit_parts(&k.parts).iter().zip(gamma_fit_parts(&gs.parts)) {
            let d = max_mat(kp, &gp);
            eprintln!("    {name}: |k − Gamma| {d:.2e}");
            assert!(d <= GAMMA_FIT_BAR, "{exx:?} {name}: {d:e}");
        }
        assert!(total <= GAMMA_FIT_BAR, "{exx:?}: {total:e}");
    }
}

/// closed-shell kUHF(D/2, D/2) ≡ kRHF(D) on the fitted path.
#[test]
fn closed_shell_kuhf_rsgdf_stress_is_the_krhf_stress() {
    let bs = pyscf_sto3g_h();
    let ks = ksys(h2_cell_g(), &bs, [1, 1, 3]);
    let fs = fitsys(&ks);
    let rs = fit_scfs(&ks, &fs);
    let mut us = fit_uscfs(&ks, &fs, 1, 1);
    for exx in EXX {
        let (r, u) = (&rs[exx_idx(exx)], &mut us[exx_idx(exx)]);
        let half: Vec<Array2<Complex64>> =
            r.densities.iter().map(|d| d.mapv(|z| z * 0.5)).collect();
        u.density_alpha = half.clone();
        u.density_beta = half;
        let sr = ks_rhf_fit(&ks, &fs, r, exx, None);
        let su = ks_uhf_fit(&ks, &fs, u, exx, None);
        let d = max_mat(&sr.de_deps, &su.de_deps);
        eprintln!("  H2 RS-GDF 1x1x3 {exx:?}: |σ_kUHF(D/2,D/2) − σ_kRHF(D)| {d:.2e}");
        assert!(d <= IDENTITY_BAR, "{exx:?}: {d:e}");
    }
}

/// H3 doublet 1x1x3: the supercell (9 atoms, multiplicity 4) Gamma RS-GDF UHF
/// stress per cell, both exxdiv, and the UHF-specific mutants.
#[test]
#[ignore = "slow (~115 s release, loaded box): 9-atom supercell Gamma RS-GDF UHF stress x2 + k stress"]
fn h3_uhf_1x1x3_rsgdf_stress_is_the_supercell_stress_per_cell() {
    let bs = pyscf_sto3g_h();
    let n = [1, 1, 3];
    let ks = ksys(h3_cell(), &bs, n);
    let fs = fitsys(&ks);
    let scfs = fit_uscfs(&ks, &fs, 2, 1);
    let g = gfit(supercell(&h3_cell(), n, 4), &bs);
    let mut refs = [ZERO; 2];
    for exx in EXX {
        let i = exx_idx(exx);
        let gscf = gamma_unrestricted(
            unfold(&ks.mesh, &scfs[i].density_alpha),
            unfold(&ks.mesh, &scfs[i].density_beta),
        );
        let k = ks_uhf_fit(&ks, &fs, &scfs[i], exx, None);
        let gs = gamma_fit_stress(&g, &gscf, exx);
        refs[i] = scale3(&gs.de_deps, 1.0 / 3.0);
        let miss = max_mat(&k.de_deps, &refs[i]);
        report(&format!("H3 UHF RS-GDF {n:?} {exx:?}"), &k);
        eprintln!("  H3 UHF RS-GDF {n:?} {exx:?}: |dE/dε_k − dE/dε_sc/N| {miss:.2e}");
        assert!(miss <= SC_FIT_BAR, "{exx:?}: {miss:e}");
    }
    // CrossSpinK acts on the dense exchange pairs only: an identity here.
    let s = ks_uhf_fit(
        &ks,
        &fs,
        &scfs[0],
        ExxDiv::None,
        Some(KStressMutation::CrossSpinK),
    );
    assert!(
        max_mat(&s.de_deps, &refs[0]) <= SC_FIT_BAR,
        "CrossSpinK must be an identity"
    );
    for m in [
        KStressMutation::SwapSpinFock,
        KStressMutation::NoPhase,
        KStressMutation::ConjD,
        KStressMutation::FitAuxPhase,
        KStressMutation::FitLrWrongQ,
    ] {
        let s = ks_uhf_fit(&ks, &fs, &scfs[0], ExxDiv::None, Some(m));
        let miss = max_mat(&s.de_deps, &refs[0]);
        eprintln!("  H3 UHF RS-GDF mutant {m:?}: miss {miss:.2e}");
        assert!(miss > MUTANT_BAR, "{m:?} not caught ({miss:e})");
    }
}

// ----------------------------------------------------------- guards

/// Typed refusal of the fitted path: an ECP in the hcore the basis does not
/// carry, on the unsplit AND the range-split source (a range-split build
/// itself is supported). Column-rotation request refusals live in
/// `pbc_sr_rotation_kderiv.rs`.
#[test]
fn rsgdf_stress_refuses_ecp() {
    let bs = pyscf_sto3g_h();
    let ks = ksys(h2_cell_g(), &bs, [1, 1, 3]);
    let fs = fitsys(&ks);
    let scf = krhf(&ks, fs.jk(&ks, ExxDiv::None));
    let run = |hk: &PeriodicHcoreK, hc: &PeriodicHcoreConfig, src: KRsGdfGradSource<'_>| {
        kpoint_rhf_stress(
            &ks.cell,
            &ks.prep,
            &ks.mesh,
            hc,
            hk,
            KStressJk::RsGdf(src),
            &scf,
            ExxDiv::None,
            &scfg(None),
        )
        .unwrap_err()
        .to_string()
    };
    // ECP in the hcore.
    let mut hk_ecp = ks.hk.clone();
    hk_ecp.v_ecp = Some(hk_ecp.h.clone());
    let e = run(&hk_ecp, &hcore_cfg(), fs.src());
    assert!(e.contains("carries V_ECP but the basis"), "{e}");
    // The same refusal on a range-split source (the SCF density is the
    // unsplit one: the refusal fires before the Fock guard).
    let sp = fitsys_cfg(&ks, split_kcfg(1.0));
    let e = run(&hk_ecp, &hcore_cfg(), sp.src());
    assert!(e.contains("carries V_ECP but the basis"), "{e}");
}

/// The Fock-consistency guard on the fitted source: it must refuse an SCF run
/// at another exxdiv, a fit on another aux basis than the SCF's, and an SCF
/// whose stored Fock was perturbed by 1e-6, and ACCEPT the matching input.
/// Both sides of the bar: matching inputs 1e-15-level (printed), the wrong
/// exxdiv / wrong fit orders of magnitude above `FOCK_CONSISTENCY_BAR`.
#[test]
fn rsgdf_stress_fock_guard_refuses_inconsistent_inputs() {
    let bs = pyscf_sto3g_h();
    let ks = ksys(h2_cell_g(), &bs, [1, 1, 3]);
    let fs = fitsys(&ks);
    let scf = krhf(&ks, fs.jk(&ks, ExxDiv::None));
    let ok = ks_rhf_fit(&ks, &fs, &scf, ExxDiv::None, None);
    eprintln!(
        "  matching RS-GDF inputs: Fock mismatch {:.2e}",
        ok.fock_mismatch
    );
    assert!(ok.fock_mismatch < 1e-12);
    // Wrong exxdiv (the SCF ran without the Madelung shift).
    let stress = |scf: &KScfResult, fs: &FitSys, exx: ExxDiv| {
        kpoint_rhf_stress(
            &ks.cell,
            &ks.prep,
            &ks.mesh,
            &hcore_cfg(),
            &ks.hk,
            KStressJk::RsGdf(fs.src()),
            scf,
            exx,
            &scfg(None),
        )
    };
    let e = stress(&scf, &fs, ExxDiv::Ewald).unwrap_err().to_string();
    eprintln!("  wrong exxdiv: {e}");
    assert!(e.contains("Fock matrix rebuilt"), "{e}");
    // A different fit (another aux basis) than the SCF's. (Another omega is
    // NOT refused, see `rsgdf_stress_accepts_a_fit_at_another_omega`.)
    let other = fitsys_with(&ks, kgdf_cfg(), &coarse_aux());
    let e = stress(&scf, &other, ExxDiv::None).unwrap_err().to_string();
    eprintln!("  wrong aux basis: {e}");
    assert!(e.contains("Fock matrix rebuilt"), "{e}");
    // A stored Fock perturbed by 1e-6 (above the 1e-8 bar).
    let mut bad = scf.clone();
    bad.fock[0][(0, 0)] += Complex64::new(1e-6, 0.0);
    let e = stress(&bad, &fs, ExxDiv::None).unwrap_err().to_string();
    assert!(e.contains("Fock matrix rebuilt"), "{e}");
}

// ------------------------------------- shifted (MP) mesh and triclinic cell

/// k-RHF RS-GDF energies `[none, ewald]` on `(cell, bs, n, centring)` with the
/// per-q kept metric counts.
fn k_fit_energies_c(
    cell: &Cell,
    bs: &BasisSet,
    n: [usize; 3],
    centring: MeshCentring,
) -> ([f64; 2], Vec<usize>) {
    let ks = ksys_c(cell.clone(), bs, n, centring);
    let fs = fitsys(&ks);
    let kept = fs.gdf.stats().per_q.iter().map(|q| q.naux_kept).collect();
    let e = [
        krhf(&ks, fs.jk(&ks, ExxDiv::None)).energy,
        krhf(&ks, fs.jk(&ks, ExxDiv::Ewald)).energy,
    ];
    (e, kept)
}

/// FD anchor of an RS-GDF k-RHF system at h and 2h, all nine components, both
/// exxdiv; returns the misses `[h, 2h]` per exxdiv.
fn fit_fd_anchor(
    name: &str,
    cell: Cell,
    bs: &BasisSet,
    n: [usize; 3],
    c: MeshCentring,
) -> (KSys, FitSys, [KScfResult; 2], [Mat3; 2]) {
    let ks = ksys_c(cell, bs, n, c);
    let fs = fitsys(&ks);
    let scfs = fit_scfs(&ks, &fs);
    let kept0: Vec<usize> = fs.gdf.stats().per_q.iter().map(|q| q.naux_kept).collect();
    let e = |cell: &Cell| {
        let (e, kept) = k_fit_energies_c(cell, bs, n, c);
        assert_eq!(
            kept, kept0,
            "the metric cut changed under strain (E(ε) piecewise)"
        );
        e
    };
    let fd1 = fd_all9(&ks.cell, FD_H, &e);
    let fd2 = fd_all9(&ks.cell, 2.0 * FD_H, &e);
    for exx in EXX {
        let x = exx_idx(exx);
        let s = ks_rhf_fit(&ks, &fs, &scfs[x], exx, None);
        let (m1, m2) = (max_mat(&s.de_deps, &fd1[x]), max_mat(&s.de_deps, &fd2[x]));
        eprintln!(
            "  {name} RS-GDF {n:?} {exx:?}: |analytic − FD| h {m1:.2e}, 2h {m2:.2e} (ratio {:.2}), \
             antisym(σ) {:.1e}",
            m2 / m1,
            s.antisymmetry()
        );
        assert!(m1 <= K_FD_BAR_FIT, "{name} {exx:?}: FD miss {m1:e}");
        assert!(s.antisymmetry() <= ANTISYM_BAR);
    }
    (ks, fs, scfs, fd1)
}

/// PHYSICS: a Monkhorst-Pack even mesh (k = ±b3/4 for N = 2) is a twisted
/// supercell with NO explicit-supercell anchor, so only the FD of ferric's
/// own k energy pins it. It is also the case where the pair-image residue
/// modulus (4) differs from the aux-image one (N = 2): a bin mix-up in either
/// dimension of the (l, t) weights fails here. Non-TRIM (2k ∉ G).
#[test]
#[ignore = "slow (~75 s release on a loaded box): 36 strained RS-GDF builds + 72 k-SCFs (h and 2h)"]
fn h2_mp_1x1x2_rsgdf_stress_matches_fd_of_own_k_energy_all_nine() {
    let c = MeshCentring::MonkhorstPack;
    let (ks, fs, scfs, fd1) = fit_fd_anchor("H2 MP", h2_cell_g(), &pyscf_sto3g_h(), [1, 1, 2], c);
    assert_eq!(ks.mesh.residue_moduli(), [1, 1, 4]);
    for exx in EXX {
        let x = exx_idx(exx);
        let s0 = ks_rhf_fit(&ks, &fs, &scfs[x], exx, None);
        for m in [KStressMutation::NoPhase, KStressMutation::NoInvNk] {
            let mm = max_mat(
                &ks_rhf_fit(&ks, &fs, &scfs[x], exx, Some(m)).de_deps,
                &fd1[x],
            );
            eprintln!("  H2 MP 1x1x2 RS-GDF {exx:?} mutant {m:?}: miss {mm:.2e}");
            assert!(mm > MUTANT_BAR, "{m:?} {exx:?}: {mm:e}");
        }
        // BLIND SPOT, asserted: every q = k' - k of this mesh is a TRIM
        // (2q is a reciprocal-lattice vector), so e^{+iq.T} = e^{-iq.T} and
        // {G + q} = {G - q} up to the sphere edge.
        for m in [KStressMutation::FitAuxPhase, KStressMutation::FitLrWrongQ] {
            let d = max_mat(
                &ks_rhf_fit(&ks, &fs, &scfs[x], exx, Some(m)).de_deps,
                &s0.de_deps,
            );
            eprintln!("  H2 MP 1x1x2 RS-GDF {exx:?} {m:?}: |mut − ref| {d:.2e} (TRIM q: blind)");
            assert!(d <= 1e-9, "{m:?} {exx:?} should be blind on TRIM q ({d:e})");
        }
    }
}

/// A triclinic lattice (TRI_A: no cell vector along a Cartesian axis, so a
/// Cartesian/fractional index mix-up in the K sets, residue bins or q phases
/// cannot cancel as it can in a cubic cell) with two s-only hydrogens, 1x1x3.
#[test]
#[ignore = "slow (~40 s release): triclinic H2 1x1x3 RS-GDF, 36 strained builds + 72 k-SCFs"]
fn triclinic_h2_1x1x3_rsgdf_stress_matches_fd_of_own_k_energy_all_nine() {
    let cell = cell_at(&[TRI_ATOMS[0], TRI_ATOMS[1]], TRI_A, 1);
    let (ks, fs, scfs, fd1) = fit_fd_anchor(
        "tri H2",
        cell,
        &pyscf_sto3g_h(),
        [1, 1, 3],
        MeshCentring::Gamma,
    );
    for exx in EXX {
        let x = exx_idx(exx);
        for m in [KStressMutation::NoPhase, KStressMutation::FitLrWrongQ] {
            let mm = max_mat(
                &ks_rhf_fit(&ks, &fs, &scfs[x], exx, Some(m)).de_deps,
                &fd1[x],
            );
            eprintln!("  tri H2 RS-GDF {exx:?} mutant {m:?}: miss {mm:.2e}");
            assert!(mm > MUTANT_BAR, "{m:?} {exx:?}: {mm:e}");
        }
    }
}

/// An ACTIVE metric lindep cut with a NEAR-dependent dropped mode: every atom
/// carries s(0.75) and s(0.76) aux shells, so each q class has one small but
/// NONZERO metric eigenvalue per atom (asserted below the cut, the rest above
/// it). An exact duplicate would make the dropped mode an exact null vector of
/// `J3`, multiplying the kept-dropped Loewner weight by zero; here that branch
/// carries weight (at the ~1e-8 level, see below), and `FitNoLoewnerKd` (kept-dropped entries zeroed) must miss
/// the supercell anchor. The Gamma supercell side drops the same modes.
#[test]
#[ignore = "slow (~50 s release): supercell anchor with a near-duplicate aux shell"]
fn h2_1x1x3_rsgdf_stress_with_an_active_lindep_cut_is_the_supercell_stress() {
    let mut aux = et_sp_aux();
    let mut near = aux.shells[&1][2].clone();
    near.exponents = vec![0.95];
    aux.shells.get_mut(&1).unwrap().push(near);
    let gdf = RsGdfConfig {
        lindep: NEAR_DUP_LINDEP,
        ..gdf_cfg()
    };
    let a = fit_supercell_anchor_cfg(
        "H2 near-dup aux",
        h2_cell_g(),
        &pyscf_sto3g_h(),
        &aux,
        &gdf,
        [1, 1, 3],
    );
    let per_q = &a.fs.gdf.stats().per_q;
    for q in per_q {
        eprintln!(
            "  q class {}: dropped {} (min eig {:.2e}), kept {}",
            q.iq, q.n_dropped, q.metric_eig_min, q.naux_kept
        );
        assert_eq!(q.n_dropped, 2, "one near-dependent mode per atom");
        assert!(
            q.metric_eig_min > 1e-14 && q.metric_eig_min < NEAR_DUP_LINDEP,
            "the dropped mode must be small but NONZERO"
        );
    }
    for m in [
        KStressMutation::FitNoLoewnerKd,
        KStressMutation::NoMetric,
        KStressMutation::NoPhase,
    ] {
        let x = a.mutant(ExxDiv::None, m);
        eprintln!(
            "  H2 near-dup aux mutant {m:?}: miss {:.2e} (worst part {:.2e})",
            x.total, x.part
        );
        // The kept-dropped weight multiplies the dropped component of H, which
        // is small for a near-dependent mode, so this defect is inherently a
        // ~1e-8 effect on the total (measured 3.6e-8; worst part 3.2e-6) -
        // far above the 1e-12 anchor floor but below MUTANT_BAR.
        let bar = if m == KStressMutation::FitNoLoewnerKd {
            100.0 * SC_FIT_BAR
        } else {
            MUTANT_BAR
        };
        assert!(x.total > bar, "{m:?} not caught ({:e})", x.total);
    }
}

/// Lindep of the near-duplicate test: above the small eigenvalue of the
/// s(0.75)/s(0.76) pair, below the rest of the spectrum (asserted).
const NEAR_DUP_LINDEP: f64 = 1e-3;

// -------------------------------------------------- aux l >= 2, omega, rotation

/// ET-sp aux plus one PURE d shell (exponent 0.8): 15 functions per H, the
/// l = 2 aux case of `aux_ft_strain_shells` and of the SR 3-centre / metric
/// derivative blocks (the production path at k; the unit test covers the FT
/// alone).
fn et_spd_aux() -> BasisSet {
    let mut aux = et_sp_aux();
    aux.shells.get_mut(&1).unwrap().push(Shell {
        l: 2,
        pure: true,
        exponents: vec![0.8],
        coefficients: vec![1.0],
    });
    aux
}

/// PHYSICS/ARTIFACT as the main supercell anchor, with a pure-d aux shell so
/// the l = 2 aux-FT strain (`aux_ft_strain_shells`, pure cart2sph) and the l = 2
/// SR derivative blocks are on the production path at k. `NoAuxFt` must be a
/// sharp per-part miss here too.
#[test]
#[ignore = "slow (~45 s release): supercell anchor with a pure-d aux shell"]
fn h2_1x1x3_rsgdf_stress_with_a_d_aux_is_the_supercell_stress() {
    let a = fit_supercell_anchor_aux(
        "H2 spd-aux",
        h2_cell_g(),
        &pyscf_sto3g_h(),
        &et_spd_aux(),
        [1, 1, 3],
    );
    for m in [KStressMutation::NoAuxFt, KStressMutation::NoAuxFt3] {
        let x = a.mutant(ExxDiv::None, m);
        eprintln!(
            "  H2 spd-aux mutant {m:?}: total miss {:.2e}, worst part {:.2e}",
            x.total, x.part
        );
        assert!(
            x.part > MUTANT_BAR,
            "{m:?} not caught per part ({:e})",
            x.part
        );
    }
}

/// The energy is omega-independent (the fit of the same aux basis, SR + LR
/// split at another omega), so an SCF run at omega 1.0 and a stress with a fit
/// at omega 1.3 is NOT inconsistent: the guard must accept it (mismatch far
/// below the bar) and the stress must agree with the omega-1.0 stress to the
/// measured level (printed; the two splits differ by the SR/LR truncation).
#[test]
fn rsgdf_stress_accepts_a_fit_at_another_omega() {
    let bs = pyscf_sto3g_h();
    let ks = ksys(h2_cell_g(), &bs, [1, 1, 3]);
    let fs = fitsys(&ks);
    let scf = krhf(&ks, fs.jk(&ks, ExxDiv::None));
    let base = ks_rhf_fit(&ks, &fs, &scf, ExxDiv::None, None);
    let other = fitsys_cfg(
        &ks,
        KRsGdfConfig {
            gdf: RsGdfConfig {
                omega: 1.3,
                ..gdf_cfg()
            },
            mutation: None,
        },
    );
    let s = ks_rhf_fit(&ks, &other, &scf, ExxDiv::None, None);
    let d = max_mat(&s.de_deps, &base.de_deps);
    eprintln!(
        "  omega 1.3 fit on an omega 1.0 SCF: Fock mismatch {:.2e}, |dE/dε − dE/dε(omega 1.0)| {d:.2e}",
        s.fock_mismatch
    );
    assert!(s.fock_mismatch < FOCK_CONSISTENCY_BAR);
    assert!(d <= OMEGA_STRESS_BAR, "{d:e}");
}

/// Measured max abs difference of dE/dε between the omega 1.0 and 1.3 fits: 2.6e-12 (H2 1x1x3).
const OMEGA_STRESS_BAR: f64 = 1e-10;

/// Auto column rotation: the energy builds and the stress walks rotate the SR
/// columns (`Auto`) and agree with the unrotated pipeline to the screening
/// precision. On a p-bearing, GENERALLY CONTRACTED k cell (cc-pVDZ / cc-pvdz-ri H2; a
/// segmented basis such as STO-3G or the s+p test basis has no rotating
/// column, so Auto is bit-identical to Off there), the stress
/// of a fully-Auto pipeline (hcore, fit, SCF) vs the fully-Off one is printed
/// together with the energy difference (zero energy difference would mean Auto
/// resolved to off here).
#[test]
#[ignore = "slow (~80 s release, loaded box): H2 cc-pVDZ 1x1x2 RS-GDF, two full pipelines"]
fn rsgdf_stress_under_auto_rotation_agrees_with_off_on_a_p_cell() {
    let bs = ferric_core::basis::bundled("cc-pvdz").unwrap();
    let aux_bs = ferric_core::basis::bundled("cc-pvdz-ri").unwrap();
    let n = [1, 1, 2];
    let run = |auto: bool| {
        let cell = h2_cell_g();
        let prep = prep_for(&cell, &bs);
        let mesh = KPointMesh::new(&cell, n, MeshCentring::Gamma).unwrap();
        let hc = if auto {
            hcore_cfg().with_sr_column_rotation(SrColumnRotation::Auto)
        } else {
            hcore_cfg()
        };
        let hk = periodic_hcore_kpts(&cell, &prep, &mesh, &hc).unwrap();
        let ks = KSys {
            cell,
            prep,
            mesh,
            hk,
        };
        let cfg = KRsGdfConfig {
            gdf: RsGdfConfig {
                sr_column_rotation: if auto {
                    SrColumnRotation::Auto
                } else {
                    SrColumnRotation::Off
                },
                ..gdf_cfg()
            },
            mutation: None,
        };
        let fs = fitsys_with(&ks, cfg, &aux_bs);
        let scf = krhf(&ks, fs.jk(&ks, ExxDiv::None));
        let st = kpoint_rhf_stress(
            &ks.cell,
            &ks.prep,
            &ks.mesh,
            &hc,
            &ks.hk,
            KStressJk::RsGdf(fs.src()),
            &scf,
            ExxDiv::None,
            &scfg(None),
        )
        .expect("stress");
        // The stress walked the columns the energy rotated (nonzero under Auto).
        assert_eq!(st.hcore_rotated_columns, ks.hk.sr_rotated_columns);
        assert_eq!(st.hcore_rotated_columns > 0, auto);
        (scf.energy, st)
    };
    let (e_auto, s_auto) = run(true);
    let (e_off, s_off) = run(false);
    let d = max_mat(&s_auto.de_deps, &s_off.de_deps);
    eprintln!(
        "  H2 cc-pVDZ 1x1x2 RS-GDF: |E(Auto) − E(Off)| {:.2e}, |dE/dε(Auto) − dE/dε(Off)| {d:.2e}",
        (e_auto - e_off).abs()
    );
    assert!(d <= AUTO_ROT_BAR, "{d:e}");
}

/// Measured |Δ dE/dε| rotated pipeline vs unrotated pipeline (hcore precision
/// 1e-14, default fit precision): 3.7e-13 (energies differ by 7.9e-14, so the
/// rotation is active and the stress walks are rotated too).
const AUTO_ROT_BAR: f64 = 1e-11;

// ------------------------------------------------------------- RS-GDF UHF FD

/// k-UHF RS-GDF energy (exxdiv none) of the H3 doublet (2, 1) on a strained
/// cell, with the per-q kept counts.
fn k_fit_energy_uhf(cell: &Cell, bs: &BasisSet, n: [usize; 3]) -> (f64, Vec<usize>) {
    let ks = ksys(cell.clone(), bs, n);
    let fs = fitsys(&ks);
    let kept = fs.gdf.stats().per_q.iter().map(|q| q.naux_kept).collect();
    (kuhf(&ks, fs.jk(&ks, ExxDiv::None), 2, 1).energy, kept)
}

/// PHYSICS: analytic UHF `dE/dε` on the fitted path vs central FD (h = 1e-4,
/// fresh fit and SCF per point, all nine components, exxdiv none) of the own
/// k-UHF RS-GDF energy: a spin-specific anchor that does not rest on the
/// supercell. The fit-path spin mutant `FitCrossSpin` (exchange pairing
/// D_α J3^H D_β) and `SwapSpinFock` must miss it.
#[test]
#[ignore = "slow (~50 s release): 18 strained RS-GDF builds + 18 k-UHF SCFs on H3"]
fn h3_uhf_1x1x3_rsgdf_stress_matches_fd_of_own_k_energy_all_nine() {
    let bs = pyscf_sto3g_h();
    let n = [1, 1, 3];
    let ks = ksys(h3_cell(), &bs, n);
    let fs = fitsys(&ks);
    let scf = kuhf(&ks, fs.jk(&ks, ExxDiv::None), 2, 1);
    let kept0: Vec<usize> = fs.gdf.stats().per_q.iter().map(|q| q.naux_kept).collect();
    let mut fd = ZERO;
    for (i, j) in ALL9 {
        let at = |h: f64| {
            let (e, kept) = k_fit_energy_uhf(&strained(&ks.cell, i, j, h), &bs, n);
            assert_eq!(kept, kept0, "the metric cut changed under strain");
            e
        };
        fd[i][j] = (at(FD_H) - at(-FD_H)) / (2.0 * FD_H);
    }
    let s = ks_uhf_fit(&ks, &fs, &scf, ExxDiv::None, None);
    let miss = max_mat(&s.de_deps, &fd);
    eprintln!("  H3 UHF RS-GDF 1x1x3: |analytic − FD| {miss:.2e}");
    assert!(miss <= K_FD_BAR_FIT, "{miss:e}");
    for m in [KStressMutation::FitCrossSpin, KStressMutation::SwapSpinFock] {
        let mm = max_mat(
            &ks_uhf_fit(&ks, &fs, &scf, ExxDiv::None, Some(m)).de_deps,
            &fd,
        );
        eprintln!("  H3 UHF RS-GDF mutant {m:?}: |mut − FD| {mm:.2e}");
        assert!(mm > MUTANT_BAR, "{m:?} not caught ({mm:e})");
    }
}

// ================================================== RANGE-SPLIT RS-GDF (k)
//
// The range split moves the smooth primitives of the SR sums to G
// space with the kernel v_SR(|G + q|) at each momentum transfer. Anchors, in
// the order they were written (each BEFORE the sweep that uses it):
//  (a) 1x1x1 == the Gamma split stress, term by term;
//  (b) 1x1x3 == the explicit split supercell / N, term by term, both exxdiv;
//  (c) split k stress vs the UNSPLIT k stress (the split is an exact
//      decomposition up to its tolerance: the difference is MEASURED and the
//      bar derived from it);
//  (d) FD of the split k energy (h^2 floor), MP [1,1,2] included;
//  (e) the Fock-consistency guard on a split source; thread bit-identity
//      lives in `pbc_parallel_bitwise.rs`.
// ARTIFACT hypothesis shared by (a)-(d): "the split stress passes because it
// silently ran the unsplit code" would make split == unsplit to 1e-13; every
// test asserts the split counters (moved primitives, shrunk SR walks), and the
// mutants delete only the NEW split terms.

/// lambda = 1 at omega = 1 moves the aux primitives s(0.75), s(0.3), p(0.5)
/// and the STO-3G orbital primitive 0.1689.
fn split_gdf_cfg(lambda: f64) -> RsGdfConfig {
    RsGdfConfig {
        range_split: Some(RangeSplit::new(lambda)),
        ..gdf_cfg()
    }
}

fn split_kcfg(lambda: f64) -> KRsGdfConfig {
    KRsGdfConfig {
        gdf: split_gdf_cfg(lambda),
        mutation: None,
    }
}

/// Panics unless the split path ran, primitives moved on both sides and the
/// SR walks shrank against the unsplit build (the artifact guard).
fn assert_moved(tag: &str, gdf: &KRsGdf, unsplit: &KRsGdf) {
    let c = |name: &str| {
        gdf.stats()
            .split_counters
            .iter()
            .find(|(n, _)| *n == name)
            .map(|&(_, v)| v)
            .unwrap_or_else(|| panic!("{tag}: counter {name:?} missing (split path did not run)"))
    };
    let (os, as_) = (
        c("rsgdf split orbital prims smooth"),
        c("rsgdf split aux prims smooth"),
    );
    let (s, u) = (gdf.stats(), unsplit.stats());
    eprintln!(
        "  {tag}: moved orbital {os}, aux {as_}; SR3 {} -> {}, SR2 {} -> {}",
        u.n_sr3_triplets, s.n_sr3_triplets, u.n_sr2_pairs, s.n_sr2_pairs
    );
    assert!(os > 0 && as_ > 0, "{tag}: nothing moved");
    assert!(s.n_sr3_triplets < u.n_sr3_triplets, "{tag}: SR3 not shrunk");
    assert!(u.split_counters.is_empty(), "{tag}: unsplit ran split path");
}

/// The NEW split terms and hooks of the k stress.
const SPLIT_MUTANTS: [KStressMutation; 6] = [
    KStressMutation::SplitNoSrKernelStrain,
    KStressMutation::SplitNoSmoothPair,
    KStressMutation::SplitFullG0,
    KStressMutation::SplitNoSmoothOverlap,
    KStressMutation::SplitFlipSrSlope,
    KStressMutation::SplitNoSmoothVolume,
];

/// PHYSICS: at 1x1x1 the k split stress is the Gamma split stress, part by
/// part (SR3, SR2, LR3, LR2, G0 overlap incl. the smooth-overlap virial, both
/// G0 volumes). ARTIFACT: independent constructions (half sphere real with
/// weight 2 vs full sphere complex), but BLIND to phases, 1/N_k and q != 0.
#[test]
fn one_point_mesh_split_rsgdf_stress_is_the_gamma_split_stress() {
    let bs = pyscf_sto3g_h();
    let ks = ksys(h2_cell_g(), &bs, [1, 1, 1]);
    let fs = fitsys_cfg(&ks, split_kcfg(1.0));
    let un = fitsys(&ks);
    assert_moved("H2 1x1x1", &fs.gdf, &un.gdf);
    let scfs = fit_scfs(&ks, &fs);
    let g = gfit_aux(h2_cell_g(), &bs, &et_sp_aux(), &split_gdf_cfg(1.0));
    for exx in EXX {
        let scf = &scfs[exx_idx(exx)];
        let gscf = gamma_restricted(real_of(&scf.densities[0]));
        let k = ks_rhf_fit(&ks, &fs, scf, exx, None);
        let gs = gamma_fit_stress(&g, &gscf, exx);
        let total = max_mat(&k.de_deps, &gs.de_deps);
        eprintln!("  H2 split 1x1x1 {exx:?}: |dE/dε k − Gamma| {total:.2e}");
        for ((name, kp), gp) in fit_parts(&k.parts).iter().zip(gamma_fit_parts(&gs.parts)) {
            let d = max_mat(kp, &gp);
            eprintln!(
                "    {name}: |k − Gamma| {d:.2e} (|part| {:.2e})",
                max_mat(kp, &ZERO)
            );
            assert!(d <= GAMMA_FIT_BAR, "{exx:?} {name}: {d:e}");
        }
        assert!(total <= GAMMA_FIT_BAR, "{exx:?}: {total:e}");
        if exx == ExxDiv::None {
            for m in SPLIT_MUTANTS {
                let x = ks_rhf_fit(&ks, &fs, scf, exx, Some(m));
                let miss = max_mat(&x.de_deps, &gs.de_deps);
                eprintln!("  1x1x1 split mutant {m:?}: miss vs Gamma {miss:.2e}");
                assert!(miss > MUTANT_BAR, "{m:?} not caught ({miss:e})");
            }
        }
    }
}

// ----------------------------------------------- supercell anchor (split)

/// PHYSICS: a Gamma-centred k mesh IS the diag(N) supercell, so per cell
/// `dE/dε = dE/dε_sc/N`, here with the RANGE-SPLIT fit on both sides (the
/// k-point kernel v_SR(|G + q|) per q vs the supercell's Gamma kernel).
/// ARTIFACT: a split stress that used the Gamma kernel at every q (the
/// k-specific defect of the energy, 1.05e-3 Ha) or mis-binned the smooth
/// pairs could not reproduce an explicit supercell; the 1x1x1 anchor cannot
/// see either (q = 0 only). NON-TRIM mesh (n_2 = 3).
#[test]
fn h2_1x1x3_split_rsgdf_stress_is_the_supercell_stress_per_cell() {
    let bs = pyscf_sto3g_h();
    let a = fit_supercell_anchor_cfg(
        "H2 split",
        h2_cell_g(),
        &bs,
        &et_sp_aux(),
        &split_gdf_cfg(1.0),
        [1, 1, 3],
    );
    assert_moved("H2 1x1x3", &a.fs.gdf, &fitsys(&a.ks).gdf);
    for m in [
        KStressMutation::NoPhase,
        KStressMutation::ConjD,
        KStressMutation::FitAuxPhase,
        KStressMutation::FitLrWrongQ,
        KStressMutation::SplitNoSrKernelStrain,
        KStressMutation::SplitNoSmoothPair,
    ] {
        let x = a.mutant(ExxDiv::None, m);
        eprintln!(
            "  H2 split mutant {m:?}: miss {:.2e} (off-diagonal {:.2e}, worst part {:.2e})",
            x.total, x.off, x.part
        );
        assert!(x.total > MUTANT_BAR, "{m:?} not caught ({:e})", x.total);
    }
}

/// Every NEW split hook against the supercell anchor, both exxdiv, per part
/// (a part-level defect hidden in the total by aux stationarity is sharp
/// per part). `SplitNoSmoothVolume` and `SplitFullG0` carry diagonal volume
/// terms: asserted diagonal-only for the volume one.
#[test]
#[ignore = "slow (~90 s release, 55 s on a quiet box): 26 stress evaluations after the supercell anchor"]
fn h2_1x1x3_split_mutants_miss_the_supercell_anchor() {
    let bs = pyscf_sto3g_h();
    let a = fit_supercell_anchor_cfg(
        "H2 split",
        h2_cell_g(),
        &bs,
        &et_sp_aux(),
        &split_gdf_cfg(1.0),
        [1, 1, 3],
    );
    for m in SPLIT_MUTANTS.into_iter().chain([
        KStressMutation::NoInvNk,
        KStressMutation::NoMetric,
        KStressMutation::NoAuxFt3,
        KStressMutation::NoJ2G0Vol,
        KStressMutation::NoJ3G0Vol,
        KStressMutation::NoG0,
        KStressMutation::NoVolume,
    ]) {
        for exx in EXX {
            let x = a.mutant(exx, m);
            eprintln!(
                "  H2 split mutant {m:?} {exx:?}: miss {:.2e} (off-diagonal {:.2e}, worst part {:.2e})",
                x.total, x.off, x.part
            );
            assert!(
                x.total > MUTANT_BAR,
                "{m:?} {exx:?} not caught ({:e})",
                x.total
            );
            // NoInvNk acts on the one-electron overlap weights, not on the
            // seven fitted parts (its part miss is the 1e-12 floor).
            if m != KStressMutation::NoInvNk {
                assert!(x.part > MUTANT_BAR, "{m:?} {exx:?}: parts ({:e})", x.part);
            }
            if m == KStressMutation::SplitNoSmoothVolume {
                assert!(x.off <= SC_FIT_BAR, "{m:?} diagonal-only ({:e})", x.off);
            }
        }
    }
}

/// PHYSICS: the split is an exact decomposition of the same fitted energy, so
/// the split k stress equals the UNSPLIT k stress up to the split's
/// own tolerance (each side at its own converged density). ARTIFACT: equality
/// to 1e-13 would mean the split ran the unsplit path (the moved counters are
/// asserted); the difference is the split's truncation, MEASURED here.
#[test]
fn h2_1x1x3_split_k_stress_matches_the_unsplit_k_stress() {
    let bs = pyscf_sto3g_h();
    let ks = ksys(h2_cell_g(), &bs, [1, 1, 3]);
    let sp = fitsys_cfg(&ks, split_kcfg(1.0));
    let un = fitsys(&ks);
    assert_moved("H2 1x1x3", &sp.gdf, &un.gdf);
    let (scf_s, scf_u) = (fit_scfs(&ks, &sp), fit_scfs(&ks, &un));
    for exx in EXX {
        let i = exx_idx(exx);
        let s = ks_rhf_fit(&ks, &sp, &scf_s[i], exx, None);
        let u = ks_rhf_fit(&ks, &un, &scf_u[i], exx, None);
        let d = max_mat(&s.de_deps, &u.de_deps);
        eprintln!(
            "  H2 1x1x3 {exx:?}: |split − unsplit| dE/dε {d:.2e}, energies {:.3e}",
            (scf_s[i].energy - scf_u[i].energy).abs()
        );
        assert!(d <= SPLIT_UNSPLIT_BAR, "{exx:?}: {d:e}");
        assert!(d > 1e-15, "{exx:?}: split ran the unsplit path ({d:e})");
    }
}

/// Measured on H2 1x1x3 (lambda 1, omega 1, each side at its own density):
/// 2.6e-12 (none) / 5.9e-13 (ewald); the split-vs-unsplit energy differs by
/// 4e-15. The bar is 40x above the floor and 7+ orders below the smallest
/// split mutant (7.8e-4).
const SPLIT_UNSPLIT_BAR: f64 = 1e-10;

// ------------------------------------------------ FD of the split k energy

/// Split k-RHF energies `[none, ewald]` on `(cell, bs, n, centring)` with the
/// per-q kept metric counts.
fn k_split_energies(
    cell: &Cell,
    bs: &BasisSet,
    n: [usize; 3],
    centring: MeshCentring,
) -> ([f64; 2], Vec<usize>) {
    let ks = ksys_c(cell.clone(), bs, n, centring);
    let fs = fitsys_cfg(&ks, split_kcfg(1.0));
    let kept = fs.gdf.stats().per_q.iter().map(|q| q.naux_kept).collect();
    let e = [
        krhf(&ks, fs.jk(&ks, ExxDiv::None)).energy,
        krhf(&ks, fs.jk(&ks, ExxDiv::Ewald)).energy,
    ];
    (e, kept)
}

/// FD anchor of a split k-RHF system at h and 2h, all nine components, both
/// exxdiv (the metric cut count is asserted frozen).
fn split_fd_anchor(
    name: &str,
    cell: Cell,
    bs: &BasisSet,
    n: [usize; 3],
    c: MeshCentring,
) -> (KSys, FitSys, [KScfResult; 2], [Mat3; 2]) {
    let ks = ksys_c(cell, bs, n, c);
    let fs = fitsys_cfg(&ks, split_kcfg(1.0));
    let scfs = fit_scfs(&ks, &fs);
    let kept0: Vec<usize> = fs.gdf.stats().per_q.iter().map(|q| q.naux_kept).collect();
    let e = |cell: &Cell| {
        let (e, kept) = k_split_energies(cell, bs, n, c);
        assert_eq!(kept, kept0, "the metric cut changed under strain");
        e
    };
    let fd1 = fd_all9(&ks.cell, FD_H, &e);
    let fd2 = fd_all9(&ks.cell, 2.0 * FD_H, &e);
    for exx in EXX {
        let x = exx_idx(exx);
        let s = ks_rhf_fit(&ks, &fs, &scfs[x], exx, None);
        let (m1, m2) = (max_mat(&s.de_deps, &fd1[x]), max_mat(&s.de_deps, &fd2[x]));
        eprintln!(
            "  {name} split {n:?} {exx:?}: |analytic − FD| h {m1:.2e}, 2h {m2:.2e} (ratio {:.2}), \
             antisym(σ) {:.1e}",
            m2 / m1,
            s.antisymmetry()
        );
        assert!(m1 <= K_FD_BAR_FIT, "{name} {exx:?}: FD miss {m1:e}");
        assert!((3.0..5.0).contains(&(m2 / m1)), "not h²: {}", m2 / m1);
        assert!(s.antisymmetry() <= ANTISYM_BAR);
    }
    (ks, fs, scfs, fd1)
}

/// PHYSICS: analytic split `dE/dε` vs central FD of ferric's own split k
/// energy (fresh split build + SCF at every strained point; lattice and atoms
/// strained, index sets frozen, k at fixed fractional coordinates). ARTIFACT:
/// the split's partition is geometry-dependent (primitive criteria are
/// strain-invariant, the pair screens are not); FD(h) vs FD(2h) must stay at
/// h² (asserted) or the split energy is piecewise under strain.
#[test]
#[ignore = "slow (~37 s release): 36 strained split RS-GDF builds + 72 k-SCFs (h and 2h)"]
fn h2_1x1x3_split_stress_matches_fd_of_own_k_energy_all_nine() {
    let (ks, fs, scfs, fd1) = split_fd_anchor(
        "H2",
        h2_cell_g(),
        &pyscf_sto3g_h(),
        [1, 1, 3],
        MeshCentring::Gamma,
    );
    for m in SPLIT_MUTANTS {
        let s = ks_rhf_fit(&ks, &fs, &scfs[0], ExxDiv::None, Some(m));
        let mm = max_mat(&s.de_deps, &fd1[0]);
        eprintln!("  H2 split FD mutant {m:?}: |mut − FD| {mm:.2e}");
        assert!(mm > MUTANT_BAR, "{m:?} not caught by FD ({mm:e})");
    }
}

/// An even MP mesh (twisted supercell, no supercell anchor; pair-image modulus
/// 4 vs aux modulus 2) with the split: only the FD pins it.
#[test]
#[ignore = "slow (~16 s release): 36 strained split RS-GDF builds + 72 k-SCFs (h and 2h)"]
fn h2_mp_1x1x2_split_stress_matches_fd_of_own_k_energy_all_nine() {
    let (ks, ..) = split_fd_anchor(
        "H2 MP",
        h2_cell_g(),
        &pyscf_sto3g_h(),
        [1, 1, 2],
        MeshCentring::MonkhorstPack,
    );
    assert_eq!(ks.mesh.residue_moduli(), [1, 1, 4]);
}

// ------------------------------------------------------------ split UHF

/// 1x1x1 H3 doublet: the k split UHF stress is the Gamma split UHF stress.
#[test]
fn one_point_mesh_split_rsgdf_uhf_stress_is_the_gamma_split_stress() {
    let bs = pyscf_sto3g_h();
    let ks = ksys(h3_cell(), &bs, [1, 1, 1]);
    let fs = fitsys_cfg(&ks, split_kcfg(1.0));
    assert_moved("H3 1x1x1", &fs.gdf, &fitsys(&ks).gdf);
    let scfs = fit_uscfs(&ks, &fs, 2, 1);
    let g = gfit_aux(h3_cell(), &bs, &et_sp_aux(), &split_gdf_cfg(1.0));
    for exx in EXX {
        let scf = &scfs[exx_idx(exx)];
        let gscf = gamma_unrestricted(
            real_of(&scf.density_alpha[0]),
            real_of(&scf.density_beta[0]),
        );
        let k = ks_uhf_fit(&ks, &fs, scf, exx, None);
        let gs = gamma_fit_stress(&g, &gscf, exx);
        let total = max_mat(&k.de_deps, &gs.de_deps);
        eprintln!("  H3 UHF split 1x1x1 {exx:?}: |dE/dε k − Gamma| {total:.2e}");
        for ((name, kp), gp) in fit_parts(&k.parts).iter().zip(gamma_fit_parts(&gs.parts)) {
            let d = max_mat(kp, &gp);
            eprintln!("    {name}: |k − Gamma| {d:.2e}");
            assert!(d <= GAMMA_FIT_BAR, "{exx:?} {name}: {d:e}");
        }
        assert!(total <= GAMMA_FIT_BAR, "{exx:?}: {total:e}");
    }
}

/// H3 doublet 1x1x3: the split supercell (9 atoms, multiplicity 4) Gamma UHF
/// stress per cell, both exxdiv, per part, and the split + spin mutants.
#[test]
#[ignore = "slow (~98 s release): 9-atom split supercell Gamma RS-GDF UHF stress x2 + k stress"]
fn h3_uhf_1x1x3_split_stress_is_the_supercell_stress_per_cell() {
    let bs = pyscf_sto3g_h();
    let n = [1, 1, 3];
    let ks = ksys(h3_cell(), &bs, n);
    let fs = fitsys_cfg(&ks, split_kcfg(1.0));
    assert_moved("H3 1x1x3", &fs.gdf, &fitsys(&ks).gdf);
    let scfs = fit_uscfs(&ks, &fs, 2, 1);
    let g = gfit_aux(
        supercell(&h3_cell(), n, 4),
        &bs,
        &et_sp_aux(),
        &split_gdf_cfg(1.0),
    );
    let mut refs = [ZERO; 2];
    let mut ref_parts = [[ZERO; 7]; 2];
    for exx in EXX {
        let i = exx_idx(exx);
        let gscf = gamma_unrestricted(
            unfold(&ks.mesh, &scfs[i].density_alpha),
            unfold(&ks.mesh, &scfs[i].density_beta),
        );
        let k = ks_uhf_fit(&ks, &fs, &scfs[i], exx, None);
        let gs = gamma_fit_stress(&g, &gscf, exx);
        refs[i] = scale3(&gs.de_deps, 1.0 / 3.0);
        let miss = max_mat(&k.de_deps, &refs[i]);
        eprintln!("  H3 UHF split {n:?} {exx:?}: |dE/dε_k − dE/dε_sc/N| {miss:.2e}");
        for (pi, ((pn, kp), gp)) in fit_parts(&k.parts)
            .iter()
            .zip(gamma_fit_parts(&gs.parts))
            .enumerate()
        {
            ref_parts[i][pi] = scale3(&gp, 1.0 / 3.0);
            let d = max_mat(kp, &ref_parts[i][pi]);
            eprintln!("    {pn}: |k − sc/N| {d:.2e}");
            assert!(d <= SC_FIT_BAR, "{exx:?} {pn}: {d:e}");
        }
        assert!(miss <= SC_FIT_BAR, "{exx:?}: {miss:e}");
    }
    for m in SPLIT_MUTANTS.into_iter().chain([
        KStressMutation::SwapSpinFock,
        KStressMutation::FitCrossSpin,
        KStressMutation::NoPhase,
        KStressMutation::ConjD,
    ]) {
        let s = ks_uhf_fit(&ks, &fs, &scfs[0], ExxDiv::None, Some(m));
        let miss = max_mat(&s.de_deps, &refs[0]);
        eprintln!("  H3 UHF split mutant {m:?}: miss {miss:.2e}");
        assert!(miss > MUTANT_BAR, "{m:?} not caught ({miss:e})");
    }
}

/// A pure-d aux shell (l = 2) with the split, 1x1x3 supercell anchor, both
/// exxdiv, per part. MEASURED at the default `lindep`: total 7.4e-12 / 7.2e-12
/// but the two J2 parts (SR and LR) are 7.7e-11..1.1e-10 and cancel in the
/// total (unsplit d-aux J2 parts: 1.7e-12..2.5e-12); the floor does not move
/// with the build precision. With the single near-null metric direction
/// (eigenvalue 2.1e-4) dropped on both sides (`lindep` 1e-3) every part is
/// <= `SC_FIT_BAR` (2.6e-12, see
/// `d_aux_part_floor_collapses_when_the_near_null_mode_is_dropped`): the floor
/// is the fit's sensitivity along that mode, not a defect. Both are asserted.
#[test]
#[ignore = "slow (~35 s release): two supercell anchors with a pure-d aux shell and the split"]
fn h2_1x1x3_split_stress_with_a_d_aux_is_the_supercell_stress() {
    let split = split_gdf_cfg(1.0);
    let bs = pyscf_sto3g_h();
    let _ = fit_supercell_anchor_bars(
        "H2 split d-aux",
        h2_cell_g(),
        &bs,
        &et_spd_aux(),
        &split,
        [1, 1, 3],
        5e-10,
    );
    let _ = fit_supercell_anchor_cfg(
        "H2 split d-aux lindep 1e-3",
        h2_cell_g(),
        &bs,
        &et_spd_aux(),
        &RsGdfConfig {
            lindep: 1e-3,
            ..split
        },
        [1, 1, 3],
    );
}

// ----------------------------------------------------------------- guard

/// The Fock-consistency guard on a SPLIT source: accepts the matching SCF,
/// refuses the wrong exxdiv and a perturbed Fock. The split SCF against the
/// UNSPLIT source is NOT inconsistent (an exact decomposition): the printed
/// mismatch is the split tolerance, far below the 1e-8 bar.
#[test]
fn split_stress_fock_guard_accepts_matching_and_refuses_inconsistent_inputs() {
    let bs = pyscf_sto3g_h();
    let ks = ksys(h2_cell_g(), &bs, [1, 1, 3]);
    let sp = fitsys_cfg(&ks, split_kcfg(1.0));
    let un = fitsys(&ks);
    let scf = krhf(&ks, sp.jk(&ks, ExxDiv::None));
    let ok = ks_rhf_fit(&ks, &sp, &scf, ExxDiv::None, None);
    eprintln!(
        "  matching split inputs: Fock mismatch {:.2e}",
        ok.fock_mismatch
    );
    assert!(ok.fock_mismatch < 1e-12);
    let cross = ks_rhf_fit(&ks, &un, &scf, ExxDiv::None, None);
    eprintln!(
        "  split SCF vs unsplit source: Fock mismatch {:.2e}",
        cross.fock_mismatch
    );
    assert!(cross.fock_mismatch < FOCK_CONSISTENCY_BAR);
    let stress = |scf: &KScfResult, exx: ExxDiv| {
        kpoint_rhf_stress(
            &ks.cell,
            &ks.prep,
            &ks.mesh,
            &hcore_cfg(),
            &ks.hk,
            KStressJk::RsGdf(sp.src()),
            scf,
            exx,
            &scfg(None),
        )
    };
    let e = stress(&scf, ExxDiv::Ewald).unwrap_err().to_string();
    assert!(e.contains("Fock matrix rebuilt"), "{e}");
    let mut bad = scf.clone();
    bad.fock[0][(0, 0)] += Complex64::new(1e-6, 0.0);
    let e = stress(&bad, ExxDiv::None).unwrap_err().to_string();
    assert!(e.contains("Fock matrix rebuilt"), "{e}");
}

// ----------------------------------------------- d-aux part floor (split)

/// The ET-sp aux plus one pure-d shell at exponent `a` (lambda 1, omega 1
/// moves the d shell iff `a <= 1`).
fn et_sp_d_aux(a: f64) -> BasisSet {
    let mut aux = et_sp_aux();
    aux.shells.get_mut(&1).unwrap().push(Shell {
        l: 2,
        pure: true,
        exponents: vec![a],
        coefficients: vec![1.0],
    });
    aux
}

/// Worst fitted-part and total miss of the 1x1x3 supercell anchor at a config
/// (both exxdiv; the identity mutant `WrongQ` reads the miss), and the
/// smallest metric eigenvalue over the q classes of the k build.
fn d_aux_floor(aux_bs: &BasisSet, gdf: &RsGdfConfig) -> (f64, f64, f64) {
    let a = fit_supercell_anchor_bars(
        "diag",
        h2_cell_g(),
        &pyscf_sto3g_h(),
        aux_bs,
        gdf,
        [1, 1, 3],
        1.0,
    );
    let (mut part, mut total) = (0.0_f64, 0.0_f64);
    for exx in EXX {
        let x = a.mutant(exx, KStressMutation::WrongQ);
        part = part.max(x.part);
        total = total.max(x.total);
    }
    let eig =
        a.fs.gdf
            .stats()
            .per_q
            .iter()
            .map(|q| q.metric_eig_min)
            .fold(f64::INFINITY, f64::min);
    (part, total, eig)
}

/// The pure-d aux PART floor of the split supercell anchor (totals agree to
/// 7e-12; the J2 SR / J2 LR parts differ by 1e-10 and cancel). The smallest
/// metric eigenvalue is 2.1e-4 (> `lindep`, so nothing is dropped by default).
/// MEASURED discriminators (release, H2 1x1x3): the floor does NOT move with
/// the build PRECISION (1.1e-10 at 1e-13, 5.1e-11 at 1e-15, 1.05e-10 at 1e-17,
/// so it is not a truncation floor), is 4.4e-11 with a COMPACT (unmoved) d
/// exponent 1.6 and 2.9e-12 unsplit; but dropping the single near-null metric
/// direction (`lindep` 1e-3, on BOTH sides) collapses it (asserted below). A
/// defect independent of that one mode would not move; this floor is the
/// fit's sensitivity along it, amplified by the split's larger SR/LR parts.
#[test]
#[ignore = "slow (~75 s release): 3 supercell anchors on H2 1x1x3"]
fn d_aux_part_floor_collapses_when_the_near_null_mode_is_dropped() {
    let split = split_gdf_cfg(1.0);
    let run = |tag: &str, g: &RsGdfConfig| {
        let r = d_aux_floor(&et_sp_d_aux(0.8), g);
        eprintln!(
            "  [{tag}] worst part {:.2e}, worst total {:.2e}, min metric eig {:.2e}",
            r.0, r.1, r.2
        );
        r
    };
    let u = run("unsplit lindep 1e-10", &gdf_cfg());
    let s = run("split   lindep 1e-10", &split);
    let l = run(
        "split   lindep 1e-3 ",
        &RsGdfConfig {
            lindep: 1e-3,
            ..split
        },
    );
    assert!(s.0 > 10.0 * u.0, "the split carries the larger part floor");
    assert!(
        l.0 < 0.1 * s.0,
        "floor must collapse when the mode is dropped"
    );
    assert!(l.0 <= SC_FIT_BAR, "{:e}", l.0);
}

/// 1x1x1 d-aux split vs the Gamma split, part by part (q = 0 only: the sphere
/// is the same set on both sides, so edge-set differences cannot enter).
/// MEASURED at the default `lindep`: total <= 4.6e-13 but the J2 SR / LR parts
/// differ by 8e-12 / 1.1e-11 (ET-sp aux without d: <= 1e-13), cancelling; with
/// the single near-null metric direction (eigenvalue 2.1e-4) dropped by
/// `lindep` = 1e-3 on both sides every part collapses to <= 4.3e-14, so the
/// floor is that mode's sensitivity, not a defect. Both are asserted.
#[test]
fn one_point_mesh_split_d_aux_stress_is_the_gamma_split_stress() {
    let (part, total, dropped, emin) = one_point_part_floor(&et_spd_aux(), &split_gdf_cfg(1.0));
    eprintln!(
        "  d-aux 1x1x1 default lindep: part {part:.2e}, total {total:.2e}, min eig {emin:.2e}"
    );
    assert_eq!(dropped, 0);
    assert!(total <= GAMMA_FIT_BAR, "{total:e}");
    assert!(part <= 5e-11, "{part:e}");
    let g = RsGdfConfig {
        lindep: 1e-3,
        ..split_gdf_cfg(1.0)
    };
    let (part, total, dropped, _) = one_point_part_floor(&et_spd_aux(), &g);
    eprintln!("  d-aux 1x1x1 lindep 1e-3 (dropped {dropped}): part {part:.2e}, total {total:.2e}");
    assert_eq!(dropped, 1);
    assert!(
        total <= GAMMA_FIT_BAR && part <= GAMMA_FIT_BAR,
        "{part:e} {total:e}"
    );
}

// ------------------------------------- split vs unsplit: other omega/lambda

/// `(orbital smooth, aux smooth)` primitive counts of a split build.
fn moved_counts(gdf: &KRsGdf) -> (usize, usize) {
    let c = |name: &str| {
        gdf.stats()
            .split_counters
            .iter()
            .find(|(n, _)| *n == name)
            .map(|&(_, v)| v)
            .unwrap_or_else(|| panic!("counter {name:?} missing (split path did not run)"))
    };
    (
        c("rsgdf split orbital prims smooth"),
        c("rsgdf split aux prims smooth"),
    )
}

/// `|split − unsplit|` of the k stress (`max` over the nine components) for
/// both exxdiv on H2 1x1x3 at `aux_bs`, each build at its own SCF density,
/// the split config `sp` against the unsplit config `un`.
fn split_vs_unsplit(aux_bs: &BasisSet, sp: &RsGdfConfig, un: &RsGdfConfig) -> ([f64; 2], FitSys) {
    let bs = pyscf_sto3g_h();
    let ks = ksys(h2_cell_g(), &bs, [1, 1, 3]);
    let kc = |g: &RsGdfConfig| KRsGdfConfig {
        gdf: *g,
        mutation: None,
    };
    let fs_s = fitsys_with(&ks, kc(sp), aux_bs);
    let fs_u = fitsys_with(&ks, kc(un), aux_bs);
    let (scf_s, scf_u) = (fit_scfs(&ks, &fs_s), fit_scfs(&ks, &fs_u));
    let mut d = [0.0; 2];
    for exx in EXX {
        let i = exx_idx(exx);
        let s = ks_rhf_fit(&ks, &fs_s, &scf_s[i], exx, None);
        let u = ks_rhf_fit(&ks, &fs_u, &scf_u[i], exx, None);
        d[i] = max_mat(&s.de_deps, &u.de_deps);
    }
    (d, fs_s)
}

/// The split is an exact decomposition at ANY omega and lambda <= 1: the
/// split k stress equals the unsplit one at omega 1.3, lambda 0.7 (the
/// omega-1 / lambda-1 test cannot see an omega or lambda that is hard-wired
/// somewhere in the new passes). Measured 3.6e-12 / 3.7e-12 (none / ewald).
#[test]
#[ignore = "slow-ish (~17 s release): 2 split + 2 unsplit builds and 4 SCFs at omega 1.3"]
fn h2_1x1x3_split_k_stress_matches_the_unsplit_one_at_omega_1p3_lambda_0p7() {
    let at = |split: Option<RangeSplit>| RsGdfConfig {
        omega: 1.3,
        range_split: split,
        ..gdf_cfg()
    };
    let (d, fs) = split_vs_unsplit(&et_sp_aux(), &at(Some(RangeSplit::new(0.7))), &at(None));
    let (o, a) = moved_counts(&fs.gdf);
    eprintln!(
        "  omega 1.3, lambda 0.7: moved orbital {o}, aux {a}; |split − unsplit| {:?}",
        d
    );
    assert!(o > 0 && a > 0, "nothing moved");
    for x in d {
        assert!(x <= SPLIT_UNSPLIT_BAR, "{x:e}");
        assert!(x > 1e-15, "split ran the unsplit path ({x:e})");
    }
}

/// The two partial-move branches of the LR strain dispatch. AUX moves, no
/// orbital does (ET-sp aux, lambda 0.32: aux s(0.3) <= 0.32 but the orbital
/// primitive 0.1689 > 0.16): the moved pass runs without a smooth-pair pass.
/// ORBITALS move, aux does not (aux s(1.9, 0.5), lambda 0.4: orbital 0.1689
/// <= 0.2 but the smallest aux exponent 0.5 > 0.4): the unsplit LR pass plus
/// the smooth-pair pass with `Xc = X`. Each agrees with the unsplit stress.
#[test]
fn split_k_stress_partial_move_branches_match_the_unsplit_stress() {
    let split = |l: f64| Some(RangeSplit::new(l));
    let at = |s: Option<RangeSplit>| RsGdfConfig {
        range_split: s,
        ..gdf_cfg()
    };
    let (d, fs) = split_vs_unsplit(&et_sp_aux(), &at(split(0.32)), &at(None));
    let (o, a) = moved_counts(&fs.gdf);
    eprintln!(
        "  aux-only move (lambda 0.32): orbital {o}, aux {a}; diff {:?}",
        d
    );
    assert_eq!(o, 0, "an orbital primitive moved");
    assert!(a > 0, "no aux primitive moved");
    for x in d {
        assert!(x <= SPLIT_UNSPLIT_BAR, "aux-only: {x:e}");
        assert!(x > 1e-15, "aux-only: split ran the unsplit path ({x:e})");
    }
    let (d, fs) = split_vs_unsplit(&coarse_aux(), &at(split(0.4)), &at(None));
    let (o, a) = moved_counts(&fs.gdf);
    eprintln!(
        "  orbital-only move (lambda 0.4): orbital {o}, aux {a}; diff {:?}",
        d
    );
    assert!(o > 0, "no orbital primitive moved");
    assert_eq!(a, 0, "an aux primitive moved");
    for x in d {
        assert!(x <= SPLIT_UNSPLIT_BAR, "orbital-only: {x:e}");
        assert!(
            x > 1e-15,
            "orbital-only: split ran the unsplit path ({x:e})"
        );
    }
}

/// TRIVIAL LIMIT: lambda = 0 builds the split plan but moves nothing, so the
/// k stress (every part and the total) is the unsplit one BIT FOR BIT (the
/// LR strain is the unsplit pass, every libint factor is 1, `q_c` is `q`, no
/// smooth overlap).
#[test]
fn nothing_moved_split_k_stress_is_the_unsplit_stress_bitwise() {
    let bs = pyscf_sto3g_h();
    let ks = ksys(h2_cell_g(), &bs, [1, 1, 3]);
    let un = fitsys(&ks);
    let sp = fitsys_cfg(&ks, split_kcfg(0.0));
    assert_eq!(moved_counts(&sp.gdf), (0, 0), "lambda = 0 moved something");
    let scfs = fit_scfs(&ks, &un);
    for exx in EXX {
        let i = exx_idx(exx);
        let u = ks_rhf_fit(&ks, &un, &scfs[i], exx, None);
        let s = ks_rhf_fit(&ks, &sp, &scfs[i], exx, None);
        let bits = |m: &Mat3| m.iter().flatten().map(|x| x.to_bits()).collect::<Vec<_>>();
        let d = max_mat(&s.de_deps, &u.de_deps);
        eprintln!("  lambda 0 {exx:?}: |split − unsplit| {d:.2e}");
        assert_eq!(bits(&s.de_deps), bits(&u.de_deps), "{exx:?} total ({d:e})");
        for ((n, a), (_, b)) in fit_parts(&s.parts).iter().zip(fit_parts(&u.parts)) {
            assert_eq!(bits(a), bits(&b), "{exx:?} {n}");
        }
    }
}

/// Typed refusals of the split source: a build and a config that disagree
/// about the split (either way), and lambda > 1.
#[test]
fn split_stress_refuses_a_mismatched_config_and_lambda_above_one() {
    let bs = pyscf_sto3g_h();
    let ks = ksys(h2_cell_g(), &bs, [1, 1, 3]);
    let sp = fitsys_cfg(&ks, split_kcfg(1.0));
    let un = fitsys(&ks);
    let scf = krhf(&ks, un.jk(&ks, ExxDiv::None));
    let run = |gdf: &KRsGdf, cfg: &KRsGdfConfig, aux: &PreparedBasis| {
        kpoint_rhf_stress(
            &ks.cell,
            &ks.prep,
            &ks.mesh,
            &hcore_cfg(),
            &ks.hk,
            KStressJk::RsGdf(KRsGdfGradSource { gdf, cfg, aux }),
            &scf,
            ExxDiv::None,
            &scfg(None),
        )
    };
    // Split build, unsplit config; unsplit build, split config.
    let e = run(&sp.gdf, &kgdf_cfg(), &sp.aux).unwrap_err().to_string();
    assert!(
        e.contains("ran the range split but the config does not"),
        "{e}"
    );
    let e = run(&un.gdf, &split_kcfg(1.0), &un.aux)
        .unwrap_err()
        .to_string();
    assert!(
        e.contains("did not run the range split but the config requests"),
        "{e}"
    );
    // Matching split build and config is accepted (an SCF on the split build
    // is not needed: the refusals fire first, the accept path is covered by
    // the anchors).
    // lambda = 2 (build and config agree): refused with the reason.
    let l2 = fitsys_cfg(&ks, split_kcfg(2.0));
    let e = run(&l2.gdf, &split_kcfg(2.0), &l2.aux)
        .unwrap_err()
        .to_string();
    assert!(e.contains("lambda = 2 > 1"), "{e}");
}

// ----------------------------------- cheap q != 0 split FD (non-ignored)

/// Central FD of the split k energy at h for two components of the H2 1x1x3
/// non-TRIM mesh: zz (the mesh direction) and xz (off-diagonal), both exxdiv.
/// A reduced version of the all-nine FD (which stays ignored) so the q != 0
/// split strain is pinned by an independent anchor in the default test run.
#[test]
fn h2_1x1x3_split_stress_matches_fd_for_zz_and_xz() {
    let bs = pyscf_sto3g_h();
    let n = [1, 1, 3];
    let ks = ksys(h2_cell_g(), &bs, n);
    let fs = fitsys_cfg(&ks, split_kcfg(1.0));
    let scfs = fit_scfs(&ks, &fs);
    let kept0: Vec<usize> = fs.gdf.stats().per_q.iter().map(|q| q.naux_kept).collect();
    for (i, j) in [(2, 2), (0, 2)] {
        let at = |h: f64| {
            let (e, kept) =
                k_split_energies(&strained(&ks.cell, i, j, h), &bs, n, MeshCentring::Gamma);
            assert_eq!(kept, kept0, "the metric cut changed under strain");
            e
        };
        let (ep, em) = (at(FD_H), at(-FD_H));
        for exx in EXX {
            let x = exx_idx(exx);
            let fd = (ep[x] - em[x]) / (2.0 * FD_H);
            let s = ks_rhf_fit(&ks, &fs, &scfs[x], exx, None);
            let miss = (s.de_deps[i][j] - fd).abs();
            eprintln!(
                "  split FD ({i},{j}) {exx:?}: analytic {:.8e}, |analytic − FD| {miss:.2e}",
                s.de_deps[i][j]
            );
            assert!(miss <= K_FD_BAR_FIT, "({i},{j}) {exx:?}: {miss:e}");
        }
    }
}

/// Worst fitted-part and total `|k − Gamma|` of the 1x1x1 split anchor (none
/// exxdiv) at `gdf`, with the number of metric eigenvalues dropped by `lindep`
/// and the smallest metric eigenvalue.
fn one_point_part_floor(aux_bs: &BasisSet, gdf: &RsGdfConfig) -> (f64, f64, usize, f64) {
    let bs = pyscf_sto3g_h();
    let ks = ksys(h2_cell_g(), &bs, [1, 1, 1]);
    let kc = KRsGdfConfig {
        gdf: *gdf,
        mutation: None,
    };
    let fs = fitsys_with(&ks, kc, aux_bs);
    let scf = krhf(&ks, fs.jk(&ks, ExxDiv::None));
    let g = gfit_aux(h2_cell_g(), &bs, aux_bs, gdf);
    let gscf = gamma_restricted(real_of(&scf.densities[0]));
    let k = ks_rhf_fit(&ks, &fs, &scf, ExxDiv::None, None);
    let gs = gamma_fit_stress(&g, &gscf, ExxDiv::None);
    let part = fit_parts(&k.parts)
        .iter()
        .zip(gamma_fit_parts(&gs.parts))
        .map(|((_, a), b)| max_mat(a, &b))
        .fold(0.0_f64, f64::max);
    let q = &fs.gdf.stats().per_q[0];
    (
        part,
        max_mat(&k.de_deps, &gs.de_deps),
        q.n_dropped,
        q.metric_eig_min,
    )
}

// ============================================================ open-shell UHF beyond H3 (#421 a)
//
// The H3 doublet anchors above are s-only with N_alpha - N_beta = 1. The
// systems here add p functions and different spin polarisation:
//   S1: H2 TRIPLET (2 alpha, 0 beta), s+p basis (nao 8): the beta density is
//       identically zero, a one-sector edge of every spin sum, with p shells
//       whose SR-derivative floor is separate from the s-only one;
//   S2: H4 TRIPLET (3 alpha, 1 beta), s+p basis (nao 16): both sectors
//       populated, unequal, with p functions.
//
// ARTIFACT hypothesis (stated before measuring). A defect that only shows
// for p shells or only for a spin-unbalanced density would be (i) a spin sum
// that is symmetric in alpha/beta by accident of N_alpha - N_beta = 1 (H3:
// the alpha and beta densities differ by a rank-1 piece), e.g. the exchange
// of the minority spin using the majority density: an O(1) miss when
// D_beta = 0 (S1: exchange of beta is exactly 0, any use of D_alpha there is
// visible), (ii) a p-shell AO-derivative or strain pair vector error that the
// closed-shell s+p H2 test sees only at N_k = 1x1x3 and RHF: the UHF Pulay
// weight W = sum_sigma D_s F_s D_s must reproduce it with F_alpha != F_beta.
// If correct: 1x1x1 == Gamma UHF <= 1e-12, k mesh == supercell/N <= 1e-10,
// FD at the h^2 floor. Independent constructions: Gamma UHF stress (real
// half-sphere kernel), the explicit supercell, central FD of the k energy.

const OS_A: f64 = 4.0;
const H4_ATOMS: [[f64; 3]; 4] = [
    [0.3, 0.2, 0.1],
    [0.35, 0.12, 1.5],
    [1.6, 0.9, 0.7],
    [1.2, 2.1, 2.3],
];
const H4_A: f64 = 4.5;

/// An open-shell test system: cell (carrying the multiplicity), the spin
/// counts and a short name.
struct OpenShell {
    name: &'static str,
    cell: Cell,
    na: usize,
    nb: usize,
}

fn s1_h2_triplet() -> OpenShell {
    OpenShell {
        name: "H2 triplet s+p",
        cell: cell_at(&H2_ATOMS, cubic(OS_A), 3),
        na: 2,
        nb: 0,
    }
}

fn s2_h4_triplet() -> OpenShell {
    OpenShell {
        name: "H4 triplet s+p",
        cell: cell_at(&H4_ATOMS, cubic(H4_A), 3),
        na: 3,
        nb: 1,
    }
}

/// k-UHF energies `[none, ewald]` of `(na, nb)` on a (strained) cell.
fn k_energies_uhf_nanb(
    cell: &Cell,
    bs: &BasisSet,
    n: [usize; 3],
    na: usize,
    nb: usize,
) -> [f64; 2] {
    let ks = ksys(cell.clone(), bs, n);
    let eri = kdense(&ks, ANCHOR_PRECISION);
    let vm = ks.mesh.madelung(&ks.cell).unwrap();
    [
        kuhf(&ks, Box::new(eri.jk_builder_with_madelung(0.0)), na, nb).energy,
        kuhf(&ks, Box::new(eri.jk_builder_with_madelung(vm)), na, nb).energy,
    ]
}

/// Both spin sectors really are populated unequally (and S1's beta is empty):
/// asserted on the SCF so a spin-restricted-looking solution cannot make the
/// anchors vacuous.
fn assert_spin_polarised(sys: &OpenShell, scf: &KUScfResult) {
    let mag: f64 = scf
        .density_alpha
        .iter()
        .zip(&scf.density_beta)
        .map(|(a, b)| (a - b).iter().map(|z| z.norm()).sum::<f64>())
        .sum();
    let beta: f64 = scf
        .density_beta
        .iter()
        .map(|b| b.iter().map(|z| z.norm()).sum::<f64>())
        .sum();
    eprintln!(
        "  {}: spin ({}, {}), sum|D_a - D_b| {mag:.3}, sum|D_b| {beta:.3e}, E {:.10}",
        sys.name, scf.nalpha, scf.nbeta, scf.energy
    );
    assert!(mag > 0.5, "{}: not spin polarised", sys.name);
    assert_eq!(beta < 1e-12, sys.nb == 0, "{}: beta sector", sys.name);
}

/// PHYSICS: at 1x1x1 the k-UHF stress is the Gamma-UHF stress, total and per
/// part, both exxdiv. ARTIFACT: blind to phases, carried by the supercell
/// anchor below; what is new is the spin-unbalanced s+p density. MEASURED
/// (S1 / S2): |k - Gamma| 3.1e-14 / 2.2e-14 (none), 9.8e-15 / 5.3e-14
/// (ewald) vs the 1e-12 bar; antisym(sigma) 7.8e-13 / 7.6e-13 (none),
/// 8.8e-13 / 1.3e-13 (ewald) vs P_ANTISYM_BAR (3.7e-11 / 9.0e-11 before #434).
fn open_shell_p_one_point(sys: OpenShell) {
    let bs = sp_basis_h();
    let ks = ksys(sys.cell.clone(), &bs, [1, 1, 1]);
    let eri = kdense(&ks, ANCHOR_PRECISION);
    let scfs = uscf_pair(&ks, &eri, sys.na, sys.nb);
    assert_spin_polarised(&sys, &scfs[0]);
    let g = gsys(sys.cell.clone(), &bs);
    for exx in EXX {
        let scf = &scfs[exx_idx(exx)];
        let gscf = gamma_unrestricted(
            real_of(&scf.density_alpha[0]),
            real_of(&scf.density_beta[0]),
        );
        let k = ks_uhf(&ks, &eri, scf, exx, None);
        let gs = gamma_stress(&g, &gscf, exx);
        assert_one_point(sys.name, &k, &gs, exx, P_ANTISYM_BAR);
    }
}

#[test]
fn open_shell_p_h2_triplet_one_point_mesh_uhf_stress_is_the_gamma_stress() {
    open_shell_p_one_point(s1_h2_triplet());
}

#[test]
#[ignore = "slow (~2 min release): H4 s+p, 2.8M SR triplets, k and Gamma, both exxdiv"]
fn open_shell_p_h4_triplet_one_point_mesh_uhf_stress_is_the_gamma_stress() {
    open_shell_p_one_point(s2_h4_triplet());
}

/// Bar of the p-shell supercell anchor at the DEFAULT model-nucleus exponent
/// (the unrotated ECP file's bar for the same floor). Before #434 was fixed
/// the SR derivative's translation noise put the miss at 5.9e-8 at the default
/// exponent (pinned at 1e7 as a workaround, 1.7e-10); with the nucleus-frame
/// evaluation it is 4.6e-13 at the default (both exxdiv), the smallest mutant
/// misses by > 1e-4 (`MUTANT_BAR`), and the bar is 11x the measurement.
/// Also the blind / diagonal-only floor of the mutant sweep.
const OS_SC_BAR: f64 = 5e-12;

/// PHYSICS: the 1x1x3 (non-TRIM) k-UHF stress per cell is the explicit
/// 3-cell supercell's Gamma-UHF stress / 3, both exxdiv, at the default
/// model-nucleus exponent; the
/// k-specific mutants and the Gamma-style term mutants all miss it. S1 only
/// (the supercell has 24 AOs).
#[test]
#[ignore = "slow (~12 min release at load 10): 6-atom s+p supercell Gamma UHF stress x2 + 26 k stress mutant evaluations"]
fn open_shell_p_1x1x3_uhf_stress_is_the_supercell_stress_per_cell() {
    let bs = sp_basis_h();
    let sys = s1_h2_triplet();
    let n = [1, 1, 3];
    let ks = ksys(sys.cell.clone(), &bs, n);
    let eri = kdense(&ks, ANCHOR_PRECISION);
    let scfs = uscf_pair(&ks, &eri, sys.na, sys.nb);
    assert_spin_polarised(&sys, &scfs[0]);
    // 3 cells x 2 unpaired electrons: multiplicity 7.
    let g = gsys(supercell(&sys.cell, n, 7), &bs);
    let mut de = [ZERO; 2];
    for (i, exx) in EXX.into_iter().enumerate() {
        let gscf = gamma_unrestricted(
            unfold(&ks.mesh, &scfs[i].density_alpha),
            unfold(&ks.mesh, &scfs[i].density_beta),
        );
        let k = ks_uhf(&ks, &eri, &scfs[i], exx, None);
        let gs = gamma_uhf_stress(
            &g.cell,
            &g.prep,
            &hcore_cfg(),
            &g.hc,
            &g.eri,
            &gscf,
            exx,
            &GammaStressConfig {
                budget_bytes: Some(AMPLE),
                ..Default::default()
            },
        )
        .expect("gamma uhf stress");
        let r = scale3(&gs.de_deps, 1.0 / 3.0);
        let miss = max_mat(&k.de_deps, &r);
        report(&format!("{} {n:?} {exx:?}", sys.name), &k);
        eprintln!(
            "  {} {n:?} {exx:?}: |dE/dε_k − dE/dε_sc/N| {miss:.2e} (bar {OS_SC_BAR:.0e})",
            sys.name
        );
        assert!(miss <= OS_SC_BAR, "{exx:?}: {miss:e}");
        assert!(k.antisymmetry() <= P_ANTISYM_BAR);
        de[i] = r;
    }
    let refs = ScRef { de };
    let run = |exx: ExxDiv, m: Option<KStressMutation>| {
        kpoint_uhf_stress(
            &ks.cell,
            &ks.prep,
            &ks.mesh,
            &hcore_cfg(),
            &ks.hk,
            KStressJk::Dense(&eri),
            &scfs[exx_idx(exx)],
            exx,
            &scfg(m),
        )
        .expect("kpoint_uhf_stress")
    };
    assert_mutants_fail_floor(
        sys.name,
        &run,
        &refs,
        &K_MUTANTS,
        &K_EWALD_MUTANTS,
        OS_SC_BAR,
    );
    assert_mutants_fail_floor(
        sys.name,
        &run,
        &refs,
        &TERM_MUTANTS,
        &TERM_EWALD_MUTANTS,
        OS_SC_BAR,
    );
}

/// PHYSICS: all nine components of the k-UHF stress of S1 vs central FD (h =
/// 1e-4, 2h) of its own k energy, a fresh UHF SCF per point, both exxdiv,
/// plus the UHF-specific and k-specific mutants. Bar K_FD_BAR_UHF (the H3
/// doublet's own measured floor, same h^2 truncation); cf. K_FD_BAR_P.
#[test]
#[ignore = "slow (~5.5 min release at load 10): 36 strained dense-AFT builds + 72 k-UHF SCFs (h and 2h), s+p H2 triplet"]
fn open_shell_p_h2_triplet_1x1x3_uhf_stress_matches_fd_all_nine() {
    let bs = sp_basis_h();
    let sys = s1_h2_triplet();
    let n = [1, 1, 3];
    let e = |c: &Cell| k_energies_uhf_nanb(c, &bs, n, sys.na, sys.nb);
    let fd1 = fd_all9(&sys.cell, FD_H, &e);
    let fd2 = fd_all9(&sys.cell, 2.0 * FD_H, &e);
    let ks = ksys(sys.cell.clone(), &bs, n);
    let eri = kdense(&ks, ANCHOR_PRECISION);
    let scfs = uscf_pair(&ks, &eri, sys.na, sys.nb);
    assert_spin_polarised(&sys, &scfs[0]);
    for (x, exx) in EXX.into_iter().enumerate() {
        let s = ks_uhf(&ks, &eri, &scfs[x], exx, None);
        let miss = max_mat(&s.de_deps, &fd1[x]);
        let miss2 = max_mat(&s.de_deps, &fd2[x]);
        eprintln!(
            "  {} 1x1x3 {exx:?}: |analytic − FD| h {miss:.2e}, 2h {miss2:.2e} (ratio {:.2})",
            sys.name,
            miss2 / miss
        );
        assert!(miss <= K_FD_BAR_UHF, "{exx:?}: FD miss {miss:e}");
        assert!(
            (3.0..5.0).contains(&(miss2 / miss)),
            "not h²: {}",
            miss2 / miss
        );
        for m in [
            KStressMutation::CrossSpinK,
            KStressMutation::SwapSpinFock,
            KStressMutation::NoPhase,
            KStressMutation::WrongQ,
            KStressMutation::ConjD,
        ] {
            let sm = ks_uhf(&ks, &eri, &scfs[x], exx, Some(m));
            let mm = max_mat(&sm.de_deps, &fd1[x]);
            eprintln!("  {} {exx:?} mutant {m:?}: miss vs FD {mm:.2e}", sys.name);
            assert!(mm > MUTANT_BAR, "{m:?} {exx:?} not caught ({mm:e})");
        }
    }
}

/// S2 (3 alpha, 1 beta, 16 AOs, both sectors populated): zz, xz and xx of the
/// k-UHF stress vs central FD at h = 1e-4 on the 1x1x3 mesh, exxdiv none and
/// ewald.
#[test]
#[ignore = "slow (~20 min release at load 10): 6 strained dense-AFT builds + 12 k-UHF SCFs on H4 s+p"]
fn open_shell_p_h4_triplet_1x1x3_uhf_stress_matches_fd_zz_xz_xx() {
    let bs = sp_basis_h();
    let sys = s2_h4_triplet();
    let n = [1, 1, 3];
    let ks = ksys(sys.cell.clone(), &bs, n);
    let eri = kdense(&ks, ANCHOR_PRECISION);
    let scfs = uscf_pair(&ks, &eri, sys.na, sys.nb);
    assert_spin_polarised(&sys, &scfs[0]);
    let mut worst_mut = [[0.0_f64; 2]; 2];
    for (i, j) in [(2usize, 2usize), (0, 2), (0, 0)] {
        let ep = k_energies_uhf_nanb(&strained(&sys.cell, i, j, FD_H), &bs, n, sys.na, sys.nb);
        let em = k_energies_uhf_nanb(&strained(&sys.cell, i, j, -FD_H), &bs, n, sys.na, sys.nb);
        for (x, exx) in EXX.into_iter().enumerate() {
            let fd = (ep[x] - em[x]) / (2.0 * FD_H);
            let s = ks_uhf(&ks, &eri, &scfs[x], exx, None);
            let miss = (s.de_deps[i][j] - fd).abs();
            eprintln!(
                "  {} 1x1x3 ({i},{j}) {exx:?}: analytic {:.8e} FD {fd:.8e} miss {miss:.2e}",
                sys.name, s.de_deps[i][j]
            );
            assert!(miss <= K_FD_BAR_UHF, "({i},{j}) {exx:?}: {miss:e}");
            for (mi, m) in [KStressMutation::CrossSpinK, KStressMutation::SwapSpinFock]
                .into_iter()
                .enumerate()
            {
                let sm = ks_uhf(&ks, &eri, &scfs[x], exx, Some(m));
                let mm = (sm.de_deps[i][j] - fd).abs();
                worst_mut[mi][x] = worst_mut[mi][x].max(mm);
            }
        }
    }
    eprintln!(
        "  {} spin mutants worst miss vs FD [none, ewald]: CrossSpinK {:?}, SwapSpinFock {:?}",
        sys.name, worst_mut[0], worst_mut[1]
    );
    for w in worst_mut.iter().flatten() {
        assert!(*w > MUTANT_BAR, "a spin mutant escaped the FD ({w:e})");
    }
}

/// PHYSICS (#434): the Gamma stress's SR-attraction term of a cell is
/// invariant under a rigid translation of every atom (the cell is periodic,
/// the stress is a derivative of a translation-invariant energy). Before the
/// SR derivative walk evaluated each triplet in the frame of its nucleus,
/// libint2 combined O(cell-size) absolute coordinates with the 1e9-sized
/// model-nucleus exponent and the p-shell `vsr_zz` of this 3-cell H2 triplet
/// supercell moved by 8.5e-8 (per cell) between translations (and sat 5.9e-8
/// off the k-point value at the default exponent). ARTIFACT vs PHYSICS: a
/// physical effect would survive a rigid translation, a numerical-noise
/// artifact does not; after the fix the translated values agree to 1e-14.
/// Density: the unfolded s+p k-UHF triplet density (`parts.vsr` is a
/// fixed-density quantity, so nothing needs to be SCF-converged Gamma-side).
#[test]
#[ignore = "slow (~1 min release): k-UHF SCF + 3 Gamma supercell hcore/ERI builds"]
fn gamma_sr_stress_is_translation_invariant_for_p_shells_at_the_default_exponent() {
    let bs = sp_basis_h();
    let sys = s1_h2_triplet();
    let n = [1, 1, 3];
    let ks = ksys(sys.cell.clone(), &bs, n);
    let eri = kdense(&ks, ANCHOR_PRECISION);
    let scfs = uscf_pair(&ks, &eri, sys.na, sys.nb);
    let sc = supercell(&sys.cell, n, 7);
    let gscf = gamma_unrestricted(
        unfold(&ks.mesh, &scfs[0].density_alpha),
        unfold(&ks.mesh, &scfs[0].density_beta),
    );
    let vsr_zz = |shift: [f64; 3]| {
        let pos: Vec<[f64; 3]> = sc
            .positions()
            .iter()
            .map(|p| [p[0] + shift[0], p[1] + shift[1], p[2] + shift[2]])
            .collect();
        let g = gsys(cell_at(&pos, *sc.lattice(), 7), &bs);
        gamma_uhf_stress(
            &g.cell,
            &g.prep,
            &hcore_cfg(),
            &g.hc,
            &g.eri,
            &gscf,
            ExxDiv::None,
            &GammaStressConfig {
                budget_bytes: Some(AMPLE),
                ..Default::default()
            },
        )
        .expect("gamma uhf stress")
        .parts
        .vsr[2][2]
            / 3.0
    };
    let dk: Vec<Array2<Complex64>> = scfs[0]
        .density_alpha
        .iter()
        .zip(&scfs[0].density_beta)
        .map(|(a, b)| a + b)
        .collect();
    let k = kpoint_hcore_stress_fixed_d(
        &ks.cell,
        &ks.prep,
        &ks.mesh,
        &hcore_cfg(),
        &ks.hk,
        &dk,
        &scfg(None),
    )
    .expect("k hcore stress")
    .vsr[2][2];
    let v0 = vsr_zz([0.0; 3]);
    let mut worst = (v0 - k).abs();
    eprintln!("  vsr_zz/3 unshifted {v0:.12}, k {k:.12}: {worst:.2e}");
    for shift in [[0.37, -0.21, 1.3], [0.0, 0.0, -3.0]] {
        let v = vsr_zz(shift);
        eprintln!("  shifted {shift:?}: {v:.12}");
        worst = worst.max((v - v0).abs()).max((v - k).abs());
    }
    assert!(
        worst <= 1e-10,
        "translation/k mismatch {worst:e} (was 8.5e-8)"
    );
}
