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
use ferric_pbc::hcore::kpoint::{periodic_hcore_kpts, PeriodicHcoreK};
use ferric_pbc::hcore::{periodic_hcore, PeriodicHcore, PeriodicHcoreConfig};
use ferric_pbc::kdense_aft::{KDenseAftConfig, KDenseAftEri};
use ferric_pbc::kgrad::KRsGdfGradSource;
use ferric_pbc::kpts::KPointMesh;
use ferric_pbc::kscf::{solve_krhf_injected, KPointInjection, KPointJk, KScfConfig, KScfResult};
use ferric_pbc::kstress::{
    kpoint_hcore_stress_fixed_d, kpoint_rhf_stress, kpoint_uhf_stress, KStress, KStressConfig,
    KStressJk, KStressMutation,
};
use ferric_pbc::kuscf::{solve_kuhf_injected, KUScfResult};
use ferric_pbc::lattice::Cell;
use ferric_pbc::pair_ft::residues::{pair_ft_residues_chunked, pair_ft_strain_residues_chunked};
use ferric_pbc::pair_ft::{pair_ft_strain_chunked, PairFtStrainTerms};
use ferric_pbc::rsgdf::kpoint::{KRsGdf, KRsGdfConfig};
use ferric_pbc::rsgdf::RsGdfConfig;
use ferric_pbc::sr_rotation::{ColumnRotation, SrColumnRotation};
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

/// Rotation OFF: the stress walks the unrotated shells (as the k forces); the
/// energy's Auto rotation agrees to the screening precision, but the FD
/// anchors are exact only against the unrotated energy.
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
    let prep = prep_for(&cell, bs);
    let mesh = KPointMesh::gamma_centred(&cell, n).expect("mesh");
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
        rohf_spin_focks: None,
    }
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
    let e2 = (ksres.e_coulomb + ksres.e_exchange - gs.e_eri).abs();
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

/// p-shell antisymmetry floor (libint2's SR-derivative precision): measured
/// 1.1e-10 (k) on the triclinic s+p cell at 1x1x1, 10x below this bar; s-only
/// cells sit at 1e-18.
const P_ANTISYM_BAR: f64 = 1e-9;

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
        let scf = krhf(&ks, Box::new(eri.jk_builder_with_madelung(0.0)));
        let g = gsys(cell, &bs);
        let gscf = gamma_restricted(real_of(&scf.densities[0]));
        for exx in EXX {
            let k = ks_rhf(&ks, &eri, &scf, exx, None);
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
    let scf = kuhf(&ks, Box::new(eri.jk_builder_with_madelung(0.0)), 2, 1);
    let g = gsys(h3_cell(), &bs);
    let gscf = gamma_unrestricted(
        real_of(&scf.density_alpha[0]),
        real_of(&scf.density_beta[0]),
    );
    for exx in EXX {
        let k = ks_uhf(&ks, &eri, &scf, exx, None);
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
        let e = e1 + st.e_coulomb + st.e_exchange + st.e_madelung + ks.hk.enn;
        eprintln!(
            "  {exx:?}: E_pieces {e:.12} vs SCF {:.12} (diff {:.2e}); E_J {:.8} E_K {:.8} E_M {:.8}",
            scf.energy,
            e - scf.energy,
            st.e_coulomb,
            st.e_exchange,
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
) -> (KSys, KDenseAftEri, KScfResult, ScRef) {
    let nc = (n[0] * n[1] * n[2]) as f64;
    let ks = ksys(cell.clone(), bs, n);
    let eri = kdense(&ks, ANCHOR_PRECISION);
    let scf = krhf(&ks, Box::new(eri.jk_builder_with_madelung(0.0)));
    let g = gsys(supercell(&cell, n, 1), bs);
    let gscf = gamma_restricted(unfold(&ks.mesh, &scf.densities));
    let mut de = [ZERO; 2];
    for (i, exx) in EXX.into_iter().enumerate() {
        let k = ks_rhf(&ks, &eri, &scf, exx, None);
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
    (ks, eri, scf, ScRef { de })
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
/// The Gamma-style term defects (the 1x1x1 Gamma anchor already pins these
/// terms; they are re-asserted against the supercell in the `#[ignore]`d
/// sweep to keep CI cheap).
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
                    off <= SC_BAR,
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
            mn <= SC_BAR,
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
    let (ks, eri, scf, refs) = rhf_supercell_anchor("H2", h2_cell_g(), &bs, [1, 1, 3]);
    assert_mutants_fail(
        "H2 1x1x3",
        &|exx, m| ks_rhf(&ks, &eri, &scf, exx, m),
        &refs,
        &K_MUTANTS,
        &K_EWALD_MUTANTS,
    );
}

/// The Gamma-style term mutants against the same supercell anchor (the
/// 1x1x1 Gamma anchor and the hcore gate already catch them in CI).
#[test]
#[ignore = "slow (~2 min release): 22 more stress evaluations after the supercell anchor"]
fn h2_1x1x3_term_mutants_miss_the_supercell_anchor() {
    let bs = pyscf_sto3g_h();
    let (ks, eri, scf, refs) = rhf_supercell_anchor("H2", h2_cell_g(), &bs, [1, 1, 3]);
    assert_mutants_fail(
        "H2 1x1x3",
        &|exx, m| ks_rhf(&ks, &eri, &scf, exx, m),
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
    let scf = kuhf(&ks, Box::new(eri.jk_builder_with_madelung(0.0)), 2, 1);
    let g = gsys(supercell(&h3_cell(), n, 4), &bs);
    let gscf = gamma_unrestricted(
        unfold(&ks.mesh, &scf.density_alpha),
        unfold(&ks.mesh, &scf.density_beta),
    );
    let mut de = [ZERO; 2];
    for (i, exx) in EXX.into_iter().enumerate() {
        let k = ks_uhf(&ks, &eri, &scf, exx, None);
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
    let run = |exx, m| ks_uhf(&ks, &eri, &scf, exx, m);
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
    let scf = krhf(&ks, Box::new(eri.jk_builder_with_madelung(0.0)));
    for exx in EXX {
        let f0 = ks_rhf(&ks, &eri, &scf, exx, None);
        for m in [KStressMutation::ConjD, KStressMutation::WrongQ] {
            let f = ks_rhf(&ks, &eri, &scf, exx, Some(m));
            let d = max_mat(&f.de_deps, &f0.de_deps);
            eprintln!("  H2 2x2x2 {exx:?} {m:?}: |mut − ref| {d:.2e} (identity on TRIM meshes)");
            assert!(
                d <= IDENTITY_BAR,
                "{m:?} should be an identity on 2x2x2 ({d:e})"
            );
        }
        for m in [KStressMutation::NoPhase, KStressMutation::NoInvNk] {
            let f = ks_rhf(&ks, &eri, &scf, exx, Some(m));
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
    let r = krhf(&ks, Box::new(eri.jk_builder_with_madelung(0.0)));
    let mut u = kuhf(&ks, Box::new(eri.jk_builder_with_madelung(0.0)), 1, 1);
    let half: Vec<Array2<Complex64>> = r.densities.iter().map(|d| d.mapv(|z| z * 0.5)).collect();
    u.density_alpha = half.clone();
    u.density_beta = half;
    for exx in EXX {
        let sr = ks_rhf(&ks, &eri, &r, exx, None);
        let su = ks_uhf(&ks, &eri, &u, exx, None);
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
fn stress_refuses_unconverged_scf_ecp_and_explicit_rotation() {
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
    // An ECP in the hcore (the k energy supports ECPs, the stress does not).
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
    assert!(e.contains("ECP strain at k is not implemented"), "{e}");
    // Explicit column rotation request.
    let rot = hcore_cfg().with_sr_column_rotation(SrColumnRotation::On(ColumnRotation::new()));
    let e = kpoint_rhf_stress(
        &ks.cell,
        &ks.prep,
        &ks.mesh,
        &rot,
        &ks.hk,
        KStressJk::Dense(&eri),
        &scf,
        ExxDiv::None,
        &scfg(None),
    )
    .unwrap_err()
    .to_string();
    assert!(e.contains("column rotation"), "{e}");
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

/// The UHF twin of the FD anchor (H3 doublet 1x1x3, both exxdiv).
#[test]
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
    let mut fd = [[[0.0; 3]; 3]; 2];
    for (i, j) in ALL9 {
        let ep = k_energies_uhf(&strained(&ks.cell, i, j, FD_H), &bs, n);
        let em = k_energies_uhf(&strained(&ks.cell, i, j, -FD_H), &bs, n);
        for x in 0..2 {
            fd[x][i][j] = (ep[x] - em[x]) / (2.0 * FD_H);
        }
    }
    for (x, exx) in EXX.into_iter().enumerate() {
        let s = ks_uhf(&ks, &eri, &scf[x], exx, None);
        let miss = max_mat(&s.de_deps, &fd[x]);
        eprintln!("  H3 UHF 1x1x3 {exx:?}: |analytic − FD| {miss:.2e}");
        assert!(miss <= K_FD_BAR_UHF, "{exx:?}: FD miss {miss:e}");
    }
}

/// H3 has a larger h² truncation than H2 (Gamma H3 UHF prototype 5.4e-8).
const K_FD_BAR_UHF: f64 = 2e-7;

/// The RS-GDF source is REFUSED with a typed error naming the missing piece
/// (PR-2 of #236), for RHF and UHF, before any work is done.
#[test]
fn stress_refuses_rsgdf_naming_the_missing_piece() {
    let bs = pyscf_sto3g_h();
    let ks = ksys(h2_cell_g(), &bs, [1, 1, 1]);
    let eri = kdense(&ks, ANCHOR_PRECISION);
    let scf = krhf(&ks, Box::new(eri.jk_builder_with_madelung(0.0)));
    let mut shells = Vec::new();
    for a in [4.7, 1.9, 0.75, 0.3] {
        shells.push(Shell {
            l: 0,
            pure: false,
            exponents: vec![a],
            coefficients: vec![1.0],
        });
    }
    let mut m = HashMap::new();
    m.insert(1, shells);
    let aux_bs = BasisSet {
        name: "s-aux-H".into(),
        shells: m,
        ecps: HashMap::new(),
    };
    let auxp = PreparedBasis::new(ks.cell.mol(), &aux_bs).unwrap();
    let cfg = KRsGdfConfig {
        gdf: RsGdfConfig {
            omega: 1.0,
            exxdiv: ExxDiv::None,
            budget_bytes: Some(AMPLE),
            ..Default::default()
        },
        mutation: None,
    };
    let gdf = KRsGdf::build(&ks.cell, &ks.prep, &auxp, &ks.mesh, &ks.hk.s, &cfg).unwrap();
    let src = KRsGdfGradSource {
        gdf: &gdf,
        cfg: &cfg,
        aux: &auxp,
    };
    let e = kpoint_rhf_stress(
        &ks.cell,
        &ks.prep,
        &ks.mesh,
        &hcore_cfg(),
        &ks.hk,
        KStressJk::RsGdf(src),
        &scf,
        ExxDiv::None,
        &scfg(None),
    )
    .unwrap_err()
    .to_string();
    assert!(e.contains("RS-GDF") && e.contains("not implemented"), "{e}");
}
