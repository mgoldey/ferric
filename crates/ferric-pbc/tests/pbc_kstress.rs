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
use ferric_core::basis::BasisSet;
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_pbc::dense_aft::{DenseAftEri, ExxDiv, DEFAULT_DENSE_AFT_MAX_BYTES};
use ferric_pbc::hcore::kpoint::{periodic_hcore_kpts, PeriodicHcoreK};
use ferric_pbc::hcore::{periodic_hcore, PeriodicHcore, PeriodicHcoreConfig};
use ferric_pbc::kdense_aft::{KDenseAftConfig, KDenseAftEri};
use ferric_pbc::kpts::KPointMesh;
use ferric_pbc::kscf::{solve_krhf_injected, KPointInjection, KPointJk, KScfConfig, KScfResult};
use ferric_pbc::kstress::{kpoint_hcore_stress_fixed_d, KStressConfig, KStressMutation};
use ferric_pbc::lattice::Cell;
use ferric_pbc::pair_ft::residues::{pair_ft_residues_chunked, pair_ft_strain_residues_chunked};
use ferric_pbc::pair_ft::{pair_ft_strain_chunked, PairFtStrainTerms};
use ferric_pbc::stress::{gamma_rhf_stress, GammaStressConfig};
use ferric_scf::result::{ScfExit, ScfResult, Spin};
use ndarray::{Array2, Array3};
use num_complex::Complex64;

type Mat3 = [[f64; 3]; 3];

const OMEGA: f64 = 0.8;
const HCORE_PRECISION: f64 = 1e-14;
/// Dense-AFT precision of the fixed-density anchors (see `pbc_kgrad.rs`).
const ANCHOR_PRECISION: f64 = 1e-12;
const AMPLE: usize = 1 << 31;
const FD_H: f64 = 1e-4;
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

fn hcore_cfg() -> PeriodicHcoreConfig {
    PeriodicHcoreConfig {
        precision: HCORE_PRECISION,
        ..PeriodicHcoreConfig::with_omega(OMEGA)
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
    for (name, cell, bs) in [
        ("H2", h2_cell_g(), pyscf_sto3g_h()),
        ("tri s+p", triclinic_cell(), sp_basis_h()),
    ] {
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
