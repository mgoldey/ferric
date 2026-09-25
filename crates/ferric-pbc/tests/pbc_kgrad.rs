//! k-point RHF / UHF analytic forces (`ferric_pbc::kgrad`), the Rust port of
//! `reference/pbc/pbc_kgrad.py` + `pbc_kgrad_gdf.py` (FINDINGS "Iteration 21
//! (Python, k-point RHF/UHF forces)").
//!
//! What is anchored against what (every anchor is taken at ONE fixed
//! density, so SCF convergence does not enter the comparison):
//! (1) 1×1×1 mesh ≡ the Gamma force (`gamma_rhf_gradient_with` /
//!     `gamma_uhf_gradient_with`, and the RS-GDF twins) at the same density,
//!     both exxdiv, ≤ 1e-12 (prototype 4.7e-16..2.8e-15; RS-GDF 1.7e-13).
//!     An independent construction: real half-sphere Gamma code vs complex
//!     full-sphere residue code, image-summed vs phase-folded weights.
//! (2) SUPERCELL ANCHOR: the k-mesh force on atom A ≡ the Gamma force of the
//!     explicit diag(N) supercell on EVERY copy of A, at the k density
//!     unfolded to the supercell (`D_sc[(c,m),(c',n)] = (1/N_k) Σ_k
//!     e^{ik·(T_c − T_c')} D(k)_mn`). Prototype ≤ 1.6e-13 dense, ≤ 6e-12
//!     RS-GDF. NOT the N× sum over copies (printed: it is (N−1)× off). The
//!     Rust `h(k)` (SR erfc V_ne on Gaussian nuclei + LR + c0 Z S(k)) was NOT
//!     prototyped at k — this anchor is what pins its derivative, since the
//!     supercell side runs the Gamma `h` derivative of `crate::grad`.
//!     Meshes: H2 1×1×3 and 2×2×2 RHF, H3 1×1×3 UHF (2,1), triclinic s+p
//!     tri2 1×1×3 RHF (`#[ignore]`d, cost unmeasured).
//! (3) Analytic vs central FD (h = 1e-4) of ferric's OWN k energy, H2 1×1×3,
//!     both exxdiv (dense and RS-GDF). The prototype's floor is the h²
//!     truncation (−2.99e-9 on (0,z), Richardson 3e-12) with a pure-AFT h;
//!     the Rust Ewald-split h adds the smeared-nucleus derivative
//!     (`GRAD_NUCLEUS_EXPONENT` 1e10 vs the energy's 1e16), so the bar is
//!     the Rust Gamma FD bar 1e-7 (`pbc_grad.rs`), residuals printed.
//! (4) ΣF ≤ 1e-10; F(ewald) ≡ F(none) on one density ≤ 1e-11; closed-shell
//!     kUHF(D/2, D/2) ≡ kRHF(D) ≤ 1e-12.
//! (5) Mutations (`KGradMutation`), miss vs the supercell anchor on 1×1×3
//!     (> 1e-4; prototype H2 1×1×3: NoPhase 9.0e-2, ConjD 7.4e-3 / 4.7e-4
//!     (none / ewald), WrongQ 9.1e-3, GammaMadelung 1.9e-1 (ewald only),
//!     NoInvNk 7.8e-1; RS-GDF FitAuxPhase 7.7e-4, FitNoMetric 4.6e-2,
//!     FitNoG0 2.8e-2). BLIND SPOT, asserted: on the TRIM-only 2×2×2 mesh
//!     `D(k)` is real and `q ≡ −q`, so ConjD and WrongQ are IDENTITIES
//!     (|ΔF| ≤ 1e-12) — a port validated only on n_i ≤ 2 meshes has not
//!     tested its phases. GammaMadelung is blind under exxdiv = none
//!     (asserted).
//!
//! Note: on H2 at TRIM meshes the SCF converges in one iteration (inversion
//! symmetry fixes the one occupied band per TRIM k); those rows test the
//! force formula, not SCF response. H2 1×1×3, H3 and tri2 carry the SCF.

mod common;

use common::*;
use ferric_core::basis::{BasisSet, Shell};
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_pbc::dense_aft::{
    DenseAftEri, ExxDiv, DEFAULT_DENSE_AFT_MAX_BYTES, DEFAULT_DENSE_AFT_PRECISION,
};
use ferric_pbc::ewald::madelung_constant;
use ferric_pbc::grad::{
    gamma_rhf_gradient_rsgdf, gamma_rhf_gradient_with, gamma_uhf_gradient_rsgdf,
    gamma_uhf_gradient_with, GammaGradConfig, RsGdfGradSource,
};
use ferric_pbc::hcore::kpoint::{periodic_hcore_kpts, PeriodicHcoreK};
use ferric_pbc::hcore::{periodic_hcore, PeriodicHcore, PeriodicHcoreConfig};
use ferric_pbc::kdense_aft::{KDenseAftConfig, KDenseAftEri};
use ferric_pbc::kgrad::{
    kpoint_rhf_gradient, kpoint_uhf_gradient, KGradConfig, KGradJk, KGradMutation, KGradient,
    KRsGdfGradSource,
};
use ferric_pbc::kpts::KPointMesh;
use ferric_pbc::kscf::{solve_krhf_injected, KPointInjection, KPointJk, KScfConfig, KScfResult};
use ferric_pbc::kuscf::{solve_kuhf_injected, KUScfResult};
use ferric_pbc::lattice::Cell;
use ferric_pbc::rsgdf::kpoint::{KRsGdf, KRsGdfConfig};
use ferric_pbc::rsgdf::{RsGdf, RsGdfConfig};
use ferric_scf::result::{ScfExit, ScfResult, Spin};
use ndarray::Array2;
use num_complex::Complex64;
use std::collections::HashMap;

/// ω of the nuclear-attraction split (as `pbc_grad.rs`, `pbc_krhf.rs`).
const OMEGA: f64 = 0.8;
const HCORE_PRECISION: f64 = 1e-14;
/// Dense-AFT precision of the fixed-density anchors (the K set and the pair
/// screen are the same on both sides at any precision; the loose value
/// keeps the supercell tensors cheap).
/// Pair-FT screening for the dense anchors. The k-point and Gamma dense
/// derivatives screen pair FTs differently, so their difference is a
/// truncation term proportional to this threshold (measured 2026-09-25, H3 UHF
/// 1x1x1 vs Gamma: 1.05e-9 / 1.41e-11 / 1.42e-13 at 1e-8 / 1e-10 / 1e-12;
/// RS-GDF, which shares one screen, agrees to 3e-14 at any setting). 1e-12
/// makes the anchors test the formula, not the screen.
const ANCHOR_PRECISION: f64 = 1e-12;
const AMPLE: usize = 1 << 31;
const FD_H: f64 = 1e-4;
const FD_BAR: f64 = 1e-7;
const GAMMA_BAR: f64 = 1e-12;
const GAMMA_BAR_GDF: f64 = 1e-12;
const SC_BAR: f64 = 1e-12;
const SC_BAR_GDF: f64 = 1e-11;
const NET_FORCE_BAR: f64 = 1e-10;
const EWALD_NONE_BAR: f64 = 1e-11;
const IDENTITY_BAR: f64 = 1e-12;
const MUTANT_BAR: f64 = 1e-4;

/// `run_kgrad_anchor.py` H2 (Bohr), a = 4, STO-3G: off-axis so every
/// Cartesian component is non-trivial.
const H2_ATOMS: [[f64; 3]; 2] = [[0.3, 0.2, 0.1], [0.35, 0.12, 1.5]];
const H2_COMPS: [(usize, usize); 3] = [(0, 0), (0, 2), (1, 1)];
/// H3 (Bohr), a = 4.5, STO-3G, doublet (2, 1) per cell.
const H3_ATOMS: [[f64; 3]; 3] = [[0.3, 0.2, 0.1], [0.35, 0.12, 1.5], [1.6, 0.9, 0.7]];
const H3_A: f64 = 4.5;
/// tri2: TRI_A with the first two TRI_MOVED hydrogens, s+p basis (nao 8).
const TRI2_ATOMS: [[f64; 3]; 2] = [[0.13, 0.25, 0.31], [0.02, 0.27, 1.66]];

const MESH_MUTANTS: [KGradMutation; 4] = [
    KGradMutation::NoPhase,
    KGradMutation::ConjD,
    KGradMutation::WrongQ,
    KGradMutation::NoInvNk,
];
const FIT_MUTANTS: [KGradMutation; 3] = [
    KGradMutation::FitAuxPhase,
    KGradMutation::FitNoMetric,
    KGradMutation::FitNoG0,
];
const EXX: [ExxDiv; 2] = [ExxDiv::None, ExxDiv::Ewald];

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

fn kgdf_cfg() -> KRsGdfConfig {
    KRsGdfConfig {
        gdf: RsGdfConfig {
            omega: 1.0,
            exxdiv: ExxDiv::None,
            budget_bytes: Some(AMPLE),
            ..Default::default()
        },
        mutation: None,
    }
}

fn gcfg(mutation: Option<KGradMutation>) -> KGradConfig {
    KGradConfig {
        budget_bytes: Some(AMPLE),
        mutation,
        ..Default::default()
    }
}

fn gamma_gcfg() -> GammaGradConfig {
    GammaGradConfig {
        budget_bytes: Some(AMPLE),
        ..Default::default()
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

fn cell_at(pos: &[[f64; 3]], lattice: [[f64; 3]; 3], mult: usize) -> Cell {
    let mut mol: Molecule = hydrogens(pos);
    mol.multiplicity = mult;
    Cell::new(mol, lattice).expect("cell")
}

fn moved(pos: &[[f64; 3]], a: usize, x: usize, h: f64) -> Vec<[f64; 3]> {
    let mut p = pos.to_vec();
    p[a][x] += h;
    p
}

fn h2_cell_g() -> Cell {
    cell_at(&H2_ATOMS, cubic(4.0), 1)
}

fn h3_cell() -> Cell {
    cell_at(&H3_ATOMS, cubic(H3_A), 2)
}

fn tri2_cell() -> Cell {
    cell_at(&TRI2_ATOMS, TRI_A, 1)
}

/// Explicit diag(n) supercell (copy c = (m0, m1, m2), m0 outer; atoms of
/// copy c at `R + Σ m_i a_i`) with multiplicity `mult`.
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
        df_jk: None,
        rohf_spin_focks: None,
    }
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
        df_jk: None,
        rohf_spin_focks: None,
    }
}

fn real_of(d: &Array2<Complex64>) -> Array2<f64> {
    let im = d.iter().fold(0.0_f64, |m, z| m.max(z.im.abs()));
    assert!(im < 1e-12, "Gamma density has an imaginary part {im:e}");
    d.mapv(|z| z.re)
}

fn max_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    assert_eq!(a.dim(), b.dim());
    a.iter()
        .zip(b.iter())
        .fold(0.0_f64, |m, (x, y)| m.max((x - y).abs()))
}

/// `max_{c, A, x} |F_k[A, x] − F_sc[c·natoms + A, x]|` (every copy).
fn sc_miss(gk: &Array2<f64>, gsc: &Array2<f64>) -> f64 {
    let na = gk.nrows();
    let nc = gsc.nrows() / na;
    assert_eq!(nc * na, gsc.nrows());
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

/// Sum of the supercell forces over the copies of each atom (what the
/// mapping is NOT).
fn copy_sum(gsc: &Array2<f64>, na: usize) -> Array2<f64> {
    let nc = gsc.nrows() / na;
    Array2::from_shape_fn((na, 3), |(a, x)| {
        (0..nc).map(|c| gsc[(c * na + a, x)]).sum()
    })
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

fn kgdf(ks: &KSys, aux: &PreparedBasis) -> KRsGdf {
    KRsGdf::build(&ks.cell, &ks.prep, aux, &ks.mesh, &ks.hk.s, &kgdf_cfg()).expect("KRsGdf")
}

fn inj<'a>(ks: &KSys, jk: Box<dyn KPointJk + 'a>) -> KPointInjection<'a> {
    KPointInjection {
        s: ks.hk.s.clone(),
        h: ks.hk.h.clone(),
        vnn: ks.hk.enn,
        jk,
    }
}

fn krhf(ks: &KSys, jk: Box<dyn KPointJk + '_>) -> KScfResult {
    let r = solve_krhf_injected(&ks.cell, &ks.mesh, &kscf_cfg(), inj(ks, jk)).expect("k-RHF");
    assert!(r.converged, "k-RHF not converged ({} it)", r.iterations);
    r
}

fn kuhf(ks: &KSys, jk: Box<dyn KPointJk + '_>, na: usize, nb: usize) -> KUScfResult {
    let r =
        solve_kuhf_injected(&ks.cell, &ks.mesh, &kscf_cfg(), inj(ks, jk), na, nb).expect("k-UHF");
    assert!(r.converged, "k-UHF not converged ({} it)", r.iterations);
    r
}

fn kg_rhf(
    ks: &KSys,
    jk: KGradJk<'_>,
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
        jk,
        scf,
        exx,
        &gcfg(m),
    )
    .expect("kpoint_rhf_gradient")
}

fn kg_uhf(
    ks: &KSys,
    jk: KGradJk<'_>,
    scf: &KUScfResult,
    exx: ExxDiv,
    m: Option<KGradMutation>,
) -> KGradient {
    kpoint_uhf_gradient(
        &ks.cell,
        &ks.prep,
        &ks.mesh,
        &hcore_cfg(),
        &ks.hk,
        jk,
        scf,
        exx,
        &gcfg(m),
    )
    .expect("kpoint_uhf_gradient")
}

/// Gamma-point setup of a cell (the explicit supercell, or the cell itself
/// for the 1×1×1 anchor): hcore + dense tensor, optionally RS-GDF.
struct GSys {
    cell: Cell,
    prep: PreparedBasis,
    hc: PeriodicHcore,
    eri: Option<DenseAftEri>,
    gdf: Option<(PreparedBasis, RsGdf)>,
}

fn gsys(cell: Cell, bs: &BasisSet, precision: Option<f64>, aux: Option<&BasisSet>) -> GSys {
    let prep = prep_for(&cell, bs);
    let hc = periodic_hcore(&cell, &prep, &hcore_cfg()).expect("gamma hcore");
    let eri = precision.map(|p| {
        DenseAftEri::build(
            &cell,
            &prep,
            &hc.s,
            ExxDiv::None,
            p,
            DEFAULT_DENSE_AFT_MAX_BYTES,
        )
        .expect("gamma dense AFT")
    });
    let gdf = aux.map(|a| {
        let auxp = PreparedBasis::new(cell.mol(), a).expect("aux prep");
        let g = RsGdf::build_for_gradient(&cell, &prep, &auxp, &hc.s, &kgdf_cfg().gdf)
            .expect("RsGdf::build_for_gradient");
        (auxp, g)
    });
    GSys {
        cell,
        prep,
        hc,
        eri,
        gdf,
    }
}

fn gamma_grad(g: &GSys, scf: &ScfResult, exx: ExxDiv) -> Array2<f64> {
    let restricted = scf.spin == Spin::Restricted;
    match (&g.eri, &g.gdf) {
        (Some(eri), _) => {
            if restricted {
                gamma_rhf_gradient_with(
                    &g.cell,
                    &g.prep,
                    &hcore_cfg(),
                    &g.hc,
                    eri,
                    scf,
                    exx,
                    &gamma_gcfg(),
                )
            } else {
                gamma_uhf_gradient_with(
                    &g.cell,
                    &g.prep,
                    &hcore_cfg(),
                    &g.hc,
                    eri,
                    scf,
                    exx,
                    &gamma_gcfg(),
                )
            }
        }
        (None, Some((auxp, gdf))) => {
            let src = RsGdfGradSource {
                gdf,
                aux: auxp,
                aux_jac: None,
            };
            if restricted {
                gamma_rhf_gradient_rsgdf(
                    &g.cell,
                    &g.prep,
                    &hcore_cfg(),
                    &g.hc,
                    &src,
                    scf,
                    exx,
                    &gamma_gcfg(),
                )
            } else {
                gamma_uhf_gradient_rsgdf(
                    &g.cell,
                    &g.prep,
                    &hcore_cfg(),
                    &g.hc,
                    &src,
                    scf,
                    exx,
                    &gamma_gcfg(),
                )
            }
        }
        _ => panic!("GSys without J/K"),
    }
    .expect("gamma gradient")
    .grad
}

fn report(tag: &str, g: &KGradient) {
    eprintln!(
        "  {tag}: |F| {:.3e} ΣF {:.1e} comm {:.1e} v_M {:.6} (images {}, SR {}, G_lr {}, K {}, \
         chunks {}, dropped {:?})",
        g.grad.iter().fold(0.0_f64, |m, v| m.max(v.abs())),
        g.net_force,
        g.commutator,
        g.madelung,
        g.n_images,
        g.n_sr_triplets,
        g.n_g_lr,
        g.n_k_eri,
        g.n_chunks,
        g.fit_dropped_max
    );
}

// ======================================================= (1) 1×1×1 ≡ Gamma

#[test]
fn one_point_mesh_rhf_is_the_gamma_force() {
    for (name, cell, bs) in [
        ("H2", h2_cell_g(), pyscf_sto3g_h()),
        ("tri2 s+p", tri2_cell(), sp_basis_h()),
    ] {
        let ks = ksys(cell.clone(), &bs, [1, 1, 1]);
        let eri = kdense(&ks, ANCHOR_PRECISION);
        let scf = krhf(&ks, Box::new(eri.jk_builder_with_madelung(0.0)));
        let g = gsys(cell, &bs, Some(ANCHOR_PRECISION), None);
        let d = real_of(&scf.densities[0]);
        let gscf = gamma_restricted(d);
        for exx in EXX {
            let fk = kg_rhf(&ks, KGradJk::Dense(&eri), &scf, exx, None);
            let fg = gamma_grad(&g, &gscf, exx);
            let dd = max_diff(&fk.grad, &fg);
            report(&format!("{name} 1x1x1 {exx:?}"), &fk);
            eprintln!("  {name} 1x1x1 {exx:?}: |F_k − F_gamma| {dd:.2e} (prototype 4.7e-16)");
            assert!(dd <= GAMMA_BAR, "{name} {exx:?}: {dd:e}");
            assert!(
                fk.net_force <= NET_FORCE_BAR,
                "{name}: ΣF {:e}",
                fk.net_force
            );
        }
    }
}

#[test]
fn one_point_mesh_uhf_is_the_gamma_force() {
    let bs = pyscf_sto3g_h();
    let ks = ksys(h3_cell(), &bs, [1, 1, 1]);
    let eri = kdense(&ks, ANCHOR_PRECISION);
    let scf = kuhf(&ks, Box::new(eri.jk_builder_with_madelung(0.0)), 2, 1);
    let g = gsys(h3_cell(), &bs, Some(ANCHOR_PRECISION), None);
    let gscf = gamma_unrestricted(
        real_of(&scf.density_alpha[0]),
        real_of(&scf.density_beta[0]),
    );
    for exx in EXX {
        let fk = kg_uhf(&ks, KGradJk::Dense(&eri), &scf, exx, None);
        let fg = gamma_grad(&g, &gscf, exx);
        let dd = max_diff(&fk.grad, &fg);
        report(&format!("H3 UHF 1x1x1 {exx:?}"), &fk);
        eprintln!("  H3 UHF 1x1x1 {exx:?}: |F_k − F_gamma| {dd:.2e} (prototype 2.8e-15)");
        assert!(dd <= GAMMA_BAR, "{exx:?}: {dd:e}");
    }
}

#[test]
fn one_point_mesh_rsgdf_is_the_gamma_rsgdf_force() {
    let bs = pyscf_sto3g_h();
    let aux_bs = et_sp_aux();
    // RHF H2 and UHF H3.
    for (name, cell, uhf) in [("H2 RHF", h2_cell_g(), false), ("H3 UHF", h3_cell(), true)] {
        let ks = ksys(cell.clone(), &bs, [1, 1, 1]);
        let auxp = PreparedBasis::new(cell.mol(), &aux_bs).unwrap();
        let gdf = kgdf(&ks, &auxp);
        let cfg = kgdf_cfg();
        let src = KRsGdfGradSource {
            gdf: &gdf,
            cfg: &cfg,
            aux: &auxp,
        };
        let g = gsys(cell, &bs, None, Some(&aux_bs));
        for exx in EXX {
            let (fk, fg) = if uhf {
                let scf = kuhf(&ks, Box::new(gdf.jk_builder_with_madelung(0.0)), 2, 1);
                let gscf = gamma_unrestricted(
                    real_of(&scf.density_alpha[0]),
                    real_of(&scf.density_beta[0]),
                );
                (
                    kg_uhf(&ks, KGradJk::RsGdf(src), &scf, exx, None),
                    gamma_grad(&g, &gscf, exx),
                )
            } else {
                let scf = krhf(&ks, Box::new(gdf.jk_builder_with_madelung(0.0)));
                let gscf = gamma_restricted(real_of(&scf.densities[0]));
                (
                    kg_rhf(&ks, KGradJk::RsGdf(src), &scf, exx, None),
                    gamma_grad(&g, &gscf, exx),
                )
            };
            let dd = max_diff(&fk.grad, &fg);
            report(&format!("{name} RS-GDF 1x1x1 {exx:?}"), &fk);
            eprintln!(
                "  {name} RS-GDF 1x1x1 {exx:?}: |F_k − F_gamma| {dd:.2e} (prototype 1.7e-13..2.1e-13)"
            );
            assert!(dd <= GAMMA_BAR_GDF, "{name} {exx:?}: {dd:e}");
        }
    }
}

// ============================================ (2) + (4) + (5) supercell anchor

/// Dense supercell anchor for an RHF system: k force vs every copy, both
/// exxdiv, at the unfolded k density; ΣF; ewald ≡ none. Returns the setup,
/// SCF and the supercell reference forces `[none, ewald]` for the mutants.
fn rhf_supercell_anchor(
    name: &str,
    cell: Cell,
    bs: &BasisSet,
    n: [usize; 3],
) -> (KSys, KDenseAftEri, KScfResult, [Array2<f64>; 2]) {
    let ks = ksys(cell.clone(), bs, n);
    let eri = kdense(&ks, ANCHOR_PRECISION);
    let scf = krhf(&ks, Box::new(eri.jk_builder_with_madelung(0.0)));
    eprintln!(
        "{name} {n:?}: k-RHF {} it, E {:.12}, nocc/k {:?}",
        scf.iterations, scf.energy, scf.nocc_per_k
    );
    let g = gsys(supercell(&cell, n, 1), bs, Some(ANCHOR_PRECISION), None);
    let gscf = gamma_restricted(unfold(&ks.mesh, &scf.densities));
    let vm_sc = madelung_constant(&g.cell).unwrap();
    let vm_k = ks.mesh.madelung(&ks.cell).unwrap();
    assert!(
        (vm_sc - vm_k).abs() < 1e-13,
        "v_M mesh {vm_k} vs supercell {vm_sc}"
    );
    let mut refs: Vec<Array2<f64>> = Vec::new();
    let mut fks: Vec<Array2<f64>> = Vec::new();
    for exx in EXX {
        let fk = kg_rhf(&ks, KGradJk::Dense(&eri), &scf, exx, None);
        let fsc = gamma_grad(&g, &gscf, exx);
        let miss = sc_miss(&fk.grad, &fsc);
        let nsum = max_diff(&copy_sum(&fsc, fk.grad.nrows()), &fk.grad);
        report(&format!("{name} {n:?} {exx:?}"), &fk);
        eprintln!(
            "  {name} {n:?} {exx:?}: max_c |F_k − F_sc(copy c)| {miss:.2e} (N× sum − F_k {nsum:.2e})"
        );
        assert!(miss <= SC_BAR, "{name} {exx:?}: supercell miss {miss:e}");
        assert!(
            fk.net_force <= NET_FORCE_BAR,
            "{name}: ΣF {:e}",
            fk.net_force
        );
        fks.push(fk.grad);
        refs.push(fsc);
    }
    let de = max_diff(&fks[0], &fks[1]);
    eprintln!("  {name} {n:?}: |F(ewald) − F(none)| {de:.2e}");
    assert!(de <= EWALD_NONE_BAR, "{name}: ewald vs none {de:e}");
    let r1 = refs.pop().unwrap();
    let r0 = refs.pop().unwrap();
    (ks, eri, scf, [r0, r1])
}

/// Each mesh mutant must miss the supercell anchor by > MUTANT_BAR (both
/// exxdiv), GammaMadelung under ewald only (and be blind under none).
fn assert_rhf_mutants_fail(
    name: &str,
    ks: &KSys,
    eri: &KDenseAftEri,
    scf: &KScfResult,
    refs: &[Array2<f64>; 2],
) {
    for m in MESH_MUTANTS {
        for (i, exx) in EXX.into_iter().enumerate() {
            let f = kg_rhf(ks, KGradJk::Dense(eri), scf, exx, Some(m));
            let miss = sc_miss(&f.grad, &refs[i]);
            eprintln!(
                "  {name} mutant {m:?} {exx:?}: miss {miss:.2e} (ΣF {:.1e})",
                f.net_force
            );
            assert!(
                miss > MUTANT_BAR,
                "{name}: mutant {m:?} {exx:?} not caught ({miss:e})"
            );
        }
    }
    let m = KGradMutation::GammaMadelung;
    let fe = kg_rhf(ks, KGradJk::Dense(eri), scf, ExxDiv::Ewald, Some(m));
    let fn_ = kg_rhf(ks, KGradJk::Dense(eri), scf, ExxDiv::None, Some(m));
    let (me, mn) = (sc_miss(&fe.grad, &refs[1]), sc_miss(&fn_.grad, &refs[0]));
    eprintln!("  {name} mutant {m:?}: ewald miss {me:.2e}, none miss {mn:.2e} (blind)");
    assert!(
        me > MUTANT_BAR,
        "{name}: GammaMadelung (ewald) not caught ({me:e})"
    );
    assert!(
        mn <= SC_BAR,
        "{name}: GammaMadelung must be blind under none ({mn:e})"
    );
}

#[test]
fn h2_1x1x3_force_is_the_supercell_force_on_one_copy() {
    let (ks, eri, scf, refs) = rhf_supercell_anchor("H2", h2_cell_g(), &pyscf_sto3g_h(), [1, 1, 3]);
    assert_rhf_mutants_fail("H2 1x1x3", &ks, &eri, &scf, &refs);
}

#[test]
fn h2_2x2x2_force_is_the_supercell_force_and_trim_mesh_is_blind_to_phase_mutants() {
    let (ks, eri, scf, refs) = rhf_supercell_anchor("H2", h2_cell_g(), &pyscf_sto3g_h(), [2, 2, 2]);
    // BLIND SPOT (FINDINGS Iteration 21): every n_i <= 2 ⇒ every k is a
    // TRIM, D(k) is real and q ≡ −q, so ConjD and WrongQ are algebraic
    // identities. A port validated on such meshes has NOT tested its phases;
    // the 1×1×3 tests above are the ones that do.
    for exx in EXX {
        let f0 = kg_rhf(&ks, KGradJk::Dense(&eri), &scf, exx, None);
        for m in [KGradMutation::ConjD, KGradMutation::WrongQ] {
            let f = kg_rhf(&ks, KGradJk::Dense(&eri), &scf, exx, Some(m));
            let d = max_diff(&f.grad, &f0.grad);
            eprintln!("  H2 2x2x2 {exx:?} {m:?}: |F_mut − F| {d:.2e} (identity on TRIM meshes)");
            assert!(
                d <= IDENTITY_BAR,
                "{m:?} should be an identity on 2x2x2 ({d:e})"
            );
        }
    }
    // NoPhase and NoInvNk are NOT blind there (prototype 6.7e-2 / 3.4).
    for m in [KGradMutation::NoPhase, KGradMutation::NoInvNk] {
        let f = kg_rhf(&ks, KGradJk::Dense(&eri), &scf, ExxDiv::None, Some(m));
        let miss = sc_miss(&f.grad, &refs[0]);
        eprintln!("  H2 2x2x2 {m:?}: miss {miss:.2e}");
        assert!(miss > MUTANT_BAR, "{m:?} not caught on 2x2x2 ({miss:e})");
    }
}

#[test]
#[ignore = "slow (unmeasured): 6-atom s+p supercell SR sum at 1e-14; run with --ignored"]
fn triclinic_sp_1x1x3_force_is_the_supercell_force_on_one_copy() {
    let (ks, eri, scf, refs) =
        rhf_supercell_anchor("tri2 s+p", tri2_cell(), &sp_basis_h(), [1, 1, 3]);
    assert_rhf_mutants_fail("tri2 1x1x3", &ks, &eri, &scf, &refs);
}

#[test]
fn h3_uhf_1x1x3_force_is_the_supercell_force_on_one_copy() {
    let bs = pyscf_sto3g_h();
    let n = [1, 1, 3];
    let ks = ksys(h3_cell(), &bs, n);
    let eri = kdense(&ks, ANCHOR_PRECISION);
    let scf = kuhf(&ks, Box::new(eri.jk_builder_with_madelung(0.0)), 2, 1);
    eprintln!(
        "H3 UHF {n:?}: {} it, E {:.12}, nocc/k α {:?} β {:?}",
        scf.iterations, scf.energy, scf.nocc_per_k_alpha, scf.nocc_per_k_beta
    );
    // Supercell (6, 3): multiplicity 4.
    let g = gsys(
        supercell(&h3_cell(), n, 4),
        &bs,
        Some(ANCHOR_PRECISION),
        None,
    );
    let gscf = gamma_unrestricted(
        unfold(&ks.mesh, &scf.density_alpha),
        unfold(&ks.mesh, &scf.density_beta),
    );
    let mut refs: Vec<Array2<f64>> = Vec::new();
    let mut fks: Vec<Array2<f64>> = Vec::new();
    for exx in EXX {
        let fk = kg_uhf(&ks, KGradJk::Dense(&eri), &scf, exx, None);
        let fsc = gamma_grad(&g, &gscf, exx);
        let miss = sc_miss(&fk.grad, &fsc);
        report(&format!("H3 UHF {n:?} {exx:?}"), &fk);
        eprintln!("  H3 UHF {n:?} {exx:?}: supercell miss {miss:.2e} (prototype 7.2e-14)");
        assert!(miss <= SC_BAR, "{exx:?}: {miss:e}");
        assert!(fk.net_force <= NET_FORCE_BAR, "ΣF {:e}", fk.net_force);
        fks.push(fk.grad);
        refs.push(fsc);
    }
    let de = max_diff(&fks[0], &fks[1]);
    eprintln!("  H3 UHF: |F(ewald) − F(none)| {de:.2e}");
    assert!(de <= EWALD_NONE_BAR, "ewald vs none {de:e}");
    for m in MESH_MUTANTS {
        for (i, exx) in EXX.into_iter().enumerate() {
            let f = kg_uhf(&ks, KGradJk::Dense(&eri), &scf, exx, Some(m));
            let miss = sc_miss(&f.grad, &refs[i]);
            eprintln!(
                "  H3 UHF mutant {m:?} {exx:?}: miss {miss:.2e} (ΣF {:.1e})",
                f.net_force
            );
            assert!(
                miss > MUTANT_BAR,
                "mutant {m:?} {exx:?} not caught ({miss:e})"
            );
        }
    }
    let f = kg_uhf(
        &ks,
        KGradJk::Dense(&eri),
        &scf,
        ExxDiv::Ewald,
        Some(KGradMutation::GammaMadelung),
    );
    let miss = sc_miss(&f.grad, &refs[1]);
    eprintln!("  H3 UHF mutant GammaMadelung ewald: miss {miss:.2e}");
    assert!(miss > MUTANT_BAR, "GammaMadelung not caught ({miss:e})");
}

#[test]
fn closed_shell_kuhf_force_is_the_krhf_force() {
    let bs = pyscf_sto3g_h();
    let ks = ksys(h2_cell_g(), &bs, [1, 1, 3]);
    let eri = kdense(&ks, ANCHOR_PRECISION);
    let r = krhf(&ks, Box::new(eri.jk_builder_with_madelung(0.0)));
    let mut u = kuhf(&ks, Box::new(eri.jk_builder_with_madelung(0.0)), 1, 1);
    let half: Vec<Array2<Complex64>> = r.densities.iter().map(|d| d.mapv(|z| z * 0.5)).collect();
    u.density_alpha = half.clone();
    u.density_beta = half;
    for exx in EXX {
        let fr = kg_rhf(&ks, KGradJk::Dense(&eri), &r, exx, None);
        let fu = kg_uhf(&ks, KGradJk::Dense(&eri), &u, exx, None);
        let d = max_diff(&fr.grad, &fu.grad);
        eprintln!("  H2 1x1x3 {exx:?}: |F_kUHF(D/2,D/2) − F_kRHF(D)| {d:.2e}");
        assert!(d <= IDENTITY_BAR, "{exx:?}: {d:e}");
    }
}

// ================================================== (3) FD of own k energy

fn fd_rows<F>(
    comps: &[(usize, usize)],
    pos: &[[f64; 3]],
    energies: F,
) -> Vec<((usize, usize), [f64; 2])>
where
    F: Fn(&[[f64; 3]]) -> [f64; 2],
{
    comps
        .iter()
        .map(|&(a, x)| {
            let ep = energies(&moved(pos, a, x, FD_H));
            let em = energies(&moved(pos, a, x, -FD_H));
            (
                (a, x),
                [
                    (ep[0] - em[0]) / (2.0 * FD_H),
                    (ep[1] - em[1]) / (2.0 * FD_H),
                ],
            )
        })
        .collect()
}

/// Dense k-RHF energies `[none, ewald]` of H2 at `pos`, 1×1×3, default
/// (converged) precision.
fn h2_dense_energies(pos: &[[f64; 3]]) -> [f64; 2] {
    let ks = ksys(cell_at(pos, cubic(4.0), 1), &pyscf_sto3g_h(), [1, 1, 3]);
    let eri = kdense(&ks, DEFAULT_DENSE_AFT_PRECISION);
    let vm = ks.mesh.madelung(&ks.cell).unwrap();
    [
        krhf(&ks, Box::new(eri.jk_builder_with_madelung(0.0))).energy,
        krhf(&ks, Box::new(eri.jk_builder_with_madelung(vm))).energy,
    ]
}

#[test]
fn h2_1x1x3_dense_force_matches_fd_of_own_k_energy() {
    let ks = ksys(h2_cell_g(), &pyscf_sto3g_h(), [1, 1, 3]);
    let eri = kdense(&ks, DEFAULT_DENSE_AFT_PRECISION);
    let vm = ks.mesh.madelung(&ks.cell).unwrap();
    let scf = [
        krhf(&ks, Box::new(eri.jk_builder_with_madelung(0.0))),
        krhf(&ks, Box::new(eri.jk_builder_with_madelung(vm))),
    ];
    let ident = scf[0].energy - scf[1].energy - vm;
    eprintln!("H2 1x1x3: E(none) − E(ewald) − nocc v_M = {ident:.1e}");
    assert!(ident.abs() < 1e-10, "{ident:e}");
    let fd = fd_rows(&H2_COMPS, &H2_ATOMS, h2_dense_energies);
    for (i, exx) in EXX.into_iter().enumerate() {
        let g = kg_rhf(&ks, KGradJk::Dense(&eri), &scf[i], exx, None);
        report(&format!("H2 1x1x3 dense {exx:?}"), &g);
        let mut worst = 0.0_f64;
        for ((a, x), v) in &fd {
            let d = g.grad[(*a, *x)] - v[i];
            eprintln!(
                "  ({a},{x}) {exx:?}: analytic {:+.10e} FD {:+.10e} diff {d:+.2e} (prototype h² floor −3e-9 on (0,z))",
                g.grad[(*a, *x)],
                v[i]
            );
            worst = worst.max(d.abs());
        }
        assert!(worst <= FD_BAR, "{exx:?}: FD miss {worst:e}");
        assert!(g.net_force <= NET_FORCE_BAR, "ΣF {:e}", g.net_force);
    }
}

/// RS-GDF k-RHF energies `[none, ewald]` of H2 at `pos`, 1×1×3.
fn h2_gdf_energies(pos: &[[f64; 3]]) -> [f64; 2] {
    let cell = cell_at(pos, cubic(4.0), 1);
    let ks = ksys(cell.clone(), &pyscf_sto3g_h(), [1, 1, 3]);
    let auxp = PreparedBasis::new(cell.mol(), &et_sp_aux()).unwrap();
    let gdf = kgdf(&ks, &auxp);
    let vm = ks.mesh.madelung(&ks.cell).unwrap();
    [
        krhf(&ks, Box::new(gdf.jk_builder_with_madelung(0.0))).energy,
        krhf(&ks, Box::new(gdf.jk_builder_with_madelung(vm))).energy,
    ]
}

#[test]
fn h2_1x1x3_rsgdf_force_matches_fd_of_own_k_energy() {
    let cell = h2_cell_g();
    let ks = ksys(cell.clone(), &pyscf_sto3g_h(), [1, 1, 3]);
    let auxp = PreparedBasis::new(cell.mol(), &et_sp_aux()).unwrap();
    let gdf = kgdf(&ks, &auxp);
    let cfg = kgdf_cfg();
    let src = KRsGdfGradSource {
        gdf: &gdf,
        cfg: &cfg,
        aux: &auxp,
    };
    let scf = krhf(&ks, Box::new(gdf.jk_builder_with_madelung(0.0)));
    let comps = [(0, 2), (1, 1)];
    let fd = fd_rows(&comps, &H2_ATOMS, h2_gdf_energies);
    for (i, exx) in EXX.into_iter().enumerate() {
        let g = kg_rhf(&ks, KGradJk::RsGdf(src), &scf, exx, None);
        report(&format!("H2 1x1x3 RS-GDF {exx:?}"), &g);
        let mut worst = 0.0_f64;
        for ((a, x), v) in &fd {
            let d = g.grad[(*a, *x)] - v[i];
            eprintln!(
                "  ({a},{x}) {exx:?}: analytic {:+.10e} FD {:+.10e} diff {d:+.2e} (prototype −2.99e-9 / +9.5e-11)",
                g.grad[(*a, *x)],
                v[i]
            );
            worst = worst.max(d.abs());
        }
        assert!(worst <= FD_BAR, "{exx:?}: FD miss {worst:e}");
    }
}

// ================================================ RS-GDF supercell anchors

#[test]
fn h2_1x1x3_rsgdf_force_is_the_supercell_rsgdf_force_and_catches_fit_mutants() {
    let n = [1, 1, 3];
    let bs = pyscf_sto3g_h();
    let aux_bs = et_sp_aux();
    let cell = h2_cell_g();
    let ks = ksys(cell.clone(), &bs, n);
    let auxp = PreparedBasis::new(cell.mol(), &aux_bs).unwrap();
    let gdf = kgdf(&ks, &auxp);
    let cfg = kgdf_cfg();
    let src = KRsGdfGradSource {
        gdf: &gdf,
        cfg: &cfg,
        aux: &auxp,
    };
    let scf = krhf(&ks, Box::new(gdf.jk_builder_with_madelung(0.0)));
    let g = gsys(supercell(&cell, n, 1), &bs, None, Some(&aux_bs));
    let gscf = gamma_restricted(unfold(&ks.mesh, &scf.densities));
    let mut refs: Vec<Array2<f64>> = Vec::new();
    let mut fks: Vec<Array2<f64>> = Vec::new();
    for exx in EXX {
        let fk = kg_rhf(&ks, KGradJk::RsGdf(src), &scf, exx, None);
        let fsc = gamma_grad(&g, &gscf, exx);
        let miss = sc_miss(&fk.grad, &fsc);
        report(&format!("H2 RS-GDF {n:?} {exx:?}"), &fk);
        eprintln!(
            "  H2 RS-GDF {n:?} {exx:?}: supercell miss {miss:.2e} (prototype 3.5e-12 / 1.6e-12)"
        );
        assert!(miss <= SC_BAR_GDF, "{exx:?}: {miss:e}");
        assert!(fk.net_force <= NET_FORCE_BAR, "ΣF {:e}", fk.net_force);
        fks.push(fk.grad);
        refs.push(fsc);
    }
    let de = max_diff(&fks[0], &fks[1]);
    eprintln!("  H2 RS-GDF: |F(ewald) − F(none)| {de:.2e} (prototype 5.5e-13)");
    assert!(de <= EWALD_NONE_BAR, "ewald vs none {de:e}");
    // WrongQ acts on the dense exchange derivative only (an identity here);
    // ConjD is unmeasured on the fitted path, so it is not asserted.
    let mesh_on_fit = [KGradMutation::NoPhase, KGradMutation::NoInvNk];
    for m in FIT_MUTANTS.into_iter().chain(mesh_on_fit) {
        let f = kg_rhf(&ks, KGradJk::RsGdf(src), &scf, ExxDiv::None, Some(m));
        let miss = sc_miss(&f.grad, &refs[0]);
        eprintln!("  H2 RS-GDF mutant {m:?}: miss {miss:.2e}");
        assert!(miss > MUTANT_BAR, "mutant {m:?} not caught ({miss:e})");
    }
}

#[test]
fn h3_uhf_1x1x3_rsgdf_force_is_the_supercell_rsgdf_force() {
    let n = [1, 1, 3];
    let bs = pyscf_sto3g_h();
    let aux_bs = et_sp_aux();
    let cell = h3_cell();
    let ks = ksys(cell.clone(), &bs, n);
    let auxp = PreparedBasis::new(cell.mol(), &aux_bs).unwrap();
    let gdf = kgdf(&ks, &auxp);
    let cfg = kgdf_cfg();
    let src = KRsGdfGradSource {
        gdf: &gdf,
        cfg: &cfg,
        aux: &auxp,
    };
    let scf = kuhf(&ks, Box::new(gdf.jk_builder_with_madelung(0.0)), 2, 1);
    let g = gsys(supercell(&cell, n, 4), &bs, None, Some(&aux_bs));
    let gscf = gamma_unrestricted(
        unfold(&ks.mesh, &scf.density_alpha),
        unfold(&ks.mesh, &scf.density_beta),
    );
    let mut refs: Vec<Array2<f64>> = Vec::new();
    for exx in EXX {
        let fk = kg_uhf(&ks, KGradJk::RsGdf(src), &scf, exx, None);
        let fsc = gamma_grad(&g, &gscf, exx);
        let miss = sc_miss(&fk.grad, &fsc);
        report(&format!("H3 UHF RS-GDF {n:?} {exx:?}"), &fk);
        eprintln!("  H3 UHF RS-GDF {n:?} {exx:?}: supercell miss {miss:.2e} (prototype 4.9e-12 / 6.1e-12)");
        assert!(miss <= SC_BAR_GDF, "{exx:?}: {miss:e}");
        refs.push(fsc);
    }
    for m in FIT_MUTANTS {
        let f = kg_uhf(&ks, KGradJk::RsGdf(src), &scf, ExxDiv::None, Some(m));
        let miss = sc_miss(&f.grad, &refs[0]);
        eprintln!("  H3 UHF RS-GDF mutant {m:?}: miss {miss:.2e}");
        assert!(miss > MUTANT_BAR, "mutant {m:?} not caught ({miss:e})");
    }
}

// ======================================================== input checks

#[test]
fn kpoint_gradient_refuses_a_mismatched_mesh_and_an_unconverged_scf() {
    let bs = pyscf_sto3g_h();
    let ks = ksys(h2_cell_g(), &bs, [1, 1, 3]);
    let eri = kdense(&ks, ANCHOR_PRECISION);
    let mut scf = krhf(&ks, Box::new(eri.jk_builder_with_madelung(0.0)));
    let other = KPointMesh::gamma_centred(&ks.cell, [1, 3, 1]).unwrap();
    let err = kpoint_rhf_gradient(
        &ks.cell,
        &ks.prep,
        &other,
        &hcore_cfg(),
        &ks.hk,
        KGradJk::Dense(&eri),
        &scf,
        ExxDiv::None,
        &gcfg(None),
    )
    .expect_err("a different mesh must be refused");
    eprintln!("mesh mismatch: {err}");
    assert!(err.to_string().contains("k-points"), "{err}");
    scf.converged = false;
    let err = kg_rhf_try(&ks, &eri, &scf).expect_err("unconverged must be refused");
    assert!(err.to_string().contains("did not converge"), "{err}");
}

fn kg_rhf_try(
    ks: &KSys,
    eri: &KDenseAftEri,
    scf: &KScfResult,
) -> Result<KGradient, ferric_core::FerricError> {
    kpoint_rhf_gradient(
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
}
