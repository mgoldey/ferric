//! Shared setup of `pbc_kstress.rs` (per commit) and
//! `validation_pbc_kstress.rs` (weekly): every non-test item of the original
//! file, made `pub`.
#![allow(dead_code, unused_imports)]

pub use crate::common::*;
pub use ferric_core::basis::{BasisSet, Shell};
pub use ferric_core::mol::Molecule;
pub use ferric_integrals::basis_bridge::PreparedBasis;
pub use ferric_pbc::dense_aft::{DenseAftEri, ExxDiv, DEFAULT_DENSE_AFT_MAX_BYTES};
pub use ferric_pbc::dft::PeriodicGridConfig;
pub use ferric_pbc::hcore::kpoint::{periodic_hcore_kpts, PeriodicHcoreK};
pub use ferric_pbc::hcore::{periodic_hcore, PeriodicHcore, PeriodicHcoreConfig};
pub use ferric_pbc::kdense_aft::{KDenseAftConfig, KDenseAftEri};
pub use ferric_pbc::kdft::{solve_krks, KRksConfig};
pub use ferric_pbc::kgrad::KRsGdfGradSource;
pub use ferric_pbc::kpts::{KPointMesh, MeshCentring};
pub use ferric_pbc::kscf::{
    solve_krhf_injected, KPointInjection, KPointJk, KScfConfig, KScfResult,
};
pub use ferric_pbc::kstress::{
    kpoint_hcore_stress_fixed_d, kpoint_rhf_stress, kpoint_uhf_stress, KStress, KStressConfig,
    KStressJk, KStressMutation, FOCK_CONSISTENCY_BAR,
};
pub use ferric_pbc::kuscf::{solve_kuhf_injected, KUScfResult};
pub use ferric_pbc::lattice::Cell;
pub use ferric_pbc::pair_ft::residues::{
    pair_ft_residues_chunked, pair_ft_strain_residues_chunked,
};
pub use ferric_pbc::pair_ft::{pair_ft_strain_chunked, PairFtStrainTerms};
pub use ferric_pbc::rsgdf::kpoint::{KRsGdf, KRsGdfConfig};
pub use ferric_pbc::rsgdf::RsGdfConfig;
pub use ferric_pbc::sr_rotation::SrColumnRotation;
pub use ferric_pbc::stress::{gamma_rhf_stress, gamma_uhf_stress, GammaStress, GammaStressConfig};
pub use ferric_scf::result::{ScfExit, ScfResult, Spin};
pub use ndarray::{Array2, Array3};
pub use num_complex::Complex64;
pub use std::collections::HashMap;

// ============================================================ RS-GDF (unsplit)
//
// The fitted J/K at k: SR 3-centre and metric strain walks with the force's
// per-(l, t) / per-t phase-folded weights, the per-q LR strain on the full
// {G + q} set (complex residue weights, aux-FT strain), the J3 G = 0 overlap
// and volume terms and the strain-only J2 G = 0 volume term.

pub use ferric_pbc::grad::RsGdfGradSource;
pub use ferric_pbc::rsgdf::{RangeSplit, RsGdf};

pub type Mat3 = [[f64; 3]; 3];

pub const OMEGA: f64 = 0.8;

pub const HCORE_PRECISION: f64 = 1e-14;

/// Dense-AFT precision of the fixed-density anchors (see `pbc_kgrad.rs`).
pub const ANCHOR_PRECISION: f64 = 1e-12;

pub const AMPLE: usize = 1 << 31;

pub const FD_H: f64 = 1e-4;

pub const GAMMA_BAR: f64 = 1e-12;

/// Measured h = 1e-4 floor of the hcore FD 6.7e-9 (h² truncation); the smallest
/// mutant miss is 7e-2, so the bar sits between at 5e-8.
pub const HCORE_FD_BAR: f64 = 5e-8;

pub const H2_ATOMS: [[f64; 3]; 2] = [[0.3, 0.2, 0.1], [0.35, 0.12, 1.5]];

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

/// Rotation pinned OFF so these anchors do not depend on the default: the
/// column-rotated stress is anchored in `pbc_sr_rotation_kderiv.rs`.
pub fn hcore_cfg() -> PeriodicHcoreConfig {
    PeriodicHcoreConfig {
        precision: HCORE_PRECISION,
        ..PeriodicHcoreConfig::with_omega(OMEGA)
    }
    .with_sr_column_rotation(SrColumnRotation::Off)
}

pub fn kscf_cfg() -> KScfConfig {
    KScfConfig {
        energy_conv: 1e-13,
        grad_conv: 1e-10,
        max_iter: 400,
        ..Default::default()
    }
}

pub fn scfg(mutation: Option<KStressMutation>) -> KStressConfig {
    KStressConfig {
        budget_bytes: Some(AMPLE),
        mutation,
        ..Default::default()
    }
}

pub fn cell_at(pos: &[[f64; 3]], lattice: [[f64; 3]; 3], mult: usize) -> Cell {
    let mut mol: Molecule = hydrogens(pos);
    mol.multiplicity = mult;
    Cell::new(mol, lattice).expect("cell")
}

pub fn h2_cell_g() -> Cell {
    cell_at(&H2_ATOMS, cubic(4.0), 1)
}

pub fn eps_at(i: usize, j: usize, h: f64) -> Mat3 {
    let mut e = [[0.0; 3]; 3];
    e[i][j] = h;
    e
}

pub fn strained(cell: &Cell, i: usize, j: usize, h: f64) -> Cell {
    cell.strained(&eps_at(i, j, h)).expect("strained cell")
}

pub fn max_mat(a: &Mat3, b: &Mat3) -> f64 {
    let mut m = 0.0_f64;
    for i in 0..3 {
        for j in 0..3 {
            m = m.max((a[i][j] - b[i][j]).abs());
        }
    }
    m
}

// ---------------------------------------------------------------- setups

pub struct KSys {
    pub cell: Cell,
    pub prep: PreparedBasis,
    pub mesh: KPointMesh,
    pub hk: PeriodicHcoreK,
}

pub fn ksys(cell: Cell, bs: &BasisSet, n: [usize; 3]) -> KSys {
    ksys_c(cell, bs, n, MeshCentring::Gamma)
}

pub fn ksys_c(cell: Cell, bs: &BasisSet, n: [usize; 3], centring: MeshCentring) -> KSys {
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

pub fn kdense(ks: &KSys, precision: f64) -> KDenseAftEri {
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

pub fn krhf(ks: &KSys, jk: Box<dyn KPointJk + '_>) -> KScfResult {
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

pub fn real_of(d: &Array2<Complex64>) -> Array2<f64> {
    let im = d.iter().fold(0.0_f64, |m, z| m.max(z.im.abs()));
    assert!(im < 1e-12, "Gamma density has an imaginary part {im:e}");
    d.mapv(|z| z.re)
}

pub fn gamma_restricted(d: Array2<f64>) -> ScfResult {
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
        precision: Default::default(),
    }
}

pub struct GSys {
    pub cell: Cell,
    pub prep: PreparedBasis,
    pub hc: PeriodicHcore,
    pub eri: DenseAftEri,
}

pub fn gsys(cell: Cell, bs: &BasisSet) -> GSys {
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
pub fn k_on(cell_ref: &Cell, cell_s: &Cell, ks: &[[f64; 3]]) -> Vec<[f64; 3]> {
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
pub fn shifted_ks(cell: &Cell, q_frac: [f64; 3], take: usize) -> Vec<[f64; 3]> {
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

pub fn collect_strain(
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

// ============================================================ (1) hcore gate

/// `(1/N_k) Σ_k Re tr[D h(k)] + E_nn` on a (strained) cell, `D(k)` FIXED.
pub fn e_hcore_fixed_d(cell: &Cell, bs: &BasisSet, n: [usize; 3], dk: &[Array2<Complex64>]) -> f64 {
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

pub fn fd_hcore(cell: &Cell, bs: &BasisSet, n: [usize; 3], dk: &[Array2<Complex64>]) -> Mat3 {
    let mut out = [[0.0; 3]; 3];
    for (i, j) in ALL9 {
        let ep = e_hcore_fixed_d(&strained(cell, i, j, FD_H), bs, n, dk);
        let em = e_hcore_fixed_d(&strained(cell, i, j, -FD_H), bs, n, dk);
        out[i][j] = (ep - em) / (2.0 * FD_H);
    }
    out
}

// ====================================================== full-stress fixtures

pub const H3_ATOMS: [[f64; 3]; 3] = [[0.3, 0.2, 0.1], [0.35, 0.12, 1.5], [1.6, 0.9, 0.7]];

pub const H3_A: f64 = 4.5;

pub const EXX: [ExxDiv; 2] = [ExxDiv::None, ExxDiv::Ewald];

pub const MUTANT_BAR: f64 = 1e-4;

pub const IDENTITY_BAR: f64 = 1e-12;

pub const ANTISYM_BAR: f64 = 1e-10;

pub fn h3_cell() -> Cell {
    cell_at(&H3_ATOMS, cubic(H3_A), 2)
}

/// Explicit diag(n) supercell (copy c = (m0, m1, m2), m0 outer) with
/// multiplicity `mult`.
pub fn supercell(cell: &Cell, n: [usize; 3], mult: usize) -> Cell {
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
pub fn unfold(mesh: &KPointMesh, dk: &[Array2<Complex64>]) -> Array2<f64> {
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

pub fn gamma_unrestricted(da: Array2<f64>, db: Array2<f64>) -> ScfResult {
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
        precision: Default::default(),
    }
}

pub fn exx_idx(e: ExxDiv) -> usize {
    match e {
        ExxDiv::None => 0,
        ExxDiv::Ewald => 1,
    }
}

/// The k-RHF SCF of each exxdiv `[none, ewald]` (the stress guards that the
/// SCF's exxdiv is the one it is asked for).
pub fn scf_pair(ks: &KSys, eri: &KDenseAftEri) -> [KScfResult; 2] {
    let vm = ks.mesh.madelung(&ks.cell).unwrap();
    [
        krhf(ks, Box::new(eri.jk_builder_with_madelung(0.0))),
        krhf(ks, Box::new(eri.jk_builder_with_madelung(vm))),
    ]
}

pub fn uscf_pair(ks: &KSys, eri: &KDenseAftEri, na: usize, nb: usize) -> [KUScfResult; 2] {
    let vm = ks.mesh.madelung(&ks.cell).unwrap();
    [
        kuhf(ks, Box::new(eri.jk_builder_with_madelung(0.0)), na, nb),
        kuhf(ks, Box::new(eri.jk_builder_with_madelung(vm)), na, nb),
    ]
}

pub fn kuhf(ks: &KSys, jk: Box<dyn KPointJk + '_>, na: usize, nb: usize) -> KUScfResult {
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

pub fn ks_rhf(
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

pub fn ks_uhf(
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

pub fn gamma_stress(g: &GSys, scf: &ScfResult, exx: ExxDiv) -> GammaStress {
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

pub fn scale3(m: &Mat3, f: f64) -> Mat3 {
    let mut o = *m;
    for r in o.iter_mut() {
        for v in r.iter_mut() {
            *v *= f;
        }
    }
    o
}

pub fn offdiag_miss(a: &Mat3, b: &Mat3) -> f64 {
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

pub fn report(tag: &str, s: &KStress) {
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

pub fn assert_one_point(
    name: &str,
    ksres: &KStress,
    gs: &GammaStress,
    exx: ExxDiv,
    antisym_bar: f64,
) {
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
pub const P_ANTISYM_BAR: f64 = 5e-12;

pub fn one_point_rhf((name, cell, bs, asym): (&str, Cell, BasisSet, f64)) {
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

// ====================================================== (3) supercell anchor

pub const SC_BAR: f64 = 1e-10;

pub struct ScRef {
    /// Gamma supercell `dE/dε / N`, per exxdiv.
    pub de: [Mat3; 2],
}

/// Supercell anchor of a closed-shell system: `dE/dε_k = dE/dε_sc / N`, σ
/// equal. Returns the references for the mutants.
pub fn rhf_supercell_anchor(
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

pub const ZERO: Mat3 = [[0.0; 3]; 3];

/// The k-specific defects (phases, q, 1/N_k, mesh Madelung): the new risk of
/// this module, run in CI.
pub const K_MUTANTS: [KStressMutation; 4] = [
    KStressMutation::NoPhase,
    KStressMutation::ConjD,
    KStressMutation::WrongQ,
    KStressMutation::NoInvNk,
];

pub const K_EWALD_MUTANTS: [KStressMutation; 2] = [
    KStressMutation::GammaMadelung,
    KStressMutation::GammaMadelungStrain,
];

/// The Gamma-style term defects. Measured against the 1x1x1 Gamma anchor
/// (`term_mutants_miss_the_one_point_gamma_anchor`) and the hcore gate; also
/// swept against the supercell in the `#[ignore]`d `h2_1x1x3_term_mutants_*`.
pub const TERM_MUTANTS: [KStressMutation; 9] = [
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

pub const TERM_EWALD_MUTANTS: [KStressMutation; 2] =
    [KStressMutation::NoMadelung, KStressMutation::MadelungS];

/// Mutants that only change the diagonal by construction (volume terms).
pub const DIAGONAL_ONLY: [KStressMutation; 3] = [
    KStressMutation::NoVolume,
    KStressMutation::NoC0Volume,
    KStressMutation::NoEwaldBg,
];

/// Every mutant must miss the supercell anchor by > MUTANT_BAR on at least one
/// component (both exxdiv; the Madelung ones under ewald only and BLIND under
/// none); the diagonal-only ones must leave the off-diagonal at the floor.
pub fn assert_mutants_fail(
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
pub fn assert_mutants_fail_floor(
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

// ====================================================== (4) FD of own k energy

/// k-RHF energies `[none, ewald]` per cell of a (strained) cell, dense AFT at
/// `ANCHOR_PRECISION`, mesh rebuilt from the same integer numerators.
pub fn k_energies(cell: &Cell, bs: &BasisSet, n: [usize; 3]) -> [f64; 2] {
    let ks = ksys(cell.clone(), bs, n);
    let eri = kdense(&ks, ANCHOR_PRECISION);
    let vm = ks.mesh.madelung(&ks.cell).unwrap();
    [
        krhf(&ks, Box::new(eri.jk_builder_with_madelung(0.0))).energy,
        krhf(&ks, Box::new(eri.jk_builder_with_madelung(vm))).energy,
    ]
}

/// Measured floor 1.0e-8 (h = 1e-4: the h² f‴/6 truncation of the central FD,
/// identical on both exxdiv); the smallest mutant miss is 4.9e-2.
pub const K_FD_BAR: f64 = 5e-8;

/// FD `dE/dε` of `energies` (`[none, ewald]`) on `cell`, all nine components, step `h`.
pub fn fd_all9(cell: &Cell, h: f64, energies: &dyn Fn(&Cell) -> [f64; 2]) -> [Mat3; 2] {
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
pub fn k_energies_c(cell: &Cell, bs: &BasisSet, n: [usize; 3], centring: MeshCentring) -> [f64; 2] {
    let ks = ksys_c(cell.clone(), bs, n, centring);
    let eri = kdense(&ks, ANCHOR_PRECISION);
    let vm = ks.mesh.madelung(&ks.cell).unwrap();
    [
        krhf(&ks, Box::new(eri.jk_builder_with_madelung(0.0))).energy,
        krhf(&ks, Box::new(eri.jk_builder_with_madelung(vm))).energy,
    ]
}

/// k-UHF energy of the H3 doublet (2, 1) on a (strained) cell, exxdiv none and ewald.
pub fn k_energies_uhf(cell: &Cell, bs: &BasisSet, n: [usize; 3]) -> [f64; 2] {
    let ks = ksys(cell.clone(), bs, n);
    let eri = kdense(&ks, ANCHOR_PRECISION);
    let vm = ks.mesh.madelung(&ks.cell).unwrap();
    [
        kuhf(&ks, Box::new(eri.jk_builder_with_madelung(0.0)), 2, 1).energy,
        kuhf(&ks, Box::new(eri.jk_builder_with_madelung(vm)), 2, 1).energy,
    ]
}

/// This test's own measured floor: 2.7e-8 at h = 1e-4, 1.2e-7 at 2h (ratio
/// 4.2, i.e. h² truncation), 2e-9 after Richardson. The bar is ~4x the floor
/// and 6 orders below the smallest mutant miss (CrossSpinK 7.9e-1,
/// SwapSpinFock 5.8e-1).
pub const K_FD_BAR_UHF: f64 = 1e-7;

// ====================================================== input consistency

pub fn rhf_stress_result(
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

/// Measured s+p floor: 9.8e-9 at the default nucleus exponent, 9.4e-9 / 3.5e-9
/// at 1e8 (pre-#434 numbers; the h = 1e-4 central-FD truncation dominates
/// exactly as for s shells); smallest mutant miss 4.6e-2. Bar 5e-8 = the
/// s-shell bar.
pub const K_FD_BAR_P: f64 = 5e-8;

/// A deliberately COARSE aux on H: s(1.9), s(0.5) (2 functions per H). Used
/// by the Fock-guard test as "a fit on another aux basis than the SCF's" (its
/// energy differs from the ET-sp fit at the 1e-3 level).
pub fn coarse_aux() -> BasisSet {
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

/// Rotation pinned OFF (see `hcore_cfg`).
pub fn gdf_cfg() -> RsGdfConfig {
    RsGdfConfig {
        omega: 1.0,
        exxdiv: ExxDiv::None,
        budget_bytes: Some(AMPLE),
        sr_column_rotation: SrColumnRotation::Off,
        ..Default::default()
    }
}

pub fn kgdf_cfg() -> KRsGdfConfig {
    KRsGdfConfig {
        gdf: gdf_cfg(),
        mutation: None,
    }
}

/// A k-point RS-GDF system: the build, its config and aux basis.
pub struct FitSys {
    pub cfg: KRsGdfConfig,
    pub aux: PreparedBasis,
    pub gdf: KRsGdf,
}

pub fn fitsys(ks: &KSys) -> FitSys {
    fitsys_cfg(ks, kgdf_cfg())
}

pub fn fitsys_cfg(ks: &KSys, cfg: KRsGdfConfig) -> FitSys {
    fitsys_with(ks, cfg, &et_sp_aux())
}

pub fn fitsys_with(ks: &KSys, cfg: KRsGdfConfig, aux_bs: &BasisSet) -> FitSys {
    let aux = PreparedBasis::new(ks.cell.mol(), aux_bs).unwrap();
    let gdf = KRsGdf::build(&ks.cell, &ks.prep, &aux, &ks.mesh, &ks.hk.s, &cfg).expect("KRsGdf");
    FitSys { cfg, aux, gdf }
}

impl FitSys {
    pub fn src(&self) -> KRsGdfGradSource<'_> {
        KRsGdfGradSource {
            gdf: &self.gdf,
            cfg: &self.cfg,
            aux: &self.aux,
        }
    }

    /// The SCF's J/K at the exxdiv.
    pub fn jk(&self, ks: &KSys, exx: ExxDiv) -> Box<dyn KPointJk + '_> {
        let vm = match exx {
            ExxDiv::None => 0.0,
            ExxDiv::Ewald => ks.mesh.madelung(&ks.cell).unwrap(),
        };
        Box::new(self.gdf.jk_builder_with_madelung(vm))
    }
}

pub fn fit_scfs(ks: &KSys, fs: &FitSys) -> [KScfResult; 2] {
    [
        krhf(ks, fs.jk(ks, ExxDiv::None)),
        krhf(ks, fs.jk(ks, ExxDiv::Ewald)),
    ]
}

pub fn ks_rhf_fit(
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
pub struct GFit {
    pub cell: Cell,
    pub prep: PreparedBasis,
    pub hc: PeriodicHcore,
    pub aux: PreparedBasis,
    pub gdf: RsGdf,
}

pub fn gfit(cell: Cell, bs: &BasisSet) -> GFit {
    gfit_aux(cell, bs, &et_sp_aux(), &gdf_cfg())
}

pub fn gfit_aux(cell: Cell, bs: &BasisSet, aux_bs: &BasisSet, cfg: &RsGdfConfig) -> GFit {
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

pub fn gamma_fit_stress(g: &GFit, scf: &ScfResult, exx: ExxDiv) -> GammaStress {
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
pub const GAMMA_FIT_BAR: f64 = 1e-11;

pub fn fit_parts(p: &ferric_pbc::kstress::KStressParts) -> [(&'static str, Mat3); 7] {
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

pub fn gamma_fit_parts(p: &ferric_pbc::stress::GammaStressParts) -> [Mat3; 7] {
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

// ----------------------------------------------- supercell anchor (RS-GDF)

/// Bar of the fitted supercell anchor (`dE/dε` per cell). Measured 2.1e-12
/// (none) / 2.3e-12 (ewald) on H2 1x1x3; the smallest sharp mutant miss is
/// 1.3e-2.
pub const SC_FIT_BAR: f64 = 1e-10;

/// Fit-specific term defects (all production-wired in `kstress`).
pub const FIT_MUTANTS: [KStressMutation; 5] = [
    KStressMutation::FitAuxPhase,
    KStressMutation::NoMetric,
    KStressMutation::NoAuxFt3,
    KStressMutation::NoJ2G0Vol,
    KStressMutation::NoJ3G0Vol,
];

/// Mutants that change only the diagonal by construction.
pub const FIT_DIAGONAL_ONLY: [KStressMutation; 2] =
    [KStressMutation::NoJ2G0Vol, KStressMutation::NoJ3G0Vol];

pub struct FitAnchor {
    pub ks: KSys,
    pub fs: FitSys,
    pub scfs: [KScfResult; 2],
    /// Gamma supercell `dE/dε / N` per exxdiv.
    pub refs: [Mat3; 2],
    /// Its seven fitted parts (`fit_parts` order) per exxdiv, `/ N`.
    pub ref_parts: [[Mat3; 7]; 2],
}

/// Miss of a mutated run against the supercell: all nine components of the
/// total, the off-diagonal of the total, and the worst of the seven FITTED
/// PARTS individually (a part-level defect whose total is hidden by aux
/// stationarity is a sharp miss here).
pub struct Miss {
    pub total: f64,
    pub off: f64,
    pub part: f64,
}

/// Supercell anchor of an RS-GDF k-RHF system (H2 or any closed-shell cell):
/// `dE/dε_k = dE/dε_sc/N`, both exxdiv, term by term; returns the setup and
/// the references for the mutants.
pub fn fit_supercell_anchor(name: &str, cell: Cell, bs: &BasisSet, n: [usize; 3]) -> FitAnchor {
    fit_supercell_anchor_aux(name, cell, bs, &et_sp_aux(), n)
}

pub fn fit_supercell_anchor_aux(
    name: &str,
    cell: Cell,
    bs: &BasisSet,
    aux_bs: &BasisSet,
    n: [usize; 3],
) -> FitAnchor {
    fit_supercell_anchor_cfg(name, cell, bs, aux_bs, &gdf_cfg(), n)
}

pub fn fit_supercell_anchor_cfg(
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
pub fn fit_supercell_anchor_bars(
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
    pub fn mutant(&self, exx: ExxDiv, m: KStressMutation) -> Miss {
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

// ------------------------------------------------ FD of the RS-GDF k energy

/// k-RHF RS-GDF energies `[none, ewald]` per cell of a (strained) cell, mesh
/// rebuilt from the same integer numerators, plus the per-q kept metric
/// counts (the FD is only smooth while the lindep cut keeps the same count).
pub fn k_fit_energies(cell: &Cell, bs: &BasisSet, n: [usize; 3]) -> ([f64; 2], Vec<usize>) {
    let ks = ksys(cell.clone(), bs, n);
    let fs = fitsys(&ks);
    let kept = fs.gdf.stats().per_q.iter().map(|q| q.naux_kept).collect();
    let e = [
        krhf(&ks, fs.jk(&ks, ExxDiv::None)).energy,
        krhf(&ks, fs.jk(&ks, ExxDiv::Ewald)).energy,
    ];
    (e, kept)
}

/// Measured floor 1.02e-8 (h = 1e-4: the h² f‴/6 truncation of the central
/// FD; 2h gives 4.31e-8, ratio 4.21, identical for both exxdiv); the smallest
/// sharp mutant miss is 1.3e-2 and the blind-on-supercell `NoAuxFt` misses
/// by 5.0e-5.
pub const K_FD_BAR_FIT: f64 = 5e-8;

// ----------------------------------------------------------- RS-GDF UHF

pub fn ks_uhf_fit(
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

pub fn fit_uscfs(ks: &KSys, fs: &FitSys, na: usize, nb: usize) -> [KUScfResult; 2] {
    [
        kuhf(ks, fs.jk(ks, ExxDiv::None), na, nb),
        kuhf(ks, fs.jk(ks, ExxDiv::Ewald), na, nb),
    ]
}

// ------------------------------------- shifted (MP) mesh and triclinic cell

/// k-RHF RS-GDF energies `[none, ewald]` on `(cell, bs, n, centring)` with the
/// per-q kept metric counts.
pub fn k_fit_energies_c(
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
pub fn fit_fd_anchor(
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

/// Lindep of the near-duplicate test: above the small eigenvalue of the
/// s(0.75)/s(0.76) pair, below the rest of the spectrum (asserted).
pub const NEAR_DUP_LINDEP: f64 = 1e-3;

// -------------------------------------------------- aux l >= 2, omega, rotation

/// ET-sp aux plus one PURE d shell (exponent 0.8): 15 functions per H, the
/// l = 2 aux case of `aux_ft_strain_shells` and of the SR 3-centre / metric
/// derivative blocks (the production path at k; the unit test covers the FT
/// alone).
pub fn et_spd_aux() -> BasisSet {
    let mut aux = et_sp_aux();
    aux.shells.get_mut(&1).unwrap().push(Shell {
        l: 2,
        pure: true,
        exponents: vec![0.8],
        coefficients: vec![1.0],
    });
    aux
}

/// Measured max abs difference of dE/dε between the omega 1.0 and 1.3 fits: 2.6e-12 (H2 1x1x3).
pub const OMEGA_STRESS_BAR: f64 = 1e-10;

/// Measured |Δ dE/dε| rotated pipeline vs unrotated pipeline (hcore precision
/// 1e-14, default fit precision): 3.7e-13 (energies differ by 7.9e-14, so the
/// rotation is active and the stress walks are rotated too).
pub const AUTO_ROT_BAR: f64 = 1e-11;

// ------------------------------------------------------------- RS-GDF UHF FD

/// k-UHF RS-GDF energy (exxdiv none) of the H3 doublet (2, 1) on a strained
/// cell, with the per-q kept counts.
pub fn k_fit_energy_uhf(cell: &Cell, bs: &BasisSet, n: [usize; 3]) -> (f64, Vec<usize>) {
    let ks = ksys(cell.clone(), bs, n);
    let fs = fitsys(&ks);
    let kept = fs.gdf.stats().per_q.iter().map(|q| q.naux_kept).collect();
    (kuhf(&ks, fs.jk(&ks, ExxDiv::None), 2, 1).energy, kept)
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
pub fn split_gdf_cfg(lambda: f64) -> RsGdfConfig {
    RsGdfConfig {
        range_split: Some(RangeSplit::new(lambda)),
        ..gdf_cfg()
    }
}

pub fn split_kcfg(lambda: f64) -> KRsGdfConfig {
    KRsGdfConfig {
        gdf: split_gdf_cfg(lambda),
        mutation: None,
    }
}

/// Panics unless the split path ran, primitives moved on both sides and the
/// SR walks shrank against the unsplit build (the artifact guard).
pub fn assert_moved(tag: &str, gdf: &KRsGdf, unsplit: &KRsGdf) {
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
pub const SPLIT_MUTANTS: [KStressMutation; 6] = [
    KStressMutation::SplitNoSrKernelStrain,
    KStressMutation::SplitNoSmoothPair,
    KStressMutation::SplitFullG0,
    KStressMutation::SplitNoSmoothOverlap,
    KStressMutation::SplitFlipSrSlope,
    KStressMutation::SplitNoSmoothVolume,
];

/// Measured on H2 1x1x3 (lambda 1, omega 1, each side at its own density):
/// 2.6e-12 (none) / 5.9e-13 (ewald); the split-vs-unsplit energy differs by
/// 4e-15. The bar is 40x above the floor and 7+ orders below the smallest
/// split mutant (7.8e-4).
pub const SPLIT_UNSPLIT_BAR: f64 = 1e-10;

// ------------------------------------------------ FD of the split k energy

/// Split k-RHF energies `[none, ewald]` on `(cell, bs, n, centring)` with the
/// per-q kept metric counts.
pub fn k_split_energies(
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
pub fn split_fd_anchor(
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

// ----------------------------------------------- d-aux part floor (split)

/// The ET-sp aux plus one pure-d shell at exponent `a` (lambda 1, omega 1
/// moves the d shell iff `a <= 1`).
pub fn et_sp_d_aux(a: f64) -> BasisSet {
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
pub fn d_aux_floor(aux_bs: &BasisSet, gdf: &RsGdfConfig) -> (f64, f64, f64) {
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

// ------------------------------------- split vs unsplit: other omega/lambda

/// `(orbital smooth, aux smooth)` primitive counts of a split build.
pub fn moved_counts(gdf: &KRsGdf) -> (usize, usize) {
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
pub fn split_vs_unsplit(
    aux_bs: &BasisSet,
    sp: &RsGdfConfig,
    un: &RsGdfConfig,
) -> ([f64; 2], FitSys) {
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

/// Worst fitted-part and total `|k − Gamma|` of the 1x1x1 split anchor (none
/// exxdiv) at `gdf`, with the number of metric eigenvalues dropped by `lindep`
/// and the smallest metric eigenvalue.
pub fn one_point_part_floor(aux_bs: &BasisSet, gdf: &RsGdfConfig) -> (f64, f64, usize, f64) {
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

pub const OS_A: f64 = 4.0;

pub const H4_ATOMS: [[f64; 3]; 4] = [
    [0.3, 0.2, 0.1],
    [0.35, 0.12, 1.5],
    [1.6, 0.9, 0.7],
    [1.2, 2.1, 2.3],
];

pub const H4_A: f64 = 4.5;

/// An open-shell test system: cell (carrying the multiplicity), the spin
/// counts and a short name.
pub struct OpenShell {
    pub name: &'static str,
    pub cell: Cell,
    pub na: usize,
    pub nb: usize,
}

pub fn s1_h2_triplet() -> OpenShell {
    OpenShell {
        name: "H2 triplet s+p",
        cell: cell_at(&H2_ATOMS, cubic(OS_A), 3),
        na: 2,
        nb: 0,
    }
}

pub fn s2_h4_triplet() -> OpenShell {
    OpenShell {
        name: "H4 triplet s+p",
        cell: cell_at(&H4_ATOMS, cubic(H4_A), 3),
        na: 3,
        nb: 1,
    }
}

/// k-UHF energies `[none, ewald]` of `(na, nb)` on a (strained) cell.
pub fn k_energies_uhf_nanb(
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
pub fn assert_spin_polarised(sys: &OpenShell, scf: &KUScfResult) {
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
pub fn open_shell_p_one_point(sys: OpenShell) {
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

/// Bar of the p-shell supercell anchor at the DEFAULT model-nucleus exponent
/// (the unrotated ECP file's bar for the same floor). Before #434 was fixed
/// the SR derivative's translation noise put the miss at 5.9e-8 at the default
/// exponent (pinned at 1e7 as a workaround, 1.7e-10); with the nucleus-frame
/// evaluation it is 4.6e-13 at the default (both exxdiv), the smallest mutant
/// misses by > 1e-4 (`MUTANT_BAR`), and the bar is 11x the measurement.
/// Also the blind / diagonal-only floor of the mutant sweep.
pub const OS_SC_BAR: f64 = 5e-12;
