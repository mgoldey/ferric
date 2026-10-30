//! Shared setup of `pbc_kgrad.rs` (per commit) and `validation_pbc_kgrad.rs`
//! (weekly): every non-test item of the original file, made `pub`.
#![allow(dead_code, unused_imports)]

pub use crate::common::*;
pub use ferric_core::basis::{BasisSet, Shell};
pub use ferric_core::mol::Molecule;
pub use ferric_integrals::basis_bridge::PreparedBasis;
pub use ferric_pbc::dense_aft::{
    DenseAftEri, ExxDiv, DEFAULT_DENSE_AFT_MAX_BYTES, DEFAULT_DENSE_AFT_PRECISION,
};
pub use ferric_pbc::ewald::madelung_constant;
pub use ferric_pbc::grad::{
    gamma_rhf_gradient_rsgdf, gamma_rhf_gradient_with, gamma_uhf_gradient_rsgdf,
    gamma_uhf_gradient_with, GammaGradConfig, RsGdfGradSource,
};
pub use ferric_pbc::hcore::kpoint::{periodic_hcore_kpts, PeriodicHcoreK};
pub use ferric_pbc::hcore::{periodic_hcore, PeriodicHcore, PeriodicHcoreConfig};
pub use ferric_pbc::kdense_aft::{KDenseAftConfig, KDenseAftEri};
pub use ferric_pbc::kgrad::{
    kpoint_rhf_gradient, kpoint_uhf_gradient, KGradConfig, KGradJk, KGradMutation, KGradient,
    KRsGdfGradSource,
};
pub use ferric_pbc::kpts::KPointMesh;
pub use ferric_pbc::kscf::{
    solve_krhf_injected, KPointInjection, KPointJk, KScfConfig, KScfResult,
};
pub use ferric_pbc::kuscf::{solve_kuhf_injected, KUScfResult};
pub use ferric_pbc::lattice::Cell;
pub use ferric_pbc::rsgdf::kpoint::{KRsGdf, KRsGdfConfig};
pub use ferric_pbc::rsgdf::{RsGdf, RsGdfConfig};
pub use ferric_scf::result::{ScfExit, ScfResult, Spin};
pub use ndarray::Array2;
pub use num_complex::Complex64;
pub use std::collections::HashMap;

/// ω of the nuclear-attraction split (as `pbc_grad.rs`, `pbc_krhf.rs`).
pub const OMEGA: f64 = 0.8;

pub const HCORE_PRECISION: f64 = 1e-14;

/// Dense-AFT precision of the fixed-density anchors (the K set and the pair
/// screen are the same on both sides at any precision; the loose value
/// keeps the supercell tensors cheap).
/// Pair-FT screening for the dense anchors. The k-point and Gamma dense
/// derivatives screen pair FTs differently, so their difference is a
/// truncation term proportional to this threshold (measured 2026-09-25, H3 UHF
/// 1x1x1 vs Gamma: 1.05e-9 / 1.41e-11 / 1.42e-13 at 1e-8 / 1e-10 / 1e-12;
/// RS-GDF, which shares one screen, agrees to 3e-14 at any setting). 1e-12
/// makes the anchors test the formula, not the screen.
pub const ANCHOR_PRECISION: f64 = 1e-12;

pub const AMPLE: usize = 1 << 31;

pub const FD_H: f64 = 1e-4;

pub const FD_BAR: f64 = 1e-7;

pub const GAMMA_BAR: f64 = 1e-12;

pub const GAMMA_BAR_GDF: f64 = 1e-12;

pub const SC_BAR: f64 = 1e-12;

pub const SC_BAR_GDF: f64 = 1e-11;

pub const NET_FORCE_BAR: f64 = 1e-10;

pub const EWALD_NONE_BAR: f64 = 1e-11;

pub const IDENTITY_BAR: f64 = 1e-12;

pub const MUTANT_BAR: f64 = 1e-4;

/// `run_kgrad_anchor.py` H2 (Bohr), a = 4, STO-3G: off-axis so every
/// Cartesian component is non-trivial.
pub const H2_ATOMS: [[f64; 3]; 2] = [[0.3, 0.2, 0.1], [0.35, 0.12, 1.5]];

pub const H2_COMPS: [(usize, usize); 3] = [(0, 0), (0, 2), (1, 1)];

/// H3 (Bohr), a = 4.5, STO-3G, doublet (2, 1) per cell.
pub const H3_ATOMS: [[f64; 3]; 3] = [[0.3, 0.2, 0.1], [0.35, 0.12, 1.5], [1.6, 0.9, 0.7]];

pub const H3_A: f64 = 4.5;

/// tri2: TRI_A with the first two TRI_MOVED hydrogens, s+p basis (nao 8).
pub const TRI2_ATOMS: [[f64; 3]; 2] = [[0.13, 0.25, 0.31], [0.02, 0.27, 1.66]];

pub const MESH_MUTANTS: [KGradMutation; 4] = [
    KGradMutation::NoPhase,
    KGradMutation::ConjD,
    KGradMutation::WrongQ,
    KGradMutation::NoInvNk,
];

pub const FIT_MUTANTS: [KGradMutation; 3] = [
    KGradMutation::FitAuxPhase,
    KGradMutation::FitNoMetric,
    KGradMutation::FitNoG0,
];

pub const EXX: [ExxDiv; 2] = [ExxDiv::None, ExxDiv::Ewald];

/// Hcore config at the file's `OMEGA` with `precision = HCORE_PRECISION`; the
/// other fields are the defaults of `with_omega`.
pub fn hcore_cfg() -> PeriodicHcoreConfig {
    PeriodicHcoreConfig {
        precision: HCORE_PRECISION,
        ..PeriodicHcoreConfig::with_omega(OMEGA)
    }
}

/// k-point SCF config with `energy_conv = 1e-13`, `grad_conv = 1e-10` and
/// `max_iter = 400`; the other fields are the defaults.
pub fn kscf_cfg() -> KScfConfig {
    KScfConfig {
        energy_conv: 1e-13,
        grad_conv: 1e-10,
        max_iter: 400,
        ..Default::default()
    }
}

pub fn kgdf_cfg() -> KRsGdfConfig {
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

/// Default k-point gradient config with budget `AMPLE` and the test-only
/// `mutation`.
pub fn gcfg(mutation: Option<KGradMutation>) -> KGradConfig {
    KGradConfig {
        budget_bytes: Some(AMPLE),
        mutation,
        ..Default::default()
    }
}

/// Default Gamma gradient config with budget `AMPLE`.
pub fn gamma_gcfg() -> GammaGradConfig {
    GammaGradConfig {
        budget_bytes: Some(AMPLE),
        ..Default::default()
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

/// A neutral hydrogen cell with atoms at `pos` (Bohr), lattice rows `lattice`
/// (Bohr) and spin multiplicity `mult`.
pub fn cell_at(pos: &[[f64; 3]], lattice: [[f64; 3]; 3], mult: usize) -> Cell {
    let mut mol: Molecule = hydrogens(pos);
    mol.multiplicity = mult;
    Cell::new(mol, lattice).expect("cell")
}

/// `pos` (Bohr) with Cartesian component `x` of atom `a` displaced by `h` Bohr.
pub fn moved(pos: &[[f64; 3]], a: usize, x: usize, h: f64) -> Vec<[f64; 3]> {
    let mut p = pos.to_vec();
    p[a][x] += h;
    p
}

/// The H2 singlet cell: `H2_ATOMS` in a cubic lattice of edge 4 Bohr.
pub fn h2_cell_g() -> Cell {
    cell_at(&H2_ATOMS, cubic(4.0), 1)
}

/// The H3 doublet cell: `H3_ATOMS` in a cubic lattice of edge `H3_A` Bohr,
/// multiplicity 2.
pub fn h3_cell() -> Cell {
    cell_at(&H3_ATOMS, cubic(H3_A), 2)
}

pub fn tri2_cell() -> Cell {
    cell_at(&TRI2_ATOMS, TRI_A, 1)
}

/// Explicit diag(n) supercell (copy c = (m0, m1, m2), m0 outer; atoms of
/// copy c at `R + Σ m_i a_i`) with multiplicity `mult`.
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
    }
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
    }
}

pub fn real_of(d: &Array2<Complex64>) -> Array2<f64> {
    let im = d.iter().fold(0.0_f64, |m, z| m.max(z.im.abs()));
    assert!(im < 1e-12, "Gamma density has an imaginary part {im:e}");
    d.mapv(|z| z.re)
}

/// Largest element-wise `|a − b|`; panics if the shapes differ.
pub fn max_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    assert_eq!(a.dim(), b.dim());
    a.iter()
        .zip(b.iter())
        .fold(0.0_f64, |m, (x, y)| m.max((x - y).abs()))
}

/// `max_{c, A, x} |F_k[A, x] − F_sc[c·natoms + A, x]|` (every copy).
pub fn sc_miss(gk: &Array2<f64>, gsc: &Array2<f64>) -> f64 {
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
pub fn copy_sum(gsc: &Array2<f64>, na: usize) -> Array2<f64> {
    let nc = gsc.nrows() / na;
    Array2::from_shape_fn((na, 3), |(a, x)| {
        (0..nc).map(|c| gsc[(c * na + a, x)]).sum()
    })
}

// ---------------------------------------------------------------- setups

pub struct KSys {
    pub cell: Cell,
    pub prep: PreparedBasis,
    pub mesh: KPointMesh,
    pub hk: PeriodicHcoreK,
}

pub fn ksys(cell: Cell, bs: &BasisSet, n: [usize; 3]) -> KSys {
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

pub fn kgdf(ks: &KSys, aux: &PreparedBasis) -> KRsGdf {
    KRsGdf::build(&ks.cell, &ks.prep, aux, &ks.mesh, &ks.hk.s, &kgdf_cfg()).expect("KRsGdf")
}

pub fn inj<'a>(ks: &KSys, jk: Box<dyn KPointJk + 'a>) -> KPointInjection<'a> {
    KPointInjection {
        s: ks.hk.s.clone(),
        h: ks.hk.h.clone(),
        vnn: ks.hk.enn,
        jk,
    }
}

pub fn krhf(ks: &KSys, jk: Box<dyn KPointJk + '_>) -> KScfResult {
    let r = solve_krhf_injected(&ks.cell, &ks.mesh, &kscf_cfg(), inj(ks, jk)).expect("k-RHF");
    assert!(r.converged, "k-RHF not converged ({} it)", r.iterations);
    r
}

pub fn kuhf(ks: &KSys, jk: Box<dyn KPointJk + '_>, na: usize, nb: usize) -> KUScfResult {
    let r =
        solve_kuhf_injected(&ks.cell, &ks.mesh, &kscf_cfg(), inj(ks, jk), na, nb).expect("k-UHF");
    assert!(r.converged, "k-UHF not converged ({} it)", r.iterations);
    r
}

pub fn kg_rhf(
    ks: &KSys,
    jk: KGradJk<'_>,
    scf: &KScfResult,
    exx: ExxDiv,
    m: Option<KGradMutation>,
) -> KGradient {
    kg_rhf_nuc(ks, jk, scf, exx, m, None)
}

/// `kg_rhf` with an explicit gradient nucleus exponent (`None` = default).
pub fn kg_rhf_nuc(
    ks: &KSys,
    jk: KGradJk<'_>,
    scf: &KScfResult,
    exx: ExxDiv,
    m: Option<KGradMutation>,
    nuc: Option<f64>,
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
        &KGradConfig {
            nucleus_exponent: nuc,
            ..gcfg(m)
        },
    )
    .expect("kpoint_rhf_gradient")
}

pub fn kg_uhf(
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
pub struct GSys {
    pub cell: Cell,
    pub prep: PreparedBasis,
    pub hc: PeriodicHcore,
    pub eri: Option<DenseAftEri>,
    pub gdf: Option<(PreparedBasis, RsGdf)>,
}

pub fn gsys(cell: Cell, bs: &BasisSet, precision: Option<f64>, aux: Option<&BasisSet>) -> GSys {
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

pub fn gamma_grad(g: &GSys, scf: &ScfResult, exx: ExxDiv) -> Array2<f64> {
    gamma_grad_nuc(g, scf, exx, None)
}

/// `gamma_grad` with an explicit gradient nucleus exponent (`None` = default).
pub fn gamma_grad_nuc(g: &GSys, scf: &ScfResult, exx: ExxDiv, nuc: Option<f64>) -> Array2<f64> {
    let restricted = scf.spin == Spin::Restricted;
    let gc = GammaGradConfig {
        nucleus_exponent: nuc,
        ..gamma_gcfg()
    };
    match (&g.eri, &g.gdf) {
        (Some(eri), _) => {
            if restricted {
                gamma_rhf_gradient_with(&g.cell, &g.prep, &hcore_cfg(), &g.hc, eri, scf, exx, &gc)
            } else {
                gamma_uhf_gradient_with(&g.cell, &g.prep, &hcore_cfg(), &g.hc, eri, scf, exx, &gc)
            }
        }
        (None, Some((auxp, gdf))) => {
            let src = RsGdfGradSource {
                gdf,
                aux: auxp,
                aux_jac: None,
            };
            if restricted {
                gamma_rhf_gradient_rsgdf(&g.cell, &g.prep, &hcore_cfg(), &g.hc, &src, scf, exx, &gc)
            } else {
                gamma_uhf_gradient_rsgdf(&g.cell, &g.prep, &hcore_cfg(), &g.hc, &src, scf, exx, &gc)
            }
        }
        _ => panic!("GSys without J/K"),
    }
    .expect("gamma gradient")
    .grad
}

pub fn report(tag: &str, g: &KGradient) {
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

// ============================================ (2) + (4) + (5) supercell anchor

/// Dense supercell anchor for an RHF system: k force vs every copy, both
/// exxdiv, at the unfolded k density; ΣF; ewald ≡ none. Returns the setup,
/// SCF and the supercell reference forces `[none, ewald]` for the mutants.
/// `nuc` is the gradient nucleus exponent used on BOTH sides (`None` =
/// default); `bar` is the supercell-miss bar (`SC_BAR`; was a looser one for
/// the p-shell case). Pass the same `nuc`/`bar` to `assert_rhf_mutants_fail`.
pub fn rhf_supercell_anchor(
    name: &str,
    cell: Cell,
    bs: &BasisSet,
    n: [usize; 3],
    nuc: Option<f64>,
    bar: f64,
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
        let fk = kg_rhf_nuc(&ks, KGradJk::Dense(&eri), &scf, exx, None, nuc);
        let fsc = gamma_grad_nuc(&g, &gscf, exx, nuc);
        let miss = sc_miss(&fk.grad, &fsc);
        let nsum = max_diff(&copy_sum(&fsc, fk.grad.nrows()), &fk.grad);
        report(&format!("{name} {n:?} {exx:?}"), &fk);
        eprintln!(
            "  {name} {n:?} {exx:?}: max_c |F_k − F_sc(copy c)| {miss:.2e} (N× sum − F_k {nsum:.2e})"
        );
        assert!(
            miss <= bar,
            "{name} {exx:?}: supercell miss {miss:e} (bar {bar:e})"
        );
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
/// exxdiv), GammaMadelung under ewald only (and be blind under none, i.e.
/// within the case's supercell `bar`). `nuc` MUST be the exponent the
/// `refs` were computed with in `rhf_supercell_anchor`.
pub fn assert_rhf_mutants_fail(
    name: &str,
    ks: &KSys,
    eri: &KDenseAftEri,
    scf: &KScfResult,
    refs: &[Array2<f64>; 2],
    nuc: Option<f64>,
    bar: f64,
) {
    for m in MESH_MUTANTS {
        for (i, exx) in EXX.into_iter().enumerate() {
            let f = kg_rhf_nuc(ks, KGradJk::Dense(eri), scf, exx, Some(m), nuc);
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
    let fe = kg_rhf_nuc(ks, KGradJk::Dense(eri), scf, ExxDiv::Ewald, Some(m), nuc);
    let fn_ = kg_rhf_nuc(ks, KGradJk::Dense(eri), scf, ExxDiv::None, Some(m), nuc);
    let (me, mn) = (sc_miss(&fe.grad, &refs[1]), sc_miss(&fn_.grad, &refs[0]));
    eprintln!("  {name} mutant {m:?}: ewald miss {me:.2e}, none miss {mn:.2e} (blind)");
    assert!(
        me > MUTANT_BAR,
        "{name}: GammaMadelung (ewald) not caught ({me:e})"
    );
    assert!(
        mn <= bar,
        "{name}: GammaMadelung must be blind under none ({mn:e})"
    );
}

// ================================================== (3) FD of own k energy

pub fn fd_rows<F>(
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
pub fn h2_dense_energies(pos: &[[f64; 3]]) -> [f64; 2] {
    let ks = ksys(cell_at(pos, cubic(4.0), 1), &pyscf_sto3g_h(), [1, 1, 3]);
    let eri = kdense(&ks, DEFAULT_DENSE_AFT_PRECISION);
    let vm = ks.mesh.madelung(&ks.cell).unwrap();
    [
        krhf(&ks, Box::new(eri.jk_builder_with_madelung(0.0))).energy,
        krhf(&ks, Box::new(eri.jk_builder_with_madelung(vm))).energy,
    ]
}

/// RS-GDF k-RHF energies `[none, ewald]` of H2 at `pos`, 1×1×3.
pub fn h2_gdf_energies(pos: &[[f64; 3]]) -> [f64; 2] {
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

pub fn kg_rhf_try(
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
