//! Stage 2: Gamma-point closed-shell KS-DFT (`ferric_pbc::dft`) — the Rust
//! port of `reference/pbc/pbc_dft.py` (FINDINGS "Iteration 8").
//!
//! What is independent of what:
//! * `uniform_grid_rks_matches_pinned_pyscf_uniform` — PySCF 2.13 pbc.dft.RKS
//!   with `UniformGrids` (AFTDF 61³, exxdiv ewald): a DIFFERENT integration
//!   construction (no partition at all) through our lattice-summed AOs, libxc
//!   kernel, XcBuilder injection and hybrid-K scaling. Fully external pins.
//! * `periodic_ssf_grid_is_the_molecular_grid_in_a_huge_box` — the partition
//!   over IMAGE atoms against the same kernel over the molecule's atoms only
//!   (no lattice at all): the exactness anchor of the image bookkeeping.
//! * `ssf_grid_rks_matches_the_prototype_construction` — the prototype's own
//!   A2/SSF numbers (`pbc_dft.PeriodicGrid(scheme="ssf", D=10)`, same TA-M4 +
//!   Lebedev-302 points): pins the whole grid construction (point count
//!   included) to an independent numpy implementation.
//! * `pbe0_box_limit_is_hyb_times_the_hf_madelung_term` — ferric's MOLECULAR
//!   KS on the identical grid rules (independent of every periodic piece).
//!
//! Mutation plan (each should turn a test red):
//! * drop the hybrid scaling of the injected K (use K, not a·K): PBE0 pins,
//!   the uniform PBE0 pin and the a⁻³ coefficient (4×) fail.
//! * image atoms not folded into the partition (neighbour list = cell atoms):
//!   the grid-integration pins and the prototype pins fail (partition of
//!   unity broken in the lattice).
//! * χ not lattice-summed (cell-0 shells only): S_grid vs S_latt and every
//!   energy fail.
//! * covering check removed: `too_small_neighbour_cutoff_is_refused` fails.
//!
//! Molecular KS unchanged (run alongside, not re-tested here): see the
//! regression list in the Stage-2 report (ferric-dft becke/grid/vxc/ks unit
//! tests, ferric-scf `dft_*`/`scf_injection` tests).

mod common;

use common::*;
use ferric_core::basis::BasisSet;
use ferric_core::mol::{Atom, Molecule};
use ferric_core::parallel::ParallelContext;
use ferric_dft::becke::{partition_weight_over, NeighbourAtom, PartitionScheme};
use ferric_dft::grid::AtomicGridConfig;
use ferric_dft::lebedev::lebedev;
use ferric_dft::prune::PruneScheme;
use ferric_dft::radial::treutler_ahlrichs_m4;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_pbc::dense_aft::{
    DenseAftEri, ExxDiv, DEFAULT_DENSE_AFT_MAX_BYTES, DEFAULT_DENSE_AFT_PRECISION,
};
use ferric_pbc::dft::{
    covering_radius_bound, gamma_rks, resolve_neighbour_cutoff, resolve_periodic_functional,
    GammaRksConfig, PeriodicDftError, PeriodicGrid, PeriodicGridConfig, PeriodicXc,
    PeriodicXcConfig,
};
use ferric_pbc::hcore::{periodic_hcore, PeriodicHcore, PeriodicHcoreConfig};
use ferric_pbc::lattice::Cell;
use ferric_pbc::uhf::GammaUhfIntegrals;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ndarray::Array2;
use ndarray_linalg::Inverse;

const OMEGA: f64 = 0.8;

// PySCF 2.13 pbc.dft.RKS, AFTDF mesh 61³, exxdiv='ewald', UniformGrids
// (test_prototype.py H2_PYSCF_UNIFORM; FINDINGS It. 8 "Our uniform-48³ RKS ≡
// PySCF UniformGrids to 1e-12"; tri: 48³ ≡ 64³).
const H2_PYSCF_UNIFORM_LDA: f64 = -1.521492168321;
const H2_PYSCF_UNIFORM_PBE0: f64 = -1.576992652865;
const TRI_PYSCF_UNIFORM_LDA: f64 = -2.205431956735;

// Prototype A2 grid, scheme="ssf", D = 10, TA-M4 75 × Lebedev 302, exxdiv
// ewald, dense pure-AFT J/K (pbc_dft.rks, conv 1e-12). Generated 2026-09-24
// with pbc_dft.PeriodicGrid(cell, 75, 302, D=10.0, scheme="ssf").
const H2_SSF_75X302_NPTS: usize = 36234;
const H2_SSF_75X302: [(&str, f64); 3] = [
    ("LDA", -1.521871150768),
    ("PBE", -1.527515243808),
    ("PBE0", -1.577295209269),
];
// Triclinic 4H, STO-3G s + Cartesian p(0.8), same construction (p functions:
// the AO-gradient and GGA paths with l = 1). Becke 75×302 of the same script
// gives 87501 points, the FINDINGS It. 8 table's count.
const TRI_SSF_75X302_NPTS: usize = 70784;
const TRI_SSF_75X302: [(&str, f64); 3] = [
    ("LDA", -2.205457322886),
    ("PBE", -2.245306875501),
    ("PBE0", -2.293844915602),
];
// Prototype uniform 48³ (≡ PySCF UniformGrids for LDA to 1e-12, above).
const TRI_PROTO_UNIFORM_PBE0: f64 = -2.293824583795;
// |∫ρ − N| and max|S_grid − S_latt| for the flat probe density D = (N/nao) S⁻¹
// (pbc_dft.grid_anchors), same grids, printed to 4 digits.
const H2_SSF_ANCHORS: [((usize, usize), f64, f64); 2] = [
    ((50, 110), 1.813e-3, 2.740e-3),
    ((75, 302), 9.501e-4, 1.657e-3),
];

fn li_atom(r: [f64; 3]) -> Atom {
    Atom {
        symbol: "Li".into(),
        z: 3,
        x: r[0],
        y: r[1],
        zpos: r[2],
        ghost: false,
        n_core_ecp: 0,
    }
}

struct Setup {
    cell: Cell,
    prep: PreparedBasis,
    bs: BasisSet,
    hc: PeriodicHcore,
    eri: DenseAftEri,
}

fn setup(cell: Cell, bs: BasisSet) -> Setup {
    let prep = prep_for(&cell, &bs);
    let hc = periodic_hcore(&cell, &prep, &PeriodicHcoreConfig::with_omega(OMEGA)).unwrap();
    // exxdiv is taken from GammaRksConfig by the builders, not from the tensor.
    let eri = DenseAftEri::build(
        &cell,
        &prep,
        &hc.s,
        ExxDiv::None,
        DEFAULT_DENSE_AFT_PRECISION,
        DEFAULT_DENSE_AFT_MAX_BYTES,
    )
    .unwrap();
    Setup {
        cell,
        prep,
        bs,
        hc,
        eri,
    }
}

fn h2_setup(a: f64) -> Setup {
    setup(h2_cell(a), pyscf_sto3g_h())
}

fn rks(su: &Setup, functional: &str, grid: PeriodicGridConfig) -> ferric_pbc::GammaRksResult {
    let mut cfg = GammaRksConfig::new(functional);
    cfg.grid = grid;
    gamma_rks(
        &su.cell,
        &su.prep,
        &su.hc,
        GammaUhfIntegrals::DenseAft(&su.eri),
        &cfg,
    )
    .unwrap_or_else(|e| panic!("gamma_rks {functional}: {e}"))
}

fn ssf_grid(n_rad: usize, n_ang: usize, d: f64) -> PeriodicGridConfig {
    PeriodicGridConfig {
        neighbour_cutoff: Some(d),
        ..PeriodicGridConfig::with_size(n_rad, n_ang)
    }
}

/// Flat probe density `D = (N/nao) S⁻¹` (`pbc_dft.probe_density`): tr(DS) = N
/// exactly, every AO occupied, ρ > 0 everywhere.
fn probe_density(su: &Setup) -> Array2<f64> {
    let n = su.cell.mol().nelec() as f64;
    let nao = su.prep.nbasis() as f64;
    su.hc.s.inv().unwrap() * (n / nao)
}

fn anchors(su: &Setup, grid: &PeriodicGrid) -> (f64, f64) {
    let xc = PeriodicXc::new(&su.cell, &su.bs, "LDA", grid, &PeriodicXcConfig::default()).unwrap();
    let d = probe_density(su);
    let n = su.cell.mol().nelec() as f64;
    let dn = (xc.integrate_density(&d).unwrap() - n).abs();
    let ds = max_abs_diff(&xc.overlap_on_grid(), &su.hc.s);
    (dn, ds)
}

// ─────────────────────────────────────────────────────────── (1) grid anchors

/// The grid integrates the lattice density to N and reproduces S_latt, with
/// the error shrinking with the (angular) grid size, and the numbers are the
/// prototype's own construction. Radial refinement at fixed 110 does NOT help
/// (the error is angular-limited, FINDINGS It. 8), so the sequence is
/// (50,110) → (75,302) → uniform (spectrally exact for all-Gaussian H).
#[test]
fn grid_integrates_lattice_density_and_overlap() {
    let su = h2_setup(4.0);
    let mut prev = (f64::INFINITY, f64::INFINITY);
    for &((nr, na), dn_ref, ds_ref) in &H2_SSF_ANCHORS {
        let grid = PeriodicGrid::build(&su.cell, &ssf_grid(nr, na, 10.0)).unwrap();
        let (dn, ds) = anchors(&su, &grid);
        eprintln!(
            "H2 a=4 SSF {nr}x{na}: npts {} <nb> {:.1} |dN| {dn:.4e} (proto {dn_ref:.3e}) \
             max|dS| {ds:.4e} (proto {ds_ref:.3e})",
            grid.len(),
            grid.mean_neighbours()
        );
        assert!(
            (dn - dn_ref).abs() < 2e-6,
            "|dN| {dn} vs prototype {dn_ref}"
        );
        assert!(
            (ds - ds_ref).abs() < 2e-6,
            "|dS| {ds} vs prototype {ds_ref}"
        );
        assert!(dn < prev.0 && ds < prev.1, "error did not decrease");
        prev = (dn, ds);
    }
    // Uniform mesh: spectral convergence, monotone, far below the atom grid.
    let mut prev_u = f64::INFINITY;
    for n in [8, 12, 16, 24] {
        let grid = PeriodicGrid::uniform(&su.cell, [n; 3]).unwrap();
        let (dn, ds) = anchors(&su, &grid);
        eprintln!("H2 a=4 uniform {n}^3: |dN| {dn:.3e} max|dS| {ds:.3e}");
        assert!(
            dn.max(ds) < prev_u,
            "uniform {n}: {dn} {ds} not below {prev_u}"
        );
        prev_u = dn.max(ds);
    }
    assert!(
        prev_u < 1e-6 * prev.0,
        "uniform 24^3 {prev_u} vs atom grid {}",
        prev.0
    );
    assert!(
        (PeriodicGrid::uniform(&su.cell, [5, 6, 7])
            .unwrap()
            .weight_sum()
            - 64.0)
            .abs()
            < 1e-12
    );
}

// ─────────────────────────────────────── (2) huge-box anchor vs molecular SSF

/// LiH (heteronuclear: the Bragg size adjustment is active) in a 30 Bohr box.
/// SSF has compact support, so near the molecule the crystal partition must
/// BE the molecular one: every periodic point within 0.08a of the centroid
/// is compared with `w_rl × SSF weight over the two molecular atoms` at the
/// bit-identical position. 0.08a, not 0.18a: the size adjustment (|a| = ½
/// for Li–H) maps ν → −1 + 4r/R, shrinking the exact zone to ~0.09 of the
/// image separation (prototype: an H point 8.7 Bohr out saw a Li image 51
/// Bohr away at a = 60). Prototype: 9.7e-17 at a = 30.
///
/// Control: the SAME comparison with Becke partitions on both sides is NOT
/// exact at a = 30 (polynomial tails see the images: prototype 4.7e-7), so
/// the test can see an image atom when one matters.
#[test]
fn periodic_ssf_grid_is_the_molecular_grid_in_a_huge_box() {
    let a = 30.0;
    let (r_li, r_h) = ([0.3, 0.2, 0.1], [0.3, 0.2, 3.1]);
    let mol = Molecule {
        atoms: vec![li_atom(r_li), h_atom(r_h)],
        charge: 0,
        multiplicity: 1,
    };
    let cell = Cell::new(mol.clone(), cubic(a)).unwrap();
    let nb_mol: Vec<NeighbourAtom> = mol
        .atoms
        .iter()
        .map(|at| NeighbourAtom {
            xyz: [at.x, at.y, at.zpos],
            z: at.z,
        })
        .collect();
    let cen = [0.3, 0.2, 1.6];
    let (n_rad, n_ang) = (30, 110);
    // Unpartitioned molecular points, keyed by (home, exact xyz bits).
    let (leb, lw) = lebedev(n_ang);
    let mut w_rl = std::collections::HashMap::new();
    for (home, at) in mol.atoms.iter().enumerate() {
        let (rs, ws) = treutler_ahlrichs_m4(at.z, n_rad);
        for (r, wr) in rs.iter().zip(&ws) {
            for (u, wl) in leb.iter().zip(&lw) {
                let p = [at.x + r * u[0], at.y + r * u[1], at.zpos + r * u[2]];
                w_rl.insert((home, p.map(f64::to_bits)), (p, wr * wl));
            }
        }
    }
    let mut worst = [0.0_f64; 2];
    let mut n_near = 0usize;
    for (k, scheme) in [PartitionScheme::Ssf, PartitionScheme::Becke]
        .into_iter()
        .enumerate()
    {
        let cfg = PeriodicGridConfig {
            partition: scheme,
            neighbour_cutoff: Some(27.0),
            ..PeriodicGridConfig::with_size(n_rad, n_ang)
        };
        let grid = PeriodicGrid::build(&cell, &cfg).unwrap();
        // Periodic weights by (home, exact xyz); a point the periodic build
        // dropped (exactly zero weight) counts as 0.
        let per: std::collections::HashMap<(usize, [u64; 3]), f64> = grid
            .points()
            .iter()
            .map(|g| ((g.home_atom, g.xyz.map(f64::to_bits)), g.weight))
            .collect();
        n_near = 0;
        for (&(home, bits), &(p, wrl)) in &w_rl {
            let dc = ((p[0] - cen[0]).powi(2) + (p[1] - cen[1]).powi(2) + (p[2] - cen[2]).powi(2))
                .sqrt();
            if dc >= 0.08 * a {
                continue;
            }
            let w_per = per.get(&(home, bits)).copied().unwrap_or(0.0);
            let w_mol = wrl * partition_weight_over(scheme, &nb_mol, home, p);
            worst[k] = worst[k].max((w_per - w_mol).abs());
            n_near += 1;
        }
        eprintln!(
            "LiH a={a} {scheme:?}: D {} (covering bound {:.3}), npts {}, near {n_near}, \
             max|dw| {:.2e}",
            grid.neighbour_cutoff(),
            grid.covering_bound(),
            grid.len(),
            worst[k]
        );
    }
    assert!(n_near > 500, "only {n_near} points within 0.08a");
    assert!(
        worst[0] < 1e-14,
        "SSF periodic vs molecular: {:.3e}",
        worst[0]
    );
    assert!(
        worst[1] > 1e-10,
        "Becke control saw no image: {:.3e}",
        worst[1]
    );
}

// ──────────────────────────────────────────────── (3) PySCF + prototype pins

/// Uniform grid (no partition) through the full KS machinery vs PySCF pbc
/// RKS UniformGrids: H2 a = 4 LDA and PBE0 (the hybrid-scaled Madelung K),
/// triclinic 4H s+p LDA (p functions → the AO gradient path).
#[test]
fn uniform_grid_rks_matches_pinned_pyscf_uniform() {
    let su = h2_setup(4.0);
    let grid = PeriodicGrid::uniform(&su.cell, [40; 3]).unwrap();
    for (xc, e_ref) in [
        ("LDA", H2_PYSCF_UNIFORM_LDA),
        ("PBE0", H2_PYSCF_UNIFORM_PBE0),
    ] {
        let e = uniform_rks(&su, &grid, xc);
        eprintln!(
            "H2 uniform 40^3 {xc}: {e:.12} (PySCF {e_ref:.12}, dE {:.2e})",
            e - e_ref
        );
        assert!((e - e_ref).abs() < 1e-8, "{xc}: {e} vs {e_ref}");
    }
    let tri = setup(triclinic_cell(), sp_basis_h());
    let grid = PeriodicGrid::uniform(&tri.cell, [48; 3]).unwrap();
    let e = uniform_rks(&tri, &grid, "LDA");
    eprintln!(
        "tri uniform 48^3 LDA: {e:.12} (PySCF {TRI_PYSCF_UNIFORM_LDA:.12}, dE {:.2e})",
        e - TRI_PYSCF_UNIFORM_LDA
    );
    assert!((e - TRI_PYSCF_UNIFORM_LDA).abs() < 1e-8);
    let e = uniform_rks(&tri, &grid, "PBE0");
    assert!((e - TRI_PROTO_UNIFORM_PBE0).abs() < 1e-8, "tri PBE0 {e}");
}

/// RKS on an explicit grid through `solve_rhf_injected` (what `gamma_rks`
/// does, with the grid supplied instead of built).
fn uniform_rks(su: &Setup, grid: &PeriodicGrid, xc: &str) -> f64 {
    use ferric_pbc::ewald::madelung_constant;
    use ferric_scf::rhf::{solve_rhf_injected, PeriodicInjection};
    let pxc = PeriodicXc::new(&su.cell, &su.bs, xc, grid, &PeriodicXcConfig::default()).unwrap();
    let vm = madelung_constant(&su.cell).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &su.prep).unwrap();
    let inj = PeriodicInjection {
        s: su.hc.s.clone(),
        h: su.hc.h.clone(),
        vnn: su.hc.enn,
        j: Box::new(su.eri.j_builder()),
        k: Box::new(su.eri.k_builder_with_madelung(vm)),
        xc: Some(Box::new(pxc)),
    };
    let r = solve_rhf_injected(
        &ParallelContext::default(),
        su.cell.mol(),
        &su.prep,
        op,
        &bounds,
        &GammaRksConfig::new(xc).scf,
        inj,
    )
    .unwrap();
    assert!(r.converged, "{xc} uniform RKS did not converge");
    r.energy
}

/// The default grid construction (SSF, D = 10, 75×302) against the numpy
/// prototype's own construction: identical point count and energies. The
/// residual vs the spectrally converged uniform reference is the partition
/// error (LDA −3.8e-4 Ha here; Becke −4.2e-4, PySCF A1 −4.4e-4).
#[test]
fn ssf_grid_rks_matches_the_prototype_construction() {
    let su = h2_setup(4.0);
    for (xc, e_ref) in H2_SSF_75X302 {
        let r = rks(&su, xc, ssf_grid(75, 302, 10.0));
        eprintln!(
            "H2 a=4 SSF 75x302 {xc}: {:.12} (proto {e_ref:.12}, dE {:.2e}), npts {}, \
             N_grid {:.6}, a_x {}",
            r.scf.energy,
            r.scf.energy - e_ref,
            r.n_grid_points,
            r.electrons_on_grid,
            r.exact_exchange_fraction
        );
        assert_eq!(r.n_grid_points, H2_SSF_75X302_NPTS);
        assert!(
            (r.scf.energy - e_ref).abs() < 1e-8,
            "{xc}: {}",
            r.scf.energy
        );
    }
    let tri = setup(triclinic_cell(), sp_basis_h());
    for (xc, e_ref) in TRI_SSF_75X302 {
        let r = rks(&tri, xc, ssf_grid(75, 302, 10.0));
        eprintln!(
            "tri 4H s+p SSF 75x302 {xc}: {:.12} (proto {e_ref:.12}, dE {:.2e}), npts {}",
            r.scf.energy,
            r.scf.energy - e_ref,
            r.n_grid_points
        );
        assert_eq!(r.n_grid_points, TRI_SSF_75X302_NPTS);
        assert!(
            (r.scf.energy - e_ref).abs() < 1e-8,
            "tri {xc}: {}",
            r.scf.energy
        );
    }
    let lda = rks(&su, "LDA", ssf_grid(75, 302, 10.0)).scf.energy;
    let d = lda - H2_PYSCF_UNIFORM_LDA;
    assert!(-4.0e-4 < d && d < -3.5e-4, "partition error {d}");
    // Default config == (75, 302) SSF with the automatic D (= 10 here).
    let def = rks(&su, "LDA", PeriodicGridConfig::default());
    assert_eq!(def.neighbour_cutoff, 10.0);
    assert!((def.scf.energy - lda).abs() < 1e-12);
}

// ─────────────────────────────────────────────── (4) hybrid box limit (a⁻³)

/// H2/STO-3G in a 16 Bohr box vs ferric's MOLECULAR RKS on the identical grid
/// rules (TA-M4 75 × Lebedev 302, Becke + size adjust, exact J/K). LDA: no
/// exact exchange and zero dipole → residual a⁻⁵ (prototype 8.7e-7). PBE0:
/// the residual is hyb × the HF exchange Makov–Payne term, c3 = −¼(4π/3)σ²
/// with σ² = 2.39406725 (PBE0 orbital, FINDINGS It. 8) = −2.50706; prototype
/// dE·a³ = −2.50645 at a = 16. Madelung on the full K would give 4×.
/// Becke partition on the periodic side so both grids share the partition.
#[test]
fn pbe0_box_limit_is_hyb_times_the_hf_madelung_term() {
    let a = 16.0;
    let su = h2_setup(a);
    let grid = PeriodicGridConfig {
        partition: PartitionScheme::Becke,
        neighbour_cutoff: Some(14.4),
        ..PeriodicGridConfig::with_size(75, 302)
    };
    let mol = hydrogens(&H2_ATOMS);
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &su.prep).unwrap();
    let molecular = |xc: &str| {
        let cfg = RhfConfig {
            xc: Some(xc.to_string()),
            dft_grid: Some(AtomicGridConfig {
                n_radial: 75,
                n_angular: 302,
                prune: None,
            }),
            // Exact four-centre J/K (no RI fitting error).
            df_j_aux: Some(String::new()),
            df_k_aux: Some(String::new()),
            density_conv: 1e-10,
            max_iter: 200,
            ..Default::default()
        };
        let r = solve_rhf(
            &ParallelContext::default(),
            &mol,
            &su.prep,
            op,
            &bounds,
            &cfg,
        )
        .unwrap();
        assert!(r.converged);
        r.energy
    };
    let sigma2 = 2.39406725_f64;
    let c3 = -0.25 * 4.0 * std::f64::consts::PI / 3.0 * sigma2;
    let d_hyb = rks(&su, "PBE0", grid.clone()).scf.energy - molecular("PBE0");
    let d_lda = rks(&su, "LDA", grid).scf.energy - molecular("LDA");
    eprintln!(
        "H2 a={a}: dE(PBE0) {d_hyb:.4e} dE*a^3 {:.5} (c3 {c3:.5}); dE(LDA) {d_lda:.3e}",
        d_hyb * a.powi(3)
    );
    assert!((d_hyb * a.powi(3) - c3).abs() < 1e-3 * c3.abs());
    assert!(d_lda.abs() < 2e-6 && d_lda.abs() < 1e-2 * c3.abs() / a.powi(3));
}

// ───────────────────────────────────────────────────── (5) named refusals

#[test]
fn too_small_neighbour_cutoff_is_refused() {
    let cell = h2_cell(4.0);
    let bound = covering_radius_bound(&cell);
    // Cubic a = 4 with a 1.4 Bohr H2: the lattice bound is 2√3 ≈ 3.46.
    assert!(bound > 1.0 && bound <= 2.0 * 3f64.sqrt() + 1e-12, "{bound}");
    match resolve_neighbour_cutoff(&cell, Some(1.0)) {
        Err(PeriodicDftError::NeighbourCutoffTooSmall { cutoff, required }) => {
            assert_eq!(cutoff, 1.0);
            assert_eq!(required, bound);
        }
        other => panic!("expected NeighbourCutoffTooSmall, got {other:?}"),
    }
    let e = PeriodicGrid::build(&cell, &ssf_grid(30, 50, 1.0)).unwrap_err();
    assert!(e.to_string().contains("neighbour_cutoff D = 1"), "{e}");
    // Automatic D: max(10, bound); a huge box raises it to the bound.
    assert_eq!(resolve_neighbour_cutoff(&cell, None).unwrap().0, 10.0);
    let big = h2_cell(40.0);
    let (d, b) = resolve_neighbour_cutoff(&big, None).unwrap();
    assert!(d == b && d > 20.0, "{d} {b}");
}

#[test]
fn unsupported_functionals_and_grid_knobs_are_refused_by_name() {
    for (name, word) in [
        ("wB97X-V", "range-separated hybrid"),
        ("SCAN", "meta-GGA"),
        ("B2PLYP", "double hybrid"),
    ] {
        let e = resolve_periodic_functional(name).unwrap_err().to_string();
        assert!(e.contains(word), "{name}: {e}");
    }
    let (_, a) = resolve_periodic_functional("PBE0").unwrap();
    assert!((a - 0.25).abs() < 1e-15);
    assert_eq!(resolve_periodic_functional("PBE").unwrap().1, 0.0);

    let cell = h2_cell(4.0);
    let pruned = PeriodicGridConfig {
        prune: Some(PruneScheme::NwchemLike),
        ..PeriodicGridConfig::default()
    };
    let e = PeriodicGrid::build(&cell, &pruned).unwrap_err().to_string();
    assert!(e.contains("prune is not supported"), "{e}");
    let e = PeriodicGrid::build(&cell, &PeriodicGridConfig::with_size(75, 590))
        .unwrap_err()
        .to_string();
    assert!(e.contains("invalid n_angular"), "{e}");
}

#[test]
fn unsupported_scf_features_are_refused_by_name() {
    let su = h2_setup(4.0);
    let run = |edit: &dyn Fn(&mut GammaRksConfig)| {
        let mut cfg = GammaRksConfig::new("PBE");
        edit(&mut cfg);
        gamma_rks(
            &su.cell,
            &su.prep,
            &su.hc,
            GammaUhfIntegrals::DenseAft(&su.eri),
            &cfg,
        )
        .unwrap_err()
        .to_string()
    };
    let cases: [(&str, &dyn Fn(&mut GammaRksConfig)); 7] = [
        ("RhfConfig.newton_trigger", &|c| c.scf.newton_trigger = 1e-2),
        ("RhfConfig.trah_trigger", &|c| {
            c.scf.trah_trigger = Some(1e-2)
        }),
        ("RhfConfig.check_stability", &|c| {
            c.scf.check_stability = true
        }),
        ("RhfConfig.xc", &|c| c.scf.xc = Some("PBE".into())),
        ("scf.xc_omega", &|c| c.scf.xc_omega = Some(0.3)),
        ("scf.dft_grid", &|c| {
            c.scf.dft_grid = Some(AtomicGridConfig::default())
        }),
        ("range-separated hybrid", &|c| {
            c.functional = "wB97X-V".into()
        }),
    ];
    for (word, edit) in cases {
        let e = run(edit);
        assert!(e.contains(word), "{word}: {e}");
    }
    // Open shell: a 3-electron cell.
    let odd = setup(
        Cell::new(hydrogens(&TRI_ATOMS[..3]), TRI_A).unwrap(),
        pyscf_sto3g_h(),
    );
    let e = gamma_rks(
        &odd.cell,
        &odd.prep,
        &odd.hc,
        GammaUhfIntegrals::DenseAft(&odd.eri),
        &GammaRksConfig::new("LDA"),
    )
    .unwrap_err()
    .to_string();
    assert!(e.contains("open shell"), "{e}");
}

/// The UHF injected path refuses an XcBuilder by name (no periodic UKS).
#[test]
fn uhf_injected_refuses_an_xc_builder() {
    use ferric_scf::rhf::PeriodicInjection;
    use ferric_scf::uhf::{solve_uhf_injected, UhfConfig};
    let su = h2_setup(4.0);
    let grid = PeriodicGrid::build(&su.cell, &ssf_grid(30, 50, 10.0)).unwrap();
    let pxc =
        PeriodicXc::new(&su.cell, &su.bs, "LDA", &grid, &PeriodicXcConfig::default()).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &su.prep).unwrap();
    let inj = PeriodicInjection {
        s: su.hc.s.clone(),
        h: su.hc.h.clone(),
        vnn: su.hc.enn,
        j: Box::new(su.eri.j_builder()),
        k: Box::new(su.eri.k_builder()),
        xc: Some(Box::new(pxc)),
    };
    let cfg = UhfConfig {
        use_sad_guess: false,
        ..Default::default()
    };
    let e = solve_uhf_injected(
        &ParallelContext::default(),
        su.cell.mol(),
        &su.prep,
        &bounds,
        &cfg,
        inj,
        None,
    )
    .unwrap_err()
    .to_string();
    assert!(e.contains("PeriodicInjection.xc"), "{e}");
}
