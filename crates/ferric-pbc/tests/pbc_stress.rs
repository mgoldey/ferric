//! Gamma-point analytic stress tensor (`ferric_pbc::stress`), the Rust port
//! of `reference/pbc/pbc_stress.py` (FINDINGS "Iteration 19").
//!
//! What is anchored against what:
//! * EXACTNESS ANCHOR: analytic `dE/dε_ab` vs central FD (h = 1e-4) of
//!   ferric's OWN Gamma energy on strained cells (`Cell::strained`: lattice
//!   AND atoms strained, fixed fractional coordinates, every G sphere and
//!   image list FROZEN as integer indices at the reference cell), ALL NINE
//!   components, both exxdiv. Prototype floors at h = 1e-4 (the h² f‴/6
//!   truncation, clean h² in every h-scan): H2 RHF 6.7e-9 / 8.5e-9, H3 UHF
//!   5.4e-8, tri RHF 3.5e-8, tri UHF 2.1e-7 ((2,2); Richardson 1.8e-10).
//!   The Rust energy differs from the prototype's pure-AFT route in `V_ne`
//!   (Ewald split ω = 0.8, Gaussian nuclei; the prototype's SR route floor was
//!   5.8e-9..7.6e-9 on s-only H2) and in the primitive screens; the forces'
//!   H2 anchor reached 2.4e-9 on the same energy. Bar `FD_BAR = 1e-7` for
//!   H2, `FD_BAR_OPEN = 2e-7` for H3 UHF (4 × its truncation floor).
//! * Mutants (`GammaStressConfig::mutation`, hidden): each must miss FD by
//!   > `MUTANT_BAR = 1e-3` on at least one of the nine components. Prototype
//!   smallest misses: H2 9.2e-2 (`NoPulay`, SR route) .. 1.3 (`MadelungS`),
//!   RS-GDF H2 3.35e-2 (`NoJ2G0Vol`), atom-grid KS 7.9e-2. The volume-only
//!   mutants (`NoVolume`, `NoEwaldBg`, `NoC0Volume`) are diagonal by
//!   construction: their OFF-diagonal miss is asserted to sit at the FD
//!   floor (a check that the anchor looks at every component, not a defect).
//!   `NoAuxFt` (aux FT strain-free in J3 AND J2) errs only by the fit error
//!   (prototype 7.7e-6 on H2 ET-40) — reported, not asserted; `NoAuxFt3` is
//!   the sharp form (prototype 7.3e-2).
//! * Identities: `|dE/dε − (dE/dε)ᵀ| ≤ 1e-10` for HF (rotation invariance
//!   with frozen sets; prototype 4e-17..1.1e-12); `dE/dε(ewald) − dE/dε(none)`
//!   = the Madelung part `−(αN/2) dv_M/dε`, whose trace is `αNv_M/2`
//!   (Euler homogeneity, `tr dv_M/dε = −v_M`); `−tr σ/3 = −dE/dΩ` vs FD of
//!   isotropic scaling (prototype 5.1e-11 on H2); UHF(D/2, D/2) ≡ RHF(D) on
//!   one density.
//! * Kernel level: Ewald `E_nn` strain vs FD and ω-independence (prototype
//!   vs PySCF-FD 1.3e-10); `dv_M/dε` vs FD of `madelung_constant`; the pair-FT
//!   strain kernel vs FD of `pair_ft` at fixed Miller indices (triclinic s+p);
//!   the strain kernel's own `P` reproduces the energy's `Σ D V_LR` and the
//!   dense-AFT `E_2e`.
//!
//! Slow (`#[ignore]`): H3 UHF, triclinic s+p RHF, H3 UKS LDA/PBE/PBE0 on the
//! atom grid (FD h = 2.5e-5: at 1e-4 grid points cross the hard `D`
//! neighbour mask under strain — prototype 1.8e-5 at 1e-4, 3.9e-9 at
//! 2.5e-5), RS-GDF H2 ET-40.
//!
//! NOT covered here: PySCF `rks_stress` end-to-end (the uniform KS grid is
//! not an energy path in Rust), k-points, ECPs, ROHF/ROKS.

mod common;

use common::*;
use ferric_core::basis::{BasisSet, Shell};
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_pbc::dense_aft::{
    DenseAftEri, ExxDiv, DEFAULT_DENSE_AFT_MAX_BYTES, DEFAULT_DENSE_AFT_PRECISION,
};
use ferric_pbc::dft::{gamma_uks, GammaUksConfig, PeriodicGridConfig};
use ferric_pbc::ewald::{
    ewald_nuclear_repulsion_with_precision, ewald_nuclear_strain, madelung_constant,
    madelung_strain, DEFAULT_EWALD_PRECISION,
};
use ferric_pbc::grad::RsGdfGradSource;
use ferric_pbc::hcore::{periodic_hcore, PeriodicHcore, PeriodicHcoreConfig};
use ferric_pbc::lattice::Cell;
use ferric_pbc::pair_ft::{pair_ft_strain_chunked, pair_ft_with_thresh, PairFtStrainTerms};
use ferric_pbc::rsgdf::{RsGdf, RsGdfConfig, DEFAULT_RSGDF_LINDEP};
use ferric_pbc::stress::{
    gamma_rhf_stress, gamma_rhf_stress_rsgdf, gamma_uhf_stress, gamma_uks_stress, GammaStress,
    GammaStressConfig, StressMutation,
};
use ferric_pbc::uhf::{gamma_uhf, GammaUhfConfig, GammaUhfIntegrals, GammaUhfResult};
use ferric_scf::result::{ScfResult, Spin};
use ndarray::{Array2, Array3};
use num_complex::Complex64;
use std::collections::HashMap;
use std::sync::OnceLock;

type Mat3 = [[f64; 3]; 3];

/// ω of the nuclear-attraction split (as `pbc_grad.rs`): not 1, so a 1/ω vs
/// 1/ω² slip in the c0 volume term is visible.
const OMEGA: f64 = 0.8;
const HCORE_PRECISION: f64 = 1e-14;
const FD_H: f64 = 1e-4;
/// Atom-grid KS step (module doc).
const FD_H_KS: f64 = 2.5e-5;
const FD_BAR: f64 = 1e-7;
const FD_BAR_OPEN: f64 = 2e-7;
const MUTANT_BAR: f64 = 1e-3;
const ANTISYM_BAR: f64 = 1e-10;
/// Triclinic s+p: the measured libint2 p-shell floor is 5.4e-8 at the
/// default nucleus exponent (see the triclinic test).
const TRI_ANTISYM_BAR: f64 = 5e-7;
/// `dE/dε(ewald) − dE/dε(none)` vs the Madelung part: two independent SCFs
/// (density_conv 1e-10), so SCF-convergence limited, not roundoff.
const MADELUNG_ID_BAR: f64 = 1e-9;
const IDENTITY_BAR: f64 = 1e-12;
/// The RS-GDF build's ω and the ample budget of `pbc_grad_rsgdf.rs`.
const GDF_OMEGA: f64 = 1.0;
const AMPLE: usize = 1 << 31;

/// `run_stress_anchor.py` H2 (Bohr), a = 4, STO-3G: off-axis so every
/// component is non-trivial.
const H2_ATOMS: [[f64; 3]; 2] = [[0.3, 0.2, 0.1], [0.35, 0.12, 1.5]];
/// `run_stress_anchor.py` H3 (Bohr), a = 4.5, STO-3G, doublet (2, 1).
const H3_ATOMS: [[f64; 3]; 3] = [[0.3, 0.2, 0.1], [0.35, 0.12, 1.5], [1.6, 0.9, 0.7]];
const H3_A: f64 = 4.5;
/// `run_grad_oracle.py` TRI_MOVED (Bohr), s+p basis.
const TRI_MOVED: [[f64; 3]; 4] = [
    [0.13, 0.25, 0.31],
    [0.02, 0.27, 1.66],
    [2.47, 2.41, 2.25],
    [3.52, 2.98, 2.71],
];

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

fn cell_at(pos: &[[f64; 3]], lattice: [[f64; 3]; 3], mult: usize) -> Cell {
    let mut mol: Molecule = hydrogens(pos);
    mol.multiplicity = mult;
    Cell::new(mol, lattice).expect("cell")
}

fn eps_at(i: usize, j: usize, h: f64) -> Mat3 {
    let mut e = [[0.0; 3]; 3];
    e[i][j] = h;
    e
}

/// `cell` strained by `h` in component `(i, j)`, index sets frozen at `cell`.
fn strained(cell: &Cell, i: usize, j: usize, h: f64) -> Cell {
    cell.strained(&eps_at(i, j, h)).expect("strained cell")
}

/// Central FD `dE/dε_ij` of every energy `energies` returns, all nine
/// components (one `Mat3` per energy).
fn fd_strain<F>(cell: &Cell, h: f64, energies: F) -> Vec<Mat3>
where
    F: Fn(&Cell) -> Vec<f64>,
{
    let mut out: Vec<Mat3> = Vec::new();
    for (i, j) in ALL9 {
        let ep = energies(&strained(cell, i, j, h));
        let em = energies(&strained(cell, i, j, -h));
        if out.is_empty() {
            out = vec![[[0.0; 3]; 3]; ep.len()];
        }
        for k in 0..ep.len() {
            out[k][i][j] = (ep[k] - em[k]) / (2.0 * h);
        }
    }
    out
}

/// `(4 FD(h/2) − FD(h))/3` from two FD tables.
fn richardson(fd_h: &Mat3, fd_h2: &Mat3) -> Mat3 {
    let mut o = [[0.0; 3]; 3];
    for a in 0..3 {
        for b in 0..3 {
            o[a][b] = (4.0 * fd_h2[a][b] - fd_h[a][b]) / 3.0;
        }
    }
    o
}

fn max_err(a: &Mat3, b: &Mat3) -> f64 {
    let mut m = 0.0_f64;
    for i in 0..3 {
        for j in 0..3 {
            let d = (a[i][j] - b[i][j]).abs();
            assert!(d.is_finite(), "non-finite stress difference");
            m = m.max(d);
        }
    }
    m
}

fn max_offdiag_err(a: &Mat3, b: &Mat3) -> f64 {
    let mut m = 0.0_f64;
    for i in 0..3 {
        for j in 0..3 {
            if i != j {
                m = m.max((a[i][j] - b[i][j]).abs());
            }
        }
    }
    m
}

fn antisym(a: &Mat3) -> f64 {
    let mut m = 0.0_f64;
    for i in 0..3 {
        for j in 0..3 {
            m = m.max((a[i][j] - a[j][i]).abs());
        }
    }
    m
}

fn diff(a: &Mat3, b: &Mat3) -> Mat3 {
    let mut o = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            o[i][j] = a[i][j] - b[i][j];
        }
    }
    o
}

fn trace(a: &Mat3) -> f64 {
    a[0][0] + a[1][1] + a[2][2]
}

fn fmt3(a: &Mat3) -> String {
    a.iter()
        .map(|r| format!("  [{:+.10e} {:+.10e} {:+.10e}]", r[0], r[1], r[2]))
        .collect::<Vec<_>>()
        .join("\n")
}

fn report(tag: &str, st: &GammaStress, fd: &Mat3) {
    eprintln!(
        "{tag}: max|an − FD| = {:.2e}, |an − anᵀ| {:.1e}, |FD − FDᵀ| {:.1e}, comm {:.1e}, \
         Ω {:.4}, P = {:+.10e}, SR triplets {}, G(LR) {}, G(ERI) {}\n analytic dE/dε:\n{}\n \
         analytic − FD:\n{}",
        max_err(&st.de_deps, fd),
        antisym(&st.de_deps),
        antisym(fd),
        st.commutator,
        st.volume,
        st.pressure(),
        st.n_sr_triplets,
        st.n_g_lr,
        st.n_g_eri,
        fmt3(&st.de_deps),
        fmt3(&diff(&st.de_deps, fd))
    );
}

// ------------------------------------------------------------ dense RHF / UHF

struct Setup {
    cell: Cell,
    prep: PreparedBasis,
    hc: PeriodicHcore,
    eri: DenseAftEri,
}

/// hcore + the dense tensor (its own exxdiv is irrelevant: `with_exxdiv` /
/// the UHF config switch the Madelung term).
fn setup(cell: Cell, bs: &BasisSet) -> Setup {
    let prep = prep_for(&cell, bs);
    let hc = periodic_hcore(&cell, &prep, &hcore_cfg()).expect("hcore");
    let eri = DenseAftEri::build(
        &cell,
        &prep,
        &hc.s,
        ExxDiv::None,
        DEFAULT_DENSE_AFT_PRECISION,
        DEFAULT_DENSE_AFT_MAX_BYTES,
    )
    .expect("dense AFT");
    Setup {
        cell,
        prep,
        hc,
        eri,
    }
}

fn rhf(su: &Setup, exx: ExxDiv) -> ScfResult {
    let eri = su.eri.clone().with_exxdiv(&su.cell, exx).unwrap();
    gamma_rhf(&su.cell, &su.prep, &su.hc, &eri)
}

fn scfg(mutation: Option<StressMutation>) -> GammaStressConfig {
    GammaStressConfig {
        mutation,
        ..Default::default()
    }
}

fn rhf_stress(su: &Setup, scf: &ScfResult, exx: ExxDiv, m: Option<StressMutation>) -> GammaStress {
    gamma_rhf_stress(
        &su.cell,
        &su.prep,
        &hcore_cfg(),
        &su.hc,
        &su.eri,
        scf,
        exx,
        &scfg(m),
    )
    .expect("gamma_rhf_stress")
}

fn uhf_stress(su: &Setup, scf: &ScfResult, exx: ExxDiv, m: Option<StressMutation>) -> GammaStress {
    gamma_uhf_stress(
        &su.cell,
        &su.prep,
        &hcore_cfg(),
        &su.hc,
        &su.eri,
        scf,
        exx,
        &scfg(m),
    )
    .expect("gamma_uhf_stress")
}

type Mos = Option<(Array2<f64>, Array2<f64>)>;

fn mos_of(r: &ScfResult) -> Mos {
    Some((r.mos_alpha.clone(), r.mos_beta.clone().expect("beta MOs")))
}

fn uhf_cfg(exx: ExxDiv, init: Mos) -> GammaUhfConfig {
    GammaUhfConfig {
        exxdiv: exx,
        initial_mos: init,
        ..GammaUhfConfig::default()
    }
}

fn uhf(su: &Setup, cfg: &GammaUhfConfig) -> GammaUhfResult {
    let r = gamma_uhf(
        &su.cell,
        &su.prep,
        &su.hc,
        GammaUhfIntegrals::DenseAft(&su.eri),
        cfg,
    )
    .expect("gamma_uhf");
    assert!(r.scf.converged);
    r
}

/// The same closed-shell density as an unrestricted result `(D/2, D/2)`.
fn as_unrestricted(r: &ScfResult) -> ScfResult {
    let mut u = r.clone();
    let half = &r.density_total * 0.5;
    u.spin = Spin::Unrestricted;
    u.density_alpha = half.clone();
    u.density_beta = Some(half);
    u.mos_beta = Some(r.mos_alpha.clone());
    u.eps_beta = Some(r.eps_alpha.clone());
    u.fock_beta = Some(r.fock_alpha.clone());
    u
}

fn h2_cell_g() -> Cell {
    cell_at(&H2_ATOMS, cubic(4.0), 1)
}

fn h3_cell() -> Cell {
    cell_at(&H3_ATOMS, cubic(H3_A), 2)
}

static H2_FD: OnceLock<Vec<Mat3>> = OnceLock::new();

/// FD `[none, ewald]` of the H2/STO-3G a = 4 RHF energy, all nine
/// components. Also checks the frozen sets: every strained build sums the
/// reference's half-sphere G count and pair-image count.
fn h2_fd() -> &'static Vec<Mat3> {
    H2_FD.get_or_init(|| {
        let bs = pyscf_sto3g_h();
        let reference = setup(h2_cell_g(), &bs);
        let (ng, ni) = (reference.eri.n_g_half(), reference.hc.n_images);
        fd_strain(&h2_cell_g(), FD_H, |c| {
            let su = setup(c.clone(), &bs);
            assert_eq!(su.eri.n_g_half(), ng, "frozen ERI G sphere changed size");
            assert_eq!(su.hc.n_images, ni, "frozen pair-image list changed size");
            [ExxDiv::None, ExxDiv::Ewald]
                .iter()
                .map(|&e| rhf(&su, e).energy)
                .collect()
        })
    })
}

// ------------------------------------------------------------- kernel level

#[test]
fn ewald_strain_matches_fd_is_omega_independent_and_madelung_is_homogeneous() {
    let h = 1e-5;
    for (tag, cell) in [
        ("H2 a=4", h2_cell_g()),
        ("tri moved", cell_at(&TRI_MOVED, TRI_A, 1)),
    ] {
        let prec = DEFAULT_EWALD_PRECISION;
        let s1 = ewald_nuclear_strain(&cell, 1.0, prec).unwrap().total();
        let s2 = ewald_nuclear_strain(&cell, 2.3, prec).unwrap().total();
        let fd = fd_strain(&cell, h, |c| {
            vec![ewald_nuclear_repulsion_with_precision(c, 1.0, prec).unwrap()]
        });
        let err = max_err(&s1, &fd[0]);
        let w_err = max_err(&s1, &s2);
        // dv_M/dε vs FD of the Madelung constant; Euler homogeneity.
        let dvm = madelung_strain(&cell).unwrap();
        let fdm = fd_strain(&cell, h, |c| vec![madelung_constant(c).unwrap()]);
        let vm = madelung_constant(&cell).unwrap();
        let m_err = max_err(&dvm, &fdm[0]);
        let euler = (trace(&dvm) + vm).abs();
        eprintln!(
            "Ewald strain {tag}: vs FD {err:.2e}, ω 1 vs 2.3 {w_err:.2e}, |tr| {:.3e}; \
             dv_M/dε vs FD {m_err:.2e}, tr + v_M {euler:.2e} (v_M {vm:.6})\n{}",
            trace(&s1).abs(),
            fmt3(&s1)
        );
        assert!(trace(&s1).abs() > 1e-2, "Ewald strain must be non-trivial");
        assert!(err < 1e-8, "{tag}: {err:e}");
        assert!(w_err < 1e-10, "{tag}: {w_err:e}");
        assert!(m_err < 1e-8, "{tag}: {m_err:e}");
        assert!(euler < 1e-12, "{tag}: {euler:e}");
        // Rotation invariance of a scalar function of the lattice.
        assert!(antisym(&s1) < 1e-12 && antisym(&dvm) < 1e-12);
    }
}

#[test]
fn pair_ft_strain_matches_fd_at_fixed_miller_index() {
    let cell = triclinic_cell();
    let bs = sp_basis_h();
    let prep = prep_for(&cell, &bs);
    let idx: Vec<[i64; 3]> = cell
        .gvector_indices(3.5)
        .unwrap()
        .into_iter()
        .filter(|n| *n != [0, 0, 0])
        .take(14)
        .collect();
    assert!(idx.len() >= 10);
    let gs: Vec<[f64; 3]> = idx.iter().map(|n| cell.gvector_from_index(*n)).collect();
    let thresh = 1e-15;
    let run = |terms: PairFtStrainTerms| -> (Array3<Complex64>, Vec<Array3<Complex64>>) {
        let mut got = None;
        pair_ft_strain_chunked(
            &cell,
            &prep,
            &gs,
            thresh,
            1 << 30,
            0,
            terms,
            |_, _, p, dp| {
                got = Some((p.clone(), dp.to_vec()));
                Ok(())
            },
        )
        .unwrap();
        got.expect("one chunk")
    };
    let (p, dp) = run(PairFtStrainTerms::ALL);
    let pref = pair_ft_with_thresh(&cell, &prep, &gs, thresh).unwrap();
    let dpmax = (&p - &pref).iter().fold(0.0_f64, |m, z| m.max(z.norm()));
    assert!(dpmax < 1e-13, "strain kernel P vs pair_ft: {dpmax:e}");
    let (_, dp_nc) = run(PairFtStrainTerms {
        centres: false,
        g_shape: true,
    });
    let (_, dp_ng) = run(PairFtStrainTerms {
        centres: true,
        g_shape: false,
    });
    let h = 1e-5;
    let n = prep.nbasis();
    let (mut worst, mut fmax, mut w_nc, mut w_ng) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
    for (a, b) in ALL9 {
        let fd_p = |s: f64| {
            let c = strained(&cell, a, b, s * h);
            let pr = prep_for(&c, &bs);
            let g: Vec<[f64; 3]> = idx.iter().map(|m| c.gvector_from_index(*m)).collect();
            pair_ft_with_thresh(&c, &pr, &g, thresh).unwrap()
        };
        let (pp, pm) = (fd_p(1.0), fd_p(-1.0));
        let k = 3 * a + b;
        for m in 0..n {
            for q in 0..n {
                for g in 0..gs.len() {
                    let f = (pp[[m, q, g]] - pm[[m, q, g]]) / (2.0 * h);
                    fmax = fmax.max(f.norm());
                    worst = worst.max((dp[k][[m, q, g]] - f).norm());
                    w_nc = w_nc.max((dp_nc[k][[m, q, g]] - f).norm());
                    w_ng = w_ng.max((dp_ng[k][[m, q, g]] - f).norm());
                }
            }
        }
    }
    eprintln!(
        "pair_ft strain vs FD: {worst:.2e} (|FD| max {fmax:.2e}); mutants: no centres \
         {w_nc:.2e}, no G shape {w_ng:.2e}"
    );
    assert!(fmax > 1e-3, "the strain must change P");
    assert!(worst < 1e-7 * fmax.max(1.0), "{worst:e}");
    assert!(w_nc > 1e-3 * fmax, "FtNoCentres escaped: {w_nc:e}");
    assert!(w_ng > 1e-3 * fmax, "GUnstrained escaped: {w_ng:e}");
}

// --------------------------------------------------------------- H2 RHF

#[test]
fn h2_rhf_stress_matches_fd_of_own_energy_all_nine_both_exxdiv() {
    let su = setup(h2_cell_g(), &pyscf_sto3g_h());
    let fdv = h2_fd();
    for (k, exx) in [ExxDiv::None, ExxDiv::Ewald].into_iter().enumerate() {
        let scf = rhf(&su, exx);
        let st = rhf_stress(&su, &scf, exx, None);
        report(&format!("H2/STO-3G a=4 RHF {exx:?}"), &st, &fdv[k]);
        let err = max_err(&st.de_deps, &fdv[k]);
        assert!(err < FD_BAR, "{exx:?}: {err:e}");
        assert!(
            antisym(&st.de_deps) < ANTISYM_BAR,
            "{exx:?}: antisymmetric part {:e}",
            antisym(&st.de_deps)
        );
        // The strain kernel's own P reproduces the energy's pieces.
        let d = &scf.density_total;
        let e_vlr = (d * &su.hc.v_lr).sum();
        assert!(
            (st.e_vlr - e_vlr).abs() < 1e-10,
            "Σ D V_LR: kernel {} vs hcore {e_vlr}",
            st.e_vlr
        );
        // Dense-AFT E_2e = ½ tr(D J[D]) − ¼ tr(D K[D]) (exxdiv none tensor).
        let n = su.prep.nbasis();
        let (mut j, mut kk) = (Array2::<f64>::zeros((n, n)), Array2::<f64>::zeros((n, n)));
        {
            use ferric_scf::fock::{JBuilder, KBuilder};
            su.eri.j_builder().build(d, &mut j).unwrap();
            su.eri
                .k_builder_with_madelung(0.0)
                .build(d, &mut kk)
                .unwrap();
        }
        let e2 = 0.5 * (d * &j).sum() - 0.25 * (d * &kk).sum();
        assert!(
            (st.e_eri - e2).abs() < 1e-10,
            "E_2e: kernel {} vs tensor {e2}",
            st.e_eri
        );
        // Every term is live (an anchor that passed with one absent would not
        // test it).
        for (name, m) in [
            ("overlap", &st.parts.overlap),
            ("kinetic", &st.parts.kinetic),
            ("vsr", &st.parts.vsr),
            ("vlr", &st.parts.vlr),
            ("eri", &st.parts.eri),
            ("nn_lr", &st.parts.nn_lr),
        ] {
            let mx = m.iter().flatten().fold(0.0_f64, |a, v| a.max(v.abs()));
            assert!(mx > 1e-3, "{name} part is {mx:e}");
        }
    }
}

/// `dE/dε(ewald) − dE/dε(none) = −(N/2) dv_M/dε` (RHF, α = 1; the S-term
/// cancels as for the forces) and the Madelung part's trace is `N v_M/2`.
#[test]
fn h2_rhf_ewald_minus_none_is_the_madelung_strain() {
    let su = setup(h2_cell_g(), &pyscf_sto3g_h());
    let sn = rhf_stress(&su, &rhf(&su, ExxDiv::None), ExxDiv::None, None);
    let se = rhf_stress(&su, &rhf(&su, ExxDiv::Ewald), ExxDiv::Ewald, None);
    let delta = diff(&se.de_deps, &sn.de_deps);
    let err = max_err(&delta, &se.parts.madelung);
    let want = se.electrons * se.madelung / 2.0;
    let tr_err = (trace(&se.parts.madelung) - want).abs();
    let mut formula = se.madelung_strain;
    for row in formula.iter_mut() {
        for v in row.iter_mut() {
            *v *= -0.5 * se.electrons;
        }
    }
    eprintln!(
        "H2 RHF: |Δ(ewald − none) − Madelung part| {err:.2e}; tr(Madelung part) {:.10} vs \
         N v_M/2 {want:.10}; N {:.12}\n{}",
        trace(&se.parts.madelung),
        se.electrons,
        fmt3(&delta)
    );
    assert!(se.madelung > 0.1);
    assert!(err < MADELUNG_ID_BAR, "{err:e}");
    assert!(tr_err < 1e-10, "{tr_err:e}");
    assert!(max_err(&se.parts.madelung, &formula) < 1e-10);
    assert!(sn.parts.madelung.iter().flatten().all(|v| *v == 0.0));
}

/// `−tr(σ)/3 = −dE/dΩ` against FD of isotropic scaling (consistency, not
/// independent of the anchor).
#[test]
fn h2_rhf_pressure_matches_isotropic_scaling() {
    let bs = pyscf_sto3g_h();
    let cell = h2_cell_g();
    let su = setup(cell.clone(), &bs);
    let st = rhf_stress(&su, &rhf(&su, ExxDiv::Ewald), ExxDiv::Ewald, None);
    let e_at = |s: f64| {
        let iso = [
            [s * FD_H, 0.0, 0.0],
            [0.0, s * FD_H, 0.0],
            [0.0, 0.0, s * FD_H],
        ];
        let c = cell.strained(&iso).unwrap();
        rhf(&setup(c, &bs), ExxDiv::Ewald).energy
    };
    // dE/de = tr(dE/dε), Ω(e) = Ω (1 + e)³.
    let de = (e_at(1.0) - e_at(-1.0)) / (2.0 * FD_H);
    let p_fd = -de / (3.0 * cell.volume());
    let err = (st.pressure() - p_fd).abs();
    eprintln!(
        "H2 RHF ewald: P = −tr σ/3 = {:.12e}, −dE/dΩ (FD) = {p_fd:.12e}, diff {err:.2e}",
        st.pressure()
    );
    assert!(err < 1e-9, "{err:e}");
}

/// UHF(D/2, D/2) ≡ RHF(D) on ONE density, both exxdiv.
#[test]
fn closed_shell_uhf_stress_is_the_rhf_stress() {
    let su = setup(h2_cell_g(), &pyscf_sto3g_h());
    for exx in [ExxDiv::None, ExxDiv::Ewald] {
        let scf = rhf(&su, exx);
        let sr = rhf_stress(&su, &scf, exx, None);
        let su_ = uhf_stress(&su, &as_unrestricted(&scf), exx, None);
        let d = max_err(&sr.de_deps, &su_.de_deps);
        eprintln!("H2 {exx:?}: |UHF(D/2,D/2) − RHF(D)| stress = {d:.2e}");
        assert!(d < IDENTITY_BAR, "{exx:?}: {d:e}");
    }
}

/// Every mutant must miss FD on at least one component; the volume-only
/// ones are diagonal by construction (their off-diagonal residual is the FD
/// floor).
#[test]
fn h2_fd_anchor_catches_each_mutant() {
    let su = setup(h2_cell_g(), &pyscf_sto3g_h());
    let fdv = h2_fd();
    let scf_n = rhf(&su, ExxDiv::None);
    let scf_e = rhf(&su, ExxDiv::Ewald);
    let diagonal_only = [
        StressMutation::NoVolume,
        StressMutation::NoEwaldBg,
        StressMutation::NoC0Volume,
    ];
    let runs = [
        (StressMutation::NoPulay, ExxDiv::None),
        (StressMutation::FtNoCentres, ExxDiv::None),
        (StressMutation::GUnstrained, ExxDiv::None),
        (StressMutation::NoVolume, ExxDiv::None),
        (StressMutation::NoEwaldLr, ExxDiv::None),
        (StressMutation::NoEwaldBg, ExxDiv::None),
        (StressMutation::NoC0Volume, ExxDiv::None),
        (StressMutation::NoSrImages, ExxDiv::None),
        (StressMutation::NoMadelung, ExxDiv::Ewald),
        (StressMutation::MadelungS, ExxDiv::Ewald),
    ];
    for (m, exx) in runs {
        let (scf, k) = match exx {
            ExxDiv::None => (&scf_n, 0),
            ExxDiv::Ewald => (&scf_e, 1),
        };
        let st = rhf_stress(&su, scf, exx, Some(m));
        let err = max_err(&st.de_deps, &fdv[k]);
        let off = max_offdiag_err(&st.de_deps, &fdv[k]);
        eprintln!(
            "H2 RHF mutant {m:?} ({exx:?}): max|an − FD| = {err:.2e} (off-diagonal {off:.2e})"
        );
        assert!(err > MUTANT_BAR, "{m:?} escaped the FD anchor: {err:e}");
        if diagonal_only.contains(&m) {
            assert!(off < FD_BAR, "{m:?} should be diagonal-only: {off:e}");
        }
    }
}

// ------------------------------------------------------------ H3 UHF (slow)

/// Reference H3 doublet UHF: the staged ewald run and its none stage.
fn h3_uhf_ref(su: &Setup) -> (ScfResult, ScfResult) {
    let r = uhf(su, &uhf_cfg(ExxDiv::Ewald, None));
    let none = r.none_stage.clone().expect("staged none stage");
    (none, r.scf)
}

#[test]
#[ignore = "slow: H3 UHF, 18 strained dense-AFT builds + 36 staged SCF stages; run in release with --ignored"]
fn h3_uhf_stress_matches_fd_all_nine_and_catches_mutants() {
    let bs = pyscf_sto3g_h();
    let su = setup(h3_cell(), &bs);
    let (none, ewald) = h3_uhf_ref(&su);
    let init = mos_of(&none);
    let fdv = fd_strain(&h3_cell(), FD_H, |c| {
        let r = uhf(
            &setup(c.clone(), &bs),
            &uhf_cfg(ExxDiv::Ewald, init.clone()),
        );
        vec![r.none_stage.expect("none stage").energy, r.scf.energy]
    });
    for (k, (scf, exx)) in [(&none, ExxDiv::None), (&ewald, ExxDiv::Ewald)]
        .into_iter()
        .enumerate()
    {
        let st = uhf_stress(&su, scf, exx, None);
        report(&format!("H3 UHF doublet {exx:?}"), &st, &fdv[k]);
        let err = max_err(&st.de_deps, &fdv[k]);
        assert!(err < FD_BAR_OPEN, "{exx:?}: {err:e}");
        assert!(antisym(&st.de_deps) < ANTISYM_BAR);
    }
    for (m, exx) in [
        (StressMutation::NoPulay, ExxDiv::Ewald),
        (StressMutation::FtNoCentres, ExxDiv::Ewald),
        (StressMutation::GUnstrained, ExxDiv::Ewald),
        (StressMutation::NoSrImages, ExxDiv::Ewald),
        (StressMutation::NoMadelung, ExxDiv::Ewald),
        (StressMutation::MadelungS, ExxDiv::Ewald),
    ] {
        let st = uhf_stress(&su, &ewald, exx, Some(m));
        let err = max_err(&st.de_deps, &fdv[1]);
        eprintln!("H3 UHF mutant {m:?}: max|an − FD| = {err:.2e}");
        assert!(err > MUTANT_BAR, "{m:?} escaped: {err:e}");
    }
    // σ(ewald) − σ(none) = −(N/2) dv_M/dε across two SCFs (UHF, α = 1).
    let sn = uhf_stress(&su, &none, ExxDiv::None, None);
    let se = uhf_stress(&su, &ewald, ExxDiv::Ewald, None);
    let err = max_err(&diff(&se.de_deps, &sn.de_deps), &se.parts.madelung);
    eprintln!("H3 UHF: |Δ(ewald − none) − Madelung part| {err:.2e}");
    assert!(err < MADELUNG_ID_BAR, "{err:e}");
}

// ------------------------------------------------ triclinic s+p RHF (slow)

#[test]
#[ignore = "slow: triclinic 4H s+p, 36 strained dense-AFT builds + 72 SCFs; run in release with --ignored"]
fn triclinic_sp_rhf_stress_matches_fd_with_richardson() {
    let bs = sp_basis_h();
    let cell = cell_at(&TRI_MOVED, TRI_A, 1);
    let su = setup(cell.clone(), &bs);
    let energies = |c: &Cell| -> Vec<f64> {
        let s = setup(c.clone(), &bs);
        [ExxDiv::None, ExxDiv::Ewald]
            .iter()
            .map(|&e| rhf(&s, e).energy)
            .collect()
    };
    let fd_h = fd_strain(&cell, FD_H, &energies);
    let fd_h2 = fd_strain(&cell, 0.5 * FD_H, &energies);
    for (k, exx) in [ExxDiv::None, ExxDiv::Ewald].into_iter().enumerate() {
        let st = rhf_stress(&su, &rhf(&su, exx), exx, None);
        let rich = richardson(&fd_h[k], &fd_h2[k]);
        report(&format!("tri s+p RHF {exx:?} (FD h)"), &st, &fd_h[k]);
        let err_h = max_err(&st.de_deps, &fd_h[k]);
        let err_r = max_err(&st.de_deps, &rich);
        eprintln!("tri s+p RHF {exx:?}: FD(h) {err_h:.2e}, Richardson {err_r:.2e}");
        // Prototype: FD(h = 1e-4) 3.5e-8, Richardson ≤ 1.3e-10. The Rust SR
        // attraction derivative carries libint2's p-shell / tight-nucleus
        // floor (forces: 1.85e-7 Ha/Bohr on this cell, bar 5e-7), contracted
        // here with pair vectors of a few Bohr: PREDICTED ≤ ~1e-6, not yet
        // measured. Bar 5e-6 sits ≥ 4 decades under every mutant.
        assert!(err_r < 5e-6, "{exx:?}: Richardson {err_r:e}");
        assert!(err_h < 5e-6, "{exx:?}: FD(h) {err_h:e}");
        // HF stress is symmetric exactly; what is left here is libint2's
        // p-shell SR-derivative floor against the tight Gaussian nucleus, not
        // a missing term (FD agreement above is unaffected). Measured
        // 2026-09-25 on this cell (exxdiv none) vs the gradient-only nucleus
        // exponent: 9.3e-10 (1e8), 9.7e-9 (1e9), 5.4e-8 (1e10, the default),
        // 1.8e-7 (1e11), 5.7e-6 (1e12). The s-only H2/H3 cells reach 1e-16.
        let a = antisym(&st.de_deps);
        assert!(a < TRI_ANTISYM_BAR, "{exx:?}: antisymmetric part {a:e}");
    }
}

// ------------------------------------------------------- H3 UKS atom grid

/// The prototype's anchor grid: (40, 50), SSF, D = 8.
fn grid_cfg() -> PeriodicGridConfig {
    PeriodicGridConfig {
        neighbour_cutoff: Some(8.0),
        ..PeriodicGridConfig::with_size(40, 50)
    }
}

fn uks_cfg(xc: &str, exx: ExxDiv, init: Mos) -> GammaUksConfig {
    GammaUksConfig {
        grid: grid_cfg(),
        exxdiv: exx,
        initial_mos: init,
        ..GammaUksConfig::new(xc)
    }
}

#[test]
#[ignore = "slow: H3 UKS LDA/PBE/PBE0 on the (40, 50) atom grid, 18 strained builds x 3 functionals; run in release with --ignored"]
fn h3_uks_atom_grid_stress_matches_fd_and_catches_xc_mutants() {
    let bs = pyscf_sto3g_h();
    let su = setup(h3_cell(), &bs);
    for (xc, exx) in [
        ("LDA", ExxDiv::None),
        ("PBE", ExxDiv::None),
        ("PBE0", ExxDiv::Ewald),
    ] {
        let run = |s: &Setup, init: Mos| {
            gamma_uks(
                &s.cell,
                &s.prep,
                &s.hc,
                GammaUhfIntegrals::DenseAft(&s.eri),
                &uks_cfg(xc, exx, init),
            )
            .unwrap_or_else(|e| panic!("gamma_uks {xc}: {e}"))
        };
        let r = run(&su, None);
        let init = mos_of(r.none_stage.as_ref().unwrap_or(&r.scf));
        let fdv = fd_strain(&h3_cell(), FD_H_KS, |c| {
            vec![run(&setup(c.clone(), &bs), init.clone()).scf.energy]
        });
        let cfg = uks_cfg(xc, exx, None);
        let stress = |m: Option<StressMutation>| {
            gamma_uks_stress(
                &su.cell,
                &su.prep,
                &hcore_cfg(),
                &su.hc,
                &su.eri,
                &r.scf,
                &cfg,
                &scfg(m),
            )
            .expect("gamma_uks_stress")
        };
        let st = stress(None);
        report(&format!("H3 UKS {xc} {exx:?} atom grid"), &st, &fdv[0]);
        let err = max_err(&st.de_deps, &fdv[0]);
        // Prototype 3.8e-9..3.9e-9 at h = 2.5e-5.
        assert!(err < FD_BAR, "{xc}: {err:e}");
        // The atom grid is NOT rotation invariant (lab-fixed Lebedev
        // offsets): the analytic antisymmetric part is the energy's own
        // (prototype 1.8e-3..2.3e-3 on both) — it must be large AND match FD.
        let an_as = antisym(&st.de_deps);
        let fd_as = antisym(&fdv[0]);
        eprintln!("H3 UKS {xc}: antisymmetric part analytic {an_as:.3e}, FD {fd_as:.3e}");
        assert!(
            an_as > 1e-5,
            "{xc}: atom-grid antisymmetry {an_as:e} suspiciously small"
        );
        assert!((an_as - fd_as).abs() < 2.0 * FD_BAR, "{xc}");
        assert!(st.e_xc.is_some() && st.n_grid_points > 0);
        for m in [
            StressMutation::XcNoWeight,
            StressMutation::XcNoAo,
            StressMutation::XcPointFixed,
        ] {
            let e = max_err(&stress(Some(m)).de_deps, &fdv[0]);
            eprintln!("H3 UKS {xc} mutant {m:?}: max|an − FD| = {e:.2e}");
            assert!(e > MUTANT_BAR, "{xc} {m:?} escaped: {e:e}");
        }
    }
}

// --------------------------------------------------------- RS-GDF H2 (slow)

/// Even-tempered s + Cartesian p aux on H (as `pbc_grad_rsgdf.rs`).
fn et_aux(a0: f64, beta: f64, n: usize) -> BasisSet {
    let mut shells = Vec::new();
    for k in 0..n {
        let a = a0 * beta.powi(k as i32);
        for l in [0, 1] {
            shells.push(Shell {
                l,
                pure: false,
                exponents: vec![a],
                coefficients: vec![1.0],
            });
        }
    }
    let mut m = HashMap::new();
    m.insert(1, shells);
    BasisSet {
        name: "et-aux-H".into(),
        shells: m,
        ecps: HashMap::new(),
    }
}

/// H2 ET l≤1 a0 0.3, β 2.5, 5 exponents: 40 aux.
fn et40() -> BasisSet {
    et_aux(0.3, 2.5, 5)
}

fn gdf_cfg() -> RsGdfConfig {
    RsGdfConfig {
        omega: GDF_OMEGA,
        lindep: DEFAULT_RSGDF_LINDEP,
        exxdiv: ExxDiv::None,
        budget_bytes: Some(AMPLE),
        ..Default::default()
    }
}

fn gdf_rhf(
    cell: &Cell,
    prep: &PreparedBasis,
    hc: &PeriodicHcore,
    gdf: &RsGdf,
    exx: ExxDiv,
) -> ScfResult {
    let g = gdf.clone().with_exxdiv(cell, exx).expect("with_exxdiv");
    gamma_rhf_jk(
        cell,
        prep,
        hc,
        Box::new(g.j_builder()),
        Box::new(g.k_builder()),
    )
}

#[test]
#[ignore = "slow: H2 ET-40 RS-GDF, 18 strained RS-GDF builds + 36 SCFs; run in release with --ignored"]
fn h2_rsgdf_rhf_stress_matches_fd_and_catches_fit_mutants() {
    let bs = pyscf_sto3g_h();
    let aux_bs = et40();
    let cell = h2_cell_g();
    let prep = prep_for(&cell, &bs);
    let hc = periodic_hcore(&cell, &prep, &hcore_cfg()).unwrap();
    let aux = PreparedBasis::new(cell.mol(), &aux_bs).unwrap();
    let gdf = RsGdf::build_for_gradient(&cell, &prep, &aux, &hc.s, &gdf_cfg()).unwrap();
    let fdv = fd_strain(&cell, FD_H, |c| {
        let p = prep_for(c, &bs);
        let h = periodic_hcore(c, &p, &hcore_cfg()).unwrap();
        let a = PreparedBasis::new(c.mol(), &aux_bs).unwrap();
        let g = RsGdf::build(c, &p, &a, &h.s, &gdf_cfg()).unwrap();
        assert_eq!(
            g.stats().n_dropped,
            gdf.stats().n_dropped,
            "the metric cut changed under strain (E(ε) discontinuous)"
        );
        [ExxDiv::None, ExxDiv::Ewald]
            .iter()
            .map(|&e| gdf_rhf(c, &p, &h, &g, e).energy)
            .collect()
    });
    let src = RsGdfGradSource {
        gdf: &gdf,
        aux: &aux,
        aux_jac: None,
    };
    let stress = |scf: &ScfResult, exx: ExxDiv, m: Option<StressMutation>| {
        gamma_rhf_stress_rsgdf(&cell, &prep, &hcore_cfg(), &hc, &src, scf, exx, &scfg(m))
            .expect("gamma_rhf_stress_rsgdf")
    };
    let scfs = [
        gdf_rhf(&cell, &prep, &hc, &gdf, ExxDiv::None),
        gdf_rhf(&cell, &prep, &hc, &gdf, ExxDiv::Ewald),
    ];
    for (k, exx) in [ExxDiv::None, ExxDiv::Ewald].into_iter().enumerate() {
        let st = stress(&scfs[k], exx, None);
        report(&format!("H2 ET-40 RS-GDF RHF {exx:?}"), &st, &fdv[k]);
        let err = max_err(&st.de_deps, &fdv[k]);
        // Prototype 6.7e-9 / 8.5e-9.
        assert!(err < FD_BAR, "{exx:?}: {err:e}");
        assert!(antisym(&st.de_deps) < ANTISYM_BAR);
        assert!(st.parts.eri.iter().flatten().all(|v| *v == 0.0));
        let j2g0 = trace(&st.parts.fit_j2_g0_volume).abs();
        eprintln!("  J2 G=0 strain-only term: tr = {j2g0:.3e} (prototype ~3e-2 per diagonal)");
        assert!(
            j2g0 > 1e-3,
            "the J2 G = 0 volume term must be live: {j2g0:e}"
        );
    }
    for m in [
        StressMutation::NoMetric,
        StressMutation::NoAuxFt3,
        StressMutation::NoJ2G0Vol,
        StressMutation::NoJ3G0Vol,
        StressMutation::NoG0,
        StressMutation::NoSrImages,
        StressMutation::GUnstrained,
        StressMutation::FtNoCentres,
    ] {
        let err = max_err(&stress(&scfs[0], ExxDiv::None, Some(m)).de_deps, &fdv[0]);
        eprintln!("H2 RS-GDF mutant {m:?}: max|an − FD| = {err:.2e}");
        assert!(err > MUTANT_BAR, "{m:?} escaped: {err:e}");
    }
    // Nearly blind by stationarity (prototype 7.7e-6): reported only.
    let err = max_err(
        &stress(&scfs[0], ExxDiv::None, Some(StressMutation::NoAuxFt)).de_deps,
        &fdv[0],
    );
    eprintln!("H2 RS-GDF mutant NoAuxFt (J3 AND J2, fit-error sized): max|an − FD| = {err:.2e}");
}
