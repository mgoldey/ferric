//! Gamma-point forces and stress with HEAVY atoms (Li, C, O): the first
//! FD anchors of `ferric_pbc::grad` / `ferric_pbc::stress` outside the H-only
//! cells of `pbc_grad*.rs` / `pbc_stress.rs`.
//!
//! What is new here (none of it was exercised by an FD anchor before):
//! * Z > 1 nuclei in the SR Gaussian-nucleus attraction derivative, the
//!   Ewald `E_nn` gradient/strain and `h`'s `c0 Z_tot D` term.
//! * Shared-exponent shells: STO-3G / 6-31G `SP` shells (ferric splits them
//!   into an s and a p shell on the SAME primitives) and cc-pVDZ general
//!   contractions (Li/C/O 1s and 2s on one set of 9 primitives, loaded as
//!   separate shells; cc-pVDZ also carries all-zero-coefficient primitives).
//! * Pure d shells in the ORBITAL basis (cc-pVDZ): pair-FT derivative /
//!   strain through `ferric_cart2sph`, libint2 1e/3-centre derivative blocks
//!   with d, the XC AO Hessians with d.
//! * Real auxiliary sets with l > 1 on heavy atoms: cc-pvdz-ri (up to f on
//!   Li/O, d on H) and def2-universal-jkfit (up to g on C/O — the l = 4 aux
//!   the real-size benchmark needs), in the RS-GDF force and stress.
//!
//! EXACTNESS ANCHOR (every test): analytic vs central FD of ferric's OWN
//! energy with the SAME truncations (the same `PeriodicHcoreConfig`,
//! `RsGdfConfig` / dense-AFT precision, grid), every displaced / strained
//! geometry rebuilt from scratch, its SCF SEEDED from the reference solution
//! (RHF/RKS: the reference density; UHF/UKS: the reference none-stage MOs) so
//! every point lands on the same state. Guards on the FD itself: the orbital
//! canonical cut drops nothing and λ_min(S) has margin (Iteration 15: LiH is
//! the lindep worst case), and the RS-GDF aux-metric drop count is the SAME
//! at every displaced geometry (a count change makes E(R) discontinuous).
//!
//! Bars (START from the known floors, FINDINGS Iterations 16-20):
//! * s-only H cells reach ≤ 1e-8 (bar 1e-7). With p shells the SR attraction
//!   derivative carries libint2's tight-Gaussian-nucleus floor, 1.85e-7
//!   Ha/Bohr on the triclinic 4H s+p cell at the default gradient nucleus
//!   exponent 1e10 (bar 5e-7). [`P_BAR`] reuses 5e-7 for the Li p-shell
//!   cases; that floor was measured with Z = 1 and may scale with Z — if a
//!   p-only case lands between 5e-7 and a few e-6, read the printed nucleus-
//!   exponent scan before calling it a defect.
//! * d shells: UNKNOWN. [`D_BAR`] = 1e-5 is PROVISIONAL UNTIL MEASURED;
//!   tighten after the first run (every test prints its residual, and the
//!   HF force tests print the residual at gradient nucleus exponents
//!   1e8/1e9/1e11 too, which separates the libint floor — it moves with the
//!   exponent — from a missing term, which does not).
//! * KS on the atom grid: FD h = 5e-5 (the hard neighbour mask; Iteration 17).
//! * Stress: all nine components, Richardson `(4 FD(h/2) − FD(h))/3` (the
//!   p-shell rule of Iteration 19); the antisymmetric part of an HF stress is
//!   the libint floor (5.4e-8 on the triclinic s+p cell) and is reported.
//!
//! Mutants (the anchor CAN fail on heavy atoms): `GradMutation::FitNoMetric`
//! and `WSign` on CO/cc-pVDZ + jkfit, `WSign` on dense-AFT LiH, `XcNoAo` on
//! CO RKS PBE, `FitNoMetric` on OH UHF, `StressMutation::NoMetric` /
//! `NoPulay` on the CO and LiH stress. H-cell magnitudes are 1e-3..1e-1; the
//! bars here are 1e-4 (fit) and 1e-3 (W sign).
//!
//! Dense-AFT is used ONLY for LiH/STO-3G: its G sphere is set by the
//! TIGHTEST orbital exponent (`gcut = 2 √(2 α_max ln 1e14)`), so Li 6-31G
//! (α_max 642) or cc-pVDZ (1469) or C/O STO-3G (≥ 71) would need 1e7..1e9 G
//! vectors. Everything else is RS-GDF.
//!
//! Everything is `#[ignore]`: an RS-GDF build with all-electron Li/C/O cores
//! is not a debug-profile test. Suggested first run:
//! `heavy_atom_shells_and_aux_are_accepted_by_forces_and_stress` (no FD; it
//! surfaces any libint2 refusal or abort on a shell/aux combination fast).

mod common;

use common::*;
use ferric_core::basis::{self, BasisSet, Shell};
use ferric_core::mol::{Atom, Molecule};
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_pbc::dense_aft::{
    DenseAftEri, ExxDiv, DEFAULT_DENSE_AFT_MAX_BYTES, DEFAULT_DENSE_AFT_PRECISION,
};
use ferric_pbc::dft::{gamma_rks, gamma_uks, GammaRksConfig, GammaUksConfig, PeriodicGridConfig};
use ferric_pbc::grad::{
    gamma_rhf_gradient_rsgdf, gamma_rhf_gradient_with, gamma_rks_gradient_rsgdf,
    gamma_uhf_gradient_rsgdf, gamma_uks_gradient_rsgdf, GammaGradConfig, GammaGradient,
    GradMutation, RsGdfGradSource,
};
use ferric_pbc::hcore::{periodic_hcore, PeriodicHcore, PeriodicHcoreConfig};
use ferric_pbc::lattice::Cell;
use ferric_pbc::lindep::{exp_to_discard, LindepReport};
use ferric_pbc::rsgdf::{RsGdf, RsGdfConfig, DEFAULT_RSGDF_LINDEP};
use ferric_pbc::stress::{
    gamma_rhf_stress_rsgdf, gamma_uhf_stress_rsgdf, GammaStress, GammaStressConfig, StressMutation,
};
use ferric_pbc::uhf::{gamma_uhf, GammaUhfConfig, GammaUhfIntegrals, GammaUhfResult};
use ferric_scf::fock::{JBuilder, KBuilder};
use ferric_scf::result::ScfResult;
use ferric_scf::rhf::{solve_rhf_injected, PeriodicInjection, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ndarray::Array2;

type Mat3 = [[f64; 3]; 3];

// ---------------------------------------------------------------- settings

/// ω of the nuclear-attraction split (as every other force/stress test).
const OMEGA: f64 = 0.8;
const HCORE_PRECISION: f64 = 1e-14;
const GDF_OMEGA: f64 = 1.0;
const AMPLE: usize = 1 << 31;
const FD_H: f64 = 1e-4;
/// Atom-grid KS step (Iteration 17: the hard neighbour mask needs ≤ 5e-5).
const FD_H_KS: f64 = 5e-5;

/// Li p shells (STO-3G / 6-31G): the H-derived p-shell floor bar (module
/// doc); PROVISIONAL for Z = 3.
const P_BAR: f64 = 5e-7;
/// d shells in the orbital basis. Provisional until measured; tighten after
/// the first run.
const D_BAR: f64 = 1e-5;
/// KS PBE with d shells on the atom grid, h = 5e-5. Provisional until
/// measured; tighten after the first run.
const KS_D_BAR: f64 = 1e-5;
/// Stress, p shells: the triclinic s+p bar (Richardson), Iteration 19.
const STRESS_P_BAR: f64 = 5e-6;
/// Stress antisymmetric part, p shells: the triclinic s+p bar (measured
/// 5.4e-8 there at Z = 1). Provisional for Z = 3.
const STRESS_P_ANTISYM_BAR: f64 = 5e-7;
/// Stress with d shells (FD and antisymmetric part). Provisional until
/// measured; tighten after the first run.
const STRESS_D_BAR: f64 = 1e-5;
const STRESS_D_ANTISYM_BAR: f64 = 1e-5;
/// ΣF: H cells reach 1e-13 (bar 1e-10). Provisional for heavy atoms (the
/// SR-derivative floor is not translation-invariant term by term).
const NET_FORCE_BAR: f64 = 1e-8;
/// F(ewald) vs F(none) from two independent SCFs (H cells 6.6e-12, bar
/// 1e-10); SCF-convergence limited, so looser with core orbitals.
const EWALD_NONE_BAR: f64 = 1e-8;
const FIT_MUTANT_BAR: f64 = 1e-4;
const MUTANT_BAR: f64 = 1e-3;
/// Orbital canonical cut of the Gamma SCF (`ferric_scf` LINDEP_THRESH).
const ORBITAL_LINDEP: f64 = 1e-6;
/// λ_min(S) must clear the cut by this much so ±h cannot move a direction
/// across it.
const S_MARGIN: f64 = 1e-5;
/// Gradient-only nucleus exponents printed (not asserted) next to the
/// default 1e10: the libint floor moves with it, a missing term does not.
const NUC_SCAN: [f64; 3] = [1e8, 1e9, 1e11];

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

// ----------------------------------------------------------------- systems

/// One LiH (Bohr), bond 3.0 along (0.36, 0.48, 0.8): off-axis so every
/// Cartesian component is non-trivial. Cubic a = 8.
const LIH_POS: [[f64; 3]; 2] = [[0.4, 0.3, 0.2], [1.48, 1.74, 2.6]];
const LIH_A: f64 = 8.0;
const LIH_COMPS: [(usize, usize); 3] = [(0, 0), (0, 2), (1, 1)];
/// One CO (Bohr), bond 2.13 along (0.48, 0.36, 0.8). Cubic a = 7.
const CO_POS: [[f64; 3]; 2] = [[0.3, 0.2, 0.1], [1.32, 0.97, 1.81]];
const CO_A: f64 = 7.0;
const CO_COMPS: [(usize, usize); 3] = [(0, 0), (1, 1), (1, 2)];
/// One OH radical (Bohr), bond 1.83 along (0.36, 0.48, 0.8). Cubic a = 7.
const OH_POS: [[f64; 3]; 2] = [[0.3, 0.2, 0.1], [0.96, 1.08, 1.56]];
const OH_A: f64 = 7.0;
const OH_COMPS: [(usize, usize); 3] = [(0, 1), (1, 0), (1, 2)];
/// Li cc-pVDZ `exp_to_discard` (applied to Li only; H's smallest cc-pVDZ
/// exponent is 0.122): removes the single-primitive diffuse s (0.02805) and
/// p (0.02403) shells and the 0.02805 tail of the 2s contraction. FINDINGS
/// "Real-size benchmark plan": LiH/cc-pVDZ reaches λ_min(S) ≈ 1e-14.
const LI_CCPVDZ_EXP_TO_DISCARD: f64 = 0.05;

#[derive(Clone)]
struct System {
    species: Vec<(&'static str, i32)>,
    pos: Vec<[f64; 3]>,
    lattice: Mat3,
    mult: usize,
}

impl System {
    fn new(species: &[(&'static str, i32)], pos: &[[f64; 3]], a: f64, mult: usize) -> Self {
        Self {
            species: species.to_vec(),
            pos: pos.to_vec(),
            lattice: cubic(a),
            mult,
        }
    }

    fn cell_with(&self, pos: &[[f64; 3]]) -> Cell {
        let atoms = self
            .species
            .iter()
            .zip(pos)
            .map(|(&(sym, z), r)| Atom {
                symbol: sym.into(),
                z,
                x: r[0],
                y: r[1],
                zpos: r[2],
                ghost: false,
                n_core_ecp: 0,
            })
            .collect();
        let mol = Molecule {
            atoms,
            charge: 0,
            multiplicity: self.mult,
        };
        Cell::new(mol, self.lattice).expect("cell")
    }

    fn cell(&self) -> Cell {
        self.cell_with(&self.pos)
    }

    fn moved(&self, a: usize, x: usize, h: f64) -> Cell {
        let mut p = self.pos.clone();
        p[a][x] += h;
        self.cell_with(&p)
    }
}

fn lih() -> System {
    System::new(&[("Li", 3), ("H", 1)], &LIH_POS, LIH_A, 1)
}

fn co() -> System {
    System::new(&[("C", 6), ("O", 8)], &CO_POS, CO_A, 1)
}

fn oh() -> System {
    System::new(&[("O", 8), ("H", 1)], &OH_POS, OH_A, 2)
}

// ------------------------------------------------------------------ bases

fn bundled(name: &str) -> BasisSet {
    basis::bundled(name).unwrap_or_else(|e| panic!("bundled {name}: {e}"))
}

/// The same functions without their all-zero-coefficient primitives
/// (cc-pVDZ's uncontracted diffuse s/p are stored as a zero column over all
/// 9 / 4 primitives but one). `exp_to_discard` keeps primitives BY EXPONENT,
/// so without this the diffuse column would survive as a zero-norm shell.
fn strip_zero_primitives(bs: &BasisSet) -> BasisSet {
    let mut out = bs.clone();
    for shells in out.shells.values_mut() {
        for sh in shells.iter_mut() {
            let keep: Vec<usize> = (0..sh.coefficients.len())
                .filter(|&p| sh.coefficients[p] != 0.0)
                .collect();
            *sh = Shell {
                l: sh.l,
                pure: sh.pure,
                exponents: keep.iter().map(|&p| sh.exponents[p]).collect(),
                coefficients: keep.iter().map(|&p| sh.coefficients[p]).collect(),
            };
        }
    }
    out
}

/// cc-pVDZ with Li's diffuse primitives discarded (module doc).
fn lih_ccpvdz() -> BasisSet {
    let bs = strip_zero_primitives(&bundled("cc-pvdz"));
    let (out, report) =
        exp_to_discard(&bs, LI_CCPVDZ_EXP_TO_DISCARD, &[3]).expect("exp_to_discard Li");
    eprintln!(
        "Li cc-pVDZ exp_to_discard {LI_CCPVDZ_EXP_TO_DISCARD}: {} shells removed, {} primitives \
         removed",
        report.n_shells_removed(),
        report.n_primitives_removed()
    );
    assert!(
        out.shells[&3].iter().any(|s| s.l == 2),
        "the Li d shell must survive"
    );
    assert!(
        out.shells[&3].iter().filter(|s| s.l == 0).count() >= 2,
        "two Li s contractions on shared primitives must survive"
    );
    out
}

// -------------------------------------------------------------- configs

fn hcore_cfg() -> PeriodicHcoreConfig {
    PeriodicHcoreConfig {
        precision: HCORE_PRECISION,
        ..PeriodicHcoreConfig::with_omega(OMEGA)
    }
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

/// (50, 110) SSF with a FIXED neighbour cutoff: `None` would take
/// `max(default, covering_radius_bound)`, which moves with the atoms. 10
/// Bohr clears the covering radius of a 7-8 Bohr cube with two atoms.
fn grid_cfg() -> PeriodicGridConfig {
    PeriodicGridConfig {
        neighbour_cutoff: Some(10.0),
        ..PeriodicGridConfig::with_size(50, 110)
    }
}

fn gcfg(mutation: Option<GradMutation>, nucleus_exponent: Option<f64>) -> GammaGradConfig {
    GammaGradConfig {
        mutation,
        budget_bytes: Some(AMPLE),
        nucleus_exponent,
    }
}

fn scfg(mutation: Option<StressMutation>, nucleus_exponent: Option<f64>) -> GammaStressConfig {
    GammaStressConfig {
        mutation,
        budget_bytes: Some(AMPLE),
        nucleus_exponent,
    }
}

fn rks_cfg(xc: &str, seed: Option<&Array2<f64>>) -> GammaRksConfig {
    let mut c = GammaRksConfig {
        grid: grid_cfg(),
        exxdiv: ExxDiv::None,
        ..GammaRksConfig::new(xc)
    };
    c.scf.init_guess_density = seed.cloned();
    c
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

fn uks_cfg(xc: &str, init: Mos) -> GammaUksConfig {
    GammaUksConfig {
        grid: grid_cfg(),
        exxdiv: ExxDiv::None,
        initial_mos: init,
        ..GammaUksConfig::new(xc)
    }
}

// ------------------------------------------------------------ builders

/// λ_min(S) and the canonical-cut count at the SCF's threshold; asserts
/// nothing is cut and the margin (module doc).
fn check_overlap(tag: &str, s: &Array2<f64>, verbose: bool) -> f64 {
    let r = LindepReport::from_real_overlap(s, ORBITAL_LINDEP).expect("S spectrum");
    let k = &r.per_k[0];
    if verbose {
        eprintln!(
            "{tag}: nao {}, λ_min(S) = {:.3e} (cut {ORBITAL_LINDEP:e}, dropped {})",
            k.nao,
            k.min_eig,
            r.total_dropped()
        );
    }
    assert_eq!(
        r.total_dropped(),
        0,
        "{tag}: the orbital canonical cut is active (λ_min {:e})",
        k.min_eig
    );
    assert!(
        k.min_eig > S_MARGIN,
        "{tag}: λ_min(S) = {:e} too close to the cut",
        k.min_eig
    );
    k.min_eig
}

/// The lattice `(prep, hc)` of `cell` in `bs`, with the overlap guard.
fn hcore_of(
    tag: &str,
    cell: &Cell,
    bs: &BasisSet,
    verbose: bool,
) -> (PreparedBasis, PeriodicHcore) {
    let prep = prep_for(cell, bs);
    let hc = periodic_hcore(cell, &prep, &hcore_cfg()).expect("periodic_hcore");
    check_overlap(tag, &hc.s, verbose);
    (prep, hc)
}

/// RS-GDF (`for_gradient` = `build_for_gradient`, else the energy-only
/// `build`, bitwise the same B).
fn gdf_of(
    cell: &Cell,
    prep: &PreparedBasis,
    hc: &PeriodicHcore,
    aux_bs: &BasisSet,
    for_gradient: bool,
) -> (PreparedBasis, RsGdf) {
    let aux = PreparedBasis::new(cell.mol(), aux_bs).expect("aux prep");
    let gdf = if for_gradient {
        RsGdf::build_for_gradient(cell, prep, &aux, &hc.s, &gdf_cfg())
    } else {
        RsGdf::build(cell, prep, &aux, &hc.s, &gdf_cfg())
    }
    .unwrap_or_else(|e| panic!("RsGdf with {}: {e}", aux_bs.name));
    (aux, gdf)
}

struct Setup {
    cell: Cell,
    prep: PreparedBasis,
    hc: PeriodicHcore,
    aux: PreparedBasis,
    gdf: RsGdf,
}

fn setup(tag: &str, cell: Cell, bs: &BasisSet, aux_bs: &BasisSet) -> Setup {
    let (prep, hc) = hcore_of(tag, &cell, bs, true);
    let (aux, gdf) = gdf_of(&cell, &prep, &hc, aux_bs, true);
    let st = gdf.stats();
    eprintln!(
        "{tag}: naux {} (kept {}, dropped {}; metric eig {:.2e}..{:.2e}), SR3 {}, half-G {}",
        st.naux,
        st.naux_kept,
        st.n_dropped,
        st.metric_eig_min,
        st.metric_eig_max,
        st.n_sr3_triplets,
        st.n_g_half
    );
    Setup {
        cell,
        prep,
        hc,
        aux,
        gdf,
    }
}

fn src(su: &Setup) -> RsGdfGradSource<'_> {
    RsGdfGradSource {
        gdf: &su.gdf,
        aux: &su.aux,
        aux_jac: None,
    }
}

/// Gamma RHF on injected J/K, optionally seeded with a density.
fn rhf_injected<'a>(
    cell: &Cell,
    prep: &PreparedBasis,
    hc: &PeriodicHcore,
    j: Box<dyn JBuilder + 'a>,
    k: Box<dyn KBuilder + 'a>,
    seed: Option<&Array2<f64>>,
) -> ScfResult {
    let ctx = ParallelContext::default();
    let op = Operator::coulomb();
    // Never read on the injected path; required by the signature.
    let bounds = SchwarzBounds::compute(op, prep).expect("schwarz");
    let cfg = RhfConfig {
        init_guess_density: seed.cloned(),
        ..gamma_config()
    };
    let inj = PeriodicInjection {
        s: hc.s.clone(),
        h: hc.h.clone(),
        vnn: hc.enn,
        j,
        k,
        xc: None,
    };
    let r = solve_rhf_injected(&ctx, cell.mol(), prep, op, &bounds, &cfg, inj)
        .expect("gamma-point RHF");
    assert!(r.converged, "gamma-point RHF did not converge");
    r
}

fn rhf_gdf(
    cell: &Cell,
    prep: &PreparedBasis,
    hc: &PeriodicHcore,
    gdf: &RsGdf,
    exx: ExxDiv,
    seed: Option<&Array2<f64>>,
) -> ScfResult {
    let g = gdf.clone().with_exxdiv(cell, exx).expect("with_exxdiv");
    rhf_injected(
        cell,
        prep,
        hc,
        Box::new(g.j_builder()),
        Box::new(g.k_builder()),
        seed,
    )
}

fn rhf_dense(
    cell: &Cell,
    prep: &PreparedBasis,
    hc: &PeriodicHcore,
    eri: &DenseAftEri,
    exx: ExxDiv,
    seed: Option<&Array2<f64>>,
) -> ScfResult {
    let e = eri.clone().with_exxdiv(cell, exx).expect("with_exxdiv");
    rhf_injected(
        cell,
        prep,
        hc,
        Box::new(e.j_builder()),
        Box::new(e.k_builder()),
        seed,
    )
}

fn uhf_on(
    su_cell: &Cell,
    prep: &PreparedBasis,
    hc: &PeriodicHcore,
    gdf: &RsGdf,
    cfg: &GammaUhfConfig,
) -> GammaUhfResult {
    let r = gamma_uhf(su_cell, prep, hc, GammaUhfIntegrals::RsGdf(gdf), cfg).expect("gamma_uhf");
    assert!(r.scf.converged, "gamma_uhf did not converge");
    r
}

// ------------------------------------------------------------ FD helpers

type Fd = Vec<((usize, usize), Vec<f64>)>;

/// Central FD `(E(+h) − E(−h))/2h` of every energy `energies` returns.
fn fd<F>(sys: &System, comps: &[(usize, usize)], h: f64, energies: F) -> Fd
where
    F: Fn(&Cell) -> Vec<f64>,
{
    comps
        .iter()
        .map(|&(a, x)| {
            let ep = energies(&sys.moved(a, x, h));
            let em = energies(&sys.moved(a, x, -h));
            let d = ep
                .iter()
                .zip(&em)
                .map(|(p, m)| (p - m) / (2.0 * h))
                .collect();
            ((a, x), d)
        })
        .collect()
}

fn max_fd_err(g: &Array2<f64>, fdv: &Fd, k: usize) -> f64 {
    fdv.iter()
        .map(|((a, x), v)| {
            let d = (g[(*a, *x)] - v[k]).abs();
            assert!(d.is_finite(), "non-finite force difference");
            d
        })
        .fold(0.0_f64, f64::max)
}

fn amax(a: &Array2<f64>) -> f64 {
    a.iter().fold(0.0_f64, |m, v| m.max(v.abs()))
}

fn max_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

fn report_force(tag: &str, g: &GammaGradient, fdv: &Fd, k: usize) -> f64 {
    let err = max_fd_err(&g.grad, fdv, k);
    let per: Vec<String> = fdv
        .iter()
        .map(|((a, x), v)| {
            format!(
                "({a},{x}) an {:+.10e} fd {:+.10e} Δ {:+.2e}",
                g.grad[(*a, *x)],
                v[k],
                g.grad[(*a, *x)] - v[k]
            )
        })
        .collect();
    let fit = g
        .fit
        .as_ref()
        .map(|f| {
            format!(
                "; fit naux {} dropped {} (s_kept_min {:.2e}), |aux| {:.2e} |metric| {:.2e} |g0| {:.2e}",
                f.naux,
                f.n_dropped,
                f.s_kept_min,
                amax(&g.parts.fit_aux_sr) + amax(&g.parts.fit_aux_lr),
                amax(&g.parts.fit_metric_sr) + amax(&g.parts.fit_metric_lr),
                amax(&g.parts.fit_g0)
            )
        })
        .unwrap_or_default();
    eprintln!(
        "{tag}: max|analytic − FD| = {err:.2e}, |ΣF| {:.1e}, comm {:.1e}, max|F| {:.3e}, \
         |vsr_basis| {:.2e}{fit}\n  {}",
        g.net_force,
        g.commutator,
        amax(&g.grad),
        amax(&g.parts.vsr_basis),
        per.join("\n  ")
    );
    err
}

/// Central FD `dE/dε_ij`, all nine, of every energy `energies` returns
/// (strained cells from `cell`, index sets frozen there).
fn fd_strain<F>(cell: &Cell, h: f64, energies: F) -> Vec<Mat3>
where
    F: Fn(&Cell) -> Vec<f64>,
{
    let mut out: Vec<Mat3> = Vec::new();
    for (i, j) in ALL9 {
        let mut e = [[0.0; 3]; 3];
        e[i][j] = h;
        let ep = energies(&cell.strained(&e).expect("strained"));
        e[i][j] = -h;
        let em = energies(&cell.strained(&e).expect("strained"));
        if out.is_empty() {
            out = vec![[[0.0; 3]; 3]; ep.len()];
        }
        for k in 0..ep.len() {
            out[k][i][j] = (ep[k] - em[k]) / (2.0 * h);
        }
    }
    out
}

fn richardson(fd_h: &Mat3, fd_h2: &Mat3) -> Mat3 {
    let mut o = [[0.0; 3]; 3];
    for a in 0..3 {
        for b in 0..3 {
            o[a][b] = (4.0 * fd_h2[a][b] - fd_h[a][b]) / 3.0;
        }
    }
    o
}

fn max_err3(a: &Mat3, b: &Mat3) -> f64 {
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

fn antisym(a: &Mat3) -> f64 {
    let mut m = 0.0_f64;
    for i in 0..3 {
        for j in 0..3 {
            m = m.max((a[i][j] - a[j][i]).abs());
        }
    }
    m
}

fn fmt3(a: &Mat3) -> String {
    a.iter()
        .map(|r| format!("  [{:+.10e} {:+.10e} {:+.10e}]", r[0], r[1], r[2]))
        .collect::<Vec<_>>()
        .join("\n")
}

fn diff3(a: &Mat3, b: &Mat3) -> Mat3 {
    let mut o = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            o[i][j] = a[i][j] - b[i][j];
        }
    }
    o
}

// ================================================================ smoke

/// No FD: every heavy-atom (orbital, aux) combination used below goes
/// through the RS-GDF build, the SCF, one force and one stress evaluation.
/// A libint2 refusal is a typed error (FFI catches C++ exceptions); a debug
/// libint2 assert would abort the process — either way this is the fastest
/// place to see it.
#[test]
#[ignore = "slow: 3 all-electron heavy-atom RS-GDF builds (jkfit g on C/O), no FD; run with --release -- --ignored"]
fn heavy_atom_shells_and_aux_are_accepted_by_forces_and_stress() {
    for (tag, sys, bs, aux_bs) in [
        (
            "LiH cc-pVDZ(Li discard) / cc-pvdz-ri",
            lih(),
            lih_ccpvdz(),
            bundled("cc-pvdz-ri"),
        ),
        (
            "CO cc-pVDZ / def2-universal-jkfit",
            co(),
            bundled("cc-pvdz"),
            bundled("def2-universal-jkfit"),
        ),
        (
            "OH cc-pVDZ / cc-pvdz-ri (UHF doublet)",
            oh(),
            bundled("cc-pvdz"),
            bundled("cc-pvdz-ri"),
        ),
    ] {
        let t0 = std::time::Instant::now();
        let su = setup(tag, sys.cell(), &bs, &aux_bs);
        let t_build = t0.elapsed().as_secs_f64();
        let (g, st) = if sys.mult == 1 {
            let scf = rhf_gdf(&su.cell, &su.prep, &su.hc, &su.gdf, ExxDiv::Ewald, None);
            let g = gamma_rhf_gradient_rsgdf(
                &su.cell,
                &su.prep,
                &hcore_cfg(),
                &su.hc,
                &src(&su),
                &scf,
                ExxDiv::Ewald,
                &gcfg(None, None),
            )
            .expect("gamma_rhf_gradient_rsgdf");
            let st = gamma_rhf_stress_rsgdf(
                &su.cell,
                &su.prep,
                &hcore_cfg(),
                &su.hc,
                &src(&su),
                &scf,
                ExxDiv::Ewald,
                &scfg(None, None),
            )
            .expect("gamma_rhf_stress_rsgdf");
            (g, st)
        } else {
            let r = uhf_on(
                &su.cell,
                &su.prep,
                &su.hc,
                &su.gdf,
                &uhf_cfg(ExxDiv::Ewald, None),
            );
            eprintln!(
                "{tag}: UHF ⟨S²⟩ {:.6}, gaps α {:?} β {:?} (v_M {:.4})",
                r.s2, r.gaps.gap_alpha, r.gaps.gap_beta, r.madelung
            );
            let g = gamma_uhf_gradient_rsgdf(
                &su.cell,
                &su.prep,
                &hcore_cfg(),
                &su.hc,
                &src(&su),
                &r.scf,
                ExxDiv::Ewald,
                &gcfg(None, None),
            )
            .expect("gamma_uhf_gradient_rsgdf");
            let st = gamma_uhf_stress_rsgdf(
                &su.cell,
                &su.prep,
                &hcore_cfg(),
                &su.hc,
                &src(&su),
                &r.scf,
                ExxDiv::Ewald,
                &scfg(None, None),
            )
            .expect("gamma_uhf_stress_rsgdf");
            (g, st)
        };
        eprintln!(
            "{tag}: build {t_build:.1} s, total {:.1} s; |ΣF| {:.2e}, max|F| {:.3e}, \
             stress |an − anᵀ| {:.2e}, P {:+.6e}\n{:.8}\n dE/dε:\n{}",
            t0.elapsed().as_secs_f64(),
            g.net_force,
            amax(&g.grad),
            antisym(&st.de_deps),
            st.pressure(),
            g.grad,
            fmt3(&st.de_deps)
        );
        assert!(g.grad.iter().all(|v| v.is_finite()));
        assert!(st.de_deps.iter().flatten().all(|v| v.is_finite()));
        assert!(amax(&g.grad) > 1e-3, "{tag}: vacuous force");
        assert!(g.net_force < NET_FORCE_BAR, "{tag}: ΣF {:e}", g.net_force);
    }
}

// ================================================================ LiH

/// Dense-AFT (pure-AFT ERI, no fit) LiH/STO-3G: SP shells on Li, Z = 3.
/// The only dense case (module doc: the G sphere follows the tightest
/// exponent, 16.1 for STO-3G Li).
#[test]
#[ignore = "slow: LiH STO-3G dense AFT (~1e6 G vectors per build), 6 displaced builds + 12 SCFs; run with --release -- --ignored"]
fn lih_sto3g_dense_aft_rhf_forces_match_fd_and_catch_wsign() {
    let sys = lih();
    let bs = bundled("sto-3g");
    let build = |c: &Cell, verbose: bool| {
        let (prep, hc) = hcore_of("LiH STO-3G dense", c, &bs, verbose);
        let eri = DenseAftEri::build(
            c,
            &prep,
            &hc.s,
            ExxDiv::None,
            DEFAULT_DENSE_AFT_PRECISION,
            DEFAULT_DENSE_AFT_MAX_BYTES,
        )
        .expect("dense AFT");
        (prep, hc, eri)
    };
    let cell = sys.cell();
    let t0 = std::time::Instant::now();
    let (prep, hc, eri) = build(&cell, true);
    eprintln!(
        "LiH STO-3G dense: build {:.1} s, half-G {}",
        t0.elapsed().as_secs_f64(),
        eri.n_g_half()
    );
    let scf_n = rhf_dense(&cell, &prep, &hc, &eri, ExxDiv::None, None);
    let scf_e = rhf_dense(
        &cell,
        &prep,
        &hc,
        &eri,
        ExxDiv::Ewald,
        Some(&scf_n.density_total),
    );
    let seed = scf_n.density_total.clone();
    let fdv = fd(&sys, &LIH_COMPS, FD_H, |c| {
        let (p, h, e) = build(c, false);
        [ExxDiv::None, ExxDiv::Ewald]
            .iter()
            .map(|&x| rhf_dense(c, &p, &h, &e, x, Some(&seed)).energy)
            .collect()
    });
    let grad = |scf: &ScfResult, exx: ExxDiv, m: Option<GradMutation>, nuc: Option<f64>| {
        gamma_rhf_gradient_with(
            &cell,
            &prep,
            &hcore_cfg(),
            &hc,
            &eri,
            scf,
            exx,
            &gcfg(m, nuc),
        )
        .expect("gamma_rhf_gradient_with")
    };
    let mut gs = Vec::new();
    for (k, (scf, exx)) in [(&scf_n, ExxDiv::None), (&scf_e, ExxDiv::Ewald)]
        .into_iter()
        .enumerate()
    {
        let g = grad(scf, exx, None, None);
        let err = report_force(&format!("LiH STO-3G dense RHF {exx:?}"), &g, &fdv, k);
        assert!(amax(&g.grad) > 1e-3, "vacuous comparison");
        assert!(err < P_BAR, "{exx:?}: {err:e}");
        assert!(g.net_force < NET_FORCE_BAR, "ΣF {:e}", g.net_force);
        gs.push(g);
    }
    for nuc in NUC_SCAN {
        let e = max_fd_err(&grad(&scf_n, ExxDiv::None, None, Some(nuc)).grad, &fdv, 0);
        eprintln!("  nucleus exponent {nuc:e}: max|analytic − FD| = {e:.2e}");
    }
    let d = max_diff(&gs[0].grad, &gs[1].grad);
    eprintln!("LiH STO-3G dense: |F_ewald − F_none| = {d:.2e}");
    assert!(d < EWALD_NONE_BAR, "{d:e}");
    let e = max_fd_err(
        &grad(&scf_n, ExxDiv::None, Some(GradMutation::WSign), None).grad,
        &fdv,
        0,
    );
    eprintln!("LiH STO-3G dense mutant WSign: max|analytic − FD| = {e:.2e}");
    assert!(e > MUTANT_BAR, "WSign escaped the heavy-atom anchor: {e:e}");
}

/// RS-GDF RHF force anchor, both exxdiv, plus the nucleus-exponent scan and
/// the listed mutants (each must miss FD by more than its bar).
fn rhf_rsgdf_force_case(
    tag: &str,
    sys: &System,
    bs: &BasisSet,
    aux_bs: &BasisSet,
    comps: &[(usize, usize)],
    bar: f64,
    mutants: &[(GradMutation, f64)],
) {
    let t0 = std::time::Instant::now();
    let su = setup(tag, sys.cell(), bs, aux_bs);
    let n_drop_ref = su.gdf.stats().n_dropped;
    let scf_n = rhf_gdf(&su.cell, &su.prep, &su.hc, &su.gdf, ExxDiv::None, None);
    let scf_e = rhf_gdf(
        &su.cell,
        &su.prep,
        &su.hc,
        &su.gdf,
        ExxDiv::Ewald,
        Some(&scf_n.density_total),
    );
    eprintln!(
        "{tag}: reference build + 2 SCFs {:.1} s (E_none {:.12}, {} it)",
        t0.elapsed().as_secs_f64(),
        scf_n.energy,
        scf_n.iterations
    );
    let seed = scf_n.density_total.clone();
    let t1 = std::time::Instant::now();
    let fdv = fd(sys, comps, FD_H, |c| {
        let (p, h) = hcore_of(tag, c, bs, false);
        let (_, g) = gdf_of(c, &p, &h, aux_bs, false);
        assert_eq!(
            g.stats().n_dropped,
            n_drop_ref,
            "{tag}: aux-metric drop count changed at a displaced geometry"
        );
        [ExxDiv::None, ExxDiv::Ewald]
            .iter()
            .map(|&x| rhf_gdf(c, &p, &h, &g, x, Some(&seed)).energy)
            .collect()
    });
    eprintln!("{tag}: FD {:.1} s", t1.elapsed().as_secs_f64());
    let grad = |scf: &ScfResult, exx: ExxDiv, m: Option<GradMutation>, nuc: Option<f64>| {
        gamma_rhf_gradient_rsgdf(
            &su.cell,
            &su.prep,
            &hcore_cfg(),
            &su.hc,
            &src(&su),
            scf,
            exx,
            &gcfg(m, nuc),
        )
        .expect("gamma_rhf_gradient_rsgdf")
    };
    let mut gs = Vec::new();
    for (k, (scf, exx)) in [(&scf_n, ExxDiv::None), (&scf_e, ExxDiv::Ewald)]
        .into_iter()
        .enumerate()
    {
        let g = grad(scf, exx, None, None);
        let err = report_force(&format!("{tag} RHF {exx:?}"), &g, &fdv, k);
        assert!(amax(&g.grad) > 1e-3, "{tag}: vacuous comparison");
        assert!(err < bar, "{tag} {exx:?}: {err:e} (bar {bar:e})");
        assert!(g.net_force < NET_FORCE_BAR, "{tag}: ΣF {:e}", g.net_force);
        // The aux-motion and metric pieces cancel to the fit error; an anchor
        // that passed with them absent would not test them.
        assert!(amax(&g.parts.fit_aux_sr) + amax(&g.parts.fit_aux_lr) > 1e-4);
        assert!(amax(&g.parts.fit_metric_sr) + amax(&g.parts.fit_metric_lr) > 1e-4);
        assert_eq!(g.fit.as_ref().unwrap().n_dropped, n_drop_ref);
        gs.push(g);
    }
    for nuc in NUC_SCAN {
        let e = max_fd_err(&grad(&scf_n, ExxDiv::None, None, Some(nuc)).grad, &fdv, 0);
        eprintln!("  {tag} nucleus exponent {nuc:e}: max|analytic − FD| = {e:.2e}");
    }
    let d = max_diff(&gs[0].grad, &gs[1].grad);
    eprintln!("{tag}: |F_ewald − F_none| = {d:.2e}");
    assert!(d < EWALD_NONE_BAR, "{tag}: {d:e}");
    for &(m, min) in mutants {
        let e = max_fd_err(&grad(&scf_n, ExxDiv::None, Some(m), None).grad, &fdv, 0);
        eprintln!("  {tag} mutant {m:?}: max|analytic − FD| = {e:.2e} (must exceed {min:e})");
        assert!(e > min, "{tag}: {m:?} escaped the heavy-atom anchor: {e:e}");
    }
    eprintln!("{tag}: total {:.1} s", t0.elapsed().as_secs_f64());
}

#[test]
#[ignore = "slow: LiH STO-3G / cc-pvdz-ri RS-GDF, 6 displaced builds + 12 SCFs; run with --release -- --ignored"]
fn lih_sto3g_rsgdf_rhf_forces_match_fd() {
    rhf_rsgdf_force_case(
        "LiH STO-3G/cc-pvdz-ri",
        &lih(),
        &bundled("sto-3g"),
        &bundled("cc-pvdz-ri"),
        &LIH_COMPS,
        P_BAR,
        &[(GradMutation::FitNoMetric, FIT_MUTANT_BAR)],
    );
}

/// 6-31G: Li 1s on 6 primitives, 2sp and 3sp shells sharing exponents
/// between the s and p shell (the SP split).
#[test]
#[ignore = "slow: LiH 6-31G / cc-pvdz-ri RS-GDF, 6 displaced builds + 12 SCFs; run with --release -- --ignored"]
fn lih_631g_rsgdf_rhf_forces_match_fd() {
    rhf_rsgdf_force_case(
        "LiH 6-31G/cc-pvdz-ri",
        &lih(),
        &bundled("6-31g"),
        &bundled("cc-pvdz-ri"),
        &LIH_COMPS,
        P_BAR,
        &[],
    );
}

/// cc-pVDZ (Li diffuse primitives discarded): pure d on Li, H p, general
/// contractions on shared primitives; aux up to f on Li.
#[test]
#[ignore = "slow: LiH cc-pVDZ / cc-pvdz-ri RS-GDF, 6 displaced builds + 12 SCFs; run with --release -- --ignored"]
fn lih_ccpvdz_rsgdf_rhf_forces_match_fd() {
    rhf_rsgdf_force_case(
        "LiH cc-pVDZ(Li discard)/cc-pvdz-ri",
        &lih(),
        &lih_ccpvdz(),
        &bundled("cc-pvdz-ri"),
        &LIH_COMPS,
        D_BAR,
        &[],
    );
}

/// RS-GDF RHF stress anchor: all nine components, both exxdiv, Richardson;
/// the antisymmetric part reported and bounded; the listed mutants must
/// miss the Richardson value by > [`MUTANT_BAR`].
fn rhf_rsgdf_stress_case(
    tag: &str,
    sys: &System,
    bs: &BasisSet,
    aux_bs: &BasisSet,
    bar: f64,
    antisym_bar: f64,
    mutants: &[StressMutation],
) {
    let t0 = std::time::Instant::now();
    let su = setup(tag, sys.cell(), bs, aux_bs);
    let n_drop_ref = su.gdf.stats().n_dropped;
    let scf_n = rhf_gdf(&su.cell, &su.prep, &su.hc, &su.gdf, ExxDiv::None, None);
    let scf_e = rhf_gdf(
        &su.cell,
        &su.prep,
        &su.hc,
        &su.gdf,
        ExxDiv::Ewald,
        Some(&scf_n.density_total),
    );
    let seed = scf_n.density_total.clone();
    let energies = |c: &Cell| -> Vec<f64> {
        let (p, h) = hcore_of(tag, c, bs, false);
        let (_, g) = gdf_of(c, &p, &h, aux_bs, false);
        assert_eq!(
            g.stats().n_dropped,
            n_drop_ref,
            "{tag}: the metric cut changed under strain (E(ε) discontinuous)"
        );
        [ExxDiv::None, ExxDiv::Ewald]
            .iter()
            .map(|&x| rhf_gdf(c, &p, &h, &g, x, Some(&seed)).energy)
            .collect()
    };
    let fd_h = fd_strain(&su.cell, FD_H, &energies);
    let fd_h2 = fd_strain(&su.cell, 0.5 * FD_H, &energies);
    eprintln!(
        "{tag}: reference + 36 strained builds {:.1} s",
        t0.elapsed().as_secs_f64()
    );
    let stress = |scf: &ScfResult,
                  exx: ExxDiv,
                  m: Option<StressMutation>,
                  nuc: Option<f64>|
     -> GammaStress {
        gamma_rhf_stress_rsgdf(
            &su.cell,
            &su.prep,
            &hcore_cfg(),
            &su.hc,
            &src(&su),
            scf,
            exx,
            &scfg(m, nuc),
        )
        .expect("gamma_rhf_stress_rsgdf")
    };
    let scfs = [&scf_n, &scf_e];
    for (k, exx) in [ExxDiv::None, ExxDiv::Ewald].into_iter().enumerate() {
        let st = stress(scfs[k], exx, None, None);
        let rich = richardson(&fd_h[k], &fd_h2[k]);
        let err_h = max_err3(&st.de_deps, &fd_h[k]);
        let err_r = max_err3(&st.de_deps, &rich);
        let an_as = antisym(&st.de_deps);
        eprintln!(
            "{tag} RHF stress {exx:?}: FD(h) {err_h:.2e}, Richardson {err_r:.2e}, \
             |an − anᵀ| {an_as:.2e}, |Rich − Richᵀ| {:.2e}, comm {:.1e}, Ω {:.3}, P {:+.10e}\n \
             analytic dE/dε:\n{}\n analytic − Richardson:\n{}",
            antisym(&rich),
            st.commutator,
            st.volume,
            st.pressure(),
            fmt3(&st.de_deps),
            fmt3(&diff3(&st.de_deps, &rich))
        );
        assert!(
            st.de_deps.iter().flatten().any(|v| v.abs() > 1e-3),
            "{tag}: vacuous stress"
        );
        assert!(
            err_r < bar,
            "{tag} {exx:?}: Richardson {err_r:e} (bar {bar:e})"
        );
        assert!(
            an_as < antisym_bar,
            "{tag} {exx:?}: antisymmetric part {an_as:e}"
        );
        let j2g0 = st
            .parts
            .fit_j2_g0_volume
            .iter()
            .flatten()
            .fold(0.0_f64, |m, v| m.max(v.abs()));
        assert!(
            j2g0 > 1e-4,
            "{tag}: J2 G = 0 volume term must be live: {j2g0:e}"
        );
    }
    for nuc in NUC_SCAN {
        let rich = richardson(&fd_h[0], &fd_h2[0]);
        let st = stress(&scf_n, ExxDiv::None, None, Some(nuc));
        eprintln!(
            "  {tag} nucleus exponent {nuc:e}: Richardson {:.2e}, |an − anᵀ| {:.2e}",
            max_err3(&st.de_deps, &rich),
            antisym(&st.de_deps)
        );
    }
    let rich0 = richardson(&fd_h[0], &fd_h2[0]);
    for &m in mutants {
        let e = max_err3(&stress(&scf_n, ExxDiv::None, Some(m), None).de_deps, &rich0);
        eprintln!("  {tag} stress mutant {m:?}: max|an − Richardson| = {e:.2e}");
        assert!(e > MUTANT_BAR, "{tag}: stress mutant {m:?} escaped: {e:e}");
    }
    eprintln!("{tag}: total {:.1} s", t0.elapsed().as_secs_f64());
}

#[test]
#[ignore = "slow: LiH 6-31G / cc-pvdz-ri RS-GDF stress, 36 strained builds + 72 SCFs; run with --release -- --ignored"]
fn lih_631g_rsgdf_rhf_stress_all_nine_richardson() {
    rhf_rsgdf_stress_case(
        "LiH 6-31G/cc-pvdz-ri",
        &lih(),
        &bundled("6-31g"),
        &bundled("cc-pvdz-ri"),
        STRESS_P_BAR,
        STRESS_P_ANTISYM_BAR,
        &[StressMutation::NoPulay],
    );
}

// ================================================================ CO

/// CO cc-pVDZ (pure d on C/O, general contractions) with
/// def2-universal-jkfit (l up to g on C/O). Mutants: the fit metric term
/// and the W sign.
#[test]
#[ignore = "slow: CO cc-pVDZ / def2-universal-jkfit RS-GDF, 6 displaced builds + 12 SCFs; run with --release -- --ignored"]
fn co_ccpvdz_jkfit_rhf_forces_match_fd_and_catch_mutants() {
    rhf_rsgdf_force_case(
        "CO cc-pVDZ/jkfit",
        &co(),
        &bundled("cc-pvdz"),
        &bundled("def2-universal-jkfit"),
        &CO_COMPS,
        D_BAR,
        &[
            (GradMutation::FitNoMetric, FIT_MUTANT_BAR),
            (GradMutation::WSign, MUTANT_BAR),
        ],
    );
}

#[test]
#[ignore = "slow: CO cc-pVDZ / def2-universal-jkfit RS-GDF stress, 36 strained builds + 72 SCFs; run with --release -- --ignored"]
fn co_ccpvdz_jkfit_rhf_stress_all_nine_richardson() {
    rhf_rsgdf_stress_case(
        "CO cc-pVDZ/jkfit",
        &co(),
        &bundled("cc-pvdz"),
        &bundled("def2-universal-jkfit"),
        STRESS_D_BAR,
        STRESS_D_ANTISYM_BAR,
        &[StressMutation::NoMetric, StressMutation::NoPulay],
    );
}

/// CO RKS PBE on the atom grid (d shells through the XC AO Hessians and
/// the grid response), RS-GDF J, FD h = 5e-5.
#[test]
#[ignore = "slow: CO cc-pVDZ RKS PBE on the (50, 110) atom grid, 6 displaced RS-GDF builds + 6 KS SCFs; run with --release -- --ignored"]
fn co_ccpvdz_rks_pbe_atom_grid_forces_match_fd() {
    let tag = "CO cc-pVDZ/jkfit RKS PBE";
    let sys = co();
    let bs = bundled("cc-pvdz");
    let aux_bs = bundled("def2-universal-jkfit");
    let t0 = std::time::Instant::now();
    let su = setup(tag, sys.cell(), &bs, &aux_bs);
    let n_drop_ref = su.gdf.stats().n_dropped;
    let rks =
        |c: &Cell, p: &PreparedBasis, h: &PeriodicHcore, g: &RsGdf, seed: Option<&Array2<f64>>| {
            let r = gamma_rks(c, p, h, GammaUhfIntegrals::RsGdf(g), &rks_cfg("PBE", seed))
                .expect("gamma_rks PBE");
            assert!(r.scf.converged, "gamma_rks PBE did not converge");
            r.scf
        };
    let scf = rks(&su.cell, &su.prep, &su.hc, &su.gdf, None);
    let seed = scf.density_total.clone();
    let fdv = fd(&sys, &CO_COMPS, FD_H_KS, |c| {
        let (p, h) = hcore_of(tag, c, &bs, false);
        let (_, g) = gdf_of(c, &p, &h, &aux_bs, false);
        assert_eq!(
            g.stats().n_dropped,
            n_drop_ref,
            "{tag}: aux drop count changed"
        );
        vec![rks(c, &p, &h, &g, Some(&seed)).energy]
    });
    let rcfg = rks_cfg("PBE", None);
    let grad = |m: Option<GradMutation>| {
        gamma_rks_gradient_rsgdf(
            &su.cell,
            &su.prep,
            &hcore_cfg(),
            &su.hc,
            &src(&su),
            &scf,
            &rcfg,
            &gcfg(m, None),
        )
        .expect("gamma_rks_gradient_rsgdf")
    };
    let g = grad(None);
    let err = report_force(tag, &g, &fdv, 0);
    eprintln!(
        "{tag}: E_xc {:?}, grid points {}, |xc_ao| {:.2e} |xc_point| {:.2e} |xc_weight| {:.2e}; {:.1} s",
        g.e_xc,
        g.n_grid_points,
        amax(&g.parts.xc_ao),
        amax(&g.parts.xc_point),
        amax(&g.parts.xc_weight),
        t0.elapsed().as_secs_f64()
    );
    assert!(g.e_xc.is_some() && g.n_grid_points > 0);
    assert!(amax(&g.grad) > 1e-3, "vacuous comparison");
    assert!(err < KS_D_BAR, "{tag}: {err:e} (bar {KS_D_BAR:e})");
    assert!(g.net_force < NET_FORCE_BAR, "{tag}: ΣF {:e}", g.net_force);
    let e = max_fd_err(&grad(Some(GradMutation::XcNoAo)).grad, &fdv, 0);
    eprintln!("  {tag} mutant XcNoAo: {e:.2e}");
    assert!(e > MUTANT_BAR, "{tag}: XcNoAo escaped: {e:e}");
    for m in [
        GradMutation::NoPointMotion,
        GradMutation::NoWeightDeriv,
        GradMutation::XcNoGga,
    ] {
        let e = max_fd_err(&grad(Some(m)).grad, &fdv, 0);
        eprintln!("  {tag} mutant {m:?}: {e:.2e} (reported only)");
    }
}

// ================================================================ OH

/// OH radical doublet, cc-pVDZ / cc-pvdz-ri: UHF (staged none → ewald)
/// forces for both stages, and the fit-metric mutant.
#[test]
#[ignore = "slow: OH cc-pVDZ / cc-pvdz-ri UHF doublet, 6 displaced RS-GDF builds + 12 UHF stages; run with --release -- --ignored"]
fn oh_ccpvdz_uhf_rsgdf_forces_match_fd() {
    let tag = "OH cc-pVDZ/cc-pvdz-ri UHF";
    let sys = oh();
    let bs = bundled("cc-pvdz");
    let aux_bs = bundled("cc-pvdz-ri");
    let t0 = std::time::Instant::now();
    let su = setup(tag, sys.cell(), &bs, &aux_bs);
    let n_drop_ref = su.gdf.stats().n_dropped;
    let r = uhf_on(
        &su.cell,
        &su.prep,
        &su.hc,
        &su.gdf,
        &uhf_cfg(ExxDiv::Ewald, None),
    );
    let none = r.none_stage.clone().expect("staged none stage");
    eprintln!(
        "{tag}: ⟨S²⟩ {:.6}, gaps α {:?} β {:?} (v_M {:.4}), E {:.12}",
        r.s2, r.gaps.gap_alpha, r.gaps.gap_beta, r.madelung, r.scf.energy
    );
    let init = mos_of(&none);
    let fdv = fd(&sys, &OH_COMPS, FD_H, |c| {
        let (p, h) = hcore_of(tag, c, &bs, false);
        let (_, g) = gdf_of(c, &p, &h, &aux_bs, false);
        assert_eq!(
            g.stats().n_dropped,
            n_drop_ref,
            "{tag}: aux drop count changed"
        );
        let rr = uhf_on(c, &p, &h, &g, &uhf_cfg(ExxDiv::Ewald, init.clone()));
        assert!(
            (rr.s2 - r.s2).abs() < 1e-3,
            "{tag}: displaced UHF landed on another state (⟨S²⟩ {} vs {})",
            rr.s2,
            r.s2
        );
        vec![rr.none_stage.expect("none stage").energy, rr.scf.energy]
    });
    let grad = |scf: &ScfResult, exx: ExxDiv, m: Option<GradMutation>, nuc: Option<f64>| {
        gamma_uhf_gradient_rsgdf(
            &su.cell,
            &su.prep,
            &hcore_cfg(),
            &su.hc,
            &src(&su),
            scf,
            exx,
            &gcfg(m, nuc),
        )
        .expect("gamma_uhf_gradient_rsgdf")
    };
    let mut gs = Vec::new();
    for (k, (scf, exx)) in [(&none, ExxDiv::None), (&r.scf, ExxDiv::Ewald)]
        .into_iter()
        .enumerate()
    {
        let g = grad(scf, exx, None, None);
        let err = report_force(&format!("{tag} {exx:?}"), &g, &fdv, k);
        assert!(amax(&g.grad) > 1e-3, "vacuous comparison");
        assert!(err < D_BAR, "{tag} {exx:?}: {err:e} (bar {D_BAR:e})");
        assert!(g.net_force < NET_FORCE_BAR, "{tag}: ΣF {:e}", g.net_force);
        gs.push(g);
    }
    for nuc in NUC_SCAN {
        let e = max_fd_err(&grad(&none, ExxDiv::None, None, Some(nuc)).grad, &fdv, 0);
        eprintln!("  {tag} nucleus exponent {nuc:e}: max|analytic − FD| = {e:.2e}");
    }
    // F(ewald) vs F(none) on ONE open-shell density.
    let d = max_diff(&grad(&r.scf, ExxDiv::None, None, None).grad, &gs[1].grad);
    eprintln!("{tag}: |F_ewald − F_none| on one density = {d:.2e}");
    assert!(d < EWALD_NONE_BAR, "{tag}: {d:e}");
    let e = max_fd_err(
        &grad(&none, ExxDiv::None, Some(GradMutation::FitNoMetric), None).grad,
        &fdv,
        0,
    );
    eprintln!("  {tag} mutant FitNoMetric: {e:.2e}");
    assert!(e > FIT_MUTANT_BAR, "{tag}: FitNoMetric escaped: {e:e}");
    let e = max_fd_err(
        &grad(&none, ExxDiv::None, Some(GradMutation::WSpinSum), None).grad,
        &fdv,
        0,
    );
    eprintln!("  {tag} mutant WSpinSum: {e:.2e} (reported only)");
    eprintln!("{tag}: total {:.1} s", t0.elapsed().as_secs_f64());
}

/// OH radical UKS PBE on the atom grid (spin-polarized XC with d shells),
/// RS-GDF J, FD h = 5e-5.
#[test]
#[ignore = "slow: OH cc-pVDZ UKS PBE on the (50, 110) atom grid, 6 displaced RS-GDF builds + 6 UKS SCFs; run with --release -- --ignored"]
fn oh_ccpvdz_uks_pbe_rsgdf_forces_match_fd() {
    let tag = "OH cc-pVDZ/cc-pvdz-ri UKS PBE";
    let sys = oh();
    let bs = bundled("cc-pvdz");
    let aux_bs = bundled("cc-pvdz-ri");
    let t0 = std::time::Instant::now();
    let su = setup(tag, sys.cell(), &bs, &aux_bs);
    let n_drop_ref = su.gdf.stats().n_dropped;
    let uks = |c: &Cell, p: &PreparedBasis, h: &PeriodicHcore, g: &RsGdf, init: Mos| {
        let r = gamma_uks(c, p, h, GammaUhfIntegrals::RsGdf(g), &uks_cfg("PBE", init))
            .expect("gamma_uks PBE");
        assert!(r.scf.converged, "gamma_uks PBE did not converge");
        r
    };
    let r = uks(&su.cell, &su.prep, &su.hc, &su.gdf, None);
    eprintln!("{tag}: ⟨S²⟩ {:.6}, E {:.12}", r.s2, r.scf.energy);
    let init = mos_of(r.none_stage.as_ref().unwrap_or(&r.scf));
    let fdv = fd(&sys, &OH_COMPS, FD_H_KS, |c| {
        let (p, h) = hcore_of(tag, c, &bs, false);
        let (_, g) = gdf_of(c, &p, &h, &aux_bs, false);
        assert_eq!(
            g.stats().n_dropped,
            n_drop_ref,
            "{tag}: aux drop count changed"
        );
        let rr = uks(c, &p, &h, &g, init.clone());
        assert!(
            (rr.s2 - r.s2).abs() < 1e-3,
            "{tag}: displaced UKS landed on another state (⟨S²⟩ {} vs {})",
            rr.s2,
            r.s2
        );
        vec![rr.scf.energy]
    });
    let cfg = uks_cfg("PBE", None);
    let g = gamma_uks_gradient_rsgdf(
        &su.cell,
        &su.prep,
        &hcore_cfg(),
        &su.hc,
        &src(&su),
        &r.scf,
        &cfg,
        &gcfg(None, None),
    )
    .expect("gamma_uks_gradient_rsgdf");
    let err = report_force(tag, &g, &fdv, 0);
    eprintln!("{tag}: {:.1} s", t0.elapsed().as_secs_f64());
    assert!(amax(&g.grad) > 1e-3, "vacuous comparison");
    assert!(err < KS_D_BAR, "{tag}: {err:e} (bar {KS_D_BAR:e})");
    assert!(g.net_force < NET_FORCE_BAR, "{tag}: ΣF {:e}", g.net_force);
}
