//! Column rotation in the k-point FORCE and STRESS derivative walks
//! (`ferric_pbc::sr_rotation`, "Forces and stress"; `rsgdf::kpoint::kderiv`
//! module doc "Column rotation"; the k sibling of the Gamma section (h) of
//! `pbc_sr_rotation.rs`, and the derivative twin of `pbc_sr_rotation_k.rs`).
//!
//! Construction under test: when the k energy build ran the rotated SR walks
//! (hcore `V_SR(k)` and the RS-GDF SR 3-centre sum), the derivative walks
//! run on the same rotated shells, contracted with the forward-transformed
//! per-residue weights `W' = Tᵀ W[r] T` (fit: the phase-folded `Zbin[r_L,
//! r_T]`; hcore: the image weight `w_d[r]`). `T` is real, geometry- and
//! k-independent. The metric, LR and G = 0 pieces stay on the parent basis.
//!
//! What each test pins (the exactness anchors were written and run first):
//!
//! (a) EXACTNESS ANCHOR (rotation vs the unrotated derivative walks of the
//!     SAME energy and density, `RotOff` mutant): they agree to the
//!     screening precision. Per PART (hcore `vsr`, fit `j3_sr`), because the
//!     totals hide a part-sized defect behind the aux stationarity.
//! (b) PHYSICS ANCHOR: rotated analytic force / stress == central FD of the
//!     ROTATED k energy (fresh hcore, fit and SCF per displacement, rotation
//!     explicitly `On` in the energy), at the O(h²) floor. This is the anchor
//!     that proves the derivative matches the rotated energy.
//! (c) 1x1x1 mesh == the Gamma rotated force / stress.
//! (d) k mesh (1x1x3, non-TRIM) == supercell / N with the rotation on both
//!     sides (Gamma on the explicit supercell).
//! (e) Bitwise across thread counts.
//! (f) MUTANTS (production-wired, `KGradMutation::Rot*` /
//!     `KStressMutation::Rot*`): no `T`, `T` for `Tᵀ`, one AO index, wrong
//!     sign, AO pair transposed, reversed residue order (fit and hcore), and
//!     the derivative walks unrotated (`RotOff`).
//! (g) Refusals: a rotated build over an `Off` config, an explicit `On` over
//!     an unrotated build, a rotation mutant in the config.
//!
//! Artifact hypothesis, stated before measuring. Correct: (a) at the
//! screening precision (measured at the default screen 1e-9/1e-10: force
//! 1.5e-10, stress 6.7e-9; ~1e-13..1e-11 at the 1e-14 screens) and shrinking
//! with it; (b) at the
//! FD floor. Forward-transform mutants: O(|T_ks|) = O(0.1..1) on the rotated
//! columns, independent of the precision. `RotOff` is NOT O(1): rotated and
//! unrotated walks differ only at the screening precision, so at the default
//! screen it is expected to be BLIND to (a) and (b) by construction (the
//! measured size is reported); a LOOSE screen makes the two walks differ by
//! ~the screen, and then (a) at that screen is the test that sees it.

mod common;

use common::*;
use ferric_core::basis::{self, BasisSet};
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_pbc::dense_aft::ExxDiv;
use ferric_pbc::grad::{gamma_rhf_gradient_rsgdf, GammaGradConfig, GammaGradient, RsGdfGradSource};
use ferric_pbc::hcore::kpoint::{periodic_hcore_kpts, PeriodicHcoreK};
use ferric_pbc::hcore::{periodic_hcore, PeriodicHcoreConfig};
use ferric_pbc::kgrad::{
    kpoint_rhf_gradient, kpoint_uhf_gradient, KGradConfig, KGradJk, KGradMutation, KGradient,
    KRsGdfGradSource,
};
use ferric_pbc::kpts::{KPointMesh, MeshCentring};
use ferric_pbc::kscf::{solve_krhf_injected, KPointInjection, KScfConfig, KScfResult};
use ferric_pbc::kstress::{
    kpoint_rhf_stress, kpoint_uhf_stress, KStress, KStressConfig, KStressJk, KStressMutation,
};
use ferric_pbc::kuscf::solve_kuhf_injected;
use ferric_pbc::lattice::Cell;
use ferric_pbc::rsgdf::kpoint::{KRsGdf, KRsGdfConfig};
use ferric_pbc::rsgdf::{RangeSplit, RsGdf, RsGdfConfig};
use ferric_pbc::sr_rotation::{ColumnRotation, ColumnRotationMutant, SrColumnRotation};
use ferric_pbc::stress::{gamma_rhf_stress_rsgdf, GammaStressConfig, Mat3};
use ferric_scf::result::{ScfExit, ScfResult, Spin};
use ndarray::Array2;
use num_complex::Complex64;

const HCORE_OMEGA: f64 = 0.8;
const OFF: SrColumnRotation = SrColumnRotation::Off;
/// Off-axis H2 (all force components nonzero), lattice 5 Bohr.
const H2_ATOMS: [[f64; 3]; 2] = [[0.3, 0.2, 0.1], [0.35, 0.12, 1.5]];

fn on() -> SrColumnRotation {
    SrColumnRotation::on()
}

fn cell_h2(pos: &[[f64; 3]]) -> Cell {
    Cell::new(hydrogens(pos), cubic(5.0)).expect("cell")
}

fn hcfg(rot: SrColumnRotation, precision: f64) -> PeriodicHcoreConfig {
    PeriodicHcoreConfig {
        precision,
        sr_column_rotation: rot,
        ..PeriodicHcoreConfig::with_omega(HCORE_OMEGA)
    }
}

fn gcfg(split: Option<RangeSplit>, rot: SrColumnRotation, precision: f64) -> KRsGdfConfig {
    KRsGdfConfig {
        gdf: RsGdfConfig {
            omega: 1.0,
            precision,
            exxdiv: ExxDiv::None,
            budget_bytes: Some(1 << 31),
            range_split: split,
            sr_column_rotation: rot,
            ..Default::default()
        },
        mutation: None,
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

fn bset(name: &str) -> BasisSet {
    basis::bundled(name).expect("bundled basis")
}

/// One k-point RS-GDF RHF run: hcore, fit and converged SCF (exxdiv none and
/// Ewald) at one rotation request / screening precision.
struct Run {
    cell: Cell,
    prep: PreparedBasis,
    aux: PreparedBasis,
    mesh: KPointMesh,
    hcfg: PeriodicHcoreConfig,
    hk: PeriodicHcoreK,
    gcfg: KRsGdfConfig,
    gdf: KRsGdf,
}

fn run(
    cell: Cell,
    n: [usize; 3],
    split: Option<RangeSplit>,
    rot: SrColumnRotation,
    prec: (f64, f64),
) -> Run {
    run_c(cell, n, MeshCentring::Gamma, split, rot, prec)
}

fn run_c(
    cell: Cell,
    n: [usize; 3],
    centring: MeshCentring,
    split: Option<RangeSplit>,
    rot: SrColumnRotation,
    (hp, gp): (f64, f64),
) -> Run {
    let prep = prep_for(&cell, &bset("cc-pvdz"));
    let aux = prep_for(&cell, &bset("cc-pvdz-ri"));
    let mesh = KPointMesh::new(&cell, n, centring).expect("mesh");
    let hcfg = hcfg(rot, hp);
    let hk = periodic_hcore_kpts(&cell, &prep, &mesh, &hcfg).expect("hcore(k)");
    let gcfg = gcfg(split, rot, gp);
    let gdf = KRsGdf::build(&cell, &prep, &aux, &mesh, &hk.s, &gcfg).expect("KRsGdf");
    Run {
        cell,
        prep,
        aux,
        mesh,
        hcfg,
        hk,
        gcfg,
        gdf,
    }
}

/// (hcore, fit) screening precision of the cheap fixtures.
const DEFAULT_PREC: (f64, f64) = (1e-9, 1e-10);

impl Run {
    fn vm(&self, exx: ExxDiv) -> f64 {
        match exx {
            ExxDiv::None => 0.0,
            ExxDiv::Ewald => self.mesh.madelung(&self.cell).unwrap(),
        }
    }

    fn scf(&self, exx: ExxDiv) -> KScfResult {
        let inj = KPointInjection {
            s: self.hk.s.clone(),
            h: self.hk.h.clone(),
            vnn: self.hk.enn,
            jk: Box::new(self.gdf.jk_builder_with_madelung(self.vm(exx))),
        };
        let r = solve_krhf_injected(&self.cell, &self.mesh, &kscf_cfg(), inj).expect("k-RHF");
        assert!(r.converged, "k-RHF not converged ({} it)", r.iterations);
        r
    }

    fn src<'a>(&'a self, cfg: &'a KRsGdfConfig) -> KRsGdfGradSource<'a> {
        KRsGdfGradSource {
            gdf: &self.gdf,
            cfg,
            aux: &self.aux,
        }
    }

    fn try_force(
        &self,
        scf: &KScfResult,
        exx: ExxDiv,
        m: Option<KGradMutation>,
    ) -> Result<KGradient, ferric_core::FerricError> {
        self.try_force_cfg(&self.hcfg, &self.gcfg, scf, exx, m)
    }

    /// The force call with explicit hcore / fit configs (which may disagree
    /// with the energy build: the refusal tests).
    fn try_force_cfg(
        &self,
        h: &PeriodicHcoreConfig,
        g: &KRsGdfConfig,
        scf: &KScfResult,
        exx: ExxDiv,
        m: Option<KGradMutation>,
    ) -> Result<KGradient, ferric_core::FerricError> {
        kpoint_rhf_gradient(
            &self.cell,
            &self.prep,
            &self.mesh,
            h,
            &self.hk,
            KGradJk::RsGdf(self.src(g)),
            scf,
            exx,
            &KGradConfig {
                mutation: m,
                budget_bytes: Some(1 << 31),
                ..Default::default()
            },
        )
    }

    fn force(&self, scf: &KScfResult, exx: ExxDiv, m: Option<KGradMutation>) -> KGradient {
        let f = self.try_force(scf, exx, m).expect("k force");
        if m.is_none() {
            // The walks ACTUALLY run rotate the columns the energy rotated.
            assert!(
                self.walked_ok(
                    f.hcore_rotated_columns,
                    f.fit_checks.map(|c| c.rotated_columns)
                ),
                "force walked hcore {} fit {:?}",
                f.hcore_rotated_columns,
                f.fit_checks.map(|c| c.rotated_columns)
            );
        }
        f
    }

    /// Whether the derivative walks' rotated-column counts equal the energy
    /// build's (hcore always; the fit when the source is an RS-GDF fit).
    fn walked_ok(&self, hcore: usize, fit: Option<usize>) -> bool {
        hcore == self.hk.sr_rotated_columns
            && fit.is_none_or(|f| f == self.gdf.stats().sr_rotated_columns)
    }

    fn try_stress(
        &self,
        scf: &KScfResult,
        exx: ExxDiv,
        m: Option<KStressMutation>,
    ) -> Result<KStress, ferric_core::FerricError> {
        self.try_stress_cfg(&self.hcfg, &self.gcfg, scf, exx, m)
    }

    fn try_stress_cfg(
        &self,
        h: &PeriodicHcoreConfig,
        g: &KRsGdfConfig,
        scf: &KScfResult,
        exx: ExxDiv,
        m: Option<KStressMutation>,
    ) -> Result<KStress, ferric_core::FerricError> {
        kpoint_rhf_stress(
            &self.cell,
            &self.prep,
            &self.mesh,
            h,
            &self.hk,
            KStressJk::RsGdf(self.src(g)),
            scf,
            exx,
            &KStressConfig {
                budget_bytes: Some(1 << 31),
                mutation: m,
                ..Default::default()
            },
        )
    }

    fn stress(&self, scf: &KScfResult, exx: ExxDiv, m: Option<KStressMutation>) -> KStress {
        let s = self.try_stress(scf, exx, m).expect("k stress");
        if m.is_none() {
            assert!(
                self.walked_ok(
                    s.hcore_rotated_columns,
                    s.fit_checks.map(|c| c.rotated_columns)
                ),
                "stress walked hcore {} fit {:?}",
                s.hcore_rotated_columns,
                s.fit_checks.map(|c| c.rotated_columns)
            );
        }
        s
    }

    /// The rotation really ran in the energy (so comparisons are not vacuous).
    fn assert_rotates(&self, tag: &str) {
        assert_eq!(self.hk.sr_rotated_columns, 2, "{tag}: hcore did not rotate");
        assert_eq!(
            self.gdf.stats().sr_rotated_columns,
            2,
            "{tag}: fit did not rotate"
        );
    }
}

fn mdiff(a: &Mat3, b: &Mat3) -> f64 {
    let mut m = 0.0_f64;
    for i in 0..3 {
        for j in 0..3 {
            m = m.max((a[i][j] - b[i][j]).abs());
        }
    }
    m
}

fn mmax(a: &Mat3) -> f64 {
    a.iter().flatten().fold(0.0_f64, |m, x| m.max(x.abs()))
}

fn msum(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    a + b
}

fn splits() -> [Option<RangeSplit>; 2] {
    [None, Some(RangeSplit::default())]
}

/// (a): rotated vs unrotated derivative walks (same energy and density).
///
/// MEASURED on the fixed build (H2, cc-pVDZ / cc-pVDZ-ri, `DEFAULT_PREC`
/// hcore 1e-9 / fit 1e-10, RS-GDF): force total 1.5e-10 (1x1x3) / 1.6e-10
/// (split 1x1x2), stress total 6.5e-9 / 6.7e-9, dominated by the hcore part.
/// This IS the screening floor of these fixtures (it scales with the screen),
/// not an artifact of the libint2 Boys table-edge read (#433, fixed in #437):
/// the survey in `libint2_overrides/README.md` found the #421 ECP k-point test
/// to be the only existing test that evaluates a Boys argument of exactly
/// 117.0 (this file was not among them; not re-instrumented here). The bar is
/// 15x above the largest measured value (stress 6.7e-9) and a rotation
/// defect moves these terms by O(0.1..1) (>= 1e6 x the bar), so it is
/// derived from the screen, not from a defective run.
const D_BAR: f64 = 1e-7;

/// (a) EXACTNESS ANCHOR, forces and stress, per part. Prints the measured
/// sizes (the bar is derived from them in the test doc).
#[test]
fn rotated_k_derivative_walks_equal_the_unrotated_walks_per_part() {
    walk_anchor(&[([1, 1, 3], None, ExxDiv::None)]);
}

/// (a) on the range-split build (1x1x2, exxdiv none): the split derivative
/// walks have a non-ignored correctness test.
#[test]
fn rotated_k_derivative_walks_equal_the_unrotated_walks_per_part_split() {
    walk_anchor(&[([1, 1, 2], Some(RangeSplit::default()), ExxDiv::None)]);
}

/// (a) on the range-split build at the Ewald exxdiv.
#[test]
#[ignore = "slow (~1 min release): range-split walks, force and stress, 1x1x3"]
fn rotated_k_derivative_walks_equal_the_unrotated_walks_per_part_split_ewald() {
    walk_anchor(&[([1, 1, 3], Some(RangeSplit::default()), ExxDiv::Ewald)]);
}

fn walk_anchor(cases: &[([usize; 3], Option<RangeSplit>, ExxDiv)]) {
    for &(n, split, exx) in cases {
        let r = run(cell_h2(&H2_ATOMS), n, split, on(), DEFAULT_PREC);
        r.assert_rotates("rotated build");
        {
            let tag = format!("{n:?} split {:?} {exx:?}", split.map(|s| s.lambda));
            let scf = r.scf(exx);
            let f_on = r.force(&scf, exx, None);
            let f_off = r.force(&scf, exx, Some(KGradMutation::RotOff));
            let s_on = r.stress(&scf, exx, None);
            let s_off = r.stress(&scf, exx, Some(KStressMutation::RotOff));
            let hc = |f: &KGradient| msum(&f.parts.vsr_basis, &f.parts.vsr_nuc);
            let d_tot = max_abs_diff(&f_on.grad, &f_off.grad);
            let d_hc = max_abs_diff(&hc(&f_on), &hc(&f_off));
            let d_fit = max_abs_diff(&f_on.parts.fit_orb_sr, &f_off.parts.fit_orb_sr);
            let ds_tot = mdiff(&s_on.de_deps, &s_off.de_deps);
            let ds_hc = mdiff(&s_on.parts.vsr, &s_off.parts.vsr);
            let ds_fit = mdiff(&s_on.parts.fit_j3_sr, &s_off.parts.fit_j3_sr);
            eprintln!(
                "{tag}: force total {d_tot:.2e} hcore {d_hc:.2e} fit {d_fit:.2e}; \
                 stress total {ds_tot:.2e} hcore {ds_hc:.2e} fit {ds_fit:.2e}; \
                 F-mismatch {:.1e}/{:.1e}; hcore SR calls on/off {} vs {}, SR3 {:?} vs {:?}",
                f_on.fock_mismatch,
                s_on.fock_mismatch,
                f_on.n_sr_triplets,
                f_off.n_sr_triplets,
                f_on.fit_checks.map(|c| c.n_sr3),
                f_off.fit_checks.map(|c| c.n_sr3),
            );
            // The walks really differ (counters from the walk ACTUALLY run):
            // the rotated one rotates the energy's columns and does fewer
            // calls; the RotOff walk rotates none.
            assert_eq!(f_off.hcore_rotated_columns, 0, "{tag}");
            assert_eq!(f_off.fit_checks.unwrap().rotated_columns, 0, "{tag}");
            assert_eq!(s_off.hcore_rotated_columns, 0, "{tag}");
            assert!(
                f_on.n_sr_triplets < f_off.n_sr_triplets,
                "{tag}: hcore calls"
            );
            assert!(
                f_on.fit_checks.unwrap().n_sr3 < f_off.fit_checks.unwrap().n_sr3,
                "{tag}: SR3 calls"
            );
            assert!(d_tot < D_BAR && d_hc < D_BAR && d_fit < D_BAR, "{tag}");
            assert!(ds_tot < D_BAR && ds_hc < D_BAR && ds_fit < D_BAR, "{tag}");
            // Non-vacuous terms, and the Fock guard keeps working.
            assert!(max_abs(&f_off.parts.fit_orb_sr) > 1e-3 && max_abs(&hc(&f_off)) > 1e-3);
            assert!(mmax(&s_off.parts.vsr) > 1e-3 && mmax(&s_off.parts.fit_j3_sr) > 1e-3);
            assert!(
                f_on.fock_mismatch < 1e-8 && s_on.fock_mismatch < 1e-8,
                "{tag}"
            );
        }
    }
}

fn max_abs(m: &Array2<f64>) -> f64 {
    m.iter().fold(0.0_f64, |a, x| a.max(x.abs()))
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

/// Smallest miss a production-wired mutant must show against the rotated
/// force / stress (measured values are printed by the tests).
const MUTANT_FLOOR: f64 = 1e-6;

/// (f): every mutant misses the rotated force, per part where it acts.
#[test]
fn rotation_mutants_miss_the_rotated_force() {
    let r = run(cell_h2(&H2_ATOMS), [1, 1, 2], None, on(), DEFAULT_PREC);
    r.assert_rotates("1x1x2");
    let scf = r.scf(ExxDiv::None);
    let f0 = r.force(&scf, ExxDiv::None, None);
    let hc = |f: &KGradient| msum(&f.parts.vsr_basis, &f.parts.vsr_nuc);
    for m in FORCE_MUTANTS {
        let f = r.force(&scf, ExxDiv::None, Some(m));
        let tot = max_abs_diff(&f.grad, &f0.grad);
        let hcm = max_abs_diff(&hc(&f), &hc(&f0));
        let fit = max_abs_diff(&f.parts.fit_orb_sr, &f0.parts.fit_orb_sr);
        eprintln!("force mutant {m:?}: total {tot:.2e} hcore {hcm:.2e} fit {fit:.2e}");
        let acts_hcore = !matches!(
            m,
            KGradMutation::RotSwapPair | KGradMutation::RotFitWrongResidue
        );
        let acts_fit = m != KGradMutation::RotHcoreWrongResidue;
        assert!(tot > MUTANT_FLOOR, "{m:?} total {tot:e}");
        assert!(!acts_hcore || hcm > MUTANT_FLOOR, "{m:?} hcore {hcm:e}");
        assert!(!acts_fit || fit > MUTANT_FLOOR, "{m:?} fit {fit:e}");
    }
}

/// (f): every mutant misses the rotated stress, per part where it acts
/// (also against FD in `rotated_k_stress_matches_fd_of_the_rotated_k_energy`).
#[test]
#[ignore = "slow (~1.5 min release): 8 stress evaluations on 1x1x3"]
fn rotation_mutants_miss_the_rotated_stress() {
    let r = run(cell_h2(&H2_ATOMS), [1, 1, 3], None, on(), DEFAULT_PREC);
    r.assert_rotates("1x1x3");
    let scf = r.scf(ExxDiv::None);
    let s0 = r.stress(&scf, ExxDiv::None, None);
    // The transposed-pair defect on the NON-TRIM mesh (n_2 = 3): on a TRIM
    // mesh (1x1x2) r == -r (mod 2), so a transposed bin is nearly the same
    // bin and the defect is weak; here the force miss is printed and bounded.
    let f0 = r.force(&scf, ExxDiv::None, None);
    let fs = r.force(&scf, ExxDiv::None, Some(KGradMutation::RotSwapPair));
    let miss = max_abs_diff(&fs.grad, &f0.grad);
    eprintln!("force RotSwapPair on 1x1x3 (non-TRIM): miss {miss:.2e}");
    assert!(miss > MUTANT_FLOOR, "{miss:e}");
    for m in STRESS_MUTANTS {
        let s = r.stress(&scf, ExxDiv::None, Some(m));
        let tot = mdiff(&s.de_deps, &s0.de_deps);
        let hcm = mdiff(&s.parts.vsr, &s0.parts.vsr);
        let fit = mdiff(&s.parts.fit_j3_sr, &s0.parts.fit_j3_sr);
        eprintln!("stress mutant {m:?}: total {tot:.2e} hcore {hcm:.2e} fit {fit:.2e}");
        let acts_hcore = !matches!(
            m,
            KStressMutation::RotSwapPair | KStressMutation::RotFitWrongResidue
        );
        let acts_fit = m != KStressMutation::RotHcoreWrongResidue;
        assert!(tot > MUTANT_FLOOR, "{m:?} total {tot:e}");
        assert!(!acts_hcore || hcm > MUTANT_FLOOR, "{m:?} hcore {hcm:e}");
        assert!(!acts_fit || fit > MUTANT_FLOOR, "{m:?} fit {fit:e}");
    }
}

fn energy_at(
    pos: &[[f64; 3]],
    strain: Option<&Mat3>,
    n: [usize; 3],
    split: Option<RangeSplit>,
) -> f64 {
    let mut cell = cell_h2(pos);
    if let Some(e) = strain {
        cell = cell.strained(e).expect("strained");
    }
    run(cell, n, split, on(), DEFAULT_PREC)
        .scf(ExxDiv::None)
        .energy
}

/// FD bars. MEASURED (H2 1x1x2, rotated, h = 1e-4, derivative at the energy's
/// 1e16 nucleus): force |analytic - FD| 2.1e-9 (split omega 1.0; RotOff
/// 2.2e-9), stress 8.7e-9 (split and unsplit); the smallest mutant misses are
/// 2.5e-5 (force) and 1.0e-3 (stress), so the bars (1e-8 / 5e-8, 4.8x / 5.8x
/// the measurement) sit 3+ decades under every mutant.
const FD_F_BAR: f64 = 1e-8;
const FD_S_BAR: f64 = 5e-8;
const FD_H: f64 = 1e-4;

/// (b) PHYSICS ANCHOR, forces: rotated analytic force == central FD of the
/// rotated k energy (rotation explicitly `On` in the energy), fresh hcore,
/// fit and SCF at every displacement. Mutants must miss the FD too.
#[test]
fn rotated_k_force_matches_fd_of_the_rotated_k_energy() {
    fd_force(None);
}

/// (b) on the range-split build.
#[test]
#[ignore = "slow (~1.5 min release): 7 rotated k pipelines, range split"]
fn rotated_k_force_matches_fd_of_the_rotated_k_energy_split() {
    fd_force(Some(RangeSplit::default()));
}

fn fd_force(split: Option<RangeSplit>) {
    let n = [1, 1, 2];
    {
        let r = run(cell_h2(&H2_ATOMS), n, split, on(), DEFAULT_PREC);
        r.assert_rotates("1x1x2");
        let scf = r.scf(ExxDiv::None);
        let f = r.force(&scf, ExxDiv::None, None);
        let fo = r.force(&scf, ExxDiv::None, Some(KGradMutation::RotOff));
        let mut worst = 0.0_f64;
        let mut worst_off = 0.0_f64;
        let mut worst_mut = [0.0_f64; 7];
        let fm: Vec<KGradient> = FORCE_MUTANTS
            .iter()
            .map(|&m| r.force(&scf, ExxDiv::None, Some(m)))
            .collect();
        for (a, x) in [(0usize, 0usize), (0, 2), (1, 1)] {
            let mut p = H2_ATOMS;
            p[a][x] += FD_H;
            let ep = energy_at(&p, None, n, split);
            p[a][x] -= 2.0 * FD_H;
            let em = energy_at(&p, None, n, split);
            let fd = (ep - em) / (2.0 * FD_H);
            worst = worst.max((f.grad[(a, x)] - fd).abs());
            worst_off = worst_off.max((fo.grad[(a, x)] - fd).abs());
            for (k, g) in fm.iter().enumerate() {
                worst_mut[k] = worst_mut[k].max((g.grad[(a, x)] - fd).abs());
            }
        }
        eprintln!(
            "split {:?}: |analytic - FD| {worst:.2e} (RotOff {worst_off:.2e}); mutants {:?}",
            split.map(|s| s.lambda),
            worst_mut.map(|x| format!("{x:.1e}"))
        );
        assert!(worst < FD_F_BAR, "FD {worst:e}");
        for (m, w) in FORCE_MUTANTS.iter().zip(worst_mut) {
            assert!(w > MUTANT_FLOOR, "mutant {m:?} escaped the FD: {w:e}");
        }
    }
}

/// (b) PHYSICS ANCHOR, stress: rotated analytic dE/d(eps) == central FD of
/// the rotated k energy under strain (components xx, zz, xz, yz).
#[test]
#[ignore = "slow: 8 strained rotated k pipelines on H2 cc-pVDZ 1x1x2 per split"]
fn rotated_k_stress_matches_fd_of_the_rotated_k_energy() {
    let n = [1, 1, 2];
    for split in splits() {
        let r = run(cell_h2(&H2_ATOMS), n, split, on(), DEFAULT_PREC);
        let scf = r.scf(ExxDiv::None);
        let s = r.stress(&scf, ExxDiv::None, None);
        let mut worst = 0.0_f64;
        let mut worst_mut = [0.0_f64; 7];
        let sm: Vec<KStress> = STRESS_MUTANTS
            .iter()
            .map(|&m| r.stress(&scf, ExxDiv::None, Some(m)))
            .collect();
        for (i, j) in [(0usize, 0usize), (2, 2), (0, 2), (1, 2)] {
            let mut e = [[0.0; 3]; 3];
            e[i][j] = FD_H;
            let ep = energy_at(&H2_ATOMS, Some(&e), n, split);
            e[i][j] = -FD_H;
            let em = energy_at(&H2_ATOMS, Some(&e), n, split);
            let fd = (ep - em) / (2.0 * FD_H);
            worst = worst.max((s.de_deps[i][j] - fd).abs());
            for (k, g) in sm.iter().enumerate() {
                worst_mut[k] = worst_mut[k].max((g.de_deps[i][j] - fd).abs());
            }
        }
        eprintln!(
            "split {:?}: stress |analytic - FD| {worst:.2e}; mutants {:?}",
            split.map(|s| s.lambda),
            worst_mut.map(|x| format!("{x:.1e}"))
        );
        assert!(worst < FD_S_BAR, "FD {worst:e}");
        for (m, w) in STRESS_MUTANTS.iter().zip(worst_mut) {
            assert!(w > MUTANT_FLOOR, "mutant {m:?} escaped the FD: {w:e}");
        }
    }
}

// ------------------------------------------------------- (c)/(d) Gamma

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
        stability_external: None,
        df_jk: None,
        cosx_final: None,
        cosx_schedule: None,
        convergence: None,
        rohf_spin_focks: None,
    }
}

/// Explicit diag(n) supercell.
fn supercell(cell: &Cell, n: [usize; 3]) -> Cell {
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
    Cell::new(hydrogens(&atoms), lat).expect("supercell")
}

/// Bar of the k vs Gamma(-supercell) anchors, force and stress, total and
/// per part (hcore SR, fit SR 3-centre). Derived from the measured values
/// printed by the tests.
const SC_BAR: f64 = 1e-7;

/// The Gamma rotated force and stress of the explicit `n`-supercell of `cell`
/// (hcore and fit rotation explicitly `On`, same precisions as `r`) at the
/// k density of `r` unfolded to the supercell, compared with the k values of
/// `r` per cell: `[force total, force hcore, force fit, stress total, stress
/// hcore, stress fit]` misses.
fn gamma_anchor(r: &Run, n: [usize; 3], exx: ExxDiv, hp: f64, gp: f64) -> [f64; 6] {
    let nc = (n[0] * n[1] * n[2]) as f64;
    let sc = supercell(&r.cell, n);
    let prep = prep_for(&sc, &bset("cc-pvdz"));
    let aux = prep_for(&sc, &bset("cc-pvdz-ri"));
    let hcfg = hcfg(on(), hp);
    let hc = periodic_hcore(&sc, &prep, &hcfg).expect("hcore");
    assert_eq!(hc.sr_rotated_columns, 2 * nc as usize);
    let gdf = RsGdf::build_for_gradient(&sc, &prep, &aux, &hc.s, &gcfg(None, on(), gp).gdf)
        .expect("RsGdf");
    assert_eq!(
        gdf.timings().counter("rsgdf SR3 rotated columns"),
        Some(2 * nc as u64)
    );
    let scf_k = r.scf(exx);
    let gscf = gamma_restricted(unfold(&r.mesh, &scf_k.densities));
    let src = RsGdfGradSource {
        gdf: &gdf,
        aux: &aux,
        aux_jac: None,
    };
    let gf = gamma_rhf_gradient_rsgdf(
        &sc,
        &prep,
        &hcfg,
        &hc,
        &src,
        &gscf,
        exx,
        &GammaGradConfig {
            budget_bytes: Some(1 << 31),
            ..Default::default()
        },
    )
    .expect("gamma force");
    let gs = gamma_rhf_stress_rsgdf(
        &sc,
        &prep,
        &hcfg,
        &hc,
        &src,
        &gscf,
        exx,
        &GammaStressConfig {
            budget_bytes: Some(1 << 31),
            ..Default::default()
        },
    )
    .expect("gamma stress");
    let kf = r.force(&scf_k, exx, None);
    let ks = r.stress(&scf_k, exx, None);
    let na = r.cell.positions().len();
    let head = |m: &Array2<f64>| m.slice(ndarray::s![0..na, ..]).to_owned();
    let hcg = msum(&gs_parts_vsr(&gf, true), &gs_parts_vsr(&gf, false));
    let hck = msum(&kf.parts.vsr_basis, &kf.parts.vsr_nuc);
    let inv = 1.0 / nc;
    let sc3 = |m: &Mat3| -> Mat3 {
        let mut o = *m;
        for row in o.iter_mut() {
            for v in row.iter_mut() {
                *v *= inv;
            }
        }
        o
    };
    [
        max_abs_diff(&kf.grad, &head(&gf.grad)),
        max_abs_diff(&hck, &head(&hcg)),
        max_abs_diff(&kf.parts.fit_orb_sr, &head(&gf.parts.fit_orb_sr)),
        mdiff(&ks.de_deps, &sc3(&gs.de_deps)),
        mdiff(&ks.parts.vsr, &sc3(&gs.parts.vsr)),
        mdiff(&ks.parts.fit_j3_sr, &sc3(&gs.parts.fit_j3_sr)),
    ]
}

fn gs_parts_vsr(g: &GammaGradient, basis: bool) -> Array2<f64> {
    if basis {
        g.parts.vsr_basis.clone()
    } else {
        g.parts.vsr_nuc.clone()
    }
}

/// (c) 1x1x1 mesh == the Gamma rotated force and stress (both exxdiv).
#[test]
fn one_point_mesh_rotated_force_and_stress_are_the_gamma_ones() {
    let (hp, gp) = (1e-11, 1e-12);
    for (split, exx) in [(None, ExxDiv::None), (None, ExxDiv::Ewald)] {
        let r = run(cell_h2(&H2_ATOMS), [1, 1, 1], split, on(), (hp, gp));
        r.assert_rotates("1x1x1");
        let d = gamma_anchor(&r, [1, 1, 1], exx, hp, gp);
        eprintln!(
            "1x1x1 {exx:?}: force/stress vs Gamma (tot, hcore, fit) {:?}",
            d.map(|x| format!("{x:.2e}"))
        );
        assert!(d.iter().all(|x| *x < SC_BAR), "{d:?}");
    }
}

/// (d) k mesh 1x1x3 (non-TRIM) == the explicit supercell / N, rotation on
/// both sides, both exxdiv.
#[test]
#[ignore = "slow: Gamma RS-GDF force + stress on the 6-atom supercell, two exxdiv"]
fn rotated_k_mesh_force_and_stress_are_the_gamma_supercell() {
    let (hp, gp) = (1e-9, 1e-10);
    let n = [1, 1, 3];
    let r = run(cell_h2(&H2_ATOMS), n, None, on(), (hp, gp));
    r.assert_rotates("1x1x3");
    for exx in [ExxDiv::None, ExxDiv::Ewald] {
        let d = gamma_anchor(&r, n, exx, hp, gp);
        eprintln!(
            "1x1x3 {exx:?}: vs supercell/N (tot, hcore, fit) {:?}",
            d.map(|x| format!("{x:.2e}"))
        );
        assert!(d.iter().all(|x| *x < SC_BAR), "{d:?}");
    }
}

// ------------------------------------------------------------- (e), (g)

fn in_pool<R: Send>(n: usize, f: impl FnOnce() -> R + Send) -> R {
    rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .build()
        .expect("rayon pool")
        .install(f)
}

/// (e) force and stress (rotated, range split) are bitwise across threads.
#[test]
fn rotated_k_force_and_stress_are_bitwise_across_thread_counts() {
    let outs: Vec<(Array2<f64>, Mat3)> = [1usize, 3]
        .iter()
        .map(|&t| {
            in_pool(t, || {
                let r = run(
                    cell_h2(&H2_ATOMS),
                    [1, 1, 2],
                    Some(RangeSplit::default()),
                    on(),
                    DEFAULT_PREC,
                );
                let scf = r.scf(ExxDiv::None);
                (
                    r.force(&scf, ExxDiv::None, None).grad,
                    r.stress(&scf, ExxDiv::None, None).de_deps,
                )
            })
        })
        .collect();
    let bits = |a: &Array2<f64>, b: &Array2<f64>| {
        a.iter()
            .zip(b)
            .filter(|(x, y)| x.to_bits() != y.to_bits())
            .count()
    };
    assert_eq!(bits(&outs[0].0, &outs[1].0), 0, "force");
    for i in 0..3 {
        for j in 0..3 {
            assert_eq!(
                outs[0].1[i][j].to_bits(),
                outs[1].1[i][j].to_bits(),
                "stress"
            );
        }
    }
}

/// (g) a derivative call that disagrees with the energy build it differentiates
/// is refused by name, never silently run on another walk.
#[test]
fn inconsistent_rotation_requests_are_refused() {
    let rot = run(cell_h2(&H2_ATOMS), [1, 1, 1], None, on(), DEFAULT_PREC);
    let unrot = run(cell_h2(&H2_ATOMS), [1, 1, 1], None, OFF, DEFAULT_PREC);
    let mutant = SrColumnRotation::On(ColumnRotation {
        mutant: ColumnRotationMutant::FlipSign,
    });
    let scf_r = rot.scf(ExxDiv::None);
    let scf_u = unrot.scf(ExxDiv::None);
    let (h_on, h_off) = (hcfg(on(), DEFAULT_PREC.0), hcfg(OFF, DEFAULT_PREC.0));
    let (g_on, g_off) = (
        gcfg(None, on(), DEFAULT_PREC.1),
        gcfg(None, OFF, DEFAULT_PREC.1),
    );
    let h_mut = hcfg(mutant, DEFAULT_PREC.0);
    let g_mut = gcfg(None, mutant, DEFAULT_PREC.1);
    let cases: [(&str, &Run, &KScfResult, &PeriodicHcoreConfig, &KRsGdfConfig); 6] = [
        (
            "rotated build, hcore config Off",
            &rot,
            &scf_r,
            &h_off,
            &g_on,
        ),
        ("rotated build, fit config Off", &rot, &scf_r, &h_on, &g_off),
        (
            "unrotated build, explicit hcore On",
            &unrot,
            &scf_u,
            &h_on,
            &g_off,
        ),
        (
            "unrotated build, explicit fit On",
            &unrot,
            &scf_u,
            &h_off,
            &g_on,
        ),
        ("rotated build, hcore mutant", &rot, &scf_r, &h_mut, &g_on),
        ("rotated build, fit mutant", &rot, &scf_r, &h_on, &g_mut),
    ];
    for (name, r, scf, h, g) in cases {
        let f = r.try_force_cfg(h, g, scf, ExxDiv::None, None);
        let s = r.try_stress_cfg(h, g, scf, ExxDiv::None, None);
        for (what, e) in [("force", f.err()), ("stress", s.err())] {
            let msg = e
                .unwrap_or_else(|| panic!("{name}: {what} not refused"))
                .to_string();
            assert!(msg.contains("rotat"), "{name}: {what}: {msg}");
        }
    }
    // The consistent requests run, on both builds.
    rot.try_force_cfg(&h_on, &g_on, &scf_r, ExxDiv::None, None)
        .expect("rotated consistent");
    unrot
        .try_force_cfg(&h_off, &g_off, &scf_u, ExxDiv::None, None)
        .expect("unrotated consistent");
}

/// (f) `RotOff` (derivative walks unrotated although the energy rotated) is
/// blind at the default screen BY CONSTRUCTION (rotated and unrotated walks
/// are the same function up to the screening). At a LOOSE screen the two
/// walks drop different triplets: the unrotated rebuild of the fit then no
/// longer reproduces the energy's `B` and the existing `‖B‖` consistency
/// guard refuses it, while the rotated derivative stays the derivative of the
/// rotated (screened) energy. ARTIFACT hypothesis: if rotation were
/// irrelevant to the derivative, `RotOff` would run and FD would not tell ON
/// from OFF; if ON is the exact derivative of the rotated screened energy,
/// ON-vs-FD stays near the FD floor at the loose screen and OFF is refused.
#[test]
fn loose_screen_refuses_the_unrotated_rebuild_and_the_rotated_force_matches_fd() {
    let n = [1, 1, 2];
    let prec = (1e-5, 1e-5);
    let h = 1e-3;
    let r = run(cell_h2(&H2_ATOMS), n, None, on(), prec);
    r.assert_rotates("loose");
    let scf = r.scf(ExxDiv::None);
    let f_on = r.force(&scf, ExxDiv::None, None);
    let off = r.try_force(&scf, ExxDiv::None, Some(KGradMutation::RotOff));
    let msg = off.expect_err("unrotated rebuild accepted").to_string();
    eprintln!("loose screen {prec:?}: RotOff refused: {msg}");
    assert!(msg.contains("rebuilt fit differs"), "{msg}");
    let mut worst_on = 0.0_f64;
    for (a, x) in [(0usize, 0usize), (0, 2), (1, 1)] {
        let mut p = H2_ATOMS;
        p[a][x] += h;
        let ep = run(cell_h2(&p), n, None, on(), prec)
            .scf(ExxDiv::None)
            .energy;
        p[a][x] -= 2.0 * h;
        let em = run(cell_h2(&p), n, None, on(), prec)
            .scf(ExxDiv::None)
            .energy;
        let fd = (ep - em) / (2.0 * h);
        worst_on = worst_on.max((f_on.grad[(a, x)] - fd).abs());
    }
    eprintln!("loose screen {prec:?}: FD miss of the rotated force {worst_on:.2e} (h = {h})");
    // O(h^2) floor 8.7e-10 at h = 1e-4 scales to ~1e-7 at h = 1e-3; measured 2.3e-7.
    assert!(worst_on < 2e-6, "{worst_on:e}");
}

/// UHF shares the rotated walks: a closed-shell k-UHF at `(D/2, D/2)` gives
/// the k-RHF rotated force and stress (the spin sums and the per-spin
/// exchange `Z` collapse to the restricted ones), rotation on, range split.
#[test]
fn closed_shell_kuhf_rotated_force_and_stress_are_the_krhf_ones() {
    let split = Some(RangeSplit::default());
    let r = run(cell_h2(&H2_ATOMS), [1, 1, 2], split, on(), DEFAULT_PREC);
    r.assert_rotates("1x1x2");
    let exx = ExxDiv::None;
    let rs = r.scf(exx);
    let inj = KPointInjection {
        s: r.hk.s.clone(),
        h: r.hk.h.clone(),
        vnn: r.hk.enn,
        jk: Box::new(r.gdf.jk_builder_with_madelung(r.vm(exx))),
    };
    let mut us = solve_kuhf_injected(&r.cell, &r.mesh, &kscf_cfg(), inj, 1, 1).expect("k-UHF");
    assert!(us.converged);
    let half: Vec<Array2<Complex64>> = rs.densities.iter().map(|d| d.mapv(|z| z * 0.5)).collect();
    us.density_alpha = half.clone();
    us.density_beta = half;
    let fr = r.force(&rs, exx, None);
    let sr = r.stress(&rs, exx, None);
    let fu = kpoint_uhf_gradient(
        &r.cell,
        &r.prep,
        &r.mesh,
        &r.hcfg,
        &r.hk,
        KGradJk::RsGdf(r.src(&r.gcfg)),
        &us,
        exx,
        &KGradConfig::default(),
    )
    .expect("k-UHF force");
    let su = kpoint_uhf_stress(
        &r.cell,
        &r.prep,
        &r.mesh,
        &r.hcfg,
        &r.hk,
        KStressJk::RsGdf(r.src(&r.gcfg)),
        &us,
        exx,
        &KStressConfig::default(),
    )
    .expect("k-UHF stress");
    let df = max_abs_diff(&fr.grad, &fu.grad);
    let ds = mdiff(&sr.de_deps, &su.de_deps);
    eprintln!("closed-shell kUHF vs kRHF (rotated, split): force {df:.2e}, stress {ds:.2e}");
    assert!(df < 1e-9 && ds < 1e-9, "{df:e} {ds:e}");
}

/// The hcore-only fall-through (`rot_walk = None` in the derivative while the
/// energy rotated) is invisible to every numerical comparison at the default
/// screen, so the walks' ACTUAL rotated-column counters are the guard:
/// production reports the energy's counts, the hcore-only defect reports 0
/// on the hcore side with the fit still rotated, and `RotOff` reports 0 on
/// both. Force and stress. Cheap (1x1x1 mesh): the rotated-column plumbing
/// does not depend on the mesh.
#[test]
fn hcore_only_unrotated_walk_is_caught_by_the_walked_column_counters() {
    let r = run(cell_h2(&H2_ATOMS), [1, 1, 1], None, on(), DEFAULT_PREC);
    r.assert_rotates("1x1x1");
    let scf = r.scf(ExxDiv::None);
    let exx = ExxDiv::None;
    let fp = r.force(&scf, exx, None);
    let sp = r.stress(&scf, exx, None);
    let fh = r.force(&scf, exx, Some(KGradMutation::RotHcoreOff));
    let sh = r.stress(&scf, exx, Some(KStressMutation::RotHcoreOff));
    let fo = r.force(&scf, exx, Some(KGradMutation::RotOff));
    let so = r.stress(&scf, exx, Some(KStressMutation::RotOff));
    let fit = |c: Option<ferric_pbc::kgrad::KFitChecks>| c.map(|c| c.rotated_columns);
    // Production walks the energy's columns.
    assert!(r.walked_ok(fp.hcore_rotated_columns, fit(fp.fit_checks)));
    assert!(r.walked_ok(sp.hcore_rotated_columns, fit(sp.fit_checks)));
    // The hcore-only defect: hcore 0, fit still 2 -> the check REJECTS it.
    assert_eq!((fh.hcore_rotated_columns, fit(fh.fit_checks)), (0, Some(2)));
    assert_eq!((sh.hcore_rotated_columns, fit(sh.fit_checks)), (0, Some(2)));
    assert!(!r.walked_ok(fh.hcore_rotated_columns, fit(fh.fit_checks)));
    assert!(!r.walked_ok(sh.hcore_rotated_columns, fit(sh.fit_checks)));
    // Full RotOff: both sides 0.
    assert_eq!((fo.hcore_rotated_columns, fit(fo.fit_checks)), (0, Some(0)));
    assert_eq!((so.hcore_rotated_columns, fit(so.fit_checks)), (0, Some(0)));
    assert!(!r.walked_ok(fo.hcore_rotated_columns, fit(fo.fit_checks)));
    // The numbers the counters stand in for (default screen): the hcore-only
    // defect moves the force by only the screening gap.
    let gap = max_abs_diff(&fp.grad, &fh.grad);
    eprintln!("default screen, hcore-only RotOff: force gap {gap:.2e} (blind to any bar > this)");
}

/// Hcore-side gap at a LOOSE screen (hcore 1e-5, fit 1e-6): the rotated
/// hcore derivative walk vs the hcore-only unrotated one (`RotHcoreOff`; the
/// fit stays rotated, so the `‖B‖` guard does not refuse it) vs FD of the
/// rotated energy. The SR derivative walk uses the nucleus exponent
/// `min(GRAD_NUCLEUS_EXPONENT, cfg)` (the energy's 1e16 by default), so the
/// only gap between either walk and the derivative of the screened energy is
/// the screen; the FD miss of each is that gap, and what separates the two
/// walks is printed, not asserted beyond a sanity bound.
#[test]
fn loose_screen_hcore_side_gap_rotated_vs_unrotated_vs_fd() {
    let n = [1, 1, 2];
    let prec = (1e-5, 1e-6);
    let h = 1e-3;
    let r = run(cell_h2(&H2_ATOMS), n, None, on(), prec);
    r.assert_rotates("loose");
    let scf = r.scf(ExxDiv::None);
    let f_on = r.force(&scf, ExxDiv::None, None);
    let f_off = r.force(&scf, ExxDiv::None, Some(KGradMutation::RotHcoreOff));
    assert_eq!(f_off.hcore_rotated_columns, 0);
    let hc = |f: &KGradient| msum(&f.parts.vsr_basis, &f.parts.vsr_nuc);
    let gap = max_abs_diff(&hc(&f_on), &hc(&f_off));
    let gap_tot = max_abs_diff(&f_on.grad, &f_off.grad);
    let (mut m_on, mut m_off) = (0.0_f64, 0.0_f64);
    for (a, x) in [(0usize, 0usize), (0, 2), (1, 1)] {
        let mut p = H2_ATOMS;
        p[a][x] += h;
        let ep = run(cell_h2(&p), n, None, on(), prec)
            .scf(ExxDiv::None)
            .energy;
        p[a][x] -= 2.0 * h;
        let em = run(cell_h2(&p), n, None, on(), prec)
            .scf(ExxDiv::None)
            .energy;
        let fd = (ep - em) / (2.0 * h);
        m_on = m_on.max((f_on.grad[(a, x)] - fd).abs());
        m_off = m_off.max((f_off.grad[(a, x)] - fd).abs());
    }
    eprintln!(
        "loose {prec:?} hcore side: rotated-vs-unrotated hcore term {gap:.2e}, total {gap_tot:.2e}; \
         FD miss rotated {m_on:.2e}, unrotated hcore walk {m_off:.2e}"
    );
    assert!(gap_tot < 1e-3 && m_on < 2e-6, "{gap_tot:e} {m_on:e}");
}

// ----------------------------------------- (#421 c) triclinic and 2x2x1 meshes
//
// The anchors above are orthorhombic (cubic 5 Bohr) and one-dimensional
// (1x1x2, 1x1x3). Untested combination: a rotating basis on a TRICLINIC cell
// (the residue bins and `Tᵀ W[r] T` weights are indexed in lattice
// coordinates, the strain acts through the full non-orthogonal lattice) and a
// 2x2x1 mesh (two mesh directions: the residue bucket of a pair image is a
// 2-D index, so a flattening or axis-order slip that a 1-D mesh cannot see).
//
// ARTIFACT hypothesis (before measuring): a defect confined to the combination
// would be (i) the residue index of the forward-transformed weights computed
// from Cartesian rather than lattice coordinates (invisible on a cubic cell:
// the two agree), (ii) the T-transform applied per residue in a different
// bucket order than the energy's back-transform on a mesh with n_1 > 1 (a 1-D
// mesh has one bucket axis), (iii) `unfold`/supercell mismatch from an
// axis-order slip. Each would be an O(0.1..1) miss against the supercell and
// FD anchors. All of those are independent constructions of the rotated
// derivative: explicit supercell (rotation on both sides) and FD of the
// ROTATED energy. If correct: the same bars as the 1-D anchors.
//
// WEAK SPOT, asserted with its measured size: every n_i <= 2 mesh is TRIM, so
// `RotSwapPair` (a transposed bin) is nearly the same bin. Measured on the
// triclinic cell: 2x2x1 force 4.72e-4 / stress 4.70e-3 (RotFitWrongResidue
// 9.2e-2 / 2.5e-1); non-TRIM 1x1x3 force 7.33e-3 / stress 2.01e-2. It is
// asserted (> 1e-4 on 2x2x1, > 1e-3 on the triclinic 1x1x3 supercell test).
// Also caught by `rotation_mutants_miss_the_rotated_force` (cubic 1x1x2,
// default), `..._stress` (cubic 1x1x3, ignored) and the ECP RS-GDF stress
// loop (1.12e-1). NOT covered: the triclinic 1x1x1 anchor runs only RotDropT;
// fit-acting FORCE mutants with an ECP are never run.

fn cell_tri() -> Cell {
    Cell::new(hydrogens(&H2_ATOMS), TRI_A).expect("triclinic cell")
}

/// Bar of the 1x1x1-vs-Gamma anchors of this section. MEASURED (triclinic,
/// both exxdiv): force total 3.0e-12, hcore 0 (bitwise), fit 1.2e-14; stress
/// total 3.9e-14, fit 4.2e-14; 25x above. The test also runs `RotDropT`
/// (force and stress) and asserts it moves the total by more than
/// `ONE_POINT_MUTANT_FLOOR`.
const ONE_POINT_BAR: f64 = 1e-10;
/// `RotDropT` at triclinic 1x1x1 must move the total by at least this (the
/// 2x2x1 triclinic value is 4.9e-1 force / 2.2e-1 stress).
const ONE_POINT_MUTANT_FLOOR: f64 = 1e-3;
/// `RotSwapPair` floors (derived from the measured values in the header).
const SWAP_PAIR_FLOOR_2X2X1: f64 = 1e-4;
const SWAP_PAIR_FLOOR_1X1X3: f64 = 1e-3;

/// (c) on the triclinic cell: 1x1x1 mesh == the Gamma rotated force and
/// stress, both exxdiv (hcore, fit and total, per part).
#[test]
#[ignore = "slow (~130 s release at load 10; 46 s before the RotDropT mutant was added): triclinic rotated 1x1x1 RS-GDF, two exxdiv anchors + one mutant"]
fn triclinic_one_point_mesh_rotated_force_and_stress_are_the_gamma_ones() {
    let (hp, gp) = (1e-11, 1e-12);
    let r = run(cell_tri(), [1, 1, 1], None, on(), (hp, gp));
    r.assert_rotates("triclinic 1x1x1");
    for exx in [ExxDiv::None, ExxDiv::Ewald] {
        let d = gamma_anchor(&r, [1, 1, 1], exx, hp, gp);
        eprintln!(
            "triclinic 1x1x1 {exx:?}: force/stress vs Gamma (tot, hcore, fit) {:?}",
            d.map(|x| format!("{x:.2e}"))
        );
        assert!(d.iter().all(|x| *x < ONE_POINT_BAR), "{d:?}");
    }
    let scf = r.scf(ExxDiv::None);
    let f0 = r.force(&scf, ExxDiv::None, None);
    let s0 = r.stress(&scf, ExxDiv::None, None);
    let fm = r.force(&scf, ExxDiv::None, Some(KGradMutation::RotDropT));
    let sm = r.stress(&scf, ExxDiv::None, Some(KStressMutation::RotDropT));
    let (mf, ms) = (
        max_abs_diff(&fm.grad, &f0.grad),
        mdiff(&sm.de_deps, &s0.de_deps),
    );
    eprintln!("triclinic 1x1x1 RotDropT: force moves {mf:.2e}, stress moves {ms:.2e}");
    assert!(mf > ONE_POINT_MUTANT_FLOOR && ms > ONE_POINT_MUTANT_FLOOR);
}

/// (d) on the triclinic cell, non-TRIM 1x1x3 mesh == supercell / N with the
/// rotation on both sides, both exxdiv.
#[test]
#[ignore = "slow (the three tests total 650 s release at load 10): Gamma RS-GDF force + stress on the 6-atom triclinic supercell, two exxdiv"]
fn triclinic_rotated_1x1x3_mesh_force_and_stress_are_the_gamma_supercell() {
    let (hp, gp) = (1e-9, 1e-10);
    let n = [1, 1, 3];
    let r = run(cell_tri(), n, None, on(), (hp, gp));
    r.assert_rotates("triclinic 1x1x3");
    for exx in [ExxDiv::None, ExxDiv::Ewald] {
        let d = gamma_anchor(&r, n, exx, hp, gp);
        eprintln!(
            "triclinic 1x1x3 {exx:?}: vs supercell/N (tot, hcore, fit) {:?}",
            d.map(|x| format!("{x:.2e}"))
        );
        assert!(d.iter().all(|x| *x < SC_BAR), "{d:?}");
    }
    // The weakest mutant on the non-TRIM triclinic mesh.
    let scf = r.scf(ExxDiv::None);
    let f0 = r.force(&scf, ExxDiv::None, None);
    let s0 = r.stress(&scf, ExxDiv::None, None);
    let fm = r.force(&scf, ExxDiv::None, Some(KGradMutation::RotSwapPair));
    let sm = r.stress(&scf, ExxDiv::None, Some(KStressMutation::RotSwapPair));
    let (mf, ms) = (
        max_abs_diff(&fm.grad, &f0.grad),
        mdiff(&sm.de_deps, &s0.de_deps),
    );
    eprintln!("triclinic 1x1x3 RotSwapPair: force moves {mf:.2e}, stress moves {ms:.2e}");
    assert!(mf > SWAP_PAIR_FLOOR_1X1X3 && ms > SWAP_PAIR_FLOOR_1X1X3);
}

/// (d) on a 2x2x1 mesh (triclinic and cubic cells): k mesh == the explicit
/// 2x2x1 supercell / N, rotation on both sides, exxdiv none.
#[test]
#[ignore = "slow (the three tests total 650 s release at load 10): Gamma RS-GDF force + stress on the 8-atom supercell, triclinic and cubic"]
fn rotated_2x2x1_mesh_force_and_stress_are_the_gamma_supercell() {
    let (hp, gp) = (1e-9, 1e-10);
    let n = [2, 2, 1];
    for (name, cell) in [("triclinic", cell_tri()), ("cubic", cell_h2(&H2_ATOMS))] {
        let r = run(cell, n, None, on(), (hp, gp));
        r.assert_rotates(name);
        let d = gamma_anchor(&r, n, ExxDiv::None, hp, gp);
        eprintln!(
            "{name} 2x2x1: vs supercell/N (tot, hcore, fit) {:?}",
            d.map(|x| format!("{x:.2e}"))
        );
        assert!(d.iter().all(|x| *x < SC_BAR), "{name}: {d:?}");
    }
}

fn energy_at_cell(cell: &Cell, n: [usize; 3]) -> f64 {
    run(cell.clone(), n, None, on(), DEFAULT_PREC)
        .scf(ExxDiv::None)
        .energy
}

fn displaced(cell: &Cell, a: usize, x: usize, h: f64) -> Cell {
    let mut p: Vec<[f64; 3]> = cell
        .positions()
        .iter()
        .map(|q| [q[0], q[1], q[2]])
        .collect();
    p[a][x] += h;
    Cell::new(hydrogens(&p), *cell.lattice()).expect("displaced cell")
}

/// (b) on a 2x2x1 mesh of the triclinic cell: rotated analytic force (H x, H
/// z, the second atom's y) and stress (xx, zz, xy, yz, xz) vs central FD of
/// the ROTATED k energy, every displaced / strained build rotated; the
/// production-wired stress and force mutants must move the total; on this
/// TRIM mesh `RotSwapPair` is weak (4.72e-4 force, 4.70e-3 stress) and is
/// asserted against `SWAP_PAIR_FLOOR_2X2X1`, the others against
/// `MUTANT_FLOOR`.
#[test]
#[ignore = "slow (the three tests total 650 s release at load 10): 1 + 16 rotated 2x2x1 triclinic k pipelines, 14 mutant evaluations"]
fn rotated_k_force_and_stress_match_fd_on_2x2x1_triclinic() {
    let n = [2, 2, 1];
    let cell = cell_tri();
    let comps_f = [(0usize, 0usize), (0, 2), (1, 1)];
    let comps_s = [(0usize, 0usize), (2, 2), (0, 1), (1, 2), (0, 2)];
    let mut fd_f = Vec::new();
    for &(a, x) in &comps_f {
        let ep = energy_at_cell(&displaced(&cell, a, x, FD_H), n);
        let em = energy_at_cell(&displaced(&cell, a, x, -FD_H), n);
        fd_f.push((ep - em) / (2.0 * FD_H));
    }
    let mut fd_s = Vec::new();
    for &(i, j) in &comps_s {
        let mut e = [[0.0; 3]; 3];
        e[i][j] = FD_H;
        let ep = energy_at_cell(&cell.strained(&e).unwrap(), n);
        e[i][j] = -FD_H;
        let em = energy_at_cell(&cell.strained(&e).unwrap(), n);
        fd_s.push((ep - em) / (2.0 * FD_H));
    }
    let r = run(cell, n, None, on(), DEFAULT_PREC);
    r.assert_rotates("triclinic 2x2x1");
    let scf = r.scf(ExxDiv::None);
    let f = r.force(&scf, ExxDiv::None, None);
    let s = r.stress(&scf, ExxDiv::None, None);
    let mut worst_f = 0.0_f64;
    for (k, &(a, x)) in comps_f.iter().enumerate() {
        let d = (f.grad[(a, x)] - fd_f[k]).abs();
        assert!(
            d.is_finite(),
            "force component ({a},{x}): non-finite difference"
        );
        worst_f = worst_f.max(d);
    }
    let mut worst_s = 0.0_f64;
    for (k, &(i, j)) in comps_s.iter().enumerate() {
        let d = (s.de_deps[i][j] - fd_s[k]).abs();
        assert!(
            d.is_finite(),
            "stress component ({i},{j}): non-finite difference"
        );
        worst_s = worst_s.max(d);
    }
    eprintln!("triclinic 2x2x1: |analytic - FD| force {worst_f:.2e}, stress {worst_s:.2e}");
    assert!(worst_f < FD_F_BAR, "force FD {worst_f:e}");
    assert!(worst_s < FD_S_BAR, "stress FD {worst_s:e}");
    for m in FORCE_MUTANTS {
        let g = r.force(&scf, ExxDiv::None, Some(m));
        let moved = max_abs_diff(&g.grad, &f.grad);
        eprintln!("  force mutant {m:?}: total moves {moved:.2e}");
        assert!(
            moved
                > if m == KGradMutation::RotSwapPair {
                    SWAP_PAIR_FLOOR_2X2X1
                } else {
                    MUTANT_FLOOR
                },
            "{m:?} escaped: {moved:e}"
        );
    }
    for m in STRESS_MUTANTS {
        let g = r.stress(&scf, ExxDiv::None, Some(m));
        let moved = mdiff(&g.de_deps, &s.de_deps);
        eprintln!("  stress mutant {m:?}: total moves {moved:.2e}");
        assert!(
            moved
                > if m == KStressMutation::RotSwapPair {
                    SWAP_PAIR_FLOOR_2X2X1
                } else {
                    MUTANT_FLOOR
                },
            "{m:?} escaped: {moved:e}"
        );
    }
}

// ------------------------------------- (#421 B) shifted (Monkhorst-Pack) mesh
//
// A Monkhorst-Pack even mesh (1x1x2: k = +-b3/4, residue modulus 4) is a
// twisted-boundary supercell, so there is NO explicit-supercell anchor; the
// rotated energy's own FD is the physics anchor. Untested combination before
// this section: the rotation (residue bins of `Tt W[r] T`, indexed through the
// mesh's residue modulus) on a mesh whose residue modulus is NOT the mesh
// size (4 vs 2 along z).
//
// ARTIFACT hypothesis (before measuring). If correct: rotated-vs-unrotated
// walks agree at the screen (D_BAR), analytic == FD at the FD floor
// (FD_F_BAR / FD_S_BAR) and every Rot* mutant misses by >= MUTANT_FLOOR. If
// the rotated weights were binned with the mesh size (2) instead of the
// residue modulus (4), or the phases were Gamma-centred, the analytic would
// miss the FD by O(0.01..1) on the rotated columns while the rotated-vs-
// unrotated anchor stays at the screen only if BOTH walks share the slip
// (the FD is the discriminating one).

// MEASURED (release): rotated vs unrotated walks on MP 1x1x2: force total
// 3.0e-10, stress total 8.9e-9 (D_BAR 1e-7, 11x). FD (h = 1e-4): force 2.35e-9
// (bar 2.5e-8, 10.6x), stress 5.5e-9 (bar 5e-8, 9.1x). FD misses of the
// mutants (force / stress): RotDropT 3.3e-1 / 4.3e-1, RotUseT 3.2e-1 / 4.4e-1,
// RotOneSided 1.5e-1 / 1.9e-1, RotFlipSign 3.9e-1 / 5.3e-1, RotSwapPair
// 2.3e-3 / 9.3e-3, RotFitWrongResidue 7.1e-2 / 1.3e-1, RotHcoreWrongResidue
// 5.5e-1 / 7.8e-1, NoPhase 1.5e-2 / 2.4e-1, ConjD 1.1e-2 / 2.2e-2 (floor
// MUTANT_FLOOR 1e-6, smallest 2.3e-3). RotOff is BLIND by design at the
// default screen (see (a)). Test mutation: rotation Off in `mp_cell_run`
// fails `assert_rotates` ("hcore did not rotate").

fn mp_cell_run() -> Run {
    run_c(
        cell_h2(&H2_ATOMS),
        [1, 1, 2],
        MeshCentring::MonkhorstPack,
        None,
        on(),
        DEFAULT_PREC,
    )
}

fn mp_energy(pos: &[[f64; 3]], strain: Option<&Mat3>) -> f64 {
    let mut cell = cell_h2(pos);
    if let Some(e) = strain {
        cell = cell.strained(e).expect("strained");
    }
    run_c(
        cell,
        [1, 1, 2],
        MeshCentring::MonkhorstPack,
        None,
        on(),
        DEFAULT_PREC,
    )
    .scf(ExxDiv::None)
    .energy
}

/// Anchor (exactness, run first): the rotated walks equal the unrotated ones
/// of the same energy and density on the MP mesh, force and stress, per part.
#[test]
fn mp_mesh_rotated_walks_equal_the_unrotated_walks() {
    let r = mp_cell_run();
    r.assert_rotates("MP 1x1x2");
    assert_eq!(r.mesh.residue_moduli(), [1, 1, 4], "not a shifted mesh");
    let scf = r.scf(ExxDiv::None);
    let f_on = r.force(&scf, ExxDiv::None, None);
    let f_off = r.force(&scf, ExxDiv::None, Some(KGradMutation::RotOff));
    let s_on = r.stress(&scf, ExxDiv::None, None);
    let s_off = r.stress(&scf, ExxDiv::None, Some(KStressMutation::RotOff));
    let hc = |f: &KGradient| msum(&f.parts.vsr_basis, &f.parts.vsr_nuc);
    let d = [
        max_abs_diff(&f_on.grad, &f_off.grad),
        max_abs_diff(&hc(&f_on), &hc(&f_off)),
        max_abs_diff(&f_on.parts.fit_orb_sr, &f_off.parts.fit_orb_sr),
        mdiff(&s_on.de_deps, &s_off.de_deps),
        mdiff(&s_on.parts.vsr, &s_off.parts.vsr),
        mdiff(&s_on.parts.fit_j3_sr, &s_off.parts.fit_j3_sr),
    ];
    eprintln!(
        "MP 1x1x2 rotated vs unrotated walks (F tot, hcore, fit; S tot, hcore, fit): {:?}",
        d.map(|x| format!("{x:.2e}"))
    );
    assert!(d.iter().all(|x| *x < D_BAR), "{d:?}");
    assert!(max_abs(&f_off.parts.fit_orb_sr) > 1e-3 && max_abs(&hc(&f_off)) > 1e-3);
    assert!(mmax(&s_off.parts.vsr) > 1e-3 && mmax(&s_off.parts.fit_j3_sr) > 1e-3);
}

/// Physics: rotated analytic force and stress on the MP 1x1x2 mesh == central
/// FD of the rotated MP k energy (rotation on in every displaced build); every
/// production-wired rotation mutant, and the mesh-sensitive `NoPhase` and
/// `ConjD` (NOT identities on a non-TRIM mesh), must miss the FD.
#[test]
#[ignore = "slow (~3 min release): 1 + 14 rotated MP k pipelines, 16 mutant evaluations"]
fn rotated_mp_mesh_force_and_stress_match_fd_of_the_rotated_k_energy() {
    let comps_f = [(0usize, 0usize), (0, 2), (1, 1)];
    let comps_s = [(0usize, 0usize), (2, 2), (0, 2), (1, 2)];
    let fd_f: Vec<f64> = comps_f
        .iter()
        .map(|&(a, x)| {
            let mut p = H2_ATOMS;
            p[a][x] += FD_H;
            let ep = mp_energy(&p, None);
            p[a][x] -= 2.0 * FD_H;
            (ep - mp_energy(&p, None)) / (2.0 * FD_H)
        })
        .collect();
    let fd_s: Vec<f64> = comps_s
        .iter()
        .map(|&(i, j)| {
            let mut e = [[0.0; 3]; 3];
            e[i][j] = FD_H;
            let ep = mp_energy(&H2_ATOMS, Some(&e));
            e[i][j] = -FD_H;
            (ep - mp_energy(&H2_ATOMS, Some(&e))) / (2.0 * FD_H)
        })
        .collect();
    let r = mp_cell_run();
    r.assert_rotates("MP 1x1x2");
    assert_eq!(r.mesh.residue_moduli(), [1, 1, 4]);
    let scf = r.scf(ExxDiv::None);
    let f = r.force(&scf, ExxDiv::None, None);
    let s = r.stress(&scf, ExxDiv::None, None);
    let miss_f = |g: &KGradient| {
        comps_f.iter().zip(&fd_f).fold(0.0_f64, |w, (&(a, x), fd)| {
            w.max((g.grad[(a, x)] - fd).abs())
        })
    };
    let miss_s = |g: &KStress| {
        comps_s.iter().zip(&fd_s).fold(0.0_f64, |w, (&(i, j), fd)| {
            w.max((g.de_deps[i][j] - fd).abs())
        })
    };
    let (wf, ws) = (miss_f(&f), miss_s(&s));
    eprintln!("MP 1x1x2 rotated: |analytic - FD| force {wf:.2e}, stress {ws:.2e}");
    assert!(wf.is_finite() && ws.is_finite());
    assert!(wf < MP_FD_F_BAR, "force FD {wf:e}");
    assert!(ws < MP_FD_S_BAR, "stress FD {ws:e}");
    let mut fm: Vec<KGradMutation> = FORCE_MUTANTS.to_vec();
    fm.extend([KGradMutation::NoPhase, KGradMutation::ConjD]);
    for m in fm {
        let w = miss_f(&r.force(&scf, ExxDiv::None, Some(m)));
        eprintln!("  force mutant {m:?}: FD miss {w:.2e}");
        assert!(w > MUTANT_FLOOR, "{m:?} escaped the FD: {w:e}");
    }
    let mut sm: Vec<KStressMutation> = STRESS_MUTANTS.to_vec();
    sm.extend([KStressMutation::NoPhase, KStressMutation::ConjD]);
    for m in sm {
        let w = miss_s(&r.stress(&scf, ExxDiv::None, Some(m)));
        eprintln!("  stress mutant {m:?}: FD miss {w:.2e}");
        assert!(w > MUTANT_FLOOR, "{m:?} escaped the FD: {w:e}");
    }
}

const MP_FD_F_BAR: f64 = 2.5e-8;
const MP_FD_S_BAR: f64 = 5e-8;
