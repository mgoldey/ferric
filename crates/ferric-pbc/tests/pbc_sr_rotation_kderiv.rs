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
//! screening precision (~1e-13..1e-11) and shrinking with it; (b) at the
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
    kpoint_rhf_gradient, KGradConfig, KGradJk, KGradMutation, KGradient, KRsGdfGradSource,
};
use ferric_pbc::kpts::KPointMesh;
use ferric_pbc::kscf::{solve_krhf_injected, KPointInjection, KScfConfig, KScfResult};
use ferric_pbc::kstress::{kpoint_rhf_stress, KStress, KStressConfig, KStressJk, KStressMutation};
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
    (hp, gp): (f64, f64),
) -> Run {
    let prep = prep_for(&cell, &bset("cc-pvdz"));
    let aux = prep_for(&cell, &bset("cc-pvdz-ri"));
    let mesh = KPointMesh::gamma_centred(&cell, n).expect("mesh");
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
        self.try_force(scf, exx, m).expect("k force")
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
        self.try_stress(scf, exx, m).expect("k stress")
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
const D_BAR: f64 = 1e-7;

/// (a) EXACTNESS ANCHOR, forces and stress, per part. Prints the measured
/// sizes (the bar is derived from them in the test doc).
#[test]
fn rotated_k_derivative_walks_equal_the_unrotated_walks_per_part() {
    let n = [1, 1, 3];
    for (split, exx) in [
        (None, ExxDiv::None),
        (Some(RangeSplit::default()), ExxDiv::Ewald),
    ] {
        let r = run(cell_h2(&H2_ATOMS), n, split, on(), DEFAULT_PREC);
        r.assert_rotates("1x1x3");
        {
            let tag = format!("1x1x3 split {:?} {exx:?}", split.map(|s| s.lambda));
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

/// (f): every mutant misses the rotated derivative, per part where it acts.
#[test]
fn rotation_mutants_miss_the_rotated_force_and_stress() {
    let r = run(cell_h2(&H2_ATOMS), [1, 1, 3], None, on(), DEFAULT_PREC);
    r.assert_rotates("1x1x3");
    let scf = r.scf(ExxDiv::None);
    let f0 = r.force(&scf, ExxDiv::None, None);
    let s0 = r.stress(&scf, ExxDiv::None, None);
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

/// Measured O(h^2) FD floor (printed); bars derived in the assertions' doc.
const FD_F_BAR: f64 = 1e-8;
const FD_S_BAR: f64 = 5e-8;
const FD_H: f64 = 1e-4;

/// (b) PHYSICS ANCHOR, forces: rotated analytic force == central FD of the
/// rotated k energy (rotation explicitly `On` in the energy), fresh hcore,
/// fit and SCF at every displacement. Mutants must miss the FD too.
#[test]
fn rotated_k_force_matches_fd_of_the_rotated_k_energy() {
    let n = [1, 1, 2];
    for split in splits() {
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
    let rot = run(cell_h2(&H2_ATOMS), [1, 1, 2], None, on(), DEFAULT_PREC);
    let unrot = run(cell_h2(&H2_ATOMS), [1, 1, 2], None, OFF, DEFAULT_PREC);
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
    let msg = off.err().expect("unrotated rebuild accepted").to_string();
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
