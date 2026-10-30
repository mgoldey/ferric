//! Shared harness of `pbc_sr_rotation.rs` (per-commit) and
//! `validation_pbc_sr_rotation.rs` (weekly): fixtures, bars, helpers and the
//! parameterised bodies of the force / stress anchors that moved weekly
//! (their per-commit canaries call the same bodies on a smaller case set).
//! Construction, bars and their derivations: `pbc_sr_rotation.rs`.
#![allow(dead_code)]

use crate::common::*;
use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_pbc::dense_aft::ExxDiv;
use ferric_pbc::grad::{
    gamma_rhf_gradient_rsgdf, GammaGradConfig, GammaGradient, GradMutation, RsGdfGradSource,
};
use ferric_pbc::hcore::{periodic_hcore, PeriodicHcore, PeriodicHcoreConfig};
use ferric_pbc::lattice::Cell;
use ferric_pbc::rsgdf::{RangeSplit, RsGdf, RsGdfConfig};
use ferric_pbc::sr_rotation::{ColumnRotation, ColumnRotationMutant, SrColumnRotation};
use ferric_pbc::stress::{
    gamma_rhf_stress_rsgdf, GammaStress, GammaStressConfig, Mat3, StressMutation,
};
use ferric_scf::result::ScfResult;
use ndarray::Array2;

pub const THREADS: [usize; 3] = [1, 2, 6];
pub const HCORE_OMEGA: f64 = 0.8;
/// (b): the tight truncation of the covariance anchor.
pub const P_TIGHT: f64 = 1e-16;
/// (b): relative bar of the covariance anchor (module doc derivation).
pub const TOL_COV: f64 = 1e-13;
/// (f): smallest relative miss a mutant must show.
pub const MUTANT_FLOOR: f64 = 1e-6;
/// (c): RHF energy bar (Ha).
pub const E_BAR: f64 = 1e-10;
/// (h): rotated vs unrotated force (Ha/Bohr) and stress dE/dε (Ha). Measured
/// (release, 3 threads): force 4.9e-15 .. 7.4e-14, stress 7.6e-14 .. 1.0e-12;
/// the smallest forward-transform mutant miss is 1.4e-1 (force) / 3.7e-1
/// (stress), so the bar sits 3 decades above the worst correct value and 9
/// below the smallest mutant.
pub const D_BAR: f64 = 1e-10;
/// (h): smallest miss a forward-transform mutant must show, per term.
pub const D_MUTANT_FLOOR: f64 = 1e-3;
/// (h): analytic vs central FD (h = 1e-4) of the ROTATED energy. Measured
/// force 1.0e-9 (rotated) and 1.1e-9 (unrotated control: the O(h²) FD
/// truncation, it scales ×4..7 at h = 2e-4), stress 4.2e-9.
pub const FD_F_BAR: f64 = 1e-8;
pub const FD_S_BAR: f64 = 5e-8;

pub const WATER: &str = "3
water
O  0.0000  0.0000  0.1173
H  0.0000  0.7572 -0.4692
H  0.0000 -0.7572 -0.4692
";

pub const FORMALDEHYDE: &str = "4
formaldehyde
C  0.0000  0.0000 -0.5290
O  0.0000  0.0000  0.6760
H  0.0000  0.9430 -1.1160
H  0.0000 -0.9430 -1.1160
";

/// Runs `f` inside a fresh rayon pool of `n` threads and returns its result.
pub fn in_pool<R: Send>(n: usize, f: impl FnOnce() -> R + Send) -> R {
    rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .build()
        .expect("rayon pool")
        .install(f)
}

pub fn cell_of(xyz: &str, a: f64) -> Cell {
    Cell::new(Molecule::parse_xyz(xyz, 0, 1).expect("xyz"), cubic(a)).expect("cell")
}

pub fn bundled(cell: &Cell, name: &str) -> PreparedBasis {
    prep_for(cell, &basis::bundled(name).expect("bundled basis"))
}

pub fn rotation(m: ColumnRotationMutant) -> SrColumnRotation {
    SrColumnRotation::On(ColumnRotation { mutant: m })
}

pub fn production() -> SrColumnRotation {
    SrColumnRotation::on()
}

/// The unrotated reference (explicit: the default `Auto` rotates the Gamma
/// energy builds).
pub const OFF: SrColumnRotation = SrColumnRotation::Off;

pub fn gdf_cfg(split: Option<RangeSplit>, rot: SrColumnRotation, precision: f64) -> RsGdfConfig {
    RsGdfConfig {
        omega: 1.0,
        precision,
        exxdiv: ExxDiv::None,
        budget_bytes: Some(1 << 30),
        range_split: split,
        sr_column_rotation: rot,
        ..Default::default()
    }
}

pub fn hcore_cfg(rot: SrColumnRotation, precision: f64) -> PeriodicHcoreConfig {
    PeriodicHcoreConfig {
        precision,
        sr_column_rotation: rot,
        ..PeriodicHcoreConfig::with_omega(HCORE_OMEGA)
    }
}

/// `(RsGdf, symmetrised J3 before the metric solve)`.
pub fn gdf_j3(
    cell: &Cell,
    prep: &PreparedBasis,
    aux: &PreparedBasis,
    s: &Array2<f64>,
    cfg: &RsGdfConfig,
) -> (RsGdf, Array2<f64>) {
    let (g, parts) = RsGdf::build_with_fit_parts(cell, prep, aux, s, cfg).expect("rsgdf");
    (g, parts.j3)
}

/// The number of elements whose bit patterns differ; panics if the shapes
/// differ.
pub fn bit_diffs(a: &Array2<f64>, b: &Array2<f64>) -> usize {
    assert_eq!(a.dim(), b.dim());
    a.iter()
        .zip(b.iter())
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count()
}

/// Asserts that `a` and `b` agree in every bit (see [`bit_diffs`]); `what`
/// labels a failure.
pub fn assert_bitwise(a: &Array2<f64>, b: &Array2<f64>, what: &str) {
    let d = bit_diffs(a, b);
    assert_eq!(d, 0, "{what}: {d} of {} elements differ in bits", a.len());
}

/// Largest element-wise `|m|` (0 for an empty matrix).
pub fn max_abs(m: &Array2<f64>) -> f64 {
    m.iter().fold(0.0_f64, |a, x| a.max(x.abs()))
}

/// `max|a − b| / max|b|`.
pub fn rel(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    let s = max_abs(b);
    assert!(
        s > 0.0,
        "reference is identically zero (vacuous comparison)"
    );
    max_abs_diff(a, b) / s
}

pub fn energy(cell: &Cell, prep: &PreparedBasis, hc: &PeriodicHcore, g: &RsGdf) -> f64 {
    gamma_rhf_jk(
        cell,
        prep,
        hc,
        Box::new(g.j_builder()),
        Box::new(g.k_builder()),
    )
    .energy
}

// ------------------------------------------------------------------ (a)

// ------------------------------------------------------- (b) and (e)

// ------------------------------------------------------------------ (c)

// ------------------------------------------------------------------ (d)

// ------------------------------------------------------------------ (f)

// ------------------------------------------------------------ refusals

// ------------------------------------------- default resolution (Auto)

// ------------------------------------------------- forces and stress (h)

/// Off-symmetry water (every force and every stress component non-zero).
pub const WATER_TILT: &str = "3
water
O  0.0300  0.0200  0.1173
H  0.0400  0.7572 -0.4692
H  0.0900 -0.7100 -0.4300
";

/// Off-axis H2 (Bohr), a = 5: the cheapest cell with something to rotate
/// (one s column per H), nine non-zero stress components.
pub const H2_OFF_AXIS: [[f64; 3]; 2] = [[0.3, 0.2, 0.1], [0.35, 0.12, 1.5]];

pub fn h2_off_axis(pos: &[[f64; 3]]) -> Cell {
    Cell::new(hydrogens(pos), cubic(5.0)).expect("cell")
}

pub fn water_tilt() -> Cell {
    cell_of(WATER_TILT, 8.0)
}

/// SCF + gradient-capable `RsGdf` + hcore of one cell at one rotation
/// request (`rot` goes to BOTH the hcore and the fit, like every driver).
pub struct Fx {
    pub cell: Cell,
    pub prep: PreparedBasis,
    pub aux: PreparedBasis,
    pub hc: PeriodicHcore,
    pub hcfg: PeriodicHcoreConfig,
    pub gdf: RsGdf,
    pub scf: ScfResult,
}

pub fn fixture(
    cell: Cell,
    prep: PreparedBasis,
    rot: SrColumnRotation,
    split: Option<RangeSplit>,
) -> Fx {
    let aux = bundled(&cell, "cc-pvdz-ri");
    let hcfg = hcore_cfg(rot, 1e-14);
    let hc = periodic_hcore(&cell, &prep, &hcfg).expect("hcore");
    let gdf = RsGdf::build_for_gradient(&cell, &prep, &aux, &hc.s, &gdf_cfg(split, rot, 1e-13))
        .expect("build_for_gradient");
    let scf = gamma_rhf_jk(
        &cell,
        &prep,
        &hc,
        Box::new(gdf.j_builder()),
        Box::new(gdf.k_builder()),
    );
    Fx {
        cell,
        prep,
        aux,
        hc,
        hcfg,
        gdf,
        scf,
    }
}

pub fn cc_fixture(cell: Cell, rot: SrColumnRotation, split: Option<RangeSplit>) -> Fx {
    let prep = bundled(&cell, "cc-pvdz");
    fixture(cell, prep, rot, split)
}

pub fn gcfg(m: Option<GradMutation>) -> GammaGradConfig {
    GammaGradConfig {
        mutation: m,
        budget_bytes: Some(1 << 30),
        ..Default::default()
    }
}

pub fn scfg(m: Option<StressMutation>) -> GammaStressConfig {
    GammaStressConfig {
        mutation: m,
        budget_bytes: Some(1 << 30),
        ..Default::default()
    }
}

pub fn try_grad(
    fx: &Fx,
    scf: &ScfResult,
    hc: &PeriodicHcore,
    hcfg: &PeriodicHcoreConfig,
    m: Option<GradMutation>,
) -> Result<GammaGradient, ferric_core::FerricError> {
    let src = RsGdfGradSource {
        gdf: &fx.gdf,
        aux: &fx.aux,
        aux_jac: None,
    };
    gamma_rhf_gradient_rsgdf(
        &fx.cell,
        &fx.prep,
        hcfg,
        hc,
        &src,
        scf,
        ExxDiv::None,
        &gcfg(m),
    )
}

/// The force of `fx` at density `scf` (its own by default).
pub fn grad_at(fx: &Fx, scf: &ScfResult, m: Option<GradMutation>) -> GammaGradient {
    try_grad(fx, scf, &fx.hc, &fx.hcfg, m).expect("gradient")
}

pub fn stress_at(fx: &Fx, scf: &ScfResult, m: Option<StressMutation>) -> GammaStress {
    let src = RsGdfGradSource {
        gdf: &fx.gdf,
        aux: &fx.aux,
        aux_jac: None,
    };
    gamma_rhf_stress_rsgdf(
        &fx.cell,
        &fx.prep,
        &fx.hcfg,
        &fx.hc,
        &src,
        scf,
        ExxDiv::None,
        &scfg(m),
    )
    .expect("stress")
}

pub fn mat3_max_diff(a: &Mat3, b: &Mat3) -> f64 {
    let mut m = 0.0_f64;
    for i in 0..3 {
        for j in 0..3 {
            m = m.max((a[i][j] - b[i][j]).abs());
        }
    }
    m
}

pub fn mat3_bits_equal(a: &Mat3, b: &Mat3) -> bool {
    (0..3).all(|i| (0..3).all(|j| a[i][j].to_bits() == b[i][j].to_bits()))
}

pub const GRAD_MUTANTS: [GradMutation; 4] = [
    GradMutation::RotDropT,
    GradMutation::RotUseT,
    GradMutation::RotOneSided,
    GradMutation::RotFlipSign,
];

pub const STRESS_MUTANTS: [StressMutation; 4] = [
    StressMutation::RotDropT,
    StressMutation::RotUseT,
    StressMutation::RotOneSided,
    StressMutation::RotFlipSign,
];

/// The rotated fixtures really rotate (so the comparisons are not vacuous).
pub fn assert_rotates(fx: &Fx, tag: &str) {
    assert!(fx.hc.sr_rotated_columns > 0, "{tag}: hcore did not rotate");
    assert_eq!(
        fx.gdf.timings().counter("rsgdf SR3 rotated columns"),
        Some(fx.hc.sr_rotated_columns as u64),
        "{tag}: the gradient build did not rotate"
    );
}

/// The force fixtures: (tag, cell, split).
pub fn force_fixtures() -> Vec<(&'static str, Cell, Option<RangeSplit>)> {
    vec![
        ("H2 unsplit", h2_off_axis(&H2_OFF_AXIS), None),
        (
            "H2 split",
            h2_off_axis(&H2_OFF_AXIS),
            Some(RangeSplit::default()),
        ),
        ("H2O split", water_tilt(), Some(RangeSplit::default())),
    ]
}

/// Largest element-wise `|m|` of a part.
pub fn amax_part(m: &Array2<f64>) -> f64 {
    max_abs(m)
}

/// The RHF energy of `cell` through the rotated (or not) pipeline at `rot`.
pub fn pipeline_energy(
    cell: &Cell,
    bs: &str,
    rot: SrColumnRotation,
    split: Option<RangeSplit>,
) -> f64 {
    let prep = bundled(cell, bs);
    let aux = bundled(cell, "cc-pvdz-ri");
    let hc = periodic_hcore(cell, &prep, &hcore_cfg(rot, 1e-14)).expect("hcore");
    let g = RsGdf::build(cell, &prep, &aux, &hc.s, &gdf_cfg(split, rot, 1e-13)).expect("rsgdf");
    energy(cell, &prep, &hc, &g)
}

// ------------------------------------- parameterised bodies (h)

/// EXACTNESS ANCHOR (forces): with the rotation requested, the analytic
/// force equals the unrotated force. Two measures: the WALK anchor (one
/// fixed density through both builds: only the derivative walks differ) and
/// the END-TO-END anchor (each build's own SCF). Mutants of the forward
/// transform must miss the walk anchor.
pub fn rotated_forces_case(fixtures: Vec<(&str, Cell, Option<RangeSplit>)>) {
    for (tag, cell, split) in fixtures {
        let on = cc_fixture(cell.clone(), production(), split);
        let off = cc_fixture(cell, OFF, split);
        assert_rotates(&on, tag);
        let g_off = grad_at(&off, &off.scf, None);
        // Walk anchor: the SAME density through the rotated and the plain walks.
        let g_walk = grad_at(&on, &off.scf, None);
        let walk = max_abs_diff(&g_walk.grad, &g_off.grad);
        // End-to-end: each build's own converged density.
        let g_own = grad_at(&on, &on.scf, None);
        let own = max_abs_diff(&g_own.grad, &g_off.grad);
        let scale = max_abs(&g_off.grad);
        eprintln!(
            "{tag}: max|F| {scale:.3e}; rot-vs-off walk {walk:.2e}, own-SCF {own:.2e}; \
             SR3 deriv calls {} vs {}",
            g_walk.fit.as_ref().map_or(0, |f| f.n_sr3_deriv),
            g_off.fit.as_ref().map_or(0, |f| f.n_sr3_deriv),
        );
        assert!(walk < D_BAR, "{tag}: walk anchor {walk:e}");
        assert!(own < D_BAR, "{tag}: own-SCF anchor {own:e}");
        // Each path is anchored on its own term, not only on the sum.
        let hcore_term = |g: &GammaGradient| &g.parts.vsr_basis + &g.parts.vsr_nuc;
        let hc_walk = max_abs_diff(&hcore_term(&g_walk), &hcore_term(&g_off));
        let fit_walk = max_abs_diff(&g_walk.parts.fit_orb_sr, &g_off.parts.fit_orb_sr);
        assert!(
            hc_walk < D_BAR && fit_walk < D_BAR,
            "{tag}: {hc_walk:e} {fit_walk:e}"
        );
        assert!(
            amax_part(&g_off.parts.fit_orb_sr) > 1e-3 && amax_part(&hcore_term(&g_off)) > 1e-3,
            "{tag}: vacuous SR terms"
        );
        // The rotated walks do fewer derivative calls (the payoff exists).
        let n3 = |g: &GammaGradient| g.fit.as_ref().expect("fit").n_sr3_deriv;
        assert!(n3(&g_walk) < n3(&g_off), "{tag}: SR3 calls");
        assert!(
            g_walk.n_sr_triplets < g_off.n_sr_triplets,
            "{tag}: hcore calls"
        );
        // Mutants: every one misses BOTH the hcore and the fit term.
        for m in GRAD_MUTANTS {
            let gm = grad_at(&on, &off.scf, Some(m));
            let tot = max_abs_diff(&gm.grad, &g_off.grad);
            let hc_miss = max_abs_diff(&hcore_term(&gm), &hcore_term(&g_off));
            let fit_miss = max_abs_diff(&gm.parts.fit_orb_sr, &g_off.parts.fit_orb_sr);
            assert!(
                tot > D_MUTANT_FLOOR && hc_miss > D_MUTANT_FLOOR && fit_miss > D_MUTANT_FLOOR,
                "{tag}: mutant {m:?} escaped: total {tot:e}, hcore {hc_miss:e}, fit {fit_miss:e}"
            );
        }
    }
}

/// EXACTNESS ANCHOR (stress), same structure as the forces'.
/// `splits`: the range splits to run; `strains`: the `(i, j)` strain
/// components whose central FD of the rotated energy is compared (all nine
/// in the weekly anchor).
pub fn rotated_stress_case(splits: &[Option<RangeSplit>], strains: &[(usize, usize)]) {
    let h = 1e-4;
    for &split in splits {
        let cell = h2_off_axis(&H2_OFF_AXIS);
        let on = cc_fixture(cell.clone(), production(), split);
        let off = cc_fixture(cell.clone(), OFF, split);
        assert_rotates(&on, "H2");
        let s_off = stress_at(&off, &off.scf, None);
        let s_walk = stress_at(&on, &off.scf, None);
        let s_own = stress_at(&on, &on.scf, None);
        let walk = mat3_max_diff(&s_walk.de_deps, &s_off.de_deps);
        let own = mat3_max_diff(&s_own.de_deps, &s_off.de_deps);
        // FD of the rotated energy under all nine strains.
        let mut fd_err = 0.0_f64;
        for &(i, j) in strains {
            let mut e = [[0.0; 3]; 3];
            e[i][j] = h;
            let ep = pipeline_energy(&cell.strained(&e).unwrap(), "cc-pvdz", production(), split);
            e[i][j] = -h;
            let em = pipeline_energy(&cell.strained(&e).unwrap(), "cc-pvdz", production(), split);
            let fd = (ep - em) / (2.0 * h);
            fd_err = fd_err.max((s_own.de_deps[i][j] - fd).abs());
        }
        assert!(walk < D_BAR, "split={split:?}: stress walk anchor {walk:e}");
        assert!(
            own < D_BAR,
            "split={split:?}: stress own-SCF anchor {own:e}"
        );
        assert!(
            fd_err < FD_S_BAR,
            "split={split:?}: stress analytic-vs-FD {fd_err:e}"
        );
        let vsr = |s: &GammaStress| s.parts.vsr;
        let zero = [[0.0; 3]; 3];
        assert!(mat3_max_diff(&vsr(&s_walk), &vsr(&s_off)) < D_BAR);
        assert!(mat3_max_diff(&s_walk.parts.fit_j3_sr, &s_off.parts.fit_j3_sr) < D_BAR);
        assert!(
            mat3_max_diff(&vsr(&s_off), &zero) > 1e-3
                && mat3_max_diff(&s_off.parts.fit_j3_sr, &zero) > 1e-3,
            "vacuous SR strain terms"
        );
        for m in STRESS_MUTANTS {
            let sm = stress_at(&on, &off.scf, Some(m));
            let tot = mat3_max_diff(&sm.de_deps, &s_off.de_deps);
            let hc_miss = mat3_max_diff(&vsr(&sm), &vsr(&s_off));
            let fit_miss = mat3_max_diff(&sm.parts.fit_j3_sr, &s_off.parts.fit_j3_sr);
            assert!(
                tot > D_MUTANT_FLOOR && hc_miss > D_MUTANT_FLOOR && fit_miss > D_MUTANT_FLOOR,
                "split={split:?}: mutant {m:?} escaped: total {tot:e}, hcore {hc_miss:e}, \
                 fit {fit_miss:e}"
            );
        }
    }
}

/// TRIVIAL LIMIT: a basis with nothing to rotate. Force and stress with the
/// rotation requested (and every forward-transform mutant, which cannot act)
/// are BITWISE the unrotated ones.
///
/// The unmutated check runs on the triclinic cell (non-orthogonal images);
/// the mutants cannot act on a basis with nothing to rotate, so they are
/// checked on the cheap cubic H2 cell (the full triclinic mutant sweep took
/// ~4.5 min and timed out the CI shard).
/// `cases`: `(cell, run the forward-transform mutants, splits)`.
pub fn identity_rotation_forces_and_stress_case(cases: Vec<(Cell, bool, Vec<Option<RangeSplit>>)>) {
    for (cell, mutants, splits) in cases {
        for split in splits {
            let on = fixture(
                cell.clone(),
                prep_for(&cell, &sp_basis_h()),
                production(),
                split,
            );
            let off = fixture(cell.clone(), prep_for(&cell, &sp_basis_h()), OFF, split);
            assert_eq!(on.hc.sr_rotated_columns, 0);
            assert_eq!(
                on.gdf.timings().counter("rsgdf SR3 rotated columns"),
                Some(0)
            );
            let g_off = grad_at(&off, &off.scf, None);
            let s_off = stress_at(&off, &off.scf, None);
            let gm: Vec<Option<GradMutation>> = if mutants {
                std::iter::once(None)
                    .chain(GRAD_MUTANTS.iter().map(|&m| Some(m)))
                    .collect()
            } else {
                vec![None]
            };
            for m in gm {
                let g = grad_at(&on, &on.scf, m);
                assert_bitwise(
                    &g.grad,
                    &g_off.grad,
                    &format!("force {m:?} split={split:?}"),
                );
            }
            let sm: Vec<Option<StressMutation>> = if mutants {
                std::iter::once(None)
                    .chain(STRESS_MUTANTS.iter().map(|&m| Some(m)))
                    .collect()
            } else {
                vec![None]
            };
            for m in sm {
                let s = stress_at(&on, &on.scf, m);
                assert!(
                    mat3_bits_equal(&s.de_deps, &s_off.de_deps),
                    "stress {m:?} split={split:?}"
                );
            }
        }
    }
}

/// The rotated force and stress are BITWISE identical across thread counts
/// (the ordered-parallel derivative walks and the per-element `Tᵀ Y T`
/// read are deterministic), and the frozen serial hcore oracle
/// (`SerialDerivWalks`) agrees bitwise with the ordered walk on the rotated
/// shells.
/// `threads`: the pools compared with 1 thread.
pub fn rotated_forces_and_stress_threads_case(splits: &[Option<RangeSplit>], threads: &[usize]) {
    for &split in splits {
        let cell = h2_off_axis(&H2_OFF_AXIS);
        let fx = cc_fixture(cell, production(), split);
        assert_rotates(&fx, "H2");
        let ref_g = in_pool(1, || grad_at(&fx, &fx.scf, None));
        let ref_s = in_pool(1, || stress_at(&fx, &fx.scf, None));
        for &n in threads {
            let g = in_pool(n, || grad_at(&fx, &fx.scf, None));
            let s = in_pool(n, || stress_at(&fx, &fx.scf, None));
            assert_bitwise(&g.grad, &ref_g.grad, &format!("force, {n} threads"));
            assert!(
                mat3_bits_equal(&s.de_deps, &ref_s.de_deps),
                "stress, {n} threads, split={split:?}"
            );
        }
        let serial = grad_at(&fx, &fx.scf, Some(GradMutation::SerialDerivWalks));
        assert_bitwise(&serial.grad, &ref_g.grad, "SerialDerivWalks vs ordered");
        let serial = stress_at(&fx, &fx.scf, Some(StressMutation::SerialDerivWalks));
        assert!(
            mat3_bits_equal(&serial.de_deps, &ref_s.de_deps),
            "stress SerialDerivWalks vs ordered, split={split:?}"
        );
    }
}
