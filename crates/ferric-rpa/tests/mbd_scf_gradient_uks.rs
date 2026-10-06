//! MBD@rsSCS-on-SCF nuclear gradient for unrestricted (UKS) references.
//!
//! The gradient is decomposed exactly as the closed-shell one
//! (`tests/mbd_scf_gradient.rs`):
//!   FD_full  — central FD of the pipeline with the UKS SCF re-solved at every
//!              displaced geometry: the true derivative.
//!   FD_orth  — the reference spin orbitals kept and only re-orthonormalized
//!              in the displaced basis: what `gradient_unrelaxed` models.
//!   FD_full − FD_orth = the orbital relaxation, against the unrestricted
//!              Z-vector term.
//!
//! Fast tests (run by default):
//!   * `uks_path_on_a_closed_shell_reproduces_the_closed_shell_gradient` — the
//!     exactness anchor: an RKS result re-labelled as UKS (α = β) must give
//!     the closed-shell gradient term by term.
//!   * `uks_mbd_unrelaxed_gradient_matches_fd_with_reorthonormalized_orbitals`
//!   * `uks_mbd_exact_gradient_matches_fd_of_full_scf_pipeline` — OH/STO-3G PBE.
//!
//! Measurement (`#[ignore]`, prints): `measure_uks_full_pipeline_fd` — NH2, OH
//! and O2 at 6-31G with PBE, PBE0 and HSE06.
//!
//!   OPENBLAS_NUM_THREADS=1 cargo test -p ferric-rpa --test mbd_scf_gradient_uks --release
//!   OPENBLAS_NUM_THREADS=1 cargo test -p ferric-rpa --test mbd_scf_gradient_uks --release \
//!     -- --ignored --nocapture

use ferric_core::basis::{self, BasisSet};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_rpa::dispersion::{
    mbd_rsscs_for_scf, mbd_rsscs_for_spin_densities, MbdFreeAtomCache, MbdRsscsConfig,
};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::{solve_uhf, solve_uhf_with_guess};
use ferric_scf::{ScfResult, Spin};
use ndarray::{s, Array2};

const H2O_XYZ: &str = "3\nwater\nO 0.000000 0.000000 0.117300\nH 0.000000 0.757200 -0.469200\nH 0.000000 -0.757200 -0.469200\n";
const OH_XYZ: &str = "2\nOH\nO 0.000000 0.000000 0.000000\nH 0.100000 0.000000 0.970000\n";
const NH2_XYZ: &str =
    "3\nNH2\nN 0.000000 0.000000 0.142000\nH 0.000000 0.802000 -0.497000\nH 0.000000 -0.802000 -0.497000\n";
const O2_XYZ: &str = "2\nO2\nO 0.000000 0.000000 0.604000\nO 0.000000 0.000000 -0.604000\n";

fn ks(xc: &str) -> RhfConfig {
    RhfConfig {
        xc: Some(xc.to_string()),
        max_iter: 300,
        energy_conv: 1e-11,
        density_conv: 1e-9,
        ..Default::default()
    }
}

/// FD step (Bohr) of the fast tests; `MBD_FD_H` overrides it. With the C2
/// proatom interpolant the volumes are smooth in the nuclear coordinates, so
/// a standard step works: OH/STO-3G PBE at h = 1e-3 measured unrelaxed vs
/// FD_orth 3.9e-11, exact vs FD_full 2.9e-11.
fn fd_step() -> f64 {
    std::env::var("MBD_FD_H")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1e-3)
}

fn max_abs(a: &Array2<f64>) -> f64 {
    a.iter().fold(0.0_f64, |m, v| m.max(v.abs()))
}

fn displaced(mol: &Molecule, atom: usize, k: usize, h: f64) -> Molecule {
    let mut m = mol.clone();
    match k {
        0 => m.atoms[atom].x += h,
        1 => m.atoms[atom].y += h,
        _ => m.atoms[atom].zpos += h,
    }
    m
}

fn bounds_for(mol: &Molecule, bs: &BasisSet) -> (PreparedBasis, SchwarzBounds) {
    let prep = PreparedBasis::new(mol, bs).expect("prepared basis");
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).expect("schwarz");
    (prep, bounds)
}

fn uks(mol: &Molecule, bs: &BasisSet, cfg: &RhfConfig, guess: Option<&ScfResult>) -> ScfResult {
    let (prep, bounds) = bounds_for(mol, bs);
    let ctx = ParallelContext::default();
    let r = match guess {
        Some(g) => solve_uhf_with_guess(
            &ctx,
            mol,
            &prep,
            &bounds,
            cfg,
            Some((&g.mos_alpha, g.mos_beta.as_ref().expect("beta MOs"))),
        ),
        None => solve_uhf(&ctx, mol, &prep, &bounds, cfg),
    }
    .expect("uks");
    assert!(r.converged, "UKS did not converge");
    assert!(matches!(r.spin, Spin::Unrestricted));
    r
}

fn n_occ(mol: &Molecule) -> (usize, usize) {
    let nelec = mol.nelec() as usize;
    let two_s = mol.multiplicity - 1;
    ((nelec + two_s) / 2, (nelec - two_s) / 2)
}

/// The reference occupied orbitals of one spin, Löwdin-orthonormalized in
/// `mol`'s (displaced) basis: D_σ = C' C'ᵀ, C' = C (Cᵀ S C)^{-1/2}.
fn reorthonormalized(s: &Array2<f64>, c: &Array2<f64>) -> Array2<f64> {
    use ndarray_linalg::{Eigh, UPLO};
    if c.ncols() == 0 {
        return Array2::zeros((c.nrows(), c.nrows()));
    }
    let m = c.t().dot(s).dot(c);
    let (w, v) = m.eigh(UPLO::Lower).expect("eigh");
    let mut vs = v.clone();
    for (j, wj) in w.iter().enumerate() {
        let f = 1.0 / wj.sqrt();
        vs.column_mut(j).mapv_inplace(|x| x * f);
    }
    let cp = c.dot(&vs.dot(&v.t()));
    cp.dot(&cp.t())
}

/// Central FD (step `h`) of E_MBD with the spin densities at each displaced
/// geometry from `densities(m)`.
fn fd_mbd(
    mol: &Molecule,
    bs: &BasisSet,
    cache: &MbdFreeAtomCache,
    mbd_cfg: &MbdRsscsConfig,
    h: f64,
    densities: impl Fn(&Molecule) -> (Array2<f64>, Array2<f64>),
) -> Array2<f64> {
    let n = mol.atoms.len();
    let mut fd = Array2::<f64>::zeros((n, 3));
    for a in 0..n {
        for k in 0..3 {
            let e = |sg: f64| {
                let m = displaced(mol, a, k, sg * h);
                let (da, db) = densities(&m);
                mbd_rsscs_for_spin_densities(cache, &m, bs, &da, &db, mbd_cfg, false)
                    .expect("mbd")
                    .energy
            };
            fd[(a, k)] = (e(1.0) - e(-1.0)) / (2.0 * h);
        }
    }
    fd
}

fn fd_orth(
    mol: &Molecule,
    bs: &BasisSet,
    cache: &MbdFreeAtomCache,
    mbd_cfg: &MbdRsscsConfig,
    r: &ScfResult,
    h: f64,
) -> Array2<f64> {
    let (na, nb) = n_occ(mol);
    let ca = r.mos_alpha.slice(s![.., ..na]).to_owned();
    let cb = r
        .mos_beta
        .as_ref()
        .expect("beta")
        .slice(s![.., ..nb])
        .to_owned();
    fd_mbd(mol, bs, cache, mbd_cfg, h, |m| {
        let prep = PreparedBasis::new(m, bs).expect("prep");
        let sm = ferric_integrals::oneelectron::overlap(&prep);
        (reorthonormalized(&sm, &ca), reorthonormalized(&sm, &cb))
    })
}

fn fd_full(
    mol: &Molecule,
    bs: &BasisSet,
    cache: &MbdFreeAtomCache,
    mbd_cfg: &MbdRsscsConfig,
    cfg: &RhfConfig,
    r: &ScfResult,
    h: f64,
) -> Array2<f64> {
    fd_mbd(mol, bs, cache, mbd_cfg, h, |m| {
        let rm = uks(m, bs, cfg, Some(r));
        (
            rm.density_alpha.clone(),
            rm.density_beta.clone().expect("beta"),
        )
    })
}

/// The converged RKS result re-expressed as an unrestricted one with α = β:
/// same MOs, orbital energies and Fock for both spins, D_σ = ½ D.
fn as_unrestricted(r: &ScfResult) -> ScfResult {
    let mut u = r.clone();
    u.spin = Spin::Unrestricted;
    u.density_alpha = 0.5 * r.density_r();
    u.density_beta = Some(0.5 * r.density_r());
    u.mos_beta = Some(r.mos_alpha.clone());
    u.eps_beta = Some(r.eps_alpha.clone());
    u.fock_beta = Some(r.fock_alpha.clone());
    u
}

/// Bar of the exactness anchor (absolute, Hartree/Bohr). Measured 3.0e-14
/// (PBE), 2.8e-14 (PBE0, HSE06) on the relaxation term; the energy and the
/// unrelaxed and orthonormality terms are bit-identical. The two paths differ
/// only by the t of their central differences (the UKS spin perturbation is ¼
/// of the closed-shell total one, so its t is 4×) and rounding.
const ANCHOR_TOL: f64 = 1e-12;

/// EXACTNESS ANCHOR: an RKS result run through the UKS path (α = β) must give
/// the closed-shell MBD gradient — unrelaxed, orthonormality and relaxation
/// terms separately. Catches: a wrong factor in the unrestricted right-hand
/// side (−2), the contraction (½), the spin-resolved exchange (c vs ½c), the
/// orthonormality term (−1 vs −½), W_σ = D_σ F_σ D_σ, or a dropped spin.
#[test]
fn uks_path_on_a_closed_shell_reproduces_the_closed_shell_gradient() {
    let mol = Molecule::parse_xyz(H2O_XYZ, 0, 1).expect("h2o");
    let bs = basis::bundled("sto-3g").expect("sto-3g");
    let op = Operator::coulomb();
    let ctx = ParallelContext::default();
    for xc in ["PBE", "PBE0", "HSE06"] {
        let cfg = ks(xc);
        let mbd_cfg = MbdRsscsConfig::for_functional(xc).expect("beta");
        let cache = MbdFreeAtomCache::build(&ctx, &mol, &bs, op, &cfg).expect("cache");
        let (prep, bounds) = bounds_for(&mol, &bs);
        let r = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg).expect("rks");
        assert!(r.converged);
        let c = mbd_rsscs_for_scf(&ctx, &cache, &mol, &bs, op, &cfg, &r, &mbd_cfg).expect("rks");
        let u = mbd_rsscs_for_scf(
            &ctx,
            &cache,
            &mol,
            &bs,
            op,
            &cfg,
            &as_unrestricted(&r),
            &mbd_cfg,
        )
        .expect("uks");
        let d_orth = max_abs(
            &(c.gradient_orthonormality.as_ref().unwrap()
                - u.gradient_orthonormality.as_ref().unwrap()),
        );
        let d_unrel = max_abs(
            &(c.gradient_unrelaxed.as_ref().unwrap() - u.gradient_unrelaxed.as_ref().unwrap()),
        );
        let gr = c.gradient_relaxation.as_ref().unwrap();
        let d_relax = max_abs(&(gr - u.gradient_relaxation.as_ref().unwrap()));
        let d_full = max_abs(&(c.gradient.as_ref().unwrap() - u.gradient.as_ref().unwrap()));
        println!(
            "{xc}: |dE| = {:.1e}; max|d orth| = {d_orth:.3e}, max|d unrelaxed| = {d_unrel:.3e}, \
             max|d relax| = {d_relax:.3e} (max|relax| = {:.3e}), max|d total| = {d_full:.3e}",
            (c.energy - u.energy).abs(),
            max_abs(gr)
        );
        assert_eq!(c.energy, u.energy, "{xc}: energy");
        for (what, d) in [
            ("orthonormality", d_orth),
            ("unrelaxed", d_unrel),
            ("relaxation", d_relax),
            ("total", d_full),
        ] {
            assert!(d < ANCHOR_TOL, "{xc}: {what} differs by {d:.3e}");
        }
        // Reachability: the relaxation term is far above the anchor bar.
        assert!(max_abs(gr) > 1e6 * ANCHOR_TOL, "{xc}: relaxation too small");
    }
}

/// Bar for the unrelaxed analytic gradient against FD_orth at `fd_step()`.
/// Measured 3.9e-11 at h = 1e-3. Dropping the orthonormality term misses by
/// its size (3.2e-6), asserted resolvable.
const FD_ORTH_TOL: f64 = 5e-10;

/// THE model test of the fixed-orbital part, OH/STO-3G PBE. Catches: the
/// open-shell orthonormality term dropped, mis-scaled (−½ instead of −1), or
/// applied to one spin only.
#[test]
fn uks_mbd_unrelaxed_gradient_matches_fd_with_reorthonormalized_orbitals() {
    let mol = Molecule::parse_xyz(OH_XYZ, 0, 2).expect("OH");
    let bs = basis::bundled("sto-3g").expect("sto-3g");
    let op = Operator::coulomb();
    let ctx = ParallelContext::default();
    let cfg = ks("PBE");
    let mbd_cfg = MbdRsscsConfig::for_functional("PBE").expect("beta");
    let cache = MbdFreeAtomCache::build(&ctx, &mol, &bs, op, &cfg).expect("cache");
    let r = uks(&mol, &bs, &cfg, None);
    let out = mbd_rsscs_for_spin_densities(
        &cache,
        &mol,
        &bs,
        &r.density_alpha,
        r.density_beta.as_ref().unwrap(),
        &mbd_cfg,
        true,
    )
    .expect("mbd");
    let g = out.gradient_unrelaxed.expect("unrelaxed");
    let orth = out.gradient_orthonormality.expect("orth");
    let fd = fd_orth(&mol, &bs, &cache, &mbd_cfg, &r, fd_step());
    let diff = max_abs(&(&g - &fd));
    println!(
        "OH/STO-3G PBE: max|g| = {:.3e}, max|orth| = {:.3e}, max|g - FD_orth| = {diff:.3e}",
        max_abs(&g),
        max_abs(&orth)
    );
    assert!(
        diff < FD_ORTH_TOL,
        "unrelaxed vs FD_orth: {diff:.3e} >= {FD_ORTH_TOL:.1e}"
    );
    assert!(
        max_abs(&orth) > 10.0 * FD_ORTH_TOL,
        "orthonormality term {:.3e} not resolvable",
        max_abs(&orth)
    );
}

/// Bar on the EXACT UKS gradient against FD_full at `fd_step()`, OH/STO-3G
/// PBE. Measured 2.9e-11 at h = 1e-3; the relaxation term against
/// FD_full − FD_orth 1.0e-11. Without the relaxation term the gradient misses
/// by 4.3e-6.
const FD_FULL_TOL: f64 = 5e-10;

/// THE exactness test, fast: OH/STO-3G PBE, exact gradient vs FD of the full
/// UKS + MBD pipeline. Catches: the relaxation term dropped or mis-scaled
/// (asserted resolvable at this bar), one spin's Z dropped, the RHS factor.
/// Also checks translation invariance and the relaxation term against
/// FD_full − FD_orth.
#[test]
fn uks_mbd_exact_gradient_matches_fd_of_full_scf_pipeline() {
    let mol = Molecule::parse_xyz(OH_XYZ, 0, 2).expect("OH");
    let bs = basis::bundled("sto-3g").expect("sto-3g");
    let op = Operator::coulomb();
    let ctx = ParallelContext::default();
    let cfg = ks("PBE");
    let mbd_cfg = MbdRsscsConfig::for_functional("PBE").expect("beta");
    let cache = MbdFreeAtomCache::build(&ctx, &mol, &bs, op, &cfg).expect("cache");
    let r = uks(&mol, &bs, &cfg, None);
    let out = mbd_rsscs_for_scf(&ctx, &cache, &mol, &bs, op, &cfg, &r, &mbd_cfg).expect("mbd");
    let g = out.gradient.expect("exact");
    let relax = out.gradient_relaxation.expect("relaxation");
    let h = fd_step();
    let fd = fd_full(&mol, &bs, &cache, &mbd_cfg, &cfg, &r, h);
    let fdo = fd_orth(&mol, &bs, &cache, &mbd_cfg, &r, h);
    let relax_fd = &fd - &fdo;
    let diff = max_abs(&(&g - &fd));
    let diff_relax = max_abs(&(&relax - &relax_fd));
    println!(
        "OH/STO-3G PBE: max|FD| = {:.3e}, max|relax| = {:.3e}, max|exact - FD| = {diff:.3e}, \
         max|relax(Z) - relax(FD)| = {diff_relax:.3e}",
        max_abs(&fd),
        max_abs(&relax)
    );
    assert!(
        diff < FD_FULL_TOL,
        "exact vs FD_full: {diff:.3e} >= {FD_FULL_TOL:.1e}"
    );
    assert!(
        diff_relax < FD_FULL_TOL,
        "relaxation vs FD: {diff_relax:.3e} >= {FD_FULL_TOL:.1e}"
    );
    for k in 0..3 {
        let sum: f64 = g.column(k).sum();
        assert!(
            sum.abs() < 1e-10 * max_abs(&g),
            "sum over atoms of the exact gradient, axis {k}: {sum:.3e}"
        );
    }
    assert!(
        max_abs(&relax) > 10.0 * FD_FULL_TOL,
        "relaxation term {:.3e} is not resolvable at the bar",
        max_abs(&relax)
    );
}

/// Measurement, not a validation: NH2, OH and O2 at 6-31G with PBE, PBE0 and
/// HSE06, and OH with PBE + RI-J and PBE0 + RI-JK; prints exact vs FD_full, relaxation vs FD_full − FD_orth, and the
/// unrelaxed gradient vs FD_orth. `MBD_FD_ONLY=<name>` and `MBD_FD_H=<h>`
/// select a molecule and the step.
#[test]
#[ignore = "measurement: full-pipeline UKS FD at 6-31G, minutes"]
fn measure_uks_full_pipeline_fd() {
    let bs = basis::bundled("6-31g").expect("6-31g");
    let op = Operator::coulomb();
    let ctx = ParallelContext::default();
    let h: f64 = std::env::var("MBD_FD_H")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1e-3);
    let only: Option<String> = std::env::var("MBD_FD_ONLY").ok();
    let only_xc: Option<String> = std::env::var("MBD_FD_XC").ok();
    let cases = [("NH2", NH2_XYZ, 2), ("OH", OH_XYZ, 2), ("O2", O2_XYZ, 3)];
    for (name, xyz, mult) in cases {
        if only.as_deref().is_some_and(|o| o != name) {
            continue;
        }
        // (label, functional, RI-J aux, RI-K aux); the fitted variants on OH.
        let jk = "def2-universal-jkfit";
        let mut variants = vec![
            ("PBE", "PBE", None, None),
            ("PBE0", "PBE0", None, None),
            ("HSE06", "HSE06", None, None),
        ];
        if name == "OH" {
            variants.push(("PBE RI-J", "PBE", Some(jk), None));
            variants.push(("PBE0 RI-JK", "PBE0", Some(jk), Some(jk)));
        }
        for (tag, xc, j_aux, k_aux) in variants {
            if only_xc.as_deref().is_some_and(|o| o != tag) {
                continue;
            }
            let label = format!("{name}/6-31G {tag}");
            let cfg = RhfConfig {
                df_j_aux: j_aux.map(str::to_string),
                df_k_aux: k_aux.map(str::to_string),
                ..ks(xc)
            };
            let mbd_cfg = MbdRsscsConfig::for_functional(xc).expect("beta");
            let mol = Molecule::parse_xyz(xyz, 0, mult).expect("mol");
            let cache = MbdFreeAtomCache::build(&ctx, &mol, &bs, op, &cfg).expect("cache");
            let r = uks(&mol, &bs, &cfg, None);
            let out =
                mbd_rsscs_for_scf(&ctx, &cache, &mol, &bs, op, &cfg, &r, &mbd_cfg).expect("mbd");
            let g = out.gradient.clone().expect("exact");
            let gu = out.gradient_unrelaxed.clone().expect("unrelaxed");
            let gr = out.gradient_relaxation.clone().expect("relaxation");
            let fd = fd_full(&mol, &bs, &cache, &mbd_cfg, &cfg, &r, h);
            let fdo = fd_orth(&mol, &bs, &cache, &mbd_cfg, &r, h);
            let relax_fd = &fd - &fdo;
            let tsum = (0..3)
                .map(|k| g.column(k).sum().abs())
                .fold(0.0_f64, f64::max);
            println!(
                "{label}: E_MBD = {:.10e}; max|FD_full| = {:.4e}; max|exact - FD_full| = {:.3e}; \
                 max|relax(Z) - relax(FD)| = {:.3e}; max|unrelaxed - FD_orth| = {:.3e}; \
                 max|relax| = {:.3e}; max|sum_B g| = {tsum:.1e}",
                out.energy,
                max_abs(&fd),
                max_abs(&(&g - &fd)),
                max_abs(&(&gr - &relax_fd)),
                max_abs(&(&gu - &fdo)),
                max_abs(&relax_fd)
            );
        }
    }
}

/// Fit-off, unscreened COSX on a flat (50,110) grid: the exchange energy the
/// open-shell COSX gradient differentiates exactly.
fn cosx_flat() -> ferric_scf::cosx_k::CosxConfig {
    ferric_scf::cosx_k::CosxConfig {
        overlap_fit: false,
        screen_thresh: None,
        half_transform: ferric_scf::cosx_k::CosxHalfTransform::Dense,
        grid: ferric_dft::grid::AtomicGridConfig {
            n_radial: 50,
            n_angular: 110,
            prune: None,
        },
        final_grid: None,
        ..ferric_scf::cosx_k::CosxConfig::flat_reference()
    }
}

/// Bar of the MBD + COSX UKS test (absolute, Ha/Bohr), derived from both
/// sides at h = 1e-3: the exact gradient measured 2.3e-11 against FD_full,
/// the relaxation term 3.2e-11 against FD_full − FD_orth; the same driver
/// with the COSX switch removed (the Z-vector contraction then differentiates
/// EXACT exchange against the COSX SCF) misses FD_full by 4.6e-10. 1e-10 sits
/// 4.3x above the first and 4.6x below the second.
const COSX_MBD_TOL: f64 = 1e-10;

/// MBD@rsSCS on a UKS-PBE0 SCF whose exchange is COSX (fit off): the exact
/// gradient (unrelaxed + orthonormality + Z-vector relaxation, the
/// contraction differentiating the COSX energy) vs central FD of the full
/// UKS(COSX) + MBD pipeline, NH2/STO-3G. NEGATIVE CONTROL: the same driver
/// with the COSX switch removed from the config (exact-K contraction against
/// the COSX SCF — the pairing the open-shell COSX refusal used to prevent)
/// must miss FD by more than the bar.
#[test]
fn uks_mbd_cosx_exact_gradient_matches_fd_of_full_scf_pipeline() {
    let mol = Molecule::parse_xyz(NH2_XYZ, 0, 2).expect("NH2");
    let bs = basis::bundled("sto-3g").expect("sto-3g");
    let op = Operator::coulomb();
    let ctx = ParallelContext::default();
    let plain = ks("PBE0");
    let cfg = RhfConfig {
        k_builder: Some("cosx".into()),
        cosx: cosx_flat(),
        ..plain.clone()
    };
    assert!(ferric_scf::cosx_gradient::scf_exchange_is_cosx(&cfg, true).unwrap());
    let mbd_cfg = MbdRsscsConfig::for_functional("PBE0").expect("beta");
    let cache = MbdFreeAtomCache::build(&ctx, &mol, &bs, op, &plain).expect("cache");
    let r = uks(&mol, &bs, &cfg, None);
    let out = mbd_rsscs_for_scf(&ctx, &cache, &mol, &bs, op, &cfg, &r, &mbd_cfg).expect("mbd");
    let g = out.gradient.expect("exact");
    let relax = out.gradient_relaxation.expect("relaxation");
    let paired = mbd_rsscs_for_scf(&ctx, &cache, &mol, &bs, op, &plain, &r, &mbd_cfg)
        .expect("exact-K contraction")
        .gradient
        .expect("exact");
    let h = fd_step();
    let fd = fd_full(&mol, &bs, &cache, &mbd_cfg, &cfg, &r, h);
    let fdo = fd_orth(&mol, &bs, &cache, &mbd_cfg, &r, h);
    let relax_fd = &fd - &fdo;
    let diff = max_abs(&(&g - &fd));
    let diff_relax = max_abs(&(&relax - &relax_fd));
    let diff_paired = max_abs(&(&paired - &fd));
    println!(
        "NH2/STO-3G UKS-PBE0 COSX + MBD: max|FD| = {:.3e}, max|relax| = {:.3e}, \
         max|exact - FD| = {diff:.3e}, max|relax(Z) - relax(FD)| = {diff_relax:.3e}; \
         exact-K contraction: max|g - FD| = {diff_paired:.3e}",
        max_abs(&fd),
        max_abs(&relax)
    );
    assert!(
        diff < COSX_MBD_TOL,
        "exact vs FD_full: {diff:.3e} >= {COSX_MBD_TOL:.1e}"
    );
    assert!(
        diff_relax < COSX_MBD_TOL,
        "relaxation vs FD: {diff_relax:.3e} >= {COSX_MBD_TOL:.1e}"
    );
    assert!(
        diff_paired > COSX_MBD_TOL,
        "NEGATIVE CONTROL blind: the exact-K contraction is only {diff_paired:.3e} off FD"
    );
}
