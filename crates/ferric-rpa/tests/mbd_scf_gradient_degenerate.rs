//! MBD@rsSCS-on-SCF exact gradient for ²Π radicals (issue #265): the
//! open-shell Z-vector on a linear molecule whose state breaks the
//! cylindrical symmetry.
//!
//! Investigation (tests/HYPOTHESES-degenerate-somo-zvector.md in ferric-scf,
//! measured by `zvector_ks::degenerate_somo_investigation`): the softest
//! eigenvector of the dense orbital Hessian is the axis-rotation mode κ_L to
//! |cos| ≥ 0.99999 (OH, CH, NO; STO-3G and 6-31G), with a gap-metric
//! eigenvalue of grid-anisotropy size and random sign (OH UKS-PBE/STO-3G:
//! +9.9e-3 at (75,110), −7.7e-5 at (75,302); PBE0 −9.4e-4; NO −2.3e-3). The
//! Z-vector now solves on the complement of κ_L.
//!
//! The non-linear anchor: NH2 UKS-PBE0 and ROKS-PBE0 / STO-3G full MBD
//! gradients are bit-identical to origin/main (compared bit for bit in the
//! PR), since no mode is detected and every projection is a no-op.
//!
//! Each case: exact gradient vs central FD of the full SCF + MBD pipeline
//! (SCF re-solved at every displaced geometry, seeded from the reference),
//! the relaxation term vs FD_full − FD_orth, and `null_mode` reported.
//!
//!   OPENBLAS_NUM_THREADS=1 cargo test -p ferric-rpa --release --test mbd_scf_gradient_degenerate \
//!     -- --include-ignored --nocapture

use ferric_core::basis::{self, BasisSet};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_rpa::dispersion::{
    mbd_rsscs_for_scf, mbd_rsscs_for_spin_densities, MbdFreeAtomCache, MbdRsscsConfig,
};
use ferric_scf::rhf::RhfConfig;
use ferric_scf::rohf::solve_rohf;
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::{solve_uhf, solve_uhf_with_guess};
use ferric_scf::{ScfResult, Spin};
use ndarray::{s, Array2};

/// Diatomics on a tilted axis (no Cartesian axis is special), Å.
const OH_XYZ: &str = "2\nOH\nO 0.000000 0.000000 0.000000\nH 0.100000 0.000000 0.970000\n";
const NO_XYZ: &str = "2\nNO\nN 0.000000 0.000000 0.000000\nO 0.200000 0.000000 1.140000\n";
const CH_XYZ: &str = "2\nCH\nC 0.000000 0.000000 0.000000\nH 0.000000 0.300000 1.080000\n";
const NH2_XYZ: &str =
    "3\nNH2\nN 0.000000 0.000000 0.142000\nH 0.000000 0.802000 -0.497000\nH 0.000000 -0.802000 -0.497000\n";

#[derive(Clone, Copy, Debug)]
enum Reference {
    Uks,
    Roks,
}

fn ks(xc: &str) -> RhfConfig {
    RhfConfig {
        xc: Some(xc.to_string()),
        max_iter: 300,
        energy_conv: 1e-11,
        density_conv: 1e-9,
        ..Default::default()
    }
}

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

fn scf(
    reference: Reference,
    mol: &Molecule,
    bs: &BasisSet,
    cfg: &RhfConfig,
    guess: Option<&ScfResult>,
) -> ScfResult {
    let (prep, bounds) = bounds_for(mol, bs);
    let ctx = ParallelContext::default();
    let r = match (reference, guess) {
        (Reference::Uks, Some(g)) => solve_uhf_with_guess(
            &ctx,
            mol,
            &prep,
            &bounds,
            cfg,
            Some((&g.mos_alpha, g.mos_beta.as_ref().expect("beta MOs"))),
        ),
        (Reference::Uks, None) => solve_uhf(&ctx, mol, &prep, &bounds, cfg),
        (Reference::Roks, g) => {
            let mut c = cfg.clone();
            if let Some(g) = g {
                c.init_guess_density = Some(g.density_total.clone());
            }
            solve_rohf(&ctx, mol, &prep, Operator::coulomb(), &bounds, &c)
        }
    }
    .expect("scf");
    assert!(r.converged, "{reference:?} did not converge");
    r
}

/// D_σ = C'C'ᵀ with C' = C (Cᵀ S C)^{-1/2}.
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

/// FD with the reference occupied orbitals kept (Löwdin re-orthonormalized
/// in the displaced basis): the model of the unrelaxed + orthonormality terms.
/// ROKS keeps closed and open orbitals as ONE set (the ROKS Z-vector's path).
fn fd_orth(
    reference: Reference,
    mol: &Molecule,
    bs: &BasisSet,
    cache: &MbdFreeAtomCache,
    mbd_cfg: &MbdRsscsConfig,
    r: &ScfResult,
    h: f64,
) -> Array2<f64> {
    let nelec = mol.nelec() as usize;
    let two_s = mol.multiplicity - 1;
    let (na, nb) = ((nelec + two_s) / 2, (nelec - two_s) / 2);
    match reference {
        Reference::Uks => {
            let ca = r.mos_alpha.slice(s![.., ..na]).to_owned();
            let cb = r
                .mos_beta
                .as_ref()
                .expect("beta")
                .slice(s![.., ..nb])
                .to_owned();
            fd_mbd(mol, bs, cache, mbd_cfg, h, |m| {
                let sm =
                    ferric_integrals::oneelectron::overlap(&PreparedBasis::new(m, bs).unwrap());
                (reorthonormalized(&sm, &ca), reorthonormalized(&sm, &cb))
            })
        }
        Reference::Roks => {
            let c = r.mos_alpha.slice(s![.., ..na]).to_owned();
            fd_mbd(mol, bs, cache, mbd_cfg, h, |m| {
                use ndarray_linalg::{Eigh, UPLO};
                let sm =
                    ferric_integrals::oneelectron::overlap(&PreparedBasis::new(m, bs).unwrap());
                let mm = c.t().dot(&sm).dot(&c);
                let (w, v) = mm.eigh(UPLO::Lower).expect("eigh");
                let mut vs = v.clone();
                for (j, wj) in w.iter().enumerate() {
                    let f = 1.0 / wj.sqrt();
                    vs.column_mut(j).mapv_inplace(|x| x * f);
                }
                let cp = c.dot(&vs.dot(&v.t()));
                let cc = cp.slice(s![.., ..nb]);
                (cp.dot(&cp.t()), cc.dot(&cc.t()))
            })
        }
    }
}

fn fd_full(
    reference: Reference,
    mol: &Molecule,
    bs: &BasisSet,
    cache: &MbdFreeAtomCache,
    mbd_cfg: &MbdRsscsConfig,
    cfg: &RhfConfig,
    r: &ScfResult,
    h: f64,
) -> Array2<f64> {
    fd_mbd(mol, bs, cache, mbd_cfg, h, |m| {
        let rm = scf(reference, m, bs, cfg, Some(r));
        (
            rm.density_alpha.clone(),
            rm.density_beta.clone().expect("beta"),
        )
    })
}

/// Bar on the exact gradient against FD_full (absolute, Ha/Bohr), h = 1e-3,
/// the bar of the non-degenerate UKS/ROKS tests. Measured on the asserted
/// cases: 7.1e-12 (CH/6-31G ROKS) to 3.0e-11 (OH/STO-3G UKS-PBE0), against
/// relaxation terms of 2.2e-7..8.7e-6. Without the projection the OH UKS-PBE0
/// solve is REFUSED (PCG meets the mode's negative curvature, Rayleigh
/// −9.4e-4), so it cannot pass silently.
const FD_FULL_TOL: f64 = 5e-10;

/// One ²Π case. Returns (max|exact − FD|, max|relax(Z) − relax(FD)|,
/// max|relax|, null-mode diagnostics).
fn case(
    label: &str,
    reference: Reference,
    xyz: &str,
    basis: &str,
    xc: &str,
) -> (
    f64,
    f64,
    f64,
    Option<ferric_scf::zvector_ks::NullModeDiagnostics>,
) {
    let mol = Molecule::parse_xyz(xyz, 0, 2).expect("mol");
    let bs = basis::bundled(basis).expect("basis");
    let op = Operator::coulomb();
    let ctx = ParallelContext::default();
    let cfg = ks(xc);
    let mbd_cfg = MbdRsscsConfig::for_functional(xc).expect("beta");
    let cache = MbdFreeAtomCache::build(&ctx, &mol, &bs, op, &cfg).expect("cache");
    let r = scf(reference, &mol, &bs, &cfg, None);
    assert!(matches!(
        (reference, r.spin),
        (Reference::Uks, Spin::Unrestricted) | (Reference::Roks, Spin::RestrictedOpen)
    ));
    let out = mbd_rsscs_for_scf(&ctx, &cache, &mol, &bs, op, &cfg, &r, &mbd_cfg)
        .unwrap_or_else(|e| panic!("{label}: {e}"));
    let g = out.gradient.clone().expect("exact");
    let relax = out.gradient_relaxation.clone().expect("relaxation");
    let null = out.null_mode;
    let h = fd_step();
    let fd = fd_full(reference, &mol, &bs, &cache, &mbd_cfg, &cfg, &r, h);
    let fdo = fd_orth(reference, &mol, &bs, &cache, &mbd_cfg, &r, h);
    let d = max_abs(&(&g - &fd));
    let dr = max_abs(&(&relax - &(&fd - &fdo)));
    println!(
        "{label}: max|FD| = {:.3e}; max|relax| = {:.3e}; max|exact - FD| = {d:.3e}; \
         max|relax(Z) - relax(FD)| = {dr:.3e}; null mode {null:?}",
        max_abs(&fd),
        max_abs(&relax)
    );
    (d, dr, max_abs(&relax), null)
}

fn assert_case(label: &str, reference: Reference, xyz: &str, basis: &str, xc: &str) {
    let (d, dr, relax, null) = case(label, reference, xyz, basis, xc);
    assert!(
        null.is_some(),
        "{label}: the axis-rotation mode was not detected"
    );
    assert!(d < FD_FULL_TOL, "{label}: exact vs FD_full {d:.3e}");
    assert!(dr < FD_FULL_TOL, "{label}: relaxation vs FD {dr:.3e}");
    assert!(
        relax > 10.0 * FD_FULL_TOL,
        "{label}: relaxation {relax:.3e} not resolvable"
    );
}

/// OH UKS-PBE0/STO-3G: the null mode's eigenvalue is NEGATIVE here (−9.4e-4),
/// the configuration whose PCG met negative curvature before the projection.
#[test]
fn oh_uks_pbe0_mbd_exact_gradient_matches_fd() {
    assert_case(
        "OH/STO-3G UKS-PBE0",
        Reference::Uks,
        OH_XYZ,
        "sto-3g",
        "PBE0",
    );
}

#[test]
#[ignore = "slow: full-pipeline FD"]
fn oh_uks_pbe_mbd_exact_gradient_matches_fd() {
    assert_case("OH/STO-3G UKS-PBE", Reference::Uks, OH_XYZ, "sto-3g", "PBE");
}

#[test]
#[ignore = "slow: full-pipeline FD"]
fn no_uks_pbe_mbd_exact_gradient_matches_fd() {
    assert_case("NO/STO-3G UKS-PBE", Reference::Uks, NO_XYZ, "sto-3g", "PBE");
}

#[test]
#[ignore = "slow: full-pipeline FD"]
fn oh_roks_pbe0_mbd_exact_gradient_matches_fd() {
    assert_case(
        "OH/STO-3G ROKS-PBE0",
        Reference::Roks,
        OH_XYZ,
        "sto-3g",
        "PBE0",
    );
}

/// CH at 6-31G: its ROKS SCF converges (STO-3G ROKS does not) and its null
/// mode is the axis rotation (Rayleigh 4.9e-3).
#[test]
#[ignore = "slow: full-pipeline FD"]
fn ch_roks_pbe_mbd_exact_gradient_matches_fd() {
    assert_case("CH/6-31G ROKS-PBE", Reference::Roks, CH_XYZ, "6-31g", "PBE");
}

/// REFUSED, NOT PROJECTED: the CH UKS-PBE state (STO-3G and 6-31G) is a
/// saddle — besides the axis-rotation null mode, its orbital Hessian has a
/// genuinely negative eigenvalue (gap metric −0.154 / −0.127, measured by
/// `zvector_ks::degenerate_somo_investigation`). Projection removes only the
/// symmetry mode, so PCG meets the other one and the gradient is refused.
#[test]
fn ch_uks_saddle_state_is_refused() {
    let mol = Molecule::parse_xyz(CH_XYZ, 0, 2).expect("CH");
    let bs = basis::bundled("sto-3g").expect("sto-3g");
    let op = Operator::coulomb();
    let ctx = ParallelContext::default();
    let cfg = ks("PBE");
    let mbd_cfg = MbdRsscsConfig::for_functional("PBE").expect("beta");
    let cache = MbdFreeAtomCache::build(&ctx, &mol, &bs, op, &cfg).expect("cache");
    let r = scf(Reference::Uks, &mol, &bs, &cfg, None);
    let err = mbd_rsscs_for_scf(&ctx, &cache, &mol, &bs, op, &cfg, &r, &mbd_cfg)
        .expect_err("a saddle-point UKS state must be refused");
    let msg = format!("{err}");
    assert!(msg.contains("not positive definite"), "{msg}");
}

/// Measurement (prints, no assertion): every ²Π case, including the refused
/// and unasserted ones.
///
/// * CH UKS-PBE (STO-3G, 6-31G): refused — a second, genuinely negative
///   Hessian eigenvalue (−0.154 / −0.127); CH ROKS/STO-3G: the SCF does not
///   converge.
/// * NO ROKS-PBE (STO-3G and 6-31G): exact − FD = 1.1e-9 / 9.6e-10 at
///   h = 1e-3, the same with and without the projection (the mode's Rayleigh
///   quotient is positive there, so the unprojected solve also runs). It is
///   an FD artifact, not a gradient error: at STO-3G it scales as 1/h
///   (2.3e-9, 1.1e-9, 6.5e-10 at h = 5e-4, 1e-3, 3e-3; a/h + b·h² with
///   a = 1.2e-12 Ha, b from the UKS run at the same steps), unchanged by
///   tightening density_conv to 1e-11 — a ~1e-12 Ha step in the ROKS
///   pipeline energy between ±h, not a constant offset of the analytic
///   gradient.
#[test]
#[ignore = "measurement"]
fn measure_degenerate_cases() {
    for (label, rf, xyz, basis, xc) in [
        ("OH/STO-3G UKS-PBE", Reference::Uks, OH_XYZ, "sto-3g", "PBE"),
        (
            "OH/STO-3G UKS-PBE0",
            Reference::Uks,
            OH_XYZ,
            "sto-3g",
            "PBE0",
        ),
        ("OH/6-31G UKS-PBE", Reference::Uks, OH_XYZ, "6-31g", "PBE"),
        ("NO/STO-3G UKS-PBE", Reference::Uks, NO_XYZ, "sto-3g", "PBE"),
        ("CH/STO-3G UKS-PBE", Reference::Uks, CH_XYZ, "sto-3g", "PBE"),
        ("CH/6-31G UKS-PBE", Reference::Uks, CH_XYZ, "6-31g", "PBE"),
        (
            "OH/STO-3G ROKS-PBE0",
            Reference::Roks,
            OH_XYZ,
            "sto-3g",
            "PBE0",
        ),
        (
            "NO/STO-3G ROKS-PBE",
            Reference::Roks,
            NO_XYZ,
            "sto-3g",
            "PBE",
        ),
        (
            "CH/STO-3G ROKS-PBE",
            Reference::Roks,
            CH_XYZ,
            "sto-3g",
            "PBE",
        ),
        ("CH/6-31G ROKS-PBE", Reference::Roks, CH_XYZ, "6-31g", "PBE"),
        ("NO/6-31G UKS-PBE", Reference::Uks, NO_XYZ, "6-31g", "PBE"),
        ("NO/6-31G ROKS-PBE", Reference::Roks, NO_XYZ, "6-31g", "PBE"),
    ] {
        if std::env::var("MBD_FD_ONLY").is_ok_and(|o| o != label) {
            continue;
        }
        let res = std::panic::catch_unwind(|| case(label, rf, xyz, basis, xc));
        if res.is_err() {
            println!("{label}: REFUSED / failed (see panic message above)");
        }
    }
}

/// ANCHOR: on a non-linear radical no mode is detected, so the solve is the
/// unprojected one (every projection is a no-op).
#[test]
fn nh2_has_no_null_mode() {
    let mol = Molecule::parse_xyz(NH2_XYZ, 0, 2).expect("NH2");
    let bs = basis::bundled("sto-3g").expect("sto-3g");
    let op = Operator::coulomb();
    let ctx = ParallelContext::default();
    let cfg = ks("PBE0");
    let mbd_cfg = MbdRsscsConfig::for_functional("PBE0").expect("beta");
    let cache = MbdFreeAtomCache::build(&ctx, &mol, &bs, op, &cfg).expect("cache");
    for rf in [Reference::Uks, Reference::Roks] {
        let r = scf(rf, &mol, &bs, &cfg, None);
        let out = mbd_rsscs_for_scf(&ctx, &cache, &mol, &bs, op, &cfg, &r, &mbd_cfg).expect("mbd");
        assert_eq!(out.null_mode, None, "{rf:?}");
    }
}
