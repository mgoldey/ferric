//! MBD@rsSCS-on-SCF nuclear gradient for restricted open-shell (ROKS)
//! references.
//!
//! Decomposed as the closed-shell and UKS gradients
//! (`tests/mbd_scf_gradient.rs`, `tests/mbd_scf_gradient_uks.rs`):
//!   FD_full  — central FD of the pipeline with the ROKS SCF re-solved at every
//!              displaced geometry: the true derivative.
//!   FD_orth  — the reference closed and open orbitals kept and Löwdin
//!              re-orthonormalized AS ONE SET in the displaced basis: what
//!              `gradient_unrelaxed` models (D_α = C'C'ᵀ, D_β from the closed
//!              columns of the same C').
//!   FD_full − FD_orth = the orbital relaxation, against the ROKS Z-vector
//!              term.
//!
//! Fast tests (run by default):
//!   * `roks_path_on_a_closed_shell_reproduces_the_closed_shell_gradient` — the
//!     exactness anchor: an RKS result re-labelled as ROKS (no open orbitals)
//!     must give the closed-shell gradient term by term.
//!   * `roks_mbd_unrelaxed_gradient_matches_fd_with_reorthonormalized_orbitals`
//!     — HCO (doublet) and CH2 (triplet) at STO-3G, PBE.
//!   * `roks_mbd_exact_gradient_matches_fd_of_full_scf_pipeline` — the same.
//!   * `roks_zvector_refuses_unsupported_references`
//!
//! Measurement (`#[ignore]`, prints): `measure_roks_full_pipeline_fd` — HCO,
//! NH2, CH2 (triplet) and O2 (triplet) at 6-31G with PBE, PBE0 and HSE06,
//! plus HCO with PBE + RI-J.
//!
//! NOT OH or another radical with a degenerate SOMO pair: the open-shell
//! orbital Hessian then has a near-null rotation mode and the Z-vector PCG
//! stops on ~zero curvature (the guard in `zvector_ks` reports it).
//!
//!   OPENBLAS_NUM_THREADS=1 cargo test -p ferric-rpa --test mbd_scf_gradient_roks --release
//!   OPENBLAS_NUM_THREADS=1 cargo test -p ferric-rpa --test mbd_scf_gradient_roks --release \
//!     -- --ignored --nocapture

use ferric_core::basis::{self, BasisSet};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_rpa::dispersion::{
    mbd_rsscs_for_restricted_open_densities, mbd_rsscs_for_scf, MbdFreeAtomCache, MbdRsscsConfig,
};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::rohf::solve_rohf;
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::{ScfResult, Spin};
use ndarray::{s, Array2};

const H2O_XYZ: &str = "3\nwater\nO 0.000000 0.000000 0.117300\nH 0.000000 0.757200 -0.469200\nH 0.000000 -0.757200 -0.469200\n";
// Bent off the symmetric geometry so that no gradient component is zero by
// symmetry (every Cartesian is exercised).
const NH2_XYZ: &str =
    "3\nNH2\nN 0.000000 0.020000 0.142000\nH 0.050000 0.802000 -0.497000\nH 0.000000 -0.782000 -0.517000\n";
// Formyl radical: the SOMO is the in-plane (a') carbon sp lobe, the same
// symmetry as closed orbitals, so the closed–open cross terms (V_co, the κ_oc
// rotations) are nonzero. In NH2 or O2 they vanish by symmetry (π SOMO
// against σ closed orbitals; g/u parity), which makes those molecules blind
// to every closed–open term.
const HCO_XYZ: &str =
    "3\nHCO\nC 0.000000 0.000000 0.000000\nO 1.180000 0.000000 0.000000\nH -0.627000 0.916000 0.000000\n";
// Triplet methylene (3B1): two open orbitals, the a1 one sharing symmetry
// with closed a1 orbitals.
const CH2_XYZ: &str =
    "3\nCH2\nC 0.000000 0.000000 0.000000\nH 0.000000 0.990000 0.620000\nH 0.000000 -0.990000 0.620000\n";
const O2_XYZ: &str = "2\nO2\nO 0.000000 0.000000 0.604000\nO 0.000000 0.020000 -0.604000\n";

fn ks(xc: &str) -> RhfConfig {
    RhfConfig {
        xc: Some(xc.to_string()),
        max_iter: 300,
        energy_conv: 1e-11,
        density_conv: 1e-9,
        ..Default::default()
    }
}

/// FD step (Bohr); `MBD_FD_H` overrides it.
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

/// ROKS at `mol`, started from `guess`'s total density when given (the
/// displaced points of FD_full), so every point lands on the same state.
fn roks(mol: &Molecule, bs: &BasisSet, cfg: &RhfConfig, guess: Option<&ScfResult>) -> ScfResult {
    let (prep, bounds) = bounds_for(mol, bs);
    let ctx = ParallelContext::default();
    let mut c = cfg.clone();
    if let Some(g) = guess {
        c.init_guess_density = Some(g.density_total.clone());
    }
    let r = solve_rohf(&ctx, mol, &prep, Operator::coulomb(), &bounds, &c).expect("roks");
    assert!(r.converged, "ROKS did not converge");
    assert!(matches!(r.spin, Spin::RestrictedOpen));
    r
}

/// (closed, open) orbital counts of a high-spin ROKS state.
fn n_closed_open(mol: &Molecule) -> (usize, usize) {
    let nelec = mol.nelec() as usize;
    let two_s = mol.multiplicity - 1;
    ((nelec - two_s) / 2, two_s)
}

/// Löwdin-orthonormalize the columns of `c` in metric `s`: C (Cᵀ S C)^{-1/2}.
fn lowdin(s: &Array2<f64>, c: &Array2<f64>) -> Array2<f64> {
    use ndarray_linalg::{Eigh, UPLO};
    let m = c.t().dot(s).dot(c);
    let (w, v) = m.eigh(UPLO::Lower).expect("eigh");
    let mut vs = v.clone();
    for (j, wj) in w.iter().enumerate() {
        let f = 1.0 / wj.sqrt();
        vs.column_mut(j).mapv_inplace(|x| x * f);
    }
    c.dot(&vs.dot(&v.t()))
}

/// Central FD (step `h`) of E_MBD with the spin densities at each displaced
/// geometry from `densities(m)`.
fn fd_mbd(
    mol: &Molecule,
    bs: &BasisSet,
    cache: &MbdFreeAtomCache,
    mbd_cfg: &MbdRsscsConfig,
    h: f64,
    mut densities: impl FnMut(&Molecule) -> (Array2<f64>, Array2<f64>),
) -> Array2<f64> {
    let n = mol.atoms.len();
    let mut fd = Array2::<f64>::zeros((n, 3));
    for a in 0..n {
        for k in 0..3 {
            let mut e = |sg: f64| {
                let m = displaced(mol, a, k, sg * h);
                let (da, db) = densities(&m);
                mbd_rsscs_for_restricted_open_densities(cache, &m, bs, &da, &db, mbd_cfg, false)
                    .expect("mbd")
                    .energy
            };
            fd[(a, k)] = (e(1.0) - e(-1.0)) / (2.0 * h);
        }
    }
    fd
}

/// FD_orth: the reference closed + open orbitals, Löwdin re-orthonormalized
/// as one set in each displaced basis.
fn fd_orth(
    mol: &Molecule,
    bs: &BasisSet,
    cache: &MbdFreeAtomCache,
    mbd_cfg: &MbdRsscsConfig,
    r: &ScfResult,
    h: f64,
) -> Array2<f64> {
    let (nc, no) = n_closed_open(mol);
    let c_occ = r.mos_alpha.slice(s![.., ..nc + no]).to_owned();
    fd_mbd(mol, bs, cache, mbd_cfg, h, |m| {
        let prep = PreparedBasis::new(m, bs).expect("prep");
        let sm = ferric_integrals::oneelectron::overlap(&prep);
        let cp = lowdin(&sm, &c_occ);
        let cc = cp.slice(s![.., ..nc]);
        (cp.dot(&cp.t()), cc.dot(&cc.t()))
    })
}

/// FD_full with the ROKS SCF re-solved at every displaced point, plus the
/// state-continuity diagnostic: max over points of max|D_σ(R ± h) − D_σ(R)| / h
/// (O(1) per Bohr on one state; a jump to another state is O(1) itself, i.e.
/// O(1/h) after the division).
#[allow(clippy::too_many_arguments)]
fn fd_full(
    mol: &Molecule,
    bs: &BasisSet,
    cache: &MbdFreeAtomCache,
    mbd_cfg: &MbdRsscsConfig,
    cfg: &RhfConfig,
    r: &ScfResult,
    h: f64,
) -> (Array2<f64>, f64) {
    let mut worst = 0.0_f64;
    let d_b0 = r.density_beta.clone().expect("beta");
    let fd = fd_mbd(mol, bs, cache, mbd_cfg, h, |m| {
        let rm = roks(m, bs, cfg, Some(r));
        let da = rm.density_alpha.clone();
        let db = rm.density_beta.clone().expect("beta");
        let jump = max_abs(&(&da - &r.density_alpha)).max(max_abs(&(&db - &d_b0))) / h;
        worst = worst.max(jump);
        (da, db)
    });
    (fd, worst)
}

/// Bar on the state-continuity diagnostic of `fd_full` (per Bohr): a smooth
/// single-state FD has max|ΔD|/h of order the density's geometric derivative,
/// a state flip is ~1/h = 1e3 at the default step. Measured ≤ 0.39 (HCO,
/// CH2 / STO-3G) and ≤ 0.34 (HCO, NH2, CH2, O2 / 6-31G) at h = 1e-3.
const STATE_JUMP_TOL: f64 = 20.0;

/// The converged RKS result re-expressed as a ROKS one with no open orbitals:
/// same MOs, D_α = D_β = ½ D, both spin Focks = F.
fn as_restricted_open(r: &ScfResult) -> ScfResult {
    let mut u = r.clone();
    u.spin = Spin::RestrictedOpen;
    u.density_alpha = 0.5 * r.density_r();
    u.density_beta = Some(0.5 * r.density_r());
    u.rohf_spin_focks = Some((r.fock_alpha.clone(), r.fock_alpha.clone()));
    u
}

/// Bar of the exactness anchor (absolute, Hartree/Bohr). The energy and the
/// unrelaxed and orthonormality terms are bit-identical; the relaxation term
/// differs only through the t of its central difference and the polarized
/// vs closed-shell XC gradient code. Measured 2.7e-14 (PBE), 2.6e-14 (PBE0),
/// 2.5e-14 (HSE06) on the relaxation term (7.7e-6); the energy, unrelaxed and
/// orthonormality terms are bit-identical.
const ANCHOR_TOL: f64 = 1e-12;

/// EXACTNESS ANCHOR: an RKS result run through the ROKS path (no open
/// orbitals) must give the closed-shell MBD gradient — unrelaxed,
/// orthonormality and relaxation terms separately. Catches: a wrong factor on
/// the vc right-hand side (−4), the contraction (½), the F⁺ = F_α + F_β
/// diagonal blocks, the spin-resolved exchange, W's closed–closed term, or
/// the ROKS orthonormality term's D_α/P_c pieces.
#[test]
fn roks_path_on_a_closed_shell_reproduces_the_closed_shell_gradient() {
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
            &as_restricted_open(&r),
            &mbd_cfg,
        )
        .expect("roks");
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
        assert!(max_abs(gr) > 1e6 * ANCHOR_TOL, "{xc}: relaxation too small");
    }
}

/// Bar for the unrelaxed analytic gradient against FD_orth at `fd_step()`.
/// Measured 2.2e-11 (HCO) and 1.4e-11 (CH2) at h = 1e-3. The closed–open cross
/// term it must resolve is 3.2e-7 (HCO) and 7.1e-8 (CH2); dropping it misses
/// by exactly that (mutation-tested).
const FD_ORTH_TOL: f64 = 5e-10;

/// Open-shell molecules of the fast tests: (name, xyz, multiplicity).
const FAST_CASES: [(&str, &str, usize); 2] = [("HCO", HCO_XYZ, 2), ("CH2", CH2_XYZ, 3)];

/// The model test of the fixed-orbital part, HCO and triplet CH2 at STO-3G,
/// PBE. Catches: the ROKS orthonormality term dropped, its closed–open cross
/// term dropped (asserted resolvable at the bar) or mis-weighted, i.e. the
/// UKS form −Σ_σ D_σ Sˣ D_σ substituted.
#[test]
fn roks_mbd_unrelaxed_gradient_matches_fd_with_reorthonormalized_orbitals() {
    let bs = basis::bundled("sto-3g").expect("sto-3g");
    let op = Operator::coulomb();
    let ctx = ParallelContext::default();
    let cfg = ks("PBE");
    let mbd_cfg = MbdRsscsConfig::for_functional("PBE").expect("beta");
    for (name, xyz, mult) in FAST_CASES {
        let mol = Molecule::parse_xyz(xyz, 0, mult).expect("mol");
        let cache = MbdFreeAtomCache::build(&ctx, &mol, &bs, op, &cfg).expect("cache");
        let r = roks(&mol, &bs, &cfg, None);
        let d_a = &r.density_alpha;
        let d_b = r.density_beta.as_ref().unwrap();
        let out =
            mbd_rsscs_for_restricted_open_densities(&cache, &mol, &bs, d_a, d_b, &mbd_cfg, true)
                .expect("mbd");
        let g = out.gradient_unrelaxed.clone().expect("unrelaxed");
        let orth = out.gradient_orthonormality.clone().expect("orth");
        // The closed–open cross piece of the orthonormality term on its own.
        let v = out.density_derivative.as_ref().expect("V");
        let p_o = d_a - d_b;
        let x = d_b.dot(v).dot(&p_o);
        let (prep, _) = bounds_for(&mol, &bs);
        let cross =
            -ferric_scf::gradient::overlap_deriv_contract(&prep, &(0.5 * (&x + &x.t()))).unwrap();
        let fd = fd_orth(&mol, &bs, &cache, &mbd_cfg, &r, fd_step());
        let diff = max_abs(&(&g - &fd));
        println!(
            "{name}/STO-3G PBE ROKS: max|g| = {:.3e}, max|orth| = {:.3e}, max|cross| = {:.3e}, \
             max|g - FD_orth| = {diff:.3e}",
            max_abs(&g),
            max_abs(&orth),
            max_abs(&cross)
        );
        assert!(
            diff < FD_ORTH_TOL,
            "{name}: unrelaxed vs FD_orth: {diff:.3e} >= {FD_ORTH_TOL:.1e}"
        );
        assert!(
            max_abs(&cross) > 10.0 * FD_ORTH_TOL,
            "{name}: closed-open cross term {:.3e} not resolvable",
            max_abs(&cross)
        );
    }
}

/// Bar on the EXACT ROKS gradient against FD_full at `fd_step()`. Measured
/// 2.1e-11 (HCO) and 1.4e-11 (CH2) at STO-3G, h = 1e-3; relaxation vs
/// FD_full − FD_orth 2.8e-12 and 6.0e-13, against relaxation terms of 2.8e-6
/// and 4.3e-7. The smallest mutation miss (a dropped off-diagonal Fock
/// coupling of the Hessian) is 3.1e-9.
const FD_FULL_TOL: f64 = 5e-10;

/// THE exactness test, fast: HCO and triplet CH2 at STO-3G, PBE, exact
/// gradient vs FD of the full ROKS + MBD pipeline. Catches: the relaxation
/// term dropped or mis-scaled (asserted resolvable), a dropped rotation block,
/// a wrong RHS factor, a missing off-diagonal Fock coupling of the Hessian, a
/// wrong Ẇ. Also checks translation invariance, the relaxation term against
/// FD_full − FD_orth, and that every FD point stayed on the reference state.
#[test]
fn roks_mbd_exact_gradient_matches_fd_of_full_scf_pipeline() {
    let bs = basis::bundled("sto-3g").expect("sto-3g");
    let op = Operator::coulomb();
    let ctx = ParallelContext::default();
    let cfg = ks("PBE");
    let mbd_cfg = MbdRsscsConfig::for_functional("PBE").expect("beta");
    for (name, xyz, mult) in FAST_CASES {
        let mol = Molecule::parse_xyz(xyz, 0, mult).expect("mol");
        let cache = MbdFreeAtomCache::build(&ctx, &mol, &bs, op, &cfg).expect("cache");
        let r = roks(&mol, &bs, &cfg, None);
        let out = mbd_rsscs_for_scf(&ctx, &cache, &mol, &bs, op, &cfg, &r, &mbd_cfg).expect("mbd");
        let g = out.gradient.expect("exact");
        let relax = out.gradient_relaxation.expect("relaxation");
        let h = fd_step();
        let (fd, jump) = fd_full(&mol, &bs, &cache, &mbd_cfg, &cfg, &r, h);
        let fdo = fd_orth(&mol, &bs, &cache, &mbd_cfg, &r, h);
        let relax_fd = &fd - &fdo;
        let diff = max_abs(&(&g - &fd));
        let diff_relax = max_abs(&(&relax - &relax_fd));
        let tsum = (0..3)
            .map(|k| g.column(k).sum().abs())
            .fold(0.0_f64, f64::max);
        println!(
            "{name}/STO-3G PBE ROKS: max|FD| = {:.3e}, max|relax| = {:.3e}, max|exact - FD| = \
             {diff:.3e}, max|relax(Z) - relax(FD)| = {diff_relax:.3e}, max|sum_B g| = \
             {tsum:.1e}, max|dD|/h = {jump:.3}",
            max_abs(&fd),
            max_abs(&relax)
        );
        assert!(
            jump < STATE_JUMP_TOL,
            "{name}: an FD point left the state: {jump:.3}"
        );
        assert!(
            diff < FD_FULL_TOL,
            "{name}: exact vs FD_full: {diff:.3e} >= {FD_FULL_TOL:.1e}"
        );
        assert!(
            diff_relax < FD_FULL_TOL,
            "{name}: relaxation vs FD: {diff_relax:.3e} >= {FD_FULL_TOL:.1e}"
        );
        assert!(
            tsum < 1e-10 * max_abs(&g),
            "{name}: sum over atoms of the exact gradient: {tsum:.3e}"
        );
        assert!(
            max_abs(&relax) > 10.0 * FD_FULL_TOL,
            "{name}: relaxation term {:.3e} is not resolvable at the bar",
            max_abs(&relax)
        );
    }
}

/// The ROKS Z-vector refuses what it cannot answer exactly, with a reason:
/// meta-GGA (no τ kernel), VV10, COSX exchange (no ROKS COSX gradient), an
/// xc_omega override, solvation and cDFT; and a plain GGA/hybrid/RSH is
/// accepted. `mbd_rsscs_for_scf` surfaces the same reason.
#[test]
fn roks_zvector_refuses_unsupported_references() {
    use ferric_scf::zvector_ks::unsupported_reason_roks;
    for xc in ["PBE", "PBE0", "HSE06", "B3LYP"] {
        assert_eq!(unsupported_reason_roks(&ks(xc)), None, "{xc}");
    }
    let cases: Vec<(&str, RhfConfig, &str)> = vec![
        ("SCAN", ks("SCAN"), "meta-GGA"),
        ("wB97X-V", ks("wB97X-V"), "VV10"),
        (
            "PBE0 cosx",
            RhfConfig {
                k_builder: Some("cosx".into()),
                ..ks("PBE0")
            },
            "COSX",
        ),
        (
            "PBE xc_omega",
            RhfConfig {
                xc_omega: Some(0.3),
                ..ks("PBE")
            },
            "xc_omega",
        ),
        (
            "PBE cosmo",
            RhfConfig {
                cosmo: Some(Default::default()),
                ..ks("PBE")
            },
            "solvation",
        ),
        ("HF", RhfConfig::default(), "Kohn-Sham"),
    ];
    for (label, cfg, word) in cases {
        let r = unsupported_reason_roks(&cfg).unwrap_or_else(|| panic!("{label} accepted"));
        assert!(r.contains(word), "{label}: {r}");
    }
    // End to end: a COSX ROKS config is refused by the MBD gradient driver
    // before anything is solved for it (the SCF here is plain exact-K).
    let mol = Molecule::parse_xyz(NH2_XYZ, 0, 2).expect("NH2");
    let bs = basis::bundled("sto-3g").expect("sto-3g");
    let op = Operator::coulomb();
    let ctx = ParallelContext::default();
    let cfg = ks("PBE0");
    let mbd_cfg = MbdRsscsConfig::for_functional("PBE0").expect("beta");
    let cache = MbdFreeAtomCache::build(&ctx, &mol, &bs, op, &cfg).expect("cache");
    let r = roks(&mol, &bs, &cfg, None);
    let cosx = RhfConfig {
        k_builder: Some("cosx".into()),
        ..cfg
    };
    let err = mbd_rsscs_for_scf(&ctx, &cache, &mol, &bs, op, &cosx, &r, &mbd_cfg)
        .expect_err("COSX ROKS must be refused");
    assert!(format!("{err:?}").contains("COSX"), "{err:?}");
}

/// Measurement, not a validation: HCO, NH2, CH2 (triplet) and O2 (triplet) at
/// 6-31G with PBE, PBE0 and HSE06, and HCO with PBE + RI-J; prints exact vs FD_full,
/// relaxation vs FD_full − FD_orth, unrelaxed vs FD_orth, translation
/// invariance and the state-continuity diagnostic. `MBD_FD_ONLY=<name>`,
/// `MBD_FD_XC=<tag>` and `MBD_FD_H=<h>` select a molecule, a variant and the
/// step.
#[test]
#[ignore = "measurement: full-pipeline ROKS FD at 6-31G, minutes"]
fn measure_roks_full_pipeline_fd() {
    let bs = basis::bundled("6-31g").expect("6-31g");
    let op = Operator::coulomb();
    let ctx = ParallelContext::default();
    let h = fd_step();
    let only: Option<String> = std::env::var("MBD_FD_ONLY").ok();
    let only_xc: Option<String> = std::env::var("MBD_FD_XC").ok();
    let cases = [
        ("HCO", HCO_XYZ, 2),
        ("NH2", NH2_XYZ, 2),
        ("CH2", CH2_XYZ, 3),
        ("O2", O2_XYZ, 3),
    ];
    for (name, xyz, mult) in cases {
        if only.as_deref().is_some_and(|o| o != name) {
            continue;
        }
        let jk = "def2-universal-jkfit";
        let mut variants = vec![
            ("PBE", "PBE", None),
            ("PBE0", "PBE0", None),
            ("HSE06", "HSE06", None),
        ];
        if name == "HCO" {
            variants.push(("PBE RI-J", "PBE", Some(jk)));
        }
        for (tag, xc, j_aux) in variants {
            if only_xc.as_deref().is_some_and(|o| o != tag) {
                continue;
            }
            let label = format!("{name}/6-31G {tag}");
            let cfg = RhfConfig {
                df_j_aux: j_aux.map(str::to_string),
                ..ks(xc)
            };
            let mbd_cfg = MbdRsscsConfig::for_functional(xc).expect("beta");
            let mol = Molecule::parse_xyz(xyz, 0, mult).expect("mol");
            let cache = MbdFreeAtomCache::build(&ctx, &mol, &bs, op, &cfg).expect("cache");
            let r = roks(&mol, &bs, &cfg, None);
            let out =
                mbd_rsscs_for_scf(&ctx, &cache, &mol, &bs, op, &cfg, &r, &mbd_cfg).expect("mbd");
            let g = out.gradient.clone().expect("exact");
            let gu = out.gradient_unrelaxed.clone().expect("unrelaxed");
            let gr = out.gradient_relaxation.clone().expect("relaxation");
            let (fd, jump) = fd_full(&mol, &bs, &cache, &mbd_cfg, &cfg, &r, h);
            let fdo = fd_orth(&mol, &bs, &cache, &mbd_cfg, &r, h);
            let relax_fd = &fd - &fdo;
            let tsum = (0..3)
                .map(|k| g.column(k).sum().abs())
                .fold(0.0_f64, f64::max);
            println!(
                "{label}: E_MBD = {:.10e}; max|FD_full| = {:.4e}; max|exact - FD_full| = {:.3e}; \
                 max|relax(Z) - relax(FD)| = {:.3e}; max|unrelaxed - FD_orth| = {:.3e}; \
                 max|relax| = {:.3e}; max|sum_B g| = {tsum:.1e}; max|dD|/h = {jump:.3}",
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
