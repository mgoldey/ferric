//! Validation tier: the MBD@rsSCS part of the open-shell (UKS / ROKS)
//! dispersion-corrected Hessian from `harmonic_frequencies_with_scf_correction`
//! against second differences of the full SCF + MBD ENERGY.
//!
//! MBD@rsSCS depends on the density through its Hirshfeld volumes, so its
//! Hessian is right only if the gradient the driver differentiates carries the
//! orbital relaxation of the reference actually solved: the unrestricted
//! Z-vector for UKS, the three-block (closed/open/virtual) Z-vector for ROKS.
//! The energy side re-solves the SCF at every displaced geometry and never
//! calls the gradient code, so it is an independent construction of the same
//! second derivative. NEGATIVE CONTROL: the same driver fed the UNRELAXED
//! gradient (no Z-vector term) must miss the bar.
//!
//! Systems (PBE/STO-3G, non-degenerate SOMOs; OH is excluded, its degenerate
//! SOMO is a separate issue): UKS NH2 doublet and CH2 triplet; ROKS HCO doublet
//! (in-plane SOMO, closed-open coupling nonzero) and CH2 triplet.
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 cargo test -p ferric-cli --release \
//!   --test validation_dispersion_frequencies_open_shell -- --ignored --nocapture
//! ```

use ferric_core::basis::{self, BasisSet};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_rpa::dispersion::{
    mbd_rsscs_for_density, mbd_rsscs_for_scf, MbdFreeAtomCache, MbdRsscsConfig,
};
use ferric_scf::frequencies::{
    harmonic_frequencies, harmonic_frequencies_with_scf_correction, FrequencyConfig,
    FrequencyReference, FrequencyResult, HessianMethod,
};
use ferric_scf::rhf::RhfConfig;
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::{ScfResult, Spin};
use ndarray::Array2;

const NH2_XYZ: &str =
    "3\nNH2\nN 0.000000 0.020000 0.142000\nH 0.050000 0.802000 -0.497000\nH 0.000000 -0.782000 -0.517000\n";
const HCO_XYZ: &str =
    "3\nHCO\nC 0.000000 0.000000 0.000000\nO 1.180000 0.000000 0.000000\nH -0.627000 0.916000 0.000000\n";
const CH2_XYZ: &str =
    "3\nCH2\nC 0.000000 0.000000 0.000000\nH 0.000000 0.990000 0.620000\nH 0.000000 -0.990000 0.620000\n";

/// Energy second-difference step, Bohr, for the 5-point stencil
/// `(-E(2h) + 16E(h) - 30E(0) + 16E(-h) - E(-2h)) / 12h^2` (O(h^4)). The MBD
/// energy of the full pipeline carries noise that the stencil amplifies as
/// 1/h^2. MEASURED worst |v^T H v - E''| over the three directions at
/// h = 0.02 / 0.04 / 0.08: CH2 triplet (UKS and ROKS alike) 4.6e-7 / 1.2e-7 /
/// 4.8e-8 (the 4x per doubling of pure noise), HCO (ROKS) 3.3e-7 / 8.1e-8 /
/// 2.3e-8, NH2 (UKS) 3.2e-9 / 9.2e-9 / 1.9e-8 (already truncation-limited).
/// h = 0.08 puts every system below 5e-8. Override with `MBD_HESS_H`.
const MBD_ENERGY_H: f64 = 0.08;
/// Bar on |v^T H_disp v - E''_MBD(v)|. Derived from both sides at h = 0.08:
/// the exact (Z-vector-relaxed) gradient gives <= 4.8e-8 on all four
/// systems; the UNRELAXED gradient (no orbital-relaxation term) gives >= 8.3e-7
/// (UKS CH2; 1.0e-6 ROKS CH2, 2.1e-6 HCO, 1.6e-5 NH2). 2e-7 sits ~4x from each.
const MBD_TOL: f64 = 2e-7;

fn pbe() -> RhfConfig {
    RhfConfig {
        xc: Some("PBE".to_string()),
        max_iter: 300,
        energy_conv: 1e-11,
        density_conv: 1e-9,
        ..Default::default()
    }
}

fn shifted(mol: &Molecule, dir: &[f64], s: f64) -> Molecule {
    let mut m = mol.clone();
    for (i, at) in m.atoms.iter_mut().enumerate() {
        at.x += s * dir[3 * i];
        at.y += s * dir[3 * i + 1];
        at.zpos += s * dir[3 * i + 2];
    }
    m
}

/// Fixed, non-symmetric probe directions (normalized), as in the closed-shell
/// test.
fn probe_directions() -> Vec<Vec<f64>> {
    let raw: [[f64; 9]; 3] = [
        [0.3, -0.1, 0.7, -0.2, 0.5, 0.1, 0.4, -0.6, -0.3],
        [0.0, 0.0, 1.0, 0.0, 0.6, -0.4, 0.0, -0.6, -0.4],
        [0.9, 0.2, -0.1, -0.5, 0.3, 0.8, 0.1, -0.7, 0.2],
    ];
    raw.iter()
        .map(|r| {
            let norm = r.iter().map(|x| x * x).sum::<f64>().sqrt();
            r.iter().map(|x| x / norm).collect()
        })
        .collect()
}

fn quad_form(h: &Array2<f64>, v: &[f64]) -> f64 {
    let n = v.len();
    let mut s = 0.0;
    for i in 0..n {
        for j in 0..n {
            s += v[i] * h[(i, j)] * v[j];
        }
    }
    s
}

/// The converged SCF of `reference` at `mol` (energy side: no gradient).
fn scf(mol: &Molecule, bs: &BasisSet, cfg: &RhfConfig, reference: FrequencyReference) -> ScfResult {
    let ctx = ParallelContext::default();
    let prep = PreparedBasis::new(mol, bs).expect("prep");
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).expect("schwarz");
    let r = match reference {
        FrequencyReference::Uhf => ferric_scf::solve_uhf(&ctx, mol, &prep, &bounds, cfg),
        FrequencyReference::Rohf => ferric_scf::solve_rohf(&ctx, mol, &prep, op, &bounds, cfg),
        FrequencyReference::Rhf => unreachable!("open-shell validation only"),
    }
    .expect("scf");
    assert!(r.converged, "SCF did not converge");
    r
}

#[allow(clippy::too_many_arguments)]
fn mbd_frequencies(
    mol: &Molecule,
    bs: &BasisSet,
    cfg: &RhfConfig,
    reference: FrequencyReference,
    spin: Spin,
    cache: &MbdFreeAtomCache,
    mcfg: &MbdRsscsConfig,
    relaxed: bool,
) -> FrequencyResult {
    let ctx = ParallelContext::default();
    harmonic_frequencies_with_scf_correction(
        &ctx,
        mol,
        &bs.name,
        Operator::coulomb(),
        cfg,
        &FrequencyConfig {
            reference,
            ..Default::default()
        },
        |m, scf| {
            assert_eq!(scf.spin, spin, "the correction saw the wrong SCF");
            let r = mbd_rsscs_for_scf(&ctx, cache, m, bs, Operator::coulomb(), cfg, scf, mcfg)?;
            // `relaxed = false` is the deliberate defect (no Z-vector term).
            let g = if relaxed {
                r.gradient
            } else {
                r.gradient_unrelaxed
            };
            Ok((r.energy, g))
        },
    )
    .expect("mbd frequencies")
}

fn check_mbd(
    label: &str,
    xyz: &str,
    multiplicity: usize,
    reference: FrequencyReference,
    spin: Spin,
) {
    let mol = Molecule::parse_xyz(xyz, 0, multiplicity).expect("molecule");
    let bs = basis::bundled("sto-3g").expect("sto-3g");
    let cfg = pbe();
    let ctx = ParallelContext::default();
    let op = Operator::coulomb();
    let mcfg = MbdRsscsConfig::for_functional("PBE").expect("beta");
    let cache = MbdFreeAtomCache::build(&ctx, &mol, &bs, op, &cfg).expect("cache");
    let plain = harmonic_frequencies(
        &ctx,
        &mol,
        "sto-3g",
        op,
        &cfg,
        &FrequencyConfig {
            reference,
            hessian: HessianMethod::FiniteDifference,
            ..Default::default()
        },
    )
    .expect("plain");
    let with = mbd_frequencies(&mol, &bs, &cfg, reference, spin, &cache, &mcfg, true);
    let h_disp = &with.cartesian_hessian - &plain.cartesian_hessian;
    let unrelaxed = mbd_frequencies(&mol, &bs, &cfg, reference, spin, &cache, &mcfg, false);
    let h_unrelaxed = &unrelaxed.cartesian_hessian - &plain.cartesian_hessian;
    // The MBD energy depends on the total density only.
    let e_mbd = |m: &Molecule| {
        let d = scf(m, &bs, &cfg, reference);
        mbd_rsscs_for_density(&cache, m, &bs, d.density_total(), &mcfg, false)
            .expect("mbd")
            .energy
    };
    let h: f64 = std::env::var("MBD_HESS_H")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(MBD_ENERGY_H);
    let e0 = e_mbd(&mol);
    assert_eq!(with.correction_energy, e0, "{label}: reported correction");
    let (mut worst, mut worst_unrelaxed, mut scale): (f64, f64, f64) = (0.0, 0.0, 0.0);
    for v in probe_directions() {
        // 5-point stencil, O(h^4) truncation.
        let at = |k: f64| e_mbd(&shifted(&mol, &v, k * h));
        let fd =
            (-at(2.0) + 16.0 * at(1.0) - 30.0 * e0 + 16.0 * at(-1.0) - at(-2.0)) / (12.0 * h * h);
        let an = quad_form(&h_disp, &v);
        let un = quad_form(&h_unrelaxed, &v);
        println!(
            "{label} MBD v^T H v: driver = {an:+.6e}, E'' (h={h}) = {fd:+.6e}, diff = {:.3e}; \
             unrelaxed diff = {:.3e}",
            an - fd,
            un - fd
        );
        worst = worst.max((an - fd).abs());
        worst_unrelaxed = worst_unrelaxed.max((un - fd).abs());
        scale = scale.max(fd.abs());
    }
    let tr_max = with
        .trans_rot_frequencies
        .iter()
        .fold(0.0_f64, |m, x| m.max(x.abs()));
    println!(
        "{label} MBD: max|v^T H v| = {scale:.3e}, worst diff = {worst:.3e}, unrelaxed worst = \
         {worst_unrelaxed:.3e}, asym(with) = {:.3e}, E_MBD = {e0:.6e}, freqs {:?} vs PBE {:?}, \
         projected trans/rot max {tr_max:.3e} cm^-1",
        with.asymmetry, with.frequencies, plain.frequencies
    );
    assert!(worst < MBD_TOL, "{label}: MBD Hessian vs E'': {worst:.3e}");
    assert!(
        scale > 10.0 * MBD_TOL,
        "{label}: MBD Hessian not resolvable at the bar"
    );
    assert!(
        worst_unrelaxed > MBD_TOL,
        "{label}: the check must SEE a missing orbital relaxation: {worst_unrelaxed:.3e}"
    );
}

#[test]
#[ignore = "validation: Harmonic frequencies with dispersion (open shell)"]
fn uks_nh2_mbd_hessian_matches_energy_second_differences() {
    check_mbd(
        "UKS NH2",
        NH2_XYZ,
        2,
        FrequencyReference::Uhf,
        Spin::Unrestricted,
    );
}

#[test]
#[ignore = "validation: Harmonic frequencies with dispersion (open shell)"]
fn uks_ch2_triplet_mbd_hessian_matches_energy_second_differences() {
    check_mbd(
        "UKS CH2(T)",
        CH2_XYZ,
        3,
        FrequencyReference::Uhf,
        Spin::Unrestricted,
    );
}

#[test]
#[ignore = "validation: Harmonic frequencies with dispersion (open shell)"]
fn roks_hco_mbd_hessian_matches_energy_second_differences() {
    check_mbd(
        "ROKS HCO",
        HCO_XYZ,
        2,
        FrequencyReference::Rohf,
        Spin::RestrictedOpen,
    );
}

#[test]
#[ignore = "validation: Harmonic frequencies with dispersion (open shell)"]
fn roks_ch2_triplet_mbd_hessian_matches_energy_second_differences() {
    check_mbd(
        "ROKS CH2(T)",
        CH2_XYZ,
        3,
        FrequencyReference::Rohf,
        Spin::RestrictedOpen,
    );
}
