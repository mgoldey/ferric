//! A converged ROHF/ROKS result must describe ONE determinant: the occupied
//! blocks of `mos_alpha` (and the densities built from them) must belong to the
//! same state as `energy`, `fock_alpha` and `rohf_spin_focks`.
//!
//! # The defect
//!
//! `solve_rohf`'s exit re-diagonalizes the converged effective Fock and used to
//! label closed/open/virtual by eigenvalue INDEX unless the F6 continuity lock
//! was engaged. OH ²Π under ROKS/PBE has its open π BELOW the closed π in the
//! effective-Fock spectrum (−0.1103 vs −0.0764), so whenever the lock was not
//! engaged — always under MOM (`mom_after_iter > 0` disables the guard) — the
//! exit handed back the OTHER π determinant (π_x²π_y¹ ↔ π_y²π_x¹) while the
//! stored spin Focks still belonged to the converged one. Semi-canonicalizing
//! that mixture put the β-occupied π ABOVE the β-empty π (β gap −0.0264 Ha on
//! OH/STO-3G), the U-PDEP χ₀ scale factors `sqrt(2e/(ω²+e²))` went NaN, and
//! every U-GW@ROKS run from Python (which sets `mom_after_iter = 5`) failed in
//! LAPACK with "Full-rank PDEP eigh failed ... return_code = 69".
//!
//! # Checks (OH/STO-3G ROKS/PBE, exact J/K, `mom_after_iter = 5`)
//!
//! 1. Independent construction: the semi-canonical energies from the SCF's
//!    STORED spin Focks (`unrestricted_reference`) equal those from spin Focks
//!    REBUILT from the returned density (`semicanonicalize` with the PBE
//!    `XcSpec`). Before the fix they differ by 0.026 Ha (stored = one state,
//!    density = the other).
//! 2. The MOM run and the default (guarded) run converge to symmetry images
//!    (π_x ↔ π_y), so their energies and semi-canonical spectra must agree.
//!    Before the fix the MOM run's β occupied/virtual π energies are swapped.
//! 3. The semi-canonical β gap is positive (−0.0264 Ha before the fix).
//!
//! Run: `OPENBLAS_NUM_THREADS=1 cargo test -p ferric-scf --release --test
//! rohf_exit_keeps_occupation`

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::RhfConfig;
use ferric_scf::rohf::solve_rohf;
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::semicanonical::{semicanonicalize, unrestricted_reference, XcSpec};
use ferric_scf::ScfResult;

/// Stored-Fock vs rebuilt-Fock energies: the stored Focks are the last
/// iteration's (one diagonalization before the returned MOs), so they agree to
/// the density-convergence level; the defect is 0.026 Ha.
const TOL_STORED_VS_REBUILT: f64 = 1e-6;
/// Symmetry-image agreement (two SCF trajectories, same convergence bars).
const TOL_IMAGE: f64 = 1e-6;

struct Run {
    mol: Molecule,
    prep: PreparedBasis,
    bounds: SchwarzBounds,
    roks: ScfResult,
}

fn roks_oh(mom_after_iter: usize) -> Run {
    let mol = Molecule::parse_xyz("2\n\nO 0 0 0\nH 0 0 0.97\n", 0, 2).unwrap();
    let prep = PreparedBasis::new(&mol, &basis::bundled("sto-3g").unwrap()).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
    let cfg = RhfConfig {
        xc: Some("pbe".into()),
        mom_after_iter,
        energy_conv: 1e-10,
        density_conv: 1e-9,
        max_iter: 200,
        ..Default::default()
    };
    let roks = solve_rohf(
        &ParallelContext::default(),
        &mol,
        &prep,
        Operator::coulomb(),
        &bounds,
        &cfg,
    )
    .unwrap();
    assert!(
        roks.converged,
        "OH ROKS/PBE (mom {mom_after_iter}) did not converge"
    );
    Run {
        mol,
        prep,
        bounds,
        roks,
    }
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()))
}

/// (stored-Fock view eps α, β), (rebuilt-Fock eps α, β), β gap of the view.
fn spectra(r: &Run) -> ((Vec<f64>, Vec<f64>), (Vec<f64>, Vec<f64>), f64) {
    let view = unrestricted_reference(&r.mol, &r.roks).unwrap();
    let rebuilt = semicanonicalize(
        &ParallelContext::default(),
        &r.mol,
        &r.prep,
        &r.bounds,
        &r.roks,
        1e-12,
        Some(&XcSpec::new("pbe")),
    )
    .unwrap();
    let nb = rebuilt.nocc_beta;
    let eb = view.eps_b();
    let gap_b = eb[nb..].iter().copied().fold(f64::INFINITY, f64::min)
        - eb[..nb].iter().copied().fold(f64::NEG_INFINITY, f64::max);
    (
        (view.eps_a().to_vec(), eb.to_vec()),
        (rebuilt.eps_alpha.clone(), rebuilt.eps_beta.clone()),
        gap_b,
    )
}

#[test]
fn roks_mom_exit_returns_the_converged_determinant() {
    let mom = roks_oh(5);
    let ((sa, sb), (ra, rb), gap_b) = spectra(&mom);
    let da = max_abs_diff(&sa, &ra);
    let db = max_abs_diff(&sb, &rb);
    eprintln!(
        "MOM run: stored-vs-rebuilt |d eps| alpha {da:.2e} beta {db:.2e}; beta gap {gap_b:.5}"
    );
    assert!(
        da < TOL_STORED_VS_REBUILT && db < TOL_STORED_VS_REBUILT,
        "returned orbitals and stored spin Focks belong to different determinants: \
         |d eps| alpha {da:.3e}, beta {db:.3e}"
    );
    assert!(gap_b > 0.0, "semi-canonical beta gap {gap_b:.5} <= 0");
}

#[test]
fn roks_mom_and_guarded_runs_are_symmetry_images() {
    let mom = roks_oh(5);
    let def = roks_oh(0);
    let de = (mom.roks.energy - def.roks.energy).abs();
    let ((ma, mb), _, gap_m) = spectra(&mom);
    let ((da, db), _, gap_d) = spectra(&def);
    let ea = max_abs_diff(&ma, &da);
    let eb = max_abs_diff(&mb, &db);
    eprintln!(
        "MOM vs default: |dE| {de:.2e}, |d eps| alpha {ea:.2e} beta {eb:.2e}; \
         beta gaps {gap_m:.5} / {gap_d:.5}"
    );
    assert!(de < 1e-8, "energies differ by {de:.3e}");
    assert!(
        ea < TOL_IMAGE && eb < TOL_IMAGE,
        "semi-canonical spectra differ: alpha {ea:.3e}, beta {eb:.3e}"
    );
}
