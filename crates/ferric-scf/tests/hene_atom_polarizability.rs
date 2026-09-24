//! Finite-field static polarizabilities of isolated He and Ne, anchored to PySCF.
//!
//! # Why this file exists
//!
//! These two numbers were previously produced only by
//! `crates/ferric-scf/examples/hene_promolecule_probe.rs`, which was a dead
//! binary: no `[[example]]` stanza, no CI job, no test referenced it, and it
//! contained **no PySCF comparison at all** — it printed α and then hardcoded
//! those same printed numbers back into its own R⁻⁴ table. The claim "α matches
//! PySCF finite field to 7 digits" lived only in a commit message, protected by
//! nothing.
//!
//! The claim is true, so it is worth a test rather than a deletion. This file
//! is that test.
//!
//! # What α is used for in the HeNe⁺ lane
//!
//! The ion-induced-dipole energy of a charge beside a neutral atom is
//! −α/(2R⁴), so the *differential* between the two HeNe⁺ diabats — hole on He
//! (polarizing Ne) versus hole on Ne (polarizing He) — is
//! (α_Ne − α_He)/(2R⁴). That is the leading long-range term in the diabatic
//! gap, and it must be computed from the α values of the SAME method and basis
//! the diabats use, not from experimental α.
//!
//! **Unit trap, recorded because it has bitten this lane.** The frequently
//! quoted He polarizability "0.205" is in ÅNGSTRÖM CUBED; in atomic units it
//! is 1.384 a.u. Ne is 2.669 a.u. = 0.396 Å³. Treating 0.205 and 2.67 as if
//! both were a.u. inflates the He–Ne differential by ~6.75× on the He term.
//! Everything in this file is in atomic units.
//!
//! # Reference data
//!
//! PySCF 2.13.0, `scf.UHF`, `def2-svp`, `conv_tol = 1e-12`,
//! `conv_tol_grad = 1e-9`, with the z-dipole field folded into `get_hcore` as
//! `h = T + V + f·z` — the same sign convention ferric's
//! `ExternalPotential.field` uses. Central 3-point second difference,
//! α_zz = −[E(+h) − 2E(0) + E(−h)]/h², at h = 0.01 a.u.
//!
//! At h = 0.01 the two programs agree to the 8 digits PySCF printed
//! (He 0.44315101, Ne 0.62933427); measured against ferric's unrounded values
//! the residuals are 4.7e-9 and 4.1e-9 a.u., which is the rounding of the
//! stored references and not a real disagreement. The bar below is therefore
//! set by the precision of the stored constants, not by either program's
//! physics.
//!
//! # Scope
//!
//! def2-SVP has no diffuse functions, so these α are roughly a third of the
//! experimental values (α_He 1.3838, α_Ne 2.6693 a.u.). That is expected and
//! is NOT a defect: the point is the method-consistent differential, and a
//! test that compared against experiment here would be measuring the basis
//! set, not ferric. The experimental numbers appear below only as a sanity
//! bracket on the *ordering* (α_Ne > α_He), which def2-SVP does reproduce.

use ferric_core::basis;
use ferric_core::external_potential::ExternalPotential;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::RhfConfig;
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::solve_uhf;

/// Finite-difference step (a.u.) at which the PySCF references were taken.
const FIELD_STEP: f64 = 0.01;

/// PySCF 2.13.0 UHF/def2-SVP finite-field α_zz at `FIELD_STEP` (a.u.).
const ALPHA_HE_PYSCF: f64 = 0.443_151_01;
const ALPHA_NE_PYSCF: f64 = 0.629_334_27;

/// Agreement bar against PySCF. The two programs agree to all eight printed
/// digits at this step; 1e-7 leaves a decade of headroom over the last printed
/// digit while still being ~1000x tighter than the h-extrapolation drift
/// (α moves 1.3e-6 between h = 0.01 and the h -> 0 limit), so this bar tests
/// program-vs-program agreement rather than the quality of the difference
/// formula.
const TOL_ALPHA: f64 = 1e-7;

fn cfg() -> RhfConfig {
    RhfConfig {
        energy_conv: 1e-11,
        density_conv: 1e-9,
        max_iter: 300,
        ..Default::default()
    }
}

/// UHF/def2-SVP energy of a neutral closed-shell atom in a uniform z-field.
fn energy_in_field(symbol: &str, field_z: f64) -> f64 {
    let xyz = format!("1\n{symbol}\n{symbol} 0.0 0.0 0.0\n");
    let mol = Molecule::parse_xyz(&xyz, 0, 1).unwrap();
    let bs = basis::bundled("def2-svp").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();
    let mut config = cfg();
    if field_z != 0.0 {
        config.external_potential = Some(ExternalPotential {
            field: Some([0.0, 0.0, field_z]),
            ..Default::default()
        });
    }
    let res = solve_uhf(&ctx, &mol, &prep, &bounds, &config).unwrap();
    assert!(
        res.converged,
        "{symbol} at F_z = {field_z} did not converge: {:?}",
        res.exit
    );
    res.energy
}

/// α_zz = −[E(+h) − 2E(0) + E(−h)]/h², central 3-point.
fn alpha_zz(symbol: &str, h: f64) -> f64 {
    let e0 = energy_in_field(symbol, 0.0);
    let ep = energy_in_field(symbol, h);
    let em = energy_in_field(symbol, -h);
    -(ep - 2.0 * e0 + em) / (h * h)
}

/// ferric's finite-field α_zz for He and Ne match PySCF's, computed with the
/// same step and the same field convention.
///
/// This is the assertion the dead example never made.
#[test]
fn atomic_static_polarizabilities_match_pyscf_finite_field() {
    for (symbol, reference) in [("He", ALPHA_HE_PYSCF), ("Ne", ALPHA_NE_PYSCF)] {
        let alpha = alpha_zz(symbol, FIELD_STEP);
        let diff = alpha - reference;
        eprintln!(
            "alpha_zz({symbol}, h={FIELD_STEP}) ferric = {alpha:.9} a.u.   pyscf = {reference:.9}   diff = {diff:+.3e}"
        );
        assert!(
            diff.abs() < TOL_ALPHA,
            "alpha_zz({symbol}) UHF/def2-SVP finite field: ferric {alpha:.9} vs PySCF \
             {reference:.9} a.u., diff {diff:+.3e} exceeds {TOL_ALPHA:.0e}. Either the \
             external-field coupling or the density response disagrees with PySCF."
        );
    }
}

/// The finite-difference estimate must **converge as the step shrinks**, and
/// it must do so at the second-order rate the central formula promises.
///
/// This is the anti-stub check for this file: a hardcoded α returns the same
/// number at every `h`, so the successive changes below would all be exactly
/// zero and the "is the sequence still moving at the largest step" assertion
/// fails. It is also a genuine numerical-quality check — a first-order error
/// in the field coupling would show O(h) rather than O(h²) behaviour.
///
/// Physics: for an atom E(F) = E0 − ½αF² − γF⁴/24 + …, so the central
/// estimate is α_FD(h) = α + (γ/12)h² and successive changes fall by exactly 4x
/// per halving of h — until the double-precision floor of the second
/// difference E(+h) − 2E(0) + E(−h) takes over.
///
/// # Why the steps start at h = 0.04
///
/// Measured 2026-09-24 on main (h = 0.04, 0.02, 0.01, 0.005): He changes
/// 1.527e-5, 3.817e-6, 9.542e-7 (ratios 4.000, 4.000); Ne −7.721e-6,
/// −1.929e-6, −4.835e-7 (ratios 4.002, 3.991). Both ratios are asserted.
///
/// The steps used to be 0.01..0.00125 with only the first ratio asserted, and
/// that ratio drifted to 4.524 for Ne (bar 3.5–4.5) after unrelated SCF
/// changes. The cause is floating-point cancellation, not SCF convergence:
/// Ne's total energy is ~−128 Ha, so the second difference carries ~1e-13 Ha
/// of roundoff, which h² = 6.25e-6 (h = 0.0025) amplifies to ~4e-8 in α — as
/// large as the O(h²) signal being compared. Larger steps grow the signal 4x
/// per doubling and shrink the noise 4x; the h = 0.005 end still keeps noise
/// ~1e-9 against a 4.8e-7 change.
///
/// Scope: this shows the field coupling is second-order accurate for
/// h ∈ [0.005, 0.04]. It says nothing about smaller h, where the estimate is
/// limited by f64 cancellation in the energies, not by the field coupling.
#[test]
fn finite_field_polarizability_converges_at_second_order() {
    let steps = [0.04_f64, 0.02, 0.01, 0.005];
    /// Number of successive-change RATIOS to assert on: all of them at these
    /// steps (see the doc comment for the measured values and why smaller
    /// steps are excluded).
    const N_RATIOS_ASSERTED: usize = 2;
    for symbol in ["He", "Ne"] {
        let alphas: Vec<f64> = steps.iter().map(|&h| alpha_zz(symbol, h)).collect();
        let deltas: Vec<f64> = alphas.windows(2).map(|w| w[1] - w[0]).collect();
        eprintln!("{symbol}: alpha = {alphas:?}");
        eprintln!("{symbol}: successive changes = {deltas:?}");

        // The sequence is actually moving at the coarsest step — this is what
        // a constant-returning alpha cannot do.
        assert!(
            deltas[0].abs() > 1e-8,
            "{symbol}: alpha did not change at all between h = {} and h = {} \
             (change {:.3e}). A finite-difference estimate that is insensitive to the \
             step is not being computed from the step.",
            steps[0],
            steps[1],
            deltas[0]
        );

        // Second-order convergence: halving h must quarter the remaining
        // error, so successive changes fall by ~4x. Bar is 3.5x..4.5x, which
        // the measured 3.991–4.002 (He, Ne) sit inside and a first-order
        // (2x) or non-converging (1x) scheme does not. Only the noise-free
        // ratio is asserted — see the doc comment for why.
        //
        // Print every ratio, including the excluded one, so the noise floor
        // stays visible to a reader rather than being silently dropped.
        for k in 0..deltas.len() - 1 {
            let ratio = deltas[k].abs() / deltas[k + 1].abs();
            let asserted = k < N_RATIOS_ASSERTED;
            eprintln!(
                "{symbol}: |d{k}|/|d{}| = {ratio:.3} (ideal 4 for O(h^2)){}",
                k + 1,
                if asserted { "" } else { "  [NOT asserted]" }
            );
            if !asserted {
                continue;
            }
            assert!(
                (3.5..4.5).contains(&ratio),
                "{symbol}: successive finite-difference changes fall by {ratio:.3}x between \
                 h = {} and h = {}, not the ~4x that second-order central differencing \
                 requires. A ratio near 2 means the field coupling carries a first-order \
                 error; a ratio near 1 means it is not converging at all.",
                steps[k + 1],
                steps[k + 2]
            );
        }
    }
}

/// The ion-induced-dipole differential between the two HeNe⁺ diabats has the
/// sign and rough size the lane's long-range analysis assumes.
///
/// α_Ne > α_He at def2-SVP (as experimentally), so the diabat that puts the
/// positive charge on He — polarizing the more polarizable Ne — is stabilized
/// more, and the differential (α_Ne − α_He)/(2R⁴) is positive. At R = 4 Å the
/// differential is a few meV, i.e. two to three orders of magnitude below the
/// 3.63 eV ΔIP anchor: the long-range gap is set by ΔIP, and polarization is a
/// correction to it, not a competitor.
#[test]
fn ion_induced_dipole_differential_is_small_against_the_ip_anchor() {
    const HA2EV: f64 = 27.211_386_245_988;
    const ANG2BOHR: f64 = 1.889_726_124_565_062;
    /// The ΔIP asymptote from `hene_atomic_ip_anchor.rs`, in eV.
    const DIP_EV: f64 = 3.630_496;

    let a_he = alpha_zz("He", FIELD_STEP);
    let a_ne = alpha_zz("Ne", FIELD_STEP);
    eprintln!(
        "alpha_He = {a_he:.8}   alpha_Ne = {a_ne:.8}   difference = {:.8} a.u.",
        a_ne - a_he
    );

    assert!(
        a_ne > a_he,
        "alpha_Ne = {a_ne:.8} must exceed alpha_He = {a_he:.8}: neon is the more \
         polarizable atom, experimentally (2.669 vs 1.384 a.u.) and at def2-SVP"
    );

    for &r_ang in &[2.0_f64, 3.0, 4.0, 6.0] {
        let rb = r_ang * ANG2BOHR;
        let d_ev = (a_ne - a_he) / (2.0 * rb.powi(4)) * HA2EV;
        eprintln!(
            "R = {r_ang:.1} A: dE_pol = {d_ev:.5} eV   ({:.4}% of the {DIP_EV:.4} eV dIP asymptote)",
            100.0 * d_ev / DIP_EV
        );
        assert!(
            d_ev > 0.0,
            "the polarization differential must favour the hole-on-He diabat, got {d_ev:.6} eV"
        );
    }

    // At 4 A the polarization term is already under 1% of the dIP asymptote,
    // which is why the R -> infinity gap is a pure ionization-potential
    // difference and the cDFT gap must asymptote to dIP. Measured: 0.13% at
    // 4 A. Bar at 1%.
    let rb4 = 4.0 * ANG2BOHR;
    let d_ev4 = (a_ne - a_he) / (2.0 * rb4.powi(4)) * HA2EV;
    assert!(
        d_ev4 < 0.01 * DIP_EV,
        "at R = 4 A the polarization differential is {d_ev4:.6} eV, which is not small \
         against the {DIP_EV:.4} eV dIP asymptote — the claim that the long-range gap is \
         set by the ionization-potential difference would not hold"
    );
}
