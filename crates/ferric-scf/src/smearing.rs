//! Finite-temperature (Fermi-Dirac) occupation smearing for the SCF.
//!
//! For systems with a near-degenerate frontier manifold (transition-metal
//! dimers such as Cu2/Ag2, metals) the integer 0/2 aufbau occupation makes the
//! SCF oscillate between which orbitals to occupy. Replacing the step function
//! with a Fermi-Dirac distribution
//!
//! ```text
//!     f_i = 1 / (1 + exp((ε_i − μ) / σ))
//! ```
//!
//! smooths the occupation boundary. Here `σ = k_B T` is the smearing width in
//! Hartree and `μ` (the chemical potential / Fermi level) is solved so that the
//! occupations reproduce the correct electron count. This is the standard
//! convergence aid used by ORCA, PySCF (`scf.addons.smearing_`) and Q-Chem.
//!
//! This module is a *pure* helper: it takes MO energies and returns
//! occupations. It performs no I/O and does not touch the SCF loop state, so it
//! is trivially unit-testable. The density build multiplies each MO's outer
//! product `c_i c_iᵀ` by its occupation instead of the flat integer factor.
//!
//! ## Restricted (closed-shell) convention
//!
//! Each *spatial* orbital holds up to 2 electrons, so its occupation is
//! `n_i = 2 · f_i ∈ [0, 2]` and `Σ_i n_i = N_elec`. The routines below return
//! the *fractional per-spin* occupation `f_i ∈ [0, 1]`; the caller scales by 2
//! for the density (`D = Σ_i 2 f_i c_i c_iᵀ`), exactly mirroring the integer
//! path where the lowest `nocc` orbitals have `f_i = 1` and the rest `f_i = 0`.

/// Fermi-Dirac occupation `f = 1/(1+exp((ε−μ)/σ))`, per spin (∈ [0, 1]).
///
/// Written to avoid overflow in `exp` for large `|ε−μ|/σ`: the argument is
/// clamped so that a deeply occupied orbital returns exactly 1.0 and a deeply
/// virtual one exactly 0.0, which also makes the σ→0 limit numerically exact.
#[inline]
pub fn fermi_dirac(eps: f64, mu: f64, sigma: f64) -> f64 {
    // σ ≤ 0 is the zero-temperature step function (used by the T→0 limit and as
    // a guard against a nonsensical negative width).
    if sigma <= 0.0 {
        return if eps < mu {
            1.0
        } else if eps > mu {
            0.0
        } else {
            0.5 // exactly at the Fermi level: half-filled (symmetric limit)
        };
    }
    let x = (eps - mu) / sigma;
    // exp(±40) is already 2e17 / 4e-18 — saturating here avoids inf/underflow
    // noise while staying far below the f=1 / f=0 rounding boundary.
    if x > 40.0 {
        0.0
    } else if x < -40.0 {
        1.0
    } else {
        1.0 / (1.0 + x.exp())
    }
}

/// Total per-spin electron count `Σ_i f_i` at chemical potential `μ`.
#[inline]
fn count_electrons(eps: &[f64], mu: f64, sigma: f64) -> f64 {
    eps.iter().map(|&e| fermi_dirac(e, mu, sigma)).sum()
}

/// Result of a Fermi-Dirac smearing solve.
#[derive(Debug, Clone)]
pub struct Smearing {
    /// Chemical potential μ (Fermi level) in Hartree.
    pub mu: f64,
    /// Per-orbital *per-spin* occupations `f_i ∈ [0, 1]`, one per MO energy.
    /// For the restricted density scale each by 2.
    pub occupations: Vec<f64>,
    /// Electronic entropy contribution `S` in units of `k_B` (dimensionless),
    /// summed over the requested spin multiplicity. For a restricted (2×)
    /// channel this already includes the spin factor, so the free-energy
    /// correction is `−σ · entropy` (σ = k_B T in Ha).
    pub entropy: f64,
}

/// Solve for the chemical potential μ so that `spin_degeneracy · Σ_i f_i(ε_i,μ,σ)`
/// equals `n_electrons`, and return the resulting per-spin occupations.
///
/// * `eps` — MO energies (Hartree), any order.
/// * `n_electrons` — the target electron count this channel must integrate to.
///   For restricted RHF this is `N_elec` and `spin_degeneracy = 2`.
/// * `sigma` — smearing width `k_B T` in Hartree (must be > 0).
/// * `spin_degeneracy` — electrons per fully-occupied orbital (2 for
///   restricted closed-shell, 1 for a single spin channel of UHF).
///
/// μ is found by **bisection**: the per-spin count `Σ f_i` is a smooth,
/// monotonically increasing function of μ (each `f_i` increases with μ), so the
/// electron-count residual has exactly one root, bracketed by
/// `[min(ε)−pad, max(ε)+pad]`, and bisection converges unconditionally. We use
/// bisection rather than Newton because it cannot overshoot a near-degenerate
/// manifold where the derivative spikes — robustness is the whole point of the
/// feature.
///
/// The entropy returned is
/// `S = −spin_degeneracy · Σ_i [f_i ln f_i + (1−f_i) ln(1−f_i)]`
/// (in units of `k_B`); the free-energy (Mermin) correction is `−σ · S`.
pub fn solve_fermi_level(
    eps: &[f64],
    n_electrons: f64,
    sigma: f64,
    spin_degeneracy: f64,
) -> Result<Smearing, ferric_core::FerricError> {
    if sigma <= 0.0 {
        return Err(ferric_core::FerricError::General(format!(
            "smearing width σ must be positive, got {sigma}"
        )));
    }
    if spin_degeneracy <= 0.0 {
        return Err(ferric_core::FerricError::General(format!(
            "spin_degeneracy must be positive, got {spin_degeneracy}"
        )));
    }
    if eps.is_empty() {
        return Err(ferric_core::FerricError::General(
            "cannot smear occupations over an empty MO set".to_string(),
        ));
    }
    // Target per-spin count. For RHF: N_elec / 2.
    let target = n_electrons / spin_degeneracy;
    let n_orb = eps.len() as f64;
    if target < -1e-12 || target > n_orb + 1e-12 {
        return Err(ferric_core::FerricError::General(format!(
            "per-spin electron target {target} outside [0, {n_orb}] — cannot be \
             reached by Fermi occupations"
        )));
    }

    let e_min = eps.iter().cloned().fold(f64::INFINITY, f64::min);
    let e_max = eps.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    // Pad the bracket so the endpoints straddle the root even when the target is
    // ~0 or ~n_orb (μ then lies well below/above the spectrum). 20σ makes the
    // Fermi factors saturate at the endpoints regardless of σ.
    let pad = 20.0 * sigma + 1.0;
    let mut lo = e_min - pad;
    let mut hi = e_max + pad;

    // Bisection: residual(μ) = Σ f_i(μ) − target is monotonically increasing.
    // 200 iterations halve a starting bracket of O(spectral width) to far below
    // machine epsilon — μ is converged to ~1e-14 Ha.
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        let residual = count_electrons(eps, mid, sigma) - target;
        if residual > 0.0 {
            hi = mid;
        } else {
            lo = mid;
        }
        if (hi - lo) < 1e-14 {
            break;
        }
    }
    let mu = 0.5 * (lo + hi);

    let occupations: Vec<f64> = eps.iter().map(|&e| fermi_dirac(e, mu, sigma)).collect();

    // Electronic entropy S = −Σ [f ln f + (1−f) ln(1−f)], per spin channel,
    // scaled by the spin degeneracy. The x·ln(x) terms vanish smoothly at the
    // integer-occupation limits (0 and 1), so guard the logs there.
    let entropy_per_spin: f64 = occupations
        .iter()
        .map(|&f| {
            let mut s = 0.0;
            if f > 0.0 && f < 1.0 {
                s -= f * f.ln();
                s -= (1.0 - f) * (1.0 - f).ln();
            }
            s
        })
        .sum();

    Ok(Smearing {
        mu,
        occupations,
        entropy: spin_degeneracy * entropy_per_spin,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A synthetic closed-shell spectrum with a clear HOMO–LUMO gap: 4 orbitals,
    /// 4 electrons (2 doubly-occupied). μ must land in the gap and the smeared
    /// count must reproduce N=4 to tolerance.
    #[test]
    fn mu_solves_electron_count() {
        let eps = [-1.0, -0.8, 0.5, 1.2]; // gap between −0.8 and 0.5
        let n_elec = 4.0;
        let sigma = 0.01; // ~3160 K — small vs the 1.3 Ha gap
        let sm = solve_fermi_level(&eps, n_elec, sigma, 2.0).unwrap();

        // Occupations sum (per spin) to N/2 = 2.
        let sum_spin: f64 = sm.occupations.iter().sum();
        assert!((sum_spin - 2.0).abs() < 1e-10, "per-spin sum {sum_spin}");
        // Total electrons = 2 · Σ f = 4.
        let n_total: f64 = 2.0 * sum_spin;
        assert!((n_total - n_elec).abs() < 1e-10, "N = {n_total}");

        // μ sits inside the gap (−0.8, 0.5).
        assert!(sm.mu > -0.8 && sm.mu < 0.5, "μ = {} not in gap", sm.mu);

        // With a small σ vs the gap, the two low orbitals are ~fully occupied
        // and the two high ones ~empty.
        assert!(sm.occupations[0] > 0.999);
        assert!(sm.occupations[1] > 0.999);
        assert!(sm.occupations[2] < 1e-3);
        assert!(sm.occupations[3] < 1e-3);
    }

    /// T→0 limit: as σ shrinks, the smeared occupation reduces to the integer
    /// aufbau occupation (lowest nocc orbitals = 1, rest = 0) and the entropy
    /// vanishes.
    #[test]
    fn zero_temperature_limit_is_integer() {
        let eps = [-2.0, -1.0, -0.3, 0.7, 1.5];
        let n_elec = 6.0; // 3 doubly-occupied spatial orbitals
        let sigma = 1e-6; // essentially T→0

        let sm = solve_fermi_level(&eps, n_elec, sigma, 2.0).unwrap();
        let expected = [1.0, 1.0, 1.0, 0.0, 0.0];
        for (i, (&f, &e)) in sm.occupations.iter().zip(expected.iter()).enumerate() {
            assert!((f - e).abs() < 1e-6, "occ[{i}] = {f}, expected {e}");
        }
        // Per-spin count still exact.
        let sum_spin: f64 = sm.occupations.iter().sum();
        assert!((sum_spin - 3.0).abs() < 1e-8);
        // Entropy → 0 at the integer limit.
        assert!(sm.entropy.abs() < 1e-4, "entropy {} should vanish", sm.entropy);
    }

    /// A near-degenerate frontier manifold (the case the feature targets): two
    /// orbitals straddle the Fermi level with a tiny gap. Smearing spreads the
    /// occupation fractionally between them while still integrating to N.
    #[test]
    fn near_degenerate_frontier_is_fractional() {
        // Two deep orbitals + two nearly-degenerate frontier orbitals holding
        // effectively 1 pair between them.
        let eps = [-1.0, -0.9, -0.001, 0.001];
        let n_elec = 6.0; // 3 pairs; the 3rd pair splits over the frontier
        let sigma = 0.02;
        let sm = solve_fermi_level(&eps, n_elec, sigma, 2.0).unwrap();

        let sum_spin: f64 = sm.occupations.iter().sum();
        assert!((sum_spin - 3.0).abs() < 1e-10, "per-spin sum {sum_spin}");
        // The two frontier orbitals are each fractionally occupied (~0.5 per
        // spin) rather than a hard 1/0 split.
        assert!(sm.occupations[2] > 0.3 && sm.occupations[2] < 0.7);
        assert!(sm.occupations[3] > 0.3 && sm.occupations[3] < 0.7);
        // Entropy is strictly positive when occupations are fractional.
        assert!(sm.entropy > 0.0, "entropy {} should be positive", sm.entropy);
    }

    /// Direct μ-solve residual check across a range of widths: the returned μ
    /// must always reproduce the target count.
    #[test]
    fn residual_is_zero_over_sigma_range() {
        let eps = [-1.5, -1.4, -0.2, 0.3, 0.35, 1.1];
        let n_elec = 6.0;
        for &sigma in &[0.001, 0.005, 0.02, 0.1, 0.5] {
            let sm = solve_fermi_level(&eps, n_elec, sigma, 2.0).unwrap();
            let n_total = 2.0 * sm.occupations.iter().sum::<f64>();
            assert!(
                (n_total - n_elec).abs() < 1e-9,
                "σ={sigma}: N={n_total} (μ={})",
                sm.mu
            );
        }
    }

    /// Single spin channel (spin_degeneracy = 1) — the UHF-style convention,
    /// exercised here only as a helper unit test (UHF wiring is out of scope).
    #[test]
    fn single_spin_channel() {
        let eps = [-1.0, -0.5, 0.2, 0.9];
        let n_up = 2.0; // 2 alpha electrons
        let sm = solve_fermi_level(&eps, n_up, 0.01, 1.0).unwrap();
        let sum: f64 = sm.occupations.iter().sum();
        assert!((sum - 2.0).abs() < 1e-10, "α count {sum}");
    }

    /// Negative σ is rejected (would silently become the step function
    /// otherwise via `fermi_dirac`, but the public solve must error).
    #[test]
    fn negative_sigma_errors() {
        let eps = [-1.0, 0.0, 1.0];
        assert!(solve_fermi_level(&eps, 2.0, -0.1, 2.0).is_err());
        assert!(solve_fermi_level(&eps, 2.0, 0.0, 2.0).is_err());
    }
}
