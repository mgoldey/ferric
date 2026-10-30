//! Spin-aware SCF result container.

use ferric_integrals::operator::Operator;
use ndarray::Array2;
use serde::{Deserialize, Serialize};

/// Spin treatment of the SCF wavefunction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Spin {
    /// Closed-shell: α and β share one set of MOs.
    Restricted,
    /// Open-shell: α and β have independent MOs.
    Unrestricted,
    /// Roothaan open-shell: α and β share spatial MOs with constrained occupations.
    RestrictedOpen,
}

/// Why the SCF loop stopped. Distinguishes acceptable exits (Converged,
/// Plateau) from failures the ladder should escalate past (Stalled, Diverged,
/// MaxIter, NotCertified).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScfExit {
    /// Standard convergence: energy + orbital gradient below thresholds.
    Converged,
    /// Near-degeneracy plateau accepted (gradient stalled below the 1e-4 floor).
    Plateau,
    /// Gradient running-minimum stopped falling above the 1e-4 floor.
    Stalled,
    /// Energy climbed beyond divergence_tol for consecutive iterations.
    Diverged,
    /// Hit max_iter without any of the above.
    MaxIter,
    /// ROHF/ROKS only: the SCF converged, but the F6 swap witness showed a
    /// one-electron move to a LOWER state, and the restart budget ran out
    /// before a state survived the check. The result is that last converged
    /// state (self-consistent energy, MOs and densities), reported
    /// `converged = false` because it is known not to be the lowest state
    /// reachable by moving one electron. See `ferric_scf::rohf_occupation`.
    NotCertified,
}

/// Converged self-consistent field solution: total energy, MO coefficients, orbital energies, and density matrices.
#[derive(Debug, Clone)]
#[must_use = "SCF result contains computed energies and orbitals"]
pub struct ScfResult {
    pub spin: Spin,
    pub energy: f64,
    /// AO total density (D_α + D_β). For Restricted this equals 2·D_α.
    pub density_total: Array2<f64>,
    /// α-spin density. Always populated.
    pub density_alpha: Array2<f64>,
    /// β-spin density. Populated for Unrestricted/RestrictedOpen; None for Restricted.
    pub density_beta: Option<Array2<f64>>,
    /// α MO coefficients (or restricted MOs).
    pub mos_alpha: Array2<f64>,
    /// β MO coefficients. `None` for Restricted.
    pub mos_beta: Option<Array2<f64>>,
    /// α orbital energies (eigenvalues of the Fock matrix in the MO basis).
    pub eps_alpha: Vec<f64>,
    /// β orbital energies. `None` for Restricted.
    pub eps_beta: Option<Vec<f64>>,
    /// α AO Fock matrix at convergence.
    pub fock_alpha: Array2<f64>,
    /// β AO Fock matrix. `None` for Restricted.
    pub fock_beta: Option<Array2<f64>>,
    /// Whether the SCF loop converged within the requested thresholds.
    pub converged: bool,
    /// Detailed exit reason (converged, plateau, stalled, diverged, max_iter).
    pub exit: ScfExit,
    /// Number of SCF iterations performed.
    pub iterations: usize,
    /// Total number of 2-electron integral quartets evaluated.
    pub computed_quartets: usize,
    /// Converged Thole-damped polarizable-embedding induced dipoles
    /// (`(n_sites, 3)`, a.u.), when `RhfConfig.polarizable` was `Some`.
    /// `None` whenever polarizable embedding was not configured — every
    /// existing constructor of `ScfResult` must set this to `None` to stay
    /// bit-identical (see `polarizable_none_is_bit_identical_to_plain_scf`).
    pub induced_dipoles: Option<Array2<f64>>,
    /// Post-convergence internal stability verdict, when
    /// `RhfConfig::check_stability` was set AND the reference was analysable.
    ///
    /// Follows the crate's solver-honesty convention (`converged`,
    /// `GwResult::outer_converged`, `LanczosResult::converged`): the verdict
    /// and the evidence for trusting it travel together on the result.
    ///
    /// **`None` means NOT CHECKED — it does NOT mean stable.** Three distinct
    /// situations all produce `None`: the flag was off (the default); the
    /// reference was skipped as un-analysable (ROHF/ROKS, range-separated,
    /// meta-GGA — each printed with its reason); or the eigensolve itself
    /// errored (also printed). A verdict of "stable" is only ever
    /// `Some(r)` with `r.converged && r.is_stable && !r.is_marginal()`, and
    /// even then it means "stable against the rotations
    /// `r.kind` covers" — see [`crate::stability`].
    ///
    /// Purely diagnostic: an instability never makes the SCF return `Err`.
    pub stability: Option<crate::stability::StabilityResult>,
    /// Post-convergence **EXTERNAL** stability verdict (RHF→UHF, the triplet
    /// channel), when `RhfConfig::check_stability` was set on an RHF/HF run.
    ///
    /// A separate field from [`stability`](Self::stability) because it is a
    /// separate OPERATOR answering a separate question, and the two verdicts
    /// routinely disagree — that disagreement is the diagnostic. Water /
    /// 6-31G at r(OH) = 2.0 Å is internally STABLE (+1.97e-2) and externally
    /// UNSTABLE (−3.07e-1): a caller reading only `stability` there concludes
    /// the solution is a minimum when it is a saddle.
    ///
    /// **`None` means NOT CHECKED, not stable.** It is `None` whenever the
    /// flag was off, the reference is not RHF (UHF-internal already spans the
    /// spin-broken space; ROHF is skipped entirely), the reference is KS (the
    /// triplet XC kernel does not exist — `StabilitySkip::TripletXcKernel`),
    /// or the eigensolve errored. Every skip prints its reason.
    ///
    /// Purely diagnostic, exactly like [`stability`](Self::stability).
    pub stability_external: Option<crate::stability::StabilityResult>,
    /// Which density-fitted (RI) Coulomb / exchange builders produced
    /// [`ScfResult::energy`], recorded by the solver that built them.
    ///
    /// `None` means every two-electron term of the energy used exact
    /// four-centre integrals (or the result was not produced by
    /// `solve_rhf`/`solve_uhf`/`solve_rohf`, e.g. a hand-built or transformed
    /// result). The analytic gradients (`rhf_gradient`, `uhf_gradient`,
    /// `rohf_gradient`, `ks_gradient_*`) read this field and differentiate the
    /// SAME approximate energy: without it they could only re-derive the
    /// solver's aux-basis resolution (auto-defaults, the `Some("")` opt-out,
    /// the consumption gates), and a re-derivation that drifts from the
    /// solver pairs an RI energy with an exact-integral gradient.
    pub df_jk: Option<DfJkRoute>,
    /// ROHF/ROKS only: the converged SPIN Fock matrices `(F_α, F_β)` (AO
    /// basis, including XC and solvent terms). `fock_alpha` holds the Roothaan
    /// EFFECTIVE Fock for ROHF, whose diagonal blocks mix `F_α` and `F_β` by a
    /// canonicalization choice and whose closed–open block is `F_β`; the
    /// gradient's energy-weighted density `W = D_α F_α D_α + D_α F_β D_β`
    /// needs the spin Focks themselves. `None` for RHF/UHF and hand-built
    /// results.
    pub rohf_spin_focks: Option<(Array2<f64>, Array2<f64>)>,
    /// COSX final-grid pass (`CosxConfig::final_grid`): the exchange energy
    /// re-evaluated ONCE on the larger final grid at the converged density
    /// (non-self-consistent, ORCA/Psi4 style). When `Some`,
    /// [`ScfResult::energy`] is `e_final` and the SCF-grid energy is kept in
    /// `e_scf_grid`; the analytic gradients differentiate the SCF-grid energy
    /// (see `crate::cosx_k::CosxFinalPass`). `None` when no final pass ran.
    pub cosx_final: Option<crate::cosx_k::CosxFinalPass>,
    /// COSX grid schedule (`CosxConfig::schedule`): the grid each iteration's
    /// K was built on and the switch iteration. `None` when no schedule ran.
    pub cosx_schedule: Option<crate::cosx_schedule::CosxScheduleRecord>,
    /// The convergence signals the gate judged at the FINAL SCF iteration and the thresholds
    /// they were judged against, so a failed (or barely passed) run is
    /// diagnosable without re-running under `FERRIC_SCF_TRACE`. Set by
    /// `solve_rhf` (converged and non-converged exits) and by `solve_uhf`'s
    /// converged and max-iteration results; `None` for ROHF/ROKS and for
    /// hand-built results. Purely a report: nothing reads it back, so
    /// energies and iteration counts are bit-identical with or without it.
    pub convergence: Option<ConvergenceReport>,
    /// The secondary-basis guess stage (`RhfConfig::guess_basis`): basis,
    /// projection, and the secondary SCF's iterations and energy. `None` when
    /// no secondary-basis guess ran.
    pub guess_basis: Option<crate::basis2::GuessBasisReport>,
    /// Precision provenance: which mixed-precision kernels this SCF ran and
    /// how (`ferric_core::gpu::provenance`). Set by `solve_rhf`/`solve_uhf`/
    /// `solve_rohf` and their KS-DFT forms from the per-kernel counters over
    /// the solve; `precision = f64` with no kernels for an all-f64 run (every
    /// default run) and for hand-built results. Purely a report: nothing reads
    /// it back, so energies are bit-identical with or without it.
    pub precision: ferric_core::gpu::provenance::PrecisionRecord,
}

/// The convergence signals the gate judged at the final iteration, plus the thresholds in force
/// (see [`ScfResult::convergence`]). Magnitudes, as gated by
/// `rhf::scf_converged`: `dp_rms < density_conv`, `dp_max < 10*density_conv`,
/// `de < energy_conv`. `err_max` (the DIIS commutator) is a diagnostic, never
/// a gate, EXCEPT under ADIIS/EDIIS: there a density-settled exit is refused
/// while `err_max` is above the energy-to-Pulay crossover and the run restarts
/// on plain Pulay DIIS (see [`DiisFallback`] and `rhf::veto_stalled_energy_diis`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConvergenceReport {
    /// Iteration whose gate decision these signals are (the values the gate judged).
    pub iteration: usize,
    /// |E - E_prev| the gate judged at that iteration.
    pub de: f64,
    /// RMS density change the gate judged (at MaxIter it lags the returned density by one step).
    pub dp_rms: f64,
    /// Max-element density change.
    pub dp_max: f64,
    /// Max |FDS - SDF| (a diagnostic, except that under ADIIS/EDIIS it vetoes a
    /// density-settled exit above the crossover; see [`Self::diis_fallback`]).
    pub err_max: f64,
    /// `RhfConfig::energy_conv` in force.
    pub energy_conv: f64,
    /// `RhfConfig::density_conv` in force.
    pub density_conv: f64,
    /// Estimated noise floor of `de` for this system
    /// ([`crate::noise_floor::de_noise_floor`]), or `None` when it is not
    /// estimable: functionals, the LinK / COSX exchange builders, PCM / COSMO,
    /// polarizable embedding, non-default screening, or a non-finite energy
    /// (none of those noise sources was measured).
    /// Report only: nothing reads it back.
    pub de_noise_floor: Option<f64>,
    /// Set once the energy-based DIIS (ADIIS/EDIIS) was abandoned for plain
    /// Pulay DIIS because it stalled (#422); `None` otherwise (always `None`
    /// for Pulay, UHF and ROHF).
    pub diis_fallback: Option<DiisFallback>,
    /// Density-settled exits refused because the state was not stationary
    /// (RHF and UHF orbital-gradient veto, `rhf::GradientVeto`); `None` when
    /// it never fired (always `None` for ROHF, whose F6 guard is separate).
    pub gradient_vetoes: Option<GradientVetoes>,
}

/// Running record of the RHF/UHF orbital-gradient veto: how many ΔP/ΔE-settled
/// exits were refused because the orbital gradient in the kept orthonormal
/// basis exceeded the bound, and the most recent one.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GradientVetoes {
    /// Number of refused exits so far.
    pub count: usize,
    /// Iteration of the most recent refusal.
    pub last_iteration: usize,
    /// Orbital gradient (max |Xᵀ(FDS−SDF)X|) at the most recent refusal.
    pub last_orbital_gradient: f64,
    /// The bound it exceeded (`max(sqrt(density_conv), 1e-3)`).
    pub bound: f64,
}

/// A record that ADIIS/EDIIS stalled and the run restarted on plain Pulay DIIS:
/// the density stopped moving while the commutator was still above the
/// energy-to-Pulay crossover, so the density-based gate was refused.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DiisFallback {
    /// Iteration whose density-settled exit was refused.
    pub iteration: usize,
    /// Commutator max-norm at that iteration.
    pub err_max: f64,
    /// The bound it exceeded (`diis_switch_thresh`, or the fixed bound used
    /// for an infinite switch).
    pub bound: f64,
    /// `"ADIIS"` or `"EDIIS"`.
    pub flavor: &'static str,
}

impl ConvergenceReport {
    /// Names of the gate criteria this report does NOT satisfy, in gate order
    /// (`dp_rms`, `dp_max`, `dE`, then `err_max` when the ADIIS/EDIIS commutator
    /// veto refused THIS iteration's exit); empty when all are met. NaN signals
    /// count as unmet.
    pub fn unmet(&self) -> Vec<&'static str> {
        let mut u = Vec::new();
        if !(self.dp_rms < self.density_conv) {
            u.push("dp_rms");
        }
        if !(self.dp_max < 10.0 * self.density_conv) {
            u.push("dp_max");
        }
        if !(self.de < self.energy_conv) {
            u.push("dE");
        }
        if self
            .diis_fallback
            .is_some_and(|f| f.iteration == self.iteration)
        {
            u.push("err_max");
        }
        if self
            .gradient_vetoes
            .is_some_and(|v| v.last_iteration == self.iteration)
        {
            u.push("orbital_gradient");
        }
        u
    }

    /// One-line human description: iteration, each signal against its
    /// threshold, and the unmet criteria.
    pub fn describe(&self) -> String {
        let unmet = self.unmet();
        let verdict = if unmet.is_empty() {
            "all criteria met".to_string()
        } else {
            format!("unmet: {}", unmet.join(", "))
        };
        let fallback_note = match self.diis_fallback {
            Some(f) => format!(
                "; {} stalled (density settled at iteration {} with err_max={:.3e} >= {:.1e}), \
                 restarted on Pulay DIIS",
                f.flavor, f.iteration, f.err_max, f.bound
            ),
            None => String::new(),
        };
        let veto_note = match self.gradient_vetoes {
            Some(v) => format!(
                "; not a stationary point: density-settled exit refused {} time(s), last at \
                 iteration {} with orbital gradient {:.3e} > {:.1e}",
                v.count, v.last_iteration, v.last_orbital_gradient, v.bound
            ),
            None => String::new(),
        };
        let floor_note = match self.de_noise_floor {
            // Only when dE is the SOLE unmet criterion: if a density criterion is
            // also unmet, the energy gate is not what stopped the run.
            Some(f)
                if unmet == ["dE"]
                    && crate::noise_floor::energy_conv_below_floor(self.energy_conv, f) =>
            {
                format!(
                    "; {}",
                    crate::noise_floor::floor_sentence(self.energy_conv, f)
                )
            }
            _ => String::new(),
        };
        format!(
            "iteration {}: dp_rms={:.3e} (need < {:.1e}), dp_max={:.3e} (need < {:.1e}), \
             dE={:.3e} (need < {:.1e}), DIIS err_max={:.3e}; {}{}{}{}",
            self.iteration,
            self.dp_rms,
            self.density_conv,
            self.dp_max,
            10.0 * self.density_conv,
            self.de,
            self.energy_conv,
            self.err_max,
            verdict,
            floor_note,
            fallback_note,
            veto_note
        )
    }
}

/// The density-fitted two-electron builders one SCF actually used.
///
/// Built only by the solvers, from the SAME effective aux names they passed to
/// `fock_assembly::build_df_jk` / `build_rsh_dfk_pair`, so the gradient cannot
/// resolve the route differently from the energy. See [`ScfResult::df_jk`].
#[derive(Debug, Clone, PartialEq)]
pub struct DfJkRoute {
    /// Auxiliary basis of the RI-J fit (`DfJ`, Cholesky-solved Coulomb-metric
    /// fit), or `None` for exact four-centre Coulomb.
    pub j_aux: Option<String>,
    /// Auxiliary basis of the ω = 0 RI-K fit (`DfK`, HF / global-hybrid
    /// exchange), or `None` when that exchange was exact or not consumed.
    pub k_aux: Option<String>,
    /// Range-separated exchange: the aux basis and ω of the SR (erfc) / LR
    /// (erf) `DfK` fitter pair. `None` for ω = 0 functionals.
    pub rsh_k: Option<(String, f64)>,
    /// Operator of the RI-J and ω = 0 RI-K fits (the SCF's Coulomb operator).
    pub op: Operator,
    /// Resolved three-index memory budget of the SCF, reused to bound the
    /// gradient's own three-index source.
    pub budget_bytes: usize,
}

impl DfJkRoute {
    /// Record the effective builders. Empty names are the "do not fit"
    /// sentinel and are dropped; returns `None` when nothing was fitted, so an
    /// all-exact SCF carries no route at all and its gradient takes the
    /// unchanged exact path.
    pub fn from_scf(
        j_aux: Option<&str>,
        k_aux: Option<&str>,
        rsh_k: Option<(&str, f64)>,
        op: Operator,
        budget_bytes: usize,
    ) -> Option<Self> {
        let clean = |s: Option<&str>| s.filter(|s| !s.is_empty()).map(str::to_string);
        let route = DfJkRoute {
            j_aux: clean(j_aux),
            k_aux: clean(k_aux),
            rsh_k: rsh_k
                .filter(|(s, _)| !s.is_empty())
                .map(|(s, w)| (s.to_string(), w)),
            op,
            budget_bytes,
        };
        route.is_active().then_some(route)
    }

    /// Whether any two-electron term was density-fitted.
    pub fn is_active(&self) -> bool {
        self.j_aux.is_some() || self.k_aux.is_some() || self.rsh_k.is_some()
    }
}

impl ScfResult {
    /// Restricted accessor: panics if spin != Restricted.
    pub fn mos_r(&self) -> &Array2<f64> {
        assert!(
            matches!(self.spin, Spin::Restricted),
            "mos_r() called on non-restricted result"
        );
        &self.mos_alpha
    }
    /// Restricted orbital energies. Panics if spin != Restricted.
    pub fn eps_r(&self) -> &[f64] {
        assert!(
            matches!(self.spin, Spin::Restricted),
            "eps_r() called on non-restricted result"
        );
        &self.eps_alpha
    }
    /// Restricted Fock matrix. Panics if spin != Restricted.
    pub fn fock_r(&self) -> &Array2<f64> {
        assert!(
            matches!(self.spin, Spin::Restricted),
            "fock_r() called on non-restricted result"
        );
        &self.fock_alpha
    }
    /// Restricted density matrix (2·D_α). Panics if spin != Restricted.
    pub fn density_r(&self) -> &Array2<f64> {
        assert!(
            matches!(self.spin, Spin::Restricted),
            "density_r() called on non-restricted result"
        );
        &self.density_total
    }
    /// Spin-summed AO density D_α + D_β. Available for all spin types
    /// (equals 2·D_α for Restricted; D_α + D_β for U/RO). Use for properties
    /// like ESP, electric field, Löwdin/Hirshfeld charges that take a
    /// total-electron density.
    pub fn density_total(&self) -> &Array2<f64> {
        &self.density_total
    }
    /// Unrestricted/ROHF accessors. Panic if called on a Restricted result.
    /// α MO coefficients. Available for all spin types.
    pub fn mos_a(&self) -> &Array2<f64> {
        &self.mos_alpha
    }
    /// β MO coefficients. Panics if spin == Restricted.
    pub fn mos_b(&self) -> &Array2<f64> {
        self.mos_beta
            .as_ref()
            .expect("mos_b() called on Restricted result")
    }
    /// α orbital energies. Available for all spin types.
    pub fn eps_a(&self) -> &[f64] {
        &self.eps_alpha
    }
    /// β orbital energies. Panics if spin == Restricted.
    pub fn eps_b(&self) -> &[f64] {
        self.eps_beta
            .as_deref()
            .expect("eps_b() called on Restricted result")
    }
}

impl std::fmt::Display for ScfResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:?} energy: {:.10} Ha ({} iters, {:?})",
            self.spin, self.energy, self.iterations, self.exit
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scf_exit_variants_distinct() {
        assert_ne!(ScfExit::Converged, ScfExit::Stalled);
        assert_ne!(ScfExit::Plateau, ScfExit::MaxIter);
        assert_ne!(ScfExit::Diverged, ScfExit::Converged);
    }
}

#[cfg(test)]
mod convergence_report_tests {
    use super::{ConvergenceReport, DiisFallback};

    fn rep(de: f64, dp_rms: f64, dp_max: f64) -> ConvergenceReport {
        ConvergenceReport {
            iteration: 7,
            de,
            dp_rms,
            dp_max,
            err_max: 1e-5,
            energy_conv: 1e-10,
            density_conv: 1e-8,
            de_noise_floor: None,
            diis_fallback: None,
            gradient_vetoes: None,
        }
    }

    /// The criteria mirror `rhf::scf_converged` exactly: dp_rms < dc,
    /// dp_max < 10 dc, dE < ec, each strict, reported in gate order.
    #[test]
    fn unmet_mirrors_the_gate_strictly_and_in_order() {
        assert!(rep(1e-11, 1e-9, 1e-8).unmet().is_empty());
        assert_eq!(rep(1e-11, 1e-8, 1e-8).unmet(), vec!["dp_rms"]); // equal is unmet
        assert_eq!(rep(1e-11, 1e-9, 1e-7).unmet(), vec!["dp_max"]);
        assert_eq!(rep(1e-10, 1e-9, 1e-8).unmet(), vec!["dE"]);
        assert_eq!(
            rep(1e-3, 1e-3, 1e-1).unmet(),
            vec!["dp_rms", "dp_max", "dE"]
        );
        // Agreement with the real gate on the same signals.
        let r = rep(1e-11, 4.7e-8, 3.8e-6);
        let sig = crate::rhf::ConvergenceSignals {
            de: r.de,
            dp_rms: r.dp_rms,
            dp_max: r.dp_max,
        };
        assert!(crate::rhf::scf_converged(sig, r.energy_conv, r.density_conv).is_none());
        assert_eq!(r.unmet(), vec!["dp_rms", "dp_max"]);
    }

    #[test]
    fn nan_signals_count_as_unmet() {
        assert_eq!(
            rep(f64::NAN, f64::NAN, f64::NAN).unmet(),
            vec!["dp_rms", "dp_max", "dE"]
        );
    }

    /// The exit report blames the noise floor only when the dE criterion is
    /// the unmet one AND energy_conv is below the estimated floor.
    #[test]
    fn describe_explains_an_unreachable_energy_gate_only_when_it_is_the_sole_unmet_criterion() {
        let with_floor = |de: f64, dp: f64, ec: f64, floor: Option<f64>| {
            ConvergenceReport {
                energy_conv: ec,
                de_noise_floor: floor,
                diis_fallback: None,
                gradient_vetoes: None,
                ..rep(de, dp, dp * 10.0)
            }
            .describe()
        };
        // dE unmet, energy_conv (1e-10) below the floor (3e-9): says so.
        let t = with_floor(2e-9, 1e-9, 1e-10, Some(3e-9));
        assert!(t.contains("unmet: dE"), "{t}");
        assert!(
            t.contains("energy_conv 1.0e-10 is below the estimated noise floor"),
            "{t}"
        );
        assert!(t.contains("3.0e-9"), "{t}");
        // dE unmet but energy_conv is comfortably above the floor: no blame.
        assert!(!with_floor(2e-3, 1e-9, 1e-6, Some(3e-9)).contains("noise floor"));
        // dE AND a density criterion unmet: the energy gate is not the cause.
        assert!(!with_floor(2e-9, 1e-6, 1e-10, Some(3e-9)).contains("noise floor"));
        // A non-finite floor makes no claim.
        assert!(!with_floor(2e-9, 1e-9, 1e-10, Some(f64::NAN)).contains("noise floor"));
        // Density is the unmet criterion (dE satisfied): no blame.
        assert!(!with_floor(1e-12, 1e-6, 1e-10, Some(3e-9)).contains("noise floor"));
        // Floor not estimable (functional / COSX runs): no claim.
        assert!(!with_floor(2e-9, 1e-9, 1e-10, None).contains("noise floor"));
    }

    #[test]
    fn describe_names_iteration_thresholds_and_unmet() {
        let t = rep(7.3e-11, 4.7e-8, 3.8e-6).describe();
        assert!(t.contains("iteration 7"), "{t}");
        assert!(t.contains("dp_rms=4.700e-8 (need < 1.0e-8)"), "{t}");
        assert!(t.contains("dE=7.300e-11 (need < 1.0e-10)"), "{t}");
        assert!(t.contains("unmet: dp_rms, dp_max"), "{t}");
        assert!(rep(1e-11, 1e-9, 1e-8)
            .describe()
            .contains("all criteria met"));
    }

    #[test]
    fn unmet_and_describe_name_the_commutator_when_the_veto_refused_this_exit() {
        let mut r = rep(1e-12, 1e-10, 1e-10);
        assert!(r.unmet().is_empty());
        r.diis_fallback = Some(DiisFallback {
            iteration: r.iteration,
            err_max: 0.5,
            bound: 0.1,
            flavor: "EDIIS",
        });
        assert_eq!(r.unmet(), vec!["err_max"]);
        let d = r.describe();
        assert!(
            d.contains("unmet: err_max") && d.contains("EDIIS stalled"),
            "{d}"
        );
        // A fallback at an EARLIER iteration is a note, not an unmet criterion.
        r.diis_fallback = Some(DiisFallback {
            iteration: r.iteration - 1,
            ..r.diis_fallback.unwrap()
        });
        assert!(r.unmet().is_empty());
        assert!(r.describe().contains("restarted on Pulay DIIS"));
    }
}
