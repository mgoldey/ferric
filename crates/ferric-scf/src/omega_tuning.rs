//! Optimal (IP-based, "gap") tuning of the range-separation parameter ω for
//! range-separated hybrid functionals — the standard Baer/Kronik ΔSCF
//! condition: choose ω so that Koopmans' theorem holds for the HOMO,
//!
//! ```text
//! J(ω) = ε_HOMO(N; ω) + IP_ΔSCF(ω),   IP = E(N−1; ω) − E(N; ω),
//! ```
//!
//! minimized as |J| by golden-section search. Each evaluation runs one
//! closed-shell RKS (neutral) and one UKS (cation, doublet) with the SAME
//! geometry, basis, grids and functional, differing only in ω via
//! `RhfConfig::xc_omega` (libxc `_omega` override; hard error for
//! functionals without one, so a tuning run can never silently fall back to
//! a fixed-ω functional).
//!
//! Scope: closed-shell neutral references (even-electron RKS) with a
//! doublet cation, the textbook case. Anions/EA-tuning and open-shell
//! references are not implemented — extend, don't approximate.
//!
//! # Cation state control
//!
//! J(ω) is only a curve if every point is the SAME electronic state. The
//! doublet cation of a symmetric molecule can have two: the symmetric one
//! (the hole delocalized over symmetry-equivalent centres) and a
//! symmetry-broken one (the hole localized on one of them). Which one an
//! independent SCF reaches depends on its guess, and [`tune_omega`](crate::omega_tuning::tune_omega)'s
//! golden-section search visits ω out of order, so independent solves can
//! return a J curve assembled from two branches.
//!
//! Two mechanisms address this, both on [`OmegaTuneConfig`](crate::omega_tuning::OmegaTuneConfig):
//!
//! * **Continuation** ([`OmegaTuneConfig::continuation`](crate::omega_tuning::OmegaTuneConfig::continuation), default OFF —
//!   see [`DEFAULT_CONTINUATION`](crate::omega_tuning::DEFAULT_CONTINUATION) for the measurement that decided that): each
//!   state is seeded from the converged orbitals of the already-evaluated ω
//!   NEAREST in ω — not the previous evaluation, which under golden section
//!   is a different thing. The first evaluation has no neighbour and falls
//!   back to the solver's own guess. [`OmegaEval::seed`](crate::omega_tuning::OmegaEval::seed) records which was used.
//! * **Branch consistency** ([`OmegaTuneConfig::branch_tol`](crate::omega_tuning::OmegaTuneConfig::branch_tol), on by default): after
//!   each cation solve, its spin-population asymmetry over symmetry-equivalent
//!   atoms ([`spin_population_asymmetry`](crate::omega_tuning::spin_population_asymmetry)) is compared with the nearest
//!   already-evaluated ω. A jump beyond the tolerance sets
//!   [`OmegaEval::branch_changed`](crate::omega_tuning::OmegaEval::branch_changed), and [`OmegaTuneResult::branch_warning`](crate::omega_tuning::OmegaTuneResult::branch_warning)
//!   carries the human-readable account. ⟨S²⟩ travels on every
//!   [`OmegaEval`](crate::omega_tuning::OmegaEval) but is NOT gated on: its smooth ω-drift is measured to be
//!   three orders of magnitude larger than the asymmetry signal that separates
//!   the branches.
//!
//! # LIMITATION: no stability verdict
//!
//! Neither mechanism can tell a converged cation that is an internal SADDLE
//! from one that is a minimum. The instrument for that is the orbital
//! Hessian. Its exchange response reproduces a range-separated Fock's SR/LR
//! split, so [`crate::stability::ks_reference_is_analysable`] accepts ω ≠ 0
//! for functionals without VV10. ωB97X-V, the functional this module is
//! validated with, carries VV10 and is still refused
//! ([`crate::stability::StabilitySkip::Vv10Kernel`]): no VV10 response kernel
//! exists. This module also does not yet report the Hessian's lowest
//! eigenvalue per ω. A tuned ω* can therefore sit on a cation branch that has
//! stopped being a minimum, and this module cannot say so. Continuation keeps
//! the curve on ONE branch; it does not certify that branch is the lowest one.

use crate::rhf::{solve_rhf, RhfConfig};
use crate::screening::SchwarzBounds;
use crate::uhf::solve_uhf_with_guess;
use ferric_core::error::FerricError;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ndarray::Array2;

/// Configuration for IP-based optimal tuning of the range-separation parameter ω.
#[derive(Debug, Clone)]
pub struct OmegaTuneConfig {
    /// RSH functional name (must carry a libxc `_omega` parameter).
    pub functional: String,
    /// Search bracket, Bohr⁻¹.
    pub omega_lo: f64,
    pub omega_hi: f64,
    /// Convergence width on ω (Bohr⁻¹).
    pub omega_tol: f64,
    /// Hard cap on J evaluations (each = 2 SCF solves).
    pub max_evals: usize,
    /// Base SCF settings applied to BOTH states (grids, convergence, ...).
    /// `xc`/`xc_omega` in it are overwritten per evaluation.
    pub scf: RhfConfig,
    /// Seed each state from the converged orbitals of the NEAREST
    /// already-evaluated ω. `false` (the default, see
    /// [`DEFAULT_CONTINUATION`]) solves every ω from the solver's own guess
    /// independently, which makes the per-ω result bit-identical to a bare
    /// [`eval_j`] call.
    ///
    /// Turn it ON for a system whose cation has more than one accessible
    /// state: continuation is what keeps a golden-section search — which
    /// visits ω out of order — on one branch instead of re-deciding the
    /// cation's state at every point from a guess.
    pub continuation: bool,
    /// Tolerance for the cation branch-consistency check: the largest change
    /// in the symmetry-equivalent spin-population asymmetry
    /// ([`spin_population_asymmetry`]) that is still attributed to ω moving
    /// rather than to the cation changing state. An absolute bar, compared
    /// against the nearest already-evaluated ω. Default
    /// [`DEFAULT_BRANCH_TOL`], which is derived from a measurement — read its
    /// doc before changing it, and read why ⟨S²⟩ is reported but not gated on.
    /// `None` disables the check, which leaves every
    /// [`OmegaEval::branch_changed`] `false` because nothing was CHECKED.
    pub branch_tol: Option<f64>,
}

impl Default for OmegaTuneConfig {
    fn default() -> Self {
        Self {
            functional: String::new(),
            omega_lo: 0.1,
            omega_hi: 1.0,
            omega_tol: 5e-3,
            max_evals: 24,
            scf: RhfConfig::default(),
            continuation: DEFAULT_CONTINUATION,
            branch_tol: Some(DEFAULT_BRANCH_TOL),
        }
    }
}

/// Default [`OmegaTuneConfig::continuation`]: OFF.
///
/// MEASURED (`tests/validation_omega_tuning_cation.rs`,
/// `omega_star_before_and_after_continuation` and
/// `h2o_continuation_dj_vs_scf_convergence_measurement`): turning continuation
/// on leaves ω* bit-identical on H2/6-31G, H2O/def2-SVP and NH3/def2-SVP at
/// 1e-10 / 1e-7, but moves J by up to 1.7e-7 Ha on H2O there. That difference
/// is SCF stopping noise in ε_HOMO, not a different solution: the cation and
/// neutral energies agree to ≤6e-12 Ha, and the J difference falls about a
/// decade per decade of `density_conv` (3.3e-6, 1.7e-7, 1.4e-8, 2.6e-10 Ha at
/// 1e-6 … 1e-9). Continuation also cuts SCF iterations by ~40 %.
///
/// It stays off because, at the thresholds callers already use, turning it on
/// would change their J values at the 1e-7 Ha level; that is a numbers change
/// for every existing caller, made only by a caller who opts in.
pub const DEFAULT_CONTINUATION: bool = false;

/// Default [`OmegaTuneConfig::branch_tol`], an absolute bar on the change in
/// the cation's spin-population asymmetry between adjacent ω.
///
/// NOT a guess, and NOT chosen before the data. MEASURED on N2/def2-SVP
/// ωB97X-V, the system whose cation is known to stop being a minimum between
/// ω = 0.53 and 0.56 (`tests/omega_tuning_cation_branch.rs`):
///
/// ```text
/// quantity                     symmetric branch, ω 0.50→0.60   broken branch (ω 0.60)
/// spin asymmetry               2.4e-10 … 3.5e-8                2.7e-7
/// ⟨S²⟩                         0.755622 → 0.756686             (see below)
/// ⟨S²⟩ drift per 0.01 Bohr⁻¹   1.06e-4                          —
/// E_cation − E_symmetric       0                                −4.6e-4 Ha
/// ```
///
/// Two consequences, both measured rather than assumed:
///
/// * ⟨S²⟩ CANNOT carry this check. Its smooth ω-drift (1.06e-4 per 0.01
///   Bohr⁻¹) is three orders of magnitude LARGER than the asymmetry signal
///   that distinguishes the branches, so any ⟨S²⟩ bar tight enough to see a
///   switch fires on every ω step. The check therefore reads ⟨S²⟩ for the
///   trace and gates only on the asymmetry.
/// * The asymmetry separates the branches by ≈1.5 decades (3.5e-8 vs 2.7e-7),
///   not the many decades a fully localized hole would give. 1e-7 sits
///   between the measured sides: above the symmetric branch's whole range
///   over a 0.10 Bohr⁻¹ sweep, below the broken branch's value.
///
/// That is a NARROW margin resting on one system, so a flag from this check
/// is a prompt to look, not a proof, and its absence is not a certificate —
/// see the module doc's limitation.
pub const DEFAULT_BRANCH_TOL: f64 = 1e-7;

/// Which guess an SCF of one [`OmegaEval`] started from.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OmegaSeed {
    /// The solver's own default guess (MINAO projection / hcore). Either
    /// continuation was off, or this was the first evaluation and there was no
    /// neighbour to continue from.
    Default,
    /// Converged orbitals of an earlier evaluation at this ω (Bohr⁻¹) — the
    /// one NEAREST in ω among those already evaluated, which under
    /// golden-section search is generally not the previous one.
    Continued { from_omega: f64 },
}

impl std::fmt::Display for OmegaSeed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OmegaSeed::Default => write!(f, "default guess"),
            OmegaSeed::Continued { from_omega } => write!(f, "continued from ω={from_omega:.6}"),
        }
    }
}

/// A single ω evaluation: HOMO eigenvalue, ΔSCF ionization potential, the
/// Koopmans residual J, and the cation-state diagnostics that say whether
/// this point belongs on the same J curve as its neighbours.
#[derive(Debug, Clone, Copy)]
pub struct OmegaEval {
    pub omega: f64,
    pub eps_homo: f64,
    pub ip_delta_scf: f64,
    pub j: f64,
    /// Cation total energy (Ha) — the ΔSCF IP's E(N−1) term, kept so a branch
    /// switch can be read off the energies and not only off the flag.
    pub e_cation: f64,
    /// Neutral total energy (Ha).
    pub e_neutral: f64,
    /// ⟨S²⟩ of the converged cation determinant.
    pub cation_s_squared: f64,
    /// Largest spin-population difference between symmetry-equivalent atoms
    /// of the cation ([`spin_population_asymmetry`]). ≈0 for a hole shared
    /// over equivalent centres; O(1) for a localized hole.
    pub cation_spin_asymmetry: f64,
    /// Which guess this evaluation's SCFs started from.
    pub seed: OmegaSeed,
    /// SCF iterations the cation took. The observable that distinguishes a
    /// continuation seed that was USED from one that was accepted and
    /// discarded: both converge to the same energy, so the energy cannot tell
    /// them apart, but a good seed converges in strictly fewer iterations.
    pub cation_iterations: usize,
    /// SCF iterations the neutral took, for the same reason.
    pub neutral_iterations: usize,
    /// `true` when the cation's spin-population asymmetry moved further than
    /// [`OmegaTuneConfig::branch_tol`] from the nearest already-evaluated ω,
    /// i.e. this point is probably not on the same branch as that neighbour.
    /// `false` with `branch_tol: None` means NOT CHECKED, not consistent.
    pub branch_changed: bool,
}

/// Result of an ω-tuning run: optimal ω, residual J, and the full evaluation trace.
#[derive(Debug, Clone)]
#[must_use]
pub struct OmegaTuneResult {
    pub omega: f64,
    /// J(ω*) — the residual Koopmans violation at the tuned ω.
    pub j: f64,
    pub evals: Vec<OmegaEval>,
    pub converged: bool,
    /// Human-readable account of every evaluation whose cation changed
    /// branch, `None` when the check found nothing (or was disabled — read
    /// [`OmegaTuneConfig::branch_tol`] to tell those apart). A J curve with
    /// this set is assembled from more than one electronic state and its ω*
    /// is not the ω* of either.
    pub branch_warning: Option<String>,
}

impl std::fmt::Display for OmegaTuneResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ω-tuning: ω* = {:.6} Bohr⁻¹ (J = {:.6}, {} evals, converged: {})",
            self.omega,
            self.j,
            self.evals.len(),
            self.converged
        )?;
        if let Some(w) = &self.branch_warning {
            write!(f, "; BRANCH WARNING: {w}")?;
        }
        Ok(())
    }
}

/// Converged state of one ω evaluation — what continuation carries forward.
///
/// Returned by [`eval_j_seeded`] and accepted by the next call, so a caller
/// driving its own ω sequence (a fixed grid, a bisection, a measurement sweep)
/// gets the same continuation [`tune_omega`] uses without re-deriving it.
#[derive(Debug, Clone)]
pub struct OmegaSeedState {
    /// The ω (Bohr⁻¹) this state was converged at.
    pub omega: f64,
    /// Neutral total AO density (D_α + D_β) → `RhfConfig::init_guess_density`.
    pub neutral_density: Array2<f64>,
    /// Cation α/β MO coefficients → `uhf::solve_uhf_with_guess`, which wants
    /// (nbasis, nbasis) and occupies the first nocc_σ columns of each.
    pub cation_mos: (Array2<f64>, Array2<f64>),
    /// Cation ⟨S²⟩. Reported in the branch diagnostic but NOT compared: its
    /// smooth ω-drift exceeds the signal (see [`DEFAULT_BRANCH_TOL`]).
    pub s_squared: f64,
    /// Cation spin-population asymmetry, for the branch comparison.
    pub spin_asymmetry: f64,
}

/// Largest spin-population difference over pairs of SYMMETRY-EQUIVALENT atoms.
///
/// The hole of a doublet cation is where the spin density is. On a molecule
/// with equivalent centres (the two N of N2, the three H of NH3) the
/// symmetric state puts equal spin population on each member of a group and a
/// localized state does not, so
///
/// ```text
/// max over groups G of   max_{A,B ∈ G} |p_A − p_B|,    p_A = Σ_{μ∈A} ((D_α − D_β)·S)_{μμ}
/// ```
///
/// is zero by symmetry for the symmetric state and O(1) for a localized hole.
/// This is the Mulliken partition of the SPIN density (`mulliken_charges`
/// partitions the total one); the partition's basis-set sensitivity does not
/// matter here because the quantity compared is a DIFFERENCE between atoms
/// the partition treats identically.
///
/// "Symmetry-equivalent" is read off the geometry, not a point group:
/// same element Z and the same multiset of interatomic distances to all other
/// atoms (to `GEOM_EQUIV_TOL` Bohr). That catches the cases this check exists
/// for — equivalent centres related by a symmetry operation have identical
/// distance multisets — and silently groups nothing when there are no
/// equivalent atoms, which makes the metric 0 and the check a no-op rather
/// than a false alarm.
///
/// Returns 0.0 when no two atoms are equivalent.
pub fn spin_population_asymmetry(
    mol: &Molecule,
    prep: &PreparedBasis,
    d_alpha: &Array2<f64>,
    d_beta: &Array2<f64>,
) -> Result<f64, FerricError> {
    let nbf = prep.nbasis();
    if d_alpha.dim() != (nbf, nbf) || d_beta.dim() != (nbf, nbf) {
        return Err(FerricError::General(format!(
            "spin_population_asymmetry: densities {:?}/{:?} != ({nbf},{nbf})",
            d_alpha.dim(),
            d_beta.dim()
        )));
    }
    let s = ferric_integrals::oneelectron::overlap(prep);
    let d_spin = d_alpha - d_beta;
    let m = d_spin.dot(&s);
    let shell_to_atom = prep.shell_to_atom();
    let shell_offsets = prep.shell_offsets();
    let natoms = mol.atoms.len();
    let mut pop = vec![0.0_f64; natoms];
    for (sh, &a) in shell_to_atom.iter().enumerate() {
        for mu in shell_offsets[sh]..shell_offsets[sh + 1] {
            pop[a] += m[(mu, mu)];
        }
    }
    let mut worst = 0.0_f64;
    for a in 0..natoms {
        for b in (a + 1)..natoms {
            if geometrically_equivalent(mol, a, b) {
                worst = worst.max((pop[a] - pop[b]).abs());
            }
        }
    }
    Ok(worst)
}

/// Distance tolerance (Bohr) for calling two atoms symmetry-equivalent.
const GEOM_EQUIV_TOL: f64 = 1e-6;

/// Same element and the same sorted multiset of distances to every other
/// atom, to [`GEOM_EQUIV_TOL`]. A sufficient geometric test for the
/// equivalence this module needs; it never claims to be a point-group
/// analysis.
fn geometrically_equivalent(mol: &Molecule, a: usize, b: usize) -> bool {
    if mol.atoms[a].z != mol.atoms[b].z {
        return false;
    }
    let dists = |i: usize| -> Vec<f64> {
        let mut v: Vec<f64> = (0..mol.atoms.len())
            .filter(|&j| j != i)
            .map(|j| {
                let (p, q) = (&mol.atoms[i], &mol.atoms[j]);
                ((p.x - q.x).powi(2) + (p.y - q.y).powi(2) + (p.zpos - q.zpos).powi(2)).sqrt()
            })
            .collect();
        v.sort_by(|x, y| x.partial_cmp(y).expect("finite coordinates"));
        v
    };
    let (da, db) = (dists(a), dists(b));
    da.len() == db.len()
        && da
            .iter()
            .zip(db.iter())
            .all(|(x, y)| (x - y).abs() < GEOM_EQUIV_TOL)
}

/// ⟨S²⟩ of a UHF/UKS determinant:
/// ⟨S²⟩ = S(S+1) + N_β − Σ_{i∈α-occ, j∈β-occ} |⟨α_i|β_j⟩|².
fn s_squared(
    c_a: &Array2<f64>,
    c_b: &Array2<f64>,
    s: &Array2<f64>,
    nocc_a: usize,
    nocc_b: usize,
) -> f64 {
    let s_true = 0.5 * (nocc_a as f64 - nocc_b as f64);
    let s_ideal = s_true * (s_true + 1.0);
    if nocc_a == 0 || nocc_b == 0 {
        return s_ideal;
    }
    let ov = c_a
        .slice(ndarray::s![.., ..nocc_a])
        .t()
        .dot(s)
        .dot(&c_b.slice(ndarray::s![.., ..nocc_b]));
    s_ideal + (nocc_b as f64) - ov.iter().map(|v| v * v).sum::<f64>()
}

/// One evaluation of the tuning objective at a fixed ω (Bohr⁻¹): converged
/// RKS neutral and UKS doublet cation with `xc = cfg.functional`,
/// `xc_omega = omega` and every other setting from `cfg.scf`. Returns
/// J = ε_HOMO(N) + E(N−1) − E(N) (signed), both states from their default
/// guesses — see [`eval_j_seeded`] for the continuation form [`tune_omega`]
/// uses. Public so J(ω) can be checked on its own.
pub fn eval_j(
    ctx: &ParallelContext,
    mol: &Molecule,
    prep: &PreparedBasis,
    bounds: &SchwarzBounds,
    cfg: &OmegaTuneConfig,
    omega: f64,
) -> Result<OmegaEval, FerricError> {
    eval_j_seeded(ctx, mol, prep, bounds, cfg, omega, None).map(|(e, _)| e)
}

/// [`eval_j`] with an optional continuation seed.
///
/// `seed` carries the converged state of an earlier ω: the neutral's total AO
/// density goes in as `RhfConfig::init_guess_density` and the cation's α/β MO
/// coefficients go to `uhf::solve_uhf_with_guess`. `None` reproduces
/// [`eval_j`] exactly, bit for bit. Returns the evaluation and the
/// [`OmegaSeedState`] a later ω can continue from, so a caller driving its own
/// ω sequence gets the same continuation [`tune_omega`] uses.
///
/// The CATION seed is MO coefficients and not a density on purpose: a density
/// guess for a UKS run is spin-summed by `uhf_guess_mos`, which would discard
/// exactly the α/β difference that distinguishes one cation branch from the
/// other.
///
/// `branch_changed` is left `false` here: it is a comparison against a
/// neighbour, which only [`tune_omega`] (holding the trace) can make. A caller
/// doing its own sweep compares the returned [`OmegaSeedState::s_squared`] and
/// [`OmegaSeedState::spin_asymmetry`] itself.
#[allow(clippy::too_many_arguments)]
pub fn eval_j_seeded(
    ctx: &ParallelContext,
    mol: &Molecule,
    prep: &PreparedBasis,
    bounds: &SchwarzBounds,
    cfg: &OmegaTuneConfig,
    omega: f64,
    seed: Option<&OmegaSeedState>,
) -> Result<(OmegaEval, OmegaSeedState), FerricError> {
    let mut scf_cfg = state_scf_config(cfg, omega)?;
    let op = Operator::coulomb();

    // Neutral: continuation goes in as an explicit guess DENSITY (rhf.rs's
    // documented entry point); with no seed the field keeps whatever the base
    // config had, which is what makes `seed: None` bit-identical to before.
    if let Some(sd) = seed {
        scf_cfg.init_guess_density = Some(sd.neutral_density.clone());
    }
    let neutral = solve_rhf(ctx, mol, prep, op, bounds, &scf_cfg)?;
    if !neutral.converged {
        return Err(FerricError::ScfConvergence {
            iterations: neutral.iterations,
            last_energy: neutral.energy,
        });
    }
    let nocc = (mol.nelec() as usize) / 2;
    let eps_homo = neutral.eps_r()[nocc - 1];

    let mut cation = mol.clone();
    cation.charge += 1;
    cation.multiplicity = 2;
    // Cation: continuation goes in as MOs, not a density — `solve_uhf_with_guess`
    // takes (C_α, C_β) of shape (nbasis, nbasis) and occupies the first
    // nocc_σ columns of each. A density guess would be spin-summed and could
    // not carry the α/β difference that IS the branch.
    let mut cat_cfg = scf_cfg.clone();
    cat_cfg.init_guess_density = None;
    let cat_seed = seed.map(|sd| (&sd.cation_mos.0, &sd.cation_mos.1));
    let cat = solve_uhf_with_guess(ctx, &cation, prep, bounds, &cat_cfg, cat_seed)?;
    if !cat.converged {
        return Err(FerricError::ScfConvergence {
            iterations: cat.iterations,
            last_energy: cat.energy,
        });
    }

    let c_a = cat.mos_alpha.clone();
    let c_b = cat
        .mos_beta
        .clone()
        .ok_or_else(|| FerricError::General("tune_omega: UKS cation has no β MOs".into()))?;
    let d_b = cat
        .density_beta
        .as_ref()
        .ok_or_else(|| FerricError::General("tune_omega: UKS cation has no β density".into()))?;
    let nelec_cat = cation.nelec() as usize;
    let nocc_a = nelec_cat.div_ceil(2);
    let nocc_b = nelec_cat / 2;
    let ovlp = ferric_integrals::oneelectron::overlap(prep);
    let s2 = s_squared(&c_a, &c_b, &ovlp, nocc_a, nocc_b);
    let asym = spin_population_asymmetry(&cation, prep, &cat.density_alpha, d_b)?;

    let ip = cat.energy - neutral.energy;
    let eval = OmegaEval {
        omega,
        eps_homo,
        ip_delta_scf: ip,
        j: eps_homo + ip,
        e_cation: cat.energy,
        e_neutral: neutral.energy,
        cation_s_squared: s2,
        cation_spin_asymmetry: asym,
        cation_iterations: cat.iterations,
        neutral_iterations: neutral.iterations,
        seed: match seed {
            None => OmegaSeed::Default,
            Some(sd) => OmegaSeed::Continued {
                from_omega: sd.omega,
            },
        },
        branch_changed: false,
    };
    let carry = OmegaSeedState {
        omega,
        neutral_density: neutral.density_total.clone(),
        cation_mos: (c_a, c_b),
        s_squared: s2,
        spin_asymmetry: asym,
    };
    Ok((eval, carry))
}

/// The SCF configuration both states of [`eval_j`] run with.
///
/// Both states must build J the same way, or the IP (and ω*) inherits the
/// difference: measured 2.3e-5 to 2.8e-5 Ha in the IP for ωB97X-V/def2-SVP
/// H2O, ≈1.4e-4 Bohr⁻¹ in ω*. The cation is range-separated UKS, and
/// `solve_uhf` never density-fits J when ω > 0: it uses exact J whatever
/// `df_j_aux` says. `solve_rhf`, by contrast, auto-selects RI-J for a
/// functional when `df_j_aux` is unset. So the only treatment both states can
/// share is exact J: an unset `df_j_aux` resolves to exact J (`Some("")`), and
/// a named aux basis is refused rather than applied to the neutral alone.
pub fn state_scf_config(cfg: &OmegaTuneConfig, omega: f64) -> Result<RhfConfig, FerricError> {
    let mut scf = RhfConfig {
        xc: Some(cfg.functional.clone()),
        xc_omega: Some(omega),
        ..cfg.scf.clone()
    };
    match scf.df_j_aux.as_deref() {
        None => scf.df_j_aux = Some(String::new()),
        Some("") => {}
        Some(aux) => {
            return Err(FerricError::General(format!(
                "tune_omega: df_j_aux = {aux:?} cannot be honoured: the range-separated \
                 UKS cation always uses exact J, so the neutral must too. Leave df_j_aux \
                 unset or set it to \"\" (exact J)."
            )))
        }
    }
    Ok(scf)
}

/// Index of the already-evaluated ω NEAREST to `omega`, or `None` when there
/// is none (the first evaluation).
///
/// Golden-section search does not visit ω monotonically, so "the previous
/// evaluation" and "the nearest ω" are different states and continuing from
/// the wrong one means continuing across a bigger ω step than necessary —
/// exactly the step most likely to jump a branch. Ties go to the earlier
/// entry, which makes the choice deterministic.
pub fn nearest_evaluated(done: &[f64], omega: f64) -> Option<usize> {
    // NOT `min_by`: `Iterator::min_by` returns the LAST of several equal
    // minima, so on an exact tie it would pick the later evaluation. Ties
    // happen whenever ω sits midway between two evaluated points, which
    // golden section produces, so the choice has to be pinned rather than
    // inherited. Fold keeping the first strict minimum instead.
    let mut best: Option<(usize, f64)> = None;
    for (i, w) in done.iter().enumerate() {
        let d = (*w - omega).abs();
        assert!(d.is_finite(), "nearest_evaluated: non-finite omega");
        match best {
            Some((_, bd)) if d >= bd => {}
            _ => best = Some((i, d)),
        }
    }
    best.map(|(i, _)| i)
}

/// The branch-consistency verdict for one evaluation against its nearest
/// already-evaluated neighbour: `Some(message)` when the cation's spin
/// asymmetry moved further than `tol`, `None` otherwise.
///
/// ⟨S²⟩ is REPORTED in the message but NOT gated on: its smooth ω-drift is
/// measured to be three orders of magnitude larger than the asymmetry signal
/// that separates the branches, so a ⟨S²⟩ bar tight enough to see a switch
/// fires on every ω step (see [`DEFAULT_BRANCH_TOL`]).
fn branch_warning_for(
    omega: f64,
    prev: &OmegaSeedState,
    now: &OmegaSeedState,
    tol: f64,
    e_cation: f64,
) -> Option<String> {
    let d_s2 = (now.s_squared - prev.s_squared).abs();
    let d_asym = (now.spin_asymmetry - prev.spin_asymmetry).abs();
    if d_asym <= tol {
        return None;
    }
    Some(format!(
        "ω={omega:.6}: cation state differs from the nearest evaluated ω={:.6} \
         (Δspin-asymmetry = {d_asym:.3e} > tol {tol:.3e}; Δ⟨S²⟩ = {d_s2:.3e}, \
         not gated on); ⟨S²⟩ {:.6}→{:.6}, asymmetry {:.3e}→{:.3e}, \
         E_cation {e_cation:.8}",
        prev.omega, prev.s_squared, now.s_squared, prev.spin_asymmetry, now.spin_asymmetry,
    ))
}

/// Golden-section minimization of |J(ω)| over the bracket.
///
/// With [`OmegaTuneConfig::continuation`] on (it is OFF by default), each
/// evaluation after the first is seeded from the nearest already-evaluated ω;
/// the first always uses the default guess. Each evaluation's cation is checked
/// against that same neighbour for a branch switch unless
/// [`OmegaTuneConfig::branch_tol`] is `None` (the check is on by default). See the module doc for what the
/// two mechanisms do and do NOT establish — in particular neither is a
/// stability verdict, which ω ≠ 0 cannot currently have.
pub fn tune_omega(
    ctx: &ParallelContext,
    mol: &Molecule,
    prep: &PreparedBasis,
    bounds: &SchwarzBounds,
    cfg: &OmegaTuneConfig,
) -> Result<OmegaTuneResult, FerricError> {
    if mol.nelec() % 2 != 0 {
        return Err(FerricError::General(
            "tune_omega: closed-shell (even-electron) neutral references only".into(),
        ));
    }
    if !(cfg.omega_lo > 0.0 && cfg.omega_hi > cfg.omega_lo) {
        return Err(FerricError::General(format!(
            "tune_omega: invalid bracket [{}, {}]",
            cfg.omega_lo, cfg.omega_hi
        )));
    }
    let mut evals: Vec<OmegaEval> = Vec::new();
    let mut seeds: Vec<OmegaSeedState> = Vec::new();
    let mut warnings: Vec<String> = Vec::new();

    let f = |w: f64,
             evals: &mut Vec<OmegaEval>,
             seeds: &mut Vec<OmegaSeedState>,
             warnings: &mut Vec<String>|
     -> Result<f64, FerricError> {
        let near = {
            let done: Vec<f64> = seeds.iter().map(|s| s.omega).collect();
            nearest_evaluated(&done, w)
        };
        let seed = if cfg.continuation {
            near.map(|i| &seeds[i])
        } else {
            None
        };
        let (mut e, carry) = eval_j_seeded(ctx, mol, prep, bounds, cfg, w, seed)?;
        // Branch check against the SAME neighbour continuation uses (or would
        // have used): the comparison and the seed must name one state, or a
        // flag could not be read as "this ω left that state".
        if let (Some(tol), Some(i)) = (cfg.branch_tol, near) {
            if let Some(msg) = branch_warning_for(w, &seeds[i], &carry, tol, e.e_cation) {
                e.branch_changed = true;
                warnings.push(msg);
            }
        }
        evals.push(e);
        seeds.push(carry);
        Ok(e.j.abs())
    };
    const INVPHI: f64 = 0.618_033_988_749_894_9;
    let (mut a, mut b) = (cfg.omega_lo, cfg.omega_hi);
    let mut c = b - (b - a) * INVPHI;
    let mut d = a + (b - a) * INVPHI;
    let mut fc = f(c, &mut evals, &mut seeds, &mut warnings)?;
    let mut fd = f(d, &mut evals, &mut seeds, &mut warnings)?;
    let mut converged = false;
    while evals.len() < cfg.max_evals {
        if (b - a) < cfg.omega_tol {
            converged = true;
            break;
        }
        if fc < fd {
            b = d;
            d = c;
            fd = fc;
            c = b - (b - a) * INVPHI;
            fc = f(c, &mut evals, &mut seeds, &mut warnings)?;
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + (b - a) * INVPHI;
            fd = f(d, &mut evals, &mut seeds, &mut warnings)?;
        }
    }
    let best = evals
        .iter()
        .cloned()
        .min_by(|x, y| x.j.abs().partial_cmp(&y.j.abs()).expect("NaN J"))
        .expect("at least two evaluations");
    let branch_warning = if warnings.is_empty() {
        None
    } else {
        Some(warnings.join(" | "))
    };
    Ok(OmegaTuneResult {
        omega: best.omega,
        j: best.j,
        evals,
        converged,
        branch_warning,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Nearest-ω selection, which is NOT previous-evaluation selection. An
    /// SCF test on a monotone ω sweep cannot tell those apart (both pick the
    /// same neighbour there), so the distinction is pinned here instead.
    #[test]
    fn nearest_evaluated_is_not_the_previous_evaluation() {
        assert_eq!(nearest_evaluated(&[], 0.4), None, "first eval has no seed");
        assert_eq!(nearest_evaluated(&[0.3], 0.9), Some(0), "sole candidate");
        // Golden-section order on [0.2, 0.8]: 0.429, 0.571, then 0.329.
        // The PREVIOUS eval is 0.571; the NEAREST is 0.429.
        let done = [0.4292, 0.5708];
        assert_eq!(nearest_evaluated(&done, 0.3292), Some(0));
        // ... and the next point 0.4708 is nearest to 0.4292 too, not to the
        // just-evaluated 0.3292.
        let done = [0.4292, 0.5708, 0.3292];
        assert_eq!(nearest_evaluated(&done, 0.4708), Some(0));
        // An EXACT tie resolves to the earlier entry, deterministically.
        // The ω must be chosen so the two distances are bit-equal: |0.3−0.4|
        // and |0.5−0.4| are NOT (0.10000000000000003 vs 0.09999999999999998),
        // so that triple tests ordinary selection, not tie-breaking. Binary
        // fractions do tie exactly.
        assert_eq!(
            (0.25_f64 - 0.5).abs(),
            (0.75_f64 - 0.5).abs(),
            "this case must be an exact tie or it is not testing tie-breaking"
        );
        assert_eq!(nearest_evaluated(&[0.25, 0.75], 0.5), Some(0));
        // ...and the near-tie above is decided by the actual f64 distance,
        // not by order: 0.5 really is marginally closer to 0.4 than 0.3 is.
        assert_eq!(nearest_evaluated(&[0.3, 0.5], 0.4), Some(1));
    }
}
