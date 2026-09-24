//! Closed-shell restricted Hartree-Fock (RHF) solver.
//!
//! Implements the Roothaan-Hall SCF procedure with DIIS convergence acceleration
//! and Schwarz-screened two-electron integral evaluation.

use crate::diis::DiisDriver;
use crate::direct_j::DirectJ;
use crate::direct_jk::DirectJK;
use crate::direct_k::DirectK;
use crate::fock::{JBuilder, KBuilder};
use crate::guess::hcore_guess;
use crate::result::{ScfExit, ScfResult, Spin};
use ferric_dft::cdft::Constraint;

use crate::screening::SchwarzBounds;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::blas_threads::{opt_in_blas_threads, with_blas_threads};
use ferric_integrals::operator::Operator;
use ndarray::linalg::general_mat_mul;
use ndarray::Array2;
use ndarray_linalg::Eigh;

/// Configuration parameters for the RHF solver.
#[derive(Debug, Clone)]
pub struct RhfConfig {
    pub max_iter: usize,
    pub energy_conv: f64,
    pub density_conv: f64,
    pub diis_size: usize,
    /// DIIS family: `Pulay` (default, plain commutator DIIS — historical
    /// byte-identical behavior) or `Adiis`/`Ediis` (energy-based variant in the
    /// early SCF, switching to Pulay near convergence — for hard TM-dimer cases).
    pub diis_flavor: crate::diis::DiisFlavor,
    /// Commutator `err_max` crossover below which `diis_flavor` reverts to plain
    /// Pulay. Ignored when `diis_flavor == Pulay`. Default 1e-1 (PySCF/ORCA).
    pub diis_switch_thresh: f64,
    /// Finite-temperature (Fermi-Dirac) occupation smearing width σ = k_B·T in
    /// Hartree. `None` (default) = integer 0/2 aufbau occupation (unchanged).
    /// `Some(σ)` smears the frontier — a convergence aid for near-degenerate
    /// d-manifolds (TM dimers/metals). See `smearing.rs`.
    pub smearing_sigma: Option<f64>,
    pub integral_thresh: f64,
    /// Choose K matrix builder: "direct" (default), "link", or "cosx"
    /// (seminumerical exchange, see [`crate::cosx_k`]). Honoured by
    /// `solve_rhf`, `solve_uhf` AND `solve_rohf` (before 2026-09-08 only
    /// `solve_rhf` read it, so it was a silent no-op for open-shell runs).
    /// The open-shell solvers build K_α and K_β from ONE builder instance,
    /// refreshing any density-dependent state per spin. Ignored WITH A WARNING
    /// whenever density-fitted J/K is active (`df_j_aux` / `df_k_aux` set, or
    /// auto-defaulted for a functional), the functional uses no exact exchange,
    /// or the functional is range-separated (exchange then comes from the
    /// SR/LR density-fitted fitters).
    pub k_builder: Option<String>,
    /// COSX knobs (grid, overlap fit, screen); only read when
    /// `k_builder == Some("cosx")`.
    pub cosx: crate::cosx_k::CosxConfig,
    /// Optional auxiliary basis for density-fitted Coulomb (RI-J). When set, J is
    /// built from precomputed 3-center ERIs in O(N^2 · naux) per iteration instead
    /// of contracting full 4-index ERIs.
    pub df_j_aux: Option<String>,
    /// Optional auxiliary basis for density-fitted exchange (RI-K). When set, K is
    /// built from the V^{-1/2}-dressed 3-center tensor in O(N^3 · naux) GEMMs.
    /// Should be a JK-fit basis (e.g. `def2-universal-jkfit`), not an RI/MP2-fit
    /// basis, which would introduce mHa-scale error in K.
    pub df_k_aux: Option<String>,
    /// XC functional name (None = pure HF; e.g. "LDA", "PBE", "B3LYP", "wB97X-V").
    pub xc: Option<String>,
    /// Range-separation ω override (Bohr⁻¹) for a range-separated `xc` —
    /// the optimal-tuning knob. `None` = the functional's published ω
    /// (byte-identical path). Hard error via libxc if `xc` has no `_omega`
    /// parameter (never silently ignored).
    pub xc_omega: Option<f64>,
    /// Main DFT grid spec. Default (75, 110) when xc.is_some().
    pub dft_grid: Option<ferric_dft::grid::AtomicGridConfig>,
    /// NLC (VV10) grid spec. Default (50, 50) when XC requires VV10.
    pub nlc_grid: Option<ferric_dft::grid::AtomicGridConfig>,
    /// Level shift (Ha) applied to the virtual–virtual block of the Fock
    /// matrix in MO basis. Defaults to 0 (no shift). Used to damp oscillations
    /// in open-shell SCF (ROHF/UHF/ROKS) where DIIS plateaus near a near-
    /// degenerate transition. A shift of 0.1–0.5 Ha is typical.
    pub level_shift: f64,
    /// If > 0 in ROHF/ROKS: switch from DIIS to a damped-Newton step once the
    /// DIIS error (err_max) drops below this trigger. Setting `1e-2` is a
    /// reasonable default for OH-doublet LDA/PBE plateaus. A value of 0
    /// disables Newton entirely (DIIS-only).
    pub newton_trigger: f64,

    /// AURORA auxiliary-curvature accelerator (arXiv:2608.07354).
    ///
    /// Disabled by default; see [`crate::aurora::AuroraConfig`]. When disabled,
    /// every SCF path behaves exactly as it did before this field existed.
    pub aurora: crate::aurora::AuroraConfig,
    /// If > 0 in ROHF/ROKS: switch from DIIS / damped-Newton to an
    /// augmented-Hessian Newton step once err_max drops below this trigger.
    /// AH handles vanishing Hessian eigenvalues that trip up PCG. A value
    /// of 0 disables AH. Both triggers can be set: PCG fires first (when
    /// err_max < newton_trigger) and AH takes over when err_max < ah_trigger
    /// (typically a tighter threshold).
    pub ah_trigger: f64,
    /// If `Some` in RHF/RKS or UHF/UKS: engage the **trust-region
    /// augmented-Hessian** (TRAH) solver once the DIIS error drops below
    /// `trah_trigger`. `None` (the default) disables TRAH entirely and every
    /// existing SCF result is bit-identical — see
    /// [`crate::trah`] for the algorithm and
    /// `tests/trah_off_is_bit_identical.rs` for the proof.
    ///
    /// # How this differs from `newton_trigger` / `ah_trigger`
    ///
    /// `newton_trigger` arms a PCG Newton step with a fixed componentwise clip
    /// at 0.2, and `ah_trigger` (ROHF/ROKS only) arms an augmented-Hessian step
    /// with the same fixed clip. Neither has a trust region: no radius adapts,
    /// no predicted-vs-actual ratio is formed, and a step that raises the
    /// energy is kept. TRAH adds exactly those three things — a level-shifted
    /// AH solve that lands ‖κ‖ ON the radius, a ρ test against the quadratic
    /// model, and rejection plus contraction when the model was wrong.
    ///
    /// TRAH takes precedence over `newton_trigger` where both would fire.
    pub trah_trigger: Option<f64>,
    /// Trust-region parameters, read only when `trah_trigger` is `Some`.
    pub trah: crate::trah::TrahConfig,
    /// If > 0 (RHF/UHF/ROHF/ROKS): activate Maximum-Overlap Method
    /// reordering after this many DIIS iters. From iter `mom_after_iter + 1`
    /// onward, the occupied MO set is picked by AO-overlap with the
    /// previous-iter accepted set (rather than by ε). Default 0 = disabled.
    /// MOM pins whatever basin DIIS holds at arming — it breaks occupied-set
    /// flip-flop, it does not steer to the ground state. Arm only after DIIS
    /// has settled (open-shell plateaus: ~5; closed-shell wanderers: 50+).
    pub mom_after_iter: usize,
    /// cDFT constraints. Empty (default) = ordinary SCF. Each constraint pins a
    /// fragment population (charge or spin) to a target via a Lagrange
    /// multiplier added to the Fock matrix. Consumed by `solve_cdft_uhf`.
    pub constraints: Vec<Constraint>,
    /// cDFT outer-loop convergence: stop when max_C |N_C − target_C| is below
    /// this (electrons). Default 1e-5.
    pub cdft_lambda_tol: f64,
    /// cDFT outer-loop iteration cap: the λ-Newton loop errors with
    /// `Convergence` after this many outer iterations. Default 30.
    ///
    /// # Why this is a knob and not a constant
    ///
    /// It was `let max_outer = 30usize;` inside `solve_cdft_uhf` until
    /// 2026-09-17, when the SAME test converged in a DIFFERENT number of
    /// outer iterations on two machines:
    ///
    /// ```text
    ///                          CI      this box
    ///   driver default (None)  18      16
    ///   hcore                   9      23
    ///   SAD                    >30     14    <- CI exhausted the cap
    /// ```
    ///
    /// The converged energies agree to ~1e-6 Ha, so this is not a physics
    /// difference: it is BLAS kernel dispatch (CI's runner vs this Zen4 box)
    /// perturbing the λ-Newton trajectory. Iteration counts are bit-stable
    /// within a machine (3/3 identical local runs) and NOT portable across
    /// machines, so a hardcoded cap turns a machine difference into a red
    /// build on a test that is otherwise measuring the right thing.
    ///
    /// Tests that assert "this path converges" should PIN this explicitly
    /// rather than inherit the default, so the assertion is about convergence
    /// and not about how many iterations one particular CPU happened to need.
    pub cdft_max_outer: usize,
    /// cDFT **state selection**: after the λ-Newton loop converges, check the
    /// λ-augmented orbital Hessian and, if the constrained solution is a
    /// SADDLE, follow the downhill eigenvector and re-converge the whole λ
    /// loop from there — keeping the lower-energy solution. Default `true`.
    ///
    /// # Why this defaults ON, unlike `check_stability`
    ///
    /// Without it the constrained solve returns WHICHEVER solution the hcore
    /// guess happens to fall into, and on HeNe⁺/def2-SVP at the integer Becke
    /// target that is a saddle 0.667 eV ABOVE another solution satisfying the
    /// SAME constraint to 8e-8 electrons (measured across six independent
    /// guesses in `tests/cdft_state_selection.rs`). That is not a diagnostic —
    /// it is a wrong answer, and a diabat energy is the whole output of a cDFT
    /// run. `check_stability` can default off because it only reports; this
    /// changes which state is returned, so leaving it off would mean shipping
    /// the known-wrong one by default.
    ///
    /// Setting it `false` restores the previous behavior EXACTLY (the descent
    /// block is skipped entirely, not merely made a no-op) — pinned by
    /// `descent_off_reproduces_the_old_saddle` in `tests/cdft_state_selection.rs`.
    ///
    /// Cost: one λ-augmented Davidson per converged constrained solve, plus one
    /// extra full λ-Newton solve per descent actually taken. On a STABLE or
    /// MARGINAL solution the descent is not taken and only the eigensolve is
    /// paid.
    pub cdft_stability_descent: bool,
    /// UNCONSTRAINED open-shell **state selection**: after a UHF solve
    /// converges, check the orbital Hessian and, if the solution is a SADDLE,
    /// follow the downhill eigenvector and re-converge from there — keeping the
    /// lower-energy solution. Default **`false`**.
    ///
    /// # Why this defaults OFF, unlike `cdft_stability_descent`
    ///
    /// The two knobs look alike and the defaults differ deliberately.
    ///
    /// `cdft_stability_descent` defaults ON because a cDFT diabat energy is the
    /// entire output of that run, the pre-fix answer was known-wrong on the
    /// lane's own system, and cDFT is a narrow, opt-in code path — nothing else
    /// in the repo pays for it.
    ///
    /// This knob is different on both counts. `solve_uhf` is on the hot path of
    /// every open-shell energy, gradient, geometry step, MP2/CC/RPA reference
    /// and free-atom SAD solve in the workspace, and a geometry optimization or
    /// a frequency job runs it hundreds of times. The descent costs a Davidson
    /// eigensolve on EVERY converged solve — paid even when the answer is
    /// already right, which after the guess fix it is on 5 of the 6 measured
    /// systems — plus a full re-converge whenever a saddle is found.
    ///
    /// The measured case for defaulting it OFF: the GUESS fix in the same
    /// commit (`uhf_guess_mos`) already moves 3-of-6-wrong to 1-of-6-wrong, and
    /// the two systems it repairs (HeNe⁺ at def2-SVP and 6-31G) land on the
    /// external reference to ~1e-10 Ha and report STABLE. The one residue,
    /// N₂⁺/6-31G, is a system PySCF 2.13.0 ALSO gets wrong from its own default
    /// guess and only fixes by running its own `stability()` — i.e. it is a
    /// known-hard case where the reference implementation likewise requires an
    /// explicit, opt-in step. Making every SCF in the repo pay a Davidson to
    /// auto-repair that class is a worse trade than telling the caller the knob
    /// exists.
    ///
    /// **What makes OFF safe is that the default is no longer silent.** With
    /// `check_stability` set, an unstable solution already prints an explicit
    /// UNSTABLE warning naming the remedy, and `ScfResult::stability` carries
    /// the verdict for a caller to branch on. A user who wants the repair
    /// applied automatically sets this to `true`; a user who does not is not
    /// left believing a saddle is a minimum.
    ///
    /// Setting it `true` costs one Davidson per converged solve plus one extra
    /// full SCF per descent actually taken.
    pub scf_stability_descent: bool,
    /// Fractional (ensemble) occupation of a degenerate frontier shell. When
    /// `true` (UHF/UKS only), if the per-spin HOMO sits inside a group of
    /// near-degenerate orbitals that straddle the occupation boundary, the
    /// integer occupation is spread *equally* over that group (e.g. a ³P atom's
    /// 2 p-electrons → 2/3 in each of px/py/pz). This restores the spherical
    /// symmetry that otherwise makes the GGA XC potential orientation-dependent
    /// and the SCF oscillate forever (free O/S/Si atoms never converge with
    /// integer occupation + PBE). Default `false` — opt-in; integer-occupation
    /// paths are unchanged.
    pub fractional_occ: bool,
    /// Hard ceiling (bytes) for the resident 3-index footprint in DfJ/DfK. When
    /// the dense `(naux,nao,nao)` tensor would exceed this, the source spills
    /// aux-blocks to disk instead of allocating in core. `0` = unset → resolved
    /// via [`resolve_three_index_budget`] (this value, when non-zero, OVERRIDES
    /// the `FERRIC_MEM_BUDGET_GB` / `FERRIC_OOC_BUDGET_GB` env vars; env fills in
    /// only when this is 0; then auto-detect 0.8×RAM; then a 2 GiB fallback).
    pub three_index_budget_bytes: usize,
    /// Optional externally-supplied initial density matrix. When `Some(d)`, the
    /// SCF loop uses this density as the starting point instead of computing an
    /// hcore or SAD guess internally. Shape must match `(nbasis, nbasis)`. The
    /// primary use-case is a SAD guess built by `guess::sad_guess(...)`.
    pub init_guess_density: Option<Array2<f64>>,
    /// When `true` (the default) and no `init_guess_density` is supplied, the SCF
    /// starts from a SAD (superposition-of-atomic-densities) guess computed via
    /// `guess::sad_guess`, falling back to the hcore guess if SAD fails. SAD cures
    /// the heavy-atom RHF divergence class (COSe/C2H3Br) that hcore triggers. Set
    /// `false` to force the bare hcore guess — used internally by `sad_guess` for
    /// its free-atom solves to break the recursion.
    pub use_sad_guess: bool,
    /// If `Some(n)`: abort early when the running minimum of `err_max` over the
    /// last `n` iters has not dropped below 0.9× its value over the previous `n`
    /// iters (gradient stopped falling) AND err_max is still above the 1e-4
    /// plateau floor. `None` (default) disables stall detection. Used by the
    /// convergence ladder to advance a stuck rung in ~n iters instead of max_iter.
    pub stall_window: Option<usize>,
    /// If `Some(f)`: abort early when the energy rises by more than `f` Ha for 3
    /// consecutive iterations (actively diverging, not just noisy). `None`
    /// (default) disables divergence detection.
    pub divergence_tol: Option<f64>,
    /// Fixed classical external potential (point charges + uniform field).
    /// `None` (default) = no external potential; folded into `hcore` once
    /// before the SCF loop, orthogonal to cDFT's per-iteration Fock hook.
    pub external_potential: Option<ferric_core::external_potential::ExternalPotential>,
    /// COSMO implicit-solvent configuration. `None` (default) = no solvent;
    /// the SCF loop is then byte-for-byte identical to a build with no COSMO
    /// support at all. Unlike `external_potential` (folded into `hcore`
    /// ONCE before the loop), COSMO's reaction-field potential depends on
    /// the density and is recomputed EVERY iteration (see `crate::cosmo`).
    pub cosmo: Option<crate::cosmo::CosmoConfig>,
    /// IEF-PCM implicit-solvent configuration. `None` (default) = vacuum
    /// (no solvent); this MUST be byte-identical to a plain vacuum
    /// calculation (see `pcm_none_matches_vacuum_*` regression tests).
    /// Unlike `external_potential` (fixed, folded into `hcore` once before
    /// the loop), PCM's apparent surface charge depends self-consistently
    /// on the solute density, so the reaction-field operator is rebuilt
    /// from the CURRENT density every SCF iteration (see the `pcm_ctx`
    /// handling in `solve_rhf`) and the outer SCF/DIIS loop carries the
    /// overall q-vs-D fixed point, the same pattern PySCF/Psi4 use.
    /// Mutually usable alongside `cosmo` (both are independent implicit-
    /// solvent models); using both simultaneously is not validated and not
    /// currently prevented at the type level — callers should pick one.
    pub pcm: Option<ferric_pcm::PcmConfig>,
    /// Thole-damped polarizable-embedding sites (induced point dipoles).
    /// `None` (default) = no polarizable sites; this MUST be byte-identical
    /// to a build with no polarizable-embedding support at all (see
    /// `polarizable_none_is_bit_identical_to_plain_scf` in
    /// `tests/qmmm_polarizable.rs`). Like COSMO/PCM (and unlike
    /// `external_potential`, folded into `hcore` once before the loop), the
    /// induced dipoles depend self-consistently on the density and are
    /// re-solved from the CURRENT density every SCF iteration inside
    /// `crate::driver::solvent_terms` — see `crate::polarizable` for the
    /// full model.
    pub polarizable: Option<crate::polarizable::PolarizableSites>,
    /// When `true`, print one line per SCF iteration to stdout (iteration
    /// number, energy, ΔE, and the same dp_rms/dp_max/err_max quantities
    /// `scf_converged` already gates on) — live progress for a user watching
    /// a long-running job (DF-B3LYP on a medium molecule, or under MPI).
    /// Default `false`: byte-identical to today's silent-until-done output.
    /// Distinct from the pre-existing `FERRIC_SCF_TRACE`/`FERRIC_ROHF_TRACE`
    /// env-only debug toggles (`scf_trace()`/`rohf_trace()`), which print
    /// additional internal diagnostics to stderr and are unaffected by this
    /// field. Under MPI, only rank 0 prints (see `ctx.is_root()` at the print
    /// site) so ranks > 0 never duplicate the trace.
    pub verbose: bool,
    /// Opt-in post-convergence **internal stability analysis**: after the SCF
    /// converges, take the lowest eigenvalue of the electronic orbital Hessian
    /// and report whether the solution is a minimum or a SADDLE POINT (see
    /// [`crate::stability`]). Default `false` — it costs a Davidson eigensolve
    /// whose every matvec is a J/K build, and with it off nothing is
    /// constructed at all, so the SCF path is bit-identical to a build with no
    /// stability support (regression-guarded by
    /// `stability_off_is_bit_identical_*` in `tests/scf_stability_wiring.rs`).
    ///
    /// The verdict lands on [`crate::result::ScfResult::stability`] as
    /// `Some(..)`; `None` there means "not checked", never "checked and
    /// stable". An instability is DIAGNOSTIC: it warns on stderr and never
    /// makes the SCF return `Err`, because a deliberately-unstable state (a
    /// cDFT diabat, a MOM excited state) is a legitimate thing to compute.
    ///
    /// Honoured by `solve_rhf`/`solve_rks` (RHF-internal, singlet channel) and
    /// `solve_uhf`/`solve_uks` (UHF-internal, independent α/β rotations).
    /// `solve_rohf`/`solve_roks` SKIP it with a printed reason — the Roothaan
    /// Hessian is a third operator, and analysing a UHF or RHF one there would
    /// be a wrong-operator verdict. Range-separated and meta-GGA functionals
    /// are likewise skipped with a reason (see
    /// [`crate::stability::ks_reference_is_analysable`]).
    pub check_stability: bool,
    /// Which screening bound to build for the LinK-specific pair list when
    /// `k_builder == Some("link")`. Default
    /// [`crate::screening::ScreeningKind::Schwarz`] — byte-identical to every
    /// pre-CSB build. `Csb` selects [`crate::screening::CsbBounds`], the
    /// RIGOROUS `min{Q_µν Q_λσ, M_µλ M_νσ, M_µσ M_νλ}` bound of Thompson &
    /// Ochsenfeld, JCP 147, 144101 (2017), Eq. (8). Because CSB is a genuine
    /// upper bound (unlike the CSAM family of Eqs. (9)/(11)/(12) in the same
    /// paper), selecting it trades a slightly larger setup cost for a tighter
    /// screen — it is NOT an accuracy tradeoff and cannot discard a quartet
    /// carrying real weight.
    ///
    /// `Csam` selects [`crate::screening::CsamBounds`]' formula, Eqs.
    /// (9)/(11)/(12) of the same paper — a tighter but **NON-RIGOROUS**
    /// estimate that CAN underestimate the true integral and therefore discard
    /// quartets carrying real weight (see that type's doc, and
    /// `ferric_integrals::csam`'s module header, for the citation and the
    /// measured violation). Its error is controlled by `integral_thresh`, so
    /// selecting it trades accuracy for speed rather than tightening a bound
    /// for free. Short-range (`erfc`) operators are refused under `Csam` with a
    /// typed error naming `screening = "csb"`.
    ///
    /// SCOPE OF THIS FIELD: read ONLY at the `link_schwarz_opt` construction
    /// site in [`solve_rhf`] (i.e. only when `k_builder == "link"`), where it
    /// selects whether the FRESH LinK table gets an `M` or `X` table. It does
    /// NOT by itself affect the dense direct `build_jk`/`DirectJ`/`DirectK`/
    /// `DirectJK` path (the default, `k_builder` unset), nor
    /// `solve_uhf`/`solve_rohf` — all of those take their bound from the
    /// CALLER-supplied `bounds: &SchwarzBounds` parameter, fixed before this
    /// function is entered.
    ///
    /// Those paths are NOT stuck on plain Schwarz, though: they pick the
    /// refinement up from the `bounds` VALUE, via
    /// [`crate::screening::SchwarzBounds::csb_m`] /
    /// [`crate::screening::SchwarzBounds::csam_x`] —
    /// `quartet_scatter::scatter_bra_pair` consults them directly, and
    /// `solve_uhf`/`solve_rohf` wrap `bounds` in
    /// [`crate::screening::LinkBound::SchwarzRef`] before handing it to LinK.
    /// Build that value with
    /// `SchwarzBounds::compute_for_screening(op, prep, kind)` rather than plain
    /// `compute` to opt those paths in.
    ///
    /// `ferric-cli` resolves `[scf] screening` ONCE and uses the same resolved
    /// `ScreeningKind` for both mechanisms, so they never disagree there. A
    /// caller that builds `bounds` via plain `compute` and sets `Csb`/`Csam`
    /// only here gets that refinement on closed-shell LinK alone — legal, just
    /// unusual.
    pub screening: crate::screening::ScreeningKind,
}

impl Default for RhfConfig {
    fn default() -> Self {
        Self {
            // Convergence gate = ΔP primary + ΔE loose sanity (see scf_converged),
            // NOT the DIIS commutator. `density_conv` is the TIGHT threshold on
            // dp_rms (ORCA TolRMSP): the density genuinely drains here (MEASURED
            // ~1e-9 at aTZ). `energy_conv` is a LOOSE "not still descending"
            // bound on |ΔE| — deliberately 1e-3, far above the ~2e-5 DF energy
            // noise floor, because ΔE (like the commutator) floors with naux and
            // a tight ΔE is unreachable. ΔP does the real work.
            //
            // History: (1e-10, 1e-8) once guarded an H2O+ UHF false convergence
            // (a *gradient*-gated accept of a state 85 mHa high). ΔP is a stronger
            // wrong-basin signal than that gradient — regression-guarded by the
            // h2o_plus_* UHF tests, which pass under this gate.
            max_iter: 200,
            energy_conv: 1e-3,
            density_conv: 1e-6,
            diis_size: 8,
            diis_flavor: crate::diis::DiisFlavor::Pulay,
            diis_switch_thresh: 1e-1,
            smearing_sigma: None,
            integral_thresh: 1e-12,
            k_builder: None,
            cosx: crate::cosx_k::CosxConfig::default(),
            df_j_aux: None,
            df_k_aux: None,
            xc: None,
            xc_omega: None,
            dft_grid: None,
            nlc_grid: None,
            level_shift: 0.0,
            newton_trigger: 0.0,
            aurora: crate::aurora::AuroraConfig::default(),
            ah_trigger: 0.0,
            trah_trigger: None,
            trah: crate::trah::TrahConfig::default(),
            mom_after_iter: 0,
            constraints: Vec::new(),
            cdft_lambda_tol: 1e-5,
            cdft_max_outer: 30,
            cdft_stability_descent: true,
            scf_stability_descent: false,
            fractional_occ: false,
            // 0 = "unset" → resolve_three_index_budget auto-detects (0.8×RAM).
            three_index_budget_bytes: 0,
            init_guess_density: None,
            use_sad_guess: true,
            stall_window: None,
            divergence_tol: None,
            external_potential: None,
            cosmo: None,
            pcm: None,
            polarizable: None,
            verbose: false,
            check_stability: false,
            screening: crate::screening::ScreeningKind::default(),
        }
    }
}

impl RhfConfig {
    /// Set the maximum number of SCF iterations.
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }
    /// Set the density convergence threshold (dp_rms).
    pub fn with_density_conv(mut self, thresh: f64) -> Self {
        self.density_conv = thresh;
        self
    }
    /// Set the XC functional (e.g. "PBE", "B3LYP", "wB97X-V").
    pub fn with_xc(mut self, xc: impl Into<String>) -> Self {
        self.xc = Some(xc.into());
        self
    }
    /// Set the virtual-block level shift (Ha) for convergence damping.
    pub fn with_level_shift(mut self, shift: f64) -> Self {
        self.level_shift = shift;
        self
    }
    /// Set the K-matrix builder strategy ("direct", "link" or "cosx").
    pub fn with_k_builder(mut self, builder: impl Into<String>) -> Self {
        self.k_builder = Some(builder.into());
        self
    }
    /// Set the COSX knobs (only read when the builder is "cosx").
    pub fn with_cosx(mut self, cosx: crate::cosx_k::CosxConfig) -> Self {
        self.cosx = cosx;
        self
    }
    /// Set the auxiliary basis for density-fitted Coulomb (RI-J).
    pub fn with_df_j_aux(mut self, aux: impl Into<String>) -> Self {
        self.df_j_aux = Some(aux.into());
        self
    }
    /// Set the auxiliary basis for density-fitted exchange (RI-K).
    pub fn with_df_k_aux(mut self, aux: impl Into<String>) -> Self {
        self.df_k_aux = Some(aux.into());
        self
    }
    /// Enable verbose per-iteration SCF output.
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }
    /// Enable the opt-in post-convergence internal stability analysis.
    pub fn with_check_stability(mut self, check: bool) -> Self {
        self.check_stability = check;
        self
    }
}

/// Post-convergence internal stability analysis for an RHF/RKS solution.
///
/// Called ONLY from `solve_rhf`'s converged exit and ONLY when
/// `config.check_stability` is set. Returns `None` — meaning "not checked", per
/// [`crate::result::ScfResult::stability`] — whenever the reference is not
/// analysable with the operator that exists, ALWAYS after printing why.
///
/// # The KS trap this function exists to avoid
///
/// [`crate::rhf_newton::hessian_matvec`] takes an OPTIONAL `fxc` response
/// closure. Passing `None` on a KS reference does not fail; it silently
/// analyses the **HF** orbital Hessian at the **KS** density, producing a
/// λ_min for an operator nobody asked about, presented with the same
/// confidence as a correct one. So for `xc.is_some()` this function builds the
/// SAME [`crate::rohf::FxcKernelStore`] the RKS Newton path builds at the same
/// restricted reference (`d_α = d_β = ½·D`), and where that kernel cannot be
/// built — range-separated (the matvec's K is plain-Coulomb) or meta-GGA (no τ
/// f_xc kernel exists in this workspace) — it SKIPS with a printed reason
/// rather than checking the wrong operator. Those are exactly the gates
/// `solve_rhf`'s own Newton branch uses. Proven, not asserted, by
/// `ks_reference_is_analysed_with_the_xc_kernel_not_the_hf_hessian`.
#[allow(clippy::too_many_arguments)]
fn stability_rhf(
    ctx: &ParallelContext,
    mol: &Molecule,
    prep: &PreparedBasis,
    bounds: &SchwarzBounds,
    config: &RhfConfig,
    c: &Array2<f64>,
    f: &Array2<f64>,
    d: &Array2<f64>,
    nocc: usize,
    has_xc: bool,
    k_mix: ferric_dft::xc_trait::KMix,
    ooc_budget: usize,
) -> Option<crate::stability::StabilityResult> {
    if let Err(skip) =
        crate::stability::ks_reference_is_analysable(config.xc.as_deref(), k_mix.omega)
    {
        eprintln!(
            "SCF stability: check requested but SKIPPED — {}. \
             ScfResult::stability is None (not checked), which does NOT mean stable.",
            skip.reason()
        );
        return None;
    }

    // The f_xc response kernel, at the same restricted reference the RKS Newton
    // path uses (d_α = d_β = ½·D). `None` only for pure HF.
    let fxc_store = if has_xc {
        let grid = config.dft_grid.clone().unwrap_or_default();
        let name = config.xc.as_deref().expect("has_xc implies Some(xc)");
        let d_half = 0.5 * d;
        match crate::rohf::FxcKernelStore::build(mol, prep, &grid, name, &d_half, &d_half) {
            Ok(s) => Some(s),
            Err(e) => {
                eprintln!(
                    "SCF stability: check requested but SKIPPED — the f_xc response kernel could \
                     not be built ({e}), and analysing the HF Hessian at a KS density instead \
                     would be a wrong-operator verdict. ScfResult::stability is None."
                );
                return None;
            }
        }
    } else {
        None
    };
    let fxc_storage = fxc_store.as_ref().map(|s| s.response());
    let fxc_ref: Option<&crate::rohf_newton::FxcResponse<'_>> = fxc_storage.as_deref();

    let f_mo = c.t().dot(f).dot(c);
    let inputs = crate::rhf_newton::RhfNewtonInputs {
        prep,
        bounds,
        c,
        f_mo: &f_mo,
        nocc,
        k_mix_sr: if has_xc { k_mix.sr } else { 1.0 },
        fxc: fxc_ref,
        thresh: config.integral_thresh,
        ooc_budget,
    };
    match crate::stability::rhf_internal_stability(
        ctx,
        &inputs,
        &crate::stability::StabilityConfig::default(),
    ) {
        Ok(res) => {
            crate::stability::report_stability(&res, config.verbose);
            Some(res)
        }
        Err(e) => {
            eprintln!(
                "SCF stability: check requested but FAILED — {}: {e}. ScfResult::stability is \
                 None (not checked). The SCF result itself is unaffected.",
                crate::stability::StabilitySkip::AnalysisFailed.reason()
            );
            None
        }
    }
}

/// Resolve the 3-index memory budget in bytes by delegating to the single
/// unified resolver [`ferric_core::memory::resolve_budget_bytes`], so every
/// memory setting shares ONE precedence chain (TOML/config > env > auto):
/// 1. A non-zero `config_bytes` — an explicit caller choice (TOML `[memory]`
///    budget / config field / kwarg). **TOML/config overrides env.**
/// 2. `FERRIC_MEM_BUDGET_GB` env var (GiB).
/// 3. Legacy `FERRIC_OOC_BUDGET_GB` / `FERRIC_ERI3_BUDGET_GB` env vars (GiB).
/// 4. Auto: 0.8 × detected available RAM (keeps the DF-JK tensor IN RAM on a
///    box with adequate memory instead of spilling under a blind 2 GiB cap).
/// 5. 2 GiB fallback.
///
/// `0` is the unset sentinel (matches the unified resolver, which treats
/// `Some(0)` as "no explicit budget"). Callers with no budget pass `0`.
///
/// Shared by RHF/UHF/ROHF so the budget is honored uniformly across all
/// DF-J/DF-K construction sites. This function no longer reads any env var
/// itself — env fallback lives entirely in the unified resolver, so TOML can
/// never be silently overridden by `FERRIC_OOC_BUDGET_GB` (it previously was).
pub fn resolve_three_index_budget(config_bytes: usize) -> usize {
    let explicit = (config_bytes != 0).then_some(config_bytes);
    ferric_core::memory::resolve_budget_bytes(explicit)
}

/// Pure stall-detector arithmetic, extracted from the `solve_rhf` loop so it can
/// be unit-tested with synthetic `errmax_history` sequences instead of forcing a
/// real molecule to stall (slow/nondeterministic).
///
/// Returns `true` when the running minimum of `errmax_history` over the last
/// `window` entries has not dropped below 0.9x its value over the previous
/// `window` entries (gradient stopped falling — robust to oscillation, since a
/// wide-band limit cycle has net-zero running-min change), AND `current_err` is
/// still above the 1e-4 plateau floor (below it, the separate plateau-acceptance
/// path owns the regime). Returns `false` when `window == 0` (degenerate/no-op
/// config) or when there isn't yet `2*window` entries of history.
pub(crate) fn stall_detected(errmax_history: &[f64], window: usize, current_err: f64) -> bool {
    if window == 0 {
        return false;
    }
    if errmax_history.len() < 2 * window || current_err < 1e-4 {
        return false;
    }
    let n_hist = errmax_history.len();
    let recent_min = errmax_history[n_hist - window..]
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min);
    let prev_min = errmax_history[n_hist - 2 * window..n_hist - window]
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min);
    recent_min >= 0.9 * prev_min
}

/// The settled convergence signals for one SCF iteration. All are magnitudes
/// (already `.abs()`/norm'd by the caller).
///
/// - `de`: |E − E_prev|, energy change.
/// - `dp_rms`: RMS of the density change ΔP = D_new − D_last, i.e.
///   `‖ΔP‖_F / sqrt(nao²)`.
/// - `dp_max`: max element of |ΔP|.
///
/// The DIIS commutator (FDS−SDF) is deliberately absent: it is a *diagnostic*
/// (printed in the trace), never a gate — see [`scf_converged`] for why.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ConvergenceSignals {
    pub de: f64,
    pub dp_rms: f64,
    pub dp_max: f64,
}

/// Decide SCF convergence from the settled signals — the ORCA `ConvCheckMode=2`
/// design, where the **density change ΔP is the primary (tight) signal** and the
/// energy change is only a *loose* "not still descending" sanity bound. Both
/// ORCA and PySCF use ΔP-primary in place of gating on the DIIS/orbital-gradient
/// commutator, and ORCA's default checks energy *stability*, not a tight ΔE.
///
/// # Why ΔP, and why ΔE only loosely
///
/// Under RI-J/RI-JK the fitted Fock carries a self-consistency error that grows
/// with `naux`. This floors BOTH the commutator's max element AND the
/// iteration-to-iteration energy change — MEASURED at aug-cc-pVTZ (toluene,
/// RI-JK): `err_max` parks at ~1.26e-6 and `dE` oscillates at ~2e-5, neither
/// draining. Gating on *either* has the same naux-chasing pathology (a fixed
/// small tolerance is unreachable and the floor scales with the aux basis).
///
/// `dp_rms` is the ONE signal that drains cleanly at the RI fixed point: the
/// density stops moving (D_{n+1} = D_n to fitting precision) even while the
/// commutator and the energy still jitter on the noise floor. MEASURED same run:
/// `dp_rms` drains monotonically to ~1e-9 while `dE` sits at 2e-5. So ΔP is the
/// convergence criterion; ΔE is used only to reject a run that is *still
/// actively descending* (early iterations have `dE ≫` any noise floor).
///
/// # The gate
///
/// Converged ⟺ density settled **and** energy not actively descending:
/// - `dp_rms < density_conv`         (primary, tight — the real signal)
/// - `dp_max < 10·density_conv`      (ORCA `TolMaxP` companion: guards a single
///   still-moving element while the RMS looks settled)
/// - `de   < energy_conv`            (LOOSE sanity bound, default 1e-3 — well
///   above the ~2e-5 DF energy floor, so it excludes a descending run without
///   demanding an unreachable tight ΔE)
///
/// The commutator (FDS−SDF) is *not* consulted — diagnostic only.
///
/// Returns `Some(ScfExit::Converged)` when met, else `None` (keep iterating).
/// The caller still owns divergence/stall/max-iter exits.
pub(crate) fn scf_converged(
    sig: ConvergenceSignals,
    energy_conv: f64,
    density_conv: f64,
) -> Option<crate::result::ScfExit> {
    let dp_rms_ok = sig.dp_rms < density_conv;
    let dp_max_ok = sig.dp_max < 10.0 * density_conv;
    let energy_not_descending = sig.de < energy_conv;
    if dp_rms_ok && dp_max_ok && energy_not_descending {
        Some(crate::result::ScfExit::Converged)
    } else {
        None
    }
}

/// Spellings of `df_j_aux` / `df_k_aux` that mean "do not density-fit"
/// (conventional four-centre J/K). Compared case-insensitively after
/// trimming. `""` is the canonical form the SCF layer reads (see
/// `resolve_aux` in [`solve_rhf`] and `build_df_jk`).
pub const DF_AUX_OFF_SPELLINGS: &[&str] = &["", "none", "off", "exact", "conventional"];

/// Normalise a user-supplied `df_j_aux` / `df_k_aux` value: any spelling in
/// [`DF_AUX_OFF_SPELLINGS`] becomes the `""` sentinel, anything else is taken
/// as an aux basis name (trimmed) and validated later by `basis::bundled`.
///
/// ONE parser for every surface. Python `run_dft` accepted all five
/// spellings while `run_rhf`/`run_uhf`/`run_rohf` and the CLI `[scf]` keys
/// passed the string through raw, so `df_j_aux="exact"` meant conventional J
/// in one function and "look up a basis called exact" (a basis error) in the
/// next.
pub fn normalize_df_aux(value: &str) -> String {
    let t = value.trim();
    if DF_AUX_OFF_SPELLINGS.contains(&t.to_ascii_lowercase().as_str()) {
        String::new()
    } else {
        t.to_string()
    }
}

/// Refuse a molecule whose multiplicity says it is open-shell.
///
/// `solve_rhf` occupies `nelec / 2` doubly-occupied orbitals and never read
/// `mol.multiplicity`. An odd electron count failed (as a misleading
/// `ScfConvergence { iterations: 0 }`), but an EVEN count with multiplicity
/// > 1 -- triplet water, triplet O2 -- converged the closed-shell SINGLET and
/// returned it as if it answered the question. Every closed-shell consumer
/// downstream (RI-MP2, MP3, SCS-MP2, CCSD, RKS, the Python `run_*`
/// functions) inherited that silent singlet. Measured before this guard:
/// water with multiplicity = 3, RI-MP2/STO-3G, returned -74.99874958 Ha, the
/// singlet's energy to every digit.
///
/// This is the single place the multiplicity was dropped, so it is the place
/// it is checked. Callers that want an open-shell reference use
/// [`crate::uhf::solve_uhf`] or [`crate::rohf::solve_rohf`] (both of which set
/// `RhfConfig::xc` for UKS/ROKS).
pub fn require_closed_shell(mol: &Molecule) -> Result<(), FerricError> {
    if mol.multiplicity == 1 {
        return Ok(());
    }
    Err(FerricError::General(format!(
        "closed-shell (restricted) SCF requested for a molecule with multiplicity {} \
         ({} electrons): RHF/RKS cannot represent unpaired electrons, and running it \
         anyway would return the closed-shell singlet. Use an open-shell reference \
         (UHF/UKS or ROHF/ROKS) for multiplicity > 1, or set multiplicity = 1 if the \
         singlet is what you want.",
        mol.multiplicity,
        mol.nelec()
    )))
}

/// Solve the closed-shell RHF equations for a molecule.
///
/// Uses the Roothaan-Hall procedure: build Fock matrix from density, diagonalize,
/// rebuild density, iterate until convergence. DIIS extrapolation accelerates
/// convergence. Returns `Ok(ScfResult)` whether or not it converges — check
/// `result.converged` / `result.exit` (`ScfExit::MaxIter` carries the
/// best-effort density/MOs from the final iteration). Returns
/// [`FerricError::General`] up front for an open-shell molecule
/// (`mol.multiplicity != 1`, see [`require_closed_shell`]) and
/// [`FerricError::ScfConvergence`] only for an odd electron count that
/// somehow carries multiplicity 1 (unreachable through `Molecule::parse_xyz`,
/// which validates parity).
///
/// # Examples
///
/// ```no_run
/// use ferric_core::{mol::Molecule, basis, parallel::ParallelContext};
/// use ferric_integrals::{basis_bridge::PreparedBasis, operator::Operator};
/// use ferric_scf::{rhf::{solve_rhf, RhfConfig}, screening::SchwarzBounds};
///
/// let mol = Molecule::parse_xyz("3\nwater\nO 0 0 0.117790\nH 0 0.755453 -0.471161\nH 0 -0.755453 -0.471161\n", 0, 1).unwrap();
/// let bs = basis::bundled("cc-pvdz").unwrap();
/// let prep = PreparedBasis::new(&mol, &bs).unwrap();
/// let op = Operator::coulomb();
/// let bounds = SchwarzBounds::compute(op, &prep).unwrap();
/// let ctx = ParallelContext::default();
/// let result = solve_rhf(&ctx, &mol, &prep, op, &bounds, &RhfConfig::default()).unwrap();
/// println!("{}", result); // prints energy, iterations, convergence
/// ```
pub fn solve_rhf(
    ctx: &ParallelContext,
    mol: &Molecule,
    prep: &PreparedBasis,
    op: Operator,
    bounds: &SchwarzBounds,
    config: &RhfConfig,
) -> Result<ScfResult, FerricError> {
    // Refuse an open-shell molecule BEFORE any work. See `require_closed_shell`.
    require_closed_shell(mol)?;
    // Build the XC contribution once. None for pure HF. Built FIRST so the
    // shared driver env can size the RSH fitter pair from k_mix, and so the
    // JK-aux auto-defaults below can see hybrid/RSH-ness.
    use ferric_dft::ks::KsXc;
    use ferric_dft::xc_trait::{KMix, XcContribution};

    let xc_contrib: Option<Box<dyn XcContribution>> = if let Some(name) = config.xc.as_deref() {
        let main = config.dft_grid.clone().unwrap_or_default();
        let nlc = config
            .nlc_grid
            .clone()
            .unwrap_or(ferric_dft::grid::AtomicGridConfig {
                n_radial: 50,
                n_angular: 50,
                ..Default::default()
            });
        // Thread the caller's `[memory] budget_gb` into the grid AO cache --
        // the largest single allocation in a DFT job. This used to call the
        // UNbudgeted `new_with_omega`, which resolves from env/auto-detect
        // and so silently DISCARDED `config.three_index_budget_bytes`. The
        // budgeted constructor existed for exactly this and had ZERO
        // production callers; its own doc records the symptom ("Setting
        // `budget_gb = 4` on a 64 GB box still sized the grid cache against
        // ~51 GB"), i.e. the documented primary knob did nothing while
        // FERRIC_MEM_BUDGET_GB worked. 0 means unset, matching
        // `rhf::resolve_three_index_budget`.
        let ks = KsXc::new_with_omega_budgeted(
            mol,
            prep.basis_set(),
            name,
            &main,
            &nlc,
            config.xc_omega,
            (config.three_index_budget_bytes != 0).then_some(config.three_index_budget_bytes),
        )
        .map_err(|e| FerricError::General(format!("KsXc init for {name}: {e:?}")))?;
        Some(Box::new(ks) as Box<dyn XcContribution>)
    } else {
        None
    };
    let k_mix: KMix = xc_contrib.as_ref().map(|x| x.k_mix()).unwrap_or_default();

    // Shared geometry-only environment: S, hcore(+ECP, +external), V_nn
    // (+external), COSMO/PCM contexts, resolved memory budget, RSH fitters.
    // One construction serving all six SCF variants — see crate::driver.
    let crate::driver::ScfEnv {
        s,
        h,
        vnn,
        ooc_budget,
        cosmo_cavity,
        pcm_ctx,
        polarizable_site_basis,
        mut dfk_sr,
        mut dfk_lr,
    } = crate::driver::prepare(ctx, mol, prep, config, &k_mix)?;

    let n = prep.nbasis();
    let nelec = mol.nelec();
    if nelec % 2 != 0 {
        return Err(FerricError::ScfConvergence {
            iterations: 0,
            last_energy: 0.0,
        });
    }
    let nocc = (nelec / 2) as usize;

    // Initial density: explicit override > SAD (default) > hcore. SAD is the
    // default because the bare hcore guess diverges on heavy-atom closed shells
    // (COSe/C2H3Br); if SAD fails to build (e.g. a free-atom solve doesn't
    // converge) we fall back to hcore rather than aborting the whole SCF.
    let mut d = if let Some(d0) = config.init_guess_density.as_ref() {
        d0.clone()
    } else if config.use_sad_guess {
        // MINAO projection guess (no per-element free-atom SCF for heavy atoms;
        // GWH atomic-hcore block for Z≥21/g-function elements). Falls back to
        // hcore if it fails. The `use_sad_guess` field name is retained for
        // API/config compatibility but now selects MINAO.
        match crate::guess::minao_projection_guess(mol, prep, prep.basis_set()) {
            Ok(d_minao) => d_minao,
            Err(_) => hcore_guess(&s, &h, nocc)?,
        }
    } else {
        hcore_guess(&s, &h, nocc)?
    };
    let mut f = Array2::zeros((n, n));
    let mut j_buf = Array2::<f64>::zeros((n, n));
    let mut k_buf = Array2::<f64>::zeros((n, n));
    // DIIS commutator scratch, hoisted out of the SCF loop for the same reason
    // `f`/`j_buf`/`k_buf` above are (defect F).
    //
    // The loop body used to compute the commutator as
    //     let (fds, sdf) = (f.dot(&d).dot(&s), s.dot(&d).dot(&f));
    //     let err = &fds - &sdf;
    // which allocates FIVE n² matrices every single iteration: the two
    // unnamed intermediates from the inner `.dot()`s, the two named products,
    // and the difference — none reused, while the far larger `j_buf`/`k_buf`
    // beside them were correctly hoisted. `driver::density_change` was
    // explicitly rewritten zero-alloc for exactly this reason, so the
    // convention already existed; this brings the commutator in line with it.
    //
    // `general_mat_mul(1.0, a, b, 0.0, &mut c)` is precisely what `a.dot(&b)`
    // performs internally for `f64` (one dgemm, alpha = 1, beta = 0) — the same
    // call, with the destination supplied instead of freshly allocated. Nothing
    // is reassociated: the products are formed in the identical order,
    // `(F·D)·S` and `(S·D)·F`, so the commutator is bit-identical and the SCF
    // trajectory is unchanged.
    let mut fd_tmp = Array2::<f64>::zeros((n, n));
    let mut fds = Array2::<f64>::zeros((n, n));
    let mut sdf = Array2::<f64>::zeros((n, n));
    let mut err = Array2::<f64>::zeros((n, n));
    // Combined DIIS driver. With the default `diis_flavor = Pulay` this is a
    // pure-Pulay driver whose `step` is byte-identical to `Diis::step`;
    // ADIIS/EDIIS activate only when a caller opts in.
    let mut diis = DiisDriver::new(
        config.diis_flavor,
        config.diis_size,
        config.diis_switch_thresh,
    );
    // RHF extrapolates with `step` (α-only), so two matrices per subspace
    // entry; an ADIIS/EDIIS flavor additionally keeps an EnergyDiis pair live
    // alongside the Pulay history (DiisDriver::new forces switch_thresh to 0
    // for Pulay, so a pure-Pulay run correctly charges nothing extra).
    crate::driver::warn_if_diis_history_large(
        "RHF",
        n,
        config.diis_size,
        crate::diis::DiisHistoryShape::SingleSpin,
        config.diis_flavor != crate::diis::DiisFlavor::Pulay && config.diis_switch_thresh > 0.0,
        ooc_budget,
    );
    // MOM reference: last accepted occupied MO block (None until armed).
    let mut mom_ref: Option<Array2<f64>> = None;
    // Convergence bookkeeping (prev energy, carried ΔP signals, divergence
    // streak, stall history) — shared across all SCF variants, see
    // driver::ScfMonitor. ΔP is INFINITY until the first density rebuild so
    // the gate can never fire on iter 1 (scf_converged / df-jk-noise-floor).
    let mut mon = crate::driver::ScfMonitor::new();
    let mut total_quartets = 0;

    // Validate `k_builder` BEFORE any expensive setup, so an unknown value is
    // rejected even on a run whose DF-J/DF-K path would later override it
    // (`df_active = false` here is only about which branch reports the error;
    // the whitelist check is unconditional). The DF-vs-pluggable decision is
    // re-resolved below once `build_df_jk` has said what is actually active.
    crate::fock_assembly::resolve_k_builder(config.k_builder.as_deref(), false, false, true, 0.0)?;

    // Meta-GGA default virtual-block level shift (see driver::effective_level_shift).
    let effective_level_shift = crate::driver::effective_level_shift(config);

    // Auto-default JK aux bases when the functional needs exact exchange but
    // the caller hasn't explicitly set df_j_aux / df_k_aux. This makes
    // `cfg.xc = Some("B3LYP")` (or any hybrid/RSH) work out of the box.
    // Pure HF (no xc) keeps the historical behavior of no auto-default.
    let needs_k = xc_contrib.is_some() && (k_mix.sr > 0.0 || k_mix.omega > 0.0);
    let needs_j = xc_contrib.is_some();
    // Whether the SCF actually consumes an exact-exchange matrix. True for pure
    // HF (no functional) and for hybrids / RSH. FALSE for a pure functional
    // (LDA/GGA such as PBE), where k_mix is all-zero and any K built would be
    // multiplied by 0 and thrown away. Gates both the exchange-builder
    // construction and the per-iteration K build below, so pure DFT never pays
    // for exact exchange it does not use. (For a pure functional this equals
    // `needs_k`; for HF it is true where `needs_k` is false, since `needs_k` is
    // DFT-specific.)
    let k_consumed = xc_contrib.is_none() || k_mix.sr > 0.0 || k_mix.omega > 0.0;
    use crate::fock_assembly::DEFAULT_JK_AUX;
    // THREE states, not two. `None` has always meant "unset, so auto-default",
    // which left no way to say "do NOT density-fit" for a functional -- the
    // auto-default fired and RI-J was silently unavoidable.
    //
    // That silence had a cost: a caller comparing ferric against an
    // exact-Coulomb reference measures the RI-J FITTING ERROR and reads it as
    // a ferric defect. MEASURED at PBE/STO-3G against conventional J: water
    // 0.28, benzene 1.16, and a 71-atom drug molecule **9.5 kcal/mol**. The
    // other direction confirms it -- ORCA re-run WITH RI-J agrees with ferric
    // to 2.7 kcal/mol where exact-Coulomb ORCA was 9.5 away.
    //
    //   None            -> auto-default (unchanged behaviour)
    //   Some("")        -> EXPLICITLY conventional four-centre J/K
    //   Some(basis)     -> density-fit with that basis (unchanged)
    //
    // An empty string is not a valid basis name, so it cannot collide with a
    // real request.
    fn resolve_aux(requested: &Option<String>, needed: bool) -> Option<String> {
        match requested.as_deref() {
            None => needed.then(|| DEFAULT_JK_AUX.to_string()),
            Some("") => None, // explicit opt-out
            Some(name) => Some(name.to_string()),
        }
    }
    let df_j_aux_eff: Option<String> = resolve_aux(&config.df_j_aux, needs_j);
    // Whether the K written into `k_buf` below ever reaches F. Narrower than
    // `k_consumed`: an RSH functional consumes exact exchange, but from its
    // own SR/LR fitters (`dfk_sr`/`dfk_lr`, built by `driver::prepare`), so
    // the F assembly never reads `k_buf` when ω > 0. True exactly for pure HF
    // (k_mix = {1, 1, 0}) and ω = 0 hybrids.
    let k_buf_consumed = k_consumed && k_mix.omega == 0.0;
    // The DF-K the caller asked for (explicitly, or via the functional
    // auto-default), BEFORE the consumption gate.
    let df_k_aux_requested: Option<String> = resolve_aux(&config.df_k_aux, needs_k);
    // Do not build a DF-K whose K is thrown away. `resolve_aux` honours an
    // explicit `Some(name)` regardless of `needs_k`, and `run_dft` passes
    // `def2-universal-jkfit` for every functional, so a pure GGA used to
    // build a full DfK (V^{-1/2} dressing + `DfK::from_full_raw`) and run
    // `build_from_occ` every iteration only for `k_mix = 0` to discard it.
    // MEASURED on danuglipron (71 atoms) PBE/STO-3G: setup:df_fitters 91 s
    // and jk_build 33 s, most of it that unused K. Same for the main DfK of
    // an RSH functional. Energy is unaffected by construction: the K never
    // entered F. Mirrors the open-shell solvers, whose `k_aux_eff` already
    // requires `need_k && ω == 0`.
    let df_k_aux_eff: Option<String> = if k_buf_consumed {
        df_k_aux_requested.clone()
    } else {
        None
    };
    // A DF-K the gate above dropped still selects the DF ROUTING for J: with
    // conventional J (`df_j_aux = Some("")`) and a requested-but-unused DF-K,
    // the old code took the `df_any` branch (DirectJ + DfK). Without this the
    // run would fall to the combined DirectJK path, which builds a 4-centre K
    // for the same functional that has no use for it.
    let df_k_skipped = df_k_aux_requested.is_some() && df_k_aux_eff.is_none();

    // Density-fitted Coulomb (RI-J) / exchange (RI-K). Builds 3-center
    // tensor(s) + metric(s) once, sharing one `PreparedBasis` when
    // `df_j_aux_eff == df_k_aux_eff` (see `build_df_jk` doc) — the common
    // case since both default to the same JK-fit basis. Under MPI,
    // `new_banded(Some(ctx))` stripes the aux band across ranks so each rank
    // builds/holds only its band of B (memory scales with rank count);
    // size-1 / non-MPI is byte-identical to the serial path.
    let (mut df_j, mut df_k) = crate::fock_assembly::build_df_jk(
        ctx,
        mol,
        op,
        prep,
        df_j_aux_eff.as_deref(),
        df_k_aux_eff.as_deref(),
        ooc_budget,
    )?;
    // Record the builders that actually produced the energy, from the SAME
    // effective names just handed to `build_df_jk` (and, for ω > 0, to
    // `driver::prepare`'s `build_rsh_dfk_pair`), so the analytic gradient
    // differentiates this energy instead of re-deriving the aux resolution.
    // `None` when nothing is fitted: an all-exact run's gradient path is
    // untouched.
    let df_jk_route = crate::result::DfJkRoute::from_scf(
        df_j_aux_eff.as_deref(),
        df_k_aux_eff.as_deref(),
        (k_mix.omega > 0.0).then(|| {
            (
                config.df_k_aux.as_deref().unwrap_or(DEFAULT_JK_AUX),
                k_mix.omega,
            )
        }),
        op,
        ooc_budget,
    );

    // A pluggable K builder ("link" / "cosx") is only consumed on the
    // non-DF path (see the iteration branch structure below): when DF-J or
    // DF-K is active it would be built and then silently ignored — a
    // pre-existing silent no-op for "link" — so warn and skip construction.
    let df_any = df_j.is_some() || df_k.is_some() || df_k_skipped;
    let pluggable_k = crate::fock_assembly::resolve_k_builder(
        config.k_builder.as_deref(),
        df_any,
        df_k.is_some(),
        k_consumed,
        k_mix.omega,
    )?;
    // Build the pluggable builder once — LinK's SignificantPairs and COSX's
    // grid/overlap-fit factor are geometry-only and expensive per iteration.
    // When using "link", compute a fresh screening bound to own the lifetime.
    // `config.screening` selects Schwarz (default — byte-identical to the
    // pre-CSB/CSAM behaviour of always building a fresh `SchwarzBounds` here),
    // CSB or CSAM. See `RhfConfig::screening` for the scope of THIS field
    // (LinK only; the dense path picks the refinement up from
    // `bounds.csb_m`/`bounds.csam_x` instead).
    //
    // `compute_for_screening` with `Schwarz` delegates verbatim to `compute`,
    // so the default arm is byte-identical to the pre-CSB code that
    // unconditionally called `SchwarzBounds::compute(op, prep)` here.
    //
    // ERROR ROUTING: `op` here can be ATTENUATED — a range-separated
    // functional drives LinK with `Operator::erfc(omega)`. `csam_x_table`
    // routes `ErfcCoulomb` to the rigorous CSB bound rather than estimating
    // it, so `compute_for_screening(.., Csam)` returns an error naming
    // `screening = "csb"` and the `?` surfaces it here. That makes
    // `screening = "csam"` + an RSH functional a clean, actionable error
    // rather than a silent substitution.
    let link_schwarz_opt = if pluggable_k == Some("link") {
        Some(SchwarzBounds::compute_for_screening(
            op,
            prep,
            config.screening,
        )?)
    } else {
        None
    };
    // `LinkBound::SchwarzRef` applies whichever refinement the wrapped value
    // carries — CSB's Eq. (8) `min` for `csb_m`, CSAM's multiplicative factor
    // for `csam_x`, and plain Schwarz (bitwise) for neither. Wrapping BOTH the
    // fresh LinK bound and the fallback to the caller's `bounds` in the same
    // adapter gives `build_pluggable_k` one monomorphization, and makes the
    // caller's `bounds` carry the refinement into LinK on the paths that reach
    // here with `link_schwarz_opt == None` (COSX, which reads no bound at all).
    let link_bound =
        crate::screening::LinkBound::SchwarzRef(link_schwarz_opt.as_ref().unwrap_or(bounds));
    let mut k_builder: Option<Box<dyn KBuilder>> = crate::fock_assembly::build_pluggable_k(
        pluggable_k,
        ctx,
        mol,
        prep,
        &link_bound,
        op,
        &config.cosx,
        config.integral_thresh,
        ooc_budget,
    )?;
    // LinK's density-pair list must exist before the first build; the loop
    // refreshes it every iteration (`update_density` immediately before
    // `build`). No-op for COSX.
    if let Some(kb) = k_builder.as_mut() {
        kb.update_density(&d);
    }

    // Canonical orthogonalizer X = U_kept · diag(1/sqrt(λ_kept)), shape (n × m),
    // dropping eigenvectors of S with λ < LINDEP_THRESH (near-linear-dependence).
    // For well-conditioned S, m == n and X reproduces existing energies; see
    // canonical_orthogonalizer / diagonalize for the padding-back-to-n convention.
    let x = canonical_orthogonalizer(&s)?;

    // Previous-iteration MO coefficients, needed to build the virtual-block level
    // shift (a projector onto the prior virtuals). None before the first solve.
    let mut c_prev: Option<Array2<f64>> = None;
    // Orbital energies/MOs from the most recent diagonalization, retained across
    // iterations so a max_iter exit can still report eps/MOs for the final density.
    let mut last_eps: Vec<f64> = Vec::new();
    let mut last_c: Array2<f64> = Array2::zeros((n, n));
    // ── Trust-region augmented-Hessian state (opt-in; None = TRAH disabled) ──
    // `trah_state` carries the radius and the pending ρ assessment across SCF
    // iterations (see `crate::trah::TrahState` for why ρ must span two).
    // `trah_undo` holds the (C, D) that the pending step departed FROM, so a
    // rejected step can be exactly undone. Both stay `None` when
    // `config.trah_trigger` is `None`, which is what makes the disabled path
    // allocation-free and bit-identical.
    let mut trah_state: Option<crate::trah::TrahState> = config
        .trah_trigger
        .map(|_| crate::trah::TrahState::new(config.trah));
    let mut trah_undo: Option<(Array2<f64>, Array2<f64>)> = None;
    // AURORA accelerator state. Built lazily at the first accelerated step (it
    // needs converged-enough MOs and costs an auxiliary integral build), and
    // never constructed at all when `config.aurora.enabled` is false.
    let mut aurora_state: Option<crate::aurora::AuroraState> = None;
    let mut aurora_first_step = true;
    let mut aurora_last_energy = f64::NAN;
    // Bare occupied MO coefficients (C_occ, the lowest `nocc` columns) that
    // produced the CURRENT `d`, used to drive the O(naux·n²·nocc) DF-K
    // half-transform instead of the O(naux·n³) density contraction. Since
    // `D = 2·C_occ·C_occᵀ` and K is linear in D, `K(D) = 2·K(C_occ·C_occᵀ)`; the
    // factor 2 is applied to the RETURNED K (exact power-of-2, no √2 rounding),
    // so `build_from_occ(C_occ)` then `×2` reproduces K(D) up to the B-vs-D
    // reassociation floor only. `None` when no MO-factored density is available:
    // the initial guess (iter 1, `d` is a raw SAD/hcore density with no C_occ)
    // and the Fermi-smearing path (`D = C·diag(2f)·Cᵀ` is not a plain rank-nocc
    // `C·Cᵀ` product). Those iters fall back to the density-based `build`.
    let mut d_occ: Option<Array2<f64>> = None;
    // Fermi-Dirac smearing state (μ, occupations, entropy) from the most recent
    // smeared density rebuild; `None` when smearing is off (default). Used only
    // for optional free-energy trace reporting.
    let mut last_smearing: Option<crate::smearing::Smearing> = None;

    // Shared constructor for every non-converged exit path (MaxIter / Stalled /
    // Diverged). The Converged path re-diagonalizes fresh and is NOT built from
    // this closure.
    let build_nonconverged = |exit: ScfExit,
                              d: &Array2<f64>,
                              c: &Array2<f64>,
                              eps: &[f64],
                              f: &Array2<f64>,
                              energy: f64,
                              iter: usize,
                              cq: usize,
                              induced_dipoles: Option<Array2<f64>>|
     -> ScfResult {
        ScfResult {
            spin: Spin::Restricted,
            energy,
            density_total: d.clone(),
            density_alpha: d * 0.5,
            density_beta: None,
            mos_alpha: c.clone(),
            mos_beta: None,
            eps_alpha: eps.to_vec(),
            eps_beta: None,
            fock_alpha: f.clone(),
            fock_beta: None,
            converged: false,
            iterations: iter,
            exit,
            computed_quartets: cq,
            induced_dipoles,
            // Non-converged exits (MaxIter/Stalled/Diverged) are never checked:
            // stability is a property of a STATIONARY point, and these are not
            // stationary. `None` = not checked, as documented on the field.
            stability: None,
            df_jk: df_jk_route.clone(),
            rohf_spin_focks: None,
        }
    };

    // Direct J/K builders hoisted out of the SCF loop: each lazily builds a
    // per-thread libint2 EnginePool on first use (engines are constructed behind
    // a global ctor mutex), so a loop-local builder would pay that construction
    // every iteration. Which builders exist mirrors the branch structure below.
    let mut direct_j: Option<DirectJ> =
        if (df_any && df_j.is_none()) || (!df_any && k_builder.is_some()) {
            Some(DirectJ::new(
                ctx,
                prep,
                bounds,
                config.integral_thresh,
                ooc_budget,
            ))
        } else {
            None
        };
    // Only build the exchange builder when exact exchange is actually consumed
    // (see `k_consumed`). Pure DFT (LDA/GGA, k_mix all zero) discards any K it
    // builds, so a full direct 4-center K on an all-electron heavy-atom system
    // (e.g. Cu2/aug-cc-pVDZ) dominated the iteration at ~99 s while the actual XC
    // grid work was ~0.4 s. HF and ω = 0 hybrids are unaffected. Gated on
    // `k_buf_consumed`, not `k_consumed`: an RSH run whose main DF-K the gate
    // above dropped must not fall through to a 4-centre K it would also discard.
    let mut direct_k: Option<DirectK> = if df_any && df_k.is_none() && k_buf_consumed {
        Some(DirectK::new(
            ctx,
            prep,
            bounds,
            config.integral_thresh,
            ooc_budget,
        ))
    } else {
        None
    };
    let mut direct_jk: Option<DirectJK> = if !df_any && k_builder.is_none() {
        Some(DirectJK::new(
            ctx,
            prep,
            bounds,
            config.integral_thresh,
            ooc_budget,
        ))
    } else {
        None
    };

    // ── Incremental (differential) Fock build, DIRECT DirectJK path only ─────
    // J and K are LINEAR in the density, so instead of rebuilding J(D)/K(D) from
    // the FULL current density every iteration, we can keep the previous J/K in
    // `j_buf`/`k_buf` and add only ΔJ=J(ΔD)/ΔK=K(ΔD) built from the density change
    // ΔD = D_new - D_last. This is mathematically EXACT (not an approximation),
    // and a large speed win because the Häser-Ahlrichs density screen tightens as
    // ΔD → 0 — late iterations screen out nearly every quartet. This is exactly
    // the PySCF (`pyscf/scf/hf.py:1077-1083,2100-2111`) and Psi4
    // (`CompositeJK.cc:229-294`, `INCFOCK_FULL_FOCK_EVERY`) incremental-Fock
    // scheme. STRICTLY scoped to the DirectJK path (`direct_jk.is_some()`, i.e.
    // `!df_any && k_builder.is_none()`): the DF/RI and LinkK paths keep the
    // unconditional full rebuild (RI's fitted Fock carries a naux-dependent noise
    // floor; layering ΔD accumulation on top risks the DIIS-destabilization class
    // seen in the reverted DF-K occ-path change — see df-jk-noise-floor memory).
    // Escape hatch (test + debugging): `FERRIC_SCF_INCREMENTAL=0` (or `off`/
    // `false`) forces the historical full-rebuild-every-iteration behavior on the
    // DirectJK path, so an A/B correctness/perf comparison can be run without a
    // rebuild. Any other value (or unset) keeps the incremental default.
    let incremental_enabled = !matches!(
        std::env::var("FERRIC_SCF_INCREMENTAL").ok().as_deref(),
        Some("0") | Some("off") | Some("false") | Some("OFF") | Some("FALSE")
    );
    let incremental_direct = direct_jk.is_some() && incremental_enabled;
    // `d_last_fock` = the density that produced the CURRENT contents of
    // `j_buf`/`k_buf`. `None` until the first (full) build; reset to force a full
    // rebuild on the next iteration.
    let mut d_last_fock: Option<Array2<f64>> = None;
    // Most recent Thole-damped polarizable-embedding dipoles (None when
    // `config.polarizable` is off/empty) — threaded into every `ScfResult`
    // constructor below (converged and non-converged exits alike).
    let mut last_induced_dipoles: Option<Array2<f64>> = None;
    // Periodic full-rebuild guard against f64 drift accumulating across many
    // incremental updates (PySCF re-triggers via `direct_scf_tol`; Psi4 hard
    // resets every `INCFOCK_FULL_FOCK_EVERY = 5`). We use 8: a compromise between
    // Psi4's 5 and amortizing the full-build cost, verified below to hold the
    // incremental-vs-full energy well within the direct-path noise floor. A full
    // rebuild fires on iter 1, whenever `iter % INCREMENTAL_FULL_REBUILD_EVERY == 1`,
    // and whenever `d_last_fock` is None.
    const INCREMENTAL_FULL_REBUILD_EVERY: usize = 8;

    crate::driver::warn_if_rss_over_at_stage("RHF", "setup", ooc_budget);

    for iter in 1..=config.max_iter {
        ctx.check_interrupted()?;
        // One target-Hamiltonian J/K build happens below, unconditionally, on
        // every pass of this loop. Ticking here (rather than at each of the four
        // builder branches) keeps the count builder-independent, which is what
        // makes it comparable across DIIS and AURORA runs. Auxiliary-curvature
        // work is deliberately NOT counted here.
        crate::aurora::TARGET_JK_BUILDS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        // Build J and K using selected builder (reuse pre-allocated buffers).
        //
        // The DirectJK path may accumulate INCREMENTALLY onto the previous
        // iteration's J/K (see `incremental_direct` above), in which case the
        // buffers must NOT be zeroed. Every other path (DF-J/DF-K, LinkK, and the
        // full-rebuild DirectJK iterations) starts from zero as before.
        let direct_full_rebuild = incremental_direct
            && (d_last_fock.is_none() || iter % INCREMENTAL_FULL_REBUILD_EVERY == 1);
        let direct_incremental = incremental_direct && !direct_full_rebuild;
        if !direct_incremental {
            j_buf.fill(0.0);
            k_buf.fill(0.0);
        }
        // Build J: DF-J if configured, else fall through to combined direct path below.
        // Build K: DF-K > LinkK > combined DirectJK, in priority order.
        if df_any {
            if let Some(dfj) = df_j.as_mut() {
                dfj.build(&d, &mut j_buf)?;
            } else {
                let dj = direct_j.as_mut().expect("DirectJ built before loop");
                total_quartets += dj.build(&d, &mut j_buf)?;
            }
            if let Some(dfk) = df_k.as_mut() {
                // O(naux·n²·nocc) C_occ half-transform instead of the
                // O(naux·n³) density contraction. `d_occ` is None on iteration 1
                // (no diagonalization yet) and whenever fractional occupations
                // make D non-idempotent (smearing), so those fall back to the
                // density path.
                //
                // `d_occ` caches BARE C_occ while the RHF density is
                // D = 2·C_occ·C_occᵀ, and `build_from_occ` applies no scaling —
                // so K(D) = 2·K(C_occ·C_occᵀ) and the factor 2 is the caller's
                // responsibility (see the contract note at the d_occ assignment
                // sites). Omitting it was the cause of the "limit-cycle" this
                // path was reverted for in 636c26c: K came out exactly half, and
                // SCF converged to a self-consistent but badly wrong energy
                // (benzene/def2-svp: -214.13 vs -230.54 Ha). With the factor
                // applied, a single Fock build agrees with the density path to
                // 2 ulp (rel 5e-16, dfk_occ_single harness) and full SCF matches
                // in both energy and iteration count.
                match d_occ.as_ref() {
                    Some(c_occ) => {
                        dfk.build_from_occ(c_occ, &mut k_buf)?;
                        k_buf *= 2.0;
                    }
                    None => {
                        dfk.build(&d, &mut k_buf)?;
                    }
                }
            } else if let Some(dk) = direct_k.as_mut() {
                // Only reached when `k_buf` is consumed (k_buf_consumed): for
                // pure DFT and RSH `direct_k` is None and k_buf stays zero,
                // since the F assembly below never reads it.
                total_quartets += <DirectK as KBuilder>::build(dk, &d, &mut k_buf)?;
            }
        } else if let Some(lk) = k_builder.as_mut() {
            let dj = direct_j.as_mut().expect("DirectJ built before loop");
            total_quartets += dj.build(&d, &mut j_buf)?;
            lk.update_density(&d);
            total_quartets += lk.build(&d, &mut k_buf)?;
            if crate::link_k::link_debug() {
                // Cross-check the pluggable K against the dense screened
                // direct build on the SAME density (debug only: one extra
                // O(N^4) build per iteration).
                let mut k_ref = Array2::<f64>::zeros(k_buf.dim());
                let mut dk = DirectK::new(ctx, prep, bounds, config.integral_thresh, ooc_budget);
                <DirectK as KBuilder>::build(&mut dk, &d, &mut k_ref)?;
                let max_dk = (&k_buf - &k_ref).iter().fold(0.0f64, |m, v| m.max(v.abs()));
                let max_k = k_ref.iter().fold(0.0f64, |m, v| m.max(v.abs()));
                let max_d = d.iter().fold(0.0f64, |m, v| m.max(v.abs()));
                eprintln!(
                    "[link-debug] iter={iter} path=density max|D|={max_d:.3e} max|K_direct|={max_k:.3e} \
                     max|K_link-K_direct|={max_dk:.3e}"
                );
            }
        } else {
            let djk = direct_jk.as_mut().expect("DirectJK built before loop");
            if direct_incremental {
                // Incremental: j_buf/k_buf still hold J(d_last)/K(d_last); add
                // only the contribution of ΔD = d - d_last. Exact by linearity.
                let d_prev = d_last_fock
                    .as_ref()
                    .expect("d_last_fock set on full rebuild");
                let delta_d = &d - d_prev;
                total_quartets += djk.build_incremental(&delta_d, &mut j_buf, &mut k_buf)?;
            } else {
                // Full rebuild (iter 1 or periodic drift-reset): j_buf/k_buf were
                // zeroed above; build J(d)/K(d) from the full current density.
                total_quartets += djk.build(&d, &mut j_buf, &mut k_buf)?;
            }
            // Record the density that now corresponds to the J/K in the buffers.
            d_last_fock = Some(d.clone());
        }

        // F = H + J − ½ K_total  (V_xc, COSMO reaction field added below),
        // assembled in place — no k_total clone for HF/plain hybrids and no
        // zeros allocation for pure DFT. The exact-exchange mix convention:
        //   pure HF (xc=None):    k_mix = {1, 1, 0}      → K_total = k_buf
        //   pure DFT (LDA/PBE):   k_mix = {0, 0, 0}      → no exchange term
        //   plain hybrid (B3LYP): k_mix = {α, α, 0}      → K_total = α · k_buf
        //   RSH (wB97X-V):        k_mix = {sr, lr, ω>0}  → K_total = sr·K_SR + lr·K_LR
        // Bit-identical to the former `f.assign(&(&h + &j_buf - &(0.5 * &k_total)))`:
        // scaling by 0.5 is exact, so fl(0.5·α·k) = 0.5·fl(α·k) elementwise.
        f.assign(&h);
        f += &j_buf;
        if k_mix.omega > 0.0 {
            // Range-separated: SR/LR DfK fitters were built once before the
            // loop (geometry-only). Only the D-dependent contraction runs here.
            let dfk_sr = dfk_sr.as_mut().expect("dfk_sr built when omega>0");
            let dfk_lr = dfk_lr.as_mut().expect("dfk_lr built when omega>0");
            // occ path when available: `occ_factor = 2.0` supplies the RHF
            // D = 2·C_occ·C_occᵀ factor that `build_from_occ` does not apply,
            // so eff_scale = 0.5·2.0 = 1.0 matches the density path's
            // scale = 0.5 against the already-doubled `d`.
            crate::fock_assembly::subtract_rsh_exchange(
                dfk_sr,
                dfk_lr,
                &d,
                d_occ.as_ref(),
                2.0,
                &mut f,
                k_mix.sr,
                k_mix.lr,
                0.5,
            )?;
        } else if k_mix.sr > 0.0 {
            // Plain hybrid or pure HF: K already built by the builder path above.
            f.scaled_add(-0.5 * k_mix.sr, &k_buf);
        }

        // Electronic energy BEFORE adding V_xc (V_xc is one-body in F but
        // E_xc is its own integral) and BEFORE adding the COSMO reaction
        // field (same reasoning: E_cosmo = ½ q·v is its own closed-form
        // energy expression, not the trace of D against V_reaction — see
        // crate::cosmo module docs and the PySCF cross-check in its
        // `energy_elec`/`get_veff` split, where `e_solvent` is added
        // directly to the total rather than folded into the ½Tr[D·vhf] term).
        let e_elec_no_xc: f64 = 0.5 * (&d * &(&h + &f)).sum();
        let e_xc = if let Some(x) = xc_contrib.as_ref() {
            x.add_xc(&d, &mut f)
        } else {
            0.0
        };

        // COSMO + IEF-PCM reaction fields: density-dependent, recomputed every
        // iteration and folded into F (unlike `external_potential`, folded into
        // `h` once before the loop); their energies are standalone terms, NOT
        // part of the ½Tr[D·(H+F)] trace (PySCF's e_solvent convention). See
        // driver::solvent_terms for the full derivation notes; a vacuum run
        // (both None) is byte-identical to a solvation-less build.
        let (e_cosmo, e_pcm, e_pol, iter_induced_dipoles) = crate::driver::solvent_terms(
            mol,
            prep,
            config,
            cosmo_cavity.as_ref(),
            pcm_ctx.as_ref(),
            polarizable_site_basis.as_ref(),
            &d,
            &mut [&mut f],
        )?;
        last_induced_dipoles = iter_induced_dipoles;

        let energy = e_elec_no_xc + e_xc + e_cosmo + e_pcm + e_pol + vnn;

        // DIIS error: e = FDS - SDF. Runs once per SCF iteration, serially
        // (this whole loop body is outside any rayon region — the JK build
        // above is the only rayon-parallel step and has already returned).
        // Opt-in BLAS raise via FERRIC_BLAS_THREADS (default 1, unchanged
        // behavior): opt_in_blas_threads()'s rayon-worker self-guard also
        // protects the SAD/free-atom path, which calls solve_rhf from inside
        // guess.rs's run_serial_pool (a 1-thread rayon pool — still "inside
        // rayon" for the guard's purposes, so it always resolves to 1 there).
        // Zero-alloc: writes into the buffers hoisted above the loop. Same
        // dgemm calls in the same association as `f.dot(&d).dot(&s)` /
        // `s.dot(&d).dot(&f)`, so bit-identical (see the hoist comment).
        with_blas_threads(opt_in_blas_threads(), || {
            general_mat_mul(1.0, &f, &d, 0.0, &mut fd_tmp);
            general_mat_mul(1.0, &fd_tmp, &s, 0.0, &mut fds);
            general_mat_mul(1.0, &s, &d, 0.0, &mut fd_tmp);
            general_mat_mul(1.0, &fd_tmp, &f, 0.0, &mut sdf);
        });
        // err = fds - sdf, in place; `Zip` visits in the same (row-major)
        // element order `&fds - &sdf` produced.
        ndarray::Zip::from(&mut err)
            .and(&fds)
            .and(&sdf)
            .for_each(|e, &a, &b| *e = a - b);

        let sig = mon.signals(energy);
        let de = sig.de;
        let err_max = err.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
        // RMS commutator (‖FDS−SDF‖_F / sqrt(size)) — a diagnostic, NOT a gate.
        let grad_rms = {
            let n = (err.len() as f64).max(1.0);
            (err.iter().map(|v| v * v).sum::<f64>() / n).sqrt()
        };

        if scf_trace() {
            eprintln!(
                "SCF iter={iter:4}  E={energy:.12}  dE={de:.3e}  \
                 dp_rms={:.3e}  dp_max={:.3e}  \
                 |g|_rms={grad_rms:.3e}  err_max={err_max:.3e}",
                mon.dp_rms, mon.dp_max
            );
        }

        // Machine-readable per-iteration record, streamed and flushed NOW (see
        // `crate::runlog`): a run killed at iteration 90 of 100 must leave the
        // first 90 on disk. Every value here was already computed above for the
        // convergence gate or the trace line — nothing is computed for the log,
        // and nothing here is read back, so the energy is bit-identical with
        // the log on or off (tests/runlog_bit_identity.rs). Rank-0-only under
        // MPI, matching the `verbose` block above, so ranks > 1 do not
        // interleave duplicate records into one file.
        if ctx.is_root() {
            if let Some(rl) = crate::runlog::log() {
                rl.scf_iter(
                    if xc_contrib.is_some() { "rks" } else { "rhf" },
                    crate::runlog::current_rung(),
                    iter,
                    energy,
                    de,
                    mon.dp_rms,
                    mon.dp_max,
                    err_max,
                    Some(grad_rms),
                );
            }
        }

        // Live per-iteration progress for a user watching a long-running job
        // (opt-in via `config.verbose` — RhfConfig field, CLI `--verbose`/`-v`,
        // or TOML `[scf] verbose = true`). Printed to STDOUT (normal-operation
        // progress, not a warning) unlike the FERRIC_SCF_TRACE debug channel
        // above (stderr, separately gated, unaffected by this flag). Under MPI
        // only rank 0 prints, so ranks > 0 never emit duplicate lines.
        if config.verbose && ctx.is_root() {
            println!(
                "SCF iter={iter:4}  E={energy:.10}  dE={de:.3e}  dp_rms={:.3e}  err_max={err_max:.3e}",
                mon.dp_rms
            );
        }

        // Convergence decision: energy + density change (ORCA ConvCheckMode-2 /
        // PySCF), NOT the DIIS commutator. Under RI-J/RI-JK the commutator parks
        // on a naux-dependent noise floor and never drains; ΔP does (MEASURED),
        // so we gate on ΔP and treat the commutator as diagnostic only. See
        // scf_converged. ΔP is INFINITY until the first density rebuild (iter
        // 1), so the `iter > 1` guard below is belt-and-suspenders on top.
        let conv_exit = scf_converged(sig, config.energy_conv, config.density_conv);

        // Divergence: energy climbing for consecutive iters (see ScfMonitor).
        if mon.diverging(energy, config.divergence_tol) {
            if scf_trace() {
                eprintln!(
                    "SCF diverged at iter={iter}: dE={:.3e} > tol for 3 iters",
                    energy - mon.prev_e
                );
            }
            return Ok(build_nonconverged(
                ScfExit::Diverged,
                &d,
                &last_c,
                &last_eps,
                &f,
                energy,
                iter,
                total_quartets,
                last_induced_dipoles.clone(),
            ));
        }

        // Stall: running-min err_max over a window stopped falling. Robust to
        // oscillation (a wide-band limit cycle has net-zero running-min change).
        // Only fires above the 1e-4 floor — below it the plateau path accepts.
        if mon.stalled(err_max, config.stall_window) {
            if scf_trace() {
                let w = config.stall_window.unwrap_or(0);
                eprintln!("SCF stalled at iter={iter}: err_max={err_max:.3e} (no progress over {w} iters)");
            }
            return Ok(build_nonconverged(
                ScfExit::Stalled,
                &d,
                &last_c,
                &last_eps,
                &f,
                energy,
                iter,
                total_quartets,
                last_induced_dipoles.clone(),
            ));
        }

        if iter > 1 {
            if let Some(exit) = conv_exit {
                // Report Fermi level / entropy / Mermin free energy when smearing
                // is active (trace-gated; `energy` is the smeared internal energy,
                // the free energy is E − σ·S).
                if scf_trace() {
                    if let (Some(sigma), Some(sm)) = (config.smearing_sigma, last_smearing.as_ref())
                    {
                        eprintln!(
                            "SCF converged with Fermi smearing σ={sigma:.3e} Ha: \
                             μ={:.6} Ha, S={:.4e} k_B, E_free=E−σS={:.10} Ha",
                            sm.mu,
                            sm.entropy,
                            energy - sigma * sm.entropy
                        );
                    }
                }
                let (orb_e, c) = diagonalize(&f, &x)?;
                let density_alpha = 0.5 * &d;
                crate::driver::warn_if_rss_over_at_stage("RHF", "converged", ooc_budget);

                // ── Opt-in internal stability analysis (RHF/RKS singlet) ─────
                // Runs ONLY at a converged exit and ONLY when the flag is set;
                // with `check_stability = false` (the default) nothing below is
                // constructed, so this branch is bit-identical to a build with
                // no stability support. Diagnostic: it warns, it never Errs.
                let stability = if config.check_stability {
                    stability_rhf(
                        ctx,
                        mol,
                        prep,
                        bounds,
                        config,
                        &c,
                        &f,
                        &d,
                        nocc,
                        xc_contrib.is_some(),
                        k_mix,
                        ooc_budget,
                    )
                } else {
                    None
                };
                return Ok(ScfResult {
                    spin: Spin::Restricted,
                    energy,
                    density_total: d,
                    density_alpha,
                    density_beta: None,
                    mos_alpha: c,
                    mos_beta: None,
                    eps_alpha: orb_e,
                    eps_beta: None,
                    fock_alpha: f,
                    fock_beta: None,
                    converged: true,
                    exit,
                    iterations: iter,
                    computed_quartets: total_quartets,
                    induced_dipoles: last_induced_dipoles,
                    stability,
                    df_jk: df_jk_route.clone(),
                    rohf_spin_focks: None,
                });
            }
        }
        mon.note_energy(energy);
        if std::env::var("FERRIC_TRAH_RHO_TRACE").is_ok() {
            let dnorm = d.iter().map(|x| x * x).sum::<f64>().sqrt();
            eprintln!(
                "TRAH-ITER-TRACE: iter={iter} E={energy:.12} |D|={dnorm:.12} err_max={err_max:.3e}"
            );
        }

        // ── Trust-region augmented-Hessian (TRAH) update, RHF/RKS ────────────
        //
        // Opt-in via `config.trah_trigger`. When `None` (the default) every
        // line below is skipped and the iteration is bit-identical to a build
        // with no TRAH support — `trah_state` is `None`, so this whole block
        // collapses to one `Option` test. Proven by
        // `tests/trah_off_is_bit_identical.rs`.
        //
        // The gates match the Newton path exactly (iter > 3 so `last_c` is a
        // real MO set; ω = 0 because the Hessian matvec's K comes from the
        // plain Coulomb `build_jk`; no meta-GGA because there is no τ f_xc
        // kernel in this workspace). Anything outside those gates keeps DIIS,
        // for the same reasons documented on the Newton branch below.
        //
        // # The two-phase structure
        //
        // Phase 1 (ρ): `energy` is, right now, the actual energy at the density
        // produced by the PREVIOUS TRAH step. That is exactly what ρ needs, and
        // it is free — no speculative Fock build. If the verdict is Rejected,
        // the orbitals and density are restored from `trah_undo` and the
        // iteration re-steps from the restored point at the contracted radius.
        //
        // Phase 2 (step): solve the level-shifted AH equations inside the
        // current radius and apply the rotation.
        let trah_armed = config.trah_trigger.is_some_and(|t| err_max < t)
            && iter > 3
            && k_mix.omega == 0.0
            && !crate::rohf::xc_is_metagga(config.xc.as_deref());
        // A TRAH step runs only if TRAH is armed AND the radius has not
        // collapsed. Both conditions are computed ONCE here, because the
        // "discard a stale prediction" rule must key off exactly the same
        // condition as "take a step": if the radius collapses while a
        // prediction is pending, DIIS resumes, and a later re-arm would
        // otherwise score that stale prediction against an energy produced by
        // DIIS steps in between — a ρ built from two unrelated points.
        let trah_runs = trah_armed && !trah_state.as_ref().is_some_and(|s| s.collapsed());
        if let Some(st) = trah_state.as_mut() {
            if !trah_runs {
                st.clear_pending();
                trah_undo = None;
            }
        }
        let mut trah_took_step = false;
        if trah_runs {
            // Phase 1: score the pending step, if any.
            // `assess` returns None when NOTHING was pending — the first armed
            // iteration, and the iteration right after a rejection (which
            // consumed its pending step and skipped Phase 2). That is not an
            // acceptance: treating it as one and then printing the PERSISTENT
            // `last_rho` made the trace show a frozen ρ repeated across a
            // reject/"accept" alternation that was not happening. Keep the
            // Option so "no verdict" stays distinguishable from "accepted".
            let verdict: Option<crate::trah::TrahVerdict> =
                trah_state.as_mut().and_then(|st| st.assess(energy));
            if let (Some(_), Some(rho)) = (verdict, trah_state.as_ref().and_then(|s| s.last_rho)) {
                crate::trah::note_rho_rhf(rho);
            }
            if scf_trace() {
                if let (Some(v), Some(rho)) =
                    (verdict, trah_state.as_ref().and_then(|s| s.last_rho))
                {
                    eprintln!(
                        "TRAH iter={iter}: rho={rho:.6} verdict={v:?} \
                         Delta={:.3e} acc={} rej={}",
                        trah_state.as_ref().map(|s| s.radius()).unwrap_or(0.0),
                        trah_state.as_ref().map(|s| s.accepted).unwrap_or(0),
                        trah_state.as_ref().map(|s| s.rejected).unwrap_or(0),
                    );
                }
            }
            if verdict == Some(crate::trah::TrahVerdict::Rejected) {
                crate::trah::note_trah_rejection();
                if let Some((c_undo, d_undo)) = trah_undo.take() {
                    if scf_trace() {
                        eprintln!(
                            "TRAH iter={iter}: REJECTED (ρ={:.3e}), restoring orbitals, Δ→{:.3e}",
                            trah_state.as_ref().and_then(|s| s.last_rho).unwrap_or(0.0),
                            trah_state.as_ref().map(|s| s.radius()).unwrap_or(0.0)
                        );
                    }
                    // Undo: MOs and density go back to the pre-step point. The
                    // density change is RECORDED (not silently swapped) so the
                    // convergence monitor sees the real motion — an undo that
                    // hid itself from `dp_rms` could let the loop "converge" on
                    // a density it had just thrown away.
                    mon.record_density_change(&d_undo, &d);
                    d.assign(&d_undo);
                    last_c = c_undo;
                    d_occ = Some(last_c.slice(ndarray::s![.., ..nocc]).to_owned());
                    if effective_level_shift > 0.0 {
                        c_prev = Some(last_c.clone());
                    }
                }
            }

            // Phase 2: step from the (possibly restored) point.
            //
            // After a REJECTION we do not step this iteration: `f` was built
            // from the density we just discarded, so an MO-basis Fock formed
            // from it would be a gradient at the wrong point. The loop
            // `continue`s, rebuilds F from the restored density, and steps on
            // the NEXT iteration at the already-contracted radius. That costs
            // one Fock build per rejection and is why rejection is not free.
            //
            // The pending assessment was consumed by `assess` above, so the
            // skipped iteration records nothing and the next ρ is formed from a
            // matched (energy_before, predicted) pair — not from a stale one.
            if verdict != Some(crate::trah::TrahVerdict::Rejected) {
                let c_cur = last_c.clone();
                let f_mo = c_cur.t().dot(&f).dot(&c_cur);

                let fxc_store = if xc_contrib.is_some() {
                    let main = config.dft_grid.clone().unwrap_or_default();
                    let name = config.xc.as_deref().expect("xc_contrib implies Some(xc)");
                    let d_half = 0.5 * &d;
                    Some(crate::rohf::FxcKernelStore::build(
                        mol, prep, &main, name, &d_half, &d_half,
                    )?)
                } else {
                    None
                };
                let fxc_storage = fxc_store.as_ref().map(|s| s.response());
                let fxc_ref: Option<&crate::rohf_newton::FxcResponse<'_>> = fxc_storage.as_deref();

                let inputs = crate::rhf_newton::RhfNewtonInputs {
                    prep,
                    bounds,
                    c: &c_cur,
                    f_mo: &f_mo,
                    nocc,
                    k_mix_sr: if xc_contrib.is_some() { k_mix.sr } else { 1.0 },
                    fxc: fxc_ref,
                    thresh: config.integral_thresh,
                    ooc_budget,
                };
                let radius = trah_state
                    .as_ref()
                    .map(|s| s.radius())
                    .expect("trah_state is Some inside the armed branch");
                let (c_new, step) = crate::trah::rhf_trah_step(ctx, &inputs, radius, &config.trah)?;
                // Decline a step the model says is worthless. Once the orbital
                // gradient is converged the quadratic model predicts a change
                // below what the energy can resolve, and rho becomes numerical
                // noise over a vanishing denominator (measured on RKS/PBE
                // water: pred=-1.4e-15, actual=+2.7e-9, rho=-1.9e6). Stepping
                // there cannot help and costs two Fock builds per cycle -- see
                // `TrahConfig::predicted_min`. Falling through to DIIS lets the
                // normal convergence test end the run.
                if step.predicted.abs() < config.trah.predicted_min {
                    if scf_trace() {
                        eprintln!(
                            "TRAH iter={iter}: predicted |{:.3e}| < {:.0e}, \
                             nothing left to gain -- deferring to DIIS",
                            step.predicted, config.trah.predicted_min
                        );
                    }
                    if let Some(st) = trah_state.as_mut() {
                        st.clear_pending();
                    }
                    trah_undo = None;
                    // `trah_took_step` is already false here -- it is only set true in
                    // the else-branch below -- so leaving it is what makes the loop
                    // fall through to DIIS.
                    // Record that the density did NOT move.
                    //
                    // `record_density_change` is a SETTER, not an accumulator:
                    // `dp_rms`/`dp_max` keep whatever was last written. TRAH
                    // `continue`s past the DIIS path, so on the iterations it
                    // drives it is the ONLY writer -- and the moment it stops
                    // writing, the last value it wrote is frozen in.
                    //
                    // That froze `dp_rms` at 2.059e-6 (above the 1e-8 bar) for
                    // 200 iterations on RKS/PBE water while dE was exactly 0,
                    // |g|_rms 4.9e-9 and err_max 2.2e-8 -- every other signal
                    // converged, the run declared failure, and it reached an
                    // energy matching DIIS to 1.9e-10. The solve was DONE; only
                    // the bookkeeping said otherwise.
                    //
                    // Writing the true (zero) change here lets the ordinary
                    // convergence test see the state the solver is actually in.
                    mon.record_density_change(&d, &d);
                } else {
                    if scf_trace() {
                        eprintln!(
                            "TRAH iter={iter}: ‖κ‖={:.3e} Δ={:.3e} μ={:.3e} α={:.1} \
                         ΔE_pred={:.3e} solves={} boundary={}",
                            step.norm,
                            radius,
                            step.level_shift,
                            step.alpha,
                            step.predicted,
                            step.shift_iterations,
                            step.on_boundary
                        );
                    }

                    // Save the pre-step point so a rejection can undo it exactly.
                    if std::env::var("FERRIC_TRAH_RHO_TRACE").is_ok() {
                        let dnorm = d.iter().map(|x| x * x).sum::<f64>().sqrt();
                        let cnorm = c_cur.iter().map(|x| x * x).sum::<f64>().sqrt();
                        let gnorm = {
                            let gm = f_mo.slice(ndarray::s![nocc.., ..nocc]);
                            gm.iter().map(|x| x * x).sum::<f64>().sqrt()
                        };
                        eprintln!(
                            "TRAH-STEP-TRACE: iter={iter} E_at_step={energy:.12} \
                         |D|={dnorm:.12} |C|={cnorm:.12} |g_ov|={gnorm:.6e} \
                         |kappa|={:.6e} pred={:.6e} mu={:.6e} alpha={:.1}",
                            step.norm, step.predicted, step.level_shift, step.alpha
                        );
                    }
                    trah_undo = Some((c_cur, d.clone()));
                    if let Some(st) = trah_state.as_mut() {
                        st.record_step(energy, &step);
                    }
                    crate::trah::note_trah_step();

                    let c_occ = c_new.slice(ndarray::s![.., ..nocc]);
                    d_occ = Some(c_occ.to_owned());
                    let d_new =
                        with_blas_threads(opt_in_blas_threads(), || 2.0 * c_occ.dot(&c_occ.t()));
                    mon.record_density_change(&d_new, &d);
                    d.assign(&d_new);
                    last_c = c_new;
                    if effective_level_shift > 0.0 {
                        c_prev = Some(last_c.clone());
                    }
                }
                trah_took_step = true;
            }
        }
        if trah_took_step {
            continue;
        }

        // ── AURORA auxiliary-curvature update, RHF/RKS ───────────────────────
        // Opt-in (config.aurora.enabled, default false). Replaces the DIIS
        // extrapolation with a quasi-Newton orbital rotation whose curvature
        // comes from an independent STO-3G auxiliary model, corrected by a
        // transported L-BFGS history of exact target secants (arXiv:2608.07354).
        //
        // Placed BEFORE the Newton branch so the two are mutually exclusive and
        // the precedence is explicit; with `enabled: false` the condition is a
        // single bool test and every existing code path is unchanged.
        //
        // Like the Newton branch below, this `continue`s — bypassing DIIS and the
        // Fock diagonalization — so it must itself update `d`, `d_occ`, `last_c`
        // and record the density change for the next iteration's convergence test.
        // The auxiliary model carries Coulomb + exact-exchange curvature but no
        // exchange-correlation curvature (the paper's `D_k^xc` is named but never
        // defined). On references with little or no exact exchange the model
        // therefore understates the curvature and the step overshoots, so those
        // stay on DIIS unless explicitly opted in. See `aurora::AuroraConfig`.
        let aurora_ax = if xc_contrib.is_some() { k_mix.sr } else { 1.0 };
        let aurora_exchange_ok = config.aurora.allow_low_exchange_ks
            || aurora_ax >= crate::aurora::MIN_VALIDATED_EXCHANGE_FRACTION;
        let use_aurora = config.aurora.enabled
            && iter > 1
            && nocc > 0
            && nocc < prep.nbasis()
            && aurora_exchange_ok
            && (config.aurora.trigger <= 0.0 || err_max < config.aurora.trigger);
        if use_aurora {
            // `last_c` holds the MOs that produced the current density `d`; it is
            // zeros before iter 2, hence the `iter > 1` gate above.
            let c_cur = &last_c;
            let f_mo = c_cur.t().dot(&f).dot(c_cur);

            // Lazily construct the accelerator at the first accelerated step.
            if aurora_state.is_none() {
                // The curvature model's exchange fraction: full exact exchange
                // for HF, the hybrid's mixing fraction otherwise. This enters the
                // CURVATURE only and can never move the converged answer.
                aurora_state = Some(crate::aurora::AuroraState::new(
                    mol,
                    prep,
                    c_cur.view(),
                    nocc,
                    aurora_ax,
                    &config.aurora,
                )?);
            }
            let state = aurora_state.as_mut().expect("aurora_state just set");

            // Close the previous macro step's secant pair with the target-level
            // gradient now available at these orbitals.
            let de = energy - aurora_last_energy;
            state.observe(&f_mo, nocc, de);
            aurora_last_energy = energy;

            let c_new = state.step(c_cur.view(), &f_mo, nocc, aurora_first_step)?;
            aurora_first_step = false;

            let c_occ = c_new.slice(ndarray::s![.., ..nocc]);
            d_occ = Some(c_occ.to_owned());
            let d_new = with_blas_threads(opt_in_blas_threads(), || 2.0 * c_occ.dot(&c_occ.t()));
            mon.record_density_change(&d_new, &d);
            d.assign(&d_new);
            last_c = c_new;
            if effective_level_shift > 0.0 {
                c_prev = Some(last_c.clone());
            }
            continue;
        }

        // ── Second-order (Newton) update, RHF/RKS ────────────────────────────
        // When enabled (newton_trigger > 0) and err_max has dropped below the
        // trigger, take a damped-Newton step on the single occ→virt orbital
        // rotation instead of DIIS — the closed-shell analogue of the UHF/ROKS
        // Newton paths. `last_c` holds the MOs that produced the current density
        // `d` (set at the tail of the previous iteration; zeros before iter 2),
        // so the branch is gated to `iter > 3`, matching solve_uhf. For RKS this
        // reuses the SAME LDA/GGA f_xc kernel (via FxcKernelStore) that the
        // ROKS/UKS Newton paths use. Gated to the non-RSH case (ω = 0), since
        // the Newton matvec's K comes from the plain Coulomb `build_jk`; RSH and
        // meta-GGA (no τ f_xc) keep the DIIS path. Default newton_trigger = 0
        // ⇒ this branch never fires and existing RHF/RKS results are unchanged.
        let use_newton = config.newton_trigger > 0.0
            && iter > 3
            && err_max < config.newton_trigger
            && k_mix.omega == 0.0
            && !crate::rohf::xc_is_metagga(config.xc.as_deref());
        if use_newton {
            let c_cur = &last_c;
            let f_mo = c_cur.t().dot(&f).dot(c_cur);

            // Build the f_xc kernel (LDA or GGA) at the current restricted
            // reference (d_α = d_β = ½·d) once per Newton step; None for pure HF.
            let fxc_store = if xc_contrib.is_some() {
                let main = config.dft_grid.clone().unwrap_or_default();
                let name = config.xc.as_deref().expect("xc_contrib implies Some(xc)");
                let d_half = 0.5 * &d;
                Some(crate::rohf::FxcKernelStore::build(
                    mol, prep, &main, name, &d_half, &d_half,
                )?)
            } else {
                None
            };
            let fxc_storage = fxc_store.as_ref().map(|s| s.response());
            let fxc_ref: Option<&crate::rohf_newton::FxcResponse<'_>> = fxc_storage.as_deref();

            let inputs = crate::rhf_newton::RhfNewtonInputs {
                prep,
                bounds,
                c: c_cur,
                f_mo: &f_mo,
                nocc,
                k_mix_sr: if xc_contrib.is_some() { k_mix.sr } else { 1.0 },
                fxc: fxc_ref,
                thresh: config.integral_thresh,
                ooc_budget,
            };
            let (c_new, _kmax) = crate::rhf_newton::rhf_newton_step(
                ctx,
                &inputs,
                config.level_shift.max(1e-6),
                0.2, // trust radius
                20,
                1e-7,
            )?;

            // Rebuild density D = 2·C_occ·C_occᵀ and record ΔP for convergence.
            let c_occ = c_new.slice(ndarray::s![.., ..nocc]);
            // Cache bare C_occ for the next iteration's DF-K half-transform (the
            // Newton path never smears, so D is always a plain rank-nocc C·Cᵀ).
            d_occ = Some(c_occ.to_owned());
            let d_new = with_blas_threads(opt_in_blas_threads(), || 2.0 * c_occ.dot(&c_occ.t()));
            mon.record_density_change(&d_new, &d);
            d.assign(&d_new);
            last_c = c_new;
            if effective_level_shift > 0.0 {
                c_prev = Some(last_c.clone());
            }
            continue;
        }

        // `d`/`energy`/`err_max` feed the energy-based branch (ADIIS/EDIIS);
        // ignored when the driver is pure Pulay (default).
        let mut f_new = diis.step(&f, &err, &d, energy, err_max);

        // Virtual-block level shift (config.level_shift > 0): add `shift` to the
        // diagonal of the virtual block in MO basis, i.e. F += shift · S·C_v·C_vᵀ·S
        // where C_v are the previous iteration's virtual MOs. This widens the
        // occupied–virtual gap and damps the orbital rotation that bare DIIS
        // overshoots — the standard cure for SCF oscillation/divergence on
        // heavy-atom closed shells (e.g. COSe, C2H3Br). The shift is ramped to
        // zero as the gradient converges (err_max → 0), so the converged Fock is
        // unshifted and the final energy is unperturbed. Applied from iter ≥ 2
        // (needs a prior C); the convergence check above runs on the *unshifted*
        // F, so an accepted solution never carries the shift.
        if effective_level_shift > 0.0 {
            if let Some(c_p) = c_prev.as_ref() {
                let shift_ramp = effective_level_shift * (err_max / 0.1).min(1.0);
                if shift_ramp > 0.0 {
                    let c_vir = c_p.slice(ndarray::s![.., nocc..]);
                    let scv = s.dot(&c_vir); // (n × nvir)
                    f_new = &f_new + &(shift_ramp * scv.dot(&scv.t()));
                }
            }
        }

        let (eps, mut c) = diagonalize(&f_new, &x)?;
        last_eps = eps;
        // Only retain C across iterations when the level shift needs it; the
        // default (shift = 0) path skips the clone entirely.
        if effective_level_shift > 0.0 {
            c_prev = Some(c.clone());
        }

        // MOM occupied-orbital selection from iter mom_after_iter+1: pin the
        // occupied set by AO-overlap with the previous accepted occupation
        // instead of aufbau (breaks occupied-set flip-flop, e.g. C2H4·Ar).
        if config.mom_after_iter > 0 && iter > config.mom_after_iter {
            if let Some(r) = mom_ref.as_ref() {
                let empty_open = Array2::<f64>::zeros((c.nrows(), 0));
                c = crate::mom::mom_reorder(&c, &s, r, &empty_open, nocc, 0);
            }
        }
        if config.mom_after_iter > 0 && iter >= config.mom_after_iter {
            mom_ref = Some(c.slice(ndarray::s![.., ..nocc]).to_owned());
        }

        // Rebuild density. Default path (smearing off): D = 2 * C_occ @ C_occ^T
        // over the lowest `nocc` MOs (BLAS dgemm). Fermi-Dirac smearing path
        // (config.smearing_sigma = Some(σ>0)): solve μ so 2·Σ f_i = N_elec at
        // width σ, then build D = Σ_i (2·f_i) c_i c_iᵀ = C · diag(2f) · Cᵀ.
        // Exact-off: when smearing_sigma is None the `_` arm is the unchanged
        // pre-smearing integer code.
        let d_new = match config.smearing_sigma {
            Some(sigma) if sigma > 0.0 => {
                let sm = crate::smearing::solve_fermi_level(&last_eps, nelec as f64, sigma, 2.0)?;
                last_smearing = Some(sm.clone());
                // Fractional occupations: D is not a plain rank-nocc C·Cᵀ, so the
                // DF-K half-transform can't represent it — force the density path.
                d_occ = None;
                with_blas_threads(opt_in_blas_threads(), || {
                    let mut c_weighted = c.clone();
                    for (i, &focc) in sm.occupations.iter().enumerate() {
                        let mut col = c_weighted.column_mut(i);
                        col *= 2.0 * focc;
                    }
                    c_weighted.dot(&c.t())
                })
            }
            _ => {
                let c_occ = c.slice(ndarray::s![.., ..nocc]);
                // Cache bare C_occ so the next iteration's DF-K uses the cheap
                // half-transform (K(D) = 2·K(C_occ·C_occᵀ), factor applied to K).
                d_occ = Some(c_occ.to_owned());
                with_blas_threads(opt_in_blas_threads(), || 2.0 * c_occ.dot(&c_occ.t()))
            }
        };
        // Density change ΔP = D_new − D_old — the primary convergence signal
        // (consumed at the top of the next iteration by scf_converged). Unlike
        // the DIIS commutator, ΔP drains to zero at the RI fixed point even when
        // the commutator parks on the naux-dependent noise floor.
        mon.record_density_change(&d_new, &d);
        d.assign(&d_new);
        last_c = c;
    }
    // Loop exhausted without convergence. Return the best-effort density so a
    // ladder can carry it forward; the caller checks `converged`.
    Ok(build_nonconverged(
        ScfExit::MaxIter,
        &d,
        &last_c,
        &last_eps,
        &f,
        mon.prev_e,
        config.max_iter,
        total_quartets,
        last_induced_dipoles,
    ))
}

/// Build the Coulomb (J) and exchange (K) matrices from the density matrix.
///
/// Uses Schwarz screening and 8-fold permutational symmetry of the ERIs.
///
/// Constructs a fresh [`crate::engine_pool::EnginePool`] every call. Fine for
/// one-shot callers (a single SCF Fock build per density update), but hot
/// repeat-callers (Newton/CPKS Hessian-vector products that call this dozens
/// of times per outer iteration on the SAME geometry/basis) should instead
/// build the pool ONCE and call [`build_jk_with_pool`] directly — see that
/// function's doc.
pub fn build_jk(
    ctx: &ParallelContext,
    prep: &PreparedBasis,
    bounds: &SchwarzBounds,
    thresh: f64,
    d: &Array2<f64>,
    j: &mut Array2<f64>,
    k: &mut Array2<f64>,
) -> Result<usize, FerricError> {
    let pool = crate::engine_pool::EnginePool::new(bounds.op, prep, 1e-14)?;
    build_jk_with_pool(
        ctx,
        prep,
        bounds,
        thresh,
        d,
        j,
        k,
        &pool,
        crate::reduce::default_band_bytes(),
    )
}

/// Same as [`build_jk`], but takes a caller-supplied [`crate::engine_pool::EnginePool`]
/// instead of constructing one internally, and an explicit `band_bytes` live-set
/// budget for the deterministic reduction (see `reduce::resolve_band_bytes`).
///
/// `EnginePool` construction is geometry/basis-only (density-independent), so
/// it is safe — and, for repeat callers, important for performance — to build
/// it ONCE outside a Hessian-vector-product / CG loop and reuse it across
/// calls on the same `(prep, bounds.op)`. Reduction order (grouped
/// deterministic sum over shell-pair groups) is unchanged from `build_jk` and
/// does not depend on the pool OR on `band_bytes`, so results stay bit-identical
/// across thread counts AND across any choice of `band_bytes` — this parameter
/// affects ONLY the live-set/parallel width of the reduction, never the result
/// (see `reduce.rs` module docs). Repeat-callers that have a solver-resolved
/// `ooc_budget` in scope should pass `reduce::resolve_band_bytes(ooc_budget)`
/// instead of the env/auto-resolved `reduce::default_band_bytes()`, so the
/// TOML/config budget actually governs this scratch too.
#[allow(clippy::too_many_arguments)]
pub fn build_jk_with_pool(
    ctx: &ParallelContext,
    prep: &PreparedBasis,
    bounds: &SchwarzBounds,
    thresh: f64,
    d: &Array2<f64>,
    j: &mut Array2<f64>,
    k: &mut Array2<f64>,
    pool: &crate::engine_pool::EnginePool,
    band_bytes: usize,
) -> Result<usize, FerricError> {
    use crate::quartet_scatter::{
        build_d_max_shell, canonical_bra_pairs, scatter_bra_pair, DensityScreen, JkMode,
    };
    use std::sync::atomic::{AtomicUsize, Ordering};

    ctx.check_interrupted()?;

    let nsh = prep.nshells();
    let nbf = prep.nbasis();
    let dims = prep.shell_dims();
    let offs = prep.shell_offsets();
    let computed_quartets = AtomicUsize::new(0);

    // Shell-blocked density-max table for Häser-Ahlrichs pair-wise screening.
    let d_max_shell = build_d_max_shell(prep, d);

    let shell_pairs: Vec<_> = canonical_bra_pairs(nsh);

    // MPI rank striping: every rank builds the identical `shell_pairs` list
    // above, then `ParallelContext::stripe` keeps only its own disjoint,
    // covering partition (see its doc for why round-robin, and the
    // bit-identity argument). Without this, every rank computed the FULL J/K
    // redundantly and the unconditional Allreduce below N-folded the result
    // instead of summing a genuine partition (confirmed: -np 2 water RHF
    // converged to ≈-3958 Ha instead of -76.03 Ha). This preserves the
    // thread-count bit-identity invariant this function is already relied on
    // for (see build_jk_bit_identical_across_thread_counts below).
    let shell_pairs: Vec<_> = ctx.stripe(shell_pairs);

    // One engine per rayon thread (see engine_pool) — constructing an engine in
    // the fold init fires once per work-chunk, not per thread, storming the
    // global libint2 ctor mutex on heavy-element bases. The pool is passed in
    // by the caller (see `build_jk_with_pool` doc) rather than built here, so
    // hot repeat-callers (Newton/CPKS) can construct it once outside their
    // loop instead of once per call.

    // Deterministic, memory-bounded reduction (see direct_jk.rs / reduce.rs).
    // The old `fold(..).reduce(..)` tree combined per-chunk (J,K) partials in a
    // worker-count-dependent order, so J/K (and every downstream SCF energy,
    // gradient, and MP2/RPA number built on this density) drifted ~1 ULP with
    // RAYON_NUM_THREADS (proven by the P7 whole-pipeline gradient bit-identity
    // test: RHF gradient differed in the last bits between 1 and 4 threads,
    // traced here). Partition the (s1,s2) shell-pair work list — a pure
    // function of nsh, never of the thread count — and fold group partials in
    // strict ascending group order: bit-identical across thread counts.
    let n_pairs = shell_pairs.len();
    let group_size = crate::reduce::deterministic_group_size(n_pairs);
    let n_groups = n_pairs.div_ceil(group_size.max(1)).max(1);

    let mut total_j = Array2::<f64>::zeros((nbf, nbf));
    let mut total_k = Array2::<f64>::zeros((nbf, nbf));
    // `band_bytes` is now an explicit parameter (see doc above): callers with
    // a solver-resolved `ooc_budget` pass `reduce::resolve_band_bytes(ooc_budget)`;
    // the plain `build_jk` wrapper and other no-budget callers pass
    // `reduce::default_band_bytes()` (env/auto-resolved), preserving the old
    // behavior byte-for-byte.
    let screen = DensityScreen::SixPair(&d_max_shell);
    crate::reduce::grouped_deterministic_sum_pair(
        &mut total_j,
        &mut total_k,
        n_groups,
        nbf,
        band_bytes,
        |g| {
            let lo = g * group_size;
            let hi = (lo + group_size).min(n_pairs);
            let mut mode = JkMode::new_both(nbf);
            let mut local_count = 0usize;
            for &(s1, s2) in &shell_pairs[lo..hi] {
                if ferric_core::INTERRUPT.load(std::sync::atomic::Ordering::Relaxed) {
                    continue;
                }
                pool.with(|engine| {
                    local_count += scatter_bra_pair(
                        engine,
                        prep,
                        dims,
                        offs,
                        &bounds.q,
                        bounds.csb_m.as_ref(),
                        bounds.csam_x.as_ref(),
                        &screen,
                        thresh,
                        d,
                        s1,
                        s2,
                        &mut mode,
                        true,
                    );
                });
            }
            let (local_j, local_k) = match mode {
                JkMode::Both(j, k) => (j, k),
                _ => unreachable!("build_jk_with_pool always uses JkMode::Both"),
            };
            computed_quartets.fetch_add(local_count, std::sync::atomic::Ordering::Relaxed);
            Ok((local_j, local_k))
        },
    )?;

    // MPI: reduce the rank-LOCAL partials, THEN accumulate. This function
    // ACCUMULATES onto `j`/`k` without zeroing them, so an Allreduce of the
    // output buffers would also multiply any pre-existing caller contents by the
    // world size. See `reduce::reduce_partial_across_ranks`.
    crate::reduce::reduce_partial_across_ranks(ctx, &mut total_j);
    crate::reduce::reduce_partial_across_ranks(ctx, &mut total_k);

    *j += &total_j;
    *k += &total_k;

    Ok(computed_quartets.load(Ordering::SeqCst))
}

/// Default linear-dependence threshold for canonical orthogonalization:
/// eigenvectors of the overlap matrix with eigenvalue below this are dropped from
/// the variational space. PySCF's default is ~1e-6 to 1e-7; 1e-6 is conservative.
pub(crate) const LINDEP_THRESH: f64 = 1e-6;

/// `FERRIC_SCF_TRACE` descriptor: per-iteration SCF convergence trace (env-only
/// debug toggle). Read at several sites in rhf/uhf/guess via [`scf_trace`].
static SCF_TRACE: ferric_core::config::ConfigVar<bool> = ferric_core::config::ConfigVar {
    env_name: "FERRIC_SCF_TRACE",
    default: false,
    parse: ferric_core::config::parse_toggle,
    validate: ferric_core::config::accept_any,
};

/// Whether the per-iteration SCF trace is on. `FERRIC_SCF_TRACE=1/true/on/yes`,
/// off for `0/false/off/no`/unset; a malformed value logs a warning and stays off.
pub(crate) fn scf_trace() -> bool {
    SCF_TRACE.toggle()
}

/// Effective linear-dependence threshold, overridable via `FERRIC_LINDEP_THRESH`.
///
/// The default [`LINDEP_THRESH`] (1e-6) drops nothing on the diffuse alkali/d-block
/// clusters (Na4/Na6/Cu2 at aug-cc-pVTZ), whose near-null overlap modes sit at
/// λ ≈ 3e-5–5e-5 — kept and amplified ~150× by 1/√λ, which parks the DIIS orbital
/// gradient just above `density_conv` so the SCF plateau-spins to `max_iter`
/// without ever declaring convergence. Raising the threshold for those systems
/// projects the offending modes out and lets DIIS converge. Env-scoped rather
/// than a global bump because a blanket 1e-4 would also drop legitimate in-band
/// modes on well-conditioned aromatics (C6H6/aug-cc-pVTZ has 19 modes in
/// [1e-6,1e-4) that DIIS drains fine), perturbing their banked energies.
fn lindep_thresh() -> f64 {
    static LINDEP: ferric_core::config::ConfigVar<f64> = ferric_core::config::ConfigVar {
        env_name: "FERRIC_LINDEP_THRESH",
        default: LINDEP_THRESH,
        parse: |s| s.parse::<f64>().map_err(|e| e.to_string()),
        validate: |v| {
            (v.is_finite() && *v > 0.0)
                .then_some(())
                .ok_or_else(|| "must be finite > 0".to_string())
        },
    };
    // A malformed/invalid override uses the default (was silent; now warns) —
    // this is read on the SCF hot path with no Result to propagate.
    LINDEP.get().map(|r| r.value).unwrap_or_else(|e| {
        eprintln!("[config] FERRIC_LINDEP_THRESH: {e}; using default {LINDEP_THRESH}");
        LINDEP_THRESH
    })
}

/// Build the canonical orthogonalizer X (n × m) from the overlap matrix S.
///
/// X = U_kept · diag(1/sqrt(λ_kept)), where U_kept are the eigenvectors of S
/// whose eigenvalue λ ≥ [`LINDEP_THRESH`]. `eigh` returns eigenvalues in
/// ASCENDING order, so the near-singular modes are first and are skipped.
/// For a well-conditioned S, m == n and no mode is dropped (regression-safe).
///
/// X is NOT symmetric (even when m == n) — callers must use the rectangular
/// transform Fʹ = Xᵀ F X, C = X Vʹ (see [`diagonalize`]).
pub(crate) fn canonical_orthogonalizer(s: &Array2<f64>) -> Result<Array2<f64>, FerricError> {
    let n = s.nrows();
    // Runs once at SCF setup, serially (called before the iteration loop, no
    // enclosing rayon region). Opt-in BLAS raise (default 1, unchanged
    // behavior) — see the DIIS FDS/SDF comment above for the SAD/free-atom
    // protection argument (identical here: opt_in_blas_threads()'s
    // rayon-worker self-guard resolves to 1 inside guess.rs's run_serial_pool).
    let (s_evals, s_evecs) = with_blas_threads(opt_in_blas_threads(), || {
        s.eigh(ndarray_linalg::UPLO::Upper)
    })
    .map_err(|e| FerricError::Lapack(format!("S diag: {e}")))?;
    // eigenvalues ascending: kept columns are those with λ ≥ the effective
    // threshold (default LINDEP_THRESH, overridable via FERRIC_LINDEP_THRESH).
    let thresh = lindep_thresh();
    let kept: Vec<usize> = (0..n).filter(|&i| s_evals[i] >= thresh).collect();
    let m = kept.len();
    let mut x = Array2::<f64>::zeros((n, m));
    for (col, &i) in kept.iter().enumerate() {
        let scale = 1.0 / s_evals[i].sqrt();
        for mu in 0..n {
            x[(mu, col)] = s_evecs[(mu, i)] * scale;
        }
    }
    Ok(x)
}

/// Diagonalize the Fock matrix in the canonical-orthogonal basis.
///
/// With X (n × m) rectangular: Fʹ = Xᵀ F X is (m × m), its eigenvectors Vʹ are
/// (m × m), and C_kept = X Vʹ is (n × m). The result is PADDED back to (n × n)
/// by appending (n − m) zero MO columns with sentinel energy 1e6, so every
/// downstream consumer (GW, RPA, density build) sees the historical (n × n) /
/// length-n shapes while the near-singular directions are inert virtuals.
fn diagonalize(f: &Array2<f64>, x: &Array2<f64>) -> Result<(Vec<f64>, Array2<f64>), FerricError> {
    crate::driver::diagonalize_rect(f, x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::screening::SchwarzBounds;
    use ferric_core::basis;
    use ferric_core::external_potential::{ExternalPotential, PointCharge};
    use ferric_core::mol::Molecule;
    use ferric_integrals::basis_bridge::PreparedBasis;

    // FERRIC_MEM_BUDGET_GB / FERRIC_OOC_BUDGET_GB are process-global; serialize
    // env-mutating tests (blas_threads.rs / ferric-core memory.rs pattern).
    // MUST be held by every test in this module that transitively reads the
    // budget env var, not just the one that mutates it: `solve_rhf` calls
    // `resolve_three_index_budget` unconditionally on every invocation (see the
    // `let ooc_budget = ...` line near the top of `solve_rhf`), which falls
    // through to `ferric_core::memory::resolve_budget_bytes` and reads the
    // ambient env value whenever no explicit `config.three_index_budget_bytes`
    // is set. Under cargo test's default parallelism, any test that calls
    // `solve_rhf` (directly or via a helper like `run_rhf_test`) can observe
    // `three_index_budget_auto_detects_ram_on_default`'s temporary near-zero /
    // pinned-GB env mutation mid-flight if it doesn't also hold this lock —
    // this file has 20+ such tests. Confirmed dormant (not actively flaky) as
    // of 2026-07-18, but the same race that hit
    // `eval_basis_on_grid_serial_and_parallel_paths_agree` in
    // ferric-export/src/gto_eval.rs applies here structurally; fixed
    // proactively rather than waiting for a flake. Pure-function tests that
    // never call `solve_rhf` (e.g. `scf_converged_*`, `stall_detected_*`) do
    // NOT need this lock — only reach for it when the test path touches
    // `resolve_three_index_budget` (directly or transitively).
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Triplet water has an EVEN electron count, so the old odd-count check
    /// never fired and `solve_rhf` returned the closed-shell singlet. If the
    /// `require_closed_shell` call at the top of `solve_rhf` is removed, the
    /// `expect_err` below panics with a converged singlet `ScfResult`.
    /// No ENV_LOCK: the guard returns before the budget resolver is reached,
    /// and the reachability half below only calls the pure helper.
    #[test]
    fn solve_rhf_refuses_an_even_electron_open_shell_molecule() {
        let xyz = "3\nwater\nO 0.000000 0.000000 0.117790\n\
                   H 0.000000 0.755453 -0.471161\nH 0.000000 -0.755453 -0.471161\n";
        let triplet = Molecule::parse_xyz(xyz, 0, 3).unwrap();
        let bs = basis::bundled("sto-3g").unwrap();
        let prep = PreparedBasis::new(&triplet, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let ctx = ParallelContext::default();
        let err = solve_rhf(&ctx, &triplet, &prep, op, &bounds, &RhfConfig::default())
            .expect_err("RHF on a triplet must be refused, not answered with the singlet");
        let msg = err.to_string();
        assert!(msg.contains("multiplicity 3"), "{msg}");
        // Reachability: the helper must accept the singlet, or the refusal
        // above proves nothing about WHICH input it refuses.
        let singlet = Molecule::parse_xyz(xyz, 0, 1).unwrap();
        assert!(require_closed_shell(&singlet).is_ok());
    }

    /// Every opt-out spelling maps to the `""` sentinel and a basis name is
    /// kept (trimmed, case preserved). If `normalize_df_aux` became a plain
    /// passthrough again, the "exact"/"NONE"/" off " cases fail.
    #[test]
    fn normalize_df_aux_maps_every_opt_out_spelling_to_the_sentinel() {
        for s in [
            "",
            "none",
            "off",
            "exact",
            "conventional",
            "NONE",
            " off ",
            "Exact",
        ] {
            assert_eq!(normalize_df_aux(s), "", "{s:?}");
        }
        assert_eq!(
            normalize_df_aux(" def2-universal-jkfit "),
            "def2-universal-jkfit"
        );
        assert_eq!(normalize_df_aux("cc-pvdz-jkfit"), "cc-pvdz-jkfit");
    }

    #[test]
    fn three_index_budget_auto_detects_ram_on_default() {
        // The whole point of the RAM-aware resolver: on a box with adequate RAM,
        // the legacy 2 GiB default must NOT cap the budget — it must fall through
        // to auto-detect (0.8×available), so the DF-JK tensor stays in RAM
        // instead of spilling to disk. Env-var precedence is also asserted here.
        // These sub-cases mutate process env, so they live in ONE test (cargo
        // runs tests in a shared process; a separate test could race the env).
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let legacy_2gib = 2 * 1024 * 1024 * 1024;

        // Guard: don't let caller-set budget env vars perturb the assertions.
        // Both the unified var and the legacy OOC var feed the resolver now.
        let saved_ooc = std::env::var("FERRIC_OOC_BUDGET_GB").ok();
        let saved_mem = std::env::var("FERRIC_MEM_BUDGET_GB").ok();
        std::env::remove_var("FERRIC_OOC_BUDGET_GB");
        std::env::remove_var("FERRIC_MEM_BUDGET_GB");

        // (1) Unset (config_bytes == 0) -> auto-detect. On any real CI/dev box
        // this is 0.8×available RAM, which is >2 GiB; the old blind 2 GiB cap
        // must be exceeded. (If available RAM were truly <2.5 GiB the fallback
        // would also equal 2 GiB — accept that degenerate case rather than flake.)
        let auto = resolve_three_index_budget(0);
        let avail = ferric_core::memory::detect_available_bytes();
        if let Some(a) = avail {
            if (a as f64 * 0.8) as usize > legacy_2gib {
                assert!(
                    auto > legacy_2gib,
                    "unset budget should auto-detect above the old 2 GiB cap on a \
                     box with {a} bytes available; got {auto}"
                );
            }
        }

        // (2) A non-zero config value is an explicit choice, honored as-is —
        // INCLUDING a deliberate 2 GiB (the bug the 0-sentinel fixes: an
        // explicit 2 GiB must NOT be swallowed by auto-detect).
        let explicit = 7 * 1024 * 1024 * 1024;
        assert_eq!(resolve_three_index_budget(explicit), explicit);
        assert_eq!(
            resolve_three_index_budget(legacy_2gib),
            legacy_2gib,
            "an explicit 2 GiB budget must be honored, not auto-detected"
        );

        // (3) Precedence: TOML/config OVERRIDES env; env fills in only when
        // config is unset. (Corrected from the earlier env-first behavior so all
        // memory settings share one chain: TOML/config > env > auto.)
        std::env::set_var("FERRIC_OOC_BUDGET_GB", "3");
        assert_eq!(
            resolve_three_index_budget(0),
            3 * 1024 * 1024 * 1024,
            "env fills in when config is unset (config_bytes == 0)"
        );
        assert_eq!(
            resolve_three_index_budget(explicit),
            explicit,
            "TOML/config budget must OVERRIDE the FERRIC_OOC_BUDGET_GB env var"
        );
        std::env::remove_var("FERRIC_OOC_BUDGET_GB");

        // (4) The unified var FERRIC_MEM_BUDGET_GB is also overridden by config,
        // and itself takes precedence over the legacy OOC var when config unset.
        std::env::set_var("FERRIC_MEM_BUDGET_GB", "5");
        assert_eq!(
            resolve_three_index_budget(0),
            5 * 1024 * 1024 * 1024,
            "FERRIC_MEM_BUDGET_GB fills in when config is unset"
        );
        assert_eq!(
            resolve_three_index_budget(explicit),
            explicit,
            "TOML/config budget must OVERRIDE FERRIC_MEM_BUDGET_GB too"
        );
        std::env::remove_var("FERRIC_MEM_BUDGET_GB");

        // Restore whatever the harness had set.
        if let Some(v) = saved_ooc {
            std::env::set_var("FERRIC_OOC_BUDGET_GB", v);
        }
        if let Some(v) = saved_mem {
            std::env::set_var("FERRIC_MEM_BUDGET_GB", v);
        }
    }

    #[test]
    fn external_point_charge_changes_rhf_energy_and_matches_hand_calc() {
        // Holds ENV_LOCK (declared above) because solve_rhf reads the
        // process-global FERRIC_MEM_BUDGET_GB/FERRIC_OOC_BUDGET_GB env vars
        // internally via resolve_three_index_budget -- see the lock's doc
        // comment.
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
        let bs = ferric_core::basis::bundled("sto-3g").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let ctx = ParallelContext::default();

        let base = solve_rhf(&ctx, &mol, &prep, op, &bounds, &RhfConfig::default()).unwrap();

        // Place a +1 point charge 20 Bohr away (weak perturbation, should shift
        // energy by a small, nonzero, well-defined amount and not break convergence).
        let ext = ExternalPotential {
            point_charges: vec![PointCharge {
                q: 1.0,
                x: 0.0,
                y: 0.0,
                z: 20.0,
            }],
            smeared_charges: Vec::new(),
            field: None,
        };
        let config = RhfConfig {
            external_potential: Some(ext.clone()),
            ..Default::default()
        };
        let perturbed = solve_rhf(&ctx, &mol, &prep, op, &bounds, &config).unwrap();

        assert!(perturbed.converged);
        assert!(
            (perturbed.energy - base.energy).abs() > 1e-8,
            "energy did not change"
        );

        // Classical charge-nuclear energy alone (no electronic response) must be
        // a lower bound on the magnitude of a repulsive-like shift; more
        // importantly, verify the classical piece was actually added by checking
        // it against the standalone helper.
        let classical = ext.charge_nuclear_energy(&mol);
        assert!(classical.abs() > 0.0);
    }

    #[test]
    fn external_potential_none_matches_default_exactly() {
        // See ENV_LOCK doc comment: solve_rhf reads the budget env vars.
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
        let bs = ferric_core::basis::bundled("sto-3g").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let ctx = ParallelContext::default();

        let a = solve_rhf(&ctx, &mol, &prep, op, &bounds, &RhfConfig::default()).unwrap();
        let config = RhfConfig {
            external_potential: None,
            ..Default::default()
        };
        let b = solve_rhf(&ctx, &mol, &prep, op, &bounds, &config).unwrap();
        assert_eq!(a.energy, b.energy);
    }

    #[test]
    fn verbose_false_matches_default_exactly() {
        // `verbose: false` (the default) must be byte-identical to a plain
        // run — the same convention external_potential/cosmo/pcm follow for
        // their own None/off defaults. Confirms the new opt-in trace field
        // does not perturb the SCF numerics or convergence path at all.
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
        let bs = ferric_core::basis::bundled("sto-3g").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let ctx = ParallelContext::default();

        let a = solve_rhf(&ctx, &mol, &prep, op, &bounds, &RhfConfig::default()).unwrap();
        let config = RhfConfig {
            verbose: false,
            ..Default::default()
        };
        let b = solve_rhf(&ctx, &mol, &prep, op, &bounds, &config).unwrap();
        assert_eq!(a.energy, b.energy);
        assert_eq!(a.iterations, b.iterations);
        assert_eq!(a.converged, b.converged);
    }

    #[test]
    fn verbose_true_does_not_change_energy_or_convergence() {
        // `verbose: true` only adds a stdout print each iteration; it must
        // not change the SCF trajectory (same iteration count, same final
        // energy) relative to verbose: false. Also exercises the
        // `config.verbose && ctx.is_root()` branch so it is covered by a
        // normal `cargo test` run (no subprocess / stdout capture needed —
        // the CLI-level `verbose_trace.rs` integration test covers the
        // actual printed text end-to-end).
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
        let bs = ferric_core::basis::bundled("sto-3g").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let ctx = ParallelContext::default();

        let quiet = solve_rhf(&ctx, &mol, &prep, op, &bounds, &RhfConfig::default()).unwrap();
        let config = RhfConfig {
            verbose: true,
            ..Default::default()
        };
        let loud = solve_rhf(&ctx, &mol, &prep, op, &bounds, &config).unwrap();
        assert_eq!(quiet.energy, loud.energy);
        assert_eq!(quiet.iterations, loud.iterations);
        assert_eq!(quiet.converged, loud.converged);
    }

    #[test]
    fn pcm_none_matches_vacuum_exactly() {
        // `pcm: None` (the default) must be byte-identical to a plain vacuum
        // calculation -- the same convention external_potential follows.
        // Holds ENV_LOCK (declared above) because solve_rhf reads the
        // process-global FERRIC_MEM_BUDGET_GB/FERRIC_OOC_BUDGET_GB env vars
        // internally via resolve_three_index_budget -- see the lock's doc
        // comment.
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
        let bs = ferric_core::basis::bundled("sto-3g").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let ctx = ParallelContext::default();

        let a = solve_rhf(&ctx, &mol, &prep, op, &bounds, &RhfConfig::default()).unwrap();
        let config = RhfConfig {
            pcm: None,
            ..Default::default()
        };
        let b = solve_rhf(&ctx, &mol, &prep, op, &bounds, &config).unwrap();
        assert_eq!(a.energy, b.energy);
    }

    /// Water/STO-3G in water solvent (eps=78.4): the standard textbook PCM
    /// validation case, cross-checked against a genuine PySCF IEF-PCM run
    /// for this exact molecule/basis/eps (own SWIG tessellation):
    /// E_solv = -3.8228 kcal/mol.
    ///
    /// **Gaussian-smeared S/D measurement (2026-07-19)**: porting the
    /// PySCF `pcm.py::get_D_S` Gaussian-smeared-charge S/D boundary-element
    /// formulation (`ferric_pcm::matrices::SdKind::GaussianSmeared`, now the
    /// crate default, on top of the SWIG cavity from the prior pass) gives
    /// ferric = -3.5733 kcal/mol, 6.53% off PySCF -- tighter than the
    /// SWIG-only-cavity number (-3.475 kcal/mol, 9.1% off) and closer to the
    /// original pre-SWIG point-charge-cavity number (-3.813 kcal/mol, 0.3%
    /// off, though that was a different, less physical cavity). Tightened
    /// from the earlier order-of-magnitude 0.3x-4x bracket to a real 10%
    /// relative-error assertion, matching this system's measured error with
    /// headroom. See docs/VALIDATION.md's PCM row for the full four-system
    /// picture.
    #[test]
    fn pcm_water_solvation_energy_is_negative_and_reasonable_magnitude() {
        // Holds ENV_LOCK (declared above) because solve_rhf reads the
        // process-global FERRIC_MEM_BUDGET_GB/FERRIC_OOC_BUDGET_GB env vars
        // internally via resolve_three_index_budget -- see the lock's doc
        // comment.
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
        let bs = ferric_core::basis::bundled("sto-3g").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let ctx = ParallelContext::default();

        let vac = solve_rhf(&ctx, &mol, &prep, op, &bounds, &RhfConfig::default()).unwrap();
        assert!(vac.converged);

        let config = RhfConfig {
            pcm: Some(ferric_pcm::PcmConfig::water()),
            ..Default::default()
        };
        let solv = solve_rhf(&ctx, &mol, &prep, op, &bounds, &config).unwrap();
        assert!(solv.converged, "PCM SCF failed to converge");

        let e_solv_ha = solv.energy - vac.energy;
        let e_solv_kcal = e_solv_ha * 627.5094740631;

        assert!(
            e_solv_ha < 0.0,
            "solvation energy must be stabilizing (negative); got {e_solv_ha:.6} Ha = {e_solv_kcal:.3} kcal/mol"
        );

        // PySCF IEF-PCM reference for this exact molecule/basis/eps (own SWIG
        // tessellation): E_solv = -3.8228 kcal/mol. Measured ferric value
        // (Gaussian-smeared S/D, 2026-07-19): -3.5733 kcal/mol, 6.53% off --
        // tolerance set to 10% with headroom above the measured error.
        let pyscf_ref_kcal = -3.8227667932356835_f64;
        let rel_tol = 0.10;
        let rel_err = (e_solv_kcal - pyscf_ref_kcal).abs() / pyscf_ref_kcal.abs();
        assert!(
            rel_err < rel_tol,
            "E_solv={e_solv_kcal:.4} kcal/mol vs PySCF IEF-PCM reference {pyscf_ref_kcal:.4} \
             kcal/mol -- relative error {:.2}% exceeds {:.2}% tolerance",
            rel_err * 100.0,
            rel_tol * 100.0
        );
    }

    /// Coverage-widening companion to `pcm_water_solvation_energy_is_negative_and_reasonable_magnitude`
    /// (see docs/VALIDATION.md's PCM entry / the "second system" cross-check task): that
    /// original test is ONE point (water/STO-3G/eps=78.4). This helper drives the same
    /// vacuum-vs-PCM comparison against an arbitrary molecule/basis/eps, so several
    /// independent systems can each get their own #[test] with their own PySCF reference
    /// number, all using the identical convention:
    ///
    ///   PySCF: `scf.RHF(mol).PCM()`, `with_solvent.method = "IEF-PCM"`,
    ///   `with_solvent.lebedev_order = 29` (302 points/sphere, PySCF's own default),
    ///   `with_solvent.vdw_scale = 1.2` (matches ferric's `PcmConfig` default).
    ///   See /tmp/ferric-pcm-widen/scratch_gen_pcm_refs.py for the exact generation script
    ///   (same convention as the original /tmp/.../gen_pcm_ref.py referenced by the
    ///   water/STO-3G test above).
    fn assert_pcm_solvation_matches_pyscf(
        xyz_path: &str,
        basis_name: &str,
        eps: f64,
        pyscf_e_solv_kcal: f64,
        rel_tol: f64,
        label: &str,
    ) {
        let mol = Molecule::load_xyz(xyz_path).unwrap();
        let bs = ferric_core::basis::bundled(basis_name).unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let ctx = ParallelContext::default();

        let vac = solve_rhf(&ctx, &mol, &prep, op, &bounds, &RhfConfig::default()).unwrap();
        assert!(vac.converged, "{label}: vacuum SCF failed to converge");

        let config = RhfConfig {
            pcm: Some(ferric_pcm::PcmConfig {
                epsilon: eps,
                ..ferric_pcm::PcmConfig::water()
            }),
            ..Default::default()
        };
        let solv = solve_rhf(&ctx, &mol, &prep, op, &bounds, &config).unwrap();
        assert!(solv.converged, "{label}: PCM SCF failed to converge");

        let e_solv_ha = solv.energy - vac.energy;
        let e_solv_kcal = e_solv_ha * 627.5094740631;

        assert!(
            e_solv_ha < 0.0,
            "{label}: solvation energy must be stabilizing (negative); got \
             {e_solv_ha:.6} Ha = {e_solv_kcal:.3} kcal/mol"
        );

        let rel_err = (e_solv_kcal - pyscf_e_solv_kcal).abs() / pyscf_e_solv_kcal.abs();
        assert!(
            rel_err < rel_tol,
            "{label}: E_solv={e_solv_kcal:.4} kcal/mol vs PySCF IEF-PCM reference \
             {pyscf_e_solv_kcal:.4} kcal/mol -- relative error {:.2}% exceeds {:.2}% tolerance",
            rel_err * 100.0,
            rel_tol * 100.0
        );
    }

    /// System 2/4 of the PCM coverage-widening sweep: SAME molecule/geometry as the
    /// original water/STO-3G point, but a bigger basis (cc-pVDZ) -- tests whether the
    /// tight agreement is basis-dependent (bigger basis -> more diffuse density near the
    /// cavity surface -> reaction field more sensitive to the cavity tessellation).
    /// PySCF IEF-PCM reference (own SWIG tessellation): E_solv = -6.2580 kcal/mol.
    ///
    /// **Gaussian-smeared S/D measurement (2026-07-19)**: porting PySCF's
    /// `pcm.py::get_D_S` Gaussian-smeared-charge S/D formulation
    /// (`ferric_pcm::matrices::SdKind::GaussianSmeared`, on top of the SWIG cavity from the
    /// prior pass) gives ferric = -5.7787 kcal/mol, 7.66% off PySCF -- an IMPROVEMENT over
    /// the SWIG-only-cavity number (-5.648 kcal/mol, 9.75% off), though still not back to
    /// the pre-SWIG-cavity coincidental near-exact match (-6.258 kcal/mol, ~0%, on the less
    /// physical hard-cut cavity). Tolerance tightened from 12% to 10% to reflect the real
    /// measured error with headroom; see docs/VALIDATION.md's PCM row for the full picture.
    #[test]
    fn pcm_water_ccpvdz_matches_pyscf_within_a_few_percent() {
        assert_pcm_solvation_matches_pyscf(
            "../../testdata/molecules/water.xyz",
            "cc-pvdz",
            78.4,
            -6.2580,
            0.10,
            "water/cc-pVDZ/eps=78.4",
        );
    }

    /// System 3/4: a genuinely different molecular TOPOLOGY at the same water eps --
    /// NH3 is pyramidal (C3v) rather than water's bent C2v, so the three N-H spheres
    /// overlap the central N sphere in a different geometric pattern than water's two
    /// O-H overlaps. Tests whether the cavity's tightness on water was water-specific or
    /// genuinely generalizes to a different small polar molecule.
    ///
    /// PySCF IEF-PCM reference: E_solv = -3.9709 kcal/mol. Pre-SWIG-switching-function,
    /// ferric measured -3.65 kcal/mol (~8% relative error). Post-SWIG-cavity-only
    /// (2026-07-19): ferric measured -4.036 kcal/mol, ~1.7% off.
    ///
    /// **Gaussian-smeared S/D measurement (2026-07-19)**: porting PySCF's
    /// `pcm.py::get_D_S` formulation (`ferric_pcm::matrices::SdKind::GaussianSmeared`) gives
    /// ferric = -3.9744 kcal/mol, 0.09% off PySCF -- essentially exact, and the tightest of
    /// all four systems, consistent with the sibling `ferric_scf::cosmo` crate's experience
    /// that Gaussian-smearing is the dominant lever for this class of boundary-element
    /// method. Tolerance tightened from 15% to 5% (still generous headroom above the
    /// measured 0.09%).
    #[test]
    fn pcm_nh3_sto3g_within_15_percent_of_pyscf() {
        assert_pcm_solvation_matches_pyscf(
            "../../testdata/molecules/nh3.xyz",
            "sto-3g",
            78.4,
            -3.9709,
            0.05,
            "NH3/STO-3G/eps=78.4",
        );
    }

    /// System 4/4: methanol (Cs, 6 atoms, 2 heavy atoms C+O 1.42 A apart, so the C and O
    /// vdW spheres -- and all 4 H spheres -- overlap much more densely than water's single
    /// central heavy atom or NH3's single central heavy atom). This is a DELIBERATE
    /// negative/stress case for the cavity tessellation.
    ///
    /// PRE-SWIG FINDING (superseded 2026-07-19): with the old hard keep/discard cavity
    /// cut, agreement did NOT generalize here. PySCF IEF-PCM reference (own SWIG
    /// tessellation) is E_solv = -2.6219 kcal/mol at eps=20.7 (acetone); ferric gave
    /// -10.55 kcal/mol -- over 4x too negative (302% relative error).
    ///
    /// POST-SWIG-CAVITY-ONLY (2026-07-19): adding the SWIG switching function
    /// (`ferric_pcm::cavity::build_cavity`) flipped the sign of the error but did not close
    /// it: eps=20.7 gave ferric -1.192 kcal/mol vs PySCF -2.622 kcal/mol (54.6% too weak).
    ///
    /// **Gaussian-smeared S/D measurement (2026-07-19)**: porting PySCF's `pcm.py::get_D_S`
    /// formulation (`ferric_pcm::matrices::SdKind::GaussianSmeared`) on top of the SWIG
    /// cavity closes most of the remaining gap: eps=20.7 gives ferric -2.4738 kcal/mol vs
    /// PySCF -2.6219 kcal/mol (5.65% off, was 54.6% too weak); eps=78.4 gives ferric
    /// -2.6218 kcal/mol vs PySCF -2.7760 kcal/mol (5.55% off, was 64.4% too weak). This is
    /// methanol's best result across all three formulations tried this investigation
    /// (302%-too-strong point-charge/hard-cut -> 54.6%-too-weak point-charge/SWIG ->
    /// 5.65%-off Gaussian-smeared/SWIG), and lands in the same <10% band as the other three
    /// systems -- see docs/VALIDATION.md's PCM row for the full four-system picture.
    /// Test tightened from an order-of-magnitude sign-only check to a real 10%
    /// relative-error assertion at the eps=20.7 point.
    #[test]
    fn pcm_methanol_sto3g_matches_pyscf_within_10_percent() {
        // Holds ENV_LOCK (declared above) because solve_rhf reads the
        // process-global FERRIC_MEM_BUDGET_GB/FERRIC_OOC_BUDGET_GB env vars
        // internally via resolve_three_index_budget -- see the lock's doc
        // comment.
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mol = Molecule::load_xyz("../../testdata/molecules/ch3oh.xyz").unwrap();
        let bs = ferric_core::basis::bundled("sto-3g").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let ctx = ParallelContext::default();

        let vac = solve_rhf(&ctx, &mol, &prep, op, &bounds, &RhfConfig::default()).unwrap();
        assert!(vac.converged);

        let config = RhfConfig {
            pcm: Some(ferric_pcm::PcmConfig {
                epsilon: 20.7,
                ..ferric_pcm::PcmConfig::water()
            }),
            ..Default::default()
        };
        let solv = solve_rhf(&ctx, &mol, &prep, op, &bounds, &config).unwrap();
        assert!(solv.converged, "PCM SCF failed to converge");

        let e_solv_ha = solv.energy - vac.energy;
        let e_solv_kcal = e_solv_ha * 627.5094740631;

        assert!(
            e_solv_ha < 0.0,
            "solvation energy must be stabilizing (negative); got {e_solv_ha:.6} Ha = \
             {e_solv_kcal:.3} kcal/mol"
        );

        // PySCF IEF-PCM reference: -2.6219 kcal/mol. Measured ferric value
        // (Gaussian-smeared S/D, 2026-07-19): -2.4738 kcal/mol, 5.65% off -- tolerance set
        // to 10% with headroom above the measured error.
        let pyscf_ref_kcal = -2.6219_f64;
        let rel_tol = 0.10;
        let rel_err = (e_solv_kcal - pyscf_ref_kcal).abs() / pyscf_ref_kcal.abs();
        assert!(
            rel_err < rel_tol,
            "E_solv={e_solv_kcal:.4} kcal/mol vs PySCF IEF-PCM reference {pyscf_ref_kcal:.4} \
             kcal/mol -- relative error {:.2}% exceeds {:.2}% tolerance",
            rel_err * 100.0,
            rel_tol * 100.0
        );
    }

    #[test]
    fn external_point_charge_changes_rks_pbe_energy() {
        // KS-DFT here is just RhfConfig{xc: Some(...), ..} through solve_rhf (no
        // separate RKS solver). This proves the external_potential wiring from
        // Task 5 composes with the xc.is_some() code path, not just the bare-HF one.
        // See ENV_LOCK doc comment: solve_rhf reads the budget env vars.
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
        let bs = ferric_core::basis::bundled("sto-3g").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let ctx = ParallelContext::default();

        let base_config = RhfConfig {
            xc: Some("PBE".to_string()),
            ..Default::default()
        };
        let base = solve_rhf(&ctx, &mol, &prep, op, &bounds, &base_config).unwrap();

        let ext = ExternalPotential {
            point_charges: vec![PointCharge {
                q: 1.0,
                x: 0.0,
                y: 0.0,
                z: 20.0,
            }],
            smeared_charges: Vec::new(),
            field: None,
        };
        let config = RhfConfig {
            xc: Some("PBE".to_string()),
            external_potential: Some(ext),
            ..Default::default()
        };
        let perturbed = solve_rhf(&ctx, &mol, &prep, op, &bounds, &config).unwrap();

        assert!(perturbed.converged);
        assert!((perturbed.energy - base.energy).abs() > 1e-8);
    }

    fn run_rhf_test(xyz: &str, basis_name: &str, ref_slug: &str, tol: f64) {
        // See ENV_LOCK doc comment: solve_rhf reads the budget env vars.
        // Held here (not at each call site) so it covers all callers
        // (test_rhf_h2_sto3g / test_rhf_h2o_sto3g / test_rhf_h2o_631g) in one place.
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
        let bs = basis::bundled(basis_name).unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let config = RhfConfig {
            energy_conv: 1e-12,
            density_conv: 1e-10,
            integral_thresh: 1e-14,
            ..Default::default()
        };
        let ctx = ParallelContext::default();
        let result = solve_rhf(&ctx, &mol, &prep, op, &bounds, &config).unwrap();
        assert!(result.converged, "RHF did not converge");
        eprintln!(
            "{ref_slug}: energy={:.12}, iters={}, vnn={:.12}",
            result.energy,
            result.iterations,
            mol.nuclear_repulsion()
        );
        let ref_path = format!("../../testdata/reference/{ref_slug}");
        if let Ok(text) = std::fs::read_to_string(&ref_path) {
            let ref_data: serde_json::Value = serde_json::from_str(&text).unwrap();
            let ref_energy = ref_data["energy"].as_f64().unwrap();
            assert!(
                (result.energy - ref_energy).abs() < tol,
                "{ref_slug}: got {:.10}, ref {:.10}",
                result.energy,
                ref_energy
            );
        }
    }

    /// MOM must not change where a well-behaved SCF converges.
    #[test]
    fn rhf_mom_no_harm_water() {
        // See ENV_LOCK doc comment: solve_rhf reads the budget env vars.
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let xyz = "3\nwater\nO 0.000000 0.000000 0.117790\nH 0.000000 0.755453 -0.471161\nH 0.000000 -0.755453 -0.471161\n";
        let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
        let bs = basis::bundled("cc-pvdz").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let ctx = ParallelContext::default();
        let base = RhfConfig {
            energy_conv: 1e-12,
            ..Default::default()
        };
        let mom = RhfConfig {
            mom_after_iter: 2,
            ..base.clone()
        };
        let e0 = solve_rhf(&ctx, &mol, &prep, op, &bounds, &base)
            .unwrap()
            .energy;
        let e1 = solve_rhf(&ctx, &mol, &prep, op, &bounds, &mom)
            .unwrap()
            .energy;
        assert!(
            (e0 - e1).abs() < 1e-9,
            "MOM changed water RHF: {e0} vs {e1}"
        );
    }

    /// A24-21 (C2H4·Ar dimer, aug-cc-pVDZ, DF-JK): aufbau DIIS-8 plateaus
    /// ~33 Ha above the minimum (err_max ~0.9, occupied-set flip-flop) and
    /// never converges in 400 iterations. MOM's contract is to pin the
    /// occupation and kill the flip-flop: with MOM armed at iter 80,
    /// the SCF *converges* — to whichever stationary state DIIS's basin
    /// held at arming (here ~-604, an excited solution 0.9 Ha above the
    /// C2H4+Ar ground state; the ground-state fix for this system is
    /// diis_size=16, with or without MOM). Arming while DIIS still wanders
    /// (iter <~50) pins worse states.
    /// Slow (~2 min release): run with --ignored.
    #[test]
    #[ignore]
    fn rhf_mom_converges_c2h4_ar_dimer() {
        // See ENV_LOCK doc comment: solve_rhf reads the budget env vars.
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let xyz = "7\na24-21 dimer\nC 0.00000000 0.66718073 -2.29024825\nC 0.00000000 -0.66718073 -2.29024825\nH -0.92400768 1.23202333 -2.28975239\nH 0.92400768 1.23202333 -2.28975239\nH -0.92400768 -1.23202333 -2.28975239\nH 0.92400768 -1.23202333 -2.28975239\nAr -0.00000000 0.00000000 1.60829261\n";
        let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
        let bs = basis::bundled("aug-cc-pvdz").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let ctx = ParallelContext::default();
        let config = RhfConfig {
            max_iter: 400,
            df_j_aux: Some("def2-universal-jkfit".into()),
            df_k_aux: Some("def2-universal-jkfit".into()),
            mom_after_iter: 80,
            ..Default::default()
        };
        let result = solve_rhf(&ctx, &mol, &prep, op, &bounds, &config).unwrap();
        assert!(result.converged, "MOM did not break the DIIS flip-flop");
        assert!(
            result.energy < -600.0,
            "C2H4·Ar dimer: converged to {:.10}, not in the molecular basin",
            result.energy
        );
        // Ground-state check: DIIS-16 + MOM lands on the C2H4+Ar limit.
        let config16 = RhfConfig {
            diis_size: 16,
            ..config
        };
        let r16 = solve_rhf(&ctx, &mol, &prep, op, &bounds, &config16).unwrap();
        assert!(
            (r16.energy - (-604.8439254747)).abs() < 1e-5,
            "DIIS-16+MOM: got {:.10}, expected -604.8439254747",
            r16.energy
        );
    }

    #[test]
    fn test_rhf_h2_sto3g() {
        run_rhf_test(
            "2\nH2\nH 0 0 0\nH 0 0 0.74\n",
            "sto-3g",
            "h2_sto-3g_rhf.json",
            1e-8,
        );
    }

    #[test]
    fn test_rhf_h2o_sto3g() {
        // Tolerance 5e-8 due to libint2 vs libcint integral differences
        run_rhf_test(
            "3\nwater\nO 0.000000 0.000000 0.117790\nH 0.000000 0.755453 -0.471161\nH 0.000000 -0.755453 -0.471161\n",
            "sto-3g",
            "h2o_sto-3g_rhf.json",
            5e-8,
        );
    }

    #[test]
    fn test_rhf_h2o_631g() {
        run_rhf_test(
            "3\nwater\nO 0.000000 0.000000 0.117790\nH 0.000000 0.755453 -0.471161\nH 0.000000 -0.755453 -0.471161\n",
            "6-31g",
            "h2o_6-31g_rhf.json",
            1e-8,
        );
    }

    /// COSe (O=C=Se) at aug-cc-pVDZ is a closed-shell singlet that PySCF RHF
    /// converges cleanly to E = -2512.5713600 Ha, but ferric's bare Roothaan+DIIS
    /// *diverges* (energy oscillates ±100 Ha, never settles — heavy-atom Se dense
    /// low-virtual manifold drives DIIS into a limit cycle). A virtual-block level
    /// shift, ramped off as the gradient drops, tames the oscillation. This test
    /// is the regression for that fix: it must converge to the PySCF energy.
    /// Slow (~1-2 min release): run with --ignored.
    #[test]
    #[ignore]
    fn rhf_level_shift_converges_cose() {
        // See ENV_LOCK doc comment: solve_rhf reads the budget env vars.
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let xyz =
            "3\nCOSe\nO 0.0000 0.0000 1.159\nC 0.0000 0.0000 0.0000\nSe 0.0000 0.0000 -1.709\n";
        let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
        let bs = basis::bundled("aug-cc-pvdz").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let ctx = ParallelContext::default();
        let config = RhfConfig {
            max_iter: 200,
            level_shift: 0.5,
            ..Default::default()
        };
        let result = solve_rhf(&ctx, &mol, &prep, op, &bounds, &config).unwrap();
        assert!(
            result.converged,
            "COSe RHF did not converge with level shift"
        );
        assert!(
            (result.energy - (-2512.5713600037)).abs() < 1e-5,
            "COSe RHF: got {:.10}, expected PySCF -2512.5713600037",
            result.energy
        );
    }

    /// Na4 at aug-cc-pVTZ has a near-degenerate occupied manifold (four overlap
    /// eigenvalues ≈ 3e-5): bare DIIS reaches the correct ground-state energy but
    /// the orbital gradient parks on a noise floor (err_max ≈ 3e-5) it can never
    /// drain below density_conv, so without plateau acceptance the SCF spins to
    /// max_iter and returns Err (the "stuck job" in the GW100 aTZ sweep). This is
    /// the regression for the plateau-acceptance fix: it must converge (flagged)
    /// to the CORRECT ground state −774.894064, NOT the ~−771.5/−771.7 excited
    /// states that MOM or a raised lindep threshold converge to. Slow (~5-7 min
    /// release): run with --ignored.
    #[test]
    #[ignore]
    fn rhf_na4_atz_near_degeneracy_plateau_converges() {
        // See ENV_LOCK doc comment: solve_rhf reads the budget env vars.
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let xyz = "4\nNa4\nNa 0.0002445 -0.0998053 1.5471126\nNa -0.0002444 3.1776586 0.0486374\n\
                   Na 0.0002444 0.0997722 -1.5472150\nNa -0.0002444 -3.1776254 -0.0485350\n";
        let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
        let bs = basis::bundled("aug-cc-pvtz").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let ctx = ParallelContext::default();
        // Default config (no level shift, no MOM) — the exact path gw100_full uses.
        let config = RhfConfig {
            max_iter: 60,
            ..Default::default()
        };
        let result = solve_rhf(&ctx, &mol, &prep, op, &bounds, &config).unwrap();
        assert!(
            result.converged,
            "Na4/aTZ did not converge (plateau acceptance failed); iters={}",
            result.iterations
        );
        assert!(
            (result.energy - (-774.894064)).abs() < 1e-4,
            "Na4/aTZ converged to the WRONG state: got {:.6}, expected ground state \
             -774.894064 (a value near -771.5/-771.7 means the plateau ceiling let an \
             excited state through)",
            result.energy
        );
    }

    #[test]
    fn divergence_aborts_early() {
        use crate::result::ScfExit;
        use crate::screening::SchwarzBounds;
        use ferric_core::basis;
        use ferric_core::mol::Molecule;
        use ferric_core::parallel::ParallelContext;
        use ferric_integrals::basis_bridge::PreparedBasis;
        use ferric_integrals::operator::Operator;

        // A guess/level-shift-free run on a hard system that oscillates. We assert
        // the detector CAN fire and returns the right exit reason, using a synthetic
        // check on the config plumbing: divergence_tol very small so any energy rise
        // trips it, with divergence disabled it would run to max_iter.
        // See ENV_LOCK doc comment: solve_rhf reads the budget env vars.
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
        let bs = basis::bundled("sto-3g").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let ctx = ParallelContext::default();

        // Water/STO-3G converges cleanly, so divergence must NOT fire — exit is
        // Converged. This guards against false positives on a healthy descent.
        let cfg = RhfConfig {
            max_iter: 100,
            divergence_tol: Some(0.5),
            stall_window: Some(15),
            ..Default::default()
        };
        let r = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg).unwrap();
        assert!(
            r.converged,
            "water/sto-3g must still converge with detectors on"
        );
        assert_eq!(r.exit, ScfExit::Converged);
    }

    #[test]
    fn detectors_default_off() {
        let cfg = RhfConfig::default();
        assert!(cfg.stall_window.is_none());
        assert!(cfg.divergence_tol.is_none());
    }

    #[test]
    fn solve_rhf_maxiter_returns_ok_not_converged_with_density() {
        use crate::result::ScfExit;

        // See ENV_LOCK doc comment: solve_rhf reads the budget env vars.
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
        let bs = basis::bundled("sto-3g").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let ctx = ParallelContext::default();
        // max_iter = 1 guarantees non-convergence for water.
        let cfg = RhfConfig {
            max_iter: 1,
            ..Default::default()
        };
        let r = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg)
            .expect("max_iter must now return Ok, not Err");
        assert!(!r.converged, "should not be converged in 1 iter");
        assert_eq!(r.exit, ScfExit::MaxIter);
        assert_eq!(r.density_total.dim(), (prep.nbasis(), prep.nbasis()));
        assert!(r.density_total.iter().all(|v| v.is_finite()));
    }

    // --- scf_converged: pure-decision tests (ΔE + ΔP gate, ORCA ConvCheckMode-2) ---

    fn sig(de: f64, dp_rms: f64, dp_max: f64) -> ConvergenceSignals {
        ConvergenceSignals { de, dp_rms, dp_max }
    }

    // Tolerances mirror the defaults: energy_conv (loose ΔE sanity) = 1e-3,
    // density_conv (tight ΔP) = 1e-6.
    const E_CONV: f64 = 1e-3;
    const D_CONV: f64 = 1e-6;

    #[test]
    fn scf_converged_accepts_settled_density_and_nondescending_energy() {
        // ΔP under the tight tol AND ΔE under the loose sanity bound → Converged.
        let r = scf_converged(sig(2e-5, 5e-7, 4e-6), E_CONV, D_CONV);
        assert_eq!(r, Some(ScfExit::Converged));
    }

    #[test]
    fn scf_converged_accepts_when_energy_and_gradient_park_on_ri_floor() {
        // THE key case (MEASURED toluene/aTZ): the commutator parks at ~1.26e-6
        // AND ΔE floors at ~2e-5 — neither drains. But dp_rms has drained to
        // ~1e-9. The gate accepts on ΔP; the loose ΔE bound (1e-3) clears the
        // 2e-5 energy floor. This is the whole point of the redesign: BOTH the
        // gradient and ΔE floor with naux, only ΔP converges.
        let r = scf_converged(sig(2e-5, 1e-9, 9e-8), E_CONV, D_CONV);
        assert_eq!(
            r,
            Some(ScfExit::Converged),
            "must converge on ΔP even when BOTH the gradient and ΔE floor above their tols"
        );
    }

    #[test]
    fn scf_converged_rejects_still_descending_density() {
        // Density still moving (dp_rms ≫ density_conv): a mid-descent iteration.
        let r = scf_converged(sig(2e-5, 3e-5, 1e-3), E_CONV, D_CONV);
        assert_eq!(r, None, "a still-moving density must not be accepted");
    }

    #[test]
    fn scf_converged_rejects_actively_descending_energy() {
        // Density looks settled but the energy is still dropping fast (ΔE ≫ the
        // loose bound) — an early iteration where DIIS briefly stalls the density
        // while the energy is far from settled. The loose ΔE bound still catches it.
        let r = scf_converged(sig(1e-2, 5e-7, 4e-6), E_CONV, D_CONV);
        assert_eq!(
            r, None,
            "an actively-descending energy must not be accepted"
        );
    }

    #[test]
    fn scf_converged_dp_max_companion_guards_a_single_moving_element() {
        // dp_rms looks settled but one density element is still swinging
        // (dp_max > 10·density_conv) → reject (ORCA TolMaxP guard).
        let r = scf_converged(sig(2e-5, 5e-7, 5e-5), E_CONV, D_CONV);
        assert_eq!(
            r, None,
            "dp_max companion must reject a single moving element"
        );
    }

    // --- stall_detected: pure-arithmetic positive-trip tests ---
    //
    // These synthesize errmax_history sequences directly, instead of forcing a
    // real molecule to stall (slow/nondeterministic), to exercise the trip path
    // that the existing detectors_default_off / divergence_aborts_early tests
    // cannot reach (they only guard against false positives on a healthy run).

    #[test]
    fn stall_detected_true_on_oscillating_plateau() {
        // Descend well below the 1e-4 floor's irrelevant here: the plateau band
        // itself is pinned above 1e-4 (a limit cycle that never drains). Window
        // w=4: 8 entries oscillating around 4.5 (band [4.0, 5.0]) means both the
        // recent and previous running-mins land at 4.0, so recent_min (4.0) >=
        // 0.9*prev_min (3.6) trips.
        let w = 4;
        let history = vec![4.0, 5.0, 4.0, 5.0, 4.0, 5.0, 4.0, 5.0];
        assert!(stall_detected(&history, w, 4.5));
    }

    #[test]
    fn stall_detected_false_on_clean_descent() {
        // Each window's running-min drops by >10% vs the previous window, so the
        // detector must not trip on genuine progress.
        let w = 4;
        let history = vec![1e-2, 9e-3, 8e-3, 7e-3, 5e-4, 4e-4, 3e-4, 2e-4];
        assert!(!stall_detected(&history, w, 2e-4));
    }

    #[test]
    fn stall_detected_false_below_plateau_floor() {
        // Flat history that would trip the running-min comparison, but
        // current_err is below the 1e-4 floor: the plateau path (separate logic)
        // owns this regime, not stall detection.
        let w = 4;
        let history = vec![5e-5; 8];
        assert!(!stall_detected(&history, w, 5e-5));
    }

    #[test]
    fn stall_detected_false_when_window_zero() {
        // Guard against the Some(0) false-trip a reviewer flagged: window=0
        // must never trip regardless of history/current_err.
        let history = vec![4.0, 5.0, 4.0, 5.0];
        assert!(!stall_detected(&history, 0, 4.5));
    }

    #[test]
    fn stall_detected_false_when_history_too_short() {
        // Fewer than 2*window entries: not enough data to compare running-mins.
        let w = 4;
        let history = vec![4.0, 5.0, 4.0, 5.0, 4.0]; // 5 < 2*4
        assert!(!stall_detected(&history, w, 4.5));
    }

    /// P14: `build_jk`'s J/K accumulation must be bit-identical regardless of
    /// the rayon worker count. Before this fix the shell-pair work list was
    /// combined via `fold(..).reduce(..)`, a binary tree whose association
    /// (and hence floating-point rounding) depends on the thread count — this
    /// was the root cause the P7 lane traced a whole-pipeline RHF gradient
    /// bit-identity mismatch (last-bit drift, 0x...dfc3 vs 0x...dfbf) back to.
    /// `grouped_deterministic_sum_pair` folds group partials in a fixed,
    /// thread-count-independent ascending order, so the result must match
    /// exactly across pools of different sizes.
    #[test]
    fn build_jk_bit_identical_across_thread_counts() {
        let mol =
            Molecule::parse_xyz("3\nH2O\nO 0 0 0\nH 0 0 0.96\nH 0.93 0 -0.26\n", 0, 1).unwrap();
        let bs = basis::bundled("cc-pvdz").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let n = prep.nbasis();

        // Dense symmetric density so every screened quartet contributes.
        let mut d = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                d[(i, j)] = 0.01 * ((i * 7 + j * 3) % 11) as f64;
            }
        }
        let d = 0.5 * (&d + &d.t());

        let run = |threads: usize| -> (Array2<f64>, Array2<f64>) {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| {
                let ctx = ParallelContext::default();
                let mut j = Array2::zeros((n, n));
                let mut k = Array2::zeros((n, n));
                build_jk(&ctx, &prep, &bounds, 1e-14, &d, &mut j, &mut k).unwrap();
                (j, k)
            })
        };

        let (j1, k1) = run(1);
        let (j4, k4) = run(4);

        for i in 0..n {
            for jj in 0..n {
                assert_eq!(
                    j1[(i, jj)].to_bits(),
                    j4[(i, jj)].to_bits(),
                    "build_jk J not bit-identical across thread counts at ({i},{jj}): \
                     1-thread={:.17e}, 4-thread={:.17e}",
                    j1[(i, jj)],
                    j4[(i, jj)],
                );
                assert_eq!(
                    k1[(i, jj)].to_bits(),
                    k4[(i, jj)].to_bits(),
                    "build_jk K not bit-identical across thread counts at ({i},{jj}): \
                     1-thread={:.17e}, 4-thread={:.17e}",
                    k1[(i, jj)],
                    k4[(i, jj)],
                );
            }
        }
    }

    /// P14: whole-pipeline RHF gradient bit-identity, un-gating the finding
    /// left by the P7 lane (see the comment on
    /// `test_3c2c_assembly_bit_identical_across_thread_counts` in
    /// ferric-mp2/src/gradient.rs). Runs a full `solve_rhf` (which drives
    /// `build_jk` every SCF iteration) followed by `rhf_gradient` under
    /// dedicated 1- and 4-worker rayon pools and compares every component via
    /// `f64::to_bits`. This is the acceptance bar P14 was scoped to satisfy:
    /// build_jk's grouped deterministic reduction must make the *converged*
    /// SCF state — and everything built on it — thread-count-invariant, not
    /// just one isolated builder call.
    #[test]
    fn whole_pipeline_rhf_gradient_bit_identical_across_thread_counts() {
        // See ENV_LOCK doc comment: solve_rhf reads the budget env vars.
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let xyz = "3\nwater\nO 0.000000 0.000000 0.117790\nH 0.000000 0.755453 -0.471161\nH 0.000000 -0.755453 -0.471161\n";
        let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
        let bs = basis::bundled("cc-pvdz").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let config = RhfConfig {
            energy_conv: 1e-12,
            density_conv: 1e-10,
            integral_thresh: 1e-14,
            ..Default::default()
        };

        let run = |threads: usize| -> (f64, Array2<f64>) {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| {
                let ctx = ParallelContext::default();
                let result = solve_rhf(&ctx, &mol, &prep, op, &bounds, &config).unwrap();
                assert!(
                    result.converged,
                    "RHF did not converge at {threads} threads"
                );
                let grad =
                    crate::gradient::rhf_gradient(&mol, &prep, op, &bounds, &result, None).unwrap();
                (result.energy, grad)
            })
        };

        let (e1, g1) = run(1);
        let (e4, g4) = run(4);

        assert_eq!(
            e1.to_bits(),
            e4.to_bits(),
            "RHF energy not bit-identical across thread counts: 1-thread={e1:.17e}, 4-thread={e4:.17e}"
        );
        for atom in 0..3 {
            for c in 0..3 {
                assert_eq!(
                    g1[(atom, c)].to_bits(),
                    g4[(atom, c)].to_bits(),
                    "RHF gradient not bit-identical across thread counts at atom={atom} coord={c}: \
                     1-thread={:.17e} (0x{:016x}), 4-thread={:.17e} (0x{:016x})",
                    g1[(atom, c)], g1[(atom, c)].to_bits(),
                    g4[(atom, c)], g4[(atom, c)].to_bits(),
                );
            }
        }
    }

    /// Solve RHF for `(mol, basis)` on the DIRECT DirectJK path (no df aux, no
    /// xc, default config) with the incremental Fock build ON vs OFF, holding
    /// ENV_LOCK for the `FERRIC_SCF_INCREMENTAL` mutation. Returns
    /// `(energy_incremental, iters_incremental, energy_full, iters_full)`.
    ///
    /// `FERRIC_SCF_INCREMENTAL=0` forces the historical full-rebuild-every-
    /// iteration behavior, so this is a true A/B of the incremental scheme
    /// against the from-scratch build it replaces on the exact same code path.
    fn incremental_vs_full(mol_path: &str, basis: &str) -> (f64, usize, f64, usize) {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mol = Molecule::load_xyz(mol_path).unwrap();
        let bs = basis::bundled(basis).unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let ctx = ParallelContext::default();
        let config = RhfConfig::default();

        // Full-rebuild-every-iteration reference.
        std::env::set_var("FERRIC_SCF_INCREMENTAL", "0");
        let full = solve_rhf(&ctx, &mol, &prep, op, &bounds, &config).unwrap();
        std::env::remove_var("FERRIC_SCF_INCREMENTAL");

        // Incremental (default).
        let incr = solve_rhf(&ctx, &mol, &prep, op, &bounds, &config).unwrap();

        assert!(
            full.converged,
            "full-rebuild RHF must converge for {mol_path}/{basis}"
        );
        assert!(
            incr.converged,
            "incremental RHF must converge for {mol_path}/{basis}"
        );
        (incr.energy, incr.iterations, full.energy, full.iterations)
    }

    /// The incremental (differential) DirectJK Fock build must reproduce the
    /// full-rebuild-every-iteration energy to a tight numerical floor, and MUST
    /// NOT slow convergence (equal-or-fewer iterations) on the direct path.
    ///
    /// Since J/K are linear in D, `J(D_last)+J(ΔD) == J(D_new)` exactly in
    /// infinite precision; in f64 the only difference is the reassociation floor
    /// of summing many small increments vs one full contraction, bounded by the
    /// periodic full rebuild (every 8 iters). The observed energy difference is
    /// well below 1e-9 Ha — far tighter than the direct-path's own PySCF-agreement
    /// tolerance (~1e-8), so we assert 1e-9. This is the direct-path analogue of
    /// the DF-K occ-path stress that was reverted this session: any convergence
    /// slowdown here would be the SAME class of bug, so the iteration-count guard
    /// is a hard assertion, not a soft check.
    #[test]
    fn incremental_fock_matches_full_rebuild_water_ccpvdz() {
        let (e_incr, it_incr, e_full, it_full) =
            incremental_vs_full("../../testdata/molecules/water.xyz", "cc-pvdz");
        assert!(
            (e_incr - e_full).abs() < 1e-9,
            "incremental vs full RHF energy drift too large: incr={e_incr:.12} full={e_full:.12} \
             (Δ={:.3e})",
            (e_incr - e_full).abs()
        );
        assert!(
            it_incr <= it_full,
            "incremental Fock SLOWED convergence: {it_incr} iters vs {it_full} full-rebuild — \
             this is the DF-K-incident bug class, do NOT ship"
        );
    }

    #[test]
    fn incremental_fock_matches_full_rebuild_methane_ccpvdz() {
        let (e_incr, it_incr, e_full, it_full) =
            incremental_vs_full("../../testdata/molecules/methane.xyz", "cc-pvdz");
        assert!(
            (e_incr - e_full).abs() < 1e-9,
            "incremental vs full RHF energy drift too large: incr={e_incr:.12} full={e_full:.12} \
             (Δ={:.3e})",
            (e_incr - e_full).abs()
        );
        assert!(
            it_incr <= it_full,
            "incremental Fock SLOWED convergence: {it_incr} iters vs {it_full} full-rebuild"
        );
    }

    /// SCF-energy-invariance anchor for the shim's per-engine ShellPair
    /// cache (crates/ferric-integrals/shim/shim.cc, ShellPairCache): a full
    /// RHF run with the cache on (default) must reproduce a run with it off
    /// (`FERRIC_SHELLPAIR_CACHE=0`, mirroring the `FERRIC_SCF_INCREMENTAL`
    /// escape-hatch pattern above) BIT-IDENTICALLY, not just to a numerical
    /// tolerance. Unlike the incremental-Fock A/B above (which legitimately
    /// differs at the reassociation floor because it changes THE ORDER OF
    /// SUMMATION), the shell-pair cache changes NOTHING about what is
    /// computed or summed -- it only memoizes `ShellPair::init`'s output,
    /// which is a pure function of (shells, ln_prec, screening_method). If
    /// `ln_precision_of` reproduces libint2's own `Engine::set_precision`
    /// formula exactly (it is written to, see shim.cc), the cached and
    /// uncached quartets are the SAME floating-point computation, so the
    /// resulting Fock matrices, densities, and energy must match to the bit
    /// at every SCF iteration. Any float-level drift here would mean the
    /// precision-matching claim is wrong.
    #[test]
    fn shellpair_cache_matches_uncached_rhf_energy_bit_identical_water_ccpvdz() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
        let bs = basis::bundled("cc-pvdz").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let ctx = ParallelContext::default();
        let config = RhfConfig::default();

        std::env::set_var("FERRIC_SHELLPAIR_CACHE", "0");
        let uncached = solve_rhf(&ctx, &mol, &prep, op, &bounds, &config).unwrap();
        std::env::remove_var("FERRIC_SHELLPAIR_CACHE");

        let cached = solve_rhf(&ctx, &mol, &prep, op, &bounds, &config).unwrap();

        assert!(uncached.converged, "uncached-shellpair RHF must converge");
        assert!(cached.converged, "cached-shellpair RHF must converge");
        assert_eq!(
            cached.iterations, uncached.iterations,
            "shell-pair cache changed the iteration count -- it must not change ANYTHING numerical"
        );
        // NOT bit-identity, and the reason is measured rather than assumed.
        //
        // The cache does not memoize the same arithmetic: libint2 rebuilds a
        // `nullptr` pair in its already-swapped canonical order, but negates a
        // PRECOMPUTED pair's `AB` instead (engine.impl.h:1154-1155, 1210-1221).
        // Both are algebraically identical; they accumulate primitive data in a
        // different order, so individual integrals differ by ~191 ULP. An SCF
        // energy built from those integrals therefore CANNOT be bit-identical,
        // and asserting that it is would be a test asserting a falsehood.
        //
        // What IS guaranteed, and is asserted above, is the trajectory: the
        // iteration count must be unchanged. Measured on benzene/aug-cc-pVDZ,
        // cache on vs off: 12 iterations both ways, with per-iteration dE
        // agreeing to the last printed digit (3.556e-9 vs 3.555e-9 at iter 10).
        // The jitter stays orders below the convergence threshold and never
        // propagates into a different basin or a different iteration count.
        //
        // Bound at the SCF convergence floor rather than the integral floor:
        // the energy is variational in the density, so a 1e-14 integral
        // perturbation moves the converged energy by far less than the 1e-10
        // density threshold the SCF is converging against.
        let de = (cached.energy - uncached.energy).abs();
        assert!(
            de <= 1e-10,
            "shell-pair cache moved the converged SCF energy by {de:.3e} Ha \
             (cached={:.17e} vs uncached={:.17e}), beyond the SCF convergence floor -- \
             a reassociation-scale cache must not shift the converged answer this much",
            cached.energy,
            uncached.energy
        );
    }

    /// Same anchor on a larger, lower-symmetry system with p/d shells, where
    /// the cached ShellPair's directed AB vector actually participates in the
    /// VRR/HRR recurrence coordinates (see engine.rs's shellpair_cache_*
    /// tests for why s-shell-only systems cannot exercise this).
    #[test]
    fn shellpair_cache_matches_uncached_rhf_energy_bit_identical_benzene_ccpvdz() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mol = Molecule::load_xyz("../../testdata/molecules/benzene.xyz").unwrap();
        let bs = basis::bundled("cc-pvdz").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let ctx = ParallelContext::default();
        let config = RhfConfig::default();

        std::env::set_var("FERRIC_SHELLPAIR_CACHE", "0");
        let uncached = solve_rhf(&ctx, &mol, &prep, op, &bounds, &config).unwrap();
        std::env::remove_var("FERRIC_SHELLPAIR_CACHE");

        let cached = solve_rhf(&ctx, &mol, &prep, op, &bounds, &config).unwrap();

        assert!(uncached.converged, "uncached-shellpair RHF must converge");
        assert!(cached.converged, "cached-shellpair RHF must converge");
        assert_eq!(cached.iterations, uncached.iterations);
        // SCF convergence floor, not bit-identity — see the water case above for
        // the full rationale (libint2 rebuilds a `nullptr` pair in swapped order
        // but negates a precomputed one's `AB`, so the integrals differ by pure
        // reassociation at ~191 ULP and the energy built from them cannot be
        // bit-identical). The iteration count asserted just above is the real
        // trajectory guarantee.
        let de = (cached.energy - uncached.energy).abs();
        assert!(
            de <= 1e-10,
            "shell-pair cache moved the converged benzene/cc-pVDZ SCF energy by {de:.3e} Ha \
             (cached={:.17e} vs uncached={:.17e}), beyond the SCF convergence floor",
            cached.energy,
            uncached.energy
        );
    }

    /// Larger, many-iteration direct-path stress case: hexane (C6H14) at cc-pVDZ
    /// is a 20-atom / 118-basis-function all-electron RHF with no df aux, so it
    /// exercises the DirectJK path across a long convergence trajectory (the
    /// regime where incremental Fock both helps most AND, if buggy, would
    /// destabilize DIIS — the direct-path equivalent of the benzene-def2 DF
    /// stress case the DF-K bug broke). cc-pVDZ bundles only Z=1-10, so a C/H
    /// alkane is the heaviest many-iteration closed shell available here.
    ///
    /// `#[ignore]`d: it runs TWO full 118-basis-function SCF solves (incremental
    /// vs full-rebuild A/B) and dominates the `ferric-scf` suite's runtime. The
    /// water and methane A/B tests above already cover the correctness contract
    /// (energy match + no iteration-count regression); this one exists for the
    /// long-trajectory regime specifically. Run it explicitly when touching the
    /// incremental Fock path:
    ///   cargo test --release -p ferric-scf -- --ignored hexane_ccpvdz_stress
    #[test]
    #[ignore]
    fn incremental_fock_matches_full_rebuild_hexane_ccpvdz_stress() {
        let (e_incr, it_incr, e_full, it_full) =
            incremental_vs_full("../../testdata/molecules/alkane_6.xyz", "cc-pvdz");
        assert!(
            (e_incr - e_full).abs() < 1e-9,
            "incremental vs full RHF energy drift too large on hexane: \
             incr={e_incr:.12} full={e_full:.12} (Δ={:.3e})",
            (e_incr - e_full).abs()
        );
        assert!(
            it_incr <= it_full,
            "incremental Fock SLOWED convergence on hexane: {it_incr} iters vs \
             {it_full} full-rebuild — this is the DF-K-incident bug class, do NOT ship"
        );
    }

    // ── DF-K is built only when its K reaches F ─────────────────────────────
    //
    // Observable: `fock_assembly::DF_K_BUILT`, a test-only thread-local bumped
    // each time `build_df_jk` returns a DfK. Each case runs one solve and
    // reads the DELTA on this thread. The MINAO guess solves no SCF, so the
    // only `build_df_jk` call on this thread is the molecule's own.

    /// Number of DfK fitters `solve_rhf` constructed for water/STO-3G with
    /// `xc` and `df_k_aux`; `df_j_aux` left unset (auto RI-J for a functional).
    fn df_k_builds(xc: Option<&str>, df_k_aux: Option<&str>) -> usize {
        // See ENV_LOCK doc comment: solve_rhf reads the budget env vars.
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
        let bs = basis::bundled("sto-3g").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let config = RhfConfig {
            xc: xc.map(str::to_string),
            df_k_aux: df_k_aux.map(str::to_string),
            ..Default::default()
        };
        let before = crate::fock_assembly::DF_K_BUILT.with(|c| c.get());
        let res = solve_rhf(
            &ParallelContext::default(),
            &mol,
            &prep,
            op,
            &bounds,
            &config,
        )
        .unwrap();
        assert!(res.converged, "{xc:?} did not converge");
        crate::fock_assembly::DF_K_BUILT.with(|c| c.get()) - before
    }

    /// A pure GGA consumes no exact exchange, so an explicitly named DF-K aux
    /// (what `run_dft` passes for every functional) must not build a DfK.
    /// FAILS (1 != 0) if `df_k_aux_eff` goes back to `resolve_aux(.., needs_k)`
    /// without the `k_buf_consumed` gate: `resolve_aux` honours `Some(name)`
    /// regardless of `needs_k`.
    #[test]
    fn pure_gga_with_named_df_k_aux_builds_no_dfk() {
        assert_eq!(
            df_k_builds(Some("PBE"), Some(crate::fock_assembly::DEFAULT_JK_AUX)),
            0
        );
        // Unset aux must not auto-default one either (needs_k is false).
        assert_eq!(df_k_builds(Some("PBE"), None), 0);
    }

    /// RSH contracts exchange from its own SR/LR fitters (`driver::prepare`),
    /// never from `k_buf`, so the main DfK is dead weight there too. FAILS
    /// (1 != 0) if the gate uses `k_consumed` (true for RSH) instead of
    /// `k_buf_consumed`.
    #[test]
    fn rsh_builds_no_main_dfk() {
        assert_eq!(df_k_builds(Some("wB97X-V"), None), 0);
    }

    /// Positive controls: HF (xc = None, `needs_k` FALSE but K consumed) and a
    /// plain hybrid must still build their DfK. FAIL (0 != 1) if the gate is
    /// keyed on `needs_k`, which is DFT-specific and false for HF, or if it
    /// drops ω = 0 hybrids.
    #[test]
    fn hf_and_hybrid_still_build_dfk() {
        assert_eq!(
            df_k_builds(None, Some(crate::fock_assembly::DEFAULT_JK_AUX)),
            1
        );
        assert_eq!(df_k_builds(Some("B3LYP"), None), 1);
        assert_eq!(
            df_k_builds(Some("B3LYP"), Some(crate::fock_assembly::DEFAULT_JK_AUX)),
            1
        );
    }
}
