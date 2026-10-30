//! The SCF precision ladder for the mixed-precision J/K kernels `dfj-pack`
//! (device RI-J, `crate::df_j_gpu`) and `dfk-occ` (device DF-K occupied path,
//! `crate::df_k_gpu`), with an f64 exit.
//!
//! The ladder is the one `xc-vxc` uses ([`ferric_core::precision_ladder`]): a
//! J/K build may run a mixed device kernel only when mixed precision was
//! requested for one of the two kernels, a previous build of the same SCF
//! exists, and some spin density moved by more than [`JK_LADDER_THRESH`](crate::jk_ladder::JK_LADDER_THRESH) (RMS
//! over the AO matrix) since it. The first build of every SCF is f64. When the
//! ladder says f64 and the resident device tensor is f32, the build runs the
//! host f64 path (the host `DfJ`/`DfK` still holds, or recomputes, the f64
//! tensor), so the device needs no f64 copy.
//!
//! The f64 exit. The SCF exit gate (ΔP below `density_conv`) is a different
//! test from the ladder's, so an exit can follow a mixed build (always under a
//! loose `density_conv`, never in practice at a tight one). The energy is
//! FIRST-order in the J/K error (unlike `xc-vxc`, whose E_xc comes from an f64
//! pass), so swapping the Fock matrix alone would not do: a converged exit whose
//! last build ran mixed is REFUSED ([`JkLadder::veto_mixed_exit`](crate::jk_ladder::JkLadder::veto_mixed_exit)), the ladder
//! is latched to f64, and the SCF continues; the iterations from there to the
//! exit, the returned energy, Fock matrix, orbitals and density are all f64.
//! The latched ladder never returns to mixed in that SCF.
//!
//! Scope. The solvers bracket each iteration's J/K build in a
//! [`JkLadder::begin_build`](crate::jk_ladder::JkLadder::begin_build) guard, which publishes the ladder's decision to the
//! device dispatchers on this thread ([`scope_allows_mixed`](crate::jk_ladder::scope_allows_mixed)). Outside any
//! guard (a one-shot build, a gradient, a test that calls `DfJ::build`
//! directly) nothing changes: the kernels behave as before this ladder. Wired:
//! the DIIS loops of `solve_rhf`, `solve_uhf` and `solve_rohf`. Not wired: the
//! second-order solvers (Newton, TRAH, AH), stability analysis and gradients.
//! With mixed precision not requested the guard is inert (no density copy, no
//! scope), so the f64 default runs exactly the code it ran before.
//!
//! # Error budget
//!
//! Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed. (SIAM
//! 2002): unit roundoff §2.2 (u32 = 2⁻²⁴ = 5.96e-8, u64 = 2⁻⁵³), γ_n(u) =
//! nu/(1−nu) §3.1, inner products and matrix products §3.5. Every bound is
//! `γ_depth · Σ|terms|`, i.e. relative error `γ_depth · κ` with the summation
//! condition number κ = Σ|terms| / |Σ terms|.
//!
//! | stage | depth n | κ | per-build bound | precision |
//! |---|---|---|---|---|
//! | `dfj-pack` pass 1, `d_P = Σ_k B̂_Pk w_k` | pair = nbf(nbf+1)/2 | κ_d = (\|B\|\|w\|)_P / \|d_P\| | `\|δd_P\| ≤ (u32 + γ_pair(u64)(1+u32))·(\|B\|\|w\|)_P` (`df_j_gpu::d_error_factor_mixed`) | f32 storage of B, f64 FMA accumulation |
//! | metric solve `c = V⁻¹d` | — | ‖V⁻¹‖₂ = 1/λ_min(V) | `δc = V⁻¹δd` (Fock-matrix amplification) | f64 Cholesky, always |
//! | `dfj-pack` pass 2, `J_k = Σ_P B̂_Pk c_P` | naux | κ_J = (\|B\|ᵀ\|c\|)_k / \|J_k\| | `\|δJ_k\| ≤ (u32 + γ_naux(u64)(1+u32))·(\|B\|ᵀ\|c\|)_k + (\|B\|ᵀ\|δc\|)_k` (`j_error_factor_mixed`) | same |
//! | `dfk-occ` stage 1, `X_P = B̂_P Ĉ_occ` | nbf, f32 panels of k_panel | κ_X (orbital nodes: κ ≫ 1) | `\|δX_P\| ≤ ε₁·\|B_P\|\|C\|`, ε₁ = `mixed_error_factor(nbf, k_panel)` (two f32 operand roundings, f32 panel sums γ_kpanel(u32), f64 sum over panels) | f32 B and C_occ, f64 flush |
//! | `dfk-occ` stage 2, `K = Σ_{P,i} X Xᵀ` | naux·nocc (k_chunk + nchunks) | diagonal κ = 1; off-diagonal κ ≫ 1 | `\|δK\| ≤ ε_K·S`, `S = Σ_P (\|B_P\|\|C\|)(\|B_P\|\|C\|)ᵀ`, ε_K = 2ε₁ + ε₁² + γ_{k_chunk+nchunks}(u64)(1+ε₁)² (`df_k_gpu::k_error_factor_mixed`) | f64 SYRK |
//!
//! What the ladder protects is the ENERGY. With `E = ½ tr D(H + F)` and
//! `F ∋ J(D) − (c_x/2) K(D)` (closed shell, `K(D) = 2 K(C_occ)`):
//!
//! * J. `E_J = ½ wᵀJ = ½ cᵀd`. With `B̂ = B + δB`, `|δB| ≤ u32|B|`, first order
//!   `½ wᵀ(δBᵀc + Bᵀ V⁻¹ δB w) = cᵀ δB w`: the metric enters through `c` only,
//!   so the energy carries NO 1/λ_min(V) (the Fock matrix does). Per build
//!   `|δE_J| ≤ ε_EJ · S_J`, `S_J = |c|ᵀ|B||w|`, `ε_EJ = u32 + γ_pair(u64) +
//!   γ_naux(u64)` (+ O(u²)); κ_EJ = S_J / |cᵀBw| = S_J / (2|E_J|).
//! * K. `|δE_K| ≤ (c_x/2) · ε_K · Σ_μν |D_μν| S_μν` (UHF: `(c_x/2) ε_K Σ_σ
//!   Σ|D_σ| S_σ` with `K_σ = K(C_σ)`).
//!
//! Through the SCF: a converged energy whose last J/K build was mixed carries
//! those FIRST-order terms evaluated at the converged density; the density is
//! the fixed point of the perturbed Fock map, displaced by about δF/gap times
//! the fixed-point amplification 1/(1−‖Jac‖), which enters the (stationary)
//! energy at SECOND order only. `dfj-pack` rounds B once, so its mixed J is a
//! fixed linear map of D (no build-to-build noise); `dfk-occ` re-rounds C_occ
//! every build, so its error moves with the density. Measured (GTX 1080, RHF,
//! `density_conv` 1e-9): with EVERY build mixed (τ = 0) that `dfk-occ` noise kept
//! C6H6/def2-SVP from converging in 200 iterations (water converged in 34 vs
//! 13), while `dfj-pack` alone converged in 14-16 vs 13-15; the production τ
//! ran 5-7 builds mixed and converged in the f64 iteration count ± 1. The exit
//! veto can only act once the exit gate is reached, so τ is what keeps
//! `dfk-occ` convergent, and the veto what keeps the energy f64.
//!
//! With the ladder and the f64 exit the returned energy is evaluated with f64
//! J/K at a density converged under f64 J/K, so no u32 term remains: its budget
//! against an all-f64 SCF of the same system is the f64 floor of either code
//! path, twice (each run is within it of exact arithmetic),
//!
//! `T_γ = (γ_pair + γ_naux)(u64)·S_J + (c_x/2)(2γ_nbf + γ_{naux·nocc})(u64)·Σ|D|S
//!        + γ_{nbf²}(u64)·½Σ|D||H+F|`,
//!
//! and the bar is `2·T_γ + 2·|ΔE_exit(f64 reference)|`: the f64 floor twice,
//! plus the realized termination of an f64 SCF under the same ΔP gate (the
//! energy change of the reference's exit iteration; ΔP stationarity puts the
//! remaining density error into E at second order only). The configured
//! `energy_conv` is NOT used: under RI it is a loose "not still descending"
//! bound, orders of magnitude above both sides of the gate. Without the exit
//! (the mutant) the first-order u32 term
//! `M = ε_EJ·S_J + (c_x/2)ε_K Σ|D|S`
//! bounds the error from above; the gap between the two is
//! `u32 / (γ_pair(u64)+γ_naux(u64)) ≈ u32 / (pair·u64) ≈ 5e8 / pair`, about
//! 1.6e4 at pair ≈ 3.1e4 (C10H22/def2-SVP), shrinking as 1/nbf² (the
//! worst-case f64 floor grows with the depth while the u32 term does not).
//! `tests/gpu_jk_ladder.rs` computes `S_J`, `S` and `T_γ` per system and
//! asserts the bar on both sides.
//!
//! Must stay f64: the metric Cholesky and solve, the V^{-1/2} lindep dressing,
//! DIIS, diagonalisation, the energy trace, every build after the exit veto.
use std::cell::Cell;

use ferric_core::precision_ladder::{LadderSite, PrecisionLadder};
use ndarray::Array2;

/// RMS change of a spin density between two consecutive J/K builds above
/// which the later build may run `dfj-pack` / `dfk-occ` mixed.
///
/// The `xc-vxc` threshold (`ferric_dft::ks::MIXED_VXC_LADDER_THRESH`, chosen by
/// measurement there) and the switch point of Laqua/Kussmann/Ochsenfeld's
/// i-sn-LinK (fp32 until the DIIS RMS error reaches 1e-5, JCP 154, 214116
/// (2021)). Its role here is convergence, not accuracy: the f64 exit makes the
/// returned energy independent of τ (any τ, including 0, gives an f64 exit).
pub const JK_LADDER_THRESH: f64 = 1e-5;

thread_local! {
    /// The ladder's decision for the J/K build in progress on this thread:
    /// `None` outside any [`JkBuildScope`].
    static SCOPE: Cell<Option<bool>> = const { Cell::new(None) };
    /// Builds on this thread that ran a mixed device kernel (monotone).
    static RAN_MIXED: Cell<u64> = const { Cell::new(0) };
}

/// The ladder's decision for the J/K build in progress on this thread:
/// `Some(true)` mixed allowed, `Some(false)` f64 required, `None` no ladder
/// (behave as without one).
pub fn scope_allows_mixed() -> Option<bool> {
    SCOPE.with(Cell::get)
}

/// A device kernel ran this build mixed (`dfj-pack` or `dfk-occ`). Called by
/// the device dispatchers; the enclosing guard keeps the build for the exit.
pub fn note_mixed_build() {
    RAN_MIXED.with(|c| c.set(c.get().wrapping_add(1)));
    // TODO(feat/precision-section-provenance): note_kernel_mixed(kernel) here,
    // with the kernel passed in by the dispatcher.
}

/// Test seams (`test-seams` feature). Never set by a shipped build.
#[cfg(feature = "test-seams")]
#[doc(hidden)]
pub mod seams {
    use std::sync::atomic::AtomicBool;
    use std::sync::Mutex;
    /// No ladder at all: the solvers open no scope, so every build runs as
    /// before the ladder (every device build mixed, no f64 exit). The kernel
    /// error maps measure the kernel itself with this.
    pub static JK_LADDER_OFF: AtomicBool = AtomicBool::new(false);
    /// Replace [`super::JK_LADDER_THRESH`] (e.g. 0: every build after the
    /// first is mixed until the exit).
    pub static JK_LADDER_THRESH_OVERRIDE: Mutex<Option<f64>> = Mutex::new(None);
}

fn thresh() -> f64 {
    #[cfg(feature = "test-seams")]
    if let Some(t) = *seams::JK_LADDER_THRESH_OVERRIDE
        .lock()
        .unwrap_or_else(|e| e.into_inner())
    {
        return t;
    }
    JK_LADDER_THRESH
}

fn ladder_off() -> bool {
    #[cfg(feature = "test-seams")]
    {
        seams::JK_LADDER_OFF.load(std::sync::atomic::Ordering::Relaxed)
    }
    #[cfg(not(feature = "test-seams"))]
    {
        false
    }
}

/// Whether a mixed J/K kernel could run in this process: `dfj-pack` or
/// `dfk-occ` in the allowlist under `precision = mixed`, a device mode other
/// than off, and a ready device. Read only for an SCF that holds a DF-J or DF-K
/// builder (`JkLadder::new([true])`), so an SCF that cannot reach a device kernel
/// never reads the GPU settings.
#[cfg(feature = "gpu")]
fn mixed_requested() -> bool {
    use ferric_core::gpu::{GpuMode, GpuStatus, MixedKernel};
    let allowed = crate::df_k_gpu::mixed_kernel_allowed(MixedKernel::DfjPack)
        || crate::df_k_gpu::mixed_kernel_allowed(MixedKernel::DfkOcc);
    allowed
        && ferric_core::gpu::settings().mode != GpuMode::Off
        && matches!(ferric_core::gpu::status(), GpuStatus::Ready(_))
}

#[cfg(not(feature = "gpu"))]
fn mixed_requested() -> bool {
    false
}

/// One SCF's J/K precision ladder (see the module docs).
pub struct JkLadder {
    ladder: PrecisionLadder<()>,
    /// Whether this SCF can reach a device J/K kernel at all.
    has_df: bool,
    /// Set by an exit veto: every later build of this SCF is f64.
    latched_f64: Cell<bool>,
}

impl JkLadder {
    /// A ladder for one SCF. `df_builders`: which DF-J / DF-K builders the SCF
    /// holds (the only J/K builders with a mixed device kernel); with none the
    /// ladder is inert.
    pub fn new<const N: usize>(df_builders: [bool; N]) -> Self {
        Self {
            ladder: PrecisionLadder::new(LadderSite::Jk),
            has_df: df_builders.contains(&true),
            latched_f64: Cell::new(false),
        }
    }

    /// Step the ladder with this build's spin densities (`[D]` closed shell,
    /// `[D_α, D_β]` open shell) and publish its decision to this thread until
    /// the guard drops. Inert (no copy, no scope) when no mixed J/K kernel is
    /// requested.
    pub fn begin_build(&self, ds: &[&Array2<f64>]) -> JkBuildScope<'_> {
        if !self.has_df || ladder_off() || !mixed_requested() {
            return JkBuildScope {
                ladder: None,
                prev: None,
                ran0: 0,
            };
        }
        // TODO(feat/precision-section-provenance): note_ladder_build(DfjPack /
        // DfkOcc) here (counted today by the shared ladder under LadderSite::Jk).
        let moving = self.ladder.step_with(true, thresh(), ds);
        let mixed = moving && !self.latched_f64.get();
        let prev = SCOPE.with(|s| s.replace(Some(mixed)));
        JkBuildScope {
            ladder: Some(self),
            prev,
            ran0: RAN_MIXED.with(Cell::get),
        }
    }

    /// The f64 exit: `conv` is the solver's converged-exit decision for this
    /// iteration. When the iteration's J/K build ran a mixed kernel the exit is
    /// refused (`None`), the ladder is latched to f64 for the rest of this SCF,
    /// and the SCF iterates on with f64 J/K; otherwise `conv` is returned
    /// unchanged.
    pub fn veto_mixed_exit<E>(&self, conv: Option<E>) -> Option<E> {
        if conv.is_some() && self.ladder.take_mixed().is_some() {
            self.latched_f64.set(true);
            // TODO(feat/precision-section-provenance): note_exit_rebuild(kernel)
            // here (counted today by the shared ladder under LadderSite::Jk).
            return None;
        }
        conv
    }

    /// Whether an exit veto has latched this SCF to f64.
    pub fn latched_f64(&self) -> bool {
        self.latched_f64.get()
    }
}

/// The scope of one J/K build (see [`JkLadder::begin_build`]). On drop it
/// restores the previous scope and, when a device kernel ran mixed inside it,
/// keeps the build for the exit veto.
#[must_use = "the ladder's decision holds only while this guard is alive"]
pub struct JkBuildScope<'a> {
    ladder: Option<&'a JkLadder>,
    prev: Option<bool>,
    ran0: u64,
}

impl Drop for JkBuildScope<'_> {
    fn drop(&mut self) {
        let Some(l) = self.ladder else {
            return;
        };
        SCOPE.with(|s| s.set(self.prev));
        if RAN_MIXED.with(Cell::get) != self.ran0 {
            l.ladder.keep_mixed(());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn guard_with(l: &JkLadder, mixed: bool) -> JkBuildScope<'_> {
        // A scope as `begin_build` opens it when mixed is requested.
        let prev = SCOPE.with(|s| s.replace(Some(mixed)));
        JkBuildScope {
            ladder: Some(l),
            prev,
            ran0: RAN_MIXED.with(Cell::get),
        }
    }

    /// Catches: a scope that leaks past its guard, a build kept for the exit
    /// when no kernel ran mixed, and a veto that does not latch or that fires
    /// on a non-converged iteration.
    #[test]
    fn scope_restores_and_only_a_mixed_build_vetoes_the_exit() {
        let l = JkLadder::new([true]);
        assert_eq!(scope_allows_mixed(), None);
        {
            let _g = guard_with(&l, true);
            assert_eq!(scope_allows_mixed(), Some(true));
            {
                let _h = guard_with(&l, false);
                assert_eq!(scope_allows_mixed(), Some(false));
            }
            assert_eq!(scope_allows_mixed(), Some(true));
        }
        assert_eq!(scope_allows_mixed(), None);
        // No kernel ran mixed: the exit stands.
        assert_eq!(l.veto_mixed_exit(Some(1)), Some(1));
        // A kernel ran mixed: a non-exit is untouched, the exit is refused once.
        {
            let _g = guard_with(&l, true);
            note_mixed_build();
        }
        assert_eq!(l.veto_mixed_exit(None::<u8>), None);
        assert!(!l.latched_f64(), "a non-exit latched");
        assert_eq!(l.veto_mixed_exit(Some(2)), None, "mixed exit accepted");
        assert!(l.latched_f64());
        assert_eq!(l.veto_mixed_exit(Some(3)), Some(3), "vetoed twice");
    }

    /// Catches: an inert ladder that opens a scope (the f64 default must run
    /// the pre-ladder code path) — without the `gpu` feature, or with no DF
    /// builder, mixed is never requested.
    #[test]
    fn an_scf_without_a_df_builder_opens_no_scope() {
        let l = JkLadder::new([false]);
        let d = Array2::<f64>::zeros((2, 2));
        let _g = l.begin_build(&[&d]);
        assert_eq!(scope_allows_mixed(), None);
        assert!(_g.ladder.is_none());
    }
}
