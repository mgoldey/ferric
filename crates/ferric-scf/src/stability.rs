//! SCF/KS **internal stability analysis**: is this converged solution a
//! minimum, or a saddle point?
//!
//! # Why this module exists
//!
//! An SCF solver converges when the orbital gradient vanishes — the Brillouin
//! condition `F^σ_{ai} = 0`. That makes the solution a *stationary point* of
//! the energy with respect to occupied↔virtual orbital rotations. It does NOT
//! make it a *minimum*. A converged SCF state can be a saddle: some rotation
//! direction lowers the energy, and the solver has no way to notice, because
//! the gradient is zero in every direction at a saddle too.
//!
//! The distinguishing quantity is the lowest eigenvalue of the **electronic
//! orbital Hessian**
//!
//! ```text
//!   H_{ai,bj} = ∂²E / ∂κ_{ai} ∂κ_{bj}
//! ```
//!
//! evaluated at the converged point. If `λ_min > 0` the point is a local
//! minimum within the rotation space considered; if `λ_min < 0` the
//! corresponding eigenvector `κ` is a downhill direction and the point is a
//! saddle — the solution is UNSTABLE.
//!
//! # Motivating measurement (the reason this landed)
//!
//! HeNe⁺ / def2-SVP / UHF at R = 2.0 Å. Ferric's unconstrained UHF converges
//! cleanly (14 iterations, ΔE < 1e-11) to `E = -130.5003466413` with a hole
//! that is 100% Ne 2p_π. PySCF 2.13.0 on the identical system reaches
//! `E = -130.5053405386` from its default guess and reports that state
//! internally AND externally STABLE. Forcing PySCF onto the π state via
//! `mom_occ` reproduces ferric's energy to 1.3e-9 Ha — and `mf.stability()`
//! reports `internal stable = False` for it. Ferric was converging, silently
//! and reproducibly, onto a saddle point 4.99e-3 Ha = 0.136 eV above the true
//! minimum, with no diagnostic that could tell the two apart.
//!
//! # What this module does and does NOT cover
//!
//! **Implemented: INTERNAL stability**, i.e. real orbital rotations *within
//! the current spin ansatz*:
//!
//! * [`uhf_internal_stability`] — UHF/UKS. Rotations are the α and β
//!   occ→virt blocks (κ_α, κ_β), varied INDEPENDENTLY. This is exactly the
//!   space the UHF energy is already stationary in, so it answers "is this
//!   UHF solution a UHF minimum?".
//! * [`rhf_internal_stability`] — RHF/RKS. The single closed-shell occ→virt
//!   rotation applied identically to both spins (the *singlet* channel,
//!   `A + B` in TDHF language). This answers "is this RHF solution an RHF
//!   minimum?".
//!
//! **NOT implemented: EXTERNAL stability.** PySCF's `stability()` also reports
//! an external verdict, which is a genuinely DIFFERENT Hessian block, not a
//! bigger version of this one:
//!
//! * RHF→UHF (spin-symmetry breaking) is the *triplet* channel: κ_α = −κ_β
//!   rather than κ_α = κ_β. In the response, the triplet Hessian has the
//!   Coulomb term `δJ` ABSENT (the α and β density perturbations cancel in
//!   δD_total) where the singlet channel has it present. Neither
//!   [`rhf_newton::hessian_matvec`](crate::rhf_newton::hessian_matvec) nor
//!   [`uhf_newton::hessian_matvec`](crate::uhf_newton::hessian_matvec) can
//!   express that: the RHF matvec hard-codes the singlet combination
//!   (`δD = 2·δD_single`, so δJ is always present), and the UHF matvec builds
//!   δJ from `δD_α + δD_β` — feeding it κ_β = −κ_α would give the triplet
//!   Coulomb cancellation but the wrong exchange bookkeeping for a
//!   *restricted* reference, because the UHF matvec reads gaps from two
//!   separate Fock matrices that are identical at an RHF point. Doing it
//!   correctly requires a new matvec, not a new caller.
//! * real→complex instabilities need a complex-κ Hessian, which does not exist
//!   anywhere in this crate.
//!
//! A `STABLE` verdict from this module therefore means "stable against the
//! rotations tested", never "stable" unqualified. [`StabilityResult`] carries
//! [`StabilityKind`] so callers can see which question was answered, and the
//! result struct is deliberately DIAGNOSTIC: nothing in the SCF path fails or
//! changes behaviour because of it. A deliberately-unstable state (a
//! constrained-DFT diabat, an excited MOM state) is a legitimate thing to
//! compute, and the caller decides what an instability means.
//!
//! # How it is computed
//!
//! The Hessian is never formed. The same
//! [`uhf_newton::hessian_matvec`](crate::uhf_newton::hessian_matvec) that the
//! Newton solver drives inside its PCG loop — already validated
//! analytic-vs-finite-difference to ~3e-10 in `uhf_newton_smoke.rs` — is
//! driven instead by a **Davidson extremal-eigenvalue solver**
//! ([`davidson_lowest`](crate::stability::davidson_lowest)). Stability
//! analysis and a Newton step are the same
//! operator asked two different questions: Newton does a linear solve `Hκ =
//! −g`, stability takes the lowest eigenpair of `H`.
//!
//! Davidson rather than Lanczos because the orbital Hessian is strongly
//! diagonally dominant — its diagonal is the orbital-energy gap
//! `ε_a − ε_i` plus a small two-electron correction — which is precisely the
//! regime where Davidson's diagonal preconditioner `t = r/(diag − λ)` pays
//! off (measured: 7 iterations on water/STO-3G, 23 on HeNe⁺/def2-SVP where
//! the rotation space is 148-dimensional). The eigensolver reports its own
//! residual and `converged` flag, because an iteration-starved eigensolve
//! returning a wrong λ_min would silently defeat the entire purpose of this
//! module.
//!
//! **It is a BLOCK Davidson, and that is load-bearing rather than an
//! optimization.** The orbital Hessian is block diagonal in the molecule's
//! irreps; a single tracked root can converge exactly inside one symmetry
//! block, report `converged = true` with residual 6e-11, and return an
//! eigenvalue that is not the lowest. Convergence is therefore declared only
//! when the WHOLE tracked block has converged — a root below the block leaves
//! at least one block residual large. See [`StabilityConfig::n_block`] for the
//! measured sweep that established this.

use crate::engine_pool::EnginePool;
use crate::rhf_newton::RhfNewtonInputs;
use crate::uhf_newton::UhfNewtonInputs;
use ferric_core::parallel::ParallelContext;
use ferric_core::FerricError;
use ndarray::{Array1, Array2};
use ndarray_linalg::{Eigh, UPLO};

/// Which rotation space the stability verdict refers to.
///
/// A verdict is only ever meaningful together with the space it was computed
/// in — "stable" always means "stable against these rotations".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StabilityKind {
    /// UHF/UKS internal: independent real α and β occ→virt rotations.
    UhfInternal,
    /// RHF/RKS internal: one real closed-shell occ→virt rotation applied to
    /// both spins (the singlet channel). Says nothing about RHF→UHF spin
    /// symmetry breaking, which is the triplet channel (see module docs).
    RhfInternal,
}

impl StabilityKind {
    /// Human-readable name for logs and assertion messages.
    pub fn label(self) -> &'static str {
        match self {
            StabilityKind::UhfInternal => "UHF internal (real, same-spin-ansatz)",
            StabilityKind::RhfInternal => "RHF internal (real, singlet channel)",
        }
    }
}

/// Outcome of an internal stability analysis.
///
/// Follows the crate's honesty convention (cf. `ScfResult::converged`,
/// `GwResult::outer_converged`): the verdict and the evidence for trusting it
/// travel together, and the caller decides what to do about them. In
/// particular [`is_stable`](Self::is_stable) must NOT be read without checking
/// [`converged`](Self::converged) — an unconverged eigensolve has not proved
/// anything in either direction.
#[derive(Debug, Clone)]
#[must_use = "a stability verdict that is not inspected is the same as not running one"]
pub struct StabilityResult {
    /// Which rotation space this verdict covers.
    pub kind: StabilityKind,
    /// Lowest eigenvalue of the electronic orbital Hessian, in Hartree per
    /// (radian²) of orbital rotation. Negative ⇒ a downhill rotation exists.
    pub lowest_eigenvalue: f64,
    /// **NOT-PROVEN-UNSTABLE**, i.e. `lowest_eigenvalue > -noise_floor`.
    ///
    /// DOCSTRING CORRECTION (2026-09-16, made while wiring this into the SCF
    /// path): this previously documented the NEGATION of the code it sits on
    /// — `lowest_eigenvalue < -tol`, which is the UNSTABLE condition. The code
    /// has always been `> -noise_floor` and is unchanged; only the prose was
    /// wrong. The distinction is not cosmetic, because it makes this field a
    /// TRAP for callers:
    ///
    /// * `true` does **not** mean "a minimum". It includes the whole marginal
    ///   band `-noise_floor < λ_min <= 0`, so a NEGATIVE eigenvalue can set it
    ///   `true`.
    /// * `false` does not mean "definitely a saddle" either when
    ///   `lowest_eigenvalue.abs() <= noise_floor`.
    ///
    /// **Do not branch on this field alone.** Use [`verdict`](Self::verdict),
    /// which is total over the four distinguishable outcomes and cannot be
    /// misread, or replicate [`summary`](Self::summary)'s discipline of
    /// checking [`converged`](Self::converged) and
    /// [`is_marginal`](Self::is_marginal) FIRST. The field is retained because
    /// it is what the analysis primitively computes and the existing tests
    /// pin it.
    pub is_stable: bool,
    /// The Hessian eigenvector for `lowest_eigenvalue`, as the α occ→virt
    /// rotation block `(nvirt_α, nocc_α)`. Normalized jointly with
    /// [`eigenvector_beta`](Self::eigenvector_beta) (the packed (α,β) vector
    /// has unit 2-norm).
    ///
    /// When `lowest_eigenvalue < 0` this is the downhill direction: rotating
    /// the MOs along it and re-converging should reach a lower state.
    pub eigenvector_alpha: Array2<f64>,
    /// The β occ→virt rotation block. `None` for [`StabilityKind::RhfInternal`],
    /// where there is only one MO set.
    pub eigenvector_beta: Option<Array2<f64>>,
    /// Whether the Davidson eigensolve met `conv_thresh`. **If this is `false`
    /// the verdict is not evidence** — `lowest_eigenvalue` is an upper bound
    /// on the true λ_min (Rayleigh–Ritz is variational), so an unconverged
    /// NEGATIVE value still proves an instability, but an unconverged
    /// POSITIVE value proves nothing.
    pub converged: bool,
    /// Final Davidson residual norm ‖Hv − λv‖ for the returned eigenpair.
    pub residual: f64,
    /// Davidson iterations taken.
    pub iterations: usize,
    /// The magnitude below which `lowest_eigenvalue` is not distinguishable
    /// from zero, given the eigensolver residual and the integral threshold.
    /// A `|λ_min|` at or below this is MARGINAL stability, reported as such
    /// rather than tuned into a clean verdict.
    pub noise_floor: f64,
}

/// The four distinguishable outcomes of a stability analysis.
///
/// Exists because [`StabilityResult::is_stable`] is a `bool` over a THREE-valued
/// question (plus a fourth "we do not know"), and a `bool` cannot carry that:
/// it is really "not proven unstable", so it reads `true` for a negative
/// `lowest_eigenvalue` anywhere in the marginal band. Any caller that branches
/// on it directly will eventually report "stable" for a saddle that happens to
/// be shallow. Branching on this enum instead makes that misreading
/// unrepresentable — the match is total and each arm names what it means.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StabilityVerdict {
    /// λ_min is positive and resolvably above the noise floor: a minimum with
    /// respect to the rotations [`StabilityResult::kind`] covers, and only
    /// those.
    Stable,
    /// λ_min is negative and resolvably below the noise floor: a SADDLE POINT.
    /// The eigenvector is a downhill direction.
    Unstable,
    /// `|λ_min|` is at or below the noise floor. Neither a proven minimum nor a
    /// proven saddle — the calculation cannot tell.
    Marginal,
    /// The eigensolve did not converge, so nothing is proven in either
    /// direction. (Rayleigh–Ritz is variational, so a negative
    /// `lowest_eigenvalue` here still proves an instability; it is a POSITIVE
    /// one that proves nothing. That asymmetry is why this is a separate arm
    /// and not folded into `Marginal`.)
    Indeterminate,
}

impl StabilityVerdict {
    /// Uppercase word for logs and warnings.
    pub fn label(self) -> &'static str {
        match self {
            StabilityVerdict::Stable => "STABLE",
            StabilityVerdict::Unstable => "UNSTABLE",
            StabilityVerdict::Marginal => "MARGINAL",
            StabilityVerdict::Indeterminate => "INDETERMINATE (eigensolver not converged)",
        }
    }
}

impl StabilityResult {
    /// True when `|λ_min|` sits at or below the numerical noise floor, i.e.
    /// the calculation cannot distinguish this point from marginally stable.
    /// Such a point is neither a proven minimum nor a proven saddle.
    pub fn is_marginal(&self) -> bool {
        self.lowest_eigenvalue.abs() <= self.noise_floor
    }

    /// The verdict, as the total four-way answer rather than the
    /// easily-misread [`is_stable`](Self::is_stable) bool. **This is what
    /// callers should branch on.**
    ///
    /// Deliberately derived from `converged`, `noise_floor` and the SIGN of
    /// `lowest_eigenvalue` — NOT from `is_stable`, so that a caller reading
    /// this can never inherit that field's marginal-band ambiguity.
    pub fn verdict(&self) -> StabilityVerdict {
        if !self.converged {
            StabilityVerdict::Indeterminate
        } else if self.is_marginal() {
            StabilityVerdict::Marginal
        } else if self.lowest_eigenvalue > 0.0 {
            StabilityVerdict::Stable
        } else {
            StabilityVerdict::Unstable
        }
    }

    /// One-line human summary for logs.
    pub fn summary(&self) -> String {
        let verdict = self.verdict().label();
        format!(
            "{}: {} — lambda_min = {:+.6e} Ha (noise floor {:.1e}), \
             Davidson residual {:.2e} in {} iters",
            self.kind.label(),
            verdict,
            self.lowest_eigenvalue,
            self.noise_floor,
            self.residual,
            self.iterations
        )
    }
}

/// Knobs for the stability eigensolve.
#[derive(Debug, Clone)]
pub struct StabilityConfig {
    /// Davidson residual-norm convergence threshold.
    pub conv_thresh: f64,
    /// Maximum Davidson subspace size before collapse/restart.
    pub max_subspace: usize,
    /// Hard Davidson iteration cap.
    pub max_iter: usize,
    /// Number of unit vectors in the initial block seed (the `n_seed` smallest
    /// Hessian-diagonal entries), and the number of Ritz vectors retained on a
    /// subspace collapse.
    ///
    /// Defaults to 8. These are a CONVERGENCE AID, not a correctness
    /// mechanism: on water/STO-3G they take the default config from 19
    /// iterations to 7. They are also, on their own, actively harmful — with
    /// `n_block = 1` they are what hands the solver a block-local eigenvector
    /// and produce a converged wrong answer (see [`n_block`](Self::n_block)).
    /// Correctness comes from `n_block >= 2`, not from this knob.
    pub n_seed: usize,
    /// Number of Ritz roots tracked simultaneously (block size), and the number
    /// of preconditioned expansion vectors added per iteration.
    ///
    /// Defaults to 6, and **must stay >= 2** — this is the knob correctness
    /// rests on. The orbital Hessian is block diagonal in the molecule's
    /// irreps, so a single tracked root can converge EXACTLY (residual 6e-11,
    /// `converged = true`, in ONE iteration) inside a small symmetry block
    /// while the true λ_min sits in another. No residual check on that root
    /// can detect the miss, because the iteration genuinely converged.
    ///
    /// Measured on water/STO-3G UHF (true λ_min = +3.6256168663e-1):
    ///
    /// ```text
    ///   n_block=1 n_seed=1  -> +3.6256168663e-1  OK    (19 iters)
    ///   n_block=1 n_seed=2  -> +3.6887050762e-1  WRONG ( 1 iter)
    ///   n_block=1 n_seed=8  -> +3.6887050762e-1  WRONG ( 1 iter)
    ///   n_block=2 n_seed=1  -> +3.6256168663e-1  OK    (11 iters)
    ///   n_block=6 n_seed=8  -> +3.6256168663e-1  OK    ( 7 iters)
    /// ```
    ///
    /// Tracking a block and requiring the WHOLE block to converge is what makes
    /// the lowest root trustworthy: a root below the block leaves at least one
    /// block residual large. Pinned by
    /// `davidson_single_root_converges_inside_a_symmetry_block`.
    pub n_block: usize,
    /// Extra safety margin folded into [`StabilityResult::noise_floor`] on top
    /// of the achieved eigensolver residual. Defaults to `1e-6` Ha: the
    /// Hessian matvec rebuilds J/K at `integral_thresh`, so eigenvalues below
    /// roughly this magnitude are not meaningfully resolved.
    pub noise_floor: f64,
}

impl Default for StabilityConfig {
    fn default() -> Self {
        Self {
            conv_thresh: 1e-6,
            max_subspace: 40,
            max_iter: 100,
            n_seed: 8,
            n_block: 6,
            noise_floor: 1e-6,
        }
    }
}

/// Internal stability of a converged **UHF/UKS** solution.
///
/// Drives [`uhf_newton::hessian_matvec`](crate::uhf_newton::hessian_matvec)
/// with a Davidson extremal-eigenvalue solver over the packed (κ_α, κ_β)
/// rotation space. `inp` is the SAME [`UhfNewtonInputs`] the Newton solver
/// builds — MO coefficients, MO-basis Fock matrices, occupations, `k_mix_sr`
/// and the optional `fxc` response closure — evaluated at the converged point.
///
/// Passing `fxc: None` on a KS reference silently analyses the WRONG operator
/// (the HF Hessian at the KS density), so a UKS caller must supply the same
/// f_xc closure the UKS Newton path uses.
///
/// # Errors
///
/// Returns [`FerricError`] if the rotation space is empty (no occupied or no
/// virtual orbitals in either spin), or if a J/K build inside the matvec fails.
pub fn uhf_internal_stability(
    ctx: &ParallelContext,
    inp: &UhfNewtonInputs,
    cfg: &StabilityConfig,
) -> Result<StabilityResult, FerricError> {
    let n = inp.c_a.nrows();
    let (na, nb) = (inp.nocc_a, inp.nocc_b);
    let (nva, nvb) = (n - na, n - nb);
    let dim_a = nva * na;
    let dim_b = nvb * nb;
    let dim = dim_a + dim_b;
    if dim == 0 {
        return Err(FerricError::General(
            "UHF stability analysis: the occ→virt rotation space is empty (no \
             occupied or no virtual orbitals in either spin); there is nothing \
             to rotate, so stability is undefined"
                .to_string(),
        ));
    }

    // EnginePool is geometry/basis-only — build once, reuse for every matvec
    // (mirrors uhf_newton_step, which does the same for its PCG loop).
    let pool = EnginePool::new(inp.bounds.op, inp.prep, 1e-14)?;

    // Diagonal preconditioner: the orbital-energy gap (F_aa - F_ii) per spin.
    // This is the dominant part of the Hessian diagonal — the reason Davidson
    // is the right solver here (see module docs).
    let fa_diag: Vec<f64> = (0..n).map(|i| inp.f_a_mo[(i, i)]).collect();
    let fb_diag: Vec<f64> = (0..n).map(|i| inp.f_b_mo[(i, i)]).collect();
    let mut diag = Vec::with_capacity(dim);
    for a in na..n {
        for i in 0..na {
            diag.push(fa_diag[a] - fa_diag[i]);
        }
    }
    for a in nb..n {
        for i in 0..nb {
            diag.push(fb_diag[a] - fb_diag[i]);
        }
    }

    let matvec = |v: &[f64]| -> Result<Vec<f64>, FerricError> {
        let k_a = Array2::from_shape_vec((nva, na), v[..dim_a].to_vec())
            .expect("alpha rotation block shape is (nvirt_a, nocc_a) by construction");
        let k_b = Array2::from_shape_vec((nvb, nb), v[dim_a..].to_vec())
            .expect("beta rotation block shape is (nvirt_b, nocc_b) by construction");
        let (h_a, h_b) = crate::uhf_newton::hessian_matvec(ctx, inp, &k_a, &k_b, &pool)?;
        let mut out = Vec::with_capacity(dim);
        out.extend(h_a.iter().copied());
        out.extend(h_b.iter().copied());
        Ok(out)
    };

    let dav = davidson_lowest(dim, matvec, &diag, cfg)?;
    let (va, vb) = dav.eigenvector.split_at(dim_a);
    Ok(StabilityResult {
        kind: StabilityKind::UhfInternal,
        lowest_eigenvalue: dav.eigenvalue,
        is_stable: dav.eigenvalue > -noise_floor_of(cfg, dav.residual),
        eigenvector_alpha: Array2::from_shape_vec((nva, na), va.to_vec())
            .expect("alpha eigenvector block shape is (nvirt_a, nocc_a) by construction"),
        eigenvector_beta: Some(
            Array2::from_shape_vec((nvb, nb), vb.to_vec())
                .expect("beta eigenvector block shape is (nvirt_b, nocc_b) by construction"),
        ),
        converged: dav.converged,
        residual: dav.residual,
        iterations: dav.iterations,
        noise_floor: noise_floor_of(cfg, dav.residual),
    })
}

/// Internal stability of a converged **RHF/RKS** solution (singlet channel).
///
/// Drives [`rhf_newton::hessian_matvec`](crate::rhf_newton::hessian_matvec),
/// whose rotation is applied identically to α and β (it builds
/// `δD = 2·δD_single`). That is the RHF→RHF singlet Hessian: it detects
/// instabilities *within* the closed-shell ansatz.
///
/// It is BLIND to RHF→UHF spin-symmetry breaking — the classic stretched-H₂
/// instability — which lives in the triplet channel (κ_α = −κ_β) where the
/// Coulomb response cancels. See the module docs: that is a different
/// operator, not a different caller of this one, and it is not implemented.
///
/// # Errors
///
/// Returns [`FerricError`] if the rotation space is empty, or if a J/K build
/// inside the matvec fails.
pub fn rhf_internal_stability(
    ctx: &ParallelContext,
    inp: &RhfNewtonInputs,
    cfg: &StabilityConfig,
) -> Result<StabilityResult, FerricError> {
    let n = inp.c.nrows();
    let no = inp.nocc;
    let nv = n - no;
    let dim = nv * no;
    if dim == 0 {
        return Err(FerricError::General(
            "RHF stability analysis: the occ→virt rotation space is empty (no \
             occupied or no virtual orbitals); there is nothing to rotate, so \
             stability is undefined"
                .to_string(),
        ));
    }

    let pool = EnginePool::new(inp.bounds.op, inp.prep, 1e-14)?;

    let f_diag: Vec<f64> = (0..n).map(|i| inp.f_mo[(i, i)]).collect();
    let mut diag = Vec::with_capacity(dim);
    for a in no..n {
        for i in 0..no {
            diag.push(f_diag[a] - f_diag[i]);
        }
    }

    let matvec = |v: &[f64]| -> Result<Vec<f64>, FerricError> {
        let k = Array2::from_shape_vec((nv, no), v.to_vec())
            .expect("rotation block shape is (nvirt, nocc) by construction");
        let h = crate::rhf_newton::hessian_matvec(ctx, inp, &k, &pool)?;
        Ok(h.iter().copied().collect())
    };

    let dav = davidson_lowest(dim, matvec, &diag, cfg)?;
    Ok(StabilityResult {
        kind: StabilityKind::RhfInternal,
        lowest_eigenvalue: dav.eigenvalue,
        is_stable: dav.eigenvalue > -noise_floor_of(cfg, dav.residual),
        eigenvector_alpha: Array2::from_shape_vec((nv, no), dav.eigenvector)
            .expect("eigenvector block shape is (nvirt, nocc) by construction"),
        eigenvector_beta: None,
        converged: dav.converged,
        residual: dav.residual,
        iterations: dav.iterations,
        noise_floor: noise_floor_of(cfg, dav.residual),
    })
}

/// The noise floor is the larger of the user's configured floor and the
/// residual the eigensolve actually achieved: an eigenvalue cannot be resolved
/// more finely than the residual of the vector it came from.
fn noise_floor_of(cfg: &StabilityConfig, residual: f64) -> f64 {
    cfg.noise_floor.max(residual)
}

// ---------------------------------------------------------------------------
// Davidson for the LOWEST eigenpair of a symmetric operator given as a matvec.
// ---------------------------------------------------------------------------

/// Lowest eigenpair from a fallible matvec, with residual reporting.
#[derive(Debug, Clone)]
pub struct LowestEigenpair {
    pub eigenvalue: f64,
    pub eigenvector: Vec<f64>,
    pub residual: f64,
    pub iterations: usize,
    pub converged: bool,
}

/// Davidson iteration for the algebraically LOWEST eigenvalue of a symmetric
/// operator supplied as `matvec: v ↦ Hv`.
///
/// This is a third Davidson in the workspace, and that is deliberate rather
/// than sloppy: `ferric_rpa::davidson` cannot be imported (ferric-rpa depends
/// on ferric-scf), the in-crate [`davidson_local`](crate::davidson_local) copy
/// takes a *projected-matrix* closure shaped for the dielectric operator
/// rather than a plain `Hv`, and `ferric_ci::davidson` lives in a crate that
/// depends on this one. What is genuinely new here is that the matvec is
/// **fallible** — a J/K build inside the orbital Hessian can fail, and that
/// must propagate as an `Err`, never be unwrapped inside the iteration.
///
/// The returned pair is variational (Rayleigh–Ritz over the built subspace),
/// so `eigenvalue` is an upper bound on the true λ_min even when `converged`
/// is false. That asymmetry is what makes an unconverged NEGATIVE verdict
/// still meaningful while an unconverged POSITIVE one is not.
///
/// # Errors
///
/// Propagates any error from `matvec`, and errors if the subspace cannot be
/// extended (which would otherwise silently return a stale eigenpair).
pub fn davidson_lowest<F>(
    dim: usize,
    matvec: F,
    diag: &[f64],
    cfg: &StabilityConfig,
) -> Result<LowestEigenpair, FerricError>
where
    F: Fn(&[f64]) -> Result<Vec<f64>, FerricError>,
{
    if dim == 0 {
        return Err(FerricError::General(
            "Davidson: empty eigenvalue problem".to_string(),
        ));
    }
    assert_eq!(diag.len(), dim, "Davidson: diagonal length must equal dim");

    // Exact 1x1 case: the matvec IS the eigenvalue.
    if dim == 1 {
        let hv = matvec(&[1.0])?;
        return Ok(LowestEigenpair {
            eigenvalue: hv[0],
            eigenvector: vec![1.0],
            residual: 0.0,
            iterations: 0,
            converged: true,
        });
    }

    let max_sub = cfg.max_subspace.max(4).min(dim);

    // BLOCK seed on the `n_seed` smallest diagonal elements, not a single unit
    // vector.
    //
    // A single-vector seed is NOT sufficient here, and this was caught by the
    // water/STO-3G anchor before any HeNe⁺ number was believed: seeded on one
    // unit vector, Davidson converged to residual 6.6e-11 on λ = +3.6887e-1
    // while the true lowest eigenvalue was +3.6256e-1 — a confidently
    // converged WRONG eigenpair. The cause is that the lowest mode of a
    // closed-shell system run through the UHF path is α/β-ANTISYMMETRIC
    // (κ_β = −κ_α, the triplet-like direction), and the crude
    // orbital-gap preconditioner `F_aa − F_ii` is a poor enough approximation
    // to the true Hessian diagonal (20.85/1.87/1.22 vs 20.08/1.36/0.72 on
    // that system) that a single seed plus one-vector restarts can lock onto
    // the second root instead. A block seed spanning the several smallest gaps
    // in BOTH spin blocks makes the correct root reachable from the start.
    let n_seed = cfg.n_seed.max(1).min(max_sub).min(dim);
    let mut order: Vec<usize> = (0..dim).collect();
    order.sort_by(|&a, &b| {
        diag[a]
            .partial_cmp(&diag[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut v_basis: Vec<Array1<f64>> = Vec::with_capacity(max_sub);

    // SYMMETRY-BREAKING SEED, first and deliberately.
    //
    // The orbital Hessian of a symmetric molecule is BLOCK DIAGONAL in the
    // irreps of the point group, and a unit-vector seed lies inside exactly
    // ONE block. A dense pseudo-random vector has nonzero overlap with every
    // block, so it is the vector that makes every root reachable at all. It is
    // generated from a fixed xorshift64 seed so the analysis stays
    // deterministic and reproducible — an eigensolver whose verdict depended
    // on a clock would not be evidence.
    //
    // NOTE (measured, not assumed): the dense seed ALONE is not sufficient.
    // With `n_block = 1`, adding the smallest-diagonal unit vectors below puts
    // a block-local eigenvector into the starting subspace, and the iteration
    // converges onto it in ONE step with residual 6e-11 and `converged = true`
    // — on the wrong root. See the `n_block` doc and the regression test
    // `davidson_single_root_converges_inside_a_symmetry_block`, which carries
    // the full n_block × n_seed sweep. Correctness comes from `n_block >= 2`;
    // this dense vector and the unit vectors below are convergence aids.
    let mut rng: u64 = 0x2545_F491_4F6C_DD1D;
    let mut dense = Array1::<f64>::zeros(dim);
    for i in 0..dim {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        dense[i] = ((rng >> 11) as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0;
    }
    let dn = dense.dot(&dense).sqrt();
    v_basis.push(&dense / dn);

    // Then the smallest-diagonal unit vectors, orthogonalized against it.
    // These carry the physics guess (the smallest orbital-energy gaps are
    // where an instability usually lives) and accelerate convergence; the
    // dense vector above is what makes correctness independent of them.
    for &idx in order.iter().take(n_seed) {
        if v_basis.len() >= max_sub {
            break;
        }
        let mut e = Array1::<f64>::zeros(dim);
        e[idx] = 1.0;
        for v in &v_basis {
            let ov = v.dot(&e);
            e.scaled_add(-ov, v);
        }
        let en = e.dot(&e).sqrt();
        if en > 1e-8 {
            v_basis.push(&e / en);
        }
    }
    let mut hv_basis: Vec<Array1<f64>> = Vec::new();

    let mut theta = diag[order[0]];
    let mut best_vec = v_basis[0].to_vec();
    let mut resid_norm = f64::INFINITY;
    let mut iters = 0usize;

    for it in 0..cfg.max_iter {
        iters = it + 1;

        // H-images for any newly added basis vectors.
        while hv_basis.len() < v_basis.len() {
            let v = v_basis[hv_basis.len()].to_vec();
            hv_basis.push(Array1::from(matvec(&v)?));
        }

        let m = v_basis.len();

        // Projected (symmetric) subspace matrix. Symmetrized explicitly so
        // round-off asymmetry cannot make `eigh` complain or drift.
        let mut sub = Array2::<f64>::zeros((m, m));
        for i in 0..m {
            for j in 0..m {
                sub[(i, j)] = v_basis[i].dot(&hv_basis[j]);
            }
        }
        let sub_sym = 0.5 * (&sub + &sub.t());

        let (evals, evecs) = sub_sym
            .eigh(UPLO::Lower)
            .map_err(|e| FerricError::Lapack(format!("Davidson subspace eigh: {e}")))?;

        // Ritz roots in ascending order.
        let mut ord: Vec<usize> = (0..m).collect();
        ord.sort_by(|&a, &b| {
            evals[a]
                .partial_cmp(&evals[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // BLOCK of the lowest `n_block` Ritz pairs, not just the lowest one.
        //
        // Tracking a block is what makes the LOWEST root trustworthy, and this
        // is the second failure the water/STO-3G anchor caught. The orbital
        // Hessian is block diagonal in the molecule's irreps, and a Ritz vector
        // can lie ENTIRELY inside one small block — `H·e_4` on that system has
        // support only on {4, 14}, a closed 2×2 block. Single-root Davidson
        // then converges to residual 6.1e-11 with `converged = true` on
        // +3.6887e-1 while the true λ_min is +3.6256e-1 in a different block:
        // it is a genuinely converged eigenpair, just not the lowest one, so
        // no residual check on that root alone can detect the miss. Expanding
        // a block keeps several roots alive simultaneously, and convergence is
        // declared only when EVERY root in the block has converged — at which
        // point the lowest of them really is the lowest, because a missed root
        // below them would leave one of the block residuals large.
        let n_block = cfg.n_block.max(1).min(m);
        theta = evals[ord[0]];

        let mut residuals: Vec<Array1<f64>> = Vec::with_capacity(n_block);
        let mut worst_resid = 0.0f64;
        for (bi, &root) in ord.iter().take(n_block).enumerate() {
            let mut x = Array1::<f64>::zeros(dim);
            let mut hx = Array1::<f64>::zeros(dim);
            for i in 0..m {
                let c = evecs[(i, root)];
                x.scaled_add(c, &v_basis[i]);
                hx.scaled_add(c, &hv_basis[i]);
            }
            let r = &hx - &(evals[root] * &x);
            let rn = r.dot(&r).sqrt();
            if bi == 0 {
                best_vec = x.to_vec();
                resid_norm = rn;
            }
            worst_resid = worst_resid.max(rn);
            residuals.push(r);
        }

        // Converged only when the WHOLE block has converged (see above), OR
        // when the subspace spans the entire space, in which case the
        // Rayleigh-Ritz values ARE the exact eigenvalues and the lowest one is
        // exact regardless of any higher root's residual.
        //
        // `worst_resid` is deliberately the block-wide worst, not the lowest
        // root's own residual: a symmetry-trapped root converges tightly while
        // the true λ_min is still undiscovered, so the lowest root's residual
        // alone is not evidence. Requiring the block means an undiscovered root
        // below the block keeps at least one block residual large.
        if worst_resid < cfg.conv_thresh || m >= dim {
            return Ok(LowestEigenpair {
                eigenvalue: theta,
                eigenvector: best_vec,
                residual: resid_norm,
                iterations: iters,
                converged: true,
            });
        }

        // Subspace full: collapse onto the LOWEST FEW Ritz vectors and restart.
        // Retaining several (not just the best one) keeps the restarted basis
        // from discarding the partially converged neighbours of the target
        // root, which is how a one-vector collapse can strand the iteration on
        // a higher root.
        // Collapse only when the basis is genuinely at the cap AND cannot be
        // the whole space (if m == dim the Ritz values are already exact and
        // the block-convergence branch below handles it). `keep` must leave
        // room for at least one new expansion vector, otherwise the iteration
        // collapses and regrows the same subspace forever: on water/STO-3G RHF
        // the rotation space is only 2×5 = 10 dimensional, and a collapse rule
        // of `m + n_block > max_sub` with `keep = max(n_block, n_seed) = 8`
        // pinned m at 8 for all 100 iterations while the lowest root sat
        // converged at 5e-16 and a higher block root stalled at 5.3e-2.
        if m >= max_sub && m < dim {
            let keep = n_block.min(m).min(max_sub.saturating_sub(n_block).max(1));
            let idx = &ord;
            let mut new_basis: Vec<Array1<f64>> = Vec::with_capacity(keep);
            for &r in idx.iter().take(keep) {
                let mut y = Array1::<f64>::zeros(dim);
                for i in 0..m {
                    y.scaled_add(evecs[(i, r)], &v_basis[i]);
                }
                // Re-orthonormalize against what is already retained.
                for v in &new_basis {
                    let ov = v.dot(&y);
                    y.scaled_add(-ov, v);
                }
                let ny = y.dot(&y).sqrt();
                if ny > 1e-10 {
                    new_basis.push(&y / ny);
                }
            }
            v_basis = new_basis;
            hv_basis.clear();
            continue;
        }

        // Preconditioned expansion vector per block root:
        // t_k = r_k / (diag − θ_k), guarded against a near-zero denominator.
        let mut added = 0usize;
        for (bi, &root) in ord.iter().take(n_block).enumerate() {
            if v_basis.len() >= max_sub {
                break;
            }
            let th = evals[root];
            let r = &residuals[bi];
            let mut t = Array1::<f64>::zeros(dim);
            for i in 0..dim {
                let d = diag[i] - th;
                t[i] = if d.abs() < 1e-8 {
                    r[i] / 1e-8
                } else {
                    r[i] / d
                };
            }
            // Modified Gram-Schmidt against the current basis (twice, for
            // numerical orthogonality on an ill-conditioned subspace).
            for _ in 0..2 {
                for v in &v_basis {
                    let ov = v.dot(&t);
                    t.scaled_add(-ov, v);
                }
            }
            let tn = t.dot(&t).sqrt();
            if tn > 1e-8 {
                v_basis.push(&t / tn);
                added += 1;
            }
        }
        if added == 0 {
            // No linearly independent expansion direction remains. Two very
            // different situations land here and they must NOT be conflated:
            //
            //  * the subspace already spans the whole space (m == dim) — then
            //    the Rayleigh-Ritz values ARE the exact eigenvalues and the
            //    lowest one is exact, however large some higher block root's
            //    residual happens to look;
            //  * the block has stagnated in a proper subspace — then the
            //    lowest root is NOT proven lowest, and `converged` must stay
            //    false so the caller cannot read the verdict as evidence.
            //
            // Without this distinction a small system (water/STO-3G RHF,
            // dim = 35) burns every iteration re-deriving the same stalled
            // block and then reports `converged = false` on an eigenvalue that
            // is in fact exact.
            if m >= dim {
                return Ok(LowestEigenpair {
                    eigenvalue: theta,
                    eigenvector: best_vec,
                    residual: resid_norm,
                    iterations: iters,
                    converged: true,
                });
            }
            break;
        }
    }

    Ok(LowestEigenpair {
        eigenvalue: theta,
        eigenvector: best_vec,
        residual: resid_norm,
        iterations: iters,
        converged: resid_norm < cfg.conv_thresh,
    })
}

// ---------------------------------------------------------------------------
// Post-SCF wiring: the opt-in check the solvers run after convergence.
// ---------------------------------------------------------------------------

/// Why a requested stability check did not produce a verdict.
///
/// This exists so a skip is never silent. `RhfConfig::check_stability` is
/// opt-in, so a user who set it and got `ScfResult::stability == None` is owed
/// an explanation of which precondition failed — the alternative (analysing
/// whatever operator happens to be available) is exactly the failure mode this
/// module was written to prevent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StabilitySkip {
    /// ROHF/ROKS reference. The Roothaan open-shell Hessian is a THIRD operator
    /// (`rohf_newton::hessian_matvec`, one MO set with closed/open/virtual
    /// blocks and Roothaan coupling), not a special case of either implemented
    /// one. Running `uhf_internal_stability` on it would analyse a UHF Hessian
    /// at MO coefficients that are not a UHF stationary point — a wrong-operator
    /// verdict that would look authoritative. Skipped deliberately.
    Rohf,
    /// Range-separated hybrid (ω ≠ 0). The Hessian matvec builds its exchange
    /// response from the plain Coulomb `build_jk`, so the LR/SR split of the
    /// converged Fock is not reproduced in the response. The Newton path gates
    /// on `k_mix.omega == 0.0` for the same reason.
    RangeSeparated,
    /// Meta-GGA functional. There is no τ-dependent f_xc kernel in this
    /// workspace (`xc_is_metagga` gates the Newton f_xc path for the same
    /// reason), so the XC response term of the Hessian cannot be formed.
    MetaGga,
    /// A per-iteration Fock modifier was active (the cDFT path,
    /// `solve_uhf_fockmod` with `fock_mod = Some(..)`). Such a run converges
    /// the CONSTRAINED Fock, so the Brillouin condition that holds at its exit
    /// is `F^constrained_{ai} = 0`; the bare UHF gradient is NOT zero there.
    /// The implemented Hessian is the bare UHF one, and its lowest eigenvalue
    /// at a non-stationary point of that same energy is not a stability
    /// verdict — a constrained diabat is *meant* not to be an unconstrained
    /// minimum. Skipped rather than answering a question nobody asked.
    FockModified,
    /// The eigensolve itself returned an error (empty rotation space, or a
    /// failed J/K build inside the matvec). The message is printed at the call
    /// site; SCF is not failed.
    AnalysisFailed,
}

impl StabilitySkip {
    /// The reason text printed to stderr when a requested check is skipped.
    pub fn reason(self) -> &'static str {
        match self {
            StabilitySkip::Rohf => {
                "the reference is ROHF/ROKS, whose Roothaan orbital Hessian is a \
                 different operator from both implemented ones (UHF-internal and \
                 RHF-internal); analysing either of those here would give a \
                 wrong-operator verdict"
            }
            StabilitySkip::RangeSeparated => {
                "the functional is range-separated (omega != 0) and the orbital-Hessian \
                 matvec builds its exchange response from the plain Coulomb kernel, so \
                 it does not reproduce the converged Fock's SR/LR split"
            }
            StabilitySkip::MetaGga => {
                "the functional is a meta-GGA and no tau-dependent f_xc kernel exists in \
                 this workspace, so the XC response term of the orbital Hessian cannot \
                 be formed"
            }
            StabilitySkip::FockModified => {
                "a per-iteration Fock modifier (cDFT constraint) was active, so the converged \
                 state is stationary for the CONSTRAINED Fock and not for the bare UHF one that \
                 the implemented Hessian belongs to"
            }
            StabilitySkip::AnalysisFailed => "the stability eigensolve returned an error",
        }
    }
}

/// Decide whether a KS/HF reference can be analysed at all, given the
/// functional's range-separation and meta-GGA status.
///
/// Returns `Ok(())` when the reference is analysable with the operators that
/// exist, or `Err(skip)` naming the precondition that failed. Mirrors EXACTLY
/// the gates the Newton paths use (`k_mix.omega == 0.0` and
/// `!rohf::xc_is_metagga(xc)`), because the stability analysis drives the same
/// `hessian_matvec` those gates protect.
///
/// It deliberately does NOT gate on `xc.is_some()`: an LDA/GGA/hybrid KS
/// reference IS analysable, provided the caller threads the same `fxc`
/// response closure the Newton path builds. Passing `fxc: None` there would
/// silently analyse the HF Hessian at the KS density — the trap this whole
/// function exists to make impossible to fall into by accident, since a caller
/// that gets `Ok(())` for a KS reference is thereby committed to supplying the
/// kernel.
pub fn ks_reference_is_analysable(xc: Option<&str>, omega: f64) -> Result<(), StabilitySkip> {
    if omega != 0.0 {
        return Err(StabilitySkip::RangeSeparated);
    }
    if crate::rohf::xc_is_metagga(xc) {
        return Err(StabilitySkip::MetaGga);
    }
    Ok(())
}

/// Print the post-SCF stability verdict.
///
/// DIAGNOSTIC ONLY, by design: an instability is information, never an error.
/// A deliberately-unstable state (a cDFT diabat, a MOM excited state) is a
/// legitimate thing to compute, so this prints and returns — it never fails
/// the SCF. See the module docs.
///
/// An UNSTABLE or INDETERMINATE verdict goes to stderr as a warning naming
/// λ_min and the remedy; a STABLE or MARGINAL one goes to stderr as a plain
/// informational line only when `verbose`, so a quiet run stays quiet.
/// Branches on [`StabilityResult::verdict`], NOT on
/// [`StabilityResult::is_stable`] — that field is "not proven unstable" and
/// reads `true` for a marginal-NEGATIVE λ_min, so branching on it here would
/// make this function print "stable" for a shallow saddle. The `match` is
/// exhaustive so a future fifth outcome is a compile error, not a silent
/// fall-through into the quiet arm.
pub fn report_stability(res: &StabilityResult, verbose: bool) {
    match res.verdict() {
        StabilityVerdict::Indeterminate => eprintln!(
            "SCF stability WARNING: {}\n  The eigensolve did NOT converge, so this verdict is not \
             evidence of stability. Because Rayleigh-Ritz is variational, lambda_min is an UPPER \
             bound on the true value: a negative value here still proves an instability, a \
             positive one proves nothing. Raise StabilityConfig::max_iter or max_subspace.",
            res.summary()
        ),
        StabilityVerdict::Marginal => eprintln!(
            "SCF stability WARNING: {}\n  |lambda_min| = {:.3e} is at or below the numerical noise \
             floor {:.1e}, so this point is NOT distinguishable from marginally stable and is \
             neither a proven minimum nor a proven saddle. Note the sign is {}: treat this as \
             'unknown', not as 'stable'. Tighten StabilityConfig::conv_thresh / noise_floor, or \
             the SCF's own density_conv, to resolve it.",
            res.summary(),
            res.lowest_eigenvalue.abs(),
            res.noise_floor,
            if res.lowest_eigenvalue < 0.0 {
                "NEGATIVE"
            } else {
                "positive"
            }
        ),
        StabilityVerdict::Stable => {
            if verbose {
                eprintln!("SCF stability: {}", res.summary());
            }
        }
        StabilityVerdict::Unstable => eprintln!(
            "SCF stability WARNING: {}\n  This converged solution is a SADDLE POINT, not a \
             minimum: lambda_min = {:+.6e} Ha < 0 means an orbital rotation exists that LOWERS \
             the energy. The SCF is stationary (the gradient vanishes at a saddle too), so \
             nothing else could have detected this.\n  REMEDY: re-converge from a guess rotated \
             along the returned Hessian eigenvector (ScfResult::stability -> eigenvector_alpha / \
             eigenvector_beta). A measured caveat: small steps fall straight back into the \
             saddle's DIIS basin, so use a rotation of order 1 radian, not 0.1.\n  SCOPE: this \
             verdict covers {} only; it says nothing about the rotations that space excludes.",
            res.summary(),
            res.lowest_eigenvalue,
            res.kind.label()
        ),
    }
}
