//! AURORA-SCF: auxiliary-curvature unified Riemannian orbital-response acceleration.
//!
//! An opt-in, closed-shell (RHF/RKS) SCF accelerator that replaces the DIIS
//! extrapolation step with a quasi-Newton orbital rotation. It follows
//! Bao & Shi, *"Beyond the Four-Decade DIIS Default: Auxiliary-Curvature
//! Acceleration of Self-Consistent-Field Calculations"*, arXiv:2608.07354
//! (2026-08-07), Supplementary Information Eqs. (S1)–(S19).
//!
//! # The idea
//!
//! Energy and gradient come **only** from the target Hamiltonian — the one the
//! user asked for. Most orbital *curvature*, however, comes from an independent
//! and much cheaper model: the full Coulomb/exchange orbital response evaluated
//! with STO-3G as the density-fitting auxiliary basis. The mismatch between that
//! model curvature and the true target curvature is learned on the fly from
//! exact target-level secant pairs through a transported, Powell-damped L-BFGS
//! history whose "initial inverse Hessian" is not a scalar but a matrix-free
//! PCG solve against the auxiliary operator.
//!
//! Because the target gradient alone decides convergence, the accelerator
//! changes the **path** to the fixed point, never the fixed point itself.
//!
//! # Scope and honesty
//!
//! Implemented here: closed-shell RHF and hybrid RKS, direct or density-fitted
//! target Fock builds — whatever [`crate::rhf::solve_rhf`] is already configured
//! to use.
//!
//! **Not** implemented, and not silently approximated:
//!
//! * `D_k^xc`, the "inexpensive Kohn-Sham diagonal correction" of Eq. (2). The
//!   paper names this term in its schematic base operator and never defines it:
//!   no formula appears in the text or the Supplementary Information, and the
//!   authors' own minimal reference implementation lists "an auxiliary LDA grid
//!   kernel for DFT" among its intentionally omitted features, saying "the
//!   missing XC curvature is learned through macro-step L-BFGS secant updates".
//!   This port therefore carries no XC curvature either. The consequence is
//!   measured, not assumed: on references with little exact exchange the model
//!   understates curvature and the step overshoots, so
//!   [`crate::aurora::AuroraConfig::allow_low_exchange_ks`] keeps them on DIIS by default.
//! * Open-shell (UHF/ROHF/ROKS) tangent blocks — the paper's Eq. (S19) sketches
//!   them, but nothing here implements them.
//! * The optional SR1 correction, the Coulomb-only, diagonal and semiempirical
//!   curvature models, and the adaptive DIIS/AURORA crossover policy of §3.2.
//! * Trial rejection with trust-radius reduction. The main text charges rejected
//!   trials to the reported cost; the minimal reference (and this port) instead
//!   clip to a fixed radius and rely on the history-restart test. So this is a
//!   trust REGION in name only — it is a fixed step-length clip.
//!
//! Default is OFF. With [`crate::aurora::AuroraConfig::enabled`] `false` the SCF driver is
//! byte-for-byte the code path it was before this module existed.
//!
//! # Layout
//!
//! * [`crate::aurora::AuxCurvature`] — the STO-3G auxiliary-response operator, Eqs. (S7)–(S9).
//! * [`crate::aurora::AuroraState`] — variable-base two-loop recursion,
//!   Eqs. (S12)–(S16), with Powell damping Eqs. (S10)–(S11).
//! * [`crate::aurora::geodesic_rotation`] — exact thin-SVD orbital exponential, Eq. (S18).
//! * [`crate::aurora::transport_tangent`] — Grassmann vector transport for the secant pairs.

use ferric_core::error::FerricError;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ndarray::{s, Array1, Array2, Array3, ArrayView2};
use ndarray_linalg::{Eigh, SVD, UPLO};

/// Counts target-Hamiltonian J/K (effective-potential) builds performed by
/// [`crate::rhf::solve_rhf`].
///
/// The paper's headline claim is a reduction in *target* builds, so measuring it
/// needs a counter that records exactly one tick per target Fock build and is
/// blind to auxiliary-curvature work — which is the point of the method and must
/// not be charged here. Auxiliary applications are tallied separately by
/// [`AuroraState::operator_calls`].
///
/// Test instrumentation: a single relaxed atomic increment per SCF iteration.
/// Because it is process-global, tests that read it must serialize (the crate's
/// existing test lock) or compare deltas taken around a single solve.
#[doc(hidden)]
pub static TARGET_JK_BUILDS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// Read the current target J/K build count.
pub fn target_jk_builds() -> usize {
    TARGET_JK_BUILDS.load(std::sync::atomic::Ordering::Relaxed)
}

thread_local! {
    /// Accelerated steps taken on this thread; see [`aurora_steps_on_this_thread`].
    static AURORA_STEPS_THIS_THREAD: std::cell::Cell<usize> =
        const { std::cell::Cell::new(0) };
}

/// Number of AURORA steps ([`AuroraState::step`]) taken on the CALLING thread
/// since it started.
///
/// Test instrumentation, so a reachability check can observe the
/// accelerator's ENGAGEMENT directly instead of inferring it from a side effect
/// such as a changed iteration count. That proxy broke once already: with the
/// DF-J metric solved by Cholesky, water/cc-pVDZ PBE converges in the same 10
/// iterations whether DIIS or AURORA drives it, so "the path changed" could no
/// longer be seen in the iteration count even though AURORA took every step.
///
/// Thread-local rather than a process-global atomic (unlike
/// [`TARGET_JK_BUILDS`]) so that tests running concurrently in one process
/// cannot pollute each other's deltas. It is sound for
/// [`crate::rhf::solve_rhf`] because the SCF loop, and so every `step` call,
/// runs on the thread that called `solve_rhf`. Take a delta around one solve.
#[doc(hidden)]
pub fn aurora_steps_on_this_thread() -> usize {
    AURORA_STEPS_THIS_THREAD.with(|c| c.get())
}

/// Tunable controls for the AURORA accelerator.
///
/// The defaults reproduce the "AURORA 0.10.0-style restricted settings" of the
/// reference implementation accompanying arXiv:2608.07354.
#[derive(Debug, Clone)]
pub struct AuroraConfig {
    /// Master switch. `false` (the default) leaves every existing SCF path
    /// bit-identical — the accelerator is never constructed and never consulted.
    pub enabled: bool,
    /// Trust radius applied to the very first accelerated step, Eq. (S17).
    pub first_trust_radius: f64,
    /// Trust radius for all later steps, Eq. (S17).
    pub trust_radius: f64,
    /// Maximum inner PCG iterations per macro step.
    pub inner_maxiter: usize,
    /// Minimum inner PCG iterations before the forcing term may stop the solve.
    pub inner_miniter: usize,
    /// Relative-residual forcing term for the inner PCG solve.
    pub inner_residual_tol: f64,
    /// Floor applied to the diagonal preconditioner, guarding small orbital gaps.
    pub diagonal_floor: f64,
    /// Number of transported secant pairs retained.
    pub history_size: usize,
    /// Pairs whose curvature falls below this are discarded rather than damped.
    pub secant_tol: f64,
    /// Rebuild the auxiliary MO tensors every this many macro steps.
    pub refresh_interval: usize,
    /// Linear-dependence threshold for the auxiliary 2-centre metric.
    pub metric_tol: f64,
    /// Auxiliary (curvature-model) basis name. The paper uses valence STO-3G.
    pub auxbasis: String,
    /// Delay the accelerator until the DIIS error has fallen below this value.
    /// `0.0` (the default) means engage from the first eligible iteration.
    pub trigger: f64,
    /// Engage on Kohn-Sham references whose exact-exchange fraction is below
    /// [`MIN_VALIDATED_EXCHANGE_FRACTION`], against the recommendation.
    ///
    /// The auxiliary model supplies Coulomb and exact-exchange curvature only.
    /// It carries **no** exchange-correlation curvature, because the paper's
    /// `D_k^xc` term (Eq. 2) is named but never defined — see the module
    /// documentation. For pure functionals the exchange response vanishes with
    /// `a_x`, the model then underestimates the true curvature, and the step
    /// overshoots. Measured on water/cc-pVDZ, PBE needed 200+ iterations at the
    /// paper's default trust radius of 0.20 against 87 for DIIS, while the same
    /// run at radius 0.10 took 45 — the signature of an overlong step, not of a
    /// wrong fixed point (all runs agreed on the energy to ~1e-8 Eh).
    ///
    /// STALE (2026-09-23): those counts were measured while DF-J still applied
    /// an explicit LU inverse of the RI metric, whose jitter floored the DIIS
    /// error and dragged every RI-J SCF out. With the Cholesky DF-J solve,
    /// DIIS converges water/cc-pVDZ PBE in 10 iterations and AURORA at radius
    /// 0.10 also in 10. The radius-0.20 figure and the overshoot argument have
    /// NOT been re-measured without that noise; the gate stays on principle
    /// (no `D_k^xc` curvature), not on the stale numbers.
    ///
    /// Default `false`: such references silently stay on DIIS rather than
    /// converge slowly. Set `true` only to measure this regime deliberately.
    pub allow_low_exchange_ks: bool,
}

/// Below this exact-exchange fraction the auxiliary curvature model is not
/// validated in this implementation, and AURORA declines to engage unless
/// [`AuroraConfig::allow_low_exchange_ks`] is set.
///
/// The threshold sits below B3LYP's 0.20 — which was measured to converge to the
/// right answer, if not always quickly — and above pure functionals' 0.0.
pub const MIN_VALIDATED_EXCHANGE_FRACTION: f64 = 0.1;

impl Default for AuroraConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            first_trust_radius: 0.28,
            trust_radius: 0.20,
            inner_maxiter: 3,
            inner_miniter: 1,
            inner_residual_tol: 0.08,
            diagonal_floor: 0.05,
            history_size: 4,
            secant_tol: 1.0e-10,
            refresh_interval: 4,
            metric_tol: 1.0e-10,
            auxbasis: "sto-3g".to_string(),
            trigger: 0.0,
            allow_low_exchange_ks: false,
        }
    }
}

/// Why the inner PCG solve stopped. Recorded for diagnostics only.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcgStop {
    /// Right-hand side was exactly zero.
    ZeroRhs,
    /// The relative-residual forcing term was met.
    ForcingTerm,
    /// A non-positive curvature direction was encountered.
    NonPositiveCurvature,
    /// The iteration cap was reached.
    Cap,
    /// The recursion broke down numerically.
    Breakdown,
}

/// Outcome of one inner PCG solve.
#[derive(Debug, Clone, Copy)]
pub struct PcgInfo {
    /// Iterations actually performed.
    pub iterations: usize,
    /// Final relative residual.
    pub relres: f64,
    /// Termination reason.
    pub stop: PcgStop,
    /// Whether a non-positive curvature direction was hit.
    pub negative_curvature: bool,
    /// Whether the diagonal fallback replaced a numerically null solution.
    pub diagonal_fallback: bool,
}

/// A transported exact secant pair in the current MO tangent frame, Eq. (S3).
#[derive(Debug, Clone)]
struct SecantPair {
    s: Array1<f64>,
    y: Array1<f64>,
}

/// A Powell-damped secant pair ready for the two-loop recursion, Eq. (S11).
#[derive(Debug, Clone)]
struct DampedPair {
    s: Array1<f64>,
    y_bar: Array1<f64>,
    rho: f64,
}

/// Exact thin-SVD orbital exponential, Eq. (S18).
///
/// For the occupied-virtual tangent block `x` (shape `(nvir, nocc)`) this
/// returns the full `(nmo, nmo)` rotation `exp[κ(X)]` where
///
/// ```text
/// κ(X) = [[0, -Xᵀ],
///         [X,   0]]
/// ```
///
/// With the thin SVD `X = U Σ Vᵀ` the exponential is available in closed form
/// and is *algebraically exact*, not a truncated series:
///
/// ```text
/// exp[κ(X)] = [[ I_o + V(cos Σ - I)Vᵀ,   -V sin Σ Uᵀ ],
///              [ U sin Σ Vᵀ,              I_v + U(cos Σ - I)Uᵀ ]]
/// ```
///
/// Orthonormality of the rotated orbitals therefore holds by construction, with
/// no a posteriori reorthogonalization.
pub fn geodesic_rotation(x: ArrayView2<f64>) -> Result<Array2<f64>, FerricError> {
    let nvir = x.nrows();
    let nocc = x.ncols();
    let nmo = nocc + nvir;
    let mut u_mo = Array2::<f64>::eye(nmo);
    if nvir == 0 || nocc == 0 {
        return Ok(u_mo);
    }
    let norm = x.iter().fold(0.0f64, |a, v| a + v * v).sqrt();
    if norm == 0.0 {
        return Ok(u_mo);
    }

    let xo = x.to_owned();
    let (uv_opt, sigma, vt_opt) = xo
        .svd(true, true)
        .map_err(|e| FerricError::Lapack(format!("AURORA geodesic thin SVD: {e}")))?;
    let uv = uv_opt.ok_or_else(|| FerricError::Lapack("AURORA SVD returned no U".into()))?;
    let vt = vt_opt.ok_or_else(|| FerricError::Lapack("AURORA SVD returned no Vt".into()))?;
    // `uv` is (nvir, k), `vt` is (k, nocc) with k = min(nvir, nocc).
    let k = sigma.len();

    // Uoo = I_o + V (cos Σ - I) Vᵀ
    for i in 0..nocc {
        for j in 0..nocc {
            let mut acc = 0.0;
            for t in 0..k {
                acc += vt[(t, i)] * (sigma[t].cos() - 1.0) * vt[(t, j)];
            }
            u_mo[(i, j)] += acc;
        }
    }
    // Uvv = I_v + U (cos Σ - I) Uᵀ
    for a in 0..nvir {
        for b in 0..nvir {
            let mut acc = 0.0;
            for t in 0..k {
                acc += uv[(a, t)] * (sigma[t].cos() - 1.0) * uv[(b, t)];
            }
            u_mo[(nocc + a, nocc + b)] += acc;
        }
    }
    // Uvo = U sin Σ Vᵀ   and   Uov = -V sin Σ Uᵀ
    for a in 0..nvir {
        for i in 0..nocc {
            let mut acc = 0.0;
            for t in 0..k {
                acc += uv[(a, t)] * sigma[t].sin() * vt[(t, i)];
            }
            u_mo[(nocc + a, i)] = acc;
            u_mo[(i, nocc + a)] = -acc;
        }
    }
    Ok(u_mo)
}

/// Grassmann vector transport of a tangent vector into the rotated MO frame.
///
/// Builds the antisymmetric `κ(v)`, conjugates it with the rotation `u_mo`, and
/// reads off the new occupied-virtual block. This is the transport
/// `𝒯ₖ` of Eq. (3) / Eq. (S19), exact for the matrix-exponential retraction
/// used by [`geodesic_rotation`].
pub fn transport_tangent(v: ArrayView2<f64>, u_mo: ArrayView2<f64>, nocc: usize) -> Array2<f64> {
    let nmo = u_mo.nrows();
    let nvir = nmo - nocc;
    let mut kappa = Array2::<f64>::zeros((nmo, nmo));
    for a in 0..nvir {
        for i in 0..nocc {
            kappa[(nocc + a, i)] = v[(a, i)];
            kappa[(i, nocc + a)] = -v[(a, i)];
        }
    }
    let t = u_mo.t().dot(&kappa).dot(&u_mo);
    t.slice(s![nocc.., ..nocc]).to_owned()
}

/// Guarded, diagonally preconditioned conjugate gradients for `A x = rhs`.
///
/// `A` is supplied matrix-free as a closure. The solve is deliberately *inexact*
/// — the forcing term `residual_tol` is loose (0.08 by default) because the
/// result is only the variable base of an L-BFGS recursion, not a Newton step
/// in its own right.
///
/// The non-positive-curvature test is **scale-relative**: an absolute floor
/// would falsely condemn the final, tiny CG search direction near convergence.
fn pcg_solve<F>(
    apply: F,
    rhs: &Array1<f64>,
    diagonal: &Array1<f64>,
    maxiter: usize,
    miniter: usize,
    residual_tol: f64,
    diagonal_floor: f64,
) -> (Array1<f64>, PcgInfo)
where
    F: Fn(&Array1<f64>) -> Array1<f64>,
{
    let n = rhs.len();
    let mut info = PcgInfo {
        iterations: 0,
        relres: f64::INFINITY,
        stop: PcgStop::Cap,
        negative_curvature: false,
        diagonal_fallback: false,
    };
    let bnorm = rhs.dot(rhs).sqrt();
    if bnorm == 0.0 {
        info.relres = 0.0;
        info.stop = PcgStop::ZeroRhs;
        return (Array1::zeros(n), info);
    }

    let invdiag: Array1<f64> = diagonal.mapv(|d| 1.0 / d.abs().max(diagonal_floor));
    let mut x = Array1::<f64>::zeros(n);
    let mut r = rhs.clone();
    let mut z = &invdiag * &r;
    let mut p = z.clone();
    let mut rz = r.dot(&z);
    let target = residual_tol * bnorm;

    for iteration in 1..=maxiter.max(1) {
        let ap = apply(&p);
        let pap = p.dot(&ap);
        // Scale-relative curvature guard.
        let threshold = 1.0e-14 * (p.dot(&p).sqrt() * ap.dot(&ap).sqrt()).max(1.0e-300);
        if !pap.is_finite() || pap <= threshold {
            info.negative_curvature = true;
            info.stop = PcgStop::NonPositiveCurvature;
            break;
        }
        let alpha = rz / pap;
        x.scaled_add(alpha, &p);
        r.scaled_add(-alpha, &ap);
        let rnorm = r.dot(&r).sqrt();
        info.iterations = iteration;
        if rnorm <= target && (iteration >= miniter || rnorm <= 1.0e-12 * bnorm) {
            info.stop = PcgStop::ForcingTerm;
            break;
        }
        z = &invdiag * &r;
        let rz_new = r.dot(&z);
        if !rz_new.is_finite() || rz.abs() < 1.0e-30 {
            info.stop = PcgStop::Breakdown;
            break;
        }
        let beta = rz_new / rz;
        p = &z + &(beta * &p);
        rz = rz_new;
    }

    if x.dot(&x).sqrt() <= 1.0e-15 * bnorm.max(1.0) {
        x = &invdiag * rhs;
        info.diagonal_fallback = true;
    }
    let resid = rhs - &apply(&x);
    info.relres = resid.dot(&resid).sqrt() / bnorm;
    (x, info)
}

/// The independent STO-3G auxiliary curvature model, Eqs. (S7)–(S9).
///
/// Holds the metric-orthogonalized three-index tensor `B̄ᴾ_pq` in the **current**
/// MO basis, built from the target orbital basis paired with an independent
/// auxiliary (fitting) basis — STO-3G by default. The auxiliary model supplies
/// only curvature; it never contributes to any reported energy, gradient, or
/// convergence decision.
pub struct AuxCurvature {
    /// `B̄ᴾ_ai`, shape `(naux, nvir, nocc)`.
    bvo: Array3<f64>,
    /// `B̄ᴾ_ab`, shape `(naux, nvir, nvir)`.
    bvv: Array3<f64>,
    /// `B̄ᴾ_ij`, shape `(naux, nocc, nocc)`.
    boo: Array3<f64>,
    /// Metric-orthogonalized AO tensor `B̄ᴾ_μν`, kept for MO refreshes.
    b_ao: Array3<f64>,
    nocc: usize,
    nvir: usize,
    /// Fraction of exact exchange in the *curvature model*, `a_x` of Eq. (S8).
    exchange_scale: f64,
    /// Diagonal of the auxiliary response, for preconditioning.
    response_diagonal: Array1<f64>,
    diagonal_floor: f64,
    /// Number of times the auxiliary operator was applied (diagnostics).
    operator_calls: usize,
}

impl AuxCurvature {
    /// Build the auxiliary curvature model for a molecule and target basis.
    ///
    /// `c` are the current MO coefficients, `nocc` the number of doubly occupied
    /// orbitals, and `exchange_scale` the exact-exchange fraction `a_x` used by
    /// the curvature model only.
    pub fn new(
        mol: &Molecule,
        prep: &PreparedBasis,
        c: ArrayView2<f64>,
        nocc: usize,
        exchange_scale: f64,
        cfg: &AuroraConfig,
    ) -> Result<Self, FerricError> {
        let auxbs = ferric_core::basis::bundled(&cfg.auxbasis)?;
        let dfbs = PreparedBasis::new(mol, &auxbs)?;
        let op = Operator::coulomb();

        // (P|Q) metric and its inverse square root, with linear-dependence
        // filtering (the auxiliary basis is minimal, but a near-linear-dependent
        // metric must drop modes rather than amplify noise).
        let v = ferric_integrals::threeindex::coulomb_metric_2c(op, &dfbs)?;
        let v_inv_sqrt = metric_inv_sqrt(&v, cfg.metric_tol)?;

        // (P|μν) three-index integrals, then dress with V^{-1/2} so that
        // (μν|λσ) ≈ Σ_P B̄ᴾ_μν B̄ᴾ_λσ.
        let eri3 = ferric_integrals::threeindex::eri3_tensor(op, prep, &dfbs)?;
        let naux = dfbs.nbasis();
        let nbas = prep.nbasis();
        let flat = eri3
            .into_shape_with_order((naux, nbas * nbas))
            .map_err(|e| FerricError::Lapack(format!("AURORA aux reshape: {e}")))?;
        let dressed = v_inv_sqrt.dot(&flat);
        let b_ao = dressed
            .into_shape_with_order((naux, nbas, nbas))
            .map_err(|e| FerricError::Lapack(format!("AURORA aux reshape back: {e}")))?;

        let nvir = c.ncols() - nocc;
        let mut me = Self {
            bvo: Array3::zeros((naux, nvir, nocc)),
            bvv: Array3::zeros((naux, nvir, nvir)),
            boo: Array3::zeros((naux, nocc, nocc)),
            b_ao,
            nocc,
            nvir,
            exchange_scale,
            response_diagonal: Array1::zeros(nvir * nocc),
            diagonal_floor: cfg.diagonal_floor,
            operator_calls: 0,
        };
        me.refresh_orbitals(c)?;
        Ok(me)
    }

    /// Build the curvature model directly from a supplied MO-basis three-index
    /// tensor, bypassing all integral construction.
    ///
    /// Exposed so the exactness anchors can feed in a tensor whose exact
    /// two-electron integrals are known by construction, and check the response
    /// against an independently written finite difference. `b` is indexed
    /// `[P, p, q]` over the **full** MO range with occupied orbitals first.
    pub fn from_mo_tensor_for_test(
        b: ndarray::ArrayView3<f64>,
        nocc: usize,
        nvir: usize,
        exchange_scale: f64,
        diagonal_floor: f64,
    ) -> Self {
        let naux = b.shape()[0];
        let mut bvo = Array3::zeros((naux, nvir, nocc));
        let mut bvv = Array3::zeros((naux, nvir, nvir));
        let mut boo = Array3::zeros((naux, nocc, nocc));
        for p in 0..naux {
            for a in 0..nvir {
                for i in 0..nocc {
                    bvo[(p, a, i)] = b[(p, nocc + a, i)];
                }
                for c in 0..nvir {
                    bvv[(p, a, c)] = b[(p, nocc + a, nocc + c)];
                }
            }
            for i in 0..nocc {
                for j in 0..nocc {
                    boo[(p, i, j)] = b[(p, i, j)];
                }
            }
        }
        let ax = exchange_scale;
        let mut diag = Array1::<f64>::zeros(nvir * nocc);
        for a in 0..nvir {
            for i in 0..nocc {
                let mut coul = 0.0;
                let mut exch = 0.0;
                for p in 0..naux {
                    let b_ai = bvo[(p, a, i)];
                    coul += b_ai * b_ai;
                    exch += bvv[(p, a, a)] * boo[(p, i, i)];
                }
                diag[a * nocc + i] = (8.0 - 2.0 * ax) * coul - 2.0 * ax * exch;
            }
        }
        Self {
            bvo,
            bvv,
            boo,
            b_ao: Array3::zeros((naux, 0, 0)),
            nocc,
            nvir,
            exchange_scale,
            response_diagonal: diag,
            diagonal_floor,
            operator_calls: 0,
        }
    }

    /// Number of auxiliary functions in the curvature model.
    pub fn naux(&self) -> usize {
        self.b_ao.shape()[0]
    }

    /// Number of auxiliary-operator applications so far (diagnostics only).
    pub fn operator_calls(&self) -> usize {
        self.operator_calls
    }

    /// Re-transform the dressed AO tensor into the current MO basis.
    ///
    /// Called when the orbitals have rotated far enough that the stored MO
    /// tensors no longer describe the current tangent space well.
    pub fn refresh_orbitals(&mut self, c: ArrayView2<f64>) -> Result<(), FerricError> {
        let naux = self.naux();
        let no = self.nocc;
        let nv = self.nvir;
        let co = c.slice(s![.., ..no]);
        let cv = c.slice(s![.., no..]);
        for p in 0..naux {
            let bp = self.b_ao.slice(s![p, .., ..]);
            // Half transforms, then the second index.
            let bp_v = cv.t().dot(&bp); // (nvir, nbas)
            let bp_o = co.t().dot(&bp); // (nocc, nbas)
            self.bvo.slice_mut(s![p, .., ..]).assign(&bp_v.dot(&co));
            self.bvv.slice_mut(s![p, .., ..]).assign(&bp_v.dot(&cv));
            self.boo.slice_mut(s![p, .., ..]).assign(&bp_o.dot(&co));
        }

        // Restricted response diagonal:
        //   (8 - 2 a_x) Σ_P (B̄ᴾ_ai)²  -  2 a_x Σ_P B̄ᴾ_aa B̄ᴾ_ii
        let ax = self.exchange_scale;
        let mut diag = Array1::<f64>::zeros(nv * no);
        for a in 0..nv {
            for i in 0..no {
                let mut coul = 0.0;
                let mut exch = 0.0;
                for p in 0..naux {
                    let b_ai = self.bvo[(p, a, i)];
                    coul += b_ai * b_ai;
                    exch += self.bvv[(p, a, a)] * self.boo[(p, i, i)];
                }
                diag[a * no + i] = (8.0 - 2.0 * ax) * coul - 2.0 * ax * exch;
            }
        }
        self.response_diagonal = diag;
        Ok(())
    }

    /// Apply the auxiliary Coulomb/exchange orbital response, Eqs. (S7)–(S8).
    ///
    /// ```text
    /// [R_aux(X)]_ai = 8 Σ_P B̄ᴾ_ai Σ_bj B̄ᴾ_bj X_bj
    ///                 - 2 a_x Σ_P,bj ( B̄ᴾ_ab B̄ᴾ_ji + B̄ᴾ_aj B̄ᴾ_bi ) X_bj
    /// ```
    ///
    /// The first term is the Coulomb response (factor 8 because both spins move
    /// together in a restricted rotation); the second retains **both** exchange
    /// contractions, which is what makes this the "aux-full" model rather than a
    /// Coulomb-only or diagonal approximation.
    pub fn response(&self, x: ArrayView2<f64>) -> Array2<f64> {
        let naux = self.naux();
        let no = self.nocc;
        let nv = self.nvir;
        let ax = self.exchange_scale;
        let mut y = Array2::<f64>::zeros((nv, no));

        // Coulomb: scalar_P = Σ_bj B̄ᴾ_bj X_bj ; y += 8 Σ_P B̄ᴾ_ai scalar_P
        for p in 0..naux {
            let bp = self.bvo.slice(s![p, .., ..]);
            let scalar: f64 = bp.iter().zip(x.iter()).map(|(b, xv)| b * xv).sum::<f64>();
            if scalar != 0.0 {
                y.scaled_add(8.0 * scalar, &bp);
            }
        }

        if ax != 0.0 {
            // term1_ai = Σ_P Σ_bj B̄ᴾ_ab X_bj B̄ᴾ_ji  = Σ_P (Bvv X Boo)_ai
            //   (B̄ᴾ_ji = B̄ᴾ_ij by symmetry of the dressed tensor)
            // term2_ai = Σ_P Σ_bj B̄ᴾ_aj X_bj B̄ᴾ_bi  = Σ_P (Bvo Xᵀ Bvo)_ai
            let mut acc = Array2::<f64>::zeros((nv, no));
            let xt = x.t().to_owned();
            for p in 0..naux {
                let bvv = self.bvv.slice(s![p, .., ..]);
                let boo = self.boo.slice(s![p, .., ..]);
                let bvo = self.bvo.slice(s![p, .., ..]);
                acc += &bvv.dot(&x).dot(&boo);
                acc += &bvo.dot(&xt).dot(&bvo);
            }
            y.scaled_add(-2.0 * ax, &acc);
        }
        y
    }

    /// Apply the unshifted base curvature operator `ℬₖ`, Eq. (S9):
    ///
    /// ```text
    /// ℬₖ[X] = 2 (F_vv X - X F_oo) + Π* R_aux(Π X)
    /// ```
    ///
    /// in the scaled convention where the gradient is `g̃ = 2 F_vo`. `shift` adds
    /// the adaptive `λₖ Dₖ` regularization of Eq. (2).
    /// Counting wrapper around [`AuxCurvature::base_apply_ref`].
    ///
    /// The formula itself lives in the `&self` variant so that the PCG closure
    /// (which must be a plain `Fn`) and this counting entry point cannot drift
    /// apart. They previously held two copies of the same expression, and a
    /// mutation test caught the duplication: a sign flip applied to one copy
    /// left every test passing because the other copy still served the path
    /// under test.
    pub fn base_apply(
        &mut self,
        x: ArrayView2<f64>,
        f_vv: &Array2<f64>,
        f_oo: &Array2<f64>,
        shift: f64,
    ) -> Array2<f64> {
        self.operator_calls += 1;
        self.base_apply_ref(x, f_vv, f_oo, shift)
    }

    /// The base curvature operator, Eq. (S9). Single source of truth.
    pub fn base_apply_ref(
        &self,
        x: ArrayView2<f64>,
        f_vv: &Array2<f64>,
        f_oo: &Array2<f64>,
        shift: f64,
    ) -> Array2<f64> {
        let mut y = 2.0 * (f_vv.dot(&x) - x.dot(f_oo));
        y += &self.response(x);
        if shift != 0.0 {
            let d = self.diagonal(f_vv, f_oo);
            let no = self.nocc;
            for a in 0..self.nvir {
                for i in 0..no {
                    y[(a, i)] += shift * d[a * no + i] * x[(a, i)];
                }
            }
        }
        y
    }

    /// Diagonal preconditioner: the orbital-energy gap plus the response
    /// diagonal, floored to keep the PCG preconditioner well conditioned.
    pub fn diagonal(&self, f_vv: &Array2<f64>, f_oo: &Array2<f64>) -> Array1<f64> {
        let no = self.nocc;
        let mut d = Array1::<f64>::zeros(self.nvir * no);
        for a in 0..self.nvir {
            for i in 0..no {
                let gap = 2.0 * (f_vv[(a, a)] - f_oo[(i, i)]);
                d[a * no + i] = (gap + self.response_diagonal[a * no + i]).max(self.diagonal_floor);
            }
        }
        d
    }
}

/// Symmetric inverse square root of the auxiliary metric with linear-dependence
/// filtering: modes below `lindep` are dropped rather than amplified.
fn metric_inv_sqrt(v: &Array2<f64>, lindep: f64) -> Result<Array2<f64>, FerricError> {
    let (evals, evecs) = v
        .eigh(UPLO::Upper)
        .map_err(|e| FerricError::Lapack(format!("AURORA aux metric eigh: {e}")))?;
    let n = v.nrows();
    let mut scaled = evecs.clone();
    for k in 0..n {
        if evals[k] < lindep {
            for r in 0..n {
                scaled[(r, k)] = 0.0;
            }
        } else {
            let f = 1.0 / evals[k].sqrt();
            for r in 0..n {
                scaled[(r, k)] *= f;
            }
        }
    }
    Ok(scaled.dot(&evecs.t()))
}

/// Per-SCF persistent AURORA state: the curvature model plus the secant history.
pub struct AuroraState {
    aux: AuxCurvature,
    history: Vec<SecantPair>,
    damped: Vec<DampedPair>,
    cfg: AuroraConfig,
    macro_count: usize,
    force_refresh: bool,
    /// Number of accelerated steps actually taken (diagnostics).
    pub steps_taken: usize,
    /// Number of times the history was cleared on a model-failure test.
    pub history_restarts: usize,
    /// How many stored secant pairs have been Powell-DAMPED (the curvature
    /// condition `sᵀy ≥ 0.2 sᵀBs` failed and `ȳ` was blended), and how many were
    /// accepted undamped. Recorded so the damping branch's reachability is an
    /// observation rather than an assumption: a mutation of the damping formula
    /// that no test notices might mean the branch never executes.
    pub pairs_damped: usize,
    /// Secant pairs accepted without damping.
    pub pairs_undamped: usize,
    /// Total inner PCG iterations (diagnostics).
    pub total_inner_iterations: usize,
    /// Frobenius norm of the most recently APPLIED tangent step, after the
    /// trust-region clip. Observable so the clip can be asserted directly: a
    /// clip that silently did nothing is otherwise invisible from outside.
    pub last_step_norm: f64,
    /// The step awaiting its target-level outcome.
    pending: Option<PendingStep>,
}

impl AuroraState {
    /// Construct the accelerator state at the current orbitals.
    pub fn new(
        mol: &Molecule,
        prep: &PreparedBasis,
        c: ArrayView2<f64>,
        nocc: usize,
        exchange_scale: f64,
        cfg: &AuroraConfig,
    ) -> Result<Self, FerricError> {
        let aux = AuxCurvature::new(mol, prep, c, nocc, exchange_scale, cfg)?;
        Ok(Self {
            aux,
            history: Vec::new(),
            damped: Vec::new(),
            cfg: cfg.clone(),
            macro_count: 0,
            force_refresh: false,
            steps_taken: 0,
            history_restarts: 0,
            pairs_damped: 0,
            pairs_undamped: 0,
            total_inner_iterations: 0,
            last_step_norm: 0.0,
            pending: None,
        })
    }

    /// Number of auxiliary-operator applications (diagnostics only).
    pub fn operator_calls(&self) -> usize {
        self.aux.operator_calls()
    }

    /// Number of secant pairs currently retained.
    pub fn history_len(&self) -> usize {
        self.history.len()
    }

    /// Rebuild the Powell-damped pairs against the *current* base operator.
    ///
    /// Damping is recomputed every macro step because `Aₖ` itself changes: a
    /// pair that satisfied the curvature condition against the previous operator
    /// need not satisfy it against this one. Eqs. (S10)–(S11):
    ///
    /// ```text
    /// bᵢ = Aₖ sᵢ ,  σᵢ = sᵢᵀbᵢ ,  δᵢ = sᵢᵀyᵢ
    /// δᵢ ≥ 0.2 σᵢ  ⇒  ȳᵢ = yᵢ
    /// otherwise      θᵢ = 0.8 σᵢ/(σᵢ - δᵢ) ,  ȳᵢ = θᵢ yᵢ + (1-θᵢ) bᵢ
    /// ```
    ///
    /// which guarantees `sᵢᵀȳᵢ = 0.2 σᵢ > 0`. A pair whose base curvature cannot
    /// be established positively is **discarded**, never forced.
    fn prepare(&mut self, c: ArrayView2<f64>, f_vv: &Array2<f64>, f_oo: &Array2<f64>) {
        self.macro_count += 1;
        let scheduled = self.macro_count > 1
            && self.cfg.refresh_interval > 0
            && (self.macro_count - 1).is_multiple_of(self.cfg.refresh_interval);
        if self.force_refresh || scheduled {
            // A refresh failure is not fatal: the stale MO tensors remain a
            // valid (if less well adapted) curvature model.
            let _ = self.aux.refresh_orbitals(c);
            self.force_refresh = false;
        }

        let tol = self.cfg.secant_tol;
        let mut pairs = Vec::with_capacity(self.history.len());
        // Collect first to avoid holding an immutable borrow of self.history
        // across the mutable `base_apply`.
        let hist: Vec<SecantPair> = self.history.clone();
        for item in &hist {
            let s_mat = to_matrix(&item.s, self.aux.nvir, self.aux.nocc);
            let bs = self.aux.base_apply(s_mat.view(), f_vv, f_oo, 0.0);
            let bs_flat = to_vector(&bs);
            let sy = item.s.dot(&item.y);
            let sbs = item.s.dot(&bs_flat);
            let scale = item.s.dot(&item.s).sqrt() * item.y.dot(&item.y).sqrt();
            if !sbs.is_finite() || sbs <= tol {
                continue;
            }
            let y_use = if sy < 0.2 * sbs {
                let denom = sbs - sy;
                if denom.abs() <= tol * sbs.abs().max(1.0) {
                    continue;
                }
                let theta = 0.8 * sbs / denom;
                self.pairs_damped += 1;
                theta * &item.y + (1.0 - theta) * &bs_flat
            } else {
                self.pairs_undamped += 1;
                item.y.clone()
            };
            let sy_use = item.s.dot(&y_use);
            if sy_use > tol * scale.max(1.0) {
                pairs.push(DampedPair {
                    s: item.s.clone(),
                    y_bar: y_use,
                    rho: 1.0 / sy_use,
                });
            }
        }
        self.damped = pairs;
    }

    /// Apply the auxiliary inverse `Aₖ⁻¹` by PCG, raising the shift `λₖ` until
    /// the base solve behaves like a positive-definite one.
    ///
    /// The ladder of shifts is tried in order; the first that yields a finite,
    /// positive-curvature solution wins. This is the `λₖ Dₖ` of Eq. (2) chosen
    /// adaptively, exactly as Eq. (S10) describes.
    fn apply_inverse_base(
        &mut self,
        rhs: &Array1<f64>,
        f_vv: &Array2<f64>,
        f_oo: &Array2<f64>,
    ) -> (Array1<f64>, PcgInfo, f64) {
        const SHIFTS: [f64; 6] = [0.0, 0.20, 0.40, 0.80, 1.60, 3.20];
        let diagonal = self.aux.diagonal(f_vv, f_oo);
        let nv = self.aux.nvir;
        let no = self.aux.nocc;
        let mut best = (Array1::<f64>::zeros(rhs.len()), None, 3.20f64);
        for &shift in SHIFTS.iter() {
            let precond: Array1<f64> = (1.0 + shift) * &diagonal;
            // The PCG closure must be a plain `Fn`, so it borrows the curvature
            // model immutably and tallies its applications in a Cell.
            let aux = &self.aux;
            let calls = std::cell::Cell::new(0usize);
            let (x, info) = pcg_solve(
                |v| {
                    calls.set(calls.get() + 1);
                    let vm = to_matrix(v, nv, no);
                    let y = aux.base_apply_ref(vm.view(), f_vv, f_oo, shift);
                    to_vector(&y)
                },
                rhs,
                &precond,
                self.cfg.inner_maxiter,
                self.cfg.inner_miniter,
                self.cfg.inner_residual_tol,
                self.cfg.diagonal_floor,
            );
            self.aux.operator_calls += calls.get();
            let ok =
                !info.negative_curvature && x.iter().all(|v| v.is_finite()) && rhs.dot(&x) > 0.0;
            best = (x, Some(info), shift);
            if ok {
                break;
            }
        }
        let (x, info, shift) = best;
        (
            x,
            info.unwrap_or(PcgInfo {
                iterations: 0,
                relres: f64::INFINITY,
                stop: PcgStop::Cap,
                negative_curvature: true,
                diagonal_fallback: false,
            }),
            shift,
        )
    }

    /// The variable-base, Powell-damped L-BFGS direction, Eqs. (S12)–(S16).
    ///
    /// The crucial departure from textbook L-BFGS is the middle step: the
    /// "initial inverse Hessian" `H₀` is **not** a scalar multiple of the
    /// identity but the matrix-free auxiliary solve `r ≈ Aₖ⁻¹ q`. The auxiliary
    /// model therefore supplies full-space curvature from the very first step,
    /// while the target secants repair only the subspace already visited.
    ///
    /// Returns the trust-region-clipped step, the PCG diagnostics and the shift
    /// that was accepted.
    fn lbfgs_direction(
        &mut self,
        gradient: &Array1<f64>,
        f_vv: &Array2<f64>,
        f_oo: &Array2<f64>,
        trust_radius: f64,
    ) -> (Array1<f64>, PcgInfo) {
        let mut q = gradient.clone();
        let mut alphas = Vec::with_capacity(self.damped.len());
        // First loop, newest to oldest.
        for pair in self.damped.iter().rev() {
            let alpha = pair.rho * pair.s.dot(&q);
            q.scaled_add(-alpha, &pair.y_bar);
            alphas.push(alpha);
        }
        // Variable base: r ≈ Aₖ⁻¹ q.
        let (mut r, info, _shift) = self.apply_inverse_base(&q, f_vv, f_oo);
        // Second loop, oldest to newest.
        for (pair, alpha) in self.damped.iter().zip(alphas.iter().rev()) {
            let beta = pair.rho * pair.y_bar.dot(&r);
            r.scaled_add(alpha - beta, &pair.s);
        }
        let mut step = -r;

        // Guaranteed-descent fallback: if the corrected direction is not a
        // descent direction, fall back to the pure auxiliary-Newton direction.
        if !step.iter().all(|v| v.is_finite()) || gradient.dot(&step) >= -1.0e-14 {
            let (base, _i2, _s2) = self.apply_inverse_base(gradient, f_vv, f_oo);
            step = -base;
        }

        let norm = step.dot(&step).sqrt();
        if norm > trust_radius && norm > 0.0 {
            step *= trust_radius / norm;
        }
        (step, info)
    }

    /// Take one AURORA macro step.
    ///
    /// Given the current MOs `c`, the target-level MO-basis Fock matrix `f_mo`,
    /// and the number of occupied orbitals, this returns the rotated MO
    /// coefficients. The caller supplies energy and gradient from the **target**
    /// Hamiltonian only; nothing from the auxiliary model reaches the answer.
    ///
    /// `first` selects the (larger) opening trust radius.
    pub fn step(
        &mut self,
        c: ArrayView2<f64>,
        f_mo: &Array2<f64>,
        nocc: usize,
        first: bool,
    ) -> Result<Array2<f64>, FerricError> {
        let nmo = c.ncols();
        let nvir = nmo - nocc;
        let f_vv = f_mo.slice(s![nocc.., nocc..]).to_owned();
        let f_oo = f_mo.slice(s![..nocc, ..nocc]).to_owned();
        // Scaled gradient g̃ = 2 F_vo (Eq. S1 note).
        let g_mat = 2.0 * f_mo.slice(s![nocc.., ..nocc]).to_owned();
        let gradient = to_vector(&g_mat);

        self.prepare(c, &f_vv, &f_oo);
        let radius = if first {
            self.cfg.first_trust_radius
        } else {
            self.cfg.trust_radius
        };
        let (step, info) = self.lbfgs_direction(&gradient, &f_vv, &f_oo, radius);
        self.total_inner_iterations += info.iterations;

        let step_mat = to_matrix(&step, nvir, nocc);
        self.last_step_norm = step.dot(&step).sqrt();
        let u_mo = geodesic_rotation(step_mat.view())?;
        let c_new = c.dot(&u_mo);

        // Transport every stored pair into the new tangent frame, then record
        // the new secant (Eq. 3 / S19). `y` needs the *next* gradient, so the
        // pending step and transported gradient are stashed until `observe`.
        for pair in self.history.iter_mut() {
            let s_m = to_matrix(&pair.s, nvir, nocc);
            let y_m = to_matrix(&pair.y, nvir, nocc);
            pair.s = to_vector(&transport_tangent(s_m.view(), u_mo.view(), nocc));
            pair.y = to_vector(&transport_tangent(y_m.view(), u_mo.view(), nocc));
        }
        self.pending = Some(PendingStep {
            s: to_vector(&transport_tangent(step_mat.view(), u_mo.view(), nocc)),
            g_transported: to_vector(&transport_tangent(g_mat.view(), u_mo.view(), nocc)),
            grad_norm: gradient.dot(&gradient).sqrt(),
        });
        self.steps_taken += 1;
        AURORA_STEPS_THIS_THREAD.with(|c| c.set(c.get() + 1));
        Ok(c_new)
    }

    /// Record the target-level outcome of the step just taken.
    ///
    /// `f_mo_new` is the MO-basis Fock matrix at the rotated orbitals (in the
    /// *rotated* MO frame) and `delta_energy` the target energy change. This
    /// closes the secant pair `yₖ = g_{k+1} - 𝒯ₖ(gₖ)` and applies the
    /// model-failure test that clears the history when the auxiliary curvature
    /// has clearly mispredicted.
    pub fn observe(&mut self, f_mo_new: &Array2<f64>, nocc: usize, delta_energy: f64) {
        let Some(pending) = self.pending.take() else {
            return;
        };
        let nvir = f_mo_new.ncols() - nocc;
        let g_new_mat = 2.0 * f_mo_new.slice(s![nocc.., ..nocc]).to_owned();
        let g_new = to_vector(&g_new_mat);
        let new_norm = g_new.dot(&g_new).sqrt();
        let ratio = new_norm / pending.grad_norm.max(1.0e-16);

        let model_failed = ratio > 1.8 || (delta_energy > 1.0e-8 && ratio > 1.25);
        if model_failed {
            self.history.clear();
            self.damped.clear();
            self.force_refresh = true;
            self.history_restarts += 1;
            return;
        }
        let y = &g_new - &pending.g_transported;
        if pending.s.dot(&pending.s).sqrt() > 1.0e-12 && y.dot(&y).sqrt() > 1.0e-12 {
            self.history.push(SecantPair { s: pending.s, y });
            let keep = self.cfg.history_size.max(1);
            if self.history.len() > keep {
                let drop = self.history.len() - keep;
                self.history.drain(..drop);
            }
        }
        let _ = nvir;
    }
}

/// A step that has been applied but whose secant pair is not yet closed.
struct PendingStep {
    s: Array1<f64>,
    g_transported: Array1<f64>,
    grad_norm: f64,
}

/// Test-only re-export of the guarded PCG solver.
///
/// Exposed so the exactness anchors can verify that the solver really solves an
/// SPD system when given enough iterations — the production call site uses a
/// deliberately loose forcing term that would mask a recursion bug.
pub fn pcg_solve_for_test<F>(
    apply: F,
    rhs: &Array1<f64>,
    diagonal: &Array1<f64>,
    maxiter: usize,
    miniter: usize,
    residual_tol: f64,
    diagonal_floor: f64,
) -> (Array1<f64>, PcgInfo)
where
    F: Fn(&Array1<f64>) -> Array1<f64>,
{
    pcg_solve(
        apply,
        rhs,
        diagonal,
        maxiter,
        miniter,
        residual_tol,
        diagonal_floor,
    )
}

/// Row-major flatten of an `(nvir, nocc)` tangent block.
fn to_vector(m: &Array2<f64>) -> Array1<f64> {
    Array1::from_iter(m.iter().copied())
}

/// Inverse of [`to_vector`].
fn to_matrix(v: &Array1<f64>, nvir: usize, nocc: usize) -> Array2<f64> {
    Array2::from_shape_vec((nvir, nocc), v.to_vec())
        .expect("AURORA tangent reshape: length matches nvir*nocc by construction")
}

/// Unused placeholder to keep the parallel context in the signature space for a
/// future MPI-aware auxiliary build.
#[allow(dead_code)]
fn _ctx_unused(_ctx: &ParallelContext) {}
