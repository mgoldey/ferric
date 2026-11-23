//! Linear-response TDDFT for closed-shell references (the restricted driver;
//! UHF/UKS references are dispatched to `crate::unrestricted`).
//!
//! Implements the Tamm–Dancoff approximation (TDA, equivalent to CIS for HF)
//! and the full Casida equations for singlet or (`M_S = 0`) triplet
//! excitations ([`TddftConfig::spin`]); the singlet matrices are:
//!
//! TDA:    A X = ω X
//! Casida: (A−B)^{1/2} (A+B) (A−B)^{1/2} Z = Ω² Z
//!
//! where A_{ia,jb} = δ_{ij}δ_{ab}(ε_a − ε_i) + 2(ia|jb) − c_HF(ij|ab) + K_{ia,jb}
//!       B_{ia,jb} = 2(ia|bj) − c_HF(ib|aj) + K_{ia,jb}
//!
//! and `K_{ia,jb} = (ia| f_αα + f_αβ |jb)` is the singlet XC-kernel block
//! (PySCF `get_ab`'s `iajb`). With real orbitals `(ia|f_xc|jb) = (ia|f_xc|bj)`,
//! so the same block enters A and B. It is built by
//! [`ferric_dft::lr_kernel::fxc_ov_block`] — the SAME code the
//! PySCF-validated `ferric_gw::tddft::run_tda_dft` uses — and the functional
//! gate [`ferric_dft::lr_kernel::resolve_response_xc`] hard-refuses every
//! functional without a complete kernel (VV10); τ-only meta-GGAs (SCAN /
//! r2SCAN / TPSS) take the τ kernel. There is no kernel-less DFT path: a KS
//! reference either gets its f_xc term or an error.
//!
//! Range-separated hybrids replace `c_HF (pq|rs)` by
//! `c_sr (pq|rs) + (c_lr − c_sr) (pq|rs)_ω`, the second term over a second set
//! of RI tensors of the erf(ωr)/r operator in its own metric. Triplets drop
//! the Coulomb term and take the `f_αα − f_αβ` kernel. Derivations:
//! `book/src/19-tddft/04-rsh.md#rsh-split` (range separation) and §9 (triplets).
//!
//! Scope: this module assembles the closed-shell (restricted) problem.
//! [`run_tddft`] sends an unrestricted (UHF / UKS) reference to
//! `crate::unrestricted`, which builds the two-spin A/B space; only ROHF /
//! ROKS references are refused (`book/src/19-tddft/06-unrestricted.md#unrestricted-spin-problem`).
//!
//! # Solvers
//!
//! [`TddftSolver::Dense`] assembles A (and B) as `dim × dim` matrices
//! (`dim = nocc·nvir`) and diagonalizes them fully; the f_xc block costs
//! `dim` AO kernel applications. [`TddftSolver::Davidson`] never forms them:
//! [`crate::davidson`] iterates on products with a block of vectors
//! (the private `matvec` module: RI Coulomb, `c_HF` RI exchange and one batched f_xc
//! application per block, [`ferric_dft::lr_kernel::SingletOvFxc`]). Both
//! report each root's residual. [`TddftSolver::Auto`] takes the dense path
//! up to [`AUTO_DENSE_MAX_DIM`].
//!
//! Under `FERRIC_TIMING` a phase table is printed to stderr (RI tensors,
//! kernel setup, assembly / products, eigensolve, oscillator strengths).

use crate::davidson::{davidson_casida, davidson_tda, DavidsonOutcome, DavidsonSettings};
use crate::matvec::ExchangeTensors;
use crate::matvec::ResponseOperator;
use ferric_core::memory::plan::{Lifetime, MemoryPlan};
use ferric_core::mol::Molecule;
use ferric_core::timing::PhaseAccumulator;
use ferric_core::FerricError;
use ferric_dft::grid::AtomicGridConfig;
pub use ferric_dft::lr_kernel::ResponseSpin;
use ferric_dft::lr_kernel::{
    fxc_ov_block, resolve_response_xc as resolve_kernel_xc, SingletKernel, SingletOvFxc,
    FXC_BLOCK_ASYMMETRY_TOL,
};
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::oneelectron;
use ferric_integrals::operator::Operator;
use ferric_integrals::three_index_source::ThreeIndexSource;
use ferric_integrals::threeindex;
use ferric_mp2::rimp2::{eri3_budget_bytes, metric_inverse_sqrt, stream_dressed_mo_band};
use ferric_scf::{PreparedAux, ScfResult};
use ndarray::{Array1, Array2};
use ndarray_linalg::{Eigh, UPLO};

/// TDDFT solution method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TddftMethod {
    /// Tamm–Dancoff approximation: diagonalize `A` alone (`B = 0`, `Y = 0`).
    Tda,
    /// Full linear response: solve the `(A, B)` problem through
    /// `(A−B)^½ (A+B) (A−B)^½ Z = Ω² Z` (dense, `dim = nocc·nvir`).
    Casida,
}

/// Largest `dim = nocc·nvir` that [`TddftSolver::Auto`] solves densely.
pub const AUTO_DENSE_MAX_DIM: usize = 200;

/// Eigensolver for the response problem.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TddftSolver {
    /// Dense A (and B) and a full `eigh`; `dim` f_xc kernel applications.
    Dense,
    /// Matrix-free block Davidson ([`crate::davidson`]).
    Davidson,
    /// [`Self::Dense`] for `dim <= AUTO_DENSE_MAX_DIM` or when the Davidson
    /// guess space would cover half of `dim`, else [`Self::Davidson`].
    #[default]
    Auto,
}

impl TddftSolver {
    /// `"dense"`, `"davidson"` or `"auto"` (case-insensitive); anything else
    /// is an error.
    pub fn parse_config_str(s: &str) -> Result<Self, FerricError> {
        match s.to_ascii_lowercase().as_str() {
            "dense" => Ok(Self::Dense),
            "davidson" => Ok(Self::Davidson),
            "auto" => Ok(Self::Auto),
            other => Err(FerricError::General(format!(
                "TDDFT solver '{other}' is not one of \"dense\", \"davidson\", \"auto\""
            ))),
        }
    }

    /// The solver `Auto` resolves to for `dim` and `n_roots` (the others
    /// resolve to themselves).
    pub fn resolve(self, dim: usize, n_roots: usize) -> Self {
        match self {
            Self::Auto if dim <= AUTO_DENSE_MAX_DIM || 4 * n_roots.max(1) >= dim => Self::Dense,
            Self::Auto => Self::Davidson,
            other => other,
        }
    }

    /// Lower-case name, as the config key spells it.
    pub fn name(self) -> &'static str {
        match self {
            Self::Dense => "dense",
            Self::Davidson => "davidson",
            Self::Auto => "auto",
        }
    }
}

/// Configuration for a TDDFT calculation.
#[derive(Debug, Clone)]
pub struct TddftConfig {
    /// Number of lowest positive singlet roots returned (capped at
    /// `nocc·nvir`).
    pub n_roots: usize,
    /// TDA or full Casida response.
    pub method: TddftMethod,
    /// Optional resident-bytes ceiling for the dense `(ia, jb)` matrices and
    /// the RI 3-index build. `None` → resolved via
    /// [`ferric_core::memory::resolve_budget_bytes`] (env / cgroup / RAM
    /// auto-detect).
    ///
    /// This field did not exist before: the crate had NO budget machinery at
    /// all, so `[memory] budget_gb` and a caller-supplied ceiling were both
    /// unreachable from here, and `build_b_tensors` hardcoded
    /// `eri3_budget_bytes(None)` — the same "silently discarded budget" defect
    /// ferric-rpa fixed three times (see `ferric_rpa::properties`). Threading
    /// it makes a pinned budget actually reach the allocations; see
    /// `caller_budget_is_honoured_not_discarded` in the tests below.
    pub memory_budget_bytes: Option<usize>,
    /// The functional the reference was converged with (e.g. `"PBE"`,
    /// `"B3LYP"`), or `None` for a Hartree–Fock reference (CIS / TDHF).
    ///
    /// It selects BOTH the exact-exchange fraction and the f_xc kernel, so it
    /// must name the reference's functional: `run_tddft` cannot check that
    /// (an `ScfResult` does not record its functional). A range-separated
    /// hybrid takes its c_sr, c_lr and ω from libxc; a functional without a
    /// complete kernel (VV10) is refused.
    pub xc: Option<String>,
    /// Integration grid for the f_xc kernel. Must match the grid the reference
    /// SCF used (`RhfConfig::dft_grid`; `AtomicGridConfig::production()` when that
    /// is `None`), or the kernel is evaluated on a different quadrature than
    /// the density it linearizes. Ignored for an HF reference.
    pub grid: AtomicGridConfig,
    /// Override the exact-exchange fraction on the `−c_HF` terms. `None`
    /// (default) takes it from `xc` (1 for an HF reference). An override is a
    /// deliberate model change and is only accepted WITH a functional: an HF
    /// reference with `c_HF != 1` is neither CIS/TDHF nor any TDDFT, and is
    /// refused.
    pub c_hf_override: Option<f64>,
    /// Eigensolver; [`TddftSolver::Auto`] by default.
    pub solver: TddftSolver,
    /// Davidson iteration controls (ignored by the dense solver).
    pub davidson: DavidsonSettings,
    /// Singlet (default) or `M_S = 0` triplet excitations.
    pub spin: ResponseSpin,
}

impl TddftConfig {
    /// `n_roots` singlets by `method` on an HF reference, default grid, no
    /// budget pin and no `c_HF` override. Set `xc` for a KS reference.
    pub fn new(n_roots: usize, method: TddftMethod) -> Self {
        Self {
            n_roots,
            method,
            ..Self::default()
        }
    }

    /// Three TDA singlets on an HF reference: production grid, automatic
    /// solver, no budget pin and no `c_HF` override.
    fn hf_singlets() -> Self {
        Self {
            n_roots: 3,
            method: TddftMethod::Tda,
            memory_budget_bytes: None,
            xc: None,
            grid: AtomicGridConfig::production(),
            c_hf_override: None,
            solver: TddftSolver::Auto,
            davidson: DavidsonSettings::default(),
            spin: ResponseSpin::Singlet,
        }
    }
}

impl Default for TddftConfig {
    fn default() -> Self {
        Self::hf_singlets()
    }
}

/// Result of a TDDFT calculation.
#[derive(Debug, Clone)]
#[must_use]
pub struct TddftResult {
    /// Lowest `n_roots` positive excitation energies (Hartree) of the run's
    /// spin, ascending.
    pub excitation_energies: Vec<f64>,
    /// Length-gauge oscillator strengths, PySCF convention
    /// (`f = (2/3) Ω |<0|r|n>|²`, `<0|r|n> = √2 Σ_ia (X+Y)_ia <i|r|a>` with
    /// `|X|² − |Y|² = 1`; `Y = 0` for TDA). Exactly zero for triplets
    /// (spin-forbidden).
    pub oscillator_strengths: Vec<f64>,
    /// `<0|r|n>` (a.u., origin at 0), same ordering as the energies.
    pub transition_dipoles: Vec<[f64; 3]>,
    /// Solution method the roots were obtained with (copied from the config).
    pub method: TddftMethod,
    /// Exact-exchange fraction actually used on the full-Coulomb `−c_HF`
    /// terms (`c_sr` of a range-separated hybrid).
    pub c_hf: f64,
    /// Long-range (erf) exact-exchange fraction; equals `c_hf` unless
    /// `omega != 0`.
    pub c_lr: f64,
    /// Range-separation parameter ω (Bohr⁻¹) of the exchange; 0 for none.
    pub omega: f64,
    /// The spin of the roots of a restricted reference (copied from the
    /// config); `None` on an unrestricted reference, whose spin-conserving
    /// roots carry no singlet / triplet label (read [`Self::s2`]).
    pub spin: Option<ResponseSpin>,
    /// The reference the roots were computed on: restricted or unrestricted.
    pub reference: ferric_scf::Spin,
    /// `<S²>` of each root, same ordering as the energies
    /// (`book/src/19-tddft/06-unrestricted.md#unrestricted-each-root`):
    /// exactly 0 (singlet) or 2 (triplet) on a restricted reference; on an
    /// unrestricted one, `<S²>` of the excited determinant expansion with
    /// coefficients `X` (TDA) or `X = ((X+Y) + (X−Y))/2` renormalized to
    /// unit length (Casida).
    pub s2: Vec<f64>,
    /// `<S²>` of the reference determinant (0 for a restricted reference).
    pub s2_reference: f64,
    /// Whether the `(ia|f_xc|jb)` kernel block was added to A (and B). True for
    /// every KS reference — there is no kernel-less DFT path — and false only
    /// for an HF reference, where the kernel is identically zero.
    pub fxc_included: bool,
    /// The solver that ran ([`TddftSolver::Dense`] or
    /// [`TddftSolver::Davidson`], never `Auto`).
    pub solver: TddftSolver,
    /// Residual norm of each returned root, same ordering as the energies:
    /// `‖A X − ω X‖` (TDA, unit X) or `√((‖r+‖² + ‖r−‖²)/2)` (Casida, see
    /// [`crate::davidson`]). For the dense solver it is evaluated with the
    /// dense matrices.
    pub residuals: Vec<f64>,
    /// Davidson iterations (0 for the dense solver).
    pub iterations: usize,
}

impl std::fmt::Display for TddftResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let ha_to_ev = 27.211_386_245_988;
        writeln!(
            f,
            "TDDFT {:?} — {} roots:",
            self.method,
            self.excitation_energies.len()
        )?;
        for (i, (&e, &osc)) in self
            .excitation_energies
            .iter()
            .zip(&self.oscillator_strengths)
            .enumerate()
        {
            writeln!(
                f,
                "  {}: {:.6} Ha  ({:.4} eV)  f = {:.6}",
                i + 1,
                e,
                e * ha_to_ev,
                osc
            )?;
        }
        Ok(())
    }
}

/// Build the RI-dressed 3-index tensors needed for TDDFT.
///
/// Returns `(b_ov, b_oo, b_vv)`:
///   - `b_ov`: (naux, nocc*nvir)  — `(ia|jb) = Σ_P b_ov[P,ia] * b_ov[P,jb]`
///   - `b_oo`: (naux, nocc*nocc)  — for exchange `(ij|ab)`
///   - `b_vv`: (naux, nvir*nvir)  — for exchange `(ij|ab)`
///
/// All three are returned to the caller and stay live for the whole of
/// [`run_tddft`], so all three are `Resident` in `plan`. `b_vv` is the largest
/// (nvir ≫ nocc at any production basis), and it is the term that shows up
/// first in the breakdown when this gate fires.
///
/// The `naux×nao²` AO source is streamed aux-blocked under the SAME budget
/// (`ThreeIndexSource::build` spills to disk rather than allocating past it),
/// so it is not charged here — but its ceiling now comes from
/// `eri3_budget_bytes(memory_budget_bytes)` rather than the hardcoded
/// `eri3_budget_bytes(None)` that used to discard a caller's budget outright.
pub(crate) fn build_b_tensors(
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    c_occ: &Array2<f64>,
    c_vir: &Array2<f64>,
    plan: &mut MemoryPlan,
    memory_budget_bytes: Option<usize>,
) -> Result<(Array2<f64>, Array2<f64>, Array2<f64>), FerricError> {
    let op = Operator::coulomb();
    let v2c = threeindex::coulomb_metric_2c(op, dfbs)?;
    let v2c_inv_sqrt = metric_inverse_sqrt(&v2c, op)?;
    let budget = eri3_budget_bytes(memory_budget_bytes);
    let mut src = ThreeIndexSource::build(op, obs, dfbs, budget)?;

    // naux is only known once the metric is built, so the RI reservations are
    // declared here rather than at the entry point — still BEFORE the first
    // large allocation (`stream_dressed_mo_band` below), which is the property
    // that matters.
    let naux = v2c_inv_sqrt.nrows();
    let (nocc, nvir) = (c_occ.ncols(), c_vir.ncols());
    plan.reserve(
        "B(P|ia) [b_ov]",
        naux.saturating_mul(nocc * nvir),
        Lifetime::Resident,
    );
    plan.reserve(
        "B(P|ij) [b_oo]",
        naux.saturating_mul(nocc * nocc),
        Lifetime::Resident,
    );
    plan.reserve(
        "B(P|ab) [b_vv]",
        naux.saturating_mul(nvir * nvir),
        Lifetime::Resident,
    );
    plan.check()?;
    let b_ov = stream_dressed_mo_band(&mut src, &v2c_inv_sqrt, c_occ, c_vir, None)?;
    let b_oo = stream_dressed_mo_band(&mut src, &v2c_inv_sqrt, c_occ, c_occ, None)?;
    let b_vv = stream_dressed_mo_band(&mut src, &v2c_inv_sqrt, c_vir, c_vir, None)?;
    Ok((b_ov, b_oo, b_vv))
}

/// The erf(ωr)/r tensors of a range-separated hybrid (`book/src/19-tddft/04-rsh.md#rsh-split`):
/// `B^ω = M_ω (Q|erf|pq)` with `M_ωᵀ M_ω` the pseudo-inverse of the erf
/// metric `(P|erf|Q)` (eigendecomposition, near-null modes dropped:
/// [`metric_inverse_sqrt`] for the erf operator), over the same auxiliary
/// basis and in the same `ov`, `oo`, `vv` layout as [`build_b_tensors`].
pub(crate) fn build_lr_tensors(
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    c_occ: &Array2<f64>,
    c_vir: &Array2<f64>,
    omega: f64,
    plan: &mut MemoryPlan,
    memory_budget_bytes: Option<usize>,
) -> Result<RiTensors, FerricError> {
    let op = Operator::erf(omega);
    let naux = dfbs.nbasis();
    let (nocc, nvir) = (c_occ.ncols(), c_vir.ncols());
    for (label, width) in [
        ("B_erf(P|ia)", nocc * nvir),
        ("B_erf(P|ij)", nocc * nocc),
        ("B_erf(P|ab)", nvir * nvir),
    ] {
        plan.reserve(label, naux.saturating_mul(width), Lifetime::Resident);
    }
    plan.check()?;
    let v2c = threeindex::coulomb_metric_2c(op, dfbs)?;
    let m = metric_inverse_sqrt(&v2c, op)?;
    let mut src = ThreeIndexSource::build(op, obs, dfbs, eri3_budget_bytes(memory_budget_bytes))?;
    let std = |b: Array2<f64>| b.as_standard_layout().into_owned();
    Ok(RiTensors {
        b_ov: std(stream_dressed_mo_band(&mut src, &m, c_occ, c_vir, None)?),
        b_oo: std(stream_dressed_mo_band(&mut src, &m, c_occ, c_occ, None)?),
        b_vv: std(stream_dressed_mo_band(&mut src, &m, c_vir, c_vir, None)?),
        lr: None,
    })
}

/// Declare every dense `(dim, dim)` matrix the requested method holds, with the
/// lifetimes read off the source rather than guessed.
///
/// This is the single place the peak is written down, and it is deliberately
/// verbose about WHICH matrices coexist, because co-residency is the part that
/// goes stale silently. Getting it wrong in EITHER direction is a defect:
/// under-counting OOMs, over-counting refuses jobs that would have fit.
///
/// # The scratch inside `build_a_matrix` / `build_b_matrix`
///
/// At most two `dim²` buffers are live at once, for every spin and exchange
/// mix:
///
/// * A: `coulomb = b_ovᵀ·b_ov` plus its `c_J * &coulomb` temporary (live for
///   one statement, and both dropped at the end of the Coulomb block); then
///   one `k_ij_ab = b_ooᵀ·b_vv` (`(nocc², nvir²)`, i.e. `dim²` elements) at a
///   time, for the Coulomb-operator exchange and then the erf one.
/// * B: `coulomb` plus its scaled temporary, the same product reused for the
///   exchange; then, for a range-separated hybrid, the erf `k_ov` alone after
///   `coulomb` is dropped.
///
/// So `2·dim²` of scratch, charged as ONE `Transient`. It is not charged
/// per-worker: neither builder uses rayon.
///
/// # TDA
///
/// Stage 1 (`build_a_matrix`): `a` + `2·dim²` scratch = `3·dim²`.
/// Stage 2 (`a.eigh`): `a` (still live — `eigh` takes `&self`) + the
/// eigenvector output = `2·dim²`.
///
/// (A KS reference adds a third alternative, see [`reserve_fxc_block`].)
///
/// Peak is stage 1. Declared as `a` resident plus the *alternatives* as
/// transients, so the plan takes `a` + max(scratch, eigenvectors) = `3·dim²`
/// and NOT `4·dim²` — charging the eigenvectors as resident on top of the
/// scratch would over-estimate by a whole matrix.
///
/// # Casida
///
/// Nine `dim²` matrices are simultaneously live at the widest point (just after
/// `m.eigh`), because nothing in the arm is dropped early: `a`, `b_mat`,
/// `a_plus_b`, `a_minus_b`, `amb_vecs`, `amb_sqrt`, `m`, `z` (the eigenvector
/// output) and `x = amb_sqrt·z`.
///
/// That is stage 3, and it dominates: the `build_a`/`build_b` stage peaks at
/// `a` + `b_mat` + `2·dim²` scratch = `4·dim²`, well under nine. So the nine
/// are declared resident and the earlier scratch adds NOTHING on top — it is
/// long dropped by the time the peak is reached, and adding it would
/// over-estimate by two matrices.
pub(crate) fn reserve_dense_response(plan: &mut MemoryPlan, method: TddftMethod, dim: usize) {
    let d2 = dim.saturating_mul(dim);
    let build_scratch = 2usize.saturating_mul(d2);

    match method {
        TddftMethod::Tda => {
            plan.reserve("A (ia,jb)", d2, Lifetime::Resident);
            // These two are the alternating occupants of the second slot, one
            // per stage — never simultaneous, so the plan takes the larger.
            plan.reserve("build_a_matrix scratch", build_scratch, Lifetime::Transient);
            plan.reserve("eigh eigenvectors", d2, Lifetime::Transient);
        }
        TddftMethod::Casida => {
            for label in [
                "A (ia,jb)",
                "B (ia,jb)",
                "A+B",
                "A-B",
                "(A-B) eigenvectors",
                "(A-B)^1/2",
                "M = (A-B)^1/2 (A+B) (A-B)^1/2",
                "M eigenvectors Z",
                "X = (A-B)^1/2 Z",
            ] {
                plan.reserve(label, d2, Lifetime::Resident);
            }
            // Deliberately NOT declared: the build-stage scratch is dropped
            // long before the nine-matrix peak above, so charging it would
            // over-estimate. See this function's docs.
        }
    }
}

/// Declare the f_xc block's scratch (KS reference only).
///
/// [`add_fxc_block`] holds the raw and the symmetrized `K` (`2·dim²`) while
/// the matrices it adds into are live.
///
/// * TDA: `a` + K + K_sym = `3·dim²` — a third alternating occupant of the
///   second slot next to [`reserve_dense_response`]'s build scratch and
///   eigenvectors, dropped before `eigh`, so it is `Transient`.
/// * Casida: `a` + `b_mat` + K + K_sym = `4·dim²`, well under the nine-matrix
///   peak, so nothing is added.
///
/// Not called for an HF reference: over-charging would refuse jobs that fit.
fn reserve_fxc_block(plan: &mut MemoryPlan, method: TddftMethod, dim: usize) {
    if method == TddftMethod::Tda {
        let k_pair = 2usize.saturating_mul(dim.saturating_mul(dim));
        plan.reserve("f_xc block (K + K_sym)", k_pair, Lifetime::Transient);
    }
}

/// Subtract `coef · (ij|ab)`, over the RI tensors `t` of one operator, from
/// the `(ia, jb)` element of `a`.
fn subtract_exchange_a(
    a: &mut Array2<f64>,
    t: ExchangeTensors<'_>,
    nocc: usize,
    nvir: usize,
    coef: f64,
) {
    // (ij|ab) = Σ_P B_oo[P, i*nocc+j] * B_vv[P, a*nvir+b]
    let k_ij_ab = t.b_oo.t().dot(t.b_vv); // (nocc*nocc, nvir*nvir)
    for i in 0..nocc {
        for a_idx in 0..nvir {
            let ia = i * nvir + a_idx;
            for j in 0..nocc {
                for b_idx in 0..nvir {
                    let jb = j * nvir + b_idx;
                    let ij = i * nocc + j;
                    let ab = a_idx * nvir + b_idx;
                    a[(ia, jb)] -= coef * k_ij_ab[(ij, ab)];
                }
            }
        }
    }
}

/// Subtract `coef · (ib|aj)` from the `(ia, jb)` element of `b`, given
/// `k_ov = B_ovᵀ B_ov` of one operator: `(ib|aj) = (ib|ja)` is its `(ib, ja)`
/// element.
fn subtract_exchange_b(
    b: &mut Array2<f64>,
    k_ov: &Array2<f64>,
    nocc: usize,
    nvir: usize,
    coef: f64,
) {
    for i in 0..nocc {
        for a_idx in 0..nvir {
            let ia = i * nvir + a_idx;
            for j in 0..nocc {
                for b_idx in 0..nvir {
                    let jb = j * nvir + b_idx;
                    let ib = i * nvir + b_idx;
                    let ja = j * nvir + a_idx;
                    b[(ia, jb)] -= coef * k_ov[(ib, ja)];
                }
            }
        }
    }
}

/// Build the TDA A matrix in the (ia, jb) compound-index space.
///
/// A_{ia,jb} = δ_{ij}δ_{ab}(ε_a − ε_i) + c_J (ia|jb) − c_sr (ij|ab) − Δc (ij|ab)_ω
///
/// with `c_J` = 2 (singlet) or 0 (triplet) and the long-range term only for a
/// range-separated hybrid (`book/src/19-tddft/04-rsh.md#rsh-split`,
/// `book/src/19-tddft/05-triplets.md#triplets-spin-blocks`). The `+ K_{ia,jb}` f_xc block is
/// added by the caller.
pub(crate) fn build_a_matrix(
    eps: &[f64],
    nocc: usize,
    nvir: usize,
    ri: &RiTensors,
    mix: ExchangeMix,
    c_j: f64,
) -> Array2<f64> {
    let dim = nocc * nvir;
    let mut a = Array2::<f64>::zeros((dim, dim));

    // Diagonal: orbital energy differences
    for i in 0..nocc {
        for a_idx in 0..nvir {
            let ia = i * nvir + a_idx;
            a[(ia, ia)] = eps[nocc + a_idx] - eps[i];
        }
    }

    // Coulomb: c_J (ia|jb) = c_J B_ovᵀ B_ov.
    if c_j != 0.0 {
        let coulomb = ri.b_ov.t().dot(&ri.b_ov);
        a += &(c_j * &coulomb);
    }
    if mix.c_sr != 0.0 {
        subtract_exchange_a(&mut a, ri.tensors(), nocc, nvir, mix.c_sr);
    }
    if let (Some(lr), Some(dc)) = (ri.lr.as_deref(), mix.long_range()) {
        subtract_exchange_a(&mut a, lr.tensors(), nocc, nvir, dc);
    }

    // The (ia|f_xc|jb) block is added by `run_tddft` after this returns (it
    // needs the kernel and the reference density, and the SAME block also
    // goes into B). For an HF reference it is identically zero: CIS.
    a
}

/// Build the TDDFT B matrix in the (ia, jb) compound-index space.
///
/// B_{ia,jb} = c_J (ia|bj) − c_sr (ib|aj) − Δc (ib|aj)_ω
///
/// (the `+ K_{ia,jb}` f_xc block is added by the caller). `(ia|bj) = (ia|jb)`
/// for real orbitals, so the Coulomb part is A's; the exchange `(ib|aj)` is
/// the `(ib, ja)` element of the same `B_ovᵀ B_ov` product.
pub(crate) fn build_b_matrix(
    nocc: usize,
    nvir: usize,
    ri: &RiTensors,
    mix: ExchangeMix,
    c_j: f64,
) -> Array2<f64> {
    let dim = nocc * nvir;
    let mut b = Array2::<f64>::zeros((dim, dim));
    if c_j != 0.0 || mix.c_sr != 0.0 {
        let coulomb = ri.b_ov.t().dot(&ri.b_ov);
        if c_j != 0.0 {
            b += &(c_j * &coulomb);
        }
        if mix.c_sr != 0.0 {
            subtract_exchange_b(&mut b, &coulomb, nocc, nvir, mix.c_sr);
        }
    }
    if let (Some(lr), Some(dc)) = (ri.lr.as_deref(), mix.long_range()) {
        let k_ov = lr.b_ov.t().dot(&lr.b_ov);
        subtract_exchange_b(&mut b, &k_ov, nocc, nvir, dc);
    }
    b
}

/// Transition dipoles and length-gauge oscillator strengths for the states in
/// `cols` (columns of `amplitudes`), in PySCF's convention:
///
/// ```text
///   <0|r|n> = √2 Σ_ia (X+Y)_{ia,n} <i|r|a>,   f_n = (2/3) Ω_n |<0|r|n>|²
/// ```
///
/// with `|X|² − |Y|² = 1` (TDA: `Y = 0`, `|X| = 1`). The `√2` is the singlet
/// spin-adaptation factor — the same one `ferric_gw::tddft` and `bse.rs` use,
/// both cross-checked against PySCF's `oscillator_strength`. This helper
/// previously omitted it (so TDA `f` was half PySCF's) and fed the Casida
/// branch an un-normalized `(A−B)^{1/2} Z`; both are fixed here.
fn oscillator_strengths(
    obs: &PreparedBasis,
    c_occ: &Array2<f64>,
    c_vir: &Array2<f64>,
    nocc: usize,
    nvir: usize,
    omegas: &Array1<f64>,
    amplitudes: &Array2<f64>,
    cols: &[usize],
) -> Result<(Vec<f64>, Vec<[f64; 3]>), FerricError> {
    let origin = [0.0, 0.0, 0.0];
    let mu_ao = oneelectron::dipole(obs, origin)?;

    // Transform dipole integrals to MO basis: μ^(MO)_{ia} = C_occ^T μ^(AO) C_vir
    let mu_ov: Vec<Array2<f64>> = (0..3)
        .map(|xyz| c_occ.t().dot(&mu_ao[xyz]).dot(c_vir))
        .collect();

    let mut osc = Vec::with_capacity(cols.len());
    let mut tdips = Vec::with_capacity(cols.len());
    for &n in cols {
        let omega_n = omegas[n];
        let mut tdip = [0.0_f64; 3];
        for xyz in 0..3 {
            let mut val = 0.0;
            for i in 0..nocc {
                for a in 0..nvir {
                    val += amplitudes[(i * nvir + a, n)] * mu_ov[xyz][(i, a)];
                }
            }
            tdip[xyz] = std::f64::consts::SQRT_2 * val;
        }
        let mu_sq = tdip[0] * tdip[0] + tdip[1] * tdip[1] + tdip[2] * tdip[2];
        osc.push((2.0 / 3.0) * omega_n * mu_sq);
        tdips.push(tdip);
    }
    Ok((osc, tdips))
}

/// Exact-exchange coefficients of a response run (`book/src/19-tddft/04-rsh.md#rsh-split`):
/// `c_sr (pq|rs) + (c_lr − c_sr) (pq|rs)_ω`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ExchangeMix {
    pub(crate) c_sr: f64,
    pub(crate) c_lr: f64,
    pub(crate) omega: f64,
}

impl ExchangeMix {
    /// A global hybrid (or HF, `c = 1`; a pure functional, `c = 0`).
    fn global(c: f64) -> Self {
        Self {
            c_sr: c,
            c_lr: c,
            omega: 0.0,
        }
    }

    /// `Δc = c_lr − c_sr` when the long-range erf term is present (ω ≠ 0;
    /// at ω = 0 the erf operator is identically zero).
    pub(crate) fn long_range(self) -> Option<f64> {
        (self.omega != 0.0).then_some(self.c_lr - self.c_sr)
    }
}

/// Resolve the exact-exchange coefficients and (for a KS reference) the XC
/// definition the f_xc kernel will be built from.
///
/// Split out of [`run_tddft`] so the refusals are testable without an SCF, and
/// so they fire BEFORE any integral work: a functional with no complete kernel
/// must cost the user nothing but the error.
pub(crate) fn resolve_response_xc(
    xc: Option<&str>,
    c_hf_override: Option<f64>,
) -> Result<(ExchangeMix, Option<ferric_dft::libxc::XcDef>), FerricError> {
    if let Some(c) = c_hf_override.filter(|c| !c.is_finite()) {
        return Err(FerricError::General(format!(
            "run_tddft: c_hf override must be finite, got {c}"
        )));
    }
    match xc {
        None => hf_exchange_mix(c_hf_override).map(|mix| (mix, None)),
        Some(name) => functional_exchange_mix(name, c_hf_override),
    }
}

/// [`resolve_response_xc`] for an HF reference: c_HF = 1 (CIS / TDHF), and
/// any other override refused.
fn hf_exchange_mix(c_hf_override: Option<f64>) -> Result<ExchangeMix, FerricError> {
    match c_hf_override {
        Some(c) if c != 1.0 => Err(FerricError::General(format!(
            "run_tddft: c_hf = {c} on a Hartree-Fock reference (no functional) is \
             neither CIS/TDHF (c_hf = 1) nor any TDDFT — there is no f_xc kernel to \
             pair it with. Name the reference functional (config xc / [tddft] xc) or \
             drop the override."
        ))),
        _ => Ok(ExchangeMix::global(1.0)),
    }
}

/// [`resolve_response_xc`] for a named functional: libxc's coefficients, or a
/// global override (refused on a range-separated functional).
fn functional_exchange_mix(
    name: &str,
    c_hf_override: Option<f64>,
) -> Result<(ExchangeMix, Option<ferric_dft::libxc::XcDef>), FerricError> {
    let spec = resolve_kernel_xc(name, "run_tddft")?;
    let mix = ExchangeMix {
        c_sr: spec.c_sr,
        c_lr: spec.c_lr,
        omega: spec.omega,
    };
    match c_hf_override {
        None => Ok((mix, Some(spec.xc_kernel))),
        Some(c) if mix.omega != 0.0 => Err(FerricError::General(format!(
            "run_tddft: c_hf = {c} on the range-separated functional '{name}' \
             (c_sr = {}, c_lr = {}, omega = {}) does not say which range it \
             replaces. Refused; drop the override.",
            mix.c_sr, mix.c_lr, mix.omega
        ))),
        Some(c) => Ok((ExchangeMix::global(c), Some(spec.xc_kernel))),
    }
}

/// `<S²>` of a spin-adapted root of a closed-shell reference.
fn restricted_s2(spin: ResponseSpin) -> f64 {
    match spin {
        ResponseSpin::Singlet => 0.0,
        ResponseSpin::Triplet => 2.0,
    }
}

/// The refusal of an ROHF / ROKS reference
/// (`book/src/19-tddft/06-unrestricted.md#unrestricted-spin-problem`).
fn roks_refusal() -> FerricError {
    FerricError::General(
        "run_tddft: ROHF/ROKS references are refused. Their orbitals diagonalize one \
         Roothaan-coupled Fock operator, not separate alpha and beta Fock operators, so the \
         unrestricted A/B (orbital-energy differences on the diagonal) is not the response \
         of that reference; the ROKS response needs the ROHF orbital-rotation Hessian, which \
         is not implemented. Use a UHF/UKS reference."
            .to_string(),
    )
}

/// Refuse any reference `run_tddft` cannot treat, and return `nocc`.
///
/// There is no spin-resolved A/B assembly in this module: `eps_r`/`mos_r` and
/// `nocc = nelec/2` are closed-shell by construction (and `eps_r`/`mos_r`
/// ASSERT a restricted reference). [`run_tddft`] routes UHF/UKS to
/// `crate::unrestricted` before reaching this check, so it only fires for a
/// direct caller of the restricted path.
fn closed_shell_nocc(mol: &Molecule, scf: &ScfResult) -> Result<usize, FerricError> {
    if !scf.converged {
        return Err(FerricError::ScfConvergence {
            iterations: scf.iterations,
            last_energy: scf.energy,
        });
    }
    if !matches!(scf.spin, ferric_scf::Spin::Restricted) {
        return Err(FerricError::General(format!(
            "run_tddft: closed-shell (restricted) references only; got a {:?} reference. \
             Open-shell TDA/TDDFT (spin-resolved A/B and the spin-flip kernel) is not \
             implemented.",
            scf.spin
        )));
    }
    let nelec = mol.nelec() as usize;
    if !nelec.is_multiple_of(2) {
        return Err(FerricError::General(format!(
            "run_tddft: closed-shell references only; {nelec} electrons is odd"
        )));
    }
    Ok(nelec / 2)
}

/// The f_xc kernel for a KS reference (`None` for HF), on `grid` — which must
/// be the reference SCF's grid — at the restricted reference D_α = D_β =
/// `density_alpha`. Its AO-on-grid cache is budget-checked inside the kernel
/// constructor.
pub(crate) fn build_kernel(
    mol: &Molecule,
    obs: &PreparedBasis,
    xc_def: Option<ferric_dft::libxc::XcDef>,
    grid: &AtomicGridConfig,
    scf: &ScfResult,
) -> Result<Option<SingletKernel>, FerricError> {
    xc_def
        .map(|def| {
            SingletKernel::new(mol, obs.basis_set(), def, grid, &scf.density_alpha).map_err(|e| {
                FerricError::General(format!("run_tddft: f_xc kernel construction: {e}"))
            })
        })
        .transpose()
}

/// Add the `K_{ia,jb} = (ia|f_αα ± f_αβ|jb)` block of `spin` to every matrix in
/// `targets` (A for TDA; A and B for Casida — with real orbitals
/// `(ia|f_xc|jb) = (ia|f_xc|bj)`, PySCF `get_ab`'s `a += iajb; b += iajb`).
/// A no-op without a kernel (HF reference).
///
/// K is built here and dropped on return, so it is live exactly as long as
/// [`reserve_fxc_block`] assumes.
pub(crate) fn add_fxc_block(
    kernel: Option<&SingletKernel>,
    spin: ResponseSpin,
    scf: &ScfResult,
    nocc: usize,
    nvir: usize,
    targets: &mut [&mut Array2<f64>],
) -> Result<(), FerricError> {
    let Some(kern) = kernel else {
        return Ok(());
    };
    let (k_fxc, asym_rel) = fxc_ov_block(kern, spin, scf.mos_r(), 0, nocc, nocc, nvir);
    // Symmetric by construction up to grid noise; a large asymmetry is an
    // adapter bug, and symmetrizing would hide it.
    if asym_rel > FXC_BLOCK_ASYMMETRY_TOL {
        return Err(FerricError::General(format!(
            "run_tddft: f_xc (ia)-block asymmetry {asym_rel:e} exceeds the grid-noise \
             tolerance {FXC_BLOCK_ASYMMETRY_TOL:e} — the AO→(ia) adapter is inconsistent"
        )));
    }
    for t in targets.iter_mut() {
        **t += &k_fxc;
    }
    Ok(())
}

/// TDA: eigenpairs of A; the eigenvectors are the unit-norm X.
pub(crate) fn solve_tda(a: &Array2<f64>) -> Result<(Array1<f64>, Array2<f64>), FerricError> {
    a.eigh(UPLO::Lower)
        .map_err(|e| FerricError::General(format!("TDDFT TDA diagonalization failed: {e}")))
}

/// Casida: `(A−B)^{1/2}(A+B)(A−B)^{1/2} Z = Ω² Z`. Returns Ω and
/// `(X+Y)_n = Ω_n^{-1/2} (A−B)^{1/2} Z_n`, which carries `|X|² − |Y|² = 1`
/// for unit `Z_n`.
pub(crate) fn solve_casida(
    a: &Array2<f64>,
    b_mat: &Array2<f64>,
) -> Result<(Array1<f64>, Array2<f64>), FerricError> {
    let dim = a.nrows();
    let a_plus_b = a + b_mat;
    let a_minus_b = a - b_mat;

    // (A−B) should be positive-definite for stable systems
    let (amb_vals, amb_vecs) = a_minus_b
        .eigh(UPLO::Lower)
        .map_err(|e| FerricError::General(format!("TDDFT: (A-B) diagonalization failed: {e}")))?;

    // (A−B)^{1/2} = U diag(√λ) U^T
    let mut amb_sqrt = Array2::<f64>::zeros((dim, dim));
    for k in 0..dim {
        let s = amb_vals[k].sqrt();
        if !s.is_finite() {
            return Err(FerricError::General(format!(
                "TDDFT: (A-B) has negative eigenvalue {:.6e} — RPA instability",
                amb_vals[k]
            )));
        }
        for i in 0..dim {
            for j in 0..dim {
                amb_sqrt[(i, j)] += s * amb_vecs[(i, k)] * amb_vecs[(j, k)];
            }
        }
    }

    // M = (A−B)^{1/2} (A+B) (A−B)^{1/2}
    let m = amb_sqrt.dot(&a_plus_b).dot(&amb_sqrt);
    let (omega_sq, z) = m
        .eigh(UPLO::Lower)
        .map_err(|e| FerricError::General(format!("TDDFT Casida diagonalization failed: {e}")))?;
    if let Some(&w2) = omega_sq.iter().find(|&&v| v < -NEGATIVE_OMEGA_SQ) {
        return Err(negative_omega_sq_error(w2));
    }
    let omega = omega_sq.mapv(|v| if v > 0.0 { v.sqrt() } else { 0.0 });

    let mut xpy = amb_sqrt.dot(&z);
    for (n, mut col) in xpy.columns_mut().into_iter().enumerate() {
        let scale = if omega[n] > 0.0 {
            1.0 / omega[n].sqrt()
        } else {
            0.0
        };
        col.mapv_inplace(|v| v * scale);
    }
    Ok((omega, xpy))
}

/// An `Ω²` below `−NEGATIVE_OMEGA_SQ` (Ha²) is an instability of the
/// reference in the run's spin channel, refused rather than dropped from the
/// root list. The dense path's positive-root cut, mirrored.
pub(crate) const NEGATIVE_OMEGA_SQ: f64 = crate::davidson::POSITIVE_ROOT;

/// The error for a Casida `Ω² < 0` (`book/src/19-tddft/05-triplets.md#triplets-spin-blocks`).
pub(crate) fn negative_omega_sq_error(w2: f64) -> FerricError {
    FerricError::General(format!(
        "TDDFT: Casida has an imaginary root (Omega^2 = {w2:.6e} Ha^2): A+B is not \
         positive definite, so the reference is unstable in this spin channel (for \
         spin = triplet, a triplet instability). Refused rather than dropped from the \
         roots; TDA has no such failure."
    ))
}

/// Indices of the lowest `n_roots` positive eigenvalues, ascending.
fn lowest_positive_roots(eigenvalues: &Array1<f64>, n_roots: usize) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..eigenvalues.len())
        .filter(|&i| eigenvalues[i] > 1e-10)
        .collect();
    indices.sort_by(|&a, &b| eigenvalues[a].total_cmp(&eigenvalues[b]));
    indices.truncate(n_roots);
    indices
}

/// Roots of one solve: eigenvalues (ascending), amplitude columns (TDA `X`,
/// Casida `X+Y`), residuals and the Davidson iteration count.
pub(crate) struct Roots {
    pub(crate) omega: Array1<f64>,
    pub(crate) amplitudes: Array2<f64>,
    pub(crate) residuals: Vec<f64>,
    pub(crate) iterations: usize,
}

/// The `(ia)` space of a closed-shell reference.
pub(crate) struct OvSpace {
    pub(crate) nocc: usize,
    pub(crate) nvir: usize,
    pub(crate) c_occ: Array2<f64>,
    pub(crate) c_vir: Array2<f64>,
}

impl OvSpace {
    fn new(scf: &ScfResult, nocc: usize) -> Result<Self, FerricError> {
        Self::from_mos(scf.mos_r(), scf.eps_r().len(), nocc)
    }

    /// The space of `nocc` occupied and `nmo − nocc` virtual columns of `c`.
    pub(crate) fn from_mos(c: &Array2<f64>, nmo: usize, nocc: usize) -> Result<Self, FerricError> {
        // nmo, not nbasis: the SCF may have dropped near-linearly-dependent
        // combinations, and the virtual space is what the MOs span.
        let nvir = nmo.saturating_sub(nocc);
        if nocc * nvir == 0 {
            return Err(FerricError::General(
                "TDDFT: no excitations possible (nocc*nvir = 0)".to_string(),
            ));
        }
        Ok(Self {
            nocc,
            nvir,
            c_occ: c.slice(ndarray::s![.., ..nocc]).to_owned(),
            c_vir: c.slice(ndarray::s![.., nocc..nmo]).to_owned(),
        })
    }

    pub(crate) fn dim(&self) -> usize {
        self.nocc * self.nvir
    }

    /// `ε_a − ε_i` in `ia = i·nvir + a` order.
    pub(crate) fn orbital_differences(&self, eps: &[f64]) -> Array1<f64> {
        let (nocc, nvir) = (self.nocc, self.nvir);
        Array1::from_shape_fn(nocc * nvir, |ia| eps[nocc + ia % nvir] - eps[ia / nvir])
    }
}

/// The RI tensors of one operator, in standard layout (the matrix-free
/// products reshape their rows).
pub(crate) struct RiTensors {
    pub(crate) b_ov: Array2<f64>,
    pub(crate) b_oo: Array2<f64>,
    pub(crate) b_vv: Array2<f64>,
    /// The erf-operator set of a range-separated hybrid (None otherwise).
    pub(crate) lr: Option<Box<RiTensors>>,
}

impl RiTensors {
    pub(crate) fn tensors(&self) -> ExchangeTensors<'_> {
        ExchangeTensors {
            b_ov: &self.b_ov,
            b_oo: &self.b_oo,
            b_vv: &self.b_vv,
        }
    }
}

/// The Coulomb coefficient `c_J` of `spin`
/// (`book/src/19-tddft/05-triplets.md#triplets-spin-blocks`).
pub(crate) fn coulomb_factor(spin: ResponseSpin) -> f64 {
    match spin {
        ResponseSpin::Singlet => 2.0,
        ResponseSpin::Triplet => 0.0,
    }
}

/// `‖A x_n − ω_n x_n‖` for each column.
pub(crate) fn tda_residuals(a: &Array2<f64>, omega: &Array1<f64>, x: &Array2<f64>) -> Vec<f64> {
    let ax = a.dot(x);
    (0..x.ncols())
        .map(|n| {
            let r = &ax.column(n) - &(omega[n] * &x.column(n));
            r.dot(&r).sqrt()
        })
        .collect()
}

/// Casida residual of each dense root, defined as in [`crate::davidson`]:
/// with `X−Y = (A+B)(X+Y)/Ω`, `r+ = 0` and `r− = (A−B)(X−Y) − Ω (X+Y)`.
pub(crate) fn casida_residuals(
    a: &Array2<f64>,
    b_mat: &Array2<f64>,
    omega: &Array1<f64>,
    xpy: &Array2<f64>,
) -> Vec<f64> {
    let xmy = (a + b_mat).dot(xpy);
    let amb = a - b_mat;
    (0..xpy.ncols())
        .map(|n| {
            let xm = &xmy.column(n) / omega[n];
            let r = &amb.dot(&xm) - &(omega[n] * &xpy.column(n));
            (0.5 * r.dot(&r)).sqrt()
        })
        .collect()
}

/// Keep the lowest `n_roots` positive roots of a dense solve.
pub(crate) fn select_roots(
    omega: &Array1<f64>,
    amps: &Array2<f64>,
    n_roots: usize,
) -> (Array1<f64>, Array2<f64>) {
    let idx = lowest_positive_roots(omega, n_roots);
    (
        idx.iter().map(|&i| omega[i]).collect(),
        amps.select(ndarray::Axis(1), &idx),
    )
}

/// The dense path: assemble A (and B), add the f_xc block, diagonalize.
fn solve_dense(
    config: &TddftConfig,
    scf: &ScfResult,
    space: &OvSpace,
    ri: &RiTensors,
    mix: ExchangeMix,
    kernel: Option<&SingletKernel>,
    acc: &mut PhaseAccumulator,
) -> Result<Roots, FerricError> {
    let (nocc, nvir) = (space.nocc, space.nvir);
    let n_roots = config.n_roots.min(space.dim());
    let eps = scf.eps_r();
    let (spin, c_j) = (config.spin, coulomb_factor(config.spin));
    let mut a = acc.time("dense: A assembly", || {
        build_a_matrix(eps, nocc, nvir, ri, mix, c_j)
    });
    match config.method {
        TddftMethod::Tda => {
            acc.time("dense: f_xc block", || {
                add_fxc_block(kernel, spin, scf, nocc, nvir, &mut [&mut a])
            })?;
            let (w, x) = acc.time("dense: eigh", || solve_tda(&a))?;
            let (omega, amplitudes) = select_roots(&w, &x, n_roots);
            let residuals = tda_residuals(&a, &omega, &amplitudes);
            Ok(Roots {
                omega,
                amplitudes,
                residuals,
                iterations: 0,
            })
        }
        TddftMethod::Casida => {
            let mut b_mat = acc.time("dense: B assembly", || {
                build_b_matrix(nocc, nvir, ri, mix, c_j)
            });
            acc.time("dense: f_xc block", || {
                add_fxc_block(kernel, spin, scf, nocc, nvir, &mut [&mut a, &mut b_mat])
            })?;
            let (w, xpy) = acc.time("dense: eigh", || solve_casida(&a, &b_mat))?;
            let (omega, amplitudes) = select_roots(&w, &xpy, n_roots);
            let residuals = casida_residuals(&a, &b_mat, &omega, &amplitudes);
            Ok(Roots {
                omega,
                amplitudes,
                residuals,
                iterations: 0,
            })
        }
    }
}

/// Columns of the Davidson subspace and its products held at once: V and
/// A·V (TDA) or V, (A+B)·V and (A−B)·V (Casida), plus one block of new
/// vectors and their products.
pub(crate) fn reserve_davidson(plan: &mut MemoryPlan, config: &TddftConfig, dim: usize) {
    let stores = match config.method {
        TddftMethod::Tda => 2,
        TddftMethod::Casida => 3,
    };
    let space = config
        .davidson
        .max_space
        .unwrap_or_else(|| (12 * config.n_roots + 8).max(50))
        .min(dim);
    plan.reserve(
        "Davidson subspace and products",
        (stores + 1usize).saturating_mul(dim).saturating_mul(space),
        Lifetime::Resident,
    );
}

/// The matrix-free operator of one run.
fn response_operator<'a>(
    space: &OvSpace,
    eps: &[f64],
    ri: &'a RiTensors,
    mix: ExchangeMix,
    spin: ResponseSpin,
    fxc: Option<SingletOvFxc>,
) -> ResponseOperator<'a> {
    ResponseOperator {
        diag: space.orbital_differences(eps),
        coul: ri.tensors(),
        nocc: space.nocc,
        nvir: space.nvir,
        c_j: coulomb_factor(spin),
        c_hf: mix.c_sr,
        long_range: ri
            .lr
            .as_deref()
            .zip(mix.long_range())
            .map(|(lr, dc)| (lr.tensors(), dc)),
        fxc,
    }
}

/// The matrix-free path.
fn solve_davidson(
    config: &TddftConfig,
    scf: &ScfResult,
    space: &OvSpace,
    ri: &RiTensors,
    mix: ExchangeMix,
    kernel: Option<&SingletKernel>,
    plan: &mut MemoryPlan,
    acc: &mut PhaseAccumulator,
) -> Result<Roots, FerricError> {
    let (nocc, nvir) = (space.nocc, space.nvir);
    if let Some(k) = kernel {
        let doubles = SingletOvFxc::resident_doubles(
            nocc,
            nvir,
            SingletOvFxc::kernel_npts(k),
            SingletOvFxc::kernel_has_gradients(k),
        );
        plan.reserve("f_xc MO values on the grid", doubles, Lifetime::Resident);
        plan.check()?;
    }
    let fxc = acc
        .time("davidson: f_xc MO grid setup", || {
            kernel
                .map(|k| {
                    SingletOvFxc::new_with_spin(
                        k,
                        config.spin,
                        &scf.density_alpha,
                        scf.mos_r(),
                        0,
                        nocc,
                        nocc,
                        nvir,
                    )
                })
                .transpose()
        })
        .map_err(|e| FerricError::General(format!("run_tddft: f_xc operator: {e}")))?;
    let op = response_operator(space, scf.eps_r(), ri, mix, config.spin, fxc);
    let s = &config.davidson;
    let out: DavidsonOutcome = match config.method {
        TddftMethod::Tda => davidson_tda(|x| op.apply_a(x, acc), &op.diag, config.n_roots, s)?,
        TddftMethod::Casida => {
            davidson_casida(|x| op.apply_pm(x, acc), &op.diag, config.n_roots, s)?
        }
    };
    Ok(Roots {
        omega: out.eigenvalues,
        amplitudes: out.vectors,
        residuals: out.residuals,
        iterations: out.iterations,
    })
}

/// [`build_kernel`] under the "f_xc kernel setup" timer.
fn timed_kernel(
    mol: &Molecule,
    obs: &PreparedBasis,
    xc_def: Option<ferric_dft::libxc::XcDef>,
    grid: &AtomicGridConfig,
    scf: &ScfResult,
    acc: &mut PhaseAccumulator,
) -> Result<Option<SingletKernel>, FerricError> {
    acc.time("f_xc kernel setup", || {
        build_kernel(mol, obs, xc_def, grid, scf)
    })
}

/// The memory plan of one `run_tddft` call: the dense matrices (and the
/// f_xc block for a KS reference) or the Davidson working set.
fn plan_response_memory(
    config: &TddftConfig,
    solver: TddftSolver,
    dim: usize,
    has_kernel: bool,
) -> MemoryPlan {
    let label = match config.method {
        TddftMethod::Tda => "TDDFT/TDA",
        TddftMethod::Casida => "TDDFT/Casida",
    };
    let mut plan = MemoryPlan::resolve(config.memory_budget_bytes, label);
    if solver == TddftSolver::Dense {
        reserve_dense_response(&mut plan, config.method, dim);
        if has_kernel {
            reserve_fxc_block(&mut plan, config.method, dim);
        }
    } else {
        reserve_davidson(&mut plan, config, dim);
    }
    plan
}

/// The Coulomb-metric RI tensors, plus the erf-metric set for a
/// range-separated hybrid (`book/src/19-tddft/04-rsh.md#rsh-split`).
pub(crate) fn build_response_tensors(
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    space: &OvSpace,
    mix: ExchangeMix,
    config: &TddftConfig,
    plan: &mut MemoryPlan,
    acc: &mut PhaseAccumulator,
) -> Result<RiTensors, FerricError> {
    let budget = config.memory_budget_bytes;
    let (c_occ, c_vir) = (&space.c_occ, &space.c_vir);
    let (b_ov, b_oo, b_vv) = acc.time("RI B tensors", || {
        build_b_tensors(obs, dfbs, c_occ, c_vir, plan, budget)
    })?;
    let lr = match mix.long_range() {
        Some(_) => Some(Box::new(acc.time("RI erf tensors", || {
            build_lr_tensors(obs, dfbs, c_occ, c_vir, mix.omega, plan, budget)
        })?)),
        None => None,
    };
    Ok(RiTensors {
        b_ov: b_ov.as_standard_layout().into_owned(),
        b_oo: b_oo.as_standard_layout().into_owned(),
        b_vv: b_vv.as_standard_layout().into_owned(),
        lr,
    })
}

/// Oscillator strengths and transition dipoles of the roots. Triplets are
/// spin-forbidden: the spin-free dipole couples the closed-shell reference
/// to no triplet, so both are exactly zero
/// (`book/src/19-tddft/05-triplets.md#triplets-spin-blocks`).
fn transition_properties(
    obs: &PreparedBasis,
    space: &OvSpace,
    roots: &Roots,
    spin: ResponseSpin,
    acc: &mut PhaseAccumulator,
) -> Result<(Vec<f64>, Vec<[f64; 3]>), FerricError> {
    let nroots = roots.omega.len();
    match spin {
        ResponseSpin::Triplet => Ok((vec![0.0; nroots], vec![[0.0; 3]; nroots])),
        ResponseSpin::Singlet => {
            let cols: Vec<usize> = (0..nroots).collect();
            acc.time("oscillator strengths", || {
                oscillator_strengths(
                    obs,
                    &space.c_occ,
                    &space.c_vir,
                    space.nocc,
                    space.nvir,
                    &roots.omega,
                    &roots.amplitudes,
                    &cols,
                )
            })
        }
    }
}

/// Run a TDDFT (TDA or Casida) singlet calculation on a closed-shell reference.
///
/// `config.xc` names the functional the reference was converged with (`None`
/// for Hartree–Fock). It fixes both the exact-exchange fraction on the
/// `−c_HF` terms and the `(ia|f_xc|jb)` kernel block added to A and B:
///
///   * HF reference: `c_HF = 1`, no kernel — CIS (TDA) or TDHF (Casida).
///   * LDA / GGA / global hybrid / τ-only meta-GGA: `c_HF` from the
///     functional, kernel from [`ferric_dft::lr_kernel`] on `config.grid`.
///
/// Coulomb and exchange integrals are RI (Coulomb metric) over `dfbs`.
/// `config.solver` picks the dense or the Davidson eigensolver (module docs).
///
/// # Memory
///
/// Dense: `dim = nocc*nvir` grows as N², so every dense matrix here is N⁴; a
/// water/cc-pVTZ Casida run holds nine co-resident dim² matrices. The plan
/// (`reserve_dense_response`, `reserve_fxc_block`, and the RI tensors in
/// `build_b_tensors`) is counted from the actual live sets in BOTH directions:
/// under-counting OOMs, over-counting refuses jobs that would have fit.
/// Davidson: the RI tensors, the subspace (`dim × max_space` per stored
/// product) and the MO values of the f_xc operator on the grid.
///
/// # Errors
///
/// Hard errors — never a warning plus a number — for: an unconverged or
/// ROHF/ROKS reference (UHF/UKS are supported); a functional without a complete kernel (VV10);
/// a `c_hf` override on an HF reference or a range-separated hybrid; an
/// empty `(ia)` space; an over-budget allocation; an f_xc block whose
/// asymmetry exceeds the grid-noise tolerance; an RPA instability (Casida,
/// either spin, both solvers);
/// a Davidson root that does not converge.
pub fn run_tddft(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    scf: &ScfResult,
    config: &TddftConfig,
) -> Result<TddftResult, FerricError> {
    if !scf.converged {
        return Err(FerricError::ScfConvergence {
            iterations: scf.iterations,
            last_energy: scf.energy,
        });
    }
    match scf.spin {
        ferric_scf::Spin::Unrestricted => {
            return crate::unrestricted::run_unrestricted(mol, obs, dfbs, scf, config)
        }
        ferric_scf::Spin::RestrictedOpen => return Err(roks_refusal()),
        ferric_scf::Spin::Restricted => {}
    }
    let mut acc = PhaseAccumulator::new().report_on_drop("TDDFT");
    let nocc = closed_shell_nocc(mol, scf)?;
    // Functional gate before any integral work.
    let (mix, xc_def) = resolve_response_xc(config.xc.as_deref(), config.c_hf_override)?;
    let space = OvSpace::new(scf, nocc)?;
    let dim = space.dim();
    let solver = config.solver.resolve(dim, config.n_roots.min(dim));

    let mut plan = plan_response_memory(config, solver, dim, xc_def.is_some());
    let ri = build_response_tensors(obs, dfbs, &space, mix, config, &mut plan, &mut acc)?;
    let kernel = timed_kernel(mol, obs, xc_def, &config.grid, scf, &mut acc)?;
    let roots = match solver {
        TddftSolver::Davidson => solve_davidson(
            config,
            scf,
            &space,
            &ri,
            mix,
            kernel.as_ref(),
            &mut plan,
            &mut acc,
        )?,
        _ => solve_dense(config, scf, &space, &ri, mix, kernel.as_ref(), &mut acc)?,
    };

    let (oscillator_strengths, transition_dipoles) =
        transition_properties(obs, &space, &roots, config.spin, &mut acc)?;

    Ok(TddftResult {
        excitation_energies: roots.omega.to_vec(),
        oscillator_strengths,
        transition_dipoles,
        method: config.method,
        c_hf: mix.c_sr,
        c_lr: mix.c_lr,
        omega: mix.omega,
        spin: Some(config.spin),
        reference: ferric_scf::Spin::Restricted,
        s2: vec![restricted_s2(config.spin); roots.omega.len()],
        s2_reference: 0.0,
        fxc_included: kernel.is_some(),
        solver,
        residuals: roots.residuals,
        iterations: roots.iterations,
    })
}

/// [`run_tddft`] on the system `aux` was prepared on; see [`PreparedAux`] for
/// what such an entry point reads from the system.
pub fn run_tddft_on(
    aux: &PreparedAux<'_>,
    scf: &ScfResult,
    config: &TddftConfig,
) -> Result<TddftResult, FerricError> {
    let sys = aux.system();
    run_tddft(sys.mol(), sys.basis(), aux.basis(), scf, config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hf_reference_is_cis_tdhf_with_no_kernel() {
        let (mix, def) = resolve_response_xc(None, None).unwrap();
        assert_eq!(mix, ExchangeMix::global(1.0));
        assert!(def.is_none(), "an HF reference has no f_xc kernel");
    }

    #[test]
    fn ks_references_always_get_a_kernel() {
        // The defect this pins: a KS reference used to run with NO kernel
        // (a warning and a kernel-less number). Every accepted functional must
        // now come back with a kernel definition.
        for (name, want_c_hf) in [("LDA", 0.0), ("PBE", 0.0), ("B3LYP", 0.2)] {
            let (mix, def) = resolve_response_xc(Some(name), None)
                .unwrap_or_else(|e| panic!("{name} must be accepted: {e}"));
            let c_hf = mix.c_sr;
            assert!(mix.long_range().is_none(), "{name}: not range-separated");
            assert!(def.is_some(), "{name}: accepted without an f_xc kernel");
            assert!(
                (c_hf - want_c_hf).abs() < 1e-12,
                "{name}: c_HF {c_hf} != {want_c_hf}"
            );
        }
    }

    #[test]
    fn tau_only_meta_gga_is_accepted_with_its_kernel() {
        for name in ["SCAN", "r2SCAN"] {
            let (mix, def) = resolve_response_xc(Some(name), None)
                .unwrap_or_else(|e| panic!("{name} has a tau f_xc kernel: {e}"));
            assert!(def.is_some(), "{name}: accepted without an f_xc kernel");
            assert_eq!(mix, ExchangeMix::global(0.0), "{name}: c_HF must be 0");
        }
    }

    #[test]
    fn vv10_is_refused_and_range_separation_resolved() {
        assert!(resolve_response_xc(Some("wB97X-V"), None).is_err());
        let (mix, def) = resolve_response_xc(Some("HYB_GGA_XC_CAM_B3LYP"), None).unwrap();
        assert!(def.is_some());
        let dc = mix.long_range().expect("CAM-B3LYP is range-separated");
        assert!(
            (dc - 0.46).abs() < 1e-12 && (mix.omega - 0.33).abs() < 1e-12,
            "{mix:?}"
        );
        // A c_hf override does not say which range it replaces.
        assert!(resolve_response_xc(Some("HYB_GGA_XC_CAM_B3LYP"), Some(0.2)).is_err());
    }

    #[test]
    fn c_hf_override_on_an_hf_reference_is_refused() {
        assert!(resolve_response_xc(None, Some(0.2)).is_err());
        // c_hf = 1 on an HF reference is just CIS/TDHF spelled out.
        assert_eq!(
            resolve_response_xc(None, Some(1.0)).unwrap().0,
            ExchangeMix::global(1.0)
        );
        // With a functional, an override is an explicit model change.
        assert_eq!(
            resolve_response_xc(Some("PBE"), Some(0.25)).unwrap().0,
            ExchangeMix::global(0.25)
        );
        assert!(resolve_response_xc(Some("PBE"), Some(f64::NAN)).is_err());
    }

    // ── Davidson route against the dense route ─────────────────────────────

    use ferric_core::parallel::ParallelContext;
    use ferric_scf::rhf::{solve_rhf, RhfConfig};
    use ferric_scf::screening::SchwarzBounds;

    struct Case {
        mol: Molecule,
        obs: PreparedBasis,
        dfbs: PreparedBasis,
        scf: ScfResult,
    }

    fn water(basis: &str, xc: Option<&str>) -> Case {
        let xyz = format!(
            "{}/../../testdata/molecules/water.xyz",
            env!("CARGO_MANIFEST_DIR")
        );
        let mol = Molecule::load_xyz(&xyz).unwrap();
        let obs = PreparedBasis::new(&mol, &ferric_core::basis::bundled(basis).unwrap()).unwrap();
        let dfbs =
            PreparedBasis::new(&mol, &ferric_core::basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let mut cfg = RhfConfig::default();
        cfg.xc = xc.map(str::to_string);
        cfg.density_conv = 1e-9;
        cfg.max_iter = 200;
        let scf = solve_rhf(&ParallelContext::default(), &mol, &obs, op, &bounds, &cfg).unwrap();
        assert!(scf.converged);
        Case {
            mol,
            obs,
            dfbs,
            scf,
        }
    }

    fn spectral_norm(m: &Array2<f64>) -> f64 {
        let s = 0.5 * (m + &m.t());
        let (w, _) = s.eigh(UPLO::Lower).unwrap();
        w.iter().fold(0.0_f64, |x, &y| x.max(y.abs()))
    }

    fn frobenius(m: &Array2<f64>) -> f64 {
        m.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    fn gap(w: &Array1<f64>, k: usize) -> f64 {
        (0..w.len())
            .filter(|&j| j != k)
            .map(|j| (w[j] - w[k]).abs())
            .fold(f64::INFINITY, f64::min)
    }

    /// Dense A, B (f_xc included), the matrix-free operator densified by
    /// applying it to the identity, and the run's `(ia)` space.
    struct Densified {
        a: Array2<f64>,
        b: Array2<f64>,
        a_mf: Array2<f64>,
        p_mf: Array2<f64>,
        m_mf: Array2<f64>,
        /// Elementwise rounding bound of the RI terms (see the test docs).
        ri_bar: f64,
        /// f_xc floor: `FXC_BLOCK_ASYMMETRY_TOL · max|K|` (0 for HF).
        fxc_bar: f64,
    }

    /// Dense A, B for `spin`, with the exchange coefficients of `xc` or, when
    /// given, `mix` (the f_xc kernel still comes from `xc`).
    fn densify_with(
        c: &Case,
        xc: Option<&str>,
        spin: ResponseSpin,
        mix: Option<ExchangeMix>,
    ) -> Densified {
        let nocc = closed_shell_nocc(&c.mol, &c.scf).unwrap();
        let (mix0, def) = resolve_response_xc(xc, None).unwrap();
        let mix = mix.unwrap_or(mix0);
        let space = OvSpace::new(&c.scf, nocc).unwrap();
        let (nvir, dim) = (space.nvir, space.dim());
        let mut plan = MemoryPlan::resolve(None, "test");
        let (b_ov, b_oo, b_vv) =
            build_b_tensors(&c.obs, &c.dfbs, &space.c_occ, &space.c_vir, &mut plan, None).unwrap();
        let lr = mix.long_range().map(|_| {
            Box::new(
                build_lr_tensors(
                    &c.obs,
                    &c.dfbs,
                    &space.c_occ,
                    &space.c_vir,
                    mix.omega,
                    &mut plan,
                    None,
                )
                .unwrap(),
            )
        });
        let ri = RiTensors {
            b_ov: b_ov.as_standard_layout().into_owned(),
            b_oo: b_oo.as_standard_layout().into_owned(),
            b_vv: b_vv.as_standard_layout().into_owned(),
            lr,
        };
        let grid = AtomicGridConfig::production();
        let kernel = build_kernel(&c.mol, &c.obs, def, &grid, &c.scf).unwrap();
        let eps = c.scf.eps_r();
        let c_j = coulomb_factor(spin);
        let mut a = build_a_matrix(eps, nocc, nvir, &ri, mix, c_j);
        let mut b = build_b_matrix(nocc, nvir, &ri, mix, c_j);
        let mut fxc_bar = 0.0;
        if let Some(k) = kernel.as_ref() {
            let (kd, _) = fxc_ov_block(k, spin, c.scf.mos_r(), 0, nocc, nocc, nvir);
            fxc_bar = FXC_BLOCK_ASYMMETRY_TOL * kd.iter().fold(0.0_f64, |m, &x| m.max(x.abs()));
        }
        add_fxc_block(
            kernel.as_ref(),
            spin,
            &c.scf,
            nocc,
            nvir,
            &mut [&mut a, &mut b],
        )
        .unwrap();
        let fxc = kernel.as_ref().map(|k| {
            SingletOvFxc::new_with_spin(
                k,
                spin,
                &c.scf.density_alpha,
                c.scf.mos_r(),
                0,
                nocc,
                nocc,
                nvir,
            )
            .unwrap()
        });
        let op = response_operator(&space, eps, &ri, mix, spin, fxc);
        let mut acc = PhaseAccumulator::new();
        let eye = Array2::<f64>::eye(dim);
        let a_mf = op.apply_a(&eye, &mut acc).unwrap();
        let (p_mf, m_mf) = op.apply_pm(&eye, &mut acc).unwrap();
        // Each RI element is a naux-term dot product, evaluated in a
        // different order by the two routes: each is within γ_n Σ|terms| of
        // the exact sum, n = naux. Σ|terms| is bounded by the largest element
        // of |B|ᵀ|B| for each term, weighted by the coefficient it enters with
        // (|A+B| has 2·c_J Coulomb + 2·c_sr exchange, + 2·|Δc| erf exchange);
        // the two routes together, 2γ_n times that.
        let naux = ri.b_ov.nrows() as f64;
        let gamma = naux * f64::EPSILON / (1.0 - naux * f64::EPSILON);
        let maxabs = |m: &Array2<f64>| m.iter().fold(0.0_f64, |x, &y| x.max(y.abs()));
        let sums = |t: &RiTensors| {
            let s_j = maxabs(&t.b_ov.mapv(f64::abs).t().dot(&t.b_ov.mapv(f64::abs)));
            let s_k = maxabs(&t.b_oo.mapv(f64::abs).t().dot(&t.b_vv.mapv(f64::abs))).max(s_j);
            (s_j, s_k)
        };
        let (s_j, s_k) = sums(&ri);
        let mut ri_bar = 2.0 * gamma * (2.0 * c_j * s_j + 2.0 * mix.c_sr.abs() * s_k);
        if let (Some(lr), Some(dc)) = (ri.lr.as_deref(), mix.long_range()) {
            ri_bar += 2.0 * gamma * 2.0 * dc.abs() * sums(lr).1;
        }
        Densified {
            a,
            b,
            a_mf,
            p_mf,
            m_mf,
            ri_bar,
            fxc_bar,
        }
    }

    /// The matrix-free products, applied to the identity, reproduce the dense
    /// A, A+B and A−B elementwise, water/cc-pVDZ, HF, PBE and B3LYP.
    ///
    /// Bar: `ri_bar` (2γ_naux times the largest |B|ᵀ|B| magnitude sum, per
    /// term coefficient; derivation in `densify`) plus, for twice the f_xc
    /// block on A+B, `2·FXC_BLOCK_ASYMMETRY_TOL·max|K|` (the floor the
    /// matrix-free f_xc block is pinned at against the dense one,
    /// `ferric_dft::lr_kernel` `ov_operator_matches_the_dense_block`).
    ///
    /// Mutations (applied and reverted, 2026-11-13): dropping f_xc from `A x`
    /// fails PBE (A off by 9.2e-2 against a 1.8e-9 bar), and from `(A+B) x`
    /// (1.8e-1); halving the Coulomb factor fails HF on A (2 → 1: 1.6e-1
    /// against 9.8e-14) and on A+B (4 → 2: 3.2e-1). Each also fails the
    /// PySCF-referenced `validation_tddft` case through the Davidson route
    /// (`h2co_631g_pbe`, `h2co_631g_hf`: an oscillator strength off by
    /// 3e-2 to 3e-1). The roots test below
    /// does not catch these by design: its bar includes the measured operator
    /// difference.
    #[test]
    fn matrix_free_products_equal_the_dense_matrices() {
        for (xc, spin) in SPIN_CASES {
            let c = water("cc-pvdz", xc);
            let d = densify_with(&c, xc, spin, None);
            let maxdiff = |x: &Array2<f64>, y: &Array2<f64>| {
                (x - y).iter().fold(0.0_f64, |m, &v| m.max(v.abs()))
            };
            let bar = d.ri_bar + 2.0 * d.fxc_bar;
            let da = maxdiff(&d.a_mf, &d.a);
            let dp = maxdiff(&d.p_mf, &(&d.a + &d.b));
            let dm = maxdiff(&d.m_mf, &(&d.a - &d.b));
            eprintln!("{xc:?} {spin:?}: A {da:.2e}, A+B {dp:.2e}, A-B {dm:.2e} (bar {bar:.2e})");
            assert!(da <= bar, "{xc:?} {spin:?}: A differs by {da:e} > {bar:e}");
            assert!(
                dp <= bar,
                "{xc:?} {spin:?}: A+B differs by {dp:e} > {bar:e}"
            );
            assert!(
                dm <= bar,
                "{xc:?} {spin:?}: A-B differs by {dm:e} > {bar:e}"
            );
        }
    }

    /// The (functional, spin) cases of the dense / matrix-free comparisons:
    /// HF, a GGA, a global and a range-separated hybrid, singlets and triplets.
    const SPIN_CASES: [(Option<&str>, ResponseSpin); 8] = [
        (None, ResponseSpin::Singlet),
        (Some("PBE"), ResponseSpin::Singlet),
        (Some("B3LYP"), ResponseSpin::Singlet),
        (Some("HYB_GGA_XC_CAM_B3LYP"), ResponseSpin::Singlet),
        (None, ResponseSpin::Triplet),
        (Some("PBE"), ResponseSpin::Triplet),
        (Some("B3LYP"), ResponseSpin::Triplet),
        (Some("HYB_GGA_XC_CAM_B3LYP"), ResponseSpin::Triplet),
    ];

    /// Exactness anchors of the range-separated exchange (`book/src/19-tddft/04-rsh.md#rsh-split`),
    /// water/cc-pVDZ on a B3LYP reference and kernel, singlet and triplet:
    ///
    /// * ω = 0 with `c_lr ≠ c_sr` (0.20, 0.65) assembles A, B and the
    ///   matrix-free A, A±B BITWISE equal to the global hybrid `c = c_sr`:
    ///   erf(0) = 0, and no erf tensors are built.
    /// * `c_sr = c_lr` = 0.20 at ω = 0.33: the erf tensors ARE built and
    ///   contracted, weighted by `Δc = 0`, so again bitwise. A long-range
    ///   term weighted by `c_lr` instead of `Δc` fails this.
    /// * Reachability: `c_lr` = 0.65 at ω = 0.33 moves A by far more than
    ///   rounding, so the comparison can see a long-range term.
    ///
    /// Mutation (2026-11-14): weighting the long-range term by `c_lr` in
    /// `ExchangeMix::long_range` fails the second anchor (A differs by
    /// 6.9e-2); dropping the long-range term fails the reachability check.
    #[test]
    fn range_separated_exchange_reduces_to_the_global_hybrid() {
        let xc = Some("B3LYP");
        let c = water("cc-pvdz", xc);
        let same = |x: &Densified, y: &Densified| {
            x.a == y.a && x.b == y.b && x.a_mf == y.a_mf && x.p_mf == y.p_mf && x.m_mf == y.m_mf
        };
        let maxdiff =
            |x: &Array2<f64>, y: &Array2<f64>| (x - y).iter().fold(0.0_f64, |m, &v| m.max(v.abs()));
        for spin in [ResponseSpin::Singlet, ResponseSpin::Triplet] {
            let mix = |c_sr, c_lr, omega| ExchangeMix { c_sr, c_lr, omega };
            let global = densify_with(&c, xc, spin, Some(ExchangeMix::global(0.2)));
            let w0 = densify_with(&c, xc, spin, Some(mix(0.2, 0.65, 0.0)));
            assert!(
                same(&w0, &global),
                "{spin:?}: omega = 0 is not the global hybrid"
            );
            let flat = densify_with(&c, xc, spin, Some(mix(0.2, 0.2, 0.33)));
            assert!(
                same(&flat, &global),
                "{spin:?}: c_sr = c_lr is not the global hybrid (A {:e}, A_mf {:e})",
                maxdiff(&flat.a, &global.a),
                maxdiff(&flat.a_mf, &global.a_mf)
            );
            let rsh = densify_with(&c, xc, spin, Some(mix(0.2, 0.65, 0.33)));
            let moved = maxdiff(&rsh.a, &global.a);
            eprintln!("{spin:?}: the long-range term moves A by {moved:.3e}");
            assert!(
                moved > 1e6 * rsh.ri_bar,
                "{spin:?}: long-range term invisible"
            );
        }
    }

    fn run_with(
        c: &Case,
        xc: Option<&str>,
        spin: ResponseSpin,
        method: TddftMethod,
        solver: TddftSolver,
    ) -> TddftResult {
        let cfg = TddftConfig {
            n_roots: 5,
            method,
            xc: xc.map(str::to_string),
            grid: AtomicGridConfig::production(),
            solver,
            spin,
            ..TddftConfig::default()
        };
        run_tddft(&c.mol, &c.obs, &c.dfbs, &c.scf, &cfg).unwrap()
    }

    /// `run_tddft` through the Davidson route equals the dense route,
    /// water/cc-pVDZ (dim 95; the default subspace of 50 restarts), HF, PBE
    /// and B3LYP, TDA and Casida.
    ///
    /// Bars (derived), per root k with Davidson residual r:
    ///
    /// * TDA: `‖r‖²/gap_k` (the Ritz value is within that of an eigenvalue of
    ///   the matrix-free operator Ã) + `‖Ã − A‖₂ ≤ ‖Ã − A‖_F` (Bauer–Fike
    ///   for the symmetric A) + `dim·ε·‖A‖₂` (the dense `eigh`).
    /// * Casida: Ω² is an eigenvalue of `(A−B)(A+B) = S H S⁻¹`
    ///   (`S = (A−B)^½`), so a perturbation `(E−, E+)` of `(A−B, A+B)` moves
    ///   it by at most `cond(S)·(‖E−‖‖A+B‖ + ‖A−B‖‖E+‖ + ‖E−‖‖E+‖)`
    ///   (Bauer–Fike with eigenvector matrix `S U`); the Davidson pair adds
    ///   `‖r_H‖/‖z‖` for `z = S(X−Y)`, whose H residual is
    ///   `r_H = S(Ω r+ + (A+B) r−)`: with `‖r±‖ ≤ √2 r`, `‖z‖ ≥ σ_min(S)‖X−Y‖`
    ///   and `‖X−Y‖ ≥ 1/‖X+Y‖` (`(X+Y)·(X−Y) = 1`), that is at most
    ///   `√2 r (Ω + ‖A+B‖) cond(S) ‖X+Y‖`, with `‖X+Y‖² ≈ Ω/λ_min(A+B)`; the
    ///   dense solve adds `dim·ε·‖A−B‖‖A+B‖`. Divided by `Ω + Ω_dense`.
    #[test]
    fn davidson_roots_equal_the_dense_roots() {
        for (xc, spin) in SPIN_CASES {
            let c = water("cc-pvdz", xc);
            let d = densify_with(&c, xc, spin, None);
            let dim = d.a.nrows() as f64;
            let eps = f64::EPSILON;
            // TDA.
            let dense = run_with(&c, xc, spin, TddftMethod::Tda, TddftSolver::Dense);
            let dav = run_with(&c, xc, spin, TddftMethod::Tda, TddftSolver::Davidson);
            assert_eq!(dav.solver, TddftSolver::Davidson);
            assert!(dav.iterations > 0);
            let (w_all, _) = solve_tda(&d.a).unwrap();
            let e_a = frobenius(&(&d.a_mf - &d.a));
            let floor = dim * eps * spectral_norm(&d.a);
            for k in 0..5 {
                let r = dav.residuals[k];
                assert!(r <= TddftConfig::default().davidson.conv_tol);
                let bar = r * r / gap(&w_all, k) + e_a + floor;
                let de = (dav.excitation_energies[k] - dense.excitation_energies[k]).abs();
                eprintln!("{xc:?} {spin:?} TDA root {k}: |dw| {de:.2e} (bar {bar:.2e}, r {r:.1e})");
                assert!(de <= bar, "{xc:?} TDA root {k}: |dw| {de:e} > bar {bar:e}");
            }
            // Casida.
            let dense = run_with(&c, xc, spin, TddftMethod::Casida, TddftSolver::Dense);
            let dav = run_with(&c, xc, spin, TddftMethod::Casida, TddftSolver::Davidson);
            let apb = &d.a + &d.b;
            let amb = &d.a - &d.b;
            let (lam, _) = amb.eigh(UPLO::Lower).unwrap();
            let (lmin, lmax) = (lam[0], lam[lam.len() - 1]);
            let cond = (lmax / lmin).sqrt();
            let e_p = frobenius(&(&d.p_mf - &apb));
            let e_m = frobenius(&(&d.m_mf - &amb));
            let (n_p, n_m) = (spectral_norm(&apb), lmax);
            let (lam_p, _) = apb.eigh(UPLO::Lower).unwrap();
            let pert = cond * (e_m * n_p + n_m * e_p + e_m * e_p);
            let floor = dim * eps * n_m * n_p;
            for k in 0..5 {
                let r = dav.residuals[k];
                let (om, om_d) = (dav.excitation_energies[k], dense.excitation_energies[k]);
                let xpy_norm = (om / lam_p[0]).sqrt();
                let solve = 2f64.sqrt() * r * (om + n_p) * cond * xpy_norm;
                let bar = (pert + solve + floor) / (om + om_d);
                let de = (om - om_d).abs();
                eprintln!(
                    "{xc:?} {spin:?} Casida root {k}: |dW| {de:.2e} (bar {bar:.2e}, r {r:.1e})"
                );
                assert!(
                    de <= bar,
                    "{xc:?} Casida root {k}: |dW| {de:e} > bar {bar:e}"
                );
            }
        }
    }
}
