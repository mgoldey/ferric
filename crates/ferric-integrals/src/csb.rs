//! CSB: the **combined Schwarz bound**, a RIGOROUS upper bound on
//! `|(µν|λσ)|` that is never looser than plain Schwarz.
//!
//! # Primary source, quoted
//!
//! Travis H. Thompson and Christian Ochsenfeld, "Distance-including rigorous
//! upper bounds and tight estimates for two-electron integrals over long- and
//! short-range operators", J. Chem. Phys. **147**, 144101 (2017),
//! doi:10.1063/1.4994190. Reprinted verbatim as "Publication I" of Thompson's
//! LMU dissertation (`edoc.ub.uni-muenchen.de/26208`), from which the text
//! below was read directly rather than reconstructed from memory.
//!
//! Section II A, "A rigorous upper bound for short-range operators":
//!
//! > The QQ estimate is the Schwarz inequality on the space of one-electron
//! > charge distributions defined as Ω_µν(r₁) = χ_µ(r₁)χ_ν(r₁). It is
//! > applicable due to the fact that the two-electron integral over some
//! > distance-dependent operator G is an inner product on this space, as long
//! > as G is positive definite. We show in Appendix A that the integral is
//! > also an inner product on the space of real continuous two-electron
//! > functions, if G is strictly positive for r₁₂ > 0. The two-electron
//! > orbital products Ω̃_µν(r₁, r₂) = χ_µ(r₁)χ_ν(r₂) are a subset of this
//! > space and the corresponding Schwarz inequality leads, due to symmetry,
//! > to the following two upper bounds:
//! >
//! > ```text
//! >   |(µν|λσ)| ≤ (µµ|λλ)^(1/2) (νν|σσ)^(1/2) = M_µλ M_νσ ,      (6)
//! >   |(µν|λσ)| ≤ (µµ|σσ)^(1/2) (νν|λλ)^(1/2) = M_µσ M_νλ .      (7)
//! > ```
//! >
//! > In contrast to the original QQ estimates, the right-hand sides of (6) and
//! > (7) inherently contain distance dependence, while lacking any overlap
//! > dependence. Equality holds in (6) and (7) for integrals of type (µµ|νν),
//! > i.e., for "perfect" overlap in forming the charge distributions. We
//! > combine the original QQ estimates (1) and the Schwarz-type inequalities
//! > (6) and (7) into a rigorous distance-including upper bound, the combined
//! > Schwarz bound (CSB),
//! >
//! > ```text
//! >   |(µν|λσ)| ≤ min{ Q_µν Q_λσ , M_µλ M_νσ , M_µσ M_νλ } .     (8)
//! > ```
//!
//! and, closing Sec. II A:
//!
//! > Lastly, we note that the CSB estimates are exact for both (µν|µν)-type
//! > and (µµ|νν)-type integrals. This is because the QQ estimates are exact
//! > for the former integral types, while Eqs. (6) and (7) are exact for the
//! > latter, and all three are upper bounds.
//!
//! # CSB is rigorous; CSAM is NOT — they are different formulas
//!
//! Immediately after Eq. (8) the paper introduces a SECOND, different family.
//! Eq. (9) defines a NORMALIZED quantity
//!
//! ```text
//!   M̃_µλ = M_µλ / ( Q_µµ^(1/2) Q_λλ^(1/2) )                     (9)
//! ```
//!
//! and the paper then says, in its own words:
//!
//! > We use M̃_µλ to formulate our non-rigorous combined Schwarz
//! > approximations (CSAs),
//! >
//! > ```text
//! >   |(µν|λσ)| ≈ Q_µν Q_λσ M_µνλσ ,                            (11)
//! > ```
//!
//! with `M_µνλσ ∈ { M̃_µλ M̃_νσ (CSA1), M̃_µσ M̃_νλ (CSA2),
//! max(M̃_µλ M̃_νσ, M̃_µσ M̃_νλ) (CSAM) }` per Eq. (12). Note the `≈`, not `≤`.
//!
//! **This module implements Eq. (8) only.** It does NOT implement Eqs.
//! (9)/(11)/(12). Psi4's `SCREENING=CSAM` ships the non-rigorous family;
//! ferric's separate `csam` work (branch `perf/csam-screening`) covers that,
//! and measured it violating by 218,493× on benzene/cc-pVDZ under `erfc(1.0)`.
//! The distinction is not stylistic:
//!
//! * CSAM **multiplies** the Schwarz product by a ratio in `[0,1]`, so a
//!   collapsed ratio drags the estimate arbitrarily far BELOW the truth.
//! * CSB takes a **`min` against that same Schwarz product**, so no matter how
//!   small the `M` factors get, the result can never exceed `Q_µν Q_λσ` and —
//!   given a correct `M` table — can never fall below `|(µν|λσ)|` either,
//!   because each of the three arguments of the `min` is independently a true
//!   upper bound.
//!
//! ## Why the `min` does NOT make M-table correctness optional
//!
//! Read this before assuming the `min` is a safety net in both directions. It
//! is not. The `min` guarantees only `CSB ≤ Schwarz` (the TIGHTNESS half). The
//! VALIDITY half — `CSB ≥ |(µν|λσ)|` — rests entirely on the `M` entries being
//! genuine `sqrt(|(µµ|λλ)|)` values, and an `M` that is too SMALL wins the
//! `min` and breaks the bound. That is precisely the direction the 218,493×
//! CSAM failure went, and precisely why this module inherits
//! [`crate::schwarz`]'s precision discipline verbatim (see below) rather than
//! treating the `min` as absolution.
//!
//! The payoff of the `min` is a different, real one: a defect in the `M` table
//! can only ever make CSB TIGHTER than Schwarz, never looser, so the failure
//! has a single sign and a single test (`csb_is_a_valid_upper_bound`) catches
//! the whole class. Without the `min`, an `M` bug could also make the bound
//! spuriously LOOSE, which no validity test would ever flag.
//!
//! # Rigor conditions (Appendix B), and what they mean for ferric's operators
//!
//! Appendix B, "A note on the applicability of the Schwarz inequality (QQ
//! bound) to integrals over multiplicative, distance based operators", gives
//! the condition. Quoting it:
//!
//! > Thus, when F_G has the positivity property defined in Appendix A, the
//! > same arguments given there apply in this case so that conditions four and
//! > five follow analogously, and the Schwarz inequality can be used with the
//! > operator G. The condition F_G(k) > 0 for k > 0 is, according to Bochner's
//! > theorem, equivalent to the statement that G is a positive-definite
//! > [function].
//!
//! where `F_G` is the 3-D Fourier transform of `G`. Table VI of the paper
//! ("Fourier transforms for some important operators in quantum chemical
//! theories. The parameters γ, ω, and α are positive real numbers. **All
//! transforms are strictly positive for k > 0.**") lists, verbatim:
//!
//! ```text
//!   Operator            Fourier transform
//!   1/r12               1 / (2π²k²)
//!   e^(-γ r12)          γ / (π²(k²+γ²)²)
//!   e^(-γ r12)/r12      1 / (2π²(k²+γ²))
//!   erf(ω r12)/r12      (1/(2π²k²)) e^(-k²/(4ω²))
//!   erfc(ω r12)/r12     (1/(2π²k²)) (1 - e^(-k²/(4ω²)))
//!   e^(-α r12²)         (1/(8π^(3/2) α^(3/2))) e^(-k²/(4α))
//! ```
//!
//! So the bound is proven for Coulomb, Yukawa, Slater-geminal, **erf** and
//! **erfc**, and the Gaussian geminal — i.e. it covers every operator
//! [`crate::schwarz::schwarz`] itself accepts, and more.
//!
//! ## Extending Table VI to ferric's `Terfc` / `Terf` (2026-09-14)
//!
//! ferric has two operators the paper does not discuss, implemented via the
//! standalone 2-D interpolation-table engine (Dutoi & Head-Gordon, JPCA 2008;
//! Goldey PhD thesis 2014) rather than libint2. Appendix B is not just a list
//! — it is a *recipe*: compute `F_G(k)` and check positivity. Applying that
//! recipe settles both, with opposite answers.
//!
//! The exact functional form, read from the shim rather than the name
//! (`crates/ferric-integrals/shim/shim.cc:717-718`):
//!
//! ```text
//!   terfc(r,r0)/r = 1/r  -  terf(r,r0)/r
//!   terf(r,r0)/r  = ( erf(w(r-r0)) + erf(w(r+r0)) ) / (2 r),   w = 1/(r0 sqrt2)
//! ```
//!
//! i.e. `G_terfc(r) = t(r)/r` with `t(r) = 1 - [erf(w(r-r0)) + erf(w(r+r0))]/2`,
//! a smoothed step falling from `t(0) = 1` to `t(inf) = 0` with its edge at
//! `r = r0` and edge width `~1/w`. (Note: `terf-tables/base_terfc_closed.py`'s
//! docstring writes this WITHOUT the factor of 2. That form goes negative past
//! `r ~ r0` and is not the shipped kernel; the shim and
//! `scripts/terfc_pd_check.py` agree on the `/2`, and
//! `tests/terfc_base_validation.rs` asserts `terfc/coulomb` stays in `(0,1)`,
//! which the un-halved form would violate. Flagged, not silently corrected.)
//!
//! **`Terfc` IS positive-definite — with a closed form, not a sweep.** Since
//! `t(r) - 1` is odd (`t(r) + t(-r) = 2`, checked numerically), `t(r) sin(kr)`
//! decomposes so that the radial transform integrates in closed form:
//!
//! ```text
//!   F_terfc(k) = (1 / (2 pi^2 k^2)) * ( 1 - cos(k r0) * e^{-k^2/(4 w^2)} )
//! ```
//!
//! This was VERIFIED against direct oscillatory quadrature of
//! `int_0^inf t(r) sin(kr) dr` at 72 combinations of
//! `r0 in {0.7, 1.0, 1.98}`, `c = r0 w in {0.3, 1/sqrt2, 2.06987, 8.0}` and
//! `k` spanning the danger points `2 pi m / r0`: **max absolute deviation
//! 5.67e-15**.
//!
//! Positivity is then immediate and needs no numerics at all: `|cos(k r0)| <= 1`
//! and `e^{-k^2/(4 w^2)} < 1` STRICTLY for every `k > 0` and finite `w`, so the
//! bracket is `>= 1 - e^{-k^2/(4w^2)} > 0`. By Bochner's theorem (the paper's
//! Appendix B) `G_terfc` is therefore a positive-definite function, and both
//! the QQ bound and CSB apply to it rigorously.
//!
//! Two independent consistency checks on that closed form:
//!
//!   * `r0 -> 0` reduces it to `(1 - e^{-k^2/(4 w^2)}) / (2 pi^2 k^2)` — the
//!     paper's own Table VI row for `erfc(w r12)/r12`, exactly. So this is a
//!     strict GENERALIZATION of a row the authors derived, not a new claim
//!     sitting beside it.
//!   * `w -> inf` (sharp cutoff) gives `(1 - cos(k r0))/(2 pi^2 k^2) >= 0`,
//!     which touches zero at `k = 2 pi m / r0`. The finite edge is exactly what
//!     lifts those touch points off zero: the `e^{-k^2/(4 w^2)}` factor
//!     multiplies the `cos` term down. This identifies the sharp-cutoff limit
//!     as the boundary case and explains WHY the smoothing matters, rather
//!     than leaving positivity as an unexplained numerical fact.
//!
//! This also corroborates, and upgrades, ferric's pre-existing sweep
//! (`scripts/terfc_pd_check.py` -> `wiki/data/terfc-pd-sweep.txt`, 2026-08-13),
//! which found `P(u) >= 5.0e-4 > 0` over `c in [0.2, 50]` numerically. A sweep
//! can only ever fail to find a violation; the closed form proves there is
//! none, for ALL `r0` and `w`, including the decoupled `(r0, omega)` family.
//!
//! **`Terf` is NOT positive-definite, and is therefore OUT OF SCOPE for CSB
//! and for plain Schwarz alike.** Since `terf = Coulomb - terfc`:
//!
//! ```text
//!   F_terf(k) = 1/(2 pi^2 k^2) - F_terfc(k)
//!             = cos(k r0) * e^{-k^2/(4 w^2)} / (2 pi^2 k^2)
//! ```
//!
//! and `cos(k r0)` changes sign — `F_terf(k) < 0` for every `k` with
//! `k r0 in (pi/2, 3 pi/2) mod 2 pi`, e.g. `k r0 = pi` where `cos = -1`.
//! Bochner's condition FAILS. Appendix A's requirement four (`<e,e> >= 0`) is
//! not merely unproven for `terf`, it is false, so the two-electron integral
//! over `terf` is NOT an inner product and the Cauchy-Schwarz step underlying
//! BOTH `Q` and `M` is invalid. This is not a tightness question: a "Schwarz
//! bound" for `terf` could be violated outright.
//!
//! Note the contrast with the paper's `erf` row, which IS positive
//! (`e^{-k^2/(4w^2)}/(2 pi^2 k^2)`, no oscillation): `terf` is the *tempered*
//! long-range complement, and it is the `cos(k r0)` introduced by the finite
//! shell radius `r0` — not the long-rangedness — that breaks positivity.
//!
//! ### What is NOT yet implemented for `Terfc`, despite being valid
//!
//! CSB needs `(PP|QQ)` (for `M`) and `(PQ|PQ)` (for `Q`) SHELL QUARTETS. The
//! terfc engine exposes only 3-centre and 2-centre entry points —
//! `scf_compute_terfc_eri3` / `scf_compute_terfc_eri2`
//! (`crates/ferric-integrals/shim/shim.h:201,206`); there is no
//! `scf_compute_terfc_eri_quartet`. So neither `csb_m_table` nor
//! `crate::schwarz::schwarz` can be built for `Terfc` today, and both correctly
//! reject it at their `match op.kind` — for a MISSING-ENGINE reason, not a
//! positivity one. The distinction matters: if a 4-centre terfc kernel is ever
//! written, CSB becomes available immediately with no new theory, whereas for
//! `Terf` no engine would make it valid.
//!
//! One caveat that would have to be settled first, and cannot be settled here:
//! the terfc engine is INTERPOLATED (2-D tables in `(S, s)`), so integrals
//! carry table error. The positivity proof above is for the EXACT kernel. A
//! `Q` or `M` entry computed slightly LOW by interpolation would break the
//! bound by that much, exactly as a precision-cliff zero would — see the
//! "Precision discipline" section below, which is the same hazard in a
//! different guise. A terfc CSB would therefore need its table error bounded
//! and folded in as a relative inflation of `Q`/`M`, not merely a floor.
//! Flagged as a prerequisite, not solved.
//!
//! **Conditions I checked for and did NOT find** (recorded because their
//! absence is load-bearing): the derivation places no restriction on angular
//! momentum, contraction depth, normalization convention, or basis-function
//! centres. Appendix A's `E₂` is "the real vector space of all continuous
//! real-valued functions of the positions of two electrons", and real
//! contracted Cartesian/solid-harmonic Gaussians are continuous real-valued
//! functions, hence members. Appendix A's own closing sentence — "Equations
//! (6) and (7) are a direct consequence of (A3) when e and f are replaced by
//! the corresponding products of Gaussian basis functions (and Mulliken
//! integral notation is used)" — is the authors making exactly that
//! substitution. The one condition that IS required, `G` positive definite,
//! Table VI discharges for every operator this module accepts.
//!
//! One scope caveat the paper states in its own voice, which is about VALUE,
//! not validity:
//!
//! > For the long-range Coulomb operator, overlap dependence is much more
//! > important than distance dependence and we find that the CSB estimate is
//! > no more useful than the QQ estimate for currently tractable systems.
//! > However, for operators with much stronger distance-decay, such as e^(-r12)
//! > and erfc(0.11·r12)/r12, we find that that the rigorous CSB provides a
//! > much tighter bound [...]
//!
//! and Table I of the paper backs that with numbers: under `1/r12` its
//! reported "F_min" for CSB is exactly `1.000` (i.e. CSB ≡ QQ on that sample),
//! while it is the attenuated/exponential kernels where CSB separates. **CSB
//! is still perfectly SOUND under Coulomb — it is a `min` of upper bounds —
//! it is just expected to buy little there.** ferric's headline profile
//! (benzene/aug-cc-pVDZ RHF, `scatter_bra_pair` at 28.96%) is a COULOMB
//! workload, so the paper predicts a small win at best on that specific
//! benchmark. That prediction is recorded here up front, BEFORE measuring, so
//! a null result on Coulomb reads as confirmation of the literature rather
//! than as evidence of a broken port. Nothing in this module has been run.
//!
//! # Precision discipline — the 218,493× failure mode, and why it cannot land here
//!
//! The `M` table faces the IDENTICAL hazard [`crate::schwarz`]'s
//! `SCHWARZ_TABLE_PRECISION` comment documents at length, and it is worth
//! being explicit about the mechanism because the CSAM branch is currently
//! sitting on a live instance of it.
//!
//! libint2 prescreens internally at the precision its engine was constructed
//! with and declines to compute a quartet whose result falls below that.
//! `Engine::compute_quartet` then returns `None`. If this module recorded that
//! as `M = 0.0`, then for every quartet touching that shell pair the CSB
//! `min` would return `0.0` — an unconditional, silent underestimate of an
//! integral that may be perfectly significant. Under an attenuated kernel this
//! is not hypothetical but the DOMINANT regime: `erfc(ω r)/r` drives
//! `(µµ|λλ)` exponentially toward zero for distant `µ, λ`, so a large fraction
//! of `M` entries sit near the engine's precision cliff — which is exactly the
//! shape of the 218,493× benzene/cc-pVDZ/`erfc(1.0)` violation the CSAM branch
//! measured.
//!
//! Three defences, applied from the start rather than retrofitted:
//!
//! 1. the table is built at [`SCHWARZ_TABLE_PRECISION`] (`0.0`, libint2's
//!    internal prescreening DISABLED) — the same constant, imported rather
//!    than copied, so the two can never drift;
//! 2. every stored entry is floored at [`SCHWARZ_Q_FLOOR`] (`1e-100`, again
//!    imported), so no entry is ever exactly zero even if the true `(µµ|λλ)`
//!    underflows;
//! 3. the `None` arm from the engine returns the floor rather than `0.0`, so a
//!    bound that could not be evaluated does not silently underestimate.
//!
//! and the invariant is asserted directly at its source by
//! `csb_m_table_never_stores_a_zero_alkane_8` below, on the same alkane_8
//! system whose Schwarz table exhibited the original defect (1049 spurious
//! zeros of 5253 pairs).
//!
//! Note the asymmetry with the Schwarz table that makes this MORE important
//! here, not less: a floored `Q` still enters as a PRODUCT, so `Q_µν · Q_λσ`
//! is small but the quartet is merely retained. A floored `M` enters a `min`,
//! where the smallest argument WINS — so a spurious zero does not soften the
//! bound, it annihilates it.
//!
//! # Cost and shape
//!
//! `nshells × nshells` `f64`, identical in shape and memory to the Schwarz `q`
//! table, geometry-only, built once per bound construction (NOT per SCF
//! iteration). It needs one `(PP|QQ)` shell quartet per unordered shell pair —
//! the same `O(nshells²)` quartet count as the Schwarz `(ij|ij)` pass, so the
//! setup cost should be the same order as `schwarz()`'s. UNMEASURED: no
//! benchmark was run for this module (the task forbade compiling). If the
//! build ever shows up in a profile, mirror `schwarz.rs`'s ROW_BLOCK rayon
//! split rather than parallelizing per pair — that file's comments record a
//! measured 26× SLOWDOWN from per-pair parallelization of a libint2 loop.

use crate::basis_bridge::PreparedBasis;
use crate::engine::Engine;
use crate::operator::{Operator, OperatorKind};
use crate::schwarz::{SCHWARZ_Q_FLOOR, SCHWARZ_TABLE_PRECISION};
use ferric_core::FerricError;
use ndarray::Array2;

/// Build the CSB `M` table: `M[P][Q] = sqrt( max_{µ∈P, λ∈Q} |(µµ|λλ)| )`,
/// symmetric, floored at [`SCHWARZ_Q_FLOOR`], built at
/// [`SCHWARZ_TABLE_PRECISION`].
///
/// # Why a shell-block MAX is the right lift to shell granularity
///
/// Eq. (6) is a statement about individual basis FUNCTIONS: `|(µν|λσ)| ≤
/// M_µλ M_νσ`. ferric screens whole shell QUARTETS, so the quantity needed is
///
/// ```text
///   max_{µ∈S1, ν∈S2, λ∈S3, σ∈S4} |(µν|λσ)|
///     ≤ max_{µ∈S1, ν∈S2, λ∈S3, σ∈S4} ( M_µλ · M_νσ )
///     ≤ ( max_{µ∈S1, λ∈S3} M_µλ ) · ( max_{ν∈S2, σ∈S4} M_νσ )
///     = M[S1][S3] · M[S2][S4]
/// ```
///
/// The second step is where the shell-block max earns its place: the two
/// factors range over DISJOINT index pairs — `(µ,λ)` and `(ν,σ)` — so
/// maximizing each independently is an over-estimate of maximizing the
/// product, never an under-estimate. (Had the two factors shared an index, a
/// product of independent maxima would still be an upper bound, but the same
/// reasoning would need restating; they do not, so it is clean.) This is the
/// identical argument by which `schwarz_pair`'s `max_{a,b} |(ab|ab)|` lifts
/// Eq. (1) to shell granularity, and it is the whole reason this table can be
/// stored per SHELL PAIR rather than per function pair.
///
/// Note this is genuinely a maximum over the `(µµ|λλ)` SUB-BLOCK of the
/// `(PP|QQ)` quartet — the entries with the bra indices equal to each other
/// and the ket indices equal to each other — not over the whole block. The
/// off-diagonal entries of `(PP|QQ)` are `(µν|λσ)` values with `µ,ν ∈ P` and
/// `λ,σ ∈ Q`; including them would produce a DIFFERENT, unproven quantity.
/// The indexing below is spelled out for that reason.
///
/// # Per-operator
///
/// The table MUST be built with the same [`Operator`] it will screen. This is
/// enforced structurally at the `ferric-scf` layer (the `CsbBounds` type owns
/// both tables and one `Operator`), and here by taking `op` as a parameter
/// rather than defaulting to Coulomb. ferric has been burned by the opposite
/// convention before: `qqr.rs` carried an extra `exp(-ω²R²)` factor that
/// double-counted attenuation already living in the erfc Schwarz factors, and
/// made the bound invalid at EVERY sampled quartet (worst |true|/bound 6.0e11
/// at benzene/cc-pVDZ). There is no such separate geometric factor here: the
/// attenuation enters through the `(µµ|λλ)` integrals themselves, exactly as
/// it enters `Q` through `(µν|µν)`.
///
/// # Accepted operators
///
/// The same set [`crate::schwarz::schwarz`] accepts — `Coulomb`,
/// `ErfCoulomb`, `ErfcCoulomb` — and for a stronger reason than mere
/// symmetry: Table VI of Thompson & Ochsenfeld proves the required
/// positive-definiteness for all three (and for Yukawa, Slater-geminal and the
/// Gaussian geminal besides). Unlike `csam_x_table` on the sibling branch,
/// which REFUSES the attenuated kernels because its non-rigorous estimate
/// diverges there, CSB has no reason to refuse them — attenuation is the
/// regime the paper says CSB is actually FOR. The restriction here is purely
/// to mirror `schwarz()`'s support surface, since a CSB bound is useless
/// without the Schwarz table it takes a `min` against.
pub fn csb_m_table(op: Operator, prep: &PreparedBasis) -> Result<Array2<f64>, FerricError> {
    match op.kind {
        OperatorKind::Coulomb | OperatorKind::ErfCoulomb | OperatorKind::ErfcCoulomb => {}
        _ => {
            return Err(FerricError::Libint(format!(
                "operator {:?} not implemented for CSB screening (mirrors schwarz())",
                op.kind
            )))
        }
    }

    let nsh = prep.nshells();
    let dims = prep.shell_dims();
    let mut eng = Engine::new_2e(op, prep, SCHWARZ_TABLE_PRECISION)?;

    let mut m = Array2::<f64>::zeros((nsh, nsh));
    for p in 0..nsh {
        let np = dims[p];
        for q in 0..=p {
            let nq = dims[q];
            // The (PP|QQ) block is laid out with shell dims (np, np, nq, nq).
            // The entries we want are (µµ|λλ) for µ ∈ P, λ ∈ Q — the
            // generalized diagonal in BOTH the bra and the ket pair — at flat
            // index ((a*np + a)*nq + b)*nq + b. See the doc above for why the
            // rest of the block must NOT be included.
            let maxv = match eng.compute_quartet(prep, p, p, q, q) {
                Some(block) => {
                    let mut acc = 0.0f64;
                    for a in 0..np {
                        for b in 0..nq {
                            let v = block[((a * np + a) * nq + b) * nq + b].abs();
                            if v > acc {
                                acc = v;
                            }
                        }
                    }
                    acc
                }
                // Unreachable at SCHWARZ_TABLE_PRECISION = 0.0 (prescreening
                // disabled), retained so an engine that declines a block still
                // yields the floor rather than a bound-destroying 0.0.
                None => 0.0,
            };
            let mv = maxv.sqrt().max(SCHWARZ_Q_FLOOR);
            m[(p, q)] = mv;
            m[(q, p)] = mv;
        }
    }
    Ok(m)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferric_core::basis;
    use ferric_core::mol::Molecule;

    fn prep_for(path: &str, basis: &str) -> PreparedBasis {
        let mol = Molecule::load_xyz(path).unwrap();
        let bs = basis::bundled(basis).unwrap();
        PreparedBasis::new(&mol, &bs).unwrap()
    }

    /// The `M` table must be symmetric, finite and non-negative. `M[P][Q]` and
    /// `M[Q][P]` are the same quantity — `(µµ|λλ)` is symmetric under
    /// bra↔ket — and are written from one computed block, so this is really a
    /// guard on the indexing arithmetic.
    #[test]
    fn csb_m_table_is_symmetric_and_finite() {
        let prep = prep_for("../../testdata/molecules/water.xyz", "cc-pvdz");
        for op in [Operator::coulomb(), Operator::erfc(1.0), Operator::erf(1.0)] {
            let m = csb_m_table(op, &prep).unwrap();
            let nsh = prep.nshells();
            for i in 0..nsh {
                for j in 0..nsh {
                    assert!(
                        (m[(i, j)] - m[(j, i)]).abs() <= 1e-12 * m[(i, j)].max(1.0),
                        "op={op:?}: M not symmetric at ({i},{j}): {} vs {}",
                        m[(i, j)],
                        m[(j, i)]
                    );
                    assert!(
                        m[(i, j)].is_finite() && m[(i, j)] >= SCHWARZ_Q_FLOOR,
                        "op={op:?}: M[{i},{j}] = {} is not a usable bound factor",
                        m[(i, j)]
                    );
                }
            }
        }
    }

    /// INVARIANT, and the direct defence against the 218,493× failure mode:
    /// the `M` table never stores an exact zero.
    ///
    /// A stored `M = 0.0` would win the CSB `min` at every quartet touching
    /// that shell pair and drive the bound to zero regardless of the true
    /// integral — a silent, unconditional underestimate. This is the same
    /// invariant `schwarz_table_never_stores_a_zero_alkane_8` asserts for the
    /// `Q` table, on the same system, chosen for the same reason: alkane_8 at
    /// cc-pVDZ is large enough that distant shell pairs fall under the
    /// engine's precision cliff, where water and CH4 show nothing at all (the
    /// pre-fix Schwarz table stored 1049 spurious zeros of 5253 pairs here).
    ///
    /// `erfc(1.0)` is included deliberately and is the discriminating arm: it
    /// drives `(µµ|λλ)` exponentially toward zero for distant `µ, λ`, so a
    /// large fraction of this table sits near the cliff. If
    /// `SCHWARZ_TABLE_PRECISION`/`SCHWARZ_Q_FLOOR` were ever dropped from
    /// `csb_m_table`, this arm is where it would show.
    #[test]
    fn csb_m_table_never_stores_a_zero_alkane_8() {
        let prep = prep_for("../../testdata/molecules/alkane_8.xyz", "cc-pvdz");
        let nsh = prep.nshells();
        assert_eq!(
            nsh, 102,
            "alkane_8/cc-pVDZ should give 102 shells; the counts quoted in this test's doc \
             were measured on that decomposition"
        );
        for op in [Operator::coulomb(), Operator::erfc(1.0), Operator::erfc(0.222)] {
            let m = csb_m_table(op, &prep).unwrap();
            let zeros = (0..nsh)
                .flat_map(|i| (0..=i).map(move |j| (i, j)))
                .filter(|&(i, j)| m[(i, j)] == 0.0)
                .count();
            assert_eq!(
                zeros, 0,
                "op={op:?}: {zeros} of {} unique shell pairs store M == 0.0. Such an entry WINS \
                 the CSB min and drives the bound to zero for every quartet touching that pair — \
                 an unconditional underestimate. Check that csb_m_table still builds at \
                 SCHWARZ_TABLE_PRECISION and floors at SCHWARZ_Q_FLOOR.",
                nsh * (nsh + 1) / 2
            );
        }
    }

    /// Eq. (6) is an EQUALITY for `(µµ|λλ)`-type integrals ("Equality holds in
    /// (6) and (7) for integrals of type (µµ|νν)"). So `M[P][Q]²` must
    /// reproduce `max_{µ∈P,λ∈Q} |(µµ|λλ)|` computed independently — this is
    /// the anchor that the indexing arithmetic picks the right sub-block of
    /// the `(PP|QQ)` quartet rather than, say, the whole block max (which
    /// would be larger and a different, unproven quantity) or a wrong stride
    /// (which could be either).
    ///
    /// Recomputed here at a DIFFERENT, explicitly-tight engine precision so
    /// the test cannot pass by both paths sharing the same defect.
    #[test]
    fn csb_m_squared_reproduces_the_mu_mu_lambda_lambda_maximum() {
        let prep = prep_for("../../testdata/molecules/water.xyz", "cc-pvdz");
        let dims = prep.shell_dims();
        let nsh = prep.nshells();
        for op in [Operator::coulomb(), Operator::erfc(1.0)] {
            let m = csb_m_table(op, &prep).unwrap();
            let mut eng = Engine::new_2e(op, &prep, 1e-30).unwrap();
            for p in 0..nsh {
                for q in 0..=p {
                    let (np, nq) = (dims[p], dims[q]);
                    let block = eng.compute_quartet(&prep, p, p, q, q).unwrap();
                    let mut want = 0.0f64;
                    for a in 0..np {
                        for b in 0..nq {
                            want = want.max(block[((a * np + a) * nq + b) * nq + b].abs());
                        }
                    }
                    let got = m[(p, q)] * m[(p, q)];
                    assert!(
                        (got - want).abs() <= 1e-10 * want.max(1e-12),
                        "op={op:?}: M[{p},{q}]^2 = {got:.6e} != max |(uu|ll)| = {want:.6e}"
                    );
                }
            }
        }
    }

    /// Operators outside `schwarz()`'s support surface are a typed error, not
    /// a silent Coulomb table. Mirrors `schwarz()`'s own restriction.
    #[test]
    fn csb_m_table_rejects_unsupported_operators() {
        let prep = prep_for("../../testdata/molecules/water.xyz", "sto-3g");
        assert!(csb_m_table(Operator::yukawa(1.0), &prep).is_err());
        // Control: the three operators `schwarz()` supports must all SUCCEED,
        // or the rejection test above would pass vacuously via a blanket
        // failure. Attenuated kernels are the regime CSB exists for; refusing
        // them (as the non-rigorous CSAM sibling must) would be a defect here.
        for op in [Operator::coulomb(), Operator::erf(1.0), Operator::erfc(1.0)] {
            assert!(
                csb_m_table(op, &prep).is_ok(),
                "csb_m_table must accept {op:?} — CSB is PROVEN for it by Table VI of \
                 Thompson & Ochsenfeld, and attenuated kernels are where CSB is most useful"
            );
        }
    }
}
