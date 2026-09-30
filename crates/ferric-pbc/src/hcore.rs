//! Gamma-point periodic one-electron matrices and nuclear repulsion
//! (Stage 1 step 4, `reference/pbc/stage1-design.md` §3).
//!
//! ```text
//! S[μ,ν] = Σ_L ⟨μ_0|ν_L⟩            T[μ,ν] = Σ_L ⟨μ_0|−½∇²|ν_L⟩
//! V      = V_SR(ω) + V_LR(ω) + V_G0(ω)          (ω-independent sum)
//! V_SR   = −Σ_L Σ_{C,M} Z_C ⟨μ_0| erfc(ω|r−R_C−M|)/|r−R_C−M| |ν_L⟩
//! V_LR   = −(1/Ω) Σ_{G≠0} (4π/G²) e^{−G²/4ω²} Re[ P*_μν(G) S(G) ],  S(G) = Σ_C Z_C e^{−iG·R_C}
//! V_G0   = + π Z_tot S / (ω² Ω)
//! h      = T + V,     E_nn = Ewald sum (neutralising-background convention)
//! ```
//!
//! with `ν_L(r) = ν(r − L)` and `P` the lattice-summed pair FT
//! ([`crate::pair_ft()`]). This is exactly `build_integrals` of
//! `reference/pbc/pbc_gamma.py`:
//!
//! * **G = 0 bookkeeping.** The real-space erfc sum implicitly contains the
//!   average potential `−Z_tot π/(ω²Ω)` (∫ erfc(ωr)/r d³r = π/ω²); `V_G0`
//!   removes it, so every term follows the standard "drop G = 0" convention
//!   (cancels against the same term of J for a neutral cell, and matches the
//!   pure reciprocal-space [`pure_aft_nuclear`] exactly).
//! * **Nuclei as Gaussians.** libint2 2.7.2's `erf_nuclear`/`erfc_nuclear`
//!   are wrong (reference/pbc/FINDINGS.md, "Iteration 2"), so the SR
//!   attraction uses the design's option (b): every nucleus is a unit-
//!   normalised s Gaussian of exponent ζ ([`GAUSSIAN_NUCLEUS_EXPONENT`],
//!   PySCF's `fakemol_for_charges` value) in a [`SiteBasis`], and
//!   `(g_C | μ ν)_erfc / ∫g_C` comes from the ordinary 3-centre engine with
//!   `Operator::erfc(ω)`. Lattice images need no image basis: the new
//!   `Engine::compute_eri3_shifted` translates the nucleus by `M` and `ν` by
//!   `L`.
//!
//! # Finite-nucleus error
//!
//! A Gaussian charge of exponent ζ has potential `erf(√ζ r)/r`, differing from
//! `1/r` only within ~`1/√ζ` of the nucleus. For a pair density `ρ` the
//! attraction error is `Z ∫ ρ (1 − erf(√ζ r))/r ≈ Z ρ(R_C) · 4π ∫ r erfc(√ζ r) dr
//! = π Z ρ(R_C)/ζ`. For an O 1s AO (`ρ(0) ≈ Z³/π ≈ 163`) that is `4e3/ζ`:
//! 4e-13 Ha at ζ = 1e16, 4e-9 at 1e12. The erfc part of the SR kernel only
//! sees `1/ω'² = 1/ω² + 1/ζ`, a relative `ω²/(2ζ)` shift — nothing at 1e16.
//! If libint2 ever mishandles ζ = 1e16 (the `tests/pbc_hcore.rs` ζ-sweep is the
//! measurement), 1e14 keeps the error below 1e-10 for first-row atoms.
//!
//! # Truncation (all derived from `precision`)
//!
//! * Pair images: `r_pair = √(2 ln(10³/t)/α_min) + 2`, `t = precision/10`
//!   (pair_ft's rule). `S`/`T` sum every image in that set.
//! * SR nucleus images, per (μ-shell, ν-shell, L): from the pair's s-type
//!   charge bound `q = max |c_a c_b| (π/p)^{3/2} e^{−ab R²/p}`
//!   (primitive-normalised coefficients), the pair is skipped when even a
//!   nucleus on top of it contributes < `precision`; otherwise nucleus images are kept
//!   within `r = √ln(q Z_max (1 + 2√(p_max/π))/precision)/ω_p + 2` Bohr of the
//!   segment `[A, B+L]` (which contains every product centre), where
//!   `ω_p = ω √(p_min/(p_min + ω²))` is the erfc range after convolution with
//!   the most diffuse product Gaussian (`erfc(x) < e^{−x²}`).
//! * LR: G on the half sphere (P(−G) = P(G)* for real AOs; weight 2) with
//!   `|G| ≤ min(2ω, 2√p_max) √ln(1/precision)`.
//!
//! # Default ω ([`default_hcore_omega`])
//!
//! ```text
//! ω_h = min( 2.5 · √π / Ω^{1/3} ,  ω_cap(precision) )
//! ω_cap(precision) = ω_gdf √( ln(1/p_gdf) / ln(1/precision) )     (0.9636 at 1e-14)
//! ```
//!
//! with `ω_gdf = 1`, `p_gdf = 1e-13` the RS-GDF defaults
//! ([`crate::rsgdf::DEFAULT_RSGDF_OMEGA`], [`crate::rsgdf::DEFAULT_RSGDF_PRECISION`]).
//! A run that sets its own RS-GDF ω (CLI `[cell] gdf_omega`, Python
//! `gdf_omega=`) caps with that ω instead ([`default_hcore_omega_for_gdf`]);
//! the cap is linear in `ω_gdf`.
//! A function of the cell volume and `precision` only: never of the thread
//! count, the budget or the basis. Any ω gives the same `V` up to truncation;
//! the rule only moves cost.
//!
//! * **Why a multiple of the Ewald ω.** Per pair image, the SR nucleus
//!   triplets go as `(N_at/Ω) r(ω)³` with `r ∝ √ln(1/precision)/ω` and the LR
//!   G count as `Ω (ω √ln(1/precision))³`. `ω = c √π/Ω^{1/3}` fixes the LR
//!   half-sphere at `(2/3) c³ ln^{3/2}(1/precision)/√π` vectors whatever the
//!   cell (≈ 1076 at `c = 2.5`, 1e-14), and both sides scale as
//!   `ln^{3/2}(1/precision)`, so the optimal `c` does not move with precision.
//! * **Why 2.5.** MODELLED 1-thread CPU of hcore SR + LR
//!   (`reference/pbc/bench/perf_hcore_omega_cost.py`: the capsule estimator
//!   of `perf_hcore_omega.py`, which reproduces ferric's SR triplet counter
//!   to 0.2% on diamond_prim, times per-triplet / per-segment-test /
//!   per-pair-FT-evaluation costs fitted on the measured diamond_prim runs at
//!   ω = 0.417 and 0.96; unfitted check: dry ice at its old default, SR 157 s
//!   modelled vs 168 s measured, LR 1.3 vs 1.85 s). cc-pVDZ:
//!
//!   | cell | old ω (√π/Ω^{1/3}) | SR+LR old | new ω | SR+LR new | best grid ω (cost) |
//!   |---|---|---|---|---|---|
//!   | diamond_prim (2 C) | 0.417 | 56.3 s | 0.964 (cap) | 30.7 s | 0.94 (30.5 s) |
//!   | diamond_conv (8 C) | 0.263 | 483 s | 0.657 | 169.0 s | 0.70 (168.6 s) |
//!   | dry ice (12 atoms) | 0.167 | 159 s | 0.417 | 44.3 s | 0.375 (43.7 s) |
//!
//!   `c = 2.5` is within 1.4% of each cell's best grid point (0.5%, 0.3%,
//!   1.4%); `c = 2.25` costs +3% on diamond_conv and `c = 2.75` +9% on dry
//!   ice. MEASURED on diamond_prim at 0.96 (reference/pbc/FINDINGS.md,
//!   "Benchmark series", 2026-09-28): hcore SR 56.1 → 26.5 s, LR 0.8 →
//!   4.7 s, total run 106.5 → 80.6 s, energy moved 1.5e-11. The diamond_conv
//!   and dry-ice rows are not measured.
//! * **Why the cap.** A ω at which the hcore LR sphere `2ω√ln(1/precision)`
//!   equals the default RS-GDF sphere `2 ω_gdf √ln(1/p_gdf)` (10.942 Bohr⁻¹)
//!   keeps every hcore G vector inside the sphere RS-GDF already walks (the
//!   precondition for fusing the two, performance plan item 9), and stops
//!   small cells (uncapped `c√π/Ω^{1/3}` > 1) from paying LR for SR savings
//!   that saturate once `ω² ≫ p_min` (`ω_p → √p_min`). diamond_prim is capped.
//! * **What it is NOT.** A constant 0.96 (the plan's "G sphere = RS-GDF's"
//!   alone) is right for diamond_prim but loses on larger cells while V_LR is
//!   a separate pair-FT pass: the modelled dry-ice LR at 0.96 is 206 s (13372
//!   half-G) against 11 s of SR, total 217 s vs 159 s today.
//! * **k-points.** [`kpoint`] uses the same rule. Its LR runs over the FULL
//!   sphere (2× the Gamma half-sphere), which moves the Gamma-calibrated
//!   optimum `c` down by ~2^{1/6} (≈ 11%, inside the flat region); the
//!   k-point SR/LR split has not been measured.
//! * **Not the nuclear-repulsion Ewald.** `E_nn`, its gradient/stress and the
//!   Madelung constant keep [`default_ewald_omega`] (point charges; a
//!   different cost balance).
//!
//! # Memory (Stage 1 step 10)
//!
//! Every buffer that grows with the lattice or the G sphere is reserved on a
//! [`crate::budget`] ledger BEFORE it is allocated, against
//! [`PeriodicHcoreConfig::budget_bytes`] (default: ferric's unified budget):
//! the `n×n` matrices, the pair-image list, the SR nucleus candidates, the G
//! list, and the `pair_ft` chunks of `V_LR` (G-chunked through
//! [`crate::pair_ft::pair_ft_chunked`], never all G at once). Chunking is
//! bit-for-bit invariant (`pair_ft_chunked` doc; the `V_LR` accumulation is
//! in G order regardless of the chunk size).
//!
//! # SR screening study (Stage 1 step 8)
//!
//! [`sr_screening_study`] re-runs the SR nucleus sum with the screen
//! threshold decoupled from the candidate set, and reports what the screen
//! skipped against the bound's own prediction. It shares the production loop
//! (`sr_attraction`), so it measures the code `periodic_hcore` runs.
//!
//! # Orbital-pair symmetry (Gamma s2)
//!
//! `(μ_0|O|ν_L) = (ν_0|O|μ_{−L})` for the translation-invariant `O` = S, T
//! and the SR attraction (nuclei at every lattice image; the candidate and
//! pair-image sets are closed under the map), so at Gamma the S, T and `V_SR`
//! lattice sums are symmetric. [`periodic_hcore`] therefore evaluates each
//! UNORDERED shell pair `i1 ≤ i2` once over every pair image `L`, with the
//! ordered loop's per-pair body and per-element addend sequence, and ONE task
//! copies the finished block into both `(μ, ν)` and `(ν, μ)`. A diagonal
//! pair also runs over every `L` and writes its `i ≤ j` elements into both
//! places (an `L = 0` + half-space scheme would save only `~1/(2 nsh)` of the
//! work and needs an `L ↔ −L` image map). The matrices are EXACTLY symmetric,
//! every element is bitwise the ordered loop's `(μ ≤ ν)` value, and each
//! element is still written by exactly one task (bitwise across thread
//! counts). The trailing `symmetrize` is then the identity (`½(x + x) = x`
//! exactly) and `sr_asymmetry` is 0. `n_sr_triplets` counts the triplets
//! COMPUTED; `n_sr_triplets_ordered` (off-diagonal × 2) is the pre-s2
//! ordered count to compare benchmarks on. [`periodic_hcore_pair_s1_oracle`]
//! is the pre-s2 build (the FROZEN serial ordered loops + averaging), bit for
//! bit. The k-point sums (`kpoint`) keep the ordered loops (`h(k)` is
//! Hermitian, not symmetric: s2 there needs the `e^{ik·L}` ↔ `e^{−ik·L}`
//! pairing), and so do the force/strain derivative walks (their own loops).

use crate::budget::{bytes_of, Ledger};
use crate::ewald::{default_ewald_omega, ewald_nuclear_repulsion};
use crate::lattice::Cell;
use crate::ordered::{ordered_units, window_budget, Stored};
use crate::pair_ft::{pair_ft_bytes_per_g, pair_ft_chunked};
use crate::rsgdf::unordered_pairs;
use crate::timing::{PbcTimings, StageClock};
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::engine::Engine;
use ferric_integrals::engine_pool::EnginePool;
use ferric_integrals::ffi;
use ferric_integrals::md3c1e::prim_norm;
use ferric_integrals::operator::Operator;
use ferric_integrals::site_basis::SiteBasis;
use ndarray::{Array2, Array3};
use num_complex::Complex64;
use rayon::prelude::*;
use std::f64::consts::PI;
use std::sync::Mutex;

/// Stage 3: the same lattice sums, phase-weighted per k-point.
pub mod kpoint;

/// Exponent (Bohr⁻²) of the unit-normalised s Gaussian standing in for a
/// point nucleus (PySCF `gto.fakemol_for_charges` uses the same 1e16). See
/// the module doc for the finite-nucleus error estimate.
pub const GAUSSIAN_NUCLEUS_EXPONENT: f64 = 1e16;

/// Default truncation target for [`periodic_hcore`] (Hartree, per neglected
/// term).
pub const DEFAULT_HCORE_PRECISION: f64 = 1e-14;

/// Precision handed to libint2 for the Gaussian-nucleus 3-centre engine.
/// libint2's primitive screening compares against `ln(precision)` using pair
/// prefactors that the 1e16-exponent, 1e11-coefficient nucleus shell skews,
/// so screening is effectively disabled (`ln` ≈ −708); our own distance
/// screen (module doc) decides what is computed.
pub(crate) const ERI3_ENGINE_PRECISION: f64 = f64::MIN_POSITIVE;
/// Precision for the overlap/kinetic engines (value used by the passing
/// `pbc_shifted_overlap` anchor, tightened).
pub(crate) const ONE_E_ENGINE_PRECISION: f64 = 1e-16;
/// Extra Bohr on every derived real-space radius (polynomial prefactors of
/// l > 0 pairs are not in the s-type bounds).
const SR_MARGIN_BOHR: f64 = 2.0;
/// Upper bound on one `pair_ft` chunk (output plus scratch), whatever the
/// budget: large enough for BLAS-friendly chunks, small enough to stay out of
/// the way of the resident matrices.
pub(crate) const G_CHUNK_BYTES: usize = 64 << 20;

/// Multiple `c` of the balanced nuclear-repulsion Ewald split
/// ([`default_ewald_omega`], `√π/Ω^{1/3}`) that [`default_hcore_omega`]
/// uses before the cap (module doc, "Default ω").
pub const HCORE_OMEGA_EWALD_FACTOR: f64 = 2.5;

/// Upper bound of the default hcore split at `precision`: the ω whose LR
/// sphere `2ω√ln(1/precision)` equals the DEFAULT RS-GDF LR sphere
/// `2 ω_gdf √ln(1/p_gdf)` ([`crate::rsgdf::DEFAULT_RSGDF_OMEGA`] = 1,
/// [`crate::rsgdf::DEFAULT_RSGDF_PRECISION`] = 1e-13; 0.9636 Bohr⁻¹ at
/// [`DEFAULT_HCORE_PRECISION`]). A run that sets its own RS-GDF ω uses
/// [`hcore_omega_cap_for_gdf`] instead.
pub fn hcore_omega_cap(precision: f64) -> f64 {
    hcore_omega_cap_for_gdf(
        precision,
        crate::rsgdf::DEFAULT_RSGDF_OMEGA,
        crate::rsgdf::DEFAULT_RSGDF_PRECISION,
    )
}

/// [`hcore_omega_cap`] for an RS-GDF build at `(gdf_omega, gdf_precision)`:
/// the ω whose hcore LR sphere `2ω√ln(1/precision)` equals that build's LR
/// sphere `2 gdf_omega √ln(1/gdf_precision)`. Linear in `gdf_omega`;
/// `hcore_omega_cap_for_gdf(p, DEFAULT_RSGDF_OMEGA, DEFAULT_RSGDF_PRECISION)`
/// is [`hcore_omega_cap`]`(p)` bit for bit.
pub fn hcore_omega_cap_for_gdf(precision: f64, gdf_omega: f64, gdf_precision: f64) -> f64 {
    let gdf = (1.0 / gdf_precision).ln();
    gdf_omega * (gdf / (1.0 / precision).ln()).sqrt()
}

/// The default nuclear-attraction split (Bohr⁻¹) for `cell` at hcore
/// truncation `precision`:
/// `min(HCORE_OMEGA_EWALD_FACTOR · √π/Ω^{1/3}, hcore_omega_cap(precision))`.
/// Deterministic in the cell volume and `precision` (no thread count, budget
/// or basis). Derivation and the measured/modelled costs: module doc,
/// "Default ω". An explicit ω ([`PeriodicHcoreConfig::with_omega`]) is never
/// replaced by this.
pub fn default_hcore_omega(cell: &Cell, precision: f64) -> f64 {
    (HCORE_OMEGA_EWALD_FACTOR * default_ewald_omega(cell)).min(hcore_omega_cap(precision))
}

/// [`default_hcore_omega`] with the cap taken from an RS-GDF build at
/// `gdf_omega` (and [`crate::rsgdf::DEFAULT_RSGDF_PRECISION`]):
/// `min(HCORE_OMEGA_EWALD_FACTOR · √π/Ω^{1/3},
/// hcore_omega_cap_for_gdf(precision, gdf_omega, DEFAULT_RSGDF_PRECISION))`,
/// so the hcore G sphere stays inside the sphere that build walks. At
/// `gdf_omega = DEFAULT_RSGDF_OMEGA` it is [`default_hcore_omega`] bit for bit.
pub fn default_hcore_omega_for_gdf(cell: &Cell, precision: f64, gdf_omega: f64) -> f64 {
    let cap = hcore_omega_cap_for_gdf(precision, gdf_omega, crate::rsgdf::DEFAULT_RSGDF_PRECISION);
    (HCORE_OMEGA_EWALD_FACTOR * default_ewald_omega(cell)).min(cap)
}

/// Settings for [`periodic_hcore`].
#[derive(Debug, Clone, Copy)]
pub struct PeriodicHcoreConfig {
    /// Ewald splitting parameter for the nuclear attraction (Bohr⁻¹). Any
    /// `ω > 0` gives the same `V` up to truncation; larger ω moves work
    /// from the real-space SR sum to reciprocal space. Default
    /// ([`PeriodicHcoreConfig::for_cell`]): [`default_hcore_omega`].
    pub omega: f64,
    /// Truncation target, in `(0, 1)`.
    pub precision: f64,
    /// Gaussian-nucleus exponent (Bohr⁻²).
    pub nucleus_exponent: f64,
    /// Memory budget (bytes) every large buffer is gated against before
    /// allocation. `None` = ferric's unified budget
    /// ([`crate::budget::resolve`]: `FERRIC_MEM_BUDGET_GB`, else 0.8 ×
    /// available RAM, else 2 GiB; `Some(0)` counts as unset, the ferric
    /// convention).
    pub budget_bytes: Option<usize>,
}

impl PeriodicHcoreConfig {
    /// Defaults ([`DEFAULT_HCORE_PRECISION`], [`GAUSSIAN_NUCLEUS_EXPONENT`])
    /// at the given ω.
    pub fn with_omega(omega: f64) -> Self {
        Self {
            omega,
            precision: DEFAULT_HCORE_PRECISION,
            nucleus_exponent: GAUSSIAN_NUCLEUS_EXPONENT,
            budget_bytes: None,
        }
    }

    /// Defaults with the default split of `cell`:
    /// [`default_hcore_omega`]`(cell, DEFAULT_HCORE_PRECISION)`.
    pub fn for_cell(cell: &Cell) -> Self {
        Self::with_omega(default_hcore_omega(cell, DEFAULT_HCORE_PRECISION))
    }

    /// [`PeriodicHcoreConfig::for_cell`] for a run whose RS-GDF build uses
    /// `gdf_omega` (Bohr⁻¹): the default split capped by that build's LR
    /// sphere ([`default_hcore_omega_for_gdf`]). `None` (the RS-GDF default,
    /// or no RS-GDF) is `for_cell`; so is `Some(DEFAULT_RSGDF_OMEGA)`, bit
    /// for bit.
    pub fn for_cell_and_gdf_omega(cell: &Cell, gdf_omega: Option<f64>) -> Self {
        match gdf_omega {
            None => Self::for_cell(cell),
            Some(w) => Self::with_omega(default_hcore_omega_for_gdf(
                cell,
                DEFAULT_HCORE_PRECISION,
                w,
            )),
        }
    }

    /// The periodic-ECP settings `periodic_hcore` uses: the same precision and
    /// budget, no caps, no mutation.
    pub(crate) fn ecp_config(&self) -> crate::ecp::PeriodicEcpConfig {
        crate::ecp::PeriodicEcpConfig {
            budget_bytes: self.budget_bytes,
            ..crate::ecp::PeriodicEcpConfig::with_precision(self.precision)
        }
    }

    fn validate(&self) -> Result<(), FerricError> {
        if !(self.omega > 0.0) || !self.omega.is_finite() {
            return Err(FerricError::General(format!(
                "periodic_hcore: omega must be finite and > 0, got {}",
                self.omega
            )));
        }
        if !(f64::MIN_POSITIVE..1.0).contains(&self.precision) {
            return Err(FerricError::General(format!(
                "periodic_hcore: precision must lie in (0, 1), got {}",
                self.precision
            )));
        }
        if !(self.nucleus_exponent > 0.0) || !self.nucleus_exponent.is_finite() {
            return Err(FerricError::General(format!(
                "periodic_hcore: nucleus_exponent must be finite and > 0, got {}",
                self.nucleus_exponent
            )));
        }
        Ok(())
    }
}

/// Output of [`periodic_hcore`]. All matrices are `(nbasis, nbasis)` in the
/// AO order of the `PreparedBasis`, symmetric.
#[derive(Debug, Clone)]
pub struct PeriodicHcore {
    /// Lattice-summed overlap.
    pub s: Array2<f64>,
    /// Lattice-summed kinetic energy.
    pub t: Array2<f64>,
    /// Real-space erfc part of the nuclear attraction (still containing its
    /// implicit G = 0 term).
    pub v_sr: Array2<f64>,
    /// Reciprocal-space erf part, G ≠ 0.
    pub v_lr: Array2<f64>,
    /// `+π Z_tot S/(ω²Ω)`: removes the SR sum's implicit G = 0 term.
    pub v_g0: Array2<f64>,
    /// `v_sr + v_lr + v_g0` (nuclear attraction with Z_eff; no ECP).
    pub v: Array2<f64>,
    /// Gamma-point periodic ECP `Σ_L V_L` ([`crate::ecp`]); `None` for an
    /// all-electron basis.
    pub v_ecp: Option<Array2<f64>>,
    /// `t + v (+ v_ecp)`.
    pub h: Array2<f64>,
    /// Ewald nuclear repulsion (PySCF `Cell.energy_nuc()` convention).
    pub enn: f64,
    /// The ω used.
    pub omega: f64,
    /// Number of pair images summed for S/T.
    pub n_images: usize,
    /// Number of shifted 3-centre calls in the SR attraction (each unordered
    /// shell pair once; module doc "Orbital-pair symmetry").
    pub n_sr_triplets: usize,
    /// `n_sr_triplets` in ordered-pair units (off-diagonal pairs × 2): the
    /// pre-s2 count, the number to compare benchmarks on.
    pub n_sr_triplets_ordered: usize,
    /// Kept (shell, shell, ECP-image) triples in `v_ecp` (0 without ECP).
    pub n_ecp_triples: usize,
    /// Number of half-sphere G vectors in the LR attraction.
    pub n_g_half: usize,
    /// max |V_SR − V_SRᵀ| before symmetrisation (a lattice sum over an
    /// image set closed under L → −L is symmetric; a large value flags a
    /// truncation or image-set defect). The s2 walk writes `V_SR` exactly
    /// symmetric, so this is 0 unless the construction breaks; the
    /// pre-s2 ordered loop ([`periodic_hcore_pair_s1_oracle`]) reports its
    /// round-off asymmetry here.
    pub sr_asymmetry: f64,
    /// The resolved memory budget (bytes).
    pub budget_bytes: usize,
    /// Bytes reserved on the budget ledger when the `V_LR` G chunks started
    /// (matrices, image lists, nucleus candidates, G list).
    pub lr_resident_bytes: usize,
    /// Bytes one `V_LR` G vector costs (`pair_ft` output + scratch).
    pub lr_bytes_per_g: usize,
    /// Number of `pair_ft` chunks the `V_LR` sum used.
    pub n_lr_chunks: usize,
    /// Stage timings (setup, S/T, SR attraction, LR attraction, ECP, Ewald)
    /// and counters ([`crate::timing`]; observation only).
    pub timings: PbcTimings,
}

struct PrimShell {
    center: [f64; 3],
    exps: Vec<f64>,
    /// Contraction coefficients with `prim_norm(a, l)` folded in.
    coefs: Vec<f64>,
    off: usize,
    dim: usize,
}

fn prim_shells(cell: &Cell, prep: &PreparedBasis) -> Result<Vec<PrimShell>, FerricError> {
    let pos = cell.positions();
    let atoms = prep.atoms();
    if atoms.len() != pos.len() {
        return Err(FerricError::General(format!(
            "periodic_hcore: PreparedBasis has {} atoms but the cell has {}",
            atoms.len(),
            pos.len()
        )));
    }
    for (k, (a, p)) in atoms.iter().zip(&pos).enumerate() {
        let d = ((a.x - p[0]).powi(2) + (a.y - p[1]).powi(2) + (a.z - p[2]).powi(2)).sqrt();
        if d > 1e-10 {
            return Err(FerricError::General(format!(
                "periodic_hcore: PreparedBasis atom {k} is {d:.3e} Bohr from the cell's atom {k}; \
                 build the PreparedBasis from cell.mol()"
            )));
        }
    }
    let dims = prep.shell_dims();
    let offs = prep.shell_offsets();
    let mut out = Vec::with_capacity(prep.nshells());
    for (s, sh) in prep.located_shells().iter().enumerate() {
        if sh.l < 0
            || sh.l as usize > ferric_integrals::md3c1e::MAX_L
            || sh.exponents.is_empty()
            || sh.exponents.len() != sh.coefficients.len()
        {
            return Err(FerricError::Basis(format!(
                "periodic_hcore: malformed or unsupported shell {s} (l={}, max l {}, {} exponents, {} coefficients)",
                sh.l,
                ferric_integrals::md3c1e::MAX_L,
                sh.exponents.len(),
                sh.coefficients.len()
            )));
        }
        let l = sh.l as usize;
        if sh.exponents.iter().any(|&a| !(a > 0.0)) {
            return Err(FerricError::Basis(format!(
                "periodic_hcore: shell {s} has a non-positive exponent"
            )));
        }
        out.push(PrimShell {
            center: sh.center,
            exps: sh.exponents.to_vec(),
            coefs: sh
                .exponents
                .iter()
                .zip(sh.coefficients)
                .map(|(&a, &c)| c * prim_norm(a, l))
                .collect(),
            off: offs[s],
            dim: dims[s],
        });
    }
    Ok(out)
}

/// s-type charge bound of a (shell, shell) primitive-pair set at separation²
/// `r2`: `max |c_a c_b| (π/p)^{3/2} e^{−ab r2/p}`, with the smallest and
/// largest pair exponent.
fn pair_bound(a: &PrimShell, b: &PrimShell, r2: f64) -> (f64, f64, f64) {
    let (mut q, mut pmin, mut pmax) = (0.0_f64, f64::INFINITY, 0.0_f64);
    for (&ea, &ca) in a.exps.iter().zip(&a.coefs) {
        for (&eb, &cb) in b.exps.iter().zip(&b.coefs) {
            let p = ea + eb;
            q = q.max((ca * cb).abs() * (PI / p).powf(1.5) * (-ea * eb / p * r2).exp());
            pmin = pmin.min(p);
            pmax = pmax.max(p);
        }
    }
    (q, pmin, pmax)
}

/// Distance from `x` to the segment `[a, b]`.
fn segment_distance(x: [f64; 3], a: [f64; 3], b: [f64; 3]) -> f64 {
    let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let ax = [x[0] - a[0], x[1] - a[1], x[2] - a[2]];
    let l2 = ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2];
    let t = if l2 > 0.0 {
        ((ax[0] * ab[0] + ax[1] * ab[1] + ax[2] * ab[2]) / l2).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let d = [ax[0] - t * ab[0], ax[1] - t * ab[1], ax[2] - t * ab[2]];
    (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
}

/// Radius beyond which a nucleus contributes < `thresh` to a pair of charge
/// bound `q` (module doc); `None` if the pair is negligible everywhere.
fn nucleus_radius(q: f64, zmax: f64, pmax: f64, omega_p: f64, thresh: f64) -> Option<f64> {
    nucleus_radius_m(q, zmax, pmax, omega_p, thresh, SR_MARGIN_BOHR)
}

/// `q Z_max (1 + 2√(p_max/π))`: the bound's value for a nucleus ON the pair
/// (the Gaussian-smeared potential at the origin is `≤ 2√(p/π)`).
fn nucleus_prefactor(q: f64, zmax: f64, pmax: f64) -> f64 {
    q * zmax * (1.0 + 2.0 * (pmax / PI).sqrt())
}

/// [`nucleus_radius`] with an explicit margin.
fn nucleus_radius_m(
    q: f64,
    zmax: f64,
    pmax: f64,
    omega_p: f64,
    thresh: f64,
    margin: f64,
) -> Option<f64> {
    let pref = nucleus_prefactor(q, zmax, pmax);
    if pref <= thresh {
        return None;
    }
    Some((pref / thresh).ln().sqrt() / omega_p + margin)
}

/// Which bound the SR nucleus screen uses. Production is always
/// [`SrBound::Derived`]; the others are deliberately BROKEN variants that
/// exist only as negative controls for [`sr_screening_study`] (a screening
/// table that cannot tell them apart from `Derived` cannot certify it).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SrBound {
    /// The module-doc bound: per-triplet `B(d) = q Z_max (1 + 2√(p_max/π))
    /// e^{−ω_p² (d − 2)²}`, `ω_p = ω √(p_min/(p_min + ω²))`, `d` = nucleus
    /// distance to the segment `[A, B+L]`.
    Derived,
    /// MUTATION: `ω_p = ω` — drops the Gaussian-extent term (the product
    /// Gaussian's convolution that slows the erfc decay). Anti-conservative
    /// for diffuse pairs (`p_min ≲ ω²`).
    NoGaussianExtent,
    /// MUTATION: no 2-Bohr margin — drops the allowance for l > 0
    /// polynomial prefactors and product centres off the segment ends.
    NoMargin,
}

impl SrBound {
    fn omega_p(self, omega: f64, pmin: f64) -> f64 {
        match self {
            SrBound::NoGaussianExtent => omega,
            _ => omega * (pmin / (pmin + omega * omega)).sqrt(),
        }
    }

    fn margin(self) -> f64 {
        match self {
            SrBound::NoMargin => 0.0,
            _ => SR_MARGIN_BOHR,
        }
    }

    /// Predicted bound of one triplet at nucleus–segment distance `d`.
    fn triplet_bound(self, pref: f64, omega_p: f64, d: f64) -> f64 {
        let x = (d - self.margin()).max(0.0) * omega_p;
        pref * (-x * x).exp()
    }
}

fn add_block(m: &mut Array2<f64>, blk: &[f64], o1: usize, n1: usize, o2: usize, n2: usize, f: f64) {
    for i in 0..n1 {
        for j in 0..n2 {
            m[(o1 + i, o2 + j)] += f * blk[i * n2 + j];
        }
    }
}

fn symmetrize(m: &Array2<f64>) -> (Array2<f64>, f64) {
    let mt = m.t();
    let asym = m
        .iter()
        .zip(mt.iter())
        .fold(0.0_f64, |acc, (a, b)| acc.max((a - b).abs()));
    (0.5 * (m + &mt), asym)
}

/// Reciprocal-lattice vectors with `0 < |G| <= gcut`, one of each `±G` pair:
/// the member whose first non-zero Cartesian component is positive.
/// `Cell::gvectors` builds `−G` as `(−n)·B`, the exact floating-point
/// negation of `G`, so exactly one of each pair is kept (even when a
/// component is a roundoff-level non-zero).
pub(crate) fn half_gvectors(cell: &Cell, gcut: f64) -> Result<Vec<[f64; 3]>, FerricError> {
    Ok(cell
        .gvectors(gcut)?
        .into_iter()
        .filter(|g| {
            if g[0] != 0.0 {
                g[0] > 0.0
            } else if g[1] != 0.0 {
                g[1] > 0.0
            } else {
                g[2] > 0.0
            }
        })
        .collect())
}

/// `−(2/Ω) Σ_{G∈half} v(G) Re[P*(G) S(G)]` with `v = 4π/G² · e^{−G²/4ω²}`
/// (`omega = None`: bare `4π/G²`). Returns the matrix and the G count.
/// G list: `Cell::gvectors` (bound × 24 B) and the half-sphere copy
/// (≤ half of it) coexist briefly.
pub(crate) fn gvector_list_bytes(cell: &Cell, gcut: f64) -> Result<usize, FerricError> {
    Ok(bytes_of(cell.gvector_count_bound(gcut)?, 36))
}

/// Output of [`reciprocal_nuclear`].
struct Reciprocal {
    v: Array2<f64>,
    n_g_half: usize,
    resident_bytes: usize,
    bytes_per_g: usize,
    n_chunks: usize,
}

fn reciprocal_nuclear(
    cell: &Cell,
    prep: &PreparedBasis,
    omega: Option<f64>,
    gcut: f64,
    pair_thresh: f64,
    ledger: &mut Ledger,
) -> Result<Reciprocal, FerricError> {
    let n = prep.nbasis();
    ledger.reserve(
        &format!("reciprocal-space G list (|G| <= {gcut:.3})"),
        gvector_list_bytes(cell, gcut)?,
    )?;
    let gv = half_gvectors(cell, gcut)?;
    let z = cell.nuclear_charges();
    let pos = cell.positions();
    let vol = cell.volume();
    let mut v = Array2::<f64>::zeros((n, n));
    let resident_bytes = ledger.resident();
    let lmax = prep
        .located_shells()
        .iter()
        .map(|s| s.l.max(0) as usize)
        .max()
        .unwrap_or(0);
    let bytes_per_g = pair_ft_bytes_per_g(n, lmax);
    let chunk_budget = ledger.remaining().min(G_CHUNK_BYTES);
    let accumulate =
        |_g0: usize, gs: &[[f64; 3]], p: &Array3<Complex64>| -> Result<(), FerricError> {
            for (g, gvec) in gs.iter().enumerate() {
                let g2 = gvec[0] * gvec[0] + gvec[1] * gvec[1] + gvec[2] * gvec[2];
                let mut kern = 4.0 * PI / g2;
                if let Some(w) = omega {
                    kern *= (-g2 / (4.0 * w * w)).exp();
                }
                // S(G) = Σ_C Z_C e^{−iG·R_C}
                let (mut sre, mut sim) = (0.0_f64, 0.0_f64);
                for (zc, r) in z.iter().zip(&pos) {
                    let ph = gvec[0] * r[0] + gvec[1] * r[1] + gvec[2] * r[2];
                    sre += zc * ph.cos();
                    sim -= zc * ph.sin();
                }
                let f = -2.0 / vol * kern;
                for m in 0..n {
                    for k in 0..n {
                        let pz = p[[m, k, g]];
                        // Re[conj(P) S] = P.re S.re + P.im S.im
                        v[(m, k)] += f * (pz.re * sre + pz.im * sim);
                    }
                }
            }
            Ok(())
        };
    let n_chunks = pair_ft_chunked(cell, prep, &gv, pair_thresh, chunk_budget, 0, accumulate)?;
    Ok(Reciprocal {
        v,
        n_g_half: gv.len(),
        resident_bytes,
        bytes_per_g,
        n_chunks,
    })
}

fn max_pair_exponent(prep: &PreparedBasis) -> f64 {
    2.0 * prep
        .located_shells()
        .iter()
        .flat_map(|s| s.exponents.iter().copied())
        .fold(0.0_f64, f64::max)
}

/// Pure reciprocal-space nuclear attraction (no Ewald split, the ω → ∞
/// limit): `V = −(1/Ω) Σ_{G≠0} (4π/G²) Re[P*(G) S(G)]` with
/// `|G| <= 2 √(p_max ln(1/precision))` (`p_max` = twice the largest
/// exponent; `pbc_gamma.py`'s `w=None` branch). An INDEPENDENT construction
/// of the same `V` as [`periodic_hcore`] (no libint2, no SR sum, no G = 0
/// bookkeeping) — the oracle for its ω-independence. Returns `(V, n_G_half)`.
pub fn pure_aft_nuclear(
    cell: &Cell,
    prep: &PreparedBasis,
    precision: f64,
) -> Result<(Array2<f64>, usize), FerricError> {
    if !(f64::MIN_POSITIVE..1.0).contains(&precision) {
        return Err(FerricError::General(format!(
            "pure_aft_nuclear: precision must lie in (0, 1), got {precision}"
        )));
    }
    prim_shells(cell, prep)?;
    let gcut = 2.0 * (max_pair_exponent(prep) * (1.0 / precision).ln()).sqrt();
    let mut ledger = Ledger::new(crate::budget::resolve(None));
    let n = prep.nbasis();
    ledger.reserve(
        &format!("pure_aft_nuclear n×n matrix (n = {n})"),
        bytes_of((n * n) as u64, 8),
    )?;
    let r = reciprocal_nuclear(cell, prep, None, gcut, 0.1 * precision, &mut ledger)?;
    Ok((r.v, r.n_g_half))
}

/// Molecular (non-periodic) attraction of Gaussian nuclei through the
/// 3-centre engine: `V[μ,ν] = −Σ_C Z_C (g_C | μ ν)_op / ∫g_C`, with `g_C`
/// the unit-normalised s Gaussian of exponent `zeta` at `R_C`. For
/// `op = coulomb` this is the finite-nucleus `⟨μ|−Σ Z_C erf(√ζ r_C)/r_C|ν⟩`;
/// for `erf(ω)`/`erfc(ω)` the attenuated kernels convolved with `g_C`.
/// Evaluated through `Engine::compute_eri3_shifted` at zero shift (the path
/// [`periodic_hcore`] uses), so the closed-form and nuclear-limit tests
/// exercise the production kernel.
pub fn gaussian_nucleus_attraction(
    prep: &PreparedBasis,
    nuclei: &[(f64, [f64; 3])],
    op: Operator,
    zeta: f64,
) -> Result<Array2<f64>, FerricError> {
    let n = prep.nbasis();
    let mut v = Array2::<f64>::zeros((n, n));
    let nuclei: Vec<&(f64, [f64; 3])> = nuclei.iter().filter(|(z, _)| *z != 0.0).collect();
    if nuclei.is_empty() {
        return Ok(v);
    }
    let sites: Vec<[f64; 4]> = nuclei
        .iter()
        .map(|(_, r)| [r[0], r[1], r[2], zeta])
        .collect();
    let site = SiteBasis::new(&sites, 0)?;
    let mut eng = Engine::new_3center(op, prep, &site.prep, ERI3_ENGINE_PRECISION)?;
    let dims = prep.shell_dims().to_vec();
    let offs = prep.shell_offsets().to_vec();
    for (k, (zc, _)) in nuclei.iter().enumerate() {
        let f = -zc / site.norm_int[k];
        for s1 in 0..prep.nshells() {
            for s2 in 0..prep.nshells() {
                if let Some(blk) = eng.compute_eri3_shifted(
                    prep,
                    &site.prep,
                    site.site_shell[k],
                    s1,
                    s2,
                    [[0.0; 3]; 3],
                )? {
                    add_block(&mut v, blk, offs[s1], dims[s1], offs[s2], dims[s2], f);
                }
            }
        }
    }
    Ok(v)
}

/// Nucleus image candidate: `(index into nuc, M, R_C + M)`.
type NucCand = (usize, [f64; 3], [f64; 3]);

/// Pair images `L` for S/T/SR at `pair_thresh` (the module doc's `r_pair`),
/// reserved on the ledger before the enumeration allocates.
fn pair_images(
    cell: &Cell,
    shells: &[PrimShell],
    pair_thresh: f64,
    ledger: &mut Ledger,
) -> Result<Vec<[f64; 3]>, FerricError> {
    let rpair = pair_radius(shells, pair_thresh);
    ledger.reserve(
        &format!("pair-image list (r_pair = {rpair:.2} Bohr)"),
        bytes_of(cell.translation_count_bound(rpair)?, 24),
    )?;
    cell.translations(rpair)
}

/// The cell's nuclei with non-zero charge, and max |Z|.
fn nonzero_nuclei(cell: &Cell) -> (Vec<(f64, [f64; 3])>, f64) {
    let nuc: Vec<(f64, [f64; 3])> = cell
        .nuclear_charges()
        .iter()
        .zip(cell.positions())
        .filter(|(z, _)| **z != 0.0)
        .map(|(z, r)| (*z, r))
        .collect();
    let zmax = nuc.iter().map(|(z, _)| z.abs()).fold(0.0_f64, f64::max);
    (nuc, zmax)
}

/// Candidate nucleus images: any (C, M) within `r_nuc_max(cand_thresh)` of a
/// segment whose endpoints lie within `rpair` of a cell atom, i.e. every
/// `M` with `|M| ≲ r_nuc_max + rpair` (the per-triplet screen then picks from
/// these).
#[allow(clippy::too_many_arguments)]
fn sr_candidates(
    cell: &Cell,
    shells: &[PrimShell],
    nuc: &[(f64, [f64; 3])],
    omega: f64,
    zmax: f64,
    cand_thresh: f64,
    rpair: f64,
    ledger: &mut Ledger,
) -> Result<Vec<NucCand>, FerricError> {
    let mut r_nuc_max = 0.0_f64;
    for a in shells {
        for b in shells {
            let (q, pmin, pmax) = pair_bound(a, b, 0.0);
            let wp = SrBound::Derived.omega_p(omega, pmin);
            if let Some(r) = nucleus_radius(q, zmax, pmax, wp, cand_thresh) {
                r_nuc_max = r_nuc_max.max(r);
            }
        }
    }
    let rc = r_nuc_max + rpair;
    let nb = cell.translation_count_bound(rc)?;
    ledger.reserve(
        &format!(
            "SR nucleus image list + candidates (r = {rc:.2} Bohr, {} nuclei)",
            nuc.len()
        ),
        bytes_of(nb, 24).saturating_add(bytes_of(
            nb.saturating_mul(nuc.len() as u64),
            std::mem::size_of::<NucCand>(),
        )),
    )?;
    let nuc_images = cell.translations(rc)?;
    let mut cands: Vec<NucCand> = Vec::with_capacity(nuc_images.len() * nuc.len());
    for m in &nuc_images {
        for (k, (_, r)) in nuc.iter().enumerate() {
            cands.push((k, *m, [r[0] + m[0], r[1] + m[1], r[2] + m[2]]));
        }
    }
    Ok(cands)
}

/// Unsymmetrised SR attraction and what the screen did.
struct SrSum {
    v: Array2<f64>,
    n_triplets: usize,
    /// `n_triplets` in ordered-pair units (off-diagonal pairs × 2; the
    /// ordered loop's count). Equal to `n_triplets` for the ordered oracle.
    n_triplets_ordered: usize,
    /// Nucleus-candidate segment-distance tests performed (the screen's own
    /// cost, FINDINGS "Performance plan" item 2).
    n_segment_tests: usize,
    /// `n_segment_tests` in ordered-pair units.
    n_segment_tests_ordered: usize,
    /// Per element: Σ over SKIPPED triplets of the bound's per-triplet
    /// prediction (only when tracked).
    predicted: Option<Array2<f64>>,
}

/// The SR nucleus sum `−Σ_{L,C,M} Z_C (g_{C,M} | μ_0 ν_L)_erfc / ∫g` over
/// `images × shells² × cands`. `screen > 0`: skip what `bound` says is below
/// `screen` (production: `SrBound::Derived` at `precision`); `screen == 0`:
/// compute every triplet (the unscreened reference). `track` accumulates the
/// bound's predicted error of the skipped triplets.
#[allow(clippy::too_many_arguments)]
fn sr_attraction(
    prep: &PreparedBasis,
    shells: &[PrimShell],
    images: &[[f64; 3]],
    cands: &[NucCand],
    nuc: &[(f64, [f64; 3])],
    zeta: f64,
    omega: f64,
    zmax: f64,
    screen: f64,
    bound: SrBound,
    track: bool,
) -> Result<SrSum, FerricError> {
    let n = prep.nbasis();
    let mut v = Array2::<f64>::zeros((n, n));
    let mut predicted = track.then(|| Array2::<f64>::zeros((n, n)));
    if nuc.is_empty() {
        return Ok(SrSum {
            v,
            n_triplets: 0,
            n_triplets_ordered: 0,
            n_segment_tests: 0,
            n_segment_tests_ordered: 0,
            predicted,
        });
    }
    let sites: Vec<[f64; 4]> = nuc.iter().map(|(_, r)| [r[0], r[1], r[2], zeta]).collect();
    let site = SiteBasis::new(&sites, 0)?;
    // PARALLEL over UNORDERED shell pairs i1 <= i2 (module doc "Orbital-pair
    // symmetry"): element (μ, ν), μ in shell i1 <= ν's shell i2, receives
    // addends ONLY from pair (i1, i2), in the order "L ascending, then
    // candidate order" — the serial `L → i1 → i2 → candidate` nest's
    // per-element sequence (every screen decision and block is a pure
    // function of (i1, i2, L, candidate)), so row μ ≤ ν is BITWISE the
    // ordered loop's (`sr_attraction_serial_oracle`). Each task accumulates
    // its block (and its predicted-skip scalar, which the ordered loop adds
    // uniformly to every element of the block) from zero with the same `+=`
    // sequence and COPIES it into BOTH (μ, ν) and (ν, μ) of the zeroed
    // output under a mutex; distinct unordered pairs own disjoint elements,
    // so the finishing order of tasks cannot change a bit. Counters are
    // integer sums. One libint engine per rayon worker
    // (`EnginePool::from_fn`); the shifted 3-centre call is stateless.
    // Per-task scratch is one (dim_i1 × dim_i2) block.
    let nsh = shells.len();
    let pool = sr_engine_pool(prep, &site, omega)?;
    let ctx = SrPairCtx {
        prep,
        shells,
        images,
        cands,
        nuc,
        site: &site,
        omega,
        zmax,
        screen,
        bound,
        track,
    };
    let out = Mutex::new((&mut v, predicted.as_mut()));
    let pairs = unordered_pairs(nsh);
    let per_pair: Vec<Result<(usize, usize), FerricError>> = pairs
        .par_iter()
        .map(|&(i1, i2)| ctx.pair_task(&pool, i1, i2, &out))
        .collect();
    let [n_triplets, n_triplets_ordered, n_segment_tests, n_segment_tests_ordered] =
        sum_pair_counts(&pairs, per_pair)?;
    Ok(SrSum {
        v,
        n_triplets,
        n_triplets_ordered,
        n_segment_tests,
        n_segment_tests_ordered,
        predicted,
    })
}

/// One erfc 3-centre engine (AO pair | nucleus site) per rayon worker.
fn sr_engine_pool(
    prep: &PreparedBasis,
    site: &SiteBasis,
    omega: f64,
) -> Result<EnginePool, FerricError> {
    EnginePool::from_fn(|| {
        Engine::new_3center(
            Operator::erfc(omega),
            prep,
            &site.prep,
            ERI3_ENGINE_PRECISION,
        )
    })
}

/// Sum per-unordered-pair `(triplets, segment tests)` in pair order into
/// `[triplets, triplets ordered-equivalent, segment tests, segment tests
/// ordered-equivalent]` (off-diagonal pairs weighted 2); the first error (in
/// pair order, so deterministic) is returned instead.
fn sum_pair_counts(
    pairs: &[(usize, usize)],
    per_pair: Vec<Result<(usize, usize), FerricError>>,
) -> Result<[usize; 4], FerricError> {
    let mut c = [0usize; 4];
    for (&(i1, i2), r) in pairs.iter().zip(per_pair) {
        let (t, sg) = r?;
        let w = if i1 == i2 { 1 } else { 2 };
        c[0] += t;
        c[1] += w * t;
        c[2] += sg;
        c[3] += w * sg;
    }
    Ok(c)
}

/// The zeroed `V` (and optional predicted-skip matrix) every
/// [`sr_attraction`] task copies its finished block into.
type SrOut<'a> = Mutex<(&'a mut Array2<f64>, Option<&'a mut Array2<f64>>)>;

/// The loop invariants of one [`sr_attraction`] call.
struct SrPairCtx<'a> {
    prep: &'a PreparedBasis,
    shells: &'a [PrimShell],
    images: &'a [[f64; 3]],
    cands: &'a [NucCand],
    nuc: &'a [(f64, [f64; 3])],
    site: &'a SiteBasis,
    omega: f64,
    zmax: f64,
    screen: f64,
    bound: SrBound,
    track: bool,
}

/// One pair's running sums: the `(dim_i1 × dim_i2)` block, the
/// predicted-skip scalar and the two counters.
struct SrPairAcc {
    acc: Vec<f64>,
    pred_acc: f64,
    n_trip: usize,
    n_seg: usize,
}

impl SrPairCtx<'_> {
    /// One task: unordered pair `(i1 <= i2)` over every image `L` ascending,
    /// then its finished block COPIED into `out` at `(μ, ν)` and `(ν, μ)`
    /// ([`write_sr_pair_block`]). Returns `(triplets, segment tests)`.
    fn pair_task(
        &self,
        pool: &EnginePool,
        i1: usize,
        i2: usize,
        out: &SrOut<'_>,
    ) -> Result<(usize, usize), FerricError> {
        let (a, b) = (&self.shells[i1], &self.shells[i2]);
        let mut st = SrPairAcc {
            acc: vec![0.0_f64; a.dim * b.dim],
            pred_acc: 0.0,
            n_trip: 0,
            n_seg: 0,
        };
        pool.with(|eng| -> Result<(), FerricError> {
            for l in self.images {
                self.pair_image(eng, i1, i2, l, &mut st)?;
            }
            Ok(())
        })?;
        if st.n_trip > 0 || st.pred_acc != 0.0 {
            write_sr_pair_block(out, a, b, i1 == i2, &st);
        }
        Ok((st.n_trip, st.n_seg))
    }

    /// Pair `(i1, i2)` at image `L`: screen, then every candidate in order.
    fn pair_image(
        &self,
        eng: &mut Engine,
        i1: usize,
        i2: usize,
        l: &[f64; 3],
        st: &mut SrPairAcc,
    ) -> Result<(), FerricError> {
        let (a, b) = (&self.shells[i1], &self.shells[i2]);
        let bound = self.bound;
        let bc = [b.center[0] + l[0], b.center[1] + l[1], b.center[2] + l[2]];
        let r2 = (a.center[0] - bc[0]).powi(2)
            + (a.center[1] - bc[1]).powi(2)
            + (a.center[2] - bc[2]).powi(2);
        let (q, pmin, pmax) = pair_bound(a, b, r2);
        let wp = bound.omega_p(self.omega, pmin);
        let rad = if self.screen > 0.0 {
            nucleus_radius_m(q, self.zmax, pmax, wp, self.screen, bound.margin())
        } else {
            Some(f64::INFINITY)
        };
        if rad.is_none() && !self.track {
            return Ok(());
        }
        let pref = nucleus_prefactor(q, self.zmax, pmax);
        let mut skipped = 0.0_f64;
        st.n_seg += self.cands.len();
        for (k, m, x) in self.cands {
            let d = segment_distance(*x, a.center, bc);
            match rad {
                Some(r) if d <= r => {}
                _ => {
                    if self.track {
                        skipped += bound.triplet_bound(pref, wp, d);
                    }
                    continue;
                }
            }
            st.n_trip += 1;
            let f = -self.nuc[*k].0 / self.site.norm_int[*k];
            if let Some(blk) = eng.compute_eri3_shifted(
                self.prep,
                &self.site.prep,
                self.site.site_shell[*k],
                i1,
                i2,
                [*m, [0.0; 3], *l],
            )? {
                // == add_block(v, blk, a.off, a.dim, b.off, b.dim, f)
                for i in 0..a.dim {
                    for j in 0..b.dim {
                        st.acc[i * b.dim + j] += f * blk[i * b.dim + j];
                    }
                }
            }
        }
        // == `p[(a.off+i, b.off+j)] += skipped` for every element
        // of the block: the same scalar sequence for each.
        if self.track && skipped > 0.0 {
            st.pred_acc += skipped;
        }
        Ok(())
    }
}

/// COPY (not add) unordered pair `(a, b)`'s finished block (and its
/// predicted-skip scalar, uniformly) into `out` under its mutex, at both
/// `(μ, ν)` and `(ν, μ)` ([`copy_block_mirrored`]), so task finishing order
/// cannot change a bit.
fn write_sr_pair_block(out: &SrOut<'_>, a: &PrimShell, b: &PrimShell, same: bool, st: &SrPairAcc) {
    let mut guard = out.lock().unwrap_or_else(|e| e.into_inner());
    let (v, p) = &mut *guard;
    copy_block_mirrored(v, a, b, same, |i, j| st.acc[i * b.dim + j]);
    if let Some(p) = p.as_mut() {
        copy_block_mirrored(p, a, b, same, |_, _| st.pred_acc);
    }
}

/// Write `val(i, j)` of unordered shell pair `(a, b)` into BOTH `m[(a.off +
/// i, b.off + j)]` and `m[(b.off + j, a.off + i)]` (module doc "Orbital-pair
/// symmetry"). A diagonal pair (`same`) writes only `i <= j`, so every
/// element receives exactly one value.
fn copy_block_mirrored<F>(m: &mut Array2<f64>, a: &PrimShell, b: &PrimShell, same: bool, val: F)
where
    F: Fn(usize, usize) -> f64,
{
    for i in 0..a.dim {
        for j in 0..b.dim {
            if same && i > j {
                continue;
            }
            let x = val(i, j);
            m[(a.off + i, b.off + j)] = x;
            m[(b.off + j, a.off + i)] = x;
        }
    }
}

/// One overlap and one kinetic 1e engine per rayon worker.
pub(crate) fn st_engine_pools(prep: &PreparedBasis) -> Result<[EnginePool; 2], FerricError> {
    Ok([
        EnginePool::from_fn(|| Engine::new_1e(ffi::OP_OVERLAP, prep, ONE_E_ENGINE_PRECISION))?,
        EnginePool::from_fn(|| Engine::new_1e(ffi::OP_KINETIC, prep, ONE_E_ENGINE_PRECISION))?,
    ])
}

/// Lattice-summed `S = Σ_L (μ_0|ν_L)` and `T` over `images`, EXACTLY
/// symmetric (module doc "Orbital-pair symmetry").
///
/// PARALLEL over UNORDERED shell pairs `i1 <= i2`: element `(μ, ν)` with `μ`
/// in shell `i1 <= ν`'s shell `i2` receives addends `1.0 · block` (= `block`
/// exactly) ONLY from pair `(i1, i2)`, in `images` order, so it is BITWISE
/// the serial ordered `L → i1 → i2` loop's element
/// ([`overlap_kinetic_serial_oracle`]); `(ν, μ)` receives a copy of it. Each
/// task sums its two blocks from `+0.0` over `L` ascending with the serial
/// `+=` expression and COPIES them into both places of the zeroed outputs
/// under a mutex; distinct unordered pairs own disjoint elements, so task
/// finishing order cannot change a bit (the construction of
/// [`sr_attraction`]). `compute_1e_block_shifted` is stateless (it moves a
/// copy of shell `i2` per call), and every pooled engine is built exactly
/// like the serial one. Errors are returned in pair order.
fn overlap_kinetic(
    prep: &PreparedBasis,
    shells: &[PrimShell],
    images: &[[f64; 3]],
) -> Result<(Array2<f64>, Array2<f64>), FerricError> {
    let n = prep.nbasis();
    let nsh = shells.len();
    let [pool_s, pool_t] = st_engine_pools(prep)?;
    let mut s = Array2::<f64>::zeros((n, n));
    let mut t = Array2::<f64>::zeros((n, n));
    let out = Mutex::new((&mut s, &mut t));
    let per_pair: Vec<Result<(), FerricError>> = unordered_pairs(nsh)
        .into_par_iter()
        .map(|(i1, i2)| {
            let (a, b) = (&shells[i1], &shells[i2]);
            let bl = a.dim * b.dim;
            let (mut acc_s, mut acc_t) = (vec![0.0_f64; bl], vec![0.0_f64; bl]);
            pool_s.with(|es| {
                pool_t.with(|et| -> Result<(), FerricError> {
                    for l in images {
                        let blk = es.compute_1e_block_shifted(prep, i1, i2, *l)?;
                        // == add_block(s, blk, a.off, a.dim, b.off, b.dim, 1.0)
                        // (`1.0 * y` is `y` exactly).
                        for (x, y) in acc_s.iter_mut().zip(&blk[..bl]) {
                            *x += *y;
                        }
                        let blk = et.compute_1e_block_shifted(prep, i1, i2, *l)?;
                        for (x, y) in acc_t.iter_mut().zip(&blk[..bl]) {
                            *x += *y;
                        }
                    }
                    Ok(())
                })
            })?;
            let mut guard = out.lock().unwrap_or_else(|e| e.into_inner());
            let (s, t) = &mut *guard;
            copy_block_mirrored(s, a, b, i1 == i2, |i, j| acc_s[i * b.dim + j]);
            copy_block_mirrored(t, a, b, i1 == i2, |i, j| acc_t[i * b.dim + j]);
            Ok(())
        })
        .collect();
    for r in per_pair {
        r?;
    }
    Ok((s, t))
}

/// The serial `S`/`T` loop as it was before the pair-parallel rewrite of
/// [`overlap_kinetic`] (FROZEN; oracle only — do not "improve").
fn overlap_kinetic_serial_oracle(
    prep: &PreparedBasis,
    shells: &[PrimShell],
    images: &[[f64; 3]],
) -> Result<(Array2<f64>, Array2<f64>), FerricError> {
    let n = prep.nbasis();
    let mut eng_s = Engine::new_1e(ffi::OP_OVERLAP, prep, ONE_E_ENGINE_PRECISION)?;
    let mut eng_t = Engine::new_1e(ffi::OP_KINETIC, prep, ONE_E_ENGINE_PRECISION)?;
    let mut s = Array2::<f64>::zeros((n, n));
    let mut t = Array2::<f64>::zeros((n, n));
    for l in images {
        for (i1, a) in shells.iter().enumerate() {
            for (i2, b) in shells.iter().enumerate() {
                let blk = eng_s.compute_1e_block_shifted(prep, i1, i2, *l)?;
                add_block(&mut s, blk, a.off, a.dim, b.off, b.dim, 1.0);
                let blk = eng_t.compute_1e_block_shifted(prep, i1, i2, *l)?;
                add_block(&mut t, blk, a.off, a.dim, b.off, b.dim, 1.0);
            }
        }
    }
    Ok((s, t))
}

/// TEST ORACLE for the parallel `S`/`T`: `[parallel, serial]` `(S, T)`
/// before `symmetrize`, over the pair images [`periodic_hcore`] uses at
/// `cfg`, plus the image count. `serial` is the pre-parallel ordered
/// `L → i1 → i2` loop, FROZEN verbatim; `parallel` is the s2 walk, whose
/// `(μ, ν)` must be BITWISE `serial[(min, max)]` (module doc "Orbital-pair
/// symmetry"; `tests/pbc_parallel_bitwise.rs`).
#[doc(hidden)]
#[allow(clippy::type_complexity)]
pub fn overlap_kinetic_parallel_and_serial(
    cell: &Cell,
    prep: &PreparedBasis,
    cfg: &PeriodicHcoreConfig,
) -> Result<([(Array2<f64>, Array2<f64>); 2], usize), FerricError> {
    cfg.validate()?;
    let shells = prim_shells(cell, prep)?;
    let mut ledger = Ledger::new(crate::budget::resolve(cfg.budget_bytes));
    let images = pair_images(cell, &shells, 0.1 * cfg.precision, &mut ledger)?;
    Ok((
        [
            overlap_kinetic(prep, &shells, &images)?,
            overlap_kinetic_serial_oracle(prep, &shells, &images)?,
        ],
        images.len(),
    ))
}

/// Build `S`, `T`, `V`, `h = T + V` and `E_nn` for a Gamma-point cell (see
/// the module doc). `prep` must be built from `cell.mol()`.
pub fn periodic_hcore(
    cell: &Cell,
    prep: &PreparedBasis,
    cfg: &PeriodicHcoreConfig,
) -> Result<PeriodicHcore, FerricError> {
    periodic_hcore_impl(cell, prep, cfg, false)
}

/// TEST ORACLE (FROZEN; do not "improve"): [`periodic_hcore`] with the
/// pre-s2 S/T and SR attraction — the FROZEN serial ORDERED loops
/// (`overlap_kinetic_serial_oracle`, `sr_attraction_serial_oracle`, bitwise
/// the pre-s2 parallel production) followed by the μ↔ν average. Bit for bit
/// the build before s2 (its `n_sr_triplets` = `n_sr_triplets_ordered` = the
/// ordered count). Serial: small test cells only.
#[doc(hidden)]
pub fn periodic_hcore_pair_s1_oracle(
    cell: &Cell,
    prep: &PreparedBasis,
    cfg: &PeriodicHcoreConfig,
) -> Result<PeriodicHcore, FerricError> {
    periodic_hcore_impl(cell, prep, cfg, true)
}

fn periodic_hcore_impl(
    cell: &Cell,
    prep: &PreparedBasis,
    cfg: &PeriodicHcoreConfig,
    s1_oracle: bool,
) -> Result<PeriodicHcore, FerricError> {
    cfg.validate()?;
    // Z_eff guard first: a bare Z is silent for every k-mesh anchor.
    crate::ecp::check_ecp_applied(cell, prep.basis_set())?;
    let total = StageClock::start();
    let mut timings = PbcTimings::default();
    let clock = StageClock::start();
    let shells = prim_shells(cell, prep)?;
    let n = prep.nbasis();
    let omega = cfg.omega;
    let thresh = cfg.precision;
    let pair_thresh = 0.1 * thresh;
    let mut ledger = Ledger::new(crate::budget::resolve(cfg.budget_bytes));
    ledger.reserve(
        &format!(
            "periodic_hcore n×n matrices (n = {n}: S, T, V_SR, V_LR, V_G0, V, h + 3 temporaries)"
        ),
        bytes_of((n * n) as u64, 8 * 10),
    )?;

    let images = pair_images(cell, &shells, pair_thresh, &mut ledger)?;
    let rpair = pair_radius(&shells, pair_thresh);
    timings.stop("hcore setup (shells, pair images)", &clock);

    // --- S, T: every pair image (parallel over unordered shell pairs, each
    // element bitwise the serial ordered loop's μ ≤ ν value: `overlap_kinetic`;
    // `symmetrize` is then the identity).
    let clock = StageClock::start();
    let (s, t) = if s1_oracle {
        overlap_kinetic_serial_oracle(prep, &shells, &images)?
    } else {
        overlap_kinetic(prep, &shells, &images)?
    };
    let (s, _) = symmetrize(&s);
    let (t, _) = symmetrize(&t);
    timings.stop("hcore S/T", &clock);

    // --- V_SR: Gaussian nuclei, erfc(ω), nucleus shifted by M, ν by L.
    let clock = StageClock::start();
    let zs = cell.nuclear_charges();
    let (nuc, zmax) = nonzero_nuclei(cell);
    let mut v_sr = Array2::<f64>::zeros((n, n));
    let mut n_sr_triplets = 0usize;
    let mut n_sr_triplets_ordered = 0usize;
    let mut sr_asymmetry = 0.0;
    if !nuc.is_empty() {
        let cands = sr_candidates(cell, &shells, &nuc, omega, zmax, thresh, rpair, &mut ledger)?;
        timings.set_counter("hcore SR nucleus candidates", cands.len() as u64);
        let sr_loop = if s1_oracle {
            sr_attraction_serial_oracle
        } else {
            sr_attraction
        };
        let sr = sr_loop(
            prep,
            &shells,
            &images,
            &cands,
            &nuc,
            cfg.nucleus_exponent,
            omega,
            zmax,
            thresh,
            SrBound::Derived,
            false,
        )?;
        n_sr_triplets = sr.n_triplets;
        n_sr_triplets_ordered = sr.n_triplets_ordered;
        timings.set_counter("hcore SR segment tests", sr.n_segment_tests as u64);
        let (sym, asym) = symmetrize(&sr.v);
        v_sr = sym;
        sr_asymmetry = asym;
    }
    timings.stop("hcore SR attraction", &clock);

    // --- V_LR (G ≠ 0) and the G = 0 correction.
    let clock = StageClock::start();
    let gcut = lr_gcut(prep, omega, thresh);
    let lr = reciprocal_nuclear(cell, prep, Some(omega), gcut, pair_thresh, &mut ledger)?;
    let v_lr = lr.v;
    let ztot: f64 = zs.iter().sum();
    let c0 = PI / (omega * omega * cell.volume());
    let v_g0 = (c0 * ztot) * &s;
    timings.stop("hcore LR attraction", &clock);

    let v = &(&v_sr + &v_lr) + &v_g0;
    // --- V_ECP (Gamma Bloch sum; `None` for an all-electron basis).
    let clock = StageClock::start();
    let ecp = crate::ecp::periodic_ecp_images_on(cell, prep, &cfg.ecp_config(), &mut ledger)?;
    let n_ecp_triples = ecp.as_ref().map_or(0, |e| e.n_triples);
    let v_ecp = ecp.map(|e| e.gamma());
    if v_ecp.is_some() {
        timings.stop("hcore ECP", &clock);
    }
    let mut h = &t + &v;
    if let Some(ve) = &v_ecp {
        h += ve;
    }
    let clock = StageClock::start();
    let enn = ewald_nuclear_repulsion(cell, default_ewald_omega(cell))?;
    timings.stop("hcore Ewald E_nn", &clock);
    ferric_core::memory::warn_if_rss_over("ferric-pbc periodic_hcore", ledger.budget(), 1.1);
    timings.set_counter("hcore pair images", images.len() as u64);
    timings.set_counter("hcore SR triplets", n_sr_triplets as u64);
    timings.set_counter(
        "hcore SR triplets (ordered-equivalent)",
        n_sr_triplets_ordered as u64,
    );
    timings.set_counter("hcore LR half-G", lr.n_g_half as u64);
    timings.set_counter("hcore LR chunks", lr.n_chunks as u64);
    if n_ecp_triples > 0 {
        timings.set_counter("hcore ECP triples", n_ecp_triples as u64);
    }
    timings.finish(&total);
    Ok(PeriodicHcore {
        s,
        t,
        v_sr,
        v_lr,
        v_g0,
        v,
        v_ecp,
        h,
        enn,
        omega,
        n_images: images.len(),
        n_sr_triplets,
        n_sr_triplets_ordered,
        n_ecp_triples,
        n_g_half: lr.n_g_half,
        sr_asymmetry,
        budget_bytes: ledger.budget(),
        lr_resident_bytes: lr.resident_bytes,
        lr_bytes_per_g: lr.bytes_per_g,
        n_lr_chunks: lr.n_chunks,
        timings,
    })
}

/// The `V_LR` G-sphere radius [`periodic_hcore`] uses:
/// `min(2ω, 2√p_max) √ln(1/precision)` (module doc). Shared with the
/// gradient (`crate::grad`) so both sum the same G set.
pub(crate) fn lr_gcut(prep: &PreparedBasis, omega: f64, precision: f64) -> f64 {
    (2.0 * omega).min(2.0 * max_pair_exponent(prep).sqrt()) * (1.0 / precision).ln().sqrt()
}

/// The pair images `L` [`periodic_hcore`] sums `S`/`T`/`V_SR` over at
/// `precision` (pair threshold `precision/10`), reserved on `ledger`.
pub(crate) fn hcore_pair_images(
    cell: &Cell,
    prep: &PreparedBasis,
    precision: f64,
    ledger: &mut Ledger,
) -> Result<Vec<[f64; 3]>, FerricError> {
    let shells = prim_shells(cell, prep)?;
    pair_images(cell, &shells, 0.1 * precision, ledger)
}

/// Gradient of the SR nuclear attraction, `Σ_{μν} D_μν ∂V_SR,μν/∂R_A`, over
/// EXACTLY the pair images, nucleus candidates and per-triplet screen
/// [`periodic_hcore`] uses at `cfg` (same `sr_attraction` loop structure,
/// derivative engine in place of the energy engine). Every term
/// `−Z_C (g_{C,M} | μ_0 ν_L)_erfc / ∫g` moves with three centres: the bra
/// function's atom, the ket function's atom (all images together) and the
/// nucleus `C` (all images `M` together). The nucleus derivative is taken
/// from translation invariance of each 3-centre integral,
/// `∂/∂C = −(∂/∂μ + ∂/∂ν)`, NOT from libint2's site block (the site is a
/// ζ = 1e16 Gaussian; the prototype used the same route, `pbc_grad.py`).
/// libint2's `erfc_nuclear` derivative operator is never used (FINDINGS
/// "Iteration 2": the same libint 2.7.2 bug as the energy).
///
/// Returns `(basis part, nucleus part, n_triplets)`, each `natoms × 3`.
/// Ordered-parallel and bit-identical to the serial walk
/// ([`sr_attraction_deriv_walk`]); `serial` runs the frozen oracle loop.
pub(crate) fn sr_attraction_gradient(
    cell: &Cell,
    prep: &PreparedBasis,
    cfg: &PeriodicHcoreConfig,
    d: &Array2<f64>,
    serial: bool,
    ledger: &mut Ledger,
) -> Result<(Array2<f64>, Array2<f64>, usize), FerricError> {
    let natoms = cell.positions().len();
    let mut g_basis = Array2::<f64>::zeros((natoms, 3));
    let mut g_nuc = Array2::<f64>::zeros((natoms, 3));
    let n_triplets = sr_attraction_deriv_walk(cell, prep, cfg, ledger, serial, |t| {
        for i in 0..t.dim1 {
            for j in 0..t.dim2 {
                let coeff = t.f * d[(t.off1 + i, t.off2 + j)];
                if coeff == 0.0 {
                    continue;
                }
                let idx = i * t.dim2 + j;
                let idx_swapped = j * t.dim1 + i;
                for c in 0..3 {
                    let d1 = coeff * t.ket_a[c * t.bs + idx_swapped];
                    let d2 = coeff * t.ket_b[c * t.bs + idx];
                    g_basis[(t.at1, c)] += d1;
                    g_basis[(t.at2, c)] += d2;
                    g_nuc[(t.atc, c)] -= d1 + d2;
                }
            }
        }
    })?;
    Ok((g_basis, g_nuc, n_triplets))
}

/// Strain derivative of the SR nuclear attraction energy `Σ D V_SR` under
/// `r → (1 + ε) r` (every centre, image and nucleus image scaled; FINDINGS
/// "Iteration 19"): each triplet `(g_{C,M} | μ_0 ν_L)` depends on its three
/// centres `A`, `B′ = B + L`, `X = R_C + M`, and by translation invariance
///
/// ```text
/// d/dε_ab = ∂_A,a (A − X)_b + ∂_B′,a (B′ − X)_b
/// ```
///
/// with the two DIRECTLY computed ket blocks of [`sr_attraction_gradient`]
/// (the nucleus block is never used). Same images, candidates and screen as
/// the energy. `drop_images` (TEST ONLY, mutation `NoSrImages`) uses the
/// un-translated `B` and `R_C` in the pair vectors. Returns `(dE/dε, n_triplets)`.
/// Ordered-parallel and bit-identical to the serial walk; `serial` runs the
/// frozen oracle loop.
pub(crate) fn sr_attraction_strain(
    cell: &Cell,
    prep: &PreparedBasis,
    cfg: &PeriodicHcoreConfig,
    d: &Array2<f64>,
    drop_images: bool,
    serial: bool,
    ledger: &mut Ledger,
) -> Result<([[f64; 3]; 3], usize), FerricError> {
    let mut out = [[0.0_f64; 3]; 3];
    let n_triplets = sr_attraction_deriv_walk(cell, prep, cfg, ledger, serial, |t| {
        let (bb, xx) = if drop_images {
            (
                [
                    t.b_image[0] - t.l[0],
                    t.b_image[1] - t.l[1],
                    t.b_image[2] - t.l[2],
                ],
                [
                    t.nucleus[0] - t.m[0],
                    t.nucleus[1] - t.m[1],
                    t.nucleus[2] - t.m[2],
                ],
            )
        } else {
            (t.b_image, t.nucleus)
        };
        let ra = [
            t.a_center[0] - xx[0],
            t.a_center[1] - xx[1],
            t.a_center[2] - xx[2],
        ];
        let rb = [bb[0] - xx[0], bb[1] - xx[1], bb[2] - xx[2]];
        let mut ga = [0.0_f64; 3];
        let mut gb = [0.0_f64; 3];
        for i in 0..t.dim1 {
            for j in 0..t.dim2 {
                let coeff = t.f * d[(t.off1 + i, t.off2 + j)];
                if coeff == 0.0 {
                    continue;
                }
                let idx = i * t.dim2 + j;
                let idx_swapped = j * t.dim1 + i;
                for c in 0..3 {
                    ga[c] += coeff * t.ket_a[c * t.bs + idx_swapped];
                    gb[c] += coeff * t.ket_b[c * t.bs + idx];
                }
            }
        }
        for a in 0..3 {
            for b in 0..3 {
                out[a][b] += ga[a] * ra[b] + gb[a] * rb[b];
            }
        }
    })?;
    Ok((out, n_triplets))
}

/// One kept SR attraction triplet `(g_{C,M} | μ_0 ν_L)` of
/// [`sr_attraction_deriv_walk`], with its two directly computed ket blocks.
pub(crate) struct SrDerivTriplet<'a> {
    /// AO offset / count of the bra (μ) and ket (ν) shells.
    pub(crate) off1: usize,
    pub(crate) dim1: usize,
    pub(crate) off2: usize,
    pub(crate) dim2: usize,
    /// `dim1 · dim2`.
    pub(crate) bs: usize,
    /// Cell atoms of the bra shell, the ket shell and the nucleus.
    pub(crate) at1: usize,
    pub(crate) at2: usize,
    pub(crate) atc: usize,
    /// `A`, `B + L`, `R_C + M` (Bohr) and the translations `L`, `M`.
    pub(crate) a_center: [f64; 3],
    pub(crate) b_image: [f64; 3],
    pub(crate) nucleus: [f64; 3],
    pub(crate) l: [f64; 3],
    pub(crate) m: [f64; 3],
    /// `−Z_C / ∫g_C`.
    pub(crate) f: f64,
    /// `d/dA` block from the swapped call, `[c · bs + j · dim1 + i]`.
    pub(crate) ket_a: &'a [f64],
    /// `d/d(B + L)` block, `[c · bs + i · dim2 + j]`.
    pub(crate) ket_b: &'a [f64],
}

/// The loop invariants of the SR attraction derivative walk: exactly
/// [`periodic_hcore`]'s pair images, nucleus candidates and screen at `cfg`,
/// with the derivative nucleus sites. Shared by the ordered-parallel walk and
/// its frozen serial oracle.
struct SrDerivSetup {
    shells: Vec<PrimShell>,
    images: Vec<[f64; 3]>,
    nuc: Vec<(f64, [f64; 3])>,
    zmax: f64,
    /// Cell atom of each `nuc` entry.
    nuc_atom: Vec<usize>,
    cands: Vec<NucCand>,
    site: SiteBasis,
    sh2at: Vec<usize>,
    thresh: f64,
    omega: f64,
}

/// `Ok(None)` when the cell has no nonzero nuclear charge (no triplets).
fn sr_deriv_setup(
    cell: &Cell,
    prep: &PreparedBasis,
    cfg: &PeriodicHcoreConfig,
    ledger: &mut Ledger,
) -> Result<Option<SrDerivSetup>, FerricError> {
    cfg.validate()?;
    let shells = prim_shells(cell, prep)?;
    let thresh = cfg.precision;
    let pair_thresh = 0.1 * thresh;
    let images = pair_images(cell, &shells, pair_thresh, ledger)?;
    let rpair = pair_radius(&shells, pair_thresh);
    let (nuc, zmax) = nonzero_nuclei(cell);
    if nuc.is_empty() {
        return Ok(None);
    }
    // nonzero_nuclei keeps the cell order of the Z != 0 atoms.
    let nuc_atom: Vec<usize> = cell
        .nuclear_charges()
        .iter()
        .enumerate()
        .filter(|(_, z)| **z != 0.0)
        .map(|(i, _)| i)
        .collect();
    let omega = cfg.omega;
    let cands = sr_candidates(cell, &shells, &nuc, omega, zmax, thresh, rpair, ledger)?;
    let sites: Vec<[f64; 4]> = nuc
        .iter()
        .map(|(_, r)| [r[0], r[1], r[2], cfg.nucleus_exponent])
        .collect();
    let site = SiteBasis::new(&sites, 0)?;
    Ok(Some(SrDerivSetup {
        shells,
        images,
        nuc,
        zmax,
        nuc_atom,
        cands,
        site,
        sh2at: prep.shell_to_atom().to_vec(),
        thresh,
        omega,
    }))
}

/// One `(L, i1, i2)` unit of the SR attraction derivative walk: the kept
/// triplet count (screen passes, blocks libint reports empty included, as
/// the serial count) and, per triplet with both blocks, its candidate index
/// and `[ket_b (3 bs) | ket_a (3 bs)]`.
#[derive(Default)]
struct SrDerivUnit {
    n_trip: usize,
    cand: Vec<usize>,
    data: Vec<f64>,
}

impl Stored for SrDerivUnit {
    fn stored_bytes(&self) -> usize {
        std::mem::size_of::<Self>() + 8 * (self.cand.len() + self.data.len())
    }
}

impl SrDerivSetup {
    fn engine(&self, prep: &PreparedBasis) -> Result<Engine, FerricError> {
        Engine::new_3center_deriv(
            Operator::erfc(self.omega),
            prep,
            &self.site.prep,
            ERI3_ENGINE_PRECISION,
        )
    }

    /// Unit `u` = `(L, i1, i2)` with `L` outermost (the serial nest order).
    fn unit_indices(&self, u: usize) -> (usize, usize, usize) {
        let nsh = self.shells.len();
        (u / (nsh * nsh), (u / nsh) % nsh, u % nsh)
    }

    /// The PURE part of unit `(L, i1, i2)`: the screen and both derivative
    /// blocks of every kept candidate, in candidate order (the serial
    /// walk's per-unit body up to `visit`).
    fn unit(
        &self,
        eng: &mut Engine,
        prep: &PreparedBasis,
        (li, i1, i2): (usize, usize, usize),
    ) -> Result<SrDerivUnit, FerricError> {
        let (a, b) = (&self.shells[i1], &self.shells[i2]);
        let l = &self.images[li];
        let bound = SrBound::Derived;
        let bc = [b.center[0] + l[0], b.center[1] + l[1], b.center[2] + l[2]];
        let r2 = (a.center[0] - bc[0]).powi(2)
            + (a.center[1] - bc[1]).powi(2)
            + (a.center[2] - bc[2]).powi(2);
        let (q, pmin, pmax) = pair_bound(a, b, r2);
        let wp = bound.omega_p(self.omega, pmin);
        let mut unit = SrDerivUnit::default();
        let Some(rad) = nucleus_radius_m(q, self.zmax, pmax, wp, self.thresh, bound.margin())
        else {
            return Ok(unit);
        };
        let bs = a.dim * b.dim;
        let site = &self.site;
        for (ci, (k, m, x)) in self.cands.iter().enumerate() {
            if segment_distance(*x, a.center, bc) > rad {
                continue;
            }
            unit.n_trip += 1;
            let start = unit.data.len();
            match eng.compute_eri3_deriv_shifted(
                prep,
                &site.prep,
                site.site_shell[*k],
                i1,
                i2,
                [*m, [0.0; 3], *l],
            )? {
                Some(blk) => unit.data.extend_from_slice(&blk[6 * bs..9 * bs]),
                None => continue,
            }
            match eng.compute_eri3_deriv_shifted(
                prep,
                &site.prep,
                site.site_shell[*k],
                i2,
                i1,
                [*m, *l, [0.0; 3]],
            )? {
                Some(blk2) => unit.data.extend_from_slice(&blk2[6 * bs..9 * bs]),
                None => {
                    unit.data.truncate(start);
                    continue;
                }
            }
            unit.cand.push(ci);
        }
        Ok(unit)
    }

    /// The ordered-parallel walk: units `(L, i1, i2)` in serial order,
    /// [`SrDerivSetup::unit`] in parallel (one derivative engine per rayon
    /// worker), [`SrDerivSetup::visit_unit`] serially. Returns the triplet
    /// count.
    fn walk_ordered<F>(
        &self,
        prep: &PreparedBasis,
        budget: usize,
        mut visit: F,
    ) -> Result<usize, FerricError>
    where
        F: FnMut(&SrDerivTriplet<'_>),
    {
        let pool = EnginePool::from_fn(|| self.engine(prep))?;
        let nsh = self.shells.len();
        let mut n_triplets = 0usize;
        ordered_units(
            self.images.len() * nsh * nsh,
            budget,
            0,
            |u| pool.with(|eng| self.unit(eng, prep, self.unit_indices(u))),
            |u, unit: SrDerivUnit| {
                n_triplets += unit.n_trip;
                self.visit_unit(self.unit_indices(u), &unit, &mut visit);
                Ok(())
            },
        )?;
        Ok(n_triplets)
    }

    /// The serial part of unit `(L, i1, i2)`: every stored triplet handed to
    /// `visit` in candidate order, with the serial walk's exact fields.
    fn visit_unit<F>(&self, (li, i1, i2): (usize, usize, usize), unit: &SrDerivUnit, visit: &mut F)
    where
        F: FnMut(&SrDerivTriplet<'_>),
    {
        let (a, b) = (&self.shells[i1], &self.shells[i2]);
        let l = &self.images[li];
        let bc = [b.center[0] + l[0], b.center[1] + l[1], b.center[2] + l[2]];
        let bs = a.dim * b.dim;
        for (t, &ci) in unit.cand.iter().enumerate() {
            let (k, m, x) = &self.cands[ci];
            let blk = &unit.data[6 * bs * t..6 * bs * (t + 1)];
            visit(&SrDerivTriplet {
                off1: a.off,
                dim1: a.dim,
                off2: b.off,
                dim2: b.dim,
                bs,
                at1: self.sh2at[i1],
                at2: self.sh2at[i2],
                atc: self.nuc_atom[*k],
                a_center: a.center,
                b_image: bc,
                nucleus: *x,
                l: *l,
                m: *m,
                f: -self.nuc[*k].0 / self.site.norm_int[*k],
                ket_a: &blk[3 * bs..],
                ket_b: &blk[..3 * bs],
            });
        }
    }
}

/// The SR attraction derivative walk shared by [`sr_attraction_gradient`]
/// and [`sr_attraction_strain`]: exactly [`periodic_hcore`]'s pair images,
/// nucleus candidates and per-triplet screen at `cfg`, with libint2's two
/// directly computed ket blocks per triplet (the bra block of a 3-centre
/// derivative is built from translation invariance, `−(site + sh2)`, and its
/// site derivative is wrong for the ~1e16 Gaussian nucleus; the ket block of
/// `(i1 | i2)` gives `d/dB`, and the ket block of the swapped call
/// `(i2 shifted by L | i1)` gives `d/dA`. Measured: the sh1 block cost
/// −0.041 Ha/Bohr on H2, FINDINGS "Iteration 16" Rust note). Returns the
/// triplet count.
///
/// ORDERED-PARALLEL ([`crate::ordered`]): the screen and the two derivative
/// calls of each `(L, i1, i2)` unit run in parallel over a window of units
/// (one erfc 3-centre derivative engine per rayon worker); `visit` then runs
/// SERIALLY, unit by unit in the serial `L → i1 → i2 → candidate` order,
/// with the same blocks. Every visitor therefore sees exactly the serial
/// sequence of triplets, and its outputs are BIT-IDENTICAL to the serial
/// walk at any thread count — even though the visitors add each triplet
/// into shared atom rows / the stress. `serial` runs the frozen pre-parallel
/// loop instead (the oracle of `tests/pbc_parallel_bitwise.rs`).
fn sr_attraction_deriv_walk<F>(
    cell: &Cell,
    prep: &PreparedBasis,
    cfg: &PeriodicHcoreConfig,
    ledger: &mut Ledger,
    serial: bool,
    visit: F,
) -> Result<usize, FerricError>
where
    F: FnMut(&SrDerivTriplet<'_>),
{
    let Some(s) = sr_deriv_setup(cell, prep, cfg, ledger)? else {
        return Ok(0);
    };
    if serial {
        return sr_attraction_deriv_walk_serial(&s, prep, visit);
    }
    s.walk_ordered(prep, window_budget(ledger.remaining()), visit)
}

/// FROZEN pre-parallel serial SR attraction derivative walk (the
/// bit-identity oracle of [`sr_attraction_deriv_walk`]; do not "improve").
fn sr_attraction_deriv_walk_serial<F>(
    s: &SrDerivSetup,
    prep: &PreparedBasis,
    mut visit: F,
) -> Result<usize, FerricError>
where
    F: FnMut(&SrDerivTriplet<'_>),
{
    let (shells, images, cands, nuc, site) = (&s.shells, &s.images, &s.cands, &s.nuc, &s.site);
    let (zmax, thresh, omega) = (s.zmax, s.thresh, s.omega);
    let mut eng = s.engine(prep)?;
    let sh2at = &s.sh2at;
    let bound = SrBound::Derived;
    let mut n_triplets = 0usize;
    for l in images {
        for (i1, a) in shells.iter().enumerate() {
            for (i2, b) in shells.iter().enumerate() {
                let bc = [b.center[0] + l[0], b.center[1] + l[1], b.center[2] + l[2]];
                let r2 = (a.center[0] - bc[0]).powi(2)
                    + (a.center[1] - bc[1]).powi(2)
                    + (a.center[2] - bc[2]).powi(2);
                let (q, pmin, pmax) = pair_bound(a, b, r2);
                let wp = bound.omega_p(omega, pmin);
                let Some(rad) = nucleus_radius_m(q, zmax, pmax, wp, thresh, bound.margin()) else {
                    continue;
                };
                let (at1, at2) = (sh2at[i1], sh2at[i2]);
                for (k, m, x) in cands {
                    if segment_distance(*x, a.center, bc) > rad {
                        continue;
                    }
                    n_triplets += 1;
                    let f = -nuc[*k].0 / site.norm_int[*k];
                    let Some(blk) = eng.compute_eri3_deriv_shifted(
                        prep,
                        &site.prep,
                        site.site_shell[*k],
                        i1,
                        i2,
                        [*m, [0.0; 3], *l],
                    )?
                    else {
                        continue;
                    };
                    let bs = a.dim * b.dim;
                    let ket_b: Vec<f64> = blk[6 * bs..9 * bs].to_vec();
                    let Some(blk2) = eng.compute_eri3_deriv_shifted(
                        prep,
                        &site.prep,
                        site.site_shell[*k],
                        i2,
                        i1,
                        [*m, *l, [0.0; 3]],
                    )?
                    else {
                        continue;
                    };
                    let ket_a: &[f64] = &blk2[6 * bs..9 * bs];
                    visit(&SrDerivTriplet {
                        off1: a.off,
                        dim1: a.dim,
                        off2: b.off,
                        dim2: b.dim,
                        bs,
                        at1,
                        at2,
                        atc: s.nuc_atom[*k],
                        a_center: a.center,
                        b_image: bc,
                        nucleus: *x,
                        l: *l,
                        m: *m,
                        f,
                        ket_a,
                        ket_b: &ket_b,
                    });
                }
            }
        }
    }
    Ok(n_triplets)
}

/// The SR attraction derivative walk of [`sr_attraction_gradient`] with each
/// triplet handed to `visit` (its image `L` included), for weights that
/// depend on `L` — the k-point forces ([`crate::kgrad`]) weight each image by
/// the Bloch-phase-folded density `(1/N_k) Σ_k e^{ik·L} D(k)`. Same pair
/// images, nucleus candidates and screen as [`periodic_hcore`] at `cfg`;
/// ordered-parallel, `visit` serial in walk order (`serial`: the frozen
/// oracle loop).
pub(crate) fn sr_attraction_deriv_visit<F>(
    cell: &Cell,
    prep: &PreparedBasis,
    cfg: &PeriodicHcoreConfig,
    ledger: &mut Ledger,
    serial: bool,
    visit: F,
) -> Result<usize, FerricError>
where
    F: FnMut(&SrDerivTriplet<'_>),
{
    sr_attraction_deriv_walk(cell, prep, cfg, ledger, serial, visit)
}

fn pair_radius(shells: &[PrimShell], pair_thresh: f64) -> f64 {
    let amin = shells
        .iter()
        .flat_map(|s| s.exps.iter().copied())
        .fold(f64::INFINITY, f64::min);
    (2.0 * (1e3 / pair_thresh).ln() / amin).sqrt() + 2.0
}

/// MEASUREMENT ONLY (Stage 1 step 8): the unsymmetrised SR attraction `V_SR`
/// of [`periodic_hcore`] with the candidate nucleus images and pair images
/// built at `cand_thresh` (exactly as `periodic_hcore` builds them at
/// `precision = cand_thresh`) and the per-triplet screen applied at
/// `screen_thresh` with `bound` (`0` = unscreened: every candidate triplet).
/// Returns `(V_SR, n_triplets computed)`; `V_SR` is the s2 walk's, exactly
/// symmetric (module doc "Orbital-pair symmetry"). At `cand_thresh =
/// screen_thresh = precision` and [`SrBound::Derived`], its symmetrisation
/// IS `periodic_hcore(..).v_sr` (anchored in `tests/pbc_sr_screening.rs`).
pub fn sr_attraction_matrix(
    cell: &Cell,
    prep: &PreparedBasis,
    omega: f64,
    cand_thresh: f64,
    screen_thresh: f64,
    bound: SrBound,
) -> Result<(Array2<f64>, usize), FerricError> {
    let st = SrStudySetup::new(cell, prep, omega, cand_thresh, None)?;
    let sr = st.run(prep, screen_thresh, bound, false)?;
    Ok((sr.v, sr.n_triplets))
}

/// One SR-attraction result as `(V_SR unsymmetrised, n_triplets,
/// n_segment_tests, predicted-skip matrix if tracked)`.
pub type SrAttractionParts = (Array2<f64>, usize, usize, Option<Array2<f64>>);

/// TEST ORACLE for the parallel SR attraction: `[parallel, serial]` over the
/// SAME shells, pair images and nucleus candidates as
/// [`sr_attraction_matrix`] (built at `cand_thresh`), screened at
/// `screen_thresh` with `bound`, optionally tracking the predicted skip.
/// `serial` is the pre-parallel ordered `L → i1 → i2 → candidate` loop,
/// FROZEN verbatim (`sr_attraction_serial_oracle`); `parallel` is the
/// production s2 walk (module doc "Orbital-pair symmetry"), whose `(μ, ν)`
/// (and predicted-skip) elements must be BITWISE `serial[(min, max)]`
/// (`tests/pbc_parallel_bitwise.rs`), the proof that the pair-outer nest kept
/// every element's summation sequence. The parallel counts are reported in
/// ORDERED-pair units (off-diagonal pairs × 2) so they compare 1:1 with the
/// serial loop's.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn sr_attraction_parallel_and_serial(
    cell: &Cell,
    prep: &PreparedBasis,
    omega: f64,
    cand_thresh: f64,
    screen_thresh: f64,
    bound: SrBound,
    track: bool,
) -> Result<[SrAttractionParts; 2], FerricError> {
    let st = SrStudySetup::new(cell, prep, omega, cand_thresh, None)?;
    let par = st.run(prep, screen_thresh, bound, track)?;
    let ser = sr_attraction_serial_oracle(
        prep,
        &st.shells,
        &st.images,
        &st.cands,
        &st.nuc,
        GAUSSIAN_NUCLEUS_EXPONENT,
        st.omega,
        st.zmax,
        screen_thresh,
        bound,
        track,
    )?;
    // The s2 counts in ordered-pair units, so they compare 1:1 with the
    // ordered loop's.
    Ok([
        (
            par.v,
            par.n_triplets_ordered,
            par.n_segment_tests_ordered,
            par.predicted,
        ),
        (ser.v, ser.n_triplets, ser.n_segment_tests, ser.predicted),
    ])
}

/// The serial SR attraction loop as it was before the pair-parallel
/// rewrite of [`sr_attraction`] (FROZEN; oracle only — do not "improve").
#[allow(clippy::too_many_arguments)]
fn sr_attraction_serial_oracle(
    prep: &PreparedBasis,
    shells: &[PrimShell],
    images: &[[f64; 3]],
    cands: &[NucCand],
    nuc: &[(f64, [f64; 3])],
    zeta: f64,
    omega: f64,
    zmax: f64,
    screen: f64,
    bound: SrBound,
    track: bool,
) -> Result<SrSum, FerricError> {
    let n = prep.nbasis();
    let mut v = Array2::<f64>::zeros((n, n));
    let mut predicted = track.then(|| Array2::<f64>::zeros((n, n)));
    let mut n_triplets = 0usize;
    let mut n_segment_tests = 0usize;
    if nuc.is_empty() {
        return Ok(SrSum {
            v,
            n_triplets,
            n_triplets_ordered: n_triplets,
            n_segment_tests,
            n_segment_tests_ordered: n_segment_tests,
            predicted,
        });
    }
    let sites: Vec<[f64; 4]> = nuc.iter().map(|(_, r)| [r[0], r[1], r[2], zeta]).collect();
    let site = SiteBasis::new(&sites, 0)?;
    let mut eng = Engine::new_3center(
        Operator::erfc(omega),
        prep,
        &site.prep,
        ERI3_ENGINE_PRECISION,
    )?;
    for l in images {
        for (i1, a) in shells.iter().enumerate() {
            for (i2, b) in shells.iter().enumerate() {
                let bc = [b.center[0] + l[0], b.center[1] + l[1], b.center[2] + l[2]];
                let r2 = (a.center[0] - bc[0]).powi(2)
                    + (a.center[1] - bc[1]).powi(2)
                    + (a.center[2] - bc[2]).powi(2);
                let (q, pmin, pmax) = pair_bound(a, b, r2);
                let wp = bound.omega_p(omega, pmin);
                let rad = if screen > 0.0 {
                    nucleus_radius_m(q, zmax, pmax, wp, screen, bound.margin())
                } else {
                    Some(f64::INFINITY)
                };
                if rad.is_none() && !track {
                    continue;
                }
                let pref = nucleus_prefactor(q, zmax, pmax);
                let mut skipped = 0.0_f64;
                n_segment_tests += cands.len();
                for (k, m, x) in cands {
                    let d = segment_distance(*x, a.center, bc);
                    match rad {
                        Some(r) if d <= r => {}
                        _ => {
                            if track {
                                skipped += bound.triplet_bound(pref, wp, d);
                            }
                            continue;
                        }
                    }
                    n_triplets += 1;
                    let f = -nuc[*k].0 / site.norm_int[*k];
                    if let Some(blk) = eng.compute_eri3_shifted(
                        prep,
                        &site.prep,
                        site.site_shell[*k],
                        i1,
                        i2,
                        [*m, [0.0; 3], *l],
                    )? {
                        add_block(&mut v, blk, a.off, a.dim, b.off, b.dim, f);
                    }
                }
                if let Some(p) = predicted.as_mut() {
                    if skipped > 0.0 {
                        for i in 0..a.dim {
                            for j in 0..b.dim {
                                p[(a.off + i, b.off + j)] += skipped;
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(SrSum {
        v,
        n_triplets,
        n_triplets_ordered: n_triplets,
        n_segment_tests,
        n_segment_tests_ordered: n_segment_tests,
        predicted,
    })
}

/// Shared setup of the step-8 measurement entry points.
struct SrStudySetup {
    shells: Vec<PrimShell>,
    images: Vec<[f64; 3]>,
    cands: Vec<NucCand>,
    nuc: Vec<(f64, [f64; 3])>,
    zmax: f64,
    omega: f64,
    ledger: Ledger,
}

impl SrStudySetup {
    fn new(
        cell: &Cell,
        prep: &PreparedBasis,
        omega: f64,
        cand_thresh: f64,
        budget_bytes: Option<usize>,
    ) -> Result<Self, FerricError> {
        PeriodicHcoreConfig {
            precision: cand_thresh,
            ..PeriodicHcoreConfig::with_omega(omega)
        }
        .validate()?;
        let shells = prim_shells(cell, prep)?;
        let mut ledger = Ledger::new(crate::budget::resolve(budget_bytes));
        let pair_thresh = 0.1 * cand_thresh;
        let images = pair_images(cell, &shells, pair_thresh, &mut ledger)?;
        let rpair = pair_radius(&shells, pair_thresh);
        let (nuc, zmax) = nonzero_nuclei(cell);
        let cands = sr_candidates(
            cell,
            &shells,
            &nuc,
            omega,
            zmax,
            cand_thresh,
            rpair,
            &mut ledger,
        )?;
        Ok(Self {
            shells,
            images,
            cands,
            nuc,
            zmax,
            omega,
            ledger,
        })
    }

    fn run(
        &self,
        prep: &PreparedBasis,
        screen: f64,
        bound: SrBound,
        track: bool,
    ) -> Result<SrSum, FerricError> {
        if !(screen >= 0.0) || !screen.is_finite() {
            return Err(FerricError::General(format!(
                "sr screening study: screen threshold must be finite and >= 0, got {screen}"
            )));
        }
        sr_attraction(
            prep,
            &self.shells,
            &self.images,
            &self.cands,
            &self.nuc,
            GAUSSIAN_NUCLEUS_EXPONENT,
            self.omega,
            self.zmax,
            screen,
            bound,
            track,
        )
    }
}

/// One row of [`sr_screening_study`].
#[derive(Debug, Clone)]
pub struct SrScreenRow {
    /// Bound the screen used.
    pub bound: SrBound,
    /// Screen threshold (`0` = unscreened).
    pub thresh: f64,
    /// SR triplets computed.
    pub n_triplets: usize,
    /// `max_ij |V_ij(thresh) − V_ij(unscreened)|` (unsymmetrised).
    pub max_abs_dv: f64,
    /// `max_ij` of the bound's predicted error (Σ over skipped triplets).
    pub max_predicted: f64,
    /// Elements with `|ΔV_ij| > predicted_ij + roundoff_floor`: the bound
    /// UNDER-predicted there (a non-conservative bound).
    pub n_violations: usize,
    /// `max_ij |ΔV_ij| / predicted_ij` over elements with
    /// `|ΔV_ij| > roundoff_floor` (0 if none): `> 1` = violation.
    pub worst_ratio: f64,
}

/// Output of [`sr_screening_study`].
#[derive(Debug, Clone)]
pub struct SrScreenStudy {
    /// ω (Bohr⁻¹).
    pub omega: f64,
    /// Threshold the candidate/pair image sets were built at.
    pub cand_thresh: f64,
    /// Pair images in the fixed set.
    pub n_images: usize,
    /// Nucleus candidates `(C, M)` in the fixed set.
    pub n_candidates: usize,
    /// Triplets in the unscreened reference.
    pub n_triplets_unscreened: usize,
    /// `16 ε max|V_unscreened|`: differences below it are summation-order
    /// roundoff, not screening error.
    pub roundoff_floor: f64,
    /// Unscreened reference (unsymmetrised).
    pub v_unscreened: Array2<f64>,
    /// One row per (bound, threshold), bounds outer, in the given orders.
    pub rows: Vec<SrScreenRow>,
}

/// MEASUREMENT HARNESS (Stage 1 step 8, not a production path): sweep the SR
/// nucleus screen threshold with the candidate and pair image sets FIXED at
/// `cand_thresh` (so the only variable is the per-triplet screen), and
/// compare each screened `V_SR` against the unscreened sum over the same sets
/// and against the bound's own predicted error, for each of `bounds` (the
/// unscreened reference is computed once). The reference is exact up to
/// the candidate truncation at `cand_thresh`, which should sit well below the
/// smallest swept threshold. Every buffer is gated against `budget_bytes`
/// (`None` = ferric's unified budget).
pub fn sr_screening_study(
    cell: &Cell,
    prep: &PreparedBasis,
    omega: f64,
    cand_thresh: f64,
    thresholds: &[f64],
    bounds: &[SrBound],
    budget_bytes: Option<usize>,
) -> Result<SrScreenStudy, FerricError> {
    let mut st = SrStudySetup::new(cell, prep, omega, cand_thresh, budget_bytes)?;
    let n = prep.nbasis();
    // Reference + one screened V + its prediction live at once.
    st.ledger.reserve(
        &format!("SR screening study n×n matrices (n = {n})"),
        bytes_of((n * n) as u64, 8 * 4),
    )?;
    let reference = st.run(prep, 0.0, SrBound::Derived, false)?;
    let vmax = reference.v.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
    let roundoff_floor = 16.0 * f64::EPSILON * vmax.max(1.0);
    let mut rows = Vec::with_capacity(thresholds.len() * bounds.len());
    for (&bound, &th) in bounds
        .iter()
        .flat_map(|b| thresholds.iter().map(move |t| (b, t)))
    {
        let sr = st.run(prep, th, bound, true)?;
        let pred = sr.predicted.as_ref().expect("tracked");
        let (mut max_dv, mut max_pred, mut nviol, mut worst) = (0.0_f64, 0.0_f64, 0usize, 0.0_f64);
        for ((x, r), p) in sr.v.iter().zip(reference.v.iter()).zip(pred.iter()) {
            let dv = (x - r).abs();
            max_dv = max_dv.max(dv);
            max_pred = max_pred.max(*p);
            if dv > p + roundoff_floor {
                nviol += 1;
            }
            if dv > roundoff_floor {
                worst = worst.max(if *p > 0.0 { dv / p } else { f64::INFINITY });
            }
        }
        rows.push(SrScreenRow {
            bound,
            thresh: th,
            n_triplets: sr.n_triplets,
            max_abs_dv: max_dv,
            max_predicted: max_pred,
            n_violations: nviol,
            worst_ratio: worst,
        });
    }
    Ok(SrScreenStudy {
        omega,
        cand_thresh,
        n_images: st.images.len(),
        n_candidates: st.cands.len(),
        n_triplets_unscreened: reference.n_triplets,
        roundoff_floor,
        v_unscreened: reference.v,
        rows,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn segment_distance_endpoints_and_interior() {
        let a = [0.0, 0.0, 0.0];
        let b = [2.0, 0.0, 0.0];
        assert!((segment_distance([1.0, 3.0, 0.0], a, b) - 3.0).abs() < 1e-15);
        assert!((segment_distance([-1.0, 0.0, 0.0], a, b) - 1.0).abs() < 1e-15);
        assert!((segment_distance([5.0, 4.0, 0.0], a, b) - 5.0).abs() < 1e-15);
        // Degenerate segment = point distance.
        assert!((segment_distance([3.0, 4.0, 0.0], a, a) - 5.0).abs() < 1e-15);
    }

    #[test]
    fn nucleus_radius_bounds_the_s_type_erfc_tail() {
        // At the returned radius (minus the margin), q Z erfc(ω_p r)/r must
        // already be below the threshold.
        let (q, z, pmax, wp, t) = (0.3, 8.0, 20.0, 0.6, 1e-14);
        let r = nucleus_radius(q, z, pmax, wp, t).unwrap() - SR_MARGIN_BOHR;
        let tail = q * z * ferric_integrals::qqr3::erfc(wp * r) / r;
        assert!(tail < t, "tail {tail:e} at r = {r}");
        assert!(nucleus_radius(1e-20, 1.0, 1.0, 1.0, 1e-14).is_none());
    }
}
