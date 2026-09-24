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

use crate::budget::{bytes_of, Ledger};
use crate::ewald::{default_ewald_omega, ewald_nuclear_repulsion};
use crate::lattice::Cell;
use crate::pair_ft::{pair_ft_bytes_per_g, pair_ft_chunked};
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::engine::Engine;
use ferric_integrals::ffi;
use ferric_integrals::md3c1e::prim_norm;
use ferric_integrals::operator::Operator;
use ferric_integrals::site_basis::SiteBasis;
use ndarray::{Array2, Array3};
use num_complex::Complex64;
use std::f64::consts::PI;

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
const ERI3_ENGINE_PRECISION: f64 = f64::MIN_POSITIVE;
/// Precision for the overlap/kinetic engines (value used by the passing
/// `pbc_shifted_overlap` anchor, tightened).
const ONE_E_ENGINE_PRECISION: f64 = 1e-16;
/// Extra Bohr on every derived real-space radius (polynomial prefactors of
/// l > 0 pairs are not in the s-type bounds).
const SR_MARGIN_BOHR: f64 = 2.0;
/// Upper bound on one `pair_ft` chunk (output plus scratch), whatever the
/// budget: large enough for BLAS-friendly chunks, small enough to stay out of
/// the way of the resident matrices.
pub(crate) const G_CHUNK_BYTES: usize = 64 << 20;

/// Settings for [`periodic_hcore`].
#[derive(Debug, Clone, Copy)]
pub struct PeriodicHcoreConfig {
    /// Ewald splitting parameter for the nuclear attraction (Bohr⁻¹). Any
    /// `ω > 0` gives the same `V` up to truncation; larger ω moves work
    /// from the real-space SR sum to reciprocal space.
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
    /// Number of shifted 3-centre calls in the SR attraction.
    pub n_sr_triplets: usize,
    /// Kept (shell, shell, ECP-image) triples in `v_ecp` (0 without ECP).
    pub n_ecp_triples: usize,
    /// Number of half-sphere G vectors in the LR attraction.
    pub n_g_half: usize,
    /// max |V_SR − V_SRᵀ| before symmetrisation (a lattice sum over an
    /// image set closed under L → −L is symmetric; a large value flags a
    /// truncation or image-set defect).
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
    let mut n_triplets = 0usize;
    if nuc.is_empty() {
        return Ok(SrSum {
            v,
            n_triplets,
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
        predicted,
    })
}

/// Build `S`, `T`, `V`, `h = T + V` and `E_nn` for a Gamma-point cell (see
/// the module doc). `prep` must be built from `cell.mol()`.
pub fn periodic_hcore(
    cell: &Cell,
    prep: &PreparedBasis,
    cfg: &PeriodicHcoreConfig,
) -> Result<PeriodicHcore, FerricError> {
    cfg.validate()?;
    // Z_eff guard first: a bare Z is silent for every k-mesh anchor.
    crate::ecp::check_ecp_applied(cell, prep.basis_set())?;
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

    // --- S, T: every pair image.
    let mut eng_s = Engine::new_1e(ffi::OP_OVERLAP, prep, ONE_E_ENGINE_PRECISION)?;
    let mut eng_t = Engine::new_1e(ffi::OP_KINETIC, prep, ONE_E_ENGINE_PRECISION)?;
    let mut s = Array2::<f64>::zeros((n, n));
    let mut t = Array2::<f64>::zeros((n, n));
    for l in &images {
        for (i1, a) in shells.iter().enumerate() {
            for (i2, b) in shells.iter().enumerate() {
                let blk = eng_s.compute_1e_block_shifted(prep, i1, i2, *l)?;
                add_block(&mut s, blk, a.off, a.dim, b.off, b.dim, 1.0);
                let blk = eng_t.compute_1e_block_shifted(prep, i1, i2, *l)?;
                add_block(&mut t, blk, a.off, a.dim, b.off, b.dim, 1.0);
            }
        }
    }
    let (s, _) = symmetrize(&s);
    let (t, _) = symmetrize(&t);

    // --- V_SR: Gaussian nuclei, erfc(ω), nucleus shifted by M, ν by L.
    let zs = cell.nuclear_charges();
    let (nuc, zmax) = nonzero_nuclei(cell);
    let mut v_sr = Array2::<f64>::zeros((n, n));
    let mut n_sr_triplets = 0usize;
    let mut sr_asymmetry = 0.0;
    if !nuc.is_empty() {
        let cands = sr_candidates(cell, &shells, &nuc, omega, zmax, thresh, rpair, &mut ledger)?;
        let sr = sr_attraction(
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
        let (sym, asym) = symmetrize(&sr.v);
        v_sr = sym;
        sr_asymmetry = asym;
    }

    // --- V_LR (G ≠ 0) and the G = 0 correction.
    let smooth = (1.0 / thresh).ln().sqrt();
    let gcut = (2.0 * omega).min(2.0 * max_pair_exponent(prep).sqrt()) * smooth;
    let lr = reciprocal_nuclear(cell, prep, Some(omega), gcut, pair_thresh, &mut ledger)?;
    let v_lr = lr.v;
    let ztot: f64 = zs.iter().sum();
    let c0 = PI / (omega * omega * cell.volume());
    let v_g0 = (c0 * ztot) * &s;

    let v = &(&v_sr + &v_lr) + &v_g0;
    // --- V_ECP (Gamma Bloch sum; `None` for an all-electron basis).
    let ecp = crate::ecp::periodic_ecp_images_on(cell, prep, &cfg.ecp_config(), &mut ledger)?;
    let n_ecp_triples = ecp.as_ref().map_or(0, |e| e.n_triples);
    let v_ecp = ecp.map(|e| e.gamma());
    let mut h = &t + &v;
    if let Some(ve) = &v_ecp {
        h += ve;
    }
    let enn = ewald_nuclear_repulsion(cell, default_ewald_omega(cell))?;
    ferric_core::memory::warn_if_rss_over("ferric-pbc periodic_hcore", ledger.budget(), 1.1);
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
        n_ecp_triples,
        n_g_half: lr.n_g_half,
        sr_asymmetry,
        budget_bytes: ledger.budget(),
        lr_resident_bytes: lr.resident_bytes,
        lr_bytes_per_g: lr.bytes_per_g,
        n_lr_chunks: lr.n_chunks,
    })
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
/// Returns `(V_SR, n_triplets)`. At `cand_thresh = screen_thresh =
/// precision` and [`SrBound::Derived`], its symmetrisation IS
/// `periodic_hcore(..).v_sr` (anchored in `tests/pbc_sr_screening.rs`).
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
