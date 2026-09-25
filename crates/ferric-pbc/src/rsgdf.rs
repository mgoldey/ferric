//! Gamma-point range-separated Gaussian density fitting (RS-GDF) and the
//! J/K builders on it (Stage 1 step 9, `reference/pbc/stage1-design.md` §4;
//! Rust port of the validated prototype `reference/pbc/pbc_gdf.py`,
//! `reference/pbc/FINDINGS.md` "Iteration 2 (Python, RS-GDF)").
//!
//! # Target
//!
//! `B[k, μν]` with `I[μν, λσ] ≈ Σ_k B[k,μν] B[k,λσ]`, where `I` is the
//! Gamma ERI of [`crate::dense_aft`] in its G = 0 convention:
//!
//! ```text
//! I[μν,λσ] = (1/Ω) Σ_{G≠0} 4π/G² conj(P_μν(G)) P_λσ(G)
//! ```
//!
//! Every fitted quantity lives in the SAME G = 0-dropped kernel
//! `v'(G) = 4π/G²` (G ≠ 0):
//!
//! ```text
//! J2[P,Q]  = (P|Q)'  = (1/Ω) Σ_{G≠0} v(G) conj(X_P(G)) X_Q(G)       (aux metric)
//! J3[μν,P] = (μν|P)' = (1/Ω) Σ_{G≠0} v(G) conj(P_μν(G)) X_P(G)
//! J2 = U s Uᵀ,  B = s^{-1/2} Uᵀ J3ᵀ   over the eigenvalues s > lindep       (Dunlap-robust)
//! ```
//!
//! Because the fitting metric IS the target kernel, the fit is variational in
//! the kernel the energy uses; charged aux functions need no compensating
//! charge because the only divergent term (G = 0) is removed identically from
//! J2 and J3. This is the PySCF RSGDF convention (`rsdf_builder.py`
//! `get_2c2e` `g0_fac`, `gen_j3c_loader` `vbar`).
//!
//! # Evaluation (Ewald split `1/r = erfc(ωr)/r + erf(ωr)/r`)
//!
//! * **SR** — real-space lattice sums of ordinary libint2 `erfc(ω)` integrals
//!   on translated shells (no image bases):
//!   `J2 += Σ_T (P_0|Q_T)` via `Engine::compute_eri2_shifted`,
//!   `J3 += Σ_{L,T} (μ_0 ν_L|P_T)` via `Engine::compute_eri3_shifted`.
//!   These sums implicitly contain the erfc G = 0 term.
//! * **LR** — `(1/Ω) Σ_{G≠0} 4π/G² e^{−G²/4ω²} conj(A(G)) X_P(G)` on the half
//!   G sphere (weight 2), `A` = aux FT ([`aux_ft`]) or pair FT
//!   ([`crate::pair_ft::pair_ft_chunked`], G-chunked, never all G at once).
//! * **G = 0** — ONE function ([`subtract_g0`]) removes `c0 q qᵀ` from J2 and
//!   `c0 S_μν q_P` from J3, `c0 = π/(ω²Ω)`, `q_P = ∫χ_P = X_P(0)`. The result
//!   is ω-independent (tested).
//!
//! # Metric solve
//!
//! Symmetric eigendecomposition with an ABSOLUTE eigenvalue cut `lindep`
//! (default 1e-10 = PySCF `LINEAR_DEP_THR`) and a REPORTED dropped count
//! ([`RsGdfStats::n_dropped`]); never plain Cholesky or LU. Diffuse aux in a
//! small cell make the periodic metric near-singular (prototype LiH: 43/240
//! eigenvalues < 1e-10), and the eig cut was stable over 8 decades of
//! threshold where PySCF's default path was not.
//!
//! # Screening (all derived from `precision`)
//!
//! For a (μ-shell, ν-shell at L) pair with s-type charge bound
//! `q_ab = max |c_a c_b| (π/p)^{3/2} e^{−ab R²/p}` and an aux shell with charge
//! bound `Q_P = max_k |c_k| (π/α_k)^{3/2} (1 + α_k^{−1/2})^{l}` (the prototype's
//! "SR cutoff scaled by |q_P|": normalised diffuse aux carry |q| ≫ 1), the
//! erfc interaction of two Gaussian distributions at distance `d` is at most
//! `q_ab Q_P (1 + 2μ/√π) e^{−ν² (d − 2)²}` with `μ² = pα/(p+α)` (largest
//! exponents) and `1/ν² = 1/p_min + 1/α_min + 1/ω²` (smallest). `d` is the
//! distance from the aux image centre to the segment `[A, B+L]` (every product
//! centre lies on it); the 2-Bohr margin absorbs the l > 0 polynomial factors
//! (the same construction as `hcore`'s SR nucleus screen). A triplet is
//! computed iff that bound is `>= precision`. `sr_screen = false` replaces
//! the per-triplet radius by the global maximum (the prototype's unscreened
//! image-shell form), for measuring what the screen drops.
//!
//! # Memory
//!
//! Every buffer that grows with the lattice, the G sphere or `naux·nao²` is
//! reserved on a [`crate::budget`] ledger before allocation against
//! [`RsGdfConfig::budget_bytes`] (default: ferric's unified budget). The
//! resident B is `8 naux nao²` bytes (dense, in core — no spill path yet).
//!
//! # Consumption (J/K)
//!
//! [`RsGdfJ`]/[`RsGdfK`] implement `ferric_scf::fock::{JBuilder, KBuilder}`
//! directly from the in-core B (J = Σ_k B_k (B_k·D), K = Σ_k B_k D B_kᵀ +
//! `v_M S D S` for `exxdiv = ewald`). `DfJ`/`DfK` were NOT reused: both
//! recompute a molecular Coulomb metric from the aux `PreparedBasis`, and
//! `DfJ` inverts it with LU (`.inv()`), which the prototype showed is the
//! wrong solve for a near-singular periodic metric; reusing them would need
//! new `from_parts` constructors plus a `ThreeIndexSource::from_blocks` in
//! ferric-scf. Using ONE B for both J and K also keeps the fitted ERI a single
//! positive-semidefinite `BᵀB`.
//!
//! # Forces
//!
//! [`RsGdf::build_for_gradient`] keeps the metric eigen-data the analytic
//! forces need (the dropped eigenvectors and `J3` projected on them, for the
//! kept–dropped Loewner term); the derivative contractions live in
//! `deriv` and share this module's SR walks
//! (`Stage::sr_three_index_walk` / `sr_metric_walk`), pair images and G
//! sphere, so the force differentiates exactly the truncated energy built
//! here. Formulas: `deriv`'s module doc and FINDINGS "Iteration 18".
//!
//! Units: Bohr and Hartree; ω in Bohr⁻¹.

use crate::budget::{bytes_of, Ledger};
use crate::dense_aft::ExxDiv;
use crate::ewald::madelung_constant;
use crate::hcore::{gvector_list_bytes, half_gvectors, G_CHUNK_BYTES};
use crate::lattice::Cell;
use crate::pair_ft::pair_ft_chunked;
use crate::timing::{CallClock, PbcTimings, StageClock};
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::engine::Engine;
use ferric_integrals::md3c1e::{cart_components, e_table, ferric_cart2sph, prim_norm, MAX_L};
use ferric_integrals::operator::Operator;
use ferric_scf::fock::{JBuilder, KBuilder};
use ndarray::linalg::general_mat_mul;
use ndarray::{Array1, Array2, Array3, ArrayView1};
use ndarray_linalg::{Eigh, UPLO};
use num_complex::Complex64;
use std::f64::consts::PI;

pub(crate) mod deriv;
pub mod kpoint;
pub(crate) mod strain;

pub use deriv::RsGdfFitDiagnostics;

/// Default Ewald split for the RS-GDF build (Bohr⁻¹; the prototype's `w=1`).
pub const DEFAULT_RSGDF_OMEGA: f64 = 1.0;
/// Default per-term truncation target (the prototype's `prec`).
pub const DEFAULT_RSGDF_PRECISION: f64 = 1e-13;
/// Default absolute metric-eigenvalue cut (PySCF `LINEAR_DEP_THR`).
pub const DEFAULT_RSGDF_LINDEP: f64 = 1e-10;
/// Default hard cap for [`RsGdf::fitted_eri`] (test/diagnostic `nao⁴`).
pub const DEFAULT_FITTED_ERI_MAX_BYTES: usize = 512 << 20;

/// libint2 engine precision: effectively off — our own distance screen
/// decides what is computed.
const ENGINE_PRECISION: f64 = 1e-20;
/// Extra Bohr on every derived real-space radius (l > 0 polynomial factors).
const SR_MARGIN_BOHR: f64 = 2.0;
/// Hard cap on the integer box one lattice-sphere visit may enumerate.
const MAX_BOX_POINTS: u64 = 50_000_000;

/// How the G = 0 term is removed. Production is ALWAYS
/// [`G0Handling::Consistent`]; the others are deliberately BROKEN variants
/// that exist only as negative controls for the exactness anchor (an anchor
/// that cannot tell them apart from `Consistent` certifies nothing).
/// Prototype measurements on the anchor (FINDINGS "Iteration 2"): metric-only
/// max|ΔI| 5.3, neither 6.1e-2, consistent 1.6e-12.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum G0Handling {
    /// Remove `c0 q qᵀ` from J2 AND `c0 S qᵀ` from J3 (PySCF RSGDF).
    Consistent,
    /// MUTATION: remove it from the metric only (one-sided).
    MetricOnlyMutant,
    /// MUTATION: remove it from neither (fits a different, ω-dependent kernel).
    NeitherMutant,
}

/// Settings for [`RsGdf::build`].
#[derive(Debug, Clone, Copy)]
pub struct RsGdfConfig {
    /// Ewald split (Bohr⁻¹). Any ω > 0 gives the same B Bᵀ up to truncation.
    pub omega: f64,
    /// Per-term truncation target, in `(0, 1)`.
    pub precision: f64,
    /// Absolute metric-eigenvalue cut (eigenvalues `<= lindep` are dropped).
    pub lindep: f64,
    /// Exchange divergence treatment applied by [`RsGdfK`].
    pub exxdiv: ExxDiv,
    /// Per-triplet SR screen (`true`, production) or the global image radius
    /// for every triplet (`false`, the prototype's unscreened form).
    pub sr_screen: bool,
    /// Memory budget in bytes (`None` = ferric's unified budget).
    pub budget_bytes: Option<usize>,
    /// G = 0 handling — [`G0Handling::Consistent`] except in mutation tests.
    pub g0: G0Handling,
}

impl Default for RsGdfConfig {
    fn default() -> Self {
        Self {
            omega: DEFAULT_RSGDF_OMEGA,
            precision: DEFAULT_RSGDF_PRECISION,
            lindep: DEFAULT_RSGDF_LINDEP,
            exxdiv: ExxDiv::Ewald,
            sr_screen: true,
            budget_bytes: None,
            g0: G0Handling::Consistent,
        }
    }
}

impl RsGdfConfig {
    fn validate(&self) -> Result<(), FerricError> {
        if !(self.omega > 0.0) || !self.omega.is_finite() {
            return Err(FerricError::General(format!(
                "RsGdf: omega must be finite and > 0, got {}",
                self.omega
            )));
        }
        if !(f64::MIN_POSITIVE..1.0).contains(&self.precision) {
            return Err(FerricError::General(format!(
                "RsGdf: precision must lie in (0, 1), got {}",
                self.precision
            )));
        }
        if !(self.lindep >= 0.0) || !self.lindep.is_finite() {
            return Err(FerricError::General(format!(
                "RsGdf: lindep must be finite and >= 0, got {}",
                self.lindep
            )));
        }
        Ok(())
    }
}

/// What the build did (counts and conditioning; the only cost statements).
#[derive(Debug, Clone)]
pub struct RsGdfStats {
    /// Aux functions.
    pub naux: usize,
    /// Metric eigenvectors kept (`naux − n_dropped`) = rows of B.
    pub naux_kept: usize,
    /// Metric eigenvalues `<= lindep` dropped.
    pub n_dropped: usize,
    /// Smallest / largest metric eigenvalue.
    pub metric_eig_min: f64,
    pub metric_eig_max: f64,
    /// The ω and precision used.
    pub omega: f64,
    pub precision: f64,
    /// Pair images `L` visited.
    pub n_pair_images: usize,
    /// Shifted 3-centre shell triplets computed (SR).
    pub n_sr3_triplets: usize,
    /// Shifted 2-centre shell pairs computed (SR metric).
    pub n_sr2_pairs: usize,
    /// Half-sphere G vectors (LR) and the `pair_ft` chunks they used.
    pub n_g_half: usize,
    pub n_g_chunks: usize,
    /// max |J2 − J2ᵀ| and max |J3\[μν\] − J3\[νμ\]| before symmetrisation
    /// (truncation / image-set defects show up here).
    pub asym_j2: f64,
    pub asym_j3: f64,
    /// Resolved budget and the bytes reserved before the LR chunks.
    pub budget_bytes: usize,
    pub resident_bytes: usize,
}

/// The fitted three-index tensor and what the K builder needs.
#[derive(Debug, Clone)]
pub struct RsGdf {
    nao: usize,
    /// `(naux_kept, nao²)`, row k, column `μ·nao+ν`; each row symmetric in (μ,ν).
    b: Array2<f64>,
    s: Array2<f64>,
    madelung: f64,
    stats: RsGdfStats,
    /// `W = U_kept s_kept^{-1/2}`, `(naux, naux_kept)`: the aux-space map of
    /// B's rows (B = Wᵀ J3ᵀ). Retained for the dRPA driver's
    /// `RpaIntermediates::v_inv_sqrt` (eigenpotential back-transform only;
    /// never part of an energy). Counted by the build ledger's metric line.
    metric_inv_sqrt: Array2<f64>,
    /// Retained only by [`RsGdf::build_for_gradient`] (the forces' metric
    /// eigen-data); `None` on every energy-only build.
    grad: Option<MetricGradParts>,
    /// Build stage timings and counters ([`crate::timing`]).
    timings: PbcTimings,
    /// Accumulated [`RsGdfJ`] / [`RsGdfK`] build calls (every builder
    /// borrowing this B, i.e. every SCF stage run on it).
    j_clock: CallClock,
    k_clock: CallClock,
}

/// The pre-solve pieces of an RS-GDF build ([`RsGdf::build_with_fit_parts`]),
/// for per-pair domain-local fits in the PERIODIC metric: with the full aux
/// set as the domain and the same eig/`lindep` pseudo-inverse,
/// `A_i J2⁺ A_jᵀ` reproduces `B_iᵀ B_j` (the trivial-radius anchor,
/// `tests/pbc_lmp2.rs`).
#[derive(Debug, Clone)]
pub struct PeriodicFitParts {
    /// Symmetrised `(P|Q)'`, `(naux, naux)` — exactly the matrix the build's
    /// metric solve decomposed.
    pub j2: Array2<f64>,
    /// Symmetrised `(μν|P)'`, `(nao², naux)`, row `μ·nao+ν`.
    pub j3: Array2<f64>,
    /// Centre of every aux function (its shell's centre), Bohr, as placed in
    /// the cell (NOT wrapped: distances to it must be minimum-image).
    pub aux_centers: Vec<[f64; 3]>,
    /// The absolute eigenvalue cut the build used.
    pub lindep: f64,
}

/// Centre of every basis function of `prep` (its shell's centre).
fn aux_function_centers(prep: &PreparedBasis) -> Vec<[f64; 3]> {
    let mut out = vec![[0.0; 3]; prep.nbasis()];
    let offs = prep.shell_offsets();
    let dims = prep.shell_dims();
    for (sh, ls) in prep.located_shells().iter().enumerate() {
        for f in 0..dims[sh] {
            out[offs[sh] + f] = ls.center;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Shell data (ferric normalisation: prim_norm folded in, CCA order, ferric
// cart→sph for pure l >= 2 — the conventions of `pair_ft`).
// ---------------------------------------------------------------------------

struct GShell {
    l: usize,
    pure: bool,
    center: [f64; 3],
    exps: Vec<f64>,
    /// Contraction coefficients WITH `prim_norm(a, l)` folded in.
    coefs: Vec<f64>,
    nfun: usize,
    off: usize,
    amin: f64,
    amax: f64,
    /// `max_k |c_k| (π/α_k)^{3/2} (1 + α_k^{−1/2})^l` (aux screening).
    qbound: f64,
}

fn gshells(prep: &PreparedBasis, who: &str) -> Result<Vec<GShell>, FerricError> {
    let dims = prep.shell_dims();
    let offs = prep.shell_offsets();
    let mut out = Vec::with_capacity(prep.nshells());
    for (s, sh) in prep.located_shells().iter().enumerate() {
        if sh.l < 0 || sh.l as usize > MAX_L {
            return Err(FerricError::Basis(format!(
                "{who}: shell {s} has l={} (supported 0..={MAX_L})",
                sh.l
            )));
        }
        let l = sh.l as usize;
        if sh.pure && l == 1 {
            return Err(FerricError::Basis(format!(
                "{who}: shell {s} is a pure p shell (libint2 orders it y,z,x); unsupported"
            )));
        }
        if sh.exponents.is_empty() || sh.exponents.len() != sh.coefficients.len() {
            return Err(FerricError::Basis(format!(
                "{who}: shell {s} has {} exponents and {} coefficients",
                sh.exponents.len(),
                sh.coefficients.len()
            )));
        }
        if sh.exponents.iter().any(|&a| !(a > 0.0) || !a.is_finite()) {
            return Err(FerricError::Basis(format!(
                "{who}: shell {s} has a non-positive or non-finite exponent"
            )));
        }
        let pure = sh.pure && l >= 2;
        let ncart = (l + 1) * (l + 2) / 2;
        let nfun = if pure { 2 * l + 1 } else { ncart };
        if nfun != dims[s] {
            return Err(FerricError::Basis(format!(
                "{who}: shell {s} (l={l}, pure={pure}) has {nfun} functions but libint2 reports {}",
                dims[s]
            )));
        }
        let coefs: Vec<f64> = sh
            .exponents
            .iter()
            .zip(sh.coefficients)
            .map(|(&a, &c)| c * prim_norm(a, l))
            .collect();
        let qbound = sh
            .exponents
            .iter()
            .zip(&coefs)
            .map(|(&a, &c)| c.abs() * (PI / a).powf(1.5) * (1.0 + a.powf(-0.5)).powi(l as i32))
            .fold(0.0_f64, f64::max);
        out.push(GShell {
            l,
            pure,
            center: sh.center,
            exps: sh.exponents.to_vec(),
            coefs,
            nfun,
            off: offs[s],
            amin: sh.exponents.iter().copied().fold(f64::INFINITY, f64::min),
            amax: sh.exponents.iter().copied().fold(0.0_f64, f64::max),
            qbound,
        });
    }
    Ok(out)
}

fn dot3(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn norm3(a: &[f64; 3]) -> f64 {
    dot3(a, a).sqrt()
}

/// Distance from `x` to the segment `[a, b]`.
fn segment_distance(x: [f64; 3], a: [f64; 3], b: [f64; 3]) -> f64 {
    let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let ax = [x[0] - a[0], x[1] - a[1], x[2] - a[2]];
    let l2 = dot3(&ab, &ab);
    let t = if l2 > 0.0 {
        (dot3(&ax, &ab) / l2).clamp(0.0, 1.0)
    } else {
        0.0
    };
    norm3(&[ax[0] - t * ab[0], ax[1] - t * ab[1], ax[2] - t * ab[2]])
}

// ---------------------------------------------------------------------------
// One-centre aux FT
// ---------------------------------------------------------------------------

/// Fourier transform of every aux function, `X[P, g] = ∫ χ_P(r) e^{−iG_g·r} dr`,
/// shape `(nbasis, gvecs.len())`, in the AO order and normalisation libint2
/// uses for `aux` (the [`mod@crate::pair_ft`] conventions: `prim_norm` folded in,
/// CCA Cartesian order, `ferric_cart2sph` for pure l >= 2). Single-Gaussian
/// case of the pair FT: `FT[x^i e^{−a x²}](G) = (π/a)^{1/2} e^{−G²/4a}
/// Σ_t E_t^{i0}(a, 0) (−iG)^t` per direction, times the phase `e^{−iG·A}`.
/// `aux` may sit on any centres (atoms, ghosts, `SiteBasis` sites).
/// `X[:, g]` at `G = 0` is the charge `q_P = ∫ χ_P`.
pub fn aux_ft(aux: &PreparedBasis, gvecs: &[[f64; 3]]) -> Result<Array2<Complex64>, FerricError> {
    if gvecs.iter().flatten().any(|v| !v.is_finite()) {
        return Err(FerricError::General("aux_ft: non-finite G vector".into()));
    }
    let shells = gshells(aux, "aux_ft")?;
    Ok(aux_ft_shells(&shells, aux.nbasis(), gvecs))
}

fn aux_ft_shells(shells: &[GShell], naux: usize, gvecs: &[[f64; 3]]) -> Array2<Complex64> {
    let ng = gvecs.len();
    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    let mut out = Array2::<Complex64>::zeros((naux, ng));
    let mut ebuf = vec![0.0_f64; (MAX_L + 1) * (MAX_L + 1)];
    for sh in shells {
        let l = sh.l;
        let comps = cart_components(l);
        let ncart = comps.len();
        let mut cart = vec![zero; ncart * ng];
        for (&a, &c) in sh.exps.iter().zip(&sh.coefs) {
            // E^{i0}_t at index i*(l+1) + t (lb = 0).
            e_table(l, 0, a, 0.0, 0.0, &mut ebuf);
            let pref = c * (PI / a).powf(1.5);
            for (g, gv) in gvecs.iter().enumerate() {
                let mag = pref * (-dot3(gv, gv) / (4.0 * a)).exp();
                let ph = dot3(gv, &sh.center);
                // e^{−iG·A}
                let common = Complex64::new(mag * ph.cos(), -mag * ph.sin());
                let mut f = [[zero; MAX_L + 1]; 3];
                for (d, fd) in f.iter_mut().enumerate() {
                    let step = Complex64::new(0.0, -gv[d]);
                    for (i, fdi) in fd.iter_mut().enumerate().take(l + 1) {
                        let mut acc = zero;
                        let mut pw = one;
                        for t in 0..=i {
                            acc += pw * ebuf[i * (l + 1) + t];
                            pw *= step;
                        }
                        *fdi = acc;
                    }
                }
                for (u, lc) in comps.iter().enumerate() {
                    cart[u * ng + g] +=
                        common * f[0][lc[0] as usize] * f[1][lc[1] as usize] * f[2][lc[2] as usize];
                }
            }
        }
        if sh.pure {
            let c2s = ferric_cart2sph(l);
            let nf = sh.nfun;
            for j in 0..nf {
                for g in 0..ng {
                    let mut acc = zero;
                    for m in 0..ncart {
                        let cm = c2s[m * nf + j];
                        if cm != 0.0 {
                            acc += cart[m * ng + g] * cm;
                        }
                    }
                    out[[sh.off + j, g]] = acc;
                }
            }
        } else {
            for u in 0..ncart {
                for g in 0..ng {
                    out[[sh.off + u, g]] = cart[u * ng + g];
                }
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// G = 0 bookkeeping — the ONE place the convention lives.
// ---------------------------------------------------------------------------

/// Remove the erfc real-space sums' implicit G = 0 term
/// `c0 = π/(ω²Ω)`: `J2 −= c0 q qᵀ` and `J3[μν,P] −= c0 S_μν q_P`
/// (`s_flat` = S raveled `μ·nao+ν`). `mode` selects the production form or a
/// negative-control mutant ([`G0Handling`]).
pub fn subtract_g0(
    j2: &mut Array2<f64>,
    j3: &mut Array2<f64>,
    s_flat: &[f64],
    q: &[f64],
    c0: f64,
    mode: G0Handling,
) {
    let (do_metric, do_three) = match mode {
        G0Handling::Consistent => (true, true),
        G0Handling::MetricOnlyMutant => (true, false),
        G0Handling::NeitherMutant => (false, false),
    };
    if do_metric {
        for (p, qp) in q.iter().enumerate() {
            for (r, qr) in q.iter().enumerate() {
                j2[(p, r)] -= c0 * qp * qr;
            }
        }
    }
    if do_three {
        for (mn, smn) in s_flat.iter().enumerate() {
            for (p, qp) in q.iter().enumerate() {
                j3[(mn, p)] -= c0 * smn * qp;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Lattice-sphere walker
// ---------------------------------------------------------------------------

/// Enumerates lattice vectors `T` with `|T − x0| <= r` by an integer box:
/// `T·b_j = 2π n_j`, so `|n_j − x0·b_j/2π| <= r |b_j|/2π`.
///
/// For a strained cell ([`Cell::strained`]) the selection is made in the
/// REFERENCE frame (`x0` mapped by fractional coordinates, distances in the
/// reference lattice) and the selected `T` is built from the strained
/// lattice — the frozen-index convention of `crate::lattice`, so the SR walks
/// do not re-select images under strain.
struct LatticeWalker {
    /// Selection lattice (the reference's for a strained cell).
    a: [[f64; 3]; 3],
    b: [[f64; 3]; 3],
    /// The strained cell itself when its index sets are frozen (`None`: the
    /// selection lattice IS the output lattice, bit for bit as before).
    frozen: Option<Cell>,
}

impl LatticeWalker {
    fn new(cell: &Cell) -> Self {
        match cell.index_reference() {
            None => Self {
                a: *cell.lattice(),
                b: cell.reciprocal(),
                frozen: None,
            },
            Some(r) => Self {
                a: *r.lattice(),
                b: r.reciprocal(),
                frozen: Some(cell.clone()),
            },
        }
    }

    fn visit<F>(&self, x0: [f64; 3], r: f64, mut f: F) -> Result<(), FerricError>
    where
        F: FnMut([f64; 3]) -> Result<(), FerricError>,
    {
        if !(r >= 0.0) || !r.is_finite() || x0.iter().any(|v| !v.is_finite()) {
            return Err(FerricError::General(format!(
                "RsGdf lattice walk: bad centre {x0:?} / radius {r}"
            )));
        }
        let x0 = match &self.frozen {
            Some(c) => c.to_index_frame(x0),
            None => x0,
        };
        let tp = 2.0 * PI;
        let mut lo = [0i64; 3];
        let mut hi = [0i64; 3];
        let mut count: u64 = 1;
        for j in 0..3 {
            let fj = dot3(&x0, &self.b[j]) / tp;
            let wj = r * norm3(&self.b[j]) / tp;
            if !(wj < 1e6) || fj.abs() > 1e9 {
                return Err(FerricError::General(format!(
                    "RsGdf lattice walk: radius {r} needs {wj:.3e} cells along b_{j}"
                )));
            }
            lo[j] = (fj - wj).floor() as i64;
            hi[j] = (fj + wj).ceil() as i64;
            count = count.saturating_mul((hi[j] - lo[j] + 1) as u64);
        }
        if count > MAX_BOX_POINTS {
            return Err(FerricError::General(format!(
                "RsGdf lattice walk: radius {r} would enumerate {count} lattice triples \
                 (cap {MAX_BOX_POINTS}); raise precision or omega"
            )));
        }
        let a = &self.a;
        let r2 = r * r;
        for n0 in lo[0]..=hi[0] {
            for n1 in lo[1]..=hi[1] {
                for n2 in lo[2]..=hi[2] {
                    let (f0, f1, f2) = (n0 as f64, n1 as f64, n2 as f64);
                    let t = [
                        f0 * a[0][0] + f1 * a[1][0] + f2 * a[2][0],
                        f0 * a[0][1] + f1 * a[1][1] + f2 * a[2][1],
                        f0 * a[0][2] + f1 * a[1][2] + f2 * a[2][2],
                    ];
                    let d = [t[0] - x0[0], t[1] - x0[1], t[2] - x0[2]];
                    if dot3(&d, &d) <= r2 {
                        match &self.frozen {
                            None => f(t)?,
                            Some(c) => f(c.translation_from_index([n0, n1, n2]))?,
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// SR bounds
// ---------------------------------------------------------------------------

/// s-type charge bound of a (shell, shell) pair at separation² `r2`, with the
/// smallest and largest pair exponent.
fn pair_bound(a: &GShell, b: &GShell, r2: f64) -> (f64, f64, f64) {
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

/// Radius (Bohr, incl. margin) beyond which the erfc interaction of two
/// Gaussian distributions with charge bounds `qa`, `qb` and exponent ranges
/// `[amin, amax]`, `[bmin, bmax]` is below `thresh`; `None` if it is below
/// everywhere (module doc, "Screening").
#[allow(clippy::too_many_arguments)]
fn sr_radius(
    qa: f64,
    qb: f64,
    amin: f64,
    amax: f64,
    bmin: f64,
    bmax: f64,
    omega: f64,
    thresh: f64,
) -> Option<f64> {
    let mu = (amax * bmax / (amax + bmax)).sqrt();
    let nu = 1.0 / (1.0 / amin + 1.0 / bmin + 1.0 / (omega * omega)).sqrt();
    let pref = qa * qb * (1.0 + 2.0 * mu / PI.sqrt());
    if !(pref > thresh) {
        return None;
    }
    Some((pref / thresh).ln().sqrt() / nu + SR_MARGIN_BOHR)
}

fn max_abs_asym(m: &Array2<f64>) -> f64 {
    let mut w = 0.0_f64;
    for i in 0..m.nrows() {
        for j in 0..i {
            w = w.max((m[(i, j)] - m[(j, i)]).abs());
        }
    }
    w
}

// ---------------------------------------------------------------------------
// The build
// ---------------------------------------------------------------------------

/// Everything the SR/LR stages share.
struct Stage<'a> {
    cell: &'a Cell,
    obs: &'a PreparedBasis,
    aux: &'a PreparedBasis,
    obs_sh: Vec<GShell>,
    aux_sh: Vec<GShell>,
    omega: f64,
    thresh: f64,
    sr_screen: bool,
    walker: LatticeWalker,
}

impl Stage<'_> {
    fn radius(&self, qa: f64, qb: f64, a: (f64, f64), b: (f64, f64)) -> Option<f64> {
        sr_radius(qa, qb, a.0, a.1, b.0, b.1, self.omega, self.thresh)
    }

    /// SR metric `Σ_T (P_0 | Q_T)_erfc` (unsymmetrised) and the pair count.
    fn sr_metric(&self) -> Result<(Array2<f64>, usize), FerricError> {
        let naux = self.aux.nbasis();
        let mut j2 = Array2::<f64>::zeros((naux, naux));
        let count = self.sr_metric_each(|p, q, _t, blk| {
            for i in 0..p.nfun {
                for j in 0..q.nfun {
                    j2[(p.off + i, q.off + j)] += blk[i * q.nfun + j];
                }
            }
        })?;
        Ok((j2, count))
    }

    /// Every SR metric block `(P_0 | Q_T)_erfc` the screen keeps, in a fixed
    /// order: `sink(P shell, Q shell, T, block (nP, nQ))`. Returns the count.
    /// [`Stage::sr_metric`] (Gamma) and the residue-binned k-point form
    /// ([`kpoint`]) share this walk, so the Gamma sums are unchanged bit for
    /// bit.
    fn sr_metric_each<F>(&self, mut sink: F) -> Result<usize, FerricError>
    where
        F: FnMut(&GShell, &GShell, [f64; 3], &[f64]),
    {
        let mut eng = Engine::new_2center(Operator::erfc(self.omega), self.aux, ENGINE_PRECISION)?;
        self.sr_metric_walk(|ip, iq, t| {
            let blk = eng.compute_eri2_shifted(self.aux, ip, iq, t)?;
            sink(&self.aux_sh[ip], &self.aux_sh[iq], t, blk);
            Ok(())
        })
    }

    /// The SR metric walk alone: `visit(P shell, Q shell, T)` for every
    /// `(P_0 | Q_T)` the screen keeps, in the fixed order
    /// [`Stage::sr_metric_each`] sums them (shared with the gradient's
    /// derivative walk, [`deriv`]). Returns the count.
    fn sr_metric_walk<F>(&self, mut visit: F) -> Result<usize, FerricError>
    where
        F: FnMut(usize, usize, [f64; 3]) -> Result<(), FerricError>,
    {
        let mut count = 0usize;
        let global = if self.sr_screen {
            0.0
        } else {
            let mut r = 0.0_f64;
            for p in &self.aux_sh {
                for q in &self.aux_sh {
                    if let Some(x) =
                        self.radius(p.qbound, q.qbound, (p.amin, p.amax), (q.amin, q.amax))
                    {
                        r = r.max(x);
                    }
                }
            }
            r
        };
        for (ip, p) in self.aux_sh.iter().enumerate() {
            for (iq, q) in self.aux_sh.iter().enumerate() {
                let rad = if self.sr_screen {
                    match self.radius(p.qbound, q.qbound, (p.amin, p.amax), (q.amin, q.amax)) {
                        Some(r) => r,
                        None => continue,
                    }
                } else {
                    global
                };
                // |R_P − (R_Q + T)| <= rad.
                let x0 = [
                    p.center[0] - q.center[0],
                    p.center[1] - q.center[1],
                    p.center[2] - q.center[2],
                ];
                self.walker.visit(x0, rad, |t| {
                    count += 1;
                    visit(ip, iq, t)
                })?;
            }
        }
        Ok(count)
    }

    /// SR 3-index `Σ_{L,T} (μ_0 ν_L | P_T)_erfc` over the pair `images`,
    /// `(nao², naux)` (unsymmetrised), and the triplet count.
    fn sr_three_index(&self, images: &[[f64; 3]]) -> Result<(Array2<f64>, usize), FerricError> {
        let n = self.obs.nbasis();
        let naux = self.aux.nbasis();
        let mut j3 = Array2::<f64>::zeros((n * n, naux));
        let count = self.sr_three_index_each(images, |a, b, p, _l, _t, blk| {
            // block layout (nP, n1, n2)
            for pp in 0..p.nfun {
                for i in 0..a.nfun {
                    let row0 = (a.off + i) * n + b.off;
                    let src = (pp * a.nfun + i) * b.nfun;
                    for j in 0..b.nfun {
                        j3[(row0 + j, p.off + pp)] += blk[src + j];
                    }
                }
            }
        })?;
        Ok((j3, count))
    }

    /// Every SR 3-centre block `(μ_0 ν_L | P_T)_erfc` the screen keeps, in a
    /// fixed order: `sink(μ shell, ν shell, P shell, L, T, block (nP, nμ, nν))`.
    /// Returns the triplet count (as [`Stage::sr_metric_each`]: one walk
    /// shared by the Gamma sum and the k-point residue bins).
    fn sr_three_index_each<F>(&self, images: &[[f64; 3]], mut sink: F) -> Result<usize, FerricError>
    where
        F: FnMut(&GShell, &GShell, &GShell, [f64; 3], [f64; 3], &[f64]),
    {
        let mut eng = Engine::new_3center(
            Operator::erfc(self.omega),
            self.obs,
            self.aux,
            ENGINE_PRECISION,
        )?;
        self.sr_three_index_walk(images, |i1, i2, ip, l, t| {
            if let Some(blk) =
                eng.compute_eri3_shifted(self.obs, self.aux, ip, i1, i2, [t, [0.0; 3], l])?
            {
                sink(
                    &self.obs_sh[i1],
                    &self.obs_sh[i2],
                    &self.aux_sh[ip],
                    l,
                    t,
                    blk,
                );
            }
            Ok(())
        })
    }

    /// The SR 3-centre walk alone: `visit(μ shell, ν shell, P shell, L, T)`
    /// for every `(μ_0 ν_L | P_T)` the screen keeps, in the fixed order
    /// [`Stage::sr_three_index_each`] sums them (shared with the gradient's
    /// derivative walk, [`deriv`], so both see the same triplet set).
    /// Returns the triplet count.
    fn sr_three_index_walk<F>(
        &self,
        images: &[[f64; 3]],
        mut visit: F,
    ) -> Result<usize, FerricError>
    where
        F: FnMut(usize, usize, usize, [f64; 3], [f64; 3]) -> Result<(), FerricError>,
    {
        let mut count = 0usize;
        let global = if self.sr_screen {
            0.0
        } else {
            let mut r = 0.0_f64;
            for a in &self.obs_sh {
                for b in &self.obs_sh {
                    let (qab, pmin, pmax) = pair_bound(a, b, 0.0);
                    for p in &self.aux_sh {
                        if let Some(x) = self.radius(qab, p.qbound, (pmin, pmax), (p.amin, p.amax))
                        {
                            r = r.max(x);
                        }
                    }
                }
            }
            r
        };
        for l in images {
            for (i1, a) in self.obs_sh.iter().enumerate() {
                for (i2, b) in self.obs_sh.iter().enumerate() {
                    let bc = [b.center[0] + l[0], b.center[1] + l[1], b.center[2] + l[2]];
                    let ab = [
                        a.center[0] - bc[0],
                        a.center[1] - bc[1],
                        a.center[2] - bc[2],
                    ];
                    let r2 = dot3(&ab, &ab);
                    let (qab, pmin, pmax) = pair_bound(a, b, r2);
                    let mid = [
                        0.5 * (a.center[0] + bc[0]),
                        0.5 * (a.center[1] + bc[1]),
                        0.5 * (a.center[2] + bc[2]),
                    ];
                    let half = 0.5 * r2.sqrt();
                    for (ip, p) in self.aux_sh.iter().enumerate() {
                        let rad = if self.sr_screen {
                            match self.radius(qab, p.qbound, (pmin, pmax), (p.amin, p.amax)) {
                                Some(r) => r,
                                None => continue,
                            }
                        } else {
                            global
                        };
                        // Every aux image within `rad` of the segment lies
                        // within `rad + half` of its midpoint.
                        let x0 = [
                            mid[0] - p.center[0],
                            mid[1] - p.center[1],
                            mid[2] - p.center[2],
                        ];
                        self.walker.visit(x0, rad + half, |t| {
                            let x = [p.center[0] + t[0], p.center[1] + t[1], p.center[2] + t[2]];
                            if segment_distance(x, a.center, bc) > rad {
                                return Ok(());
                            }
                            count += 1;
                            visit(i1, i2, ip, *l, t)
                        })?;
                    }
                }
            }
        }
        Ok(count)
    }

    /// LR `(2/Ω) Σ_{G∈half} 4π/G² e^{−G²/4ω²} Re[conj(A) X]` added into `j2`
    /// and `j3`; `pair_ft` G-chunked within `chunk_budget`. Returns the chunk
    /// count and the `(wall, CPU)` seconds spent in the per-chunk sink (aux
    /// FT, packing and the four GEMMs); the rest of the call is the pair FT.
    fn lr_accumulate(
        &self,
        gv: &[[f64; 3]],
        j2: &mut Array2<f64>,
        j3: &mut Array2<f64>,
        chunk_budget: usize,
    ) -> Result<(usize, f64, Option<f64>), FerricError> {
        let n = self.obs.nbasis();
        let n2 = n * n;
        let naux = self.aux.nbasis();
        let vol = self.cell.volume();
        let omega = self.omega;
        // pr, pi (n² reals each) + X (naux complex) + xr, xi, xrw, xiw.
        let extra_per_g = n2
            .saturating_mul(16)
            .saturating_add(naux.saturating_mul(48))
            .saturating_add(64);
        let pair_ft_thresh = (0.01 * self.thresh).min(crate::pair_ft::DEFAULT_PAIR_FT_THRESH);
        let aux_sh = &self.aux_sh;
        let (mut sink_wall, mut sink_cpu) = (0.0_f64, None::<f64>);
        let sink =
            |_g0: usize, gs: &[[f64; 3]], pft: &Array3<Complex64>| -> Result<(), FerricError> {
                let clock = StageClock::start();
                let ng = gs.len();
                let x = aux_ft_shells(aux_sh, naux, gs);
                let mut pr = Array2::<f64>::zeros((n2, ng));
                let mut pim = Array2::<f64>::zeros((n2, ng));
                let mut xr = Array2::<f64>::zeros((naux, ng));
                let mut xi = Array2::<f64>::zeros((naux, ng));
                let mut xrw = Array2::<f64>::zeros((naux, ng));
                let mut xiw = Array2::<f64>::zeros((naux, ng));
                for (g, gvec) in gs.iter().enumerate() {
                    let g2 = dot3(gvec, gvec);
                    let w = 2.0 / vol * 4.0 * PI / g2 * (-g2 / (4.0 * omega * omega)).exp();
                    for m in 0..n {
                        for k in 0..n {
                            let z = pft[[m, k, g]];
                            pr[(m * n + k, g)] = w * z.re;
                            pim[(m * n + k, g)] = w * z.im;
                        }
                    }
                    for p in 0..naux {
                        let z = x[(p, g)];
                        xr[(p, g)] = z.re;
                        xi[(p, g)] = z.im;
                        xrw[(p, g)] = w * z.re;
                        xiw[(p, g)] = w * z.im;
                    }
                }
                // Re[conj(A) X] = A.re X.re + A.im X.im
                general_mat_mul(1.0, &pr, &xr.t(), 1.0, &mut *j3);
                general_mat_mul(1.0, &pim, &xi.t(), 1.0, &mut *j3);
                general_mat_mul(1.0, &xrw, &xr.t(), 1.0, &mut *j2);
                general_mat_mul(1.0, &xiw, &xi.t(), 1.0, &mut *j2);
                let (w, c) = clock.elapsed();
                sink_wall += w;
                if let Some(c) = c {
                    sink_cpu = Some(sink_cpu.unwrap_or(0.0) + c);
                }
                Ok(())
            };
        let n_chunks = pair_ft_chunked(
            self.cell,
            self.obs,
            gv,
            pair_ft_thresh,
            chunk_budget,
            extra_per_g,
            sink,
        )?;
        Ok((n_chunks, sink_wall, sink_cpu))
    }
}

/// Pair-image radius of the SR 3-centre sum: every pair whose charge bound
/// times the largest aux charge and potential factor reaches `precision`
/// (pair_ft's radius rule at the equivalent pair threshold). Shared by the
/// Gamma and k-point builds.
fn pair_image_radius(st: &Stage<'_>, precision: f64) -> f64 {
    let amin_orb = st
        .obs_sh
        .iter()
        .map(|s| s.amin)
        .fold(f64::INFINITY, f64::min);
    let pmax_orb = 2.0 * st.obs_sh.iter().map(|s| s.amax).fold(0.0_f64, f64::max);
    let qaux_max = st.aux_sh.iter().map(|p| p.qbound).fold(0.0_f64, f64::max);
    let vfac_max = st
        .aux_sh
        .iter()
        .map(|p| 1.0 + 2.0 * (pmax_orb * p.amax / (pmax_orb + p.amax)).sqrt() / PI.sqrt())
        .fold(1.0_f64, f64::max);
    let pair_thresh = 0.1 * precision / (qaux_max * vfac_max).max(1.0);
    (2.0 * (1e3 / pair_thresh).ln() / amin_orb).sqrt() + SR_MARGIN_BOHR
}

/// The pair images are generated from the cell's atoms, so the orbital basis
/// must sit on exactly those atoms (`pair_ft` re-checks this).
fn check_obs_on_cell(cell: &Cell, obs: &PreparedBasis) -> Result<(), FerricError> {
    let pos = cell.positions();
    if obs.atoms().len() != pos.len() {
        return Err(FerricError::General(format!(
            "RsGdf: orbital PreparedBasis has {} atoms but the cell has {}; build it from cell.mol()",
            obs.atoms().len(),
            pos.len()
        )));
    }
    for (k, (a, p)) in obs.atoms().iter().zip(&pos).enumerate() {
        let d = ((a.x - p[0]).powi(2) + (a.y - p[1]).powi(2) + (a.z - p[2]).powi(2)).sqrt();
        if d > 1e-10 {
            return Err(FerricError::General(format!(
                "RsGdf: orbital PreparedBasis atom {k} is {d:.3e} Bohr from the cell's atom {k}"
            )));
        }
    }
    Ok(())
}

/// Symmetrise `j3` over μ↔ν in place; returns the largest asymmetry seen.
fn symmetrize_pairs(j3: &mut Array2<f64>, n: usize) -> f64 {
    let naux = j3.ncols();
    let mut asym = 0.0_f64;
    for m in 0..n {
        for k in 0..m {
            let (r1, r2) = (m * n + k, k * n + m);
            for p in 0..naux {
                let (x, y) = (j3[(r1, p)], j3[(r2, p)]);
                asym = asym.max((x - y).abs());
                let avg = 0.5 * (x + y);
                j3[(r1, p)] = avg;
                j3[(r2, p)] = avg;
            }
        }
    }
    asym
}

/// What the RS-GDF FORCES need from the metric solve beyond `B` and `W`
/// ([`RsGdf::build_for_gradient`]; FINDINGS "Iteration 18"): the kept and
/// dropped eigenvalues, the dropped eigenvectors `U_d` and `J3` projected on
/// them, `J_d = U_dᵀ J3ᵀ` — the pieces of the kept–dropped (Loewner /
/// Daleckii–Krein) block of the metric derivative. Energies never read it.
#[derive(Debug, Clone)]
pub(crate) struct MetricGradParts {
    /// Kept eigenvalues, in the column order of `W` (`> lindep`).
    pub(crate) s_kept: Vec<f64>,
    /// Dropped eigenvalues (`<= lindep`), in the column order of `u_drop`.
    pub(crate) s_drop: Vec<f64>,
    /// `(naux, n_dropped)` dropped eigenvectors.
    pub(crate) u_drop: Array2<f64>,
    /// `(n_dropped, nao²)` = `U_dᵀ J3ᵀ` (symmetrised J3, G = 0 removed).
    pub(crate) jd: Array2<f64>,
    /// The build's SR screen mode (the derivative walk must visit the same
    /// triplets).
    pub(crate) sr_screen: bool,
    pub(crate) lindep: f64,
}

/// `(s_kept, s_drop, U_d, J_d)` of [`MetricGradParts`].
type DroppedParts = (Vec<f64>, Vec<f64>, Array2<f64>, Array2<f64>);

/// `(B, eigenvalues ascending, W, gradient parts)` of [`fit_with_metric`].
type MetricFit = (Array2<f64>, Vec<f64>, Array2<f64>, Option<DroppedParts>);

/// The dropped-subspace pieces of [`MetricGradParts`] from `J2 = U s Uᵀ`
/// (`evals`, `evecs`) and the kept indices: `(s_kept, s_drop, U_d, J_d)`,
/// reserved on `ledger` once the dropped count is known. `None` (energy-only
/// build, no ledger) forms nothing.
fn dropped_subspace_parts(
    evals: &Array1<f64>,
    evecs: &Array2<f64>,
    keep: &[usize],
    j3: &Array2<f64>,
    lindep: f64,
    ledger: Option<&mut Ledger>,
) -> Result<Option<DroppedParts>, FerricError> {
    let Some(ledger) = ledger else {
        return Ok(None);
    };
    let naux = evecs.nrows();
    let drop_idx: Vec<usize> = (0..naux).filter(|&k| !(evals[k] > lindep)).collect();
    let nd = drop_idx.len();
    ledger.reserve(
        &format!(
            "RsGdf gradient parts: dropped eigenvectors + projected J3 \
             (n_dropped = {nd}, naux = {naux}, nao² = {})",
            j3.nrows()
        ),
        bytes_of(
            (nd as u64).saturating_mul((naux as u64).saturating_add(j3.nrows() as u64)),
            8,
        ),
    )?;
    let mut u_drop = Array2::<f64>::zeros((naux, nd));
    for (c, &k) in drop_idx.iter().enumerate() {
        for r in 0..naux {
            u_drop[(r, c)] = evecs[(r, k)];
        }
    }
    // J_d = U_dᵀ J3ᵀ = (J3 U_d)ᵀ, (nd, n²)
    let jd = j3.dot(&u_drop).t().as_standard_layout().into_owned();
    let s_kept: Vec<f64> = keep.iter().map(|&k| evals[k]).collect();
    let s_drop: Vec<f64> = drop_idx.iter().map(|&k| evals[k]).collect();
    Ok(Some((s_kept, s_drop, u_drop, jd)))
}

/// `B = (J3 W)ᵀ`, `W = U_keep s_keep^{-1/2}` from `J2 = U s Uᵀ` with the
/// eigenvalues `<= lindep` dropped. Returns `(B, eigenvalues ascending, W,
/// gradient parts)` (`W` is `(naux, naux_kept)`; B is computed from it
/// exactly as before). With `grad_ledger = Some(..)` the dropped-subspace
/// pieces of [`MetricGradParts`] are also formed (reserved on that ledger
/// once the dropped count is known); B is bitwise the same either way.
fn fit_with_metric(
    j2: &Array2<f64>,
    j3: Array2<f64>,
    lindep: f64,
    grad_ledger: Option<&mut Ledger>,
    timings: &mut PbcTimings,
) -> Result<MetricFit, FerricError> {
    let naux = j2.nrows();
    let clock = StageClock::start();
    let (evals, evecs) = j2
        .eigh(UPLO::Upper)
        .map_err(|e| FerricError::Lapack(format!("RsGdf metric eigh: {e}")))?;
    timings.stop("rsgdf metric eigh", &clock);
    let clock = StageClock::start();
    let keep: Vec<usize> = (0..naux).filter(|&k| evals[k] > lindep).collect();
    if keep.is_empty() {
        return Err(FerricError::General(format!(
            "RsGdf: every metric eigenvalue is <= lindep {lindep:e} (max {:e}); the aux basis \
             has no fitting power in the G = 0-dropped kernel",
            evals[naux - 1]
        )));
    }
    let mut w = Array2::<f64>::zeros((naux, keep.len()));
    for (c, &k) in keep.iter().enumerate() {
        let sc = 1.0 / evals[k].sqrt();
        for r in 0..naux {
            w[(r, c)] = evecs[(r, k)] * sc;
        }
    }
    let bt = j3.dot(&w); // (n², nkeep)
    let grad = dropped_subspace_parts(&evals, &evecs, &keep, &j3, lindep, grad_ledger)?;
    drop(j3);
    let b = bt.t().as_standard_layout().into_owned(); // (nkeep, n²)
    timings.stop("rsgdf B = (J3 W)^T", &clock);
    Ok((b, evals.to_vec(), w, grad))
}

/// libint2 (as built for ferric) assumes solid-harmonic shells for l > 1 in
/// 2- and 3-centre two-body integrals and ABORTS the process on a Cartesian
/// one (`engine.impl.h`, `ERI2_PURE_SH`). Refuse such an aux basis up front
/// with a typed error instead.
pub(crate) fn require_pure_aux(aux: &PreparedBasis, who: &str) -> Result<(), FerricError> {
    for (i, sh) in aux.located_shells().iter().enumerate() {
        if sh.l > 1 && !sh.pure {
            return Err(FerricError::Basis(format!(
                "{who}: aux shell {i} has l = {} and is Cartesian; libint2's 2/3-centre \
                 integrals require solid-harmonic (pure) shells above l = 1",
                sh.l
            )));
        }
    }
    Ok(())
}

impl RsGdf {
    /// Build B for `cell` with orbital basis `obs` (built from `cell.mol()`)
    /// and aux basis `aux` (any centres: `PreparedBasis::new(cell.mol(),
    /// aux_set)` for atom-centred aux, or a `SiteBasis`). `s` is the lattice
    /// overlap (e.g. `PeriodicHcore::s`), used by the G = 0 term and the
    /// Madelung shift. See the module doc for the method.
    pub fn build(
        cell: &Cell,
        obs: &PreparedBasis,
        aux: &PreparedBasis,
        s: &Array2<f64>,
        cfg: &RsGdfConfig,
    ) -> Result<Self, FerricError> {
        Self::build_impl(cell, obs, aux, s, cfg, false, false).map(|(gdf, _)| gdf)
    }

    /// [`RsGdf::build`] that also retains what the analytic RS-GDF forces
    /// need from the metric solve ([`crate::grad`]'s `*_rsgdf` entry points;
    /// FINDINGS "Iteration 18"): the kept/dropped eigenvalues, the dropped
    /// eigenvectors `U_d` and `U_dᵀ J3ᵀ` (`n_dropped × nao²`, the kept–dropped
    /// Loewner term). B, W, the stats and therefore every energy are bitwise
    /// those of [`RsGdf::build`] (same code path; the extra pieces are copies
    /// taken inside the metric solve, reserved on the build ledger once the
    /// dropped count is known).
    pub fn build_for_gradient(
        cell: &Cell,
        obs: &PreparedBasis,
        aux: &PreparedBasis,
        s: &Array2<f64>,
        cfg: &RsGdfConfig,
    ) -> Result<Self, FerricError> {
        Self::build_impl(cell, obs, aux, s, cfg, false, true).map(|(gdf, _)| gdf)
    }

    /// Whether this B carries the gradient parts
    /// ([`RsGdf::build_for_gradient`]).
    pub fn has_gradient_parts(&self) -> bool {
        self.grad.is_some()
    }

    pub(crate) fn gradient_parts(&self) -> Option<&MetricGradParts> {
        self.grad.as_ref()
    }

    /// [`RsGdf::build`] that ALSO returns the symmetrised periodic metric
    /// `J2`, the 3-index `J3` (both in the G = 0-dropped kernel, before the
    /// metric solve) and the aux function centres — what a per-pair
    /// domain-local fit in the PERIODIC metric needs
    /// ([`crate::lmp2`]). The returned `RsGdf` is bitwise the one
    /// [`RsGdf::build`] gives (same code path; the parts are copies taken
    /// just before the metric solve). The extra `8 naux (naux + nao²)`
    /// bytes are reserved on the build ledger.
    pub fn build_with_fit_parts(
        cell: &Cell,
        obs: &PreparedBasis,
        aux: &PreparedBasis,
        s: &Array2<f64>,
        cfg: &RsGdfConfig,
    ) -> Result<(Self, PeriodicFitParts), FerricError> {
        let (gdf, parts) = Self::build_impl(cell, obs, aux, s, cfg, true, false)?;
        let parts = parts.ok_or_else(|| {
            FerricError::General("RsGdf::build_with_fit_parts: parts not retained".into())
        })?;
        Ok((gdf, parts))
    }

    fn build_impl(
        cell: &Cell,
        obs: &PreparedBasis,
        aux: &PreparedBasis,
        s: &Array2<f64>,
        cfg: &RsGdfConfig,
        retain_parts: bool,
        retain_grad: bool,
    ) -> Result<(Self, Option<PeriodicFitParts>), FerricError> {
        cfg.validate()?;
        require_pure_aux(aux, "RsGdf")?;
        let n = obs.nbasis();
        let n2 = n * n;
        let naux = aux.nbasis();
        if s.dim() != (n, n) {
            return Err(FerricError::General(format!(
                "RsGdf: overlap has shape {:?}, expected ({n}, {n})",
                s.dim()
            )));
        }
        if naux == 0 || n == 0 {
            return Err(FerricError::General(format!(
                "RsGdf: empty basis (nao = {n}, naux = {naux})"
            )));
        }
        check_obs_on_cell(cell, obs)?;
        let total = StageClock::start();
        let mut timings = PbcTimings::default();
        let clock = StageClock::start();
        let st = Stage {
            cell,
            obs,
            aux,
            obs_sh: gshells(obs, "RsGdf orbital basis")?,
            aux_sh: gshells(aux, "RsGdf aux basis")?,
            omega: cfg.omega,
            thresh: cfg.precision,
            sr_screen: cfg.sr_screen,
            walker: LatticeWalker::new(cell),
        };

        let mut ledger = Ledger::new(crate::budget::resolve(cfg.budget_bytes));
        ledger.reserve(
            &format!("RsGdf n×n matrices (n = {n}: S copy + Madelung S D S temporaries)"),
            bytes_of(n2 as u64, 8 * 4),
        )?;
        ledger.reserve(
            &format!("RsGdf 3-index J3 + B + transpose temporary (naux = {naux}, nao = {n})"),
            bytes_of((n2 as u64).saturating_mul(naux as u64), 8 * 3),
        )?;
        ledger.reserve(
            &format!("RsGdf aux metric + eigenvectors + scaled copy (naux = {naux})"),
            bytes_of((naux as u64).saturating_mul(naux as u64), 8 * 4),
        )?;
        if retain_parts {
            ledger.reserve(
                &format!("RsGdf retained fit parts: J2 + J3 copies (naux = {naux}, nao = {n})"),
                bytes_of(
                    (naux as u64).saturating_mul((naux as u64).saturating_add(n2 as u64)),
                    8,
                ),
            )?;
        }

        // Pair images: every pair whose charge bound times the largest aux
        // charge and potential factor reaches `precision` (pair_ft's radius
        // rule at the equivalent pair threshold).
        let rpair = pair_image_radius(&st, cfg.precision);
        ledger.reserve(
            &format!("RsGdf pair-image list (r_pair = {rpair:.2} Bohr)"),
            bytes_of(cell.translation_count_bound(rpair)?, 24),
        )?;
        let images = cell.translations(rpair)?;

        let gcut = 2.0 * cfg.omega * (1.0 / cfg.precision).ln().sqrt();
        ledger.reserve(
            &format!("RsGdf LR G list (|G| <= {gcut:.3})"),
            gvector_list_bytes(cell, gcut)?,
        )?;
        let gv = half_gvectors(cell, gcut)?;
        let resident_bytes = ledger.resident();
        let chunk_budget = ledger.remaining().min(G_CHUNK_BYTES);
        timings.stop("rsgdf setup (shells, pair images, G list)", &clock);

        // --- SR (real space), LR (G ≠ 0), then the G = 0 term.
        let clock = StageClock::start();
        let (mut j2, n_sr2) = st.sr_metric()?;
        timings.stop("rsgdf SR metric (2-centre)", &clock);
        let clock = StageClock::start();
        let (mut j3, n_sr3) = st.sr_three_index(&images)?;
        timings.stop("rsgdf SR 3-centre", &clock);
        let clock = StageClock::start();
        let (n_g_chunks, sink_wall, sink_cpu) =
            st.lr_accumulate(&gv, &mut j2, &mut j3, chunk_budget)?;
        let (lr_wall, lr_cpu) = clock.elapsed();
        timings.add(
            "rsgdf LR pair FT",
            (lr_wall - sink_wall).max(0.0),
            match (lr_cpu, sink_cpu) {
                (Some(a), Some(b)) => Some((a - b).max(0.0)),
                (a, None) => a,
                (None, Some(_)) => None,
            },
            1,
        );
        timings.add(
            "rsgdf LR aux FT + GEMM",
            sink_wall,
            sink_cpu.or(lr_cpu.map(|_| 0.0)),
            n_g_chunks as u64,
        );
        let clock = StageClock::start();
        let q: Vec<f64> = aux_ft_shells(&st.aux_sh, naux, &[[0.0; 3]])
            .column(0)
            .iter()
            .map(|z| z.re)
            .collect();
        let s_std = s.as_standard_layout();
        let s_flat = s_std.as_slice().expect("standard layout");
        let c0 = PI / (cfg.omega * cfg.omega * cell.volume());
        subtract_g0(&mut j2, &mut j3, s_flat, &q, c0, cfg.g0);

        let asym_j2 = max_abs_asym(&j2);
        let j2 = 0.5 * (&j2 + &j2.t());
        let asym_j3 = symmetrize_pairs(&mut j3, n);
        timings.stop("rsgdf G=0 + symmetrise", &clock);
        let parts = if retain_parts {
            Some(PeriodicFitParts {
                j2: j2.clone(),
                j3: j3.clone(),
                aux_centers: aux_function_centers(aux),
                lindep: cfg.lindep,
            })
        } else {
            None
        };
        let (b, evals, metric_inv_sqrt, grad_eig) = fit_with_metric(
            &j2,
            j3,
            cfg.lindep,
            retain_grad.then_some(&mut ledger),
            &mut timings,
        )?;
        let nkeep = b.nrows();
        let grad = grad_eig.map(|(s_kept, s_drop, u_drop, jd)| MetricGradParts {
            s_kept,
            s_drop,
            u_drop,
            jd,
            sr_screen: cfg.sr_screen,
            lindep: cfg.lindep,
        });

        let madelung = match cfg.exxdiv {
            ExxDiv::None => 0.0,
            ExxDiv::Ewald => madelung_constant(cell)?,
        };
        ferric_core::memory::warn_if_rss_over("ferric-pbc RsGdf", ledger.budget(), 1.1);
        let stats = RsGdfStats {
            naux,
            naux_kept: nkeep,
            n_dropped: naux - nkeep,
            metric_eig_min: evals[0],
            metric_eig_max: evals[naux - 1],
            omega: cfg.omega,
            precision: cfg.precision,
            n_pair_images: images.len(),
            n_sr3_triplets: n_sr3,
            n_sr2_pairs: n_sr2,
            n_g_half: gv.len(),
            n_g_chunks,
            asym_j2,
            asym_j3,
            budget_bytes: ledger.budget(),
            resident_bytes,
        };
        for (name, v) in [
            ("rsgdf pair images", stats.n_pair_images),
            ("rsgdf SR2 pairs", stats.n_sr2_pairs),
            ("rsgdf SR3 triplets", stats.n_sr3_triplets),
            ("rsgdf LR half-G", stats.n_g_half),
            ("rsgdf LR chunks", stats.n_g_chunks),
            ("rsgdf naux", stats.naux),
            ("rsgdf naux kept", stats.naux_kept),
            ("rsgdf aux dropped", stats.n_dropped),
        ] {
            timings.set_counter(name, v as u64);
        }
        timings.finish(&total);
        Ok((
            Self {
                nao: n,
                b,
                s: s.clone(),
                madelung,
                stats,
                metric_inv_sqrt,
                grad,
                timings,
                j_clock: CallClock::default(),
                k_clock: CallClock::default(),
            },
            parts,
        ))
    }

    /// The lattice overlap `S` B was built with (the SCF's `S`).
    pub fn overlap(&self) -> &Array2<f64> {
        &self.s
    }

    /// The same B with another exchange-divergence treatment (B does not
    /// depend on it; only the K builder's `v_M` changes). `cell` must be the
    /// cell B was built for.
    pub fn with_exxdiv(mut self, cell: &Cell, exxdiv: ExxDiv) -> Result<Self, FerricError> {
        self.madelung = match exxdiv {
            ExxDiv::None => 0.0,
            ExxDiv::Ewald => madelung_constant(cell)?,
        };
        Ok(self)
    }

    /// `(naux_kept, nao²)` fitted tensor, row k, column `μ·nao+ν`.
    pub fn b(&self) -> &Array2<f64> {
        &self.b
    }

    /// `U_kept s_kept^{-1/2}`, `(naux, naux_kept)`, from the metric solve
    /// (`B = (J3 W)ᵀ` with this `W`).
    pub fn metric_inv_sqrt(&self) -> &Array2<f64> {
        &self.metric_inv_sqrt
    }

    /// Build statistics (counts, conditioning, dropped eigenvalues).
    pub fn stats(&self) -> &RsGdfStats {
        &self.stats
    }

    /// Build stage timings and counters, plus the accumulated J and K build
    /// calls of every builder borrowed from this B so far (`"scf J (rsgdf)"`,
    /// `"scf K (rsgdf)"`). `wall_s` is the BUILD total; the J/K stages lie
    /// outside it (they run later, in the SCF).
    pub fn timings(&self) -> PbcTimings {
        let mut t = self.timings.clone();
        t.add_stage(&self.j_clock.timing("scf J (rsgdf)"));
        t.add_stage(&self.k_clock.timing("scf K (rsgdf)"));
        t
    }

    /// `v_M` applied in K (0 for [`ExxDiv::None`]).
    pub fn madelung(&self) -> f64 {
        self.madelung
    }

    /// TEST/DIAGNOSTIC: the fitted ERI `BᵀB`, `(nao², nao²)` with the
    /// [`crate::dense_aft::DenseAftEri::eri`] layout. Errors above `max_bytes`.
    pub fn fitted_eri(&self, max_bytes: usize) -> Result<Array2<f64>, FerricError> {
        let n2 = self.nao * self.nao;
        let bytes = 8u128 * (n2 as u128) * (n2 as u128);
        if bytes > max_bytes as u128 {
            return Err(FerricError::General(format!(
                "RsGdf::fitted_eri: nao^4 tensor needs {bytes} bytes (nao = {}) > cap {max_bytes} bytes",
                self.nao
            )));
        }
        Ok(self.b.t().dot(&self.b))
    }

    /// J builder borrowing this B.
    pub fn j_builder(&self) -> RsGdfJ<'_> {
        RsGdfJ { gdf: self }
    }

    /// K builder (with the Madelung term) borrowing this B.
    pub fn k_builder(&self) -> RsGdfK<'_> {
        RsGdfK {
            gdf: self,
            madelung: self.madelung,
        }
    }

    /// K builder with an explicit Madelung shift `v_M` (0 = `exxdiv=None`),
    /// ignoring B's own [`RsGdf::madelung`]. Lets one B serve both stages of
    /// the staged Gamma UHF (`crate::uhf::gamma_uhf`) without cloning it.
    pub fn k_builder_with_madelung(&self, madelung: f64) -> RsGdfK<'_> {
        RsGdfK {
            gdf: self,
            madelung,
        }
    }

    fn check(&self, d: &Array2<f64>, out: &Array2<f64>, who: &str) -> Result<(), FerricError> {
        let n = self.nao;
        if d.dim() != (n, n) || out.dim() != (n, n) {
            return Err(FerricError::General(format!(
                "{who}: density {:?} / output {:?}, expected ({n}, {n})",
                d.dim(),
                out.dim()
            )));
        }
        Ok(())
    }
}

/// [`JBuilder`] over an [`RsGdf`]: `J = Σ_k B_k (B_k · D)`. Overwrites `j`.
pub struct RsGdfJ<'a> {
    gdf: &'a RsGdf,
}

impl JBuilder for RsGdfJ<'_> {
    fn build(&mut self, d: &Array2<f64>, j: &mut Array2<f64>) -> Result<usize, FerricError> {
        let gdf = self.gdf;
        gdf.j_clock.time(|| self.build_untimed(d, j))
    }

    fn reset(&mut self) {}
}

impl RsGdfJ<'_> {
    fn build_untimed(
        &mut self,
        d: &Array2<f64>,
        j: &mut Array2<f64>,
    ) -> Result<usize, FerricError> {
        self.gdf.check(d, j, "RsGdfJ")?;
        let n = self.gdf.nao;
        let d_std = d.as_standard_layout();
        let dflat = ArrayView1::from(d_std.as_slice().expect("standard layout"));
        let c = self.gdf.b.dot(&dflat); // (nkeep)
        let jflat = self.gdf.b.t().dot(&c); // (n²)
        for m in 0..n {
            for k in 0..n {
                j[(m, k)] = jflat[m * n + k];
            }
        }
        Ok(self.gdf.b.len())
    }
}

/// [`KBuilder`] over an [`RsGdf`]: `K = Σ_k B_k D B_kᵀ + v_M S D S`.
/// Overwrites `k`.
///
/// Does NOT override [`KBuilder::build_from_occ`] (audited for Stage 4): the
/// trait default reconstructs `D = C Cᵀ` and calls [`KBuilder::build`], so
/// the Madelung term is kept. A future DfK-style half-transform override
/// (`Σ_k (B_k C)(B_k C)ᵀ`) MUST also add `v_M S C Cᵀ S`, or it silently drops
/// the Madelung term on the occupied path only (guarded by
/// `tests/pbc_uhf.rs::k_builders_keep_madelung_on_the_occupied_path`).
pub struct RsGdfK<'a> {
    gdf: &'a RsGdf,
    madelung: f64,
}

impl KBuilder for RsGdfK<'_> {
    fn build(&mut self, d: &Array2<f64>, k: &mut Array2<f64>) -> Result<usize, FerricError> {
        let gdf = self.gdf;
        gdf.k_clock.time(|| self.build_untimed(d, k))
    }

    fn update_density(&mut self, _d: &Array2<f64>) {}

    fn reset(&mut self) {}
}

impl RsGdfK<'_> {
    fn build_untimed(
        &mut self,
        d: &Array2<f64>,
        k: &mut Array2<f64>,
    ) -> Result<usize, FerricError> {
        self.gdf.check(d, k, "RsGdfK")?;
        let n = self.gdf.nao;
        k.fill(0.0);
        for row in self.gdf.b.rows() {
            let bk = row
                .into_shape_with_order((n, n))
                .map_err(|e| FerricError::General(format!("RsGdfK: B row reshape: {e}")))?;
            let tmp = bk.dot(d);
            general_mat_mul(1.0, &tmp, &bk.t(), 1.0, &mut *k);
        }
        if self.madelung != 0.0 {
            let sds = self.gdf.s.dot(d).dot(&self.gdf.s);
            k.scaled_add(self.madelung, &sds);
        }
        Ok(self.gdf.b.len() * n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferric_core::basis::{BasisSet, Shell};
    use ferric_core::mol::{Atom, Molecule};
    use std::collections::HashMap;

    fn one_shell_prep(l: i32, pure: bool, a: f64, at: [f64; 3]) -> PreparedBasis {
        let mut shells = HashMap::new();
        shells.insert(
            1,
            vec![Shell {
                l,
                pure,
                exponents: vec![a],
                coefficients: vec![1.0],
            }],
        );
        let bs = BasisSet {
            name: "one".into(),
            shells,
            ecps: HashMap::new(),
        };
        let mol = Molecule {
            atoms: vec![Atom {
                symbol: "H".into(),
                z: 1,
                x: at[0],
                y: at[1],
                zpos: at[2],
                ghost: false,
                n_core_ecp: 0,
            }],
            charge: 0,
            multiplicity: 2,
        };
        PreparedBasis::new(&mol, &bs).unwrap()
    }

    #[test]
    fn aux_ft_charge_of_s_is_the_closed_form_and_pure_d_is_neutral() {
        let a = 0.37;
        let s = one_shell_prep(0, false, a, [0.3, -0.2, 1.1]);
        let x = aux_ft(&s, &[[0.0; 3]]).unwrap();
        // unit-self-overlap s: ∫g = 2^{3/4} π^{3/4} a^{-3/4}
        let want = 2f64.powf(0.75) * PI.powf(0.75) * a.powf(-0.75);
        assert!(
            (x[(0, 0)].re - want).abs() < 1e-13 * want,
            "{} vs {want}",
            x[(0, 0)].re
        );
        assert!(x[(0, 0)].im.abs() < 1e-15);
        let d = one_shell_prep(2, true, a, [0.3, -0.2, 1.1]);
        let xd = aux_ft(&d, &[[0.0; 3]]).unwrap();
        assert_eq!(xd.nrows(), 5);
        for p in 0..5 {
            assert!(
                xd[(p, 0)].norm() < 1e-13,
                "pure d charge {p}: {}",
                xd[(p, 0)]
            );
        }
        // Cartesian d: xx/yy/zz carry equal nonzero charge, off-diagonal none.
        let dc = one_shell_prep(2, false, a, [0.0; 3]);
        let xc = aux_ft(&dc, &[[0.0; 3]]).unwrap();
        let q: Vec<f64> = (0..6).map(|p| xc[(p, 0)].re).collect();
        // CCA order: xx, xy, xz, yy, yz, zz
        assert!(q[0] > 0.1 && (q[0] - q[3]).abs() < 1e-13 && (q[0] - q[5]).abs() < 1e-13);
        assert!(q[1].abs() < 1e-15 && q[2].abs() < 1e-15 && q[4].abs() < 1e-15);
    }

    #[test]
    fn aux_ft_phase_is_minus_i_g_dot_a() {
        // Translating the function by A multiplies X(G) by e^{−iG·A}.
        let a = 0.8;
        let g = [[0.7, -0.3, 1.2]];
        let at = [0.4, 1.1, -0.6];
        let x0 = aux_ft(&one_shell_prep(1, false, a, [0.0; 3]), &g).unwrap();
        let x1 = aux_ft(&one_shell_prep(1, false, a, at), &g).unwrap();
        let ph = dot3(&g[0], &at);
        let f = Complex64::new(ph.cos(), -ph.sin());
        for p in 0..3 {
            assert!((x1[(p, 0)] - x0[(p, 0)] * f).norm() < 1e-14);
            assert!(x0[(p, 0)].norm() > 1e-3, "vacuous");
        }
    }

    #[test]
    fn lattice_walker_is_complete() {
        let mol = Molecule {
            atoms: vec![Atom {
                symbol: "H".into(),
                z: 1,
                x: 0.0,
                y: 0.0,
                zpos: 0.0,
                ghost: false,
                n_core_ecp: 0,
            }],
            charge: 0,
            multiplicity: 2,
        };
        let cell = Cell::new(mol, [[4.6, 0.0, 0.0], [0.9, 4.3, 0.0], [0.5, 0.7, 4.8]]).unwrap();
        let w = LatticeWalker::new(&cell);
        let x0 = [1.7, -2.3, 0.4];
        let r = 9.5;
        let mut got = 0usize;
        w.visit(x0, r, |_| {
            got += 1;
            Ok(())
        })
        .unwrap();
        let a = cell.lattice();
        let mut brute = 0usize;
        for n0 in -10i32..=10 {
            for n1 in -10i32..=10 {
                for n2 in -10i32..=10 {
                    let t: Vec<f64> = (0..3)
                        .map(|k| n0 as f64 * a[0][k] + n1 as f64 * a[1][k] + n2 as f64 * a[2][k])
                        .collect();
                    let d =
                        ((t[0] - x0[0]).powi(2) + (t[1] - x0[1]).powi(2) + (t[2] - x0[2]).powi(2))
                            .sqrt();
                    if d <= r {
                        brute += 1;
                    }
                }
            }
        }
        assert!(brute > 20);
        assert_eq!(got, brute);
    }

    #[test]
    fn g0_mutants_differ_only_where_they_should() {
        let mut j2 = Array2::<f64>::zeros((2, 2));
        let mut j3 = Array2::<f64>::zeros((1, 2));
        subtract_g0(
            &mut j2,
            &mut j3,
            &[2.0],
            &[1.0, 3.0],
            0.5,
            G0Handling::Consistent,
        );
        assert_eq!(j2[(0, 1)], -1.5);
        assert_eq!(j3[(0, 1)], -3.0);
        let mut j2m = Array2::<f64>::zeros((2, 2));
        let mut j3m = Array2::<f64>::zeros((1, 2));
        subtract_g0(
            &mut j2m,
            &mut j3m,
            &[2.0],
            &[1.0, 3.0],
            0.5,
            G0Handling::MetricOnlyMutant,
        );
        assert_eq!(j2m, j2);
        assert_eq!(j3m[(0, 1)], 0.0);
    }
}
