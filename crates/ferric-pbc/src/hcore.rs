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

use crate::ewald::{default_ewald_omega, ewald_nuclear_repulsion};
use crate::lattice::Cell;
use crate::pair_ft::pair_ft_with_thresh;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::engine::Engine;
use ferric_integrals::ffi;
use ferric_integrals::md3c1e::prim_norm;
use ferric_integrals::operator::Operator;
use ferric_integrals::site_basis::SiteBasis;
use ndarray::Array2;
use std::f64::consts::PI;

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
/// Upper bound on one `pair_ft` chunk (`16 · nao² · n_G` bytes).
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
}

impl PeriodicHcoreConfig {
    /// Defaults ([`DEFAULT_HCORE_PRECISION`], [`GAUSSIAN_NUCLEUS_EXPONENT`])
    /// at the given ω.
    pub fn with_omega(omega: f64) -> Self {
        Self {
            omega,
            precision: DEFAULT_HCORE_PRECISION,
            nucleus_exponent: GAUSSIAN_NUCLEUS_EXPONENT,
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
    /// `v_sr + v_lr + v_g0`.
    pub v: Array2<f64>,
    /// `t + v`.
    pub h: Array2<f64>,
    /// Ewald nuclear repulsion (PySCF `Cell.energy_nuc()` convention).
    pub enn: f64,
    /// The ω used.
    pub omega: f64,
    /// Number of pair images summed for S/T.
    pub n_images: usize,
    /// Number of shifted 3-centre calls in the SR attraction.
    pub n_sr_triplets: usize,
    /// Number of half-sphere G vectors in the LR attraction.
    pub n_g_half: usize,
    /// max |V_SR − V_SRᵀ| before symmetrisation (a lattice sum over an
    /// image set closed under L → −L is symmetric; a large value flags a
    /// truncation or image-set defect).
    pub sr_asymmetry: f64,
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
        if sh.l < 0 || sh.exponents.is_empty() || sh.exponents.len() != sh.coefficients.len() {
            return Err(FerricError::Basis(format!(
                "periodic_hcore: malformed shell {s} (l={}, {} exponents, {} coefficients)",
                sh.l,
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
    let pref = q * zmax * (1.0 + 2.0 * (pmax / PI).sqrt());
    if pref <= thresh {
        return None;
    }
    Some((pref / thresh).ln().sqrt() / omega_p + SR_MARGIN_BOHR)
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
fn reciprocal_nuclear(
    cell: &Cell,
    prep: &PreparedBasis,
    omega: Option<f64>,
    gcut: f64,
    pair_thresh: f64,
) -> Result<(Array2<f64>, usize), FerricError> {
    let n = prep.nbasis();
    let gv = half_gvectors(cell, gcut)?;
    let z = cell.nuclear_charges();
    let pos = cell.positions();
    let vol = cell.volume();
    let mut v = Array2::<f64>::zeros((n, n));
    let chunk = (G_CHUNK_BYTES / (16 * n * n).max(1)).max(1);
    for gs in gv.chunks(chunk) {
        let p = pair_ft_with_thresh(cell, prep, gs, pair_thresh)?;
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
    }
    Ok((v, gv.len()))
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
    reciprocal_nuclear(cell, prep, None, gcut, 0.1 * precision)
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

/// Build `S`, `T`, `V`, `h = T + V` and `E_nn` for a Gamma-point cell (see
/// the module doc). `prep` must be built from `cell.mol()`.
pub fn periodic_hcore(
    cell: &Cell,
    prep: &PreparedBasis,
    cfg: &PeriodicHcoreConfig,
) -> Result<PeriodicHcore, FerricError> {
    cfg.validate()?;
    let shells = prim_shells(cell, prep)?;
    let n = prep.nbasis();
    let omega = cfg.omega;
    let thresh = cfg.precision;
    let pair_thresh = 0.1 * thresh;

    let amin = shells
        .iter()
        .flat_map(|s| s.exps.iter().copied())
        .fold(f64::INFINITY, f64::min);
    let rpair = (2.0 * (1e3 / pair_thresh).ln() / amin).sqrt() + 2.0;
    let images = cell.translations(rpair)?;

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
    let pos = cell.positions();
    let nuc: Vec<(f64, [f64; 3])> = zs
        .iter()
        .zip(&pos)
        .filter(|(z, _)| **z != 0.0)
        .map(|(z, r)| (*z, *r))
        .collect();
    let mut v_sr = Array2::<f64>::zeros((n, n));
    let mut n_sr_triplets = 0usize;
    let mut sr_asymmetry = 0.0;
    if !nuc.is_empty() {
        let zmax = nuc.iter().map(|(z, _)| z.abs()).fold(0.0_f64, f64::max);
        let sites: Vec<[f64; 4]> = nuc
            .iter()
            .map(|(_, r)| [r[0], r[1], r[2], cfg.nucleus_exponent])
            .collect();
        let site = SiteBasis::new(&sites, 0)?;
        let mut eng = Engine::new_3center(
            Operator::erfc(omega),
            prep,
            &site.prep,
            ERI3_ENGINE_PRECISION,
        )?;

        // Candidate nucleus images: any (C, M) within r_nuc_max of a segment
        // whose endpoints lie within r_pair of a cell atom.
        let mut r_nuc_max = 0.0_f64;
        for a in &shells {
            for b in &shells {
                let (q, pmin, pmax) = pair_bound(a, b, 0.0);
                let wp = omega * (pmin / (pmin + omega * omega)).sqrt();
                if let Some(r) = nucleus_radius(q, zmax, pmax, wp, thresh) {
                    r_nuc_max = r_nuc_max.max(r);
                }
            }
        }
        let nuc_images = cell.translations(r_nuc_max + rpair)?;
        let mut cands: Vec<(usize, [f64; 3], [f64; 3])> = Vec::new();
        for m in &nuc_images {
            for (k, (_, r)) in nuc.iter().enumerate() {
                cands.push((k, *m, [r[0] + m[0], r[1] + m[1], r[2] + m[2]]));
            }
        }

        for l in &images {
            for (i1, a) in shells.iter().enumerate() {
                for (i2, b) in shells.iter().enumerate() {
                    let bc = [b.center[0] + l[0], b.center[1] + l[1], b.center[2] + l[2]];
                    let r2 = (a.center[0] - bc[0]).powi(2)
                        + (a.center[1] - bc[1]).powi(2)
                        + (a.center[2] - bc[2]).powi(2);
                    let (q, pmin, pmax) = pair_bound(a, b, r2);
                    let wp = omega * (pmin / (pmin + omega * omega)).sqrt();
                    let Some(rad) = nucleus_radius(q, zmax, pmax, wp, thresh) else {
                        continue;
                    };
                    for (k, m, x) in &cands {
                        if segment_distance(*x, a.center, bc) > rad {
                            continue;
                        }
                        n_sr_triplets += 1;
                        let f = -nuc[*k].0 / site.norm_int[*k];
                        if let Some(blk) = eng.compute_eri3_shifted(
                            prep,
                            &site.prep,
                            site.site_shell[*k],
                            i1,
                            i2,
                            [*m, [0.0; 3], *l],
                        )? {
                            add_block(&mut v_sr, blk, a.off, a.dim, b.off, b.dim, f);
                        }
                    }
                }
            }
        }
        let (sym, asym) = symmetrize(&v_sr);
        v_sr = sym;
        sr_asymmetry = asym;
    }

    // --- V_LR (G ≠ 0) and the G = 0 correction.
    let smooth = (1.0 / thresh).ln().sqrt();
    let gcut = (2.0 * omega).min(2.0 * max_pair_exponent(prep).sqrt()) * smooth;
    let (v_lr, n_g_half) = reciprocal_nuclear(cell, prep, Some(omega), gcut, pair_thresh)?;
    let ztot: f64 = zs.iter().sum();
    let c0 = PI / (omega * omega * cell.volume());
    let v_g0 = (c0 * ztot) * &s;

    let v = &(&v_sr + &v_lr) + &v_g0;
    let h = &t + &v;
    let enn = ewald_nuclear_repulsion(cell, default_ewald_omega(cell))?;
    Ok(PeriodicHcore {
        s,
        t,
        v_sr,
        v_lr,
        v_g0,
        v,
        h,
        enn,
        omega,
        n_images: images.len(),
        n_sr_triplets,
        n_g_half,
        sr_asymmetry,
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
