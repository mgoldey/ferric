//! Stage 3: k-point sampled closed-shell RHF (`solve_krhf`), a SEPARATE
//! driver from `ferric_scf::rhf` (FINDINGS "Iteration 9", "For the Rust
//! port": `solve_rhf_impl` and its byte-identity contract stay untouched;
//! `ScfResult`/`Diis` are not generified).
//!
//! ```text
//! F(k) = h(k) + J(k) − ½ K(k)                      (K includes v_M S D S for exxdiv = ewald)
//! F(k) C(k) = S(k) C(k) ε(k)                        (complex Hermitian; canonical orthogonaliser per k)
//! D(k) = 2 C_occ(k) C_occ(k)^H                      (global aufbau over all k, PySCF KRHF get_occ)
//! E/cell = (1/N_k) Σ_k ½ Re tr[(h(k) + F(k)) D(k)] + E_nn
//! ```
//!
//! * Time reversal: only one of each `{k, −k}` pair is diagonalised;
//!   `C(−k) = C(k)*`, `ε(−k) = ε(k)` (`X(−k) = X(k)*` holds EXACTLY for the
//!   one-electron matrices, [`crate::kpts`]).
//! * Canonical orthogonaliser per k: eigenvectors of `S(k)` with
//!   `λ >= lindep` (default `1e-6`, ferric's Gamma `LINDEP_THRESH`), scaled
//!   by `λ^{−1/2}`.
//! * DIIS over the stacked k blocks: error `X^H (F D S − S D F) X` per k,
//!   Gram `B_ij = Re Σ_k ⟨e_i(k), e_j(k)⟩`, REAL extrapolation weights
//!   (so time reversal survives extrapolation).
//! * Occupations: global aufbau (`N_k · N_e/2` lowest levels over all k).
//!   Occupations may change between iterations; a closed gap
//!   (`ε_LUMO − ε_HOMO < min_gap`, which includes a Fermi level splitting a
//!   degenerate set such as a `±k` pair) is a hard error — metals and
//!   fractional occupation are out of scope.
//!
//! # J/K sources ([`KRhfConfig::jk`])
//!
//! * `dense` — the pure-AFT oracle ([`crate::kdense_aft`]; `N_k² nao⁴`
//!   memory, hard-capped): toy cells and exactness anchors.
//! * `rsgdf` — complex RS-GDF per momentum transfer q
//!   ([`crate::rsgdf::kpoint::KRsGdf`]): aux FT at `G + q`, a Hermitian
//!   metric `J2(q)` with its own eig + lindep cut (only q = 0 carries the
//!   Gamma G = 0 bookkeeping), SR 3-centre lattice sums binned by residue and
//!   weighted by `e^{ik'·L} e^{−iq·T}`. Needs an aux basis.
//!
//! Shifted (Monkhorst-Pack, even N) meshes run through the same code but
//! have no supercell anchor.

use crate::dense_aft::ExxDiv;
use crate::ewald::default_ewald_omega;
use crate::hcore::kpoint::{hermitize, periodic_hcore_kpts};
use crate::hcore::PeriodicHcoreConfig;
use crate::kdense_aft::{KDenseAftConfig, KDenseAftEri};
use crate::kpts::KPointMesh;
use crate::lattice::Cell;
use crate::lindep::{overlap_abs_row_sum, KLindep, LindepReport};
use crate::rsgdf::kpoint::{KRsGdf, KRsGdfConfig};
use crate::timing::{PbcTimings, StageClock};
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ndarray::{s, Array2};
use ndarray_linalg::{Eigh, UPLO};
use num_complex::Complex64;
use std::collections::VecDeque;

/// Coulomb + exchange builder for the k-point SCF: J needs only the
/// k-summed density (q = 0), K needs every `(k, k')` pair, so one trait
/// carries both. `k` must include any exchange-divergence (Madelung) term.
pub trait KPointJk {
    /// Overwrite `j[k]`, `k[k]` for densities `dm[k]` (one per mesh point).
    fn build(
        &mut self,
        dm: &[Array2<Complex64>],
        j: &mut [Array2<Complex64>],
        k: &mut [Array2<Complex64>],
    ) -> Result<(), FerricError>;
}

/// What [`solve_krhf_injected`] needs besides the mesh.
pub struct KPointInjection<'a> {
    /// `S(k)`, mesh order.
    pub s: Vec<Array2<Complex64>>,
    /// `h(k)`, mesh order.
    pub h: Vec<Array2<Complex64>>,
    /// Nuclear repulsion per cell.
    pub vnn: f64,
    /// J/K builder.
    pub jk: Box<dyn KPointJk + 'a>,
}

/// SCF settings.
#[derive(Debug, Clone, Copy)]
pub struct KScfConfig {
    /// Iteration cap.
    pub max_iter: usize,
    /// `|ΔE|` per cell below which (with `grad_conv`) the SCF has converged.
    pub energy_conv: f64,
    /// Max `|X^H (F D S − S D F) X|` element over all k.
    pub grad_conv: f64,
    /// DIIS subspace (0 or 1 disables extrapolation).
    pub diis_space: usize,
    /// Canonical-orthogonaliser cut on the eigenvalues of `S(k)`.
    pub lindep: f64,
    /// Smallest allowed HOMO-LUMO gap (Hartree) in the global aufbau.
    pub min_gap: f64,
}

impl Default for KScfConfig {
    fn default() -> Self {
        Self {
            max_iter: 200,
            energy_conv: 1e-12,
            grad_conv: 1e-9,
            diis_space: 8,
            lindep: 1e-6,
            min_gap: 1e-6,
        }
    }
}

/// Result of a k-point RHF.
#[derive(Debug, Clone)]
pub struct KScfResult {
    /// Total energy PER CELL (Hartree), of the density in `densities`.
    pub energy: f64,
    /// Nuclear repulsion per cell.
    pub e_nuc: f64,
    /// Orbital energies per k (ascending, real), mesh order.
    pub eps: Vec<Vec<f64>>,
    /// MO coefficients per k, `(nao, n_orth)`, complex.
    pub mos: Vec<Array2<Complex64>>,
    /// `D(k)` (Hermitian, occupation 2), the density `energy` belongs to.
    pub densities: Vec<Array2<Complex64>>,
    /// `F(k)` of `densities` (not extrapolated).
    pub fock: Vec<Array2<Complex64>>,
    /// Occupation numbers (2 or 0) per k, aligned with `eps`.
    pub occupations: Vec<Vec<f64>>,
    /// Doubly occupied levels per k.
    pub nocc_per_k: Vec<usize>,
    /// Highest occupied / lowest unoccupied level over all k.
    pub homo: f64,
    /// See `homo` (`+∞` if no virtuals).
    pub lumo: f64,
    /// Converged flag.
    pub converged: bool,
    /// SCF iterations run.
    pub iterations: usize,
    /// Final max orbital-gradient element.
    pub max_error: f64,
    /// Cartesian k-points (Bohr⁻¹), mesh order.
    pub kpts: Vec<[f64; 3]>,
    /// Per-k canonical-cut diagnostics (kept counts, smallest / largest
    /// dropped eigenvalue of `S(k)`, noise-floor flag; [`crate::lindep`]).
    pub lindep: LindepReport,
    /// Coarse stages from [`solve_krhf`] (`"k hcore"`, `"k J/K build"`,
    /// `"k SCF"`) and the J/K build's counters; empty from
    /// [`solve_krhf_injected`], whose caller owns the builds.
    pub timings: PbcTimings,
}

/// Which J/K builder [`solve_krhf`] uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KJkKind {
    /// Dense pure-AFT kernels (oracle; toy scale).
    Dense,
    /// k-point RS-GDF (needs an aux basis).
    RsGdf,
}

impl KJkKind {
    /// Strict parse: `"dense"` or `"rsgdf"` (case-insensitive); anything
    /// else is an error (config honesty — no silent default).
    pub fn parse_config_str(s: &str) -> Result<Self, FerricError> {
        match s.to_ascii_lowercase().as_str() {
            "dense" => Ok(KJkKind::Dense),
            "rsgdf" => Ok(KJkKind::RsGdf),
            other => Err(FerricError::General(format!(
                "k-point jk must be \"dense\" or \"rsgdf\", got {other:?}"
            ))),
        }
    }
}

/// Top-level settings for [`solve_krhf`].
#[derive(Debug, Clone, Copy)]
pub struct KRhfConfig {
    /// SCF settings.
    pub scf: KScfConfig,
    /// Exchange-divergence treatment (overrides `rsgdf.gdf.exxdiv`).
    pub exxdiv: ExxDiv,
    /// One-electron lattice sums (as `periodic_hcore`).
    pub hcore: PeriodicHcoreConfig,
    /// J/K builder.
    pub jk: KJkKind,
    /// Dense-AFT J/K oracle settings (`jk = Dense`).
    pub dense: KDenseAftConfig,
    /// k-point RS-GDF settings (`jk = RsGdf`).
    pub rsgdf: KRsGdfConfig,
}

impl KRhfConfig {
    /// Defaults (dense J/K) with the balanced Ewald ω of `cell` for the
    /// nuclear split.
    pub fn for_cell(cell: &Cell, exxdiv: ExxDiv) -> Self {
        Self {
            scf: KScfConfig::default(),
            exxdiv,
            hcore: PeriodicHcoreConfig::with_omega(default_ewald_omega(cell)),
            jk: KJkKind::Dense,
            dense: KDenseAftConfig::default(),
            rsgdf: KRsGdfConfig::default(),
        }
    }
}

/// k-point RHF: builds `S(k)`, `h(k)` ([`periodic_hcore_kpts`]) and the J/K
/// source selected by `cfg.jk` — the dense pure-AFT kernels
/// ([`KDenseAftEri`], `aux` must be `None`) or k-point RS-GDF ([`KRsGdf`],
/// `aux` REQUIRED) — then runs [`solve_krhf_injected`].
pub fn solve_krhf(
    cell: &Cell,
    prep: &PreparedBasis,
    aux: Option<&PreparedBasis>,
    mesh: &KPointMesh,
    cfg: &KRhfConfig,
) -> Result<KScfResult, FerricError> {
    match (cfg.jk, aux) {
        (KJkKind::Dense, Some(_)) => {
            return Err(FerricError::General(
                "solve_krhf: an aux basis was given but jk = dense does not use it; set jk = \
                 rsgdf or pass no aux basis"
                    .into(),
            ))
        }
        (KJkKind::RsGdf, None) => {
            return Err(FerricError::General(
                "solve_krhf: jk = rsgdf requires an auxbasis (none given)".into(),
            ))
        }
        _ => {}
    }
    let total = StageClock::start();
    let mut timings = PbcTimings::default();
    let clock = StageClock::start();
    let hk = periodic_hcore_kpts(cell, prep, mesh, &cfg.hcore)?;
    timings.stop("k hcore", &clock);
    let clock = StageClock::start();
    let mut r = match aux {
        None => {
            let eri = KDenseAftEri::build(cell, prep, mesh, &hk.s, cfg.exxdiv, &cfg.dense)?;
            timings.stop("k J/K build", &clock);
            let inj = KPointInjection {
                s: hk.s,
                h: hk.h,
                vnn: hk.enn,
                jk: Box::new(eri.jk_builder()),
            };
            let clock = StageClock::start();
            let r = solve_krhf_injected(cell, mesh, &cfg.scf, inj)?;
            timings.stop("k SCF", &clock);
            r
        }
        Some(aux) => {
            let gdf =
                KRsGdf::build(cell, prep, aux, mesh, &hk.s, &cfg.rsgdf)?.with_exxdiv(cfg.exxdiv);
            timings.stop("k J/K build", &clock);
            crate::rsgdf::kpoint::record_stats(&mut timings, gdf.stats());
            let inj = KPointInjection {
                s: hk.s,
                h: hk.h,
                vnn: hk.enn,
                jk: Box::new(gdf.jk_builder()),
            };
            let clock = StageClock::start();
            let r = solve_krhf_injected(cell, mesh, &cfg.scf, inj)?;
            timings.stop("k SCF", &clock);
            r
        }
    };
    timings.finish(&total);
    r.timings = timings;
    Ok(r)
}

pub(crate) fn herm_t(m: &Array2<Complex64>) -> Array2<Complex64> {
    m.t().mapv(|z| z.conj())
}

/// Hermitian eigendecomposition of a complex matrix, copied to column-major
/// (Fortran) layout first. `ndarray-linalg`'s `eigh` on a ROW-MAJOR complex
/// Hermitian matrix returns eigenvectors that do not satisfy `A v = v w`
/// (measured residual 1.2 on a 2x2 example vs 2e-16 column-major); real
/// matrices are unaffected, which is why every Gamma / real-k path worked.
pub(crate) fn eigh_herm(
    m: &Array2<Complex64>,
) -> Result<(ndarray::Array1<f64>, Array2<Complex64>), ndarray_linalg::error::LinalgError> {
    use ndarray::ShapeBuilder;
    let mut f = Array2::<Complex64>::zeros(m.raw_dim().f());
    f.assign(m);
    f.eigh(UPLO::Upper)
}

/// Complex canonical orthogonaliser: `X = U_kept λ_kept^{−1/2}` over the
/// eigenpairs of the Hermitian `S` with `λ >= lindep` (ascending).
pub fn complex_canonical_orthogonalizer(
    s: &Array2<Complex64>,
    lindep: f64,
) -> Result<Array2<Complex64>, FerricError> {
    Ok(complex_canonical_orthogonalizer_with_stats(s, lindep)?.0)
}

/// [`complex_canonical_orthogonalizer`] plus the cut's statistics from the
/// SAME eigendecomposition (kept count, smallest eigenvalue, largest dropped
/// eigenvalue). The cut is on the RAW eigenvalues of the unnormalised `S`
/// (FINDINGS Iteration 15: a normalised cut breaks k-mesh ≡ supercell).
pub fn complex_canonical_orthogonalizer_with_stats(
    s: &Array2<Complex64>,
    lindep: f64,
) -> Result<(Array2<Complex64>, KLindep), FerricError> {
    let (w, u) =
        eigh_herm(&hermitize(s)).map_err(|e| FerricError::Lapack(format!("S(k) diag: {e}")))?;
    let kept: Vec<usize> = (0..w.len()).filter(|&i| w[i] >= lindep).collect();
    if kept.is_empty() {
        return Err(FerricError::General(format!(
            "complex_canonical_orthogonalizer: every eigenvalue of S(k) is below lindep {lindep:e} \
             (max {:e})",
            w.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        )));
    }
    let n = s.nrows();
    let mut x = Array2::<Complex64>::zeros((n, kept.len()));
    for (c, &i) in kept.iter().enumerate() {
        let f = 1.0 / w[i].sqrt();
        for r in 0..n {
            x[(r, c)] = u[(r, i)] * f;
        }
    }
    Ok((x, KLindep::from_eigenvalues(&w.to_vec(), lindep)))
}

/// Per-k orthogonalisers for the time-reversal representatives (partners by
/// conjugation: `S(−k) = S(k)*`, same spectrum) and the [`LindepReport`];
/// prints the noise-floor warning (as `who`) when flagged.
pub(crate) fn orthogonalizers_with_report(
    mesh: &KPointMesh,
    s: &[Array2<Complex64>],
    lindep: f64,
    who: &str,
) -> Result<(Vec<Array2<Complex64>>, LindepReport), FerricError> {
    let nk = mesh.nk();
    let mut x: Vec<Array2<Complex64>> = vec![Array2::zeros((0, 0)); nk];
    let mut stats: Vec<KLindep> = vec![KLindep::default(); nk];
    for k in mesh.tr_representatives() {
        let (xk, st) = complex_canonical_orthogonalizer_with_stats(&s[k], lindep)?;
        let p = mesh.minus(k);
        if p != k {
            x[p] = xk.mapv(|z| z.conj());
            stats[p] = st.clone();
        }
        x[k] = xk;
        stats[k] = st;
    }
    let abs_sum = s.iter().map(overlap_abs_row_sum).fold(0.0_f64, f64::max);
    let report = LindepReport::from_parts(lindep, stats, abs_sum);
    report.warn_if_near_noise_floor(who);
    Ok((x, report))
}

/// DIIS over stacked k blocks with real weights.
pub(crate) struct KDiis {
    cap: usize,
    focks: VecDeque<Vec<Array2<Complex64>>>,
    errs: VecDeque<Vec<Array2<Complex64>>>,
}

fn gram(a: &[Array2<Complex64>], b: &[Array2<Complex64>]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            x.iter()
                .zip(y.iter())
                .map(|(p, q)| (p.conj() * q).re)
                .sum::<f64>()
        })
        .sum()
}

/// Solve the bordered DIIS system on the newest `m` vectors (Gram block
/// normalised to unit scale); `None` if singular.
fn diis_coeffs(b: &[Vec<f64>]) -> Option<Vec<f64>> {
    let m = b.len();
    let dim = m + 1;
    let scale = b
        .iter()
        .flat_map(|r| r.iter())
        .fold(0.0_f64, |a, x| a.max(x.abs()));
    if !(scale > 0.0) || !scale.is_finite() {
        return None;
    }
    let mut a = vec![0.0_f64; dim * dim];
    let mut rhs = vec![0.0_f64; dim];
    for i in 0..m {
        for j in 0..m {
            a[i * dim + j] = b[i][j] / scale;
        }
        a[i * dim + m] = 1.0;
        a[m * dim + i] = 1.0;
    }
    rhs[m] = 1.0;
    // Gaussian elimination with partial pivoting.
    for col in 0..dim {
        let piv = (col..dim)
            .max_by(|&x, &y| a[x * dim + col].abs().total_cmp(&a[y * dim + col].abs()))?;
        if a[piv * dim + col].abs() < 1e-14 {
            return None;
        }
        if piv != col {
            for c in 0..dim {
                a.swap(piv * dim + c, col * dim + c);
            }
            rhs.swap(piv, col);
        }
        for r in (col + 1)..dim {
            let f = a[r * dim + col] / a[col * dim + col];
            if f != 0.0 {
                for c in col..dim {
                    a[r * dim + c] -= f * a[col * dim + c];
                }
                rhs[r] -= f * rhs[col];
            }
        }
    }
    let mut x = vec![0.0_f64; dim];
    for r in (0..dim).rev() {
        let mut acc = rhs[r];
        for c in (r + 1)..dim {
            acc -= a[r * dim + c] * x[c];
        }
        x[r] = acc / a[r * dim + r];
    }
    let c: Vec<f64> = x[..m].to_vec();
    c.iter().all(|v| v.is_finite()).then_some(c)
}

impl KDiis {
    pub(crate) fn new(cap: usize) -> Self {
        Self {
            cap,
            focks: VecDeque::new(),
            errs: VecDeque::new(),
        }
    }

    pub(crate) fn step(
        &mut self,
        f: &[Array2<Complex64>],
        e: &[Array2<Complex64>],
    ) -> Vec<Array2<Complex64>> {
        if self.cap < 2 {
            return f.to_vec();
        }
        self.focks.push_back(f.to_vec());
        self.errs.push_back(e.to_vec());
        while self.focks.len() > self.cap {
            self.focks.pop_front();
            self.errs.pop_front();
        }
        let len = self.errs.len();
        if len < 2 {
            return f.to_vec();
        }
        // Newest m vectors, m = len … 2 (drop the oldest on a singular B).
        for m in (2..=len).rev() {
            let off = len - m;
            let b: Vec<Vec<f64>> = (0..m)
                .map(|i| {
                    (0..m)
                        .map(|j| gram(&self.errs[off + i], &self.errs[off + j]))
                        .collect()
                })
                .collect();
            if let Some(c) = diis_coeffs(&b) {
                let mut out: Vec<Array2<Complex64>> =
                    f.iter().map(|x| Array2::zeros(x.dim())).collect();
                for (i, ci) in c.iter().enumerate() {
                    for (o, fi) in out.iter_mut().zip(&self.focks[off + i]) {
                        o.scaled_add(Complex64::new(*ci, 0.0), fi);
                    }
                }
                return out;
            }
        }
        f.to_vec()
    }
}

/// Diagonalise `F(k)` for the time-reversal representatives and fill the
/// partners by conjugation. Returns `(ε(k), C(k))` for every k.
pub(crate) fn diagonalize_all(
    mesh: &KPointMesh,
    f: &[Array2<Complex64>],
    x: &[Array2<Complex64>],
) -> Result<(Vec<Vec<f64>>, Vec<Array2<Complex64>>), FerricError> {
    let nk = mesh.nk();
    let mut eps: Vec<Vec<f64>> = vec![Vec::new(); nk];
    let mut cs: Vec<Array2<Complex64>> = vec![Array2::zeros((0, 0)); nk];
    for k in mesh.tr_representatives() {
        let fo = hermitize(&herm_t(&x[k]).dot(&f[k]).dot(&x[k]));
        let (w, v) = eigh_herm(&fo)
            .map_err(|e| FerricError::Lapack(format!("F(k) diag at k = {k}: {e}")))?;
        let c = x[k].dot(&v);
        let p = mesh.minus(k);
        if p != k {
            eps[p] = w.to_vec();
            cs[p] = c.mapv(|z| z.conj());
        }
        eps[k] = w.to_vec();
        cs[k] = c;
    }
    Ok((eps, cs))
}

/// Global aufbau: the `n_occ_total` lowest levels over all k. Errors if the
/// gap closes (module doc).
pub(crate) fn aufbau(
    eps: &[Vec<f64>],
    n_occ_total: usize,
    min_gap: f64,
) -> Result<(Vec<Vec<f64>>, f64, f64), FerricError> {
    let mut all: Vec<(f64, usize, usize)> = eps
        .iter()
        .enumerate()
        .flat_map(|(k, e)| e.iter().enumerate().map(move |(i, &v)| (v, k, i)))
        .collect();
    if all.len() < n_occ_total || n_occ_total == 0 {
        return Err(FerricError::General(format!(
            "solve_krhf: {n_occ_total} doubly occupied levels requested but {} orbitals exist",
            all.len()
        )));
    }
    all.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
    let homo = all[n_occ_total - 1].0;
    let lumo = all.get(n_occ_total).map_or(f64::INFINITY, |t| t.0);
    if lumo - homo < min_gap {
        return Err(FerricError::General(format!(
            "solve_krhf: HOMO-LUMO gap {:.3e} Ha < min_gap {min_gap:.1e} (HOMO {homo:.10}, LUMO \
             {lumo:.10}); metallic / fractional occupation is not supported",
            lumo - homo
        )));
    }
    let mut occ: Vec<Vec<f64>> = eps.iter().map(|e| vec![0.0; e.len()]).collect();
    for &(_, k, i) in &all[..n_occ_total] {
        occ[k][i] = 2.0;
    }
    Ok((occ, homo, lumo))
}

fn density(c: &Array2<Complex64>, occ: &[f64]) -> Array2<Complex64> {
    occupied_projector(c, occ).mapv(|z| z * 2.0)
}

/// `C_occ C_occ^H` over the columns with `occ[i] > 0` (unit occupation; the
/// RHF density is twice this, the per-spin UHF density is this).
pub(crate) fn occupied_projector(c: &Array2<Complex64>, occ: &[f64]) -> Array2<Complex64> {
    let occ_idx: Vec<usize> = (0..occ.len()).filter(|&i| occ[i] > 0.0).collect();
    let n = c.nrows();
    let mut co = Array2::<Complex64>::zeros((n, occ_idx.len()));
    for (j, &i) in occ_idx.iter().enumerate() {
        co.slice_mut(s![.., j]).assign(&c.slice(s![.., i]));
    }
    co.dot(&herm_t(&co))
}

/// k-point closed-shell RHF on injected `S(k)`, `h(k)`, `E_nn` and J/K
/// (module doc). The cell supplies the electron count (`mol().nelec()`,
/// must be even and positive); `inj.s`/`inj.h` must have one
/// `(n, n)` matrix per mesh point.
pub fn solve_krhf_injected(
    cell: &Cell,
    mesh: &KPointMesh,
    cfg: &KScfConfig,
    mut inj: KPointInjection<'_>,
) -> Result<KScfResult, FerricError> {
    let nk = mesh.nk();
    if inj.s.len() != nk || inj.h.len() != nk {
        return Err(FerricError::General(format!(
            "solve_krhf: {} S / {} h matrices for {nk} k-points",
            inj.s.len(),
            inj.h.len()
        )));
    }
    let n = inj.s[0].nrows();
    if inj.s.iter().chain(inj.h.iter()).any(|m| m.dim() != (n, n)) {
        return Err(FerricError::General(
            "solve_krhf: S(k)/h(k) must all be square of one size".into(),
        ));
    }
    if !inj.vnn.is_finite() {
        return Err(FerricError::General(format!(
            "solve_krhf: non-finite E_nn {}",
            inj.vnn
        )));
    }
    let nelec = cell.mol().nelec();
    if nelec <= 0 || nelec % 2 != 0 {
        return Err(FerricError::General(format!(
            "solve_krhf: closed-shell RHF needs an even, positive electron count per cell, got {nelec}"
        )));
    }
    if !(cfg.lindep > 0.0) || !(cfg.min_gap >= 0.0) || cfg.max_iter == 0 {
        return Err(FerricError::General(format!(
            "solve_krhf: invalid config (lindep {}, min_gap {}, max_iter {})",
            cfg.lindep, cfg.min_gap, cfg.max_iter
        )));
    }
    let nocc = (nelec / 2) as usize;

    // Orthogonalisers: representatives explicitly, partners by conjugation
    // (S(−k) = S(k)* exactly); per-k kept counts may differ (lindep module).
    let (x, lindep_report) = orthogonalizers_with_report(mesh, &inj.s, cfg.lindep, "solve_krhf")?;
    for (k, xk) in x.iter().enumerate() {
        if xk.ncols() < nocc {
            return Err(FerricError::General(format!(
                "solve_krhf: {nocc} occupied orbitals but S(k={k}) keeps only {} of {n} after the \
                 lindep cut",
                xk.ncols()
            )));
        }
    }

    let zero = || Array2::<Complex64>::zeros((n, n));
    let mut dm: Vec<Array2<Complex64>> = (0..nk).map(|_| zero()).collect();
    let mut jm: Vec<Array2<Complex64>> = (0..nk).map(|_| zero()).collect();
    let mut km: Vec<Array2<Complex64>> = (0..nk).map(|_| zero()).collect();
    let mut diis = KDiis::new(cfg.diis_space);
    let mut e_old = f64::NAN;
    let inv_nk = 1.0 / nk as f64;
    let mut last: Option<(f64, Vec<Array2<Complex64>>, f64)> = None;
    let mut last_diag: Option<(
        Vec<Vec<f64>>,
        Vec<Array2<Complex64>>,
        Vec<Vec<f64>>,
        f64,
        f64,
    )> = None;

    for it in 0..cfg.max_iter {
        inj.jk.build(&dm, &mut jm, &mut km)?;
        let f: Vec<Array2<Complex64>> = (0..nk)
            .map(|k| hermitize(&(&(&inj.h[k] + &jm[k]) - &km[k].mapv(|z| z * 0.5))))
            .collect();
        let mut e_elec = 0.0;
        for k in 0..nk {
            let hf = &inj.h[k] + &f[k];
            // Re tr[(h + F) D] = Re Σ_mn (h+F)_mn D_nm
            let mut tr = 0.0;
            for m in 0..n {
                for nu in 0..n {
                    tr += (hf[(m, nu)] * dm[k][(nu, m)]).re;
                }
            }
            e_elec += 0.5 * tr;
        }
        let energy = e_elec * inv_nk + inj.vnn;
        let errs: Vec<Array2<Complex64>> = (0..nk)
            .map(|k| {
                let fds = f[k].dot(&dm[k]).dot(&inj.s[k]);
                let comm = &fds - &herm_t(&fds);
                herm_t(&x[k]).dot(&comm).dot(&x[k])
            })
            .collect();
        let emax = errs
            .iter()
            .flat_map(|e| e.iter())
            .fold(0.0_f64, |a, z| a.max(z.norm()));
        if it > 0 && (energy - e_old).abs() < cfg.energy_conv && emax < cfg.grad_conv {
            let (eps, cs) = diagonalize_all(mesh, &f, &x)?;
            let (occ, homo, lumo) = aufbau(&eps, nocc * nk, cfg.min_gap)?;
            return Ok(finish(
                mesh,
                energy,
                inj.vnn,
                eps,
                cs,
                dm,
                f,
                occ,
                homo,
                lumo,
                true,
                it,
                emax,
                lindep_report.clone(),
            ));
        }
        e_old = energy;
        let f_use = if it > 0 {
            diis.step(&f, &errs)
        } else {
            f.clone()
        };
        let (eps, cs) = diagonalize_all(mesh, &f_use, &x)?;
        // Intermediate iterations: a degenerate level at the cut (e.g. the core
        // guess of an atom with degenerate p levels) is broken by index, not
        // refused; the gap is enforced on the CONVERGED result above.
        let (occ, homo, lumo) = aufbau(&eps, nocc * nk, 0.0)?;
        let new_dm: Vec<Array2<Complex64>> = (0..nk).map(|k| density(&cs[k], &occ[k])).collect();
        last = Some((energy, f, emax));
        last_diag = Some((eps, cs, occ, homo, lumo));
        dm = new_dm;
    }
    // Not converged: report the last evaluated energy/Fock with the density
    // that produced them is gone; report the latest orbitals and density.
    let (energy, f, emax) = last.expect("max_iter >= 1");
    let (eps, cs, occ, homo, lumo) = last_diag.expect("max_iter >= 1");
    Ok(finish(
        mesh,
        energy,
        inj.vnn,
        eps,
        cs,
        dm,
        f,
        occ,
        homo,
        lumo,
        false,
        cfg.max_iter,
        emax,
        lindep_report,
    ))
}

#[allow(clippy::too_many_arguments)]
fn finish(
    mesh: &KPointMesh,
    energy: f64,
    e_nuc: f64,
    eps: Vec<Vec<f64>>,
    mos: Vec<Array2<Complex64>>,
    densities: Vec<Array2<Complex64>>,
    fock: Vec<Array2<Complex64>>,
    occupations: Vec<Vec<f64>>,
    homo: f64,
    lumo: f64,
    converged: bool,
    iterations: usize,
    max_error: f64,
    lindep: LindepReport,
) -> KScfResult {
    let nocc_per_k = occupations
        .iter()
        .map(|o| o.iter().filter(|&&v| v > 0.0).count())
        .collect();
    KScfResult {
        energy,
        e_nuc,
        eps,
        mos,
        densities,
        fock,
        occupations,
        nocc_per_k,
        homo,
        lumo,
        converged,
        iterations,
        max_error,
        kpts: mesh.kpts().to_vec(),
        lindep,
        timings: PbcTimings::default(),
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn complex_eigh_eigenvectors_satisfy_the_eigen_equation() {
        use num_complex::Complex64 as C;
        // Row-major on purpose: the layout that broke ndarray-linalg's eigh.
        let a: Array2<C> = ndarray::array![
            [C::new(2.0, 0.0), C::new(0.3, 0.7), C::new(-0.2, 0.1)],
            [C::new(0.3, -0.7), C::new(1.0, 0.0), C::new(0.05, -0.4)],
            [C::new(-0.2, -0.1), C::new(0.05, 0.4), C::new(-0.5, 0.0)]
        ];
        assert!(a.is_standard_layout());
        let (w, v) = eigh_herm(&a).unwrap();
        let res = (&a.dot(&v) - &(&v * &w.mapv(|x| C::new(x, 0.0))))
            .iter()
            .fold(0.0_f64, |m, z| m.max(z.norm()));
        assert!(res < 1e-13, "eigen residual {res:e}");
    }

    use super::*;

    #[test]
    fn diis_coeffs_sum_to_one_and_reject_singular() {
        let b = vec![vec![2.0, 0.5], vec![0.5, 1.0]];
        let c = diis_coeffs(&b).unwrap();
        assert!((c.iter().sum::<f64>() - 1.0).abs() < 1e-14);
        // minimiser of cᵀBc on Σc = 1: c ∝ B⁻¹ 1
        let (d0, d1) = (1.0 - 0.5, 2.0 - 0.5);
        assert!((c[0] - d0 / (d0 + d1)).abs() < 1e-14, "{c:?}");
        assert!(diis_coeffs(&[vec![0.0, 0.0], vec![0.0, 0.0]]).is_none());
    }

    #[test]
    fn aufbau_refuses_a_split_degenerate_pair() {
        let eps = vec![vec![-1.0, 0.5], vec![-1.0, 0.5]];
        assert!(aufbau(&eps, 1, 1e-6).is_err());
        let (occ, h, l) = aufbau(&eps, 2, 1e-6).unwrap();
        assert_eq!(occ, vec![vec![2.0, 0.0], vec![2.0, 0.0]]);
        assert_eq!((h, l), (-1.0, 0.5));
    }
}
