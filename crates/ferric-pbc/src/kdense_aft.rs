//! TEST/ORACLE ONLY (Stage 3): dense pure-AFT k-point Coulomb/exchange
//! kernels — the k-mesh generalisation of [`crate::dense_aft`].
//!
//! ```text
//! P^{k'}(K)_μν = Σ_L e^{ik'·L} ∫ φ_μ φ_ν(·−L) e^{−iK·r}       (Bloch pair χ*_{μk} χ_{νk'}, K = G + k' − k)
//! Jker[k,k'][(μν),(λσ)] = (1/Ω) Σ_{G≠0}        v(G)   P^{k}_μν(G)   conj P^{k'}_λσ(G)
//! Kker[k,k'][(μλ),(νσ)] = (1/Ω) Σ_{K∈G+q, K≠0} v(K)   P^{k'}_μλ(K)  conj P^{k'}_νσ(K),   q = k' − k
//! J_μν(k) = (1/N_k) Σ_{k'} Σ_λσ Jker[k,k'][(μν),(λσ)] D_λσ(k')
//! K_μν(k) = (1/N_k) Σ_{k'} Σ_λσ Kker[k,k'][(μλ),(νσ)] D_λσ(k')  +  v_M (S D S)(k)   (exxdiv = ewald)
//! ```
//!
//! with `v(K) = 4π/|K|²` and `D(k) = 2 C_occ(k) C_occ(k)^H`, so the SCF forms
//! `F(k) = h(k) + J(k) − ½K(k)` exactly as at Gamma. The ONLY dropped term
//! is `K = 0`, the q = 0 head of exchange (and G = 0 of J, cancelling the
//! nuclear G = 0 in a neutral cell) — term by term what the Gamma RHF of the
//! `diag(N)` supercell drops, because `{G + q}` over a Gamma-centred mesh is
//! the supercell's reciprocal lattice. `v_M` is the SUPERCELL Madelung
//! constant ([`KPointMesh::madelung`]), added per k with no `1/N_k` (PySCF
//! `df_jk._ewald_exxdiv_for_G0`). Derivation and conventions:
//! `reference/pbc/FINDINGS.md` "Iteration 9", `reference/pbc/pbc_kpts.py`.
//!
//! # Build
//!
//! One residue-resolved pair FT per momentum-transfer class `q` gives
//! `P^{k'}(G + q)` for EVERY `k'` by an `(N_k × R)` phase sum
//! ([`crate::pair_ft::residues`]); time reversal (`Kker[−k,−k'] =
//! conj Kker[k,k']`, real AOs) skips the `−q` pass. Cost ≈ `N_k/2` full-sphere
//! Gamma pair FTs plus `N_k` GEMMs `(nao² × n_K)(n_K × nao²)` per class.
//!
//! # Scope
//!
//! Memory is `2 N_k² nao⁴ × 16` bytes (HARD-capped by
//! [`KDenseAftConfig::max_bytes`], error not warning) and the K sphere grows
//! as `Ω p_max^{3/2}`: toy cells and the exactness anchors only. Production
//! k-point J/K is complex RS-GDF per q ([`crate::rsgdf::kpoint`]).

use crate::budget::{bytes_of, Ledger};
use crate::dense_aft::{ExxDiv, DEFAULT_DENSE_AFT_MAX_BYTES, DEFAULT_DENSE_AFT_PRECISION};
use crate::ewald::madelung_constant;
use crate::hcore::{gvector_list_bytes, G_CHUNK_BYTES};
use crate::kpts::KPointMesh;
use crate::kscf::KPointJk;
use crate::lattice::Cell;
use crate::pair_ft::residues::{pair_ft_residues_chunked, residue_coords};
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ndarray::{Array2, Array3};
use num_complex::Complex64;
use std::f64::consts::PI;

/// TEST-ONLY deliberate defects for the mutation tests of
/// `tests/pbc_krhf.rs` (the prototype's `_MUTANT` switch). Never set in
/// production; each must break the k-mesh ≡ supercell anchor.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KMutation {
    /// `e^{−ik'·L}` in the ERI pair FT only (S, T, V keep `e^{+ik·L}`).
    PhaseSignInPairFt,
    /// Exchange kernel `v(G)` instead of `v(G + q)` (`v(0) := 0`).
    KernelAtG,
    /// Primitive-cell Madelung constant instead of the supercell's.
    PrimitiveMadelung,
}

/// Settings for [`KDenseAftEri::build`].
#[derive(Debug, Clone, Copy)]
pub struct KDenseAftConfig {
    /// K sphere `|K| <= 2 √(p_max ln(1/precision))` (the Gamma
    /// [`crate::dense_aft`] rule, so a Gamma-centred mesh and its supercell
    /// sum the SAME K set at any precision); pair screen `0.1 · precision`.
    pub precision: f64,
    /// Hard cap on the two kernels, `2 N_k² nao⁴ × 16` bytes.
    pub max_bytes: usize,
    /// Memory budget (`None` = ferric's unified budget).
    pub budget_bytes: Option<usize>,
    #[doc(hidden)]
    pub mutation: Option<KMutation>,
    /// Range-separated exchange ([`crate::rsh`]): when set, the K kernel is
    /// `4π/K² [c_sr + (c_lr − c_sr) e^{−K²/4ω²}]` (the fractions are INSIDE
    /// the kernel, so the SCF must use an exact-exchange fraction of 1) and
    /// the `ExxDiv::Ewald` constant is the mixed supercell Madelung constant.
    /// J is unaffected.
    pub rsh: Option<crate::rsh::RshParams>,
}

impl Default for KDenseAftConfig {
    /// [`DEFAULT_DENSE_AFT_PRECISION`], [`DEFAULT_DENSE_AFT_MAX_BYTES`], unified budget, no mutation.
    fn default() -> Self {
        Self {
            precision: DEFAULT_DENSE_AFT_PRECISION,
            max_bytes: DEFAULT_DENSE_AFT_MAX_BYTES,
            budget_bytes: None,
            mutation: None,
            rsh: None,
        }
    }
}

/// Dense k-point J/K kernels plus what the Madelung term needs.
#[derive(Debug, Clone)]
pub struct KDenseAftEri {
    nao: usize,
    nk: usize,
    /// `[k * N_k + k']`, each `(nao², nao²)`.
    jker: Vec<Array2<Complex64>>,
    kker: Vec<Array2<Complex64>>,
    s: Vec<Array2<Complex64>>,
    /// `v_M` applied in K (0 for `ExxDiv::None`).
    madelung: f64,
    /// The mesh's `v_M` (what `ExxDiv::Ewald` applies).
    madelung_ewald: f64,
    n_k_total: usize,
    n_q_passes: usize,
    gcut: f64,
    /// Primitive-pair screen of the ERI pair FT (`0.1 · precision`).
    pair_thresh: f64,
}

/// Input validation of [`KDenseAftEri::build`]; returns the kernel byte count.
fn validate_build_inputs(
    nao: usize,
    nk: usize,
    s_k: &[Array2<Complex64>],
    cfg: &KDenseAftConfig,
) -> Result<u128, FerricError> {
    let n2 = nao * nao;
    let kbytes = 2u128 * (nk as u128).pow(2) * (n2 as u128).pow(2) * 16;
    if kbytes > cfg.max_bytes as u128 {
        return Err(FerricError::General(format!(
            "KDenseAftEri: dense k-point kernels need {kbytes} bytes (nao = {nao}, N_k = {nk}) \
             > cap {} bytes; this is a toy-cell oracle (use jk = rsgdf)",
            cfg.max_bytes
        )));
    }
    if s_k.len() != nk || s_k.iter().any(|s| s.dim() != (nao, nao)) {
        return Err(FerricError::General(format!(
            "KDenseAftEri: need {nk} overlap matrices of shape ({nao}, {nao})"
        )));
    }
    if let Some(r) = &cfg.rsh {
        r.validate()?;
    }
    if !(f64::MIN_POSITIVE..1.0).contains(&cfg.precision) {
        return Err(FerricError::General(format!(
            "KDenseAftEri: precision must lie in (0, 1), got {}",
            cfg.precision
        )));
    }
    Ok(kbytes)
}

/// K-sphere radius `2 sqrt(p_max ln(1/precision))`.
fn k_sphere_radius(prep: &PreparedBasis, precision: f64) -> f64 {
    let pmax = 2.0
        * prep
            .located_shells()
            .iter()
            .flat_map(|sh| sh.exponents.iter().copied())
            .fold(0.0_f64, f64::max);
    2.0 * (pmax * (1.0 / precision).ln()).sqrt()
}

/// Reserve the build's memory on a fresh ledger (kernels, overlap copies, K list).
fn reserve_build_memory(
    cell: &Cell,
    mesh: &KPointMesh,
    nao: usize,
    kbytes: u128,
    gcut: f64,
    cfg: &KDenseAftConfig,
) -> Result<Ledger, FerricError> {
    let nk = mesh.nk();
    let n2 = nao * nao;
    let mut ledger = Ledger::new(crate::budget::resolve(cfg.budget_bytes));
    ledger.reserve(
        &format!("KDenseAftEri J/K kernels (nao = {nao}, N_k = {nk})"),
        usize::try_from(kbytes).unwrap_or(usize::MAX),
    )?;
    ledger.reserve(
        &format!("KDenseAftEri overlap copies + per-block GEMM output (nao = {nao})"),
        bytes_of((nk * n2 + n2 * n2) as u64, 16),
    )?;
    let qmax = (0..nk)
        .map(|iq| norm3(mesh.q_class(iq).0))
        .fold(0.0_f64, f64::max);
    // Cell::gvectors at gcut + |q| plus the K copy (same bound).
    ledger.reserve(
        &format!("KDenseAftEri K list (|K| <= {gcut:.3}, |q| <= {qmax:.3})"),
        gvector_list_bytes(cell, gcut + qmax)?.saturating_mul(2),
    )?;
    Ok(ledger)
}

fn norm3(q: [f64; 3]) -> f64 {
    (q[0] * q[0] + q[1] * q[1] + q[2] * q[2]).sqrt()
}

/// Mesh phases per `(k, residue)`; conjugated for the
/// [`KMutation::PhaseSignInPairFt`] defect.
fn residue_phases(mesh: &KPointMesh, moduli: [usize; 3], flip: bool) -> Vec<Vec<Complex64>> {
    let nr = moduli[0] * moduli[1] * moduli[2];
    (0..mesh.nk())
        .map(|k| {
            (0..nr)
                .map(|r| {
                    let p = mesh.phase(k, residue_coords(r, moduli));
                    if flip {
                        p.conj()
                    } else {
                        p
                    }
                })
                .collect()
        })
        .collect()
}

/// The `ExxDiv::Ewald` constant: mixed supercell Madelung for RSH, else the mesh's.
fn ewald_madelung(
    cell: &Cell,
    mesh: &KPointMesh,
    cfg: &KDenseAftConfig,
) -> Result<f64, FerricError> {
    let primitive = cfg.mutation == Some(KMutation::PrimitiveMadelung);
    match cfg.rsh {
        Some(r) => {
            let lat = if primitive {
                *cell.lattice()
            } else {
                mesh.supercell_lattice()
            };
            r.madelung(&Cell::new(cell.mol().clone(), lat)?)
        }
        None if primitive => madelung_constant(cell),
        None => mesh.madelung(cell),
    }
}

/// Everything one momentum-transfer pass needs (shared across passes).
struct QPass<'a> {
    cell: &'a Cell,
    prep: &'a PreparedBasis,
    mesh: &'a KPointMesh,
    cfg: &'a KDenseAftConfig,
    phr: &'a [Vec<Complex64>],
    moduli: [usize; 3],
    gcut: f64,
    thresh: f64,
    chunk_budget: usize,
    extra_per_g: usize,
}

impl QPass<'_> {
    /// Run pass `iq`, accumulating into `jker`/`kker`; returns the pass's K count.
    fn run(
        &self,
        iq: usize,
        jker: &mut [Array2<Complex64>],
        kker: &mut [Array2<Complex64>],
    ) -> Result<usize, FerricError> {
        let (mesh, nk) = (self.mesh, self.mesh.nk());
        let (q, mq) = mesh.q_class(iq);
        let ks = shifted_k_vectors(self.cell, q, self.gcut)?;
        let chunk = ChunkCtx {
            q,
            vol: self.cell.volume(),
            kernel_at_g: self.cfg.mutation == Some(KMutation::KernelAtG),
            rsh: self.cfg.rsh,
            nao: self.prep.nbasis(),
            nk,
            iq,
            time_reversal: mesh.q_minus(iq) != iq,
            kof: (0..nk).map(|j| mesh.k_minus_q(j, mq)).collect(),
            minus: (0..nk).map(|k| mesh.minus(k)).collect(),
            phr: self.phr,
        };
        let sink =
            |_k0: usize, kv: &[[f64; 3]], qs: &[Array3<Complex64>]| -> Result<(), FerricError> {
                chunk.accumulate(kv, qs, jker, kker);
                Ok(())
            };
        pair_ft_residues_chunked(
            self.cell,
            self.prep,
            &ks,
            self.moduli,
            self.thresh,
            self.chunk_budget,
            self.extra_per_g,
            sink,
        )?;
        Ok(ks.len())
    }
}

/// `K = G + q` with `0 < |K| <= gcut` (`Cell::gvectors` at `gcut + |q|`).
fn shifted_k_vectors(cell: &Cell, q: [f64; 3], gcut: f64) -> Result<Vec<[f64; 3]>, FerricError> {
    let g2cut = gcut * gcut;
    Ok(cell
        .gvectors(gcut + norm3(q))?
        .into_iter()
        .map(|g| [g[0] + q[0], g[1] + q[1], g[2] + q[2]])
        .filter(|k| {
            let k2 = k[0] * k[0] + k[1] * k[1] + k[2] * k[2];
            k2 > 0.0 && k2 <= g2cut
        })
        .collect())
}

/// Per-pass context of the chunk accumulation.
struct ChunkCtx<'a> {
    q: [f64; 3],
    vol: f64,
    kernel_at_g: bool,
    rsh: Option<crate::rsh::RshParams>,
    nao: usize,
    nk: usize,
    iq: usize,
    /// `-q` is a different class: fill the mirrored block by conjugation.
    time_reversal: bool,
    kof: Vec<usize>,
    minus: Vec<usize>,
    phr: &'a [Vec<Complex64>],
}

impl ChunkCtx<'_> {
    /// `|K|^2` of the kernel argument (`K - q` under the `KernelAtG` defect).
    fn kernel_arg_sq(&self, k: &[f64; 3]) -> f64 {
        let x = if self.kernel_at_g {
            [k[0] - self.q[0], k[1] - self.q[1], k[2] - self.q[2]]
        } else {
            *k
        };
        x[0] * x[0] + x[1] * x[1] + x[2] * x[2]
    }

    /// Coulomb weights `4 pi / (K^2 Omega)` (0 at the head).
    fn coulomb_weights(&self, kv: &[[f64; 3]]) -> Vec<f64> {
        kv.iter()
            .map(|k| {
                let x2 = self.kernel_arg_sq(k);
                if x2 > 1e-20 {
                    4.0 * PI / x2 / self.vol
                } else {
                    0.0
                }
            })
            .collect()
    }

    /// Exchange weights: `v(K)` times the range-separation factor (if any).
    fn exchange_weights(&self, kv: &[[f64; 3]], vv: &[f64]) -> Vec<f64> {
        match self.rsh {
            None => vv.to_vec(),
            Some(r) => kv
                .iter()
                .zip(vv)
                .map(|(k, v)| v * r.kernel_factor(self.kernel_arg_sq(k)))
                .collect(),
        }
    }

    /// `P^{k'}(K)` for every `k'`, each `(nao^2, n_K)`.
    fn pair_ft_per_k(&self, qs: &[Array3<Complex64>], ng: usize) -> Vec<Array2<Complex64>> {
        let nao = self.nao;
        (0..self.nk)
            .map(|j| {
                let mut p = Array2::<Complex64>::zeros((nao * nao, ng));
                for (qr, ph) in qs.iter().zip(&self.phr[j]) {
                    for m in 0..nao {
                        for l in 0..nao {
                            let row = m * nao + l;
                            for g in 0..ng {
                                p[(row, g)] += *ph * qr[[m, l, g]];
                            }
                        }
                    }
                }
                p
            })
            .collect()
    }

    /// Fold one K chunk into the J/K kernels.
    fn accumulate(
        &self,
        kv: &[[f64; 3]],
        qs: &[Array3<Complex64>],
        jker: &mut [Array2<Complex64>],
        kker: &mut [Array2<Complex64>],
    ) {
        let nk = self.nk;
        let vv = self.coulomb_weights(kv);
        let pj = self.pair_ft_per_k(qs, kv.len());
        let vk_w = self.exchange_weights(kv, &vv);
        for j in 0..nk {
            let blk = scale_columns(&pj[j], &vk_w).dot(&herm(&pj[j]));
            let k = self.kof[j];
            kker[k * nk + j] += &blk;
            if self.time_reversal {
                kker[self.minus[k] * nk + self.minus[j]] += &blk.mapv(|z| z.conj());
            }
        }
        if self.iq == 0 {
            for k in 0..nk {
                let pv = scale_columns(&pj[k], &vv);
                for j in 0..nk {
                    jker[k * nk + j] += &pv.dot(&herm(&pj[j]));
                }
            }
        }
    }
}

fn scale_columns(p: &Array2<Complex64>, w: &[f64]) -> Array2<Complex64> {
    let mut pv = p.clone();
    for (mut col, v) in pv.columns_mut().into_iter().zip(w) {
        col.mapv_inplace(|z| z * *v);
    }
    pv
}

fn herm(p: &Array2<Complex64>) -> Array2<Complex64> {
    p.t().mapv(|z| z.conj())
}

impl KDenseAftEri {
    /// Build the kernels for `mesh` on `cell` in the AO basis `prep` (built
    /// from `cell.mol()`); `s_k` = `S(k)` per mesh point (for the Madelung
    /// term only).
    pub fn build(
        cell: &Cell,
        prep: &PreparedBasis,
        mesh: &KPointMesh,
        s_k: &[Array2<Complex64>],
        exxdiv: ExxDiv,
        cfg: &KDenseAftConfig,
    ) -> Result<Self, FerricError> {
        let nao = prep.nbasis();
        let nk = mesh.nk();
        let n2 = nao * nao;
        let kbytes = validate_build_inputs(nao, nk, s_k, cfg)?;
        let gcut = k_sphere_radius(prep, cfg.precision);
        let ledger = reserve_build_memory(cell, mesh, nao, kbytes, gcut, cfg)?;

        let moduli = mesh.residue_moduli();
        let flip = cfg.mutation == Some(KMutation::PhaseSignInPairFt);
        let phr = residue_phases(mesh, moduli, flip);

        let zero_blk = || Array2::<Complex64>::zeros((n2, n2));
        let mut jker: Vec<Array2<Complex64>> = (0..nk * nk).map(|_| zero_blk()).collect();
        let mut kker: Vec<Array2<Complex64>> = (0..nk * nk).map(|_| zero_blk()).collect();
        let thresh = 0.1 * cfg.precision;
        let pass = QPass {
            cell,
            prep,
            mesh,
            cfg,
            phr: &phr,
            moduli,
            gcut,
            thresh,
            chunk_budget: ledger.remaining().min(G_CHUNK_BYTES),
            // per K: N_k P columns + scaled copy + conjugate transpose.
            extra_per_g: bytes_of(((nk + 2) * n2) as u64, 16),
        };
        let mut n_k_total = 0usize;
        let mut n_q_passes = 0usize;

        for iq in 0..nk {
            if mesh.q_minus(iq) < iq {
                continue; // filled by time reversal from pass iqm
            }
            n_k_total += pass.run(iq, &mut jker, &mut kker)?;
            n_q_passes += 1;
        }
        let madelung_ewald = ewald_madelung(cell, mesh, cfg)?;
        let madelung = match exxdiv {
            ExxDiv::None => 0.0,
            ExxDiv::Ewald => madelung_ewald,
        };
        ferric_core::memory::warn_if_rss_over("ferric-pbc KDenseAftEri", ledger.budget(), 1.1);
        Ok(Self {
            nao,
            nk,
            jker,
            kker,
            s: s_k.to_vec(),
            madelung,
            madelung_ewald,
            n_k_total,
            n_q_passes,
            gcut,
            pair_thresh: thresh,
        })
    }

    /// The same kernels with another exchange-divergence treatment.
    pub fn with_exxdiv(mut self, exxdiv: ExxDiv) -> Self {
        self.madelung = match exxdiv {
            ExxDiv::None => 0.0,
            ExxDiv::Ewald => self.madelung_ewald,
        };
        self
    }

    /// `v_M` applied in K (0 for `ExxDiv::None`).
    pub fn madelung(&self) -> f64 {
        self.madelung
    }

    /// The mesh (supercell) Madelung constant `ExxDiv::Ewald` applies.
    pub fn madelung_ewald(&self) -> f64 {
        self.madelung_ewald
    }

    /// Number of k-points.
    pub fn nk(&self) -> usize {
        self.nk
    }

    /// Total K vectors summed over the computed q passes.
    pub fn n_k_total(&self) -> usize {
        self.n_k_total
    }

    /// q passes computed (the rest came from time reversal).
    pub fn n_q_passes(&self) -> usize {
        self.n_q_passes
    }

    /// K-sphere radius (Bohr⁻¹).
    pub fn gcut(&self) -> f64 {
        self.gcut
    }

    /// Primitive-pair screen of the ERI pair FT (`0.1 · precision`); the
    /// k-point forces ([`crate::kgrad`]) reuse it.
    pub fn pair_thresh(&self) -> f64 {
        self.pair_thresh
    }

    /// Coulomb kernel block `[k, k']`, `(nao², nao²)`.
    pub fn jker(&self, k: usize, kp: usize) -> &Array2<Complex64> {
        &self.jker[k * self.nk + kp]
    }

    /// Exchange kernel block `[k, k']`, `(nao², nao²)`.
    pub fn kker(&self, k: usize, kp: usize) -> &Array2<Complex64> {
        &self.kker[k * self.nk + kp]
    }

    /// J/K builder borrowing the kernels, with this tensor's `v_M`.
    pub fn jk_builder(&self) -> KDenseAftJk<'_> {
        KDenseAftJk {
            eri: self,
            madelung: self.madelung,
        }
    }

    /// J/K builder with an explicit `v_M` (0 = `exxdiv=None`).
    pub fn jk_builder_with_madelung(&self, madelung: f64) -> KDenseAftJk<'_> {
        KDenseAftJk {
            eri: self,
            madelung,
        }
    }

    /// `J(k)`, `K(k)` (K including `v_M S D S`) for densities `dm` (one per
    /// mesh point, Hermitian). Overwrites `j`, `k`.
    pub fn contract(
        &self,
        dm: &[Array2<Complex64>],
        madelung: f64,
        j: &mut [Array2<Complex64>],
        k: &mut [Array2<Complex64>],
    ) -> Result<(), FerricError> {
        let (n, nk) = (self.nao, self.nk);
        if dm.len() != nk || j.len() != nk || k.len() != nk {
            return Err(FerricError::General(format!(
                "KDenseAftJk: {} densities / {} J / {} K, expected {nk}",
                dm.len(),
                j.len(),
                k.len()
            )));
        }
        for x in dm.iter().chain(j.iter()).chain(k.iter()) {
            if x.dim() != (n, n) {
                return Err(FerricError::General(format!(
                    "KDenseAftJk: matrix of shape {:?}, expected ({n}, {n})",
                    x.dim()
                )));
            }
        }
        let inv = 1.0 / nk as f64;
        let zero = Complex64::new(0.0, 0.0);
        for kk in 0..nk {
            let jk = &mut j[kk];
            let kx = &mut k[kk];
            jk.fill(zero);
            kx.fill(zero);
            for kp in 0..nk {
                let jb = &self.jker[kk * nk + kp];
                let kb = &self.kker[kk * nk + kp];
                let d = &dm[kp];
                for m in 0..n {
                    for nu in 0..n {
                        let (mut aj, mut ak) = (zero, zero);
                        for l in 0..n {
                            for s in 0..n {
                                let dls = d[(l, s)];
                                aj += jb[(m * n + nu, l * n + s)] * dls;
                                ak += kb[(m * n + l, nu * n + s)] * dls;
                            }
                        }
                        jk[(m, nu)] += aj * inv;
                        kx[(m, nu)] += ak * inv;
                    }
                }
            }
            if madelung != 0.0 {
                let s = &self.s[kk];
                let sds = s.dot(&dm[kk]).dot(s);
                kx.scaled_add(Complex64::new(madelung, 0.0), &sds);
            }
        }
        Ok(())
    }
}

/// [`KPointJk`] over a [`KDenseAftEri`], including the Madelung shift.
pub struct KDenseAftJk<'a> {
    eri: &'a KDenseAftEri,
    madelung: f64,
}

impl KPointJk for KDenseAftJk<'_> {
    /// Contract the per-k densities `dm` into `j` and `k` (Madelung shift included in `k`).
    fn build(
        &mut self,
        dm: &[Array2<Complex64>],
        j: &mut [Array2<Complex64>],
        k: &mut [Array2<Complex64>],
    ) -> Result<(), FerricError> {
        self.eri.contract(dm, self.madelung, j, k)
    }
}
