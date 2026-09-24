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
//! k-point exchange needs complex RS-GDF per q (not implemented; see
//! `kscf` module doc).

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
}

impl Default for KDenseAftConfig {
    fn default() -> Self {
        Self {
            precision: DEFAULT_DENSE_AFT_PRECISION,
            max_bytes: DEFAULT_DENSE_AFT_MAX_BYTES,
            budget_bytes: None,
            mutation: None,
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
        let kbytes = 2u128 * (nk as u128).pow(2) * (n2 as u128).pow(2) * 16;
        if kbytes > cfg.max_bytes as u128 {
            return Err(FerricError::General(format!(
                "KDenseAftEri: dense k-point kernels need {kbytes} bytes (nao = {nao}, N_k = {nk}) \
                 > cap {} bytes; this is a toy-cell oracle (k-point RS-GDF is not implemented)",
                cfg.max_bytes
            )));
        }
        if s_k.len() != nk || s_k.iter().any(|s| s.dim() != (nao, nao)) {
            return Err(FerricError::General(format!(
                "KDenseAftEri: need {nk} overlap matrices of shape ({nao}, {nao})"
            )));
        }
        if !(f64::MIN_POSITIVE..1.0).contains(&cfg.precision) {
            return Err(FerricError::General(format!(
                "KDenseAftEri: precision must lie in (0, 1), got {}",
                cfg.precision
            )));
        }
        let pmax = 2.0
            * prep
                .located_shells()
                .iter()
                .flat_map(|sh| sh.exponents.iter().copied())
                .fold(0.0_f64, f64::max);
        let gcut = 2.0 * (pmax * (1.0 / cfg.precision).ln()).sqrt();
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
            .map(|iq| {
                let q = mesh.q_class(iq).0;
                (q[0] * q[0] + q[1] * q[1] + q[2] * q[2]).sqrt()
            })
            .fold(0.0_f64, f64::max);
        // Cell::gvectors at gcut + |q| plus the K copy (same bound).
        ledger.reserve(
            &format!("KDenseAftEri K list (|K| <= {gcut:.3}, |q| <= {qmax:.3})"),
            gvector_list_bytes(cell, gcut + qmax)?.saturating_mul(2),
        )?;

        let moduli = mesh.residue_moduli();
        let nr = moduli[0] * moduli[1] * moduli[2];
        let flip = cfg.mutation == Some(KMutation::PhaseSignInPairFt);
        let phr: Vec<Vec<Complex64>> = (0..nk)
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
            .collect();

        let zero_blk = || Array2::<Complex64>::zeros((n2, n2));
        let mut jker: Vec<Array2<Complex64>> = (0..nk * nk).map(|_| zero_blk()).collect();
        let mut kker: Vec<Array2<Complex64>> = (0..nk * nk).map(|_| zero_blk()).collect();
        let vol = cell.volume();
        let thresh = 0.1 * cfg.precision;
        // per K: N_k P columns + scaled copy + conjugate transpose.
        let extra_per_g = bytes_of(((nk + 2) * n2) as u64, 16);
        let mut n_k_total = 0usize;
        let mut n_q_passes = 0usize;
        let g2cut = gcut * gcut;

        for iq in 0..nk {
            let iqm = mesh.q_minus(iq);
            if iqm < iq {
                continue; // filled by time reversal from pass iqm
            }
            let (q, mq) = mesh.q_class(iq);
            let qn = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2]).sqrt();
            let ks: Vec<[f64; 3]> = cell
                .gvectors(gcut + qn)?
                .into_iter()
                .map(|g| [g[0] + q[0], g[1] + q[1], g[2] + q[2]])
                .filter(|k| {
                    let k2 = k[0] * k[0] + k[1] * k[1] + k[2] * k[2];
                    k2 > 0.0 && k2 <= g2cut
                })
                .collect();
            n_k_total += ks.len();
            n_q_passes += 1;
            let kof: Vec<usize> = (0..nk).map(|j| mesh.k_minus_q(j, mq)).collect();
            let minus: Vec<usize> = (0..nk).map(|k| mesh.minus(k)).collect();
            let kernel_at_g = cfg.mutation == Some(KMutation::KernelAtG);
            let chunk_budget = ledger.remaining().min(G_CHUNK_BYTES);
            let jref = &mut jker;
            let kref = &mut kker;
            let phr = &phr;
            let sink = |_k0: usize,
                        kv: &[[f64; 3]],
                        qs: &[Array3<Complex64>]|
             -> Result<(), FerricError> {
                let ng = kv.len();
                let vv: Vec<f64> = kv
                    .iter()
                    .map(|k| {
                        let x = if kernel_at_g {
                            [k[0] - q[0], k[1] - q[1], k[2] - q[2]]
                        } else {
                            *k
                        };
                        let x2 = x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
                        if x2 > 1e-20 {
                            4.0 * PI / x2 / vol
                        } else {
                            0.0
                        }
                    })
                    .collect();
                // P^{k'}(K) for every k', as (nao², n_K).
                let pj: Vec<Array2<Complex64>> = (0..nk)
                    .map(|j| {
                        let mut p = Array2::<Complex64>::zeros((n2, ng));
                        for (qr, ph) in qs.iter().zip(&phr[j]) {
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
                    .collect();
                let scaled = |p: &Array2<Complex64>| {
                    let mut pv = p.clone();
                    for (mut col, v) in pv.columns_mut().into_iter().zip(&vv) {
                        col.mapv_inplace(|z| z * *v);
                    }
                    pv
                };
                let herm = |p: &Array2<Complex64>| p.t().mapv(|z| z.conj());
                for j in 0..nk {
                    let blk = scaled(&pj[j]).dot(&herm(&pj[j]));
                    let k = kof[j];
                    kref[k * nk + j] += &blk;
                    if iqm != iq {
                        kref[minus[k] * nk + minus[j]] += &blk.mapv(|z| z.conj());
                    }
                }
                if iq == 0 {
                    for k in 0..nk {
                        let pv = scaled(&pj[k]);
                        for j in 0..nk {
                            jref[k * nk + j] += &pv.dot(&herm(&pj[j]));
                        }
                    }
                }
                Ok(())
            };
            pair_ft_residues_chunked(
                cell,
                prep,
                &ks,
                moduli,
                thresh,
                chunk_budget,
                extra_per_g,
                sink,
            )?;
        }
        let madelung_ewald = if cfg.mutation == Some(KMutation::PrimitiveMadelung) {
            madelung_constant(cell)?
        } else {
            mesh.madelung(cell)?
        };
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
    fn build(
        &mut self,
        dm: &[Array2<Complex64>],
        j: &mut [Array2<Complex64>],
        k: &mut [Array2<Complex64>],
    ) -> Result<(), FerricError> {
        self.eri.contract(dm, self.madelung, j, k)
    }
}
