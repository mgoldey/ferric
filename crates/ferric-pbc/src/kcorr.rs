//! Stage 9: k-point closed-shell MP2 ([`kpoint_mp2`]) and direct RPA
//! ([`kpoint_drpa`]) on a k-point RHF ([`crate::kscf::KScfResult`]). Rust
//! port of `reference/pbc/pbc_kcorr.py` (FINDINGS "Iteration 12 (Python,
//! k-point MP2/dRPA)" and its "For the Rust port (stage 9)" section, which
//! this module follows).
//!
//! # Conventions (Iteration 9 Bloch AOs, per-cell normalised `C(k)`)
//!
//! ```text
//! B(k, k')[P, ml]   the pair tensor of class q = k' − k (aux P of class q), Kker ≈ B^T conj(B)
//! Bov_ia(k, k')     = Σ_ml conj C_i(k)_m  B_ml(k, k')  C_a(k')_l
//! Bvo_bj(kb, kj)    = Σ_ns conj C_b(kb)_n B_ns(kb, kj) C_j(kj)_s          (built EXPLICITLY, see below)
//! V[ki,kj,ka]_iajb  = (i ki a ka | j kj b kb) = Σ_P Bov_ia(ki,ka) conj Bvo_bj(kb,kj),  kb = ki + kj − ka
//! E_MP2 / cell      = (1/N_k³) Σ_{ki,kj,ka} Σ_ijab conj(V_iajb) [2 V_iajb − V_ibja] / (ε_i + ε_j − ε_a − ε_b)
//! Π(q, iω)          = (4/N_k) Σ_k' Σ_ia Bov_ia(k'−q, k') conj Bov_ia(k'−q, k')ᵀ e/(ω² + e²),  e = ε_a(k') − ε_i(k'−q)
//! E_dRPA / cell     = (1/N_k) Σ_q (1/2π) ∫_0^∞ dω [ln det(1 + Π(q)) − tr Π(q)]
//! ```
//!
//! Both legs of `V` lie in the SAME q class (`ka − ki = kj − kb`), so they
//! share the class's aux set. With these normalisations an `N₁×N₂×N₃`
//! Gamma-centred mesh is, term by term, the Gamma MP2/dRPA of the explicit
//! `diag(N)` supercell per cell (prototype: exact, 2e-15).
//!
//! `Bvo` is transformed from `B(kb, kj)` itself — it is NOT derived by
//! conjugating the `−q` class: that shortcut needs the time-reversal aux
//! gauge `U(−q) = conj U(q)` AND `C(−k) = conj C(k)`, and nothing forces an
//! SCF (or a caller-built `KScfResult`) to deliver the second.
//!
//! # Integral sources ([`KCorrIntegrals`])
//!
//! * [`KCorrIntegrals::RsGdf`] — production: the SCF's own complex
//!   [`KRsGdf`] blocks `B(k, k')` (`KRsGdf::block`).
//! * [`KCorrIntegrals::DenseAft`] — TEST/ORACLE ONLY: [`KDenseAftPairs`], the
//!   exact pure-AFT pair tensors. [`crate::kdense_aft::KDenseAftEri`] cannot
//!   serve here: it stores only the contracted exchange kernels
//!   `Kker[k,k'] = Σ_K v P^{kk'} (P^{kk'})ᴴ` (the SAME pair on both sides),
//!   while `V` needs the cross-pair `Σ_K v P^{ki ka} (P^{kb kj})ᴴ` and `Π(q)`
//!   needs every pair of the class. `KDenseAftPairs` accumulates the full
//!   per-class Gram `G_q[(k',ml),(k'',ns)] = (1/Ω) Σ_{K∈G+q,K≠0} v(K)
//!   P^{k'}_ml(K) conj P^{k''}_ns(K)` over the SAME K sphere, pair screen and
//!   phases as `KDenseAftEri` and factors it exactly (`eigh_herm`,
//!   `B = Λ^{1/2} Uᵀ`, rows with `λ <= 1e-15 λ_max` dropped), so its diagonal
//!   blocks reproduce `KDenseAftEri::kker` (asserted in the tests).
//!
//! # dRPA energy ([`KDrpaEnergy`]): REALIFY, not a zheev summand
//!
//! `Π(q)` is complex Hermitian. [`KDrpaEnergy::Quadrature`] (production)
//! REALIFIES it: with `Bq = X + iY` (rows P, columns (k', i, a), scaled by
//! `N_k^{-1/2}`), the real matrix `B_R = [[X, −Y], [Y, X]]` with every
//! excitation energy duplicated gives `B_R F B_Rᵀ = [[Re Π, −Im Π], [Im Π,
//! Re Π]]`, whose spectrum is that of `Π` with every eigenvalue doubled, so
//! `E_q = ½ ×` the unchanged real pipeline
//! ([`crate::drpa::drpa_from_b_ov`] → ferric-rpa `run_pdep_rpa_from_parts`:
//! full-rank dense eigensolve, the same GL nodes and log-det summand as
//! `gamma_drpa`). The real path takes `(naux, nocc·nvir)` with
//! `e_ia = ε_a − ε_i`; the class's excitations are NOT a product set
//! (`ε_a(k') − ε_i(k'−q)`), so they are passed as `nocc = 1`, `ε_occ = [0]`,
//! `ε_vir = e_c` (column c). Chosen over a new complex summand because it
//! reuses the validated energy integrator verbatim; the cost is 4× the
//! memory of `Bq` and 8× the eigensolve flops (toy-to-small scale today).
//! [`KDrpaEnergy::Plasmon`] (TEST/ORACLE) is the INDEPENDENT construction:
//! the per-q frequency integral done analytically on the ov Gram,
//! `E_q = ½ [Σ √eig(D² + 4 D^{1/2} K_q D^{1/2}) − tr D − 2 tr K_q]`, with
//! `kscf::eigh_herm` (never ndarray-linalg `eigh` on a row-major
//! complex matrix). [`KDrpaEnergy::SecondOrder`] (TEST/ORACLE) is the
//! `O(Π²)` term, which must equal [`KMp2Result::e_direct`].
//!
//! # Denominators (config honesty; the Gamma table of [`crate::mp2`])
//!
//! A `KScfResult` does not record its exxdiv, so the caller STATES it
//! (`reference_exxdiv`); [`crate::mp2::occupied_shift`] with the MESH
//! (supercell) Madelung constant turns it into the requested
//! [`Mp2Denominators`]. `exxdiv = ewald` moves every occupied level of every
//! k by `−v_M(mesh)` at fixed C (Iteration 9), so the conversion is exact.
//! `MadelungShifted` is the physical convention (N_k⁻¹ convergence);
//! `Unshifted` is what PySCF KMP2 gives after an `exxdiv=None` KRHF
//! (N_k^{−1/3}, 10 % of E_corr at 4³ in the prototype).
//!
//! # Scope
//!
//! Closed shell, uniform occupation (every k must hold the same number of
//! doubly occupied levels — the global aufbau of `solve_krhf` can violate
//! it; refused). `nvir` may vary per k (lindep cut). Frozen core per k
//! through `active_occ`. Gamma-centred meshes have the supercell anchor;
//! shifted (MP, even N) meshes run through the same code, unanchored.
//! NOT measured: meshes > 3 per axis in Rust, real aux fitting error at
//! scale, cost, the q = 0 head correction (not ported — the prototype's
//! fixed-k head is exact only for flat bands).
//!
//! # Memory
//!
//! Every buffer is reserved on a [`crate::budget`] ledger first. MP2: all
//! `N_k²` explicit `Bvo` blocks, one ki row of `Bov`, one ki slice of `V`
//! (`N_k² (nocc·nvir)²`, the exchange partner `V[ki,kj,kb]` lives in the same
//! slice). dRPA: one class's `Bq`, its realified copy (or the plasmon Gram).
//! ferric-rpa then runs its own ceiling check on the whole budget.
//!
//! Units: Bohr and Hartree.

use crate::budget::{bytes_of, Ledger};
use crate::dense_aft::ExxDiv;
use crate::drpa::{drpa_from_b_ov, validate_quad_points, GAMMA_DRPA_QUAD_U0};
use crate::hcore::kpoint::hermitize;
use crate::hcore::{gvector_list_bytes, G_CHUNK_BYTES};
use crate::kdense_aft::KDenseAftConfig;
use crate::kpts::KPointMesh;
use crate::kscf::{eigh_herm, KScfResult};
use crate::lattice::Cell;
use crate::mp2::{occupied_shift, Mp2Denominators};
use crate::pair_ft::residues::{pair_ft_residues_chunked, residue_coords};
use crate::rsgdf::kpoint::KRsGdf;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_mp2::rimp2::active_occ;
use ndarray::linalg::general_mat_mul;
use ndarray::{s, Array2, Array3, ArrayView2};
use num_complex::Complex64 as C64;
use std::f64::consts::PI;

/// Relative eigenvalue cut of the [`KDenseAftPairs`] Gram factorisation
/// (drops the roundoff-negative and numerically-null directions only).
pub const PAIR_GRAM_REL_CUT: f64 = 1e-15;

/// Default frequency grid (the Gamma dRPA's, same reason).
pub const DEFAULT_KDRPA_QUAD_POINTS: usize = crate::drpa::DEFAULT_GAMMA_DRPA_QUAD_POINTS;

/// TEST-ONLY deliberate defects (the prototype's `_MUTANT` switch plus its
/// normalisation code mutation). Never set in production; each must break
/// the k-mesh ≡ supercell anchor (`tests/pbc_kcorr.rs`).
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KCorrMutation {
    /// MP2: `kb = ki − kj + ka` in the denominator and the exchange lookup
    /// (V itself stays correct, as in the prototype). dRPA untouched.
    KbIndex,
    /// MP2: no conjugation on the second leg (`V = Bovᵀ Bvo`).
    NoConj,
    /// Both: the occupied Madelung shift is never applied (occ shift 0
    /// whatever the stated reference / requested convention).
    MissingMadelungShift,
    /// dRPA: occupied energy of `k' + q` instead of `k' − q` in `Π(q)`.
    RpaOccupiedKPlusQ,
    /// Both: MP2 `/N_k²` instead of `/N_k³`; `Π` without its `1/N_k`.
    /// Invisible at N_k = 1 by construction (the artifact hypothesis).
    NkNormalization,
}

// ============================================================ dense oracle

/// TEST/ORACLE ONLY: exact pure-AFT pair tensors `B(k, k')` per
/// momentum-transfer class, from the factorised per-class Gram (module
/// doc). Memory `N_k (N_k nao²)²` complex: toy cells only (hard-capped by
/// [`KDenseAftConfig::max_bytes`]).
#[derive(Debug, Clone)]
pub struct KDenseAftPairs {
    nao: usize,
    nk: usize,
    /// Per class iq: `b[j] = B(kof[j], j)`, `(rank_q, nao²)`.
    classes: Vec<Vec<Array2<C64>>>,
    /// `[k * N_k + k']` → (class, slot j = k').
    pair_loc: Vec<(usize, usize)>,
    rank: Vec<usize>,
    kpts: Vec<[f64; 3]>,
    madelung_ewald: f64,
    n_k_total: usize,
    gcut: f64,
}

impl KDenseAftPairs {
    /// Build every class explicitly (no time-reversal fill) with the
    /// K sphere, pair screen and Bloch phases of
    /// [`crate::kdense_aft::KDenseAftEri::build`] at the same `cfg`
    /// (`cfg.mutation` must be `None`).
    pub fn build(
        cell: &Cell,
        prep: &PreparedBasis,
        mesh: &KPointMesh,
        cfg: &KDenseAftConfig,
    ) -> Result<Self, FerricError> {
        if cfg.mutation.is_some() {
            return Err(FerricError::General(
                "KDenseAftPairs: KMutation is a KDenseAftEri test switch; not supported here"
                    .into(),
            ));
        }
        let nao = prep.nbasis();
        let nk = mesh.nk();
        let n2 = nao * nao;
        let m = nk * n2;
        let m2 = (m as u128) * (m as u128);
        let stored = (nk as u128) * m2 * 16;
        let work = 4 * m2 * 16;
        if stored + work > cfg.max_bytes as u128 {
            return Err(FerricError::General(format!(
                "KDenseAftPairs: dense pair tensors need {} bytes (nao = {nao}, N_k = {nk}) > cap {} \
                 bytes; this is a toy-cell oracle (use KRsGdf)",
                stored + work,
                cfg.max_bytes
            )));
        }
        if !(f64::MIN_POSITIVE..1.0).contains(&cfg.precision) {
            return Err(FerricError::General(format!(
                "KDenseAftPairs: precision must lie in (0, 1), got {}",
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
            &format!(
                "KDenseAftPairs factored B, all classes (nao = {nao}, N_k = {nk}, upper bound)"
            ),
            usize::try_from(stored).unwrap_or(usize::MAX),
        )?;
        ledger.reserve(
            &format!("KDenseAftPairs per-class Gram + eigh copies (N_k·nao² = {m})"),
            usize::try_from(work).unwrap_or(usize::MAX),
        )?;
        let qmax = (0..nk)
            .map(|iq| {
                let q = mesh.q_class(iq).0;
                (q[0] * q[0] + q[1] * q[1] + q[2] * q[2]).sqrt()
            })
            .fold(0.0_f64, f64::max);
        ledger.reserve(
            &format!("KDenseAftPairs K list (|K| <= {gcut:.3}, |q| <= {qmax:.3})"),
            gvector_list_bytes(cell, gcut + qmax)?.saturating_mul(2),
        )?;

        let moduli = mesh.residue_moduli();
        let nr = moduli[0] * moduli[1] * moduli[2];
        let phr: Vec<Vec<C64>> = (0..nk)
            .map(|k| {
                (0..nr)
                    .map(|r| mesh.phase(k, residue_coords(r, moduli)))
                    .collect()
            })
            .collect();
        let vol = cell.volume();
        let thresh = 0.1 * cfg.precision;
        let g2cut = gcut * gcut;
        // Per K: one column of A and one of Aᴴ.
        let extra_per_g = bytes_of((2 * m) as u64, 16);
        let one = C64::new(1.0, 0.0);

        let mut classes = Vec::with_capacity(nk);
        let mut rank = Vec::with_capacity(nk);
        let mut pair_loc = vec![(usize::MAX, usize::MAX); nk * nk];
        let mut n_k_total = 0usize;
        for iq in 0..nk {
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
            let kof: Vec<usize> = (0..nk).map(|j| mesh.k_minus_q(j, mq)).collect();
            let mut g = Array2::<C64>::zeros((m, m));
            {
                let gref = &mut g;
                let phr = &phr;
                let sink =
                    |_k0: usize, kv: &[[f64; 3]], qs: &[Array3<C64>]| -> Result<(), FerricError> {
                        let ng = kv.len();
                        let sv: Vec<f64> = kv
                            .iter()
                            .map(|k| {
                                let k2 = k[0] * k[0] + k[1] * k[1] + k[2] * k[2];
                                if k2 > 1e-20 {
                                    (4.0 * PI / k2 / vol).sqrt()
                                } else {
                                    0.0
                                }
                            })
                            .collect();
                        // A[(k', ml), K] = √(v(K)/Ω) P^{k'}_ml(K), pair (k'−q, k').
                        let mut a = Array2::<C64>::zeros((m, ng));
                        for (j, ph_j) in phr.iter().enumerate() {
                            for (qr, ph) in qs.iter().zip(ph_j) {
                                for mm in 0..nao {
                                    for l in 0..nao {
                                        let row = j * n2 + mm * nao + l;
                                        for gi in 0..ng {
                                            a[(row, gi)] += *ph * qr[[mm, l, gi]];
                                        }
                                    }
                                }
                            }
                        }
                        for (mut col, s) in a.columns_mut().into_iter().zip(&sv) {
                            col.mapv_inplace(|z| z * *s);
                        }
                        let ah = a.t().mapv(|z| z.conj());
                        general_mat_mul(one, &a, &ah, one, &mut *gref);
                        Ok(())
                    };
                let chunk_budget = ledger.remaining().min(G_CHUNK_BYTES);
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
            let (w, u) = eigh_herm(&hermitize(&g)).map_err(|e| {
                FerricError::Lapack(format!("KDenseAftPairs: Gram eigh at q class {iq}: {e}"))
            })?;
            let wmax = w.iter().copied().fold(0.0_f64, f64::max);
            let kept: Vec<usize> = (0..m)
                .filter(|&a| w[a] > PAIR_GRAM_REL_CUT * wmax)
                .collect();
            if kept.is_empty() {
                return Err(FerricError::General(format!(
                    "KDenseAftPairs: q class {iq} has an empty pair Gram (max eigenvalue {wmax:e})"
                )));
            }
            let r = kept.len();
            let mut b: Vec<Array2<C64>> = (0..nk).map(|_| Array2::zeros((r, n2))).collect();
            for (row, &a) in kept.iter().enumerate() {
                let sq = w[a].sqrt();
                for (j, bj) in b.iter_mut().enumerate() {
                    for x in 0..n2 {
                        bj[(row, x)] = u[(j * n2 + x, a)] * sq;
                    }
                }
            }
            for (j, &k) in kof.iter().enumerate() {
                pair_loc[k * nk + j] = (iq, j);
            }
            classes.push(b);
            rank.push(r);
        }
        if pair_loc.iter().any(|&(c, _)| c == usize::MAX) {
            return Err(FerricError::General(
                "KDenseAftPairs: internal error — a (k, k') pair was not covered by any q class"
                    .into(),
            ));
        }
        Ok(Self {
            nao,
            nk,
            classes,
            pair_loc,
            rank,
            kpts: mesh.kpts().to_vec(),
            madelung_ewald: mesh.madelung(cell)?,
            n_k_total,
            gcut,
        })
    }

    /// `B(k, k')`, `(rank_q, nao²)`, row a, column `m·nao + l`, so that
    /// `B(k,k')ᵀ conj B(k,k')` is `KDenseAftEri::kker(k, k')`.
    pub fn block(&self, k: usize, kp: usize) -> &Array2<C64> {
        let (c, j) = self.pair_loc[k * self.nk + kp];
        &self.classes[c][j]
    }

    /// Rows kept for class `iq` ([`KPointMesh::q_class`] order).
    pub fn rank(&self, iq: usize) -> usize {
        self.rank[iq]
    }

    /// Number of k-points.
    pub fn nk(&self) -> usize {
        self.nk
    }

    /// AO count.
    pub fn nao(&self) -> usize {
        self.nao
    }

    /// Cartesian k-points of the mesh it was built on.
    pub fn kpts(&self) -> &[[f64; 3]] {
        &self.kpts
    }

    /// The mesh (supercell) Madelung constant.
    pub fn madelung_ewald(&self) -> f64 {
        self.madelung_ewald
    }

    /// Total K vectors summed over all classes.
    pub fn n_k_total(&self) -> usize {
        self.n_k_total
    }

    /// K-sphere radius (Bohr⁻¹).
    pub fn gcut(&self) -> f64 {
        self.gcut
    }
}

// ============================================================ inputs

/// Where the pair tensors come from.
#[derive(Debug, Clone, Copy)]
pub enum KCorrIntegrals<'a> {
    /// Production: the SCF's own complex RS-GDF blocks.
    RsGdf(&'a KRsGdf),
    /// TEST/ORACLE ONLY: exact dense pure-AFT pair tensors.
    DenseAft(&'a KDenseAftPairs),
}

impl KCorrIntegrals<'_> {
    fn nk(&self) -> usize {
        match self {
            KCorrIntegrals::RsGdf(g) => g.nk(),
            KCorrIntegrals::DenseAft(d) => d.nk(),
        }
    }

    fn madelung_ewald(&self) -> f64 {
        match self {
            KCorrIntegrals::RsGdf(g) => g.madelung_ewald(),
            KCorrIntegrals::DenseAft(d) => d.madelung_ewald(),
        }
    }

    /// Largest row count of any class (budget sizing).
    fn max_rows(&self) -> usize {
        match self {
            KCorrIntegrals::RsGdf(g) => g
                .stats()
                .per_q
                .iter()
                .map(|q| q.naux_kept)
                .max()
                .unwrap_or(0),
            KCorrIntegrals::DenseAft(d) => d.rank.iter().copied().max().unwrap_or(0),
        }
    }

    fn block(&self, k: usize, kp: usize, nao: usize) -> Result<Array2<C64>, FerricError> {
        let b = match self {
            KCorrIntegrals::RsGdf(g) => g.block(k, kp),
            KCorrIntegrals::DenseAft(d) => d.block(k, kp).clone(),
        };
        if b.ncols() != nao * nao {
            return Err(FerricError::General(format!(
                "k-point correlation: B({k},{kp}) has {} columns, the reference has nao = {nao}",
                b.ncols()
            )));
        }
        Ok(b)
    }
}

/// `out[P, x·n2 + y] = Σ_ml conj(left[m,x]) B_P[m,l] right[l,y]`, the
/// `(rows, n1·n2)` layout. Every `dot` output is read by index only (a
/// complex `dot` may come back column-major).
fn half_transform(
    b: &Array2<C64>,
    nao: usize,
    left: ArrayView2<'_, C64>,
    right: ArrayView2<'_, C64>,
) -> Result<Array2<C64>, FerricError> {
    let rows = b.nrows();
    let (n1, n2) = (left.ncols(), right.ncols());
    let b_std = b.as_standard_layout();
    let flat = b_std
        .as_slice()
        .ok_or_else(|| FerricError::General("k-point correlation: B not contiguous".into()))?;
    let bv = ArrayView2::from_shape((rows * nao, nao), flat)
        .map_err(|e| FerricError::General(format!("k-point correlation: B reshape: {e}")))?;
    let x = bv.dot(&right); // (rows·nao, n2)
    let lh = left.t().mapv(|z| z.conj()); // (n1, nao)
    let mut out = Array2::<C64>::zeros((rows, n1 * n2));
    for p in 0..rows {
        let xp = x.slice(s![p * nao..(p + 1) * nao, ..]);
        let ov = lh.dot(&xp); // (n1, n2)
        for i in 0..n1 {
            for a in 0..n2 {
                out[(p, i * n2 + a)] = ov[(i, a)];
            }
        }
    }
    Ok(out)
}

/// Mesh index of `k1 + s2·k2 + s3·k3` (numerators over 2N).
fn k_combo(mesh: &KPointMesh, k1: usize, (s2, k2): (i64, usize), (s3, k3): (i64, usize)) -> usize {
    let (a, b, c) = (
        mesh.numerators(k1),
        mesh.numerators(k2),
        mesh.numerators(k3),
    );
    mesh.index_of_nums([
        a[0] + s2 * b[0] + s3 * c[0],
        a[1] + s2 * b[1] + s3 * c[1],
        a[2] + s2 * b[2] + s3 * c[2],
    ])
    .expect("a signed sum of three mesh points with net coefficient 1 is on the mesh")
}

/// Validated reference: active orbitals and shifted denominators per k.
struct KRef {
    nk: usize,
    nao: usize,
    nocc: usize,
    co: Vec<Array2<C64>>,
    cv: Vec<Array2<C64>>,
    eo: Vec<Vec<f64>>,
    ev: Vec<Vec<f64>>,
    madelung: f64,
    occ_shift: f64,
}

#[allow(clippy::too_many_arguments)]
fn prepare(
    what: &str,
    cell: &Cell,
    mesh: &KPointMesh,
    kscf: &KScfResult,
    ints: &KCorrIntegrals<'_>,
    frozen_core: usize,
    reference_exxdiv: ExxDiv,
    denominators: Mp2Denominators,
    mutation: Option<KCorrMutation>,
) -> Result<KRef, FerricError> {
    let err = |m: String| Err(FerricError::General(format!("{what}: {m}")));
    let nk = mesh.nk();
    if !kscf.converged {
        return err("the k-point RHF reference did not converge".into());
    }
    if ints.nk() != nk {
        return err(format!(
            "the integrals were built for {} k-points, the mesh has {nk}",
            ints.nk()
        ));
    }
    if let KCorrIntegrals::DenseAft(d) = ints {
        if !same_kpts(d.kpts(), mesh.kpts()) {
            return err("the dense pair tensors were built on a different mesh".into());
        }
    }
    if kscf.eps.len() != nk
        || kscf.mos.len() != nk
        || kscf.occupations.len() != nk
        || kscf.nocc_per_k.len() != nk
        || !same_kpts(&kscf.kpts, mesh.kpts())
    {
        return err(format!(
            "the reference does not match the {nk}-point mesh (eps {}, mos {}, occupations {}, \
             k-points {})",
            kscf.eps.len(),
            kscf.mos.len(),
            kscf.occupations.len(),
            kscf.kpts.len()
        ));
    }
    let nelec = cell.mol().nelec();
    if nelec <= 0 || nelec % 2 != 0 {
        return err(format!(
            "closed shell needs an even, positive electron count per cell (got {nelec})"
        ));
    }
    let nocc_total = (nelec / 2) as usize;
    if kscf.nocc_per_k.iter().any(|&n| n != nocc_total) {
        return err(format!(
            "non-uniform occupation {:?} (need {nocc_total} doubly occupied levels at EVERY k; \
             the global aufbau moved electrons between k-points)",
            kscf.nocc_per_k
        ));
    }
    let nocc = active_occ(nocc_total, frozen_core)?;
    let nao = kscf.mos[0].nrows();
    let madelung = ints.madelung_ewald();
    let occ_shift = if mutation == Some(KCorrMutation::MissingMadelungShift) {
        0.0
    } else {
        occupied_shift(reference_exxdiv, denominators, madelung)
    };
    let mut co = Vec::with_capacity(nk);
    let mut cv = Vec::with_capacity(nk);
    let mut eo = Vec::with_capacity(nk);
    let mut ev = Vec::with_capacity(nk);
    for k in 0..nk {
        let c = &kscf.mos[k];
        let e = &kscf.eps[k];
        let nmo = c.ncols();
        if c.nrows() != nao || e.len() != nmo || kscf.occupations[k].len() != nmo {
            return err(format!(
                "k = {k}: C {:?}, {} eps, {} occupations inconsistent (nao = {nao})",
                c.dim(),
                e.len(),
                kscf.occupations[k].len()
            ));
        }
        if nmo <= nocc_total {
            return err(format!("k = {k}: no virtual orbitals ({nmo} MOs)"));
        }
        if kscf.occupations[k][..nocc_total].iter().any(|&o| o != 2.0)
            || kscf.occupations[k][nocc_total..].iter().any(|&o| o != 0.0)
        {
            return err(format!(
                "k = {k}: the occupied levels are not the lowest {nocc_total} (occupations {:?})",
                kscf.occupations[k]
            ));
        }
        co.push(c.slice(s![.., frozen_core..nocc_total]).to_owned());
        cv.push(c.slice(s![.., nocc_total..]).to_owned());
        eo.push(
            e[frozen_core..nocc_total]
                .iter()
                .map(|x| x + occ_shift)
                .collect(),
        );
        ev.push(e[nocc_total..].to_vec());
    }
    Ok(KRef {
        nk,
        nao,
        nocc,
        co,
        cv,
        eo,
        ev,
        madelung,
        occ_shift,
    })
}

fn same_kpts(a: &[[f64; 3]], b: &[[f64; 3]]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(x, y)| (0..3).all(|d| (x[d] - y[d]).abs() <= 1e-12 * (1.0 + y[d].abs())))
}

// ============================================================ MP2

/// Settings for [`kpoint_mp2`]. Deliberately no `Default`: the reference
/// exxdiv and the denominator convention must be stated.
#[derive(Debug, Clone, Copy)]
pub struct KMp2Config {
    /// Core orbitals excluded at every k (validated by `active_occ`).
    pub frozen_core: usize,
    /// The `exxdiv` the k-point RHF REFERENCE was converged with.
    pub reference_exxdiv: ExxDiv,
    /// Denominator convention.
    pub denominators: Mp2Denominators,
    /// Memory budget in bytes (`None` = ferric's unified budget).
    pub budget_bytes: Option<usize>,
    #[doc(hidden)]
    pub mutation: Option<KCorrMutation>,
}

impl KMp2Config {
    /// Madelung-shifted denominators, no frozen core, unified budget.
    pub fn shifted(reference_exxdiv: ExxDiv) -> Self {
        Self {
            frozen_core: 0,
            reference_exxdiv,
            denominators: Mp2Denominators::MadelungShifted,
            budget_bytes: None,
            mutation: None,
        }
    }
}

/// k-point MP2 result (every energy PER CELL).
#[derive(Debug, Clone)]
#[must_use]
pub struct KMp2Result {
    /// MP2 correlation energy per cell (`e_os + e_ss`).
    pub mp2_corr: f64,
    /// `(1/N_k³) Σ |V|² / D` (prototype `E_os`).
    pub e_os: f64,
    /// `(1/N_k³) Σ conj(V) (V − V_ibja) / D` (prototype `E_ss`).
    pub e_ss: f64,
    /// Direct (Coulomb-only) MP2 `(1/N_k³) Σ 2 |V|² / D`: the `O(Π²)` term
    /// of dRPA (a test anchor).
    pub e_direct: f64,
    /// `kscf.energy + mp2_corr` (the reference energy is its own; see
    /// [`crate::mp2::GammaMp2Result::total_energy`]).
    pub total_energy: f64,
    /// Mesh (supercell) Madelung constant.
    pub madelung: f64,
    /// Shift added to every active occupied ε at every k.
    pub occ_shift: f64,
    pub nocc_active: usize,
    /// Virtuals per k.
    pub nvir: Vec<usize>,
}

/// k-point closed-shell MP2 (module doc) on a converged k-point RHF `kscf`
/// over `mesh` (the mesh `kscf` and `ints` were built on).
pub fn kpoint_mp2(
    cell: &Cell,
    mesh: &KPointMesh,
    kscf: &KScfResult,
    ints: KCorrIntegrals<'_>,
    cfg: &KMp2Config,
) -> Result<KMp2Result, FerricError> {
    let r = prepare(
        "kpoint_mp2",
        cell,
        mesh,
        kscf,
        &ints,
        cfg.frozen_core,
        cfg.reference_exxdiv,
        cfg.denominators,
        cfg.mutation,
    )?;
    let (nk, nao, no) = (r.nk, r.nao, r.nocc);
    let nvir: Vec<usize> = r.ev.iter().map(|v| v.len()).collect();
    let nvmax = nvir.iter().copied().max().unwrap_or(0);
    let nov = no * nvmax;
    let rows = ints.max_rows() as u64;
    let nk64 = nk as u64;

    let mut ledger = Ledger::new(crate::budget::resolve(cfg.budget_bytes));
    ledger.reserve(
        &format!("k-point MP2 explicit Bvo(kb,kj), all N_k² pairs (N_k = {nk}, rows <= {rows}, nocc·nvir = {nov})"),
        bytes_of(nk64.saturating_mul(nk64).saturating_mul(rows), nov.saturating_mul(16)),
    )?;
    ledger.reserve(
        &format!(
            "k-point MP2 Bov(ki,ka), one ki row (N_k = {nk}, rows <= {rows}, nocc·nvir = {nov})"
        ),
        bytes_of(nk64.saturating_mul(rows), nov.saturating_mul(16)),
    )?;
    ledger.reserve(
        &format!(
            "k-point MP2 V[ki,kj,ka], one ki slice (N_k² = {}, nocc·nvir = {nov})",
            nk * nk
        ),
        bytes_of(
            nk64.saturating_mul(nk64).saturating_mul(nov as u64),
            nov.saturating_mul(16),
        ),
    )?;
    ledger.reserve(
        &format!("k-point MP2 block copy + half transform (rows <= {rows}, nao = {nao})"),
        bytes_of(
            rows.saturating_mul(nao as u64),
            (nao + nvmax).saturating_mul(16),
        ),
    )?;

    let conj_leg = cfg.mutation != Some(KCorrMutation::NoConj);
    let mut bvo: Vec<Array2<C64>> = Vec::with_capacity(nk * nk);
    for kb in 0..nk {
        for kj in 0..nk {
            let b = ints.block(kb, kj, nao)?;
            let mut t = half_transform(&b, nao, r.cv[kb].view(), r.co[kj].view())?;
            if conj_leg {
                t.mapv_inplace(|z| z.conj());
            }
            bvo.push(t);
        }
    }

    let (mut e_os, mut e_ss, mut e_dir) = (0.0_f64, 0.0_f64, 0.0_f64);
    for ki in 0..nk {
        let bov: Vec<Array2<C64>> = (0..nk)
            .map(|ka| {
                let b = ints.block(ki, ka, nao)?;
                half_transform(&b, nao, r.co[ki].view(), r.cv[ka].view())
            })
            .collect::<Result<_, _>>()?;
        // v[kj·N_k + ka] = (no·nv(ka)) × (nv(kb)·no), element (i·nv+a, b·no+j).
        let mut v: Vec<Array2<C64>> = Vec::with_capacity(nk * nk);
        for kj in 0..nk {
            for (ka, bov_a) in bov.iter().enumerate() {
                let kb = k_combo(mesh, ki, (1, kj), (-1, ka));
                let y = &bvo[kb * nk + kj];
                if bov_a.nrows() != y.nrows() {
                    return Err(FerricError::General(format!(
                        "kpoint_mp2: B({ki},{ka}) has {} rows but B({kb},{kj}) has {} — the two \
                         legs must share one q class's aux set",
                        bov_a.nrows(),
                        y.nrows()
                    )));
                }
                v.push(bov_a.t().dot(y));
            }
        }
        for kj in 0..nk {
            for ka in 0..nk {
                let kb_true = k_combo(mesh, ki, (1, kj), (-1, ka));
                let kb = if cfg.mutation == Some(KCorrMutation::KbIndex) {
                    k_combo(mesh, ki, (-1, kj), (1, ka))
                } else {
                    kb_true
                };
                let (nva, nvb) = (nvir[ka], nvir[kb]);
                if nvb != nvir[kb_true] {
                    return Err(FerricError::General(
                        "kpoint_mp2: KbIndex mutation needs a uniform nvir".into(),
                    ));
                }
                let m = &v[kj * nk + ka];
                let xm = &v[kj * nk + kb];
                let (eoi, eoj, eva, evb) = (&r.eo[ki], &r.eo[kj], &r.ev[ka], &r.ev[kb]);
                for i in 0..no {
                    for a in 0..nva {
                        for j in 0..no {
                            for b in 0..nvb {
                                let d = eoi[i] + eoj[j] - eva[a] - evb[b];
                                let vv = m[(i * nva + a, b * no + j)];
                                let xx = xm[(i * nvb + b, a * no + j)];
                                let v2 = vv.norm_sqr();
                                e_os += v2 / d;
                                e_ss += (vv.conj() * (vv - xx)).re / d;
                                e_dir += 2.0 * v2 / d;
                            }
                        }
                    }
                }
            }
        }
    }
    let nkf = nk as f64;
    let scale = if cfg.mutation == Some(KCorrMutation::NkNormalization) {
        1.0 / (nkf * nkf)
    } else {
        1.0 / (nkf * nkf * nkf)
    };
    let (e_os, e_ss, e_direct) = (e_os * scale, e_ss * scale, e_dir * scale);
    let mp2_corr = e_os + e_ss;
    if !(mp2_corr.is_finite() && e_direct.is_finite()) {
        return Err(FerricError::General(format!(
            "kpoint_mp2: non-finite correlation energy {mp2_corr} (zero denominator? occupied/virtual \
             degeneracy after a {:+} occupied shift)",
            r.occ_shift
        )));
    }
    Ok(KMp2Result {
        mp2_corr,
        e_os,
        e_ss,
        e_direct,
        total_energy: kscf.energy + mp2_corr,
        madelung: r.madelung,
        occ_shift: r.occ_shift,
        nocc_active: no,
        nvir,
    })
}

// ============================================================ dRPA

/// How [`kpoint_drpa`] evaluates each `E_q` (module doc).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KDrpaEnergy {
    /// Production: realified `Π(q)` through ferric-rpa's full-rank log-det
    /// quadrature (`quad_points` GL nodes, u0 = 0.5).
    Quadrature,
    /// TEST/ORACLE: analytic per-q frequency integral on the ov Gram.
    Plasmon,
    /// TEST/ORACLE: the `O(Π²)` term on the GL grid (= direct KMP2).
    SecondOrder,
}

/// Settings for [`kpoint_drpa`]. Deliberately no `Default`.
#[derive(Debug, Clone, Copy)]
pub struct KDrpaConfig {
    /// Core orbitals excluded at every k.
    pub frozen_core: usize,
    /// The `exxdiv` the k-point RHF REFERENCE was converged with.
    pub reference_exxdiv: ExxDiv,
    /// Denominator convention (shared with MP2).
    pub denominators: Mp2Denominators,
    /// Frequency points (validated as the Gamma dRPA's; unused by
    /// `Plasmon` but still validated).
    pub quad_points: usize,
    /// Energy construction.
    pub energy: KDrpaEnergy,
    /// Memory budget in bytes (`None` = ferric's unified budget).
    pub budget_bytes: Option<usize>,
    #[doc(hidden)]
    pub mutation: Option<KCorrMutation>,
}

impl KDrpaConfig {
    /// Madelung-shifted, no frozen core, 40-point quadrature, unified budget.
    pub fn shifted(reference_exxdiv: ExxDiv) -> Self {
        Self {
            frozen_core: 0,
            reference_exxdiv,
            denominators: Mp2Denominators::MadelungShifted,
            quad_points: DEFAULT_KDRPA_QUAD_POINTS,
            energy: KDrpaEnergy::Quadrature,
            budget_bytes: None,
            mutation: None,
        }
    }
}

/// k-point dRPA result (per cell).
#[derive(Debug, Clone)]
#[must_use]
pub struct KDrpaResult {
    /// dRPA correlation energy per cell, `(1/N_k) Σ_q E_q`.
    pub drpa_corr: f64,
    /// `E_q` per class ([`KPointMesh::q_class`] order), before the `1/N_k`.
    pub per_q: Vec<f64>,
    /// `kscf.energy + drpa_corr`.
    pub total_energy: f64,
    pub madelung: f64,
    pub occ_shift: f64,
    pub nocc_active: usize,
    pub nvir: Vec<usize>,
    /// Rows of `Bq` per class.
    pub naux: Vec<usize>,
    /// Energy construction used.
    pub energy: KDrpaEnergy,
}

/// `E_q` from ferric-rpa's real pipeline on the realified `Bq` (module doc).
fn drpa_q_realified(
    bq: &Array2<C64>,
    e: &[f64],
    quad_points: usize,
    budget: usize,
) -> Result<f64, FerricError> {
    let (nr, nc) = bq.dim();
    // Columns go in ASCENDING excitation energy. The energy is invariant
    // under a column permutation (B F Bᵀ with F diagonal), but ferric-rpa's
    // scale-factor guard reads an out-of-order ε_vir as a rotated
    // (non-canonical) basis and debug-asserts; these ε_vir are per-column
    // excitation energies of a canonical k basis, so sorting is exact.
    let mut order: Vec<usize> = (0..nc).collect();
    order.sort_by(|&a, &b| e[a].total_cmp(&e[b]));
    let mut br = Array2::<f64>::zeros((2 * nr, 2 * nc));
    let mut ed = vec![0.0; 2 * nc];
    for (j, &c) in order.iter().enumerate() {
        // Each energy appears twice (Re and Im halves); interleave them so
        // the doubled list stays ascending too.
        ed[2 * j] = e[c];
        ed[2 * j + 1] = e[c];
        for p in 0..nr {
            let z = bq[(p, c)];
            br[(p, 2 * j)] = z.re;
            br[(p, 2 * j + 1)] = -z.im;
            br[(nr + p, 2 * j)] = z.im;
            br[(nr + p, 2 * j + 1)] = z.re;
        }
    }
    let res = drpa_from_b_ov(br, None, &[0.0], &ed, quad_points, Some(budget))?;
    if !res.eigensolver_converged {
        return Err(FerricError::General(
            "kpoint_drpa: static dielectric eigensolve reported non-convergence".into(),
        ));
    }
    Ok(0.5 * res.e_rpa)
}

/// `K_q = Bqᵀ conj(Bq)` (the ov Gram over (k', i, a)).
fn ov_gram(bq: &Array2<C64>) -> Array2<C64> {
    bq.t().dot(&bq.mapv(|z| z.conj()))
}

/// TEST/ORACLE: `E_q = ½ [Σ √eig(D² + 4 D^{1/2} K D^{1/2}) − tr D − 2 tr K]`.
fn drpa_q_plasmon(bq: &Array2<C64>, e: &[f64]) -> Result<f64, FerricError> {
    let k = ov_gram(bq);
    let n = e.len();
    let sd: Vec<f64> = e.iter().map(|x| x.sqrt()).collect();
    let mut m = Array2::<C64>::zeros((n, n));
    for p in 0..n {
        for q in 0..n {
            m[(p, q)] = k[(p, q)] * (4.0 * sd[p] * sd[q]);
        }
        m[(p, p)] += C64::new(e[p] * e[p], 0.0);
    }
    let (lam, _) = eigh_herm(&hermitize(&m))
        .map_err(|err| FerricError::Lapack(format!("kpoint_drpa plasmon eigh: {err}")))?;
    let lmin = lam.iter().copied().fold(f64::INFINITY, f64::min);
    if !(lmin > 0.0) {
        return Err(FerricError::General(format!(
            "kpoint_drpa: dRPA instability (smallest Ω² = {lmin:e})"
        )));
    }
    let sum_omega: f64 = lam.iter().map(|x| x.sqrt()).sum();
    let tr_d: f64 = e.iter().sum();
    let tr_k: f64 = (0..n).map(|p| k[(p, p)].re).sum();
    Ok(0.5 * (sum_omega - tr_d - 2.0 * tr_k))
}

/// TEST/ORACLE: `−(1/2π) ∫ dω tr Π(q)² / 2` on the GL grid.
fn drpa_q_second_order(bq: &Array2<C64>, e: &[f64], quad_points: usize) -> f64 {
    let (freqs, weights) =
        ferric_rpa::quadrature::gauss_legendre_nodes(quad_points, GAMMA_DRPA_QUAD_U0);
    let (nr, nc) = bq.dim();
    let mut total = 0.0_f64;
    for (&w, &wk) in freqs.iter().zip(&weights) {
        let mut bs = bq.to_owned();
        for (c, &ec) in e.iter().enumerate() {
            let f = (4.0 * ec / (w * w + ec * ec)).sqrt();
            bs.column_mut(c).mapv_inplace(|z| z * f);
        }
        // tr Π² = ‖Π‖_F² = ‖the smaller Gram‖_F² (same nonzero spectrum).
        let g = if nr <= nc {
            bs.dot(&bs.t().mapv(|z| z.conj()))
        } else {
            bs.t().dot(&bs.mapv(|z| z.conj()))
        };
        let tr_pi2: f64 = g.iter().map(|z| z.norm_sqr()).sum();
        total += wk * (-0.5 * tr_pi2);
    }
    total / (2.0 * PI)
}

/// k-point closed-shell dRPA (module doc) on a converged k-point RHF
/// `kscf` over `mesh`.
pub fn kpoint_drpa(
    cell: &Cell,
    mesh: &KPointMesh,
    kscf: &KScfResult,
    ints: KCorrIntegrals<'_>,
    cfg: &KDrpaConfig,
) -> Result<KDrpaResult, FerricError> {
    validate_quad_points(cfg.quad_points)?;
    let r = prepare(
        "kpoint_drpa",
        cell,
        mesh,
        kscf,
        &ints,
        cfg.frozen_core,
        cfg.reference_exxdiv,
        cfg.denominators,
        cfg.mutation,
    )?;
    let (nk, nao, no) = (r.nk, r.nao, r.nocc);
    let nvir: Vec<usize> = r.ev.iter().map(|v| v.len()).collect();
    let ncols: usize = nvir.iter().map(|v| no * v).sum();
    let rows = ints.max_rows();
    let budget = crate::budget::resolve(cfg.budget_bytes);
    let mut ledger = Ledger::new(budget);
    ledger.reserve(
        &format!("k-point dRPA Bq of one class (rows <= {rows}, N_k·nocc·nvir = {ncols})"),
        bytes_of(rows as u64, ncols.saturating_mul(16)),
    )?;
    match cfg.energy {
        KDrpaEnergy::Quadrature => ledger.reserve(
            &format!("k-point dRPA realified Bq (2·{rows} × 2·{ncols}, real)"),
            bytes_of((2 * rows) as u64, (2 * ncols).saturating_mul(8)),
        )?,
        KDrpaEnergy::Plasmon => ledger.reserve(
            &format!("k-point dRPA plasmon Gram + matrix + eigh copies ({ncols}²)"),
            bytes_of(ncols as u64, ncols.saturating_mul(64)),
        )?,
        KDrpaEnergy::SecondOrder => ledger.reserve(
            &format!("k-point dRPA scaled Bq + Gram (rows <= {rows}, {ncols})"),
            bytes_of(rows as u64, ncols.saturating_add(rows).saturating_mul(16)),
        )?,
    }
    ledger.reserve(
        &format!("k-point dRPA block copy + half transform (rows <= {rows}, nao = {nao})"),
        bytes_of((rows * nao) as u64, (2 * nao).saturating_mul(16)),
    )?;

    let inv_sqrt_nk = if cfg.mutation == Some(KCorrMutation::NkNormalization) {
        1.0
    } else {
        1.0 / (nk as f64).sqrt()
    };
    let mut per_q = Vec::with_capacity(nk);
    let mut naux = Vec::with_capacity(nk);
    for iq in 0..nk {
        let (_, mq) = mesh.q_class(iq);
        let mut parts: Vec<Array2<C64>> = Vec::with_capacity(nk);
        let mut e: Vec<f64> = Vec::with_capacity(ncols);
        for j in 0..nk {
            let k = mesh.k_minus_q(j, mq);
            let k_occ = if cfg.mutation == Some(KCorrMutation::RpaOccupiedKPlusQ) {
                let x = mesh.numerators(j);
                mesh.index_of_nums([x[0] + 2 * mq[0], x[1] + 2 * mq[1], x[2] + 2 * mq[2]])
                    .expect("k' + q is on the mesh")
            } else {
                k
            };
            let b = ints.block(k, j, nao)?;
            parts.push(half_transform(&b, nao, r.co[k].view(), r.cv[j].view())?);
            for &ei in &r.eo[k_occ] {
                for &ea in &r.ev[j] {
                    let d = ea - ei;
                    if !(d.is_finite() && d > 0.0) {
                        return Err(FerricError::General(format!(
                            "kpoint_drpa: non-positive or non-finite e_ia = {d} (ε_i({k_occ}) {ei}, \
                             ε_a({j}) {ea}); the dRPA frequency integral is undefined"
                        )));
                    }
                    e.push(d);
                }
            }
        }
        let nr = parts[0].nrows();
        if parts.iter().any(|p| p.nrows() != nr) {
            return Err(FerricError::General(format!(
                "kpoint_drpa: the pairs of q class {iq} do not share one aux set (row counts {:?})",
                parts.iter().map(|p| p.nrows()).collect::<Vec<_>>()
            )));
        }
        let mut bq = Array2::<C64>::zeros((nr, ncols));
        let mut c0 = 0;
        for p in &parts {
            let w = p.ncols();
            for row in 0..nr {
                for c in 0..w {
                    bq[(row, c0 + c)] = p[(row, c)] * inv_sqrt_nk;
                }
            }
            c0 += w;
        }
        drop(parts);
        let eq = match cfg.energy {
            KDrpaEnergy::Quadrature => drpa_q_realified(&bq, &e, cfg.quad_points, budget)?,
            KDrpaEnergy::Plasmon => drpa_q_plasmon(&bq, &e)?,
            KDrpaEnergy::SecondOrder => drpa_q_second_order(&bq, &e, cfg.quad_points),
        };
        per_q.push(eq);
        naux.push(nr);
    }
    let drpa_corr = per_q.iter().sum::<f64>() / nk as f64;
    if !drpa_corr.is_finite() {
        return Err(FerricError::General(format!(
            "kpoint_drpa: non-finite correlation energy {drpa_corr}"
        )));
    }
    Ok(KDrpaResult {
        drpa_corr,
        per_q,
        total_energy: kscf.energy + drpa_corr,
        madelung: r.madelung,
        occ_shift: r.occ_shift,
        nocc_active: no,
        nvir,
        naux,
        energy: cfg.energy,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rnd_c(seed: &mut u64) -> C64 {
        let mut next = || {
            *seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((*seed >> 11) as f64 / (1u64 << 53) as f64) - 0.5
        };
        C64::new(next(), next())
    }

    /// Realified quadrature ≡ complex plasmon ≡ (at second order) the
    /// direct sum, on a random complex Bq: the realification doubles every
    /// eigenvalue exactly, so the halved real energy is the complex one.
    #[test]
    fn realified_quadrature_matches_the_complex_plasmon() {
        let mut seed = 99u64;
        let (nr, nc) = (5usize, 7usize);
        let bq = Array2::from_shape_fn((nr, nc), |_| rnd_c(&mut seed) * 0.3);
        let e: Vec<f64> = (0..nc).map(|c| 0.6 + 0.37 * c as f64).collect();
        let pl = drpa_q_plasmon(&bq, &e).unwrap();
        let qd = drpa_q_realified(&bq, &e, 200, 1 << 30).unwrap();
        assert!((pl - qd).abs() < 1e-10, "plasmon {pl} quad {qd}");
        // Second order vs the explicit 2 Σ |K_cd|² / (−e_c − e_d) sum.
        let k = ov_gram(&bq);
        let mut direct = 0.0;
        for c in 0..nc {
            for d in 0..nc {
                direct += 2.0 * k[(c, d)].norm_sqr() / (-e[c] - e[d]);
            }
        }
        // ∫_0^∞ g_c g_d dω = 8π/(e_c + e_d) with g = 4e/(ω² + e²), so
        // −(1/2π)·½·Σ |K_cd|² g_c g_d integrates to −2 Σ |K_cd|²/(e_c + e_d).
        let so = drpa_q_second_order(&bq, &e, 512);
        assert!(
            (so - direct).abs() < 1e-10,
            "second order {so} direct {direct}"
        );
    }

    #[test]
    fn half_transform_matches_the_explicit_sum() {
        let mut seed = 7u64;
        let (rows, nao, n1, n2) = (3usize, 3usize, 2usize, 1usize);
        let b = Array2::from_shape_fn((rows, nao * nao), |_| rnd_c(&mut seed));
        let l = Array2::from_shape_fn((nao, n1), |_| rnd_c(&mut seed));
        let r = Array2::from_shape_fn((nao, n2), |_| rnd_c(&mut seed));
        let out = half_transform(&b, nao, l.view(), r.view()).unwrap();
        let mut d = 0.0_f64;
        for p in 0..rows {
            for x in 0..n1 {
                for y in 0..n2 {
                    let mut acc = C64::new(0.0, 0.0);
                    for m in 0..nao {
                        for ll in 0..nao {
                            acc += l[(m, x)].conj() * b[(p, m * nao + ll)] * r[(ll, y)];
                        }
                    }
                    d = d.max((acc - out[(p, x * n2 + y)]).norm());
                }
            }
        }
        assert!(d < 1e-14, "{d:e}");
    }
}
