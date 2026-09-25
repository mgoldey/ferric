//! TEST/ORACLE ONLY: dense pure-AFT Gamma-point ERI tensor and the J/K
//! builders on it (Stage 1 step 5, `reference/pbc/stage1-design.md` §5-6).
//!
//! ```text
//! I[μν, λσ] = (1/Ω) Σ_{G≠0} (4π/G²) Re[ P*_μν(G) P_λσ(G) ]
//!           = (2/Ω) Σ_{G∈half} (4π/G²) (P^re_μν P^re_λσ + P^im_μν P^im_λσ)
//! J[μ,ν] = Σ_λσ I[μν, λσ] D[λ,σ]
//! K[μ,ν] = Σ_λσ I[μλ, νσ] D[λ,σ]  +  v_M (S D S)[μ,ν]     (exxdiv = ewald)
//! ```
//!
//! This is `pbc_gamma.py`'s `w=None` path: all Coulomb in reciprocal space,
//! G = 0 dropped (the PySCF `exxdiv=None` convention for K; cancels for J in a
//! neutral cell), optional Madelung probe-charge correction `K += v_M S D S`
//! (`exxdiv='ewald'`, PySCF `df_jk._ewald_exxdiv_for_G0`). The K definition is
//! ferric's (`KBuilder` doc: `K_μν = Σ (μλ|νσ) D_λσ` with the total density),
//! so the SCF assembles `F = h + J − ½K` unchanged.
//!
//! Memory is `8 nao⁴` bytes and the G count grows as `Ω p_max^{3/2}`: this is
//! a correctness oracle for toy cells, never a production builder. The tensor
//! size is HARD-capped (error, not a warning) by the caller's `max_bytes`,
//! and, like every ferric-pbc buffer, everything the build holds (tensor +
//! GEMM temporary, G list, `pair_ft` chunks) is reserved against ferric's
//! memory budget before allocation ([`DenseAftEri::build_budgeted`]).
//! G is processed in `pair_ft` chunks; the chunk size changes the GEMM
//! summation order, so different budgets agree to roundoff, not bitwise
//! (`tests/pbc_memory_gates.rs` measures it).

use crate::budget::{bytes_of, Ledger};
use crate::ewald::madelung_constant;
use crate::hcore::{gvector_list_bytes, half_gvectors, G_CHUNK_BYTES};
use crate::lattice::Cell;
use crate::pair_ft::{pair_ft_bytes_per_g, pair_ft_chunked};
use crate::timing::{CallClock, PbcTimings, StageClock};
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_scf::fock::{JBuilder, KBuilder};
use ndarray::{Array2, Array3};
use num_complex::Complex64;
use std::f64::consts::PI;

/// Default hard cap on the dense tensor (`8 nao⁴` bytes): 512 MiB (nao 90).
pub const DEFAULT_DENSE_AFT_MAX_BYTES: usize = 512 << 20;

/// Default truncation target for the pure-AFT G sphere.
pub const DEFAULT_DENSE_AFT_PRECISION: f64 = 1e-14;

/// Treatment of the exchange G = 0 divergence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExxDiv {
    /// Drop the G = 0 term (PySCF `exxdiv=None`). Converges only as 1/L.
    None,
    /// Madelung probe-charge correction `K += v_M S D S` (PySCF
    /// `exxdiv='ewald'`, the Stage-1 default). Converges as L⁻³.
    Ewald,
}

impl ExxDiv {
    /// Strict parse: `"none"` or `"ewald"` (case-insensitive); anything else
    /// is an error (config honesty — no silent default).
    pub fn parse_config_str(s: &str) -> Result<Self, FerricError> {
        match s.to_ascii_lowercase().as_str() {
            "none" => Ok(ExxDiv::None),
            "ewald" => Ok(ExxDiv::Ewald),
            other => Err(FerricError::General(format!(
                "exxdiv must be \"none\" or \"ewald\", got {other:?}"
            ))),
        }
    }
}

/// The dense pure-AFT tensor plus what the K builder needs.
#[derive(Debug, Clone)]
pub struct DenseAftEri {
    nao: usize,
    /// `(nao², nao²)`, row `μ·nao+ν`, column `λ·nao+σ`.
    eri: Array2<f64>,
    s: Array2<f64>,
    madelung: f64,
    n_g_half: usize,
    n_g_chunks: usize,
    resident_bytes: usize,
    bytes_per_g: usize,
    /// G-sphere radius of the tensor (Bohr⁻¹).
    gcut: f64,
    /// `pair_ft` primitive screen used for the tensor.
    pair_thresh: f64,
    /// Build stage timings and counters ([`crate::timing`]).
    timings: PbcTimings,
    /// Accumulated [`DenseAftJ`] / [`DenseAftK`] build calls.
    j_clock: CallClock,
    k_clock: CallClock,
}

impl DenseAftEri {
    /// Build `I` for `cell` in the AO basis `prep` (built from `cell.mol()`).
    ///
    /// * `s` — the lattice overlap (e.g. `PeriodicHcore::s`), used only by
    ///   the Madelung term.
    /// * `precision` — G sphere `|G| <= 2 √(p_max ln(1/precision))`,
    ///   `p_max` = twice the largest exponent (the prototype's rule).
    /// * `max_bytes` — hard cap on `8 nao⁴`; exceeded ⇒ `Err` before any work.
    ///
    /// Memory is gated against ferric's unified budget
    /// ([`crate::budget::resolve`]`(None)`); see [`DenseAftEri::build_budgeted`].
    pub fn build(
        cell: &Cell,
        prep: &PreparedBasis,
        s: &Array2<f64>,
        exxdiv: ExxDiv,
        precision: f64,
        max_bytes: usize,
    ) -> Result<Self, FerricError> {
        Self::build_budgeted(cell, prep, s, exxdiv, precision, max_bytes, None)
    }

    /// [`DenseAftEri::build`] with an explicit memory budget (`None` =
    /// ferric's unified budget). Reserved before allocation, in order: the
    /// tensor plus its same-size GEMM/symmetrisation temporary
    /// (`2 · 8 nao⁴`), the overlap copy, the G list; then each `pair_ft`
    /// chunk (plus the four `nao² × n_G` real staging arrays) must fit in
    /// what is left (capped at 64 MiB).
    pub fn build_budgeted(
        cell: &Cell,
        prep: &PreparedBasis,
        s: &Array2<f64>,
        exxdiv: ExxDiv,
        precision: f64,
        max_bytes: usize,
        budget_bytes: Option<usize>,
    ) -> Result<Self, FerricError> {
        let nao = prep.nbasis();
        let bytes = 8u128 * (nao as u128).pow(4);
        if bytes > max_bytes as u128 {
            return Err(FerricError::General(format!(
                "DenseAftEri: dense nao^4 tensor needs {bytes} bytes (nao = {nao}) > cap \
                 {max_bytes} bytes; this is a toy-cell oracle, use a periodic density-fitted \
                 builder for real systems"
            )));
        }
        if s.dim() != (nao, nao) {
            return Err(FerricError::General(format!(
                "DenseAftEri: overlap has shape {:?}, expected ({nao}, {nao})",
                s.dim()
            )));
        }
        if !(f64::MIN_POSITIVE..1.0).contains(&precision) {
            return Err(FerricError::General(format!(
                "DenseAftEri: precision must lie in (0, 1), got {precision}"
            )));
        }
        let pmax = 2.0
            * prep
                .located_shells()
                .iter()
                .flat_map(|sh| sh.exponents.iter().copied())
                .fold(0.0_f64, f64::max);
        let gcut = 2.0 * (pmax * (1.0 / precision).ln()).sqrt();
        let n2 = nao * nao;
        let mut ledger = Ledger::new(crate::budget::resolve(budget_bytes));
        ledger.reserve(
            &format!("DenseAftEri nao^4 tensor + same-size GEMM temporary (nao = {nao})"),
            bytes_of(n2 as u64, n2).saturating_mul(16),
        )?;
        ledger.reserve(
            &format!("DenseAftEri overlap copy (nao = {nao})"),
            bytes_of(n2 as u64, 8),
        )?;
        ledger.reserve(
            &format!("DenseAftEri G list (|G| <= {gcut:.3})"),
            gvector_list_bytes(cell, gcut)?,
        )?;
        let total = StageClock::start();
        let mut timings = PbcTimings::default();
        let gv = half_gvectors(cell, gcut)?;
        let vol = cell.volume();
        let mut eri = Array2::<f64>::zeros((n2, n2));
        let chunk_budget = ledger.remaining().min(G_CHUNK_BYTES);
        let resident_bytes = ledger.resident();
        // a_re, a_im, b_re, b_im: 4 real nao² columns per G.
        let staging_per_g = n2.saturating_mul(32);
        let lmax = prep
            .located_shells()
            .iter()
            .map(|sh| sh.l.max(0) as usize)
            .max()
            .unwrap_or(0);
        let bytes_per_g = pair_ft_bytes_per_g(nao, lmax).saturating_add(staging_per_g);
        let (mut sink_wall, mut sink_cpu) = (0.0_f64, None::<f64>);
        let accumulate =
            |_g0: usize, gs: &[[f64; 3]], p: &Array3<Complex64>| -> Result<(), FerricError> {
                let clock = StageClock::start();
                let ng = gs.len();
                let mut a_re = Array2::<f64>::zeros((n2, ng));
                let mut a_im = Array2::<f64>::zeros((n2, ng));
                let mut b_re = Array2::<f64>::zeros((n2, ng));
                let mut b_im = Array2::<f64>::zeros((n2, ng));
                for (g, gvec) in gs.iter().enumerate() {
                    let g2 = gvec[0] * gvec[0] + gvec[1] * gvec[1] + gvec[2] * gvec[2];
                    let w = 2.0 / vol * 4.0 * PI / g2;
                    for m in 0..nao {
                        for k in 0..nao {
                            let z = p[[m, k, g]];
                            let r = m * nao + k;
                            a_re[(r, g)] = z.re;
                            a_im[(r, g)] = z.im;
                            b_re[(r, g)] = w * z.re;
                            b_im[(r, g)] = w * z.im;
                        }
                    }
                }
                eri += &b_re.dot(&a_re.t());
                eri += &b_im.dot(&a_im.t());
                let (w, c) = clock.elapsed();
                sink_wall += w;
                if let Some(c) = c {
                    sink_cpu = Some(sink_cpu.unwrap_or(0.0) + c);
                }
                Ok(())
            };
        let clock = StageClock::start();
        let n_g_chunks = pair_ft_chunked(
            cell,
            prep,
            &gv,
            0.1 * precision,
            chunk_budget,
            staging_per_g,
            accumulate,
        )?;
        let (lr_wall, lr_cpu) = clock.elapsed();
        timings.add(
            "dense AFT pair FT",
            (lr_wall - sink_wall).max(0.0),
            match (lr_cpu, sink_cpu) {
                (Some(a), Some(b)) => Some((a - b).max(0.0)),
                (a, None) => a,
                (None, Some(_)) => None,
            },
            1,
        );
        timings.add(
            "dense AFT GEMM",
            sink_wall,
            sink_cpu.or(lr_cpu.map(|_| 0.0)),
            n_g_chunks as u64,
        );
        // Exactly symmetric in exact arithmetic; remove GEMM rounding.
        let eri = 0.5 * (&eri + &eri.t());
        let madelung = match exxdiv {
            ExxDiv::None => 0.0,
            ExxDiv::Ewald => madelung_constant(cell)?,
        };
        ferric_core::memory::warn_if_rss_over("ferric-pbc DenseAftEri", ledger.budget(), 1.1);
        timings.set_counter("dense AFT half-G", gv.len() as u64);
        timings.set_counter("dense AFT chunks", n_g_chunks as u64);
        timings.finish(&total);
        Ok(Self {
            nao,
            eri,
            s: s.clone(),
            madelung,
            n_g_half: gv.len(),
            n_g_chunks,
            resident_bytes,
            bytes_per_g,
            gcut,
            pair_thresh: 0.1 * precision,
            timings,
            j_clock: CallClock::default(),
            k_clock: CallClock::default(),
        })
    }

    /// The same tensor with another exchange-divergence treatment (the
    /// tensor does not depend on it; only the K builder's `v_M` changes).
    /// `cell` must be the cell the tensor was built for.
    pub fn with_exxdiv(mut self, cell: &Cell, exxdiv: ExxDiv) -> Result<Self, FerricError> {
        self.madelung = match exxdiv {
            ExxDiv::None => 0.0,
            ExxDiv::Ewald => madelung_constant(cell)?,
        };
        Ok(self)
    }

    /// `(nao², nao²)` tensor, row `μ·nao+ν`, column `λ·nao+σ`.
    pub fn eri(&self) -> &Array2<f64> {
        &self.eri
    }

    /// `v_M` applied in K (0 for [`ExxDiv::None`]).
    pub fn madelung(&self) -> f64 {
        self.madelung
    }

    /// Number of half-sphere G vectors summed.
    pub fn n_g_half(&self) -> usize {
        self.n_g_half
    }

    /// G-sphere radius `|G| <= gcut` (Bohr⁻¹) the tensor summed (half
    /// sphere, `G ≠ 0`). The Gamma gradient (`crate::grad`) differentiates
    /// over exactly this set.
    pub fn gcut(&self) -> f64 {
        self.gcut
    }

    /// Primitive-pair screening threshold handed to `pair_ft` for the tensor.
    pub fn pair_thresh(&self) -> f64 {
        self.pair_thresh
    }

    /// Number of `pair_ft` G chunks the build used.
    pub fn n_g_chunks(&self) -> usize {
        self.n_g_chunks
    }

    /// Bytes reserved on the budget ledger before the G chunks started
    /// (tensor + temporary, overlap copy, G list).
    pub fn resident_bytes(&self) -> usize {
        self.resident_bytes
    }

    /// Bytes one G vector costs in a chunk (`pair_ft` output + scratch +
    /// the four real staging columns).
    pub fn bytes_per_g(&self) -> usize {
        self.bytes_per_g
    }

    /// Build stage timings and counters (G list through symmetrisation),
    /// plus the accumulated J and K build calls of every builder borrowed
    /// from this tensor (`"scf J (dense AFT)"`, `"scf K (dense AFT)"`).
    /// `wall_s` is the BUILD total.
    pub fn timings(&self) -> PbcTimings {
        let mut t = self.timings.clone();
        t.add_stage(&self.j_clock.timing("scf J (dense AFT)"));
        t.add_stage(&self.k_clock.timing("scf K (dense AFT)"));
        t
    }

    /// J builder borrowing this tensor.
    pub fn j_builder(&self) -> DenseAftJ<'_> {
        DenseAftJ { eri: self }
    }

    /// K builder (with the Madelung term) borrowing this tensor.
    pub fn k_builder(&self) -> DenseAftK<'_> {
        DenseAftK {
            eri: self,
            madelung: self.madelung,
        }
    }

    /// K builder with an explicit Madelung shift `v_M` (0 = `exxdiv=None`),
    /// ignoring the tensor's own [`DenseAftEri::madelung`]. Lets one tensor
    /// serve both stages of the staged Gamma UHF (`crate::uhf::gamma_uhf`)
    /// without cloning `nao⁴` floats.
    pub fn k_builder_with_madelung(&self, madelung: f64) -> DenseAftK<'_> {
        DenseAftK {
            eri: self,
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

/// [`JBuilder`] over a [`DenseAftEri`]. Overwrites `j`.
pub struct DenseAftJ<'a> {
    eri: &'a DenseAftEri,
}

impl JBuilder for DenseAftJ<'_> {
    fn build(&mut self, d: &Array2<f64>, j: &mut Array2<f64>) -> Result<usize, FerricError> {
        let eri = self.eri;
        eri.j_clock.time(|| self.build_untimed(d, j))
    }

    fn reset(&mut self) {}
}

impl DenseAftJ<'_> {
    fn build_untimed(
        &mut self,
        d: &Array2<f64>,
        j: &mut Array2<f64>,
    ) -> Result<usize, FerricError> {
        self.eri.check(d, j, "DenseAftJ")?;
        let n = self.eri.nao;
        let i = &self.eri.eri;
        for m in 0..n {
            for k in 0..n {
                let row = m * n + k;
                let mut acc = 0.0;
                for l in 0..n {
                    for s in 0..n {
                        acc += i[(row, l * n + s)] * d[(l, s)];
                    }
                }
                j[(m, k)] = acc;
            }
        }
        Ok(n.pow(4))
    }
}

/// [`KBuilder`] over a [`DenseAftEri`], including the Madelung shift.
/// Overwrites `k`.
///
/// Does NOT override [`KBuilder::build_from_occ`]: the trait default
/// reconstructs `D = C Cᵀ` and calls [`KBuilder::build`], so the Madelung
/// term is kept on the occupied-MO path too. An override must add
/// `v_M S C Cᵀ S` itself (FINDINGS "Iteration 6"; guarded by
/// `tests/pbc_uhf.rs::k_builders_keep_madelung_on_the_occupied_path`).
pub struct DenseAftK<'a> {
    eri: &'a DenseAftEri,
    madelung: f64,
}

impl KBuilder for DenseAftK<'_> {
    fn build(&mut self, d: &Array2<f64>, k: &mut Array2<f64>) -> Result<usize, FerricError> {
        let eri = self.eri;
        eri.k_clock.time(|| self.build_untimed(d, k))
    }

    fn update_density(&mut self, _d: &Array2<f64>) {}

    fn reset(&mut self) {}
}

impl DenseAftK<'_> {
    fn build_untimed(
        &mut self,
        d: &Array2<f64>,
        k: &mut Array2<f64>,
    ) -> Result<usize, FerricError> {
        self.eri.check(d, k, "DenseAftK")?;
        let n = self.eri.nao;
        let i = &self.eri.eri;
        for m in 0..n {
            for nu in 0..n {
                let mut acc = 0.0;
                for l in 0..n {
                    let row = m * n + l;
                    for s in 0..n {
                        acc += i[(row, nu * n + s)] * d[(l, s)];
                    }
                }
                k[(m, nu)] = acc;
            }
        }
        if self.madelung != 0.0 {
            let sds = self.eri.s.dot(d).dot(&self.eri.s);
            k.scaled_add(self.madelung, &sds);
        }
        Ok(n.pow(4))
    }
}
