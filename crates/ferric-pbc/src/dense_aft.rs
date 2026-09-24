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
//! size is HARD-capped (error, not a warning) by the caller's `max_bytes`.

use crate::ewald::madelung_constant;
use crate::hcore::{half_gvectors, G_CHUNK_BYTES};
use crate::lattice::Cell;
use crate::pair_ft::pair_ft_with_thresh;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_scf::fock::{JBuilder, KBuilder};
use ndarray::Array2;
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
}

impl DenseAftEri {
    /// Build `I` for `cell` in the AO basis `prep` (built from `cell.mol()`).
    ///
    /// * `s` — the lattice overlap (e.g. `PeriodicHcore::s`), used only by
    ///   the Madelung term.
    /// * `precision` — G sphere `|G| <= 2 √(p_max ln(1/precision))`,
    ///   `p_max` = twice the largest exponent (the prototype's rule).
    /// * `max_bytes` — hard cap on `8 nao⁴`; exceeded ⇒ `Err` before any work.
    pub fn build(
        cell: &Cell,
        prep: &PreparedBasis,
        s: &Array2<f64>,
        exxdiv: ExxDiv,
        precision: f64,
        max_bytes: usize,
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
        let gv = half_gvectors(cell, gcut)?;
        let vol = cell.volume();
        let n2 = nao * nao;
        let mut eri = Array2::<f64>::zeros((n2, n2));
        let chunk = (G_CHUNK_BYTES / (16 * n2).max(1)).max(1);
        for gs in gv.chunks(chunk) {
            let p = pair_ft_with_thresh(cell, prep, gs, 0.1 * precision)?;
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
        }
        // Exactly symmetric in exact arithmetic; remove GEMM rounding.
        let eri = 0.5 * (&eri + &eri.t());
        let madelung = match exxdiv {
            ExxDiv::None => 0.0,
            ExxDiv::Ewald => madelung_constant(cell)?,
        };
        Ok(Self {
            nao,
            eri,
            s: s.clone(),
            madelung,
            n_g_half: gv.len(),
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

    /// J builder borrowing this tensor.
    pub fn j_builder(&self) -> DenseAftJ<'_> {
        DenseAftJ { eri: self }
    }

    /// K builder (with the Madelung term) borrowing this tensor.
    pub fn k_builder(&self) -> DenseAftK<'_> {
        DenseAftK { eri: self }
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

    fn reset(&mut self) {}
}

/// [`KBuilder`] over a [`DenseAftEri`], including the Madelung shift.
/// Overwrites `k`.
pub struct DenseAftK<'a> {
    eri: &'a DenseAftEri,
}

impl KBuilder for DenseAftK<'_> {
    fn build(&mut self, d: &Array2<f64>, k: &mut Array2<f64>) -> Result<usize, FerricError> {
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
        if self.eri.madelung != 0.0 {
            let sds = self.eri.s.dot(d).dot(&self.eri.s);
            k.scaled_add(self.eri.madelung, &sds);
        }
        Ok(n.pow(4))
    }

    fn update_density(&mut self, _d: &Array2<f64>) {}

    fn reset(&mut self) {}
}
