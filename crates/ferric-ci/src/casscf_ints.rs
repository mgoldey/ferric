//! The two-electron integrals of a two-step CASSCF macro iteration.
//!
//! Each [`crate::casscf::CasScfProblem::evaluate`] needs three things from
//! the AO ERIs, at the current orbitals:
//!
//! * `J[D]` and `K[D]` for the core density `D^c` (inactive Fock `F^I`);
//! * the same for the active density `D^a` (active Fock `F^A`);
//! * `(pu|vw)`, `p` over all MOs, `u, v, w` active.
//!
//! [`CasScfIntegrals::Direct`] builds the first two with
//! ferric-scf's direct J/K builder ([`ferric_scf::rhf::build_jk_with_pool`],
//! Schwarz and density screened, deterministic reduction) and the third with
//! a direct half transform (`direct_paaa`); no `nbas^4` array exists. The
//! peak is the half-transformed block `(mn|vw)`, `nbas^2 n_act^2` doubles.
//!
//! [`CasScfIntegrals::Dense`] holds the full `nbas^4` AO block and contracts
//! it by parallel GEMVs (`dense_jk`) and quarter-transform GEMMs. It is the test oracle for the direct route. The default,
//! [`CasScfIntegrals::Auto`], takes the dense route when its block fits in
//! half of the memory budget and the direct route otherwise.

use crate::casscf::dense_jk;
use crate::integrals::{dense_ao_eri, transform_eri_mixed};
use ferric_core::parallel::ParallelContext;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::blas_threads::{opt_in_blas_threads, with_blas_threads};
use ferric_integrals::engine_pool::EnginePool;
use ferric_integrals::operator::Operator;
use ferric_scf::screening::SchwarzBounds;
use ndarray::{Array2, ArrayView2};
use rayon::prelude::*;

/// Default Schwarz threshold of the direct route: the
/// [`ferric_scf::rhf::RhfConfig`] default `integral_thresh`.
pub const DEFAULT_INTEGRAL_THRESH: f64 = 1e-12;

/// How a CASSCF problem gets its two-electron integrals.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CasScfIntegrals {
    /// Direct J/K builds and a direct `(pu|vw)` half transform, recomputed
    /// every macro iteration. Shell quartets with Schwarz product
    /// `Q_bra Q_ket` (times `max|D|` in the J/K builds) below `thresh` are
    /// skipped; `thresh = 0` keeps every quartet.
    Direct {
        /// Integral screening threshold.
        thresh: f64,
    },
    /// The dense `nbas^4` AO block, built once; J/K by parallel GEMVs over
    /// it (`dense_jk`). The oracle of the direct route.
    Dense,
    /// [`CasScfIntegrals::Dense`] when its arrays fit in half of the memory
    /// budget, otherwise [`CasScfIntegrals::Direct`] at
    /// [`DEFAULT_INTEGRAL_THRESH`]. The dense route re-reads one block per
    /// iteration where the direct one recomputes every shell quartet three
    /// times (benzene/cc-pVDZ CAS(6,6): 4.1 s per macro iteration direct,
    /// 0.29 s dense, `crates/ferric-ci/theory.md`), so the direct route is
    /// for the bases whose block does not fit.
    Auto,
}

impl Default for CasScfIntegrals {
    fn default() -> Self {
        CasScfIntegrals::Auto
    }
}

impl CasScfIntegrals {
    /// Elements resident for the life of the problem, and the transient peak
    /// of one `(pu|vw)` build with `nmo` output rows, for the memory plan.
    pub(crate) fn plan_elems(&self, nbas: usize, nmo: usize, nact: usize) -> (usize, usize) {
        let n2 = nbas.saturating_mul(nbas);
        let a2 = nact.saturating_mul(nact);
        let a3 = a2.saturating_mul(nact);
        match self {
            CasScfIntegrals::Dense | CasScfIntegrals::Auto => (
                n2.saturating_mul(n2),
                // First two quarter outputs of `transform_eri_mixed`.
                nact.saturating_mul(n2.saturating_mul(nbas))
                    .saturating_add(a2.saturating_mul(n2)),
            ),
            CasScfIntegrals::Direct { .. } => (
                // Schwarz table (nshells^2 <= nbas^2).
                n2,
                // `(mn|vw)` plus the per-pair blocks it is assembled from,
                // then `(m u|vw)` beside it, then the output.
                direct_paaa_peak_elems(nbas, nmo, nact).max(a3),
            ),
        }
    }

    /// Strict parse of a config route: `"auto"`, `"direct"` (with `thresh`,
    /// default [`DEFAULT_INTEGRAL_THRESH`]) or `"dense"`. Any other spelling,
    /// or a threshold given without `"direct"`, is an error.
    pub fn parse_config_str(route: &str, thresh: Option<f64>) -> Result<Self, String> {
        let r = match (route, thresh) {
            ("direct", t) => CasScfIntegrals::Direct {
                thresh: t.unwrap_or(DEFAULT_INTEGRAL_THRESH),
            },
            ("dense", None) => CasScfIntegrals::Dense,
            ("auto", None) => CasScfIntegrals::Auto,
            ("dense" | "auto", Some(_)) => {
                return Err(format!(
                    "integral_thresh is read by integrals = \"direct\" only, not by {route:?}"
                ))
            }
            (other, _) => {
                return Err(format!(
                    "integrals must be \"auto\", \"direct\" or \"dense\", got {other:?}"
                ))
            }
        };
        r.validate().map_err(|e| e.to_string())?;
        Ok(r)
    }

    /// `"auto"`, `"direct"` or `"dense"`.
    pub fn as_str(&self) -> &'static str {
        match self {
            CasScfIntegrals::Direct { .. } => "direct",
            CasScfIntegrals::Dense => "dense",
            CasScfIntegrals::Auto => "auto",
        }
    }

    /// The concrete route: `Auto` becomes `Dense` when the dense route's
    /// resident and transient arrays (`plan_elems`) take at most half of
    /// `budget_bytes`, else `Direct` at the default threshold.
    pub(crate) fn resolve(self, nbas: usize, nmo: usize, nact: usize, budget_bytes: usize) -> Self {
        match self {
            CasScfIntegrals::Auto => {
                let (r, t) = CasScfIntegrals::Dense.plan_elems(nbas, nmo, nact);
                let bytes = r
                    .saturating_add(t)
                    .saturating_mul(std::mem::size_of::<f64>());
                if bytes <= budget_bytes / 2 {
                    CasScfIntegrals::Dense
                } else {
                    CasScfIntegrals::default_direct()
                }
            }
            other => other,
        }
    }

    /// `Direct` at [`DEFAULT_INTEGRAL_THRESH`].
    pub fn default_direct() -> Self {
        CasScfIntegrals::Direct {
            thresh: DEFAULT_INTEGRAL_THRESH,
        }
    }

    pub(crate) fn validate(&self) -> Result<(), FerricError> {
        if let CasScfIntegrals::Direct { thresh } = self {
            if !(thresh.is_finite() && *thresh >= 0.0) {
                return Err(FerricError::General(format!(
                    "CASSCF: integral threshold must be a finite number >= 0, got {thresh}"
                )));
            }
        }
        Ok(())
    }
}

/// The integral source of one problem.
pub(crate) enum IntegralBackend {
    Dense {
        eri: Vec<f64>,
        nbas: usize,
    },
    // Boxed so the enum is the size of the dense variant.
    Direct {
        bounds: Box<SchwarzBounds>,
        pool: Box<EnginePool>,
        thresh: f64,
    },
}

impl IntegralBackend {
    pub(crate) fn new(prep: &PreparedBasis, route: CasScfIntegrals) -> Result<Self, FerricError> {
        route.validate()?;
        match route {
            CasScfIntegrals::Dense | CasScfIntegrals::Auto => Ok(IntegralBackend::Dense {
                eri: dense_ao_eri(prep)?,
                nbas: prep.nbasis(),
            }),
            CasScfIntegrals::Direct { thresh } => {
                let op = Operator::coulomb();
                Ok(IntegralBackend::Direct {
                    bounds: Box::new(SchwarzBounds::compute(op, prep)?),
                    pool: Box::new(EnginePool::new(
                        op,
                        prep,
                        ferric_integrals::engine_pool::eri_precision(),
                    )?),
                    thresh,
                })
            }
        }
    }

    /// `J[D]`, `K[D]` (AO basis), `J_mn = Σ (mn|ls) D_ls`,
    /// `K_mn = Σ (ml|ns) D_ls`, for a symmetric `d`.
    pub(crate) fn jk(
        &self,
        prep: &PreparedBasis,
        d: &Array2<f64>,
    ) -> Result<(Array2<f64>, Array2<f64>), FerricError> {
        match self {
            IntegralBackend::Dense { eri, nbas } => Ok(dense_jk(eri, *nbas, d)),
            IntegralBackend::Direct {
                bounds,
                pool,
                thresh,
            } => {
                let n = prep.nbasis();
                let mut j = Array2::<f64>::zeros((n, n));
                let mut k = Array2::<f64>::zeros((n, n));
                ferric_scf::rhf::build_jk_with_pool(
                    &ParallelContext::default(),
                    prep,
                    bounds,
                    *thresh,
                    d,
                    &mut j,
                    &mut k,
                    pool,
                    ferric_scf::reduce::default_band_bytes(),
                )?;
                Ok((j, k))
            }
        }
    }

    /// `(pu|vw) = Σ C1[m,p] Ca[n,u] Ca[l,v] Ca[s,w] (mn|ls)`, flat
    /// `[((p n_a + u) n_a + v) n_a + w]`, `p` over the columns of `c1`.
    pub(crate) fn paaa<'c>(
        &self,
        prep: &PreparedBasis,
        c1: ArrayView2<'c, f64>,
        ca: ArrayView2<'c, f64>,
    ) -> Result<Vec<f64>, FerricError> {
        match self {
            // The quarter transforms are four GEMMs outside any parallel
            // region, so they run at the serial-section BLAS thread count.
            IntegralBackend::Dense { eri, nbas } => {
                with_blas_threads(opt_in_blas_threads(), || {
                    transform_eri_mixed(eri, *nbas, [c1, ca, ca, ca])
                })
            }
            IntegralBackend::Direct {
                bounds,
                pool,
                thresh,
            } => direct_paaa(prep, pool, &bounds.q, *thresh, c1, ca),
        }
    }
}

/// Peak elements of [`direct_paaa`] with `nmo` output rows: the per-pair
/// blocks and `(mn|vw)` they are scattered into (each `nbas^2 n_a^2`, the
/// former half of it for the canonical pairs, counted whole), then
/// `(mn|vw)` beside `(m u|vw)` (`nbas n_a^3`), then `(m u|vw)` beside the
/// output (`nmo n_a^3`).
pub(crate) fn direct_paaa_peak_elems(nbas: usize, nmo: usize, nact: usize) -> usize {
    let a2 = nact.saturating_mul(nact);
    let a3 = a2.saturating_mul(nact);
    let x = nbas.saturating_mul(nbas).saturating_mul(a2);
    let y = nbas.saturating_mul(a3);
    let out = nmo.saturating_mul(a3);
    x.saturating_mul(2)
        .max(x.saturating_add(y))
        .max(y.saturating_add(out))
}

/// Direct `(pu|vw)` with `u, v, w` the columns of `ca` and `p` those of `c1`.
///
/// Pass 1, parallel over canonical bra shell pairs `(s1 >= s2)`: every ket
/// shell pair `(s3 >= s4)` with `q[s1,s2] q[s3,s4] >= thresh` is computed
/// (four-fold symmetry: each bra pair owns its row block, so no reduction
/// and the result does not depend on the thread count), and its ket is
/// contracted with `Ca` in two steps,
///
/// ```text
///   T[mn, l, w]  = Σ_s (mn|ls) Ca[s,w]          (both ket orders when s3 != s4)
///   X[mn, v, w]  = Σ_l Ca[l,v] T[mn, l, w]      (one GEMM per bra function pair)
/// ```
///
/// Pass 2: `Y[m, u, vw] = Σ_n Ca[n,u] X[m, n, vw]` (one GEMM per `m`, in
/// parallel) and `(pu|vw) = C1ᵀ Y` (one GEMM).
///
/// Each skipped quartet's integrals are bounded by `q q < thresh`, so every
/// output element moves by at most `thresh` times the product of the four
/// coefficient-column 1-norms per skipped quartet function.
pub(crate) fn direct_paaa(
    prep: &PreparedBasis,
    pool: &EnginePool,
    q: &Array2<f64>,
    thresh: f64,
    c1: ArrayView2<'_, f64>,
    ca: ArrayView2<'_, f64>,
) -> Result<Vec<f64>, FerricError> {
    let nbas = prep.nbasis();
    let na = ca.ncols();
    let np = c1.ncols();
    if ca.nrows() != nbas || c1.nrows() != nbas {
        return Err(FerricError::General(format!(
            "CASSCF direct (pu|vw): coefficient rows {} / {} != nbas {nbas}",
            c1.nrows(),
            ca.nrows()
        )));
    }
    let na2 = na * na;
    if na == 0 || np == 0 {
        return Ok(Vec::new());
    }
    let nsh = prep.nshells();
    let dims = prep.shell_dims();
    let offs = prep.shell_offsets();
    let ca_std = ca.as_standard_layout().into_owned();
    let ca_rows = ca_std
        .as_slice()
        .ok_or_else(|| FerricError::General("CASSCF direct (pu|vw): Ca layout".to_string()))?;
    let pairs: Vec<(usize, usize)> = (0..nsh)
        .flat_map(|s1| (0..=s1).map(move |s2| (s1, s2)))
        .collect();

    // Pass 1: one (n1 n2) x na² block of X per canonical bra pair.
    let blocks: Vec<Vec<f64>> = pairs
        .par_iter()
        .map(|&(s1, s2)| {
            let (n1, n2) = (dims[s1], dims[s2]);
            let n12 = n1 * n2;
            let mut t = vec![0.0f64; n12 * nbas * na];
            let q12 = q[(s1, s2)];
            pool.with(|eng| {
                for &(s3, s4) in &pairs {
                    if q12 * q[(s3, s4)] < thresh {
                        continue;
                    }
                    let Some(blk) = eng.compute_quartet(prep, s1, s2, s3, s4) else {
                        continue;
                    };
                    let (n3, n4) = (dims[s3], dims[s4]);
                    let (o3, o4) = (offs[s3], offs[s4]);
                    for ab in 0..n12 {
                        let tab = &mut t[ab * nbas * na..(ab + 1) * nbas * na];
                        for c in 0..n3 {
                            let la = o3 + c;
                            for d in 0..n4 {
                                let sg = o4 + d;
                                let v = blk[(ab * n3 + c) * n4 + d];
                                if v == 0.0 {
                                    continue;
                                }
                                let cs = &ca_rows[sg * na..(sg + 1) * na];
                                let tl = &mut tab[la * na..(la + 1) * na];
                                for (x, &cw) in tl.iter_mut().zip(cs) {
                                    *x += v * cw;
                                }
                                if s3 != s4 {
                                    let cl = &ca_rows[la * na..(la + 1) * na];
                                    let ts = &mut tab[sg * na..(sg + 1) * na];
                                    for (x, &cw) in ts.iter_mut().zip(cl) {
                                        *x += v * cw;
                                    }
                                }
                            }
                        }
                    }
                }
            });
            // X[ab, v, w] = Σ_l Ca[l, v] T[ab, l, w]: Caᵀ (na x nbas) · T_ab (nbas x na).
            let mut out = vec![0.0f64; n12 * na2];
            for ab in 0..n12 {
                let tab =
                    ArrayView2::from_shape((nbas, na), &t[ab * nbas * na..(ab + 1) * nbas * na])
                        .expect("T block shape is (nbas, na) by construction");
                let xab = ca_std.t().dot(&tab);
                for (dst, src) in out[ab * na2..(ab + 1) * na2].iter_mut().zip(xab.iter()) {
                    *dst = *src;
                }
            }
            out
        })
        .collect();

    // Scatter into X[m, n, vw] and its bra transpose.
    let mut x = vec![0.0f64; nbas * nbas * na2];
    for (&(s1, s2), blk) in pairs.iter().zip(&blocks) {
        let (n1, n2) = (dims[s1], dims[s2]);
        let (o1, o2) = (offs[s1], offs[s2]);
        for a in 0..n1 {
            for b in 0..n2 {
                let src = &blk[(a * n2 + b) * na2..(a * n2 + b + 1) * na2];
                let (m, n) = (o1 + a, o2 + b);
                x[(m * nbas + n) * na2..(m * nbas + n + 1) * na2].copy_from_slice(src);
                x[(n * nbas + m) * na2..(n * nbas + m + 1) * na2].copy_from_slice(src);
            }
        }
    }
    drop(blocks);

    // Pass 2: Y[m, u, vw] = Σ_n Ca[n, u] X[m, n, vw].
    let mut y = vec![0.0f64; nbas * na * na2];
    y.par_chunks_mut(na * na2)
        .zip(x.par_chunks(nbas * na2))
        .for_each(|(ym, xm)| {
            let xm = ArrayView2::from_shape((nbas, na2), xm)
                .expect("X row block shape is (nbas, na²) by construction");
            let prod = ca_std.t().dot(&xm);
            for (dst, src) in ym.iter_mut().zip(prod.iter()) {
                *dst = *src;
            }
        });
    drop(x);
    let ym = ArrayView2::from_shape((nbas, na * na2), &y)
        .map_err(|e| FerricError::General(format!("CASSCF direct (pu|vw) Y shape: {e}")))?;
    let c1t = c1.t().as_standard_layout().into_owned();
    let out = c1t.dot(&ym);
    let out = out.as_standard_layout().into_owned();
    let (v, off) = out.into_raw_vec_and_offset();
    if off.unwrap_or(0) != 0 {
        return Err(FerricError::General(
            "CASSCF direct (pu|vw): unexpected non-zero array offset".to_string(),
        ));
    }
    Ok(v)
}
