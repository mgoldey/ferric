//! Analytic nuclear forces of the k-point closed-shell KS-DFT energy
//! ([`crate::kdft::solve_krks`]) for LDA and GGA functionals (exact-exchange
//! fraction 0), dense-AFT and RS-GDF J.
//!
//! # Energy (per cell; [`crate::kdft`] conventions)
//!
//! ```text
//! E = (1/N_k) Σ_k Re tr[(h(k) + ½ J(k)) D(k)] + E_xc[D] + E_nn
//! E_xc = Σ_g w_g ε(ρ_g, σ_g),     ρ_g   = (1/N_k) Σ_k Σ_μν D(k)_μν φ^k_μ(r_g) φ^k_ν(r_g)*
//!                                 ∇ρ_g  = (1/N_k) Σ_k 2 Re Σ_μν D(k)_μν ∇φ^k_μ φ^k_ν*
//! φ^k_μ(r) = Σ_L e^{ik·L} χ_μ(r − R_A − L)        (μ on atom A)
//! ```
//!
//! `ε = ρ ε_xc`, `σ = |∇ρ|²`, `V_xc(k)_μν = ∂E_xc/∂D(k)_νμ`-equivalent as in
//! [`crate::kdft::KPeriodicXc`].
//!
//! # What is new relative to the k-point HF force ([`crate::kgrad`])
//!
//! A pure functional has no exchange: the whole HF derivative is reused with
//! the exchange fraction `a = 0` (`assemble(.., exch_scale = 0)`: the K
//! derivative walks, the RS-GDF fitted-exchange weights and the Madelung
//! M-term are absent; the Coulomb parts are unchanged) and the energy-weighted
//! Pulay term built from the Fock matrix INCLUDING `V_xc(k)`. The only new
//! derivative terms are those of `E_xc` at fixed density matrix:
//!
//! ```text
//! dE_xc/dR_{A,j} =  X1 (AO motion)  +  X3 (grid-point motion)  +  X4 (weight response)
//! ```
//!
//! # X1: AO derivative at fixed grid
//!
//! Bloch phases `e^{ik·L}` do not depend on the atoms, so an atom moves only
//! the shell centre inside each image: `∂φ^k_μ/∂R_{A,j} = −∂_jφ^k_μ` for
//! `μ ∈ A`, with `∂_jφ^k_μ = Σ_L e^{ik·L} ∂_jχ_μ(r − R_A − L)` (at FIXED
//! fractional `k` the phases are also strain/displacement invariant). With
//! `p^k_μ = Σ_ν D(k)_μν φ^k_ν*` and `m^{k,i}_μ = Σ_ν D(k)_μν (∂_iφ^k_ν)*`
//! (Hermitian `D`, so the `ν ∈ A` terms are the complex conjugates of the
//! `μ ∈ A` ones):
//!
//! ```text
//! dρ/dR_{A,j}     = −(2/N_k) Σ_k Re Σ_{μ∈A} ∂_jφ^k_μ p^k_μ
//! d∇_iρ/dR_{A,j}  = −(2/N_k) Σ_k Re Σ_{μ∈A} [∂_i∂_jφ^k_μ p^k_μ + ∂_jφ^k_μ m^{k,i}_μ]
//! a_{g,A,j} = w_g [ v_ρ dρ/dR_{A,j} + Σ_i c_i d∇_iρ/dR_{A,j} ],   c = 2 v_σ ∇ρ
//! ```
//!
//! It is the Gamma term of [`crate::grad`] with the Bloch AOs complex, `Re`
//! taken after the contraction and a `1/N_k`; GGA needs the AO Hessian
//! (`∂_i∂_jφ`), which is NOT cached (the energy cache holds `φ, ∇φ`), so it is
//! recomputed chunk by chunk ([`crate::dft::LatticeAoHess::eval_bloch`]) with
//! all k inside the chunk. The kernel gates (`ρ > DENSITY_FLOOR`) are the
//! energy's.
//!
//! # X3, X4: the atom-centred grid
//!
//! A grid point is rigidly attached to its home atom `H`: it moves with it.
//! Holding `D` fixed, the integrand at a moving point changes by the point
//! motion, which is exactly minus the sum of the AO-motion terms over all atoms
//! (translating atoms and point together leaves `φ(r)` unchanged):
//!
//! ```text
//! X3:  dE/dR_{H(g),j} −= Σ_A a_{g,A,j}
//! X4:  dE/dR_{A,j}    += Σ_g ε_g · d w_g/dR_{A,j}        (SSF partition response,
//!                         [`crate::dft::build_response_grid`], k-INDEPENDENT)
//! ```
//!
//! The grid, its weights and `dw/dR` do not depend on `k` at all, so X3/X4 are
//! the Gamma code verbatim (`ε_g = ρ_g ε_xc,g` from the k density). On a
//! uniform grid ([`crate::dft::PeriodicGrid::uniform`]) the points are fixed in
//! space and the weights are constants: X3 and X4 are absent by construction
//! (and must be: an XC force that kept them would be wrong), only X1 remains.
//!
//! # Fock guard
//!
//! The Pulay term needs `F(k) = h + J + V_xc`. The entry rebuilds it from the
//! density and compares with the SCF's stored (non-extrapolated) Fock against
//! [`crate::kgrad::FOCK_CONSISTENCY_BAR`], with NO Madelung-shift allowance
//! (a pure functional has no `K`). A result from another functional, grid or
//! J route does not match.
//!
//! # Memory
//!
//! The Bloch AO cache of the energy is `npts·nbf·N_k·16 B` per plane and is
//! never retained here. The derivative pass holds, per concurrently live
//! chunk, `13·nbf·N_k·chunk` complex values (`φ`, `∇φ`, `∇∇φ` for every k) plus
//! one `D·conj(φ)` GEMM buffer and a partial `V_xc` (`2 N_k nbf²`); the chunk
//! shrinks with `N_k·nbf`, the number of concurrent chunks shrinks with the
//! remaining budget, and the pass is pre-flighted on a [`crate::budget::Ledger`]
//! (a typed error when not even one chunk fits). The `ResponseGrid` weight
//! derivatives (`npts·natoms·24 B`) are reserved too; the transient of the XC
//! pass is released before the HF assembly reserves its own arrays.
//!
//! # Scope
//!
//! Pure LDA/GGA only. Global hybrids (exchange fraction ≠ 0), range-separated
//! hybrids, meta-GGA, VV10/double hybrids are refused with typed errors
//! ([`PeriodicDftError::Unsupported`]); unrestricted/open-shell k-point KS has
//! no entry point.

use crate::budget::{bytes_of, Ledger};
use crate::dft::{
    build_response_grid, BlochAoHess, BlochPhase, LatticeAoHess, PeriodicDftError, PeriodicGrid,
    PeriodicGridConfig, DEFAULT_AO_THRESHOLD,
};
use crate::grad::ao_atoms;
use crate::hcore::kpoint::PeriodicHcoreK;
use crate::hcore::PeriodicHcoreConfig;
use crate::kgrad::{
    assemble, check_fock, check_inputs, jk_build, open_ledger, KGradConfig, KGradJk,
    KGradient, KSpins,
};
use crate::kpts::KPointMesh;
use crate::kscf::KScfResult;
use crate::lattice::Cell;
use crate::rsh::resolve_periodic_functional_rsh;
use ferric_core::FerricError;
use ferric_dft::density_on_grid::DensityGrid;
use ferric_dft::grid::GridPoint;
use ferric_dft::libxc::{FunctionalFamily, XcDef};
use ferric_dft::vxc::{closed_kernel, DENSITY_FLOOR};
use ferric_integrals::basis_bridge::PreparedBasis;
use ndarray::{Array1, Array2, Array3, Axis};
use num_complex::Complex64 as C64;
use rayon::prelude::*;

/// Upper bound on points per derivative chunk (the Gamma `XC_GRAD_CHUNK`).
const MAX_CHUNK: usize = 128;
/// Lower bound on points per derivative chunk.
const MIN_CHUNK: usize = 8;
/// Target bytes of AO planes per live chunk (shrinks the chunk with `N_k·nbf`).
const CHUNK_TARGET_BYTES: usize = 64 << 20;
/// Complex AO planes per k and point held by a chunk: `φ`, `∇φ` (3), `∇∇φ` (9).
const AO_PLANES: usize = 13;

/// The grid the energy integrated on, rebuilt for the derivative.
#[derive(Debug, Clone)]
pub enum KXcGrid {
    /// The atom-centred SSF grid of [`PeriodicGrid::build`] (point motion and
    /// weight response included). The SAME config the energy used.
    Atom(PeriodicGridConfig),
    /// [`PeriodicGrid::uniform`] with these mesh counts (points fixed in space,
    /// constant weights: no X3/X4 terms).
    Uniform([usize; 3]),
}

impl Default for KXcGrid {
    fn default() -> Self {
        Self::Atom(PeriodicGridConfig::default())
    }
}

/// Deliberate defects for the mutation tests (`tests/pbc_kxcgrad.rs`). Never
/// set in production.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KXcMutation {
    /// Drop the AO-motion term X1 (and with it X3, which is built from it).
    NoAo,
    /// Drop the grid-point motion X3.
    NoPointMotion,
    /// Drop the weight response X4.
    NoWeightDeriv,
    /// Drop the GGA part of the AO derivative (`c_i` terms).
    NoGga,
    /// Keep the GGA `∂_jφ m^i` term but drop the Hessian term `∂_i∂_jφ p`.
    NoHessian,
    /// Force-side AOs without the Bloch phase (the density keeps its own).
    NoPhase,
    /// Force-side AOs with `e^{−ik·L}` (identity on TRIM-only meshes).
    ConjPhase,
    /// `−1` instead of `−2` in the AO-derivative prefactor (a spin-factor slip).
    WrongFactor,
    /// Missing `1/N_k` in the AO-derivative term.
    NoInvNk,
    /// The Pulay term built from a Fock matrix WITHOUT `V_xc` (the guard is
    /// skipped for this defect: it would refuse it).
    PulayNoVxc,
}

/// Settings of [`kpoint_rks_gradient`].
#[derive(Debug, Clone)]
pub struct KRksGradConfig {
    /// The HF-part settings (budget, nucleus exponent, HF mutation).
    pub grad: KGradConfig,
    /// The grid of the energy.
    pub grid: KXcGrid,
    /// AO truncation of the energy ([`crate::dft::PeriodicXcConfig::ao_threshold`]).
    pub ao_threshold: f64,
    /// TEST ONLY: a deliberate XC-force defect.
    #[doc(hidden)]
    pub mutation: Option<KXcMutation>,
}

impl Default for KRksGradConfig {
    fn default() -> Self {
        Self {
            grad: KGradConfig::default(),
            grid: KXcGrid::default(),
            ao_threshold: DEFAULT_AO_THRESHOLD,
            mutation: None,
        }
    }
}

/// Output of [`kpoint_rks_gradient`].
#[derive(Debug, Clone)]
pub struct KRksGradient {
    /// `dE/dR_A` per cell atom (`natoms × 3`, Hartree/Bohr per cell): HF part
    /// plus XC.
    pub grad: Array2<f64>,
    /// The k-point HF machinery's result with exchange fraction 0 (its `grad`
    /// is the non-XC part; per-term breakdown in `hf.parts`).
    pub hf: KGradient,
    /// XC AO-motion term X1 (before the point-motion X3 is subtracted).
    pub xc_ao: Array2<f64>,
    /// XC grid-point motion X3 (zero on a uniform grid).
    pub xc_point: Array2<f64>,
    /// XC weight response X4 (zero on a uniform grid).
    pub xc_weight: Array2<f64>,
    /// `E_xc` at the SCF density.
    pub e_xc: f64,
    /// `∫ρ` at the SCF density (the grid error against `N_e`).
    pub electrons_on_grid: f64,
    /// Grid points.
    pub n_grid_points: usize,
    /// Points per derivative chunk (shrinks with `N_k · nbf`).
    pub chunk_points: usize,
    /// Chunks evaluated concurrently (bounded by the remaining budget).
    pub live_chunks: usize,
}

/// What an XC response returns (internal: lets tests inject a null XC).
pub(crate) struct XcOut {
    pub(crate) e_xc: f64,
    pub(crate) electrons: f64,
    /// `V_xc(k)`, Hermitian.
    pub(crate) v: Vec<Array2<C64>>,
    pub(crate) ao: Array2<f64>,
    pub(crate) point: Array2<f64>,
    pub(crate) weight: Array2<f64>,
    pub(crate) npts: usize,
    pub(crate) chunk_points: usize,
    pub(crate) live_chunks: usize,
}

/// The XC energy functional's value, potential and fixed-`D` force terms.
pub(crate) trait KXcResponse {
    /// Everything the force needs from the XC energy at density `d`; `ledger`
    /// is the caller's budget (a transient clone is used inside).
    fn evaluate(&self, d: &[Array2<C64>], ledger: &Ledger) -> Result<XcOut, FerricError>;
}

/// The zero XC (`E_xc = 0`, `V_xc = 0`, no force): the exactness anchor
/// against the exchange-free HF machinery.
pub(crate) struct NullXc;

impl KXcResponse for NullXc {
    fn evaluate(&self, d: &[Array2<C64>], _ledger: &Ledger) -> Result<XcOut, FerricError> {
        let n = d[0].nrows();
        Ok(XcOut {
            e_xc: 0.0,
            electrons: 0.0,
            v: d.iter().map(|_| Array2::zeros((n, n))).collect(),
            ao: Array2::zeros((0, 3)),
            point: Array2::zeros((0, 3)),
            weight: Array2::zeros((0, 3)),
            npts: 0,
            chunk_points: 0,
            live_chunks: 0,
        })
    }
}

/// The real XC on the energy's grid.
struct GridXc<'a> {
    cell: &'a Cell,
    prep: &'a PreparedBasis,
    xc: XcDef,
    gga: bool,
    kpts: Vec<[f64; 3]>,
    aoat: Vec<usize>,
    grid: &'a KXcGrid,
    ao_threshold: f64,
    mutation: Option<KXcMutation>,
}

/// `ρ`, `∇ρ`, `σ` of one chunk (the arithmetic of `KPeriodicXc::chunk_density`).
fn chunk_density(
    d: &[Array2<C64>],
    chi: &[Array2<C64>],
    dchi: &[Array3<C64>],
    gga: bool,
) -> DensityGrid {
    let nk = d.len();
    let nbf = d[0].nrows();
    let np = chi[0].ncols();
    let inv_nk = 1.0 / nk as f64;
    let mut rho = Array1::<f64>::zeros(np);
    let mut grad = Array2::<f64>::zeros((3, np));
    for k in 0..nk {
        let p = d[k].dot(&conj(&chi[k]));
        for g in 0..np {
            let mut r = C64::new(0.0, 0.0);
            for mu in 0..nbf {
                r += chi[k][(mu, g)] * p[(mu, g)];
            }
            rho[g] += inv_nk * r.re;
            if gga {
                for ax in 0..3 {
                    let mut gr = C64::new(0.0, 0.0);
                    for mu in 0..nbf {
                        gr += dchi[k][(ax, mu, g)] * p[(mu, g)];
                    }
                    grad[(ax, g)] += inv_nk * 2.0 * gr.re;
                }
            }
        }
    }
    let sigma = (0..np)
        .map(|g| grad[(0, g)].powi(2) + grad[(1, g)].powi(2) + grad[(2, g)].powi(2))
        .collect();
    DensityGrid { rho, grad, sigma }
}

fn conj(m: &Array2<C64>) -> Array2<C64> {
    m.mapv(|z| z.conj())
}

/// One chunk's contributions.
struct Chunk {
    e: f64,
    ne: f64,
    v: Vec<Array2<C64>>,
    ao: Array2<f64>,
    point: Array2<f64>,
    weight: Array2<f64>,
}

impl KXcResponse for GridXc<'_> {
    fn evaluate(&self, d: &[Array2<C64>], ledger: &Ledger) -> Result<XcOut, FerricError> {
        let mut led = ledger.clone();
        let natoms = self.cell.positions().len();
        let nbf = self.prep.nbasis();
        let nk = d.len();
        let mutation = self.mutation;
        // --- Grid (and, for the atom grid, the weight derivatives).
        let (points, dweight): (Vec<GridPoint>, Option<Vec<[f64; 3]>>) = match self.grid {
            KXcGrid::Atom(cfg) => {
                let rg = build_response_grid(self.cell, cfg, &mut led)?;
                (rg.points, Some(rg.dweight))
            }
            KXcGrid::Uniform(n) => (PeriodicGrid::uniform(self.cell, *n)?.points().to_vec(), None),
        };
        let ev = LatticeAoHess::new(
            self.cell,
            self.prep.basis_set(),
            &points,
            self.ao_threshold,
            &led,
        )?;
        if ev.nbf() != nbf || self.aoat.len() != nbf {
            return Err(FerricError::General(format!(
                "k-point XC force: lattice AO count {} != prepared basis {nbf}",
                ev.nbf()
            )));
        }
        // --- Chunking and the memory pre-flight.
        let per_point = |np: usize| -> usize {
            bytes_of((nk * nbf * np) as u64, 16 * AO_PLANES)
                .saturating_add(bytes_of((nbf * np) as u64, 16 * 6))
                .saturating_add(bytes_of((np * natoms * 3) as u64, 8))
        };
        let fixed = bytes_of((nk * nbf * nbf) as u64, 16 * 2);
        let mut chunk_points = MAX_CHUNK;
        while chunk_points > MIN_CHUNK && per_point(chunk_points) > CHUNK_TARGET_BYTES {
            chunk_points /= 2;
        }
        let per_chunk = per_point(chunk_points).saturating_add(fixed);
        // One chunk must fit; more run concurrently while the budget allows.
        led.check(
            &format!(
                "k-point XC force AO/Hessian chunk ({chunk_points} points, nbf = {nbf}, N_k = {nk})"
            ),
            per_chunk,
        )?;
        let par = rayon::current_num_threads()
            .max(1)
            .min((led.remaining() / per_chunk.max(1)).max(1));
        led.reserve(
            &format!("k-point XC force chunks ({par} x {chunk_points} points)"),
            per_chunk.saturating_mul(par),
        )?;
        let chunks = LatticeAoHess::chunks(&points, chunk_points);
        let inv_nk = 1.0 / nk as f64;
        let use_gga = self.gga && mutation != Some(KXcMutation::NoGga);
        let alt_phase = match mutation {
            Some(KXcMutation::NoPhase) => Some(BlochPhase::None),
            Some(KXcMutation::ConjPhase) => Some(BlochPhase::Conj),
            _ => None,
        };

        let run = |idx: &Vec<usize>| -> Result<Chunk, FerricError> {
            let pts: Vec<GridPoint> = idx.iter().map(|&i| points[i]).collect();
            let xyz: Vec<[f64; 3]> = pts.iter().map(|g| g.xyz).collect();
            let np = pts.len();
            let bl: BlochAoHess = ev.eval_bloch(&xyz, &self.kpts, BlochPhase::True, self.gga)?;
            let dens = chunk_density(d, &bl.chi, &bl.dchi, self.gga);
            let kern = closed_kernel(&dens, None, &self.xc);
            let live = |g: usize| dens.rho[g] > DENSITY_FLOOR;
            let mut e = 0.0;
            let mut ne = 0.0;
            for g in 0..np {
                e += pts[g].weight * dens.rho[g] * kern.exc[g];
                ne += pts[g].weight * dens.rho[g];
            }
            // Gated kernel vectors (the energy's gates).
            let vr: Vec<f64> = (0..np)
                .map(|g| if live(g) { kern.vrho[g] } else { 0.0 })
                .collect();
            let c = Array2::from_shape_fn((3, np), |(i, g)| {
                if self.gga && live(g) {
                    2.0 * kern.vsigma[g] * dens.grad[(i, g)]
                } else {
                    0.0
                }
            });
            // --- V_xc(k) (the arithmetic of `KPeriodicXc::build`).
            let a: Vec<f64> = (0..np).map(|g| pts[g].weight * vr[g]).collect();
            let mut v: Vec<Array2<C64>> = Vec::with_capacity(nk);
            for k in 0..nk {
                let scale = |fac: &[f64]| {
                    let mut m = conj(&bl.chi[k]);
                    for g in 0..np {
                        m.column_mut(g).mapv_inplace(|z| z * fac[g]);
                    }
                    m
                };
                let mut vk = scale(&a).dot(&bl.chi[k].t());
                if self.gga {
                    for ax in 0..3 {
                        let f: Vec<f64> = (0..np).map(|g| pts[g].weight * c[(ax, g)]).collect();
                        let d_ax = bl.dchi[k].index_axis(Axis(0), ax);
                        let m = scale(&f).dot(&d_ax.t());
                        let mh = conj(&m).t().to_owned();
                        vk += &m;
                        vk += &mh;
                    }
                }
                v.push(vk);
            }
            // --- Force: a[g, A, j] (module doc, X1).
            let alt: Option<BlochAoHess> = alt_phase
                .map(|ph| ev.eval_bloch(&xyz, &self.kpts, ph, self.gga))
                .transpose()?;
            let fb: &BlochAoHess = alt.as_ref().unwrap_or(&bl);
            let pref = if mutation == Some(KXcMutation::WrongFactor) {
                -1.0
            } else {
                -2.0
            };
            let kscale = if mutation == Some(KXcMutation::NoInvNk) {
                1.0
            } else {
                inv_nk
            };
            let mut a_pt = Array3::<f64>::zeros((np, natoms, 3));
            if mutation != Some(KXcMutation::NoAo) {
                for k in 0..nk {
                    let p = d[k].dot(&conj(&fb.chi[k]));
                    let md: Option<[Array2<C64>; 3]> = use_gga.then(|| {
                        std::array::from_fn(|i| {
                            d[k].dot(&conj(&fb.dchi[k].index_axis(Axis(0), i).to_owned()))
                        })
                    });
                    for g in 0..np {
                        let wg = pref * pts[g].weight * kscale;
                        for mu in 0..nbf {
                            let at = self.aoat[mu];
                            let pm = p[(mu, g)];
                            for j in 0..3 {
                                let dj = fb.dchi[k][(j, mu, g)];
                                let mut t = dj * pm * vr[g];
                                if let Some(md) = md.as_ref() {
                                    for i in 0..3 {
                                        let mut inner = dj * md[i][(mu, g)];
                                        if mutation != Some(KXcMutation::NoHessian) {
                                            inner += fb.ddchi[k][(j, i, mu, g)] * pm;
                                        }
                                        t += inner * c[(i, g)];
                                    }
                                }
                                a_pt[(g, at, j)] += wg * t.re;
                            }
                        }
                    }
                }
            }
            let mut ao = Array2::<f64>::zeros((natoms, 3));
            let mut point = Array2::<f64>::zeros((natoms, 3));
            let mut weight = Array2::<f64>::zeros((natoms, 3));
            for (g, &gi) in idx.iter().enumerate() {
                let home = pts[g].home_atom;
                let mut tot = [0.0_f64; 3];
                for at in 0..natoms {
                    for j in 0..3 {
                        let val = a_pt[(g, at, j)];
                        ao[(at, j)] += val;
                        tot[j] += val;
                        if let Some(dw) = dweight.as_ref() {
                            let eps = dens.rho[g] * kern.exc[g];
                            weight[(at, j)] += eps * dw[gi * natoms + at][j];
                        }
                    }
                }
                if dweight.is_some() {
                    for j in 0..3 {
                        point[(home, j)] -= tot[j];
                    }
                }
            }
            Ok(Chunk {
                e,
                ne,
                v,
                ao,
                point,
                weight,
            })
        };

        // Concurrency-sized batches reduced in chunk order: memory is bounded by
        // `par` chunks and the sums do not depend on the thread count.
        let mut e_xc = 0.0;
        let mut electrons = 0.0;
        let mut v: Vec<Array2<C64>> = (0..nk).map(|_| Array2::zeros((nbf, nbf))).collect();
        let mut ao = Array2::<f64>::zeros((natoms, 3));
        let mut point = Array2::<f64>::zeros((natoms, 3));
        let mut weight = Array2::<f64>::zeros((natoms, 3));
        for batch in chunks.chunks(par) {
            let outs: Vec<Result<Chunk, FerricError>> = batch.par_iter().map(&run).collect();
            for out in outs {
                let c = out?;
                e_xc += c.e;
                electrons += c.ne;
                for (acc, x) in v.iter_mut().zip(&c.v) {
                    *acc += x;
                }
                ao += &c.ao;
                point += &c.point;
                weight += &c.weight;
            }
        }
        if !e_xc.is_finite()
            || v.iter().any(|m| m.iter().any(|z| !z.re.is_finite()))
            || ao
                .iter()
                .chain(point.iter())
                .chain(weight.iter())
                .any(|x| !x.is_finite())
        {
            return Err(FerricError::General(format!(
                "k-point XC force: non-finite E_xc/V_xc/force"
            )));
        }
        // Hermitise (as `KPeriodicXc::build`).
        for m in v.iter_mut() {
            let h = conj(m).t().to_owned();
            *m = (&*m + &h).mapv(|z| z * 0.5);
        }
        match mutation {
            Some(KXcMutation::NoPointMotion) => point.fill(0.0),
            Some(KXcMutation::NoWeightDeriv) => weight.fill(0.0),
            _ => {}
        }
        Ok(XcOut {
            e_xc,
            electrons,
            v,
            ao,
            point,
            weight,
            npts: points.len(),
            chunk_points,
            live_chunks: par,
        })
    }
}

/// Hermitian part.
fn hermitize(m: &Array2<C64>) -> Array2<C64> {
    let h = conj(m).t().to_owned();
    (m + &h).mapv(|z| z * 0.5)
}

/// Refuse what the pure-functional force does not cover; returns the
/// resolved functional.
fn resolve_pure(functional: &str) -> Result<XcDef, FerricError> {
    let (xc, exx, rsh) = resolve_periodic_functional_rsh(functional)?;
    if rsh.is_some() {
        return Err(PeriodicDftError::Unsupported {
            feature: "range-separated hybrid force at k",
            reason: format!(
                "{functional}: the k-point XC force covers LDA and GGA functionals with no exact \
                 exchange; the screened-exchange derivative is not implemented"
            ),
        }
        .into());
    }
    if exx != 0.0 {
        return Err(PeriodicDftError::Unsupported {
            feature: "global hybrid force at k",
            reason: format!(
                "{functional} has exact-exchange fraction {exx}; the k-point XC force covers pure \
                 LDA/GGA functionals only (the exchange derivative with a scaled K is not \
                 validated)"
            ),
        }
        .into());
    }
    if xc
        .funcs
        .iter()
        .any(|f| matches!(f.family(), FunctionalFamily::MetaGga))
    {
        return Err(PeriodicDftError::Unsupported {
            feature: "meta-GGA force at k",
            reason: format!("{functional}: the tau derivative (AO Hessian terms) is not implemented"),
        }
        .into());
    }
    Ok(xc)
}

/// k-point closed-shell RKS nuclear gradient (module doc) of a converged
/// [`crate::kdft::solve_krks`] result for a PURE LDA/GGA `functional`.
///
/// * `hcore_cfg`, `hk`, `jk` — the config/output of the hcore and the J source
///   the SCF used (dense AFT or RS-GDF), as for [`crate::kgrad::kpoint_rhf_gradient`]
///   (including the SR column-rotation setting of the energy).
/// * `scf` — `KRksResult::scf`.
/// * `cfg.grid`/`cfg.ao_threshold` — the grid and AO truncation of the energy.
///
/// Refuses (typed errors) hybrids, range-separated hybrids, meta-GGA,
/// VV10/double hybrids, and an SCF whose Fock the rebuilt `h + J + V_xc` does
/// not reproduce.
#[allow(clippy::too_many_arguments)]
pub fn kpoint_rks_gradient(
    cell: &Cell,
    prep: &PreparedBasis,
    mesh: &KPointMesh,
    hcore_cfg: &PeriodicHcoreConfig,
    hk: &PeriodicHcoreK,
    jk: KGradJk<'_>,
    scf: &KScfResult,
    functional: &str,
    cfg: &KRksGradConfig,
) -> Result<KRksGradient, FerricError> {
    let xc = resolve_pure(functional)?;
    let gga = xc
        .funcs
        .iter()
        .any(|f| !matches!(f.family(), FunctionalFamily::Lda));
    let resp = GridXc {
        cell,
        prep,
        xc,
        gga,
        kpts: mesh.kpts().to_vec(),
        aoat: ao_atoms(prep),
        grid: &cfg.grid,
        ao_threshold: cfg.ao_threshold,
        mutation: cfg.mutation,
    };
    rks_gradient_with(
        "kpoint_rks_gradient",
        cell,
        prep,
        mesh,
        hcore_cfg,
        hk,
        jk,
        scf,
        &resp,
        cfg,
    )
}

/// TEST/DIAGNOSTIC: the exchange-free machinery with a zero XC (`E_xc = 0`,
/// `V_xc = 0`) on a Hartree-only SCF (`solve_krks_injected` with a zero XC
/// builder and `a = 0`). The exactness anchor of the XC plumbing against
/// [`kpoint_hartree_gradient`].
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn kpoint_null_xc_gradient(
    cell: &Cell,
    prep: &PreparedBasis,
    mesh: &KPointMesh,
    hcore_cfg: &PeriodicHcoreConfig,
    hk: &PeriodicHcoreK,
    jk: KGradJk<'_>,
    scf: &KScfResult,
    cfg: &KRksGradConfig,
) -> Result<KRksGradient, FerricError> {
    rks_gradient_with(
        "kpoint_null_xc_gradient",
        cell,
        prep,
        mesh,
        hcore_cfg,
        hk,
        jk,
        scf,
        &NullXc,
        cfg,
    )
}

/// The shared body: XC response, Fock guard, exchange-free HF assembly.
#[allow(clippy::too_many_arguments)]
fn rks_gradient_with(
    who: &str,
    cell: &Cell,
    prep: &PreparedBasis,
    mesh: &KPointMesh,
    hcore_cfg: &PeriodicHcoreConfig,
    hk: &PeriodicHcoreK,
    jk: KGradJk<'_>,
    scf: &KScfResult,
    xc: &dyn KXcResponse,
    cfg: &KRksGradConfig,
) -> Result<KRksGradient, FerricError> {
    check_inputs(
        who,
        cell,
        prep,
        mesh,
        hcore_cfg,
        hk,
        &jk,
        &[&scf.densities[..]],
        scf.converged,
        &scf.kpts,
    )?;
    let natoms = cell.positions().len();
    let nk = mesh.nk();
    let mut ledger = open_ledger(cell, prep, mesh, &cfg.grad, false)?;
    let d = scf.densities.clone();
    let mut xo = xc.evaluate(&d, &ledger)?;
    if xo.ao.nrows() == 0 {
        xo.ao = Array2::zeros((natoms, 3));
        xo.point = Array2::zeros((natoms, 3));
        xo.weight = Array2::zeros((natoms, 3));
    }
    // F = h + J + V_xc (no exchange); the stored Fock of the SCF must agree.
    let (j, _k) = jk_build(&jk, 0.0, &d)?;
    let with_vxc = cfg.mutation != Some(KXcMutation::PulayNoVxc);
    let f: Vec<Array2<C64>> = (0..nk)
        .map(|q| {
            let base = hermitize(&(&hk.h[q] + &j[q]));
            if with_vxc {
                hermitize(&(&base + &xo.v[q]))
            } else {
                base
            }
        })
        .collect();
    let mismatch = if with_vxc {
        check_fock(who, "fock", &f, &scf.fock)?
    } else {
        0.0
    };
    let mut hf = assemble(
        cell,
        prep,
        mesh,
        hcore_cfg,
        hk,
        &jk,
        KSpins::Restricted { d, f },
        0.0,
        0.0,
        &cfg.grad,
        &mut ledger,
    )?;
    hf.fock_mismatch = mismatch;
    let grad = &(&(&hf.grad + &xo.ao) + &xo.point) + &xo.weight;
    if grad.iter().any(|v| !v.is_finite()) {
        return Err(FerricError::General(format!("{who}: non-finite force")));
    }
    Ok(KRksGradient {
        grad,
        hf,
        xc_ao: xo.ao,
        xc_point: xo.point,
        xc_weight: xo.weight,
        e_xc: xo.e_xc,
        electrons_on_grid: xo.electrons,
        n_grid_points: xo.npts,
        chunk_points: xo.chunk_points,
        live_chunks: xo.live_chunks,
    })
}

/// TEST/DIAGNOSTIC: the exchange-free HF machinery on a Hartree-only SCF
/// (`F = h + J`), the independent twin of [`kpoint_null_xc_gradient`] (it
/// shares `assemble` but none of the XC plumbing or the KS Fock guard).
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn kpoint_hartree_gradient(
    cell: &Cell,
    prep: &PreparedBasis,
    mesh: &KPointMesh,
    hcore_cfg: &PeriodicHcoreConfig,
    hk: &PeriodicHcoreK,
    jk: KGradJk<'_>,
    scf: &KScfResult,
    cfg: &KGradConfig,
) -> Result<KGradient, FerricError> {
    let who = "kpoint_hartree_gradient";
    check_inputs(
        who,
        cell,
        prep,
        mesh,
        hcore_cfg,
        hk,
        &jk,
        &[&scf.densities[..]],
        scf.converged,
        &scf.kpts,
    )?;
    let mut ledger = open_ledger(cell, prep, mesh, cfg, false)?;
    let d = scf.densities.clone();
    let (j, _k) = jk_build(&jk, 0.0, &d)?;
    let f: Vec<Array2<C64>> = (0..mesh.nk())
        .map(|q| hermitize(&(&hk.h[q] + &j[q])))
        .collect();
    let mismatch = check_fock(who, "fock", &f, &scf.fock)?;
    let mut out = assemble(
        cell,
        prep,
        mesh,
        hcore_cfg,
        hk,
        &jk,
        KSpins::Restricted { d, f },
        0.0,
        0.0,
        cfg,
        &mut ledger,
    )?;
    out.fock_mismatch = mismatch;
    Ok(out)
}
