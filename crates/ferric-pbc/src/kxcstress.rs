//! Analytic stress tensor of the k-point closed-shell KS-DFT energy
//! ([`crate::kdft::solve_krks`]) for LDA and GGA functionals (exact-exchange
//! fraction 0), dense-AFT and RS-GDF J: the strain twin of [`crate::kxcgrad`]
//! and the k-point twin of the Gamma XC stress (`crate::stress`).
//!
//! # Convention
//!
//! That of [`crate::kstress`]: `σ_ab = (1/Ω) dE/dε_ab`, `a_i → (1 + ε) a_i`,
//! `R_A → (1 + ε) R_A`, Miller indices and fractional `k` FIXED (so the Bloch
//! phases `e^{ik·L}` and the residue binning are strain-invariant and there is
//! no k-strain term).
//!
//! # Energy (per cell; [`crate::kxcgrad`] conventions)
//!
//! ```text
//! E = (1/N_k) Σ_k Re tr[(h(k) + ½ J(k)) D(k)] + E_xc[D] + E_nn
//! E_xc = Σ_g w_g ε(ρ_g, σ_g)
//! ```
//!
//! # Terms
//!
//! 1. **Exchange-free HF part**: the whole k HF stress with exchange fraction
//!    `a = 0` ([`crate::kstress`]'s `assemble(.., exch_scale = 0)`: the dense K
//!    walks, the RS-GDF fitted-exchange weights and the Madelung M-term and
//!    strain are absent; `exch_scale = 1` is bitwise the HF stress), with the
//!    Pulay weight `W = D F D` built from `F = h + J + V_xc`.
//! 2. **XC at fixed `D`**, `dE_xc/dε_ab` = AO strain + grid-point motion
//!    (`xc_ao`) + weight response (`xc_weight`). Holding `D` fixed, a grid point
//!    `r_g = O_g + δ_g` rigidly attached to an anchor `O_g` (offset `δ_g` not
//!    strained) and an image shell centre `X_L → (1 + ε) X_L` give
//!
//!    ```text
//!    dφ^k_μ(r_g)/dε_ab        = Σ_L e^{ik·L} ∂_aχ_μ(r_g − X_L) (O_g − X_L)_b     =: m1^k[a,b]_μ
//!    d(∂_iφ^k_μ)/dε_ab        = Σ_L e^{ik·L} ∂_i∂_aχ_μ(r_g − X_L) (O_g − X_L)_b  =: m2^k[3i+a,b]_μ
//!    ```
//!
//!    (the gradient operator acts on the strained function: no further term;
//!    the Cartesian gradient of the Bloch sum is evaluated at the strained
//!    argument). With `p^k_μ = Σ_ν D(k)_μν φ^k_ν*` and `md^{k,i}_μ =
//!    Σ_ν D(k)_μν (∂_iφ^k_ν)*` (Hermitian `D`, as in [`crate::kxcgrad`]):
//!
//!    ```text
//!    dρ/dε_ab        = (2/N_k) Σ_k Re Σ_μ m1^k[a,b]_μ p^k_μ
//!    d∇_iρ/dε_ab     = (2/N_k) Σ_k Re Σ_μ [ m2^k[3i+a,b]_μ p^k_μ + m1^k[a,b]_μ md^{k,i}_μ ]
//!    xc_ao[a,b]      = Σ_g w_g [ v_ρ dρ/dε_ab + Σ_i c_i d∇_iρ/dε_ab ],    c = 2 v_σ ∇ρ
//!    xc_weight[a,b]  = Σ_g ε_g dw_g/dε_ab,      ε_g = ρ_g ε_xc,g
//!    ```
//!
//!    This is the Gamma `xc_stress` with a phase per image, complex arithmetic,
//!    `Re` after the contraction and a `1/N_k`; the weight response is
//!    k-independent.
//!
//! # Which grid, which anchor, which weight response
//!
//! * **Uniform grid** ([`crate::dft::PeriodicGrid::uniform`], points at fixed
//!   FRACTIONAL coordinates, weight `Ω/N`): every point is its own anchor
//!   (`O_g = r_g`: the point moves as `r → (1 + ε) r`), and the weight response
//!   is the pure volume term `dw_g/dε_ab = w_g δ_ab`. There is no partition and
//!   no home atom; the rotation-invariant energy gives a symmetric stress.
//! * **Atom-centred grid** ([`crate::dft::PeriodicGrid::build`]): the anchor is
//!   the home atom (the point moves with it, offset unstrained, so `w_rl` is
//!   unchanged) and the weight response is the SSF partition derivative of
//!   `dft::build_strain_grid` (k-independent, `dW/dε` per point).
//!   The grid is built from lab-fixed Lebedev offsets, so the ENERGY is not
//!   invariant under a rigid rotation of the cell and the stress carries an
//!   antisymmetric part `σ_ab − σ_ba` that vanishes only as the grid is
//!   refined. It is the stress of the energy as computed (FD of `E(ε)` for each
//!   of the nine components reproduces it, the antisymmetric part included);
//!   [`KRksStress::sigma`] is returned UNSYMMETRISED and
//!   [`KRksStress::symmetric`] / [`KRksStress::antisymmetry`] are the
//!   diagnostics (the Gamma convention). Measured `max|σ_ab − σ_ba|` for H2 in
//!   the 4 Bohr cube (1x1x3, PBE; `|σ|` is about 6e-3 Ha/Bohr³): 1.1e-4 on a
//!   (20, 50) grid, 7.1e-5 (30, 110), 7.8e-5 (40, 110), 7.6e-6 on the default
//!   (75, 302); 4e-19 on a uniform grid. The 1x1x3 mesh and the explicit
//!   supercell give the identical number (1.14e-4 at (20, 50)), so it is a
//!   property of the grid, not of the k sampling. A cell optimizer should use
//!   [`KRksStress::symmetric`] on a grid at least as fine as the default.
//!
//! The grid, its weights and `dW/dε` do not depend on `k`; the AO list, the
//! per-shell extents, the density gates (`ρ > DENSITY_FLOOR`) and the `ε_g`
//! (ungated) are the energy's, as in the force.
//!
//! # Fock guard
//!
//! As [`crate::kxcgrad`]: `F = h + J + V_xc` is rebuilt and compared with the
//! SCF's stored Fock at [`crate::kgrad::FOCK_CONSISTENCY_BAR`].
//!
//! # Memory
//!
//! The Bloch AO cache of the energy is never retained. A live chunk holds
//! `40·nbf·N_k·chunk` complex values (`φ`, `∇φ`, `m1` (9), `m2` (27) for every
//! k) plus six `nbf·chunk` work planes and a partial `V_xc`; the chunk shrinks
//! with `N_k·nbf`, the number of concurrent chunks with the remaining budget,
//! and the pass is pre-flighted on the memory ledger (typed error
//! when not even one chunk fits). The strain-grid derivatives (`npts·96` B) are
//! reserved with the points.
//!
//! # Scope
//!
//! Pure LDA/GGA only; hybrids (exact-exchange fraction ≠ 0), range-separated
//! hybrids, meta-GGA, VV10/double hybrids and unrestricted KS keep typed
//! errors.

use crate::budget::{bytes_of, Ledger};
use crate::dft::{
    build_strain_grid, BlochAoStrain, BlochPhase, LatticeAoHess, PeriodicDftError, PeriodicGrid,
    DEFAULT_AO_THRESHOLD,
};
use crate::grad::ao_atoms;
use crate::hcore::kpoint::PeriodicHcoreK;
use crate::hcore::PeriodicHcoreConfig;
use crate::kgrad::{check_fock, check_inputs, jk_build, KGradJk, KSpins};
use crate::kpts::KPointMesh;
use crate::kscf::KScfResult;
use crate::kstress::{
    assemble, grad_jk, open_ledger, refuse_unsupported, KStress, KStressConfig, KStressJk,
};
use crate::kxcgrad::{chunk_density, chunk_vxc, conj, hermitize, KXcGrid};
use crate::lattice::Cell;
use crate::rsh::resolve_periodic_functional_rsh;
use crate::stress::{add_into, scaled, Mat3, ZERO3};
use ferric_core::FerricError;
use ferric_dft::grid::GridPoint;
use ferric_dft::libxc::{FunctionalFamily, XcDef};
use ferric_dft::vxc::{closed_kernel, DENSITY_FLOOR};
use ferric_integrals::basis_bridge::PreparedBasis;
use ndarray::{Array2, Axis};
use num_complex::Complex64 as C64;
use rayon::prelude::*;

/// Remedy text of the KS stress's Fock guard.
const KS_REMEDY: &str = "the functional, grid, AO threshold and J/K route must be the SCF's own (a different functional gives a different V_xc)";

/// Upper bound on points per strain chunk.
const MAX_CHUNK: usize = 128;
/// Lower bound on points per strain chunk.
const MIN_CHUNK: usize = 8;
/// Target bytes of AO planes per live chunk.
const CHUNK_TARGET_BYTES: usize = 64 << 20;
/// Complex AO planes per k and point held by a chunk: `φ` (1), `∇φ` (3), `m1`
/// (9), `m2` (27).
const AO_PLANES: usize = 40;

/// Deliberate defects for the mutation tests (`tests/pbc_kxcstress.rs`). Never
/// set in production.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KXcStressMutation {
    /// Drop the whole XC AO-strain term (point motion included).
    NoAo,
    /// Drop the grid-point motion: anchors at the origin, so only the image
    /// centres strain (`O_g = 0`).
    NoPointMotion,
    /// Atom grid: anchor the point at ITSELF instead of its home atom (the
    /// Gamma `XcPointFixed`); an identity on a uniform grid.
    PointSelf,
    /// Drop the weight response (the partition derivative on an atom grid, the
    /// `w δ_ab` volume term on a uniform grid).
    NoWeightDeriv,
    /// Drop the GGA part of the AO strain (the `c_i` terms).
    NoGga,
    /// Drop the Hessian-moment term `m2 p` of the GGA strain.
    NoHessian,
    /// Drop the first-moment gradient term `m1 md` of the GGA strain.
    NoStrainMoment,
    /// Strain-side AOs without the Bloch phase (the density keeps its own).
    NoPhase,
    /// Strain-side AOs with `e^{−ik·L}` (identity on TRIM-only meshes).
    ConjPhase,
    /// `1` instead of `2` in the AO prefactor (a spin-factor slip).
    WrongFactor,
    /// Missing `1/N_k` in the AO strain term.
    NoInvNk,
    /// The Pulay term built from a Fock matrix WITHOUT `V_xc` (the guard is
    /// skipped for this defect: it would refuse it).
    PulayNoVxc,
}

/// Settings of [`kpoint_rks_stress`].
#[derive(Debug, Clone)]
pub struct KRksStressConfig {
    /// The HF-part settings (budget, nucleus exponent, HF mutation).
    pub stress: KStressConfig,
    /// The grid of the energy.
    pub grid: KXcGrid,
    /// AO truncation of the energy ([`crate::dft::PeriodicXcConfig::ao_threshold`]).
    pub ao_threshold: f64,
    /// Cap on concurrently live strain chunks (`None` = the thread count,
    /// further limited by the remaining budget). The result does not depend on it.
    pub max_live_chunks: Option<usize>,
    /// TEST ONLY: a deliberate XC-stress defect.
    #[doc(hidden)]
    pub mutation: Option<KXcStressMutation>,
}

impl Default for KRksStressConfig {
    fn default() -> Self {
        Self {
            stress: KStressConfig::default(),
            grid: KXcGrid::default(),
            ao_threshold: DEFAULT_AO_THRESHOLD,
            max_live_chunks: None,
            mutation: None,
        }
    }
}

/// Output of [`kpoint_rks_stress`].
#[derive(Debug, Clone)]
pub struct KRksStress {
    /// `dE/dε_ab` (Hartree per cell): HF part + XC.
    pub de_deps: Mat3,
    /// `σ_ab = (1/Ω) dE/dε_ab` (Hartree/Bohr³). NOT symmetrised.
    pub sigma: Mat3,
    /// The exchange-free k HF stress (its `de_deps`/`sigma` are the non-XC
    /// part; per-term breakdown in `hf.parts`).
    pub hf: KStress,
    /// XC AO strain plus grid-point motion (`dE_xc/dε` through the strained
    /// Bloch AOs; `Σ_g w_g [..]` of the module doc).
    pub xc_ao: Mat3,
    /// XC weight response: the SSF partition derivative on an atom grid, the
    /// volume term `ε_g w_g δ_ab` on a uniform grid.
    pub xc_weight: Mat3,
    /// `E_xc` at the SCF density.
    pub e_xc: f64,
    /// `∫ρ` at the SCF density.
    pub electrons_on_grid: f64,
    /// Grid points.
    pub n_grid_points: usize,
    /// Points per strain chunk.
    pub chunk_points: usize,
    /// Chunks evaluated concurrently.
    pub live_chunks: usize,
    /// Cell volume (Bohr³).
    pub volume: f64,
}

impl KRksStress {
    /// `(σ + σᵀ)/2`.
    pub fn symmetric(&self) -> Mat3 {
        let mut o = ZERO3;
        for a in 0..3 {
            for b in 0..3 {
                o[a][b] = 0.5 * (self.sigma[a][b] + self.sigma[b][a]);
            }
        }
        o
    }

    /// `max_ab |σ_ab − σ_ba|` (Hartree/Bohr³): roundoff on a uniform grid; on
    /// an atom grid a grid-size-dependent number (module doc).
    pub fn antisymmetry(&self) -> f64 {
        let mut m = 0.0_f64;
        for a in 0..3 {
            for b in 0..3 {
                m = m.max((self.sigma[a][b] - self.sigma[b][a]).abs());
            }
        }
        m
    }

    /// Pressure `P = −tr(σ)/3` (Hartree/Bohr³).
    pub fn pressure(&self) -> f64 {
        -(self.sigma[0][0] + self.sigma[1][1] + self.sigma[2][2]) / 3.0
    }
}

/// What an XC stress response returns (internal: lets tests inject a null XC).
pub(crate) struct XcStressOut {
    pub(crate) e_xc: f64,
    pub(crate) electrons: f64,
    /// `V_xc(k)`, Hermitian.
    pub(crate) v: Vec<Array2<C64>>,
    pub(crate) ao: Mat3,
    pub(crate) weight: Mat3,
    pub(crate) npts: usize,
    pub(crate) chunk_points: usize,
    pub(crate) live_chunks: usize,
}

/// The XC energy functional's value, potential and fixed-`D` strain terms.
pub(crate) trait KXcStressResponse {
    /// Everything the stress needs from the XC energy at density `d`; `ledger`
    /// is the caller's budget (a transient clone is used inside).
    fn evaluate(&self, d: &[Array2<C64>], ledger: &Ledger) -> Result<XcStressOut, FerricError>;
}

/// The zero XC: the exactness anchor against the exchange-free HF machinery.
pub(crate) struct NullXc;

impl KXcStressResponse for NullXc {
    fn evaluate(&self, d: &[Array2<C64>], _ledger: &Ledger) -> Result<XcStressOut, FerricError> {
        let n = d[0].nrows();
        Ok(XcStressOut {
            e_xc: 0.0,
            electrons: 0.0,
            v: d.iter().map(|_| Array2::zeros((n, n))).collect(),
            ao: ZERO3,
            weight: ZERO3,
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
    grid: &'a KXcGrid,
    ao_threshold: f64,
    max_live_chunks: Option<usize>,
    mutation: Option<KXcStressMutation>,
}

/// One chunk's contributions.
struct Chunk {
    e: f64,
    ne: f64,
    v: Vec<Array2<C64>>,
    ao: Mat3,
    weight: Mat3,
}

/// Grid points, per-point strain derivative of the weight and anchors.
struct StrainPoints {
    points: Vec<GridPoint>,
    /// `dw_g/dε_ab`.
    dweight: Vec<Mat3>,
    /// `O_g`.
    anchors: Vec<[f64; 3]>,
}

impl GridXc<'_> {
    fn strain_points(&self, led: &mut Ledger) -> Result<StrainPoints, FerricError> {
        match self.grid {
            KXcGrid::Atom(cfg) => {
                let sg = build_strain_grid(self.cell, cfg, led)?;
                Ok(StrainPoints {
                    points: sg.points,
                    dweight: sg.dweight,
                    anchors: sg.anchors,
                })
            }
            KXcGrid::Uniform(n) => {
                let points = PeriodicGrid::uniform(self.cell, *n)?.points().to_vec();
                // Points at fixed fractional coordinates: `r → (1 + ε) r`
                // (the point is its own anchor) and `w = Ω/N` scales with the
                // volume: `dw/dε_ab = w δ_ab`.
                let dweight = points
                    .iter()
                    .map(|p| {
                        let mut m = ZERO3;
                        for (a, row) in m.iter_mut().enumerate() {
                            row[a] = p.weight;
                        }
                        m
                    })
                    .collect();
                let anchors = points.iter().map(|p| p.xyz).collect();
                Ok(StrainPoints {
                    points,
                    dweight,
                    anchors,
                })
            }
        }
    }
}

impl KXcStressResponse for GridXc<'_> {
    fn evaluate(&self, d: &[Array2<C64>], ledger: &Ledger) -> Result<XcStressOut, FerricError> {
        let mut led = ledger.clone();
        let nbf = self.prep.nbasis();
        let nk = d.len();
        let mutation = self.mutation;
        let sp = self.strain_points(&mut led)?;
        let points = &sp.points;
        let ev = LatticeAoHess::new(
            self.cell,
            self.prep.basis_set(),
            points,
            self.ao_threshold,
            &led,
        )?;
        if ev.nbf() != nbf || ao_atoms(self.prep).len() != nbf {
            return Err(FerricError::General(format!(
                "k-point XC stress: lattice AO count {} != prepared basis {nbf}",
                ev.nbf()
            )));
        }
        // --- Chunking and the memory pre-flight.
        let per_point = |np: usize| -> usize {
            bytes_of((nk * nbf * np) as u64, 16 * AO_PLANES)
                .saturating_add(bytes_of((nbf * np) as u64, 16 * 6))
        };
        let fixed = bytes_of((nk * nbf * nbf) as u64, 16 * 2);
        let mut chunk_points = MAX_CHUNK;
        while chunk_points > MIN_CHUNK && per_point(chunk_points) > CHUNK_TARGET_BYTES {
            chunk_points /= 2;
        }
        let per_chunk = per_point(chunk_points).saturating_add(fixed);
        led.check(
            &format!(
                "k-point XC stress AO moment chunk ({chunk_points} points, nbf = {nbf}, N_k = {nk})"
            ),
            per_chunk,
        )?;
        let par = rayon::current_num_threads()
            .max(1)
            .min(self.max_live_chunks.unwrap_or(usize::MAX).max(1))
            .min((led.remaining() / per_chunk.max(1)).max(1));
        led.reserve(
            &format!("k-point XC stress chunks ({par} x {chunk_points} points)"),
            per_chunk.saturating_mul(par),
        )?;
        let chunks = LatticeAoHess::chunks(points, chunk_points);
        let inv_nk = 1.0 / nk as f64;
        let use_gga = self.gga && mutation != Some(KXcStressMutation::NoGga);
        let alt_phase = match mutation {
            Some(KXcStressMutation::NoPhase) => Some(BlochPhase::None),
            Some(KXcStressMutation::ConjPhase) => Some(BlochPhase::Conj),
            _ => None,
        };
        let pref = if mutation == Some(KXcStressMutation::WrongFactor) {
            1.0
        } else {
            2.0
        };
        let kscale = if mutation == Some(KXcStressMutation::NoInvNk) {
            1.0
        } else {
            inv_nk
        };

        let run = |idx: &Vec<usize>| -> Result<Chunk, FerricError> {
            let pts: Vec<GridPoint> = idx.iter().map(|&i| points[i]).collect();
            let xyz: Vec<[f64; 3]> = pts.iter().map(|g| g.xyz).collect();
            let anchors: Vec<[f64; 3]> = match mutation {
                Some(KXcStressMutation::NoPointMotion) => vec![[0.0; 3]; idx.len()],
                Some(KXcStressMutation::PointSelf) => xyz.clone(),
                _ => idx.iter().map(|&i| sp.anchors[i]).collect(),
            };
            let np = pts.len();
            let bl: BlochAoStrain =
                ev.eval_bloch_strain(&xyz, &anchors, &self.kpts, BlochPhase::True, self.gga)?;
            let dens = chunk_density(d, &bl.chi, &bl.dchi, self.gga);
            let kern = closed_kernel(&dens, None, &self.xc);
            let live = |g: usize| dens.rho[g] > DENSITY_FLOOR;
            let mut e = 0.0;
            let mut ne = 0.0;
            for g in 0..np {
                e += pts[g].weight * dens.rho[g] * kern.exc[g];
                ne += pts[g].weight * dens.rho[g];
            }
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
            let wts: Vec<f64> = pts.iter().map(|p| p.weight).collect();
            let v = chunk_vxc(&bl.chi, &bl.dchi, &wts, &vr, &c, self.gga);

            // --- Strain side (module doc, item 2). The mutant phases evaluate
            // their own moments; the density above keeps the true phase.
            let alt: Option<BlochAoStrain> = alt_phase
                .map(|ph| ev.eval_bloch_strain(&xyz, &anchors, &self.kpts, ph, self.gga))
                .transpose()?;
            let fb: &BlochAoStrain = alt.as_ref().unwrap_or(&bl);
            let mut ao = ZERO3;
            if mutation != Some(KXcStressMutation::NoAo) {
                for k in 0..nk {
                    let p = d[k].dot(&conj(&fb.chi[k]));
                    let md: Option<[Array2<C64>; 3]> = use_gga.then(|| {
                        std::array::from_fn(|i| {
                            d[k].dot(&conj(&fb.dchi[k].index_axis(Axis(0), i).to_owned()))
                        })
                    });
                    for g in 0..np {
                        let wg = pref * pts[g].weight * kscale;
                        // q_μ = v_ρ p_μ + Σ_i c_i md^i_μ  (contracts with m1).
                        let q: Vec<C64> = (0..nbf)
                            .map(|mu| {
                                let mut t = p[(mu, g)] * vr[g];
                                if let Some(md) = md.as_ref() {
                                    if mutation != Some(KXcStressMutation::NoStrainMoment) {
                                        for i in 0..3 {
                                            t += md[i][(mu, g)] * c[(i, g)];
                                        }
                                    }
                                }
                                t
                            })
                            .collect();
                        for a in 0..3 {
                            for b in 0..3 {
                                let mut t = C64::new(0.0, 0.0);
                                for mu in 0..nbf {
                                    t += fb.m1[k][(a, b, mu, g)] * q[mu];
                                }
                                if use_gga && mutation != Some(KXcStressMutation::NoHessian) {
                                    for i in 0..3 {
                                        let mut h = C64::new(0.0, 0.0);
                                        for mu in 0..nbf {
                                            h += fb.m2[k][(3 * i + a, b, mu, g)] * p[(mu, g)];
                                        }
                                        t += h * c[(i, g)];
                                    }
                                }
                                ao[a][b] += wg * t.re;
                            }
                        }
                    }
                }
            }
            let mut weight = ZERO3;
            for (g, &gi) in idx.iter().enumerate() {
                let eps = dens.rho[g] * kern.exc[g];
                add_into(&mut weight, &sp.dweight[gi], eps);
            }
            Ok(Chunk {
                e,
                ne,
                v,
                ao,
                weight,
            })
        };

        // Concurrency-sized batches reduced in chunk order: memory is bounded by
        // `par` chunks and the sums do not depend on the thread count.
        let mut e_xc = 0.0;
        let mut electrons = 0.0;
        let mut v: Vec<Array2<C64>> = (0..nk).map(|_| Array2::zeros((nbf, nbf))).collect();
        let mut ao = ZERO3;
        let mut weight = ZERO3;
        for batch in chunks.chunks(par) {
            let outs: Vec<Result<Chunk, FerricError>> = batch.par_iter().map(&run).collect();
            for out in outs {
                let c = out?;
                e_xc += c.e;
                electrons += c.ne;
                for (acc, x) in v.iter_mut().zip(&c.v) {
                    *acc += x;
                }
                add_into(&mut ao, &c.ao, 1.0);
                add_into(&mut weight, &c.weight, 1.0);
            }
        }
        if !e_xc.is_finite()
            || v.iter().any(|m| m.iter().any(|z| !z.re.is_finite()))
            || ao
                .iter()
                .chain(weight.iter())
                .flatten()
                .any(|x| !x.is_finite())
        {
            return Err(FerricError::General(
                "k-point XC stress: non-finite E_xc/V_xc/stress".into(),
            ));
        }
        for m in v.iter_mut() {
            *m = hermitize(m);
        }
        if mutation == Some(KXcStressMutation::NoWeightDeriv) {
            weight = ZERO3;
        }
        Ok(XcStressOut {
            e_xc,
            electrons,
            v,
            ao,
            weight,
            npts: points.len(),
            chunk_points,
            live_chunks: par,
        })
    }
}

/// Refuse what the pure-functional stress does not cover; returns the
/// resolved functional.
fn resolve_pure(functional: &str) -> Result<XcDef, FerricError> {
    let (xc, exx, rsh) = resolve_periodic_functional_rsh(functional)?;
    if rsh.is_some() {
        return Err(PeriodicDftError::Unsupported {
            feature: "range-separated hybrid stress at k",
            reason: format!(
                "{functional}: the k-point XC stress covers LDA and GGA functionals with no exact \
                 exchange; the screened-exchange strain is not implemented"
            ),
        }
        .into());
    }
    if exx != 0.0 {
        return Err(PeriodicDftError::Unsupported {
            feature: "global hybrid stress at k",
            reason: format!(
                "{functional} has exact-exchange fraction {exx}; the k-point XC stress covers \
                 pure LDA/GGA functionals only (the exchange strain with a scaled K and its \
                 Madelung strain is not validated)"
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
            feature: "meta-GGA stress at k",
            reason: format!(
                "{functional}: the tau strain (Hessian-moment terms) is not implemented"
            ),
        }
        .into());
    }
    Ok(xc)
}

/// k-point closed-shell RKS stress (module doc) of a converged
/// [`crate::kdft::solve_krks`] result for a PURE LDA/GGA `functional`.
///
/// * `hcore_cfg`, `hk`, `jk` — the config/output of the hcore and the J source
///   the SCF used (dense AFT or RS-GDF), as for [`crate::kstress::kpoint_rhf_stress`]
///   (including the SR column-rotation setting of the energy).
/// * `scf` — `KRksResult::scf`.
/// * `cfg.grid`/`cfg.ao_threshold` — the grid and AO truncation of the energy.
///
/// Refuses (typed errors) hybrids, range-separated hybrids, meta-GGA,
/// VV10/double hybrids, and an SCF whose Fock the rebuilt `h + J + V_xc` does
/// not reproduce.
#[allow(clippy::too_many_arguments)]
pub fn kpoint_rks_stress(
    cell: &Cell,
    prep: &PreparedBasis,
    mesh: &KPointMesh,
    hcore_cfg: &PeriodicHcoreConfig,
    hk: &PeriodicHcoreK,
    jk: KStressJk<'_>,
    scf: &KScfResult,
    functional: &str,
    cfg: &KRksStressConfig,
) -> Result<KRksStress, FerricError> {
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
        grid: &cfg.grid,
        ao_threshold: cfg.ao_threshold,
        max_live_chunks: cfg.max_live_chunks,
        mutation: cfg.mutation,
    };
    rks_stress_with(
        "kpoint_rks_stress",
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
/// `V_xc = 0`) on a Hartree-only SCF. The exactness anchor of the XC plumbing
/// against [`kpoint_hartree_stress`].
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn kpoint_null_xc_stress(
    cell: &Cell,
    prep: &PreparedBasis,
    mesh: &KPointMesh,
    hcore_cfg: &PeriodicHcoreConfig,
    hk: &PeriodicHcoreK,
    jk: KStressJk<'_>,
    scf: &KScfResult,
    cfg: &KRksStressConfig,
) -> Result<KRksStress, FerricError> {
    rks_stress_with(
        "kpoint_null_xc_stress",
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
fn rks_stress_with(
    who: &str,
    cell: &Cell,
    prep: &PreparedBasis,
    mesh: &KPointMesh,
    hcore_cfg: &PeriodicHcoreConfig,
    hk: &PeriodicHcoreK,
    jk: KStressJk<'_>,
    scf: &KScfResult,
    xc: &dyn KXcStressResponse,
    cfg: &KRksStressConfig,
) -> Result<KRksStress, FerricError> {
    refuse_unsupported(who, &jk)?;
    let gj: KGradJk<'_> = grad_jk(&jk);
    check_inputs(
        who,
        cell,
        prep,
        mesh,
        hcore_cfg,
        hk,
        &gj,
        &[&scf.densities[..]],
        scf.converged,
        &scf.kpts,
    )?;
    let nk = mesh.nk();
    let mut ledger = open_ledger(prep, mesh, &cfg.stress, false)?;
    let d = scf.densities.clone();
    let xo = xc.evaluate(&d, &ledger)?;
    // F = h + J + V_xc (no exchange); the stored Fock of the SCF must agree.
    let (j, _k) = jk_build(&gj, 0.0, &d)?;
    let with_vxc = cfg.mutation != Some(KXcStressMutation::PulayNoVxc);
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
        check_fock(who, "fock", &f, &scf.fock, KS_REMEDY)?
    } else {
        0.0
    };
    let mut hf = assemble(
        cell,
        prep,
        mesh,
        hcore_cfg,
        hk,
        &gj,
        KSpins::Restricted { d, f },
        0.0,
        0.0,
        &cfg.stress,
        &mut ledger,
    )?;
    hf.fock_mismatch = mismatch;
    finish(hf, xo)
}

/// Sum the parts: `dE/dε = HF + xc_ao + xc_weight`, `σ = (1/Ω) dE/dε`.
fn finish(hf: KStress, xo: XcStressOut) -> Result<KRksStress, FerricError> {
    let mut de = hf.de_deps;
    add_into(&mut de, &xo.ao, 1.0);
    add_into(&mut de, &xo.weight, 1.0);
    if de.iter().flatten().any(|v| !v.is_finite()) {
        return Err(FerricError::General(
            "k-point RKS stress: non-finite strain derivative".into(),
        ));
    }
    let vol = hf.volume;
    Ok(KRksStress {
        de_deps: de,
        sigma: scaled(&de, 1.0 / vol),
        xc_ao: xo.ao,
        xc_weight: xo.weight,
        e_xc: xo.e_xc,
        electrons_on_grid: xo.electrons,
        n_grid_points: xo.npts,
        chunk_points: xo.chunk_points,
        live_chunks: xo.live_chunks,
        volume: vol,
        hf,
    })
}

/// TEST/DIAGNOSTIC: the exchange-free HF stress on a Hartree-only SCF
/// (`F = h + J`), the independent twin of [`kpoint_null_xc_stress`] (it shares
/// `assemble` but none of the XC plumbing or the KS Fock guard).
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn kpoint_hartree_stress(
    cell: &Cell,
    prep: &PreparedBasis,
    mesh: &KPointMesh,
    hcore_cfg: &PeriodicHcoreConfig,
    hk: &PeriodicHcoreK,
    jk: KStressJk<'_>,
    scf: &KScfResult,
    cfg: &KStressConfig,
) -> Result<KStress, FerricError> {
    let who = "kpoint_hartree_stress";
    refuse_unsupported(who, &jk)?;
    let gj = grad_jk(&jk);
    check_inputs(
        who,
        cell,
        prep,
        mesh,
        hcore_cfg,
        hk,
        &gj,
        &[&scf.densities[..]],
        scf.converged,
        &scf.kpts,
    )?;
    let mut ledger = open_ledger(prep, mesh, cfg, false)?;
    let d = scf.densities.clone();
    let (j, _k) = jk_build(&gj, 0.0, &d)?;
    let f: Vec<Array2<C64>> = (0..mesh.nk())
        .map(|q| hermitize(&(&hk.h[q] + &j[q])))
        .collect();
    let mismatch = check_fock(who, "fock", &f, &scf.fock, KS_REMEDY)?;
    let mut out = assemble(
        cell,
        prep,
        mesh,
        hcore_cfg,
        hk,
        &gj,
        KSpins::Restricted { d, f },
        0.0,
        0.0,
        cfg,
        &mut ledger,
    )?;
    out.fock_mismatch = mismatch;
    Ok(out)
}
