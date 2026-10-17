//! Closed-shell LDA exchange-correlation on a grid FIXED IN SPACE, and its
//! first and second nuclear derivatives.
//!
//! The energy is `E_xc = Σ_g w_g f(ρ(r_g))`, `f = ρ ε_xc`, with the points
//! `r_g` and weights `w_g` built once (at the reference geometry, by
//! `ferric_dft::grid::build_atomic_grid`) and then held constant while the nuclei move. Only
//! the atom-centred AOs depend on the geometry, so every nuclear derivative is
//! an AO derivative and no Becke-weight or grid-point response appears:
//!
//! ```text
//! ρ_{Bα}      = −2 Σ_{μ∈B} ∂_αχ_μ (Dχ)_μ
//! ρ_{Bα,Cβ}   = 2 δ_BC Σ_{μ∈B} ∂_α∂_βχ_μ (Dχ)_μ + 2 Σ_{μ∈B,ν∈C} D_μν ∂_αχ_μ ∂_βχ_ν
//! ∂E/∂R_{Bα}            = Σ_g w f' ρ_{Bα}
//! ∂²E/∂R_{Bα}∂R_{Cβ}|_D = Σ_g w [ f'' ρ_{Bα} ρ_{Cβ} + f' ρ_{Bα,Cβ} ]
//! ∂V_μν/∂R_{Bα}|_D     = Σ_g w [ f'' ρ_{Bα} χ_μ χ_ν − f' (δ_{μ∈B} ∂_αχ_μ χ_ν + χ_μ δ_{ν∈B} ∂_αχ_ν) ]
//! V1_μν[D1]             = Σ_g w f'' (Σ_κλ D1_κλ χ_κ χ_λ) χ_μ χ_ν
//! ```
//!
//! This is the convention of PySCF's `grid_response=False` (the grid is not
//! differentiated). It is NOT the derivative of the energy the production
//! Kohn-Sham SCF defines on a grid that follows the atoms, whose gradient
//! carries the weight-response term; the two differ by the grid's
//! translation error and the weight derivatives (see `crate::hessian`).
//!
//! Only the closed-shell LDA case (pure, or a global hybrid whose exact-exchange
//! fraction `a` enters the Fock matrix as `−(a/2)K`) is implemented: libxc's
//! `v2rho2` is the whole kernel, and the highest AO derivative is the second one.

use crate::rhf::XcBuilder;
use ferric_core::basis::BasisSet;
use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_dft::grid::{build_atomic_grid, AtomicGridConfig};
use ferric_dft::libxc::{k_mix_from_xc_def, xc_def_from_name, FunctionalFamily, XcDef};
use ferric_dft::vxc::DENSITY_FLOOR;
use ferric_integrals::ao_grid::{
    eval_basis_and_grad_on_points, eval_basis_grad_hess_on_points, eval_basis_on_points,
};
use ferric_integrals::basis_bridge::PreparedBasis;
use ndarray::{s, Array1, Array2, Array3, Array4, Axis};

/// Grid points per batch (AO planes are `nbf × BATCH`; the Hessian-level
/// planes are 13 of them).
const BATCH: usize = 2048;

/// Largest `χ` table (bytes) the response kernel keeps resident for the CPHF
/// matrix-vector products; a larger system is a typed error, not a silent
/// re-evaluation per iteration.
const RESPONSE_TABLE_MAX_BYTES: usize = 2 << 30;

/// Scale factors on the four XC pieces of the Hessian, for mutation tests.
/// `Default` (all 1) is production; 0 drops a piece, 2 doubles it.
#[derive(Debug, Clone, Copy)]
pub struct XcMutants {
    /// `f'' ρ_{Bα} ρ_{Cβ}` in the skeleton.
    pub skeleton_fxc: f64,
    /// `f' ρ_{Bα,Cβ}` in the skeleton.
    pub skeleton_vxc: f64,
    /// `f'' ρ_{Bα} χ χ` in the first-order Fock matrix.
    pub fock_fxc: f64,
    /// `f' ∂(χχ)` in the first-order Fock matrix.
    pub fock_vxc: f64,
}

impl Default for XcMutants {
    fn default() -> Self {
        XcMutants {
            skeleton_fxc: 1.0,
            skeleton_vxc: 1.0,
            fock_fxc: 1.0,
            fock_vxc: 1.0,
        }
    }
}

/// The nuclei, basis and AO-to-atom map a derivative is taken at.
pub struct XcGeometry<'a> {
    mol: &'a Molecule,
    bs: &'a BasisSet,
    ao_atom: Vec<usize>,
    ranges: Vec<(usize, usize)>,
}

impl<'a> XcGeometry<'a> {
    /// The AOs of each atom must be contiguous (they are, shell by shell).
    pub fn new(mol: &'a Molecule, prep: &'a PreparedBasis) -> Result<Self, FerricError> {
        let offsets = prep.shell_offsets();
        let mut ao_atom = Vec::with_capacity(prep.nbasis());
        for (sh, &atom) in prep.shell_to_atom().iter().enumerate() {
            ao_atom.resize(ao_atom.len() + offsets[sh + 1] - offsets[sh], atom);
        }
        let mut ranges = vec![(usize::MAX, 0usize); mol.atoms.len()];
        for (mu, &a) in ao_atom.iter().enumerate() {
            let r = &mut ranges[a];
            if r.0 == usize::MAX {
                r.0 = mu;
            } else if r.1 != mu {
                return Err(FerricError::General(
                    "fixed-grid XC derivatives need the AOs of each atom to be contiguous".into(),
                ));
            }
            r.1 = mu + 1;
        }
        Ok(XcGeometry {
            mol,
            bs: prep.basis_set(),
            ao_atom,
            ranges,
        })
    }

    fn natoms(&self) -> usize {
        self.mol.atoms.len()
    }
}

/// A frozen LDA grid with its functional.
pub struct FixedGridLda {
    xc: XcDef,
    /// Global-hybrid fraction of exact exchange `a` (0 for a pure functional).
    exx: f64,
    pts: Vec<[f64; 3]>,
    w: Vec<f64>,
}

/// Which AO derivative planes a batch needs.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Level {
    Value,
    Grad,
    Hess,
}

struct Ao {
    chi: Array2<f64>,
    dchi: Option<Array3<f64>>,
    ddchi: Option<Array4<f64>>,
}

struct Kernel {
    exc: Array1<f64>,
    v1: Array1<f64>,
    v2: Array1<f64>,
}

struct Batch {
    ao: Ao,
    p: Array2<f64>,
    rho: Array1<f64>,
    w: Array1<f64>,
    k: Kernel,
}

fn ao_error(e: impl std::fmt::Debug) -> FerricError {
    FerricError::General(format!("fixed-grid XC: AO evaluation failed: {e:?}"))
}

fn eval_ao(geo: &XcGeometry<'_>, pts: &[[f64; 3]], level: Level) -> Result<Ao, FerricError> {
    match level {
        Level::Value => Ok(Ao {
            chi: eval_basis_on_points(geo.mol, geo.bs, pts).map_err(ao_error)?,
            dchi: None,
            ddchi: None,
        }),
        Level::Grad => {
            let (chi, dchi) =
                eval_basis_and_grad_on_points(geo.mol, geo.bs, pts).map_err(ao_error)?;
            Ok(Ao {
                chi,
                dchi: Some(dchi),
                ddchi: None,
            })
        }
        Level::Hess => {
            let (chi, dchi, ddchi) =
                eval_basis_grad_hess_on_points(geo.mol, geo.bs, pts).map_err(ao_error)?;
            Ok(Ao {
                chi,
                dchi: Some(dchi),
                ddchi: Some(ddchi),
            })
        }
    }
}

/// `row_scale ⊙ columns`: scale column `g` of `m` by `v[g]`.
fn scale_cols(m: &Array2<f64>, v: &Array1<f64>) -> Array2<f64> {
    m * &v.view().insert_axis(Axis(0))
}

impl FixedGridLda {
    /// Build the grid of `mol` (the reference geometry) with `cfg`. A
    /// functional that is not pure or global-hybrid LDA is a typed error.
    pub fn new(mol: &Molecule, xc_name: &str, cfg: &AtomicGridConfig) -> Result<Self, FerricError> {
        if cfg.prune.is_some() {
            return Err(FerricError::General(
                "fixed-grid XC: an angular-pruned grid (AtomicGridConfig::prune) is not \
                 supported; the production SCF builds the pruned grid but this unit freezes \
                 the flat grid, so the two would silently differ (use prune = None)"
                    .into(),
            ));
        }
        let grid = build_atomic_grid(mol, cfg);
        Self::from_points(
            xc_name,
            grid.iter().map(|g| g.xyz).collect(),
            grid.iter().map(|g| g.weight).collect(),
        )
    }

    /// The grid points (Bohr) and weights.
    pub fn points_and_weights(&self) -> (&[[f64; 3]], &[f64]) {
        (&self.pts, &self.w)
    }

    /// A frozen grid from explicit points (Bohr) and weights, e.g. another
    /// code's grid for a like-for-like comparison.
    pub fn from_points(
        xc_name: &str,
        pts: Vec<[f64; 3]>,
        w: Vec<f64>,
    ) -> Result<Self, FerricError> {
        if pts.len() != w.len() {
            return Err(FerricError::General(format!(
                "fixed-grid XC: {} points but {} weights",
                pts.len(),
                w.len()
            )));
        }
        let xc = xc_def_from_name(xc_name)
            .map_err(|e| FerricError::General(format!("fixed-grid XC: libxc: {e:?}")))?;
        let mix = k_mix_from_xc_def(&xc);
        let global_lda = xc.funcs.iter().all(|f| f.family() == FunctionalFamily::Lda)
            && mix.omega == 0.0
            && mix.sr == mix.lr
            && xc.vv10.is_none();
        if !global_lda {
            return Err(FerricError::General(format!(
                "fixed-grid XC: '{xc_name}' is not a pure or global-hybrid LDA functional"
            )));
        }
        Ok(FixedGridLda {
            xc,
            exx: mix.sr,
            pts,
            w,
        })
    }

    /// The global-hybrid fraction `a` of exact exchange (0 for a pure functional).
    pub fn exact_exchange(&self) -> f64 {
        self.exx
    }

    /// Number of grid points.
    pub fn npts(&self) -> usize {
        self.pts.len()
    }

    fn ranges(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        (0..self.pts.len())
            .step_by(BATCH)
            .map(|g0| (g0, (g0 + BATCH).min(self.pts.len())))
    }

    fn kernel(&self, rho: &Array1<f64>, second: bool) -> Kernel {
        let n = rho.len();
        let r = rho.as_slice().expect("contiguous");
        let (mut exc, mut v1, mut v2) = (Array1::zeros(n), Array1::zeros(n), Array1::zeros(n));
        for (i, f) in self.xc.funcs.iter().enumerate() {
            let wi = self.xc.weights.as_ref().map_or(1.0, |ws| ws[i]);
            let (mut e, mut v) = (vec![0.0; n], vec![0.0; n]);
            f.eval_lda_unpolarized(r, &mut e, &mut v);
            exc.iter_mut().zip(&e).for_each(|(t, x)| *t += wi * x);
            v1.iter_mut().zip(&v).for_each(|(t, x)| *t += wi * x);
            if second {
                let mut f2 = vec![0.0; n];
                f.eval_lda_fxc_unpolarized(r, &mut f2);
                v2.iter_mut().zip(&f2).for_each(|(t, x)| *t += wi * x);
            }
        }
        for (g, &rg) in r.iter().enumerate() {
            if rg < DENSITY_FLOOR {
                exc[g] = 0.0;
                v1[g] = 0.0;
                v2[g] = 0.0;
            }
        }
        Kernel { exc, v1, v2 }
    }

    fn batch(
        &self,
        geo: &XcGeometry<'_>,
        d: &Array2<f64>,
        range: (usize, usize),
        level: Level,
        second: bool,
    ) -> Result<Batch, FerricError> {
        let (g0, g1) = range;
        let ao = eval_ao(geo, &self.pts[g0..g1], level)?;
        let p = d.dot(&ao.chi);
        let rho = (&ao.chi * &p).sum_axis(Axis(0));
        let k = self.kernel(&rho, second);
        let w = Array1::from(self.w[g0..g1].to_vec());
        Ok(Batch { ao, p, rho, w, k })
    }

    /// `(E_xc, V_xc)` at the density `d` for the AOs of `geo`.
    pub fn energy_vxc(
        &self,
        geo: &XcGeometry<'_>,
        d: &Array2<f64>,
    ) -> Result<(f64, Array2<f64>), FerricError> {
        let mut e = 0.0;
        let mut v = Array2::<f64>::zeros(d.dim());
        for range in self.ranges() {
            let b = self.batch(geo, d, range, Level::Value, false)?;
            e += (&b.w * &b.rho * &b.k.exc).sum();
            let scaled = scale_cols(&b.ao.chi, &(&b.w * &b.k.v1));
            v += &scaled.dot(&b.ao.chi.t());
        }
        let vt = v.t().to_owned();
        Ok((e, 0.5 * (v + vt)))
    }

    /// `∂ρ/∂R_{Bα}` for every coordinate, `(3N, npts_batch)`.
    fn rho_derivative(geo: &XcGeometry<'_>, b: &Batch) -> Array2<f64> {
        let dchi = b.ao.dchi.as_ref().expect("gradient planes");
        let mut rg = Array2::<f64>::zeros((3 * geo.natoms(), b.w.len()));
        for alpha in 0..3 {
            let prod = &dchi.index_axis(Axis(0), alpha) * &b.p;
            for (mu, &a) in geo.ao_atom.iter().enumerate() {
                rg.row_mut(3 * a + alpha).scaled_add(-2.0, &prod.row(mu));
            }
        }
        rg
    }

    /// The AO-derivative nuclear gradient of `E_xc` on the fixed grid,
    /// `(natoms, 3)`.
    pub fn gradient(
        &self,
        geo: &XcGeometry<'_>,
        d: &Array2<f64>,
    ) -> Result<Array2<f64>, FerricError> {
        let mut grad = Array2::<f64>::zeros((geo.natoms(), 3));
        for range in self.ranges() {
            let b = self.batch(geo, d, range, Level::Grad, false)?;
            let rg = Self::rho_derivative(geo, &b);
            let wf1 = &b.w * &b.k.v1;
            let g = rg.dot(&wf1);
            for x in 0..g.len() {
                grad[(x / 3, x % 3)] += g[x];
            }
        }
        Ok(grad)
    }

    /// The second nuclear derivative of `E_xc` at fixed density, `(3N, 3N)`.
    pub fn skeleton_hessian(
        &self,
        geo: &XcGeometry<'_>,
        d: &Array2<f64>,
        m: &XcMutants,
    ) -> Result<Array2<f64>, FerricError> {
        let n3 = 3 * geo.natoms();
        let mut h = Array2::<f64>::zeros((n3, n3));
        for range in self.ranges() {
            let b = self.batch(geo, d, range, Level::Hess, true)?;
            let rg = Self::rho_derivative(geo, &b);
            let wf2 = &b.w * &b.k.v2;
            h.scaled_add(m.skeleton_fxc, &scale_cols(&rg, &wf2).dot(&rg.t()));
            let wf1 = &b.w * &b.k.v1;
            Self::add_second_derivative_of_rho(geo, d, &b, &wf1, m.skeleton_vxc, &mut h);
        }
        Ok(h)
    }

    /// `Σ_g w f' ρ_{Bα,Cβ}` added into `h`.
    fn add_second_derivative_of_rho(
        geo: &XcGeometry<'_>,
        d: &Array2<f64>,
        b: &Batch,
        wf1: &Array1<f64>,
        scale: f64,
        h: &mut Array2<f64>,
    ) {
        let dchi = b.ao.dchi.as_ref().expect("gradient planes");
        let ddchi = b.ao.ddchi.as_ref().expect("hessian planes");
        for alpha in 0..3 {
            for beta in 0..3 {
                // Same-centre term 2 Σ_{μ∈B} ∂α∂βχ_μ (Dχ)_μ.
                let prod = &ddchi.index_axis(Axis(0), alpha).index_axis(Axis(0), beta) * &b.p;
                let s_mu = prod.dot(wf1);
                for (mu, &a) in geo.ao_atom.iter().enumerate() {
                    h[(3 * a + alpha, 3 * a + beta)] += 2.0 * scale * s_mu[mu];
                }
            }
        }
        for (c, &(lo, hi)) in geo.ranges.iter().enumerate() {
            for beta in 0..3 {
                // Σ_{ν∈C} D_μν ∂βχ_ν for every μ.
                let pd = d
                    .slice(s![.., lo..hi])
                    .dot(&dchi.slice(s![beta, lo..hi, ..]));
                for alpha in 0..3 {
                    let prod: Array2<f64> = &dchi.index_axis(Axis(0), alpha) * &pd;
                    let t = prod.dot(wf1);
                    for (mu, &a) in geo.ao_atom.iter().enumerate() {
                        h[(3 * a + alpha, 3 * c + beta)] += 2.0 * scale * t[mu];
                    }
                }
            }
        }
    }

    /// `∂V_xc/∂R_x` at fixed density for every coordinate `x = 3·atom + axis`
    /// (AO basis, symmetric, `nbf × nbf` each).
    pub fn first_order_fock(
        &self,
        geo: &XcGeometry<'_>,
        d: &Array2<f64>,
        m: &XcMutants,
    ) -> Result<Vec<Array2<f64>>, FerricError> {
        let nbf = d.nrows();
        let n3 = 3 * geo.natoms();
        let mut v = vec![Array2::<f64>::zeros((nbf, nbf)); n3];
        for range in self.ranges() {
            let b = self.batch(geo, d, range, Level::Grad, true)?;
            let rg = Self::rho_derivative(geo, &b);
            let wf2 = &b.w * &b.k.v2;
            let wf1 = &b.w * &b.k.v1;
            let chi = &b.ao.chi;
            for (x, vx) in v.iter_mut().enumerate() {
                let coef = &wf2 * &rg.row(x);
                *vx += &(m.fock_fxc * scale_cols(chi, &coef).dot(&chi.t()));
            }
            let dchi = b.ao.dchi.as_ref().expect("gradient planes");
            for alpha in 0..3 {
                let a_mat =
                    scale_cols(&dchi.index_axis(Axis(0), alpha).to_owned(), &wf1).dot(&chi.t());
                for (mu, &atom) in geo.ao_atom.iter().enumerate() {
                    let vx = &mut v[3 * atom + alpha];
                    vx.row_mut(mu).scaled_add(-m.fock_vxc, &a_mat.row(mu));
                    vx.column_mut(mu).scaled_add(-m.fock_vxc, &a_mat.row(mu));
                }
            }
        }
        Ok(v)
    }

    /// The kernel `D1 ↦ V1[D1]` of the response, with `χ` and `w f''` at the
    /// converged density `d` kept resident.
    pub fn response(
        &self,
        geo: &XcGeometry<'_>,
        d: &Array2<f64>,
        factor: f64,
    ) -> Result<LdaResponse, FerricError> {
        let nbf = d.nrows();
        let bytes = nbf
            .saturating_mul(self.npts())
            .saturating_mul(std::mem::size_of::<f64>());
        if bytes > RESPONSE_TABLE_MAX_BYTES {
            return Err(FerricError::General(format!(
                "fixed-grid LDA Hessian: the resident AO table ({:.1} GiB) exceeds {:.0} GiB",
                bytes as f64 / f64::powi(2.0, 30),
                RESPONSE_TABLE_MAX_BYTES as f64 / f64::powi(2.0, 30)
            )));
        }
        let mut chi = Array2::<f64>::zeros((nbf, self.npts()));
        let mut wf2 = Array1::<f64>::zeros(self.npts());
        for (g0, g1) in self.ranges() {
            let b = self.batch(geo, d, (g0, g1), Level::Value, true)?;
            chi.slice_mut(s![.., g0..g1]).assign(&b.ao.chi);
            wf2.slice_mut(s![g0..g1]).assign(&(factor * &b.w * &b.k.v2));
        }
        Ok(LdaResponse { chi, wf2 })
    }
}

/// `V1[D1] = Σ_g w f'' ρ1(g) χ_μ χ_ν` for a symmetric response density `D1`.
pub struct LdaResponse {
    chi: Array2<f64>,
    wf2: Array1<f64>,
}

impl LdaResponse {
    /// The XC potential response to the density change `d1`.
    pub fn apply(&self, d1: &Array2<f64>) -> Array2<f64> {
        let rho1 = (&self.chi * &d1.dot(&self.chi)).sum_axis(Axis(0));
        scale_cols(&self.chi, &(&self.wf2 * &rho1)).dot(&self.chi.t())
    }
}

/// [`FixedGridLda`] at one geometry as the SCF's [`XcBuilder`] (global-hybrid
/// fraction `a` of the grid's functional), for the injected SCF that the finite-difference anchors run at
/// displaced nuclei on the reference grid.
pub struct FixedGridXc<'a> {
    /// The frozen grid.
    pub grid: &'a FixedGridLda,
    /// The geometry and basis the AOs are evaluated at.
    pub geo: &'a XcGeometry<'a>,
}

impl XcBuilder for FixedGridXc<'_> {
    fn build(&mut self, d: &Array2<f64>) -> Result<(f64, Array2<f64>), FerricError> {
        self.grid.energy_vxc(self.geo, d)
    }

    fn exact_exchange_fraction(&self) -> f64 {
        self.grid.exact_exchange()
    }
}

/// The nuclear gradient of the fixed-grid closed-shell LDA energy with exact J:
/// nuclear repulsion, one-electron, Pulay and Coulomb terms plus
/// [`FixedGridLda::gradient`], `(natoms, 3)`. `rks` is the converged restricted
/// result on `grid` (see [`crate::hessian::rks_lda_fixed_grid_hessian`]).
pub fn rks_lda_fixed_grid_gradient(
    mol: &Molecule,
    prep: &PreparedBasis,
    op: ferric_integrals::operator::Operator,
    bounds: &crate::screening::SchwarzBounds,
    grid: &FixedGridLda,
    rks: &crate::result::ScfResult,
) -> Result<Array2<f64>, FerricError> {
    let nocc = (mol.nelec() / 2) as usize;
    let d = rks.density_r();
    let w = crate::gradient::build_energy_weighted_density(rks, nocc);
    let mut grad = crate::gradient::oneelectron_gradient(mol, prep, d, &w, None)?;
    grad += &crate::ks_gradient::twoelectron_gradient_scaled_k(
        prep,
        op,
        bounds,
        d,
        grid.exact_exchange(),
    )?;
    grad += &grid.gradient(&XcGeometry::new(mol, prep)?, d)?;
    Ok(grad)
}

/// Converge the closed-shell LDA SCF of `mol` with the XC on the frozen `grid`
/// and exact four-centre J (the SCF at a displaced geometry whose energy
/// surface the fixed-grid gradient and Hessian differentiate).
///
/// `config` is the injected-path configuration (see
/// [`crate::rhf::validate_injected`]: no `xc`, no density fitting, no SAD
/// guess); the functional comes from `grid`.
pub fn solve_rks_fixed_grid(
    ctx: &ferric_core::parallel::ParallelContext,
    mol: &Molecule,
    prep: &PreparedBasis,
    op: ferric_integrals::operator::Operator,
    bounds: &crate::screening::SchwarzBounds,
    grid: &FixedGridLda,
    config: &crate::rhf::RhfConfig,
) -> Result<crate::result::ScfResult, FerricError> {
    use crate::rhf::{resolve_three_index_budget, solve_rhf_injected, PeriodicInjection};
    let budget = resolve_three_index_budget(config.three_index_budget_bytes);
    let geo = XcGeometry::new(mol, prep)?;
    let inj = PeriodicInjection {
        s: ferric_integrals::oneelectron::overlap(prep),
        h: ferric_integrals::oneelectron::hcore_ecp_with_external(
            prep,
            mol,
            prep.basis_set(),
            None,
        )?,
        vnn: mol.nuclear_repulsion(),
        j: Box::new(crate::direct_j::DirectJ::new(
            ctx,
            prep,
            bounds,
            config.integral_thresh,
            budget,
        )),
        k: Box::new(crate::direct_k::DirectK::new(
            ctx,
            prep,
            bounds,
            config.integral_thresh,
            budget,
        )),
        xc: Some(Box::new(FixedGridXc { grid, geo: &geo })),
    };
    solve_rhf_injected(ctx, mol, prep, op, bounds, config, inj)
}
