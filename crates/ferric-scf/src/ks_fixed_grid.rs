//! Closed-shell LDA and GGA exchange-correlation on a grid FIXED IN SPACE, and
//! its first and second nuclear derivatives.
//!
//! The energy is `E_xc = Σ_g w_g f(ρ(r_g), σ(r_g))`, `f = ρ ε_xc`,
//! `σ = |∇ρ|²`, with the points `r_g` and weights `w_g` built once (at the
//! reference geometry, by `ferric_dft::grid::build_atomic_grid`) and then held
//! constant while the nuclei move. Only the atom-centred AOs depend on the
//! geometry, so every nuclear derivative is an AO derivative and no Becke-weight
//! or grid-point response appears.
//!
//! With `P = Dχ`, `Q_γ = D ∂_γχ` and `∂_αχ` the electron-coordinate derivative
//! (the nuclear derivative of an AO on atom `B` is its negative), the
//! first-order densities of the coordinate `x = (B, α)` are
//!
//! ```text
//! ρ_x       = −2 Σ_{μ∈B} ∂_αχ_μ P_μ
//! g_{γ,x}   = −2 Σ_{μ∈B} [∂_γ∂_αχ_μ P_μ + ∂_αχ_μ Q_{μγ}]               (g = ∇ρ)
//! ρ_xy      = 2 δ_BC Σ_{μ∈B} ∂_α∂_βχ_μ P_μ + 2 Σ_{μ∈B,ν∈C} D_μν ∂_αχ_μ ∂_βχ_ν
//! g_{γ,xy}  = 2 δ_BC Σ_{μ∈B} [∂_γ∂_α∂_βχ_μ P_μ + ∂_α∂_βχ_μ Q_{μγ}]
//!             + 2 Σ_{μ∈B} [∂_γ∂_αχ_μ (D ∂_βχ)_{μ,C} + ∂_αχ_μ (D ∂_γ∂_βχ)_{μ,C}]
//! ```
//!
//! and with the per-point weights `F = w f_ρ`, `G_γ = 2 w f_σ g_γ` (`G = 0` for
//! LDA):
//!
//! ```text
//! V_μν[F,G]  = Σ_g F χ_μ χ_ν + G_γ (∂_γχ_μ χ_ν + χ_μ ∂_γχ_ν)
//! dF[ρ1,g1]  = w (f_ρρ ρ1 + 2 f_ρσ g·g1)
//! dG[ρ1,g1]  = 2 w [(f_ρσ ρ1 + 2 f_σσ g·g1) g_γ + f_σ g1_γ]
//! ∂E/∂x             = Σ_g F ρ_x + G·g_x
//! ∂²E/∂x∂y|_D      = Σ_g [ρ_x dF[y] + g_x·dG[y]] + Σ_g [F ρ_xy + G·g_xy]
//! ∂V/∂x|_D          = V[dF[x], dG[x]] + (AO-derivative rows and columns of V[F,G])
//! V1[D1]            = V[dF[D1], dG[D1]],  ρ1 = Σ D1 χχ,  g1 = 2 Σ D1 ∂χ χ
//! ```
//!
//! This is the convention of PySCF's `grid_response=False` (the grid is not
//! differentiated). It is NOT the derivative of the energy the production
//! Kohn-Sham SCF defines on a grid that follows the atoms, whose gradient
//! carries the weight-response term; the two differ by the grid's
//! translation error and the weight derivatives (see `crate::hessian`).
//!
//! Implemented: closed-shell LDA, GGA and global-hybrid versions of both (the
//! hybrid exchange fraction is carried here, its Fock/Hessian terms live in
//! `crate::hessian`). The highest AO derivative is the third one, for the
//! GGA skeleton only (`ferric_integrals::ao_grid_third`).

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
use ferric_integrals::ao_grid_third::{eval_basis_third_on_points, THIRD_ORDER};
use ferric_integrals::basis_bridge::PreparedBasis;
use ndarray::{s, Array1, Array2, Array3, Array4, Axis};

/// Grid points per batch (AO planes are `nbf × BATCH`; the Hessian-level
/// planes are 13 of them, the third-derivative level 23).
const BATCH: usize = 2048;

/// Largest `χ` table (bytes) the response kernel keeps resident for the CPHF
/// matrix-vector products; a larger system is a typed error, not a silent
/// re-evaluation per iteration.
const RESPONSE_TABLE_MAX_BYTES: usize = 2 << 30;

/// Scale factors on the pieces of the Hessian, for mutation tests.
/// `Default` (all 1) is production; 0 drops a piece, 2 doubles it.
#[derive(Debug, Clone, Copy)]
pub struct XcMutants {
    /// `ρ_x dF[y] + g_x·dG[y]` in the skeleton (the kernel part).
    pub skeleton_fxc: f64,
    /// `F ρ_xy + G·g_xy` in the skeleton.
    pub skeleton_vxc: f64,
    /// `V[dF[x], dG[x]]` in the first-order Fock matrix.
    pub fock_fxc: f64,
    /// The AO-derivative rows and columns of `V[F, G]` in the first-order Fock
    /// matrix.
    pub fock_vxc: f64,
    /// Scale on every gradient-dependent functional derivative (`f_σ`,
    /// `f_ρσ`, `f_σσ`) inside the skeleton and first-order Fock matrix: 0
    /// treats a GGA as an LDA there.
    pub gradient_terms: f64,
}

impl Default for XcMutants {
    fn default() -> Self {
        XcMutants {
            skeleton_fxc: 1.0,
            skeleton_vxc: 1.0,
            fock_fxc: 1.0,
            fock_vxc: 1.0,
            gradient_terms: 1.0,
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

/// A frozen XC grid with its functional (closed-shell LDA or GGA, pure or
/// global hybrid).
pub struct FixedGridKs {
    xc: XcDef,
    /// Global-hybrid fraction of exact exchange `a` (0 for a pure functional).
    exx: f64,
    /// The functional depends on `∇ρ` (or the gradient path is forced).
    gga: bool,
    pts: Vec<[f64; 3]>,
    w: Vec<f64>,
}

/// Which AO derivative planes a batch needs.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Level {
    Value,
    Grad,
    Hess,
    Third,
}

struct Ao {
    chi: Array2<f64>,
    dchi: Option<Array3<f64>>,
    ddchi: Option<Array4<f64>>,
    /// The ten unique third derivatives, `THIRD_ORDER`.
    third: Option<Array3<f64>>,
}

impl Ao {
    fn dchi(&self) -> &Array3<f64> {
        self.dchi.as_ref().expect("gradient planes")
    }

    fn ddchi(&self) -> &Array4<f64> {
        self.ddchi.as_ref().expect("hessian planes")
    }

    /// `∂_γ∂_α∂_β χ` for every function, `(nbf, npts)`.
    fn third(&self, g: usize, a: usize, b: usize) -> ndarray::ArrayView2<'_, f64> {
        let mut c = [0usize; 3];
        for ax in [g, a, b] {
            c[ax] += 1;
        }
        let idx = THIRD_ORDER
            .iter()
            .position(|&t| t == (c[0], c[1], c[2]))
            .expect("third derivative multiset");
        self.third
            .as_ref()
            .expect("third-derivative planes")
            .index_axis(Axis(0), idx)
    }

    /// `∂_γ∂_α χ` for every function, `(nbf, npts)`.
    fn second(&self, g: usize, a: usize) -> ndarray::ArrayView2<'_, f64> {
        self.ddchi().slice(s![g, a, .., ..])
    }
}

/// libxc derivatives of `f = ρ ε_xc` summed over the component functionals.
/// `fs`, `frs`, `fss` are zero for an LDA.
struct Kernel {
    exc: Array1<f64>,
    fr: Array1<f64>,
    fs: Array1<f64>,
    frr: Array1<f64>,
    frs: Array1<f64>,
    fss: Array1<f64>,
}

impl Kernel {
    fn zeros(n: usize) -> Self {
        let z = || Array1::<f64>::zeros(n);
        Kernel {
            exc: z(),
            fr: z(),
            fs: z(),
            frr: z(),
            frs: z(),
            fss: z(),
        }
    }
}

/// The gradient-dependent part of a batch.
struct GradPart {
    /// `Q_γ = D ∂_γχ`, `(3, nbf, npts)`.
    q: Array3<f64>,
    /// `g = ∇ρ`, `(3, npts)`.
    g: Array2<f64>,
    /// `G_γ = 2 w f_σ g_γ`, `(3, npts)`.
    gw: Array2<f64>,
}

struct Batch {
    ao: Ao,
    p: Array2<f64>,
    rho: Array1<f64>,
    w: Array1<f64>,
    k: Kernel,
    /// `F = w f_ρ`.
    fw: Array1<f64>,
    grad: Option<GradPart>,
}

fn ao_error(e: impl std::fmt::Debug) -> FerricError {
    FerricError::General(format!("fixed-grid XC: AO evaluation failed: {e:?}"))
}

fn eval_ao(geo: &XcGeometry<'_>, pts: &[[f64; 3]], level: Level) -> Result<Ao, FerricError> {
    let (mol, bs) = (geo.mol, geo.bs);
    let (chi, dchi, ddchi) = match level {
        Level::Value => (
            eval_basis_on_points(mol, bs, pts).map_err(ao_error)?,
            None,
            None,
        ),
        Level::Grad => {
            let (chi, dchi) = eval_basis_and_grad_on_points(mol, bs, pts).map_err(ao_error)?;
            (chi, Some(dchi), None)
        }
        Level::Hess | Level::Third => {
            let (chi, dchi, ddchi) =
                eval_basis_grad_hess_on_points(mol, bs, pts).map_err(ao_error)?;
            (chi, Some(dchi), Some(ddchi))
        }
    };
    let third = if level == Level::Third {
        Some(eval_basis_third_on_points(mol, bs, pts).map_err(ao_error)?)
    } else {
        None
    };
    Ok(Ao {
        chi,
        dchi,
        ddchi,
        third,
    })
}

/// `row_scale ⊙ columns`: scale column `g` of `m` by `v[g]`.
fn scale_cols(m: &Array2<f64>, v: &Array1<f64>) -> Array2<f64> {
    m * &v.view().insert_axis(Axis(0))
}

/// [`scale_cols`] for views.
fn scale_view(m: ndarray::ArrayView2<'_, f64>, v: &Array1<f64>) -> Array2<f64> {
    &m * &v.view().insert_axis(Axis(0))
}

/// `V[F, G]` for the AOs of `ao` (symmetric by construction).
fn vmat(ao: &Ao, fw: &Array1<f64>, gw: Option<&Array2<f64>>) -> Array2<f64> {
    let mut v = scale_cols(&ao.chi, fw).dot(&ao.chi.t());
    if let Some(gw) = gw {
        let dchi = ao.dchi();
        for gam in 0..3 {
            let t =
                scale_view(dchi.index_axis(Axis(0), gam), &gw.row(gam).to_owned()).dot(&ao.chi.t());
            v += &t;
            v += &t.t();
        }
    }
    v
}

/// Response of the weights `(F, G)` to a first-order density `(ρ1, g1)`.
fn weights_response(
    k: &Kernel,
    w: &Array1<f64>,
    g: Option<&Array2<f64>>,
    rho1: &Array1<f64>,
    g1: Option<&Array2<f64>>,
) -> (Array1<f64>, Option<Array2<f64>>) {
    let (Some(g), Some(g1)) = (g, g1) else {
        return (&(w * &k.frr) * rho1, None);
    };
    let gg1: Array1<f64> = (g * g1).sum_axis(Axis(0));
    let df = w * &(&(&k.frr * rho1) + &(2.0 * &(&k.frs * &gg1)));
    let coef = 2.0 * &(w * &(&(&k.frs * rho1) + &(2.0 * &(&k.fss * &gg1))));
    let wfs = 2.0 * &(w * &k.fs);
    let mut dg = Array2::<f64>::zeros(g.dim());
    for gam in 0..3 {
        let row = &(&coef * &g.row(gam)) + &(&wfs * &g1.row(gam));
        dg.row_mut(gam).assign(&row);
    }
    (df, Some(dg))
}

/// Sum the rows `lo..hi` of every atom: `(natoms, ncols)` from `(nbf, ncols)`.
fn atom_row_sums(geo: &XcGeometry<'_>, m: &Array2<f64>) -> Array2<f64> {
    let mut out = Array2::<f64>::zeros((geo.natoms(), m.ncols()));
    for (a, &(lo, hi)) in geo.ranges.iter().enumerate() {
        out.row_mut(a)
            .assign(&m.slice(s![lo..hi, ..]).sum_axis(Axis(0)));
    }
    out
}

/// `Σ_g` of each row, summed per atom: `(natoms,)`.
fn atom_sums(geo: &XcGeometry<'_>, m: &Array2<f64>) -> Array1<f64> {
    let per_function = m.sum_axis(Axis(1)).insert_axis(Axis(1));
    atom_row_sums(geo, &per_function).column(0).to_owned()
}

impl FixedGridKs {
    /// Build the grid of `mol` (the reference geometry) with `cfg`. A functional
    /// outside closed-shell LDA/GGA, pure or global hybrid, is a typed error.
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
        let semilocal = |f: FunctionalFamily| {
            matches!(
                f,
                FunctionalFamily::Lda | FunctionalFamily::Gga | FunctionalFamily::HybridGga
            )
        };
        if xc.is_double_hybrid() {
            return Err(FerricError::General(format!(
                "fixed-grid XC: '{xc_name}' is a double hybrid; its post-SCF MP2 term has no \
                 fixed-grid Hessian here, so only the Kohn-Sham reference would be differentiated"
            )));
        }
        let supported = xc.funcs.iter().all(|f| semilocal(f.family()))
            && mix.omega == 0.0
            && mix.sr == mix.lr
            && xc.vv10.is_none();
        if !supported {
            return Err(FerricError::General(format!(
                "fixed-grid XC: '{xc_name}' is not a pure or global-hybrid LDA/GGA functional"
            )));
        }
        let gga = xc.funcs.iter().any(|f| f.family() != FunctionalFamily::Lda);
        Ok(FixedGridKs {
            xc,
            exx: mix.sr,
            gga,
            pts,
            w,
        })
    }

    /// Force the gradient-dependent code path on a functional without
    /// gradient dependence (`f_σ = f_ρσ = f_σσ = 0`): the exactness anchor that
    /// the GGA assembly reduces to the LDA one.
    #[doc(hidden)]
    pub fn with_forced_gradient_path(mut self) -> Self {
        self.gga = true;
        self
    }

    /// The global-hybrid fraction `a` of exact exchange (0 for a pure functional).
    pub fn exact_exchange(&self) -> f64 {
        self.exx
    }

    /// Whether the functional (or the forced path) uses `∇ρ`.
    pub fn is_gradient_dependent(&self) -> bool {
        self.gga
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

    /// libxc derivatives at `(ρ, σ)`; the second derivatives only when `second`.
    fn kernel(&self, rho: &Array1<f64>, sigma: &Array1<f64>, second: bool) -> Kernel {
        let n = rho.len();
        let r = rho.as_slice().expect("contiguous");
        let sg = sigma.as_slice().expect("contiguous");
        let mut k = Kernel::zeros(n);
        for (i, f) in self.xc.funcs.iter().enumerate() {
            let wi = self.xc.weights.as_ref().map_or(1.0, |ws| ws[i]);
            let mut e = vec![0.0; n];
            let mut v = vec![0.0; n];
            let (mut frr, mut frs, mut fss) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
            let mut vs = vec![0.0; n];
            if f.family() == FunctionalFamily::Lda {
                f.eval_lda_unpolarized(r, &mut e, &mut v);
                if second {
                    f.eval_lda_fxc_unpolarized(r, &mut frr);
                }
            } else {
                f.eval_gga_unpolarized(r, sg, &mut e, &mut v, &mut vs);
                if second {
                    f.eval_gga_fxc_unpolarized(r, sg, &mut frr, &mut frs, &mut fss);
                }
            }
            for g in 0..n {
                k.exc[g] += wi * e[g];
                k.fr[g] += wi * v[g];
                k.fs[g] += wi * vs[g];
                k.frr[g] += wi * frr[g];
                k.frs[g] += wi * frs[g];
                k.fss[g] += wi * fss[g];
            }
        }
        for (g, &rg) in r.iter().enumerate() {
            if rg < DENSITY_FLOOR {
                k.exc[g] = 0.0;
                k.fr[g] = 0.0;
                k.fs[g] = 0.0;
                k.frr[g] = 0.0;
                k.frs[g] = 0.0;
                k.fss[g] = 0.0;
            }
        }
        k
    }

    /// `(Q, ∇ρ)` of a gradient-dependent batch.
    fn gradient_densities(d: &Array2<f64>, ao: &Ao, p: &Array2<f64>) -> (Array3<f64>, Array2<f64>) {
        let dchi = ao.dchi();
        let mut q = Array3::<f64>::zeros(dchi.dim());
        let mut g = Array2::<f64>::zeros((3, p.ncols()));
        for gam in 0..3 {
            let dg = dchi.index_axis(Axis(0), gam);
            q.index_axis_mut(Axis(0), gam).assign(&d.dot(&dg));
            g.row_mut(gam).assign(&(2.0 * (&dg * p).sum_axis(Axis(0))));
        }
        (q, g)
    }

    /// One batch at density `d`. `gradient_terms` scales `f_σ, f_ρσ, f_σσ`
    /// (1 in production).
    fn batch(
        &self,
        geo: &XcGeometry<'_>,
        d: &Array2<f64>,
        range: (usize, usize),
        level: Level,
        second: bool,
        gradient_terms: f64,
    ) -> Result<Batch, FerricError> {
        let (g0, g1) = range;
        let level = match (self.gga, level) {
            (true, l) => l.max(Level::Grad),
            (false, Level::Third) => Level::Hess,
            (false, l) => l,
        };
        let ao = eval_ao(geo, &self.pts[g0..g1], level)?;
        let p = d.dot(&ao.chi);
        let rho = (&ao.chi * &p).sum_axis(Axis(0));
        let gdens = self.gga.then(|| Self::gradient_densities(d, &ao, &p));
        let sigma = gdens.as_ref().map_or_else(
            || Array1::zeros(g1 - g0),
            |(_, g)| (g * g).sum_axis(Axis(0)),
        );
        let mut k = self.kernel(&rho, &sigma, second);
        k.fs *= gradient_terms;
        k.frs *= gradient_terms;
        k.fss *= gradient_terms;
        let w = Array1::from(self.w[g0..g1].to_vec());
        let fw = &w * &k.fr;
        let grad = gdens.map(|(q, g)| {
            let two_w_fs = 2.0 * &(&w * &k.fs);
            let gw = &g * &two_w_fs.insert_axis(Axis(0));
            GradPart { q, g, gw }
        });
        Ok(Batch {
            ao,
            p,
            rho,
            w,
            k,
            fw,
            grad,
        })
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
            let b = self.batch(geo, d, range, Level::Value, false, 1.0)?;
            e += (&b.w * &b.rho * &b.k.exc).sum();
            v += &vmat(&b.ao, &b.fw, b.grad.as_ref().map(|g| &g.gw));
        }
        let vt = v.t().to_owned();
        Ok((e, 0.5 * (v + vt)))
    }

    /// `∂ρ/∂R_x` for every coordinate `x = 3·atom + axis`, `(3N, npts_batch)`.
    fn rho_derivative(geo: &XcGeometry<'_>, b: &Batch) -> Array2<f64> {
        let dchi = b.ao.dchi();
        let mut rg = Array2::<f64>::zeros((3 * geo.natoms(), b.w.len()));
        for alpha in 0..3 {
            let prod = &dchi.index_axis(Axis(0), alpha) * &b.p;
            let sums = atom_row_sums(geo, &prod);
            for a in 0..geo.natoms() {
                rg.row_mut(3 * a + alpha).scaled_add(-2.0, &sums.row(a));
            }
        }
        rg
    }

    /// `∂(∇ρ)/∂R_x`, `(3N, 3, npts_batch)` (GGA batches only).
    fn grad_rho_derivative(geo: &XcGeometry<'_>, b: &Batch) -> Array3<f64> {
        let gp = b.grad.as_ref().expect("gradient part");
        let dchi = b.ao.dchi();
        let mut out = Array3::<f64>::zeros((3 * geo.natoms(), 3, b.w.len()));
        for alpha in 0..3 {
            for gam in 0..3 {
                let t = &b.ao.second(gam, alpha) * &b.p
                    + &dchi.index_axis(Axis(0), alpha) * &gp.q.index_axis(Axis(0), gam);
                let sums = atom_row_sums(geo, &t);
                for a in 0..geo.natoms() {
                    out.index_axis_mut(Axis(0), 3 * a + alpha)
                        .row_mut(gam)
                        .scaled_add(-2.0, &sums.row(a));
                }
            }
        }
        out
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
            let level = if self.gga { Level::Hess } else { Level::Grad };
            let b = self.batch(geo, d, range, level, false, 1.0)?;
            let mut g = Self::rho_derivative(geo, &b).dot(&b.fw);
            if let Some(gp) = &b.grad {
                let gx = Self::grad_rho_derivative(geo, &b);
                for gam in 0..3 {
                    g += &gx.index_axis(Axis(1), gam).dot(&gp.gw.row(gam));
                }
            }
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
            let b = self.batch(geo, d, range, Level::Third, true, m.gradient_terms)?;
            h.scaled_add(m.skeleton_fxc, &Self::kernel_part(geo, &b));
            let mut second = Array2::<f64>::zeros((n3, n3));
            Self::add_same_centre(geo, &b, &mut second);
            for (c, &(lo, hi)) in geo.ranges.iter().enumerate() {
                Self::add_cross_centre(geo, d, &b, (c, lo, hi), &mut second);
            }
            h.scaled_add(m.skeleton_vxc, &second);
        }
        Ok(h)
    }

    /// `Σ_g [ρ_x dF[y] + g_x·dG[y]]` of one batch.
    fn kernel_part(geo: &XcGeometry<'_>, b: &Batch) -> Array2<f64> {
        let n3 = 3 * geo.natoms();
        let rx = Self::rho_derivative(geo, b);
        let gx = b.grad.as_ref().map(|_| Self::grad_rho_derivative(geo, b));
        let npts = b.w.len();
        let mut df = Array2::<f64>::zeros((n3, npts));
        let mut dg = gx.as_ref().map(|_| Array3::<f64>::zeros((n3, 3, npts)));
        for x in 0..n3 {
            let g1 = gx.as_ref().map(|gx| gx.index_axis(Axis(0), x).to_owned());
            let (f, gdf) = weights_response(
                &b.k,
                &b.w,
                b.grad.as_ref().map(|g| &g.g),
                &rx.row(x).to_owned(),
                g1.as_ref(),
            );
            df.row_mut(x).assign(&f);
            if let (Some(dg), Some(gdf)) = (dg.as_mut(), gdf) {
                dg.index_axis_mut(Axis(0), x).assign(&gdf);
            }
        }
        let mut h = rx.dot(&df.t());
        if let (Some(gx), Some(dg)) = (&gx, &dg) {
            for gam in 0..3 {
                h += &gx
                    .index_axis(Axis(1), gam)
                    .dot(&dg.index_axis(Axis(1), gam).t());
            }
        }
        h
    }

    /// The `δ_BC` terms of `Σ_g [F ρ_xy + G·g_xy]`:
    /// `2 Σ_{μ∈B} [∂α∂βχ_μ (F P_μ + G·Q_μ) + (G_γ ∂γ∂α∂βχ_μ) P_μ]`.
    fn add_same_centre(geo: &XcGeometry<'_>, b: &Batch, h: &mut Array2<f64>) {
        // X = F P + Σ_γ G_γ Q_γ
        let mut x = scale_cols(&b.p, &b.fw);
        if let Some(gp) = &b.grad {
            for gam in 0..3 {
                x += &scale_view(gp.q.index_axis(Axis(0), gam), &gp.gw.row(gam).to_owned());
            }
        }
        for alpha in 0..3 {
            for beta in 0..3 {
                let mut t = &b.ao.second(alpha, beta) * &x;
                if let Some(gp) = &b.grad {
                    for gam in 0..3 {
                        let t3 =
                            scale_view(b.ao.third(gam, alpha, beta), &gp.gw.row(gam).to_owned());
                        t += &(&t3 * &b.p);
                    }
                }
                let per_atom = atom_sums(geo, &t);
                for a in 0..geo.natoms() {
                    h[(3 * a + alpha, 3 * a + beta)] += 2.0 * per_atom[a];
                }
            }
        }
    }

    /// The terms of `Σ_g [F ρ_xy + G·g_xy]` that couple atom `B` to the atom
    /// `C = (c, lo..hi)`:
    /// `2 Σ_{μ∈B} [∂αχ_μ (D ∂βχ)_{μ,C} F + G_γ (∂γ∂αχ_μ (D ∂βχ)_{μ,C} + ∂αχ_μ (D ∂γ∂βχ)_{μ,C})]`.
    fn add_cross_centre(
        geo: &XcGeometry<'_>,
        d: &Array2<f64>,
        b: &Batch,
        (c, lo, hi): (usize, usize, usize),
        h: &mut Array2<f64>,
    ) {
        let dchi = b.ao.dchi();
        // H_α = Σ_γ G_γ ∂γ∂αχ  (nbf, npts)
        let h_alpha: Option<Vec<Array2<f64>>> = b.grad.as_ref().map(|gp| {
            (0..3)
                .map(|alpha| {
                    let mut t = Array2::<f64>::zeros(b.p.dim());
                    for gam in 0..3 {
                        t += &scale_view(b.ao.second(gam, alpha), &gp.gw.row(gam).to_owned());
                    }
                    t
                })
                .collect()
        });
        let d_c = d.slice(s![.., lo..hi]);
        for beta in 0..3 {
            // pd = Σ_{ν∈C} D_μν ∂βχ_ν
            let pd = d_c.dot(&dchi.slice(s![beta, lo..hi, ..]));
            // wv = F pd + Σ_γ G_γ (D ∂γ∂βχ)_{μ,C}
            let mut wv = scale_cols(&pd, &b.fw);
            if let Some(gp) = &b.grad {
                for gam in 0..3 {
                    let pdg = d_c.dot(&b.ao.second(gam, beta).slice(s![lo..hi, ..]));
                    wv += &scale_cols(&pdg, &gp.gw.row(gam).to_owned());
                }
            }
            for alpha in 0..3 {
                let mut arr = &dchi.index_axis(Axis(0), alpha) * &wv;
                if let Some(ha) = &h_alpha {
                    arr += &(&ha[alpha] * &pd);
                }
                let per_atom = atom_sums(geo, &arr);
                for a in 0..geo.natoms() {
                    h[(3 * a + alpha, 3 * c + beta)] += 2.0 * per_atom[a];
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
            let b = self.batch(geo, d, range, Level::Hess, true, m.gradient_terms)?;
            let rx = Self::rho_derivative(geo, &b);
            let gx = b.grad.as_ref().map(|_| Self::grad_rho_derivative(geo, &b));
            for (x, vx) in v.iter_mut().enumerate() {
                let g1 = gx.as_ref().map(|gx| gx.index_axis(Axis(0), x).to_owned());
                let (df, dg) = weights_response(
                    &b.k,
                    &b.w,
                    b.grad.as_ref().map(|g| &g.g),
                    &rx.row(x).to_owned(),
                    g1.as_ref(),
                );
                *vx += &(m.fock_fxc * vmat(&b.ao, &df, dg.as_ref()));
            }
            Self::add_ao_derivative_fock(geo, &b, m.fock_vxc, &mut v);
        }
        Ok(v)
    }

    /// The rows and columns of `V[F, G]` whose AO moves with the nucleus:
    /// `−T_μ` with `T_μν = Σ_g F ∂αχ_μ χ_ν + G_γ (∂γ∂αχ_μ χ_ν + ∂αχ_μ ∂γχ_ν)`.
    fn add_ao_derivative_fock(geo: &XcGeometry<'_>, b: &Batch, scale: f64, v: &mut [Array2<f64>]) {
        let (chi, dchi) = (&b.ao.chi, b.ao.dchi());
        for alpha in 0..3 {
            let da = dchi.index_axis(Axis(0), alpha);
            let mut t = scale_view(da, &b.fw).dot(&chi.t());
            if let Some(gp) = &b.grad {
                for gam in 0..3 {
                    let gw = gp.gw.row(gam).to_owned();
                    t += &scale_view(b.ao.second(gam, alpha), &gw).dot(&chi.t());
                    t += &scale_view(da, &gw).dot(&dchi.index_axis(Axis(0), gam).t());
                }
            }
            for (mu, &atom) in geo.ao_atom.iter().enumerate() {
                let vx = &mut v[3 * atom + alpha];
                vx.row_mut(mu).scaled_add(-scale, &t.row(mu));
                vx.column_mut(mu).scaled_add(-scale, &t.row(mu));
            }
        }
    }

    /// The kernel `D1 ↦ V1[D1]` of the response, with the AOs and the
    /// functional's second derivatives at the converged density `d` kept
    /// resident. `gradient_terms` scales `f_σ, f_ρσ, f_σσ` (1 in production).
    pub fn response(
        &self,
        geo: &XcGeometry<'_>,
        d: &Array2<f64>,
        factor: f64,
        gradient_terms: f64,
    ) -> Result<XcResponse, FerricError> {
        let nbf = d.nrows();
        let npts = self.npts();
        let planes = if self.gga { 4 } else { 1 };
        let bytes = nbf
            .saturating_mul(npts)
            .saturating_mul(planes)
            .saturating_mul(std::mem::size_of::<f64>());
        if bytes > RESPONSE_TABLE_MAX_BYTES {
            return Err(FerricError::General(format!(
                "fixed-grid KS Hessian: the resident AO table ({:.1} GiB) exceeds {:.0} GiB",
                bytes as f64 / f64::powi(2.0, 30),
                RESPONSE_TABLE_MAX_BYTES as f64 / f64::powi(2.0, 30)
            )));
        }
        let mut resp = XcResponse {
            chi: Array2::<f64>::zeros((nbf, npts)),
            dchi: self.gga.then(|| Array3::<f64>::zeros((3, nbf, npts))),
            w: Array1::from(self.w.clone()),
            g: self.gga.then(|| Array2::<f64>::zeros((3, npts))),
            k: Kernel::zeros(npts),
            factor,
        };
        for (g0, g1) in self.ranges() {
            let b = self.batch(geo, d, (g0, g1), Level::Value, true, gradient_terms)?;
            resp.chi.slice_mut(s![.., g0..g1]).assign(&b.ao.chi);
            if let (Some(gp), Some(dst), Some(gd)) = (&b.grad, resp.dchi.as_mut(), resp.g.as_mut())
            {
                dst.slice_mut(s![.., .., g0..g1]).assign(b.ao.dchi());
                gd.slice_mut(s![.., g0..g1]).assign(&gp.g);
            }
            resp.k.fs.slice_mut(s![g0..g1]).assign(&b.k.fs);
            resp.k.frr.slice_mut(s![g0..g1]).assign(&b.k.frr);
            resp.k.frs.slice_mut(s![g0..g1]).assign(&b.k.frs);
            resp.k.fss.slice_mut(s![g0..g1]).assign(&b.k.fss);
        }
        Ok(resp)
    }
}

/// `V1[D1] = V[dF[ρ1, g1], dG[ρ1, g1]]` for a symmetric response density `D1`.
pub struct XcResponse {
    chi: Array2<f64>,
    dchi: Option<Array3<f64>>,
    w: Array1<f64>,
    g: Option<Array2<f64>>,
    k: Kernel,
    factor: f64,
}

impl XcResponse {
    /// The XC potential response to the density change `d1`.
    pub fn apply(&self, d1: &Array2<f64>) -> Array2<f64> {
        let pd = d1.dot(&self.chi);
        let rho1 = (&self.chi * &pd).sum_axis(Axis(0));
        let g1 = self.dchi.as_ref().map(|dchi| {
            let mut g1 = Array2::<f64>::zeros((3, self.chi.ncols()));
            for gam in 0..3 {
                g1.row_mut(gam)
                    .assign(&(2.0 * (&dchi.index_axis(Axis(0), gam) * &pd).sum_axis(Axis(0))));
            }
            g1
        });
        let (df, dg) = weights_response(&self.k, &self.w, self.g.as_ref(), &rho1, g1.as_ref());
        let mut v = scale_cols(&self.chi, &(self.factor * &df)).dot(&self.chi.t());
        if let (Some(dg), Some(dchi)) = (dg, self.dchi.as_ref()) {
            for gam in 0..3 {
                let gw = self.factor * &dg.row(gam).to_owned();
                let t = scale_view(dchi.index_axis(Axis(0), gam), &gw).dot(&self.chi.t());
                v += &t;
                v += &t.t();
            }
        }
        v
    }
}

/// [`FixedGridKs`] at one geometry as the SCF's [`XcBuilder`] (global-hybrid
/// fraction `a` of the grid's functional), for the injected SCF that the finite-difference anchors run at
/// displaced nuclei on the reference grid.
pub struct FixedGridXc<'a> {
    /// The frozen grid.
    pub grid: &'a FixedGridKs,
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

/// The nuclear gradient of the fixed-grid closed-shell KS energy with exact J
/// and `a`-scaled exact K:
/// nuclear repulsion, one-electron, Pulay and Coulomb terms plus
/// [`FixedGridKs::gradient`], `(natoms, 3)`. `rks` is the converged restricted
/// result on `grid` (see [`crate::hessian::rks_fixed_grid_hessian`]).
pub fn rks_fixed_grid_gradient(
    mol: &Molecule,
    prep: &PreparedBasis,
    op: ferric_integrals::operator::Operator,
    bounds: &crate::screening::SchwarzBounds,
    grid: &FixedGridKs,
    rks: &crate::result::ScfResult,
) -> Result<Array2<f64>, FerricError> {
    rks_fixed_grid_gradient_df(mol, prep, op, bounds, grid, rks, None)
}

/// [`rks_fixed_grid_gradient`] for an SCF whose Coulomb matrix is the RI-J
/// (`DfJ`) one over the auxiliary basis `dfbs` (at THIS geometry): the Coulomb
/// term is the fitted `dE_J^DF/dR` ([`crate::df_gradient::df_jk_gradient`]),
/// the exchange stays the exact `a`-scaled four-centre one. `None` is exactly
/// [`rks_fixed_grid_gradient`].
pub fn rks_fixed_grid_gradient_df(
    mol: &Molecule,
    prep: &PreparedBasis,
    op: ferric_integrals::operator::Operator,
    bounds: &crate::screening::SchwarzBounds,
    grid: &FixedGridKs,
    rks: &crate::result::ScfResult,
    dfbs: Option<&PreparedBasis>,
) -> Result<Array2<f64>, FerricError> {
    rks_fixed_grid_gradient_dfjk(mol, prep, op, bounds, grid, rks, dfbs, None)
}

/// [`rks_fixed_grid_gradient_df`] with the exchange of a hybrid fitted too:
/// `k_dfbs` is the RI-K (`DfK`) auxiliary basis at THIS geometry, and the
/// exchange term is the fitted `−(a/4) dE_K^DF/dR`
/// ([`crate::df_gradient::df_jk_gradient`], one closed-shell channel). `None`
/// for both is exactly [`rks_fixed_grid_gradient`]; a pure functional has no
/// exchange term whatever `k_dfbs` says.
#[allow(clippy::too_many_arguments)]
pub fn rks_fixed_grid_gradient_dfjk(
    mol: &Molecule,
    prep: &PreparedBasis,
    op: ferric_integrals::operator::Operator,
    bounds: &crate::screening::SchwarzBounds,
    grid: &FixedGridKs,
    rks: &crate::result::ScfResult,
    j_dfbs: Option<&PreparedBasis>,
    k_dfbs: Option<&PreparedBasis>,
) -> Result<Array2<f64>, FerricError> {
    let nocc = (mol.nelec() / 2) as usize;
    let natoms = mol.atoms.len();
    let d = rks.density_r();
    let a = grid.exact_exchange();
    let w = crate::gradient::build_energy_weighted_density(rks, nocc);
    let mut grad = crate::gradient::oneelectron_gradient(mol, prep, d, &w, None)?;
    let k_dfbs = k_dfbs.filter(|_| a != 0.0);
    match (j_dfbs, k_dfbs) {
        (None, None) => {
            grad += &crate::ks_gradient::twoelectron_gradient_scaled_k(prep, op, bounds, d, a)?;
        }
        (None, Some(_)) => {
            grad += &crate::ks_gradient::twoelectron_gradient_scaled_k(prep, op, bounds, d, 0.0)?;
        }
        (Some(_), None) => {
            grad += &crate::ks_gradient::twoelectron_k_gradient(prep, op, bounds, d, a)?;
        }
        (Some(_), Some(_)) => {}
    }
    if let Some(aux) = j_dfbs {
        grad +=
            &crate::df_gradient::df_jk_gradient(natoms, prep, aux, op, Some(d), &[], usize::MAX)?;
    }
    if let Some(aux) = k_dfbs {
        let channel = crate::df_gradient::DfKChannel {
            density: d,
            alpha: -0.25 * a,
        };
        grad += &crate::df_gradient::df_jk_gradient(
            natoms,
            prep,
            aux,
            op,
            None,
            &[channel],
            usize::MAX,
        )?;
    }
    grad += &grid.gradient(&XcGeometry::new(mol, prep)?, d)?;
    Ok(grad)
}

/// Converge the closed-shell LDA/GGA SCF of `mol` with the XC on the frozen `grid`
/// and exact four-centre J/K (the SCF at a displaced geometry whose energy
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
    grid: &FixedGridKs,
    config: &crate::rhf::RhfConfig,
) -> Result<crate::result::ScfResult, FerricError> {
    solve_rks_fixed_grid_df(ctx, mol, prep, op, bounds, grid, None, config)
}

/// [`solve_rks_fixed_grid`] with the Coulomb matrix fitted over `dfbs` (the
/// SCF's own `DfJ`; the exchange of a hybrid stays exact four-centre). `None`
/// is exactly [`solve_rks_fixed_grid`]. The result carries no `df_jk` route
/// (injected path); the Hessian takes the auxiliary basis from its caller.
#[allow(clippy::too_many_arguments)]
pub fn solve_rks_fixed_grid_df(
    ctx: &ferric_core::parallel::ParallelContext,
    mol: &Molecule,
    prep: &PreparedBasis,
    op: ferric_integrals::operator::Operator,
    bounds: &crate::screening::SchwarzBounds,
    grid: &FixedGridKs,
    dfbs: Option<&PreparedBasis>,
    config: &crate::rhf::RhfConfig,
) -> Result<crate::result::ScfResult, FerricError> {
    solve_rks_fixed_grid_dfjk(ctx, mol, prep, op, bounds, grid, dfbs, None, config)
}

/// [`solve_rks_fixed_grid_df`] with the exchange fitted too: the SCF's K is
/// `DfK` over `k_dfbs` (scaled by the hybrid fraction like the exact one).
/// `None` for `k_dfbs` is exactly [`solve_rks_fixed_grid_df`]. The result
/// carries no `df_jk` route (injected path); the Hessian takes the auxiliary
/// bases from its caller.
#[allow(clippy::too_many_arguments)]
pub fn solve_rks_fixed_grid_dfjk(
    ctx: &ferric_core::parallel::ParallelContext,
    mol: &Molecule,
    prep: &PreparedBasis,
    op: ferric_integrals::operator::Operator,
    bounds: &crate::screening::SchwarzBounds,
    grid: &FixedGridKs,
    j_dfbs: Option<&PreparedBasis>,
    k_dfbs: Option<&PreparedBasis>,
    config: &crate::rhf::RhfConfig,
) -> Result<crate::result::ScfResult, FerricError> {
    use crate::fock::{JBuilder, KBuilder};
    use crate::rhf::{resolve_three_index_budget, solve_rhf_injected, PeriodicInjection};
    let budget = resolve_three_index_budget(config.three_index_budget_bytes);
    let geo = XcGeometry::new(mol, prep)?;
    let j: Box<dyn JBuilder + '_> = match j_dfbs {
        Some(aux) => Box::new(crate::df_j::DfJ::new(op, prep, aux, budget)?),
        None => Box::new(crate::direct_j::DirectJ::new(
            ctx,
            prep,
            bounds,
            config.integral_thresh,
            budget,
        )),
    };
    let k: Box<dyn KBuilder + '_> = match k_dfbs {
        Some(aux) => Box::new(crate::df_k::DfK::new(op, prep, aux, budget)?),
        None => Box::new(crate::direct_k::DirectK::new(
            ctx,
            prep,
            bounds,
            config.integral_thresh,
            budget,
        )),
    };
    let inj = PeriodicInjection {
        s: ferric_integrals::oneelectron::overlap(prep),
        h: ferric_integrals::oneelectron::hcore_ecp_with_external(
            prep,
            mol,
            prep.basis_set(),
            None,
        )?,
        vnn: mol.nuclear_repulsion(),
        j,
        k,
        xc: Some(Box::new(FixedGridXc { grid, geo: &geo })),
    };
    solve_rhf_injected(ctx, mol, prep, op, bounds, config, inj)
}
