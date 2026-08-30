//! Analytical nuclear gradient of the semilocal XC energy on a Becke-Lebedev grid.
//!
//! This implements the "no grid response" approximation: the Becke partition
//! and radial-grid weights are treated as nuclear-position-independent. The
//! resulting gradient has an error of ~1e-5 Ha/Bohr at typical (75, 110) grids;
//! sufficient for geometry optimization but not spectroscopic accuracy.
//!
//! Closed-shell LDA gradient per atom A:
//!
//! ```text
//!     ∂E_xc/∂R_A,axis = -2 · Σ_g w_g · v_ρ(r_g) ·
//!                       Σ_ν χ_ν(r_g) · Σ_{μ ∈ A} D_μν · ∂_axis χ_μ(r_g)
//! ```
//!
//! (The factor of 2 accounts for the symmetric μ↔ν pair: when μ ∈ A we get
//! one contribution; when ν ∈ A the symmetric one. D is symmetric, so these
//! are equal — hence the 2.)

use ferric_core::memory::plan::{Lifetime, MemoryPlan};
use ferric_core::mol::Molecule;
use ferric_integrals::blas_threads::{opt_in_blas_threads, with_blas_threads};
use crate::ao_grid::AoGridKind;
use ndarray::{Array2, Array3, ArrayView1, ArrayView2, Axis, Zip};
use rayon::prelude::*;

use crate::density_on_grid::{eval_density_closed, eval_density_uks, eval_tau_closed};
use crate::grid::{
    build_atomic_grid, build_atomic_grid_with_response, grid_response_bytes, AtomicGridConfig,
};
use crate::libxc::{xc_def_from_name, xc_def_from_name_nspin, FunctionalFamily, LibxcError, XcDef};

/// Total element-count (`nbf·npts`) below which the row/column contractions stay
/// serial — rayon spawn/join overhead dwarfs the reduction on tiny grids (the
/// free-atom SCF case). Mirrors the `PAR_MIN_PTS` guard in `vv10`/`ao_grid`.
const PAR_MIN_ELEMS: usize = 128 * 128;

/// Row-wise contraction `out[μ] = Σ_g a[μ,g] · b[μ,g]` over two `(nbf, npts)`
/// operands sharing the row index μ.
///
/// This is the diagonal of `a · bᵀ`; forming the full GEMM would compute `nbf×`
/// too much work (the off-diagonal μ≠ν terms are never used), so we fuse the
/// element-wise product and the row reduction into one contiguous pass instead.
/// The μ rows are fanned out over rayon — disjoint reads, one output slot per
/// row, each row a fixed-order left-to-right fold — so the result is
/// bit-identical to the serial reduction regardless of thread count.
#[inline]
fn row_dot(a: &ArrayView2<'_, f64>, b: &ArrayView2<'_, f64>) -> ndarray::Array1<f64> {
    debug_assert_eq!(a.dim(), b.dim());
    let (nbf, npts) = a.dim();
    // Per-row deterministic left-to-right fold: row μ of `a` · row μ of `b`.
    // Each row is a fully independent reduction, so parallel and serial paths
    // give bit-identical results.
    let reduce = |mu: usize| -> f64 {
        let ar = a.row(mu);
        let br = b.row(mu);
        Zip::from(ar).and(br).fold(0.0_f64, |acc, &x, &y| acc + x * y)
    };
    let out: Vec<f64> = if nbf.saturating_mul(npts) >= PAR_MIN_ELEMS {
        (0..nbf).into_par_iter().map(reduce).collect()
    } else {
        (0..nbf).map(reduce).collect()
    };
    ndarray::Array1::from_vec(out)
}

/// Column-wise contraction `out[g] = Σ_μ a[μ,g] · b[μ,g]` over two `(nbf, npts)`
/// operands sharing the row index μ, reducing along μ to a length-`npts` vector.
///
/// Used by the GGA grid-response path to form ∂²ρ on the grid. Fans out over
/// grid-point columns; each column's μ-reduction is an independent, fixed-order
/// left-to-right fold, so the output is bit-identical across thread counts.
fn col_dot(a: &ArrayView2<'_, f64>, b: &ArrayView2<'_, f64>) -> ndarray::Array1<f64> {
    debug_assert_eq!(a.dim(), b.dim());
    let (nbf, npts) = a.dim();
    let mut out = ndarray::Array1::<f64>::zeros(npts);
    let reduce = |g: usize| -> f64 {
        let mut acc = 0.0_f64;
        for mu in 0..nbf {
            acc += a[(mu, g)] * b[(mu, g)];
        }
        acc
    };
    if nbf.saturating_mul(npts) >= PAR_MIN_ELEMS {
        out.as_slice_mut()
            .expect("contiguous")
            .par_iter_mut()
            .enumerate()
            .for_each(|(g, o)| *o = reduce(g));
    } else {
        for g in 0..npts {
            out[g] = reduce(g);
        }
    }
    out
}

/// Scatter per-basis-function, per-axis partial sums into the `(natoms, 3)`
/// gradient: `grad[atom(μ), axis] -= 2 · partial[axis][μ]`.
///
/// `partial[axis]` has length nbf. The scatter itself is a short serial pass
/// (nbf ≪ the npts contraction it follows), and keeping it serial preserves a
/// fixed summation order over μ so the result is thread-count independent.
fn scatter_partials(
    partials: &[ndarray::Array1<f64>; 3],
    bf_to_atom_map: &[usize],
    grad: &mut Array2<f64>,
) {
    let nbf = partials[0].len();
    for (axis, part) in partials.iter().enumerate() {
        for mu in 0..nbf {
            let atom = bf_to_atom_map[mu];
            grad[(atom, axis)] -= 2.0 * part[mu];
        }
    }
}

/// `out[(μ, g)] = a[(μ, g)] · s[g]` — fused row-major column scaling into a
/// fresh array. (The gradient sites' analogue of `vxc::scale_columns_into`, but
/// allocating since the operands are reused unscaled elsewhere.)
#[inline]
fn scale_cols(a: &ArrayView2<'_, f64>, s: &ArrayView1<'_, f64>) -> Array2<f64> {
    let mut out = a.to_owned();
    Zip::from(out.rows_mut()).for_each(|mut row| {
        Zip::from(&mut row).and(s).for_each(|v, &sg| *v *= sg);
    });
    out
}

/// Assemble the per-axis, per-μ AO-derivative partial sums for a GGA-form
/// gradient (closed shell, or one UKS spin channel).
///
/// Computes, for each Cartesian `axis` and basis function μ:
/// ```text
///   partial[axis][μ] = Σ_g [ t_rho[g] · m[μ,g] · ∂_axis χ[μ,g]
///     + Σ_b c[b,g] · ( ∂_axis χ[μ,g] · mdchi[b,μ,g]
///                     + ∂²_{axis,b} χ[μ,g] · m[μ,g] ) ]
/// ```
/// where `c[b,g]` is the caller-supplied per-direction GGA weight column
/// (closed shell: `t_sig[g]·∇ρ_b[g]`; UKS: the mixed `G^σ_b(g)`).
///
/// Each term is a row-wise contraction (see [`row_dot`]) of two pre-scaled
/// `(nbf, npts)` operands; the μ-row reductions fan out over rayon. The caller
/// scatters the result via [`scatter_partials`].
fn gga_ao_partials(
    m: &Array2<f64>,
    mdchi: &Array3<f64>,
    dchi: &Array3<f64>,
    ddchi: &ndarray::Array4<f64>,
    c: &Array2<f64>, // (3, npts) per-direction GGA weight column
    t_rho: &[f64],
) -> [ndarray::Array1<f64>; 3] {
    let t_rho_v = ArrayView1::from(t_rho);
    // LDA-like operand: mt[μ,g] = m[μ,g] · t_rho[g].
    let mt = scale_cols(&m.view(), &t_rho_v);
    std::array::from_fn(|axis| {
        let dchi_axis = dchi.index_axis(Axis(0), axis);
        // LDA piece.
        let mut partial = row_dot(&mt.view(), &dchi_axis);
        for b in 0..3 {
            let cb_v = c.index_axis(Axis(0), b);
            // term1: ∂_axis χ · (mdchi_b · c_b)
            let mdchi_b_scaled = scale_cols(&mdchi.index_axis(Axis(0), b), &cb_v);
            partial = partial + row_dot(&dchi_axis, &mdchi_b_scaled.view());
            // term2: ∂²_{axis,b} χ · (m · c_b)
            let m_scaled = scale_cols(&m.view(), &cb_v);
            let ddchi_a = ddchi.index_axis(Axis(0), axis);
            let ddchi_ab = ddchi_a.index_axis(Axis(0), b);
            partial = partial + row_dot(&ddchi_ab, &m_scaled.view());
        }
        partial
    })
}

/// Build the closed-shell GGA weight columns `c[b,g] = t_sig[g] · ∇ρ_b[g]`.
fn gga_weight_columns(t_sig: &[f64], grad_rho: &Array2<f64>) -> Array2<f64> {
    let t_sig_v = ArrayView1::from(t_sig);
    let mut c = Array2::<f64>::zeros((3, grad_rho.ncols()));
    for b in 0..3 {
        Zip::from(&mut c.index_axis_mut(Axis(0), b))
            .and(&t_sig_v)
            .and(&grad_rho.index_axis(Axis(0), b))
            .for_each(|o, &t, &gr| *o = t * gr);
    }
    c
}

/// Meta-GGA τ contribution to the per-μ, per-axis AO-derivative partial sums.
///
/// Derivation (closed shell; one spin channel behaves identically with D → D_σ
/// and τ → τ_σ). The kinetic-energy density is
///
/// ```text
///   τ(r) = ½ Σ_b Σ_{μν} D_μν ∂_b χ_μ(r) ∂_b χ_ν(r)
/// ```
///
/// Translating nucleus A moves only the AOs centred on A, and for those
/// `∂χ_μ/∂R_{A,α} = −∂_α χ_μ` (χ_μ depends on r − R_A). Differentiating τ at
/// fixed D and fixed grid therefore hits **one** of the two ∂_b χ factors per
/// term:
///
/// ```text
///   ∂τ/∂R_{A,α} = ½ Σ_b Σ_{μν} D_μν [ (−∂²_{αb} χ_μ)·∂_b χ_ν  (μ ∈ A)
///                                    + ∂_b χ_μ·(−∂²_{αb} χ_ν) (ν ∈ A) ]
/// ```
///
/// D is symmetric and the two bracket terms map onto each other under μ↔ν, so
/// they are equal and the ½ cancels:
///
/// ```text
///   ∂τ/∂R_{A,α} = − Σ_b Σ_{μ∈A, ν} D_μν ∂²_{αb} χ_μ ∂_b χ_ν
///               = − Σ_b Σ_{μ∈A} ∂²_{αb} χ_μ · (D ∂_b χ)_μ
/// ```
///
/// and the energy contribution is `Σ_g w_g v_τ(g) ∂τ/∂R_{A,α}`.
///
/// [`scatter_partials`] applies `grad[atom(μ), axis] -= 2 · partial[axis][μ]`,
/// so this routine returns **half** the μ-resolved sum:
///
/// ```text
///   partial[α][μ] = ½ Σ_g (w_g v_τ(g)) Σ_b ∂²_{αb} χ_μ(g) · (D ∂_b χ)_μ(g)
/// ```
///
/// Cross-check against the Fock-matrix analogue in `vxc.rs`
/// (`V^τ_μν = ½ Σ_g w_g v_τ Σ_b ∂_b χ_μ ∂_b χ_ν`): contracting `V^τ` with the
/// AO-derivative of D reproduces exactly the expression above, including the
/// cancelled ½ — the factor lives in the μ↔ν doubling, not in the τ definition.
fn mgga_tau_partials(
    mdchi: &Array3<f64>,          // (3, nbf, npts): (D ∂_b χ)_μ
    ddchi: &ndarray::Array4<f64>, // (3, 3, nbf, npts)
    t_tau: &[f64],                // w_g · v_τ(g)
) -> [ndarray::Array1<f64>; 3] {
    let t_tau_v = ArrayView1::from(t_tau);
    std::array::from_fn(|axis| {
        let ddchi_a = ddchi.index_axis(Axis(0), axis);
        let mut partial: Option<ndarray::Array1<f64>> = None;
        for b in 0..3 {
            let mdchi_b_scaled = scale_cols(&mdchi.index_axis(Axis(0), b), &t_tau_v);
            let ddchi_ab = ddchi_a.index_axis(Axis(0), b);
            let term = row_dot(&ddchi_ab, &mdchi_b_scaled.view());
            partial = Some(match partial {
                None => term,
                Some(p) => p + term,
            });
        }
        let mut p = partial.expect("3 axes");
        p.mapv_inplace(|v| 0.5 * v);
        p
    })
}

/// Electron-coordinate gradient of τ on the grid: `∂_α τ(r_g)` for each axis.
///
/// ```text
///   ∂_α τ = ½ Σ_b Σ_{μν} D_μν [∂²_{αb} χ_μ ∂_b χ_ν + ∂_b χ_μ ∂²_{αb} χ_ν]
///         = Σ_b Σ_μ ∂²_{αb} χ_μ · (D ∂_b χ)_μ
/// ```
///
/// (Same μ↔ν doubling as [`mgga_tau_partials`], hence no residual ½.) Needed by
/// the grid-response "home-translation" correction, which differentiates the
/// integrand with respect to the grid point itself.
fn tau_spatial_grad(
    mdchi: &Array3<f64>,
    ddchi: &ndarray::Array4<f64>,
) -> [ndarray::Array1<f64>; 3] {
    std::array::from_fn(|axis| {
        let ddchi_a = ddchi.index_axis(Axis(0), axis);
        let mut acc: Option<ndarray::Array1<f64>> = None;
        for b in 0..3 {
            let mdchi_b = mdchi.index_axis(Axis(0), b);
            let ddchi_ab = ddchi_a.index_axis(Axis(0), b);
            let term = col_dot(&ddchi_ab, &mdchi_b);
            acc = Some(match acc {
                None => term,
                Some(a) => a + term,
            });
        }
        acc.expect("3 axes")
    })
}

/// `Mdχ[b, μ, g] = Σ_ν D_μν ∂_b χ_ν(r_g)` — one GEMM per Cartesian direction.
fn build_mdchi(d: &Array2<f64>, dchi: &Array3<f64>) -> Array3<f64> {
    let (_, nbf, npts) = dchi.dim();
    let mut mdchi = Array3::<f64>::zeros((3, nbf, npts));
    for b in 0..3 {
        let slice = dchi.index_axis(Axis(0), b);
        let prod: Array2<f64> = with_blas_threads(opt_in_blas_threads(), || d.dot(&slice));
        mdchi.index_axis_mut(Axis(0), b).assign(&prod);
    }
    mdchi
}

/// Errors from the KS-DFT nuclear gradient path.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum KsGradError {
    #[error("XC family {0:?} not supported in this gradient path")]
    UnsupportedFamily(FunctionalFamily),
    #[error("libxc resolver failed: {0}")]
    Libxc(LibxcError),
    #[error("AO eval failed: {0:?}")]
    AoEval(crate::ao_grid::GtoEvalError),
    /// The gradient's declared working set does not fit the memory budget.
    /// Carries the per-term breakdown from [`MemoryPlan::check`], so the
    /// message names which tensor blew up rather than only the total.
    #[error("{0}")]
    OverBudget(ferric_core::error::FerricError),
}

impl From<ferric_core::error::FerricError> for KsGradError {
    fn from(e: ferric_core::error::FerricError) -> Self {
        Self::OverBudget(e)
    }
}

/// Declare the full working set of a KS-DFT XC nuclear-gradient path.
///
/// # Why this exists
///
/// `ao_grid::check_ao_grid_budget(ValueGradHess, …)` correctly approves the 13
/// `(nbf, npts)` planes of χ + ∇χ + ∇∇χ that
/// `eval_basis_grad_hess_on_points` allocates — and then every function in this
/// module allocates *more*, after that check has already returned `Ok`. The
/// UKS meta-GGA path peaks around 24 planes against a gate that approved 13,
/// and there was no `check_alloc` anywhere in this file to catch the
/// difference. This plan is the missing gate: it is built and checked **before
/// the AO evaluation**, and it declares everything, not just the AO tensors.
///
/// # The plane census, by allocation site
///
/// Resident for the whole call:
///
/// | planes | what | site |
/// |---|---|---|
/// | `kind.planes()` | χ (+ ∇χ, + ∇∇χ) | `ao_grid::eval_basis_grad_hess_on_points` |
/// | 1 (×2 UKS) | `m` = `D·χ` | `let m = d.dot(&chi)` in every path |
/// | 3 (×2 UKS) | `mdchi` = `D·∂χ` | `build_mdchi`, and its open-coded twins |
///
/// Largest transient stage (these do not coexist — each is dropped before the
/// next allocates):
///
/// | planes | what | site |
/// |---|---|---|
/// | 3 | `mt`, `mdchi_b_scaled`, `m_scaled` | `gga_ao_partials` |
/// | 3 | `psi[0..3]` | `eval_tau_closed`, meta-GGA only |
/// | 2 | `pa`/`pb` per direction | the UKS `mdchi_a`/`mdchi_b` build loop |
/// | 1 (×2 UKS) | `phi` = `D·χ` | `eval_density_closed`/`_uks` |
///
/// so 3 covers all of them.
///
/// # Where this is deliberately conservative
///
/// `xc_gradient_uks_from_density` builds one spin's `m`/`mdchi` at a time in
/// its AO-derivative phase (4 planes + 3 transient) and only later holds both
/// spins' (8 planes + 2 transient); its true peak is 23, not the 24 declared
/// here. One plane of slack out of 24 is worth not making the caller reason
/// about which phase it is in — but it is slack, not an unaccounted term, and
/// it is stated so nobody "fixes" it by shaving the real terms.
///
/// `natoms` is only used for the grid-response `weight1` term; pass
/// `with_grid_response = false` for the paths (VV10) that do not build it.
fn xc_gradient_plan(
    label: &'static str,
    nbf: usize,
    npts: usize,
    natoms: usize,
    kind: AoGridKind,
    is_uks: bool,
    with_grid_response: bool,
) -> MemoryPlan {
    let plane = nbf.saturating_mul(npts);
    let mut plan = MemoryPlan::resolve(None, label);

    // Already allocated by the time this runs (the grid is built first, since
    // it is what determines `npts`) — declared so the AO tensors are sized
    // against what is left, not against an empty budget.
    if with_grid_response {
        plan.reserve_sized(
            "grid + weight1 dw/dR (already resident)",
            grid_response_bytes(npts, natoms),
            1,
            Lifetime::Resident,
            1,
        );
    } else {
        plan.reserve_sized(
            "grid points (already resident)",
            npts,
            std::mem::size_of::<crate::grid::GridPoint>(),
            Lifetime::Resident,
            1,
        );
    }

    plan.reserve(
        match kind {
            AoGridKind::ValueOnly => "chi",
            AoGridKind::ValueAndGrad => "chi + dchi",
            AoGridKind::ValueGradHess => "chi + dchi + ddchi",
        },
        kind.planes().saturating_mul(plane),
        Lifetime::Resident,
    );

    // m = D·chi (1) and mdchi = D·dchi (3), per spin.
    let spins: usize = if is_uks { 2 } else { 1 };
    plan.reserve(
        if is_uks { "M = D_s·chi and Mdchi = D_s·dchi (both spins)" } else { "M = D·chi and Mdchi = D·dchi" },
        spins.saturating_mul(4).saturating_mul(plane),
        Lifetime::Resident,
    );

    // Largest transient stage — see the census above.
    plan.reserve("AO-partial scratch (scale_cols / psi)", 3usize.saturating_mul(plane), Lifetime::Transient);

    // O(npts) companion vectors: the libxc in/out buffers, the pre-scaled
    // per-point coefficients, ρ/∇ρ/σ (and τ), the GGA weight columns, and the
    // `pts`/`weights` copies of the grid. Counted the same way as `ks.rs`'s
    // `batch_point_doubles` — negligible against a plane at production `nbf`,
    // but not at small-basis/large-grid shapes, where they would otherwise be
    // the whole unaccounted difference.
    plan.reserve(
        "per-grid-point work vectors",
        (if is_uks { 48usize } else { 24 }).saturating_mul(npts),
        Lifetime::Resident,
    );

    plan
}

impl From<LibxcError> for KsGradError { fn from(e: LibxcError) -> Self { Self::Libxc(e) } }
impl From<crate::ao_grid::GtoEvalError> for KsGradError {
    fn from(e: crate::ao_grid::GtoEvalError) -> Self { Self::AoEval(e) }
}

impl From<KsGradError> for ferric_core::error::FerricError {
    fn from(e: KsGradError) -> Self { Self::General(e.to_string()) }
}

/// Compute the AO-basis index → atom index map.
///
/// `bf_to_atom[μ]` returns the index of the atom whose basis function μ
/// belongs to.
fn bf_to_atom(
    shell_to_atom: &[usize],
    shell_offsets: &[usize],
    shell_dims: &[usize],
    nbf: usize,
) -> Vec<usize> {
    let mut map = vec![0_usize; nbf];
    for (sh_idx, &atom) in shell_to_atom.iter().enumerate() {
        let off = shell_offsets[sh_idx];
        let dim = shell_dims[sh_idx];
        for i in 0..dim {
            map[off + i] = atom;
        }
    }
    map
}

/// Compute the closed-shell semilocal XC nuclear gradient.
///
/// Currently supports **LDA only**. GGA / hybrid-GGA / RSH require AO Hessians
/// (second derivatives of χ) which are not yet implemented in ao_grid.rs.
///
/// Arguments:
/// - `mol`: molecule (used for `natoms`)
/// - `d_total`: closed-shell total density matrix (= 2·D_α)
/// - `xc_name`: functional name (must be LDA-family)
/// - `dft_grid`: main grid spec
/// - `bf_to_atom_map`: precomputed AO→atom mapping (length nbf)
/// - `chi`: AO values on the grid, shape `(nbf, npts)`
/// - `dchi`: AO gradients on the grid, shape `(3, nbf, npts)`
///
/// Returns the XC contribution to the gradient as a `(natoms, 3)` array.
#[allow(clippy::too_many_arguments)]
pub fn xc_gradient_closed_lda(
    mol: &Molecule,
    d_total: &Array2<f64>,
    xc_name: &str,
    bf_to_atom_map: &[usize],
    chi: &Array2<f64>,
    dchi: &Array3<f64>,
    grid_weights: &[f64],
) -> Result<Array2<f64>, KsGradError> {
    let xc: XcDef = xc_def_from_name(xc_name)?;
    // Ensure all component functionals are LDA family.
    for f in &xc.funcs {
        if !matches!(f.family(), FunctionalFamily::Lda) {
            return Err(KsGradError::UnsupportedFamily(f.family()));
        }
    }

    let (nbf, npts) = chi.dim();
    debug_assert_eq!(dchi.dim(), (3, nbf, npts));
    debug_assert_eq!(d_total.dim(), (nbf, nbf));
    debug_assert_eq!(grid_weights.len(), npts);
    debug_assert_eq!(bf_to_atom_map.len(), nbf);

    // Evaluate ρ and v_ρ on the grid (D_total → ρ via the closed-shell eval).
    let dens = eval_density_closed(d_total, chi, dchi);

    let rho_slice = dens.rho.as_slice().expect("rho is contiguous");

    // Accumulate v_ρ across all component functionals.
    let mut vrho_total = vec![0.0_f64; npts];
    for (i, func) in xc.funcs.iter().enumerate() {
        let w_i = xc.weights.as_ref().map_or(1.0, |ws| ws[i]);
        let mut exc = vec![0.0_f64; npts];
        let mut vrho = vec![0.0_f64; npts];
        func.eval_lda_unpolarized(rho_slice, &mut exc, &mut vrho);
        for g in 0..npts {
            vrho_total[g] += w_i * vrho[g];
        }
    }

    // Build per-grid-point pre-scaled coefficient: t_g = w_g · v_ρ(r_g).
    let mut t = vec![0.0_f64; npts];
    for g in 0..npts {
        t[g] = grid_weights[g] * vrho_total[g];
    }

    // ∂E_xc/∂R_A,axis = -2 Σ_g t_g · M_μ(g) · ∂χ_μ/∂x_axis(g)  for μ ∈ A
    //
    // where:
    //   t_g     = w_g · v_ρ(r_g)
    //   M_μ(g)  = Σ_ν D_μν · χ_ν(r_g)
    //   dchi    = ∂χ/∂x (electron-position gradient; ∂χ/∂R_A = -∂χ/∂x for μ ∈ A)
    //
    // Derivation:
    //   ∂ρ/∂R_A,axis = -2 · Σ_{μ∈A, ν} D_μν · ∂_axis χ_μ · χ_ν
    //   ∂E_xc/∂R_A   = Σ_g w_g · v_ρ · ∂ρ/∂R_A
    //                = -2 · Σ_g w_g · v_ρ · Σ_{μ∈A, ν} D_μν · ∂_axis χ_μ · χ_ν
    //
    // M = D · χ as a matrix product (D is (nbf, nbf), χ is (nbf, npts)). FD-verified
    // against H2/STO-3G LDA — see tests/dft_gradient_lda.rs. Runs before the
    // rayon-gated row_dot contraction below starts. Opt-in BLAS raise via
    // FERRIC_BLAS_THREADS (default 1, unchanged behavior).
    let m: Array2<f64> = with_blas_threads(opt_in_blas_threads(), || d_total.dot(chi));

    let natoms = mol.atoms.len();
    let mut grad = Array2::<f64>::zeros((natoms, 3));

    // Pre-scale M by the per-point coefficient t once (fused, reused across the
    // three axes): mt[μ,g] = m[μ,g] · t[g]. Then each axis is a row-wise
    // contraction Σ_g mt[μ,g] · ∂_axis χ[μ,g].
    let mt = scale_cols(&m.view(), &ArrayView1::from(&t[..]));
    let partials: [ndarray::Array1<f64>; 3] = std::array::from_fn(|axis| {
        row_dot(&mt.view(), &dchi.index_axis(Axis(0), axis))
    });
    scatter_partials(&partials, bf_to_atom_map, &mut grad);

    Ok(grad)
}

/// Convenience wrapper: build the molecular grid, evaluate AOs, then call
/// `xc_gradient_closed_lda`. Used by ferric-scf's KS gradient driver. Adds
/// the Becke partition-weight grid-response correction (P2.1, PySCF
/// convention) automatically.
// Eight args carry the molecule, density, XC choice, grid config, and the
// shell-decomposition arrays the caller already has; bundling them into a struct
// would only move the boilerplate to the call site.
#[allow(clippy::too_many_arguments)]
pub fn xc_gradient_closed_lda_from_density(
    mol: &Molecule,
    bs: &ferric_core::basis::BasisSet,
    d_total: &Array2<f64>,
    xc_name: &str,
    grid_cfg: &AtomicGridConfig,
    shell_to_atom: &[usize],
    shell_offsets: &[usize],
    shell_dims: &[usize],
) -> Result<Array2<f64>, KsGradError> {
    let nbf = d_total.nrows();
    let (grid, weight1) = build_atomic_grid_with_response(mol, grid_cfg)?;
    // Gate the AO tensors (and everything allocated on top of them) against
    // what the grid + weight1 already left resident — see `xc_gradient_plan`.
    // The LDA path needs only χ + ∇χ, and never builds `mdchi`, but declaring
    // the shared shape keeps one census for the whole module.
    xc_gradient_plan(
        "KS-DFT LDA XC gradient",
        nbf,
        grid.len(),
        mol.atoms.len(),
        crate::ao_grid::AoGridKind::ValueAndGrad,
        false,
        true,
    )
    .check()?;
    let pts: Vec<[f64; 3]> = grid.iter().map(|g| g.xyz).collect();
    let (chi, dchi) = crate::ao_grid::eval_basis_and_grad_on_points(mol, bs, &pts)?;
    let weights: Vec<f64> = grid.iter().map(|g| g.weight).collect();
    let map = bf_to_atom(shell_to_atom, shell_offsets, shell_dims, nbf);
    let mut grad =
        xc_gradient_closed_lda(mol, d_total, xc_name, &map, &chi, &dchi, &weights)?;

    // Grid-response correction (PySCF grids_response_cc convention). Two
    // pieces sum to translational invariance against the existing AO-derivative
    // term in `xc_gradient_closed_lda`:
    //
    //   (1) Weight response (all atoms B):
    //         Δgrad[B,α] += Σ_g weight1[g,B,α] · ε_xc(r_g) · ρ(r_g)
    //   (2) Grid-coord response (B = home(g) only):
    //         Δgrad[A,α] += Σ_{g: home=A} w_g · v_ρ(r_g) · ∂ρ/∂r^α(r_g)
    //
    // weight1 here already includes the home-translation ∇_r piece needed for
    // exact Σ_B weight1[g,B,α] = 0 (see build_atomic_grid_with_response).
    let xc: XcDef = xc_def_from_name(xc_name)?;
    let dens = eval_density_closed(d_total, &chi, &dchi);
    let rho_slice = dens.rho.as_slice().expect("rho is contiguous");
    let npts = rho_slice.len();
    let mut eps_total = vec![0.0_f64; npts];
    let mut vrho_total = vec![0.0_f64; npts];
    for (i, func) in xc.funcs.iter().enumerate() {
        let w_i = xc.weights.as_ref().map_or(1.0, |ws| ws[i]);
        let mut exc = vec![0.0_f64; npts];
        let mut vrho = vec![0.0_f64; npts];
        func.eval_lda_unpolarized(rho_slice, &mut exc, &mut vrho);
        for g in 0..npts {
            eps_total[g] += w_i * exc[g];
            vrho_total[g] += w_i * vrho[g];
        }
    }
    let natoms = mol.atoms.len();
    // (1) weight response.
    for g in 0..npts {
        let f = eps_total[g] * rho_slice[g];
        for b in 0..natoms {
            grad[(b, 0)] += weight1[g][b][0] * f;
            grad[(b, 1)] += weight1[g][b][1] * f;
            grad[(b, 2)] += weight1[g][b][2] * f;
        }
    }
    // (2) grid-coordinate response (home atom of each grid point).
    for (gi, gp) in grid.iter().enumerate() {
        let a = gp.home_atom;
        let w = gp.weight;
        let vr = vrho_total[gi];
        grad[(a, 0)] += w * vr * dens.grad[(0, gi)];
        grad[(a, 1)] += w * vr * dens.grad[(1, gi)];
        grad[(a, 2)] += w * vr * dens.grad[(2, gi)];
    }

    Ok(grad)
}

/// Low-level GGA-style gradient assembly from precomputed per-grid-point
/// potentials. Reused by:
///   * the semilocal-XC gradient (v_ρ, v_σ from libxc)
///   * the VV10 nonlocal gradient (v_ρ, v_σ from the VV10 pair sum)
///
/// Formula (closed shell, AO basis derivative only — no grid response):
/// ```text
///   ∂E/∂R_A,axis = -2 Σ_g · Σ_{μ∈A, ν} D_μν · [
///       w·v_ρ · ∂_axis χ_μ · χ_ν
///     + Σ_b w·2·v_σ·∇ρ_b · (∂_axis χ_μ · ∂_b χ_ν + ∂²_{axis,b} χ_μ · χ_ν)
///   ]
/// ```
/// where v_ρ and v_σ are the input potentials supplied per grid point.
#[allow(clippy::too_many_arguments)]
pub fn gga_gradient_from_potentials(
    natoms: usize,
    d_total: &Array2<f64>,
    chi: &Array2<f64>,
    dchi: &Array3<f64>,
    ddchi: &ndarray::Array4<f64>,
    grad_rho: &Array2<f64>,        // shape (3, npts) — ∇ρ on the grid
    weights: &[f64],
    vrho: &[f64],
    vsig: &[f64],
    bf_to_atom_map: &[usize],
    rho_floor: f64,
    rho: &[f64],
) -> Array2<f64> {
    let (nbf, npts) = chi.dim();
    debug_assert_eq!(dchi.dim(), (3, nbf, npts));
    debug_assert_eq!(ddchi.dim(), (3, 3, nbf, npts));
    debug_assert_eq!(weights.len(), npts);
    debug_assert_eq!(vrho.len(), npts);
    debug_assert_eq!(vsig.len(), npts);
    debug_assert_eq!(rho.len(), npts);
    debug_assert_eq!(bf_to_atom_map.len(), nbf);

    // Precompute per-point pre-scaled coefficients (zero out low-ρ points).
    let mut t_rho = vec![0.0_f64; npts];
    let mut t_sig = vec![0.0_f64; npts];
    for g in 0..npts {
        if rho[g] > rho_floor {
            t_rho[g] = weights[g] * vrho[g];
            t_sig[g] = weights[g] * 2.0 * vsig[g];
        }
    }

    // M_μ(g) = Σ_ν D_μν · χ_ν(r_g). Runs before the rayon-gated ao-partial
    // reductions below start. Opt-in BLAS raise via FERRIC_BLAS_THREADS
    // (default 1, unchanged behavior).
    let m: Array2<f64> = with_blas_threads(opt_in_blas_threads(), || d_total.dot(chi));
    // Mdχ[b, μ, g] = Σ_ν D_μν · ∂_b χ_ν(r_g)
    let mut mdchi = ndarray::Array3::<f64>::zeros((3, nbf, npts));
    for b in 0..3 {
        let slice = dchi.index_axis(ndarray::Axis(0), b);
        let prod: Array2<f64> =
            with_blas_threads(opt_in_blas_threads(), || d_total.dot(&slice));
        mdchi.index_axis_mut(ndarray::Axis(0), b).assign(&prod);
    }

    let mut grad = Array2::<f64>::zeros((natoms, 3));
    let c = gga_weight_columns(&t_sig, grad_rho);
    let partials = gga_ao_partials(&m, &mdchi, dchi, ddchi, &c, &t_rho);
    scatter_partials(&partials, bf_to_atom_map, &mut grad);
    grad
}

/// VV10 nuclear gradient (closed shell, no grid response).
///
/// Reuses the VV10 pair-sum from `vv10::add_vv10` to compute per-grid-point
/// v_ρ and v_σ, then assembles via `gga_gradient_from_potentials`. The NLC
/// grid is built fresh (the SCF's NLC grid is not persisted on the result).
///
/// Like all semilocal gradients in this round, the grid-weight response
/// (Becke partition derivative) is dropped — see module doc.
#[allow(clippy::too_many_arguments)]
pub fn vv10_gradient_from_density(
    mol: &Molecule,
    bs: &ferric_core::basis::BasisSet,
    d_total: &Array2<f64>,
    params: &crate::libxc::Vv10Params,
    nlc_grid_cfg: &AtomicGridConfig,
    shell_to_atom: &[usize],
    shell_offsets: &[usize],
    shell_dims: &[usize],
) -> Result<Array2<f64>, KsGradError> {
    let nbf = d_total.nrows();
    let grid = build_atomic_grid(mol, nlc_grid_cfg);
    // No grid response on this path (`build_atomic_grid`, not
    // `_with_response`), so no `weight1` term — but the AO Hessian and the
    // `m`/`mdchi` planes inside `gga_gradient_from_potentials` are the same.
    xc_gradient_plan(
        "VV10 nonlocal gradient",
        nbf,
        grid.len(),
        mol.atoms.len(),
        crate::ao_grid::AoGridKind::ValueGradHess,
        false,
        false,
    )
    .check()?;
    let pts: Vec<[f64; 3]> = grid.iter().map(|g| g.xyz).collect();
    let (chi, dchi, ddchi) = crate::ao_grid::eval_basis_grad_hess_on_points(mol, bs, &pts)?;

    // Density on the NLC grid.
    let dens = crate::density_on_grid::eval_density_closed(d_total, &chi, &dchi);

    // VV10 potentials from the pair-sum.
    let (vrho, vsig) = crate::vv10::compute_vv10_potentials(&grid, &dens, params);

    let weights: Vec<f64> = grid.iter().map(|g| g.weight).collect();
    let map = bf_to_atom(shell_to_atom, shell_offsets, shell_dims, nbf);

    let rho_slice: Vec<f64> = dens.rho.iter().copied().collect();
    let grad = gga_gradient_from_potentials(
        mol.atoms.len(),
        d_total,
        &chi, &dchi, &ddchi,
        &dens.grad,
        &weights,
        &vrho, &vsig,
        &map,
        1e-10,
        &rho_slice,
    );
    Ok(grad)
}

/// GGA gradient (closed shell). Requires AO Hessians; currently only supports
/// s and p shells. Caller is expected to validate the functional family.
///
/// Formula:
/// ```text
///   ∂E_xc/∂R_A,axis = -2 Σ_g · Σ_{μ∈A, ν} D_μν · [
///       w·v_ρ · ∂_axis χ_μ · χ_ν
///     + Σ_b w·2·v_σ·∇ρ_b · (∂_axis χ_μ · ∂_b χ_ν + ∂²_{axis,b} χ_μ · χ_ν)
///   ]
/// ```
#[allow(clippy::too_many_arguments)]
pub fn xc_gradient_closed_gga_from_density(
    mol: &Molecule,
    bs: &ferric_core::basis::BasisSet,
    d_total: &Array2<f64>,
    xc_name: &str,
    grid_cfg: &AtomicGridConfig,
    shell_to_atom: &[usize],
    shell_offsets: &[usize],
    shell_dims: &[usize],
) -> Result<Array2<f64>, KsGradError> {
    let xc: XcDef = xc_def_from_name(xc_name)?;
    // Accept all families: caller handles the exact-exchange piece (via
    // ks_gradient_closed's K-gradient calls); the semilocal piece is what
    // we compute here, which works the same way for plain GGA, hybrid GGA,
    // and range-separated GGA. (VV10 nonlocal piece is NOT computed here.)
    // LDA family also accepted — falls back to vsigma=0.

    let nbf = d_total.nrows();
    let (grid, weight1) = build_atomic_grid_with_response(mol, grid_cfg)?;
    // Gate the whole working set — the 13 AO planes AND the m/mdchi planes
    // allocated after `check_ao_grid_budget` has already returned — against
    // what the grid + weight1 left resident. See `xc_gradient_plan`.
    xc_gradient_plan(
        "KS-DFT GGA XC gradient",
        nbf,
        grid.len(),
        mol.atoms.len(),
        crate::ao_grid::AoGridKind::ValueGradHess,
        false,
        true,
    )
    .check()?;
    let pts: Vec<[f64; 3]> = grid.iter().map(|g| g.xyz).collect();
    let (chi, dchi, ddchi) = crate::ao_grid::eval_basis_grad_hess_on_points(mol, bs, &pts)?;
    let weights: Vec<f64> = grid.iter().map(|g| g.weight).collect();
    let map = bf_to_atom(shell_to_atom, shell_offsets, shell_dims, nbf);

    let npts = chi.ncols();
    debug_assert_eq!(dchi.dim(), (3, nbf, npts));
    debug_assert_eq!(ddchi.dim(), (3, 3, nbf, npts));

    // Evaluate ρ, ∇ρ, σ, then v_ρ and v_σ per grid point.
    let dens = crate::density_on_grid::eval_density_closed(d_total, &chi, &dchi);
    let rho_slice = dens.rho.as_slice().expect("rho is contiguous");
    let sigma_slice = dens.sigma.as_slice().expect("sigma is contiguous");

    // Accumulate ε_xc(r_g) along with v_ρ and v_σ for the grid-response piece.
    let mut eps_total = vec![0.0_f64; npts];
    let mut vrho_total = vec![0.0_f64; npts];
    let mut vsigma_total = vec![0.0_f64; npts];
    for (i, func) in xc.funcs.iter().enumerate() {
        let w_i = xc.weights.as_ref().map_or(1.0, |ws| ws[i]);
        let mut exc = vec![0.0_f64; npts];
        let mut vrho = vec![0.0_f64; npts];
        match func.family() {
            FunctionalFamily::Lda => {
                func.eval_lda_unpolarized(rho_slice, &mut exc, &mut vrho);
            }
            _ => {
                let mut vsigma = vec![0.0_f64; npts];
                func.eval_gga_unpolarized(rho_slice, sigma_slice, &mut exc, &mut vrho, &mut vsigma);
                for g in 0..npts {
                    vsigma_total[g] += w_i * vsigma[g];
                }
            }
        }
        for g in 0..npts {
            eps_total[g] += w_i * exc[g];
            vrho_total[g] += w_i * vrho[g];
        }
    }

    // Pre-scale per-point quantities for tight inner loops.
    //   t_rho[g]  = w_g · v_ρ
    //   t_sig[g]  = w_g · 2 · v_σ          (multiplied by ∇ρ at inner-loop time)
    let mut t_rho = vec![0.0_f64; npts];
    let mut t_sig = vec![0.0_f64; npts];
    const RHO_FLOOR: f64 = 1e-10;
    for g in 0..npts {
        if dens.rho[g] > RHO_FLOOR {
            t_rho[g] = weights[g] * vrho_total[g];
            t_sig[g] = weights[g] * 2.0 * vsigma_total[g];
        }
    }

    // Precompute Σ_ν D_μν · χ_ν (= M_μ) and Σ_ν D_μν · ∂_b χ_ν (= Mdχ[b, μ, g])
    // via matrix products. Both run before the rayon-gated ao-partial
    // reductions below start. Opt-in BLAS raise via FERRIC_BLAS_THREADS
    // (default 1, unchanged behavior).
    let m: Array2<f64> = with_blas_threads(opt_in_blas_threads(), || d_total.dot(&chi));   // (nbf, npts)
    let mut mdchi = ndarray::Array3::<f64>::zeros((3, nbf, npts));
    for b in 0..3 {
        let slice = dchi.index_axis(ndarray::Axis(0), b);   // (nbf, npts)
        let prod: Array2<f64> =
            with_blas_threads(opt_in_blas_threads(), || d_total.dot(&slice));
        mdchi.index_axis_mut(ndarray::Axis(0), b).assign(&prod);
    }

    let natoms = mol.atoms.len();
    let mut grad = Array2::<f64>::zeros((natoms, 3));

    let c = gga_weight_columns(&t_sig, &dens.grad);
    let partials = gga_ao_partials(&m, &mdchi, &dchi, &ddchi, &c, &t_rho);
    scatter_partials(&partials, &map, &mut grad);

    // ── Grid-response correction (P2.1, PySCF convention) ──
    //
    // (1) Weight response: Σ_g weight1[g, B, α] · ε_xc · ρ for every atom B.
    // (2) Grid-coord response for B = home(g):
    //       w_g · [v_ρ · ∂_α ρ + 2 v_σ · Σ_b ∇ρ_b · ∂²_{αb} ρ]
    //
    // For the GGA piece we need ∂²_{αb} ρ on the grid, derived from D, χ,
    // ∇χ, and the AO Hessian we already evaluated for the AO-gradient sum:
    //   ∂²_{αb} ρ = 2 Σ_μν D · [∂_α χ_μ · ∂_b χ_ν + χ_ν · ∂²_{αb} χ_μ]
    //             = 2 Σ_μ ∂_α χ_μ · (D · ∂_b χ)_μ
    //             + 2 Σ_μ m_μ · ∂²_{αb} χ_μ
    for g in 0..npts {
        let f = eps_total[g] * rho_slice[g];
        for b in 0..natoms {
            grad[(b, 0)] += weight1[g][b][0] * f;
            grad[(b, 1)] += weight1[g][b][1] * f;
            grad[(b, 2)] += weight1[g][b][2] * f;
        }
    }
    // ∂²_{αb} ρ(r_g) = 2 Σ_μ [ ∂_α χ_μ · (D ∂_b χ)_μ + m_μ · ∂²_{αb} χ_μ ].
    // Precompute the per-(axis,b) column over the grid (reduction over μ) so the
    // per-point scatter below is a short serial pass. `hess_col[axis][b]` is a
    // length-npts array; the μ-reduction fans out over grid points via rayon.
    let hess_col: [[ndarray::Array1<f64>; 3]; 3] = std::array::from_fn(|axis| {
        let dchi_axis = dchi.index_axis(Axis(0), axis);
        let ddchi_a = ddchi.index_axis(Axis(0), axis);
        std::array::from_fn(|b| {
            let mdchi_b = mdchi.index_axis(Axis(0), b);
            let ddchi_ab = ddchi_a.index_axis(Axis(0), b);
            let mut hb = col_dot(&dchi_axis, &mdchi_b);
            hb += &col_dot(&m.view(), &ddchi_ab);
            hb.mapv_inplace(|v| 2.0 * v);
            hb
        })
    });
    for (gi, gp) in grid.iter().enumerate() {
        if dens.rho[gi] <= RHO_FLOOR {
            continue;
        }
        let a = gp.home_atom;
        let w = gp.weight;
        let vr = vrho_total[gi];
        let vs = vsigma_total[gi];
        // ∂_α ρ already in dens.grad.
        for k in 0..3 {
            grad[(a, k)] += w * vr * dens.grad[(k, gi)];
        }
        // 2 v_σ Σ_b ∇ρ_b · ∂²_{αb} ρ.
        for axis in 0..3 {
            let mut sum_b = 0.0_f64;
            for b in 0..3 {
                sum_b += dens.grad[(b, gi)] * hess_col[axis][b][gi];
            }
            grad[(a, axis)] += w * 2.0 * vs * sum_b;
        }
    }

    Ok(grad)
}

/// Meta-GGA gradient (closed shell). SCAN / r2SCAN / TPSS — τ-dependent, no
/// density Laplacian (matching the SCF energy path, which passes a zero `lapl`
/// buffer and discards `vlapl`).
///
/// Extends [`xc_gradient_closed_gga_from_density`] with the two τ terms:
///
/// ```text
///   AO-derivative:   ∂E/∂R_{A,α} += −Σ_g w_g v_τ Σ_b Σ_{μ∈A} ∂²_{αb} χ_μ (D ∂_b χ)_μ
///   grid response:   ∂E/∂R_{A,α} += Σ_{g: home=A} w_g v_τ ∂_α τ(r_g)
/// ```
///
/// See [`mgga_tau_partials`] for the derivation of the first (including why the
/// ½ in τ's definition cancels against the μ↔ν doubling) and
/// [`tau_spatial_grad`] for the second.
#[allow(clippy::too_many_arguments)]
pub fn xc_gradient_closed_mgga_from_density(
    mol: &Molecule,
    bs: &ferric_core::basis::BasisSet,
    d_total: &Array2<f64>,
    xc_name: &str,
    grid_cfg: &AtomicGridConfig,
    shell_to_atom: &[usize],
    shell_offsets: &[usize],
    shell_dims: &[usize],
) -> Result<Array2<f64>, KsGradError> {
    let xc: XcDef = xc_def_from_name(xc_name)?;

    let nbf = d_total.nrows();
    let (grid, weight1) = build_atomic_grid_with_response(mol, grid_cfg)?;
    // Gate the whole working set — the 13 AO planes AND the m/mdchi planes
    // allocated after `check_ao_grid_budget` has already returned — against
    // what the grid + weight1 left resident. See `xc_gradient_plan`.
    xc_gradient_plan(
        "KS-DFT meta-GGA XC gradient",
        nbf,
        grid.len(),
        mol.atoms.len(),
        crate::ao_grid::AoGridKind::ValueGradHess,
        false,
        true,
    )
    .check()?;
    let pts: Vec<[f64; 3]> = grid.iter().map(|g| g.xyz).collect();
    let (chi, dchi, ddchi) = crate::ao_grid::eval_basis_grad_hess_on_points(mol, bs, &pts)?;
    let weights: Vec<f64> = grid.iter().map(|g| g.weight).collect();
    let map = bf_to_atom(shell_to_atom, shell_offsets, shell_dims, nbf);

    let npts = chi.ncols();
    debug_assert_eq!(dchi.dim(), (3, nbf, npts));
    debug_assert_eq!(ddchi.dim(), (3, 3, nbf, npts));

    // ρ, ∇ρ, σ and τ on the grid. `eval_tau_closed` fed the TOTAL D returns the
    // total τ, which is the unpolarized libxc convention (τ = τ_α + τ_β) — the
    // same call the SCF energy path makes.
    let dens = eval_density_closed(d_total, &chi, &dchi);
    let tau = eval_tau_closed(d_total, &dchi);
    let rho_slice = dens.rho.as_slice().expect("rho is contiguous");
    let sigma_slice = dens.sigma.as_slice().expect("sigma is contiguous");
    let tau_slice = tau.as_slice().expect("tau is contiguous");

    let mut eps_total = vec![0.0_f64; npts];
    let mut vrho_total = vec![0.0_f64; npts];
    let mut vsigma_total = vec![0.0_f64; npts];
    let mut vtau_total = vec![0.0_f64; npts];
    for (i, func) in xc.funcs.iter().enumerate() {
        let w_i = xc.weights.as_ref().map_or(1.0, |ws| ws[i]);
        let mut exc = vec![0.0_f64; npts];
        let mut vrho = vec![0.0_f64; npts];
        match func.family() {
            FunctionalFamily::Lda => {
                func.eval_lda_unpolarized(rho_slice, &mut exc, &mut vrho);
            }
            FunctionalFamily::MetaGga => {
                let mut vsigma = vec![0.0_f64; npts];
                let mut vtau = vec![0.0_f64; npts];
                func.eval_mgga_unpolarized(
                    rho_slice, sigma_slice, tau_slice,
                    &mut exc, &mut vrho, &mut vsigma, &mut vtau,
                );
                for g in 0..npts {
                    vsigma_total[g] += w_i * vsigma[g];
                    vtau_total[g] += w_i * vtau[g];
                }
            }
            _ => {
                let mut vsigma = vec![0.0_f64; npts];
                func.eval_gga_unpolarized(rho_slice, sigma_slice, &mut exc, &mut vrho, &mut vsigma);
                for g in 0..npts {
                    vsigma_total[g] += w_i * vsigma[g];
                }
            }
        }
        for g in 0..npts {
            eps_total[g] += w_i * exc[g];
            vrho_total[g] += w_i * vrho[g];
        }
    }

    const RHO_FLOOR: f64 = 1e-10;
    let mut t_rho = vec![0.0_f64; npts];
    let mut t_sig = vec![0.0_f64; npts];
    let mut t_tau = vec![0.0_f64; npts];
    for g in 0..npts {
        if dens.rho[g] > RHO_FLOOR {
            t_rho[g] = weights[g] * vrho_total[g];
            t_sig[g] = weights[g] * 2.0 * vsigma_total[g];
            t_tau[g] = weights[g] * vtau_total[g];
        }
    }

    let m: Array2<f64> = with_blas_threads(opt_in_blas_threads(), || d_total.dot(&chi));
    let mdchi = build_mdchi(d_total, &dchi);

    let natoms = mol.atoms.len();
    let mut grad = Array2::<f64>::zeros((natoms, 3));

    // ρ + σ AO-derivative terms (identical to the GGA path).
    let c = gga_weight_columns(&t_sig, &dens.grad);
    let partials = gga_ao_partials(&m, &mdchi, &dchi, &ddchi, &c, &t_rho);
    scatter_partials(&partials, &map, &mut grad);
    // τ AO-derivative term.
    let tau_partials = mgga_tau_partials(&mdchi, &ddchi, &t_tau);
    scatter_partials(&tau_partials, &map, &mut grad);

    // ── Grid-response correction (same convention as the GGA path) ──
    // (1) weight response.
    for g in 0..npts {
        let f = eps_total[g] * rho_slice[g];
        for b in 0..natoms {
            grad[(b, 0)] += weight1[g][b][0] * f;
            grad[(b, 1)] += weight1[g][b][1] * f;
            grad[(b, 2)] += weight1[g][b][2] * f;
        }
    }
    // (2) home-translation of the integrand: ρ, σ and τ pieces.
    let hess_col: [[ndarray::Array1<f64>; 3]; 3] = std::array::from_fn(|axis| {
        let dchi_axis = dchi.index_axis(Axis(0), axis);
        let ddchi_a = ddchi.index_axis(Axis(0), axis);
        std::array::from_fn(|b| {
            let mdchi_b = mdchi.index_axis(Axis(0), b);
            let ddchi_ab = ddchi_a.index_axis(Axis(0), b);
            let mut hb = col_dot(&dchi_axis, &mdchi_b);
            hb += &col_dot(&m.view(), &ddchi_ab);
            hb.mapv_inplace(|v| 2.0 * v);
            hb
        })
    });
    let dtau = tau_spatial_grad(&mdchi, &ddchi);
    for (gi, gp) in grid.iter().enumerate() {
        if dens.rho[gi] <= RHO_FLOOR {
            continue;
        }
        let a = gp.home_atom;
        let w = gp.weight;
        let vr = vrho_total[gi];
        let vs = vsigma_total[gi];
        let vt = vtau_total[gi];
        for k in 0..3 {
            grad[(a, k)] += w * vr * dens.grad[(k, gi)];
        }
        for axis in 0..3 {
            let mut sum_b = 0.0_f64;
            for b in 0..3 {
                sum_b += dens.grad[(b, gi)] * hess_col[axis][b][gi];
            }
            grad[(a, axis)] += w * 2.0 * vs * sum_b;
            grad[(a, axis)] += w * vt * dtau[axis][gi];
        }
    }

    Ok(grad)
}

/// Spin-polarized (UKS) analytic XC gradient.
///
/// Per-spin gradient (AO-derivative term only — no grid response):
/// ```text
///   ∂E_xc/∂R_A,axis = -2 Σ_σ Σ_g Σ_{μ∈A, ν} D_σ_μν · [
///       w · v_ρσ · ∂_axis χ_μ · χ_ν
///     + Σ_b w · G^σ_b · (∂_axis χ_μ · ∂_b χ_ν + ∂²_{axis,b} χ_μ · χ_ν)
///   ]
/// ```
/// where
///   `G^α_b = 2 v_σαα · ∇ρ_α_b + v_σαβ · ∇ρ_β_b`
///   `G^β_b = 2 v_σββ · ∇ρ_β_b + v_σαβ · ∇ρ_α_b`
#[allow(clippy::too_many_arguments)]
pub fn xc_gradient_uks_from_density(
    mol: &Molecule,
    bs: &ferric_core::basis::BasisSet,
    d_a: &Array2<f64>,
    d_b: &Array2<f64>,
    xc_name: &str,
    grid_cfg: &AtomicGridConfig,
    shell_to_atom: &[usize],
    shell_offsets: &[usize],
    shell_dims: &[usize],
) -> Result<Array2<f64>, KsGradError> {
    let xc: XcDef = xc_def_from_name_nspin(xc_name, 2)?;

    let nbf = d_a.nrows();
    let (grid, weight1) = build_atomic_grid_with_response(mol, grid_cfg)?;
    // Gate the whole working set — the 13 AO planes AND the m/mdchi planes
    // allocated after `check_ao_grid_budget` has already returned — against
    // what the grid + weight1 left resident. See `xc_gradient_plan`.
    xc_gradient_plan(
        "KS-DFT UKS GGA XC gradient",
        nbf,
        grid.len(),
        mol.atoms.len(),
        crate::ao_grid::AoGridKind::ValueGradHess,
        true,
        true,
    )
    .check()?;
    let pts: Vec<[f64; 3]> = grid.iter().map(|g| g.xyz).collect();
    let (chi, dchi, ddchi) = crate::ao_grid::eval_basis_grad_hess_on_points(mol, bs, &pts)?;
    let weights: Vec<f64> = grid.iter().map(|g| g.weight).collect();
    let map = bf_to_atom(shell_to_atom, shell_offsets, shell_dims, nbf);
    let npts = chi.ncols();

    // Polarized densities.
    let dens = eval_density_uks(d_a, d_b, &chi, &dchi);

    // Build interleaved libxc input.
    let mut rho_in = vec![0.0_f64; 2 * npts];
    let mut sigma_in = vec![0.0_f64; 3 * npts];
    for g in 0..npts {
        rho_in[2 * g + 0] = dens.rho_a[g];
        rho_in[2 * g + 1] = dens.rho_b[g];
        sigma_in[3 * g + 0] = dens.sigma[(0, g)];
        sigma_in[3 * g + 1] = dens.sigma[(1, g)];
        sigma_in[3 * g + 2] = dens.sigma[(2, g)];
    }

    let mut eps_total = vec![0.0_f64; npts];
    let mut vrho_a = vec![0.0_f64; npts];
    let mut vrho_b = vec![0.0_f64; npts];
    let mut vsig_aa = vec![0.0_f64; npts];
    let mut vsig_ab = vec![0.0_f64; npts];
    let mut vsig_bb = vec![0.0_f64; npts];

    for (i, func) in xc.funcs.iter().enumerate() {
        let w_i = xc.weights.as_ref().map_or(1.0, |ws| ws[i]);
        let mut exc = vec![0.0_f64; npts];
        let mut vrho = vec![0.0_f64; 2 * npts];
        match func.family() {
            FunctionalFamily::Lda => {
                func.eval_lda_polarized(&rho_in, &mut exc, &mut vrho);
            }
            _ => {
                let mut vsig = vec![0.0_f64; 3 * npts];
                func.eval_gga_polarized(
                    &rho_in, &sigma_in,
                    &mut exc, &mut vrho, &mut vsig,
                );
                for g in 0..npts {
                    vsig_aa[g] += w_i * vsig[3 * g + 0];
                    vsig_ab[g] += w_i * vsig[3 * g + 1];
                    vsig_bb[g] += w_i * vsig[3 * g + 2];
                }
            }
        }
        for g in 0..npts {
            eps_total[g] += w_i * exc[g];
            vrho_a[g] += w_i * vrho[2 * g + 0];
            vrho_b[g] += w_i * vrho[2 * g + 1];
        }
    }

    const RHO_FLOOR: f64 = 1e-10;
    let natoms = mol.atoms.len();
    let mut grad = Array2::<f64>::zeros((natoms, 3));

    // Inner kernel per spin σ.
    let add_spin_contribution = |d_sigma: &Array2<f64>,
                                     vrho_sigma: &[f64],
                                     vsig_same: &[f64],
                                     vsig_cross: &[f64],
                                     grad_same: &Array2<f64>,
                                     grad_cross: &Array2<f64>,
                                     rho_sigma: &ndarray::Array1<f64>,
                                     grad_out: &mut Array2<f64>| {
        let mut t_rho = vec![0.0_f64; npts];
        // c[b,g] = G^σ_b(g) = w_g · (2 v_σ(σσ) · ∇ρ_σ_b + v_σ(αβ) · ∇ρ_other_b).
        // (For UKS the weight w and the factor 2 are folded into c directly, so
        // the LDA weight t_rho carries no extra factor — matching the original.)
        let mut c = Array2::<f64>::zeros((3, npts));
        for g in 0..npts {
            if rho_sigma[g] > RHO_FLOOR {
                t_rho[g] = weights[g] * vrho_sigma[g];
                let w = weights[g];
                for b in 0..3 {
                    c[(b, g)] = w * (
                          2.0 * vsig_same[g] * grad_same[(b, g)]
                        +       vsig_cross[g] * grad_cross[(b, g)]
                    );
                }
            }
        }
        // Both GEMMs run before the rayon-gated ao-partial reductions below
        // start; add_spin_contribution itself is called sequentially (twice),
        // never from inside rayon. Opt-in BLAS raise via FERRIC_BLAS_THREADS
        // (default 1, unchanged behavior).
        let m: Array2<f64> = with_blas_threads(opt_in_blas_threads(), || d_sigma.dot(&chi));
        let mut mdchi = ndarray::Array3::<f64>::zeros((3, nbf, npts));
        for b in 0..3 {
            let slice = dchi.index_axis(ndarray::Axis(0), b);
            let prod: Array2<f64> =
                with_blas_threads(opt_in_blas_threads(), || d_sigma.dot(&slice));
            mdchi.index_axis_mut(ndarray::Axis(0), b).assign(&prod);
        }
        let partials = gga_ao_partials(&m, &mdchi, &dchi, &ddchi, &c, &t_rho);
        scatter_partials(&partials, &map, grad_out);
    };

    add_spin_contribution(
        d_a, &vrho_a, &vsig_aa, &vsig_ab,
        &dens.grad_a, &dens.grad_b, &dens.rho_a,
        &mut grad,
    );
    add_spin_contribution(
        d_b, &vrho_b, &vsig_bb, &vsig_ab,
        &dens.grad_b, &dens.grad_a, &dens.rho_b,
        &mut grad,
    );

    // ── Grid-response correction (P2.1, PySCF convention) ──
    //
    // (1) Weight response: Σ_g weight1[g, B, α] · ε_xc · ρ_tot for every atom B.
    //     ε_xc is the polarized exchange-correlation energy density (libxc-returned).
    //     ρ_tot = ρ_α + ρ_β.
    //
    // (2) Grid-coord response for B = home(g): the total derivative of the
    //     polarized integrand under r_g translation.
    //       ∂(ε_xc ρ)/∂r^α
    //         = v_{ρα} · ∂_α ρ_α + v_{ρβ} · ∂_α ρ_β
    //         + 2 v_{σαα} · Σ_b ∇ρ_α_b · ∂²_{αb} ρ_α
    //         + 2 v_{σββ} · Σ_b ∇ρ_β_b · ∂²_{αb} ρ_β
    //         + v_{σαβ} · Σ_b (∇ρ_α_b · ∂²_{αb} ρ_β + ∇ρ_β_b · ∂²_{αb} ρ_α)
    //
    // The mixed second derivative ∂²_{αb} ρ_σ is reconstructed from D_σ, χ,
    // ∇χ, and the AO Hessian (which the AO-derivative path already uses).
    //
    // Translational invariance is exact: Σ_b weight1[g,b,α] = 0 by
    // construction (lab-fixed ∂w/∂R sums to zero across atoms plus
    // the ∇_r piece reattributed to home), and the AO-derivative term
    // implements the lab-fixed-r path so the home-translation correction
    // exactly closes the chain rule.

    // Precompute per-spin (D · χ) and (D · ∂χ). Runs before the rayon-gated
    // hess_col reduction below starts. Opt-in BLAS raise via
    // FERRIC_BLAS_THREADS (default 1, unchanged behavior).
    let m_a: Array2<f64> = with_blas_threads(opt_in_blas_threads(), || d_a.dot(&chi));
    let m_b: Array2<f64> = with_blas_threads(opt_in_blas_threads(), || d_b.dot(&chi));
    let mut mdchi_a = Array3::<f64>::zeros((3, nbf, npts));
    let mut mdchi_b = Array3::<f64>::zeros((3, nbf, npts));
    for b in 0..3 {
        let slice = dchi.index_axis(ndarray::Axis(0), b);
        let pa: Array2<f64> = with_blas_threads(opt_in_blas_threads(), || d_a.dot(&slice));
        let pb: Array2<f64> = with_blas_threads(opt_in_blas_threads(), || d_b.dot(&slice));
        mdchi_a.index_axis_mut(ndarray::Axis(0), b).assign(&pa);
        mdchi_b.index_axis_mut(ndarray::Axis(0), b).assign(&pb);
    }

    // ∂²_{αb} ρ_σ(r_g) = 2 Σ_μ [ ∂_α χ_μ · (D_σ ∂_b χ)_μ + (D_σ χ)_μ · ∂²_{αb} χ_μ ].
    // (Factor 2 from the μ↔ν symmetry of the per-spin density; see the original
    // derivation comment retained here.) Precompute the full per-spin, per-(axis,b)
    // column over the grid via a μ-reduction so the per-point scatter is cheap.
    let hess_cols = |m_s: &Array2<f64>, mdchi_s: &Array3<f64>| -> [[ndarray::Array1<f64>; 3]; 3] {
        std::array::from_fn(|axis| {
            let dchi_axis = dchi.index_axis(Axis(0), axis);
            let ddchi_a = ddchi.index_axis(Axis(0), axis);
            std::array::from_fn(|b| {
                let mdchi_sb = mdchi_s.index_axis(Axis(0), b);
                let ddchi_ab = ddchi_a.index_axis(Axis(0), b);
                let mut hb = col_dot(&dchi_axis, &mdchi_sb);
                hb += &col_dot(&m_s.view(), &ddchi_ab);
                hb.mapv_inplace(|v| 2.0 * v);
                hb
            })
        })
    };
    let hess_a = hess_cols(&m_a, &mdchi_a);
    let hess_b = hess_cols(&m_b, &mdchi_b);

    // (1) weight response.
    for g in 0..npts {
        let f = eps_total[g] * (dens.rho_a[g] + dens.rho_b[g]);
        for b in 0..natoms {
            grad[(b, 0)] += weight1[g][b][0] * f;
            grad[(b, 1)] += weight1[g][b][1] * f;
            grad[(b, 2)] += weight1[g][b][2] * f;
        }
    }
    // (2) home-translation of the integrand.
    for (gi, gp) in grid.iter().enumerate() {
        if dens.rho_a[gi] + dens.rho_b[gi] <= RHO_FLOOR {
            continue;
        }
        let a = gp.home_atom;
        let w = gp.weight;
        let vra = vrho_a[gi];
        let vrb = vrho_b[gi];
        let vsaa = vsig_aa[gi];
        let vsbb = vsig_bb[gi];
        let vsab = vsig_ab[gi];
        for axis in 0..3 {
            // ρ-derivative piece.
            let rho_piece = vra * dens.grad_a[(axis, gi)] + vrb * dens.grad_b[(axis, gi)];
            // σ-derivative piece.
            let mut sig_piece = 0.0_f64;
            for b in 0..3 {
                let gba = dens.grad_a[(b, gi)];
                let gbb = dens.grad_b[(b, gi)];
                let h_aa = hess_a[axis][b][gi];
                let h_bb = hess_b[axis][b][gi];
                sig_piece += 2.0 * vsaa * gba * h_aa
                    + 2.0 * vsbb * gbb * h_bb
                    + vsab * (gba * h_bb + gbb * h_aa);
            }
            grad[(a, axis)] += w * (rho_piece + sig_piece);
        }
    }

    Ok(grad)
}

/// Spin-polarized (UKS / ROKS) meta-GGA analytic XC gradient.
///
/// Extends [`xc_gradient_uks_from_density`] with the per-spin τ terms. Each spin
/// channel contributes independently — the polarized meta-GGA kernel returns a
/// separate `v_τσ`, and τ_σ depends only on `D_σ`, so there is no cross-spin τ
/// coupling to worry about (unlike σ, which carries the αβ cross term):
///
/// ```text
///   AO-derivative:  ∂E/∂R_{A,α} += −Σ_σ Σ_g w_g v_τσ Σ_b Σ_{μ∈A} ∂²_{αb} χ_μ (D_σ ∂_b χ)_μ
///   grid response:  ∂E/∂R_{A,α} += Σ_σ Σ_{g: home=A} w_g v_τσ ∂_α τ_σ(r_g)
/// ```
///
/// See [`mgga_tau_partials`] for the derivation (with D → D_σ, τ → τ_σ).
#[allow(clippy::too_many_arguments)]
pub fn xc_gradient_uks_mgga_from_density(
    mol: &Molecule,
    bs: &ferric_core::basis::BasisSet,
    d_a: &Array2<f64>,
    d_b: &Array2<f64>,
    xc_name: &str,
    grid_cfg: &AtomicGridConfig,
    shell_to_atom: &[usize],
    shell_offsets: &[usize],
    shell_dims: &[usize],
) -> Result<Array2<f64>, KsGradError> {
    let xc: XcDef = xc_def_from_name_nspin(xc_name, 2)?;

    let nbf = d_a.nrows();
    let (grid, weight1) = build_atomic_grid_with_response(mol, grid_cfg)?;
    // Gate the whole working set — the 13 AO planes AND the m/mdchi planes
    // allocated after `check_ao_grid_budget` has already returned — against
    // what the grid + weight1 left resident. See `xc_gradient_plan`.
    xc_gradient_plan(
        "KS-DFT UKS meta-GGA XC gradient",
        nbf,
        grid.len(),
        mol.atoms.len(),
        crate::ao_grid::AoGridKind::ValueGradHess,
        true,
        true,
    )
    .check()?;
    let pts: Vec<[f64; 3]> = grid.iter().map(|g| g.xyz).collect();
    let (chi, dchi, ddchi) = crate::ao_grid::eval_basis_grad_hess_on_points(mol, bs, &pts)?;
    let weights: Vec<f64> = grid.iter().map(|g| g.weight).collect();
    let map = bf_to_atom(shell_to_atom, shell_offsets, shell_dims, nbf);
    let npts = chi.ncols();

    let dens = eval_density_uks(d_a, d_b, &chi, &dchi);
    let (tau_a, tau_b) = crate::density_on_grid::eval_tau_uks(d_a, d_b, &dchi);

    // Interleaved libxc input (per-spin components adjacent per point).
    let mut rho_in = vec![0.0_f64; 2 * npts];
    let mut sigma_in = vec![0.0_f64; 3 * npts];
    let mut tau_in = vec![0.0_f64; 2 * npts];
    for g in 0..npts {
        rho_in[2 * g] = dens.rho_a[g];
        rho_in[2 * g + 1] = dens.rho_b[g];
        sigma_in[3 * g] = dens.sigma[(0, g)];
        sigma_in[3 * g + 1] = dens.sigma[(1, g)];
        sigma_in[3 * g + 2] = dens.sigma[(2, g)];
        tau_in[2 * g] = tau_a[g];
        tau_in[2 * g + 1] = tau_b[g];
    }

    let mut eps_total = vec![0.0_f64; npts];
    let mut vrho_a = vec![0.0_f64; npts];
    let mut vrho_b = vec![0.0_f64; npts];
    let mut vsig_aa = vec![0.0_f64; npts];
    let mut vsig_ab = vec![0.0_f64; npts];
    let mut vsig_bb = vec![0.0_f64; npts];
    let mut vtau_a = vec![0.0_f64; npts];
    let mut vtau_b = vec![0.0_f64; npts];

    for (i, func) in xc.funcs.iter().enumerate() {
        let w_i = xc.weights.as_ref().map_or(1.0, |ws| ws[i]);
        let mut exc = vec![0.0_f64; npts];
        let mut vrho = vec![0.0_f64; 2 * npts];
        match func.family() {
            FunctionalFamily::Lda => {
                func.eval_lda_polarized(&rho_in, &mut exc, &mut vrho);
            }
            FunctionalFamily::MetaGga => {
                let mut vsig = vec![0.0_f64; 3 * npts];
                let mut vtau = vec![0.0_f64; 2 * npts];
                func.eval_mgga_polarized(
                    &rho_in, &sigma_in, &tau_in,
                    &mut exc, &mut vrho, &mut vsig, &mut vtau,
                );
                for g in 0..npts {
                    vsig_aa[g] += w_i * vsig[3 * g];
                    vsig_ab[g] += w_i * vsig[3 * g + 1];
                    vsig_bb[g] += w_i * vsig[3 * g + 2];
                    vtau_a[g] += w_i * vtau[2 * g];
                    vtau_b[g] += w_i * vtau[2 * g + 1];
                }
            }
            _ => {
                let mut vsig = vec![0.0_f64; 3 * npts];
                func.eval_gga_polarized(&rho_in, &sigma_in, &mut exc, &mut vrho, &mut vsig);
                for g in 0..npts {
                    vsig_aa[g] += w_i * vsig[3 * g];
                    vsig_ab[g] += w_i * vsig[3 * g + 1];
                    vsig_bb[g] += w_i * vsig[3 * g + 2];
                }
            }
        }
        for g in 0..npts {
            eps_total[g] += w_i * exc[g];
            vrho_a[g] += w_i * vrho[2 * g];
            vrho_b[g] += w_i * vrho[2 * g + 1];
        }
    }

    const RHO_FLOOR: f64 = 1e-10;
    let natoms = mol.atoms.len();
    let mut grad = Array2::<f64>::zeros((natoms, 3));

    // Per-spin (D_σ χ) and (D_σ ∂_b χ), reused by both the AO-derivative sum and
    // the grid-response second-derivative columns below.
    let m_a: Array2<f64> = with_blas_threads(opt_in_blas_threads(), || d_a.dot(&chi));
    let m_b: Array2<f64> = with_blas_threads(opt_in_blas_threads(), || d_b.dot(&chi));
    let mdchi_a = build_mdchi(d_a, &dchi);
    let mdchi_b = build_mdchi(d_b, &dchi);

    // ── AO-derivative term, per spin ──
    let add_spin = |m_s: &Array2<f64>,
                        mdchi_s: &Array3<f64>,
                        vrho_sigma: &[f64],
                        vsig_same: &[f64],
                        vsig_cross: &[f64],
                        vtau_sigma: &[f64],
                        grad_same: &Array2<f64>,
                        grad_cross: &Array2<f64>,
                        rho_sigma: &ndarray::Array1<f64>,
                        grad_out: &mut Array2<f64>| {
        let mut t_rho = vec![0.0_f64; npts];
        let mut t_tau = vec![0.0_f64; npts];
        let mut c = Array2::<f64>::zeros((3, npts));
        for g in 0..npts {
            if rho_sigma[g] > RHO_FLOOR {
                let w = weights[g];
                t_rho[g] = w * vrho_sigma[g];
                t_tau[g] = w * vtau_sigma[g];
                for b in 0..3 {
                    c[(b, g)] = w
                        * (2.0 * vsig_same[g] * grad_same[(b, g)]
                            + vsig_cross[g] * grad_cross[(b, g)]);
                }
            }
        }
        let partials = gga_ao_partials(m_s, mdchi_s, &dchi, &ddchi, &c, &t_rho);
        scatter_partials(&partials, &map, grad_out);
        let tau_partials = mgga_tau_partials(mdchi_s, &ddchi, &t_tau);
        scatter_partials(&tau_partials, &map, grad_out);
    };

    add_spin(
        &m_a, &mdchi_a, &vrho_a, &vsig_aa, &vsig_ab, &vtau_a,
        &dens.grad_a, &dens.grad_b, &dens.rho_a, &mut grad,
    );
    add_spin(
        &m_b, &mdchi_b, &vrho_b, &vsig_bb, &vsig_ab, &vtau_b,
        &dens.grad_b, &dens.grad_a, &dens.rho_b, &mut grad,
    );

    // ── Grid-response correction ──
    // (1) weight response.
    for g in 0..npts {
        let f = eps_total[g] * (dens.rho_a[g] + dens.rho_b[g]);
        for b in 0..natoms {
            grad[(b, 0)] += weight1[g][b][0] * f;
            grad[(b, 1)] += weight1[g][b][1] * f;
            grad[(b, 2)] += weight1[g][b][2] * f;
        }
    }

    // ∂²_{αb} ρ_σ columns, and ∂_α τ_σ columns.
    let hess_cols = |m_s: &Array2<f64>, mdchi_s: &Array3<f64>| -> [[ndarray::Array1<f64>; 3]; 3] {
        std::array::from_fn(|axis| {
            let dchi_axis = dchi.index_axis(Axis(0), axis);
            let ddchi_a = ddchi.index_axis(Axis(0), axis);
            std::array::from_fn(|b| {
                let mdchi_sb = mdchi_s.index_axis(Axis(0), b);
                let ddchi_ab = ddchi_a.index_axis(Axis(0), b);
                let mut hb = col_dot(&dchi_axis, &mdchi_sb);
                hb += &col_dot(&m_s.view(), &ddchi_ab);
                hb.mapv_inplace(|v| 2.0 * v);
                hb
            })
        })
    };
    let hess_a = hess_cols(&m_a, &mdchi_a);
    let hess_b = hess_cols(&m_b, &mdchi_b);
    let dtau_a = tau_spatial_grad(&mdchi_a, &ddchi);
    let dtau_b = tau_spatial_grad(&mdchi_b, &ddchi);

    // (2) home-translation of the integrand.
    for (gi, gp) in grid.iter().enumerate() {
        if dens.rho_a[gi] + dens.rho_b[gi] <= RHO_FLOOR {
            continue;
        }
        let a = gp.home_atom;
        let w = gp.weight;
        let vra = vrho_a[gi];
        let vrb = vrho_b[gi];
        let vsaa = vsig_aa[gi];
        let vsbb = vsig_bb[gi];
        let vsab = vsig_ab[gi];
        let vta = vtau_a[gi];
        let vtb = vtau_b[gi];
        for axis in 0..3 {
            let rho_piece = vra * dens.grad_a[(axis, gi)] + vrb * dens.grad_b[(axis, gi)];
            let mut sig_piece = 0.0_f64;
            for b in 0..3 {
                let gba = dens.grad_a[(b, gi)];
                let gbb = dens.grad_b[(b, gi)];
                let h_aa = hess_a[axis][b][gi];
                let h_bb = hess_b[axis][b][gi];
                sig_piece += 2.0 * vsaa * gba * h_aa
                    + 2.0 * vsbb * gbb * h_bb
                    + vsab * (gba * h_bb + gbb * h_aa);
            }
            let tau_piece = vta * dtau_a[axis][gi] + vtb * dtau_b[axis][gi];
            grad[(a, axis)] += w * (rho_piece + sig_piece + tau_piece);
        }
    }

    Ok(grad)
}

#[cfg(test)]
mod budget_tests {
    use super::*;

    // FERRIC_MEM_BUDGET_GB is process-global; serialize every test that sets
    // it against the crate-wide lock (same convention as ks.rs / ao_grid.rs).
    use crate::TEST_BUDGET_ENV_LOCK as ENV_LOCK;
    const VAR: &str = ferric_core::memory::ENV_UNIFIED;

    /// A shape whose declared working set exceeds the budget must be refused
    /// **before** the AO Hessian is allocated, and the message must name the
    /// term that dominates — a bare total is what made the historical
    /// incidents slow to diagnose.
    #[test]
    fn the_gate_rejects_an_over_budget_shape_and_names_the_largest_term() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        std::env::set_var(VAR, "1"); // 1 GiB

        // 2000 basis functions on a 500k-point grid: the AO tensors alone are
        // 13 · 2000 · 500_000 · 8 = 104 TB. Nothing subtle about this one.
        let plan = xc_gradient_plan(
            "KS-DFT UKS meta-GGA XC gradient",
            2000,
            500_000,
            50,
            AoGridKind::ValueGradHess,
            true,
            true,
        );
        let err = plan.check().unwrap_err().to_string();
        std::env::remove_var(VAR);

        assert!(err.contains("KS-DFT UKS meta-GGA XC gradient"), "{err}");
        assert!(err.contains("chi + dchi + ddchi"), "breakdown must name the AO term: {err}");
        // Largest contributor sorts first, so it is the first row of the table.
        let ao = err.find("chi + dchi + ddchi").expect("AO term present");
        let md = err.find("Mdchi").expect("m/mdchi term present");
        assert!(ao < md, "largest term must sort first:\n{err}");
    }

    /// The UKS peak really is larger than the closed-shell one, and both are
    /// larger than the 13 planes `check_ao_grid_budget` approves on its own —
    /// which is the entire bug this gate closes.
    #[test]
    fn the_declared_peak_exceeds_what_check_ao_grid_budget_alone_approves() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        std::env::set_var(VAR, "1024"); // large, so check() is not the subject here

        let (nbf, npts, natoms) = (300usize, 200_000usize, 20usize);
        let plane_bytes = nbf * npts * 8;
        let ao_only = AoGridKind::ValueGradHess.planes() * plane_bytes;

        let closed =
            xc_gradient_plan("closed", nbf, npts, natoms, AoGridKind::ValueGradHess, false, true);
        let uks =
            xc_gradient_plan("uks", nbf, npts, natoms, AoGridKind::ValueGradHess, true, true);
        std::env::remove_var(VAR);

        // Closed shell: 13 AO + 4 (m, mdchi) resident + 3 transient = 20 planes.
        assert!(
            closed.peak_bytes() >= ao_only + 7 * plane_bytes,
            "closed peak {} must exceed the AO-only gate {} by >= 7 planes",
            closed.peak_bytes(),
            ao_only,
        );
        // UKS: 13 + 8 + 3 = 24 planes.
        assert!(
            uks.peak_bytes() >= ao_only + 11 * plane_bytes,
            "UKS peak {} must exceed the AO-only gate {} by >= 11 planes",
            uks.peak_bytes(),
            ao_only,
        );
        assert!(uks.peak_bytes() > closed.peak_bytes(), "UKS holds two spins' m/mdchi");
    }

    /// An over-estimating guard is also a bug: a budget that genuinely fits the
    /// declared peak must pass, and a real (small) molecular shape must not be
    /// refused under an ordinary budget.
    #[test]
    fn an_ample_budget_still_passes_the_gate() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        std::env::set_var(VAR, "8"); // 8 GiB — an unremarkable desktop budget

        // H2O-sized: nbf 25, a default 75x302 grid over 3 atoms ~ 68k points.
        for &(kind, uks) in &[
            (AoGridKind::ValueAndGrad, false),
            (AoGridKind::ValueGradHess, false),
            (AoGridKind::ValueGradHess, true),
        ] {
            let plan = xc_gradient_plan("t", 25, 68_000, 3, kind, uks, true);
            assert!(
                plan.check().is_ok(),
                "a 25-function, 68k-point gradient must fit 8 GiB:\n{}",
                plan.report(),
            );
        }

        // And a mid-size one: 300 functions, 200k points, 20 atoms — 24 planes
        // is ~11.5 GB, so it must NOT fit 8 GiB, but must fit 32.
        let plan = xc_gradient_plan("t", 300, 200_000, 20, AoGridKind::ValueGradHess, true, true);
        assert!(plan.check().is_err(), "24 planes of 300x200k does not fit 8 GiB");
        std::env::set_var(VAR, "32");
        let plan = xc_gradient_plan("t", 300, 200_000, 20, AoGridKind::ValueGradHess, true, true);
        assert!(plan.check().is_ok(), "...but it does fit 32 GiB:\n{}", plan.report());

        std::env::remove_var(VAR);
    }

    /// `weight1` scales as `npts × natoms`, and the gate must see that: the
    /// same grid on ten times the atoms must cost about ten times as much.
    #[test]
    fn the_weight1_term_scales_with_both_npts_and_natoms() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        std::env::set_var(VAR, "1024");
        let small = crate::grid::grid_response_plan(100_000, 5).peak_bytes();
        let many_atoms = crate::grid::grid_response_plan(100_000, 50).peak_bytes();
        let many_pts = crate::grid::grid_response_plan(1_000_000, 5).peak_bytes();
        std::env::remove_var(VAR);

        assert!(many_atoms > 5 * small, "{many_atoms} vs {small}");
        assert!(many_pts > 9 * small, "{many_pts} vs {small}");
    }
}
