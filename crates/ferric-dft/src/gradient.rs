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

use ferric_core::mol::Molecule;
use ndarray::{Array2, Array3, ArrayView1, ArrayView2, Axis, Zip};
use rayon::prelude::*;

use crate::density_on_grid::{eval_density_closed, eval_density_uks};
use crate::grid::{build_atomic_grid, build_atomic_grid_with_response, AtomicGridConfig};
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

#[derive(Debug, thiserror::Error)]
pub enum KsGradError {
    #[error("XC family {0:?} not supported in this gradient path")]
    UnsupportedFamily(FunctionalFamily),
    #[error("libxc resolver failed: {0}")]
    Libxc(LibxcError),
    #[error("AO eval failed: {0:?}")]
    AoEval(crate::ao_grid::GtoEvalError),
}

impl From<LibxcError> for KsGradError { fn from(e: LibxcError) -> Self { Self::Libxc(e) } }
impl From<crate::ao_grid::GtoEvalError> for KsGradError {
    fn from(e: crate::ao_grid::GtoEvalError) -> Self { Self::AoEval(e) }
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
    for func in &xc.funcs {
        let mut exc = vec![0.0_f64; npts];
        let mut vrho = vec![0.0_f64; npts];
        func.eval_lda_unpolarized(rho_slice, &mut exc, &mut vrho);
        for g in 0..npts {
            vrho_total[g] += vrho[g];
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
    // against H2/STO-3G LDA — see tests/dft_gradient_lda.rs.
    let m: Array2<f64> = d_total.dot(chi);

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
    let (grid, weight1) = build_atomic_grid_with_response(mol, grid_cfg);
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
    for func in &xc.funcs {
        let mut exc = vec![0.0_f64; npts];
        let mut vrho = vec![0.0_f64; npts];
        func.eval_lda_unpolarized(rho_slice, &mut exc, &mut vrho);
        for g in 0..npts {
            eps_total[g] += exc[g];
            vrho_total[g] += vrho[g];
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

    // M_μ(g) = Σ_ν D_μν · χ_ν(r_g)
    let m: Array2<f64> = d_total.dot(chi);
    // Mdχ[b, μ, g] = Σ_ν D_μν · ∂_b χ_ν(r_g)
    let mut mdchi = ndarray::Array3::<f64>::zeros((3, nbf, npts));
    for b in 0..3 {
        let slice = dchi.index_axis(ndarray::Axis(0), b);
        let prod: Array2<f64> = d_total.dot(&slice);
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
    let (grid, weight1) = build_atomic_grid_with_response(mol, grid_cfg);
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
    for func in &xc.funcs {
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
                    vsigma_total[g] += vsigma[g];
                }
            }
        }
        for g in 0..npts {
            eps_total[g] += exc[g];
            vrho_total[g] += vrho[g];
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
    // via matrix products.
    let m: Array2<f64> = d_total.dot(&chi);   // (nbf, npts)
    let mut mdchi = ndarray::Array3::<f64>::zeros((3, nbf, npts));
    for b in 0..3 {
        let slice = dchi.index_axis(ndarray::Axis(0), b);   // (nbf, npts)
        let prod: Array2<f64> = d_total.dot(&slice);
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
    let (grid, weight1) = build_atomic_grid_with_response(mol, grid_cfg);
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

    for func in &xc.funcs {
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
                    vsig_aa[g] += vsig[3 * g + 0];
                    vsig_ab[g] += vsig[3 * g + 1];
                    vsig_bb[g] += vsig[3 * g + 2];
                }
            }
        }
        for g in 0..npts {
            eps_total[g] += exc[g];
            vrho_a[g] += vrho[2 * g + 0];
            vrho_b[g] += vrho[2 * g + 1];
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
        let m: Array2<f64> = d_sigma.dot(&chi);
        let mut mdchi = ndarray::Array3::<f64>::zeros((3, nbf, npts));
        for b in 0..3 {
            let slice = dchi.index_axis(ndarray::Axis(0), b);
            let prod: Array2<f64> = d_sigma.dot(&slice);
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

    // Precompute per-spin (D · χ) and (D · ∂χ).
    let m_a: Array2<f64> = d_a.dot(&chi);
    let m_b: Array2<f64> = d_b.dot(&chi);
    let mut mdchi_a = Array3::<f64>::zeros((3, nbf, npts));
    let mut mdchi_b = Array3::<f64>::zeros((3, nbf, npts));
    for b in 0..3 {
        let slice = dchi.index_axis(ndarray::Axis(0), b);
        let pa: Array2<f64> = d_a.dot(&slice);
        let pb: Array2<f64> = d_b.dot(&slice);
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
