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
use ndarray::{Array2, Array3};

use crate::density_on_grid::eval_density_closed;
use crate::grid::{build_atomic_grid, AtomicGridConfig};
use crate::libxc::{xc_def_from_name, FunctionalFamily, LibxcError, XcDef};

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

    for axis in 0..3 {
        for mu in 0..nbf {
            let atom = bf_to_atom_map[mu];
            let mut sum = 0.0_f64;
            for g in 0..npts {
                sum += t[g] * m[(mu, g)] * dchi[(axis, mu, g)];
            }
            grad[(atom, axis)] -= 2.0 * sum;
        }
    }

    Ok(grad)
}

/// Convenience wrapper: build the molecular grid, evaluate AOs, then call
/// `xc_gradient_closed_lda`. Used by ferric-scf's KS gradient driver.
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
    let grid = build_atomic_grid(mol, grid_cfg);
    let pts: Vec<[f64; 3]> = grid.iter().map(|g| g.xyz).collect();
    let (chi, dchi) = crate::ao_grid::eval_basis_and_grad_on_points(mol, bs, &pts)?;
    let weights: Vec<f64> = grid.iter().map(|g| g.weight).collect();
    let map = bf_to_atom(shell_to_atom, shell_offsets, shell_dims, nbf);
    xc_gradient_closed_lda(mol, d_total, xc_name, &map, &chi, &dchi, &weights)
}
