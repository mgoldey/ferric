//! GTO evaluation for cube-file grids.  Core AO evaluator lives in
//! `ferric_dft::ao_grid`; this module re-exports it and adds the
//! `GridSpec`-aware `eval_basis_on_grid` wrapper.

pub use ferric_dft::ao_grid::{eval_basis_on_points, nbasis, GtoEvalError};
use ferric_dft::ao_grid::{collect_shells, eval_shell};

use ferric_core::basis::num_functions;
use ferric_core::basis::BasisSet;
use ferric_core::mol::Molecule;
use ndarray::Array2;

use crate::cube::GridSpec;

/// Evaluate all basis functions on every grid point.
///
/// Returns `Array2` of shape `(nbasis, n_x*n_y*n_z)` where rows are basis functions
/// and columns are grid points in (ix, iy, iz) row-major order matching `export_cube`.
///
/// Memory: `nbasis × n_grid × 8` bytes. For a 50³ grid + 100 basis functions ≈ 100 MB.
/// Caller should keep this in mind for large systems / fine grids.
pub fn eval_basis_on_grid(
    mol: &Molecule,
    bs: &BasisSet,
    grid: &GridSpec,
) -> Result<Array2<f64>, GtoEvalError> {
    let shells = collect_shells(mol, bs)?;
    let nbf: usize = shells.iter().map(|s| num_functions(s.l, s.pure)).sum();
    let npts = grid.n_x * grid.n_y * grid.n_z;

    // Fail-fast size guard: the dense AO-on-grid buffer chi is (nbf, npts) f64
    // (:30). GridSpec::for_molecule derives npts = n_x·n_y·n_z from a
    // user-chosen spacing with no upper bound (cube.rs:53-55), so a fine grid on
    // a large molecule can request an unbounded allocation. No config budget on
    // this export path, so resolve from env/auto. Keep next to the chi alloc.
    let peak = nbf.saturating_mul(npts).saturating_mul(8); // f64
    ferric_core::memory::check_alloc(
        &format!("cube AO-on-grid (nbf={nbf}, npts={npts} = {}×{}×{})", grid.n_x, grid.n_y, grid.n_z),
        peak,
        ferric_core::memory::resolve_budget_bytes(None),
    )
    .map_err(|e| GtoEvalError::OutOfBudget(e.to_string()))?;

    let mut chi = Array2::<f64>::zeros((nbf, npts));

    // Assume orthogonal grid (step vectors are axis-aligned). The cube exporter
    // already enforces this; nonorthogonal grids would need a more general loop.
    let hx = grid.step_x[0];
    let hy = grid.step_y[1];
    let hz = grid.step_z[2];

    // Max num_functions for l ≤ 3: pure 7, Cartesian 10.
    let mut shell_buf = [0.0f64; 10];

    for ix in 0..grid.n_x {
        let x = grid.origin[0] + ix as f64 * hx;
        for iy in 0..grid.n_y {
            let y = grid.origin[1] + iy as f64 * hy;
            for iz in 0..grid.n_z {
                let z = grid.origin[2] + iz as f64 * hz;
                let g = (ix * grid.n_y + iy) * grid.n_z + iz;

                let mut row_offset = 0usize;
                for sh in &shells {
                    let n = num_functions(sh.l, sh.pure);
                    let buf = &mut shell_buf[..n];
                    let dx = x - sh.center[0];
                    let dy = y - sh.center[1];
                    let dz = z - sh.center[2];
                    eval_shell(sh, dx, dy, dz, buf)?;
                    for (i, &v) in buf.iter().enumerate() {
                        chi[(row_offset + i, g)] = v;
                    }
                    row_offset += n;
                }
            }
        }
    }
    Ok(chi)
}

/// Index from (ix, iy, iz) to flat grid offset (matches cube row-major order).
pub fn grid_index(grid: &GridSpec, ix: usize, iy: usize, iz: usize) -> usize {
    (ix * grid.n_y + iy) * grid.n_z + iz
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferric_core::basis;
    use ferric_core::mol::Molecule;

    fn h2_mol() -> Molecule {
        Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 0.7408\n", 0, 1).unwrap()
    }

    #[test]
    fn h2_sto3g_s_function_normalization() {
        // At the H nucleus, the STO-3G 1s contracted GTO should be positive and finite.
        let mol = h2_mol();
        let bs = basis::bundled("sto-3g").unwrap();
        let grid = GridSpec {
            origin: [0.0, 0.0, 0.0],
            n_x: 1, n_y: 1, n_z: 1,
            step_x: [1.0, 0.0, 0.0],
            step_y: [0.0, 1.0, 0.0],
            step_z: [0.0, 0.0, 1.0],
        };
        let chi = eval_basis_on_grid(&mol, &bs, &grid).unwrap();
        // 2 H atoms × 1 STO-3G 1s function each = 2 basis functions
        assert_eq!(chi.nrows(), 2);
        assert_eq!(chi.ncols(), 1);
        // First H at origin: 1s should be largest there
        assert!(chi[(0, 0)] > 0.0);
        // Second H at z=1.4: its 1s should be smaller at origin
        assert!(chi[(1, 0)] > 0.0);
        assert!(chi[(0, 0)] > chi[(1, 0)]);
    }

    // FERRIC_MEM_BUDGET_GB is process-global; serialize env-mutating tests
    // (blas_threads.rs / ferric-core memory.rs pattern).
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn eval_basis_on_grid_fails_fast_under_tiny_env_budget() {
        // M2 size guard: an unbounded npts (fine spacing) must ERROR cleanly
        // before the (nbf, npts) allocation when the budget is tiny.
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mol = h2_mol();
        let bs = basis::bundled("sto-3g").unwrap();
        let grid = GridSpec {
            origin: [0.0, 0.0, 0.0],
            n_x: 50, n_y: 50, n_z: 50, // 125k pts × 2 bf × 8 B = 2 MB > tiny budget
            step_x: [0.1, 0.0, 0.0],
            step_y: [0.0, 0.1, 0.0],
            step_z: [0.0, 0.0, 0.1],
        };
        std::env::set_var("FERRIC_MEM_BUDGET_GB", "0.000001");
        let res = eval_basis_on_grid(&mol, &bs, &grid);
        std::env::remove_var("FERRIC_MEM_BUDGET_GB");
        let err = res.unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("cube AO-on-grid") && msg.contains("budget is"),
            "unexpected: {msg}"
        );
    }
}
