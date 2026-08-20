//! GTO evaluation for cube-file grids.  Core AO evaluator lives in
//! `ferric_dft::ao_grid`; this module re-exports it and adds the
//! `GridSpec`-aware `eval_basis_on_grid` wrapper.

/// Re-exports of the core AO evaluator from `ferric_dft::ao_grid`.
pub use ferric_dft::ao_grid::{eval_basis_on_points, nbasis, GtoEvalError};
use ferric_dft::ao_grid::{collect_shells, eval_shell, LocatedShell};

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

    // Precompute each shell's row offset once (shared by serial and parallel
    // paths) instead of recomputing the running sum per grid point.
    let shell_offsets: Vec<usize> = {
        let mut off = 0usize;
        shells
            .iter()
            .map(|sh| {
                let o = off;
                off += num_functions(sh.l, sh.pure);
                o
            })
            .collect()
    };

    if npts < PAR_GRID_POINT_THRESHOLD {
        // Max num_functions for l ≤ 3: pure 7, Cartesian 10.
        let mut shell_buf = [0.0f64; 10];
        for ix in 0..grid.n_x {
            let x = grid.origin[0] + ix as f64 * hx;
            for iy in 0..grid.n_y {
                let y = grid.origin[1] + iy as f64 * hy;
                for iz in 0..grid.n_z {
                    let z = grid.origin[2] + iz as f64 * hz;
                    let g = grid_index(grid, ix, iy, iz);
                    eval_point_into(&shells, &shell_offsets, x, y, z, &mut shell_buf, |row, v| {
                        chi[(row, g)] = v;
                    })?;
                }
            }
        }
        return Ok(chi);
    }

    // Parallelize over grid points `g`. Each `g` is one full *column* of `chi`
    // (shape (nbf, npts), row-major: element (row, g) at row*npts + g) and no
    // two grid points ever write the same element — write-once, disjoint
    // columns, no accumulation — so the scatter below is data-race-free and
    // bit-identical to the serial loop regardless of thread count or
    // scheduling order. `eval_shell`/`radial` are pure scalar/vector math (no
    // BLAS, no engine, no shared mutable state), so each rayon worker only
    // needs its own `shell_buf` scratch, built once via `try_for_each_init`
    // (never per-point). `try_for_each_init` (not `for_each_init`) so a
    // `GtoEvalError` from any worker short-circuits and propagates instead of
    // being silently dropped.
    use rayon::prelude::*;
    let chi_ptr = chi.as_mut_ptr() as usize;
    let stride = npts; // row-major (nbf, npts): element (row, g) at row*stride + g
    let n_y = grid.n_y;
    let n_z = grid.n_z;
    let (ox, oy, oz) = (grid.origin[0], grid.origin[1], grid.origin[2]);

    (0..npts).into_par_iter().try_for_each_init(
        || [0.0f64; 10],
        |shell_buf, g| -> Result<(), GtoEvalError> {
            let ix = g / (n_y * n_z);
            let rem = g % (n_y * n_z);
            let iy = rem / n_z;
            let iz = rem % n_z;
            let x = ox + ix as f64 * hx;
            let y = oy + iy as f64 * hy;
            let z = oz + iz as f64 * hz;
            eval_point_into(&shells, &shell_offsets, x, y, z, shell_buf, |row, v| {
                let base = chi_ptr as *mut f64;
                // SAFETY: distinct `g` values (rayon work items) write to
                // distinct columns `row*stride + g` for every `row`, so
                // concurrent workers never touch the same element. See the
                // disjoint-column argument above.
                unsafe {
                    *base.add(row * stride + g) = v;
                }
            })
        },
    )?;
    Ok(chi)
}

/// Below this many grid points, run the serial loop directly — avoids
/// rayon/scratch-buffer overhead for small cube grids (mirrors the
/// `PAR_*_THRESHOLD` convention used across ferric-integrals/ferric-scf).
const PAR_GRID_POINT_THRESHOLD: usize = 512;

/// Evaluate every shell at one grid point `(x, y, z)` and hand each computed
/// AO value to `write(row, value)`. Shared by the serial and parallel paths
/// so both compute in exactly the same order (shell order, then
/// within-shell function order) and thus produce bit-identical output.
#[inline]
fn eval_point_into(
    shells: &[LocatedShell],
    shell_offsets: &[usize],
    x: f64,
    y: f64,
    z: f64,
    shell_buf: &mut [f64; 10],
    mut write: impl FnMut(usize, f64),
) -> Result<(), GtoEvalError> {
    for (sh, &row_offset) in shells.iter().zip(shell_offsets) {
        let n = num_functions(sh.l, sh.pure);
        let buf = &mut shell_buf[..n];
        let dx = x - sh.center[0];
        let dy = y - sh.center[1];
        let dz = z - sh.center[2];
        eval_shell(sh, dx, dy, dz, buf)?;
        for (i, &v) in buf.iter().enumerate() {
            write(row_offset + i, v);
        }
    }
    Ok(())
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
        // Holds ENV_LOCK (declared below) because eval_basis_on_grid reads the
        // process-global FERRIC_MEM_BUDGET_GB internally (resolve_budget_bytes(None))
        // -- any test calling it can observe another test's tiny-budget env
        // mutation under cargo test's default parallelism, not just the test
        // that sets the var itself.
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
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
    // (blas_threads.rs / ferric-core memory.rs pattern). MUST be held by
    // every test in this module that calls eval_basis_on_grid, not just the
    // one that sets the var -- resolve_budget_bytes(None) reads the ambient
    // env value internally, so any concurrent caller can observe another
    // test's tiny-budget mutation under cargo test's default parallelism
    // (found 2026-07-18: eval_basis_on_grid_serial_and_parallel_paths_agree
    // flaked with OutOfBudget from exactly this race).
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

    /// P10 regression guard: `eval_basis_on_grid`'s parallel grid-point path
    /// must be bit-identical across rayon thread counts. Water/cc-pVDZ on a
    /// 12×12×12 grid (1728 points) clears `PAR_GRID_POINT_THRESHOLD` (512),
    /// so the parallel scatter path is exercised in both dedicated pools.
    #[test]
    fn eval_basis_on_grid_bit_identical_across_thread_counts() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let xyz = "3\nwater\nO 0.000000 0.000000 0.117790\nH 0.000000 0.755453 -0.471161\nH 0.000000 -0.755453 -0.471161\n";
        let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
        let bs = basis::bundled("cc-pvdz").unwrap();
        let grid = GridSpec {
            origin: [-3.0, -3.0, -3.0],
            n_x: 12,
            n_y: 12,
            n_z: 12, // 1728 pts, clears PAR_GRID_POINT_THRESHOLD (512)
            step_x: [0.5, 0.0, 0.0],
            step_y: [0.0, 0.5, 0.0],
            step_z: [0.0, 0.0, 0.5],
        };

        let run = |threads: usize| -> Array2<f64> {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
            pool.install(|| eval_basis_on_grid(&mol, &bs, &grid).unwrap())
        };
        let chi1 = run(1);
        let chi4 = run(4);
        assert_eq!(chi1.shape(), chi4.shape());
        for (idx, (v1, v4)) in chi1.iter().zip(chi4.iter()).enumerate() {
            assert_eq!(
                v1.to_bits(),
                v4.to_bits(),
                "grid element {idx} differs across thread counts: {v1:e} vs {v4:e}"
            );
        }
    }

    /// Companion to the bit-identity test above: confirms the *serial*
    /// below-threshold path and the *parallel* above-threshold path compute
    /// the same values, not just that the parallel path is internally
    /// consistent across thread counts. Uses two grids sharing the same
    /// origin/spacing — one below `PAR_GRID_POINT_THRESHOLD` (serial path),
    /// one above it (parallel path) — and checks the points common to both
    /// (matching (ix, iy, iz)) agree exactly.
    #[test]
    fn eval_basis_on_grid_serial_and_parallel_paths_agree() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let xyz = "3\nwater\nO 0.000000 0.000000 0.117790\nH 0.000000 0.755453 -0.471161\nH 0.000000 -0.755453 -0.471161\n";
        let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
        let bs = basis::bundled("cc-pvdz").unwrap();

        // Small grid: below PAR_GRID_POINT_THRESHOLD (512) -> serial path.
        let small = GridSpec {
            origin: [-1.0, -1.0, -1.0],
            n_x: 5,
            n_y: 5,
            n_z: 5, // 125 pts < 512
            step_x: [0.5, 0.0, 0.0],
            step_y: [0.0, 0.5, 0.0],
            step_z: [0.0, 0.0, 0.5],
        };
        // Large grid sharing the same origin/spacing but extended so the
        // first 5x5x5 block of points is identical to `small`: above
        // PAR_GRID_POINT_THRESHOLD (512) -> parallel path.
        let large = GridSpec {
            origin: [-1.0, -1.0, -1.0],
            n_x: 5,
            n_y: 5,
            n_z: 25, // 625 pts >= 512; first 5 z-points per (x,y) match `small`
            step_x: [0.5, 0.0, 0.0],
            step_y: [0.0, 0.5, 0.0],
            step_z: [0.0, 0.0, 0.5],
        };

        let chi_small = eval_basis_on_grid(&mol, &bs, &small).unwrap();
        let chi_large = eval_basis_on_grid(&mol, &bs, &large).unwrap();

        for ix in 0..small.n_x {
            for iy in 0..small.n_y {
                for iz in 0..small.n_z {
                    let g_small = grid_index(&small, ix, iy, iz);
                    let g_large = grid_index(&large, ix, iy, iz);
                    for row in 0..chi_small.nrows() {
                        assert_eq!(
                            chi_small[(row, g_small)].to_bits(),
                            chi_large[(row, g_large)].to_bits(),
                            "row {row} at (ix={ix},iy={iy},iz={iz}): serial vs parallel mismatch"
                        );
                    }
                }
            }
        }
    }
}
