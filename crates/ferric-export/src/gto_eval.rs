//! Re-exports from `ferric_integrals::ao_grid`.

pub use ferric_integrals::ao_grid::{
    eval_basis_on_grid, eval_basis_on_points, grid_index, nbasis, GtoEvalError, GridSpec,
};

#[cfg(test)]
mod tests {
    use super::*;
    use ferric_core::basis;
    use ferric_core::mol::Molecule;

    fn h2_mol() -> Molecule {
        Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 0.7408\n", 0, 1).unwrap()
    }

    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn h2_sto3g_s_function_normalization() {
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
        assert_eq!(chi.nrows(), 2);
        assert_eq!(chi.ncols(), 1);
        assert!(chi[(0, 0)] > 0.0);
        assert!(chi[(1, 0)] > 0.0);
        assert!(chi[(0, 0)] > chi[(1, 0)]);
    }

    #[test]
    fn eval_basis_on_grid_fails_fast_under_tiny_env_budget() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mol = h2_mol();
        let bs = basis::bundled("sto-3g").unwrap();
        let grid = GridSpec {
            origin: [0.0, 0.0, 0.0],
            n_x: 50, n_y: 50, n_z: 50,
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

    #[test]
    fn eval_basis_on_grid_bit_identical_across_thread_counts() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let xyz = "3\nwater\nO 0.000000 0.000000 0.117790\nH 0.000000 0.755453 -0.471161\nH 0.000000 -0.755453 -0.471161\n";
        let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
        let bs = basis::bundled("cc-pvdz").unwrap();
        let grid = GridSpec {
            origin: [-3.0, -3.0, -3.0],
            n_x: 12, n_y: 12, n_z: 12,
            step_x: [0.5, 0.0, 0.0],
            step_y: [0.0, 0.5, 0.0],
            step_z: [0.0, 0.0, 0.5],
        };

        let run = |threads: usize| -> ndarray::Array2<f64> {
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

    #[test]
    fn eval_basis_on_grid_serial_and_parallel_paths_agree() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let xyz = "3\nwater\nO 0.000000 0.000000 0.117790\nH 0.000000 0.755453 -0.471161\nH 0.000000 -0.755453 -0.471161\n";
        let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
        let bs = basis::bundled("cc-pvdz").unwrap();

        let small = GridSpec {
            origin: [-1.0, -1.0, -1.0],
            n_x: 5, n_y: 5, n_z: 5,
            step_x: [0.5, 0.0, 0.0],
            step_y: [0.0, 0.5, 0.0],
            step_z: [0.0, 0.0, 0.5],
        };
        let large = GridSpec {
            origin: [-1.0, -1.0, -1.0],
            n_x: 5, n_y: 5, n_z: 25,
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
