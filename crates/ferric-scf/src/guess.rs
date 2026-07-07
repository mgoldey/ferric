//! Initial density matrix guess for the SCF procedure.

use ferric_core::FerricError;
use ndarray::Array2;
use ndarray_linalg::Eigh;

/// Generate an initial density matrix from the core Hamiltonian eigenvectors.
///
/// Diagonalizes H in the canonically-orthogonalized basis and occupies the
/// lowest `nocc` orbitals to form D = 2 * C_occ * C_occ^T. Uses the same
/// linear-dependence filtering as the SCF loop's orthogonalizer, so a
/// near-singular overlap (e.g. aug bases on clustered atoms) is dropped from
/// the guess instead of seeding the SCF with an Inf/NaN density.
pub fn hcore_guess(
    s: &Array2<f64>,
    h: &Array2<f64>,
    nocc: usize,
) -> Result<Array2<f64>, FerricError> {
    let x = crate::rhf::canonical_orthogonalizer(s)?; // (n, m), m ≤ n
    let m = x.ncols();
    if nocc > m {
        return Err(FerricError::General(format!(
            "nocc = {nocc} exceeds the orthogonalized basis dimension {m} (nbasis = {}) — check charge and basis set",
            s.nrows()
        )));
    }
    // H' = Xᵀ H X
    let h_prime = x.t().dot(h).dot(&x);
    let (_, c_prime) = h_prime
        .eigh(ndarray_linalg::UPLO::Upper)
        .map_err(|e| FerricError::Lapack(format!("H' diag: {e}")))?;
    // C = X C'
    let c = x.dot(&c_prime);
    // D = 2 C_occ C_occ^T
    let c_occ = c.slice(ndarray::s![.., ..nocc]);
    let d = c_occ.dot(&c_occ.t()) * 2.0;
    Ok(d)
}

/// Ground-state multiplicity for a neutral free atom (Z = 1..118).
///
/// Matches the proatom_gs_mult closure in ferric-cli/src/main.rs so the
/// SAD free-atom solves are consistent with the Hirshfeld proatom path.
fn atom_ground_state_mult(z: i32) -> usize {
    match z {
        // Doublets (²S or ²P): H, Li, B, F, Na, Al, Cl, Ga, Br, I, ...
        1 | 3 | 5 | 9 | 11 | 13 | 17 | 31 | 35 | 53 => 2,
        // Triplets (³P): C, O, Si, S, Ge, Se, Te
        6 | 8 | 14 | 16 | 32 | 34 | 52 => 3,
        // Quartets (⁴S): N, P, As, Sb
        7 | 15 | 33 | 51 => 4,
        // Everything else (closed-shell or handled by UHF): singlet / even electron
        _ => 1,
    }
}

/// Superposition-of-Atomic-Densities (SAD) initial density matrix guess.
///
/// For each unique element in the molecule this function runs a free-atom
/// SCF (UHF for open-shell multiplicities, RHF for singlets) in the molecular
/// basis restricted to that atom's shells. The converged atomic density blocks
/// are placed on the block-diagonal of the full molecular density matrix; all
/// off-diagonal atom-atom blocks remain zero (the standard SAD guess). This
/// gives a physically sane starting point that is far superior to the bare
/// hcore guess for heavy-atom closed-shell systems (Br, Se, …) where the
/// hcore density is so poor that DIIS diverges.
///
/// # Arguments
/// * `mol`  – the molecular geometry (atoms' Z and positions)
/// * `prep` – the full molecular prepared basis (provides atom→basis mapping)
/// * `bs_name` – name of the orbital basis set (looked up via `ferric_core::basis::bundled`)
///
/// # Returns
/// A block-diagonal density matrix D of shape `(nbasis, nbasis)` where each
/// atom's block is the converged atomic spin-summed density in its own shells.
/// `tr(D · S) ≈ nelec` to within SCF convergence.
pub fn sad_guess(
    mol: &ferric_core::mol::Molecule,
    prep: &ferric_integrals::basis_bridge::PreparedBasis,
    bs_name: &str,
) -> Result<Array2<f64>, FerricError> {
    use ferric_core::basis;
    use ferric_core::mol::Molecule;
    use ferric_integrals::basis_bridge::PreparedBasis;
    use ferric_core::parallel::ParallelContext;
    use ferric_integrals::operator::Operator;
    use crate::screening::SchwarzBounds;
    use crate::rhf::{RhfConfig, solve_rhf};
    use crate::uhf::solve_uhf;
    use std::collections::HashMap;

    let n = prep.nbasis();
    let mut d = Array2::<f64>::zeros((n, n));

    // Compute per-atom AO offsets and sizes from the full molecular basis.
    // shell_to_atom[sh] = atom index; shell_offsets[sh] = start AO of shell sh.
    let shell_to_atom = prep.shell_to_atom();
    let shell_offsets = prep.shell_offsets();
    let shell_dims = prep.shell_dims();
    let natoms = mol.atoms.len();

    // Build per-atom AO offset and count: atom_ao_start[a], atom_ao_count[a].
    let mut atom_ao_start = vec![0usize; natoms];
    let mut atom_ao_count = vec![0usize; natoms];
    for sh in 0..prep.nshells() {
        let a = shell_to_atom[sh];
        if atom_ao_count[a] == 0 {
            atom_ao_start[a] = shell_offsets[sh];
        }
        atom_ao_count[a] += shell_dims[sh];
    }

    // Cache atomic density blocks by Z to avoid recomputing the same element twice.
    // Key = Z; Value = atomic density matrix (nao_atom × nao_atom).
    let mut atom_density_cache: HashMap<i32, Array2<f64>> = HashMap::new();

    let bs = basis::bundled(bs_name)?;
    let ctx = ParallelContext::default();
    let op = Operator::coulomb();

    for (ai, atom) in mol.atoms.iter().enumerate() {
        if atom.ghost { continue; }
        let z = atom.z;
        let nao = atom_ao_count[ai];
        if nao == 0 { continue; }

        // Build or retrieve the atomic density block for this Z.
        let atom_d = if let Some(cached) = atom_density_cache.get(&z) {
            cached.clone()
        } else {
            // Build a single-atom Molecule at the origin (SCF is translationally invariant).
            let sym = ferric_core::elements::z_to_symbol(z)
                .ok_or_else(|| FerricError::General(format!("unknown Z={z}")))?;
            let axyz = format!("1\n{sym}\n{sym} 0 0 0\n");
            let mult = atom_ground_state_mult(z);
            let amol = Molecule::parse_xyz(&axyz, 0, mult)?;
            let aprep = PreparedBasis::new(&amol, &bs)?;
            let abounds = SchwarzBounds::compute(op, &aprep)?;

            // Run the free-atom SCF on a serial 1-thread rayon pool to avoid the
            // 18× rayon overhead on single-atom systems (see rayon-penalty-on-free-atom-scf
            // memory). Serial pool ensures the atom solve doesn't starve the caller.
            let atom_d_full: Array2<f64> = run_serial_pool(|| {
                if mult == 1 {
                    // Closed-shell atom: use RHF.
                    let acfg = RhfConfig {
                        max_iter: 200,
                        energy_conv: 1e-8,
                        density_conv: 1e-6,
                        ..Default::default()
                    };
                    solve_rhf(&ctx, &amol, &aprep, op, &abounds, &acfg)
                        .map(|r| r.density_total().to_owned())
                } else {
                    // Open-shell atom: use UHF + MOM to pin the occupation.
                    let acfg = RhfConfig {
                        max_iter: 200,
                        energy_conv: 1e-8,
                        density_conv: 1e-6,
                        mom_after_iter: 5,
                        ..Default::default()
                    };
                    solve_uhf(&ctx, &amol, &aprep, &abounds, &acfg)
                        .map(|r| r.density_total().to_owned())
                }
            })
            .map_err(|e| FerricError::General(format!(
                "SAD free-atom SCF failed for {sym} (Z={z}): {e:?}"
            )))?;

            let d_block = atom_d_full.slice(ndarray::s![..nao, ..nao]).to_owned();
            atom_density_cache.insert(z, d_block.clone());
            d_block
        };

        // Place the atomic density block on the block-diagonal of D.
        let off = atom_ao_start[ai];
        let blk_n = atom_ao_count[ai];
        if atom_d.nrows() != blk_n || atom_d.ncols() != blk_n {
            return Err(FerricError::General(format!(
                "SAD: atom {ai} (Z={z}) atomic density shape {:?} ≠ expected ({blk_n},{blk_n})",
                atom_d.dim()
            )));
        }
        let mut block = d.slice_mut(ndarray::s![off..off + blk_n, off..off + blk_n]);
        block.assign(&atom_d);
    }

    Ok(d)
}

/// Run a closure on a freshly spawned 1-thread rayon pool.
///
/// Free-atom SCFs are tiny (~10-30 basis functions) but the global rayon pool's
/// work-stealing overhead makes them 18× slower than single-threaded execution.
/// This matches the `run_serial` pattern in ferric-cli/src/main.rs.
fn run_serial_pool<F, R>(f: F) -> R
where
    F: FnOnce() -> R + Send,
    R: Send,
{
    match rayon::ThreadPoolBuilder::new().num_threads(1).build() {
        Ok(pool) => pool.install(f),
        Err(_) => f(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferric_core::basis;
    use ferric_core::mol::Molecule;
    use ferric_integrals::basis_bridge::PreparedBasis;
    use ferric_integrals::oneelectron;

    #[test]
    fn hcore_guess_near_singular_overlap_is_finite() {
        // Overlap with an exactly-zero eigenvalue (perfect linear dependence).
        // The guess must drop the singular mode like the SCF loop's canonical
        // orthogonalizer — not seed the SCF with an Inf/NaN density.
        let s = ndarray::array![[1.0, 1.0], [1.0, 1.0]];
        let h = ndarray::array![[-1.0, -0.5], [-0.5, -1.0]];
        let d = hcore_guess(&s, &h, 1).unwrap();
        assert!(
            d.iter().all(|v| v.is_finite()),
            "guess density has non-finite entries: {d:?}"
        );
    }

    #[test]
    fn hcore_guess_nocc_exceeding_basis_is_an_error() {
        // More occupied orbitals than basis functions (e.g. a charge typo in
        // the input) must be a clean Err, not a slice panic.
        let s = ndarray::array![[1.0, 0.0], [0.0, 1.0]];
        let h = ndarray::array![[-1.0, 0.0], [0.0, -1.0]];
        let res = hcore_guess(&s, &h, 3);
        assert!(res.is_err(), "nocc > nbasis must be an error, got {res:?}");
    }

    #[test]
    fn test_hcore_guess_water_trace() {
        let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
        let bs = basis::bundled("sto-3g").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let s = oneelectron::overlap(&prep);
        let h = oneelectron::hcore(&prep);
        let d = hcore_guess(&s, &h, 5).unwrap();
        let n = prep.nbasis();
        // D symmetric
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (d[(i, j)] - d[(j, i)]).abs() < 1e-10,
                    "D not symmetric at ({i},{j})"
                );
            }
        }
        // tr(DS) = nelec = 10
        let tr: f64 = (0..n)
            .map(|i| (0..n).map(|j| d[(i, j)] * s[(i, j)]).sum::<f64>())
            .sum();
        assert!((tr - 10.0).abs() < 1e-6, "tr(DS) = {tr}, expected 10");
    }

    /// SAD guess for water/STO-3G: verify the density is symmetric and
    /// tr(DS) ≈ 10 (the correct electron count).
    #[test]
    fn sad_guess_water_sto3g_trace() {
        let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
        let bs = basis::bundled("sto-3g").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let s = oneelectron::overlap(&prep);
        let d = sad_guess(&mol, &prep, "sto-3g").unwrap();
        let n = prep.nbasis();
        // D symmetric
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (d[(i, j)] - d[(j, i)]).abs() < 1e-10,
                    "SAD D not symmetric at ({i},{j})"
                );
            }
        }
        // tr(DS) ≈ nelec = 10. SAD is exact to SCF convergence.
        let tr: f64 = (0..n)
            .map(|i| (0..n).map(|j| d[(i, j)] * s[(i, j)]).sum::<f64>())
            .sum();
        assert!((tr - 10.0).abs() < 0.1, "SAD tr(DS) = {tr}, expected ≈10");
    }

    /// SAD guess + RHF on C2H3Br / aug-cc-pVDZ with NO level shift must converge
    /// to the PySCF reference energy −2649.796875 within 1e-5 Ha.
    ///
    /// Without a SAD guess, ferric's hcore initial guess diverges on this system
    /// unless a level_shift ≥ 1.0 Ha is applied (57 iterations). SAD provides a
    /// physically sane starting point that converges with zero level shift.
    ///
    /// Marked #[ignore] because it takes ~1-2 min in release mode.
    #[test]
    #[ignore]
    fn rhf_sad_guess_c2h3br_aug_cc_pvdz_no_shift() {
        use crate::rhf::{RhfConfig, solve_rhf};
        use crate::screening::SchwarzBounds;
        use ferric_core::parallel::ParallelContext;
        use ferric_integrals::operator::Operator;

        let xyz = "6\nC2H3Br\nC 0.000000 0.000000 0.000000\nC 0.000000 0.000000 1.325600\nH -0.895976 0.000000 -0.602298\nH -0.894897 0.000000 1.927173\nH 0.908386 0.000000 -0.581003\nBr 1.357668 0.000000 2.194533\n";
        let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
        let bs = basis::bundled("aug-cc-pvdz").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let ctx = ParallelContext::default();

        // Build SAD initial density
        let d_sad = sad_guess(&mol, &prep, "aug-cc-pvdz").expect("SAD guess must succeed");

        // Run RHF starting from the SAD density, NO level shift
        let config = RhfConfig {
            max_iter: 200,
            init_guess_density: Some(d_sad),
            level_shift: 0.0,
            ..Default::default()
        };
        let result = solve_rhf(&ctx, &mol, &prep, op, &bounds, &config)
            .expect("RHF with SAD guess must converge");

        eprintln!(
            "C2H3Br SAD/aug-cc-pVDZ: E={:.10}, iters={}",
            result.energy, result.iterations
        );
        assert!(result.converged, "RHF with SAD guess did not converge");
        assert!(
            (result.energy - (-2649.796875)).abs() < 1e-5,
            "C2H3Br SAD RHF: got {:.10}, expected ≈-2649.796875",
            result.energy
        );
    }
}
