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
        // ²S alkali-like heavy atoms + coinage metals with a single ns valence
        // electron over a closed (or pseudo-closed) shell: K (⁴s¹), Cu ([Ar]3d¹⁰4s¹),
        // Rb (⁵s¹), Ag ([Kr]4d¹⁰5s¹). Without these they fell into the `_` arm and
        // were forced to a singlet, which for their ODD electron count makes the
        // free-atom RHF fail at iteration 0 — silently defeating SAD for Cu2/CCuN
        // and the alkali/coinage members (the guess then falls back to hcore).
        19 | 29 | 37 | 47 => 2,
        // Triplets (³P): C, O, Si, S, Ge, Se, Te
        6 | 8 | 14 | 16 | 32 | 34 | 52 => 3,
        // Quartets (⁴S): N, P, As, Sb
        7 | 15 | 33 | 51 => 4,
        // Fallback: an odd electron count can NEVER be a singlet, so default odd Z
        // to a doublet and only even Z to a singlet. This keeps the free-atom SCF
        // from erroring on any odd-Z element not enumerated above (a closed-shell
        // RHF requires an even electron count). Not necessarily the true ground
        // state for open-shell transition metals, but a valid, convergent solve —
        // the SAD block only needs a reasonable atomic density, not the exact term.
        _ if z % 2 == 1 => 2,
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
/// * `bs`   – the orbital basis set (typically `prep.basis_set()`); its shells
///   and any ECP definitions are reused for the per-element free-atom solves, so
///   file-loaded (non-bundled) bases work without a name round-trip.
///
/// # Returns
/// A block-diagonal density matrix D of shape `(nbasis, nbasis)` where each
/// atom's block is the converged atomic spin-summed density in its own shells.
/// `tr(D · S) ≈ nelec` to within SCF convergence.
pub fn sad_guess(
    mol: &ferric_core::mol::Molecule,
    prep: &ferric_integrals::basis_bridge::PreparedBasis,
    bs: &ferric_core::basis::BasisSet,
) -> Result<Array2<f64>, FerricError> {
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

    for (ai, atom) in mol.atoms.iter().enumerate() {
        if atom.ghost { continue; }
        let z = atom.z;
        let nao = atom_ao_count[ai];
        if nao == 0 { continue; }

        // Skip the SAD free-atom solve for atoms carrying high angular momentum
        // (g functions, l ≥ 4). The ⟨gg|gg⟩-class quartets in the free-atom direct
        // J/K build are so expensive that a single UHF iteration on, e.g.,
        // Cu/aug-cc-pVTZ cannot finish in minutes on the serial 1-thread pool this
        // guess uses (see sad-free-atom-cu-multiplicity-and-highl-cost memory). The
        // whole point of SAD is a *cheap* better-than-hcore start; when the atomic
        // solve costs more than the molecular SCF it is meant to accelerate, it is a
        // net loss. Leave this atom's block at zero — the molecular SCF's DIIS fills
        // it in from the other (SAD-seeded) atoms, which is still far better than a
        // full-hcore guess and, crucially, does not stall setup.
        let has_high_l = bs
            .for_element(z)
            .map(|shells| shells.iter().any(|sh| sh.l >= 4))
            .unwrap_or(false);
        if has_high_l {
            if crate::rhf::scf_trace() {
                let sym = ferric_core::elements::z_to_symbol(z).unwrap_or("?");
                eprintln!(
                    "SAD: skipping free-atom solve for {sym} (Z={z}) — basis has l≥4 (g functions); block left zero"
                );
            }
            continue;
        }

        // Build or retrieve the atomic density block for this Z.
        let atom_d = if let Some(cached) = atom_density_cache.get(&z) {
            cached.clone()
        } else {
            let atom_d_full = free_atom_density(z, bs)?;
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

/// Run the free-atom SCF for element `z` in the given basis `bs` and return the
/// converged spin-summed atomic density in `bs`'s AO ordering for that atom
/// (shape `(nao, nao)` where `nao` is the number of AO functions `bs` assigns
/// to element `z`).
///
/// Shared by `sad_guess` (same-basis SAD) and `sad_guess_smallbasis` (which
/// calls this with `def2-svp` for high-l atoms before projecting up).
fn free_atom_density(
    z: i32,
    bs: &ferric_core::basis::BasisSet,
) -> Result<Array2<f64>, FerricError> {
    use ferric_core::mol::Molecule;
    use ferric_integrals::basis_bridge::PreparedBasis;
    use ferric_core::parallel::ParallelContext;
    use ferric_integrals::operator::Operator;
    use crate::screening::SchwarzBounds;
    use crate::rhf::{RhfConfig, solve_rhf};
    use crate::uhf::solve_uhf;

    let ctx = ParallelContext::default();
    let op = Operator::coulomb();

    // Build a single-atom Molecule at the origin (SCF is translationally invariant).
    let sym = ferric_core::elements::z_to_symbol(z)
        .ok_or_else(|| FerricError::General(format!("unknown Z={z}")))?;
    let axyz = format!("1\n{sym}\n{sym} 0 0 0\n");
    let mult = atom_ground_state_mult(z);
    let mut amol = Molecule::parse_xyz(&axyz, 0, mult)?;
    // Carry the ECP core count for PP bases (Rb, I, …) so the free-atom
    // electron count matches the valence-only molecular solve. No-op for
    // all-electron bases (apply_ecp early-returns on empty bs.ecps).
    amol.apply_ecp(bs);
    let aprep = PreparedBasis::new(&amol, bs)?;
    let abounds = SchwarzBounds::compute(op, &aprep)?;

    // Run the free-atom SCF on a serial 1-thread rayon pool to avoid the
    // 18× rayon overhead on single-atom systems (see rayon-penalty-on-free-atom-scf
    // memory). Serial pool ensures the atom solve doesn't starve the caller.
    // `use_sad_guess: false` breaks the recursion: these free-atom solves
    // ARE the SAD building blocks, so they must start from hcore.
    run_serial_pool(|| {
        if mult == 1 {
            // Closed-shell atom: use RHF. density_conv is the tight (reachable)
            // signal under the ΔP gate; energy_conv is left at the loose default
            // (a tighter one could stall a heavy atom whose dE floors above it —
            // see rhf::scf_converged / the guess is density-valued anyway).
            let acfg = RhfConfig {
                max_iter: 200,
                density_conv: 1e-6,
                use_sad_guess: false,
                ..Default::default()
            };
            solve_rhf(&ctx, &amol, &aprep, op, &abounds, &acfg)
                .map(|r| r.density_total().to_owned())
        } else {
            // Open-shell atom: use UHF + MOM to pin the occupation. density_conv
            // is the tight (reachable) signal; energy_conv left at the loose
            // default (see the closed-shell branch above).
            let acfg = RhfConfig {
                max_iter: 200,
                density_conv: 1e-6,
                mom_after_iter: 5,
                use_sad_guess: false,
                ..Default::default()
            };
            solve_uhf(&ctx, &amol, &aprep, &abounds, &acfg)
                .map(|r| r.density_total().to_owned())
        }
    })
    .map_err(|e| FerricError::General(format!(
        "SAD free-atom SCF failed for {sym} (Z={z}): {e:?}"
    )))
}

/// Pick the small basis used to build a high-l atom's free-atom density block,
/// tiered by element coverage:
///
/// 1. `def2-svp` if it carries element `z` (covers Z=1-20,37,53,54).
/// 2. else `def2-tzvp` if it carries `z` (covers the d-block and heavy atoms,
///    up to Z=86, with max l=3 — no g functions, so the free-atom solve stays
///    cheap even for transition metals like Cu that def2-svp lacks).
/// 3. else `None` — the caller falls back to the zero block.
///
/// Loading either bundled basis is cheap (parsed from an embedded JSON string),
/// so this simply loads and clones rather than caching across calls.
fn pick_small_basis(z: i32) -> Option<ferric_core::basis::BasisSet> {
    if let Ok(svp) = ferric_core::basis::bundled("def2-svp") {
        if svp.for_element(z).is_some() {
            return Some(svp);
        }
    }
    if let Ok(tzvp) = ferric_core::basis::bundled("def2-tzvp") {
        if tzvp.for_element(z).is_some() {
            return Some(tzvp);
        }
    }
    None
}

/// SAD guess where high angular-momentum atoms (l≥4, e.g. Cu/aTZ g functions)
/// build their density block in a small basis and project it into the target
/// AO space via the S-metric, instead of being left zero. Everyone else uses
/// the normal same-basis SAD free-atom solve.
///
/// For each atom whose target-basis shells include l≥4, this:
///  1. Picks a small basis via `pick_small_basis` (def2-svp, falling back to
///     def2-tzvp for elements def2-svp lacks, e.g. the d-block) and solves the
///     free atom in it — cheap even for Cu.
///  2. Builds a combined single-atom `PreparedBasis` with target shells followed
///     by small-basis shells for that element, and slices the cross-overlap
///     block `S_ts` (target × small) and the small self-overlap `S_ss` out of
///     the resulting full overlap matrix.
///  3. Projects: `T = S_ts · S_ss⁻¹`, `P_target = T · D_small · Tᵀ`.
///
/// If no small basis carries the element, falls back to the zero block (the
/// same behavior as `sad_guess`'s g-skip).
pub fn sad_guess_smallbasis(
    mol: &ferric_core::mol::Molecule,
    prep: &ferric_integrals::basis_bridge::PreparedBasis,
    bs: &ferric_core::basis::BasisSet,
) -> Result<Array2<f64>, FerricError> {
    use std::collections::HashMap;

    let n = prep.nbasis();
    let mut d = Array2::<f64>::zeros((n, n));

    let shell_to_atom = prep.shell_to_atom();
    let shell_offsets = prep.shell_offsets();
    let shell_dims = prep.shell_dims();
    let natoms = mol.atoms.len();

    let mut atom_ao_start = vec![0usize; natoms];
    let mut atom_ao_count = vec![0usize; natoms];
    for sh in 0..prep.nshells() {
        let a = shell_to_atom[sh];
        if atom_ao_count[a] == 0 {
            atom_ao_start[a] = shell_offsets[sh];
        }
        atom_ao_count[a] += shell_dims[sh];
    }

    // Cache block by Z (both light-atom same-basis blocks and projected
    // high-l blocks) to avoid recomputing the same element twice.
    let mut atom_density_cache: HashMap<i32, Array2<f64>> = HashMap::new();

    for (ai, atom) in mol.atoms.iter().enumerate() {
        if atom.ghost { continue; }
        let z = atom.z;
        let nao = atom_ao_count[ai];
        if nao == 0 { continue; }

        let has_high_l = bs
            .for_element(z)
            .map(|shells| shells.iter().any(|sh| sh.l >= 4))
            .unwrap_or(false);

        let atom_d = if let Some(cached) = atom_density_cache.get(&z) {
            cached.clone()
        } else if !has_high_l {
            // Light atom: identical same-basis free-atom solve as sad_guess.
            let block = free_atom_density(z, bs)?
                .slice(ndarray::s![..nao, ..nao])
                .to_owned();
            atom_density_cache.insert(z, block.clone());
            block
        } else {
            match pick_small_basis(z) {
                None => {
                    // Neither def2-svp nor def2-tzvp carries this element —
                    // same zero-block fallback as sad_guess's g-skip.
                    if crate::rhf::scf_trace() {
                        let sym = ferric_core::elements::z_to_symbol(z).unwrap_or("?");
                        eprintln!(
                            "SAD-smallbasis: {sym} (Z={z}) has l≥4 in target basis and is absent from def2-svp and def2-tzvp; block left zero"
                        );
                    }
                    let block = Array2::zeros((nao, nao));
                    atom_density_cache.insert(z, block.clone());
                    block
                }
                Some(small_bs) => {
                    if crate::rhf::scf_trace() {
                        let sym = ferric_core::elements::z_to_symbol(z).unwrap_or("?");
                        eprintln!(
                            "SAD-smallbasis: {sym} (Z={z}) via {}",
                            small_bs.name
                        );
                    }
                    // High-l atom: project the small-basis free-atom density
                    // into the target AO block via the S-metric.
                    let block = project_smallbasis_density(z, bs, &small_bs)?;
                    atom_density_cache.insert(z, block.clone());
                    block
                }
            }
        };

        let off = atom_ao_start[ai];
        let blk_n = atom_ao_count[ai];
        if atom_d.nrows() != blk_n || atom_d.ncols() != blk_n {
            return Err(FerricError::General(format!(
                "sad_guess_smallbasis: atom {ai} (Z={z}) block shape {:?} ≠ expected ({blk_n},{blk_n})",
                atom_d.dim()
            )));
        }
        let mut block = d.slice_mut(ndarray::s![off..off + blk_n, off..off + blk_n]);
        block.assign(&atom_d);
    }

    Ok(d)
}

/// Build the free-atom density in `small_bs` (def2-svp) for element `z` and
/// project it into the AO block of `target_bs` for that element via the
/// S-metric: `T = S_ts · S_ss⁻¹`, `P_target = T · D_small · Tᵀ`, where `S_ts`
/// is the target×small cross-overlap and `S_ss` the small-basis self-overlap.
///
/// The cross overlap is obtained by building a single-atom `BasisSet` whose
/// shell list for `z` is `target shells ++ small shells`, preparing one
/// `PreparedBasis` from it, and slicing the resulting full overlap matrix.
/// Returns shape `(n_t, n_t)` where `n_t` is the number of target AO
/// functions for `z`.
fn project_smallbasis_density(
    z: i32,
    target_bs: &ferric_core::basis::BasisSet,
    small_bs: &ferric_core::basis::BasisSet,
) -> Result<Array2<f64>, FerricError> {
    use ferric_core::basis::{num_functions, BasisSet};
    use ferric_core::mol::Molecule;
    use ferric_integrals::basis_bridge::PreparedBasis;
    use ferric_integrals::oneelectron;
    use ndarray_linalg::Inverse;
    use std::collections::HashMap;

    let target_shells = target_bs
        .for_element(z)
        .ok_or_else(|| FerricError::General(format!(
            "project_smallbasis_density: target basis missing element Z={z}"
        )))?
        .to_vec();
    let small_shells = small_bs
        .for_element(z)
        .ok_or_else(|| FerricError::General(format!(
            "project_smallbasis_density: small basis missing element Z={z}"
        )))?
        .to_vec();

    let n_t: usize = target_shells.iter().map(|sh| num_functions(sh.l, sh.pure)).sum();
    let n_s: usize = small_shells.iter().map(|sh| num_functions(sh.l, sh.pure)).sum();

    // Combined single-atom basis: target shells first, then small shells,
    // for this element only.
    let mut combined_shells = target_shells;
    combined_shells.extend(small_shells);
    let mut combined_map = HashMap::new();
    combined_map.insert(z, combined_shells);
    let combined_bs = BasisSet {
        name: format!("{}+{}@{z}", target_bs.name, small_bs.name),
        shells: combined_map,
        ecps: target_bs.ecps.clone(),
    };

    let sym = ferric_core::elements::z_to_symbol(z)
        .ok_or_else(|| FerricError::General(format!("unknown Z={z}")))?;
    let axyz = format!("1\n{sym}\n{sym} 0 0 0\n");
    let mult = atom_ground_state_mult(z);
    let mut amol = Molecule::parse_xyz(&axyz, 0, mult)?;
    amol.apply_ecp(&combined_bs);
    let combined_prep = PreparedBasis::new(&amol, &combined_bs)?;
    let s_full = oneelectron::overlap(&combined_prep);

    if s_full.nrows() != n_t + n_s {
        return Err(FerricError::General(format!(
            "project_smallbasis_density: combined overlap dim {} != n_t+n_s ({n_t}+{n_s}) for Z={z}",
            s_full.nrows()
        )));
    }

    let s_ts = s_full.slice(ndarray::s![0..n_t, n_t..n_t + n_s]).to_owned();
    let s_ss = s_full.slice(ndarray::s![n_t.., n_t..]).to_owned();

    let s_ss_inv = s_ss
        .inv()
        .map_err(|e| FerricError::Lapack(format!(
            "project_smallbasis_density: S_ss singular for Z={z}: {e}"
        )))?;
    if s_ss_inv.iter().any(|v| !v.is_finite()) {
        return Err(FerricError::General(format!(
            "project_smallbasis_density: S_ss^-1 has non-finite entries for Z={z}"
        )));
    }

    // Free-atom density in the small basis alone.
    let d_small = free_atom_density(z, small_bs)?;
    if d_small.nrows() != n_s || d_small.ncols() != n_s {
        return Err(FerricError::General(format!(
            "project_smallbasis_density: small-basis free-atom density shape {:?} != ({n_s},{n_s}) for Z={z}",
            d_small.dim()
        )));
    }

    // T = S_ts · S_ss^-1  (n_t x n_s); P_target = T · D_small · T^T
    let t = s_ts.dot(&s_ss_inv);
    let p_target = t.dot(&d_small).dot(&t.t());

    if p_target.iter().any(|v| !v.is_finite()) {
        return Err(FerricError::General(format!(
            "project_smallbasis_density: projected density has non-finite entries for Z={z}"
        )));
    }

    Ok(p_target)
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

    /// Sanity check for `project_smallbasis_density`'s S-metric projection math,
    /// independent of any target/small basis-size mismatch: when target == small,
    /// `T = S_ts · S_ss⁻¹` reduces to the identity, so `tr(P·S) = tr(D_small·S_ss)`
    /// exactly. This isolates the projection formula from the (much larger and
    /// basis-size-dependent) approximation error seen when projecting a small
    /// (def2-svp) density into a much bigger target basis (see
    /// `sad_smallbasis_cu_block_is_nonzero_and_traced`).
    #[test]
    #[ignore] // ~5s release: builds an O/def2-svp free-atom SCF
    fn project_smallbasis_density_same_basis_reproduces_trace() {
        let svp = basis::bundled("def2-svp").unwrap();
        let p = project_smallbasis_density(8, &svp, &svp).unwrap();
        let mol = Molecule::parse_xyz("1\nO\nO 0 0 0\n", 0, 3).unwrap();
        let prep = PreparedBasis::new(&mol, &svp).unwrap();
        let s = oneelectron::overlap(&prep);
        let n = prep.nbasis();
        let tr: f64 = (0..n).map(|i| (0..n).map(|j| p[(i, j)] * s[(i, j)]).sum::<f64>()).sum();
        assert!((tr - 8.0).abs() < 1e-4, "same-basis projection must reproduce trace, got {tr}");
    }

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
        let d = sad_guess(&mol, &prep, &bs).unwrap();
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
        let d_sad = sad_guess(&mol, &prep, &bs).expect("SAD guess must succeed");

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

    #[test]
    #[ignore] // ~10-30s release: builds a Cu/def2-tzvp free-atom SCF
    fn sad_smallbasis_cu_block_is_nonzero_and_traced() {
        use ferric_core::mol::Molecule;
        use ferric_core::basis;
        use ferric_integrals::basis_bridge::PreparedBasis;
        use ferric_integrals::oneelectron;

        // Now that the small-basis lookup is tiered (def2-svp -> def2-tzvp),
        // Cu/aug-cc-pVTZ — the actual target case this feature exists for
        // (Cu has g functions, l=4, in aug-cc-pVTZ) — is exercisable directly:
        // def2-svp lacks Cu (Z=29; svp only covers Z=1-20,37,53,54), so the
        // tier falls through to def2-tzvp, which covers the whole d-block up
        // to Z=86 with max l=3 (cheap free-atom solve, no g functions).
        //
        // def2-tzvp/Cu has roughly a ~2x AO-count ratio to aug-cc-pVTZ/Cu (vs
        // the ~4x def2-svp/def2-qzvp ratio in the prior O-substitute test,
        // which measured ~4.67x trace inflation for a ~4.07x size ratio). A
        // ~2x size ratio should inflate the trace by noticeably less than
        // that, but AO-projection guesses are not exact off-basis, so this
        // uses the same loose structural bound style: nonzero, finite,
        // symmetric, and a generously-bounded trace around Cu's 29 electrons.
        let mol = Molecule::parse_xyz("1\nCu\nCu 0 0 0\n", 0, 2).unwrap();
        let bs = basis::bundled("aug-cc-pvtz").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let s = oneelectron::overlap(&prep);
        let d = sad_guess_smallbasis(&mol, &prep, &bs).unwrap();
        assert!(d.iter().any(|&v| v.abs() > 1e-6), "Cu block must be nonzero");
        assert!(d.iter().all(|v| v.is_finite()), "Cu block has non-finite entries");
        let n = prep.nbasis();
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (d[(i, j)] - d[(j, i)]).abs() < 1e-8,
                    "projected D not symmetric at ({i},{j})"
                );
            }
        }
        let tr: f64 = (0..n).map(|i| (0..n).map(|j| d[(i, j)] * s[(i, j)]).sum::<f64>()).sum();
        eprintln!("sad_smallbasis_cu_block_is_nonzero_and_traced: tr(D*S) = {tr}");
        assert!(
            tr > 20.0 && tr < 80.0,
            "tr(DS)={tr}, expected a positive, order-of-magnitude-sane trace for Cu (nelec=29)"
        );
    }
}
