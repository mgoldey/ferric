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

/// Number of electrons to occupy for a neutral free atom, split into the
/// "doubly-occupied core+valence pair count" plus the number of singly-occupied
/// (open-shell) spatial orbitals. `(n_doubly, n_singly)` where
/// `2*n_doubly + n_singly == z - n_core_ecp` (the ECP-adjusted valence count).
///
/// This is a *density-shape* helper, not a term-symbol solver: we only need a
/// physically-sane radial occupation to seed the SCF, so the open-shell electrons
/// are spread as `n_singly` singly-occupied orbitals via `atom_ground_state_mult`
/// (which already encodes the neutral-atom ground multiplicity). The remaining
/// electrons pair up. For a closed-shell atom `n_singly == 0`.
fn aufbau_occupation(nvalence: usize, mult: usize) -> (usize, usize) {
    // mult = 2S+1 ⇒ number of unpaired electrons = mult - 1.
    let n_unpaired = mult.saturating_sub(1).min(nvalence);
    let n_paired_elec = nvalence - n_unpaired;
    (n_paired_elec / 2, n_unpaired)
}

/// Build a free-atom density block for element `z` in basis `bs` **without any
/// SCF** — the no-per-element-SCF core of the MINAO-style guess.
///
/// Builds the single-atom effective Fock via the Generalized Wolfsberg–Helmholtz
/// (GWH) recipe from the *atomic* core Hamiltonian `H = T + V` (single atom at the
/// origin, ECP-adjusted point charge and, when present, the ECP projector) —
/// diagonal `H_ii`, off-diagonals `F_ij = ½·1.75·S_ij·(H_ii+H_jj)` — diagonalizes
/// it in the canonically-orthogonalized atomic basis, and fills the lowest
/// orbitals by aufbau to the neutral-atom valence electron count:
///   * the lowest `n_doubly` spatial orbitals doubly (occupation 2),
///   * the open-shell electrons spread EQUALLY over the (near-)degenerate frontier,
/// returning `D = Σ_i f_i c_i c_iᵀ` (spin-summed), shape `(nao, nao)`.
///
/// A single-atom hcore eigensolve is O(nao³) on a ~15-40-function block — a few
/// microseconds — versus the seconds-to-minutes of a full free-atom UHF/MOM SCF
/// (and with none of its wrong-spin-basin risk). The nuclear-attraction well of
/// a *single* atom, filled by aufbau, gives the correct radial ordering and node
/// structure; unlike the *molecular* hcore guess (whose multi-center potential
/// over-contracts heavy atoms and diverges), the per-atom aufbau density is a
/// sound MINAO source. The exchange-correlation refinement is left to the
/// molecular SCF this density seeds.
fn atomic_hcore_density(
    z: i32,
    bs: &ferric_core::basis::BasisSet,
) -> Result<Array2<f64>, FerricError> {
    use ferric_core::mol::Molecule;
    use ferric_integrals::basis_bridge::PreparedBasis;
    use ferric_integrals::oneelectron;

    let sym = ferric_core::elements::z_to_symbol(z)
        .ok_or_else(|| FerricError::General(format!("unknown Z={z}")))?;
    let axyz = format!("1\n{sym}\n{sym} 0 0 0\n");
    let mult = atom_ground_state_mult(z);
    // Neutral atom (charge 0). apply_ecp both trims the point charge to the
    // valence Z and records the core count so the valence electron count matches
    // the molecular (ECP) solve. No-op for all-electron bases.
    let mut amol = Molecule::parse_xyz(&axyz, 0, mult)?;
    amol.apply_ecp(bs);
    let aprep = PreparedBasis::new(&amol, bs)?;

    let s = oneelectron::overlap(&aprep);
    // ECP-aware core Hamiltonian: adds V_ECP when bs carries ECPs, else plain T+V.
    let h = oneelectron::hcore_ecp(&aprep, &amol, bs);

    // Valence electron count after ECP: nelec() already accounts for the ECP core.
    let nvalence = usize::try_from(amol.nelec()).map_err(|_| {
        FerricError::General(format!("atomic_hcore_density: negative nelec for Z={z}"))
    })?;
    let (n_doubly, n_singly) = aufbau_occupation(nvalence, mult);

    // Build the effective one-electron matrix to diagonalize. Bare atomic hcore
    // over-contracts the orbitals (no electron–electron repulsion), which seeds a
    // density that DIIS can limit-cycle around at ultra-tight thresholds
    // (water/STO-3G/PBE stalled at density_conv=1e-8). The Generalized
    // Wolfsberg–Helmholtz (GWH) recipe gives a markedly better SCF-free seed at
    // essentially no cost: keep the diagonal H_ii but replace the off-diagonals
    // with F_ij = ½·K·S_ij·(H_ii + H_jj), K = 1.75 (Pople's standard value). This
    // is the same guess many production codes expose as "gwh" and converges
    // water/STO-3G/PBE cleanly to 1e-8 in a handful of iterations.
    const GWH_K: f64 = 1.75;
    let nao_h = h.nrows();
    let mut f_eff = Array2::<f64>::zeros((nao_h, nao_h));
    for i in 0..nao_h {
        f_eff[(i, i)] = h[(i, i)];
        for j in (i + 1)..nao_h {
            let fij = 0.5 * GWH_K * s[(i, j)] * (h[(i, i)] + h[(j, j)]);
            f_eff[(i, j)] = fij;
            f_eff[(j, i)] = fij;
        }
    }

    // Diagonalize F_eff in the canonically-orthogonalized basis (lindep-filtered,
    // same path as hcore_guess), so a near-singular atomic overlap can't seed a
    // non-finite density.
    let x = crate::rhf::canonical_orthogonalizer(&s)?; // (n, m), m ≤ n
    let m = x.ncols();
    let n_occ_orb = n_doubly + n_singly;
    if n_occ_orb > m {
        return Err(FerricError::General(format!(
            "atomic_hcore_density: {sym} (Z={z}) needs {n_occ_orb} occupied orbitals but orthogonalized atomic basis has only {m}"
        )));
    }
    let h_prime = x.t().dot(&f_eff).dot(&x);
    let (eps, c_prime) = h_prime
        .eigh(ndarray_linalg::UPLO::Upper)
        .map_err(|e| FerricError::Lapack(format!("atomic H' diag for Z={z}: {e}")))?;
    let c = x.dot(&c_prime); // (nao, m), columns ascending in energy

    // Per-orbital occupation numbers (spin-summed): the lowest `n_doubly`
    // orbitals get 2.0. The `n_singly` open-shell electrons are spread EQUALLY
    // over all orbitals (near-)degenerate with the frontier, not dumped onto an
    // arbitrary subset — a bare-hcore atomic problem has exactly-degenerate p/d
    // shells, so picking specific SOMOs would break the atom's spherical
    // symmetry and seed a lopsided molecular density that DIIS then has to
    // laboriously un-tilt (observed: water/STO-3G PBE limit-cycled to MaxIter
    // when O's 2p SOMOs were placed on two of the three 2p orbitals). Equal
    // fractional occupation over the degenerate frontier restores a spherical
    // atomic density and converges cleanly.
    let mut occ = vec![0.0_f64; m];
    for o in occ.iter_mut().take(n_doubly) {
        *o = 2.0;
    }
    if n_singly > 0 {
        // Frontier group = all orbitals within EPS_DEGEN of the first partially
        // occupied level eps[n_doubly], starting from n_doubly. Spread the
        // n_singly electrons equally over that whole group.
        const EPS_DEGEN: f64 = 1e-4;
        let e_front = eps[n_doubly];
        let mut group_end = n_doubly;
        while group_end < m && (eps[group_end] - e_front).abs() <= EPS_DEGEN {
            group_end += 1;
        }
        let group = group_end - n_doubly;
        let frac = n_singly as f64 / group as f64; // ≤ 1.0
        for o in occ.iter_mut().take(group_end).skip(n_doubly) {
            *o = frac;
        }
    }

    let nao = s.nrows();
    let mut d = Array2::<f64>::zeros((nao, nao));
    // D = Σ_i occ_i · c_i c_iᵀ (spin-summed AO density).
    for (i, &f) in occ.iter().enumerate() {
        if f == 0.0 { continue; }
        let ci = c.slice(ndarray::s![.., i]);
        // rank-1 update scaled by the occupation number
        let outer = ci.to_owned().insert_axis(ndarray::Axis(1));
        d = d + outer.dot(&outer.t()) * f;
    }
    if d.iter().any(|v| !v.is_finite()) {
        return Err(FerricError::General(format!(
            "atomic_hcore_density: non-finite atomic density for Z={z}"
        )));
    }
    Ok(d)
}

/// Default SCF initial-density guess: a superposition of per-atom densities,
/// built so that NO expensive per-element SCF ever runs.
///
/// The atomic block for each element is built by one of two routes, chosen by the
/// element so the ~20-minute free-atom-SCF pathology can never occur:
///
///  * **Heavy atoms** — transition metals and heavier (`Z ≥ 21`) or any element
///    whose target shells include high angular momentum (`l ≥ 4`, g functions):
///    a **no-SCF** GWH atomic block from [`atomic_hcore_density`] built directly
///    in the target basis (one small eigensolve; ONE-electron integrals only, so
///    even Cu/aug-cc-pVTZ g functions cost nothing). This is exactly the class the
///    guess exists to fix: a free-atom SCF on Cu/aug-cc-pVDZ (no g, but a heavy
///    d-block UHF) took minutes and could land in the wrong spin basin; GWH sidesteps
///    both. Cu2/aug-cc-pVDZ RKS-PBE from this guess converges in ~26 iters to the
///    physical −3280.641 Ha / HOMO −0.174 Ha / 1.70 eV-gap state (matches PySCF).
///
///  * **Light main-group atoms** (`Z ≤ 20`, no g functions): a proper free-atom
///    SCF block via [`free_atom_density`]. These solves are milliseconds (they were
///    never the pathology), and their self-consistent blocks are a higher-quality
///    seed than GWH — enough that tight-gate light-molecule DFT SCFs (methane/water
///    cc-pVDZ PBE at energy_conv 1e-10) still converge cleanly, matching the prior
///    SAD default.
///
/// Off-diagonal atom-atom blocks stay zero (standard SAD/MINAO structure); the
/// block diagonal is trace-exact (`tr(D·S) == nelec`). If a per-element build
/// fails, that atom's block is left zero and the molecular SCF fills it in from
/// the other (seeded) atoms — still far better than a bare molecular-hcore guess.
pub fn minao_projection_guess(
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

    // Collect the unique non-ghost (Z, nao) pairs needing a cache entry, in
    // first-encountered order (BTreeMap/HashMap iteration order is not
    // guaranteed, so we track order explicitly for determinism — though the
    // final result does not depend on cache build order at all: each entry
    // is an independent per-Z build, and placement below is a separate,
    // strictly serial, ascending-atom-index loop, so this is bit-identical
    // regardless of which order the unique Zs are computed in).
    let mut seen: HashMap<i32, usize> = HashMap::new(); // z -> nao
    let mut unique_zs: Vec<i32> = Vec::new();
    for (ai, atom) in mol.atoms.iter().enumerate() {
        if atom.ghost { continue; }
        let nao = atom_ao_count[ai];
        if nao == 0 { continue; }
        seen.entry(atom.z).or_insert_with(|| {
            unique_zs.push(atom.z);
            nao
        });
    }

    // Build each unique Z's target-AO density block. Density blocks are
    // per-Z independent (each is a self-contained free-atom SCF or no-SCF
    // GWH build, never touching another element's data), so this is safe to
    // parallelize across elements — unlike the per-Z SCF ITSELF, which stays
    // on `run_serial_pool`'s single-thread rayon pool internally (existing
    // behavior, unchanged: see free_atom_density/atomic_hcore_density). The
    // final per-atom placement loop below is untouched and still strictly
    // serial in ascending atom order, so the result is bit-identical to the
    // old fully-serial cache-build — only the WALL-CLOCK order of the
    // independent per-Z builds changes, not the math or the placement.
    let build_one = |z: i32| -> Array2<f64> {
        let nao = seen[&z];
        // Choose the no-SCF (GWH) route for the atoms whose free-atom SCF is
        // the pathology this guess exists to eliminate — transition metals and
        // heavier (Z ≥ 21), or any element carrying g (l ≥ 4) shells in the
        // target basis. Light main-group atoms (Z ≤ 20, no g) keep the cheap,
        // higher-quality free-atom SCF block.
        let has_high_l = bs
            .for_element(z)
            .map(|shells| shells.iter().any(|sh| sh.l >= 4))
            .unwrap_or(false);
        let use_no_scf = z >= 21 || has_high_l;

        let built = if use_no_scf {
            // No-SCF GWH block directly in the target basis (g-safe: 1e ints only).
            atomic_hcore_density(z, bs)
        } else {
            // Cheap, high-quality light-atom free-atom SCF block.
            free_atom_density(z, bs)
        };

        match built {
            Ok(full) => full.slice(ndarray::s![..nao, ..nao]).to_owned(),
            Err(e) => {
                if crate::rhf::scf_trace() {
                    let sym = ferric_core::elements::z_to_symbol(z).unwrap_or("?");
                    let route = if use_no_scf { "GWH no-SCF" } else { "free-atom SCF" };
                    eprintln!("MINAO: {route} atomic density failed for {sym} (Z={z}): {e:?}; block left zero");
                }
                Array2::zeros((nao, nao))
            }
        }
    };

    // Guard: skip the parallel machinery entirely for the (very common)
    // single-unique-Z case — nothing to parallelize, and it avoids spinning
    // up rayon's fan-out for a single item.
    let atom_density_cache: HashMap<i32, Array2<f64>> = if unique_zs.len() <= 1 {
        unique_zs.iter().map(|&z| (z, build_one(z))).collect()
    } else {
        use rayon::prelude::*;
        unique_zs
            .par_iter()
            .map(|&z| (z, build_one(z)))
            .collect::<Vec<_>>()
            .into_iter()
            .collect()
    };

    for (ai, atom) in mol.atoms.iter().enumerate() {
        if atom.ghost { continue; }
        let z = atom.z;
        let nao = atom_ao_count[ai];
        if nao == 0 { continue; }

        let atom_d = &atom_density_cache[&z];

        let off = atom_ao_start[ai];
        let blk_n = atom_ao_count[ai];
        if atom_d.nrows() != blk_n || atom_d.ncols() != blk_n {
            return Err(FerricError::General(format!(
                "minao_projection_guess: atom {ai} (Z={z}) block shape {:?} ≠ expected ({blk_n},{blk_n})",
                atom_d.dim()
            )));
        }
        let mut block = d.slice_mut(ndarray::s![off..off + blk_n, off..off + blk_n]);
        block.assign(atom_d);
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

    /// MINAO no-SCF guess for water/STO-3G: density must be symmetric and
    /// tr(DS) ≈ 10. The atomic-hcore aufbau density is not exact (no XC), so the
    /// trace is looser than the SCF-based SAD test, but must be in the right
    /// ballpark (correct electron count to within projection/hcore error).
    #[test]
    fn minao_guess_water_sto3g_trace() {
        let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
        let bs = basis::bundled("sto-3g").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let s = oneelectron::overlap(&prep);
        let d = minao_projection_guess(&mol, &prep, &bs).unwrap();
        let n = prep.nbasis();
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (d[(i, j)] - d[(j, i)]).abs() < 1e-10,
                    "MINAO D not symmetric at ({i},{j})"
                );
            }
        }
        assert!(d.iter().all(|v| v.is_finite()), "MINAO D has non-finite entries");
        let tr: f64 = (0..n)
            .map(|i| (0..n).map(|j| d[(i, j)] * s[(i, j)]).sum::<f64>())
            .sum();
        // Same-basis aufbau-hcore density is trace-exact by construction
        // (tr(2 C_occ C_occᵀ S) = 2·n_occ for orthonormal C), so this is tight.
        assert!((tr - 10.0).abs() < 1e-6, "MINAO tr(DS) = {tr}, expected 10");
    }

    /// The no-SCF atomic-hcore density for a heavy transition metal (Cu, Z=29)
    /// in aug-cc-pVDZ must be finite, symmetric, and trace-exact (29 electrons)
    /// — with NO free-atom SCF. This is the element whose free-atom SCF was the
    /// ~minutes-and-wrong-spin pathology the MINAO guess exists to eliminate.
    #[test]
    fn atomic_hcore_density_cu_aug_cc_pvdz_no_scf() {
        let bs = basis::bundled("aug-cc-pvdz").unwrap();
        // Cu in aug-cc-pVDZ is all-electron, maxL=3 (no g), so the density is
        // built directly in this basis (no small-basis projection).
        let d = atomic_hcore_density(29, &bs).unwrap();
        assert!(d.iter().all(|v| v.is_finite()), "Cu atomic density non-finite");
        let n = d.nrows();
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (d[(i, j)] - d[(j, i)]).abs() < 1e-8,
                    "Cu atomic D not symmetric at ({i},{j})"
                );
            }
        }
        // Trace against the atomic overlap must equal Cu's 29 electrons.
        let mol = Molecule::parse_xyz("1\nCu\nCu 0 0 0\n", 0, 2).unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let s = oneelectron::overlap(&prep);
        let tr: f64 = (0..n).map(|i| (0..n).map(|j| d[(i, j)] * s[(i, j)]).sum::<f64>()).sum();
        assert!((tr - 29.0).abs() < 1e-6, "Cu atomic tr(DS) = {tr}, expected 29");
    }

    /// MINAO for Cu/aug-cc-pVTZ (Cu has g functions, l=4). Because the guess
    /// uses only ONE-electron integrals it builds the density directly in the
    /// g-containing target basis — no small-basis projection — so it stays
    /// trace-exact (29 electrons), nonzero, finite, symmetric, with NO free-atom SCF.
    #[test]
    fn minao_cu_aug_cc_pvtz_no_scf() {
        let mol = Molecule::parse_xyz("1\nCu\nCu 0 0 0\n", 0, 2).unwrap();
        let bs = basis::bundled("aug-cc-pvtz").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let s = oneelectron::overlap(&prep);
        let d = minao_projection_guess(&mol, &prep, &bs).unwrap();
        assert!(d.iter().any(|&v| v.abs() > 1e-6), "Cu MINAO block must be nonzero");
        assert!(d.iter().all(|v| v.is_finite()), "Cu MINAO block non-finite");
        let n = prep.nbasis();
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (d[(i, j)] - d[(j, i)]).abs() < 1e-8,
                    "MINAO D not symmetric at ({i},{j})"
                );
            }
        }
        let tr: f64 = (0..n).map(|i| (0..n).map(|j| d[(i, j)] * s[(i, j)]).sum::<f64>()).sum();
        eprintln!("minao_cu_aug_cc_pvtz_no_scf: tr(D*S) = {tr}");
        assert!((tr - 29.0).abs() < 1e-6, "Cu MINAO tr(DS)={tr}, expected 29 (trace-exact)");
    }

    /// End-to-end: Cu2 / aug-cc-pVDZ RKS-PBE from the default (MINAO) guess must
    /// converge FAST to the physical state (E≈-3280.641 Ha, HOMO ε≈-0.174 Ha,
    /// gap≈1.70 eV per PySCF), with no per-element free-atom SCF in setup.
    /// Ignored by default (minutes in release even when fast).
    #[test]
    #[ignore]
    fn minao_cu2_pbe_converges_fast_and_physical() {
        // The DFT grid path stack-allocates large scratch; the cargo-test worker
        // thread's default 2 MiB stack overflows, so run the body on a thread with
        // a generous stack (the CLI already runs SCF on such a main thread).
        std::thread::Builder::new()
            .stack_size(256 * 1024 * 1024)
            .spawn(cu2_pbe_minao_body)
            .unwrap()
            .join()
            .unwrap();
    }

    fn cu2_pbe_minao_body() {
        use crate::rhf::{RhfConfig, solve_rhf};
        use crate::screening::SchwarzBounds;
        use ferric_core::parallel::ParallelContext;
        use ferric_integrals::operator::Operator;
        use ferric_dft::grid::AtomicGridConfig;

        // Cu2 at 2.2197 Å (task geometry).
        let xyz = "2\nCu2\nCu 0.0 0.0 0.0\nCu 0.0 0.0 2.2197\n";
        let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
        let bs = basis::bundled("aug-cc-pvdz").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let ctx = ParallelContext::default();

        let t0 = std::time::Instant::now();
        let cfg = RhfConfig {
            max_iter: 200,
            xc: Some("PBE".to_string()),
            dft_grid: Some(AtomicGridConfig { n_radial: 75, n_angular: 110, ..Default::default() }),
            ..Default::default() // use_sad_guess defaults true ⇒ MINAO
        };
        let r = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg).expect("Cu2 RKS-PBE");
        let dt = t0.elapsed().as_secs_f64();
        let eps = r.eps_r();
        let nocc = (mol.nelec() / 2) as usize;
        let homo = eps[nocc - 1];
        let lumo = eps[nocc];
        let gap_ev = (lumo - homo) * 27.211386;
        eprintln!(
            "Cu2 MINAO/PBE: E={:.6} iters={} conv={} HOMO={:.6} gap={:.4} eV wall={:.1}s",
            r.energy, r.iterations, r.converged, homo, gap_ev, dt
        );
        assert!(r.converged, "Cu2 RKS-PBE did not converge from MINAO guess");
        assert!((r.energy - (-3280.641)).abs() < 0.05, "Cu2 E={:.6}, expected ≈-3280.641", r.energy);
    }

    /// MINAO must converge a light closed-shell DFT case (water/STO-3G/PBE) at the
    /// production density_conv (1e-6) to the PySCF energy, from the default guess.
    /// Water's O/H are light (Z ≤ 20) so this exercises the free-atom-SCF block
    /// route of the hybrid guess end-to-end. Ignored (~10s release: a DF-PBE SCF).
    #[test]
    #[ignore]
    fn minao_water_pbe_converges() {
        use crate::rhf::{RhfConfig, solve_rhf};
        use crate::screening::SchwarzBounds;
        use ferric_core::parallel::ParallelContext;
        use ferric_integrals::operator::Operator;
        std::thread::Builder::new().stack_size(256 * 1024 * 1024).spawn(|| {
            let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
            let bs = basis::bundled("sto-3g").unwrap();
            let prep = PreparedBasis::new(&mol, &bs).unwrap();
            let op = Operator::coulomb();
            let bounds = SchwarzBounds::compute(op, &prep).unwrap();
            let ctx = ParallelContext::default();
            let cfg = RhfConfig {
                xc: Some("PBE".into()),
                df_j_aux: Some("def2-universal-jkfit".into()),
                density_conv: 1e-6,
                max_iter: 200,
                ..Default::default() // MINAO (use_sad_guess defaults true)
            };
            let r = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg).unwrap();
            eprintln!("water MINAO/PBE: E={:.10} iters={} conv={}", r.energy, r.iterations, r.converged);
            assert!(r.converged, "MINAO water/STO-3G PBE did not converge (exit={:?})", r.exit);
        }).unwrap().join().unwrap();
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
