//! Density/charge/ESP molecular properties derived purely from an
//! [`ScfResult`](crate::result::ScfResult) — electrostatic potential,
//! electric field, effective volumes, and atomic partial-charge schemes
//! (Becke, Löwdin, Mulliken, CHELPG, RESP).
//!
//! These routines were moved here from `ferric-rpa::properties` (2026-07)
//! because none of them actually depend on RPA (PDEP eigenpairs, Lanczos,
//! or the screened dielectric): each one only needs one-electron/grid
//! integrals over an SCF density, which `ferric-scf` (this crate) already
//! has access to via its existing `ferric-core`/`ferric-integrals`/
//! `ferric-dft`/`ferric-pcm` dependencies. Living in the lower crate lets
//! any future non-RPA consumer (e.g. a plain-SCF CLI path) use them without
//! pulling in `ferric-rpa`. `ferric-rpa::properties` re-exports the public
//! functions below unchanged, so existing call sites
//! (`ferric_rpa::properties::hirshfeld_charges` etc.) are unaffected.
//!
//! A handful of small helpers here (`debug_toggle`, `positive_f64`,
//! `hirshfeld_spacing`, `hirshfeld_margin`, `slater_xi_for_z`, `eig3_sym`)
//! are `pub` rather than private: they are also used by RPA-dependent
//! sibling functions that legitimately stay in `ferric-rpa::properties`
//! (e.g. `pdep_polarizability_hirshfeld`, `pdep_polarizability_static`), so
//! `ferric-rpa` calls back into these via `ferric_scf::properties::*`
//! instead of duplicating them.
//!
//! Three RPA-independent Hirshfeld routines — `atomic_effective_volumes_hirshfeld`,
//! `hirshfeld_i_charges` and `hirshfeld_charges` — are defined in
//! `ferric-rpa::properties`, not here. The first two integrate on the uniform
//! Cartesian lattice of `ferric_integrals::ao_grid::GridSpec`; `hirshfeld_charges`
//! integrates on the atom-centred Becke–Lebedev grid of `ferric_dft::grid`. The
//! proatom machinery they consume —
//! [`RadialProatom`](crate::properties::RadialProatom),
//! [`ProatomProvider`](crate::properties::ProatomProvider),
//! [`spherically_averaged_proatom`](crate::properties::spherically_averaged_proatom),
//! [`slater_xi_for_z`](crate::properties::slater_xi_for_z) and the free-atom SCF
//! provider [`scf_proatom_provider`](crate::properties::scf_proatom_provider) —
//! lives here.

use std::os::raw::c_int;

use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::engine::Engine;
use ferric_integrals::ffi::{self, CAtom};
use ferric_integrals::oneelectron;
use ndarray::Array2;

/// Debug-print toggles for the property/Hirshfeld/α diagnostics (env-only).
/// NOTE behavior change: all three were previously read via `.is_ok()` (any
/// value, incl. `=0`, enabled them); now `=0`/`false`/`off` disable them, via
/// the shared [`ferric_core::config::parse_toggle`].
pub fn debug_toggle(env_name: &'static str) -> bool {
    let var = ferric_core::config::ConfigVar::<bool> {
        env_name,
        default: false,
        parse: ferric_core::config::parse_toggle,
        validate: ferric_core::config::accept_any,
    };
    var.toggle()
}

/// Hirshfeld/α bounding-box grid knobs (Bohr), read at 5 sites here with one
/// shared default each (verified identical: spacing 0.20, margin 6.0).
/// These are result-affecting, so the value is VALIDATED (finite > 0); a
/// malformed override logs a warning and uses the default rather than aborting a
/// deep property calc (2 of the 5 call sites don't return Result, so a hard Err
/// can't propagate uniformly without a signature change — deferred).
pub fn positive_f64(env_name: &'static str, default: f64) -> f64 {
    let var = ferric_core::config::ConfigVar::<f64> {
        env_name,
        default,
        parse: |s| s.parse::<f64>().map_err(|e| e.to_string()),
        validate: |v| {
            (v.is_finite() && *v > 0.0)
                .then_some(())
                .ok_or_else(|| "must be finite > 0".to_string())
        },
    };
    var.get().map(|r| r.value).unwrap_or_else(|e| {
        eprintln!("[config] {env_name}: {e}; using default {default}");
        default
    })
}

/// Grid spacing (Bohr) for the Hirshfeld/α bounding-box grid. `FERRIC_HIRSHFELD_SPACING`.
pub fn hirshfeld_spacing() -> f64 {
    positive_f64("FERRIC_HIRSHFELD_SPACING", 0.20)
}

/// Bounding-box margin (Bohr) covering the diffuse α tail. `FERRIC_HIRSHFELD_MARGIN`.
pub fn hirshfeld_margin() -> f64 {
    positive_f64("FERRIC_HIRSHFELD_MARGIN", 6.0)
}

/// Evaluate the electrostatic potential V(R_A) at each nuclear position,
/// excluding the divergent self-interaction Z_A/|R_A − R_A|.
///
/// ```text
///     V(R_A) = Σ_{B ≠ A} Z_B / |R_A − R_B|
///            − Σ_{μν} D_{μν} ⟨μ| 1/|r − R_A| |ν⟩
/// ```
///
/// Returns a `Vec<f64>` of length `mol.atoms.len()`, in Hartree (a.u.).
///
/// # Implementation note
///
/// The electronic term reuses the nuclear-attraction engine — we override
/// the point-charge list to a single Z=1 charge at R_A so libint integrates
/// `⟨μ| −1/|r−R_A| |ν⟩`.  Flipping sign gives the matrix M^A; contracting
/// with D and adding the nuclear sum gives V.
pub fn esp_at_atoms(
    mol: &Molecule,
    prep: &PreparedBasis,
    density: &Array2<f64>,
) -> Result<Vec<f64>, FerricError> {
    use ferric_integrals::blas_threads::with_blas_threads;
    use rayon::prelude::*;

    let natoms = mol.atoms.len();
    let nbas = prep.nbasis();
    if density.shape() != [nbas, nbas] {
        return Err(FerricError::General(format!(
            "esp_at_atoms: density shape {:?} != ({nbas},{nbas})",
            density.shape()
        )));
    }

    let dims = prep.shell_dims();
    let offs = prep.shell_offsets();
    let nsh = prep.nshells();

    // Each atom is an independent probe: a fresh point-charge list is
    // written into the engine, then contracted with D. The engine is
    // stateful (Send, not Sync) so each rayon worker gets its own via
    // map_init instead of sharing/cloning per element. BLAS is pinned to 1
    // inside the rayon region per repo convention (no GEMM here, but keeps
    // the discipline uniform with the other P5 sites).
    let out: Vec<f64> = with_blas_threads(1, || {
        (0..natoms)
            .into_par_iter()
            .map_init(
                || Engine::new_1e(ffi::OP_NUCLEAR, prep, 1e-14),
                |eng, a| -> Result<f64, FerricError> {
                    let eng = eng.as_mut().map_err(|e| {
                        FerricError::General(format!("esp_at_atoms: engine init failed: {e}"))
                    })?;
                    let atom_a = &mol.atoms[a];

                    // Override engine params with a single Z=1 charge at R_A.
                    // libint's nuclear-attraction operator returns
                    //   ⟨μ| −Z / |r − R| |ν⟩
                    // so for Z=1 we get −⟨μ| 1/|r−R_A| |ν⟩ in the engine output.
                    let probe = [CAtom {
                        atomic_number: 1.0,
                        x: atom_a.x,
                        y: atom_a.y,
                        z: atom_a.zpos,
                    }];
                    // SAFETY: probe is a stack-local CAtom slice; handle_mut() is the live engine
                    // pointer; probe.len() fits in c_int. Shim catches C++ exceptions → negative rc.
                    let rc = unsafe {
                        ffi::scf_engine_set_point_charges(
                            eng.handle_mut(),
                            probe.as_ptr(),
                            probe.len() as c_int,
                        )
                    };
                    if rc < 0 {
                        return Err(FerricError::General(format!(
                            "esp_at_atoms: set_point_charges failed (rc={rc}) for atom {a}"
                        )));
                    }

                    // Build M^A_μν = ⟨μ| −1/|r−R_A| |ν⟩ from libint with Z=+1.
                    //
                    //   V_elec(R_A) = − ∫ ρ(r) / |r − R_A| dr
                    //               = − Σ_{μν} D_{μν} T_{μν}      with T_{μν} = ⟨μ|1/r|ν⟩
                    //               = + Σ_{μν} D_{μν} M^A_{μν}    since M^A = −T.
                    //
                    // So summing density · block directly (full square, no symmetry
                    // collapse) yields V_elec.
                    // Iterate upper-triangle shell pairs and contribute both (μν) and
                    // (νμ) by symmetry: the operator and density are symmetric so the
                    // two contributions are equal.
                    let mut v_elec = 0.0_f64;
                    for s1 in 0..nsh {
                        for s2 in 0..=s1 {
                            let block = eng.compute_1e_block(prep, s1, s2);
                            let n1 = dims[s1];
                            let n2 = dims[s2];
                            let o1 = offs[s1];
                            let o2 = offs[s2];
                            if s1 == s2 {
                                for i in 0..n1 {
                                    for j in 0..n2 {
                                        // Full block entries, no symmetry collapse needed.
                                        v_elec += density[(o1 + i, o2 + j)] * block[i * n2 + j];
                                    }
                                }
                            } else {
                                // Off-diagonal shell pair: block covers (s1,s2); add
                                // 2× since (s2,s1) is the symmetric partner.
                                for i in 0..n1 {
                                    for j in 0..n2 {
                                        v_elec +=
                                            2.0 * density[(o1 + i, o2 + j)] * block[i * n2 + j];
                                    }
                                }
                            }
                        }
                    }

                    // Nuclear sum: Σ_{B ≠ A} Z_B / |R_A − R_B|
                    let mut v_nuc = 0.0_f64;
                    for b in 0..natoms {
                        if b == a {
                            continue;
                        }
                        let atom_b = &mol.atoms[b];
                        let dx = atom_a.x - atom_b.x;
                        let dy = atom_a.y - atom_b.y;
                        let dz = atom_a.zpos - atom_b.zpos;
                        let r = (dx * dx + dy * dy + dz * dz).sqrt();
                        if r < 1e-12 {
                            return Err(FerricError::General(format!(
                                "esp_at_atoms: atoms {a} and {b} coincide"
                            )));
                        }
                        v_nuc += atom_b.z as f64 / r;
                    }

                    Ok(v_nuc + v_elec)
                },
            )
            .collect::<Result<Vec<f64>, FerricError>>()
    })?;

    Ok(out)
}

/// Evaluate the electric field **E**(R_A) = −∇V(R_A) at each nuclear
/// position.  Returns one `[f64; 3]` per atom, in atomic units (Hartree / Bohr).
///
/// ```text
///     E_d(R_A) = E^elec_d(R_A) + E^nuc_d(R_A)
///     E^elec_d(R_A) = + Σ_{μν} D_{μν} ⟨μ| (r − R_A)_d / |r − R_A|³ |ν⟩
///     E^nuc_d (R_A) = Σ_{B ≠ A} Z_B (R_A − R_B)_d / |R_A − R_B|³
/// ```
///
/// Sign convention matches [`esp_at_atoms`]: V is the electrostatic potential
/// felt by a unit positive test charge, and **E** = −∇V.
///
/// # Derivation of the electronic sign
///
/// ```text
///     V_elec(R) = − ∫ ρ(r) / |r − R| dr             (electrons are negative)
///     E_elec(R) = −∇V_elec(R) = + ∫ ρ(r) (r − R)/|r − R|³ dr
/// ```
///
/// so for a closed-shell AO density `D_{μν}`:
///
/// ```text
///     E^elec_d(R_A) = + Σ_{μν} D_{μν} ⟨μ|(r − R_A)_d/|r − R_A|³|ν⟩.
/// ```
///
/// # Implementation (Path A)
///
/// Re-uses libint2's first-derivative nuclear-attraction engine.  For each
/// atom A we set a single point charge of Z=+1 at R_A, then the derivative
/// engine returns 6 + 3·N_charges = 9 derivative blocks per shell pair:
///
///  - blocks 0..6: derivatives w.r.t. the two shell centers (Pulay terms,
///    irrelevant here — we only want the charge-center derivative);
///  - blocks 6,7,8: d/dR_A_{x,y,z} of `M^A_{μν} = ⟨μ| −1/|r − R_A| |ν⟩`
///    (libint's nuclear operator is −Z/|r − R|, with Z=+1 here).
///
/// With ∇_{R}(1/|r − R|) = (r − R)/|r − R|³,
///
/// ```text
///     dM^A/dR_A_d  =  − ⟨μ|(r − R_A)_d/|r − R_A|³|ν⟩
///     ⇒ ⟨μ|(r−R_A)_d/|r−R_A|³|ν⟩  =  − dM^A/dR_A_d.
/// ```
///
/// Therefore the electronic contribution is **minus** the contraction of D
/// with the libint charge-center derivative:
///
/// ```text
///     E^elec_d  =  + Σ D_{μν} ⟨μ|(r−R_A)_d/|r−R_A|³|ν⟩
///               =  − Σ D_{μν} · (dM^A/dR_A_d).
/// ```
pub fn electric_field_at_atoms(
    mol: &Molecule,
    prep: &PreparedBasis,
    density: &Array2<f64>,
) -> Result<Vec<[f64; 3]>, FerricError> {
    use ferric_integrals::blas_threads::with_blas_threads;
    use rayon::prelude::*;

    let natoms = mol.atoms.len();
    let nbas = prep.nbasis();
    if density.shape() != [nbas, nbas] {
        return Err(FerricError::General(format!(
            "electric_field_at_atoms: density shape {:?} != ({nbas},{nbas})",
            density.shape()
        )));
    }

    let dims = prep.shell_dims();
    let offs = prep.shell_offsets();
    let nsh = prep.nshells();
    let max_fn = dims.iter().copied().max().unwrap_or(1);

    // With a single point charge, libint returns 6 (shell) + 3 (charge) = 9
    // derivative blocks of size n1*n2 each.
    let nderiv = 6 + 3; // 6 shell-center + 3 (xyz) charge-center derivatives
    let max_block = max_fn * max_fn;

    // Each atom probe needs its own stateful derivative engine plus its own
    // hand-sized raw-FFI scratch buffer (reliability convention: 1e-deriv
    // sizing is per-caller, not per-engine). map_init hands each rayon
    // worker exactly one (engine, buf) pair, reused across the atoms that
    // worker processes — never cloned per atom, never shared across workers.
    let out: Vec<[f64; 3]> = with_blas_threads(1, || {
        (0..natoms)
            .into_par_iter()
            .map_init(
                || {
                    let eng = Engine::new_1e_deriv(ffi::OP_NUCLEAR, prep, 1e-14);
                    let buf = vec![0.0_f64; nderiv * max_block];
                    (eng, buf)
                },
                |(eng, buf), a| -> Result<[f64; 3], FerricError> {
                    let eng = eng.as_mut().map_err(|e| {
                        FerricError::General(format!(
                            "electric_field_at_atoms: engine init failed: {e}"
                        ))
                    })?;
                    let atom_a = &mol.atoms[a];

                    // Override point charges: single Z=+1 probe at R_A.
                    let probe = [CAtom {
                        atomic_number: 1.0,
                        x: atom_a.x,
                        y: atom_a.y,
                        z: atom_a.zpos,
                    }];
                    // SAFETY: probe is a stack-local CAtom slice; handle_mut() is the live engine
                    // pointer; probe.len() fits in c_int. Shim catches C++ exceptions → negative rc.
                    let rc = unsafe {
                        ffi::scf_engine_set_point_charges(
                            eng.handle_mut(),
                            probe.as_ptr(),
                            probe.len() as c_int,
                        )
                    };
                    if rc < 0 {
                        return Err(FerricError::General(format!(
                            "electric_field_at_atoms: set_point_charges failed (rc={rc}) for atom {a}"
                        )));
                    }

                    // Contract D with charge-center derivative blocks (indices 6,7,8).
                    let mut e_elec = [0.0_f64; 3];
                    for s1 in 0..nsh {
                        for s2 in 0..=s1 {
                            let n1 = dims[s1];
                            let n2 = dims[s2];
                            let block_sz = n1 * n2;
                            let total = nderiv * block_sz;
                            if buf.len() < total {
                                buf.resize(total, 0.0);
                            }
                            // SAFETY: buf is pre-sized to nderiv * block_sz; handle_mut()/handle()
                            // are live pointers; shell indices are in range. Shim returns written >= 0.
                            let written = unsafe {
                                ffi::scf_compute_1e_deriv_block(
                                    eng.handle_mut(),
                                    prep.handle(),
                                    s1 as c_int,
                                    s2 as c_int,
                                    buf.as_mut_ptr(),
                                )
                            };
                            assert!(written >= 0, "libint2 internal error in nuclear deriv block ({s1},{s2}): status {written}");
                            if written == 0 {
                                continue;
                            }
                            let o1 = offs[s1];
                            let o2 = offs[s2];
                            // Charge-center derivative blocks start at index 6.
                            // dM^A/dR_A_d = -<μ|(r-R_A)_d/|r-R_A|³|ν>
                            // E^elec_d = +Σ D * <μ|(r-R_A)_d/|r-R_A|³|ν> = -Σ D * dM^A/dR_A_d
                            for d in 0..3 {
                                let blk_off = (6 + d) * block_sz;
                                let mut acc = 0.0_f64;
                                if s1 == s2 {
                                    for i in 0..n1 {
                                        for j in 0..n2 {
                                            acc += density[(o1 + i, o2 + j)]
                                                * buf[blk_off + i * n2 + j];
                                        }
                                    }
                                } else {
                                    // Off-diagonal shell pair: density and operator are
                                    // both symmetric in (μ,ν), so the (s2,s1) partner
                                    // contributes equally → factor 2.
                                    for i in 0..n1 {
                                        for j in 0..n2 {
                                            acc += 2.0
                                                * density[(o1 + i, o2 + j)]
                                                * buf[blk_off + i * n2 + j];
                                        }
                                    }
                                }
                                e_elec[d] -= acc;
                            }
                        }
                    }

                    // Nuclear contribution: Σ_{B≠A} Z_B (R_A − R_B)_d / |R_A − R_B|³
                    let mut e_nuc = [0.0_f64; 3];
                    for b in 0..natoms {
                        if b == a {
                            continue;
                        }
                        let atom_b = &mol.atoms[b];
                        let dx = atom_a.x - atom_b.x;
                        let dy = atom_a.y - atom_b.y;
                        let dz = atom_a.zpos - atom_b.zpos;
                        let r2 = dx * dx + dy * dy + dz * dz;
                        let r = r2.sqrt();
                        if r < 1e-12 {
                            return Err(FerricError::General(format!(
                                "electric_field_at_atoms: atoms {a} and {b} coincide"
                            )));
                        }
                        let inv_r3 = 1.0 / (r2 * r);
                        let zb = atom_b.z as f64;
                        e_nuc[0] += zb * dx * inv_r3;
                        e_nuc[1] += zb * dy * inv_r3;
                        e_nuc[2] += zb * dz * inv_r3;
                    }

                    Ok([
                        e_elec[0] + e_nuc[0],
                        e_elec[1] + e_nuc[1],
                        e_elec[2] + e_nuc[2],
                    ])
                },
            )
            .collect::<Result<Vec<[f64; 3]>, FerricError>>()
    })?;

    Ok(out)
}

/// Per-atom effective volume via Becke partitioning:
/// ```text
///   v_A = ∫ w^A_Becke(r) ρ(r) |r − R_A|³ dr
/// ```
/// Returned in a.u. (Bohr³·e). The TS volume *ratio* is `v_A / v_free[Z_A]`,
/// where `v_free` is computed by running this same integral on a live
/// free-atom SCF density (ferric-cli's TS-C6 branch), NOT read from a table:
/// `ferric_rpa::dispersion::free_atom_ref::ts_free_atom`'s `vol_free` is `None`
/// for every Z (no sourced hardcoded free-atom volume — see that module's
/// doc and docs/vol-free-verification.md). The live-SCF free-atom volume is
/// the only denominator on a scale consistent with `v_A`.
pub fn atomic_effective_volumes_becke(
    mol: &Molecule,
    _prep: &PreparedBasis,
    obs_bs: &ferric_core::basis::BasisSet,
    density: &Array2<f64>,
) -> Result<Vec<f64>, FerricError> {
    atomic_effective_volumes_becke_chunked(mol, _prep, obs_bs, density, None)
}

/// [`atomic_effective_volumes_becke`] with an explicit grid chunk width.
///
/// `None` uses the production width. A test hook: chunking here is meant to be
/// numerically INERT, and the only honest way to assert that is to run TWO
/// widths on the SAME machine and compare them to each other.
///
/// The first version of `mwe_scf_grid_chunking_is_inert.rs` instead pinned
/// hardcoded reference values captured locally, and CI failed on them by
/// 3.3e-7 — because CI forces `OPENBLAS_CORETYPE=Haswell` while the dev box
/// uses its native kernels, so the SCF converges to a slightly different
/// density and the CHARGES differ before chunking is even reached. That test
/// was pinning a machine-dependent SCF result, not the chunking property.
#[doc(hidden)]
pub fn atomic_effective_volumes_becke_chunked(
    mol: &Molecule,
    _prep: &PreparedBasis,
    obs_bs: &ferric_core::basis::BasisSet,
    density: &Array2<f64>,
    chunk_override: Option<usize>,
) -> Result<Vec<f64>, FerricError> {
    use ferric_dft::ao_grid::eval_basis_on_points;
    use ferric_dft::grid::{build_atomic_grid, AtomicGridConfig};

    let natoms = mol.atoms.len();
    let grid_cfg = AtomicGridConfig::default();
    let grid = build_atomic_grid(mol, &grid_cfg);
    let points: Vec<[f64; 3]> = grid.iter().map(|g| g.xyz).collect();
    let weights: Vec<f64> = grid.iter().map(|g| g.weight).collect();
    let home_atom: Vec<usize> = grid.iter().map(|g| g.home_atom).collect();
    let npts = points.len();

    let pos: Vec<[f64; 3]> = mol.atoms.iter().map(|at| [at.x, at.y, at.zpos]).collect();

    // Evaluate chi in CHUNKS of grid points instead of materialising the whole
    // (nbf, npts) matrix up front.
    //
    // The full matrix is 3.4 GB at danuglipron/def2-SVP scale (73 atoms, the
    // default 75x110 grid => 602,250 points, nbf ~ 700) and 7.7 GB at
    // def2-TZVP, none of it bounded by `[memory] budget_gb` — this function
    // consults no budget, and `eval_basis_on_points` is not one of the sites
    // `check_ao_grid_budget` guards. That is the allocation shape behind the
    // 2026-07-13 incidents where ferric-cli reached 16-17 GB anon-RSS on the
    // Becke-grid property path. Every consumer below reads chi one grid point
    // at a time, so nothing needed the whole thing resident.
    //
    // Numerically INERT, and the reason is worth stating because "chunking is
    // obviously safe" is how block-boundary regressions get in here (see
    // `DRESS_ROW_BLOCK`'s doc for ~7e-15 from an odd GEMM row split):
    //   * chi is a materialised lookup table, not a reduction — chi[mu, g] is a
    //     pure function of point g's coordinates, so grouping points changes
    //     nothing about the values.
    //   * The one real reduction, `vol[a] += ...`, still runs over g in strict
    //     ascending order: the chunk loop is serial and outer, the point loop
    //     serial and inner, so the addition sequence is exactly what it was.
    // `mwe_scf_grid_chunking_is_inert.rs` pins this bit-for-bit.
    let chunk = chunk_override
        .unwrap_or_else(|| crate::reduce::deterministic_group_size(npts))
        .max(1);
    let mut vol = vec![0.0_f64; natoms];
    let mut g0 = 0usize;
    while g0 < npts {
        let g1 = (g0 + chunk).min(npts);
        let chi = eval_basis_on_points(mol, obs_bs, &points[g0..g1]).map_err(|e| {
            FerricError::General(format!(
                "atomic_effective_volumes_becke: chi eval failed: {e}"
            ))
        })?;
        let nbf = chi.nrows();
        for g in g0..g1 {
            let gc = g - g0; // column index within this chunk
            let a = home_atom[g];
            let mut rho = 0.0;
            for mu in 0..nbf {
                let cm = chi[(mu, gc)];
                if cm.abs() < 1e-30 {
                    continue;
                }
                for nu in 0..nbf {
                    rho += density[(mu, nu)] * cm * chi[(nu, gc)];
                }
            }
            let dx = points[g][0] - pos[a][0];
            let dy = points[g][1] - pos[a][1];
            let dz = points[g][2] - pos[a][2];
            let r3 = (dx * dx + dy * dy + dz * dz).powf(1.5);
            vol[a] += weights[g] * rho * r3;
        }
        g0 = g1;
    }
    Ok(vol)
}

/// Becke atomic charges via fuzzy partitioning of the molecular density.
///
/// `q_A = Z_A − ∫ w^A_Becke(r) ρ(r) dV` evaluated on the Becke-Lebedev
/// grid. Becke partition is geometry-only (no proatom density model),
/// fixing the C-O charge-inversion bug of single-exp Slater Hirshfeld
/// (memory [[lowdin-over-single-exp-hirshfeld]]).
///
/// Sum-rule renormalization: rescales so `Σ_A (Z_A − q_A) = N_e` exactly
/// (compensates ~0.003 e grid quadrature noise on H2O).
///
/// Closed-shell: pass `density = D_total` (= 2·D_α in restricted).
/// Open-shell: pass `D_α + D_β`.
pub fn becke_charges(
    mol: &Molecule,
    _prep: &PreparedBasis,
    obs_bs: &ferric_core::basis::BasisSet,
    density: &Array2<f64>,
) -> Result<Vec<f64>, FerricError> {
    becke_charges_chunked(mol, _prep, obs_bs, density, None)
}

/// [`becke_charges`] with an explicit grid chunk width. See
/// [`atomic_effective_volumes_becke_chunked`] for why this hook exists.
#[doc(hidden)]
pub fn becke_charges_chunked(
    mol: &Molecule,
    _prep: &PreparedBasis,
    obs_bs: &ferric_core::basis::BasisSet,
    density: &Array2<f64>,
    chunk_override: Option<usize>,
) -> Result<Vec<f64>, FerricError> {
    use ferric_dft::ao_grid::eval_basis_on_points;
    use ferric_dft::grid::{build_atomic_grid, AtomicGridConfig};

    let natoms = mol.atoms.len();
    let grid = build_atomic_grid(mol, &AtomicGridConfig::default());
    let points: Vec<[f64; 3]> = grid.iter().map(|g| g.xyz).collect();
    let weights: Vec<f64> = grid.iter().map(|g| g.weight).collect();
    let home_atom: Vec<usize> = grid.iter().map(|g| g.home_atom).collect();
    let npts = points.len();

    let nbf = density.nrows();
    if density.ncols() != nbf {
        return Err(FerricError::General(format!(
            "becke_charges: density shape {:?} is not square",
            density.dim()
        )));
    }

    // Chunk the grid instead of materialising the full (nbf, npts) chi AND its
    // same-shaped product `D·chi` — two co-resident matrices, so this path's
    // peak was TWICE the 3.4 GB / 7.7 GB figures quoted on
    // `atomic_effective_volumes_becke` above (6.7 GB at danuglipron/def2-SVP,
    // 15.4 GB at def2-TZVP), with no budget bounding either.
    //
    // Numerically INERT, and this site needs the argument spelled out because
    // it contains a GEMM, which is exactly where block boundaries have bitten
    // this repo before:
    //   * `d_chi = D.dot(chi)` reduces over nbf (the SHARED index); g is a FREE
    //     index of that product. Chunking g therefore splits independent output
    //     COLUMNS and leaves the k-axis accumulation completely untouched.
    //     Contrast `DRESS_ROW_BLOCK`, where the split moved a GEMM's output
    //     ROWS and shifted OpenBLAS's accumulation lanes (~7e-15 per odd split).
    //   * `rho[g]`'s sum over mu stays whole and in order within each g.
    //   * `n_e[home] += w*rho` runs over g in strict ascending order: the chunk
    //     loop is serial and outer.
    // `mwe_scf_grid_chunking_is_inert.rs` pins this bit-for-bit, including
    // across worker counts.
    let chunk = chunk_override
        .unwrap_or_else(|| crate::reduce::deterministic_group_size(npts))
        .max(1);
    let mut n_e = vec![0.0_f64; natoms];
    let mut g0 = 0usize;
    while g0 < npts {
        let g1 = (g0 + chunk).min(npts);
        let chi = eval_basis_on_points(mol, obs_bs, &points[g0..g1])
            .map_err(|e| FerricError::General(format!("becke_charges: chi eval failed: {e}")))?;
        if chi.nrows() != nbf {
            return Err(FerricError::General(format!(
                "becke_charges: density shape {:?} != nbf {}",
                density.dim(),
                chi.nrows()
            )));
        }
        // ρ(r_g) = Σ_μν D_μν χ_μ(g) χ_ν(g) = Σ_μ χ_μ · (D·χ)_μ
        let d_chi = density.dot(&chi);
        let w = g1 - g0;
        let mut rho = vec![0.0_f64; w];
        for mu in 0..nbf {
            for gc in 0..w {
                rho[gc] += chi[(mu, gc)] * d_chi[(mu, gc)];
            }
        }
        // Per-atom electron count via Becke partition:
        //   n^A = Σ_{g: home=A} w_g ρ(r_g)
        // (Becke partition baked into `w_g · 1[home=A]`.)
        for gc in 0..w {
            n_e[home_atom[g0 + gc]] += weights[g0 + gc] * rho[gc];
        }
        g0 = g1;
    }

    // Mild renormalization: rescale to enforce Σ_A n^A = N_e (corrects
    // residual grid-quadrature error of ~0.003 e on N_e=10).
    let n_target = mol.nelec() as f64;
    let n_sum: f64 = n_e.iter().sum();
    let scale = if n_sum.abs() > 1e-12 {
        n_target / n_sum
    } else {
        1.0
    };

    Ok((0..natoms)
        .map(|a| mol.atoms[a].z as f64 - scale * n_e[a])
        .collect())
}

/// A spherically averaged free-atom radial density ρ⁰(r), tabulated on a
/// radial grid and interpolated smoothly, for use as a Hirshfeld proatom.
///
/// Built by [`spherically_averaged_proatom`] from a free-atom SCF density in the
/// molecule's own basis (so the Hirshfeld weight ratio is basis-consistent with
/// the molecular density); [`scf_proatom_provider`] builds one per element.
///
/// # Interpolant
///
/// [`at`](Self::at) evaluates `exp(y(r))`, where `y` is a cubic spline through
/// `y_k = ln ρ_k` at the tabulated radii `r_k`:
///
/// * **Positive everywhere** by construction: the Hirshfeld weight divides by
///   `Σ_B ρ⁰_B`, and a cubic spline of ρ itself undershoots below zero in an
///   exponentially decaying tail.
/// * **Accurate for a decaying density**: `ln ρ` of a Gaussian-basis atom is a
///   slowly curving function of r (exactly quadratic, `−2a r²`, in a
///   single-Gaussian tail), so the spline's O(h⁴) error is on a much flatter
///   function than ρ.
/// * **Even at the nucleus**: the spline is the restriction to r ≥ 0 of the
///   spline through the mirrored table `{(±r_k, y_k)}`. A spherically averaged
///   Gaussian-basis density is an even function of r, and so is this
///   extension: on `[0, r_0]` the mirrored segment is the even cubic
///   `y_0 + M_0 (r² − r_0²)/2`, so `y'(0) = 0` and ρ is C2 through r = 0.
///   With `r_0 = 0` the same equations reduce to the clamped condition
///   `y'(0) = 0`.
/// * **Smooth exponential tail**: the far end is a natural end
///   (`y''(r_last) = 0`) and `y` continues LINEARLY beyond the last used
///   radius with its end slope, which must be negative. ρ then decays
///   exponentially to zero instead of jumping to it, and the continuation is
///   C2 because the spline's second derivative is already zero there.
///   Knots from the first `ρ_k <= `[`Self::TAIL_RHO_MIN`] onward are not used
///   (they are underflow, not data).
///
/// The result is C2 in r everywhere, and [`deriv`](Self::deriv) is its exact
/// derivative at every r, knots included. Evaluation is O(1) per point on a
/// uniform table (direct index) and O(log n) otherwise.
///
/// A table that is zero at every radius (a bare nucleus, e.g. H⁺ in
/// Hirshfeld-I) is the identically zero proatom.
#[derive(Debug, Clone)]
pub struct RadialProatom {
    radii: Vec<f64>,
    rho: Vec<f64>,
    /// `None` for the identically zero proatom.
    spline: Option<LogSpline>,
}

/// Cubic spline in `ln ρ` over the used knots (see [`RadialProatom`]).
#[derive(Debug, Clone)]
struct LogSpline {
    /// Used knots `r_0 < … < r_{n-1}`, `n >= 2`, `r_0 >= 0`.
    r: Vec<f64>,
    /// `ln ρ_k`.
    y: Vec<f64>,
    /// `y''(r_k)`; `m[n-1] = 0` (natural end).
    m: Vec<f64>,
    /// `y'(r_{n-1}) < 0`, the slope of the linear continuation.
    tail_slope: f64,
    /// `(r_0, 1/h)` when the knots are uniform: the segment is a direct index.
    uniform: Option<(f64, f64)>,
}

impl LogSpline {
    fn build(r: Vec<f64>, y: Vec<f64>) -> Result<Self, FerricError> {
        let n = r.len();
        debug_assert!(n >= 2 && y.len() == n);
        // Second-derivative (moment) equations. Row 0 is the continuity row
        // at r_0 with the mirrored segment [-r_0, r_0] (length 2 r_0, moments
        // M_0 at both ends by symmetry, equal values so zero secant):
        //   (r_0 + h_0/3) M_0 + h_0/6 M_1 = (y_1 - y_0)/h_0.
        // Interior rows are the standard ones; the last row is M_{n-1} = 0.
        // Strictly diagonally dominant, so the Thomas sweep is stable.
        let h: Vec<f64> = (0..n - 1).map(|i| r[i + 1] - r[i]).collect();
        let mut diag = vec![0.0_f64; n];
        let mut upper = vec![0.0_f64; n];
        let mut lower = vec![0.0_f64; n];
        let mut rhs = vec![0.0_f64; n];
        diag[0] = r[0] + h[0] / 3.0;
        upper[0] = h[0] / 6.0;
        rhs[0] = (y[1] - y[0]) / h[0];
        for i in 1..n - 1 {
            lower[i] = h[i - 1] / 6.0;
            diag[i] = (h[i - 1] + h[i]) / 3.0;
            upper[i] = h[i] / 6.0;
            rhs[i] = (y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1];
        }
        diag[n - 1] = 1.0;
        rhs[n - 1] = 0.0;
        // Forward elimination.
        for i in 1..n {
            let w = lower[i] / diag[i - 1];
            diag[i] -= w * upper[i - 1];
            rhs[i] -= w * rhs[i - 1];
        }
        let mut m = vec![0.0_f64; n];
        m[n - 1] = rhs[n - 1] / diag[n - 1];
        for i in (0..n - 1).rev() {
            m[i] = (rhs[i] - upper[i] * m[i + 1]) / diag[i];
        }
        let hl = h[n - 2];
        let tail_slope = (y[n - 1] - y[n - 2]) / hl + hl * (m[n - 2] + 2.0 * m[n - 1]) / 6.0;
        if !(tail_slope < 0.0) {
            return Err(FerricError::General(format!(
                "RadialProatom: ln ρ has slope {tail_slope:.3e} at the last used radius \
                 {:.4} Bohr; a proatom must decay there",
                r[n - 1]
            )));
        }
        let span = r[n - 1] - r[0];
        let step = span / (n - 1) as f64;
        let uniform = (0..n)
            .all(|k| (r[k] - (r[0] + k as f64 * step)).abs() <= 1e-9 * step)
            .then_some((r[0], 1.0 / step));
        Ok(Self {
            r,
            y,
            m,
            tail_slope,
            uniform,
        })
    }

    /// `(y(r), y'(r))` for `r >= 0`.
    fn eval(&self, r: f64) -> (f64, f64) {
        let n = self.r.len();
        let i: isize = if r < self.r[0] {
            -1
        } else if r >= self.r[n - 1] {
            (n - 1) as isize
        } else {
            (match self.uniform {
                Some((a, inv_h)) => (((r - a) * inv_h) as usize).min(n - 2),
                None => self.r.partition_point(|&x| x <= r).clamp(1, n - 1) - 1,
            }) as isize
        };
        self.piece(i, r)
    }

    /// `(y, y')` of polynomial piece `i` at `r` (any r, no range check):
    /// `-1` the even mirrored segment `[0, r_0]`, `0..n-1` the cubic on
    /// `[r_i, r_{i+1}]`, `n-1` the linear tail beyond `r_{n-1}`.
    fn piece(&self, i: isize, r: f64) -> (f64, f64) {
        let n = self.r.len();
        if i < 0 {
            // Even mirrored segment: y_0 + M_0 (r² − r_0²)/2.
            let r0 = self.r[0];
            return (
                self.y[0] + 0.5 * self.m[0] * (r * r - r0 * r0),
                self.m[0] * r,
            );
        }
        let i = i as usize;
        if i >= n - 1 {
            let rl = self.r[n - 1];
            return (self.y[n - 1] + self.tail_slope * (r - rl), self.tail_slope);
        }
        let h = self.r[i + 1] - self.r[i];
        let a = (self.r[i + 1] - r) / h;
        let b = 1.0 - a;
        let (mi, mj) = (self.m[i], self.m[i + 1]);
        let y = a * self.y[i]
            + b * self.y[i + 1]
            + ((a * a * a - a) * mi + (b * b * b - b) * mj) * h * h / 6.0;
        let dy = (self.y[i + 1] - self.y[i]) / h - (3.0 * a * a - 1.0) * h * mi / 6.0
            + (3.0 * b * b - 1.0) * h * mj / 6.0;
        (y, dy)
    }
}

impl RadialProatom {
    /// Tabulated densities at or below this value end the used table (see
    /// the type doc). It is far below anything a consumer resolves (the
    /// Hirshfeld weight's denominator floor is 1e-12) and far above the
    /// subnormal range, where `ln ρ` of a tabulated value loses precision.
    pub const TAIL_RHO_MIN: f64 = 1e-200;

    /// Build the smooth proatom through the table `(radii[k], rho[k])`.
    ///
    /// # Errors
    ///
    /// Mismatched or empty arrays; a radius that is not finite, a negative
    /// first radius, or radii that are not strictly ascending; a density that
    /// is not finite or is negative; a table that is not identically zero but
    /// has fewer than two leading values above [`Self::TAIL_RHO_MIN`]; and a
    /// table whose `ln ρ` does not decrease at the last used radius (no
    /// decaying tail to continue).
    pub fn new(radii: Vec<f64>, rho: Vec<f64>) -> Result<Self, FerricError> {
        let err = |m: String| FerricError::General(format!("RadialProatom: {m}"));
        if radii.is_empty() || radii.len() != rho.len() {
            return Err(err(format!(
                "{} radii and {} densities; need equal, non-zero lengths",
                radii.len(),
                rho.len()
            )));
        }
        if radii.iter().any(|r| !r.is_finite()) || radii[0] < 0.0 {
            return Err(err("radii must be finite and start at r >= 0".into()));
        }
        if radii.windows(2).any(|w| w[1] <= w[0]) {
            return Err(err("radii must be strictly ascending".into()));
        }
        if rho.iter().any(|p| !p.is_finite() || *p < 0.0) {
            return Err(err("densities must be finite and >= 0".into()));
        }
        if rho.iter().all(|&p| p == 0.0) {
            return Ok(Self {
                radii,
                rho,
                spline: None,
            });
        }
        let used = rho
            .iter()
            .position(|&p| p <= Self::TAIL_RHO_MIN)
            .unwrap_or(rho.len());
        if used < 2 {
            return Err(err(format!(
                "only {used} leading densities above {:e}; need at least 2",
                Self::TAIL_RHO_MIN
            )));
        }
        let spline = LogSpline::build(
            radii[..used].to_vec(),
            rho[..used].iter().map(|p| p.ln()).collect(),
        )?;
        Ok(Self {
            radii,
            rho,
            spline: Some(spline),
        })
    }

    /// Tabulated radii (Bohr), strictly ascending.
    pub fn radii(&self) -> &[f64] {
        &self.radii
    }

    /// Tabulated densities ρ⁰(r_k) (a.u.).
    pub fn rho(&self) -> &[f64] {
        &self.rho
    }

    /// `(ρ⁰(r), dρ⁰/dr)` of the smooth interpolant, as an even function of r.
    pub fn value_and_deriv(&self, r: f64) -> (f64, f64) {
        let Some(s) = &self.spline else {
            return (0.0, 0.0);
        };
        let (y, dy) = s.eval(r.abs());
        let v = y.exp();
        (v, v * dy * r.signum())
    }

    /// ρ⁰ at distance `r` (Bohr); see the type doc for the interpolant.
    pub fn at(&self, r: f64) -> f64 {
        self.value_and_deriv(r).0
    }

    /// dρ⁰/dr of [`at`](Self::at): the exact derivative of the interpolant
    /// (not of the free-atom density it interpolates) at every r, knots
    /// included. 0 at r = 0.
    pub fn deriv(&self, r: f64) -> f64 {
        self.value_and_deriv(r).1
    }
}

/// Spherically average a single-atom density (atom at the origin) onto a radial
/// grid via Lebedev angular quadrature. `atom_density` is the atomic SCF AO
/// density matrix in `atom_bs`; the returned [`RadialProatom`] is the proatom
/// reference for Hirshfeld partitioning.
///
/// # Errors
///
/// An AO evaluation failure, or a table [`RadialProatom::new`] refuses (e.g.
/// radii that are not strictly ascending, or a density that does not decay).
pub fn spherically_averaged_proatom(
    z: i32,
    atom_bs: &ferric_core::basis::BasisSet,
    atom_density: &Array2<f64>,
    radii: &[f64],
) -> Result<RadialProatom, FerricError> {
    use ferric_core::mol::{Atom, Molecule};
    use ferric_dft::ao_grid::eval_basis_on_points;
    use ferric_dft::lebedev::lebedev;

    let sym = ferric_core::elements::z_to_symbol(z).unwrap_or("X");
    let atom_mol = Molecule {
        atoms: vec![Atom {
            symbol: sym.to_string(),
            z,
            x: 0.0,
            y: 0.0,
            zpos: 0.0,
            ghost: false,
            n_core_ecp: 0,
        }],
        charge: 0,
        multiplicity: 1,
    };
    let (dirs, wts) = lebedev(110);
    let mut rho = vec![0.0_f64; radii.len()];
    for (ri, &r) in radii.iter().enumerate() {
        // Build the sphere of radius r and evaluate the density on it.
        let pts: Vec<[f64; 3]> = dirs
            .iter()
            .map(|d| [d[0] * r, d[1] * r, d[2] * r])
            .collect();
        let chi = eval_basis_on_points(&atom_mol, atom_bs, &pts)
            .map_err(|e| FerricError::General(format!("proatom chi eval: {e}")))?;
        let nbf = chi.nrows();
        let d_chi = atom_density.dot(&chi);
        // Angular average: Σ_k w_k ρ(r,Ω_k), weights sum to 1.
        let mut acc = 0.0;
        for (k, &w) in wts.iter().enumerate() {
            let mut rho_k = 0.0;
            for mu in 0..nbf {
                rho_k += chi[(mu, k)] * d_chi[(mu, k)];
            }
            acc += w * rho_k;
        }
        rho[ri] = acc.max(0.0);
    }
    RadialProatom::new(radii.to_vec(), rho)
}

/// Provider of spherically-averaged free-atom proatom densities: given element
/// `z` and integer charge state `q`, returns the radial proatom (or `None` if
/// unavailable). Built by the caller from atomic SCF in the molecule's basis.
pub type ProatomProvider<'a> = dyn Fn(i32, i32) -> Option<RadialProatom> + 'a;

/// Spin multiplicity of the neutral free-atom SCF behind [`scf_proatom_provider`]
/// (and ferric-cli's TS free-atom volumes).
pub fn proatom_ground_state_mult(z: i32) -> usize {
    match z {
        // Doublets: H, Li, B, F, Na, Al, Cl, Ga, Br (one unpaired p/s e⁻)
        1 | 3 | 5 | 9 | 11 | 13 | 17 | 31 | 35 | 53 => 2,
        // ²S alkali-like heavy atoms + coinage metals (single ns valence e⁻):
        // K, Cu, Rb, Ag. Without these an odd-electron atom hits `_ => 1` and
        // its closed-shell proatom RHF fails at iter 0.
        19 | 29 | 37 | 47 => 2,
        // Triplets (³P): C, O, Si, S, Ge, Se
        6 | 8 | 14 | 16 | 32 | 34 => 3,
        // Quartets (⁴S): N, P, As
        7 | 15 | 33 => 4,
        // Odd electron count can never be a singlet: default odd Z to a doublet.
        _ if z % 2 == 1 => 2,
        _ => 1,
    }
}

/// Radii (Bohr) on which [`scf_proatom_provider`] tabulates each proatom:
/// 0.05 to 30 Bohr in 0.05 Bohr steps (600 points).
pub fn scf_proatom_radii() -> Vec<f64> {
    (1..=600).map(|k| k as f64 * 0.05).collect()
}

/// Hirshfeld proatom provider built from free-atom SCF densities in the
/// molecule's own basis — the proatom ferric-cli and the Python
/// `ferric.hirshfeld_charges` pass by default.
///
/// For a neutral atom (`q == 0`) of element `z`, the returned closure solves the
/// isolated atom at the origin in `bs` with multiplicity
/// [`proatom_ground_state_mult`], using `config` (the molecule's own SCF
/// settings: method, functional, density fitting, thresholds):
///
/// * singlet → `solve_rhf`;
/// * otherwise → `solve_uhf` with MOM armed after iteration 5, and, when
///   `config.xc` is set, fractional (ensemble) occupation of the degenerate
///   frontier orbitals so the open-shell GGA atom stays spherical and converges.
///
/// The total density is spherically averaged by [`spherically_averaged_proatom`]
/// onto [`scf_proatom_radii`]. It returns `None` — so the Hirshfeld routine
/// falls back to the Slater proatom of [`slater_xi_for_z`] for that atom — for a
/// charged state (`q != 0`), for `z - q <= 0`, and when the free-atom SCF errors
/// or does not converge (`solve_rhf`/`solve_uhf` return `Ok` at `max_iter`, so
/// convergence is checked explicitly). Each free-atom SCF runs on a one-thread
/// rayon pool. Nothing is cached: every call re-solves the atom.
pub fn scf_proatom_provider<'a>(
    ctx: &'a ferric_core::parallel::ParallelContext,
    bs: &'a ferric_core::basis::BasisSet,
    op: ferric_integrals::operator::Operator,
    config: &'a crate::rhf::RhfConfig,
) -> impl Fn(i32, i32) -> Option<RadialProatom> + 'a {
    let radii = scf_proatom_radii();
    move |z: i32, qi: i32| -> Option<RadialProatom> {
        if qi != 0 || z - qi <= 0 {
            return None;
        }
        let mult = proatom_ground_state_mult(z);
        let sym = ferric_core::elements::z_to_symbol(z).unwrap_or("X");
        let axyz = format!("1\n{sym}\n{sym} 0 0 0\n");
        let amol = Molecule::parse_xyz(&axyz, 0, mult).ok()?;
        let aobs = PreparedBasis::new(&amol, bs).ok()?;
        let abounds = crate::screening::SchwarzBounds::compute(op, &aobs).ok()?;
        let mut acfg = config.clone();
        // A proatom is the ISOLATED free atom: the molecule's environment
        // (point charges/field, implicit solvent, polarizable sites, cDFT
        // constraints) is not applied to it.
        acfg.external_potential = None;
        acfg.cosmo = None;
        acfg.pcm = None;
        acfg.polarizable = None;
        acfg.constraints.clear();
        if mult != 1 {
            acfg.mom_after_iter = 5;
            // Pure HF free atoms do not need this (K is orbital-invariant in
            // the degenerate subspace), so only enable it when xc is set.
            if acfg.xc.is_some() {
                acfg.fractional_occ = true;
            }
        }
        let solve = || {
            if mult == 1 {
                crate::rhf::solve_rhf(ctx, &amol, &aobs, op, &abounds, &acfg)
                    .ok()
                    .filter(|r| r.converged)
                    .map(|r| r.density_r().to_owned())
            } else {
                crate::uhf::solve_uhf(ctx, &amol, &aobs, &abounds, &acfg)
                    .ok()
                    .filter(|r| r.converged)
                    .map(|r| r.density_total().to_owned())
            }
        };
        // One-thread pool (inline if it cannot be built): on the global pool
        // rayon's coordination overhead dwarfs a one-atom Fock build — a single
        // S atom at aug-cc-pVDZ took 179 s with RAYON_NUM_THREADS=8 vs 9.6 s
        // with 1.
        let adens = match rayon::ThreadPoolBuilder::new().num_threads(1).build() {
            Ok(pool) => pool.install(solve),
            Err(_) => solve(),
        }?;
        spherically_averaged_proatom(z, bs, &adens, &radii).ok()
    }
}

/// Löwdin atomic charges from symmetrically orthogonalized AOs.
///
/// In the Löwdin basis χ̃ = S^{-1/2} χ, the AOs are orthonormal and
/// remain atom-centered (no shape redistribution to other centers).
/// Per-atom populations are
///
/// ```text
///     n_A = Σ_{μ ∈ A} (S^{1/2} D S^{1/2})_{μμ}
///     q_A = Z_A − n_A
/// ```
///
/// Compared to Mulliken: less basis-set sensitive (no off-diagonal D·S
/// terms that can go negative). Compared to grid Hirshfeld with a
/// single-exponential proatom: no proatom shape required, so the C–O
/// inversion that plagues simple proatom models is gone.
///
/// Total electron count is conserved exactly by construction
/// (Tr[S^{1/2} D S^{1/2}] = Tr[D S] = N_e).
///
/// # Convention note
///
/// Löwdin charges depend on the *AO ordering convention* of the basis set.
/// Within a given convention the answer is well-defined and self-consistent
/// (and conserves N_e exactly). Across conventions the per-atom split can
/// differ — e.g. ferric (libint conventions) returns q_O = −0.48 on H2O /
/// cc-pVDZ while PySCF returns q_O = −0.10 on the same density. Both are
/// "Löwdin charges"; the difference reflects how each library orders d-
/// functions (libint: l-major Cartesian or pure-spherical depending on
/// shell setup; PySCF: spherical with its own canonical order).
///
/// This matters downstream only if you intend to mix Löwdin charges across
/// engines. As a baseline for CM5 within ferric the result is fully
/// self-consistent.
///
/// Closed-shell only.
pub fn lowdin_charges(
    mol: &Molecule,
    prep: &PreparedBasis,
    density: &Array2<f64>,
) -> Result<Vec<f64>, FerricError> {
    use ndarray_linalg::Eigh;

    let nbf = prep.nbasis();
    if density.nrows() != nbf || density.ncols() != nbf {
        return Err(FerricError::General(format!(
            "lowdin_charges: density {:?} != nbf {}",
            density.dim(),
            nbf
        )));
    }

    let s = oneelectron::overlap(prep);

    // S = U diag(λ) U^T → S^{1/2} = U diag(√λ) U^T.
    let (eigvals, eigvecs) = s
        .eigh(ndarray_linalg::UPLO::Upper)
        .map_err(|e| FerricError::General(format!("lowdin_charges: S eigh failed: {e}")))?;
    let mut sqrt_lambda = Array2::<f64>::zeros((nbf, nbf));
    for i in 0..nbf {
        if eigvals[i] <= 0.0 {
            return Err(FerricError::General(format!(
                "lowdin_charges: overlap eigenvalue {} <= 0 (linear dependence)",
                eigvals[i]
            )));
        }
        sqrt_lambda[(i, i)] = eigvals[i].sqrt();
    }
    let s_half = eigvecs.dot(&sqrt_lambda).dot(&eigvecs.t());

    // M = S^{1/2} · D · S^{1/2}
    let m = s_half.dot(density).dot(&s_half);

    // shell → atom; shell_offsets gives [start_μ for each shell].
    let shell_to_atom = prep.shell_to_atom();
    let shell_offsets = prep.shell_offsets();
    let natoms = mol.atoms.len();

    let mut atom_pop = vec![0.0_f64; natoms];
    for (sh_idx, &atom_idx) in shell_to_atom.iter().enumerate() {
        let mu0 = shell_offsets[sh_idx];
        let mu1 = shell_offsets[sh_idx + 1];
        for mu in mu0..mu1 {
            atom_pop[atom_idx] += m[(mu, mu)];
        }
    }

    Ok((0..natoms)
        .map(|a| mol.atoms[a].z as f64 - atom_pop[a])
        .collect())
}

/// Mulliken partial charges (units of e), the standard population analysis:
/// q_A = Z_A - Σ_{μ∈A} (D·S)_{μμ}.
///
/// Unlike Löwdin (which symmetrically orthogonalizes via S^{1/2}), Mulliken
/// splits each off-diagonal (D·S) contribution evenly between its two AO
/// centers with no basis-set-size correction — the textbook population
/// analysis, well known to be basis-set-sensitive (can misbehave badly with
/// diffuse/augmented functions) but included here as the standard baseline
/// every QC package provides, not as a recommended charge scheme. Prefer
/// `lowdin_charges` for a more basis-stable partition. Closed-shell only.
pub fn mulliken_charges(
    mol: &Molecule,
    prep: &PreparedBasis,
    density: &Array2<f64>,
) -> Result<Vec<f64>, FerricError> {
    let nbf = prep.nbasis();
    if density.nrows() != nbf || density.ncols() != nbf {
        return Err(FerricError::General(format!(
            "mulliken_charges: density {:?} != nbf {}",
            density.dim(),
            nbf
        )));
    }

    let s = oneelectron::overlap(prep);

    // M = D · S; the Mulliken atomic population is the sum of M's diagonal
    // over AOs centered on that atom (trace(D·S) = N_e exactly).
    let m = density.dot(&s);

    let shell_to_atom = prep.shell_to_atom();
    let shell_offsets = prep.shell_offsets();
    let natoms = mol.atoms.len();

    let mut atom_pop = vec![0.0_f64; natoms];
    for (sh_idx, &atom_idx) in shell_to_atom.iter().enumerate() {
        let mu0 = shell_offsets[sh_idx];
        let mu1 = shell_offsets[sh_idx + 1];
        for mu in mu0..mu1 {
            atom_pop[atom_idx] += m[(mu, mu)];
        }
    }

    Ok((0..natoms)
        .map(|a| mol.atoms[a].z as f64 - atom_pop[a])
        .collect())
}

/// One grid point surviving CHELPG's vdW-exclusion / outer-cutoff filter,
/// paired with the molecular ESP `V_QM(r)` evaluated there.
struct EspGridPoint {
    r: [f64; 3],
    v: f64,
}

/// Build the CHELPG grid: a cubic grid of `spacing` (Bohr) spanning the
/// molecule's bounding box plus `margin` (Bohr) in every direction, then
/// evaluate `V_QM(r) = Σ_B Z_B/|r−R_B| − Σ_μν D_μν ⟨μ|1/|r−r_g||ν⟩` at every
/// surviving point.
///
/// A grid point survives iff:
///   * it lies **outside** `vdw_scale × bondi_radius(Z_A)` of every atom A
///     (excludes the region where the point-charge model of the ESP is least
///     accurate — close to a nucleus the true multi-center ESP is dominated
///     by the local cusp, not the far-field 1/r tail a fitted point charge
///     reproduces); and
///   * it lies **within** `outer_cutoff` (Bohr) of at least one atom (bounds
///     the fit region to where the ESP is still chemically meaningful — far
///     outside the molecule V_QM → 0 and contributes no information, only
///     numerical noise, to the fit).
///
/// This is the standard CHELPG (Breneman & Wiberg 1990) grid definition,
/// implemented directly in Bohr (this codebase's native length unit)
/// rather than the paper's Å values — 0.3 Bohr spacing / 2.8 Bohr margin
/// is a deliberate same-shape, tighter-in-absolute-terms grid, not a
/// unit-conversion slip.
///
/// V(r) evaluation reuses the exact same libint nuclear-attraction
/// probe-charge trick as [`esp_at_atoms`] (Z=+1 point charge, sign-flip
/// convention) and [`ferric_pcm::potential::solute_potential_at_tesserae`]
/// (which documents the same derivation for cavity tesserae) — this is the
/// third call site of that pattern, now at freely-placed grid points rather
/// than nuclei or cavity surface points. Grid points are independent probes,
/// so they're processed in rayon like `esp_at_atoms`' per-atom loop (not
/// serial like the tesserae version, since CHELPG grids run to thousands of
/// points rather than a few hundred).
fn chelpg_grid_esp(
    mol: &Molecule,
    prep: &PreparedBasis,
    density: &Array2<f64>,
    spacing: f64,
    margin: f64,
    vdw_scale: f64,
    outer_cutoff: f64,
) -> Result<Vec<EspGridPoint>, FerricError> {
    use ferric_pcm::radii::bondi_radius_bohr;

    let nbas = prep.nbasis();
    if density.shape() != [nbas, nbas] {
        return Err(FerricError::General(format!(
            "chelpg_grid_esp: density shape {:?} != ({nbas},{nbas})",
            density.shape()
        )));
    }
    if !(spacing.is_finite() && spacing > 0.0) {
        return Err(FerricError::General(format!(
            "chelpg_grid_esp: spacing must be finite > 0, got {spacing}"
        )));
    }

    let natoms = mol.atoms.len();
    if natoms == 0 {
        return Err(FerricError::General(
            "chelpg_grid_esp: empty molecule".into(),
        ));
    }

    // Bounding box (Bohr) + margin, same convention as
    // `ferric_export::cube::GridSpec::bounding_box`.
    let mut lo = [f64::MAX; 3];
    let mut hi = [f64::MIN; 3];
    let atom_pos: Vec<[f64; 3]> = mol.atoms.iter().map(|a| [a.x, a.y, a.zpos]).collect();
    let atom_r_excl: Vec<f64> = mol
        .atoms
        .iter()
        .map(|a| vdw_scale * bondi_radius_bohr(a.z))
        .collect();
    for p in &atom_pos {
        for d in 0..3 {
            lo[d] = lo[d].min(p[d]);
            hi[d] = hi[d].max(p[d]);
        }
    }
    // Grid is built symmetric about the bounding box's own CENTER (not
    // anchored at `lo - margin` and stepped forward), so that a molecule
    // with an exact point-group symmetry (e.g. water's C2v mirror plane)
    // gets a grid that respects that symmetry too. An origin-anchored,
    // ceil-rounded grid is generically NOT symmetric under the molecule's
    // own symmetry operations (confirmed: water's C2v mirror maps its grid
    // to a copy offset by a fraction of `spacing`), which silently breaks
    // exact charge-symmetry between symmetry-equivalent atoms at the
    // ~1e-4-e level — small numerically, but a real, avoidable artifact
    // rather than physics. `half_pts` on each axis is the number of grid
    // steps needed to cover the half-extent (bounding-box half-width +
    // margin), so the full per-axis point count is always `2*half_pts + 1`
    // (odd, with a point exactly at the center) — symmetric by construction
    // for any bounding box, not just symmetric molecules.
    let center = [
        0.5 * (lo[0] + hi[0]),
        0.5 * (lo[1] + hi[1]),
        0.5 * (lo[2] + hi[2]),
    ];
    let half_pts = [
        ((0.5 * (hi[0] - lo[0]) + margin) / spacing).ceil().max(1.0) as usize,
        ((0.5 * (hi[1] - lo[1]) + margin) / spacing).ceil().max(1.0) as usize,
        ((0.5 * (hi[2] - lo[2]) + margin) / spacing).ceil().max(1.0) as usize,
    ];
    let origin = [
        center[0] - half_pts[0] as f64 * spacing,
        center[1] - half_pts[1] as f64 * spacing,
        center[2] - half_pts[2] as f64 * spacing,
    ];
    let n = [
        2 * half_pts[0] + 1,
        2 * half_pts[1] + 1,
        2 * half_pts[2] + 1,
    ];
    let npts_total = n[0] * n[1] * n[2];

    // Size guard, mirroring `eval_basis_on_grid`'s fail-fast convention:
    // don't silently build an unbounded candidate-point list for a very
    // fine spacing / large molecule.
    let peak_bytes = npts_total.saturating_mul(std::mem::size_of::<[f64; 3]>());
    ferric_core::memory::check_alloc(
        &format!(
            "chelpg candidate grid ({}×{}×{} = {npts_total} pts before vdW filtering)",
            n[0], n[1], n[2]
        ),
        peak_bytes,
        ferric_core::memory::resolve_budget_bytes(None),
    )
    .map_err(|e| FerricError::General(e.to_string()))?;

    // Filter candidate points to the CHELPG shell (outside vdW, inside outer
    // cutoff) BEFORE the expensive V(r) evaluation — most of a generous
    // bounding-box grid is either buried inside an atom or wasted empty
    // space far from the molecule.
    let mut kept: Vec<[f64; 3]> = Vec::new();
    for ix in 0..n[0] {
        let x = origin[0] + ix as f64 * spacing;
        for iy in 0..n[1] {
            let y = origin[1] + iy as f64 * spacing;
            for iz in 0..n[2] {
                let z = origin[2] + iz as f64 * spacing;
                let r = [x, y, z];

                let mut inside_any_vdw = false;
                let mut within_outer_cutoff = false;
                for a in 0..natoms {
                    let dx = r[0] - atom_pos[a][0];
                    let dy = r[1] - atom_pos[a][1];
                    let dz = r[2] - atom_pos[a][2];
                    let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                    if dist < atom_r_excl[a] {
                        inside_any_vdw = true;
                        break;
                    }
                    if dist <= atom_r_excl[a] + outer_cutoff {
                        within_outer_cutoff = true;
                    }
                }
                if !inside_any_vdw && within_outer_cutoff {
                    kept.push(r);
                }
            }
        }
    }

    if kept.is_empty() {
        return Err(FerricError::General(
            "chelpg_grid_esp: no grid points survived the vdW-exclusion/outer-cutoff filter \
             (spacing too coarse, or vdw_scale/outer_cutoff too tight)"
                .into(),
        ));
    }

    let values = esp_at_points(mol, prep, density, &kept)?;

    Ok(kept
        .into_iter()
        .zip(values)
        .map(|(r, v)| EspGridPoint { r, v })
        .collect())
}

/// Evaluate the molecular electrostatic potential `V_QM(r) = Σ_B Z_B/|r−R_B|
/// − Σ_μν D_μν ⟨μ|1/|r−r||ν⟩` at an explicit, caller-supplied list of
/// points (in Bohr).
///
/// The general-purpose primitive behind `chelpg_grid_esp` (which supplies
/// the CHELPG/RESP vdW-filtered grid) — factored out so it can also be
/// called directly against a fixed point list for a strict apples-to-apples
/// cross-check against an external reference (see
/// `crates/ferric-rpa/tests/properties_chelpg_resp.rs`'s PySCF cross-check,
/// which asks PySCF for `V_QM` at the exact same points via its own
/// `Vnuc − Vele` primitives). Same sign convention as [`esp_at_atoms`] and
/// [`ferric_pcm::potential::solute_potential_at_tesserae`] — see
/// `esp_at_atoms`'s doc comment for the libint probe-charge derivation.
pub fn esp_at_points(
    mol: &Molecule,
    prep: &PreparedBasis,
    density: &Array2<f64>,
    points: &[[f64; 3]],
) -> Result<Vec<f64>, FerricError> {
    use ferric_integrals::blas_threads::with_blas_threads;
    use rayon::prelude::*;

    let nbas = prep.nbasis();
    if density.shape() != [nbas, nbas] {
        return Err(FerricError::General(format!(
            "esp_at_points: density shape {:?} != ({nbas},{nbas})",
            density.shape()
        )));
    }

    let natoms = mol.atoms.len();
    let dims = prep.shell_dims();
    let offs = prep.shell_offsets();
    let nsh = prep.nshells();

    // Same per-worker stateful-engine pattern as `esp_at_atoms`: each point
    // is an independent probe, engine is Send-not-Sync so map_init hands
    // one engine per rayon worker rather than sharing/cloning.
    with_blas_threads(1, || {
        points
            .par_iter()
            .map_init(
                || Engine::new_1e(ffi::OP_NUCLEAR, prep, 1e-14),
                |eng, &r| -> Result<f64, FerricError> {
                    let eng = eng.as_mut().map_err(|e| {
                        FerricError::General(format!("esp_at_points: engine init failed: {e}"))
                    })?;

                    let probe = [CAtom {
                        atomic_number: 1.0,
                        x: r[0],
                        y: r[1],
                        z: r[2],
                    }];
                    // SAFETY: probe is a stack-local CAtom slice; handle_mut() is the live engine
                    // pointer; probe.len() fits in c_int. Shim catches C++ exceptions → negative rc.
                    let rc = unsafe {
                        ffi::scf_engine_set_point_charges(
                            eng.handle_mut(),
                            probe.as_ptr(),
                            probe.len() as c_int,
                        )
                    };
                    if rc < 0 {
                        return Err(FerricError::General(format!(
                            "esp_at_points: set_point_charges failed (rc={rc})"
                        )));
                    }

                    // V_elec(r) = + Σ_μν D_μν ⟨μ|−1/|r−r_g||ν⟩ (see
                    // `esp_at_atoms`'s doc comment for the sign derivation;
                    // identical here, just at an arbitrary point rather than
                    // a nucleus).
                    let mut v_elec = 0.0_f64;
                    for s1 in 0..nsh {
                        for s2 in 0..=s1 {
                            let block = eng.compute_1e_block(prep, s1, s2);
                            let n1 = dims[s1];
                            let n2 = dims[s2];
                            let o1 = offs[s1];
                            let o2 = offs[s2];
                            if s1 == s2 {
                                for i in 0..n1 {
                                    for j in 0..n2 {
                                        v_elec += density[(o1 + i, o2 + j)] * block[i * n2 + j];
                                    }
                                }
                            } else {
                                for i in 0..n1 {
                                    for j in 0..n2 {
                                        v_elec +=
                                            2.0 * density[(o1 + i, o2 + j)] * block[i * n2 + j];
                                    }
                                }
                            }
                        }
                    }

                    let mut v_nuc = 0.0_f64;
                    for b in 0..natoms {
                        let atom_b = &mol.atoms[b];
                        let dx = r[0] - atom_b.x;
                        let dy = r[1] - atom_b.y;
                        let dz = r[2] - atom_b.zpos;
                        let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                        v_nuc += atom_b.z as f64 / dist;
                    }

                    Ok(v_nuc + v_elec)
                },
            )
            .collect::<Result<Vec<f64>, FerricError>>()
    })
}

/// Solve the CHELPG-style constrained linear least-squares fit
///
/// ```text
///     minimize   Σ_g (V_QM(r_g) − Σ_A q_A/|r_g−R_A|)²
///     subject to Σ_A q_A = q_total
/// ```
///
/// via the standard Lagrange-multiplier normal-equations system (Breneman &
/// Wiberg, *J. Comput. Chem.* **11**, 361 (1990), Eq. 5-6): with
/// `A_{AB} = Σ_g 1/(|r_g−R_A| |r_g−R_B|)` and `b_A = Σ_g V_QM(r_g)/|r_g−R_A|`,
/// solve the `(natoms+1)×(natoms+1)` bordered system
///
/// ```text
///     [ A   1 ] [ q ]   [ b       ]
///     [ 1^T 0 ] [ λ ] = [ q_total ]
/// ```
///
/// This is a single direct linear solve, not an iterative optimizer — exact
/// up to the linear system's conditioning.
fn solve_chelpg_normal_equations(
    atom_pos: &[[f64; 3]],
    grid: &[EspGridPoint],
    q_total: f64,
) -> Result<Vec<f64>, FerricError> {
    use ndarray_linalg::Solve;

    let natoms = atom_pos.len();
    let n = natoms + 1;
    let mut mat = Array2::<f64>::zeros((n, n));
    let mut rhs = ndarray::Array1::<f64>::zeros(n);

    // Per-point, per-atom inverse distances (reused for both A_AB and b_A).
    let mut inv_r = vec![0.0_f64; natoms];
    for pt in grid {
        for a in 0..natoms {
            let dx = pt.r[0] - atom_pos[a][0];
            let dy = pt.r[1] - atom_pos[a][1];
            let dz = pt.r[2] - atom_pos[a][2];
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();
            inv_r[a] = 1.0 / dist;
        }
        for a in 0..natoms {
            rhs[a] += pt.v * inv_r[a];
            for b in 0..=a {
                let contrib = inv_r[a] * inv_r[b];
                mat[(a, b)] += contrib;
                if a != b {
                    mat[(b, a)] += contrib;
                }
            }
        }
    }

    // Border: Lagrange-multiplier row/column enforcing Σ q_A = q_total.
    for a in 0..natoms {
        mat[(a, natoms)] = 1.0;
        mat[(natoms, a)] = 1.0;
    }
    rhs[natoms] = q_total;

    let sol = mat.solve(&rhs).map_err(|e| {
        FerricError::Lapack(format!(
            "solve_chelpg_normal_equations: bordered normal-equations solve failed \
             (grid too small/degenerate, or atoms nearly coincident): {e}"
        ))
    })?;

    Ok(sol.iter().take(natoms).copied().collect())
}

/// CHELPG (CHarges from Electrostatic Potentials, Grid-based) atomic partial
/// charges.
///
/// Breneman, C. M.; Wiberg, K. B. "Determining Atom-Centered Monopoles from
/// Molecular Electrostatic Potentials." *J. Comput. Chem.* **1990**, *11*,
/// 361–373.
///
/// Structurally different from `hirshfeld_charges`/[`lowdin_charges`]/
/// [`mulliken_charges`]: those are **population-partition** schemes that
/// split the electron density directly among atoms. CHELPG instead chooses
/// atom-centered point charges `q_A` that best reproduce the *molecular
/// electrostatic potential* `V_QM(r)` on a grid of points around the
/// molecule, in a constrained least-squares sense — the standard charge
/// scheme for downstream force-field electrostatics.
///
/// # Grid
///
/// Cubic grid, `spacing` (default 0.3 Bohr) inside the molecule's bounding
/// box extended by `margin` (default 2.8 Bohr) in every direction, excluding
/// points within `vdw_scale × bondi_radius(Z_A)` (default scale 1.0) of any
/// atom A and points beyond `outer_cutoff` (default 2.8 Bohr) past the
/// nearest atom's vdW-scaled radius. Uses the same Bondi radii table as
/// `ferric_pcm`'s PCM/COSMO cavity construction
/// (`ferric_pcm::radii::bondi_radius_bohr`) — not a second hand-rolled
/// table.
///
/// # Fit
///
/// Solves the Lagrange-multiplier-constrained normal equations (a single
/// `(natoms+1)×(natoms+1)` linear solve, not an iterative optimizer) — see
/// `solve_chelpg_normal_equations`.
///
/// Returns `Vec<f64>` of length `mol.atoms.len()`, units of e, summing to
/// `mol.charge` (up to the linear solve's numerical precision — see the
/// `sum_matches_total_charge` regression tests for the achieved tolerance).
///
/// Closed-shell only (uses the total density; open-shell references should
/// pass `rhf.density_total()`, which is spin-summed and therefore already
/// correct here — no open-shell-specific machinery is needed for a
/// classical electrostatic-potential fit).
#[allow(clippy::too_many_arguments)]
pub fn chelpg_charges(
    mol: &Molecule,
    prep: &PreparedBasis,
    density: &Array2<f64>,
) -> Result<Vec<f64>, FerricError> {
    let grid = chelpg_grid_esp(
        mol,
        prep,
        density,
        chelpg_spacing(),
        chelpg_margin(),
        chelpg_vdw_scale(),
        chelpg_outer_cutoff(),
    )?;
    let atom_pos: Vec<[f64; 3]> = mol.atoms.iter().map(|a| [a.x, a.y, a.zpos]).collect();
    solve_chelpg_normal_equations(&atom_pos, &grid, mol.charge as f64)
}

/// Grid spacing (Bohr) for CHELPG/RESP. `FERRIC_CHELPG_SPACING`.
fn chelpg_spacing() -> f64 {
    positive_f64("FERRIC_CHELPG_SPACING", 0.3)
}

/// Bounding-box margin (Bohr) for CHELPG/RESP. `FERRIC_CHELPG_MARGIN`.
fn chelpg_margin() -> f64 {
    positive_f64("FERRIC_CHELPG_MARGIN", 2.8)
}

/// vdW-radius exclusion scale for CHELPG/RESP (grid points inside
/// `vdw_scale × bondi_radius` of any atom are dropped). `FERRIC_CHELPG_VDW_SCALE`.
fn chelpg_vdw_scale() -> f64 {
    positive_f64("FERRIC_CHELPG_VDW_SCALE", 1.0)
}

/// Outer cutoff (Bohr) past an atom's vdW-scaled radius beyond which grid
/// points are dropped. `FERRIC_CHELPG_OUTER_CUTOFF`.
fn chelpg_outer_cutoff() -> f64 {
    positive_f64("FERRIC_CHELPG_OUTER_CUTOFF", 2.8)
}

/// RESP (Restrained ElectroStatic Potential) atomic partial charges.
///
/// Bayly, C. I.; Cieplak, P.; Cornell, W. D.; Kollman, P. A. "A Well-behaved
/// Electrostatic Potential Based Method Using Charge Restraints for Deriving
/// Atomic Charges: The RESP Model." *J. Phys. Chem.* **1993**, *97*,
/// 10269–10280.
///
/// Same ESP grid (`chelpg_grid_esp`) and least-squares objective as
/// [`chelpg_charges`], plus a hyperbolic restraint that damps charges on
/// **non-hydrogen** atoms toward zero (mitigates overfitting/unphysically
/// large charges on buried heavy atoms):
///
/// ```text
///     minimize  Σ_g (V_QM(r_g) − V_fit(r_g))²
///               + restraint_weight · Σ_{A: Z_A≠1} (√(q_A² + b²) − b)
///     subject to Σ_A q_A = q_total
/// ```
///
/// # Scope
///
/// This is a **single-stage** restrained fit with the standard literature
/// weight/tightness parameters (`restraint_weight = 0.0005`, `b = 0.1 e`),
/// applied uniformly to every non-hydrogen atom. The full published RESP
/// recipe additionally runs a *second* stage that re-fits with a tighter
/// restraint applied only to specific chemically-equivalenced atom groups
/// (and, for force-field parameterization, averages over multiple
/// conformers) — that multi-stage/multi-conformer averaging is explicitly
/// OUT OF SCOPE here; this is a single-conformer, single-stage restrained
/// fit, an honest subset of the full RESP procedure rather than a full
/// reimplementation.
///
/// # Solving the nonlinear restraint
///
/// The restraint term `√(q_A²+b²) − b` is nonlinear in `q_A`, so the fit is
/// not a single linear solve. Standard RESP practice (and the approach here)
/// is a fixed-point/Newton iteration: at each iteration, linearize the
/// restraint's contribution to the gradient by evaluating its second
/// derivative at the *current* charge estimate,
///
/// ```text
///     d/dq_A [ restraint_weight · (√(q_A²+b²) − b) ] = restraint_weight · q_A / √(q_A²+b²)
///     ≈ restraint_weight / √(q_A²+b²) · q_A     (holding the denominator fixed within an iteration)
/// ```
///
/// which just adds a diagonal term `restraint_weight / √(q_A^(k)²+b²)` to the
/// CHELPG normal-equations matrix `A` (non-hydrogen rows only) at each
/// iteration `k`, then re-solves the same bordered linear system with the
/// updated diagonal — a short Newton/fixed-point loop over an otherwise
/// unchanged linear solve, not a black-box nonlinear optimizer.
///
/// Returns `Vec<f64>` of length `mol.atoms.len()`, units of e.
pub fn resp_charges(
    mol: &Molecule,
    prep: &PreparedBasis,
    density: &Array2<f64>,
) -> Result<Vec<f64>, FerricError> {
    let grid = chelpg_grid_esp(
        mol,
        prep,
        density,
        chelpg_spacing(),
        chelpg_margin(),
        chelpg_vdw_scale(),
        chelpg_outer_cutoff(),
    )?;
    let atom_pos: Vec<[f64; 3]> = mol.atoms.iter().map(|a| [a.x, a.y, a.zpos]).collect();
    let restraint_weight = resp_restraint_weight();
    let b = resp_restraint_b();
    let is_heavy: Vec<bool> = mol.atoms.iter().map(|a| a.z != 1).collect();

    solve_resp_restrained(
        &atom_pos,
        &grid,
        mol.charge as f64,
        &is_heavy,
        restraint_weight,
        b,
    )
}

/// CHELPG **and** RESP charges from ONE shared ESP grid.
///
/// [`chelpg_charges`] and [`resp_charges`] each call `chelpg_grid_esp` with the
/// identical arguments and differ only in the least-squares solve that follows.
/// Callers that want both (the CLI's `export_npz` path defaults BOTH to on) were
/// therefore evaluating the same molecular ESP over the same several-thousand
/// point grid twice: MEASURED 2.2 s each at benzene/def2-SVP with 12 threads
/// (11.6 s serial), i.e. ~2.2 s of pure duplicate work per run, scaling with
/// nsh² × npoints.
///
/// Returns `(chelpg, resp)`. Numerically identical to calling the two functions
/// separately — same grid, same solvers, just evaluated once.
pub fn chelpg_and_resp_charges(
    mol: &Molecule,
    prep: &PreparedBasis,
    density: &Array2<f64>,
) -> Result<(Vec<f64>, Vec<f64>), FerricError> {
    let grid = chelpg_grid_esp(
        mol,
        prep,
        density,
        chelpg_spacing(),
        chelpg_margin(),
        chelpg_vdw_scale(),
        chelpg_outer_cutoff(),
    )?;
    let atom_pos: Vec<[f64; 3]> = mol.atoms.iter().map(|a| [a.x, a.y, a.zpos]).collect();
    let chelpg = solve_chelpg_normal_equations(&atom_pos, &grid, mol.charge as f64)?;
    let is_heavy: Vec<bool> = mol.atoms.iter().map(|a| a.z != 1).collect();
    let resp = solve_resp_restrained(
        &atom_pos,
        &grid,
        mol.charge as f64,
        &is_heavy,
        resp_restraint_weight(),
        resp_restraint_b(),
    )?;
    Ok((chelpg, resp))
}

/// RESP hyperbolic restraint weight (e⁻¹, standard literature default
/// 0.0005). `FERRIC_RESP_RESTRAINT_WEIGHT`.
fn resp_restraint_weight() -> f64 {
    positive_f64("FERRIC_RESP_RESTRAINT_WEIGHT", 0.0005)
}

/// RESP hyperbolic restraint tightness parameter `b` (e, standard literature
/// default 0.1). `FERRIC_RESP_RESTRAINT_B`.
fn resp_restraint_b() -> f64 {
    positive_f64("FERRIC_RESP_RESTRAINT_B", 0.1)
}

/// Newton/fixed-point iteration solving the RESP-restrained bordered normal
/// equations. See [`resp_charges`]'s doc comment for the derivation.
///
/// Starts from the unrestrained CHELPG solution (iteration 0's diagonal
/// correction uses q_A=0 as the initial linearization point, which for the
/// hyperbolic penalty is a finite, well-defined starting slope
/// `restraint_weight / b` — no singularity at q=0 the way a bare `|q|`
/// restraint would have).
fn solve_resp_restrained(
    atom_pos: &[[f64; 3]],
    grid: &[EspGridPoint],
    q_total: f64,
    is_heavy: &[bool],
    restraint_weight: f64,
    b: f64,
) -> Result<Vec<f64>, FerricError> {
    use ndarray_linalg::Solve;

    let natoms = atom_pos.len();
    let n = natoms + 1;

    // Build the unrestrained normal-equations matrix/rhs once (A, b are
    // charge-independent; only the diagonal restraint correction changes
    // per iteration).
    let mut base_mat = Array2::<f64>::zeros((n, n));
    let mut rhs = ndarray::Array1::<f64>::zeros(n);
    let mut inv_r = vec![0.0_f64; natoms];
    for pt in grid {
        for a in 0..natoms {
            let dx = pt.r[0] - atom_pos[a][0];
            let dy = pt.r[1] - atom_pos[a][1];
            let dz = pt.r[2] - atom_pos[a][2];
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();
            inv_r[a] = 1.0 / dist;
        }
        for a in 0..natoms {
            rhs[a] += pt.v * inv_r[a];
            for b_idx in 0..=a {
                let contrib = inv_r[a] * inv_r[b_idx];
                base_mat[(a, b_idx)] += contrib;
                if a != b_idx {
                    base_mat[(b_idx, a)] += contrib;
                }
            }
        }
    }
    for a in 0..natoms {
        base_mat[(a, natoms)] = 1.0;
        base_mat[(natoms, a)] = 1.0;
    }
    rhs[natoms] = q_total;

    // Fixed-point/Newton loop on the restraint diagonal.
    let mut q = vec![0.0_f64; natoms];
    const MAX_ITER: usize = 50;
    const TOL: f64 = 1e-8;
    for _iter in 0..MAX_ITER {
        let mut mat = base_mat.clone();
        for a in 0..natoms {
            if is_heavy[a] {
                mat[(a, a)] += restraint_weight / (q[a] * q[a] + b * b).sqrt();
            }
        }
        let sol = mat.solve(&rhs).map_err(|e| {
            FerricError::Lapack(format!(
                "solve_resp_restrained: restrained normal-equations solve failed at \
                 iteration {_iter}: {e}"
            ))
        })?;
        let q_new: Vec<f64> = sol.iter().take(natoms).copied().collect();
        let max_dq = q_new
            .iter()
            .zip(&q)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        q = q_new;
        if max_dq < TOL {
            break;
        }
    }

    Ok(q)
}

/// Slater single-exponential proatom exponent ξ (Bohr⁻¹) for element Z.
///
/// Derived from Bragg-Slater empirical atomic radii R_BS:
///     ξ = 1 / (R_BS in Bohr),
/// giving the proatom ρ⁰(r) = Z ξ³/π · exp(−2ξr) (normalized to Z electrons).
/// Elements beyond Z = 18 all get R_BS = 1.00 Å.
///
/// This is the fallback proatom of the Hirshfeld routines in
/// `ferric-rpa::properties`: they use it for any atom their proatom provider
/// returns `None` for, and for every atom when no provider is passed. A single
/// exponential has no core peak, so Hirshfeld *charges* built on it are
/// qualitative: on H2O, CO and CH3OH they differ from the free-atom SCF
/// proatom charges by 0.23–0.72 e and flip the sign of the CH3OH carbon
/// (`crates/ferric-rpa/tests/validation_hirshfeld.rs`).
/// [`scf_proatom_provider`] supplies the free-atom SCF proatoms. The additive per-atom
/// polarizability partition in `pdep_polarizability_hirshfeld` is less
/// sensitive to the proatom shape, since its sum rule is enforced numerically.
pub fn slater_xi_for_z(z: i32) -> f64 {
    // Bragg-Slater radii in Angstrom (1 Å = 1.8897259886 Bohr).
    // Values from Slater J. Chem. Phys. 41, 3199 (1964) for Z=1..18.
    let r_bs_ang: f64 = match z {
        1 => 0.25,
        2 => 0.30,
        3 => 1.45,
        4 => 1.05,
        5 => 0.85,
        6 => 0.70,
        7 => 0.65,
        8 => 0.60,
        9 => 0.50,
        10 => 0.45,
        11 => 1.80,
        12 => 1.50,
        13 => 1.25,
        14 => 1.10,
        15 => 1.00,
        16 => 1.00,
        17 => 1.00,
        18 => 0.71,
        _ => 1.00,
    };
    let r_bs_bohr = r_bs_ang * 1.8897259886;
    1.0 / r_bs_bohr
}

/// 3x3 symmetric eigenvalue solver via Jacobi rotations.  Returns the three
/// eigenvalues sorted ascending.  Used to report principal polarizabilities.
pub fn eig3_sym(a: [[f64; 3]; 3]) -> Result<[f64; 3], FerricError> {
    // Use ndarray-linalg for robustness.
    use ndarray::arr2;
    use ndarray_linalg::Eigh;
    let m = arr2(&a);
    let (vals, _) = m
        .eigh(ndarray_linalg::UPLO::Upper)
        .map_err(|e| FerricError::Lapack(format!("principal-axis eigh: {e}")))?;
    let mut v = [vals[0], vals[1], vals[2]];
    v.sort_by(|a, b| a.total_cmp(b));
    Ok(v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rhf::{solve_rhf, RhfConfig};
    use crate::screening::SchwarzBounds;
    use ferric_core::basis;
    use ferric_core::parallel::ParallelContext;
    use ferric_integrals::operator::Operator;

    fn build_h2() -> (
        Molecule,
        PreparedBasis,
        PreparedBasis,
        Operator,
        crate::result::ScfResult,
    ) {
        // H2 at 1.4 Bohr, cc-pVDZ orbital + cc-pVDZ-RI aux.
        let xyz = "2\nH2\nH 0 0 0\nH 0 0 0.74083\n";
        let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
        let obs_bs = basis::bundled("cc-pvdz").unwrap();
        let dfbs_bs = basis::bundled("cc-pvdz-ri").unwrap();
        let op = Operator::coulomb();
        let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
        let dfbs = PreparedBasis::new(&mol, &dfbs_bs).unwrap();
        let ctx = ParallelContext::default();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default()).unwrap();
        (mol, obs, dfbs, op, rhf)
    }

    #[test]
    fn esp_at_h_in_h2_finite() {
        // Sanity: ESP at H in H2 is finite and on the order of −1 to +1 Ha
        // (electron cloud screens the other proton).
        let (mol, obs, _dfbs, _op, rhf) = build_h2();
        let v = esp_at_atoms(&mol, &obs, rhf.density_r()).unwrap();
        assert_eq!(v.len(), 2);
        // Symmetry: V(H1) == V(H2)
        assert!((v[0] - v[1]).abs() < 1e-8, "H2 ESP asymmetric: {v:?}");
        assert!(v[0].is_finite(), "ESP not finite");
        // Sanity-bound: bare-proton ESP at the bond partner is +1/1.4 ≈ 0.714,
        // electronic shielding brings it down well below that.  Just check
        // the value is within a wide physical band.
        assert!(
            v[0].abs() < 5.0,
            "ESP at H in H2 = {} Ha; outside physical band",
            v[0]
        );
    }

    #[test]
    fn becke_effective_volume_h2_finite_positive() {
        let (mol, obs, _dfbs, _op, rhf) = build_h2();
        let bs = basis::bundled("cc-pvdz").unwrap();
        let v = atomic_effective_volumes_becke(&mol, &obs, &bs, rhf.density_r()).unwrap();
        assert_eq!(v.len(), 2);
        assert!(v[0] > 0.0 && v[0].is_finite(), "vol[0]={}", v[0]);
        // H2 symmetric: equal volumes.
        assert!((v[0] - v[1]).abs() / v[0] < 1e-6, "asymmetric: {v:?}");
    }
}

/// Electric dipole moment, in atomic units (e·a₀).
///
/// `mu = Σ_A Z_A^eff R_A − Σ_μν D_μν ⟨μ|r|ν⟩`, with the electronic part from
/// the AO dipole integrals and the nuclear part from the EFFECTIVE nuclear
/// charges.
///
/// # Why `effective_z()` and not `z`
///
/// The dipole must be consistent with the density it is computed from. Under an
/// ECP the density only carries the VALENCE electrons, so pairing it with the
/// full `Z` double-counts the core and gives a dipole that is wrong by a large
/// amount on any heavy-element system. This mirrors the fix applied to the
/// nuclear-repulsion gradient (`gradient.rs`), where the same `z`-vs-
/// `effective_z()` inconsistency was a live bug. `effective_z()` equals `z` for
/// an all-electron atom, so ordinary systems are unaffected.
///
/// Note the origin dependence: for a NEUTRAL system the dipole is
/// origin-independent, but for a charged one it is not, and this uses the input
/// coordinate origin. Callers reporting a dipole for an ion must say which
/// origin they used.
pub fn dipole_moment(
    mol: &Molecule,
    prep: &PreparedBasis,
    density_total: &Array2<f64>,
) -> Result<[f64; 3], FerricError> {
    let nbas = prep.nbasis();
    if density_total.shape() != [nbas, nbas] {
        return Err(FerricError::General(format!(
            "dipole_moment: density shape {:?} != ({nbas},{nbas})",
            density_total.shape()
        )));
    }
    let dip_ao = oneelectron::dipole(prep, [0.0, 0.0, 0.0])?;
    let mut mu = [0.0f64; 3];
    for (d, mu_d) in mu.iter_mut().enumerate() {
        let elec = (density_total * &dip_ao[d]).sum();
        let nuc: f64 = mol
            .atoms
            .iter()
            .map(|a| a.effective_z() as f64 * [a.x, a.y, a.zpos][d])
            .sum();
        *mu_d = nuc - elec;
    }
    Ok(mu)
}

/// `|mu|` in atomic units.
pub fn dipole_magnitude(mu: &[f64; 3]) -> f64 {
    (mu[0] * mu[0] + mu[1] * mu[1] + mu[2] * mu[2]).sqrt()
}

/// Conversion factor from atomic units (e·a₀) to Debye.
///
/// CODATA: 1 e·a₀ = 8.4783536255e-30 C·m, and 1 D = 3.33564e-30 C·m.
pub const DEBYE_PER_AU: f64 = 2.541_746_473_1;

/// Electrostatic potential sampled on a molecular **surface**, not at nuclei.
///
/// Returns `(points, esp)` where `points` are Cartesian coordinates (Bohr) on a
/// solvent-accessible-style shell around the molecule and `esp` is the
/// potential there, in Hartree/e, with the same sign convention as
/// [`esp_at_atoms`].
///
/// # Why a surface and not the nuclei
///
/// [`esp_at_atoms`] evaluates V at each nucleus, where the `Z_A/|r − R_A|`
/// term of the *other* nuclei is finite but the local electronic cusp
/// dominates. The result is essentially a function of the atom's own nuclear
/// charge: measured over 500 QM9 molecules, the per-element ranges do not even
/// overlap (H −1.13..−0.92, C −14.75..−14.31, N −18.40..−18.16,
/// O −22.39..−22.15, F −26.49). As a conditioning signal for a generative
/// model that must *predict* element identity, that is a label, not a
/// descriptor.
///
/// The potential a *binding partner* feels is the one outside the van der
/// Waals surface, and that is what shape/electrostatics-conditioned generative
/// models (e.g. ShEPhERD) actually use. This function samples it.
///
/// # Construction
///
/// A Lebedev sphere of `n_angular` points is placed at `vdw_scale ×` the Bondi
/// radius of each atom, and any point falling inside another atom's scaled
/// radius is dropped, leaving the solvent-exposed envelope. Lebedev order is
/// used rather than the Cartesian lattice of [`chelpg_charges`] so the sample
/// is rotationally balanced and the count per atom is fixed, which matters when
/// the result is fed to a model as a per-atom feature.
pub fn esp_on_surface(
    mol: &Molecule,
    prep: &PreparedBasis,
    density: &Array2<f64>,
    vdw_scale: f64,
    n_angular: usize,
) -> Result<(Vec<[f64; 3]>, Vec<f64>), FerricError> {
    use ferric_pcm::radii::bondi_radius_bohr;
    use ferric_quadrature::lebedev::lebedev;

    if !(vdw_scale.is_finite() && vdw_scale > 0.0) {
        return Err(FerricError::General(format!(
            "esp_on_surface: vdw_scale must be finite > 0, got {vdw_scale}"
        )));
    }
    let (unit, _w) = lebedev(n_angular);
    let pos: Vec<[f64; 3]> = mol.atoms.iter().map(|a| [a.x, a.y, a.zpos]).collect();
    let radii: Vec<f64> = mol
        .atoms
        .iter()
        .map(|a| vdw_scale * bondi_radius_bohr(a.z))
        .collect();

    let mut points: Vec<[f64; 3]> = Vec::with_capacity(pos.len() * unit.len());
    for (a, c) in pos.iter().enumerate() {
        for u in &unit {
            let p = [
                c[0] + radii[a] * u[0],
                c[1] + radii[a] * u[1],
                c[2] + radii[a] * u[2],
            ];
            // Keep only the solvent-exposed envelope: drop points buried
            // inside a neighbour's shell.
            let buried = pos.iter().enumerate().any(|(b, q)| {
                if b == a {
                    return false;
                }
                let d2 = (p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2);
                d2 < radii[b] * radii[b]
            });
            if !buried {
                points.push(p);
            }
        }
    }
    if points.is_empty() {
        return Err(FerricError::General(
            "esp_on_surface: every sample point was buried; check vdw_scale".into(),
        ));
    }
    let esp = esp_at_points(mol, prep, density, &points)?;
    Ok((points, esp))
}

#[cfg(test)]
mod dipole_tests {
    use super::*;
    use ferric_core::basis;
    use ferric_core::parallel::ParallelContext;
    use ferric_integrals::operator::Operator;

    /// Water's RHF dipole, against PySCF.
    ///
    /// MEASURED reference, not recalled: PySCF 2.x
    /// `scf.RHF(mol).run(conv_tol=1e-12).dip_moment(unit='Debye')` on this
    /// exact geometry/basis gives
    ///
    ///     Dipole moment(X, Y, Z, Debye):  0.00000, 0.00000, -1.72748
    ///
    /// i.e. |mu| = 1.7274787296 D = 0.6796424222 a.u. ferric returns
    /// -0.6796424654 a.u., agreeing to 4.3e-8.
    ///
    /// (An earlier version of this test asserted ~1.53 D from memory and
    /// failed. The lesson is the obvious one: generate the reference, do not
    /// recall it.)
    ///
    /// An EXTERNAL reference is the point — a sign error, a missing nuclear
    /// term, or an origin mistake all break it, whereas comparing against
    /// ferric's own numbers would not.
    #[test]
    fn water_sto3g_dipole_matches_reference() {
        let ctx = ParallelContext::default();
        let mol = Molecule::load_xyz(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../testdata/molecules/water.xyz"
        ))
        .unwrap();
        let bs = basis::bundled("sto-3g").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = crate::screening::SchwarzBounds::compute(op, &prep).unwrap();
        let rhf = crate::rhf::solve_rhf(
            &ctx,
            &mol,
            &prep,
            op,
            &bounds,
            &crate::rhf::RhfConfig {
                density_conv: 1e-10,
                ..Default::default()
            },
        )
        .unwrap();

        let mu = dipole_moment(&mol, &prep, rhf.density_total()).unwrap();
        let mag_au = dipole_magnitude(&mu);
        let mag_d = mag_au * DEBYE_PER_AU;
        eprintln!("water/STO-3G dipole: {mu:?} a.u.  |mu| = {mag_au:.6} a.u. = {mag_d:.4} D");

        // Water's C2v axis is z in this geometry, so x/y must vanish by symmetry.
        // That is a structural check a wrong-axis bug fails immediately.
        assert!(
            mu[0].abs() < 1e-8,
            "x component must vanish by symmetry: {}",
            mu[0]
        );
        assert!(
            mu[1].abs() < 1e-8,
            "y component must vanish by symmetry: {}",
            mu[1]
        );
        // PySCF: 0.6796424222 a.u. Tolerance is 1e-6, ~20x the observed
        // 4.3e-8 residual, which is SCF-convergence noise rather than a
        // method difference (both are plain RHF on the same geometry/basis).
        const PYSCF_AU: f64 = 0.679_642_422_194_317;
        assert!(
            (mag_au - PYSCF_AU).abs() < 1e-6,
            "water/STO-3G RHF dipole {mag_au:.10} a.u. vs PySCF {PYSCF_AU:.10}"
        );
    }

    /// `effective_z()`, not `z`: under an ECP the density carries only valence
    /// electrons, so pairing it with the full nuclear charge double-counts the
    /// core. This pins that the ECP path uses the consistent charge.
    #[test]
    fn ecp_dipole_uses_the_effective_nuclear_charge() {
        let ctx = ParallelContext::default();
        let bs = basis::bundled("def2-svp").unwrap();
        let mut mol = Molecule::parse_xyz("2\n\nI 0.0 0.0 0.0\nH 0.0 0.0 1.61\n", 0, 1).unwrap();
        mol.apply_ecp(&bs);
        // TEETH: without an active ECP this test says nothing about ECPs.
        assert!(
            mol.atoms.iter().any(|a| a.n_core_ecp > 0),
            "def2-svp must carry an ECP for I"
        );

        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = crate::screening::SchwarzBounds::compute(op, &prep).unwrap();
        let rhf = crate::rhf::solve_rhf(
            &ctx,
            &mol,
            &prep,
            op,
            &bounds,
            &crate::rhf::RhfConfig::default(),
        )
        .unwrap();

        let mu = dipole_moment(&mol, &prep, rhf.density_total()).unwrap();
        let mag_d = dipole_magnitude(&mu) * DEBYE_PER_AU;
        eprintln!("HI/def2-SVP(ECP) dipole = {mag_d:.4} D");

        // HI's experimental dipole is ~0.45 D; any small-basis RHF value is in
        // the same ballpark. Using the BARE z instead of effective_z() would
        // add 46 e * 1.61 A of spurious nuclear charge-separation and give a
        // dipole of order 100 D, so a loose physical bound is a decisive test.
        assert!(
            mag_d < 10.0,
            "HI dipole {mag_d:.4} D is far too large — the nuclear term is \
             almost certainly using the bare Z instead of effective_z()"
        );
    }
}

#[cfg(test)]
mod proatom_tests {
    //! The smooth proatom interpolant of [`RadialProatom`]. No SCF: synthetic
    //! sums of Gaussians stand in for a spherically averaged Gaussian-basis
    //! atom (whose density is exactly such an even function of r).
    use super::*;

    /// Two-shell "atom": an even, positive sum of Gaussians and its derivative.
    /// The 0.6 tail exponent underflows the table below `TAIL_RHO_MIN` at
    /// r ≈ 27.7 Bohr, so the truncated-table path is exercised.
    fn model(r: f64) -> (f64, f64) {
        let terms = [(300.0, 60.0), (5.0, 3.0), (0.3, 0.9), (0.05, 0.6)];
        let mut v = 0.0;
        let mut d = 0.0;
        for (c, a) in terms {
            let e = c * (-a * r * r).exp();
            v += e;
            d += -2.0 * a * r * e;
        }
        (v, d)
    }

    fn table(radii: &[f64]) -> RadialProatom {
        RadialProatom::new(radii.to_vec(), radii.iter().map(|&r| model(r).0).collect()).unwrap()
    }

    fn uniform(step: f64, n: usize) -> Vec<f64> {
        (1..=n).map(|k| k as f64 * step).collect()
    }

    /// Dense probe points on [0, rmax], deliberately including every knot
    /// and r = 0.
    fn probes(p: &RadialProatom, rmax: f64) -> Vec<f64> {
        let mut out: Vec<f64> = (0..=20_000).map(|i| rmax * i as f64 / 20_000.0).collect();
        out.extend(p.radii().iter().copied().filter(|&r| r <= rmax));
        out
    }

    /// EXACTNESS ANCHOR: the interpolant passes through every tabulated value
    /// (up to the exp∘ln round trip). Catches: an off-by-one segment index, a
    /// wrong basis-function weight (A/B swapped), a table shifted by a knot.
    #[test]
    fn reproduces_the_table_at_every_knot() {
        let p = table(&scf_proatom_radii());
        let mut worst = 0.0_f64;
        for (k, (&r, &v)) in p.radii().iter().zip(p.rho()).enumerate() {
            if v <= RadialProatom::TAIL_RHO_MIN {
                break;
            }
            let rel = (p.at(r) - v).abs() / v;
            worst = worst.max(rel);
            assert!(
                rel < 1e-12,
                "knot {k} r={r}: {} vs {v} (rel {rel:.2e})",
                p.at(r)
            );
        }
        println!("max rel error at the knots: {worst:.2e}");
    }

    /// Accuracy and ORDER on a smooth analytic density: halving the step cuts
    /// the max relative error on [0, 10] Bohr by ~16 (O(h⁴)), and at the
    /// production step it is far below the piecewise-linear error.
    #[test]
    fn converges_at_fourth_order_on_an_analytic_density() {
        let err = |step: f64| -> (f64, f64) {
            let radii = uniform(step, (30.0 / step).round() as usize);
            let p = table(&radii);
            let mut e_spline = 0.0_f64;
            let mut e_linear = 0.0_f64;
            for i in 0..=10_000 {
                let r = 10.0 * i as f64 / 10_000.0;
                let t = model(r).0;
                e_spline = e_spline.max((p.at(r) - t).abs() / t);
                // The piecewise-linear interpolant (constant below radii[0]),
                // for the record.
                let lin = if r <= radii[0] {
                    model(radii[0]).0
                } else {
                    let k = ((r / step).floor() as usize).clamp(1, radii.len() - 1);
                    let (a, b) = (radii[k - 1], radii[k]);
                    let f = (r - a) / (b - a);
                    (1.0 - f) * model(a).0 + f * model(b).0
                };
                e_linear = e_linear.max((lin - t).abs() / t);
            }
            (e_spline, e_linear)
        };
        let (e1, l1) = err(0.05);
        let (e2, l2) = err(0.025);
        let (e3, _) = err(0.0125);
        let order = (e2 / e3).log2();
        println!(
            "max rel error on [0, 10], spline: h=0.05 {e1:.3e}, h=0.025 {e2:.3e}, \
             h=0.0125 {e3:.3e}, observed order {:.2} then {order:.2}; piecewise linear: \
             {l1:.3e}, {l2:.3e}",
            (e1 / e2).log2()
        );
        assert!(e1 < SPLINE_REL_ERR_005, "h = 0.05: {e1:.3e}");
        assert!((3.5..5.0).contains(&order), "observed order {order:.2}");
    }

    /// Max relative error at the production step; measured 4.6e-3 (the
    /// piecewise-linear interpolant: 1.5e-1) on this steep (exponent 60) core.
    const SPLINE_REL_ERR_005: f64 = 6e-3;

    /// `deriv` is the exact derivative of `at` EVERYWHERE — at knots, at
    /// r = 0, in the mirrored segment, across the last used knot and in the
    /// tail — by central FD, with no special-casing. Catches: a wrong
    /// moment term in y', the A/B sign of the slope, the tail slope, the
    /// mirrored-segment derivative, a dropped chain-rule factor ρ.
    #[test]
    fn deriv_matches_fd_everywhere_including_knots() {
        let p = table(&scf_proatom_radii());
        let last = p.spline.as_ref().unwrap().r.last().copied().unwrap();
        let mut pts = probes(&p, 35.0);
        pts.push(last);
        let h = 1e-6;
        let mut worst = 0.0_f64;
        for &r in &pts {
            let fd = (p.at(r + h) - p.at(r - h)) / (2.0 * h);
            let an = p.deriv(r);
            let scale = p.at(r).max(1e-300);
            let rel = (an - fd).abs() / scale;
            worst = worst.max(rel);
            assert!(rel < DERIV_FD_REL, "r={r}: deriv {an:.12e} vs FD {fd:.12e}");
        }
        println!(
            "max |deriv - FD| / rho = {worst:.2e} over {} points",
            pts.len()
        );
        assert_eq!(p.deriv(0.0), 0.0);
    }

    /// Bar on |deriv − FD(h=1e-6)| / ρ; measured 1.2e-7 (FD truncation and
    /// rounding), knots included.
    const DERIV_FD_REL: f64 = 1e-6;

    /// `ln ρ`, and with it ρ, ρ' and ρ'', is continuous across every knot
    /// (C2): the two polynomial pieces meeting at a knot agree there in
    /// value, slope and curvature. Includes the mirrored piece at r_0 and the
    /// linear tail at the last used knot. Curvature is the central difference
    /// of y' along ONE piece, exact for its quadratic y' up to rounding. A
    /// kink leaves an O(1) slope mismatch; a C1-only spline (e.g. a wrong
    /// row-0 or natural-end equation) a curvature one.
    #[test]
    fn value_slope_and_curvature_are_continuous_across_knots() {
        let p = table(&scf_proatom_radii());
        let s = p.spline.as_ref().unwrap();
        let n = s.r.len() as isize;
        let curv = |i: isize, r: f64| (s.piece(i, r + 1e-3).1 - s.piece(i, r - 1e-3).1) / 2e-3;
        let (mut j0, mut j1, mut j2) = (0.0_f64, 0.0_f64, 0.0_f64);
        for k in 0..n {
            let rk = s.r[k as usize];
            let (l, r) = (s.piece(k - 1, rk), s.piece(k, rk));
            let scale = 1.0 + l.0.abs();
            j0 = j0.max((l.0 - r.0).abs() / scale);
            j1 = j1.max((l.1 - r.1).abs() / scale);
            j2 = j2.max((curv(k - 1, rk) - curv(k, rk)).abs() / scale);
        }
        println!("max jumps across knots (/(1+|ln rho|)): ln rho {j0:.2e}, slope {j1:.2e}, curvature {j2:.2e}");
        assert!(j0 < JUMP_Y, "ln rho jump {j0:.2e}");
        assert!(j1 < JUMP_DY, "slope jump {j1:.2e}");
        assert!(j2 < JUMP_D2Y, "curvature jump {j2:.2e}");
    }

    const JUMP_Y: f64 = 1e-12;
    const JUMP_DY: f64 = 1e-9;
    const JUMP_D2Y: f64 = 1e-6;

    /// Even at the nucleus: ρ(−r) = ρ(r), ρ'(0) = 0, ρ'(r) ≈ ρ''(0) r near 0,
    /// and positive everywhere, including far beyond the table, where it decays
    /// monotonically to zero instead of jumping to it.
    #[test]
    fn even_at_the_nucleus_positive_and_decaying_in_the_tail() {
        let p = table(&scf_proatom_radii());
        for &r in &[1e-9, 0.01, 0.03, 0.05, 0.07, 1.0] {
            assert_eq!(p.at(-r).to_bits(), p.at(r).to_bits());
            assert_eq!(p.deriv(-r), -p.deriv(r));
        }
        assert_eq!(p.deriv(0.0), 0.0);
        // Below radii[0] the model is ~flat; the interpolant must be too.
        let (t0, _) = model(0.0);
        assert!((p.at(0.0) - t0).abs() / t0 < 1e-3, "{} vs {t0}", p.at(0.0));
        // Positive until exp(ln ρ) underflows f64 (ln ρ < −745), which this
        // model's steep Gaussian tail reaches ~3 Bohr past the last used knot.
        let mut prev = f64::INFINITY;
        for i in 0..=8000 {
            let r = 80.0 * i as f64 / 8000.0;
            let v = p.at(r);
            let (y, _) = p.spline.as_ref().unwrap().eval(r);
            assert!(v > 0.0 || y < -745.0, "rho({r}) = {v}, ln rho {y}");
            assert!(v >= 0.0 && v.is_finite());
            if r > 1.0 {
                assert!(v <= prev, "tail not decaying at r={r}");
            }
            prev = v;
        }
    }

    /// Non-uniform radii take the binary-search path; it must give the same
    /// interpolant as the direct index on a table where both apply, and still
    /// pass through the knots with FD-exact derivatives.
    #[test]
    fn non_uniform_and_uniform_paths_agree() {
        let p = table(&scf_proatom_radii());
        let mut q = p.clone();
        q.spline.as_mut().unwrap().uniform = None;
        assert!(p.spline.as_ref().unwrap().uniform.is_some());
        for &r in &probes(&p, 35.0) {
            let (a, b) = (p.value_and_deriv(r), q.value_and_deriv(r));
            // Both paths pick a valid piece; at a knot they may pick the two
            // neighbours, which agree to rounding in ln ρ (|ln ρ| up to ~460):
            // measured 1.4e-14 relative.
            assert!(
                (a.0 - b.0).abs() <= 1e-12 * a.0.abs(),
                "r={r}: {a:?} vs {b:?}"
            );
            assert!((a.1 - b.1).abs() <= 1e-10 * a.0.abs().max(1e-300), "r={r}");
        }
        // A genuinely non-uniform table, starting at r = 0.
        let radii: Vec<f64> = (0..400)
            .map(|k| 0.05 * k as f64 + 2e-4 * (k * k) as f64)
            .collect();
        let s = table(&radii);
        assert!(s.spline.as_ref().unwrap().uniform.is_none());
        let mut checked = 0;
        for (&r, &v) in s.radii().iter().zip(s.rho()) {
            if v <= RadialProatom::TAIL_RHO_MIN {
                break;
            }
            checked += 1;
            assert!((s.at(r) - v).abs() <= 1e-12 * v, "knot r={r}");
            let fd = (s.at(r + 1e-6) - s.at(r - 1e-6)) / 2e-6;
            assert!((s.deriv(r) - fd).abs() <= DERIV_FD_REL * v, "r={r}");
        }
        assert!(checked > 200, "{checked} knots");
        // With a knot AT r = 0 the zero slope is the clamped end condition,
        // satisfied to rounding (not by an explicit zero as for r_0 > 0).
        assert!(s.deriv(0.0).abs() <= 1e-12 * s.at(0.0), "{}", s.deriv(0.0));
    }

    /// A bare nucleus (all zeros) is the zero proatom; invalid tables are
    /// errors, not silently clamped.
    #[test]
    fn zero_table_and_invalid_tables() {
        let r = scf_proatom_radii();
        let z = RadialProatom::new(r.clone(), vec![0.0; r.len()]).unwrap();
        assert_eq!(z.value_and_deriv(1.0), (0.0, 0.0));
        let bad = |radii: Vec<f64>, rho: Vec<f64>| RadialProatom::new(radii, rho).is_err();
        assert!(bad(vec![], vec![]));
        assert!(bad(vec![1.0, 2.0], vec![1.0]));
        assert!(bad(vec![-0.1, 1.0, 2.0], vec![3.0, 2.0, 1.0]));
        assert!(bad(vec![1.0, 1.0, 2.0], vec![3.0, 2.0, 1.0]));
        assert!(bad(vec![1.0, 2.0, 3.0], vec![3.0, f64::NAN, 1.0]));
        assert!(bad(vec![1.0, 2.0, 3.0], vec![3.0, -1.0, 1.0]));
        // One usable leading value only.
        assert!(bad(vec![1.0, 2.0, 3.0], vec![3.0, 0.0, 0.0]));
        // No decaying tail: ln rho rises at the end.
        assert!(bad(vec![1.0, 2.0, 3.0], vec![1.0, 1.0, 2.0]));
        // A trailing underflow is dropped, not an error.
        let t = RadialProatom::new(vec![1.0, 2.0, 3.0, 4.0], vec![1.0, 0.1, 0.01, 0.0]).unwrap();
        assert!(t.at(4.0) > 0.0 && t.at(4.0) < 0.01);
    }
}
