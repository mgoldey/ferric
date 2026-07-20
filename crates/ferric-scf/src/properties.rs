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
//! Three otherwise-pure functions — `atomic_effective_volumes_hirshfeld`,
//! `hirshfeld_i_charges`, and `hirshfeld_charges` — were NOT moved here:
//! they depend on `ferric_export::cube::GridSpec` /
//! `ferric_export::gto_eval::eval_basis_on_grid`, and `ferric-export`
//! itself depends on `ferric-scf`, so moving them would create a Cargo
//! dependency cycle (`ferric-scf` → `ferric-export` → `ferric-scf`). They
//! remain defined in `ferric-rpa::properties` alongside the RPA-dependent
//! functions, even though they have no RPA dependency of their own.

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
                                        v_elec +=
                                            density[(o1 + i, o2 + j)] * block[i * n2 + j];
                                    }
                                }
                            } else {
                                // Off-diagonal shell pair: block covers (s1,s2); add
                                // 2× since (s2,s1) is the symmetric partner.
                                for i in 0..n1 {
                                    for j in 0..n2 {
                                        v_elec += 2.0
                                            * density[(o1 + i, o2 + j)]
                                            * block[i * n2 + j];
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
/// Per-atom effective volume via Becke partitioning:
/// ```text
///   v_A = ∫ w^A_Becke(r) ρ(r) |r − R_A|³ dr
/// ```
/// Returned in a.u. (Bohr³·e). The TS volume *ratio* is v_A / v_free[Z_A],
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
    use ferric_dft::ao_grid::eval_basis_on_points;
    use ferric_dft::grid::{build_atomic_grid, AtomicGridConfig};

    let natoms = mol.atoms.len();
    let grid_cfg = AtomicGridConfig::default();
    let grid = build_atomic_grid(mol, &grid_cfg);
    let points: Vec<[f64; 3]> = grid.iter().map(|g| g.xyz).collect();
    let weights: Vec<f64> = grid.iter().map(|g| g.weight).collect();
    let home_atom: Vec<usize> = grid.iter().map(|g| g.home_atom).collect();
    let npts = points.len();

    let chi = eval_basis_on_points(mol, obs_bs, &points).map_err(|e| {
        FerricError::General(format!(
            "atomic_effective_volumes_becke: chi eval failed: {e}"
        ))
    })?;
    let nbf = chi.nrows();

    let pos: Vec<[f64; 3]> = mol
        .atoms
        .iter()
        .map(|at| [at.x, at.y, at.zpos])
        .collect();

    let mut vol = vec![0.0_f64; natoms];
    for g in 0..npts {
        let a = home_atom[g];
        let mut rho = 0.0;
        for mu in 0..nbf {
            let cm = chi[(mu, g)];
            if cm.abs() < 1e-30 {
                continue;
            }
            for nu in 0..nbf {
                rho += density[(mu, nu)] * cm * chi[(nu, g)];
            }
        }
        let dx = points[g][0] - pos[a][0];
        let dy = points[g][1] - pos[a][1];
        let dz = points[g][2] - pos[a][2];
        let r3 = (dx * dx + dy * dy + dz * dz).powf(1.5);
        vol[a] += weights[g] * rho * r3;
    }
    Ok(vol)
}

pub fn becke_charges(
    mol: &Molecule,
    _prep: &PreparedBasis,
    obs_bs: &ferric_core::basis::BasisSet,
    density: &Array2<f64>,
) -> Result<Vec<f64>, FerricError> {
    use ferric_dft::grid::{build_atomic_grid, AtomicGridConfig};
    use ferric_dft::ao_grid::eval_basis_on_points;

    let natoms = mol.atoms.len();
    let grid = build_atomic_grid(mol, &AtomicGridConfig::default());
    let points: Vec<[f64; 3]> = grid.iter().map(|g| g.xyz).collect();
    let weights: Vec<f64> = grid.iter().map(|g| g.weight).collect();
    let home_atom: Vec<usize> = grid.iter().map(|g| g.home_atom).collect();
    let npts = points.len();

    let chi = eval_basis_on_points(mol, obs_bs, &points).map_err(|e| {
        FerricError::General(format!("becke_charges: chi eval failed: {e}"))
    })?;
    let nbf = chi.nrows();
    if density.nrows() != nbf || density.ncols() != nbf {
        return Err(FerricError::General(format!(
            "becke_charges: density shape {:?} != nbf {nbf}", density.dim()
        )));
    }

    // ρ(r_g) = Σ_μν D_μν χ_μ(g) χ_ν(g) = Σ_μ χ_μ · (D·χ)_μ
    let d_chi = density.dot(&chi);
    let mut rho = vec![0.0_f64; npts];
    for mu in 0..nbf {
        for g in 0..npts {
            rho[g] += chi[(mu, g)] * d_chi[(mu, g)];
        }
    }

    // Per-atom electron count via Becke partition:
    //   n^A = Σ_{g: home=A} w_g ρ(r_g)
    // (Becke partition baked into `w_g · 1[home=A]`.)
    let mut n_e = vec![0.0_f64; natoms];
    for g in 0..npts {
        n_e[home_atom[g]] += weights[g] * rho[g];
    }

    // Mild renormalization: rescale to enforce Σ_A n^A = N_e (corrects
    // residual grid-quadrature error of ~0.003 e on N_e=10).
    let n_target = mol.nelec() as f64;
    let n_sum: f64 = n_e.iter().sum();
    let scale = if n_sum.abs() > 1e-12 { n_target / n_sum } else { 1.0 };

    Ok((0..natoms)
        .map(|a| mol.atoms[a].z as f64 - scale * n_e[a])
        .collect())
}

/// Hirshfeld atomic charges q_A = Z_A − ∫ ρ(r) w^A(r) dr.
///
/// Uses the same Slater-proatom Hirshfeld weights as
/// [`pdep_polarizability_hirshfeld`] and a regular real-space grid, so
/// charge magnitudes and signs are directly comparable to the per-atom
/// polarizability tensors exported alongside.
///
/// # Known limitation: single-exponential proatom
///
/// The proatom density is a Slater monoexponential with ξ derived from
/// Slater's rules for the *outermost* shell. This is fine for the additive
/// partitioning needed by `pdep_polarizability_hirshfeld` (where the sum
/// rule is enforced numerically and individual α^A magnitudes are within
/// ~50% of literature). For *charges*, however, the proatom shape directly
/// sets sign and magnitude — and a single-exponential model has no core
/// peak, so it badly mis-allocates valence density for systems where
/// proatoms of similar ξ compete (e.g. C–O bonds). Until proper
/// Roothaan-Hartree-Fock spherical proatom densities (Bunge 1993) are
/// wired in, the absolute charge values exported here should be treated
/// as a *baseline for downstream CM5 correction* on small molecules only,
/// not as production-quality population analysis.
/// A spherically-averaged free-atom radial density ρ_free(r), tabulated on a
/// shared radial grid, for use as a Hirshfeld proatom. Built from an atomic SCF
/// density in the *molecule's own basis* (basis-consistent Hirshfeld weights).
#[derive(Clone)]
pub struct RadialProatom {
    /// Radii (Bohr), ascending. Shared across all atoms.
    pub radii: Vec<f64>,
    /// ρ_free at each radius (a.u.).
    pub rho: Vec<f64>,
}

impl RadialProatom {
    /// Linear-interpolate ρ_free at distance `r` (clamped/zero outside range).
    pub fn at(&self, r: f64) -> f64 {
        let n = self.radii.len();
        if n == 0 || r <= self.radii[0] {
            return self.rho.first().copied().unwrap_or(0.0);
        }
        if r >= self.radii[n - 1] {
            return 0.0; // tail beyond the grid is negligible
        }
        // Binary search for the bracketing interval.
        let mut lo = 0usize;
        let mut hi = n - 1;
        while hi - lo > 1 {
            let mid = (lo + hi) / 2;
            if self.radii[mid] <= r { lo = mid; } else { hi = mid; }
        }
        let t = (r - self.radii[lo]) / (self.radii[hi] - self.radii[lo]);
        (1.0 - t) * self.rho[lo] + t * self.rho[hi]
    }
}

/// Spherically average a single-atom density (atom at the origin) onto a radial
/// grid via Lebedev angular quadrature. `atom_density` is the atomic SCF AO
/// density matrix in `atom_bs`; the returned [`RadialProatom`] is the proatom
/// reference for Hirshfeld partitioning.
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
        atoms: vec![Atom { symbol: sym.to_string(), z, x: 0.0, y: 0.0, zpos: 0.0, ghost: false, n_core_ecp: 0 }],
        charge: 0,
        multiplicity: 1,
    };
    let (dirs, wts) = lebedev(110);
    let mut rho = vec![0.0_f64; radii.len()];
    for (ri, &r) in radii.iter().enumerate() {
        // Build the sphere of radius r and evaluate the density on it.
        let pts: Vec<[f64; 3]> = dirs.iter().map(|d| [d[0] * r, d[1] * r, d[2] * r]).collect();
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
    Ok(RadialProatom { radii: radii.to_vec(), rho })
}

/// Provider of spherically-averaged free-atom proatom densities: given element
/// `z` and integer charge state `q`, returns the radial proatom (or `None` if
/// unavailable). Built by the caller from atomic SCF in the molecule's basis.
pub type ProatomProvider<'a> = dyn Fn(i32, i32) -> Option<RadialProatom> + 'a;

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
        return Err(FerricError::General("chelpg_grid_esp: empty molecule".into()));
    }

    // Bounding box (Bohr) + margin, same convention as
    // `ferric_export::cube::GridSpec::bounding_box`.
    let mut lo = [f64::MAX; 3];
    let mut hi = [f64::MIN; 3];
    let atom_pos: Vec<[f64; 3]> = mol
        .atoms
        .iter()
        .map(|a| [a.x, a.y, a.zpos])
        .collect();
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
    let n = [2 * half_pts[0] + 1, 2 * half_pts[1] + 1, 2 * half_pts[2] + 1];
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
/// The general-purpose primitive behind [`chelpg_grid_esp`] (which supplies
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

                    let probe = [CAtom { atomic_number: 1.0, x: r[0], y: r[1], z: r[2] }];
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
/// Structurally different from [`hirshfeld_charges`]/[`lowdin_charges`]/
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
/// [`solve_chelpg_normal_equations`].
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
/// Same ESP grid ([`chelpg_grid_esp`]) and least-squares objective as
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

    solve_resp_restrained(&atom_pos, &grid, mol.charge as f64, &is_heavy, restraint_weight, b)
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
///     ξ = 1 / (R_BS in Bohr).
///
/// The Hirshfeld partition is robust to the precise proatom radial shape;
/// the sum rule inside `pdep_polarizability_hirshfeld` is the gate.
pub fn slater_xi_for_z(z: i32) -> f64 {
    // Bragg-Slater radii in Bohr (1 Å = 1.8897259886 Bohr).
    // Values from Slater J. Chem. Phys. 41, 3199 (1964) for Z=1..18.
    //
    // This is the legacy single-exponential ξ. New code should call
    // [`proatom_density_two_exp`] instead — it splits the density into a
    // tight 1s core + a diffuse Slater valence with proper Slater's-rules
    // screening per shell, which fixes the C-O charge inversion that
    // afflicts the single-exponential form (see
    // [[lowdin-over-single-exp-hirshfeld]] memory).
    //
    // Kept here for back-compat with [`pdep_polarizability_hirshfeld`]
    // which uses the renormalized additive partition and is more tolerant
    // of proatom shape errors than absolute charge analysis.
    let r_bs_ang: f64 = match z {
        1 => 0.25,  2 => 0.30,
        3 => 1.45,  4 => 1.05,  5 => 0.85,  6 => 0.70,  7 => 0.65,  8 => 0.60,
        9 => 0.50, 10 => 0.45,
        11 => 1.80, 12 => 1.50, 13 => 1.25, 14 => 1.10, 15 => 1.00, 16 => 1.00,
        17 => 1.00, 18 => 0.71,
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
    use ferric_core::basis;
    use ferric_core::parallel::ParallelContext;
    use ferric_integrals::operator::Operator;
    use crate::rhf::{solve_rhf, RhfConfig};
    use crate::screening::SchwarzBounds;

    fn build_h2() -> (Molecule, PreparedBasis, PreparedBasis, Operator, crate::result::ScfResult) {
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
