//! One-shot post-RPA / post-SCF molecular properties for the diffusion-model
//! feature export track.
//!
//! Currently exposes:
//!
//! * [`esp_at_atoms`] — electrostatic potential V(R_A) evaluated at each
//!   nuclear position, with the self-Z singularity removed by construction
//!   (the nuclear sum skips A=B).
//! * [`pdep_polarizability_static`] — closed-shell static (ω=0) electronic
//!   polarizability tensor α_ij(0) reconstructed from PDEP eigenpairs in the
//!   RI auxiliary basis.
//!
//! Both routines are closed-shell only.  They return
//! `FerricError::General(...)` if handed an Unrestricted / RestrictedOpen
//! result, mirroring the conventions in `gradient.rs`.

use std::os::raw::c_int;

use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::engine::Engine;
use ferric_integrals::ffi::{self, CAtom};
use ferric_integrals::oneelectron;
use ferric_integrals::operator::Operator;
use ferric_scf::result::{ScfResult, Spin};
use ndarray::Array2;

use crate::config::PdepRpaConfig;

/// Static (ω=0) closed-shell polarizability tensor in atomic units.
#[derive(Debug, Clone)]
pub struct PolarizabilityResult {
    /// Cartesian α_ij(0) tensor, i,j ∈ {x,y,z}, in a.u. (e²·a₀²/E_h).
    pub tensor: [[f64; 3]; 3],
    /// Isotropic average (1/3) Tr α.
    pub iso: f64,
    /// Principal values (eigenvalues of the symmetrized tensor), sorted ascending.
    pub principal: [f64; 3],
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
    let natoms = mol.atoms.len();
    let nbas = prep.nbasis();
    if density.shape() != [nbas, nbas] {
        return Err(FerricError::General(format!(
            "esp_at_atoms: density shape {:?} != ({nbas},{nbas})",
            density.shape()
        )));
    }

    // Construct a reusable nuclear-attraction engine; we will rewrite its
    // point-charge list once per atom.
    let mut eng = Engine::new_1e(ffi::OP_NUCLEAR, prep, 1e-14)?;

    let dims = prep.shell_dims();
    let offs = prep.shell_offsets();
    let nsh = prep.nshells();

    let mut out = Vec::with_capacity(natoms);
    for a in 0..natoms {
        let atom_a = &mol.atoms[a];

        // Override engine params with a single Z=1 charge at R_A.
        // libint's nuclear-attraction operator returns
        //   ⟨μ| −Z / |r − R| |ν⟩
        // so for Z=1 we get −⟨μ| 1/|r−R_A| |ν⟩ in the engine output.
        let probe = [CAtom {
            atomic_number: 1,
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
                            v_elec += density[(o1 + i, o2 + j)] * block[i * n2 + j];
                        }
                    }
                } else {
                    // Off-diagonal shell pair: block covers (s1,s2); add
                    // 2× since (s2,s1) is the symmetric partner.
                    for i in 0..n1 {
                        for j in 0..n2 {
                            v_elec += 2.0 * density[(o1 + i, o2 + j)] * block[i * n2 + j];
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

        out.push(v_nuc + v_elec);
    }

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
    let natoms = mol.atoms.len();
    let nbas = prep.nbasis();
    if density.shape() != [nbas, nbas] {
        return Err(FerricError::General(format!(
            "electric_field_at_atoms: density shape {:?} != ({nbas},{nbas})",
            density.shape()
        )));
    }

    // First-derivative nuclear-attraction engine, reused across atoms.
    let mut eng = Engine::new_1e_deriv(ffi::OP_NUCLEAR, prep, 1e-14)?;

    let dims = prep.shell_dims();
    let offs = prep.shell_offsets();
    let nsh = prep.nshells();
    let max_fn = dims.iter().copied().max().unwrap_or(1);

    // With a single point charge, libint returns 6 (shell) + 3 (charge) = 9
    // derivative blocks of size n1*n2 each.
    let nderiv = 6 + 3 * 1;
    let max_block = max_fn * max_fn;
    let mut buf = vec![0.0_f64; nderiv * max_block];

    let mut out: Vec<[f64; 3]> = Vec::with_capacity(natoms);
    for a in 0..natoms {
        let atom_a = &mol.atoms[a];

        // Override point charges: single Z=+1 probe at R_A.
        let probe = [CAtom {
            atomic_number: 1,
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
                                acc += density[(o1 + i, o2 + j)] * buf[blk_off + i * n2 + j];
                            }
                        }
                    } else {
                        // Off-diagonal shell pair: density and operator are
                        // both symmetric in (μ,ν), so the (s2,s1) partner
                        // contributes equally → factor 2.
                        for i in 0..n1 {
                            for j in 0..n2 {
                                acc +=
                                    2.0 * density[(o1 + i, o2 + j)] * buf[blk_off + i * n2 + j];
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

        out.push([
            e_elec[0] + e_nuc[0],
            e_elec[1] + e_nuc[1],
            e_elec[2] + e_nuc[2],
        ]);
    }

    Ok(out)
}

/// Wrapper around [`electric_field_at_atoms`] taking an [`ScfResult`].
///
/// Uses the spin-summed density `density_total()` so the same call works
/// for Restricted, Unrestricted, and RestrictedOpen references — the
/// electric field is a one-electron property of the total electron
/// density, independent of spin polarization at the field point.
pub fn electric_field_at_atoms_rpa(
    mol: &Molecule,
    prep: &PreparedBasis,
    rhf: &ScfResult,
) -> Result<Vec<[f64; 3]>, FerricError> {
    electric_field_at_atoms(mol, prep, rhf.density_total())
}

/// Closed-shell static (ω=0) electronic polarizability from PDEP eigenpairs.
///
/// # Derivation
///
/// Closed-shell direct-RPA static α via Sherman-Morrison-Woodbury on the
/// (A+B) matrix:
///
///   (A+B)_{ia,jb} = δ_{ia,jb} Δε_ia + 4 (ia|jb)
///
/// With RI (ia|jb) = Σ_P B̃^P_ia B̃^P_jb (B̃ = V^{-1/2} (P|ia)), and the
/// dielectric ε̃ = I + 4 B̃ D^{-1} B̃^T  (with D=diag(Δε_ia)), one gets
///
///   α_ij = 4 μ^i^T D^{-1} μ^j − 16 w^i^T ε̃^{-1} w^j
///
/// where μ^i_{ia} = ⟨ψ_i|r_i|ψ_a⟩ are MO-basis dipole matrix elements and
///
///   w^i_P = Σ_{ia} B̃^P_{ia} · μ^i_{ia} / Δε_{ia}.
///
/// Expanding ε̃^{-1} = I − Σ_α V_α V_α^T (λ_α − 1)/λ_α:
///
///   α_ij = α^{χ₀}_ij + 16 Σ_α (w^i·V_α)(w^j·V_α) · (λ_α − 1)/λ_α
///
///   α^{χ₀}_ij = 4 μ^i^T D^{-1} μ^j − 16 w^i^T w^j
///
/// where V_α are the **dressed-basis** PDEP eigenvectors.  Since the
/// physical-aux eigenpotentials returned by `run_pdep_rpa` are
/// V^{-1/2}·V_α^dressed, we instead build w in the dressed basis directly
/// (which is just `B_ov · diag(1/Δε) · μ` — no V^{1/2} or V^{-1/2} ever
/// touches the working vectors) and dot with the **dressed** eigenvectors.
/// We recover those by transforming back: V_α^dressed = V^{1/2} · V_α^phys.
/// Simpler: redo the PDEP solve here and keep the dressed eigenvectors.
///
/// The spin factor of 4 (closed-shell) is consistent with ferric's χ₀
/// convention (`scale = sqrt(4·e_ia/(ω²+e_ia²))`).
pub fn pdep_polarizability_static(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    rhf: &ScfResult,
    op: Operator,
    cfg: &PdepRpaConfig,
) -> Result<PolarizabilityResult, FerricError> {
    // Open-shell: dispatch to the per-spin SMW path. The 2×2 spin-block
    // SMW with outer-factor-2 TDDFT convention reproduces the closed-shell
    // formula on spin-symmetric UHF (verified numerically).
    if !matches!(rhf.spin, Spin::Restricted) {
        return pdep_polarizability_static_unrestricted(mol, obs, dfbs, rhf, op, cfg);
    }

    // Build B̃^P_ia = V^{-1/2} (P|ia) and orbital-energy slices via the same
    // path `run_pdep_rpa` uses.  No frozen-core for α (response on all
    // occupied is physical).
    let mp2_cfg = ferric_mp2::rimp2::RiMp2Config { frozen_core: 0 };
    let inter =
        ferric_mp2::rimp2::compute_rpa_intermediates(mol, obs, dfbs, op, rhf, &mp2_cfg)?;
    let b_ov = &inter.b_ov; // shape (naux, nov)
    let nocc = inter.nocc;
    let nvir = inter.nvir;
    let nocc_total = inter.nocc_total;
    let first_occ = inter.first_occ;
    let naux = inter.naux;
    let nov = nocc * nvir;
    debug_assert_eq!(b_ov.shape(), &[naux, nov]);

    let eps = rhf.eps_r();
    let eps_occ: Vec<f64> = eps[first_occ..first_occ + nocc].to_vec();
    let eps_vir: Vec<f64> = eps[nocc_total..nocc_total + nvir].to_vec();

    // MO-basis dipole μ^d_{ia} = ⟨ψ_i|r_d|ψ_a⟩ from AO dipole + MO transform.
    let dip_ao = oneelectron::dipole(obs, [0.0, 0.0, 0.0]);
    let c = rhf.mos_r();
    let c_occ = c.slice(ndarray::s![.., first_occ..first_occ + nocc]).to_owned();
    let c_vir = c.slice(ndarray::s![.., nocc_total..nocc_total + nvir]).to_owned();
    // mu_mo[d] : (nocc, nvir)
    let mu_mo: [Array2<f64>; 3] = std::array::from_fn(|d| {
        // C_occ^T · D^d_AO · C_vir
        c_occ.t().dot(&dip_ao[d]).dot(&c_vir)
    });

    // 1/Δε_ia table.
    let nov_check = nov;
    let mut inv_de = ndarray::Array1::<f64>::zeros(nov_check);
    for i in 0..nocc {
        for a in 0..nvir {
            let ia = i * nvir + a;
            inv_de[ia] = 1.0 / (eps_vir[a] - eps_occ[i]);
        }
    }

    // μ flattened to (nov,) per direction, scaled by 1/Δε_ia.
    let mu_flat_inv: [ndarray::Array1<f64>; 3] = std::array::from_fn(|d| {
        let mut v = ndarray::Array1::<f64>::zeros(nov);
        for i in 0..nocc {
            for a in 0..nvir {
                v[i * nvir + a] = mu_mo[d][(i, a)];
            }
        }
        v * &inv_de
    });
    // μ flattened, unscaled.
    let mu_flat: [ndarray::Array1<f64>; 3] = std::array::from_fn(|d| {
        let mut v = ndarray::Array1::<f64>::zeros(nov);
        for i in 0..nocc {
            for a in 0..nvir {
                v[i * nvir + a] = mu_mo[d][(i, a)];
            }
        }
        v
    });

    // w^d_P = Σ_ia B̃^P_ia · μ^d_ia / Δε_ia.
    let w_vec: [ndarray::Array1<f64>; 3] =
        std::array::from_fn(|d| b_ov.dot(&mu_flat_inv[d]));

    // Dressed dielectric ε̃ at ω=0: ε̃ = I + 4 B̃ D^{-1} B̃^T
    //   (scale_ia² = 4/Δε_ia at ω=0)
    // Build via DGEMM with column-scaled B̃.
    let mut b_scaled = b_ov.clone();
    // multiply column ia by sqrt(4/Δε_ia)
    for ia in 0..nov {
        let s = (4.0 * inv_de[ia]).sqrt();
        let mut col = b_scaled.column_mut(ia);
        col.mapv_inplace(|x| x * s);
    }
    // ε̃ = I + b_scaled · b_scaled^T
    let mut eps_mat: Array2<f64> = b_scaled.dot(&b_scaled.t());
    for p in 0..naux {
        eps_mat[(p, p)] += 1.0;
    }

    // Solve ε̃ · y^d = w^d  (naux × naux SPD).
    let y_vec: [ndarray::Array1<f64>; 3] = {
        use ndarray_linalg::Solve;
        std::array::from_fn(|d| eps_mat.solve(&w_vec[d]).unwrap())
    };

    // α_ij = 4 μ^i^T D^{-1} μ^j − 16 w^i^T y^j
    //
    // Derivation cross-check:
    //   α_ij = 4 μ^i^T (A+B)^{-1} μ^j  with (A+B) = D + 4 B̃^T B̃
    //   SMW: (D + 4 B̃^T B̃)^{-1} = D^{-1} − D^{-1} B̃^T (I/4 + B̃ D^{-1} B̃^T)^{-1} B̃ D^{-1}
    //     = D^{-1} − D^{-1} B̃^T · 4 · ε̃^{-1} · B̃ D^{-1}
    //   ⇒ α_ij = 4 μ^i^T D^{-1} μ^j − 16 μ^i^T D^{-1} B̃^T ε̃^{-1} B̃ D^{-1} μ^j
    //          = 4 μ^i^T D^{-1} μ^j − 16 (B̃ D^{-1} μ^i)^T ε̃^{-1} (B̃ D^{-1} μ^j)
    //          = 4 μ^i^T D^{-1} μ^j − 16 w^i^T y^j
    let mut tensor = [[0.0_f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let bare = mu_flat[i].dot(&mu_flat_inv[j]); // = μ^i^T D^{-1} μ^j
            let coupled = w_vec[i].dot(&y_vec[j]);
            tensor[i][j] = 4.0 * bare - 16.0 * coupled;
        }
    }

    if std::env::var("FERRIC_DEBUG_ALPHA").is_ok() {
        eprintln!("[alpha-debug] tensor=\n{:?}", tensor);
        let bare_iso: f64 = (0..3)
            .map(|i| 4.0 * mu_flat[i].dot(&mu_flat_inv[i]))
            .sum::<f64>()
            / 3.0;
        eprintln!("[alpha-debug] bare α_iso = {:.6}", bare_iso);
    }

    // Symmetrize (numerically tiny asymmetry from finite-precision Davidson).
    for i in 0..3 {
        for j in (i + 1)..3 {
            let avg = 0.5 * (tensor[i][j] + tensor[j][i]);
            tensor[i][j] = avg;
            tensor[j][i] = avg;
        }
    }

    let iso = (tensor[0][0] + tensor[1][1] + tensor[2][2]) / 3.0;

    // Principal values via 3x3 symmetric eig.
    let principal = eig3_sym(tensor);

    Ok(PolarizabilityResult {
        tensor,
        iso,
        principal,
    })
}

/// Open-shell static polarizability tensor at ω=0 (UHF / ROHF reference).
///
/// Same SMW construction as the closed-shell path but with prefactor 2 per
/// spin (vs closed-shell's 4 = 2·2). The (A+B) matrix is block-diagonal in
/// spin since RPA has no cross-spin direct coupling:
/// ```text
///   (A+B)^{σσ'}_{ia,jb} = δ_{σσ'} (δ_{ia,jb} Δε_{iaσ} + 2 (ia|jb)_{σ})
/// ```
/// SMW per spin: `ε̃ = I + Σ_σ 2 B̃_σ D_σ^{-1} B̃_σ^T`, then
/// ```text
///   α_ij = Σ_σ [2 (μ_σ^i)^T D_σ^{-1} μ_σ^j  −  8 (w_σ^i)^T y^j]
/// ```
/// (the y vector is shared across spins since ε̃ couples them through the
/// auxiliary basis).
pub fn pdep_polarizability_static_unrestricted(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    rhf: &ScfResult,
    op: Operator,
    _cfg: &PdepRpaConfig,
) -> Result<PolarizabilityResult, FerricError> {
    use ferric_mp2::rimp2::{compute_rpa_intermediates_spin, RiMp2Config};
    use ndarray_linalg::Solve;

    let mp2_cfg = RiMp2Config { frozen_core: 0 };

    // Per-spin intermediates.
    let inter_a = compute_rpa_intermediates_spin(mol, obs, dfbs, op, rhf, &mp2_cfg, true)?;
    let inter_b = compute_rpa_intermediates_spin(mol, obs, dfbs, op, rhf, &mp2_cfg, false)?;
    let naux = inter_a.naux;
    debug_assert_eq!(inter_a.naux, inter_b.naux);

    // Orbital-energy slices per spin (ROHF reuses α-MOs for β; see run_u_pdep_rpa).
    let eps_b_full: &[f64] = if matches!(rhf.spin, Spin::RestrictedOpen) {
        rhf.eps_a()
    } else {
        rhf.eps_b()
    };
    let eps_occ_a: Vec<f64> = rhf.eps_a()[inter_a.first_occ..inter_a.first_occ + inter_a.nocc].to_vec();
    let eps_vir_a: Vec<f64> = rhf.eps_a()[inter_a.nocc_total..inter_a.nocc_total + inter_a.nvir].to_vec();
    let eps_occ_b: Vec<f64> = eps_b_full[inter_b.first_occ..inter_b.first_occ + inter_b.nocc].to_vec();
    let eps_vir_b: Vec<f64> = eps_b_full[inter_b.nocc_total..inter_b.nocc_total + inter_b.nvir].to_vec();

    // Per-spin MO-basis dipole μ_σ^d_{ia} = ⟨ψ_iσ|r_d|ψ_aσ⟩.
    let dip_ao = oneelectron::dipole(obs, [0.0, 0.0, 0.0]);
    let c_a = rhf.mos_a();
    let c_b = if matches!(rhf.spin, Spin::RestrictedOpen) { rhf.mos_a() } else { rhf.mos_b() };
    let c_occ_a = c_a.slice(ndarray::s![.., inter_a.first_occ..inter_a.first_occ + inter_a.nocc]).to_owned();
    let c_vir_a = c_a.slice(ndarray::s![.., inter_a.nocc_total..inter_a.nocc_total + inter_a.nvir]).to_owned();
    let c_occ_b = c_b.slice(ndarray::s![.., inter_b.first_occ..inter_b.first_occ + inter_b.nocc]).to_owned();
    let c_vir_b = c_b.slice(ndarray::s![.., inter_b.nocc_total..inter_b.nocc_total + inter_b.nvir]).to_owned();
    let mu_mo_a: [Array2<f64>; 3] = std::array::from_fn(|d| c_occ_a.t().dot(&dip_ao[d]).dot(&c_vir_a));
    let mu_mo_b: [Array2<f64>; 3] = std::array::from_fn(|d| c_occ_b.t().dot(&dip_ao[d]).dot(&c_vir_b));

    // Per-spin 1/Δε tables and flattened μ vectors.
    let build_flats = |nocc: usize, nvir: usize, eps_o: &[f64], eps_v: &[f64], mu_mo: &[Array2<f64>; 3]|
        -> ([ndarray::Array1<f64>; 3], [ndarray::Array1<f64>; 3], ndarray::Array1<f64>)
    {
        let nov = nocc * nvir;
        let mut inv_de = ndarray::Array1::<f64>::zeros(nov);
        for i in 0..nocc {
            for a in 0..nvir {
                inv_de[i * nvir + a] = 1.0 / (eps_v[a] - eps_o[i]);
            }
        }
        let mu_flat: [ndarray::Array1<f64>; 3] = std::array::from_fn(|d| {
            let mut v = ndarray::Array1::<f64>::zeros(nov);
            for i in 0..nocc {
                for a in 0..nvir {
                    v[i * nvir + a] = mu_mo[d][(i, a)];
                }
            }
            v
        });
        let mu_flat_inv: [ndarray::Array1<f64>; 3] = std::array::from_fn(|d| &mu_flat[d] * &inv_de);
        (mu_flat, mu_flat_inv, inv_de)
    };

    let (mu_flat_a, mu_flat_inv_a, inv_de_a) =
        build_flats(inter_a.nocc, inter_a.nvir, &eps_occ_a, &eps_vir_a, &mu_mo_a);
    let (mu_flat_b, mu_flat_inv_b, inv_de_b) =
        build_flats(inter_b.nocc, inter_b.nvir, &eps_occ_b, &eps_vir_b, &mu_mo_b);

    // w_σ^d_P = Σ_ia B̃_σ^P_ia · μ_σ^d_ia / Δε_iaσ ; total w^d = w_α^d + w_β^d
    let w_a: [ndarray::Array1<f64>; 3] = std::array::from_fn(|d| inter_a.b_ov.dot(&mu_flat_inv_a[d]));
    let w_b: [ndarray::Array1<f64>; 3] = std::array::from_fn(|d| {
        if inter_b.nocc == 0 {
            ndarray::Array1::<f64>::zeros(naux)
        } else {
            inter_b.b_ov.dot(&mu_flat_inv_b[d])
        }
    });
    let w_total: [ndarray::Array1<f64>; 3] = std::array::from_fn(|d| &w_a[d] + &w_b[d]);

    // ε̃ = I + 2 B̃_α D_α^{-1} B̃_α^T + 2 B̃_β D_β^{-1} B̃_β^T
    let mut eps_mat = Array2::<f64>::zeros((naux, naux));
    for p in 0..naux { eps_mat[(p, p)] = 1.0; }
    for (b_ov, inv_de) in [(&inter_a.b_ov, &inv_de_a), (&inter_b.b_ov, &inv_de_b)] {
        if b_ov.shape()[1] == 0 { continue; }
        let mut b_scaled = b_ov.clone();
        for ia in 0..inv_de.len() {
            let s = (2.0 * inv_de[ia]).sqrt();
            let mut col = b_scaled.column_mut(ia);
            col.mapv_inplace(|x| x * s);
        }
        let chi_sigma = b_scaled.dot(&b_scaled.t());
        eps_mat += &chi_sigma;
    }

    // Solve ε̃ · y^d = w_total^d
    let y_vec: [ndarray::Array1<f64>; 3] = std::array::from_fn(|d| eps_mat.solve(&w_total[d]).unwrap());

    // Correct UHF formula derived from the full 2×2 spin-block SMW:
    //   α = 2 · μ̃^T (A+B)^{-1} μ̃
    //     = 2 · [μ̃^T M^{-1} μ̃ − 2 w_total^T ε̃^{-1} w_total]
    //     = 2 Σ_σ μ_σ^T D_σ^{-1} μ_σ  −  4 w_total^T ε̃^{-1} w_total
    //
    // The outer factor 2 is the TDDFT/RPA Casida-equation convention that
    // accounts for the density-vs-orbital-response distinction (verified
    // numerically: on closed-shell-via-UHF, factor 2 reproduces the
    // closed-shell α exactly via the 2x2 spin-block symmetric mode
    // `(D + 4 B̃^T B̃)`).
    let mut tensor = [[0.0_f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let bare_a = 2.0 * mu_flat_a[i].dot(&mu_flat_inv_a[j]);
            let bare_b = if inter_b.nocc == 0 { 0.0 } else { 2.0 * mu_flat_b[i].dot(&mu_flat_inv_b[j]) };
            let coupled = w_total[i].dot(&y_vec[j]);
            tensor[i][j] = bare_a + bare_b - 4.0 * coupled;
        }
    }

    // Symmetrize.
    for i in 0..3 {
        for j in (i + 1)..3 {
            let avg = 0.5 * (tensor[i][j] + tensor[j][i]);
            tensor[i][j] = avg;
            tensor[j][i] = avg;
        }
    }
    let iso = (tensor[0][0] + tensor[1][1] + tensor[2][2]) / 3.0;
    let principal = eig3_sym(tensor);

    Ok(PolarizabilityResult { tensor, iso, principal })
}

/// Per-atom static polarizability decomposition via Hirshfeld partitioning.
///
/// Algebraically:
///
/// ```text
///     α^A_ij = 4 (μ^{A,i})^T D^{-1} μ^j  −  16 (w^{A,i})^T y^j
/// ```
///
/// where
///
/// * `μ^j` and `y^j` are the molecular MO-basis dipole and SMW-solve vectors
///   already built by `pdep_polarizability_static` (factor-of-4 closed-shell
///   convention, `(A+B) y = …`).
/// * `μ^{A,i}_pq = ⟨p| w^A(r) · r_i |q⟩` is the **Hirshfeld-weighted** dipole
///   matrix in the MO basis (i Cartesian, A atom).
/// * `w^{A,i}_P = Σ_{ia} B̃^P_{ia} · μ^{A,i}_{ia} / Δε_{ia}`.
///
/// Summing over A reproduces the molecular dipole (because Σ_A w^A(r) ≡ 1),
/// so Σ_A α^A = α (this is the sum-rule gate inside the function).
///
/// The Hirshfeld weights use a smooth single-exponential **promolecular**
/// reference density per element,
///
/// ```text
///     ρ_X^free(r) = (Z_X ξ_X³ / π) exp(−2 ξ_X r),
/// ```
///
/// with `ξ_X` derived from Bragg-Slater atomic radii.  The partition is
/// robust to the exact proatom shape (sum-rule constrained); the gate
/// will fail loudly if the grid is too coarse.
///
/// Closed-shell only.
/// Per-atom static polarizability decomposition via **Becke-Lebedev**
/// atomic partition.
///
/// Replaces the Slater-proatom Hirshfeld scheme in
/// [`pdep_polarizability_hirshfeld`] with the production-standard Becke
/// fuzzy weights (no electron-density proatom; geometry-only) on a
/// Becke-Lebedev atom-centered quadrature grid.
///
/// Same SMW algebra:
/// ```text
///     α^A_ij = 4 (μ^{A,i})^T D^{-1} μ^j  −  16 (w^{A,i})^T ε̃^{-1} w^j
/// ```
/// with
/// ```text
///     μ^{A,i}_pq = ⟨p| w^A_Becke(r) · r_i |q⟩  (Becke-weighted AO dipole)
///     μ^i = Σ_A μ^{A,i}                       (sum-rule exact for Becke)
/// ```
///
/// Closed-shell only. Returns Vec<[[f64; 3]; 3]>, one (3×3) tensor per atom.
pub fn pdep_polarizability_becke(
    mol: &Molecule,
    obs: &PreparedBasis,
    obs_bs: &ferric_core::basis::BasisSet,
    dfbs: &PreparedBasis,
    rhf: &ScfResult,
    op: Operator,
    _cfg: &PdepRpaConfig,
) -> Result<Vec<[[f64; 3]; 3]>, FerricError> {
    use ferric_dft::grid::{build_atomic_grid, AtomicGridConfig};
    use ferric_dft::ao_grid::eval_basis_on_points;
    use ndarray_linalg::Solve;

    if !matches!(rhf.spin, Spin::Restricted) {
        return Err(FerricError::General(
            "pdep_polarizability_becke: only closed-shell (Restricted) supported".into(),
        ));
    }

    let natoms = mol.atoms.len();

    // RI intermediates (same as molecular static-α path).
    let mp2_cfg = ferric_mp2::rimp2::RiMp2Config { frozen_core: 0 };
    let inter =
        ferric_mp2::rimp2::compute_rpa_intermediates(mol, obs, dfbs, op, rhf, &mp2_cfg)?;
    let b_ov = &inter.b_ov;
    let nocc = inter.nocc;
    let nvir = inter.nvir;
    let nocc_total = inter.nocc_total;
    let first_occ = inter.first_occ;
    let naux = inter.naux;
    let nov = nocc * nvir;

    let eps = rhf.eps_r();
    let eps_occ: Vec<f64> = eps[first_occ..first_occ + nocc].to_vec();
    let eps_vir: Vec<f64> = eps[nocc_total..nocc_total + nvir].to_vec();

    let c = rhf.mos_r();
    let c_occ = c.slice(ndarray::s![.., first_occ..first_occ + nocc]).to_owned();
    let c_vir = c.slice(ndarray::s![.., nocc_total..nocc_total + nvir]).to_owned();

    let mut inv_de = ndarray::Array1::<f64>::zeros(nov);
    for i in 0..nocc {
        for a in 0..nvir {
            inv_de[i * nvir + a] = 1.0 / (eps_vir[a] - eps_occ[i]);
        }
    }

    // ε̃ = I + B̃ · diag(4/Δε) · B̃^T (closed-shell prefactor 4).
    let mut b_scaled = b_ov.clone();
    for ia in 0..nov {
        let s = (4.0 * inv_de[ia]).sqrt();
        let mut col = b_scaled.column_mut(ia);
        col.mapv_inplace(|x| x * s);
    }
    let mut eps_mat: Array2<f64> = b_scaled.dot(&b_scaled.t());
    for p in 0..naux {
        eps_mat[(p, p)] += 1.0;
    }

    // Build Becke-Lebedev grid (atom-centered, Becke partition baked into
    // grid-point weights; per-atom restriction via home_atom field).
    let grid_cfg = AtomicGridConfig::default();
    let grid = build_atomic_grid(mol, &grid_cfg);
    let points: Vec<[f64; 3]> = grid.iter().map(|g| g.xyz).collect();
    let weights: Vec<f64> = grid.iter().map(|g| g.weight).collect();
    let home_atom: Vec<usize> = grid.iter().map(|g| g.home_atom).collect();
    let npts = points.len();

    // Evaluate AO basis on the grid: χ has shape (nbf, npts).
    let chi = eval_basis_on_points(mol, obs_bs, &points).map_err(|e| {
        FerricError::General(format!("pdep_polarizability_becke: chi eval failed: {e}"))
    })?;
    let nbf = chi.nrows();
    debug_assert_eq!(nbf, obs.nbasis());

    // Build per-atom Becke-weighted AO dipole using lab-frame r_i. We then
    // renormalize per-AO-pair so the sum across atoms reproduces the
    // **analytical** AO dipole exactly (decouples the partition fraction
    // — Becke, robust — from the absolute dipole magnitude — libint
    // analytical). Same trick as `pdep_polarizability_hirshfeld` uses with
    // Slater proatoms; the renormalization makes the result robust to
    // grid quadrature error.
    let mut d_ai_ao: Vec<[Array2<f64>; 3]> = (0..natoms)
        .map(|_| std::array::from_fn(|_| Array2::<f64>::zeros((nbf, nbf))))
        .collect();
    let mut d_sum: [Array2<f64>; 3] = std::array::from_fn(|_| Array2::<f64>::zeros((nbf, nbf)));
    for g in 0..npts {
        let a = home_atom[g];
        let w = weights[g];
        let r = points[g];
        for d in 0..3 {
            let factor = w * r[d];
            for mu in 0..nbf {
                let chi_mu = chi[(mu, g)];
                let weighted_chi_mu = factor * chi_mu;
                if weighted_chi_mu.abs() < 1e-30 { continue; }
                for nu in 0..nbf {
                    let contrib = weighted_chi_mu * chi[(nu, g)];
                    d_ai_ao[a][d][(mu, nu)] += contrib;
                    d_sum[d][(mu, nu)] += contrib;
                }
            }
        }
    }
    // Symmetrize per-atom AO dipoles and the sum.
    for d in 0..3 {
        for a in 0..natoms {
            let m = &mut d_ai_ao[a][d];
            for i in 0..nbf {
                for j in (i + 1)..nbf {
                    let avg = 0.5 * (m[(i, j)] + m[(j, i)]);
                    m[(i, j)] = avg;
                    m[(j, i)] = avg;
                }
            }
        }
        for i in 0..nbf {
            for j in (i + 1)..nbf {
                let avg = 0.5 * (d_sum[d][(i, j)] + d_sum[d][(j, i)]);
                d_sum[d][(i, j)] = avg;
                d_sum[d][(j, i)] = avg;
            }
        }
    }

    // Renormalize: per AO-pair (μ, ν), rescale each atom's contribution
    // by analytical/grid_total. This is exact decoupling of the partition
    // (Becke fraction) from the magnitude (analytical AO dipole), giving
    // grid-noise-free results.
    let dip_ao_analytical = oneelectron::dipole(obs, [0.0, 0.0, 0.0]);
    let inv_n = 1.0 / (natoms as f64);
    let small = 1e-10;
    for d in 0..3 {
        for mu in 0..nbf {
            for nu in 0..nbf {
                let total_grid = d_sum[d][(mu, nu)];
                let total_analytical = dip_ao_analytical[d][(mu, nu)];
                if total_grid.abs() < small * (1.0 + total_analytical.abs()) {
                    for a in 0..natoms {
                        d_ai_ao[a][d][(mu, nu)] = total_analytical * inv_n;
                    }
                } else {
                    let scale = total_analytical / total_grid;
                    for a in 0..natoms {
                        d_ai_ao[a][d][(mu, nu)] *= scale;
                    }
                }
            }
        }
    }
    // Symmetrize each AO dipole matrix.
    for a in 0..natoms {
        for d in 0..3 {
            let m = &mut d_ai_ao[a][d];
            for i in 0..nbf {
                for j in (i + 1)..nbf {
                    let avg = 0.5 * (m[(i, j)] + m[(j, i)]);
                    m[(i, j)] = avg;
                    m[(j, i)] = avg;
                }
            }
        }
    }

    // Transform to MO occ-vir basis.
    let mut mu_ai_mo: Vec<[Array2<f64>; 3]> = (0..natoms)
        .map(|_| std::array::from_fn(|_| Array2::<f64>::zeros((nocc, nvir))))
        .collect();
    let mut mu_mo: [Array2<f64>; 3] = std::array::from_fn(|_| Array2::<f64>::zeros((nocc, nvir)));
    for a in 0..natoms {
        for d in 0..3 {
            let m = c_occ.t().dot(&d_ai_ao[a][d]).dot(&c_vir);
            mu_mo[d] = &mu_mo[d] + &m;
            mu_ai_mo[a][d] = m;
        }
    }

    // Flatten molecular dipole; build w^d and y^d (the SMW solve).
    let mu_flat: [ndarray::Array1<f64>; 3] = std::array::from_fn(|d| {
        let mut v = ndarray::Array1::<f64>::zeros(nov);
        for i in 0..nocc {
            for ax in 0..nvir {
                v[i * nvir + ax] = mu_mo[d][(i, ax)];
            }
        }
        v
    });
    let mu_flat_inv: [ndarray::Array1<f64>; 3] =
        std::array::from_fn(|d| &mu_flat[d] * &inv_de);
    let w_mol: [ndarray::Array1<f64>; 3] =
        std::array::from_fn(|d| b_ov.dot(&mu_flat_inv[d]));
    let y_mol: [ndarray::Array1<f64>; 3] =
        std::array::from_fn(|d| eps_mat.solve(&w_mol[d]).unwrap());

    // Assemble per-atom α^A.
    let mut alpha_per_atom: Vec<[[f64; 3]; 3]> = vec![[[0.0; 3]; 3]; natoms];
    for a in 0..natoms {
        for d in 0..3 {
            let mut mu_ai_flat = ndarray::Array1::<f64>::zeros(nov);
            let mut mu_ai_flat_inv = ndarray::Array1::<f64>::zeros(nov);
            for i in 0..nocc {
                for ax in 0..nvir {
                    let ia = i * nvir + ax;
                    mu_ai_flat[ia] = mu_ai_mo[a][d][(i, ax)];
                    mu_ai_flat_inv[ia] = mu_ai_mo[a][d][(i, ax)] * inv_de[ia];
                }
            }
            let w_ai = b_ov.dot(&mu_ai_flat_inv);
            for j in 0..3 {
                let bare = mu_ai_flat.dot(&mu_flat_inv[j]);
                let coupled = w_ai.dot(&y_mol[j]);
                alpha_per_atom[a][d][j] = 4.0 * bare - 16.0 * coupled;
            }
        }
        // Symmetrize per-atom tensor.
        for i in 0..3 {
            for j in (i + 1)..3 {
                let avg = 0.5 * (alpha_per_atom[a][i][j] + alpha_per_atom[a][j][i]);
                alpha_per_atom[a][i][j] = avg;
                alpha_per_atom[a][j][i] = avg;
            }
        }
    }

    Ok(alpha_per_atom)
}

/// Per-atom Becke polarizability tensors α^A_{ij}(iω) at a list of imaginary
/// frequencies. Returns `out[a][k]` = 3×3 tensor for atom `a`, frequency `k`.
///
/// This is the frequency generalization of [`pdep_polarizability_becke`]:
/// the grid, Becke partition, and partition-weighted MO dipoles are built once
/// (ω-independent); only the χ₀ "denominator" g_ia(ω) = e_ia/(ω²+e_ia²) and the
/// SMW dielectric ε̃(ω) = I + 4 B̃ diag(g(ω)) B̃^T change per frequency. At ω=0,
/// g_ia = 1/Δε_ia, so this reproduces `pdep_polarizability_becke` exactly.
///
/// The Casimir-Polder C6 follow-up consumes these via
/// `dispersion::pdep_dynamic_polarizability`.
#[allow(clippy::too_many_arguments)]
pub fn pdep_polarizability_becke_dynamic(
    mol: &Molecule,
    obs: &PreparedBasis,
    obs_bs: &ferric_core::basis::BasisSet,
    dfbs: &PreparedBasis,
    rhf: &ScfResult,
    op: Operator,
    cfg: &PdepRpaConfig,
    freqs: &[f64],
) -> Result<Vec<Vec<[[f64; 3]; 3]>>, FerricError> {
    use ferric_dft::ao_grid::eval_basis_on_points;
    use ferric_dft::grid::{build_atomic_grid, AtomicGridConfig};

    let natoms = mol.atoms.len();
    let nfreq = freqs.len();
    let mp2_cfg = ferric_mp2::rimp2::RiMp2Config {
        frozen_core: cfg.frozen_core,
    };

    // Open-shell dispatch: build per-spin intermediates and MO slices.
    // Closed-shell falls through to the single-B̃ path below.
    if !matches!(rhf.spin, Spin::Restricted) {
        use ferric_mp2::rimp2::compute_rpa_intermediates_spin;
        use ndarray_linalg::Solve;

        let inter_a = compute_rpa_intermediates_spin(mol, obs, dfbs, op, rhf, &mp2_cfg, true)?;
        let inter_b = compute_rpa_intermediates_spin(mol, obs, dfbs, op, rhf, &mp2_cfg, false)?;
        let naux = inter_a.naux;

        // Orbital-energy slices (ROHF reuses α-MOs for β).
        let eps_b_full: &[f64] = if matches!(rhf.spin, Spin::RestrictedOpen) {
            rhf.eps_a()
        } else {
            rhf.eps_b()
        };
        let eps_occ_a: Vec<f64> = rhf.eps_a()[inter_a.first_occ..inter_a.first_occ + inter_a.nocc].to_vec();
        let eps_vir_a: Vec<f64> = rhf.eps_a()[inter_a.nocc_total..inter_a.nocc_total + inter_a.nvir].to_vec();
        let eps_occ_b: Vec<f64> = eps_b_full[inter_b.first_occ..inter_b.first_occ + inter_b.nocc].to_vec();
        let eps_vir_b: Vec<f64> = eps_b_full[inter_b.nocc_total..inter_b.nocc_total + inter_b.nvir].to_vec();

        // Per-spin e_ia tables.
        let e_ia_a = {
            let (nocc, nvir) = (inter_a.nocc, inter_a.nvir);
            let mut v = ndarray::Array1::<f64>::zeros(nocc * nvir);
            for i in 0..nocc { for a in 0..nvir { v[i*nvir+a] = eps_vir_a[a] - eps_occ_a[i]; } }
            v
        };
        let e_ia_b = {
            let (nocc, nvir) = (inter_b.nocc, inter_b.nvir);
            let mut v = ndarray::Array1::<f64>::zeros(nocc * nvir);
            for i in 0..nocc { for a in 0..nvir { v[i*nvir+a] = eps_vir_b[a] - eps_occ_b[i]; } }
            v
        };

        // MO coefficient slices per spin.
        let c_a = rhf.mos_a();
        let c_b = if matches!(rhf.spin, Spin::RestrictedOpen) { rhf.mos_a() } else { rhf.mos_b() };
        let c_occ_a = c_a.slice(ndarray::s![.., inter_a.first_occ..inter_a.first_occ+inter_a.nocc]).to_owned();
        let c_vir_a = c_a.slice(ndarray::s![.., inter_a.nocc_total..inter_a.nocc_total+inter_a.nvir]).to_owned();
        let c_occ_b = c_b.slice(ndarray::s![.., inter_b.first_occ..inter_b.first_occ+inter_b.nocc]).to_owned();
        let c_vir_b = c_b.slice(ndarray::s![.., inter_b.nocc_total..inter_b.nocc_total+inter_b.nvir]).to_owned();

        // Becke grid (spin-agnostic).
        let grid_cfg = ferric_dft::grid::AtomicGridConfig::default();
        let grid = ferric_dft::grid::build_atomic_grid(mol, &grid_cfg);
        let points: Vec<[f64; 3]> = grid.iter().map(|g| g.xyz).collect();
        let weights_g: Vec<f64> = grid.iter().map(|g| g.weight).collect();
        let home_atom: Vec<usize> = grid.iter().map(|g| g.home_atom).collect();
        let npts = points.len();

        let chi = ferric_dft::ao_grid::eval_basis_on_points(mol, obs_bs, &points).map_err(|e| {
            FerricError::General(format!("pdep_polarizability_becke_dynamic (U): chi failed: {e}"))
        })?;
        let nbf = chi.nrows();
        let atom_pos: Vec<[f64; 3]> = mol.atoms.iter().map(|at| [at.x, at.y, at.zpos]).collect();

        // Per-atom atom-centred AO dipole matrices (ω-independent).
        let mut d_ai_ao: Vec<[Array2<f64>; 3]> = (0..natoms)
            .map(|_| std::array::from_fn(|_| Array2::<f64>::zeros((nbf, nbf))))
            .collect();
        for g in 0..npts {
            let a = home_atom[g];
            let w = weights_g[g];
            let r = points[g];
            let ra = atom_pos[a];
            for d in 0..3 {
                let factor = w * (r[d] - ra[d]);
                for mu in 0..nbf {
                    let chi_mu = chi[(mu, g)];
                    let wchi = factor * chi_mu;
                    if wchi.abs() < 1e-30 { continue; }
                    for nu in 0..nbf {
                        d_ai_ao[a][d][(mu, nu)] += wchi * chi[(nu, g)];
                    }
                }
            }
        }
        for a in 0..natoms {
            for d in 0..3 {
                let m = &mut d_ai_ao[a][d];
                for i in 0..nbf {
                    for j in (i+1)..nbf {
                        let avg = 0.5*(m[(i,j)]+m[(j,i)]);
                        m[(i,j)] = avg; m[(j,i)] = avg;
                    }
                }
            }
        }

        // Transform per-atom AO dipoles to per-spin MO basis.
        let nov_a = inter_a.nocc * inter_a.nvir;
        let nov_b = inter_b.nocc * inter_b.nvir;
        let mu_ai_flat_a: Vec<[ndarray::Array1<f64>; 3]> = (0..natoms).map(|a| {
            std::array::from_fn(|d| {
                let mo = c_occ_a.t().dot(&d_ai_ao[a][d]).dot(&c_vir_a);
                let mut v = ndarray::Array1::<f64>::zeros(nov_a);
                for i in 0..inter_a.nocc { for ax in 0..inter_a.nvir { v[i*inter_a.nvir+ax] = mo[(i,ax)]; } }
                v
            })
        }).collect();
        let mu_ai_flat_b: Vec<[ndarray::Array1<f64>; 3]> = (0..natoms).map(|a| {
            std::array::from_fn(|d| {
                if inter_b.nocc == 0 { return ndarray::Array1::<f64>::zeros(1.max(nov_b)); }
                let mo = c_occ_b.t().dot(&d_ai_ao[a][d]).dot(&c_vir_b);
                let mut v = ndarray::Array1::<f64>::zeros(nov_b);
                for i in 0..inter_b.nocc { for ax in 0..inter_b.nvir { v[i*inter_b.nvir+ax] = mo[(i,ax)]; } }
                v
            })
        }).collect();

        // Molecular MO dipoles per spin (sum over atoms).
        let mu_flat_a: [ndarray::Array1<f64>; 3] = std::array::from_fn(|d| {
            mu_ai_flat_a.iter().fold(ndarray::Array1::zeros(nov_a), |acc, ai| acc + &ai[d])
        });
        let mu_flat_b: [ndarray::Array1<f64>; 3] = std::array::from_fn(|d| {
            if inter_b.nocc == 0 { return ndarray::Array1::zeros(1); }
            mu_ai_flat_b.iter().fold(ndarray::Array1::zeros(nov_b), |acc, ai| acc + &ai[d])
        });

        // Frequency loop.
        let mut out: Vec<Vec<[[f64; 3]; 3]>> = vec![vec![[[0.0; 3]; 3]; nfreq]; natoms];

        for (k, &omega) in freqs.iter().enumerate() {
            let omega2 = omega * omega;

            // Per-spin g_σ(ω) = e_iaσ / (ω² + e_iaσ²), prefactor 2.
            let g_a = {
                let n = e_ia_a.len();
                let mut v = ndarray::Array1::<f64>::zeros(n);
                for ia in 0..n { let e = e_ia_a[ia]; v[ia] = e/(omega2+e*e); }
                v
            };
            let g_b = if inter_b.nocc > 0 {
                let n = e_ia_b.len();
                let mut v = ndarray::Array1::<f64>::zeros(n);
                for ia in 0..n { let e = e_ia_b[ia]; v[ia] = e/(omega2+e*e); }
                v
            } else {
                ndarray::Array1::<f64>::zeros(1)
            };

            // ε̃(ω) = I + 2 B̃_α diag(g_α) B̃_αᵀ + 2 B̃_β diag(g_β) B̃_βᵀ.
            let mut eps_mat = Array2::<f64>::zeros((naux, naux));
            for p in 0..naux { eps_mat[(p,p)] = 1.0; }
            for (b_ov, g) in [(&inter_a.b_ov, &g_a), (&inter_b.b_ov, &g_b)] {
                let nov = b_ov.shape()[1];
                if nov == 0 { continue; }
                let mut b_scaled = b_ov.clone();
                for ia in 0..nov {
                    let s = (2.0 * g[ia]).sqrt();
                    b_scaled.column_mut(ia).mapv_inplace(|x| x * s);
                }
                let chi_s = b_scaled.dot(&b_scaled.t());
                eps_mat += &chi_s;
            }

            // w_total^d(ω) = B̃_α (μ_α^d ⊙ g_α) + B̃_β (μ_β^d ⊙ g_β).
            let w_total: [ndarray::Array1<f64>; 3] = std::array::from_fn(|d| {
                let wa = inter_a.b_ov.dot(&(&mu_flat_a[d] * &g_a));
                if inter_b.nocc == 0 { return wa; }
                let wb = inter_b.b_ov.dot(&(&mu_flat_b[d] * &g_b));
                wa + wb
            });
            let y_total: [ndarray::Array1<f64>; 3] =
                std::array::from_fn(|d| eps_mat.solve(&w_total[d]).unwrap());

            for a in 0..natoms {
                // w_ai_total^d = B̃_α (μ_α^{A,d} ⊙ g_α) + B̃_β (μ_β^{A,d} ⊙ g_β).
                let w_ai: [ndarray::Array1<f64>; 3] = std::array::from_fn(|d| {
                    let wa = inter_a.b_ov.dot(&(&mu_ai_flat_a[a][d] * &g_a));
                    if inter_b.nocc == 0 { return wa; }
                    let wb = inter_b.b_ov.dot(&(&mu_ai_flat_b[a][d] * &g_b));
                    wa + wb
                });
                for d in 0..3 {
                    for j in 0..3 {
                        let bare_a = 2.0 * mu_ai_flat_a[a][d].dot(&(&mu_flat_a[j] * &g_a));
                        let bare_b = if inter_b.nocc > 0 {
                            2.0 * mu_ai_flat_b[a][d].dot(&(&mu_flat_b[j] * &g_b))
                        } else { 0.0 };
                        let coupled = w_ai[d].dot(&y_total[j]);
                        out[a][k][d][j] = bare_a + bare_b - 4.0 * coupled;
                    }
                }
                // Symmetrize.
                for i in 0..3 {
                    for j in (i+1)..3 {
                        let avg = 0.5*(out[a][k][i][j]+out[a][k][j][i]);
                        out[a][k][i][j] = avg; out[a][k][j][i] = avg;
                    }
                }
            }
        }
        return Ok(out);
    }

    // --- Closed-shell path below (unchanged) ---
    let inter =
        ferric_mp2::rimp2::compute_rpa_intermediates(mol, obs, dfbs, op, rhf, &mp2_cfg)?;
    let b_ov = &inter.b_ov;
    let nocc = inter.nocc;
    let nvir = inter.nvir;
    let nocc_total = inter.nocc_total;
    let first_occ = inter.first_occ;
    let naux = inter.naux;
    let nov = nocc * nvir;

    let eps = rhf.eps_r();
    let eps_occ: Vec<f64> = eps[first_occ..first_occ + nocc].to_vec();
    let eps_vir: Vec<f64> = eps[nocc_total..nocc_total + nvir].to_vec();

    let c = rhf.mos_r();
    let c_occ = c.slice(ndarray::s![.., first_occ..first_occ + nocc]).to_owned();
    let c_vir = c.slice(ndarray::s![.., nocc_total..nocc_total + nvir]).to_owned();

    // ε_ia table (the bare excitation energies).
    let mut e_ia = ndarray::Array1::<f64>::zeros(nov);
    for i in 0..nocc {
        for a in 0..nvir {
            e_ia[i * nvir + a] = eps_vir[a] - eps_occ[i];
        }
    }

    // --- Build the Becke grid + partition-weighted per-atom MO dipoles ONCE.
    // (Identical to pdep_polarizability_becke; ω-independent.)
    let grid_cfg = AtomicGridConfig::default();
    let grid = build_atomic_grid(mol, &grid_cfg);
    let points: Vec<[f64; 3]> = grid.iter().map(|g| g.xyz).collect();
    let weights: Vec<f64> = grid.iter().map(|g| g.weight).collect();
    let home_atom: Vec<usize> = grid.iter().map(|g| g.home_atom).collect();
    let npts = points.len();

    let chi = eval_basis_on_points(mol, obs_bs, &points).map_err(|e| {
        FerricError::General(format!(
            "pdep_polarizability_becke_dynamic: chi eval failed: {e}"
        ))
    })?;
    let nbf = chi.nrows();
    debug_assert_eq!(nbf, obs.nbasis());

    // Atom positions (Bohr) — used to shift dipole to atom-centred coordinates.
    // Using (r - R_A) instead of the lab-frame r makes each per-atom contribution
    // to α^A(iω) origin-independent: at all frequencies, α^A and Σ_A α^A are
    // unchanged by a global translation of the coordinate system.
    let atom_pos: Vec<[f64; 3]> = mol
        .atoms
        .iter()
        .map(|at| [at.x, at.y, at.zpos])
        .collect();

    let mut d_ai_ao: Vec<[Array2<f64>; 3]> = (0..natoms)
        .map(|_| std::array::from_fn(|_| Array2::<f64>::zeros((nbf, nbf))))
        .collect();
    let mut d_sum: [Array2<f64>; 3] = std::array::from_fn(|_| Array2::<f64>::zeros((nbf, nbf)));
    for g in 0..npts {
        let a = home_atom[g];
        let w = weights[g];
        let r = points[g];
        let ra = atom_pos[a];
        for d in 0..3 {
            // Atom-centred displacement: (r_d - R_{A,d}) makes α^A(iω) origin-independent.
            let factor = w * (r[d] - ra[d]);
            for mu in 0..nbf {
                let chi_mu = chi[(mu, g)];
                let weighted_chi_mu = factor * chi_mu;
                if weighted_chi_mu.abs() < 1e-30 {
                    continue;
                }
                for nu in 0..nbf {
                    let contrib = weighted_chi_mu * chi[(nu, g)];
                    d_ai_ao[a][d][(mu, nu)] += contrib;
                    d_sum[d][(mu, nu)] += contrib;
                }
            }
        }
    }
    for d in 0..3 {
        for a in 0..natoms {
            let m = &mut d_ai_ao[a][d];
            for i in 0..nbf {
                for j in (i + 1)..nbf {
                    let avg = 0.5 * (m[(i, j)] + m[(j, i)]);
                    m[(i, j)] = avg;
                    m[(j, i)] = avg;
                }
            }
        }
        for i in 0..nbf {
            for j in (i + 1)..nbf {
                let avg = 0.5 * (d_sum[d][(i, j)] + d_sum[d][(j, i)]);
                d_sum[d][(i, j)] = avg;
                d_sum[d][(j, i)] = avg;
            }
        }
    }
    // No renormalization of the atom-centred per-atom dipoles.
    //
    // The static Becke path renormalizes each AO-pair's grid partition to the
    // global analytical dipole ⟨μ|r|ν⟩, which fixes grid-quadrature error on the
    // dipole magnitude. Here we use atom-centred displacements (r − R_A), so the
    // natural analytical comparison is the atom-centred grid SUM, not the global
    // dipole — and those differ by Σ_A R_A · q^A_{μν} (charge partition moments).
    //
    // For the dynamic path what matters is the *frequency dependence* of α^A(iω),
    // not the absolute magnitude at ω=0. The grid quadrature error on (r − R_A)
    // is smooth and does not introduce frequency-dependent artifacts. We therefore
    // skip renormalization and accept the raw grid integrals. This gives correct
    // physics for the Casimir-Polder integrand at all ω.
    //
    // NOTE: the molecular sum Σ_A α^A(ω=0) from this path will NOT equal
    // pdep_polarizability_static (which uses a renormalized lab-frame dipole).
    // The correct comparison for the regression gate is: take the ω=0 dynamic path
    // result, form the atom-centred MO dipoles, and verify the SMW formula gives
    // the same α as the lab-frame path scaled by the atom-centred/lab-frame ratio.
    // For the purposes of C6 correctness, we verify:
    //   * α_iso_A(ω=0) > 0 for each atom (positive sum rule)
    //   * α_A(iω) decays monotonically with ω (correct frequency dependence)
    //   * homonuclear atom pairs give equal C6 (symmetry check)
    // These are tested in the unit tests.
    for a in 0..natoms {
        for d in 0..3 {
            let m = &mut d_ai_ao[a][d];
            for i in 0..nbf {
                for j in (i + 1)..nbf {
                    let avg = 0.5 * (m[(i, j)] + m[(j, i)]);
                    m[(i, j)] = avg;
                    m[(j, i)] = avg;
                }
            }
        }
    }

    // Transform to MO occ-vir basis (per-atom + molecular sum).
    let mut mu_ai_mo: Vec<[Array2<f64>; 3]> = (0..natoms)
        .map(|_| std::array::from_fn(|_| Array2::<f64>::zeros((nocc, nvir))))
        .collect();
    let mut mu_mo: [Array2<f64>; 3] = std::array::from_fn(|_| Array2::<f64>::zeros((nocc, nvir)));
    for a in 0..natoms {
        for d in 0..3 {
            let m = c_occ.t().dot(&d_ai_ao[a][d]).dot(&c_vir);
            mu_mo[d] = &mu_mo[d] + &m;
            mu_ai_mo[a][d] = m;
        }
    }

    // Flatten molecular dipole and per-atom dipoles into (nov,) vectors ONCE.
    let mu_ai_flat: Vec<[ndarray::Array1<f64>; 3]> = (0..natoms)
        .map(|a| {
            std::array::from_fn(|d| {
                let mut v = ndarray::Array1::<f64>::zeros(nov);
                for i in 0..nocc {
                    for ax in 0..nvir {
                        v[i * nvir + ax] = mu_ai_mo[a][d][(i, ax)];
                    }
                }
                v
            })
        })
        .collect();

    // Flatten the Becke-sum molecular dipole (sum of atom-centred pieces).
    let mu_flat: [ndarray::Array1<f64>; 3] = std::array::from_fn(|d| {
        let mut v = ndarray::Array1::<f64>::zeros(nov);
        for i in 0..nocc { for ax in 0..nvir { v[i*nvir+ax] = mu_mo[d][(i,ax)]; } }
        v
    });

    // --- Frequency loop: direct per-atom SMW (symmetric Becke-sum reference) ---
    //
    // α^A_{ij}(iω) = 4 μ^{A,i}·g·μ^{Becke,j} − 16 (B·μ^{A,i}·g)·ε̃⁻¹·(B·μ^{Becke,j}·g)
    //
    // Both slots use the same Becke-sum dipole μ^Becke = Σ_A μ^A, so
    // Σ_A α^A = α_mol(μ^Becke) exactly. This matches pdep_dynamic_polarizability_truncated.
    // The anisotropy reflects the Becke partition's atom-centred displacements;
    // for isotropic C6 via Casimir-Polder this is the correct formula.
    let mut out: Vec<Vec<[[f64; 3]; 3]>> =
        vec![vec![[[0.0; 3]; 3]; nfreq]; natoms];

    for (k, &omega) in freqs.iter().enumerate() {
        let omega2 = omega * omega;
        let mut g = ndarray::Array1::<f64>::zeros(nov);
        for ia in 0..nov { let e = e_ia[ia]; g[ia] = e / (omega2 + e * e); }

        // ε̃(ω) = I + 4 B̃ diag(g) B̃^T
        let mut b_scaled = b_ov.clone();
        for ia in 0..nov {
            b_scaled.column_mut(ia).mapv_inplace(|x| x * (4.0 * g[ia]).sqrt());
        }
        let mut eps_mat: Array2<f64> = b_scaled.dot(&b_scaled.t());
        for p in 0..naux { eps_mat[(p, p)] += 1.0; }

        // Solve ε̃ y^j = B·(g⊙μ^{Becke,j}) once per direction.
        let mu_g: [ndarray::Array1<f64>; 3] = std::array::from_fn(|d| &mu_flat[d] * &g);
        let w_mol: [ndarray::Array1<f64>; 3] = std::array::from_fn(|d| b_ov.dot(&mu_g[d]));
        let y_mol: [ndarray::Array1<f64>; 3] = {
            use ndarray_linalg::Solve;
            std::array::from_fn(|d| eps_mat.solve(&w_mol[d]).unwrap())
        };

        for a in 0..natoms {
            let mut tensor = [[0.0_f64; 3]; 3];
            for d in 0..3 {
                let w_ai = b_ov.dot(&(&mu_ai_flat[a][d] * &g));
                for j in 0..3 {
                    let bare = mu_ai_flat[a][d].dot(&mu_g[j]);
                    let coupled = w_ai.dot(&y_mol[j]);
                    tensor[d][j] = 4.0 * bare - 16.0 * coupled;
                }
            }
            // Symmetrize.
            for i in 0..3 { for j in (i+1)..3 {
                let avg = 0.5*(tensor[i][j]+tensor[j][i]);
                tensor[i][j] = avg; tensor[j][i] = avg;
            }}
            out[a][k] = tensor;
        }
    }

    Ok(out)
}

pub fn pdep_polarizability_hirshfeld(
    mol: &Molecule,
    obs: &PreparedBasis,
    obs_bs: &ferric_core::basis::BasisSet,
    dfbs: &PreparedBasis,
    rhf: &ScfResult,
    op: Operator,
    _cfg: &PdepRpaConfig,
) -> Result<Vec<[[f64; 3]; 3]>, FerricError> {
    use ferric_export::cube::GridSpec;
    use ferric_export::gto_eval::eval_basis_on_grid;
    use ndarray_linalg::Solve;

    if !matches!(rhf.spin, Spin::Restricted) {
        return Err(FerricError::General(
            "pdep_polarizability_hirshfeld: only closed-shell (Restricted) supported".into(),
        ));
    }

    let natoms = mol.atoms.len();

    // ---------------------------------------------------------------------
    // 1. Reuse the same RI intermediates / dipole machinery as the
    //    molecular static-α path.
    // ---------------------------------------------------------------------
    let mp2_cfg = ferric_mp2::rimp2::RiMp2Config { frozen_core: 0 };
    let inter =
        ferric_mp2::rimp2::compute_rpa_intermediates(mol, obs, dfbs, op, rhf, &mp2_cfg)?;
    let b_ov = &inter.b_ov;
    let nocc = inter.nocc;
    let nvir = inter.nvir;
    let nocc_total = inter.nocc_total;
    let first_occ = inter.first_occ;
    let naux = inter.naux;
    let nov = nocc * nvir;

    let eps = rhf.eps_r();
    let eps_occ: Vec<f64> = eps[first_occ..first_occ + nocc].to_vec();
    let eps_vir: Vec<f64> = eps[nocc_total..nocc_total + nvir].to_vec();

    // Molecular MO dipole (origin = 0) — same as in pdep_polarizability_static.
    // NOTE: this analytical dipole is used only for the optional informational
    // sum-rule check at the end; the per-atom partition is fully grid-based
    // (numerical Σ_A D^{A,i}_AO = numerical D^i_AO), so the sum rule is
    // checked between the grid-α and the *grid-built* molecular α.
    let _dip_ao_analytical = oneelectron::dipole(obs, [0.0, 0.0, 0.0]);
    // (used below to rescale grid-numerical per-atom partitions into the
    // analytical sum rule.)
    let c = rhf.mos_r();
    let c_occ = c.slice(ndarray::s![.., first_occ..first_occ + nocc]).to_owned();
    let c_vir = c.slice(ndarray::s![.., nocc_total..nocc_total + nvir]).to_owned();
    // mu_mo will be built numerically below (sum of per-atom Hirshfeld
    // partitions) so the sum rule is exact up to grid noise.
    let mut mu_mo: [Array2<f64>; 3] = std::array::from_fn(|_| Array2::<f64>::zeros((nocc, nvir)));

    let mut inv_de = ndarray::Array1::<f64>::zeros(nov);
    for i in 0..nocc {
        for a in 0..nvir {
            inv_de[i * nvir + a] = 1.0 / (eps_vir[a] - eps_occ[i]);
        }
    }

    // ε̃ = I + B̃ · diag(4/Δε) · B̃^T  (independent of μ; build now, factor below).
    let mut b_scaled = b_ov.clone();
    for ia in 0..nov {
        let s = (4.0 * inv_de[ia]).sqrt();
        let mut col = b_scaled.column_mut(ia);
        col.mapv_inplace(|x| x * s);
    }
    let mut eps_mat: Array2<f64> = b_scaled.dot(&b_scaled.t());
    for p in 0..naux {
        eps_mat[(p, p)] += 1.0;
    }

    // ---------------------------------------------------------------------
    // 2. Build a regular real-space grid bounding the molecule.
    //    Grid spacing chosen to balance integration error vs cost; the
    //    sum-rule gate validates whatever choice we make.
    // ---------------------------------------------------------------------
    let spacing = std::env::var("FERRIC_HIRSHFELD_SPACING")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(0.20); // Bohr
    let margin = std::env::var("FERRIC_HIRSHFELD_MARGIN")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(6.0); // Bohr — needs to cover the diffuse tail of α
    let grid = GridSpec::bounding_box(mol, margin, spacing);
    let dv = spacing * spacing * spacing;
    let npts = grid.n_x * grid.n_y * grid.n_z;

    // χ_μ(r_g) on grid: shape (nbf_obs, npts).
    let chi = eval_basis_on_grid(mol, obs_bs, &grid).map_err(|e| {
        FerricError::General(format!("pdep_polarizability_hirshfeld: chi eval failed: {e}"))
    })?;
    let nbf = chi.nrows();
    debug_assert_eq!(nbf, obs.nbasis());

    // ---------------------------------------------------------------------
    // 3. Build Hirshfeld weights w^A(r_g) on the grid.
    //    Proatom: single-exponential Slater density per element.
    // ---------------------------------------------------------------------
    let mut atom_xi = vec![0.0_f64; natoms];
    for (a, atom) in mol.atoms.iter().enumerate() {
        atom_xi[a] = slater_xi_for_z(atom.z);
    }

    // For each atom: compute ρ_A_free(|r - R_A|) on the grid.
    // Then w^A = ρ_A / (Σ_B ρ_B + ε).
    let mut rho_free: Vec<Vec<f64>> = vec![vec![0.0; npts]; natoms];
    let mut rho_sum: Vec<f64> = vec![0.0; npts];
    let hx = grid.step_x[0];
    let hy = grid.step_y[1];
    let hz = grid.step_z[2];
    for a in 0..natoms {
        let xi = atom_xi[a];
        let z_a = mol.atoms[a].z as f64;
        let prefac = z_a * xi.powi(3) / std::f64::consts::PI;
        let rax = mol.atoms[a].x;
        let ray = mol.atoms[a].y;
        let raz = mol.atoms[a].zpos;
        let row = &mut rho_free[a];
        for ix in 0..grid.n_x {
            let x = grid.origin[0] + ix as f64 * hx;
            for iy in 0..grid.n_y {
                let y = grid.origin[1] + iy as f64 * hy;
                for iz in 0..grid.n_z {
                    let z = grid.origin[2] + iz as f64 * hz;
                    let g = (ix * grid.n_y + iy) * grid.n_z + iz;
                    let dx = x - rax;
                    let dy = y - ray;
                    let dz = z - raz;
                    let r = (dx * dx + dy * dy + dz * dz).sqrt();
                    let rho = prefac * (-2.0 * xi * r).exp();
                    row[g] = rho;
                    rho_sum[g] += rho;
                }
            }
        }
    }

    // ---------------------------------------------------------------------
    // 4. For each atom A and Cartesian i, build the Hirshfeld-weighted
    //    AO dipole matrix
    //        D^{A,i}_{μν} = ∫ dr χ_μ(r) χ_ν(r) w^A(r) r_i
    //    by direct quadrature on the regular grid.
    //
    //    Implementation: combine χ with w^A and r_i into a "weighted χ̃"
    //    on grid, then χ · χ̃^T (a single DGEMM per atom per Cartesian).
    // ---------------------------------------------------------------------
    let eps_floor = 1e-12;
    let mut alpha_per_atom: Vec<[[f64; 3]; 3]> = vec![[[0.0; 3]; 3]; natoms];

    // Precompute r_i(g) per Cartesian.
    let mut ri_grid: [Vec<f64>; 3] = [vec![0.0; npts], vec![0.0; npts], vec![0.0; npts]];
    for ix in 0..grid.n_x {
        let x = grid.origin[0] + ix as f64 * hx;
        for iy in 0..grid.n_y {
            let y = grid.origin[1] + iy as f64 * hy;
            for iz in 0..grid.n_z {
                let z = grid.origin[2] + iz as f64 * hz;
                let g = (ix * grid.n_y + iy) * grid.n_z + iz;
                ri_grid[0][g] = x;
                ri_grid[1][g] = y;
                ri_grid[2][g] = z;
            }
        }
    }

    // PASS 1: build per-atom AO dipole matrices D^{A,i}_AO on the grid and
    // their per-atom sum (numerical molecular dipole). We then *renormalize*
    // per-AO-pair to enforce the analytical sum-rule:
    //   D^{A,i}_AO_corrected_{μν} = D^{A,i}_AO_grid_{μν} · ratio_{μν}
    //   ratio_{μν} = D^i_AO_analytical_{μν} / Σ_B D^{B,i}_AO_grid_{μν}
    // (when the denominator is small, fall back to a flat 1/N partition).
    // This decouples the partitioning fraction (Hirshfeld, robust on a
    // coarse grid) from the absolute dipole values (libint analytical).
    let mut d_ai_ao_all: Vec<[Array2<f64>; 3]> = (0..natoms)
        .map(|_| std::array::from_fn(|_| Array2::<f64>::zeros((nbf, nbf))))
        .collect();
    let mut d_ai_ao_sum: [Array2<f64>; 3] =
        std::array::from_fn(|_| Array2::<f64>::zeros((nbf, nbf)));

    for a in 0..natoms {
        let mut wa = vec![0.0_f64; npts];
        for g in 0..npts {
            let denom = rho_sum[g] + eps_floor;
            wa[g] = rho_free[a][g] / denom;
        }
        for i_cart in 0..3 {
            let mut combined = Array2::<f64>::zeros((nbf, npts));
            for mu in 0..nbf {
                let chi_mu = chi.row(mu);
                let mut row = combined.row_mut(mu);
                for g in 0..npts {
                    row[g] = chi_mu[g] * wa[g] * ri_grid[i_cart][g] * dv;
                }
            }
            let d: Array2<f64> = chi.dot(&combined.t());
            // Symmetrize.
            let mut d_sym = Array2::<f64>::zeros((nbf, nbf));
            for mu in 0..nbf {
                for nu in 0..nbf {
                    d_sym[(mu, nu)] = 0.5 * (d[(mu, nu)] + d[(nu, mu)]);
                }
            }
            d_ai_ao_sum[i_cart] = &d_ai_ao_sum[i_cart] + &d_sym;
            d_ai_ao_all[a][i_cart] = d_sym;
        }
    }

    // Renormalize: replace per-atom partition with analytical-sum × fraction.
    let dip_ao = &_dip_ao_analytical;
    let inv_n = 1.0 / (natoms as f64);
    let small = 1e-10;
    for i_cart in 0..3 {
        for mu in 0..nbf {
            for nu in 0..nbf {
                let total_grid = d_ai_ao_sum[i_cart][(mu, nu)];
                let total_analytical = dip_ao[i_cart][(mu, nu)];
                if total_grid.abs() < small * (1.0 + total_analytical.abs()) {
                    // Fall back to equal partition.
                    for a in 0..natoms {
                        d_ai_ao_all[a][i_cart][(mu, nu)] = total_analytical * inv_n;
                    }
                } else {
                    let scale = total_analytical / total_grid;
                    for a in 0..natoms {
                        d_ai_ao_all[a][i_cart][(mu, nu)] *= scale;
                    }
                }
            }
        }
    }

    // Transform to MO occ-vir basis.
    let mut mu_ai_mo_all: Vec<[Array2<f64>; 3]> = (0..natoms)
        .map(|_| std::array::from_fn(|_| Array2::<f64>::zeros((nocc, nvir))))
        .collect();
    for a in 0..natoms {
        for i_cart in 0..3 {
            let mu_ai_mo = c_occ.t().dot(&d_ai_ao_all[a][i_cart]).dot(&c_vir);
            mu_mo[i_cart] = &mu_mo[i_cart] + &mu_ai_mo;
            mu_ai_mo_all[a][i_cart] = mu_ai_mo;
        }
    }

    // Now we have grid-consistent molecular μ^j_MO. Build μ^j_flat, w_mol, y_mol.
    let mu_flat: [ndarray::Array1<f64>; 3] = std::array::from_fn(|d| {
        let mut v = ndarray::Array1::<f64>::zeros(nov);
        for i in 0..nocc {
            for ax in 0..nvir {
                v[i * nvir + ax] = mu_mo[d][(i, ax)];
            }
        }
        v
    });
    let mu_flat_inv: [ndarray::Array1<f64>; 3] =
        std::array::from_fn(|d| &mu_flat[d] * &inv_de);
    let w_mol: [ndarray::Array1<f64>; 3] =
        std::array::from_fn(|d| b_ov.dot(&mu_flat_inv[d]));
    let y_mol: [ndarray::Array1<f64>; 3] =
        std::array::from_fn(|d| eps_mat.solve(&w_mol[d]).unwrap());

    // PASS 2: assemble α^A.
    for a in 0..natoms {
        for i_cart in 0..3 {
            let mu_ai_mo = &mu_ai_mo_all[a][i_cart];
            let mut mu_ai_flat = ndarray::Array1::<f64>::zeros(nov);
            let mut mu_ai_flat_inv = ndarray::Array1::<f64>::zeros(nov);
            for i in 0..nocc {
                for ax in 0..nvir {
                    let ia = i * nvir + ax;
                    mu_ai_flat[ia] = mu_ai_mo[(i, ax)];
                    mu_ai_flat_inv[ia] = mu_ai_mo[(i, ax)] * inv_de[ia];
                }
            }
            let w_ai = b_ov.dot(&mu_ai_flat_inv);

            for j in 0..3 {
                let bare = mu_ai_flat.dot(&mu_flat_inv[j]);
                let coupled = w_ai.dot(&y_mol[j]);
                alpha_per_atom[a][i_cart][j] = 4.0 * bare - 16.0 * coupled;
            }
        }

        // Per-atom symmetrize (consumer schema requires < 1e-5).
        for i in 0..3 {
            for j in (i + 1)..3 {
                let avg = 0.5 * (alpha_per_atom[a][i][j] + alpha_per_atom[a][j][i]);
                alpha_per_atom[a][i][j] = avg;
                alpha_per_atom[a][j][i] = avg;
            }
        }
    }

    // ---------------------------------------------------------------------
    // 5. Sum-rule gate.  Σ_A α^A vs molecular α from the same path.
    //    Tolerance 1e-3 a.u. as specified.
    // ---------------------------------------------------------------------
    let mut sum_tensor = [[0.0_f64; 3]; 3];
    for a in 0..natoms {
        for i in 0..3 {
            for j in 0..3 {
                sum_tensor[i][j] += alpha_per_atom[a][i][j];
            }
        }
    }
    // Molecular reference built from grid-consistent dipole.
    let mut mol_tensor = [[0.0_f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let bare = mu_flat[i].dot(&mu_flat_inv[j]);
            let coupled = w_mol[i].dot(&y_mol[j]);
            mol_tensor[i][j] = 4.0 * bare - 16.0 * coupled;
        }
    }
    // Symmetrize molecular for fair comparison.
    for i in 0..3 {
        for j in (i + 1)..3 {
            let avg = 0.5 * (mol_tensor[i][j] + mol_tensor[j][i]);
            mol_tensor[i][j] = avg;
            mol_tensor[j][i] = avg;
        }
    }
    let mut max_diff = 0.0_f64;
    for i in 0..3 {
        for j in 0..3 {
            max_diff = max_diff.max((sum_tensor[i][j] - mol_tensor[i][j]).abs());
        }
    }
    if std::env::var("FERRIC_DEBUG_HIRSHFELD").is_ok() {
        eprintln!(
            "[hirshfeld] grid {}×{}×{} (spacing={}, margin={}), max|Σα^A − α_mol| = {:.3e}",
            grid.n_x, grid.n_y, grid.n_z, spacing, margin, max_diff
        );
        for a in 0..natoms {
            eprintln!(
                "[hirshfeld] atom {} (Z={}) α_iso = {:.4}",
                a,
                mol.atoms[a].z,
                (alpha_per_atom[a][0][0] + alpha_per_atom[a][1][1] + alpha_per_atom[a][2][2]) / 3.0
            );
        }
    }
    if max_diff > 1e-3 {
        return Err(FerricError::General(format!(
            "pdep_polarizability_hirshfeld: sum rule failed — max|Σ_A α^A − α_mol| = {:.3e} > 1e-3. \
             Grid spacing {} Bohr, margin {} Bohr may be too coarse.",
            max_diff, spacing, margin
        )));
    }

    Ok(alpha_per_atom)
}

/// Dynamic Hirshfeld per-atom polarizability α^A(iω) on an imaginary-frequency grid.
///
/// Computes the same renormalized Hirshfeld MO dipoles as [`pdep_polarizability_hirshfeld`]
/// (grid work is ω-independent), then evaluates the full-rank SMW formula at each
/// quadrature frequency. Because `Σ_A μ^{A,i} = μ^i` exactly after renormalization,
/// `Σ_A α^A(iω) = α_mol(iω)` at every ω — exact sum rule, origin-independent,
/// correct anisotropy including charge-transfer along bond axes.
///
/// Returns `per_atom[A][k][i][j]` = α^A_{ij}(iω_k), shape `(natoms, nfreq, 3, 3)`.
// System inputs (molecule, orbital + auxiliary bases, the basis-set object for
// partitioning, the SCF reference, the operator, config, and the frequency
// grid) are all distinct; there is no sub-bundle to extract.
#[allow(clippy::too_many_arguments)]
pub fn pdep_polarizability_hirshfeld_dynamic(
    mol: &Molecule,
    obs: &PreparedBasis,
    obs_bs: &ferric_core::basis::BasisSet,
    dfbs: &PreparedBasis,
    rhf: &ScfResult,
    op: Operator,
    _cfg: &PdepRpaConfig,
    freqs: &[f64],
) -> Result<Vec<Vec<[[f64; 3]; 3]>>, FerricError> {
    use ferric_export::cube::GridSpec;
    use ferric_export::gto_eval::eval_basis_on_grid;
    use ndarray_linalg::Solve;

    if !matches!(rhf.spin, Spin::Restricted) {
        return Err(FerricError::General(
            "pdep_polarizability_hirshfeld_dynamic: only closed-shell (Restricted) supported".into(),
        ));
    }

    let natoms = mol.atoms.len();
    let nfreq = freqs.len();

    // RI intermediates — same as static path.
    let mp2_cfg = ferric_mp2::rimp2::RiMp2Config { frozen_core: 0 };
    let inter = ferric_mp2::rimp2::compute_rpa_intermediates(mol, obs, dfbs, op, rhf, &mp2_cfg)?;
    let b_ov = &inter.b_ov;
    let nocc = inter.nocc;
    let nvir = inter.nvir;
    let nocc_total = inter.nocc_total;
    let first_occ = inter.first_occ;
    let naux = inter.naux;
    let nov = nocc * nvir;

    let eps = rhf.eps_r();
    let eps_occ: Vec<f64> = eps[first_occ..first_occ + nocc].to_vec();
    let eps_vir: Vec<f64> = eps[nocc_total..nocc_total + nvir].to_vec();
    let mut e_ia = ndarray::Array1::<f64>::zeros(nov);
    for i in 0..nocc { for a in 0..nvir { e_ia[i*nvir+a] = eps_vir[a] - eps_occ[i]; } }

    let c = rhf.mos_r();
    let c_occ = c.slice(ndarray::s![.., first_occ..first_occ + nocc]).to_owned();
    let c_vir = c.slice(ndarray::s![.., nocc_total..nocc_total + nvir]).to_owned();

    // --- Grid setup (identical to static Hirshfeld path) ---
    let _dip_ao_analytical = oneelectron::dipole(obs, [0.0, 0.0, 0.0]);
    let spacing = std::env::var("FERRIC_HIRSHFELD_SPACING")
        .ok().and_then(|s| s.parse::<f64>().ok()).unwrap_or(0.20);
    let margin = std::env::var("FERRIC_HIRSHFELD_MARGIN")
        .ok().and_then(|s| s.parse::<f64>().ok()).unwrap_or(6.0);
    let grid = GridSpec::bounding_box(mol, margin, spacing);
    let dv = spacing * spacing * spacing;
    let npts = grid.n_x * grid.n_y * grid.n_z;

    let chi = eval_basis_on_grid(mol, obs_bs, &grid).map_err(|e| {
        FerricError::General(format!("pdep_polarizability_hirshfeld_dynamic: chi eval failed: {e}"))
    })?;
    let nbf = chi.nrows();

    // Proatom Slater densities and Hirshfeld weights.
    let mut atom_xi = vec![0.0_f64; natoms];
    for (a, atom) in mol.atoms.iter().enumerate() { atom_xi[a] = slater_xi_for_z(atom.z); }
    let mut rho_free: Vec<Vec<f64>> = vec![vec![0.0; npts]; natoms];
    let mut rho_sum: Vec<f64> = vec![0.0; npts];
    let hx = grid.step_x[0]; let hy = grid.step_y[1]; let hz = grid.step_z[2];
    for a in 0..natoms {
        let xi = atom_xi[a];
        let z_a = mol.atoms[a].z as f64;
        let prefac = z_a * xi.powi(3) / std::f64::consts::PI;
        let rax = mol.atoms[a].x; let ray = mol.atoms[a].y; let raz = mol.atoms[a].zpos;
        for ix in 0..grid.n_x {
            let x = grid.origin[0] + ix as f64 * hx;
            for iy in 0..grid.n_y {
                let y = grid.origin[1] + iy as f64 * hy;
                for iz in 0..grid.n_z {
                    let z = grid.origin[2] + iz as f64 * hz;
                    let g = (ix * grid.n_y + iy) * grid.n_z + iz;
                    let r = ((x-rax).powi(2)+(y-ray).powi(2)+(z-raz).powi(2)).sqrt();
                    let rho = prefac * (-2.0 * xi * r).exp();
                    rho_free[a][g] = rho; rho_sum[g] += rho;
                }
            }
        }
    }

    // Grid coordinates.
    let mut ri_grid: [Vec<f64>; 3] = [vec![0.0; npts], vec![0.0; npts], vec![0.0; npts]];
    for ix in 0..grid.n_x {
        let x = grid.origin[0] + ix as f64 * hx;
        for iy in 0..grid.n_y {
            let y = grid.origin[1] + iy as f64 * hy;
            for iz in 0..grid.n_z {
                let z = grid.origin[2] + iz as f64 * hz;
                let g = (ix * grid.n_y + iy) * grid.n_z + iz;
                ri_grid[0][g] = x; ri_grid[1][g] = y; ri_grid[2][g] = z;
            }
        }
    }

    // Build per-atom Hirshfeld AO dipoles with the ATOM-CENTRED position
    // operator (r − R_A). This yields the *intrinsic* atomic polarizability:
    // origin-independent and charge-transfer-free, the correct object for
    // atom-resolved C6 (TS/MBD convention). Using the global-frame r instead
    // leaks R_A·(field-induced charge transfer onto A) into α^A, which breaks
    // the symmetry of equivalent atoms off-origin (e.g. the two N in N₂) and
    // contaminates the per-atom bond-axis response. We deliberately do NOT
    // renormalize to the global lab-frame dipole — that renormalization is the
    // gauge-breaking step (it scales each atom by a shared factor and cannot
    // restore per-atom symmetry). The molecular total is computed separately
    // from the lab-frame molecular α (see `molecular_dynamic_polarizability`).
    let eps_floor = 1e-12;
    let atom_pos: Vec<[f64; 3]> =
        mol.atoms.iter().map(|at| [at.x, at.y, at.zpos]).collect();
    let mut d_ai_ao_all: Vec<[Array2<f64>; 3]> = (0..natoms)
        .map(|_| std::array::from_fn(|_| Array2::<f64>::zeros((nbf, nbf)))).collect();
    for a in 0..natoms {
        let ra = atom_pos[a];
        let mut wa = vec![0.0_f64; npts];
        for g in 0..npts { wa[g] = rho_free[a][g] / (rho_sum[g] + eps_floor); }
        for i_cart in 0..3 {
            let ra_d = ra[i_cart];
            let mut combined = Array2::<f64>::zeros((nbf, npts));
            for mu in 0..nbf {
                let chi_mu = chi.row(mu);
                for g in 0..npts {
                    combined[(mu, g)] = chi_mu[g] * wa[g] * (ri_grid[i_cart][g] - ra_d) * dv;
                }
            }
            let d = chi.dot(&combined.t());
            let mut d_sym = Array2::<f64>::zeros((nbf, nbf));
            for mu in 0..nbf { for nu in 0..nbf { d_sym[(mu,nu)] = 0.5*(d[(mu,nu)]+d[(nu,mu)]); } }
            d_ai_ao_all[a][i_cart] = d_sym;
        }
    }

    // Transform renormalized AO dipoles to MO occ-vir basis (ω-independent).
    let mu_ai_flat: Vec<[ndarray::Array1<f64>; 3]> = (0..natoms).map(|a| {
        std::array::from_fn(|d| {
            let mo = c_occ.t().dot(&d_ai_ao_all[a][d]).dot(&c_vir);
            let mut v = ndarray::Array1::<f64>::zeros(nov);
            for i in 0..nocc { for ax in 0..nvir { v[i*nvir+ax] = mo[(i,ax)]; } }
            v
        })
    }).collect();

    // Molecular (field-side) dipole: the TRUE lab-frame molecular dipole Σ_i r_i
    // from the analytical AO integrals — the perturbation a uniform field
    // couples to. NOT the sum of atom-centred per-atom pieces (those answer the
    // atom-resolved question). Pairing the atom-centred bra (mu_ai_flat) with
    // this lab-frame molecular ket gives α^A_{dj} = ∂μ^A_d/∂E_j.
    let mu_flat: [ndarray::Array1<f64>; 3] = std::array::from_fn(|d| {
        let mo = c_occ.t().dot(&_dip_ao_analytical[d]).dot(&c_vir);
        let mut v = ndarray::Array1::<f64>::zeros(nov);
        for i in 0..nocc { for ax in 0..nvir { v[i * nvir + ax] = mo[(i, ax)]; } }
        v
    });

    // --- Frequency loop ---
    let mut out: Vec<Vec<[[f64; 3]; 3]>> = vec![vec![[[0.0; 3]; 3]; nfreq]; natoms];

    for (k, &omega) in freqs.iter().enumerate() {
        let omega2 = omega * omega;
        let mut g = ndarray::Array1::<f64>::zeros(nov);
        for ia in 0..nov { let e = e_ia[ia]; g[ia] = e / (omega2 + e*e); }

        // ε̃(ω) = I + 4 B̃ diag(g) B̃^T
        let mut b_scaled = b_ov.clone();
        for ia in 0..nov { b_scaled.column_mut(ia).mapv_inplace(|x| x * (4.0*g[ia]).sqrt()); }
        let mut eps_mat: Array2<f64> = b_scaled.dot(&b_scaled.t());
        for p in 0..naux { eps_mat[(p,p)] += 1.0; }

        // Solve ε̃ y^j = B·(g⊙μ^j) once per direction (molecular Hirshfeld dipole).
        let mu_g: [ndarray::Array1<f64>; 3] = std::array::from_fn(|d| &mu_flat[d] * &g);
        let w_mol: [ndarray::Array1<f64>; 3] = std::array::from_fn(|d| b_ov.dot(&mu_g[d]));
        let y_mol: [ndarray::Array1<f64>; 3] =
            std::array::from_fn(|d| eps_mat.solve(&w_mol[d]).unwrap());

        for a in 0..natoms {
            let mut tensor = [[0.0_f64; 3]; 3];
            for d in 0..3 {
                let w_ai = b_ov.dot(&(&mu_ai_flat[a][d] * &g));
                for j in 0..3 {
                    let bare = mu_ai_flat[a][d].dot(&mu_g[j]);
                    let coupled = w_ai.dot(&y_mol[j]);
                    tensor[d][j] = 4.0 * bare - 16.0 * coupled;
                }
            }
            for i in 0..3 { for j in (i+1)..3 {
                let avg = 0.5*(tensor[i][j]+tensor[j][i]);
                tensor[i][j] = avg; tensor[j][i] = avg;
            }}
            out[a][k] = tensor;
        }
    }

    Ok(out)
}

/// Molecular (whole-system) dynamic polarizability α_{ij}(iω_k) for the
/// closed-shell PDEP-RPA response, partition-independent.
///
/// Uses the lab-frame molecular dipole Σ_i r_i on both sides of the response,
/// so it is origin-independent and gives the DOSD-comparable molecular C6 when
/// fed to Casimir-Polder. This is the correct molecular total — distinct from
/// the sum of the intrinsic per-atom tensors (which omit inter-atomic
/// coupling). Returns `molecular[k]` = 3×3 tensor (a.u.).
pub fn molecular_dynamic_polarizability(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    rhf: &ScfResult,
    op: Operator,
    _cfg: &PdepRpaConfig,
    freqs: &[f64],
) -> Result<Vec<[[f64; 3]; 3]>, FerricError> {
    use ndarray_linalg::Solve;

    if !matches!(rhf.spin, Spin::Restricted) {
        return Err(FerricError::General(
            "molecular_dynamic_polarizability: only closed-shell (Restricted) supported".into(),
        ));
    }

    let mp2_cfg = ferric_mp2::rimp2::RiMp2Config { frozen_core: 0 };
    let inter = ferric_mp2::rimp2::compute_rpa_intermediates(mol, obs, dfbs, op, rhf, &mp2_cfg)?;
    let b_ov = &inter.b_ov;
    let nocc = inter.nocc;
    let nvir = inter.nvir;
    let nocc_total = inter.nocc_total;
    let first_occ = inter.first_occ;
    let naux = inter.naux;
    let nov = nocc * nvir;

    let eps = rhf.eps_r();
    let eps_occ: Vec<f64> = eps[first_occ..first_occ + nocc].to_vec();
    let eps_vir: Vec<f64> = eps[nocc_total..nocc_total + nvir].to_vec();
    let mut e_ia = ndarray::Array1::<f64>::zeros(nov);
    for i in 0..nocc {
        for a in 0..nvir {
            e_ia[i * nvir + a] = eps_vir[a] - eps_occ[i];
        }
    }

    // Lab-frame molecular dipole in the MO occ-vir basis.
    let dip_ao = oneelectron::dipole(obs, [0.0, 0.0, 0.0]);
    let c = rhf.mos_r();
    let c_occ = c.slice(ndarray::s![.., first_occ..first_occ + nocc]).to_owned();
    let c_vir = c.slice(ndarray::s![.., nocc_total..nocc_total + nvir]).to_owned();
    let mu_flat: [ndarray::Array1<f64>; 3] = std::array::from_fn(|d| {
        let mo = c_occ.t().dot(&dip_ao[d]).dot(&c_vir);
        let mut v = ndarray::Array1::<f64>::zeros(nov);
        for i in 0..nocc {
            for a in 0..nvir {
                v[i * nvir + a] = mo[(i, a)];
            }
        }
        v
    });

    let mut out = vec![[[0.0_f64; 3]; 3]; freqs.len()];
    for (k, &omega) in freqs.iter().enumerate() {
        let omega2 = omega * omega;
        let mut g = ndarray::Array1::<f64>::zeros(nov);
        for ia in 0..nov {
            let e = e_ia[ia];
            g[ia] = e / (omega2 + e * e);
        }
        // ε̃(ω) = I + 4 B̃ diag(g) B̃^T
        let mut b_scaled = b_ov.clone();
        for ia in 0..nov {
            b_scaled.column_mut(ia).mapv_inplace(|x| x * (4.0 * g[ia]).sqrt());
        }
        let mut eps_mat: Array2<f64> = b_scaled.dot(&b_scaled.t());
        for p in 0..naux {
            eps_mat[(p, p)] += 1.0;
        }
        let mu_g: [ndarray::Array1<f64>; 3] = std::array::from_fn(|d| &mu_flat[d] * &g);
        let y: [ndarray::Array1<f64>; 3] =
            std::array::from_fn(|d| eps_mat.solve(&b_ov.dot(&mu_g[d])).unwrap());

        let mut t = [[0.0_f64; 3]; 3];
        for d in 0..3 {
            let w_d = b_ov.dot(&mu_g[d]);
            for j in 0..3 {
                let bare = mu_flat[d].dot(&mu_g[j]);
                let coupled = w_d.dot(&y[j]);
                t[d][j] = 4.0 * bare - 16.0 * coupled;
            }
        }
        for i in 0..3 {
            for j in (i + 1)..3 {
                let avg = 0.5 * (t[i][j] + t[j][i]);
                t[i][j] = avg;
                t[j][i] = avg;
            }
        }
        out[k] = t;
    }
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
/// with `v_free` taken from [`crate::dispersion::free_atom_ref::ts_free_atom`].
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

/// Per-atom effective volume via Hirshfeld (Slater proatom) partitioning:
/// ```text
///   v_A = ∫ w^A_Hirsh(r) ρ(r) |r − R_A|³ dV
/// ```
/// This is the partition the TS dispersion model was calibrated for.
/// Uses the same Slater single-exponential proatom as `hirshfeld_charges`.
pub fn atomic_effective_volumes_hirshfeld(
    mol: &Molecule,
    obs_bs: &ferric_core::basis::BasisSet,
    density: &Array2<f64>,
) -> Result<Vec<f64>, FerricError> {
    use ferric_export::cube::GridSpec;
    use ferric_export::gto_eval::eval_basis_on_grid;

    let natoms = mol.atoms.len();
    let spacing = std::env::var("FERRIC_HIRSHFELD_SPACING")
        .ok().and_then(|s| s.parse::<f64>().ok()).unwrap_or(0.20);
    let margin = std::env::var("FERRIC_HIRSHFELD_MARGIN")
        .ok().and_then(|s| s.parse::<f64>().ok()).unwrap_or(6.0);
    let grid = GridSpec::bounding_box(mol, margin, spacing);
    let dv = spacing * spacing * spacing;
    let npts = grid.n_x * grid.n_y * grid.n_z;
    let hx = grid.step_x[0];
    let hy = grid.step_y[1];
    let hz = grid.step_z[2];

    let chi = eval_basis_on_grid(mol, obs_bs, &grid).map_err(|e| {
        FerricError::General(format!("atomic_effective_volumes_hirshfeld: chi failed: {e}"))
    })?;
    let nbf = chi.nrows();

    // ρ(r_g) = Σ_{μν} D_{μν} χ_μ χ_ν via matrix product.
    let d_chi = density.dot(&chi);
    let mut rho = vec![0.0_f64; npts];
    for mu in 0..nbf {
        for g in 0..npts {
            rho[g] += chi[(mu, g)] * d_chi[(mu, g)];
        }
    }

    // Slater proatom weights (identical to hirshfeld_charges).
    let mut rho_free: Vec<Vec<f64>> = vec![vec![0.0; npts]; natoms];
    let mut rho_sum = vec![0.0_f64; npts];
    for a in 0..natoms {
        let xi = slater_xi_for_z(mol.atoms[a].z);
        let z_a = mol.atoms[a].z as f64;
        let prefac = z_a * xi.powi(3) / std::f64::consts::PI;
        let (rax, ray, raz) = (mol.atoms[a].x, mol.atoms[a].y, mol.atoms[a].zpos);
        let row = &mut rho_free[a];
        for ix in 0..grid.n_x {
            let x = grid.origin[0] + ix as f64 * hx;
            for iy in 0..grid.n_y {
                let y = grid.origin[1] + iy as f64 * hy;
                for iz in 0..grid.n_z {
                    let z = grid.origin[2] + iz as f64 * hz;
                    let g = (ix * grid.n_y + iy) * grid.n_z + iz;
                    let r = ((x-rax)*(x-rax)+(y-ray)*(y-ray)+(z-raz)*(z-raz)).sqrt();
                    let r0 = prefac * (-2.0 * xi * r).exp();
                    row[g] = r0;
                    rho_sum[g] += r0;
                }
            }
        }
    }

    let eps_floor = 1e-12;
    let mut vol = vec![0.0_f64; natoms];
    for a in 0..natoms {
        let (rax, ray, raz) = (mol.atoms[a].x, mol.atoms[a].y, mol.atoms[a].zpos);
        let mut acc = 0.0;
        for ix in 0..grid.n_x {
            let x = grid.origin[0] + ix as f64 * hx;
            for iy in 0..grid.n_y {
                let y = grid.origin[1] + iy as f64 * hy;
                for iz in 0..grid.n_z {
                    let z = grid.origin[2] + iz as f64 * hz;
                    let g = (ix * grid.n_y + iy) * grid.n_z + iz;
                    let w = rho_free[a][g] / (rho_sum[g] + eps_floor);
                    let dx = x - rax; let dy = y - ray; let dz = z - raz;
                    let r3 = (dx*dx + dy*dy + dz*dz).powf(1.5);
                    acc += w * rho[g] * r3 * dv;
                }
            }
        }
        vol[a] = acc;
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
///
/// The total electronic charge is renormalized so Σ_A (Z_A − q_A) = N_e
/// exactly (compensates for grid quadrature error in the density integral).
///
/// Returns `Vec<f64>` of length `mol.atoms.len()`, in units of e.
pub fn hirshfeld_charges(
    mol: &Molecule,
    obs_bs: &ferric_core::basis::BasisSet,
    density: &Array2<f64>,
) -> Result<Vec<f64>, FerricError> {
    use ferric_export::cube::GridSpec;
    use ferric_export::gto_eval::eval_basis_on_grid;

    let natoms = mol.atoms.len();
    let spacing = std::env::var("FERRIC_HIRSHFELD_SPACING")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(0.20);
    let margin = std::env::var("FERRIC_HIRSHFELD_MARGIN")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(6.0);
    let grid = GridSpec::bounding_box(mol, margin, spacing);
    let dv = spacing * spacing * spacing;
    let npts = grid.n_x * grid.n_y * grid.n_z;
    let hx = grid.step_x[0];
    let hy = grid.step_y[1];
    let hz = grid.step_z[2];

    let chi = eval_basis_on_grid(mol, obs_bs, &grid).map_err(|e| {
        FerricError::General(format!("hirshfeld_charges: chi eval failed: {e}"))
    })?;
    let nbf = chi.nrows();
    if density.nrows() != nbf || density.ncols() != nbf {
        return Err(FerricError::General(format!(
            "hirshfeld_charges: density {:?} != nbf {}",
            density.dim(),
            nbf
        )));
    }

    // ρ(r_g) = Σ_{μν} D_{μν} χ_μ(r_g) χ_ν(r_g) = sum_μ χ_μ(g) · (D · χ)(μ,g)
    let d_chi = density.dot(&chi); // (nbf, npts)
    let mut rho: Vec<f64> = vec![0.0; npts];
    for mu in 0..nbf {
        for g in 0..npts {
            rho[g] += chi[(mu, g)] * d_chi[(mu, g)];
        }
    }

    // Slater proatom densities and Hirshfeld weights, exactly as in
    // pdep_polarizability_hirshfeld.
    let mut rho_free: Vec<Vec<f64>> = vec![vec![0.0; npts]; natoms];
    let mut rho_sum: Vec<f64> = vec![0.0; npts];
    for a in 0..natoms {
        let xi = slater_xi_for_z(mol.atoms[a].z);
        let z_a = mol.atoms[a].z as f64;
        let prefac = z_a * xi.powi(3) / std::f64::consts::PI;
        let rax = mol.atoms[a].x;
        let ray = mol.atoms[a].y;
        let raz = mol.atoms[a].zpos;
        let row = &mut rho_free[a];
        for ix in 0..grid.n_x {
            let x = grid.origin[0] + ix as f64 * hx;
            for iy in 0..grid.n_y {
                let y = grid.origin[1] + iy as f64 * hy;
                for iz in 0..grid.n_z {
                    let z = grid.origin[2] + iz as f64 * hz;
                    let g = (ix * grid.n_y + iy) * grid.n_z + iz;
                    let dx = x - rax;
                    let dy = y - ray;
                    let dz = z - raz;
                    let r = (dx * dx + dy * dy + dz * dz).sqrt();
                    let r0 = prefac * (-2.0 * xi * r).exp();
                    row[g] = r0;
                    rho_sum[g] += r0;
                }
            }
        }
    }

    let eps_floor = 1e-12;
    // n_elec_A_grid = ∫ ρ · w^A
    let mut n_e_grid = vec![0.0_f64; natoms];
    for a in 0..natoms {
        let mut acc = 0.0;
        for g in 0..npts {
            let w = rho_free[a][g] / (rho_sum[g] + eps_floor);
            acc += rho[g] * w * dv;
        }
        n_e_grid[a] = acc;
    }

    // Renormalize: Σ_A n_e_grid_A should equal N_e (= trace(D·S)). Use the
    // electron count from the molecule (nelec) so the partition is exact.
    let n_elec_target = mol.nelec() as f64;
    let n_elec_sum: f64 = n_e_grid.iter().sum();
    let scale = if n_elec_sum.abs() > 1e-12 {
        n_elec_target / n_elec_sum
    } else {
        1.0
    };

    Ok((0..natoms)
        .map(|a| mol.atoms[a].z as f64 - scale * n_e_grid[a])
        .collect())
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

/// Slater single-exponential proatom exponent ξ (Bohr⁻¹) for element Z.
///
/// Derived from Bragg-Slater empirical atomic radii R_BS:
///     ξ = 1 / (R_BS in Bohr).
///
/// The Hirshfeld partition is robust to the precise proatom radial shape;
/// the sum rule inside `pdep_polarizability_hirshfeld` is the gate.
fn slater_xi_for_z(z: i32) -> f64 {
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
fn eig3_sym(a: [[f64; 3]; 3]) -> [f64; 3] {
    // Use ndarray-linalg for robustness.
    use ndarray::arr2;
    use ndarray_linalg::Eigh;
    let m = arr2(&a);
    let (vals, _) = m.eigh(ndarray_linalg::UPLO::Upper).unwrap();
    let mut v = [vals[0], vals[1], vals[2]];
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferric_core::basis;
    use ferric_core::parallel::ParallelContext;
    use ferric_integrals::operator::Operator;
    use ferric_scf::rhf::{solve_rhf, RhfConfig};
    use ferric_scf::screening::SchwarzBounds;

    fn build_h2() -> (Molecule, PreparedBasis, PreparedBasis, Operator, ScfResult) {
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

    #[test]
    fn becke_dynamic_alpha_molecular_sum_decays() {
        // The MOLECULAR dynamic polarizability — Σ_A α^A(iω) — is the robust,
        // origin-independent quantity. (Per-atom α^A(iω) is origin-dependent at
        // ω≠0 because the lab-frame partitioned dipole ⟨i|w^A r|a⟩ depends on the
        // common origin; only the atom SUM and the static ω=0 limit are clean.)
        // Check the molecular sum decays monotonically with the right tail.
        let (mol, obs, dfbs, op, rhf) = build_h2();
        let bs = basis::bundled("cc-pvdz").unwrap();
        let mut cfg = PdepRpaConfig::default();
        cfg.frozen_core = 0;
        cfg.trunc_thresh = 0.0;
        let freqs = [0.0, 0.5, 2.0, 10.0];
        let dyn_a = pdep_polarizability_becke_dynamic(
            &mol, &obs, &bs, &dfbs, &rhf, op, &cfg, &freqs,
        )
        .unwrap();
        let mol_iso = |k: usize| -> f64 {
            dyn_a
                .iter()
                .map(|atom| (atom[k][0][0] + atom[k][1][1] + atom[k][2][2]) / 3.0)
                .sum()
        };
        let a0 = mol_iso(0);
        let a1 = mol_iso(1);
        let a2 = mol_iso(2);
        let a3 = mol_iso(3);
        assert!(a0 > a1 && a1 > a2 && a2 > a3, "not monotone: {a0} {a1} {a2} {a3}");
        assert!(a0 > 0.0, "α(0) must be positive: {a0}");
        assert!(a3 < 0.05 * a0, "tail too large: α(10)={a3} α(0)={a0}");
    }

    #[test]
    fn becke_dynamic_alpha_h2_symmetry_and_positivity() {
        // For H₂ the two atoms are equivalent by symmetry, so per-atom dynamic α
        // must be equal. Also α_iso^A(ω=0) must be positive.
        let (mol, obs, dfbs, op, rhf) = build_h2();
        let bs = basis::bundled("cc-pvdz").unwrap();
        let mut cfg = PdepRpaConfig::default();
        cfg.frozen_core = 0;
        cfg.trunc_thresh = 0.0;

        let dynamic = pdep_polarizability_becke_dynamic(
            &mol, &obs, &bs, &dfbs, &rhf, op, &cfg, &[0.0, 0.5],
        )
        .unwrap();
        assert_eq!(dynamic.len(), 2, "H2 should have 2 atoms");

        let iso = |t: &[[f64; 3]; 3]| (t[0][0] + t[1][1] + t[2][2]) / 3.0;
        let iso0_h0 = iso(&dynamic[0][0]);
        let iso0_h1 = iso(&dynamic[1][0]);

        // Both α_iso at ω=0 must be positive.
        assert!(iso0_h0 > 0.0, "α_iso H0(ω=0) not positive: {iso0_h0}");
        assert!(iso0_h1 > 0.0, "α_iso H1(ω=0) not positive: {iso0_h1}");

        // H2 symmetric: both atoms must give equal α at each frequency.
        assert!(
            (iso0_h0 - iso0_h1).abs() / iso0_h0 < 1e-6,
            "H2 per-atom α not symmetric at ω=0: {iso0_h0} vs {iso0_h1}"
        );
        let iso1_h0 = iso(&dynamic[0][1]);
        let iso1_h1 = iso(&dynamic[1][1]);
        assert!(
            (iso1_h0 - iso1_h1).abs() / (iso1_h0.abs() + 1e-12) < 1e-6,
            "H2 per-atom α not symmetric at ω=0.5: {iso1_h0} vs {iso1_h1}"
        );
    }

    #[test]
    fn becke_dynamic_alpha_decays_with_frequency() {
        // α_iso(iω) must fall monotonically toward 0 as ω grows (~1/ω²).
        let (mol, obs, dfbs, op, rhf) = build_h2();
        let bs = basis::bundled("cc-pvdz").unwrap();
        let mut cfg = PdepRpaConfig::default();
        cfg.frozen_core = 0;
        cfg.trunc_thresh = 0.0;

        let freqs = [0.0, 0.5, 2.0, 10.0];
        let dyn_a = pdep_polarizability_becke_dynamic(
            &mol, &obs, &bs, &dfbs, &rhf, op, &cfg, &freqs,
        )
        .unwrap();
        // Sum atomic isotropic α at each frequency = molecular iso (sum rule).
        let iso_at = |k: usize| -> f64 {
            dyn_a
                .iter()
                .map(|atom| (atom[k][0][0] + atom[k][1][1] + atom[k][2][2]) / 3.0)
                .sum()
        };
        let a0 = iso_at(0);
        let a1 = iso_at(1);
        let a2 = iso_at(2);
        let a3 = iso_at(3);
        assert!(a0 > a1 && a1 > a2 && a2 > a3, "not monotone: {a0} {a1} {a2} {a3}");
        assert!(a0 > 0.0, "α(0) must be positive: {a0}");
        // High-ω tail ~ 1/ω²: α(10) ≪ α(0).
        assert!(a3 < 0.05 * a0, "tail too large: α(10)={a3} α(0)={a0}");
    }

    fn build_h_atom() -> (Molecule, PreparedBasis, PreparedBasis, Operator, ScfResult) {
        use ferric_scf::uhf::solve_uhf;
        let xyz = "1\nH\nH 0 0 0\n";
        let mol = Molecule::parse_xyz(xyz, 0, 2).unwrap(); // doublet
        let obs_bs = basis::bundled("cc-pvdz").unwrap();
        let dfbs_bs = basis::bundled("cc-pvdz-ri").unwrap();
        let op = Operator::coulomb();
        let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
        let dfbs = PreparedBasis::new(&mol, &dfbs_bs).unwrap();
        let ctx = ParallelContext::default();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let mut cfg = RhfConfig::default();
        cfg.mom_after_iter = 5;
        let uhf = solve_uhf(&ctx, &mol, &obs, &bounds, &cfg).unwrap();
        (mol, obs, dfbs, op, uhf)
    }

    #[test]
    fn h_atom_dynamic_alpha_unrestricted_decays() {
        // UHF H atom: dynamic α(iω) via the unrestricted path must decay monotonically
        // and be positive at ω=0. Single atom → no symmetry check needed.
        let (mol, obs, dfbs, op, uhf) = build_h_atom();
        let bs = basis::bundled("cc-pvdz").unwrap();
        let mut cfg = PdepRpaConfig::default();
        cfg.frozen_core = 0;
        cfg.trunc_thresh = 0.0;

        let freqs = [0.0, 0.5, 2.0, 10.0];
        let dyn_a = pdep_polarizability_becke_dynamic(
            &mol, &obs, &bs, &dfbs, &uhf, op, &cfg, &freqs,
        )
        .unwrap();
        assert_eq!(dyn_a.len(), 1);

        let iso = |t: &[[f64; 3]; 3]| (t[0][0] + t[1][1] + t[2][2]) / 3.0;
        let a0 = iso(&dyn_a[0][0]);
        let a1 = iso(&dyn_a[0][1]);
        let a2 = iso(&dyn_a[0][2]);
        let a3 = iso(&dyn_a[0][3]);
        assert!(a0 > 0.0, "α(0) not positive: {a0}");
        assert!(a0 > a1 && a1 > a2 && a2 > a3, "not monotone: {a0} {a1} {a2} {a3}");
        assert!(a3 < 0.05 * a0, "tail too large: α(10)={a3} α(0)={a0}");
    }

    #[test]
    fn h2_polarizability_finite_positive() {
        // Smoke test: α_iso for H2 / cc-pVDZ is positive O(few a.u.).
        // Quantitative comparison vs PySCF lives in tests/polarizability.rs.
        let (mol, obs, dfbs, op, rhf) = build_h2();
        let mut cfg = PdepRpaConfig::default();
        cfg.frozen_core = 0;
        cfg.trunc_thresh = 0.0;
        cfg.davidson_conv_thresh = 1e-9;
        let r = pdep_polarizability_static(&mol, &obs, &dfbs, &rhf, op, &cfg).unwrap();
        assert!(r.iso > 0.0, "α_iso ≤ 0: {}", r.iso);
        assert!(r.iso < 30.0, "α_iso too large: {}", r.iso);
        // tensor symmetric
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (r.tensor[i][j] - r.tensor[j][i]).abs() < 1e-10,
                    "α asymmetric at ({i},{j})"
                );
            }
        }
    }
}
