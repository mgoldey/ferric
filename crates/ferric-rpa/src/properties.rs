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
            ffi::goscf_engine_set_point_charges(
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
    _cfg: &PdepRpaConfig,
) -> Result<PolarizabilityResult, FerricError> {
    if !matches!(rhf.spin, Spin::Restricted) {
        return Err(FerricError::General(
            "pdep_polarizability_static: only closed-shell (Restricted) supported".into(),
        ));
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
