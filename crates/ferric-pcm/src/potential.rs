//! Solute electrostatic potential at cavity tesserae, and the reaction-field
//! one-electron AO operator built from the resulting apparent surface
//! charges.
//!
//! Mirrors `ferric_rpa::properties::esp_at_atoms` exactly, just evaluated at
//! cavity tesserae instead of nuclear positions — see that module's doc
//! comment for the libint sign-convention derivation this reuses
//! (`⟨μ| −1/|r−R| |ν⟩` from a Z=+1 probe charge).

use std::os::raw::c_int;

use ferric_core::external_potential::PointCharge;
use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::engine::Engine;
use ferric_integrals::ffi::{self, CAtom};
use ndarray::Array2;

use crate::cavity::Tessera;

/// Evaluate the total (nuclear + electronic) solute electrostatic potential
/// at each cavity tessera, in Hartree.
///
/// ```text
///     v_k = Σ_B Z_B / |r_k − R_B|  −  Σ_{μν} D_{μν} ⟨μ| 1/|r − r_k| |ν⟩
/// ```
///
/// No self-interaction subtlety here (unlike `esp_at_atoms`'s atom-centered
/// case): every tessera sits strictly outside every real nucleus by
/// cavity construction, so the nuclear sum has no `A == k` singularity to
/// skip.
pub fn solute_potential_at_tesserae(
    mol: &Molecule,
    prep: &PreparedBasis,
    density: &Array2<f64>,
    tess: &[Tessera],
) -> Result<Vec<f64>, FerricError> {
    let nbas = prep.nbasis();
    if density.shape() != [nbas, nbas] {
        return Err(FerricError::General(format!(
            "solute_potential_at_tesserae: density shape {:?} != ({nbas},{nbas})",
            density.shape()
        )));
    }

    let dims = prep.shell_dims();
    let offs = prep.shell_offsets();
    let nsh = prep.nshells();

    // Serial (not rayon) — cavity tessera counts for small/medium molecules
    // (few hundred to a couple thousand) are far below the shell-pair
    // parallel-worthwhile threshold used elsewhere (oneelectron.rs's
    // PAR_SHELL_PAIR_THRESHOLD), and this routine is called once per SCF
    // iteration, so keeping it simple/serial avoids re-litigating the
    // engine-pool-per-worker plumbing for a first correct implementation.
    let mut eng = Engine::new_1e(ffi::OP_NUCLEAR, prep, 1e-14)
        .map_err(|e| FerricError::General(format!("solute_potential_at_tesserae: engine init failed: {e}")))?;

    let mut out = Vec::with_capacity(tess.len());
    for t in tess {
        let probe = [CAtom {
            atomic_number: 1.0,
            x: t.position[0],
            y: t.position[1],
            z: t.position[2],
        }];
        // SAFETY: probe is a stack-local [CAtom; 1] that outlives the FFI call;
        // eng is a valid Engine handle; len=1 matches the array.
        let rc = unsafe {
            ffi::scf_engine_set_point_charges(eng.handle_mut(), probe.as_ptr(), probe.len() as c_int)
        };
        if rc < 0 {
            return Err(FerricError::General(format!(
                "solute_potential_at_tesserae: set_point_charges failed (rc={rc})"
            )));
        }

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
                            v_elec += 2.0 * density[(o1 + i, o2 + j)] * block[i * n2 + j];
                        }
                    }
                }
            }
        }

        let mut v_nuc = 0.0_f64;
        for atom in &mol.atoms {
            if atom.ghost {
                continue;
            }
            let dx = t.position[0] - atom.x;
            let dy = t.position[1] - atom.y;
            let dz = t.position[2] - atom.zpos;
            let r = (dx * dx + dy * dy + dz * dz).sqrt();
            if r < 1e-8 {
                return Err(FerricError::General(
                    "solute_potential_at_tesserae: tessera coincides with a nucleus (degenerate cavity)".into(),
                ));
            }
            v_nuc += atom.effective_z() as f64 / r;
        }

        out.push(v_nuc + v_elec);
    }

    Ok(out)
}

/// Build the one-electron reaction-field AO operator `V_pcm` from the
/// apparent surface charges `q` at cavity tesserae:
///
/// ```text
///     V_pcm_{μν} = Σ_k q_k · ⟨μ| −1/|r − r_k| |ν⟩
/// ```
///
/// This is exactly the same sign convention `nuclear_with_external` uses
/// for a generic classical `PointCharge` (treating each apparent surface
/// charge as a fixed point charge for the purpose of the one-electron
/// integral) — see the module doc for the derivation of why this is the
/// correct one-electron potential energy operator for an electron
/// interacting with the reaction field.
pub fn build_reaction_field_operator(
    prep: &PreparedBasis,
    tess: &[Tessera],
    q: &[f64],
) -> Result<Array2<f64>, FerricError> {
    if tess.len() != q.len() {
        return Err(FerricError::General(format!(
            "build_reaction_field_operator: {} tesserae but {} charges",
            tess.len(),
            q.len()
        )));
    }
    let point_charges: Vec<PointCharge> = tess
        .iter()
        .zip(q.iter())
        .map(|(t, &qk)| PointCharge {
            q: qk,
            x: t.position[0],
            y: t.position[1],
            z: t.position[2],
        })
        .collect();

    let mut eng = Engine::new_1e(ffi::OP_NUCLEAR, prep, 1e-14)
        .map_err(|e| FerricError::General(format!("build_reaction_field_operator: engine init failed: {e}")))?;
    eng.set_point_charges_extra(prep, &point_charges)?;

    // Reuse the same "real atoms zeroed out, only extra charges active"
    // trick as nuclear_with_external: we want ONLY the tessera-charge
    // contribution, not the real nuclear attraction (that's already in
    // hcore separately). set_point_charges_extra appends `extra` AFTER
    // prep.atoms() — but prep.atoms() carries the REAL nuclear charges, so
    // calling it directly would double-count V_nuc. Instead, build a probe
    // atom list with all-real-atom charges zeroed (position kept, charge 0)
    // plus the tessera charges appended; a zero-charge point contributes
    // nothing to the nuclear-attraction integral.
    let zeroed_atoms: Vec<CAtom> = prep
        .atoms()
        .iter()
        .map(|a| CAtom { atomic_number: 0.0, x: a.x, y: a.y, z: a.z })
        .collect();
    let mut all_atoms = zeroed_atoms;
    all_atoms.extend(point_charges.iter().map(|pc| CAtom {
        atomic_number: pc.q,
        x: pc.x,
        y: pc.y,
        z: pc.z,
    }));
    // SAFETY: all_atoms is a Vec<CAtom> that outlives the FFI call;
    // eng is a valid Engine handle; len matches the vec length.
    let rc = unsafe {
        ffi::scf_engine_set_point_charges(eng.handle_mut(), all_atoms.as_ptr(), all_atoms.len() as c_int)
    };
    if rc < 0 {
        return Err(FerricError::General(format!(
            "build_reaction_field_operator: set_point_charges failed (rc={rc})"
        )));
    }

    let n = prep.nbasis();
    let nsh = prep.nshells();
    let dims = prep.shell_dims();
    let offs = prep.shell_offsets();
    let mut out = Array2::<f64>::zeros((n, n));
    for s1 in 0..nsh {
        for s2 in 0..=s1 {
            let block = eng.compute_1e_block(prep, s1, s2);
            let n1 = dims[s1];
            let n2 = dims[s2];
            for i in 0..n1 {
                for j in 0..n2 {
                    let v = block[i * n2 + j];
                    out[(offs[s1] + i, offs[s2] + j)] = v;
                    out[(offs[s2] + j, offs[s1] + i)] = v;
                }
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cavity::{build_cavity, CavityConfig};
    use ferric_core::basis;

    fn water_sto3g() -> (Molecule, PreparedBasis) {
        let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
        let bs = basis::bundled("sto-3g").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        (mol, prep)
    }

    #[test]
    fn potential_at_tesserae_matches_hand_calc_for_zero_density() {
        // With D=0, v_k should equal the pure nuclear-sum ESP, hand-computable.
        let (mol, prep) = water_sto3g();
        let tess = build_cavity(&mol, &CavityConfig::default()).unwrap();
        let d0 = Array2::<f64>::zeros((prep.nbasis(), prep.nbasis()));
        let v = solute_potential_at_tesserae(&mol, &prep, &d0, &tess).unwrap();
        assert_eq!(v.len(), tess.len());
        // Hand check tessera 0 against a direct sum.
        let mut expected = 0.0;
        for atom in &mol.atoms {
            let dx = tess[0].position[0] - atom.x;
            let dy = tess[0].position[1] - atom.y;
            let dz = tess[0].position[2] - atom.zpos;
            let r = (dx * dx + dy * dy + dz * dz).sqrt();
            expected += atom.z as f64 / r;
        }
        assert!((v[0] - expected).abs() < 1e-10, "got {}, expected {}", v[0], expected);
    }

    #[test]
    fn reaction_field_operator_is_symmetric_and_zero_for_zero_charges() {
        let (mol, prep) = water_sto3g();
        let tess = build_cavity(&mol, &CavityConfig::default()).unwrap();
        let q = vec![0.0; tess.len()];
        let vmat = build_reaction_field_operator(&prep, &tess, &q).unwrap();
        assert!(vmat.iter().all(|&x| x == 0.0));

        let mut q_nonzero = vec![0.0; tess.len()];
        q_nonzero[0] = 0.01;
        let vmat2 = build_reaction_field_operator(&prep, &tess, &q_nonzero).unwrap();
        let n = prep.nbasis();
        for i in 0..n {
            for j in 0..n {
                assert!((vmat2[(i, j)] - vmat2[(j, i)]).abs() < 1e-12);
            }
        }
        assert!(vmat2.iter().any(|&x| x != 0.0));
    }

    #[test]
    fn mismatched_lengths_error() {
        let (mol, prep) = water_sto3g();
        let tess = build_cavity(&mol, &CavityConfig::default()).unwrap();
        let q = vec![0.0; tess.len() + 1];
        assert!(build_reaction_field_operator(&prep, &tess, &q).is_err());
    }
}
