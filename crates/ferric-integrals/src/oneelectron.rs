//! One-electron integral matrices: overlap (S), kinetic (T), nuclear (V), and core Hamiltonian (H).

use crate::basis_bridge::PreparedBasis;
use crate::ecp::{ecp_matrix_spherical, gto_norm, EcpCenter, EcpGaussianShell};
use crate::engine::Engine;
use crate::ffi;
use ferric_core::basis::BasisSet;
use ferric_core::mol::Molecule;
use ndarray::Array2;

/// Build a symmetric matrix from a one-electron engine by iterating over upper-triangle shell pairs.
fn build_symmetric(prep: &PreparedBasis, mut eng: Engine) -> Array2<f64> {
    let n = prep.nbasis();
    let nsh = prep.nshells();
    let dims = prep.shell_dims();
    let offs = prep.shell_offsets();
    let mut out = Array2::zeros((n, n));
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
    out
}

/// Compute the overlap matrix S, shape (nbasis, nbasis).
pub fn overlap(prep: &PreparedBasis) -> Array2<f64> {
    let eng = Engine::new_1e(ffi::OP_OVERLAP, prep, 1e-14).unwrap();
    build_symmetric(prep, eng)
}

/// Compute the kinetic energy matrix T, shape (nbasis, nbasis).
pub fn kinetic(prep: &PreparedBasis) -> Array2<f64> {
    let eng = Engine::new_1e(ffi::OP_KINETIC, prep, 1e-14).unwrap();
    build_symmetric(prep, eng)
}

/// Compute the nuclear attraction matrix V, shape (nbasis, nbasis).
pub fn nuclear(prep: &PreparedBasis) -> Array2<f64> {
    let mut eng = Engine::new_1e(ffi::OP_NUCLEAR, prep, 1e-14).unwrap();
    eng.set_point_charges(prep);
    build_symmetric(prep, eng)
}

/// Compute the core Hamiltonian H = T + V, shape (nbasis, nbasis).
///
/// `V` is the nuclear-attraction matrix; for ECP-treated atoms its point charges
/// are already the effective (valence-only) `Z − n_core` set up in
/// [`PreparedBasis::new`]. This function does **not** add the ECP projector
/// `V_ECP` — use [`hcore_ecp`] when the basis carries ECPs.
pub fn hcore(prep: &PreparedBasis) -> Array2<f64> {
    let t = kinetic(prep);
    let v = nuclear(prep);
    t + v
}

/// Compute the dense spherical ECP projector matrix `V_ECP`, shape
/// (nbasis, nbasis), for the given molecule + ECP-carrying basis set.
///
/// Returns `None` (and does zero work) when the basis carries no ECPs. The shell
/// iteration order mirrors [`PreparedBasis::new`] exactly, so the resulting AO
/// ordering matches libint's spherical basis — the matrix can be added to
/// `hcore` directly.
///
/// Per the libecpint convention (see [`crate::ecp`]), each primitive's stored
/// contraction coefficient is multiplied by `gto_norm(l, α)` before being handed
/// to the shim.
pub fn ecp_potential(mol: &Molecule, bs: &BasisSet) -> Option<Array2<f64>> {
    if bs.ecps.is_empty() {
        return None;
    }
    let mut shells: Vec<EcpGaussianShell> = Vec::new();
    let mut ecps: Vec<EcpCenter> = Vec::new();
    for atom in &mol.atoms {
        let center = [atom.x, atom.y, atom.zpos];
        // Gaussian shells (real Z → real basis), gto_norm folded per primitive.
        if let Some(tmpls) = bs.for_element(atom.z) {
            for sh in tmpls {
                let coefs: Vec<f64> = sh
                    .exponents
                    .iter()
                    .zip(sh.coefficients.iter())
                    .map(|(&a, &c)| c * gto_norm(sh.l, a))
                    .collect();
                shells.push(EcpGaussianShell {
                    l: sh.l,
                    center,
                    exponents: sh.exponents.clone(),
                    coefficients: coefs,
                });
            }
        }
        // ECP center for this atom, if any. Flatten EcpDef channels into the
        // (am, n, exp, coef) primitive lists libecpint expects.
        if let Some(def) = bs.ecp_for_element(atom.z) {
            let mut ams = Vec::new();
            let mut ns = Vec::new();
            let mut exponents = Vec::new();
            let mut coefficients = Vec::new();
            for ch in &def.shells {
                for t in &ch.terms {
                    ams.push(ch.angular_momentum);
                    ns.push(t.r_exp);
                    exponents.push(t.gexp);
                    coefficients.push(t.coef);
                }
            }
            ecps.push(EcpCenter { center, ams, ns, exponents, coefficients });
        }
    }
    if ecps.is_empty() {
        return None;
    }
    let n: usize = shells.iter().map(|s| (2 * s.l + 1) as usize).sum();
    let flat = ecp_matrix_spherical(&shells, &ecps)
        .expect("ecp_matrix_spherical failed building V_ECP");
    Some(Array2::from_shape_vec((n, n), flat).expect("V_ECP shape"))
}

/// Core Hamiltonian including the ECP projector:
/// `H = T + V_nuc(effective Z) + V_ECP`.
///
/// Identical to [`hcore`] when `bs` carries no ECPs (V_ECP is skipped at zero
/// cost). `prep` must have been built from the same `mol` + `bs`.
pub fn hcore_ecp(prep: &PreparedBasis, mol: &Molecule, bs: &BasisSet) -> Array2<f64> {
    let mut h = hcore(prep);
    if let Some(vecp) = ecp_potential(mol, bs) {
        assert_eq!(
            vecp.dim(),
            h.dim(),
            "V_ECP dimension {:?} != hcore dimension {:?}",
            vecp.dim(),
            h.dim()
        );
        h += &vecp;
    }
    h
}

/// Compute the 3 electric dipole matrices ⟨μ|(r - origin)|ν⟩, shape (nbasis, nbasis) each.
/// `origin` is in Bohr. Returns [x_mat, y_mat, z_mat].
pub fn dipole(prep: &PreparedBasis, origin: [f64; 3]) -> [Array2<f64>; 3] {
    let nbas = prep.nbasis();
    let mut flat = vec![0.0f64; 3 * nbas * nbas];
    let ret = unsafe {
        ffi::scf_compute_dipole(
            prep.handle(),
            origin.as_ptr(),
            nbas as std::os::raw::c_int,
            flat.as_mut_ptr(),
        )
    };
    assert!(ret >= 0, "scf_compute_dipole failed: {}", ret);
    let make_mat = |offset: usize| {
        let slice = &flat[offset..offset + nbas * nbas];
        Array2::from_shape_vec((nbas, nbas), slice.to_vec()).unwrap()
    };
    [make_mat(0), make_mat(nbas * nbas), make_mat(2 * nbas * nbas)]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis_bridge::PreparedBasis;
    use ferric_core::basis;
    use ferric_core::mol::Molecule;

    fn water_sto3g() -> PreparedBasis {
        let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
        let bs = basis::bundled("sto-3g").unwrap();
        PreparedBasis::new(&mol, &bs).unwrap()
    }

    #[test]
    fn test_overlap_diagonal_ones() {
        let prep = water_sto3g();
        let s = overlap(&prep);
        for i in 0..prep.nbasis() {
            assert!((s[(i, i)] - 1.0).abs() < 1e-8, "S[{i},{i}] = {}", s[(i, i)]);
        }
    }

    #[test]
    fn test_overlap_symmetric() {
        let prep = water_sto3g();
        let s = overlap(&prep);
        let n = prep.nbasis();
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (s[(i, j)] - s[(j, i)]).abs() < 1e-12,
                    "S not symmetric at ({i},{j})"
                );
            }
        }
    }
}
