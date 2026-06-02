//! Bridge between Rust basis/molecule types and the libint2 C++ engine.
//!
//! [`PreparedBasis`] translates a [`Molecule`] + [`BasisSet`] pair into the
//! internal libint2 representation, managing the C++ handle lifetime.

use crate::ffi::{self, CAtom, CShell};
use ferric_core::basis::BasisSet;
use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use std::os::raw::{c_int, c_void};
use std::sync::Once;

static LIBINT_INIT: Once = Once::new();

fn ensure_init() {
    LIBINT_INIT.call_once(|| unsafe { ffi::scf_libint_init() });
}

/// A molecule + basis set prepared for integral evaluation.
///
/// Owns the opaque libint2 `BasisSet` handle and precomputed shell metadata
/// (dimensions, offsets, atom mapping). Thread-safe (`Send + Sync`).
pub struct PreparedBasis {
    handle: *mut c_void,
    basis_set: BasisSet,
    atoms: Vec<CAtom>,
    shell_dims: Vec<usize>,
    shell_offsets: Vec<usize>,
    shell_to_atom: Vec<usize>,
    nbasis: usize,
    nshells: usize,
    max_nprim: i32,
    max_l: i32,
}

unsafe impl Send for PreparedBasis {}
unsafe impl Sync for PreparedBasis {}

impl PreparedBasis {
    /// Create a prepared basis from a molecule and basis set.
    ///
    /// Initializes libint2 on first call and constructs the internal C++ basis handle.
    pub fn new(mol: &Molecule, bs: &BasisSet) -> Result<Self, FerricError> {
        ensure_init();
        let c_atoms: Vec<CAtom> = mol.atoms.iter().map(|a| CAtom {
            atomic_number: a.z as c_int, x: a.x, y: a.y, z: a.zpos,
        }).collect();

        let mut c_shells = Vec::new();
        let mut shell_to_atom = Vec::new();
        let mut keep_exps = Vec::new();
        let mut keep_coefs = Vec::new();
        for (ai, atom) in mol.atoms.iter().enumerate() {
            let tmpls = bs.for_element(atom.z).ok_or_else(|| {
                FerricError::Basis(format!(
                    "no basis shells for Z={} ({}) in {:?}",
                    atom.z, atom.symbol, bs.name
                ))
            })?;
            for sh in tmpls {
                let exps: Vec<f64> = sh.exponents.clone();
                let coefs: Vec<f64> = sh.coefficients.clone();
                c_shells.push(CShell {
                    l: sh.l as c_int,
                    nprim: exps.len() as c_int,
                    atom_index: ai as c_int,
                    pure: if sh.pure { 1 } else { 0 },
                    exponents: exps.as_ptr(),
                    coefficients: coefs.as_ptr(),
                });
                shell_to_atom.push(ai);
                keep_exps.push(exps);
                keep_coefs.push(coefs);
            }
        }

        let handle = unsafe {
            ffi::scf_basis_create(
                c_shells.as_ptr(),
                c_shells.len() as c_int,
                c_atoms.as_ptr(),
                c_atoms.len() as c_int,
            )
        };
        // keep_exps and keep_coefs must stay alive until after scf_basis_create returns
        // (the C++ side copies the data)
        drop(keep_exps);
        drop(keep_coefs);

        if handle.is_null() {
            return Err(FerricError::Libint("scf_basis_create returned null".into()));
        }
        let nbasis = unsafe { ffi::scf_basis_nbasis(handle) } as usize;
        let nshells = unsafe { ffi::scf_basis_nshells(handle) } as usize;
        let mut dims_raw = vec![0i32; nshells];
        unsafe { ffi::scf_basis_shell_dims(handle, dims_raw.as_mut_ptr()) };
        let shell_dims: Vec<usize> = dims_raw.iter().map(|&d| d as usize).collect();
        let mut shell_offsets = vec![0usize; nshells + 1];
        for i in 0..nshells {
            shell_offsets[i + 1] = shell_offsets[i] + shell_dims[i];
        }
        let mut mp: c_int = 0;
        let mut ml: c_int = 0;
        unsafe { ffi::scf_basis_max_dims(handle, &mut mp, &mut ml) };
        Ok(PreparedBasis {
            handle,
            basis_set: bs.clone(),
            atoms: c_atoms,
            shell_dims,
            shell_offsets,
            shell_to_atom,
            nbasis,
            nshells,
            max_nprim: mp,
            max_l: ml,
        })
    }

    /// Get the underlying basis set used to build this prepared basis.
    pub fn basis_set(&self) -> &BasisSet { &self.basis_set }
    /// Raw pointer to the C++ basis handle (for FFI calls).
    pub fn handle(&self) -> *const c_void { self.handle }
    /// Atom array passed to libint2.
    pub fn atoms(&self) -> &[CAtom] { &self.atoms }
    /// Total number of basis functions.
    pub fn nbasis(&self) -> usize { self.nbasis }
    /// Number of shells.
    pub fn nshells(&self) -> usize { self.nshells }
    /// Number of basis functions per shell.
    pub fn shell_dims(&self) -> &[usize] { &self.shell_dims }
    /// Cumulative offset of each shell into the full basis (length nshells+1).
    pub fn shell_offsets(&self) -> &[usize] { &self.shell_offsets }
    /// Atom index that each shell belongs to.
    pub fn shell_to_atom(&self) -> &[usize] { &self.shell_to_atom }
    /// Number of atoms.
    pub fn natoms(&self) -> usize { self.atoms.len() }
    /// Maximum number of primitives across all shells.
    pub fn max_nprim(&self) -> i32 { self.max_nprim }
    /// Maximum angular momentum across all shells.
    pub fn max_l(&self) -> i32 { self.max_l }
    /// Cartesian center of each shell (from its atom).
    pub fn shell_centers(&self) -> Vec<[f64; 3]> {
        self.shell_to_atom.iter().map(|&ai| {
            let a = &self.atoms[ai];
            [a.x, a.y, a.z]
        }).collect()
    }
}

impl Drop for PreparedBasis {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { ffi::scf_basis_destroy(self.handle) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferric_core::basis;
    use ferric_core::mol::Molecule;

    #[test]
    fn test_prepared_basis_h2o_sto3g() {
        let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
        let bs = basis::bundled("sto-3g").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        assert_eq!(prep.nbasis(), 7);
        assert_eq!(prep.nshells(), 5);
    }

    #[test]
    fn test_prepared_basis_h2o_631g() {
        let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
        let bs = basis::bundled("6-31g").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        assert_eq!(prep.nbasis(), 13);
    }
}
