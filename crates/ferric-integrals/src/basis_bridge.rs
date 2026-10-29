//! Bridge between Rust basis/molecule types and the libint2 C++ engine.
//!
//! [`crate::basis_bridge::PreparedBasis`] translates a [`ferric_core::mol::Molecule`] + [`ferric_core::basis::BasisSet`] pair into the
//! internal libint2 representation, managing the C++ handle lifetime.

use crate::ao_grid::LocatedShell;
use crate::ffi::{self, CAtom, CShell};
use ferric_core::basis::BasisSet;
use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use std::os::raw::{c_int, c_void};
use std::sync::Once;

static LIBINT_INIT: Once = Once::new();

pub(crate) fn ensure_init() {
    // SAFETY: Called exactly once via std::sync::Once. scf_libint_init
    // initializes libint2's global state (no arguments, no preconditions).
    LIBINT_INIT.call_once(|| unsafe { ffi::scf_libint_init() });
}

/// A molecule + basis set prepared for integral evaluation.
///
/// Owns the opaque libint2 `BasisSet` handle and precomputed shell metadata
/// (dimensions, offsets, atom mapping). Thread-safe (`Send + Sync`).
#[must_use = "PreparedBasis is expensive to construct; dropping it wastes work"]
pub struct PreparedBasis {
    handle: *mut c_void,
    basis_set: BasisSet,
    atoms: Vec<CAtom>,
    shell_dims: Vec<usize>,
    shell_offsets: Vec<usize>,
    shell_to_atom: Vec<usize>,
    /// Per-shell angular momentum, pure flag and primitive data, in shell
    /// order (the same data handed to libint2; kept so pure-Rust kernels such
    /// as `md3c1e` can be built from a `PreparedBasis` alone).
    shell_l: Vec<i32>,
    shell_pure: Vec<bool>,
    shell_exps: Vec<Vec<f64>>,
    shell_coefs: Vec<Vec<f64>>,
    nbasis: usize,
    nshells: usize,
    max_nprim: i32,
    max_l: i32,
}

// SAFETY: PreparedBasis owns its libint2 handle exclusively and all fields
// are immutable after construction. The C++ BasisSet is a read-only data
// structure (shell metadata, no mutable scratch), so concurrent &-access
// from multiple threads is sound. The handle is destroyed in Drop
// (single-owner, no aliasing).
unsafe impl Send for PreparedBasis {}
unsafe impl Sync for PreparedBasis {}

/// Refuse, at setup, a molecule whose V_ECP the ECP integral code cannot
/// build: when any real atom carries an ECP, V_ECP couples EVERY shell of the
/// molecule, and both ECP backends stop at l = [`crate::ecp_quad::MAX_SHELL_L`]
/// (4, g). Without this the first hcore build panics mid-SCF. Reached by
/// e.g. Pd or Ag in cc-pVQZ-PP, whose h (l = 5) shells exceed the limit.
///
/// V_ECP is also built in the SPHERICAL AO basis only (`ecp_potential` sizes
/// it as 2l+1 per shell), so a Cartesian l >= 2 shell anywhere in an ECP
/// molecule (e.g. a user JSON typed `gto_cartesian`) would mis-size it and
/// panic; that is refused here too.
fn check_ecp_shell_l(mol: &Molecule, bs: &BasisSet) -> Result<(), FerricError> {
    let Some(ecp_atom) = mol
        .atoms
        .iter()
        .find(|a| !a.ghost && bs.ecp_for_element(a.z).is_some())
    else {
        return Ok(());
    };
    for atom in &mol.atoms {
        let shells = bs.for_element(atom.z).unwrap_or_default();
        if let Some(why) = ecp_incompatible_shell(shells) {
            return Err(FerricError::Basis(format!(
                "basis {:?} puts {why} on {} but {} carries an ECP, and ECP integrals \
                 support only spherical shells up to l={}; use a smaller basis (e.g. \
                 the TZ member of the family) or a spherical (gto_spherical) one",
                bs.name,
                atom.symbol,
                ecp_atom.symbol,
                crate::ecp_quad::MAX_SHELL_L
            )));
        }
    }
    Ok(())
}

/// Why `shells` cannot share a molecule with an ECP, if they cannot.
fn ecp_incompatible_shell(shells: &[ferric_core::basis::Shell]) -> Option<String> {
    let max_l = crate::ecp_quad::MAX_SHELL_L as i32;
    if let Some(sh) = shells.iter().find(|s| s.l > max_l) {
        return Some(format!("an l={} shell (above l={max_l})", sh.l));
    }
    shells
        .iter()
        .find(|s| s.l >= 2 && !s.pure)
        .map(|sh| format!("a Cartesian l={} shell", sh.l))
}

impl PreparedBasis {
    /// Create a prepared basis from a molecule and basis set.
    ///
    /// Initializes libint2 on first call and constructs the internal C++ basis handle.
    pub fn new(mol: &Molecule, bs: &BasisSet) -> Result<Self, FerricError> {
        ensure_init();
        check_ecp_shell_l(mol, bs)?;
        // Effective nuclear charge for the `nuclear` 1e operator and point-charge
        // list: 0 for a ghost (basis-only) center so libint2 excludes it from the
        // nuclear-attraction operator; otherwise Z − n_core for ECP atoms, plain Z
        // for all-electron atoms. The ECP n_core is derived from the basis set's
        // ECP table so it is correct even if the caller did not call
        // `Molecule::apply_ecp`. The basis-set SHELL lookup below still uses the
        // REAL `atom.z`, so ghost/ECP atoms keep their full basis.
        let c_atoms: Vec<CAtom> = mol
            .atoms
            .iter()
            .map(|a| {
                let z_eff = if a.ghost {
                    0
                } else {
                    match bs.ecp_for_element(a.z) {
                        Some(def) => a.z - def.n_core,
                        None => a.z,
                    }
                };
                CAtom {
                    atomic_number: z_eff as f64,
                    x: a.x,
                    y: a.y,
                    z: a.zpos,
                }
            })
            .collect();

        let mut c_shells = Vec::new();
        let mut shell_to_atom = Vec::new();
        let mut shell_l = Vec::new();
        let mut shell_pure = Vec::new();
        let mut keep_exps = Vec::new();
        let mut keep_coefs = Vec::new();
        for (ai, atom) in mol.atoms.iter().enumerate() {
            let tmpls = bs
                .for_element(atom.z)
                .ok_or_else(|| bs.missing_element_error(atom.z))?;
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
                shell_l.push(sh.l);
                shell_pure.push(sh.pure);
                keep_exps.push(exps);
                keep_coefs.push(coefs);
            }
        }

        // SAFETY: c_shells and c_atoms are valid slices with matching lengths.
        // keep_exps/keep_coefs keep the pointed-to exponent/coefficient arrays
        // alive until scf_basis_create returns (the C++ side copies the data);
        // they are then retained as `shell_exps`/`shell_coefs` for Rust-side
        // consumers. Null-checked below.
        let handle = unsafe {
            ffi::scf_basis_create(
                c_shells.as_ptr(),
                c_shells.len() as c_int,
                c_atoms.as_ptr(),
                c_atoms.len() as c_int,
            )
        };

        if handle.is_null() {
            return Err(FerricError::Libint("scf_basis_create returned null".into()));
        }
        // SAFETY: handle is a valid (non-null, just-created) basis handle.
        // These accessors read immutable metadata from the C++ BasisSet object.
        let nbasis = unsafe { ffi::scf_basis_nbasis(handle) } as usize;
        let nshells = unsafe { ffi::scf_basis_nshells(handle) } as usize;
        if nshells != c_shells.len() {
            // SAFETY: handle is non-null and was created just above; it is not
            // used again after this point.
            unsafe { ffi::scf_basis_destroy(handle) };
            return Err(FerricError::Libint(format!(
                "scf_basis_create: libint2 reports {nshells} shells, {} were passed",
                c_shells.len()
            )));
        }
        let mut dims_raw = vec![0i32; nshells];
        // SAFETY: `dims_raw` has exactly `nshells` elements, matching the
        // number of shells in the C++ basis. The shim writes exactly that many.
        unsafe { ffi::scf_basis_shell_dims(handle, dims_raw.as_mut_ptr()) };
        let shell_dims: Vec<usize> = dims_raw.iter().map(|&d| d as usize).collect();
        let mut shell_offsets = vec![0usize; nshells + 1];
        for i in 0..nshells {
            shell_offsets[i + 1] = shell_offsets[i] + shell_dims[i];
        }
        let mut mp: c_int = 0;
        let mut ml: c_int = 0;
        // SAFETY: Valid handle; writes two c_int values into stack locals.
        unsafe { ffi::scf_basis_max_dims(handle, &mut mp, &mut ml) };
        Ok(PreparedBasis {
            handle,
            basis_set: bs.clone(),
            atoms: c_atoms,
            shell_dims,
            shell_offsets,
            shell_to_atom,
            shell_l,
            shell_pure,
            shell_exps: keep_exps,
            shell_coefs: keep_coefs,
            nbasis,
            nshells,
            max_nprim: mp,
            max_l: ml,
        })
    }

    /// Get the underlying basis set used to build this prepared basis.
    pub fn basis_set(&self) -> &BasisSet {
        &self.basis_set
    }
    /// Raw pointer to the C++ basis handle (for FFI calls).
    pub fn handle(&self) -> *const c_void {
        self.handle
    }
    /// Atom array passed to libint2.
    pub fn atoms(&self) -> &[CAtom] {
        &self.atoms
    }
    /// Total number of basis functions.
    pub fn nbasis(&self) -> usize {
        self.nbasis
    }
    /// Number of shells.
    pub fn nshells(&self) -> usize {
        self.nshells
    }
    /// Number of basis functions per shell.
    pub fn shell_dims(&self) -> &[usize] {
        &self.shell_dims
    }
    /// Cumulative offset of each shell into the full basis (length nshells+1).
    pub fn shell_offsets(&self) -> &[usize] {
        &self.shell_offsets
    }
    /// Atom index that each shell belongs to.
    pub fn shell_to_atom(&self) -> &[usize] {
        &self.shell_to_atom
    }
    /// Number of atoms.
    pub fn natoms(&self) -> usize {
        self.atoms.len()
    }
    /// Maximum number of primitives across all shells.
    pub fn max_nprim(&self) -> i32 {
        self.max_nprim
    }
    /// Maximum angular momentum across all shells.
    pub fn max_l(&self) -> i32 {
        self.max_l
    }
    /// Cartesian center of each shell (from its atom).
    pub fn shell_centers(&self) -> Vec<[f64; 3]> {
        self.shell_to_atom
            .iter()
            .map(|&ai| {
                let a = &self.atoms[ai];
                [a.x, a.y, a.z]
            })
            .collect()
    }
    /// The shells of this basis, in `PreparedBasis` shell order, as located
    /// shells (angular momentum, pure flag, primitive exponents and the
    /// contraction coefficients exactly as stored in the `BasisSet`, i.e.
    /// WITHOUT the per-primitive normalization — see `ao_grid`).
    ///
    /// This is the same shell list that was handed to libint2, so a Rust-side
    /// kernel built from it produces blocks in the same AO order/offsets as
    /// `shell_offsets()`. Centres are the parent atom's coordinates (Bohr).
    pub fn located_shells(&self) -> Vec<LocatedShell<'_>> {
        (0..self.nshells)
            .map(|s| {
                let a = &self.atoms[self.shell_to_atom[s]];
                LocatedShell {
                    l: self.shell_l[s],
                    pure: self.shell_pure[s],
                    exponents: &self.shell_exps[s],
                    coefficients: &self.shell_coefs[s],
                    center: [a.x, a.y, a.z],
                }
            })
            .collect()
    }
}

impl std::fmt::Debug for PreparedBasis {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparedBasis")
            .field("nbasis", &self.nbasis)
            .field("nshells", &self.nshells)
            .field("basis_set", &self.basis_set.name)
            .finish_non_exhaustive()
    }
}

impl Drop for PreparedBasis {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            // SAFETY: handle is non-null and was created by scf_basis_create;
            // Drop runs exactly once per PreparedBasis.
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

    /// A user BSE JSON for HI: aug-cc-pVDZ-PP's H and I (I with its ECP),
    /// with every l >= 2 shell retyped `function_type`.
    fn hi_user_basis(function_type: &str) -> basis::BasisSet {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../ferric-core/src/basis/bundled/aug-cc-pvdz-pp.json"
        );
        let mut doc: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        let els = doc["elements"].as_object_mut().unwrap();
        els.retain(|z, _| z == "1" || z == "53");
        for el in els.values_mut() {
            for sh in el["electron_shells"].as_array_mut().unwrap() {
                let max_l = sh["angular_momentum"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .filter_map(serde_json::Value::as_i64)
                    .max()
                    .unwrap();
                if max_l >= 2 {
                    sh["function_type"] = function_type.into();
                }
            }
        }
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("hi-user.json");
        std::fs::write(&file, doc.to_string()).unwrap();
        basis::load_bse_json(file.to_str().unwrap()).unwrap()
    }

    /// V_ECP is spherical-only; a Cartesian d shell on an ECP molecule used to
    /// panic in `ecp_potential`'s `.expect`. It is now a setup error, and the
    /// same basis typed spherical is accepted (so the refusal is the d type).
    #[test]
    fn ecp_with_cartesian_d_shell_is_a_clean_setup_error() {
        let xyz = "2\nHI\nH 0 0 0\nI 0 0 1.609\n";
        let cart = hi_user_basis("gto_cartesian");
        let mut mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
        mol.apply_ecp(&cart);
        let msg = match PreparedBasis::new(&mol, &cart) {
            Ok(_) => panic!("Cartesian d with an ECP must be refused"),
            Err(e) => e.to_string(),
        };
        assert!(
            msg.contains("Cartesian l=2 shell on I but I carries an ECP"),
            "{msg}"
        );
        let sph = hi_user_basis("gto_spherical");
        let mut mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
        mol.apply_ecp(&sph);
        let prep = PreparedBasis::new(&mol, &sph).unwrap();
        let v = crate::oneelectron::ecp_potential(&mol, &sph).unwrap();
        assert_eq!(v.nrows(), prep.nbasis());
    }
}
