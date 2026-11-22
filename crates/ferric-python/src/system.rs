//! The prepared system of a molecular binding.
//!
//! The SCF entry points (`run_rhf`, `run_uhf`, `run_rohf`, `run_dft`,
//! `run_uks`, `run_roks`) and the correlated `run_*` drivers build one
//! `ferric_scf::PreparedSystem` (molecule with the basis set's ECP applied,
//! prepared orbital basis, Coulomb Schwarz bounds, parallel context) by its
//! single constructor, through [`prepare`] (or, where the ECP molecule is
//! checked before any integral, `EcpMolecule` directly), and prepare any
//! fitting basis on the same molecule with [`prepare_aux`]. The correlated
//! drivers solve their reference on the borrowed system (`solve_rhf` /
//! `solve_uhf` / `solve_reference`) and refuse a non-converged SCF
//! (`ScfResult::into_converged`). Every later use of the molecule reads
//! `sys.mol()`, so the kernels see the valence electron count the reference
//! was solved for. (Kernels that cannot handle ECP atoms refuse them
//! themselves: the amplitude-threshold local MP2, dRPA and LinLCCD in
//! `ferric_mp2::lmp2_amplitude`.)
//!
//! [`prepare`], [`prepare_ecp`] and [`prepare_aux`] build the basis, the
//! bounds and the fitting basis with the GIL released (they copy what they
//! need from the Python objects first); a binding that already runs inside
//! `allow_threads` builds its system there directly.

use super::*;

/// A copy of `mol` with `basis_set`'s ECP applied (`EcpMolecule::new`, the
/// one place the ECP enters a molecule), for a binding that needs the
/// valence molecule (electron count, effective nuclear charges) but prepares
/// its own bases, or none, and needs no Schwarz bounds: the integral and
/// property helpers, `AoIntegrals`, `response_modes`.
pub(crate) fn ecp_molecule(mol: &PyMolecule, basis_set: &PyBasisSet) -> Molecule {
    EcpMolecule::new(mol.inner.clone(), &basis_set.inner).into_mol()
}

/// `f` with the GIL released, from a binding that holds it (the integral
/// work of preparing a system or a fitting basis touches no Python object).
fn without_gil<T: Send>(f: impl FnOnce() -> T + Send) -> T {
    Python::with_gil(|py| py.allow_threads(f))
}

/// `PreparedSystem::new` on a copy of `mol`, for a binding returning
/// `PyResult`.
pub(crate) fn prepare(mol: &PyMolecule, basis: &PyBasisSet) -> PyResult<PreparedSystem> {
    let mol = mol.inner.clone();
    without_gil(|| PreparedSystem::new(mol, &basis.inner)).map_err(make_err)
}

/// `EcpMolecule::prepare` with the default options, for a binding that
/// checked the ECP molecule first.
pub(crate) fn prepare_ecp(ecp: EcpMolecule<'_>) -> PyResult<PreparedSystem> {
    without_gil(|| ecp.prepare(PrepareOptions::default())).map_err(make_err)
}

/// `PreparedSystem::prepare_aux` of the fitting basis `aux` on `sys`'s
/// molecule, for a binding returning `PyResult`.
pub(crate) fn prepare_aux<'s>(
    sys: &'s PreparedSystem,
    aux: &PyBasisSet,
) -> PyResult<PreparedAux<'s>> {
    without_gil(|| sys.prepare_aux(&aux.inner)).map_err(make_err)
}
