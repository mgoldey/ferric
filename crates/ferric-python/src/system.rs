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

use super::*;

/// `PreparedSystem::new` on a copy of `mol`, for a binding returning
/// `PyResult`.
pub(crate) fn prepare(mol: &PyMolecule, basis: &PyBasisSet) -> PyResult<PreparedSystem> {
    PreparedSystem::new(mol.inner.clone(), &basis.inner).map_err(make_err)
}

/// `PreparedSystem::prepare_aux` of the fitting basis `aux` on `sys`'s
/// molecule, for a binding returning `PyResult`.
pub(crate) fn prepare_aux<'s>(
    sys: &'s PreparedSystem,
    aux: &PyBasisSet,
) -> PyResult<PreparedAux<'s>> {
    sys.prepare_aux(&aux.inner).map_err(make_err)
}
