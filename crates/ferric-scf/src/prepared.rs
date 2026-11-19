//! [`PreparedSystem`](crate::prepared::PreparedSystem): the molecule, its
//! prepared orbital basis, the Coulomb operator, Schwarz bounds and the
//! parallel context, built once by a single constructor and only borrowed
//! afterwards.
//!
//! The constructor is where the basis set's ECP enters the molecule
//! ([`ferric_core::mol::Molecule::apply_ecp`]), so a solver handed a
//! `&PreparedSystem` cannot receive a molecule without its ECP core counts
//! while its basis carries the ECP potential.

use ferric_core::basis::BasisSet;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;

use crate::ladder::{LadderResult, Rung};
use crate::result::ScfResult;
use crate::rhf::RhfConfig;
use crate::screening::SchwarzBounds;

/// A molecule with its ECP applied, the prepared orbital basis, the Coulomb
/// operator, its Schwarz bounds and a parallel context.
///
/// Built by [`PreparedSystem::new`] only, and borrowed afterwards: fields are
/// private and read through `&self` accessors, so the molecule, the basis it
/// was prepared on and the bounds of that basis cannot drift apart.
pub struct PreparedSystem {
    mol: Molecule,
    basis: PreparedBasis,
    op: Operator,
    bounds: SchwarzBounds,
    ctx: ParallelContext,
}

impl PreparedSystem {
    /// Apply the basis set's ECP core counts to `mol`
    /// ([`Molecule::apply_ecp`], a no-op without an ECP), prepare `basis` on
    /// it, and compute the Coulomb Schwarz bounds of that basis with a default
    /// [`ParallelContext`].
    ///
    /// The ECP is applied unconditionally: [`PreparedBasis::new`] always
    /// reduces each ECP atom's nuclear charge to `Z - n_core` and the core
    /// Hamiltonian adds `V_ECP`, so the molecule's `n_core_ecp` (and with it
    /// `nelec()` and `nuclear_repulsion()`) must agree.
    ///
    /// Errors are those of [`PreparedBasis::new`] (e.g. an element the basis
    /// does not cover, or a shell the ECP integrals cannot handle) and of
    /// [`SchwarzBounds::compute`].
    pub fn new(mut mol: Molecule, basis: &BasisSet) -> Result<Self, FerricError> {
        mol.apply_ecp(basis);
        let basis = PreparedBasis::new(&mol, basis)?;
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &basis)?;
        Ok(Self {
            mol,
            basis,
            op,
            bounds,
            ctx: ParallelContext::default(),
        })
    }

    /// An auxiliary (fitting) basis prepared on the same molecule.
    pub fn with_aux(&self, aux: &BasisSet) -> Result<PreparedBasis, FerricError> {
        PreparedBasis::new(&self.mol, aux)
    }

    /// The molecule, with the basis set's ECP core counts applied.
    pub fn mol(&self) -> &Molecule {
        &self.mol
    }

    /// The orbital basis prepared on [`Self::mol`].
    pub fn basis(&self) -> &PreparedBasis {
        &self.basis
    }

    /// The two-electron operator of the reference SCF (Coulomb).
    pub fn op(&self) -> Operator {
        self.op
    }

    /// Schwarz bounds of [`Self::basis`] for [`Self::op`].
    pub fn bounds(&self) -> &SchwarzBounds {
        &self.bounds
    }

    /// The parallel execution context.
    pub fn ctx(&self) -> &ParallelContext {
        &self.ctx
    }

    /// [`solve_rhf`](crate::rhf::solve_rhf) on this system.
    pub fn solve_rhf(&self, config: &RhfConfig) -> Result<ScfResult, FerricError> {
        crate::rhf::solve_rhf(
            &self.ctx,
            &self.mol,
            &self.basis,
            self.op,
            &self.bounds,
            config,
        )
    }

    /// [`solve_uhf`](crate::uhf::solve_uhf) on this system.
    pub fn solve_uhf(&self, config: &RhfConfig) -> Result<ScfResult, FerricError> {
        crate::uhf::solve_uhf(&self.ctx, &self.mol, &self.basis, &self.bounds, config)
    }

    /// [`solve_rohf`](crate::rohf::solve_rohf) on this system.
    pub fn solve_rohf(&self, config: &RhfConfig) -> Result<ScfResult, FerricError> {
        crate::rohf::solve_rohf(
            &self.ctx,
            &self.mol,
            &self.basis,
            self.op,
            &self.bounds,
            config,
        )
    }

    /// [`solve_rhf_ladder`](crate::ladder::solve_rhf_ladder) on this system.
    pub fn solve_rhf_ladder(&self, ladder: &[Rung]) -> Result<LadderResult, FerricError> {
        crate::ladder::solve_rhf_ladder(
            &self.ctx,
            &self.mol,
            &self.basis,
            self.op,
            &self.bounds,
            ladder,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferric_core::basis;

    fn hi() -> Molecule {
        Molecule::parse_xyz("2\n\nH 0 0 0\nI 0 0 1.61\n", 0, 1).unwrap()
    }

    #[test]
    fn the_constructor_applies_the_ecp_core_counts() {
        let bs = basis::bundled("def2-svp").unwrap();
        let sys = PreparedSystem::new(hi(), &bs).unwrap();
        assert_eq!(hi().nelec(), 54);
        assert_eq!(sys.mol().nelec(), 26);
        assert_eq!(sys.mol().atoms[1].n_core_ecp, 28);
        let mut applied = hi();
        applied.apply_ecp(&bs);
        assert_eq!(sys.mol().nuclear_repulsion(), applied.nuclear_repulsion());
    }

    #[test]
    fn the_system_matches_a_hand_built_basis_bounds_and_aux() {
        let mol = Molecule::parse_xyz(
            "3\n\nO 0 0 0.117\nH 0 0.757 -0.467\nH 0 -0.757 -0.467\n",
            0,
            1,
        )
        .unwrap();
        let bs = basis::bundled("sto-3g").unwrap();
        let aux_bs = basis::bundled("cc-pvdz-ri").unwrap();
        let sys = PreparedSystem::new(mol.clone(), &bs).unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        assert_eq!(sys.mol().nelec(), mol.nelec());
        assert_eq!(sys.basis().nbasis(), prep.nbasis());
        assert_eq!(sys.op(), Operator::coulomb());
        let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
        assert_eq!(sys.bounds().q, bounds.q);
        let aux = sys.with_aux(&aux_bs).unwrap();
        assert_eq!(
            aux.nbasis(),
            PreparedBasis::new(&mol, &aux_bs).unwrap().nbasis()
        );
    }
}
