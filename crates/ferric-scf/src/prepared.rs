//! [`PreparedSystem`](crate::prepared::PreparedSystem): the molecule, its
//! prepared orbital basis, the Coulomb operator, Schwarz bounds and the
//! parallel context, built once by a single constructor and only borrowed
//! afterwards.
//!
//! Construction has two stages. [`EcpMolecule::new`] is where the basis set's
//! ECP enters the molecule ([`ferric_core::mol::Molecule::apply_ecp`]);
//! [`EcpMolecule::prepare`] then prepares the basis and its Schwarz bounds
//! with [`PrepareOptions`]. A frontend that validates the ECP molecule before
//! any integral work (its electron count, a frozen core) does so between the
//! two; [`PreparedSystem::new`] runs both with the default options. So a
//! solver handed a `&PreparedSystem` cannot receive a molecule without its ECP
//! core counts while its basis carries the ECP potential.
//!
//! A fitting basis is prepared on the system's molecule by
//! [`PreparedSystem::prepare_aux`], which returns a [`PreparedAux`] that
//! borrows the system it was prepared for and names it
//! ([`PreparedAux::system`]).
//!
//! The system owns its screening: the bounds were built for one
//! [`ScreeningKind`] ([`PrepareOptions::screening`]), and the solve methods
//! run every config with that kind ([`PreparedSystem::screening`]), whatever
//! its `RhfConfig::screening` says, so direct J/K (which reads the bounds'
//! tables) and LinK (which reads the config) cannot screen with two kinds.

use std::borrow::Cow;
use std::ops::Deref;

use ferric_core::basis::BasisSet;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;

use crate::ladder::{LadderResult, Rung};
use crate::result::ScfResult;
use crate::rhf::RhfConfig;
use crate::screening::{SchwarzBounds, ScreeningKind};

/// How [`EcpMolecule::prepare`] builds a [`PreparedSystem`]. The `Default` is
/// plain Cauchy-Schwarz bounds and a default [`ParallelContext`], what
/// [`PreparedSystem::new`] uses.
#[derive(Default)]
pub struct PrepareOptions {
    /// The shell-quartet screening of the bounds
    /// ([`SchwarzBounds::compute_for_screening`]): `Schwarz` leaves both
    /// refinement tables unset, `Csb` / `Csam` attach theirs. A solver whose
    /// `RhfConfig::screening` should agree reads it back through
    /// [`PreparedSystem::screening`], which derives it from the bounds; the
    /// system's own solve methods apply it to every config they run.
    pub screening: ScreeningKind,
    /// The parallel execution context the system's solvers run in.
    pub ctx: ParallelContext,
}

/// A molecule with a basis set's ECP core counts applied
/// ([`Molecule::apply_ecp`]), not yet prepared: the first stage of a
/// [`PreparedSystem`].
///
/// The molecule is readable ([`Self::mol`]) so a caller can check what
/// depends on the valence electron count (spin, frozen core, active space)
/// before the basis is prepared, and the only way forward is
/// [`Self::prepare`] with the same basis set.
pub struct EcpMolecule<'b> {
    mol: Molecule,
    basis: &'b BasisSet,
}

impl<'b> EcpMolecule<'b> {
    /// Set `mol`'s ECP core counts to what `basis` implies
    /// ([`Molecule::apply_ecp`]: the basis set's `n_core` on each ECP atom, 0
    /// elsewhere, whatever counts `mol` arrived with).
    ///
    /// The counts are set unconditionally: [`PreparedBasis::new`] derives
    /// each atom's nuclear charge (`Z - n_core` on an ECP atom) and `V_ECP`
    /// from the basis set alone, so the molecule's `n_core_ecp` (and with it
    /// `nelec()` and `nuclear_repulsion()`) must agree with the same basis.
    pub fn new(mut mol: Molecule, basis: &'b BasisSet) -> Self {
        mol.apply_ecp(basis);
        Self { mol, basis }
    }

    /// The molecule, with the basis set's ECP core counts applied.
    pub fn mol(&self) -> &Molecule {
        &self.mol
    }

    /// The molecule with the ECP core counts applied, consuming the stage:
    /// for a caller that needs the valence molecule (its electron count,
    /// nuclear repulsion, effective charges) but no prepared basis or bounds,
    /// such as one that prepares a basis per geometry itself.
    pub fn into_mol(self) -> Molecule {
        self.mol
    }

    /// The basis set whose ECP was applied and that [`Self::prepare`]
    /// prepares.
    pub fn basis_set(&self) -> &'b BasisSet {
        self.basis
    }

    /// Prepare the basis set on [`Self::mol`] and compute its Coulomb
    /// Schwarz bounds with `options.screening`
    /// ([`SchwarzBounds::compute_for_screening`]; `Schwarz` is
    /// byte-identical to [`SchwarzBounds::compute`]).
    ///
    /// Errors are those of [`PreparedBasis::new`] (e.g. an element the basis
    /// does not cover, or a shell the ECP integrals cannot handle), then
    /// those of the bounds (e.g. `Csam` refuses a short-range operator).
    pub fn prepare(self, options: PrepareOptions) -> Result<PreparedSystem, FerricError> {
        let PrepareOptions { screening, ctx } = options;
        let basis = PreparedBasis::new(&self.mol, self.basis)?;
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute_for_screening(op, &basis, screening)?;
        Ok(PreparedSystem {
            mol: self.mol,
            basis,
            bounds,
            ctx,
        })
    }
}

/// A molecule with its ECP applied, the prepared orbital basis, the Coulomb
/// Schwarz bounds of that basis and a parallel context.
///
/// Built by [`EcpMolecule::prepare`] only ([`Self::new`] is that with the
/// default options), and borrowed afterwards: fields are private and read
/// through `&self` accessors, so the molecule, the basis it was prepared on
/// and the bounds of that basis cannot drift apart.
pub struct PreparedSystem {
    mol: Molecule,
    basis: PreparedBasis,
    bounds: SchwarzBounds,
    ctx: ParallelContext,
}

impl PreparedSystem {
    /// [`EcpMolecule::new`] then [`EcpMolecule::prepare`] with the default
    /// [`PrepareOptions`]: plain Schwarz bounds and a default
    /// [`ParallelContext`].
    pub fn new(mol: Molecule, basis: &BasisSet) -> Result<Self, FerricError> {
        EcpMolecule::new(mol, basis).prepare(PrepareOptions::default())
    }

    /// An auxiliary (fitting) basis prepared on the same molecule. The
    /// returned [`PreparedAux`] borrows this system, so it cannot outlive
    /// it, and names it ([`PreparedAux::system`]).
    pub fn prepare_aux(&self, aux: &BasisSet) -> Result<PreparedAux<'_>, FerricError> {
        Ok(PreparedAux {
            system: self,
            basis: PreparedBasis::new(&self.mol, aux)?,
        })
    }

    /// The molecule, with the basis set's ECP core counts applied.
    pub fn mol(&self) -> &Molecule {
        &self.mol
    }

    /// The orbital basis prepared on [`Self::mol`].
    pub fn basis(&self) -> &PreparedBasis {
        &self.basis
    }

    /// The two-electron operator of the reference SCF and of the bounds:
    /// always Coulomb.
    pub fn op(&self) -> Operator {
        Operator::coulomb()
    }

    /// Schwarz bounds of [`Self::basis`] for [`Self::op`].
    pub fn bounds(&self) -> &SchwarzBounds {
        &self.bounds
    }

    /// The screening the bounds were built with
    /// ([`PrepareOptions::screening`]), read from the bounds themselves:
    /// [`SchwarzBounds::compute_for_screening`] attaches the CSB table for
    /// `Csb`, the CSAM table for `Csam` and neither for `Schwarz`.
    pub fn screening(&self) -> ScreeningKind {
        match (&self.bounds.csb_m, &self.bounds.csam_x) {
            (Some(_), _) => ScreeningKind::Csb,
            (None, Some(_)) => ScreeningKind::Csam,
            (None, None) => ScreeningKind::Schwarz,
        }
    }

    /// The molecule (with its ECP core counts applied), consuming the
    /// system: for a result that keeps the molecule after the solve.
    pub fn into_mol(self) -> Molecule {
        self.mol
    }

    /// The parallel execution context.
    pub fn ctx(&self) -> &ParallelContext {
        &self.ctx
    }

    /// `config` with [`Self::screening`] in place of its own
    /// `RhfConfig::screening`: borrowed when the two agree (every caller
    /// today), else a copy. A copy shares the original's internal records
    /// (`eri_route`, `raw_handoff` are shared handles), so what the solver
    /// writes there is still read through `config`.
    pub fn screened<'c>(&self, config: &'c RhfConfig) -> Cow<'c, RhfConfig> {
        let screening = self.screening();
        if config.screening == screening {
            Cow::Borrowed(config)
        } else {
            Cow::Owned(RhfConfig {
                screening,
                ..config.clone()
            })
        }
    }

    /// [`solve_rhf`](crate::rhf::solve_rhf) on this system, with the
    /// system's screening ([`Self::screened`]).
    pub fn solve_rhf(&self, config: &RhfConfig) -> Result<ScfResult, FerricError> {
        crate::rhf::solve_rhf(
            &self.ctx,
            &self.mol,
            &self.basis,
            self.op(),
            &self.bounds,
            &self.screened(config),
        )
    }

    /// [`solve_uhf`](crate::uhf::solve_uhf) on this system, with the
    /// system's screening ([`Self::screened`]).
    pub fn solve_uhf(&self, config: &RhfConfig) -> Result<ScfResult, FerricError> {
        crate::uhf::solve_uhf(
            &self.ctx,
            &self.mol,
            &self.basis,
            &self.bounds,
            &self.screened(config),
        )
    }

    /// [`solve_rohf`](crate::rohf::solve_rohf) on this system, with the
    /// system's screening ([`Self::screened`]).
    pub fn solve_rohf(&self, config: &RhfConfig) -> Result<ScfResult, FerricError> {
        crate::rohf::solve_rohf(
            &self.ctx,
            &self.mol,
            &self.basis,
            self.op(),
            &self.bounds,
            &self.screened(config),
        )
    }

    /// The converged reference by the spin rule of the correlated drivers:
    /// [`Self::solve_uhf`] when the multiplicity is above 1, else
    /// [`Self::solve_rhf`]; non-convergence is
    /// [`FerricError::ScfConvergence`] ([`ScfResult::into_converged`]).
    pub fn solve_reference(&self, config: &RhfConfig) -> Result<ScfResult, FerricError> {
        let scf = if self.mol.multiplicity > 1 {
            self.solve_uhf(config)?
        } else {
            self.solve_rhf(config)?
        };
        scf.into_converged()
    }

    /// [`solve_rhf_ladder`](crate::ladder::solve_rhf_ladder) on this system,
    /// every rung with the system's screening ([`Self::screened`]).
    pub fn solve_rhf_ladder(&self, ladder: &[Rung]) -> Result<LadderResult, FerricError> {
        let screening = self.screening();
        let ladder: Cow<'_, [Rung]> = if ladder.iter().all(|r| r.config.screening == screening) {
            Cow::Borrowed(ladder)
        } else {
            Cow::Owned(
                ladder
                    .iter()
                    .map(|r| Rung {
                        config: self.screened(&r.config).into_owned(),
                        restart: r.restart,
                    })
                    .collect(),
            )
        };
        crate::ladder::solve_rhf_ladder(
            &self.ctx,
            &self.mol,
            &self.basis,
            self.op(),
            &self.bounds,
            &ladder,
        )
    }
}

/// A fitting basis prepared on a [`PreparedSystem`]'s molecule by
/// [`PreparedSystem::prepare_aux`], the only constructor.
///
/// It borrows the system, so it cannot outlive it, and [`Self::system`]
/// returns that system: code handed a `&PreparedAux` reads the matching
/// orbital basis and molecule from it rather than pairing the aux with a
/// system of its own. It dereferences to the [`PreparedBasis`] that the
/// fitting kernels take.
///
/// The fitting basis takes its nuclear charges from its own basis set's ECP
/// table ([`PreparedBasis::new`]), so on an ECP atom an aux set without ECP
/// data carries the all-electron charge `Z`. Only a fitting kernel that read
/// the aux basis's charges would see that; the orbital basis is the one that
/// carries `Z - n_core` and `V_ECP`.
pub struct PreparedAux<'s> {
    system: &'s PreparedSystem,
    basis: PreparedBasis,
}

impl<'s> PreparedAux<'s> {
    /// The system whose molecule this basis was prepared on.
    pub fn system(&self) -> &'s PreparedSystem {
        self.system
    }

    /// The prepared fitting basis.
    pub fn basis(&self) -> &PreparedBasis {
        &self.basis
    }

    /// The prepared fitting basis, for a kernel that also holds `sys`: an
    /// error unless this basis was prepared on `sys` (the same system, not an
    /// equal one), so a kernel taking both cannot pair the fitting basis of
    /// one system with the orbital basis of another.
    pub fn basis_for(&self, sys: &PreparedSystem) -> Result<&PreparedBasis, FerricError> {
        if std::ptr::eq(self.system, sys) {
            Ok(&self.basis)
        } else {
            Err(FerricError::General(
                "a fitting basis was paired with a system it was not prepared on".to_string(),
            ))
        }
    }
}

/// The fitting basis only: the system it borrows is not printed.
impl std::fmt::Debug for PreparedAux<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparedAux")
            .field("basis", &self.basis)
            .finish_non_exhaustive()
    }
}

impl Deref for PreparedAux<'_> {
    type Target = PreparedBasis;

    fn deref(&self) -> &PreparedBasis {
        &self.basis
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
    fn an_already_applied_molecule_prepares_like_a_fresh_one() {
        let ecp = basis::bundled("def2-svp").unwrap();
        let fresh = PreparedSystem::new(hi(), &ecp).unwrap();
        let again = PreparedSystem::new(fresh.mol().clone(), &ecp).unwrap();
        assert_eq!(again.mol().nelec(), fresh.mol().nelec());
        assert_eq!(
            again.mol().nuclear_repulsion(),
            fresh.mol().nuclear_repulsion()
        );
        assert_eq!(again.bounds().q, fresh.bounds().q);
        // The same molecule in an all-electron basis is all-electron again.
        let sto = basis::bundled("sto-3g").unwrap();
        let all_electron = PreparedSystem::new(fresh.mol().clone(), &sto).unwrap();
        assert_eq!(all_electron.mol().nelec(), 54);
        assert_eq!(
            all_electron.mol().nuclear_repulsion(),
            hi().nuclear_repulsion()
        );
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
        let aux = sys.prepare_aux(&aux_bs).unwrap();
        assert_eq!(
            aux.nbasis(),
            PreparedBasis::new(&mol, &aux_bs).unwrap().nbasis()
        );
    }

    #[test]
    fn the_ecp_stage_exposes_the_valence_molecule_before_any_integral() {
        let bs = basis::bundled("def2-svp").unwrap();
        let ecp = EcpMolecule::new(hi(), &bs);
        assert_eq!(ecp.mol().nelec(), 26);
        assert!(std::ptr::eq(ecp.basis_set(), &bs));
        let mut by_hand = hi();
        by_hand.apply_ecp(&bs);
        let applied = EcpMolecule::new(hi(), &bs).into_mol();
        assert_eq!(applied.nelec(), by_hand.nelec());
        assert_eq!(applied.nuclear_repulsion(), by_hand.nuclear_repulsion());
        let sys = ecp.prepare(PrepareOptions::default()).unwrap();
        assert_eq!(sys.mol().nelec(), 26);
        assert_eq!(sys.screening(), ScreeningKind::Schwarz);
        assert_eq!(sys.into_mol().nelec(), 26);
    }

    #[test]
    fn the_screening_option_builds_the_bounds_compute_for_screening_builds() {
        let bs = basis::bundled("sto-3g").unwrap();
        for kind in [
            ScreeningKind::Schwarz,
            ScreeningKind::Csb,
            ScreeningKind::Csam,
        ] {
            let options = PrepareOptions {
                screening: kind,
                ..Default::default()
            };
            let sys = EcpMolecule::new(hi(), &bs).prepare(options).unwrap();
            let prep = PreparedBasis::new(&hi(), &bs).unwrap();
            let want =
                SchwarzBounds::compute_for_screening(Operator::coulomb(), &prep, kind).unwrap();
            assert_eq!(sys.screening(), kind);
            assert_eq!(sys.bounds().q, want.q);
            assert_eq!(sys.bounds().csb_m, want.csb_m);
            assert_eq!(sys.bounds().csam_x, want.csam_x);
            assert_eq!(sys.bounds().csb_m.is_some(), kind == ScreeningKind::Csb);
            assert_eq!(sys.bounds().csam_x.is_some(), kind == ScreeningKind::Csam);
        }
    }

    #[test]
    fn the_solve_methods_run_with_the_systems_screening() {
        let mol = Molecule::parse_xyz(
            "3\n\nO 0 0 0.117\nH 0 0.757 -0.467\nH 0 -0.757 -0.467\n",
            0,
            1,
        )
        .unwrap();
        let bs = basis::bundled("sto-3g").unwrap();
        let options = PrepareOptions {
            screening: ScreeningKind::Csb,
            ..Default::default()
        };
        let sys = EcpMolecule::new(mol, &bs).prepare(options).unwrap();
        let agreeing = RhfConfig {
            screening: ScreeningKind::Csb,
            ..Default::default()
        };
        assert!(matches!(sys.screened(&agreeing), Cow::Borrowed(_)));
        // A config that disagrees runs as the agreeing one: the same
        // energy, bit for bit, as the loose solver with the system's kind.
        let disagreeing = RhfConfig::default();
        assert_eq!(disagreeing.screening, ScreeningKind::Schwarz);
        let screened = sys.screened(&disagreeing);
        assert!(matches!(screened, Cow::Owned(_)));
        assert_eq!(screened.screening, ScreeningKind::Csb);
        let want = crate::rhf::solve_rhf(
            sys.ctx(),
            sys.mol(),
            sys.basis(),
            sys.op(),
            sys.bounds(),
            &agreeing,
        )
        .unwrap();
        let got = sys.solve_rhf(&disagreeing).unwrap();
        assert_eq!(got.energy.to_bits(), want.energy.to_bits());
        let rung = |config: RhfConfig| Rung {
            config,
            restart: false,
        };
        let lr = sys.solve_rhf_ladder(&[rung(disagreeing.clone())]).unwrap();
        let lw = sys.solve_rhf_ladder(&[rung(agreeing)]).unwrap();
        assert_eq!(lr.result.energy.to_bits(), lw.result.energy.to_bits());
        // The solver's route record is still read through the caller's config.
        let route = disagreeing.eri_route.get().expect("an in-core route");
        assert_eq!(route.nbasis, sys.basis().nbasis());
    }

    #[test]
    fn an_aux_names_the_system_it_was_prepared_on() {
        let bs = basis::bundled("def2-svp").unwrap();
        let aux_bs = basis::bundled("def2-svp-ri").unwrap();
        let sys = PreparedSystem::new(hi(), &bs).unwrap();
        let aux = sys.prepare_aux(&aux_bs).unwrap();
        assert!(std::ptr::eq(aux.system(), &sys));
        assert!(std::ptr::eq(aux.basis_for(&sys).unwrap(), aux.basis()));
        let other = PreparedSystem::new(hi(), &bs).unwrap();
        assert!(aux.basis_for(&other).is_err());
        // The fitting basis, not the orbital one.
        let by_hand = PreparedBasis::new(&hi(), &aux_bs).unwrap();
        assert_eq!(aux.nbasis(), by_hand.nbasis());
        assert_ne!(aux.nbasis(), sys.basis().nbasis());
    }
}
