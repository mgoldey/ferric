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
//! refuse a config whose `RhfConfig::screening` names another kind, so
//! direct J/K (which reads the bounds' tables) and LinK (which reads the
//! config) cannot screen with two kinds. A driver that builds its own config
//! sets `screening: sys.screening()`.

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
            basis_set: self.basis.clone(),
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
    basis_set: BasisSet,
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

    /// The basis set [`Self::basis`] was prepared from (a copy of the one
    /// [`EcpMolecule::new`] was given), for the kernels that also read the
    /// shells by element (the local correlation models' atom domains).
    pub fn basis_set(&self) -> &BasisSet {
        &self.basis_set
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

    /// `Ok` when `config` screens with the kind the bounds were built for
    /// ([`Self::screening`]); otherwise an error naming both kinds. The one
    /// check behind every solve method.
    fn check_screening(&self, config: &RhfConfig) -> Result<(), FerricError> {
        let built = self.screening();
        if config.screening == built {
            Ok(())
        } else {
            Err(FerricError::General(format!(
                "the config screens with {:?} but the system's bounds were built for {:?} \
                 (PrepareOptions::screening); set RhfConfig::screening to \
                 PreparedSystem::screening()",
                config.screening, built
            )))
        }
    }

    /// `Ok` when `scf` can be a reference on this system; otherwise an error
    /// saying what does not fit. The one check behind every correlated
    /// `*_on` entry point that takes a reference (see [`PreparedAux`]).
    ///
    /// An [`ScfResult`] carries no molecule, so this checks what a kernel
    /// would index by the system:
    /// - the β fields match [`ScfResult::spin`]: a `Restricted` result has
    ///   no β MOs or orbital energies, an `Unrestricted` one has both (a
    ///   `RestrictedOpen` result may have either);
    /// - the total density is `nbasis × nbasis` ([`Self::basis`]);
    /// - each spin's MO coefficients have `nbasis` AO rows and one orbital
    ///   energy per MO, and the occupied orbitals of [`Self::mol`]'s
    ///   electron count and multiplicity fit in its MOs (a result without β
    ///   fields is checked for both spins on its α ones).
    ///
    /// A reference of another molecule in a basis of the same size (or of
    /// another geometry) is not detected.
    pub fn check_reference(&self, scf: &ScfResult) -> Result<(), FerricError> {
        use crate::result::Spin;
        let beta = (scf.mos_beta.is_some(), scf.eps_beta.is_some());
        let spin_fields_ok = match scf.spin {
            Spin::Restricted => beta == (false, false),
            Spin::Unrestricted => beta == (true, true),
            Spin::RestrictedOpen => beta.0 == beta.1,
        };
        if !spin_fields_ok {
            return reference_mismatch(format!(
                "it is {:?} but has beta MO coefficients: {}, beta orbital energies: {}",
                scf.spin, beta.0, beta.1
            ));
        }
        let nbasis = self.basis.nbasis();
        let (rows, cols) = scf.density_total.dim();
        if (rows, cols) != (nbasis, nbasis) {
            return reference_mismatch(format!(
                "its total density is {rows}x{cols}, but the system's orbital basis has \
                 {nbasis} functions"
            ));
        }
        let (n_alpha, n_beta) = self.occupied_counts();
        let subject = "the reference SCF result does";
        self.check_spin_orbitals(subject, "alpha", &scf.mos_alpha, &scf.eps_alpha, n_alpha)?;
        self.check_spin_orbitals(
            subject,
            "beta",
            scf.mos_beta.as_ref().unwrap_or(&scf.mos_alpha),
            scf.eps_beta.as_ref().unwrap_or(&scf.eps_alpha),
            n_beta,
        )
    }

    /// The per-spin part of [`Self::check_reference`] for closed-shell
    /// orbitals that are not an [`ScfResult`] (e.g. the optimised orbitals
    /// of an orbital-optimised MP2): `mos` has `nbasis` AO rows, `eps` one
    /// entry per MO, and the system's doubly occupied orbitals fit.
    pub fn check_closed_shell_orbitals(
        &self,
        mos: &ndarray::Array2<f64>,
        eps: &[f64],
    ) -> Result<(), FerricError> {
        let (n_alpha, _) = self.occupied_counts();
        self.check_spin_orbitals("the orbitals do", "alpha", mos, eps, n_alpha)
    }

    /// `(n_alpha, n_beta)` of [`Self::mol`]'s electron count and multiplicity.
    fn occupied_counts(&self) -> (usize, usize) {
        let nelec = i64::from(self.mol.nelec());
        let mult = self.mol.multiplicity as i64;
        let count = |n: i64| usize::try_from(n.max(0) / 2).unwrap_or(0);
        (count(nelec + mult - 1), count(nelec - mult + 1))
    }

    fn check_spin_orbitals(
        &self,
        subject: &str,
        spin: &str,
        mos: &ndarray::Array2<f64>,
        eps: &[f64],
        n_occ: usize,
    ) -> Result<(), FerricError> {
        let nbasis = self.basis.nbasis();
        let n_eps = eps.len();
        let (rows, nmo) = mos.dim();
        if rows != nbasis {
            return orbitals_mismatch(
                subject,
                format!(
                    "its {spin} MO coefficients have {rows} AO rows, but the system's orbital \
                 basis has {nbasis} functions"
                ),
            );
        }
        if n_eps != nmo {
            return orbitals_mismatch(
                subject,
                format!("it has {n_eps} {spin} orbital energies for {nmo} {spin} MOs"),
            );
        }
        if n_occ > nmo {
            return orbitals_mismatch(
                subject,
                format!(
                    "the system's molecule occupies {n_occ} {spin} orbitals, but the result has \
                 {nmo} {spin} MOs"
                ),
            );
        }
        Ok(())
    }

    /// [`solve_rhf`](crate::rhf::solve_rhf) on this system; a config
    /// whose screening is not the system's is an error.
    pub fn solve_rhf(&self, config: &RhfConfig) -> Result<ScfResult, FerricError> {
        self.check_screening(config)?;
        crate::rhf::solve_rhf(
            &self.ctx,
            &self.mol,
            &self.basis,
            self.op(),
            &self.bounds,
            config,
        )
    }

    /// [`solve_uhf`](crate::uhf::solve_uhf) on this system; a config
    /// whose screening is not the system's is an error.
    pub fn solve_uhf(&self, config: &RhfConfig) -> Result<ScfResult, FerricError> {
        self.check_screening(config)?;
        crate::uhf::solve_uhf(&self.ctx, &self.mol, &self.basis, &self.bounds, config)
    }

    /// [`solve_rohf`](crate::rohf::solve_rohf) on this system; a config
    /// whose screening is not the system's is an error.
    pub fn solve_rohf(&self, config: &RhfConfig) -> Result<ScfResult, FerricError> {
        self.check_screening(config)?;
        crate::rohf::solve_rohf(
            &self.ctx,
            &self.mol,
            &self.basis,
            self.op(),
            &self.bounds,
            config,
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

    /// [`solve_rhf_ladder`](crate::ladder::solve_rhf_ladder) on this system;
    /// a rung whose screening is not the system's is an error.
    pub fn solve_rhf_ladder(&self, ladder: &[Rung]) -> Result<LadderResult, FerricError> {
        for rung in ladder {
            self.check_screening(&rung.config)?;
        }
        crate::ladder::solve_rhf_ladder(
            &self.ctx,
            &self.mol,
            &self.basis,
            self.op(),
            &self.bounds,
            ladder,
        )
    }
}

/// The error of [`PreparedSystem::check_reference`]: `what` does not fit.
fn reference_mismatch(what: String) -> Result<(), FerricError> {
    orbitals_mismatch("the reference SCF result does", what)
}

/// `subject` (with its verb: "the reference SCF result does", or "the
/// orbitals do" for orbitals passed in its place) does not fit the system:
/// `what`.
fn orbitals_mismatch(subject: &str, what: String) -> Result<(), FerricError> {
    Err(FerricError::General(format!(
        "{subject} not belong to this system: {what}; pass the result of an SCF on \
         the same molecule and orbital basis"
    )))
}

/// A fitting basis prepared on a [`PreparedSystem`]'s molecule by
/// [`PreparedSystem::prepare_aux`], the only constructor.
///
/// It borrows the system, so it cannot outlive it, and [`Self::system`]
/// returns that system: code handed a `&PreparedAux` reads the matching
/// orbital basis and molecule from it rather than pairing the aux with a
/// system of its own. It does not dereference to its [`PreparedBasis`]: a
/// loose fitting kernel is reached through [`Self::basis`], so a frontend
/// goes through the `*_on` entry points.
///
/// The correlated crates' `*_on(aux, ..)` entry points (`ri_mp2_on`,
/// `ccsd_closed_shell_on`, `run_gw_on`, `run_tddft_on`, ...) take a
/// `&PreparedAux` and read the molecule (with the ECP applied), the orbital
/// basis (and, where the kernel takes them, the basis set, bounds, context
/// and the Coulomb operator) from [`Self::system`], with this basis as the
/// fitting basis, so a frontend cannot pair the fitting basis of one system
/// with the orbital basis of another. Each checks its reference SCF result
/// against the system first ([`Self::check_reference`]), so a result of
/// another system is an error rather than an out-of-bounds slice. An entry
/// whose kernel runs on a correlation operator other than the system's
/// Coulomb one takes that operator as an argument.
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

    /// [`PreparedSystem::check_reference`] on [`Self::system`]: `Ok` when
    /// `scf` can be a reference on the system this basis was prepared on.
    pub fn check_reference(&self, scf: &ScfResult) -> Result<(), FerricError> {
        self.system.check_reference(scf)
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
            aux.basis().nbasis(),
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

    fn csb_water() -> PreparedSystem {
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
        EcpMolecule::new(mol, &bs).prepare(options).unwrap()
    }

    fn rung(config: RhfConfig) -> Rung {
        Rung {
            config,
            restart: false,
        }
    }

    #[test]
    fn the_solve_methods_refuse_a_config_screening_with_another_kind() {
        let sys = csb_water();
        let schwarz = RhfConfig::default();
        assert_eq!(schwarz.screening, ScreeningKind::Schwarz);
        let errors = [
            sys.solve_rhf(&schwarz).map(|_| ()).unwrap_err(),
            sys.solve_uhf(&schwarz).map(|_| ()).unwrap_err(),
            sys.solve_rohf(&schwarz).map(|_| ()).unwrap_err(),
            sys.solve_reference(&schwarz).map(|_| ()).unwrap_err(),
        ];
        for e in errors {
            let msg = e.to_string();
            assert!(msg.contains("Schwarz") && msg.contains("Csb"), "{msg}");
            assert!(msg.contains("PrepareOptions::screening"), "{msg}");
        }
        // One disagreeing rung refuses the whole ladder, before any SCF.
        let csb = RhfConfig {
            screening: ScreeningKind::Csb,
            ..Default::default()
        };
        assert!(sys
            .solve_rhf_ladder(&[rung(csb), rung(schwarz.clone())])
            .is_err());
        assert!(schwarz.eri_route.get().is_none());
    }

    #[test]
    fn a_csb_system_solves_a_csb_config_as_the_loose_solver_does() {
        let sys = csb_water();
        // LinK reads `RhfConfig::screening`; direct J reads the bounds' table.
        let csb = RhfConfig {
            screening: ScreeningKind::Csb,
            k_builder: Some(crate::options::KBuilderKind::Link),
            ..Default::default()
        };
        let want = crate::rhf::solve_rhf(
            sys.ctx(),
            sys.mol(),
            sys.basis(),
            sys.op(),
            sys.bounds(),
            &csb,
        )
        .unwrap();
        let got = sys.solve_rhf(&csb).unwrap();
        assert!(got.converged);
        assert_eq!(got.energy.to_bits(), want.energy.to_bits());
        let lr = sys.solve_rhf_ladder(&[rung(csb)]).unwrap();
        assert!(lr.converged);
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
        assert_eq!(aux.basis().nbasis(), by_hand.nbasis());
        assert_ne!(aux.basis().nbasis(), sys.basis().nbasis());
    }

    fn water(basis_name: &str, charge: i32, mult: usize) -> PreparedSystem {
        let mol = Molecule::parse_xyz(
            "3\n\nO 0 0 0.117\nH 0 0.757 -0.467\nH 0 -0.757 -0.467\n",
            charge,
            mult,
        )
        .unwrap();
        PreparedSystem::new(mol, &basis::bundled(basis_name).unwrap()).unwrap()
    }

    #[test]
    fn the_system_keeps_the_basis_set_it_was_prepared_from() {
        let bs = basis::bundled("def2-svp").unwrap();
        let sys = PreparedSystem::new(hi(), &bs).unwrap();
        assert_eq!(sys.basis_set().name, bs.name);
        assert_eq!(sys.basis_set().shells, bs.shells);
        let via_stage = EcpMolecule::new(hi(), &bs)
            .prepare(PrepareOptions::default())
            .unwrap();
        assert_eq!(via_stage.basis_set().shells, bs.shells);
    }

    #[test]
    fn a_reference_of_the_system_passes_the_reference_check() {
        let sys = water("sto-3g", 0, 1);
        let rhf = sys.solve_rhf(&RhfConfig::default()).unwrap();
        sys.check_reference(&rhf).unwrap();
        let aux = sys
            .prepare_aux(&basis::bundled("cc-pvdz-ri").unwrap())
            .unwrap();
        aux.check_reference(&rhf).unwrap();
        // Open shell: UHF and ROHF of the cation.
        let cation = water("sto-3g", 1, 2);
        let uhf = cation.solve_uhf(&RhfConfig::default()).unwrap();
        cation.check_reference(&uhf).unwrap();
        let rohf = cation.solve_rohf(&RhfConfig::default()).unwrap();
        cation.check_reference(&rohf).unwrap();
    }

    #[test]
    fn a_reference_of_another_system_is_an_error_naming_what_does_not_fit() {
        let small = water("sto-3g", 0, 1);
        let large = water("6-31g", 0, 1);
        let rhf_small = small.solve_rhf(&RhfConfig::default()).unwrap();
        let rhf_large = large.solve_rhf(&RhfConfig::default()).unwrap();
        for (sys, rhf, rows, nb) in [(&small, &rhf_large, 13, 7), (&large, &rhf_small, 7, 13)] {
            let msg = sys.check_reference(rhf).unwrap_err().to_string();
            assert!(msg.contains("does not belong to this system"), "{msg}");
            assert!(
                msg.contains(&format!("{rows} AO rows"))
                    && msg.contains(&format!("{nb} functions")),
                "{msg}"
            );
        }
        // Orbital energies that do not match the MOs.
        let mut short = rhf_small.clone();
        short.eps_alpha.pop();
        let msg = small.check_reference(&short).unwrap_err().to_string();
        assert!(
            msg.contains("6 alpha orbital energies for 7 alpha MOs"),
            "{msg}"
        );
        // Fewer MOs than the molecule's occupied orbitals.
        let mut few = rhf_small.clone();
        few.mos_alpha = few.mos_alpha.slice(ndarray::s![.., ..4]).to_owned();
        few.eps_alpha.truncate(4);
        let msg = small.check_reference(&few).unwrap_err().to_string();
        assert!(msg.contains("occupies 5 alpha orbitals"), "{msg}");
        // A UHF result's beta channel is checked too.
        let cation = water("sto-3g", 1, 2);
        let mut uhf = cation.solve_uhf(&RhfConfig::default()).unwrap();
        uhf.mos_beta = Some(rhf_large.mos_alpha.clone());
        let msg = cation.check_reference(&uhf).unwrap_err().to_string();
        assert!(
            msg.contains("beta MO coefficients have 13 AO rows"),
            "{msg}"
        );
    }

    #[test]
    fn the_reference_check_wants_beta_fields_matching_the_spin_and_a_square_density() {
        let sys = water("sto-3g", 0, 1);
        let rhf = sys.solve_rhf(&RhfConfig::default()).unwrap();
        // A restricted result with beta orbitals.
        let mut r = rhf.clone();
        r.mos_beta = Some(rhf.mos_alpha.clone());
        r.eps_beta = Some(rhf.eps_alpha.clone());
        let msg = sys.check_reference(&r).unwrap_err().to_string();
        assert!(
            msg.contains("it is Restricted but has beta MO coefficients: true"),
            "{msg}"
        );
        // An unrestricted result without beta orbital energies.
        let cation = water("sto-3g", 1, 2);
        let mut u = cation.solve_uhf(&RhfConfig::default()).unwrap();
        u.eps_beta = None;
        let msg = cation.check_reference(&u).unwrap_err().to_string();
        assert!(msg.contains("it is Unrestricted but has beta MO coefficients: true, beta orbital energies: false"), "{msg}");
        // A density of the wrong size.
        let mut d = rhf.clone();
        d.density_total = ndarray::Array2::zeros((6, 6));
        let msg = sys.check_reference(&d).unwrap_err().to_string();
        assert!(
            msg.contains("its total density is 6x6") && msg.contains("7 functions"),
            "{msg}"
        );
        // Orbitals that are not an SCF result.
        sys.check_closed_shell_orbitals(&rhf.mos_alpha, &rhf.eps_alpha)
            .unwrap();
        let large = water("6-31g", 0, 1)
            .solve_rhf(&RhfConfig::default())
            .unwrap();
        let msg = sys
            .check_closed_shell_orbitals(&large.mos_alpha, &large.eps_alpha)
            .unwrap_err()
            .to_string();
        assert!(
            msg.starts_with("General error: the orbitals do not belong to this system"),
            "{msg}"
        );
    }
}
