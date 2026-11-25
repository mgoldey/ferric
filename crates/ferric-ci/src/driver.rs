//! Molecule-to-energy drivers for CASSCF and SC-NEVPT2.
//!
//! Each driver runs the exact-J/K Coulomb RHF that starts the calculation
//! ([`rhf_start`](crate::driver::rhf_start)), then two-step CASSCF
//! ([`casscf_on`](crate::driver::casscf_on)), CAS-CI
//! ([`casci_on`](crate::driver::casci_on)) or a CASSCF or CAS-CI
//! reference plus SC-NEVPT2 ([`nevpt2_on`](crate::driver::nevpt2_on)).
//! The Python bindings (`ferric.run_casci`, `ferric.run_casscf`,
//! `ferric.run_nevpt2`) and the CLI kinds (`method.kind = "casscf"`,
//! `"nevpt2"`) call these functions, so the same input gives the same number
//! on both surfaces.
//!
//! Every driver takes the [`PreparedSystem`](ferric_scf::PreparedSystem)
//! (molecule with the basis set's ECP applied, its prepared basis), so a
//! frontend cannot hand the kernels a molecule and a basis that disagree
//! about the ECP. The kernels underneath
//! ([`run_casscf`], [`run_cas_ci`], [`sc_nevpt2`]) take the molecule and the
//! basis separately.

use crate::{
    run_cas_ci, run_casscf, sc_nevpt2, CasCiConfig, CasCiResult, CasScfConfig, CasScfResult,
    Nevpt2Config, Nevpt2Integrals, Nevpt2Reference, Nevpt2Result,
};
use ferric_core::FerricError;
use ferric_scf::rhf::RhfConfig;
use ferric_scf::{PreparedSystem, ScfResult};

/// Convergence settings of the RHF start; `None` keeps the [`RhfConfig`]
/// default.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ScfStart {
    /// SCF iteration cap.
    pub max_iter: Option<usize>,
    /// SCF energy-change threshold (Ha).
    pub energy_conv: Option<f64>,
    /// SCF density-change threshold.
    pub density_conv: Option<f64>,
    /// `RhfConfig::gc_rotation` of the start (the direct J/K contraction
    /// rotation, `ferric_scf::direct_jk::gc_rotation`); `None` = unset.
    /// Pass the same value as `CasScfConfig::gc_rotation` so the CASSCF
    /// direct builds follow the start.
    pub gc_rotation: Option<ferric_scf::direct_jk::GcRotate>,
    /// `RhfConfig::jk_digest` of the start (the direct J/K digest,
    /// `ferric_scf::quartet_scatter::JkDigest`); `None` = unset. Pass the
    /// same value as `CasScfConfig::jk_digest`.
    pub jk_digest: Option<ferric_scf::quartet_scatter::JkDigest>,
}

/// The exact-J/K Coulomb RHF of `sys` (its molecule with the basis set's
/// ECP applied, its prepared basis and Schwarz bounds) with the
/// [`RhfConfig`] defaults, `start`'s overrides and the memory budget
/// (`three_index_budget_bytes`; `None` is the auto-detected budget),
/// screening with the system's kind ([`PreparedSystem::screening`]). An
/// unconverged SCF is a [`FerricError::ScfConvergence`].
pub fn rhf_start(
    sys: &PreparedSystem,
    start: ScfStart,
    memory_budget_bytes: Option<usize>,
) -> Result<ScfResult, FerricError> {
    let cfg = start_config(start, memory_budget_bytes, sys.screening())?;
    sys.solve_rhf(&cfg)?.into_converged()
}

/// The [`RhfConfig`] of the RHF start, lowered through
/// [`ScfOptions`](ferric_scf::options::ScfOptions): `start`'s thresholds,
/// the budget and the screening kind, every other field at its default.
fn start_config(
    start: ScfStart,
    memory_budget_bytes: Option<usize>,
    screening: ferric_scf::screening::ScreeningKind,
) -> Result<RhfConfig, FerricError> {
    let ScfStart {
        max_iter,
        energy_conv,
        density_conv,
        gc_rotation,
        jk_digest,
    } = start;
    let mut cfg = ferric_scf::options::ScfOptions {
        max_iter,
        energy_conv,
        density_conv,
        three_index_budget_bytes: memory_budget_bytes,
        screening: Some(screening),
        ..Default::default()
    }
    .into_rhf_config(ferric_scf::df_j::RiJTask::Energy)?;
    cfg.gc_rotation = gc_rotation;
    cfg.jk_digest = jk_digest;
    Ok(cfg)
}

/// The doubly occupied core count `(nelec − n_α − n_β) / 2` that the active
/// electrons leave out of `nelec`. An odd or negative remainder is an error
/// whose text starts with `caller`.
pub fn default_n_core(
    nelec: usize,
    n_elec_active: (usize, usize),
    caller: &str,
) -> Result<usize, String> {
    let n_act_el = n_elec_active.0 + n_elec_active.1;
    if n_act_el > nelec || !(nelec - n_act_el).is_multiple_of(2) {
        return Err(format!(
            "{caller}: {n_act_el} active electrons do not leave an even core count out of {nelec}"
        ));
    }
    Ok((nelec - n_act_el) / 2)
}

/// The reference SC-NEVPT2 is computed on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Nevpt2On {
    /// Two-step CASSCF from the RHF orbitals.
    Casscf,
    /// CAS-CI on the canonical RHF orbitals.
    Casci,
}

impl Nevpt2On {
    /// `"casscf"` or `"casci"`.
    pub fn as_str(self) -> &'static str {
        match self {
            Nevpt2On::Casscf => "casscf",
            Nevpt2On::Casci => "casci",
        }
    }

    /// Strict parse of `"casscf"` / `"casci"`; any other spelling is an
    /// error, never a default.
    pub fn parse_config_str(s: &str) -> Result<Self, String> {
        match s {
            "casscf" => Ok(Nevpt2On::Casscf),
            "casci" => Ok(Nevpt2On::Casci),
            other => Err(format!(
                "reference must be \"casscf\" or \"casci\", got {other:?}"
            )),
        }
    }
}

/// SC-NEVPT2 on a converged reference, with the reference's own counters.
#[derive(Debug, Clone)]
pub struct Nevpt2Run {
    /// The SC-NEVPT2 result.
    pub result: Nevpt2Result,
    /// CASSCF macro iterations, or CAS-CI Davidson iterations.
    pub reference_iterations: usize,
    /// The CASSCF orbital-gradient norm; `None` on a CAS-CI reference.
    pub reference_grad_norm: Option<f64>,
    /// The CASSCF reference's second-order check; `None` on a CAS-CI
    /// reference or with `stability = "off"`.
    pub reference_stability: Option<crate::CasScfStability>,
}

/// The CAS-CI settings of the [`Nevpt2On::Casci`] reference: `cas`'s
/// partition, CI settings and budget, the active window starting at
/// `cas.n_core`, no RDMs.
pub fn casci_config_of(cas: &CasScfConfig) -> CasCiConfig {
    CasCiConfig {
        n_active: cas.n_active,
        n_elec_active: cas.n_elec_active,
        active_start: cas.n_core,
        conv_thresh: cas.ci_conv_thresh,
        max_subspace: cas.ci_max_subspace,
        max_iter: cas.ci_max_iter,
        memory_budget_bytes: cas.memory_budget_bytes,
        compute_rdms: false,
    }
}

/// Two-step CASSCF ([`run_casscf`]) on `sys` from its converged RHF `rhf`.
pub fn casscf_on(
    sys: &PreparedSystem,
    rhf: &ScfResult,
    cas: &CasScfConfig,
) -> Result<CasScfResult, FerricError> {
    sys.check_reference(rhf)?;
    run_casscf(sys.mol(), sys.basis(), rhf, cas)
}

/// CAS-CI ([`run_cas_ci`]) on `sys` from its converged RHF `rhf`.
pub fn casci_on(
    sys: &PreparedSystem,
    rhf: &ScfResult,
    cas: &CasCiConfig,
) -> Result<CasCiResult, FerricError> {
    sys.check_reference(rhf)?;
    run_cas_ci(sys.mol(), sys.basis(), rhf, *cas)
}

/// SC-NEVPT2 on `sys` and the reference `on` built from its converged RHF
/// `rhf`: [`run_casscf`] with `cas`, or [`run_cas_ci`] with [`casci_config_of`]
/// (`cas`'s orbital-optimization settings are then not read). The
/// SC-NEVPT2 settings are the [`Nevpt2Config`] defaults with `cas`'s memory
/// budget and the Hamiltonian of `cas.integrals`
/// ([`Nevpt2Integrals::of_casscf`]: density-fitted on the RI route). An
/// unconverged reference is an error, and so is the RI route on a CAS-CI
/// reference (the CAS-CI runs on the exact integrals).
pub fn nevpt2_on(
    sys: &PreparedSystem,
    rhf: &ScfResult,
    on: Nevpt2On,
    cas: &CasScfConfig,
) -> Result<Nevpt2Run, FerricError> {
    sys.check_reference(rhf)?;
    let nev = Nevpt2Config {
        memory_budget_bytes: cas.memory_budget_bytes,
        integrals: Nevpt2Integrals::of_casscf(cas.integrals),
        ..Default::default()
    };
    let (mol, prep) = (sys.mol(), sys.basis());
    if on == Nevpt2On::Casci && nev.integrals != Nevpt2Integrals::Dense {
        return Err(FerricError::General(
            "SC-NEVPT2: integrals = \"ri\" sets the CASSCF Hamiltonian; a CAS-CI reference \
             runs on the exact integrals, so it is refused there"
                .to_string(),
        ));
    }
    match on {
        Nevpt2On::Casscf => {
            let r = run_casscf(mol, prep, rhf, cas)?;
            let reference = Nevpt2Reference::from_casscf(&r, cas)?;
            let result = sc_nevpt2(mol, prep, &reference, &nev)?;
            Ok(Nevpt2Run {
                result,
                reference_iterations: r.iterations,
                reference_grad_norm: Some(r.grad_norm),
                reference_stability: r.stability,
            })
        }
        Nevpt2On::Casci => {
            let cfg = casci_config_of(cas);
            let ci = run_cas_ci(mol, prep, rhf, cfg)?;
            let reference = Nevpt2Reference::from_casci(&ci, rhf.mos_r(), &cfg)?;
            let result = sc_nevpt2(mol, prep, &reference, &nev)?;
            Ok(Nevpt2Run {
                result,
                reference_iterations: ci.iterations,
                reference_grad_norm: None,
                reference_stability: None,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `start_config` (through `ScfOptions`) reproduces the config the RHF
    /// start built before roadmap item 11(a): the
    /// `RhfConfig { three_index_budget_bytes, ..Default::default() }` literal,
    /// the three threshold overrides, then `screening` (the literal's
    /// `..start_config(..)` update in `rhf_start`), field for field
    /// (`Debug` text; `RhfConfig` has no `PartialEq`).
    #[test]
    fn start_config_is_the_old_literal() {
        use ferric_scf::screening::ScreeningKind;
        let old = |start: ScfStart, budget: Option<usize>, screening| {
            let mut cfg = RhfConfig::default();
            cfg.three_index_budget_bytes = budget.unwrap_or(0);
            cfg.max_iter = start.max_iter.unwrap_or(cfg.max_iter);
            cfg.energy_conv = start.energy_conv.unwrap_or(cfg.energy_conv);
            cfg.density_conv = start.density_conv.unwrap_or(cfg.density_conv);
            cfg.screening = screening;
            cfg.gc_rotation = start.gc_rotation;
            cfg.jk_digest = start.jk_digest;
            cfg
        };
        let all = ScfStart {
            max_iter: Some(77),
            energy_conv: Some(1e-9),
            density_conv: Some(3e-8),
            gc_rotation: Some(ferric_scf::direct_jk::GcRotate::Off),
            jk_digest: Some(ferric_scf::quartet_scatter::JkDigest::Legacy),
        };
        for (start, budget, screening) in [
            (ScfStart::default(), None, ScreeningKind::default()),
            (all, Some(1 << 30), ScreeningKind::Csb),
        ] {
            let new = start_config(start, budget, screening).unwrap();
            assert_eq!(
                format!("{:?}", old(start, budget, screening)),
                format!("{new:?}")
            );
        }
    }

    /// `rhf_start` builds its own config; on a CSB system it must screen
    /// with CSB (the system's solve methods refuse any other kind).
    #[test]
    fn rhf_start_runs_on_a_csb_system() {
        use ferric_scf::screening::ScreeningKind;
        use ferric_scf::{EcpMolecule, PrepareOptions};
        let mol = ferric_core::mol::Molecule::parse_xyz(
            "3\n\nO 0 0 0.117\nH 0 0.757 -0.467\nH 0 -0.757 -0.467\n",
            0,
            1,
        )
        .unwrap();
        let bs = ferric_core::basis::bundled("sto-3g").unwrap();
        let options = PrepareOptions {
            screening: ScreeningKind::Csb,
            ..Default::default()
        };
        let sys = EcpMolecule::new(mol, &bs).prepare(options).unwrap();
        assert!(
            rhf_start(&sys, ScfStart::default(), None)
                .unwrap()
                .converged
        );
    }

    #[test]
    fn default_n_core_needs_an_even_remainder() {
        assert_eq!(default_n_core(10, (2, 2), "x"), Ok(3));
        assert_eq!(default_n_core(16, (5, 3), "x"), Ok(4));
        let e = default_n_core(10, (2, 1), "run_casscf").unwrap_err();
        assert_eq!(
            e,
            "run_casscf: 3 active electrons do not leave an even core count out of 10"
        );
        assert!(default_n_core(2, (2, 2), "x").is_err());
    }

    #[test]
    fn nevpt2_reference_parse_is_strict() {
        for on in [Nevpt2On::Casscf, Nevpt2On::Casci] {
            assert_eq!(Nevpt2On::parse_config_str(on.as_str()), Ok(on));
        }
        for bad in ["CASSCF", "rhf", ""] {
            let e = Nevpt2On::parse_config_str(bad).unwrap_err();
            assert!(e.starts_with("reference must be"), "{e}");
        }
    }

    #[test]
    fn casci_config_takes_the_partition_and_ci_settings() {
        let cas = CasScfConfig {
            n_active: 4,
            n_elec_active: (2, 2),
            n_core: 3,
            ci_conv_thresh: 1e-11,
            ci_max_subspace: 30,
            ci_max_iter: 77,
            memory_budget_bytes: Some(1 << 30),
            ..Default::default()
        };
        let c = casci_config_of(&cas);
        assert_eq!(
            c,
            CasCiConfig {
                n_active: 4,
                n_elec_active: (2, 2),
                active_start: 3,
                conv_thresh: 1e-11,
                max_subspace: 30,
                max_iter: 77,
                memory_budget_bytes: Some(1 << 30),
                compute_rdms: false,
            }
        );
    }

    /// Water in `basis_name` (STO-3G: 7 functions, 6-31G: 13), with its
    /// converged RHF, for the reference-check tests of the `*_on` entries.
    fn reference_check_water(
        basis_name: &str,
    ) -> (ferric_scf::PreparedSystem, ferric_scf::ScfResult) {
        let mol = ferric_core::mol::Molecule::parse_xyz(
            "3\n\nO 0 0 0.117\nH 0 0.757 -0.467\nH 0 -0.757 -0.467\n",
            0,
            1,
        )
        .unwrap();
        let bs = ferric_core::basis::bundled(basis_name).unwrap();
        let sys = ferric_scf::PreparedSystem::new(mol, &bs).unwrap();
        let rhf = sys
            .solve_rhf(&ferric_scf::rhf::RhfConfig::default())
            .unwrap();
        (sys, rhf)
    }

    fn assert_reference_refused<T>(who: &str, r: Result<T, ferric_core::FerricError>) {
        let msg = r.map(|_| ()).unwrap_err().to_string();
        assert!(
            msg.contains("does not belong to this system"),
            "{who}: {msg}"
        );
    }

    #[test]
    fn on_entries_refuse_a_reference_of_another_system() {
        let (small, _) = reference_check_water("sto-3g");
        let (_large, rhf_large) = reference_check_water("6-31g");
        let cas = CasScfConfig {
            n_active: 2,
            n_elec_active: (1, 1),
            n_core: 4,
            ..Default::default()
        };
        assert_reference_refused("casscf_on", casscf_on(&small, &rhf_large, &cas));
        assert_reference_refused(
            "casci_on",
            casci_on(&small, &rhf_large, &casci_config_of(&cas)),
        );
        assert_reference_refused(
            "nevpt2_on",
            nevpt2_on(&small, &rhf_large, Nevpt2On::Casscf, &cas),
        );
    }
}
