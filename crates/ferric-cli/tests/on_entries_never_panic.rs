//! Every correlated `*_on` entry point, handed a reference that does not
//! suit it, returns an error (or a result) instead of panicking.
//!
//! The entries check their reference against the system
//! (`PreparedSystem::check_reference`), but a dimensionally valid reference
//! can still be wrong for a kernel: an unrestricted (or restricted-open)
//! result handed to a closed-shell kernel, a result with fewer MOs than basis
//! functions, or a system with no virtual orbitals (He / STO-3G). Each case
//! below is run under `catch_unwind` for every entry; a panic fails the test
//! and names the entry and the case. The expected outcome per case is
//! asserted where it is fixed (restricted-only kernels must refuse an
//! open-shell reference; every entry must refuse a rectangular MO matrix).
//!
//! The table lives in this crate because it is the one that depends on every
//! correlated crate.

use std::panic::{catch_unwind, AssertUnwindSafe};

use ferric_cc::CcConfig;
use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_integrals::operator::Operator;
use ferric_mp2::rimp2::RiMp2Config;
use ferric_scf::rhf::RhfConfig;
use ferric_scf::{PreparedAux, PreparedSystem, ScfResult};

const WATER: &str = "3\n\nO 0 0 0.117\nH 0 0.757 -0.467\nH 0 -0.757 -0.467\n";
const HE: &str = "1\n\nHe 0 0 0\n";

/// What an entry did with a reference.
#[derive(Debug)]
enum Outcome {
    Ok,
    Err(String),
    Panic(String),
}

fn run<T>(f: impl FnOnce() -> Result<T, FerricError>) -> Outcome {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(_)) => Outcome::Ok,
        Ok(Err(e)) => Outcome::Err(e.to_string()),
        Err(p) => Outcome::Panic(
            p.downcast_ref::<String>()
                .cloned()
                .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_default(),
        ),
    }
}

/// Which spin a kernel accepts.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Accepts {
    /// Closed-shell (RHF / RKS) references only.
    Restricted,
    /// Any spin.
    Any,
}

type Entry<'a> = (
    &'static str,
    Accepts,
    Box<dyn Fn(&PreparedAux<'_>, &ScfResult) -> Outcome + 'a>,
);

/// Every entry that takes a reference, with tiny settings.
fn entries<'a>() -> Vec<Entry<'a>> {
    use Accepts::*;
    let mp2 = RiMp2Config::default;
    let cc = CcConfig::default;
    let coulomb = Operator::coulomb();
    let pdep = || ferric_rpa::PdepRpaConfig {
        quadrature: ferric_rpa::config::QuadratureConfig {
            n_points: 4,
            ..Default::default()
        },
        ..Default::default()
    };
    let dh = || ferric_mp2::double_hybrid::DoubleHybridKind::B2plyp.mp2_config();
    vec![
        (
            "ri_mp2_on",
            Restricted,
            Box::new(move |a, r| run(|| ferric_mp2::rimp2::ri_mp2_on(a, r, &mp2()))),
        ),
        (
            "ri_mp2_with_op_on",
            Restricted,
            Box::new(move |a, r| {
                run(|| ferric_mp2::rimp2::ri_mp2_with_op_on(a, coulomb, r, &mp2()))
            }),
        ),
        (
            "ri_mp2_fitted_on",
            Restricted,
            Box::new(move |a, r| {
                run(|| {
                    ferric_mp2::rimp2::ri_mp2_fitted_on(
                        a,
                        r,
                        &ferric_mp2::omega_ri::RiFit::Coulomb,
                        &mp2(),
                    )
                })
            }),
        ),
        (
            "ri_mp2_spin_components_fitted_on",
            Restricted,
            Box::new(move |a, r| {
                run(|| {
                    ferric_mp2::rimp2::ri_mp2_spin_components_fitted_on(
                        a,
                        r,
                        &mp2(),
                        &ferric_mp2::omega_ri::RiFit::Coulomb,
                    )
                })
            }),
        ),
        (
            "u_ri_mp2_on",
            Any,
            Box::new(move |a, r| run(|| ferric_mp2::u_rimp2::u_ri_mp2_on(a, r, &mp2()))),
        ),
        (
            "u_ri_mp2_with_op_on",
            Any,
            Box::new(move |a, r| {
                run(|| ferric_mp2::u_rimp2::u_ri_mp2_with_op_on(a, coulomb, r, &mp2()))
            }),
        ),
        (
            "ri_mp2_spin_components_any_on",
            Any,
            Box::new(move |a, r| {
                run(|| ferric_mp2::u_rimp2::ri_mp2_spin_components_any_on(a, coulomb, r, &mp2()))
            }),
        ),
        (
            "mp3_energy_on",
            Restricted,
            Box::new(|a, r| run(|| ferric_mp2::mp3::mp3_energy_on(a, r, 0, None))),
        ),
        (
            "oo_ri_mp2_on",
            Restricted,
            Box::new(|a, r| {
                run(|| ferric_mp2::oo_rimp2::oo_ri_mp2_on(a, r, &Default::default(), None))
            }),
        ),
        (
            "u_oo_ri_mp2_on",
            Any,
            Box::new(|a, r| {
                run(|| ferric_mp2::u_oo_rimp2::u_oo_ri_mp2_on(a, r, &Default::default(), None))
            }),
        ),
        (
            "attenuated_ri_mp2_on",
            Any,
            Box::new(|a, r| {
                run(|| ferric_mp2::attenuated::attenuated_ri_mp2_on(a, r, &Default::default()))
            }),
        ),
        (
            "scs_mp2_on",
            Any,
            Box::new(|a, r| run(|| ferric_mp2::scs::scs_mp2_on(a, r, &Default::default()))),
        ),
        (
            "scs_mp2_2terfc_on",
            Any,
            Box::new(|a, r| run(|| ferric_mp2::scs::scs_mp2_2terfc_on(a, r, &Default::default()))),
        ),
        (
            "att_mp2_vv10_on",
            Restricted,
            Box::new(|a, r| {
                run(|| ferric_mp2::att_vv10::att_mp2_vv10_on(a, r, &Default::default()))
            }),
        ),
        (
            "u_att_mp2_vv10_on",
            Any,
            Box::new(|a, r| {
                run(|| ferric_mp2::att_vv10::u_att_mp2_vv10_on(a, r, &Default::default()))
            }),
        ),
        (
            "mp2_double_hybrid_on",
            Restricted,
            Box::new(move |a, r| {
                run(|| ferric_mp2::double_hybrid::mp2_double_hybrid_on(a, r, &dh()))
            }),
        ),
        (
            "u_mp2_double_hybrid_on",
            Any,
            Box::new(move |a, r| {
                run(|| ferric_mp2::double_hybrid::u_mp2_double_hybrid_on(a, r, &dh()))
            }),
        ),
        (
            "omega_ri_cdd_mp2_on",
            Restricted,
            Box::new(|a, r| {
                run(|| ferric_mp2::omega_ri_cdd::omega_ri_cdd_mp2_on(a, r, &Default::default()))
            }),
        ),
        (
            "laplace_ri_mp2_on",
            Restricted,
            Box::new(|a, r| run(|| ferric_mp2::laplace::laplace_ri_mp2_on(a, r, 4, 0, None))),
        ),
        (
            "laplace_sos_mp2_on(mo)",
            Restricted,
            Box::new(|a, r| {
                run(|| {
                    ferric_mp2::laplace::laplace_sos_mp2_on(
                        a,
                        r,
                        &Default::default(),
                        ferric_mp2::laplace::SosFormulation::Mo,
                    )
                })
            }),
        ),
        (
            "laplace_sos_mp2_on(ao)",
            Restricted,
            Box::new(|a, r| {
                run(|| {
                    ferric_mp2::laplace::laplace_sos_mp2_on(
                        a,
                        r,
                        &Default::default(),
                        ferric_mp2::laplace::SosFormulation::Ao,
                    )
                })
            }),
        ),
        (
            "laplace_sos_mp2_on(local)",
            Restricted,
            Box::new(|a, r| {
                run(|| {
                    ferric_mp2::laplace::laplace_sos_mp2_on(
                        a,
                        r,
                        &Default::default(),
                        ferric_mp2::laplace::SosFormulation::Local(8.0),
                    )
                })
            }),
        ),
        (
            "rimp2_gradient_analytical_on",
            Restricted,
            Box::new(move |a, r| {
                run(|| ferric_mp2::gradient::rimp2_gradient_analytical_on(a, r, &mp2(), None))
            }),
        ),
        (
            "amplitude_lmp2_on",
            Restricted,
            Box::new(move |a, r| {
                run(|| {
                    ferric_mp2::lmp2_amplitude::amplitude_lmp2_on(
                        a,
                        coulomb,
                        r,
                        &Default::default(),
                    )
                })
            }),
        ),
        (
            "amplitude_lmp2_direct_on",
            Restricted,
            Box::new(move |a, r| {
                run(|| {
                    ferric_mp2::lmp2_direct::amplitude_lmp2_direct_on(
                        a,
                        coulomb,
                        r,
                        &Default::default(),
                        &Default::default(),
                    )
                })
            }),
        ),
        (
            "att_mp2_vv10_scan_on",
            Restricted,
            Box::new(|a, r| {
                run(|| {
                    ferric_mp2::att_vv10::att_mp2_vv10_scan_on(
                        a,
                        r,
                        &Default::default(),
                        &[ferric_mp2::att_vv10::AttVv10ScanArm {
                            r0_bohr: 2.0,
                            omega: None,
                        }],
                        &[6.0],
                        false,
                    )
                })
            }),
        ),
        (
            "amplitude_drpa_scan_timed_on",
            Restricted,
            Box::new(|a, r| {
                run(|| {
                    ferric_mp2::drpa_amplitude::amplitude_drpa_scan_timed_on(
                        a,
                        r,
                        &Default::default(),
                        &[1e-4],
                    )
                })
            }),
        ),
        (
            "amplitude_drpa_on",
            Restricted,
            Box::new(|a, r| {
                run(|| ferric_mp2::drpa_amplitude::amplitude_drpa_on(a, r, &Default::default()))
            }),
        ),
        (
            "ccd_on",
            Restricted,
            Box::new(move |a, r| run(|| ferric_cc::ccd::ccd_on(a, r, &cc()))),
        ),
        (
            "linlccd_on",
            Restricted,
            Box::new(move |a, r| {
                run(|| {
                    ferric_cc::linlccd::linlccd_on(
                        a,
                        r,
                        &cc(),
                        ferric_cc::linlccd::LadderVariant::Hh,
                    )
                })
            }),
        ),
        (
            "amplitude_linlccd_on",
            Restricted,
            Box::new(|a, r| {
                run(|| {
                    ferric_cc::linlccd_amplitude::amplitude_linlccd_on(
                        a,
                        r,
                        &Default::default(),
                        ferric_cc::linlccd::LadderVariant::Hh,
                    )
                })
            }),
        ),
        (
            "ccsd_closed_shell_on",
            Restricted,
            Box::new(move |a, r| {
                run(|| ferric_cc::ccsd_closed_shell::ccsd_closed_shell_on(a, r, &cc()))
            }),
        ),
        (
            "u_ccsd_on",
            Any,
            Box::new(move |a, r| run(|| ferric_cc::u_ccsd::u_ccsd_on(a, r, &cc()))),
        ),
        (
            "run_pdep_rpa_on",
            Restricted,
            Box::new(move |a, r| run(|| ferric_rpa::run_pdep_rpa_on(a, r, &pdep()))),
        ),
        (
            "run_pdep_rpa_fitted_on",
            Restricted,
            Box::new(move |a, r| {
                run(|| {
                    ferric_rpa::run_pdep_rpa_fitted_on(
                        a,
                        r,
                        &pdep(),
                        &ferric_mp2::omega_ri::RiFit::Coulomb,
                    )
                })
            }),
        ),
        (
            "run_u_pdep_rpa_fitted_on",
            Any,
            Box::new(move |a, r| {
                run(|| {
                    ferric_rpa::run_u_pdep_rpa_fitted_on(
                        a,
                        r,
                        &pdep(),
                        &ferric_mp2::omega_ri::RiFit::Coulomb,
                    )
                })
            }),
        ),
        (
            "run_pdep_rpa_energy_on",
            Restricted,
            Box::new(move |a, r| run(|| ferric_rpa::run_pdep_rpa_energy_on(a, r, &pdep()))),
        ),
        (
            "rs_mp2_lr_rpa_on",
            Any,
            Box::new(|a, r| run(|| ferric_rpa::rs_mp2_lr_rpa_on(a, r, &Default::default()))),
        ),
        (
            "rpa_analytic_gradient_on",
            Restricted,
            Box::new(move |a, r| {
                run(|| {
                    ferric_rpa::gradient_analytic::rpa_analytic_gradient_on(
                        a,
                        r,
                        &ferric_rpa::PdepRpaConfig {
                            trunc_thresh: 0.0,
                            ..pdep()
                        },
                        ferric_rpa::gradient_analytic::RingOrder::Full,
                    )
                })
            }),
        ),
        (
            "run_gw_on",
            Restricted,
            Box::new(move |a, r| {
                run(|| ferric_gw::run_gw_on(a, r, None, &pdep(), &Default::default()))
            }),
        ),
        (
            "run_u_gw_on",
            Any,
            Box::new(move |a, r| {
                run(|| ferric_gw::run_u_gw_on(a, r, None, &pdep(), &Default::default()))
            }),
        ),
        (
            "run_bse_tda_on",
            Restricted,
            Box::new(move |a, r| run(|| ferric_gw::bse::run_bse_tda_on(a, r, &pdep(), 0))),
        ),
        (
            "run_rpax_static_polarizability_on",
            Restricted,
            Box::new(move |a, r| {
                run(|| ferric_gw::bse::run_rpax_static_polarizability_on(a, r, &pdep(), 0, 0.3))
            }),
        ),
        (
            "opmp3_on",
            Any,
            Box::new(|a, r| {
                let cfg = ferric_mp2::opmp3_driver::Opmp3Config::new(
                    ferric_mp2::opmp3_driver::Opmp3RefKind::Hf,
                );
                run(|| {
                    ferric_mp2::opmp3_driver::opmp3_on(
                        a.system(),
                        r,
                        ferric_mp2::opmp3_driver::Opmp3Integrals::Ri(a),
                        &cfg,
                    )
                })
            }),
        ),
        (
            "local_opmp3_on",
            Restricted,
            Box::new(|a, r| {
                let cfg = ferric_mp2::opmp3_driver::Opmp3Config::new(
                    ferric_mp2::opmp3_driver::Opmp3RefKind::Hf,
                );
                let local = ferric_mp2::lmp2_amplitude::AmplitudeLmp2Config {
                    eps: 1e-5,
                    ..Default::default()
                };
                run(|| ferric_mp2::local_opmp3::local_opmp3_on(a, r, &cfg, &local))
            }),
        ),
        (
            "run_tddft_on",
            Any,
            Box::new(|a, r| run(|| ferric_tddft::run_tddft_on(a, r, &Default::default()))),
        ),
    ]
}

struct Case {
    sys: PreparedSystem,
    aux_bs: basis::BasisSet,
    reference: ScfResult,
}

fn system(
    xyz: &str,
    charge: i32,
    mult: usize,
    bs: &str,
    aux: &str,
) -> (PreparedSystem, basis::BasisSet) {
    let mol = Molecule::parse_xyz(xyz, charge, mult).unwrap();
    let sys = PreparedSystem::new(mol, &basis::bundled(bs).unwrap()).unwrap();
    (sys, basis::bundled(aux).unwrap())
}

/// Run every entry on `case`; fail on any panic, and return the outcomes.
fn sweep(label: &str, case: &Case) -> Vec<(&'static str, Accepts, Outcome)> {
    let aux = case.sys.prepare_aux(&case.aux_bs).unwrap();
    let out: Vec<_> = entries()
        .into_iter()
        .map(|(name, accepts, f)| (name, accepts, f(&aux, &case.reference)))
        .collect();
    for (name, _, o) in &out {
        eprintln!("[{label}] {name}: {o:?}");
    }
    let panics: Vec<_> = out
        .iter()
        .filter_map(|(n, _, o)| match o {
            Outcome::Panic(msg) => Some(format!("{n}: {msg}")),
            _ => None,
        })
        .collect();
    assert!(
        panics.is_empty(),
        "[{label}] entries panicked:\n{}",
        panics.join("\n")
    );
    out
}

#[test]
fn an_open_shell_reference_is_refused_by_the_closed_shell_entries() {
    for (label, rohf) in [("uhf", false), ("rohf", true)] {
        let (sys, aux_bs) = system(WATER, 1, 2, "sto-3g", "cc-pvdz-ri");
        let cfg = RhfConfig::default();
        let reference = if rohf {
            sys.solve_rohf(&cfg)
        } else {
            sys.solve_uhf(&cfg)
        }
        .unwrap();
        let out = sweep(
            label,
            &Case {
                sys,
                aux_bs,
                reference,
            },
        );
        for (name, accepts, o) in out {
            if accepts == Accepts::Restricted {
                assert!(
                    matches!(o, Outcome::Err(_)),
                    "[{label}] {name} must refuse: {o:?}"
                );
            }
            // Positive controls: open-shell entries that need nothing else
            // must still accept a valid UHF / ROHF reference.
            if ["u_ri_mp2_on", "rs_mp2_lr_rpa_on"].contains(&name) {
                assert!(
                    matches!(o, Outcome::Ok),
                    "[{label}] {name} must accept: {o:?}"
                );
            }
        }
    }
}

#[test]
fn a_reference_with_fewer_mos_than_basis_functions_is_refused() {
    let (sys, aux_bs) = system(WATER, 0, 1, "sto-3g", "cc-pvdz-ri");
    let mut reference = sys.solve_rhf(&RhfConfig::default()).unwrap();
    reference.mos_alpha = reference.mos_alpha.slice(ndarray::s![.., ..6]).to_owned();
    reference.eps_alpha.truncate(6);
    let out = sweep(
        "rectangular",
        &Case {
            sys,
            aux_bs,
            reference,
        },
    );
    for (name, _, o) in out {
        match o {
            Outcome::Err(msg) => assert!(
                msg.contains("does not belong to this system"),
                "{name}: {msg}"
            ),
            other => panic!("[rectangular] {name} must refuse: {other:?}"),
        }
    }
}

#[test]
fn a_system_without_virtual_orbitals_never_panics() {
    // Closed shell: He / STO-3G, one MO, doubly occupied.
    let (sys, aux_bs) = system(HE, 0, 1, "sto-3g", "cc-pvdz-ri");
    let reference = sys.solve_rhf(&RhfConfig::default()).unwrap();
    sweep(
        "he-sto-3g",
        &Case {
            sys,
            aux_bs,
            reference,
        },
    );
    // Open shell: the H atom in STO-3G, its one alpha MO occupied.
    let (sys, aux_bs) = system("1\n\nH 0 0 0\n", 0, 2, "sto-3g", "cc-pvdz-ri");
    let reference = sys.solve_uhf(&RhfConfig::default()).unwrap();
    sweep(
        "h-atom-uhf-sto-3g",
        &Case {
            sys,
            aux_bs,
            reference,
        },
    );
}
