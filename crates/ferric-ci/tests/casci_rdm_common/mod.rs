//! Systems and the CAS-CI run shared by `tests/casci_rdms.rs` and
//! `examples/casci_rdm_dump.rs` (the input of `scripts/gen_casci_rdm_refs.py`).
//! Both must run the same RHF and window, or the reference orbitals and the
//! tested orbitals would differ.

#![allow(dead_code)]

use ferric_ci::hamiltonian::DeterminantSpace;
use ferric_ci::strings::enumerate_strings;
use ferric_ci::{run_cas_ci, CasCiConfig, CasCiResult};
use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::ScfResult;
use std::path::PathBuf;

/// Davidson residual-norm threshold for every RDM run.
pub const CONV_THRESH: f64 = 1e-10;

/// One CAS-CI system.
#[derive(Debug, Clone, Copy)]
pub struct System {
    pub name: &'static str,
    pub xyz: &'static str,
    pub basis: &'static str,
    pub active_start: usize,
    pub n_active: usize,
    pub n_elec: (usize, usize),
}

/// The validation systems of `tests/validation_casci.rs`, same windows.
pub const SYSTEMS: [System; 2] = [
    System {
        name: "n2_r1.10",
        xyz: "n2_r1.10.xyz",
        basis: "cc-pvdz",
        active_start: 4,
        n_active: 6,
        n_elec: (3, 3),
    },
    System {
        name: "h2o",
        xyz: "h2o.xyz",
        basis: "6-31g",
        active_start: 3,
        n_active: 4,
        n_elec: (2, 2),
    },
];

pub fn system(name: &str) -> System {
    *SYSTEMS
        .iter()
        .find(|s| s.name == name)
        .unwrap_or_else(|| panic!("unknown system {name}"))
}

pub fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("<root>/crates/ferric-ci")
        .to_path_buf()
}

pub struct Run {
    pub mol: Molecule,
    pub prep: PreparedBasis,
    pub rhf: ScfResult,
    pub space: DeterminantSpace,
    pub res: CasCiResult,
}

/// Exact-J/K RHF (the settings of `validation_casci.rs`) and CAS-CI with RDMs.
pub fn run_system(sys: &System) -> Run {
    let xyz = workspace_root()
        .join("testdata/molecules/validation")
        .join(sys.xyz);
    let mol = Molecule::load_xyz_with_charge(xyz.to_str().unwrap(), 0, 1)
        .unwrap_or_else(|e| panic!("{}: {e:?}", xyz.display()));
    let prep = PreparedBasis::new(&mol, &basis::bundled(sys.basis).unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let cfg = RhfConfig {
        max_iter: 300,
        energy_conv: 1e-11,
        density_conv: 1e-9,
        ..Default::default()
    };
    let rhf = solve_rhf(&ParallelContext::default(), &mol, &prep, op, &bounds, &cfg)
        .unwrap_or_else(|e| panic!("{}: RHF failed: {e:?}", sys.name));
    assert!(rhf.converged, "{}: RHF did not converge", sys.name);
    assert!(rhf.df_jk.is_none(), "{}: RHF must use exact J/K", sys.name);
    let ci_cfg = CasCiConfig {
        n_active: sys.n_active,
        n_elec_active: sys.n_elec,
        active_start: sys.active_start,
        conv_thresh: CONV_THRESH,
        compute_rdms: true,
        ..Default::default()
    };
    let res = run_cas_ci(&mol, &prep, &rhf, ci_cfg)
        .unwrap_or_else(|e| panic!("{}: CAS-CI failed: {e:?}", sys.name));
    assert!(res.converged, "{}: Davidson did not converge", sys.name);
    // The RDM bounds are derived from CONV_THRESH, so the root must meet it.
    assert!(
        res.residual_norm < CONV_THRESH,
        "{}: CI residual {:e} >= CONV_THRESH {CONV_THRESH:e}",
        sys.name,
        res.residual_norm
    );
    let space = DeterminantSpace {
        alpha_strings: enumerate_strings(sys.n_active, sys.n_elec.0),
        beta_strings: enumerate_strings(sys.n_active, sys.n_elec.1),
    };
    Run {
        mol,
        prep,
        rhf,
        space,
        res,
    }
}
