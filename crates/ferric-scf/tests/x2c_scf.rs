//! X2C-1e as an SCF reference (`RhfConfig::relativistic`).
//!
//! * Exactness anchor: `reference_hcore(None)` is the nonrelativistic core
//!   Hamiltonian bit for bit (the `None` SCF route is checked bitwise against
//!   the base build by the CLI comparison in the commit message).
//! * The driver route equals the block route: an SCF with
//!   `relativistic = X2c1e` against the same SCF with the constant Fock hook
//!   `V = h_X2C - h_NR`, `E_hook = tr(D V)`, for RHF, UHF, ROHF and UKS. The
//!   two differ by the rounding of `h_NR + V` and by the guess (the ROHF guess
//!   Fock reads `h`), so they agree to their convergence: each energy is
//!   within `energy_conv` of the stationary value, bar `2 energy_conv`.
//! * PySCF `sfx2c1e()` (`testdata/reference/x2c_scf_sfx2c1e.json`,
//!   `scripts/gen_pyscf_x2c_scf_refs.py`): HBr and H2Se RHF at cc-pVDZ and its
//!   uncontracted form. Bar: the two codes' energy convergence (`ENERGY_CONV`
//!   here, PySCF `conv_tol`) plus the nonrelativistic disagreement of the same
//!   two codes on the same system (`|E_NR(ferric) - E_NR(PySCF)|`, measured in
//!   the test), which carries the shared integral and convergence floors.
//! * Refusals: gradients (by config and by result), Hessians, frequencies,
//!   geometry optimization, injected (periodic) solves, ECP atoms, strict
//!   parsing; the certificate records X2C and is Ungraded.

use ferric_core::basis::{self, BasisSet};
use ferric_core::external_potential::{ExternalPotential, PointCharge};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::engine::libint_max_deriv_order;
use ferric_integrals::oneelectron;
use ferric_integrals::operator::Operator;
use ferric_scf::result::ScfResult;
use ferric_scf::rhf::{solve_rhf, solve_rhf_with_hook, RhfConfig};
use ferric_scf::rohf::{solve_rohf, solve_rohf_with_hook};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::{solve_uhf, solve_uhf_with_hook};
use ferric_scf::x2c::{
    reference_hcore, uncontracted_basis, x2c1e_hcore, Relativistic, LIGHT_SPEED,
};
use ndarray::Array2;

/// Energy convergence of every SCF here (Ha).
const ENERGY_CONV: f64 = 1e-10;

fn skip() -> bool {
    if libint_max_deriv_order() < 2 {
        eprintln!("SKIP: libint2 generated without second derivatives (no pVp)");
        return true;
    }
    false
}

fn mol(xyz_angstrom: &str, charge: i32, mult: usize) -> Molecule {
    let lines: Vec<&str> = xyz_angstrom.lines().collect();
    Molecule::parse_xyz(
        &format!("{}\n\n{}\n", lines.len(), xyz_angstrom),
        charge,
        mult,
    )
    .unwrap()
}

const HBR: &str = "H 0 0 0\nBr 0 0 1.414";

fn tight(xc: Option<&str>) -> RhfConfig {
    RhfConfig {
        energy_conv: ENERGY_CONV,
        density_conv: 1e-8,
        xc: xc.map(str::to_string),
        ..Default::default()
    }
}

fn x2c(cfg: &RhfConfig) -> RhfConfig {
    RhfConfig {
        relativistic: Some(Relativistic::X2c1e),
        ..cfg.clone()
    }
}

struct Sys {
    mol: Molecule,
    prep: PreparedBasis,
    bounds: SchwarzBounds,
}

fn sys(m: Molecule, bs: &BasisSet) -> Sys {
    let prep = PreparedBasis::new(&m, bs).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
    Sys {
        mol: m,
        prep,
        bounds,
    }
}

fn bits(a: &Array2<f64>) -> Vec<u64> {
    a.iter().map(|x| x.to_bits()).collect()
}

fn max_abs(a: &Array2<f64>) -> f64 {
    a.iter().fold(0.0, |m, x| m.max(x.abs()))
}

/// `h_X2C - h_NR` in the orbital basis of `s`.
fn x2c_shift(s: &Sys) -> Array2<f64> {
    let bs = s.prep.basis_set();
    x2c1e_hcore(&s.mol, bs, LIGHT_SPEED).unwrap()
        - oneelectron::hcore_ecp_with_external(&s.prep, &s.mol, bs, None).unwrap()
}

#[test]
fn none_reference_hcore_is_the_nonrelativistic_hcore_bitwise() {
    if skip() {
        return;
    }
    let bs = basis::bundled("cc-pvdz").unwrap();
    let s = sys(mol(HBR, 0, 1), &bs);
    let ext = ExternalPotential {
        point_charges: vec![PointCharge {
            q: -0.4,
            x: 1.0,
            y: 2.0,
            z: -3.0,
        }],
        field: Some([0.0, 0.0, 0.01]),
        ..Default::default()
    };
    for e in [None, Some(&ext)] {
        let nr = oneelectron::hcore_ecp_with_external(&s.prep, &s.mol, &bs, e).unwrap();
        let got = reference_hcore(&s.prep, &s.mol, &bs, e, None).unwrap();
        assert_eq!(bits(&got), bits(&nr), "ext = {}", e.is_some());
    }
    // X2C keeps the external potential, in the nonrelativistic picture:
    // h(X2C, ext) - h(X2C) = h(NR, ext) - h(NR) to the rounding of the
    // matrix entries (max|h| ~ 1e3 Ha here, 1e-14 relative per term).
    let rel = Some(Relativistic::X2c1e);
    let with = reference_hcore(&s.prep, &s.mol, &bs, Some(&ext), rel).unwrap();
    let without = reference_hcore(&s.prep, &s.mol, &bs, None, rel).unwrap();
    let nr_with = oneelectron::hcore_ecp_with_external(&s.prep, &s.mol, &bs, Some(&ext)).unwrap();
    let nr = oneelectron::hcore(&s.prep);
    let dev = max_abs(&(&(&with - &without) - &(&nr_with - &nr)));
    let bar = 1e-14 * 4.0 * (max_abs(&with) + max_abs(&nr_with));
    assert!(dev <= bar, "external term {dev:e} > {bar:e}");
    assert!(
        max_abs(&(&nr_with - &nr)) > 1e-3,
        "the external potential is not vacuous"
    );
}

/// `(E_hook, V)` of the constant shift `dh` for a closed-shell total density.
fn closed_hook(
    dh: &Array2<f64>,
) -> impl FnMut(&Array2<f64>) -> Result<(f64, Array2<f64>), ferric_core::FerricError> + '_ {
    move |d: &Array2<f64>| Ok(((d * dh).sum(), dh.clone()))
}

type SpinHookOut = Result<(f64, Array2<f64>, Array2<f64>), ferric_core::FerricError>;

fn spin_hook(dh: &Array2<f64>) -> impl FnMut(&Array2<f64>, &Array2<f64>) -> SpinHookOut + '_ {
    move |da: &Array2<f64>, db: &Array2<f64>| Ok((((da + db) * dh).sum(), dh.clone(), dh.clone()))
}

fn assert_same(label: &str, driver: &ScfResult, block: &ScfResult) {
    assert!(
        driver.converged && block.converged,
        "{label}: not converged"
    );
    assert_eq!(driver.relativistic, Some(Relativistic::X2c1e), "{label}");
    assert_eq!(block.relativistic, None, "{label}");
    let d = (driver.energy - block.energy).abs();
    eprintln!(
        "{label}: driver {:.12} block {:.12} |d| {d:.2e}",
        driver.energy, block.energy
    );
    assert!(
        d <= 2.0 * ENERGY_CONV,
        "{label}: driver vs block route {d:e}"
    );
}

#[test]
fn driver_matches_the_fock_hook_block_route() {
    if skip() {
        return;
    }
    let ctx = ParallelContext::default();
    let op = Operator::coulomb();
    let bs = basis::bundled("cc-pvdz").unwrap();

    let s = sys(mol(HBR, 0, 1), &bs);
    let dh = x2c_shift(&s);
    let cfg = tight(None);
    let driver = solve_rhf(&ctx, &s.mol, &s.prep, op, &s.bounds, &x2c(&cfg)).unwrap();
    let mut hook = closed_hook(&dh);
    let block = solve_rhf_with_hook(&ctx, &s.mol, &s.prep, op, &s.bounds, &cfg, &mut hook).unwrap();
    assert_same("RHF HBr", &driver, &block);
    // The shift is far above the bar, so the comparison is not vacuous.
    let nr = solve_rhf(&ctx, &s.mol, &s.prep, op, &s.bounds, &cfg).unwrap();
    assert!(
        (driver.energy - nr.energy).abs() > 1.0,
        "X2C shift is vacuous"
    );

    let s = sys(mol(HBR, 1, 2), &bs);
    let dh = x2c_shift(&s);
    for (label, cfg) in [
        ("UHF HBr+", tight(None)),
        ("UKS/PBE HBr+", tight(Some("PBE"))),
    ] {
        let driver = solve_uhf(&ctx, &s.mol, &s.prep, &s.bounds, &x2c(&cfg)).unwrap();
        let mut hook = spin_hook(&dh);
        let block = solve_uhf_with_hook(&ctx, &s.mol, &s.prep, &s.bounds, &cfg, &mut hook).unwrap();
        assert_same(label, &driver, &block);
    }
    let cfg = tight(None);
    let driver = solve_rohf(&ctx, &s.mol, &s.prep, op, &s.bounds, &x2c(&cfg)).unwrap();
    let mut hook = spin_hook(&dh);
    let block = solve_rohf_with_hook(&ctx, &s.mol, &s.prep, &s.bounds, &cfg, &mut hook).unwrap();
    assert_same("ROHF HBr+", &driver, &block);
}

fn reference() -> serde_json::Value {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../testdata/reference/x2c_scf_sfx2c1e.json"
    );
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

#[test]
fn rhf_matches_pyscf_sfx2c1e_contracted_and_uncontracted() {
    if skip() {
        return;
    }
    let refs = reference();
    let pyscf_conv = refs["conv_tol"].as_f64().unwrap();
    let ctx = ParallelContext::default();
    let op = Operator::coulomb();
    let contracted = basis::bundled("cc-pvdz").unwrap();
    let uncontracted = uncontracted_basis(&contracted).unwrap();
    for label in ["HBr", "H2Se"] {
        let r = &refs["systems"][label];
        let m = mol(r["xyz_angstrom"].as_str().unwrap(), 0, 1);
        for (bname, bs) in [("cc-pvdz", &contracted), ("unc-cc-pvdz", &uncontracted)] {
            let p = &r["bases"][bname];
            let s = sys(m.clone(), bs);
            assert_eq!(
                s.prep.nbasis() as u64,
                p["nao"].as_u64().unwrap(),
                "{label} {bname}"
            );
            let cfg = tight(None);
            let nr = solve_rhf(&ctx, &s.mol, &s.prep, op, &s.bounds, &cfg).unwrap();
            let rel = solve_rhf(&ctx, &s.mol, &s.prep, op, &s.bounds, &x2c(&cfg)).unwrap();
            assert!(nr.converged && rel.converged, "{label} {bname}");
            let d_nr = (nr.energy - p["e_nr"].as_f64().unwrap()).abs();
            let d_x2c = (rel.energy - p["e_x2c"].as_f64().unwrap()).abs();
            let bar = 2.0 * (ENERGY_CONV + pyscf_conv) + d_nr;
            eprintln!(
                "{label} {bname}: nbf {} E_x2c {:.10} dE(x2c) {d_x2c:.2e} dE(nr) {d_nr:.2e} \
                 bar {bar:.2e} shift {:.6}",
                s.prep.nbasis(),
                rel.energy,
                rel.energy - nr.energy
            );
            assert!(
                d_x2c <= bar,
                "{label} {bname}: |E_x2c - PySCF| {d_x2c:e} > {bar:e}"
            );
        }
    }
}

fn has<T: std::fmt::Debug>(r: Result<T, ferric_core::FerricError>, needle: &str, what: &str) {
    match r {
        Ok(v) => panic!("{what}: not refused ({v:?})"),
        Err(e) => assert!(e.to_string().contains(needle), "{what}: {e}"),
    }
}

const NO_DERIV: &str = "relativistic = \"x2c1e\" has no";

#[test]
fn gradients_hessians_and_geometry_tasks_refuse_x2c() {
    use ferric_scf::frequencies::{harmonic_frequencies, FrequencyConfig, HessianMethod};
    use ferric_scf::gradient::{
        preflight_cosx_restricted, preflight_cosx_restricted_open, preflight_cosx_unrestricted,
        refuse_relativistic_gradient, restricted_scf_gradient, rhf_gradient,
    };
    use ferric_scf::hessian::{rhf_hessian, rhf_hessian_preflight};
    use ferric_scf::optimize::{optimize_geometry, OptimizeConfig};
    if skip() {
        return;
    }
    let ctx = ParallelContext::default();
    let op = Operator::coulomb();
    let rel = x2c(&RhfConfig::default());
    let vac = RhfConfig::default();
    refuse_relativistic_gradient(&vac).expect("nonrelativistic config passes");
    has(refuse_relativistic_gradient(&rel), NO_DERIV, "helper");
    has(
        preflight_cosx_restricted(&rel),
        NO_DERIV,
        "restricted preflight",
    );
    has(
        preflight_cosx_unrestricted(&rel),
        NO_DERIV,
        "unrestricted preflight",
    );
    has(
        preflight_cosx_restricted_open(&rel),
        NO_DERIV,
        "restricted-open preflight",
    );
    has(
        rhf_hessian_preflight(op, &rel),
        "relativistic",
        "Hessian preflight",
    );
    preflight_cosx_restricted(&vac).unwrap();
    rhf_hessian_preflight(op, &vac).unwrap();

    let bs = basis::bundled("sto-3g").unwrap();
    let s = sys(mol("H 0 0 0\nF 0 0 0.92", 0, 1), &bs);
    let res = solve_rhf(&ctx, &s.mol, &s.prep, op, &s.bounds, &rel).unwrap();
    assert_eq!(res.relativistic, Some(Relativistic::X2c1e));
    // The config guard of the dispatcher, then the result guard of the
    // gradient itself (an X2C result handed in with a nonrelativistic config).
    has(
        restricted_scf_gradient(&s.mol, &s.prep, &bs, op, &s.bounds, &rel, &res),
        NO_DERIV,
        "dispatcher, X2C config",
    );
    has(
        restricted_scf_gradient(&s.mol, &s.prep, &bs, op, &s.bounds, &vac, &res),
        NO_DERIV,
        "dispatcher, X2C result",
    );
    has(
        rhf_gradient(&s.mol, &s.prep, op, &s.bounds, &res, None),
        NO_DERIV,
        "rhf_gradient",
    );
    has(
        rhf_hessian(&ctx, &s.mol, &s.prep, op, &s.bounds, &res, &vac),
        NO_DERIV,
        "rhf_hessian, X2C result",
    );
    // Anchor: the nonrelativistic result differentiates.
    let nr = solve_rhf(&ctx, &s.mol, &s.prep, op, &s.bounds, &vac).unwrap();
    rhf_gradient(&s.mol, &s.prep, op, &s.bounds, &nr, None).expect("NR gradient");

    for hessian in [
        HessianMethod::Auto,
        HessianMethod::Analytic,
        HessianMethod::FiniteDifference,
    ] {
        let fcfg = FrequencyConfig {
            hessian,
            ..Default::default()
        };
        has(
            harmonic_frequencies(&ctx, &s.mol, "sto-3g", op, &rel, &fcfg),
            NO_DERIV,
            &format!("harmonic_frequencies {hessian:?}"),
        );
    }
    has(
        optimize_geometry(&ctx, &s.mol, "sto-3g", op, &rel, &OptimizeConfig::default()),
        NO_DERIV,
        "optimize_geometry",
    );
}

#[test]
fn injected_solves_ecp_atoms_and_unknown_spellings_are_refused() {
    // An injected solve takes the hcore guess (the SAD guess is refused
    // there for its own reason), so the base config is one it accepts.
    let base = RhfConfig {
        use_sad_guess: false,
        ..Default::default()
    };
    ferric_scf::rhf::validate_injected(&base).unwrap();
    let rel = x2c(&base);
    let e = ferric_scf::rhf::validate_injected(&rel).unwrap_err();
    assert_eq!(e.field, "relativistic", "{e}");

    assert_eq!(
        Relativistic::parse_config_str("x2c1e").unwrap(),
        Relativistic::X2c1e
    );
    for bad in ["X2C1E", "x2c", "sfx2c1e", ""] {
        assert!(
            Relativistic::parse_config_str(bad).is_err(),
            "{bad:?} accepted"
        );
    }
    if skip() {
        return;
    }
    // def2-SVP carries an ECP for I.
    let ctx = ParallelContext::default();
    let bs = basis::bundled("def2-svp").unwrap();
    let mut m = mol("H 0 0 0\nI 0 0 1.61", 0, 1);
    m.apply_ecp(&bs);
    let s = sys(m, &bs);
    has(
        solve_rhf(&ctx, &s.mol, &s.prep, Operator::coulomb(), &s.bounds, &rel),
        "carries an ECP",
        "ECP atom",
    );
}

#[test]
fn certificate_records_x2c_and_is_ungraded() {
    use ferric_core::certificate::Grade;
    if skip() {
        return;
    }
    let ctx = ParallelContext::default();
    let op = Operator::coulomb();
    let bs = basis::bundled("sto-3g").unwrap();
    let s = sys(mol("H 0 0 0\nF 0 0 0.92", 0, 1), &bs);
    let rel = x2c(&RhfConfig::default());
    let res = solve_rhf(&ctx, &s.mol, &s.prep, op, &s.bounds, &rel).unwrap();
    let c = ferric_scf::certificate::scf_certificate(&s.mol, &res, &rel);
    assert!(
        c.approximations
            .iter()
            .any(|a| a.name == "x2c1e" && a.detail.contains("X2C-1e")),
        "{c}"
    );
    assert_eq!(c.grade, Grade::Ungraded, "{c}");
    // Anchor: the nonrelativistic certificate carries no X2C record and its
    // grade is not capped by it.
    let vac = RhfConfig::default();
    let nr = solve_rhf(&ctx, &s.mol, &s.prep, op, &s.bounds, &vac).unwrap();
    let c = ferric_scf::certificate::scf_certificate(&s.mol, &nr, &vac);
    assert!(c.approximations.iter().all(|a| a.name != "x2c1e"), "{c}");
    assert_ne!(c.grade, Grade::Ungraded, "{c}");
}
