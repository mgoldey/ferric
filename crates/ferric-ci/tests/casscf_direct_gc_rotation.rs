//! The direct CASSCF integral route ([`CasScfIntegrals::Direct`]) follows
//! `CasScfConfig::gc_rotation`, the contraction rotation of the direct J/K
//! builds (`ferric_scf::direct_jk::gc_rotation`).
//!
//! * `Some(Off)` is the unrotated build, and bitwise equal to an unset
//!   config under `FERRIC_GC_ROTATE=off` (a child process of this binary
//!   with the variable set; no test mutates its own environment).
//! * `Some(Auto)` rotates (the bits differ) and agrees with `Off` at one
//!   orbital set (water / cc-pVDZ, CAS(4,4), the RHF orbitals): energy
//!   within 1e-11 Ha, orbital gradient and `F^I`, `F^A` within 1e-10. The
//!   measured differences are a few 1e-13 (printed).
//! * An unset config with the variable unset is `Auto`.

mod casscf_common;

use casscf_common::molecule_text;
use ferric_ci::casscf::{CasScfConfig, CasScfIntegrals, CasScfProblem};
use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::direct_jk::GcRotate;
use ferric_scf::incore_jk::EriMode;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ndarray::Array2;
use std::hash::{Hash, Hasher};

/// Set in the environment of the child runs.
const CHILD: &str = "FERRIC_TEST_GC_ROTATION_CHILD";

struct Point {
    e: f64,
    g: Vec<f64>,
    fi: Array2<f64>,
    fa: Array2<f64>,
}

/// The direct-route CASSCF point at the RHF orbitals, `gc_rotation = gc`.
/// The RHF itself runs unrotated, so every run starts from the same bits.
fn point(gc: Option<GcRotate>) -> Point {
    let mol = Molecule::parse_xyz(&molecule_text("water.xyz"), 0, 1).unwrap();
    let prep = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let mut rc = RhfConfig::default()
        .with_energy_conv(1e-12)
        .with_density_conv(1e-10)
        .with_gc_rotation(GcRotate::Off);
    rc.eri = EriMode::Direct;
    let rhf = solve_rhf(&ParallelContext::default(), &mol, &prep, op, &bounds, &rc).unwrap();
    assert!(rhf.converged);
    let c = rhf.mos_r().clone();
    let cfg = CasScfConfig {
        n_active: 4,
        n_elec_active: (2, 2),
        n_core: 3,
        ci_conv_thresh: 1e-12,
        integrals: CasScfIntegrals::Direct { thresh: 1e-12 },
        gc_rotation: gc,
        ..Default::default()
    };
    let p = CasScfProblem::new(&mol, &prep, c.ncols(), &cfg).unwrap();
    let pt = p.evaluate(&c).unwrap();
    Point {
        e: pt.e_total,
        g: pt.gradient.clone(),
        fi: pt.f_inactive.clone(),
        fa: pt.f_active.clone(),
    }
}

fn bits(p: &Point) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    p.e.to_bits().hash(&mut h);
    for v in p.g.iter().chain(p.fi.iter()).chain(p.fa.iter()) {
        v.to_bits().hash(&mut h);
    }
    h.finish()
}

fn maxdiff<'a>(a: impl Iterator<Item = &'a f64>, b: impl Iterator<Item = &'a f64>) -> f64 {
    // NaN-safe: a NaN difference is +inf.
    a.zip(b).fold(0.0f64, |m, (x, y)| {
        let d = (x - y).abs();
        if d.is_nan() {
            f64::INFINITY
        } else {
            m.max(d)
        }
    })
}

/// The bits of an unset-config run in a child process with
/// `FERRIC_GC_ROTATE` set to `env` (`None`: removed).
fn child_bits(env: Option<&str>) -> u64 {
    let mut cmd = std::process::Command::new(std::env::current_exe().unwrap());
    cmd.args(["child_prints_unset_config_bits", "--exact", "--nocapture"])
        .env(CHILD, "1");
    match env {
        Some(v) => cmd.env("FERRIC_GC_ROTATE", v),
        None => cmd.env_remove("FERRIC_GC_ROTATE"),
    };
    let out = cmd.output().expect("spawn the child");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "child failed: {stdout}");
    let line = stdout
        .lines()
        .find_map(|l| l.strip_prefix("GC_BITS "))
        .unwrap_or_else(|| panic!("no GC_BITS line: {stdout}"));
    u64::from_str_radix(line.trim(), 16).unwrap()
}

#[test]
fn direct_route_follows_the_config_rotation() {
    let off = point(Some(GcRotate::Off));
    let auto = point(Some(GcRotate::Auto));
    let de = (auto.e - off.e).abs();
    let dg = maxdiff(auto.g.iter(), off.g.iter());
    let df = maxdiff(auto.fi.iter(), off.fi.iter()).max(maxdiff(auto.fa.iter(), off.fa.iter()));
    eprintln!("CASSCF direct auto vs off: |dE| {de:.2e}, max|dg| {dg:.2e}, max|dF| {df:.2e}");
    assert!(de <= 1e-11, "|dE| = {de:e}");
    assert!(dg <= 1e-10, "max|dg| = {dg:e}");
    assert!(df <= 1e-10, "max|dF| = {df:e}");
    assert_ne!(bits(&auto), bits(&off), "the auto run did not rotate");
    assert_eq!(
        child_bits(Some("off")),
        bits(&off),
        "config off is not the unset config under FERRIC_GC_ROTATE=off"
    );
    assert_eq!(
        child_bits(None),
        bits(&auto),
        "an unset config with the variable unset is not auto"
    );
}

/// Child entry point of `direct_route_follows_the_config_rotation`: an
/// unset-config run, its bits printed. A no-op outside the child.
#[test]
fn child_prints_unset_config_bits() {
    if std::env::var_os(CHILD).is_none() {
        return;
    }
    println!("GC_BITS {:016x}", bits(&point(None)));
}
