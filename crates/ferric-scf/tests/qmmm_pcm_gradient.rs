//! `qmmm_qm_energy_and_gradient` (the QM-region gradient inside
//! `optimize_qmmm`) routes through the SCF gradient dispatchers, so the
//! IEF-PCM term is included and COSMO is refused.
//!
//! Physics vs artifact hypothesis: if the PCM term is present, analytic - FD of
//! the solvated QM/MM SCF energy stays inside the step-derived bound; if the
//! bare vacuum-form `rhf_gradient` were called, every coordinate would miss by
//! the PCM gradient itself (asserted as a control, and by mutation).
//!
//! FD bound: central differences at `h` and `2h` differ by three times the
//! O(h^2) truncation error at `h`; the bar is that difference plus `4 dE/h`,
//! `dE` the SCF energy convergence.

use ferric_core::basis;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_pcm::PcmConfig;
use ferric_scf::gradient::rhf_gradient;
use ferric_scf::qmmm::{
    qmmm_qm_energy_and_gradient, BoundaryChargeScheme, QmSelection, QmmmAtom, QmmmMethod,
    QmmmSystem, DEFAULT_LINK_SCALE,
};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

const ANG2BOHR: f64 = 1.0 / 0.529_177_210_92;
const H: f64 = 1e-3;
const E_CONV: f64 = 1e-11;

/// Capped ethane, QM = one methyl (C0 + H2 H4 H6) + link H across C0-C1,
/// MM = the other methyl with RCD boundary charges.
fn system() -> QmmmSystem {
    let cc = 1.53 * ANG2BOHR;
    let ch = 1.09 * ANG2BOHR;
    let theta = 109.5_f64.to_radians();
    let (s, c) = (theta.sin(), theta.cos());
    let mut atoms = vec![
        QmmmAtom::new("C", 6, 0.0, 0.0, 0.0, -0.1),
        QmmmAtom::new("C", 6, 0.0, 0.0, cc, -0.1),
    ];
    for k in 0..3 {
        let phi = 2.0 * std::f64::consts::PI * (k as f64) / 3.0;
        let (x, y) = (ch * s * phi.cos(), ch * s * phi.sin());
        atoms.push(QmmmAtom::new("H", 1, x, y, ch * c, 0.033));
        atoms.push(QmmmAtom::new("H", 1, x, y, cc - ch * c, 0.033));
    }
    let bonds = vec![(0, 1), (0, 2), (0, 4), (0, 6), (1, 3), (1, 5), (1, 7)];
    QmmmSystem::new(&atoms, QmSelection::Indices(vec![0, 2, 4, 6]), 0, 1)
        .unwrap()
        .with_link_atoms(&bonds, DEFAULT_LINK_SCALE)
        .unwrap()
        .with_boundary_charges(&bonds, BoundaryChargeScheme::RedistributedChargeDipole)
        .unwrap()
}

fn scf_cfg(pcm: Option<PcmConfig>) -> RhfConfig {
    RhfConfig {
        max_iter: 300,
        energy_conv: E_CONV,
        density_conv: 1e-9,
        pcm,
        ..Default::default()
    }
}

/// SCF energy of the QM molecule `mol` in the (fixed) embedding of `sys`.
fn energy(sys: &QmmmSystem, mol: &ferric_core::mol::Molecule, cfg: &RhfConfig) -> f64 {
    let bs = basis::bundled("sto-3g").unwrap();
    let prep = PreparedBasis::new(mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let mut c = cfg.clone();
    c.external_potential = sys.to_external_potential();
    let r = solve_rhf(&ParallelContext::default(), mol, &prep, op, &bounds, &c).unwrap();
    assert!(r.converged);
    r.energy
}

fn displaced(
    mol: &ferric_core::mol::Molecule,
    a: usize,
    ax: usize,
    dx: f64,
) -> ferric_core::mol::Molecule {
    let mut m = mol.clone();
    match ax {
        0 => m.atoms[a].x += dx,
        1 => m.atoms[a].y += dx,
        _ => m.atoms[a].zpos += dx,
    }
    m
}

fn central(
    sys: &QmmmSystem,
    mol: &ferric_core::mol::Molecule,
    cfg: &RhfConfig,
    a: usize,
    ax: usize,
    h: f64,
) -> f64 {
    (energy(sys, &displaced(mol, a, ax, h), cfg) - energy(sys, &displaced(mol, a, ax, -h), cfg))
        / (2.0 * h)
}

#[test]
fn qmmm_pcm_qm_gradient_matches_fd() {
    let sys = system();
    let mol = sys.to_qm_molecule();
    let cfg = scf_cfg(Some(PcmConfig::water()));
    let bs = basis::bundled("sto-3g").unwrap();
    let (_, g, _) = qmmm_qm_energy_and_gradient(
        &ParallelContext::default(),
        &sys,
        &bs,
        &QmmmMethod::Rhf,
        &cfg,
    )
    .unwrap();

    // Vacuum-form control on the SAME solvated SCF.
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let mut c = cfg.clone();
    c.external_potential = sys.to_external_potential();
    let r = solve_rhf(&ParallelContext::default(), &mol, &prep, op, &bounds, &c).unwrap();
    let g_bare = rhf_gradient(&mol, &prep, op, &bounds, &r, c.external_potential.as_ref()).unwrap();

    // Every atom (link H included), z axis plus x on the first two atoms.
    let mut coords: Vec<(usize, usize)> = (0..mol.atoms.len()).map(|a| (a, 2)).collect();
    coords.extend([(0, 0), (1, 0)]);
    let mut worst_control = 0.0_f64;
    for (a, ax) in coords {
        let fd_h = central(&sys, &mol, &cfg, a, ax, H);
        let fd_2h = central(&sys, &mol, &cfg, a, ax, 2.0 * H);
        let tol = (fd_2h - fd_h).abs() + 4.0 * E_CONV / H;
        let err = (g[(a, ax)] - fd_h).abs();
        assert!(
            err < tol,
            "atom {a} axis {ax}: analytic {:.10e} FD {fd_h:.10e} |diff| {err:.3e} > tol {tol:.3e}",
            g[(a, ax)]
        );
        let miss = (g_bare[(a, ax)] - fd_h).abs();
        assert!(
            miss > 10.0 * tol,
            "control: bare gradient miss {miss:.3e} not > 10 tol {tol:.3e} at atom {a} axis {ax}"
        );
        worst_control = worst_control.max(miss);
    }
    eprintln!("QM/MM+PCM: bare-gradient control miss up to {worst_control:.3e} Ha/Bohr");
}

/// Without PCM the dispatcher route is exactly the bare `rhf_gradient`.
#[test]
fn qmmm_without_pcm_is_bit_identical_to_bare_rhf_gradient() {
    let sys = system();
    let mol = sys.to_qm_molecule();
    let cfg = scf_cfg(None);
    let bs = basis::bundled("sto-3g").unwrap();
    let (e, g, _) = qmmm_qm_energy_and_gradient(
        &ParallelContext::default(),
        &sys,
        &bs,
        &QmmmMethod::Rhf,
        &cfg,
    )
    .unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let mut c = cfg.clone();
    c.external_potential = sys.to_external_potential();
    let r = solve_rhf(&ParallelContext::default(), &mol, &prep, op, &bounds, &c).unwrap();
    let g0 = rhf_gradient(&mol, &prep, op, &bounds, &r, c.external_potential.as_ref()).unwrap();
    assert_eq!(e.to_bits(), r.energy.to_bits());
    assert_eq!(g, g0, "no-PCM QM/MM gradient must be bit-identical");
}

#[test]
fn qmmm_cosmo_gradient_is_refused() {
    let sys = system();
    let bs = basis::bundled("sto-3g").unwrap();
    let cfg = RhfConfig {
        cosmo: Some(ferric_scf::cosmo::CosmoConfig::default()),
        ..Default::default()
    };
    for m in [QmmmMethod::Rhf, QmmmMethod::Uhf] {
        let err = qmmm_qm_energy_and_gradient(&ParallelContext::default(), &sys, &bs, &m, &cfg)
            .unwrap_err();
        assert!(
            err.to_string().contains("COSMO has no analytic gradient"),
            "{err}"
        );
    }
}
