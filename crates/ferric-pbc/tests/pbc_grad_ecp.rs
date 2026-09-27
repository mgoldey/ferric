// LANL2DZ exponents such as 0.318 are basis data, not approximations of 1/pi.
#![allow(clippy::approx_constant)]
//! Gamma-point forces with a periodic ECP (`ferric_pbc::ecp::periodic_ecp_gradient`
//! + its wiring into `ferric_pbc::grad`): the Rust port of
//! `reference/pbc/pbc_grad_ecp.py` / `run_grad_ecp_anchor.py` (FINDINGS
//! "Iteration 22" and its 2026-09-27 addendum).
//!
//! System: HI, H STO-3G (PySCF digits), I LANL2DZ basis + LANL2DZ ECP (Z_eff
//! 7), the Iteration 14 fixtures of `pbc_ecp.rs` (full basis, or the compact
//! I s 0.4653 + p 0.318 variant). Anchor cell diag(6, 6, 7) with the atoms
//! OFF the common axis, H (0.3, 0.2, 0.4), I (0.9, −0.4, 3.3), so no force
//! component vanishes by symmetry.
//!
//! What is anchored against what:
//! * TRIVIAL LIMIT (fast): in a 40-Bohr box only L = M = 0 survive, so the
//!   periodic ECP force term IS the molecular `ferric_scf::gradient::
//!   ecp_gradient` (libecpint `compute_first_derivs`, a different code path
//!   for the centre term) at the same density. The image mutants `L0Only` /
//!   `M0Only` are INVISIBLE there by construction (printed, asserted as the
//!   blind spot) — which is why the small cell below exists.
//! * TERM ANCHOR (fast): the ECP term at FIXED density vs central FD of
//!   `E_ECP(R) = Σ D V_ECP(R)` (ferric's own `periodic_ecp_images`, screen
//!   recomputed at every displaced geometry: triples flipping in/out of the
//!   1e-14 screen are ≤ ~precision/h in the FD). Prototype raw h = 1e-4
//!   floor 1.5e-9..1.8e-9 (h² truncation). First Rust run at h = 1e-4:
//!   1.37e-7 (atom 0 z) / 3.1e-8 (atom 1 x) — plausibly libecpint's
//!   geometry-dependent radial-quadrature jitter (the molecular
//!   `ecp_matrix_deriv.rs` measured an h-independent 1.4e-7 plateau on
//!   I2/def2-SVP). The test now prints an h ladder (1e-3 .. 3e-5, plus
//!   Richardson) that separates jitter (∝ 1/h) from truncation (∝ h²) and
//!   asserts the best estimate against 1e-7, PROVISIONAL.
//! * WIRING (fast): the RHF force's `parts.ecp` IS `periodic_ecp_gradient`
//!   at the SCF density; its triple count is the hcore's; the UHF path on
//!   the same closed-shell density gives the same force (all six Gamma
//!   entry points share `grad::assemble`); stress refuses an ECP cell; a
//!   hcore built at another ECP precision is refused.
//! * SCF FD ANCHOR (slow, #[ignore]): analytic RHF force vs central FD (h =
//!   1e-4) of ferric's OWN Gamma energy, both exxdiv, every displaced SCF
//!   seeded from the reference density. Prototype (Python, frozen image
//!   sets, pure AFT): 1.49e-9 none / 1.50e-9 ewald. The Rust SR attraction
//!   derivative carries libint2's p-shell floor at the 1e9 gradient nucleus
//!   exponent (6.4e-9 on the H s+p cell; Z = 7 here) and the ECP term
//!   libecpint's jitter. MEASURED 2026-09-27 (compact HI): 2.52e-7 for both
//!   exxdiv → bar 5e-7; ΣF 1.9e-11 (all of it the dense-AFT `eri` part at
//!   precision 1e-10; ECP part 3e-16) → bar 1e-9. Mutants, measured (compact;
//!   prototype full-LANL2DZ values in brackets): `EcpNoCentre` 0.495 [3.1e-1]
//!   and `EcpCentreSign` 0.989 [6.3e-1] (ΣF sees both); `EcpL0Only` 2.5e-2
//!   [1.6e-2] and `EcpM0Only` 2.4e-2 [4.2e-3] (ΣF BLIND — only this FD
//!   anchor catches them). Each must miss by > 10x the bar; ΣF visibility
//!   is asserted on the ECP part. The full-LANL2DZ case is not yet measured.
//!
//! Not pinned to PySCF: 2.13's `ECPscalar_ipnuc` is 1.1e-7 off on off-centre
//! elements and its periodic `ecp_int` 3.8e-6 off (FINDINGS 14, 22).

mod common;

use common::{gamma_config, max_abs_diff};
use ferric_core::basis::{BasisSet, Shell};
use ferric_core::ecp::{EcpDef, EcpShell, EcpTerm};
use ferric_core::mol::{Atom, Molecule};
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_pbc::dense_aft::{DenseAftEri, ExxDiv, DEFAULT_DENSE_AFT_MAX_BYTES};
use ferric_pbc::ecp::{
    periodic_ecp_gradient, periodic_ecp_gradient_with, periodic_ecp_images, EcpGradMutation,
    PeriodicEcpConfig,
};
use ferric_pbc::grad::{
    gamma_rhf_gradient_with, gamma_uhf_gradient_with, GammaGradConfig, GammaGradient, GradMutation,
};
use ferric_pbc::hcore::{periodic_hcore, PeriodicHcore, PeriodicHcoreConfig};
use ferric_pbc::lattice::Cell;
use ferric_pbc::stress::{gamma_rhf_stress, GammaStressConfig};
use ferric_scf::result::{ScfResult, Spin};
use ferric_scf::rhf::{solve_rhf_injected, PeriodicInjection, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ndarray::Array2;
use std::collections::HashMap;

const OMEGA: f64 = 0.8;
/// hcore (and so ECP) truncation: the production default, stated.
const HCORE_PRECISION: f64 = 1e-14;
/// Dense-AFT precision. The FD anchor alone needs only a CONSISTENT energy
/// (the G sphere does not move with the atoms), but at 1e-6 the TOTAL RHF
/// force was not translation invariant (first run, 2026-09-27: ΣF 2.62e-7,
/// while the ECP part alone sums to ~1e-14) — the suspected source is the
/// AFT pair screen; the per-part ΣF is printed to confirm. 1e-10 as the
/// k-point gradient anchors needed; the ΣF bar is NOT loosened.
const AFT_PRECISION: f64 = 1e-10;
const HI_A: [[f64; 3]; 3] = [[6.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 7.0]];
/// `run_grad_ecp_anchor.py` MOVED (Bohr).
const MOVED: [[f64; 3]; 2] = [[0.3, 0.2, 0.4], [0.9, -0.4, 3.3]];
const FD_H: f64 = 1e-4;
const TERM_FD_BAR: f64 = 1e-7;
/// Measured 2026-09-27 (compact HI, release): 2.52e-7 for both exxdiv.
const SCF_FD_BAR: f64 = 5e-7;
/// ΣF of the TOTAL force. Measured 1.9e-11 at AFT precision 1e-10; the
/// per-part printout put all of it in `eri` (dense-AFT pair screening), the
/// ECP part at 3e-16.
const NET_FORCE_BAR: f64 = 1e-9;

// ------------------------------------------------------------ fixtures
// (pbc_ecp.rs's Iteration-14 fixtures; test files cannot share them.)

fn shell(l: i32, exps: &[f64], coefs: &[f64]) -> Shell {
    let lf = l as f64;
    let mut s = 0.0;
    for (a, ca) in exps.iter().zip(coefs) {
        for (b, cb) in exps.iter().zip(coefs) {
            s += ca * cb * (2.0 * (a * b).sqrt() / (a + b)).powf(lf + 1.5);
        }
    }
    Shell {
        l,
        pure: false,
        exponents: exps.to_vec(),
        coefficients: coefs.iter().map(|c| c / s.sqrt()).collect(),
    }
}

fn lanl2dz_i_ecp() -> EcpDef {
    let ch = |l: i32, t: &[(i32, f64, f64)]| EcpShell {
        angular_momentum: l,
        terms: t
            .iter()
            .map(|&(n, z, d)| EcpTerm {
                coef: d,
                r_exp: n,
                gexp: z,
            })
            .collect(),
    };
    EcpDef {
        n_core: 46,
        shells: vec![
            ch(
                3,
                &[
                    (0, 1.0715702, -0.0747621),
                    (1, 44.1936028, -30.0811224),
                    (2, 12.9367609, -75.3722721),
                    (2, 3.1956412, -22.0563758),
                    (2, 0.8589806, -1.6979585),
                ],
            ),
            ch(
                0,
                &[
                    (0, 127.9202670, 2.9380036),
                    (1, 78.6211465, 41.2471267),
                    (2, 36.5146237, 287.8680095),
                    (2, 9.9065681, 114.3758506),
                    (2, 1.9420086, 37.6547714),
                ],
            ),
            ch(
                1,
                &[
                    (0, 13.0035304, 2.2222630),
                    (1, 76.0331404, 39.4090831),
                    (2, 24.1961684, 177.4075002),
                    (2, 6.4053433, 77.9889462),
                    (2, 1.5851786, 25.7547641),
                ],
            ),
            ch(
                2,
                &[
                    (0, 40.4278108, 7.0524360),
                    (1, 28.9084375, 33.3041635),
                    (2, 15.6268936, 186.9453875),
                    (2, 4.1442856, 71.9688361),
                    (2, 0.9377235, 9.3630657),
                ],
            ),
        ],
    }
}

/// H STO-3G + I LANL2DZ (`full`) or the compact s+p I basis, LANL2DZ ECP on I.
fn hi_basis(full: bool) -> BasisSet {
    let mut shells = HashMap::new();
    shells.insert(
        1,
        vec![shell(
            0,
            &[3.42525091, 0.62391373, 0.1688554],
            &[0.15432897, 0.53532814, 0.44463454],
        )],
    );
    let i_shells = if full {
        vec![
            shell(0, &[0.7242, 0.4653], &[-2.9731048, 3.4827643]),
            shell(0, &[0.1336], &[1.0]),
            shell(1, &[1.29, 0.318], &[-0.2092377, 1.1035347]),
            shell(1, &[0.1053], &[1.0]),
        ]
    } else {
        vec![shell(0, &[0.4653], &[1.0]), shell(1, &[0.318], &[1.0])]
    };
    shells.insert(53, i_shells);
    let mut ecps = HashMap::new();
    ecps.insert(53, lanl2dz_i_ecp());
    BasisSet {
        name: if full { "HI-lanl2dz" } else { "HI-compact" }.into(),
        shells,
        ecps,
    }
}

fn atom(symbol: &str, z: i32, r: [f64; 3]) -> Atom {
    Atom {
        symbol: symbol.into(),
        z,
        x: r[0],
        y: r[1],
        zpos: r[2],
        ghost: false,
        n_core_ecp: 0,
    }
}

/// H at pos[0], I at pos[1], through `apply_ecp(bs)`.
fn hi_cell(pos: &[[f64; 3]], lattice: [[f64; 3]; 3], bs: &BasisSet) -> Cell {
    let mut mol = Molecule {
        atoms: vec![atom("H", 1, pos[0]), atom("I", 53, pos[1])],
        charge: 0,
        multiplicity: 1,
    };
    mol.apply_ecp(bs);
    Cell::new(mol, lattice).expect("HI cell")
}

fn prep(cell: &Cell, bs: &BasisSet) -> PreparedBasis {
    PreparedBasis::new(cell.mol(), bs).expect("prep")
}

fn moved(pos: &[[f64; 3]], a: usize, x: usize, h: f64) -> Vec<[f64; 3]> {
    let mut p = pos.to_vec();
    p[a][x] += h;
    p
}

fn hcore_cfg() -> PeriodicHcoreConfig {
    PeriodicHcoreConfig {
        precision: HCORE_PRECISION,
        ..PeriodicHcoreConfig::with_omega(OMEGA)
    }
}

/// The ECP settings `periodic_hcore` uses at [`hcore_cfg`] (same precision;
/// the budget does not enter the screen).
fn ecp_cfg() -> PeriodicEcpConfig {
    PeriodicEcpConfig::with_precision(HCORE_PRECISION)
}

/// A fixed symmetric, non-trivial "density" for the term-level anchors.
fn sym_density(n: usize) -> Array2<f64> {
    Array2::from_shape_fn((n, n), |(i, j)| {
        let (a, b) = (i as f64, j as f64);
        0.3 * (1.1 * a + 0.7 * b).cos()
            + 0.3 * (1.1 * b + 0.7 * a).cos()
            + if i == j { 0.5 } else { 0.0 }
    })
}

fn amax(a: &Array2<f64>) -> f64 {
    a.iter().fold(0.0_f64, |m, x| m.max(x.abs()))
}

fn net(a: &Array2<f64>) -> f64 {
    (0..3)
        .map(|c| a.column(c).sum().abs())
        .fold(0.0_f64, f64::max)
}

/// `|ΣF|` of every non-zero part of a Gamma force, for locating a
/// translation-invariance leak.
fn parts_net(g: &GammaGradient) -> String {
    let p = &g.parts;
    [
        ("overlap", &p.overlap),
        ("kinetic", &p.kinetic),
        ("ecp", &p.ecp),
        ("vsr_basis", &p.vsr_basis),
        ("vsr_nuc", &p.vsr_nuc),
        ("vsr_basis+nuc", &(&p.vsr_basis + &p.vsr_nuc)),
        ("vlr_basis", &p.vlr_basis),
        ("vlr_nuc", &p.vlr_nuc),
        ("vlr_basis+nuc", &(&p.vlr_basis + &p.vlr_nuc)),
        ("eri", &p.eri),
        ("nn_sr", &p.nn_sr),
        ("nn_lr", &p.nn_lr),
    ]
    .iter()
    .map(|(name, a)| format!("{name} {:.1e}", net(a)))
    .collect::<Vec<_>>()
    .join(", ")
}

// ------------------------------------------------------------ SCF drivers

fn build(cell: &Cell, bs: &BasisSet) -> (PreparedBasis, PeriodicHcore, DenseAftEri) {
    let p = prep(cell, bs);
    let hc = periodic_hcore(cell, &p, &hcore_cfg()).expect("hcore");
    assert!(hc.n_ecp_triples > 0, "no ECP triples: the cell is vacuous");
    let eri = DenseAftEri::build(
        cell,
        &p,
        &hc.s,
        ExxDiv::None,
        AFT_PRECISION,
        DEFAULT_DENSE_AFT_MAX_BYTES,
    )
    .expect("dense AFT");
    (p, hc, eri)
}

fn rhf(
    cell: &Cell,
    p: &PreparedBasis,
    hc: &PeriodicHcore,
    eri: &DenseAftEri,
    seed: Option<&Array2<f64>>,
) -> ScfResult {
    let ctx = ParallelContext::default();
    let op = Operator::coulomb();
    // Never read on the injected path; required by the signature.
    let bounds = SchwarzBounds::compute(op, p).expect("schwarz");
    let cfg = RhfConfig {
        init_guess_density: seed.cloned(),
        max_iter: 300,
        ..gamma_config()
    };
    let inj = PeriodicInjection {
        s: hc.s.clone(),
        h: hc.h.clone(),
        vnn: hc.enn,
        j: Box::new(eri.j_builder()),
        k: Box::new(eri.k_builder()),
        xc: None,
    };
    let r = solve_rhf_injected(&ctx, cell.mol(), p, op, &bounds, &cfg, inj).expect("gamma RHF");
    assert!(r.converged, "gamma RHF did not converge");
    r
}

fn grad(
    cell: &Cell,
    p: &PreparedBasis,
    hc: &PeriodicHcore,
    eri: &DenseAftEri,
    scf: &ScfResult,
    exx: ExxDiv,
    mutation: Option<GradMutation>,
) -> GammaGradient {
    let cfg = GammaGradConfig {
        mutation,
        ..Default::default()
    };
    gamma_rhf_gradient_with(cell, p, &hcore_cfg(), hc, eri, scf, exx, &cfg).expect("gamma gradient")
}

/// The same closed-shell density as an unrestricted result `(D/2, D/2)`.
fn as_unrestricted(r: &ScfResult) -> ScfResult {
    let mut u = r.clone();
    let half = &r.density_total * 0.5;
    u.spin = Spin::Unrestricted;
    u.density_alpha = half.clone();
    u.density_beta = Some(half);
    u.mos_beta = Some(r.mos_alpha.clone());
    u.eps_beta = Some(r.eps_alpha.clone());
    u.fock_beta = Some(r.fock_alpha.clone());
    u
}

// ============================================================ trivial limit

/// 40-Bohr box: only L = M = 0 survive, so the periodic ECP force term IS
/// the molecular libecpint gradient (`compute_first_derivs`, per-atom
/// centre attribution by libecpint) at the same density. The centre mutants
/// are visible here (and to ΣF); the image mutants are NOT (blind spot,
/// asserted), hence the small-cell anchors.
#[test]
fn big_box_ecp_force_term_is_the_molecular_ecp_gradient() {
    let bs = hi_basis(false);
    let a = 40.0;
    let shift = [0.37, 0.21, 0.5 * a - 1.52];
    let mol_pos = [[0.0, 0.0, 0.0], [0.4, -0.3, 3.0]];
    let pos: Vec<[f64; 3]> = mol_pos
        .iter()
        .map(|r| [r[0] + shift[0], r[1] + shift[1], r[2] + shift[2]])
        .collect();
    let cell = hi_cell(&pos, [[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]], &bs);
    let p = prep(&cell, &bs);
    let d = sym_density(p.nbasis());
    let g = periodic_ecp_gradient(&cell, &p, &ecp_cfg(), &d)
        .expect("periodic ECP gradient")
        .expect("ECP cell");
    let gm = ferric_scf::gradient::ecp_gradient(cell.mol(), &p, &d).expect("molecular");
    let diff = max_abs_diff(&g.grad, &gm);
    eprintln!(
        "40-Bohr box: |periodic − molecular ECP gradient| {diff:.2e} (max {:.3e}), ΣF {:.1e}, \
         triples {}\n{:.10}",
        amax(&gm),
        net(&g.grad),
        g.n_triples,
        g.grad
    );
    assert!(amax(&gm) > 1e-3, "vacuous: molecular ECP gradient ~0");
    assert!(diff < 1e-10, "{diff:e}");
    assert!(net(&g.grad) < 1e-12, "ΣF {:e}", net(&g.grad));
    for (m, visible) in [
        (EcpGradMutation::NoCentre, true),
        (EcpGradMutation::CentreSign, true),
        (EcpGradMutation::L0Only, false),
        (EcpGradMutation::M0Only, false),
    ] {
        let gm_ = periodic_ecp_gradient_with(&cell, &p, &ecp_cfg(), &d, Some(m))
            .unwrap()
            .unwrap();
        let dm = max_abs_diff(&gm_.grad, &g.grad);
        eprintln!(
            "  box mutant {m:?}: |Δ| {dm:.2e}, ΣF {:.1e}",
            net(&gm_.grad)
        );
        if visible {
            assert!(dm > 1e-3, "{m:?} invisible in the box: {dm:e}");
            assert!(net(&gm_.grad) > 1e-3, "{m:?}: ΣF should see it");
        } else {
            // Blind spot, by construction: no image survives in the box.
            assert!(
                dm < 1e-10,
                "{m:?} visible in the box ({dm:e}): images leaked in"
            );
        }
    }
}

// ============================================================ term anchor

/// ECP term at a FIXED density vs central FD of `Σ D V_ECP(R)` (ferric's
/// own lattice sum) on the 6×6×7 anchor cell, compact basis. Two
/// components (one per atom) keep it fast; the SCF anchor covers all six.
#[test]
fn ecp_term_matches_fd_of_its_energy_at_fixed_density() {
    let bs = hi_basis(false);
    let cell = hi_cell(&MOVED, HI_A, &bs);
    let p = prep(&cell, &bs);
    let d = sym_density(p.nbasis());
    let g = periodic_ecp_gradient(&cell, &p, &ecp_cfg(), &d)
        .unwrap()
        .unwrap();
    let e_ecp = |pos: &[[f64; 3]]| -> f64 {
        let c = hi_cell(pos, HI_A, &bs);
        let pc = prep(&c, &bs);
        let v = periodic_ecp_images(&c, &pc, &ecp_cfg())
            .unwrap()
            .unwrap()
            .gamma();
        (&d * &v).sum()
    };
    // Step ladder: a smooth-function residual is TRUNCATION (∝ h², falls
    // 11x per ladder step) at large h; value-kernel JITTER (libecpint's
    // geometry-dependent radial grid, and the deriv = 1 vs deriv = 0
    // engine's grid set-up) is h-independent in E, so ∝ 1/h in the FD and
    // RISES as h shrinks. First run (h = 1e-4 only): 1.37e-7 on atom 0 z,
    // 3.1e-8 on atom 1 x. Richardson on (1e-3, 5e-4) removes the h² part at
    // steps where jitter/h is smallest. The asserted error is the BEST
    // estimate of the ladder; all are printed — set the bar from them.
    let ladder = [1e-3, 3e-4, 1e-4, 3e-5];
    let fd_at = |a: usize, x: usize, h: f64| {
        (e_ecp(&moved(&MOVED, a, x, h)) - e_ecp(&moved(&MOVED, a, x, -h))) / (2.0 * h)
    };
    let mut worst = 0.0_f64;
    for (a, x) in [(0usize, 2usize), (1, 0)] {
        let an = g.grad[(a, x)];
        let errs: Vec<(f64, f64)> = ladder
            .iter()
            .map(|&h| (h, (an - fd_at(a, x, h)).abs()))
            .collect();
        let rich = (4.0 * fd_at(a, x, 5e-4) - fd_at(a, x, 1e-3)) / 3.0;
        let err_rich = (an - rich).abs();
        let best = errs.iter().map(|e| e.1).fold(err_rich, f64::min);
        eprintln!(
            "ECP term atom {a} x{x}: analytic {an:.12e} (bra {:.3e} ket {:.3e} centre {:.3e})\n  \
             |an − FD(h)|: {}\n  |an − Richardson(1e-3, 5e-4)| {err_rich:.2e}; best {best:.2e}",
            g.bra[(a, x)],
            g.ket[(a, x)],
            g.centre[(a, x)],
            errs.iter()
                .map(|(h, e)| format!("h {h:.0e}: {e:.2e}"))
                .collect::<Vec<_>>()
                .join(", ")
        );
        assert!(an.abs() > 1e-6, "vacuous component ({a}, {x})");
        worst = worst.max(best);
    }
    eprintln!(
        "ECP term: max over components of the best |analytic − FD| {worst:.2e}, ΣF {:.1e}, \
         triples {} ({} calls)",
        net(&g.grad),
        g.n_triples,
        g.n_calls
    );
    // PROVISIONAL (TERM_FD_BAR 1e-7): on the best ladder estimate.
    assert!(worst < TERM_FD_BAR, "{worst:e}");
    assert!(net(&g.grad) < 1e-12, "ΣF {:e}", net(&g.grad));
    // All three centres move: none of the parts is idle in this cell.
    for (name, part) in [("bra", &g.bra), ("ket", &g.ket), ("centre", &g.centre)] {
        assert!(amax(part) > 1e-4, "{name} part ~0");
    }
}

// ============================================================ wiring

/// The RHF force carries exactly `periodic_ecp_gradient` at the SCF density
/// over the hcore's own triples; the UHF path on the same density gives the
/// same force; the stress refuses the ECP cell; a hcore built at another ECP
/// precision is refused.
#[test]
fn gamma_forces_carry_the_ecp_term_and_stress_refuses_it() {
    let bs = hi_basis(false);
    let cell = hi_cell(&MOVED, HI_A, &bs);
    let (p, hc, eri) = build(&cell, &bs);
    let scf = rhf(&cell, &p, &hc, &eri, None);
    let g = grad(&cell, &p, &hc, &eri, &scf, ExxDiv::None, None);
    let e = periodic_ecp_gradient(&cell, &p, &ecp_cfg(), &scf.density_total)
        .unwrap()
        .unwrap();
    let d = max_abs_diff(&g.parts.ecp, &e.grad);
    eprintln!(
        "RHF force: parts.ecp vs periodic_ecp_gradient {d:.1e}; ECP part max {:.3e}; triples \
         {} (hcore {}); ΣF {:.1e} (AFT precision {AFT_PRECISION:.0e}); per part: {}\n{:.10}",
        amax(&g.parts.ecp),
        g.n_ecp_triples,
        hc.n_ecp_triples,
        g.net_force,
        parts_net(&g),
        g.grad
    );
    assert_eq!(
        d, 0.0,
        "the force's ECP part is not periodic_ecp_gradient: {d:e}"
    );
    assert!(amax(&g.parts.ecp) > 1e-3, "ECP part ~0");
    assert_eq!(g.n_ecp_triples, hc.n_ecp_triples);
    assert_eq!(e.n_triples, hc.n_ecp_triples);
    assert!(
        net(&g.parts.ecp) < 1e-12,
        "ECP part ΣF {:e}",
        net(&g.parts.ecp)
    );
    assert!(
        g.net_force < NET_FORCE_BAR,
        "ΣF {:e}; per part: {}",
        g.net_force,
        parts_net(&g)
    );

    // UHF entry point on the same closed-shell density.
    let u = gamma_uhf_gradient_with(
        &cell,
        &p,
        &hcore_cfg(),
        &hc,
        &eri,
        &as_unrestricted(&scf),
        ExxDiv::None,
        &GammaGradConfig::default(),
    )
    .expect("UHF gradient");
    let du = max_abs_diff(&u.grad, &g.grad);
    eprintln!("UHF(D/2, D/2) vs RHF force: {du:.1e}");
    assert!(du < 1e-10, "{du:e}");
    assert!(max_abs_diff(&u.parts.ecp, &g.parts.ecp) < 1e-14);

    // Stress with an ECP: refused (no strain derivative of V_ECP).
    let st = gamma_rhf_stress(
        &cell,
        &p,
        &hcore_cfg(),
        &hc,
        &eri,
        &scf,
        ExxDiv::None,
        &GammaStressConfig::default(),
    );
    let msg = st.expect_err("stress must refuse an ECP cell").to_string();
    assert!(msg.contains("ECP"), "{msg}");

    // A hcore whose V_ECP came from another screen: refused, not mixed.
    let other = PeriodicHcoreConfig {
        precision: 1e-8,
        ..hcore_cfg()
    };
    let err = gamma_rhf_gradient_with(
        &cell,
        &p,
        &other,
        &hc,
        &eri,
        &scf,
        ExxDiv::None,
        &GammaGradConfig::default(),
    )
    .expect_err("mismatched ECP screen must error")
    .to_string();
    assert!(err.contains("ECP force screen"), "{err}");
}

// ============================================================ SCF FD anchor

type Fd = Vec<((usize, usize), [f64; 2])>;

/// Gamma RHF energies `[none, ewald]` at `pos`, seeded from `seed`.
fn energies(pos: &[[f64; 3]], bs: &BasisSet, seed: &Array2<f64>) -> [f64; 2] {
    let cell = hi_cell(pos, HI_A, bs);
    let (p, hc, base) = build(&cell, bs);
    let mut out = [0.0; 2];
    for (k, exx) in [ExxDiv::None, ExxDiv::Ewald].into_iter().enumerate() {
        let eri = base.clone().with_exxdiv(&cell, exx).unwrap();
        out[k] = rhf(&cell, &p, &hc, &eri, Some(seed)).energy;
    }
    out
}

fn max_fd_err(g: &Array2<f64>, fdv: &Fd, k: usize) -> f64 {
    fdv.iter()
        .map(|((a, x), v)| (g[(*a, *x)] - v[k]).abs())
        .fold(0.0_f64, f64::max)
}

/// The anchor on the 6×6×7 cell with the given basis (`full` = the
/// prototype's LANL2DZ system). Asserts the FD bar both exxdiv, ΣF,
/// F(ewald) = F(none), and that each ECP mutant misses the anchor by
/// > 10x the bar with the predicted ΣF (in)visibility.
fn scf_fd_anchor(full: bool) {
    let label = if full { "HI/LANL2DZ" } else { "HI/compact" };
    let bs = hi_basis(full);
    let cell = hi_cell(&MOVED, HI_A, &bs);
    let (p, hc, base) = build(&cell, &bs);
    let eri_n = base.clone().with_exxdiv(&cell, ExxDiv::None).unwrap();
    let eri_e = base.with_exxdiv(&cell, ExxDiv::Ewald).unwrap();
    let scf_n = rhf(&cell, &p, &hc, &eri_n, None);
    let seed = scf_n.density_total.clone();
    let scf_e = rhf(&cell, &p, &hc, &eri_e, Some(&seed));
    let g = [
        grad(&cell, &p, &hc, &eri_n, &scf_n, ExxDiv::None, None),
        grad(&cell, &p, &hc, &eri_e, &scf_e, ExxDiv::Ewald, None),
    ];
    let comps = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)];
    let fdv: Fd = comps
        .iter()
        .map(|&(a, x)| {
            let ep = energies(&moved(&MOVED, a, x, FD_H), &bs, &seed);
            let em = energies(&moved(&MOVED, a, x, -FD_H), &bs, &seed);
            (
                (a, x),
                [
                    (ep[0] - em[0]) / (2.0 * FD_H),
                    (ep[1] - em[1]) / (2.0 * FD_H),
                ],
            )
        })
        .collect();
    for (k, gg) in g.iter().enumerate() {
        let err = max_fd_err(&gg.grad, &fdv, k);
        eprintln!(
            "{label} exxdiv {}: max|analytic − FD| = {err:.2e}, ΣF {:.1e}, comm {:.1e}, \
             ECP part max {:.3e}, ECP triples {}; ΣF per part: {}\n{:.10}",
            ["none", "ewald"][k],
            gg.net_force,
            gg.commutator,
            amax(&gg.parts.ecp),
            gg.n_ecp_triples,
            parts_net(gg),
            gg.grad
        );
        assert!(err < SCF_FD_BAR, "{label} exxdiv {k}: {err:e}");
        assert!(
            gg.net_force < NET_FORCE_BAR,
            "ΣF {:e}; per part: {}",
            gg.net_force,
            parts_net(gg)
        );
    }
    let dex = max_abs_diff(&g[0].grad, &g[1].grad);
    eprintln!("{label}: |F(ewald) − F(none)| {dex:.1e}");
    assert!(dex < 1e-9, "{dex:e}");

    for (m, sigma_f_sees_it) in [
        (GradMutation::EcpNoCentre, true),
        (GradMutation::EcpCentreSign, true),
        (GradMutation::EcpL0Only, false),
        (GradMutation::EcpM0Only, false),
    ] {
        let gm = grad(&cell, &p, &hc, &eri_n, &scf_n, ExxDiv::None, Some(m));
        let err = max_fd_err(&gm.grad, &fdv, 0);
        // ΣF visibility is judged on the ECP PART (the mutants touch only
        // it), so the other parts' translation-invariance floor cannot mask
        // or fake it.
        let nm = net(&gm.parts.ecp);
        eprintln!(
            "{label} mutant {m:?}: max|analytic − FD| = {err:.2e}, |ΣF| total {:.1e}, \
             ECP part {nm:.1e}",
            gm.net_force
        );
        assert!(
            err > 10.0 * SCF_FD_BAR,
            "{m:?} escaped the FD anchor: {err:e}"
        );
        if sigma_f_sees_it {
            assert!(nm > 1e-3, "{m:?}: ΣF should see it ({nm:e})");
        } else {
            // Predicted blind spot: every kept triple stays translation
            // invariant, so only the FD anchor can catch these.
            assert!(nm < 1e-12, "{m:?}: ΣF unexpectedly sees it ({nm:e})");
        }
    }
}

#[test]
#[ignore = "slow: compact HI 6x6x7, 12 displaced periodic hcore (ECP lattice sum) + dense-AFT builds, 26 SCFs, 6 gradients; run in release with --ignored, serially"]
fn hi_compact_rhf_force_matches_fd_of_own_energy_and_catches_ecp_mutants() {
    scf_fd_anchor(false);
}

#[test]
#[ignore = "slow: full HI/LANL2DZ 6x6x7 (prototype system; ~14 s per ECP lattice-sum build in release), 13 hcore builds, 26 SCFs, 6 gradients; run in release with --ignored, serially"]
fn hi_lanl2dz_rhf_force_matches_fd_of_own_energy_and_catches_ecp_mutants() {
    scf_fd_anchor(true);
}
