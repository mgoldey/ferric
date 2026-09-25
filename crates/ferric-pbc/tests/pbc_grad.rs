//! Gamma-point periodic RHF analytic forces (`ferric_pbc::grad`), the Rust
//! port of `reference/pbc/pbc_grad.py` (FINDINGS "Iteration 16").
//!
//! What is anchored against what:
//! * EXACTNESS ANCHOR: analytic vs central finite difference (h = 1e-4) of
//!   ferric's OWN Gamma energy (`periodic_hcore` + `DenseAftEri` +
//!   `solve_rhf_injected`), both `exxdiv`. The G spheres do not move with
//!   the atoms, so the only FD floor is `h² f'''/6` (prototype: 2.4e-9 on
//!   H₂, 3.3e-9 on the triclinic cell) plus the primitive screens at the
//!   default 1e-14 precision. Bar 1e-7 Ha/Bohr. The Rust SR images are
//!   derived from `precision` (not the prototype's fixed SR cutoffs, which
//!   left 5.6e-7..2.9e-6 on STO-3G), and the dense-AFT ERI has no SR part.
//! * `exxdiv = ewald` and `none` forces equal to 1e-12 (holds only with the
//!   `−(v_M/2) DSD` term in `M`).
//! * Ewald `E_nn` gradient vs PySCF `pbc.grad.krhf.grad_nuc` pins (computed
//!   2026-09-24 with the prototype's PySCF 2.13, `cart=True`, precision 1e-12)
//!   and vs FD of `ewald_nuclear_repulsion`.
//! * Integral-level identities: shifted derivative blocks at zero shift are
//!   bitwise the unshifted ones; ket blocks vs FD of the shifted energy
//!   integrals; 3-centre translation invariance; pair-FT derivative
//!   `Q_mn + Q_nm = −iG P_mn` and vs FD of `pair_ft`.
//!
//! Mutations (`GammaGradConfig::mutation`, a hidden knob, EXECUTED in
//! `h2_fd_anchor_catches_each_mutant`; prototype magnitudes on H₂ pure AFT):
//! * `VneNoBasis` (drop basis-centre motion in V_ne): 2.1e-1
//! * `WSign` (+W): 2.4e-2 (none)
//! * `NoEwaldLr` (drop the Ewald LR derivative): 2.4e-1
//! * `NoMadelungS` (drop −(v_M/2) DSD, exxdiv = ewald): 6.1e-2
//!
//! Each must exceed the FD floor by orders; the test asserts > 1e-3.
//! The sum of forces is BLIND to all four (prototype: ΣF ≤ 1e-15 for every
//! mutant) — it is kept only as a sanity assert on the real gradient.
//!
//! Integral-level mutations (by hand, not knobs; PREDICTED, NOT YET RUN):
//! swapping the bra/ket blocks in `add_pair_deriv` should fail the anchor on
//! the moved-atom triclinic cell; dropping the `− i F[i−1]` term of the
//! pair-FT derivative should fail `pair_ft_derivative_identities` (p shells,
//! both the `Q + Qᵀ = −iGP` identity and the FD check).

mod common;

use common::*;
use ferric_core::basis::BasisSet;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::engine::Engine;
use ferric_integrals::ffi;
use ferric_integrals::operator::Operator;
use ferric_integrals::site_basis::SiteBasis;
use ferric_pbc::dense_aft::{
    DenseAftEri, ExxDiv, DEFAULT_DENSE_AFT_MAX_BYTES, DEFAULT_DENSE_AFT_PRECISION,
};
use ferric_pbc::ewald::{ewald_nuclear_gradient, ewald_nuclear_repulsion, DEFAULT_EWALD_PRECISION};
use ferric_pbc::grad::{gamma_rhf_gradient_with, GammaGradConfig, GammaRhfGradient, GradMutation};
use ferric_pbc::hcore::{periodic_hcore, PeriodicHcore, PeriodicHcoreConfig};
use ferric_pbc::lattice::Cell;
use ferric_pbc::pair_ft::{pair_ft_deriv, pair_ft_with_thresh};
use ndarray::Array2;
use num_complex::Complex64;
use std::sync::OnceLock;

/// ω of the nuclear-attraction split: not 1, so a 1/ω vs 1/ω² slip in the
/// G = 0 term (c0) is visible.
const OMEGA: f64 = 0.8;
/// hcore truncation (the default, stated explicitly: the SR image sets and
/// the SR screen are derived from it).
const HCORE_PRECISION: f64 = 1e-14;
const FD_H: f64 = 1e-4;
const FD_BAR: f64 = 1e-7;

/// `test_prototype.py` GRAD_H2_ATOMS (Bohr): off-axis so every Cartesian
/// component is non-trivial.
const GRAD_H2_ATOMS: [[f64; 3]; 2] = [[0.3, 0.2, 0.1], [0.35, 0.12, 1.5]];
const GRAD_COMPS: [(usize, usize); 3] = [(0, 0), (0, 2), (1, 1)];
/// `run_grad_oracle.py` TRI_MOVED (Bohr).
const TRI_MOVED: [[f64; 3]; 4] = [
    [0.13, 0.25, 0.31],
    [0.02, 0.27, 1.66],
    [2.47, 2.41, 2.25],
    [3.52, 2.98, 2.71],
];
const TRI_COMPS: [(usize, usize); 4] = [(0, 0), (1, 2), (2, 1), (3, 0)];

/// PySCF 2.13 `pbc.grad.krhf.grad_nuc` (Hartree/Bohr).
const H2_GRAD_NUC: [[f64; 3]; 2] = [
    [
        1.685386359805511e-02,
        -2.695954703521714e-02,
        3.798615889971087e-01,
    ],
    [
        -1.685386359805510e-02,
        2.695954703521714e-02,
        -3.798615889971086e-01,
    ],
];
const TRI_MOVED_GRAD_NUC: [[f64; 3]; 4] = [
    [
        -3.639564784001081e-02,
        -1.201925628299193e-03,
        5.128752303644706e-01,
    ],
    [
        3.222712644079274e-02,
        -3.698830776099783e-02,
        -3.962273112353235e-01,
    ],
    [
        4.062916476945971e-01,
        2.587515728403524e-01,
        1.493228667620570e-01,
    ],
    [
        -4.021231262953789e-01,
        -2.205613394510553e-01,
        -2.659707858912042e-01,
    ],
];
/// PySCF `Cell.energy_nuc()` for the same cells (the energy the gradient
/// differentiates).
const H2_ENUC: f64 = -0.6296103246662934;
const TRI_MOVED_ENUC: f64 = -1.4485937832181088;

fn hcore_cfg() -> PeriodicHcoreConfig {
    PeriodicHcoreConfig {
        precision: HCORE_PRECISION,
        ..PeriodicHcoreConfig::with_omega(OMEGA)
    }
}

fn cell_at(pos: &[[f64; 3]], lattice: [[f64; 3]; 3]) -> Cell {
    Cell::new(hydrogens(pos), lattice).expect("cell")
}

fn moved(pos: &[[f64; 3]], a: usize, x: usize, h: f64) -> Vec<[f64; 3]> {
    let mut p = pos.to_vec();
    p[a][x] += h;
    p
}

/// hcore + the exxdiv-none dense tensor (the tensor does not depend on
/// exxdiv; `with_exxdiv` switches the K builder's v_M).
fn build(cell: &Cell, bs: &BasisSet) -> (PreparedBasis, PeriodicHcore, DenseAftEri) {
    let prep = prep_for(cell, bs);
    let hc = periodic_hcore(cell, &prep, &hcore_cfg()).expect("hcore");
    let eri = DenseAftEri::build(
        cell,
        &prep,
        &hc.s,
        ExxDiv::None,
        DEFAULT_DENSE_AFT_PRECISION,
        DEFAULT_DENSE_AFT_MAX_BYTES,
    )
    .expect("dense AFT");
    (prep, hc, eri)
}

/// Gamma RHF energies `[none, ewald]` at one geometry.
fn energies(cell: &Cell, bs: &BasisSet) -> [f64; 2] {
    let (prep, hc, base) = build(cell, bs);
    let mut out = [0.0; 2];
    for (k, exx) in [ExxDiv::None, ExxDiv::Ewald].into_iter().enumerate() {
        let eri = base.clone().with_exxdiv(cell, exx).unwrap();
        out[k] = gamma_rhf(cell, &prep, &hc, &eri).energy;
    }
    out
}

/// Analytic gradients for `exxdivs` (one SCF per exxdiv, one build).
fn analytic(
    cell: &Cell,
    bs: &BasisSet,
    runs: &[(ExxDiv, Option<GradMutation>)],
) -> Vec<GammaRhfGradient> {
    let (prep, hc, base) = build(cell, bs);
    runs.iter()
        .map(|&(exx, mutation)| {
            let eri = base.clone().with_exxdiv(cell, exx).unwrap();
            let scf = gamma_rhf(cell, &prep, &hc, &eri);
            let cfg = GammaGradConfig {
                mutation,
                ..Default::default()
            };
            gamma_rhf_gradient_with(cell, &prep, &hcore_cfg(), &hc, &eri, &scf, exx, &cfg)
                .expect("gamma gradient")
        })
        .collect()
}

/// Central FD `[none, ewald]` of the energy for each component.
fn fd(
    pos: &[[f64; 3]],
    lattice: [[f64; 3]; 3],
    bs: &BasisSet,
    comps: &[(usize, usize)],
) -> Vec<((usize, usize), [f64; 2])> {
    comps
        .iter()
        .map(|&(a, x)| {
            let ep = energies(&cell_at(&moved(pos, a, x, FD_H), lattice), bs);
            let em = energies(&cell_at(&moved(pos, a, x, -FD_H), lattice), bs);
            (
                (a, x),
                [
                    (ep[0] - em[0]) / (2.0 * FD_H),
                    (ep[1] - em[1]) / (2.0 * FD_H),
                ],
            )
        })
        .collect()
}

static H2_FD: OnceLock<Vec<((usize, usize), [f64; 2])>> = OnceLock::new();

fn h2_fd() -> &'static Vec<((usize, usize), [f64; 2])> {
    H2_FD.get_or_init(|| fd(&GRAD_H2_ATOMS, cubic(4.0), &pyscf_sto3g_h(), &GRAD_COMPS))
}

fn max_fd_err(g: &Array2<f64>, fdv: &[((usize, usize), [f64; 2])], k: usize) -> f64 {
    fdv.iter()
        .map(|((a, x), v)| (g[(*a, *x)] - v[k]).abs())
        .fold(0.0_f64, f64::max)
}

fn max_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

// ------------------------------------------------------------ integral level

#[test]
fn shifted_1e_deriv_blocks_match_unshifted_and_fd() {
    let cell = triclinic_cell();
    let prep = prep_for(&cell, &sp_basis_h());
    let nsh = prep.nshells();
    let dims = prep.shell_dims().to_vec();
    let shift = [1.3, -0.4, 0.7];
    let h = 1e-5;
    let mut worst_fd = 0.0_f64;
    for op in [ffi::OP_OVERLAP, ffi::OP_KINETIC] {
        let mut eng = Engine::new_1e_deriv(op, &prep, 1e-16).unwrap();
        let mut e1 = Engine::new_1e(op, &prep, 1e-16).unwrap();
        for s1 in 0..nsh {
            for s2 in 0..nsh {
                let n = dims[s1] * dims[s2];
                let plain = eng
                    .compute_1e_deriv_block(&prep, s1, s2)
                    .map(|b| b.to_vec());
                let zero = eng
                    .compute_1e_deriv_block_shifted(&prep, s1, s2, [0.0; 3])
                    .unwrap()
                    .map(|b| b.to_vec());
                assert_eq!(plain, zero, "op {op} ({s1},{s2}): zero shift not bitwise");
                let blk = eng
                    .compute_1e_deriv_block_shifted(&prep, s1, s2, shift)
                    .unwrap()
                    .map(|b| b.to_vec())
                    .unwrap_or_else(|| vec![0.0; 6 * n]);
                assert_eq!(blk.len(), 6 * n);
                // Ket block = d/d(shift) of <s1 | op | s2(r − shift)>.
                for c in 0..3 {
                    let mut sp = shift;
                    sp[c] += h;
                    let mut sm = shift;
                    sm[c] -= h;
                    let ep = e1
                        .compute_1e_block_shifted(&prep, s1, s2, sp)
                        .unwrap()
                        .to_vec();
                    let em = e1
                        .compute_1e_block_shifted(&prep, s1, s2, sm)
                        .unwrap()
                        .to_vec();
                    for i in 0..n {
                        let f = (ep[i] - em[i]) / (2.0 * h);
                        let ket = blk[(3 + c) * n + i];
                        worst_fd = worst_fd.max((ket - f).abs());
                    }
                    // Translation invariance: bra + ket = 0 per element.
                    for i in 0..n {
                        let t = blk[c * n + i] + blk[(3 + c) * n + i];
                        assert!(t.abs() < 1e-12, "op {op}: bra+ket {t:e}");
                    }
                }
            }
        }
    }
    eprintln!("shifted 1e deriv ket block vs FD: {worst_fd:.2e}");
    assert!(worst_fd < 1e-8, "{worst_fd:e}");

    // A nuclear engine writes 3(2 + natoms) blocks: the capacity check must
    // turn that into an error, never an overrun.
    let mut nuc = Engine::new_1e_deriv(ffi::OP_NUCLEAR, &prep, 1e-16).unwrap();
    nuc.set_point_charges(&prep).unwrap();
    assert!(nuc
        .compute_1e_deriv_block_shifted(&prep, 0, 0, shift)
        .is_err());
}

#[test]
fn shifted_eri3_deriv_matches_unshifted_translation_and_fd() {
    let cell = triclinic_cell();
    let prep = prep_for(&cell, &sp_basis_h());
    let site = SiteBasis::new(&[[0.4, -0.3, 0.9, 5.0]], 0).unwrap();
    let op = Operator::erfc(OMEGA);
    let mut d = Engine::new_3center_deriv(op, &prep, &site.prep, f64::MIN_POSITIVE).unwrap();
    let mut e = Engine::new_3center(op, &prep, &site.prep, f64::MIN_POSITIVE).unwrap();
    let dims = prep.shell_dims().to_vec();
    let shifts = [[0.5, 0.1, -0.2], [0.0; 3], [1.3, -0.4, 0.7]];
    let h = 1e-5;
    let mut worst_fd = 0.0_f64;
    for s1 in 0..prep.nshells() {
        for s2 in 0..prep.nshells() {
            let n = dims[s1] * dims[s2];
            let plain = d
                .compute_eri3_deriv(&prep, &site.prep, 0, s1, s2)
                .map(|b| b.to_vec());
            let zero = d
                .compute_eri3_deriv_shifted(&prep, &site.prep, 0, s1, s2, [[0.0; 3]; 3])
                .unwrap()
                .map(|b| b.to_vec());
            assert_eq!(plain, zero, "({s1},{s2}): zero shifts not bitwise");
            let blk = d
                .compute_eri3_deriv_shifted(&prep, &site.prep, 0, s1, s2, shifts)
                .unwrap()
                .expect("not screened")
                .to_vec();
            for c in 0..3 {
                for i in 0..n {
                    let t = blk[c * n + i] + blk[(3 + c) * n + i] + blk[(6 + c) * n + i];
                    assert!(t.abs() < 1e-10, "3c translation invariance {t:e}");
                }
                let mut sp = shifts;
                sp[2][c] += h;
                let mut sm = shifts;
                sm[2][c] -= h;
                let ep = e
                    .compute_eri3_shifted(&prep, &site.prep, 0, s1, s2, sp)
                    .unwrap()
                    .map(|b| b.to_vec())
                    .unwrap_or_else(|| vec![0.0; n]);
                let em = e
                    .compute_eri3_shifted(&prep, &site.prep, 0, s1, s2, sm)
                    .unwrap()
                    .map(|b| b.to_vec())
                    .unwrap_or_else(|| vec![0.0; n]);
                for i in 0..n {
                    let f = (ep[i] - em[i]) / (2.0 * h);
                    worst_fd = worst_fd.max((blk[(6 + c) * n + i] - f).abs());
                }
            }
        }
    }
    eprintln!("shifted eri3 deriv sh2 block vs FD: {worst_fd:.2e}");
    assert!(worst_fd < 1e-8, "{worst_fd:e}");
}

#[test]
fn pair_ft_derivative_identities() {
    let cell = triclinic_cell();
    let bs = sp_basis_h();
    let prep = prep_for(&cell, &bs);
    let gs: Vec<[f64; 3]> = cell
        .gvectors(3.5)
        .unwrap()
        .into_iter()
        .filter(|g| g.iter().any(|v| *v != 0.0))
        .take(14)
        .collect();
    assert!(gs.len() >= 10);
    let thresh = 1e-15;
    let (p, q) = pair_ft_deriv(&cell, &prep, &gs, thresh).unwrap();
    let pref = pair_ft_with_thresh(&cell, &prep, &gs, thresh).unwrap();
    let n = prep.nbasis();
    let mut dp = 0.0_f64;
    let mut dsym = 0.0_f64;
    for m in 0..n {
        for k in 0..n {
            for (g, gv) in gs.iter().enumerate() {
                dp = dp.max((p[[m, k, g]] - pref[[m, k, g]]).norm());
                for c in 0..3 {
                    // Q_mk + Q_km = −iG P_mk
                    let lhs = q[c][[m, k, g]] + q[c][[k, m, g]];
                    let rhs = Complex64::new(0.0, -gv[c]) * p[[m, k, g]];
                    dsym = dsym.max((lhs - rhs).norm());
                }
            }
        }
    }
    eprintln!("pair_ft_deriv: |P - pair_ft| {dp:.2e}, |Q + Q^T + iGP| {dsym:.2e}");
    assert!(dp < 1e-13, "{dp:e}");
    assert!(dsym < 1e-12, "{dsym:e}");

    // Move atom 2 (all images of its functions) along y: FD of pair_ft vs
    // δ_{m∈A} Q_mn + δ_{n∈A} Q_nm.
    let (a, x, h) = (2usize, 1usize, 1e-5);
    let pos = cell.positions();
    let fd_p = |sgn: f64| {
        let c = cell_at(&moved(&pos, a, x, sgn * h), *cell.lattice());
        let pr = prep_for(&c, &bs);
        pair_ft_with_thresh(&c, &pr, &gs, thresh).unwrap()
    };
    let (pp, pm) = (fd_p(1.0), fd_p(-1.0));
    let sh2at = prep.shell_to_atom();
    let mut aoat = vec![0usize; n];
    for (sh, &at) in sh2at.iter().enumerate() {
        for k in 0..prep.shell_dims()[sh] {
            aoat[prep.shell_offsets()[sh] + k] = at;
        }
    }
    let (mut worst, mut fmax) = (0.0_f64, 0.0_f64);
    for m in 0..n {
        for k in 0..n {
            for g in 0..gs.len() {
                let f = (pp[[m, k, g]] - pm[[m, k, g]]) / (2.0 * h);
                let mut an = Complex64::new(0.0, 0.0);
                if aoat[m] == a {
                    an += q[x][[m, k, g]];
                }
                if aoat[k] == a {
                    an += q[x][[k, m, g]];
                }
                worst = worst.max((an - f).norm());
                fmax = fmax.max(f.norm());
            }
        }
    }
    eprintln!("pair_ft_deriv vs FD (atom {a}, dir {x}): {worst:.2e} (|FD| max {fmax:.2e})");
    assert!(fmax > 1e-3, "the moved atom must change P");
    assert!(worst < 1e-7 * fmax.max(1.0), "{worst:e}");
}

// ------------------------------------------------------------------ Ewald

#[test]
fn ewald_gradient_matches_pyscf_grad_nuc_and_fd() {
    for (pos, lattice, pin, enuc) in [
        (&GRAD_H2_ATOMS[..], cubic(4.0), &H2_GRAD_NUC[..], H2_ENUC),
        (
            &TRI_MOVED[..],
            TRI_A,
            &TRI_MOVED_GRAD_NUC[..],
            TRI_MOVED_ENUC,
        ),
    ] {
        let cell = cell_at(pos, lattice);
        let e = ewald_nuclear_repulsion(&cell, 1.0).unwrap();
        assert!((e - enuc).abs() < 1e-10, "E_nn {e} vs PySCF {enuc}");
        let g = ewald_nuclear_gradient(&cell, 1.0).unwrap();
        let g2 = ewald_nuclear_gradient(&cell, 2.3).unwrap();
        let mut worst = 0.0_f64;
        let mut w_omega = 0.0_f64;
        for a in 0..pos.len() {
            for c in 0..3 {
                worst = worst.max((g[a][c] - pin[a][c]).abs());
                w_omega = w_omega.max((g[a][c] - g2[a][c]).abs());
            }
        }
        let sum: f64 = (0..3)
            .map(|c| g.iter().map(|r| r[c]).sum::<f64>().abs())
            .fold(0.0, f64::max);
        // FD of the energy, one component per atom.
        let h = 1e-5;
        let mut worst_fd = 0.0_f64;
        for a in 0..pos.len() {
            let x = a % 3;
            let ep = ewald_nuclear_repulsion(&cell_at(&moved(pos, a, x, h), lattice), 1.0).unwrap();
            let em =
                ewald_nuclear_repulsion(&cell_at(&moved(pos, a, x, -h), lattice), 1.0).unwrap();
            worst_fd = worst_fd.max((g[a][x] - (ep - em) / (2.0 * h)).abs());
        }
        eprintln!(
            "Ewald gradient: vs PySCF {worst:.2e}, omega 1 vs 2.3 {w_omega:.2e}, \
             vs FD {worst_fd:.2e}, |sum| {sum:.2e} (precision {DEFAULT_EWALD_PRECISION:e})"
        );
        assert!(worst < 1e-10, "{worst:e}");
        assert!(w_omega < 1e-11, "{w_omega:e}");
        assert!(worst_fd < 1e-8, "{worst_fd:e}");
        assert!(sum < 1e-12, "{sum:e}");
    }
}

// --------------------------------------------------------- total force

#[test]
fn h2_force_matches_fd_of_own_energy_both_exxdiv() {
    let cell = cell_at(&GRAD_H2_ATOMS, cubic(4.0));
    let bs = pyscf_sto3g_h();
    let r = analytic(&cell, &bs, &[(ExxDiv::None, None), (ExxDiv::Ewald, None)]);
    let fdv = h2_fd();
    for (k, rr) in r.iter().enumerate() {
        let err = max_fd_err(&rr.grad, fdv, k);
        eprintln!(
            "H2/STO-3G a=4 exxdiv {}: max|analytic − FD| = {err:.2e}, |ΣF| {:.1e}, \
             comm {:.1e}, SR triplets {}, G(LR) {}, G(ERI) {}\n{:.10}",
            ["none", "ewald"][k],
            rr.net_force,
            rr.commutator,
            rr.n_sr_triplets,
            rr.n_g_lr,
            rr.n_g_eri,
            rr.grad
        );
        assert!(err < FD_BAR, "exxdiv {k}: {err:e}");
        // Sanity only: blind to every mutation (see module doc).
        assert!(rr.net_force < 1e-10, "ΣF {:e}", rr.net_force);
        // The SR route is live (the anchor would not see a dead one).
        let vsr = rr
            .parts
            .vsr_basis
            .iter()
            .fold(0.0_f64, |m, v| m.max(v.abs()));
        assert!(vsr > 1e-3, "V_SR basis part {vsr:e}");
    }
    // FD itself: exxdiv shifts E by the constant −v_M N/2.
    for (_, v) in fdv {
        assert!(
            (v[0] - v[1]).abs() < 1e-8,
            "FD none {} vs ewald {}",
            v[0],
            v[1]
        );
    }
}

#[test]
fn h2_ewald_and_none_forces_agree() {
    let cell = cell_at(&GRAD_H2_ATOMS, cubic(4.0));
    let r = analytic(
        &cell,
        &pyscf_sto3g_h(),
        &[(ExxDiv::None, None), (ExxDiv::Ewald, None)],
    );
    let d = max_diff(&r[0].grad, &r[1].grad);
    eprintln!("H2: |F_ewald − F_none| = {d:.2e} (v_M = {})", r[1].madelung);
    assert!(r[1].madelung > 0.1);
    assert!(d < 1e-12, "{d:e}");
}

#[test]
fn h2_fd_anchor_catches_each_mutant() {
    let cell = cell_at(&GRAD_H2_ATOMS, cubic(4.0));
    let fdv = h2_fd();
    let runs = [
        (ExxDiv::None, Some(GradMutation::VneNoBasis)),
        (ExxDiv::None, Some(GradMutation::WSign)),
        (ExxDiv::None, Some(GradMutation::NoEwaldLr)),
        (ExxDiv::Ewald, Some(GradMutation::NoMadelungS)),
    ];
    let r = analytic(&cell, &pyscf_sto3g_h(), &runs);
    for ((exx, mutation), rr) in runs.iter().zip(&r) {
        let k = if *exx == ExxDiv::None { 0 } else { 1 };
        let err = max_fd_err(&rr.grad, fdv, k);
        eprintln!(
            "mutant {mutation:?} ({exx:?}): max|analytic − FD| = {err:.2e}, |ΣF| = {:.1e} \
             (translation invariance cannot see it)",
            rr.net_force
        );
        assert!(err > 1e-3, "{mutation:?} escaped the FD anchor: {err:e}");
    }
}

#[test]
fn gradient_rejects_mismatched_omega() {
    let cell = h2_cell(4.0);
    let (prep, hc, eri) = build(&cell, &pyscf_sto3g_h());
    let scf = gamma_rhf(&cell, &prep, &hc, &eri);
    let wrong = PeriodicHcoreConfig {
        omega: OMEGA + 0.1,
        ..hcore_cfg()
    };
    let err = gamma_rhf_gradient_with(
        &cell,
        &prep,
        &wrong,
        &hc,
        &eri,
        &scf,
        ExxDiv::None,
        &GammaGradConfig::default(),
    )
    .expect_err("omega mismatch must error");
    assert!(err.to_string().contains("omega"), "{err}");
}

#[test]
#[ignore = "slow: triclinic 4H s+p, 8 displaced dense-AFT builds + 16 SCFs; run in release with --ignored"]
fn triclinic_sp_force_matches_fd_of_own_energy() {
    let cell = cell_at(&TRI_MOVED, TRI_A);
    let bs = sp_basis_h();
    let r = analytic(&cell, &bs, &[(ExxDiv::None, None), (ExxDiv::Ewald, None)]);
    let fdv = fd(&TRI_MOVED, TRI_A, &bs, &TRI_COMPS);
    for (k, rr) in r.iter().enumerate() {
        let err = max_fd_err(&rr.grad, &fdv, k);
        eprintln!(
            "triclinic s+p exxdiv {}: max|analytic − FD| = {err:.2e}, |ΣF| {:.1e}, comm {:.1e}\n{:.10}",
            ["none", "ewald"][k],
            rr.net_force,
            rr.commutator,
            rr.grad
        );
        // p shells: libint2's 3-centre derivative loses precision for tight
        // Gaussian nuclei. Measured FD error vs the gradient-only nucleus
        // exponent: 8.3e-2 (1e16), 4.3e-6 (1e12), 1.85e-7 (1e10; the default
        // GRAD_NUCLEUS_EXPONENT). The same 1.85e-7 with the energy at 1e10 too,
        // so this is libint precision, not an energy/gradient mismatch. The bar
        // is set from that floor (H2, s only, reaches 2.4e-9 under FD_BAR).
        assert!(err < 5e-7, "exxdiv {k}: {err:e}");
        assert!(rr.net_force < 1e-10, "ΣF {:e}", rr.net_force);
    }
    let d = max_diff(&r[0].grad, &r[1].grad);
    eprintln!("triclinic: |F_ewald − F_none| = {d:.2e}");
    // Two INDEPENDENT SCFs (commutators 4e-12 / 6e-11 here): measured
    // 6.58e-12 (2026-09-25, before and after the UHF/KS port, bit-identical).
    // The defect this guards, a missing −(v_M/2) DSD overlap term, is
    // v_M tr(D dS/dR) ~ 6e-2 (FINDINGS It. 16), so 1e-10 sits between.
    assert!(d < 1e-10, "{d:e}");
}
