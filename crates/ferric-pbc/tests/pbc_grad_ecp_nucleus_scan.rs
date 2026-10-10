// LANL2DZ exponents such as 0.318 are basis data, not approximations of 1/pi.
#![allow(
    clippy::approx_constant,
    clippy::needless_range_loop,
    clippy::float_cmp
)]
//! DIAGNOSTIC (`#[ignore]`, prints a table; asserts only harness validity):
//! is the ~4e-7 big-box ECP force scatter (FINDINGS "ECP periodic force
//! big-box limit (measured 2026-09-28)") the gradient-only Gaussian-nucleus
//! exponent floor ([`GRAD_NUCLEUS_EXPONENT`] = 1e9, libint2's p/d-shell
//! precision on the SR attraction DERIVATIVE; the energy keeps 1e16)?
//!
//! Run (release, serially, one BLAS thread):
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 cargo test --release -p ferric-pbc \
//!     --test pbc_grad_ecp_nucleus_scan -- --ignored --nocapture --test-threads=1
//! ```
//!
//! Optional: `FERRIC_NUC_EXP_SCAN="1e7,1e8,1e9,1e10,1e11"` overrides the
//! scanned exponents (comma list; each must be ≤ the energy's 1e16).
//!
//! System: HI (H STO-3G, I LANL2DZ + LANL2DZ ECP, Z_eff 7), off-axis in a
//! cubic a = 24 Bohr box (the smallest box of the 2026-09-28 fit window,
//! whose scatter was a-independent), exxdiv ewald, dense AFT, Gamma RHF.
//! CONTROL: the same geometry and AO basis with the ECP REMOVED and the
//! iodine nucleus replaced by a Z = 7 nucleus (so the nuclear charges, the
//! electron count and the AO layout are identical — the ECP operator is the
//! ONLY difference).
//!
//! What is measured, per scanned exponent ζ (energy and density fixed, only
//! the gradient's `GammaGradConfig::nucleus_exponent` changes — the existing
//! test-facing override; no production default is touched):
//! * TOTAL: analytic force − Richardson central FD (h = 2e-3, 1e-3) of
//!   ferric's OWN Gamma energy (every displaced SCF seeded from the
//!   reference density), iodine x/y/z (ΣF ≈ 0, so H mirrors it).
//! * PER TERM, at the FIXED reference density D (no SCF noise; the hcore
//!   pieces are linear in D): analytic part − Richardson FD of `Σ D X(R)`
//!   for X = T (`kinetic`), V_SR (`vsr_basis + vsr_nuc`), V_LR
//!   (`vlr_basis + vlr_nuc`), V_ECP (`ecp`), and E_nn (`nn_sr + nn_lr`).
//!   The remainder (TOTAL − Σ terms) is the `overlap` (W, v_g0, Madelung) +
//!   `eri` residual, i.e. the 2e/SCF side, by difference.
//! * vs MOLECULAR: box force − Richardson FD of ferric's molecular RHF on
//!   the same geometry (the 2026-09-28 comparison). This carries the
//!   physical ζ-independent c3'/a³ finite-size term, so read its CHANGE
//!   across ζ, not its value.
//!
//! Only `vsr_basis`/`vsr_nuc` can depend on ζ (grad.rs `assemble`: the
//! override enters only `sr_attraction_gradient`); every other part is
//! checked (and asserted) ζ-independent to 1e-13, and ζ = default is
//! asserted bit-identical to `nucleus_exponent: None`.
//!
//! PREDICTIONS (write-down before measuring):
//! * IF the floor is the nucleus exponent: the V_SR term residual carries
//!   the TOTAL residual, has the scan's U shape (rising ~10-30x per decade
//!   above ~1e9, rising again at 1e7 from the 1e16-energy mismatch), and the
//!   box−molecular difference moves across ζ by about the scatter (~4e-7);
//!   the ECP term residual is flat and small; the no-ECP control shows the
//!   same V_SR behaviour.
//! * IF it is something else (ECP quadrature, lattice-sum/image cutoff, SCF
//!   convergence, 2e side): the V_SR residual at 1e8-1e9 is ≪ 4e-7, and the
//!   TOTAL residual / box−molecular difference is FLAT across ζ; the
//!   non-flat piece sits in the ECP term (ECP quadrature — then the control
//!   lacks it), in the remainder (2e/SCF side), or nowhere in the own-energy
//!   FD at all (then the energies themselves disagree box vs molecular, a
//!   truncation of the ENERGY that an own-FD check is blind to by
//!   construction; the ζ-independence of box−molecular then says the same).
//!
//! Harness-validity asserts (not physics bars): SCFs converge; the
//! ζ-independent kinetic and E_nn term residuals are < 1e-8 (a failure there
//! means the FD/density convention of this harness is wrong — do not read
//! the table); the ECP system's ECP part is non-vacuous (> 1e-3); the
//! control's is exactly zero.
//!
//! Cost (estimate, NOT measured): per system 13 hcore + 13 dense-AFT builds
//! (a = 24, ~2e6 half-sphere G) + 13 SCFs + 12 molecular SCFs + one gradient
//! per ζ; a few minutes per test in release.

mod common;

use common::{gamma_config, max_abs_diff};
use ferric_core::basis::{BasisSet, Shell};
use ferric_core::ecp::{EcpDef, EcpShell, EcpTerm};
use ferric_core::mol::{Atom, Molecule};
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_pbc::dense_aft::{DenseAftEri, ExxDiv, DEFAULT_DENSE_AFT_MAX_BYTES};
use ferric_pbc::grad::{
    gamma_rhf_gradient_with, GammaGradConfig, GammaGradient, GRAD_NUCLEUS_EXPONENT,
};
use ferric_pbc::hcore::{periodic_hcore, PeriodicHcore, PeriodicHcoreConfig};
use ferric_pbc::lattice::Cell;
use ferric_scf::result::ScfResult;
use ferric_scf::rhf::{solve_rhf, solve_rhf_injected, PeriodicInjection, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ndarray::Array2;
use std::collections::HashMap;

const OMEGA: f64 = 0.8;
const HCORE_PRECISION: f64 = 1e-14;
const AFT_PRECISION: f64 = 1e-10;
const BOX: f64 = 24.0;
/// Richardson pair: central differences at H1 and H1/2.
const H1: f64 = 2e-3;
const SCAN_DEFAULT: [f64; 5] = [1e7, 1e8, 1e9, 1e10, 1e11];
/// Iodine (atom 1) x, y, z.
const COMPS: [(usize, usize); 3] = [(1, 0), (1, 1), (1, 2)];
/// Term order: kinetic, V_SR, V_LR, V_ECP, E_nn.
const TERMS: [&str; 5] = ["kin", "V_SR", "V_LR", "V_ECP", "E_nn"];

// ------------------------------------------------------------ fixtures
// (copied from pbc_grad_ecp.rs's Iteration-14 fixtures; test files cannot
// share them.)

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

/// The system under test: HI/LANL2DZ with the ECP, or the control (same
/// AO shells on a Z = 7 nucleus, no ECP).
#[derive(Clone, Copy, PartialEq)]
enum Sys {
    HiEcp,
    NoEcpControl,
}

impl Sys {
    fn label(self) -> &'static str {
        match self {
            Sys::HiEcp => "HI/LANL2DZ+ECP",
            Sys::NoEcpControl => "control: I shells on Z=7, no ECP",
        }
    }
    fn heavy_z(self) -> i32 {
        match self {
            Sys::HiEcp => 53,
            Sys::NoEcpControl => 7,
        }
    }
    fn basis(self) -> BasisSet {
        let mut shells = HashMap::new();
        shells.insert(
            1,
            vec![shell(
                0,
                &[3.42525091, 0.62391373, 0.1688554],
                &[0.15432897, 0.53532814, 0.44463454],
            )],
        );
        shells.insert(
            self.heavy_z(),
            vec![
                shell(0, &[0.7242, 0.4653], &[-2.9731048, 3.4827643]),
                shell(0, &[0.1336], &[1.0]),
                shell(1, &[1.29, 0.318], &[-0.2092377, 1.1035347]),
                shell(1, &[0.1053], &[1.0]),
            ],
        );
        let mut ecps = HashMap::new();
        if self == Sys::HiEcp {
            ecps.insert(53, lanl2dz_i_ecp());
        }
        BasisSet {
            name: "HI-nucleus-scan".into(),
            shells,
            ecps,
        }
    }
    fn cell(self, pos: &[[f64; 3]], bs: &BasisSet) -> Cell {
        let heavy = match self {
            Sys::HiEcp => "I",
            Sys::NoEcpControl => "N",
        };
        let atom = |symbol: &str, z: i32, r: [f64; 3]| Atom {
            symbol: symbol.into(),
            z,
            x: r[0],
            y: r[1],
            zpos: r[2],
            ghost: false,
            n_core_ecp: 0,
        };
        let mut mol = Molecule {
            atoms: vec![atom("H", 1, pos[0]), atom(heavy, self.heavy_z(), pos[1])],
            charge: 0,
            multiplicity: 1,
        };
        mol.apply_ecp(bs);
        Cell::new(mol, [[BOX, 0.0, 0.0], [0.0, BOX, 0.0], [0.0, 0.0, BOX]]).expect("cell")
    }
}

/// Off-axis HI (no force component zero by symmetry), centred-ish in the box
/// (the geometry of `big_box_ecp_force_term_is_the_molecular_ecp_gradient`).
fn ref_positions() -> Vec<[f64; 3]> {
    let shift = [0.37, 0.21, 0.5 * BOX - 1.52];
    [[0.0, 0.0, 0.0], [0.4, -0.3, 3.0]]
        .iter()
        .map(|r| [r[0] + shift[0], r[1] + shift[1], r[2] + shift[2]])
        .collect()
}

/// `pos` (Bohr) with Cartesian component `x` of atom `a` displaced by `h` Bohr.
fn moved(pos: &[[f64; 3]], a: usize, x: usize, h: f64) -> Vec<[f64; 3]> {
    let mut p = pos.to_vec();
    p[a][x] += h;
    p
}

/// Hcore config at the file's `OMEGA` with `precision = HCORE_PRECISION`; the
/// other fields are the defaults of `with_omega`.
fn hcore_cfg() -> PeriodicHcoreConfig {
    PeriodicHcoreConfig {
        precision: HCORE_PRECISION,
        ..PeriodicHcoreConfig::with_omega(OMEGA)
    }
}

/// The nucleus exponents to scan: the comma-separated
/// `FERRIC_NUC_EXP_SCAN` list when set and non-empty (panics on an
/// unparsable entry), else `SCAN_DEFAULT`.
fn scan_exponents() -> Vec<f64> {
    match std::env::var("FERRIC_NUC_EXP_SCAN") {
        Ok(s) if !s.trim().is_empty() => s
            .split(',')
            .map(|t| {
                t.trim()
                    .parse::<f64>()
                    .unwrap_or_else(|e| panic!("FERRIC_NUC_EXP_SCAN entry {t:?}: {e}"))
            })
            .collect(),
        _ => SCAN_DEFAULT.to_vec(),
    }
}

/// Largest element-wise `|a|` (0 for an empty matrix).
fn amax(a: &Array2<f64>) -> f64 {
    a.iter().fold(0.0_f64, |m, x| m.max(x.abs()))
}

/// `Σ_ij d_ij x_ij`: the element-wise product sum, i.e. `tr(d xᵀ)`.
fn tr(d: &Array2<f64>, x: &Array2<f64>) -> f64 {
    (d * x).sum()
}

/// Richardson from central differences at `h` and `h/2`.
fn richardson(c_h: f64, c_h2: f64) -> f64 {
    (4.0 * c_h2 - c_h) / 3.0
}

// ------------------------------------------------------------ drivers

struct Built {
    cell: Cell,
    prep: PreparedBasis,
    hc: PeriodicHcore,
}

/// Builds the system's cell at `pos` and its periodic hcore; asserts that
/// the ECP is really present (`HiEcp`: `n_ecp_triples > 0`) or absent
/// (`NoEcpControl`: no `v_ecp`).
fn build_hcore(sys: Sys, pos: &[[f64; 3]], bs: &BasisSet) -> Built {
    let cell = sys.cell(pos, bs);
    let prep = PreparedBasis::new(cell.mol(), bs).expect("prep");
    let hc = periodic_hcore(&cell, &prep, &hcore_cfg()).expect("hcore");
    match sys {
        Sys::HiEcp => assert!(hc.n_ecp_triples > 0, "no ECP triples: vacuous"),
        Sys::NoEcpControl => assert!(hc.v_ecp.is_none(), "control carries an ECP"),
    }
    Built { cell, prep, hc }
}

/// The `ExxDiv::Ewald` dense-AFT ERI tensor of the built system at
/// `AFT_PRECISION`; panics on error.
fn aft(b: &Built) -> DenseAftEri {
    DenseAftEri::build(
        &b.cell,
        &b.prep,
        &b.hc.s,
        ExxDiv::Ewald,
        AFT_PRECISION,
        DEFAULT_DENSE_AFT_MAX_BYTES,
    )
    .expect("dense AFT")
}

fn gamma_rhf(b: &Built, eri: &DenseAftEri, seed: Option<&Array2<f64>>) -> ScfResult {
    let ctx = ParallelContext::default();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &b.prep).expect("schwarz");
    let cfg = RhfConfig {
        init_guess_density: seed.cloned(),
        max_iter: 300,
        ..gamma_config()
    };
    let inj = PeriodicInjection {
        s: b.hc.s.clone(),
        h: b.hc.h.clone(),
        vnn: b.hc.enn,
        j: Box::new(eri.j_builder()),
        k: Box::new(eri.k_builder()),
        xc: None,
    };
    let r =
        solve_rhf_injected(&ctx, b.cell.mol(), &b.prep, op, &bounds, &cfg, inj).expect("gamma RHF");
    assert!(r.converged, "gamma RHF did not converge");
    r
}

/// The molecular (non-periodic) RHF of the cell's molecule, starting from
/// the optional `seed` density, with `max_iter = 300`; panics if it does not
/// converge.
fn molecular_energy(b: &Built, seed: Option<&Array2<f64>>) -> ScfResult {
    let ctx = ParallelContext::default();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &b.prep).expect("schwarz");
    let cfg = RhfConfig {
        init_guess_density: seed.cloned(),
        max_iter: 300,
        ..gamma_config()
    };
    let r = solve_rhf(&ctx, b.cell.mol(), &b.prep, op, &bounds, &cfg).expect("molecular RHF");
    assert!(r.converged, "molecular RHF did not converge");
    r
}

/// `[Σ D T, Σ D V_SR, Σ D V_LR, Σ D V_ECP, E_nn]` at fixed `d`.
fn term_energies(hc: &PeriodicHcore, d: &Array2<f64>) -> [f64; 5] {
    [
        tr(d, &hc.t),
        tr(d, &hc.v_sr),
        tr(d, &hc.v_lr),
        hc.v_ecp.as_ref().map_or(0.0, |v| tr(d, v)),
        hc.enn,
    ]
}

/// Analytic per-term forces in [`TERMS`] order, plus the remainder
/// (`grad − Σ terms` = overlap + eri).
fn analytic_terms(g: &GammaGradient) -> [Array2<f64>; 6] {
    let p = &g.parts;
    let terms = [
        p.kinetic.clone(),
        &p.vsr_basis + &p.vsr_nuc,
        &p.vlr_basis + &p.vlr_nuc,
        p.ecp.clone(),
        &p.nn_sr + &p.nn_lr,
    ];
    let mut rest = g.grad.clone();
    for t in &terms {
        rest -= t;
    }
    let [a, b, c, d, e] = terms;
    [a, b, c, d, e, rest]
}

/// Per displaced geometry: the fixed-D term energies, the Gamma SCF energy,
/// the molecular SCF energy.
struct Point {
    terms: [f64; 5],
    e_box: f64,
    e_mol: f64,
}

fn scan(sys: Sys) {
    let label = sys.label();
    let bs = sys.basis();
    let pos0 = ref_positions();
    let t0 = std::time::Instant::now();

    // --- reference geometry: SCF, molecular SCF.
    let b0 = build_hcore(sys, &pos0, &bs);
    let eri0 = aft(&b0);
    let scf0 = gamma_rhf(&b0, &eri0, None);
    let d0 = scf0.density_total.clone();
    let mol0 = molecular_energy(&b0, Some(&d0));
    let dmol0 = mol0.density_total.clone();
    eprintln!(
        "[{label}] a = {BOX}: E_box {:.12}, E_mol {:.12}, nao {}, ECP triples {}, \
         half-G {}, SR triplets {} ({:.1} s)",
        scf0.energy,
        mol0.energy,
        b0.prep.nbasis(),
        b0.hc.n_ecp_triples,
        eri0.n_g_half(),
        b0.hc.n_sr_triplets,
        t0.elapsed().as_secs_f64()
    );

    // --- displaced geometries (ζ-independent: done once).
    let mut fd_terms = vec![[0.0_f64; 5]; COMPS.len()];
    let mut fd_box = vec![0.0_f64; COMPS.len()];
    let mut fd_mol = vec![0.0_f64; COMPS.len()];
    for (ci, &(a, x)) in COMPS.iter().enumerate() {
        let mut central = Vec::new(); // at h, h/2
        for h in [H1, 0.5 * H1] {
            let pt = |s: f64| -> Point {
                let b = build_hcore(sys, &moved(&pos0, a, x, s * h), &bs);
                let terms = term_energies(&b.hc, &d0);
                let eri = aft(&b);
                let e_box = gamma_rhf(&b, &eri, Some(&d0)).energy;
                let e_mol = molecular_energy(&b, Some(&dmol0)).energy;
                Point {
                    terms,
                    e_box,
                    e_mol,
                }
            };
            let (p, m) = (pt(1.0), pt(-1.0));
            let mut c_terms = [0.0; 5];
            for k in 0..5 {
                c_terms[k] = (p.terms[k] - m.terms[k]) / (2.0 * h);
            }
            central.push((
                c_terms,
                (p.e_box - m.e_box) / (2.0 * h),
                (p.e_mol - m.e_mol) / (2.0 * h),
            ));
        }
        let (h_a, h_b) = (&central[0], &central[1]);
        for k in 0..5 {
            fd_terms[ci][k] = richardson(h_a.0[k], h_b.0[k]);
        }
        fd_box[ci] = richardson(h_a.1, h_b.1);
        fd_mol[ci] = richardson(h_a.2, h_b.2);
        eprintln!(
            "[{label}] comp (atom {a}, {x}): FD box {:+.10e} (|c(h)−c(h/2)| {:.1e}), \
             FD mol {:+.10e}, terms [{}] ({:.1} s)",
            fd_box[ci],
            (h_a.1 - h_b.1).abs(),
            fd_mol[ci],
            fd_terms[ci]
                .iter()
                .zip(TERMS)
                .map(|(v, n)| format!("{n} {v:+.6e}"))
                .collect::<Vec<_>>()
                .join(", "),
            t0.elapsed().as_secs_f64()
        );
    }

    // --- default path: None == Some(GRAD_NUCLEUS_EXPONENT), bitwise.
    let grad_at = |z: Option<f64>| -> GammaGradient {
        let cfg = GammaGradConfig {
            nucleus_exponent: z,
            ..Default::default()
        };
        gamma_rhf_gradient_with(
            &b0.cell,
            &b0.prep,
            &hcore_cfg(),
            &b0.hc,
            &eri0,
            &scf0,
            ExxDiv::Ewald,
            &cfg,
        )
        .expect("gamma gradient")
    };
    let g_default = grad_at(None);
    let g_explicit = grad_at(Some(GRAD_NUCLEUS_EXPONENT));
    let dd = max_abs_diff(&g_default.grad, &g_explicit.grad);
    assert!(
        dd == 0.0,
        "default vs explicit {GRAD_NUCLEUS_EXPONENT:e}: {dd:e}"
    );
    match sys {
        Sys::HiEcp => assert!(amax(&g_default.parts.ecp) > 1e-3, "vacuous ECP part"),
        Sys::NoEcpControl => assert!(amax(&g_default.parts.ecp) == 0.0),
    }
    eprintln!(
        "[{label}] default ζ = {GRAD_NUCLEUS_EXPONENT:e}: commutator {:.1e}, ΣF {:.1e}, \
         |F_I| {:.3e}, ECP part max {:.3e}",
        g_default.commutator,
        g_default.net_force,
        amax(&g_default.grad),
        amax(&g_default.parts.ecp)
    );

    // --- the scan.
    let zetas = scan_exponents();
    let first = analytic_terms(&g_default);
    eprintln!(
        "[{label}] residual = analytic − Richardson FD, max over iodine x/y/z (Ha/Bohr); \
         rest = TOTAL − Σ terms (overlap/W + eri); box−mol per component"
    );
    eprintln!(
        "{:>8} | {:>9} | {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} | {:>9} | {:>13} {:>13} {:>13}",
        "zeta",
        "TOTAL",
        "kin",
        "V_SR",
        "V_LR",
        "V_SR+LR",
        "V_ECP",
        "E_nn",
        "rest",
        "box−mol x",
        "box−mol y",
        "box−mol z"
    );
    for &z in &zetas {
        assert!(
            z > 0.0 && z <= hcore_cfg().nucleus_exponent,
            "ζ {z:e} out of range"
        );
        let g = grad_at(Some(z));
        let an = analytic_terms(&g);
        // ζ-independent parts must not move (only V_SR may).
        for (k, name) in [
            (0, "kin"),
            (2, "V_LR"),
            (3, "V_ECP"),
            (4, "E_nn"),
            (5, "rest"),
        ] {
            let dz = max_abs_diff(&an[k], &first[k]);
            assert!(dz < 1e-13, "{name} moved with ζ = {z:e}: {dz:e}");
        }
        let mut r_tot = 0.0_f64;
        let mut r_term = [0.0_f64; 5];
        let mut r_rest = 0.0_f64;
        let mut r_srlr = 0.0_f64;
        let mut box_mol = [0.0_f64; 3];
        for (ci, &(a, x)) in COMPS.iter().enumerate() {
            let rt = g.grad[(a, x)] - fd_box[ci];
            r_tot = r_tot.max(rt.abs());
            let mut sum_terms = 0.0;
            for k in 0..5 {
                let r = an[k][(a, x)] - fd_terms[ci][k];
                r_term[k] = r_term[k].max(r.abs());
                sum_terms += r;
            }
            r_rest = r_rest.max((rt - sum_terms).abs());
            let srlr = an[1][(a, x)] + an[2][(a, x)] - fd_terms[ci][1] - fd_terms[ci][2];
            r_srlr = r_srlr.max(srlr.abs());
            box_mol[ci] = g.grad[(a, x)] - fd_mol[ci];
        }
        eprintln!(
            "{z:>8.0e} | {r_tot:>9.2e} | {:>9.2e} {:>9.2e} {:>9.2e} {r_srlr:>9.2e} {:>9.2e} \
             {:>9.2e} | {r_rest:>9.2e} | {:>+13.6e} {:>+13.6e} {:>+13.6e}",
            r_term[0],
            r_term[1],
            r_term[2],
            r_term[3],
            r_term[4],
            box_mol[0],
            box_mol[1],
            box_mol[2]
        );
        // Harness validity: ζ-independent, integral-precision-limited terms.
        assert!(
            r_term[0] < 1e-8,
            "kinetic term FD residual {:e}: harness wrong",
            r_term[0]
        );
        assert!(
            r_term[4] < 1e-8,
            "E_nn term FD residual {:e}: harness wrong",
            r_term[4]
        );
    }
    eprintln!("[{label}] done ({:.1} s)", t0.elapsed().as_secs_f64());
}

#[test]
#[ignore = "diagnostic: HI/LANL2DZ a = 24 box, 13 hcore + dense-AFT builds + 13 Gamma SCFs + 12 molecular SCFs, gradient per nucleus exponent; run in release with --ignored --nocapture, serially"]
fn hi_ecp_big_box_force_residual_vs_gradient_nucleus_exponent() {
    scan(Sys::HiEcp);
}

#[test]
#[ignore = "diagnostic control: same geometry/AO shells on a Z = 7 nucleus, no ECP; same cost as the ECP scan; run in release with --ignored --nocapture, serially"]
fn no_ecp_control_force_residual_vs_gradient_nucleus_exponent() {
    scan(Sys::NoEcpControl);
}
