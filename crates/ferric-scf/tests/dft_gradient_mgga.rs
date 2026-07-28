//! FD validation of the meta-GGA (SCAN / r2SCAN) nuclear gradient.
//!
//! The τ-dependent term is the piece that distinguishes this from the GGA
//! gradient in `dft_gradient_gga.rs`:
//!
//! ```text
//!   ∂E/∂R_{A,α} += −Σ_g w_g v_τ Σ_b Σ_{μ∈A} ∂²_{αb} χ_μ (D ∂_b χ)_μ   (AO deriv)
//!                +  Σ_{g: home=A} w_g v_τ ∂_α τ(r_g)                   (grid response)
//! ```
//!
//! A wrong factor or a missing μ↔ν symmetrization in that term shows up here
//! immediately: on H2O/STO-3G the τ contribution is O(1e-2) Ha/Bohr, two orders
//! of magnitude above the FD noise floor, so a factor-2 error cannot hide.
//!
//! AO Hessians are only implemented for s/p shells, so these tests use STO-3G
//! and 6-31G.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::ks_gradient::{ks_gradient_closed, ks_gradient_uks};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::solve_uhf;
use ndarray::Array2;
use rayon::prelude::*;
use serde::Deserialize;
use std::fs;
use std::path::PathBuf;

/// Meta-GGA SCF is stiffer than GGA (the τ Fock amplifies grid noise), so the
/// energy parks on a ~1e-8 Ha floor — matching `dft_scan.rs`'s thresholds.
/// A tighter energy_conv would limit-cycle and leave perturbed geometries with
/// a stale density, poisoning the FD reference.
fn cfg_for(xc: &str) -> RhfConfig {
    RhfConfig {
        xc: Some(xc.into()),
        df_j_aux: Some("def2-universal-jkfit".into()),
        energy_conv: 1e-9,
        density_conv: 1e-8,
        max_iter: 500,
        ..Default::default()
    }
}

fn fd_gradient_closed(xyz: &str, basis_name: &str, xc: &str, delta: f64) -> Array2<f64> {
    let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
    let natoms = mol.atoms.len();
    let bs = basis::bundled(basis_name).unwrap();

    // Seed every displaced SCF from the equilibrium density (same reasoning as
    // dft_gradient_gga.rs: a fresh guess can land a ± twin in a different SCF
    // basin and fabricate a nonphysical FD "gradient").
    let base_cfg = cfg_for(xc);
    let prep0 = PreparedBasis::new(&mol, &bs).unwrap();
    let bounds0 = SchwarzBounds::compute(Operator::coulomb(), &prep0).unwrap();
    let res0 = solve_rhf(
        &ParallelContext::default(), &mol, &prep0, Operator::coulomb(), &bounds0, &base_cfg,
    )
    .unwrap();
    let cfg = RhfConfig {
        init_guess_density: Some(res0.density_r().clone()),
        use_sad_guess: false,
        ..base_cfg
    };

    let pairs: Vec<(usize, usize)> =
        (0..natoms).flat_map(|a| (0..3).map(move |c| (a, c))).collect();
    let results: Vec<((usize, usize), f64)> = pairs
        .par_iter()
        .map(|&(atom, coord)| {
            let mut mol_p = mol.clone();
            let mut mol_m = mol.clone();
            match coord {
                0 => { mol_p.atoms[atom].x += delta; mol_m.atoms[atom].x -= delta; }
                1 => { mol_p.atoms[atom].y += delta; mol_m.atoms[atom].y -= delta; }
                _ => { mol_p.atoms[atom].zpos += delta; mol_m.atoms[atom].zpos -= delta; }
            }
            let mut e = [0.0_f64; 2];
            for (i, m) in [mol_p, mol_m].iter().enumerate() {
                let prep = PreparedBasis::new(m, &bs).unwrap();
                let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
                let r = solve_rhf(
                    &ParallelContext::default(), m, &prep, Operator::coulomb(), &bounds, &cfg,
                )
                .unwrap();
                assert!(r.converged, "FD solve diverged at atom={atom} coord={coord} sign={i}");
                e[i] = r.energy;
            }
            ((atom, coord), (e[0] - e[1]) / (2.0 * delta))
        })
        .collect();

    let mut grad = Array2::<f64>::zeros((natoms, 3));
    for ((atom, coord), g) in results {
        grad[(atom, coord)] = g;
    }
    grad
}

fn run_closed(label: &str, xyz: &str, basis_name: &str, xc: &str, tol: f64) {
    let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
    let bs = basis::bundled(basis_name).unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let res = solve_rhf(&ParallelContext::default(), &mol, &prep, op, &bounds, &cfg_for(xc))
        .unwrap();
    assert!(res.converged, "{label} {xc}: reference SCF did not converge");

    let g_ana = ks_gradient_closed(&mol, &prep, &bs, op, &bounds, xc, &res, None).unwrap();
    let g_fd = fd_gradient_closed(xyz, basis_name, xc, 1e-3);

    eprintln!("=== {label} {xc} RKS gradient (analytic vs FD) ===");
    let mut max_diff = 0.0_f64;
    for a in 0..mol.atoms.len() {
        for c in 0..3 {
            let diff = (g_ana[(a, c)] - g_fd[(a, c)]).abs();
            max_diff = max_diff.max(diff);
            eprintln!(
                "  atom={a} coord={c}: ana={:+.6e} fd={:+.6e} diff={:.2e}",
                g_ana[(a, c)], g_fd[(a, c)], diff
            );
        }
    }
    eprintln!("  max diff: {max_diff:.2e}, tol: {tol:.0e}");
    assert!(max_diff < tol, "{label} {xc}: max |ana-fd| = {max_diff:.3e} exceeds {tol:.0e}");
}

// ── Closed shell (RKS) ────────────────────────────────────────────────────

#[test]
fn scan_gradient_h2_sto3g_vs_fd() {
    run_closed("H2/sto-3g", "2\nH2\nH 0 0 0\nH 0 0 0.74\n", "sto-3g", "SCAN", 1e-4);
}

#[test]
fn scan_gradient_h2o_sto3g_vs_fd() {
    run_closed(
        "H2O/sto-3g",
        "3\nH2O\nO 0 0 0\nH 0 0.7572 0.5868\nH 0 -0.7572 0.5868\n",
        "sto-3g", "SCAN", 3e-4,
    );
}

#[test]
fn r2scan_gradient_h2o_sto3g_vs_fd() {
    run_closed(
        "H2O/sto-3g",
        "3\nH2O\nO 0 0 0\nH 0 0.7572 0.5868\nH 0 -0.7572 0.5868\n",
        "sto-3g", "r2SCAN", 3e-4,
    );
}

#[test]
fn scan_gradient_h2o_631g_vs_fd() {
    run_closed(
        "H2O/6-31G",
        "3\nH2O\nO 0 0 0\nH 0 0.7572 0.5868\nH 0 -0.7572 0.5868\n",
        "6-31g", "SCAN", 3e-4,
    );
}

// ── PySCF cross-check (closed shell) ─────────────────────────────────────
//
// Independent of the FD check above: FD validates the analytic gradient against
// ferric's OWN energy, PySCF validates it against an external implementation of
// the same physics. References from `scripts/gen_pyscf_mgga_grad_refs.py`, run
// with `grid_response = True` to match ferric's P2.1 corrections.

#[derive(Deserialize)]
struct GradRef {
    grad: Vec<[f64; 3]>,
    e_total: f64,
    #[serde(default)]
    converged: bool,
}

fn ref_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../testdata/reference")
        .join(name)
}

fn load_ref(name: &str) -> GradRef {
    let txt = fs::read_to_string(ref_path(name))
        .unwrap_or_else(|e| panic!("missing reference {name}: {e} — regenerate with \
                                    scripts/gen_pyscf_mgga_grad_refs.py"));
    let r: GradRef = serde_json::from_str(&txt).unwrap();
    assert!(r.converged, "reference {name} did not converge in PySCF");
    r
}

fn run_vs_pyscf(label: &str, xyz: &str, basis_name: &str, xc: &str, ref_file: &str, tol: f64) {
    let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
    let bs = basis::bundled(basis_name).unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let res = solve_rhf(&ParallelContext::default(), &mol, &prep, op, &bounds, &cfg_for(xc))
        .unwrap();
    assert!(res.converged, "{label} {xc}: SCF did not converge");

    let g_ana = ks_gradient_closed(&mol, &prep, &bs, op, &bounds, xc, &res, None).unwrap();
    let r = load_ref(ref_file);

    eprintln!("=== {label} {xc} RKS gradient (ferric vs PySCF) ===");
    eprintln!("  E: ferric={:.10} pyscf={:.10} diff={:.2e}",
              res.energy, r.e_total, (res.energy - r.e_total).abs());
    let mut max_diff = 0.0_f64;
    for a in 0..mol.atoms.len() {
        for c in 0..3 {
            let diff = (g_ana[(a, c)] - r.grad[a][c]).abs();
            max_diff = max_diff.max(diff);
            eprintln!(
                "  atom={a} coord={c}: ferric={:+.6e} pyscf={:+.6e} diff={:.2e}",
                g_ana[(a, c)], r.grad[a][c], diff
            );
        }
    }
    eprintln!("  max diff: {max_diff:.2e}, tol: {tol:.0e}");
    assert!(max_diff < tol, "{label} {xc}: max |ferric-pyscf| = {max_diff:.3e} exceeds {tol:.0e}");
}

#[test]
fn scan_gradient_h2_sto3g_vs_pyscf() {
    run_vs_pyscf("H2/sto-3g", "2\nH2\nH 0 0 0\nH 0 0 0.74\n", "sto-3g", "SCAN",
                 "h2_sto-3g_scan_grad.json", 1e-4);
}

#[test]
fn scan_gradient_h2o_sto3g_vs_pyscf() {
    run_vs_pyscf("H2O/sto-3g",
                 "3\nH2O\nO 0 0 0\nH 0 0.7572 0.5868\nH 0 -0.7572 0.5868\n",
                 "sto-3g", "SCAN", "h2o_sto-3g_scan_grad.json", 1e-4);
}

#[test]
fn r2scan_gradient_h2o_sto3g_vs_pyscf() {
    run_vs_pyscf("H2O/sto-3g",
                 "3\nH2O\nO 0 0 0\nH 0 0.7572 0.5868\nH 0 -0.7572 0.5868\n",
                 "sto-3g", "r2SCAN", "h2o_sto-3g_r2scan_grad.json", 1e-4);
}

#[test]
fn scan_gradient_h2o_631g_vs_pyscf() {
    run_vs_pyscf("H2O/6-31G",
                 "3\nH2O\nO 0 0 0\nH 0 0.7572 0.5868\nH 0 -0.7572 0.5868\n",
                 "6-31g", "SCAN", "h2o_6-31g_scan_grad.json", 1e-4);
}

// ── Open shell (UKS) ──────────────────────────────────────────────────────
//
// HISTORY (corrected 2026-07-27). This block previously asserted a
// "pre-existing defect in the spin-polarized SCAN SCF ENERGY", inferred from a
// 2.2e-4 Ha ferric-vs-PySCF gap on OH/STO-3G SCAN plus a non-smooth E(R). That
// diagnosis was WRONG. There is no polarized meta-GGA defect. What was actually
// measured was OH being a pathological SCF system for meta-GGA, compounded by
// SCAN's own grid pathology. The evidence that overturned it:
//
//  1. libxc packing/kernel is EXACT. Feeding hand-built polarized
//     (ρ_α, ρ_β, σ_αα, σ_αβ, σ_ββ, τ_α, τ_β) with ρ_α≠ρ_β and τ_α≠τ_β through
//     `eval_mgga_polarized` reproduces PySCF's `libxc.eval_xc(..., spin=1)`
//     BIT-FOR-BIT (all of exc/vrho/vsigma/vtau, MGGA_{X,C}_{SCAN,R2SCAN}).
//     A transposed or mis-strided τ buffer is therefore ruled out directly.
//     The strided parallel-chunk path is bit-identical to the serial path too.
//
//  2. E_xc on an IDENTICAL polarized density agrees. Evaluating ferric's
//     polarized meta-GGA E_xc on PySCF's own converged OH/STO-3G UKS density:
//       SCAN  : PySCF -9.008668117931  ferric -9.008668456727  → 3.4e-7
//       r2SCAN: PySCF -8.997832541635  ferric -8.997832880573  → 3.4e-7
//     The residual is the SAME constant 3.39e-7 for both functionals — a
//     grid/Becke-partition difference. A wrong τ factor would scale with each
//     functional's v_τ magnitude (which differ by ~10%), not stay constant.
//
//  3. Other genuinely polarized systems agree to ~1e-8, same code path.
//     ferric − PySCF at (99,302), UKS, RI-J def2-universal-jkfit:
//       NH2 doublet /sto-3g: PBE -2.4e-8  r2SCAN -2.9e-8  SCAN -2.7e-8
//       O2  triplet /sto-3g: PBE -4.8e-8  r2SCAN -9.1e-8  SCAN -4.2e-8
//       CH3 doublet /sto-3g: PBE +4.9e-9  r2SCAN -1.3e-8  SCAN -1.3e-9
//       (6-31G likewise, all ≤ 6.5e-7). Only OH is an outlier.
//
//  4. OH is the pathology, not the kernel. Its β-HOMO converges to ~-0.0004 Ha,
//     i.e. essentially zero gap, so the surface is near-flat: ferric needs 443
//     (SCAN) / 692 (r2SCAN) iterations at conv 1e-11, and the two codes settle
//     on marginally different points. PySCF is not a 1e-7 reference for this
//     system — its OWN OH/SCAN energy moves 1.9e-5 Ha for nothing but a
//     conv_tol change (1e-11 → 1e-12: -74.646861758 → -74.646843050).
//
//  5. The E(R) non-smoothness is SCAN-specific and grid-refinement-RESISTANT,
//     which is a known SCAN trait (r2SCAN was designed to fix exactly it), not
//     a spin bug. Over 1e-3 Å steps on OH/STO-3G:
//       SCAN   (75,110) : sign flips (+1.16e-4, +1.06e-5 against a -1.5e-4 trend)
//       SCAN   (150,302): STILL non-smooth (-2.35e-4, -1.04e-4, ..., -2.50e-4)
//       r2SCAN (75,110) : already monotone (-1.54e-4 → -1.22e-4)
//       r2SCAN (150,302): monotone, smooth to ~1e-6 (-1.587e-4 → -1.475e-4)
//     Same buffers, same assembly, same driver — only the functional differs.
//
// The OH/SCAN case below therefore keeps a LOOSE 1e-3 bound, but for the honest
// reason: OH is a near-degenerate SCF system and SCAN is grid-pathological, so
// neither code's answer is reference-grade there. It stays as a regression guard
// (a factor-2 or sign error in the τ term blows past it by ~2 orders of
// magnitude). The well-behaved polarized cases (NH2, CH3) carry the real bar at
// 1e-4 — the same tolerance as closed-shell.

fn uhf_cfg(xc: &str) -> RhfConfig {
    RhfConfig {
        xc: Some(xc.into()),
        df_j_aux: Some("def2-universal-jkfit".into()),
        energy_conv: 1e-9,
        density_conv: 1e-8,
        max_iter: 500,
        ..Default::default()
    }
}

fn run_uks_vs_pyscf(
    label: &str, xyz: &str, charge: i32, mult: usize, basis_name: &str,
    xc: &str, ref_file: &str, tol: f64,
) {
    let mol = Molecule::parse_xyz(xyz, charge, mult).unwrap();
    let bs = basis::bundled(basis_name).unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let res =
        solve_uhf(&ParallelContext::default(), &mol, &prep, &bounds, &uhf_cfg(xc)).unwrap();
    assert!(res.converged, "{label} {xc}: UKS did not converge");

    let g_ana = ks_gradient_uks(&mol, &prep, &bs, op, &bounds, xc, &res, None).unwrap();
    let r = load_ref(ref_file);

    eprintln!("=== {label} {xc} UKS gradient (ferric vs PySCF) ===");
    eprintln!("  E: ferric={:.10} pyscf={:.10} diff={:.2e}",
              res.energy, r.e_total, (res.energy - r.e_total).abs());
    let mut max_diff = 0.0_f64;
    for a in 0..mol.atoms.len() {
        for c in 0..3 {
            let diff = (g_ana[(a, c)] - r.grad[a][c]).abs();
            max_diff = max_diff.max(diff);
            eprintln!(
                "  atom={a} coord={c}: ferric={:+.6e} pyscf={:+.6e} diff={:.2e}",
                g_ana[(a, c)], r.grad[a][c], diff
            );
        }
    }
    eprintln!("  max diff: {max_diff:.2e}, tol: {tol:.0e}");
    assert!(max_diff < tol, "{label} {xc}: max |ferric-pyscf| = {max_diff:.3e} exceeds {tol:.0e}");
}

/// Control: the SAME UKS meta-GGA-capable driver on a GGA functional agrees with
/// PySCF to 3e-6. This is what pins the OH/SCAN gap on the polarized SCAN
/// energy rather than on `ks_gradient_uks` or the shared gradient assembly.
#[test]
fn pbe_gradient_oh_sto3g_uks_vs_pyscf() {
    // PySCF: E = -74.5726577938, ∂E/∂z(O) = +9.6437700e-2 (grid_response=True,
    // (75,110) unpruned Becke-1988 grid, RI-J def2-universal-jkfit, conv_tol 1e-10).
    let mol = Molecule::parse_xyz("2\nOH\nO 0 0 0\nH 0 0 0.97\n", 0, 2).unwrap();
    let bs = basis::bundled("sto-3g").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let res =
        solve_uhf(&ParallelContext::default(), &mol, &prep, &bounds, &uhf_cfg("PBE")).unwrap();
    assert!(res.converged);
    let g = ks_gradient_uks(&mol, &prep, &bs, op, &bounds, "PBE", &res, None).unwrap();
    let e_diff = (res.energy - (-74.5726577938)).abs();
    let g_diff = (g[(0, 2)] - 9.6437700e-2).abs();
    eprintln!("OH/sto-3g PBE UKS: dE={e_diff:.2e}  d(gz)={g_diff:.2e}");
    assert!(e_diff < 1e-6, "OH/PBE UKS energy vs PySCF: {e_diff:.3e}");
    assert!(g_diff < 1e-4, "OH/PBE UKS gradient vs PySCF: {g_diff:.3e}");
}

/// OH is a near-degenerate open-shell system (β-HOMO ≈ -4e-4 Ha) AND SCAN is
/// grid-pathological, so neither ferric nor PySCF is reference-grade here — see
/// the block comment above for the measurements. Kept as a coarse regression
/// guard on the τ gradient assembly only. The real bar is `nh2_*` / `ch3_*`.
#[test]
fn scan_gradient_oh_sto3g_uks_vs_pyscf() {
    run_uks_vs_pyscf("OH/sto-3g", "2\nOH\nO 0 0 0\nH 0 0 0.97\n", 0, 2, "sto-3g",
                     "SCAN", "oh_sto-3g_scan_grad.json", 1e-3);
}

const NH2: &str = "3\nNH2\nN 0 0 0.1414\nH 0 0.8067 -0.4950\nH 0 -0.8067 -0.4950\n";
const CH3: &str = "4\nCH3\nC 0 0 0\nH 0 1.0790 0\n\
                   H 0.9344 -0.5395 0\nH -0.9344 -0.5395 0\n";

/// Well-behaved polarized meta-GGA gradient: NH2 doublet. Its SCF energy agrees
/// with PySCF to 2.7e-8 Ha (SCAN) — same UKS + polarized-meta-GGA code path OH
/// uses — so this case CAN carry the closed-shell 1e-4 bar.
#[test]
fn scan_gradient_nh2_sto3g_uks_vs_pyscf() {
    run_uks_vs_pyscf("NH2/sto-3g", NH2, 0, 2, "sto-3g",
                     "SCAN", "nh2_sto-3g_scan_grad.json", 1e-4);
}

/// r2SCAN sibling of the NH2 case: exercises the same polarized τ assembly with
/// a different v_τ magnitude, which is what would expose a τ scaling error.
#[test]
fn r2scan_gradient_nh2_sto3g_uks_vs_pyscf() {
    run_uks_vs_pyscf("NH2/sto-3g", NH2, 0, 2, "sto-3g",
                     "R2SCAN", "nh2_sto-3g_r2scan_grad.json", 1e-4);
}

/// CH3 doublet — a second well-behaved polarized system, with the spin density
/// on carbon rather than on the electronegative centre.
#[test]
fn scan_gradient_ch3_sto3g_uks_vs_pyscf() {
    run_uks_vs_pyscf("CH3/sto-3g", CH3, 0, 2, "sto-3g",
                     "SCAN", "ch3_sto-3g_scan_grad.json", 1e-4);
}
