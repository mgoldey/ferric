//! Hybrid meta-GGA (TPSSh, M06-2X) nuclear gradients, closed and open shell.
//!
//! The gradient is the meta-GGA XC gradient (τ AO-derivative and grid-response
//! terms, `dft_gradient_mgga.rs`) plus the hybrid exact-exchange gradient
//! scaled by the libxc fraction (`c_K = k_mix.sr`, `dft_gradient_gga.rs`),
//! composed by `ks_gradient_*` with no functional-specific code. Two checks:
//!
//! * **FD of ferric's own energy** (central, δ = 1e-3 Bohr, every SCF seeded
//!   from the reference density). Exact four-centre J/K and the XC grid
//!   response make the analytic gradient the derivative of the energy the SCF
//!   defines, so the residual is FD truncation, O(δ²·E''') ~ 1e-7, plus SCF
//!   noise 2·1e-11/(2δ) ~ 1e-8. Bar `FD_TOL = 1e-5` Ha/Bohr (the FD-noise
//!   estimate with two decades of margin); measured 1.0e-7..1.4e-7. FD checks
//!   consistency with ferric's own energy only: it cannot see a missing exact
//!   exchange that the energy also lacks (the PySCF check does).
//! * **PySCF** `grid_response = True` on the same flat (75, 110) grid, exact
//!   J/K, from `scripts/gen_pyscf_hybrid_mgga_refs.py`. Bar `PYSCF_TOL =
//!   1e-6` Ha/Bohr: the SCAN / r2SCAN gradients reach 4e-8..5e-8 against
//!   PySCF at the same settings (`dft_gradient_mgga.rs`), and these measure
//!   2e-9..4e-8; 1e-6 leaves the PySCF SCF and grid tolerances room while a
//!   dropped exact-exchange or τ gradient term (3e-2..7e-2 Ha/Bohr, measured
//!   by mutation) is five orders above it.
//!
//! The 6-31G total energies are printed, not asserted: ferric's bundled
//! 6-31G and PySCF's differ slightly (RHF on this H2O: 7e-9 Ha), so they
//! differ by up to 1e-6 Ha on NH2 for reasons unrelated to the functional.
//! The cc-pVDZ energies in `dft_hybrid_mgga.rs` agree to 2e-13.
//!
//! 6-31G: the meta-GGA gradient needs AO Hessians, implemented for s/p
//! shells only.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_dft::grid::AtomicGridConfig;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::ks_gradient::{
    ks_gradient_closed_with_exchange_on_grids, ks_gradient_uks_with_exchange_on_grids, KsGradGrids,
};
use ferric_scf::result::ScfResult;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::solve_uhf;
use ndarray::Array2;
use rayon::prelude::*;
use serde::Deserialize;
use std::fs;
use std::path::PathBuf;

const FD_TOL: f64 = 1e-5;
const PYSCF_TOL: f64 = 1e-6;

const H2O: &str = "3\nH2O\nO 0 0 0\nH 0 0.7572 0.5868\nH 0 -0.7572 0.5868\n";
const NH2: &str = "3\nNH2\nN 0 0 0.1414\nH 0 0.8067 -0.4950\nH 0 -0.8067 -0.4950\n";
const BASIS: &str = "6-31g";

fn cfg(xc: &str) -> RhfConfig {
    RhfConfig::default()
        .with_dft_grid(AtomicGridConfig::default())
        .with_xc(xc)
        .with_df_j_aux(String::new())
        .with_df_k_aux(String::new())
        .with_energy_conv(1e-11)
        .with_density_conv(1e-9)
        .with_max_iter(300)
}

fn scf(mol: &Molecule, xc: &str, c: &RhfConfig) -> ScfResult {
    let bs = basis::bundled(BASIS).unwrap();
    let prep = PreparedBasis::new(mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();
    let r = if mol.multiplicity == 1 {
        solve_rhf(&ctx, mol, &prep, op, &bounds, c)
    } else {
        solve_uhf(&ctx, mol, &prep, &bounds, c)
    }
    .unwrap();
    assert!(r.converged, "{xc} SCF did not converge");
    r
}

fn analytic(mol: &Molecule, xc: &str) -> (f64, Array2<f64>, ScfResult) {
    let bs = basis::bundled(BASIS).unwrap();
    let prep = PreparedBasis::new(mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let c = cfg(xc);
    let res = scf(mol, xc, &c);
    let grids = KsGradGrids::from_config(&c);
    let g = if mol.multiplicity == 1 {
        ks_gradient_closed_with_exchange_on_grids(
            mol, &prep, &bs, op, &bounds, xc, &res, None, None, &grids,
        )
    } else {
        ks_gradient_uks_with_exchange_on_grids(
            mol, &prep, &bs, op, &bounds, xc, &res, None, None, &grids,
        )
    }
    .unwrap();
    (res.energy, g, res)
}

fn fd(mol: &Molecule, xc: &str, res0: &ScfResult, delta: f64) -> Array2<f64> {
    let mut seeded = cfg(xc);
    // Spin-summed (UKS splits it D/2 per spin and reconverges).
    seeded.init_guess_density = Some(res0.density_total.clone());
    seeded.use_sad_guess = false;
    let n = mol.atoms.len();
    let pairs: Vec<(usize, usize)> = (0..n).flat_map(|a| (0..3).map(move |c| (a, c))).collect();
    let vals: Vec<f64> = pairs
        .par_iter()
        .map(|&(a, c)| {
            let e = |s: f64| {
                let mut m = mol.clone();
                match c {
                    0 => m.atoms[a].x += s * delta,
                    1 => m.atoms[a].y += s * delta,
                    _ => m.atoms[a].zpos += s * delta,
                }
                scf(&m, xc, &seeded).energy
            };
            (e(1.0) - e(-1.0)) / (2.0 * delta)
        })
        .collect();
    Array2::from_shape_vec((n, 3), vals).unwrap()
}

fn max_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

#[derive(Deserialize)]
struct GradRef {
    e_total: f64,
    grad: Vec<[f64; 3]>,
    converged: bool,
}

fn pyscf_grad(name: &str) -> GradRef {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../testdata/reference")
        .join(name);
    let r: GradRef = serde_json::from_str(&fs::read_to_string(&p).unwrap_or_else(|e| {
        panic!("{name}: {e}; regenerate with scripts/gen_pyscf_hybrid_mgga_refs.py")
    }))
    .unwrap();
    assert!(r.converged, "{name}: PySCF did not converge");
    r
}

fn vs_pyscf(label: &str, xyz: &str, mult: usize, xc: &str, ref_file: &str) {
    let mol = Molecule::parse_xyz(xyz, 0, mult).unwrap();
    let (e, g, _) = analytic(&mol, xc);
    let r = pyscf_grad(ref_file);
    let gr = Array2::from_shape_vec(
        (r.grad.len(), 3),
        r.grad.iter().flatten().copied().collect(),
    )
    .unwrap();
    let d = max_diff(&g, &gr);
    eprintln!(
        "[{label}/{xc}] E ferric {e:.10} PySCF {:.10} (diff {:.1e}); max|g - g_PySCF| = {d:.2e}",
        r.e_total,
        (e - r.e_total).abs()
    );
    assert!(d < PYSCF_TOL, "{label}/{xc}: gradient vs PySCF {d:.3e}");
}

fn vs_fd(label: &str, xyz: &str, mult: usize, xc: &str) {
    let mol = Molecule::parse_xyz(xyz, 0, mult).unwrap();
    let (_, g, res) = analytic(&mol, xc);
    let g_fd = fd(&mol, xc, &res, 1e-3);
    let d = max_diff(&g, &g_fd);
    eprintln!("[{label}/{xc}] max|g - g_FD| = {d:.2e}");
    assert!(d < FD_TOL, "{label}/{xc}: gradient vs FD {d:.3e}");
}

#[test]
fn tpssh_h2o_gradient_vs_pyscf() {
    vs_pyscf("H2O", H2O, 1, "TPSSh", "h2o_6-31g_tpssh_grad.json");
}

#[test]
fn m062x_h2o_gradient_vs_pyscf() {
    vs_pyscf("H2O", H2O, 1, "M06-2X", "h2o_6-31g_m062x_grad.json");
}

#[test]
fn tpssh_nh2_uks_gradient_vs_pyscf() {
    vs_pyscf("NH2", NH2, 2, "TPSSh", "nh2_6-31g_tpssh_grad.json");
}

#[test]
fn m062x_nh2_uks_gradient_vs_pyscf() {
    vs_pyscf("NH2", NH2, 2, "M06-2X", "nh2_6-31g_m062x_grad.json");
}

#[test]
fn tpssh_h2o_gradient_vs_fd() {
    vs_fd("H2O", H2O, 1, "TPSSh");
}

#[test]
fn m062x_nh2_uks_gradient_vs_fd() {
    vs_fd("NH2", NH2, 2, "M06-2X");
}
