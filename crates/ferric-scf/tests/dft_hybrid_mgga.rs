//! Hybrid meta-GGA (TPSSh, M06-2X, SCAN0, r2SCAN0, wB97M-V) SCF energies.
//!
//! A hybrid meta-GGA is the τ-dependent semilocal path of the pure meta-GGAs
//! (`dft_scan.rs`) plus the exact-exchange fraction libxc reports, applied
//! through the same `k_mix` as the hybrid GGAs. Nothing in the SCF is
//! functional-specific, so the evidence is:
//!
//! * **Anchor (fixed density)**: at a converged TPSSh density `D`, the SCF
//!   energy equals `tr(Dh) + ½tr(DJ) − (a/4)tr(DK) + E_xc^{TPSSh}[D] + V_nn`
//!   with `a = 0.10` and J/K/E_xc rebuilt independently of the SCF loop
//!   (exact four-centre J/K, `KsXc::add_xc` on the SCF's grid), and the
//!   semilocal part decomposes as `E_xc^{TPSSh} = E_xc^{TPSS} − a·E_x^{TPSS}`
//!   (libxc's `0.9·TPSS_X + TPSS_C`). The same reconstruction with `a = 1`
//!   and no XC is checked on HF first, which pins the J/K conventions. The
//!   bitwise trivial limit (SCAN0 with `_exx = 0` IS SCAN) is in
//!   `ferric_dft::libxc`'s unit tests.
//! * **PySCF**: RKS (H2O), UKS and ROKS (NH2 doublet) energies at cc-pVDZ on
//!   the flat (75, 110) Becke grid with EXACT four-centre J and K in both
//!   codes, from `scripts/gen_pyscf_hybrid_mgga_refs.py`. Exact J/K, not
//!   RI: with def2-universal-jkfit both codes' fitted energies differ by
//!   ~3e-7 Ha on H2O for PBE0 and pure TPSS alike (an RI-J/RI-K convention
//!   difference, not a functional one), which would sit above this bar.
//!
//! Bar: `TOL = 1e-8` Ha, the meta-GGA SCF energy noise floor (`dft_scan.rs`
//! ENERGY_CONV: the τ Fock parks the energy near 1e-8). Measured on these
//! systems: 1e-13..1e-10. A wrong exact-exchange fraction moves the energy by
//! mHa (TPSSh `0.1·(E_x^HF − E_x^TPSS)` is 7.6e-3 Ha on H2O/6-31G).

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_dft::grid::AtomicGridConfig;
use ferric_dft::ks::KsXc;
use ferric_dft::xc_trait::XcContribution;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::oneelectron;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::{build_jk, solve_rhf, RhfConfig};
use ferric_scf::rohf::solve_rohf;
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::solve_uhf;
use ndarray::Array2;
use serde::Deserialize;
use std::fs;
use std::path::PathBuf;

const TOL: f64 = 1e-8;

/// wB97M-V bar. RI-J (f64) and RI-K with the same auxiliary basis in both
/// codes plus VV10 on the (50, 50) NLC grid; measured 5.9e-8 Ha. The fitted
/// hybrid GGAs and pure TPSS agree with PySCF at matched fitting to ~3e-7
/// with ferric's f32 RI-J default (`dft_b3lyp.rs` / `dft_wb97xv.rs` use 5e-5);
/// 1e-6 sits above the measured fitted agreement and two to three decades
/// below what a lost exchange range (1.1 Ha with c_sr zeroed) or τ Fock term
/// (5.6e-4 Ha) moves this energy.
const RSH_TOL: f64 = 1e-6;

/// SCF thresholds two decades inside `TOL` (energy) so the comparison
/// measures the functional, not the stopping point.
const ENERGY_CONV: f64 = 1e-10;
const DENSITY_CONV: f64 = 1e-9;

const H2O: &str = "3\nH2O\nO 0 0 0\nH 0 0.7572 0.5868\nH 0 -0.7572 0.5868\n";
const NH2: &str = "3\nNH2\nN 0 0 0.1414\nH 0 0.8067 -0.4950\nH 0 -0.8067 -0.4950\n";

#[derive(Deserialize)]
struct Ref {
    e_total: f64,
    converged: bool,
}

fn load_ref(name: &str) -> Ref {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../testdata/reference")
        .join(name);
    let r: Ref = serde_json::from_str(&fs::read_to_string(&p).unwrap_or_else(|e| {
        panic!("{name}: {e}; regenerate with scripts/gen_pyscf_hybrid_mgga_refs.py")
    }))
    .unwrap();
    assert!(r.converged, "PySCF reference {name} did not converge");
    r
}

/// The PySCF-matched configuration: flat (75, 110) grid, exact four-centre
/// J and K (an empty aux name turns density fitting off).
fn cfg(xc: &str) -> RhfConfig {
    RhfConfig {
        dft_grid: Some(AtomicGridConfig::default()),
        xc: Some(xc.into()),
        df_j_aux: Some(String::new()),
        df_k_aux: Some(String::new()),
        energy_conv: ENERGY_CONV,
        density_conv: DENSITY_CONV,
        max_iter: 300,
        ..Default::default()
    }
}

#[derive(Clone, Copy)]
enum Reference {
    Rks,
    Uks,
    Roks,
}

fn energy(xyz: &str, mult: usize, basis_name: &str, xc: &str, kind: Reference) -> f64 {
    let mol = Molecule::parse_xyz(xyz, 0, mult).unwrap();
    let bs = basis::bundled(basis_name).unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();
    let c = cfg(xc);
    let res = match kind {
        Reference::Rks => solve_rhf(&ctx, &mol, &prep, op, &bounds, &c),
        Reference::Uks => solve_uhf(&ctx, &mol, &prep, &bounds, &c),
        Reference::Roks => solve_rohf(&ctx, &mol, &prep, op, &bounds, &c),
    }
    .unwrap();
    assert!(res.converged, "{xc} SCF did not converge");
    res.energy
}

fn check(label: &str, xyz: &str, mult: usize, xc: &str, kind: Reference, ref_file: &str) {
    let e = energy(xyz, mult, "cc-pvdz", xc, kind);
    let r = load_ref(ref_file);
    let err = (e - r.e_total).abs();
    eprintln!(
        "[{label}/{xc}] ferric {e:.10}  PySCF {:.10}  err {err:.2e}",
        r.e_total
    );
    assert!(err < TOL, "{label}/{xc}: |ferric - PySCF| = {err:.3e}");
}

#[test]
fn tpssh_h2o_rks_vs_pyscf() {
    check(
        "H2O",
        H2O,
        1,
        "TPSSh",
        Reference::Rks,
        "h2o_cc-pvdz_tpssh.json",
    );
}

#[test]
fn tpssh_nh2_uks_vs_pyscf() {
    check(
        "NH2",
        NH2,
        2,
        "TPSSh",
        Reference::Uks,
        "nh2_cc-pvdz_tpssh.json",
    );
}

#[test]
fn tpssh_nh2_roks_vs_pyscf() {
    check(
        "NH2",
        NH2,
        2,
        "TPSSh",
        Reference::Roks,
        "nh2_cc-pvdz_tpssh_roks.json",
    );
}

#[test]
fn m062x_h2o_rks_vs_pyscf() {
    check(
        "H2O",
        H2O,
        1,
        "M06-2X",
        Reference::Rks,
        "h2o_cc-pvdz_m062x.json",
    );
}

#[test]
fn m062x_nh2_uks_vs_pyscf() {
    check(
        "NH2",
        NH2,
        2,
        "M06-2X",
        Reference::Uks,
        "nh2_cc-pvdz_m062x.json",
    );
}

#[test]
fn scan0_h2o_rks_vs_pyscf() {
    check(
        "H2O",
        H2O,
        1,
        "SCAN0",
        Reference::Rks,
        "h2o_cc-pvdz_scan0.json",
    );
}

#[test]
fn scan0_nh2_uks_vs_pyscf() {
    check(
        "NH2",
        NH2,
        2,
        "SCAN0",
        Reference::Uks,
        "nh2_cc-pvdz_scan0.json",
    );
}

/// r2SCAN0 (libxc HYB_MGGA_XC_R2SCAN0, 25% exact exchange; libxc >= 6.1).
/// One combined libxc component, so the exact-exchange fraction is the one
/// `xc_hyb_exx_coef` reports for it.
#[test]
fn r2scan0_h2o_rks_vs_pyscf() {
    check(
        "H2O",
        H2O,
        1,
        "r2SCAN0",
        Reference::Rks,
        "h2o_cc-pvdz_r2scan0.json",
    );
}

#[test]
fn r2scan0_nh2_uks_vs_pyscf() {
    check(
        "NH2",
        NH2,
        2,
        "r2SCAN0",
        Reference::Uks,
        "nh2_cc-pvdz_r2scan0.json",
    );
}

/// ωB97M-V: range-separated (erf/erfc exact exchange, ω = 0.3) hybrid
/// meta-GGA plus VV10 on the (50, 50) NLC grid, through the paths the GGA
/// ωB97X-V uses. ferric builds range-separated exchange only by RI-K, so this
/// case fits J and K with def2-universal-jkfit in both codes, with ferric's
/// RI-J in f64 (the f32 default is a separate, documented approximation).
/// Bar: `RSH_TOL`, set from the measured agreement (see the constant).
#[test]
fn wb97m_v_h2o_rks_vs_pyscf() {
    let mol = Molecule::parse_xyz(H2O, 0, 1).unwrap();
    let bs = basis::bundled("cc-pvdz").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let c = RhfConfig {
        df_j_aux: Some("def2-universal-jkfit".into()),
        df_k_aux: Some("def2-universal-jkfit".into()),
        ri_j_precision: Some(ferric_scf::df_j::RiJPrecision::F64),
        ..cfg("wB97M-V")
    };
    let res = solve_rhf(&ParallelContext::default(), &mol, &prep, op, &bounds, &c).unwrap();
    assert!(res.converged);
    let r = load_ref("h2o_cc-pvdz_wb97m_v.json");
    let err = (res.energy - r.e_total).abs();
    eprintln!(
        "[H2O/wB97M-V] ferric {:.10}  PySCF {:.10}  err {err:.2e}",
        res.energy, r.e_total
    );
    assert!(err < RSH_TOL, "H2O/wB97M-V: |ferric - PySCF| = {err:.3e}");
}

/// Exactness anchor at fixed density (see the module doc). Exact four-centre
/// J/K so that `build_jk` rebuilds the SCF's own two-electron terms.
#[test]
fn tpssh_energy_decomposes_at_fixed_density() {
    let mol = Molecule::parse_xyz(H2O, 0, 1).unwrap();
    let bs = basis::bundled("6-31g").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();
    let exact = |xc: Option<&str>| RhfConfig {
        dft_grid: Some(AtomicGridConfig::default()),
        xc: xc.map(Into::into),
        df_j_aux: Some(String::new()),
        df_k_aux: Some(String::new()),
        energy_conv: 1e-10,
        density_conv: 1e-9,
        max_iter: 300,
        ..Default::default()
    };
    let h = oneelectron::hcore(&prep);
    let vnn = mol.nuclear_repulsion();
    let two_e = |d: &Array2<f64>| {
        let n = d.nrows();
        let (mut j, mut k) = (Array2::zeros((n, n)), Array2::zeros((n, n)));
        build_jk(&ctx, &prep, &bounds, 1e-14, d, &mut j, &mut k).unwrap();
        ((d * &h).sum(), 0.5 * (d * &j).sum(), (d * &k).sum())
    };
    let grid = AtomicGridConfig::default();
    let nlc = AtomicGridConfig::default_nlc();
    let exc = |name: &str, d: &Array2<f64>| {
        let ks = KsXc::new(&mol, &bs, name, &grid, &nlc).unwrap();
        let mut f = Array2::zeros(d.raw_dim());
        ks.add_xc(d, &mut f)
    };

    // HF calibration: a = 1, no XC.
    let hf = solve_rhf(&ctx, &mol, &prep, op, &bounds, &exact(None)).unwrap();
    assert!(hf.converged);
    let (e1, ej, tr_dk) = two_e(hf.density_r());
    let e_hf = e1 + ej - 0.25 * tr_dk + vnn;
    let d_hf = (e_hf - hf.energy).abs();
    eprintln!(
        "HF: SCF {:.12} rebuilt {e_hf:.12} diff {d_hf:.2e}",
        hf.energy
    );
    assert!(d_hf < 1e-8, "HF energy reconstruction off by {d_hf:.3e}");

    let a = 0.10;
    let res = solve_rhf(&ctx, &mol, &prep, op, &bounds, &exact(Some("TPSSh"))).unwrap();
    assert!(res.converged);
    let d = res.density_r();
    let (e1, ej, tr_dk) = two_e(d);
    let (e_tpssh, e_tpss, e_x) = (exc("TPSSh", d), exc("TPSS", d), exc("MGGA_X_TPSS", d));
    let rebuilt = e1 + ej - 0.25 * a * tr_dk + e_tpssh + vnn;
    let d_scf = (rebuilt - res.energy).abs();
    let d_sl = (e_tpssh - (e_tpss - a * e_x)).abs();
    eprintln!(
        "TPSSh: SCF {:.12} rebuilt {rebuilt:.12} diff {d_scf:.2e}; \
         E_xc(TPSSh) - [E_xc(TPSS) - a E_x(TPSS)] = {d_sl:.2e}; a(E_x^HF - E_x^TPSS) = {:.6e}",
        res.energy,
        a * (-0.25 * tr_dk - e_x)
    );
    assert!(
        d_scf < 1e-8,
        "TPSSh energy reconstruction off by {d_scf:.3e}"
    );
    assert!(
        d_sl < 1e-10,
        "TPSSh semilocal decomposition off by {d_sl:.3e}"
    );
}

// ── Response consumers: internal stability and Newton ─────────────────────
//
// The orbital Hessian of a hybrid meta-GGA is the meta-GGA one (τ f_xc kernel,
// `mgga_response_vs_pyscf.rs`) plus `a`·K response. References: PySCF
// `newton_ah.gen_g_hop_{rhf,uhf}` dense spectra on the same grid and exact
// J/K (`gen_pyscf_hybrid_mgga_refs.py stability`). Bar `TOL_LAMBDA = 1e-8`,
// the r2SCAN bar (`mgga_response_vs_pyscf.rs`, measured 1.5e-10 there): λ_min
// is first order in the Davidson residual (2e-8 here) and in the SCF density
// error. Measured 6.3e-9 (RKS H2O) and 6.6e-10 Ha (UKS NH2). Dropping the
// 0.1·K response from the Newton/stability Hessian moves λ_min by 3.8e-2 and
// 6.2e-2 Ha (mutation).

const TOL_LAMBDA: f64 = 1e-8;

#[derive(Deserialize)]
struct StabRef {
    e_total: f64,
    hessian_lowest: Vec<f64>,
    converged: bool,
}

fn stability_case(xyz: &str, mult: usize, ref_file: &str) {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../testdata/reference")
        .join(ref_file);
    let r: StabRef = serde_json::from_str(&fs::read_to_string(&p).unwrap()).unwrap();
    assert!(r.converged);
    let mol = Molecule::parse_xyz(xyz, 0, mult).unwrap();
    let bs = basis::bundled("cc-pvdz").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();
    let c = RhfConfig {
        check_stability: true,
        energy_conv: 1e-11,
        density_conv: 1e-10,
        ..cfg("TPSSh")
    };
    let res = if mult == 1 {
        solve_rhf(&ctx, &mol, &prep, op, &bounds, &c)
    } else {
        solve_uhf(&ctx, &mol, &prep, &bounds, &c)
    }
    .unwrap();
    assert!(res.converged);
    let st = res.stability.as_ref().expect("check_stability verdict");
    eprintln!("[{ref_file}] {}", st.summary());
    let de = (res.energy - r.e_total).abs();
    let dl = (st.lowest_eigenvalue - r.hessian_lowest[0]).abs();
    eprintln!(
        "[{ref_file}] E diff {de:.2e}; λ_min ferric {:+.12e} PySCF {:+.12e} diff {dl:.2e}",
        st.lowest_eigenvalue, r.hessian_lowest[0]
    );
    assert!(de < TOL, "{ref_file}: energy diff {de:.3e}");
    assert!(st.converged, "{ref_file}: stability Davidson not converged");
    assert!(dl < TOL_LAMBDA, "{ref_file}: λ_min diff {dl:.3e}");
}

#[test]
fn tpssh_h2o_rks_stability_matches_pyscf() {
    stability_case(H2O, 1, "h2o_cc-pvdz_tpssh_stability.json");
}

#[test]
fn tpssh_nh2_uks_stability_matches_pyscf() {
    stability_case(NH2, 2, "nh2_cc-pvdz_tpssh_stability.json");
}

/// Newton (`newton_trigger`) lands on the DIIS energy for RKS and UKS TPSSh
/// (measured 0 and 2e-14 Ha). The energy comparison cannot see a slightly
/// wrong Hessian (Newton still reaches the same stationary point); it pins
/// that the hybrid meta-GGA Newton path runs and converges. With the K
/// response dropped from the Hessian the Newton runs do not converge
/// (mutation).
#[test]
fn tpssh_newton_matches_diis() {
    for (xyz, mult) in [(H2O, 1), (NH2, 2)] {
        let mol = Molecule::parse_xyz(xyz, 0, mult).unwrap();
        let bs = basis::bundled("cc-pvdz").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let ctx = ParallelContext::default();
        let run = |c: &RhfConfig| {
            if mult == 1 {
                solve_rhf(&ctx, &mol, &prep, op, &bounds, c)
            } else {
                solve_uhf(&ctx, &mol, &prep, &bounds, c)
            }
            .unwrap()
        };
        let diis = run(&cfg("TPSSh"));
        let newton = run(&RhfConfig {
            newton_trigger: 1e-2,
            ..cfg("TPSSh")
        });
        let d = (diis.energy - newton.energy).abs();
        eprintln!(
            "mult {mult}: DIIS {} it, Newton {} it, |dE| {d:.2e}",
            diis.iterations, newton.iterations
        );
        assert!(diis.converged && newton.converged);
        assert!(d < TOL, "mult {mult}: Newton vs DIIS {d:.3e}");
    }
}
