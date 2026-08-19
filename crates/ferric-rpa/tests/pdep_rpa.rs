use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_rpa::config::{QuadratureConfig, QuadratureScheme};
use ferric_rpa::{run_pdep_rpa, PdepRpaConfig};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::ScfResult;
use ferric_scf::screening::SchwarzBounds;
use serde::Deserialize;

#[derive(Deserialize)]
struct RpaRef {
    e_corr: f64,
}

fn load_ref(path: &str) -> f64 {
    let s = std::fs::read_to_string(path).unwrap_or_else(|_| panic!("missing ref: {path}"));
    let r: RpaRef = serde_json::from_str(&s).unwrap();
    r.e_corr
}

fn setup(
    xyz: &str,
    obs_name: &str,
    dfbs_name: &str,
) -> (Molecule, PreparedBasis, PreparedBasis, Operator, ScfResult) {
    let ctx = ParallelContext::default();
    let mol = Molecule::load_xyz(xyz).unwrap();
    let obs_bs = basis::bundled(obs_name).unwrap();
    let dfbs_bs = basis::bundled(dfbs_name).unwrap();
    let op = Operator::coulomb();
    let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
    let dfbs = PreparedBasis::new(&mol, &dfbs_bs).unwrap();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default()).unwrap();
    (mol, obs, dfbs, op, rhf)
}

fn pyscf_compat_config(n_quad: usize) -> PdepRpaConfig {
    PdepRpaConfig {
        quadrature: QuadratureConfig {
            scheme: QuadratureScheme::GaussLegendre,
            n_points: n_quad,
            u0: 0.5,
        },
        frozen_core: 0,
        // Disable PDEP truncation: compare full-basis dielectric to PySCF's full RI-RPA.
        trunc_thresh: 0.0,
        eigensolver_conv_thresh: 1e-10,
        ..Default::default()
    }
}

#[test]
fn h2_sto3g_rpa_energy_sign() {
    let (mol, obs, dfbs, op, rhf) = setup("../../testdata/molecules/h2.xyz", "sto-3g", "sto-3g");
    let result = run_pdep_rpa(&mol, &obs, &dfbs, op, &rhf, &PdepRpaConfig::default()).unwrap();
    assert!(result.e_rpa < 0.0, "E_c should be negative, got {}", result.e_rpa);
    assert!(result.n_eigenpotentials > 0);
}

#[test]
fn energy_only_run_does_not_materialize_inv_dielectric_freq() {
    // M9: the default (energy-only) config must NOT build the nquad × M²
    // inverse-dielectric stack — it is only consumed by GW/BSE/property paths.
    let (mol, obs, dfbs, op, rhf) = setup("../../testdata/molecules/h2.xyz", "sto-3g", "sto-3g");
    let cfg = PdepRpaConfig::default();
    assert!(!cfg.need_inv_dielectric_freq, "default config must be energy-only");
    let result = run_pdep_rpa(&mol, &obs, &dfbs, op, &rhf, &cfg).unwrap();
    assert!(
        result.inv_dielectric_freq.is_none(),
        "energy-only run must leave inv_dielectric_freq None (never allocate the stack)"
    );
    assert!(result.e_rpa < 0.0);
}

#[test]
fn inv_dielectric_freq_built_when_flag_set() {
    // M9: setting the flag materializes the per-frequency stack for GW/property
    // consumers, one (M×M) matrix per quadrature point.
    let (mol, obs, dfbs, op, rhf) = setup("../../testdata/molecules/h2.xyz", "sto-3g", "sto-3g");
    let cfg = PdepRpaConfig {
        need_inv_dielectric_freq: true,
        need_eigenvalues_freq: true,
        ..PdepRpaConfig::default()
    };
    let result = run_pdep_rpa(&mol, &obs, &dfbs, op, &rhf, &cfg).unwrap();
    let winv = result
        .inv_dielectric_freq
        .as_ref()
        .expect("flag set → inv_dielectric_freq must be Some");
    assert_eq!(
        winv.len(),
        result.quad_freqs.len(),
        "one inverse-dielectric matrix per quadrature frequency"
    );
    let m = result.n_eigenpotentials;
    assert_eq!(winv[0].shape(), &[m, m], "each matrix is M×M in the PDEP basis");
}

#[test]
fn h2_sto3g_pdep_rpa_matches_pyscf() {
    // Reference uses STO-3G/STO-3G-RI to match ferric's RI basis exactly.
    let e_ref = load_ref("../../testdata/reference/h2_sto-3g_rpa.json");
    let (mol, obs, dfbs, op, rhf) = setup("../../testdata/molecules/h2.xyz", "sto-3g", "sto-3g");
    let cfg = pyscf_compat_config(40);
    let result = run_pdep_rpa(&mol, &obs, &dfbs, op, &rhf, &cfg).unwrap();
    let diff = (result.e_rpa - e_ref).abs();
    println!("H2/STO-3G  ferric={:.10} pyscf={:.10}  diff={:.2e}", result.e_rpa, e_ref, diff);
    assert!(diff < 1e-6, "H2/STO-3G PDEP-RPA differs by {:.2e}", diff);
}

#[test]
fn h2o_cc_pvdz_pdep_rpa_matches_pyscf() {
    let e_ref = load_ref("../../testdata/reference/h2o_cc-pvdz_rpa.json");
    let (mol, obs, dfbs, op, rhf) =
        setup("../../testdata/molecules/water.xyz", "cc-pvdz", "cc-pvdz-ri");
    let cfg = pyscf_compat_config(40);
    let result = run_pdep_rpa(&mol, &obs, &dfbs, op, &rhf, &cfg).unwrap();
    let diff = (result.e_rpa - e_ref).abs();
    println!("H2O/cc-pVDZ ferric={:.10} pyscf={:.10}  diff={:.2e}", result.e_rpa, e_ref, diff);
    assert!(diff < 1e-6, "H2O/cc-pVDZ PDEP-RPA differs by {:.2e}", diff);
}

#[test]
fn h2o_aug_cc_pvdz_pdep_rpa_matches_pyscf() {
    let e_ref = load_ref("../../testdata/reference/h2o_aug-cc-pvdz_rpa.json");
    let (mol, obs, dfbs, op, rhf) = setup(
        "../../testdata/molecules/water.xyz",
        "aug-cc-pvdz",
        "aug-cc-pvdz-rifit",
    );
    let cfg = pyscf_compat_config(40);
    let result = run_pdep_rpa(&mol, &obs, &dfbs, op, &rhf, &cfg).unwrap();
    let diff = (result.e_rpa - e_ref).abs();
    println!("H2O/aug-cc-pVDZ ferric={:.10} pyscf={:.10}  diff={:.2e}", result.e_rpa, e_ref, diff);
    assert!(diff < 1e-6, "H2O/aug-cc-pVDZ PDEP-RPA differs by {:.2e}", diff);
}

#[test]
fn h2o_aug_cc_pvtz_pdep_rpa_matches_pyscf() {
    let e_ref = load_ref("../../testdata/reference/h2o_aug-cc-pvtz_rpa.json");
    let (mol, obs, dfbs, op, rhf) = setup(
        "../../testdata/molecules/water.xyz",
        "aug-cc-pvtz",
        "aug-cc-pvtz-rifit",
    );
    let cfg = pyscf_compat_config(40);
    let result = run_pdep_rpa(&mol, &obs, &dfbs, op, &rhf, &cfg).unwrap();
    let diff = (result.e_rpa - e_ref).abs();
    println!("H2O/aug-cc-pVTZ ferric={:.10} pyscf={:.10}  diff={:.2e}", result.e_rpa, e_ref, diff);
    assert!(diff < 1e-6, "H2O/aug-cc-pVTZ PDEP-RPA differs by {:.2e}", diff);
}

#[test]
fn h2_sto3g_pdep_rpa_vs_ri_drpa() {
    // Sanity: PDEP-RPA == RI-dRPA when no truncation is applied (PDEP keeps all naux modes).
    let (mol, obs, dfbs, op, rhf) = setup("../../testdata/molecules/h2.xyz", "sto-3g", "sto-3g");
    let cfg = PdepRpaConfig {
        trunc_thresh: 0.0,
        eigensolver_conv_thresh: 1e-10,
        run_diagnostics: true,
        ..Default::default()
    };
    let result = run_pdep_rpa(&mol, &obs, &dfbs, op, &rhf, &cfg).unwrap();
    let e_diag = result.e_rpa_dft_diag.expect("diagnostic should be present");
    let diff = (result.e_rpa - e_diag).abs();
    assert!(diff < 1e-8, "PDEP-RPA vs RI-dRPA differ by {:.2e}", diff);
}

#[test]
fn h2o_cc_pvdz_quadrature_convergence() {
    // 20 vs 40 GL points should both match PySCF to ≤1e-5.
    let e_ref = load_ref("../../testdata/reference/h2o_cc-pvdz_rpa.json");
    let (mol, obs, dfbs, op, rhf) =
        setup("../../testdata/molecules/water.xyz", "cc-pvdz", "cc-pvdz-ri");
    for &n in &[20usize, 40] {
        let cfg = pyscf_compat_config(n);
        let result = run_pdep_rpa(&mol, &obs, &dfbs, op, &rhf, &cfg).unwrap();
        let diff = (result.e_rpa - e_ref).abs();
        println!("H2O/cc-pVDZ n_quad={} diff={:.2e}", n, diff);
        assert!(diff < 1e-5, "n_quad={}: |ΔE| = {:.2e} > 1e-5", n, diff);
    }
}

#[test]
fn h2o_cc_pvdz_pdep_truncation_convergence() {
    // PDEP test: can we recover full RPA energy with M ≈ 3 × N_atoms eigenpotentials?
    // H2O has 3 atoms → target M ≈ 9. Full naux=84.
    let e_ref = load_ref("../../testdata/reference/h2o_cc-pvdz_rpa.json");
    let (mol, obs, dfbs, op, rhf) =
        setup("../../testdata/molecules/water.xyz", "cc-pvdz", "cc-pvdz-ri");

    println!("\nPDEP truncation study (H2O/cc-pVDZ, naux=84):");
    println!("PySCF (full):      {:.10}", e_ref);

    for thresh in &[1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-10] {
        let mut cfg = pyscf_compat_config(40);
        cfg.trunc_thresh = *thresh;
        cfg.eigensolver_conv_thresh = 1e-10;
        let result = run_pdep_rpa(&mol, &obs, &dfbs, op, &rhf, &cfg).unwrap();
        let diff = result.e_rpa - e_ref;
        println!(
            "  trunc={:.0e}: M={:3} E_c={:.10}  diff={:+.2e}  λ_max={:.3} λ_min={:.3e}",
            thresh, result.n_eigenpotentials, result.e_rpa, diff,
            result.eigenvalues_static.first().copied().unwrap_or(0.0),
            result.eigenvalues_static.last().copied().unwrap_or(0.0),
        );
    }
}

#[test]
#[ignore = "slow: 5 full/truncated PDEP-RPA solves at aug-cc-pVTZ (naux=198), \
            pure timing printout with no assertions -- see benzene_cc_pvdz_timing \
            below for the same pattern already ignored at a smaller basis"]
fn h2o_aug_cc_pvtz_timing_comparison() {
    use std::time::Instant;
    let (mol, obs, dfbs, op, rhf) = setup(
        "../../testdata/molecules/water.xyz",
        "aug-cc-pvtz",
        "aug-cc-pvtz-rifit",
    );
    println!("\nH2O/aug-cc-pVTZ timing (naux=198, 40 GL points):");

    // Full RI-RPA (no truncation)
    let mut cfg_full = pyscf_compat_config(40);
    cfg_full.trunc_thresh = 0.0;
    let t0 = Instant::now();
    let r_full = run_pdep_rpa(&mol, &obs, &dfbs, op, &rhf, &cfg_full).unwrap();
    let dt_full = t0.elapsed();
    println!("  full (M=naux={}):  E_c={:.10}  t={:.2}s",
        r_full.n_eigenpotentials, r_full.e_rpa, dt_full.as_secs_f64());

    // PDEP truncated
    for &th in &[1e-1, 1e-2, 1e-3, 1e-4] {
        let mut cfg = pyscf_compat_config(40);
        cfg.trunc_thresh = th;
        let t0 = Instant::now();
        let r = run_pdep_rpa(&mol, &obs, &dfbs, op, &rhf, &cfg).unwrap();
        let dt = t0.elapsed();
        let diff = (r.e_rpa - r_full.e_rpa).abs();
        println!("  trunc={:.0e}: M={:3} E_c={:.10}  diff_vs_full={:.2e}  t={:.2}s ({:.0}% of full)",
            th, r.n_eigenpotentials, r.e_rpa, diff,
            dt.as_secs_f64(), 100.0 * dt.as_secs_f64() / dt_full.as_secs_f64());
    }
}

#[test]
#[ignore]
fn benzene_cc_pvdz_timing() {
    use std::time::Instant;
    let (mol, obs, dfbs, op, rhf) = setup(
        "../../testdata/molecules/benzene.xyz",
        "cc-pvdz",
        "cc-pvdz-ri",
    );
    println!("\nBenzene/cc-pVDZ PDEP-RPA timing (40 GL points):");

    let mut cfg = pyscf_compat_config(40);
    cfg.trunc_thresh = 1e-4;
    cfg.frozen_core = 6;

    let t0 = Instant::now();
    let r = run_pdep_rpa(&mol, &obs, &dfbs, op, &rhf, &cfg).unwrap();
    let dt = t0.elapsed();
    println!("  trunc=1e-4 frozen_core=6: M={} E_c={:.10} t={:.2}s",
        r.n_eigenpotentials, r.e_rpa, dt.as_secs_f64());
}

/// Lanczos-vs-Davidson agreement under PDEP truncation: both eigensolvers
/// do a full-rank eigensolve then post-filter by |λ−1| > thresh. The
/// energies must agree at every truncation threshold.
#[test]
fn lanczos_matches_davidson_under_truncation() {
    use ferric_rpa::config::Eigensolver;

    let (mol, obs, dfbs, op, rhf) = setup(
        "../../testdata/molecules/water.xyz",
        "aug-cc-pvtz",
        "aug-cc-pvtz-rifit",
    );

    // Full-rank reference at τ=0 for the vacuousness check.
    let mut cfg_full = pyscf_compat_config(40);
    cfg_full.trunc_thresh = 0.0;
    cfg_full.eigensolver = Eigensolver::Davidson;
    let r_full = run_pdep_rpa(&mol, &obs, &dfbs, op, &rhf, &cfg_full).unwrap();

    for &thresh in &[1e-2, 1e-3, 1e-4] {
        // Davidson reference: full eigensolve + post-filter.
        let mut cfg_ref = pyscf_compat_config(40);
        cfg_ref.trunc_thresh = thresh;
        cfg_ref.eigensolver = Eigensolver::Davidson;
        let r_ref = run_pdep_rpa(&mol, &obs, &dfbs, op, &rhf, &cfg_ref).unwrap();

        // Targeted Lanczos (test subject).
        let mut cfg_targeted = pyscf_compat_config(40);
        cfg_targeted.trunc_thresh = thresh;
        cfg_targeted.eigensolver = Eigensolver::Lanczos;
        let r_targeted = run_pdep_rpa(&mol, &obs, &dfbs, op, &rhf, &cfg_targeted).unwrap();

        let d = (r_targeted.e_rpa - r_ref.e_rpa).abs();
        eprintln!(
            "thresh={thresh:.0e}: Lanczos M={} E={:.10} vs Davidson M={} E={:.10} d={d:.2e}",
            r_targeted.n_eigenpotentials,
            r_targeted.e_rpa,
            r_ref.n_eigenpotentials,
            r_ref.e_rpa,
        );
        assert_eq!(
            r_targeted.n_eigenpotentials, r_ref.n_eigenpotentials,
            "Lanczos kept a different number of modes than Davidson \
             at thresh={thresh:.0e}: {} vs {}",
            r_targeted.n_eigenpotentials, r_ref.n_eigenpotentials,
        );
        assert!(
            d < 1e-7,
            "Lanczos energy disagrees with Davidson at \
             thresh={thresh:.0e}: {d:.2e} (> 1e-7)",
        );
    }

    // Sanity: truncation must actually discard modes.
    let mut cfg_trunc = pyscf_compat_config(40);
    cfg_trunc.trunc_thresh = 1e-2;
    cfg_trunc.eigensolver = Eigensolver::Lanczos;
    let r_trunc = run_pdep_rpa(&mol, &obs, &dfbs, op, &rhf, &cfg_trunc).unwrap();
    assert!(
        r_trunc.n_eigenpotentials < r_full.n_eigenpotentials,
        "truncation at 1e-2 kept all modes — test is vacuous"
    );
}

fn run_lanczos_eigensolve_benchmark(label: &str, xyz: &str, obs_name: &str, dfbs_name: &str) {
    use ferric_mp2::rimp2::{compute_rpa_intermediates, RiMp2Config};
    use ferric_rpa::lanczos;
    use ferric_rpa::sternheimer;
    use ndarray::Array2;
    use ndarray_linalg::QR;
    use std::time::Instant;

    let (mol, obs, dfbs, op, rhf) = setup(xyz, obs_name, dfbs_name);

    let mp2_cfg = RiMp2Config { frozen_core: 8, ..Default::default() };
    let inter = compute_rpa_intermediates(&mol, &obs, &dfbs, op, &rhf, &mp2_cfg).unwrap();
    let naux = inter.naux;
    let nocc = inter.nocc;
    let nvir = inter.nvir;
    let b_ov = &inter.b_ov;
    let nocc_total = inter.nocc_total;
    let first_occ = inter.first_occ;
    let eps_occ: Vec<f64> = rhf.eps_r()[first_occ..first_occ + nocc].to_vec();
    let eps_vir: Vec<f64> = rhf.eps_r()[nocc_total..nocc_total + nvir].to_vec();

    eprintln!("{label}: naux={naux} nocc={nocc} nvir={nvir} nov={}", nocc * nvir);

    let thresh = 1e-2;

    let t0 = Instant::now();
    let dense = lanczos::run_lanczos_full_rank_budgeted(
        naux, nocc * nvir,
        |v: &Array2<f64>| sternheimer::dielectric_apply(v, b_ov, &eps_occ, &eps_vir, 0.0),
        naux, None,
    ).unwrap();
    let dt_dense = t0.elapsed();

    let dense_kept: Vec<f64> = dense.eigenvalues.iter()
        .copied()
        .filter(|&lam| (lam - 1.0).abs() > thresh)
        .collect();
    eprintln!("dense: {naux} modes, {} above thresh={thresh:.0e}, {:.1}ms",
        dense_kept.len(), dt_dense.as_secs_f64() * 1e3);

    let dense_sum: f64 = dense_kept.iter().map(|&l| l - 1.0 - l.ln()).sum();

    for block_size in [256, 128, 64] {
        if block_size >= naux { continue; }
        let mut raw = Array2::<f64>::zeros((naux, block_size));
        let mut state: u64 = 0xDEAD_BEEF_CAFE_BABEu64;
        for x in raw.iter_mut() {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            *x = (state >> 33) as f64 / (1u64 << 31) as f64 - 1.0;
        }
        let (seed, _) = raw.qr().unwrap();

        let max_iter = naux / block_size + 1;
        let t0 = Instant::now();
        let lz = lanczos::run_lanczos_targeted(
            seed,
            |v: &Array2<f64>| sternheimer::dielectric_apply(v, b_ov, &eps_occ, &eps_vir, 0.0),
            thresh, max_iter, 1e-6, false,
        ).unwrap();
        let dt_lz = t0.elapsed();

        let speedup = dt_dense.as_secs_f64() / dt_lz.as_secs_f64();
        let lz_sum: f64 = lz.eigenvalues.iter().map(|&l| l - 1.0 - l.ln()).sum();
        let d = (lz_sum - dense_sum).abs();
        eprintln!(
            "targeted m={block_size:3}: {} modes, conv={}, resid={:.1e}, \
             {:.1}ms ({:.2}x vs dense), energy_d={d:.2e}",
            lz.eigenvalues.len(), lz.converged, lz.max_resid,
            dt_lz.as_secs_f64() * 1e3, speedup,
        );
    }
}

#[test]
#[ignore]
fn targeted_lanczos_benchmark_dz() {
    run_lanczos_eigensolve_benchmark(
        "C8/DZ", "../../testdata/molecules/alkane_8.xyz", "cc-pvdz", "cc-pvdz-ri",
    );
}

#[test]
#[ignore]
fn targeted_lanczos_benchmark_tz() {
    run_lanczos_eigensolve_benchmark(
        "C8/TZ", "../../testdata/molecules/alkane_8.xyz", "cc-pvtz", "cc-pvtz-rifit",
    );
}
