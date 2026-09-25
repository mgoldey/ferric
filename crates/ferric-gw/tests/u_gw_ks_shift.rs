//! Where the Kohn–Sham static shift Σ_x − v_xc enters the unrestricted GW
//! solve (`run_u_gw(.., vxc_diag)`). Cheap (OH/STO-3G UHF, 8-point
//! quadrature), so NOT ignored.
//!
//! 1. `vxc_diag = None` (UHF/ROHF reference) solves the unshifted QP
//!    equation: bit-identical to `Some(v)` with `v_p = Σ_x,p`, i.e. a shift of
//!    exactly +0.0 fed through the same solver.
//! 2. With a non-zero shift Δ_p = Σ_x,p − v_p the returned roots satisfy
//!    ε_QP − ε_mf − Δ − Σ_c(ε_QP) = 0 (Σ_c at the SHIFTED root), and differ
//!    from the post-hoc recipe (unshifted root + Δ) because Σ_c is
//!    energy-dependent. For U-COHSEX (static) ε_QP = ε_mf + Σ_c + Δ exactly.
//!
//! The v_xc here is synthetic (Σ_x plus a fixed per-MO offset): these tests
//! pin where the shift goes, not the value of v_xc. The physics anchor is
//! `validation_gw.rs` `u_g0w0_uks_pbe_*_qp` (PySCF `ugw_ac`, real UKS/PBE).
//!
//! MUTATION (run by hand): in `u_sigma.rs` `qp_per_spin_g0w0`, pass `0.0`
//! instead of `shift` to `solve_qp_for_mo` (the pre-fix behaviour) —
//! `g0w0_ks_shift_enters_the_qp_equation` fails on the residual (|resid| =
//! |Δ| ≥ 0.05 Ha), and the `u_g0w0_uks_pbe_*_qp` validation rows fail by
//! 379–956 meV. The same edit in `run_u_evgw0` (`shifts_a[idx]` → `0.0`)
//! fails `evgw0_ks_shift_enters_the_qp_equation`.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_gw::{run_u_gw, GwConfig, GwMethod, UGwResult};
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_rpa::config::{
    Chi0Backend, Chi0Sparsity, Eigensolver, PdepRpaConfig, QuadratureConfig, QuadratureScheme,
    SternheimerConfig,
};
use ferric_scf::rhf::RhfConfig;
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::solve_uhf;
use ferric_scf::ScfResult;
use ndarray::Array1;

/// QP residual bar: the Newton solve stops at |step| < 1e-7 Ha, so the
/// residual is ≤ 1e-7·|1 − Σc'|; 1e-6 is the same bar `validation_gw.rs`
/// (`TOL_RESID`) uses.
const TOL_RESID: f64 = 1e-6;

struct Oh {
    mol: Molecule,
    obs: PreparedBasis,
    dfbs: PreparedBasis,
    uhf: ScfResult,
}

fn oh_sto3g() -> Oh {
    let mol = Molecule::parse_xyz("2\nOH\nO 0 0 0\nH 0 0 0.9697\n", 0, 2).unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled("sto-3g").unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &obs).unwrap();
    let cfg = RhfConfig {
        max_iter: 200,
        ..Default::default()
    };
    let uhf = solve_uhf(&ParallelContext::default(), &mol, &obs, &bounds, &cfg).unwrap();
    assert!(uhf.converged, "OH/STO-3G UHF did not converge");
    Oh {
        mol,
        obs,
        dfbs,
        uhf,
    }
}

fn pdep_cfg() -> PdepRpaConfig {
    PdepRpaConfig {
        quadrature: QuadratureConfig {
            scheme: QuadratureScheme::GaussLegendre,
            n_points: 8,
            u0: 0.5,
        },
        eigensolver_conv_thresh: 1e-8,
        eigensolver_max_vecs: 0,
        trunc_thresh: 0.0,
        run_diagnostics: false,
        frozen_core: 0,
        chi0_backend: Chi0Backend::Dense,
        chi0_sparsity: Chi0Sparsity::Dense,
        eigensolver: Eigensolver::Davidson,
        sternheimer: SternheimerConfig::default(),
        memory_budget_bytes: None,
        need_inv_dielectric_freq: false, // run_u_gw forces this on
        need_eigenvalues_freq: true,
        verbose: false,
    }
}

fn run(oh: &Oh, method: GwMethod, vxc: Option<(&Array1<f64>, &Array1<f64>)>) -> UGwResult {
    let gw_cfg = GwConfig {
        method,
        max_ev_iter: 30,
        ev_conv_thresh: 1e-8,
        ..Default::default()
    };
    run_u_gw(
        &oh.mol,
        &oh.obs,
        &oh.dfbs,
        Operator::coulomb(),
        &oh.uhf,
        &pdep_cfg(),
        &gw_cfg,
        vxc,
    )
    .unwrap_or_else(|e| panic!("run_u_gw({method:?}): {e}"))
}

/// Absolute-MO-indexed synthetic v_xc per spin (length `nmo`, as run_u_gw
/// requires): v_p = Σ_x,p − Δ_p on the QP window, so the shift Σ_x − v_xc is
/// exactly Δ_p there; 0 elsewhere. `delta` maps the window position to Δ.
fn synthetic_vxc(
    res: &UGwResult,
    nmo: usize,
    delta: impl Fn(usize) -> f64,
) -> (Array1<f64>, Array1<f64>) {
    let mut va = Array1::<f64>::zeros(nmo);
    let mut vb = Array1::<f64>::zeros(nmo);
    for (idx, &mo) in res.mo_indices.iter().enumerate() {
        va[mo] = res.sigma_x_a[idx] - delta(idx);
        vb[mo] = res.sigma_x_b[idx] - delta(idx);
    }
    (va, vb)
}

fn bits(a: &Array1<f64>) -> Vec<u64> {
    a.iter().map(|x| x.to_bits()).collect()
}

fn assert_bit_identical(method: GwMethod) {
    let oh = oh_sto3g();
    let none = run(&oh, method, None);
    // Shift exactly +0.0 on every window state (Σ_x − Σ_x).
    let (va, vb) = synthetic_vxc(&none, oh.uhf.eps_alpha.len(), |_| 0.0);
    let zero = run(&oh, method, Some((&va, &vb)));
    for (name, a, b) in [
        ("eps_qp_a", &none.eps_qp_a, &zero.eps_qp_a),
        ("eps_qp_b", &none.eps_qp_b, &zero.eps_qp_b),
        ("sigma_c_a", &none.sigma_c_a, &zero.sigma_c_a),
        ("sigma_c_b", &none.sigma_c_b, &zero.sigma_c_b),
        ("z_factor_a", &none.z_factor_a, &zero.z_factor_a),
        ("z_factor_b", &none.z_factor_b, &zero.z_factor_b),
    ] {
        assert_eq!(
            bits(a),
            bits(b),
            "{method:?}: {name} with vxc_diag = None is not bit-identical to the zero-shift \
             KS path: {a} vs {b}"
        );
    }
}

#[test]
fn g0w0_uhf_reference_none_is_bit_identical_to_zero_shift() {
    assert_bit_identical(GwMethod::G0W0);
}

#[test]
fn evgw0_uhf_reference_none_is_bit_identical_to_zero_shift() {
    assert_bit_identical(GwMethod::EvGw0);
}

#[test]
fn cohsex_uhf_reference_none_is_bit_identical_to_zero_shift() {
    assert_bit_identical(GwMethod::Cohsex);
}

/// Δ_p = −0.05 − 0.01·p Ha: non-zero on every state and state-dependent, so a
/// spin- or index-mixup in the shift also moves the residual.
fn delta(idx: usize) -> f64 {
    -0.05 - 0.01 * idx as f64
}

fn assert_shift_inside(method: GwMethod) {
    let oh = oh_sto3g();
    let raw = run(&oh, method, None);
    let (va, vb) = synthetic_vxc(&raw, oh.uhf.eps_alpha.len(), delta);
    let ks = run(&oh, method, Some((&va, &vb)));
    assert_eq!(ks.mo_indices, raw.mo_indices);
    let homo_a = raw.mo_indices.iter().position(|&m| m == 4).expect("α HOMO");
    let homo_b = raw.mo_indices.iter().position(|&m| m == 3).expect("β HOMO");
    let mut max_posthoc_gap = 0.0_f64;
    for (spin, homo, res_ks, res_raw) in [
        (
            "alpha",
            homo_a,
            (
                &ks.eps_qp_a,
                &ks.eps_mf_a,
                &ks.sigma_c_a,
                &ks.qp_converged_a,
            ),
            &raw.eps_qp_a,
        ),
        (
            "beta",
            homo_b,
            (
                &ks.eps_qp_b,
                &ks.eps_mf_b,
                &ks.sigma_c_b,
                &ks.qp_converged_b,
            ),
            &raw.eps_qp_b,
        ),
    ] {
        let (eps_qp, eps_mf, sc, conv) = res_ks;
        assert!(
            conv[homo],
            "{method:?} {spin}: HOMO QP Newton did not converge"
        );
        for (k, &mo) in ks.mo_indices.iter().enumerate() {
            if !conv[k] {
                continue;
            }
            // Σ_c is reported at the root, so this residual is the shifted QP
            // equation itself.
            let resid = eps_qp[k] - eps_mf[k] - delta(k) - sc[k];
            assert!(
                resid.abs() < TOL_RESID,
                "{method:?} {spin} MO {mo}: shifted-QP residual {resid:.3e} Ha (shift {:.3} Ha \
                 not inside the QP equation?)",
                delta(k)
            );
            max_posthoc_gap = max_posthoc_gap.max((eps_qp[k] - (res_raw[k] + delta(k))).abs());
        }
    }
    if method == GwMethod::Cohsex {
        // Static self-energy: inside and post hoc coincide (to rounding).
        assert!(
            max_posthoc_gap < 1e-12,
            "COHSEX: shifted QP differs from ε_raw + Δ by {max_posthoc_gap:.3e} Ha"
        );
    } else {
        // Σ_c(ω) is energy-dependent, so solving at the shifted root must
        // differ from shifting the unshifted root by ~|Δ|·|Σc'|·Z ≫ Newton tol.
        assert!(
            max_posthoc_gap > 100.0 * TOL_RESID,
            "{method:?}: shifted QP matches the post-hoc recipe to {max_posthoc_gap:.3e} Ha — \
             the test cannot tell inside from post hoc"
        );
    }
}

#[test]
fn g0w0_ks_shift_enters_the_qp_equation() {
    assert_shift_inside(GwMethod::G0W0);
}

#[test]
fn evgw0_ks_shift_enters_the_qp_equation() {
    assert_shift_inside(GwMethod::EvGw0);
}

#[test]
fn cohsex_ks_shift_is_additive() {
    assert_shift_inside(GwMethod::Cohsex);
}

#[test]
fn run_u_gw_rejects_wrong_length_vxc() {
    let oh = oh_sto3g();
    let nmo = oh.uhf.eps_alpha.len();
    let short = Array1::<f64>::zeros(nmo - 1);
    let ok = Array1::<f64>::zeros(nmo);
    let err = run_u_gw(
        &oh.mol,
        &oh.obs,
        &oh.dfbs,
        Operator::coulomb(),
        &oh.uhf,
        &pdep_cfg(),
        &GwConfig::default(),
        Some((&ok, &short)),
    )
    .unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("vxc_diag") && msg.contains(&format!("β {}", nmo - 1)),
        "unexpected error: {msg}"
    );
}
