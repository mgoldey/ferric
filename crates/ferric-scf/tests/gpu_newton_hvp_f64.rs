#![cfg(all(feature = "gpu", feature = "test-seams"))]
//! The Newton HVP's density-fitted response J/K run f64 on the device under
//! mixed-precision settings (`rhf_newton::ScfJkResponse`, guarded by
//! `jk_ladder::F64BuildScope`).
//!
//! Why: the HVP builds run outside the SCF's J/K ladder scope, where the device
//! kernels treat every build as mixed-eligible. The response δK comes from the
//! ± occupied factor `K_occ(X+Y) − K_occ(X−Y)`, so an f32 `dfk-occ` build would
//! leave an absolute error of about `ε₃₂·|K_occ(X)|` whatever `|κ|` is: O(1)
//! relative error late in PCG. A mixed build there would also be counted in the
//! kernel provenance of a run whose energy builds were all f64.
//!
//! The test enables `dfk-occ` and `dfj-pack` (plus the shipped mixed kernels) through the
//! `MIXED_SETTINGS_OVERRIDE` seam, as `gpu_dfk_mixed_kernel.rs` does, then:
//!  * positive controls: one `build_from_occ` on the same `DfK` and one `build`
//!    on the same `DfJ`, OUTSIDE any scope, run mixed (`dfk-occ` and `dfj-pack`
//!    mixed counters +1 each), so the settings reach both kernels and the main
//!    asserts are not vacuous;
//!  * an HVP through `hessian_matvec_jk` with the fitted builders lent adds ZERO
//!    to the `dfk-occ` and `dfj-pack` mixed counters.
//!
//! Mutation (applied and reverted, 2026-11-11): removing the
//! `F64BuildScope::enter()` guard from `rhf_newton::response_jk` makes the HVP
//! run 2 `dfk-occ` and 1 `dfj-pack` builds mixed (GTX 1080), failing the
//! zero-count assert. The controls read 1 and 1 in both runs.
//!
//! Open shell (`open_shell_newton_hvp_fitted_jk_runs_f64_under_mixed_settings`):
//! the same check for the UHF/UKS and ROHF/ROKS matvecs
//! (`uhf_newton::hessian_matvec`, `rohf_newton::hessian_matvec` with fitted
//! `j_response`/`k_response`, scoped in their `response_fock`). Mutation
//! (applied and reverted, 2026-11-11): removing the guard from both
//! `response_fock`s makes each HVP run 4 `dfk-occ` and 1 `dfj-pack` builds
//! mixed (UHF and ROHF alike; GTX 1080), failing the zero-count asserts.
//!
//! Its own binary: the kernel counters and GPU settings are process-wide, and
//! the two tests serialise on `SERIAL`.
use ferric_core::basis;
use ferric_core::gpu::config::{GpuSettings, GpuSettingsExplicit};
use ferric_core::gpu::stats::kernel_counters;
use ferric_core::gpu::{
    install, probe, GpuMode, GpuStatus, MixedKernel, MixedKernelSet, Precision,
};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::df_j::{DfJ, RiJPrecision};
use ferric_scf::df_k::DfK;
use ferric_scf::df_k_gpu::MIXED_SETTINGS_OVERRIDE;
use ferric_scf::fock::KBuilder;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::rhf_newton::{hessian_matvec_jk, RhfNewtonInputs, ScfJkResponse};
use ferric_scf::screening::SchwarzBounds;
use ndarray::Array2;

const AUX: &str = "def2-universal-jkfit";

/// The kernel counters and the settings override are process-wide.
static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn ready() -> bool {
    if !matches!(probe(0), GpuStatus::Ready(_)) {
        eprintln!("skipping: no CUDA device");
        assert!(
            std::env::var("FERRIC_GPU_TESTS_REQUIRED").ok().as_deref() != Some("1"),
            "FERRIC_GPU_TESTS_REQUIRED=1 but no device"
        );
        return false;
    }
    install(GpuSettingsExplicit {
        mode: Some(GpuMode::Auto),
        memory_gb: Some(4.0),
        ..Default::default()
    })
    .expect("install");
    true
}

/// Mixed settings with `dfk-occ` allowed, installed in the override seam for
/// the guard's lifetime.
struct Mixed;
impl Mixed {
    fn on() -> Self {
        let shipped = MixedKernelSet::SHIPPED
            .with(MixedKernel::DfkOcc)
            .with(MixedKernel::DfjPack);
        let (s, _) = GpuSettings::resolve_with_default(
            GpuSettingsExplicit {
                mode: Some(GpuMode::Auto),
                precision: Some(Precision::Mixed),
                mixed_kernels: Some(shipped),
                memory_gb: Some(4.0),
                ..Default::default()
            },
            |_| None,
            Precision::F64,
            shipped,
        )
        .expect("resolve");
        *MIXED_SETTINGS_OVERRIDE.lock().unwrap() = Some(s);
        Mixed
    }
}
impl Drop for Mixed {
    fn drop(&mut self) {
        *MIXED_SETTINGS_OVERRIDE
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = None;
    }
}

fn mixed_count(k: MixedKernel) -> u64 {
    kernel_counters()[k.index()].mixed
}

#[test]
fn newton_hvp_fitted_jk_runs_f64_under_mixed_settings() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    if !ready() {
        return;
    }
    let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
    let prep = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
    let aux = PreparedBasis::new(&mol, &basis::bundled(AUX).unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();
    let n = prep.nbasis();
    let nocc = mol.nelec() as usize / 2;

    // Orbitals from ferric's own f64 SCF, before any mixed setting is on.
    let cfg = RhfConfig {
        energy_conv: 1e-9,
        density_conv: 1e-7,
        df_j_aux: Some(AUX.into()),
        df_k_aux: Some(AUX.into()),
        ri_j_precision: Some(RiJPrecision::F64),
        ..Default::default()
    };
    let res = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg).unwrap();
    assert!(res.converged);
    let c = res.mos_alpha.clone();
    let f_mo = c.t().dot(&res.fock_alpha).dot(&c);

    let _mixed = Mixed::on();
    let budget = ferric_core::memory::resolve_budget_bytes(None);
    let mut dfj = DfJ::new_with_precision(op, &prep, &aux, budget, RiJPrecision::F64).unwrap();
    let mut dfk = DfK::new(op, &prep, &aux, budget).unwrap();

    // Positive control: outside any scope the same DfK runs dfk-occ mixed.
    let k0 = mixed_count(MixedKernel::DfkOcc);
    let c_occ = c.slice(ndarray::s![.., ..nocc]).to_owned();
    let mut k_probe = Array2::<f64>::zeros((n, n));
    dfk.build_from_occ(&c_occ, &mut k_probe).unwrap();
    let control = mixed_count(MixedKernel::DfkOcc) - k0;
    let j0 = mixed_count(MixedKernel::DfjPack);
    let d_occ = 2.0 * c_occ.dot(&c_occ.t());
    let mut j_probe = Array2::<f64>::zeros((n, n));
    ferric_scf::fock::JBuilder::build(&mut dfj, &d_occ, &mut j_probe).unwrap();
    let control_j = mixed_count(MixedKernel::DfjPack) - j0;
    assert!(
        control_j >= 1,
        "positive control: an unscoped RI-J build must run dfj-pack mixed (got {control_j})"
    );
    assert!(
        control >= 1,
        "positive control: an unscoped dfk-occ build must run mixed under these settings \
         (got {control}); without it the assert below proves nothing"
    );

    let nv = n - nocc;
    let kappa = Array2::<f64>::from_shape_fn((nv, nocc), |(a, i)| {
        1e-3 * (((a * 7 + i * 3) % 11) as f64 - 5.0)
    });
    let inputs = RhfNewtonInputs {
        prep: &prep,
        bounds: &bounds,
        c: &c,
        f_mo: &f_mo,
        nocc,
        k_mix_sr: 1.0,
        rsh: None,
        fxc: None,
        thresh: 1e-12,
        ooc_budget: budget,
    };
    let pool = ferric_scf::engine_pool::EnginePool::new(op, &prep, 1e-14).unwrap();
    let (k1, j1) = (
        mixed_count(MixedKernel::DfkOcc),
        mixed_count(MixedKernel::DfjPack),
    );
    let hk = {
        let resp = ScfJkResponse::new(Some(&mut dfj), Some(&mut dfk));
        hessian_matvec_jk(&ctx, &inputs, &kappa, &pool, Some(&resp)).unwrap()
    };
    let (dk, dj) = (
        mixed_count(MixedKernel::DfkOcc) - k1,
        mixed_count(MixedKernel::DfjPack) - j1,
    );
    eprintln!("control dfk-occ mixed={control} dfj-pack mixed={control_j}; HVP dfk-occ mixed={dk} dfj-pack mixed={dj}");
    assert!(hk.iter().all(|v| v.is_finite()));
    assert_eq!(dk, 0, "the HVP's response K ran dfk-occ mixed {dk} times");
    assert_eq!(dj, 0, "the HVP's response J ran dfj-pack mixed {dj} times");
}

#[test]
fn open_shell_newton_hvp_fitted_jk_runs_f64_under_mixed_settings() {
    use ferric_scf::uhf_newton::{AoResponse, OccKResponse};
    use std::cell::RefCell;
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    if !ready() {
        return;
    }
    let mol = Molecule::parse_xyz("2\nOH\nO 0 0 0\nH 0 0 0.97\n", 0, 2).unwrap();
    let prep = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
    let aux = PreparedBasis::new(&mol, &basis::bundled(AUX).unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();
    let n = prep.nbasis();
    let (na, nb) = (5usize, 4usize);

    // Orbitals from ferric's own f64 UHF, before any mixed setting is on.
    let cfg = RhfConfig {
        energy_conv: 1e-9,
        density_conv: 1e-7,
        df_j_aux: Some(AUX.into()),
        df_k_aux: Some(AUX.into()),
        ri_j_precision: Some(RiJPrecision::F64),
        ..Default::default()
    };
    let res = ferric_scf::uhf::solve_uhf(&ctx, &mol, &prep, &bounds, &cfg).unwrap();
    assert!(res.converged);
    let ca = res.mos_alpha.clone();
    let cb = res.mos_beta.clone().unwrap();
    let fa_mo = ca.t().dot(&res.fock_alpha).dot(&ca);
    let fb_mo = cb.t().dot(res.fock_beta.as_ref().unwrap()).dot(&cb);

    let _mixed = Mixed::on();
    let budget = ferric_core::memory::resolve_budget_bytes(None);
    let dfj =
        RefCell::new(DfJ::new_with_precision(op, &prep, &aux, budget, RiJPrecision::F64).unwrap());
    let dfk = RefCell::new(DfK::new(op, &prep, &aux, budget).unwrap());

    // Positive control: outside any scope the same DfK runs dfk-occ mixed.
    let k0 = mixed_count(MixedKernel::DfkOcc);
    let c_occ = ca.slice(ndarray::s![.., ..na]).to_owned();
    let mut k_probe = Array2::<f64>::zeros((n, n));
    dfk.borrow_mut()
        .build_from_occ(&c_occ, &mut k_probe)
        .unwrap();
    let control = mixed_count(MixedKernel::DfkOcc) - k0;
    assert!(
        control >= 1,
        "positive control: an unscoped dfk-occ build must run mixed (got {control})"
    );

    let jr = Box::new(|dd: &Array2<f64>| {
        let mut out = Array2::<f64>::zeros(dd.raw_dim());
        ferric_scf::fock::JBuilder::build(&mut *dfj.borrow_mut(), dd, &mut out)?;
        Ok(out)
    }) as Box<AoResponse<'_>>;
    let kr = Box::new(|cf: &Array2<f64>| {
        let mut out = Array2::<f64>::zeros((n, n));
        dfk.borrow_mut().build_from_occ(cf, &mut out)?;
        Ok(out)
    }) as Box<OccKResponse<'_>>;
    let pool = ferric_scf::engine_pool::EnginePool::new(op, &prep, 1e-14).unwrap();
    let kap = |r: usize, c: usize| {
        Array2::<f64>::from_shape_fn((r, c), |(a, i)| {
            1e-3 * (((a * 7 + i * 3) % 11) as f64 - 5.0)
        })
    };
    let counts = || {
        (
            mixed_count(MixedKernel::DfkOcc),
            mixed_count(MixedKernel::DfjPack),
        )
    };

    // UHF/UKS matvec.
    let inp = ferric_scf::uhf_newton::UhfNewtonInputs {
        prep: &prep,
        bounds: &bounds,
        c_a: &ca,
        c_b: &cb,
        f_a_mo: &fa_mo,
        f_b_mo: &fb_mo,
        nocc_a: na,
        nocc_b: nb,
        k_mix_sr: 1.0,
        rsh: None,
        fxc: None,
        thresh: 1e-12,
        ooc_budget: budget,
        j_response: Some(&*jr),
        k_response: Some(ferric_scf::uhf_newton::KResponse::Occ(&*kr)),
    };
    let (k1, j1) = counts();
    let (ha, hb) = ferric_scf::uhf_newton::hessian_matvec(
        &ctx,
        &inp,
        &kap(n - na, na),
        &kap(n - nb, nb),
        &pool,
    )
    .unwrap();
    let (k2, j2) = counts();
    let (uk, uj) = (k2 - k1, j2 - j1);

    // ROHF/ROKS matvec on the α MOs (nc = 4 closed, 1 open).
    let (nc, no) = (nb, na - nb);
    let rinp = ferric_scf::rohf_newton::RohfNewtonInputs {
        prep: &prep,
        bounds: &bounds,
        c: &ca,
        f_a_mo: &fa_mo,
        f_b_mo: &fa_mo,
        nocc_double: nc,
        nocc_open: no,
        k_mix_sr: 1.0,
        rsh: None,
        fxc: None,
        thresh: 1e-12,
        ooc_budget: budget,
        j_response: Some(&*jr),
        k_response: Some(ferric_scf::uhf_newton::KResponse::Occ(&*kr)),
    };
    let (r_vc, r_vo, r_oc) = ferric_scf::rohf_newton::hessian_matvec(
        &ctx,
        &rinp,
        &kap(n - na, nc),
        &kap(n - na, no),
        &kap(no, nc),
        &pool,
    )
    .unwrap();
    let (k3, j3) = counts();
    let (rk, rj) = (k3 - k2, j3 - j2);
    eprintln!(
        "control dfk-occ mixed={control}; UHF HVP dfk-occ mixed={uk} dfj-pack mixed={uj}; \
         ROHF HVP dfk-occ mixed={rk} dfj-pack mixed={rj}"
    );
    for m in [&ha, &hb, &r_vc, &r_vo, &r_oc] {
        assert!(m.iter().all(|v| v.is_finite()));
    }
    assert_eq!((uk, uj), (0, 0), "the UHF HVP's response J/K ran mixed");
    assert_eq!((rk, rj), (0, 0), "the ROHF HVP's response J/K ran mixed");
}
