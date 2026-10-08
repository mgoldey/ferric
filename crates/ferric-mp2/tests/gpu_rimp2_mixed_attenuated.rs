#![cfg(all(feature = "gpu", feature = "test-seams"))]
//! Mixed-precision RI-MP2 energy (kernel `rimp2-energy`) under the attenuated
//! operators erfc and terfc: closed-shell (water) and unrestricted (O2 triplet,
//! CH3 doublet), against the CPU f64 energy of the SAME dressed `B_ov`.
//! Operators at the repo defaults: erfc ω = 0.420 Å⁻¹
//! (`AttenuatedMp2Config::default()`), terfc r0 = 1.05 Å
//! (`ATT_RIMP2_TERFC_DEFAULT_R0_ANG`, curvature-constrained ω).
//!
//! BOUND, re-derived per operator. The kernel's per-element error is
//! eps_device(naux, b)·S_ab + η with S_ab = Σ_P |B_P,ia||B_P,jb| (rounding
//! analysis in `common/rimp2_eps.rs`): the factor is a property of the
//! arithmetic, but S_ab and η are read from THIS operator's dressed `B_ov`
//! (B = V⁻¹ᐟ²·(P|ia) with the attenuated metric V). A worse-conditioned
//! attenuated metric shows up as larger |B| and as S_ab ≫ |g_ab| (cancellation),
//! which the bound carries element by element; the measured max|B| and the
//! energy-level κ_E are printed per operator next to the Coulomb values. No
//! Coulomb constant is reused. ASSUMPTION: the reference is the f64 energy of
//! the same f64 `B_ov`; the RI fitting error of the attenuated metric (and any
//! loss in forming V⁻¹ᐟ² in f64) is common to both sides and is not part of
//! this bound. The f32 range refusal (max|B| ≤ sqrt(f32::MAX/b)) is checked on
//! the attenuated `B_ov` as on the Coulomb one.
//!
//! SHIP GATE |ΔE| ≤ 1.0e-3 Eh per heavy atom and per electron, asserted on the
//! DERIVED bound (so it holds by construction for these systems) and on the
//! measured error.
//!
//! TWO-SIDED. The threshold that separates an f64 result from a mixed one is
//! the f64 device gate (2γ_naux(u64)·S, the summation-order bound): the f64
//! device kernel is inside it (`gpu_rimp2_resident.rs`,
//! `gpu_u_rimp2_resident.rs`), and the f32-storage part of the mixed error is
//! measured OUTSIDE it here for every operator and block, so the pin
//! (`non_widened_callers_stay_f64_under_mixed`) that compares those callers with
//! the f64 device result bit for bit cannot be passed by a silent mixed run.
//! The kernel-structure defects (f32 accumulation, no f64 flush) are operator
//! independent and pinned by the flush constructions in `gpu_rimp2_mixed.rs`
//! and `gpu_u_rimp2_mixed.rs`.
//!
//! MEASURED (b = 64, cc-pVDZ / cc-pvdz-ri; |ΔE| total, per heavy atom, per
//! electron; derived bound):
//!   H2O    erfc  1.7e-9  1.7e-9  1.7e-10  bound 4.3e-6   | terfc 6.1e-9 6.1e-9 6.1e-10 bound 4.3e-6
//!   O2     erfc  2.2e-9  1.1e-9  1.4e-10  bound 3.8e-6   | terfc 4.5e-9 2.3e-9 2.8e-10 bound 3.9e-6
//!   CH3    erfc  6.5e-11 6.5e-11 7.2e-12  bound 1.5e-6   | terfc 1.8e-9 1.8e-9 2.1e-10 bound 1.6e-6
//! Water max|B|: Coulomb 0.201, erfc 0.209, terfc 0.211; κ_E 1.304 / 1.306 / 1.305.
//!
//! MUTATION RUN (once, temporary source edit, reverted):
//! `rimp2::mixed_energy_operator` returning `true` for every operator. The pin
//! FAILED (erf: mixed GEMMs counted where none are allowed).
use ferric_core::gpu::device::device;
use ferric_core::gpu::mixed::effective_k_panel;
use ferric_core::gpu::mixed_host::{gamma, round_trip_f32, U64};
use ferric_core::gpu::{
    install, pool, probe, stats, GpuMode, GpuSettingsExplicit, GpuStatus, MixedKernel,
    MixedKernelSet, Precision,
};
use ferric_core::units::ANGSTROM_TO_BOHR;
use ferric_integrals::operator::Operator;
use ferric_mp2::attenuated::AttenuatedMp2Config;
use ferric_mp2::rimp2::{spin_components_from_b_ov_kappa_cpu, SpinComponents};
use ferric_mp2::rimp2_gpu::{
    spin_components_on_device, u_opposite_spin_on_device_prec, u_same_spin_on_device_prec,
};
use ndarray::Array2;

#[path = "common/rimp2_error_bound.rs"]
mod bound;
#[path = "common/rimp2_gpu_fixture.rs"]
mod csfix;
#[path = "common/rimp2_eps.rs"]
#[allow(dead_code)]
mod eps;
#[path = "common/u_rimp2_gpu_fixture.rs"]
mod fixture;
#[path = "common/u_rimp2_gpu_real.rs"]
mod real;
#[path = "common/u_rimp2_gpu_bound.rs"]
mod ubound;
use csfix::Prepared;
use fixture::{cpu_opp, cpu_same, Chan};
use ubound::ElemModel;

static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());

const SHIP_GATE_EH: f64 = 1.0e-3;
const TERFC_DEFAULT_R0_ANG: f64 = 1.05;

fn ready() -> bool {
    if !matches!(probe(0), GpuStatus::Ready(_)) {
        eprintln!("skipping: no CUDA device");
        assert!(
            std::env::var("FERRIC_GPU_TESTS_REQUIRED").ok().as_deref() != Some("1"),
            "FERRIC_GPU_TESTS_REQUIRED=1 but no device"
        );
        return false;
    }
    static INSTALL: std::sync::Once = std::sync::Once::new();
    INSTALL.call_once(|| {
        install(GpuSettingsExplicit {
            mode: Some(GpuMode::Auto),
            memory_gb: Some(2.0),
            precision: Some(Precision::Mixed),
            mixed_kernels: Some(MixedKernelSet::EMPTY.with(MixedKernel::RiMp2Energy)),
            ..Default::default()
        })
        .expect("install");
    });
    true
}

fn erfc_default() -> Operator {
    Operator::erfc(AttenuatedMp2Config::default().omega)
}

/// The terfc table engine needs `FERRIC_TERF_TABLE_DIR`; without it the terfc
/// arms are skipped with a printed notice (the erfc arms still run).
fn terfc_tables() -> bool {
    let ok = std::env::var("FERRIC_TERF_TABLE_DIR").is_ok();
    if !ok {
        eprintln!("SKIP terfc arms: FERRIC_TERF_TABLE_DIR unset");
    }
    ok
}

/// The operators gated here: erfc always, terfc when its tables are present.
fn attenuated_ops() -> Vec<(&'static str, Operator)> {
    let mut v = vec![("erfc", erfc_default())];
    if terfc_tables() {
        v.push(("terfc", terfc_default()));
    }
    v
}

fn terfc_default() -> Operator {
    Operator::terfc(TERFC_DEFAULT_R0_ANG * ANGSTROM_TO_BOHR)
}

fn max_abs(b: &Array2<f64>) -> f64 {
    b.iter().fold(0.0f64, |m, x| m.max(x.abs()))
}

fn on_device(p: &Prepared, precision: Precision) -> SpinComponents {
    let dev = device(0).unwrap();
    let pool = pool().unwrap();
    spin_components_on_device(
        &dev,
        &pool,
        &p.b_ov,
        &p.eps,
        p.nocc,
        p.nvir,
        p.first_occ,
        p.nocc_total,
        None,
        precision,
        true,
    )
    .unwrap()
}

fn cs_bound(p: &Prepared, eps_g: f64, eta: f64) -> bound::EnergyBound {
    bound::energy_bound(
        &p.b_ov,
        &p.eps,
        p.nocc,
        p.nvir,
        p.first_occ,
        p.nocc_total,
        bound::BoundSpec {
            eps_g,
            eta,
            kappa: None,
        },
    )
}

/// Water (1 heavy atom, 10 electrons), closed shell, one operator.
fn closed_shell_gate(label: &str, op: Operator) {
    let (p, cpu) = csfix::prepare_scf_op("h2o", "cc-pvdz", "cc-pvdz-ri", 0, op);
    let (n_heavy, n_elec) = (1.0, 10.0);
    let (naux, b) = (p.b_ov.nrows(), effective_k_panel());
    let eta = eps::eta_abs(naux, max_abs(&p.b_ov));
    let s0 = stats();
    let mixed = on_device(&p, Precision::Mixed);
    let s1 = stats();
    assert_eq!(
        (
            s1.gemm_mixed - s0.gemm_mixed,
            s1.mixed_fallback_f64 - s0.mixed_fallback_f64
        ),
        (p.nocc as u64, 0),
        "{label}: the mixed path was not taken for every block"
    );
    let full = cs_bound(&p, eps::eps_device(naux, b), eta);
    let f64g = cs_bound(&p, 2.0 * gamma(naux, U64), 0.0);
    let storage = spin_components_from_b_ov_kappa_cpu(
        &round_trip_f32(&p.b_ov.view()),
        &p.eps,
        p.nocc,
        p.nvir,
        p.first_occ,
        p.nocc_total,
        None,
    );
    let (d_os, d_ss) = (mixed.e_os - cpu.e_os, mixed.e_ss - cpu.e_ss);
    let (s_os, s_ss) = (storage.e_os - cpu.e_os, storage.e_ss - cpu.e_ss);
    let tot = mixed.e_total - cpu.e_total;
    let b_tot = full.os + full.ss;
    eprintln!(
        "error map mixed {label} H2O/cc-pVDZ: nocc {} nvir {} naux {naux} | max|B| {:.3e} kappa_E {:.3} p99 kappa(i=0) {:.2e} | E_corr {:.9} | dE_os {d_os:+.3e} dE_ss {d_ss:+.3e} total {tot:+.3e} Eh | per heavy atom {:.3e} per electron {:.3e} | bound os {:.3e} ss {:.3e} total {b_tot:.3e} (per heavy atom {:.3e}, per electron {:.3e}) | f32-storage {s_os:+.3e} {s_ss:+.3e} vs f64 gate {:.3e} {:.3e}",
        p.nocc, p.nvir, max_abs(&p.b_ov), full.kappa_e_os, full.kappa_p99_block0, cpu.e_total,
        tot.abs() / n_heavy, tot.abs() / n_elec, full.os, full.ss, b_tot / n_heavy, b_tot / n_elec,
        f64g.os, f64g.ss
    );
    assert!(
        d_os.abs() <= full.os && d_ss.abs() <= full.ss,
        "{label}: outside the derived bound"
    );
    assert!(
        b_tot / n_heavy <= SHIP_GATE_EH && b_tot / n_elec <= SHIP_GATE_EH,
        "{label}: ship gate"
    );
    assert!(
        s_os.abs() > f64g.os && s_ss.abs() > f64g.ss,
        "{label}: the f32-storage defect is inside the f64 gate"
    );
    assert!(
        d_os.abs() + d_ss.abs() >= 1e3 * U64 * cpu.e_total.abs(),
        "{label}: mixed did not move the energy off the f64 noise floor"
    );
}

fn u_mixed(a: &Chan, b: &Chan) -> [f64; 3] {
    let dev = device(0).unwrap();
    let pool = pool().unwrap();
    let m = Precision::Mixed;
    [
        u_same_spin_on_device_prec(&dev, &pool, a.ch(), m).unwrap(),
        u_same_spin_on_device_prec(&dev, &pool, b.ch(), m).unwrap(),
        u_opposite_spin_on_device_prec(&dev, &pool, a.ch(), b.ch(), m).unwrap(),
    ]
}

/// One open-shell case under one operator; also runs `u_ri_mp2` end to end.
fn open_shell_gate(label: &str, case: &real::Case, op: Operator) {
    let (a, b) = real::chans_for_op(case, op);
    let n_heavy = case.mol.atoms.iter().filter(|x| x.z > 1).count() as f64;
    let n_elec = case.mol.nelec() as f64;
    let cpu = [cpu_same(&a), cpu_same(&b), cpu_opp(&a, &b)];
    let blocks = (2 * a.nocc + b.nocc) as u64;
    let s0 = stats();
    let dev = u_mixed(&a, &b);
    let s1 = stats();
    assert_eq!(
        s1.gemm_mixed - s0.gemm_mixed,
        blocks,
        "{label}: mixed blocks"
    );
    let naux = a.b.nrows();
    let max_b = max_abs(&a.b).max(max_abs(&b.b));
    let m = ElemModel {
        eps_g: eps::eps_device(naux, effective_k_panel()),
        eta: eps::eta_abs(naux, max_b),
    };
    let bounds = [
        ubound::same_spin_bound_with(&a, m),
        ubound::same_spin_bound_with(&b, m),
        ubound::opposite_spin_bound_with(&a, &b, m),
    ];
    let f64g = [
        ubound::same_spin_bound(&a),
        ubound::same_spin_bound(&b),
        ubound::opposite_spin_bound(&a, &b),
    ];
    let fr = |c: &Chan| Chan {
        b: round_trip_f32(&c.b.view()),
        eps: c.eps.clone(),
        nocc: c.nocc,
        nvir: c.nvir,
        first_occ: c.first_occ,
        nocc_total: c.nocc_total,
    };
    let (fa, fb) = (fr(&a), fr(&b));
    let storage = [cpu_same(&fa), cpu_same(&fb), cpu_opp(&fa, &fb)];
    let d: Vec<f64> = (0..3).map(|k| dev[k] - cpu[k]).collect();
    let tot: f64 = d.iter().sum();
    let b_tot: f64 = bounds.iter().sum();
    eprintln!(
        "error map mixed {label}: naux {naux} max|B| {max_b:.3e} | E_corr {:.9} | dE_aa {:+.3e} dE_bb {:+.3e} dE_ab {:+.3e} total {tot:+.3e} Eh | per heavy atom {:.3e} per electron {:.3e} | bound {:.3e} {:.3e} {:.3e} total {b_tot:.3e} (per heavy atom {:.3e}, per electron {:.3e})",
        cpu.iter().sum::<f64>(), d[0], d[1], d[2], tot.abs() / n_heavy, tot.abs() / n_elec,
        bounds[0], bounds[1], bounds[2], b_tot / n_heavy, b_tot / n_elec
    );
    for k in 0..3 {
        assert!(
            d[k].abs() <= bounds[k],
            "{label} block {k}: outside the derived bound"
        );
        assert!(
            (storage[k] - cpu[k]).abs() > f64g[k],
            "{label} block {k}: f32-storage defect inside the f64 gate"
        );
    }
    assert!(
        b_tot / n_heavy <= SHIP_GATE_EH && b_tot / n_elec <= SHIP_GATE_EH,
        "{label}: ship gate"
    );
    let e0 = stats();
    let e2e = ferric_mp2::u_rimp2::u_ri_mp2(
        &case.mol,
        &case.obs,
        &case.dfbs,
        op,
        &case.scf,
        &ferric_mp2::rimp2::RiMp2Config::default(),
    )
    .unwrap()
    .components;
    let e1 = stats();
    assert_eq!(
        (
            e1.gemm_mixed - e0.gemm_mixed,
            e1.gemm_offloaded - e0.gemm_offloaded
        ),
        (blocks, 0),
        "{label}: e2e u_ri_mp2 did not run the mixed kernel"
    );
    for (k, e) in [e2e.e_aa, e2e.e_bb, e2e.e_ab].into_iter().enumerate() {
        assert!((e - cpu[k]).abs() <= bounds[k], "{label}: e2e block {k}");
    }
}

#[test]
fn erfc_and_terfc_energies_are_inside_their_derived_bounds_and_the_ship_gate() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if !ready() {
        return;
    }
    // Coulomb context for the per-operator norms (printed, not gated here)
    let (pc, _) = csfix::prepare_scf("h2o", "cc-pvdz", "cc-pvdz-ri", 0);
    let fc = cs_bound(
        &pc,
        eps::eps_device(pc.b_ov.nrows(), effective_k_panel()),
        0.0,
    );
    eprintln!(
        "coulomb context H2O/cc-pVDZ: max|B| {:.3e} kappa_E {:.3} p99 kappa(i=0) {:.2e}",
        max_abs(&pc.b_ov),
        fc.kappa_e_os,
        fc.kappa_p99_block0
    );
    for (label, op) in attenuated_ops() {
        closed_shell_gate(label, op);
    }
    for (name, xyz, mult) in [
        ("O2 triplet", "o2.xyz", 3usize),
        ("CH3 doublet", "validation/ch3.xyz", 2),
    ] {
        let case = real::real_case(xyz, 0, mult);
        for (label, op) in attenuated_ops() {
            open_shell_gate(&format!("{label} {name}"), &case, op);
        }
    }
}

/// The closed-shell erfc/terfc ENERGY entries (`ri_mp2_spin_components`,
/// `attenuated_ri_mp2` dense and QQR-screened) take the mixed kernel; the
/// callers that are not widened run the f64 device kernel: erf
/// (`attenuated_ri_mp2_long_range`), the composite fitted terfc
/// (`Operator::terfc_fit`), kappa-regularised RI-MP2, and
/// `spin_components_from_b_ov` (the entry of SR-MP2 inside RS-MP2+RPA and of
/// OO-MP2), the last bit for bit equal to the f64 device result; U-RI-MP2
/// under erf likewise.
#[test]
fn non_widened_callers_stay_f64_under_mixed() {
    use ferric_mp2::attenuated::{attenuated_ri_mp2, attenuated_ri_mp2_long_range};
    use ferric_mp2::rimp2::{ri_mp2_spin_components, spin_components_from_b_ov, RiMp2Config};
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if !ready() {
        return;
    }
    let root = env!("CARGO_MANIFEST_DIR");
    let mol = ferric_core::mol::Molecule::load_xyz(&format!(
        "{root}/../../testdata/molecules/validation/h2o.xyz"
    ))
    .unwrap();
    let obs_bs = ferric_core::basis::bundled("cc-pvdz").unwrap();
    let obs = ferric_integrals::basis_bridge::PreparedBasis::new(&mol, &obs_bs).unwrap();
    let dfbs = ferric_integrals::basis_bridge::PreparedBasis::new(
        &mol,
        &ferric_core::basis::bundled("cc-pvdz-ri").unwrap(),
    )
    .unwrap();
    let op = Operator::coulomb();
    let bounds = ferric_scf::screening::SchwarzBounds::compute(op, &obs).unwrap();
    let rhf = ferric_scf::rhf::solve_rhf(
        &ferric_core::parallel::ParallelContext::default(),
        &mol,
        &obs,
        op,
        &bounds,
        &ferric_scf::rhf::RhfConfig {
            df_j_aux: Some("def2-universal-jkfit".into()),
            df_k_aux: Some("def2-universal-jkfit".into()),
            energy_conv: 1e-8,
            density_conv: 1e-6,
            ..Default::default()
        },
    )
    .unwrap();
    let nocc = (mol.nelec() / 2) as u64;
    let count = |f: &dyn Fn()| {
        let s0 = stats();
        f();
        let s1 = stats();
        (
            s1.gemm_mixed - s0.gemm_mixed,
            s1.gemm_offloaded - s0.gemm_offloaded,
        )
    };
    let cfg = AttenuatedMp2Config::default();
    let rcfg = RiMp2Config::default();
    // widened: the erfc/terfc energy entries
    let mut widened = vec![
        count(&|| {
            let _ = attenuated_ri_mp2(&mol, &obs, &dfbs, &rhf, &cfg).unwrap();
        }),
        count(&|| {
            let c = AttenuatedMp2Config {
                screen_thresh: Some(1e-10),
                ..AttenuatedMp2Config::default()
            };
            let _ = attenuated_ri_mp2(&mol, &obs, &dfbs, &rhf, &c).unwrap();
        }),
    ];
    if terfc_tables() {
        widened.push(count(&|| {
            ri_mp2_spin_components(&mol, &obs, &dfbs, terfc_default(), &rhf, &rcfg).unwrap();
        }));
    }
    for (k, c) in widened.iter().enumerate() {
        assert_eq!(*c, (nocc, 0), "widened entry {k} did not run mixed");
    }
    // not widened
    let kcfg = RiMp2Config {
        kappa: Some(1.45),
        ..RiMp2Config::default()
    };
    let r0 = TERFC_DEFAULT_R0_ANG * ANGSTROM_TO_BOHR;
    let kept = [
        count(&|| {
            let _ = attenuated_ri_mp2_long_range(&mol, &obs, &dfbs, &rhf, &cfg).unwrap();
        }),
        count(&|| {
            ri_mp2_spin_components(&mol, &obs, &dfbs, Operator::terfc_fit(r0), &rhf, &rcfg)
                .unwrap();
        }),
        count(&|| {
            ri_mp2_spin_components(&mol, &obs, &dfbs, erfc_default(), &rhf, &kcfg).unwrap();
        }),
    ];
    for (k, c) in kept.iter().enumerate() {
        assert_eq!(*c, (0, nocc), "non-widened entry {k} ran mixed");
    }
    // spin_components_from_b_ov on an erfc B_ov: f64, bit for bit
    let (p, _) = csfix::prepare_scf_op("h2o", "cc-pvdz", "cc-pvdz-ri", 0, erfc_default());
    let f64dev = on_device(&p, Precision::F64);
    let mixed = on_device(&p, Precision::Mixed);
    assert_ne!(f64dev.e_total.to_bits(), mixed.e_total.to_bits());
    let s0 = stats();
    let sc = spin_components_from_b_ov(&p.b_ov, &p.eps, p.nocc, p.nvir, p.first_occ, p.nocc_total);
    let s1 = stats();
    assert_eq!(
        s1.gemm_mixed, s0.gemm_mixed,
        "SR-MP2/OO-MP2 entry ran mixed"
    );
    assert_eq!(sc.e_total.to_bits(), f64dev.e_total.to_bits());
    // U-RI-MP2 under erf: f64
    let case = real::real_case("validation/ch3.xyz", 0, 2);
    let (ua, ub) = real::chans(&case.amps);
    let s0 = stats();
    let _ = ferric_mp2::u_rimp2::u_ri_mp2(
        &case.mol,
        &case.obs,
        &case.dfbs,
        Operator::erf(cfg.omega),
        &case.scf,
        &rcfg,
    )
    .unwrap();
    let s1 = stats();
    assert_eq!(
        (
            s1.gemm_mixed - s0.gemm_mixed,
            s1.gemm_offloaded - s0.gemm_offloaded
        ),
        (0, (2 * ua.nocc + ub.nocc) as u64),
        "U-RI-MP2 under erf ran mixed"
    );
}
