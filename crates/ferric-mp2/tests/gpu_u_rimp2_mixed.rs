#![cfg(all(feature = "gpu", feature = "test-seams"))]
//! Mixed-precision UNRESTRICTED RI-MP2 energy (kernel `rimp2-energy`): one
//! f32-resident `B_ov` per spin, the shipped k-panelled SGEMM with f64
//! accumulation for every block of E_αα, E_ββ and E_αβ, f64 pair arithmetic,
//! against the CPU f64 energy of the same `B_ov`.
//!
//! BOUND (derived, not tuned). Per element of G_i = B_iᵀ·B_other (depth naux,
//! panels of b = `effective_k_panel()`), the closed-shell kernel's factor
//! eps_device = (1+u32)²(1+γ_b(u32))(1+γ_⌈naux/b⌉(u64)) − 1 + γ_naux(u64) times
//! S_ab = Σ_P|B||B|, plus the f32 underflow term η (`common/rimp2_eps.rs`),
//! propagated through the SS antisymmetrised K = g_ab − g_ba and the OS g² by
//! `common/u_rimp2_gpu_bound.rs` (the same propagation as the f64 device gate,
//! only the per-element model differs). The SHIP GATE is |ΔE| ≤ 1.0e-3 Eh per
//! heavy atom and per electron; the derived bound itself (not only the measured
//! error) is asserted under it, so the gate holds by construction on every
//! system the bound is evaluated on, and the bound is a sum over pair terms, so
//! it grows with the number of pairs (size-extensive, not per-system tuned).
//!
//! TWO-SIDED. The derived energy bound is a worst case over rounding signs; like
//! the closed-shell one it sits decades above the measured error and is blind
//! to a subtle kernel degradation (f32 accumulation over the whole contraction,
//! or the panels summed in f32 before one flush). Those are pinned by the
//! deterministic construction `the_f64_flush_keeps_what_f32_accumulation_loses_in_both_spin_blocks`:
//! an element whose exact value 1 + 2⁻³⁰ needs the f64 flush across two panels.
//! Correct side: device error 0 (measured). Mutant side: an f32 sum returns 1,
//! so the energy moves by 2⁻²⁹ ≈ 1.9e-9 Eh per block (computed exactly on the
//! host by dropping the 2⁻³⁰ term). Threshold: half the mutant gap, 2⁻³⁰.
//!
//! MUTATION RUN (once, by a temporary source edit, reverted): the mixed GEMM in
//! `u_g_block_on_device` given one panel of width naux (f32 accumulation over
//! the whole contraction). The flush construction FAILED (device −1.0, error
//! 1.863e-9 against the threshold 9.31e-10); the panel counter in the
//! energy test FAILED (one panel per block instead of ⌈naux/b⌉); with that
//! counter assertion disabled, the derived energy bound PASSED the mutant on
//! both molecules (O2 ΔE +9.2e-9, CH3 +5.9e-10 Eh, bounds 3.9e-6 and 1.6e-6):
//! the energy bound is a gross-defect gate, not a kernel-structure gate.
//!
//! MEASURED (cc-pVDZ / cc-pvdz-ri, RI-JK UHF reference, b = 64; printed by
//! `mixed_unrestricted_energy_is_inside_the_derived_bound_and_the_ship_gate`):
//!   system       nocc a/b nvir a/b naux  E_corr      dE_aa     dE_bb     dE_ab     total     /heavy atom /electron  bound (Eh)  bound/heavy atom
//!   O2 triplet   9/7      19/21    112   -0.348888  -2.4e-9   +1.1e-8   +5.5e-9   +1.4e-8   7.0e-9      8.7e-10    3.9e-6      2.0e-6
//!   CH3 doublet  5/4      24/25     98   -0.129040  +1.5e-10  +2.3e-10  +1.3e-10  +5.0e-10  5.0e-10     5.5e-11    1.6e-6      1.6e-6
//! Largest error/bound per block: 1.2e-2 (O2 E_bb).
use ferric_core::gpu::device::device;
use ferric_core::gpu::mixed::effective_k_panel;
use ferric_core::gpu::mixed_host::{round_trip_f32, U64};
use ferric_core::gpu::{
    install, pool, probe, stats, GpuMode, GpuSettingsExplicit, GpuStatus, MixedKernel,
    MixedKernelSet, Precision,
};
use ferric_mp2::rimp2_gpu::{
    try_u_opposite_spin_on_device, try_u_same_spin_on_device, u_opposite_spin_on_device_prec,
    u_same_spin_on_device_prec, FAIL_AS_KERNEL, FAIL_AT_BLOCK,
};
use ndarray::Array2;
use std::sync::atomic::Ordering;

#[path = "common/u_rimp2_gpu_bound.rs"]
mod bound;
#[path = "common/rimp2_eps.rs"]
#[allow(dead_code)]
mod eps;
#[path = "common/u_rimp2_gpu_fixture.rs"]
mod fixture;
#[path = "common/u_rimp2_gpu_real.rs"]
mod real;
#[path = "common/u_rimp2_gpu_synth.rs"]
mod synth;
use bound::ElemModel;
use fixture::{cpu_opp, cpu_same, Chan};

static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());

const SHIP_GATE_EH: f64 = 1.0e-3;

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
        memory_gb: Some(2.0),
        precision: Some(Precision::Mixed),
        mixed_kernels: Some(MixedKernelSet::EMPTY.with(MixedKernel::RiMp2Energy)),
        ..Default::default()
    })
    .expect("install");
    true
}

/// The three block energies on the device at `p` against the process pool.
fn device_energies(a: &Chan, b: &Chan, p: Precision) -> [f64; 3] {
    let dev = device(0).unwrap();
    let pool = pool().unwrap();
    [
        u_same_spin_on_device_prec(&dev, &pool, a.ch(), p).unwrap(),
        u_same_spin_on_device_prec(&dev, &pool, b.ch(), p).unwrap(),
        u_opposite_spin_on_device_prec(&dev, &pool, a.ch(), b.ch(), p).unwrap(),
    ]
}

/// The mixed kernel's per-element model for a channel pair of depth naux.
fn mixed_model(a: &Chan, b: &Chan) -> ElemModel {
    let naux = a.b.nrows();
    let max_b =
        a.b.iter()
            .chain(b.b.iter())
            .fold(0.0f64, |m, x| m.max(x.abs()));
    ElemModel {
        eps_g: eps::eps_device(naux, effective_k_panel()),
        eta: eps::eta_abs(naux, max_b),
    }
}

fn f32_rounded(c: &Chan) -> Chan {
    Chan {
        b: round_trip_f32(&c.b.view()),
        eps: c.eps.clone(),
        nocc: c.nocc,
        nvir: c.nvir,
        first_occ: c.first_occ,
        nocc_total: c.nocc_total,
    }
}

#[test]
fn mixed_unrestricted_energy_is_inside_the_derived_bound_and_the_ship_gate() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if !ready() {
        return;
    }
    for (name, xyz, mult) in [
        ("O2 triplet", "o2.xyz", 3usize),
        ("CH3 doublet", "validation/ch3.xyz", 2),
    ] {
        let case = real::real_case(xyz, 0, mult);
        let (a, b) = real::chans(&case.amps);
        let n_heavy = case.mol.atoms.iter().filter(|x| x.z > 1).count();
        let n_elec = case.mol.nelec() as usize;
        let cpu = [cpu_same(&a), cpu_same(&b), cpu_opp(&a, &b)];
        let s0 = stats();
        let dev = device_energies(&a, &b, Precision::Mixed);
        let s1 = stats();
        // the mixed kernel ran for every block of all three energies, nothing fell back
        let blocks = (2 * a.nocc + b.nocc) as u64;
        assert_eq!(
            (
                s1.gemm_mixed - s0.gemm_mixed,
                s1.mixed_fallback_f64 - s0.mixed_fallback_f64,
                s1.gemm_offloaded - s0.gemm_offloaded,
            ),
            (blocks, 0, 0),
            "{name}: the mixed path was not taken for every block"
        );
        let panels = a.b.nrows().div_ceil(effective_k_panel()) as u64;
        assert_eq!(s1.mixed_panels - s0.mixed_panels, blocks * panels);
        let m = mixed_model(&a, &b);
        let bounds = [
            bound::same_spin_bound_with(&a, m),
            bound::same_spin_bound_with(&b, m),
            bound::opposite_spin_bound_with(&a, &b, m),
        ];
        let f64_gate = [
            bound::same_spin_bound(&a),
            bound::same_spin_bound(&b),
            bound::opposite_spin_bound(&a, &b),
        ];
        let (fa, fb) = (f32_rounded(&a), f32_rounded(&b));
        let storage = [cpu_same(&fa), cpu_same(&fb), cpu_opp(&fa, &fb)];
        let mut d_tot = 0.0;
        for (k, blk) in ["aa", "bb", "ab"].iter().enumerate() {
            let d = dev[k] - cpu[k];
            let ds = storage[k] - cpu[k];
            d_tot += d;
            eprintln!(
                "error map mixed U {name} E_{blk}: E {:.9} dE {d:+.3e} bound {:.3e} (dE/bound {:.2e}) | f32-storage dE {ds:+.3e} vs f64 device gate {:.3e}",
                cpu[k],
                bounds[k],
                d.abs() / bounds[k],
                f64_gate[k]
            );
            assert!(
                d.abs() <= bounds[k],
                "{name} E_{blk}: {d:e} > bound {:e}",
                bounds[k]
            );
            // teeth: the mixed error class is outside the f64 device gate, so
            // that gate would catch a silent mixed run on an f64-only caller
            assert!(
                ds.abs() > f64_gate[k],
                "{name} E_{blk}: f32 storage {ds:e} inside the f64 gate {:e}",
                f64_gate[k]
            );
        }
        let b_tot: f64 = bounds.iter().sum();
        let e_corr: f64 = cpu.iter().sum();
        eprintln!(
            "ship gate U {name}: nocc a/b {}/{} nvir a/b {}/{} naux {} | E_corr {e_corr:.9} | dE {d_tot:+.3e} Eh ({:+.3e} kcal/mol) | per heavy atom ({n_heavy}) {:.3e} | per electron ({n_elec}) {:.3e} | bound {b_tot:.3e} Eh, per heavy atom {:.3e}, per electron {:.3e} | gate {SHIP_GATE_EH:e}",
            a.nocc, b.nocc, a.nvir, b.nvir, a.b.nrows(),
            d_tot * 627.5094740631,
            d_tot.abs() / n_heavy as f64,
            d_tot.abs() / n_elec as f64,
            b_tot / n_heavy as f64,
            b_tot / n_elec as f64,
        );
        assert!(
            b_tot / n_heavy as f64 <= SHIP_GATE_EH,
            "{name}: derived bound per heavy atom"
        );
        assert!(
            b_tot / n_elec as f64 <= SHIP_GATE_EH,
            "{name}: derived bound per electron"
        );
        // positive control: the mixed path moved the energy off the f64 noise class
        assert!(
            d_tot.abs() >= 1e3 * U64 * e_corr.abs(),
            "{name}: mixed moved the energy by less than the f64 noise floor; not engaged?"
        );
        // end to end: u_ri_mp2 (Coulomb, no kappa) takes the mixed kernel for
        // every block and lands inside the same bounds
        let e0 = stats();
        let e2e = ferric_mp2::u_rimp2::u_ri_mp2(
            &case.mol,
            &case.obs,
            &case.dfbs,
            ferric_integrals::operator::Operator::coulomb(),
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
            "{name}: e2e u_ri_mp2 did not run the mixed kernel for every block"
        );
        for (k, e) in [e2e.e_aa, e2e.e_bb, e2e.e_ab].into_iter().enumerate() {
            assert!((e - cpu[k]).abs() <= bounds[k], "{name}: e2e block {k}");
        }
    }
}

/// A channel whose B has only two non-zero rows per used column: row 0 (panel
/// 0) and row `bp` (panel 1). `cols` lists the columns that carry the pattern
/// (1 at row 0, 2⁻¹⁵ at row bp), so each product of two such columns is
/// 1 + 2⁻³⁰: exact in f64 and through the f64 flush, 1 in any f32 sum.
fn flush_channel(nocc: usize, nvir: usize, cols: &[usize], tiny: bool) -> Chan {
    let bp = effective_k_panel();
    let mut b = Array2::<f64>::zeros((2 * bp, nocc * nvir));
    for &c in cols {
        b[(0, c)] = 1.0;
        if tiny {
            b[(bp, c)] = 2f64.powi(-15);
        }
    }
    let mut eps = vec![-0.5; nocc];
    eps.extend(std::iter::repeat_n(0.0, nvir)); // every denominator is exactly -1
    Chan {
        b,
        eps,
        nocc,
        nvir,
        first_occ: 0,
        nocc_total: nocc,
    }
}

#[test]
fn the_f64_flush_keeps_what_f32_accumulation_loses_in_both_spin_blocks() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if !ready() {
        return;
    }
    let dev = device(0).unwrap();
    let pool = pool().unwrap();
    // SS: nocc 2, nvir 2; columns (0,a=0) and (1,b=1) carry the pattern, so
    // K = (00|11) − (01|10) = 1 + 2⁻³⁰ for (i,j) = (0,1).
    let ss = flush_channel(2, 2, &[0, 3], true);
    let ss_mutant = flush_channel(2, 2, &[0, 3], false); // what an f32 sum returns
                                                         // OS: nocc 1, nvir 1 on both sides, g = (00|00) = 1 + 2⁻³⁰.
    let (oa, ob) = (
        flush_channel(1, 1, &[0], true),
        flush_channel(1, 1, &[0], true),
    );
    let (ma, mb) = (
        flush_channel(1, 1, &[0], false),
        flush_channel(1, 1, &[0], false),
    );
    let cases = [
        (
            "ss",
            cpu_same(&ss),
            cpu_same(&ss_mutant),
            u_same_spin_on_device_prec(&dev, &pool, ss.ch(), Precision::Mixed).unwrap(),
        ),
        (
            "os",
            cpu_opp(&oa, &ob),
            cpu_opp(&ma, &mb),
            u_opposite_spin_on_device_prec(&dev, &pool, oa.ch(), ob.ch(), Precision::Mixed)
                .unwrap(),
        ),
    ];
    for (blk, exact, mutant, dev_e) in cases {
        let gap = (mutant - exact).abs();
        let threshold = 0.5 * gap;
        eprintln!(
            "flush construction {blk}: exact {exact:.17e} device {dev_e:.17e} (err {:.3e}) | f32-sum mutant {mutant:.17e} (err {gap:.3e}) | threshold {threshold:.3e}",
            (dev_e - exact).abs()
        );
        assert!(
            gap >= 2f64.powi(-29) * 0.99,
            "{blk}: construction has no teeth ({gap:e})"
        );
        assert!(
            (dev_e - exact).abs() < threshold,
            "{blk}: the device lost the second panel (f32 accumulation?)"
        );
    }
}

#[test]
fn a_mixed_path_failure_runs_the_f64_device_path_with_a_counter() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if !ready() {
        return;
    }
    let a = synth::synthetic(150, 4, 17, 0, 61);
    let b = synth::synthetic(150, 3, 18, 0, 62);
    let want = device_energies(&a, &b, Precision::F64);
    FAIL_AS_KERNEL.store(true, Ordering::Relaxed);
    FAIL_AT_BLOCK.store(1, Ordering::Relaxed);
    let s0 = stats();
    let ss = try_u_same_spin_on_device(a.ch(), true);
    let os = try_u_opposite_spin_on_device(a.ch(), b.ch(), true);
    let s1 = stats();
    FAIL_AT_BLOCK.store(usize::MAX, Ordering::Relaxed);
    FAIL_AS_KERNEL.store(false, Ordering::Relaxed);
    assert_eq!(
        (
            s1.mixed_fallback_f64 - s0.mixed_fallback_f64,
            s1.gemm_mixed - s0.gemm_mixed,
            s1.mixed_panels - s0.mixed_panels,
            s1.gemm_offloaded - s0.gemm_offloaded,
            s1.gemm_cpu_cuda_error - s0.gemm_cpu_cuda_error,
        ),
        (2, 0, 0, (2 * a.nocc) as u64, 0),
        "a mixed failure is counted once per call, its finished mixed blocks are not, and the f64 device path completes"
    );
    assert_eq!(ss.unwrap().to_bits(), want[0].to_bits());
    assert_eq!(os.unwrap().to_bits(), want[2].to_bits());
}

/// Exactness anchor: with `mixed_ok = false` the dispatchers run the f64
/// device kernel and return its energies bit for bit, with no mixed GEMM
/// counted, even though the settings allow `rimp2-energy`. (The default
/// `precision = "f64"` half is in `gpu_u_rimp2_resident.rs`: settings are
/// installed once per process.)
#[test]
fn without_mixed_the_unrestricted_energy_is_the_f64_kernel_bit_for_bit() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if !ready() {
        return;
    }
    let a = synth::synthetic(140, 5, 16, 1, 71);
    let b = synth::synthetic(140, 4, 17, 1, 72);
    let f64dev = device_energies(&a, &b, Precision::F64);
    let mixed = device_energies(&a, &b, Precision::Mixed);
    assert_ne!(
        f64dev[0].to_bits(),
        mixed[0].to_bits(),
        "fixture blind to mixed"
    );
    let run = |ok: bool| {
        [
            try_u_same_spin_on_device(a.ch(), ok).unwrap(),
            try_u_same_spin_on_device(b.ch(), ok).unwrap(),
            try_u_opposite_spin_on_device(a.ch(), b.ch(), ok).unwrap(),
        ]
    };
    // settings allow mixed, the caller says no (every caller but Coulomb u_ri_mp2)
    let s0 = stats();
    let gated = run(false);
    let s1 = stats();
    assert_eq!(
        s1.gemm_mixed, s0.gemm_mixed,
        "a mixed GEMM ran with mixed_ok = false"
    );
    // positive control: the same dispatchers with mixed_ok = true take the mixed kernel
    let allowed = run(true);
    for k in 0..3 {
        assert_eq!(
            gated[k].to_bits(),
            f64dev[k].to_bits(),
            "block {k}, mixed_ok false"
        );
        assert_eq!(
            allowed[k].to_bits(),
            mixed[k].to_bits(),
            "block {k}, mixed_ok true"
        );
    }
    // and the f64 kernel is the f64 kernel: inside the summation-order gate
    assert!((f64dev[0] - cpu_same(&a)).abs() <= bound::same_spin_bound(&a));
    assert!((f64dev[2] - cpu_opp(&a, &b)).abs() <= bound::opposite_spin_bound(&a, &b));
}
