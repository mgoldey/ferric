//! Memory pre-flight of the UHF RI-K Hessian: one exchange channel per spin
//! (`ferric_scf::hessian_df_k::rik_peak_bytes_channels`), charged to the global
//! pool before the first allocation; the same estimate is the pre-SCF probe
//! `Auto` consults (`analytic_uhf_hessian_available`), with the α and β
//! occupied counts of the molecule.
//!
//! Own test binary: the pool is process-global, so installing a small one must
//! not share a process with the numerical tests; ONE test function, so that
//! nothing runs concurrently with an installed pool.
//!
//! Hypotheses. Right: a pool one byte below the documented two-channel peak is
//! refused with a typed error naming the RI-K Hessian reservation, a pool at
//! the peak plus the raw three-index tensor runs and returns the skeleton it
//! returns with no pool; the pre-SCF probe refuses under the small pool and
//! accepts under the larger; `Auto` then falls back to finite differences while
//! `Analytic` stays an error; the two-channel charge exceeds the one-channel
//! charge (a probe that sized only one channel would be too small). Artifact:
//! the charge ignores the second channel (a pool at the one-channel peak would
//! run), is too large (the pool just below it succeeds), or the estimate stops
//! growing like N^4.

use ferric_core::basis;
use ferric_core::memory::pool::{clear_global, install_global, MemoryPool};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::frequencies::{
    harmonic_frequencies, FrequencyConfig, FrequencyReference, HessianMethod, HessianSource,
};
use ferric_scf::hessian::analytic_uhf_hessian_available;
use ferric_scf::hessian::rij_support::{
    rik_channel_terms_with_aux, RikChannel, RikMutants, RIK_ALPHA_SPIN,
};
use ferric_scf::hessian_df_k::{rik_peak_bytes, rik_peak_bytes_channels};
use ferric_scf::rhf::RhfConfig;
use ndarray::Array2;

const NH2: &str = "3\nNH2\nN 0.0 0.0 0.0\nH 0.01 0.81 -0.63\nH -0.02 -0.79 -0.64\n";

/// Documented peaks (module doc of `hessian_df_k`), within 1 %.
fn assert_documented(label: &str, got: usize, documented: f64) {
    let rel = (got as f64 - documented).abs() / documented;
    eprintln!("{label}: {got} bytes, documented {documented:.4e}, relative difference {rel:.1e}");
    assert!(rel < 0.01, "{label}: {got} vs documented {documented:.4e}");
}

#[test]
fn the_two_channel_charge_gates_the_run_and_the_probe() {
    // --- the documented numbers and the scaling (no pool involved) ----------
    // One channel is the closed-shell estimate, bit for bit.
    assert_eq!(
        rik_peak_bytes_channels(30, 300, &[60], 1500),
        rik_peak_bytes(30, 300, 60, 1500)
    );
    // UHF doublet NH2 cc-pVDZ / def2-universal-jkfit shape (3 atoms, 24 basis
    // functions, 4 + 3 occupied, 113 auxiliary functions): two channels cost
    // more than the larger one alone.
    let two = rik_peak_bytes_channels(3, 24, &[4, 3], 113);
    let one = rik_peak_bytes_channels(3, 24, &[4], 113);
    eprintln!("NH2 cc-pVDZ/jkfit shape: two channels {two} bytes, alpha alone {one}");
    // Channels run one after the other, so the peak is the largest phase of
    // any channel plus what the finished ones leave resident: a second channel
    // can never LOWER the estimate, and raises it when its phases (with the
    // first channel's outputs resident) are the widest.
    assert!(two >= one);
    assert!(
        rik_peak_bytes_channels(5, 200, &[2, 8], 100) > rik_peak_bytes_channels(5, 200, &[2], 100)
    );
    assert_documented("NH2 two channels (4 + 3)", two, 2.6340e6);
    // 30-atom shape of the closed-shell documentation with an unrestricted
    // pair of channels (60 + 59 occupied).
    assert_documented(
        "30 atoms, 300 bf, m = 60 + 59, A = 1500",
        rik_peak_bytes_channels(30, 300, &[60, 59], 1500),
        9.8244e9,
    );
    // N^4 growth with the per-atom shape A = 50 N, n = 10 N, m = 2 N (+/- 1).
    let at = |n: usize| rik_peak_bytes_channels(n, 10 * n, &[2 * n, 2 * n - 1], 50 * n) as f64;
    let slope = (at(60) / at(30)).log2();
    eprintln!("two-channel scaling exponent 30 -> 60 atoms {slope:.2}");
    assert!((3.6..4.1).contains(&slope), "exponent {slope}");

    if ferric_integrals::engine::libint_max_deriv_order() < 1 {
        eprintln!("SKIP: libint2 without first derivatives");
        return;
    }
    // --- the reservation inside the call -------------------------------------
    let mol = Molecule::parse_xyz(NH2, 0, 2).unwrap();
    let prep = PreparedBasis::new(&mol, &basis::bundled("6-31g").unwrap()).unwrap();
    let aux = || PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let n = prep.nbasis();
    let naux = aux().nbasis();
    // A one-column channel followed by a four-column one: the second channel's
    // phases (first channel's outputs resident) set the peak.
    let xa = Array2::from_shape_fn((n, 1), |(i, c)| 0.3 + 0.1 * (i + c) as f64);
    let xb = Array2::from_shape_fn((n, 4), |(i, c)| 0.2 + 0.1 * (i + 2 * c) as f64);
    let run = |two_channels: bool| {
        let mut ch = vec![RikChannel {
            x: &xa,
            alpha: RIK_ALPHA_SPIN,
        }];
        if two_channels {
            ch.push(RikChannel {
                x: &xb,
                alpha: RIK_ALPHA_SPIN,
            });
        }
        rik_channel_terms_with_aux(
            3,
            &prep,
            aux(),
            Operator::coulomb(),
            usize::MAX,
            &ch,
            &RikMutants::default(),
        )
    };
    clear_global();
    let reference = run(true).expect("no pool: inert");
    let need = rik_peak_bytes_channels(3, n, &[1, 4], naux);
    let need_one = rik_peak_bytes_channels(3, n, &[1], naux);
    let raw = 8 * naux * n * n;
    assert!(need > need_one);

    install_global(MemoryPool::with_capacity_bytes(need - 1));
    let err = run(true)
        .expect_err("one byte below the two-channel peak must refuse")
        .to_string();
    eprintln!("pool at peak - 1: {err}");
    assert!(err.contains("RI-K Hessian"), "{err}");

    // A pool that fits ONE channel's peak but not two refuses the two-channel
    // run (artifact: a charge that ignored the second channel would run).
    install_global(MemoryPool::with_capacity_bytes(need_one + raw));
    assert!(need > need_one, "test shape cannot separate the charges");
    run(true).expect_err("a pool sized for one channel must refuse two");
    run(false).expect("and still runs the one-channel call");

    install_global(MemoryPool::with_capacity_bytes(need + 2 * raw));
    let got = run(true).expect("the peak plus the raw tensor must fit");
    let d = got
        .skeleton
        .iter()
        .zip(reference.skeleton.iter())
        .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
    eprintln!("peak {need} bytes (+ raw {raw}); skeleton difference with/without pool {d:.1e}");
    assert_eq!(d, 0.0);

    // --- the pre-SCF probe and the dispatch ----------------------------------
    // The probe sizes the molecule's own (alpha, beta) = (5, 4)... NH2 doublet
    // has 9 electrons: 5 alpha, 4 beta.
    let cfg = RhfConfig {
        df_j_aux: Some(String::new()),
        df_k_aux: Some("cc-pvdz-ri".into()),
        ..Default::default()
    };
    let op = Operator::coulomb();
    let probe_need = rik_peak_bytes_channels(3, n, &[5, 4], naux);
    install_global(MemoryPool::with_capacity_bytes(probe_need - 1));
    let e = analytic_uhf_hessian_available(&mol, &prep, op, &cfg)
        .expect_err("probe must refuse below the two-channel estimate")
        .to_string();
    eprintln!("probe below: {e}");
    assert!(e.contains("memory pool") && e.contains("UHF"), "{e}");
    install_global(MemoryPool::with_capacity_bytes(probe_need + 2 * raw));
    if ferric_integrals::engine::libint_max_deriv_order() >= 2 {
        analytic_uhf_hessian_available(&mol, &prep, op, &cfg).expect("probe accepts above");
    }

    // Auto falls back to finite differences, Analytic stays an error.
    install_global(MemoryPool::with_capacity_bytes(probe_need - 1));
    let freq = |method| {
        harmonic_frequencies(
            &ParallelContext::default(),
            &mol,
            "6-31g",
            op,
            &cfg,
            &FrequencyConfig {
                hessian: method,
                reference: FrequencyReference::Uhf,
                ..Default::default()
            },
        )
    };
    let auto = freq(HessianMethod::Auto);
    let strict = freq(HessianMethod::Analytic);
    clear_global();
    let auto = auto.expect("Auto must fall back to finite differences");
    assert_eq!(auto.hessian_source, HessianSource::FiniteDifference);
    assert!(strict.is_err(), "Analytic must stay an error");
}

/// Live and peak bytes the process has allocated through the global allocator.
/// An exact instrument for "what does the call allocate", unlike the resident
/// set (freed memory the allocator keeps is not an allocation).
struct CountingAlloc;

static LIVE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static PEAK: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

// SAFETY: forwards every call to the system allocator unchanged and only
// updates two atomic counters.
unsafe impl std::alloc::GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, l: std::alloc::Layout) -> *mut u8 {
        use std::sync::atomic::Ordering::Relaxed;
        let p = std::alloc::System.alloc(l);
        if !p.is_null() {
            let now = LIVE.fetch_add(l.size(), Relaxed) + l.size();
            PEAK.fetch_max(now, Relaxed);
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, l: std::alloc::Layout) {
        LIVE.fetch_sub(l.size(), std::sync::atomic::Ordering::Relaxed);
        std::alloc::System.dealloc(p, l)
    }
    unsafe fn alloc_zeroed(&self, l: std::alloc::Layout) -> *mut u8 {
        use std::sync::atomic::Ordering::Relaxed;
        let p = std::alloc::System.alloc_zeroed(l);
        if !p.is_null() {
            let now = LIVE.fetch_add(l.size(), Relaxed) + l.size();
            PEAK.fetch_max(now, Relaxed);
        }
        p
    }
    unsafe fn realloc(&self, p: *mut u8, l: std::alloc::Layout, new: usize) -> *mut u8 {
        use std::sync::atomic::Ordering::Relaxed;
        let q = std::alloc::System.realloc(p, l, new);
        if !q.is_null() {
            if new >= l.size() {
                let now = LIVE.fetch_add(new - l.size(), Relaxed) + (new - l.size());
                PEAK.fetch_max(now, Relaxed);
            } else {
                LIVE.fetch_sub(l.size() - new, Relaxed);
            }
        }
        q
    }
}

#[global_allocator]
static ALLOC: CountingAlloc = CountingAlloc;

/// Peak live allocation (bytes) above the live level at the call, over `f`.
fn peak_allocation_of<T>(f: impl FnOnce() -> T) -> (T, usize) {
    use std::sync::atomic::Ordering::Relaxed;
    let base = LIVE.load(Relaxed);
    PEAK.store(base, Relaxed);
    let out = f();
    (out, PEAK.load(Relaxed).saturating_sub(base))
}

fn bound_check(
    label: &str,
    natoms: usize,
    prep: &PreparedBasis,
    aux: PreparedBasis,
    occ: &[usize],
    slack: usize,
) -> (usize, usize, usize) {
    let (n, naux) = (prep.nbasis(), aux.nbasis());
    let xs: Vec<Array2<f64>> = occ
        .iter()
        .enumerate()
        .map(|(k, &m)| {
            Array2::from_shape_fn((n, m), |(i, c)| {
                ((1 + i * (7 + 2 * k) + c * (3 + k)) % 11) as f64 * 0.05
            })
        })
        .collect();
    let channels: Vec<RikChannel<'_>> = xs
        .iter()
        .map(|x| RikChannel {
            x,
            alpha: RIK_ALPHA_SPIN,
        })
        .collect();
    let charge = rik_peak_bytes_channels(natoms, n, occ, naux);
    let raw = 8 * naux * n * n;
    let t = std::time::Instant::now();
    let (res, peak) = peak_allocation_of(|| {
        rik_channel_terms_with_aux(
            natoms,
            prep,
            aux,
            Operator::coulomb(),
            usize::MAX,
            &channels,
            &RikMutants::default(),
        )
    });
    drop(res.unwrap());
    eprintln!(
        "{label}: {natoms} atoms, n = {n}, A = {naux}, m = {occ:?}: charge {charge} B + raw {raw} B, \
         measured peak allocation {peak} B = {:.2}x (charge + raw), {:.1} s",
        peak as f64 / (charge + raw) as f64,
        t.elapsed().as_secs_f64()
    );
    assert!(
        peak <= charge + raw + slack,
        "{label}: measured peak allocation {peak} B exceeds charge {charge} + raw {raw} + slack {slack}"
    );
    (peak, charge, raw)
}

#[test]
#[ignore = "measurement: peak allocation of the RI-K terms against the charge (counting allocator); measured 31 s (box load ~8)"]
fn the_charge_bounds_the_measured_peak_allocation() {
    // Hypothesis (right): the peak live allocation of the call stays below
    // charge + raw three-index tensor + a small allowance for per-thread
    // engine buffers, i.e. the documented peak is an upper bound of what the
    // call allocates. Artifact (estimate too small): the measured peak exceeds
    // it; the ratio is reported either way. The allowance is the
    // thread count times a per-thread engine/partial buffer, measured as the
    // difference between the 1- and N-thread runs when this test is run with
    // RAYON_NUM_THREADS=1 and without.
    if ferric_integrals::engine::libint_max_deriv_order() < 1 {
        eprintln!("SKIP: libint2 without first derivatives");
        return;
    }
    let xyz = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../testdata/molecules/validation/alkane_4.xyz"
    ))
    .expect("alkane_4.xyz");
    let mol = Molecule::parse_xyz(&xyz, 0, 1).unwrap();
    let prep = PreparedBasis::new(&mol, &basis::bundled("sto-3g").unwrap()).unwrap();
    let aux = PreparedBasis::new(&mol, &basis::bundled("def2-universal-jkfit").unwrap()).unwrap();
    bound_check(
        "butane STO-3G/jkfit",
        mol.atoms.len(),
        &prep,
        aux,
        &[8, 7],
        0,
    );
    // Other shapes: a larger basis (n^2 terms), one channel, a channel with
    // very few columns (the metric terms dominate), and unequal channels.
    let big = PreparedBasis::new(&mol, &basis::bundled("6-31g").unwrap()).unwrap();
    let big_aux =
        PreparedBasis::new(&mol, &basis::bundled("def2-universal-jkfit").unwrap()).unwrap();
    bound_check("butane 6-31G/jkfit", 14, &big, big_aux, &[15, 14], 0);
    let aux = || PreparedBasis::new(&mol, &basis::bundled("def2-universal-jkfit").unwrap()).unwrap();
    bound_check("butane 6-31G/jkfit, one channel", 14, &big, aux(), &[15], 0);
    bound_check("butane 6-31G/jkfit, m = 1 + 3", 14, &big, aux(), &[1, 3], 0);
    bound_check("butane 6-31G/jkfit, m = 3 + 30", 14, &big, aux(), &[3, 30], 0);
}
