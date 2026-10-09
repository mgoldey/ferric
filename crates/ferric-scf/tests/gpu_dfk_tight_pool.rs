#![cfg(all(feature = "gpu", feature = "test-seams"))]
//! A process pool too small for the dressed tensor: the dispatcher runs the CPU
//! path, counts `PoolFull` exactly ONCE (the decline is sticky, not per
//! iteration), uploads and computes nothing on the device, leaks no
//! reservation, and returns the forced-host K bit for bit. The ample direction
//! is gpu_dfk_resident.rs. The RSH-pair direction (first DfK fits, second does
//! not) is the last test.
use ferric_core::basis;
use ferric_core::gpu::{install, pool, probe, stats, GpuMode, GpuSettingsExplicit, GpuStatus};
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::df_k::DfK;
use ferric_scf::df_k_gpu::FORCE_HOST;
use ferric_scf::fock::KBuilder;
use ndarray::Array2;
use std::sync::atomic::Ordering;

static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Pool of 100 MB: benzene/def2-SVP's B is 58 MB, so ONE tensor fits and two do not
/// (asserted below, not assumed); 1 MB would also make the first test tight.
const POOL_GB: f64 = 0.100;

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
            memory_gb: Some(POOL_GB),
            ..Default::default()
        })
        .expect("install");
    });
    true
}

fn benzene_dfk() -> (DfK<'static>, usize) {
    let mol = Molecule::load_xyz("../../testdata/molecules/benzene.xyz").unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled("def2-svp").unwrap()).unwrap();
    let aux = PreparedBasis::new(&mol, &basis::bundled("def2-universal-jkfit").unwrap()).unwrap();
    let n = obs.nbasis();
    (
        DfK::new(Operator::coulomb(), &obs, &aux, usize::MAX).unwrap(),
        n,
    )
}

fn c_occ(n: usize, nocc: usize) -> Array2<f64> {
    Array2::from_shape_fn((n, nocc), |(mu, i)| {
        0.05 * (((mu * 5 + i * 3) % 17) as f64 - 8.0)
    })
}

fn bits(k: &Array2<f64>) -> Vec<u64> {
    k.iter().map(|v| v.to_bits()).collect()
}

#[test]
fn a_tensor_larger_than_the_pool_runs_the_cpu_path_bit_identically_and_counts_it_once() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if !ready() {
        return;
    }
    let (mut dfk, n) = benzene_dfk();
    let band = dfk.dressed_incore_flat_for_test().unwrap().nrows();
    let b_bytes = 8 * band * n * n;
    let cap = pool().unwrap().capacity_bytes();
    assert_eq!(
        pool().unwrap().available_bytes(),
        cap,
        "another test left a reservation behind"
    );
    // second tensor must NOT fit (so the RSH test below is a real exhaustion test)
    assert!(
        b_bytes < cap && 2 * b_bytes > cap,
        "fixture no longer exercises one-fits-two-do-not: B {b_bytes} B, pool {cap} B"
    );
    // make THIS test tight: occupy the pool until B cannot fit
    let hog = pool()
        .unwrap()
        .reserve("test hog", cap - b_bytes / 2)
        .unwrap();
    let c = c_occ(n, 21);
    FORCE_HOST.store(true, Ordering::SeqCst);
    let mut k_host = Array2::zeros((n, n));
    dfk.build_from_occ(&c, &mut k_host).unwrap();
    FORCE_HOST.store(false, Ordering::SeqCst);
    let before = pool().unwrap().available_bytes();
    let a = stats();
    let mut k = Array2::zeros((n, n));
    for _ in 0..3 {
        dfk.build_from_occ(&c, &mut k).unwrap();
        assert_eq!(
            bits(&k),
            bits(&k_host),
            "a tight pool must give the forced-host K bit for bit"
        );
    }
    let b = stats();
    assert_eq!(
        b.gemm_cpu_pool_full,
        a.gemm_cpu_pool_full + 1,
        "PoolFull must be counted once, not per build"
    );
    assert_eq!(b.dfk_declined, a.dfk_declined + 1);
    assert_eq!(b.dfk_device_builds, a.dfk_device_builds);
    assert_eq!(
        b.resident_uploads, a.resident_uploads,
        "a refused reservation must move no bytes"
    );
    assert_eq!(b.bytes_h2d, a.bytes_h2d);
    assert_eq!(
        pool().unwrap().available_bytes(),
        before,
        "the failed attempt leaked a reservation"
    );
    drop(hog);
}

#[test]
fn the_second_dfk_of_a_pair_stays_on_the_cpu_when_only_one_fits() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if !ready() {
        return;
    }
    let ((mut first, n), (mut second, _)) = (benzene_dfk(), benzene_dfk());
    let c = c_occ(n, 21);
    assert_eq!(
        pool().unwrap().available_bytes(),
        pool().unwrap().capacity_bytes(),
        "another test left a reservation behind"
    );
    let a = stats();
    let (mut k1, mut k2) = (Array2::zeros((n, n)), Array2::zeros((n, n)));
    first.build_from_occ(&c, &mut k1).unwrap(); // takes the card
    second.build_from_occ(&c, &mut k2).unwrap(); // pool exhausted: CPU
    let b = stats();
    assert_eq!(
        b.dfk_device_builds,
        a.dfk_device_builds + 1,
        "exactly the first DfK must run on the device"
    );
    assert_eq!(b.gemm_cpu_pool_full, a.gemm_cpu_pool_full + 1);
    assert_eq!(b.dfk_declined, a.dfk_declined + 1);
    // the second tensor's K is the CPU K: compare with a forced-host build of the first tensor (same B)
    FORCE_HOST.store(true, Ordering::SeqCst);
    let mut k_host = Array2::zeros((n, n));
    second.build_from_occ(&c, &mut k_host).unwrap();
    FORCE_HOST.store(false, Ordering::SeqCst);
    assert_eq!(bits(&k2), bits(&k_host));
    drop(first); // its device memory is returned to the pool when the DfK drops
    assert_eq!(
        pool().unwrap().available_bytes(),
        pool().unwrap().capacity_bytes(),
        "dropping the DfK must free its device memory"
    );
}

/// Same SCF, same binary: once with the device path forced off (the code path of a
/// mode-off or non-gpu build) and once with a pool that refuses the dressed tensor.
/// The energies must agree to the last bit, because the tight-pool run IS the CPU run.
#[test]
fn scf_energy_under_a_refusing_pool_is_bit_identical_to_the_forced_host_run() {
    use ferric_core::gpu::device::GpuError;
    use ferric_core::parallel::ParallelContext;
    use ferric_core::FerricError;
    use ferric_scf::df_k_gpu::DeviceDfK;
    use ferric_scf::rhf::{solve_rhf, RhfConfig};
    use ferric_scf::screening::SchwarzBounds;

    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if !ready() {
        return;
    }
    let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
    let prep = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
    let aux = PreparedBasis::new(&mol, &basis::bundled("def2-universal-jkfit").unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let cfg = RhfConfig {
        df_j_aux: Some("def2-universal-jkfit".into()),
        df_k_aux: Some("def2-universal-jkfit".into()),
        ..Default::default()
    };
    let run = || match solve_rhf(&ParallelContext::default(), &mol, &prep, op, &bounds, &cfg) {
        Ok(r) => (r.energy.to_bits(), r.iterations),
        Err(FerricError::ScfConvergence { last_energy, .. }) => (last_energy.to_bits(), usize::MAX),
        Err(e) => panic!("unexpected SCF error: {e:?}"),
    };
    let cap = pool().unwrap().capacity_bytes();
    assert_eq!(
        pool().unwrap().available_bytes(),
        cap,
        "another test left a reservation behind"
    );

    // The forced-host run puts RI-J on the CPU too: the tight run below cannot
    // hold the RI-J tensor either, and RI-J on the device is not bit-identical to
    // the host.
    FORCE_HOST.store(true, Ordering::SeqCst);
    ferric_scf::df_j_gpu::FORCE_HOST.store(true, Ordering::SeqCst);
    let host = run();
    FORCE_HOST.store(false, Ordering::SeqCst);
    ferric_scf::df_j_gpu::FORCE_HOST.store(false, Ordering::SeqCst);

    // Leave room for the small K accumulator but not for B (B is ~0.5 MB here).
    const LEFT: usize = 100_000;
    let hog = pool().unwrap().reserve("test hog", cap - LEFT).unwrap();
    let holder = DfK::new(op, &prep, &aux, usize::MAX).unwrap();
    let flat = holder.dressed_incore_flat_for_test().unwrap().to_owned();
    assert!(
        8 * flat.len() > LEFT,
        "fixture: B must not fit in the bytes left"
    );
    let dev = ferric_core::gpu::device::device(0).unwrap();
    match DeviceDfK::upload(&dev, &pool().unwrap(), &flat.view(), prep.nbasis()) {
        Err(GpuError::PoolFull { label, .. }) => assert_eq!(label, "DF-K dressed B"),
        Ok(_) => panic!("B fit in the pool the test made too small"),
        Err(e) => panic!("{e:?}"),
    }
    assert_eq!(
        pool().unwrap().available_bytes(),
        LEFT,
        "the refused upload leaked its K accumulator"
    );

    let a = stats();
    let tight = run();
    let b = stats();
    drop(hog);
    assert_eq!(
        tight, host,
        "the refused-pool SCF is not the CPU SCF (energy bits, iterations)"
    );
    assert!(
        b.gemm_cpu_pool_full > a.gemm_cpu_pool_full,
        "no PoolFull was counted: the device path was never tried"
    );
    assert_eq!(
        b.dfk_declined - a.dfk_declined,
        b.gemm_cpu_pool_full - a.gemm_cpu_pool_full
    );
    assert_eq!(b.dfk_device_builds, a.dfk_device_builds);
    assert_eq!(
        b.resident_uploads, a.resident_uploads,
        "a refused reservation must move no bytes"
    );
    assert_eq!(
        pool().unwrap().available_bytes(),
        cap,
        "the SCF leaked a reservation"
    );
}

fn water_dfk(budget: usize) -> (DfK<'static>, usize) {
    let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
    let aux = PreparedBasis::new(&mol, &basis::bundled("def2-universal-jkfit").unwrap()).unwrap();
    let n = obs.nbasis();
    (
        DfK::new(Operator::coulomb(), &obs, &aux, budget).unwrap(),
        n,
    )
}

/// B and the K accumulator fit, the half-transform scratch does not: the first build
/// declines (after the upload), counts PoolFull once, returns the forced-host K bit for
/// bit, and gives every byte back.
#[test]
fn b_fits_but_the_scratch_is_refused_declines_on_the_first_build() {
    use ferric_scf::df_k_gpu::resident_bytes;

    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if !ready() {
        return;
    }
    let (mut dfk, n) = water_dfk(usize::MAX);
    let nocc = 5;
    let band = dfk.dressed_incore_flat_for_test().unwrap().nrows();
    let cap = pool().unwrap().capacity_bytes();
    assert_eq!(
        pool().unwrap().available_bytes(),
        cap,
        "another test left a reservation behind"
    );
    // after B and K are resident, one byte less than C_occ's own reservation remains
    let left = resident_bytes(band, n).unwrap() + 8 * n * nocc - 1;
    let hog = pool().unwrap().reserve("test hog", cap - left).unwrap();
    let c = c_occ(n, nocc);
    FORCE_HOST.store(true, Ordering::SeqCst);
    let mut k_host = Array2::zeros((n, n));
    dfk.build_from_occ(&c, &mut k_host).unwrap();
    FORCE_HOST.store(false, Ordering::SeqCst);
    let a = stats();
    let mut k = Array2::zeros((n, n));
    for _ in 0..3 {
        dfk.build_from_occ(&c, &mut k).unwrap();
        assert_eq!(bits(&k), bits(&k_host));
    }
    let b = stats();
    assert_eq!(
        b.gemm_cpu_pool_full,
        a.gemm_cpu_pool_full + 1,
        "PoolFull once, not per build"
    );
    assert_eq!(b.dfk_declined, a.dfk_declined + 1);
    assert_eq!(b.dfk_device_builds, a.dfk_device_builds);
    assert_eq!(
        b.resident_uploads,
        a.resident_uploads + 1,
        "B was uploaded before the scratch refusal"
    );
    assert_eq!(
        pool().unwrap().available_bytes(),
        left,
        "the decline leaked device memory"
    );
    drop(hog);
    assert_eq!(pool().unwrap().available_bytes(), cap);
}

/// A source that spilled (tiny budget) cannot be copied to the device: one sticky
/// decline, nothing uploaded, the CPU K.
#[test]
fn a_spilled_source_declines_once_and_moves_no_bytes() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if !ready() {
        return;
    }
    let n0 = water_dfk(usize::MAX).1;
    let (mut dfk, n) = water_dfk(n0 * n0 * 8 * 3);
    assert!(
        dfk.dressed_incore_flat_for_test().is_none(),
        "fixture: the source must have spilled"
    );
    let c = c_occ(n, 5);
    FORCE_HOST.store(true, Ordering::SeqCst);
    let mut k_host = Array2::zeros((n, n));
    dfk.build_from_occ(&c, &mut k_host).unwrap();
    FORCE_HOST.store(false, Ordering::SeqCst);
    let a = stats();
    let mut k = Array2::zeros((n, n));
    for _ in 0..3 {
        dfk.build_from_occ(&c, &mut k).unwrap();
        assert_eq!(bits(&k), bits(&k_host));
    }
    let b = stats();
    assert_eq!(b.dfk_declined, a.dfk_declined + 1, "declined once, sticky");
    assert_eq!(b.dfk_device_builds, a.dfk_device_builds);
    assert_eq!(b.resident_uploads, a.resident_uploads);
    assert_eq!(b.bytes_h2d, a.bytes_h2d);
    assert_eq!(
        pool().unwrap().available_bytes(),
        pool().unwrap().capacity_bytes()
    );
}
