#![cfg(feature = "gpu")]
//! Device pool priority between RI-J and DF-K of one SCF. RI-J is built before
//! DF-K each iteration, so an RI-J upload that takes the card first would leave
//! the larger dressed K tensor permanently on the CPU. With a pool that holds
//! the K tensor but not J and K together, J must stay on the CPU and K must run
//! on the device. benzene/def2-SVP: dressed B 58 MB, packed raw B 29 MB; the pool
//! is 75 MB (asserted below from the real tensor sizes, not assumed).
use ferric_core::basis;
use ferric_core::gpu::{install, pool, probe, stats, GpuMode, GpuSettingsExplicit, GpuStatus};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

const POOL_GB: f64 = 0.075;

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
        memory_gb: Some(POOL_GB),
        ..Default::default()
    })
    .expect("install");
    true
}

#[test]
fn a_pool_that_fits_k_but_not_j_and_k_gives_the_card_to_k() {
    if !ready() {
        return;
    }
    let mol = Molecule::load_xyz("../../testdata/molecules/benzene.xyz").unwrap();
    let prep = PreparedBasis::new(&mol, &basis::bundled("def2-svp").unwrap()).unwrap();
    let aux = PreparedBasis::new(&mol, &basis::bundled("def2-universal-jkfit").unwrap()).unwrap();
    let (n, naux) = (prep.nbasis(), aux.nbasis());
    let k_bytes = 8 * (naux * n * n + n * n);
    let j_bytes = 8 * naux * (n * (n + 1) / 2);
    let cap = pool().unwrap().capacity_bytes();
    assert!(
        k_bytes <= cap && cap < k_bytes + j_bytes,
        "fixture no longer fits K but not J+K: K {k_bytes} B, J {j_bytes} B, pool {cap} B"
    );
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let cfg = RhfConfig {
        density_conv: 1e-6,
        max_iter: 4,
        df_j_aux: Some("def2-universal-jkfit".into()),
        df_k_aux: Some("def2-universal-jkfit".into()),
        ..Default::default()
    };
    let a = stats();
    // A few iterations are enough: both tensors are placed on the first build.
    let _ = solve_rhf(&ParallelContext::default(), &mol, &prep, op, &bounds, &cfg);
    let b = stats();
    assert!(
        b.dfk_device_builds > a.dfk_device_builds,
        "DF-K never ran on the device (K declined: {})",
        b.dfk_declined - a.dfk_declined
    );
    assert_eq!(
        b.dfj_device_builds, a.dfj_device_builds,
        "RI-J took the card from DF-K"
    );
    assert_eq!(b.dfj_declined, a.dfj_declined + 1);
}
