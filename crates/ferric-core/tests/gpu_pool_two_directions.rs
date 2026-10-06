#![cfg(feature = "gpu")]
//! A device-memory gate must refuse what does not fit AND admit what does.
//! `pool_capacity_above_physical_free_falls_back_per_call` is Review Focus #2:
//! a pool that believes more is free than the card has must not abort the run.
use ferric_core::gpu::pool::DevicePool;
use ferric_core::gpu::{device::device, probe, GpuStatus};

fn skip_without_device() -> bool {
    match probe(0) {
        GpuStatus::Ready(_) => false,
        other => {
            eprintln!("skipping: no CUDA device ({other:?})");
            assert!(std::env::var("FERRIC_GPU_TESTS_REQUIRED").ok().as_deref() != Some("1"));
            true
        }
    }
}

#[test]
fn tight_pool_refuses_and_names_the_shortfall() {
    let pool = DevicePool::with_capacity_bytes(1 << 20);
    let err = pool.reserve("test plane", 2 << 20).unwrap_err().to_string();
    assert!(
        err.contains("test plane") && err.contains("short by"),
        "{err}"
    );
    assert_eq!(
        pool.available_bytes(),
        1 << 20,
        "a refused reserve must not debit"
    );
}

#[test]
fn ample_pool_admits_and_credits_back_on_drop() {
    let pool = DevicePool::with_capacity_bytes(8 << 20);
    {
        let _r = pool.reserve("a", 3 << 20).unwrap();
        let _s = pool.reserve("b", 3 << 20).unwrap();
        assert!(
            pool.reserve("c", 3 << 20).is_err(),
            "two live reservations must compose"
        );
        assert_eq!(pool.available_bytes(), 2 << 20);
    }
    assert_eq!(pool.available_bytes(), 8 << 20);
}

#[test]
fn pool_capacity_above_physical_free_falls_back_per_call() {
    if skip_without_device() {
        return;
    }
    let dev = device(0).unwrap();
    let (free, _total) = dev.mem_info().unwrap();
    // Lie to the ledger: claim 4x the real free memory, then ask the DEVICE for
    // more than it has. The reservation succeeds (ledger) and the allocation
    // must fail as a typed error, not abort.
    let pool = DevicePool::with_capacity_bytes(free.saturating_mul(4));
    let want = free.saturating_add(1 << 30);
    let _r = pool.reserve("oversubscribed", want).unwrap();
    let alloc = dev.stream.alloc_zeros::<u8>(want);
    assert!(alloc.is_err(), "the card cannot hand out more than it has");
}
