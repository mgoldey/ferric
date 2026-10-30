#![cfg(feature = "gpu")]
//! A page-locked [`HostBuf`] on the device: it is granted (a GTX-class box has
//! far more than a few KiB of lockable memory), starts zeroed, hands out exact
//! heads and refuses an overrun with a typed `Layout` error, and carries an
//! upload to a device buffer and a download back bit for bit, f64 and f32.
//!
//! Limit of the zero check: on this driver (GTX 1080, Linux) a fresh
//! page-locked allocation is already zero-filled by the OS, also after a
//! same-size buffer was dirtied and freed, so the check pins the contract but
//! does NOT catch the removal of `HostBuf::pinned`'s explicit zeroing (mutation
//! checked: it survives).
use ferric_core::gpu::device::{device, GpuError};
use ferric_core::gpu::host::HostBuf;
use ferric_core::gpu::{probe, GpuStatus};

static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn skip() -> bool {
    match probe(0) {
        GpuStatus::Ready(_) => false,
        o => {
            eprintln!("skipping: no CUDA device ({o:?})");
            assert!(
                std::env::var("FERRIC_GPU_TESTS_REQUIRED").ok().as_deref() != Some("1"),
                "FERRIC_GPU_TESTS_REQUIRED=1 but no device"
            );
            true
        }
    }
}

#[test]
fn a_pinned_buffer_is_zeroed_bounded_and_round_trips_through_the_device() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if skip() {
        return;
    }
    let dev = device(0).unwrap();
    // Not a multiple of anything convenient, so the tail is exercised.
    let n = 4099usize;
    let mut up = HostBuf::<f64>::pinned(&dev, n).expect("page-locked allocation refused");
    assert!(up.is_pinned());
    assert_eq!((up.len(), up.bytes()), (n, 8 * n));
    assert!(
        up.head(n).unwrap().iter().all(|v| v.to_bits() == 0),
        "a fresh page-locked buffer is not zeroed"
    );
    // Exact heads, typed refusal one past the end (read and write).
    assert_eq!(up.head(0).unwrap().len(), 0);
    assert_eq!(up.head_mut(n).unwrap().len(), n);
    assert!(matches!(up.head(n + 1), Err(GpuError::Layout(_))));
    assert!(matches!(up.head_mut(n + 1), Err(GpuError::Layout(_))));

    // Values with every kind of bit pattern a wave carries, incl. -0.0.
    let want: Vec<f64> = (0..n)
        .map(|i| match i % 5 {
            0 => -0.0,
            1 => (i as f64).sqrt() * 1e-300,
            2 => -(i as f64) * 1.25e17,
            3 => 1.0 / (i as f64 + 3.0),
            _ => f64::MIN_POSITIVE * i as f64,
        })
        .collect();
    up.head_mut(n).unwrap().copy_from_slice(&want);
    let mut d = dev.stream.alloc_zeros::<f64>(n).unwrap();
    dev.stream.memcpy_htod(up.head(n).unwrap(), &mut d).unwrap();
    let mut down = HostBuf::<f64>::pinned(&dev, n).unwrap();
    // A partial download fills exactly its head; the rest stays zero.
    let half = n / 2;
    dev.stream
        .memcpy_dtoh(&d.slice(..half), down.head_mut(half).unwrap())
        .unwrap();
    dev.stream.synchronize().unwrap();
    let got = down.head(n).unwrap();
    assert!(got[..half]
        .iter()
        .zip(&want[..half])
        .all(|(a, b)| a.to_bits() == b.to_bits()));
    assert!(got[half..].iter().all(|v| v.to_bits() == 0));
    dev.stream
        .memcpy_dtoh(&d, down.head_mut(n).unwrap())
        .unwrap();
    dev.stream.synchronize().unwrap();
    assert!(
        down.head(n)
            .unwrap()
            .iter()
            .zip(&want)
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "the f64 round trip through the device changed a value"
    );

    // f32 (the mixed operands' stage) too.
    let mut up32 = HostBuf::<f32>::pinned(&dev, 257).unwrap();
    assert!(up32.is_pinned() && up32.head(257).unwrap().iter().all(|v| *v == 0.0));
    let want32: Vec<f32> = (0..257).map(|i| (i as f32 - 128.0) * 0.375).collect();
    up32.head_mut(257).unwrap().copy_from_slice(&want32);
    let mut d32 = dev.stream.alloc_zeros::<f32>(257).unwrap();
    dev.stream
        .memcpy_htod(up32.head(257).unwrap(), &mut d32)
        .unwrap();
    let mut back32 = vec![0.0f32; 257];
    dev.stream.memcpy_dtoh(&d32, &mut back32).unwrap();
    dev.stream.synchronize().unwrap();
    assert_eq!(back32, want32);
}
