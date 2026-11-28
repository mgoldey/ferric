//! `memory::sys::release_free_heap` makes RSS count only memory in use.
//!
//! Its own test binary: RSS is process-wide, so a concurrent test in the same
//! process could move it between the two readings.

/// Catches: a `release_free_heap` that does not return the arenas' free pages
/// (a no-op, or a trim of the main arena only: this test frees from a rayon-free
/// spawned test thread, so its blocks sit in a non-main arena).
///
/// 256 MiB in 64 KiB blocks stays under glibc's default mmap threshold
/// (128 KiB), so freeing them leaves the pages in the arena; the fence allocated
/// after them keeps the arena's top from being trimmed by `free` itself. That
/// is the retained-heap state a second SCF in one process used to see.
#[cfg(all(target_os = "linux", target_env = "gnu"))]
#[test]
fn freed_heap_leaves_rss_after_release() {
    use ferric_core::memory::sys::{own_rss_bytes, release_free_heap};
    const BLOCK: usize = 64 << 10;
    const TOTAL: usize = 256 << 20;
    let blocks: Vec<Vec<u8>> = (0..TOTAL / BLOCK)
        .map(|i| vec![(i % 251) as u8 + 1; BLOCK])
        .collect();
    let fence = vec![1u8; 4096];
    let live = own_rss_bytes().expect("VmRSS readable on Linux");
    drop(blocks);
    let retained = own_rss_bytes().unwrap();
    assert!(release_free_heap(), "malloc_trim reported nothing released");
    let after = own_rss_bytes().unwrap();
    std::hint::black_box(&fence);
    // The precondition, so a glibc that starts trimming on free cannot make the
    // assertion below pass vacuously.
    assert!(
        retained + (TOTAL / 2) > live,
        "free returned the blocks itself (live {live}, retained {retained}); the test no longer \
         exercises the retained-heap case"
    );
    assert!(
        retained >= after + TOTAL / 2,
        "release_free_heap left the freed heap resident: {retained} -> {after} bytes"
    );
}
