//! Memory-budgeted aux-blocked 3-index (P|μν) integral source.
//!
//! Serves RAW (un-dressed) 3-center integrals in aux-blocks under a fixed byte
//! budget. In-core when the full tensor fits the budget; disk-spill otherwise.
//! Consumers apply their own metric (V^{-1} for J, V^{-1/2} for K).

use crate::basis_bridge::PreparedBasis;
use crate::operator::Operator;
use ferric_core::memory::plan::{Lifetime, MemoryPlan};
use ferric_core::FerricError;
use ndarray::{Array2, Array3, ArrayView3};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::os::unix::io::AsRawFd;

/// `FERRIC_OOC_TRACE` descriptor: 3-index in-core/spill decision trace (env-only
/// debug toggle). NOTE behavior change: previously `.is_ok()` (any value, incl.
/// `=0`, enabled it); now `=0`/`false`/`off` disable it.
static OOC_TRACE: ferric_core::config::ConfigVar<bool> = ferric_core::config::ConfigVar {
    env_name: "FERRIC_OOC_TRACE",
    default: false,
    parse: ferric_core::config::parse_toggle,
    validate: ferric_core::config::accept_any,
};
fn ooc_trace() -> bool {
    OOC_TRACE.toggle()
}

/// Largest number of aux rows whose (block_naux × nao × nao × 8) bytes fit the
/// budget; at least 1 (a single aux row must always be representable).
fn block_naux_for(budget_bytes: usize, nao: usize) -> usize {
    let row_bytes = nao.saturating_mul(nao).saturating_mul(8).max(1);
    (budget_bytes / row_bytes).max(1)
}

/// Spill-path block size honoring the double-buffered pipeline: the compute
/// thread holds one block (N+1) while the write thread holds another (N), so at
/// most TWO blocks are resident at once. Sizing each block to *half* the budget
/// keeps that pair inside the ceiling. The `scratch` read-back buffer allocated
/// for `DiskSpill` is one block of the same size, so a later streaming pass
/// (one live scratch block) also stays well within budget.
fn spill_block_naux_for(budget_bytes: usize, nao: usize) -> usize {
    block_naux_for(budget_bytes / 2, nao)
}

/// Pure decision + report for the SEQUENTIAL blocked/spilled DF-dressing
/// path's over-budget-by-construction warning (see the comment at its call
/// site in `build_dressed_band`).
///
/// Mirrors the exact `MemoryPlan` reservations the sequential dressing loop
/// makes: the optional in-core `dressed band` (only when `in_core`), plus the
/// TWO co-resident `block_naux·nao²` blocks (`accum` and the `contrib` GEMM
/// result added into it every iteration) — this is the same shape as
/// `spill_block_naux_for` halves for on the RAW path, but `block_naux` here
/// comes from the UN-halved `block_naux_for`, so the pair is ~2x budget by
/// construction on a spilled source.
///
/// Returns `Some(report)` (the human-readable breakdown) when the plan does
/// NOT fit the budget, `None` when it does. Extracted as a pure function
/// (explicit shape arguments, no disk I/O, no `PreparedBasis`) so the
/// warn-unconditionally behavior can be pinned in milliseconds instead of via
/// the full spill machinery.
fn blocked_dressing_overshoot_report(
    budget_bytes: usize, in_core: bool, band: usize, block_naux: usize, nao: usize,
) -> Option<String> {
    let mut plan = MemoryPlan::with_budget_bytes(budget_bytes, "DF dressing (blocked)");
    if in_core {
        plan.reserve("dressed band B[P,mu,nu]", band * nao * nao, Lifetime::Resident);
    }
    plan.reserve("dressing accum block", block_naux * nao * nao, Lifetime::Resident);
    plan.reserve("dressing GEMM contrib block", block_naux * nao * nao, Lifetime::Resident);
    if plan.check().is_err() {
        Some(plan.report())
    } else {
        None
    }
}

#[doc(hidden)]
pub fn blocked_dressing_overshoot_report_for_test(
    budget_bytes: usize, in_core: bool, band: usize, block_naux: usize, nao: usize,
) -> Option<String> {
    blocked_dressing_overshoot_report(budget_bytes, in_core, band, block_naux, nao)
}

/// Output aux rows per dressing GEMM in `build_dressed_band`'s parallel fast
/// path.
///
/// A COMPILE-TIME constant, and an EVEN one, on purpose. Splitting a GEMM's
/// output rows is only bit-identical to the unsplit product when the row count
/// keeps OpenBLAS's inner accumulation on the same vector lanes. Measured on
/// this box (a (558,558)x(558,2000) product, split every `blk` rows, comparing
/// against the single GEMM):
///   blk = 63 -> 15005 elements differ | 64 -> 0 | 65 -> 14979 differ
///   blk = 127 -> 7459 differ          | 128 -> 0 | 129 -> 7405 differ
///   blk = 279 -> 3732 differ          | 280 -> 0
/// i.e. EVERY odd block size perturbs the result (~7e-15) and every even one is
/// exact — a 2-wide SIMD accumulation effect, not a k-blocking effect as first
/// assumed. So a thread-derived block count (`band / nthreads`) would silently
/// change the dressed tensor, and hence the SCF energy, whenever it landed on an
/// odd value. 64 is even, keeps each GEMM wide enough for BLAS3 efficiency, and
/// gives ~9 blocks at benzene/aTZ (naux = 558) — enough to fill a 12-core box.
///
/// `dressed_tensor_bitidentical_across_thread_counts` pins this; it deliberately
/// forces an ODD split to stay mutation-sensitive.
const DRESS_ROW_BLOCK: usize = 64;

/// Q (k-dimension) rows accumulated per partial GEMM in the dressing.
///
/// An ACCURACY knob as much as a determinism one. Summing the k axis in
/// fixed-size blocks and accumulating the partials is a coarse pairwise
/// summation: partial sums stay smaller, so less magnitude is lost to rounding
/// than in one full-k reduction. Note the in-core path previously did NO
/// blocking at all — an in-core source reports `block_naux = band`, so
/// `block_edges()` yields a single block — i.e. the least accurate option.
///
/// Both axes measured; 128 is the knee, not a guess:
/// ```text
///   k-block | DfK::new (benzene/aTZ, 12 thr) | RMS err vs fsum ref
///   full-k  |  2157 ms                       | 1.00x (baseline)
///   279     |  2232 ms                       | 1.19x better
///   128     |  2494 ms  (+16%)               | 1.42x better   <- chosen
///   64      |  3144 ms  (+46%)               | 1.82x better
///   32      |  4072 ms  (+89%)               | 2.02x better
/// ```
/// Cross-checked on ferric's REAL dressing operands (benzene/def2-SVP, versus a
/// Kahan-compensated reference over the same integrals): full-k 1.18e-16 RMS vs
/// k-blocked 3.07e-17 — 3.9x better. The SCF energy shifts by ~5e-9 Ha when this
/// changes, and that shift is TOWARD the reference, not away.
const DRESS_K_BLOCK: usize = 128;

/// Evict the file's pages from the OS page cache.
///
/// CRITICAL for memory safety under a cgroup budget: written/read file pages are
/// charged to the cgroup's `memory.current` (the `file` component of
/// `memory.stat`) until the kernel reclaims them. When spilling a tensor far
/// larger than the budget (e.g. a 15 GB temp file under an 8 GB cap), that page
/// cache accumulates unbounded and OOM-kills the process even though our heap
/// (`anon`) stays within budget. We flush dirty pages to disk then advise the
/// kernel to drop the whole file from cache, keeping the cgroup footprint bound
/// to the heap working set. Best-effort: errors are ignored (cache eviction is
/// an optimization, not a correctness requirement on systems where it's a no-op).
fn drop_page_cache(file: &File) {
    // DONTNEED only drops CLEAN pages; flush dirty pages to disk first.
    let _ = file.sync_data();
    let fd = file.as_raw_fd();
    // offset 0, len 0 == "to end of file".
    // SAFETY: fd is a valid open file descriptor (from as_raw_fd on a live
    // File). posix_fadvise with DONTNEED is advisory and cannot corrupt data.
    unsafe {
        libc::posix_fadvise(fd, 0, 0, libc::POSIX_FADV_DONTNEED);
    }
}

// ---------------------------------------------------------------------------
// Disk-spill preflight.
//
// The spill machinery below (double-buffered producer/writer, rendezvous
// channel, half-budget block sizing) is sound. What was missing was any check
// that the destination could HOLD the write, and any notice to the caller that
// they had crossed a performance cliff.
//
// Measured consequences of the missing check: at alkane_32/def2-TZVP the
// spilled tensor is ~55 GB and `DfK::build` re-reads the WHOLE thing every SCF
// iteration (~6 minutes of pure IO per iteration). A C48/def2-TZVP run would
// write ~185 GB and a C32/def2-QZVP run ~415 GB — the latter being 2x the free
// space on the partition this was measured on, which sat at 94-95% full for the
// duration of that campaign. `cosx_k::check_budget` refuses a comparable
// overcommit with a typed error naming the requirement; this path silently
// degraded instead. That asymmetry is what is fixed here.
// ---------------------------------------------------------------------------

/// Absolute headroom floor: never leave the spill partition with less than this.
///
/// 2 GB is chosen to be larger than the working set of the things that share a
/// `/tmp` with a ferric run and break badly when it fills — the OS/journal
/// scratch, other jobs' temp files, and (on many boxes) the cargo/rustc
/// incremental scratch. Filling the last byte of a shared partition does not
/// just fail *this* job, it fails every unrelated process that needs to write,
/// which is exactly the collateral-damage mode ferric already fights with
/// `scripts/ferric-limited`. This is deliberately NOT tuned to any measurement:
/// the cost of being 2 GB conservative on a 400 GB write is nil, and the cost of
/// being 0 GB conservative is a wedged box.
const SPILL_HEADROOM_BYTES: u64 = 2_000_000_000;

/// Proportional headroom: also never consume more than `1 - this` of the free
/// space. On a multi-TB scratch filesystem a flat 2 GB is noise, and filesystems
/// (ext4/XFS/btrfs) degrade in allocation quality and can fail metadata writes
/// well before literal zero. 2% is the smaller of the two arms on anything under
/// 100 GB free, so on the small partitions where the flat floor matters it is the
/// flat floor that binds; the fraction only takes over on large ones.
const SPILL_HEADROOM_FRACTION: f64 = 0.02;

/// Once-latch for the "you are now spilling to disk" warning. The spill loop
/// runs once per aux BLOCK (thousands of times on a large tensor), so the
/// warning must be latched or it becomes the output. Same idiom as
/// `ferric_gw::cohsex`'s `M_PROJ_WARNED`.
static SPILL_WARNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// The directory the spill file is created in.
///
/// `tempfile::tempfile()` (used below) creates an unlinked file in
/// `std::env::temp_dir()`, which on unix is `$TMPDIR` when set and `/tmp`
/// otherwise. There is NO ferric-specific spill-directory variable — `TMPDIR` is
/// the only override, and this function exists so the preflight stats exactly
/// the directory the writer will use rather than assuming `/tmp`.
fn spill_dir() -> std::path::PathBuf {
    std::env::temp_dir()
}

/// Free bytes available to an unprivileged process on the filesystem holding
/// `path`, or `None` if the path cannot be statted.
///
/// `f_bavail` (not `f_bfree`): the reserved-blocks pool that `f_bfree` includes
/// is unavailable to a normal ferric process, so `f_bfree` would over-promise by
/// the usual 5% root reservation.
///
/// `None` on failure is deliberate. A missing/unstattable spill directory means
/// the probe could not measure, not that the disk is full: fabricating `0` there
/// would refuse every spill on any system whose `statvfs` we cannot read, which
/// is a worse failure than the one being fixed. The caller degrades to the old
/// (unchecked) behaviour in that case — but still warns.
fn free_bytes_at(path: &std::path::Path) -> Option<u64> {
    use std::os::unix::ffi::OsStrExt;
    let c_path = std::ffi::CString::new(path.as_os_str().as_bytes()).ok()?;
    // SAFETY: `c_path` is a valid NUL-terminated C string that outlives the
    // call, and `buf` is a correctly-sized, properly-aligned `statvfs` we hand
    // over exclusively for the duration of the call. `statvfs` only writes into
    // `buf` and returns 0/-1.
    let mut buf = std::mem::MaybeUninit::<libc::statvfs>::uninit();
    let rc = unsafe { libc::statvfs(c_path.as_ptr(), buf.as_mut_ptr()) };
    if rc != 0 {
        return None;
    }
    // SAFETY: statvfs returned 0, so `buf` is fully initialized.
    let st = unsafe { buf.assume_init() };
    // f_frsize is the fragment size the f_b* counts are expressed in.
    //
    // The widening goes through `try_from` because libc's statvfs field types
    // are PLATFORM-DEPENDENT (`u64` on linux-gnu, `u32` on some 32-bit targets,
    // `c_ulong` in general). `as` would be a same-type cast here and a lossy one
    // elsewhere; `u64::from` is not implemented for every candidate type. This
    // form compiles unchanged on all of them, and the `ok()?` arm is genuinely
    // unreachable on any target where the field is unsigned and ≤64 bits —
    // which is all of them — so it degrades to "could not measure", never to a
    // fabricated figure.
    //
    // clippy's `useless_conversion` fires because on THIS target (linux-gnu)
    // the fields already are `u64`, making the conversion an identity. Allowed
    // rather than removed: dropping it would silently truncate on any target
    // where the field is narrower, and re-adding it under a `cfg` for a
    // one-line widening is worse than the lint.
    #[allow(clippy::useless_conversion)]
    let frsize = u64::try_from(st.f_frsize).ok()?;
    #[allow(clippy::useless_conversion)]
    let bavail = u64::try_from(st.f_bavail).ok()?;
    frsize.checked_mul(bavail)
}

/// The spill preflight: refuse a write that the destination cannot hold with
/// headroom to spare.
///
/// Split out as a PURE function of `(needed, free, dir)` so it is testable
/// without filling a real disk — the live path supplies `free` from
/// [`free_bytes_at`]. The message names the size, the free space and the path,
/// and lists every remedy, so a user never has to read this file to understand
/// the refusal (the standard `cosx_k::check_budget` sets).
fn check_spill_disk(needed: u64, free: u64, dir: &std::path::Path) -> Result<(), FerricError> {
    let headroom = SPILL_HEADROOM_BYTES.max((free as f64 * SPILL_HEADROOM_FRACTION) as u64);
    if needed.saturating_add(headroom) <= free {
        return Ok(());
    }
    Err(FerricError::General(format!(
        "3-index disk spill needs {:.2} GB in {} but only {:.2} GB is free there \
         (a {:.2} GB safety margin is reserved so the spill cannot fill the partition \
         out from under the rest of the system). Remedies: raise the in-core ceiling so \
         the tensor never spills ([memory] budget_gb / FERRIC_MEM_BUDGET_GB); point the \
         spill elsewhere (TMPDIR=/path/with/room); use a smaller auxiliary basis; or use \
         a K builder that does not materialize the 3-index tensor ([scf] k_builder).",
        needed as f64 / 1e9,
        dir.display(),
        free as f64 / 1e9,
        headroom as f64 / 1e9,
    )))
}

/// The text of the spill warning. Pure so its content is testable; emitted by
/// [`warn_spill_once`].
fn spill_warning_text(needed: u64, dir: &std::path::Path) -> String {
    format!(
        "[ferric] warning: 3-index tensor ({:.2} GB) exceeds the memory budget and is being \
         spilled to disk in {}. This is a performance cliff, not just a memory tradeoff: \
         DfK::build re-reads the ENTIRE spilled tensor on every SCF iteration, so each \
         iteration pays roughly (tensor size / disk bandwidth) in pure IO on top of its \
         compute — at {:.2} GB and ~150 MB/s that is ~{:.0} min per iteration. Raise \
         [memory] budget_gb / FERRIC_MEM_BUDGET_GB to keep it in core, or use a smaller \
         auxiliary basis.",
        needed as f64 / 1e9,
        dir.display(),
        needed as f64 / 1e9,
        (needed as f64 / 150e6) / 60.0,
    )
}

/// Emit the spill warning at most once per process. Returns whether it fired
/// (the return value is what makes the once-ness testable).
fn warn_spill_once(needed: u64, dir: &std::path::Path) -> bool {
    if SPILL_WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
        return false;
    }
    eprintln!("{}", spill_warning_text(needed, dir));
    true
}

/// The full preflight run by both spill sites: warn once, and refuse if the
/// destination cannot hold `needed` bytes.
///
/// Called ONLY from inside a spill branch — the in-core path must not stat
/// anything (an unwritable `TMPDIR` has no bearing on a job that never touches
/// disk, and `in_core_path_never_touches_the_spill_directory` pins that).
fn preflight_spill(needed_bytes: usize) -> Result<(), FerricError> {
    let dir = spill_dir();
    let needed = needed_bytes as u64;
    warn_spill_once(needed, &dir);
    match free_bytes_at(&dir) {
        Some(free) => check_spill_disk(needed, free, &dir),
        // Could not measure: proceed (the pre-existing behaviour) rather than
        // refuse a job on an unreadable statvfs. The warning above already told
        // the user a large write is starting.
        None => Ok(()),
    }
}

/// Packed μν-triangle length for `nao` basis functions: `nao*(nao+1)/2`.
///
/// PySCF calls this `nao_pair`. `(P|μν)` is symmetric in μν — `eri3_block`
/// computes only `s2 in 0..=s1` and then writes BOTH `(μν)` and `(νμ)` — so a
/// spill file storing the full square held every off-diagonal twice.
#[inline]
fn packed_pair_len(nao: usize) -> usize {
    nao * (nao + 1) / 2
}

/// Pack `src` (b, nao, nao) into `dst` (b * packed_pair_len(nao)) lower
/// triangle, row-major in (μ, ν≤μ) order.
fn pack_lower_triangle(src: &ArrayView3<'_, f64>, nao: usize, dst: &mut [f64]) {
    let b = src.shape()[0];
    let pair = packed_pair_len(nao);
    debug_assert!(dst.len() >= b * pair);
    for pl in 0..b {
        let base = pl * pair;
        let mut k = 0usize;
        for mu in 0..nao {
            for nu in 0..=mu {
                dst[base + k] = src[(pl, mu, nu)];
                k += 1;
            }
        }
        debug_assert_eq!(k, pair);
    }
}

/// Inverse of [`pack_lower_triangle`]: expand `src` back into the full
/// symmetric `(b, nao, nao)` block, writing BOTH `(μν)` and `(νμ)`.
///
/// This is what keeps the packing a pure STORAGE change: `for_each_block`
/// already hands out a view of `scratch`, so unpacking here means every
/// consumer still sees the identical `(b, nao, nao)` values it saw before.
fn unpack_lower_triangle(src: &[f64], nao: usize, b: usize, dst: &mut Array3<f64>) {
    let pair = packed_pair_len(nao);
    debug_assert!(src.len() >= b * pair);
    for pl in 0..b {
        let base = pl * pair;
        let mut k = 0usize;
        for mu in 0..nao {
            for nu in 0..=mu {
                let v = src[base + k];
                dst[(pl, mu, nu)] = v;
                dst[(pl, nu, mu)] = v;
                k += 1;
            }
        }
    }
}

/// One aux-block of raw (P|μν), rows `[p0, p0+data.shape()[0])`.
#[derive(Debug)]
pub struct AuxBlock<'a> {
    pub p0: usize,
    pub data: ArrayView3<'a, f64>,
}

enum Backend {
    InCore(Array3<f64>),
    DiskSpill { file: File, scratch: Array3<f64> },
    /// Rebuild each aux block on demand instead of storing the tensor.
    ///
    /// The middle ground this type used to lack. There were only two modes --
    /// whole tensor resident, or whole tensor on disk -- so a budget either
    /// admitted an allocation up to 100% of itself or fell off the cliff
    /// [`spill_warning_text`] describes: `DfK::build` re-reads the ENTIRE
    /// spilled tensor on every SCF iteration.
    ///
    /// Recompute has the SAME one-block footprint as the spill path but writes
    /// nothing and re-reads nothing, trading integral recompute for that IO.
    ///
    /// ```text
    ///   backend      resident        disk    per-pass cost
    ///   InCore       naux·nao²·8     none    none
    ///   Recompute    block·nao²·8    NONE    recompute integrals
    ///   DiskSpill    block·nao²·8    full    re-read the whole file
    /// ```
    ///
    /// Holds `Arc<PreparedBasis>` rather than borrowing, deliberately. A
    /// borrow forces a lifetime onto `ThreeIndexSource`, and
    /// `fock_assembly::build_df_jk` builds its `dfbs` as a LOCAL and returns
    /// the `DfJ`/`DfK` that hold the source — so a borrowing backend cannot
    /// outlive it, on exactly the SCF J/K path that benefits most. An `Arc`
    /// owns the base, keeps it alive, and leaves every existing call site and
    /// struct field untouched. `PreparedBasis` is already `Send + Sync` with a
    /// proper `Drop`, so this needs no new unsafe.
    Recompute {
        op: Operator,
        obs: std::sync::Arc<PreparedBasis>,
        dfbs: std::sync::Arc<PreparedBasis>,
        scratch: Array3<f64>,
    },
}

impl std::fmt::Debug for ThreeIndexSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ThreeIndexSource")
            .field("naux", &self.naux)
            .field("nao", &self.nao)
            .field("block_naux", &self.block_naux)
            .field("band", &(self.band_p0..self.band_p1))
            .finish_non_exhaustive()
    }
}

/// Memory-managed source for 3-index integrals (P|μν), supporting blocked I/O.
pub struct ThreeIndexSource {
    /// GLOBAL number of aux functions (full tensor height), NOT the band height.
    /// Consumers slice a (naux, naux) metric / (naux,) coefficient vector with the
    /// GLOBAL aux index reported by `for_each_block`, so this stays global even
    /// when only a band `[band_p0, band_p1)` is resident.
    naux: usize,
    nao: usize,
    block_naux: usize,
    /// GLOBAL aux range this source actually holds: `[band_p0, band_p1)`.
    /// For a non-banded (full) source this is `0..naux`. `for_each_block` reports
    /// GLOBAL aux indices (`blk.p0 ∈ [band_p0, band_p1)`); the underlying storage
    /// is band-local (height `band_p1 - band_p0`).
    band_p0: usize,
    band_p1: usize,
    backend: Backend,
}

impl ThreeIndexSource {
    /// `budget_bytes` is the hard ceiling for the resident raw 3-index footprint.
    /// Builds the FULL aux range `[0, naux)`.
    /// Like [`Self::build`], but RECOMPUTES over-budget blocks instead of
    /// spilling them to disk.
    ///
    /// In-core when the tensor fits the budget — byte-identical to
    /// [`Self::build`] in that case. When it does not fit, this holds one
    /// block and rebuilds each from `eri3_block` as `for_each_block` walks the
    /// tensor, so it writes nothing and re-reads nothing.
    ///
    /// # Which to call
    ///
    /// Prefer this wherever the source is streamed more than once — an SCF
    /// J/K build streams it every iteration, so the spill path pays its whole
    /// file in IO per iteration (11.4 min/iteration at danuglipron/def2-TZVP
    /// scale even after the μν packing). Recompute pays integrals instead.
    ///
    /// [`Self::build`] remains for callers that stream ONCE, where reading a
    /// spilled file back is cheaper than recomputing integrals, and for the
    /// DRESSED path, which cannot be rebuilt from the bases alone (it needs
    /// the metric).
    ///
    /// Takes `Arc`s rather than references so the source can outlive the
    /// caller's bases. A borrow would force a lifetime onto
    /// `ThreeIndexSource`, and `fock_assembly::build_df_jk` builds its `dfbs`
    /// as a LOCAL and returns the `DfJ`/`DfK` that hold the source — so a
    /// borrowing backend cannot outlive it, on exactly the SCF J/K path that
    /// benefits most. An `Arc` owns the base and keeps it alive, which is also
    /// why every existing call site and struct field is untouched.
    /// `PreparedBasis` is already `Send + Sync` with a proper `Drop`, so this
    /// needs no new unsafe.
    pub fn build_recomputing(
        op: Operator,
        obs: std::sync::Arc<PreparedBasis>,
        dfbs: std::sync::Arc<PreparedBasis>,
        budget_bytes: usize,
    ) -> Result<Self, FerricError> {
        let naux = dfbs.nbasis();
        let nao = obs.nbasis();
        let needed = naux.saturating_mul(nao).saturating_mul(nao).saturating_mul(8);
        if needed <= budget_bytes {
            // Fits: identical to `build`'s in-core branch.
            let eri = crate::threeindex::eri3_block(op, &obs, &dfbs, 0, naux)?;
            return Ok(Self {
                naux,
                nao,
                block_naux: naux.max(1),
                band_p0: 0,
                band_p1: naux,
                backend: Backend::InCore(eri),
            });
        }
        // Over budget: one resident block, rebuilt per pass. `spill_block_naux_for`
        // is reused for the width; only ONE block is live here (versus two on the
        // spill pipeline), so that sizing is conservative for this backend.
        let block_naux = spill_block_naux_for(budget_bytes, nao).min(naux.max(1));
        let scratch = Array3::<f64>::zeros((block_naux, nao, nao));
        Ok(Self {
            naux,
            nao,
            block_naux,
            band_p0: 0,
            band_p1: naux,
            backend: Backend::Recompute { op, obs, dfbs, scratch },
        })
    }

    pub fn build(
        op: Operator, obs: &PreparedBasis, dfbs: &PreparedBasis, budget_bytes: usize,
    ) -> Result<Self, FerricError> {
        let naux = dfbs.nbasis();
        Self::build_band(op, obs, dfbs, budget_bytes, 0, naux)
    }

    /// Build only the GLOBAL aux band `[band_p0, band_p1)` of the raw (P|μν)
    /// tensor. The resulting source holds ONLY that band in memory (or spilled),
    /// so its resident footprint is `(band_p1 - band_p0) · nao² · 8`, not the full
    /// tensor — this is the memory lever for MPI aux-band striping: each rank
    /// builds/holds its own band. `for_each_block` reports GLOBAL aux indices.
    ///
    /// `naux()` still returns the GLOBAL count so consumers can size and slice the
    /// full (naux, naux) metric with the global aux index.
    pub fn build_band(
        op: Operator, obs: &PreparedBasis, dfbs: &PreparedBasis, budget_bytes: usize,
        band_p0: usize, band_p1: usize,
    ) -> Result<Self, FerricError> {
        let naux = dfbs.nbasis();
        let nao = obs.nbasis();
        assert!(band_p0 <= band_p1 && band_p1 <= naux, "invalid aux band [{band_p0},{band_p1}) for naux={naux}");
        let band = band_p1 - band_p0;
        let needed = band.saturating_mul(nao).saturating_mul(nao).saturating_mul(8);
        if ooc_trace() {
            eprintln!(
                "[OOC build] naux={naux} band=[{band_p0},{band_p1}) nao={nao} needed={:.2}GB budget={:.2}GB -> {}",
                needed as f64 / 1e9, budget_bytes as f64 / 1e9,
                if needed <= budget_bytes { "InCore" } else { "Spill" },
            );
        }
        if needed <= budget_bytes {
            // In-core: build exactly the band (global rows [band_p0, band_p1)).
            // eri3_block returns a (band, nao, nao) tensor indexed band-locally.
            let eri = crate::threeindex::eri3_block(op, obs, dfbs, band_p0, band_p1)?;
            Ok(Self { naux, nao, block_naux: band.max(1), band_p0, band_p1, backend: Backend::InCore(eri) })
        } else {
            // Double-buffered spill: a producer thread computes block N+1 (via the
            // rayon-parallel `eri3_block`) while this thread writes block N to
            // disk, so compute and I/O overlap instead of serializing. A rendezvous
            // `sync_channel(0)` bounds the pipeline to exactly TWO live blocks at
            // any instant (one being written, one being computed) — the producer's
            // `send` blocks until the writer takes the previous block, so it never
            // runs more than one block ahead. `spill_block_naux_for` sizes each
            // block to half the budget so that pair stays inside the ceiling
            // (budget-honest). The write thread does pure I/O — no rayon here.
            //
            // Block content is byte-identical to the old serial loop: `eri3_block`
            // is write-once per element (see threeindex.rs) and blocks are written
            // in the same p0-ascending order, so the on-disk file — and thus the
            // read-back path in `for_each_block` — is unchanged.
            // Preflight BEFORE creating the file or computing a single block:
            // refuse a spill the destination cannot hold, and announce the
            // per-SCF-iteration re-read cost the caller did not ask for.
            // `needed` is the UNPACKED band size, used for the in-core decision
            // above. What this branch actually WRITES is the packed triangle,
            // ~half that -- so preflight the packed figure or the disk guard
            // refuses spills that would comfortably fit, and the warning quotes
            // a size and an IO time that are both 2x too large.
            let spill_bytes = (band as u64)
                .saturating_mul(packed_pair_len(nao) as u64)
                .saturating_mul(8);
            preflight_spill(spill_bytes as usize)?;
            let block_naux = spill_block_naux_for(budget_bytes, nao);
            let mut file = tempfile::tempfile()
                .map_err(|e| FerricError::General(format!("tempfile: {e}")))?;

            // Channel carries either a computed block or a producer-side error.
            let (tx, rx) = std::sync::mpsc::sync_channel::<Result<Array3<f64>, FerricError>>(0);

            std::thread::scope(|s| -> Result<(), FerricError> {
                // Producer: compute blocks in ascending GLOBAL p0 order over the
                // band [band_p0, band_p1), hand each off. Only band rows are ever
                // computed or spilled — the on-disk file holds exactly the band.
                s.spawn(move || {
                    let mut p0 = band_p0;
                    while p0 < band_p1 {
                        let p1 = (p0 + block_naux).min(band_p1);
                        let blk = crate::threeindex::eri3_block(op, obs, dfbs, p0, p1);
                        let is_err = blk.is_err();
                        // If the receiver hung up (writer hit an I/O error and
                        // returned early), stop producing.
                        if tx.send(blk).is_err() || is_err {
                            return;
                        }
                        p0 = p1;
                    }
                });

                // Consumer (this thread): write each block as it arrives. Pure I/O.
                // Reused across blocks so the pack buffer is allocated once.
                let mut packbuf: Vec<f64> = Vec::new();
                for blk in rx.iter() {
                    let blk = blk?;
                    // Store the PACKED μν triangle: half the bytes written, and
                    // half the bytes re-read on every subsequent streaming pass
                    // (which for DfK is every SCF iteration -- the cost the
                    // spill warning quantifies).
                    let b = blk.shape()[0];
                    let pair = packed_pair_len(nao);
                    packbuf.resize(b * pair, 0.0);
                    pack_lower_triangle(&blk.view(), nao, &mut packbuf);
                    let bytes: &[u8] = bytemuck::cast_slice(&packbuf[..b * pair]);
                    file.write_all(bytes)
                        .map_err(|e| FerricError::General(format!("spill write: {e}")))?;
                    // Evict just-written pages so the cgroup-charged page cache does
                    // not accumulate the whole (>budget) file. See drop_page_cache.
                    drop_page_cache(&file);
                }
                Ok(())
            })?;

            file.flush().ok();
            drop_page_cache(&file);
            let scratch = Array3::<f64>::zeros((block_naux, nao, nao));
            Ok(Self { naux, nao, block_naux, band_p0, band_p1, backend: Backend::DiskSpill { file, scratch } })
        }
    }

    /// Build a DRESSED source: `out[P,:,:] = Σ_Q m[P,Q] · raw[Q,:,:]`, honoring budget.
    /// `raw` is consumed (streamed) and `m` is (naux, naux). Produces the FULL
    /// aux range `[0, naux)`.
    pub fn build_dressed(
        raw: &mut ThreeIndexSource, m: &Array2<f64>, budget_bytes: usize,
    ) -> Result<Self, FerricError> {
        let naux = raw.naux();
        Self::build_dressed_band(raw, m, budget_bytes, 0, naux)
    }

    /// Build a DRESSED source restricted to the GLOBAL aux band `[band_p0,
    /// band_p1)`:  `out[P,:,:] = Σ_Q m[P,Q] · raw[Q,:,:]`  for P ∈ `[band_p0,
    /// band_p1)`. The dressing SUM runs over ALL Q, so `raw` MUST be the FULL
    /// `[0, naux)` source (it is streamed block-by-block; `raw` may itself be
    /// budget-bounded / disk-spilled so its full footprint need not be resident).
    /// The OUTPUT holds only the band — this is the memory lever for MPI DF-K:
    /// each rank dresses/holds only its own aux-band of `B[P,μ,ν]`.
    pub fn build_dressed_band(
        raw: &mut ThreeIndexSource, m: &Array2<f64>, budget_bytes: usize,
        band_p0: usize, band_p1: usize,
    ) -> Result<Self, FerricError> {
        let naux = raw.naux();
        assert!(
            raw.band_p0 == 0 && raw.band_p1 == naux,
            "build_dressed_band requires a FULL raw source (all Q); got raw band [{},{})",
            raw.band_p0, raw.band_p1,
        );
        assert!(band_p0 <= band_p1 && band_p1 <= naux, "invalid aux band [{band_p0},{band_p1}) for naux={naux}");
        let nao = raw.nao();
        let band = band_p1 - band_p0;
        // Sizing is on the BAND footprint (what this rank actually holds), not the
        // full tensor — so a rank's budget applies to ITS band.
        let needed = band.saturating_mul(nao).saturating_mul(nao).saturating_mul(8);
        let block_naux = block_naux_for(budget_bytes, nao);
        let in_core = needed <= budget_bytes;
        if ooc_trace() {
            eprintln!(
                "[OOC dress] naux={naux} band=[{band_p0},{band_p1}) nao={nao} needed={:.2}GB budget={:.2}GB block_naux={block_naux} -> {}",
                needed as f64 / 1e9, budget_bytes as f64 / 1e9,
                if in_core { "InCore" } else { "Spill" },
            );
        }
        // Output storage is band-local (height `band`), addressed by (P - band_p0).
        let mut out_incore: Option<Array3<f64>> =
            if in_core { Some(Array3::zeros((band, nao, nao))) } else { None };
        let mut file: Option<File> =
            if in_core { None } else {
                // Same preflight as the raw path, and for the same reason
                // sized on the PACKED figure: `needed` is the unpacked band
                // (used for the in-core decision above), but what is written is
                // the μν triangle -- roughly half. Preflighting the unpacked
                // number would refuse spills that fit and overstate the IO time
                // in the warning by 2x.
                let spill_bytes = band
                    .saturating_mul(packed_pair_len(nao))
                    .saturating_mul(8);
                preflight_spill(spill_bytes)?;
                Some(tempfile::tempfile().map_err(|e| FerricError::General(format!("tempfile: {e}")))?)
            };
        // FAST PATH: raw and output both fully in core. Each output block is an
        // independent linear combination over Q, so the blocks can be computed
        // concurrently — but ONLY with the SAME boundaries the sequential loop
        // below uses.
        //
        // Bit-identity depends on the block boundaries, not on execution order.
        // Within a block, each output element accumulates over Q-blocks in
        // ascending order; BLAS additionally chooses its internal k-blocking from
        // the operand's row count, so re-chunking the output rows (e.g. into
        // `nthreads` even pieces) changes the summation order and perturbs the
        // result — measured 3.6e-15 elementwise, which moved the benzene/aTZ SCF
        // energy by 2.9e-6 Ha and made it vary with RAYON_NUM_THREADS. Reusing
        // the identical `block_naux` boundaries and identical per-block Q loop
        // makes every output element see the exact same addition sequence, so
        // this is bit-identical to the sequential path at any thread count
        // (df_k.rs's `df_k_bit_identical_across_thread_counts` covers it).
        //
        // Why it matters: the dressing is ~107 GFLOP at benzene/aug-cc-pVTZ and
        // ran entirely on ONE core (BLAS is pinned to 1 thread under rayon per
        // the project convention), so DfK::new scaled only 1.35x across 12 cores.
        if in_core && raw.is_incore() {

            use rayon::prelude::*;
            let raw_flat = raw.incore_flat().expect("is_incore checked");
            let raw_p0 = raw.band().0;
            let arr = out_incore.as_mut().expect("in_core implies Some");
            let q_edges = raw.block_edges();
            // Output-block boundaries from a COMPILE-TIME CONSTANT, never from
            // `block_naux` (budget-derived) or the thread count. This is what
            // makes the result reproducible: BLAS picks its internal k-blocking
            // from the operand's row count, so the boundaries fix the summation
            // order, and holding them constant fixes the result for every thread
            // count, memory budget, and execution order. Deriving them from
            // `nthreads` was the bug that perturbed the benzene/aTZ SCF energy by
            // 2.9e-6 Ha and made it vary with RAYON_NUM_THREADS.
            // Memory accounting for this path. Historically NOTHING here was
            // charged: the `in_core` decision above gates only `out_incore`
            // (one `band·nao²·8` copy), yet this path used to `collect()` every
            // output block into a `Vec<(usize, Array2)>` BEFORE scattering, so
            // a full SECOND copy of the dressed band (another `band·nao²·8`)
            // was co-resident with `out_incore` at the moment the collect
            // finished. At benzene/aug-cc-pVTZ (naux=1512, nao=414) that is
            // 2.1 GB charged as 1.05 GB — the classic under-count that OOMs.
            // The per-worker `acc`/`contrib` scratch was unaccounted too.
            //
            // Two independent fixes, both applied:
            //  1. The second copy is GONE, not merely charged for. Each block
            //     now accumulates into its own destination rows of `arr`
            //     directly via rayon's slice `par_chunks_mut`, so peak
            //     residency is one copy of the band plus per-worker scratch.
            //     This is a pure removal of a copy: the GEMM boundaries
            //     (`DRESS_ROW_BLOCK`), the Q sweep, and the accumulation order
            //     are byte-for-byte what they were, so the dressed tensor is
            //     bit-identical (pinned by
            //     `dressed_tensor_bitidentical_across_thread_counts`).
            //  2. What genuinely remains resident is now DECLARED, so an
            //     over-budget job fails fast with a breakdown instead of
            //     walking into the allocation. `contrib` is the only real
            //     per-worker term left (`acc` is gone with the copy).
            //
            // Note the asymmetry this repairs: the RAW spill path sizes blocks
            // with `spill_block_naux_for` = `block_naux_for(budget/2, nao)`,
            // HALVED because two blocks are live at once. This dressing path
            // used the UN-halved `block_naux_for` while `accum` and `contrib`
            // coexist on the sequential path below — see the guard there.
            let mut out_edges: Vec<(usize, usize)> = Vec::new();
            let mut l = 0usize;
            while l < band {
                let r = (l + DRESS_ROW_BLOCK).min(band);
                out_edges.push((l, r));
                l = r;
            }
            // Concurrency is bounded by the number of blocks as well as by the
            // pool: charging `nthreads` copies when there are only 3 blocks
            // would over-count, and an over-estimating guard refuses jobs that
            // would have fit — as much a defect as an under-estimate.
            let workers = rayon::current_num_threads().max(1).min(out_edges.len().max(1));
            let mut plan = MemoryPlan::with_budget_bytes(budget_bytes, "DF dressing (in-core)");
            plan.reserve("dressed band B[P,mu,nu]", band * nao * nao, Lifetime::Resident);
            // Each worker holds one `contrib` GEMM result: `b × nao²` where
            // `b ≤ DRESS_ROW_BLOCK`. This is the term the old code never
            // charged for at all.
            plan.reserve_per_worker(
                "dressing GEMM contrib (per worker)",
                DRESS_ROW_BLOCK.min(band.max(1)) * nao * nao,
                workers,
            );
            plan.check()?;
            // Scatter-free: each block writes its OWN disjoint rows of `arr`.
            // `arr` is a freshly-allocated contiguous `Array3`, so its backing
            // slice is row-major and output row `l` occupies exactly
            // `[l·nao², (l+1)·nao²)`. `par_chunks_mut(DRESS_ROW_BLOCK·nao²)`
            // therefore yields precisely the `out_edges` blocks, in order, as
            // disjoint mutable slices — the same partition the collect used,
            // without ever materialising a second copy of the band.
            let row_elems = nao * nao;
            let arr_slice = arr.as_slice_mut().ok_or_else(|| {
                FerricError::General("dress out: non-contiguous output band".into())
            })?;
            arr_slice
                .par_chunks_mut(DRESS_ROW_BLOCK * row_elems)
                .enumerate()
                .try_for_each(|(bi, dst)| -> Result<(), FerricError> {
                    let (l0, l1) = out_edges[bi];
                    let b = l1 - l0;
                    let p0 = band_p0 + l0;
                    let mut acc = ndarray::ArrayViewMut2::from_shape((b, row_elems), dst)
                        .map_err(|e| {
                            FerricError::General(format!("dress out reshape: {e}"))
                        })?;
                    // Ascending Q sweep, sub-blocked to DRESS_K_BLOCK. The outer
                    // edges follow the source's own blocks (so a spilled source
                    // keeps its streaming order); the inner split is a fixed
                    // constant, which both pins the summation order and improves
                    // accuracy (see DRESS_K_BLOCK).
                    //
                    // `acc` starts at zero (`arr` was zero-initialised), so
                    // `acc += contrib` accumulates the identical term sequence
                    // the old private buffer did, in the identical order.
                    for &(q0, q1) in q_edges.iter() {
                        let mut qa = q0;
                        while qa < q1 {
                            let qb = (qa + DRESS_K_BLOCK).min(q1);
                            let msub = m.slice(ndarray::s![p0..p0 + b, qa..qb]);
                            let rblk = raw_flat.slice(ndarray::s![qa - raw_p0..qb - raw_p0, ..]);
                            let contrib: Array2<f64> = msub.dot(&rblk);
                            acc += &contrib;
                            qa = qb;
                        }
                    }
                    Ok(())
                })?;
            return Ok(Self {
                naux,
                nao,
                block_naux: band.max(1),
                band_p0,
                band_p1,
                backend: Backend::InCore(out_incore.expect("in_core implies Some")),
            });
        }

        // SEQUENTIAL path accounting.
        //
        // The defect: `block_naux` is `block_naux_for(budget_bytes, nao)` — one
        // block sized to the WHOLE budget — yet the loop below holds TWO such
        // blocks live simultaneously (`accum`, and the `contrib` GEMM result
        // added into it), plus `out_incore` (band·nao²) underneath on the
        // in-core branch. Compare `spill_block_naux_for` on the RAW path, which
        // halves the budget for exactly this two-live-blocks reason; the
        // dressing path never did, so its gate covered at most half of what it
        // actually held.
        //
        // Why this is a WARNING and not a hard `check()?`: the block size is
        // load-bearing for the RESULT. `block_naux` is the GEMM's m dimension
        // (`m.slice(p0..p1)`), so shrinking it to make the pair fit would change
        // BLAS's internal blocking, the summation order, and hence the dressed
        // tensor and the SCF energy — precisely the class of change
        // `DRESS_ROW_BLOCK` exists to prevent. And hard-failing instead would
        // refuse every spilled dressing that runs correctly today, since
        // `block_naux·nao²·8 ≈ budget` makes the honest two-block total ≈ 2×
        // budget by construction. Over-counting that refuses jobs which would
        // have fit is as much a bug as under-counting that OOMs, so the honest
        // figure is REPORTED (with the breakdown) rather than enforced.
        //
        // FIXED 2026-09-10: the report was gated on `ooc_trace()` (the
        // `FERRIC_OOC_TRACE` debug env var, default OFF), so in every
        // production run the "REPORTED" promise above was false — the ~2x
        // overshoot was computed and then silently discarded, and an operator
        // whose job died in a cgroup MemoryMax got neither an error nor a
        // warning. `ooc_trace()` still controls the VERBOSE per-decision
        // in-core/spill trace elsewhere in this file; here it gated the ONLY
        // channel that could ever tell anyone this path is running ~2x over
        // budget by construction, so the two purposes should not share a
        // toggle. Print unconditionally on `check()` failure instead — this
        // adds a stderr line on an already-slow disk-spill path, allocates
        // nothing, and changes no arithmetic (`plan.check()`'s Err is now
        // read, not just computed).
        //
        // The decision (does this shape overshoot the budget?) is pulled out
        // into `blocked_dressing_overshoot_report` so it is testable without
        // driving the whole spill machinery (disk files, a real
        // `PreparedBasis`) — see `blocked_dressing_overshoot_report_for_test`.
        if let Some(report) = blocked_dressing_overshoot_report(budget_bytes, in_core, band, block_naux, nao) {
            eprintln!(
                "[ferric] WARNING: DF dressing (blocked/spilled) exceeds the byte budget \
                 by construction (two live blocks per iteration; block size is fixed \
                 because it sets the GEMM shape and hence the numerical result, so it is \
                 not shrunk to fit):\n{report}"
            );
        }

        let mut p0 = band_p0;
        while p0 < band_p1 {
            let p1 = (p0 + block_naux).min(band_p1);
            let b = p1 - p0;
            let mut accum = Array3::<f64>::zeros((b, nao, nao));
            // Dressing out[P] = Σ_Q m[P,Q] raw[Q] sums over ALL Q — `raw` is the
            // full source, streamed block-by-block. Only this band's output rows
            // [p0, p1) are accumulated.
            raw.for_each_block(|rb| {
                let rb_b = rb.data.shape()[0];
                let raw_flat = rb.data.into_shape_with_order((rb_b, nao * nao))
                    .map_err(|e| FerricError::General(format!("raw reshape: {e}")))?;
                let msub = m.slice(ndarray::s![p0..p1, rb.p0..rb.p0 + rb_b]); // (b, rb_b)
                let contrib = msub.dot(&raw_flat); // (b, nao*nao)
                let mut acc_flat = accum.view_mut().into_shape_with_order((b, nao * nao)).unwrap();
                acc_flat += &contrib;
                Ok(())
            })?;
            if let Some(arr) = out_incore.as_mut() {
                // Band-local destination: global P maps to row (P - band_p0).
                arr.slice_mut(ndarray::s![p0 - band_p0..p1 - band_p0, .., ..]).assign(&accum);
            } else if let Some(f) = file.as_mut() {
                // PACKED, matching the raw spill path -- both are read back by
                // the same `for_each_block` arm, so the two formats MUST agree.
                // (Writing this one unpacked while the reader unpacks produced
                // pure garbage: 5458316/5458320 elements wrong, caught by
                // `spilled_dressed_tensor_stays_within_a_few_ulp_of_in_core`.)
                //
                // Valid here for the same reason as the raw path: the dressing
                // is out[P,μν] = Σ_Q m[P,Q]·raw[Q,μν], and `m` does not touch
                // the μν indices, so a μν-symmetric raw tensor stays symmetric.
                let bl = accum.shape()[0];
                let pair = packed_pair_len(nao);
                let mut packbuf = vec![0.0f64; bl * pair];
                pack_lower_triangle(&accum.view(), nao, &mut packbuf);
                let bytes: &[u8] = bytemuck::cast_slice(&packbuf);
                f.write_all(bytes).map_err(|e| FerricError::General(format!("dress write: {e}")))?;
                drop_page_cache(f);
            }
            p0 = p1;
        }
        let backend = match (out_incore, file) {
            (Some(arr), _) => Backend::InCore(arr),
            (None, Some(f)) => {
                f.sync_all().ok();
                Backend::DiskSpill { file: f, scratch: Array3::zeros((block_naux, nao, nao)) }
            }
            _ => unreachable!(),
        };
        Ok(Self {
            naux,
            nao,
            block_naux: if in_core { band.max(1) } else { block_naux },
            band_p0,
            band_p1,
            backend,
        })
    }

    /// True when the whole band is resident (no disk spill).
    pub fn is_incore(&self) -> bool { matches!(self.backend, Backend::InCore(_)) }

    /// The resident tensor as a flat `(band_naux, nao*nao)` view; `None` when
    /// spilled. Rows are band-local (global P maps to `P - band_p0`).
    pub fn incore_flat(&self) -> Option<ndarray::ArrayView2<'_, f64>> {
        match &self.backend {
            Backend::InCore(eri) => {
                let (b, n1, n2) = eri.dim();
                eri.view().into_shape_with_order((b, n1 * n2)).ok()
            }
            _ => None,
        }
    }

    /// GLOBAL `[q0, q1)` aux ranges that [`Self::for_each_block`] yields, in the
    /// same ascending order. Callers that replace `for_each_block` with their own
    /// loop MUST sweep these exact edges to keep floating-point accumulation
    /// order (and hence bit-identity) unchanged.
    pub fn block_edges(&self) -> Vec<(usize, usize)> {
        let band = self.band_naux();
        let step = self.block_naux.max(1);
        let mut v = Vec::with_capacity(band.div_ceil(step));
        let mut l0 = 0;
        while l0 < band {
            let l1 = (l0 + step).min(band);
            v.push((self.band_p0 + l0, self.band_p0 + l1));
            l0 = l1;
        }
        v
    }

    /// Is this source backed by on-demand recompute?
    ///
    /// Test hook, paired with [`Self::is_spilled_for_test`]: a contract that
    /// cannot see WHICH backend it got cannot distinguish "recomputed
    /// correctly" from "silently still spilling".
    #[doc(hidden)]
    pub fn is_recompute_for_test(&self) -> bool {
        matches!(self.backend, Backend::Recompute { .. })
    }

    /// Is this source backed by the disk spill?
    ///
    /// Test hook. A reachability guard that cannot see which backend it got
    /// asserts nothing — a "spilled" source that silently stayed in core makes
    /// every downstream contract vacuous.
    #[doc(hidden)]
    pub fn is_spilled_for_test(&self) -> bool {
        matches!(self.backend, Backend::DiskSpill { .. })
    }

    /// GLOBAL number of aux functions (full tensor height), regardless of band.
    pub fn naux(&self) -> usize { self.naux }
    /// Number of AO basis functions.
    pub fn nao(&self) -> usize { self.nao }
    /// The GLOBAL aux range `[p0, p1)` this source actually holds. `0..naux` for
    /// a full (non-banded) source.
    pub fn band(&self) -> (usize, usize) { (self.band_p0, self.band_p1) }
    /// Number of aux rows resident in THIS source's band (`band_p1 - band_p0`).
    pub fn band_naux(&self) -> usize { self.band_p1 - self.band_p0 }
    /// Number of aux-blocks in this source's band.
    pub fn n_blocks(&self) -> usize {
        self.band_naux().div_ceil(self.block_naux.max(1))
    }
    /// Aux rows per block (last block may be smaller).
    pub fn block_naux(&self) -> usize { self.block_naux }

    /// Primary iteration API. Calls `f` once per aux-block, in order, over the
    /// resident band. `blk.p0` is the GLOBAL aux index of the block's first row
    /// (∈ [band_p0, band_p1)); the block data is the band-local storage sliced to
    /// that block. Consumers slice a (naux, naux) metric / (naux,) coefficient
    /// vector with `blk.p0` (global), so J/K contributions are placed correctly
    /// whether the source is full or a per-rank band.
    pub fn for_each_block(
        &mut self,
        mut f: impl FnMut(AuxBlock<'_>) -> Result<(), FerricError>,
    ) -> Result<(), FerricError> {
        let band = self.band_naux();
        let band_p0 = self.band_p0;
        match &mut self.backend {
            Backend::InCore(eri) => {
                let nb = band.div_ceil(self.block_naux.max(1));
                for i in 0..nb {
                    // local rows into the band-local storage; global p0 reported.
                    let l0 = i * self.block_naux;
                    let l1 = (l0 + self.block_naux).min(band);
                    let view = eri.slice(ndarray::s![l0..l1, .., ..]);
                    f(AuxBlock { p0: band_p0 + l0, data: view })?;
                }
                Ok(())
            }
            Backend::Recompute { op, obs, dfbs, scratch } => {
                // Rebuild each block from the bases rather than reading it back.
                //
                // BIT-IDENTICAL to what the in-core backend would have stored:
                // `eri3_block` is a pure function of (op, obs, dfbs, p0, p1)
                // and write-once per element, and nothing is summed across
                // blocks here, so there is no reassociation to worry about.
                // Pinned by `mwe_recompute_backend_avoids_disk.rs` CONTRACT 1.
                let nb = band.div_ceil(self.block_naux.max(1));
                for i in 0..nb {
                    let l0 = i * self.block_naux;
                    let l1 = (l0 + self.block_naux).min(band);
                    let p0 = band_p0 + l0;
                    let p1 = band_p0 + l1;
                    let blk = crate::threeindex::eri3_block(*op, obs, dfbs, p0, p1)?;
                    let b = l1 - l0;
                    scratch.slice_mut(ndarray::s![0..b, .., ..]).assign(&blk);
                    let view = scratch.slice(ndarray::s![0..b, .., ..]);
                    f(AuxBlock { p0, data: view })?;
                }
                Ok(())
            }
            Backend::DiskSpill { file, scratch } => {
                file.seek(SeekFrom::Start(0)).map_err(|e| FerricError::General(format!("seek: {e}")))?;
                let mut packbuf: Vec<f64> = Vec::new();
                let nb = band.div_ceil(self.block_naux.max(1));
                for i in 0..nb {
                    let l0 = i * self.block_naux;
                    let l1 = (l0 + self.block_naux).min(band);
                    let b = l1 - l0;
                    // The file holds the PACKED triangle; read that, then expand
                    // into `scratch` so the yielded view keeps its historical
                    // (b, nao, nao) shape and every consumer is untouched.
                    let pair = packed_pair_len(self.nao);
                    let elems = b * pair;
                    packbuf.resize(elems, 0.0);
                    let bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut packbuf[..elems]);
                    file.read_exact(bytes).map_err(|e| FerricError::General(format!("spill read: {e}")))?;
                    unpack_lower_triangle(&packbuf[..elems], self.nao, b, scratch);
                    let view = scratch.slice(ndarray::s![0..b, .., ..]);
                    f(AuxBlock { p0: band_p0 + l0, data: view })?;
                }
                // Reads also populate the cgroup-charged page cache; drop them so
                // a full streaming pass doesn't pull the entire file into cache.
                drop_page_cache(file);
                Ok(())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operator::Operator;
    use crate::basis_bridge::PreparedBasis;
    use ferric_core::basis;
    use ferric_core::mol::Molecule;

    fn water() -> (Molecule,) { (Molecule::parse_xyz("3\nH2O\nO 0 0 0\nH 0 0 0.96\nH 0.93 0 -0.26\n", 0, 1).unwrap(),) }

    /// The dressed tensor must not depend on RAYON_NUM_THREADS.
    ///
    /// `build_dressed_band`'s parallel fast path splits OUTPUT aux rows across
    /// workers. BLAS chooses its internal k-blocking from the operand's row
    /// count, so the row-block boundaries fix the floating-point summation
    /// order — deriving them from the thread count (or from the budget-derived
    /// `block_naux`) silently changes the result per machine. That regression
    /// shifted the benzene/aug-cc-pVTZ SCF energy by 2.9e-6 Ha and only appeared
    /// at RAYON=2 and 12. `DRESS_ROW_BLOCK` is a compile-time constant to
    /// prevent it; this test pins that.
    #[test]
    fn dressed_tensor_bitidentical_across_thread_counts() {
        // Benzene/def2-SVP with the JK-fit aux: naux = 558, so DRESS_ROW_BLOCK
        // gives 9 blocks and each dressing GEMM is large enough that BLAS's
        // internal k-blocking genuinely differs with the operand row count.
        // SMALLER SYSTEMS DO NOT WORK: at methane/aug-cc-pVDZ the band is only
        // 147 and the rounding does not diverge, so the test passes even with
        // thread-derived boundaries (verified by mutation). If this test is ever
        // made cheaper, re-run that mutation check.
        let mol = Molecule::load_xyz("../../testdata/molecules/benzene.xyz").unwrap();
        let obs = PreparedBasis::new(&mol, &basis::bundled("def2-svp").unwrap()).unwrap();
        let aux = PreparedBasis::new(&mol, &basis::bundled("def2-universal-jkfit").unwrap()).unwrap();
        let op = Operator::coulomb();
        let naux = aux.nbasis();
        // A non-trivial, deterministic metric (symmetric, well-conditioned).
        let mut m = Array2::<f64>::zeros((naux, naux));
        for i in 0..naux {
            for j in 0..naux {
                m[(i, j)] = if i == j { 1.5 } else { 0.01 / ((i as f64 - j as f64).abs() + 1.0) };
            }
        }
        let budget = usize::MAX / 4; // force the in-core fast path

        let dress_with = |threads: usize| -> Array3<f64> {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
            pool.install(|| {
                let mut raw = ThreeIndexSource::build(op, &obs, &aux, budget).unwrap();
                let d = ThreeIndexSource::build_dressed(&mut raw, &m, budget).unwrap();
                match d.backend {
                    Backend::InCore(arr) => arr,
                    _ => panic!("expected in-core dressed source"),
                }
            })
        };

        let a = dress_with(1);
        for threads in [2usize, 3, 5, 8] {
            let b = dress_with(threads);
            assert_eq!(a.dim(), b.dim());
            // BIT-identical, not approximately equal — that is the invariant.
            let differing = a
                .iter()
                .zip(b.iter())
                .filter(|(x, y)| x.to_bits() != y.to_bits())
                .count();
            assert_eq!(
                differing, 0,
                "dressed tensor differs at {differing} elements between 1 and {threads} threads"
            );
        }

        // The invariant above holds because DRESS_ROW_BLOCK is EVEN (see its
        // doc: odd row splits perturb OpenBLAS's vectorized accumulation by
        // ~7e-15). Guard the property directly — a thread-derived block count
        // would land on odd values and reintroduce the drift, and the loop above
        // cannot catch that on its own because the thread counts it happens to
        // exercise may all yield even blocks.
        assert_eq!(
            DRESS_ROW_BLOCK % 2,
            0,
            "DRESS_ROW_BLOCK must be EVEN: an odd output-row split changes \
             OpenBLAS's accumulation order and makes the dressed tensor (and the \
             SCF energy) machine-dependent"
        );
    }

    /// The spilled dressed tensor must stay within a few ulp of the in-core one,
    /// and the deviation must NOT grow as the budget shrinks.
    ///
    /// # What this covers that the sibling test does not
    ///
    /// `dressed_tensor_bitidentical_across_thread_counts` sets
    /// `budget = usize::MAX / 4` with the comment "force the in-core fast path",
    /// so the whole spill path — including the band width
    /// [`spill_block_naux_for`] derives straight from the budget — is outside
    /// its coverage.
    ///
    /// # Why the bar here is ulp-bounded and not bit-identity
    ///
    /// Bit-identity is the right bar ACROSS THREAD COUNTS (the sibling test)
    /// because the thread count must not change the arithmetic at all. It is
    /// the wrong bar across BUDGETS on this path: a spilled tensor is
    /// necessarily assembled in different-sized pieces than an in-core one, so
    /// the k-axis summation is grouped differently, and [`DRESS_K_BLOCK`]'s doc
    /// documents that k-blocking is deliberately an accuracy knob here (its
    /// measured table shows blocked summation is 1.19-3.9x MORE accurate than
    /// full-k, and moves the SCF energy ~5e-9 Ha TOWARD a Kahan reference).
    ///
    /// Measured (benzene/cc-pVDZ, cc-pVDZ-RI): the spilled tensor differs from
    /// in-core at ~4.26M of 5.46M elements, max abs 1.24e-14 against a largest
    /// element of 9.96 — **1.25e-15 relative, 5.6 ulp**. That is ordinary
    /// reassociation.
    ///
    /// # The property that actually matters, and is asserted
    ///
    /// The deviation must be BOUNDED and must not DRIFT with the budget. A
    /// budget-derived band that degraded the tensor progressively — or that
    /// tripped a genuine construction bug at some width — would show up as a
    /// growing or width-correlated error, and that is what this pins.
    ///
    /// # The mechanism, RESOLVED 2026-09-10
    ///
    /// The deviation is introduced entirely by the DRESSING, and its cause is
    /// that [`DRESS_K_BLOCK`] is defeated by the source's block boundaries.
    ///
    /// The k-axis sweep is `for (q0, q1) in q_edges { split at DRESS_K_BLOCK }`,
    /// and `q_edges` comes from `block_edges()`, which steps by
    /// `self.block_naux` — the whole band for an in-core source, but
    /// `spill_block_naux_for(budget/2, nao)` for a spilled one. The fixed
    /// 128-wide split is then CLIPPED at every block boundary, so the number of
    /// partial sums is budget-derived. Measured at benzene/cc-pVDZ (naux = 420):
    ///
    /// ```text
    ///   in-core  band = 420  ->   4 partial sums (128/128/128/36)
    ///   spilled  band =  52  ->   9 partial sums
    ///   spilled  band =  21  ->  20 partial sums
    ///   spilled  band =  10  ->  42 partial sums
    /// ```
    ///
    /// `DRESS_K_BLOCK`'s doc says it "pins the summation order"; on a spilled
    /// source it does not, because the outer edges move underneath it.
    ///
    /// The RAW (undressed) spill path is BIT-IDENTICAL to in-core at every band
    /// width — 0 of 5,458,320 elements differ, verified by
    /// `probe_raw_spilled_tensor_vs_in_core`. So there is no second spill-path
    /// difference upstream; the dressing is the whole of it.
    ///
    /// # Why this is not fixed here
    ///
    /// Making the sum budget-independent needs the k-blocks to span source
    /// blocks, i.e. buffering `DRESS_K_BLOCK` raw Q-rows before each GEMM. That
    /// buffer is `128 * nao^2 * 8` — 13 MB at benzene/cc-pVDZ, 502 MB at
    /// danuglipron/def2-SVP, 2.6 GB at def2-TZVP — held ON TOP of the spill
    /// band, on the path that exists precisely because memory is short. At the
    /// benzene/2 MB budget that buffer is 6.6x the entire budget. Trading the
    /// memory the spill path is there to save, to remove a few ulp, is the
    /// wrong trade.
    ///
    /// A partial measure (sub-blocking each source block at `DRESS_K_BLOCK`)
    /// was implemented and MEASURED INERT: whenever the spill band is narrower
    /// than 128 — which is the entire interesting regime — it splits nothing
    /// and the deviation is unchanged at 5.62 ulp. It was reverted rather than
    /// left in as a fix-shaped no-op.
    ///
    /// So the reassociation is inherent to streaming here, and the honest
    /// engineering answer is the bound this test asserts.
    ///
    /// # The SCF-level number: RESOLVED — it is benzene/PBE, not the spill
    ///
    /// Low-memory validation showed a 1.6e-5 Ha spread in the benzene/cc-pVDZ
    /// **PBE** energy across spilled budgets (identical at every in-core budget
    /// from 0.2 to 8 GiB) — ~1e9x the tensor deviation bounded here. Three
    /// explanations were proposed and tested; the first three all failed, which
    /// is worth recording so nobody re-runs them:
    ///
    /// 1. *Convergence-tolerance artifact.* REFUTED: at `energy_conv = 1e-8`
    ///    the gap is 4.5e-6 Ha and does not move between `density_conv` 1e-6
    ///    and 1e-7, while iteration counts diverge (60 in-core vs 35 spilled).
    /// 2. *A second spill difference upstream of the dressing.* REFUTED: the
    ///    RAW spilled tensor is BIT-IDENTICAL to in-core at every band width
    ///    (0 of 5,458,320 elements differ — `probe_raw_spilled_tensor_vs_in_core`).
    ///    The dressing is the whole of the tensor deviation.
    /// 3. *DIIS amplifying the ulp difference.* REFUTED: shrinking `diis_size`
    ///    8 -> 2 leaves the gap the same order (6.5e-6 -> 2.3e-6), and the
    ///    IN-CORE energy alone moves 3.2e-6 Ha across that change.
    ///
    ///    Do NOT extend that sweep to `diis_size = 1` and read it as more of
    ///    the same: it converges to -222.04 Ha, ~10 Ha from the -231.95 every
    ///    other setting reaches — a DIFFERENT electronic state. Its
    ///    in-core/spilled gap (1.4e-5 Ha) is therefore not comparable and says
    ///    nothing here. Measured, and recorded so the row is not mistaken for
    ///    supporting data.
    ///
    /// The discriminating sweep — same molecule, same basis, same spilled
    /// tensor, varying only the method:
    ///
    /// ```text
    ///   benzene/cc-pVDZ, in-core (0.2 GB) vs spilled (0.002 GB)
    ///     RHF    -230.7261263707  vs  -230.7261263690    gap 1.7e-09 Ha
    ///     BLYP   -232.1531991510  vs  -232.1531991550    gap 4.0e-09 Ha
    ///     PBE    -231.9508313583  vs  -231.9508248546    gap 6.5e-06 Ha
    /// ```
    ///
    /// **BLYP is a GGA on the same Becke-Lebedev grid and is 1600x tighter than
    /// PBE.** So this is not "DFT grid amplification" (an earlier draft of this
    /// note said that, on the RHF row alone, before the BLYP row landed — it was
    /// wrong). RHF and BLYP both show that the 5.6-ulp tensor deviation is worth
    /// ~2-4e-9 Ha, i.e. negligible, exactly as this test's bound implies.
    ///
    /// What is left is a benzene/**PBE**-specific SCF sensitivity: that
    /// particular surface is ill-conditioned enough that two trajectories from
    /// ulp-different starting tensors converge ~6e-6 Ha apart. The `diis_size`
    /// evidence corroborates it — the in-core PBE energy alone moves 3.2e-6 Ha
    /// on a pure solver-setting change, while RHF and BLYP do not.
    ///
    /// Practical reading: the spill path is sound. A method whose own SCF is
    /// well-conditioned reproduces to ~1e-9 Ha across the in-core/spilled
    /// boundary. If a specific system/functional shows more, suspect that SCF,
    /// not the tensor — and check it in-core by perturbing `diis_size` first.
    ///
    #[test]
    fn spilled_dressed_tensor_stays_within_a_few_ulp_of_in_core() {
        let mol = Molecule::load_xyz("../../testdata/molecules/benzene.xyz").unwrap();
        let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
        let aux = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
        let op = Operator::coulomb();
        let naux = aux.nbasis();
        let nao = obs.nbasis();
        let mut m = Array2::<f64>::zeros((naux, naux));
        for i in 0..naux {
            for j in 0..naux {
                m[(i, j)] = if i == j { 1.5 } else { 0.01 / ((i as f64 - j as f64).abs() + 1.0) };
            }
        }

        let dress_at = |budget: usize| -> (Array3<f64>, bool) {
            let mut raw = ThreeIndexSource::build(op, &obs, &aux, budget).unwrap();
            let mut d = ThreeIndexSource::build_dressed(&mut raw, &m, budget).unwrap();
            let spilled = matches!(d.backend, Backend::DiskSpill { .. });
            let mut out = Array3::<f64>::zeros((naux, nao, nao));
            d.for_each_block(&mut |blk: AuxBlock| {
                let n = blk.data.shape()[0];
                out.slice_mut(ndarray::s![blk.p0..blk.p0 + n, .., ..]).assign(&blk.data);
                Ok(())
            })
            .unwrap();
            (out, spilled)
        };

        let row_bytes = nao * nao * 8;
        let (reference, ref_spilled) = dress_at(usize::MAX / 4);
        assert!(!ref_spilled, "the reference must be the in-core backend");
        let max_elem = reference.iter().fold(0.0f64, |a, &x| a.max(x.abs()));
        assert!(max_elem > 1.0, "sanity: the dressed tensor should not be ~zero");

        // 64 ulp of the largest element: an order of magnitude above the 5.6 ulp
        // measured, so ordinary BLAS variation across machines passes, while a
        // real construction defect (which shows up at 1e-3 relative or worse,
        // not 1e-14) fails loudly.
        let tol = 64.0 * f64::EPSILON * max_elem;

        let mut seen: Vec<(usize, f64)> = Vec::new();
        for band in [10usize, 11, 12, 20, 21, 51, 52] {
            let budget = (band * row_bytes + row_bytes / 2) * 2;
            let (got, spilled) = dress_at(budget);
            if !spilled {
                continue;
            }
            let max_abs = reference
                .iter()
                .zip(got.iter())
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f64, f64::max);
            let differing =
                reference.iter().zip(got.iter()).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
            eprintln!(
                "DRESSED band={band} differing={differing}/{} max_abs={max_abs:.3e} ({:.2} ulp)",
                reference.len(),
                if max_elem > 0.0 { max_abs / (f64::EPSILON * max_elem) } else { 0.0 },
            );
            assert!(
                max_abs <= tol,
                "spilled dressed tensor (band={band}) deviates {max_abs:.3e} from in-core \
                 ({:.1} ulp of the largest element {max_elem:.3e}), above the {:.1}-ulp bar. \
                 A few ulp is expected reassociation; this much is a construction defect.",
                max_abs / (f64::EPSILON * max_elem),
                tol / (f64::EPSILON * max_elem)
            );
            seen.push((band, max_abs));
        }

        // Reachability: if nothing spilled, this asserted nothing.
        assert!(
            seen.len() >= 3,
            "only {} budgets forced a spill, too few to see a trend — re-derive them from \
             spill_block_naux_for",
            seen.len()
        );

        // The deviation must not DRIFT with the band width. A narrower band means
        // more partial sums, so a progressive degradation would show as a
        // monotone rise; a construction bug keyed to some width would show as an
        // outlier. Require the spread across widths to stay inside the same bar.
        let worst = seen.iter().map(|&(_, e)| e).fold(0.0f64, f64::max);
        let best = seen.iter().map(|&(_, e)| e).fold(f64::INFINITY, f64::min);
        assert!(
            worst <= tol,
            "worst spilled deviation {worst:.3e} exceeds the {tol:.3e} bar: {seen:?}"
        );
        assert!(
            worst / best.max(f64::MIN_POSITIVE) < 1e3,
            "the deviation varies by {:.1e}x across band widths ({seen:?}), which suggests a \
             width-dependent construction error rather than uniform reassociation",
            worst / best.max(f64::MIN_POSITIVE)
        );
    }

    /// DIAGNOSTIC: is the RAW (undressed) spilled tensor bit-identical to in-core?
    ///
    /// Discriminates H2 for the unresolved spill/SCF question recorded on
    /// `spilled_dressed_tensor_stays_within_a_few_ulp_of_in_core`: if the RAW
    /// spill path already differs, the 5.6-ulp dressed deviation is downstream
    /// of a more basic difference and the search moves there. If the raw path
    /// is bit-identical, the deviation is introduced by the DRESSING, which is
    /// where the k-blocking lives.
    #[test]
    #[ignore = "diagnostic: prints, no assertions; run with --ignored --nocapture"]
    fn probe_raw_spilled_tensor_vs_in_core() {
        let mol = Molecule::load_xyz("../../testdata/molecules/benzene.xyz").unwrap();
        let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
        let aux = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
        let op = Operator::coulomb();
        let naux = aux.nbasis();
        let nao = obs.nbasis();

        let raw_at = |budget: usize| -> (Array3<f64>, bool) {
            let mut r = ThreeIndexSource::build(op, &obs, &aux, budget).unwrap();
            let spilled = matches!(r.backend, Backend::DiskSpill { .. });
            let mut out = Array3::<f64>::zeros((naux, nao, nao));
            r.for_each_block(&mut |blk: AuxBlock| {
                let n = blk.data.shape()[0];
                out.slice_mut(ndarray::s![blk.p0..blk.p0 + n, .., ..]).assign(&blk.data);
                Ok(())
            })
            .unwrap();
            (out, spilled)
        };

        let row_bytes = nao * nao * 8;
        let (reference, ref_spilled) = raw_at(usize::MAX / 4);
        assert!(!ref_spilled, "reference must be in-core");
        let max_elem = reference.iter().fold(0.0f64, |a, &x| a.max(x.abs()));

        for band in [10usize, 21, 52] {
            let budget = (band * row_bytes + row_bytes / 2) * 2;
            let (got, spilled) = raw_at(budget);
            let differing =
                reference.iter().zip(got.iter()).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
            let max_abs = reference
                .iter()
                .zip(got.iter())
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f64, f64::max);
            eprintln!(
                "RAW band={band} spilled={spilled} differing={differing}/{} max_abs={max_abs:.3e} \
                 ({:.2} ulp of max|elem|={max_elem:.3e})",
                reference.len(),
                if max_elem > 0.0 { max_abs / (f64::EPSILON * max_elem) } else { 0.0 },
            );
        }
    }

    /// REGRESSION (defect A): removing the second copy must not move a bit.
    ///
    /// `build_dressed_band`'s parallel fast path used to `collect()` every
    /// output block into a `Vec<(usize, Array2)>` and then scatter, so at the
    /// instant the collect finished a FULL SECOND COPY of the dressed band was
    /// co-resident with `out_incore` — while the `in_core` gate above charged
    /// for only ONE. It now accumulates straight into the destination rows via
    /// `par_chunks_mut`.
    ///
    /// `dressed_tensor_bitidentical_across_thread_counts` proves the new path
    /// agrees with ITSELF at every thread count; it cannot prove it agrees with
    /// the code that was replaced. This does: the reference below reproduces
    /// the OLD structure literally (a private zeroed `Array2` per block,
    /// accumulated, then assigned into the output), so any change to the GEMM
    /// boundaries, the Q sweep, or the accumulation order shows up as a
    /// differing bit.
    #[test]
    fn dressed_fast_path_bitidentical_to_owned_buffer_reference() {
        let (mol,) = water();
        let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
        let dfbs =
            PreparedBasis::new(&mol, &basis::bundled("def2-universal-jkfit").unwrap()).unwrap();
        let op = Operator::coulomb();
        let naux = dfbs.nbasis();
        let nao = obs.nbasis();
        let mut m = Array2::<f64>::zeros((naux, naux));
        for p in 0..naux {
            for q in 0..naux {
                m[(p, q)] = 0.001 * (((p * 7 + q * 3) % 13) as f64) + if p == q { 1.0 } else { 0.0 };
            }
        }
        let budget = usize::MAX / 4; // force the in-core fast path

        // Live path.
        let mut raw = ThreeIndexSource::build(op, &obs, &dfbs, budget).unwrap();
        let dressed = ThreeIndexSource::build_dressed(&mut raw, &m, budget).unwrap();
        let got = match dressed.backend {
            Backend::InCore(ref a) => a.clone(),
            _ => panic!("expected the in-core fast path"),
        };

        // Reference: the OLD collect-into-owned-buffer-then-scatter structure,
        // written out by hand and run SERIALLY.
        let raw_ref = ThreeIndexSource::build(op, &obs, &dfbs, budget).unwrap();
        let q_edges = raw_ref.block_edges();
        let raw_flat = raw_ref.incore_flat().expect("in-core");
        let raw_p0 = raw_ref.band().0;
        let mut expect = Array3::<f64>::zeros((naux, nao, nao));
        let mut l = 0usize;
        while l < naux {
            let r = (l + DRESS_ROW_BLOCK).min(naux);
            let b = r - l;
            // A PRIVATE owned buffer, exactly as the removed code allocated.
            let mut acc = Array2::<f64>::zeros((b, nao * nao));
            for &(q0, q1) in q_edges.iter() {
                let mut qa = q0;
                while qa < q1 {
                    let qb = (qa + DRESS_K_BLOCK).min(q1);
                    let msub = m.slice(ndarray::s![l..l + b, qa..qb]);
                    let rblk = raw_flat.slice(ndarray::s![qa - raw_p0..qb - raw_p0, ..]);
                    let contrib: Array2<f64> = msub.dot(&rblk);
                    acc += &contrib;
                    qa = qb;
                }
            }
            // ...then scattered into the output.
            let mut dst = expect
                .slice_mut(ndarray::s![l..l + b, .., ..])
                .into_shape_with_order((b, nao * nao))
                .unwrap();
            dst.assign(&acc);
            l = r;
        }

        assert_eq!(got.dim(), expect.dim());
        let differing = got
            .iter()
            .zip(expect.iter())
            .filter(|(x, y)| x.to_bits() != y.to_bits())
            .count();
        assert_eq!(
            differing, 0,
            "removing the redundant second copy changed {differing} elements — the \
             scatter-free path must be a pure memory fix, never a numerical one"
        );
    }

    /// Defect A, the accounting half: the in-core dressing gate must refuse a
    /// band that does not fit AND name the term in the breakdown.
    ///
    /// The old gate compared only `band·nao²·8` against the budget, so it was
    /// impossible for it to fail here for the right reason. Both directions are
    /// checked: a budget that genuinely cannot hold the band is refused, and an
    /// ample budget still runs (an over-estimating guard that refuses a job
    /// which would have fit is as much a defect as an under-estimate).
    #[test]
    fn dressed_over_budget_is_refused_and_named_ample_budget_still_runs() {
        let (mol,) = water();
        let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
        let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
        let op = Operator::coulomb();
        let naux = dfbs.nbasis();
        let nao = obs.nbasis();
        let mut m = Array2::<f64>::zeros((naux, naux));
        for p in 0..naux {
            m[(p, p)] = 1.0;
        }

        // AMPLE: the band plus every worker's contrib comfortably fits.
        let ample = usize::MAX / 4;
        let mut raw = ThreeIndexSource::build(op, &obs, &dfbs, ample).unwrap();
        assert!(
            ThreeIndexSource::build_dressed(&mut raw, &m, ample).is_ok(),
            "an ample budget must still run — over-counting refuses jobs that fit"
        );

        // TIGHT-BUT-FAST-PATH: a budget that admits the band itself (so the
        // in-core fast path is entered) but cannot also hold the per-worker
        // GEMM scratch the old gate ignored entirely.
        let band_bytes = naux * nao * nao * 8;
        let mut raw2 = ThreeIndexSource::build(op, &obs, &dfbs, ample).unwrap();
        let err = ThreeIndexSource::build_dressed(&mut raw2, &m, band_bytes)
            .expect_err("band + per-worker scratch exceeds a band-sized budget");
        let msg = err.to_string();
        assert!(msg.contains("DF dressing (in-core)"), "must name the stage: {msg}");
        assert!(
            msg.contains("dressing GEMM contrib"),
            "the breakdown must name the previously-uncharged per-worker term: {msg}"
        );
    }

    // ------------------------------------------------------------------
    // Disk-spill preflight anchors.
    //
    // The defect these pin: `build_band` used to choose in-core vs spill with a
    // bare `if needed <= budget_bytes`, with NO disk-space check and NO warning.
    // A C48/def2-TZVP job writes ~185 GB and a C32/def2-QZVP job ~415 GB into
    // `std::env::temp_dir()`; nothing looked at the free space, so the run
    // filled the partition instead of refusing. `cosx_k::check_budget` refuses
    // with a diagnosis in the same situation — the asymmetry WAS the bug.
    // ------------------------------------------------------------------

    /// (a) REFUSAL FIRES. The spill preflight must return a typed error naming
    /// the tensor size, the free space AND the path when the write cannot fit.
    ///
    /// Free space is INJECTED (the `free` argument), never produced by actually
    /// filling a disk: the check is a pure function of (needed, free, dir) so it
    /// can be exercised deterministically. `check_spill_disk` is the seam the
    /// live path calls with a real `statvfs` result.
    ///
    /// MUTATION: delete the `needed + SPILL_HEADROOM_BYTES > free` branch from
    /// `check_spill_disk` and this goes RED (`expect_err` panics).
    #[test]
    fn spill_refused_when_free_space_cannot_hold_the_tensor() {
        let dir = std::path::Path::new("/tmp/ferric-spill-anchor");
        // 100 GB wanted, 1 GB free.
        let needed = 100_000_000_000u64;
        let free = 1_000_000_000u64;
        let err = check_spill_disk(needed, free, dir)
            .expect_err("a 100 GB spill into 1 GB of free space must be refused");
        let msg = err.to_string();
        // The three facts a user needs without reading source.
        assert!(msg.contains("100.00 GB"), "must name the tensor size: {msg}");
        assert!(msg.contains("1.00 GB"), "must name the free space: {msg}");
        assert!(
            msg.contains("/tmp/ferric-spill-anchor"),
            "must name the spill directory: {msg}"
        );
        // The actionable remedies.
        assert!(msg.contains("budget_gb"), "must name the budget knob: {msg}");
        assert!(msg.contains("TMPDIR"), "must name the spill-dir override: {msg}");
    }

    /// (a′) The refusal must NOT fire when the write genuinely fits — an
    /// over-refusing guard rejects jobs that would have run, which is as much a
    /// defect as no guard at all (cf.
    /// `dressed_over_budget_is_refused_and_named_ample_budget_still_runs`).
    #[test]
    fn spill_allowed_when_free_space_is_ample() {
        let dir = std::path::Path::new("/tmp");
        assert!(
            check_spill_disk(1_000_000_000, 500_000_000_000, dir).is_ok(),
            "1 GB into 500 GB free must be allowed"
        );
    }

    /// (4) THE HEADROOM MARGIN. A write that fits arithmetically but would leave
    /// the partition essentially full must still be refused.
    ///
    /// MUTATION: set `SPILL_HEADROOM_BYTES` to 0 and `SPILL_HEADROOM_FRACTION`
    /// to 0.0 and this goes RED — `needed == free` then passes.
    #[test]
    fn spill_refused_when_it_would_leave_no_headroom() {
        let dir = std::path::Path::new("/tmp");
        // Exactly fills the partition: arithmetically OK, operationally fatal.
        assert!(
            check_spill_disk(100_000_000_000, 100_000_000_000, dir).is_err(),
            "a write that consumes 100% of free space must be refused"
        );
        // Just inside the absolute floor (2 GB) — still refused.
        assert!(
            check_spill_disk(99_000_000_000, 100_000_000_000, dir).is_err(),
            "leaving 1 GB free is below the absolute headroom floor"
        );
        // The percentage arm bites on a large partition where 2 GB would not:
        // 10 TB free, want 9.95 TB → leaves 50 GB, which is under 2% of free.
        assert!(
            check_spill_disk(9_950_000_000_000, 10_000_000_000_000, dir).is_err(),
            "leaving <2% of a large partition must be refused"
        );
        // And the same partition with a comfortable remainder is fine.
        assert!(
            check_spill_disk(5_000_000_000_000, 10_000_000_000_000, dir).is_ok(),
            "leaving 50% of the partition must be allowed"
        );
    }

    /// (d) THE WARNING FIRES ONCE, NOT PER BLOCK.
    ///
    /// Entering the spill path is a performance cliff (`DfK::build` re-reads the
    /// whole tensor every SCF iteration), so it must be announced — but the spill
    /// loop runs once per aux BLOCK, and a per-block warning would emit thousands
    /// of lines. `spill_warning_text` is pure (returns the line) and
    /// `SPILL_WARNED` is the once-latch; this asserts the latch admits exactly
    /// one caller and that the line names the re-read cost.
    ///
    /// MUTATION: remove the `SPILL_WARNED.swap` guard from `warn_spill_once` (or
    /// make it always return true) and this goes RED.
    #[test]
    fn spill_warning_is_emitted_once_and_names_the_per_iteration_reread() {
        let dir = std::path::Path::new("/tmp/ferric-spill-anchor");
        let text = spill_warning_text(55_000_000_000, dir);
        assert!(text.starts_with("[ferric] warning:"), "must match ferric's warning convention: {text}");
        assert!(text.contains("55.00 GB"), "must name the tensor size: {text}");
        assert!(text.contains("/tmp/ferric-spill-anchor"), "must name the spill dir: {text}");
        assert!(
            text.contains("every SCF iteration"),
            "the point of the warning is the per-iteration re-read cost: {text}"
        );

        // The once-latch: a fresh latch admits exactly one caller.
        let latch = std::sync::atomic::AtomicBool::new(false);
        let fired: usize = (0..1000)
            .filter(|_| !latch.swap(true, std::sync::atomic::Ordering::Relaxed))
            .count();
        assert_eq!(fired, 1, "the spill warning must fire once, not per block");
        // That the LIVE latch is actually consulted is a separate, behavioural
        // claim — see `spill_warning_latch_is_consulted`.
    }

    /// (d′) BEHAVIOURAL: the live `warn_spill_once` must consult the latch. Runs
    /// the real function twice against a locally-reset latch and asserts the
    /// second call reports "already warned".
    ///
    /// MUTATION: make `warn_spill_once` unconditional and this goes RED.
    #[test]
    fn spill_warning_latch_is_consulted() {
        // Serialize against any other test that might trip the latch.
        static GUARD: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let _g = GUARD.lock().unwrap_or_else(|e| e.into_inner());
        SPILL_WARNED.store(false, std::sync::atomic::Ordering::Relaxed);
        assert!(warn_spill_once(1_000, std::path::Path::new("/tmp")), "first call must warn");
        assert!(
            !warn_spill_once(1_000, std::path::Path::new("/tmp")),
            "second call must be suppressed by the once-latch"
        );
        SPILL_WARNED.store(false, std::sync::atomic::Ordering::Relaxed);
    }

    /// (b) IN-CORE IS UNTOUCHED. When the tensor fits the budget the new code
    /// must not run, warn, or perturb a single bit.
    ///
    /// The bit-identity half is already carried by
    /// `in_core_block_equals_dense_eri3`; what this adds is that the in-core
    /// decision NEVER consults the disk. A stat call on the in-core path would be
    /// a behaviour change (and would make an unwritable TMPDIR fail an in-core
    /// job), so the guard is asserted to live strictly inside the spill branch:
    /// the source builds fine with TMPDIR pointed at a nonexistent path.
    #[test]
    fn in_core_path_never_touches_the_spill_directory() {
        let (mol,) = water();
        let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
        let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
        let op = Operator::coulomb();
        let dense = crate::threeindex::eri3_tensor(op, &obs, &dfbs).unwrap();

        // In-core (huge budget): must succeed and be bit-identical to dense.
        let mut src = ThreeIndexSource::build(op, &obs, &dfbs, usize::MAX).unwrap();
        assert!(src.is_incore(), "usize::MAX budget must stay in core");
        let mut reassembled = ndarray::Array3::<f64>::zeros(dense.dim());
        src.for_each_block(|blk| {
            reassembled
                .slice_mut(ndarray::s![blk.p0..blk.p0 + blk.data.shape()[0], .., ..])
                .assign(&blk.data);
            Ok(())
        })
        .unwrap();
        let n_diff = reassembled
            .iter()
            .zip(dense.iter())
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert_eq!(n_diff, 0, "the disk guard must not perturb the in-core tensor");
    }

    /// (c) SPILL STILL WORKS WHEN IT LEGITIMATELY FITS.
    ///
    /// A tiny budget forces the spill path on a tensor of a few MB; the real
    /// `/tmp` has room, so the preflight must pass and the streamed tensor must
    /// still equal the dense build bit-for-bit. (This duplicates
    /// `spill_blocks_equal_dense_eri3`'s content check on purpose: that test is
    /// the pre-existing green one this change must not break, and this one
    /// states the intent — "the guard admits a legitimate small spill" —
    /// explicitly.)
    #[test]
    fn small_legitimate_spill_passes_the_disk_guard() {
        let (mol,) = water();
        let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
        let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
        let op = Operator::coulomb();
        let dense = crate::threeindex::eri3_tensor(op, &obs, &dfbs).unwrap();
        let (naux, nao, _) = dense.dim();
        let tiny = nao * nao * 8 * 3;
        let mut src = ThreeIndexSource::build(op, &obs, &dfbs, tiny)
            .expect("a few-MB spill must pass the disk preflight on a normal /tmp");
        assert!(!src.is_incore(), "tiny budget must have spilled");
        let mut reassembled = ndarray::Array3::<f64>::zeros((naux, nao, nao));
        src.for_each_block(|blk| {
            let b = blk.data.shape()[0];
            reassembled
                .slice_mut(ndarray::s![blk.p0..blk.p0 + b, .., ..])
                .assign(&blk.data);
            Ok(())
        })
        .unwrap();
        let n_diff = reassembled
            .iter()
            .zip(dense.iter())
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert_eq!(n_diff, 0, "a legitimate spill must still be bit-exact");
    }

    /// The free-space probe must return a plausible answer for a real directory
    /// and `None` (never a panic, never a bogus 0 that would refuse every spill)
    /// for a path that does not exist.
    #[test]
    fn free_space_probe_reports_real_paths_and_declines_missing_ones() {
        let free = free_bytes_at(std::path::Path::new("/tmp"))
            .expect("/tmp must report free space");
        assert!(free > 0, "/tmp reported 0 bytes free");
        assert_eq!(
            free_bytes_at(std::path::Path::new("/nonexistent-ferric-anchor-path")),
            None,
            "a missing path must decline (None), not fabricate a free-space figure"
        );
    }

    #[test]
    fn block_naux_respects_budget() {
        // nao=10 → one aux row is 10*10*8 = 800 bytes.
        // budget 4000 bytes → block_naux = 4000/800 = 5.
        assert_eq!(block_naux_for(4000, 10), 5);
        // budget smaller than one row → at least 1.
        assert_eq!(block_naux_for(500, 10), 1);
    }

    #[test]
    fn spill_block_sizing_counts_both_live_blocks() {
        // Double-buffered spill holds two blocks at once (one computing, one
        // writing): the pair must fit the budget. nao=10 → row = 800 bytes.
        // budget 4000 → 2 × (2 rows × 800) = 3200 ≤ 4000. block_naux_for
        // would have said 5 rows (4000 bytes), whose pair would bust the budget.
        assert_eq!(spill_block_naux_for(4000, 10), 2);
        // Degenerate floor: even a budget below one row yields 1 (a single aux
        // row must always be representable) — pre-existing behavior.
        assert_eq!(spill_block_naux_for(500, 10), 1);
    }

    #[test]
    fn spill_single_row_blocks_equal_dense_eri3() {
        // Degenerate pipeline: budget below one aux row forces block_naux = 1,
        // maximizing producer/consumer handoffs (one rendezvous per aux row).
        // Content must still be bit-identical to the dense build.
        let (mol,) = water();
        let obs = PreparedBasis::new(&mol, &basis::bundled("sto-3g").unwrap()).unwrap();
        let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
        let op = Operator::coulomb();
        let dense = crate::threeindex::eri3_tensor(op, &obs, &dfbs).unwrap();
        let (naux, nao, _) = dense.dim();
        let mut src = ThreeIndexSource::build(op, &obs, &dfbs, 1).unwrap();
        assert_eq!(src.n_blocks(), naux, "budget=1 byte should force 1-row blocks");
        let mut reassembled = ndarray::Array3::<f64>::zeros((naux, nao, nao));
        src.for_each_block(|blk| {
            let b = blk.data.shape()[0];
            reassembled.slice_mut(ndarray::s![blk.p0..blk.p0 + b, .., ..]).assign(&blk.data);
            Ok(())
        }).unwrap();
        let n_diff = reassembled.iter().zip(dense.iter())
            .filter(|(a, b)| a.to_bits() != b.to_bits()).count();
        assert_eq!(n_diff, 0, "1-row spill blocks differ bitwise from dense eri3");
    }

    #[test]
    fn spill_blocks_equal_dense_eri3() {
        let (mol,) = water();
        let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
        let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
        let op = Operator::coulomb();
        let dense = crate::threeindex::eri3_tensor(op, &obs, &dfbs).unwrap();
        let (naux, nao, _) = dense.dim();
        // Tiny budget → force spill into several blocks.
        let tiny = nao * nao * 8 * 3; // ~3 aux rows per block
        let mut src = ThreeIndexSource::build(op, &obs, &dfbs, tiny).unwrap();
        assert!(src.n_blocks() > 1, "expected spill into >1 block, got {}", src.n_blocks());
        let mut reassembled = ndarray::Array3::<f64>::zeros((naux, nao, nao));
        src.for_each_block(|blk| {
            let b = blk.data.shape()[0];
            reassembled.slice_mut(ndarray::s![blk.p0..blk.p0 + b, .., ..]).assign(&blk.data);
            Ok(())
        }).unwrap();
        let maxdiff = (&reassembled - &dense).iter().map(|v| v.abs()).fold(0.0, f64::max);
        assert!(maxdiff == 0.0, "spill blocks != dense eri3, maxdiff={maxdiff}");
    }

    #[test]
    fn bands_reassemble_to_full_raw_tensor() {
        // The MPI aux-band striping invariant: building disjoint bands
        // [0,k), [k,naux) and concatenating them (each block reports its GLOBAL
        // p0) must reproduce the full dense (P|μν) tensor bit-for-bit. This is
        // what makes summing per-rank partials equal the serial result.
        let (mol,) = water();
        let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
        let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
        let op = Operator::coulomb();
        let dense = crate::threeindex::eri3_tensor(op, &obs, &dfbs).unwrap();
        let (naux, nao, _) = dense.dim();
        let k = naux / 2;

        let mut reassembled = ndarray::Array3::<f64>::zeros((naux, nao, nao));
        for &(p0, p1) in &[(0usize, k), (k, naux)] {
            let mut src = ThreeIndexSource::build_band(op, &obs, &dfbs, usize::MAX, p0, p1).unwrap();
            assert_eq!(src.band(), (p0, p1));
            assert_eq!(src.band_naux(), p1 - p0);
            // naux stays GLOBAL even for a band.
            assert_eq!(src.naux(), naux);
            src.for_each_block(|blk| {
                let b = blk.data.shape()[0];
                // blk.p0 is the GLOBAL aux index.
                assert!(blk.p0 >= p0 && blk.p0 + b <= p1);
                reassembled
                    .slice_mut(ndarray::s![blk.p0..blk.p0 + b, .., ..])
                    .assign(&blk.data);
                Ok(())
            })
            .unwrap();
        }
        let n_diff = reassembled
            .iter()
            .zip(dense.iter())
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert_eq!(n_diff, 0, "reassembled bands differ bitwise from dense eri3");
    }

    #[test]
    fn dressed_bands_reassemble_to_full_dressed_tensor() {
        // Same invariant for the DRESSED (V^{-1/2}-mixed) source used by DF-K:
        // each rank dresses only its band (summing over ALL Q of the full raw
        // source), and concatenating the bands reproduces the full dressed
        // tensor bit-for-bit.
        let (mol,) = water();
        let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
        let dfbs =
            PreparedBasis::new(&mol, &basis::bundled("def2-universal-jkfit").unwrap()).unwrap();
        let op = Operator::coulomb();
        let naux = dfbs.nbasis();
        let nao = obs.nbasis();

        // A deterministic non-trivial (naux, naux) mixing matrix (stand-in for
        // V^{-1/2}); the invariant is purely algebraic so any m works.
        let mut m = Array2::<f64>::zeros((naux, naux));
        for p in 0..naux {
            for q in 0..naux {
                m[(p, q)] = 0.001 * (((p * 7 + q * 3) % 13) as f64) + if p == q { 1.0 } else { 0.0 };
            }
        }

        // Full dressed tensor (reference).
        let mut raw_full = ThreeIndexSource::build(op, &obs, &dfbs, usize::MAX).unwrap();
        let mut full = ThreeIndexSource::build_dressed(&mut raw_full, &m, usize::MAX).unwrap();
        let mut full_dense = ndarray::Array3::<f64>::zeros((naux, nao, nao));
        full.for_each_block(|blk| {
            let b = blk.data.shape()[0];
            full_dense
                .slice_mut(ndarray::s![blk.p0..blk.p0 + b, .., ..])
                .assign(&blk.data);
            Ok(())
        })
        .unwrap();

        // Banded dressed tensors, concatenated.
        let k = naux / 3;
        let mut reassembled = ndarray::Array3::<f64>::zeros((naux, nao, nao));
        for &(p0, p1) in &[(0usize, k), (k, naux)] {
            let mut raw = ThreeIndexSource::build(op, &obs, &dfbs, usize::MAX).unwrap();
            let mut band =
                ThreeIndexSource::build_dressed_band(&mut raw, &m, usize::MAX, p0, p1).unwrap();
            assert_eq!(band.band(), (p0, p1));
            band.for_each_block(|blk| {
                let b = blk.data.shape()[0];
                reassembled
                    .slice_mut(ndarray::s![blk.p0..blk.p0 + b, .., ..])
                    .assign(&blk.data);
                Ok(())
            })
            .unwrap();
        }
        // The dressing is a GEMM `out[P,:] = Σ_Q m[P,Q] raw[Q,:]`. BLAS may pick
        // a different internal tiling for a band's smaller M dimension than for
        // the full M, so the result is NOT guaranteed bit-for-bit across a
        // band-vs-full split (only the RAW integral band reassembly is bitwise —
        // see `bands_reassemble_to_full_raw_tensor`). What MUST hold is numerical
        // equivalence to machine precision: each output row P is the same linear
        // combination of the same raw rows. The end-to-end MPI correctness bar
        // (2-rank ≡ 1-rank ≤ 1e-12 Ha on real SCF energies) is verified
        // separately in tests/mpi_dfjk_banding.rs.
        let maxdiff = (&reassembled - &full_dense)
            .iter()
            .map(|v| v.abs())
            .fold(0.0, f64::max);
        assert!(
            maxdiff < 1e-12,
            "reassembled dressed bands differ from full dressed tensor: maxdiff={maxdiff}"
        );
    }

    #[test]
    fn banded_spill_reassembles_to_full() {
        // Band + disk-spill together: a band built under a tiny budget must
        // stream-reassemble bit-for-bit into the dense tensor's band. Exercises
        // the global-index reporting on the spill read-back path.
        let (mol,) = water();
        let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
        let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
        let op = Operator::coulomb();
        let dense = crate::threeindex::eri3_tensor(op, &obs, &dfbs).unwrap();
        let (naux, nao, _) = dense.dim();
        let (p0, p1) = (naux / 4, naux); // an off-zero band
        let tiny = nao * nao * 8 * 2; // ~2 aux rows per block → forces spill
        let mut src = ThreeIndexSource::build_band(op, &obs, &dfbs, tiny, p0, p1).unwrap();
        assert!(src.n_blocks() > 1, "expected spill into >1 block");
        let mut reassembled = ndarray::Array3::<f64>::zeros((naux, nao, nao));
        src.for_each_block(|blk| {
            let b = blk.data.shape()[0];
            assert!(blk.p0 >= p0 && blk.p0 + b <= p1);
            reassembled
                .slice_mut(ndarray::s![blk.p0..blk.p0 + b, .., ..])
                .assign(&blk.data);
            Ok(())
        })
        .unwrap();
        let band_diff = (&reassembled.slice(ndarray::s![p0..p1, .., ..])
            - &dense.slice(ndarray::s![p0..p1, .., ..]))
            .iter()
            .map(|v| v.abs())
            .fold(0.0, f64::max);
        assert_eq!(band_diff, 0.0, "spilled band != dense eri3 band");
    }

    #[test]
    fn in_core_block_equals_dense_eri3() {
        let (mol,) = water();
        let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
        let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
        let op = Operator::coulomb();
        let dense = crate::threeindex::eri3_tensor(op, &obs, &dfbs).unwrap();
        // Huge budget → single in-core block, raw (un-dressed).
        let mut src = ThreeIndexSource::build(op, &obs, &dfbs, usize::MAX).unwrap();
        assert_eq!(src.n_blocks(), 1);
        let mut reassembled = ndarray::Array3::<f64>::zeros(dense.dim());
        src.for_each_block(|blk| {
            reassembled.slice_mut(ndarray::s![blk.p0..blk.p0 + blk.data.shape()[0], .., ..])
                .assign(&blk.data);
            Ok(())
        }).unwrap();
        let maxdiff = (&reassembled - &dense).iter().map(|v| v.abs()).fold(0.0, f64::max);
        assert!(maxdiff == 0.0, "in-core raw block != dense eri3, maxdiff={maxdiff}");
    }
}
