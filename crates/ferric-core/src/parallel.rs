//! Global parallelization context (Rayon + MPI).

// mpi 0.7 renamed the concrete world communicator to `SimpleCommunicator`
// (the old `SystemCommunicator` alias was dropped). `SimpleCommunicator::world()`
// yields MPI_COMM_WORLD, and `rank`/`size`/`all_reduce_into` come from the
// `Communicator` / `CommunicatorCollectives` traits.
#[cfg(feature = "mpi")]
use mpi::topology::SimpleCommunicator;
#[cfg(feature = "mpi")]
use mpi::traits::Communicator;

// MPI is initialized exactly once per process. `mpi::initialize()` returns a
// `Universe` RAII guard that finalizes MPI on drop; we must keep it alive for
// the whole process. Crucially, `SimpleCommunicator`/`Universe` embed a raw
// `*mut ompi_communicator_t`, so they are `!Send + !Sync`. `ParallelContext` is
// threaded into rayon closures (which require `Send`/`Sync`), so it MUST NOT
// hold any MPI object. We therefore park the `Universe` in a process-global and
// keep only `rank`/`size`/`mpi_active` (all plain `Copy`) in the context, and
// reconstruct the world communicator on demand via `world()` inside the
// (serial, non-rayon) reduction points. This is what keeps `ParallelContext`
// `Send + Sync` while still driving real MPI collectives.
#[cfg(feature = "mpi")]
static MPI_UNIVERSE: std::sync::OnceLock<Option<MpiUniverseHolder>> = std::sync::OnceLock::new();

// SAFETY: MPI_COMM_WORLD is a process-global handle valid on every thread once
// `MPI_Init(_thread)` has run. rsmpi conservatively marks the universe `!Send`
// because a *user-created* communicator's lifetime is thread-sensitive, but the
// world universe lives for the whole process and we only ever read it. Parking
// it behind this wrapper lets us stash it in a `OnceLock` (which requires the
// contents be `Send + Sync`) without ever moving it across a thread in a way
// that touches MPI state. The held `Universe` is only dropped at process exit.
#[cfg(feature = "mpi")]
struct MpiUniverseHolder(#[allow(dead_code)] mpi::environment::Universe);
#[cfg(feature = "mpi")]
unsafe impl Send for MpiUniverseHolder {}
#[cfg(feature = "mpi")]
unsafe impl Sync for MpiUniverseHolder {}

/// A context representing the parallel execution environment.
///
/// Handles both single-node (Rayon) and multi-node (MPI) parallelization.
/// Deliberately holds NO MPI object (only `rank`/`size`/`mpi_active`) so it is
/// `Send + Sync` and can be borrowed into rayon parallel regions. Obtain the
/// world communicator for a collective via [`ParallelContext::world`].
pub struct ParallelContext {
    /// Whether a real, multi-rank MPI world is active (feature on, initialized,
    /// size > 1). When false, [`world`](Self::world) returns `None` and every
    /// reduction path is a no-op — byte-identical to the serial engine.
    #[cfg(feature = "mpi")]
    mpi_active: bool,
    pub rank: usize,
    pub size: usize,
}

impl ParallelContext {
    /// Create a new context. If the `mpi` feature is on, initializes MPI once
    /// per process and records this rank's rank/size.
    pub fn new() -> Self {
        #[cfg(feature = "mpi")]
        {
            // Initialize MPI exactly once and park the Universe process-globally
            // so it is never finalized until process exit.
            let holder = MPI_UNIVERSE.get_or_init(|| {
                mpi::initialize().map(MpiUniverseHolder)
            });
            if holder.is_some() {
                let w = SimpleCommunicator::world();
                let rank = w.rank() as usize;
                let size = w.size() as usize;
                return Self { mpi_active: size > 1, rank, size };
            }
        }

        Self {
            rank: 0,
            size: 1,
            #[cfg(feature = "mpi")]
            mpi_active: false,
        }
    }

    /// The world communicator (MPI_COMM_WORLD) for a collective, or `None` when
    /// there is no active multi-rank MPI world (single rank, or feature off).
    /// Reconstructed on demand — cheap, just wraps the global handle — and must
    /// only be called from a serial reduction point (NOT inside a rayon closure;
    /// the returned communicator is `!Send`, which is exactly why the context
    /// itself does not carry it).
    #[cfg(feature = "mpi")]
    pub fn world(&self) -> Option<SimpleCommunicator> {
        if self.mpi_active {
            Some(SimpleCommunicator::world())
        } else {
            None
        }
    }

    /// Construct a context for a SPECIFIC (rank, size) WITHOUT an active MPI
    /// world — the reduction paths are no-ops (`world()` is `None`). Used by
    /// tests to simulate rank sharding (build the per-rank partial and sum the
    /// partials by hand) without launching real MPI. Since the private
    /// `mpi_active` field can't be set via a struct literal from another crate,
    /// this is the supported way to make a rank/size-pinned context.
    pub fn for_rank(rank: usize, size: usize) -> Self {
        Self {
            rank,
            size,
            #[cfg(feature = "mpi")]
            mpi_active: false,
        }
    }

    pub fn is_root(&self) -> bool {
        self.rank == 0
    }

    /// Contiguous aux-band `[p0, p1)` this rank owns when `n` items (aux
    /// functions) are striped across `size` ranks in balanced contiguous bands.
    ///
    /// The first `n % size` ranks get `⌈n/size⌉` items; the rest get `⌊n/size⌋`.
    /// Bands are DISJOINT and their union is exactly `[0, n)`, so summing each
    /// rank's band-restricted partial reproduces the full sum. With `size == 1`
    /// (or the `mpi` feature off) this is the full range `[0, n)`, so non-MPI /
    /// single-rank behavior is unchanged.
    ///
    /// Contiguous (not round-robin `i % size`) bands are used deliberately: the
    /// DF-JK B tensor is aux-major, so a contiguous aux band is a contiguous
    /// slice each rank can build and hold on its own — that is what makes the
    /// resident memory (not just the FLOPs) scale with rank count.
    pub fn aux_band(&self, n: usize) -> (usize, usize) {
        aux_band_for(n, self.rank, self.size)
    }
}

/// Pure balanced contiguous partition: item range `[p0, p1)` for `rank` of
/// `size` over `n` items. First `n % size` ranks get one extra item. Extracted
/// as a free function so it is unit-testable without an MPI context.
pub fn aux_band_for(n: usize, rank: usize, size: usize) -> (usize, usize) {
    let size = size.max(1);
    let base = n / size;
    let rem = n % size;
    // Ranks [0, rem) get base+1; ranks [rem, size) get base.
    let p0 = rank * base + rank.min(rem);
    let count = base + if rank < rem { 1 } else { 0 };
    (p0, p0 + count)
}

impl ParallelContext {

    /// Run a task only on the root process.
    pub fn root_only<F, R>(&self, f: F) -> Option<R>
    where
        F: FnOnce() -> R,
    {
        if self.is_root() {
            Some(f())
        } else {
            None
        }
    }

    /// Check if a global interrupt (SIGINT) has been requested.
    pub fn check_interrupted(&self) -> Result<(), crate::FerricError> {
        if crate::INTERRUPT.load(std::sync::atomic::Ordering::Relaxed) {
            Err(crate::FerricError::Libint("Interrupted by user".into()))
        } else {
            Ok(())
        }
    }
}

impl Default for ParallelContext {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::aux_band_for;

    #[test]
    fn aux_band_single_rank_is_full_range() {
        assert_eq!(aux_band_for(113, 0, 1), (0, 113));
        // size 0 is treated as 1 (defensive).
        assert_eq!(aux_band_for(113, 0, 0), (0, 113));
    }

    #[test]
    fn aux_bands_partition_disjointly_and_cover() {
        // naux not divisible by size: 113 over 4 → 29,28,28,28.
        let n = 113;
        let size = 4;
        let bands: Vec<_> = (0..size).map(|r| aux_band_for(n, r, size)).collect();
        assert_eq!(bands, vec![(0, 29), (29, 57), (57, 85), (85, 113)]);
        // Disjoint + covering: concatenation is exactly [0, n).
        let mut cursor = 0;
        let mut total = 0;
        for &(p0, p1) in &bands {
            assert_eq!(p0, cursor, "bands must be contiguous with no gap/overlap");
            assert!(p1 >= p0);
            total += p1 - p0;
            cursor = p1;
        }
        assert_eq!(cursor, n);
        assert_eq!(total, n);
    }

    #[test]
    fn aux_bands_balanced_within_one() {
        // Every band width is either ⌊n/size⌋ or ⌈n/size⌉.
        let n = 450;
        let size = 7;
        let lo = n / size;
        let hi = lo + if n % size == 0 { 0 } else { 1 };
        for r in 0..size {
            let (p0, p1) = aux_band_for(n, r, size);
            let w = p1 - p0;
            assert!(w == lo || w == hi, "band width {w} not in {{{lo},{hi}}}");
        }
    }

    #[test]
    fn aux_bands_more_ranks_than_items() {
        // 3 items over 8 ranks: first 3 ranks get 1 each; ranks 3..8 own nothing
        // (empty bands). With base=0, rem=3, every empty rank's p0 = rank*0 +
        // min(rank,3) = 3, so their bands are the empty range (3,3). An empty
        // band is fine: it contributes no aux rows to the reduction.
        let bands: Vec<_> = (0..8).map(|r| aux_band_for(3, r, 8)).collect();
        assert_eq!(
            bands,
            vec![(0, 1), (1, 2), (2, 3), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3)]
        );
        // Every rank width is 0 or 1, and the non-empty bands still tile [0,3).
        for (r, &(p0, p1)) in bands.iter().enumerate() {
            assert!(p1 - p0 <= 1, "rank {r} band too wide");
        }
    }
}
