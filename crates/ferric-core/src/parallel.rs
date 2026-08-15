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

/// Whether this build of `ferric-core` has the `mpi` feature compiled in.
///
/// Exported so DOWNSTREAM crates can statically assert that their own `mpi`
/// feature was enabled alongside core's. Enabling `ferric-core/mpi` *bare* is a
/// silent-wrong-answer configuration: [`ParallelContext`] then reports a real
/// multi-rank world (so e.g. `ferric_scf::df_j::DfJ` stripes the aux band), but
/// the matching `#[cfg(feature = "mpi")]` Allreduce lives in the DOWNSTREAM
/// crate and never compiles in — each rank converges "successfully" to a
/// different wrong energy built from only its own band. See the comment on
/// `ferric-mp2`'s `mpi` feature for the empirical trace of that failure.
///
/// Every crate's own `mpi` feature already chains `ferric-scf/mpi`, so this
/// only fires for a hand-rolled `--features ferric-core/mpi`.
pub const MPI_ENABLED: bool = cfg!(feature = "mpi");

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

    /// Round-robin MPI rank striping of a flat work list: keep only the items
    /// whose index is congruent to this rank mod world size.
    ///
    /// This is a DISJOINT, COVERING partition of `items` — every index `i`
    /// lands on exactly one rank (`i % size == rank` for exactly one `rank` in
    /// `0..size`), and concatenating every rank's kept items (in original
    /// index order, which this preserves since `Vec::into_iter` is order-
    /// preserving) reproduces `items` exactly. Summing each rank's
    /// stripe-restricted partial therefore reproduces the full sum — the same
    /// property `aux_band` provides for contiguous bands, just round-robin
    /// instead of contiguous.
    ///
    /// Round-robin (not contiguous) striping is used here because the callers
    /// (direct J/K/JK Fock builders, LinK) stripe a canonical shell-*pair*
    /// list whose per-pair cost is wildly uneven (a pair's ket loop scales
    /// with its own screened partner count) — round-robin spreads that
    /// unevenness across ranks, where a contiguous band could hand one rank
    /// an unlucky run of expensive pairs.
    ///
    /// With `size == 1` (feature off, or a single rank), `idx % 1 == 0` is
    /// trivially true for every idx, so this is a no-op and the returned list
    /// is byte-identical to `items` — preserving single-rank/non-MPI
    /// behavior, and the thread-count/rank-count bit-identity invariant the
    /// direct builders rely on (see their `*_bit_identical_across_thread_counts`
    /// tests).
    pub fn stripe<T>(&self, items: Vec<T>) -> Vec<T> {
        let (rank, size) = (self.rank, self.size);
        items
            .into_iter()
            .enumerate()
            .filter(|(idx, _)| idx % size == rank)
            .map(|(_, item)| item)
            .collect()
    }

    /// How many rayon worker threads THIS rank should use, given how many
    /// sibling ranks share its node.
    ///
    /// See [`threads_per_rank`] for the policy and [`local_ranks_per_node`] for
    /// how the local rank count is detected. This is the method form: it reads
    /// the launcher's environment and this machine's physical core count.
    ///
    /// This is a RECOMMENDATION, not an enforcement — nothing here touches the
    /// global rayon pool. The caller decides what to do with it (build a
    /// bounded pool, pass it to `ThreadPoolBuilder::num_threads`, …). Keeping
    /// it a pure query is what makes it testable without an MPI launcher and
    /// what keeps a single-rank run byte-identical: at 1 local rank the answer
    /// is simply "every physical core", which is what an unbounded rayon pool
    /// would already have used on a 1-socket box.
    pub fn rayon_threads(&self) -> usize {
        threads_per_rank(local_ranks_per_node(), physical_cores())
    }
}

/// Threads each rank should take when `local_ranks` ranks share a node with
/// `physical_cores` physical cores.
///
/// ## Why this exists
///
/// Nothing previously derived the rayon pool width from the local rank count,
/// so `mpirun -np 4` spawned FOUR full-width rayon pools on the SAME cores —
/// 4× oversubscription. That is the same hazard class [`crate::blas_threads`]
/// documents for rayon × OpenBLAS (a *product* of threads, 3–5× slowdown),
/// just with MPI ranks as the outer factor instead of BLAS. A hybrid
/// rank×thread decomposition only pays off if the product is bounded, so the
/// pool width has to come from the local rank count.
///
/// ## Policy (stated, not an accident of integer division)
///
/// `threads = max(1, floor(physical_cores / local_ranks))`.
///
/// * **Floor, not round or ceil.** `ranks × threads <= physical_cores` is the
///   invariant worth keeping; rounding up breaks it. On 6 cores with 4 ranks
///   the answer is `floor(6/4) = 1`, so 4 ranks × 1 thread = 4 cores used and
///   2 left IDLE. Deliberately: 4 ranks × 2 threads would be 8 threads on 6
///   cores, and reintroducing oversubscription to chase 2 idle cores trades a
///   bounded loss (33% idle) for an unbounded one (contention). A caller who
///   wants the remainder must hand out uneven widths itself; this function
///   returns ONE number that is correct for EVERY rank, which is what makes it
///   safe to call independently on each rank with no collective agreement.
/// * **Floored at 1.** More local ranks than cores (`local_ranks > cores`) is
///   already oversubscribed at the RANK level — the launcher did that, not us —
///   and returning 0 would build an empty/defaulted rayon pool, which rayon
///   interprets as "use all cores" and would make the oversubscription
///   dramatically worse. 1 is the least-bad answer.
/// * **PHYSICAL cores, not logical.** See [`physical_cores`]. Sizing to
///   SMT siblings hands each rank threads that share an execution unit, which
///   for the GEMM-bound work in RI-MP2 is contention, not parallelism.
/// * **`local_ranks == 0` is treated as 1** (defensive; a launcher reporting 0
///   is nonsense, and the single-rank answer is the safe interpretation).
///
/// ## Single-rank invariant
///
/// `threads_per_rank(1, c) == c` for every `c >= 1`. A non-MPI run, a
/// `-np 1` run, and a feature-off build therefore all get the full core count
/// — exactly the unbounded-pool width they had before this function existed.
/// Pinned by `threads_per_rank_single_rank_is_all_cores`.
pub fn threads_per_rank(local_ranks: usize, physical_cores: usize) -> usize {
    let local_ranks = local_ranks.max(1);
    let physical_cores = physical_cores.max(1);
    (physical_cores / local_ranks).max(1)
}

/// Number of MPI ranks sharing THIS node ("local size"), or 1 when not under a
/// recognized launcher.
///
/// **Local, not world.** World size is the wrong number: 8 ranks spread over 4
/// nodes is 2 ranks per node, and sizing the pool to 8 would leave 3/4 of each
/// node idle. Only the ranks that actually contend for THIS node's cores
/// matter. (On this project's single-node box the two happen to coincide, which
/// is exactly why it would be easy to get wrong and never notice.)
///
/// Launcher variables, in precedence order:
///
/// * `OMPI_COMM_WORLD_LOCAL_SIZE` — Open MPI (what ferric uses; 4.1.6 here).
/// * `MV2_COMM_WORLD_LOCAL_SIZE` — MVAPICH2.
/// * `MPI_LOCALNRANKS` — MPICH / Hydra.
/// * `SLURM_NTASKS_PER_NODE` — Slurm's `srun` when it launches ranks directly
///   (no `mpirun` in between), so none of the above are set.
///
/// **Fallback is 1, deliberately.** An unrecognized launcher yields the
/// single-rank answer, i.e. the full-width pool ferric used before — the
/// pre-existing behavior, which is at worst as oversubscribed as today and
/// never worse. The alternative (guessing from world size, or from the number
/// of sibling processes) would silently NARROW the pool on a machine where the
/// guess is wrong, turning a missing env var into a performance regression on
/// runs that have nothing to do with MPI.
pub fn local_ranks_per_node() -> usize {
    local_ranks_per_node_with(|k| std::env::var(k).ok())
}

/// [`local_ranks_per_node`] with an injected env lookup — the testable core.
///
/// Same injection rationale as [`crate::blas_threads::opt_in_blas_threads_with`]:
/// the process environment is global and the test harness is multi-threaded, so
/// a `set_var`-based test races every other test in the binary.
#[doc(hidden)]
pub fn local_ranks_per_node_with(get: impl Fn(&str) -> Option<String>) -> usize {
    const LOCAL_SIZE_VARS: [&str; 4] = [
        "OMPI_COMM_WORLD_LOCAL_SIZE",
        "MV2_COMM_WORLD_LOCAL_SIZE",
        "MPI_LOCALNRANKS",
        "SLURM_NTASKS_PER_NODE",
    ];
    for key in LOCAL_SIZE_VARS {
        if let Some(v) = get(key) {
            // Garbage parses to nothing and we fall through to the next var /
            // the default. A thread count is a performance knob, not a
            // result-affecting one, so a malformed value must degrade rather
            // than abort a converged run (the same exception class
            // `blas_threads::BLAS_THREADS` documents). Note SLURM_NTASKS_PER_NODE
            // can carry a comma list ("4,2") on heterogeneous allocations; the
            // parse fails on that and we fall back to 1 rather than guess.
            if let Ok(n) = v.trim().parse::<usize>() {
                if n >= 1 {
                    return n;
                }
            }
        }
    }
    1
}

/// Physical core count of this machine (SMT siblings collapsed), falling back
/// to the logical count when topology is unreadable.
///
/// **Why physical.** On the 6-core / 12-thread box this was written for,
/// `std::thread::available_parallelism()` reports 12. Sizing a rank×thread grid
/// to 12 makes `2 ranks × 6 threads` look like "all cores" when it is really 2
/// ranks × 6 HYPERTHREADS on 6 physical cores — the two threads in a pair share
/// one set of execution units, so for the BLAS3-bound GEMMs in RI-MP2 the
/// second thread mostly adds contention. Worse, it makes an SMT effect
/// indistinguishable from a hybrid-decomposition effect in any measurement.
///
/// Derived from `/sys/devices/system/cpu/cpu*/topology/thread_siblings_list`:
/// each online CPU reports the set of logical CPUs sharing its physical core,
/// so the number of DISTINCT sibling sets is the physical core count. This is
/// the same information `lscpu` computes "Core(s) per socket × Socket(s)" from.
///
/// Fallbacks, in order: sysfs topology → `available_parallelism()` (logical) →
/// 1. The logical fallback is intentionally NOT halved: on a machine without
/// SMT that would waste half the box, and we cannot tell the two cases apart
/// without the topology we just failed to read.
pub fn physical_cores() -> usize {
    if let Some(n) = physical_cores_from_sysfs("/sys/devices/system/cpu") {
        return n;
    }
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

/// [`physical_cores`] with an injected sysfs root — the testable core. Returns
/// `None` when the topology is unreadable (non-Linux, container without sysfs,
/// or a kernel that does not export `thread_siblings_list`), so the caller can
/// apply its documented fallback chain.
#[doc(hidden)]
pub fn physical_cores_from_sysfs(cpu_root: &str) -> Option<usize> {
    let entries = std::fs::read_dir(cpu_root).ok()?;
    let mut sibling_sets: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        // Only cpuN directories; skip cpuidle/cpufreq/etc.
        if !name.starts_with("cpu") || !name[3..].chars().all(|c| c.is_ascii_digit()) || name.len() == 3 {
            continue;
        }
        let path = entry.path().join("topology/thread_siblings_list");
        // An OFFLINE cpu has no topology/ dir — read failure just skips it,
        // which is correct: an offline SMT sibling is not a core we can use.
        if let Ok(list) = std::fs::read_to_string(&path) {
            // e.g. "0,6" (SMT pair) or "3" (no SMT). The string is canonical
            // per core — every sibling reports the SAME list — so counting
            // distinct strings counts distinct physical cores.
            sibling_sets.insert(list.trim().to_string());
        }
    }
    if sibling_sets.is_empty() {
        None
    } else {
        Some(sibling_sets.len())
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
    use super::{local_ranks_per_node_with, physical_cores, physical_cores_from_sysfs, threads_per_rank};

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

    // ---- rank-aware thread-pool sizing -------------------------------------

    /// THE regression guard: a single-rank (or non-MPI, or feature-off) run
    /// must get the FULL core count — byte-identical pool width to before
    /// rank-aware sizing existed. A binding bug that narrows the single-rank
    /// pool is the one failure mode that would silently slow every ordinary
    /// serial run in the project.
    #[test]
    fn threads_per_rank_single_rank_is_all_cores() {
        for cores in 1..=64 {
            assert_eq!(
                threads_per_rank(1, cores),
                cores,
                "1 local rank must own every physical core (cores={cores})"
            );
        }
        // 0 local ranks (nonsense launcher value) is treated as 1.
        assert_eq!(threads_per_rank(0, 6), 6);
    }

    /// The invariant that makes hybrid worth doing at all: ranks × threads
    /// never exceeds the physical core count. This is what FAILS if the
    /// binding logic is broken to hand every rank the full core count (the
    /// mutation this test is designed to catch).
    #[test]
    fn ranks_times_threads_never_oversubscribes_physical_cores() {
        for cores in 1..=32 {
            for ranks in 1..=cores {
                let t = threads_per_rank(ranks, cores);
                assert!(t >= 1, "must never hand out a zero-width pool");
                assert!(
                    ranks * t <= cores,
                    "ranks({ranks}) x threads({t}) = {} exceeds physical cores ({cores})",
                    ranks * t
                );
            }
        }
    }

    /// The documented 6-core grid, spelled out. 1x6, 2x3, 3x2, 6x1 tile the box
    /// exactly; 4 ranks is the non-divisible case where the floor policy
    /// deliberately leaves 2 cores idle rather than oversubscribing to 8.
    #[test]
    fn threads_per_rank_six_core_grid() {
        assert_eq!(threads_per_rank(1, 6), 6);
        assert_eq!(threads_per_rank(2, 6), 3);
        assert_eq!(threads_per_rank(3, 6), 2);
        assert_eq!(threads_per_rank(6, 6), 1);
        // Non-divisible: floor(6/4) = 1, so 4 ranks use 4 of 6 cores. NOT 2
        // (which would be 8 threads on 6 cores).
        assert_eq!(threads_per_rank(4, 6), 1);
        assert_eq!(threads_per_rank(5, 6), 1);
        // 12 cores, 5 ranks: floor(12/5) = 2 -> 10 of 12 used, 2 idle.
        assert_eq!(threads_per_rank(5, 12), 2);
    }

    /// More local ranks than cores: already oversubscribed at the rank level by
    /// the launcher; we return 1 rather than 0 (a 0-width rayon pool means
    /// "all cores", which would make it far worse).
    #[test]
    fn threads_per_rank_floors_at_one_when_more_ranks_than_cores() {
        assert_eq!(threads_per_rank(12, 6), 1);
        assert_eq!(threads_per_rank(1000, 6), 1);
        assert_eq!(threads_per_rank(2, 1), 1);
    }

    /// Local size comes from the launcher env, in the documented precedence
    /// order, with a fallback of 1 (== today's full-width behavior) when
    /// nothing is set or the value is unparseable.
    #[test]
    fn local_ranks_detection_precedence_and_fallback() {
        // Nothing set -> 1 (unrecognized launcher / plain `cargo run`).
        assert_eq!(local_ranks_per_node_with(|_| None), 1);

        // Open MPI wins over every other var.
        let get = |k: &str| match k {
            "OMPI_COMM_WORLD_LOCAL_SIZE" => Some("3".to_string()),
            "MPI_LOCALNRANKS" => Some("7".to_string()),
            "SLURM_NTASKS_PER_NODE" => Some("9".to_string()),
            _ => None,
        };
        assert_eq!(local_ranks_per_node_with(get), 3);

        // MVAPICH2, then MPICH/Hydra, then Slurm, each when the ones above are
        // absent.
        assert_eq!(
            local_ranks_per_node_with(|k| (k == "MV2_COMM_WORLD_LOCAL_SIZE").then(|| "4".into())),
            4
        );
        assert_eq!(
            local_ranks_per_node_with(|k| (k == "MPI_LOCALNRANKS").then(|| "5".into())),
            5
        );
        assert_eq!(
            local_ranks_per_node_with(|k| (k == "SLURM_NTASKS_PER_NODE").then(|| "2".into())),
            2
        );

        // Garbage / zero degrade to the single-rank default rather than
        // aborting a run over a performance knob. A comma list (heterogeneous
        // Slurm allocation) is exactly this case.
        assert_eq!(
            local_ranks_per_node_with(|k| (k == "OMPI_COMM_WORLD_LOCAL_SIZE").then(|| "nope".into())),
            1
        );
        assert_eq!(
            local_ranks_per_node_with(|k| (k == "OMPI_COMM_WORLD_LOCAL_SIZE").then(|| "0".into())),
            1
        );
        assert_eq!(
            local_ranks_per_node_with(|k| (k == "SLURM_NTASKS_PER_NODE").then(|| "4,2".into())),
            1
        );

        // Unset OMPI + set MPICH: the loop must not stop at the first MISSING
        // var (a `for` that returned on the first key regardless would).
        assert_eq!(
            local_ranks_per_node_with(|k| (k == "MPI_LOCALNRANKS").then(|| "6".into())),
            6
        );
    }

    /// Physical-core detection must (a) return something usable and (b) never
    /// exceed the logical count. On an SMT box it should be strictly less; on a
    /// non-SMT box equal — this asserts the bound that holds either way.
    #[test]
    fn physical_cores_is_sane_and_at_most_logical() {
        let phys = physical_cores();
        let logical = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
        assert!(phys >= 1, "physical core count must be at least 1");
        assert!(
            phys <= logical,
            "physical cores ({phys}) cannot exceed logical cores ({logical})"
        );
    }

    /// The pool width a SINGLE-rank run installs must equal the physical core
    /// count — i.e. the width an unbounded rayon pool would have used on this
    /// single-socket box. This is the end-to-end form of the single-rank
    /// invariant: it goes through the SAME `ParallelContext::rayon_threads`
    /// call `run_mpi_ri_mp2` uses, rather than re-deriving it, so a future
    /// refactor that changes the method (not just `threads_per_rank`) is still
    /// caught.
    ///
    /// `ParallelContext::default()` is rank 0 of size 1 in a non-MPI test
    /// binary, and no launcher env var is set, so `local_ranks_per_node()`
    /// returns 1 — the exact configuration every ordinary serial ferric run
    /// has.
    #[test]
    fn single_rank_context_installs_full_width_pool() {
        let ctx = super::ParallelContext::default();
        assert_eq!(ctx.size, 1, "a non-MPI test binary must report a single rank");
        assert_eq!(
            ctx.rayon_threads(),
            physical_cores(),
            "a single-rank run must get the FULL physical core count — narrowing it \
             would silently slow every ordinary serial run"
        );
    }

    /// The sysfs parser counts DISTINCT sibling sets, so an SMT pair collapses
    /// to one core. Driven off a synthetic tree so it is deterministic on any
    /// machine (the real-hardware read is covered by the test above).
    #[test]
    fn physical_cores_from_sysfs_collapses_smt_siblings() {
        let root = std::env::temp_dir().join(format!(
            "ferric-cpu-topo-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        // 4 logical CPUs, 2 physical cores: (0,2) and (1,3).
        for (cpu, siblings) in [(0, "0,2"), (1, "1,3"), (2, "0,2"), (3, "1,3")] {
            let d = root.join(format!("cpu{cpu}/topology"));
            std::fs::create_dir_all(&d).unwrap();
            std::fs::write(d.join("thread_siblings_list"), format!("{siblings}\n")).unwrap();
        }
        // Decoys that must be ignored: a non-cpuN dir, and the bare `cpu` dir.
        std::fs::create_dir_all(root.join("cpuidle")).unwrap();
        std::fs::create_dir_all(root.join("cpufreq")).unwrap();

        assert_eq!(
            physical_cores_from_sysfs(root.to_str().unwrap()),
            Some(2),
            "an SMT pair must count as ONE physical core"
        );

        // Unreadable root -> None, so the caller falls back to the logical count.
        assert_eq!(physical_cores_from_sysfs("/definitely/not/a/sysfs/root"), None);

        let _ = std::fs::remove_dir_all(&root);
    }
}
