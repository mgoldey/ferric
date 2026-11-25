//! The shared-primitive kernel switch of the rotated direct J/K walk
//! (`[scf] gc_kernel`, `FERRIC_GC_KERNEL`).
//!
//! # What it does
//!
//! On the contraction-rotated walk ([`super::gc_rotation`]) the shells of one
//! atom and `l` that share primitives form groups
//! ([`ferric_integrals::gc_kernel`]). With the kernel on, every surviving
//! quartet with `l1 + l2 + l3 + l4 <= 1` (`(ss|ss)`, and `(ss|sp)` with the
//! p anywhere) is taken from a block the kernel evaluates once per *group*
//! quartet, from one pass over the union primitives, instead of from a
//! libint call per column quartet:
//!
//! * the bra-pair list is reordered *group first*: the first bra pair (in
//!   the production order) of an ordered bra group pair with `l1 + l2 <= 1`
//!   pulls the other members of that group pair in right after it, so the
//!   kernel blocks of a bra group pair are evaluated once and reused by all
//!   its bra pairs ([`group_first`]);
//! * the deterministic group partition, and the MPI striping, take such a
//!   run of members (a *bucket*) as one unit, so no reduction group and no
//!   rank splits a bucket ([`Partition`]); both stay pure functions of the
//!   bra-pair list;
//! * in the ket loop the screen is unchanged (per column quartet); the
//!   group block is computed on the bucket's first survivor in that ket
//!   group pair and cached for the rest of the bucket ([`KernelCache`]);
//!   each surviving column quartet is gathered into libint's layout and
//!   digested by the same pair digest, in the same canonical `(s3, s4)`
//!   order; everything else goes to libint unchanged.
//!
//! The integrals differ from libint's at the 1e-15 relative level, so J and
//! K move by about 1e-14 relative and energies by ≤ 1e-11 Ha. Results stay
//! bitwise identical across thread counts: the cached blocks are functions
//! of the group quartet alone, and the partition and fold order are
//! functions of the work list.
//!
//! # Where it runs
//!
//! Only on the rotated walk of [`super::DirectJK`] (closed and open shell,
//! full and incremental) with the pair digest
//! ([`crate::quartet_scatter::JkDigest::Pair`]) and the Coulomb operator.
//! Everything else keeps libint: the unrotated walk (segmented bases,
//! `gc_rotation = "off"`), the legacy digest, `DirectJ` / `DirectK`, the
//! pooled builds (`rhf::build_jk_with_pool`: Newton, stability, CPHF,
//! Z-vector), the in-core store, LinK, CFMM, density fitting, gradients and
//! every non-Coulomb operator.
//!
//! # The switch
//!
//! `auto` or `off`, recorded on the bounds like the digest
//! ([`GcKernelSlot`]): a config's value (`RhfConfig::gc_kernel`), else
//! `FERRIC_GC_KERNEL`, else **`off`** (the default for this release). `off`
//! is today's walk bit for bit, its bra-pair order included. The setting is
//! agreed across MPI ranks before a rotated build dispatches.

use ferric_core::FerricError;
use ferric_integrals::gc_kernel::{GcKernelBasis, KernelScratch, PairId, PairKind};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Environment variable selecting the kernel where a config left it unset.
pub const GC_KERNEL_ENV: &str = "FERRIC_GC_KERNEL";

/// Whether the rotated direct walk takes the low-`l` classes from the
/// shared-primitive kernel (module doc).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GcKernel {
    /// The kernel for `l1 + l2 + l3 + l4 <= 1` wherever it applies.
    Auto,
    /// libint for every quartet, bit for bit the walk without the kernel.
    /// The default.
    #[default]
    Off,
}

/// `FERRIC_GC_KERNEL`: `auto` or `off`, case-insensitive, blanks ignored;
/// anything else is refused. Unset means `off`.
static GC_KERNEL: ferric_core::config::ConfigVar<GcKernel> = ferric_core::config::ConfigVar {
    env_name: GC_KERNEL_ENV,
    default: GcKernel::Off,
    parse: |s| GcKernel::parse_str(&s.trim().to_ascii_lowercase()),
    validate: ferric_core::config::accept_any,
};

impl GcKernel {
    /// The canonical spelling (`auto` / `off`).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Off => "off",
        }
    }

    /// Strict parse of a config value (`[scf] gc_kernel`, the Python
    /// `gc_kernel=` kwarg): exactly `auto` or `off`.
    pub fn parse_str(s: &str) -> Result<Self, String> {
        match s {
            "auto" => Ok(Self::Auto),
            "off" => Ok(Self::Off),
            _ => Err(format!(
                "invalid value {s:?}: expected exactly \"auto\" or \"off\""
            )),
        }
    }

    /// The setting a build runs: `explicit` (a config's) if set, else
    /// `FERRIC_GC_KERNEL`, else [`GcKernel::Off`]. A malformed variable is an
    /// error naming it.
    pub fn resolve(explicit: Option<GcKernel>) -> Result<Self, FerricError> {
        GC_KERNEL
            .resolve(explicit, ferric_core::config::env_lookup)
            .map(|r| r.value)
            .map_err(|e| FerricError::General(format!("{GC_KERNEL_ENV}: {e}")))
    }

    /// [`resolve`](Self::resolve) for a record (a run log, a checkpoint
    /// identity): the canonical spelling, or for a malformed variable
    /// `unresolved` with its raw text (the builds refuse it).
    pub fn resolved_label(explicit: Option<GcKernel>) -> String {
        match Self::resolve(explicit) {
            Ok(d) => d.as_str().to_string(),
            Err(_) => format!("unresolved {:?}", std::env::var(GC_KERNEL_ENV).ok()),
        }
    }

    /// The parsed `FERRIC_GC_KERNEL` of this process (unset = `off`).
    pub fn env_setting() -> Result<Self, FerricError> {
        Self::resolve(None)
    }
}

impl std::fmt::Display for GcKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The kernel setting a config recorded on one
/// [`SchwarzBounds`](crate::screening::SchwarzBounds) (its `gc_kernel`
/// field), shared by its clones and by the rotated bounds of its
/// contraction rotation, like [`crate::quartet_scatter::DigestSlot`].
/// Nothing recorded, or a config that left it unset, means
/// `FERRIC_GC_KERNEL`, then `off`. The last writer wins.
#[derive(Clone, Default)]
pub struct GcKernelSlot(Arc<RwLock<Option<GcKernel>>>);

impl std::fmt::Debug for GcKernelSlot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("GcKernelSlot")
            .field(&self.recorded())
            .finish()
    }
}

impl GcKernelSlot {
    /// Record a config's `gc_kernel` (`None`: unset).
    pub fn set(&self, mode: Option<GcKernel>) {
        if let Ok(mut g) = self.0.write() {
            *g = mode;
        }
    }

    /// What a config recorded (`None`: nothing, or unset).
    pub fn recorded(&self) -> Option<GcKernel> {
        self.0.read().map(|g| *g).unwrap_or_default()
    }

    /// Copy `other`'s record onto these bounds.
    pub fn copy_from(&self, other: &GcKernelSlot) {
        self.set(other.recorded());
    }

    /// The setting the builds on these bounds run.
    pub fn mode(&self) -> Result<GcKernel, FerricError> {
        GcKernel::resolve(self.recorded())
    }

    /// [`mode`](Self::mode) agreed across the MPI ranks of `ctx`: every rank
    /// runs the same setting (the kernel changes the bra-pair order and the
    /// rank striping), or every rank returns an error before the build's
    /// first collective. `mode` at one rank or without MPI.
    pub(crate) fn mode_collective(
        &self,
        ctx: &ferric_core::parallel::ParallelContext,
    ) -> Result<GcKernel, FerricError> {
        use super::gc_rotation::all_ranks_agree;
        let local = self.mode();
        let all_auto = all_ranks_agree(ctx, matches!(local, Ok(GcKernel::Auto)));
        let all_off = all_ranks_agree(ctx, matches!(local, Ok(GcKernel::Off)));
        match local {
            Ok(m) if all_auto || all_off => Ok(m),
            Ok(m) => Err(FerricError::General(format!(
                "gc kernel: this rank resolved {m}, another MPI rank a different setting or \
                 none ({GC_KERNEL_ENV} differs between ranks); set [scf] gc_kernel, or the \
                 variable alike on every rank"
            ))),
            Err(e) => Err(e),
        }
    }
}

// ---------------------------------------------------------------------------
// The work list: group-first order and the bucket-aware partition.
// ---------------------------------------------------------------------------

/// The kernel bucket of bra pair `(s1, s2)`: its ordered group pair, when
/// the kernel has data for it (`l1 + l2 <= 1`).
fn bucket(kb: &GcKernelBasis, s1: usize, s2: usize) -> Option<PairId> {
    if kb.shell_l(s1) + kb.shell_l(s2) > 1 {
        return None;
    }
    match kb.pair_of(s1, s2) {
        PairKind::Data(id) => Some(id),
        _ => None,
    }
}

/// `pairs` in group-first order, and the start of each unit (a bucket's run
/// of members, or a single bra pair) in it, with a final `len` sentinel.
/// The production order, except that the first member of a bucket pulls the
/// others in right after it (in their production order).
pub(crate) fn group_first(
    pairs: &[(usize, usize)],
    kb: &GcKernelBasis,
) -> (Vec<(usize, usize)>, Vec<usize>) {
    let mut members: HashMap<PairId, Vec<(usize, usize)>> = HashMap::new();
    for &(s1, s2) in pairs {
        if let Some(b) = bucket(kb, s1, s2) {
            members.entry(b).or_default().push((s1, s2));
        }
    }
    let mut out = Vec::with_capacity(pairs.len());
    let mut units = Vec::new();
    for &(s1, s2) in pairs {
        match bucket(kb, s1, s2) {
            Some(b) => {
                if let Some(m) = members.remove(&b) {
                    units.push(out.len());
                    out.extend(m);
                }
            }
            None => {
                units.push(out.len());
                out.push((s1, s2));
            }
        }
    }
    debug_assert_eq!(out.len(), pairs.len());
    units.push(out.len());
    (out, units)
}

/// How a walk's (rank-local) bra-pair list is cut into reduction groups.
#[derive(Debug, Clone)]
pub(crate) enum Partition {
    /// `deterministic_group_size(n)` pairs per group (the walk without the
    /// kernel, unchanged).
    Even { n: usize, size: usize },
    /// Group `g` is `bounds[g]..bounds[g + 1]`: whole units, at least
    /// `deterministic_group_size(n)` pairs per group except the last.
    Units(Vec<usize>),
}

impl Partition {
    /// The partition of `n` pairs without the kernel.
    pub(crate) fn even(n: usize) -> Self {
        Partition::Even {
            n,
            size: crate::reduce::deterministic_group_size(n),
        }
    }

    /// Whole units (`units`: unit starts with a final sentinel) of a list of
    /// `n` pairs, filled to at least `deterministic_group_size(n)` each.
    fn units(units: &[usize]) -> Self {
        let n = units.last().copied().unwrap_or(0);
        let size = crate::reduce::deterministic_group_size(n);
        let mut bounds = vec![0];
        for &u in &units[1..] {
            if u - bounds.last().copied().unwrap_or(0) >= size {
                bounds.push(u);
            }
        }
        if *bounds.last().unwrap_or(&0) != n {
            bounds.push(n);
        }
        Partition::Units(bounds)
    }

    /// Number of groups (at least one).
    pub(crate) fn n_groups(&self) -> usize {
        match self {
            Partition::Even { n, size } => n.div_ceil((*size).max(1)).max(1),
            Partition::Units(b) => (b.len() - 1).max(1),
        }
    }

    /// The pairs of group `g`.
    pub(crate) fn range(&self, g: usize) -> std::ops::Range<usize> {
        match self {
            Partition::Even { n, size } => {
                let lo = (g * size).min(*n);
                lo..(lo + size).min(*n)
            }
            Partition::Units(b) => match (b.get(g), b.get(g + 1)) {
                (Some(&lo), Some(&hi)) => lo..hi,
                _ => 0..0,
            },
        }
    }
}

/// The kernel's work list for this rank: the group-first order of the full
/// list `pairs`, striped across the ranks of `ctx` by whole units
/// (round-robin over units, as `ParallelContext::stripe` does over pairs),
/// and its unit partition.
pub(crate) fn work_list(
    ctx: &ferric_core::parallel::ParallelContext,
    pairs: &[(usize, usize)],
    kb: &GcKernelBasis,
) -> (Vec<(usize, usize)>, Partition) {
    let (order, units) = group_first(pairs, kb);
    let (rank, size) = (ctx.rank, ctx.size.max(1));
    let mut local = Vec::with_capacity(order.len() / size + 1);
    let mut starts = Vec::new();
    for (u, w) in units.windows(2).enumerate() {
        if u % size == rank {
            starts.push(local.len());
            local.extend_from_slice(&order[w[0]..w[1]]);
        }
    }
    starts.push(local.len());
    (local, Partition::units(&starts))
}

// ---------------------------------------------------------------------------
// The per-group-accumulator block cache.
// ---------------------------------------------------------------------------

/// A multiplicative hash for the `u32` ket-pair keys of [`KernelCache`]
/// (the default SipHash costs more than the lookup it serves).
#[derive(Default, Clone, Copy)]
struct IdHasher(u64);

impl std::hash::Hasher for IdHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0 ^ u64::from(b)).wrapping_mul(0x100_0000_01b3);
        }
    }
    fn write_u32(&mut self, i: u32) {
        self.0 = u64::from(i).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    }
}

type IdBuild = std::hash::BuildHasherDefault<IdHasher>;

/// Counters of the kernel over a build.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct KernelCounts {
    /// Surviving column quartets served by the kernel (and digested).
    pub quartets: usize,
    /// Group quartets the kernel evaluated.
    pub evals: usize,
    /// Column quartets those evaluations computed (surviving or not).
    pub columns: usize,
}

impl KernelCounts {
    pub(crate) fn add(&mut self, o: &KernelCounts) {
        self.quartets += o.quartets;
        self.evals += o.evals;
        self.columns += o.columns;
    }
}

/// The kernel blocks of one bucket (the current bra group pair), owned by a
/// group accumulator of the pair digest: keyed by the ket group pair,
/// dropped when the bra group pair changes.
pub(crate) struct KernelCache<'k> {
    kb: &'k GcKernelBasis,
    /// The bucket the cached blocks belong to.
    bra: Option<PairId>,
    /// Ket pair → offset of its block in `arena`.
    map: HashMap<u32, usize, IdBuild>,
    /// The last ket pair looked up and its offset (consecutive survivors
    /// often share the ket group pair).
    last: Option<(u32, usize)>,
    arena: Vec<f64>,
    scratch: KernelScratch,
    /// The gathered column quartet (at most 3 integrals).
    pub(crate) q: [f64; 3],
    pub(crate) counts: KernelCounts,
}

impl std::fmt::Debug for KernelCache<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KernelCache")
            .field("bra", &self.bra)
            .field("counts", &self.counts)
            .finish_non_exhaustive()
    }
}

impl<'k> KernelCache<'k> {
    pub(crate) fn new(kb: &'k GcKernelBasis) -> Self {
        KernelCache {
            kb,
            bra: None,
            map: HashMap::default(),
            last: None,
            arena: Vec::new(),
            scratch: KernelScratch::default(),
            q: [0.0; 3],
            counts: KernelCounts::default(),
        }
    }

    /// The bucket of bra pair `(s1, s2)` when the kernel serves its
    /// `l <= 1` kets (dropping the cached blocks of another bucket), with
    /// `l1 + l2`.
    #[inline]
    pub(crate) fn begin(&mut self, s1: usize, s2: usize) -> Option<(PairId, i32)> {
        let b = bucket(self.kb, s1, s2)?;
        if self.bra != Some(b) {
            self.bra = Some(b);
            self.map.clear();
            self.last = None;
            self.arena.clear();
        }
        Some((b, self.kb.shell_l(s1) + self.kb.shell_l(s2)))
    }

    /// Whether the kernel serves `(s3, s4)` against a bra of `l12`.
    #[inline(always)]
    pub(crate) fn covers(&self, l12: i32, s3: usize, s4: usize) -> bool {
        l12 + self.kb.shell_l(s3) + self.kb.shell_l(s4) <= 1
    }

    /// Gather column quartet `(s1 s2|s3 s4)` of bucket `bra` into
    /// [`q`](Self::q), evaluating its group block on first use; `None`
    /// (libint serves it) when the kernel has no data for the ket group
    /// pair. The number of integrals otherwise.
    pub(crate) fn gather(&mut self, bra: PairId, s: [usize; 4]) -> Option<usize> {
        let PairKind::Data(ket) = self.kb.pair_of(s[2], s[3]) else {
            return None;
        };
        let key = ket.index() as u32;
        let hit = match self.last {
            Some((k, o)) if k == key => Some(o),
            _ => self.map.get(&key).copied(),
        };
        let off = match hit {
            Some(o) => o,
            None => {
                let o = self.arena.len();
                let len = self.kb.block_len(bra, ket);
                self.arena.resize(o + len, 0.0);
                self.kb
                    .eval(bra, ket, &mut self.arena[o..], &mut self.scratch);
                self.counts.evals += 1;
                self.counts.columns += self.kb.columns(bra, ket);
                self.map.insert(key, o);
                o
            }
        };
        self.last = Some((key, off));
        let n = self.kb.gather(&self.arena[off..], bra, ket, s, &mut self.q);
        self.counts.quartets += 1;
        Some(n)
    }
}

/// A test probe: the kernel counts of the closed- and open-shell direct
/// builds run on this thread since [`start`](kernel_probe::start). Not part
/// of the API.
#[doc(hidden)]
pub mod kernel_probe {
    use super::KernelCounts;
    use std::cell::RefCell;

    thread_local! {
        static LOG: RefCell<Option<Vec<KernelCounts>>> = const { RefCell::new(None) };
    }

    /// Start recording this thread's builds (clearing any record).
    pub fn start() {
        LOG.with(|l| *l.borrow_mut() = Some(Vec::new()));
    }

    /// One entry per rotated pair-digest build since [`start`] (all zero for
    /// a build with the kernel off), and stop recording.
    pub fn take() -> Vec<KernelCounts> {
        LOG.with(|l| l.borrow_mut().take().unwrap_or_default())
    }

    pub(crate) fn record(c: KernelCounts) {
        LOG.with(|l| {
            if let Some(v) = l.borrow_mut().as_mut() {
                v.push(c);
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spelling_is_strict_and_default_is_off() {
        assert_eq!(GcKernel::parse_str("auto"), Ok(GcKernel::Auto));
        assert_eq!(GcKernel::parse_str("off"), Ok(GcKernel::Off));
        for bad in ["", "on", "Auto", " auto", "OFF", "pair", "1"] {
            assert!(GcKernel::parse_str(bad).is_err(), "{bad:?}");
        }
        assert_eq!(GcKernel::default(), GcKernel::Off);
        assert_eq!(
            GcKernel::resolve(Some(GcKernel::Auto)).unwrap(),
            GcKernel::Auto
        );
    }

    #[test]
    fn slot_is_shared_by_clones_and_copied() {
        let a = GcKernelSlot::default();
        assert_eq!(a.recorded(), None);
        let b = a.clone();
        a.set(Some(GcKernel::Auto));
        assert_eq!(b.recorded(), Some(GcKernel::Auto));
        let c = GcKernelSlot::default();
        c.copy_from(&b);
        assert_eq!(c.mode().unwrap(), GcKernel::Auto);
        let ctx = ferric_core::parallel::ParallelContext::default();
        assert_eq!(c.mode_collective(&ctx).unwrap(), GcKernel::Auto);
    }

    /// The block cache against a fresh evaluation, bitwise, for every
    /// covered quartet of a rotated cc-pVDZ basis, with the bra pairs in
    /// shell order so the bucket changes between most of them: a stale
    /// block (one cached for another bra bucket) or a ket key that aliases
    /// another ket group pair would differ.
    #[test]
    fn cache_serves_the_block_of_the_current_bucket() {
        use ferric_integrals::basis_bridge::PreparedBasis;
        use ferric_integrals::contraction_rotation::{RotatedBasis, RotationLevel};
        let mol =
            ferric_core::mol::Molecule::load_xyz("../../testdata/molecules/c2h4.xyz").unwrap();
        let parent =
            PreparedBasis::new(&mol, &ferric_core::basis::bundled("cc-pvdz").unwrap()).unwrap();
        let rot = RotatedBasis::detect_prepared(&parent, RotationLevel::Ogc)
            .unwrap()
            .unwrap();
        let kb = GcKernelBasis::for_rotated(&rot, 1e-20).unwrap();
        let nsh = rot.prep.nshells();
        let mut cache = KernelCache::new(&kb);
        let mut scratch = KernelScratch::default();
        let (mut block, mut want) = (Vec::new(), [0.0; 3]);
        let (mut served, mut switches, mut last) = (0usize, 0usize, None);
        for s1 in 0..nsh {
            for s2 in 0..=s1 {
                let Some((bra, l12)) = cache.begin(s1, s2) else {
                    continue;
                };
                if last != Some(bra) {
                    switches += 1;
                    last = Some(bra);
                }
                for s3 in 0..nsh {
                    for s4 in 0..nsh {
                        if !cache.covers(l12, s3, s4) {
                            continue;
                        }
                        let Some(n) = cache.gather(bra, [s1, s2, s3, s4]) else {
                            continue;
                        };
                        let PairKind::Data(ket) = kb.pair_of(s3, s4) else {
                            unreachable!("gather served a ket without data");
                        };
                        block.resize(kb.block_len(bra, ket), 0.0);
                        kb.eval(bra, ket, &mut block, &mut scratch);
                        assert_eq!(kb.gather(&block, bra, ket, [s1, s2, s3, s4], &mut want), n);
                        for (x, y) in cache.q[..n].iter().zip(&want[..n]) {
                            assert_eq!(x.to_bits(), y.to_bits(), "({s1} {s2}|{s3} {s4})");
                        }
                        served += 1;
                    }
                }
            }
        }
        assert!(served > 1000 && switches > 50, "{served} {switches}");
        assert_eq!(cache.counts.quartets, served);
        assert!(cache.counts.evals < served, "no block was reused");
    }

    /// The unit partition never splits a unit, covers the list in order and
    /// fills each group (but the last) to the even partition's size.
    #[test]
    fn unit_partition_keeps_units_whole() {
        // Units of sizes 1, 4, 1, 1, 9, 2, ... over 3000 pairs.
        let mut starts = vec![0usize];
        let sizes = [1usize, 4, 1, 1, 9, 2, 3, 1];
        let mut i = 0;
        while *starts.last().unwrap() < 3000 {
            let next = (starts.last().unwrap() + sizes[i % sizes.len()]).min(3000);
            starts.push(next);
            i += 1;
        }
        let p = Partition::units(&starts);
        let size = crate::reduce::deterministic_group_size(3000);
        let mut covered = 0;
        for g in 0..p.n_groups() {
            let r = p.range(g);
            assert_eq!(r.start, covered);
            assert!(starts.contains(&r.start) && starts.contains(&r.end));
            if g + 1 < p.n_groups() {
                assert!(r.len() >= size);
            }
            covered = r.end;
        }
        assert_eq!(covered, 3000);
        // An empty list is one empty group.
        let p = Partition::units(&[0]);
        assert_eq!(p.n_groups(), 1);
        assert_eq!(p.range(0), 0..0);
        // The even partition is the walk's own arithmetic.
        let e = Partition::even(3000);
        assert_eq!(e.range(1), size..2 * size);
    }
}
