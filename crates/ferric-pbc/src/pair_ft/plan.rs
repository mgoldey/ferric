//! The production pair-FT kernel: the primitive-pair screen walked ONCE per
//! call, then every G chunk evaluated task-parallel from the cached site
//! lists (FINDINGS "Performance plan (research) — 2026-09-25" §5, "Pair-FT
//! re-walk", and §4, "General-contraction duplication").
//!
//! # Why
//!
//! The serial kernels (`super::pair_ft_block`, `super::residues::residue_block`,
//! kept FROZEN as the test oracles) re-walk every `(shell pair, image, bra
//! primitive, ket primitive)` combination — a `powf`, an `exp` and a `ln`
//! each — for EVERY G chunk, although both the screen and the per-pair G
//! window are chunk-independent (the window uses `max |G|` over the whole G
//! set). On dry ice (cc-pVDZ, range split on) that re-walk was the dominant,
//! serial "rsgdf LR pair FT" stage (326.8 s of 576 s, FINDINGS "Benchmark
//! series — quiet box").
//!
//! Per (surviving primitive pair, G) the serial kernels then evaluate one
//! `exp` (`e^{−G²/4p}`), one `cos` and one `sin` (`e^{−iG·P}`). Neither
//! depends on the contraction coefficients, and the `exp` does not depend on
//! the centres either, so this kernel evaluates each of them ONCE and reuses
//! the value (same expression, same inputs, hence the same bits):
//!
//! * `e^{−G²/4p}` depends only on the exponent pair `(a, b)`: one table row
//!   per distinct exponent pair (`slot`) and chunk ([`ChunkG::etab`]), shared
//!   by every image, shell pair and atom pair.
//! * `cos(G·P)`, `sin(G·P)` and the E tables depend on `(A, B + L, a, b, la,
//!   lb)` only. The columns of a general contraction (cc-pVDZ C/O: three `s`
//!   shells on the same 9 exponents, two `p` shells on the same 4) are
//!   separate shells with identical centre, `l` and exponents: a SHELL GROUP
//!   (a maximal run of consecutive such shells). Each `(image, bra primitive,
//!   ket primitive)` of a GROUP pair is a SITE; the trig values and the
//!   `F = E·(−iG)^t` rows are evaluated once per site and used by every
//!   member shell pair `(i, j)` of the group pair that has a survivor there
//!   (an ENTRY: the member, its `cc` and its window).
//!
//! # Bit-identity (per element, by construction)
//!
//! An output element `P[r][m, n, g]` is the sum, in serial walk order
//! (image ascending, then bra primitive, then ket primitive), of the terms of
//! the primitive pairs of `(shell(m), shell(n))` that pass
//! `mag >= thresh` AND whose window reaches `g` (`|G_g|² <= g2max`), followed
//! by the same Cartesian→AO transform. Nothing couples different G columns
//! or different shell pairs. So:
//!
//! * The sites of a group pair are in walk order and, within a site, one
//!   entry per member that passes the screen, with the screen's own `cc` and
//!   `g2max` (computed with the serial kernel's expressions from the member's
//!   own coefficients). A member's entries therefore appear in its serial
//!   walk order, and it accumulates into its OWN Cartesian block, so its
//!   addend sequence is the serial one. An entry whose window reaches no G
//!   of the whole set (`g2max < min |G|²`) contributed zero to every chunk
//!   in the serial kernel (`ngp == 0`) and is dropped.
//! * Every scalar of a term is the serial kernel's expression on bitwise the
//!   serial inputs: the site's `B + L`, `A − B′`, `p`, `P_c`, `G·P_c`, `cos`,
//!   `sin` and E tables are computed from the group's first shell, whose
//!   centre, `l` and exponents are bitwise every member's (the group key);
//!   the table value `e^{−G²/4p}` is `(−G² / (4 p)).exp()` with `p = a + b`
//!   of the same two exponent bit patterns; a member's term is then
//!   `mag = cc · e`, `(mag cos, −mag sin) · Fx · Fy · Fz` exactly as serial.
//!   The shared values are computed up to the LARGEST window of the site's
//!   entries (resp. the slot's); a member reads only its own window, and a
//!   value at `g` does not depend on how far past `g` it was computed.
//! * The parallel split is over TASKS `(bra shell group, contiguous ket-group
//!   range)`, formed once per plan ([`PairFtPlan::new`], deterministic greedy
//!   on a per-group-pair work weight, [`PAIR_FT_TASK_SPLIT`]): a task owns
//!   the rows of its bra group's shells in its ket range's columns in every
//!   `P[r]`, walks its ket groups in order and writes each of their elements
//!   exactly once, each member pair from a Cartesian accumulator it zeroes
//!   itself. No element is shared between tasks, and neither the split nor
//!   any element's term order depends on the thread count. (Finer than
//!   bra-group rows: with few shells a heavy contracted bra row is a tail at
//!   every per-chunk barrier. Coarser than single pairs, whose split
//!   measurably raised the serial kernel time: FINDINGS "Pair FT per-pair
//!   tasks: sub-stage measurement (2026-09-28)". The `pair FT cost max *`
//!   counters of [`PairFtPlan::record_stats`] measure the balance.)
//! * Beyond a shell pair's window (`g >= n_used`, the largest `ngp` of its
//!   entries, `n_used = 0` for a pair without entries) no term reaches its
//!   Cartesian accumulator, which therefore stays exactly `+0.0`, and the
//!   serial transform of `+0.0` is `+0.0` (`+0.0 + (±0.0) = +0.0`, the
//!   pure-shell accumulators start at `+0.0`). The kernel writes that `+0.0`
//!   directly there instead of zeroing, transforming and scattering it.
//!   Both shortcuts are counted ([`CTR_PARTIAL_WINDOW`]: `0 < n_used <
//!   n_G`; [`CTR_EMPTY_WINDOW`]: `n_used = 0`), and the bitwise test
//!   `pair_ft_kernels_are_bitwise_vs_frozen_serial_kernels_in_partial_windows`
//!   asserts the partial one is reached by both kernels.
//! * Every element of every `P[r]` is written (the tasks tile the matrix:
//!   `check_contiguous`, asserted again per chunk by
//!   [`PairFtPlan::segments`]; the scatter covers every `(i, j, g)`, the
//!   `+0.0` beyond the window included), so the output buffers are REUSED
//!   across chunks without zeroing ([`PairFtPlan::block_timed`]): a stale
//!   value is always overwritten. The multi-chunk bitwise tests would see a
//!   missed write, since the oracle starts every chunk from zero.
//!
//! Hence every element carries the serial scalar sequence: the output is BIT
//! FOR BIT the frozen serial kernel's, at any thread count and any G
//! chunking (`tests/pbc_parallel_bitwise.rs`; the shared-site path, i.e.
//! groups of more than one shell, by
//! `pair_ft_kernels_are_bitwise_vs_frozen_serial_kernels_with_general_contraction`,
//! which asserts [`CTR_SITES`] `<` survivors).
//!
//! # Memory
//!
//! The site and entry lists ([`SITE_BYTES`], [`SURVIVOR_BYTES`] each) are
//! counted before they are allocated and checked against ferric's unified
//! budget. Each rayon worker holds one group pair's scratch at a time
//! (`members × R` Cartesian blocks, the F rows, the trig row and the pure
//! half-transform, all `× n_G(chunk)`); with `T` threads `T` such sets are
//! live (a few KiB per G per member per thread against the `16 nao²` bytes
//! per G of the output itself). The per-chunk exp table is `8 × slots` bytes
//! per G (`slots` ≤ distinct exponent pairs).

use super::{build_shells, FtShell, WINDOW_MARGIN};
use crate::budget::{bytes_of, Ledger};
use crate::kpts::lattice_coords;
use crate::lattice::Cell;
use crate::pair_ft::residues::residue_index;
use crate::timing::{PbcTimings, StageClock};
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::md3c1e::{cart_components, e_table, ferric_cart2sph};
use ndarray::Array3;
use num_complex::Complex64;
use rayon::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// Sub-stage: the once-per-call screen walk ([`PairFtPlan::new`]).
pub(crate) const SUB_PLAN: &str = "pair FT: plan (screen walk, survivor lists)";
/// Sub-stage: the serial per-chunk setup (G sort, exp table, buffers, task
/// slices).
pub(crate) const SUB_SETUP: &str = "pair FT: chunk setup (serial)";
/// Sub-stage: the parallel per-chunk kernel.
pub(crate) const SUB_KERNEL: &str = "pair FT: chunk kernel (parallel)";

/// Counter: (shell pair, G chunk) combinations whose window ends INSIDE the
/// chunk (`0 < n_used < n_G`), i.e. that took the accumulate-then-`+0.0`-tail
/// path of the per-pair kernel. Summed over chunks and calls.
pub const CTR_PARTIAL_WINDOW: &str = "pair FT partial-window pair-chunks";
/// Counter: (shell pair, G chunk) combinations with `n_used == 0` (no
/// survivor reaches any G of the chunk; written as `+0.0` directly).
pub const CTR_EMPTY_WINDOW: &str = "pair FT empty-window pair-chunks";
/// Counter: cached primitive-pair survivors (entries), summed over plans.
pub const CTR_SURVIVORS: &str = "pair FT survivors (sum over plans)";
/// Counter: sites (distinct `(group pair, image, bra prim, ket prim)`),
/// summed over plans. `survivors − sites` trig/E evaluations per G are
/// saved by the shell groups; equal when every group is one shell.
pub const CTR_SITES: &str = "pair FT sites (sum over plans)";
/// Counter: shells in the largest shell group (max over plans).
pub const CTR_LARGEST_GROUP: &str = "pair FT largest shell group";

/// Task granularity: a task's work weight is capped at
/// `max(ceil(W / PAIR_FT_TASK_SPLIT), heaviest group pair)`, `W` the plan's
/// total weight (a shell pair weighs `(survivors + 1) × ncart(bra) ×
/// ncart(ket)`, the `+ 1` standing for its transform and scatter, which run
/// for every pair; a group pair the sum of its members). About
/// `PAIR_FT_TASK_SPLIT + ngroups` tasks per chunk. A constant, NOT a
/// function of the thread count: the split (hence every element's term
/// order, which it does not touch anyway) is the same on any pool.
pub(crate) const PAIR_FT_TASK_SPLIT: u64 = 256;

/// One `(image, bra primitive, ket primitive)` of a group pair with at
/// least one entry: `entries[e0..e1]` of its [`GroupPairList`], and the
/// exp-table `slot` of its exponent pair.
#[derive(Clone, Copy)]
struct Site {
    image: u32,
    ia: u16,
    ib: u16,
    slot: u32,
    e0: u32,
    e1: u32,
}

/// One screened primitive pair of one member shell pair at a site: the
/// member (`(i − s0(bra group)) × len(ket group) + (j − s0(ket group))`)
/// and the two screen scalars (`cc = c_a c_b (π/p)^{3/2}` and the G window
/// `g2max`) exactly as the serial kernel computed them.
#[derive(Clone, Copy)]
struct Entry {
    cc: f64,
    g2max: f64,
    member: u16,
}

/// Bytes of one cached survivor (entry).
pub(crate) const SURVIVOR_BYTES: usize = std::mem::size_of::<Entry>();
/// Bytes of one cached site.
pub(crate) const SITE_BYTES: usize = std::mem::size_of::<Site>();

/// The sites of one group pair in walk order, with their entries.
#[derive(Default)]
struct GroupPairList {
    sites: Vec<Site>,
    entries: Vec<Entry>,
}

impl GroupPairList {
    /// Append an entry at site `key` (a new site unless it is the last one).
    fn push(&mut self, key: (u32, u16, u16), slot: u32, e: Entry) {
        let n = self.entries.len() as u32;
        match self.sites.last_mut() {
            Some(s) if (s.image, s.ia, s.ib) == key => s.e1 = n + 1,
            _ => self.sites.push(Site {
                image: key.0,
                ia: key.1,
                ib: key.2,
                slot,
                e0: n,
                e1: n + 1,
            }),
        }
        self.entries.push(e);
    }
}

/// A maximal run `s0..s1` of consecutive shells with bitwise the same
/// centre, `l` and exponents (the columns of a general contraction).
#[derive(Clone, Copy)]
struct ShellGroup {
    s0: usize,
    s1: usize,
}

impl ShellGroup {
    fn len(&self) -> usize {
        self.s1 - self.s0
    }
}

/// The screen of one pair-FT call, walked once: shells, groups, images,
/// residue buckets and the per-group-pair site lists.
pub(crate) struct PairFtPlan {
    shells: Vec<FtShell>,
    groups: Vec<ShellGroup>,
    images: Vec<[f64; 3]>,
    /// Residue bucket of each image (all 0 for the Gamma kernel).
    bucket: Vec<usize>,
    nr: usize,
    nbf: usize,
    lmax: usize,
    /// `lists[ga * ngroups + gb]`, sites in serial walk order.
    lists: Vec<GroupPairList>,
    /// `p = a + b` of each exp-table slot (distinct exponent pair in use).
    slot_p: Vec<f64>,
    /// Largest entry window `g2max` of each slot.
    slot_g2max: Vec<f64>,
    /// Entries of each SHELL pair (`ia * nshells + ib`).
    nsurv_pair: Vec<u64>,
    /// Largest entry window `g2max` of each shell pair (`−∞` without
    /// entries): `n_used` of a chunk is its `partition_point`.
    g2max_pair: Vec<f64>,
    /// The per-chunk tasks, ordered by bra group then ket range; the ket
    /// ranges of a bra group tile `0..ngroups` in order.
    tasks: Vec<PairTaskSpec>,
    comps: Vec<Vec<[u8; 3]>>,
    c2s: Vec<Vec<f64>>,
    who: &'static str,
}

fn norm2(g: &[f64; 3]) -> f64 {
    g[0] * g[0] + g[1] * g[1] + g[2] * g[2]
}

fn same_bits(x: &[f64], y: &[f64]) -> bool {
    x.len() == y.len() && x.iter().zip(y).all(|(a, b)| a.to_bits() == b.to_bits())
}

/// The shell groups: maximal runs of consecutive shells with bitwise equal
/// centre, `l` and exponents.
fn build_groups(shells: &[FtShell]) -> Vec<ShellGroup> {
    let mut groups: Vec<ShellGroup> = Vec::new();
    for (s, sh) in shells.iter().enumerate() {
        if let Some(g) = groups.last_mut() {
            let rep = &shells[g.s0];
            if rep.l == sh.l && same_bits(&rep.center, &sh.center) && same_bits(&rep.exps, &sh.exps)
            {
                g.s1 = s + 1;
                continue;
            }
        }
        groups.push(ShellGroup { s0: s, s1: s + 1 });
    }
    groups
}

/// Per shell, the index of each primitive exponent in the sorted list of
/// distinct exponent bit patterns; and that list's length.
fn exponent_ids(shells: &[FtShell]) -> (Vec<Vec<u32>>, Vec<f64>) {
    let mut bits: Vec<u64> = shells
        .iter()
        .flat_map(|s| s.exps.iter().map(|e| e.to_bits()))
        .collect();
    bits.sort_unstable();
    bits.dedup();
    let ids = shells
        .iter()
        .map(|s| {
            s.exps
                .iter()
                .map(|e| bits.binary_search(&e.to_bits()).expect("exponent listed") as u32)
                .collect()
        })
        .collect();
    (ids, bits.into_iter().map(f64::from_bits).collect())
}

/// Site and entry counts of one group pair (the budget pass).
#[derive(Default, Clone, Copy)]
struct ListCount {
    sites: u64,
    entries: u64,
}

impl PairFtPlan {
    /// Walk the screen of `super::pair_ft_block` (`moduli = None`) or of
    /// `super::residues::residue_block` (`Some(moduli)`) once for a G set
    /// whose `max |G|` is `gmax_window` and whose smallest `|G|²` is `g2min`.
    /// `who` names the kernel in error messages (the serial kernels' names).
    pub(crate) fn new(
        cell: &Cell,
        prep: &PreparedBasis,
        thresh: f64,
        gmax_window: f64,
        g2min: f64,
        moduli: Option<[usize; 3]>,
        who: &'static str,
    ) -> Result<Self, FerricError> {
        let shells = build_shells(cell, prep)?;
        let nr = moduli.map_or(1, |m| m[0] * m[1] * m[2]);
        let lmax = shells.iter().map(|s| s.l).max().unwrap_or(0);
        let mut plan = Self {
            groups: build_groups(&shells),
            images: Vec::new(),
            bucket: Vec::new(),
            nr,
            nbf: prep.nbasis(),
            lmax,
            lists: Vec::new(),
            slot_p: Vec::new(),
            slot_g2max: Vec::new(),
            nsurv_pair: Vec::new(),
            g2max_pair: Vec::new(),
            tasks: Vec::new(),
            comps: (0..=lmax).map(cart_components).collect(),
            c2s: (0..=lmax).map(ferric_cart2sph).collect(),
            shells,
            who,
        };
        plan.check_contiguous()?;
        if plan.shells.is_empty() {
            return Ok(plan);
        }
        let amin = plan
            .shells
            .iter()
            .flat_map(|s| s.exps.iter().copied())
            .fold(f64::INFINITY, f64::min);
        if !(amin > 0.0) {
            return Err(FerricError::Basis(format!(
                "{who}: smallest exponent is {amin}; must be > 0"
            )));
        }
        let rpair = (2.0 * (1e3 / thresh).ln() / amin).sqrt() + 2.0;
        plan.images = cell.translations(rpair)?;
        plan.check_index_widths()?;
        plan.bucket = match moduli {
            None => vec![0; plan.images.len()],
            Some(m) => {
                let b = cell.reciprocal();
                plan.images
                    .iter()
                    .map(|l| residue_index(lattice_coords(&b, l), m))
                    .collect()
            }
        };
        plan.fill_lists(thresh, gmax_window, g2min)?;
        plan.tasks = plan.build_tasks();
        Ok(plan)
    }

    /// Image, primitive and member indices must fit the site/entry fields.
    fn check_index_widths(&self) -> Result<(), FerricError> {
        let wide_group = self
            .groups
            .iter()
            .any(|g| g.len() * g.len() > usize::from(u16::MAX));
        if u32::try_from(self.images.len()).is_err()
            || wide_group
            || self
                .shells
                .iter()
                .any(|s| s.exps.len() > usize::from(u16::MAX))
        {
            return Err(FerricError::General(format!(
                "{}: {} images, a contraction longer than {} primitives or a shell group too large",
                self.who,
                self.images.len(),
                u16::MAX
            )));
        }
        Ok(())
    }

    /// Count, check against the budget, then fill the site lists (so they
    /// are never allocated past the budget), then the exp-table slots and
    /// the per-shell-pair tallies. Both walks are parallel over GROUP PAIRS
    /// and collected in pair order: the lists do not depend on the thread
    /// count.
    fn fill_lists(&mut self, thresh: f64, gmax: f64, g2min: f64) -> Result<(), FerricError> {
        let who = self.who;
        let ngr = self.groups.len();
        let (eids, uexp) = exponent_ids(&self.shells);
        let nu = uexp.len();
        if nu > usize::from(u16::MAX) {
            return Err(FerricError::General(format!(
                "{who}: {nu} distinct exponents (the exp-table index holds {})",
                u16::MAX
            )));
        }
        let counts: Vec<ListCount> = (0..ngr * ngr)
            .into_par_iter()
            .map(|gp| {
                let mut c = ListCount::default();
                let mut last = None;
                self.walk(gp / ngr, gp % ngr, thresh, gmax, g2min, |key, _| {
                    c.entries += 1;
                    if last != Some(key) {
                        c.sites += 1;
                        last = Some(key);
                    }
                });
                c
            })
            .collect();
        if counts.iter().any(|c| c.entries >= u64::from(u32::MAX)) {
            return Err(FerricError::General(format!(
                "{who}: a group pair has more than {} survivors",
                u32::MAX
            )));
        }
        let (ns_tot, ne_tot) = counts
            .iter()
            .fold((0u64, 0u64), |(s, e), c| (s + c.sites, e + c.entries));
        Ledger::new(crate::budget::resolve(None)).check(
            &format!("{who} cached primitive-pair survivors ({ne_tot}) and sites ({ns_tot})"),
            bytes_of(ne_tot, SURVIVOR_BYTES).saturating_add(bytes_of(ns_tot, SITE_BYTES)),
        )?;
        self.lists = (0..ngr * ngr)
            .into_par_iter()
            .map(|gp| {
                let (ga, gb) = (gp / ngr, gp % ngr);
                let (ea, eb) = (&eids[self.groups[ga].s0], &eids[self.groups[gb].s0]);
                let mut v = GroupPairList {
                    sites: Vec::with_capacity(counts[gp].sites as usize),
                    entries: Vec::with_capacity(counts[gp].entries as usize),
                };
                self.walk(ga, gb, thresh, gmax, g2min, |key, e| {
                    let raw =
                        ea[usize::from(key.1)] as usize * nu + eb[usize::from(key.2)] as usize;
                    v.push(key, raw as u32, e);
                });
                v
            })
            .collect();
        self.assign_slots(&uexp);
        self.tally_pairs();
        Ok(())
    }

    /// Relabel the raw exponent-pair index of every site (`id_a × nu +
    /// id_b`) to a dense slot, and record each slot's `p = a + b` and
    /// largest entry window (serial, in list order: deterministic).
    fn assign_slots(&mut self, uexp: &[f64]) {
        let nu = uexp.len();
        let mut map = vec![u32::MAX; nu * nu];
        let (mut slot_p, mut slot_g2max) = (Vec::new(), Vec::new());
        for list in &mut self.lists {
            for s in &mut list.sites {
                let raw = s.slot as usize;
                if map[raw] == u32::MAX {
                    map[raw] = slot_p.len() as u32;
                    slot_p.push(uexp[raw / nu] + uexp[raw % nu]);
                    slot_g2max.push(f64::NEG_INFINITY);
                }
                s.slot = map[raw];
                let w = &mut slot_g2max[s.slot as usize];
                for e in &list.entries[s.e0 as usize..s.e1 as usize] {
                    *w = f64::max(*w, e.g2max);
                }
            }
        }
        self.slot_p = slot_p;
        self.slot_g2max = slot_g2max;
    }

    /// Entries and largest window of every SHELL pair.
    fn tally_pairs(&mut self) {
        let ns = self.shells.len();
        let ngr = self.groups.len();
        self.nsurv_pair = vec![0; ns * ns];
        self.g2max_pair = vec![f64::NEG_INFINITY; ns * ns];
        for (gp, list) in self.lists.iter().enumerate() {
            let (ga, gb) = (self.groups[gp / ngr], self.groups[gp % ngr]);
            for e in &list.entries {
                let m = usize::from(e.member);
                let (i, j) = (ga.s0 + m / gb.len(), gb.s0 + m % gb.len());
                self.nsurv_pair[i * ns + j] += 1;
                let w = &mut self.g2max_pair[i * ns + j];
                *w = f64::max(*w, e.g2max);
            }
        }
    }

    /// Work weight of shell pair `(ia, ib)` for the task split (not a
    /// numerical input): `(survivors + 1) × ncart(ia) × ncart(ib)`.
    fn pair_weight(&self, ia: usize, ib: usize) -> u64 {
        let ns = self.shells.len();
        let n = self.nsurv_pair[ia * ns + ib] + 1;
        n * (self.shells[ia].ncart * self.shells[ib].ncart) as u64
    }

    /// Work weight of group pair `(ga, gb)`: the sum over its members.
    fn group_weight(&self, ga: usize, gb: usize) -> u64 {
        let (a, b) = (self.groups[ga], self.groups[gb]);
        (a.s0..a.s1)
            .flat_map(|i| (b.s0..b.s1).map(move |j| (i, j)))
            .map(|(i, j)| self.pair_weight(i, j))
            .sum()
    }

    /// The deterministic greedy split ([`PAIR_FT_TASK_SPLIT`]): per bra
    /// group, ket groups in order, a range is closed before the group pair
    /// that would push it past the cap. Every group pair lands in exactly
    /// one task; each bra group has at least one.
    fn build_tasks(&self) -> Vec<PairTaskSpec> {
        let ngr = self.groups.len();
        let (mut total, mut heaviest) = (0u64, 0u64);
        for ga in 0..ngr {
            for gb in 0..ngr {
                let w = self.group_weight(ga, gb);
                total += w;
                heaviest = heaviest.max(w);
            }
        }
        let cap = total.div_ceil(PAIR_FT_TASK_SPLIT).max(heaviest);
        let col = |gb: usize| {
            if gb < ngr {
                self.shells[self.groups[gb].s0].off
            } else {
                self.nbf
            }
        };
        let mut tasks = Vec::new();
        for ga in 0..ngr {
            let g = self.groups[ga];
            let r0 = self.shells[g.s0].off;
            let nrows: usize = (g.s0..g.s1).map(|s| self.shells[s].nfun).sum();
            let task = |gb0: usize, gb1: usize| PairTaskSpec {
                ga,
                gb0,
                gb1,
                c0: col(gb0),
                c1: col(gb1),
                r0,
                nrows,
            };
            let (mut gb0, mut acc) = (0usize, 0u64);
            for gb in 0..ngr {
                let w = self.group_weight(ga, gb);
                if gb > gb0 && acc + w > cap {
                    tasks.push(task(gb0, gb));
                    (gb0, acc) = (gb, 0);
                }
                acc += w;
            }
            tasks.push(task(gb0, ngr));
        }
        tasks
    }

    /// Load-balance and sharing counters of this plan (max over the plans
    /// of a call sequence, e.g. the main and smooth LR passes): tasks per
    /// chunk, and the work proxy `survivors × ncart(bra) × ncart(ket)`
    /// summed, of the largest task, of the largest single shell pair and of
    /// the largest bra-group row; survivors and sites (summed); groups.
    /// `max task / (total / tasks)` is the imbalance factor.
    pub(crate) fn record_stats(&self, t: &mut PbcTimings) {
        let ns = self.shells.len();
        let cost = |ia: usize, ib: usize| {
            self.nsurv_pair[ia * ns + ib] * (self.shells[ia].ncart * self.shells[ib].ncart) as u64
        };
        let (mut total, mut max_pair, mut max_bra) = (0u64, 0u64, 0u64);
        let task_cost = |k: &PairTaskSpec| {
            let (a, b0, b1) = (
                self.groups[k.ga],
                self.groups.get(k.gb0).map_or(ns, |g| g.s0),
                self.groups.get(k.gb1).map_or(ns, |g| g.s0),
            );
            (a.s0..a.s1)
                .map(|i| (b0..b1).map(|j| cost(i, j)).sum::<u64>())
                .sum::<u64>()
        };
        for g in &self.groups {
            let mut bra = 0u64;
            for ia in g.s0..g.s1 {
                for ib in 0..ns {
                    let c = cost(ia, ib);
                    bra += c;
                    max_pair = max_pair.max(c);
                }
            }
            total += bra;
            max_bra = max_bra.max(bra);
        }
        let max_task = self.tasks.iter().map(task_cost).max().unwrap_or(0);
        let nsites: u64 = self.lists.iter().map(|l| l.sites.len() as u64).sum();
        t.add_counter(CTR_SURVIVORS, self.nsurv_pair.iter().sum());
        t.add_counter(CTR_SITES, nsites);
        t.max_counter("pair FT shells", ns as u64);
        t.max_counter("pair FT shell groups", self.groups.len() as u64);
        t.max_counter(
            CTR_LARGEST_GROUP,
            self.groups
                .iter()
                .map(|g| g.len() as u64)
                .max()
                .unwrap_or(0),
        );
        t.max_counter("pair FT exp-table slots", self.slot_p.len() as u64);
        t.max_counter("pair FT tasks per chunk", self.tasks.len() as u64);
        t.max_counter("pair FT cost total", total);
        t.max_counter("pair FT cost max task", max_task);
        t.max_counter("pair FT cost max pair", max_pair);
        t.max_counter("pair FT cost max bra group", max_bra);
    }

    /// Shells must tile `0..nbf` in order (the tasks' row segments tile
    /// every `P[r]`).
    fn check_contiguous(&self) -> Result<(), FerricError> {
        let mut next = 0usize;
        for (s, sh) in self.shells.iter().enumerate() {
            if sh.off != next {
                return Err(FerricError::General(format!(
                    "{}: shell {s} starts at AO {} but the previous shells end at {next}",
                    self.who, sh.off
                )));
            }
            next += sh.nfun;
        }
        if next != self.nbf {
            return Err(FerricError::General(format!(
                "{}: shells cover {next} AOs of {}",
                self.who, self.nbf
            )));
        }
        Ok(())
    }

    /// The serial kernels' screen for every member shell pair of group pair
    /// `(ga, gb)`, in walk order (image, bra primitive, ket primitive; then
    /// members), with their exact expressions: `f(site key, entry)` per
    /// survivor. `(π/p)^{3/2}` and `e^{−ab r²/p}` are the serial kernel's
    /// subexpressions, evaluated once per site instead of once per member
    /// (same inputs, same bits).
    fn walk(
        &self,
        ga: usize,
        gb: usize,
        thresh: f64,
        gmax: f64,
        g2min: f64,
        mut f: impl FnMut((u32, u16, u16), Entry),
    ) {
        let (gra, grb) = (self.groups[ga], self.groups[gb]);
        let (sa, sb) = (&self.shells[gra.s0], &self.shells[grb.s0]);
        let (la, lb) = (sa.l, sb.l);
        let lgmax = (la + lb) as f64 * gmax.max(1.0).ln();
        let pi = std::f64::consts::PI;
        for (il, l) in self.images.iter().enumerate() {
            let bc = [
                sb.center[0] + l[0],
                sb.center[1] + l[1],
                sb.center[2] + l[2],
            ];
            let ab = [
                sa.center[0] - bc[0],
                sa.center[1] - bc[1],
                sa.center[2] - bc[2],
            ];
            let r2 = ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2];
            for (ka, &a) in sa.exps.iter().enumerate() {
                for (kb, &b) in sb.exps.iter().enumerate() {
                    let p = a + b;
                    let pw = (pi / p).powf(1.5);
                    let gauss = (-a * b / p * r2).exp();
                    let key = (il as u32, ka as u16, kb as u16);
                    for (mi, i) in (gra.s0..gra.s1).enumerate() {
                        let ca = self.shells[i].coefs[ka];
                        for (mj, j) in (grb.s0..grb.s1).enumerate() {
                            let cb = self.shells[j].coefs[kb];
                            let cc = ca * cb * pw;
                            let mag = (cc * gauss).abs();
                            if mag < thresh {
                                continue;
                            }
                            let g2max =
                                4.0 * p * ((mag / thresh).ln().max(0.0) + WINDOW_MARGIN + lgmax);
                            // The serial kernel's `ngp == 0` in every chunk.
                            if g2max < g2min {
                                continue;
                            }
                            let member = (mi * grb.len() + mj) as u16;
                            f(key, Entry { cc, g2max, member });
                        }
                    }
                }
            }
        }
    }

    /// `P[r]` (each `(nbf, nbf, gvecs.len())`) for one G chunk in fresh
    /// buffers; bit for bit the serial kernel's output (module doc).
    pub(crate) fn block(&self, gvecs: &[[f64; 3]]) -> Vec<Array3<Complex64>> {
        self.block_timed(gvecs, &mut Vec::new(), &mut PbcTimings::default())
    }

    /// [`PairFtPlan::block`] into buffers taken from `pool` (return them with
    /// [`recycle`] after the sink), with sub-stage timers and task counters
    /// in `t`. A reused buffer is NOT zeroed: every element is overwritten
    /// (module doc), so the values are the fresh-buffer ones bit for bit.
    pub(crate) fn block_timed(
        &self,
        gvecs: &[[f64; 3]],
        pool: &mut Vec<Vec<Complex64>>,
        t: &mut PbcTimings,
    ) -> Vec<Array3<Complex64>> {
        let clock = StageClock::start();
        let (nbf, ng, nr) = (self.nbf, gvecs.len(), self.nr);
        let mut out: Vec<Array3<Complex64>> = (0..nr)
            .map(|_| {
                let v = take_buffer(pool, nbf * nbf * ng);
                Array3::from_shape_vec((nbf, nbf, ng), v).expect("buffer of the block's length")
            })
            .collect();
        if ng == 0 || self.shells.is_empty() {
            // Zero elements (nbf = 0 without shells): nothing stale.
            t.stop_sub(SUB_SETUP, &clock);
            return out;
        }
        let chunk = ChunkG::new(gvecs, self.lmax, &self.slot_p, &self.slot_g2max);
        let mut segs = self.segments(&mut out, ng);
        // Task `k` owns `segs[seg(k)..seg(k) + nr * nrows(k)]`: the tasks
        // take consecutive runs of the task-major segment list.
        let mut rest: &mut [&mut [Complex64]] = &mut segs;
        let views: Vec<(&PairTaskSpec, &mut [&mut [Complex64]])> = self
            .tasks
            .iter()
            .map(|k| {
                let (mine, tail) = std::mem::take(&mut rest).split_at_mut(self.nr * k.nrows);
                rest = tail;
                (k, mine)
            })
            .collect();
        assert!(rest.is_empty(), "{}: task segments left over", self.who);
        let n_tasks = views.len() as u64;
        t.stop_sub(SUB_SETUP, &clock);

        let clock = StageClock::start();
        let (busy, longest, pairs) = (AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0));
        let (partial, empty) = (AtomicU64::new(0), AtomicU64::new(0));
        views
            .into_par_iter()
            .for_each_init(PairScratch::default, |scr, (k, dst)| {
                let t0 = Instant::now();
                let w = self.task_block(k, dst, scr, &chunk);
                let ns = u64::try_from(t0.elapsed().as_nanos()).unwrap_or(u64::MAX);
                busy.fetch_add(ns, Ordering::Relaxed);
                longest.fetch_max(ns, Ordering::Relaxed);
                pairs.fetch_add(w.pairs, Ordering::Relaxed);
                partial.fetch_add(w.partial, Ordering::Relaxed);
                empty.fetch_add(w.empty, Ordering::Relaxed);
            });
        t.stop_sub(SUB_KERNEL, &clock);
        t.add_counter("pair FT tasks run", n_tasks);
        t.add_counter("pair FT shell pairs visited", pairs.into_inner());
        t.add_counter(CTR_PARTIAL_WINDOW, partial.into_inner());
        t.add_counter(CTR_EMPTY_WINDOW, empty.into_inner());
        t.add_counter(
            "pair FT kernel busy us (sum of task walls)",
            busy.into_inner() / 1000,
        );
        t.max_counter("pair FT longest task us", longest.into_inner() / 1000);
        out
    }

    /// Every task's row segments of every `P[r]`, TASK-major: task `k`'s
    /// run is `r * nrows(k) + i` -> row `r0(k) + i` of `P[r]`, columns
    /// `c0(k)..c1(k)`, all `ng` G. Three allocations per chunk (rows,
    /// segments, and the caller's task views), none per task. Asserts that
    /// the tasks consumed every row whole (they tile each `P[r]`).
    fn segments<'a>(
        &self,
        out: &'a mut [Array3<Complex64>],
        ng: usize,
    ) -> Vec<&'a mut [Complex64]> {
        let nbf = self.nbf;
        let mut rows: Vec<&'a mut [Complex64]> = Vec::with_capacity(self.nr * nbf);
        for arr in out.iter_mut() {
            let s = arr.as_slice_mut().expect("standard-layout block");
            rows.extend(s.chunks_mut(nbf * ng));
        }
        let nseg: usize = self.tasks.iter().map(|k| self.nr * k.nrows).sum();
        let mut segs = Vec::with_capacity(nseg);
        for k in &self.tasks {
            let width = (k.c1 - k.c0) * ng;
            for r in 0..self.nr {
                for i in 0..k.nrows {
                    let row = &mut rows[r * nbf + k.r0 + i];
                    let (head, tail) = std::mem::take(row).split_at_mut(width);
                    *row = tail;
                    segs.push(head);
                }
            }
        }
        assert!(
            rows.iter().all(|r| r.is_empty()),
            "{}: the tasks do not tile the output rows",
            self.who
        );
        segs
    }

    /// Task `k`: its ket groups in order, each group pair into its rows and
    /// columns of the task's segments. Returns the task's tallies (counted
    /// locally, added to the shared counters once per task).
    fn task_block(
        &self,
        k: &PairTaskSpec,
        dst: &mut [&mut [Complex64]],
        scr: &mut PairScratch,
        ch: &ChunkG,
    ) -> WindowTally {
        let mut w = WindowTally::default();
        for gb in k.gb0..k.gb1 {
            self.group_pair_block(k, gb, dst, scr, ch, &mut w);
        }
        w
    }

    /// Group pair `(k.ga, gb)`: every member's window in this chunk, the
    /// sites' terms into the members' Cartesian accumulators, then each
    /// member's Cartesian→AO scatter into its rows (`dst[r * nrows(k) +
    /// off(i) − r0(k) + ·]`) and columns (from local column `off(j) −
    /// c0(k)`).
    fn group_pair_block(
        &self,
        k: &PairTaskSpec,
        gb: usize,
        dst: &mut [&mut [Complex64]],
        scr: &mut PairScratch,
        ch: &ChunkG,
        w: &mut WindowTally,
    ) {
        let (gra, grb) = (self.groups[k.ga], self.groups[gb]);
        let ns = self.shells.len();
        let nr = self.nr;
        scr.n_used.clear();
        for i in gra.s0..gra.s1 {
            for j in grb.s0..grb.s1 {
                // No entry reaches g >= n_used (module doc).
                let n = ch.g2.partition_point(|&x| x <= self.g2max_pair[i * ns + j]);
                w.count(n, ch.ng);
                scr.n_used.push(n);
            }
        }
        if scr.n_used.iter().any(|&n| n > 0) {
            let (sa, sb) = (&self.shells[gra.s0], &self.shells[grb.s0]);
            scr.prepare(sa, sb, nr, ch.ng);
            let list = &self.lists[k.ga * self.groups.len() + gb];
            for site in &list.sites {
                let entries = &list.entries[site.e0 as usize..site.e1 as usize];
                self.site_terms(site, entries, sa, sb, scr, ch);
            }
        }
        for (m, (i, j)) in (gra.s0..gra.s1)
            .flat_map(|i| (grb.s0..grb.s1).map(move |j| (i, j)))
            .enumerate()
        {
            let (si, sj) = (&self.shells[i], &self.shells[j]);
            let (col0, roff) = (sj.off - k.c0, si.off - k.r0);
            let n = scr.n_used[m];
            for r in 0..nr {
                let rows = &mut dst[r * k.nrows + roff..r * k.nrows + roff + si.nfun];
                let cart: &[Complex64] = if n == 0 { &[] } else { &scr.cart[m * nr + r] };
                self.scatter(rows, col0, cart, n, &mut scr.tmp, si, sj, &ch.order);
            }
        }
    }

    /// One site: its entries' windows; the shared `G·P` trig row, E tables
    /// and F rows up to the largest window; then each entry's term into its
    /// member's accumulator (bucket of the site's image) — the serial
    /// kernel's inner loop body, expression for expression.
    fn site_terms(
        &self,
        site: &Site,
        entries: &[Entry],
        sa: &FtShell,
        sb: &FtShell,
        scr: &mut PairScratch,
        ch: &ChunkG,
    ) {
        let PairScratch {
            common,
            cs,
            sn,
            ngp,
            ebuf,
            fbuf,
            cart,
            ..
        } = scr;
        ngp.clear();
        ngp.extend(
            entries
                .iter()
                .map(|e| ch.g2.partition_point(|&x| x <= e.g2max)),
        );
        let nmax = ngp.iter().copied().max().unwrap_or(0);
        if nmax == 0 {
            return;
        }
        let l = &self.images[site.image as usize];
        let bucket = self.bucket[site.image as usize];
        let bc = [
            sb.center[0] + l[0],
            sb.center[1] + l[1],
            sb.center[2] + l[2],
        ];
        let ab = [
            sa.center[0] - bc[0],
            sa.center[1] - bc[1],
            sa.center[2] - bc[2],
        ];
        let a = sa.exps[usize::from(site.ia)];
        let b = sb.exps[usize::from(site.ib)];
        let p = a + b;
        let pc = [
            (a * sa.center[0] + b * bc[0]) / p,
            (a * sa.center[1] + b * bc[1]) / p,
            (a * sa.center[2] + b * bc[2]) / p,
        ];
        for ((c, s), gv) in cs[..nmax]
            .iter_mut()
            .zip(&mut sn[..nmax])
            .zip(&ch.gsorted[..nmax])
        {
            let ph = gv[0] * pc[0] + gv[1] * pc[1] + gv[2] * pc[2];
            *c = ph.cos();
            *s = ph.sin();
        }
        f_rows(sa.l, sb.l, a, b, ab, nmax, ch, ebuf, fbuf);
        let et = &ch.etab[site.slot as usize * ch.ng..site.slot as usize * ch.ng + nmax];
        let (ca, cb) = (&self.comps[sa.l], &self.comps[sb.l]);
        for (e, &n) in entries.iter().zip(ngp.iter()) {
            if n == 0 {
                continue;
            }
            let cc = e.cc;
            for (((o, &c), &s), &x) in common[..n].iter_mut().zip(&cs[..n]).zip(&sn[..n]).zip(et) {
                let mag = cc * x;
                // e^{−iG·P}
                *o = Complex64::new(mag * c, -mag * s);
            }
            let acc = &mut cart[usize::from(e.member) * self.nr + bucket];
            accumulate(acc, &common[..n], fbuf, ca, cb, sb.l, ch.ng);
        }
    }

    /// Cartesian `[nca][ncb][ng]` -> the pair's AO elements (`rows[i]`,
    /// local column `col0 + j`), G scattered back through `order`: the
    /// serial kernels' transform, expression for expression, for
    /// `g < n_used`; `+0.0` (its value there, module doc) for the rest.
    /// `cart` is read on `0..n_used` of each row only; `tmp` is scratch for
    /// the pure half-transform (restarted from +0.0 on that range).
    #[allow(clippy::too_many_arguments)]
    fn scatter(
        &self,
        rows: &mut [&mut [Complex64]],
        col0: usize,
        cart: &[Complex64],
        n_used: usize,
        tmp: &mut Vec<Complex64>,
        sa: &FtShell,
        sb: &FtShell,
        order: &[usize],
    ) {
        let zero = Complex64::new(0.0, 0.0);
        let ng = order.len();
        let (la, lb) = (sa.l, sb.l);
        let (nca, ncb) = (sa.ncart, sb.ncart);
        let (nfa, nfb) = (sa.nfun, sb.nfun);
        let right: &[Complex64] = if n_used > 0 && sb.pure {
            let cbm = &self.c2s[lb];
            tmp.resize(nca * nfb * ng, zero);
            for row in tmp.chunks_mut(ng) {
                row[..n_used].fill(zero);
            }
            for m in 0..nca {
                for j in 0..nfb {
                    for n in 0..ncb {
                        let c = cbm[n * nfb + j];
                        if c == 0.0 {
                            continue;
                        }
                        for g in 0..n_used {
                            tmp[(m * nfb + j) * ng + g] += cart[(m * ncb + n) * ng + g] * c;
                        }
                    }
                }
            }
            &tmp[..]
        } else {
            cart
        };
        for (i, dst) in rows.iter_mut().enumerate().take(nfa) {
            for j in 0..nfb {
                let base = (col0 + j) * ng;
                for g in 0..n_used {
                    let val = if sa.pure {
                        let cam = &self.c2s[la];
                        let mut acc = zero;
                        for m in 0..nca {
                            let c = cam[m * nfa + i];
                            if c != 0.0 {
                                acc += right[(m * nfb + j) * ng + g] * c;
                            }
                        }
                        acc
                    } else {
                        right[(i * nfb + j) * ng + g]
                    };
                    dst[base + order[g]] = val;
                }
                for &o in &order[n_used..] {
                    dst[base + o] = zero;
                }
            }
        }
    }
}

/// The serial kernel's F rows `F_d[ij][g] = Σ_t E_d[ij][t] (−iG_d)^t` for
/// `g < n`, per Cartesian direction (E tables rebuilt in `ebuf`).
#[allow(clippy::too_many_arguments)]
fn f_rows(
    la: usize,
    lb: usize,
    a: f64,
    b: f64,
    ab: [f64; 3],
    n: usize,
    ch: &ChunkG,
    ebuf: &mut [Vec<f64>; 3],
    fbuf: &mut [Vec<Complex64>; 3],
) {
    let zero = Complex64::new(0.0, 0.0);
    let (ng, st, nij) = (ch.ng, la + lb + 1, (la + 1) * (lb + 1));
    for d in 0..3 {
        e_table(la, lb, a, b, ab[d], &mut ebuf[d]);
        let (e, f, w) = (&ebuf[d], &mut fbuf[d], &ch.pw[d]);
        for ij in 0..nij {
            let frow = &mut f[ij * ng..ij * ng + n];
            frow.fill(zero);
            for t in 0..st {
                let et = e[ij * st + t];
                if et == 0.0 {
                    continue;
                }
                for (x, wv) in frow.iter_mut().zip(&w[t * ng..t * ng + n]) {
                    *x += *wv * et;
                }
            }
        }
    }
}

/// `cart[(u, v)][g] += common[g] · Fx · Fy · Fz` for `g < common.len()`
/// (the serial kernel's accumulate, same product order).
fn accumulate(
    cart: &mut [Complex64],
    common: &[Complex64],
    fbuf: &[Vec<Complex64>; 3],
    ca: &[[u8; 3]],
    cb: &[[u8; 3]],
    lb: usize,
    ng: usize,
) {
    let n = common.len();
    let ncb = cb.len();
    let (fx, fy, fz) = (&fbuf[0], &fbuf[1], &fbuf[2]);
    for (u, ac) in ca.iter().enumerate() {
        for (v, bcmp) in cb.iter().enumerate() {
            let ix = (ac[0] as usize * (lb + 1) + bcmp[0] as usize) * ng;
            let iy = (ac[1] as usize * (lb + 1) + bcmp[1] as usize) * ng;
            let iz = (ac[2] as usize * (lb + 1) + bcmp[2] as usize) * ng;
            let dst = &mut cart[(u * ncb + v) * ng..(u * ncb + v) * ng + n];
            for ((((d, c), x), y), z) in dst
                .iter_mut()
                .zip(common)
                .zip(&fx[ix..ix + n])
                .zip(&fy[iy..iy + n])
                .zip(&fz[iz..iz + n])
            {
                *d += *c * *x * *y * *z;
            }
        }
    }
}

/// One task of every chunk: bra group `ga` (AO rows `r0..r0 + nrows`), ket
/// groups `gb0..gb1`, whose AO columns are `c0..c1`
/// ([`PairFtPlan::build_tasks`]).
struct PairTaskSpec {
    ga: usize,
    gb0: usize,
    gb1: usize,
    c0: usize,
    c1: usize,
    r0: usize,
    nrows: usize,
}

/// Per-task tallies of the (shell pair, G chunk) windows: `partial` counts
/// `0 < n_used < n_G(chunk)` (accumulated on `0..n_used`, `+0.0` written
/// beyond), `empty` counts `n_used == 0` (the all-`+0.0` scatter). Pairs
/// with `n_used == n_G(chunk)` are not counted; `pairs` counts all.
#[derive(Default)]
struct WindowTally {
    pairs: u64,
    partial: u64,
    empty: u64,
}

impl WindowTally {
    fn count(&mut self, n_used: usize, ng: usize) {
        self.pairs += 1;
        if n_used == 0 {
            self.empty += 1;
        } else if n_used < ng {
            self.partial += 1;
        }
    }
}

/// Per-worker scratch of [`PairFtPlan::group_pair_block`], reused across
/// group pairs and tasks (never read stale: `cs`/`sn`, `common` and the F
/// rows are written for `g <` the window before being read there,
/// `e_table` zeroes its range, the Cartesian accumulators restart from
/// +0.0 on `0..n_used` (as the serial kernel's fresh `vec![zero; ..]`) and
/// nothing reads them beyond it).
#[derive(Default)]
struct PairScratch {
    common: Vec<Complex64>,
    cs: Vec<f64>,
    sn: Vec<f64>,
    ngp: Vec<usize>,
    n_used: Vec<usize>,
    ebuf: [Vec<f64>; 3],
    fbuf: [Vec<Complex64>; 3],
    cart: Vec<Vec<Complex64>>,
    tmp: Vec<Complex64>,
}

impl PairScratch {
    /// Size the buffers for a group pair of `sa`-like and `sb`-like shells
    /// with `n_used.len()` members, and zero each member's accumulators on
    /// its `0..n_used`.
    fn prepare(&mut self, sa: &FtShell, sb: &FtShell, nr: usize, ng: usize) {
        let zero = Complex64::new(0.0, 0.0);
        let (st, nij) = (sa.l + sb.l + 1, (sa.l + 1) * (sb.l + 1));
        self.common.resize(ng, zero);
        self.cs.resize(ng, 0.0);
        self.sn.resize(ng, 0.0);
        for d in 0..3 {
            self.ebuf[d].resize(nij * st, 0.0);
            self.fbuf[d].resize(nij * ng, zero);
        }
        self.cart.resize_with(self.n_used.len() * nr, Vec::new);
        for (m, &n) in self.n_used.iter().enumerate() {
            for c in &mut self.cart[m * nr..(m + 1) * nr] {
                c.resize(sa.ncart * sb.ncart * ng, zero);
                for row in c.chunks_mut(ng) {
                    row[..n].fill(zero);
                }
            }
        }
    }
}

/// A buffer of exactly `n` elements from `pool` (or new). Existing elements
/// keep their (stale) values; growth is zero-filled in parallel, so a first
/// chunk faults its pages in on every worker instead of one.
fn take_buffer(pool: &mut Vec<Vec<Complex64>>, n: usize) -> Vec<Complex64> {
    let mut v = pool.pop().unwrap_or_default();
    if v.len() >= n {
        v.truncate(n);
    } else {
        let grow = n - v.len();
        v.reserve_exact(grow);
        v.par_extend(rayon::iter::repeat_n(Complex64::new(0.0, 0.0), grow));
    }
    v
}

/// Return a chunk's blocks to `pool` for the next [`PairFtPlan::block_timed`].
pub(crate) fn recycle(pool: &mut Vec<Vec<Complex64>>, blocks: Vec<Array3<Complex64>>) {
    for b in blocks {
        let (v, offset) = b.into_raw_vec_and_offset();
        if offset.unwrap_or(0) == 0 {
            pool.push(v);
        }
    }
}

/// One G chunk in `|G|`-ascending order (stable over the caller's order,
/// exactly as the serial kernels sort it) with its `|G|²`, `(−iG_d)^t` and
/// the exp table.
struct ChunkG {
    ng: usize,
    order: Vec<usize>,
    gsorted: Vec<[f64; 3]>,
    g2: Vec<f64>,
    pw: Vec<Vec<Complex64>>,
    /// `etab[slot * ng + g] = (−|G_g|² / (4 p_slot)).exp()` for `g` inside
    /// the slot's largest window (never read beyond it).
    etab: Vec<f64>,
}

impl ChunkG {
    fn new(gvecs: &[[f64; 3]], lmax: usize, slot_p: &[f64], slot_g2max: &[f64]) -> Self {
        let ng = gvecs.len();
        let mut order: Vec<usize> = (0..ng).collect();
        order.sort_by(|&x, &y| norm2(&gvecs[x]).total_cmp(&norm2(&gvecs[y])));
        let gsorted: Vec<[f64; 3]> = order.iter().map(|&i| gvecs[i]).collect();
        let nt = 2 * lmax + 1;
        let g2: Vec<f64> = gsorted.iter().map(norm2).collect();
        // pw[d][t * ng + g] = (−i G_d)^t.
        let pw: Vec<Vec<Complex64>> = (0..3)
            .map(|d| {
                let mut v = vec![Complex64::new(0.0, 0.0); nt * ng];
                for (g, gv) in gsorted.iter().enumerate() {
                    let step = Complex64::new(0.0, -gv[d]);
                    let mut acc = Complex64::new(1.0, 0.0);
                    for t in 0..nt {
                        v[t * ng + g] = acc;
                        acc *= step;
                    }
                }
                v
            })
            .collect();
        let mut etab = vec![0.0; slot_p.len() * ng];
        if ng > 0 {
            etab.par_chunks_mut(ng)
                .zip(slot_p.par_iter().zip(slot_g2max))
                .for_each(|(row, (&p, &w))| {
                    let n = g2.partition_point(|&x| x <= w);
                    for (e, &x) in row[..n].iter_mut().zip(&g2) {
                        *e = (-x / (4.0 * p)).exp();
                    }
                });
        }
        Self {
            ng,
            order,
            gsorted,
            g2,
            pw,
            etab,
        }
    }
}

/// Smallest `|G|²` of a G set (the serial kernels' expression); `+∞` for an
/// empty set.
pub(crate) fn min_gnorm2(gvecs: &[[f64; 3]]) -> f64 {
    gvecs.iter().map(norm2).fold(f64::INFINITY, f64::min)
}
