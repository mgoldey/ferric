//! The production pair-FT kernel: the primitive-pair screen walked ONCE per
//! call, then every G chunk evaluated shell-pair-parallel from the cached
//! survivor lists (FINDINGS "Performance plan (research) — 2026-09-25" §5,
//! "Pair-FT re-walk").
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
//! # Bit-identity (per element, by construction)
//!
//! An output element `P[r][m, n, g]` is the sum, in serial walk order
//! (image ascending, then bra primitive, then ket primitive), of the terms of
//! the primitive pairs of `(shell(m), shell(n))` that pass
//! `mag >= thresh` AND whose window reaches `g` (`|G_g|² <= g2max`), followed
//! by the same Cartesian→AO transform. Nothing couples different G columns
//! or different shell pairs. So:
//!
//! * The survivor list of a shell pair holds exactly the combinations that
//!   pass the screen, in walk order, with the screen's own `cc` and `g2max`
//!   (computed with the serial kernel's expressions); every other scalar the
//!   term needs (`B + L`, `A − B′`, `p`, `P_c`) is recomputed per chunk with
//!   the serial kernel's expressions. A survivor whose window reaches no G of
//!   the whole set (`g2max < min |G|²`) contributed zero to every chunk in the
//!   serial kernel (`ngp == 0`) and is dropped.
//! * The parallel split is over TASKS `(bra shell, contiguous ket-shell
//!   range)`, formed once per plan ([`PairFtPlan::new`], deterministic
//!   greedy on a per-pair work weight, [`PAIR_FT_TASK_SPLIT`]): a task owns
//!   the `nfun(bra)` row segments of its ket range (the columns of those ket
//!   shells) in every `P[r]`, walks its ket shells in order and writes each
//!   of their elements exactly once, each pair from a Cartesian accumulator
//!   it zeroes itself. No element is shared between tasks, and neither the
//!   split nor any element's term order depends on the thread count. (Finer
//!   than bra-shell rows: with few shells a heavy contracted bra row is a
//!   tail at every per-chunk barrier. Coarser than single pairs, whose split
//!   measurably raised the serial kernel time: FINDINGS "Pair FT per-pair
//!   tasks: sub-stage measurement (2026-09-28)". The `pair FT cost max *`
//!   counters of [`PairFtPlan::record_stats`] measure the balance.)
//! * Beyond a pair's window (`g >= n_used`, the largest `ngp` of its
//!   survivors, `n_used = 0` for a pair without survivors) no term reaches
//!   the Cartesian accumulator, which therefore stays exactly `+0.0`, and
//!   the serial transform of `+0.0` is `+0.0` (`+0.0 + (±0.0) = +0.0`, the
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
//! chunking (`tests/pbc_parallel_bitwise.rs`).
//!
//! # Memory
//!
//! The survivor lists ([`SURVIVOR_BYTES`] each) are counted before they are
//! allocated and checked against ferric's unified budget. Each rayon worker
//! holds one shell pair's scratch at a time (`R` Cartesian blocks, the F
//! rows and the pure half-transform, all `× n_G(chunk)`), which is the per-G
//! scratch [`super::pair_ft_bytes_per_g`] already charges once per G; with `T`
//! threads `T` such sets are live (a few KiB per G per thread against the
//! `16 nao²` bytes per G of the output itself).

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
/// Sub-stage: the serial per-chunk setup (G sort, buffers, task slices).
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

/// Task granularity: a task's work weight is capped at
/// `max(ceil(W / PAIR_FT_TASK_SPLIT), heaviest pair)`, `W` the plan's total
/// weight (a pair weighs `(survivors + 1) × ncart(bra) × ncart(ket)`, the
/// `+ 1` standing for its transform and scatter, which run for every pair).
/// About `PAIR_FT_TASK_SPLIT + nshells` tasks per chunk. A constant, NOT a
/// function of the thread count: the split (hence every element's term
/// order, which it does not touch anyway) is the same on any pool.
pub(crate) const PAIR_FT_TASK_SPLIT: u64 = 256;

/// One screened primitive pair of a shell pair: the image and primitive
/// indices, and the two screen scalars (`cc = c_a c_b (π/p)^{3/2}` and the G
/// window `g2max`) exactly as the serial kernel computed them.
#[derive(Clone, Copy)]
struct Survivor {
    image: u32,
    ia: u16,
    ib: u16,
    cc: f64,
    g2max: f64,
}

/// Bytes of one cached survivor.
pub(crate) const SURVIVOR_BYTES: usize = std::mem::size_of::<Survivor>();

/// The screen of one pair-FT call, walked once: shells, images, residue
/// buckets and the per-shell-pair survivor lists.
pub(crate) struct PairFtPlan {
    shells: Vec<FtShell>,
    images: Vec<[f64; 3]>,
    /// Residue bucket of each image (all 0 for the Gamma kernel).
    bucket: Vec<usize>,
    nr: usize,
    nbf: usize,
    lmax: usize,
    /// `survivors[ia * nshells + ib]`, in serial walk order.
    survivors: Vec<Vec<Survivor>>,
    /// Largest survivor window `g2max` of each shell pair (`−∞` without
    /// survivors): `n_used` of a chunk is its `partition_point`.
    g2max_pair: Vec<f64>,
    /// The per-chunk tasks, ordered by bra shell then ket range; the ket
    /// ranges of a bra shell tile `0..nshells` in order.
    tasks: Vec<PairTaskSpec>,
    comps: Vec<Vec<[u8; 3]>>,
    c2s: Vec<Vec<f64>>,
    who: &'static str,
}

fn norm2(g: &[f64; 3]) -> f64 {
    g[0] * g[0] + g[1] * g[1] + g[2] * g[2]
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
            images: Vec::new(),
            bucket: Vec::new(),
            nr,
            nbf: prep.nbasis(),
            lmax,
            survivors: Vec::new(),
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
        if u32::try_from(plan.images.len()).is_err()
            || plan
                .shells
                .iter()
                .any(|s| s.exps.len() > usize::from(u16::MAX))
        {
            return Err(FerricError::General(format!(
                "{who}: {} images or a contraction longer than {} primitives",
                plan.images.len(),
                u16::MAX
            )));
        }
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

        // Count, check, then fill (so the lists are never allocated past the
        // budget). Both passes are parallel over SHELL PAIRS and collected in
        // pair order: the lists do not depend on the thread count.
        let ns = plan.shells.len();
        let counts: Vec<usize> = (0..ns * ns)
            .into_par_iter()
            .map(|pair| {
                let mut c = 0usize;
                plan.walk(pair / ns, pair % ns, thresh, gmax_window, g2min, |_| c += 1);
                c
            })
            .collect();
        let total: usize = counts.iter().sum();
        Ledger::new(crate::budget::resolve(None)).check(
            &format!("{who} cached primitive-pair survivors ({total})"),
            bytes_of(total as u64, SURVIVOR_BYTES),
        )?;
        plan.survivors = (0..ns * ns)
            .into_par_iter()
            .map(|pair| {
                let mut v = Vec::with_capacity(counts[pair]);
                plan.walk(pair / ns, pair % ns, thresh, gmax_window, g2min, |s| {
                    v.push(s)
                });
                v
            })
            .collect();
        plan.g2max_pair = plan
            .survivors
            .iter()
            .map(|v| v.iter().map(|s| s.g2max).fold(f64::NEG_INFINITY, f64::max))
            .collect();
        plan.tasks = plan.build_tasks();
        Ok(plan)
    }

    /// Work weight of shell pair `(ia, ib)` for the task split (not a
    /// numerical input): `(survivors + 1) × ncart(ia) × ncart(ib)`.
    fn pair_weight(&self, ia: usize, ib: usize) -> u64 {
        let ns = self.shells.len();
        let n = self.survivors[ia * ns + ib].len() as u64 + 1;
        n * (self.shells[ia].ncart * self.shells[ib].ncart) as u64
    }

    /// The deterministic greedy split ([`PAIR_FT_TASK_SPLIT`]): per bra
    /// shell, ket shells in order, a range is closed before the pair that
    /// would push it past the cap. Every shell pair lands in exactly one
    /// task; each bra shell has at least one.
    fn build_tasks(&self) -> Vec<PairTaskSpec> {
        let ns = self.shells.len();
        let (mut total, mut heaviest) = (0u64, 0u64);
        for ia in 0..ns {
            for ib in 0..ns {
                let w = self.pair_weight(ia, ib);
                total += w;
                heaviest = heaviest.max(w);
            }
        }
        let cap = total.div_ceil(PAIR_FT_TASK_SPLIT).max(heaviest);
        let col = |kb: usize| {
            if kb < ns {
                self.shells[kb].off
            } else {
                self.nbf
            }
        };
        let mut tasks = Vec::new();
        for ia in 0..ns {
            let (mut kb0, mut acc) = (0usize, 0u64);
            for ib in 0..ns {
                let w = self.pair_weight(ia, ib);
                if ib > kb0 && acc + w > cap {
                    tasks.push(PairTaskSpec {
                        ia,
                        kb0,
                        kb1: ib,
                        c0: col(kb0),
                        c1: col(ib),
                    });
                    (kb0, acc) = (ib, 0);
                }
                acc += w;
            }
            tasks.push(PairTaskSpec {
                ia,
                kb0,
                kb1: ns,
                c0: col(kb0),
                c1: col(ns),
            });
        }
        tasks
    }

    /// Load-balance counters of this plan (max over the plans of a call
    /// sequence, e.g. the main and smooth LR passes): tasks per chunk, and
    /// the work proxy `survivors × ncart(bra) × ncart(ket)` summed, of the
    /// largest task, of the largest single pair and of the largest bra-shell
    /// row (the pre-pair task granularity). `max task / (total / tasks)` is
    /// the imbalance factor.
    pub(crate) fn record_stats(&self, t: &mut PbcTimings) {
        let ns = self.shells.len();
        let cost = |ia: usize, ib: usize| {
            self.survivors[ia * ns + ib].len() as u64
                * (self.shells[ia].ncart * self.shells[ib].ncart) as u64
        };
        let (mut total, mut max_pair, mut max_bra, mut nsurv) = (0u64, 0u64, 0u64, 0u64);
        for ia in 0..ns {
            let mut bra = 0u64;
            for ib in 0..ns {
                let c = cost(ia, ib);
                nsurv += self.survivors[ia * ns + ib].len() as u64;
                bra += c;
                max_pair = max_pair.max(c);
            }
            total += bra;
            max_bra = max_bra.max(bra);
        }
        let max_task = self
            .tasks
            .iter()
            .map(|k| (k.kb0..k.kb1).map(|ib| cost(k.ia, ib)).sum::<u64>())
            .max()
            .unwrap_or(0);
        t.add_counter("pair FT survivors (sum over plans)", nsurv);
        t.max_counter("pair FT shells", ns as u64);
        t.max_counter("pair FT tasks per chunk", self.tasks.len() as u64);
        t.max_counter("pair FT cost total", total);
        t.max_counter("pair FT cost max task", max_task);
        t.max_counter("pair FT cost max pair", max_pair);
        t.max_counter("pair FT cost max bra shell", max_bra);
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

    /// The serial kernels' screen for shell pair `(ia, ib)`, in walk order
    /// (image, bra primitive, ket primitive), with their exact expressions.
    fn walk(
        &self,
        ia: usize,
        ib: usize,
        thresh: f64,
        gmax: f64,
        g2min: f64,
        mut f: impl FnMut(Survivor),
    ) {
        let (sa, sb) = (&self.shells[ia], &self.shells[ib]);
        let (la, lb) = (sa.l, sb.l);
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
            for (ka, (&a, &ca)) in sa.exps.iter().zip(&sa.coefs).enumerate() {
                for (kb, (&b, &cb)) in sb.exps.iter().zip(&sb.coefs).enumerate() {
                    let p = a + b;
                    let cc = ca * cb * (pi / p).powf(1.5);
                    let mag = (cc * (-a * b / p * r2).exp()).abs();
                    if mag < thresh {
                        continue;
                    }
                    let g2max = 4.0
                        * p
                        * ((mag / thresh).ln().max(0.0)
                            + WINDOW_MARGIN
                            + (la + lb) as f64 * gmax.max(1.0).ln());
                    // The serial kernel's `ngp == 0` in every chunk.
                    if g2max < g2min {
                        continue;
                    }
                    f(Survivor {
                        image: il as u32,
                        ia: ka as u16,
                        ib: kb as u16,
                        cc,
                        g2max,
                    });
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
        let chunk = ChunkG::new(gvecs, self.lmax);
        let mut segs = self.segments(&mut out, ng);
        // Task `k` owns `segs[seg(k)..seg(k) + nr * nfun(bra)]`: the tasks
        // take consecutive runs of the task-major segment list.
        let mut rest: &mut [&mut [Complex64]] = &mut segs;
        let views: Vec<(&PairTaskSpec, &mut [&mut [Complex64]])> = self
            .tasks
            .iter()
            .map(|k| {
                let n = self.nr * self.shells[k.ia].nfun;
                let (mine, tail) = std::mem::take(&mut rest).split_at_mut(n);
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
                pairs.fetch_add((k.kb1 - k.kb0) as u64, Ordering::Relaxed);
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
    /// run is `r * nfun(bra) + i` -> row `off(bra) + i` of `P[r]`, columns
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
        let nseg: usize = self
            .tasks
            .iter()
            .map(|k| self.nr * self.shells[k.ia].nfun)
            .sum();
        let mut segs = Vec::with_capacity(nseg);
        for k in &self.tasks {
            let sa = &self.shells[k.ia];
            let width = (k.c1 - k.c0) * ng;
            for r in 0..self.nr {
                for i in 0..sa.nfun {
                    let row = &mut rows[r * nbf + sa.off + i];
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

    /// Task `k`: its ket shells in order, each pair into its columns of the
    /// task's row segments (`dst[r * nfun(bra) + i]`, local column
    /// `off(ket) − c0(k) + j`). Returns the task's window tallies (counted
    /// locally, added to the shared counters once per task).
    fn task_block(
        &self,
        k: &PairTaskSpec,
        dst: &mut [&mut [Complex64]],
        scr: &mut PairScratch,
        ch: &ChunkG,
    ) -> WindowTally {
        let mut w = WindowTally::default();
        for ib in k.kb0..k.kb1 {
            let col0 = self.shells[ib].off - k.c0;
            let n_used = self.pair_block(k.ia, ib, dst, col0, scr, ch);
            if n_used == 0 {
                w.empty += 1;
            } else if n_used < ch.ng {
                w.partial += 1;
            }
        }
        w
    }

    /// Shell pair `(ia, ib)` into its columns (from local column `col0`) of
    /// the row segments `dst`: the serial kernels' inner loop over the
    /// cached survivors, then the Cartesian→AO scatter. Returns the pair's
    /// window `n_used` in this chunk (observation only).
    fn pair_block(
        &self,
        ia: usize,
        ib: usize,
        dst: &mut [&mut [Complex64]],
        col0: usize,
        scr: &mut PairScratch,
        ch: &ChunkG,
    ) -> usize {
        let zero = Complex64::new(0.0, 0.0);
        let (ng, gvecs, g2, pw) = (ch.ng, &ch.gsorted, &ch.g2, &ch.pw);
        let ns = self.shells.len();
        let (sa, sb) = (&self.shells[ia], &self.shells[ib]);
        let (la, lb) = (sa.l, sb.l);
        let (nca, ncb) = (sa.ncart, sb.ncart);
        let st = la + lb + 1;
        let nij = (la + 1) * (lb + 1);
        let nfa = sa.nfun;
        // No survivor reaches g >= n_used (module doc): only 0..n_used of
        // the Cartesian accumulators is zeroed, accumulated and transformed.
        let n_used = g2.partition_point(|&x| x <= self.g2max_pair[ia * ns + ib]);
        if n_used == 0 {
            for rows in dst.chunks_mut(nfa) {
                self.scatter(rows, col0, &[], 0, &mut scr.tmp, sa, sb, &ch.order);
            }
            return 0;
        }
        // Stale scratch is never read: `common` and the F rows are written
        // for g < ngp before being read there, `e_table` zeroes its range,
        // the Cartesian accumulators restart from +0.0 on 0..n_used (as the
        // serial kernel's fresh `vec![zero; ..]`) and nothing reads them
        // beyond it.
        scr.common.resize(ng, zero);
        for d in 0..3 {
            scr.ebuf[d].resize(nij * st, 0.0);
            scr.fbuf[d].resize(nij * ng, zero);
        }
        scr.cart.resize_with(self.nr, Vec::new);
        for c in scr.cart.iter_mut() {
            c.resize(nca * ncb * ng, zero);
            for row in c.chunks_mut(ng) {
                row[..n_used].fill(zero);
            }
        }
        let ca_comps = &self.comps[la];
        let cb_comps = &self.comps[lb];
        let PairScratch {
            common,
            ebuf,
            fbuf,
            cart,
            tmp,
        } = scr;

        for s in &self.survivors[ia * ns + ib] {
            let ngp = g2.partition_point(|&x| x <= s.g2max);
            if ngp == 0 {
                continue;
            }
            let l = &self.images[s.image as usize];
            let cart = &mut cart[self.bucket[s.image as usize]];
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
            let a = sa.exps[s.ia as usize];
            let b = sb.exps[s.ib as usize];
            let p = a + b;
            let cc = s.cc;
            let pc = [
                (a * sa.center[0] + b * bc[0]) / p,
                (a * sa.center[1] + b * bc[1]) / p,
                (a * sa.center[2] + b * bc[2]) / p,
            ];
            for (g, gv) in gvecs.iter().enumerate().take(ngp) {
                let mag = cc * (-g2[g] / (4.0 * p)).exp();
                let ph = gv[0] * pc[0] + gv[1] * pc[1] + gv[2] * pc[2];
                // e^{−iG·P}
                common[g] = Complex64::new(mag * ph.cos(), -mag * ph.sin());
            }
            for d in 0..3 {
                e_table(la, lb, a, b, ab[d], &mut ebuf[d]);
                let (e, f, w) = (&ebuf[d], &mut fbuf[d], &pw[d]);
                for ij in 0..nij {
                    let frow = &mut f[ij * ng..ij * ng + ngp];
                    frow.fill(zero);
                    for t in 0..st {
                        let et = e[ij * st + t];
                        if et == 0.0 {
                            continue;
                        }
                        let wrow = &w[t * ng..t * ng + ngp];
                        for (x, wv) in frow.iter_mut().zip(wrow) {
                            *x += *wv * et;
                        }
                    }
                }
            }
            let (fx, fy, fz) = (&fbuf[0], &fbuf[1], &fbuf[2]);
            for (u, ac) in ca_comps.iter().enumerate() {
                for (v, bcmp) in cb_comps.iter().enumerate() {
                    let ix = (ac[0] as usize * (lb + 1) + bcmp[0] as usize) * ng;
                    let iy = (ac[1] as usize * (lb + 1) + bcmp[1] as usize) * ng;
                    let iz = (ac[2] as usize * (lb + 1) + bcmp[2] as usize) * ng;
                    let dst = &mut cart[(u * ncb + v) * ng..(u * ncb + v) * ng + ngp];
                    for g in 0..ngp {
                        dst[g] += common[g] * fx[ix + g] * fy[iy + g] * fz[iz + g];
                    }
                }
            }
        }
        for (rows, cart) in dst.chunks_mut(nfa).zip(cart.iter()) {
            self.scatter(rows, col0, cart, n_used, tmp, sa, sb, &ch.order);
        }
        n_used
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

/// One task of every chunk: bra shell `ia`, ket shells `kb0..kb1`, whose
/// AO columns are `c0..c1` ([`PairFtPlan::build_tasks`]).
struct PairTaskSpec {
    ia: usize,
    kb0: usize,
    kb1: usize,
    c0: usize,
    c1: usize,
}

/// Per-task tallies of the (shell pair, G chunk) windows: `partial` counts
/// `0 < n_used < n_G(chunk)` (accumulated on `0..n_used`, `+0.0` written
/// beyond), `empty` counts `n_used == 0` (the early all-`+0.0` branch).
/// Pairs with `n_used == n_G(chunk)` are not counted.
#[derive(Default)]
struct WindowTally {
    partial: u64,
    empty: u64,
}

/// Per-worker scratch of [`PairFtPlan::pair_block`], reused across pairs
/// and tasks (never read stale; see there).
#[derive(Default)]
struct PairScratch {
    common: Vec<Complex64>,
    ebuf: [Vec<f64>; 3],
    fbuf: [Vec<Complex64>; 3],
    cart: Vec<Vec<Complex64>>,
    tmp: Vec<Complex64>,
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
/// exactly as the serial kernels sort it) with its `|G|²` and `(−iG_d)^t`.
struct ChunkG {
    ng: usize,
    order: Vec<usize>,
    gsorted: Vec<[f64; 3]>,
    g2: Vec<f64>,
    pw: Vec<Vec<Complex64>>,
}

impl ChunkG {
    fn new(gvecs: &[[f64; 3]], lmax: usize) -> Self {
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
        Self {
            ng,
            order,
            gsorted,
            g2,
            pw,
        }
    }
}

/// Smallest `|G|²` of a G set (the serial kernels' expression); `+∞` for an
/// empty set.
pub(crate) fn min_gnorm2(gvecs: &[[f64; 3]]) -> f64 {
    gvecs.iter().map(norm2).fold(f64::INFINITY, f64::min)
}
