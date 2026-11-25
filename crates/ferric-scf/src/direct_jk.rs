//! Combined direct Coulomb + exchange (J+K) matrix construction from a single quartet pass.

use crate::quartet_scatter::{JkDigest, ScatterTimes};
use crate::screening::SchwarzBounds;
use ferric_core::parallel::ParallelContext;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ndarray::Array2;

pub mod gc_rotation;
pub use gc_rotation::{GcRotate, GC_ROTATE_ENV};
use gc_rotation::{GcState, RotatedJk};

/// `energy_conv` below this multiple of `integral_thresh` turns the
/// closed-shell incremental Fock build off by default (#415). The measured dE
/// noise of the incremental path is up to ~250x `integral_thresh` (table on
/// [`closed_shell_incremental_enabled`]), so this leaves a ~40x margin.
pub const INCREMENTAL_ENERGY_CONV_FACTOR: f64 = 1e4;

/// `density_conv` below this multiple of `integral_thresh` turns the
/// closed-shell incremental Fock build off by default (#415). The lowest
/// dp_rms the incremental path reached was 0.07-12x `integral_thresh`
/// (benzene/cc-pVDZ: 1.2e-11 at 1e-12), so this leaves a ~80x margin.
pub const INCREMENTAL_DENSITY_CONV_FACTOR: f64 = 1e3;

/// Is the closed-shell INCREMENTAL (ΔD) Fock build in `solve_rhf` enabled?
///
/// `FERRIC_SCF_INCREMENTAL=1` (or `on`/`true`) forces it on, `0` (or
/// `off`/`false`) forces the full rebuild every iteration. Unset (or any
/// other value), it is on unless the requested convergence is tighter than its
/// noise allows: `energy_conv < INCREMENTAL_ENERGY_CONV_FACTOR *
/// integral_thresh` or `density_conv < INCREMENTAL_DENSITY_CONV_FACTOR *
/// integral_thresh`. The defaults (1e-3, 1e-6, 1e-12) keep it on.
///
/// # Why, with the measurements (#415)
///
/// Each incremental iteration adds `J(ΔD)`/`K(ΔD)` built under the
/// Häser-Ahlrichs density screen, so the quartets the screen drops (at
/// `integral_thresh` scaled by `max|ΔD|`) are missing from the running J/K
/// until the next periodic full rebuild. That shows up as noise in the
/// iteration-to-iteration `dE` and density change, roughly linear in
/// `integral_thresh`, and makes a very tight `energy_conv` reachable only by
/// chance (#410):
///
/// ```text
/// (scripts/incr_fock_ab.py noise: median |dE| once dp_rms < 1e-9)
///                     thresh   incremental   full rebuild
///   water/cc-pVDZ      1e-12     3.3e-12       3.6e-14
///   benzene/cc-pVDZ    1e-12     8.0e-11       1.7e-13
///   HBr/def2-SVP       1e-12     1.4e-11       9.1e-13
///   SiH4/cc-pVDZ       1e-12     2.6e-11       2.3e-13
///   C10H22/STO-3G      1e-10     1.9e-8        2.3e-13
///   C10H22/STO-3G      1e-12     2.5e-10       2.3e-13
///   C10H22/STO-3G      1e-14     4.5e-13       2.3e-13
/// ```
///
/// It is not turned off outright because it does pay on extended systems,
/// where the ΔD screen prunes. Default settings, one SCF per fresh process
/// (scripts/incr_fock_ab.py cost; wall times from a shared 4-core box):
///
/// ```text
///                     nbf  iters   quartets incr/full   wall full/incr
///   water/cc-pVDZ      24  10/10        1.000               ~1.0
///   benzene/cc-pVDZ   114   9/9         0.990               1.01
///   HBr/def2-SVP       37  11/11        0.999               ~1.0
///   C10H22/STO-3G      72   8/8         0.783               1.11
///   C20H42/STO-3G     142   8/8         0.744               1.23
///   C32H66/STO-3G     226   8/8         0.733               1.21
///   C48H98/STO-3G     338   8/8         0.727               1.18
///   C10H22/6-31G      134   9/9         0.857               1.09
/// ```
pub fn closed_shell_incremental_enabled(
    energy_conv: f64,
    density_conv: f64,
    integral_thresh: f64,
) -> bool {
    match std::env::var("FERRIC_SCF_INCREMENTAL").ok().as_deref() {
        Some("1") | Some("on") | Some("true") | Some("ON") | Some("TRUE") => true,
        Some("0") | Some("off") | Some("false") | Some("OFF") | Some("FALSE") => false,
        _ => {
            energy_conv >= INCREMENTAL_ENERGY_CONV_FACTOR * integral_thresh
                && density_conv >= INCREMENTAL_DENSITY_CONV_FACTOR * integral_thresh
        }
    }
}

/// Is the open-shell INCREMENTAL (ΔD) Fock build enabled?
///
/// **Default OFF**, unlike the closed-shell `solve_rhf` path (see
/// [`closed_shell_incremental_enabled`]). Opt in with
/// `FERRIC_SCF_UHF_INCREMENTAL=1`.
///
/// # Why it is off, with the measurement
///
/// The implementation is correct — `build_uhf_incremental` reproduces a
/// from-scratch rebuild to ~1e-14, including the adversarial opposing-spin-delta
/// case (see the direct_jk.rs unit tests and `open_shell_combined_jk.rs`). It
/// simply does not PAY at any size measured here, because the
/// Häser-Ahlrichs screen only starts rejecting quartets once `max|ΔD|` drops
/// below ~1e-8, and an SCF run reaches that only in its last iteration or two:
///
/// ```text
/// (measured, CH3/cc-pVDZ, quartet counts; the DEFAULT density_conv is 1e-6)
///   density_conv 1e-6 : full 87120  incr 87120   -> 1.000x
///   density_conv 1e-8 : full 101640 incr 101636  -> 1.000x
///   density_conv 1e-10: full 123420 incr 122984  -> 1.004x
///   density_conv 1e-12: full 152460 incr 154530  -> 0.987x  (WORSE, +4 iters)
///
/// (measured, larger systems, default thresholds)
///   alkane_4 (+1,doublet)/cc-pVDZ: 21412773 -> 21201021 quartets = 1.010x
///   alkane_8 (+1,doublet)/6-31G:   90222585 -> 86214445 quartets = 1.046x
/// ```
///
/// Wall-clock showed no reliable win either: on this (shared, load-average-16)
/// box the alkane_8 timing moved between 1.006x and 1.740x across repeat runs
/// while the quartet counts were bit-stable, i.e. the timing spread was machine
/// contention, not signal. A 4.6% work reduction cannot produce a 74% wall
/// reduction; the large figures were artifacts and are not claimed.
///
/// The CLOSED-shell incremental path was found to give 1.000x on
/// water/cc-pVDZ and methane/cc-pVDZ and 1.016x on alkane_4/cc-pVDZ, but #415
/// measured 0.73-0.86x quartets on alkane chains at STO-3G/6-31G (table on
/// [`closed_shell_incremental_enabled`]), so the closed-shell default stays
/// on there. These open-shell figures predate that and were not re-measured.
///
/// Keeping the code, tested and switchable, rather than deleting it: the
/// mechanism is sound and becomes profitable if `integral_thresh` is ever
/// loosened relative to `density_conv`, or on systems large enough for
/// locality to bite. Defaulting it ON today would buy a per-iteration density
/// clone and extra SCF-loop state for ~1% fewer quartets.
pub fn open_shell_incremental_enabled() -> bool {
    matches!(
        std::env::var("FERRIC_SCF_UHF_INCREMENTAL").ok().as_deref(),
        Some("1") | Some("on") | Some("true") | Some("ON") | Some("TRUE")
    )
}

/// Is the combined single-pass open-shell J+K build enabled?
///
/// Default ON. `FERRIC_SCF_COMBINED_JK=0` (or `off`/`false`) restores the
/// historical three-pass open-shell Fock build (`DirectJ(D_total)` +
/// `DirectK(D_α)` + `DirectK(D_β)`) in `solve_uhf`/`solve_rohf`, so the two can
/// be compared for correctness and timing in one binary. Same string
/// convention as `FERRIC_SCF_INCREMENTAL`.
pub fn combined_open_shell_jk_enabled() -> bool {
    !matches!(
        std::env::var("FERRIC_SCF_COMBINED_JK").ok().as_deref(),
        Some("0") | Some("off") | Some("false") | Some("OFF") | Some("FALSE")
    )
}

/// Combined Coulomb (J) and exchange (K) matrix builder.
///
/// Iterates all screened canonical (s1,s2,s3,s4) quartets **once** and accumulates
/// both J and K from the same computed integral, avoiding the 2× ERI evaluation cost
/// of calling DirectJ and DirectK separately.
///
/// The work list is the screened bra-pair list (O(nsh²) memory); the ket loop and
/// the Häser-Ahlrichs density screen run inside the parallel group workers, with
/// ~n_pairs/1024 pairs per group amortizing load imbalance across rayon's
/// dynamic scheduling (same structure as `rhf::build_jk`). The former flat
/// quartet pre-enumeration was a serial O(nsh⁴) loop with an unbounded
/// surviving-quartet Vec reallocated every SCF iteration.
pub struct DirectJK<'a> {
    ctx: &'a ParallelContext,
    prep: &'a PreparedBasis,
    bounds: &'a SchwarzBounds,
    thresh: f64,
    /// Fully-resolved unified memory budget (TOML > env > auto), passed in by
    /// the solver — caps the reduction band scratch via `resolve_band_bytes`.
    mem_budget: usize,
    // Lazily built on first build() and reused for the builder's lifetime:
    // libint2 engine construction is serialized behind a global ctor mutex,
    // so hoist the builder out of the SCF loop to pay it once, not per iteration.
    pool: Option<crate::engine_pool::EnginePool>,
    /// The internal contraction rotation ([`gc_rotation`]): requested until
    /// the first direct build, then resolved (the rotated basis and bounds
    /// live in `bounds.gc`, shared with every later build on these bounds).
    gc: gc_rotation::GcState,
    /// Engines on the rotated shells, owned like `pool`.
    rot_pool: Option<crate::engine_pool::EnginePool>,
    /// `max|D|` per exchange channel of the last full build's walk density
    /// (one entry closed shell, two open shell; `None` before the first
    /// full build, or when it ran the legacy digest): the reference of the
    /// pair digest's symmetry check on an incremental build's `ΔD`
    /// ([`crate::quartet_scatter::SymScale::Delta`]).
    sym_ref: Option<Vec<f64>>,
}

impl<'a> std::fmt::Debug for DirectJK<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DirectJK")
            .field("thresh", &self.thresh)
            .field("mem_budget", &self.mem_budget)
            .finish_non_exhaustive()
    }
}

impl<'a> DirectJK<'a> {
    /// Whether the incremental (`ΔD`) build saves work: not when the bounds
    /// carry the in-core ERI store, whose J/K cost does not shrink with `ΔD`.
    pub fn incremental_pays(&self) -> bool {
        self.bounds.incore.get().is_none()
    }

    /// Create a combined screened J+K builder.
    pub fn new(
        ctx: &'a ParallelContext,
        prep: &'a PreparedBasis,
        bounds: &'a SchwarzBounds,
        thresh: f64,
        mem_budget: usize,
    ) -> Self {
        DirectJK {
            ctx,
            prep,
            bounds,
            thresh,
            mem_budget,
            pool: None,
            gc: GcState::Pending(None),
            rot_pool: None,
            sym_ref: None,
        }
    }

    /// Choose the internal contraction rotation ([`gc_rotation`]) instead
    /// of following the bounds. It takes effect at the first direct build;
    /// a builder that has built already keeps its choice.
    #[must_use]
    pub fn with_gc_rotation(self, mode: GcRotate) -> Self {
        self.with_gc_request(Some(mode))
    }

    /// [`with_gc_rotation`](Self::with_gc_rotation) from a config's
    /// `gc_rotation` field (an SCF's): `None` = unset, `FERRIC_GC_ROTATE`
    /// and then `auto` decide. A builder given neither follows its bounds
    /// ([`gc_rotation::GcSlot::mode`]: unrotated unless a config recorded a
    /// request on them).
    #[must_use]
    pub fn with_gc_request(mut self, mode: Option<GcRotate>) -> Self {
        if let GcState::Pending(_) = self.gc {
            self.gc = GcState::Pending(Some(mode));
        }
        self
    }

    /// Columns the contraction rotation rotated: `None` before the first
    /// direct build (or while every build was served by the in-core store),
    /// `Some(0)` when the builder runs unrotated (switch off, or nothing in
    /// the basis to rotate).
    pub fn gc_rotated_columns(&self) -> Option<usize> {
        match &self.gc {
            GcState::Pending(_) => None,
            GcState::Resolved(r) => Some(r.as_ref().map_or(0, |r| r.basis.n_rotated_columns)),
        }
    }

    /// Resolve a pending rotation request: the builder's own (config)
    /// request, else the bounds' [`gc_rotation::GcSlot::mode`].
    /// The rotated basis and bounds come from the bounds' cache (built at
    /// the first rotated build on these bounds, by whichever builder runs
    /// first).
    fn resolve_gc(&mut self) -> Result<(), FerricError> {
        if let GcState::Pending(request) = self.gc {
            let mode = match request {
                Some(v) => GcRotate::resolve(v)?,
                None => self.bounds.gc.mode()?,
            };
            let r = self.bounds.gc.rotation(self.prep, self.bounds, mode)?;
            self.gc = GcState::Resolved(r);
        }
        Ok(())
    }

    /// The unrotated walk over `self`'s own basis and bounds.
    fn walk(&mut self) -> Walk<'_> {
        Walk {
            ctx: self.ctx,
            prep: self.prep,
            bounds: self.bounds,
            thresh: self.thresh,
            mem_budget: self.mem_budget,
            pool: &mut self.pool,
        }
    }

    /// Incremental Fock build: given the DENSITY CHANGE `delta_d = D_new - D_last`,
    /// ACCUMULATE `ΔJ = J(delta_d)` and `ΔK = K(delta_d)` onto the caller's
    /// existing `j`/`k` buffers (which must already hold `J(D_last)`/`K(D_last)`),
    /// yielding `J(D_new)`/`K(D_new)`. The caller must NOT zero `j`/`k` first.
    ///
    /// This is mathematically EXACT (not an approximation): J and K are linear in
    /// D, so `J(D_last) + J(ΔD) == J(D_last + ΔD) == J(D_new)` in infinite
    /// precision. In f64 it differs from a from-scratch `build(&D_new, ..)` only
    /// by the reassociation floor of summing many small increments vs one full
    /// contraction — kept small by a periodic full rebuild in the SCF loop (see
    /// `crate::driver::INCREMENTAL_FULL_REBUILD_EVERY`). The screen inside
    /// `build_d_max_shell` is driven by `delta_d`, so as SCF converges (ΔD → 0)
    /// almost every quartet is Häser-Ahlrichs-screened out and late iterations
    /// become nearly free (the PySCF `pyscf/scf/hf.py` / Psi4 `CompositeJK.cc`
    /// incremental-Fock scheme).
    ///
    /// Bit-identity across `RAYON_NUM_THREADS` is preserved: this shares the exact
    /// same deterministic grouped reduction as `build` — only the input density
    /// (a delta vs the full D) and the caller's zero-vs-accumulate choice differ.
    pub fn build_incremental(
        &mut self,
        delta_d: &Array2<f64>,
        j: &mut Array2<f64>,
        k: &mut Array2<f64>,
    ) -> Result<usize, FerricError> {
        // `build` already accumulates `J(d)`/`K(d)` onto the caller's buffers via
        // `*j += &total_j; *k += &total_k`, and screens with `build_d_max_shell(d)`.
        // Passing `delta_d` (without the caller zeroing) is therefore exactly the
        // incremental update — no separate kernel needed. The one difference:
        // the pair digest judges `ΔD`'s symmetry against the last full
        // density's scale (`quartet_scatter::is_symmetric_to_roundoff`).
        self.build_with(delta_d, j, k, true)
    }

    /// Open-shell combined build: ONE quartet pass producing `J[D_α+D_β]`,
    /// `K[D_α]` and `K[D_β]`, ACCUMULATED onto the caller's three buffers.
    ///
    /// This replaces the previous open-shell three-pass Fock build
    /// (`DirectJ(D_total)` + `DirectK(D_α)` + `DirectK(D_β)`), which evaluated
    /// every surviving shell quartet THREE times and screened each pass on the
    /// loose single global `max|D|` scalar. Here each quartet's integral is
    /// computed once and scattered into all three matrices, under the tight
    /// six-pairwise Häser-Ahlrichs shell screen the closed-shell path already
    /// used.
    ///
    /// The screening key is `|D_α| + |D_β|` elementwise — the cheapest key that
    /// is simultaneously an upper bound on all three contracted densities; see
    /// `quartet_scatter::build_d_max_shell_spin_sum` (private — a code span,
    /// not a doc link) for why the sum rather than the (tighter but unsound
    /// for J) max.
    ///
    /// `d_total` is passed explicitly rather than formed here so the caller's
    /// existing `&d_a + &d_b` temporary is reused, and so the INCREMENTAL path
    /// can pass `ΔD_total` — which is *not* recoverable from the two spin
    /// deltas without an extra allocation.
    ///
    /// Bit-identity across `RAYON_NUM_THREADS` holds by the same argument as
    /// [`build`](Self::build): the pair list, group partition and fold order are
    /// pure functions of the work list, never of the thread count.
    pub fn build_uhf(
        &mut self,
        d_total: &Array2<f64>,
        d_a: &Array2<f64>,
        d_b: &Array2<f64>,
        j: &mut Array2<f64>,
        k_a: &mut Array2<f64>,
        k_b: &mut Array2<f64>,
    ) -> Result<usize, FerricError> {
        self.build_uhf_with([d_total, d_a, d_b], j, k_a, k_b, false)
    }

    /// [`build_uhf`](Self::build_uhf) on full densities, or on the deltas of
    /// [`build_uhf_incremental`](Self::build_uhf_incremental). `d` is
    /// `[D_total, D_α, D_β]` (or their deltas).
    fn build_uhf_with(
        &mut self,
        d: [&Array2<f64>; 3],
        j: &mut Array2<f64>,
        k_a: &mut Array2<f64>,
        k_b: &mut Array2<f64>,
        incremental: bool,
    ) -> Result<usize, FerricError> {
        let [d_total, d_a, d_b] = d;
        self.ctx.check_interrupted()?;
        if let Some(store) = crate::incore_jk::store_for(self.bounds, self.prep)? {
            store.add_uhf(d_total, d_a, d_b, j, k_a, k_b)?;
            return Ok(store.quartets());
        }
        self.resolve_gc()?;
        let reference = self.sym_ref.take();
        let mut scales = None;
        let sym = sym_scale(incremental, reference.as_deref());
        let computed = if !self.rotated() {
            self.walk()
                .build_uhf([d_total, d_a, d_b], j, k_a, k_b, sym, &mut scales)
        } else {
            self.run_rotated(&[d_total, d_a, d_b], &mut [j, k_a, k_b], |mut w, d, x| {
                let [jr, kar, kbr] = x else {
                    unreachable!("three outputs")
                };
                w.build_uhf([&d[0], &d[1], &d[2]], jr, kar, kbr, sym, &mut scales)
            })
        };
        self.sym_ref = if incremental { reference } else { scales };
        computed
    }

    /// Incremental open-shell Fock build: given the per-spin density CHANGES
    /// (`Δ_total = ΔD_α + ΔD_β`), ACCUMULATE the corresponding `ΔJ`/`ΔK_α`/`ΔK_β`
    /// onto buffers that already hold the previous iteration's `J`/`K_α`/`K_β`.
    /// The caller must NOT zero them first.
    ///
    /// Exact by linearity, exactly as in the closed-shell
    /// [`build_incremental`](Self::build_incremental): J and K are linear in the
    /// density, so `K(D_last) + K(ΔD) == K(D_new)` in infinite precision, and in
    /// f64 the difference from a from-scratch rebuild is the reassociation floor
    /// — bounded by the caller's periodic full rebuild.
    ///
    /// ## Screening key (the open-shell-specific decision)
    ///
    /// The screen runs on `|ΔD_α| + |ΔD_β|`, NOT on `|ΔD_total|`. The two spin
    /// channels converge independently and routinely move in opposite
    /// directions — a spin-flip rearrangement can leave `ΔD_total ≈ 0` while
    /// `ΔD_α` and `ΔD_β` are both large. Screening on the total would then drop
    /// quartets that carry a large, real contribution to `ΔK_α` and `ΔK_β`
    /// individually, silently corrupting the spin Focks while the Coulomb term
    /// looked fine. The spin sum bounds all three deltas (triangle inequality),
    /// so it is safe for whichever channel is largest, and it costs nothing
    /// extra: as SCF converges BOTH channels go to zero together and the screen
    /// tightens just as hard as the closed-shell one.
    pub fn build_uhf_incremental(
        &mut self,
        delta_total: &Array2<f64>,
        delta_a: &Array2<f64>,
        delta_b: &Array2<f64>,
        j: &mut Array2<f64>,
        k_a: &mut Array2<f64>,
        k_b: &mut Array2<f64>,
    ) -> Result<usize, FerricError> {
        // `build_uhf` already ACCUMULATES onto the caller's buffers and screens
        // on the spin sum of whatever densities it is handed, so passing the
        // deltas (without the caller zeroing) IS the incremental update — the
        // same relationship `build_incremental` has to `build` (the symmetry
        // reference included).
        self.build_uhf_with([delta_total, delta_a, delta_b], j, k_a, k_b, true)
    }

    /// Build J and K matrices simultaneously from a single pass over shell quartets.
    /// Returns the number of unique quartets computed.
    ///
    /// ACCUMULATES onto `j`/`k` (`*j += J(d)`, `*k += K(d)`) — callers doing a
    /// full rebuild zero the buffers first; the incremental path
    /// ([`build_incremental`](Self::build_incremental)) passes `ΔD` and does not.
    pub fn build(
        &mut self,
        d: &Array2<f64>,
        j: &mut Array2<f64>,
        k: &mut Array2<f64>,
    ) -> Result<usize, FerricError> {
        self.build_with(d, j, k, false)
    }

    /// [`build`](Self::build) on a full density, or on the `ΔD` of
    /// [`build_incremental`](Self::build_incremental).
    fn build_with(
        &mut self,
        d: &Array2<f64>,
        j: &mut Array2<f64>,
        k: &mut Array2<f64>,
        incremental: bool,
    ) -> Result<usize, FerricError> {
        self.ctx.check_interrupted()?;
        if let Some(store) = crate::incore_jk::store_for(self.bounds, self.prep)? {
            store.add_jk(d, j, k)?;
            return Ok(store.quartets());
        }
        self.resolve_gc()?;
        let reference = self.sym_ref.take();
        let mut scales = None;
        let sym = sym_scale(incremental, reference.as_deref());
        let computed = if !self.rotated() {
            self.walk().build(d, j, k, sym, &mut scales)
        } else {
            self.run_rotated(&[d], &mut [j, k], |mut w, d, x| {
                let [jr, kr] = x else {
                    unreachable!("two outputs")
                };
                w.build(&d[0], jr, kr, sym, &mut scales)
            })
        };
        self.sym_ref = if incremental { reference } else { scales };
        computed
    }

    /// Whether the resolved rotation is active.
    fn rotated(&self) -> bool {
        matches!(self.gc, GcState::Resolved(Some(_)))
    }

    /// A build on the rotated shells: `D' = Tᵀ D T` for each of `ds`, `run`
    /// on fresh rotated outputs, then `out += T X' Tᵀ` in place for each of
    /// `outs`. The `D'` and `X'` buffers (`(ds + outs) · n²` doubles) are
    /// charged to the memory pool for the build.
    fn run_rotated(
        &mut self,
        ds: &[&Array2<f64>],
        outs: &mut [&mut Array2<f64>],
        run: impl FnOnce(Walk<'_>, &[Array2<f64>], &mut [Array2<f64>]) -> Result<usize, FerricError>,
    ) -> Result<usize, FerricError> {
        let (ctx, thresh, mem_budget) = (self.ctx, self.thresh, self.mem_budget);
        let GcState::Resolved(Some(r)) = &self.gc else {
            unreachable!("run_rotated without a rotation");
        };
        let r: &RotatedJk = r;
        let n = r.basis.nbasis();
        let _charge = r.charge_collective(ctx, ds.len() + outs.len())?;
        let t0 = std::time::Instant::now();
        let dr: Vec<Array2<f64>> = ds.iter().map(|d| r.forward(d)).collect();
        let t_fwd = t0.elapsed().as_secs_f64();
        let mut xr: Vec<Array2<f64>> = (0..outs.len()).map(|_| zeros(n)).collect();
        let computed = run(
            Walk {
                ctx,
                prep: &r.basis.prep,
                bounds: &r.bounds,
                thresh,
                mem_budget,
                pool: &mut self.rot_pool,
            },
            &dr,
            &mut xr,
        )?;
        let t1 = std::time::Instant::now();
        for (x, out) in xr.iter().zip(outs.iter_mut()) {
            r.back_add(x, out)?;
        }
        report_gc_transforms(
            t_fwd,
            t1.elapsed().as_secs_f64(),
            t0.elapsed().as_secs_f64(),
        );
        Ok(computed)
    }
}

/// The pair digest's symmetry scale of a build: a full density's own, or
/// for an incremental `ΔD` the last full build's per-channel `max|D|`
/// (its own when no full build recorded one).
fn sym_scale(incremental: bool, reference: Option<&[f64]>) -> crate::quartet_scatter::SymScale<'_> {
    use crate::quartet_scatter::SymScale;
    match (incremental, reference) {
        (true, Some(r)) => SymScale::Delta(r),
        _ => SymScale::Own,
    }
}

/// An `n × n` zero matrix.
fn zeros(n: usize) -> Array2<f64> {
    Array2::<f64>::zeros((n, n))
}

/// `FERRIC_TIMING` line of a rotated build: the forward and back transforms
/// and the build's total wall (walk included).
fn report_gc_transforms(forward: f64, back: f64, total: f64) {
    if ferric_core::timing::enabled() {
        eprintln!(
            "[timing] direct-jk   gc rotation: forward {forward:.4} s + back {back:.4} s; \
             rotated build total {total:.3} s"
        );
    }
}

/// One direct walk: a basis, its bounds and engine pool. [`DirectJK`] runs
/// it on its own basis, or on the rotated basis of [`gc_rotation`].
struct Walk<'w> {
    ctx: &'w ParallelContext,
    prep: &'w PreparedBasis,
    bounds: &'w SchwarzBounds,
    thresh: f64,
    mem_budget: usize,
    pool: &'w mut Option<crate::engine_pool::EnginePool>,
}

impl Walk<'_> {
    /// The screened canonical bra-pair work list for a given shell-blocked
    /// density-max table, plus the lazily-constructed engine pool.
    ///
    /// Factored out of [`build`](Self::build) / [`build_uhf`](Self::build_uhf)
    /// because both need the IDENTICAL list construction: the `bra_thresh`
    /// pre-filter, the MPI rank striping, and the one-engine-per-rayon-thread
    /// pool hoist. Keeping it in one place is what makes the open-shell path
    /// screen exactly as tightly as the closed-shell one instead of drifting.
    fn screened_bra_pairs(
        &mut self,
        d_max_shell: &ndarray::Array2<f64>,
    ) -> Result<Vec<(usize, usize)>, FerricError> {
        let nsh = self.prep.nshells();
        let max_d = d_max_shell.iter().cloned().fold(0.0f64, f64::max);
        let max_q: f64 = self.bounds.q.iter().cloned().fold(0.0f64, f64::max);
        let bra_thresh = if max_q > 0.0 {
            self.thresh / (max_q * max_d.max(1e-30))
        } else {
            self.thresh
        };
        let q_table = &self.bounds.q;
        let mut shell_pairs: Vec<(usize, usize)> = Vec::new();
        for s1 in 0..nsh {
            for s2 in 0..=s1 {
                if q_table[(s1, s2)] > bra_thresh {
                    shell_pairs.push((s1, s2));
                }
            }
        }
        // MPI rank striping (see `ParallelContext::stripe` doc; size == 1
        // keeps the full list unchanged).
        let shell_pairs: Vec<(usize, usize)> = self.ctx.stripe(shell_pairs);

        // One engine per rayon thread (see engine_pool): constructing it in the
        // fold init below would fire once per work-chunk and storm the global
        // libint2 ctor mutex (catastrophic for heavy-element bases).
        if self.pool.is_none() {
            *self.pool = Some(crate::engine_pool::EnginePool::new(
                self.bounds.op,
                self.prep,
                ferric_integrals::engine_pool::eri_precision(),
            )?);
        }
        Ok(shell_pairs)
    }

    /// The open-shell walk. `sym` is the pair digest's symmetry scale;
    /// `scales` receives its per-spin `max|D|` when the pair digest ran.
    /// `d` is `[D_total, D_α, D_β]`.
    fn build_uhf(
        &mut self,
        d: [&Array2<f64>; 3],
        j: &mut Array2<f64>,
        k_a: &mut Array2<f64>,
        k_b: &mut Array2<f64>,
        sym: crate::quartet_scatter::SymScale<'_>,
        scales: &mut Option<Vec<f64>>,
    ) -> Result<usize, FerricError> {
        use crate::quartet_scatter::{
            build_d_max_shell_spin_sum, scatter_bra_pair, DensityScreen, JkMode, PairSpec,
        };
        use std::sync::atomic::{AtomicUsize, Ordering};

        let [d_total, d_a, d_b] = d;
        let dims = self.prep.shell_dims();
        let offs = self.prep.shell_offsets();
        let thresh = self.thresh;
        let computed_quartets = AtomicUsize::new(0);

        let d_max_shell = build_d_max_shell_spin_sum(self.prep, d_a, d_b);
        let shell_pairs = self.screened_bra_pairs(&d_max_shell)?;

        let q_table = &self.bounds.q;
        // CSB `M` table when `[scf] screening = "csb"` selected it; `None`
        // (the default) leaves the hot loop on plain Schwarz, byte-identical.
        //
        // NOT applied to the BRA-PAIR prefilter above, and that is not an
        // omission: the prefilter screens the DIAGONAL quartet (s1,s2|s1,s2),
        // where plain Schwarz is EXACT — Cauchy-Schwarz is an equality there,
        // so Q(s1,s2)^2 == max |(s1 s2|s1 s2)| — and no valid upper bound can
        // be smaller than an exact one. CSB's `min` therefore selects the
        // Schwarz term at every diagonal quartet, making it provably inert on
        // this loop. Same structural fact `qqr.rs` records for QQR (see
        // `tests/qqr_diagonal_noop.rs`), and the same consequence: CSB acts
        // only on the innermost per-quartet test, never on a pair list.
        let m_table = self.bounds.csb_m.as_ref();
        // CSAM `X` table when `[scf] screening = "csam"` selected it. Mutually
        // exclusive with `m_table` by construction; `None` for both is the
        // byte-identical plain-Schwarz default. Like CSB it is NOT applied to the
        // bra-pair prefilter — but for a different reason: CSAM's diagonal
        // refinement factor is exactly 1 (X's diagonal is 1.0 by Cauchy-Schwarz
        // equality), so it too is provably inert on `(s1,s2|s1,s2)`.
        let x_table = self.bounds.csam_x.as_ref();
        let prep = self.prep;
        let nbf = prep.nbasis();
        let pool = self
            .pool
            .as_ref()
            .expect("pool initialized by screened_bra_pairs");

        let n_pairs = shell_pairs.len();
        let group_size = crate::reduce::deterministic_group_size(n_pairs);
        let n_groups = n_pairs.div_ceil(group_size.max(1)).max(1);
        let screen = DensityScreen::SixPair(&d_max_shell);

        let band_bytes = crate::reduce::resolve_band_bytes(self.mem_budget);
        // One group's bra pairs scattered into `mode`.
        let run = |g: usize, mode: &mut JkMode<'_>| {
            let lo = g * group_size;
            let hi = (lo + group_size).min(n_pairs);
            let mut local_count = 0usize;
            for &(s1, s2) in &shell_pairs[lo..hi] {
                if ferric_core::INTERRUPT.load(Ordering::Relaxed) {
                    continue;
                }
                pool.with(|engine| {
                    local_count += scatter_bra_pair(
                        engine, prep, dims, offs, q_table, m_table, x_table, &screen, thresh,
                        d_total, s1, s2, mode, true,
                    );
                });
            }
            computed_quartets.fetch_add(local_count, Ordering::Relaxed);
        };
        let (mut total_j, mut total_k_a, mut total_k_b) = match self.bounds.digest.mode()? {
            JkDigest::Legacy => {
                let mut total_j = Array2::<f64>::zeros((nbf, nbf));
                let mut total_k_a = Array2::<f64>::zeros((nbf, nbf));
                let mut total_k_b = Array2::<f64>::zeros((nbf, nbf));
                crate::reduce::ordered_stream_sum_triple(
                    &mut total_j,
                    &mut total_k_a,
                    &mut total_k_b,
                    n_groups,
                    nbf,
                    band_bytes,
                    |g| {
                        let mut mode = JkMode::new_uhf(nbf, d_a, d_b);
                        run(g, &mut mode);
                        match mode {
                            JkMode::Uhf(j, ka, kb, _, _) => Ok((j, ka, kb)),
                            _ => unreachable!("DirectJK::build_uhf's legacy walk uses JkMode::Uhf"),
                        }
                    },
                )?;
                (total_j, total_k_a, total_k_b)
            }
            JkDigest::Pair => {
                let spec = PairSpec::uhf(nbf, dims, d_total, d_a, d_b, sym);
                let out = spec.sum_groups(self.ctx, n_groups, band_bytes, None, run)?;
                *scales = Some(spec.scales());
                let [k_a, k_b]: [Array2<f64>; 2] = out
                    .k
                    .try_into()
                    .unwrap_or_else(|_| unreachable!("two spin K matrices"));
                (
                    out.j.expect("the open-shell pair digest builds J"),
                    k_a,
                    k_b,
                )
            }
        };

        // MPI: reduce the rank-LOCAL partials, THEN accumulate. Reducing `j`/`k_a`/
        // `k_b` instead would also sum the caller's pre-existing contents, which on
        // the incremental path (`build_uhf_incremental`) hold the previous
        // iteration's already-global J/K — identical on every rank, so an output-
        // buffer Allreduce would multiply them by the world size every iteration.
        // See `reduce::reduce_partial_across_ranks` for the measured failure.
        crate::reduce::reduce_partial_across_ranks(self.ctx, &mut total_j);
        crate::reduce::reduce_partial_across_ranks(self.ctx, &mut total_k_a);
        crate::reduce::reduce_partial_across_ranks(self.ctx, &mut total_k_b);

        *j += &total_j;
        *k_a += &total_k_a;
        *k_b += &total_k_b;

        Ok(computed_quartets.load(Ordering::SeqCst))
    }

    /// The closed-shell walk of [`DirectJK::build`].
    /// The closed-shell walk. `sym` is the pair digest's symmetry scale;
    /// `scales` receives its `max|D|` when the pair digest ran.
    fn build(
        &mut self,
        d: &Array2<f64>,
        j: &mut Array2<f64>,
        k: &mut Array2<f64>,
        sym: crate::quartet_scatter::SymScale<'_>,
        scales: &mut Option<Vec<f64>>,
    ) -> Result<usize, FerricError> {
        use crate::quartet_scatter::{build_d_max_shell, DensityScreen, JkMode, PairSpec};
        use std::sync::atomic::{AtomicUsize, Ordering};

        let profile_t0 = ferric_core::timing::enabled().then(std::time::Instant::now);

        let computed_quartets = AtomicUsize::new(0);

        // Shell-blocked density-max table d_max_shell[(si, sj)] = max |D_μν| over
        // (μ ∈ shell si, ν ∈ shell sj). The Häser-Ahlrichs density-weighted screen
        // uses the max of all six pair maxima (12,34,13,14,23,24) because J contracts
        // D against (λ,σ),(μ,ν) and K against (μ,λ),(μ,σ),(ν,λ),(ν,σ).
        let d_max_shell = build_d_max_shell(self.prep, d);

        // Work list = screened canonical bra pairs (s1,s2), O(nsh²) memory. The
        // ket (s3,s4) loop and the pair-wise density screen —
        //   sqrt(Q_{12}) * sqrt(Q_{34}) * D_max_pair, with
        //   D_max_pair = max(d12, d34, d13, d14, d23, d24)
        // — run inside the parallel group workers below. The previous flat
        // quartet pre-enumeration was a serial O(nsh⁴) loop rebuilt (and its
        // Vec reallocated) EVERY SCF iteration, with unbounded memory in the
        // surviving-quartet count (32 B/entry — GB-scale on large direct
        // jobs); the pair list is the same structure `rhf::build_jk` already
        // uses with this reduction, and moves the screening work onto the
        // rayon workers. Shared verbatim with `build_uhf` via
        // `screened_bra_pairs` (which also hoists the engine pool).
        let shell_pairs = self.screened_bra_pairs(&d_max_shell)?;

        let q_table = &self.bounds.q;
        // CSB `M` table when `[scf] screening = "csb"` selected it; `None`
        // (the default) leaves the hot loop on plain Schwarz, byte-identical.
        //
        // NOT applied to the BRA-PAIR prefilter above, and that is not an
        // omission: the prefilter screens the DIAGONAL quartet (s1,s2|s1,s2),
        // where plain Schwarz is EXACT — Cauchy-Schwarz is an equality there,
        // so Q(s1,s2)^2 == max |(s1 s2|s1 s2)| — and no valid upper bound can
        // be smaller than an exact one. CSB's `min` therefore selects the
        // Schwarz term at every diagonal quartet, making it provably inert on
        // this loop. Same structural fact `qqr.rs` records for QQR (see
        // `tests/qqr_diagonal_noop.rs`), and the same consequence: CSB acts
        // only on the innermost per-quartet test, never on a pair list.
        let m_table = self.bounds.csb_m.as_ref();
        // CSAM `X` table when `[scf] screening = "csam"` selected it. Mutually
        // exclusive with `m_table` by construction; `None` for both is the
        // byte-identical plain-Schwarz default. Like CSB it is NOT applied to the
        // bra-pair prefilter — but for a different reason: CSAM's diagonal
        // refinement factor is exactly 1 (X's diagonal is 1.0 by Cauchy-Schwarz
        // equality), so it too is provably inert on `(s1,s2|s1,s2)`.
        let x_table = self.bounds.csam_x.as_ref();
        let prep = self.prep;
        let nbf = prep.nbasis();
        let pool = self
            .pool
            .as_ref()
            .expect("pool initialized by screened_bra_pairs");

        // Deterministic, memory-bounded reduction (see direct_k / reduce.rs). The
        // old `fold(..).reduce(..)` tree held one J and one K nbf² partial per
        // work-chunk (~2× the direct-K footprint, ~21 GB at 50-atom/aug-cc-pVTZ,
        // 32 threads) and combined them in a worker-count-dependent order. Group
        // the canonical quartet list, fold each group's (J,K) serially, and sum
        // group partials in strict group order — bit-identical across thread
        // counts, live set bounded by a byte-budgeted window of partials.
        // Group partition is a pure function of the quartet list (never the
        // thread count): group boundaries set the floating-point association of
        // the per-group folds, so a thread-dependent partition would break
        // bit-identity across RAYON_NUM_THREADS.
        let n_pairs = shell_pairs.len();
        let group_size = crate::reduce::deterministic_group_size(n_pairs);
        let n_groups = n_pairs.div_ceil(group_size.max(1)).max(1);
        let screen = DensityScreen::SixPair(&d_max_shell);

        let band_bytes = crate::reduce::resolve_band_bytes(self.mem_budget);
        let digest = self.bounds.digest.mode()?;
        let profile = profile_t0.map(|t0| BuildProfile::new(t0, n_pairs, n_groups, digest));
        let job = GroupJob {
            pool,
            prep,
            q_table,
            m_table,
            x_table,
            screen: &screen,
            thresh: self.thresh,
            d,
        };
        let mut band_times = crate::reduce::StreamTimes::default();
        let mut pair_report = None;
        let (mut total_j, mut total_k) = match digest {
            JkDigest::Legacy => {
                let mut total_j = Array2::<f64>::zeros((nbf, nbf));
                let mut total_k = Array2::<f64>::zeros((nbf, nbf));
                crate::reduce::ordered_stream_sum_pair(
                    &mut total_j,
                    &mut total_k,
                    n_groups,
                    nbf,
                    band_bytes,
                    |g| {
                        let lo = g * group_size;
                        let pairs = &shell_pairs[lo..(lo + group_size).min(n_pairs)];
                        let mut mode = JkMode::new_both(nbf);
                        let n = job.run(pairs, profile.as_ref(), &mut mode);
                        computed_quartets.fetch_add(n, Ordering::Relaxed);
                        match mode {
                            JkMode::Both(j, k) => Ok((j, k)),
                            _ => unreachable!("DirectJK::build's legacy walk uses JkMode::Both"),
                        }
                    },
                    profile.as_ref().map(|_| &mut band_times),
                )?;
                (total_j, total_k)
            }
            JkDigest::Pair => {
                // Half partials summed in group order, then `finish` (linear,
                // so it commutes with the rank reduction below).
                let spec = PairSpec::both(nbf, prep.shell_dims(), d, sym);
                let mut out = spec.sum_groups(
                    self.ctx,
                    n_groups,
                    band_bytes,
                    profile.as_ref().map(|_| &mut band_times),
                    |g, mode| {
                        let lo = g * group_size;
                        let pairs = &shell_pairs[lo..(lo + group_size).min(n_pairs)];
                        let n = job.run(pairs, profile.as_ref(), mode);
                        computed_quartets.fetch_add(n, Ordering::Relaxed);
                    },
                )?;
                *scales = Some(spec.scales());
                let k = out.k.pop().expect("the J+K pair digest builds K");
                let j = out.j.take().expect("the J+K pair digest builds J");
                pair_report = Some(out);
                (j, k)
            }
        };

        // MPI: reduce the rank-LOCAL partials, THEN accumulate — NOT the other way
        // round. `build_incremental` delegates here with `j`/`k` already holding
        // the previous iteration's global `J(D_last)`/`K(D_last)`, so Allreducing
        // the output buffers would return `N·J(D_last) + Σ_r ΔJ_r`. See
        // `reduce::reduce_partial_across_ranks`.
        crate::reduce::reduce_partial_across_ranks(self.ctx, &mut total_j);
        crate::reduce::reduce_partial_across_ranks(self.ctx, &mut total_k);

        *j += &total_j;
        *k += &total_k;

        let computed = computed_quartets.load(Ordering::SeqCst);
        if let Some(p) = profile {
            p.report(
                nbf,
                self.prep.nshells(),
                computed,
                &band_times,
                pair_report.as_ref(),
            );
        }
        Ok(computed)
    }
}

/// The inputs every reduction group of [`DirectJK::build`] shares.
struct GroupJob<'g> {
    pool: &'g crate::engine_pool::EnginePool,
    prep: &'g PreparedBasis,
    q_table: &'g Array2<f64>,
    m_table: Option<&'g Array2<f64>>,
    x_table: Option<&'g Array2<f64>>,
    screen: &'g crate::quartet_scatter::DensityScreen<'g>,
    thresh: f64,
    d: &'g Array2<f64>,
}

impl GroupJob<'_> {
    /// The bra pairs `pairs` scattered into `mode` (fresh `nbf × nbf`
    /// partials: `JkMode::Both` for the legacy digest, a `PairSpec`'s mode
    /// for the pair digest); returns the number of quartets computed.
    /// With `profile`, the group's stage times are recorded into it.
    fn run(
        &self,
        pairs: &[(usize, usize)],
        profile: Option<&BuildProfile>,
        mode: &mut crate::quartet_scatter::JkMode<'_>,
    ) -> usize {
        use crate::quartet_scatter::{scatter_bra_pair, scatter_bra_pair_probed};
        use std::sync::atomic::Ordering;
        let t_group = std::time::Instant::now();
        let (prep, d) = (self.prep, self.d);
        let (dims, offs) = (prep.shell_dims(), prep.shell_offsets());
        let mut count = 0usize;
        let mut times = ScatterTimes::default();
        for &(s1, s2) in pairs {
            if ferric_core::INTERRUPT.load(Ordering::Relaxed) {
                continue;
            }
            let (q, m, x, scr, t) = (
                self.q_table,
                self.m_table,
                self.x_table,
                self.screen,
                self.thresh,
            );
            self.pool.with(|engine| {
                count += match profile {
                    Some(_) => scatter_bra_pair_probed::<true>(
                        engine, prep, dims, offs, q, m, x, scr, t, d, s1, s2, mode, true,
                        &mut times,
                    ),
                    None => scatter_bra_pair(
                        engine, prep, dims, offs, q, m, x, scr, t, d, s1, s2, mode, true,
                    ),
                };
            });
        }
        if let Some(p) = profile {
            p.record_group(t_group, &times);
        }
        count
    }
}

/// `FERRIC_TIMING` profile of one closed-shell [`DirectJK::build`]: per-group
/// wall times and stage sums, printed as one block of `[timing]` lines.
///
/// The phases it separates, and how each is measured:
///
/// * **setup** — density-max table, bra-pair list and engine pool, from entry
///   to the first band;
/// * **libint** — inside `Engine::compute_quartet` (summed over threads);
/// * **digest** — contracting each returned block into J and K (for the
///   pair digest also its per-bra-pair `begin`/`end`); the pair digest's
///   serial density preparation and `finish` are on a line of their own
///   (in the wall's `setup`, which is everything outside the groups);
/// * **loop+screen** — the rest of each group's wall time: the `(s3, s4)`
///   candidate loop, the Schwarz × density test, and the group's two `nbf²`
///   partial allocations;
/// * **folds** — the in-order folds of the group partials into J and K (and
///   their release), run by whichever worker finds the next group ready while
///   the others keep computing;
/// * **outside groups** — `threads × (groups and folds wall) − Σ group wall`:
///   worker time spent folding, waiting for the fold window, or idle at the
///   end (load imbalance).
///
/// The per-quartet clock reads are themselves a cost (two reads per computed
/// quartet); compare the profiled wall with an unprofiled one before reading
/// small phases.
struct BuildProfile {
    t0: std::time::Instant,
    n_pairs: usize,
    n_groups: usize,
    digest: JkDigest,
    /// `(rayon thread, group wall ns, stage times)` per group.
    groups: std::sync::Mutex<Vec<(usize, u64, ScatterTimes)>>,
}

impl BuildProfile {
    fn new(t0: std::time::Instant, n_pairs: usize, n_groups: usize, digest: JkDigest) -> Self {
        BuildProfile {
            t0,
            n_pairs,
            n_groups,
            digest,
            groups: std::sync::Mutex::new(Vec::with_capacity(n_groups)),
        }
    }

    fn record_group(&self, t: std::time::Instant, times: &ScatterTimes) {
        let wall = t.elapsed().as_nanos() as u64;
        let tid = rayon::current_thread_index().unwrap_or(usize::MAX);
        if let Ok(mut g) = self.groups.lock() {
            g.push((tid, wall, *times));
        }
    }

    fn report(
        &self,
        nbf: usize,
        nsh: usize,
        computed: usize,
        bands: &crate::reduce::StreamTimes,
        pair: Option<&crate::quartet_scatter::PairOut>,
    ) {
        let wall = self.t0.elapsed().as_secs_f64();
        let groups = self.groups.lock().map(|g| g.clone()).unwrap_or_default();
        let threads = rayon::current_num_threads();
        let mut sum = ScatterTimes::default();
        let mut group_ns = 0u64;
        let mut busy = std::collections::BTreeMap::<usize, u64>::new();
        for (tid, w, t) in &groups {
            sum.add(t);
            group_ns += w;
            *busy.entry(*tid).or_default() += w;
        }
        let s = |ns: u64| ns as f64 * 1e-9;
        let produce = s(bands.wall_ns);
        let fold = s(bands.fold_ns);
        let setup = wall - produce;
        let idle = threads as f64 * produce - s(group_ns);
        let loop_screen = s(group_ns) - s(sum.libint_ns) - s(sum.digest_ns);
        let (bmin, bmax) = busy
            .values()
            .fold((u64::MAX, 0u64), |(a, b), &v| (a.min(v), b.max(v)));
        let kernel = pair.map_or(String::new(), |p| format!(" (K kernel {})", p.kernel));
        eprintln!(
            "[timing] direct-jk build: nbf {nbf} nsh {nsh} bra pairs {} groups {} \
             threads {threads} digest {}{kernel}",
            self.n_pairs, self.n_groups, self.digest
        );
        eprintln!(
            "[timing] direct-jk   wall {wall:.3} s = setup {setup:.3} + groups and folds \
             {produce:.3} (in-order folds {fold:.3}, overlapping the groups)"
        );
        eprintln!(
            "[timing] direct-jk   thread-seconds: groups {:.3} = libint {:.3} + digest {:.3} \
             + loop/screen/alloc {loop_screen:.3}; outside groups (folds, waits) {idle:.3}",
            s(group_ns),
            s(sum.libint_ns),
            s(sum.digest_ns),
        );
        if let Some(p) = pair {
            // Serial, outside the groups: in `setup` (the preparation) and
            // after the groups (the finish), not in the digest above.
            eprintln!(
                "[timing] direct-jk   pair digest serial: density preparation {:.4} s + \
                 finish {:.4} s (the per-bra-pair begin/end are in the digest)",
                p.prep_s, p.finish_s
            );
        }
        eprintln!(
            "[timing] direct-jk   per-thread busy min {:.3} max {:.3} s over {} threads; \
             candidates {} survivors {} computed {computed}",
            s(if bmin == u64::MAX { 0 } else { bmin }),
            s(bmax),
            busy.len(),
            sum.candidates,
            sum.survivors,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::direct_j::DirectJ;
    use crate::direct_k::DirectK;
    use crate::fock::{JBuilder, KBuilder};
    use ferric_core::basis;
    use ferric_core::mol::Molecule;
    use ferric_integrals::operator::Operator;

    /// Regression guard for the grouped deterministic reduction: J and K from
    /// the direct builders (DirectJK combined, DirectJ, DirectK) must be
    /// bit-identical regardless of the rayon worker count. The old
    /// `fold(..).reduce(..)` trees combined per-chunk partials in a
    /// worker-count-dependent order, so J/K (and the SCF energy) drifted ~µHa
    /// with RAYON_NUM_THREADS. The group partition is a pure function of the
    /// work list and group partials are summed in ascending group order, so the
    /// result cannot depend on the thread count.
    /// Two open-shell test densities (α ≠ β, both symmetric) plus their sum,
    /// on H2O/cc-pVDZ. Shared by the `build_uhf` anchors below.
    fn uhf_test_densities(n: usize) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
        let mut d_a = Array2::<f64>::zeros((n, n));
        let mut d_b = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                d_a[(i, j)] = 0.01 * ((i * 7 + j * 3) % 11) as f64;
                // Deliberately DIFFERENT structure, so any code path that
                // silently used one spin density for both K matrices (or the
                // total for either) is caught.
                d_b[(i, j)] = 0.007 * ((i * 5 + j * 13) % 9) as f64 - 0.002;
            }
        }
        let d_a = 0.5 * (&d_a + &d_a.t());
        let d_b = 0.5 * (&d_b + &d_b.t());
        let d_total = &d_a + &d_b;
        (d_a, d_b, d_total)
    }

    /// EXACTNESS ANCHOR for the combined open-shell build.
    ///
    /// `build_uhf` must reproduce, to a tight numerical tolerance, exactly what
    /// the three-pass path it replaces produced: `J` from `DirectJ(D_total)`,
    /// `K_α` from `DirectK(D_α)`, `K_β` from `DirectK(D_β)`.
    ///
    /// The tolerance is not zero because this is deliberately NOT a
    /// bit-identity claim: the combined build screens on the six-pairwise
    /// spin-sum shell table while `DirectJ`/`DirectK` screen on a single global
    /// `max|D|` scalar, so the two admit different quartet SETS (the combined
    /// one is TIGHTER — it may legitimately drop quartets the loose screen
    /// kept). Screening is an approximation controlled by `thresh`, so the
    /// agreement bar is the screening threshold, not the f64 floor. Run at a
    /// very tight `thresh` (1e-14) both paths converge to the exact J/K.
    ///
    /// ARTIFACT HYPOTHESIS: if the α/β densities were crossed, or one spin
    /// density were used for both K buffers, K_α and K_β would come out equal
    /// (or swapped) — the asymmetric `d_a`/`d_b` above make that a loud
    /// failure rather than a silent pass.
    #[test]
    fn build_uhf_matches_three_pass_directj_plus_two_directk() {
        let mol =
            Molecule::parse_xyz("3\nH2O\nO 0 0 0\nH 0 0 0.96\nH 0.93 0 -0.26\n", 0, 1).unwrap();
        let bs = basis::bundled("cc-pvdz").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let n = prep.nbasis();
        let (d_a, d_b, d_total) = uhf_test_densities(n);
        let ctx = ParallelContext::default();
        let thresh = 1e-14;

        // Reference: the three separate passes the open-shell solvers used to run.
        let mut j_ref = Array2::zeros((n, n));
        let mut ka_ref = Array2::zeros((n, n));
        let mut kb_ref = Array2::zeros((n, n));
        let mut dj = DirectJ::new(&ctx, &prep, &bounds, thresh, usize::MAX);
        <DirectJ as JBuilder>::build(&mut dj, &d_total, &mut j_ref).unwrap();
        let mut dk = DirectK::new(&ctx, &prep, &bounds, thresh, usize::MAX);
        <DirectK as KBuilder>::build(&mut dk, &d_a, &mut ka_ref).unwrap();
        <DirectK as KBuilder>::build(&mut dk, &d_b, &mut kb_ref).unwrap();

        // Combined: one pass.
        let mut j = Array2::zeros((n, n));
        let mut ka = Array2::zeros((n, n));
        let mut kb = Array2::zeros((n, n));
        let mut djk = DirectJK::new(&ctx, &prep, &bounds, thresh, usize::MAX);
        djk.build_uhf(&d_total, &d_a, &d_b, &mut j, &mut ka, &mut kb)
            .unwrap();

        let max_abs_diff = |x: &Array2<f64>, y: &Array2<f64>| -> f64 {
            (x - y).iter().map(|v| v.abs()).fold(0.0f64, f64::max)
        };
        let tol = 1e-10;
        assert!(
            max_abs_diff(&j, &j_ref) < tol,
            "combined J differs from DirectJ(D_total) by {}",
            max_abs_diff(&j, &j_ref)
        );
        assert!(
            max_abs_diff(&ka, &ka_ref) < tol,
            "combined K_alpha differs from DirectK(D_alpha) by {}",
            max_abs_diff(&ka, &ka_ref)
        );
        assert!(
            max_abs_diff(&kb, &kb_ref) < tol,
            "combined K_beta differs from DirectK(D_beta) by {}",
            max_abs_diff(&kb, &kb_ref)
        );
        // The two spin exchange matrices must actually DIFFER — otherwise the
        // test above would pass just as well for a builder that ignored one of
        // the two densities.
        assert!(
            max_abs_diff(&ka, &kb) > 1e-3,
            "test densities failed to produce distinct K_alpha/K_beta — this \
             test cannot detect a crossed-density bug"
        );
    }

    /// EXACTNESS ANCHOR (trivial limit): in the CLOSED-SHELL limit `D_α = D_β`,
    /// the open-shell build must collapse onto the closed-shell one — both spin
    /// exchange matrices equal to `K(D_α)`, and `J` equal to `J(2·D_α)`.
    ///
    /// This is the "vacuous approximation" anchor the experimental protocol
    /// requires: it fixes the *relationship* between the new path and the
    /// long-trusted one, independent of any screening difference, and it is
    /// bit-exact because with `D_α = D_β` the spin-sum screen table equals
    /// `build_d_max_shell(2·D_α)`'s and the two builds traverse the same
    /// quartets in the same order.
    #[test]
    fn build_uhf_matches_closed_shell_build_when_spins_are_equal() {
        let mol =
            Molecule::parse_xyz("3\nH2O\nO 0 0 0\nH 0 0 0.96\nH 0.93 0 -0.26\n", 0, 1).unwrap();
        let bs = basis::bundled("cc-pvdz").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let n = prep.nbasis();
        let (d_half, _, _) = uhf_test_densities(n);
        let d_total = &d_half + &d_half;
        let ctx = ParallelContext::default();
        let thresh = 1e-12;

        // Closed-shell reference: J(D_total) and K(D_half) from `build`.
        // (`build` uses ONE density for both, so call it twice with the density
        // each matrix actually needs and keep only the relevant output.)
        let mut j_ref = Array2::zeros((n, n));
        let mut k_scratch = Array2::zeros((n, n));
        let mut djk_ref = DirectJK::new(&ctx, &prep, &bounds, thresh, usize::MAX);
        djk_ref.build(&d_total, &mut j_ref, &mut k_scratch).unwrap();
        let mut j_scratch = Array2::zeros((n, n));
        let mut k_ref = Array2::zeros((n, n));
        djk_ref.build(&d_half, &mut j_scratch, &mut k_ref).unwrap();

        let mut j = Array2::zeros((n, n));
        let mut ka = Array2::zeros((n, n));
        let mut kb = Array2::zeros((n, n));
        let mut djk = DirectJK::new(&ctx, &prep, &bounds, thresh, usize::MAX);
        djk.build_uhf(&d_total, &d_half, &d_half, &mut j, &mut ka, &mut kb)
            .unwrap();

        assert_eq!(
            ka, kb,
            "equal spin densities must give bit-identical K_alpha/K_beta"
        );
        let max_abs_diff = |x: &Array2<f64>, y: &Array2<f64>| -> f64 {
            (x - y).iter().map(|v| v.abs()).fold(0.0f64, f64::max)
        };
        assert!(
            max_abs_diff(&j, &j_ref) < 1e-10,
            "closed-shell limit J mismatch: {}",
            max_abs_diff(&j, &j_ref)
        );
        assert!(
            max_abs_diff(&ka, &k_ref) < 1e-10,
            "closed-shell limit K mismatch: {}",
            max_abs_diff(&ka, &k_ref)
        );
    }

    /// The incremental open-shell update must reproduce a from-scratch build.
    ///
    /// `build_uhf(D1) ==  build_uhf(D0) then build_uhf_incremental(D1 - D0)`,
    /// to the f64 reassociation floor. The deltas here move the two spin
    /// channels in OPPOSITE directions (ΔD_α > 0, ΔD_β < 0) so that ΔD_total
    /// is much smaller than either channel — precisely the configuration in
    /// which screening on ΔD_total instead of |ΔD_α| + |ΔD_β| would drop real
    /// contributions to the spin exchange matrices.
    #[test]
    fn build_uhf_incremental_matches_full_rebuild_with_opposing_spin_deltas() {
        let mol =
            Molecule::parse_xyz("3\nH2O\nO 0 0 0\nH 0 0 0.96\nH 0.93 0 -0.26\n", 0, 1).unwrap();
        let bs = basis::bundled("cc-pvdz").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let n = prep.nbasis();
        let (a0, b0, _) = uhf_test_densities(n);
        let ctx = ParallelContext::default();
        let thresh = 1e-12;

        // Opposing spin deltas: the total barely moves, each channel moves a lot.
        let mut bump = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                bump[(i, j)] = 0.004 * (((i * 11 + j * 17) % 7) as f64 - 3.0);
            }
        }
        let bump = 0.5 * (&bump + &bump.t());
        let a1 = &a0 + &bump;
        let b1 = &b0 - &bump; // ΔD_total == 0 exactly by construction
        let t0 = &a0 + &b0;
        let t1 = &a1 + &b1;

        // From-scratch build at the new density.
        let mut j_full = Array2::zeros((n, n));
        let mut ka_full = Array2::zeros((n, n));
        let mut kb_full = Array2::zeros((n, n));
        let mut djk_f = DirectJK::new(&ctx, &prep, &bounds, thresh, usize::MAX);
        djk_f
            .build_uhf(&t1, &a1, &b1, &mut j_full, &mut ka_full, &mut kb_full)
            .unwrap();

        // Build at the old density, then accumulate the delta.
        let mut j_inc = Array2::zeros((n, n));
        let mut ka_inc = Array2::zeros((n, n));
        let mut kb_inc = Array2::zeros((n, n));
        let mut djk_i = DirectJK::new(&ctx, &prep, &bounds, thresh, usize::MAX);
        djk_i
            .build_uhf(&t0, &a0, &b0, &mut j_inc, &mut ka_inc, &mut kb_inc)
            .unwrap();
        let da = &a1 - &a0;
        let db = &b1 - &b0;
        let dt = &t1 - &t0;
        djk_i
            .build_uhf_incremental(&dt, &da, &db, &mut j_inc, &mut ka_inc, &mut kb_inc)
            .unwrap();

        let max_abs_diff = |x: &Array2<f64>, y: &Array2<f64>| -> f64 {
            (x - y).iter().map(|v| v.abs()).fold(0.0f64, f64::max)
        };
        let tol = 1e-10;
        assert!(
            max_abs_diff(&j_inc, &j_full) < tol,
            "incremental J drift {}",
            max_abs_diff(&j_inc, &j_full)
        );
        assert!(
            max_abs_diff(&ka_inc, &ka_full) < tol,
            "incremental K_alpha drift {} — a ΔD_total screening key would fail \
             HERE, where ΔD_total is zero but each spin channel moved",
            max_abs_diff(&ka_inc, &ka_full)
        );
        assert!(
            max_abs_diff(&kb_inc, &kb_full) < tol,
            "incremental K_beta drift {}",
            max_abs_diff(&kb_inc, &kb_full)
        );
        // Confirm the test is actually in the regime it claims: the spin deltas
        // must be large while the total delta is negligible.
        let max_dt = dt.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
        let max_da = da.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
        assert!(
            max_da > 1e-3 && max_dt < 1e-12,
            "test is not in the opposing-delta regime: max|dD_a|={max_da}, max|dD_total|={max_dt}"
        );
    }

    /// Thread-count bit-identity for the new open-shell path, matching the
    /// existing guarantee for `build`/`DirectJ`/`DirectK`.
    #[test]
    fn build_uhf_bit_identical_across_thread_counts() {
        let mol =
            Molecule::parse_xyz("3\nH2O\nO 0 0 0\nH 0 0 0.96\nH 0.93 0 -0.26\n", 0, 1).unwrap();
        let bs = basis::bundled("cc-pvdz").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let n = prep.nbasis();
        let (d_a, d_b, d_total) = uhf_test_densities(n);

        let run = |threads: usize| -> (Array2<f64>, Array2<f64>, Array2<f64>) {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| {
                let ctx = ParallelContext::default();
                let mut j = Array2::zeros((n, n));
                let mut ka = Array2::zeros((n, n));
                let mut kb = Array2::zeros((n, n));
                let mut djk = DirectJK::new(&ctx, &prep, &bounds, 1e-14, usize::MAX);
                djk.build_uhf(&d_total, &d_a, &d_b, &mut j, &mut ka, &mut kb)
                    .unwrap();
                (j, ka, kb)
            })
        };
        let r1 = run(1);
        let r4 = run(4);
        assert_eq!(
            r1.0, r4.0,
            "combined UHF J must be bit-identical across thread counts"
        );
        assert_eq!(
            r1.1, r4.1,
            "combined UHF K_alpha must be bit-identical across thread counts"
        );
        assert_eq!(
            r1.2, r4.2,
            "combined UHF K_beta must be bit-identical across thread counts"
        );
    }

    #[test]
    fn direct_builders_bit_identical_across_thread_counts() {
        let mol =
            Molecule::parse_xyz("3\nH2O\nO 0 0 0\nH 0 0 0.96\nH 0.93 0 -0.26\n", 0, 1).unwrap();
        let bs = basis::bundled("cc-pvdz").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let n = prep.nbasis();

        // Dense symmetric density so every quartet contributes.
        let mut d = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                d[(i, j)] = 0.01 * ((i * 7 + j * 3) % 11) as f64;
            }
        }
        let d = 0.5 * (&d + &d.t());

        let build_all = |threads: usize| -> (Array2<f64>, Array2<f64>, Array2<f64>, Array2<f64>) {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| {
                let ctx = ParallelContext::default();
                let mut j_jk = Array2::zeros((n, n));
                let mut k_jk = Array2::zeros((n, n));
                let mut djk = DirectJK::new(&ctx, &prep, &bounds, 1e-14, usize::MAX);
                djk.build(&d, &mut j_jk, &mut k_jk).unwrap();

                let mut j_only = Array2::zeros((n, n));
                let mut dj = DirectJ::new(&ctx, &prep, &bounds, 1e-14, usize::MAX);
                <DirectJ as JBuilder>::build(&mut dj, &d, &mut j_only).unwrap();

                let mut k_only = Array2::zeros((n, n));
                let mut dk = DirectK::new(&ctx, &prep, &bounds, 1e-14, usize::MAX);
                <DirectK as KBuilder>::build(&mut dk, &d, &mut k_only).unwrap();

                (j_jk, k_jk, j_only, k_only)
            })
        };

        let r1 = build_all(1);
        let r4 = build_all(4);
        assert_eq!(
            r1.0, r4.0,
            "DirectJK J must be bit-identical across thread counts"
        );
        assert_eq!(
            r1.1, r4.1,
            "DirectJK K must be bit-identical across thread counts"
        );
        assert_eq!(
            r1.2, r4.2,
            "DirectJ J must be bit-identical across thread counts"
        );
        assert_eq!(
            r1.3, r4.3,
            "DirectK K must be bit-identical across thread counts"
        );
    }
}
