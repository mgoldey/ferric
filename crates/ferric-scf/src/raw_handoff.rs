//! Hand the SCF's raw RI-JK tensor `(P|μν)` to the analytic gradient that
//! follows it, so the gradient's `dfg:3c-pass` does not regenerate it.
//!
//! When RI-J and RI-K share one auxiliary basis (the default), the SCF builds
//! the raw tensor once: DF-K dresses from it and RI-J keeps it, unscreened
//! (a source shared with DF-K is never pair-screened, see
//! `fock_assembly::build_df_jk`) and in f64 on a geometry task. The gradient
//! contracts exactly these integrals with the converged density, and
//! regenerating them cost ~1.7 s of a ~10.5 s gradient (n-dodecane/def2-TZVP
//! B3LYP, 2026-11-10).
//!
//! The hand-off is NOT part of [`ScfResult`](crate::result::ScfResult): a
//! result is public, cloned and returned to Python, and would pin the tensor
//! for as long as any caller holds it. Instead a driver that runs an SCF and
//! then its gradient installs a fresh [`RawHandoff`] in the config it hands
//! both ([`with_fresh_handoff`]); the solver deposits the tensor at its
//! converged exit, and the gradient dispatchers take it (`with_kept_raw`)
//! and drop it when the gradient returns. A driver that never asks for a
//! gradient never installs one, so nothing is held past the SCF.
//!
//! Kept only when: J and K share the aux basis, one rank, the tensor is held
//! whole in memory in f64 (unpacked or packed tier), and the tensor plus the
//! gradient's metric and exchange intermediates fit the three-index budget.
//! Otherwise the gradient recomputes exactly as before; under `FERRIC_TIMING`
//! both sides say which happened and why. Before using a kept tensor the
//! gradient checks that its aux basis, operator and geometry match
//! ([`KeptRaw::check`]).

use crate::result::DfJkRoute;
use crate::rhf::RhfConfig;
use crate::split_rij::RiJ;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_integrals::three_index_source::ThreeIndexSource;
use std::cell::RefCell;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};

/// A raw `(P|μν)` tensor kept from an SCF, with what it was built for.
///
/// Holding it keeps its memory-pool charge (the source owns it), so the
/// bytes stay accounted for until the gradient drops it.
pub struct KeptRaw {
    raw: ThreeIndexSource,
    aux: String,
    op: Operator,
    geometry_hash: u64,
}

impl std::fmt::Debug for KeptRaw {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KeptRaw")
            .field("aux", &self.aux)
            .field("op", &self.op)
            .field("geometry_hash", &self.geometry_hash)
            .field("bytes", &self.bytes())
            .finish()
    }
}

impl KeptRaw {
    /// The auxiliary basis name the tensor was built with.
    pub fn aux(&self) -> &str {
        &self.aux
    }

    /// The operator the tensor was built with.
    pub fn op(&self) -> Operator {
        self.op
    }

    /// Resident bytes of the tensor.
    pub fn bytes(&self) -> usize {
        self.raw.resident_f64_bytes().unwrap_or(0)
    }

    /// The tensor itself.
    pub(crate) fn source(&self) -> &ThreeIndexSource {
        &self.raw
    }

    /// `Ok` when this tensor holds `(P|μν)` for `aux`/`op` at `mol` in `obs`
    /// and `dfbs`; `Err(why)` otherwise (the caller recomputes).
    pub fn check(
        &self,
        mol: &Molecule,
        obs: &PreparedBasis,
        dfbs: &PreparedBasis,
        aux: &str,
        op: Operator,
    ) -> Result<(), &'static str> {
        if self.aux != aux {
            return Err("auxiliary basis differs");
        }
        if self.op != op {
            return Err("operator differs");
        }
        let full = (0, dfbs.nbasis());
        if self.raw.nao() != obs.nbasis() || self.raw.naux() != dfbs.nbasis() {
            return Err("tensor shape differs");
        }
        if self.raw.band() != full {
            return Err("tensor holds an aux band, not the full range");
        }
        if self.geometry_hash != geometry_hash(mol, obs) {
            return Err("geometry or orbital basis differs");
        }
        Ok(())
    }
}

/// A shared slot through which one SCF hands its raw tensor to the gradient
/// that follows it (see the module docs). Clones share the slot.
#[derive(Clone, Default)]
pub struct RawHandoff(Arc<Mutex<Option<KeptRaw>>>);

impl std::fmt::Debug for RawHandoff {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RawHandoff")
            .field("filled", &self.is_filled())
            .finish()
    }
}

impl RawHandoff {
    /// An empty slot.
    pub fn new() -> Self {
        Self::default()
    }

    /// Take the kept tensor, leaving the slot empty.
    pub fn take(&self) -> Option<KeptRaw> {
        self.lock().take()
    }

    /// Does the slot hold a tensor?
    pub fn is_filled(&self) -> bool {
        self.lock().is_some()
    }

    /// Put `kept` in the slot, dropping whatever it held. Test hook (the
    /// solvers fill it themselves).
    #[doc(hidden)]
    pub fn put_for_test(&self, kept: KeptRaw) {
        *self.lock() = Some(kept);
    }

    fn put(&self, kept: KeptRaw) {
        *self.lock() = Some(kept);
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Option<KeptRaw>> {
        // A poisoned slot only means a panic elsewhere; its content is whole.
        self.0.lock().unwrap_or_else(|e| e.into_inner())
    }
}

/// `config` with a fresh, empty [`RawHandoff`]: what a driver that runs an
/// SCF and then its gradient hands both (one per evaluation, so a tensor
/// that is never taken is dropped with the config).
pub fn with_fresh_handoff(config: &RhfConfig) -> RhfConfig {
    RhfConfig {
        raw_handoff: Some(RawHandoff::new()),
        ..config.clone()
    }
}

/// `config` with no hand-off: for a sub-solve inside a solver (the
/// `guess_basis` secondary SCF, stability-descent candidates), whose tensor
/// must never reach the slot the outer SCF's gradient reads.
pub(crate) fn without_handoff(config: &RhfConfig) -> RhfConfig {
    RhfConfig {
        raw_handoff: None,
        ..config.clone()
    }
}

/// Empty `config`'s slot (dropping any tensor in it). Every solver calls this
/// before it builds its own RI-JK tensor, so a tensor from an earlier solve
/// through the same config is never held beside the new one, and never
/// outlives an SCF that does not keep its own.
pub(crate) fn clear_slot(config: &RhfConfig) {
    if let Some(h) = config.raw_handoff.as_ref() {
        drop(h.take());
    }
}

/// Identity of a geometry in an orbital basis: atomic numbers, ghost flags,
/// ECP cores, the exact coordinate bits, and the basis name and size.
pub fn geometry_hash(mol: &Molecule, obs: &PreparedBasis) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    mol.atoms.len().hash(&mut h);
    for a in &mol.atoms {
        a.z.hash(&mut h);
        a.ghost.hash(&mut h);
        a.n_core_ecp.hash(&mut h);
        a.x.to_bits().hash(&mut h);
        a.y.to_bits().hash(&mut h);
        a.zpos.to_bits().hash(&mut h);
    }
    obs.basis_set().name.hash(&mut h);
    obs.nbasis().hash(&mut h);
    h.finish()
}

/// Under `FERRIC_TIMING`, one line saying what happened to the raw tensor.
pub(crate) fn note(msg: &str) {
    if ferric_core::timing::enabled() {
        eprintln!("[timing] dfg:raw-handoff {msg}");
    }
}

/// Bytes the DF-JK gradient holds beside the three-index tensor (its memory
/// plane in `df_gradient`): five `(naux, naux)` metric matrices and two
/// `(naux, m²)` matrices per exchange channel of rank `m`. (A packed kept
/// tensor also needs one unpacked block, `df_gradient::pass_block_bytes`.)
#[doc(hidden)]
pub fn gradient_intermediate_bytes(naux: usize, ranks: &[usize]) -> usize {
    let k: usize = ranks
        .iter()
        .map(|&m| {
            2usize
                .saturating_mul(naux)
                .saturating_mul(m * m)
                .saturating_mul(8)
        })
        .fold(0usize, usize::saturating_add);
    k.saturating_add(
        5usize
            .saturating_mul(naux)
            .saturating_mul(naux)
            .saturating_mul(8),
    )
}

/// At a converged SCF exit: move RI-J's raw tensor into `config`'s hand-off
/// when one is installed and the tensor qualifies (module docs). `occ` are
/// the exchange channels' occupied counts (closed shell: `[nocc]`; open
/// shell: `[nα, nβ]`), for the gradient's memory estimate. Leaves `df_j`
/// empty when it keeps the tensor; the SCF must not build J afterwards.
pub(crate) fn keep_for_gradient(
    config: &RhfConfig,
    ctx: &ParallelContext,
    mol: &Molecule,
    prep: &PreparedBasis,
    df_j: &mut Option<RiJ<'_>>,
    route: Option<&DfJkRoute>,
    occ: &[usize],
) {
    let Some(handoff) = config.raw_handoff.as_ref() else {
        return;
    };
    // Whatever the slot held is stale now: this SCF's tensor replaces it, or
    // nothing does.
    drop(handoff.take());
    match kept_raw(ctx, mol, prep, df_j, route, occ) {
        Ok(kept) => {
            note(&format!(
                "kept: {:.2} GB raw tensor ({}) handed to the gradient",
                kept.bytes() as f64 / 1e9,
                kept.aux
            ));
            handoff.put(kept);
        }
        Err(why) => note(&format!("not kept ({why}): the gradient recomputes it")),
    }
}

fn kept_raw(
    ctx: &ParallelContext,
    mol: &Molecule,
    prep: &PreparedBasis,
    df_j: &mut Option<RiJ<'_>>,
    route: Option<&DfJkRoute>,
    occ: &[usize],
) -> Result<KeptRaw, &'static str> {
    let route = route.ok_or("nothing density-fitted")?;
    let aux = match (route.j_aux.as_deref(), route.k_aux.as_deref()) {
        (Some(j), Some(k)) if j == k => j.to_owned(),
        _ => return Err("RI-J and RI-K do not share one aux basis"),
    };
    if ctx.size > 1 {
        return Err("several ranks");
    }
    let raw_bytes = match df_j.as_ref() {
        Some(RiJ::Stored(j)) if j.precision() == crate::df_j::RiJPrecision::F64 => j
            .resident_f64_bytes()
            .ok_or("the tensor is not held whole in memory in f64")?,
        Some(RiJ::Stored(_)) => return Err("the RI-J tensor is stored in f32"),
        Some(RiJ::Split(_)) => return Err("Split-RI-J holds no tensor"),
        None => return Err("no RI-J builder"),
    };
    let (naux, packed) = match df_j.as_ref() {
        Some(RiJ::Stored(j)) => (j.naux(), j.is_packed_raw()),
        _ => unreachable!("checked above"),
    };
    // A packed tensor is read by unpacking one block at a time.
    let block = if packed {
        crate::df_gradient::pass_block_bytes_upper_bound(prep.nbasis(), naux, route.budget_bytes)
    } else {
        0
    };
    let held = raw_bytes
        .saturating_add(gradient_intermediate_bytes(naux, occ))
        .saturating_add(block);
    if held > route.budget_bytes {
        return Err("tensor and gradient intermediates exceed the three-index budget");
    }
    let Some(RiJ::Stored(j)) = df_j.take() else {
        unreachable!("checked above")
    };
    Ok(KeptRaw {
        raw: j.into_source(),
        aux,
        op: route.op,
        geometry_hash: geometry_hash(mol, prep),
    })
}

thread_local! {
    /// The kept tensor of the gradient running on this thread
    /// ([`with_kept_raw`]); taken by the first fitted pass that can use it.
    static ACTIVE: RefCell<Option<KeptRaw>> = const { RefCell::new(None) };
}

/// Restores [`ACTIVE`] when the gradient returns (or unwinds), dropping a
/// tensor no pass took.
struct ActiveGuard(Option<KeptRaw>);

impl Drop for ActiveGuard {
    fn drop(&mut self) {
        let prev = self.0.take();
        ACTIVE.with(|a| *a.borrow_mut() = prev);
    }
}

/// Run the gradient `f` with the tensor `config`'s SCF kept (if any) made
/// available to its fitted pass ([`take_active`]). The tensor is dropped
/// when `f` returns, used or not.
///
/// A scoped thread-local rather than a parameter because the KS and HF
/// gradient entry points are public, many and deep; the fitted pass runs on
/// the calling thread (no `rayon::join` around it). A pass on another thread
/// would simply not see it and recompute, which is the correct fallback.
///
/// Without a kept tensor the scope still shadows any outer one with `None`,
/// so a nested gradient that has no hand-off of its own cannot consume it.
pub(crate) fn with_kept_raw<T>(config: &RhfConfig, f: impl FnOnce() -> T) -> T {
    let kept = config.raw_handoff.as_ref().and_then(RawHandoff::take);
    let _guard = ActiveGuard(ACTIVE.with(|a| a.replace(kept)));
    f()
}

/// The kept tensor installed by [`with_kept_raw`] on this thread, taken only
/// when it was built for `aux`/`op` at `mol` in `obs`/`dfbs`
/// ([`KeptRaw::check`]); otherwise it is left in place (and reported), so a
/// pass that cannot use it does not drop a tensor another pass could.
pub(crate) fn take_active(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    aux: &str,
    op: Operator,
) -> Option<KeptRaw> {
    ACTIVE.with(|a| {
        let mut slot = a.borrow_mut();
        match slot.as_ref().map(|k| k.check(mol, obs, dfbs, aux, op)) {
            Some(Ok(())) => slot.take(),
            Some(Err(why)) => {
                note(&format!("kept tensor not used ({why}): recomputing"));
                None
            }
            None => None,
        }
    })
}
