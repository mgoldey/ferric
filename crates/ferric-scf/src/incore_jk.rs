//! In-core four-centre J and K: the AO electron-repulsion integrals of one
//! geometry held in memory, and J/K of any number of densities by GEMM.
//!
//! # Why
//!
//! A direct J/K build recomputes every significant shell quartet on every
//! call. Below about 200 basis functions the whole integral set fits in a few
//! GB, and contracting it with a density is memory-bandwidth work: on benzene /
//! cc-pVDZ (114 functions, 0.34 GB store, 12 rayon threads) one direct J/K
//! build takes 2.3 s, building the store 2.2 s, and J + K from the store
//! 0.13 s (J 0.02 s, K 0.11 s). The exact four-centre route carries the RHF
//! reference of every correlated method, its CPHF, the RI-MP2 Z-vector and the
//! OO-MP2 loop, which call J/K tens to hundreds of times on one geometry.
//!
//! # Layout
//!
//! One `npair × npair` row-major matrix, `npair = n(n+1)/2`, element
//! `[pq, rs] = (pq|rs)` with `pq = p(p+1)/2 + q` for `p ≥ q`. It keeps the
//! two pair symmetries (4-fold, `n^4/4` doubles) and stores both triangles of
//! the pair matrix, so:
//!
//! * **J** is one GEMM: `J_pq = Σ_rs (pq|rs) w_rs` with the pair-packed density
//!   `w_rs = D_rs + D_sr` (`r > s`) and `w_rr = D_rr`. The off-diagonal pair
//!   weight is what folds the `(rs)`/`(sr)` halves of the sum into one column;
//!   `J[D]` reads only the symmetric part of `D`, as the direct builder does.
//! * **K** is sliced GEMMs: for each row `m`, the slab `(m l|a b)` over
//!   `(l, a, b)` is gathered from rows `pair(m, l)` and unpacked to full
//!   `(a, b)`, and `K_mb = Σ_{l a} (m l|a b) D_la` is one GEMM of the slab's
//!   transpose with the `(n², ndens)` matrix of vectorised densities. The
//!   slab is gathered and contracted in fixed blocks of [`K_L_BLOCK`](crate::incore_jk::K_L_BLOCK) values of
//!   `l`, so a worker holds `K_L_BLOCK · n²` doubles, not `n³`. Any `D` is
//!   accepted, symmetric or not: `K[D]_mb = Σ (m l|a b) D_la` exactly as the
//!   direct builder's 8-fold scatter forms it.
//!
//! A second, exchange-ordered copy `[(m b), (l a)] = (m l|b a)` would make K a
//! plain GEMM without the gather, at three times the memory (`n^4/4` for J plus
//! `n^4/2` for K). The gather costs `n^4` copies per K build, the same order as
//! the GEMM's own reads: on benzene / cc-pVDZ it is 0.065-0.078 s of a
//! 0.11 s single-density K, so the second copy would save under 0.08 s per
//! build against a 2.3 s direct build, and the memory saved is what lets the
//! store reach about 200 basis functions inside a quarter of a 12 GB budget.
//! Forty densities in one call take 0.24 s: the gather is shared by the
//! columns.
//!
//! Each J row block and each K row is computed by one task over a fixed
//! partition (row blocks of [`J_ROW_BLOCK`](crate::incore_jk::J_ROW_BLOCK), one task per `m`, `l` blocks of
//! [`K_L_BLOCK`](crate::incore_jk::K_L_BLOCK)), so the result does not depend on the thread count.
//!
//! # Lifetime and staleness
//!
//! The store belongs to ONE geometry. It lives in the
//! [`SchwarzBounds`](crate::screening::SchwarzBounds) of that geometry
//! (`bounds.incore`), which every driver already rebuilds per geometry
//! (optimizer step, finite-difference displacement), so the store is dropped
//! with the bounds it was built for. As a second line, every use checks the
//! store against the [`PreparedBasis`](ferric_integrals::basis_bridge::PreparedBasis) it is about to serve (atom positions
//! and charges, shell layout and exponents, operator) and refuses a mismatch
//! with an error rather than returning another geometry's J/K.
//!
//! # Who builds it and who uses it
//!
//! `solve_rhf` / `solve_uhf` / `solve_rohf` clear the slot on entry
//! ([`begin_scf`](crate::incore_jk::begin_scf)), so the guess Fock build is always direct and an SCF's
//! bits do not depend on what ran before it on the same bounds; a repeat SCF
//! on one geometry (a CDFT outer loop, a cold retry) rebuilds the store, at
//! about the cost of one direct J/K build. [`attach_for_scf`](crate::incore_jk::attach_for_scf) is called by
//! the same drivers once their J/K route is known. It builds the store when `[scf] eri`
//! ([`EriMode`](crate::incore_jk::EriMode)) asks for it and the route is exact four-centre J and K with
//! the plain Coulomb operator (no RI, COSX, LinK or range separation), and
//! clears the slot otherwise. The direct builders (`DirectJK::build`,
//! `DirectJK::build_uhf`, `rhf::build_jk*`, the block and spin response
//! builders) serve from the store whenever the bounds carry one, so the SCF,
//! the CPHF of the Hessian, the RI-MP2 Z-vector and the OO-MP2 loop that
//! follow it on the same bounds all use it.

use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use ferric_core::memory::plan::{Lifetime, MemoryPlan};
use ferric_core::memory::pool::Reservation;
use ferric_core::parallel::ParallelContext;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ndarray::{s, Array2, ArrayView2};
use rayon::prelude::*;

use crate::rhf::RhfConfig;
use crate::screening::SchwarzBounds;

/// The accepted `[scf] eri` spellings, in the order error messages list them.
pub const ERI_MODES: &[&str] = &["auto", "direct", "incore"];

/// `auto` takes the store when it fits in `budget / AUTO_BUDGET_DIVISOR`: the
/// store stays resident for the SCF and every response build that follows on
/// the same bounds (a correlated method's own tensors among them), so it may
/// claim a quarter of the budget, not all of it.
pub const AUTO_BUDGET_DIVISOR: usize = 4;

/// Rows of the pair-packed store per J task.
pub const J_ROW_BLOCK: usize = 256;

/// Values of `l` per K slab block (a worker holds `K_L_BLOCK · n²` doubles).
pub const K_L_BLOCK: usize = 16;

/// How the exact four-centre route builds J and K (`[scf] eri`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EriMode {
    /// The in-core store when it fits [`AUTO_BUDGET_DIVISOR`] of the budget
    /// and the route is exact four-centre J and K; direct otherwise.
    #[default]
    Auto,
    /// Always the integral-direct builders.
    Direct,
    /// Always the in-core store; an error when it does not fit the budget or
    /// the route is not exact four-centre J and K.
    Incore,
}

impl EriMode {
    /// Strict parse: exactly `"auto"`, `"direct"` or `"incore"`.
    pub fn parse_config_str(s: &str) -> Result<Self, String> {
        match s {
            "auto" => Ok(Self::Auto),
            "direct" => Ok(Self::Direct),
            "incore" => Ok(Self::Incore),
            other => Err(format!(
                "unknown value \"{other}\"; expected one of {ERI_MODES:?}"
            )),
        }
    }

    /// The canonical spelling.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Direct => "direct",
            Self::Incore => "incore",
        }
    }
}

/// Packed index of the AO pair `(p, q)`, `p ≥ q`.
#[inline(always)]
pub fn pair_index(p: usize, q: usize) -> usize {
    debug_assert!(p >= q);
    p * (p + 1) / 2 + q
}

/// Elements of the store for `n` basis functions (`npair²`), `None` on
/// overflow.
pub fn store_elems(n: usize) -> Option<usize> {
    let npair = n.checked_mul(n + 1)? / 2;
    npair.checked_mul(npair)
}

/// The identity of the geometry and basis a store serves: atom positions and
/// charges, the shell layout and exponents, and the operator.
fn geometry_key(prep: &PreparedBasis, op: Operator) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    prep.nbasis().hash(&mut h);
    for a in prep.atoms() {
        for v in [a.atomic_number, a.x, a.y, a.z] {
            v.to_bits().hash(&mut h);
        }
    }
    prep.shell_dims().hash(&mut h);
    prep.shell_to_atom().hash(&mut h);
    for sh in prep.located_shells() {
        sh.l.hash(&mut h);
        for e in sh.exponents {
            e.to_bits().hash(&mut h);
        }
    }
    format!("{op:?}").hash(&mut h);
    h.finish()
}

/// The AO ERIs of one geometry, pair-packed (see the module doc).
pub struct IncoreEri {
    n: usize,
    npair: usize,
    eri: Vec<f64>,
    op: Operator,
    key: u64,
    quartets: usize,
    build_seconds: f64,
    _charge: Reservation,
}

impl std::fmt::Debug for IncoreEri {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IncoreEri")
            .field("n", &self.n)
            .field("bytes", &self.bytes())
            .field("quartets", &self.quartets)
            .finish_non_exhaustive()
    }
}

/// `(m l|a b)` for `l` in `l0..l0+nl`, unpacked to `out[((l-l0) n + a) n + b]`.
fn gather_slab(store: &IncoreEri, m: usize, l0: usize, nl: usize, out: &mut [f64]) {
    let (n, np) = (store.n, store.npair);
    for dl in 0..nl {
        let l = l0 + dl;
        let r = pair_index(m.max(l), m.min(l));
        let row = &store.eri[r * np..(r + 1) * np];
        let blk = &mut out[dl * n * n..(dl + 1) * n * n];
        for a in 0..n {
            let base = a * (a + 1) / 2;
            for b in 0..=a {
                let v = row[base + b];
                blk[a * n + b] = v;
                blk[b * n + a] = v;
            }
        }
    }
}

impl IncoreEri {
    /// The memory plan of a store for `prep` against `budget_bytes` (and the
    /// process pool when one is installed): the store itself, resident, and
    /// one K slab block per worker.
    pub fn plan(prep: &PreparedBasis, budget_bytes: usize) -> Result<MemoryPlan, FerricError> {
        let n = prep.nbasis();
        let elems = store_elems(n).ok_or_else(|| {
            FerricError::General(format!("in-core ERI store for {n} functions overflows"))
        })?;
        let mut plan = MemoryPlan::from_global_pool(Some(budget_bytes), "in-core ERI store");
        plan.reserve("in-core ERI (pq|rs)", elems, Lifetime::Resident);
        plan.reserve_per_worker(
            "in-core K slab block",
            K_L_BLOCK * n * n,
            rayon::current_num_threads().max(1),
        );
        Ok(plan)
    }

    /// Compute every unique shell quartet of `op` over `prep` once (no
    /// Schwarz or density screening) and store it pair-packed. `charge` is
    /// the pool reservation the store holds for its lifetime.
    pub fn build(
        ctx: &ParallelContext,
        prep: &PreparedBasis,
        op: Operator,
        plan: &MemoryPlan,
        charge: Reservation,
    ) -> Result<Self, FerricError> {
        let t0 = std::time::Instant::now();
        let n = prep.nbasis();
        let npair = n * (n + 1) / 2;
        let mut eri = plan.alloc_vec("in-core ERI (pq|rs)", npair * npair)?;
        let pool = crate::engine_pool::EnginePool::new(
            op,
            prep,
            ferric_integrals::engine_pool::eri_precision(),
        )?;
        let dims = prep.shell_dims();
        let offs = prep.shell_offsets();
        let pairs = crate::quartet_scatter::canonical_bra_pairs(prep.nshells());
        let quartets = AtomicUsize::new(0);

        struct StorePtr(*mut f64);
        // SAFETY: every element (pq, rs) and its mirror (rs, pq) belongs to
        // exactly one unique shell quartet, and each unique quartet is
        // computed by exactly one task (the one owning its larger bra pair),
        // so no two tasks write the same element.
        unsafe impl Send for StorePtr {}
        unsafe impl Sync for StorePtr {}
        let ptr = StorePtr(eri.as_mut_ptr());
        let ptr = &ptr;
        let pairs_ref = &pairs;
        pairs_ref
            .par_iter()
            .enumerate()
            .for_each(|(p12, &(s1, s2))| {
                if ferric_core::INTERRUPT.load(Ordering::Relaxed) {
                    return;
                }
                let base = ptr.0;
                let mut local = 0usize;
                pool.with(|eng| {
                    for &(s3, s4) in &pairs_ref[..=p12] {
                        let Some(q) = eng.compute_quartet(prep, s1, s2, s3, s4) else {
                            continue;
                        };
                        local += 1;
                        let (n1, n2, n3, n4) = (dims[s1], dims[s2], dims[s3], dims[s4]);
                        let (o1, o2, o3, o4) = (offs[s1], offs[s2], offs[s3], offs[s4]);
                        for a in 0..n1 {
                            let mu = o1 + a;
                            for b in 0..n2 {
                                let nu = o2 + b;
                                if nu > mu {
                                    continue;
                                }
                                let pq = pair_index(mu, nu);
                                for c in 0..n3 {
                                    let la = o3 + c;
                                    for d in 0..n4 {
                                        let sg = o4 + d;
                                        if sg > la {
                                            continue;
                                        }
                                        let rs = pair_index(la, sg);
                                        let v = q[((a * n2 + b) * n3 + c) * n4 + d];
                                        // SAFETY: pq, rs < npair; disjoint
                                        // ownership as argued at `StorePtr`.
                                        unsafe {
                                            *base.add(pq * npair + rs) = v;
                                            *base.add(rs * npair + pq) = v;
                                        }
                                    }
                                }
                            }
                        }
                    }
                });
                quartets.fetch_add(local, Ordering::Relaxed);
            });
        ctx.check_interrupted()?;
        Ok(IncoreEri {
            n,
            npair,
            eri,
            op,
            key: geometry_key(prep, op),
            quartets: quartets.load(Ordering::SeqCst),
            build_seconds: t0.elapsed().as_secs_f64(),
            _charge: charge,
        })
    }

    /// Basis functions.
    pub fn nbasis(&self) -> usize {
        self.n
    }

    /// Resident bytes of the store.
    pub fn bytes(&self) -> usize {
        self.eri.len() * std::mem::size_of::<f64>()
    }

    /// Unique shell quartets computed into the store.
    pub fn quartets(&self) -> usize {
        self.quartets
    }

    /// Wall seconds the store took to build.
    pub fn build_seconds(&self) -> f64 {
        self.build_seconds
    }

    /// The operator the store holds.
    pub fn op(&self) -> Operator {
        self.op
    }

    /// Does this store hold the integrals of `prep` under `op`?
    pub fn serves(&self, prep: &PreparedBasis, op: Operator) -> bool {
        self.n == prep.nbasis() && self.key == geometry_key(prep, op)
    }

    /// A copy whose integrals are `|(pq|rs)|`. With absolute densities its
    /// J and K are `Σ |(pq|rs)| |D_rs|`-type sums, the scale of a
    /// floating-point rounding bound on the same contraction.
    #[doc(hidden)]
    pub fn abs_copy(&self) -> Self {
        IncoreEri {
            n: self.n,
            npair: self.npair,
            eri: self.eri.iter().map(|v| v.abs()).collect(),
            op: self.op,
            key: self.key,
            quartets: self.quartets,
            build_seconds: 0.0,
            _charge: Reservation::inert("in-core ERI abs copy"),
        }
    }

    /// `J[D_i]` for every density (symmetric `n × n` results).
    pub fn coulomb(&self, ds: &[&Array2<f64>]) -> Vec<Array2<f64>> {
        let (n, np, m) = (self.n, self.npair, ds.len());
        if m == 0 {
            return Vec::new();
        }
        let mut w = Array2::<f64>::zeros((np, m));
        for (i, d) in ds.iter().enumerate() {
            for p in 0..n {
                for q in 0..p {
                    w[(pair_index(p, q), i)] = d[(p, q)] + d[(q, p)];
                }
                w[(pair_index(p, p), i)] = d[(p, p)];
            }
        }
        let nblk = np.div_ceil(J_ROW_BLOCK);
        let blocks: Vec<Array2<f64>> = (0..nblk)
            .into_par_iter()
            .map(|b| {
                let lo = b * J_ROW_BLOCK;
                let rows = J_ROW_BLOCK.min(np - lo);
                let pblk = ArrayView2::from_shape((rows, np), &self.eri[lo * np..(lo + rows) * np])
                    .expect("row block of the npair x npair store");
                pblk.dot(&w)
            })
            .collect();
        let mut jp = Array2::<f64>::zeros((np, m));
        for (b, blk) in blocks.iter().enumerate() {
            let lo = b * J_ROW_BLOCK;
            jp.slice_mut(s![lo..lo + blk.nrows(), ..]).assign(blk);
        }
        let mut out = vec![Array2::<f64>::zeros((n, n)); m];
        for (i, j) in out.iter_mut().enumerate() {
            for p in 0..n {
                for q in 0..=p {
                    let v = jp[(pair_index(p, q), i)];
                    j[(p, q)] = v;
                    j[(q, p)] = v;
                }
            }
        }
        out
    }

    /// `K[D_i]_mb = Σ_la (m l|a b) D_i[l, a]` for every density (any `D`).
    pub fn exchange(&self, ds: &[&Array2<f64>]) -> Vec<Array2<f64>> {
        let (n, m) = (self.n, ds.len());
        if m == 0 {
            return Vec::new();
        }
        let n2 = n * n;
        // dmat[(l n + a), i] = D_i[l, a]
        let mut dmat = Array2::<f64>::zeros((n2, m));
        for (i, d) in ds.iter().enumerate() {
            for ((l, a), &x) in d.indexed_iter() {
                dmat[(l * n + a, i)] = x;
            }
        }
        let rows: Vec<Array2<f64>> = (0..n)
            .into_par_iter()
            .map_init(
                || vec![0.0f64; K_L_BLOCK * n2],
                |slab, row| {
                    let mut kr = Array2::<f64>::zeros((n, m));
                    let mut l0 = 0;
                    while l0 < n {
                        let nl = K_L_BLOCK.min(n - l0);
                        gather_slab(self, row, l0, nl, slab);
                        let v = ArrayView2::from_shape((nl * n, n), &slab[..nl * n2])
                            .expect("slab block is (nl n) x n");
                        let dblk = dmat.slice(s![l0 * n..(l0 + nl) * n, ..]);
                        ndarray::linalg::general_mat_mul(1.0, &v.t(), &dblk, 1.0, &mut kr);
                        l0 += nl;
                    }
                    kr
                },
            )
            .collect();
        let mut out = vec![Array2::<f64>::zeros((n, n)); m];
        for (row, kr) in rows.iter().enumerate() {
            for (i, k) in out.iter_mut().enumerate() {
                for b in 0..n {
                    k[(row, b)] = kr[(b, i)];
                }
            }
        }
        out
    }

    /// `J[jd_i]` and `K[kd_i]` in one call.
    pub fn jk_multi(
        &self,
        jd: &[&Array2<f64>],
        kd: &[&Array2<f64>],
    ) -> (Vec<Array2<f64>>, Vec<Array2<f64>>) {
        (self.coulomb(jd), self.exchange(kd))
    }

    /// `J[D_a + D_b]` and, when `exchange`, `[K[D_a], K[D_b]]`: the spin
    /// response builder's contract.
    pub fn spin_jk(
        &self,
        d: [&Array2<f64>; 2],
        exchange: bool,
    ) -> (Array2<f64>, Option<[Array2<f64>; 2]>) {
        let total = d[0] + d[1];
        let kd: &[&Array2<f64>] = if exchange { &d } else { &[] };
        let (j, k) = self.jk_multi(&[&total], kd);
        let mut j = j.into_iter();
        let k = <[Array2<f64>; 2]>::try_from(k).ok();
        (j.next().unwrap_or_default(), k)
    }

    /// `j += J[d]`, `k += K[d]` (accumulating, as the direct builders do).
    pub fn add_jk(&self, d: &Array2<f64>, j: &mut Array2<f64>, k: &mut Array2<f64>) {
        let (jj, kk) = self.jk_multi(&[d], &[d]);
        *j += &jj[0];
        *k += &kk[0];
    }

    /// `j += J[d_total]`, `k_a += K[d_a]`, `k_b += K[d_b]`.
    pub fn add_uhf(
        &self,
        d_total: &Array2<f64>,
        d_a: &Array2<f64>,
        d_b: &Array2<f64>,
        j: &mut Array2<f64>,
        k_a: &mut Array2<f64>,
        k_b: &mut Array2<f64>,
    ) {
        let (jj, kk) = self.jk_multi(&[d_total], &[d_a, d_b]);
        *j += &jj[0];
        *k_a += &kk[0];
        *k_b += &kk[1];
    }
}

/// The store slot of one [`SchwarzBounds`]. Clones share the slot (a cloned
/// bounds describes the same geometry).
#[derive(Clone, Default)]
pub struct IncoreSlot(Arc<RwLock<Option<Arc<IncoreEri>>>>);

impl std::fmt::Debug for IncoreSlot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.get() {
            Some(s) => write!(f, "IncoreSlot({s:?})"),
            None => write!(f, "IncoreSlot(empty)"),
        }
    }
}

impl IncoreSlot {
    /// The attached store, if any.
    pub fn get(&self) -> Option<Arc<IncoreEri>> {
        self.0.read().ok().and_then(|g| g.clone())
    }

    /// Attach (`Some`) or clear (`None`) the store.
    pub fn set(&self, store: Option<Arc<IncoreEri>>) {
        if let Ok(mut g) = self.0.write() {
            *g = store;
        }
    }
}

/// Detach any store from `bounds` at the start of an SCF. The guess Fock
/// build runs before the SCF knows its J/K route (and so before
/// [`attach_for_scf`]); a store left by an earlier SCF on the same bounds
/// would serve it, and a repeat SCF would then differ from the first in the
/// last bits. With the slot cleared, every SCF builds its guess the same way
/// (direct) and builds its own store; the response solves that follow the
/// SCF on the same bounds (CPHF, Z-vector, OO-MP2) still find it.
pub fn begin_scf(bounds: &SchwarzBounds) {
    bounds.incore.set(None);
}

/// The store `bounds` carries for `prep`: `Ok(None)` when it carries none,
/// an error when it carries one built for another geometry, basis or
/// operator (a stale store must never answer).
pub fn store_for(
    bounds: &SchwarzBounds,
    prep: &PreparedBasis,
) -> Result<Option<Arc<IncoreEri>>, FerricError> {
    let Some(s) = bounds.incore.get() else {
        return Ok(None);
    };
    if !s.serves(prep, bounds.op) {
        return Err(FerricError::General(
            "in-core ERI store: the bounds carry a store built for another geometry, basis \
             or operator; it is never reused across geometries (rebuild the bounds, or set \
             [scf] eri = \"direct\")"
                .into(),
        ));
    }
    Ok(Some(s))
}

/// What the in-core route of the last SCF on a config was: shared by the
/// config's clones, written by [`attach_for_scf`], read by the certificate.
#[derive(Clone, Default)]
pub struct EriRouteRecord(Arc<Mutex<Option<EriRoute>>>);

impl std::fmt::Debug for EriRouteRecord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "EriRouteRecord({:?})", self.get())
    }
}

impl EriRouteRecord {
    /// The recorded route of the last SCF (`None`: direct, or no SCF yet).
    pub fn get(&self) -> Option<EriRoute> {
        self.0.lock().ok().and_then(|g| g.clone())
    }

    fn set(&self, r: Option<EriRoute>) {
        if let Ok(mut g) = self.0.lock() {
            *g = r;
        }
    }
}

/// An SCF that served J/K from the in-core store.
#[derive(Debug, Clone, PartialEq)]
pub struct EriRoute {
    /// Basis functions.
    pub nbasis: usize,
    /// Resident bytes of the store.
    pub bytes: usize,
    /// Unique shell quartets held.
    pub quartets: usize,
    /// Wall seconds of the store build (0 when an existing store was reused).
    pub build_seconds: f64,
    /// The `[scf] eri` mode that selected it.
    pub mode: EriMode,
}

impl EriRoute {
    /// The run-log / certificate text.
    pub fn describe(&self) -> String {
        format!(
            "in-core four-centre ERIs ([scf] eri = \"{}\"): {} functions, {:.3} GB, {} \
             unique shell quartets, no Schwarz or density screening; built in {:.2} s",
            self.mode.as_str(),
            self.nbasis,
            self.bytes as f64 / 1e9,
            self.quartets,
            self.build_seconds
        )
    }
}

/// The exact four-centre builder for an SCF, with the in-core store attached
/// first ([`attach_for_scf`]); `None` when the route is not `direct_route`.
pub fn direct_builder_for_scf<'a>(
    ctx: &'a ParallelContext,
    prep: &'a PreparedBasis,
    bounds: &'a SchwarzBounds,
    config: &RhfConfig,
    budget_bytes: usize,
    (direct_route, omega): (bool, f64),
) -> Result<Option<crate::direct_jk::DirectJK<'a>>, FerricError> {
    attach_for_scf(
        ctx,
        prep,
        bounds,
        config,
        budget_bytes,
        (direct_route, omega),
    )?;
    Ok(direct_route.then(|| {
        crate::direct_jk::DirectJK::new(ctx, prep, bounds, config.integral_thresh, budget_bytes)
    }))
}

/// Decide, and when chosen build or reuse, the in-core store for an SCF on
/// `bounds`.
///
/// `(direct_route, omega)` is the driver's own test that J and K both come
/// from the combined direct four-centre builder (no RI-J/RI-K, no LinK/COSX,
/// no injected J/K) and the exchange's range-separation parameter; the route
/// is exact when it is direct, `omega` is 0 and the operator is plain
/// Coulomb. Returns the
/// store the SCF will serve from, or `None` (slot cleared) for the direct
/// route. Errors only for `eri = "incore"` on a route or budget that cannot
/// take it, and for a failed store build.
pub fn attach_for_scf(
    ctx: &ParallelContext,
    prep: &PreparedBasis,
    bounds: &SchwarzBounds,
    config: &RhfConfig,
    budget_bytes: usize,
    (direct_route, omega): (bool, f64),
) -> Result<Option<Arc<IncoreEri>>, FerricError> {
    let mode = config.eri;
    let clear = || {
        bounds.incore.set(None);
        config.eri_route.set(None);
        Ok(None)
    };
    let exact = direct_route && omega == 0.0 && bounds.op == Operator::coulomb();
    match (mode, exact) {
        (EriMode::Direct, _) => return clear(),
        (EriMode::Auto, false) => return clear(),
        (EriMode::Incore, false) => {
            return Err(FerricError::General(
                "[scf] eri = \"incore\" needs the exact four-centre J and K route; this SCF \
                 builds J/K by density fitting, LinK, COSX or a range-separated operator. Use \
                 [scf] eri = \"auto\" or \"direct\""
                    .into(),
            ))
        }
        _ => {}
    }
    if mode == EriMode::Auto && ctx.size > 1 {
        return clear();
    }
    let record = |s: &Arc<IncoreEri>, built: bool| {
        let route = EriRoute {
            nbasis: s.nbasis(),
            bytes: s.bytes(),
            quartets: s.quartets(),
            build_seconds: if built { s.build_seconds() } else { 0.0 },
            mode,
        };
        config.eri_route.set(Some(route.clone()));
        route
    };
    if let Some(s) = bounds.incore.get() {
        if s.serves(prep, bounds.op) {
            record(&s, false);
            return Ok(Some(s));
        }
        bounds.incore.set(None);
    }
    let plan = IncoreEri::plan(prep, budget_bytes)?;
    let store_bytes = store_elems(prep.nbasis())
        .unwrap_or(usize::MAX)
        .saturating_mul(8);
    if mode == EriMode::Auto && store_bytes > plan.budget_bytes() / AUTO_BUDGET_DIVISOR {
        if config.verbose {
            eprintln!(
                "[ferric] SCF J/K: direct four-centre ([scf] eri = \"auto\": the in-core store \
                 needs {:.3} GB, more than 1/{AUTO_BUDGET_DIVISOR} of the {:.3} GB budget)",
                store_bytes as f64 / 1e9,
                plan.budget_bytes() as f64 / 1e9
            );
        }
        return clear();
    }
    let charge = match (mode, plan.commit()) {
        (_, Ok(c)) => c,
        (EriMode::Auto, Err(_)) => return clear(),
        (_, Err(e)) => return Err(e),
    };
    let store = Arc::new(IncoreEri::build(ctx, prep, bounds.op, &plan, charge)?);
    bounds.incore.set(Some(store.clone()));
    let route = record(&store, true);
    if config.verbose {
        eprintln!("[ferric] SCF J/K: {}", route.describe());
    }
    if let Some(log) = crate::runlog::log() {
        log.note(
            "eri_store",
            serde_json::json!({
                "route": "incore",
                "mode": mode.as_str(),
                "nbasis": route.nbasis,
                "bytes": route.bytes,
                "quartets": route.quartets,
                "build_seconds": route.build_seconds,
            }),
        );
    }
    Ok(Some(store))
}
