//! perf5 d3 prototype: a standalone measurement harness for a bra-pair-local
//! J/K digest. NOT wired into any production path: nothing in the library
//! calls this file, and it changes no behaviour.
//!
//! `digest <xyz> <basis> <dens.bin> <rot|off> [reps] [fill_threads]`
//!
//! 1. Replays `DirectJK`'s closed-shell walk exactly as
//!    `gc_shared_prim_bench profile` does (bra prefilter `Q > thresh / (max Q
//!    · max D)`, canonical ket loop, six-pair `Q·Q·Dmax < 1e-12`) on the
//!    rotated (`rot`: the OGC level `auto` uses) or parent shells, with a
//!    converged density written by `gc_shared_prim_bench density`.
//! 2. Computes every surviving quartet ONCE with libint (untimed, in chunks
//!    of at most `CHUNK_DOUBLES` integrals, `fill_threads` threads) and keeps
//!    the blocks.
//! 3. Times ONLY the digest, single thread, interleaving the variants
//!    (forward order on even reps, reverse on odd reps):
//!    * `A16`: today's digest (`prod_digest.rs`, a verbatim copy of
//!      `quartet_scatter.rs`'s closed-shell code), `BLOCKED_MIN_INTEGRALS =
//!      16`; also `A1` (blocked for every non-all-s quartet) and `Ainf` (the
//!      per-element path for everything);
//!    * `Bsym`: bra-pair-local digest assuming a symmetric density: J of the
//!      bra pair as a running dot product written once, K only into rows of
//!      the two bra shells (`K = Kb + Kbᵀ` at the end), the ket J block
//!      written once per quartet into a half matrix (`J = Jh + Jhᵀ`);
//!    * `Bgen`: the same for a general density: a second bra-row accumulator
//!      `Kt` against `Dᵀ` (`K = Kb + Ktᵀ`);
//!    * `Bsym4`: `Bsym` with the innermost (ket index 4) loop const-generic
//!      for n4 ∈ {1, 3, 5} (the common pure s / p / d shells).
//! 4. Checks: `A16` against the real `scatter_bra_pair` (the source file is
//!    compiled into this example) on the same walk, bitwise; every B variant
//!    against `A16` (max |ΔJ|, |ΔK|, absolute and relative); the first-order
//!    energy change `½ tr D(ΔJ − ½ΔK)`; and `Bsym4` run in the deterministic
//!    group partition (`reduce::deterministic_group_size`, partials summed in
//!    group order) on 1, 2 and 4 threads, bitwise.

use std::time::Instant;

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::contraction_rotation::{DerivRotMutant, RotatedBasis, RotationLevel};
use ferric_integrals::engine::Engine;
use ferric_integrals::engine_pool::ERI_PRECISION;
use ferric_integrals::operator::Operator;
use ferric_scf::screening::SchwarzBounds;
use ndarray::Array2;
use rayon::prelude::*;

mod prod_digest;
#[allow(dead_code, unused_imports, clippy::all)]
#[path = "../../src/quartet_scatter.rs"]
mod real_scatter;

const THRESH: f64 = 1e-12;
/// Integrals held per chunk (8 B each): 1.2e8 = 0.96 GB.
const CHUNK_DOUBLES: usize = 120_000_000;

fn load_mol(path: &str) -> Molecule {
    let text = std::fs::read_to_string(path).expect("read xyz");
    Molecule::parse_xyz(&text, 0, 1).expect("parse xyz")
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    match a.get(1).map(String::as_str) {
        Some("digest") => {
            let reps = a.get(6).map_or(5, |s| s.parse().unwrap());
            let fill = a.get(7).map_or(4, |s| s.parse().unwrap());
            let walk = a.get(8).map_or(0, |s| s.parse().unwrap());
            digest(&a[2], &a[3], &a[4], a[5] == "rot", reps, fill, walk)
        }
        _ => eprintln!(
            "usage: digest <xyz> <basis> <dens.bin> <rot|off> [reps] [fill_threads] [walk_reps]"
        ),
    }
}

fn read_density(path: &str, n: usize) -> Array2<f64> {
    let b = std::fs::read(path).unwrap();
    assert_eq!(b.len(), n * n * 8, "density size");
    let v: Vec<f64> = b
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect();
    Array2::from_shape_vec((n, n), v).unwrap()
}

/// The walk's basis, density, Q table, shell-blocked max |D| and bra pairs.
struct Setup {
    prep: PreparedBasis,
    d: Array2<f64>,
    q: Array2<f64>,
    dmax: Array2<f64>,
    pairs: Vec<(usize, usize)>,
}

fn setup(xyz: &str, name: &str, dens: &str, rot: bool) -> Setup {
    let mol = load_mol(xyz);
    let parent = PreparedBasis::new(&mol, &basis::bundled(name).unwrap()).unwrap();
    let d = read_density(dens, parent.nbasis());
    let (prep, d) = if rot {
        match RotatedBasis::detect_prepared(&parent, RotationLevel::Ogc).unwrap() {
            Some(rb) => {
                let dp = rb.ao_transpose(DerivRotMutant::None).matrix(&d);
                println!("rotated columns: {}", rb.n_rotated_columns);
                (rb.prep, dp)
            }
            None => (parent, d),
        }
    } else {
        (parent, d)
    };
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
    let q = bounds.q.clone();
    let dmax = real_scatter::build_d_max_shell(&prep, &d);
    let nsh = prep.nshells();
    let max_d = dmax.iter().cloned().fold(0.0f64, f64::max);
    let max_q = q.iter().cloned().fold(0.0f64, f64::max);
    let bra_thresh = THRESH / (max_q * max_d.max(1e-30));
    let mut pairs = Vec::new();
    for s1 in 0..nsh {
        for s2 in 0..=s1 {
            if q[(s1, s2)] > bra_thresh {
                pairs.push((s1, s2));
            }
        }
    }
    Setup {
        prep,
        d,
        q,
        dmax,
        pairs,
    }
}

/// One bra pair's computed survivors: `(s3, s4, offset into ints)`.
struct Bra {
    s1: usize,
    s2: usize,
    kets: Vec<(u32, u32, u64)>,
    ints: Vec<f64>,
}

impl Setup {
    /// Survivors of bra pair `(s1, s2)` computed by libint, in walk order.
    fn fill(&self, eng: &mut Engine, s1: usize, s2: usize) -> Bra {
        let mut kets = Vec::new();
        let mut ints = Vec::new();
        let b12 = self.q[(s1, s2)];
        let dm = &self.dmax;
        for s3 in 0..=s1 {
            let s4max = if s3 == s1 { s2 } else { s3 };
            for s4 in 0..=s4max {
                let d = dm[(s1, s2)]
                    .max(dm[(s3, s4)])
                    .max(dm[(s1, s3)])
                    .max(dm[(s1, s4)])
                    .max(dm[(s2, s3)])
                    .max(dm[(s2, s4)]);
                if b12 * self.q[(s3, s4)] * d < THRESH {
                    continue;
                }
                if let Some(blk) = eng.compute_quartet(&self.prep, s1, s2, s3, s4) {
                    kets.push((s3 as u32, s4 as u32, ints.len() as u64));
                    ints.extend_from_slice(blk);
                }
            }
        }
        Bra { s1, s2, kets, ints }
    }
}

// ---------------------------------------------------------------------------
// The bra-pair-local digest ("B").
// ---------------------------------------------------------------------------

/// Outputs of a B digest: `J = jh + jhᵀ`; `K = kb + kbᵀ` (symmetric density)
/// or `K = kb + ktᵀ` (general).
struct BOut {
    jh: Array2<f64>,
    kb: Array2<f64>,
    kt: Array2<f64>,
}

impl BOut {
    fn new(n: usize, general: bool) -> Self {
        BOut {
            jh: Array2::zeros((n, n)),
            kb: Array2::zeros((n, n)),
            kt: Array2::zeros((if general { n } else { 0 }, if general { n } else { 0 })),
        }
    }

    /// `(J, K)` from the half accumulators.
    fn finish(&self) -> (Array2<f64>, Array2<f64>) {
        let j = &self.jh + &self.jh.t();
        let k = if self.kt.is_empty() {
            &self.kb + &self.kb.t()
        } else {
            &self.kb + &self.kt.t()
        };
        (j, k)
    }

    fn add(&mut self, o: &BOut) {
        self.jh += &o.jh;
        self.kb += &o.kb;
        if !self.kt.is_empty() {
            self.kt += &o.kt;
        }
    }
}

/// Per-bra-pair scratch of the B digest.
#[derive(Default)]
struct BScratch {
    jb: Vec<f64>,
    e12: Vec<f64>,
    e34: Vec<f64>,
    s34: Vec<f64>,
}

#[inline(always)]
unsafe fn dotp<const N: usize>(a: *const f64, b: *const f64, n: usize) -> f64 {
    let n = if N > 0 { N } else { n };
    let mut s = 0.0;
    for i in 0..n {
        s += *a.add(i) * *b.add(i);
    }
    s
}

#[inline(always)]
unsafe fn axpyp<const N: usize>(y: *mut f64, a: f64, x: *const f64, n: usize) {
    let n = if N > 0 { N } else { n };
    for i in 0..n {
        *y.add(i) += a * *x.add(i);
    }
}

/// One quartet of the B digest. `N4 > 0` fixes `n4` at compile time.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn b_quartet<const N4: usize, const GEN: bool>(
    q: &[f64],
    n: usize,
    dp: *const f64,
    dtp: *const f64,
    kbp: *mut f64,
    ktp: *mut f64,
    jhp: *mut f64,
    s: &mut BScratch,
    [n1, n2, n3, n4]: [usize; 4],
    [o1, o2, o3, o4]: [usize; 4],
    [sym12, sym34, sym1234]: [bool; 3],
) {
    let n4 = if N4 > 0 { N4 } else { n4 };
    let n34 = n3 * n4;
    let e34 = s.e34.as_mut_ptr();
    for c in 0..n3 {
        for dd in 0..n4 {
            let mut v = *dp.add((o3 + c) * n + o4 + dd);
            if sym34 {
                v += *dp.add((o4 + dd) * n + o3 + c);
            }
            *e34.add(c * n4 + dd) = v;
        }
    }
    let s34 = s.s34.as_mut_ptr();
    if sym1234 {
        for i in 0..n34 {
            *s34.add(i) = 0.0;
        }
    }
    // Symmetric density: the self-paired quartet ((s3,s4) == (s1,s2)) has no
    // ket-row images, and `K = Kb + Kbᵀ` would add them: halve it.
    let w = if GEN || sym1234 { 1.0 } else { 0.5 };
    let qp = q.as_ptr();
    for a in 0..n1 {
        let mu = o1 + a;
        let dmu = dp.add(mu * n);
        let kmu = kbp.add(mu * n);
        for b in 0..n2 {
            let nu = o2 + b;
            let dnu = dp.add(nu * n);
            let knu = kbp.add(nu * n);
            let ab = a * n2 + b;
            let e12 = *s.e12.as_ptr().add(ab);
            let mut jacc = 0.0;
            for c in 0..n3 {
                let la = o3 + c;
                let r = qp.add((ab * n3 + c) * n4);
                jacc += dotp::<N4>(r, e34.add(c * n4), n4);
                if sym1234 {
                    axpyp::<N4>(s34.add(c * n4), e12, r, n4);
                }
                *kmu.add(la) += w * dotp::<N4>(r, dnu.add(o4), n4);
                if sym12 {
                    *knu.add(la) += w * dotp::<N4>(r, dmu.add(o4), n4);
                }
                if sym34 {
                    axpyp::<N4>(kmu.add(o4), w * *dnu.add(la), r, n4);
                    if sym12 {
                        axpyp::<N4>(knu.add(o4), w * *dmu.add(la), r, n4);
                    }
                }
                if GEN && sym1234 {
                    // Ket-row images K[λ,μ] += D[σ,ν] v ... stored transposed
                    // in kt's bra rows against Dᵀ's rows.
                    let tmu = ktp.add(mu * n);
                    let tnu = ktp.add(nu * n);
                    let dtmu = dtp.add(mu * n);
                    let dtnu = dtp.add(nu * n);
                    *tmu.add(la) += dotp::<N4>(r, dtnu.add(o4), n4);
                    if sym12 {
                        *tnu.add(la) += dotp::<N4>(r, dtmu.add(o4), n4);
                    }
                    if sym34 {
                        axpyp::<N4>(tmu.add(o4), *dtnu.add(la), r, n4);
                        if sym12 {
                            axpyp::<N4>(tnu.add(o4), *dtmu.add(la), r, n4);
                        }
                    }
                }
            }
            *s.jb.as_mut_ptr().add(ab) += jacc;
        }
    }
    if sym1234 {
        let w34 = if sym34 { 1.0 } else { 0.5 };
        for c in 0..n3 {
            for dd in 0..n4 {
                *jhp.add((o3 + c) * n + o4 + dd) += w34 * *s34.add(c * n4 + dd);
            }
        }
    }
}

/// The per-bra-pair constants of the B digest.
#[derive(Clone, Copy)]
struct BraB {
    nn: [usize; 2],
    oo: [usize; 2],
    sym12: bool,
    s: [usize; 2],
}

/// Start bra pair `(s1, s2)`: zero the bra J accumulator, form
/// `E12 = D[μν] (+ D[νμ])`.
fn b_begin(
    dims: &[usize],
    offs: &[usize],
    d: &Array2<f64>,
    s1: usize,
    s2: usize,
    s: &mut BScratch,
) -> BraB {
    let (n1, n2) = (dims[s1], dims[s2]);
    let (o1, o2) = (offs[s1], offs[s2]);
    let sym12 = s1 != s2;
    let m = dims.iter().copied().max().unwrap_or(1);
    for v in [&mut s.jb, &mut s.e12, &mut s.e34, &mut s.s34] {
        if v.len() < m * m {
            v.resize(m * m, 0.0);
        }
    }
    s.jb[..n1 * n2].fill(0.0);
    for a in 0..n1 {
        for b in 0..n2 {
            let mut v = d[(o1 + a, o2 + b)];
            if sym12 {
                v += d[(o2 + b, o1 + a)];
            }
            s.e12[a * n2 + b] = v;
        }
    }
    BraB {
        nn: [n1, n2],
        oo: [o1, o2],
        sym12,
        s: [s1, s2],
    }
}

/// Digest one quartet `(s1 s2|s3 s4)` of the current bra pair.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn b_one<const SPEC: bool, const GEN: bool>(
    br: &BraB,
    c: &Ctx,
    out: &mut BOut,
    s: &mut BScratch,
    s3: usize,
    s4: usize,
    q: &[f64],
) {
    let (dims, offs) = (c.dims, c.offs);
    let n = c.d.nrows();
    let (n3, n4) = (dims[s3], dims[s4]);
    let [n1, n2] = br.nn;
    let q = &q[..n1 * n2 * n3 * n4];
    let nn = [n1, n2, n3, n4];
    let oo = [br.oo[0], br.oo[1], offs[s3], offs[s4]];
    let sy = [br.sym12, s3 != s4, (br.s[0], br.s[1]) != (s3, s4)];
    let (dp, dtp) = (c.d.as_ptr(), c.dt.as_ptr());
    let (kbp, ktp, jhp) = (
        out.kb.as_mut_ptr(),
        out.kt.as_mut_ptr(),
        out.jh.as_mut_ptr(),
    );
    // SAFETY: every index is inside its shell block of the n×n matrices and
    // the quartet block has n1·n2·n3·n4 entries.
    unsafe {
        match (SPEC, n4) {
            (true, 1) => b_quartet::<1, GEN>(q, n, dp, dtp, kbp, ktp, jhp, s, nn, oo, sy),
            (true, 3) => b_quartet::<3, GEN>(q, n, dp, dtp, kbp, ktp, jhp, s, nn, oo, sy),
            (true, 5) => b_quartet::<5, GEN>(q, n, dp, dtp, kbp, ktp, jhp, s, nn, oo, sy),
            _ => b_quartet::<0, GEN>(q, n, dp, dtp, kbp, ktp, jhp, s, nn, oo, sy),
        }
    }
}

/// Flush the bra pair's J (written once).
fn b_end(br: &BraB, out: &mut BOut, s: &BScratch) {
    let [n1, n2] = br.nn;
    let [o1, o2] = br.oo;
    let w12 = if br.sym12 { 1.0 } else { 0.5 };
    for a in 0..n1 {
        for b in 0..n2 {
            out.jh[(o1 + a, o2 + b)] += w12 * s.jb[a * n2 + b];
        }
    }
}

/// One stored bra pair of the B digest. `SPEC`: const-generic n4 ∈ {1, 3, 5}.
fn b_bra_pair<const SPEC: bool, const GEN: bool>(
    c: &Ctx,
    bra: &Bra,
    out: &mut BOut,
    s: &mut BScratch,
) {
    let br = b_begin(c.dims, c.offs, c.d, bra.s1, bra.s2, s);
    for &(s3, s4, off) in &bra.kets {
        b_one::<SPEC, GEN>(
            &br,
            c,
            out,
            s,
            s3 as usize,
            s4 as usize,
            &bra.ints[off as usize..],
        );
    }
    b_end(&br, out, s);
}

// ---------------------------------------------------------------------------
// Driver.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Var {
    A16,
    A1,
    Ainf,
    Bsym,
    Bgen,
    Bsym4,
    Bgen4,
}

const VARS: [Var; 7] = [
    Var::A16,
    Var::Bsym4,
    Var::A1,
    Var::Bsym,
    Var::Ainf,
    Var::Bgen,
    Var::Bgen4,
];

enum Acc {
    A(Array2<f64>, Array2<f64>),
    B(BOut),
}

impl Var {
    fn is_b(self) -> bool {
        matches!(self, Var::Bsym | Var::Bgen | Var::Bsym4 | Var::Bgen4)
    }
    fn acc(self, n: usize) -> Acc {
        if self.is_b() {
            Acc::B(BOut::new(n, matches!(self, Var::Bgen | Var::Bgen4)))
        } else {
            Acc::A(Array2::zeros((n, n)), Array2::zeros((n, n)))
        }
    }
}

struct Ctx<'a> {
    dims: &'a [usize],
    offs: &'a [usize],
    d: &'a Array2<f64>,
    dt: &'a Array2<f64>,
}

/// Digest the bras of one chunk with variant `v` into `acc`.
fn run_var(c: &Ctx, v: Var, bras: &[Bra], acc: &mut Acc) {
    let (dims, offs, d) = (c.dims, c.offs, c.d);
    match acc {
        Acc::A(j, k) => {
            let mut mode = prod_digest::JkMode::Both(std::mem::take(j), std::mem::take(k));
            for b in bras {
                let f = match v {
                    Var::A16 => prod_digest::digest_bra_pair::<16>,
                    Var::A1 => prod_digest::digest_bra_pair::<1>,
                    _ => prod_digest::digest_bra_pair::<{ usize::MAX }>,
                };
                f(dims, offs, d, b.s1, b.s2, &b.kets, &b.ints, &mut mode);
            }
            let prod_digest::JkMode::Both(jj, kk) = mode else {
                unreachable!()
            };
            *j = jj;
            *k = kk;
        }
        Acc::B(out) => {
            let mut s = BScratch::default();
            for b in bras {
                match v {
                    Var::Bsym => b_bra_pair::<false, false>(c, b, out, &mut s),
                    Var::Bgen => b_bra_pair::<false, true>(c, b, out, &mut s),
                    Var::Bgen4 => b_bra_pair::<true, true>(c, b, out, &mut s),
                    _ => b_bra_pair::<true, false>(c, b, out, &mut s),
                }
            }
        }
    }
}

fn finish(acc: &Acc) -> (Array2<f64>, Array2<f64>) {
    match acc {
        Acc::A(j, k) => (j.clone(), k.clone()),
        Acc::B(o) => o.finish(),
    }
}

fn maxabs(a: &Array2<f64>) -> f64 {
    a.iter().fold(0.0f64, |m, x| m.max(x.abs()))
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

fn uptime() -> String {
    std::fs::read_to_string("/proc/loadavg")
        .unwrap_or_default()
        .trim()
        .to_string()
}

fn digest(
    xyz: &str,
    name: &str,
    dens: &str,
    rot: bool,
    reps: usize,
    fill_threads: usize,
    walk_reps: usize,
) {
    let st = setup(xyz, name, dens, rot);
    let prep = &st.prep;
    let n = prep.nbasis();
    let (dims, offs) = (prep.shell_dims().to_vec(), prep.shell_offsets().to_vec());
    let dt = st.d.t().to_owned();
    let dsym =
        st.d.iter()
            .zip(dt.iter())
            .all(|(a, b)| a.to_bits() == b.to_bits());
    let dasym = maxabs(&(&st.d - &dt));
    println!(
        "basis {name} rot {rot}: nbf {n} nsh {} bra pairs {} | D bitwise symmetric {dsym} (max |D-Dt| {dasym:.1e}) | reps {reps} | load {}",
        prep.nshells(),
        st.pairs.len(),
        uptime()
    );
    let c = Ctx {
        dims: &dims,
        offs: &offs,
        d: &st.d,
        dt: &dt,
    };
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(fill_threads)
        .build()
        .unwrap();
    let engines: Vec<std::sync::Mutex<Engine>> = (0..fill_threads)
        .map(|_| {
            std::sync::Mutex::new(Engine::new_2e(Operator::coulomb(), prep, ERI_PRECISION).unwrap())
        })
        .collect();

    // Accumulators: rep 0 into `keep`, later reps into `junk` (same shapes).
    let mut keep: Vec<Acc> = VARS.iter().map(|v| v.acc(n)).collect();
    let mut junk: Vec<Acc> = VARS.iter().map(|v| v.acc(n)).collect();
    let mut t = vec![vec![0.0f64; reps]; VARS.len()];
    let (mut quartets, mut nints, mut small_q, mut small_i) = (0u64, 0u64, 0u64, 0u64);
    let mut n_chunks = 0usize;
    let mut fill_s = 0.0;
    let mut chunk_bras: Vec<Bra> = Vec::new();
    let mut held = 0usize;
    let mut pi = 0usize;
    let mut single_chunk: Option<Vec<Bra>> = None;
    while pi < st.pairs.len() {
        // Fill up to CHUNK_DOUBLES integrals in batches of 256 bra pairs.
        let t0 = Instant::now();
        while pi < st.pairs.len() && held < CHUNK_DOUBLES {
            let hi = (pi + 256).min(st.pairs.len());
            let batch: Vec<Bra> = pool.install(|| {
                st.pairs[pi..hi]
                    .par_iter()
                    .map(|&(s1, s2)| {
                        let tid = rayon::current_thread_index().unwrap_or(0);
                        let mut eng = engines[tid].lock().unwrap();
                        st.fill(&mut eng, s1, s2)
                    })
                    .collect()
            });
            for b in &batch {
                held += b.ints.len();
                quartets += b.kets.len() as u64;
                nints += b.ints.len() as u64;
                for &(s3, s4, _) in &b.kets {
                    let ni = dims[b.s1] * dims[b.s2] * dims[s3 as usize] * dims[s4 as usize];
                    if ni < 16 {
                        small_q += 1;
                        small_i += ni as u64;
                    }
                }
            }
            chunk_bras.extend(batch);
            pi = hi;
        }
        fill_s += t0.elapsed().as_secs_f64();
        n_chunks += 1;
        // Time the digest variants on this chunk, interleaved.
        for rep in 0..reps {
            let order: Vec<usize> = if rep % 2 == 0 {
                (0..VARS.len()).collect()
            } else {
                (0..VARS.len()).rev().collect()
            };
            for vi in order {
                let acc = if rep == 0 {
                    &mut keep[vi]
                } else {
                    &mut junk[vi]
                };
                let t0 = Instant::now();
                run_var(&c, VARS[vi], &chunk_bras, acc);
                t[vi][rep] += t0.elapsed().as_secs_f64();
            }
        }
        held = 0;
        if pi >= st.pairs.len() && n_chunks == 1 {
            single_chunk = Some(std::mem::take(&mut chunk_bras));
        } else {
            chunk_bras.clear();
        }
    }
    // The O(n²) finishing step of B, timed and added.
    for (vi, v) in VARS.iter().enumerate() {
        if v.is_b() {
            for r in t[vi].iter_mut() {
                let t0 = Instant::now();
                let x = finish(&keep[vi]);
                std::hint::black_box(&x);
                *r += t0.elapsed().as_secs_f64();
            }
        }
    }
    println!(
        "walk: {quartets} quartets, {nints} integrals ({:.2} GB), {n_chunks} chunk(s), fill {fill_s:.1} s on {fill_threads} threads; \
         quartets < 16 integrals: {:.1}% of quartets, {:.1}% of integrals",
        nints as f64 * 8e-9,
        100.0 * small_q as f64 / quartets as f64,
        100.0 * small_i as f64 / nints as f64
    );
    println!("load after timing: {}", uptime());

    // Timings.
    let ia = VARS.iter().position(|&v| v == Var::A16).unwrap();
    let mut ta = t[ia].clone();
    let med_a = median(&mut ta);
    println!(
        "{:>6} {:>9} {:>9} {:>9} {:>8} {:>8}   per-rep s",
        "var", "median s", "min s", "ns/int", "xA16med", "xA16min"
    );
    let min_a = t[ia].iter().cloned().fold(f64::INFINITY, f64::min);
    for (vi, v) in VARS.iter().enumerate() {
        let mut tv = t[vi].clone();
        let med = median(&mut tv);
        let mn = t[vi].iter().cloned().fold(f64::INFINITY, f64::min);
        let reps_s: Vec<String> = t[vi].iter().map(|x| format!("{x:.3}")).collect();
        println!(
            "{:>6} {:>9.3} {:>9.3} {:>9.2} {:>8.2} {:>8.2}   {}",
            format!("{v:?}"),
            med,
            mn,
            med * 1e9 / nints as f64,
            med_a / med,
            min_a / mn,
            reps_s.join(" ")
        );
    }
    // Paired per-rep ratios A16 / Bsym4.
    let ib = VARS.iter().position(|&v| v == Var::Bsym4).unwrap();
    let mut pr: Vec<f64> = (0..reps).map(|r| t[ia][r] / t[ib][r]).collect();
    let prs: Vec<String> = pr.iter().map(|x| format!("{x:.2}")).collect();
    println!(
        "paired A16/Bsym4 per rep: {} (median {:.2})",
        prs.join(" "),
        median(&mut pr)
    );

    // Accuracy against A16, and A16 against the real scatter_bra_pair.
    let (ja, ka) = finish(&keep[ia]);
    let (mj, mk) = (maxabs(&ja), maxabs(&ka));
    println!("max|J| {mj:.3e} max|K| {mk:.3e}");
    let e2 = |j: &Array2<f64>, k: &Array2<f64>| -> f64 { 0.5 * (&st.d * &(j - &(k * 0.5))).sum() };
    let e_a = e2(&ja, &ka);
    for (vi, v) in VARS.iter().enumerate() {
        if vi == ia {
            continue;
        }
        let (j, k) = finish(&keep[vi]);
        let (dj, dk) = (maxabs(&(&j - &ja)), maxabs(&(&k - &ka)));
        let de = e2(&j, &k) - e_a;
        println!(
            "{:>6} vs A16: max|dJ| {dj:.2e} ({:.2e} rel)  max|dK| {dk:.2e} ({:.2e} rel)  dE(1st order) {de:.2e} Ha",
            format!("{v:?}"),
            dj / mj,
            dk / mk
        );
    }
    println!("E2 = 1/2 tr D(J - K/2) (A16) = {e_a:.10} Ha");
    validate_a(&st, &ja, &ka);

    // Thread-count determinism of Bsym4 in the deterministic group partition.
    if let Some(bras) = single_chunk {
        let mut res: Vec<(usize, Array2<f64>, Array2<f64>)> = Vec::new();
        for nt in [1usize, 2, 4] {
            let p = rayon::ThreadPoolBuilder::new()
                .num_threads(nt)
                .build()
                .unwrap();
            let t0 = Instant::now();
            let (j, k) = p.install(|| grouped_b(&c, &bras));
            println!(
                "grouped Bsym4 on {nt} thread(s): {:.3} s wall",
                t0.elapsed().as_secs_f64()
            );
            res.push((nt, j, k));
        }
        let bits = |a: &Array2<f64>, b: &Array2<f64>| {
            a.iter()
                .zip(b.iter())
                .all(|(x, y)| x.to_bits() == y.to_bits())
        };
        for r in &res[1..] {
            println!(
                "grouped Bsym4: {} threads bitwise == 1 thread: J {} K {}",
                r.0,
                bits(&r.1, &res[0].1),
                bits(&r.2, &res[0].2)
            );
        }
        let (dj, dk) = (maxabs(&(&res[0].1 - &ja)), maxabs(&(&res[0].2 - &ka)));
        println!(
            "grouped Bsym4 vs serial A16: max|dJ| {:.2e} rel, max|dK| {:.2e} rel",
            dj / mj,
            dk / mk
        );
    } else {
        println!("(walk spans several chunks: thread-count check skipped)");
    }
    if walk_reps > 0 {
        walks(&st, &c, walk_reps, &ja, &ka);
    }
}

/// In-situ comparison, single thread: the whole walk with libint in the loop,
/// digested by the real `scatter_bra_pair` (A, production code as is) or by
/// `Bsym4` right after each `compute_quartet` (B). Interleaved ABAB...
fn walks(st: &Setup, c: &Ctx, reps: usize, ja: &Array2<f64>, ka: &Array2<f64>) {
    let prep = &st.prep;
    let n = prep.nbasis();
    let (dims, offs) = (prep.shell_dims(), prep.shell_offsets());
    let mut eng = Engine::new_2e(Operator::coulomb(), prep, ERI_PRECISION).unwrap();
    let screen = real_scatter::DensityScreen::SixPair(&st.dmax);
    let (mut ta, mut tb) = (Vec::new(), Vec::new());
    let mut last_b = None;
    println!("in-situ walks: load {}", uptime());
    for rep in 0..2 * reps {
        let do_a = (rep % 2 == 0) == (rep / 2 % 2 == 0);
        let t0 = Instant::now();
        if do_a {
            let mut mode = real_scatter::JkMode::new_both(n);
            for &(s1, s2) in &st.pairs {
                real_scatter::scatter_bra_pair(
                    &mut eng, prep, dims, offs, &st.q, None, None, &screen, THRESH, &st.d, s1, s2,
                    &mut mode, false,
                );
            }
            std::hint::black_box(&mode);
            ta.push(t0.elapsed().as_secs_f64());
        } else {
            let mut out = BOut::new(n, false);
            let mut s = BScratch::default();
            let dm = &st.dmax;
            for &(s1, s2) in &st.pairs {
                let br = b_begin(dims, offs, &st.d, s1, s2, &mut s);
                let b12 = st.q[(s1, s2)];
                for s3 in 0..=s1 {
                    let s4max = if s3 == s1 { s2 } else { s3 };
                    for s4 in 0..=s4max {
                        let d = dm[(s1, s2)]
                            .max(dm[(s3, s4)])
                            .max(dm[(s1, s3)])
                            .max(dm[(s1, s4)])
                            .max(dm[(s2, s3)])
                            .max(dm[(s2, s4)]);
                        if b12 * st.q[(s3, s4)] * d < THRESH {
                            continue;
                        }
                        if let Some(q) = eng.compute_quartet(prep, s1, s2, s3, s4) {
                            b_one::<true, false>(&br, c, &mut out, &mut s, s3, s4, q);
                        }
                    }
                }
                b_end(&br, &mut out, &s);
            }
            let jk = out.finish();
            tb.push(t0.elapsed().as_secs_f64());
            last_b = Some(jk);
        }
    }
    let (ma, mb) = (median(&mut ta.clone()), median(&mut tb.clone()));
    let mina = ta.iter().cloned().fold(f64::INFINITY, f64::min);
    let minb = tb.iter().cloned().fold(f64::INFINITY, f64::min);
    let f = |v: &[f64]| {
        v.iter()
            .map(|x| format!("{x:.2}"))
            .collect::<Vec<_>>()
            .join(" ")
    };
    println!(
        "in-situ walk A (real scatter_bra_pair): median {ma:.3} s min {mina:.3} s  [{}]",
        f(&ta)
    );
    println!(
        "in-situ walk B (Bsym4 after libint):    median {mb:.3} s min {minb:.3} s  [{}]",
        f(&tb)
    );
    println!(
        "in-situ walk speed-up A/B: median {:.3}x, min {:.3}x; saved {:.3} s/walk (median); load {}",
        ma / mb,
        mina / minb,
        ma - mb,
        uptime()
    );
    if let Some((j, k)) = last_b {
        println!(
            "in-situ B vs stored A16: max|dJ| {:.2e} rel max|dK| {:.2e} rel",
            maxabs(&(&j - ja)) / maxabs(ja),
            maxabs(&(&k - ka)) / maxabs(ka)
        );
    }
}

/// `Bsym4` in `DirectJK`'s deterministic group partition: groups of
/// `deterministic_group_size(n_pairs)` consecutive bra pairs, each folded
/// serially into fresh half partials, partials summed in ascending group
/// order (in bands, as `reduce::ordered_stream_sum_pair` does), then
/// `J = Jh + Jhᵀ`, `K = Kb + Kbᵀ` once at the end.
fn grouped_b(c: &Ctx, bras: &[Bra]) -> (Array2<f64>, Array2<f64>) {
    let n = c.d.nrows();
    let gs = ferric_scf::reduce::deterministic_group_size(bras.len());
    let groups: Vec<&[Bra]> = bras.chunks(gs).collect();
    let mut total = BOut::new(n, false);
    for band in groups.chunks(32) {
        let parts: Vec<BOut> = band
            .par_iter()
            .map(|g| {
                let mut o = BOut::new(n, false);
                let mut s = BScratch::default();
                for b in g.iter() {
                    b_bra_pair::<true, false>(c, b, &mut o, &mut s);
                }
                o
            })
            .collect();
        for p in &parts {
            total.add(p);
        }
    }
    total.finish()
}

/// `A16` (the copy) against the real `scatter_bra_pair` with libint on the
/// same walk, serially in the same bra-pair order: must be bitwise equal.
fn validate_a(st: &Setup, ja: &Array2<f64>, ka: &Array2<f64>) {
    let prep = &st.prep;
    let n = prep.nbasis();
    let (dims, offs) = (prep.shell_dims(), prep.shell_offsets());
    let mut eng = Engine::new_2e(Operator::coulomb(), prep, ERI_PRECISION).unwrap();
    let screen = real_scatter::DensityScreen::SixPair(&st.dmax);
    let mut mode = real_scatter::JkMode::new_both(n);
    let t0 = Instant::now();
    let mut count = 0usize;
    for &(s1, s2) in &st.pairs {
        count += real_scatter::scatter_bra_pair(
            &mut eng, prep, dims, offs, &st.q, None, None, &screen, THRESH, &st.d, s1, s2,
            &mut mode, false,
        );
    }
    let real_scatter::JkMode::Both(j, k) = mode else {
        unreachable!()
    };
    let bits = |a: &Array2<f64>, b: &Array2<f64>| {
        a.iter()
            .zip(b.iter())
            .all(|(x, y)| x.to_bits() == y.to_bits())
    };
    println!(
        "A16 copy vs real scatter_bra_pair ({count} quartets, {:.1} s serial): J bitwise {} K bitwise {}",
        t0.elapsed().as_secs_f64(),
        bits(&j, ja),
        bits(&k, ka)
    );
}
