//! perf5 phase 3 (design pass): a standalone measurement harness for the
//! shared-primitive ("b2") kernel idea. NOT wired into any production path:
//! nothing in the library calls this file, and it changes no behaviour.
//!
//! Modes (argv):
//!
//! * `density <xyz> <basis> <out.bin>`: converge an RHF (direct, `gc_rotation
//!   = auto`, 1e-10 / 1e-8: every build is a full build, so `FERRIC_TIMING=1`
//!   gives clean per-build profiles) and write the AO density (raw little-endian
//!   f64, row-major) so the other modes screen with a realistic density.
//! * `profile <xyz> <basis> <dens.bin> <rot|off>`: replay `DirectJK`'s
//!   closed-shell walk (bra prefilter `Q > thresh / (max Q · max D)`,
//!   canonical ket loop, Häser–Ahlrichs six-pair `Q·Q·Dmax < 1e-12`) on the
//!   rotated (`rot`, the OGC level `auto` uses) or parent shells, call libint
//!   for every survivor and report, per angular-momentum class: shell
//!   quartets, primitive quartets, libint thread-seconds, and the primitive
//!   quartets a shared-primitive (group) evaluation would need. Index types:
//!   `S` = an s shell with >= 5 primitives (the C core/valence columns),
//!   `s` = any other s shell, `p`, `d`, `f`.
//! * `kernel <xyz> <basis> <dens.bin> <rot|off>`: the prototype kernel for
//!   every class with l <= 1 on each index, bucketed by l-class. Each (atom,
//!   l) group of columns sharing primitives is evaluated once per group
//!   quartet from one primitive pass and contracted into every column
//!   quartet ((ss|ss): a dedicated F0 kernel; the rest: Obara–Saika VRR per
//!   primitive quartet, contraction, HRR once per column quartet). Checked
//!   against libint's contracted integrals on the same shells (max abs and
//!   relative error) and timed against libint on exactly the screened
//!   quartets the production walk computes, both per group quartet and per
//!   shell quartet (no sharing). Ends with a hybrid walk: the kernel for
//!   (ss|ss) and (ss|sp), libint for every other survivor.
//! * `hybrid <xyz> <basis> <dens.bin> <rot|off> [pairs]` (perf5f): the whole
//!   walk, single thread, digested by the production bra-pair-local digest
//!   (`[scf] jk_digest = "pair"`). `A` is the real code: `PairSpec::both` +
//!   `sum_groups` (one group) around the real `scatter_bra_pair`, compiled in
//!   from `src/quartet_scatter.rs`. `A'` and `B` run a copy of the pair
//!   digest's closed-shell symmetric kernel (checked bitwise against `A` on
//!   the production bra-pair order) on a group-first bra-pair order: `A'`
//!   with libint for every survivor, `B` with the shared-primitive kernel
//!   for every surviving quartet with l1+l2+l3+l4 <= 1 ((ss|ss), (ss|sp) in
//!   any position), evaluated once per group quartet (lazily, on the first
//!   surviving member, cached for the bra group pair's member bra pairs)
//!   and libint for the rest. Only surviving column quartets are digested.
//!   Timed interleaved (A A' B B A' A per pair), with the pair-data setup,
//!   a probed stage breakdown and J/K accuracy against `A`.
//!
//! Run with `RAYON_NUM_THREADS` set; timings in `kernel` and `hybrid` are
//! single-thread.

use std::collections::HashMap;
use std::time::Instant;

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::contraction_rotation::{DerivRotMutant, RotatedBasis, RotationLevel};
use ferric_integrals::engine::Engine;
use ferric_integrals::engine_pool::ERI_PRECISION;
use ferric_integrals::operator::Operator;
use ferric_scf::direct_jk::GcRotate;
use ferric_scf::incore_jk::EriMode;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ndarray::Array2;
use rayon::prelude::*;

// The production J/K scatter and pair digest, compiled in as is. Its few
// `crate::` paths resolve to the shims below (serial, no MPI).
#[allow(dead_code, unused_imports, clippy::all)]
#[path = "../src/quartet_scatter.rs"]
mod real_scatter;

/// Shim for `crate::direct_jk::gc_rotation` (one rank).
mod direct_jk {
    pub mod gc_rotation {
        use ferric_core::memory::pool::{reserve_global, Reservation};
        use ferric_core::parallel::ParallelContext;
        use ferric_core::FerricError;
        pub fn all_ranks_agree(_ctx: &ParallelContext, local: bool) -> bool {
            local
        }
        pub fn reserve_collective(
            _ctx: &ParallelContext,
            label: &str,
            bytes: usize,
        ) -> Result<Reservation, FerricError> {
            reserve_global(label, bytes)
        }
    }
}

/// Shim for `crate::reduce`.
mod reduce {
    pub use ferric_scf::reduce::*;
}

const THRESH: f64 = 1e-12;

fn load_mol(path: &str) -> Molecule {
    let text = std::fs::read_to_string(path).expect("read xyz");
    Molecule::parse_xyz(&text, 0, 1).expect("parse xyz")
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    match a.get(1).map(String::as_str) {
        Some("density") => density(&a[2], &a[3], &a[4]),
        Some("profile") => profile(&a[2], &a[3], &a[4], a[5] == "rot"),
        Some("kernel") => kernel(&a[2], &a[3], &a[4], a[5] == "rot"),
        Some("hybrid") => hybrid(
            &a[2],
            &a[3],
            &a[4],
            a[5] == "rot",
            a.get(6).map_or(5, |s| s.parse().unwrap()),
        ),
        _ => eprintln!("usage: density|profile|kernel ... (see the module doc)"),
    }
}

fn density(xyz: &str, name: &str, out: &str) {
    let mol = load_mol(xyz);
    let prep = PreparedBasis::new(&mol, &basis::bundled(name).unwrap()).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
    let ctx = ParallelContext::default();
    let mut cfg = RhfConfig::default()
        .with_energy_conv(1e-10)
        .with_density_conv(1e-8);
    cfg.eri = EriMode::Direct;
    let cfg = cfg.with_gc_rotation(GcRotate::Auto);
    let r = solve_rhf(&ctx, &mol, &prep, Operator::coulomb(), &bounds, &cfg).unwrap();
    println!("E = {:.12} converged {}", r.energy, r.converged);
    let bytes: Vec<u8> = r
        .density_total
        .iter()
        .flat_map(|x| x.to_le_bytes())
        .collect();
    std::fs::write(out, bytes).unwrap();
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

/// The basis the walk runs on (rotated or parent), its density, Q table and
/// shell-blocked max |D|.
struct Setup {
    prep: PreparedBasis,
    /// The density in the walk's basis.
    d: Array2<f64>,
    q: Array2<f64>,
    dmax: Array2<f64>,
    pairs: Vec<(usize, usize)>,
    /// Per shell: (l, nonzero primitives, type letter, group id).
    info: Vec<(i32, usize, u8, usize)>,
    /// Per group: union primitive count.
    gn: Vec<usize>,
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
    let nsh = prep.nshells();
    // The production shell-blocked max |D| (the same table `DirectJK` screens with).
    let dmax = real_scatter::build_d_max_shell(&prep, &d);
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
    // Groups: shells on one atom with the same l whose exponent sets
    // intersect (bitwise), joined transitively.
    let ls = prep.located_shells();
    let s2a = prep.shell_to_atom().to_vec();
    let nz = |s: usize| -> Vec<u64> {
        ls[s]
            .exponents
            .iter()
            .zip(ls[s].coefficients)
            .filter(|(_, c)| **c != 0.0)
            .map(|(e, _)| e.to_bits())
            .collect()
    };
    let mut parent_of: Vec<usize> = (0..nsh).collect();
    fn find(p: &mut [usize], x: usize) -> usize {
        let mut r = x;
        while p[r] != r {
            r = p[r];
        }
        p[x] = r;
        r
    }
    for a in 0..nsh {
        for b in 0..a {
            if s2a[a] == s2a[b] && ls[a].l == ls[b].l {
                let (ea, eb) = (nz(a), nz(b));
                if ea.iter().any(|x| eb.contains(x)) {
                    let (ra, rb) = (find(&mut parent_of, a), find(&mut parent_of, b));
                    parent_of[ra.max(rb)] = ra.min(rb);
                }
            }
        }
    }
    let mut gid = vec![usize::MAX; nsh];
    let mut root_gid = HashMap::new();
    let mut gexp: Vec<Vec<u64>> = Vec::new();
    for s in 0..nsh {
        let r = find(&mut parent_of, s);
        let g = *root_gid.entry(r).or_insert_with(|| {
            gexp.push(Vec::new());
            gexp.len() - 1
        });
        gid[s] = g;
        for e in nz(s) {
            if !gexp[g].contains(&e) {
                gexp[g].push(e);
            }
        }
    }
    let gn: Vec<usize> = gexp.iter().map(Vec::len).collect();
    let info = (0..nsh)
        .map(|s| {
            let n = nz(s).len();
            let t = match ls[s].l {
                0 if n >= 5 => b'S',
                0 => b's',
                1 => b'p',
                2 => b'd',
                _ => b'f',
            };
            (ls[s].l, n, t, gid[s])
        })
        .collect();
    Setup {
        prep,
        d,
        q,
        dmax,
        pairs,
        info,
        gn,
    }
}

impl Setup {
    /// The screened canonical quartets of bra pair `(s1, s2)`, in walk order.
    fn kets(&self, s1: usize, s2: usize, out: &mut Vec<[usize; 4]>) {
        out.clear();
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
                out.push([s1, s2, s3, s4]);
            }
        }
    }

    fn rank(t: u8) -> u8 {
        match t {
            b's' => 0,
            b'S' => 1,
            b'p' => 2,
            b'd' => 3,
            _ => 4,
        }
    }

    /// Canonical class label `(xy|zw)` from the four index types.
    fn class(&self, q: [usize; 4]) -> [u8; 4] {
        let t = q.map(|s| self.info[s].2);
        let srt = |a: u8, b: u8| {
            if Self::rank(a) >= Self::rank(b) {
                [a, b]
            } else {
                [b, a]
            }
        };
        let (b, k) = (srt(t[0], t[1]), srt(t[2], t[3]));
        let key = |x: [u8; 2]| (Self::rank(x[0]), Self::rank(x[1]));
        if key(b) >= key(k) {
            [b[0], b[1], k[0], k[1]]
        } else {
            [k[0], k[1], b[0], b[1]]
        }
    }

    /// Canonical group-quartet key.
    fn gkey(&self, q: [usize; 4]) -> u64 {
        let g = q.map(|s| self.info[s].3 as u64);
        let (b0, b1) = (g[0].max(g[1]), g[0].min(g[1]));
        let (k0, k1) = (g[2].max(g[3]), g[2].min(g[3]));
        let (bp, kp) = ((b0 << 16) | b1, (k0 << 16) | k1);
        (bp.max(kp) << 32) | bp.min(kp)
    }

    /// Primitive pairs a group pair needs (same group: unordered).
    fn gpair_prims(&self, g1: usize, g2: usize) -> f64 {
        let (a, b) = (self.gn[g1] as f64, self.gn[g2] as f64);
        if g1 == g2 {
            a * (a + 1.0) / 2.0
        } else {
            a * b
        }
    }
}

fn label(c: [u8; 4]) -> String {
    format!(
        "({}{}|{}{})",
        c[0] as char, c[1] as char, c[2] as char, c[3] as char
    )
}

#[derive(Default, Clone)]
struct Acc {
    shellq: u64,
    primq: f64,
    ns: u64,
    groupq: u64,
    gprimq: f64,
}

fn profile(xyz: &str, name: &str, dens: &str, rot: bool) {
    let st = setup(xyz, name, dens, rot);
    let nthreads = rayon::current_num_threads();
    let engines: Vec<std::sync::Mutex<Engine>> = (0..nthreads)
        .map(|_| {
            std::sync::Mutex::new(
                Engine::new_2e(Operator::coulomb(), &st.prep, ERI_PRECISION).unwrap(),
            )
        })
        .collect();
    let t0 = Instant::now();
    type Part = (HashMap<[u8; 4], Acc>, Vec<u64>);
    let (acc, mut gkeys): Part = st
        .pairs
        .par_iter()
        .fold(
            || (HashMap::new(), Vec::new(), Vec::new()),
            |(mut m, mut keys, mut buf): (HashMap<[u8; 4], Acc>, Vec<u64>, Vec<[usize; 4]>),
             &(s1, s2)| {
                st.kets(s1, s2, &mut buf);
                let tid = rayon::current_thread_index().unwrap_or(0);
                let mut eng = engines[tid].lock().unwrap();
                for &q in &buf {
                    let t = Instant::now();
                    let blk = eng.compute_quartet(&st.prep, q[0], q[1], q[2], q[3]);
                    let ok = blk.is_some();
                    let ns = t.elapsed().as_nanos() as u64;
                    let e = m.entry(st.class(q)).or_default();
                    e.shellq += 1;
                    e.ns += ns;
                    if ok {
                        e.primq += q.iter().map(|&s| st.info[s].1 as f64).product::<f64>();
                    }
                    keys.push(st.gkey(q));
                }
                (m, keys, buf)
            },
        )
        .map(|(m, k, _)| (m, k))
        .reduce(
            || (HashMap::new(), Vec::new()),
            |(mut a, mut ka), (b, kb)| {
                for (k, v) in b {
                    let e = a.entry(k).or_default();
                    e.shellq += v.shellq;
                    e.primq += v.primq;
                    e.ns += v.ns;
                }
                ka.extend(kb);
                (a, ka)
            },
        );
    let wall = t0.elapsed().as_secs_f64();
    let mut acc = acc;
    gkeys.sort_unstable();
    gkeys.dedup();
    // Group quartets: class from a representative shell of each group.
    let mut rep = vec![usize::MAX; st.gn.len()];
    for (s, i) in st.info.iter().enumerate() {
        if rep[i.3] == usize::MAX {
            rep[i.3] = s;
        }
    }
    for k in &gkeys {
        let g = [
            ((k >> 48) & 0xffff) as usize,
            ((k >> 32) & 0xffff) as usize,
            ((k >> 16) & 0xffff) as usize,
            (k & 0xffff) as usize,
        ];
        let c = st.class(g.map(|x| rep[x]));
        let e = acc.entry(c).or_default();
        e.groupq += 1;
        e.gprimq += st.gpair_prims(g[0], g[1]) * st.gpair_prims(g[2], g[3]);
    }
    let tot = acc.values().fold(Acc::default(), |mut a, v| {
        a.shellq += v.shellq;
        a.primq += v.primq;
        a.ns += v.ns;
        a.groupq += v.groupq;
        a.gprimq += v.gprimq;
        a
    });
    println!(
        "basis {name} rot {rot}: nsh {} groups {} bra pairs {} threads {nthreads} walk wall {wall:.2} s",
        st.prep.nshells(),
        st.gn.len(),
        st.pairs.len()
    );
    println!(
        "TOTAL shellq {} primq {:.4e} libint {:.3} s-thread  groupq {} group-primq {:.4e}",
        tot.shellq,
        tot.primq,
        tot.ns as f64 * 1e-9,
        tot.groupq,
        tot.gprimq
    );
    let mut rows: Vec<_> = acc.into_iter().collect();
    rows.sort_by_key(|r| std::cmp::Reverse(r.1.ns));
    println!(
        "{:>9} {:>10} {:>11} {:>9} {:>7} {:>8} {:>9} {:>11}",
        "class", "shellq", "primq", "libint s", "%lib", "us/q", "groupq", "g-primq"
    );
    for (c, v) in rows {
        println!(
            "{:>9} {:>10} {:>11.4e} {:>9.3} {:>6.1}% {:>8.3} {:>9} {:>11.4e}",
            label(c),
            v.shellq,
            v.primq,
            v.ns as f64 * 1e-9,
            100.0 * v.ns as f64 / tot.ns as f64,
            v.ns as f64 * 1e-3 / v.shellq.max(1) as f64,
            v.groupq,
            v.gprimq
        );
    }
}

// ---------------------------------------------------------------------------
// The prototype shared-primitive kernel (l <= 1 per index).
// ---------------------------------------------------------------------------

/// Boys F0 by a 7th-order Taylor table on a 0.1 grid (|δ| <= 0.05), the
/// asymptotic form past T = 40. Truncation <= δ^8/8! F_8 ~ 6e-17.
struct Boys {
    /// Per grid point: (-1)^n F_n(T_k)/n!, n = 0..=7 (and F_1.. for l > 0).
    tab: Vec<[f64; 12]>,
}

const BOYS_H: f64 = 0.1;
const BOYS_TMAX: f64 = 40.0;
const BOYS_M: usize = 12; // F_0 .. F_11 stored; F_m..F_{m+7} used for Fm, m <= 4

impl Boys {
    fn new() -> Self {
        let npt = (BOYS_TMAX / BOYS_H) as usize + 2;
        let tab = (0..npt)
            .map(|k| {
                let t = k as f64 * BOYS_H;
                let mtop = BOYS_M - 1;
                // Series for F_mtop, then downward recursion.
                let mut term = 1.0 / (2 * mtop + 1) as f64;
                let mut sum = term;
                let mut i = 1;
                while term > 1e-18 * sum || i < 5 {
                    term *= 2.0 * t / (2 * mtop + 2 * i + 1) as f64;
                    sum += term;
                    i += 1;
                    if i > 2000 {
                        break;
                    }
                }
                let et = (-t).exp();
                let mut f = [0.0; BOYS_M];
                f[mtop] = et * sum;
                for m in (0..mtop).rev() {
                    f[m] = (2.0 * t * f[m + 1] + et) / (2 * m + 1) as f64;
                }
                f
            })
            .collect();
        Boys { tab }
    }

    #[inline(always)]
    fn f0(&self, t: f64) -> f64 {
        if t >= BOYS_TMAX {
            return 0.5 * (std::f64::consts::PI / t).sqrt();
        }
        let k = (t * (1.0 / BOYS_H) + 0.5) as usize;
        let d = t - k as f64 * BOYS_H;
        let f = &self.tab[k];
        // Σ_n (-d)^n F_n / n!
        let md = -d;
        f[0] + md
            * (f[1]
                + md * (f[2] * 0.5
                    + md * (f[3] * (1.0 / 6.0)
                        + md * (f[4] * (1.0 / 24.0)
                            + md * (f[5] * (1.0 / 120.0)
                                + md * (f[6] * (1.0 / 720.0) + md * f[7] * (1.0 / 5040.0)))))))
    }

    /// F_0..=F_mmax at t (mmax <= 4).
    #[inline(always)]
    fn fm(&self, t: f64, mmax: usize, out: &mut [f64; 5]) {
        if t >= BOYS_TMAX {
            let mut f = 0.5 * (std::f64::consts::PI / t).sqrt();
            let it = 0.5 / t;
            for (m, o) in out.iter_mut().enumerate().take(mmax + 1) {
                *o = f;
                f *= (2 * m + 1) as f64 * it;
            }
            return;
        }
        let k = (t * (1.0 / BOYS_H) + 0.5) as usize;
        let d = t - k as f64 * BOYS_H;
        let f = &self.tab[k];
        let md = -d;
        for (m, o) in out.iter_mut().enumerate().take(mmax + 1) {
            let g = &f[m..m + 8];
            *o = g[0]
                + md * (g[1]
                    + md * (g[2] * 0.5
                        + md * (g[3] * (1.0 / 6.0)
                            + md * (g[4] * (1.0 / 24.0)
                                + md * (g[5] * (1.0 / 120.0)
                                    + md * (g[6] * (1.0 / 720.0) + md * g[7] * (1.0 / 5040.0)))))));
        }
    }
}

/// One evaluation group: the union primitives of the columns (shells) of one
/// (atom, l) group, and their libint-normalised coefficients (`n × m`, zero
/// where a column lacks a primitive).
struct Group {
    l: i32,
    center: [f64; 3],
    exps: Vec<f64>,
    coef: Vec<f64>,
    shells: Vec<usize>,
    /// max_col |c| per primitive (libint's pair screen input).
    cmax: Vec<f64>,
}

fn double_factorial(n: i32) -> f64 {
    let mut r = 1.0;
    let mut k = n;
    while k > 1 {
        r *= k as f64;
        k -= 2;
    }
    r
}

/// libint's coefficient handling: fold the primitive norm, then rescale the
/// contraction to unit self-overlap.
fn libint_coefs(l: i32, exps: &[f64], c: &[f64]) -> Vec<f64> {
    use std::f64::consts::PI;
    let lf = l as f64;
    let mut v: Vec<f64> = exps
        .iter()
        .zip(c)
        .map(|(&a, &c)| {
            let two_a = 2.0 * a;
            let n = (2f64.powf(lf) * two_a.powf(lf + 1.5)
                / (PI.powf(1.5) * double_factorial(2 * l - 1)))
            .sqrt();
            c * n
        })
        .collect();
    let mut s = 0.0;
    for (i, &a) in exps.iter().enumerate() {
        for (j, &b) in exps.iter().enumerate() {
            s += v[i] * v[j] * double_factorial(2 * l - 1) * PI.powf(1.5)
                / (2f64.powf(lf) * (a + b).powf(lf + 1.5));
        }
    }
    let f = 1.0 / s.sqrt();
    for x in &mut v {
        *x *= f;
    }
    v
}

fn build_groups(st: &Setup) -> (Vec<Group>, Vec<(usize, usize)>) {
    let ls = st.prep.located_shells();
    let ng = st.gn.len();
    let mut members: Vec<Vec<usize>> = vec![Vec::new(); ng];
    for (s, i) in st.info.iter().enumerate() {
        members[i.3].push(s);
    }
    let mut shell_slot = vec![(0usize, 0usize); st.info.len()];
    let groups = members
        .into_iter()
        .enumerate()
        .map(|(g, sh)| {
            let mut exps: Vec<f64> = Vec::new();
            for &s in &sh {
                for (e, c) in ls[s].exponents.iter().zip(ls[s].coefficients) {
                    if *c != 0.0 && !exps.iter().any(|x| x.to_bits() == e.to_bits()) {
                        exps.push(*e);
                    }
                }
            }
            let (n, m) = (exps.len(), sh.len());
            let mut coef = vec![0.0; n * m];
            for (col, &s) in sh.iter().enumerate() {
                shell_slot[s] = (g, col);
                let (e, c): (Vec<f64>, Vec<f64>) = ls[s]
                    .exponents
                    .iter()
                    .zip(ls[s].coefficients)
                    .filter(|(_, c)| **c != 0.0)
                    .map(|(e, c)| (*e, *c))
                    .unzip();
                let cn = libint_coefs(ls[s].l, &e, &c);
                for (ei, ci) in e.iter().zip(cn) {
                    let i = exps
                        .iter()
                        .position(|x| x.to_bits() == ei.to_bits())
                        .unwrap();
                    coef[i * m + col] = ci;
                }
            }
            let cmax = (0..n)
                .map(|i| (0..m).map(|c| coef[i * m + c].abs()).fold(0.0, f64::max))
                .collect();
            Group {
                l: ls[sh[0]].l,
                center: ls[sh[0]].center,
                exps,
                coef,
                shells: sh,
                cmax,
            }
        })
        .collect();
    (groups, shell_slot)
}

/// Primitive-pair data of a group pair (libint `Original` screen at the
/// engine precision; a same-group pair keeps i <= j and folds (j, i) in).
struct Pair {
    la: i32,
    lb: i32,
    ma: usize,
    mb: usize,
    n: usize,
    p: Vec<f64>,
    pc: Vec<[f64; 3]>,
    /// P - A (for the l = 1 recursions)
    pa: Vec<[f64; 3]>,
    u: Vec<f64>,
    /// `n × (ma·mb)` coefficient products (s|s pairs only; for p pairs the
    /// (j, i) fold is not done and these are plain products).
    cw: Vec<f64>,
    /// HRR terms (contracted side), see [`hrr_terms`].
    hrr: Vec<[(usize, f64); 2]>,
}

fn make_pair(ga: &Group, gb: &Group, same: bool) -> Pair {
    let (ma, mb) = (ga.shells.len(), gb.shells.len());
    let a = ga.center;
    let b = gb.center;
    let ab = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    let r2 = ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2];
    let ln_eps = ERI_PRECISION.ln();
    // The (j, i) fold needs both indices to be s (it swaps the centres' roles).
    let fold = same && ga.l == 0;
    let mut pr = Pair {
        la: ga.l,
        lb: gb.l,
        ma,
        mb,
        n: 0,
        p: vec![],
        pc: vec![],
        pa: vec![],
        u: vec![],
        cw: vec![],
        hrr: hrr_terms(ga.l, gb.l, ab),
    };
    let pref = 2f64.sqrt() * std::f64::consts::PI.powf(1.25);
    for i in 0..ga.exps.len() {
        let j0 = if fold { i } else { 0 };
        for j in j0..gb.exps.len() {
            let (ea, eb) = (ga.exps[i], gb.exps[j]);
            let p = ea + eb;
            let mu = ea * eb / p;
            if ga.cmax[i].ln() + gb.cmax[j].ln() - mu * r2 < ln_eps {
                continue;
            }
            let pc = [
                (ea * a[0] + eb * b[0]) / p,
                (ea * a[1] + eb * b[1]) / p,
                (ea * a[2] + eb * b[2]) / p,
            ];
            pr.p.push(p);
            pr.pc.push(pc);
            pr.pa.push([pc[0] - a[0], pc[1] - a[1], pc[2] - a[2]]);
            pr.u.push(pref * (-mu * r2).exp() / p);
            for ca in 0..ma {
                for cb in 0..mb {
                    let mut w = ga.coef[i * ma + ca] * gb.coef[j * mb + cb];
                    if fold && i != j {
                        w += ga.coef[j * ma + ca] * gb.coef[i * mb + cb];
                    }
                    pr.cw.push(w);
                }
            }
            pr.n += 1;
        }
    }
    pr
}

/// (ss|ss) for every column quartet of a group quartet: `out[(ab)·mcd + cd]`.
#[inline(never)]
fn kernel_ssss(boys: &Boys, bra: &Pair, ket: &Pair, out: &mut [f64], tmp: &mut [f64]) {
    let mab = bra.ma * bra.mb;
    let mcd = ket.ma * ket.mb;
    out[..mab * mcd].fill(0.0);
    for ij in 0..bra.n {
        let p = bra.p[ij];
        let pc = bra.pc[ij];
        let t = &mut tmp[..mcd];
        t.fill(0.0);
        for kl in 0..ket.n {
            let q = ket.p[kl];
            let qc = ket.pc[kl];
            let pq = p + q;
            let rho = p * q / pq;
            let dx = pc[0] - qc[0];
            let dy = pc[1] - qc[1];
            let dz = pc[2] - qc[2];
            let tt = rho * (dx * dx + dy * dy + dz * dz);
            let v = ket.u[kl] / pq.sqrt() * boys.f0(tt);
            let cw = &ket.cw[kl * mcd..(kl + 1) * mcd];
            for (x, w) in t.iter_mut().zip(cw) {
                *x += v * w;
            }
        }
        let ub = bra.u[ij];
        let cw = &bra.cw[ij * mab..(ij + 1) * mab];
        for (ab, w) in cw.iter().enumerate() {
            let w = ub * w;
            for (o, x) in out[ab * mcd..(ab + 1) * mcd].iter_mut().zip(t.iter()) {
                *o += w * x;
            }
        }
    }
}

/// Cartesian components up to l = 2 (s; x, y, z; xx, xy, xz, yy, yz, zz).
const COMP: [[usize; 3]; 10] = [
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [2, 0, 0],
    [1, 1, 0],
    [1, 0, 1],
    [0, 2, 0],
    [0, 1, 1],
    [0, 0, 2],
];
/// Number of Cartesian functions with total l <= L.
const NCUM: [usize; 3] = [1, 4, 10];
/// `INC[e][i]`: index of e + 1_i (e of l <= 1).
const INC: [[usize; 3]; 4] = [[1, 2, 3], [4, 5, 6], [5, 7, 8], [6, 8, 9]];
/// Build rule of e (1..10) by Obara–Saika: (direction i, e - 1_i, N_i(e - 1_i),
/// e - 2·1_i or 0 when N is 0).
const BUILD: [(usize, usize, f64, usize); 10] = [
    (0, 0, 0.0, 0),
    (0, 0, 0.0, 0),
    (1, 0, 0.0, 0),
    (2, 0, 0.0, 0),
    (0, 1, 1.0, 0),
    (1, 1, 0.0, 0),
    (2, 1, 0.0, 0),
    (1, 2, 1.0, 0),
    (2, 2, 0.0, 0),
    (2, 3, 1.0, 0),
];
/// `DEC[e][i]`: e - 1_i, or 99 when e has no i component.
const DEC: [[usize; 3]; 10] = [
    [99, 99, 99],
    [0, 99, 99],
    [99, 0, 99],
    [99, 99, 0],
    [1, 99, 99],
    [2, 1, 99],
    [3, 99, 1],
    [99, 2, 99],
    [99, 3, 2],
    [99, 99, 3],
];

/// Obara–Saika VRR for one primitive quartet: `[e 0 | f 0]` for every
/// |e| <= LAB, |f| <= LCD, into `out[e·NF + f]`. `f` holds F_m (m <= LAB + LCD)
/// already scaled by the primitive prefactor.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn vrr<const LAB: usize, const LCD: usize>(
    f: &[f64; 5],
    p: f64,
    q: f64,
    pq: f64,
    pa: [f64; 3],
    wp: [f64; 3],
    qc: [f64; 3],
    wq: [f64; 3],
    out: &mut [f64; 100],
    ss: &mut [[f64; 5]; 10],
    ef: &mut [[[f64; 3]; 10]; 10],
) {
    let ltot = LAB + LCD;
    let ne = NCUM[LAB];
    let nf = NCUM[LCD];
    ss[0][..=ltot].copy_from_slice(&f[..=ltot]);
    let o2p = 0.5 / p;
    let rp = q / pq;
    for e in 1..ne {
        let (i, e1, n, e2) = BUILD[e];
        let le = COMP[e][0] + COMP[e][1] + COMP[e][2];
        for m in 0..=(ltot - le) {
            let mut v = pa[i] * ss[e1][m] + wp[i] * ss[e1][m + 1];
            if n != 0.0 {
                v += n * o2p * (ss[e2][m] - rp * ss[e2][m + 1]);
            }
            ss[e][m] = v;
        }
    }
    if LCD == 0 {
        for e in 0..ne {
            out[e] = ss[e][0];
        }
        return;
    }
    let o2q = 0.5 / q;
    let rq = p / pq;
    let o2pq = 0.5 / pq;
    // ef[e][f][m], m <= LCD - |f|
    for e in 0..ne {
        ef[e][0][..=LCD].copy_from_slice(&ss[e][..=LCD]);
    }
    for fi in 1..nf {
        let (i, f1, n, f2) = BUILD[fi];
        let lf = COMP[fi][0] + COMP[fi][1] + COMP[fi][2];
        for e in 0..ne {
            let ne_i = COMP[e][i] as f64;
            let em = DEC[e][i];
            for m in 0..=(LCD - lf) {
                let mut v = qc[i] * ef[e][f1][m] + wq[i] * ef[e][f1][m + 1];
                if n != 0.0 {
                    v += n * o2q * (ef[e][f2][m] - rq * ef[e][f2][m + 1]);
                }
                if em != 99 {
                    v += ne_i * o2pq * ef[em][f1][m + 1];
                }
                ef[e][fi][m] = v;
            }
        }
    }
    for e in 0..ne {
        for fi in 0..nf {
            out[e * nf + fi] = ef[e][fi][0];
        }
    }
}

/// HRR terms of one side: per output function (a-major over the two
/// shells' Cartesian functions), up to two `(e index, coefficient)`.
fn hrr_terms(la: i32, lb: i32, ab: [f64; 3]) -> Vec<[(usize, f64); 2]> {
    let fa: Vec<usize> = if la == 0 { vec![0] } else { vec![1, 2, 3] };
    let mut t = Vec::new();
    for &a in &fa {
        if lb == 0 {
            t.push([(a, 1.0), (0, 0.0)]);
        } else {
            for i in 0..3 {
                t.push([(INC[a][i], 1.0), (a, ab[i])]);
            }
        }
    }
    t
}

/// Shared-primitive kernel for l <= 1 per index (Head-Gordon–Pople: VRR per
/// primitive quartet, contraction into every column pair, HRR once per
/// contracted column quartet). Cartesian p in x, y, z order. Output layout
/// `out[(ab·nfab + fab)·(mcd·nfcd) + cd·nfcd + fcd]`.
#[inline(never)]
fn kernel_l1(boys: &Boys, bra: &Pair, ket: &Pair, out: &mut [f64], scratch: &mut Vec<f64>) {
    let lab = (bra.la + bra.lb) as usize;
    let lcd = (ket.la + ket.lb) as usize;
    match (lab, lcd) {
        (0, 0) => kernel_l1_t::<0, 0>(boys, bra, ket, out, scratch),
        (1, 0) => kernel_l1_t::<1, 0>(boys, bra, ket, out, scratch),
        (0, 1) => kernel_l1_t::<0, 1>(boys, bra, ket, out, scratch),
        (1, 1) => kernel_l1_t::<1, 1>(boys, bra, ket, out, scratch),
        (2, 0) => kernel_l1_t::<2, 0>(boys, bra, ket, out, scratch),
        (0, 2) => kernel_l1_t::<0, 2>(boys, bra, ket, out, scratch),
        (2, 1) => kernel_l1_t::<2, 1>(boys, bra, ket, out, scratch),
        (1, 2) => kernel_l1_t::<1, 2>(boys, bra, ket, out, scratch),
        (2, 2) => kernel_l1_t::<2, 2>(boys, bra, ket, out, scratch),
        _ => unreachable!(),
    }
}

fn kernel_l1_t<const LAB: usize, const LCD: usize>(
    boys: &Boys,
    bra: &Pair,
    ket: &Pair,
    out: &mut [f64],
    scratch: &mut Vec<f64>,
) {
    let ne = NCUM[LAB];
    let nf = NCUM[LCD];
    let nef = ne * nf;
    let mab = bra.ma * bra.mb;
    let mcd = ket.ma * ket.mb;
    let need = (mcd + mab * mcd) * nef;
    if scratch.len() < need {
        scratch.resize(need, 0.0);
    }
    let (h, acc) = scratch.split_at_mut(mcd * nef);
    let acc = &mut acc[..mab * mcd * nef];
    acc.fill(0.0);
    let ltot = LAB + LCD;
    let mut v = [0.0f64; 100];
    let mut ss = [[0.0f64; 5]; 10];
    let mut ef = [[[0.0f64; 3]; 10]; 10];
    for ij in 0..bra.n {
        let p = bra.p[ij];
        let pc = bra.pc[ij];
        let pa = bra.pa[ij];
        h.fill(0.0);
        for kl in 0..ket.n {
            let q = ket.p[kl];
            let qc = ket.pc[kl];
            let pq = p + q;
            let rho = p * q / pq;
            let d = [pc[0] - qc[0], pc[1] - qc[1], pc[2] - qc[2]];
            let tt = rho * (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
            let mut f = [0.0; 5];
            boys.fm(tt, ltot, &mut f);
            let pre = ket.u[kl] / pq.sqrt();
            for x in f.iter_mut().take(ltot + 1) {
                *x *= pre;
            }
            let a = q / pq;
            let b = p / pq;
            let wp = [-a * d[0], -a * d[1], -a * d[2]];
            let wq = [b * d[0], b * d[1], b * d[2]];
            vrr::<LAB, LCD>(
                &f, p, q, pq, pa, wp, ket.pa[kl], wq, &mut v, &mut ss, &mut ef,
            );
            let cw = &ket.cw[kl * mcd..(kl + 1) * mcd];
            for (cd, w) in cw.iter().enumerate() {
                for (o, x) in h[cd * nef..(cd + 1) * nef].iter_mut().zip(&v[..nef]) {
                    *o += w * x;
                }
            }
        }
        let ub = bra.u[ij];
        let cw = &bra.cw[ij * mab..(ij + 1) * mab];
        for (ab, w) in cw.iter().enumerate() {
            let w = ub * w;
            for (o, x) in acc[ab * mcd * nef..(ab + 1) * mcd * nef]
                .iter_mut()
                .zip(h.iter())
            {
                *o += w * x;
            }
        }
    }
    // HRR on the contracted [e0|f0], once per column quartet.
    let hb = &bra.hrr;
    let hk = &ket.hrr;
    let (nfab, nfcd) = (hb.len(), hk.len());
    for ab in 0..mab {
        for cd in 0..mcd {
            let src = &acc[(ab * mcd + cd) * nef..(ab * mcd + cd + 1) * nef];
            for (fab, tb) in hb.iter().enumerate() {
                let dst0 = (ab * nfab + fab) * (mcd * nfcd) + cd * nfcd;
                for (fcd, tk) in hk.iter().enumerate() {
                    let mut s = 0.0;
                    for &(e, cb) in tb {
                        if cb == 0.0 {
                            continue;
                        }
                        for &(fi, ck) in tk {
                            if ck == 0.0 {
                                continue;
                            }
                            s += cb * ck * src[e * nf + fi];
                        }
                    }
                    out[dst0 + fcd] = s;
                }
            }
        }
    }
}

fn kernel(xyz: &str, name: &str, dens: &str, rot: bool) {
    let st = setup(xyz, name, dens, rot);
    let boys = Boys::new();
    let (groups, slot) = build_groups(&st);
    let mut eng = Engine::new_2e(Operator::coulomb(), &st.prep, ERI_PRECISION).unwrap();
    let dims = st.prep.shell_dims().to_vec();
    // Surviving canonical shell quartets with l <= 1 everywhere, bucketed by
    // all-s vs l <= 1 with a p, and the group quartets they need.
    let mut buf = Vec::new();
    let mut by_class: std::collections::BTreeMap<String, Vec<[usize; 4]>> = Default::default();
    for &(s1, s2) in &st.pairs {
        st.kets(s1, s2, &mut buf);
        for &q in &buf {
            let ls = q.map(|s| st.info[s].0);
            if ls.iter().all(|&l| l <= 1) {
                let c = st.class(q).map(|t| if t == b'S' { b's' } else { t });
                // l-class label with S merged into s, bra/ket re-sorted.
                let mut b = [c[0], c[1]];
                let mut k = [c[2], c[3]];
                b.sort_by(|x, y| y.cmp(x));
                k.sort_by(|x, y| y.cmp(x));
                let (b, k) = if b >= k { (b, k) } else { (k, b) };
                by_class
                    .entry(label([b[0], b[1], k[0], k[1]]))
                    .or_default()
                    .push(q);
            }
        }
    }
    println!("basis {name} rot {rot}: groups {}", groups.len());
    // Hybrid walk: the kernel for (ss|ss) and (ss|sp), libint for the rest.
    let (mut t_lib_k, mut t_grp_k) = (0.0, 0.0);
    for (label, qs) in &by_class {
        let (tl, tg) = run_class(label, qs, &st, &groups, &slot, &boys, &mut eng, &dims);
        if label == "(ss|ss)" || label == "(ss|sp)" {
            t_lib_k += tl;
            t_grp_k += tg;
        }
    }
    let mut rest: Vec<[usize; 4]> = Vec::new();
    for &(s1, s2) in &st.pairs {
        st.kets(s1, s2, &mut buf);
        for &q in &buf {
            let n_s = q.iter().filter(|&&s| st.info[s].0 == 0).count();
            let n_p = q.iter().filter(|&&s| st.info[s].0 == 1).count();
            if !(n_s == 4 || (n_s == 3 && n_p == 1)) {
                rest.push(q);
            }
        }
    }
    let mut t_rest = f64::MAX;
    let mut sink = 0.0;
    for _ in 0..2 {
        let t = Instant::now();
        for q in &rest {
            if let Some(v) = eng.compute_quartet(&st.prep, q[0], q[1], q[2], q[3]) {
                sink += v[0];
            }
        }
        t_rest = t_rest.min(t.elapsed().as_secs_f64());
    }
    let base = t_rest + t_lib_k;
    let hyb = t_rest + t_grp_k;
    println!(
        "[hybrid walk, 1 thread, integrals only] libint everywhere {base:.3} s | kernel for \
         (ss|ss)+(ss|sp), libint elsewhere {hyb:.3} s | integral-time speed-up {:.3}x \
         ({} other quartets on libint) [sink {sink:.3e}]",
        base / hyb,
        rest.len()
    );
}

/// Group quartets (bra group pair, ket group pair) in first-seen order, with
/// the shell quartets each covers.
fn group_quartets(
    qs: &[[usize; 4]],
    st: &Setup,
) -> Vec<((usize, usize), (usize, usize), Vec<[usize; 4]>)> {
    let mut idx: HashMap<(usize, usize, usize, usize), usize> = HashMap::new();
    let mut out: Vec<((usize, usize), (usize, usize), Vec<[usize; 4]>)> = Vec::new();
    for &q in qs {
        // Orient: bra pair (g1 >= g2) as ordered shells, so the group pair is
        // the shells' groups in the quartet's own order.
        let g = q.map(|s| st.info[s].3);
        let key = (g[0], g[1], g[2], g[3]);
        let i = *idx.entry(key).or_insert_with(|| {
            out.push(((g[0], g[1]), (g[2], g[3]), Vec::new()));
            out.len() - 1
        });
        out[i].2.push(q);
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn run_class(
    label: &str,
    qs: &[[usize; 4]],
    st: &Setup,
    groups: &[Group],
    slot: &[(usize, usize)],
    boys: &Boys,
    eng: &mut Engine,
    dims: &[usize],
) -> (f64, f64) {
    if qs.is_empty() {
        return (0.0, 0.0);
    }
    let gq = group_quartets(qs, st);
    // Pair data, cached per ordered group pair.
    let mut pairs: HashMap<(usize, usize), Pair> = HashMap::new();
    let tp = Instant::now();
    for (b, k, _) in &gq {
        for &(x, y) in [b, k] {
            pairs
                .entry((x, y))
                .or_insert_with(|| make_pair(&groups[x], &groups[y], x == y));
        }
    }
    let pair_s = tp.elapsed().as_secs_f64();
    // Per-shell-quartet pairs (no sharing: each shell its own one-column group).
    let single: Vec<Group> = (0..st.info.len())
        .map(|s| {
            let g = &groups[slot[s].0];
            let col = slot[s].1;
            let m = g.shells.len();
            let keep: Vec<usize> = (0..g.exps.len())
                .filter(|&i| g.coef[i * m + col] != 0.0)
                .collect();
            Group {
                l: g.l,
                center: g.center,
                exps: keep.iter().map(|&i| g.exps[i]).collect(),
                coef: keep.iter().map(|&i| g.coef[i * m + col]).collect(),
                shells: vec![s],
                cmax: keep.iter().map(|&i| g.coef[i * m + col].abs()).collect(),
            }
        })
        .collect();
    let mut spairs: HashMap<(usize, usize), Pair> = HashMap::new();
    for q in qs {
        for (x, y) in [(q[0], q[1]), (q[2], q[3])] {
            spairs
                .entry((x, y))
                .or_insert_with(|| make_pair(&single[x], &single[y], x == y));
        }
    }
    let is_ss = qs[0].iter().all(|&s| st.info[s].0 == 0);
    let mut out = vec![0.0; 1 << 16];
    let mut tmp = vec![0.0; 1 << 12];
    let mut prim = Vec::new();
    let mut run = |b: &Pair, k: &Pair, out: &mut [f64]| {
        if is_ss {
            kernel_ssss(boys, b, k, out, &mut tmp);
        } else {
            kernel_l1(boys, b, k, out, &mut prim);
        }
    };
    // --- accuracy: every covered shell quartet against libint ---
    let mut max_abs = 0.0f64;
    let mut max_val = 0.0f64;
    let mut max_rel = 0.0f64; // over |I| > 1e-10
    let mut n_checked = 0usize;
    // libint pure p is (y, z, x) = Cartesian index (1, 2, 0)
    let pure_p = st.prep.located_shells().iter().any(|s| s.l == 1 && s.pure);
    let pmap = |f: usize| if pure_p { [1, 2, 0][f] } else { f };
    for (b, k, members) in &gq {
        let (pb, pk) = (&pairs[b], &pairs[k]);
        run(pb, pk, &mut out);
        let nfab = [1, 3][pb.la as usize] * [1, 3][pb.lb as usize];
        let nfcd = [1, 3][pk.la as usize] * [1, 3][pk.lb as usize];
        let mcd = pk.ma * pk.mb;
        for q in members {
            let lib = eng
                .compute_quartet(&st.prep, q[0], q[1], q[2], q[3])
                .map(<[f64]>::to_vec);
            let (c1, c2, c3, c4) = (slot[q[0]].1, slot[q[1]].1, slot[q[2]].1, slot[q[3]].1);
            let ab = c1 * pb.mb + c2;
            let cd = c3 * pk.mb + c4;
            let d = [dims[q[0]], dims[q[1]], dims[q[2]], dims[q[3]]];
            for i in 0..d[0] {
                for j in 0..d[1] {
                    for kk in 0..d[2] {
                        for l in 0..d[3] {
                            let li = ((i * d[1] + j) * d[2] + kk) * d[3] + l;
                            let ia = if d[0] == 3 { pmap(i) } else { 0 };
                            let ib = if d[1] == 3 { pmap(j) } else { 0 };
                            let ic = if d[2] == 3 { pmap(kk) } else { 0 };
                            let id = if d[3] == 3 { pmap(l) } else { 0 };
                            let fab = ia * d[1] + ib;
                            let fcd = ic * d[3] + id;
                            let mine = out[(ab * nfab + fab) * (mcd * nfcd) + cd * nfcd + fcd];
                            let theirs = lib.as_ref().map_or(0.0, |v| v[li]);
                            let e = (mine - theirs).abs();
                            max_abs = max_abs.max(e);
                            max_val = max_val.max(theirs.abs());
                            if theirs.abs() > 1e-10 {
                                max_rel = max_rel.max(e / theirs.abs());
                            }
                            n_checked += 1;
                        }
                    }
                }
            }
        }
    }
    // --- timing (single thread, best of 3) ---
    let reps = 3;
    let mut t_lib = f64::MAX;
    let mut sink = 0.0;
    for _ in 0..reps {
        let t = Instant::now();
        for q in qs {
            if let Some(v) = eng.compute_quartet(&st.prep, q[0], q[1], q[2], q[3]) {
                sink += v[0];
            }
        }
        t_lib = t_lib.min(t.elapsed().as_secs_f64());
    }
    let mut t_grp = f64::MAX;
    for _ in 0..reps {
        let t = Instant::now();
        for (b, k, _) in &gq {
            run(&pairs[b], &pairs[k], &mut out);
            sink += out[0];
        }
        t_grp = t_grp.min(t.elapsed().as_secs_f64());
    }
    let mut t_one = f64::MAX;
    for _ in 0..reps {
        let t = Instant::now();
        for q in qs {
            run(&spairs[&(q[0], q[1])], &spairs[&(q[2], q[3])], &mut out);
            sink += out[0];
        }
        t_one = t_one.min(t.elapsed().as_secs_f64());
    }
    let prim_lib: f64 = qs
        .iter()
        .map(|q| (spairs[&(q[0], q[1])].n * spairs[&(q[2], q[3])].n) as f64)
        .sum();
    let colq: usize = gq
        .iter()
        .map(|(b, k, _)| pairs[b].ma * pairs[b].mb * pairs[k].ma * pairs[k].mb)
        .sum();
    let prim_grp: f64 = gq
        .iter()
        .map(|(b, k, _)| (pairs[b].n * pairs[k].n) as f64)
        .sum();
    println!(
        "[{label}] shell quartets {} -> group quartets {} computing {colq} column quartets; \
         screened primitive quartets: per shell {prim_lib:.4e}, per group {prim_grp:.4e} \
         ({:.2}x fewer)",
        qs.len(),
        gq.len(),
        prim_lib / prim_grp.max(1.0)
    );
    println!(
        "[{label}] accuracy over {n_checked} integrals: max |err| {max_abs:.3e}, max |I| \
         {max_val:.3e}, err/scale {:.3e}, max rel (|I|>1e-10) {max_rel:.3e}",
        max_abs / max_val.max(1e-300)
    );
    println!(
        "[{label}] 1-thread time: libint {t_lib:.4} s ({:.3} us/q) | kernel per shell \
         quartet {t_one:.4} s ({:.2}x) | kernel per group quartet {t_grp:.4} s ({:.2}x) \
         | pair setup {pair_s:.4} s (once per geometry) [sink {sink:.3e}]",
        t_lib * 1e6 / qs.len() as f64,
        t_lib / t_one,
        t_lib / t_grp
    );
    (t_lib, t_grp)
}

// ---------------------------------------------------------------------------
// perf5f: the hybrid walk with the production pair digest (`hybrid` mode).

/// `Σ a[i] b[i]` over `N` (or, for `N = 0`, `n`) entries (copy of the pair
/// digest's `dotp`).
#[inline(always)]
unsafe fn dotp<const N: usize>(a: *const f64, b: *const f64, n: usize) -> f64 {
    let n = if N > 0 { N } else { n };
    let mut s = 0.0;
    for i in 0..n {
        s += *a.add(i) * *b.add(i);
    }
    s
}

/// `y[i] += a x[i]` (copy of the pair digest's `axpyp`).
#[inline(always)]
unsafe fn axpyp<const N: usize>(y: *mut f64, a: f64, x: *const f64, n: usize) {
    let n = if N > 0 { N } else { n };
    for i in 0..n {
        *y.add(i) += a * *x.add(i);
    }
}

/// A copy of the production pair digest's closed-shell symmetric kernel
/// (`PairJk::{begin, quartet, end}` with J, one exchange channel and
/// `GEN = false`; `pair_quartet::<N4, true, 1, false>`), on the symmetrised
/// density. The copy exists only because those methods are private to
/// `quartet_scatter`; every `hybrid` run checks it bitwise against the real
/// `PairSpec::sum_groups` + `scatter_bra_pair` on the production order.
struct PairCopy {
    n: usize,
    /// `½(D + Dᵀ)` (what `PairSpec::materialize` makes for a density
    /// symmetric to roundoff); also the Coulomb density (`Channel0`).
    ds: Array2<f64>,
    jh: Array2<f64>,
    kb: Array2<f64>,
    jb: Vec<f64>,
    e12: Vec<f64>,
    e34: Vec<f64>,
    s34: Vec<f64>,
    bn: [usize; 2],
    bo: [usize; 2],
    bs: [usize; 2],
    sym12: bool,
    quartets: usize,
}

impl PairCopy {
    fn new(d: &Array2<f64>, max_dim: usize) -> Self {
        let n = d.nrows();
        let mut ds = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..=i {
                let (a, b) = (d[(i, j)], d[(j, i)]);
                let s = a + b;
                let v = if s.is_finite() {
                    0.5 * s
                } else {
                    0.5 * a + 0.5 * b
                };
                ds[(i, j)] = v;
                ds[(j, i)] = v;
            }
        }
        let m = max_dim * max_dim;
        PairCopy {
            n,
            ds,
            jh: Array2::zeros((n, n)),
            kb: Array2::zeros((n, n)),
            jb: vec![0.0; m],
            e12: vec![0.0; m],
            e34: vec![0.0; m],
            s34: vec![0.0; m],
            bn: [0; 2],
            bo: [0; 2],
            bs: [0; 2],
            sym12: false,
            quartets: 0,
        }
    }

    fn begin(&mut self, dims: &[usize], offs: &[usize], s1: usize, s2: usize) {
        let (n1, n2) = (dims[s1], dims[s2]);
        let (o1, o2) = (offs[s1], offs[s2]);
        let sym12 = s1 != s2;
        self.bn = [n1, n2];
        self.bo = [o1, o2];
        self.bs = [s1, s2];
        self.sym12 = sym12;
        self.jb[..n1 * n2].fill(0.0);
        for a in 0..n1 {
            for b in 0..n2 {
                let mut v = self.ds[(o1 + a, o2 + b)];
                if sym12 {
                    v += self.ds[(o2 + b, o1 + a)];
                }
                self.e12[a * n2 + b] = v;
            }
        }
    }

    fn end(&mut self) {
        let [n1, n2] = self.bn;
        let [o1, o2] = self.bo;
        let w12 = if self.sym12 { 1.0 } else { 0.5 };
        for a in 0..n1 {
            for b in 0..n2 {
                self.jh[(o1 + a, o2 + b)] += w12 * self.jb[a * n2 + b];
            }
        }
    }

    fn quartet(&mut self, q: &[f64], dims: &[usize], offs: &[usize], s3: usize, s4: usize) {
        self.quartets += 1;
        let [n1, n2] = self.bn;
        let (n3, n4) = (dims[s3], dims[s4]);
        assert!(q.len() >= n1 * n2 * n3 * n4, "pair digest: short block");
        // SAFETY: every index is inside its shell block of an n × n matrix,
        // the scratch holds max_dim² entries, q holds n1·n2·n3·n4 integrals.
        unsafe {
            match n4 {
                1 => self.quartet_t::<1>(q.as_ptr(), dims, offs, s3, s4),
                3 => self.quartet_t::<3>(q.as_ptr(), dims, offs, s3, s4),
                5 => self.quartet_t::<5>(q.as_ptr(), dims, offs, s3, s4),
                _ => self.quartet_t::<0>(q.as_ptr(), dims, offs, s3, s4),
            }
        }
    }

    #[inline(always)]
    unsafe fn quartet_t<const N4: usize>(
        &mut self,
        q: *const f64,
        dims: &[usize],
        offs: &[usize],
        s3: usize,
        s4: usize,
    ) {
        let [n1, n2] = self.bn;
        let (n3, n4) = (dims[s3], dims[s4]);
        let [o1, o2] = self.bo;
        let (o3, o4) = (offs[s3], offs[s4]);
        let sym12 = self.sym12;
        let sym34 = s3 != s4;
        let sym1234 = (self.bs[0], self.bs[1]) != (s3, s4);
        let n4 = if N4 > 0 { N4 } else { n4 };
        let n = self.n;
        let n34 = n3 * n4;
        let jd = self.ds.as_ptr();
        let dk = jd;
        let jh = self.jh.as_mut_ptr();
        let kb = self.kb.as_mut_ptr();
        let e34 = self.e34.as_mut_ptr();
        let s34 = self.s34.as_mut_ptr();
        let e12p = self.e12.as_ptr();
        let jbp = self.jb.as_mut_ptr();
        for c in 0..n3 {
            for dd in 0..n4 {
                let mut v = *jd.add((o3 + c) * n + o4 + dd);
                if sym34 {
                    v += *jd.add((o4 + dd) * n + o3 + c);
                }
                *e34.add(c * n4 + dd) = v;
            }
        }
        if sym1234 {
            for i in 0..n34 {
                *s34.add(i) = 0.0;
            }
        }
        let w = if sym1234 { 1.0 } else { 0.5 };
        for a in 0..n1 {
            let mu = o1 + a;
            for b in 0..n2 {
                let nu = o2 + b;
                let ab = a * n2 + b;
                let e12 = *e12p.add(ab);
                let mut jacc = 0.0;
                for c in 0..n3 {
                    let la = o3 + c;
                    let r = q.add((ab * n3 + c) * n4);
                    jacc += dotp::<N4>(r, e34.add(c * n4), n4);
                    if sym1234 {
                        axpyp::<N4>(s34.add(c * n4), e12, r, n4);
                    }
                    let (dmu, dnu) = (dk.add(mu * n), dk.add(nu * n));
                    let (kmu, knu) = (kb.add(mu * n), kb.add(nu * n));
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
                }
                *jbp.add(ab) += jacc;
            }
        }
        if sym1234 {
            let w34 = if sym34 { 1.0 } else { 0.5 };
            for c in 0..n3 {
                for dd in 0..n4 {
                    *jh.add((o3 + c) * n + o4 + dd) += w34 * *s34.add(c * n4 + dd);
                }
            }
        }
    }
}

/// Counters and (probed) stage times of one copy walk.
#[derive(Default, Debug, Clone)]
struct HybStats {
    survivors: u64,
    libint_calls: u64,
    libint_digested: u64,
    kernel_quartets: u64,
    kernel_group_evals: u64,
    kernel_column_quartets: u64,
    t_lib: f64,
    t_ker: f64,
    t_dig: f64,
}

/// Everything a copy walk needs besides the engine.
struct Hyb<'a> {
    st: &'a Setup,
    slot: &'a [(usize, usize)],
    boys: &'a Boys,
    ng: usize,
    /// Pair data index per ordered group pair (`u32::MAX`: none, l_a+l_b > 1).
    pidx: Vec<u32>,
    pairs: Vec<Pair>,
    pure_p: bool,
    max_dim: usize,
}

/// Bra pairs of the walk in a group-first order: production order, except
/// that the first bra pair with l1 + l2 <= 1 of an ordered bra group pair
/// pulls in all the other members of that group pair right after it.
fn group_first_order(st: &Setup) -> Vec<(usize, usize)> {
    let gid = |s: usize| st.info[s].3;
    let l = |s: usize| st.info[s].0;
    let mut members: HashMap<(usize, usize), Vec<(usize, usize)>> = HashMap::new();
    for &(s1, s2) in &st.pairs {
        if l(s1) + l(s2) <= 1 {
            members
                .entry((gid(s1), gid(s2)))
                .or_default()
                .push((s1, s2));
        }
    }
    let mut out = Vec::with_capacity(st.pairs.len());
    for &(s1, s2) in &st.pairs {
        if l(s1) + l(s2) <= 1 {
            if let Some(m) = members.remove(&(gid(s1), gid(s2))) {
                out.extend(m);
            }
        } else {
            out.push((s1, s2));
        }
    }
    assert_eq!(out.len(), st.pairs.len());
    out
}

/// The real production pair-digest walk: `PairSpec::both` +
/// `sum_groups` (one group) around the real `scatter_bra_pair`.
fn walk_real(
    st: &Setup,
    eng: &std::sync::Mutex<Engine>,
    order: &[(usize, usize)],
) -> (Array2<f64>, Array2<f64>) {
    let prep = &st.prep;
    let n = prep.nbasis();
    let (dims, offs) = (prep.shell_dims(), prep.shell_offsets());
    let screen = real_scatter::DensityScreen::SixPair(&st.dmax);
    let ctx = ParallelContext::default();
    let mut spec = real_scatter::PairSpec::both(n, dims, &st.d, real_scatter::SymScale::Own);
    assert_eq!(spec.kernel_label(), "symmetric");
    let out = spec
        .sum_groups(&ctx, 1, 1 << 30, None, |_, mode| {
            let mut e = eng.lock().unwrap();
            for &(s1, s2) in order {
                real_scatter::scatter_bra_pair(
                    &mut e, prep, dims, offs, &st.q, None, None, &screen, THRESH, &st.d, s1, s2,
                    mode, false,
                );
            }
        })
        .unwrap();
    (out.j.unwrap(), out.k.into_iter().next().unwrap())
}

/// The copy walk: `KERNEL = false` is `A'` (libint everywhere), `true` is
/// `B` (the shared-primitive kernel for every surviving quartet with
/// l1+l2+l3+l4 <= 1). `PROBE` adds stage clocks.
fn walk_copy<const KERNEL: bool, const PROBE: bool>(
    h: &Hyb,
    eng: &mut Engine,
    order: &[(usize, usize)],
    stats: &mut HybStats,
) -> (Array2<f64>, Array2<f64>) {
    let st = h.st;
    let prep = &st.prep;
    let n = prep.nbasis();
    let (dims, offs) = (prep.shell_dims(), prep.shell_offsets());
    let screen = real_scatter::DensityScreen::SixPair(&st.dmax);
    let mut pc = PairCopy::new(&st.d, h.max_dim);
    let ng = h.ng;
    let gid = |s: usize| st.info[s].3;
    let lv = |s: usize| st.info[s].0 as usize;
    // Kernel cache: one output block per ordered ket group pair, valid for
    // the current bra group pair (generation stamp).
    let mut stamp: Vec<u32> = if KERNEL { vec![0; ng * ng] } else { Vec::new() };
    let mut cache: Vec<Vec<f64>> = if KERNEL {
        vec![Vec::new(); ng * ng]
    } else {
        Vec::new()
    };
    let mut gen = 0u32;
    let mut cur_bra = usize::MAX;
    let mut tmp = vec![0.0; 1 << 12];
    let mut prim = Vec::new();
    let mut qbuf = [0.0f64; 3];
    let pmap = |f: usize| if h.pure_p { [1, 2, 0][f] } else { f };
    for &(s1, s2) in order {
        let l12 = lv(s1) + lv(s2);
        let kbra = KERNEL && l12 <= 1;
        let bkey = gid(s1) * ng + gid(s2);
        if kbra && bkey != cur_bra {
            cur_bra = bkey;
            gen += 1;
        }
        let t = PROBE.then(Instant::now);
        pc.begin(dims, offs, s1, s2);
        if let Some(t) = t {
            stats.t_dig += t.elapsed().as_secs_f64();
        }
        let b12 = st.q[(s1, s2)];
        let (c1, c2) = (h.slot[s1].1, h.slot[s2].1);
        for s3 in 0..=s1 {
            let s4max = if s3 == s1 { s2 } else { s3 };
            for s4 in 0..=s4max {
                let b34 = st.q[(s3, s4)];
                let dmax = screen.dmax(s1, s2, s3, s4);
                let bound = b12 * b34;
                if bound * dmax < THRESH {
                    continue;
                }
                stats.survivors += 1;
                if kbra && l12 + lv(s3) + lv(s4) <= 1 {
                    let key = gid(s3) * ng + gid(s4);
                    let pb = &h.pairs[h.pidx[bkey] as usize];
                    let pk = &h.pairs[h.pidx[key] as usize];
                    let out = &mut cache[key];
                    if stamp[key] != gen {
                        stamp[key] = gen;
                        let t = PROBE.then(Instant::now);
                        let len = pb.ma * pb.mb * pk.ma * pk.mb * pb.hrr.len() * pk.hrr.len();
                        if out.len() < len {
                            out.resize(len, 0.0);
                        }
                        if l12 == 0 && lv(s3) + lv(s4) == 0 {
                            kernel_ssss(h.boys, pb, pk, out, &mut tmp);
                        } else {
                            kernel_l1(h.boys, pb, pk, out, &mut prim);
                        }
                        stats.kernel_group_evals += 1;
                        stats.kernel_column_quartets += (pb.ma * pb.mb * pk.ma * pk.mb) as u64;
                        if let Some(t) = t {
                            stats.t_ker += t.elapsed().as_secs_f64();
                        }
                    }
                    // Gather the column quartet in libint's layout.
                    let (c3, c4) = (h.slot[s3].1, h.slot[s4].1);
                    let ab = c1 * pb.mb + c2;
                    let cd = c3 * pk.mb + c4;
                    let (nfab, nfcd) = (pb.hrr.len(), pk.hrr.len());
                    let mcd = pk.ma * pk.mb;
                    let d = [dims[s1], dims[s2], dims[s3], dims[s4]];
                    let mut li = 0;
                    for i0 in 0..d[0] {
                        for i1 in 0..d[1] {
                            for i2 in 0..d[2] {
                                for i3 in 0..d[3] {
                                    let fa = if d[0] == 3 { pmap(i0) } else { 0 };
                                    let fb = if d[1] == 3 { pmap(i1) } else { 0 };
                                    let fc = if d[2] == 3 { pmap(i2) } else { 0 };
                                    let fd = if d[3] == 3 { pmap(i3) } else { 0 };
                                    let fab = fa * d[1] + fb;
                                    let fcd = fc * d[3] + fd;
                                    qbuf[li] =
                                        out[(ab * nfab + fab) * (mcd * nfcd) + cd * nfcd + fcd];
                                    li += 1;
                                }
                            }
                        }
                    }
                    stats.kernel_quartets += 1;
                    let t = PROBE.then(Instant::now);
                    pc.quartet(&qbuf[..li], dims, offs, s3, s4);
                    if let Some(t) = t {
                        stats.t_dig += t.elapsed().as_secs_f64();
                    }
                } else {
                    let t = PROBE.then(Instant::now);
                    let block = eng.compute_quartet(prep, s1, s2, s3, s4);
                    let t2 = PROBE.then(Instant::now);
                    if let (Some(a), Some(b)) = (t, t2) {
                        stats.t_lib += (b - a).as_secs_f64();
                    }
                    stats.libint_calls += 1;
                    if let Some(q) = block {
                        stats.libint_digested += 1;
                        pc.quartet(q, dims, offs, s3, s4);
                        if let Some(t) = t2 {
                            stats.t_dig += t.elapsed().as_secs_f64();
                        }
                    }
                }
            }
        }
        let t = PROBE.then(Instant::now);
        pc.end();
        if let Some(t) = t {
            stats.t_dig += t.elapsed().as_secs_f64();
        }
    }
    // The real finish (J = Jh + Jhᵀ, K = Kb + Kbᵀ).
    let spec = real_scatter::PairSpec::both(n, dims, &st.d, real_scatter::SymScale::Own);
    let (j, mut k) = spec.finish(vec![pc.jh, pc.kb]);
    (j.unwrap(), k.remove(0))
}

fn bitwise_eq(a: &Array2<f64>, b: &Array2<f64>) -> bool {
    // `==` (±0 equal): the real path adds the group partial onto zeros.
    a.iter().zip(b.iter()).all(|(x, y)| x == y)
}

fn maxabs(a: &Array2<f64>) -> f64 {
    a.iter().fold(0.0f64, |m, x| m.max(x.abs()))
}

fn median(v: &[f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    let mut v = v.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

fn loadavg() -> String {
    std::fs::read_to_string("/proc/loadavg")
        .unwrap_or_default()
        .trim()
        .to_string()
}

fn hybrid(xyz: &str, name: &str, dens: &str, rot: bool, npairs: usize) {
    let st = setup(xyz, name, dens, rot);
    let boys = Boys::new();
    let (groups, slot) = build_groups(&st);
    let ng = groups.len();
    let prep = &st.prep;
    let max_dim = prep.shell_dims().iter().copied().max().unwrap();
    // Pair data for every ordered group pair with l_a + l_b <= 1, once per
    // geometry.
    let tp = Instant::now();
    let mut pidx = vec![u32::MAX; ng * ng];
    let mut pairs = Vec::new();
    for x in 0..ng {
        for y in 0..ng {
            if groups[x].l + groups[y].l <= 1 {
                pidx[x * ng + y] = pairs.len() as u32;
                pairs.push(make_pair(&groups[x], &groups[y], x == y));
            }
        }
    }
    let pair_s = tp.elapsed().as_secs_f64();
    let pure_p = prep.located_shells().iter().any(|s| s.l == 1 && s.pure);
    let h = Hyb {
        st: &st,
        slot: &slot,
        boys: &boys,
        ng,
        pidx,
        pairs,
        pure_p,
        max_dim,
    };
    println!(
        "basis {name} rot {rot}: nbf {} shells {} groups {ng} bra pairs {}; pair data for {} \
         group pairs (l_a+l_b <= 1) in {pair_s:.4} s (once per geometry)",
        prep.nbasis(),
        prep.nshells(),
        st.pairs.len(),
        h.pairs.len()
    );
    let eng =
        std::sync::Mutex::new(Engine::new_2e(Operator::coulomb(), prep, ERI_PRECISION).unwrap());
    let gorder = group_first_order(&st);

    // 1. The copy against the real pair digest, production order: bitwise.
    let t0 = Instant::now();
    let (ja, ka) = walk_real(&st, &eng, &st.pairs);
    let ta0 = t0.elapsed().as_secs_f64();
    let mut s0 = HybStats::default();
    let (jc, kc) = walk_copy::<false, false>(&h, &mut eng.lock().unwrap(), &st.pairs, &mut s0);
    println!(
        "copy (libint, production order) vs real PairSpec::sum_groups + scatter_bra_pair: \
         J bitwise {} K bitwise {} ({} survivors, {} digested; real walk {ta0:.2} s)",
        bitwise_eq(&ja, &jc),
        bitwise_eq(&ka, &kc),
        s0.survivors,
        s0.libint_digested
    );

    // 2. Probed stage breakdown (one run each; clocks add overhead).
    let mut sp_a = HybStats::default();
    let t0 = Instant::now();
    let (ja2, ka2) = walk_copy::<false, true>(&h, &mut eng.lock().unwrap(), &gorder, &mut sp_a);
    let tpa = t0.elapsed().as_secs_f64();
    let mut sp_b = HybStats::default();
    let t0 = Instant::now();
    let (jb, kb) = walk_copy::<true, true>(&h, &mut eng.lock().unwrap(), &gorder, &mut sp_b);
    let tpb = t0.elapsed().as_secs_f64();
    for (lab, s, t) in [("A' probed", &sp_a, tpa), ("B  probed", &sp_b, tpb)] {
        println!(
            "[{lab}] walk {t:.3} s = libint {:.3} + kernel {:.3} + digest {:.3} + rest \
             (loop/screen/lookup/gather) {:.3} | survivors {} libint calls {} (digested {}) \
             kernel quartets {} group evals {} column quartets computed {}",
            s.t_lib,
            s.t_ker,
            s.t_dig,
            t - s.t_lib - s.t_ker - s.t_dig,
            s.survivors,
            s.libint_calls,
            s.libint_digested,
            s.kernel_quartets,
            s.kernel_group_evals,
            s.kernel_column_quartets
        );
    }

    // 3. Accuracy: B and A' (group-first order) against A.
    let ea = 0.5
        * st.d
            .iter()
            .zip(ja.iter().zip(ka.iter()))
            .map(|(d, (j, k))| d * (j - 0.5 * k))
            .sum::<f64>();
    for (lab, j, k) in [("A' (order only)", &ja2, &ka2), ("B (kernel)", &jb, &kb)] {
        let dj = j - &ja;
        let dk = k - &ka;
        let de = 0.5
            * st.d
                .iter()
                .zip(dj.iter().zip(dk.iter()))
                .map(|(d, (j, k))| d * (j - 0.5 * k))
                .sum::<f64>();
        println!(
            "[accuracy {lab} vs A] max|J| {:.3} max|dJ| {:.2e} (rel {:.2e}) max|K| {:.3} \
             max|dK| {:.2e} (rel {:.2e}) | dE2 (1st order) {de:.2e} Ha of E2 {ea:.6}",
            maxabs(&ja),
            maxabs(&dj),
            maxabs(&dj) / maxabs(&ja),
            maxabs(&ka),
            maxabs(&dk),
            maxabs(&dk) / maxabs(&ka)
        );
    }

    // 4. Timing, single thread, interleaved A A' B B A' A per pair.
    println!(
        "timing: {npairs} pairs of (A A' B B A' A), load {}",
        loadavg()
    );
    let (mut ta, mut tc, mut tb) = (Vec::new(), Vec::new(), Vec::new());
    let mut ratio_ab = Vec::new();
    let mut ratio_cb = Vec::new();
    for rep in 0..npairs {
        let (mut pa, mut pcc, mut pb) = (0.0, 0.0, 0.0);
        for v in [0u8, 1, 2, 2, 1, 0] {
            let t0 = Instant::now();
            match v {
                0 => {
                    std::hint::black_box(walk_real(&st, &eng, &st.pairs));
                }
                1 => {
                    let mut s = HybStats::default();
                    std::hint::black_box(walk_copy::<false, false>(
                        &h,
                        &mut eng.lock().unwrap(),
                        &gorder,
                        &mut s,
                    ));
                }
                _ => {
                    let mut s = HybStats::default();
                    std::hint::black_box(walk_copy::<true, false>(
                        &h,
                        &mut eng.lock().unwrap(),
                        &gorder,
                        &mut s,
                    ));
                }
            }
            let t = t0.elapsed().as_secs_f64();
            match v {
                0 => {
                    ta.push(t);
                    pa += t;
                }
                1 => {
                    tc.push(t);
                    pcc += t;
                }
                _ => {
                    tb.push(t);
                    pb += t;
                }
            }
        }
        ratio_ab.push(pa / pb);
        ratio_cb.push(pcc / pb);
        println!(
            "  pair {rep}: A {:.3} {:.3} | A' {:.3} {:.3} | B {:.3} {:.3} | A/B {:.3} A'/B {:.3} | load {}",
            ta[2 * rep],
            ta[2 * rep + 1],
            tc[2 * rep],
            tc[2 * rep + 1],
            tb[2 * rep],
            tb[2 * rep + 1],
            pa / pb,
            pcc / pb,
            loadavg()
        );
    }
    let mm = |v: &[f64]| {
        (
            median(v),
            v.iter().cloned().fold(f64::INFINITY, f64::min),
            v.iter().cloned().fold(0.0f64, f64::max),
        )
    };
    for (lab, v) in [
        ("A  (real, prod order)", &ta),
        ("A' (copy, group-first)", &tc),
        ("B  (kernel)", &tb),
    ] {
        let (m, lo, hi) = mm(v);
        println!("[{lab}] walk s: median {m:.3} min {lo:.3} max {hi:.3}");
    }
    let (m, lo, hi) = mm(&ratio_ab);
    println!("[A/B per pair] median {m:.3} min {lo:.3} max {hi:.3}");
    let (m, lo, hi) = mm(&ratio_cb);
    println!("[A'/B per pair] median {m:.3} min {lo:.3} max {hi:.3}");
    println!(
        "saved per walk (median A - median B): {:.3} s; pair setup {pair_s:.4} s once per \
         geometry; load {}",
        median(&ta) - median(&tb),
        loadavg()
    );
}
