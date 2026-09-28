//! The production pair-FT kernel: the primitive-pair screen walked ONCE per
//! call, then every G chunk evaluated shell-row-parallel from the cached
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
//! * The parallel split is over BRA SHELLS: a task owns the output rows of
//!   its bra shell (a contiguous block of every `P[r]`), walks the ket shells
//!   in order and writes each element exactly once. No element is shared
//!   between tasks, and the split does not depend on the thread count.
//!
//! Hence every element carries the serial scalar sequence: the output is BIT
//! FOR BIT the frozen serial kernel's, at any thread count and any G
//! chunking (`tests/pbc_parallel_bitwise.rs`).
//!
//! # Memory
//!
//! The survivor lists ([`SURVIVOR_BYTES`] each) are counted before they are
//! allocated and checked against ferric's unified budget. Each rayon task
//! holds one shell pair's scratch (`R` Cartesian blocks, the F rows and the
//! pure half-transform, all `× n_G(chunk)`), which is the per-G scratch
//! [`super::pair_ft_bytes_per_g`] already charges once per G; with `T`
//! threads `T` such sets are live (a few KiB per G per thread against the
//! `16 nao²` bytes per G of the output itself).

use super::{build_shells, FtShell, WINDOW_MARGIN};
use crate::budget::{bytes_of, Ledger};
use crate::kpts::lattice_coords;
use crate::lattice::Cell;
use crate::pair_ft::residues::residue_index;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::md3c1e::{cart_components, e_table, ferric_cart2sph};
use ndarray::Array3;
use num_complex::Complex64;
use rayon::prelude::*;

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
        // budget). Both passes are parallel over bra shells and collected in
        // shell order: the lists do not depend on the thread count.
        let ns = plan.shells.len();
        let counts: Vec<usize> = (0..ns)
            .into_par_iter()
            .map(|ia| {
                let mut c = 0usize;
                for ib in 0..ns {
                    plan.walk(ia, ib, thresh, gmax_window, g2min, |_| c += 1);
                }
                c
            })
            .collect();
        let total: usize = counts.iter().sum();
        Ledger::new(crate::budget::resolve(None)).check(
            &format!("{who} cached primitive-pair survivors ({total})"),
            bytes_of(total as u64, SURVIVOR_BYTES),
        )?;
        let rows: Vec<Vec<Vec<Survivor>>> = (0..ns)
            .into_par_iter()
            .map(|ia| {
                (0..ns)
                    .map(|ib| {
                        let mut v = Vec::new();
                        plan.walk(ia, ib, thresh, gmax_window, g2min, |s| v.push(s));
                        v.shrink_to_fit();
                        v
                    })
                    .collect()
            })
            .collect();
        plan.survivors = rows.into_iter().flatten().collect();
        Ok(plan)
    }

    /// Shells must tile `0..nbf` in order (each task owns a contiguous row
    /// block).
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

    /// `P[r]` (each `(nbf, nbf, gvecs.len())`) for one G chunk; bit for bit
    /// the serial kernel's output for the same chunk (module doc).
    pub(crate) fn block(&self, gvecs: &[[f64; 3]]) -> Vec<Array3<Complex64>> {
        let (nbf, ng, nr) = (self.nbf, gvecs.len(), self.nr);
        let mut out: Vec<Array3<Complex64>> = (0..nr)
            .map(|_| Array3::<Complex64>::zeros((nbf, nbf, ng)))
            .collect();
        if ng == 0 || self.shells.is_empty() {
            return out;
        }
        let chunk = ChunkG::new(gvecs, self.lmax);

        // Per bra shell, its row block of every P[r].
        let ns = self.shells.len();
        let mut rows: Vec<Vec<&mut [Complex64]>> =
            (0..ns).map(|_| Vec::with_capacity(nr)).collect();
        for arr in out.iter_mut() {
            let mut rest = arr.as_slice_mut().expect("fresh standard-layout array");
            for (sh, row) in self.shells.iter().zip(rows.iter_mut()) {
                let (head, tail) = std::mem::take(&mut rest).split_at_mut(sh.nfun * nbf * ng);
                row.push(head);
                rest = tail;
            }
        }
        rows.into_par_iter()
            .enumerate()
            .for_each(|(ia, mut dst)| self.row_block(ia, &mut dst, &chunk));
        out
    }

    /// Every ket shell of bra shell `ia`, in order, into `dst[r]` (the rows
    /// of shell `ia` in `P[r]`). The body is the serial kernels' inner loop
    /// over the cached survivors.
    fn row_block(&self, ia: usize, dst: &mut [&mut [Complex64]], ch: &ChunkG) {
        let zero = Complex64::new(0.0, 0.0);
        let (ng, gvecs, g2, pw) = (ch.ng, &ch.gsorted, &ch.g2, &ch.pw);
        let sa = &self.shells[ia];
        let ns = self.shells.len();
        let mut common = vec![zero; ng];
        let mut ebuf: [Vec<f64>; 3] = [Vec::new(), Vec::new(), Vec::new()];
        let mut fbuf: [Vec<Complex64>; 3] = [Vec::new(), Vec::new(), Vec::new()];
        for (ib, sb) in self.shells.iter().enumerate() {
            let (la, lb) = (sa.l, sb.l);
            let (nca, ncb) = (sa.ncart, sb.ncart);
            let st = la + lb + 1;
            let nij = (la + 1) * (lb + 1);
            for d in 0..3 {
                ebuf[d].resize(nij * st, 0.0);
                fbuf[d].resize(nij * ng, zero);
            }
            let mut cart: Vec<Vec<Complex64>> =
                (0..self.nr).map(|_| vec![zero; nca * ncb * ng]).collect();
            let ca_comps = &self.comps[la];
            let cb_comps = &self.comps[lb];

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
            for (r, cart) in cart.into_iter().enumerate() {
                self.scatter(&mut *dst[r], cart, sa, sb, &ch.order);
            }
        }
    }

    /// Cartesian `[nca][ncb][ng]` -> AO rows of `sa` (local row index) at
    /// columns `sb.off..`, G scattered back through `order`: the serial
    /// kernels' transform, expression for expression.
    fn scatter(
        &self,
        dst: &mut [Complex64],
        cart: Vec<Complex64>,
        sa: &FtShell,
        sb: &FtShell,
        order: &[usize],
    ) {
        let zero = Complex64::new(0.0, 0.0);
        let (nbf, ng) = (self.nbf, order.len());
        let (la, lb) = (sa.l, sb.l);
        let (nca, ncb) = (sa.ncart, sb.ncart);
        let (nfa, nfb) = (sa.nfun, sb.nfun);
        let right: Vec<Complex64> = if sb.pure {
            let cbm = &self.c2s[lb];
            let mut tmp = vec![zero; nca * nfb * ng];
            for m in 0..nca {
                for j in 0..nfb {
                    for n in 0..ncb {
                        let c = cbm[n * nfb + j];
                        if c == 0.0 {
                            continue;
                        }
                        for g in 0..ng {
                            tmp[(m * nfb + j) * ng + g] += cart[(m * ncb + n) * ng + g] * c;
                        }
                    }
                }
            }
            tmp
        } else {
            cart
        };
        for i in 0..nfa {
            for j in 0..nfb {
                let base = (i * nbf + sb.off + j) * ng;
                for g in 0..ng {
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
            }
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
