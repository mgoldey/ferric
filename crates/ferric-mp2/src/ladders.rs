//! hh / pp ladder application on the ragged pair-block pattern — the ladder
//! part of the amplitude-threshold LinLCCD matvec, shared so other ladder
//! members of the local family (a local OP-MP3) reuse it rather than copy it.
//!
//! For each OUTPUT pair `(i, j)`:
//!
//! ```text
//! hh: R_ij += Σ_(k,l) (ik|jl) · gather(T_kl)          (coefficients: OoGram)
//! pp: R_ij[a,b] += Σ_(c,d) (ac|bd) T_ij[c,d]          (source: PpSource)
//! ```
//!
//! Moved verbatim from `ferric_cc::linlccd_amplitude` (ferric-cc depends on
//! ferric-mp2, not the reverse); the per-pair arithmetic order is unchanged,
//! so LinLCCD energies are bit-identical.

use ndarray::Array2;
use rayon::prelude::*;

use crate::lmp2_direct::{OoGram, PpFitted};
use crate::ragged::{apply_pattern, gather_into, Ragged};

/// pp-ladder source.
pub enum PpSource<'a> {
    /// No pp ladder (DriversOnly / Hh tiers).
    None,
    /// Licensed consistent Gram: per-pair gathers from ONE whitened global
    /// Bvv `(naux, nv²)` — provably structured, the original path.
    GlobalWhitened {
        /// Whitened `B_vv`, `(naux, nv²)`, column `a·nv + c`.
        bvv_t: &'a Array2<f64>,
        /// Number of localized virtuals.
        nv: usize,
    },
    /// Integral-direct per-pair domain-FITTED factors — the PSD contract
    /// here is the MEASURED one (see [`PpFitted`]'s doc for the license).
    Fitted(&'a [PpFitted]),
}

/// pp block application shared by both pp sources: given the pair's block
/// `m` (rows (a,c), cols (b,d)), accumulate `out[a,b] += Σ_cd m T[c,d]`.
fn apply_pp_block(rp: &mut Array2<f64>, m: &Array2<f64>, nda: usize, ndb: usize, tp: &Array2<f64>) {
    for ra in 0..nda {
        for cb in 0..ndb {
            let mut acc = 0.0;
            for rc in 0..nda {
                for cd in 0..ndb {
                    acc += m[(ra * nda + rc, cb * ndb + cd)] * tp[(rc, cd)];
                }
            }
            rp[(ra, cb)] += acc;
        }
    }
}

/// Add the hh ladder (`oo`, when given) and the pp ladder (`pp`) applied to
/// `t` into `r`, then pattern-project every block of `r`; `flops` gets the
/// ladder multiply-adds.
///
/// Each output block is computed wholly inside ONE rayon task keyed by the
/// OUTPUT pair (disjoint writes only), with the serial loop's per-block
/// arithmetic order, then added to `r[p]` in pair order — deterministic and
/// bit-identical under any schedule.
pub fn apply_ladders_into(
    rg: &Ragged,
    t: &[Array2<f64>],
    oo: Option<&OoGram>,
    pp: &PpSource<'_>,
    r: &mut [Array2<f64>],
    flops: &mut u64,
) {
    let ladder: Vec<(Array2<f64>, u64)> = rg
        .pairs
        .par_iter()
        .enumerate()
        .map(|(p, pb)| {
            let mut rp = Array2::<f64>::zeros((pb.da.len(), pb.db.len()));
            let mut fl = 0u64;
            // hh: out_ij += Σ_(k,l) (ik|jl) · gather(T_kl). For a FIXED
            // output (i,j), sum over source pairs (k,l) = (p_src.i,
            // p_src.j) with a nonzero OOOO coefficient. Iterate the
            // source pairs (same set the serial loop scanned) but write
            // only into THIS output's rp.
            if let Some(oo) = oo {
                for (p_src, pb_src) in rg.pairs.iter().enumerate() {
                    let coeff = oo.coeff(pb.i, pb_src.i, pb.j, pb_src.j);
                    if coeff == 0.0 {
                        continue;
                    }
                    gather_into(&mut rp, pb_src, pb, coeff, &t[p_src], &mut fl);
                }
            }
            match pp {
                PpSource::None => {}
                PpSource::GlobalWhitened { bvv_t: bvv, nv } => {
                    let nv = *nv;
                    let naux = bvv.nrows();
                    let (da, db) = (&pb.da, &pb.db);
                    let (nda, ndb) = (da.len(), db.len());
                    let mut ba = Array2::<f64>::zeros((naux, nda * nda));
                    for (ra, &a) in da.iter().enumerate() {
                        for (rc, &c) in da.iter().enumerate() {
                            ba.column_mut(ra * nda + rc).assign(&bvv.column(a * nv + c));
                        }
                    }
                    let mut bb = Array2::<f64>::zeros((naux, ndb * ndb));
                    for (cb, &b) in db.iter().enumerate() {
                        for (cd, &d) in db.iter().enumerate() {
                            bb.column_mut(cb * ndb + cd).assign(&bvv.column(b * nv + d));
                        }
                    }
                    let m = ba.t().dot(&bb); // rows (a,c), cols (b,d)
                    fl += (nda * nda * ndb * ndb * naux) as u64;
                    apply_pp_block(&mut rp, &m, nda, ndb, &t[p]);
                }
                PpSource::Fitted(factors) => {
                    let (nda, ndb) = (pb.da.len(), pb.db.len());
                    let f = &factors[p];
                    let m = f.a_rows.dot(&f.gb); // rows (a,c), cols (b,d)
                    fl += (nda * nda * ndb * ndb * f.gb.nrows()) as u64;
                    apply_pp_block(&mut rp, &m, nda, ndb, &t[p]);
                }
            }
            (rp, fl)
        })
        .collect();
    for (p, (contrib, fl)) in ladder.into_iter().enumerate() {
        r[p] += &contrib;
        *flops += fl;
        apply_pattern(&mut r[p], &rg.pairs[p].pat, rg.pairs[p].db.len());
    }
}
