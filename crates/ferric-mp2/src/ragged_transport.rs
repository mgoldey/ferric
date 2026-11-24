//! Moving ragged pair blocks between MPI ranks.
//!
//! The integral-direct LMP2 pair fits stripe their domain groups across
//! ranks ([`crate::lmp2_direct::DirectConfig::mpi`]), so after the fits each
//! rank holds the blocks of its own pair slots only. The solve needs every
//! block, because the occupied Fock couples pairs. This module carries the
//! blocks across with two sum-reductions:
//!
//! 1. a per-slot length table (each slot has exactly one owner, the other
//!    ranks contribute zero), and
//! 2. one flat `f64` buffer each rank fills at its own slots' offsets.
//!
//! A [`PairBlock`] is ragged and its shape at `eps > 0` comes out of the fit,
//! which is why the lengths travel first. Indices and the mask ride as `f64`
//! (exact below 2^53). The position maps `pos_da` / `pos_db` are not sent:
//! they are the inverse of `da` / `db` over the virtual space (`usize::MAX`
//! where a virtual is absent), so only their length travels and the receiver
//! rebuilds them. Summing with the other ranks' zeros returns every
//! value unchanged, except that a `-0.0` comes back as `+0.0`.
//!
//! The three steps ([`slot_lengths`], [`pack_slots`], [`unpack_slots`]) are
//! pure, so a test can play several ranks in one process; [`gather_slots`]
//! strings them together over a real [`ParallelContext`].

use crate::ragged::PairBlock;
use ferric_core::parallel::ParallelContext;
use ferric_core::FerricError;
use ndarray::Array2;

/// The blocks of one unique-pair slot, `None` until some rank has fitted it.
pub(crate) type Slot = Option<Vec<PairBlock>>;

/// Elements per reduction call: 2^24 `f64` (128 MiB), far below MPI's `i32`
/// count limit, and it bounds the scratch copy `all_reduce_sum` allocates.
const REDUCE_CHUNK: usize = 1 << 24;

/// Header elements per block: `i`, `j`, `|da|`, `|db|`, `|pos_da|`, `|pos_db|`.
const BLOCK_HEADER: usize = 6;

/// The position map of `dom` over `n` virtuals: `pos[dom[k]] = k`,
/// `usize::MAX` elsewhere. `None` if an index is out of range.
fn position_map(dom: &[usize], n: usize) -> Option<Vec<usize>> {
    let mut pos = vec![usize::MAX; n];
    for (k, &a) in dom.iter().enumerate() {
        *pos.get_mut(a)? = k;
    }
    Some(pos)
}

fn block_len(pb: &PairBlock) -> usize {
    let (na, nb) = (pb.da.len(), pb.db.len());
    // da, db, then pat, j_blk, denom (na·nb each) and the two Fock squares.
    BLOCK_HEADER + na + nb + 3 * na * nb + na * na + nb * nb
}

/// Encoded length of a slot: one count element plus its blocks.
fn slot_len(blocks: &[PairBlock]) -> usize {
    1 + blocks.iter().map(block_len).sum::<usize>()
}

fn encode_block(pb: &PairBlock, out: &mut Vec<f64>) -> Result<(), FerricError> {
    let (na, nb) = (pb.da.len(), pb.db.len());
    // The receiver rebuilds the position maps, so they must be what
    // `position_map` gives.
    let shapes_ok = position_map(&pb.da, pb.pos_da.len()).as_ref() == Some(&pb.pos_da)
        && position_map(&pb.db, pb.pos_db.len()).as_ref() == Some(&pb.pos_db)
        && pb.pat.len() == na * nb
        && pb.fvv_aa.dim() == (na, na)
        && pb.fvv_bb.dim() == (nb, nb)
        && pb.j_blk.dim() == (na, nb)
        && pb.denom.dim() == (na, nb);
    if !shapes_ok {
        return Err(FerricError::General(format!(
            "ragged_transport: pair ({}, {}) has block shapes or position maps this \
             transport cannot rebuild",
            pb.i, pb.j
        )));
    }
    out.extend([pb.i, pb.j, na, nb, pb.pos_da.len(), pb.pos_db.len()].map(|v| v as f64));
    for idx in [&pb.da, &pb.db] {
        out.extend(idx.iter().map(|&v| v as f64));
    }
    out.extend(pb.pat.iter().map(|&b| f64::from(u8::from(b))));
    // `iter()` walks in logical (row-major) order whatever the layout.
    for arr in [&pb.fvv_aa, &pb.fvv_bb, &pb.j_blk, &pb.denom] {
        out.extend(arr.iter().copied());
    }
    Ok(())
}

/// A cursor over an encoded buffer that fails instead of reading past it.
struct Reader<'a> {
    buf: &'a [f64],
    at: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [f64], FerricError> {
        let end = self
            .at
            .checked_add(n)
            .filter(|&e| e <= self.buf.len())
            .ok_or_else(|| {
                FerricError::General("ragged_transport: encoded slot is truncated".into())
            })?;
        let s = &self.buf[self.at..end];
        self.at = end;
        Ok(s)
    }

    fn index(&mut self) -> Result<usize, FerricError> {
        to_index(self.take(1)?[0])
    }

    fn indices(&mut self, n: usize) -> Result<Vec<usize>, FerricError> {
        self.take(n)?.iter().map(|&v| to_index(v)).collect()
    }

    fn matrix(&mut self, rows: usize, cols: usize) -> Result<Array2<f64>, FerricError> {
        Array2::from_shape_vec((rows, cols), self.take(rows * cols)?.to_vec())
            .map_err(|e| FerricError::General(format!("ragged_transport: {e}")))
    }
}

/// An index that travelled as `f64`. Anything but a non-negative integer
/// means two ranks wrote the same element or the buffer is corrupt.
fn to_index(v: f64) -> Result<usize, FerricError> {
    if v >= 0.0 && v.fract() == 0.0 && v < 9.0e15 {
        Ok(v as usize)
    } else {
        Err(FerricError::General(format!(
            "ragged_transport: {v} is not an index (slot written by more than one rank?)"
        )))
    }
}

fn decode_block(r: &mut Reader<'_>) -> Result<PairBlock, FerricError> {
    let (i, j, na, nb) = (r.index()?, r.index()?, r.index()?, r.index()?);
    let (npa, npb) = (r.index()?, r.index()?);
    let (da, db) = (r.indices(na)?, r.indices(nb)?);
    let out_of_range = || {
        FerricError::General(format!(
            "ragged_transport: pair ({i}, {j}) has a virtual index outside its position map"
        ))
    };
    let pos_da = position_map(&da, npa).ok_or_else(out_of_range)?;
    let pos_db = position_map(&db, npb).ok_or_else(out_of_range)?;
    let pat = r.take(na * nb)?.iter().map(|&v| v != 0.0).collect();
    let fvv_aa = r.matrix(na, na)?;
    let fvv_bb = r.matrix(nb, nb)?;
    let j_blk = r.matrix(na, nb)?;
    let denom = r.matrix(na, nb)?;
    Ok(PairBlock {
        i,
        j,
        da,
        db,
        pat,
        fvv_aa,
        fvv_bb,
        j_blk,
        denom,
        pos_da,
        pos_db,
    })
}

/// Encoded length of every slot this rank holds; 0.0 for the others.
pub(crate) fn slot_lengths(slots: &[Slot]) -> Vec<f64> {
    slots
        .iter()
        .map(|s| s.as_deref().map_or(0.0, |b| slot_len(b) as f64))
        .collect()
}

/// Start offset of every slot in the flat buffer, and the buffer length.
fn offsets(global_lens: &[f64]) -> Result<(Vec<usize>, usize), FerricError> {
    let mut at = 0usize;
    let mut offs = Vec::with_capacity(global_lens.len());
    for &l in global_lens {
        offs.push(at);
        at = at.checked_add(to_index(l)?).ok_or_else(|| {
            FerricError::General("ragged_transport: gather buffer length overflows".into())
        })?;
    }
    Ok((offs, at))
}

/// The flat buffer with this rank's slots written at their offsets and
/// zeros everywhere else. `global_lens` is the reduced length table.
pub(crate) fn pack_slots(slots: &[Slot], global_lens: &[f64]) -> Result<Vec<f64>, FerricError> {
    let (offs, total) = offsets(global_lens)?;
    let mut buf = vec![0.0_f64; total];
    let mut enc: Vec<f64> = Vec::new();
    for (px, slot) in slots.iter().enumerate() {
        let Some(blocks) = slot else { continue };
        enc.clear();
        enc.push(blocks.len() as f64);
        for pb in blocks {
            encode_block(pb, &mut enc)?;
        }
        if enc.len() as f64 != global_lens[px] {
            return Err(FerricError::General(format!(
                "ragged_transport: slot {px} is {} elements here but {} after the reduction \
                 (fitted on more than one rank?)",
                enc.len(),
                global_lens[px]
            )));
        }
        buf[offs[px]..offs[px] + enc.len()].copy_from_slice(&enc);
    }
    Ok(buf)
}

/// Fill every slot this rank does not hold from the reduced buffer. A slot
/// no rank fitted (length 0) stays `None` for the caller to report.
pub(crate) fn unpack_slots(
    slots: &mut [Slot],
    global_lens: &[f64],
    global_buf: &[f64],
) -> Result<(), FerricError> {
    let (offs, total) = offsets(global_lens)?;
    if global_buf.len() != total {
        return Err(FerricError::General(format!(
            "ragged_transport: gather buffer is {} elements, the length table says {total}",
            global_buf.len()
        )));
    }
    for (px, slot) in slots.iter_mut().enumerate() {
        let len = to_index(global_lens[px])?;
        if slot.is_some() || len == 0 {
            continue;
        }
        let mut r = Reader {
            buf: &global_buf[offs[px]..offs[px] + len],
            at: 0,
        };
        let n = r.index()?;
        let blocks = (0..n)
            .map(|_| decode_block(&mut r))
            .collect::<Result<Vec<_>, _>>()?;
        if r.at != len {
            return Err(FerricError::General(format!(
                "ragged_transport: slot {px} decoded {} of {len} elements",
                r.at
            )));
        }
        *slot = Some(blocks);
    }
    Ok(())
}

/// End of the window of consecutive slots starting at `start`: the longest
/// run whose encoded length stays within [`REDUCE_CHUNK`] (at least one
/// slot, so a slot longer than a chunk is a window of its own).
fn window_end(global_lens: &[f64], start: usize) -> Result<usize, FerricError> {
    let (mut end, mut len) = (start, 0usize);
    while end < global_lens.len() {
        let add = to_index(global_lens[end])?;
        if end > start && len.saturating_add(add) > REDUCE_CHUNK {
            break;
        }
        len = len.saturating_add(add);
        end += 1;
    }
    Ok(end)
}

/// Bytes [`gather_slots`] holds at its peak beside the blocks: the largest
/// window's flat buffer plus one reduction chunk.
pub(crate) fn gather_bytes(global_lens: &[f64]) -> Result<usize, FerricError> {
    let (mut start, mut peak) = (0usize, 0usize);
    while start < global_lens.len() {
        let end = window_end(global_lens, start)?;
        let (_, total) = offsets(&global_lens[start..end])?;
        peak = peak.max(total + total.min(REDUCE_CHUNK));
        start = end;
    }
    Ok(8 * peak)
}

/// Give every rank every slot: reduce the length table, then, one window of
/// slots at a time, the packed buffer, decoding what this rank did not fit.
/// The transport buffer is one window, not the whole system. `avail_bytes`,
/// when set, is what the memory budget has left for it; exceeding that is an
/// error before any buffer is allocated.
///
/// Call from the rank's serial point (see [`ParallelContext::all_reduce_sum`]).
/// On a context with no MPI world the reductions do nothing and the slots
/// come back as they went in.
pub(crate) fn gather_slots(
    ctx: &ParallelContext,
    slots: &mut [Slot],
    avail_bytes: Option<usize>,
) -> Result<(), FerricError> {
    let mut lens = slot_lengths(slots);
    ctx.all_reduce_sum(&mut lens);
    if let Some(avail) = avail_bytes {
        ferric_core::memory::check_alloc(
            "LMP2 MPI pair-block gather buffer",
            gather_bytes(&lens)?,
            avail,
        )?;
    }
    let mut start = 0usize;
    while start < slots.len() {
        let end = window_end(&lens, start)?;
        let (win, win_lens) = (&mut slots[start..end], &lens[start..end]);
        let mut buf = pack_slots(win, win_lens)?;
        for chunk in buf.chunks_mut(REDUCE_CHUNK) {
            ctx.all_reduce_sum(chunk);
        }
        unpack_slots(win, win_lens, &buf)?;
        start = end;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A block with distinct values in every field, non-square, with a
    /// column-major array among them.
    fn block(i: usize, j: usize, na: usize, nb: usize, seed: f64) -> PairBlock {
        let fill = |r: usize, c: usize, k: f64| {
            Array2::from_shape_fn((r, c), |(a, b)| seed + k + (a * c + b) as f64 * 0.125)
        };
        // Same logical values as `fill`, stored column-major.
        let j_blk = fill(nb, na, 3.0).reversed_axes();
        assert!(na < 2 || nb < 2 || !j_blk.is_standard_layout());
        let da: Vec<usize> = (0..na).map(|a| 3 * a + i).collect();
        let db: Vec<usize> = (0..nb).map(|b| 5 * b + j).collect();
        const NV: usize = 40;
        PairBlock {
            i,
            j,
            pat: (0..na * nb).map(|k| k % 3 != 1).collect(),
            fvv_aa: fill(na, na, 1.0),
            fvv_bb: fill(nb, nb, 2.0),
            j_blk,
            denom: fill(na, nb, 4.0),
            pos_da: position_map(&da, NV).unwrap(),
            pos_db: position_map(&db, NV).unwrap(),
            da,
            db,
        }
    }

    fn encoded(slot: &Slot) -> Option<Vec<u64>> {
        slot.as_ref().map(|blocks| {
            let mut v = vec![blocks.len() as f64];
            for pb in blocks {
                encode_block(pb, &mut v).unwrap();
            }
            v.iter().map(|x| x.to_bits()).collect()
        })
    }

    fn serial_slots() -> Vec<Slot> {
        vec![
            Some(vec![block(0, 0, 2, 2, 0.5)]),
            Some(vec![block(0, 1, 3, 2, -1.25), block(1, 0, 2, 3, 9.0)]),
            Some(vec![]),
            Some(vec![block(2, 5, 1, 4, 100.0)]),
            Some(vec![block(3, 3, 4, 1, -7.5)]),
        ]
    }

    /// Elementwise sum, the reduction `gather_slots` performs across ranks.
    fn summed(parts: &[Vec<f64>]) -> Vec<f64> {
        let mut out = vec![0.0; parts[0].len()];
        for p in parts {
            for (o, v) in out.iter_mut().zip(p) {
                *o += v;
            }
        }
        out
    }

    /// Play `size` ranks in one process: each holds the slots `px % size ==
    /// rank`, and after the two reductions every rank holds all of them,
    /// bit for bit.
    #[test]
    fn simulated_ranks_recover_the_serial_slots() {
        let serial = serial_slots();
        for size in [1usize, 2, 3, 7] {
            let mut ranks: Vec<Vec<Slot>> = (0..size)
                .map(|r| {
                    serial
                        .iter()
                        .enumerate()
                        .map(|(px, s)| if px % size == r { s.clone() } else { None })
                        .collect()
                })
                .collect();
            let lens = summed(&ranks.iter().map(|s| slot_lengths(s)).collect::<Vec<_>>());
            let bufs: Vec<Vec<f64>> = ranks
                .iter()
                .map(|s| pack_slots(s, &lens).unwrap())
                .collect();
            let buf = summed(&bufs);
            for slots in ranks.iter_mut() {
                unpack_slots(slots, &lens, &buf).unwrap();
                for (got, want) in slots.iter().zip(&serial) {
                    assert_eq!(encoded(got), encoded(want), "size {size}");
                }
            }
        }
    }

    #[test]
    fn a_decoded_block_keeps_every_field() {
        let pb = block(4, 9, 3, 2, 0.75);
        let slots = vec![Some(vec![pb.clone()])];
        let lens = slot_lengths(&slots);
        let buf = pack_slots(&slots, &lens).unwrap();
        let mut empty: Vec<Slot> = vec![None];
        unpack_slots(&mut empty, &lens, &buf).unwrap();
        let got = &empty[0].as_ref().unwrap()[0];
        assert_eq!((got.i, got.j), (4, 9));
        assert_eq!(got.da, pb.da);
        assert_eq!(got.db, pb.db);
        assert_eq!(got.pos_da, pb.pos_da);
        assert_eq!(got.pos_db, pb.pos_db);
        assert_eq!(got.pat, pb.pat);
        assert_eq!(got.fvv_aa, pb.fvv_aa);
        assert_eq!(got.fvv_bb, pb.fvv_bb);
        assert_eq!(got.j_blk, pb.j_blk);
        assert_eq!(got.denom, pb.denom);
        assert!(got.j_blk.is_standard_layout());
    }

    /// A slot no rank fitted stays empty for the caller to report.
    #[test]
    fn an_unfitted_slot_stays_none() {
        let mut slots: Vec<Slot> = vec![Some(vec![block(0, 0, 1, 1, 1.0)]), None];
        let lens = slot_lengths(&slots);
        let buf = pack_slots(&slots, &lens).unwrap();
        unpack_slots(&mut slots, &lens, &buf).unwrap();
        assert!(slots[0].is_some() && slots[1].is_none());
    }

    /// Two ranks fitting the same slot double its length and its values;
    /// both ranks must refuse rather than decode garbage.
    #[test]
    fn a_slot_fitted_twice_is_refused() {
        let slots: Vec<Slot> = vec![Some(vec![block(0, 0, 2, 2, 1.0)])];
        let lens = summed(&[slot_lengths(&slots), slot_lengths(&slots)]);
        let err = pack_slots(&slots, &lens).unwrap_err();
        assert!(format!("{err}").contains("more than one rank"), "{err}");
    }

    /// The receiver rebuilds the position maps, so a block whose map is not
    /// the inverse of its domain must be refused at the sender.
    #[test]
    fn a_position_map_the_receiver_cannot_rebuild_is_refused() {
        let mut pb = block(0, 0, 2, 2, 1.0);
        pb.pos_da.swap(pb.da[0], pb.da[1]);
        let slots: Vec<Slot> = vec![Some(vec![pb])];
        let err = pack_slots(&slots, &slot_lengths(&slots)).unwrap_err();
        assert!(format!("{err}").contains("position maps"), "{err}");
    }

    #[test]
    fn a_short_buffer_is_refused() {
        let slots: Vec<Slot> = vec![Some(vec![block(0, 0, 2, 2, 1.0)])];
        let lens = slot_lengths(&slots);
        let buf = pack_slots(&slots, &lens).unwrap();
        let mut empty: Vec<Slot> = vec![None];
        let err = unpack_slots(&mut empty, &lens, &buf[..buf.len() - 1]).unwrap_err();
        assert!(format!("{err}").contains("length table"), "{err}");
    }

    /// With no MPI world the reductions are no-ops: a rank's own slots come
    /// back untouched and the others stay empty.
    #[test]
    fn gather_without_a_world_changes_nothing() {
        let ctx = ParallelContext::for_rank(0, 2);
        let mut slots: Vec<Slot> = vec![Some(vec![block(0, 0, 2, 2, 1.0)]), None];
        let before = encoded(&slots[0]);
        gather_slots(&ctx, &mut slots, None).unwrap();
        assert_eq!(encoded(&slots[0]), before);
        assert!(slots[1].is_none());
    }

    #[test]
    fn gather_respects_the_budget() {
        let ctx = ParallelContext::for_rank(0, 1);
        let mut slots: Vec<Slot> = vec![Some(vec![block(0, 0, 8, 8, 1.0)])];
        let err = gather_slots(&ctx, &mut slots, Some(16)).unwrap_err();
        assert!(format!("{err}").contains("gather buffer"), "{err}");
    }
}
