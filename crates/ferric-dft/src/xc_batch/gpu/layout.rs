//! Zero-padded, bucketed layouts that turn a wave's per-batch GEMMs into a few
//! strided-batched calls.
//!
//! A strided-batched GEMM needs one shape and one stride per call. Batches
//! differ in `nact` (active functions) and `nb` (points), so every batch is
//! padded: `nact` up to its bucket, a multiple of [`NACT_BUCKET`], and `nb` up
//! to the grid's largest batch `NB`. Buckets of 16 wasted the least padded work
//! on the measured grids (10.5% benzene, 9.4% n-octane PBE/cc-pVDZ, against
//! 18.6% and 37.9% padding every batch of a wave to the wave's largest).
//!
//! * The resident AO blocks are stored bucket by bucket, each bucket's blocks in
//!   ascending batch order, every block `(nact_pad, 4·NB)` row-major with
//!   component `c` (χ, ∂xχ, ∂yχ, ∂zχ) in columns `c·NB .. c·NB + nb` ([`AoLayout`]).
//!   A wave's batches of one bucket are then consecutive in that bucket's
//!   storage: one stride reaches all of them.
//! * Inside a wave the staging buffers are ordered group by group (one group =
//!   the wave's batches of one bucket, ascending), then item by item (density or
//!   V matrix), then batch by batch ([`WaveLayout`]): each (group, item) is one
//!   compact strided batch.
//!
//! Every padded row and column is an exact zero in at least one operand of each
//! product, so it adds exact zeros to the real elements and produces exact
//! zeros in the padded part of the output, which is never read back.
use ferric_core::gpu::device::GpuError;

/// `nact` is padded up to a multiple of this.
pub(crate) const NACT_BUCKET: usize = 16;

/// `na` rounded up to its bucket.
pub(crate) fn pad_nact(na: usize) -> usize {
    na.div_ceil(NACT_BUCKET) * NACT_BUCKET
}

fn overflow() -> GpuError {
    GpuError::Layout("padded XC layout overflows usize".into())
}

/// Where every batch's padded AO block lives on the device.
#[derive(Debug, PartialEq)]
pub(crate) struct AoLayout {
    /// `NB`: the most points of any batch (every block's component width).
    pub nb_pad: usize,
    /// Per batch: its bucket (`None` when it has no active function).
    pub bucket: Vec<Option<usize>>,
    /// Per batch: its index among its bucket's batches (ascending batch order).
    pub rank: Vec<usize>,
    /// Per bucket, ascending: the padded `nact`.
    pub na_pad: Vec<usize>,
    /// Per bucket: element offset of its first block.
    pub base: Vec<usize>,
    /// Elements of the whole padded AO copy.
    pub total: usize,
}

impl AoLayout {
    /// The layout of a grid whose batches have `(nact, nb)` = `shapes[bi]`.
    pub(crate) fn new(shapes: &[(usize, usize)]) -> Result<Self, GpuError> {
        let nb_pad = shapes
            .iter()
            .filter(|s| s.0 > 0)
            .map(|s| s.1)
            .max()
            .unwrap_or(0);
        let mut na_pad: Vec<usize> = shapes
            .iter()
            .filter(|s| s.0 > 0)
            .map(|s| pad_nact(s.0))
            .collect();
        na_pad.sort_unstable();
        na_pad.dedup();
        let mut count = vec![0usize; na_pad.len()];
        let mut bucket = Vec::with_capacity(shapes.len());
        let mut rank = Vec::with_capacity(shapes.len());
        for &(na, _) in shapes {
            if na == 0 {
                bucket.push(None);
                rank.push(0);
                continue;
            }
            let q = na_pad
                .binary_search(&pad_nact(na))
                .map_err(|_| overflow())?;
            bucket.push(Some(q));
            rank.push(count[q]);
            count[q] += 1;
        }
        let mut base = Vec::with_capacity(na_pad.len());
        let mut total = 0usize;
        for (q, &np) in na_pad.iter().enumerate() {
            base.push(total);
            let block = np.checked_mul(4 * nb_pad).ok_or_else(overflow)?;
            let all = block.checked_mul(count[q]).ok_or_else(overflow)?;
            total = total.checked_add(all).ok_or_else(overflow)?;
        }
        Ok(Self {
            nb_pad,
            bucket,
            rank,
            na_pad,
            base,
            total,
        })
    }

    /// Elements of one block of bucket `q` (its stride in the AO copy).
    pub(crate) fn block_len(&self, q: usize) -> usize {
        self.na_pad[q] * 4 * self.nb_pad
    }

    /// Element offset of batch `bi`'s block (`None`: no active function).
    pub(crate) fn offset(&self, bi: usize) -> Option<usize> {
        let q = self.bucket[bi]?;
        Some(self.base[q] + self.rank[bi] * self.block_len(q))
    }

    /// `(nact_pad, NB)` of batch `bi`, `(0, NB)` without active functions.
    pub(crate) fn padded(&self, bi: usize) -> (usize, usize) {
        let na = self.bucket[bi].map_or(0, |q| self.na_pad[q]);
        (na, self.nb_pad)
    }
}

/// One group of a wave: its bucket's batches, ascending.
#[derive(Debug, PartialEq)]
pub(crate) struct Group {
    pub bucket: usize,
    /// Wave-local batch indices (batch − b0), ascending.
    pub batches: Vec<usize>,
    /// Per buffer, the element offset of the group's first item.
    pub base: [usize; 3],
    /// Per buffer, the per-batch length of each item (uniform in the group).
    pub lens: [Vec<usize>; 3],
}

impl Group {
    /// Offset in buffer `i` of item `d`'s compact batch (its first matrix).
    pub(crate) fn item_base(&self, i: usize, d: usize) -> usize {
        let before: usize = self.lens[i][..d].iter().sum();
        self.base[i] + self.batches.len() * before
    }
}

/// The staging layout of the wave `[b0, b0 + n)`.
#[derive(Debug, PartialEq)]
pub(crate) struct WaveLayout {
    pub groups: Vec<Group>,
    /// Per wave-local batch: `(group, index in the group)`, `None` for a batch
    /// without active functions (it owns no staging).
    pub slot: Vec<Option<(usize, usize)>>,
    /// Per buffer, elements used.
    pub total: [usize; 3],
}

impl WaveLayout {
    /// Groups in ascending bucket order. `lens(bi)[i]` lists batch `bi`'s item
    /// lengths in buffer `i`; it must be the same for every batch of a bucket.
    pub(crate) fn new(
        ao: &AoLayout,
        b0: usize,
        n: usize,
        lens: impl Fn(usize) -> [Vec<usize>; 3],
    ) -> Result<Self, GpuError> {
        let mut by_bucket: Vec<Vec<usize>> = vec![Vec::new(); ao.na_pad.len()];
        for j in 0..n {
            if let Some(q) = ao.bucket[b0 + j] {
                by_bucket[q].push(j);
            }
        }
        let mut groups = Vec::new();
        let mut slot = vec![None; n];
        let mut total = [0usize; 3];
        for (q, batches) in by_bucket.into_iter().enumerate() {
            let Some(&first) = batches.first() else {
                continue;
            };
            let l = lens(b0 + first);
            if let Some(&j) = batches.iter().find(|&&j| lens(b0 + j) != l) {
                return Err(GpuError::Layout(format!(
                    "XC batch {} has other item lengths than batch {} of its bucket",
                    b0 + j,
                    b0 + first
                )));
            }
            let mut base = [0usize; 3];
            for i in 0..3 {
                base[i] = total[i];
                let per: usize = l[i].iter().sum();
                let all = per.checked_mul(batches.len()).ok_or_else(overflow)?;
                total[i] = total[i].checked_add(all).ok_or_else(overflow)?;
            }
            for (jg, &j) in batches.iter().enumerate() {
                slot[j] = Some((groups.len(), jg));
            }
            groups.push(Group {
                bucket: q,
                batches,
                base,
                lens: l,
            });
        }
        Ok(Self {
            groups,
            slot,
            total,
        })
    }

    /// Offset of wave-local batch `j`'s item `d` in buffer `i` (0 for a batch
    /// without active functions, whose items are empty).
    pub(crate) fn off(&self, j: usize, i: usize, d: usize) -> usize {
        match self.slot[j] {
            Some((g, jg)) => {
                let grp = &self.groups[g];
                grp.item_base(i, d) + jg * grp.lens[i][d]
            }
            None => 0,
        }
    }

    /// Buffer `i` (`buf.len() == total[i]`) cut into every batch's items:
    /// `out[j][d]` is wave-local batch `j`'s item `d` (empty for a batch
    /// without active functions). The pieces are disjoint and cover `buf`.
    pub(crate) fn split<'a, T>(
        &self,
        i: usize,
        n: usize,
        buf: &'a mut [T],
    ) -> Vec<Vec<&'a mut [T]>> {
        let mut out: Vec<Vec<&'a mut [T]>> = (0..n).map(|_| Vec::new()).collect();
        let mut rest = buf;
        for g in &self.groups {
            for &len in &g.lens[i] {
                for &j in &g.batches {
                    let (head, tail) = std::mem::take(&mut rest).split_at_mut(len);
                    out[j].push(head);
                    rest = tail;
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buckets_pad_nact_to_sixteen_and_keep_batch_order() {
        assert_eq!(
            [0, 1, 16, 17, 32, 33].map(pad_nact),
            [0, 16, 16, 32, 32, 48]
        );
        // (nact, nb): two buckets (16, 48), an empty batch, a short batch.
        let shapes = [
            (40, 128),
            (10, 128),
            (0, 128),
            (33, 100),
            (16, 128),
            (48, 7),
        ];
        let ao = AoLayout::new(&shapes).unwrap();
        assert_eq!(ao.nb_pad, 128);
        assert_eq!(ao.na_pad, vec![16, 48]);
        assert_eq!(
            ao.bucket,
            vec![Some(1), Some(0), None, Some(1), Some(0), Some(1)]
        );
        assert_eq!(ao.rank, vec![0, 0, 0, 1, 1, 2]);
        let (b16, b48) = (16 * 512, 48 * 512);
        assert_eq!(ao.base, vec![0, 2 * b16]);
        assert_eq!(ao.total, 2 * b16 + 3 * b48);
        assert_eq!(
            (0..6).map(|bi| ao.offset(bi)).collect::<Vec<_>>(),
            vec![
                Some(2 * b16),
                Some(0),
                None,
                Some(2 * b16 + b48),
                Some(b16),
                Some(2 * b16 + 2 * b48)
            ]
        );
        assert_eq!(ao.padded(3), (48, 128));
        assert_eq!(ao.padded(2), (0, 128));
        // Blocks tile the copy exactly: disjoint and covering.
        let mut cover = vec![0u8; ao.total];
        for bi in 0..6 {
            if let Some(at) = ao.offset(bi) {
                let len = ao.block_len(ao.bucket[bi].unwrap());
                cover[at..at + len].iter_mut().for_each(|c| *c += 1);
            }
        }
        assert!(cover.iter().all(|&c| c == 1));
    }

    #[test]
    fn a_wave_is_grouped_by_bucket_then_item_then_batch() {
        let shapes = [
            (40, 128),
            (10, 128),
            (0, 128),
            (33, 100),
            (16, 128),
            (48, 7),
        ];
        let ao = AoLayout::new(&shapes).unwrap();
        // Two items (e.g. two densities): in = na_pad², out = na_pad·NB.
        let lens = |bi: usize| {
            let (np, nb) = ao.padded(bi);
            [vec![np * np; 2], vec![], vec![np * nb; 2]]
        };
        // The wave holds batches 1..6.
        let w = WaveLayout::new(&ao, 1, 5, lens).unwrap();
        assert_eq!(w.groups.len(), 2);
        assert_eq!((w.groups[0].bucket, &w.groups[0].batches), (0, &vec![0, 3]));
        assert_eq!((w.groups[1].bucket, &w.groups[1].batches), (1, &vec![2, 4]));
        assert_eq!(
            w.slot,
            vec![Some((0, 0)), None, Some((1, 0)), Some((0, 1)), Some((1, 1))]
        );
        let (i16, i48) = (16 * 16, 48 * 48);
        assert_eq!(w.total[0], 2 * 2 * i16 + 2 * 2 * i48);
        // Group 0: [item0 b0, item0 b3, item1 b0, item1 b3], then group 1.
        assert_eq!(w.off(0, 0, 0), 0);
        assert_eq!(w.off(3, 0, 0), i16);
        assert_eq!(w.off(0, 0, 1), 2 * i16);
        assert_eq!(w.off(3, 0, 1), 3 * i16);
        assert_eq!(w.off(2, 0, 0), 4 * i16);
        assert_eq!(w.off(4, 0, 1), 4 * i16 + 3 * i48);
        assert_eq!(w.groups[1].item_base(0, 1), 4 * i16 + 2 * i48);
        assert_eq!(w.off(1, 0, 0), 0, "an empty batch owns nothing");
        // `split` hands every batch exactly its items, in buffer order.
        let mut buf: Vec<usize> = (0..w.total[0]).collect();
        let parts = w.split(0, 5, &mut buf);
        assert!(parts[1].is_empty());
        for j in [0, 2, 3, 4] {
            assert_eq!(parts[j].len(), 2);
            for (d, p) in parts[j].iter().enumerate() {
                assert_eq!(p[0], w.off(j, 0, d));
                assert_eq!(p.len(), lens(1 + j)[0][d]);
            }
        }
        // A bucket whose batches disagree on item lengths is refused.
        let bad = |bi: usize| [vec![bi], vec![], vec![]];
        assert!(matches!(
            WaveLayout::new(&ao, 1, 5, bad),
            Err(GpuError::Layout(_))
        ));
    }
}
