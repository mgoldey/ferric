//! Schwarz-screened AO shell pairs for a compressed RI-J tensor.
//!
//! RI-J stores `B[P, μν] = (P|μν)` for every AO pair, but on a molecule of any
//! size most pairs are numerically zero: two functions on distant atoms barely
//! overlap. Cauchy-Schwarz bounds every element of a pair's row,
//!
//! ```text
//!   |(P|μν)| <= Q_{μν} · Q_P,    Q_{μν} = sqrt((μν|μν)),  Q_P = sqrt((P|P)),
//! ```
//!
//! so a shell pair with `Q[s1, s2] · max_P Q_P <= thresh` has EVERY element of
//! its rows at or below `thresh`, and storing it buys nothing a `thresh`-level
//! calculation can see. Measured on danuglipron (71 atoms, def2-SVP,
//! def2-universal-jkfit, nbf 724, naux 3635): at `thresh = 1e-12`, 56.0% of
//! the 56,616 shell pairs and 52.0% of the 262,450 packed function pairs
//! survive, so the stored tensor drops from 7.63 GB to 3.97 GB.
//!
//! [`PairScreen`] is the kept-pair list and the stored-column layout;
//! [`fill_eri3_pairs`] computes ONLY the kept shell pairs straight into that
//! layout, so the dropped pairs cost neither memory nor integral time.
//!
//! # Error bound on J
//!
//! A dropped pair's Coulomb element is set to zero. Its true value is
//! `J_μν = Σ_P (P|μν) c_P`, so the error is
//!
//! ```text
//!   |J_μν| <= Σ_P |(P|μν)| |c_P| <= Q_{μν} Σ_P Q_P |c_P|
//!          <= thresh · Σ_P |c_P| · Q_P / max_P Q_P,
//! ```
//!
//! which is at most `thresh · Σ_P |c_P|`. Kept pairs are computed exactly as
//! the unscreened build computes them (same engine call), so they carry no
//! screening error; their contribution to `d_P` loses only the dropped pairs'
//! `Σ (P|μν) D_μν`, bounded the same way with `|D_μν|` in place of `|c_P|`.

use crate::basis_bridge::PreparedBasis;
use crate::engine::Engine;
use crate::operator::Operator;
use ferric_core::FerricError;
use ndarray::Array2;

/// The AO shell pairs whose RI-J rows survive a Schwarz screen, and the column
/// layout the compressed tensor stores them in.
///
/// Columns are grouped by kept shell pair, in `(s1, s2 <= s1)` row-major
/// order; within a pair they run `(i, j)` row-major with `j <= i` on a diagonal
/// pair. Every column is a function pair `(μ, ν)` with `ν <= μ`, the same
/// triangle the packed tier stores, so a consumer only needs [`Self::cols`].
#[derive(Debug, Clone)]
pub struct PairScreen {
    shell_pairs: Vec<(u32, u32)>,
    /// First stored column of each kept shell pair; one trailing entry = ncols.
    col_start: Vec<usize>,
    cols: Vec<(u32, u32)>,
    total_pairs: usize,
    total_shell_pairs: usize,
    thresh: f64,
}

impl PairScreen {
    /// Screen `obs`'s shell pairs with the operator's own Schwarz table:
    /// keep `(s1, s2)` iff `Q[s1, s2] · q_aux_max > thresh`, where `q_aux_max`
    /// is `max_P sqrt((P|P))` of the SAME operator's aux metric.
    pub fn schwarz(
        op: Operator,
        obs: &PreparedBasis,
        q_aux_max: f64,
        thresh: f64,
    ) -> Result<Self, FerricError> {
        let q = crate::schwarz::schwarz(op, obs)?;
        Ok(Self::from_shell_bounds(
            &q,
            obs.shell_dims(),
            obs.shell_offsets(),
            q_aux_max,
            thresh,
        ))
    }

    /// The pure part of [`Self::schwarz`]: build the layout from a shell-pair
    /// bound table `q` (`nsh × nsh`), the shell sizes and offsets.
    pub fn from_shell_bounds(
        q: &Array2<f64>,
        dims: &[usize],
        offs: &[usize],
        q_aux_max: f64,
        thresh: f64,
    ) -> Self {
        let nsh = dims.len();
        let nao = offs.get(nsh).copied().unwrap_or(0);
        let mut screen = Self {
            shell_pairs: Vec::new(),
            col_start: vec![0],
            cols: Vec::new(),
            total_pairs: nao * (nao + 1) / 2,
            total_shell_pairs: nsh * (nsh + 1) / 2,
            thresh,
        };
        for s1 in 0..nsh {
            for s2 in 0..=s1 {
                if q[(s1, s2)] * q_aux_max > thresh {
                    screen.push_shell_pair(s1, s2, dims, offs);
                }
            }
        }
        screen
    }

    fn push_shell_pair(&mut self, s1: usize, s2: usize, dims: &[usize], offs: &[usize]) {
        self.shell_pairs.push((s1 as u32, s2 as u32));
        for i in 0..dims[s1] {
            let jmax = if s1 == s2 { i + 1 } else { dims[s2] };
            for j in 0..jmax {
                self.cols
                    .push(((offs[s1] + i) as u32, (offs[s2] + j) as u32));
            }
        }
        self.col_start.push(self.cols.len());
    }

    /// Stored columns (kept function pairs).
    pub fn ncols(&self) -> usize {
        self.cols.len()
    }
    /// `(μ, ν <= μ)` of every stored column, in storage order.
    pub fn cols(&self) -> &[(u32, u32)] {
        &self.cols
    }
    /// Function pairs of the full packed triangle, `nao(nao+1)/2`.
    pub fn total_pairs(&self) -> usize {
        self.total_pairs
    }
    /// Kept and total shell pairs.
    pub fn shell_pair_counts(&self) -> (usize, usize) {
        (self.shell_pairs.len(), self.total_shell_pairs)
    }
    /// The screening threshold this layout was built with.
    pub fn thresh(&self) -> f64 {
        self.thresh
    }
}

/// Fill `out` (`(p1 - p0, screen.ncols())`) with `(P|μν)` for aux rows
/// `[p0, p1)` and the kept pairs only.
///
/// Every stored value comes from the same `Engine::compute_eri3` call, with the
/// same engine precision, as `threeindex::eri3_block_screened(.., None)`, so it
/// is BIT-IDENTICAL to the corresponding element of the unscreened tensor.
/// Parallel over aux shells; each writes a disjoint band of rows.
pub fn fill_eri3_pairs(
    op: Operator,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    (p0, p1): (usize, usize),
    screen: &PairScreen,
    out: &mut Array2<f64>,
) -> Result<(), FerricError> {
    use rayon::prelude::*;

    check_band(dfbs, (p0, p1))?;
    check_layout(obs, screen)?;
    let ncols = screen.ncols();
    if out.dim() != (p1 - p0, ncols) {
        return Err(FerricError::General(format!(
            "pair-screened block: out is {:?}, expected ({}, {ncols})",
            out.dim(),
            p1 - p0
        )));
    }
    Engine::new_3center(op, obs, dfbs, 1e-14)?;
    let rows = out
        .as_slice_mut()
        .ok_or_else(|| FerricError::General("pair-screened block not contiguous".into()))?;
    let out_ptr = rows.as_mut_ptr() as usize;
    let (dims_df, offs_df) = (dfbs.shell_dims(), dfbs.shell_offsets());
    (0..dfbs.nshells()).into_par_iter().for_each_init(
        || Engine::new_3center(op, obs, dfbs, 1e-14).expect("3-center engine (pre-validated)"),
        |eng, sp| {
            let (pbase, np) = (offs_df[sp], dims_df[sp]);
            if pbase + np <= p0 || pbase >= p1 {
                return;
            }
            // SAFETY: aux shell `sp` owns rows [pbase, pbase+np) ∩ [p0, p1),
            // disjoint from every other shell's, and `out` outlives the loop.
            let base = out_ptr as *mut f64;
            for (k, &(s1, s2)) in screen.shell_pairs.iter().enumerate() {
                let (s1, s2) = (s1 as usize, s2 as usize);
                let Some(block) = eng.compute_eri3(obs, dfbs, sp, s1, s2) else {
                    continue; // engine-screened: the row stays zero, as unscreened
                };
                let (n1, n2) = (obs.shell_dims()[s1], obs.shell_dims()[s2]);
                for p in (pbase.max(p0) - pbase)..(np.min(p1 - pbase)) {
                    let row = unsafe { base.add((pbase + p - p0) * ncols) };
                    scatter_pair_row(block, (p, n1, n2), s1 == s2, screen.col_start[k], row);
                }
            }
        },
    );
    Ok(())
}

/// Refuse an aux band that is reversed or runs past `dfbs`. Called before any
/// `p1 - p0`, so a bad band is an error, not an underflow panic or silently
/// zero trailing rows.
pub(crate) fn check_band(
    dfbs: &PreparedBasis,
    (p0, p1): (usize, usize),
) -> Result<(), FerricError> {
    if p0 > p1 || p1 > dfbs.nbasis() {
        return Err(FerricError::General(format!(
            "pair-screened block: invalid aux band [{p0}, {p1}) for naux = {}",
            dfbs.nbasis()
        )));
    }
    Ok(())
}

/// Refuse a screen built for a different orbital basis: every kept shell pair
/// must index a shell of `obs` and own exactly as many stored columns as that
/// pair has function pairs. This is what keeps every `scatter_pair_row` write
/// inside its row, since `PairScreen` is public and could be built from other
/// shell dimensions.
fn check_layout(obs: &PreparedBasis, screen: &PairScreen) -> Result<(), FerricError> {
    let dims = obs.shell_dims();
    let width = |s1: usize, s2: usize| -> Option<usize> {
        let (a, b) = (*dims.get(s1)?, *dims.get(s2)?);
        Some(if s1 == s2 { a * (a + 1) / 2 } else { a * b })
    };
    for (k, &(s1, s2)) in screen.shell_pairs.iter().enumerate() {
        let stored = screen.col_start[k + 1] - screen.col_start[k];
        if width(s1 as usize, s2 as usize) != Some(stored) {
            return Err(FerricError::General(format!(
                "pair-screened block: the screen does not match the orbital basis \
                 (shell pair ({s1}, {s2}) stores {stored} columns)"
            )));
        }
    }
    Ok(())
}

/// Copy one aux function's `(n1 × n2)` shell-pair block into its stored
/// columns starting at `col0` (lower triangle only on a diagonal pair).
fn scatter_pair_row(
    block: &[f64],
    (p, n1, n2): (usize, usize, usize),
    diagonal: bool,
    col0: usize,
    row: *mut f64,
) {
    let mut c = col0;
    for i in 0..n1 {
        let jmax = if diagonal { i + 1 } else { n2 };
        for j in 0..jmax {
            // SAFETY: `c` < the pair's col_start[k+1] <= ncols, and `row`
            // points at this aux row's ncols contiguous elements.
            unsafe { *row.add(c) = block[(p * n1 + i) * n2 + j] };
            c += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The layout is the packed triangle restricted to kept shell pairs: with
    /// nothing dropped it enumerates every `(μ, ν <= μ)` exactly once.
    #[test]
    fn keeping_everything_covers_the_packed_triangle_once() {
        let dims = [1usize, 3, 5, 1];
        let offs = [0usize, 1, 4, 9, 10];
        let q = Array2::from_elem((4, 4), 1.0);
        let s = PairScreen::from_shell_bounds(&q, &dims, &offs, 1.0, 0.0);
        assert_eq!(s.ncols(), s.total_pairs());
        let mut seen = std::collections::BTreeSet::new();
        for &(mu, nu) in s.cols() {
            assert!(nu <= mu);
            assert!(seen.insert((mu, nu)), "({mu},{nu}) stored twice");
        }
        assert_eq!(seen.len(), 10 * 11 / 2);
        assert_eq!(s.shell_pair_counts(), (10, 10));
    }

    /// A pair is kept iff its bound times `q_aux_max` EXCEEDS the threshold
    /// (equal is dropped: every element is then <= thresh).
    #[test]
    fn the_threshold_is_strict() {
        let dims = [2usize, 2];
        let offs = [0usize, 2, 4];
        // Exactly representable: 0.5 * 2.0 == 1.0 == thresh.
        let q = Array2::from_shape_vec((2, 2), vec![4.0, 0.5, 0.5, 4.0]).unwrap();
        let s = PairScreen::from_shell_bounds(&q, &dims, &offs, 2.0, 1.0);
        assert_eq!(
            s.shell_pair_counts(),
            (2, 3),
            "the (1,0) pair is exactly at thresh"
        );
        assert_eq!(s.ncols(), 3 + 3);
        assert!(s.cols().iter().all(|&(mu, nu)| (mu < 2) == (nu < 2)));
    }

    fn water(basis: &str) -> PreparedBasis {
        let mol = ferric_core::mol::Molecule::parse_xyz(
            "3\n\nO 0 0 0.117\nH 0 0.757 -0.467\nH 0 -0.757 -0.467\n",
            0,
            1,
        )
        .unwrap();
        PreparedBasis::new(&mol, &ferric_core::basis::bundled(basis).unwrap()).unwrap()
    }

    fn fill_err(
        obs: &PreparedBasis,
        dfbs: &PreparedBasis,
        band: (usize, usize),
        screen: &PairScreen,
    ) -> String {
        let mut out = Array2::<f64>::zeros((1, screen.ncols()));
        fill_eri3_pairs(Operator::coulomb(), obs, dfbs, band, screen, &mut out)
            .expect_err("must be refused")
            .to_string()
    }

    /// `PairScreen` and `fill_eri3_pairs` are public, so a screen built for a
    /// different orbital basis must be refused before the raw-pointer scatter,
    /// not written past its rows.
    #[test]
    fn a_screen_for_another_basis_is_refused() {
        let (svp, sto, aux) = (
            water("def2-svp"),
            water("sto-3g"),
            water("def2-universal-jkfit"),
        );
        let screen = PairScreen::schwarz(Operator::coulomb(), &svp, 1.0, 0.0).unwrap();
        let msg = fill_err(&sto, &aux, (0, 1), &screen);
        assert!(msg.contains("does not match the orbital basis"), "{msg}");
    }

    /// A reversed band or one past naux is an error, not an underflow panic or
    /// silently zero trailing rows.
    #[test]
    fn an_invalid_aux_band_is_refused() {
        let (obs, aux) = (water("sto-3g"), water("def2-universal-jkfit"));
        let screen = PairScreen::schwarz(Operator::coulomb(), &obs, 1.0, 0.0).unwrap();
        let naux = aux.nbasis();
        for band in [(3, 1), (0, naux + 1)] {
            let msg = fill_err(&obs, &aux, band, &screen);
            assert!(msg.contains("invalid aux band"), "{band:?}: {msg}");
        }
    }
}
