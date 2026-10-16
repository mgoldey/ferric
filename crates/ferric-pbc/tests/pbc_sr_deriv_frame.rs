//! The nucleus-frame evaluation of the SR attraction derivative blocks (#434).
//!
//! PHYSICS: a 3-centre integral and its derivative are invariant under a common
//! translation of the three centres. In floating point libint2 combines the
//! ABSOLUTE coordinates with the zeta-sized exponents of the Gaussian nucleus,
//! so far from the origin the derivative blocks (site, sh1 and sh2 alike)
//! carry rounding that grows with zeta: 4.7e-8 of the block at zeta = 1e9,
//! 2e-4 at 1e12 and 44-100% at 1e16 (measured below on a cell offset by
//! (11.3, -7.7, 9.1) Bohr). Evaluating in the frame whose origin is the
//! nucleus (`hcore::nucleus_frame_shifts`: all three shells shifted by `-x`)
//! gives blocks that agree with central FD of the energy block to <= 1e-8 of
//! the block (<= 9.5e-9, the h = 1e-4 FD floor) at every zeta.
//!
//! ARTIFACT vs PHYSICS: if the old "wrong site/bra derivative at 1e16" were a
//! libint2 limit it would survive the frame change; it does not. The unframed
//! evaluation is asserted to be BAD at zeta = 1e16 (the witness that this
//! test can fail), the framed one good at every zeta. FD step h = 1e-4 on the
//! z coordinate of each centre in turn; the FD is taken in the SAME frame as
//! the analytic block it checks.
mod common;
use common::*;
use ferric_integrals::engine::Engine;
use ferric_integrals::operator::Operator;
use ferric_integrals::site_basis::SiteBasis;

const FD_H: f64 = 1e-4;
/// Framed blocks vs FD, as a fraction of the largest block. MEASURED worst
/// 9.5e-9 (site block, the same at every zeta: the FD step is the floor).
const FRAMED_BAR: f64 = 1e-7;
/// The unframed blocks at zeta = 1e16 must miss by more than this fraction
/// (MEASURED 0.44 for sh2, 1.0 for sh1 and site).
const UNFRAMED_WITNESS: f64 = 1e-2;

/// Worst `|analytic - FD|` of the z-derivative block of centre `which`
/// (0 site, 1 sh1, 2 sh2) over every shell pair, in the nucleus frame
/// (`framed`) or the lab frame, and the largest block.
fn worst_fd_miss(zeta: f64, framed: bool, which: usize) -> (f64, f64) {
    let off = [11.3, -7.7, 9.1];
    let pos: Vec<[f64; 3]> = H2_ATOMS
        .iter()
        .map(|p| [p[0] + off[0], p[1] + off[1], p[2] + off[2]])
        .collect();
    let cell = ferric_pbc::lattice::Cell::new(hydrogens(&pos), cubic(30.0)).unwrap();
    let prep = prep_for(&cell, &sp_basis_h());
    let nsh = prep.shell_dims().len();
    let x = pos[1];
    let site = SiteBasis::new(&[[x[0], x[1], x[2], zeta]], 0).unwrap();
    let op = Operator::erfc(0.8);
    let mut deriv =
        Engine::new_3center_deriv(op, &prep, &site.prep, f64::MIN_POSITIVE).expect("deriv engine");
    let mut value =
        Engine::new_3center(op, &prep, &site.prep, f64::MIN_POSITIVE).expect("value engine");
    let mx = [-x[0], -x[1], -x[2]];
    let base = if framed { [mx, mx, mx] } else { [[0.0; 3]; 3] };
    let (mut miss, mut scale) = (0.0_f64, 0.0_f64);
    for i1 in 0..nsh {
        for i2 in 0..nsh {
            let Some(blk) = deriv
                .compute_eri3_deriv_shifted(&prep, &site.prep, site.site_shell[0], i1, i2, base)
                .unwrap()
                .map(<[f64]>::to_vec)
            else {
                continue;
            };
            let bs = blk.len() / 9;
            let mut energy = |sign: f64| {
                let mut sh = base;
                sh[which][2] += sign * FD_H;
                value
                    .compute_eri3_shifted(&prep, &site.prep, site.site_shell[0], i1, i2, sh)
                    .unwrap()
                    .map_or_else(|| vec![0.0; bs], <[f64]>::to_vec)
            };
            let (ep, em) = (energy(1.0), energy(-1.0));
            for e in 0..bs {
                let fd = (ep[e] - em[e]) / (2.0 * FD_H);
                let a = blk[(3 * which + 2) * bs + e];
                miss = miss.max((a - fd).abs());
            }
            for v in &blk {
                scale = scale.max(v.abs());
            }
        }
    }
    (miss, scale)
}

#[test]
fn nucleus_frame_derivative_blocks_match_fd_at_every_exponent_and_the_lab_frame_does_not() {
    for zeta in [1e9, 1e12, 1e16] {
        for which in 0..3 {
            let name = ["site", "sh1", "sh2"][which];
            let (mf, sf) = worst_fd_miss(zeta, true, which);
            let (mu, su) = worst_fd_miss(zeta, false, which);
            eprintln!(
                "  zeta {zeta:.0e} {name}_z: framed {:.2e} of block, lab frame {:.2e}",
                mf / sf,
                mu / su
            );
            assert!(
                mf <= FRAMED_BAR * sf,
                "{name} zeta {zeta:e}: {mf:e} vs {sf:e}"
            );
            if zeta >= 1e16 {
                assert!(
                    mu > UNFRAMED_WITNESS * su,
                    "witness: the lab frame should be bad at 1e16 ({name}: {mu:e} vs {su:e})"
                );
            }
        }
    }
}
