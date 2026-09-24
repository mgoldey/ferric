//! Stage-1 PBC commit 2 anchor: the lattice-summed overlap built from the new
//! shim primitive, `S_latt[μ,ν] = Σ_L ⟨μ_0 | ν(r − L)⟩` via
//! `Engine::compute_1e_block_shifted`, equals the analytic pair-density FT at
//! G = 0 (`ferric_pbc::pair_ft`) — two INDEPENDENT constructions (libint2
//! Obara-Saika overlap vs McMurchie-Davidson Hermite E_0 in Rust).
//!
//! Artifact hypotheses: a wrong shift sign is INVISIBLE here (the image set is
//! closed under L → −L), which is why the direction anchor lives in
//! `ferric-integrals/tests/one_electron_shifted.rs`; a dropped image or a
//! per-image block written transposed (non-symmetric per L) shows up as an
//! asymmetric or wrong S_latt, which this test sees element by element.

use ferric_core::basis::bundled;
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::engine::Engine;
use ferric_integrals::ffi;
use ferric_pbc::lattice::Cell;
use ferric_pbc::pair_ft::{pair_ft, DEFAULT_PAIR_FT_THRESH};
use ndarray::Array2;

const WATER: &str = "3
water
O  0.0000  0.0000  0.1173
H  0.0000  0.7572 -0.4692
H  0.0000 -0.7572 -0.4692
";

/// Σ_L ⟨μ_0|ν_L⟩ over every image whose closest atom pair is within `rcut`.
fn shifted_lattice_overlap(cell: &Cell, prep: &PreparedBasis, rcut: f64) -> (Array2<f64>, usize) {
    let n = prep.nbasis();
    let nsh = prep.nshells();
    let dims = prep.shell_dims().to_vec();
    let offs = prep.shell_offsets().to_vec();
    let mut eng = Engine::new_1e(ffi::OP_OVERLAP, prep, 1e-14).expect("overlap engine");
    let images = cell.translations(rcut).expect("translations");
    let mut s = Array2::<f64>::zeros((n, n));
    for l in &images {
        for s1 in 0..nsh {
            for s2 in 0..nsh {
                let block = eng
                    .compute_1e_block_shifted(prep, s1, s2, *l)
                    .expect("shifted overlap block");
                let n2 = dims[s2];
                for i in 0..dims[s1] {
                    for j in 0..n2 {
                        s[(offs[s1] + i, offs[s2] + j)] += block[i * n2 + j];
                    }
                }
            }
        }
    }
    (s, images.len())
}

#[test]
fn shifted_lattice_overlap_equals_pair_ft_at_g0() {
    let mol = Molecule::parse_xyz(WATER, 0, 1).expect("water");
    // Triclinic and small enough that many images overlap the reference cell.
    let lattice = [[5.5, 0.0, 0.0], [1.1, 6.0, 0.0], [0.7, -0.9, 6.5]];
    let cell = Cell::new(mol, lattice).expect("cell");
    for basis in ["sto-3g", "cc-pvdz"] {
        let bs = bundled(basis).expect("basis");
        let prep = PreparedBasis::new(cell.mol(), &bs).expect("prep");
        let n = prep.nbasis();

        // Image range: a superset of pair_ft's own (its rpair + 2 Bohr).
        let amin = cell
            .mol()
            .atoms
            .iter()
            .flat_map(|t| bs.for_element(t.z).expect("element").iter())
            .flat_map(|sh| sh.exponents.iter().copied())
            .fold(f64::INFINITY, f64::min);
        let rcut = (2.0 * (1e3 / DEFAULT_PAIR_FT_THRESH).ln() / amin).sqrt() + 4.0;
        let (s, nimg) = shifted_lattice_overlap(&cell, &prep, rcut);
        let p = pair_ft(&cell, &prep, &[[0.0; 3]]).expect("pair_ft");

        let (mut err, mut err_im, mut asym, mut s0_diff) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
        let s0 = ferric_integrals::oneelectron::overlap(&prep);
        for m in 0..n {
            for k in 0..n {
                let z = p[[m, k, 0]];
                err = err.max((z.re - s[(m, k)]).abs());
                err_im = err_im.max(z.im.abs());
                asym = asym.max((s[(m, k)] - s[(k, m)]).abs());
                s0_diff = s0_diff.max((s[(m, k)] - s0[(m, k)]).abs());
            }
        }
        eprintln!(
            "{basis}: nao={n} images={nimg} max|S_latt(shifted) - Re P(0)| = {err:.2e}, \
             max|Im P(0)| = {err_im:.2e}, asym = {asym:.2e}, max|S_latt - S_mol| = {s0_diff:.2e}"
        );
        // Non-vacuity: the lattice sum must differ materially from the
        // molecular overlap, or images were not actually included.
        assert!(nimg > 1 && s0_diff > 1e-3, "{basis}: vacuous lattice sum");
        assert!(err < 1e-12, "{basis}: S_latt vs pair_ft(G=0) {err:.3e}");
        assert!(err_im < 1e-12, "{basis}: Im P(0) {err_im:.3e}");
        assert!(asym < 1e-13, "{basis}: S_latt asymmetry {asym:.3e}");
    }
}
