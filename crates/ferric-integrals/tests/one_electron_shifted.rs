//! `Engine::compute_1e_block_shifted` / `scf_compute_1e_block_shifted`
//! (Stage-1 PBC commit 2, reference/pbc/stage1-design.md §3).
//!
//! * Exactness anchor: `shift = 0` is BITWISE equal to `compute_1e_block` for
//!   overlap, kinetic and nuclear on every shell pair (same shell data, only a
//!   copied-and-moved origin).
//! * Direction anchor: a nonzero shift `L` equals the off-diagonal block of an
//!   ordinary two-copy molecule whose second copy (ghost atoms) sits at `+L`.
//!   A lattice SUM over ±L is blind to the sign of the shift; this is not.
//! * Honesty: a non-finite shift and an out-of-range shell are `Err`, not a
//!   panic or garbage.
//!
//! The lattice-sum anchor (`Σ_L` shifted overlap ≡ `pair_ft(G=0)`) lives in
//! `crates/ferric-pbc/tests/pbc_shifted_overlap.rs` (ferric-integrals must not
//! depend on ferric-pbc).

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::engine::Engine;
use ferric_integrals::ffi;

const WATER: &str = "3
water
O  0.0000  0.0000  0.1173
H  0.0000  0.7572 -0.4692
H  0.0000 -0.7572 -0.4692
";

fn water() -> Molecule {
    Molecule::parse_xyz(WATER, 0, 1).expect("water")
}

fn engine(op: std::os::raw::c_int, prep: &PreparedBasis) -> Engine {
    let mut eng = Engine::new_1e(op, prep, 1e-14).expect("1e engine");
    if op == ffi::OP_NUCLEAR {
        eng.set_point_charges(prep).expect("point charges");
    }
    eng
}

#[test]
fn zero_shift_is_bitwise_identical_to_compute_1e_block() {
    let mol = water();
    let bs = basis::bundled("cc-pvdz").expect("basis");
    let prep = PreparedBasis::new(&mol, &bs).expect("prep");
    let nsh = prep.nshells();
    for (name, op) in [
        ("overlap", ffi::OP_OVERLAP),
        ("kinetic", ffi::OP_KINETIC),
        ("nuclear", ffi::OP_NUCLEAR),
    ] {
        let mut plain = engine(op, &prep);
        let mut shifted = engine(op, &prep);
        let mut nonzero = 0usize;
        for s1 in 0..nsh {
            for s2 in 0..nsh {
                let a = plain.compute_1e_block(&prep, s1, s2).to_vec();
                let b = shifted
                    .compute_1e_block_shifted(&prep, s1, s2, [0.0; 3])
                    .expect("shifted block")
                    .to_vec();
                assert_eq!(a.len(), b.len(), "{name} ({s1},{s2}) length");
                for (i, (x, y)) in a.iter().zip(&b).enumerate() {
                    assert_eq!(
                        x.to_bits(),
                        y.to_bits(),
                        "{name} ({s1},{s2})[{i}]: {x} vs {y}"
                    );
                }
                nonzero += a.iter().filter(|v| **v != 0.0).count();
            }
        }
        assert!(nonzero > 0, "{name}: vacuous (all blocks zero)");
    }
}

/// `⟨μ_0 | ν(r − L)⟩` must equal the (cell-0, image) block of a molecule that
/// contains a ghost copy of every atom displaced by `+L`.
#[test]
fn nonzero_shift_matches_a_ghost_copy_displaced_by_plus_l() {
    let mol = water();
    let bs = basis::bundled("cc-pvdz").expect("basis");
    let prep0 = PreparedBasis::new(&mol, &bs).expect("prep0");
    let nsh0 = prep0.nshells();
    // Asymmetric shift: a sign or axis mix-up cannot pass by symmetry.
    let l = [1.3, -0.7, 2.1];
    let mut atoms = mol.atoms.clone();
    for a in &mol.atoms {
        let mut g = a.clone();
        g.x += l[0];
        g.y += l[1];
        g.zpos += l[2];
        g.ghost = true;
        atoms.push(g);
    }
    let pair = Molecule {
        atoms,
        charge: 0,
        multiplicity: 1,
    };
    let prep2 = PreparedBasis::new(&pair, &bs).expect("prep2");
    assert_eq!(prep2.nshells(), 2 * nsh0, "ghost copy shell count");

    for (name, op) in [("overlap", ffi::OP_OVERLAP), ("kinetic", ffi::OP_KINETIC)] {
        let mut e0 = engine(op, &prep0);
        let mut e2 = engine(op, &prep2);
        let mut max_err = 0.0_f64;
        let mut max_val = 0.0_f64;
        for s1 in 0..nsh0 {
            for s2 in 0..nsh0 {
                let want = e2.compute_1e_block(&prep2, s1, nsh0 + s2).to_vec();
                let got = e0
                    .compute_1e_block_shifted(&prep0, s1, s2, l)
                    .expect("shifted")
                    .to_vec();
                assert_eq!(want.len(), got.len());
                for (x, y) in want.iter().zip(&got) {
                    max_err = max_err.max((x - y).abs());
                    max_val = max_val.max(x.abs());
                }
            }
        }
        eprintln!("{name}: max|shifted - ghost block| = {max_err:.2e} (max|block| {max_val:.2e})");
        assert!(max_val > 1e-3, "{name}: vacuous, image too far");
        assert!(max_err < 1e-14, "{name}: {max_err:.3e}");

        // Negative control: the OPPOSITE shift must NOT match (else the test
        // cannot see the sign).
        let mut worst = 0.0_f64;
        for s1 in 0..nsh0 {
            for s2 in 0..nsh0 {
                let want = e2.compute_1e_block(&prep2, s1, nsh0 + s2).to_vec();
                let got = e0
                    .compute_1e_block_shifted(&prep0, s1, s2, [-l[0], -l[1], -l[2]])
                    .expect("shifted")
                    .to_vec();
                for (x, y) in want.iter().zip(&got) {
                    worst = worst.max((x - y).abs());
                }
            }
        }
        assert!(
            worst > 1e-3,
            "{name}: -L indistinguishable from +L ({worst:.2e})"
        );
    }
}

#[test]
fn nonfinite_shift_and_bad_shell_are_errors() {
    let mol = water();
    let bs = basis::bundled("sto-3g").expect("basis");
    let prep = PreparedBasis::new(&mol, &bs).expect("prep");
    let mut eng = engine(ffi::OP_OVERLAP, &prep);
    for bad in [
        [f64::NAN, 0.0, 0.0],
        [0.0, f64::INFINITY, 0.0],
        [0.0, 0.0, f64::NEG_INFINITY],
    ] {
        let e = eng
            .compute_1e_block_shifted(&prep, 0, 0, bad)
            .expect_err("non-finite shift must error")
            .to_string();
        assert!(e.contains("non-finite shift"), "{e}");
    }
    let nsh = prep.nshells();
    let e = eng
        .compute_1e_block_shifted(&prep, 0, nsh, [0.0; 3])
        .expect_err("out-of-range shell must error")
        .to_string();
    assert!(e.contains(&format!("(0,{nsh}) out of range")), "{e}");
}
