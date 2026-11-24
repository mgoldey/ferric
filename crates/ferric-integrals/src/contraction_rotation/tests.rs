use super::*;
use ferric_core::basis;

/// A Cartesian (non-pure) shell of angular momentum `l` with the given
/// exponents `e` and coefficients `c`.
fn sh(l: i32, e: &[f64], c: &[f64]) -> Shell {
    Shell {
        l,
        pure: false,
        exponents: e.to_vec(),
        coefficients: c.to_vec(),
    }
}

fn norm(s: &Shell) -> f64 {
    contraction_norm2(s.l, &s.exponents, &s.coefficients).sqrt()
}

/// The rotated column reproduces the parent as a function, primitive by
/// primitive over the libint-normalised shells:
/// `c_k / √s(c_k) = T_kk c'_k / √s(c'_k) + T_ks c_s / √s(c_s)`.
#[test]
fn rotated_row_reconstructs_the_parent_column() {
    let e = [30.0, 5.0, 1.0, 0.2];
    let big = sh(0, &e, &[0.1, 0.4, 0.5, 0.3]);
    let single = sh(0, &e, &[0.0, 0.0, 0.0, 1.3]);
    let t = rotate_templates(
        &[big.clone(), single],
        ColumnRotationMutant::Production,
        RotationLevel::Rot1,
    );
    assert_eq!(t.n_rotated, 1);
    assert_eq!(t.shells[0].exponents, vec![30.0, 5.0, 1.0]);
    assert_eq!(t.shells[1].exponents, vec![0.2]);
    let row = &t.rows[0];
    assert_eq!(row[0].0, 0);
    assert_eq!(row[1].0, 1);
    let (nk, nr, ns) = (norm(&big), norm(&t.shells[0]), norm(&t.shells[1]));
    for p in 0..4 {
        let parent = big.coefficients[p] / nk;
        let rot = if p < 3 {
            t.shells[0].coefficients[p] / nr
        } else {
            0.0
        };
        let s = if p == 3 {
            t.shells[1].coefficients[0] / ns
        } else {
            0.0
        };
        let got = row[0].1 * rot + row[1].1 * s;
        assert!(
            (got - parent).abs() <= 1e-15,
            "primitive {p}: {got} vs {parent}"
        );
    }
    // Nothing to rotate: identity rows, templates untouched.
    let seg = rotate_templates(
        &[big, sh(1, &e, &[0.0, 0.0, 0.0, 1.0])],
        ColumnRotationMutant::Production,
        RotationLevel::Ogc,
    );
    assert_eq!(seg.n_rotated, 0);
    assert!(seg
        .rows
        .iter()
        .enumerate()
        .all(|(k, r)| r == &vec![(k, 1.0)]));
}

/// Largest deviation, over every parent shell of `tmpls` and every primitive
/// of its exponent list, between the normalized parent contraction and
/// `Σ_m T_km` (normalized rotated contraction `m`), with the rotated
/// primitives mapped onto the parent's list by exponent bits.
fn reconstruction_error(tmpls: &[Shell], tr: &TemplateRotation) -> f64 {
    let mut worst = 0.0f64;
    for (k, parent) in tmpls.iter().enumerate() {
        let nk = norm(parent);
        let mut got = vec![0.0; parent.exponents.len()];
        for &(m, t) in &tr.rows[k] {
            let r = &tr.shells[m];
            let nr = norm(r);
            for (e, c) in r.exponents.iter().zip(&r.coefficients) {
                let p = parent
                    .exponents
                    .iter()
                    .position(|x| x.to_bits() == e.to_bits())
                    .expect("rotated primitive in the parent's exponent list");
                got[p] += t * c / nr;
            }
        }
        for (p, g) in got.iter().enumerate() {
            worst = worst.max((g - parent.coefficients[p] / nk).abs());
        }
    }
    worst
}

/// cc-pVDZ / cc-pVTZ carbon, oxygen and hydrogen: both stages reproduce every
/// parent column to round-off, and the OGC stage gives the design's
/// primitive counts (cc-pVDZ C: s 7,7,1, p 3,1, d 1; H: s 3,1, p 1).
#[test]
fn ogc_reconstructs_every_parent_column_and_counts_match_the_design() {
    for name in ["cc-pvdz", "cc-pvtz", "aug-cc-pvdz"] {
        let bs = basis::bundled(name).unwrap();
        for z in [1, 6, 8] {
            let tmpls = bs.for_element(z).unwrap();
            for level in [RotationLevel::Rot1, RotationLevel::Ogc] {
                let tr = rotate_templates(tmpls, ColumnRotationMutant::Production, level);
                assert!(tr.n_rotated > 0, "{name} Z={z} {level:?} rotates nothing");
                let err = reconstruction_error(tmpls, &tr);
                assert!(err < 1e-14, "{name} Z={z} {level:?}: {err:e}");
            }
        }
    }
    let bs = basis::bundled("cc-pvdz").unwrap();
    let nprims = |z: i32, level| -> Vec<usize> {
        rotate_templates(
            bs.for_element(z).unwrap(),
            ColumnRotationMutant::Production,
            level,
        )
        .shells
        .iter()
        .map(|s| s.exponents.len())
        .collect()
    };
    assert_eq!(nprims(6, RotationLevel::Rot1), vec![8, 8, 1, 3, 1, 1]);
    assert_eq!(nprims(6, RotationLevel::Ogc), vec![7, 7, 1, 3, 1, 1]);
    assert_eq!(nprims(1, RotationLevel::Ogc), vec![3, 1, 1]);
}

fn water() -> Molecule {
    Molecule::parse_xyz("3\nH2O\nO 0 0 0.1\nH 0 0.76 -0.47\nH 0 -0.76 -0.47\n", 0, 1).unwrap()
}

fn ethylene() -> Molecule {
    Molecule::parse_xyz(
        "6\nC2H4\nC 0 0 0.667\nC 0 0 -0.667\nH 0 0.923 1.237\nH 0 -0.923 1.237\n\
         H 0 0.923 -1.237\nH 0 -0.923 -1.237\n",
        0,
        1,
    )
    .unwrap()
}

/// `T` is invertible and well conditioned, the rotated basis keeps the AO
/// layout, and the parent overlap is recovered as `S = T S' Tᵀ` (so the
/// rotated basis spans exactly the parent's space, function by function).
#[test]
fn rotated_basis_is_invertible_and_reproduces_the_overlap() {
    for (mol, name) in [
        (water(), "cc-pvdz"),
        (water(), "cc-pvtz"),
        (ethylene(), "cc-pvdz"),
        (ethylene(), "aug-cc-pvdz"),
    ] {
        let bs = basis::bundled(name).unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let s = crate::oneelectron::overlap(&prep);
        for level in [RotationLevel::Rot1, RotationLevel::Ogc] {
            let rb = RotatedBasis::detect_prepared(&prep, level)
                .unwrap()
                .expect("a generally contracted basis rotates");
            assert_eq!(rb.prep.shell_offsets(), prep.shell_offsets());
            let t = rb.dense_t();
            let n = t.nrows();
            let rows: Vec<Vec<f64>> = (0..n).map(|i| t.row(i).to_vec()).collect();
            let tinv = invert(&rows).expect("T invertible");
            let mut worst_id = 0.0f64;
            for i in 0..n {
                for j in 0..n {
                    let v: f64 = (0..n).map(|k| rows[i][k] * tinv[k][j]).sum();
                    let target = if i == j { 1.0 } else { 0.0 };
                    worst_id = worst_id.max((v - target).abs());
                }
            }
            assert!(
                worst_id < 1e-13,
                "{name} {level:?}: T T⁻¹ − I = {worst_id:e}"
            );
            let s_rot = crate::oneelectron::overlap(&rb.prep);
            let back = rb.back_transform_general(&s_rot).unwrap();
            let sym = rb.back_transform_matrix(&s_rot).unwrap();
            let dense = t.dot(&s_rot).dot(&t.t());
            let err = (&back - &s).iter().fold(0.0f64, |m, x| m.max(x.abs()));
            let err_sym = (&sym - &s).iter().fold(0.0f64, |m, x| m.max(x.abs()));
            let err_dense = (&dense - &back).iter().fold(0.0f64, |m, x| m.max(x.abs()));
            assert!(err < 1e-13, "{name} {level:?}: |T S' Tᵀ − S| = {err:e}");
            assert!(
                err_sym < 1e-13,
                "{name} {level:?}: symmetric pass {err_sym:e}"
            );
            assert!(err_dense < 1e-14, "{name} {level:?}: dense T {err_dense:e}");
            // The forward transform is the transpose map: Tᵀ X T.
            let fwd = rb.ao_transpose(DerivRotMutant::None).matrix(&s);
            let fwd_dense = t.t().dot(&s).dot(&t);
            let err_fwd = (&fwd - &fwd_dense)
                .iter()
                .fold(0.0f64, |m, x| m.max(x.abs()));
            assert!(err_fwd < 1e-14, "{name} {level:?}: forward {err_fwd:e}");
        }
    }
}

/// Segmented bases (no shared exponent lists with a single-primitive column
/// or two long columns) rotate nothing: `None`, the caller's old path.
#[test]
fn segmented_bases_rotate_nothing() {
    for name in ["sto-3g", "6-31g*", "def2-svp", "def2-tzvp", "pcseg-1"] {
        let bs = basis::bundled(name).unwrap();
        let prep = PreparedBasis::new(&ethylene(), &bs).unwrap();
        for level in [RotationLevel::Rot1, RotationLevel::Ogc] {
            assert!(
                RotatedBasis::detect_prepared(&prep, level)
                    .unwrap()
                    .is_none(),
                "{name} {level:?} rotated"
            );
        }
    }
}

/// The prepared-shell detection and the element-template detection (the
/// periodic walks' entry) give the same rotation, bit for bit.
#[test]
fn detect_prepared_matches_detect() {
    let mol = ethylene();
    let bs = basis::bundled("cc-pvdz").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    for level in [RotationLevel::Rot1, RotationLevel::Ogc] {
        let a = RotatedBasis::detect(&mol, &prep, ColumnRotationMutant::Production, level, "t")
            .unwrap()
            .unwrap();
        let b = RotatedBasis::detect_prepared(&prep, level)
            .unwrap()
            .unwrap();
        assert_eq!(a.rows, b.rows);
        assert_eq!(a.groups, b.groups);
        // The molecular path counts non-identity rows; stage 1 alone counts
        // the same columns on both paths.
        if level == RotationLevel::Rot1 {
            assert_eq!(a.n_rotated_columns, b.n_rotated_columns);
        }
        let (la, lb) = (a.prep.located_shells(), b.prep.located_shells());
        for (x, y) in la.iter().zip(&lb) {
            assert_eq!(x.exponents, y.exponents);
            assert_eq!(x.coefficients, y.coefficients);
        }
    }
}

/// The in-place `out += T X' Tᵀ` equals the dense product, for a symmetric
/// input (then exactly symmetric) and a non-symmetric one, on top of a
/// non-zero `out`.
#[test]
fn back_transform_add_matches_dense_t() {
    let mol = ethylene();
    let bs = basis::bundled("cc-pvdz").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let rb = RotatedBasis::detect_prepared(&prep, RotationLevel::Ogc)
        .unwrap()
        .unwrap();
    let t = rb.dense_t();
    let n = t.nrows();
    let x = Array2::from_shape_fn((n, n), |(i, j)| ((i * 7 + j * 3) % 13) as f64 * 0.1 - 0.4);
    let sym = 0.5 * (&x + &x.t());
    let base = Array2::from_shape_fn((n, n), |(i, j)| (i + 2 * j) as f64 * 1e-3);
    let base = 0.5 * (&base + &base.t());
    for (input, symmetric) in [(&sym, true), (&x, false)] {
        let mut out = base.clone();
        rb.back_transform_add(input, &mut out).unwrap();
        let dense = &base + &t.dot(input).dot(&t.t());
        let err = (&out - &dense).iter().fold(0.0f64, |m, v| m.max(v.abs()));
        assert!(err < 1e-14, "symmetric {symmetric}: {err:e}");
        if symmetric {
            assert!(out
                .iter()
                .zip(out.t().iter())
                .all(|(a, b)| a.to_bits() == b.to_bits()));
        }
    }
}

/// The molecular path's conditioning gate: a group whose stage-1 block is
/// worse conditioned than `OGC_MAX_COND` stays unrotated (the periodic path
/// keeps it).
#[test]
fn ill_conditioned_stage_one_group_is_left_unrotated() {
    // The single column's primitive carries almost all of the long column's
    // weight, so T_kk is tiny (but above the stage-1 guard) and κ(T) ≫ 100.
    let e = [10.0, 1.0, 0.1];
    let tmpls = vec![sh(0, &e, &[0.01, 0.01, 1.0]), sh(0, &e, &[0.0, 0.0, 1.0])];
    let mut tr = rotate_templates(
        &tmpls,
        ColumnRotationMutant::Production,
        RotationLevel::Rot1,
    );
    assert_eq!(tr.n_rotated, 1, "stage 1 rotates it");
    gate_conditioning(&tmpls, &mut tr);
    assert_eq!(tr.n_rotated, 0);
    assert!(tr
        .rows
        .iter()
        .enumerate()
        .all(|(k, r)| r == &vec![(k, 1.0)]));
    assert_eq!(tr.shells, tmpls);
    // A well-conditioned group passes the gate unchanged.
    let good = vec![sh(0, &e, &[0.3, 0.6, 0.2]), sh(0, &e, &[0.0, 0.0, 1.0])];
    let mut tg = rotate_templates(&good, ColumnRotationMutant::Production, RotationLevel::Rot1);
    let before = tg.rows.clone();
    gate_conditioning(&good, &mut tg);
    assert_eq!(tg.rows, before);
    assert_eq!(tg.n_rotated, 1);
}
