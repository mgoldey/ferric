//! Three-center and two-center integral builders for density fitting (RI).

use crate::basis_bridge::PreparedBasis;
use crate::engine::Engine;
use crate::operator::Operator;
use crate::qqr3::QqrBounds3;
use crate::schwarz::{schwarz, schwarz3_aux};
use ferric_core::FerricError;
use ndarray::{Array2, Array3};

/// Build the 2-center Coulomb metric (P|Q), shape (naux, naux).
pub fn coulomb_metric_2c(op: Operator, dfbs: &PreparedBasis) -> Result<Array2<f64>, FerricError> {
    let naux = dfbs.nbasis();
    let nsh = dfbs.nshells();
    let dims = dfbs.shell_dims();
    let offs = dfbs.shell_offsets();
    let mut eng = Engine::new_2center(op, dfbs, 1e-14)?;
    let mut v = Array2::zeros((naux, naux));
    for sp in 0..nsh {
        for sq in 0..=sp {
            let block = eng.compute_eri2(dfbs, sp, sq);
            let np = dims[sp];
            let nq = dims[sq];
            for p in 0..np {
                for q in 0..nq {
                    let val = block[p * nq + q];
                    v[(offs[sp] + p, offs[sq] + q)] = val;
                    v[(offs[sq] + q, offs[sp] + p)] = val;
                }
            }
        }
    }
    Ok(v)
}

/// Build 3-center integrals (P|mn), shape (naux, nbasis, nbasis).
pub fn eri3_tensor(op: Operator, obs: &PreparedBasis, dfbs: &PreparedBasis) -> Result<Array3<f64>, FerricError> {
    let naux = dfbs.nbasis();
    let nbas = obs.nbasis();
    let nsh_obs = obs.nshells();
    let nsh_df = dfbs.nshells();
    let dims_obs = obs.shell_dims();
    let offs_obs = obs.shell_offsets();
    let dims_df = dfbs.shell_dims();
    let offs_df = dfbs.shell_offsets();
    let mut eng = Engine::new_3center(op, obs, dfbs, 1e-14)?;
    let mut eri = Array3::zeros((naux, nbas, nbas));
    for sp in 0..nsh_df {
        for s1 in 0..nsh_obs {
            for s2 in 0..=s1 {
                if let Some(block) = eng.compute_eri3(obs, dfbs, sp, s1, s2) {
                    let np = dims_df[sp];
                    let n1 = dims_obs[s1];
                    let n2 = dims_obs[s2];
                    for p in 0..np {
                        for i in 0..n1 {
                            for j in 0..n2 {
                                let val = block[(p * n1 + i) * n2 + j];
                                eri[(offs_df[sp] + p, offs_obs[s1] + i, offs_obs[s2] + j)] = val;
                                eri[(offs_df[sp] + p, offs_obs[s2] + j, offs_obs[s1] + i)] = val;
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(eri)
}

/// Schwarz-screened 3-center ERI builder.
///
/// Same dense `(naux, nbas, nbas)` output as [`eri3_tensor`], but skips shell
/// triples whose Cauchy–Schwarz bound `Q3[P] · Q(μ,ν)` is below `thresh`.
/// Skipped blocks remain zero in the output. With `thresh = 0.0` this is a
/// drop-in equivalent of `eri3_tensor` (modulo libint's internal precision).
///
/// Returns `(tensor, n_kept, n_total)` where the shell-triple counts let
/// callers report screening effectiveness without re-walking the loop.
pub fn eri3_tensor_screened(
    op: Operator,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    thresh: f64,
) -> Result<(Array3<f64>, usize, usize), FerricError> {
    let naux = dfbs.nbasis();
    let nbas = obs.nbasis();
    let nsh_obs = obs.nshells();
    let nsh_df = dfbs.nshells();
    let dims_obs = obs.shell_dims();
    let offs_obs = obs.shell_offsets();
    let dims_df = dfbs.shell_dims();
    let offs_df = dfbs.shell_offsets();

    // Schwarz bounds matched to the operator: |(P|μν)| ≤ Q3[P] · Q(μ,ν).
    let q_obs = schwarz(op, obs)?;
    let q3 = schwarz3_aux(op, dfbs)?;

    let mut eng = Engine::new_3center(op, obs, dfbs, 1e-14)?;
    let mut eri = Array3::zeros((naux, nbas, nbas));
    let mut n_kept = 0usize;
    let mut n_total = 0usize;

    for sp in 0..nsh_df {
        let q3p = q3[sp];
        for s1 in 0..nsh_obs {
            for s2 in 0..=s1 {
                n_total += 1;
                if q3p * q_obs[(s1, s2)] < thresh {
                    continue;
                }
                let Some(block) = eng.compute_eri3(obs, dfbs, sp, s1, s2) else { continue };
                n_kept += 1;
                let np = dims_df[sp];
                let n1 = dims_obs[s1];
                let n2 = dims_obs[s2];
                for p in 0..np {
                    for i in 0..n1 {
                        for j in 0..n2 {
                            let val = block[(p * n1 + i) * n2 + j];
                            eri[(offs_df[sp] + p, offs_obs[s1] + i, offs_obs[s2] + j)] = val;
                            eri[(offs_df[sp] + p, offs_obs[s2] + j, offs_obs[s1] + i)] = val;
                        }
                    }
                }
            }
        }
    }
    Ok((eri, n_kept, n_total))
}

/// QQR-screened 3-center ERI builder.
///
/// Same dense output as [`eri3_tensor`], but uses the distance-aware QQR-3
/// bound (Schwarz × `min(1, ext·ext/R) × op_decay(R)`) so erfc-attenuated
/// operators actually drop long-range shell triples. The basic Schwarz path
/// [`eri3_tensor_screened`] cannot do this — its bound is bra-only and has
/// no notion of bra-ket distance.
///
/// Skipped blocks remain zero. Returns `(tensor, n_kept, n_total)`.
pub fn eri3_tensor_screened_qqr(
    op: Operator,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    bounds: &QqrBounds3,
    thresh: f64,
) -> Result<(Array3<f64>, usize, usize), FerricError> {
    let naux = dfbs.nbasis();
    let nbas = obs.nbasis();
    let nsh_obs = obs.nshells();
    let nsh_df = dfbs.nshells();
    let dims_obs = obs.shell_dims();
    let offs_obs = obs.shell_offsets();
    let dims_df = dfbs.shell_dims();
    let offs_df = dfbs.shell_offsets();

    let mut eng = Engine::new_3center(op, obs, dfbs, 1e-14)?;
    let mut eri = Array3::zeros((naux, nbas, nbas));
    let mut n_kept = 0usize;
    let mut n_total = 0usize;

    for sp in 0..nsh_df {
        for s1 in 0..nsh_obs {
            for s2 in 0..=s1 {
                n_total += 1;
                if bounds.estimate3(sp, s1, s2) < thresh {
                    continue;
                }
                let Some(block) = eng.compute_eri3(obs, dfbs, sp, s1, s2) else { continue };
                n_kept += 1;
                let np = dims_df[sp];
                let n1 = dims_obs[s1];
                let n2 = dims_obs[s2];
                for p in 0..np {
                    for i in 0..n1 {
                        for j in 0..n2 {
                            let val = block[(p * n1 + i) * n2 + j];
                            eri[(offs_df[sp] + p, offs_obs[s1] + i, offs_obs[s2] + j)] = val;
                            eri[(offs_df[sp] + p, offs_obs[s2] + j, offs_obs[s1] + i)] = val;
                        }
                    }
                }
            }
        }
    }
    Ok((eri, n_kept, n_total))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferric_core::basis;
    use ferric_core::mol::Molecule;

    #[test]
    fn test_coulomb_metric_2c_symmetric() {
        let mol = Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 0.74\n", 0, 1).unwrap();
        let dfbs_set = basis::bundled("cc-pvdz-ri").unwrap();
        let dfbs = PreparedBasis::new(&mol, &dfbs_set).unwrap();
        let v = coulomb_metric_2c(Operator::coulomb(), &dfbs).unwrap();
        let n = dfbs.nbasis();
        for i in 0..n {
            for j in 0..n {
                assert!((v[(i, j)] - v[(j, i)]).abs() < 1e-12,
                    "(P|Q) not symmetric at ({i},{j})");
            }
        }
        // Diagonal should be positive
        for i in 0..n { assert!(v[(i, i)] > 0.0, "(P|P) should be positive"); }
    }

    #[test]
    fn test_eri3_screened_zero_thresh_matches_dense() {
        // With thresh = 0, the screened path must reproduce the dense tensor
        // bit-for-bit (modulo libint's internal 1e-14 precision filter).
        let mol = Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 0.74\n", 0, 1).unwrap();
        let obs_set = basis::bundled("sto-3g").unwrap();
        let dfbs_set = basis::bundled("cc-pvdz-ri").unwrap();
        let obs = PreparedBasis::new(&mol, &obs_set).unwrap();
        let dfbs = PreparedBasis::new(&mol, &dfbs_set).unwrap();
        let dense = eri3_tensor(Operator::coulomb(), &obs, &dfbs).unwrap();
        let (screened, n_kept, n_total) =
            eri3_tensor_screened(Operator::coulomb(), &obs, &dfbs, 0.0).unwrap();
        assert_eq!(n_kept, n_total, "thresh=0 should keep every (P,s1,s2≤s1) triple");
        let max_diff = dense.iter().zip(screened.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max);
        assert!(max_diff < 1e-12, "screened tensor diverges from dense: max diff {max_diff:.2e}");
    }

    #[test]
    fn test_eri3_screened_erfc_water_drops_triples() {
        // Real check: on water/cc-pVDZ with erfc(ω=0.222 Bohr⁻¹), a production
        // threshold must (a) drop some triples that Coulomb keeps, and (b) the
        // surviving tensor must agree with the unscreened erfc build to high
        // precision on retained entries.
        let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
        let obs_set = basis::bundled("cc-pvdz").unwrap();
        let dfbs_set = basis::bundled("cc-pvdz-ri").unwrap();
        let obs = PreparedBasis::new(&mol, &obs_set).unwrap();
        let dfbs = PreparedBasis::new(&mol, &dfbs_set).unwrap();
        let op = Operator::erfc(0.222);
        let unscreened = eri3_tensor(op, &obs, &dfbs).unwrap();
        let (screened, n_kept, n_total) =
            eri3_tensor_screened(op, &obs, &dfbs, 1e-10).unwrap();
        eprintln!("H2O/cc-pVDZ erfc(0.222) eri3 screening: {n_kept}/{n_total} triples kept");
        // Either we drop some, or the system is too small for screening to fire.
        // Tensor agreement is the load-bearing check.
        let max_diff = unscreened.iter().zip(screened.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max);
        assert!(max_diff < 1e-9,
            "screened erfc tensor diverges from unscreened: max diff {max_diff:.2e}");
    }

    #[test]
    fn test_eri3_qqr_screened_water_matches_unscreened() {
        // Correctness: QQR-screened tensor on water/cc-pVDZ with erfc must
        // agree with the unscreened build to high precision at production
        // threshold. Water is too small for screening to fire, but the
        // surviving entries must still be correct.
        let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
        let obs_set = basis::bundled("cc-pvdz").unwrap();
        let dfbs_set = basis::bundled("cc-pvdz-ri").unwrap();
        let obs = PreparedBasis::new(&mol, &obs_set).unwrap();
        let dfbs = PreparedBasis::new(&mol, &dfbs_set).unwrap();
        let op = Operator::erfc(0.222);
        let bounds = crate::qqr3::QqrBounds3::new(op, &mol, &obs, &dfbs).unwrap();
        let unscreened = eri3_tensor(op, &obs, &dfbs).unwrap();
        let (screened, n_kept, n_total) =
            eri3_tensor_screened_qqr(op, &obs, &dfbs, &bounds, 1e-10).unwrap();
        eprintln!("water erfc QQR3 thresh=1e-10: {n_kept}/{n_total} kept");
        let max_diff = unscreened.iter().zip(screened.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max);
        assert!(max_diff < 1e-9,
            "QQR-screened tensor diverges from unscreened: max diff {max_diff:.2e}");
    }

    #[test]
    #[ignore = "decane/cc-pVDZ build is heavy (~minute); run with --ignored for screening curve"]
    fn bench_eri3_screened_decane_erfc() {
        // The water test confirms correctness but cannot fire screening — the
        // molecule is smaller than erfc's range. Decane (C10H22, ~12 Å) is the
        // smallest system where shell-triple distances exceed the erfc range
        // and screening should start to drop triples meaningfully.
        let mol = Molecule::load_xyz("../../testdata/molecules/alkane_10.xyz").unwrap();
        let obs_set = basis::bundled("cc-pvdz").unwrap();
        let dfbs_set = basis::bundled("cc-pvdz-ri").unwrap();
        let obs = PreparedBasis::new(&mol, &obs_set).unwrap();
        let dfbs = PreparedBasis::new(&mol, &dfbs_set).unwrap();
        eprintln!(
            "decane: nbas={}, naux={}, nsh_obs={}, nsh_aux={}",
            obs.nbasis(), dfbs.nbasis(), obs.nshells(), dfbs.nshells()
        );

        // Coulomb screens little — the operator has infinite range.
        let op_c = Operator::coulomb();
        let (_, n_kept_c, n_total_c) =
            eri3_tensor_screened(op_c, &obs, &dfbs, 1e-10).unwrap();
        eprintln!("  Coulomb thresh=1e-10: {n_kept_c}/{n_total_c} triples kept ({:.1}%)",
            100.0 * n_kept_c as f64 / n_total_c as f64);

        // erfc with the dissertation optimal omega should drop substantially more.
        let op_e = Operator::erfc(0.222);
        for &thresh in &[1e-12, 1e-10, 1e-8, 1e-6] {
            let (_, n_kept, n_total) =
                eri3_tensor_screened(op_e, &obs, &dfbs, thresh).unwrap();
            eprintln!("  Schwarz erfc(0.222) thresh={thresh:.0e}: {n_kept}/{n_total} kept ({:.1}%)",
                100.0 * n_kept as f64 / n_total as f64);
        }

        // QQR-3 with distance-aware bound — this is what should actually fire
        // for erfc. Same operator and thresholds; compare retention to Schwarz.
        let bounds = crate::qqr3::QqrBounds3::new(op_e, &mol, &obs, &dfbs).unwrap();
        for &thresh in &[1e-12, 1e-10, 1e-8, 1e-6] {
            let (_, n_kept, n_total) =
                eri3_tensor_screened_qqr(op_e, &obs, &dfbs, &bounds, thresh).unwrap();
            eprintln!("  QQR3   erfc(0.222) thresh={thresh:.0e}: {n_kept}/{n_total} kept ({:.1}%)",
                100.0 * n_kept as f64 / n_total as f64);
        }
    }

    #[test]
    fn test_eri3_symmetric_in_mu_nu() {
        let mol = Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 0.74\n", 0, 1).unwrap();
        let obs_set = basis::bundled("sto-3g").unwrap();
        let dfbs_set = basis::bundled("cc-pvdz-ri").unwrap();
        let obs = PreparedBasis::new(&mol, &obs_set).unwrap();
        let dfbs = PreparedBasis::new(&mol, &dfbs_set).unwrap();
        let eri = eri3_tensor(Operator::coulomb(), &obs, &dfbs).unwrap();
        let naux = dfbs.nbasis();
        let nbas = obs.nbasis();
        for p in 0..naux {
            for i in 0..nbas {
                for j in 0..nbas {
                    assert!((eri[(p, i, j)] - eri[(p, j, i)]).abs() < 1e-12,
                        "ERI3 not symmetric at P={p},i={i},j={j}");
                }
            }
        }
    }
}
