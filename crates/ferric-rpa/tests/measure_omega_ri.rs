//! Measurement (not a test): omega-RI accuracy and cost against the
//! Coulomb-fit O(N^5) baselines on the n-alkane series, one SCF per molecule.
//!
//! ```text
//! FERRIC_OMEGA_RI_SIZES=4,8,12,16,20,24 OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=4 \
//!   cargo test --release -p ferric-rpa --test measure_omega_ri -- --ignored --nocapture
//! ```
//!
//! cc-pVDZ / cc-pVDZ-RI, frozen 1s cores, n_quad = 7. Prints one JSON line per
//! molecule (`OMEGA_RI_ROW {...}`): energies (Eh) and wall seconds of
//! * `ri_mp2` (Coulomb) and `ri_mp2_fitted` at omega = 0.05, 0.1, 0.2;
//! * `LaplaceMp2::compute_mo` (Coulomb, the MO Laplace baseline) and
//!   `omega_ri_cdd_mp2` at omega = 0.1: unscreened, paper screens
//!   (theta_schwarz 1e-9, theta_ij 1e-6), each screen alone, and cdd_tol 1e-6;
//! * `run_pdep_rpa` (Coulomb) and `run_pdep_rpa_fitted` at omega = 0.1
//!   (skipped above `FERRIC_OMEGA_RI_RPA_MAX` carbons, default 16).
//!
//! `FERRIC_OMEGA_RI_PROFILE=1`: only `ri_mp2` (Coulomb) and the unscreened /
//! paper-screened CDD. Every CDD row carries its stage wall times (`stages`)
//! and the exchange GEMM work fractions (`kept_work`, `range_work`).
//!
//! Artifact hypothesis for the screening-error growth: if the growth were a
//! construction error in the screen (wrong index set), the pair-only and
//! virtual-only errors would not add up to the combined one and the error
//! at zero thresholds would be nonzero; both are printed.

use std::path::PathBuf;
use std::time::Instant;

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::laplace::LaplaceMp2;
use ferric_mp2::omega_ri::RiFit;
use ferric_mp2::omega_ri_cdd::{omega_ri_cdd_mp2, OmegaRiCddConfig, PseudoOrbitals};
use ferric_mp2::rimp2::{ri_mp2_fitted, RiMp2Config};
use ferric_rpa::config::PdepRpaConfig;
use ferric_rpa::{run_pdep_rpa, run_pdep_rpa_fitted};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

fn timed<T>(f: impl FnOnce() -> T) -> (T, f64) {
    let t0 = Instant::now();
    let v = f();
    (v, t0.elapsed().as_secs_f64())
}

#[test]
#[ignore = "measurement: omega-RI accuracy and cost on n-alkanes"]
fn measure_omega_ri_alkanes() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let sizes: Vec<usize> = std::env::var("FERRIC_OMEGA_RI_SIZES")
        .unwrap_or_else(|_| "4,8".into())
        .split(',')
        .map(|s| s.trim().parse().unwrap())
        .collect();
    let rpa_max: usize = std::env::var("FERRIC_OMEGA_RI_RPA_MAX")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(16);
    let lap_max: usize = std::env::var("FERRIC_OMEGA_RI_LAPMO_MAX")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(24);
    let rijk = std::env::var("FERRIC_OMEGA_RI_RIJK").is_ok_and(|v| v == "1");
    let profile = std::env::var("FERRIC_OMEGA_RI_PROFILE").is_ok_and(|v| v == "1");
    let op = Operator::coulomb();
    for n in sizes {
        let xyz = root.join(format!("testdata/molecules/alkane_{n}.xyz"));
        let mol = Molecule::load_xyz(xyz.to_str().unwrap()).unwrap();
        let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
        let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let (rhf, t_scf) = timed(|| {
            solve_rhf(&ParallelContext::default(), &mol, &obs, op, &bounds, &{
                let mut c = RhfConfig::default()
                    .with_energy_conv(1e-10)
                    .with_density_conv(1e-8)
                    .with_max_iter(200);
                // FERRIC_OMEGA_RI_RIJK=1: RI-JK SCF (def2-universal-jkfit) to reach
                // larger sizes; every method then reads the same orbitals.
                c.df_j_aux = rijk.then(|| "def2-universal-jkfit".to_string());
                c.df_k_aux = rijk.then(|| "def2-universal-jkfit".to_string());
                c
            })
            .unwrap()
        });
        assert!(rhf.converged);
        let fc = n; // carbon 1s
        let mp2_cfg = RiMp2Config {
            frozen_core: fc,
            ..Default::default()
        };
        let mut row = serde_json::json!({
            "n_carbon": n, "nbf": obs.nbasis(), "naux": dfbs.nbasis(), "t_scf": t_scf, "scf_rijk": rijk,
        });
        let (rc, t) = timed(|| {
            ri_mp2_fitted(&mol, &obs, &dfbs, op, &rhf, &mp2_cfg, &RiFit::Coulomb).unwrap()
        });
        row["rimp2_coulomb"] = serde_json::json!({"e": rc.mp2_corr, "t": t});
        let omegas: &[f64] = if profile { &[] } else { &[0.05, 0.1, 0.2] };
        for &w in omegas {
            let fit = RiFit::omega_ri(w).unwrap();
            let (r, t) =
                timed(|| ri_mp2_fitted(&mol, &obs, &dfbs, op, &rhf, &mp2_cfg, &fit).unwrap());
            row[format!("rimp2_omega_{w}")] = serde_json::json!({"e": r.mp2_corr, "t": t});
        }
        if n <= lap_max && !profile {
            let (lap, t) = timed(|| {
                LaplaceMp2::new(7)
                    .compute_mo(&mol, &obs, &dfbs, op, &rhf, fc)
                    .unwrap()
            });
            row["laplace_mo_coulomb"] = serde_json::json!({"e": lap, "t": t});
        }
        let base = OmegaRiCddConfig {
            fit: RiFit::omega_ri(0.1).unwrap(),
            frozen_core: fc,
            ..Default::default()
        };
        let variants: [(&str, OmegaRiCddConfig); 5] = [
            ("cdd_unscreened", base.clone()),
            (
                "cdd_paper_screens",
                OmegaRiCddConfig {
                    schwarz_thresh: 1e-9,
                    pair_thresh: 1e-6,
                    ..base.clone()
                },
            ),
            (
                "cdd_pair_only",
                OmegaRiCddConfig {
                    pair_thresh: 1e-6,
                    ..base.clone()
                },
            ),
            (
                "cdd_schwarz_only",
                OmegaRiCddConfig {
                    schwarz_thresh: 1e-9,
                    ..base.clone()
                },
            ),
            (
                "cdd_tol_1e-6",
                OmegaRiCddConfig {
                    orbitals: PseudoOrbitals::Cholesky { tol: 1e-6 },
                    ..base.clone()
                },
            ),
        ];
        let keep = |tag: &str| !profile || tag == "cdd_unscreened" || tag == "cdd_paper_screens";
        for (tag, cfg) in variants.into_iter().filter(|(t, _)| keep(t)) {
            let (r, t) = timed(|| omega_ri_cdd_mp2(&mol, &obs, &dfbs, op, &rhf, &cfg).unwrap());
            let tm = r.timings;
            row[tag] = serde_json::json!({
                "e": r.e_corr, "t": t, "e_x": r.e_exchange, "pairs": r.pair_fraction,
                "virt": r.virtual_fraction, "rank_occ": r.mean_rank_occ, "rank_vir": r.mean_rank_vir,
                "nocc": r.nocc, "nvir": r.nvir,
                "kept_work": r.exchange_kept_work, "range_work": r.exchange_range_work,
                "stages": {"dress": tm.dress, "b_ov": tm.b_ov, "cholesky": tm.cholesky,
                    "transform": tm.transform, "coulomb": tm.coulomb, "exchange": tm.exchange},
            });
        }
        if n <= rpa_max && !profile {
            let cfg = PdepRpaConfig {
                frozen_core: fc,
                ..Default::default()
            };
            let (r, t) = timed(|| run_pdep_rpa(&mol, &obs, &dfbs, op, &rhf, &cfg).unwrap());
            row["rpa_coulomb"] = serde_json::json!({"e": r.e_rpa, "t": t});
            let fit = RiFit::omega_ri(0.1).unwrap();
            let (r, t) =
                timed(|| run_pdep_rpa_fitted(&mol, &obs, &dfbs, op, &rhf, &cfg, &fit).unwrap());
            row["rpa_omega_0.1"] = serde_json::json!({"e": r.e_rpa, "t": t});
        }
        println!("OMEGA_RI_ROW {row}");
    }
}
