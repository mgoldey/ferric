//! att-MP2 static α and frozen-amplitude MP2 C6 attenuation sweep drivers.
//!
//! Construction under test: Coulomb RHF reference and Coulomb HF response,
//! erfc(ωr) on the MP2 correlation only (`mp2_polarizability_analytic` with
//! Coulomb `bounds`; `frozen_mp2_c6_molecular`).
//!
//! Sweep (ignored; one machine-readable `ROW` line per point):
//!   SWEEP_MOLS=h2o,n2 SWEEP_BASIS=aug-cc-pvdz SWEEP_C6=1 \
//!   SWEEP_OMEGAS=0,0.1,0.42 OPENBLAS_NUM_THREADS=1 cargo test --release \
//!   -p ferric-mp2 --test attenuation_rerun attn_sweep -- --ignored --nocapture
//! `ω = 0` means `Operator::coulomb()`. The auxiliary basis is the basis's
//! RI fit (`cc-pvdz-ri`, `aug-cc-pvdz-rifit`, ...). With `SWEEP_C6=1` each row
//! comes from `frozen_mp2_c6_molecular` on the 12-point Casimir-Polder grid
//! (u0 = 0.6, the grid of `cpks_frozen_mp2_c6_sweep`); otherwise from
//! `mp2_polarizability_analytic` (full tensor).
use ferric_core::basis;
use ferric_core::external_potential::ExternalPotential;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::cpks_polar::{frozen_mp2_c6_molecular, mp2_polarizability_analytic};
use ferric_mp2::rimp2::{ri_mp2, RiMp2Config};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

fn geometry(label: &str) -> &'static str {
    match label {
        "h2o" => "3\nh2o\nO 0 0 0.117790\nH 0 0.755453 -0.471161\nH 0 -0.755453 -0.471161\n",
        "n2" => "2\nn2\nN 0 0 0.0\nN 0 0 1.0977\n",
        "nh3" => {
            "4\nnh3\nN 0 0 0.0\nH 0 0.9377 -0.3816\nH 0.8120 -0.4689 -0.3816\n\
             H -0.8120 -0.4689 -0.3816\n"
        }
        // NH3 in a generic orientation (the PySCF FF reference geometry).
        "nh3g" => {
            "4\nnh3\nN 0.001600 -0.055708 0.103215\nH -0.560130 0.790564 0.124323\n\
             H 0.928089 0.227984 -0.202386\nH -0.379098 -0.628661 -0.644480\n"
        }
        "co2" => "3\nco2\nC 0 0 0.0\nO 0 0 1.1621\nO 0 0 -1.1621\n",
        "ch4" => {
            "5\nch4\nC 0 0 0\nH 0.6276 0.6276 0.6276\nH -0.6276 -0.6276 0.6276\n\
             H -0.6276 0.6276 -0.6276\nH 0.6276 -0.6276 -0.6276\n"
        }
        other => panic!("unknown molecule {other}"),
    }
}

fn aux_for(obs: &str) -> &'static str {
    match obs {
        "sto-3g" | "cc-pvdz" => "cc-pvdz-ri",
        "aug-cc-pvdz" => "aug-cc-pvdz-rifit",
        other => panic!("no aux mapping for {other}"),
    }
}

fn op_for(w: f64) -> Operator {
    if w == 0.0 {
        Operator::coulomb()
    } else {
        Operator::erfc(w)
    }
}

/// 12-point Gauss-Legendre mapped to [0,∞) by ω = u0(1+x)/(1−x) (same grid as
/// `cpks_polar.rs::cp_grid`).
fn cp_grid(u0: f64) -> (Vec<f64>, Vec<f64>) {
    let x = [
        -0.9815606342467192,
        -0.9041172563704749,
        -0.7699026741943047,
        -0.5873179542866175,
        -0.3678314989981802,
        -0.1252334085114689,
        0.1252334085114689,
        0.3678314989981802,
        0.5873179542866175,
        0.7699026741943047,
        0.9041172563704749,
        0.9815606342467192,
    ];
    let w = [
        0.0471753363865118,
        0.1069393259953184,
        0.1600783285433462,
        0.2031674267230659,
        0.2334925365383548,
        0.2491470458134028,
        0.2491470458134028,
        0.2334925365383548,
        0.2031674267230659,
        0.1600783285433462,
        0.1069393259953184,
        0.0471753363865118,
    ];
    let freqs = x.iter().map(|&xi| u0 * (1.0 + xi) / (1.0 - xi)).collect();
    let wts = x
        .iter()
        .zip(w.iter())
        .map(|(&xi, &wi)| wi * 2.0 * u0 / (1.0 - xi).powi(2))
        .collect();
    (freqs, wts)
}

struct Setup {
    mol: Molecule,
    obs: PreparedBasis,
    dfbs: PreparedBasis,
    bounds: SchwarzBounds,
    ctx: ParallelContext,
    rhf: ferric_scf::ScfResult,
}

fn setup(label: &str, obs_name: &str) -> Setup {
    let mol = Molecule::parse_xyz(geometry(label), 0, 1).unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled(obs_name).unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled(aux_for(obs_name)).unwrap()).unwrap();
    let ctx = ParallelContext::default();
    let cb = Operator::coulomb();
    let bounds = SchwarzBounds::compute(cb, &obs).unwrap();
    let cfg = RhfConfig {
        energy_conv: 1e-12,
        density_conv: 1e-10,
        ..Default::default()
    };
    let rhf = solve_rhf(&ctx, &mol, &obs, cb, &bounds, &cfg).unwrap();
    Setup {
        mol,
        obs,
        dfbs,
        bounds,
        ctx,
        rhf,
    }
}

fn mp2_cfg() -> RiMp2Config {
    RiMp2Config {
        frozen_core: 0,
        memory_budget_bytes: None,
        ..Default::default()
    }
}

/// Exactness anchor: erfc(ωr)/r → 1/r as ω → 0, so the attenuated α and
/// frozen C6 must reproduce the Coulomb ones. ω = 0 passed as an erfc
/// operator must be the Coulomb number to round-off; a small ω differs only
/// at O(ω³) (the O(ω) constant shift −2ω/√π integrates to zero against
/// orthogonal occ-vir pairs).
#[test]
fn attenuated_alpha_and_c6_match_coulomb_in_the_omega_zero_limit() {
    let s = setup("h2o", "sto-3g");
    let cfg = mp2_cfg();
    let (freqs, weights) = cp_grid(0.6);
    let run = |op: Operator| {
        let a = mp2_polarizability_analytic(
            &s.ctx, &s.mol, &s.obs, &s.dfbs, op, &s.bounds, &s.rhf, &cfg,
        )
        .unwrap();
        let (c6, _, _, _) = frozen_mp2_c6_molecular(
            &s.ctx, &s.mol, &s.obs, &s.dfbs, op, &s.bounds, &s.rhf, &cfg, &freqs, &weights,
        )
        .unwrap();
        (a.tensor, c6)
    };
    let (t_cb, c6_cb) = run(Operator::coulomb());
    for (w, bar) in [(0.0, 1e-12), (1e-3, 1e-8)] {
        let (t, c6) = run(Operator::erfc(w));
        let mut dmax: f64 = 0.0;
        for x in 0..3 {
            for y in 0..3 {
                dmax = dmax.max((t[x][y] - t_cb[x][y]).abs());
            }
        }
        let dc6 = (c6 - c6_cb).abs();
        eprintln!("erfc(ω={w:e}) vs Coulomb: max|Δα| = {dmax:.3e}, |ΔC6| = {dc6:.3e}");
        assert!(dmax < bar, "ω={w}: max|Δα| = {dmax:e} ≥ {bar:e}");
        assert!(dc6 < bar * 10.0, "ω={w}: |ΔC6| = {dc6:e}");
    }
}

/// Second-molecule check of the erfc construction against the energy-only
/// finite field `α_dd = −∂²E/∂F_d²` of `E_RHF(F) + E_MP2[erfc](F)` (5-point,
/// h = 2e-3), each field point a re-converged Coulomb RHF. Generic-orientation
/// NH3/STO-3G at ω = 0.42 and 1.0 Bohr⁻¹. Same bar as
/// `attenuated_relaxed_alpha_matches_energy_finite_field`.
#[test]
#[ignore = "tier slow (>=60s): CI runs it when an affected crate changes (scripts/ci_test_tiers.py)"]
fn attenuated_alpha_finite_field_nh3() {
    let s = setup("nh3g", "sto-3g");
    let cfg = mp2_cfg();
    let cb = Operator::coulomb();
    let scf_cfg = |field: Option<[f64; 3]>| RhfConfig {
        energy_conv: 1e-13,
        density_conv: 1e-11,
        external_potential: field.map(|f| ExternalPotential {
            field: Some(f),
            ..Default::default()
        }),
        ..Default::default()
    };
    for w in [0.42, 1.0] {
        let op = Operator::erfc(w);
        let energy = |field: Option<[f64; 3]>| -> f64 {
            let rhf = solve_rhf(&s.ctx, &s.mol, &s.obs, cb, &s.bounds, &scf_cfg(field)).unwrap();
            ri_mp2(&s.mol, &s.obs, &s.dfbs, op, &rhf, &cfg)
                .unwrap()
                .total_energy
        };
        let rhf = solve_rhf(&s.ctx, &s.mol, &s.obs, cb, &s.bounds, &scf_cfg(None)).unwrap();
        let a =
            mp2_polarizability_analytic(&s.ctx, &s.mol, &s.obs, &s.dfbs, op, &s.bounds, &rhf, &cfg)
                .unwrap();
        let h = 2e-3;
        let e0 = energy(None);
        for d in 0..3 {
            let at = |sg: f64| {
                let mut f = [0.0; 3];
                f[d] = sg * h;
                energy(Some(f))
            };
            let fd = -(-at(2.0) + 16.0 * at(1.0) - 30.0 * e0 + 16.0 * at(-1.0) - at(-2.0))
                / (12.0 * h * h);
            let rel = (a.tensor[d][d] - fd).abs() / fd.abs().max(a.iso);
            eprintln!(
                "NH3 ω={w} α[{d}{d}] analytic {:.7} finite field {fd:.7} rel {rel:.1e}",
                a.tensor[d][d]
            );
            assert!(
                rel < 1e-5,
                "ω={w} α[{d}{d}] = {} vs FF {fd}",
                a.tensor[d][d]
            );
        }
    }
}

#[test]
#[ignore = "measurement: attenuation sweep driver (SWEEP_* env vars), not a regression test; see wiki/notes/attenuated-mp2-alpha-c6-rerun-2026-11-05.md"]
fn attn_sweep() {
    let env = |k: &str, d: &str| std::env::var(k).unwrap_or_else(|_| d.to_string());
    let mols = env("SWEEP_MOLS", "h2o");
    let obs_name = env("SWEEP_BASIS", "cc-pvdz");
    let do_c6 = env("SWEEP_C6", "0") == "1";
    let omegas: Vec<f64> = env("SWEEP_OMEGAS", "0,0.1,0.2,0.3,0.42,0.6,0.8,1.0")
        .split(',')
        .map(|x| x.trim().parse().unwrap())
        .collect();
    let cfg = mp2_cfg();
    let (freqs, weights) = cp_grid(0.6);
    eprintln!("ROW\tmol\tbasis\tomega\ta_iso\ta_xx\ta_yy\ta_zz\tc6\ta_hf0");
    for label in mols.split(',') {
        let s = setup(label, &obs_name);
        for &w in &omegas {
            let op = op_for(w);
            if do_c6 {
                let (c6, _, a_mp2, a_hf) = frozen_mp2_c6_molecular(
                    &s.ctx, &s.mol, &s.obs, &s.dfbs, op, &s.bounds, &s.rhf, &cfg, &freqs, &weights,
                )
                .unwrap();
                eprintln!("ROW\t{label}\t{obs_name}\t{w}\t{a_mp2:.6}\t-\t-\t-\t{c6:.4}\t{a_hf:.6}");
            } else {
                let a = mp2_polarizability_analytic(
                    &s.ctx, &s.mol, &s.obs, &s.dfbs, op, &s.bounds, &s.rhf, &cfg,
                )
                .unwrap();
                eprintln!(
                    "ROW\t{label}\t{obs_name}\t{w}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t-\t-",
                    a.iso, a.tensor[0][0], a.tensor[1][1], a.tensor[2][2]
                );
            }
        }
    }
}
