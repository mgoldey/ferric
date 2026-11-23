//! rs-mp2-rpa charges the AO tensor at the tier the build takes.
//!
//! # What is claimed
//!
//! 1. `scan_charge` plans the tier `ThreeIndexSource::build` takes at the
//!    budget the driver hands it, and the charge at that tier fits the budget.
//!    At budgets where the in-core tensor does not fit, the packed and spill
//!    tiers are reachable and admitted.
//! 2. With a budget above need, the driver takes the in-core tier and its
//!    energies are bitwise those of the pre-change path (the public pieces
//!    called with the full budget).
//! 3. The packed and spilled runs agree with the in-core run within a bar
//!    derived from the reduction order of the `b_ov` stream, for both
//!    formulations and two ω points.
//!
//! # The bar (claim 3)
//!
//! Only `b = F·(Q|ia)` depends on the tier. The stream accumulates
//! `b[P,ia] = Σ_Q F[P,Q] raw[Q,ia]` in chunks that start at each source
//! block, so the tiers sum the same `naux` terms in different groupings.
//! Two orders of one sum of n terms differ by at most `2γ_n Σ|x|`
//! (`γ_n = nu/(1-nu)`, u = 2⁻⁵³), so elementwise
//!
//! ```text
//!   |δb| ≤ 2γ_naux · |F|·|raw|
//! ```
//!
//! which `b_ov_differs_by_at_most_the_reduction_order_bound` checks. That
//! majorant is rigorous but loose for the erf metric, whose `F` has large
//! entries that cancel. The energy bar therefore propagates the MEASURED
//! `δb` (bounded above) to first order, with `V = |b|`, `U = |δb|`:
//!
//! ```text
//!   g = bᵀb:  |g| ≤ G = VᵀV,   |δg| ≤ Gδ = UᵀV + VᵀU
//!   T1 = Σ G·Gδ/|Δ|,  T2 = Σ G'·Gδ/|Δ|,  T3 = Σ G·Gδ'/|Δ|   (' = i↔j swap)
//!   |δ(2E_OS)| ≤ 4 T1,     |δE_MP2| ≤ 4 T1 + T2 + T3      (E_MP2 = Σ g(2g−g')/Δ)
//! ```
//!
//! dRPA: `E = (1/2π)∫₀^∞ tr[ln(1+Π) − Π] dω` with `Π ⪰ 0`, so
//! `|δE| ≤ (1/2π)∫ ‖Π‖_F ‖δΠ‖_F`. `Π` and `δΠ` are bounded elementwise by
//! the same sums built from `V` and from `U, V`; with
//! `∫₀^∞ w_ia w_jb dω = π/(2(Δ_ia+Δ_jb))` and Cauchy-Schwarz in ω,
//!
//! ```text
//!   |δE_dRPA| ≤ 4 √(S·R),   S = Σ G²/|Δ|,   R = Σ (UᵀU∘VᵀV + UᵀV∘(UᵀV)ᵀ)/|Δ|
//! ```
//!
//! The code integrates on a quadrature, so this term carries a factor 2.
//! Each run also rounds its own evaluation differently; their difference is
//! bounded by twice the forward error at the probabilistic rate `√N u`
//! (Higham & Mary 2019, `N = (nocc·nvir)²` terms) on the majorants
//! `|E_MP2| ≤ 3S`, `|2E_OS| ≤ 2S`, `|E_dRPA| ≤ 2S`. The bar of a formulation
//! is the sum of these over its terms (B: `E_MP2[C] + E_dRPA[lr] − 2E_OS[lr]`;
//! T: `E_MP2[C] + (E_dRPA − 2E_OS)[C] − (E_dRPA − 2E_OS)[sr]`).
//!
//! # Mutations (see the commit report)
//!
//! * `scan_charge` charging the full in-core tensor whatever the tier: claim 1
//!   fails on its `total <= budget` assertion.
//! * the driver handing the full budget to the build (`ri_cfg`): the pool
//!   test in `rs_mp2_rpa_spill_tier_pool.rs` fails on its admission assertion.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_integrals::three_index_source::jk_storage::Tier;
use ferric_integrals::three_index_source::ThreeIndexSource;
use ferric_mp2::rimp2::{
    compute_rpa_intermediates, spin_components_from_b_ov, stream_dressed_mo_band, RiMp2Config,
};
use ferric_rpa::budget::ScanCharge;
use ferric_rpa::rs_mp2_rpa::{
    rs_mp2_lr_rpa_scan, scan_charge_for, RsMp2RpaConfig, RsMp2RpaFormulation,
};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::ScfResult;
use ndarray::Array2;

const ANG_INV: f64 = ferric_mp2::attenuated::BOHR_INV_PER_ANG_INV;
const BOTH: [RsMp2RpaFormulation; 2] = [
    RsMp2RpaFormulation::DeltaLr,
    RsMp2RpaFormulation::CoupledRings,
];
/// Fixed so the per-worker terms do not read the machine.
const WORKERS: usize = 2;
const AMPLE: usize = 1 << 40;

struct Sys {
    mol: Molecule,
    obs: PreparedBasis,
    dfbs: PreparedBasis,
    rhf: ScfResult,
}

fn setup() -> Sys {
    let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../testdata/molecules/s22/water_dimer.xyz");
    let mol = Molecule::parse_xyz(&std::fs::read_to_string(p).unwrap(), 0, 1).unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled("aug-cc-pvdz").unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("aug-cc-pvdz-rifit").unwrap()).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &obs).unwrap();
    let rhf = solve_rhf(
        &ParallelContext::default(),
        &mol,
        &obs,
        Operator::coulomb(),
        &bounds,
        &RhfConfig::default()
            .with_max_iter(400)
            .with_energy_conv(1e-11)
            .with_density_conv(1e-10),
    )
    .unwrap();
    assert!(rhf.converged);
    Sys {
        mol,
        obs,
        dfbs,
        rhf,
    }
}

fn pool() -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(WORKERS)
        .build()
        .unwrap()
}

fn cfg_at(budget: usize) -> RsMp2RpaConfig {
    let mut cfg = RsMp2RpaConfig {
        frozen_core: 2,
        ..Default::default()
    };
    cfg.drpa.memory_budget_bytes = Some(budget);
    cfg
}

fn charge_at(sys: &Sys, budget: usize) -> ScanCharge {
    scan_charge_for(&sys.mol, &sys.obs, &sys.dfbs, &cfg_at(budget))
        .unwrap()
        .1
}

/// The largest budget below the in-core need whose planned tier is `tier`,
/// scanning down in 2% steps. `None` when no such budget exists. Selected on
/// the tier alone, so whether the charge fits is left to the assertions.
fn admitted_budget_for(sys: &Sys, tier: Tier) -> Option<usize> {
    let full = charge_at(sys, AMPLE);
    assert_eq!(full.ao_tier, Tier::InCore);
    let mut b = full.total;
    while b > full.persistent {
        if charge_at(sys, b).ao_tier == tier {
            return Some(b);
        }
        b = b * 49 / 50;
    }
    None
}

fn tier_of(src: &ThreeIndexSource) -> Tier {
    if src.is_incore() {
        Tier::InCore
    } else if src.packed_flat().is_some() {
        Tier::InCorePacked
    } else {
        assert!(!src.is_resident());
        Tier::Spill
    }
}

/// Claim 1.
#[test]
fn the_charge_follows_the_tier_the_build_takes() {
    let sys = setup();
    pool().install(|| {
        let full = charge_at(&sys, AMPLE);
        let in_core_tensor = sys.dfbs.nbasis() * sys.obs.nbasis().pow(2) * 8;
        assert_eq!(full.ao_resident, in_core_tensor);
        for tier in [Tier::InCore, Tier::InCorePacked, Tier::Spill] {
            let budget = if tier == Tier::InCore {
                full.total
            } else {
                admitted_budget_for(&sys, tier)
                    .unwrap_or_else(|| panic!("no admitted budget reaches the {tier:?} tier"))
            };
            let c = charge_at(&sys, budget);
            assert_eq!(c.ao_tier, tier, "planned tier at {budget} B");
            assert!(
                c.total <= budget,
                "{tier:?}: charge {} B exceeds the budget {budget} B",
                c.total
            );
            if tier != Tier::InCore {
                assert!(c.ao_resident < in_core_tensor, "{tier:?}: {c:?}");
            }
            // The build at the budget the driver hands it takes that tier.
            let src =
                ThreeIndexSource::build(Operator::coulomb(), &sys.obs, &sys.dfbs, c.ao_budget)
                    .unwrap();
            assert_eq!(
                tier_of(&src),
                tier,
                "built tier at ao_budget {}",
                c.ao_budget
            );
            // And the driver admits the run.
            let r = rs_mp2_lr_rpa_scan(
                &sys.mol,
                &sys.obs,
                &sys.dfbs,
                &sys.rhf,
                &cfg_at(budget),
                &[0.42 * ANG_INV],
                &BOTH,
            );
            assert!(r.is_ok(), "{tier:?} at {budget} B refused: {:?}", r.err());
        }
    });
}

/// Claim 2: the trivial limit. The pre-change driver built each operator's
/// intermediates at the full budget; at an ample budget that is the in-core
/// tier, and so is the new driver's.
#[test]
fn ample_budget_is_bitwise_the_in_core_path() {
    let sys = setup();
    pool().install(|| {
        let w = 0.42 * ANG_INV;
        let cfg = cfg_at(AMPLE);
        assert_eq!(charge_at(&sys, AMPLE).ao_tier, Tier::InCore);
        let got =
            rs_mp2_lr_rpa_scan(&sys.mol, &sys.obs, &sys.dfbs, &sys.rhf, &cfg, &[w], &BOTH).unwrap();
        let ri = RiMp2Config {
            frozen_core: cfg.frozen_core,
            memory_budget_bytes: Some(AMPLE),
            ..Default::default()
        };
        let mut drpa_cfg = cfg.drpa.clone();
        drpa_cfg.frozen_core = cfg.frozen_core;
        let eps = sys.rhf.eps_r();
        let parts = |op: Operator| {
            let it = compute_rpa_intermediates(&sys.mol, &sys.obs, &sys.dfbs, op, &sys.rhf, &ri)
                .unwrap();
            let sc = spin_components_from_b_ov(
                &it.b_ov,
                eps,
                it.nocc,
                it.nvir,
                it.first_occ,
                it.nocc_total,
            );
            let e = ferric_rpa::run_pdep_rpa_from_intermediates(
                &it, &sys.mol, &sys.obs, &sys.dfbs, op, &sys.rhf, &drpa_cfg,
            )
            .unwrap()
            .e_rpa;
            (sc, e)
        };
        let (sc_c, rpa_c) = parts(Operator::coulomb());
        let (sc_lr, rpa_lr) = parts(Operator::erf(w));
        let (sc_sr, rpa_sr) = parts(Operator::erfc(w));
        let e_b = sc_c.e_total + rpa_lr - 2.0 * sc_lr.e_os;
        let e_t = sc_c.e_total + (rpa_c - 2.0 * sc_c.e_os) - (rpa_sr - 2.0 * sc_sr.e_os);
        assert_eq!(got[0].e_corr.to_bits(), e_b.to_bits(), "delta-lr");
        assert_eq!(got[1].e_corr.to_bits(), e_t.to_bits(), "coupled-rings");
        assert_eq!(got[0].e_sr_mp2.to_bits(), sc_sr.e_total.to_bits());
        assert_eq!(got[0].e_lr_mp2.to_bits(), sc_lr.e_total.to_bits());
    });
}

/// `|F|·|raw|`, the elementwise majorant of `b` for `op`.
fn abs_majorant(sys: &Sys, op: Operator) -> Array2<f64> {
    let it = intermediates(sys, op, AMPLE);
    let c = sys.rhf.mos_r();
    let c_occ = c
        .slice(ndarray::s![.., it.first_occ..it.first_occ + it.nocc])
        .to_owned();
    let c_vir = c.slice(ndarray::s![.., it.nocc_total..]).to_owned();
    let naux = sys.dfbs.nbasis();
    let mut src = ThreeIndexSource::build(op, &sys.obs, &sys.dfbs, AMPLE).unwrap();
    let raw = stream_dressed_mo_band(&mut src, &Array2::eye(naux), &c_occ, &c_vir, None).unwrap();
    it.v_inv_sqrt.mapv(f64::abs).dot(&raw.mapv(f64::abs))
}

fn intermediates(sys: &Sys, op: Operator, budget: usize) -> ferric_mp2::rimp2::RpaIntermediates {
    let ri = RiMp2Config {
        frozen_core: 2,
        memory_budget_bytes: Some(budget),
        ..Default::default()
    };
    compute_rpa_intermediates(&sys.mol, &sys.obs, &sys.dfbs, op, &sys.rhf, &ri).unwrap()
}

/// The first-order sensitivity sums of the bar, for one operator, from the
/// in-core `b` and the measured `δb` of a tier.
struct Sens {
    s: f64,
    t1: f64,
    t2: f64,
    t3: f64,
    r: f64,
}

impl Sens {
    /// Bound on `|δ(2E_OS)|`, and on the evaluation majorant `|2E_OS|`.
    fn os2(&self) -> (f64, f64) {
        (4.0 * self.t1, 2.0 * self.s)
    }
    fn mp2(&self) -> (f64, f64) {
        (4.0 * self.t1 + self.t2 + self.t3, 3.0 * self.s)
    }
    /// The factor 2 covers the quadrature (module doc).
    fn drpa(&self) -> (f64, f64) {
        (2.0 * 4.0 * (self.s * self.r).sqrt(), 2.0 * self.s)
    }
}

fn sens(sys: &Sys, op: Operator, tier_ao_budget: usize) -> Sens {
    let it = intermediates(sys, op, AMPLE);
    let other = intermediates(sys, op, tier_ao_budget).b_ov;
    let v = it.b_ov.mapv(f64::abs);
    let u = (&other - &it.b_ov).mapv(f64::abs);
    let g = v.t().dot(&v);
    let uv = u.t().dot(&v);
    let gd = &uv + &uv.t();
    let uu = u.t().dot(&u);
    let eps = sys.rhf.eps_r();
    let (no, nv) = (it.nocc, it.nvir);
    let d = |i: usize, a: usize| eps[it.nocc_total + a] - eps[it.first_occ + i];
    let (mut out_s, mut t1, mut t2, mut t3, mut r) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for i in 0..no {
        for a in 0..nv {
            let ia = i * nv + a;
            for j in 0..no {
                for bb in 0..nv {
                    let jb = j * nv + bb;
                    // The exchange partner (ib|ja).
                    let (ib, ja) = (i * nv + bb, j * nv + a);
                    let den = d(i, a) + d(j, bb);
                    out_s += g[[ia, jb]] * g[[ia, jb]] / den;
                    t1 += g[[ia, jb]] * gd[[ia, jb]] / den;
                    t2 += g[[ib, ja]] * gd[[ia, jb]] / den;
                    t3 += g[[ia, jb]] * gd[[ib, ja]] / den;
                    r += (uu[[ia, jb]] * g[[ia, jb]] + uv[[ia, jb]] * uv[[jb, ia]]) / den;
                }
            }
        }
    }
    Sens {
        s: out_s,
        t1,
        t2,
        t3,
        r,
    }
}

fn gamma(n: usize) -> f64 {
    let nu = n as f64 * f64::EPSILON / 2.0;
    nu / (1.0 - nu)
}

/// The premise of claim 3: elementwise, the tiers' `b_ov` differ by at most
/// `ε·B`.
#[test]
fn b_ov_differs_by_at_most_the_reduction_order_bound() {
    let sys = setup();
    pool().install(|| {
        let eps_b = 2.0 * gamma(sys.dfbs.nbasis());
        let w = 0.42 * ANG_INV;
        let mut differed = false;
        for tier in [Tier::InCorePacked, Tier::Spill] {
            let budget = admitted_budget_for(&sys, tier).unwrap();
            let ao_budget = charge_at(&sys, budget).ao_budget;
            for op in [Operator::coulomb(), Operator::erf(w), Operator::erfc(w)] {
                let m_b = abs_majorant(&sys, op);
                let at = |b: usize| {
                    let ri = RiMp2Config {
                        frozen_core: 2,
                        memory_budget_bytes: Some(b),
                        ..Default::default()
                    };
                    compute_rpa_intermediates(&sys.mol, &sys.obs, &sys.dfbs, op, &sys.rhf, &ri)
                        .unwrap()
                        .b_ov
                };
                let (a, b) = (at(AMPLE), at(ao_budget));
                let mut worst: f64 = 0.0;
                for ((x, y), bound) in a.iter().zip(b.iter()).zip(m_b.iter()) {
                    let d = (x - y).abs();
                    differed |= d > 0.0;
                    if *bound > 0.0 {
                        worst = worst.max(d / (eps_b * bound));
                    } else {
                        assert_eq!(d, 0.0);
                    }
                }
                eprintln!("{tier:?} {op:?}: max |db|/(eps B) = {worst:.3e}");
                assert!(
                    worst <= 1.0,
                    "{tier:?} {op:?}: |db| exceeds eps*B by {worst}"
                );
            }
        }
        // Reachability: a tier whose stream equals the in-core one bit for bit
        // would make claim 3 vacuous.
        assert!(differed, "no tier changed b_ov: the comparison is vacuous");
    });
}

/// Claim 3.
#[test]
fn packed_and_spilled_match_in_core_within_the_reduction_order_bar() {
    let sys = setup();
    pool().install(|| {
        let omegas = [0.42 * ANG_INV, 0.673 * ANG_INV];
        let reference = rs_mp2_lr_rpa_scan(
            &sys.mol,
            &sys.obs,
            &sys.dfbs,
            &sys.rhf,
            &cfg_at(AMPLE),
            &omegas,
            &BOTH,
        )
        .unwrap();
        let nov = intermediates(&sys, Operator::coulomb(), AMPLE).b_ov.ncols() as f64;
        // 2·√N·u with N = nov².
        let eval = 2.0 * nov * f64::EPSILON / 2.0;
        let bar_of =
            |terms: &[(f64, f64)]| -> f64 { terms.iter().map(|(p, m)| p + eval * m).sum() };
        for tier in [Tier::InCorePacked, Tier::Spill] {
            let budget = admitted_budget_for(&sys, tier).unwrap();
            let c = charge_at(&sys, budget);
            assert_eq!(c.ao_tier, tier);
            let got = rs_mp2_lr_rpa_scan(
                &sys.mol,
                &sys.obs,
                &sys.dfbs,
                &sys.rhf,
                &cfg_at(budget),
                &omegas,
                &BOTH,
            )
            .unwrap();
            let s_c = sens(&sys, Operator::coulomb(), c.ao_budget);
            for (k, &w) in omegas.iter().enumerate() {
                let s_lr = sens(&sys, Operator::erf(w), c.ao_budget);
                let s_sr = sens(&sys, Operator::erfc(w), c.ao_budget);
                for (j, f) in BOTH.iter().enumerate() {
                    let i = k * BOTH.len() + j;
                    let bar_corr = match f {
                        RsMp2RpaFormulation::DeltaLr => {
                            bar_of(&[s_c.mp2(), s_lr.drpa(), s_lr.os2()])
                        }
                        RsMp2RpaFormulation::CoupledRings => {
                            bar_of(&[s_c.mp2(), s_c.drpa(), s_c.os2(), s_sr.drpa(), s_sr.os2()])
                        }
                    };
                    let checks = [
                        ("e_corr", reference[i].e_corr, got[i].e_corr, bar_corr),
                        (
                            "e_mp2_full",
                            reference[i].e_mp2_full,
                            got[i].e_mp2_full,
                            bar_of(&[s_c.mp2()]),
                        ),
                        (
                            "e_sr_mp2",
                            reference[i].e_sr_mp2,
                            got[i].e_sr_mp2,
                            bar_of(&[s_sr.mp2()]),
                        ),
                        (
                            "e_lr_mp2",
                            reference[i].e_lr_mp2,
                            got[i].e_lr_mp2,
                            bar_of(&[s_lr.mp2()]),
                        ),
                    ];
                    for (name, a, b, bar) in checks {
                        let d = (a - b).abs();
                        eprintln!("{tier:?} w={w:.4} {f:?} {name}: |dE| = {d:.3e}, bar {bar:.3e}");
                        assert!(d <= bar, "{tier:?} w={w} {f:?} {name}: {d:e} > {bar:e}");
                    }
                }
            }
        }
    });
}
