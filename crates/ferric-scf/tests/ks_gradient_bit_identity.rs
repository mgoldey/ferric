//! Bit-identity guard for the KS gradient entry points in `ks_gradient.rs`.
//!
//! Every closed-shell (RKS), UKS and ROKS public entry point is evaluated on a
//! tiny system and the f64 bit patterns of the result are compared against
//! values recorded from the implementation before the spin-generic refactor
//! (ARCHITECTURE.md roadmap item 14). The cases cover each two-electron
//! branch: density-fitted route (default RI-J / RSH fitted K), exact
//! four-centre scaled-K (B3LYP with fitting off), exact range-separated
//! erfc/erf (wB97X with the recorded route removed), COSX (RIJCOSX for RKS,
//! four-centre J for UKS / ROKS), plus the polarizable wrappers, the
//! `_on_grids` grid plumbing, meta-GGA and VV10 XC dispatch, and the
//! standalone `xc_gradient_*` helpers.
//!
//! The recorded bits come from one machine. Another CPU or BLAS build
//! reorders floating-point sums and moves the last bits (~1e-12 relative),
//! so by default each component must match to `TOL` (1e-10 Hartree/Bohr):
//! far below any change a refactor that drops, doubles or reorders a term
//! would make, and far above cross-machine rounding. Set
//! `FERRIC_GRADIENT_BIT_EXACT=1` to demand identical bits instead, which is
//! the check to run on the machine that recorded the table when refactoring.
//!
//! Re-record (only when a numerical change is intended):
//!   OPENBLAS_NUM_THREADS=1 cargo test -p ferric-scf --test ks_gradient_bit_identity -- --nocapture
//! with `EXPECTED` emptied, and paste the printed table.

use ferric_core::basis::{bundled, BasisSet};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_dft::grid::AtomicGridConfig;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::cosx_k::{CosxConfig, CosxHalfTransform};
use ferric_scf::ks_gradient::{
    ks_gradient_closed, ks_gradient_closed_for_density, ks_gradient_closed_with_exchange,
    ks_gradient_closed_with_exchange_on_grids, ks_gradient_closed_with_polarizable,
    ks_gradient_closed_with_polarizable_on_grids, ks_gradient_roks, ks_gradient_roks_for_density,
    ks_gradient_roks_with_exchange, ks_gradient_roks_with_exchange_on_grids, ks_gradient_uks,
    ks_gradient_uks_for_density, ks_gradient_uks_with_exchange,
    ks_gradient_uks_with_exchange_on_grids, ks_gradient_uks_with_polarizable, xc_gradient_closed,
    xc_gradient_open, KsGradGrids,
};
use ferric_scf::polarizable::{PolarizableSite, PolarizableSites};
use ferric_scf::result::ScfResult;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::rohf::solve_rohf;
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::solve_uhf;
use ndarray::Array2;

const WATER: &str = "3\nwater\nO 0.00 0.02 0.11\nH 0.03 0.76 -0.47\nH -0.02 -0.75 -0.45\n";
const OH: &str = "2\nOH\nO 0.0 0.01 0.0\nH 0.02 0.0 0.98\n";

struct Sys {
    mol: Molecule,
    bs: BasisSet,
    prep: PreparedBasis,
    bounds: SchwarzBounds,
}

fn sys(xyz: &str, mult: usize) -> Sys {
    let mol = Molecule::parse_xyz(xyz, 0, mult).unwrap();
    let bs = bundled("sto-3g").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
    Sys {
        mol,
        bs,
        prep,
        bounds,
    }
}

fn cosx() -> CosxConfig {
    CosxConfig {
        overlap_fit: false,
        screen_thresh: None,
        half_transform: CosxHalfTransform::Dense,
        ..CosxConfig::flat_reference()
    }
}

/// `exact` turns density fitting off (four-centre J and K).
fn cfg(xc: &str, exact: bool, cosx_cfg: Option<CosxConfig>) -> RhfConfig {
    let off = exact.then(String::new);
    RhfConfig {
        xc: Some(xc.into()),
        k_builder: cosx_cfg.as_ref().map(|_| "cosx".to_string()),
        cosx: cosx_cfg.unwrap_or_default(),
        df_j_aux: off.clone(),
        df_k_aux: off,
        energy_conv: 1e-8,
        density_conv: 1e-6,
        max_iter: 400,
        level_shift: 0.2,
        mom_after_iter: 5,
        ..Default::default()
    }
}

#[derive(Clone, Copy)]
enum Ref {
    Rks,
    Uks,
    Roks,
}

fn scf(s: &Sys, r: Ref, c: &RhfConfig) -> ScfResult {
    let ctx = ParallelContext::default();
    let op = Operator::coulomb();
    let res = match r {
        Ref::Rks => solve_rhf(&ctx, &s.mol, &s.prep, op, &s.bounds, c),
        Ref::Uks => solve_uhf(&ctx, &s.mol, &s.prep, &s.bounds, c),
        Ref::Roks => solve_rohf(&ctx, &s.mol, &s.prep, op, &s.bounds, c),
    }
    .unwrap();
    assert!(res.converged, "SCF did not converge");
    res
}

fn without_route(r: &ScfResult) -> ScfResult {
    let mut r = r.clone();
    r.df_jk = None;
    r
}

fn coarse_grids() -> KsGradGrids {
    KsGradGrids {
        main: AtomicGridConfig {
            n_radial: 40,
            n_angular: 86,
            prune: None,
            weight_cut: None,
        },
        nlc: AtomicGridConfig {
            n_radial: 30,
            n_angular: 50,
            prune: None,
            weight_cut: None,
        },
    }
}

fn sites() -> PolarizableSites {
    PolarizableSites {
        sites: vec![
            PolarizableSite {
                x: 4.0,
                y: -1.0,
                z: 2.0,
                alpha: 8.0,
            },
            PolarizableSite {
                x: -3.5,
                y: 2.0,
                z: -2.5,
                alpha: 6.0,
            },
        ],
        thole_a: Some(2.1304),
        exclusions: vec![],
        dipole_zeta: 1e4,
        max_sites_dense: 4000,
    }
}

fn dipoles() -> Array2<f64> {
    Array2::from_shape_vec((2, 3), vec![0.03, -0.01, 0.02, -0.02, 0.015, 0.01]).unwrap()
}

/// A density perturbation (symmetric) for the `_for_density` entry points.
fn bump(d: &Array2<f64>) -> Array2<f64> {
    let n = d.nrows();
    Array2::from_shape_fn((n, n), |(i, j)| {
        d[(i, j)] + 1e-3 * (((i + j) % 5) as f64 - 2.0)
    })
}

fn bits(g: &Array2<f64>) -> Vec<u64> {
    g.iter().map(|v| v.to_bits()).collect()
}

fn all_cases() -> Vec<(&'static str, Array2<f64>)> {
    let op = Operator::coulomb();
    let mut out: Vec<(&'static str, Array2<f64>)> = Vec::new();

    // ---- closed shell (RKS) on water ------------------------------------
    let w = sys(WATER, 1);
    let (m, p, b, bs) = (&w.mol, &w.prep, &w.bounds, &w.bs);
    let pbe = scf(&w, Ref::Rks, &cfg("pbe", false, None));
    out.push((
        "rks_pbe_ri",
        ks_gradient_closed(m, p, bs, op, b, "pbe", &pbe, None).unwrap(),
    ));
    out.push((
        "rks_pbe_coarse",
        ks_gradient_closed_with_exchange_on_grids(
            m,
            p,
            bs,
            op,
            b,
            "pbe",
            &pbe,
            None,
            None,
            &coarse_grids(),
        )
        .unwrap(),
    ));
    out.push((
        "rks_pbe_polar",
        ks_gradient_closed_with_polarizable(
            m,
            p,
            bs,
            op,
            b,
            "pbe",
            &pbe,
            None,
            Some(&sites()),
            Some(&dipoles()),
        )
        .unwrap(),
    ));
    out.push((
        "rks_pbe_polar_coarse",
        ks_gradient_closed_with_polarizable_on_grids(
            m,
            p,
            bs,
            op,
            b,
            "pbe",
            &pbe,
            None,
            Some(&sites()),
            Some(&dipoles()),
            &coarse_grids(),
        )
        .unwrap(),
    ));
    {
        let d = bump(pbe.density_r());
        let wd = bump(&ferric_scf::gradient::build_energy_weighted_density(
            &pbe, 5,
        ));
        out.push((
            "rks_pbe_for_density",
            ks_gradient_closed_for_density(
                m,
                p,
                bs,
                op,
                b,
                "pbe",
                &pbe,
                None,
                None,
                &d,
                &wd,
                &KsGradGrids::default(),
            )
            .unwrap(),
        ));
    }
    out.push((
        "rks_pbe_xc_only",
        xc_gradient_closed(m, p, bs, "pbe", pbe.density_r(), &KsGradGrids::default()).unwrap(),
    ));
    let b3 = scf(&w, Ref::Rks, &cfg("b3lyp", true, None));
    out.push((
        "rks_b3lyp_exact",
        ks_gradient_closed(m, p, bs, op, b, "b3lyp", &b3, None).unwrap(),
    ));
    let wb = scf(&w, Ref::Rks, &cfg("HYB_GGA_XC_WB97X", false, None));
    out.push((
        "rks_wb97x_routed",
        ks_gradient_closed(m, p, bs, op, b, "HYB_GGA_XC_WB97X", &wb, None).unwrap(),
    ));
    out.push((
        "rks_wb97x_exact_rsh",
        ks_gradient_closed(
            m,
            p,
            bs,
            op,
            b,
            "HYB_GGA_XC_WB97X",
            &without_route(&wb),
            None,
        )
        .unwrap(),
    ));
    let cx = cosx();
    let b3c = scf(&w, Ref::Rks, &cfg("b3lyp", false, Some(cx.clone())));
    out.push((
        "rks_b3lyp_rijcosx",
        ks_gradient_closed_with_exchange(m, p, bs, op, b, "b3lyp", &b3c, None, Some(&cx)).unwrap(),
    ));
    out.push((
        "rks_b3lyp_cosx_4c_j",
        ks_gradient_closed_with_exchange(
            m,
            p,
            bs,
            op,
            b,
            "b3lyp",
            &without_route(&b3c),
            None,
            Some(&cx),
        )
        .unwrap(),
    ));
    let scan = scf(&w, Ref::Rks, &cfg("scan", false, None));
    out.push((
        "rks_scan",
        ks_gradient_closed(m, p, bs, op, b, "scan", &scan, None).unwrap(),
    ));
    let lda = scf(&w, Ref::Rks, &cfg("lda", false, None));
    out.push((
        "rks_lda",
        ks_gradient_closed(m, p, bs, op, b, "lda", &lda, None).unwrap(),
    ));
    let wbv = scf(&w, Ref::Rks, &cfg("wB97X-V", false, None));
    out.push((
        "rks_wb97xv_coarse",
        ks_gradient_closed_with_exchange_on_grids(
            m,
            p,
            bs,
            op,
            b,
            "wB97X-V",
            &wbv,
            None,
            None,
            &coarse_grids(),
        )
        .unwrap(),
    ));

    // ---- open shell (UKS, ROKS) on OH -----------------------------------
    let o = sys(OH, 2);
    let (m, p, b, bs) = (&o.mol, &o.prep, &o.bounds, &o.bs);
    for (r, tag) in [(Ref::Uks, "uks"), (Ref::Roks, "roks")] {
        let plain = |xc: &str, res: &ScfResult| match r {
            Ref::Uks => ks_gradient_uks(m, p, bs, op, b, xc, res, None).unwrap(),
            _ => ks_gradient_roks(m, p, bs, op, b, xc, res, None).unwrap(),
        };
        let with_x = |xc: &str, res: &ScfResult, c: Option<&CosxConfig>| match r {
            Ref::Uks => ks_gradient_uks_with_exchange(m, p, bs, op, b, xc, res, None, c).unwrap(),
            _ => ks_gradient_roks_with_exchange(m, p, bs, op, b, xc, res, None, c).unwrap(),
        };
        let on_grids = |xc: &str, res: &ScfResult| match r {
            Ref::Uks => ks_gradient_uks_with_exchange_on_grids(
                m,
                p,
                bs,
                op,
                b,
                xc,
                res,
                None,
                None,
                &coarse_grids(),
            )
            .unwrap(),
            _ => ks_gradient_roks_with_exchange_on_grids(
                m,
                p,
                bs,
                op,
                b,
                xc,
                res,
                None,
                None,
                &coarse_grids(),
            )
            .unwrap(),
        };
        let name = |s: &str| -> &'static str { Box::leak(format!("{tag}_{s}").into_boxed_str()) };

        let pbe = scf(&o, r, &cfg("pbe", false, None));
        out.push((name("pbe_ri"), plain("pbe", &pbe)));
        out.push((name("pbe_coarse"), on_grids("pbe", &pbe)));
        let b3 = scf(&o, r, &cfg("b3lyp", true, None));
        out.push((name("b3lyp_exact"), plain("b3lyp", &b3)));
        let wb = scf(&o, r, &cfg("HYB_GGA_XC_WB97X", false, None));
        out.push((name("wb97x_routed"), plain("HYB_GGA_XC_WB97X", &wb)));
        out.push((
            name("wb97x_exact_rsh"),
            plain("HYB_GGA_XC_WB97X", &without_route(&wb)),
        ));
        let cx = cosx();
        let b3c = scf(&o, r, &cfg("b3lyp", true, Some(cx.clone())));
        out.push((name("b3lyp_cosx"), with_x("b3lyp", &b3c, Some(&cx))));
        out.push((name("b3lyp_cosx_none"), with_x("b3lyp", &b3c, None)));
        {
            let da = bump(&pbe.density_alpha);
            let db = pbe.density_beta.clone().unwrap();
            let wd = bump(&da);
            let f = match r {
                Ref::Uks => ks_gradient_uks_for_density,
                _ => ks_gradient_roks_for_density,
            };
            out.push((
                name("pbe_for_density"),
                f(
                    m,
                    p,
                    bs,
                    op,
                    b,
                    "pbe",
                    &pbe,
                    None,
                    None,
                    &da,
                    &db,
                    &wd,
                    &KsGradGrids::default(),
                )
                .unwrap(),
            ));
        }
        if let Ref::Uks = r {
            out.push((
                name("pbe_polar"),
                ks_gradient_uks_with_polarizable(
                    m,
                    p,
                    bs,
                    op,
                    b,
                    "pbe",
                    &pbe,
                    None,
                    Some(&sites()),
                    Some(&dipoles()),
                )
                .unwrap(),
            ));
            out.push((
                name("pbe_xc_only"),
                xc_gradient_open(
                    m,
                    p,
                    bs,
                    "pbe",
                    (&pbe.density_alpha, pbe.density_beta.as_ref().unwrap()),
                    &KsGradGrids::default(),
                )
                .unwrap(),
            ));
            let scan = scf(&o, r, &cfg("scan", false, None));
            out.push((name("scan"), plain("scan", &scan)));
            let wbv = scf(&o, r, &cfg("wB97X-V", false, None));
            out.push((name("wb97xv_coarse"), on_grids("wB97X-V", &wbv)));
        }
    }
    out
}

/// Bits recorded from the pre-refactor implementation.
const EXPECTED: &[(&str, &[u64])] = &[
    (
        "rks_pbe_ri",
        &[
            0x3f5d7b0b4c4313da,
            0x3f87dd467414e398,
            0xbfc1ca7af8c64510,
            0xbf692791e228f6c0,
            0xbfb1a38afd6caeb8,
            0x3fb37e75b3754940,
            0x3f54d418780e21d8,
            0x3fad4fc45dd424e0,
            0x3fb016803e173996,
        ],
    ),
    (
        "rks_pbe_coarse",
        &[
            0x3f5d7e5f380bbac6,
            0x3f87bca683583680,
            0xbfc1e9855686d72c,
            0xbf693e5a3dc71faa,
            0xbfb18b8bcb80d29e,
            0x3fb397c820af4d4c,
            0x3f54fe5543829784,
            0x3fad27edf62b9678,
            0x3fb03b428c5e57d0,
        ],
    ),
    (
        "rks_pbe_polar",
        &[
            0x3f61f3f31460b0ed,
            0x3f878fc62808aecb,
            0xbfc1dabc1f8e3746,
            0xbf690ca5afad623d,
            0xbfb1ae9a76c3b4ac,
            0x3fb391be4993e42b,
            0x3f540bd1835e818c,
            0x3fad6bac2e10e669,
            0x3fb0255828ca278c,
        ],
    ),
    (
        "rks_pbe_polar_coarse",
        &[
            0x3f61f59d0a450463,
            0x3f876f26374c01b3,
            0xbfc1f9c67d4ec962,
            0xbf69236e0b4b8b27,
            0xbfb1969b44d7d892,
            0x3fb3ab10b6cde837,
            0x3f54360e4ed2f738,
            0x3fad43d5c6685801,
            0x3fb04a1a771145c6,
        ],
    ),
    (
        "rks_pbe_for_density",
        &[
            0x3f699170533254de,
            0x3f9323ee091e8a3d,
            0xbfc2d2ada2464afe,
            0xbf6c09811428709e,
            0xbfb3ec3efadc928a,
            0x3fb48e85b7925c76,
            0x3f33c08607b2eab0,
            0x3fae4686f129e4b0,
            0x3fb116d58cfa340a,
        ],
    ),
    (
        "rks_pbe_xc_only",
        &[
            0xbf64d51d03e0078b,
            0x3f7ba5eef5a4f6c0,
            0x3fd2dd1d2b3f2df8,
            0x3f7f940e1966be50,
            0x3fc88dde75b3b7fc,
            0xbfc3360bf878c8b8,
            0xbf75297f9776e8ba,
            0xbfc96b0ded60dfc8,
            0xbfc2842e5e059305,
        ],
    ),
    (
        "rks_b3lyp_exact",
        &[
            0x3f5ba639ee5a08d4,
            0x3f887aaabf9c5862,
            0xbfc04cb916178c31,
            0xbf6754f080f1cc62,
            0xbfb066476fdf7cb2,
            0x3fb1fc00356fc2b5,
            0x3f5303a71389854c,
            0x3faaade42fd7ea44,
            0x3fad3ae3ed7e99f2,
        ],
    ),
    (
        "rks_wb97x_routed",
        &[
            0x3f5a027b41d2562c,
            0x3f894d28f768996e,
            0xbfbee5b897f46ce2,
            0xbf65596e8bee5ef2,
            0xbfae4dd985ed6fc0,
            0x3fb1219551898999,
            0x3f50b061d60ae0c0,
            0x3fa7fa8f48134bd8,
            0x3fab88468cd5bb6e,
        ],
    ),
    (
        "rks_wb97x_exact_rsh",
        &[
            0x3f5a03256c6bcd2c,
            0x3f894e521634ce9e,
            0xbfbee64da9e2c066,
            0xbf6559e1aae11372,
            0xbfae4e7ee443a91c,
            0x3fb121efb0245361,
            0x3f50b09de956d618,
            0x3fa7faea5eb677a0,
            0x3fab88bbf37ccaee,
        ],
    ),
    (
        "rks_b3lyp_rijcosx",
        &[
            0x3f5bd6dba8b426a8,
            0x3f88749074a20dce,
            0xbfc049f5969b3aeb,
            0xbf676c0a055bfbea,
            0xbfb067cb81db196a,
            0x3fb1f96e9c82b170,
            0x3f53013862024c38,
            0x3faab272e68db240,
            0x3fad34f921677674,
        ],
    ),
    (
        "rks_b3lyp_cosx_4c_j",
        &[
            0x3f5bd79c30cf34e8,
            0x3f8875db0ab72e7e,
            0xbfc04a4b165c66fb,
            0xbf676c97232c6f0a,
            0xbfb068330c41aefa,
            0x3fb1f9d5f30fdde0,
            0x3f530192158825b8,
            0x3faab2ef55d594c0,
            0x3fad35807351cd34,
        ],
    ),
    (
        "rks_scan",
        &[
            0x3f5a434252318442,
            0x3f8885cdcc4024f6,
            0xbfbfff272def16c4,
            0xbf64e0ec7cc55854,
            0xbfaebc1b14937ffc,
            0x3fb1a55d6cb23f96,
            0x3f4efd2d4eb421a0,
            0x3fa89aa7a1837514,
            0x3facb3938279a244,
        ],
    ),
    (
        "rks_lda",
        &[
            0x3f5c1effdd692260,
            0x3f885b0353237776,
            0xbfc08913dab6ad12,
            0xbf680c03f1a8ee6c,
            0xbfb0ea1b384bbafc,
            0x3fb23a8243a53a30,
            0x3f53f90805e92258,
            0x3fabbd759bce8fe8,
            0x3fadaf4ae3902818,
        ],
    ),
    (
        "rks_wb97xv_coarse",
        &[
            0x3f5ae148145946d9,
            0x3f88e8b19fa448d9,
            0xbfbfb3d2e3e7201c,
            0xbf663c237133cbd9,
            0xbfaf0f6aec9ad9a1,
            0x3fb187aa75765aeb,
            0x3f519c5968b08121,
            0x3fa8d572d6c6aad2,
            0x3fac57d39027abf1,
        ],
    ),
    (
        "uks_pbe_ri",
        &[
            0x3f5c60db14b7e608,
            0xbf4c6ca8e347a2e0,
            0x3fb5396a0640f00e,
            0xbf5c60db14b73dd8,
            0x3f4c6ca8e348fae0,
            0xbfb5396a0640ef12,
        ],
    ),
    (
        "uks_pbe_coarse",
        &[
            0x3f5b3404adad7fd8,
            0xbf4ae24d580bb604,
            0x3fb51fe49e715c2c,
            0xbf5b3404adaefee0,
            0x3f4ae24d580c4924,
            0xbfb51fe49e715d02,
        ],
    ),
    (
        "uks_b3lyp_exact",
        &[
            0x3f59c7d00f84f76a,
            0xbf49ca41f82a189e,
            0x3fb3473cb56e25ae,
            0xbf59c7d00f812472,
            0x3f49ca41f822a7c8,
            0xbfb3473cb56e5238,
        ],
    ),
    (
        "uks_wb97x_routed",
        &[
            0x3f55f4b30ca7429e,
            0xbf4640f5ec298060,
            0x3fb1901aed800df2,
            0xbf55f4b30ca815fc,
            0x3f4640f5ec286d0e,
            0xbfb1901aed801012,
        ],
    ),
    (
        "uks_wb97x_exact_rsh",
        &[
            0x3f55f4aad716c5f6,
            0xbf4640edb60634c0,
            0x3fb19014a44e09d2,
            0xbf55f4aad7179954,
            0x3f4640edb605216a,
            0xbfb19014a44e0bf0,
        ],
    ),
    (
        "uks_b3lyp_cosx",
        &[
            0x3f59e50a9dff87d6,
            0xbf49e7a7eb99dc02,
            0x3fb346ef2762b472,
            0xbf59e50a9dfbd340,
            0x3f49e7a7eb92d99a,
            0xbfb346ef2762e064,
        ],
    ),
    (
        "uks_b3lyp_cosx_none",
        &[
            0x3f59c80209844296,
            0xbf49ca882bf15802,
            0x3fb347231f7735f2,
            0xbf59c80209808d60,
            0x3f49ca882bea551a,
            0xbfb347231f776204,
        ],
    ),
    (
        "uks_pbe_for_density",
        &[
            0xbf683ff2fc2c5fac,
            0x3f67015c84a6894c,
            0xbfc281f60ddeecda,
            0x3f683ff2fc2c911e,
            0xbf67015c84a5cdbd,
            0x3fc281f60ddeee2b,
        ],
    ),
    (
        "uks_pbe_polar",
        &[
            0x3f5bee51c5a5cde2,
            0xbf4c67d71c4d9383,
            0x3fb54d095cc15db3,
            0xbf5f07d48093dec4,
            0x3f4db229a4725d84,
            0xbfb54abc77a3c98a,
        ],
    ),
    (
        "uks_pbe_xc_only",
        &[
            0xbf72d78e26e1aa7e,
            0x3f62d4bc234e82d8,
            0xbfcd1aa527d223a9,
            0x3f72d78e26e1d462,
            0xbf62d4bc234e2cd8,
            0x3fcd1aa527d22467,
        ],
    ),
    (
        "uks_scan",
        &[
            0x3f54603d8436a96c,
            0xbf43f6f863c558e4,
            0x3fb2168dd53614a2,
            0xbf54603d8458788c,
            0x3f43f6f863fdc428,
            0xbfb2168dd536f732,
        ],
    ),
    (
        "uks_wb97xv_coarse",
        &[
            0x3f5769d2aba6f67f,
            0xbf47373084a0d0f5,
            0x3fb204d7b20870f3,
            0xbf5765d8b9a237d3,
            0x3f47323bff5f316b,
            0xbfb204a7cc9cff6e,
        ],
    ),
    (
        "roks_pbe_ri",
        &[
            0x3f5c51839212b2fc,
            0xbf4c5ec38229d298,
            0x3fb52ed0682faadc,
            0xbf5c518392127900,
            0x3f4c5ec3822d2fbc,
            0xbfb52ed0682fabca,
        ],
    ),
    (
        "roks_pbe_coarse",
        &[
            0x3f5b2f7ebd0a6500,
            0xbf4acac797c19bac,
            0x3fb51534df9a9234,
            0xbf5b2f7ebd0b84b8,
            0x3f4acac797c0d5d4,
            0xbfb51534df9a916a,
        ],
    ),
    (
        "roks_b3lyp_exact",
        &[
            0x3f59b5c1f6ae59ae,
            0xbf49bbe33e9a4074,
            0x3fb33ae8e6afd5d8,
            0xbf59b5c1f6aa1814,
            0x3f49bbe33e9389fa,
            0xbfb33ae8e6b001ae,
        ],
    ),
    (
        "roks_wb97x_routed",
        &[
            0x3f56137127ab9238,
            0xbf45d4785affe72a,
            0x3fb1794b1967b1f8,
            0xbf56137127ac96f8,
            0x3f45d4785aff0798,
            0xbfb1794b1967b1e2,
        ],
    ),
    (
        "roks_wb97x_exact_rsh",
        &[
            0x3f56136884dafd94,
            0xbf45d46fb7bad8c2,
            0x3fb179447c557268,
            0xbf56136884dc0230,
            0x3f45d46fb7b9f930,
            0xbfb179447c557252,
        ],
    ),
    (
        "roks_b3lyp_cosx",
        &[
            0x3f59d2e9e3766454,
            0xbf49da0c6be9ea0a,
            0x3fb33a9aa986f5f0,
            0xbf59d2e9e3727d34,
            0x3f49da0c6be438a4,
            0xbfb33a9aa9872496,
        ],
    ),
    (
        "roks_b3lyp_cosx_none",
        &[
            0x3f59b5fe89ccabd4,
            0xbf49bc1aed8fb80a,
            0x3fb33acf4784ce10,
            0xbf59b5fe89c8c474,
            0x3f49bc1aed8a0624,
            0xbfb33acf4784fcb6,
        ],
    ),
    (
        "roks_pbe_for_density",
        &[
            0xbf688b75ddb40d6e,
            0x3f672b2044db4bfa,
            0xbfc2c67511f0cac2,
            0x3f688b75ddb3cc3a,
            0xbf672b2044db395c,
            0x3fc2c67511f0c927,
        ],
    ),
];

/// Per-component tolerance (Hartree/Bohr) for the default, cross-machine mode.
const TOL: f64 = 1e-10;

#[test]
fn ks_gradients_match_recorded() {
    let cases = all_cases();
    if EXPECTED.is_empty() {
        for (name, g) in &cases {
            let hex: Vec<String> = bits(g).iter().map(|b| format!("0x{b:016x}")).collect();
            println!("    (\"{name}\", &[{}]),", hex.join(", "));
        }
        panic!("EXPECTED is empty: recorded table printed above");
    }
    let bit_exact = std::env::var("FERRIC_GRADIENT_BIT_EXACT").is_ok_and(|v| v == "1");
    assert_eq!(cases.len(), EXPECTED.len(), "case count");
    for ((name, g), (ename, ebits)) in cases.iter().zip(EXPECTED) {
        assert_eq!(name, ename);
        assert_eq!(
            g.ncols(),
            3,
            "{name}: gradient must have three Cartesian columns"
        );
        if bit_exact {
            assert_eq!(bits(g).as_slice(), *ebits, "{name}: gradient bits changed");
            continue;
        }
        let got: Vec<f64> = g.iter().copied().collect();
        assert_eq!(got.len(), ebits.len(), "{name}: component count");
        for (i, (x, b)) in got.iter().zip(ebits.iter()).enumerate() {
            let want = f64::from_bits(*b);
            assert!(
                (x - want).abs() <= TOL,
                "{name}[{i}]: {x:e} vs recorded {want:e} (|diff| {:e} > {TOL:e})",
                (x - want).abs()
            );
        }
    }
}
