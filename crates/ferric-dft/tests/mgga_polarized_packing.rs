//! Regression guard on the SPIN-POLARIZED meta-GGA libxc packing.
//!
//! libxc's nspin=2 meta-GGA ABI interleaves per point:
//!   rho[2g+σ], sigma[3g+{αα,αβ,ββ}], tau[2g+σ]
//! and ferric's `eval_mgga_polarized` must honour all three strides — including
//! across the parallel chunk boundaries, where each chunk offsets by 2*g0 / 3*g0
//! rather than g0.
//!
//! A transposed or mis-strided τ buffer is INVISIBLE whenever τ_α == τ_β, which
//! is exactly the degenerate case a closed-shell test exercises. In 2026-07 an
//! open-shell SCAN energy discrepancy was misattributed to precisely such a bug;
//! the hypothesis was only ruled out by checking these strides against PySCF's
//! `libxc.eval_xc(..., spin=1)` on inputs with ρ_α≠ρ_β AND τ_α≠τ_β. This test
//! keeps that check permanently, so the same hypothesis never has to be
//! re-litigated by hand.
//!
//! The PySCF values below were produced with (pyscf.dft.libxc.eval_xc, deriv=1)
//! on the same five points and matched ferric BIT-FOR-BIT at the time of
//! writing; they are stored to 17 significant digits and asserted to 1e-12
//! relative (loose enough for a libxc patch-version rebuild, tight enough that
//! any stride/factor error — all of which are O(1) relative — fails loudly).

use ferric_dft::libxc::XcFunctional;

/// (ρ_α, ρ_β, σ_αα, σ_αβ, σ_ββ, τ_α, τ_β) — every point has ρ_α≠ρ_β and
/// τ_α≠τ_β except the last, which is the degenerate control.
const PTS: [[f64; 7]; 5] = [
    [0.30, 0.10, 0.20, 0.06, 0.03, 0.25, 0.05],
    [0.05, 0.04, 0.01, 0.008, 0.007, 0.02, 0.015],
    [1.20, 0.90, 2.50, 1.90, 1.60, 1.10, 0.80],
    [0.02, 0.0005, 0.003, 0.0001, 0.00002, 0.004, 0.0001],
    [0.50, 0.50, 0.40, 0.40, 0.40, 0.30, 0.30],
];

/// Per functional: for each of the 5 points, (exc, vrho_a, vrho_b, vtau_a, vtau_b).
/// vsigma is checked structurally (see `sigma_stride_is_honoured`) rather than
/// pinned here.
struct Case {
    name: &'static str,
    vals: [[f64; 5]; 5],
}

const CASES: [Case; 2] = [
    Case {
        name: "MGGA_X_SCAN",
        vals: [
            [-6.5710158612955627e-1, -9.7421204848721643e-1, -6.5929790410911293e-1,
              4.8260949522549682e-2, 5.1534253132780400e-2],
            [-3.8885384001161238e-1, -5.1834436722163357e-1, -4.7995514557138069e-1,
              5.1086020560353163e-2, 5.4436087200114953e-2],
            [-1.0989214002799412e0, -1.5496883437928213e0, -1.4068329907227120e0,
              2.5721338839697516e-2, 2.8751349155763172e-2],
            [-2.9086805305390517e-1, -3.8281019326160365e-1, -8.0807573490160101e-2,
              7.0064852605350073e-2, 1.8726775444683116e-1],
            [-8.5464014618431405e-1, -1.1546356325890939e0, -1.1546356325890939e0,
              3.4207265995712263e-2, 3.4207265995712263e-2],
        ],
    },
    Case {
        name: "MGGA_C_R2SCAN",
        vals: [
            [-2.6871348592676380e-2, -1.5025159974549727e-2, -5.3154939053378480e-2,
             -1.6840515249517348e-2, -1.6840515249517348e-2],
            [-2.0564829783331080e-2, -3.0472610883333369e-2, -3.5301907032786753e-2,
             -1.8457136993702196e-2, -1.8457136993702196e-2],
            [-2.9917733817732550e-2, -2.1016104564960912e-2, -3.1078972541447585e-2,
             -8.1746651206136400e-3, -8.1746651206136400e-3],
            [-2.6616626666151871e-3, -6.0734945194064159e-3, -1.4818531314519098e-1,
             -1.9917393442853155e-2, -1.9917393442853169e-2],
            [-2.8840146712116571e-2, -2.6458840915514217e-2, -2.6458840915514217e-2,
             -1.1796661666463744e-2, -1.1796661666463744e-2],
        ],
    },
];

fn pack() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = PTS.len();
    let mut rho = vec![0.0; 2 * n];
    let mut sig = vec![0.0; 3 * n];
    let mut tau = vec![0.0; 2 * n];
    for (g, p) in PTS.iter().enumerate() {
        rho[2 * g] = p[0];
        rho[2 * g + 1] = p[1];
        sig[3 * g] = p[2];
        sig[3 * g + 1] = p[3];
        sig[3 * g + 2] = p[4];
        tau[2 * g] = p[5];
        tau[2 * g + 1] = p[6];
    }
    (rho, sig, tau)
}

fn eval(name: &str) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = PTS.len();
    let f = XcFunctional::new(name, 2).unwrap();
    let (rho, sig, tau) = pack();
    let mut exc = vec![0.0; n];
    let mut vrho = vec![0.0; 2 * n];
    let mut vsig = vec![0.0; 3 * n];
    let mut vtau = vec![0.0; 2 * n];
    f.eval_mgga_polarized(&rho, &sig, &tau, &mut exc, &mut vrho, &mut vsig, &mut vtau);
    (exc, vrho, vsig, vtau)
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-12 * a.abs().max(b.abs()).max(1e-30)
}

#[test]
fn polarized_mgga_matches_pyscf_eval_xc() {
    for case in &CASES {
        let (exc, vrho, _, vtau) = eval(case.name);
        for (g, want) in case.vals.iter().enumerate() {
            let got = [exc[g], vrho[2 * g], vrho[2 * g + 1], vtau[2 * g], vtau[2 * g + 1]];
            let labels = ["exc", "vrho_a", "vrho_b", "vtau_a", "vtau_b"];
            for k in 0..5 {
                assert!(
                    close(got[k], want[k]),
                    "{} point {g} {}: ferric {:.17e} vs PySCF {:.17e}",
                    case.name, labels[k], got[k], want[k]
                );
            }
        }
    }
}

/// The α/β τ channels must be genuinely distinct — a buffer that dropped or
/// duplicated a spin would still pass a τ_α==τ_β test. Points 0-3 have
/// τ_α ≠ τ_β; for the exchange functional (no cross-spin coupling) that MUST
/// produce v_τα ≠ v_τβ.
#[test]
fn tau_alpha_and_beta_are_independent_channels() {
    let (_, _, _, vtau) = eval("MGGA_X_SCAN");
    for g in 0..4 {
        let (a, b) = (vtau[2 * g], vtau[2 * g + 1]);
        assert!(
            (a - b).abs() > 1e-6 * a.abs().max(b.abs()),
            "point {g}: v_τα ({a:.6e}) and v_τβ ({b:.6e}) are indistinguishable — \
             the τ buffer is not spin-resolved"
        );
    }
    // The degenerate control point MUST give equal channels.
    assert_eq!(vtau[8].to_bits(), vtau[9].to_bits(),
               "τ_α == τ_β point must yield v_τα == v_τβ");
}

/// The σ channel is 3-strided while ρ/τ are 2-strided. For a pure EXCHANGE
/// meta-GGA there is no αβ cross term, so v_σαβ must be exactly zero — which
/// only holds if the 3-stride is honoured (a 2-stride read would smear the ββ
/// channel into it).
#[test]
fn sigma_stride_is_honoured() {
    let (_, _, vsig, _) = eval("MGGA_X_SCAN");
    for g in 0..PTS.len() {
        assert_eq!(
            vsig[3 * g + 1], 0.0,
            "point {g}: exchange-only v_σαβ must be exactly 0, got {:e} — \
             sigma is being read with the wrong stride",
            vsig[3 * g + 1]
        );
        assert!(vsig[3 * g] < 0.0 && vsig[3 * g + 2] < 0.0,
                "point {g}: v_σαα / v_σββ must both be negative for SCAN exchange");
    }
}

/// The parallel chunked path offsets by 2*g0 / 3*g0 per chunk; it must be
/// bit-identical to the serial path. `PAR_MIN_PTS` is 50_000, so this sizes
/// above it and compares against sub-problems small enough to stay serial.
#[test]
fn chunked_path_is_bit_identical_to_serial() {
    let n = 60_000;
    let build = |n: usize| {
        let (mut rho, mut sig, mut tau) = (vec![0.0; 2 * n], vec![0.0; 3 * n], vec![0.0; 2 * n]);
        for g in 0..n {
            let t = (g as f64) * 1e-4;
            let (ga, gb) = (0.3 + 0.5 * (2.0 * t).sin().abs(), 0.2 + 0.3 * (3.0 * t).cos().abs());
            rho[2 * g] = 0.05 + 0.9 * t.sin().abs();
            rho[2 * g + 1] = 0.01 + 0.4 * t.cos().abs();
            sig[3 * g] = ga * ga;
            sig[3 * g + 1] = ga * gb;
            sig[3 * g + 2] = gb * gb;
            tau[2 * g] = 0.5 + 0.4 * t.sin().abs();
            tau[2 * g + 1] = 0.2 + 0.3 * t.cos().abs();
        }
        (rho, sig, tau)
    };
    for name in ["MGGA_X_SCAN", "MGGA_C_SCAN", "MGGA_X_R2SCAN", "MGGA_C_R2SCAN"] {
        let f = XcFunctional::new(name, 2).unwrap();
        let (rho, sig, tau) = build(n);
        let (mut exc, mut vrho) = (vec![0.0; n], vec![0.0; 2 * n]);
        let (mut vsig, mut vtau) = (vec![0.0; 3 * n], vec![0.0; 2 * n]);
        f.eval_mgga_polarized(&rho, &sig, &tau, &mut exc, &mut vrho, &mut vsig, &mut vtau);

        let step = 1000;
        let mut g0 = 0;
        while g0 < n {
            let g1 = (g0 + step).min(n);
            let m = g1 - g0;
            let (mut e, mut vr) = (vec![0.0; m], vec![0.0; 2 * m]);
            let (mut vs, mut vt) = (vec![0.0; 3 * m], vec![0.0; 2 * m]);
            f.eval_mgga_polarized(
                &rho[2 * g0..2 * g1], &sig[3 * g0..3 * g1], &tau[2 * g0..2 * g1],
                &mut e, &mut vr, &mut vs, &mut vt,
            );
            for k in 0..m {
                assert_eq!(exc[g0 + k].to_bits(), e[k].to_bits(), "{name} exc at {}", g0 + k);
                for s in 0..2 {
                    assert_eq!(vrho[2 * (g0 + k) + s].to_bits(), vr[2 * k + s].to_bits(),
                               "{name} vrho[{s}] at {}", g0 + k);
                    assert_eq!(vtau[2 * (g0 + k) + s].to_bits(), vt[2 * k + s].to_bits(),
                               "{name} vtau[{s}] at {}", g0 + k);
                }
                for s in 0..3 {
                    assert_eq!(vsig[3 * (g0 + k) + s].to_bits(), vs[3 * k + s].to_bits(),
                               "{name} vsigma[{s}] at {}", g0 + k);
                }
            }
            g0 = g1;
        }
    }
}
