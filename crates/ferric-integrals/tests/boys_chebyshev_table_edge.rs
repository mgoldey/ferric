//! libint2's `FmEval_Chebyshev7` read one row past its table when the Boys
//! argument was EXACTLY `T_crit = 117` (issue #433).
//!
//! ROOT CAUSE. `eval` takes the asymptotic branch for `x > T_crit` and the
//! interpolation branch otherwise, with `iv = int(x * 7)` indexing a table of
//! `819 = 117 * 7` intervals (rows 0..=818). At `x == 117.0` the strict `>`
//! sends the call to interpolation with `iv == 819`: the 8 coefficients per
//! `m` are read from the 64-byte-aligned heap block just past the table, i.e.
//! uninitialised memory. Any other `x` is fine (just below 117 gives
//! `iv == 818`, above gives the asymptotic branch). The engine precision only
//! decides whether the one primitive quartet with `T == 117` survives primitive
//! screening: at 1e-14 and 1e-10 it is screened out, at 1e-20 and below it is
//! kept (so "garbage at 2.2e-308, 1e-100, 1e-20; clean at 1e-14"). The exact
//! hit needs `rho * |PQ|^2 == 117` in floating point, which lattice-integer
//! shifts with "nice" exponents produce (here `0.9 * 130`).
//!
//! ARTIFACT vs PHYSICS hypothesis (written before the root cause was known).
//! Physics: a far-field erfc integral is ~1e-22, so a 1e288 value cannot be
//! real. Artifact: the value changes with allocator fill (`MALLOC_PERTURB_`)
//! and with nothing else; both were observed, so it is an uninitialised read,
//! and the C++ reproducer showed one primitive quartet's `Gm[0]` (and only
//! that one) as the garbage carrier.
//!
//! Tests here:
//! * `t_crit_hit_matches_the_closed_form`: the exact-hit geometry against an
//!   independent closed form (Gaussian product theorem + erf/erfc potential of
//!   a Gaussian charge), relative 1e-12, for Coulomb and a long-range erfc, and
//!   absolutely (< 1e-15) for the issue's omega = 0.8, where the true value is
//!   ~1e-23 and the engine's cancellation noise is ~5e-17. The closed form is
//!   first checked at a generic geometry (anchor: a wrong formula would fail
//!   there, where no table edge is involved).
//! * `sweep_is_independent_of_allocator_fill`: a shells x geometries x
//!   operators x precisions sweep, hashed bit-for-bit in a child process under
//!   several `MALLOC_PERTURB_` values; every hash must agree.

use ferric_core::basis::{BasisSet, Shell};
use ferric_core::mol::{Atom, Molecule};
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::engine::Engine;
use ferric_integrals::operator::Operator;
use ferric_integrals::site_basis::SiteBasis;
use std::collections::HashMap;
use std::f64::consts::PI;

// libm's C99 erf/erfc, bound directly (same convention as ferric-pcm).
extern "C" {
    fn erf(x: f64) -> f64;
    fn erfc(x: f64) -> f64;
}

/// The "I" centre of the issue's reproducer (Bohr).
const I_POS: [f64; 3] = [0.9, -0.4, 3.3];
/// Exponent of the point-like site (the hcore Gaussian nucleus).
const SITE_ZETA: f64 = 1e16;
/// The reproducer's image shifts: aux site by `M`, second orbital by `L`.
const M_SHIFT: [f64; 3] = [6.0, 0.0, -7.0];
const L_SHIFT: [f64; 3] = [-6.0, 0.0, 0.0];

/// An s shell with ferric's convention: coefficients refer to unit-normalised
/// primitives, contraction scaled to unit self-overlap.
fn s_shell(exps: &[f64], coefs: &[f64]) -> Shell {
    let mut s = 0.0;
    for (a, ca) in exps.iter().zip(coefs) {
        for (b, cb) in exps.iter().zip(coefs) {
            s += ca * cb * (2.0 * (a * b).sqrt() / (a + b)).powf(1.5);
        }
    }
    Shell {
        l: 0,
        pure: false,
        exponents: exps.to_vec(),
        coefficients: coefs.iter().map(|c| c / s.sqrt()).collect(),
    }
}

fn p_shell(exps: &[f64], coefs: &[f64]) -> Shell {
    let mut s = 0.0;
    for (a, ca) in exps.iter().zip(coefs) {
        for (b, cb) in exps.iter().zip(coefs) {
            s += ca * cb * (2.0 * (a * b).sqrt() / (a + b)).powf(2.5);
        }
    }
    Shell {
        l: 1,
        pure: false,
        exponents: exps.to_vec(),
        coefficients: coefs.iter().map(|c| c / s.sqrt()).collect(),
    }
}

fn prep_of(shell: Shell) -> PreparedBasis {
    let mol = Molecule {
        atoms: vec![Atom {
            symbol: "I".into(),
            z: 53,
            x: I_POS[0],
            y: I_POS[1],
            zpos: I_POS[2],
            ghost: false,
            n_core_ecp: 0,
        }],
        charge: 0,
        multiplicity: 1,
    };
    let mut shells = HashMap::new();
    shells.insert(53, vec![shell]);
    let bs = BasisSet {
        name: "t-crit".into(),
        shells,
        ecps: HashMap::new(),
    };
    PreparedBasis::new(&mol, &bs).expect("prep")
}

fn site() -> SiteBasis {
    SiteBasis::new(&[[I_POS[0], I_POS[1], I_POS[2], SITE_ZETA]], 0).expect("site")
}

/// `(P | mu nu)` with the aux site shifted by `m`, the second orbital by `l`.
fn block(
    shell: &Shell,
    op: Operator,
    precision: f64,
    m: [f64; 3],
    l: [f64; 3],
) -> Option<Vec<f64>> {
    let prep = prep_of(shell.clone());
    let site = site();
    let mut eng = Engine::new_3center(op, &prep, &site.prep, precision).expect("engine");
    eng.compute_eri3_shifted(
        &prep,
        &site.prep,
        site.site_shell[0],
        0,
        0,
        [m, [0.0; 3], l],
    )
    .expect("compute")
    .map(<[f64]>::to_vec)
}

/// Closed form of `(P | mu nu)` for a contracted s shell and a Gaussian site:
/// Gaussian product theorem over primitive pairs, and the potential of a
/// Gaussian charge for the kernel `erf(wr)/r` (`omega = None`: Coulomb) or its
/// complement `erfc(wr)/r`. `erfc_omega = Some(w)` selects `erfc(w r)/r`.
fn closed_form(shell: &Shell, erfc_omega: Option<f64>, m: [f64; 3], l: [f64; 3]) -> f64 {
    let pc = [I_POS[0] + m[0], I_POS[1] + m[1], I_POS[2] + m[2]];
    let a = I_POS;
    let b = [I_POS[0] + l[0], I_POS[1] + l[1], I_POS[2] + l[2]];
    let ab2: f64 = (0..3).map(|k| (a[k] - b[k]).powi(2)).sum();
    // The aux shell is unit-normalised: weight (2 z / pi)^(3/4), integral
    // of the Gaussian (pi / z)^(3/2).
    let aux_w = (2.0 * SITE_ZETA / PI).powf(0.75) * (PI / SITE_ZETA).powf(1.5);
    let mut total = 0.0;
    for (ap, cp) in shell.exponents.iter().zip(&shell.coefficients) {
        for (bq, cq) in shell.exponents.iter().zip(&shell.coefficients) {
            let np = (2.0 * ap / PI).powf(0.75);
            let nq = (2.0 * bq / PI).powf(0.75);
            let g = ap + bq;
            let k = (-(ap * bq / g) * ab2).exp();
            let q: Vec<f64> = (0..3).map(|i| (ap * a[i] + bq * b[i]) / g).collect();
            let r =
                ((pc[0] - q[0]).powi(2) + (pc[1] - q[1]).powi(2) + (pc[2] - q[2]).powi(2)).sqrt();
            // Two Gaussian charges: effective exponent g z / (g + z).
            let ge = g * SITE_ZETA / (g + SITE_ZETA);
            let v = match erfc_omega {
                None => (unsafe { erf(ge.sqrt() * r) }) / r,
                Some(w) => {
                    let kappa = (ge * w * w / (ge + w * w)).sqrt();
                    (unsafe { erfc(kappa * r) - erfc(ge.sqrt() * r) }) / r
                }
            };
            total += cp * np * cq * nq * k * (PI / g).powf(1.5) * aux_w * v;
        }
    }
    total
}

fn contracted_i_s() -> Shell {
    s_shell(&[1.2, 0.45], &[0.25, 0.55])
}

/// A generic geometry (no Boys argument near 117): anchors the closed form.
#[test]
fn closed_form_matches_the_engine_at_a_generic_geometry() {
    let shell = contracted_i_s();
    let m = [2.3, 0.4, -1.9];
    let l = [-1.7, 0.6, 0.8];
    for (op, om) in [
        (Operator::coulomb(), None),
        (Operator::erfc(0.3), Some(0.3)),
        (Operator::erfc(0.8), Some(0.8)),
    ] {
        let got = block(&shell, op, 1e-20, m, l).expect("not screened")[0];
        let want = closed_form(&shell, om, m, l);
        let rel = (got - want).abs() / want.abs();
        assert!(
            rel < 1e-10,
            "{om:?}: engine {got:e} vs closed form {want:e}"
        );
    }
}

/// The exact-hit geometry (a Boys argument of exactly 117.0 in the
/// (0.45, 0.45) primitive pair) must give the closed-form value.
#[test]
fn t_crit_hit_matches_the_closed_form() {
    let shell = contracted_i_s();
    // Long-range kernels keep the value above rounding noise. The whole block
    // is ~1e-17 (the site integrates to ~1e-12), so use the hcore engine
    // precision: at 1e-20 primitive screening would drop ~3e-7 of it.
    for (op, om) in [
        (Operator::coulomb(), None),
        (Operator::erfc(0.1), Some(0.1)),
    ] {
        let got = block(&shell, op, f64::MIN_POSITIVE, M_SHIFT, L_SHIFT).expect("not screened")[0];
        let want = closed_form(&shell, om, M_SHIFT, L_SHIFT);
        let rel = (got - want).abs() / want.abs();
        assert!(
            rel < 1e-12,
            "{om:?}: engine {got:e} vs closed form {want:e} (rel {rel:.1e})"
        );
    }
    // The issue's operator and the hcore engine precision: the true value is
    // ~1e-23, the engine's cancellation noise ~5e-17.
    let got = block(
        &shell,
        Operator::erfc(0.8),
        f64::MIN_POSITIVE,
        M_SHIFT,
        L_SHIFT,
    )
    .expect("kept at MIN_POSITIVE")[0];
    let want = closed_form(&shell, Some(0.8), M_SHIFT, L_SHIFT);
    assert!(want.abs() < 1e-20, "closed form {want:e}");
    assert!(got.abs() < 1e-15, "engine {got:e}");
}

// ----------------------------------------------------------------- sweep

fn fnv(h: &mut u64, x: f64) {
    for b in x.to_bits().to_le_bytes() {
        *h ^= u64::from(b);
        *h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
}

/// Displacements `(m, l)` whose aux-to-pair-centre distance squared is 130
/// (so `T = 0.9 * 130` near 117 for the (0.45, 0.45) pair) in every axis
/// assignment, plus generic ones.
fn geometries() -> Vec<(&'static str, [f64; 3], [f64; 3])> {
    let mut v = vec![
        ("t117", M_SHIFT, L_SHIFT),
        ("t117", [0.0, 9.0, -7.0], [0.0, -6.0, 0.0]),
        ("t117", [-9.0, 0.0, 7.0], [6.0, 0.0, 0.0]),
        ("t117", [3.0, 0.0, 11.0], [0.0, 0.0, 0.0]),
        ("t117", [7.0, 9.0, 0.0], [0.0, 0.0, 0.0]),
        ("t117", [0.0, 0.0, 0.0], [18.0, 0.0, -14.0]),
        ("t117", [12.0, 0.0, 0.0], [-6.0, 0.0, 14.0]),
    ];
    for k in 0..24 {
        let t = 0.37 * f64::from(k);
        v.push((
            "generic",
            [4.0 * t.sin(), 3.0 * t.cos(), -5.0 + 0.4 * f64::from(k)],
            [-2.5 * t.cos(), 1.5 + 0.1 * f64::from(k), 2.0 * t.sin()],
        ));
    }
    v
}

fn sweep() -> (u64, u64, usize) {
    let shells = [
        s_shell(&[1.2, 0.45], &[0.25, 0.55]),
        s_shell(&[0.45], &[1.0]),
        s_shell(&[3.0, 1.0, 0.35], &[0.3, 0.5, 0.4]),
        p_shell(&[2.0, 0.7], &[0.4, 0.8]),
        p_shell(&[0.45], &[1.0]),
    ];
    let ops: [(Operator, f64); 6] = [
        (Operator::erfc(0.8), f64::MIN_POSITIVE),
        (Operator::erfc(0.8), 1e-14),
        (Operator::erfc(0.1), 1e-20),
        (Operator::erf(0.5), 1e-20),
        (Operator::coulomb(), 1e-20),
        (Operator::coulomb(), 1e-14),
    ];
    let (mut h117, mut hgen, mut n) = (0xcbf2_9ce4_8422_2325_u64, 0xcbf2_9ce4_8422_2325_u64, 0);
    for shell in &shells {
        for (op, prec) in ops {
            for (tag, m, l) in geometries() {
                let h = if tag == "t117" { &mut h117 } else { &mut hgen };
                match block(shell, op, prec, m, l) {
                    Some(b) => b.iter().for_each(|&x| fnv(h, x)),
                    None => fnv(h, f64::NAN),
                }
                n += 1;
            }
        }
    }
    (h117, hgen, n)
}

/// Child entry point: prints the two hashes. Run by
/// `sweep_is_independent_of_allocator_fill` under several allocator fills.
#[test]
fn sweep_child_prints_hashes() {
    let (h117, hgen, n) = sweep();
    println!("SWEEP_HASH t117={h117:016x} generic={hgen:016x} n={n}");
}

#[test]
fn sweep_is_independent_of_allocator_fill() {
    let exe = std::env::current_exe().expect("current exe");
    let run = |perturb: Option<&str>| -> String {
        let mut cmd = std::process::Command::new(&exe);
        cmd.args(["sweep_child_prints_hashes", "--exact", "--nocapture"]);
        cmd.env_remove("MALLOC_PERTURB_");
        if let Some(p) = perturb {
            cmd.env("MALLOC_PERTURB_", p);
        }
        let out = cmd.output().expect("spawn");
        assert!(out.status.success(), "child failed");
        String::from_utf8_lossy(&out.stdout)
            .lines()
            .find(|l| l.starts_with("SWEEP_HASH"))
            .expect("hash line")
            .to_string()
    };
    let base = run(None);
    for p in ["1", "170", "255"] {
        let got = run(Some(p));
        assert_eq!(
            got, base,
            "MALLOC_PERTURB_={p} changed integral bits (uninitialised read)"
        );
    }
}
