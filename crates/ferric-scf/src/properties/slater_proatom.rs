//! The Slater fallback proatom of the Hirshfeld routines, ECP-aware.
//!
//! An all-electron atom keeps the single-exponential model of
//! [`slater_xi_for_z`] (`ρ⁰(r) = Z ξ³/π · exp(−2ξr)`), evaluated by exactly
//! the expression the routines used inline before this type existed.
//!
//! That model has no shell structure, so it cannot shed a core: rescaling it
//! to `Z − n_core` would keep its nuclear cusp, the most core-like feature a
//! density can have, while an ECP basis holds a valence density that is small
//! at the nucleus and peaks in the valence region. An atom carrying an ECP
//! (`Atom::n_core_ecp > 0`) therefore gets a valence proatom built from
//! Slater-type shells instead: the neutral atom's ground configuration
//! (Madelung filling), shell exponents from Slater's rules on the FULL
//! configuration (the core still screens the valence shells; the ECP mimics
//! exactly that), then the `n_core` innermost electrons removed in (n, l)
//! order and the remaining occupied shells summed as spherically averaged
//! normalized STO densities. It integrates to exactly `Z − n_core`.
//!
//! The core is removed in (n, l) order, not filling (Madelung) order: an ECP
//! core is the radially innermost shells, and the closed-shell core counts
//! 10, 18, 28, 36, 46, 60, 78 are the n, l prefixes 1s–2p, –3p, –3d, –4p,
//! –4d, –4f, –5d. Madelung order would remove 4s before 3d and reach none of
//! 28, 46, 60 or 78 at a shell boundary.

use ferric_core::mol::Atom;

use super::slater_xi_for_z;

/// Slater fallback proatom ρ⁰(r) of one atom; see the module doc.
#[derive(Debug, Clone)]
pub struct SlaterProatom {
    kind: Kind,
}

#[derive(Debug, Clone)]
enum Kind {
    /// `prefac · exp(−2 ξ r)`, `prefac = Z ξ³/π`.
    Single { xi: f64, prefac: f64 },
    /// `Σ_s c_s r^{p_s} exp(−a_s r)` over the occupied valence shells.
    Valence(Vec<StoShell>),
}

/// One spherically averaged STO shell density `c r^p exp(−a r)`, with
/// `p = 2n* − 2`, `a = 2ζ` and `c = N (2ζ)^{2n*+1} / (4π Γ(2n*+1))`.
#[derive(Debug, Clone, Copy)]
struct StoShell {
    c: f64,
    p: f64,
    a: f64,
}

/// Subshells in Madelung filling order, `(n, l)`.
const MADELUNG: [(u32, u32); 19] = [
    (1, 0),
    (2, 0),
    (2, 1),
    (3, 0),
    (3, 1),
    (4, 0),
    (3, 2),
    (4, 1),
    (5, 0),
    (4, 2),
    (5, 1),
    (6, 0),
    (4, 3),
    (5, 2),
    (6, 1),
    (7, 0),
    (5, 3),
    (6, 2),
    (7, 1),
];

/// Slater's effective principal quantum number n*.
fn n_star(n: u32) -> f64 {
    match n {
        1 => 1.0,
        2 => 2.0,
        3 => 3.0,
        4 => 3.7,
        5 => 4.0,
        6 => 4.2,
        _ => 4.3,
    }
}

/// Slater's-rules group of subshell `(n, l)`: s and p share a group, d and f
/// each form their own. Groups order lexicographically as Slater lists them
/// (1s; 2sp; 3sp; 3d; 4sp; 4d; 4f; 5sp; ...).
fn group(n: u32, l: u32) -> (u32, u32) {
    (n, if l <= 1 { 0 } else { l })
}

/// ln Γ(x) for x > 0 (Lanczos, g = 7, 9 terms; ~1e-15 relative).
fn ln_gamma(x: f64) -> f64 {
    const G: f64 = 7.0;
    const C: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    let x = x - 1.0;
    let mut a = C[0];
    let t = x + G + 0.5;
    for (i, &c) in C.iter().enumerate().skip(1) {
        a += c / (x + i as f64);
    }
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

impl SlaterProatom {
    /// The fallback proatom of `atom`: the single exponential for an
    /// all-electron atom (ghosts included, as before), the valence STO-shell
    /// density for an atom carrying an ECP (`n_core_ecp > 0`).
    pub fn for_atom(atom: &Atom) -> Self {
        if atom.n_core_ecp <= 0 {
            return Self::single(atom.z);
        }
        Self::valence(atom.z, atom.n_core_ecp)
    }

    /// The single-exponential proatom of element `z`, normalized to `z`.
    pub fn single(z: i32) -> Self {
        let xi = slater_xi_for_z(z);
        let prefac = z as f64 * xi.powi(3) / std::f64::consts::PI;
        Self {
            kind: Kind::Single { xi, prefac },
        }
    }

    /// The valence proatom of element `z` with `n_core` core electrons
    /// removed; see the module doc. Integrates to `z − n_core`.
    pub fn valence(z: i32, n_core: i32) -> Self {
        // Neutral ground configuration by Madelung filling.
        let mut left = z.max(0) as u32;
        let mut occ: Vec<(u32, u32, u32)> = Vec::new(); // (n, l, electrons)
        for &(n, l) in &MADELUNG {
            if left == 0 {
                break;
            }
            let k = left.min(2 * (2 * l + 1));
            occ.push((n, l, k));
            left -= k;
        }
        // Slater's-rules exponent of each subshell on the full configuration.
        let zeta: Vec<f64> = occ
            .iter()
            .map(|&(n, l, _)| {
                let g = group(n, l);
                let mut s = 0.0;
                for &(n2, l2, k2) in &occ {
                    let g2 = group(n2, l2);
                    let k2 = f64::from(k2);
                    if g2 == g {
                        let same = if (n, l) == (n2, l2) { k2 - 1.0 } else { k2 };
                        s += same * if n == 1 { 0.30 } else { 0.35 };
                    } else if g2 < g {
                        // An s/p electron: 0.85 from shell n - 1, 1.00 from
                        // deeper. A d/f electron: 1.00 from every group
                        // to its left.
                        s += k2 * if l < 2 && n2 + 1 == n { 0.85 } else { 1.0 };
                    }
                }
                (f64::from(z) - s) / n_star(n)
            })
            .collect();
        // Remove the core from the innermost subshells, in (n, l) order.
        let mut order: Vec<usize> = (0..occ.len()).collect();
        order.sort_by_key(|&i| (occ[i].0, occ[i].1));
        let mut core = n_core.max(0) as u32;
        let mut valence = occ.iter().map(|o| o.2).collect::<Vec<u32>>();
        for &i in &order {
            if core == 0 {
                break;
            }
            let take = core.min(valence[i]);
            valence[i] -= take;
            core -= take;
        }
        let shells = occ
            .iter()
            .zip(&zeta)
            .zip(&valence)
            .filter(|(_, &nv)| nv > 0)
            .map(|((&(n, _, _), &zt), &nv)| {
                let ns = n_star(n);
                let a = 2.0 * zt;
                let ln_c = f64::from(nv).ln() + (2.0 * ns + 1.0) * a.ln()
                    - (4.0 * std::f64::consts::PI).ln()
                    - ln_gamma(2.0 * ns + 1.0);
                StoShell {
                    c: ln_c.exp(),
                    p: 2.0 * ns - 2.0,
                    a,
                }
            })
            .collect();
        Self {
            kind: Kind::Valence(shells),
        }
    }

    /// `(ρ⁰(r), dρ⁰/dr)` at distance `r >= 0` (Bohr).
    pub fn value_and_deriv(&self, r: f64) -> (f64, f64) {
        match &self.kind {
            Kind::Single { xi, prefac } => {
                let v = prefac * (-2.0 * xi * r).exp();
                (v, -2.0 * xi * v)
            }
            Kind::Valence(shells) => {
                let (mut v, mut d) = (0.0, 0.0);
                for s in shells {
                    let e = s.c * (-s.a * r).exp();
                    if s.p == 0.0 {
                        v += e;
                        d -= s.a * e;
                    } else if r > 0.0 {
                        let rp1 = r.powf(s.p - 1.0);
                        v += e * rp1 * r;
                        d += e * rp1 * (s.p - s.a * r);
                    }
                }
                (v, d)
            }
        }
    }

    /// ρ⁰ at distance `r` (Bohr).
    pub fn at(&self, r: f64) -> f64 {
        match &self.kind {
            Kind::Single { xi, prefac } => prefac * (-2.0 * xi * r).exp(),
            Kind::Valence(_) => self.value_and_deriv(r).0,
        }
    }

    /// Electrons the proatom holds (its exact integral).
    pub fn n_electrons(&self) -> f64 {
        match &self.kind {
            Kind::Single { xi, prefac } => prefac * std::f64::consts::PI / xi.powi(3),
            Kind::Valence(shells) => shells
                .iter()
                .map(|s| {
                    // 4π c Γ(p+3) / a^{p+3}
                    4.0 * std::f64::consts::PI
                        * s.c
                        * (ln_gamma(s.p + 3.0) - (s.p + 3.0) * s.a.ln()).exp()
                })
                .sum(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn atom(z: i32, n_core_ecp: i32) -> Atom {
        Atom {
            symbol: "X".into(),
            z,
            x: 0.0,
            y: 0.0,
            zpos: 0.0,
            ghost: false,
            n_core_ecp,
        }
    }

    /// 4π ∫ r² ρ dr by the trapezoid rule on a fine grid.
    fn integral(p: &SlaterProatom) -> f64 {
        let h = 1e-3;
        (1..=40_000)
            .map(|k| {
                let r = k as f64 * h;
                let w = if k == 40_000 { 0.5 } else { 1.0 };
                w * r * r * p.at(r) * h
            })
            .sum::<f64>()
            * 4.0
            * std::f64::consts::PI
    }

    #[test]
    fn ln_gamma_matches_factorials_and_half_integers() {
        for (x, want) in [(1.0, 1.0_f64), (5.0, 24.0), (9.0, 40320.0)] {
            assert!((ln_gamma(x) - want.ln()).abs() < 1e-13, "Γ({x})");
        }
        let half = std::f64::consts::PI.sqrt();
        assert!((ln_gamma(0.5) - half.ln()).abs() < 1e-13);
    }

    /// An all-electron atom keeps the single-exponential expression the
    /// routines evaluated inline before. Compared to 2 ulp here, not bitwise:
    /// with `z` known at compile time LLVM may constant-fold `xi.powi(3)`
    /// through the host `pow` instead of two runtime multiplies (seen as a
    /// 1-ulp difference in this test). The production call sites' bitwise
    /// identity is checked end to end (Python, float hex, against the
    /// previous build).
    #[test]
    fn all_electron_atom_is_the_single_exponential() {
        let close = |a: f64, b: f64| (a - b).abs() <= 2.0 * f64::EPSILON * b.abs();
        for z in [1, 8, 35, 53] {
            let p = SlaterProatom::for_atom(&atom(z, 0));
            let xi = slater_xi_for_z(z);
            let prefac = z as f64 * xi.powi(3) / std::f64::consts::PI;
            assert!((p.n_electrons() - f64::from(z)).abs() < 1e-12);
            for r in [0.0, 0.3, 1.7, 6.0] {
                let v = prefac * (-2.0 * xi * r).exp();
                let (pv, pd) = p.value_and_deriv(r);
                assert!(close(p.at(r), v), "Z={z} r={r}");
                assert!(close(pv, v), "Z={z} r={r}");
                assert!(close(pd, -2.0 * xi * v), "Z={z} r={r}");
            }
        }
    }

    /// Iodine with def2's 28-electron core: 25 valence electrons, a density
    /// that vanishes at the nucleus (no core cusp) and peaks in the valence
    /// region; the derivative is the exact one.
    #[test]
    fn iodine_valence_proatom_holds_the_valence_electrons() {
        let p = SlaterProatom::for_atom(&atom(53, 28));
        assert!(
            (p.n_electrons() - 25.0).abs() < 1e-12,
            "{}",
            p.n_electrons()
        );
        assert!((integral(&p) - 25.0).abs() < 1e-6, "{}", integral(&p));
        assert_eq!(p.at(0.0), 0.0);
        let peak = (1..4000)
            .map(|k| k as f64 * 1e-3)
            .max_by(|a, b| (a * a * p.at(*a)).total_cmp(&(b * b * p.at(*b))))
            .unwrap();
        assert!(peak > 0.2, "radial peak at {peak} Bohr");
        for r in [0.4, 1.1, 2.5, 5.0] {
            let h = 1e-6;
            let fd = (p.at(r + h) - p.at(r - h)) / (2.0 * h);
            let d = p.value_and_deriv(r).1;
            assert!(
                (fd - d).abs() <= 1e-6 * d.abs().max(1e-6),
                "r={r}: {d} vs {fd}"
            );
        }
    }

    /// Every closed-shell core count sits on a shell boundary in (n, l)
    /// order and leaves Z − n_core electrons.
    #[test]
    fn closed_shell_cores_leave_the_valence_count() {
        for (z, core) in [(37, 28), (53, 28), (55, 46), (79, 60), (86, 78), (35, 10)] {
            let p = SlaterProatom::valence(z, core);
            let want = f64::from(z - core);
            assert!((p.n_electrons() - want).abs() < 1e-10, "Z={z} core={core}");
        }
    }
}
