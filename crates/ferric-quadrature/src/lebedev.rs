//! Lebedev angular quadrature on the unit sphere.
//!
//! Returns `(points, weights)` where `points` are unit-vector directions
//! `(x, y, z)` and `weights` integrate `1/(4π)` on the sphere (i.e., they
//! sum to 1, not 4π — multiply by 4π·r² for spherical integration).
//!
//! Supported orders: 6, 14, 26, 50, 110, 302, 434, 590. These are the
//! standard tabulated Lebedev rules suitable for atomic integration (302 is
//! the Becke/Furche-default for production DFT; 434 and 590 integrate
//! spherical harmonics exactly through degree 35 and 41 respectively).
//!
//! Reference: Lebedev & Laikov, Russian Acad. Sci. Dokl. Math. 59, 477 (1999).
//! Tables transcribed from the canonical C source at
//! `https://people.sc.fsu.edu/~jburkardt/datasets/sphere_lebedev_rule/`.
//! Orders 434 and 590 were recovered from PySCF's compiled Lebedev-Laikov
//! generator (`pyscf.dft.gen_grid.MakeAngularGrid`); see the per-order
//! comments in [`lebedev`] for the cross-check.

/// Generate Lebedev nodes by group symmetry.
///
/// Lebedev rules expand a small set of "generator" points into the full
/// node set using the O_h symmetry of the unit sphere. The five generator
/// classes are:
///   * 6  vertices of an octahedron (a1)
///   * 8  vertices of a cube (a2)
///   * 12 mid-edge points (a3)
///   * 24 (b)  with one parameter — orbit of axis-perpendicular plane
///   * 24 (c)  with one parameter — orbit of arbitrary axis
///   * 48 (d)  with two parameters — fully general orbit
///
/// `weights` are uniform within each class; the table below gives the
/// per-class point count, weight, and parameter(s).
type Triplet = (f64, f64, f64);

fn class_a1(w: f64) -> Vec<(Triplet, f64)> {
    vec![
        ((1.0, 0.0, 0.0), w),
        ((-1.0, 0.0, 0.0), w),
        ((0.0, 1.0, 0.0), w),
        ((0.0, -1.0, 0.0), w),
        ((0.0, 0.0, 1.0), w),
        ((0.0, 0.0, -1.0), w),
    ]
}

fn class_a2(w: f64) -> Vec<(Triplet, f64)> {
    let s = (1.0_f64 / 3.0).sqrt();
    let mut v = Vec::with_capacity(8);
    for &sx in &[1.0, -1.0] {
        for &sy in &[1.0, -1.0] {
            for &sz in &[1.0, -1.0] {
                v.push(((sx * s, sy * s, sz * s), w));
            }
        }
    }
    v
}

fn class_a3(w: f64) -> Vec<(Triplet, f64)> {
    let s = (0.5_f64).sqrt();
    let mut v = Vec::with_capacity(12);
    for &sx in &[1.0, -1.0] {
        for &sy in &[1.0, -1.0] {
            v.push(((sx * s, sy * s, 0.0), w));
            v.push(((sx * s, 0.0, sy * s), w));
            v.push(((0.0, sx * s, sy * s), w));
        }
    }
    v
}

fn class_b(w: f64, p: f64) -> Vec<(Triplet, f64)> {
    // p is the parameter; the orbit is {(±p, ±p, ±q) and permutations} with
    // q = sqrt(1 - 2 p²).
    let q = (1.0 - 2.0 * p * p).max(0.0).sqrt();
    let mut v = Vec::with_capacity(24);
    for &sp1 in &[1.0_f64, -1.0] {
        for &sp2 in &[1.0_f64, -1.0] {
            for &sq in &[1.0_f64, -1.0] {
                v.push(((sp1 * p, sp2 * p, sq * q), w));
                v.push(((sp1 * p, sq * q, sp2 * p), w));
                v.push(((sq * q, sp1 * p, sp2 * p), w));
            }
        }
    }
    v
}

fn class_c(w: f64, p: f64) -> Vec<(Triplet, f64)> {
    // p is parameter; orbit is {(±p, ±q, 0) and permutations}, q = sqrt(1 - p²).
    let q = (1.0 - p * p).max(0.0).sqrt();
    let mut v = Vec::with_capacity(24);
    for &sp in &[1.0_f64, -1.0] {
        for &sq in &[1.0_f64, -1.0] {
            v.push(((sp * p, sq * q, 0.0), w));
            v.push(((sp * p, 0.0, sq * q), w));
            v.push(((0.0, sp * p, sq * q), w));
            v.push(((sq * q, sp * p, 0.0), w));
            v.push(((sq * q, 0.0, sp * p), w));
            v.push(((0.0, sq * q, sp * p), w));
        }
    }
    v
}

fn class_d(w: f64, p: f64, q: f64) -> Vec<(Triplet, f64)> {
    let r = (1.0 - p * p - q * q).max(0.0).sqrt();
    let mut v = Vec::with_capacity(48);
    for &sp in &[1.0_f64, -1.0] {
        for &sq in &[1.0_f64, -1.0] {
            for &sr in &[1.0_f64, -1.0] {
                // All 6 permutations of (p, q, r).
                let triplets = [
                    (sp * p, sq * q, sr * r),
                    (sp * p, sr * r, sq * q),
                    (sq * q, sp * p, sr * r),
                    (sq * q, sr * r, sp * p),
                    (sr * r, sp * p, sq * q),
                    (sr * r, sq * q, sp * p),
                ];
                for t in triplets {
                    v.push((t, w));
                }
            }
        }
    }
    v
}

/// Lebedev order 434 (see the comment inside for provenance).
fn rule_434() -> Vec<(Triplet, f64)> {
    // Degree 35. 6 (a1) + 8 (a2) + 12 (a3) + 7·24 (b) + 2·24 (c)
    // + 4·48 (d) = 434 points. Lebedev-Laikov (Dokl. Math. 59, 477
    // (1999)) parameters, recovered from PySCF 2.13.1's compiled
    // generator `pyscf.dft.gen_grid.MakeAngularGrid(434)` (libdft,
    // CxLebedevGrid) by grouping its nodes into O_h orbits and taking
    // one representative per orbit. Cross-check (2026-09-25): this
    // class list, expanded by the class_* generators above, reproduces
    // PySCF's 434 points and weights to max |Δ| = 1.1e-16 (both sets
    // sorted by coordinate).
    let mut v = class_a1(0.0005265897968224436_f64);
    v.extend(class_a2(0.002512317418927307_f64));
    v.extend(class_a3(0.002548219972002607_f64));
    // 7 b-orbits (24 pts each)
    v.extend(class_b(0.001462495621594614_f64, 0.07568084367178018_f64));
    v.extend(class_b(0.002014279020918528_f64, 0.1774836054609158_f64));
    v.extend(class_b(0.002530403801186355_f64, 0.6909346307509111_f64));
    v.extend(class_b(0.002302694782227416_f64, 0.2861289010307638_f64));
    v.extend(class_b(0.00244537343731298_f64, 0.3927259763368002_f64));
    v.extend(class_b(0.002513267174597564_f64, 0.6456664707424256_f64));
    v.extend(class_b(0.002501725168402936_f64, 0.4914342637784746_f64));
    // 2 c-orbits (24 pts each)
    v.extend(class_c(0.001910951282179532_f64, 0.2102725228573068_f64));
    v.extend(class_c(0.002417442375638981_f64, 0.471598691151316_f64));
    // 4 d-orbits (48 pts each)
    v.extend(class_d(
        0.002236607760437849_f64,
        0.09921769636429248_f64,
        0.3344363145343455_f64,
    ));
    v.extend(class_d(
        0.002512236854563495_f64,
        0.10680182607580488_f64,
        0.5905157048925271_f64,
    ));
    v.extend(class_d(
        0.002416930044324775_f64,
        0.2054823696403044_f64,
        0.4502330382582625_f64,
    ));
    v.extend(class_d(
        0.002496644054553086_f64,
        0.31042840351665446_f64,
        0.5550152361076807_f64,
    ));
    v
}

/// Lebedev order 590 (see the comment inside for provenance).
fn rule_590() -> Vec<(Triplet, f64)> {
    // Degree 41. 6 (a1) + 8 (a2) + 9·24 (b) + 3·24 (c) + 6·48 (d)
    // = 590 points (no a3 orbit). Same provenance and cross-check as
    // 434: recovered from `pyscf.dft.gen_grid.MakeAngularGrid(590)`
    // (PySCF 2.13.1); expanded here it reproduces PySCF's points and
    // weights to max |Δ| = 1.1e-16.
    let mut v = class_a1(0.0003095121295306187_f64);
    v.extend(class_a2(0.001852379698597489_f64));
    // 9 b-orbits (24 pts each)
    v.extend(class_b(0.000976433116505105_f64, 0.06095034115507196_f64));
    v.extend(class_b(0.001871790639277744_f64, 0.7040954938227469_f64));
    v.extend(class_b(0.001384737234851692_f64, 0.1459036449157763_f64));
    v.extend(class_b(0.001617210647254411_f64, 0.2384736701421887_f64));
    v.extend(class_b(0.001858812585438317_f64, 0.6807744066455244_f64));
    v.extend(class_b(0.001749564657281154_f64, 0.3317920736472123_f64));
    v.extend(class_b(0.001818471778162769_f64, 0.4215761784010967_f64));
    v.extend(class_b(0.001852028828296213_f64, 0.6372546939258752_f64));
    v.extend(class_b(0.001846715956151242_f64, 0.5044419707800358_f64));
    // 3 c-orbits (24 pts each)
    v.extend(class_c(0.001300321685886048_f64, 0.1724782009907724_f64));
    v.extend(class_c(0.001705153996395864_f64, 0.3964755348199858_f64));
    v.extend(class_c(0.001857161196774078_f64, 0.6116843442009876_f64));
    // 6 d-orbits (48 pts each)
    v.extend(class_d(
        0.001555213603396808_f64,
        0.08213021581932511_f64,
        0.2778673190586244_f64,
    ));
    v.extend(class_d(
        0.001802239128008525_f64,
        0.08999205842074876_f64,
        0.5033564271075117_f64,
    ));
    v.extend(class_d(
        0.001713904507106709_f64,
        0.1720795225656878_f64,
        0.3791035407695563_f64,
    ));
    v.extend(class_d(
        0.00184983056044366_f64,
        0.1816640840360209_f64,
        0.598412649788538_f64,
    ));
    v.extend(class_d(
        0.001802658934377451_f64,
        0.263471665593795_f64,
        0.474239284255198_f64,
    ));
    v.extend(class_d(
        0.001842866472905286_f64,
        0.3518280927733519_f64,
        0.561026380862206_f64,
    ));
    v
}

/// Return Lebedev `(unit-vectors, weights)` summing to 1 for the given order.
///
/// Supported: 6, 14, 26, 50, 110, 302, 434, 590. Panics on any other order.
pub fn lebedev(order: usize) -> (Vec<[f64; 3]>, Vec<f64>) {
    let raw: Vec<(Triplet, f64)> = match order {
        6 => class_a1(1.0 / 6.0),
        14 => {
            // 6 octahedron (w=1/15) + 8 cube (w=3/40)
            let mut v = class_a1(1.0 / 15.0);
            v.extend(class_a2(3.0 / 40.0));
            v
        }
        26 => {
            // 6 + 12 + 8
            let mut v = class_a1(1.0 / 21.0);
            v.extend(class_a3(4.0 / 105.0));
            v.extend(class_a2(27.0 / 840.0));
            v
        }
        50 => {
            // 6 + 12 + 8 + 24 (one b orbit, p ~ 0.30..)
            let mut v = class_a1(4.0 / 315.0);
            v.extend(class_a3(64.0 / 2835.0));
            v.extend(class_a2(27.0 / 1280.0));
            // 24-point class_b, parameter from Lebedev's tables for order-50:
            //   p ≈ 0.30151134457776 (canonical value)
            let p = 0.301511344577763955_f64;
            let w_b = 14641.0 / 725760.0;
            v.extend(class_b(w_b, p));
            v
        }
        110 => {
            // 6 (a1) + 8 (a2) + 3·24 (b) + 1·24 (c) = 110 points.
            // Parameters from Lebedev-Laikov canonical tables (Dokl. Math. 59,
            // 477 (1999)), as transcribed in PySCF's CxLebedevGrid.c. Ferric's
            // class_b corresponds to PySCF's case 3 (24 pts at (±a,±a,±b),
            // b=√(1-2a²)); class_c corresponds to PySCF's case 4 (24 pts at
            // (±a,±b,0), b=√(1-a²)).
            let mut v = class_a1(0.3828270494937162e-2_f64);
            v.extend(class_a2(0.9793737512487512e-2_f64));
            v.extend(class_b(0.8211737283191111e-2_f64, 0.1851156353447362_f64));
            v.extend(class_b(0.9942814891178103e-2_f64, 0.6904210483822922_f64));
            v.extend(class_b(0.9595471336070963e-2_f64, 0.3956894730559419_f64));
            v.extend(class_c(0.9694996361663028e-2_f64, 0.4783690288121502_f64));
            v
        }
        302 => {
            // 6 (a1) + 8 (a2) + 6·24 (b) + 2·24 (c) + 2·48 (d) = 302 points.
            // Parameters from Lebedev-Laikov canonical tables (Dokl. Math. 59,
            // 477 (1999)), as transcribed in PySCF's CxLebedevGrid::MakeAngularGrid_302.
            // Ferric's class_d corresponds to PySCF's case 5 (48 points at
            // (±a, ±b, ±c) with c = √(1-a²-b²)).
            let mut v = class_a1(0.8545911725128148e-3_f64);
            v.extend(class_a2(0.3599119285025571e-2_f64));
            // 6 b-orbits (case 3: 24 pts each)
            v.extend(class_b(0.3449788424305883e-2_f64, 0.3515640345570105_f64));
            v.extend(class_b(0.3604822601419882e-2_f64, 0.6566329410219612_f64));
            v.extend(class_b(0.3576729661743367e-2_f64, 0.4729054132581005_f64));
            v.extend(class_b(0.2352101413689164e-2_f64, 0.09618308522614784_f64));
            v.extend(class_b(0.3108953122413675e-2_f64, 0.2219645236294178_f64));
            v.extend(class_b(0.3650045807677255e-2_f64, 0.7011766416089545_f64));
            // 2 c-orbits (case 4: 24 pts each)
            v.extend(class_c(0.2982344963171804e-2_f64, 0.2644152887060663_f64));
            v.extend(class_c(0.3600820932216460e-2_f64, 0.5718955891878961_f64));
            // 2 d-orbits (case 5: 48 pts each), two-parameter (a, b)
            v.extend(class_d(
                0.3571540554273387e-2_f64,
                0.2510034751770465_f64,
                0.8000727494073952_f64,
            ));
            v.extend(class_d(
                0.3392312205006170e-2_f64,
                0.1233548532583327_f64,
                0.4127724083168531_f64,
            ));
            v
        }
        434 => rule_434(),
        590 => rule_590(),
        _ => panic!("lebedev: unsupported order {order} (try 6, 14, 26, 50, 110, 302, 434, 590)"),
    };
    let mut pts = Vec::with_capacity(raw.len());
    let mut wts = Vec::with_capacity(raw.len());
    for ((x, y, z), w) in raw {
        pts.push([x, y, z]);
        wts.push(w);
    }
    (pts, wts)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check_weight_sum(order: usize, tol: f64) {
        let (pts, wts) = lebedev(order);
        let sum: f64 = wts.iter().sum();
        assert!(
            (sum - 1.0).abs() < tol,
            "order {order}: weight sum {sum} != 1 (n_pts={})",
            pts.len()
        );
    }

    fn check_unit_norm(order: usize) {
        let (pts, _) = lebedev(order);
        for p in &pts {
            let n2: f64 = p.iter().map(|x| x * x).sum();
            assert!(
                (n2 - 1.0).abs() < 1e-12,
                "non-unit point {:?}, |·|²={n2}",
                p
            );
        }
    }

    #[test]
    fn lebedev_6_sum_and_norm() {
        check_weight_sum(6, 1e-12);
        check_unit_norm(6);
    }
    #[test]
    fn lebedev_14_sum_and_norm() {
        check_weight_sum(14, 1e-12);
        check_unit_norm(14);
    }
    #[test]
    fn lebedev_26_sum_and_norm() {
        check_weight_sum(26, 1e-12);
        check_unit_norm(26);
    }
    #[test]
    fn lebedev_50_sum_and_norm() {
        check_weight_sum(50, 1e-10);
        check_unit_norm(50);
    }
    #[test]
    fn lebedev_110_sum_and_norm() {
        check_weight_sum(110, 1e-9);
        check_unit_norm(110);
    }

    #[test]
    fn lebedev_434_sum_and_norm() {
        let (pts, _) = lebedev(434);
        assert_eq!(pts.len(), 434);
        check_weight_sum(434, 1e-13);
        check_unit_norm(434);
    }
    #[test]
    fn lebedev_590_sum_and_norm() {
        let (pts, _) = lebedev(590);
        assert_eq!(pts.len(), 590);
        check_weight_sum(590, 1e-13);
        check_unit_norm(590);
    }

    /// (n-1)!! style double factorial in f64; `double_factorial(-1) = 1`.
    fn double_factorial(n: i64) -> f64 {
        let mut r = 1.0;
        let mut k = n;
        while k > 1 {
            r *= k as f64;
            k -= 2;
        }
        r
    }

    /// Exact ∫ x^a y^b z^c dΩ / (4π): zero unless a, b, c are all even,
    /// else (a−1)!!(b−1)!!(c−1)!! / (a+b+c+1)!!.
    fn sphere_monomial_mean(a: i32, b: i32, c: i32) -> f64 {
        if a % 2 != 0 || b % 2 != 0 || c % 2 != 0 {
            return 0.0;
        }
        double_factorial((a - 1) as i64)
            * double_factorial((b - 1) as i64)
            * double_factorial((c - 1) as i64)
            / double_factorial((a + b + c + 1) as i64)
    }

    /// Worst quadrature error over every monomial x^a y^b z^c of total degree
    /// exactly `deg`. The error is RELATIVE where the exact integral is
    /// nonzero (high-degree monomial integrals are small, ~1e-2..1e-8, so an
    /// absolute bar would be blind) and absolute where it is zero.
    fn worst_monomial_error(order: usize, deg: i32) -> (f64, (i32, i32, i32)) {
        let (pts, wts) = lebedev(order);
        let mut worst = (0.0_f64, (0, 0, 0));
        for a in 0..=deg {
            for b in 0..=(deg - a) {
                let c = deg - a - b;
                let q: f64 = pts
                    .iter()
                    .zip(wts.iter())
                    .map(|(p, w)| w * p[0].powi(a) * p[1].powi(b) * p[2].powi(c))
                    .sum();
                let exact = sphere_monomial_mean(a, b, c);
                let err = if exact != 0.0 {
                    ((q - exact) / exact).abs()
                } else {
                    q.abs()
                };
                if err > worst.0 {
                    worst = (err, (a, b, c));
                }
            }
        }
        worst
    }

    /// Exactness through `degree` (every monomial of every total degree
    /// 0..=degree) plus a negative control at `degree + 1`. Odd total degrees
    /// integrate to zero by inversion symmetry for ANY centrosymmetric rule,
    /// so the control must be the next EVEN degree; degree is odd for all
    /// Lebedev rules, so degree + 1 is even.
    ///
    /// Measured with PySCF's own 434/590 nodes (2026-09-25): exact through
    /// degree worst rel err ≤ 2.1e-15; at degree + 1 the worst rel err is
    /// 8.3e-4 (302), 1.1e-4 (434), 1.5e-5 (590). The 1e-12 / 1e-7 bars sit
    /// between the two sides.
    fn check_monomial_exactness(order: usize, degree: i32) {
        for deg in 0..=degree {
            let (err, abc) = worst_monomial_error(order, deg);
            assert!(
                err < 1e-12,
                "Lebedev-{order} should integrate degree-{deg} monomial x^{}y^{}z^{} exactly, err {err:.3e}",
                abc.0,
                abc.1,
                abc.2
            );
        }
        let (err, abc) = worst_monomial_error(order, degree + 1);
        assert!(
            err > 1e-7,
            "negative control: Lebedev-{order} is only degree {degree}, yet every degree-{} monomial \
             integrated to within {err:.3e} (worst x^{}y^{}z^{}) — the exactness check cannot fail",
            degree + 1,
            abc.0,
            abc.1,
            abc.2
        );
    }

    #[test]
    fn lebedev_302_integrates_monomials_through_degree_29() {
        check_monomial_exactness(302, 29);
    }
    #[test]
    fn lebedev_434_integrates_monomials_through_degree_35() {
        check_monomial_exactness(434, 35);
    }
    #[test]
    fn lebedev_590_integrates_monomials_through_degree_41() {
        check_monomial_exactness(590, 41);
    }

    /// Lebedev integrates spherical harmonics Y_lm exactly up to degree L.
    /// For order 110, L=15 — so r² Y_2m should integrate to 0 over the sphere.
    #[test]
    fn lebedev_110_integrates_y22_to_zero() {
        let (pts, wts) = lebedev(110);
        // Y_2,2 ∝ x² - y² (real form). ∫ Y_2,2 = 0.
        let int_y22: f64 = pts
            .iter()
            .zip(wts.iter())
            .map(|(p, w)| w * (p[0] * p[0] - p[1] * p[1]))
            .sum();
        assert!(
            int_y22.abs() < 1e-10,
            "Lebedev-110 should integrate Y_2,2 to 0, got {int_y22:.3e}"
        );
    }
}
