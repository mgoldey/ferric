//! Lebedev angular quadrature on the unit sphere.
//!
//! Returns `(points, weights)` where `points` are unit-vector directions
//! `(x, y, z)` and `weights` integrate `1/(4π)` on the sphere (i.e., they
//! sum to 1, not 4π — multiply by 4π·r² for spherical integration).
//!
//! Supported orders ([`SUPPORTED_ORDERS`]): every Lebedev-Laikov rule PySCF
//! builds, 6 through 5810 points (algebraic degree 3 through 131). 302 is
//! the Becke/Furche-default for production DFT; 194 is the valence order of
//! the pruned COSX exchange grids, as in ORCA's GridX3 and PySCF's SGX
//! level 2. Three rules carry negative weights: 74 (degree 13), 230
//! (degree 25) and 266 (degree 27); [`NEGATIVE_WEIGHT_ORDERS`] lists them.
//!
//! Reference: Lebedev & Laikov, Russian Acad. Sci. Dokl. Math. 59, 477 (1999).
//!
//! Provenance and licence. ferric is MIT OR Apache-2.0. Orders 6-110 and 302
//! carry the published Lebedev-Laikov parameters as written out in the match
//! arms of [`try_lebedev`]; orders 194, 434 and 590 were recovered from
//! PySCF's compiled Lebedev-Laikov generator (`pyscf.dft.gen_grid.MakeAngularGrid`)
//! as described in their builders. Every other order comes from the const
//! tables in `lebedev/tables.rs`, which `scripts/gen_lebedev_tables.py`
//! writes from the same PySCF generator's OUTPUT (PySCF is Apache-2.0): its
//! nodes are grouped into O_h orbits and one (weight, parameters) row is
//! kept per orbit. No generator code is copied from PySCF or from any other
//! Lebedev implementation; numbers are the published rule parameters, and
//! the `class_*` expanders below are ferric's own. The same script writes
//! `testdata/reference/lebedev_pyscf_orbits.json`, against which the tests
//! compare every order to 1e-14.

use ferric_core::FerricError;

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

/// Lebedev order 194 (see the comment inside for provenance).
fn rule_194() -> Vec<(Triplet, f64)> {
    // Degree 23. 6 (a1) + 8 (a2) + 12 (a3) + 4·24 (b) + 1·24 (c) + 1·48 (d)
    // = 194 points. Lebedev-Laikov (Dokl. Math. 59, 477 (1999)) LD0194
    // parameters, read from PySCF 2.13.1's compiled generator
    // `pyscf.dft.gen_grid.MakeAngularGrid(194)` by grouping its nodes into O_h
    // orbits (one representative per orbit, full repr precision). They agree
    // digit for digit with the published LD0194 table (Lebedev-Laikov C
    // source: V = 0.1782340447244611e-2 / 0.5716905949977102e-2 /
    // 0.5573383178848738e-2; A = 0.6712973442695226, 0.2892465627575439,
    // 0.4446933178717437, 0.1299335447650067; A = 0.3457702197611283;
    // A, B = 0.1590417105383530, 0.8360360154824589). Cross-check
    // (2026-10-01): a numpy transcription of the class_* expansion with these
    // parameters reproduces PySCF's 194 points and weights exactly (max |Δ| =
    // 0 with both sets sorted by coordinate); exactness of THIS code is pinned
    // by `lebedev_194_integrates_monomials_through_degree_23`.
    //
    // Orbit naming: ferric's class_a2 is the 8-point cube (Lebedev-Laikov's
    // code 3) and class_a3 the 12 edge midpoints (code 2); class_b is code 4
    // (±a, ±a, ±b), class_c code 5 (±a, ±b, 0), class_d code 6.
    let mut v = class_a1(0.001782340447244611_f64);
    v.extend(class_a2(0.005573383178848738_f64));
    v.extend(class_a3(0.005716905949977102_f64));
    // 4 b-orbits (24 pts each)
    v.extend(class_b(0.005608704082587997_f64, 0.6712973442695226_f64));
    v.extend(class_b(0.005158237711805383_f64, 0.2892465627575439_f64));
    v.extend(class_b(0.005518771467273614_f64, 0.4446933178717437_f64));
    v.extend(class_b(0.004106777028169394_f64, 0.1299335447650067_f64));
    // 1 c-orbit (24 pts)
    v.extend(class_c(0.005051846064614808_f64, 0.3457702197611283_f64));
    // 1 d-orbit (48 pts)
    v.extend(class_d(
        0.005530248916233094_f64,
        0.159041710538353_f64,
        0.8360360154824589_f64,
    ));
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

mod tables;

/// One Lebedev-Laikov rule as orbit parameters: the optional single orbits
/// of the octahedron vertices (`a1`), cube vertices (`a2`) and edge midpoints
/// (`a3`) by weight, then `[weight, p]` rows for [`class_b`] and [`class_c`]
/// and `[weight, p, q]` rows for [`class_d`].
struct Rule {
    order: usize,
    /// Algebraic degree; read by the tests.
    #[cfg_attr(not(test), allow(dead_code))]
    degree: usize,
    a1: Option<f64>,
    a2: Option<f64>,
    a3: Option<f64>,
    b: &'static [[f64; 2]],
    c: &'static [[f64; 2]],
    d: &'static [[f64; 3]],
}

impl Rule {
    fn expand(&self) -> Vec<(Triplet, f64)> {
        let mut v = Vec::with_capacity(self.order);
        if let Some(w) = self.a1 {
            v.extend(class_a1(w));
        }
        if let Some(w) = self.a2 {
            v.extend(class_a2(w));
        }
        if let Some(w) = self.a3 {
            v.extend(class_a3(w));
        }
        for &[w, p] in self.b {
            v.extend(class_b(w, p));
        }
        for &[w, p] in self.c {
            v.extend(class_c(w, p));
        }
        for &[w, p, q] in self.d {
            v.extend(class_d(w, p, q));
        }
        v
    }
}

/// The orders [`lebedev`] and [`try_lebedev`] tabulate, ascending.
pub const SUPPORTED_ORDERS: [usize; 32] = [
    6, 14, 26, 38, 50, 74, 86, 110, 146, 170, 194, 230, 266, 302, 350, 434, 590, 770, 974, 1202,
    1454, 1730, 2030, 2354, 2702, 3074, 3470, 3890, 4334, 4802, 5294, 5810,
];

/// The supported orders whose rule has negative weights (Lebedev-Laikov
/// 74, 230 and 266). A consumer that reads a weight as an area, such as a
/// solvation cavity's tesserae, must refuse these.
pub const NEGATIVE_WEIGHT_ORDERS: [usize; 3] = [74, 230, 266];

/// [`SUPPORTED_ORDERS`] without [`NEGATIVE_WEIGHT_ORDERS`], ascending: the
/// rules whose weights are all positive.
pub const POSITIVE_WEIGHT_ORDERS: [usize; 29] = [
    6, 14, 26, 38, 50, 86, 110, 146, 170, 194, 302, 350, 434, 590, 770, 974, 1202, 1454, 1730,
    2030, 2354, 2702, 3074, 3470, 3890, 4334, 4802, 5294, 5810,
];

/// Return Lebedev `(unit-vectors, weights)` summing to 1 for the given order.
///
/// Supported: [`SUPPORTED_ORDERS`]. Panics on any other order; use
/// [`try_lebedev`] where the order comes from user input.
pub fn lebedev(order: usize) -> (Vec<[f64; 3]>, Vec<f64>) {
    try_lebedev(order).unwrap_or_else(|e| panic!("{e}"))
}

/// [`lebedev`] returning an error for an order outside [`SUPPORTED_ORDERS`].
///
/// # Errors
///
/// `FerricError::General` naming the order and the supported set.
pub fn try_lebedev(order: usize) -> Result<(Vec<[f64; 3]>, Vec<f64>), FerricError> {
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
        194 => rule_194(),
        434 => rule_434(),
        590 => rule_590(),
        _ => match tables::RULES.iter().find(|r| r.order == order) {
            Some(rule) => rule.expand(),
            None => {
                let listed: Vec<String> = SUPPORTED_ORDERS.iter().map(usize::to_string).collect();
                return Err(FerricError::General(format!(
                    "lebedev: unsupported order {order} (try {})",
                    listed.join(", ")
                )));
            }
        },
    };
    let mut pts = Vec::with_capacity(raw.len());
    let mut wts = Vec::with_capacity(raw.len());
    for ((x, y, z), w) in raw {
        pts.push([x, y, z]);
        wts.push(w);
    }
    Ok((pts, wts))
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

    /// Lebedev-194 is degree 23. Measured 2026-10-01: worst relative error
    /// through degree 23 is at the f64 floor; at degree 24 it is O(1e-3)
    /// (the negative control in `check_monomial_exactness`).
    #[test]
    fn lebedev_194_integrates_monomials_through_degree_23() {
        check_monomial_exactness(194, 23);
    }
    #[test]
    fn lebedev_194_sum_and_norm() {
        check_weight_sum(194, 1e-14);
        check_unit_norm(194);
        assert_eq!(lebedev(194).0.len(), 194);
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

    /// Every supported order builds, with as many points as its name; the
    /// orders between them are refused by `try_lebedev` and panic in
    /// `lebedev`.
    #[test]
    fn try_lebedev_accepts_exactly_the_supported_orders() {
        for order in 0..=SUPPORTED_ORDERS[SUPPORTED_ORDERS.len() - 1] + 1 {
            match try_lebedev(order) {
                Ok((pts, wts)) => {
                    assert!(SUPPORTED_ORDERS.contains(&order), "order {order} accepted");
                    assert_eq!((pts.len(), wts.len()), (order, order));
                }
                Err(e) => {
                    assert!(!SUPPORTED_ORDERS.contains(&order), "order {order}: {e}");
                    assert!(e
                        .to_string()
                        .contains(&format!("unsupported order {order}")));
                }
            }
        }
    }

    /// `(order, degree)` of every supported rule: the hand-written arms'
    /// degrees, then each table rule's own.
    fn all_degrees() -> Vec<(usize, usize)> {
        let hand = [
            (6, 3),
            (14, 5),
            (26, 7),
            (50, 11),
            (110, 17),
            (194, 23),
            (302, 29),
            (434, 35),
            (590, 41),
        ];
        let mut v: Vec<(usize, usize)> = hand.to_vec();
        v.extend(tables::RULES.iter().map(|r| (r.order, r.degree)));
        v.sort_unstable();
        v
    }

    /// Exactness on the all-even monomials x^2i y^2j z^2l of every total
    /// degree through `degree` (relative error, since the exact integrals are
    /// small), then zonal harmonics `P_l(u·r)` about four fixed generic axes
    /// `u` for every `l` through `degree` (absolute: `|P_l| <= 1`, exact
    /// integral 0 for `l >= 1`) and, as the negative control, at
    /// `l = degree + 1`. Odd-exponent monomials are covered by
    /// `every_order_is_closed_under_sign_flips` (a sign-flip-closed rule with
    /// equal partner weights sums each of them to zero pairwise).
    /// Precomputed even powers keep the 5810-point, degree-131 check cheap;
    /// the generic `check_monomial_exactness` stays for the hand-written
    /// orders.
    ///
    /// The control is a zonal harmonic, not a monomial: at high degree a
    /// harmonic's monomial expansion cancels heavily, so a rule that misses
    /// `Y_{L+1}` by O(1e-4) still integrates every degree-(L+1) monomial to
    /// ~1e-14 relative (measured: 3890-5810 at 1.4e-14 to 2.3e-14), and a
    /// monomial control cannot fail.
    ///
    /// Bars (measured over the 23 table orders, 2026-11-12; each test prints
    /// its own `MEASURED` line): worst in-degree monomial relative error
    /// 5.1e-14 (4334) against the 1e-12 bar, worst in-degree zonal sum
    /// 4.1e-16 against 1e-13; the smallest control is 5.7e-3 (2354) against
    /// 1e-7.
    fn check_even_monomial_exactness(order: usize, degree: usize) -> (f64, f64, f64) {
        let (pts, wts) = lebedev(order);
        let kmax = degree / 2;
        let pow = |c: usize| -> Vec<Vec<f64>> {
            pts.iter()
                .map(|p| {
                    let s = p[c] * p[c];
                    let mut row = vec![1.0; kmax + 1];
                    for k in 1..=kmax {
                        row[k] = row[k - 1] * s;
                    }
                    row
                })
                .collect()
        };
        let (xs, ys, zs) = (pow(0), pow(1), pow(2));
        let mut inside = 0.0_f64;
        for k in 0..=kmax {
            for i in 0..=k {
                for j in 0..=(k - i) {
                    let l = k - i - j;
                    let q: f64 = (0..pts.len())
                        .map(|g| wts[g] * xs[g][i] * ys[g][j] * zs[g][l])
                        .sum();
                    let exact = sphere_monomial_mean(2 * i as i32, 2 * j as i32, 2 * l as i32);
                    let err = ((q - exact) / exact).abs();
                    assert!(
                        err < 1e-12,
                        "Lebedev-{order} should integrate x^{}y^{}z^{} exactly, rel err {err:.3e}",
                        2 * i,
                        2 * j,
                        2 * l
                    );
                    inside = inside.max(err);
                }
            }
        }
        // Zonal harmonics: S_l = sum_g w_g P_l(u . r_g), Legendre recurrence.
        let axes: [[f64; 3]; 4] = [
            [1.0, 0.3, 0.1],
            [0.2, 0.7, -0.5],
            [-0.6, 0.15, 0.8],
            [0.35, -0.9, 0.25],
        ];
        let lmax = degree + 1;
        let mut zonal_inside = 0.0_f64;
        let mut control = 0.0_f64;
        for u in axes {
            let n = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt();
            let u = u.map(|c| c / n);
            let mut sums = vec![0.0_f64; lmax + 1];
            for (p, &w) in pts.iter().zip(&wts) {
                let t = p[0] * u[0] + p[1] * u[1] + p[2] * u[2];
                let (mut pm, mut pl) = (1.0_f64, t);
                sums[0] += w;
                sums[1] += w * t;
                for l in 1..lmax {
                    let next = ((2 * l + 1) as f64 * t * pl - l as f64 * pm) / (l + 1) as f64;
                    pm = pl;
                    pl = next;
                    sums[l + 1] += w * pl;
                }
            }
            for (l, &s_l) in sums.iter().enumerate().take(degree + 1).skip(1) {
                assert!(
                    s_l.abs() < 1e-13,
                    "Lebedev-{order} should integrate P_{l}(u.r) to 0, got {s_l:.3e} (u {u:?})"
                );
                zonal_inside = zonal_inside.max(s_l.abs());
            }
            control = control.max(sums[lmax].abs());
        }
        assert!(
            control > 1e-7,
            "negative control: Lebedev-{order} is only degree {degree}, yet P_{lmax}(u.r) \
             integrated to within {control:.3e} about every axis"
        );
        (inside, zonal_inside, control)
    }

    macro_rules! exactness_tests {
        ($($name:ident: $order:expr, $degree:expr;)*) => {$(
            #[test]
            fn $name() {
                let (pts, _) = lebedev($order);
                assert_eq!(pts.len(), $order);
                check_weight_sum($order, 1e-13);
                check_unit_norm($order);
                let (inside, zonal, control) = check_even_monomial_exactness($order, $degree);
                println!(
                    "MEASURED {}: monomial rel {inside:.2e}, zonal {zonal:.2e} through degree {}; \
                     zonal {control:.2e} at {}",
                    $order, $degree, $degree + 1
                );
            }
        )*};
    }

    exactness_tests! {
        lebedev_38_is_exact_through_degree_9: 38, 9;
        lebedev_74_is_exact_through_degree_13: 74, 13;
        lebedev_86_is_exact_through_degree_15: 86, 15;
        lebedev_146_is_exact_through_degree_19: 146, 19;
        lebedev_170_is_exact_through_degree_21: 170, 21;
        lebedev_230_is_exact_through_degree_25: 230, 25;
        lebedev_266_is_exact_through_degree_27: 266, 27;
        lebedev_350_is_exact_through_degree_31: 350, 31;
        lebedev_770_is_exact_through_degree_47: 770, 47;
        lebedev_974_is_exact_through_degree_53: 974, 53;
        lebedev_1202_is_exact_through_degree_59: 1202, 59;
        lebedev_1454_is_exact_through_degree_65: 1454, 65;
        lebedev_1730_is_exact_through_degree_71: 1730, 71;
        lebedev_2030_is_exact_through_degree_77: 2030, 77;
        lebedev_2354_is_exact_through_degree_83: 2354, 83;
        lebedev_2702_is_exact_through_degree_89: 2702, 89;
        lebedev_3074_is_exact_through_degree_95: 3074, 95;
        lebedev_3470_is_exact_through_degree_101: 3470, 101;
        lebedev_3890_is_exact_through_degree_107: 3890, 107;
        lebedev_4334_is_exact_through_degree_113: 4334, 113;
        lebedev_4802_is_exact_through_degree_119: 4802, 119;
        lebedev_5294_is_exact_through_degree_125: 5294, 125;
        lebedev_5810_is_exact_through_degree_131: 5810, 131;
    }

    /// The macro list above, the table rules and [`SUPPORTED_ORDERS`] name
    /// the same orders and degrees; the even-monomial check also holds for
    /// the hand-written orders.
    #[test]
    fn degrees_cover_every_supported_order() {
        let degs = all_degrees();
        let orders: Vec<usize> = degs.iter().map(|d| d.0).collect();
        assert_eq!(orders, SUPPORTED_ORDERS.to_vec());
        for (order, degree) in degs {
            // Lebedev-Laikov degree ladder: 3, 5, 7, then odd steps.
            assert_eq!(degree % 2, 1, "order {order}");
        }
        for (order, degree) in [(194, 23), (302, 29), (434, 35), (590, 41)] {
            check_even_monomial_exactness(order, degree);
        }
    }

    /// Every rule is closed under each coordinate sign flip with equal
    /// partner weights, so each monomial with an odd exponent sums to zero.
    #[test]
    fn every_order_is_closed_under_sign_flips() {
        for order in SUPPORTED_ORDERS {
            let (pts, wts) = lebedev(order);
            let key = |p: [f64; 3]| p.map(|x| (x * 1e12).round() as i64);
            let index: std::collections::HashMap<[i64; 3], f64> =
                pts.iter().zip(&wts).map(|(p, &w)| (key(*p), w)).collect();
            assert_eq!(index.len(), order, "order {order}: repeated points");
            for (p, &w) in pts.iter().zip(&wts) {
                for axis in 0..3 {
                    let mut f = *p;
                    f[axis] = -f[axis];
                    let partner = index.get(&key(f)).copied();
                    assert_eq!(
                        partner,
                        Some(w),
                        "order {order}: {p:?} flipped on axis {axis}"
                    );
                }
            }
        }
    }

    /// One orbit of a rule in the reference layout: canonical `(a, b, c)`
    /// with `a >= b >= c >= 0`, weight, point count.
    type Orbit = ([f64; 3], f64, usize);

    /// Reduce a rule to its orbits, the same reduction the generator applies
    /// to PySCF's nodes.
    fn canonical_orbits(order: usize) -> Vec<Orbit> {
        let (pts, wts) = lebedev(order);
        let mut canon: Vec<([f64; 3], f64)> = pts
            .iter()
            .zip(&wts)
            .map(|(p, &w)| {
                let mut a = p.map(f64::abs);
                a.sort_by(|x, y| y.total_cmp(x));
                (a, w)
            })
            .collect();
        canon.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap());
        let mut out: Vec<Orbit> = Vec::new();
        for (a, w) in canon {
            match out.last_mut() {
                Some((rep, rw, n)) if (0..3).all(|k| (rep[k] - a[k]).abs() < 1e-12) => {
                    assert_eq!(
                        *rw, w,
                        "order {order}: weights differ within the orbit of {rep:?}"
                    );
                    *n += 1;
                }
                _ => out.push((a, w, 1)),
            }
        }
        out
    }

    /// Every order equals PySCF's rule (`MakeAngularGrid`) as orbit lists:
    /// same orbits, same point counts, coordinates and weights within 1e-14.
    /// Both rules consist of whole O_h orbits with one weight each (the
    /// generator asserts it for PySCF; `every_order_is_closed_under_sign_flips`
    /// and the per-orbit weight assert here for ferric), so this is the
    /// comparison of the sorted point and weight sets.
    ///
    /// Measured 2026-11-12: max |Δ coordinate| 3.9e-16, max |Δ weight| / w
    /// 3.3e-16 over all 32 orders.
    #[test]
    fn every_order_matches_pyscf() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../testdata/reference/lebedev_pyscf_orbits.json");
        let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path:?}: {e}"));
        let json: serde_json::Value = serde_json::from_str(&text).unwrap();
        let orders = json["orders"].as_object().unwrap();
        let mut listed: Vec<usize> = orders.keys().map(|k| k.parse().unwrap()).collect();
        listed.sort_unstable();
        assert_eq!(listed, SUPPORTED_ORDERS.to_vec());
        let (mut dx, mut dw) = (0.0_f64, 0.0_f64);
        for (order, degree) in all_degrees() {
            let entry = &orders[&order.to_string()];
            assert_eq!(
                entry["degree"].as_u64(),
                Some(degree as u64),
                "order {order}"
            );
            let reference: Vec<Orbit> = entry["orbits"]
                .as_array()
                .unwrap()
                .iter()
                .map(|o| {
                    let f = |k: usize| o[k].as_f64().unwrap();
                    ([f(0), f(1), f(2)], f(3), o[4].as_u64().unwrap() as usize)
                })
                .collect();
            let ours = canonical_orbits(order);
            assert_eq!(ours.len(), reference.len(), "order {order}: orbit count");
            for (o, r) in ours.iter().zip(&reference) {
                assert_eq!(
                    o.2, r.2,
                    "order {order}: orbit {:?} has {} points, PySCF {}",
                    r.0, o.2, r.2
                );
                for k in 0..3 {
                    dx = dx.max((o.0[k] - r.0[k]).abs());
                }
                dw = dw.max(((o.1 - r.1) / r.1).abs());
                assert!(
                    (0..3).all(|k| (o.0[k] - r.0[k]).abs() < 1e-14)
                        && ((o.1 - r.1) / r.1).abs() < 1e-14,
                    "order {order}: orbit {:?} w {:e} vs PySCF {:?} w {:e}",
                    o.0,
                    o.1,
                    r.0,
                    r.1
                );
            }
        }
        println!("max |dx| {dx:.2e}, max rel |dw| {dw:.2e}");
    }

    /// Exactly [`NEGATIVE_WEIGHT_ORDERS`] have a negative weight, and
    /// [`POSITIVE_WEIGHT_ORDERS`] is the rest.
    #[test]
    fn negative_weight_orders_are_exactly_listed() {
        for order in SUPPORTED_ORDERS {
            let negative = lebedev(order).1.iter().any(|&w| w < 0.0);
            assert_eq!(
                negative,
                NEGATIVE_WEIGHT_ORDERS.contains(&order),
                "order {order}"
            );
            assert_eq!(
                !negative,
                POSITIVE_WEIGHT_ORDERS.contains(&order),
                "order {order}"
            );
        }
        assert_eq!(
            POSITIVE_WEIGHT_ORDERS.len() + NEGATIVE_WEIGHT_ORDERS.len(),
            SUPPORTED_ORDERS.len()
        );
        assert!(POSITIVE_WEIGHT_ORDERS.windows(2).all(|p| p[0] < p[1]));
    }

    #[test]
    #[should_panic(expected = "lebedev: unsupported order 7")]
    fn lebedev_panics_on_an_unsupported_order() {
        let _ = lebedev(7);
    }
}
