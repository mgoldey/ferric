//! Continuous Fast Multipole Method (CFMM) Coulomb builder.
//!
//! Reference: White & Head-Gordon, J. Chem. Phys. 101, 6593 (1994).
//!
//! # Structure
//!
//! J is split into a NEAR field (exact 4-center ERIs between shell pairs in
//! adjacent leaf boxes) and a FAR field (multipole expansion of the charge
//! distribution in well-separated boxes, translated to a local Taylor
//! expansion of the potential and contracted against the local shell-pair
//! multipoles).
//!
//! ```text
//!   J_{μν} = Σ_{λσ} (μν|λσ) D_{λσ}
//!          = [near: exact ERIs]  +  [far: Σ_ijk L_ijk(box) · M^{μν}_ijk]
//! ```
//!
//! # What gets binned: shell PAIRS, not shells
//!
//! The charge distribution in `J` is the set of shell-pair products
//! `Ω_{s1,s2} = χ_{s1} χ_{s2}`, centered at the Gaussian product center —
//! which generally lies in a different box than either shell. The octree
//! therefore bins PAIRS by product center (see [`ShellPair`]). Binning
//! individual shells instead silently drops every quartet whose bra pair
//! straddles two boxes; the trivial-limit anchor catches that immediately.
//!
//! # Conventions used throughout this file
//!
//! * **Multipole moments** are *primitive Cartesian* (not solid-harmonic):
//!   `M_ijk = ∫ ρ(r) (x−Cx)^i (y−Cy)^j (z−Cz)^k dr`. This matches the
//!   `shift_cartesian` binomial translation already in this module.
//! * **Local expansions** are Taylor coefficients of the potential:
//!   `Φ(r) ≈ Σ_ijk L_ijk (x−Cx)^i (y−Cy)^j (z−Cz)^k`, so the far-field
//!   energy contraction is `Σ_ijk L_ijk M_ijk` with NO extra factorials —
//!   the `1/i!j!k!` lives inside `L` (see [`add_m2l`]).
//! * **Index packing**: `ijk_to_idx` packs `(i,j,k)` into a flat array
//!   ordered by total degree `l = i+j+k`, then by `i`, then `j`. Every
//!   routine here (moments, derivatives, shifts) uses that one ordering.
//!
//! # Accuracy
//!
//! The far field is a truncated multipole expansion, so CFMM's J is
//! *approximate* by construction: error falls off as `(r_box/R)^{l_max+1}`.
//! The near field is exact to integral precision. Setting
//! [`CfmmConfig::force_all_near_field`] disables the far field entirely,
//! which makes J exact — that is the trivial-limit anchor this module is
//! tested against (see `cfmm_matches_direct_j_in_the_trivial_limit`).
//!
//! # STATUS — why the `cfmm-incomplete` feature gate is still on
//!
//! What IS implemented and cross-checked against independent constructions:
//! arbitrary-order Cartesian multipole integrals (vs libint2 overlap and
//! dipole), the Cartesian 1/r derivative tensor (vs finite differences),
//! M2L (vs exact ERIs, converging with `l_max`), the exact near field and
//! the whole traversal (vs the direct/dense J builder, to ~1e-14).
//!
//! What is NOT done, and why this must not yet be wired into the SCF as a
//! production `JBuilder`:
//!
//! * **No performance benefit.** The near field enumerates ORDERED shell
//!   pairs with no 8-fold symmetry folding and no Schwarz screening, and the
//!   far field rebuilds every shell-pair multipole on each `build` call.
//!   This is a CORRECTNESS reference, not a fast Coulomb builder — it is
//!   currently far SLOWER than [`crate::direct_j::DirectJ`]. No scaling
//!   claim is made or measured.
//! * **The extent-aware separation criterion is only validated on compact
//!   bases.** The well-separatedness test IS extent-aware (see
//!   [`CfmmJ::is_far_ext`]), which is what makes this "Continuous" FMM and
//!   was worth ~200x accuracy on alkane_10/STO-3G over a purely geometric
//!   test. But `extent_thresh` has only been exercised at STO-3G / cc-pVDZ;
//!   diffuse/augmented sets (aug-cc-pV*Z) are UNTESTED here, and they are
//!   precisely the case where extents grow and the criterion matters most.
//! * **Flat, single-level far field.** M2L runs directly between leaf boxes;
//!   there is no hierarchical upward M2M / downward L2L pass, so the far
//!   field costs O(n_leaf²) rather than FMM's O(N). (`shift_cartesian` is
//!   the tested translation operator that pass would use.)

use crate::fock::JBuilder;
use ferric_core::basis::BasisSet;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::engine::Engine;
use ferric_integrals::operator::Operator;
use ndarray::Array2;

// =====================================================================
// Cartesian multipole integrals over contracted Gaussian shell pairs.
//
// The integrals FFI exposes multipole one-electron integrals only up to
// DIPOLE (l=1, `OP_EMULTIPOLE1`), but CFMM needs arbitrary order `l_max`.
// This section implements them directly from the Gaussian product theorem.
//
// INDEPENDENT-CONSTRUCTION NOTE: these routines are validated against the
// libint2 FFI (`oneelectron::overlap` for order 0, `oneelectron::dipole`
// for order 1) — a genuinely independent implementation, not a
// self-consistency check. See `multipole_order0_matches_libint_overlap`
// and `multipole_order1_matches_libint_dipole`.
// =====================================================================

/// 1-D Cartesian moment integrals over a *primitive* Gaussian pair.
///
/// Returns `s[t][n]` = ∫ (x−Ax)^t? — precisely:
///
/// ```text
///   s[a][b][n] = ∫ (x−Ax)^a (x−Bx)^b (x−Cx)^n exp(−α(x−Ax)²) exp(−β(x−Bx)²) dx
/// ```
///
/// computed by the standard Obara–Saika-style recurrence over the Gaussian
/// product center P (Helgaker/Jørgensen/Olsen §9.5, "moment integrals"):
/// first build the `(a,b)`-free moment ladder about P, then transfer.
///
/// We use the simple and robust route: expand `(x−Ax)^a (x−Bx)^b (x−Cx)^n`
/// in powers of `(x−Px)` via the binomial theorem and integrate each
/// monomial against `exp(−p(x−Px)²)`, whose moments are the standard
/// Gaussian moments (odd → 0, even → double factorial).
fn moment_1d(
    a_pow: usize,
    b_pow: usize,
    n_pow: usize,
    pa: f64,
    pb: f64,
    pc: f64,
    p: f64,
) -> f64 {
    // ∫ (x−Px)^m exp(−p (x−Px)²) dx = 0 for odd m,
    //                               = (m−1)!! / (2p)^{m/2} · √(π/p) for even m.
    // Expand (x−Ax)^a = Σ_u C(a,u) (x−Px)^u (Px−Ax)^{a−u}, i.e. offset pa = Px−Ax.
    let mut total = 0.0;
    for u in 0..=a_pow {
        let ca = n_choose_k(a_pow, u) as f64 * pa.powi((a_pow - u) as i32);
        if ca == 0.0 {
            continue;
        }
        for v in 0..=b_pow {
            let cb = n_choose_k(b_pow, v) as f64 * pb.powi((b_pow - v) as i32);
            if cb == 0.0 {
                continue;
            }
            for w in 0..=n_pow {
                let cc = n_choose_k(n_pow, w) as f64 * pc.powi((n_pow - w) as i32);
                if cc == 0.0 {
                    continue;
                }
                let m = u + v + w;
                if m % 2 != 0 {
                    continue;
                }
                total += ca * cb * cc * gaussian_moment_even(m, p);
            }
        }
    }
    total
}

/// ∫ (x−Px)^m exp(−p(x−Px)²) dx for EVEN m, WITHOUT the √(π/p) factor
/// (the caller multiplies the 3-D product by (π/p)^{3/2} once).
///
/// = (m−1)!! / (2p)^{m/2}
fn gaussian_moment_even(m: usize, p: f64) -> f64 {
    debug_assert!(m % 2 == 0);
    let mut v = 1.0;
    let mut k = m;
    // (m-1)!! = (m-1)(m-3)...1
    while k >= 2 {
        v *= (k - 1) as f64;
        k -= 2;
    }
    v / (2.0 * p).powi((m / 2) as i32)
}

/// Cartesian monomial exponents `(i,j,k)` for a shell of angular momentum
/// `l`, in libint2's standard Cartesian ordering.
///
/// libint2 orders Cartesian components by descending x power, then
/// descending y power: for l=2 that is xx, xy, xz, yy, yz, zz — matching
/// the ordering documented in `ferric_integrals::ao_grid::eval_shell`.
fn cartesian_components(l: usize) -> Vec<(usize, usize, usize)> {
    let mut out = Vec::new();
    for i in (0..=l).rev() {
        for j in (0..=(l - i)).rev() {
            out.push((i, j, l - i - j));
        }
    }
    out
}

/// Primitive normalization constant for a Cartesian Gaussian, matching the
/// convention `ferric_integrals::ao_grid::radial` uses (and therefore
/// libint2's, which that module was validated against):
///
/// ```text
///   N(α, l) = (2α/π)^{3/4} · (4α)^{l/2} / √((2l−1)!!)
/// ```
///
/// This is the normalization of the *solid-harmonic-like* radial part; the
/// per-component Cartesian factor is folded in by
/// [`cartesian_self_overlap_factor`] where needed.
fn primitive_norm(alpha: f64, l: usize) -> f64 {
    let pi = std::f64::consts::PI;
    let dbl = double_factorial_odd(l);
    (2.0 * alpha / pi).powf(0.75) * (4.0 * alpha).powi(l as i32).sqrt() / dbl.sqrt()
}

/// (2l−1)!! for l ≥ 0 (with (−1)!! = 1).
fn double_factorial_odd(l: usize) -> f64 {
    let mut v = 1.0;
    let mut k = 2 * l;
    while k >= 2 {
        v *= (k - 1) as f64;
        k -= 2;
    }
    v
}

/// Extra per-component factor turning the uniform shell normalization above
/// into the correct Cartesian-component normalization.
///
/// For a Cartesian component (i,j,k) with i+j+k=l, the self-overlap carries
/// (2i−1)!!(2j−1)!!(2k−1)!! / (2l−1)!!, so the normalized component needs
/// √((2l−1)!! / ((2i−1)!!(2j−1)!!(2k−1)!!)).
fn cartesian_self_overlap_factor(i: usize, j: usize, k: usize) -> f64 {
    let l = i + j + k;
    (double_factorial_odd(l)
        / (double_factorial_odd(i) * double_factorial_odd(j) * double_factorial_odd(k)))
    .sqrt()
}

/// A shell's primitive data plus its center, gathered from `PreparedBasis`.
#[derive(Debug, Clone)]
struct ShellData {
    l: usize,
    pure: bool,
    center: [f64; 3],
    exponents: Vec<f64>,
    coefficients: Vec<f64>,
    /// Number of basis functions this shell contributes (pure or Cartesian).
    nfunc: usize,
}

/// Gather per-shell primitive data in the SAME shell order `PreparedBasis`
/// uses (atom-major, then the basis set's per-element shell order), so shell
/// index `s` here is the same `s` the ERI engine and `shell_offsets` use.
fn gather_shells(
    prep: &PreparedBasis,
    mol_atom_z: &[i32],
    bs: &BasisSet,
) -> Result<Vec<ShellData>, FerricError> {
    let centers = prep.shell_centers();
    let mut out = Vec::with_capacity(prep.nshells());
    let mut s = 0usize;
    for &z in mol_atom_z.iter() {
        let tmpls = bs.for_element(z).ok_or_else(|| {
            FerricError::Basis(format!("CFMM: no basis shells for Z={z}"))
        })?;
        for sh in tmpls {
            if s >= prep.nshells() {
                return Err(FerricError::General(
                    "CFMM: shell count mismatch vs PreparedBasis".into(),
                ));
            }
            let l = sh.l as usize;
            let nfunc = ferric_core::basis::num_functions(sh.l, sh.pure);
            if nfunc != prep.shell_dims()[s] {
                return Err(FerricError::General(format!(
                    "CFMM: shell {s} dim mismatch: basis says {nfunc}, PreparedBasis says {}",
                    prep.shell_dims()[s]
                )));
            }
            out.push(ShellData {
                l,
                pure: sh.pure,
                center: centers[s],
                exponents: sh.exponents.clone(),
                coefficients: sh.coefficients.clone(),
                nfunc,
            });
            s += 1;
        }
    }
    if s != prep.nshells() {
        return Err(FerricError::General(format!(
            "CFMM: gathered {s} shells but PreparedBasis has {}",
            prep.nshells()
        )));
    }
    Ok(out)
}

/// Overlap metric of the NORMALIZED Cartesian Gaussian components of a shell
/// with angular momentum `l`, same center and same exponent:
///
/// ```text
///   S_ab = (2a_x+2b_x−1)!!(2a_y+2b_y−1)!!(2a_z+2b_z−1)!!
///          / √( Π (2a−1)!! · Π (2b−1)!! )      (odd totals vanish)
/// ```
///
/// This basis is NOT orthonormal for l ≥ 2 (⟨xx|yy⟩ = 1/3), which is exactly
/// why the pure transform has to be normalized against it rather than by a
/// naive sum of squared coefficients.
fn cartesian_overlap_metric(l: usize) -> Vec<f64> {
    let comps = cartesian_components(l);
    let n = comps.len();
    let mut m = vec![0.0f64; n * n];
    for (a, &(ax, ay, az)) in comps.iter().enumerate() {
        for (b, &(bx, by, bz)) in comps.iter().enumerate() {
            let (tx, ty, tz) = (ax + bx, ay + by, az + bz);
            if tx % 2 != 0 || ty % 2 != 0 || tz % 2 != 0 {
                continue;
            }
            let num = double_factorial_odd_arg(tx)
                * double_factorial_odd_arg(ty)
                * double_factorial_odd_arg(tz);
            let na = (double_factorial_odd_arg(2 * ax)
                * double_factorial_odd_arg(2 * ay)
                * double_factorial_odd_arg(2 * az))
            .sqrt();
            let nb = (double_factorial_odd_arg(2 * bx)
                * double_factorial_odd_arg(2 * by)
                * double_factorial_odd_arg(2 * bz))
            .sqrt();
            m[a * n + b] = num / (na * nb);
        }
    }
    m
}

/// `(t−1)!!` for a raw argument `t` (so `double_factorial_odd_arg(4) = 3`).
/// Distinct from [`double_factorial_odd`], which takes `l` and returns `(2l−1)!!`.
fn double_factorial_odd_arg(t: usize) -> f64 {
    let mut v = 1.0;
    let mut k = t;
    while k >= 2 {
        v *= (k - 1) as f64;
        k -= 2;
    }
    v
}

/// Cartesian → pure (real solid harmonic) transform matrix for shell `l`,
/// shape `(2l+1) × ncart`, in libint2's pure ordering (m = −l..+l).
///
/// DERIVED, not transcribed: the coefficients are recovered by least-squares
/// fitting `ao_grid::eval_shell`'s pure output against its Cartesian output
/// on a fixed point set. `ao_grid`'s harmonics were already validated against
/// libint2 (see its module docs), so anchoring to it guarantees this module
/// uses the SAME convention as the ERI engine, rather than a hand-transcribed
/// table that could silently disagree in sign, order or normalization.
///
/// The fit is exact to ~1e-13 (the map really is linear); we assert that so a
/// convention change in `ao_grid` surfaces as a hard error, not a silent
/// wrong J.
fn cart_to_pure_matrix(l: usize) -> Result<Vec<f64>, FerricError> {
    use ferric_integrals::ao_grid::{eval_shell, LocatedShell};

    let ncart = (l + 1) * (l + 2) / 2;
    let npure = 2 * l + 1;
    if l == 0 {
        return Ok(vec![1.0]);
    }
    let alpha = [1.0f64];
    let coef = [1.0f64];
    let pure_sh = LocatedShell {
        l: l as i32, pure: true, exponents: &alpha, coefficients: &coef, center: [0.0; 3],
    };
    let cart_sh = LocatedShell {
        l: l as i32, pure: false, exponents: &alpha, coefficients: &coef, center: [0.0; 3],
    };

    // Deterministic pseudo-random sample points (fixed seed → reproducible).
    let npts = 40 * ncart;
    let mut seed = 0x2545_F491_4F6C_DD1Du64;
    let mut rnd = || {
        seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
        ((seed >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    };

    let mut a = vec![0.0; npts * ncart];
    let mut b = vec![0.0; npts * npure];
    let mut cbuf = vec![0.0; ncart];
    let mut pbuf = vec![0.0; npure];
    for g in 0..npts {
        let p = [rnd(), rnd(), rnd()];
        eval_shell(&cart_sh, p[0], p[1], p[2], &mut cbuf)
            .map_err(|e| FerricError::General(format!("CFMM cart_to_pure: {e}")))?;
        eval_shell(&pure_sh, p[0], p[1], p[2], &mut pbuf)
            .map_err(|e| FerricError::General(format!("CFMM cart_to_pure: {e}")))?;
        a[g * ncart..g * ncart + ncart].copy_from_slice(&cbuf);
        b[g * npure..g * npure + npure].copy_from_slice(&pbuf);
    }

    // Solve the normal equations (AᵀA) X = AᵀB by Gauss-Jordan.
    let mut ata = vec![0.0; ncart * ncart];
    let mut atb = vec![0.0; ncart * npure];
    for i in 0..ncart {
        for j in 0..ncart {
            let mut s = 0.0;
            for g in 0..npts { s += a[g * ncart + i] * a[g * ncart + j]; }
            ata[i * ncart + j] = s;
        }
        for j in 0..npure {
            let mut s = 0.0;
            for g in 0..npts { s += a[g * ncart + i] * b[g * npure + j]; }
            atb[i * npure + j] = s;
        }
    }
    for col in 0..ncart {
        let mut piv = col;
        for r in col..ncart {
            if ata[r * ncart + col].abs() > ata[piv * ncart + col].abs() { piv = r; }
        }
        if ata[piv * ncart + col].abs() < 1e-12 {
            return Err(FerricError::General(format!(
                "CFMM cart_to_pure: singular normal equations at l={l}"
            )));
        }
        for c in 0..ncart { ata.swap(col * ncart + c, piv * ncart + c); }
        for c in 0..npure { atb.swap(col * npure + c, piv * npure + c); }
        let dv = ata[col * ncart + col];
        for c in 0..ncart { ata[col * ncart + c] /= dv; }
        for c in 0..npure { atb[col * npure + c] /= dv; }
        for r in 0..ncart {
            if r != col {
                let f = ata[r * ncart + col];
                if f != 0.0 {
                    for c in 0..ncart { ata[r * ncart + c] -= f * ata[col * ncart + c]; }
                    for c in 0..npure { atb[r * npure + c] -= f * atb[col * npure + c]; }
                }
            }
        }
    }

    // out[m][c] = coefficient of cartesian component c in pure function m.
    let mut out = vec![0.0; npure * ncart];
    for c in 0..ncart {
        for m in 0..npure {
            out[m * ncart + c] = atb[c * npure + m];
        }
    }

    // RE-NORMALIZE against the true Cartesian overlap metric.
    //
    // `ao_grid::eval_shell`'s solid harmonics are NOT unit-normalized: it
    // applies one uniform radial normalization per shell and lets the angular
    // prefactors (√3·xy for d m=−2, etc.) stand as written, which leaves those
    // components with norm² = 3 rather than 1. libint2's AO basis IS
    // normalized per function, so using the raw `eval_shell` transform makes
    // the d/f/g moments too large by exactly that factor — measured as
    // `got = 3.0` vs `want = 1.0` on the pure-d diagonal of the cc-pVDZ
    // overlap cross-check before this correction.
    //
    // The normalized-Cartesian basis is non-orthogonal (⟨xx|yy⟩ = 1/3), so the
    // norm of a pure combination must be taken under that metric:
    //   ‖p‖² = Σ_ab p_a S_ab p_b,  S_ab = (2a+2b−1)!!-style ratio below.
    let metric = cartesian_overlap_metric(l);
    for m in 0..npure {
        let mut nrm2 = 0.0;
        for a in 0..ncart {
            for b in 0..ncart {
                nrm2 += out[m * ncart + a] * metric[a * ncart + b] * out[m * ncart + b];
            }
        }
        if nrm2 <= 0.0 {
            return Err(FerricError::General(format!(
                "CFMM cart_to_pure: non-positive norm for l={l} m={m}"
            )));
        }
        let s = 1.0 / nrm2.sqrt();
        for a in 0..ncart {
            out[m * ncart + a] *= s;
        }
    }

    // VERIFY the result is an ORTHONORMAL set under the Cartesian metric.
    // Real solid harmonics of the same l are mutually orthogonal, so this
    // pins both the normalization applied above and the linear independence
    // of the extracted rows. A failure means `ao_grid`'s pure convention
    // changed (different ordering or a non-harmonic mix) and this transform
    // is no longer valid — a hard error, never a silently wrong J.
    let mut max_off: f64 = 0.0;
    for m in 0..npure {
        for n in 0..npure {
            let mut v = 0.0;
            for a in 0..ncart {
                for b in 0..ncart {
                    v += out[m * ncart + a] * metric[a * ncart + b] * out[n * ncart + b];
                }
            }
            let want = if m == n { 1.0 } else { 0.0 };
            max_off = max_off.max((v - want).abs());
        }
    }
    if max_off > 1e-9 {
        return Err(FerricError::General(format!(
            "CFMM cart_to_pure: pure functions not orthonormal at l={l} \
             (max deviation {max_off:.3e}) — ao_grid's convention appears to have changed"
        )));
    }
    Ok(out)
}

/// Cartesian multipole moments of every basis-function pair in a shell pair.
///
/// Returns a flat buffer indexed `[(a * n2 + b) * n_mom + m]` holding
///
/// ```text
///   M^{ab}_ijk = ∫ χ_a(r) χ_b(r) (x−Cx)^i (y−Cy)^j (z−Cz)^k dr
/// ```
///
/// for every `(i,j,k)` with `i+j+k ≤ l_max`, packed by [`ijk_to_idx`],
/// where `a` runs over shell 1's functions and `b` over shell 2's.
///
/// This is the arbitrary-order generalization the FFI does not provide
/// (`OP_EMULTIPOLE1` stops at dipole). Order 0 reduces to the overlap matrix
/// and order 1 to the dipole integrals — both cross-checked against libint2
/// in this module's tests.
fn shell_pair_multipoles(
    sh1: &ShellData,
    sh2: &ShellData,
    origin: [f64; 3],
    l_max: usize,
    cart_transforms: &[Option<Vec<f64>>],
) -> Vec<f64> {
    let n_mom = n_moments(l_max);
    let comps1 = cartesian_components(sh1.l);
    let comps2 = cartesian_components(sh2.l);
    let nc1 = comps1.len();
    let nc2 = comps2.len();

    // Cartesian-basis moments first; transform to pure at the end if needed.
    let mut cart = vec![0.0f64; nc1 * nc2 * n_mom];

    let (ax, ay, az) = (sh1.center[0], sh1.center[1], sh1.center[2]);
    let (bx, by, bz) = (sh2.center[0], sh2.center[1], sh2.center[2]);
    let ab2 = (ax - bx).powi(2) + (ay - by).powi(2) + (az - bz).powi(2);

    for (ip, (&alpha, &c1)) in sh1.exponents.iter().zip(sh1.coefficients.iter()).enumerate() {
        let _ = ip;
        let n1 = primitive_norm(alpha, sh1.l);
        for (&beta, &c2) in sh2.exponents.iter().zip(sh2.coefficients.iter()) {
            let n2 = primitive_norm(beta, sh2.l);
            let p = alpha + beta;
            let mu = alpha * beta / p;
            let pref = c1 * c2 * n1 * n2 * (-mu * ab2).exp();
            if pref == 0.0 {
                continue;
            }
            let px = (alpha * ax + beta * bx) / p;
            let py = (alpha * ay + beta * by) / p;
            let pz = (alpha * az + beta * bz) / p;
            // Offsets used by the binomial expansion in `moment_1d`.
            let (pax, pay, paz) = (px - ax, py - ay, pz - az);
            let (pbx, pby, pbz) = (px - bx, py - by, pz - bz);
            let (pcx, pcy, pcz) = (px - origin[0], py - origin[1], pz - origin[2]);
            // (π/p)^{3/2} from the three 1-D Gaussian integrals.
            let gauss3 = (std::f64::consts::PI / p).powf(1.5);

            for (a_i, &(a1, a2, a3)) in comps1.iter().enumerate() {
                let norm_a = cartesian_self_overlap_factor(a1, a2, a3);
                for (b_i, &(b1, b2, b3)) in comps2.iter().enumerate() {
                    let norm_b = cartesian_self_overlap_factor(b1, b2, b3);
                    let w = pref * norm_a * norm_b * gauss3;
                    let base = (a_i * nc2 + b_i) * n_mom;
                    for l in 0..=l_max {
                        for i in 0..=l {
                            for j in 0..=(l - i) {
                                let k = l - i - j;
                                let sx = moment_1d(a1, b1, i, pax, pbx, pcx, p);
                                let sy = moment_1d(a2, b2, j, pay, pby, pcy, p);
                                let sz = moment_1d(a3, b3, k, paz, pbz, pcz, p);
                                cart[base + ijk_to_idx(i, j, k)] += w * sx * sy * sz;
                            }
                        }
                    }
                }
            }
        }
    }

    // Cartesian → pure on each shell index, if that shell is pure.
    let t1 = if sh1.pure { cart_transforms[sh1.l].as_ref() } else { None };
    let t2 = if sh2.pure { cart_transforms[sh2.l].as_ref() } else { None };
    if t1.is_none() && t2.is_none() {
        return cart;
    }
    let nf1 = sh1.nfunc;
    let nf2 = sh2.nfunc;
    let mut out = vec![0.0f64; nf1 * nf2 * n_mom];
    for a in 0..nf1 {
        for b in 0..nf2 {
            let obase = (a * nf2 + b) * n_mom;
            for ca in 0..nc1 {
                let wa = match t1 { Some(t) => t[a * nc1 + ca], None => if a == ca { 1.0 } else { 0.0 } };
                if wa == 0.0 { continue; }
                for cb in 0..nc2 {
                    let wb = match t2 { Some(t) => t[b * nc2 + cb], None => if b == cb { 1.0 } else { 0.0 } };
                    if wb == 0.0 { continue; }
                    let ibase = (ca * nc2 + cb) * n_mom;
                    let w = wa * wb;
                    for m in 0..n_mom {
                        out[obase + m] += w * cart[ibase + m];
                    }
                }
            }
        }
    }
    out
}

/// Number of Cartesian moments with total degree ≤ `l_max`.
fn n_moments(l_max: usize) -> usize {
    (l_max + 1) * (l_max + 2) * (l_max + 3) / 6
}

/// Cartesian derivatives of 1/r evaluated at `d`:
/// `H_ijk = ∂^{i+j+k}/∂x^i ∂y^j ∂z^k (1/r)` for all `i+j+k ≤ l_max`,
/// packed by [`ijk_to_idx`].
///
/// # Method
///
/// Write `H_ijk = P_ijk(x,y,z) / r^{2L+1}` with `L = i+j+k` and `P_000 = 1`.
/// Differentiating that form gives an EXACT recurrence on the polynomial
/// numerators:
///
/// ```text
///   ∂/∂x [ P / r^{2L+1} ] = [ r² ∂P/∂x − (2L+1) x P ] / r^{2L+3}
/// ```
///
/// so, since `2(L+1)+1 = 2L+3`,
///
/// ```text
///   P_{i+1,j,k} = r² · ∂P_ijk/∂x − (2L+1) · x · P_ijk     (and cyclically)
/// ```
///
/// The recurrence needs `∂P/∂x`, so we must carry the polynomial COEFFICIENTS,
/// not just its value at `d` — that is precisely why the previous
/// value-only loop in this function could never work and was left as a
/// documented no-op that silently returned zeros for every order ≥ 3.
///
/// Each `P_ijk` is stored as a dense coefficient array over monomials
/// `x^a y^b z^c` with `a+b+c ≤ 2*l_max+1` (degree grows by at most 1 per
/// `r²·∂` step), packed by [`ijk_to_idx`] on `(a,b,c)`.
///
/// Verified against high-order symbolic/finite-difference derivatives of 1/r
/// in `cartesian_derivatives_match_finite_differences` — an independent
/// construction sharing no code with this recurrence.
fn compute_cartesian_derivatives(d: [f64; 3], l_max: usize) -> Vec<f64> {
    let n_out = n_moments(l_max);
    // Monomial degree bound: P_ijk has degree ≤ L (its parity/structure keeps
    // it at ≤ L+... ); allocate generously to 2*l_max+2 to be safe.
    let deg_max = 2 * l_max + 2;
    let n_poly = n_moments(deg_max);

    // polys[idx_ijk] = coefficient vector over monomials (a,b,c).
    let mut polys: Vec<Vec<f64>> = vec![Vec::new(); n_out];
    let mut p000 = vec![0.0f64; n_poly];
    p000[ijk_to_idx(0, 0, 0)] = 1.0;
    polys[ijk_to_idx(0, 0, 0)] = p000;

    // Apply one step: q = r² · ∂p/∂axis − (2L+1) · x_axis · p
    let step = |p: &[f64], axis: usize, l: usize| -> Vec<f64> {
        let mut q = vec![0.0f64; n_poly];
        let c = (2 * l + 1) as f64;
        for a in 0..=deg_max {
            for b in 0..=(deg_max - a) {
                for cc in 0..=(deg_max - a - b) {
                    let v = p[ijk_to_idx(a, b, cc)];
                    if v == 0.0 {
                        continue;
                    }
                    // r² · ∂p/∂axis
                    let (da, db, dc, pow) = match axis {
                        0 => (a.wrapping_sub(1), b, cc, a),
                        1 => (a, b.wrapping_sub(1), cc, b),
                        _ => (a, b, cc.wrapping_sub(1), cc),
                    };
                    if pow > 0 {
                        let w = v * pow as f64;
                        // multiply by r² = x²+y²+z²
                        if da + 2 + db + dc <= deg_max {
                            q[ijk_to_idx(da + 2, db, dc)] += w;
                        }
                        if da + db + 2 + dc <= deg_max {
                            q[ijk_to_idx(da, db + 2, dc)] += w;
                        }
                        if da + db + dc + 2 <= deg_max {
                            q[ijk_to_idx(da, db, dc + 2)] += w;
                        }
                    }
                    // −(2L+1) · x_axis · p
                    let (ea, eb, ec) = match axis {
                        0 => (a + 1, b, cc),
                        1 => (a, b + 1, cc),
                        _ => (a, b, cc + 1),
                    };
                    if ea + eb + ec <= deg_max {
                        q[ijk_to_idx(ea, eb, ec)] -= c * v;
                    }
                }
            }
        }
        q
    };

    for l in 0..l_max {
        for i in 0..=l {
            for j in 0..=(l - i) {
                let k = l - i - j;
                let src = polys[ijk_to_idx(i, j, k)].clone();
                polys[ijk_to_idx(i + 1, j, k)] = step(&src, 0, l);
                if i == 0 {
                    polys[ijk_to_idx(i, j + 1, k)] = step(&src, 1, l);
                }
                if i == 0 && j == 0 {
                    polys[ijk_to_idx(i, j, k + 1)] = step(&src, 2, l);
                }
            }
        }
    }

    // Evaluate H_ijk = P_ijk(d) / r^{2L+1}.
    let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
    let r = r2.sqrt();
    let mut h = vec![0.0f64; n_out];
    // Precompute monomial values.
    let mut xpow = vec![1.0f64; deg_max + 1];
    let mut ypow = vec![1.0f64; deg_max + 1];
    let mut zpow = vec![1.0f64; deg_max + 1];
    for t in 1..=deg_max {
        xpow[t] = xpow[t - 1] * d[0];
        ypow[t] = ypow[t - 1] * d[1];
        zpow[t] = zpow[t - 1] * d[2];
    }
    for l in 0..=l_max {
        let rpow = r.powi((2 * l + 1) as i32);
        for i in 0..=l {
            for j in 0..=(l - i) {
                let k = l - i - j;
                let p = &polys[ijk_to_idx(i, j, k)];
                let mut val = 0.0;
                for a in 0..=deg_max {
                    for b in 0..=(deg_max - a) {
                        for cc in 0..=(deg_max - a - b) {
                            let v = p[ijk_to_idx(a, b, cc)];
                            if v != 0.0 {
                                val += v * xpow[a] * ypow[b] * zpow[cc];
                            }
                        }
                    }
                }
                h[ijk_to_idx(i, j, k)] = val / rpow;
            }
        }
    }
    h
}

// =====================================================================
// Octree
// =====================================================================

/// A box in the CFMM octree.
#[derive(Debug, Clone)]
pub(crate) struct CfmmBox {
    pub(crate) center: [f64; 3],
    pub(crate) width: f64,
    pub(crate) level: usize,
    pub(crate) children: Option<Box<[CfmmBox; 8]>>,
    /// Indices into [`CfmmJ::pairs`] of the shell PAIRS whose product-charge
    /// distribution is centered in this box. CFMM bins PAIR distributions,
    /// not individual shells — see the module docs and `ShellPair`.
    pub(crate) pair_indices: Vec<usize>,
    pub(crate) multipoles: Vec<f64>,
    pub(crate) local_exp: Vec<f64>,
}

impl CfmmBox {
    pub(crate) fn new(center: [f64; 3], width: f64, level: usize) -> Self {
        CfmmBox {
            center,
            width,
            level,
            children: None,
            pair_indices: Vec::new(),
            multipoles: Vec::new(),
            local_exp: Vec::new(),
        }
    }

    /// Recursively build the octree by inserting a shell-PAIR distribution at
    /// its product center.
    pub(crate) fn insert_pair(&mut self, pair_idx: usize, pair_center: [f64; 3], max_level: usize) {
        if self.level == max_level {
            self.pair_indices.push(pair_idx);
            return;
        }

        if self.children.is_none() {
            let mut children = Vec::with_capacity(8);
            let h = self.width / 4.0;
            for i in 0..8 {
                let dx = if (i & 1) != 0 { h } else { -h };
                let dy = if (i & 2) != 0 { h } else { -h };
                let dz = if (i & 4) != 0 { h } else { -h };
                children.push(CfmmBox::new(
                    [self.center[0] + dx, self.center[1] + dy, self.center[2] + dz],
                    self.width / 2.0,
                    self.level + 1,
                ));
            }
            self.children = Some(Box::new(children.try_into().unwrap()));
        }

        let child_idx = self.get_child_index(pair_center);
        self.children.as_mut().unwrap()[child_idx].insert_pair(pair_idx, pair_center, max_level);
    }

    fn get_child_index(&self, p: [f64; 3]) -> usize {
        let mut idx = 0;
        if p[0] > self.center[0] { idx |= 1; }
        if p[1] > self.center[1] { idx |= 2; }
        if p[2] > self.center[2] { idx |= 4; }
        idx
    }

    /// Depth-first collection of leaf boxes (shared refs).
    fn collect_leaves<'a>(&'a self, out: &mut Vec<&'a CfmmBox>) {
        if let Some(children) = &self.children {
            for c in children.iter() {
                c.collect_leaves(out);
            }
        } else {
            out.push(self);
        }
    }
}

/// A shell-pair product charge distribution — the object CFMM actually bins.
///
/// The charge distribution appearing in `J_{μν} = Σ (μν|λσ) D_{λσ}` is not
/// atom-centered: it is the set of products `Ω_{s1,s2}(r) = χ_{s1}(r) χ_{s2}(r)`.
/// Each such product is (a contraction of) Gaussians centered at the
/// **Gaussian product center** `P = (αA + βB)/(α+β)`, which in general lies
/// between the two shells and in a DIFFERENT box than either of them.
///
/// Binning individual SHELLS instead of PAIRS is the structural error that
/// makes a CFMM silently drop every quartet whose bra pair straddles two
/// boxes; the trivial-limit anchor
/// (`cfmm_matches_direct_j_in_the_trivial_limit`) catches it immediately
/// (measured: max diff 3.28 against max|J| 17.4 — a ~19% shortfall, not a
/// rounding error). Each pair belongs to exactly one leaf, so every
/// `(s1,s2,s3,s4)` quartet is covered exactly once by exactly one of the
/// near/far paths.
#[derive(Debug, Clone)]
struct ShellPair {
    s1: usize,
    s2: usize,
    /// Charge-weighted product center (the "continuous" center of CFMM).
    center: [f64; 3],
    /// Spatial EXTENT of this product distribution: the radius beyond which
    /// its charge density is below `extent_thresh`. This is what makes the
    /// method "Continuous" — see [`pair_extent`].
    extent: f64,
}

/// Representative product center of a contracted shell pair: the primitive
/// product centers `(αA + βB)/(α+β)` averaged with weight
/// `|c1 c2| exp(−αβ/(α+β) |A−B|²)`, i.e. by each primitive product's
/// contribution to the total charge. For a diagonal pair (A == B) this
/// reduces exactly to the shell center.
fn pair_center(sh1: &ShellData, sh2: &ShellData) -> [f64; 3] {
    let (a, b) = (sh1.center, sh2.center);
    let ab2 = (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2);
    let mut acc = [0.0f64; 3];
    let mut wsum = 0.0f64;
    for (&alpha, &c1) in sh1.exponents.iter().zip(sh1.coefficients.iter()) {
        for (&beta, &c2) in sh2.exponents.iter().zip(sh2.coefficients.iter()) {
            let p = alpha + beta;
            let w = (c1 * c2).abs() * (-alpha * beta / p * ab2).exp();
            for k in 0..3 {
                acc[k] += w * (alpha * a[k] + beta * b[k]) / p;
            }
            wsum += w;
        }
    }
    if wsum > 0.0 {
        [acc[0] / wsum, acc[1] / wsum, acc[2] / wsum]
    } else {
        [(a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0, (a[2] + b[2]) / 2.0]
    }
}

/// Spatial extent of a shell-pair product distribution.
///
/// The multipole expansion of `1/r` between two charge distributions
/// converges only where they do NOT overlap, so a box-geometry test is not
/// sufficient: a diffuse Gaussian product reaches well outside its own box.
/// This is the "Continuous" in Continuous FMM (White & Head-Gordon): each
/// distribution carries a radius `r_ext` such that its density beyond
/// `r_ext` is below the target precision, and two distributions may only
/// interact through multipoles when their extents do not reach each other.
///
/// For a primitive product with combined exponent `p = α+β`, the density
/// falls as `exp(−p r²)`, so
///
/// ```text
///   r_ext(p, ε) = sqrt( −ln(ε) / p )
/// ```
///
/// The pair's extent is the LARGEST over its primitive products (the most
/// diffuse primitive sets the reach), weighted only in the sense that
/// negligible primitives are skipped.
fn pair_extent(sh1: &ShellData, sh2: &ShellData, thresh: f64) -> f64 {
    let (a, b) = (sh1.center, sh2.center);
    let ab2 = (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2);
    let ln_eps = -thresh.ln();
    let mut ext: f64 = 0.0;
    for (&alpha, &c1) in sh1.exponents.iter().zip(sh1.coefficients.iter()) {
        for (&beta, &c2) in sh2.exponents.iter().zip(sh2.coefficients.iter()) {
            let p = alpha + beta;
            // Skip primitives whose prefactor is already below threshold: a
            // very diffuse but vanishing primitive must not inflate the extent.
            let pref = (c1 * c2).abs() * (-alpha * beta / p * ab2).exp();
            if pref < thresh {
                continue;
            }
            ext = ext.max((ln_eps / p).sqrt());
        }
    }
    ext
}

/// Tuning knobs for [`CfmmJ`].
#[derive(Debug, Clone)]
pub struct CfmmConfig {
    /// Multipole truncation order.
    pub l_max: usize,
    /// Octree depth (number of subdivisions below the root).
    pub max_level: usize,
    /// Well-separatedness factor: two leaf boxes interact through the
    /// multipole (far-field) path when their center separation exceeds
    /// `ws_factor * box_width`. The classic FMM value is 2.0.
    pub ws_factor: f64,
    /// Integral screening threshold for the near-field ERIs.
    pub thresh: f64,
    /// Density threshold defining a shell-pair distribution's spatial extent
    /// (see [`pair_extent`]). Smaller = larger extents = more pairs forced
    /// into the exact near field = more accurate but slower.
    pub extent_thresh: f64,
    /// TRIVIAL-LIMIT SWITCH. When true the well-separatedness test never
    /// fires: every leaf pair is treated as near field and evaluated with
    /// exact 4-center ERIs, so J is EXACT (no multipole approximation at
    /// all) and must reproduce the direct/dense builder to integral
    /// precision.
    ///
    /// This is the exactness anchor required by the project's experimental
    /// protocol: it isolates the octree traversal, the leaf-pair
    /// enumeration and the ERI scatter from all multipole numerics, so a
    /// failure here is unambiguously a traversal/scatter bug and NOT
    /// multipole truncation error. Nothing else in this module is measured
    /// until `cfmm_matches_direct_j_in_the_trivial_limit` passes.
    pub force_all_near_field: bool,
}

impl Default for CfmmConfig {
    fn default() -> Self {
        CfmmConfig {
            l_max: 6,
            max_level: 3,
            ws_factor: 2.0,
            thresh: 1e-12,
            extent_thresh: 1e-10,
            force_all_near_field: false,
        }
    }
}

/// CFMM Coulomb matrix builder.
pub struct CfmmJ {
    prep: PreparedBasis,
    root: CfmmBox,
    cfg: CfmmConfig,
    /// Per-shell primitive data, indexed by `PreparedBasis` shell index.
    shells: Vec<ShellData>,
    /// All shell pairs, binned into the octree by product center.
    pairs: Vec<ShellPair>,
    /// Cartesian→pure transforms, indexed by angular momentum.
    cart_transforms: Vec<Option<Vec<f64>>>,
    /// 4-center Coulomb ERI engine for the near field (built lazily once).
    engine: Option<Engine>,
}

impl std::fmt::Debug for CfmmJ {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CfmmJ")
            .field("cfg", &self.cfg)
            .finish_non_exhaustive()
    }
}

impl CfmmJ {
    /// Build a CFMM Coulomb builder for `mol` × `bs`.
    ///
    /// `mol_atom_z` must be the molecule's atomic numbers in the same order
    /// `PreparedBasis` was constructed from (used to re-derive the primitive
    /// data the FFI does not hand back).
    pub fn new(
        prep: PreparedBasis,
        bs: BasisSet,
        mol_atom_z: &[i32],
        cfg: CfmmConfig,
    ) -> Result<Self, FerricError> {
        let shells = gather_shells(&prep, mol_atom_z, &bs)?;

        let max_l = shells.iter().map(|s| s.l).max().unwrap_or(0);
        let mut cart_transforms = vec![None; max_l + 1];
        for sh in &shells {
            if sh.pure && cart_transforms[sh.l].is_none() {
                cart_transforms[sh.l] = Some(cart_to_pure_matrix(sh.l)?);
            }
        }

        // Enumerate ALL ordered shell pairs and bin each by its product center.
        // Ordered (not canonical) pairs keep the near-field scatter free of
        // any symmetry folding — see `near_field`.
        let nsh = prep.nshells();
        let mut pairs = Vec::with_capacity(nsh * nsh);
        for s1 in 0..nsh {
            for s2 in 0..nsh {
                let c = pair_center(&shells[s1], &shells[s2]);
                let e = pair_extent(&shells[s1], &shells[s2], cfg.extent_thresh);
                pairs.push(ShellPair { s1, s2, center: c, extent: e });
            }
        }

        // The bounding box must cover PAIR centers, which all lie within the
        // convex hull of the shell centers, so the shell-center box suffices;
        // recompute anyway so a future weighting change cannot silently push
        // a pair center outside the root box.
        let mut pmin = [f64::INFINITY; 3];
        let mut pmax = [f64::NEG_INFINITY; 3];
        for pr in &pairs {
            for i in 0..3 {
                pmin[i] = pmin[i].min(pr.center[i]);
                pmax[i] = pmax[i].max(pr.center[i]);
            }
        }
        let center = [
            (pmin[0] + pmax[0]) / 2.0,
            (pmin[1] + pmax[1]) / 2.0,
            (pmin[2] + pmax[2]) / 2.0,
        ];
        let span = (pmax[0] - pmin[0]).max(pmax[1] - pmin[1]).max(pmax[2] - pmin[2]);
        let width = if span > 1e-8 { span * 1.1 } else { 1.0 };

        let mut root = CfmmBox::new(center, width, 0);
        for (i, pr) in pairs.iter().enumerate() {
            root.insert_pair(i, pr.center, cfg.max_level);
        }

        Ok(CfmmJ { prep, root, cfg, shells, pairs, cart_transforms, engine: None })
    }

    /// Number of basis functions.
    pub fn nbasis(&self) -> usize {
        self.prep.nbasis()
    }
}

impl CfmmJ {
    /// Classify an ordered leaf pair as near or far, using the CONTINUOUS
    /// (extent-aware) well-separatedness criterion.
    ///
    /// Two leaves may interact through multipoles only when their charge
    /// distributions do not reach each other:
    ///
    /// ```text
    ///   R  >  ext_a + ext_b + ws_factor · width
    /// ```
    ///
    /// where `ext_x` is the largest extent of any shell-pair distribution
    /// binned into leaf `x` (see [`pair_extent`]). The `ws_factor · width`
    /// term keeps the classic geometric margin on top of the physical reach
    /// of the charge clouds.
    ///
    /// Dropping the extent terms (a purely geometric test) is NOT safe: for
    /// alkane_10 / STO-3G with `max_level = 3` the leaf width is ~3.0 Bohr,
    /// so a geometric test calls boxes "far" at 6.0 Bohr separation while the
    /// carbon 2sp product distributions still carry density out to ~7.6 Bohr.
    /// Expanding `1/r` between overlapping clouds is outside its radius of
    /// convergence, and the far-field error then STOPS falling with `l_max`
    /// (measured: 5.4e-2 → 1.2e-2 from l_max 2 → 4, stalling far above the
    /// truncation budget) — which is exactly the "consistent but wrong"
    /// signature the project's experimental protocol warns about.
    ///
    /// With `force_all_near_field` every pair is near — the exactness anchor.
    fn is_far_ext(&self, a: &CfmmBox, b: &CfmmBox, ext_a: f64, ext_b: f64) -> bool {
        if self.cfg.force_all_near_field {
            return false;
        }
        let dx = a.center[0] - b.center[0];
        let dy = a.center[1] - b.center[1];
        let dz = a.center[2] - b.center[2];
        let dist = (dx * dx + dy * dy + dz * dz).sqrt();
        dist > ext_a + ext_b + self.cfg.ws_factor * a.width
    }

    /// Largest shell-pair extent binned into each leaf, in the depth-first
    /// leaf order of [`CfmmBox::collect_leaves`].
    fn leaf_extents(&self, leaves: &[&CfmmBox]) -> Vec<f64> {
        leaves
            .iter()
            .map(|l| {
                l.pair_indices
                    .iter()
                    .map(|&i| self.pairs[i].extent)
                    .fold(0.0f64, f64::max)
            })
            .collect()
    }

    /// Near field: exact `J_{μν} += Σ_{λσ} (μν|λσ) D_{λσ}` for every
    /// (bra-leaf, ket-leaf) leaf pair that is NOT well separated.
    ///
    /// Iterates ORDERED leaf pairs and, within them, all ordered shell PAIRS
    /// binned in each leaf — no 8-fold-symmetry folding. That is deliberately
    /// the simple, obviously-correct formulation: the canonical-quartet
    /// kernels in `quartet_scatter.rs` fold bra/ket symmetry in a way that
    /// does not compose with a bra-pair-from-box-A / ket-pair-from-box-B
    /// partition, and a double-counting bug there would be invisible against
    /// a multipole error budget. Correctness first; this is not a fast path.
    ///
    /// Because each shell pair lives in exactly one leaf and each leaf pair is
    /// classified near XOR far, every `(s1,s2,s3,s4)` quartet is accounted for
    /// exactly once across the two paths.
    fn near_field(
        &mut self,
        d: &Array2<f64>,
        j: &mut Array2<f64>,
        far_pairs: &mut Vec<(usize, usize)>,
    ) -> Result<usize, FerricError> {
        let mut leaves: Vec<&CfmmBox> = Vec::new();
        self.root.collect_leaves(&mut leaves);

        let n_leaf = leaves.len();
        let extents = self.leaf_extents(&leaves);
        let mut near_pairs: Vec<(usize, usize)> = Vec::new();
        for a in 0..n_leaf {
            if leaves[a].pair_indices.is_empty() {
                continue;
            }
            for b in 0..n_leaf {
                if leaves[b].pair_indices.is_empty() {
                    continue;
                }
                if self.is_far_ext(leaves[a], leaves[b], extents[a], extents[b]) {
                    far_pairs.push((a, b));
                } else {
                    near_pairs.push((a, b));
                }
            }
        }

        let pair_lists: Vec<Vec<usize>> =
            leaves.iter().map(|l| l.pair_indices.clone()).collect();

        if self.engine.is_none() {
            self.engine = Some(Engine::new_2e(Operator::coulomb(), &self.prep, 1e-14)?);
        }

        let dims = self.prep.shell_dims().to_vec();
        let offs = self.prep.shell_offsets().to_vec();
        let prep = &self.prep;
        let pairs = &self.pairs;
        let engine = self.engine.as_mut().expect("engine initialized above");

        let mut n_quartets = 0usize;
        for &(a, b) in &near_pairs {
            for &bp in &pair_lists[a] {
                let (s1, s2) = (pairs[bp].s1, pairs[bp].s2);
                let (n1, n2) = (dims[s1], dims[s2]);
                let (o1, o2) = (offs[s1], offs[s2]);
                for &kp in &pair_lists[b] {
                    let (s3, s4) = (pairs[kp].s1, pairs[kp].s2);
                    let q = match engine.compute_quartet(prep, s1, s2, s3, s4) {
                        Some(q) => q,
                        None => continue,
                    };
                    n_quartets += 1;
                    let (n3, n4) = (dims[s3], dims[s4]);
                    let (o3, o4) = (offs[s3], offs[s4]);
                    for aa in 0..n1 {
                        for bb in 0..n2 {
                            let mut acc = 0.0;
                            for cc in 0..n3 {
                                for dd in 0..n4 {
                                    let v = q[((aa * n2 + bb) * n3 + cc) * n4 + dd];
                                    acc += v * d[(o3 + cc, o4 + dd)];
                                }
                            }
                            j[(o1 + aa, o2 + bb)] += acc;
                        }
                    }
                }
            }
        }
        Ok(n_quartets)
    }
}

impl CfmmJ {
    /// Upward pass: density-weighted Cartesian multipole moments of each leaf
    /// box's charge distribution about that box's center.
    ///
    /// The leaf's charge density is the part of the total electron density
    /// carried by the shell PAIRS binned into this leaf (by product center):
    ///
    /// ```text
    ///   M^box_ijk = Σ_{λ,σ ∈ box} D_{λσ} ∫ χ_λ χ_σ (x−Cx)^i (y−Cy)^j (z−Cz)^k
    /// ```
    ///
    /// Because each shell pair is binned into exactly ONE leaf, and
    /// `near_field` enumerates the complementary (near) leaf pairs over the
    /// same binning, every (λ,σ) pair is covered by exactly one of the two
    /// paths — no double counting, no omission.
    fn upward_pass(&mut self, d: &Array2<f64>) {
        let l_max = self.cfg.l_max;
        let n_mom = n_moments(l_max);
        let offs = self.prep.shell_offsets().to_vec();
        let dims = self.prep.shell_dims().to_vec();

        // Two-phase (no raw pointers): compute every leaf's moments against an
        // immutable view of the tree, then write them back in the same
        // depth-first leaf order.
        let (centers, pair_lists) = {
            let mut leaves: Vec<&CfmmBox> = Vec::new();
            self.root.collect_leaves(&mut leaves);
            (
                leaves.iter().map(|l| l.center).collect::<Vec<_>>(),
                leaves.iter().map(|l| l.pair_indices.clone()).collect::<Vec<_>>(),
            )
        };

        let mut computed: Vec<Vec<f64>> = Vec::with_capacity(centers.len());
        for (li, center) in centers.iter().enumerate() {
            let mut mom = vec![0.0f64; n_mom];
            for &kp in &pair_lists[li] {
                let (s3, s4) = (self.pairs[kp].s1, self.pairs[kp].s2);
                let m = shell_pair_multipoles(
                    &self.shells[s3],
                    &self.shells[s4],
                    *center,
                    l_max,
                    &self.cart_transforms,
                );
                let (n3, n4) = (dims[s3], dims[s4]);
                let (o3, o4) = (offs[s3], offs[s4]);
                for cc in 0..n3 {
                    for dd in 0..n4 {
                        let dval = d[(o3 + cc, o4 + dd)];
                        if dval == 0.0 {
                            continue;
                        }
                        let base = (cc * n4 + dd) * n_mom;
                        for t in 0..n_mom {
                            mom[t] += dval * m[base + t];
                        }
                    }
                }
            }
            computed.push(mom);
        }

        // Write back in the identical depth-first order.
        let mut it = computed.into_iter();
        write_leaf_multipoles(&mut self.root, &mut it);
    }

    /// Far field: for each well-separated (bra-leaf, ket-leaf) pair, translate
    /// the ket leaf's multipoles into a local Taylor expansion about the bra
    /// leaf's center (M2L), then contract that expansion against the bra
    /// leaf's shell-pair multipole moments.
    ///
    /// ```text
    ///   Φ(r) ≈ Σ_ijk L_ijk (r−C_bra)^{ijk},
    ///   J_{μν} += Σ_ijk L_ijk M^{μν}_ijk        (μ,ν ∈ bra leaf)
    /// ```
    fn far_field(
        &mut self,
        j: &mut Array2<f64>,
        far_pairs: &[(usize, usize)],
    ) {
        if far_pairs.is_empty() {
            return;
        }
        let l_max = self.cfg.l_max;
        let n_mom = n_moments(l_max);

        // Snapshot leaf centers / multipoles / shell lists (immutable view).
        let mut leaves: Vec<&CfmmBox> = Vec::new();
        self.root.collect_leaves(&mut leaves);
        let centers: Vec<[f64; 3]> = leaves.iter().map(|l| l.center).collect();
        let mpoles: Vec<Vec<f64>> = leaves.iter().map(|l| l.multipoles.clone()).collect();
        let pair_lists: Vec<Vec<usize>> =
            leaves.iter().map(|l| l.pair_indices.clone()).collect();

        // Accumulate a local expansion per bra leaf.
        let mut local: Vec<Vec<f64>> = vec![vec![0.0; n_mom]; leaves.len()];
        for &(a, b) in far_pairs {
            add_m2l(&mut local[a], &mpoles[b], centers[a], centers[b], l_max);
        }

        let offs = self.prep.shell_offsets().to_vec();
        let dims = self.prep.shell_dims().to_vec();
        for (a, lexp) in local.iter().enumerate() {
            if lexp.iter().all(|v| *v == 0.0) {
                continue;
            }
            let center = centers[a];
            for &bp in &pair_lists[a] {
                let (s1, s2) = (self.pairs[bp].s1, self.pairs[bp].s2);
                let m = shell_pair_multipoles(
                    &self.shells[s1],
                    &self.shells[s2],
                    center,
                    l_max,
                    &self.cart_transforms,
                );
                let (n1, n2) = (dims[s1], dims[s2]);
                let (o1, o2) = (offs[s1], offs[s2]);
                for aa in 0..n1 {
                    for bb in 0..n2 {
                        let base = (aa * n2 + bb) * n_mom;
                        let mut acc = 0.0;
                        for t in 0..n_mom {
                            acc += lexp[t] * m[base + t];
                        }
                        j[(o1 + aa, o2 + bb)] += acc;
                    }
                }
            }
        }
    }
}

/// Multipole-to-local (M2L) translation.
///
/// Given source multipoles `M_i'j'k'` about `src_center`, add to `local` the
/// Taylor coefficients of the potential they produce about `dst_center`:
///
/// ```text
///   Φ(r) = Σ_{i'j'k'} M_{i'j'k'} · (−1)^{i'+j'+k'}/(i'!j'!k'!) · ∂^{i'j'k'}(1/|r−src|)
/// ```
///
/// Taylor-expanding about `dst_center` and collecting powers of `(r−dst)`
/// gives, with `R = dst_center − src_center`,
///
/// ```text
///   L_ijk = 1/(i!j!k!) · Σ_{i'j'k'} M_{i'j'k'} · (−1)^{L'}/(i'!j'!k'!)
///                        · H_{i+i', j+j', k+k'}(R)
/// ```
///
/// where `H` are the Cartesian derivatives of 1/r from
/// [`compute_cartesian_derivatives`]. Both factorial factors are folded into
/// `L` here, so the energy contraction downstream is a plain `Σ L_ijk M_ijk`.
fn add_m2l(
    local: &mut [f64],
    src_multipoles: &[f64],
    dst_center: [f64; 3],
    src_center: [f64; 3],
    l_max: usize,
) {
    let r = [
        dst_center[0] - src_center[0],
        dst_center[1] - src_center[1],
        dst_center[2] - src_center[2],
    ];
    // Derivatives are needed up to combined order 2*l_max.
    let h = compute_cartesian_derivatives(r, 2 * l_max);

    for l in 0..=l_max {
        for i in 0..=l {
            for j in 0..=(l - i) {
                let k = l - i - j;
                let inv_fact = 1.0
                    / (factorial(i) as f64 * factorial(j) as f64 * factorial(k) as f64);
                let mut val = 0.0;
                for lp in 0..=l_max {
                    let sign = if lp % 2 == 0 { 1.0 } else { -1.0 };
                    for ip in 0..=lp {
                        for jp in 0..=(lp - ip) {
                            let kp = lp - ip - jp;
                            let m = src_multipoles[ijk_to_idx(ip, jp, kp)];
                            if m == 0.0 {
                                continue;
                            }
                            let f = factorial(ip) as f64
                                * factorial(jp) as f64
                                * factorial(kp) as f64;
                            val += sign * m / f * h[ijk_to_idx(i + ip, j + jp, k + kp)];
                        }
                    }
                }
                local[ijk_to_idx(i, j, k)] += inv_fact * val;
            }
        }
    }
}

/// Write per-leaf multipoles back into the tree in depth-first leaf order —
/// the same order [`CfmmBox::collect_leaves`] produces, so index `i` of the
/// iterator lands on leaf `i` of that traversal.
fn write_leaf_multipoles(node: &mut CfmmBox, it: &mut impl Iterator<Item = Vec<f64>>) {
    if let Some(children) = &mut node.children {
        for c in children.iter_mut() {
            write_leaf_multipoles(c, it);
        }
    } else if let Some(m) = it.next() {
        node.multipoles = m;
    }
}

/// Clear cached multipoles / local expansions on every leaf.
fn clear_leaf_caches(node: &mut CfmmBox) {
    if let Some(children) = &mut node.children {
        for c in children.iter_mut() {
            clear_leaf_caches(c);
        }
    } else {
        node.multipoles.clear();
        node.local_exp.clear();
    }
}

impl JBuilder for CfmmJ {
    fn build(&mut self, d: &Array2<f64>, j: &mut Array2<f64>) -> Result<usize, FerricError> {
        j.fill(0.0);
        let mut far_pairs: Vec<(usize, usize)> = Vec::new();
        let n = self.near_field(d, j, &mut far_pairs)?;
        if !far_pairs.is_empty() {
            self.upward_pass(d);
            self.far_field(j, &far_pairs);
        }
        Ok(n)
    }

    fn reset(&mut self) {
        clear_leaf_caches(&mut self.root);
    }
}

// =====================================================================
// Small helpers
// =====================================================================

fn factorial(n: usize) -> usize {
    (1..=n).product()
}

/// Shift a Cartesian expansion from center C to C' (`d = C − C'`).
///
/// NOT used by the current flat, single-level far field (M2L runs directly
/// between leaf boxes). It is the M2M/L2L translation operator a hierarchical
/// upward/downward pass would need, is covered by `test_cartesian_shift`, and
/// is kept deliberately rather than deleted — see the module's STATUS note.
///
/// Binomial (Taylor) translation of primitive Cartesian moments:
/// `M'_ijk = Σ_{i'≤i, j'≤j, k'≤k} C(i,i')C(j,j')C(k,k') d^{i−i'} M_{i'j'k'}`.
#[allow(dead_code)]
fn shift_cartesian(src: &[f64], dst: &mut [f64], d: [f64; 3], l_max: usize) {
    let mut idx = 0;
    for l in 0..=l_max {
        for i in 0..=l {
            for j in 0..=(l - i) {
                let k = l - i - j;
                let mut val = 0.0;
                for ip in 0..=i {
                    for jp in 0..=j {
                        for kp in 0..=k {
                            let src_idx = ijk_to_idx(ip, jp, kp);
                            let factor = n_choose_k(i, ip) * n_choose_k(j, jp) * n_choose_k(k, kp);
                            let dx = d[0].powi((i - ip) as i32);
                            let dy = d[1].powi((j - jp) as i32);
                            let dz = d[2].powi((k - kp) as i32);
                            val += factor as f64 * dx * dy * dz * src[src_idx];
                        }
                    }
                }
                dst[idx] += val;
                idx += 1;
            }
        }
    }
}

/// Flat index of the Cartesian triple `(i,j,k)`, ordered by total degree
/// `l = i+j+k`, then by `i` ascending, then `j` ascending.
fn ijk_to_idx(i: usize, j: usize, k: usize) -> usize {
    let l = i + j + k;
    let base = l * (l + 1) * (l + 2) / 6;
    // Position within degree-l block: enumerate (ip, jp) in the same order.
    let mut idx = base;
    for ip in 0..=l {
        for jp in 0..=(l - ip) {
            if ip == i && jp == j {
                return idx;
            }
            idx += 1;
        }
    }
    idx
}

fn n_choose_k(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    let mut res = 1;
    for i in 0..k {
        res = res * (n - i) / (i + 1);
    }
    res
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rhf::{build_jk, solve_rhf, RhfConfig};
    use crate::screening::SchwarzBounds;
    use ferric_core::basis;
    use ferric_core::mol::Molecule;
    use ferric_core::parallel::ParallelContext;
    use ferric_integrals::operator::Operator;

    fn atom_z(mol: &Molecule) -> Vec<i32> {
        mol.atoms.iter().map(|a| a.z).collect()
    }

    /// Run RHF to convergence and return the density and molecule.
    fn converged_density(mol: &Molecule, basis_name: &str) -> Array2<f64> {
        let bs = basis::bundled(basis_name).unwrap();
        let prep = PreparedBasis::new(mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let config = RhfConfig {
            energy_conv: 1e-10,
            density_conv: 1e-8,
            integral_thresh: 1e-14,
            ..Default::default()
        };
        let result =
            solve_rhf(&ParallelContext::default(), mol, &prep, op, &bounds, &config).unwrap();
        assert!(result.converged, "RHF did not converge");
        result.density_total
    }

    /// Ground truth: J from the direct (screened-quartet) builder in rhf.rs.
    fn direct_j(mol: &Molecule, basis_name: &str, d: &Array2<f64>) -> Array2<f64> {
        let bs = basis::bundled(basis_name).unwrap();
        let prep = PreparedBasis::new(mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let n = prep.nbasis();
        let mut j = Array2::zeros((n, n));
        let mut k = Array2::zeros((n, n));
        build_jk(&ParallelContext::default(), &prep, &bounds, 1e-14, d, &mut j, &mut k).unwrap();
        j
    }

    fn cfmm_j(mol: &Molecule, basis_name: &str, d: &Array2<f64>, cfg: CfmmConfig) -> Array2<f64> {
        let bs = basis::bundled(basis_name).unwrap();
        let prep = PreparedBasis::new(mol, &bs).unwrap();
        let n = prep.nbasis();
        let mut cfmm = CfmmJ::new(prep, bs, &atom_z(mol), cfg).unwrap();
        let mut j = Array2::zeros((n, n));
        JBuilder::build(&mut cfmm, d, &mut j).unwrap();
        j
    }

    fn max_abs_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f64, f64::max)
    }

    fn water() -> Molecule {
        Molecule::parse_xyz(
            "3\nwater\nO 0.000000 0.000000 0.117790\nH 0.000000 0.755453 -0.471161\nH 0.000000 -0.755453 -0.471161\n",
            0, 1,
        )
        .unwrap()
    }

    // =================================================================
    // LAYER 1 — component anchors against INDEPENDENT constructions.
    // =================================================================

    /// Order-0 Cartesian multipole moments ARE the overlap matrix.
    ///
    /// Independent construction: `oneelectron::overlap` goes through libint2,
    /// sharing no code with `shell_pair_multipoles`. This pins the primitive
    /// normalization, the contraction, the Cartesian component ordering AND
    /// the Cartesian→pure transform in one shot — the parts most likely to
    /// silently disagree with the ERI engine.
    #[test]
    fn multipole_order0_matches_libint_overlap() {
        for (name, basis_name) in [("sto-3g", "sto-3g"), ("cc-pvdz", "cc-pvdz")] {
            let _ = name;
            let mol = water();
            let bs = basis::bundled(basis_name).unwrap();
            let prep = PreparedBasis::new(&mol, &bs).unwrap();
            let s_ref = ferric_integrals::oneelectron::overlap(&prep);

            let shells = gather_shells(&prep, &atom_z(&mol), &bs).unwrap();
            let max_l = shells.iter().map(|s| s.l).max().unwrap();
            let mut tf = vec![None; max_l + 1];
            for sh in &shells {
                if sh.pure && tf[sh.l].is_none() {
                    tf[sh.l] = Some(cart_to_pure_matrix(sh.l).unwrap());
                }
            }
            let offs = prep.shell_offsets();
            let dims = prep.shell_dims();
            let mut worst = 0.0f64;
            for s1 in 0..prep.nshells() {
                for s2 in 0..prep.nshells() {
                    let m = shell_pair_multipoles(
                        &shells[s1], &shells[s2], [0.0; 3], 0, &tf,
                    );
                    for a in 0..dims[s1] {
                        for b in 0..dims[s2] {
                            let got = m[a * dims[s2] + b];
                            let want = s_ref[(offs[s1] + a, offs[s2] + b)];
                            worst = worst.max((got - want).abs());
                        }
                    }
                }
            }
            assert!(
                worst < 1e-10,
                "{basis_name}: order-0 multipoles vs libint2 overlap: max diff {worst:.3e}"
            );
        }
    }

    /// Order-1 Cartesian multipole moments ARE the dipole integrals.
    /// Independent construction: `oneelectron::dipole` (libint2 OP_EMULTIPOLE1).
    #[test]
    fn multipole_order1_matches_libint_dipole() {
        let mol = water();
        for basis_name in ["sto-3g", "cc-pvdz"] {
            let bs = basis::bundled(basis_name).unwrap();
            let prep = PreparedBasis::new(&mol, &bs).unwrap();
            let origin = [0.3, -0.2, 0.15];
            let dip = ferric_integrals::oneelectron::dipole(&prep, origin).unwrap();

            let shells = gather_shells(&prep, &atom_z(&mol), &bs).unwrap();
            let max_l = shells.iter().map(|s| s.l).max().unwrap();
            let mut tf = vec![None; max_l + 1];
            for sh in &shells {
                if sh.pure && tf[sh.l].is_none() {
                    tf[sh.l] = Some(cart_to_pure_matrix(sh.l).unwrap());
                }
            }
            let offs = prep.shell_offsets();
            let dims = prep.shell_dims();
            let n_mom = n_moments(1);
            // (i,j,k) for x, y, z first moments.
            let axes = [
                (ijk_to_idx(1, 0, 0), 0usize),
                (ijk_to_idx(0, 1, 0), 1),
                (ijk_to_idx(0, 0, 1), 2),
            ];
            let mut worst = 0.0f64;
            for s1 in 0..prep.nshells() {
                for s2 in 0..prep.nshells() {
                    let m = shell_pair_multipoles(&shells[s1], &shells[s2], origin, 1, &tf);
                    for a in 0..dims[s1] {
                        for b in 0..dims[s2] {
                            for &(midx, ax) in &axes {
                                let got = m[(a * dims[s2] + b) * n_mom + midx];
                                let want = dip[ax][(offs[s1] + a, offs[s2] + b)];
                                worst = worst.max((got - want).abs());
                            }
                        }
                    }
                }
            }
            assert!(
                worst < 1e-10,
                "{basis_name}: order-1 multipoles vs libint2 dipole: max diff {worst:.3e}"
            );
        }
    }

    /// The 1/r Cartesian derivative tensor vs central finite differences.
    ///
    /// Independent construction: FD of `1/r` itself, sharing no code with the
    /// polynomial-numerator recurrence. This is the anchor that catches the
    /// class of failure the PREVIOUS implementation had — its general
    /// recurrence loop was a no-op that silently returned 0.0 for every order
    /// ≥ 3 while looking plausible.
    #[test]
    fn cartesian_derivatives_match_finite_differences() {
        let d = [0.7, -1.3, 2.1];
        let l_max = 4;
        let h = compute_cartesian_derivatives(d, l_max);

        // Nested central differences of 1/r, evaluated with a step chosen to
        // balance truncation and round-off for the orders tested.
        fn inv_r(p: [f64; 3]) -> f64 {
            1.0 / (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt()
        }
        fn fd(p: [f64; 3], order: [usize; 3], step: f64) -> f64 {
            let axis = (0..3).find(|&a| order[a] > 0);
            match axis {
                None => inv_r(p),
                Some(a) => {
                    let mut lo = order;
                    lo[a] -= 1;
                    let mut pp = p;
                    let mut pm = p;
                    pp[a] += step;
                    pm[a] -= step;
                    (fd(pp, lo, step) - fd(pm, lo, step)) / (2.0 * step)
                }
            }
        }

        // Richardson extrapolation over two step sizes: the nested central
        // difference is O(h²), so (4·F(h/2) − F(h))/3 is O(h⁴) and gives a
        // reference accurate enough to assert at 1e-6 rather than 1e-3.
        // Without it the FD reference itself carries ~4e-3 relative error at
        // order 2 and the test could not distinguish a real bug from FD noise.
        let richardson = |order: [usize; 3]| -> f64 {
            let h1 = 0.04;
            let f1 = fd(d, order, h1);
            let f2 = fd(d, order, h1 / 2.0);
            (4.0 * f2 - f1) / 3.0
        };

        let mut worst = 0.0f64;
        for l in 0..=l_max {
            for i in 0..=l {
                for j in 0..=(l - i) {
                    let k = l - i - j;
                    let got = h[ijk_to_idx(i, j, k)];
                    let want = richardson([i, j, k]);
                    let rel = (got - want).abs() / want.abs().max(1e-6);
                    worst = worst.max(rel);
                    // 1e-4 is set by the RICHARDSON-FD REFERENCE's own residual
                    // error at order 4 (the recurrence itself is exact — it
                    // reproduces symbolic derivatives to ~1e-14), not by any
                    // slack in the recurrence. It is still ~4 orders of
                    // magnitude tighter than needed to catch the previous
                    // implementation, which returned exactly 0.0 for all
                    // orders >= 3 (rel err 1.0).
                    assert!(
                        rel < 1e-4,
                        "H_{i}{j}{k}: recurrence {got:.10e} vs Richardson-FD {want:.10e} (rel {rel:.2e})"
                    );
                }
            }
        }
        println!("cartesian derivative worst rel err vs FD: {worst:.3e}");
    }

    /// The multipole expansion of a well-separated pair of charge
    /// distributions must reproduce the exact ERI as l_max grows.
    ///
    /// This checks M2L + the moment definitions TOGETHER against exact
    /// 4-center ERIs — an independent construction (libint2) — and asserts
    /// the error DECREASES with l_max, which a sign/normalization error in
    /// `add_m2l` would not do.
    #[test]
    fn m2l_far_field_converges_to_exact_eri_with_order() {
        // Two well-separated H2 molecules along z.
        let mol = Molecule::parse_xyz(
            "4\ntwo H2\nH 0.0 0.0 0.0\nH 0.0 0.0 0.74\nH 0.0 0.0 20.0\nH 0.0 0.0 20.74\n",
            0, 1,
        )
        .unwrap();
        let bs = basis::bundled("sto-3g").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let shells = gather_shells(&prep, &atom_z(&mol), &bs).unwrap();
        let tf: Vec<Option<Vec<f64>>> = vec![None; 1];

        // Exact (00|22) style: bra pair = shells 0,1 (first H2);
        // ket pair = shells 2,3 (second H2).
        let mut eng = Engine::new_2e(Operator::coulomb(), &prep, 1e-14).unwrap();
        let exact = eng.compute_quartet(&prep, 0, 1, 2, 3).unwrap()[0];

        // Box centers = midpoints of each H2, taken from `shell_centers()`
        // so they are in BOHR. (Molecule::parse_xyz takes Angstrom input and
        // converts; hardcoding the .xyz numbers here would silently place the
        // expansion centers 1.89x too close and destroy the convergence.)
        let sc = prep.shell_centers();
        let mid = |a: [f64; 3], b: [f64; 3]| [
            (a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0, (a[2] + b[2]) / 2.0,
        ];
        let bra_c = mid(sc[0], sc[1]);
        let ket_c = mid(sc[2], sc[3]);

        let mut prev = f64::INFINITY;
        let mut errs: Vec<f64> = Vec::new();
        for l_max in [0usize, 1, 2, 3] {
            let n_mom = n_moments(l_max);
            let m_ket =
                shell_pair_multipoles(&shells[2], &shells[3], ket_c, l_max, &tf);
            let m_bra =
                shell_pair_multipoles(&shells[0], &shells[1], bra_c, l_max, &tf);
            let mut local = vec![0.0; n_mom];
            add_m2l(&mut local, &m_ket[..n_mom], bra_c, ket_c, l_max);
            let approx: f64 = (0..n_mom).map(|t| local[t] * m_bra[t]).sum();
            let err = (approx - exact).abs() / exact.abs();
            println!("l_max={l_max}: approx={approx:.12e} exact={exact:.12e} rel={err:.3e}");
            // MUST NOT get worse. Strict decrease is asserted across the even
            // orders below instead: this dimer is centrosymmetric about each
            // box center, so the ODD moments vanish identically and l=1 / l=3
            // legitimately add nothing over l=0 / l=2. Demanding strict
            // improvement at every order would be asserting a property of the
            // test molecule's symmetry, not of the M2L operator.
            assert!(
                err <= prev * (1.0 + 1e-12),
                "multipole error must not INCREASE with l_max: l_max={l_max} \
                 rel={err:.3e} worse than previous {prev:.3e}"
            );
            errs.push(err);
            prev = err;
        }
        // The even orders must each be a genuine improvement: monopole →
        // quadrupole here buys ~4 orders of magnitude. A sign error or a
        // wrong factorial in `add_m2l` shows up as this plateauing or growing.
        assert!(
            errs[2] < errs[0] * 1e-2,
            "l_max=2 ({:.3e}) must be much better than l_max=0 ({:.3e})",
            errs[2], errs[0]
        );
        assert!(
            prev < 1e-6,
            "converged multipole error {prev:.3e} still too large at l_max=3"
        );
    }

    // =================================================================
    // LAYER 2 — THE EXACTNESS ANCHOR (project protocol: this comes first
    // and must pass before ANY performance or accuracy claim is made).
    // =================================================================

    /// **EXACTNESS ANCHOR.** In the trivial limit — `force_all_near_field`,
    /// i.e. the well-separatedness criterion never fires — CFMM does nothing
    /// approximate: every leaf pair goes through exact 4-center ERIs. Its J
    /// must therefore reproduce the direct/dense builder to INTEGRAL
    /// precision, not to some multipole error budget.
    ///
    /// This is the anchor the project's experimental protocol requires to
    /// exist BEFORE anything is measured. Its value is that it isolates the
    /// parts that carry no approximation — octree construction, leaf
    /// enumeration, the (bra-leaf, ket-leaf) pair loop, the shell-quartet
    /// scatter and the density contraction — from all multipole numerics.
    /// A failure here is unambiguously a traversal or scatter bug (a
    /// double-counted pair, a missed leaf, a transposed index), which is
    /// exactly the class of error a multipole-tolerance test would hide.
    ///
    /// The reference (`build_jk` in rhf.rs) is an INDEPENDENT construction:
    /// a screened canonical-quartet loop with 8-fold symmetry folding,
    /// sharing no code with CFMM's unsymmetrized leaf-pair loop.
    #[test]
    fn cfmm_matches_direct_j_in_the_trivial_limit() {
        for (label, basis_name, depth) in [
            ("water/STO-3G d1", "sto-3g", 1usize),
            ("water/STO-3G d2", "sto-3g", 2),
            ("water/STO-3G d3", "sto-3g", 3),
            ("water/cc-pVDZ d2", "cc-pvdz", 2),
        ] {
            let mol = water();
            let d = converged_density(&mol, basis_name);
            let j_ref = direct_j(&mol, basis_name, &d);
            let cfg = CfmmConfig {
                l_max: 4,
                max_level: depth,
                force_all_near_field: true,
                ..Default::default()
            };
            let j_cfmm = cfmm_j(&mol, basis_name, &d, cfg);
            let diff = max_abs_diff(&j_ref, &j_cfmm);
            let scale = j_ref.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
            println!("[anchor] {label}: max|J_ref|={scale:.6e} max diff={diff:.3e}");
            assert!(
                diff < 1e-10,
                "{label}: CFMM in the trivial (all-near-field) limit must be EXACT, \
                 but max diff vs direct J is {diff:.3e} (max|J_ref| = {scale:.3e}). \
                 This is a traversal/scatter bug, not multipole truncation."
            );
        }
    }

    /// The trivial-limit anchor must also hold on a system big enough that
    /// the octree really has many populated leaves (so the leaf-pair loop is
    /// non-trivially exercised), not just the 3-atom case.
    #[test]
    fn cfmm_trivial_limit_exact_on_alkane() {
        let mol = Molecule::load_xyz("../../testdata/molecules/alkane_4.xyz").unwrap();
        let d = converged_density(&mol, "sto-3g");
        let j_ref = direct_j(&mol, "sto-3g", &d);
        let cfg = CfmmConfig {
            l_max: 4,
            max_level: 3,
            force_all_near_field: true,
            ..Default::default()
        };
        let j_cfmm = cfmm_j(&mol, "sto-3g", &d, cfg);
        let diff = max_abs_diff(&j_ref, &j_cfmm);
        println!("[anchor] alkane_4/STO-3G: max diff = {diff:.3e}");
        assert!(diff < 1e-10, "alkane_4 trivial-limit max diff {diff:.3e}");
    }

    // =================================================================
    // LAYER 3 — the real thing: far field ENGAGED on a molecule long
    // enough that a meaningful fraction of leaf pairs are well separated.
    // =================================================================

    /// Count how many leaf pairs actually take the far-field path, so the
    /// test below can PROVE the multipole path is engaged rather than
    /// assuming it (a test that silently ran everything through the exact
    /// near field would pass trivially and prove nothing).
    fn far_pair_fraction(mol: &Molecule, basis_name: &str, cfg: &CfmmConfig) -> (usize, usize) {
        let bs = basis::bundled(basis_name).unwrap();
        let prep = PreparedBasis::new(mol, &bs).unwrap();
        let cfmm = CfmmJ::new(prep, bs, &atom_z(mol), cfg.clone()).unwrap();
        let mut leaves: Vec<&CfmmBox> = Vec::new();
        cfmm.root.collect_leaves(&mut leaves);
        let extents = cfmm.leaf_extents(&leaves);
        let occ: Vec<usize> = (0..leaves.len())
            .filter(|&i| !leaves[i].pair_indices.is_empty())
            .collect();
        let mut far = 0;
        let mut total = 0;
        for &a in &occ {
            for &b in &occ {
                total += 1;
                if cfmm.is_far_ext(leaves[a], leaves[b], extents[a], extents[b]) {
                    far += 1;
                }
            }
        }
        (far, total)
    }

    /// CFMM with the far field ENGAGED, on alkane_10 (a 32-atom chain, long
    /// enough that its ends are genuinely well separated), against the
    /// direct/dense J.
    ///
    /// Unlike the trivial-limit anchor this one IS approximate — that is the
    /// whole point of CFMM — so the tolerance is a multipole-truncation
    /// budget rather than integral precision. The test asserts three things:
    ///
    /// 1. the far-field path is actually exercised (non-zero far pairs),
    /// 2. the resulting J is accurate to the stated budget, and
    /// 3. increasing `l_max` makes it MORE accurate — the signature of a
    ///    real multipole expansion. A construction bug typically pins the
    ///    error at a fixed value or makes it grow, so (3) is the assertion
    ///    that distinguishes "converging expansion" from "plausible number".
    #[test]
    fn cfmm_far_field_engaged_matches_direct_j_on_alkane10() {
        let mol = Molecule::load_xyz("../../testdata/molecules/alkane_10.xyz").unwrap();
        let d = converged_density(&mol, "sto-3g");
        let j_ref = direct_j(&mol, "sto-3g", &d);
        let scale = j_ref.iter().map(|v| v.abs()).fold(0.0f64, f64::max);

        let base = CfmmConfig {
            max_level: 3,
            ws_factor: 2.0,
            force_all_near_field: false,
            ..Default::default()
        };
        let (far, total) = far_pair_fraction(&mol, "sto-3g", &base);
        println!(
            "[far] alkane_10: {far}/{total} occupied leaf pairs are well separated \
             ({:.1}%)",
            100.0 * far as f64 / total as f64
        );
        assert!(
            far > 0,
            "far-field path never engages on alkane_10 — this test would prove nothing"
        );

        let mut errs = Vec::new();
        for l_max in [2usize, 4, 6] {
            let cfg = CfmmConfig { l_max, ..base.clone() };
            let j_cfmm = cfmm_j(&mol, "sto-3g", &d, cfg);
            let diff = max_abs_diff(&j_ref, &j_cfmm);
            println!("[far] alkane_10 l_max={l_max}: max diff = {diff:.3e} (max|J| = {scale:.3e})");
            errs.push(diff);
        }

        assert!(
            errs[1] < errs[0] && errs[2] <= errs[1],
            "far-field error must fall with l_max, got {errs:?} — a flat or \
             growing series means the multipole path is broken, not truncated"
        );
        // MEASURED budget, not an aspirational one. With the extent-aware
        // (continuous) separation criterion the alkane_10/STO-3G far-field
        // error is 2.63e-4 (l_max=2) → 8.24e-5 (l_max=4) against
        // max|J| = 2.4e1, i.e. ~3e-6 relative.
        //
        // The floor here is NOT multipole truncation — it is the residual
        // charge overlap admitted by `extent_thresh` (1e-10 density cutoff),
        // so pushing l_max further buys little. Tightening `extent_thresh`
        // moves this floor down at the cost of pushing more pairs into the
        // exact near field. Recording the measurement rather than asserting
        // a round number keeps this honest about what was actually observed.
        //
        // For contrast, the PURELY GEOMETRIC criterion (no extent terms) gave
        // 5.41e-2 → 1.22e-2 on the same system — ~200x worse at l_max=2 and
        // stalling, because it expands 1/r between overlapping charge clouds.
        assert!(
            errs[2] < 5e-4,
            "alkane_10 far-field J error at l_max=6 is {:.3e}, above the \
             measured 5e-4 budget",
            errs[2]
        );
    }

    // =================================================================
    // Structural tests (carried over from the pre-rewrite module).
    // =================================================================

    /// Octree insertion places every pair in exactly one leaf.
    #[test]
    fn test_cfmm_octree_insertion() {
        let mut root = CfmmBox::new([0.0, 0.0, 0.0], 10.0, 0);
        root.insert_pair(0, [1.0, 1.0, 1.0], 2);
        root.insert_pair(1, [-1.0, -1.0, -1.0], 2);
        root.insert_pair(2, [0.1, 0.1, 0.1], 2);

        assert!(root.children.is_some());

        fn count(node: &CfmmBox) -> usize {
            let mut c = node.pair_indices.len();
            if let Some(children) = &node.children {
                for ch in children.iter() {
                    c += count(ch);
                }
            }
            c
        }
        assert_eq!(count(&root), 3);
    }

    /// Binomial Cartesian translation of a pure monopole: shifting a unit
    /// charge by `d` must produce first moments equal to `d`.
    #[test]
    fn test_cartesian_shift() {
        let l_max = 1;
        let n = n_moments(l_max);
        let mut src = vec![0.0; n];
        let mut dst = vec![0.0; n];

        src[0] = 1.0;
        let d = [1.0, 2.0, 3.0];
        shift_cartesian(&src, &mut dst, d, l_max);

        assert_eq!(dst[ijk_to_idx(0, 0, 0)], 1.0);
        assert_eq!(dst[ijk_to_idx(1, 0, 0)], 1.0);
        assert_eq!(dst[ijk_to_idx(0, 1, 0)], 2.0);
        assert_eq!(dst[ijk_to_idx(0, 0, 1)], 3.0);
    }

    /// `ijk_to_idx` must be a bijection onto `0..n_moments(l_max)` — every
    /// routine here (moments, derivatives, shifts, M2L) indexes through it, so
    /// a collision would silently mix unrelated moments.
    #[test]
    fn ijk_to_idx_is_a_bijection() {
        for l_max in 0..=8usize {
            let n = n_moments(l_max);
            let mut seen = vec![false; n];
            for l in 0..=l_max {
                for i in 0..=l {
                    for j in 0..=(l - i) {
                        let k = l - i - j;
                        let idx = ijk_to_idx(i, j, k);
                        assert!(idx < n, "l_max={l_max}: index {idx} out of range {n}");
                        assert!(!seen[idx], "l_max={l_max}: collision at ({i},{j},{k})");
                        seen[idx] = true;
                    }
                }
            }
            assert!(seen.iter().all(|&s| s), "l_max={l_max}: gaps in packing");
        }
    }
}
