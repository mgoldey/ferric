//! Safe wrapper over the libecpint ECP shim ([`crate::ecp_ffi`]).
//!
//! Computes the dense **spherical** ECP matrix `V_ECP` for a molecule with ECP
//! centers, matching libint's spherical AO basis (and PySCF's `ECPscalar`).
//!
//! libecpint computes integrals over **bare Cartesian** Gaussians and applies no
//! internal normalization. To match the production (spherical) convention this
//! wrapper:
//!   1. requires the caller to supply contraction coefficients with the
//!      primitive normalization `gto_norm(l, α)` folded in (the bare-Cartesian
//!      convention libcint uses with `cart=True`);
//!   2. calls the shim to get the Cartesian `V_ECP`;
//!   3. applies the per-shell Cartesian→spherical transform `Cᵀ V Cᵀ` using the
//!      libcint `cart2sph` matrices.
//!
//! Verified: `c2sᵀ · (gto_norm-folded libecpint Cartesian) · c2s` reproduces
//! PySCF's spherical `ECPscalar` to ~1e-9 (see `tests/ecp_matrix.rs`).

use crate::ecp_ffi::{
    ferric_ecp_matrix, ferric_ecp_matrix_deriv, ferric_ecp_natoms, CEcpCenter, CEcpGShell,
    FERRIC_ECP_OK,
};
use ferric_core::FerricError;
use std::os::raw::c_int;

/// A Cartesian Gaussian basis shell for ECP evaluation.
///
/// `coefficients` must already include the primitive normalization
/// `gto_norm(l, α)` (i.e. be in the bare-Cartesian convention libecpint expects).
#[derive(Debug, Clone)]
pub struct EcpGaussianShell {
    pub l: i32,
    pub center: [f64; 3],
    pub exponents: Vec<f64>,
    pub coefficients: Vec<f64>,
}

/// One ECP center: a flat list of semilocal primitives, each tagged with its
/// angular momentum, r-power, exponent, and coefficient. The maximum-`am`
/// channel is the local term (libecpint determines this).
#[derive(Debug, Clone)]
pub struct EcpCenter {
    pub center: [f64; 3],
    pub ams: Vec<i32>,
    pub ns: Vec<i32>,
    pub exponents: Vec<f64>,
    pub coefficients: Vec<f64>,
}

/// libcint primitive Gaussian normalization `gto_norm(l, α)`:
/// `sqrt( 2^(2l+3) (l+1)! (2α)^(l+3/2) / ( (2l+2)! √π ) )`.
///
/// Fold this into a stored (non-primitive-normalized) contraction coefficient to
/// obtain the bare-Cartesian coefficient libecpint expects.
pub fn gto_norm(l: i32, alpha: f64) -> f64 {
    fn factorial(n: u64) -> f64 {
        (1..=n).map(|k| k as f64).product()
    }
    let lf = l as f64;
    let num = 2f64.powf(2.0 * lf + 3.0) * factorial((l + 1) as u64) * (2.0 * alpha).powf(lf + 1.5);
    let den = factorial((2 * l + 2) as u64) * std::f64::consts::PI.sqrt();
    (num / den).sqrt()
}

#[inline]
fn ncart(l: i32) -> usize {
    (((l + 1) * (l + 2)) / 2) as usize
}

#[inline]
fn nsph(l: i32) -> usize {
    (2 * l + 1) as usize
}

/// Per-shell Cartesian→spherical transform matrices `C` (ncart × nsph), in
/// libcint convention, row-major. `V_sph = Cᵀ V_cart C`.
/// Supported up to l = 4 (g) — covers def2 / cc-pVnZ-PP orbital bases.
fn cart2sph(l: i32) -> &'static [f64] {
    match l {
        0 => &C2S0,
        1 => &C2S1,
        2 => &C2S2,
        3 => &C2S3,
        4 => &C2S4,
        _ => &[],
    }
}

// l=0: (1,1)
static C2S0: [f64; 1] = [0.282_094_791_773_878_14];
// l=1: (3,3)
static C2S1: [f64; 9] = [
    0.488_602_511_902_919_9, 0.0, 0.0,
    0.0, 0.488_602_511_902_919_9, 0.0,
    0.0, 0.0, 0.488_602_511_902_919_9,
];
// l=2: (6,5)
static C2S2: [f64; 30] = [
    0.0, 0.0, -0.315_391_565_252_52, 0.0, 0.546_274_215_296_039_6,
    1.092_548_430_592_079_2, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 1.092_548_430_592_079_2, 0.0,
    0.0, 0.0, -0.315_391_565_252_52, 0.0, -0.546_274_215_296_039_6,
    0.0, 1.092_548_430_592_079_2, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.630_783_130_505_04, 0.0, 0.0,
];
// l=3: (10,7)
static C2S3: [f64; 70] = [
    0.0, 0.0, 0.0, 0.0, -0.457_045_799_464_465_7, 0.0, 0.590_043_589_926_643_5,
    1.770_130_769_779_930_4, 0.0, -0.457_045_799_464_465_7, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, -1.119_528_997_770_346_2, 0.0, 1.445_305_721_320_277_1, 0.0,
    0.0, 0.0, 0.0, 0.0, -0.457_045_799_464_465_7, 0.0, -1.770_130_769_779_930_4,
    0.0, 2.890_611_442_640_554_3, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 1.828_183_197_857_862_9, 0.0, 0.0,
    -0.590_043_589_926_643_5, 0.0, -0.457_045_799_464_465_7, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, -1.119_528_997_770_346_2, 0.0, -1.445_305_721_320_277_1, 0.0,
    0.0, 0.0, 1.828_183_197_857_862_9, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.746_352_665_180_230_8, 0.0, 0.0, 0.0,
];
// l=4: (15,9)
static C2S4: [f64; 135] = [
    0.0, 0.0, 0.0, 0.0, 0.317_356_640_745_612_93, 0.0, -0.473_087_347_878_78, 0.0, 0.625_835_735_449_176_1,
    2.503_342_941_796_704_6, 0.0, -0.946_174_695_757_56, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, -2.007_139_630_671_867_6, 0.0, 1.770_130_769_779_930_4, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.634_713_281_491_225_9, 0.0, 0.0, 0.0, -3.755_014_412_695_057,
    0.0, 5.310_392_309_339_791, 0.0, -2.007_139_630_671_867_6, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, -2.538_853_125_964_903_4, 0.0, 2.838_524_087_272_680_2, 0.0, 0.0,
    -2.503_342_941_796_704_6, 0.0, -0.946_174_695_757_56, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, -2.007_139_630_671_867_6, 0.0, -5.310_392_309_339_791, 0.0,
    0.0, 0.0, 5.677_048_174_545_360_5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 2.676_186_174_229_157, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.317_356_640_745_612_93, 0.0, 0.473_087_347_878_78, 0.0, 0.625_835_735_449_176_1,
    0.0, -1.770_130_769_779_930_4, 0.0, -2.007_139_630_671_867_6, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, -2.538_853_125_964_903_4, 0.0, -2.838_524_087_272_680_2, 0.0, 0.0,
    0.0, 0.0, 0.0, 2.676_186_174_229_157, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.846_284_375_321_634_5, 0.0, 0.0, 0.0, 0.0,
];

/// Compute the dense spherical ECP matrix `V_ECP` for the given Cartesian shells
/// and ECP centers. Returns an `nsph × nsph` matrix (row-major), where `nsph`
/// is the total number of spherical basis functions (`Σ_s (2 l_s + 1)`).
///
/// `shells` coefficients must be in the bare-Cartesian convention (primitive
/// `gto_norm` folded in). The shell order defines the output AO order, matching
/// libint's spherical ordering per shell.
pub fn ecp_matrix_spherical(
    shells: &[EcpGaussianShell],
    ecps: &[EcpCenter],
) -> Result<Vec<f64>, FerricError> {
    if shells.is_empty() || ecps.is_empty() {
        return Err(FerricError::Libint("ecp_matrix_spherical: empty input".into()));
    }
    for sh in shells {
        if sh.l > 4 {
            return Err(FerricError::Libint(format!(
                "ECP: angular momentum l={} > 4 not supported by cart2sph table",
                sh.l
            )));
        }
    }

    let (c_shells, c_ecps, _keep) = build_c_arrays(shells, ecps);

    let ncart_total: usize = shells.iter().map(|s| ncart(s.l)).sum();
    let mut v_cart = vec![0.0f64; ncart_total * ncart_total];
    // SAFETY: c_shells/c_ecps are valid C-repr arrays built by build_c_arrays
    // (backed by _keep). v_cart is pre-sized to ncart_total². Status checked.
    let status = unsafe {
        ferric_ecp_matrix(
            c_shells.as_ptr(),
            c_shells.len() as c_int,
            c_ecps.as_ptr(),
            c_ecps.len() as c_int,
            v_cart.as_mut_ptr(),
        )
    };
    if status != FERRIC_ECP_OK {
        return Err(FerricError::Libint(format!("ferric_ecp_matrix failed: {status}")));
    }

    Ok(cart_to_sph(shells, &v_cart))
}

/// Owns the `c_int` conversions and per-ECP vectors that the `CEcpCenter`
/// pointers alias. Must outlive any FFI call using those pointers.
struct CEcpBacking {
    _ams: Vec<Vec<c_int>>,
    _ns: Vec<Vec<c_int>>,
}

/// Build the C-ABI shell + ECP arrays. The returned [`CEcpBacking`] owns the
/// storage the `CEcpCenter` pointers alias and must be kept alive across the
/// FFI call (the `shells`/`ecps` slices themselves back the other pointers).
fn build_c_arrays(
    shells: &[EcpGaussianShell],
    ecps: &[EcpCenter],
) -> (Vec<CEcpGShell>, Vec<CEcpCenter>, CEcpBacking) {
    let c_shells: Vec<CEcpGShell> = shells
        .iter()
        .map(|s| CEcpGShell {
            l: s.l as c_int,
            nprim: s.exponents.len() as c_int,
            x: s.center[0],
            y: s.center[1],
            z: s.center[2],
            exponents: s.exponents.as_ptr(),
            coefficients: s.coefficients.as_ptr(),
        })
        .collect();

    // ECP int arrays need stable storage of the c_int conversions.
    let ams_store: Vec<Vec<c_int>> = ecps.iter().map(|e| e.ams.iter().map(|&a| a as c_int).collect()).collect();
    let ns_store: Vec<Vec<c_int>> = ecps.iter().map(|e| e.ns.iter().map(|&n| n as c_int).collect()).collect();
    let c_ecps: Vec<CEcpCenter> = ecps
        .iter()
        .enumerate()
        .map(|(i, e)| CEcpCenter {
            x: e.center[0],
            y: e.center[1],
            z: e.center[2],
            nterm: e.ams.len() as c_int,
            ams: ams_store[i].as_ptr(),
            ns: ns_store[i].as_ptr(),
            exponents: e.exponents.as_ptr(),
            coefficients: e.coefficients.as_ptr(),
        })
        .collect();

    (c_shells, c_ecps, CEcpBacking { _ams: ams_store, _ns: ns_store })
}

/// Cartesian -> spherical for one dense `ncart x ncart` matrix:
/// `V_sph = Cᵀ V_cart C`, applied block-diagonally per shell pair.
/// Returns an `nsph x nsph` row-major matrix.
fn cart_to_sph(shells: &[EcpGaussianShell], v_cart: &[f64]) -> Vec<f64> {
    let ncart_total: usize = shells.iter().map(|s| ncart(s.l)).sum();
    let nsph_total: usize = shells.iter().map(|s| nsph(s.l)).sum();
    // Cartesian and spherical offsets per shell.
    let mut cart_off = Vec::with_capacity(shells.len());
    let mut sph_off = Vec::with_capacity(shells.len());
    {
        let mut c = 0;
        let mut s = 0;
        for sh in shells {
            cart_off.push(c);
            sph_off.push(s);
            c += ncart(sh.l);
            s += nsph(sh.l);
        }
    }

    let mut v_sph = vec![0.0f64; nsph_total * nsph_total];
    // For each shell pair (A, B): V_sph[A,B] = C_Aᵀ V_cart[A,B] C_B.
    for (a, sha) in shells.iter().enumerate() {
        let nca = ncart(sha.l);
        let nsa = nsph(sha.l);
        let ca = cart2sph(sha.l); // (nca x nsa)
        for (b, shb) in shells.iter().enumerate() {
            let ncb = ncart(shb.l);
            let nsb = nsph(shb.l);
            let cb = cart2sph(shb.l); // (ncb x nsb)

            // tmp = V_cart[A,B] (nca x ncb) · C_B (ncb x nsb)  ->  (nca x nsb)
            let mut tmp = vec![0.0f64; nca * nsb];
            for i in 0..nca {
                for q in 0..nsb {
                    let mut acc = 0.0;
                    for k in 0..ncb {
                        let vc = v_cart[(cart_off[a] + i) * ncart_total + (cart_off[b] + k)];
                        acc += vc * cb[k * nsb + q];
                    }
                    tmp[i * nsb + q] = acc;
                }
            }
            // V_sph[A,B] = C_Aᵀ (nsa x nca) · tmp (nca x nsb) -> (nsa x nsb)
            for p in 0..nsa {
                for q in 0..nsb {
                    let mut acc = 0.0;
                    for i in 0..nca {
                        acc += ca[i * nsa + p] * tmp[i * nsb + q];
                    }
                    v_sph[(sph_off[a] + p) * nsph_total + (sph_off[b] + q)] = acc;
                }
            }
        }
    }

    v_sph
}

/// First derivatives of the spherical ECP matrix with respect to every atomic
/// coordinate: `dV_ECP/dR`.
///
/// Returns `(derivs, natoms)` where `derivs` has length `3 * natoms`, each entry
/// an `nsph × nsph` row-major matrix, ordered `{A_x, A_y, A_z, B_x, ...}`.
///
/// **The atom ordering is libecpint's, not the caller's.** libecpint takes no
/// atom list: it infers atom ids by deduplicating shell and ECP centers (1e-4
/// Bohr tolerance, shells first then ECP centers, in order of first appearance —
/// see `ECPIntegrator::init`). For a molecule where every atom carries at least
/// one basis shell and shells are emitted atom-by-atom, this coincides with the
/// caller's atom order; it does **not** in general. Callers must map ids back
/// via [`ecp_deriv_atom_ids`] rather than assuming 1:1 — see
/// [`crate::oneelectron::ecp_potential_deriv`], which does exactly that.
///
/// The A/B/C (bra, ket, ECP center) contributions are already summed per atom by
/// libecpint, so each matrix is the total derivative w.r.t. that one coordinate.
pub fn ecp_matrix_deriv_spherical(
    shells: &[EcpGaussianShell],
    ecps: &[EcpCenter],
) -> Result<(Vec<Vec<f64>>, usize), FerricError> {
    if shells.is_empty() || ecps.is_empty() {
        return Err(FerricError::Libint(
            "ecp_matrix_deriv_spherical: empty input".into(),
        ));
    }
    for sh in shells {
        if sh.l > 4 {
            return Err(FerricError::Libint(format!(
                "ECP gradient: angular momentum l={} > 4 not supported by cart2sph table",
                sh.l
            )));
        }
    }

    let (c_shells, c_ecps, _keep) = build_c_arrays(shells, ecps);

    // SAFETY: c_shells/c_ecps are valid C-repr arrays (backed by _keep).
    // Returns the number of atoms or a negative error code.
    let natoms = unsafe {
        ferric_ecp_natoms(
            c_shells.as_ptr(),
            c_shells.len() as c_int,
            c_ecps.as_ptr(),
            c_ecps.len() as c_int,
        )
    };
    if natoms <= 0 {
        return Err(FerricError::Libint(format!(
            "ferric_ecp_natoms failed: {natoms}"
        )));
    }
    let natoms = natoms as usize;

    let ncart_total: usize = shells.iter().map(|s| ncart(s.l)).sum();
    let mut d_cart = vec![0.0f64; 3 * natoms * ncart_total * ncart_total];
    let mut got_natoms: c_int = 0;
    // SAFETY: same arrays as above; d_cart sized for 3*natoms*ncart² doubles.
    let status = unsafe {
        ferric_ecp_matrix_deriv(
            c_shells.as_ptr(),
            c_shells.len() as c_int,
            c_ecps.as_ptr(),
            c_ecps.len() as c_int,
            d_cart.as_mut_ptr(),
            &mut got_natoms,
        )
    };
    if status != FERRIC_ECP_OK {
        return Err(FerricError::Libint(format!(
            "ferric_ecp_matrix_deriv failed: {status}"
        )));
    }
    if got_natoms as usize != natoms {
        return Err(FerricError::Libint(format!(
            "ECP gradient: natoms disagreement ({natoms} predicted, {got_natoms} computed)"
        )));
    }

    let block = ncart_total * ncart_total;
    let derivs = (0..3 * natoms)
        .map(|c| cart_to_sph(shells, &d_cart[c * block..(c + 1) * block]))
        .collect();
    Ok((derivs, natoms))
}

/// Map each of libecpint's inferred atom ids back to an index into `centers`
/// (the caller's own atom list, in Bohr).
///
/// Replicates `ECPIntegrator::init`'s dedup order — shell centers first in the
/// order given, then any ECP center not already seen — using the same 1e-4 Bohr
/// L1 tolerance. Returns a vector of length `natoms` whose `i`-th entry is the
/// index into `centers` that libecpint atom `i` corresponds to.
///
/// Errors if any inferred center cannot be matched to a caller atom, which would
/// mean the derivative rows could not be attributed and the gradient would be
/// silently misassigned.
pub fn ecp_deriv_atom_ids(
    shells: &[EcpGaussianShell],
    ecps: &[EcpCenter],
    centers: &[[f64; 3]],
) -> Result<Vec<usize>, FerricError> {
    const TOL: f64 = 1e-4;
    let close = |a: &[f64; 3], b: &[f64; 3]| {
        (a[0] - b[0]).abs() + (a[1] - b[1]).abs() + (a[2] - b[2]).abs() < TOL
    };

    let mut inferred: Vec<[f64; 3]> = Vec::new();
    let intern = |c: [f64; 3], inferred: &mut Vec<[f64; 3]>| {
        if !inferred.iter().any(|e| close(e, &c)) {
            inferred.push(c);
        }
    };
    for s in shells {
        intern(s.center, &mut inferred);
    }
    for e in ecps {
        intern(e.center, &mut inferred);
    }

    inferred
        .iter()
        .map(|c| {
            centers.iter().position(|a| close(a, c)).ok_or_else(|| {
                FerricError::Libint(format!(
                    "ECP gradient: libecpint center [{:.6}, {:.6}, {:.6}] matches no atom; \
                     derivative rows cannot be attributed",
                    c[0], c[1], c[2]
                ))
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gto_norm_matches_libcint() {
        // Cross-checked against pyscf.gto.gto_norm.
        assert!((gto_norm(0, 1.0) - 2.526_475_110_984_259).abs() < 1e-12);
        assert!((gto_norm(1, 1.0) - 2.917_322_170_855_303).abs() < 1e-12);
        assert!((gto_norm(2, 1.0) - 2.609_332_274_519_885).abs() < 1e-12);
        assert!((gto_norm(3, 2.5) - 15.501_559_129_019_617).abs() < 1e-10);
    }
}
