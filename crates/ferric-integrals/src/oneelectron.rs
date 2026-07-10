//! One-electron integral matrices: overlap (S), kinetic (T), nuclear (V), and core Hamiltonian (H).

use crate::basis_bridge::PreparedBasis;
use crate::ecp::{ecp_matrix_spherical, gto_norm, EcpCenter, EcpGaussianShell};
use crate::engine::Engine;
use crate::ffi;
use ferric_core::basis::BasisSet;
use ferric_core::mol::Molecule;
use ndarray::Array2;

/// Below this many shell-pair units, run the serial loop directly — avoids
/// rayon/engine-construction overhead for free-atom/tiny-basis jobs (see the
/// free-atom rule: single-atom SCF must not pay a parallel-setup tax).
const PAR_SHELL_PAIR_THRESHOLD: usize = 64;

/// Build a symmetric one-electron matrix by iterating over upper-triangle shell
/// pairs. `make_eng` constructs a fresh, fully-configured [`Engine`] (overlap /
/// kinetic / nuclear-with-point-charges) — called once for the serial path and
/// once per rayon worker via `for_each_init`, never per shell pair (engine
/// construction runs under a global ctor mutex; per-item construction would
/// serialize on that mutex and defeat the parallelism).
///
/// Parallelized over the outer shell index `s1` (independent row bands) once
/// `nsh` clears [`PAR_SHELL_PAIR_THRESHOLD`]. For a fixed `s1`, the write set is
/// `{(offs[s1]+i, offs[s2]+j), (offs[s2]+j, offs[s1]+i) : s2 ≤ s1}` — every
/// written row index is either in `[offs[s1], offs[s1]+n1)` (first form) or
/// equals `offs[s1]+i` for the same range (second form, transposed). So each
/// `s1` owns a disjoint band of *rows* `[offs[s1], offs[s1]+dims[s1])` in
/// `out`; distinct `s1` values touch disjoint row bands, so the raw-pointer
/// scatter below is data-race-free and, since every element is written
/// exactly once (no accumulation), bit-identical to the serial loop
/// regardless of thread count or scheduling order.
fn build_symmetric(prep: &PreparedBasis, make_eng: impl Fn() -> Engine + Sync) -> Array2<f64> {
    let n = prep.nbasis();
    let nsh = prep.nshells();
    let dims = prep.shell_dims();
    let offs = prep.shell_offsets();
    let mut out = Array2::zeros((n, n));

    if nsh < PAR_SHELL_PAIR_THRESHOLD {
        let mut eng = make_eng();
        for s1 in 0..nsh {
            for s2 in 0..=s1 {
                let block = eng.compute_1e_block(prep, s1, s2);
                let n1 = dims[s1];
                let n2 = dims[s2];
                for i in 0..n1 {
                    for j in 0..n2 {
                        let v = block[i * n2 + j];
                        out[(offs[s1] + i, offs[s2] + j)] = v;
                        out[(offs[s2] + j, offs[s1] + i)] = v;
                    }
                }
            }
        }
        return out;
    }

    use rayon::prelude::*;
    let out_ptr = out.as_mut_ptr() as usize;
    let stride = n; // row-major (n, n): element (r, c) at r*stride + c

    (0..nsh).into_par_iter().for_each_init(
        &make_eng,
        |worker_eng, s1| {
            let n1 = dims[s1];
            let o1 = offs[s1];
            for s2 in 0..=s1 {
                let block = worker_eng.compute_1e_block(prep, s1, s2);
                let n2 = dims[s2];
                let o2 = offs[s2];
                for i in 0..n1 {
                    for j in 0..n2 {
                        let v = block[i * n2 + j];
                        let r = o1 + i;
                        let c = o2 + j;
                        unsafe {
                            let base = out_ptr as *mut f64;
                            *base.add(r * stride + c) = v;
                            *base.add(c * stride + r) = v;
                        }
                    }
                }
            }
        },
    );
    out
}

/// Compute the overlap matrix S, shape (nbasis, nbasis).
pub fn overlap(prep: &PreparedBasis) -> Array2<f64> {
    build_symmetric(prep, || Engine::new_1e(ffi::OP_OVERLAP, prep, 1e-14).unwrap())
}

/// Compute the kinetic energy matrix T, shape (nbasis, nbasis).
pub fn kinetic(prep: &PreparedBasis) -> Array2<f64> {
    build_symmetric(prep, || Engine::new_1e(ffi::OP_KINETIC, prep, 1e-14).unwrap())
}

/// Compute the nuclear attraction matrix V, shape (nbasis, nbasis).
pub fn nuclear(prep: &PreparedBasis) -> Array2<f64> {
    build_symmetric(prep, || {
        let mut eng = Engine::new_1e(ffi::OP_NUCLEAR, prep, 1e-14).unwrap();
        eng.set_point_charges(prep);
        eng
    })
}

/// Compute the core Hamiltonian H = T + V, shape (nbasis, nbasis).
///
/// `V` is the nuclear-attraction matrix; for ECP-treated atoms its point charges
/// are already the effective (valence-only) `Z − n_core` set up in
/// [`PreparedBasis::new`]. This function does **not** add the ECP projector
/// `V_ECP` — use [`hcore_ecp`] when the basis carries ECPs.
pub fn hcore(prep: &PreparedBasis) -> Array2<f64> {
    let t = kinetic(prep);
    let v = nuclear(prep);
    t + v
}

/// Compute the dense spherical ECP projector matrix `V_ECP`, shape
/// (nbasis, nbasis), for the given molecule + ECP-carrying basis set.
///
/// Returns `None` (and does zero work) when the basis carries no ECPs. The shell
/// iteration order mirrors [`PreparedBasis::new`] exactly, so the resulting AO
/// ordering matches libint's spherical basis — the matrix can be added to
/// `hcore` directly.
///
/// Per the libecpint convention (see [`crate::ecp`]), each primitive's stored
/// contraction coefficient is multiplied by `gto_norm(l, α)` before being handed
/// to the shim.
pub fn ecp_potential(mol: &Molecule, bs: &BasisSet) -> Option<Array2<f64>> {
    if bs.ecps.is_empty() {
        return None;
    }
    let mut shells: Vec<EcpGaussianShell> = Vec::new();
    let mut ecps: Vec<EcpCenter> = Vec::new();
    for atom in &mol.atoms {
        let center = [atom.x, atom.y, atom.zpos];
        // Gaussian shells (real Z → real basis), gto_norm folded per primitive.
        if let Some(tmpls) = bs.for_element(atom.z) {
            for sh in tmpls {
                let coefs: Vec<f64> = sh
                    .exponents
                    .iter()
                    .zip(sh.coefficients.iter())
                    .map(|(&a, &c)| c * gto_norm(sh.l, a))
                    .collect();
                shells.push(EcpGaussianShell {
                    l: sh.l,
                    center,
                    exponents: sh.exponents.clone(),
                    coefficients: coefs,
                });
            }
        }
        // ECP center for this atom, if any. Flatten EcpDef channels into the
        // (am, n, exp, coef) primitive lists libecpint expects.
        if let Some(def) = bs.ecp_for_element(atom.z) {
            let mut ams = Vec::new();
            let mut ns = Vec::new();
            let mut exponents = Vec::new();
            let mut coefficients = Vec::new();
            for ch in &def.shells {
                for t in &ch.terms {
                    ams.push(ch.angular_momentum);
                    ns.push(t.r_exp);
                    exponents.push(t.gexp);
                    coefficients.push(t.coef);
                }
            }
            ecps.push(EcpCenter { center, ams, ns, exponents, coefficients });
        }
    }
    if ecps.is_empty() {
        return None;
    }
    let n: usize = shells.iter().map(|s| (2 * s.l + 1) as usize).sum();
    let flat = ecp_matrix_spherical(&shells, &ecps)
        .expect("ecp_matrix_spherical failed building V_ECP");
    Some(Array2::from_shape_vec((n, n), flat).expect("V_ECP shape"))
}

/// Core Hamiltonian including the ECP projector:
/// `H = T + V_nuc(effective Z) + V_ECP`.
///
/// Identical to [`hcore`] when `bs` carries no ECPs (V_ECP is skipped at zero
/// cost). `prep` must have been built from the same `mol` + `bs`.
pub fn hcore_ecp(prep: &PreparedBasis, mol: &Molecule, bs: &BasisSet) -> Array2<f64> {
    let mut h = hcore(prep);
    if let Some(vecp) = ecp_potential(mol, bs) {
        assert_eq!(
            vecp.dim(),
            h.dim(),
            "V_ECP dimension {:?} != hcore dimension {:?}",
            vecp.dim(),
            h.dim()
        );
        h += &vecp;
    }
    h
}

/// Compute the 3 electric dipole matrices ⟨μ|(r - origin)|ν⟩, shape (nbasis, nbasis) each.
/// `origin` is in Bohr. Returns [x_mat, y_mat, z_mat].
pub fn dipole(prep: &PreparedBasis, origin: [f64; 3]) -> [Array2<f64>; 3] {
    let nbas = prep.nbasis();
    let mut flat = vec![0.0f64; 3 * nbas * nbas];
    let ret = unsafe {
        ffi::scf_compute_dipole(
            prep.handle(),
            origin.as_ptr(),
            nbas as std::os::raw::c_int,
            flat.as_mut_ptr(),
        )
    };
    assert!(ret >= 0, "scf_compute_dipole failed: {}", ret);
    let make_mat = |offset: usize| {
        let slice = &flat[offset..offset + nbas * nbas];
        Array2::from_shape_vec((nbas, nbas), slice.to_vec()).unwrap()
    };
    [make_mat(0), make_mat(nbas * nbas), make_mat(2 * nbas * nbas)]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis_bridge::PreparedBasis;
    use ferric_core::basis;
    use ferric_core::mol::Molecule;

    fn water_sto3g() -> PreparedBasis {
        let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
        let bs = basis::bundled("sto-3g").unwrap();
        PreparedBasis::new(&mol, &bs).unwrap()
    }

    #[test]
    fn test_overlap_diagonal_ones() {
        let prep = water_sto3g();
        let s = overlap(&prep);
        for i in 0..prep.nbasis() {
            assert!((s[(i, i)] - 1.0).abs() < 1e-8, "S[{i},{i}] = {}", s[(i, i)]);
        }
    }

    #[test]
    fn test_overlap_symmetric() {
        let prep = water_sto3g();
        let s = overlap(&prep);
        let n = prep.nbasis();
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (s[(i, j)] - s[(j, i)]).abs() < 1e-12,
                    "S not symmetric at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn test_dipole_symmetric() {
        // ⟨μ|r_d|ν⟩ is symmetric in (μ,ν) — r is a multiplicative operator.
        let prep = water_sto3g();
        let dip = dipole(&prep, [0.0, 0.0, 0.0]);
        let n = prep.nbasis();
        for (d, mat) in dip.iter().enumerate() {
            for i in 0..n {
                for j in 0..n {
                    assert!(
                        (mat[(i, j)] - mat[(j, i)]).abs() < 1e-12,
                        "dipole axis {d} not symmetric at ({i},{j})"
                    );
                }
            }
        }
    }

    #[test]
    fn test_dipole_origin_shift_diagonal() {
        // Shifting the origin by δ subtracts δ·S from ⟨μ|r|ν⟩ (since
        // ⟨μ|(r−δ)|ν⟩ = ⟨μ|r|ν⟩ − δ⟨μ|ν⟩). Validates the origin argument wiring.
        let prep = water_sto3g();
        let s = overlap(&prep);
        let d0 = dipole(&prep, [0.0, 0.0, 0.0]);
        let delta = [0.3, -0.7, 1.1];
        let dshift = dipole(&prep, delta);
        let n = prep.nbasis();
        for (ax, dl) in delta.iter().enumerate() {
            for i in 0..n {
                for j in 0..n {
                    let expected = d0[ax][(i, j)] - dl * s[(i, j)];
                    assert!(
                        (dshift[ax][(i, j)] - expected).abs() < 1e-10,
                        "axis {ax} origin-shift mismatch at ({i},{j})"
                    );
                }
            }
        }
    }

    /// Serial reference for `build_symmetric` (pre-parallelization implementation,
    /// kept verbatim). The parallel version must reproduce it bit-for-bit.
    fn build_symmetric_serial(prep: &PreparedBasis, mut eng: Engine) -> Array2<f64> {
        let n = prep.nbasis();
        let nsh = prep.nshells();
        let dims = prep.shell_dims();
        let offs = prep.shell_offsets();
        let mut out = Array2::zeros((n, n));
        for s1 in 0..nsh {
            for s2 in 0..=s1 {
                let block = eng.compute_1e_block(prep, s1, s2);
                let n1 = dims[s1];
                let n2 = dims[s2];
                for i in 0..n1 {
                    for j in 0..n2 {
                        let v = block[i * n2 + j];
                        out[(offs[s1] + i, offs[s2] + j)] = v;
                        out[(offs[s2] + j, offs[s1] + i)] = v;
                    }
                }
            }
        }
        out
    }

    fn assert_bit_identical(a: &Array2<f64>, b: &Array2<f64>, what: &str) {
        assert_eq!(a.dim(), b.dim(), "{what}: shape mismatch");
        let n_diff = a.iter().zip(b.iter()).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
        assert_eq!(n_diff, 0, "{what}: {n_diff} elements differ bitwise");
    }

    /// alkane_6/cc-pVDZ clears `PAR_SHELL_PAIR_THRESHOLD` (64 shells), so this
    /// test actually exercises the rayon path, not just the serial fallback.
    fn alkane6_cc_pvdz() -> PreparedBasis {
        let mol = Molecule::load_xyz("../../testdata/molecules/alkane_6.xyz").unwrap();
        let bs = basis::bundled("cc-pvdz").unwrap();
        PreparedBasis::new(&mol, &bs).unwrap()
    }

    #[test]
    fn test_build_symmetric_bitidentical_to_serial_overlap() {
        let prep = alkane6_cc_pvdz();
        assert!(prep.nshells() >= PAR_SHELL_PAIR_THRESHOLD,
            "test molecule too small to exercise the parallel path: {} shells", prep.nshells());
        let par = overlap(&prep);
        let eng_ref = Engine::new_1e(ffi::OP_OVERLAP, &prep, 1e-14).unwrap();
        let ser = build_symmetric_serial(&prep, eng_ref);
        assert_bit_identical(&ser, &par, "overlap");
    }

    #[test]
    fn test_build_symmetric_bitidentical_to_serial_kinetic() {
        let prep = alkane6_cc_pvdz();
        let par = kinetic(&prep);
        let eng_ref = Engine::new_1e(ffi::OP_KINETIC, &prep, 1e-14).unwrap();
        let ser = build_symmetric_serial(&prep, eng_ref);
        assert_bit_identical(&ser, &par, "kinetic");
    }

    #[test]
    fn test_build_symmetric_bitidentical_to_serial_nuclear() {
        let prep = alkane6_cc_pvdz();
        let par = nuclear(&prep);
        let mut eng_ref = Engine::new_1e(ffi::OP_NUCLEAR, &prep, 1e-14).unwrap();
        eng_ref.set_point_charges(&prep);
        let ser = build_symmetric_serial(&prep, eng_ref);
        assert_bit_identical(&ser, &par, "nuclear");
    }
}
