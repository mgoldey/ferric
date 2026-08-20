//! One-electron integral matrices: overlap (S), kinetic (T), nuclear (V), and core Hamiltonian (H).

use crate::basis_bridge::PreparedBasis;
use crate::ecp::{
    ecp_deriv_atom_ids, ecp_matrix_deriv_spherical, ecp_matrix_spherical, gto_norm, EcpCenter,
    EcpGaussianShell,
};
use crate::engine::Engine;
use crate::ffi;
use ferric_core::basis::BasisSet;
use ferric_core::external_potential::ExternalPotential;
use ferric_core::mol::Molecule;
use ferric_core::FerricError;
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
                        // SAFETY: rayon work items write to disjoint
                        // (s1,s2) shell-pair blocks in `out`; the symmetric
                        // write (r,c)/(c,r) is within the same block's
                        // triangle. out_ptr is an AtomicPtr shared across
                        // workers; no two workers touch the same (r,c) pair.
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
        eng.set_point_charges(prep).unwrap();
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

/// Nuclear attraction matrix including external point charges (appended
/// after the real atoms). Falls back to plain `nuclear(prep)` semantics
/// when `ext.point_charges` is empty.
pub fn nuclear_with_external(prep: &PreparedBasis, ext: &ExternalPotential) -> Array2<f64> {
    build_symmetric(prep, || {
        let mut eng = Engine::new_1e(ffi::OP_NUCLEAR, prep, 1e-14).unwrap();
        eng.set_point_charges_extra(prep, &ext.point_charges).unwrap();
        eng
    })
}

/// One-electron term for a uniform external field: H' = +E·r per electron
/// (i.e. V = -E·mu), built from the dipole integrals about the origin.
/// Returns the zero matrix when `field == [0,0,0]`.
pub fn field_hcore_term(prep: &PreparedBasis, field: [f64; 3]) -> Result<Array2<f64>, FerricError> {
    let n = prep.nbasis();
    if field == [0.0, 0.0, 0.0] {
        return Ok(Array2::zeros((n, n)));
    }
    let dip = dipole(prep, [0.0, 0.0, 0.0])?;
    Ok(field[0] * &dip[0] + field[1] * &dip[1] + field[2] * &dip[2])
}

/// Core Hamiltonian H = T + V, optionally including an external potential's
/// point-charge nuclear-attraction term and uniform-field term. `ext = None`
/// is byte-for-byte identical to `hcore(prep)`.
pub fn hcore_with_external(prep: &PreparedBasis, ext: Option<&ExternalPotential>) -> Result<Array2<f64>, FerricError> {
    let t = kinetic(prep);
    let Some(ext) = ext else { return Ok(t + nuclear(prep)) };
    if ext.is_empty() {
        return Ok(t + nuclear(prep));
    }
    let v = if ext.point_charges.is_empty() {
        nuclear(prep)
    } else {
        nuclear_with_external(prep, ext)
    };
    let mut h = t + v;
    if let Some(field) = ext.field {
        h += &field_hcore_term(prep, field)?;
    }
    Ok(h)
}

/// `hcore_ecp` extended with an external potential — see [`hcore_with_external`]
/// and [`hcore_ecp`]. `ext = None` is byte-for-byte identical to `hcore_ecp`.
pub fn hcore_ecp_with_external(
    prep: &PreparedBasis,
    mol: &Molecule,
    bs: &BasisSet,
    ext: Option<&ExternalPotential>,
) -> Result<Array2<f64>, FerricError> {
    let mut h = hcore_with_external(prep, ext)?;
    if let Some(vecp) = ecp_potential(mol, bs) {
        assert_eq!(vecp.dim(), h.dim(), "V_ECP dimension {:?} != hcore dimension {:?}", vecp.dim(), h.dim());
        h += &vecp;
    }
    Ok(h)
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
    let (shells, ecps) = build_ecp_inputs(mol, bs)?;
    let n: usize = shells.iter().map(|s| (2 * s.l + 1) as usize).sum();
    let flat = ecp_matrix_spherical(&shells, &ecps)
        .expect("ecp_matrix_spherical failed building V_ECP");
    Some(Array2::from_shape_vec((n, n), flat).expect("V_ECP shape"))
}

/// Build the (Gaussian shells, ECP centers) input pair that the libecpint
/// wrapper consumes, in exactly the order [`PreparedBasis::new`] emits shells.
///
/// Returns `None` when the basis carries no ECPs or the molecule has no
/// ECP-carrying atom. Shared by [`ecp_potential`] and [`ecp_potential_deriv`] so
/// the energy and gradient can never disagree about the basis they describe.
fn build_ecp_inputs(
    mol: &Molecule,
    bs: &BasisSet,
) -> Option<(Vec<EcpGaussianShell>, Vec<EcpCenter>)> {
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
    Some((shells, ecps))
}

/// First derivatives of the spherical ECP matrix with respect to each atomic
/// coordinate: `dV_ECP/dR_{A,c}`.
///
/// Returns `Ok(None)` (zero work) when the basis carries no ECPs, mirroring
/// [`ecp_potential`]. Otherwise returns a `(natoms, 3)`-shaped nesting: element
/// `[a][c]` is the `(nbasis, nbasis)` derivative w.r.t. atom `a`'s coordinate
/// `c`, indexed by **the molecule's** atom order. Atoms libecpint never sees
/// (no basis shells and no ECP) correctly get zero matrices.
///
/// AO ordering matches [`ecp_potential`] and hence `hcore`, so the result can be
/// contracted directly with a density matrix in the AO basis.
#[allow(clippy::type_complexity)]
pub fn ecp_potential_deriv(
    mol: &Molecule,
    bs: &BasisSet,
) -> Result<Option<Vec<[Array2<f64>; 3]>>, FerricError> {
    let Some((shells, ecps)) = build_ecp_inputs(mol, bs) else {
        return Ok(None);
    };
    let n: usize = shells.iter().map(|s| (2 * s.l + 1) as usize).sum();

    let (flat_derivs, natoms_ecpint) = ecp_matrix_deriv_spherical(&shells, &ecps)?;

    // libecpint indexes derivatives by ITS OWN inferred atom ids (deduplicated
    // shell/ECP centers), which need not match mol.atoms 1:1. Map them back
    // explicitly rather than assuming the orders coincide.
    let centers: Vec<[f64; 3]> = mol.atoms.iter().map(|a| [a.x, a.y, a.zpos]).collect();
    let ids = ecp_deriv_atom_ids(&shells, &ecps, &centers)?;
    if ids.len() != natoms_ecpint {
        return Err(FerricError::Libint(format!(
            "ECP gradient: mapped {} atom ids but libecpint reported {natoms_ecpint}",
            ids.len()
        )));
    }

    let mut out: Vec<[Array2<f64>; 3]> = (0..mol.atoms.len())
        .map(|_| {
            [
                Array2::zeros((n, n)),
                Array2::zeros((n, n)),
                Array2::zeros((n, n)),
            ]
        })
        .collect();

    for (ecpint_atom, &mol_atom) in ids.iter().enumerate() {
        for c in 0..3 {
            let flat = &flat_derivs[3 * ecpint_atom + c];
            if flat.len() != n * n {
                return Err(FerricError::Libint(format!(
                    "ECP gradient: derivative block {} has {} elements, expected {}",
                    3 * ecpint_atom + c,
                    flat.len(),
                    n * n
                )));
            }
            out[mol_atom][c] = Array2::from_shape_vec((n, n), flat.clone())
                .map_err(|e| FerricError::Libint(format!("dV_ECP shape: {e}")))?;
        }
    }

    Ok(Some(out))
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
/// `origin` is in Bohr. Returns `[x_mat, y_mat, z_mat]`.
///
/// Returns `Err(FerricError::Libint(..))` instead of panicking if the
/// underlying `scf_compute_dipole` shim call reports a libint2-internal
/// failure (negative status) — see the FFI exception-safety convention:
/// every `scf_*` shim call catches C++ exceptions and returns a status code
/// that must be checked, never silently trusted.
pub fn dipole(prep: &PreparedBasis, origin: [f64; 3]) -> Result<[Array2<f64>; 3], FerricError> {
    let nbas = prep.nbasis();
    let mut flat = vec![0.0f64; 3 * nbas * nbas];
    // SAFETY: prep.handle() is a valid basis handle; origin and flat are
    // stack/heap arrays with correct sizes. Status checked below.
    let ret = unsafe {
        ffi::scf_compute_dipole(
            prep.handle(),
            origin.as_ptr(),
            nbas as std::os::raw::c_int,
            flat.as_mut_ptr(),
        )
    };
    if ret < 0 {
        return Err(FerricError::Libint(format!("scf_compute_dipole failed: {ret}")));
    }
    let make_mat = |offset: usize| {
        let slice = &flat[offset..offset + nbas * nbas];
        Array2::from_shape_vec((nbas, nbas), slice.to_vec()).unwrap()
    };
    Ok([make_mat(0), make_mat(nbas * nbas), make_mat(2 * nbas * nbas)])
}

/// Cartesian second-moment integrals ⟨μ|(r−O)_p (r−O)_q|ν⟩ about `origin`,
/// returned in the order [xx, xy, xz, yy, yz, zz] (libint `emultipole2`).
///
/// Verified by the exact translational identity against [`dipole`] and
/// [`overlap`] (see the tests): shifting the origin by d maps
/// ⟨pq⟩_O' = ⟨pq⟩_O − d_p⟨q⟩_O − d_q⟨p⟩_O + d_p d_q S, which pins every
/// component against two independent engines.
pub fn second_moment(
    prep: &PreparedBasis,
    origin: [f64; 3],
) -> Result<[Array2<f64>; 6], FerricError> {
    let nbas = prep.nbasis();
    let mut flat = vec![0.0f64; 6 * nbas * nbas];
    // SAFETY: same contract as dipole — valid handle, correctly-sized buffers.
    let ret = unsafe {
        ffi::scf_compute_second_moment(
            prep.handle(),
            origin.as_ptr(),
            nbas as std::os::raw::c_int,
            flat.as_mut_ptr(),
        )
    };
    if ret < 0 {
        return Err(FerricError::Libint(format!(
            "scf_compute_second_moment failed: {ret}"
        )));
    }
    let make_mat = |k: usize| {
        let slice = &flat[k * nbas * nbas..(k + 1) * nbas * nbas];
        Array2::from_shape_vec((nbas, nbas), slice.to_vec()).unwrap()
    };
    Ok([make_mat(0), make_mat(1), make_mat(2), make_mat(3), make_mat(4), make_mat(5)])
}

/// ⟨μ|(r−O)²|ν⟩ = xx + yy + zz about `origin` — the operator orbital
/// spreads are built from (σ² = ⟨r²⟩ − ⟨r⟩²).
pub fn r2_moment(prep: &PreparedBasis, origin: [f64; 3]) -> Result<Array2<f64>, FerricError> {
    let m = second_moment(prep, origin)?;
    Ok(&(&m[0] + &m[3]) + &m[5])
}

/// Per-orbital centroids ⟨p|r|p⟩ (rows of the returned (n, 3) array) and
/// spatial spreads σ_p = sqrt(⟨p|r²|p⟩ − |⟨p|r|p⟩|²) for the columns of `c`.
///
/// The spread is origin-independent by construction; centroids are reported
/// in the lab frame (origin at 0). Mirrors Q-Chem's "second moments of
/// orbitals" property; useful as an ML descriptor and as the pair-gate
/// input the amplitude-threshold family uses.
pub fn orbital_moments(
    prep: &PreparedBasis,
    c: &Array2<f64>,
) -> Result<(Array2<f64>, Vec<f64>), FerricError> {
    let dip = dipole(prep, [0.0; 3])?;
    let r2 = r2_moment(prep, [0.0; 3])?;
    let n = c.ncols();
    let mut centers = Array2::<f64>::zeros((n, 3));
    let mut spreads = Vec::with_capacity(n);
    for p in 0..n {
        let col = c.column(p);
        let mut c2 = 0.0;
        for (x, dm) in dip.iter().enumerate() {
            let v = col.dot(&dm.dot(&col));
            centers[(p, x)] = v;
            c2 += v * v;
        }
        let r2v = col.dot(&r2.dot(&col));
        spreads.push((r2v - c2).max(0.0).sqrt());
    }
    Ok((centers, spreads))
}

/// Electronic-density second-moment tensor about `origin`:
/// `M_xy = Σ_μν D_μν ⟨μ|(r−O)_x (r−O)_y|ν⟩` (symmetric 3×3; trace =
/// ⟨r²⟩ of the density — the spatial-extent measure).
///
/// `d` is the total (spin-summed) AO density matrix. Verified by the exact
/// density-level translational identity against [`dipole`]/[`overlap`]
/// (see the tests): M(O+d) = M(O) − d⊗μ − μ⊗d + (d⊗d)·N with μ the
/// electronic dipole-integral contraction and N = Tr(D S).
pub fn density_second_moment(
    prep: &PreparedBasis,
    d: &Array2<f64>,
    origin: [f64; 3],
) -> Result<[[f64; 3]; 3], FerricError> {
    let m = second_moment(prep, origin)?;
    // component order [xx, xy, xz, yy, yz, zz] -> (p,q)
    let pq: [(usize, usize); 6] = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)];
    let mut out = [[0.0f64; 3]; 3];
    for (k, &(p, q)) in pq.iter().enumerate() {
        let v = (d * &m[k]).sum();
        out[p][q] = v;
        out[q][p] = v;
    }
    Ok(out)
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

    /// EXACT translational identity pinning all six second-moment
    /// components against the independent dipole + overlap engines:
    ///   ⟨pq⟩_{O+d} = ⟨pq⟩_O − d_p⟨q⟩_O − d_q⟨p⟩_O + d_p d_q S.
    /// Mutation arm: with one term of the identity deliberately dropped the
    /// residual must be LARGE (proves the check is not trivially satisfied).
    #[test]
    fn second_moment_translational_identity() {
        let prep = water_sto3g();
        let n = prep.nbasis();
        let o1 = [0.0, 0.0, 0.0];
        let d = [0.37, -0.81, 0.55]; // deliberately incommensurate shift
        let o2 = [o1[0] + d[0], o1[1] + d[1], o1[2] + d[2]];
        let s = overlap(&prep);
        let dip1 = dipole(&prep, o1).unwrap();
        let m1 = second_moment(&prep, o1).unwrap();
        let m2 = second_moment(&prep, o2).unwrap();
        // component order [xx, xy, xz, yy, yz, zz] -> (p,q) axis pairs
        let pq: [(usize, usize); 6] = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)];
        let mut max_dev: f64 = 0.0;
        let mut max_dropped: f64 = 0.0;
        for (k, &(p, q)) in pq.iter().enumerate() {
            for i in 0..n {
                for j in 0..n {
                    let expect = m1[k][(i, j)] - d[p] * dip1[q][(i, j)] - d[q] * dip1[p][(i, j)]
                        + d[p] * d[q] * s[(i, j)];
                    max_dev = max_dev.max((m2[k][(i, j)] - expect).abs());
                    // mutation: drop the d_p⟨q⟩ term
                    let broken = m1[k][(i, j)] - d[q] * dip1[p][(i, j)] + d[p] * d[q] * s[(i, j)];
                    max_dropped = max_dropped.max((m2[k][(i, j)] - broken).abs());
                }
            }
        }
        assert!(max_dev < 1e-10, "translational identity dev {max_dev:.2e}");
        assert!(
            max_dropped > 1e-3,
            "identity check vacuous: dropped-term residual only {max_dropped:.2e}"
        );
    }

    /// Density-level translational identity for the second-moment tensor:
    /// with N = Tr(D S) and electronic dipole μ_x = Σ D ∘ ⟨x⟩ about O,
    ///   M(O+d)_pq = M(O)_pq − d_p μ_q − d_q μ_p + d_p d_q N — exact.
    /// Uses a deliberately asymmetric D (outer product of two AO vectors,
    /// symmetrized) so the check is not vacuous for off-diagonal components.
    #[test]
    fn density_second_moment_translational_identity() {
        let prep = water_sto3g();
        let nb = prep.nbasis();
        // synthetic symmetric "density" with full off-diagonal structure
        let mut d = Array2::<f64>::zeros((nb, nb));
        for i in 0..nb {
            for j in 0..nb {
                d[(i, j)] = 0.3 + 0.1 * ((i * 7 + j * 3) % 5) as f64;
            }
        }
        let d = &d + &d.t().to_owned();
        let s = overlap(&prep);
        let n_e = (&d * &s).sum();
        let o1 = [0.0; 3];
        let sh = [0.29, -0.63, 0.41];
        let o2 = [sh[0], sh[1], sh[2]];
        let dip = dipole(&prep, o1).unwrap();
        let mu: Vec<f64> = (0..3).map(|x| (&d * &dip[x]).sum()).collect();
        let m1 = density_second_moment(&prep, &d, o1).unwrap();
        let m2 = density_second_moment(&prep, &d, o2).unwrap();
        let mut max_dev: f64 = 0.0;
        for p in 0..3 {
            for q in 0..3 {
                let expect = m1[p][q] - sh[p] * mu[q] - sh[q] * mu[p] + sh[p] * sh[q] * n_e;
                max_dev = max_dev.max((m2[p][q] - expect).abs());
            }
        }
        assert!(max_dev < 1e-10, "density translational identity dev {max_dev:.2e}");
    }

    /// Orbital moments: spreads strictly positive; centroid of a symmetric
    /// (normalized-eigenvector) orbital set finite; and σ² must equal the
    /// direct ⟨r²⟩ − |⟨r⟩|² evaluation elementwise.
    #[test]
    fn orbital_moments_match_direct_evaluation() {
        let prep = water_sto3g();
        let nb = prep.nbasis();
        // normalized AO unit vectors (orbital_moments takes any columns)
        let s = overlap(&prep);
        let mut c = Array2::<f64>::zeros((nb, nb));
        for k in 0..nb {
            c[(k, k)] = 1.0 / s[(k, k)].sqrt();
        }
        let (centers, spreads) = orbital_moments(&prep, &c).unwrap();
        let dipm = dipole(&prep, [0.0; 3]).unwrap();
        let r2 = r2_moment(&prep, [0.0; 3]).unwrap();
        for p in 0..nb {
            assert!(spreads[p] > 0.0, "spread[{p}] not positive");
            let col = c.column(p);
            let mut c2 = 0.0;
            for (xax, dm) in dipm.iter().enumerate() {
                let v = col.dot(&dm.dot(&col));
                assert!((centers[(p, xax)] - v).abs() < 1e-12);
                c2 += v * v;
            }
            let sig = (col.dot(&r2.dot(&col)) - c2).max(0.0).sqrt();
            assert!((spreads[p] - sig).abs() < 1e-12);
        }
    }

    /// r² diagonal must be strictly positive and symmetric, and the trace
    /// wrapper must equal xx+yy+zz elementwise.
    #[test]
    fn r2_moment_is_positive_and_consistent() {
        let prep = water_sto3g();
        let n = prep.nbasis();
        let m = second_moment(&prep, [0.0; 3]).unwrap();
        let r2 = r2_moment(&prep, [0.0; 3]).unwrap();
        for i in 0..n {
            assert!(r2[(i, i)] > 0.0, "⟨{i}|r²|{i}⟩ = {} not positive", r2[(i, i)]);
            for j in 0..n {
                let expect = m[0][(i, j)] + m[3][(i, j)] + m[5][(i, j)];
                assert!((r2[(i, j)] - expect).abs() < 1e-14);
                assert!((r2[(i, j)] - r2[(j, i)]).abs() < 1e-12);
            }
        }
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
        let dip = dipole(&prep, [0.0, 0.0, 0.0]).unwrap();
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
        let d0 = dipole(&prep, [0.0, 0.0, 0.0]).unwrap();
        let delta = [0.3, -0.7, 1.1];
        let dshift = dipole(&prep, delta).unwrap();
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
        eng_ref.set_point_charges(&prep).unwrap();
        let ser = build_symmetric_serial(&prep, eng_ref);
        assert_bit_identical(&ser, &par, "nuclear");
    }

    use ferric_core::external_potential::{ExternalPotential, PointCharge};

    #[test]
    fn hcore_with_external_none_matches_hcore() {
        let prep = water_sto3g();
        let h_orig = hcore(&prep);
        let h_new = hcore_with_external(&prep, None).unwrap();
        assert_eq!(h_orig, h_new);
    }

    #[test]
    fn hcore_with_external_empty_matches_hcore() {
        let prep = water_sto3g();
        let h_orig = hcore(&prep);
        let ext = ExternalPotential::default();
        let h_new = hcore_with_external(&prep, Some(&ext)).unwrap();
        assert_eq!(h_orig, h_new);
    }

    #[test]
    fn nuclear_with_external_adds_point_charge_attraction() {
        let prep = water_sto3g();
        let v_orig = nuclear(&prep);
        let ext = ExternalPotential {
            point_charges: vec![PointCharge { q: 1.0, x: 0.0, y: 0.0, z: 10.0 }],
            field: None,
        };
        let v_new = nuclear_with_external(&prep, &ext);
        // The external charge must change every element that has nonzero AO
        // density overlap; at minimum, the matrix must differ from v_orig.
        let diff: f64 = (&v_new - &v_orig).iter().map(|x| x.abs()).sum();
        assert!(diff > 1e-10, "external point charge had no effect on V");
        // Symmetric.
        let n = prep.nbasis();
        for i in 0..n {
            for j in 0..n {
                assert!((v_new[(i, j)] - v_new[(j, i)]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn field_hcore_term_zero_field_is_zero_matrix() {
        let prep = water_sto3g();
        let h = field_hcore_term(&prep, [0.0, 0.0, 0.0]).unwrap();
        assert!(h.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn field_hcore_term_matches_dipole_integral_scaled() {
        let prep = water_sto3g();
        let dip = dipole(&prep, [0.0, 0.0, 0.0]).unwrap();
        let field = [0.0, 0.0, 0.02];
        let h = field_hcore_term(&prep, field).unwrap();
        // H' = +E·r per electron => h = field[2] * dip[2] (using dip = <mu|r|nu>)
        let expected = &dip[2] * field[2];
        let n = prep.nbasis();
        for i in 0..n {
            for j in 0..n {
                assert!((h[(i, j)] - expected[(i, j)]).abs() < 1e-12);
            }
        }
    }
}
