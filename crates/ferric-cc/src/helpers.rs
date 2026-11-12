use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_scf::ScfResult;
use ferric_tensors::{einsum, permute_to_owned, Axis, Tensor};
use ndarray::{Array4, ArrayD, IxDyn};

/// Orbital counts of a closed-shell RHF reference, `(nocc, nmo)`.
///
/// `nocc` is half of `mol.nelec()`, the doubly occupied count the RHF solver
/// filled. `nelec` counts valence electrons only once
/// [`Molecule::apply_ecp`] has run, which is the electron count the SCF
/// with that basis solved for. The sign of the orbital energies plays no
/// part: the HOMO of an anion in a basis without diffuse functions can lie
/// above zero, and counting `eps < 0` then drops an occupied orbital.
///
/// `nmo` is the number of MO columns the SCF retained. When the overlap
/// has near-null modes, the SCF drops them and pads `mos_r()` back to `nbas`
/// columns with trailing zero columns (orbital energy 1e6); those columns are
/// not orbitals, so the virtual space is `nocc..nmo`.
///
/// # Errors
///
/// An odd or negative electron count (not a closed shell), padding columns
/// that are not trailing, or more occupied orbitals than retained MOs.
pub fn closed_shell_counts(mol: &Molecule, rhf: &ScfResult) -> Result<(usize, usize), FerricError> {
    let nelec = mol.nelec();
    if nelec < 0 || nelec % 2 != 0 {
        return Err(FerricError::General(format!(
            "closed-shell CC needs an even, non-negative electron count; this molecule has {nelec}"
        )));
    }
    let nocc = (nelec / 2) as usize;
    let c = rhf.mos_r();
    let is_zero = |p: usize| c.column(p).iter().all(|&x| x == 0.0);
    let nmo = (0..c.ncols()).take_while(|&p| !is_zero(p)).count();
    if (nmo..c.ncols()).any(|p| !is_zero(p)) {
        return Err(FerricError::General(format!(
            "RHF MO coefficients have a zero column at {nmo} followed by nonzero columns; \
             the dropped linear-dependence columns must trail the retained ones"
        )));
    }
    if nocc > nmo {
        return Err(FerricError::General(format!(
            "{nelec} electrons need {nocc} occupied orbitals but the RHF reference retains \
             only {nmo} MOs"
        )));
    }
    Ok((nocc, nmo))
}

// --- Permutation antisymmetrizers on the i,j,a,b axes of X[i,j,a,b] ---
// Shared by the spin-orbital CCD and CCSD residual builders (ccd.rs, ccsd.rs),
// which apply the same P(ij)/P(ab) projectors to their `[O,O,V,V]` tensors.

// Both swaps go through `permute_to_owned` rather than ndarray's bare
// `as_standard_layout().into_owned()`. The gather is the same; the difference
// is that `permute_to_owned` fans it out over rayon above 64K elements, and
// these run on `[O,O,V,V]` residual tensors — (2·no)²(2·nv)² in the
// spin-orbital drivers — every CC iteration, so they clear that threshold at
// any interesting system size. `einsum.rs` measures this exact strided copy at
// 47% of a comparable contraction at nv=40 and 70% at nv=80: it is
// memory-bandwidth-bound and gets relatively WORSE with size.
//
// Safe by construction: a permutation is pure data movement with no
// accumulation, so every output element is written exactly once with the same
// value regardless of thread count. Bit-identical, unlike a reduction.

/// Swap the i,j axes (0,1) of an `[i,j,a,b]` tensor.
fn swap_ij(x: &ArrayD<f64>) -> ArrayD<f64> {
    permute_to_owned(x.view().permuted_axes(IxDyn(&[1, 0, 2, 3])))
}
/// Swap the a,b axes (2,3) of an `[i,j,a,b]` tensor.
fn swap_ab(x: &ArrayD<f64>) -> ArrayD<f64> {
    permute_to_owned(x.view().permuted_axes(IxDyn(&[0, 1, 3, 2])))
}
/// P(ij) x = x - swap_ij(x)
pub fn p_ij(x: &ArrayD<f64>) -> ArrayD<f64> {
    x - &swap_ij(x)
}
/// P(ab) x = x - swap_ab(x)
pub fn p_ab(x: &ArrayD<f64>) -> ArrayD<f64> {
    x - &swap_ab(x)
}
/// P(ij)P(ab) x = x - swap_ij(x) - swap_ab(x) + swap_ab(swap_ij(x))
pub fn p_ij_ab(x: &ArrayD<f64>) -> ArrayD<f64> {
    let sij = swap_ij(x);
    let sab = swap_ab(x);
    let sijab = swap_ab(&sij);
    x - &sij - &sab + &sijab
}

/// Compute the Particle-Particle ladder term: L_iajb = sum_P B^P_ab * (sum_cd B^P_cd * t_icjd)
/// RI complexity: O(N^5). All contractions routed through `einsum!` (BLAS3 GEMM).
///
/// `b_ab_t` is the dressed RI block `[Aux, V, V]` (loop-invariant — wrapped once
/// by the caller and reused across CCD iterations). `t2_t` is the current
/// amplitude tensor stored `(i,c,j,d)` labeled `[O, V, O, V]`. Intermediate
/// `X[P,i,j] = sum_cd B^P_cd t2[i,c,j,d]`, then `R[i,a,j,b] = sum_P B^P_ab X[P,i,j]`.
/// Output layout `(i,a,j,b)`.
pub fn contract_pp_ladder(b_ab_t: &Tensor<3>, t2_t: &Tensor<4>) -> Array4<f64> {
    let nocc = t2_t.shape()[0];
    let nvir = t2_t.shape()[1];

    // X[P,i,j] = sum_cd B^P_cd t2[i,c,j,d]  (contract c,d=V)
    let x: ndarray::ArrayD<f64> = einsum!("Pcd,icjd->Pij", b_ab_t, t2_t);
    let x_t = Tensor::new(x, [Axis::Aux, Axis::O, Axis::O]);

    // R[a,b,i,j] = sum_P B^P_ab X[P,i,j]; then permute (a,b,i,j)->(i,a,j,b) = axes [2,0,3,1]
    let r_abij: ndarray::ArrayD<f64> = einsum!("Pab,Pij->abij", b_ab_t, &x_t);
    let res = permute_to_owned(r_abij.permuted_axes(IxDyn(&[2, 0, 3, 1])).view());
    res.into_dimensionality::<ndarray::Ix4>()
        .unwrap()
        .into_shape_with_order((nocc, nvir, nocc, nvir))
        .unwrap()
}

/// Compute the Hole-Hole ladder term: H_iajb = sum_P B^P_ij * (sum_kl B^P_kl * t_kalb)
/// RI complexity: O(N^5). All contractions routed through `einsum!` (BLAS3 GEMM).
///
/// `b_ij_t` is the dressed RI block `[Aux, O, O]` (loop-invariant — wrapped once
/// by the caller and reused across CCD iterations). `t2_t` is the current
/// amplitude tensor stored `(k,a,l,b)` labeled `[O, V, O, V]`. Intermediate
/// `Y[P,a,b] = sum_kl B^P_kl t2[k,a,l,b]`, then `R[i,a,j,b] = sum_P B^P_ij Y[P,a,b]`.
/// Output layout `(i,a,j,b)`.
pub fn contract_hh_ladder(b_ij_t: &Tensor<3>, t2_t: &Tensor<4>) -> Array4<f64> {
    let nocc = t2_t.shape()[0];
    let nvir = t2_t.shape()[1];

    // Y[P,a,b] = sum_kl B^P_kl t2[k,a,l,b]  (contract k,l=O)
    let y: ndarray::ArrayD<f64> = einsum!("Pkl,kalb->Pab", b_ij_t, t2_t);
    let y_t = Tensor::new(y, [Axis::Aux, Axis::V, Axis::V]);

    // R[i,j,a,b] = sum_P B^P_ij Y[P,a,b]; then permute (i,j,a,b)->(i,a,j,b) = axes [0,2,1,3]
    let r_ijab: ndarray::ArrayD<f64> = einsum!("Pij,Pab->ijab", b_ij_t, &y_t);
    let res = permute_to_owned(r_ijab.permuted_axes(IxDyn(&[0, 2, 1, 3])).view());
    res.into_dimensionality::<ndarray::Ix4>()
        .unwrap()
        .into_shape_with_order((nocc, nvir, nocc, nvir))
        .unwrap()
}
