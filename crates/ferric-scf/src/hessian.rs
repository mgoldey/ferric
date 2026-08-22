//! Analytic RHF second derivatives (Hessian).
//!
//! Computes the full (3N × 3N) Cartesian Hessian as five terms:
//!
//! 1. Nuclear repulsion:  `d²V_nn / dR_A dR_B`
//! 2. Skeleton 1e:        `Tr[D · h^{(2)}]`  (kinetic + nuclear-attraction second derivs)
//! 3. Overlap × W:        `−Tr[W · S^{(2)}]`
//! 4. Skeleton 2e:        J and K second derivatives contracted with D
//! 5. CPKS response:      orbital relaxation via coupled-perturbed HF
//!
//! **Status**: Only term (1) is fully implemented. Terms (2)–(5) require
//! second-derivative integrals from libint2 (`deriv_order=2`), which need
//! `LIBINT2_MAX_DERIV_ORDER >= 2`. The current libint2 installation has
//! `LIBINT2_MAX_DERIV_ORDER = 1` (first derivatives only). The structure and
//! flow for all five terms is laid out here with TODO markers at the integral
//! engine boundaries.
//!
//! The existing finite-difference Hessian in [`super::frequencies`] remains the
//! production path until these stubs are filled in.

use crate::result::ScfResult;
use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ndarray::{s, Array2};

/// Compute the full analytic RHF Hessian.
///
/// Returns a (3N, 3N) Cartesian Hessian in Hartree/Bohr².
pub fn rhf_hessian(
    _mol: &Molecule,
    _prep: &PreparedBasis,
    _op: Operator,
    rhf: &ScfResult,
) -> Result<Array2<f64>, FerricError> {
    if !rhf.converged {
        return Err(FerricError::ScfConvergence {
            iterations: rhf.iterations,
            last_energy: rhf.energy,
        });
    }

    // Terms 2-5 (1e/overlap/2e skeleton + CPKS response) are unimplemented and
    // each return a ZERO block. Assembling them would hand back a
    // nuclear-repulsion-only matrix that LOOKS like a Hessian -- right shape,
    // plausible magnitudes, silently missing every electronic contribution.
    // Refusing is the honest behaviour; see `hess_nuclear_repulsion` below,
    // which is complete and callable on its own if you genuinely want term 1.
    //
    // Lift this only when terms 2-5 are implemented AND validated against
    // finite differences of `rhf_gradient` (an independent construction).
    Err(FerricError::General(
        "analytic RHF Hessian is not implemented: terms 2-5 need libint2 \
         deriv_order=2, which the mpqc4 export does not provide (it is fixed at \
         export time; see README prerequisites). Use \
         ferric_scf::frequencies::harmonic_frequencies, which differentiates \
         analytic gradients numerically and is the validated path."
            .to_string(),
    ))
}

// ---------------------------------------------------------------------------
// Term 1: Nuclear repulsion second derivatives
// ---------------------------------------------------------------------------

/// Nuclear repulsion Hessian: `d²V_nn / dR_{A,x} dR_{B,y}`.
///
/// Shape: (3*natom, 3*natom). This is a pure geometry term — no integrals.
///
/// For A ≠ B:
///   d²(Z_A Z_B / r_AB) / dR_{Ax} dR_{By}
///     = Z_A Z_B * (−3 (R_Ax − R_Bx)(R_Ay − R_By) / r^5 + δ_{xy} / r^3)
///
/// Diagonal blocks (A = A) are the negative sum of all off-diagonal blocks.
pub fn hess_nuclear_repulsion(mol: &Molecule) -> Array2<f64> {
    let natoms = mol.atoms.len();
    let n3 = 3 * natoms;
    let mut h = Array2::<f64>::zeros((n3, n3));

    for i in 0..natoms {
        let ai = &mol.atoms[i];
        if ai.ghost {
            continue;
        }
        let za = ai.effective_z() as f64;
        let ri = [ai.x, ai.y, ai.zpos];

        for j in 0..natoms {
            if i == j {
                continue;
            }
            let aj = &mol.atoms[j];
            if aj.ghost {
                continue;
            }
            let zb = aj.effective_z() as f64;
            let rj = [aj.x, aj.y, aj.zpos];

            let d = [ri[0] - rj[0], ri[1] - rj[1], ri[2] - rj[2]];
            let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            let r = r2.sqrt();
            let r3 = r * r2;
            let r5 = r3 * r2;
            let zz = za * zb;

            // Off-diagonal block (i, j)
            for x in 0..3 {
                for y in 0..3 {
                    let val = zz * (-3.0 * d[x] * d[y] / r5
                        + if x == y { 1.0 / r3 } else { 0.0 });
                    h[(3 * i + x, 3 * j + y)] = val;
                    // Diagonal block accumulation: h[i,i] -= h[i,j]
                    h[(3 * i + x, 3 * i + y)] -= val;
                }
            }
        }
    }

    h
}

// ---------------------------------------------------------------------------
// Term 2: Skeleton one-electron Hessian
// ---------------------------------------------------------------------------

/// Skeleton one-electron Hessian: `Σ_μν D_μν · d²h_μν / dR_A dR_B`.
///
/// `h = T + V_ne` (kinetic energy + nuclear attraction).
///
/// **TODO**: Requires second-derivative integral engine support:
/// - `compute_1e_hessian_block` for kinetic integrals (∂²T/∂R_A∂R_B)
/// - Per-nucleus `compute_1e_hessian_rinv_block` for nuclear attraction
///   (decompose V_ne as sum over nuclei C: V = Σ_C -Z_C/|r-R_C|,
///   then d²/dR_A dR_B has shell-center and nuclear-center contributions)
///
/// The shim needs a `scf_engine_create_deriv2` function wrapping libint2's
/// `Engine(op, max_nprim, max_l, 2, precision)` (deriv_order=2), which
/// requires `LIBINT2_MAX_DERIV_ORDER >= 2` at libint2 compile time.
/// Current installation has `LIBINT2_MAX_DERIV_ORDER = 1`.
#[allow(dead_code)] // scaffold: kept as the implementation roadmap for the
// deriv_order=2 work; unreachable until rhf_hessian's guard is lifted.
fn skeleton_hess_1e(
    _mol: &Molecule,
    prep: &PreparedBasis,
    _d: &Array2<f64>,
) -> Result<Array2<f64>, FerricError> {
    let natoms = prep.shell_to_atom().iter().copied().max().unwrap_or(0) + 1;
    let n3 = 3 * natoms;

    // TODO: Implement when second-derivative integral engine is available.
    //
    // Structure:
    //   1. Loop over canonical shell pairs (s1, s2)
    //   2. For kinetic: compute d²T_μν/dR_A dR_B blocks
    //      - Same-center (A=A): d²/dR_Ax dR_Ay on the bra or ket center
    //        ("ipip" integrals: int1e_ipipkin for same-atom, int1e_ipkinip for cross)
    //      - Cross-center (A≠B): d/dR_Ax on bra, d/dR_By on ket
    //   3. For nuclear attraction: decompose per nucleus C
    //      - Involves int1e_ipiprinv (same-atom), int1e_iprinvip (cross-atom)
    //      - Must set rinv_at_nucleus for each nuclear center C
    //      - Three types of center pairs: (shell,shell), (shell,nuc), (nuc,nuc)
    //   4. Contract each block with D: hess[3A+x, 3B+y] += Σ_μν D_μν · d²h_μν
    //
    // The deriv_order=2 engine returns derivative blocks indexed by
    // pairs of center-coordinate indices. For 1e with 2 shell centers:
    // overlap/kinetic: (2 centers × 3 coords)^2 / 2 = 21 unique blocks
    // (triangular: xx,xy,xz,yy,yz,zz for each center pair)
    //
    // For nuclear attraction with N charge centers:
    // (2 + N) centers, ((2+N)*3 choose 2 + (2+N)*3) unique blocks

    Ok(Array2::<f64>::zeros((n3, n3)))
}

// ---------------------------------------------------------------------------
// Term 3: Overlap × energy-weighted density
// ---------------------------------------------------------------------------

/// Overlap Hessian contribution: `−Σ_μν W_μν · d²S_μν / dR_A dR_B`.
///
/// **TODO**: Requires second-derivative overlap integrals:
/// - `int1e_ipipovlp` (same-center: d²S/dR_Ax dR_Ay)
/// - `int1e_ipovlpip` (cross-center: dS/dR_Ax · dS/dR_By)
///
/// Same libint2 `deriv_order=2` requirement as the 1e skeleton.
#[allow(dead_code)] // scaffold: kept as the implementation roadmap for the
// deriv_order=2 work; unreachable until rhf_hessian's guard is lifted.
fn skeleton_hess_overlap(
    _mol: &Molecule,
    prep: &PreparedBasis,
    _w: &Array2<f64>,
) -> Result<Array2<f64>, FerricError> {
    let natoms = prep.shell_to_atom().iter().copied().max().unwrap_or(0) + 1;
    let n3 = 3 * natoms;

    // TODO: Implement when second-derivative overlap engine is available.
    //
    // Structure:
    //   1. Compute int1e_ipipovlp (9 components = 3×3, same-center second deriv)
    //   2. Compute int1e_ipovlpip (9 components, cross-center)
    //   3. For each atom A:
    //      - Same-atom block: hess[A,A] -= Σ_μ∈A Σ_ν W_μν · S^{(2)}_μν(AA) × 2
    //   4. For each pair (A, B):
    //      - Cross block: hess[A,B] -= Σ_μ∈A Σ_ν∈B W_μν · S^{(2)}_μν(AB) × 2

    Ok(Array2::<f64>::zeros((n3, n3)))
}

// ---------------------------------------------------------------------------
// Term 4: Skeleton two-electron Hessian
// ---------------------------------------------------------------------------

/// Skeleton two-electron Hessian: J and K second derivative contributions.
///
/// J contribution: `Σ_μνλσ D_μν D_λσ · d²(μν|λσ) / dR_A dR_B`
/// K contribution: `−½ Σ_μνλσ D_μλ D_νσ · d²(μν|λσ) / dR_A dR_B`
///
/// **TODO**: Requires second-derivative ERI engine (deriv_order=2):
/// - `int2e_ipip1` (both derivatives on first pair, same center)
/// - `int2e_ip1ip2` (one derivative on each pair, cross center)
/// - `int2e_ipvip1` (derivative on bra, derivative on ket of first pair)
///
/// For 4 shell centers, the second-derivative ERI has
/// (4×3 + 4×3 choose 2) / 2 = 78 unique derivative blocks.
#[allow(dead_code)] // scaffold: kept as the implementation roadmap for the
// deriv_order=2 work; unreachable until rhf_hessian's guard is lifted.
fn skeleton_hess_2e(
    _mol: &Molecule,
    prep: &PreparedBasis,
    _d: &Array2<f64>,
) -> Result<Array2<f64>, FerricError> {
    let natoms = prep.shell_to_atom().iter().copied().max().unwrap_or(0) + 1;
    let n3 = 3 * natoms;

    // TODO: Implement when second-derivative ERI engine is available.
    //
    // Structure (mirrors PySCF's _partial_hess_ejk):
    //   1. Compute same-atom diagonal contributions using int2e_ipip1
    //      (both derivatives on the same center of the first pair)
    //   2. For each atom A, compute cross-atom contributions:
    //      - int2e_ip1ip2: derivative on center A of pair 1, on center B of pair 2
    //      - int2e_ipvip1: derivative on bra of pair 1, on ket of pair 1
    //   3. Contract with density matrix for J and K separately:
    //      - J: 2e Coulomb via D_μν D_λσ contraction
    //      - K: exchange via D_μλ D_νσ contraction (with -0.5 prefactor for RHF)
    //   4. Combine: h_2e = h_J - h_K
    //
    // This is typically the most expensive component and benefits from
    // the same Schwarz-screened, deterministic-parallel shell-quartet
    // enumeration that gradient.rs uses for the first-derivative path.

    Ok(Array2::<f64>::zeros((n3, n3)))
}

// ---------------------------------------------------------------------------
// Term 5: CPKS (coupled-perturbed HF) orbital response
// ---------------------------------------------------------------------------

/// CPKS orbital response contribution to the Hessian.
///
/// The skeleton (partial) Hessian from terms 2–4 holds the orbitals fixed.
/// The CPKS response accounts for how the MO coefficients relax under a
/// nuclear perturbation. This adds the missing term:
///
/// ```text
/// d²E/dR_A dR_B |_response = 4 Σ_ia U^B_ai h^A_ia
///     − 4 Σ_ia U^B_ai ε_i S^A_ia − 2 Σ_ij S^A_ij ε^B_ij
/// ```
///
/// where `U^A` is the orbital rotation parameter from solving the CPKS
/// equations: `(A − ε_a + ε_i) U^A_ai = −h^A_ai + ε_i S^A_ai`.
///
/// **TODO**: The CPKS solve requires:
/// 1. First-derivative Fock matrix in AO basis for each atom (h^A)
///    — this REUSES the existing first-derivative infrastructure from gradient.rs
/// 2. First-derivative overlap in AO basis (S^A) — already available
/// 3. Iterative solve of the CPKS linear system
///
/// The CPKS solve itself does NOT need second-derivative integrals — it uses
/// only first derivatives and the converged MO coefficients/energies. However,
/// the response terms it produces are contracted with the skeleton Hessian from
/// terms 2–4, so the full analytic Hessian needs all components to be nonzero.
#[allow(dead_code)] // scaffold: kept as the implementation roadmap for the
// deriv_order=2 work; unreachable until rhf_hessian's guard is lifted.
fn cpks_response(
    mol: &Molecule,
    prep: &PreparedBasis,
    rhf: &ScfResult,
    h_partial: &Array2<f64>,
) -> Result<Array2<f64>, FerricError> {
    let natoms = mol.atoms.len();
    let _n3 = 3 * natoms;
    let _nbas = prep.nbasis();
    let nocc = (mol.nelec() / 2) as usize;
    let c = rhf.mos_r();
    let eps = rhf.eps_r();
    let nmo = c.ncols();
    let nvir = nmo - nocc;
    let _c_occ = c.slice(s![.., ..nocc]);
    let _c_vir = c.slice(s![.., nocc..]);

    // TODO: Full CPKS implementation requires:
    //
    // 1. Build h1_ao[atom][coord] = dF/dR_{A,x} (first-derivative Fock matrix)
    //    This needs:
    //    - dH_core/dR (kinetic + nuclear attraction first derivatives — AVAILABLE
    //      via existing Engine::new_1e_deriv)
    //    - dJ/dR and dK/dR (first-derivative ERIs — AVAILABLE via existing
    //      Engine::new_2e_deriv and compute_eri_deriv_quartet)
    //    So this part CAN be built with the current shim.
    //
    // 2. Build s1_ao[atom][coord] = dS/dR_{A,x} (first-derivative overlap — AVAILABLE)
    //
    // 3. Transform to MO basis:
    //    h1_mo = C^T · h1_ao · C_occ
    //    s1_mo = C^T · s1_ao · C_occ
    //
    // 4. Solve CPKS equations for each perturbation (3*natom equations):
    //    For each atom A and coordinate x:
    //      (ε_a - ε_i) U^{Ax}_{ai} + Σ_{bj} A_{ai,bj} U^{Ax}_{bj}
    //        = -h1_mo[Ax]_{ai} + ε_i · s1_mo[Ax]_{ai}
    //
    //    where A_{ai,bj} = 4(ai|bj) - (ab|ij) - (aj|bi) is the orbital Hessian.
    //    This is solved iteratively (DIIS or conjugate gradient).
    //
    // 5. Assemble response contribution:
    //    For each pair (A, B):
    //      de2[A,B] += 4 Σ_{ai} U^B_{ai} · h1_mo[A]_{ai}
    //      de2[A,B] -= 4 Σ_{ai} U^B_{ai} · ε_i · s1_mo[A]_{ai}
    //      de2[A,B] -= 2 Σ_{ij} s1_oo[A]_{ij} · e1[B]_{ij}
    //
    //    where e1[B]_{ij} = orbital energy response from the CPKS solve.
    //
    // NOTE: Steps 1–5 use ONLY first-derivative integrals (deriv_order=1),
    // which ARE available. The CPKS infrastructure could be implemented
    // independently of the second-derivative stubs in terms 2–4. However,
    // the total Hessian is only useful when ALL five terms are present.

    let _eps_occ = &eps[..nocc];
    let _eps_vir = &eps[nocc..nocc + nvir];

    Ok(h_partial.clone())
}

// ---------------------------------------------------------------------------
// Entry point for frequencies from analytic Hessian
// ---------------------------------------------------------------------------

/// Compute harmonic frequencies from the analytic RHF Hessian.
///
/// This is the analytic alternative to the finite-difference path in
/// [`super::frequencies::harmonic_frequencies`]. Uses [`rhf_hessian`] for
/// the Hessian, then feeds it into the existing mass-weighting + projection
/// + eigenvalue machinery.
///
/// **Currently returns an error** because the analytic Hessian requires
/// second-derivative integrals (`LIBINT2_MAX_DERIV_ORDER >= 2`) that are
/// not available in this libint2 installation.
pub fn analytic_frequencies(
    mol: &Molecule,
    prep: &PreparedBasis,
    op: Operator,
    rhf: &ScfResult,
) -> Result<super::frequencies::FrequencyResult, FerricError> {
    let hessian = rhf_hessian(mol, prep, op, rhf)?;

    let masses = super::frequencies::atom_masses(mol)?;
    super::frequencies::frequencies_from_cartesian_hessian(mol, &hessian, &masses)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn water() -> Molecule {
        Molecule::parse_xyz(
            "3\nwater\nO 0.000000 0.000000 0.117300\n\
             H 0.000000 0.757200 -0.469200\nH 0.000000 -0.757200 -0.469200\n",
            0, 1,
        )
        .unwrap()
    }

    fn h2() -> Molecule {
        Molecule::parse_xyz("2\nH2\nH 0.0 0.0 0.0\nH 0.0 0.0 0.74\n", 0, 1).unwrap()
    }

    #[test]
    fn nuclear_hessian_h2_symmetry() {
        let m = h2();
        let h = hess_nuclear_repulsion(&m);
        assert_eq!(h.dim(), (6, 6));

        // H2 along z: only zz blocks are nonzero
        // d²(1/r)/dz1 dz2 = d²(1/|z1-z2|)/dz1 dz2 = -2/r³ (off-diag)
        // d²(1/r)/dz1 dz1 = +2/r³ (diagonal, from -sum rule)
        let r = 0.74 * 1.8897259886; // Angstrom -> Bohr
        let expected = 2.0 / (r * r * r);

        // h[z1,z1] should be positive (restoring force)
        assert!(
            (h[(2, 2)] - expected).abs() < 1e-6,
            "h[z1,z1] = {}, expected {}",
            h[(2, 2)],
            expected
        );
        // h[z1,z2] should be negative
        assert!(
            (h[(2, 5)] + expected).abs() < 1e-6,
            "h[z1,z2] = {}, expected {}",
            h[(2, 5)],
            -expected
        );

        // Perpendicular (x) blocks: h[x1,x2] = ZZ*(−3·0·0/r⁵ + 1/r³) = 1/r³
        // Diagonal sum rule: h[x1,x1] = −h[x1,x2] = −1/r³
        let expected_diag_perp = -1.0 / (r * r * r);
        assert!(
            (h[(0, 0)] - expected_diag_perp).abs() < 1e-6,
            "h[x1,x1] = {}, expected {}",
            h[(0, 0)],
            expected_diag_perp
        );
    }

    #[test]
    fn nuclear_hessian_symmetric() {
        let m = water();
        let h = hess_nuclear_repulsion(&m);
        let n3 = 3 * m.atoms.len();
        for i in 0..n3 {
            for j in 0..n3 {
                assert!(
                    (h[(i, j)] - h[(j, i)]).abs() < 1e-12,
                    "nuclear Hessian not symmetric at ({i},{j}): {} vs {}",
                    h[(i, j)],
                    h[(j, i)]
                );
            }
        }
    }

    #[test]
    fn nuclear_hessian_translational_invariance() {
        let m = water();
        let h = hess_nuclear_repulsion(&m);
        let natoms = m.atoms.len();

        // Translational invariance: Σ_B d²V/dR_A dR_B = 0 for all A
        for a in 0..natoms {
            for x in 0..3 {
                let mut sum = [0.0; 3];
                for b in 0..natoms {
                    for y in 0..3 {
                        sum[y] += h[(3 * a + x, 3 * b + y)];
                    }
                }
                for y in 0..3 {
                    assert!(
                        sum[y].abs() < 1e-10,
                        "translational invariance violated: Σ_B h[{a}{x},{b}{y}] = {}",
                        sum[y],
                        b = "B",
                        y = y
                    );
                }
            }
        }
    }
}
