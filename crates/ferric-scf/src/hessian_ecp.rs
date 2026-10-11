//! Effective core potentials in the analytic Hessian (issue #285).
//!
//! `V_ECP` is a plain one-electron operator of the core Hamiltonian, so it
//! enters the analytic RHF/UHF Hessian through exactly two terms at fixed
//! density `D`:
//!
//! * the skeleton term `Σ_μν D_μν ∂²V_μν/∂R_{A,x}∂R_{B,y}`
//!   ([`ecp_skeleton_hessian`]);
//! * `∂V_ECP/∂R_{A,x}` in the first-order Fock matrices `F^x` of the CPHF
//!   response (`ecp_first_derivative_matrices`, crate-private).
//!
//! The nuclear repulsion already uses the screened charges (`effective_z`),
//! and the nuclear-attraction operator the effective charges of the prepared
//! basis, so no other term changes.
//!
//! # How the second derivative is obtained
//!
//! Neither libint2 nor ferric's ECP quadrature has second derivatives, and
//! libecpint's own are inconsistent with its first derivatives at the
//! 1e-6 relative level (sizing report `benchmarks/hessian-sizing/REPORT.md`
//! on `research/hessian-sizing`). So the second derivative is a fourth-order
//! central stencil of the analytic FIRST-derivative block kernel
//! ([`ecp_block_deriv_spherical`], the backend selected by
//! `FERRIC_ECP_BACKEND`, i.e. whichever the gradient uses), step [`STEP`].
//!
//! The displacement is RIGID PER ATOM: displacing atom `b` moves every shell
//! and the ECP centre sitting on `b`. A triple `(bra shell, ket shell, ECP)`
//! whose three centres do not include `b` is untouched, and one with all three
//! centres ON `b` is translated as a whole, so its first-derivative partials
//! are unchanged (translation invariance). Both cancel exactly in the stencil
//! (the weights sum to zero), so the kernel is run only on the remaining
//! triples (the "mask"). Every shell centre of the first-derivative block is
//! then folded onto its atom, as `matrix_deriv_cart` does for the gradient:
//! bra partial to the bra shell's atom, ket partial to the ket shell's atom,
//! centre partial to the ECP's atom.
//!
//! Angular momentum: the AO basis must stay within the 4-centre second
//! derivatives of the SCF Hessian (l <= 3, i.e. up to f functions); a higher
//! shell is a typed error here as well.

use ferric_core::basis::BasisSet;
use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_integrals::ecp::ecp_block_deriv_spherical;
use ferric_integrals::oneelectron::{ecp_inputs_with_atoms, ecp_potential_deriv, EcpInputs};
use ndarray::Array2;

/// Displacement step (Bohr) of the fourth-order stencil. The sizing probe
/// measured `|S(h) - S(2h)|` = 7.8e-10 on HI/def2-SVP at `h` = 2e-3.
pub const STEP: f64 = 2e-3;
/// `(offset in steps, weight)` of the first-derivative stencil (÷ `12·h`).
const D1: [(f64, f64); 4] = [(-2.0, 1.0), (-1.0, -8.0), (1.0, 8.0), (2.0, -1.0)];
/// Largest shell angular momentum of the second-derivative AO basis.
pub(crate) const MAX_L: i32 = 3;

/// Whether the (molecule, basis) pair carries an ECP at all: `false` means
/// the ECP Hessian terms are exactly zero and are skipped (bit-identical to
/// the all-electron Hessian).
pub(crate) fn has_ecp(mol: &Molecule, bs: &BasisSet) -> bool {
    !bs.ecps.is_empty() && mol.atoms.iter().any(|a| bs.ecp_for_element(a.z).is_some())
}

/// Whether the core charge some atom carries differs from the basis ECP's
/// (`Molecule::apply_ecp` not run, or run with another basis): the nuclear
/// charges would not match the potential, so the Hessian is refused.
pub(crate) fn ecp_inconsistent(mol: &Molecule, bs: &BasisSet) -> bool {
    mol.atoms
        .iter()
        .any(|a| a.n_core_ecp != bs.ecp_for_element(a.z).map_or(0, |d| d.n_core))
}

/// `∂V_ECP/∂R_{A,x}` as AO matrices, index `3A + x` (length `3·natoms`);
/// empty when the basis carries no ECP. Backend as the ECP gradient.
pub(crate) fn ecp_first_derivative_matrices(
    mol: &Molecule,
    bs: &BasisSet,
) -> Result<Vec<Array2<f64>>, FerricError> {
    let Some(derivs) = ecp_potential_deriv(mol, bs)? else {
        return Ok(Vec::new());
    };
    if derivs.len() != mol.atoms.len() {
        return Err(FerricError::Libint(format!(
            "ECP Hessian: {} atom blocks of dV_ECP/dR, expected {}",
            derivs.len(),
            mol.atoms.len()
        )));
    }
    Ok(derivs.into_iter().flatten().collect())
}

/// `Σ_μν D_μν ∂²V_ECP,μν/∂R_{A,x}∂R_{B,y}` (3N × 3N, Ha/Bohr²), `D` the
/// spin-summed, symmetric AO density. Zero (no work) without an ECP.
pub fn ecp_skeleton_hessian(
    mol: &Molecule,
    bs: &BasisSet,
    d: &Array2<f64>,
) -> Result<Array2<f64>, FerricError> {
    ecp_skeleton_hessian_step(mol, bs, d, STEP)
}

/// [`ecp_skeleton_hessian`] with an explicit stencil step (step-sweep anchor).
pub fn ecp_skeleton_hessian_step(
    mol: &Molecule,
    bs: &BasisSet,
    d: &Array2<f64>,
    step: f64,
) -> Result<Array2<f64>, FerricError> {
    let n3 = 3 * mol.atoms.len();
    let mut h = Array2::<f64>::zeros((n3, n3));
    let Some(base) = ecp_inputs_with_atoms(mol, bs) else {
        return Ok(h);
    };
    if let Some(sh) = base.shells.iter().find(|s| s.l > MAX_L) {
        return Err(FerricError::General(format!(
            "analytic ECP Hessian: a shell with l = {} exceeds the second-derivative \
             angular momentum (l <= {MAX_L})",
            sh.l
        )));
    }
    for b in 0..mol.atoms.len() {
        for y in 0..3 {
            for (off, wt) in D1 {
                let g = displaced(mol, b, y, off * step);
                let inp = ecp_inputs_with_atoms(&g, bs).ok_or_else(|| {
                    FerricError::General("analytic ECP Hessian: ECP lost on displacement".into())
                })?;
                let f = wt / (12.0 * step);
                if let Some(rows) = folded_first_derivative(&inp, mol.atoms.len(), b, d)? {
                    for (r, v) in rows.iter().enumerate() {
                        h[(r, 3 * b + y)] += f * v;
                    }
                }
            }
        }
    }
    // The exact skeleton is symmetric (`D` symmetric, `V` a plain operator), and
    // so is the stencil of it up to its truncation error, which is the only
    // asymmetry present (the sweep test bounds it at ~1e-9). Folding in
    // `(H + Hᵀ)/2` halves that error rather than hiding anything: the reported
    // `response_asymmetry` probes the CPHF response alone and never covered the
    // skeleton terms, which are symmetric by construction in the other terms.
    Ok((&h + &h.t()) * 0.5)
}

/// `mol` with atom `atom` moved by `d` along `axis`.
fn displaced(mol: &Molecule, atom: usize, axis: usize, d: f64) -> Molecule {
    let mut m = mol.clone();
    match axis {
        0 => m.atoms[atom].x += d,
        1 => m.atoms[atom].y += d,
        _ => m.atoms[atom].zpos += d,
    }
    m
}

/// Triples `(bra shell, ket shell, ECP)` whose first-derivative partials
/// change when atom `b` moves rigidly: some centre on `b`, but not all three
/// (module doc), for the shell pairs `i <= j` only (`V` and `D` are symmetric,
/// so the `(j, i)` block repeats the `(i, j)` one; [`folded_first_derivative`]
/// doubles the off-diagonal pairs). Index `(i · nshell + j) · necp + u`;
/// `None` when empty.
fn moving_triples(inp: &EcpInputs, b: usize) -> Option<Vec<u8>> {
    let (ns, ne) = (inp.shells.len(), inp.ecps.len());
    let mut mask = vec![0u8; ns * ns * ne];
    let mut any = false;
    for i in 0..ns {
        for j in i..ns {
            for u in 0..ne {
                let on_b = [inp.shell_atom[i], inp.shell_atom[j], inp.ecp_atom[u]]
                    .iter()
                    .filter(|&&a| a == b)
                    .count();
                if on_b == 1 || on_b == 2 {
                    mask[(i * ns + j) * ne + u] = 1;
                    any = true;
                }
            }
        }
    }
    any.then_some(mask)
}

/// `Σ_μν D_μν ∂V_μν/∂R_{A,x}` (index `3A + x`) over the triples that move
/// with atom `b`, folded onto atoms; `None` when no triple moves. `d` must be
/// symmetric.
fn folded_first_derivative(
    inp: &EcpInputs,
    natoms: usize,
    b: usize,
    d: &Array2<f64>,
) -> Result<Option<Vec<f64>>, FerricError> {
    let Some(mask) = moving_triples(inp, b) else {
        return Ok(None);
    };
    let ne = inp.ecps.len();
    let groups: Vec<usize> = (0..ne).collect();
    let blk = ecp_block_deriv_spherical(
        &inp.shells,
        &inp.shells,
        &inp.ecps,
        Some(&mask),
        &groups,
        ne,
    )?;
    let (ao_atom, ao_shell) = ao_maps(inp);
    let nbf = ao_atom.len();
    if blk.nrow != nbf || blk.ncol != nbf || d.dim() != (nbf, nbf) {
        return Err(FerricError::Libint(format!(
            "ECP Hessian: derivative block {}x{} / density {:?} vs {nbf} basis functions",
            blk.nrow,
            blk.ncol,
            d.dim()
        )));
    }
    // Only the `i <= j` blocks were computed: weight the off-diagonal shell
    // pairs by two (symmetric `D`), the diagonal ones by one.
    let dw = Array2::from_shape_fn((nbf, nbf), |(mu, nu)| {
        d[(mu, nu)]
            * if ao_shell[mu] == ao_shell[nu] {
                1.0
            } else {
                2.0
            }
    });
    let d = &dw;
    let mut out = vec![0.0; 3 * natoms];
    for x in 0..3 {
        for mu in 0..nbf {
            for nu in 0..nbf {
                let dmn = d[(mu, nu)];
                out[3 * ao_atom[mu] + x] += dmn * blk.bra[x][mu * nbf + nu];
                out[3 * ao_atom[nu] + x] += dmn * blk.ket[x][mu * nbf + nu];
            }
        }
        for (u, cen) in blk.centre.iter().enumerate() {
            let dot: f64 = d.iter().zip(&cen[x]).map(|(p, q)| p * q).sum();
            out[3 * inp.ecp_atom[u] + x] += dot;
        }
    }
    Ok(Some(out))
}

/// The atom and the shell of every spherical AO (shell order, `2l + 1`
/// functions each).
fn ao_maps(inp: &EcpInputs) -> (Vec<usize>, Vec<usize>) {
    let mut atom = Vec::new();
    let mut shell = Vec::new();
    for (i, (s, &a)) in inp.shells.iter().zip(&inp.shell_atom).enumerate() {
        for _ in 0..(2 * s.l + 1) {
            atom.push(a);
            shell.push(i);
        }
    }
    (atom, shell)
}
