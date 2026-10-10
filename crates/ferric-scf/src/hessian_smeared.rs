//! Gaussian-smeared external charges in the analytic Hessian (issue #358).
//!
//! The smeared-charge attraction is `V_μν = −Σ_i q_i (μν|g_i)/norm_i` (see
//! [`ferric_integrals::oneelectron::smeared_attraction`]), a three-centre
//! Coulomb integral against one s-Gaussian `g_i` per site. The sites are fixed
//! in space (QM–QM block only, as for point charges), so the QM-atom Hessian
//! needs, at fixed density `D`,
//!
//! * the skeleton term `Σ_μν D_μν ∂²V_μν/∂x∂y` ([`smeared_skeleton_hessian`]),
//! * the first-derivative AO matrices `∂V/∂x` for `F^x`
//!   ([`smeared_first_derivative_matrices`]),
//!
//! plus the classical charge–nucleus term, which lives in
//! `ExternalPotential::charge_nuclear_hessian`.
//!
//! # How the second derivatives are obtained
//!
//! The wheel's libint2 (2.7.2 small export) has no three-centre SECOND
//! derivatives (`LIBINT2_DERIV_ERI3_ORDER 1`), so the second derivative is a
//! fourth-order central stencil of the FIRST-derivative blocks
//! ([`Engine::compute_eri3_deriv_shifted`]), step [`STEP`]. The displacement
//! must be RIGID PER ATOM: every shell sitting on the displaced atom moves.
//! Moving one shell of a same-atom pair is not an atomic displacement and is
//! off by ~1e-3 (negative control in
//! `crates/ferric-integrals/tests/smeared_deriv2_stencil.rs`). The sizing
//! report (benchmarks/hessian-sizing/REPORT.md on `research/hessian-sizing`)
//! measured the stencil against independent mixed differences of the value
//! blocks at 7e-10 (width 1.2 Bohr) down to 4e-13 (width 0.01 Bohr).

use ferric_core::external_potential::SmearedCharge;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::engine::Engine;
use ferric_integrals::operator::Operator;
use ferric_integrals::site_basis::SiteBasis;
use ndarray::Array2;

/// Displacement step (Bohr) of the fourth-order stencil.
const STEP: f64 = 2e-3;
/// `(offset in steps, weight)` of the first-derivative stencil (÷ `12·STEP`).
const D1: [(f64, f64); 4] = [(-2.0, 1.0), (-1.0, -8.0), (1.0, 8.0), (2.0, -1.0)];
const ZERO: [f64; 3] = [0.0; 3];

/// The synthetic site basis and its three-centre first-derivative engine.
struct Sites {
    basis: SiteBasis,
    eng: Engine,
}

impl Sites {
    fn new(prep: &PreparedBasis, smeared: &[SmearedCharge]) -> Result<Self, FerricError> {
        let sites: Vec<[f64; 4]> = smeared
            .iter()
            .map(|s| [s.x, s.y, s.z, 1.0 / (s.width * s.width)])
            .collect();
        let basis = SiteBasis::new(&sites, 0)?;
        let eng = Engine::new_3center_deriv(Operator::coulomb(), prep, &basis.prep, 1e-14)?;
        Ok(Sites { basis, eng })
    }

    /// `−q_i/norm_i`: the factor taking `(μν|g_i)` to the attraction `V_μν`.
    fn coeff(&self, smeared: &[SmearedCharge], i: usize) -> f64 {
        -smeared[i].q / self.basis.norm_int[i]
    }
}

/// `∂V/∂R_{A,x}` for every QM coordinate (index `3A + x`): the basis-centre
/// dependence of the smeared attraction (the sites are fixed).
pub(crate) fn smeared_first_derivative_matrices(
    prep: &PreparedBasis,
    smeared: &[SmearedCharge],
) -> Result<Vec<Array2<f64>>, FerricError> {
    let natoms = prep.atoms().len();
    let nbf = prep.nbasis();
    let mut mats = vec![Array2::<f64>::zeros((nbf, nbf)); 3 * natoms];
    if smeared.is_empty() {
        return Ok(mats);
    }
    let mut st = Sites::new(prep, smeared)?;
    let (dims, offs, sh2at) = (
        prep.shell_dims(),
        prep.shell_offsets(),
        prep.shell_to_atom(),
    );
    for i in 0..smeared.len() {
        let (sh_p, c) = (st.basis.site_shell[i], st.coeff(smeared, i));
        for s1 in 0..prep.nshells() {
            for s2 in 0..=s1 {
                let Some(blk) = st
                    .eng
                    .compute_eri3_deriv(prep, &st.basis.prep, sh_p, s1, s2)
                else {
                    continue;
                };
                let (n1, n2) = (dims[s1], dims[s2]);
                let n = n1 * n2;
                for (k, atom) in [sh2at[s1], sh2at[s2]].into_iter().enumerate() {
                    for x in 0..3 {
                        let src = &blk[(3 + 3 * k + x) * n..(4 + 3 * k + x) * n];
                        let m = &mut mats[3 * atom + x];
                        scatter(m, src, (offs[s1], n1, offs[s2], n2), c, s1 != s2);
                    }
                }
            }
        }
    }
    Ok(mats)
}

/// `m[(o1+a, o2+b)] += c·src[a·n2+b]`, and the transpose when `mirror`.
fn scatter(
    m: &mut Array2<f64>,
    src: &[f64],
    pair: (usize, usize, usize, usize),
    c: f64,
    mirror: bool,
) {
    let (o1, n1, o2, n2) = pair;
    for a in 0..n1 {
        for b in 0..n2 {
            let v = c * src[a * n2 + b];
            m[(o1 + a, o2 + b)] += v;
            if mirror {
                m[(o2 + b, o1 + a)] += v;
            }
        }
    }
}

/// Shifts `[site, sh1, sh2]` displacing ATOM `b` along `axis` by `d`: every
/// shell of the pair sitting on `b` moves.
fn atom_shift(atoms: [usize; 2], b: usize, axis: usize, d: f64) -> [[f64; 3]; 3] {
    let mut sh = [ZERO; 3];
    for (k, &a) in atoms.iter().enumerate() {
        if a == b {
            sh[1 + k][axis] = d;
        }
    }
    sh
}

/// AO weights `D_μν` (+ `D_νμ` when the canonical pair `s1 > s2` stands for
/// both orders) of one shell pair, times `c`.
fn weights(d: &Array2<f64>, pair: (usize, usize, usize, usize), c: f64, both: bool) -> Vec<f64> {
    let (o1, n1, o2, n2) = pair;
    let mut w = Vec::with_capacity(n1 * n2);
    for a in 0..n1 {
        for b in 0..n2 {
            let (mu, nu) = (o1 + a, o2 + b);
            w.push(
                c * if both {
                    d[(mu, nu)] + d[(nu, mu)]
                } else {
                    d[(mu, nu)]
                },
            );
        }
    }
    w
}

/// Skeleton term `Σ_μν D_μν ∂²V_μν/∂R_{A,x}∂R_{B,y}` (3N × 3N, Ha/Bohr²).
pub(crate) fn smeared_skeleton_hessian(
    prep: &PreparedBasis,
    smeared: &[SmearedCharge],
    d: &Array2<f64>,
) -> Result<Array2<f64>, FerricError> {
    let natoms = prep.atoms().len();
    let mut h = Array2::<f64>::zeros((3 * natoms, 3 * natoms));
    if smeared.is_empty() {
        return Ok(h);
    }
    let mut st = Sites::new(prep, smeared)?;
    let (dims, offs, sh2at) = (
        prep.shell_dims(),
        prep.shell_offsets(),
        prep.shell_to_atom(),
    );
    for i in 0..smeared.len() {
        let (sh_p, c) = (st.basis.site_shell[i], st.coeff(smeared, i));
        for s1 in 0..prep.nshells() {
            for s2 in 0..=s1 {
                let pair = (offs[s1], dims[s1], offs[s2], dims[s2]);
                let w = weights(d, pair, c, s1 != s2);
                let atoms = [sh2at[s1], sh2at[s2]];
                pair_stencil(&mut h, &mut st, prep, (sh_p, s1, s2), atoms, &w)?;
            }
        }
    }
    Ok(h)
}

/// Stencil contribution of one (site, shell pair): for every atom `b` carrying
/// a shell of the pair, every axis `y` and stencil point, the first-derivative
/// blocks give `∂/∂R_{A,x}` of the contracted integral for each atom `A` of
/// the pair.
fn pair_stencil(
    h: &mut Array2<f64>,
    st: &mut Sites,
    prep: &PreparedBasis,
    shells: (usize, usize, usize),
    atoms: [usize; 2],
    w: &[f64],
) -> Result<(), FerricError> {
    let (sh_p, s1, s2) = shells;
    let n = w.len();
    let distinct: &[usize] = if atoms[0] == atoms[1] {
        &atoms[..1]
    } else {
        &atoms
    };
    for &b in distinct {
        for y in 0..3 {
            for (off, wt) in D1 {
                let sh = atom_shift(atoms, b, y, off * STEP);
                let Some(blk) =
                    st.eng
                        .compute_eri3_deriv_shifted(prep, &st.basis.prep, sh_p, s1, s2, sh)?
                else {
                    continue;
                };
                let f = wt / (12.0 * STEP);
                for (k, &a) in atoms.iter().enumerate() {
                    for x in 0..3 {
                        let col = &blk[(3 + 3 * k + x) * n..(4 + 3 * k + x) * n];
                        let v: f64 = col.iter().zip(w).map(|(g, wi)| g * wi).sum();
                        h[(3 * a + x, 3 * b + y)] += f * v;
                    }
                }
            }
        }
    }
    Ok(())
}
