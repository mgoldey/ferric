//! Spin-free exact-two-component (X2C-1e) core Hamiltonian.
//!
//! The one-electron Dirac equation of the nuclear potential is solved in the
//! uncontracted basis, its electronic states are decoupled from the
//! positronic ones, and the resulting two-component (spin-free) Hamiltonian is
//! contracted back to the orbital basis (Dyall, JCP 106, 9618 (1997);
//! Kutzelnigg and Liu, JCP 123, 241102 (2005); Liu and Peng, JCP 131, 031104
//! (2009)). Only the core Hamiltonian is transformed: the two-electron
//! interaction and property operators stay in the nonrelativistic picture.
//!
//! In a basis with overlap `S`, kinetic `T`, nuclear attraction `V` and the
//! pVp block `W = sum_a <d_a m|V|d_a n>` ([`ferric_integrals::pvp`]), the
//! modified Dirac equation is
//!
//! ```text
//! [ V  T            ] [C_L]   [ S  0         ] [C_L]
//! [ T  W/(4c^2) - T ] [C_S] = [ 0  T/(2c^2)  ] [C_S] E
//! ```
//!
//! Its `n` highest solutions are electronic and give `X = C_S C_L^-1`. With
//! `S~ = S + X^T T X / (2c^2)` and
//! `R = S^-1/2 (S^-1/2 S~ S^-1/2)^-1/2 S^1/2`,
//!
//! ```text
//! h = R^T (V + T X + X^T T - X^T T X + X^T W X / (4c^2)) R.
//! ```
//!
//! The uncontracted basis has one shell per unique exponent and angular
//! momentum of each element (PySCF's `xuncontract = True`), and
//! `h(contracted) = C^T h(uncontracted) C` with `C` from
//! [`contraction_matrix`]. These are the settings of PySCF
//! `scf.RHF(mol).sfx2c1e()` with `approx = '1e'`.
//!
//! Molecules with an ECP atom are refused: the ECP already carries the scalar
//! relativistic effects of its core, and X2C on top counts them twice.
//!
//! Equations and the numpy reference: `scripts/prototypes/x2c_derivation.md`
//! and `scripts/prototypes/x2c_proto.py`; tests: `crates/ferric-scf/tests/x2c.rs`.

use ferric_core::basis::{BasisSet, Shell};
use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::{oneelectron, pvp};
use ndarray::{s, Array2};
use ndarray_linalg::{Eigh, FactorizeInto, Solve, UPLO};
use std::collections::HashMap;

/// Speed of light in atomic units used by PySCF (`lib.param.LIGHT_SPEED`).
pub const LIGHT_SPEED: f64 = 137.035_999_679_94;

/// Number of functions of a shell of angular momentum `l`.
fn ncomp(l: i32, pure: bool) -> usize {
    let l = l as usize;
    if pure {
        2 * l + 1
    } else {
        (l + 1) * (l + 2) / 2
    }
}

/// The uncontracted shells of one element: one single-primitive shell per
/// unique exponent and `l`, angular momenta in order of first appearance,
/// exponents in file order.
fn uncontract_shells(z: i32, shells: &[Shell]) -> Result<Vec<Shell>, FerricError> {
    // (l, pure, exponents) in order of first appearance of l.
    let mut by_l: Vec<(i32, bool, Vec<f64>)> = Vec::new();
    for sh in shells {
        let pos = match by_l.iter().position(|(l, _, _)| *l == sh.l) {
            Some(p) => p,
            None => {
                by_l.push((sh.l, sh.pure, Vec::new()));
                by_l.len() - 1
            }
        };
        let (_, pure, exps) = &mut by_l[pos];
        if *pure != sh.pure {
            return Err(FerricError::Basis(format!(
                "X2C: element Z={z} mixes spherical and Cartesian l={} shells, so the \
                 uncontracted basis has no single form for that angular momentum",
                sh.l
            )));
        }
        for &e in &sh.exponents {
            if !exps.iter().any(|&x| x.to_bits() == e.to_bits()) {
                exps.push(e);
            }
        }
    }
    Ok(by_l
        .into_iter()
        .flat_map(|(l, pure, exps)| {
            exps.into_iter().map(move |e| Shell {
                l,
                pure,
                exponents: vec![e],
                coefficients: vec![1.0],
            })
        })
        .collect())
}

/// The uncontracted form of `bs`: one single-primitive shell per unique
/// exponent and angular momentum of every element. ECP tables are dropped.
pub fn uncontracted_basis(bs: &BasisSet) -> Result<BasisSet, FerricError> {
    let mut shells = HashMap::new();
    for (&z, shs) in &bs.shells {
        shells.insert(z, uncontract_shells(z, shs)?);
    }
    Ok(BasisSet {
        name: format!("{}-uncontracted", bs.name),
        shells,
        ecps: HashMap::new(),
    })
}

/// `C` of shape `(n_uncontracted, n_contracted)` with
/// `mu = sum_k C[k, mu] chi_k`: each contracted AO of `bs` on `mol` as a
/// combination of the AOs of [`uncontracted_basis`]`(bs)` on the same centre,
/// in ferric's shell order for both bases.
///
/// The coefficients multiply unit-normalized primitives and are divided by
/// the norm of the contraction, `sqrt(c^T S_l c)` with the same-centre
/// primitive overlap `S_l(a, b) = (2 sqrt(ab) / (a + b))^(l + 3/2)`, as
/// libint2 normalizes a contracted shell.
pub fn contraction_matrix(mol: &Molecule, bs: &BasisSet) -> Result<Array2<f64>, FerricError> {
    let mut n_unc = 0usize;
    let mut n_con = 0usize;
    let mut unc_per_atom = Vec::with_capacity(mol.atoms.len());
    for atom in &mol.atoms {
        let con = bs
            .for_element(atom.z)
            .ok_or_else(|| bs.missing_element_error(atom.z))?;
        let unc = uncontract_shells(atom.z, con)?;
        n_unc += unc.iter().map(|s| ncomp(s.l, s.pure)).sum::<usize>();
        n_con += con.iter().map(|s| ncomp(s.l, s.pure)).sum::<usize>();
        unc_per_atom.push(unc);
    }
    let mut c = Array2::<f64>::zeros((n_unc, n_con));
    let mut row0 = 0usize;
    let mut col = 0usize;
    for (atom, unc) in mol.atoms.iter().zip(&unc_per_atom) {
        let mut index: HashMap<(i32, u64), usize> = HashMap::new();
        let mut r = row0;
        for sh in unc {
            index.insert((sh.l, sh.exponents[0].to_bits()), r);
            r += ncomp(sh.l, sh.pure);
        }
        for sh in bs.for_element(atom.z).unwrap_or_default() {
            let lf = sh.l as f64;
            let mut norm2 = 0.0;
            for (a, ca) in sh.exponents.iter().zip(&sh.coefficients) {
                for (b, cb) in sh.exponents.iter().zip(&sh.coefficients) {
                    norm2 += ca * cb * (2.0 * (a * b).sqrt() / (a + b)).powf(lf + 1.5);
                }
            }
            if !(norm2 > 0.0 && norm2.is_finite()) {
                return Err(FerricError::Basis(format!(
                    "X2C: a contracted l={} shell of {} has self-overlap {norm2}",
                    sh.l, atom.symbol
                )));
            }
            let scale = 1.0 / norm2.sqrt();
            for m in 0..ncomp(sh.l, sh.pure) {
                for (e, ck) in sh.exponents.iter().zip(&sh.coefficients) {
                    let k = index[&(sh.l, e.to_bits())];
                    c[(k + m, col)] += ck * scale;
                }
                col += 1;
            }
        }
        row0 = r;
    }
    Ok(c)
}

fn lapack(what: &str, e: impl std::fmt::Display) -> FerricError {
    FerricError::Lapack(format!("X2C {what}: {e}"))
}

/// `S^p` of a symmetric positive definite matrix, by its eigendecomposition.
fn sym_pow(s: &Array2<f64>, p: f64, what: &str) -> Result<Array2<f64>, FerricError> {
    let (e, u) = s.eigh(UPLO::Lower).map_err(|err| lapack(what, err))?;
    if let Some(bad) = e.iter().find(|&&x| !(x > 0.0)) {
        return Err(FerricError::Lapack(format!(
            "X2C {what}: matrix is not positive definite (eigenvalue {bad:e})"
        )));
    }
    let mut us = u.clone();
    for (j, &ej) in e.iter().enumerate() {
        us.column_mut(j).mapv_inplace(|x| x * ej.powf(p));
    }
    Ok(us.dot(&u.t()))
}

/// The spin-free X2C-1e Hamiltonian in the basis of `s`, `t`, `v`, `w`
/// (overlap, kinetic, nuclear attraction, pVp), speed of light `c`. This basis
/// must be flexible enough to represent the small component; use the
/// uncontracted basis ([`x2c1e_hcore`] does).
pub fn x2c1e_transform(
    s: &Array2<f64>,
    t: &Array2<f64>,
    v: &Array2<f64>,
    w: &Array2<f64>,
    c: f64,
) -> Result<Array2<f64>, FerricError> {
    check_light_speed(c)?;
    let n = s.nrows();
    for (name, m) in [("T", t), ("V", v), ("W", w)] {
        if m.dim() != (n, n) {
            return Err(FerricError::General(format!(
                "X2C: {name} is {:?}, S is ({n}, {n})",
                m.dim()
            )));
        }
    }
    let c2 = c * c;
    let mut h4 = Array2::<f64>::zeros((2 * n, 2 * n));
    let mut m4 = Array2::<f64>::zeros((2 * n, 2 * n));
    h4.slice_mut(s![..n, ..n]).assign(v);
    h4.slice_mut(s![..n, n..]).assign(t);
    h4.slice_mut(s![n.., ..n]).assign(t);
    h4.slice_mut(s![n.., n..]).assign(&(w / (4.0 * c2) - t));
    m4.slice_mut(s![..n, ..n]).assign(s);
    m4.slice_mut(s![n.., n..]).assign(&(t / (2.0 * c2)));
    let (_, (vec, _)) = (h4, m4)
        .eigh(UPLO::Lower)
        .map_err(|e| lapack("Dirac eigenproblem", e))?;
    // Electronic solutions: the n highest eigenvalues (ascending order).
    let cl = vec.slice(s![..n, n..]).to_owned();
    let cs = vec.slice(s![n.., n..]).to_owned();
    // X = C_S C_L^-1, row by row: C_L^T X[j, :]^T = C_S[j, :]^T.
    let lu = cl
        .t()
        .to_owned()
        .factorize_into()
        .map_err(|e| lapack("X", e))?;
    let mut x = Array2::<f64>::zeros((n, n));
    for j in 0..n {
        let row = lu
            .solve(&cs.row(j).to_owned())
            .map_err(|e| lapack("X", e))?;
        x.row_mut(j).assign(&row);
    }
    let tx = t.dot(&x);
    let xt_tx = x.t().dot(&tx);
    let s_tilde = s + &(&xt_tx / (2.0 * c2));
    let lag = v + &tx + tx.t() - &xt_tx + &(x.t().dot(w).dot(&x) / (4.0 * c2));
    let s_half = sym_pow(s, 0.5, "S^1/2")?;
    let s_mhalf = sym_pow(s, -0.5, "S^-1/2")?;
    let inner = sym_pow(&s_mhalf.dot(&s_tilde).dot(&s_mhalf), -0.5, "R")?;
    let r = s_mhalf.dot(&inner).dot(&s_half);
    let h = r.t().dot(&lag).dot(&r);
    Ok(0.5 * (&h + &h.t()))
}

fn check_light_speed(c: f64) -> Result<(), FerricError> {
    if c.is_finite() && c > 0.0 {
        Ok(())
    } else {
        Err(FerricError::General(format!(
            "X2C: the speed of light must be finite and positive, got {c}"
        )))
    }
}

/// Spin-free X2C-1e core Hamiltonian of `mol` in the AO basis of `bs`
/// (ferric's shell order, as [`oneelectron::hcore`]), speed of light `c` in
/// atomic units ([`LIGHT_SPEED`] for PySCF's value). It replaces `T + V`.
///
/// Refuses a molecule with an ECP atom (the ECP's scalar relativity would be
/// counted twice) and a basis whose second-derivative nuclear integrals the
/// linked libint2 cannot build (the pVp block).
pub fn x2c1e_hcore(mol: &Molecule, bs: &BasisSet, c: f64) -> Result<Array2<f64>, FerricError> {
    check_light_speed(c)?;
    if let Some(a) = mol
        .atoms
        .iter()
        .find(|a| !a.ghost && bs.ecp_for_element(a.z).is_some())
    {
        return Err(FerricError::Basis(format!(
            "X2C: basis {:?} carries an ECP for {}; the ECP already includes the \
             scalar relativistic effects of the core, so X2C on top would count them \
             twice. Use an all-electron basis",
            bs.name, a.symbol
        )));
    }
    let unc = uncontracted_basis(bs)?;
    let prep = PreparedBasis::new(mol, &unc)?;
    let cmat = contraction_matrix(mol, bs)?;
    if cmat.nrows() != prep.nbasis() {
        return Err(FerricError::Basis(format!(
            "X2C: contraction matrix has {} rows, the uncontracted basis {} functions",
            cmat.nrows(),
            prep.nbasis()
        )));
    }
    let s_u = oneelectron::overlap(&prep);
    let t_u = oneelectron::kinetic(&prep);
    let v_u = oneelectron::nuclear(&prep);
    let w_u = pvp::pvp_nuclear(&prep)?;
    let h_u = x2c1e_transform(&s_u, &t_u, &v_u, &w_u, c)?;
    let h = cmat.t().dot(&h_u).dot(&cmat);
    Ok(0.5 * (&h + &h.t()))
}
