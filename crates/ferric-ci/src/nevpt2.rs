//! Strongly contracted NEVPT2 (SC-NEVPT2) on a CAS-CI or CASSCF reference.
//!
//! Angeli, Cimiraglia and Malrieu, JCP 117, 9138 (2002): the zeroth-order
//! Hamiltonian is Dyall's, the first-order space is spanned by one strongly
//! contracted function `Ψ_l = P_l H Ψ_0` per external label `l`, and the
//! second-order energy is the sum of eight class energies
//!
//! ```text
//!   E_class = − Σ_l N_l / (Δ_l + K_l / N_l),   N_l = ⟨Ψ_l|Ψ_l⟩,
//!   K_l = ⟨Ψ_l| H_act − E_0 |Ψ_l⟩,   Δ_l = Σ_{r∈l} ε_r − Σ_{i∈l} ε_i,
//! ```
//!
//! skipping labels with `|N_l| <= 1e-14`
//! ([`NORM_ZERO`](crate::nevpt2::NORM_ZERO)). Definitions, conventions and
//! the derivation: `scripts/prototypes/nevpt2_derivation.md`
//! and `crates/ferric-ci/theory.md` ("SC-NEVPT2").
//!
//! # Orbitals
//!
//! `F^I = h + J[D^c] − ½K[D^c]` (`D^c = 2 C_c C_cᵀ`) and
//! `F = F^I + F^A`, `F^A = J[D^a] − ½K[D^a]`, `D^a = C_a γ C_aᵀ`. The core
//! and virtual orbitals are the eigenvectors of the core and virtual blocks of
//! `F` (PySCF `mcscf.CASCI.canonicalize`), and `ε` their eigenvalues. The
//! active orbitals are used as given: every active index is summed, so the
//! energy does not depend on the active basis (PySCF rotates to natural
//! orbitals; the result is the same). With [`Nevpt2Config::canonicalize`]
//! off the given core and virtual orbitals are used, `ε` is the diagonal of
//! `F`, and an off-diagonal element of either block above
//! [`CANONICAL_TOL`](crate::nevpt2::CANONICAL_TOL) is an error.
//!
//! # Evaluation
//!
//! The class formulas are those of PySCF `mrpt.nevpt2` (`Sr`, `Si`, `Sijrs`,
//! `Sijr`, `Srsi`, `Srs`, `Sij`, `Sir` and the `make_a*`/`make_k27`
//! intermediates), transcribed index for index as `einsum` specs evaluated
//! by [`einsum`](crate::nevpt2::einsum) on the E-product RDMs `⟨E E⟩`,
//! `⟨E E E⟩` ([`crate::rdm3::e_product_rdms`]), the hole RDMs
//! ([`crate::rdm3::hole_rdms`]) and the `(aa|aa)`-contracted 4-RDM
//! intermediates `f3ca`/`f3ac` ([`crate::rdm3::nevpt_rdms`]). No 4-RDM is
//! formed. `h1e` below is the active block of `F^I` and `h2e[p,q,r,s] =
//! (pr|qs)`, as in PySCF.
//!
//! # Cost and memory
//!
//! The AO two-electron integrals are held densely (`nbas^4`), as in
//! [`crate::casscf`], and ten MO blocks are transformed from them. The
//! largest active intermediates are `n^6` (`n` active orbitals), and the
//! most expensive single contraction is `O(n^8)` (the `a16`/`a22` terms).
//! The external contractions cost `O(n_v n^6)` (class r), `O(n_c n^6)`
//! (class i), `O(n_v^2 n^4)` (rs), `O(n_c^2 n^4)` (ij), `O(n_c n_v n^4)`
//! (ir) and `O(n_c^2 n_v^2)` (ijrs). Everything is reserved through one
//! [`MemoryPlan`] before the AO block is built; a run past the budget is a
//! [`FerricError::MemoryExhausted`] before any large allocation.

use crate::casscf::{dense_jk, CasScfConfig, CasScfResult};
use crate::hamiltonian::DeterminantSpace;
use crate::integrals::{dense_ao_eri, transform_eri_mixed, ActiveSpaceIntegrals};
use crate::rdm3::{e_product_rdms, hole_rdms, nevpt_rdms};
use crate::sigma::DirectSigma;
use crate::{strings, CasCiConfig, CasCiResult};
use ferric_core::memory::plan::{Lifetime, MemoryPlan};
use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::oneelectron::hcore_ecp;
use ndarray::{s, Array2, ArrayD, ArrayView2, IxDyn};
use ndarray_linalg::{Eigh, UPLO};
use rayon::prelude::*;

/// Labels with `|N_l|` at or below this are skipped (PySCF
/// `mrpt.nevpt2.NUMERICAL_ZERO`).
pub const NORM_ZERO: f64 = 1e-14;

/// Largest off-diagonal element of the core or virtual block of `F`
/// accepted when [`Nevpt2Config::canonicalize`] is off (Ha).
pub const CANONICAL_TOL: f64 = 1e-8;

/// Settings of [`sc_nevpt2`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Nevpt2Config {
    /// Diagonalize the core and virtual blocks of `F` (true), or use the
    /// given core and virtual orbitals after checking that both blocks are
    /// diagonal within [`CANONICAL_TOL`] (false).
    pub canonicalize: bool,
    /// Largest accepted `‖H c − E c‖` of the reference CI vector. The class
    /// formulas assume `Ψ_0` is an eigenvector of the active Hamiltonian.
    pub max_ci_residual: f64,
    /// Memory ceiling in bytes (see [`CasCiConfig::memory_budget_bytes`]).
    pub memory_budget_bytes: Option<usize>,
}

impl Default for Nevpt2Config {
    fn default() -> Self {
        Nevpt2Config {
            canonicalize: true,
            max_ci_residual: 1e-8,
            memory_budget_bytes: None,
        }
    }
}

/// A CAS reference: orbitals, partition and CI vector.
#[derive(Debug, Clone, Copy)]
pub struct Nevpt2Reference<'a> {
    /// Orbitals `nbas x nmo`: core, active, virtual columns.
    pub mo_coeff: &'a Array2<f64>,
    /// Doubly occupied core orbitals.
    pub n_core: usize,
    /// Active orbitals.
    pub n_active: usize,
    /// Active electrons `(n_alpha, n_beta)`; must be equal.
    pub n_elec_active: (usize, usize),
    /// Normalized CI vector in the active orbitals of `mo_coeff`, layout of
    /// [`CasCiResult::ci_vector`].
    pub ci_vector: &'a [f64],
}

impl<'a> Nevpt2Reference<'a> {
    /// The reference of a converged CASSCF run; `config` is the one the run
    /// used. An unconverged run is an error.
    pub fn from_casscf(
        result: &'a CasScfResult,
        config: &CasScfConfig,
    ) -> Result<Self, FerricError> {
        if !result.converged {
            return Err(FerricError::Convergence(format!(
                "NEVPT2: the CASSCF reference did not converge ({} macro iterations, |g| = {:.2e})",
                result.iterations, result.grad_norm
            )));
        }
        Ok(Nevpt2Reference {
            mo_coeff: &result.mo_coeff,
            n_core: config.n_core,
            n_active: config.n_active,
            n_elec_active: config.n_elec_active,
            ci_vector: &result.ci_vector,
        })
    }

    /// The reference of a converged CAS-CI run on the orbitals `mo_coeff`
    /// (the RHF orbitals the run used); `config` is the one the run used.
    pub fn from_casci(
        result: &'a CasCiResult,
        mo_coeff: &'a Array2<f64>,
        config: &CasCiConfig,
    ) -> Result<Self, FerricError> {
        if !result.converged {
            return Err(FerricError::Convergence(format!(
                "NEVPT2: the CAS-CI reference did not converge (residual {:.2e})",
                result.residual_norm
            )));
        }
        Ok(Nevpt2Reference {
            mo_coeff,
            n_core: config.active_start,
            n_active: config.n_active,
            n_elec_active: config.n_elec_active,
            ci_vector: &result.ci_vector,
        })
    }
}

/// One number per SC-NEVPT2 class, named as in PySCF (`Sr` = class r, ...).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Nevpt2Classes {
    /// One virtual particle, one active hole (`V^(-1)'`, label r).
    pub sr: f64,
    /// One core hole, one active particle (`V^(+1)'`, label i).
    pub si: f64,
    /// Two core holes, two virtual particles (`V^(0)`, labels ij, rs).
    pub sijrs: f64,
    /// Two core holes, one virtual particle (`V^(+1)`).
    pub sijr: f64,
    /// One core hole, two virtual particles (`V^(-1)`).
    pub srsi: f64,
    /// Two virtual particles, two active holes (`V^(-2)`).
    pub srs: f64,
    /// Two core holes, two active particles (`V^(+2)`).
    pub sij: f64,
    /// One core hole, one virtual particle (`V^(0)'`).
    pub sir: f64,
}

impl Nevpt2Classes {
    /// PySCF names, in the order of [`Self::to_array`].
    pub const NAMES: [&'static str; 8] = ["Sr", "Si", "Sijrs", "Sijr", "Srsi", "Srs", "Sij", "Sir"];

    /// The eight values in the order of [`Self::NAMES`].
    pub fn to_array(&self) -> [f64; 8] {
        [
            self.sr, self.si, self.sijrs, self.sijr, self.srsi, self.srs, self.sij, self.sir,
        ]
    }

    /// Sum of the eight values.
    pub fn total(&self) -> f64 {
        self.to_array().iter().sum()
    }
}

/// Result of [`sc_nevpt2`].
#[derive(Debug, Clone)]
#[must_use]
pub struct Nevpt2Result {
    /// Second-order energy, the sum of [`Self::energies`] (Ha).
    pub e_corr: f64,
    /// Class energies (Ha).
    pub energies: Nevpt2Classes,
    /// Class norms `Σ_l N_l` over every label (PySCF's `norm`).
    pub norms: Nevpt2Classes,
    /// Reference energy `E_core + ⟨Ψ_0|H_act|Ψ_0⟩` at the given orbitals (Ha).
    pub e_reference: f64,
    /// `‖H c − E c‖` of the reference CI vector.
    pub ci_residual_norm: f64,
    /// Orbitals used: canonical core and virtual columns, the given active
    /// columns.
    pub mo_coeff: Array2<f64>,
    /// Diagonal of `F^I + F^A` in [`Self::mo_coeff`] (Ha); for the core and
    /// virtual orbitals these are the `ε` of the denominators.
    pub mo_energy: Vec<f64>,
    /// Core orbitals.
    pub n_core: usize,
    /// Active orbitals.
    pub n_active: usize,
    /// Virtual orbitals.
    pub n_virtual: usize,
}

type T = ArrayD<f64>;

/// Most operands of one [`einsum`] call.
const MAX_OPERANDS: usize = 4;
/// Most distinct index labels of one [`einsum`] call.
const MAX_LABELS: usize = 16;

/// `numpy.einsum(spec, *ops)` for explicit-output specs (`"ab,bc->ac"`), any
/// number of operands up to 4, repeated labels within an operand (traces and
/// diagonals) allowed. Each output element is one sequential sum over the
/// summed labels in row-major order, so the result does not depend on the
/// thread count. Cost: the product of all label extents.
pub fn einsum(spec: &str, ops: &[&T]) -> Result<T, FerricError> {
    let bad = |m: &str| FerricError::General(format!("NEVPT2 einsum '{spec}': {m}"));
    let (lhs, out) = spec.split_once("->").ok_or_else(|| bad("no '->'"))?;
    let ins: Vec<&[u8]> = lhs.split(',').map(str::as_bytes).collect();
    if ins.len() != ops.len() || ops.len() > MAX_OPERANDS {
        return Err(bad("operand count"));
    }
    let mut labels: Vec<u8> = Vec::new();
    for &c in out.as_bytes() {
        if labels.contains(&c) {
            return Err(bad("repeated output label"));
        }
        labels.push(c);
    }
    let nout = labels.len();
    for term in &ins {
        for &c in *term {
            if !labels.contains(&c) {
                labels.push(c);
            }
        }
    }
    let nl = labels.len();
    if nl > MAX_LABELS {
        return Err(bad("too many labels"));
    }
    let nops = ops.len();
    let mut ext: Vec<Option<usize>> = vec![None; nl];
    let mut strides = [[0usize; MAX_LABELS]; MAX_OPERANDS];
    for (k, (term, op)) in ins.iter().zip(ops).enumerate() {
        let shape = op.shape();
        if term.len() != shape.len() {
            return Err(bad("operand rank"));
        }
        let mut row_major = vec![0usize; shape.len()];
        let mut st = 1usize;
        for a in (0..shape.len()).rev() {
            row_major[a] = st;
            st *= shape[a];
        }
        for (a, &c) in term.iter().enumerate() {
            let li = labels
                .iter()
                .position(|&x| x == c)
                .ok_or_else(|| bad("label"))?;
            match ext[li] {
                None => ext[li] = Some(shape[a]),
                Some(e) if e != shape[a] => return Err(bad("extent mismatch")),
                Some(_) => {}
            }
            strides[k][li] += row_major[a];
        }
    }
    let ext: Vec<usize> = ext
        .into_iter()
        .map(|e| e.ok_or_else(|| bad("output label not in any operand")))
        .collect::<Result<_, _>>()?;
    let data: Vec<_> = ops.iter().map(|o| o.as_standard_layout()).collect();
    let slices: Vec<&[f64]> = data
        .iter()
        .map(|d| d.as_slice().ok_or_else(|| bad("layout")))
        .collect::<Result<_, _>>()?;
    let out_shape = ext[..nout].to_vec();
    let nsum: usize = ext[nout..].iter().product();
    let n_out: usize = out_shape.iter().product();
    let mut res = vec![0.0f64; n_out];
    res.par_iter_mut().enumerate().for_each(|(o, r)| {
        if nsum == 0 {
            return;
        }
        let mut off = [0usize; MAX_OPERANDS];
        let mut rem = o;
        for li in (0..nout).rev() {
            let d = rem % ext[li];
            rem /= ext[li];
            for k in 0..nops {
                off[k] += d * strides[k][li];
            }
        }
        let ns = nl - nout;
        let mut dig = [0usize; MAX_LABELS];
        let mut acc = 0.0;
        'sum: loop {
            let mut p = 1.0;
            for k in 0..nops {
                p *= slices[k][off[k]];
            }
            acc += p;
            let mut j = ns;
            loop {
                if j == 0 {
                    break 'sum;
                }
                j -= 1;
                let li = nout + j;
                dig[j] += 1;
                for k in 0..nops {
                    off[k] += strides[k][li];
                }
                if dig[j] < ext[li] {
                    break;
                }
                for k in 0..nops {
                    off[k] -= strides[k][li] * ext[li];
                }
                dig[j] = 0;
            }
        }
        *r = acc;
    });
    ArrayD::from_shape_vec(IxDyn(&out_shape), res).map_err(|e| bad(&e.to_string()))
}

/// `a.transpose(axes)` (numpy semantics) as an owned standard-layout array.
fn perm(a: &T, axes: &[usize]) -> T {
    a.view()
        .permuted_axes(IxDyn(axes))
        .as_standard_layout()
        .into_owned()
}

fn eye(n: usize) -> T {
    Array2::<f64>::eye(n).into_dyn()
}

/// PySCF `_norm_to_energy` over the given labels: `(Σ N, −Σ N/(Δ + K/N))`,
/// skipping `|N| <= NORM_ZERO`.
fn norm_to_energy(labels: impl Iterator<Item = (f64, f64, f64)>) -> (f64, f64) {
    let (mut nt, mut e) = (0.0, 0.0);
    for (n, h, d) in labels {
        nt += n;
        if n.abs() > NORM_ZERO {
            e -= n / (d + h / n);
        }
    }
    (nt, e)
}

/// The inputs of the class formulas, in PySCF's layout.
struct Inputs<'a> {
    nc: usize,
    nv: usize,
    eps_c: &'a [f64],
    eps_v: &'a [f64],
    /// Active block of `F^I`.
    h1e: T,
    /// `h2e[p,q,r,s] = (pr|qs)` over active orbitals.
    h2e: T,
    dm1: T,
    /// `⟨E_pq E_rs⟩`.
    dm2: T,
    /// `⟨E_pq E_rs E_tu⟩`.
    dm3: T,
    f3ca: T,
    f3ac: T,
    hdm1: T,
    hdm2: T,
    hdm3: T,
}

fn make_a16(x: &Inputs<'_>) -> Result<T, FerricError> {
    let (h1e, h2e, dm3) = (&x.h1e, &x.h2e, &x.dm3);
    let n = h1e.shape()[0];
    let mut a = -einsum("ib,rpqiac->pqrabc", &[h1e, dm3])?;
    a += &einsum("ia,rpqbic->pqrabc", &[h1e, dm3])?;
    a -= &einsum("ci,rpqbai->pqrabc", &[h1e, dm3])?;
    a -= &perm(&x.f3ca, &[1, 4, 0, 2, 5, 3]);
    a -= &einsum("kbia,rpqcki->pqrabc", &[h2e, dm3])?;
    a -= &einsum("kbaj,rpqjkc->pqrabc", &[h2e, dm3])?;
    a += &einsum("cbij,rpqjai->pqrabc", &[h2e, dm3])?;
    let fdm2 = einsum("kbij,rpajki->prab", &[h2e, dm3])?;
    a += &einsum("prab,qc->pqrabc", &[&fdm2, &eye(n)])?;
    a += &perm(&x.f3ac, &[1, 2, 0, 4, 3, 5]);
    a -= &perm(&x.f3ca, &[1, 2, 0, 4, 3, 5]);
    a += &einsum("jbij,rpqiac->pqrabc", &[h2e, dm3])?;
    a -= &einsum("cjka,rpqbjk->pqrabc", &[h2e, dm3])?;
    a += &einsum("jcij,rpqbai->pqrabc", &[h2e, dm3])?;
    Ok(a)
}

fn make_a22(x: &Inputs<'_>) -> Result<T, FerricError> {
    let (h1e, h2e, dm2, dm3) = (&x.h1e, &x.h2e, &x.dm2, &x.dm3);
    let n = h1e.shape()[0];
    let d = eye(n);
    let mut a = -einsum("pb,kipjac->ijkabc", &[h1e, dm3])?;
    a -= &einsum("pa,kibjpc->ijkabc", &[h1e, dm3])?;
    a += &einsum("cp,kibjap->ijkabc", &[h1e, dm3])?;
    a += &einsum("cqra,kibjqr->ijkabc", &[h2e, dm3])?;
    a -= &einsum("qcpq,kibjap->ijkabc", &[h2e, dm3])?;
    a -= &perm(&x.f3ac, &[1, 5, 0, 2, 4, 3]);
    let fdm2 = einsum("pqrb,kiqcpr->ikbc", &[h2e, dm3])?;
    a -= &einsum("ikbc,ja->ijkabc", &[&fdm2, &d])?;
    a -= &einsum("pqab,kiqjpc->ijkabc", &[h2e, dm3])?;
    a += &einsum("pcrb,kiajpr->ijkabc", &[h2e, dm3])?;
    a += &einsum("cqrb,kiqjar->ijkabc", &[h2e, dm3])?;
    a -= &perm(&x.f3ac, &[1, 3, 0, 4, 2, 5]);
    a += &perm(&x.f3ca, &[1, 3, 0, 4, 2, 5]);
    a += &(2.0 * &einsum("jb,kiac->ijkabc", &[h1e, dm2])?);
    a += &(2.0 * &einsum("pjrb,kiprac->ijkabc", &[h2e, dm3])?);
    let mut f = einsum("pa,kipc->ikac", &[h1e, dm2])?;
    f -= &einsum("cp,kiap->ikac", &[h1e, dm2])?;
    f -= &einsum("cqra,kiqr->ikac", &[h2e, dm2])?;
    f += &einsum("qcpq,kiap->ikac", &[h2e, dm2])?;
    f += &einsum("pqra,kiqcpr->ikac", &[h2e, dm3])?;
    f -= &einsum("rcpq,kiaqrp->ikac", &[h2e, dm3])?;
    a += &(2.0 * &einsum("ikac,jb->ijkabc", &[&f, &d])?);
    Ok(a)
}

/// `h1e − Σ_j h2e[m,j,j,n]` (the shift PySCF applies in `make_a17/a19`).
fn h1e_shifted(h1e: &T, h2e: &T) -> Result<T, FerricError> {
    Ok(h1e - &einsum("mjjn->mn", &[h2e])?)
}

fn make_a17(x: &Inputs<'_>) -> Result<T, FerricError> {
    let h = h1e_shifted(&x.h1e, &x.h2e)?;
    Ok(-einsum("pi,cabi->abcp", &[&h, &x.dm2])? - &einsum("kpij,cabjki->abcp", &[&x.h2e, &x.dm3])?)
}

fn make_a19(x: &Inputs<'_>) -> Result<T, FerricError> {
    let h = h1e_shifted(&x.h1e, &x.h2e)?;
    Ok(-einsum("pi,ai->ap", &[&h, &x.dm1])? - &einsum("kpij,ajki->ap", &[&x.h2e, &x.dm2])?)
}

fn make_a23(x: &Inputs<'_>) -> Result<T, FerricError> {
    let (h1e, h2e) = (&x.h1e, &x.h2e);
    let mut a = -einsum("ip,caib->abcp", &[h1e, &x.dm2])?;
    a -= &einsum("pijk,cajbik->abcp", &[h2e, &x.dm3])?;
    a += &(2.0 * &einsum("bp,ca->abcp", &[h1e, &x.dm1])?);
    a += &(2.0 * &einsum("pibk,caik->abcp", &[h2e, &x.dm2])?);
    Ok(a)
}

fn make_a25(x: &Inputs<'_>) -> Result<T, FerricError> {
    let (h1e, h2e) = (&x.h1e, &x.h2e);
    let mut a = -einsum("pi,ai->ap", &[h1e, &x.dm1])?;
    a -= &einsum("pijk,jaik->ap", &[h2e, &x.dm2])?;
    a += &(2.0 * &einsum("ap->pa", &[h1e])?);
    a += &(2.0 * &einsum("piaj,ij->ap", &[h2e, &x.dm1])?);
    Ok(a)
}

fn make_a3(x: &Inputs<'_>) -> Result<T, FerricError> {
    let (h1e, h2e) = (&x.h1e, &x.h2e);
    let d = eye(h1e.shape()[0]);
    let mut a = einsum("ia,ip->pa", &[h1e, &x.hdm1])?;
    a += &(2.0 * &einsum("ijka,pj,ik->pa", &[h2e, &d, &x.dm1])?);
    a -= &einsum("ijka,jpik->pa", &[h2e, &x.dm2])?;
    Ok(a)
}

fn make_k27(x: &Inputs<'_>) -> Result<T, FerricError> {
    let (h1e, h2e) = (&x.h1e, &x.h2e);
    let mut k = -einsum("ai,pi->pa", &[h1e, &x.dm1])?;
    k -= &einsum("iajk,pkij->pa", &[h2e, &x.dm2])?;
    k += &einsum("iaji,pj->pa", &[h2e, &x.dm1])?;
    Ok(k)
}

/// `(rm2, a7)`: the normal-ordered 2-RDM in PySCF's `make_a7` layout and the
/// class-rs Koopmans matrix.
fn make_a7(x: &Inputs<'_>) -> Result<(T, T), FerricError> {
    let (h1e, h2e) = (&x.h1e, &x.h2e);
    let d = eye(h1e.shape()[0]);
    let rm2 = einsum("iljk->ijkl", &[&x.dm2])? - &einsum("ik,jl->ijkl", &[&x.dm1, &d])?;
    let mut rm3 = einsum("injmkl->ijklmn", &[&x.dm3])?;
    rm3 -= &einsum("jn,imkl->ijklmn", &[&d, &x.dm2])?;
    rm3 -= &einsum("km,ijln->ijklmn", &[&d, &rm2])?;
    rm3 -= &einsum("kn,ijml->ijklmn", &[&d, &rm2])?;
    let mut a7 = -einsum("bi,pqia->pqab", &[h1e, &rm2])?;
    a7 -= &einsum("ai,pqbi->pqab", &[h1e, &rm2])?;
    a7 -= &einsum("kbij,pqkija->pqab", &[h2e, &rm3])?;
    a7 -= &einsum("kaij,pqkibj->pqab", &[h2e, &rm3])?;
    a7 -= &einsum("baij,pqij->pqab", &[h2e, &rm2])?;
    Ok((rm2, a7))
}

fn make_a9(x: &Inputs<'_>) -> Result<T, FerricError> {
    let (h1e, h2e, hdm2, hdm3) = (&x.h1e, &x.h2e, &x.hdm2, &x.hdm3);
    let mut a = einsum("ib,pqai->pqab", &[h1e, hdm2])?;
    a += &(2.0 * &einsum("ijib,pqaj->pqab", &[h2e, hdm2])?);
    a -= &einsum("ijjb,pqai->pqab", &[h2e, hdm2])?;
    a -= &einsum("ijkb,pkqaij->pqab", &[h2e, hdm3])?;
    a += &einsum("ia,pqib->pqab", &[h1e, hdm2])?;
    a -= &einsum("ijja,pqib->pqab", &[h2e, hdm2])?;
    a -= &einsum("ijba,pqji->pqab", &[h2e, hdm2])?;
    a += &(2.0 * &einsum("ijia,pqjb->pqab", &[h2e, hdm2])?);
    a -= &einsum("ijka,pqkjbi->pqab", &[h2e, hdm3])?;
    Ok(a)
}

fn make_a12(x: &Inputs<'_>) -> Result<T, FerricError> {
    let (h1e, h2e, dm2, dm3) = (&x.h1e, &x.h2e, &x.dm2, &x.dm3);
    let mut a = einsum("ia,qpib->pqab", &[h1e, dm2])?;
    a -= &einsum("bi,qpai->pqab", &[h1e, dm2])?;
    a += &einsum("ijka,qpjbik->pqab", &[h2e, dm3])?;
    a -= &einsum("kbij,qpajki->pqab", &[h2e, dm3])?;
    a -= &einsum("bjka,qpjk->pqab", &[h2e, dm2])?;
    a += &einsum("jbij,qpai->pqab", &[h2e, dm2])?;
    Ok(a)
}

fn make_a13(x: &Inputs<'_>) -> Result<T, FerricError> {
    let (h1e, h2e, dm1, dm2, dm3) = (&x.h1e, &x.h2e, &x.dm1, &x.dm2, &x.dm3);
    let d = eye(h1e.shape()[0]);
    let mut a = -einsum("ia,qbip->pqab", &[h1e, dm2])?;
    a += &(2.0 * &einsum("pa,qb->pqab", &[h1e, dm1])?);
    a += &einsum("bi,qiap->pqab", &[h1e, dm2])?;
    a -= &(2.0 * &einsum("pa,bi,qi->pqab", &[&d, h1e, dm1])?);
    a -= &einsum("ijka,qbjpik->pqab", &[h2e, dm3])?;
    a += &einsum("kbij,qjapki->pqab", &[h2e, dm3])?;
    a += &einsum("blma,qmlp->pqab", &[h2e, dm2])?;
    a += &(2.0 * &einsum("kpma,qbkm->pqab", &[h2e, dm2])?);
    a -= &(2.0 * &einsum("bpma,qm->pqab", &[h2e, dm1])?);
    a -= &einsum("lbkl,qkap->pqab", &[h2e, dm2])?;
    a -= &(2.0 * &einsum("ap,mbkl,qlmk->pqab", &[&d, h2e, dm2])?);
    a += &(2.0 * &einsum("ap,lbkl,qk->pqab", &[&d, h2e, dm1])?);
    Ok(a)
}

/// The MO integral blocks of the external classes, in PySCF's layouts.
struct Blocks {
    /// Sr: `[r,q,p,s] = (rp|qs)`, `r` virtual.
    v_sr: T,
    /// Sr: `F^I[r,p] − Σ_b v_sr[r,b,b,p]`.
    h_sr: T,
    /// Si: `[p,q,i,r] = (pi|qr)`, `i` core.
    v_si: T,
    /// Si: `F^I[p,i]`, `p` active, `i` core.
    h_si: T,
    /// Sijrs: `(ia|jb)`, `[i,a,j,b]`.
    cvcv: T,
    /// Sijr: `[r,p,i,j] = (jp|ir)`.
    v_sijr: T,
    /// Srsi: `[r,s,i,p] = (sp|ir)`.
    v_srsi: T,
    /// Srs: `[r,s,p,q] = (rp|sq)`.
    v_srs: T,
    /// Sij: `[p,q,i,j] = (ip|jq)`.
    v_sij: T,
    /// Sir: `[r,p,i,q] = (ri|pq)`.
    v1_sir: T,
    /// Sir: `[r,q,p,i] = (rp|iq)`.
    v2_sir: T,
    /// Sir: `F^I[r,i]`.
    h_sir: T,
}

fn sr(x: &Inputs<'_>, b: &Blocks) -> Result<(f64, f64), FerricError> {
    let (v, h) = (&b.v_sr, &b.h_sr);
    let a16 = make_a16(x)?;
    let a17 = make_a17(x)?;
    let a19 = make_a19(x)?;
    let ener = einsum("ipqr,pqrabc,iabc->i", &[v, &a16, v])?
        + &(2.0 * &einsum("ipqr,pqra,ia->i", &[v, &a17, h])?)
        + &einsum("ip,pa,ia->i", &[h, &a19, h])?;
    let norm = einsum("ipqr,rpqbac,iabc->i", &[v, &x.dm3, v])?
        + &(2.0 * &einsum("ipqr,rpqa,ia->i", &[v, &x.dm2, h])?)
        + &einsum("ip,pa,ia->i", &[h, &x.dm1, h])?;
    Ok(norm_to_energy(
        (0..x.nv).map(|r| (norm[[r]], ener[[r]], x.eps_v[r])),
    ))
}

fn si(x: &Inputs<'_>, b: &Blocks) -> Result<(f64, f64), FerricError> {
    let (v, h) = (&b.v_si, &b.h_si);
    let n = x.h1e.shape()[0];
    let d = eye(n);
    let a22 = make_a22(x)?;
    let a23 = make_a23(x)?;
    let a25 = make_a25(x)?;
    let dm3_h =
        2.0 * &einsum("abef,cd->abcdef", &[&x.dm2, &d])? - &perm(&x.dm3, &[0, 1, 3, 2, 4, 5]);
    let dm2_h = 2.0 * &einsum("ab,cd->abcd", &[&x.dm1, &d])? - &perm(&x.dm2, &[0, 1, 3, 2]);
    let dm1_h = 2.0 * &d - &perm(&x.dm1, &[1, 0]);
    let ener = einsum("qpir,pqrabc,baic->i", &[v, &a22, v])?
        + &(2.0 * &einsum("qpir,pqra,ai->i", &[v, &a23, h])?)
        + &einsum("pi,pa,ai->i", &[h, &a25, h])?;
    let norm = einsum("qpir,rpqbac,baic->i", &[v, &dm3_h, v])?
        + &(2.0 * &einsum("qpir,rpqa,ai->i", &[v, &dm2_h, h])?)
        + &einsum("pi,pa,ai->i", &[h, &dm1_h, h])?;
    Ok(norm_to_energy(
        (0..x.nc).map(|i| (norm[[i]], ener[[i]], -x.eps_c[i])),
    ))
}

fn sijrs(x: &Inputs<'_>, b: &Blocks) -> (f64, f64) {
    let g = &b.cvcv;
    let (mut norm, mut e) = (0.0, 0.0);
    for i in 0..x.nc {
        for a in 0..x.nv {
            for j in 0..x.nc {
                for bb in 0..x.nv {
                    let v = g[[i, a, j, bb]];
                    let theta = 2.0 * v - g[[i, bb, j, a]];
                    norm += v * theta;
                    e += v * theta / (x.eps_c[i] + x.eps_c[j] - x.eps_v[a] - x.eps_v[bb]);
                }
            }
        }
    }
    (norm, e)
}

/// `2 ein(s1) − ein(s2)` symmetrized over the two pair axes `(ax1, ax2)` with
/// the pair diagonal halved (PySCF's `Sijr`/`Srsi` construction).
fn pair_symmetrized(s1: &str, s2: &str, v: &T, m: &T, swap: &[usize]) -> Result<T, FerricError> {
    let mut t = 2.0 * &einsum(s1, &[v, v, m])? - &einsum(s2, &[v, v, m])?;
    t += &perm(&t, swap);
    Ok(t)
}

fn sijr(x: &Inputs<'_>, b: &Blocks) -> Result<(f64, f64), FerricError> {
    let v = &b.v_sijr;
    let a3 = make_a3(x)?;
    let (s1, s2) = ("rpji,raji,pa->rji", "rpji,raij,pa->rji");
    let norm = pair_symmetrized(s1, s2, v, &x.hdm1, &[0, 2, 1])?;
    let h = pair_symmetrized(s1, s2, v, &a3, &[0, 2, 1])?;
    let mut labels = Vec::new();
    for r in 0..x.nv {
        for j in 0..x.nc {
            for i in j..x.nc {
                let f = if i == j { 0.5 } else { 1.0 };
                let diff = x.eps_v[r] - x.eps_c[j] - x.eps_c[i];
                labels.push((f * norm[[r, j, i]], f * h[[r, j, i]], diff));
            }
        }
    }
    Ok(norm_to_energy(labels.into_iter()))
}

fn srsi(x: &Inputs<'_>, b: &Blocks) -> Result<(f64, f64), FerricError> {
    let v = &b.v_srsi;
    let k27 = make_k27(x)?;
    let (s1, s2) = ("rsip,rsia,pa->rsi", "rsip,sria,pa->rsi");
    let norm = pair_symmetrized(s1, s2, v, &x.dm1, &[1, 0, 2])?;
    let h = pair_symmetrized(s1, s2, v, &k27, &[1, 0, 2])?;
    let mut labels = Vec::new();
    for r in 0..x.nv {
        for s_ in r..x.nv {
            let f = if r == s_ { 0.5 } else { 1.0 };
            for i in 0..x.nc {
                let diff = x.eps_v[r] + x.eps_v[s_] - x.eps_c[i];
                labels.push((f * norm[[r, s_, i]], f * h[[r, s_, i]], diff));
            }
        }
    }
    Ok(norm_to_energy(labels.into_iter()))
}

fn srs(x: &Inputs<'_>, b: &Blocks) -> Result<(f64, f64), FerricError> {
    let v = &b.v_srs;
    let (rm2, a7) = make_a7(x)?;
    let norm = 0.5 * &einsum("rsqp,rsba,pqba->rs", &[v, v, &rm2])?;
    let h = 0.5 * &einsum("rsqp,rsba,pqab->rs", &[v, v, &a7])?;
    let mut labels = Vec::new();
    for r in 0..x.nv {
        for s_ in 0..x.nv {
            labels.push((norm[[r, s_]], h[[r, s_]], x.eps_v[r] + x.eps_v[s_]));
        }
    }
    Ok(norm_to_energy(labels.into_iter()))
}

fn sij(x: &Inputs<'_>, b: &Blocks) -> Result<(f64, f64), FerricError> {
    let v = &b.v_sij;
    let a9 = make_a9(x)?;
    let norm = 0.5 * &einsum("qpij,baij,pqab->ij", &[v, v, &x.hdm2])?;
    let h = 0.5 * &einsum("qpij,baij,pqab->ij", &[v, v, &a9])?;
    let mut labels = Vec::new();
    for i in 0..x.nc {
        for j in 0..x.nc {
            labels.push((norm[[i, j]], h[[i, j]], -(x.eps_c[i] + x.eps_c[j])));
        }
    }
    Ok(norm_to_energy(labels.into_iter()))
}

fn sir(x: &Inputs<'_>, b: &Blocks) -> Result<(f64, f64), FerricError> {
    let (v1, v2, h1) = (&b.v1_sir, &b.v2_sir, &b.h_sir);
    let (dm1, dm2) = (&x.dm1, &x.dm2);
    let mut norm = 2.0 * &einsum("rpiq,raib,qpab->ir", &[v1, v1, dm2])?;
    norm -= &einsum("rpiq,rabi,qpab->ir", &[v1, v2, dm2])?;
    norm -= &einsum("rpqi,raib,qpab->ir", &[v2, v1, dm2])?;
    norm += &(2.0 * &einsum("raqi,rabi,qb->ir", &[v2, v2, dm1])?);
    norm -= &einsum("rpqi,rabi,qbap->ir", &[v2, v2, dm2])?;
    norm += &einsum("rpqi,raai,qp->ir", &[v2, v2, dm1])?;
    norm += &(4.0 * &einsum("rpiq,ri,qp->ir", &[v1, h1, dm1])?);
    norm -= &(2.0 * &einsum("rpqi,ri,qp->ir", &[v2, h1, dm1])?);
    norm += &(2.0 * &einsum("ri,ri->ir", &[h1, h1])?);
    let a12 = make_a12(x)?;
    let a13 = make_a13(x)?;
    let mut h = 2.0 * &einsum("rpiq,raib,pqab->ir", &[v1, v1, &a12])?;
    h -= &einsum("rpiq,rabi,pqab->ir", &[v1, v2, &a12])?;
    h -= &einsum("rpqi,raib,pqab->ir", &[v2, v1, &a12])?;
    h += &einsum("rpqi,rabi,pqab->ir", &[v2, v2, &a13])?;
    let mut labels = Vec::new();
    for i in 0..x.nc {
        for r in 0..x.nv {
            labels.push((norm[[i, r]], h[[i, r]], x.eps_v[r] - x.eps_c[i]));
        }
    }
    Ok(norm_to_energy(labels.into_iter()))
}

/// The eight class `(norm, energy)` pairs. A class whose external space is
/// empty is exactly zero.
fn classes(x: &Inputs<'_>, b: &Blocks) -> Result<(Nevpt2Classes, Nevpt2Classes), FerricError> {
    let (has_c, has_v) = (x.nc > 0, x.nv > 0);
    let zero = (0.0, 0.0);
    let r = if has_v { sr(x, b)? } else { zero };
    let i = if has_c { si(x, b)? } else { zero };
    let ijrs = if has_c && has_v { sijrs(x, b) } else { zero };
    let ijr = if has_c && has_v { sijr(x, b)? } else { zero };
    let rsi = if has_c && has_v { srsi(x, b)? } else { zero };
    let rs = if has_v { srs(x, b)? } else { zero };
    let ij = if has_c { sij(x, b)? } else { zero };
    let ir = if has_c && has_v { sir(x, b)? } else { zero };
    let pick = |k: usize| Nevpt2Classes {
        sr: [r.0, r.1][k],
        si: [i.0, i.1][k],
        sijrs: [ijrs.0, ijrs.1][k],
        sijr: [ijr.0, ijr.1][k],
        srsi: [rsi.0, rsi.1][k],
        srs: [rs.0, rs.1][k],
        sij: [ij.0, ij.1][k],
        sir: [ir.0, ir.1][k],
    };
    Ok((pick(0), pick(1)))
}

/// The 8-fold-symmetric orderings of `(C1 C2|C3 C4)`, as permutations of
/// the requested axes.
const ERI_ORDERINGS: [[usize; 4]; 8] = [
    [0, 1, 2, 3],
    [1, 0, 2, 3],
    [0, 1, 3, 2],
    [1, 0, 3, 2],
    [2, 3, 0, 1],
    [3, 2, 0, 1],
    [2, 3, 1, 0],
    [3, 2, 1, 0],
];

/// The ordering that contracts the narrowest block first (it is passed
/// last to [`transform_eri_mixed`]), and the transient elements of that
/// transform's first two quarter outputs.
fn eri_ordering(widths: [usize; 4], nbas: usize) -> ([usize; 4], usize) {
    let mut best = (ERI_ORDERINGS[0], usize::MAX);
    for o in ERI_ORDERINGS {
        let (w4, w3) = (widths[o[3]], widths[o[2]]);
        let peak = w4
            .saturating_mul(nbas.saturating_pow(3))
            .saturating_add(w4.saturating_mul(w3).saturating_mul(nbas * nbas));
        if peak < best.1 {
            best = (o, peak);
        }
    }
    best
}

/// `(C1 C2|C3 C4)` as an `[n1, n2, n3, n4]` array.
fn mo_block(eri_ao: &[f64], nbas: usize, c: [ArrayView2<'_, f64>; 4]) -> Result<T, FerricError> {
    let widths = [c[0].ncols(), c[1].ncols(), c[2].ncols(), c[3].ncols()];
    if widths.contains(&0) {
        return Ok(ArrayD::zeros(IxDyn(&widths)));
    }
    let (o, _) = eri_ordering(widths, nbas);
    let flat = transform_eri_mixed(eri_ao, nbas, [c[o[0]], c[o[1]], c[o[2]], c[o[3]]])?;
    let shape: Vec<usize> = o.iter().map(|&k| widths[k]).collect();
    let t = ArrayD::from_shape_vec(IxDyn(&shape), flat)
        .map_err(|e| FerricError::General(format!("NEVPT2 MO block shape: {e}")))?;
    // Axis k of `t` is requested axis o[k]; requested axis j is axis inv[j] of `t`.
    let mut inv = [0usize; 4];
    for (k, &j) in o.iter().enumerate() {
        inv[j] = k;
    }
    Ok(perm(&t, &inv))
}

fn validate(
    mol: &Molecule,
    nbas: usize,
    reference: &Nevpt2Reference<'_>,
    config: &Nevpt2Config,
) -> Result<(), FerricError> {
    let err = |m: String| Err(FerricError::General(format!("NEVPT2: {m}")));
    let (nc, na) = (reference.n_core, reference.n_active);
    let (ea, eb) = reference.n_elec_active;
    let c = reference.mo_coeff;
    if ea != eb {
        return err(format!(
            "open-shell active spaces (n_alpha = {ea}, n_beta = {eb}) are not supported"
        ));
    }
    if na == 0 || na > strings::MAX_ACTIVE_ORBITALS || ea > na {
        return err(format!(
            "invalid active space (n_active = {na}, n_elec = ({ea}, {eb}))"
        ));
    }
    if c.nrows() != nbas || nc + na > c.ncols() {
        return err(format!(
            "orbitals have shape {:?}; need {nbas} rows and at least n_core + n_active = {} columns",
            c.dim(),
            nc + na
        ));
    }
    let nelec = mol.nelec() as usize;
    if 2 * nc + ea + eb != nelec {
        return err(format!(
            "2*n_core({nc}) + active({ea}+{eb}) != molecule nelec {nelec}"
        ));
    }
    let ndet = strings::count_strings(na, ea)
        .and_then(|x| strings::count_strings(na, eb).map(|y| x.saturating_mul(y)))
        .unwrap_or(usize::MAX);
    if reference.ci_vector.len() != ndet {
        return err(format!(
            "CI vector has length {} but the active space has {ndet} determinants",
            reference.ci_vector.len()
        ));
    }
    let norm: f64 = reference
        .ci_vector
        .iter()
        .map(|x| x * x)
        .sum::<f64>()
        .sqrt();
    if (norm - 1.0).abs() > 1e-10 {
        return err(format!("CI vector norm {norm} is not 1"));
    }
    if !(config.max_ci_residual.is_finite() && config.max_ci_residual > 0.0) {
        return err(format!(
            "max_ci_residual must be a positive finite number, got {}",
            config.max_ci_residual
        ));
    }
    Ok(())
}

/// Elements of the resident and transient stages of [`sc_nevpt2`], charged
/// to `plan`.
fn reserve(plan: &mut MemoryPlan, nbas: usize, nc: usize, na: usize, nv: usize, ndet: usize) {
    let n2 = na * na;
    let n4 = n2 * n2;
    let n6 = n4 * n2;
    plan.reserve("dense AO ERIs", nbas.saturating_pow(4), Lifetime::Resident);
    plan.reserve(
        "hcore, J, K, F^I, F^A, F (AO and MO), orbitals",
        nbas.saturating_mul(nbas).saturating_mul(12),
        Lifetime::Resident,
    );
    let blocks: [[usize; 4]; 10] = [
        [na, na, na, na],
        [nv, na, na, na],
        [na, nc, na, na],
        [nc, nv, nc, nv],
        [nc, na, nc, nv],
        [nv, na, nc, nv],
        [nv, na, nv, na],
        [nc, na, nc, na],
        [nv, nc, na, na],
        [nv, na, nc, na],
    ];
    let mut peak = 0usize;
    let mut resident = 0usize;
    for w in blocks {
        peak = peak.max(eri_ordering(w, nbas).1);
        resident = resident.saturating_add(w.iter().product::<usize>());
    }
    plan.reserve(
        "MO transform chain (first two quarter outputs)",
        peak,
        Lifetime::Transient,
    );
    // Each block is held in its transformed and its permuted layout.
    plan.reserve(
        "MO integral blocks",
        resident.saturating_mul(2),
        Lifetime::Resident,
    );
    plan.reserve(
        "CI residual: sigma vector",
        ndet.saturating_mul(2),
        Lifetime::Resident,
    );
    // dm3, f3ca, f3ac, G3, hdm3, dm3_h, rm3, a16 or a22, and up to four
    // einsum results or permuted copies alive at once.
    plan.reserve(
        "n^6 active intermediates",
        n6.saturating_mul(12),
        Lifetime::Resident,
    );
    // Per-label tensors of Sijr, Srsi, Sir (two einsums and the symmetrized
    // copy of each of norm and K).
    let ext = nv
        .saturating_mul(nc)
        .saturating_mul(nc.max(nv))
        .saturating_add(nv.saturating_mul(nv));
    plan.reserve(
        "per-label norms and K",
        ext.saturating_mul(8),
        Lifetime::Transient,
    );
}

/// SC-NEVPT2 on `reference`.
///
/// The Hamiltonian is the molecule's: `h = T + V_nuc + V_ECP` (the ECP of
/// `prep`'s basis; apply it to `mol` with `Molecule::apply_ecp` first, as
/// the CASSCF run did) and the exact AO ERIs. The reference CI vector must
/// be an eigenvector of the active Hamiltonian within
/// [`Nevpt2Config::max_ci_residual`]; open-shell active spaces
/// (`n_alpha != n_beta`) are refused.
pub fn sc_nevpt2(
    mol: &Molecule,
    prep: &PreparedBasis,
    reference: &Nevpt2Reference<'_>,
    config: &Nevpt2Config,
) -> Result<Nevpt2Result, FerricError> {
    let nbas = prep.nbasis();
    validate(mol, nbas, reference, config)?;
    let c = reference.mo_coeff;
    let (nc, na) = (reference.n_core, reference.n_active);
    let nmo = c.ncols();
    let nv = nmo - nc - na;
    let n_elec = reference.n_elec_active;
    let ci = reference.ci_vector;

    let mut plan = MemoryPlan::from_global_pool(
        config.memory_budget_bytes,
        format!("SC-NEVPT2 (nbas={nbas}, n_core={nc}, n_active={na}, n_virtual={nv})"),
    );
    reserve(&mut plan, nbas, nc, na, nv, ci.len());
    let _mem = plan.commit()?;
    // The RDM builders get what this plan leaves of the same ceiling.
    let rest = Some(plan.budget_bytes().saturating_sub(plan.peak_bytes()));

    let h_ao = hcore_ecp(prep, mol, prep.basis_set());
    let eri_ao = dense_ao_eri(prep)?;

    // F^I and E_core.
    let (fi_ao, e_core) = if nc == 0 {
        (h_ao.clone(), mol.nuclear_repulsion())
    } else {
        let cc = c.slice(s![.., 0..nc]);
        let dc = 2.0 * cc.dot(&cc.t());
        let (j, k) = dense_jk(&eri_ao, nbas, &dc);
        let fi = &h_ao + &j - &(0.5 * &k);
        let e = mol.nuclear_repulsion() + 0.5 * (&dc * &(&h_ao + &fi)).sum();
        (fi, e)
    };
    let ca = c.slice(s![.., nc..nc + na]);
    let ints = ActiveSpaceIntegrals {
        n_active: na,
        h: ca.t().dot(&fi_ao).dot(&ca),
        eri: transform_eri_mixed(&eri_ao, nbas, [ca, ca, ca, ca])?,
        e_core,
    };

    // The reference must be an eigenvector of the active Hamiltonian.
    let space = DeterminantSpace {
        alpha_strings: strings::enumerate_strings(na, n_elec.0),
        beta_strings: strings::enumerate_strings(na, n_elec.1),
    };
    // The sigma vector carries `e_core` on the diagonal, so `c . Hc` is the
    // total reference energy.
    let sigma = DirectSigma::new(&ints, &space).apply(ci);
    let e_ref: f64 = ci.iter().zip(&sigma).map(|(a, b)| a * b).sum();
    let residual = ci
        .iter()
        .zip(&sigma)
        .map(|(a, b)| (b - e_ref * a).powi(2))
        .sum::<f64>()
        .sqrt();
    drop(sigma);
    if !(residual <= config.max_ci_residual) {
        return Err(FerricError::General(format!(
            "NEVPT2: the reference CI vector has residual |Hc - Ec| = {residual:.2e} above \
             max_ci_residual {:e}; it is not an eigenvector of the active Hamiltonian",
            config.max_ci_residual
        )));
    }

    let rd = nevpt_rdms(&ints, n_elec, ci, rest)?;

    // Generalized Fock and the canonical core and virtual orbitals.
    let da = ca.dot(&rd.dm1).dot(&ca.t());
    let (j, k) = dense_jk(&eri_ao, nbas, &da);
    let f_ao = &fi_ao + &j - &(0.5 * &k);
    let f_mo = c.t().dot(&f_ao).dot(c);
    let mut u = Array2::<f64>::eye(nmo);
    for (lo, hi) in [(0, nc), (nc + na, nmo)] {
        if hi <= lo {
            continue;
        }
        let blk = f_mo.slice(s![lo..hi, lo..hi]);
        if config.canonicalize {
            let sym = 0.5 * (&blk + &blk.t());
            let (_, v) = sym
                .eigh(UPLO::Lower)
                .map_err(|e| FerricError::Lapack(format!("NEVPT2 canonicalize: {e}")))?;
            u.slice_mut(s![lo..hi, lo..hi]).assign(&v);
        } else {
            let mut off = 0.0f64;
            for ((p, q), &x) in blk.indexed_iter() {
                if p != q {
                    off = off.max(x.abs());
                }
            }
            if off > CANONICAL_TOL {
                return Err(FerricError::General(format!(
                    "NEVPT2: canonicalize is off but the {} block of F has an off-diagonal \
                     element {off:.2e} above {CANONICAL_TOL:e}",
                    if lo == 0 { "core" } else { "virtual" }
                )));
            }
        }
    }
    let cn = c.dot(&u);
    let fock = cn.t().dot(&f_ao).dot(&cn);
    let mo_energy: Vec<f64> = fock.diag().to_vec();
    let fi = cn.t().dot(&fi_ao).dot(&cn);

    let (cc, cv) = (cn.slice(s![.., 0..nc]), cn.slice(s![.., nc + na..]));
    let ca = cn.slice(s![.., nc..nc + na]);
    let blk = |b: [ArrayView2<'_, f64>; 4]| mo_block(&eri_ao, nbas, b);
    let mat = |r: std::ops::Range<usize>, q: std::ops::Range<usize>| {
        fi.slice(s![r, q]).to_owned().into_dyn()
    };
    let (ra, rc, rv) = (nc..nc + na, 0..nc, nc + na..nmo);
    let aaaa = ArrayD::from_shape_vec(IxDyn(&[na; 4]), ints.eri.clone())
        .map_err(|e| FerricError::General(format!("NEVPT2 (aa|aa) shape: {e}")))?;
    let h2e = perm(&aaaa, &[0, 2, 1, 3]);
    let v_sr = perm(&blk([cv, ca, ca, ca])?, &[0, 2, 1, 3]);
    let h_sr = mat(rv.clone(), ra.clone()) - &einsum("mbbn->mn", &[&v_sr])?;
    let b = Blocks {
        h_sr,
        v_sr,
        v_si: perm(&blk([ca, cc, ca, ca])?, &[0, 2, 1, 3]),
        h_si: mat(ra.clone(), rc.clone()),
        cvcv: blk([cc, cv, cc, cv])?,
        v_sijr: perm(&blk([cc, ca, cc, cv])?, &[3, 1, 2, 0]),
        v_srsi: perm(&blk([cv, ca, cc, cv])?, &[3, 0, 2, 1]),
        v_srs: perm(&blk([cv, ca, cv, ca])?, &[0, 2, 1, 3]),
        v_sij: perm(&blk([cc, ca, cc, ca])?, &[1, 3, 0, 2]),
        v1_sir: perm(&blk([cv, cc, ca, ca])?, &[0, 2, 1, 3]),
        v2_sir: perm(&blk([cv, ca, cc, ca])?, &[0, 3, 1, 2]),
        h_sir: mat(rv, rc),
    };
    drop(eri_ao);

    let (g2, g3) = e_product_rdms(&rd.dm1, &rd.dm2, &rd.dm3);
    let hole = hole_rdms(&rd.dm1, &rd.dm2, &rd.dm3, rest)?;
    let x = Inputs {
        nc,
        nv,
        eps_c: &mo_energy[0..nc],
        eps_v: &mo_energy[nc + na..],
        h1e: ints.h.clone().into_dyn(),
        h2e,
        dm1: rd.dm1.into_dyn(),
        dm2: g2.into_dyn(),
        dm3: g3.into_dyn(),
        f3ca: rd.f3ca.into_dyn(),
        f3ac: rd.f3ac.into_dyn(),
        hdm1: hole.hdm1.into_dyn(),
        hdm2: hole.hdm2.into_dyn(),
        hdm3: hole.hdm3.into_dyn(),
    };
    let (norms, energies) = classes(&x, &b)?;
    Ok(Nevpt2Result {
        e_corr: energies.total(),
        energies,
        norms,
        e_reference: e_ref,
        ci_residual_norm: residual,
        mo_coeff: cn,
        mo_energy,
        n_core: nc,
        n_active: na,
        n_virtual: nv,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arr(shape: &[usize], seed: u64) -> T {
        let mut st = seed;
        let n: usize = shape.iter().product();
        let v = (0..n)
            .map(|_| {
                st = st
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((st >> 11) as f64 / (1u64 << 53) as f64) - 0.5
            })
            .collect();
        ArrayD::from_shape_vec(IxDyn(shape), v).unwrap()
    }

    /// `einsum` against explicit loops: a GEMM, a three-operand contraction
    /// with a trace inside one operand, a transpose and a diagonal. Bar:
    /// sums of at most 5^3 products of numbers below 1/2, rounding below
    /// 1e-14.
    #[test]
    fn einsum_matches_explicit_loops() {
        let a = arr(&[3, 4], 1);
        let b = arr(&[4, 5], 2);
        let ab = einsum("ij,jk->ik", &[&a, &b]).unwrap();
        for i in 0..3 {
            for k in 0..5 {
                let want: f64 = (0..4).map(|j| a[[i, j]] * b[[j, k]]).sum();
                assert!((ab[[i, k]] - want).abs() < 1e-14);
            }
        }
        let g = arr(&[5, 5, 5], 3);
        let h = arr(&[5, 3], 4);
        let w = einsum(
            "pqq,pk,ki->ip",
            &[&g, &h, &a.slice(s![.., 0..3]).to_owned().into_dyn()],
        );
        let w = w.unwrap();
        for i in 0..3 {
            for p in 0..5 {
                let mut want = 0.0;
                for q in 0..5 {
                    for k in 0..3 {
                        want += g[[p, q, q]] * h[[p, k]] * a[[k, i]];
                    }
                }
                assert!((w[[i, p]] - want).abs() < 1e-14);
            }
        }
        let t = einsum("ij->ji", &[&a]).unwrap();
        assert_eq!(t[[2, 1]], a[[1, 2]]);
        let sq = arr(&[4, 4], 5);
        let d = einsum("ii->i", &[&sq]).unwrap();
        assert_eq!(d[[3]], sq[[3, 3]]);
        assert!(einsum("ij,jk->ik", &[&a, &a]).is_err());
        assert!(einsum("ij->ik", &[&a]).is_err());
    }

    /// The ordering chosen for a block is one of the 8 ERI symmetries and
    /// puts the narrowest width last.
    #[test]
    fn eri_ordering_contracts_the_narrowest_block_first() {
        let (o, _) = eri_ordering([9, 2, 7, 5], 10);
        assert_eq!(o[3], 1);
        let (o, _) = eri_ordering([3, 9, 9, 9], 10);
        assert_eq!(o[3], 0);
    }
}
