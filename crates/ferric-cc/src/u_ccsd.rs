//! Spin-blocked unrestricted CCSD (UCCSD) on a UHF reference.
//!
//! The amplitudes are the five spin blocks of the spin-orbital `t1`/`t2`:
//! `t1a[i,a]`, `t1b[I,A]`, `t2aa[i,j,a,b]`, `t2ab[i,J,a,B]`, `t2bb[I,J,A,B]`
//! (lower case alpha, upper case beta). `t2aa` and `t2bb` are antisymmetric in
//! `(i,j)` and in `(a,b)`; `t2ab[i,J,a,B]` is the spin-orbital amplitude
//! `t[i alpha, J beta, a alpha, B beta]`. These are PySCF's `uccsd`
//! conventions, and the residual equations transcribe
//! `pyscf/cc/uccsd.py::update_amps` with the canonical-HF simplification
//! (`f_ov = 0`, diagonal `f_oo`/`f_vv`), matching the restricted solvers.
//! Derivation: `book/src/16-coupled-cluster/08-uccsd.md#uccsd-equations`.
//!
//! Integrals are chemist blocks `(pq|rs) = sum_P B^P_pq B^P_rs` from per-spin
//! dressed RI 3-index blocks
//! ([`URiBlocks`](crate::u_ccsd::URiBlocks)), so the method is DF-UCCSD on
//! the auxiliary basis passed in. The `vvvv`, `vvVV` and `VVVV` blocks are held
//! dense: the open-shell solver has no integral-direct ladder.
//!
//! The independent check is the spin-orbital solver of [`crate::ccsd`] run on
//! the same UHF orbitals ([`crate::ccsd::ccsd_uhf_spin_orbital`]): its tensors
//! span `(no_a + no_b)` occupied and `(nv_a + nv_b)` virtual spin orbitals and
//! it never uses spin blocks.

use crate::CcConfig;
use ferric_core::memory::plan::{Lifetime, MemoryPlan};
use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::mo_transform::{dress_3index, transform_3center_ov};
use ferric_mp2::rimp2::{active_occ, cholesky_inverse_sqrt};
use ferric_scf::result::Spin;
use ferric_scf::ScfResult;
use ferric_tensors::{einsum_dyn, permute_to_owned};
use ndarray::{Array2, Array3, Array4, ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// einsum with a free output order
// ---------------------------------------------------------------------------

/// `numpy.einsum` for two or three operands with any output order.
///
/// Each binary step calls [`einsum_dyn`] with the engine's natural output
/// order (batch, left-free, right-free) and the result is permuted to the
/// requested order at the end. An index carried by a later operand or by the
/// output is kept through a step; an index of one operand that nothing else
/// carries is a single-operand sum, which this helper refuses (panics), as
/// the engine does.
pub(crate) fn ein(spec: &str, ops: &[&ArrayD<f64>]) -> ArrayD<f64> {
    let spec: String = spec.chars().filter(|c| !c.is_whitespace()).collect();
    let (ins, out) = spec.split_once("->").expect("einsum spec needs '->'");
    let terms: Vec<Vec<char>> = ins.split(',').map(|t| t.chars().collect()).collect();
    assert_eq!(terms.len(), ops.len(), "einsum {spec}: operand count");
    assert!(ops.len() >= 2, "einsum {spec}: needs two operands");
    let out: Vec<char> = out.chars().collect();
    let mut acc_term = terms[0].clone();
    let mut acc = ops[0].clone();
    for k in 1..ops.len() {
        let right = &terms[k];
        // indices still needed after this step
        let mut keep: Vec<char> = out.clone();
        for t in &terms[k + 1..] {
            keep.extend(t.iter().copied());
        }
        let both = |c: &char| acc_term.contains(c) && right.contains(c);
        let batch: Vec<char> = acc_term
            .iter()
            .copied()
            .filter(|c| both(c) && keep.contains(c))
            .collect();
        let lfree: Vec<char> = acc_term
            .iter()
            .copied()
            .filter(|c| !right.contains(c))
            .collect();
        let rfree: Vec<char> = right
            .iter()
            .copied()
            .filter(|c| !acc_term.contains(c))
            .collect();
        for c in lfree.iter().chain(&rfree) {
            assert!(
                keep.contains(c),
                "einsum {spec}: index {c} is summed over one operand only"
            );
        }
        let natural: Vec<char> = batch.iter().chain(&lfree).chain(&rfree).copied().collect();
        let step: String = format!(
            "{},{}->{}",
            acc_term.iter().collect::<String>(),
            right.iter().collect::<String>(),
            natural.iter().collect::<String>()
        );
        acc = einsum_dyn(&step, acc.view(), ops[k].view(), 1.0)
            .unwrap_or_else(|e| panic!("einsum {spec} (step {step}): {e}"));
        acc_term = natural;
    }
    if acc_term == out {
        return acc;
    }
    let axes: Vec<usize> = out
        .iter()
        .map(|c| acc_term.iter().position(|x| x == c).expect("output index"))
        .collect();
    permute_to_owned(acc.view().permuted_axes(IxDyn(&axes)))
}

/// Scalar `numpy.einsum` (`"ijab,ijab->"`).
pub(crate) fn ein0(spec: &str, ops: &[&ArrayD<f64>]) -> f64 {
    let r = ein(spec, ops);
    *r.iter().next().unwrap_or(&0.0)
}

/// `x.transpose(axes)` as an owned standard-layout array.
pub(crate) fn tr(x: &ArrayD<f64>, axes: &[usize]) -> ArrayD<f64> {
    permute_to_owned(x.view().permuted_axes(IxDyn(axes)))
}

// ---------------------------------------------------------------------------
// per-spin RI blocks
// ---------------------------------------------------------------------------

/// Orbital spaces and dressed RI 3-index blocks of a UHF reference, per spin
/// (index 0 = alpha, 1 = beta).
///
/// `b_ov[s][P,i,a]`, `b_oo[s][P,i,j]` and `b_vv[s][P,a,b]` are
/// `sum_Q (V^-1/2)_PQ (Q|pq)` over the active occupied orbitals
/// (`frozen_core..nocc[s]`) and all virtual orbitals of spin `s`. `eo[s]` and
/// `ev[s]` are the matching orbital energies.
pub struct URiBlocks {
    /// Active occupied orbitals per spin.
    pub no: [usize; 2],
    /// Virtual orbitals per spin.
    pub nv: [usize; 2],
    /// Active occupied orbital energies per spin.
    pub eo: [Vec<f64>; 2],
    /// Virtual orbital energies per spin.
    pub ev: [Vec<f64>; 2],
    /// `B^P_ia` per spin.
    pub b_ov: [Array3<f64>; 2],
    /// `B^P_ij` per spin.
    pub b_oo: [Array3<f64>; 2],
    /// `B^P_ab` per spin.
    pub b_vv: [Array3<f64>; 2],
}

/// Occupied counts `(nocc_a, nocc_b)` and retained MO counts `(nmo_a, nmo_b)`
/// of a UHF reference for `mol`.
///
/// `nocc` comes from the electron count and multiplicity, as the UHF solver
/// filled it. `nmo` drops the trailing all-zero columns the SCF pads in when
/// it removes near-linear-dependent overlap modes.
///
/// # Errors
///
/// A reference that is not `Spin::Unrestricted`, an unconverged SCF, an
/// impossible electron count, or more occupied orbitals than retained MOs.
pub fn uhf_counts(
    mol: &Molecule,
    scf: &ScfResult,
) -> Result<([usize; 2], [usize; 2]), FerricError> {
    match scf.spin {
        Spin::Unrestricted => {}
        Spin::RestrictedOpen => {
            return Err(FerricError::General(
                "UCCSD needs a UHF reference: a ROHF result has no per-spin orbital energies \
                 (semi-canonicalize first: ferric_scf::semicanonical)"
                    .into(),
            ))
        }
        Spin::Restricted => {
            return Err(FerricError::General(
                "UCCSD needs a UHF reference; for a restricted reference use \
                 ferric_cc::ccsd_closed_shell"
                    .into(),
            ))
        }
    }
    if !scf.converged {
        return Err(FerricError::ScfConvergence {
            iterations: scf.iterations,
            last_energy: scf.energy,
        });
    }
    let nelec = i64::from(mol.nelec());
    let two_s = mol.multiplicity as i64 - 1;
    if nelec < two_s || (nelec + two_s) % 2 != 0 {
        return Err(FerricError::General(format!(
            "UCCSD: {nelec} electrons cannot have multiplicity {}",
            mol.multiplicity
        )));
    }
    let nocc = [
        ((nelec + two_s) / 2) as usize,
        ((nelec - two_s) / 2) as usize,
    ];
    let mos = [scf.mos_a(), scf.mos_b()];
    let mut nmo = [0usize; 2];
    for s in 0..2 {
        let c = mos[s];
        let is_zero = |p: usize| c.column(p).iter().all(|&x| x == 0.0);
        nmo[s] = (0..c.ncols()).take_while(|&p| !is_zero(p)).count();
        if nocc[s] > nmo[s] {
            return Err(FerricError::General(format!(
                "UCCSD: {} occupied orbitals of spin {s} but only {} retained MOs",
                nocc[s], nmo[s]
            )));
        }
    }
    Ok((nocc, nmo))
}

impl URiBlocks {
    /// Build the per-spin blocks for `scf` (UHF) with `frozen_core` frozen
    /// orbitals in each spin.
    ///
    /// # Errors
    ///
    /// See [`uhf_counts`]; also a frozen core larger than either occupied
    /// count, and integral failures.
    pub fn build(
        mol: &Molecule,
        obs: &PreparedBasis,
        dfbs: &PreparedBasis,
        op: Operator,
        scf: &ScfResult,
        frozen_core: usize,
    ) -> Result<Self, FerricError> {
        let (nocc, nmo) = uhf_counts(mol, scf)?;
        let no = [
            active_occ(nocc[0], frozen_core)?,
            active_occ(nocc[1], frozen_core)?,
        ];
        let nv = [nmo[0] - nocc[0], nmo[1] - nocc[1]];
        let mos = [scf.mos_a(), scf.mos_b()];
        let eps = [scf.eps_a(), scf.eps_b()];
        let v2c = ferric_integrals::threeindex::coulomb_metric_2c(op, dfbs)?;
        let v_inv_sqrt = cholesky_inverse_sqrt(&v2c)?;
        let eri3_ao = ferric_integrals::threeindex::eri3_tensor(op, obs, dfbs)?;
        let mut b_ov = Vec::with_capacity(2);
        let mut b_oo = Vec::with_capacity(2);
        let mut b_vv = Vec::with_capacity(2);
        let mut eo = Vec::with_capacity(2);
        let mut ev = Vec::with_capacity(2);
        for s in 0..2 {
            let c = mos[s];
            let c_occ = c.slice(ndarray::s![.., frozen_core..nocc[s]]).to_owned();
            let c_vir = c.slice(ndarray::s![.., nocc[s]..nmo[s]]).to_owned();
            let dress = |c1: &Array2<f64>, c2: &Array2<f64>| {
                dress_3index(&transform_3center_ov(&eri3_ao, c1, c2), &v_inv_sqrt)
            };
            b_ov.push(dress(&c_occ, &c_vir));
            b_oo.push(dress(&c_occ, &c_occ));
            b_vv.push(dress(&c_vir, &c_vir));
            eo.push(eps[s][frozen_core..nocc[s]].to_vec());
            ev.push(eps[s][nocc[s]..nmo[s]].to_vec());
        }
        let pair = |mut v: Vec<Array3<f64>>| {
            let b = v.pop().expect("beta");
            let a = v.pop().expect("alpha");
            [a, b]
        };
        let [eo_a, eo_b]: [Vec<f64>; 2] = eo.try_into().expect("two spins");
        let [ev_a, ev_b]: [Vec<f64>; 2] = ev.try_into().expect("two spins");
        Ok(Self {
            no,
            nv,
            eo: [eo_a, eo_b],
            ev: [ev_a, ev_b],
            b_ov: pair(b_ov),
            b_oo: pair(b_oo),
            b_vv: pair(b_vv),
        })
    }

    /// `B^P_ai` (virtual-occupied) of spin `s`.
    pub fn b_vo(&self, s: usize) -> Array3<f64> {
        self.b_ov[s]
            .view()
            .permuted_axes([0, 2, 1])
            .as_standard_layout()
            .into_owned()
    }
}

/// Chemist block `(pq|rs) = sum_P x[P,p,q] y[P,r,s]`, indexed `[p,q,r,s]`.
pub(crate) fn chem(x: &Array3<f64>, y: &Array3<f64>) -> ArrayD<f64> {
    let (naux, p, q) = x.dim();
    let (_, r, s) = y.dim();
    let xm = x
        .view()
        .into_shape_with_order((naux, p * q))
        .expect("standard layout")
        .to_owned();
    let ym = y
        .view()
        .into_shape_with_order((naux, r * s))
        .expect("standard layout")
        .to_owned();
    let m = xm.t().dot(&ym);
    m.into_shape_with_order(IxDyn(&[p, q, r, s]))
        .expect("chemist reshape")
}

// ---------------------------------------------------------------------------
// integrals in PySCF's `uccsd._ChemistsERIs` layout
// ---------------------------------------------------------------------------

/// The chemist integral blocks the UCCSD equations read (lower case = alpha,
/// upper case = beta; e.g. `ov_vv[i,a,B,C] = (ia|BC)`).
struct UEris {
    ovov: ArrayD<f64>,
    ov_ov: ArrayD<f64>,
    ovov_b: ArrayD<f64>,
    ovoo: ArrayD<f64>,
    ovoo_b: ArrayD<f64>,
    ov_oo: ArrayD<f64>,
    ov_oo_ba: ArrayD<f64>,
    oooo: ArrayD<f64>,
    oooo_b: ArrayD<f64>,
    oo_oo: ArrayD<f64>,
    oovv: ArrayD<f64>,
    oovv_b: ArrayD<f64>,
    oo_vv: ArrayD<f64>,
    oo_vv_ba: ArrayD<f64>,
    ovvo: ArrayD<f64>,
    ovvo_b: ArrayD<f64>,
    ov_vo: ArrayD<f64>,
    ov_vo_ba: ArrayD<f64>,
    ovvv: ArrayD<f64>,
    ovvv_b: ArrayD<f64>,
    ov_vv: ArrayD<f64>,
    ov_vv_ba: ArrayD<f64>,
    vvvv: ArrayD<f64>,
    vvvv_b: ArrayD<f64>,
    vv_vv: ArrayD<f64>,
}

impl UEris {
    fn build(r: &URiBlocks) -> Self {
        let (a, b) = (0, 1);
        let vo = [r.b_vo(a), r.b_vo(b)];
        let ov = &r.b_ov;
        let oo = &r.b_oo;
        let vv = &r.b_vv;
        Self {
            ovov: chem(&ov[a], &ov[a]),
            ov_ov: chem(&ov[a], &ov[b]),
            ovov_b: chem(&ov[b], &ov[b]),
            ovoo: chem(&ov[a], &oo[a]),
            ovoo_b: chem(&ov[b], &oo[b]),
            ov_oo: chem(&ov[a], &oo[b]),
            ov_oo_ba: chem(&ov[b], &oo[a]),
            oooo: chem(&oo[a], &oo[a]),
            oooo_b: chem(&oo[b], &oo[b]),
            oo_oo: chem(&oo[a], &oo[b]),
            oovv: chem(&oo[a], &vv[a]),
            oovv_b: chem(&oo[b], &vv[b]),
            oo_vv: chem(&oo[a], &vv[b]),
            oo_vv_ba: chem(&oo[b], &vv[a]),
            ovvo: chem(&ov[a], &vo[a]),
            ovvo_b: chem(&ov[b], &vo[b]),
            ov_vo: chem(&ov[a], &vo[b]),
            ov_vo_ba: chem(&ov[b], &vo[a]),
            ovvv: chem(&ov[a], &vv[a]),
            ovvv_b: chem(&ov[b], &vv[b]),
            ov_vv: chem(&ov[a], &vv[b]),
            ov_vv_ba: chem(&ov[b], &vv[a]),
            vvvv: chem(&vv[a], &vv[a]),
            vvvv_b: chem(&vv[b], &vv[b]),
            vv_vv: chem(&vv[a], &vv[b]),
        }
    }
}

// ---------------------------------------------------------------------------
// amplitudes
// ---------------------------------------------------------------------------

/// The five UCCSD amplitude blocks (see the module docs for the layout).
#[derive(Debug, Clone)]
pub struct UAmplitudes {
    /// `t1[i alpha, a alpha]`.
    pub t1a: Array2<f64>,
    /// `t1[I beta, A beta]`.
    pub t1b: Array2<f64>,
    /// `t2[i alpha, j alpha, a alpha, b alpha]`, antisymmetric.
    pub t2aa: Array4<f64>,
    /// `t2[i alpha, J beta, a alpha, B beta]`.
    pub t2ab: Array4<f64>,
    /// `t2[I beta, J beta, A beta, B beta]`, antisymmetric.
    pub t2bb: Array4<f64>,
}

/// Converged UCCSD result.
#[derive(Debug, Clone)]
#[must_use]
pub struct UCcsdResult {
    /// Total correlation energy `e_aa + e_ab + e_bb` (Hartree).
    pub correlation_energy: f64,
    /// Same-spin alpha-alpha part: `1/4 <ij||ab> t2aa + 1/2 <ij||ab> t1a t1a`.
    pub e_aa: f64,
    /// Opposite-spin part: `(ia|JB) (t2ab + t1a t1b)`.
    pub e_ab: f64,
    /// Same-spin beta-beta part.
    pub e_bb: f64,
    /// Converged amplitudes.
    pub amplitudes: UAmplitudes,
    /// Amplitude iterations taken.
    pub iterations: usize,
}

/// `tau = t2 + fac/2 * P(ij)P(ab) t1 t1` for a same-spin block
/// (PySCF `make_tau_aa` with `r1 = t1`).
fn make_tau_aa(t2: &ArrayD<f64>, t1: &ArrayD<f64>, fac: f64) -> ArrayD<f64> {
    let x = ein("ia,jb->ijab", &[t1, t1]);
    let mut y = &x - &tr(&x, &[1, 0, 2, 3]);
    y = &y - &tr(&y, &[0, 1, 3, 2]);
    t2 + &(fac * 0.5 * y)
}

/// `tau_ab = t2ab + fac * t1a t1b` (PySCF `make_tau_ab` with `r1 = t1`).
fn make_tau_ab(t2ab: &ArrayD<f64>, t1a: &ArrayD<f64>, t1b: &ArrayD<f64>, fac: f64) -> ArrayD<f64> {
    t2ab + &(fac * ein("ia,jb->ijab", &[t1a, t1b]))
}

/// Antisymmetrize over the occupied pair and the virtual pair:
/// `x - x(ijba) - x(jiab) + x(jiba)`.
fn asym_oovv(x: &ArrayD<f64>) -> ArrayD<f64> {
    let y = x - &tr(x, &[0, 1, 3, 2]);
    &y - &tr(&y, &[1, 0, 2, 3])
}

struct Amps {
    t1a: ArrayD<f64>,
    t1b: ArrayD<f64>,
    t2aa: ArrayD<f64>,
    t2ab: ArrayD<f64>,
    t2bb: ArrayD<f64>,
}

/// Denominators per block.
struct Denoms {
    d1a: ArrayD<f64>,
    d1b: ArrayD<f64>,
    d2aa: ArrayD<f64>,
    d2ab: ArrayD<f64>,
    d2bb: ArrayD<f64>,
}

impl Denoms {
    fn new(r: &URiBlocks) -> Self {
        let dia = |s: usize| {
            let (no, nv) = (r.no[s], r.nv[s]);
            let mut d = ArrayD::zeros(IxDyn(&[no, nv]));
            for i in 0..no {
                for a in 0..nv {
                    d[[i, a]] = r.eo[s][i] - r.ev[s][a];
                }
            }
            d
        };
        let d1a = dia(0);
        let d1b = dia(1);
        let d2 = |x: &ArrayD<f64>, y: &ArrayD<f64>| {
            let (no1, nv1) = (x.shape()[0], x.shape()[1]);
            let (no2, nv2) = (y.shape()[0], y.shape()[1]);
            let mut d = ArrayD::zeros(IxDyn(&[no1, no2, nv1, nv2]));
            for i in 0..no1 {
                for j in 0..no2 {
                    for a in 0..nv1 {
                        for b in 0..nv2 {
                            d[[i, j, a, b]] = x[[i, a]] + y[[j, b]];
                        }
                    }
                }
            }
            d
        };
        Self {
            d2aa: d2(&d1a, &d1a),
            d2ab: d2(&d1a, &d1b),
            d2bb: d2(&d1b, &d1b),
            d1a,
            d1b,
        }
    }
}

/// UCCSD energy components `(e_aa, e_ab, e_bb)` (PySCF `uccsd.energy`, `f_ov = 0`).
fn energy(t: &Amps, e: &UEris) -> (f64, f64, f64) {
    let same = |t1: &ArrayD<f64>, t2: &ArrayD<f64>, ovov: &ArrayD<f64>| {
        0.25 * ein0("ijab,iajb->", &[t2, ovov]) - 0.25 * ein0("ijab,ibja->", &[t2, ovov])
            + 0.5 * ein0("ia,jb,iajb->", &[t1, t1, ovov])
            - 0.5 * ein0("ia,jb,ibja->", &[t1, t1, ovov])
    };
    let e_aa = same(&t.t1a, &t.t2aa, &e.ovov);
    let e_bb = same(&t.t1b, &t.t2bb, &e.ovov_b);
    let e_ab = ein0("iJaB,iaJB->", &[&t.t2ab, &e.ov_ov])
        + ein0("ia,JB,iaJB->", &[&t.t1a, &t.t1b, &e.ov_ov]);
    (e_aa, e_ab, e_bb)
}

/// One Jacobi update of the UCCSD amplitudes: the right-hand sides of
/// PySCF `uccsd.update_amps` divided by the orbital-energy denominators.
#[allow(clippy::too_many_lines)]
fn update_amps(t: &Amps, e: &UEris, d: &Denoms) -> Amps {
    let (t1a, t1b, t2aa, t2ab, t2bb) = (&t.t1a, &t.t1b, &t.t2aa, &t.t2ab, &t.t2bb);
    let (noa, nva) = (t1a.shape()[0], t1a.shape()[1]);
    let (nob, nvb) = (t1b.shape()[0], t1b.shape()[1]);
    let z = |s: &[usize]| ArrayD::<f64>::zeros(IxDyn(s));

    let tauaa = make_tau_aa(t2aa, t1a, 1.0);
    let taubb = make_tau_aa(t2bb, t1b, 1.0);
    let tauab = make_tau_ab(t2ab, t1a, t1b, 1.0);

    // particle-particle ladders: (ac|bd) tau_ijcd
    let mut u2aa = 0.5 * ein("ijcd,acbd->ijab", &[&tauaa, &e.vvvv]);
    let mut u2bb = 0.5 * ein("ijcd,acbd->ijab", &[&taubb, &e.vvvv_b]);
    let mut u2ab = ein("iJcD,acBD->iJaB", &[&tauab, &e.vv_vv]);

    let mut u1a = z(&[noa, nva]);
    let mut u1b = z(&[nob, nvb]);
    let mut fooa = z(&[noa, noa]);
    let mut foob = z(&[nob, nob]);
    let mut fvva = z(&[nva, nva]);
    let mut fvvb = z(&[nvb, nvb]);
    let mut wovvo = z(&[noa, nva, nva, noa]);
    let mut w_ovvo_b = z(&[nob, nvb, nvb, nob]);
    let mut wo_vv_o = z(&[noa, nvb, nva, nob]); // woVvO
    let mut wo_vvo_x = z(&[noa, nvb, nvb, noa]); // woVVo
    let mut w_ovv_o = z(&[nob, nva, nvb, noa]); // wOvVo
    let mut w_ovvo_x = z(&[nob, nva, nva, nob]); // wOvvO

    // --- (ov|vv) same spin ---
    {
        let ovvv = &e.ovvv - &tr(&e.ovvv, &[0, 3, 2, 1]);
        fvva += &ein("mf,mfae->ae", &[t1a, &ovvv]);
        wovvo += &ein("jf,mebf->mbej", &[t1a, &ovvv]);
        u1a += &(0.5 * ein("mief,meaf->ia", &[t2aa, &ovvv]));
        u2aa += &ein("ie,mbea->imab", &[t1a, &ovvv]);
        let tmp = ein("ijef,mebf->ijmb", &[&tauaa, &ovvv]);
        u2aa -= &(0.5 * ein("ijmb,ma->ijab", &[&tmp, t1a]));
    }
    {
        let ovvv = &e.ovvv_b - &tr(&e.ovvv_b, &[0, 3, 2, 1]);
        fvvb += &ein("mf,mfae->ae", &[t1b, &ovvv]);
        w_ovvo_b += &ein("jf,mebf->mbej", &[t1b, &ovvv]);
        u1b += &(0.5 * ein("mief,meaf->ia", &[t2bb, &ovvv]));
        u2bb += &ein("ie,mbea->imab", &[t1b, &ovvv]);
        let tmp = ein("ijef,mebf->ijmb", &[&taubb, &ovvv]);
        u2bb -= &(0.5 * ein("ijmb,ma->ijab", &[&tmp, t1b]));
    }
    // --- (ov|VV) and (OV|vv) ---
    {
        let x = &e.ov_vv;
        fvvb += &ein("mf,mfAE->AE", &[t1a, x]);
        wo_vv_o += &ein("JF,meBF->mBeJ", &[t1b, x]);
        wo_vvo_x -= &ein("jf,mfBE->mBEj", &[t1a, x]);
        u1b += &ein("mIeF,meAF->IA", &[t2ab, x]);
        u2ab += &ein("IE,maEB->mIaB", &[t1b, x]);
        let tmp = ein("iJeF,meBF->iJmB", &[&tauab, x]);
        u2ab -= &ein("iJmB,ma->iJaB", &[&tmp, t1a]);
    }
    {
        let x = &e.ov_vv_ba;
        fvva += &ein("MF,MFae->ae", &[t1b, x]);
        w_ovv_o += &ein("jf,MEbf->MbEj", &[t1a, x]);
        w_ovvo_x -= &ein("JF,MFbe->MbeJ", &[t1b, x]);
        u1a += &ein("iMfE,MEaf->ia", &[t2ab, x]);
        u2ab += &ein("ie,MBea->iMaB", &[t1a, x]);
        let tmp = ein("iJeF,MFbe->iJbM", &[&tauab, x]);
        u2ab -= &ein("iJbM,MA->iJbA", &[&tmp, t1b]);
    }

    // --- alpha (oo|oo), (ov|oo), (ov|ov) ---
    let fova;
    {
        let mut w = ein("je,nemi->mnij", &[t1a, &e.ovoo]);
        w = &w - &tr(&w, &[0, 1, 3, 2]);
        w += &tr(&e.oooo, &[0, 2, 1, 3]);
        w += &(0.5 * ein("ijef,menf->mnij", &[&tauaa, &e.ovov]));
        u2aa += &(0.5 * ein("mnab,mnij->ijab", &[&tauaa, &w]));
        let ovoo = &e.ovoo - &tr(&e.ovoo, &[2, 1, 0, 3]);
        fooa += &ein("ne,nemi->mi", &[t1a, &ovoo]);
        u1a += &(0.5 * ein("mnae,meni->ia", &[t2aa, &ovoo]));
        wovvo += &ein("nb,nemj->mbej", &[t1a, &ovoo]);

        let til = make_tau_aa(t2aa, t1a, 0.5);
        let ovov = &e.ovov - &tr(&e.ovov, &[0, 3, 2, 1]);
        fvva -= &(0.5 * ein("mnaf,menf->ae", &[&til, &ovov]));
        fooa += &(0.5 * ein("inef,menf->mi", &[&til, &ovov]));
        fova = ein("nf,menf->me", &[t1a, &ovov]);
        u2aa += &(0.5 * tr(&ovov, &[0, 2, 1, 3]));
        wovvo -= &(0.5 * ein("jnfb,menf->mbej", &[t2aa, &ovov]));
        wo_vv_o += &(0.5 * ein("nJfB,menf->mBeJ", &[t2ab, &ovov]));
        let tmp = ein("jf,menf->mnej", &[t1a, &ovov]);
        wovvo -= &ein("nb,mnej->mbej", &[t1a, &tmp]);
    }
    // --- beta ---
    let fovb;
    {
        let mut w = ein("je,nemi->mnij", &[t1b, &e.ovoo_b]);
        w = &w - &tr(&w, &[0, 1, 3, 2]);
        w += &tr(&e.oooo_b, &[0, 2, 1, 3]);
        w += &(0.5 * ein("ijef,menf->mnij", &[&taubb, &e.ovov_b]));
        u2bb += &(0.5 * ein("mnab,mnij->ijab", &[&taubb, &w]));
        let ovoo = &e.ovoo_b - &tr(&e.ovoo_b, &[2, 1, 0, 3]);
        foob += &ein("ne,nemi->mi", &[t1b, &ovoo]);
        u1b += &(0.5 * ein("mnae,meni->ia", &[t2bb, &ovoo]));
        w_ovvo_b += &ein("nb,nemj->mbej", &[t1b, &ovoo]);

        let til = make_tau_aa(t2bb, t1b, 0.5);
        let ovov = &e.ovov_b - &tr(&e.ovov_b, &[0, 3, 2, 1]);
        fvvb -= &(0.5 * ein("MNAF,MENF->AE", &[&til, &ovov]));
        foob += &(0.5 * ein("inef,menf->mi", &[&til, &ovov]));
        fovb = ein("nf,menf->me", &[t1b, &ovov]);
        u2bb += &(0.5 * tr(&ovov, &[0, 2, 1, 3]));
        w_ovvo_b -= &(0.5 * ein("jnfb,menf->mbej", &[t2bb, &ovov]));
        w_ovv_o += &(0.5 * ein("jNbF,MENF->MbEj", &[t2ab, &ovov]));
        let tmp = ein("jf,menf->mnej", &[t1b, &ovov]);
        w_ovvo_b -= &ein("nb,mnej->mbej", &[t1b, &tmp]);
    }
    let mut fova = fova;
    let mut fovb = fovb;

    // --- mixed (OV|oo), (ov|OO) ---
    {
        let ovoo_ba = &e.ov_oo_ba; // (IA|jk)
        let ovoo_ab = &e.ov_oo; // (ia|JK)
        fooa += &ein("NE,NEmi->mi", &[t1b, ovoo_ba]);
        u1a -= &ein("nMaE,MEni->ia", &[t2ab, ovoo_ba]);
        w_ovv_o -= &ein("nb,MEnj->MbEj", &[t1a, ovoo_ba]);
        wo_vvo_x += &ein("NB,NEmj->mBEj", &[t1b, ovoo_ba]);
        foob += &ein("ne,neMI->MI", &[t1a, ovoo_ab]);
        u1b -= &ein("mNeA,meNI->IA", &[t2ab, ovoo_ab]);
        wo_vv_o -= &ein("NB,meNJ->mBeJ", &[t1b, ovoo_ab]);
        w_ovvo_x += &ein("nb,neMJ->MbeJ", &[t1a, ovoo_ab]);
        let mut w = ein("JE,NEmi->mNiJ", &[t1b, ovoo_ba]);
        w += &ein("je,neMI->nMjI", &[t1a, ovoo_ab]);
        w += &tr(&e.oo_oo, &[0, 2, 1, 3]);
        w += &ein("iJeF,meNF->mNiJ", &[&tauab, &e.ov_ov]);
        u2ab += &ein("mNaB,mNiJ->iJaB", &[&tauab, &w]);
    }
    {
        let x = &e.ov_ov;
        let til = make_tau_ab(t2ab, t1a, t1b, 0.5);
        fvva -= &ein("mNaF,meNF->ae", &[&til, x]);
        fvvb -= &ein("nMfA,nfME->AE", &[&til, x]);
        fooa += &ein("iNeF,meNF->mi", &[&til, x]);
        foob += &ein("nIfE,nfME->MI", &[&til, x]);
        fova += &ein("NF,meNF->me", &[t1b, x]);
        fovb += &ein("nf,nfME->ME", &[t1a, x]);
        u2ab += &tr(x, &[0, 2, 1, 3]);
        wovvo += &(0.5 * ein("jNbF,meNF->mbej", &[t2ab, x]));
        w_ovvo_b += &(0.5 * ein("nJfB,nfME->MBEJ", &[t2ab, x]));
        w_ovv_o -= &(0.5 * ein("jnfb,nfME->MbEj", &[t2aa, x]));
        wo_vv_o -= &(0.5 * ein("JNFB,meNF->mBeJ", &[t2bb, x]));
        wo_vvo_x += &(0.5 * ein("jNfB,mfNE->mBEj", &[t2ab, x]));
        w_ovvo_x += &(0.5 * ein("nJbF,neMF->MbeJ", &[t2ab, x]));
        let tmpabab = ein("JF,meNF->mNeJ", &[t1b, x]);
        let tmpbaba = ein("jf,nfME->MnEj", &[t1a, x]);
        wo_vv_o -= &ein("NB,mNeJ->mBeJ", &[t1b, &tmpabab]);
        w_ovv_o -= &ein("nb,MnEj->MbEj", &[t1a, &tmpbaba]);
        wo_vvo_x += &ein("NB,NmEj->mBEj", &[t1b, &tmpbaba]);
        w_ovvo_x += &ein("nb,nMeJ->MbeJ", &[t1a, &tmpabab]);
    }

    // --- T1 from the F intermediates ---
    u1a += &ein("ie,ae->ia", &[t1a, &fvva]);
    u1a -= &ein("ma,mi->ia", &[t1a, &fooa]);
    u1a -= &ein("imea,me->ia", &[t2aa, &fova]);
    u1a += &ein("iMaE,ME->ia", &[t2ab, &fovb]);
    u1b += &ein("ie,ae->ia", &[t1b, &fvvb]);
    u1b -= &ein("ma,mi->ia", &[t1b, &foob]);
    u1b -= &ein("imea,me->ia", &[t2bb, &fovb]);
    u1b += &ein("mIeA,me->IA", &[t2ab, &fova]);

    // --- (oo|vv), (ov|vo) ---
    {
        wovvo -= &tr(&e.oovv, &[0, 2, 3, 1]);
        wovvo += &tr(&e.ovvo, &[0, 2, 1, 3]);
        let oovv = &e.oovv - &tr(&e.ovvo, &[0, 3, 2, 1]);
        u1a -= &ein("nf,niaf->ia", &[t1a, &oovv]);
        let tmp = ein("ie,mjbe->mbij", &[t1a, &oovv]);
        u2aa += &(2.0 * ein("ma,mbij->ijab", &[t1a, &tmp]));
    }
    {
        w_ovvo_b -= &tr(&e.oovv_b, &[0, 2, 3, 1]);
        w_ovvo_b += &tr(&e.ovvo_b, &[0, 2, 1, 3]);
        let oovv = &e.oovv_b - &tr(&e.ovvo_b, &[0, 3, 2, 1]);
        u1b -= &ein("nf,niaf->ia", &[t1b, &oovv]);
        let tmp = ein("ie,mjbe->mbij", &[t1b, &oovv]);
        u2bb += &(2.0 * ein("ma,mbij->ijab", &[t1b, &tmp]));
    }
    {
        let oo_vv = &e.oo_vv; // (ij|AB)
        let ov_vo = &e.ov_vo; // (ia|BJ)
        wo_vvo_x -= &tr(oo_vv, &[0, 2, 3, 1]);
        wo_vv_o += &tr(ov_vo, &[0, 2, 1, 3]);
        u1b += &ein("nf,nfAI->IA", &[t1a, ov_vo]);
        let mut tmp = ein("ie,meBJ->mBiJ", &[t1a, ov_vo]);
        tmp += &ein("IE,mjBE->mBjI", &[t1b, oo_vv]);
        u2ab -= &ein("ma,mBiJ->iJaB", &[t1a, &tmp]);
    }
    {
        let oo_vv = &e.oo_vv_ba; // (IJ|ab)
        let ov_vo = &e.ov_vo_ba; // (IA|bj)
        w_ovvo_x -= &tr(oo_vv, &[0, 2, 3, 1]);
        w_ovv_o += &tr(ov_vo, &[0, 2, 1, 3]);
        u1a += &ein("NF,NFai->ia", &[t1b, ov_vo]);
        let mut tmp = ein("IE,MEbj->MbIj", &[t1b, ov_vo]);
        tmp += &ein("ie,MJbe->MbJi", &[t1a, oo_vv]);
        u2ab -= &ein("MA,MbIj->jIbA", &[t1b, &tmp]);
    }

    // --- ring terms ---
    u2aa += &(2.0 * ein("imae,mbej->ijab", &[t2aa, &wovvo]));
    u2aa += &(2.0 * ein("iMaE,MbEj->ijab", &[t2ab, &w_ovv_o]));
    u2bb += &(2.0 * ein("imae,mbej->ijab", &[t2bb, &w_ovvo_b]));
    u2bb += &(2.0 * ein("mIeA,mBeJ->IJAB", &[t2ab, &wo_vv_o]));
    u2ab += &ein("imae,mBeJ->iJaB", &[t2aa, &wo_vv_o]);
    u2ab += &ein("iMaE,MBEJ->iJaB", &[t2ab, &w_ovvo_b]);
    u2ab += &ein("iMeA,MbeJ->iJbA", &[t2ab, &w_ovvo_x]);
    u2ab += &ein("IMAE,MbEj->jIbA", &[t2bb, &w_ovv_o]);
    u2ab += &ein("mIeA,mbej->jIbA", &[t2ab, &wovvo]);
    u2ab += &ein("mIaE,mBEj->jIaB", &[t2ab, &wo_vvo_x]);

    // --- F_vv and F_oo on T2 ---
    {
        let fa = &fvva - &(0.5 * ein("mb,me->be", &[t1a, &fova]));
        let fb = &fvvb - &(0.5 * ein("mb,me->be", &[t1b, &fovb]));
        u2aa += &ein("ijae,be->ijab", &[t2aa, &fa]);
        u2bb += &ein("ijae,be->ijab", &[t2bb, &fb]);
        u2ab += &ein("iJaE,BE->iJaB", &[t2ab, &fb]);
        u2ab += &ein("iJeA,be->iJbA", &[t2ab, &fa]);
        let fa = &fooa + &(0.5 * ein("je,me->mj", &[t1a, &fova]));
        let fb = &foob + &(0.5 * ein("je,me->mj", &[t1b, &fovb]));
        u2aa -= &ein("imab,mj->ijab", &[t2aa, &fa]);
        u2bb -= &ein("imab,mj->ijab", &[t2bb, &fb]);
        u2ab -= &ein("iMaB,MJ->iJaB", &[t2ab, &fb]);
        u2ab -= &ein("mIaB,mj->jIaB", &[t2ab, &fa]);
    }
    // --- t1 (ov|oo) ---
    {
        let ovoo = &e.ovoo - &tr(&e.ovoo, &[2, 1, 0, 3]);
        let ovoo_b = &e.ovoo_b - &tr(&e.ovoo_b, &[2, 1, 0, 3]);
        u2aa -= &ein("ma,jbim->ijab", &[t1a, &ovoo]);
        u2bb -= &ein("ma,jbim->ijab", &[t1b, &ovoo_b]);
        u2ab -= &ein("ma,JBim->iJaB", &[t1a, &e.ov_oo_ba]);
        u2ab -= &ein("MA,ibJM->iJbA", &[t1b, &e.ov_oo]);
    }

    let u2aa = asym_oovv(&(0.5 * u2aa));
    let u2bb = asym_oovv(&(0.5 * u2bb));
    Amps {
        t1a: u1a / &d.d1a,
        t1b: u1b / &d.d1b,
        t2aa: u2aa / &d.d2aa,
        t2ab: u2ab / &d.d2ab,
        t2bb: u2bb / &d.d2bb,
    }
}

impl Amps {
    fn blocks(&self) -> [&ArrayD<f64>; 5] {
        [&self.t1a, &self.t1b, &self.t2aa, &self.t2ab, &self.t2bb]
    }

    fn len(&self) -> usize {
        self.blocks().iter().map(|b| b.len()).sum()
    }

    fn to_column(&self) -> Array2<f64> {
        let mut v = Array2::<f64>::zeros((self.len(), 1));
        let mut k = 0;
        for b in self.blocks() {
            for x in b.iter() {
                v[[k, 0]] = *x;
                k += 1;
            }
        }
        v
    }

    fn unflatten_like(&self, v: &Array2<f64>) -> Amps {
        let mut k = 0;
        let mut take = |b: &ArrayD<f64>| {
            let mut out = ArrayD::<f64>::zeros(b.raw_dim());
            for slot in out.iter_mut() {
                *slot = v[[k, 0]];
                k += 1;
            }
            out
        };
        Amps {
            t1a: take(&self.t1a),
            t1b: take(&self.t1b),
            t2aa: take(&self.t2aa),
            t2ab: take(&self.t2ab),
            t2bb: take(&self.t2bb),
        }
    }

    fn max_abs_diff(&self, other: &Amps) -> f64 {
        self.blocks()
            .iter()
            .zip(other.blocks())
            .flat_map(|(a, b)| a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()))
            .fold(0.0, f64::max)
    }
}

/// The dimensions the UCCSD memory plan is a pure function of.
#[derive(Debug, Clone, Copy)]
pub struct UCcsdShape {
    /// Active occupied orbitals per spin.
    pub no: [usize; 2],
    /// Virtual orbitals per spin.
    pub nv: [usize; 2],
    /// AO basis functions.
    pub nbas: usize,
    /// Auxiliary basis functions.
    pub naux: usize,
    /// DIIS subspace depth.
    pub diis_subspace: usize,
}

/// The UCCSD [`MemoryPlan`] for `shape`.
///
/// Charged: the AO 3-index tensor (transient), the per-spin RI blocks, every
/// chemist integral block (the three `vv|vv` blocks dominate), eight copies
/// of the amplitude set (current, update, previous, intermediates of the same
/// shape) and the DIIS history.
pub fn u_ccsd_memory_plan(shape: UCcsdShape, budget_bytes: Option<usize>) -> MemoryPlan {
    let UCcsdShape {
        no,
        nv,
        nbas,
        naux,
        diis_subspace,
    } = shape;
    let [oa, ob] = no;
    let [va, vb] = nv;
    let mut plan = MemoryPlan::resolve(
        budget_bytes,
        format!("UCCSD (no_a={oa}, no_b={ob}, nv_a={va}, nv_b={vb})"),
    );
    let m = |xs: &[usize]| xs.iter().fold(1usize, |acc, &x| acc.saturating_mul(x));
    plan.reserve(
        "eri3_ao (P|mn)",
        m(&[naux, nbas, nbas]),
        Lifetime::Transient,
    );
    let nmo = [oa + va, ob + vb];
    plan.reserve(
        "per-spin RI blocks B_oo/B_ov/B_vv",
        m(&[naux, nmo[0], nmo[0]]).saturating_add(m(&[naux, nmo[1], nmo[1]])),
        Lifetime::Resident,
    );
    plan.reserve(
        "vvvv/vvVV/VVVV chemist blocks",
        m(&[va, va, va, va])
            .saturating_add(m(&[va, va, vb, vb]))
            .saturating_add(m(&[vb, vb, vb, vb])),
        Lifetime::Resident,
    );
    plan.reserve(
        "ovvv/ovVV/OVvv/OVVV chemist blocks (+ antisymmetrized copy)",
        m(&[oa, va, va, va])
            .saturating_add(m(&[oa, va, vb, vb]))
            .saturating_add(m(&[ob, vb, va, va]))
            .saturating_add(m(&[ob, vb, vb, vb]))
            .saturating_add(m(&[oa.max(ob), va.max(vb), va.max(vb), va.max(vb)])),
        Lifetime::Resident,
    );
    let ovov = m(&[oa + ob, va + vb, oa + ob, va + vb]);
    plan.reserve(
        "oooo/ovoo/oovv/ovvo/ovov chemist blocks",
        ovov.saturating_mul(4),
        Lifetime::Resident,
    );
    let amps = m(&[oa, oa, va, va])
        .saturating_add(m(&[oa, ob, va, vb]))
        .saturating_add(m(&[ob, ob, vb, vb]))
        .saturating_add(oa * va + ob * vb);
    plan.reserve(
        "amplitude working set (x8: t, update, previous, denominators, W intermediates)",
        amps.saturating_mul(8),
        Lifetime::Resident,
    );
    plan.reserve(
        "DIIS amplitude + error history (2 x diis_subspace)",
        crate::diis_history_elems(amps, diis_subspace),
        Lifetime::Resident,
    );
    plan
}

/// Spin-blocked RI-UCCSD correlation energy on a UHF reference.
///
/// `cfg.frozen_core` orbitals are frozen in EACH spin (the lowest
/// `frozen_core` alpha and the lowest `frozen_core` beta MOs), the convention
/// of the restricted solvers and of PySCF `UCCSD(frozen=n)`. The amplitudes
/// start from the MP2 guess; DIIS extrapolates all five blocks jointly.
/// Convergence is `|dE| < min(cfg.energy_conv, 1e-9)` Hartree.
///
/// # Errors
///
/// A non-UHF or unconverged reference ([`uhf_counts`]), a memory plan over
/// budget (before any integral is built), or `FerricError::Convergence`
/// after `cfg.max_iter` iterations.
pub fn u_ccsd(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    scf: &ScfResult,
    cfg: &CcConfig,
) -> Result<UCcsdResult, FerricError> {
    let (nocc, nmo) = uhf_counts(mol, scf)?;
    let no = [
        active_occ(nocc[0], cfg.frozen_core)?,
        active_occ(nocc[1], cfg.frozen_core)?,
    ];
    let nv = [nmo[0] - nocc[0], nmo[1] - nocc[1]];
    let mut plan = u_ccsd_memory_plan(
        UCcsdShape {
            no,
            nv,
            nbas: obs.nbasis(),
            naux: dfbs.nbasis(),
            diis_subspace: cfg.diis_subspace,
        },
        cfg.memory_budget_bytes,
    );
    if let Some(pool) = ferric_core::memory::pool::global() {
        plan = plan.with_pool(&pool);
    }
    // HARD: every block is dense with no streaming fallback.
    let _charge = plan.commit()?;

    let ri = URiBlocks::build(mol, obs, dfbs, op, scf, cfg.frozen_core)?;
    let eris = UEris::build(&ri);
    let den = Denoms::new(&ri);
    solve(&eris, &den, cfg)
}

fn solve(eris: &UEris, den: &Denoms, cfg: &CcConfig) -> Result<UCcsdResult, FerricError> {
    // MP2 guess: t1 = 0, t2 = <ij||ab> / D.
    let ovov_ijab = tr(&eris.ovov, &[0, 2, 1, 3]);
    let ovov_b_ijab = tr(&eris.ovov_b, &[0, 2, 1, 3]);
    let mut t = Amps {
        t1a: ArrayD::zeros(den.d1a.raw_dim()),
        t1b: ArrayD::zeros(den.d1b.raw_dim()),
        t2aa: (&ovov_ijab - &tr(&ovov_ijab, &[0, 1, 3, 2])) / &den.d2aa,
        t2ab: tr(&eris.ov_ov, &[0, 2, 1, 3]) / &den.d2ab,
        t2bb: (&ovov_b_ijab - &tr(&ovov_b_ijab, &[0, 1, 3, 2])) / &den.d2bb,
    };
    let mut diis = ferric_scf::diis::Diis::new(cfg.diis_subspace.max(1));
    let mut e_old = {
        let (a, b, c) = energy(&t, eris);
        a + b + c
    };
    let tol = cfg.energy_conv.min(1e-9);
    for iter in 0..cfg.max_iter {
        let t_new = update_amps(&t, eris, den);
        let v_new = t_new.to_column();
        let err = &v_new - &t.to_column();
        let ext = diis.step(&v_new, &err);
        let t_next = t.unflatten_like(&ext);
        let step = t_next.max_abs_diff(&t);
        t = t_next;
        let (e_aa, e_ab, e_bb) = energy(&t, eris);
        let e_corr = e_aa + e_ab + e_bb;
        if (e_corr - e_old).abs() < tol && iter > 0 {
            println!(
                "UCCSD converged in {} iterations. E_corr = {e_corr:.10} (last step {step:.1e})",
                iter + 1
            );
            let fix2 = |x: ArrayD<f64>| x.into_dimensionality::<ndarray::Ix2>().expect("2-D");
            let fix4 = |x: ArrayD<f64>| x.into_dimensionality::<ndarray::Ix4>().expect("4-D");
            return Ok(UCcsdResult {
                correlation_energy: e_corr,
                e_aa,
                e_ab,
                e_bb,
                amplitudes: UAmplitudes {
                    t1a: fix2(t.t1a),
                    t1b: fix2(t.t1b),
                    t2aa: fix4(t.t2aa),
                    t2ab: fix4(t.t2ab),
                    t2bb: fix4(t.t2bb),
                },
                iterations: iter + 1,
            });
        }
        e_old = e_corr;
    }
    Err(FerricError::Convergence(format!(
        "UCCSD did not converge in {} iterations",
        cfg.max_iter
    )))
}
