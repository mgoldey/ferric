//! EE-EOM-CCSD singlet excitation energies on a closed-shell RHF reference.
//!
//! The right eigenproblem `(H̄ − E_CC) R = ω R` with `H̄ = e^{−T} H e^{T}`
//! from converged spin-adapted CCSD amplitudes ([`crate::ccsd_closed_shell`]),
//! on the singlet singles+doubles space, solved by the non-symmetric block
//! Davidson of [`crate::eom_davidson`].
//!
//! # Space and vector
//!
//! A singlet `R` is `r1[i,a]` and `r2[i,j,a,b]` over spatial active
//! orbitals with `r2[i,j,a,b] = r2[j,i,b,a]` (the αβ doubles amplitude; the
//! same-spin block is `r2 − r2ᵀ_{ab}`). The Davidson vector packs `r1` and
//! the lower triangle of the symmetric `(ia, jb)` matrix of `r2`, the layout
//! of PySCF's `eom_rccsd.EOMEESinglet`. The doubles part of the product is
//! symmetric by construction, so the packed operator is the restriction of
//! `H̄ − E_CC` to the singlet space.
//!
//! # Product
//!
//! `σ = (H̄ − E_CC) R` is assembled from the spin-adapted EE intermediates of
//! PySCF `cc/eom_rccsd.py` (`_IMDS.make_ee`, `eeccsd_matvec_singlet`),
//! written here unblocked with the canonical-HF simplification (`f_ov = 0`,
//! diagonal `f_oo`, `f_vv`). The derivation check is the prototype
//! `scripts/prototypes/eom_ccsd_proto.py`: there the product equals the CC
//! Jacobian `∂Ω/∂t · R` of PySCF's RCCSD residual, taken by an exact
//! five-point difference (the residual is a quartic polynomial in `t`), a
//! construction that shares no code with PySCF's EOM module. At converged
//! amplitudes that Jacobian is `H̄ − E_CC` on singles+doubles.
//!
//! Integrals are the RI chemist blocks `(pq|rs) = Σ_P B^P_pq B^P_rs` of
//! [`crate::ccsd_closed_shell`], from the same auxiliary basis. The
//! particle-particle ladder `Σ_ef τ_ijef (ae|bf)` takes the dense `nv⁴`
//! block when its memory plan fits and otherwise rebuilds `(ae|bf)` from the
//! RI factors in `blk·nv³` panels each product; both routes are the same sum.
//!
//! # Anchors (tests)
//!
//! * `t = 0`: the singles-singles block is the CIS (TDA-HF) matrix, and its
//!   eigenvalues equal `ferric_gw::bse::run_cis_tda`'s on the same RI
//!   integrals.
//! * H2 / 6-31G: the dense packed `H̄ − E_CC` diagonalized directly equals
//!   the Davidson roots.
//! * PySCF `eomee_ccsd_singlet` on like-for-like DF integrals (water and
//!   C2H4), `scripts/validation/gen_eom_ccsd.py`.
//!
//! Right eigenvectors only: no left vectors, so no oscillator strengths or
//! EOM properties. Singlets only (no triplets, no spin flip). Closed-shell
//! RHF reference only.

use crate::ccsd_closed_shell::LadderRoute;
use crate::eom_davidson::{davidson_nonsymmetric, NonsymDavidsonSettings};
use crate::CcResult;
use ferric_core::memory::plan::{Lifetime, MemoryPlan};
use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::mo_transform::{transform_3center_oo, transform_3center_ov, transform_3center_vv};
use ferric_mp2::rimp2::{active_occ, cholesky_inverse_sqrt};
use ferric_mp2::spinorbital::build_b;
use ferric_scf::ScfResult;
use ferric_tensors::{einsum_dyn, permute_to_owned, Axis};
use ndarray::{s, Array1, Array2, Array4, ArrayD, Ix2, Ix4, IxDyn};

/// CCSD energy-change threshold the EOM drivers converge the amplitudes to
/// (`CcConfig::energy_conv`). An amplitude error `δt` moves ω at first order,
/// so the CCSD stage is run tighter than the energy-only default (1e-8).
pub const EOM_CCSD_ENERGY_CONV: f64 = 1e-10;

/// Controls of an EOM-CCSD singlet solve.
#[derive(Debug, Clone, PartialEq)]
pub struct EomCcsdConfig {
    /// Number of lowest singlet roots.
    pub n_roots: usize,
    /// Davidson residual-norm threshold (Hartree). The excitation-energy
    /// error is first order in it for this non-symmetric problem.
    pub conv_tol: f64,
    /// Davidson iteration cap; an unconverged root is an error.
    pub max_iter: usize,
    /// Davidson subspace size before a collapse (`None`: the solver default).
    pub max_space: Option<usize>,
    /// Resident-bytes ceiling for the EOM tensors (`None`: resolved by
    /// [`ferric_core::memory::resolve_budget_bytes`]).
    pub memory_budget_bytes: Option<usize>,
}

impl Default for EomCcsdConfig {
    fn default() -> Self {
        Self {
            n_roots: 5,
            conv_tol: 1e-6,
            max_iter: 100,
            max_space: None,
            memory_budget_bytes: None,
        }
    }
}

/// Converged EOM-CCSD singlet roots.
#[derive(Debug, Clone)]
pub struct EomCcsdResult {
    /// Excitation energies ω (Hartree), ascending.
    pub excitation_energies: Vec<f64>,
    /// Singles part `r1[i,a]` of each right eigenvector.
    pub r1: Vec<Array2<f64>>,
    /// Doubles part `r2[i,j,a,b]` (αβ amplitude) of each right eigenvector.
    pub r2: Vec<Array4<f64>>,
    /// Singles weight `2‖r1‖² / ‖R‖²` in the spin-orbital norm
    /// `‖R‖² = 2‖r1‖² + ‖r2‖² + ½‖r2 − r2ᵀ_{ab}‖²` (each vector is scaled
    /// to `‖R‖ = 1`).
    pub singles_weight: Vec<f64>,
    /// Final Davidson residual norm of each root (packed vector, unit norm).
    pub residuals: Vec<f64>,
    /// Davidson iterations.
    pub iterations: usize,
    /// Products `σ = H̄ R` evaluated.
    pub products: usize,
    /// The particle-particle ladder route the products took.
    pub ladder_route: LadderRoute,
}

/// The dimensions the EOM memory plan depends on.
#[derive(Debug, Clone, Copy)]
pub struct EomShape {
    /// Active occupied spatial orbitals.
    pub no: usize,
    /// Virtual spatial orbitals.
    pub nv: usize,
    /// AO basis functions.
    pub nbas: usize,
    /// Auxiliary basis functions.
    pub naux: usize,
    /// Roots requested.
    pub n_roots: usize,
    /// Davidson subspace columns held (vectors and products each).
    pub max_space: usize,
}

impl EomShape {
    /// Length of the packed singlet vector, `nov + nov(nov+1)/2`.
    pub fn dim(&self) -> usize {
        let nov = self.no * self.nv;
        nov + nov * (nov + 1) / 2
    }
}

/// The EOM-CCSD memory plan for `shape` and `route`.
///
/// Resident: the RI blocks, the chemist blocks (`ovvv` is the `no·nv³`
/// term), the dense `(ae|bf)` block on the dense route, the EE intermediates
/// (`W_vovv` is `no·nv³`), the amplitudes, and the Davidson vectors and
/// products (`2·max_space·dim`). Transient: the AO 3-index tensor, the
/// intermediate build (`3·no·nv³ + 6·no²nv²`) and one product's working set
/// (`10·no²nv²`, an `no³nv` panel, and the RI ladder panel `blk·nv³`).
pub fn eom_memory_plan(shape: EomShape, route: LadderRoute, budget_bytes: usize) -> MemoryPlan {
    let EomShape {
        no,
        nv,
        nbas,
        naux,
        n_roots: _,
        max_space,
    } = shape;
    let nv2 = nv.saturating_pow(2);
    let nv3 = nv.saturating_pow(3);
    let no2 = no.saturating_pow(2);
    let oovv = no2.saturating_mul(nv2);
    let label = match route {
        LadderRoute::Dense => format!("EOM-CCSD singlets (no={no}, nv={nv}; dense ladder)"),
        LadderRoute::Ri { block } => {
            format!("EOM-CCSD singlets (no={no}, nv={nv}; RI ladder, block {block})")
        }
    };
    let mut plan = MemoryPlan::with_budget_bytes(budget_bytes, label);
    plan.reserve(
        "eri3_ao (P|mn)",
        naux.saturating_mul(nbas).saturating_mul(nbas),
        Lifetime::Transient,
    );
    plan.reserve(
        "b_ov/b_oo/b_vv RI blocks",
        naux.saturating_mul(
            no.saturating_mul(nv)
                .saturating_add(no2)
                .saturating_add(nv2),
        ),
        Lifetime::Resident,
    );
    plan.reserve(
        "chemist blocks oooo/ovoo/ovov/oovv/ovvo/ovvv",
        no2.saturating_mul(no2)
            .saturating_add(no2.saturating_mul(no).saturating_mul(nv))
            .saturating_add(oovv.saturating_mul(3))
            .saturating_add(no.saturating_mul(nv3)),
        Lifetime::Resident,
    );
    plan.reserve(
        "EE intermediates (W_oooo, W_ovoo, W_ovvo x2, W_ooov, W_vovv) + t1/t2",
        no2.saturating_mul(no2)
            .saturating_add(no2.saturating_mul(no).saturating_mul(nv).saturating_mul(2))
            .saturating_add(oovv.saturating_mul(3))
            .saturating_add(no.saturating_mul(nv3)),
        Lifetime::Resident,
    );
    plan.reserve(
        "Davidson vectors and products (2 x max_space x dim)",
        shape.dim().saturating_mul(max_space).saturating_mul(2),
        Lifetime::Resident,
    );
    plan.reserve(
        "intermediate build scratch (3 no nv^3 + 6 no^2 nv^2)",
        no.saturating_mul(nv3)
            .saturating_mul(3)
            .saturating_add(oovv.saturating_mul(6)),
        Lifetime::Transient,
    );
    let ladder_panel = match route {
        LadderRoute::Dense => {
            plan.reserve(
                "(ae|bf) dense ladder block",
                nv.saturating_pow(4),
                Lifetime::Resident,
            );
            0
        }
        LadderRoute::Ri { block } => block.saturating_mul(nv3),
    };
    plan.reserve(
        "one product: 10 no^2 nv^2 + no^3 nv panel + ladder panel",
        oovv.saturating_mul(10)
            .saturating_add(no2.saturating_mul(no).saturating_mul(nv))
            .saturating_add(ladder_panel),
        Lifetime::Transient,
    );
    plan
}

/// Dense when it fits, otherwise the widest RI ladder block that fits;
/// an error naming the plan when even block 1 does not.
fn choose_route(shape: EomShape, budget: usize) -> Result<(LadderRoute, MemoryPlan), FerricError> {
    let dense = eom_memory_plan(shape, LadderRoute::Dense, budget);
    if dense.check().is_ok() {
        return Ok((LadderRoute::Dense, dense));
    }
    let fits = |block| {
        eom_memory_plan(shape, LadderRoute::Ri { block }, budget)
            .check()
            .is_ok()
    };
    let nv = shape.nv.max(1);
    let block = if fits(nv) {
        nv
    } else if !fits(1) {
        1
    } else {
        let (mut lo, mut hi) = (1usize, nv);
        while lo + 1 < hi {
            let mid = lo + (hi - lo) / 2;
            if fits(mid) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        lo
    };
    let plan = eom_memory_plan(shape, LadderRoute::Ri { block }, budget);
    plan.check()?;
    Ok((LadderRoute::Ri { block }, plan))
}

// ------------------------------------------------------------ tensor helpers

/// Run-time einsum whose output may list its indices in any order: the
/// contraction runs in the engine's (batch, left-free, right-free) order and
/// is then permuted to `out`.
fn ein(spec: &str, a: &ArrayD<f64>, b: &ArrayD<f64>) -> Result<ArrayD<f64>, FerricError> {
    let err = |m: String| FerricError::General(format!("EOM-CCSD einsum '{spec}': {m}"));
    let (ins, out) = spec.split_once("->").ok_or_else(|| err("no '->'".into()))?;
    let (l, r) = ins
        .split_once(',')
        .ok_or_else(|| err("not two operands".into()))?;
    let batch = l.chars().filter(|c| r.contains(*c) && out.contains(*c));
    let left = l.chars().filter(|c| !r.contains(*c) && out.contains(*c));
    let right = r.chars().filter(|c| !l.contains(*c) && out.contains(*c));
    let natural: String = batch.chain(left).chain(right).collect();
    let res = einsum_dyn(&format!("{l},{r}->{natural}"), a.view(), b.view(), 1.0)
        .map_err(|e| err(e.to_string()))?;
    if natural == out {
        return Ok(res);
    }
    let axes: Vec<usize> = out
        .chars()
        .map(|c| natural.find(c).expect("output index in natural order"))
        .collect();
    Ok(permute_to_owned(res.view().permuted_axes(IxDyn(&axes))))
}

/// `x` with its axes permuted, standard layout.
fn perm(x: &ArrayD<f64>, axes: &[usize]) -> ArrayD<f64> {
    permute_to_owned(x.view().permuted_axes(IxDyn(axes)))
}

/// `t2 + fac/2 · (t1_ia r1_jb + t1_jb r1_ia)` (PySCF `_make_tau`).
fn make_tau(
    t2: &ArrayD<f64>,
    t1: &ArrayD<f64>,
    r1: &ArrayD<f64>,
    fac: f64,
) -> Result<ArrayD<f64>, FerricError> {
    let x = ein("ia,jb->ijab", t1, r1)?;
    let sym = &x + &perm(&x, &[1, 0, 3, 2]);
    Ok(t2 + &(sym * (0.5 * fac)))
}

/// `2 x − x` with the last two axes swapped.
fn theta_of(x: &ArrayD<f64>) -> ArrayD<f64> {
    &(2.0 * x) - &perm(x, &[0, 1, 3, 2])
}

// ----------------------------------------------------------- integral blocks

/// Chemist MO blocks over the active space, PySCF `eris` layout:
/// `oooo[i,j,k,l] = (ij|kl)`, `ovoo[i,a,j,k] = (ia|jk)`, `ovov = (ia|jb)`,
/// `oovv[i,j,a,b] = (ij|ab)`, `ovvo[i,a,b,j] = (ia|bj)`, `ovvv[i,a,b,c] =
/// (ia|bc)`; the ladder holds `(ae|bf)` densely or the RI factor `B_vv`.
struct Blocks {
    oooo: ArrayD<f64>,
    ovoo: ArrayD<f64>,
    ovov: ArrayD<f64>,
    oovv: ArrayD<f64>,
    ovvo: ArrayD<f64>,
    ovvv: ArrayD<f64>,
    ladder: Ladder,
}

/// The particle-particle ladder `Z_ijab = Σ_ef τ_ijef (ae|bf)`.
enum Ladder {
    /// `vvvv[a,e,b,f] = (ae|bf)`.
    Dense(ArrayD<f64>),
    /// `B^P_ae` and the `a`-panel width.
    Ri { b_vv: ArrayD<f64>, block: usize },
}

impl Ladder {
    fn apply(&self, tau: &ArrayD<f64>) -> Result<ArrayD<f64>, FerricError> {
        match self {
            Ladder::Dense(vvvv) => ein("ijef,aebf->ijab", tau, vvvv),
            Ladder::Ri { b_vv, block } => {
                let sh = tau.shape();
                let (no, nv) = (sh[0], sh[2]);
                let mut z = ArrayD::<f64>::zeros(IxDyn(&[no, no, nv, nv]));
                let mut a0 = 0;
                while a0 < nv {
                    let a1 = (a0 + block).min(nv);
                    let b_a = b_vv.slice(s![.., a0..a1, ..]).to_owned().into_dyn();
                    let panel = ein("Pae,Pbf->aebf", &b_a, b_vv)?;
                    let part = ein("ijef,aebf->ijab", tau, &panel)?;
                    z.slice_mut(s![.., .., a0..a1, ..]).assign(&part);
                    a0 = a1;
                }
                Ok(z)
            }
        }
    }
}

fn build_blocks(
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    c_occ: &Array2<f64>,
    c_vir: &Array2<f64>,
    route: LadderRoute,
) -> Result<Blocks, FerricError> {
    let v2c = ferric_integrals::threeindex::coulomb_metric_2c(op, dfbs)?;
    let v_inv_sqrt = cholesky_inverse_sqrt(&v2c)?;
    let eri3_ao = ferric_integrals::threeindex::eri3_tensor(op, obs, dfbs)?;
    use Axis::{O, V};
    let b_ov = build_b(
        &transform_3center_ov(&eri3_ao, c_occ, c_vir),
        &v_inv_sqrt,
        O,
        V,
    )
    .into_inner();
    let b_oo = build_b(&transform_3center_oo(&eri3_ao, c_occ), &v_inv_sqrt, O, O).into_inner();
    let b_vv = build_b(&transform_3center_vv(&eri3_ao, c_vir), &v_inv_sqrt, V, V).into_inner();
    drop(eri3_ao);
    let b_vo = perm(&b_ov, &[0, 2, 1]);
    let ladder = match route {
        LadderRoute::Dense => Ladder::Dense(ein("Pae,Pbf->aebf", &b_vv, &b_vv)?),
        LadderRoute::Ri { block } => Ladder::Ri {
            b_vv: b_vv.clone(),
            block,
        },
    };
    Ok(Blocks {
        oooo: ein("Pij,Pkl->ijkl", &b_oo, &b_oo)?,
        ovoo: ein("Pia,Pjk->iajk", &b_ov, &b_oo)?,
        ovov: ein("Pia,Pjb->iajb", &b_ov, &b_ov)?,
        oovv: ein("Pij,Pab->ijab", &b_oo, &b_vv)?,
        ovvo: ein("Pia,Pbj->iabj", &b_ov, &b_vo)?,
        ovvv: ein("Pia,Pbc->iabc", &b_ov, &b_vv)?,
        ladder,
    })
}

// ------------------------------------------------------- EE intermediates

/// Spin-adapted EE intermediates (PySCF `_IMDS.make_ee` names).
struct Imds {
    f_oo: ArrayD<f64>,
    f_vv: ArrayD<f64>,
    fov: ArrayD<f64>,
    w_oooo: ArrayD<f64>,
    w_ovoo: ArrayD<f64>,
    w_ovvo: ArrayD<f64>,
    w_ovvo_x: ArrayD<f64>,
    w_ooov: ArrayD<f64>,
    w_vovv: ArrayD<f64>,
}

fn diag_matrix(e: &[f64]) -> ArrayD<f64> {
    Array2::from_diag(&Array1::from(e.to_vec())).into_dyn()
}

/// The intermediates for amplitudes `t1`, `t2` and orbital energies `eo`,
/// `ev` (canonical HF: `f_ov = 0`, `f_oo`, `f_vv` diagonal).
fn intermediates(
    t1: &ArrayD<f64>,
    t2: &ArrayD<f64>,
    g: &Blocks,
    eo: &[f64],
    ev: &[f64],
) -> Result<Imds, FerricError> {
    let (ovoo, ovov, oovv, ovvo, ovvv, oooo) =
        (&g.ovoo, &g.ovov, &g.oovv, &g.ovvo, &g.ovvv, &g.oooo);
    let tau = make_tau(t2, t1, t1, 1.0)?;
    let theta = theta_of(t2);
    // L[m,e,n,f] = 2(me|nf) − (mf|ne)
    let l_ovov = &(2.0 * ovov) - &perm(ovov, &[0, 3, 2, 1]);

    let mut f_vv = &(2.0 * &ein("mf,mfae->ae", t1, ovvv)?) - &ein("mf,meaf->ae", t1, ovvv)?;
    let mut w_ovoo = ein("mebf,ijef->mbij", ovvv, &tau)?;
    // W_mbej: `w_ovvo` (PySCF woVvO) and the exchange-type `w_ovvo_x` (woVVo).
    let mut w_ovvo = ein("jf,mebf->mbej", t1, ovvv)?;
    let mut w_ovvo_x = -ein("jf,mfbe->mbej", t1, ovvv)?;
    let w_ooov = &ein("if,mfne->mnie", t1, ovov)? + &perm(ovoo, &[2, 0, 3, 1]);
    let tmp = ein("njbf,mfne->mbej", t2, ovov)?;
    w_ovvo = &w_ovvo - &(0.5 * &tmp);
    w_ovvo_x = &w_ovvo_x + &tmp;
    let x = ein("menf,jf->menj", ovov, t1)?;
    w_ovvo = &w_ovvo - &ein("nb,menj->mbej", t1, &x)?;
    let x = ein("mfne,jf->menj", ovov, t1)?;
    w_ovvo_x = &w_ovvo_x + &ein("nb,menj->mbej", t1, &x)?;
    w_ovvo = &w_ovvo + &(0.5 * &ein("njfb,menf->mbej", &theta, &l_ovov)?);
    let fov = ein("nf,menf->me", t1, &l_ovov)?;
    let tilab = &(0.5 * &ein("ia,jb->ijab", t1, t1)?) + t2;
    let mut f_oo = ein("mief,menf->ni", &tilab, &l_ovov)?;
    f_vv = &f_vv - &ein("mnaf,menf->ae", &tilab, &l_ovov)?;
    w_ovvo = &w_ovvo - &ein("nb,menj->mbej", t1, ovoo)?;
    w_ovvo_x = &w_ovvo_x + &ein("nb,nemj->mbej", t1, ovoo)?;
    w_ovvo = &w_ovvo + &perm(ovvo, &[0, 2, 1, 3]);
    w_ovvo_x = &w_ovvo_x - &perm(oovv, &[0, 2, 3, 1]);

    f_oo = &f_oo + &diag_matrix(eo) + &(0.5 * &ein("me,ie->mi", &fov, t1)?);
    f_vv = &f_vv + &diag_matrix(ev) - &(0.5 * &ein("me,ma->ae", &fov, t1)?);

    let w1 = ein("je,nemi->mnij", t1, ovoo)?;
    let mut w_oooo = &(&w1 + &perm(&w1, &[1, 0, 3, 2])) + &perm(oooo, &[0, 2, 1, 3]);
    let tmp = ein("meni,jneb->mbji", ovoo, t2)?;
    w_ovoo = &w_ovoo - &(0.5 * &perm(&tmp, &[0, 1, 3, 2]));
    w_ovoo = &w_ovoo - &tmp;
    let x = &(2.0 * ovoo) - &perm(ovoo, &[2, 1, 0, 3]);
    w_ovoo = &w_ovoo + &(0.5 * &ein("nemi,njeb->mbij", &x, &theta)?);
    f_oo = &f_oo + &ein("ne,nemi->mi", t1, &x)?;
    w_oooo = &w_oooo + &ein("ijef,menf->mnij", &tau, ovov)?;
    w_ovoo = &w_ovoo - &ein("nb,mnij->mbij", t1, &w_oooo)?;
    let mut t_oovv = ein("njbf,nemf->ejmb", t2, ovov)?;
    let mut t_ovvo = &(-0.5 * &ein("nifb,menf->eimb", &theta, &l_ovov)?) + &(0.5 * &t_oovv);
    w_ovoo = &w_ovoo - &ein("ie,ejmb->mbij", t1, &t_ovvo)?;
    w_ovoo = &w_ovoo - &ein("ie,ejmb->mbji", t1, &t_oovv)?;
    w_ovoo = &w_ovoo + &perm(ovoo, &[3, 1, 2, 0]);
    t_ovvo = &t_ovvo - &perm(ovvo, &[1, 3, 0, 2]);
    t_oovv = &t_oovv - &perm(oovv, &[3, 1, 0, 2]);
    w_ovoo = &w_ovoo + &ein("mebj,ie->mbij", ovvo, t1)?;
    w_ovoo = &w_ovoo + &ein("mjbe,ie->mbji", oovv, t1)?;
    w_ovoo = &w_ovoo + &ein("me,ijeb->mbij", &fov, t2)?;

    // ebmf[e,b,m,f] = (me|bf)
    let ebmf = perm(ovvv, &[1, 2, 0, 3]);
    let w = ein("ebmf,miaf->eiab", &ebmf, t2)?;
    let mut w_vovv = &(-0.5 * &perm(&w, &[0, 1, 3, 2])) - &w;
    w_vovv = &w_vovv + &perm(&ebmf, &[1, 2, 3, 0]);
    let tmp = &(-0.5 * &ebmf) + &perm(&ebmf, &[1, 0, 2, 3]);
    w_vovv = &w_vovv + &ein("efmb,mifa->eiba", &tmp, &theta)?;
    w_vovv = &w_vovv + &ein("meni,mnab->eiab", ovoo, &tau)?;
    w_vovv = &w_vovv - &ein("me,miab->eiab", &fov, t2)?;
    w_vovv = &w_vovv + &ein("ma,eimb->eiab", t1, &t_ovvo)?;
    w_vovv = &w_vovv + &ein("ma,eimb->eiba", t1, &t_oovv)?;
    Ok(Imds {
        f_oo,
        f_vv,
        fov,
        w_oooo,
        w_ovoo,
        w_ovvo,
        w_ovvo_x,
        w_ooov,
        w_vovv,
    })
}

/// `σ = (H̄ − E_CC) R` on the singlet space; returns `(σ1, σ2)` with `σ2`
/// symmetric under `(i,a) ↔ (j,b)`.
fn sigma(
    r1: &ArrayD<f64>,
    r2: &ArrayD<f64>,
    t1: &ArrayD<f64>,
    t2: &ArrayD<f64>,
    g: &Blocks,
    im: &Imds,
) -> Result<(ArrayD<f64>, ArrayD<f64>), FerricError> {
    let (ovov, ovvv) = (&g.ovov, &g.ovvv);
    let mut hr1 = &ein("ae,ie->ia", &im.f_vv, r1)? - &ein("mi,ma->ia", &im.f_oo, r1)?;
    hr1 = &hr1 + &(2.0 * &ein("me,imae->ia", &im.fov, r2)?);
    hr1 = &hr1 - &ein("me,imea->ia", &im.fov, r2)?;

    let tau2 = make_tau(r2, r1, t1, 2.0)?;
    let mut hr2 = g.ladder.apply(&tau2)?;
    hr2 = &hr2 + &ein("mnij,mnab->ijab", &im.w_oooo, r2)?;
    hr2 *= 0.5;
    hr2 = &hr2 + &ein("be,ijae->ijab", &im.f_vv, r2)?;
    hr2 = &hr2 - &ein("mj,imab->ijab", &im.f_oo, r2)?;

    let theta = theta_of(r2);
    hr1 = &hr1 + &ein("mfae,mife->ia", ovvv, &theta)?;
    let tmp = ein("meaf,ijef->maij", ovvv, &tau2)?;
    hr2 = &hr2 - &ein("ma,mbij->ijab", t1, &tmp)?;
    // Doubles <- singles through one ovvv integral and t2.
    let tmp = &(2.0 * &ein("meaf,me->af", ovvv, r1)?) - &ein("mfae,me->af", ovvv, r1)?;
    hr2 = &hr2 + &ein("af,ijfb->ijab", &tmp, t2)?;
    hr2 = &hr2 - &ein("mbij,ma->ijab", &im.w_ovoo, r1)?;
    hr2 = &hr2 + &ein("ejab,ie->ijab", &im.w_vovv, r1)?;

    let tmp = ein("mbej,imea->jiab", &im.w_ovvo_x, r2)?;
    hr2 = &hr2 + &tmp;
    hr2 = &hr2 + &(0.5 * &perm(&tmp, &[0, 1, 3, 2]));

    let wvo = &(0.5 * &im.w_ovvo_x) + &im.w_ovvo;
    hr1 = &hr1 + &(2.0 * &ein("maei,me->ia", &wvo, r1)?);
    hr2 = &hr2 + &ein("mbej,imae->ijab", &wvo, &theta)?;

    hr1 = &hr1 - &ein("mnie,mnae->ia", &im.w_ooov, &theta)?;
    let tmp = &(2.0 * &ein("nmie,me->ni", &im.w_ooov, r1)?) - &ein("mnie,me->ni", &im.w_ooov, r1)?;
    hr2 = &hr2 - &ein("ni,njab->ijab", &tmp, t2)?;

    let tmp = &(2.0 * &ein("mfne,mf->en", ovov, r1)?) - &ein("menf,mf->en", ovov, r1)?;
    let tmp = &ein("en,nb->eb", &tmp, t1)? + &ein("menf,mnbf->eb", ovov, &theta)?;
    hr2 = &hr2 - &ein("eb,ijea->jiab", &tmp, t2)?;

    let tmp = ein("nemf,imef->ni", ovov, &theta)?;
    hr1 = &hr1 - &ein("na,ni->ia", t1, &tmp)?;
    hr2 = &hr2 - &ein("mj,miab->ijba", &tmp, t2)?;

    let tmp = ein("menf,ijef->mnij", ovov, &tau2)?;
    let tau = 0.5 * &make_tau(t2, t1, t1, 1.0)?;
    hr2 = &hr2 + &ein("mnij,mnab->ijab", &tmp, &tau)?;

    let hr2 = &hr2 + &perm(&hr2, &[1, 0, 3, 2]);
    Ok((hr1, hr2))
}

// --------------------------------------------------------------- packing

/// `r1` then the lower triangle (row-major, `q ≤ p`) of `M[ia,jb] = r2[i,j,a,b]`.
fn pack(r1: &ArrayD<f64>, r2: &ArrayD<f64>, no: usize, nv: usize) -> Array1<f64> {
    let nov = no * nv;
    let mut out = Array1::<f64>::zeros(nov + nov * (nov + 1) / 2);
    let mut k = 0;
    for i in 0..no {
        for a in 0..nv {
            out[k] = r1[[i, a]];
            k += 1;
        }
    }
    for p in 0..nov {
        let (i, a) = (p / nv, p % nv);
        for q in 0..=p {
            let (j, b) = (q / nv, q % nv);
            out[k] = r2[[i, j, a, b]];
            k += 1;
        }
    }
    out
}

/// Inverse of [`pack`]: `r2` filled symmetrically.
fn unpack(x: ndarray::ArrayView1<f64>, no: usize, nv: usize) -> (ArrayD<f64>, ArrayD<f64>) {
    let nov = no * nv;
    let mut r1 = ArrayD::<f64>::zeros(IxDyn(&[no, nv]));
    let mut r2 = ArrayD::<f64>::zeros(IxDyn(&[no, no, nv, nv]));
    let mut k = 0;
    for i in 0..no {
        for a in 0..nv {
            r1[[i, a]] = x[k];
            k += 1;
        }
    }
    for p in 0..nov {
        let (i, a) = (p / nv, p % nv);
        for q in 0..=p {
            let (j, b) = (q / nv, q % nv);
            r2[[i, j, a, b]] = x[k];
            r2[[j, i, b, a]] = x[k];
            k += 1;
        }
    }
    (r1, r2)
}

// ------------------------------------------------------------- operator

/// The packed singlet operator `H̄ − E_CC` for one set of amplitudes.
pub struct EomCcsdOperator {
    no: usize,
    nv: usize,
    t1: ArrayD<f64>,
    t2: ArrayD<f64>,
    blocks: Blocks,
    imds: Imds,
    route: LadderRoute,
    // Holds the plan's debit against the global pool while the tensors live.
    _charge: ferric_core::memory::pool::Reservation,
}

impl EomCcsdOperator {
    /// Build the operator from spin-adapted CCSD amplitudes `t1[i,a]`,
    /// `t2[i,j,a,b]` over the active space `frozen_core..nocc` × virtuals.
    /// `max_space` is the Davidson subspace size the plan charges.
    ///
    /// # Errors
    ///
    /// A non-restricted reference; amplitudes whose shape is not
    /// `(no, nv)` / `(no, no, nv, nv)` (e.g. spin-orbital ones); a memory plan
    /// that does not fit; an integral failure.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        mol: &Molecule,
        obs: &PreparedBasis,
        dfbs: &PreparedBasis,
        op: Operator,
        rhf: &ScfResult,
        t1: &Array2<f64>,
        t2: &Array4<f64>,
        frozen_core: usize,
        n_roots: usize,
        max_space: Option<usize>,
        memory_budget_bytes: Option<usize>,
    ) -> Result<Self, FerricError> {
        if !matches!(rhf.spin, ferric_scf::Spin::Restricted) {
            return Err(FerricError::General(
                "EOM-CCSD singlets need a closed-shell RHF reference".into(),
            ));
        }
        let (nocc_total, nmo) = crate::helpers::closed_shell_counts(mol, rhf)?;
        let no = active_occ(nocc_total, frozen_core)?;
        let nv = nmo - nocc_total;
        if t1.dim() != (no, nv) || t2.dim() != (no, no, nv, nv) {
            return Err(FerricError::General(format!(
                "EOM-CCSD needs spin-adapted amplitudes t1 ({no}, {nv}) and t2 ({no}, {no}, {nv}, \
                 {nv}); got t1 {:?} and t2 {:?}",
                t1.dim(),
                t2.dim()
            )));
        }
        let nov = no * nv;
        let dim = nov + nov * (nov + 1) / 2;
        let n_guess = (2 * n_roots).max(n_roots + 8).min(dim);
        let space = max_space
            .unwrap_or((n_guess + 10 * n_roots).max(50))
            .max(n_guess + 2 * n_roots)
            .min(dim);
        let shape = EomShape {
            no,
            nv,
            nbas: obs.nbasis(),
            naux: dfbs.nbasis(),
            n_roots,
            max_space: space,
        };
        let pool = ferric_core::memory::pool::global();
        let mut budget = ferric_core::memory::resolve_budget_bytes(memory_budget_bytes);
        if let Some(p) = &pool {
            budget = budget.min(p.available_bytes());
        }
        let (route, mut plan) = choose_route(shape, budget)?;
        if let Some(p) = pool {
            plan = plan.with_pool(&p);
        }
        let charge = plan.commit()?;

        let eps = rhf.eps_r();
        let c = rhf.mos_r();
        let c_occ = c.slice(s![.., frozen_core..nocc_total]).to_owned();
        let c_vir = c.slice(s![.., nocc_total..nmo]).to_owned();
        let eo: Vec<f64> = (frozen_core..nocc_total).map(|p| eps[p]).collect();
        let ev: Vec<f64> = (nocc_total..nmo).map(|p| eps[p]).collect();
        let blocks = build_blocks(obs, dfbs, op, &c_occ, &c_vir, route)?;
        let t1 = t1.clone().into_dyn();
        let t2 = t2.clone().into_dyn();
        let imds = intermediates(&t1, &t2, &blocks, &eo, &ev)?;
        Ok(Self {
            no,
            nv,
            t1,
            t2,
            blocks,
            imds,
            route,
            _charge: charge,
        })
    }

    /// Length of the packed singlet vector.
    pub fn dim(&self) -> usize {
        let nov = self.no * self.nv;
        nov + nov * (nov + 1) / 2
    }

    /// `(no, nv)`: active occupied and virtual counts.
    pub fn sizes(&self) -> (usize, usize) {
        (self.no, self.nv)
    }

    /// The ladder route the products take.
    pub fn ladder_route(&self) -> LadderRoute {
        self.route
    }

    /// `(H̄ − E_CC) x` for one packed vector.
    ///
    /// # Errors
    ///
    /// A wrong vector length, or an internal contraction failure.
    pub fn apply(&self, x: ndarray::ArrayView1<f64>) -> Result<Array1<f64>, FerricError> {
        if x.len() != self.dim() {
            return Err(FerricError::General(format!(
                "EOM-CCSD product: vector length {} != {}",
                x.len(),
                self.dim()
            )));
        }
        let (r1, r2) = unpack(x, self.no, self.nv);
        let (h1, h2) = sigma(&r1, &r2, &self.t1, &self.t2, &self.blocks, &self.imds)?;
        Ok(pack(&h1, &h2, self.no, self.nv))
    }

    /// The Davidson preconditioner: `F_aa − F_ii` on singles and the sum of
    /// two such on doubles, from the dressed `F_oo`, `F_vv` diagonals.
    pub fn diagonal(&self) -> Array1<f64> {
        let (no, nv) = (self.no, self.nv);
        let mut d1 = ArrayD::<f64>::zeros(IxDyn(&[no, nv]));
        for i in 0..no {
            for a in 0..nv {
                d1[[i, a]] = self.imds.f_vv[[a, a]] - self.imds.f_oo[[i, i]];
            }
        }
        let mut d2 = ArrayD::<f64>::zeros(IxDyn(&[no, no, nv, nv]));
        for ((i, j, a, b), v) in d2
            .view_mut()
            .into_dimensionality::<Ix4>()
            .expect("rank 4")
            .indexed_iter_mut()
        {
            *v = d1[[i, a]] + d1[[j, b]];
        }
        pack(&d1, &d2, no, nv)
    }

    /// The packed vector as `(r1, r2)`.
    pub fn split(&self, x: ndarray::ArrayView1<f64>) -> (Array2<f64>, Array4<f64>) {
        let (r1, r2) = unpack(x, self.no, self.nv);
        (
            r1.into_dimensionality::<Ix2>().expect("rank 2"),
            r2.into_dimensionality::<Ix4>().expect("rank 4"),
        )
    }

    /// The singles-singles block `nov × nov` (columns: unit singles).
    ///
    /// # Errors
    ///
    /// A failed product.
    pub fn singles_block(&self) -> Result<Array2<f64>, FerricError> {
        let nov = self.no * self.nv;
        let mut s = Array2::<f64>::zeros((nov, nov));
        let mut e = Array1::<f64>::zeros(self.dim());
        for k in 0..nov {
            e[k] = 1.0;
            let col = self.apply(e.view())?;
            s.column_mut(k).assign(&col.slice(s![..nov]));
            e[k] = 0.0;
        }
        Ok(s)
    }

    /// The whole packed operator as a dense `dim × dim` matrix (test oracle).
    ///
    /// # Errors
    ///
    /// A failed product.
    pub fn dense(&self) -> Result<Array2<f64>, FerricError> {
        let n = self.dim();
        let mut m = Array2::<f64>::zeros((n, n));
        let mut e = Array1::<f64>::zeros(n);
        for k in 0..n {
            e[k] = 1.0;
            m.column_mut(k).assign(&self.apply(e.view())?);
            e[k] = 0.0;
        }
        Ok(m)
    }
}

/// `(singles weight, scale to ‖R‖ = 1)` in the spin-orbital norm
/// `‖R‖² = 2‖r1‖² + ‖r2‖² + ½‖r2 − r2ᵀ_{ab}‖²`.
fn spin_norm(r1: &Array2<f64>, r2: &Array4<f64>) -> (f64, f64) {
    let s1 = 2.0 * r1.iter().map(|x| x * x).sum::<f64>();
    let ab = r2.iter().map(|x| x * x).sum::<f64>();
    let swapped = r2.view().permuted_axes([0, 1, 3, 2]);
    let same: f64 = r2
        .iter()
        .zip(swapped.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum();
    let total = s1 + ab + 0.5 * same;
    (s1 / total, 1.0 / total.sqrt())
}

/// The lowest `cfg.n_roots` EE-EOM-CCSD singlet excitation energies from
/// converged spin-adapted CCSD amplitudes `cc` (from
/// [`crate::ccsd_closed_shell::ccsd_closed_shell`] with the same
/// `frozen_core`, operator and auxiliary basis).
///
/// # Errors
///
/// See [`EomCcsdOperator::new`]; amplitudes without `t1`; an unconverged
/// Davidson solve.
#[allow(clippy::too_many_arguments)]
pub fn eom_ccsd_singlets(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    rhf: &ScfResult,
    cc: &CcResult,
    frozen_core: usize,
    cfg: &EomCcsdConfig,
) -> Result<EomCcsdResult, FerricError> {
    let t1 = cc.t1.as_ref().ok_or_else(|| {
        FerricError::General("EOM-CCSD needs CCSD amplitudes with singles (t1)".into())
    })?;
    if cfg.n_roots == 0 {
        return Err(FerricError::General(
            "EOM-CCSD: n_roots must be at least 1".into(),
        ));
    }
    let opr = EomCcsdOperator::new(
        mol,
        obs,
        dfbs,
        op,
        rhf,
        t1,
        &cc.t2,
        frozen_core,
        cfg.n_roots,
        cfg.max_space,
        cfg.memory_budget_bytes,
    )?;
    let diag = opr.diagonal();
    let settings = NonsymDavidsonSettings {
        conv_tol: cfg.conv_tol,
        max_iter: cfg.max_iter,
        max_space: cfg.max_space,
    };
    let apply = |x: &Array2<f64>| -> Result<Array2<f64>, FerricError> {
        let mut out = Array2::<f64>::zeros(x.raw_dim());
        for (k, col) in x.columns().into_iter().enumerate() {
            out.column_mut(k).assign(&opr.apply(col)?);
        }
        Ok(out)
    };
    let sol = davidson_nonsymmetric(apply, &diag, cfg.n_roots, &settings)?;
    let mut r1s = Vec::new();
    let mut r2s = Vec::new();
    let mut weights = Vec::new();
    for col in sol.vectors.columns() {
        let (r1, r2) = opr.split(col);
        let (w, scale) = spin_norm(&r1, &r2);
        r1s.push(r1 * scale);
        r2s.push(r2 * scale);
        weights.push(w);
    }
    Ok(EomCcsdResult {
        excitation_energies: sol.eigenvalues,
        r1: r1s,
        r2: r2s,
        singles_weight: weights,
        residuals: sol.residuals,
        iterations: sol.iterations,
        products: sol.products,
        ladder_route: opr.ladder_route(),
    })
}

/// The certificate of an `eom-ccsd` run: grade from
/// `ferric_core::certificate::METHOD_KIND_GRADES` folded with the SCF
/// reference, the RI basis, the frozen core and, on the RI route, the ladder
/// block.
pub fn eom_ccsd_certificate(
    reference: &ferric_core::certificate::Certificate,
    aux: &str,
    frozen_core: usize,
    r: &EomCcsdResult,
) -> ferric_core::certificate::Certificate {
    let c = ferric_core::certificate::Certificate::for_post_scf("eom-ccsd", reference)
        .with_ri(aux)
        .with_frozen_core(frozen_core);
    match r.ladder_route {
        LadderRoute::Ri { block } => c.with_approximation(
            "ri-ladder",
            format!("integral-direct VVVV ladder, block {block}"),
        ),
        LadderRoute::Dense => c,
    }
}
