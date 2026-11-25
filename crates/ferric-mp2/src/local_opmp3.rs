//! Amplitude-threshold local OP-MP3: the OP-MP3 models
//! ([`crate::opmp3`]) from first- and second-order amplitudes solved on ONE
//! amplitude-threshold pattern in localized orbitals.
//!
//! ```text
//! P A P c1 = −P K,      P A P c2 = −P R(c1),      A = Fock superoperator
//! W(2) = W[c1, K],      W(3) = W[c2, K],          E3 = c2·K = c1·R(c1)
//! ```
//!
//! `P` is the local MP2 pattern (WSHG23 Eq-8 mask, optionally widened by the
//! Fock-propagated screen, [`crate::ragged::pair_blocks_screened`]) over
//! (λ-)Boys occupieds and VV-HV virtuals. Equations, the invariance and
//! one-mask (Wigner 2n+1) arguments and the KS-reference rules:
//! `book/src/13-local-mp2/04-local-opmp3.md#local-opmp3`.
//!
//! # Reference
//!
//! The reference is the OP-MP3 one ([`crate::opmp3_reference`]): the HF
//! Fock matrix of the converged density (RHF or RKS), semicanonical in the
//! occupied and virtual blocks. The localization (λ-Boys weighting, VV-HV
//! hard-virtual pseudo-canonicalization) and the `F_oo` / `F_vv` blocks of
//! `A` all use that HF Fock matrix, never a Kohn–Sham one; the frozen core
//! is the lowest semicanonical occupieds, as in the canonical driver.
//! `E_NBS` is reference-wide and mask-independent, so it is taken from the
//! reference unchanged.
//!
//! # V1 scope
//!
//! The pattern solves are ragged (one block per retained pair), but the
//! residual `R(c1)` is evaluated by the canonical spin-blocked MP3 residual
//! (`u_mp3_residual`) over the localized orbitals — dense
//! `(no·nv)²` amplitudes and RI factors — and only its pattern part is kept.
//! The cost is therefore that of canonical RI-MP3 plus the solves: no cost
//! or scaling claim attaches to this module; it is the correctness
//! reference a ragged residual must reproduce. Closed shell only.

use crate::lmp2_amplitude::{
    checked_vvhv_with_fock, localized_spaces_with_fock, AmplitudeLmp2Config,
};
use crate::opmp3::{w_channels, Opmp3Models, WMatrices};
use crate::opmp3_driver::{Opmp3Config, Opmp3Result, OPMP3_FOCK_THRESH};
use crate::opmp3_reference::opmp3_reference;
use crate::ragged::{
    check_swap_closed, kept_fraction, pair_blocks_screened, solve_ragged_rhs, PairBlock, Ragged,
    RaggedSolve,
};
use crate::u_mp3::{
    build_source, channel_energy, doubles_integrals, u_mp3_peak_bytes, SpinBlockedDoubles,
    UMp3Dims, UMp3Eri,
};
use crate::u_mp3_eri::SpinOrbitals;
use crate::u_mp3_residual::mp3_residual;
use ferric_core::FerricError;
use ferric_integrals::operator::Operator;
use ferric_scf::{PreparedAux, ScfResult, Spin};
use ndarray::{s, Array2, Array4};

/// Relative CG residual of both pattern solves when the caller sets none.
/// The W matrices are linear in the amplitudes, so their error is first
/// order in it: about `1e-11 · |E_c|`.
pub const LOCAL_OPMP3_CG_RTOL: f64 = 1e-11;

/// Result of [`local_opmp3_on`]: the OP-MP3 result of the local
/// amplitudes, and what the pattern and its solves did.
#[derive(Debug, Clone)]
pub struct LocalOpmp3Result {
    /// Energies, W (in the localized occupied basis) and models, in the
    /// canonical driver's result type (`dims` are the localized spaces,
    /// `peak_bytes` the residual stage's pre-flighted peak).
    pub opmp3: Opmp3Result,
    /// `E3` by the Wigner 2n+1 route, `c1·R(c1)` (total over channels).
    pub e3_2n1: f64,
    /// Threshold ε of the pattern.
    pub eps: f64,
    /// Retained fraction of the `(no·nv)²` amplitude space.
    pub keep_fraction: f64,
    /// Ordered occupied pairs with a retained block.
    pub n_pairs: usize,
    /// CG iterations of the `c1` and `c2` solves.
    pub cg_iterations: [usize; 2],
    /// Final relative residuals of the `c1` and `c2` solves.
    pub cg_relres: [f64; 2],
    /// The occupied localization.
    pub lambda_boys: crate::boys::LambdaBoysReport,
    /// The virtual-space build.
    pub virtual_space: crate::vvhv_local::VirtualSpaceReport,
}

impl LocalOpmp3Result {
    /// `|c2·K − c1·R(c1)|`: zero up to the solver residuals when both orders
    /// are solved on one pattern.
    pub fn e3_identity_gap(&self) -> f64 {
        (self.opmp3.e3.total() - self.e3_2n1).abs()
    }
}

/// The [`AmplitudeLmp2Config`] fields local OP-MP3 reads are `eps`,
/// `fock_screen`, `lambda_boys`, `virtual_space`, `cg_rtol`, `cg_max_iter`
/// and `frozen_core` (which must equal the OP-MP3 one). The others select
/// LMP2 assembly paths this driver does not have; setting one is an error.
fn check_local_config(
    local: &AmplitudeLmp2Config,
    config: &Opmp3Config,
) -> Result<(), FerricError> {
    let unsupported = |what: &str| {
        Err(FerricError::Unsupported {
            feature: format!("local OP-MP3 with {what}"),
            reason: "the in-core local OP-MP3 driver builds the full pattern from dense \
                     localized integrals; that LMP2 assembly option has no counterpart here"
                .to_string(),
        })
    };
    if !(local.eps.is_finite() && local.eps >= 0.0) {
        return Err(FerricError::General(format!(
            "local OP-MP3: eps must be finite and >= 0, got {}",
            local.eps
        )));
    }
    if local.frozen_core != config.frozen_core {
        return Err(FerricError::General(format!(
            "local OP-MP3: the local frozen_core ({}) differs from the OP-MP3 one ({})",
            local.frozen_core, config.frozen_core
        )));
    }
    if local.pair_gate_cal.is_some() {
        return unsupported("a pair gate (pair_gate_cal)");
    }
    if local.fit_radius_bohr.is_some() {
        return unsupported("domain-local RI fits (fit_radius_bohr)");
    }
    if local.aux_tail_frac.is_some() {
        return unsupported("an aux tail truncation (aux_tail_frac)");
    }
    if local.fvv_sparse_rho != 0.0 {
        return unsupported("a sparse Fvv product (fvv_sparse_rho)");
    }
    Ok(())
}

/// The ragged pattern over the dense localized `(ia|jb)` block `g_ab`
/// (`[i, j, a, b]`): every pair `i ≤ j` and its mirror through the shared
/// LMP2 pair builder, so the mask is the local MP2 one bit for bit (the
/// Schwarz candidate prefilter of the RI assembly never drops a retained
/// element, so it is left out here).
fn pattern(
    g_ab: &Array4<f64>,
    f_oo: &Array2<f64>,
    f_vv: &Array2<f64>,
    eps: f64,
    fock_screen: Option<f64>,
) -> Ragged {
    let (no, _, nv, _) = g_ab.dim();
    let fo: Vec<f64> = (0..no).map(|i| f_oo[(i, i)]).collect();
    let fv: Vec<f64> = (0..nv).map(|a| f_vv[(a, a)]).collect();
    let cand: Vec<usize> = (0..nv).collect();
    let mut pairs: Vec<PairBlock> = Vec::new();
    for i in 0..no {
        for j in i..no {
            let g = g_ab.slice(s![i, j, .., ..]).to_owned();
            pairs.extend(pair_blocks_screened(
                i,
                j,
                &g,
                &cand,
                nv,
                f_vv,
                &fo,
                &fv,
                eps,
                fock_screen,
            ));
        }
    }
    Ragged::from_pairs(pairs)
}

/// Ragged blocks `x[i, j, da, db]` of a dense `[i, j, a, b]` tensor, scaled.
fn gather(rg: &Ragged, x: &Array4<f64>, scale: f64) -> Vec<Array2<f64>> {
    rg.pairs
        .iter()
        .map(|pb| {
            Array2::from_shape_fn((pb.da.len(), pb.db.len()), |(r, c)| {
                scale * x[(pb.i, pb.j, pb.da[r], pb.db[c])]
            })
        })
        .collect()
}

/// The dense `[i, j, a, b]` αβ amplitudes of ragged blocks `t` (zero off the
/// pattern).
fn scatter(rg: &Ragged, t: &[Array2<f64>], no: usize, nv: usize) -> Array4<f64> {
    let mut out = Array4::<f64>::zeros((no, no, nv, nv));
    for (pb, blk) in rg.pairs.iter().zip(t) {
        let nb = pb.db.len();
        for (r, &a) in pb.da.iter().enumerate() {
            for (c, &b) in pb.db.iter().enumerate() {
                if pb.pat[r * nb + c] {
                    out[(pb.i, pb.j, a, b)] = blk[(r, c)];
                }
            }
        }
    }
    out
}

/// Closed-shell spin blocks of spatial amplitudes `t[i, j, a, b]`:
/// `t^αα_ijab = t_ijab − t_ijba`, `t^ββ = t^αα`, `t^αβ = t`.
fn closed_shell_doubles(t: Array4<f64>) -> SpinBlockedDoubles {
    let mut aa = t.clone();
    aa -= &t.view().permuted_axes([0, 1, 3, 2]);
    SpinBlockedDoubles {
        bb: aa.clone(),
        aa,
        ab: t,
    }
}

/// One pattern solve `P A P x = P rhs`; non-convergence is an error.
fn solve(
    rg: &Ragged,
    f_oo: &Array2<f64>,
    rhs: &[Array2<f64>],
    rtol: f64,
    max_iter: usize,
    what: &str,
) -> Result<RaggedSolve, FerricError> {
    let sol = solve_ragged_rhs(rg, f_oo, rhs, rtol, max_iter);
    if !sol.converged {
        return Err(FerricError::Convergence(format!(
            "local OP-MP3 {what} solve: relative residual {:.2e} after {} iterations \
             (rtol {rtol:.0e}, max_iter {max_iter})",
            sol.relres, sol.iterations
        )));
    }
    Ok(sol)
}

/// Local OP-MP3 on an existing converged closed-shell reference `scf` (RHF
/// or RKS) of `aux`'s system ([`PreparedAux::system`]: molecule, orbital
/// basis and the basis set it was prepared from, bounds and context), RI
/// integrals over `aux`, after [`PreparedAux::check_reference`]. `config` gives
/// the OP-MP3 settings (`x_ss`, frozen core), `local` the pattern (see
/// `check_local_config`).
pub fn local_opmp3_on(
    aux: &PreparedAux<'_>,
    scf: &ScfResult,
    config: &Opmp3Config,
    local: &AmplitudeLmp2Config,
) -> Result<LocalOpmp3Result, FerricError> {
    aux.check_reference(scf)?;
    let sys = aux.system();
    let (ctx, mol, obs, bounds) = (sys.ctx(), sys.mol(), sys.basis(), sys.bounds());
    let obs_bs = sys.basis_set();
    let aux = aux.basis();
    let x_ss = config.resolved_x_ss()?;
    check_local_config(local, config)?;
    if mol.multiplicity != 1 || !matches!(scf.spin, Spin::Restricted) {
        return Err(FerricError::Unsupported {
            feature: "local OP-MP3 on an open-shell or unrestricted reference".to_string(),
            reason: "the local pattern solves are closed-shell (RHF / RKS references)".to_string(),
        });
    }
    let r = opmp3_reference(ctx, mol, obs, bounds, OPMP3_FOCK_THRESH, scf)?;
    let e_nbs = r.e_nbs(config.frozen_core)?;
    let f_hf = &r.f_ao[0];
    // The semicanonical reference as an SCF result: same occupied and
    // virtual spaces as `scf`, ordered by the HF Fock matrix, so the frozen
    // core matches the canonical driver's for a KS reference too.
    let mut scf_sc = scf.clone();
    scf_sc.mos_alpha = r.spins[0].mos.clone();
    let (vvhv, _, virtual_space) = checked_vvhv_with_fock(
        mol,
        obs,
        obs_bs,
        &scf_sc,
        f_hf,
        "local_opmp3",
        &local.virtual_space,
    )?;
    let sp = localized_spaces_with_fock(
        mol,
        obs,
        &scf_sc,
        f_hf,
        config.frozen_core,
        &vvhv,
        local.lambda_boys_spec(),
    )?;
    let (no, nv) = (sp.no, sp.nv);
    // the residual reads no orbital energy; the Fock diagonals are recorded
    let diag = |f: &Array2<f64>| -> Vec<f64> { f.diag().to_vec() };
    let spin = SpinOrbitals {
        c_occ: sp.c_locc.clone(),
        c_vir: sp.c_vloc.clone(),
        e_occ: diag(&sp.f_oo),
        e_vir: diag(&sp.f_vv),
    };
    let orbs = [spin.clone(), spin];
    let dims = UMp3Dims {
        no_a: no,
        no_b: no,
        nv_a: nv,
        nv_b: nv,
        nbas: obs.nbasis(),
        naux: aux.nbasis(),
        frozen_core: config.frozen_core,
    };
    let label = format!(
        "local OP-MP3 (RI residual stage; occ {no}, vir {nv}, nbas {}, naux {})",
        dims.nbas, dims.naux
    );
    let peak = u_mp3_peak_bytes(&dims, false);
    ferric_core::memory::check_alloc(
        &label,
        peak,
        ferric_core::memory::resolve_budget_bytes(config.memory_budget_bytes),
    )?;
    let _peak_charge = crate::rimp2::charge_mo_side(&label, peak)?;

    let op = Operator::coulomb();
    let src = build_source(obs, op, UMp3Eri::Ri { aux }, &orbs)?;
    let (k, ring) = doubles_integrals(&src)?;
    let rg = pattern(&k.ab, &sp.f_oo, &sp.f_vv, local.eps, local.fock_screen);
    check_swap_closed(&rg).map_err(|e| {
        FerricError::General(format!("local OP-MP3: pattern is not swap-closed ({e})"))
    })?;
    let rtol = local.cg_rtol.unwrap_or(LOCAL_OPMP3_CG_RTOL);
    let s1 = solve(
        &rg,
        &sp.f_oo,
        &gather(&rg, &k.ab, -1.0),
        rtol,
        local.cg_max_iter,
        "c1",
    )?;
    let c1 = closed_shell_doubles(scatter(&rg, &s1.t, no, nv));
    let res = mp3_residual(&src, ring, &c1)?;
    let s2 = solve(
        &rg,
        &sp.f_oo,
        &gather(&rg, &res.ab, -1.0),
        rtol,
        local.cg_max_iter,
        "c2",
    )?;
    let c2 = closed_shell_doubles(scatter(&rg, &s2.t, no, nv));

    let w = WMatrices {
        w2: w_channels(&c1, &k),
        w3: w_channels(&c2, &k),
    };
    let models = Opmp3Models::evaluate(&w, x_ss)?;
    let e2 = channel_energy(&c1, &k);
    let e3 = channel_energy(&c2, &k);
    Ok(LocalOpmp3Result {
        opmp3: Opmp3Result {
            e_scf: r.e_scf,
            e_hf: r.e_hf,
            e_nbs,
            e2,
            e3,
            w,
            models,
            max_ov: [r.spins[0].max_ov(), r.spins[1].max_ov()],
            dims,
            peak_bytes: peak,
            scf_iterations: scf.iterations,
        },
        e3_2n1: channel_energy(&c1, &res).total(),
        eps: local.eps,
        keep_fraction: kept_fraction(&rg, no, nv),
        n_pairs: rg.pairs.len(),
        cg_iterations: [s1.iterations, s2.iterations],
        cg_relres: [s1.relres, s2.relres],
        lambda_boys: sp.lambda_boys,
        virtual_space,
    })
}

/// The whole run on `aux`'s system: the OP-MP3 reference SCF
/// ([`crate::opmp3_driver::opmp3_scf`]), then [`local_opmp3_on`].
pub fn run_local_opmp3(
    aux: &PreparedAux<'_>,
    config: &Opmp3Config,
    local: &AmplitudeLmp2Config,
) -> Result<LocalOpmp3Result, FerricError> {
    config.resolved_x_ss()?;
    check_local_config(local, config)?;
    let scf = crate::opmp3_driver::opmp3_scf(aux.system(), config)?;
    local_opmp3_on(aux, &scf, config, local)
}
