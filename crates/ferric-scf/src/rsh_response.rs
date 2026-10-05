//! The range-separated exchange **response** for the orbital-Hessian matvecs.
//!
//! # Why this module exists
//!
//! The converged RSH Fock and the orbital Hessian used to disagree about which
//! operator exchange is built with. The Fock (`uhf.rs`, `rhf.rs`, `rohf.rs`)
//! assembles
//!
//! ```text
//!   K_total,σ = c_SR · K[erfc(ω)](D_σ) + c_LR · K[erf(ω)](D_σ)
//! ```
//!
//! via [`crate::fock_assembly::subtract_rsh_exchange`] and two [`DfK`] fitters.
//! The Hessian matvecs built their response from ONE `build_jk_with_pool` call
//! at the ambient (Coulomb) operator scaled by the single scalar `k_mix.sr`,
//! which is not the ω ≠ 0 kernel at all. `stability::ks_reference_is_analysable`
//! therefore refused every range-separated hybrid, and the Newton accelerators
//! either used the wrong kernel or (ROHF, `rohf.rs`) passed `k_mix_sr = 0.0`
//! and dropped the exchange response entirely.
//!
//! # Why density fitting, not direct four-centre erf/erfc integrals
//!
//! libint2 can compute four-centre `erf`/`erfc` ERIs, and
//! [`crate::screening`] admits Schwarz bounds for both kernels, so a direct
//! path is *buildable*. It would nevertheless be the WRONG operator here.
//! `crate::driver::prepare` states the constraint:
//!
//! > RSH exchange is density-fitted ONLY: there is no conventional four-centre
//! > erf/erfc exchange path.
//!
//! The energy the SCF minimizes is therefore the density-FITTED RSH energy. Its
//! second derivative is the response of the density-fitted exchange, built from
//! the SAME `B[P,μ,ν]` tensors. A direct four-centre response would be the
//! second derivative of a different functional: it would differ from the true
//! Hessian of the converged energy by the DF fitting error, which is orders of
//! magnitude above the ~3e-10 analytic-vs-finite-difference bar the ω = 0
//! matvec holds, and — critically — would NOT shrink as the finite-difference
//! step is refined. That floor-not-step-error signature is what distinguishes
//! a wrong operator from a coarse difference, and it is why the fitters are
//! shared rather than rebuilt with a different kernel.
//!
//! # Interior mutability, and why this BORROWS the fitters
//!
//! [`DfK::build`] needs `&mut self` (it streams its dressed tensor through a
//! budgeted reduction). The Hessian matvecs take `&inputs` and are driven
//! inside a PCG loop and a Davidson eigensolve that call them many times with
//! the same inputs, so the fitters sit behind a [`RefCell`] and the borrow is
//! taken and released inside [`RshResponse::exchange_response`] — no guard is
//! ever held across a call back into caller code.
//!
//! This type BORROWS the cells rather than owning the fitters. That is
//! load-bearing: inside the SCF loop the same two fitters assemble the
//! converged Fock every iteration, so a Newton or TRAH step taken mid-loop must
//! reach the identical `B[P,μ,ν]` tensors. Owning them would force either a
//! restructure of the loop's ownership or — worse — a SECOND pair built with
//! the same arguments, which pays the DF build twice and reintroduces exactly
//! the Fock/Hessian divergence this module exists to make impossible.

use crate::df_k::DfK;
use crate::fock::KBuilder;
use ferric_core::FerricError;
use ndarray::Array2;
use std::cell::RefCell;

/// The SR/LR exchange-response builder a Hessian matvec uses when ω ≠ 0.
///
/// Holds the two [`DfK`] fitters the converged Fock was assembled from —
/// `Operator::erfc(ω)` and `Operator::erf(ω)` — plus the coefficients, so the
/// response and the Fock cannot diverge: both read the same `KMix` and the same
/// dressed three-index tensors.
pub struct RshResponse<'a> {
    /// `δD ↦ c_SR·δK[erfc(ω)](δD) + c_LR·δK[erf(ω)](δD)`.
    ///
    /// A boxed closure rather than the two `&RefCell<DfK<'d>>` borrows
    /// directly, for the same reason
    /// [`crate::rohf_newton::FxcResponse`] is one: the `DfK<'d>` inner lifetime
    /// is INVARIANT under `RefCell`, so carrying it in the type would force
    /// every `RhfNewtonInputs<'a>` / `UhfNewtonInputs<'a>` to unify the
    /// borrow of the fitters with the borrows of `prep`, `bounds`, `c` and the
    /// MO-basis Fock — which they cannot, since those are shorter-lived
    /// stack locals in the post-SCF stability path. Boxing erases `'d` and
    /// leaves one covariant `'a`.
    /// NOT `+ Sync`, unlike [`crate::rohf_newton::FxcResponse`]: `RefCell` is
    /// not `Sync`, and this closure does not need to be. Every driver of the
    /// Hessian matvec — the PCG loop in `*_newton_step`, the Davidson
    /// eigensolve in [`crate::stability`], the TRAH loop in [`crate::trah`] —
    /// is a SEQUENTIAL loop; the parallelism lives inside `DfK::build`'s own
    /// rayon region, below this call. A future caller that wanted to drive the
    /// matvec from several threads at once would have to revisit this, and the
    /// compiler would tell it so rather than letting it share a `RefCell`.
    apply: Box<dyn Fn(&Array2<f64>) -> Result<Array2<f64>, FerricError> + 'a>,
    c_sr: f64,
    c_lr: f64,
    omega: f64,
}

impl std::fmt::Debug for RshResponse<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RshResponse")
            .field("c_sr", &self.c_sr)
            .field("c_lr", &self.c_lr)
            .field("omega", &self.omega)
            .finish()
    }
}

impl<'a> RshResponse<'a> {
    /// Wrap the solver's own SR/LR fitters and `KMix` coefficients.
    ///
    /// `omega` is carried for diagnostics and for [`RshResponse::omega`]; the
    /// fitters themselves already embed it (they were built at
    /// `Operator::erfc(omega)` / `Operator::erf(omega)`), so nothing here can
    /// reinterpret it. Taking `c_sr`/`c_lr`/`omega` as three scalars rather
    /// than a `KMix` keeps `ferric-dft` out of this module's signature; every
    /// caller passes `k_mix.sr`, `k_mix.lr`, `k_mix.omega` from the one `KMix`
    /// the Fock read, which is what makes divergence structural.
    pub fn new<'d: 'a>(
        sr: &'a RefCell<DfK<'d>>,
        lr: &'a RefCell<DfK<'d>>,
        c_sr: f64,
        c_lr: f64,
        omega: f64,
    ) -> Self {
        Self {
            apply: Box::new(move |dd: &Array2<f64>| {
                let n = dd.nrows();
                let mut k_sr = Array2::<f64>::zeros((n, n));
                let mut k_lr = Array2::<f64>::zeros((n, n));
                // `KBuilder::build` is the SAME entry point
                // `subtract_rsh_exchange` reaches through for the converged
                // Fock's density path, on the SAME dressed B[P,mu,nu].
                KBuilder::build(&mut *sr.borrow_mut(), dd, &mut k_sr)?;
                KBuilder::build(&mut *lr.borrow_mut(), dd, &mut k_lr)?;
                // Same element-wise association as `subtract_rsh_exchange`:
                // scale SR in place, then `scaled_add(c_lr, &k_lr)`.
                k_sr *= c_sr;
                k_sr.scaled_add(c_lr, &k_lr);
                Ok(k_sr)
            }),
            c_sr,
            c_lr,
            omega,
        }
    }

    /// The range-separation parameter these fitters were built at.
    pub fn omega(&self) -> f64 {
        self.omega
    }

    /// The SR and LR exact-exchange coefficients, in that order.
    pub fn coefficients(&self) -> (f64, f64) {
        (self.c_sr, self.c_lr)
    }

    /// `c_SR · δK[erfc(ω)](δD) + c_LR · δK[erf(ω)](δD)`, the exchange response
    /// of one spin's density perturbation.
    ///
    /// Element-wise association matches
    /// [`crate::fock_assembly::subtract_rsh_exchange`]: scale the SR matrix by
    /// `c_sr` in place, then `scaled_add(c_lr, &k_lr)`. The caller applies the
    /// Fock's own sign and factor (−1 per spin for UHF/ROHF, −½ for the
    /// restricted total density), exactly as it does for the Coulomb-kernel
    /// `δK` at ω = 0.
    ///
    /// # Errors
    ///
    /// Propagates a failed DF-K contraction.
    pub fn exchange_response(&self, dd: &Array2<f64>) -> Result<Array2<f64>, FerricError> {
        (self.apply)(dd)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rhf::{build_jk, canonical_orthogonalizer};
    use crate::screening::SchwarzBounds;
    use crate::uhf_newton::{hessian_matvec, UhfNewtonInputs};
    use ferric_core::basis;
    use ferric_core::mol::Molecule;
    use ferric_core::parallel::ParallelContext;
    use ferric_integrals::basis_bridge::PreparedBasis;
    use ferric_integrals::oneelectron;
    use ferric_integrals::operator::Operator;
    use ndarray_linalg::{Eigh, Solve, UPLO};

    /// Deterministic xorshift64 PRNG (mirrors `uhf_newton_smoke.rs`).
    struct Xorshift64(u64);
    impl Xorshift64 {
        fn next_f64(&mut self) -> f64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            ((x >> 11) as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0
        }
    }

    /// Everything the FD check needs for ONE (ω, c_SR, c_LR) triple.
    struct Harness<'a> {
        ctx: ParallelContext,
        prep: PreparedBasis,
        bounds: SchwarzBounds,
        h: Array2<f64>,
        x: Array2<f64>,
        nocc_a: usize,
        nocc_b: usize,
        sr: RefCell<DfK<'a>>,
        lr: RefCell<DfK<'a>>,
        c_sr: f64,
        c_lr: f64,
        omega: f64,
    }

    /// The Fock expression the matvec differentiates, built from the SAME
    /// fitters: `F_σ = h + J(D_α + D_β) − (c_SR·K_SR + c_LR·K_LR)(D_σ)`.
    ///
    /// Pure Hartree–Fock in SR/LR form — no functional — so the FD reference
    /// needs no `V_xc` and the whole operator is expressible here.
    impl Harness<'_> {
        fn fock(&self, d_a: &Array2<f64>, d_b: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
            let n = self.h.nrows();
            let d_tot = d_a + d_b;
            let mut j = Array2::<f64>::zeros((n, n));
            let mut k_dum = Array2::<f64>::zeros((n, n));
            build_jk(
                &self.ctx,
                &self.prep,
                &self.bounds,
                1e-14,
                &d_tot,
                &mut j,
                &mut k_dum,
            )
            .unwrap();
            let rsh = RshResponse::new(&self.sr, &self.lr, self.c_sr, self.c_lr, self.omega);
            let k_a = rsh.exchange_response(d_a).unwrap();
            let k_b = rsh.exchange_response(d_b).unwrap();
            (&self.h + &j - &k_a, &self.h + &j - &k_b)
        }

        /// Occupied-block density for a spin.
        fn density(c: &Array2<f64>, nocc: usize) -> Array2<f64> {
            let occ = c.slice(ndarray::s![.., ..nocc]);
            occ.dot(&occ.t())
        }

        /// Diagonalize F in the orthogonal basis X and return the full MO set.
        fn diag(&self, f: &Array2<f64>) -> Array2<f64> {
            let ft = self.x.t().dot(f).dot(&self.x);
            let (_e, v) = ft.eigh(UPLO::Upper).unwrap();
            self.x.dot(&v)
        }

        /// Occupied→virtual gradient block `g^σ_{ai} = F^σ_{ai}` at given MOs.
        fn gradient(&self, c_a: &Array2<f64>, c_b: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
            let d_a = Self::density(c_a, self.nocc_a);
            let d_b = Self::density(c_b, self.nocc_b);
            let (f_a, f_b) = self.fock(&d_a, &d_b);
            let n = self.h.nrows();
            let ov = |m: &Array2<f64>, nocc: usize| -> Array2<f64> {
                let nv = n - nocc;
                let mut o = Array2::<f64>::zeros((nv, nocc));
                for (ir, a) in (nocc..n).enumerate() {
                    for i in 0..nocc {
                        o[(ir, i)] = m[(a, i)];
                    }
                }
                o
            };
            let fa_mo = c_a.t().dot(&f_a).dot(c_a);
            let fb_mo = c_b.t().dot(&f_b).dot(c_b);
            (ov(&fa_mo, self.nocc_a), ov(&fb_mo, self.nocc_b))
        }
    }

    /// Cayley rotation of the occ→virt block, exactly as `uhf_newton` applies it.
    fn rotate(c: &Array2<f64>, k_ov: &Array2<f64>, nocc: usize, eps: f64) -> Array2<f64> {
        let n = c.nrows();
        let mut kappa = Array2::<f64>::zeros((n, n));
        for (ir, a) in (nocc..n).enumerate() {
            for i in 0..nocc {
                let v = eps * k_ov[(ir, i)];
                kappa[(a, i)] = v;
                kappa[(i, a)] = -v;
            }
        }
        let half = 0.5 * &kappa;
        let eye = Array2::<f64>::eye(n);
        let am = &eye - &half;
        let bm = &eye + &half;
        let mut u = Array2::<f64>::zeros((n, n));
        for col in 0..n {
            let sol = am.solve(&bm.column(col).to_owned()).unwrap();
            for row in 0..n {
                u[(row, col)] = sol[row];
            }
        }
        c.dot(&u)
    }

    /// Converge the SR/LR Hartree–Fock state, then measure
    /// `|H_analytic·κ − H_FD·κ|` at two finite-difference steps.
    ///
    /// Returns `(shrink_ratio, relative_residual_at_the_finer_step)`.
    fn fd_residual_slope(omega: f64, c_sr: f64, c_lr: f64) -> (f64, f64) {
        let mol = Molecule::parse_xyz("2\nOH\nO 0 0 0\nH 0 0 0.97\n", 0, 2).unwrap();
        let bs = basis::bundled("sto-3g").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let dfbs = basis::bundled("def2-universal-jkfit").unwrap();
        let dfbs_prep = PreparedBasis::new(&mol, &dfbs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let ctx = ParallelContext::default();
        let budget = ferric_core::memory::resolve_budget_bytes(None);
        let s = oneelectron::overlap(&prep);
        let h = oneelectron::hcore(&prep);
        let x = canonical_orthogonalizer(&s).unwrap();
        let nelec = mol.nelec() as usize;
        let two_s = mol.multiplicity - 1;

        // SAFETY of the lifetimes: both fitters and `dfbs_prep` live for the
        // whole function, so the `RefCell` borrows in `Harness` are valid.
        let sr = RefCell::new(DfK::new(Operator::erfc(omega), &prep, &dfbs_prep, budget).unwrap());
        let lr = RefCell::new(DfK::new(Operator::erf(omega), &prep, &dfbs_prep, budget).unwrap());
        let hn = Harness {
            ctx,
            prep,
            bounds,
            h: h.clone(),
            x,
            nocc_a: (nelec + two_s) / 2,
            nocc_b: (nelec - two_s) / 2,
            sr,
            lr,
            c_sr,
            c_lr,
            omega,
        };
        let n = h.nrows();

        // Plain Roothaan iteration from the hcore guess. No DIIS: this is a
        // tiny system and the FD check only needs a CONVERGED stationary point,
        // which the gradient assertion below proves it reached.
        let mut c_a = hn.diag(&h);
        let mut c_b = c_a.clone();
        for _ in 0..200 {
            let d_a = Harness::density(&c_a, hn.nocc_a);
            let d_b = Harness::density(&c_b, hn.nocc_b);
            let (f_a, f_b) = hn.fock(&d_a, &d_b);
            c_a = hn.diag(&f_a);
            c_b = hn.diag(&f_b);
        }
        let (g_a, g_b) = hn.gradient(&c_a, &c_b);
        let gmax = g_a
            .iter()
            .chain(g_b.iter())
            .fold(0.0f64, |m, v| m.max(v.abs()));
        assert!(
            gmax < 1e-7,
            "SR/LR HF (omega = {omega}, c_SR = {c_sr}, c_LR = {c_lr}) did not reach a \
             stationary point: max|g_ai| = {gmax:.3e}. A finite-difference Hessian check \
             at a non-stationary point measures nothing."
        );

        let d_a = Harness::density(&c_a, hn.nocc_a);
        let d_b = Harness::density(&c_b, hn.nocc_b);
        let (f_a, f_b) = hn.fock(&d_a, &d_b);
        let f_a_mo = c_a.t().dot(&f_a).dot(&c_a);
        let f_b_mo = c_b.t().dot(&f_b).dot(&c_b);

        let mut rng = Xorshift64(0x243F6A8885A308D3);
        let k_a =
            Array2::<f64>::from_shape_fn((n - hn.nocc_a, hn.nocc_a), |_| 0.01 * rng.next_f64());
        let k_b =
            Array2::<f64>::from_shape_fn((n - hn.nocc_b, hn.nocc_b), |_| 0.01 * rng.next_f64());

        let rsh = RshResponse::new(&hn.sr, &hn.lr, c_sr, c_lr, omega);
        let inputs = UhfNewtonInputs {
            prep: &hn.prep,
            bounds: &hn.bounds,
            c_a: &c_a,
            c_b: &c_b,
            f_a_mo: &f_a_mo,
            f_b_mo: &f_b_mo,
            nocc_a: hn.nocc_a,
            nocc_b: hn.nocc_b,
            // Deliberately a VALUE THAT WOULD BE WRONG if it were read: the
            // RSH arm must ignore `k_mix_sr` entirely. If a future edit let the
            // Coulomb arm run for omega != 0, this 0.0 makes the FD check fail
            // loudly rather than agreeing by coincidence.
            k_mix_sr: 0.0,
            rsh: Some(&rsh),
            fxc: None,
            thresh: 1e-14,
            ooc_budget: ferric_core::memory::resolve_budget_bytes(None),
        };
        let pool = crate::engine_pool::EnginePool::new(
            hn.bounds.op,
            &hn.prep,
            ferric_integrals::engine_pool::eri_precision(),
        )
        .unwrap();
        let (ha, hb) = hessian_matvec(&hn.ctx, &inputs, &k_a, &k_b, &pool).unwrap();

        let mut out = Vec::new();
        for eps in [4e-3_f64, 1e-3_f64] {
            let (gp_a, gp_b) = hn.gradient(
                &rotate(&c_a, &k_a, hn.nocc_a, eps),
                &rotate(&c_b, &k_b, hn.nocc_b, eps),
            );
            let (gm_a, gm_b) = hn.gradient(
                &rotate(&c_a, &k_a, hn.nocc_a, -eps),
                &rotate(&c_b, &k_b, hn.nocc_b, -eps),
            );
            let fd_a = (&gp_a - &gm_a) / (2.0 * eps);
            let fd_b = (&gp_b - &gm_b) / (2.0 * eps);
            let dev = ha
                .iter()
                .zip(fd_a.iter())
                .chain(hb.iter().zip(fd_b.iter()))
                .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
            let scale = ha
                .iter()
                .chain(hb.iter())
                .fold(0.0f64, |m, v| m.max(v.abs()));
            eprintln!(
                "  omega {omega} c_SR {c_sr} c_LR {c_lr}  eps {eps:.1e}: \
                 max|H_an - H_fd| = {dev:.3e}  (rel {:.3e}, scale {scale:.3e})",
                dev / scale
            );
            assert!(
                scale > 1e-6,
                "the matvec returned a ~zero block; test is vacuous"
            );
            out.push((dev, dev / scale));
        }
        let ratio = out[0].0 / out[1].0.max(f64::MIN_POSITIVE);
        eprintln!("  residual shrank {ratio:.1}x for a 4x smaller step");
        (ratio, out[1].1)
    }

    /// **Analytic-vs-finite-difference at ω ≠ 0, with c_SR = c_LR = 1.**
    ///
    /// `erfc(ωr)/r + erf(ωr)/r ≡ 1/r` for any ω, so the converged state here is
    /// ordinary UHF and the SR/LR response must reproduce the full exchange
    /// response. The two-step residual slope is the discriminator: a correct
    /// operator's residual falls as `O(eps²)`; a wrong one (c_LR dropped,
    /// `k_mix.sr` used for both, or a direct four-centre response differentiating
    /// a density-fitted Fock) leaves a FLOOR that the step does not move.
    ///
    /// Note what this case CANNOT see: with c_SR = c_LR, swapping erf↔erfc
    /// changes nothing by symmetry. That artifact is covered by
    /// `fd_holds_when_sr_and_lr_coefficients_differ` below, which uses
    /// c_SR ≠ c_LR so the two kernels are no longer interchangeable.
    #[test]
    fn fd_holds_at_nonzero_omega_with_equal_coefficients() {
        let (ratio, rel) = fd_residual_slope(0.40, 1.0, 1.0);
        assert!(
            ratio > 3.0,
            "FD residual is a FLOOR, not a step error (only {ratio:.1}x for a 4x step \
             reduction) — the signature of a wrong exchange operator in the response"
        );
        assert!(
            rel < 1e-6,
            "analytic-vs-FD relative residual {rel:.3e} above the bar"
        );
    }

    /// The same check with **c_SR ≠ c_LR**, where erf and erfc are no longer
    /// interchangeable — so this is the case that an erf↔erfc swap breaks.
    #[test]
    fn fd_holds_when_sr_and_lr_coefficients_differ() {
        let (ratio, rel) = fd_residual_slope(0.40, 1.0, 0.3);
        assert!(
            ratio > 3.0,
            "c_SR != c_LR: FD residual is a floor, not a step error ({ratio:.1}x for 4x)"
        );
        assert!(
            rel < 1e-6,
            "c_SR != c_LR: analytic-vs-FD residual {rel:.3e} above the bar"
        );
    }
}
