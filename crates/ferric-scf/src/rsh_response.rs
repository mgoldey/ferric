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

    // -----------------------------------------------------------------------
    // THE DECISIVE MEASUREMENT: the N2+ lambda_min sign flip across the onset.
    // -----------------------------------------------------------------------

    /// Workspace root (walk up from the CWD; `CARGO_MANIFEST_DIR` is baked in
    /// at compile time and wrong inside a nextest archive).
    fn workspace_root() -> std::path::PathBuf {
        use std::path::{Path, PathBuf};
        let looks_like_root = |p: &Path| {
            p.join("Cargo.toml").is_file()
                && p.join("testdata").is_dir()
                && p.join("crates").is_dir()
        };
        if let Ok(cwd) = std::env::current_dir() {
            let mut here: Option<&Path> = Some(cwd.as_path());
            while let Some(p) = here {
                if looks_like_root(p) {
                    return p.to_path_buf();
                }
                here = p.parent();
            }
        }
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|p| p.parent())
            .expect("manifest dir should be <root>/crates/ferric-scf")
            .to_path_buf()
    }

    /// Converge the symmetric N2+ wB97X-V UKS state at `omega` and return
    /// `(lambda_min, Davidson residual, converged, SCF energy)`.
    ///
    /// SCF recipe verbatim from `tests/validation_rsh_omega.rs` — exact J for
    /// both states, DF-K `def2-universal-jkfit`, default (75,110) Becke grid —
    /// which is the recipe `scripts/validation/gen_rsh_omega.py` matches in
    /// PySCF.
    fn n2_cation_lambda_min(omega: f64) -> (f64, f64, bool, f64) {
        use crate::stability::{uhf_internal_stability, StabilityConfig};
        use crate::uhf::solve_uhf;

        // Same geometry as testdata/molecules/validation/n2.xyz (r_e = 1.0977 A).
        let mol = Molecule::parse_xyz("2\nN2+ 2Sg+\nN 0 0 0\nN 0 0 1.09770000\n", 1, 2).unwrap();
        let bs = basis::bundled("def2-svp").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let ctx = ParallelContext::default();
        let cfg = crate::rhf::RhfConfig {
            xc: Some("wB97X-V".into()),
            xc_omega: Some(omega),
            df_j_aux: Some(String::new()),
            df_k_aux: Some("def2-universal-jkfit".into()),
            energy_conv: 1e-10,
            density_conv: 1e-7,
            max_iter: 500,
            ..Default::default()
        };
        let res = solve_uhf(&ctx, &mol, &prep, &bounds, &cfg).unwrap();
        assert!(
            res.converged,
            "N2+ wB97X-V at omega = {omega} did not converge ({} iters)",
            res.iterations
        );

        let c_a = res.mos_alpha.clone();
        let c_b = res.mos_beta.clone().unwrap();
        let f_a_mo = c_a.t().dot(&res.fock_alpha).dot(&c_a);
        let f_b_mo = c_b.t().dot(res.fock_beta.as_ref().unwrap()).dot(&c_b);
        let nelec = mol.nelec() as usize;
        let two_s = mol.multiplicity - 1;
        let nocc_a = (nelec + two_s) / 2;
        let nocc_b = (nelec - two_s) / 2;

        // The SAME SR/LR kernel the converged Fock used. Rebuilt from the
        // identical (operator, obs, aux, budget) arguments rather than handed
        // back by `solve_uhf`, so the dressed B[P,mu,nu] is the same tensor.
        let xc_def = ferric_dft::libxc::xc_def_from_name_nspin_omega("wB97X-V", 2, omega)
            .expect("wB97X-V at the overridden omega");
        let k_mix = ferric_dft::libxc::k_mix_from_xc_def(&xc_def);
        assert!(
            k_mix.omega > 0.0,
            "the omega override did not reach the functional: k_mix.omega = {}",
            k_mix.omega
        );
        let dfbs = basis::bundled("def2-universal-jkfit").unwrap();
        let dfbs_prep = PreparedBasis::new(&mol, &dfbs).unwrap();
        let budget = ferric_core::memory::resolve_budget_bytes(None);
        let sr =
            RefCell::new(DfK::new(Operator::erfc(k_mix.omega), &prep, &dfbs_prep, budget).unwrap());
        let lr =
            RefCell::new(DfK::new(Operator::erf(k_mix.omega), &prep, &dfbs_prep, budget).unwrap());
        let rsh = RshResponse::new(&sr, &lr, k_mix.sr, k_mix.lr, k_mix.omega);

        // The f_xc response kernel at the converged densities, exactly as
        // `uhf::stability_uhf` builds it. VV10 is NOT in it, and PySCF's probe
        // omits it too — see the test's doc comment.
        let d_a = {
            let o = c_a.slice(ndarray::s![.., ..nocc_a]);
            o.dot(&o.t())
        };
        let d_b = {
            let o = c_b.slice(ndarray::s![.., ..nocc_b]);
            o.dot(&o.t())
        };
        let grid = cfg.dft_grid.clone().unwrap_or_default();
        let fxc_store = crate::rohf::FxcKernelStore::build(
            &mol,
            &prep,
            &grid,
            "wB97X-V",
            cfg.xc_omega,
            &d_a,
            &d_b,
        )
        .expect("wB97X-V f_xc kernel (GGA family)");
        let fxc_storage = fxc_store.response();

        let inputs = UhfNewtonInputs {
            prep: &prep,
            bounds: &bounds,
            c_a: &c_a,
            c_b: &c_b,
            f_a_mo: &f_a_mo,
            f_b_mo: &f_b_mo,
            nocc_a,
            nocc_b,
            // Not read on the RSH arm; a value that would be WRONG if it were.
            k_mix_sr: 0.0,
            rsh: Some(&rsh),
            fxc: Some(&*fxc_storage),
            thresh: cfg.integral_thresh,
            ooc_budget: budget,
        };
        let st = uhf_internal_stability(&ctx, &inputs, &StabilityConfig::default())
            .expect("UHF internal stability eigensolve");
        (st.lowest_eigenvalue, st.residual, st.converged, res.energy)
    }

    /// **THE DECISIVE TEST.** N2+ / def2-SVP / wB97X-V: lambda_min of the UKS
    /// orbital Hessian must be POSITIVE at omega = 0.53 and NEGATIVE at
    /// omega = 0.56, matching PySCF's `symmetric_cation_probe` in sign and
    /// roughly in magnitude. The reference values (+1.129e-4 and -2.712e-3) are
    /// READ from `testdata/reference/validation/rsh_omega/n2_def2-svp.json`,
    /// not hardcoded here.
    ///
    /// This is the negative control #288 could not satisfy and #313 could not
    /// compute: J and <S^2> are perfectly smooth through the onset (ferric
    /// matches PySCF's J there to 7.4e-8 Ha), so only a curvature can see it.
    ///
    /// # Scope: what this comparison does and does NOT establish
    ///
    /// `RhfConfig::check_stability` REFUSES wB97X-V, via
    /// [`crate::stability::StabilitySkip::Vv10Kernel`]: the functional carries
    /// VV10 nonlocal correlation and no VV10 response kernel exists anywhere in
    /// this workspace. That refusal is correct and is asserted by
    /// `tests/rsh_orbital_hessian.rs::wb97xv_is_still_refused_but_for_the_vv10_reason`.
    /// This test drives `uhf_internal_stability` directly, accepting the same
    /// omission PySCF's probe makes — its own provenance records
    /// "PySCF KS response omits the VV10 kernel" — so the comparison is
    /// like-for-like and isolates exactly what #314 changed: the exchange
    /// kernel. It does NOT establish that this lambda_min is the complete
    /// second derivative of wB97X-V's energy. It is not, on either side.
    ///
    /// # Reachability (artifact hypothesis A6)
    ///
    /// Asserting only `sign(lambda_0.53) != sign(lambda_0.56)` would be
    /// arithmetic if the values did not actually straddle zero. Each side is
    /// asserted separately with its own margin, the reference is first checked
    /// to straddle zero itself, and the Davidson `converged` flag is required
    /// (artifact A5: an iteration-starved eigensolve returning a wrong lambda
    /// would otherwise look like success).
    #[test]
    #[ignore = "validation: RSH orbital Hessian (two RSH UKS SCFs + two Davidson eigensolves)"]
    fn n2_cation_lambda_min_flips_sign_across_the_onset() {
        let path =
            workspace_root().join("testdata/reference/validation/rsh_omega/n2_def2-svp.json");
        let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
            panic!(
                "missing validation reference {} ({e}); regenerate with \
                 scripts/validation/gen_rsh_omega.py — a missing reference is a failure, \
                 never a skip",
                path.display()
            )
        });
        let r: serde_json::Value = serde_json::from_str(&text)
            .unwrap_or_else(|e| panic!("{}: bad JSON: {e}", path.display()));
        let probe = r["symmetric_cation_probe"]
            .as_array()
            .expect("reference symmetric_cation_probe");
        assert_eq!(
            probe.len(),
            2,
            "the reference probe should carry exactly the two bracketing omegas"
        );

        let mut measured: Vec<(f64, f64, f64, f64, bool)> = Vec::new();
        for p in probe {
            let omega = p["omega"].as_f64().expect("probe omega");
            let lam_ref = p["cation_lambda_min"].as_f64().expect("probe lambda_min");
            let j_ref = p["j_on_this_branch"].as_f64().expect("probe j");
            let (lam, resid, conv, e) = n2_cation_lambda_min(omega);
            eprintln!(
                "omega {omega:.2}:  lambda_min ferric {lam:+.6e}  PySCF {lam_ref:+.6e}  \
                 E_cation {e:.9}  (Davidson resid {resid:.2e}, converged {conv}; \
                 reference J on this branch {j_ref:+.6e})"
            );
            measured.push((omega, lam, lam_ref, resid, conv));
        }

        // A5.
        for (omega, _, _, resid, conv) in &measured {
            assert!(
                *conv,
                "omega {omega}: the Davidson eigensolve did NOT converge (residual \
                 {resid:.3e}); an unconverged lambda_min is not a stability verdict"
            );
        }

        let (w_lo, lam_lo, ref_lo, _, _) = measured[0];
        let (w_hi, lam_hi, ref_hi, _, _) = measured[1];
        assert!(w_lo < w_hi, "reference probe omegas are not ordered");
        // The PREMISE on the reference side: if PySCF's own values did not
        // straddle zero this test could not discriminate at all.
        assert!(
            ref_lo > 0.0 && ref_hi < 0.0,
            "the REFERENCE itself does not straddle zero ({ref_lo:+.3e}, {ref_hi:+.3e}); \
             this test cannot discriminate and must be redesigned, not relaxed"
        );
        // The PREMISE on ferric's side, each with its own margin (A6).
        let margin_lo = 0.1 * ref_lo.abs();
        let margin_hi = 0.1 * ref_hi.abs();
        assert!(
            lam_lo > margin_lo,
            "omega {w_lo}: lambda_min = {lam_lo:+.6e} is not clearly POSITIVE (margin \
             {margin_lo:.3e}); PySCF says {ref_lo:+.6e}. Without this side the sign-flip \
             claim is arithmetic, not measurement."
        );
        assert!(
            lam_hi < -margin_hi,
            "omega {w_hi}: lambda_min = {lam_hi:+.6e} is not clearly NEGATIVE (margin \
             {margin_hi:.3e}); PySCF says {ref_hi:+.6e}"
        );
        // The CONCLUSION: roughly the right magnitude, not only the right sign.
        // A factor bar, not an absolute one: ferric's own DF fitting error on a
        // comparable quantity is 1.7e-3 RELATIVE (see
        // `tests/rsh_orbital_hessian.rs::erfc_plus_erf_response_reproduces_the_coulomb_response`),
        // and the two codes differ in DF-K aux handling and grid details.
        for (omega, lam, lam_ref, _, _) in &measured {
            let ratio = lam.abs() / lam_ref.abs();
            eprintln!("omega {omega:.2}: |ferric| / |PySCF| = {ratio:.3}");
            assert!(
                (0.2..=5.0).contains(&ratio),
                "omega {omega}: lambda_min magnitude {lam:+.6e} is not within a factor 5 \
                 of PySCF's {lam_ref:+.6e} (ratio {ratio:.3}). The SIGN may be right, but \
                 'roughly in magnitude' is part of the acceptance and this misses it."
            );
        }
    }

    /// SCRATCH diagnostic (temporary): lambda_min across a wider omega range,
    /// to see where ferric's curve sits relative to PySCF's two points.
    #[test]
    #[ignore = "scratch diagnostic"]
    fn scratch_lambda_min_sweep() {
        for w in [0.20_f64, 0.30, 0.40, 0.50, 0.53, 0.56, 0.60] {
            let (lam, resid, conv, e) = n2_cation_lambda_min(w);
            eprintln!("SWEEP omega {w:.2}  lambda_min {lam:+.6e}  E {e:.9}  resid {resid:.2e} conv {conv}");
        }
    }

    /// SCRATCH: is the lambda_min(omega) slope discrepancy the VV10 response
    /// omission? Compare wB97X-V (VV10, response incomplete) against HSE06
    /// (range-separated, NO VV10, so ferric's Hessian IS complete for it).
    #[test]
    #[ignore = "scratch diagnostic"]
    fn scratch_vv10_vs_not() {
        for name in ["wB97X-V", "HSE06"] {
            let def = ferric_dft::libxc::xc_def_from_name(name).unwrap();
            let km = ferric_dft::libxc::k_mix_from_xc_def(&def);
            eprintln!(
                "FUNC {name}: vv10 {:?}  k_mix sr {} lr {} omega {}",
                def.vv10.is_some(),
                km.sr,
                km.lr,
                km.omega
            );
        }
    }

    /// SCRATCH: the independent control. O2/def2-SVP lambda_min vs PySCF for
    /// PBE (omega=0), B3LYP (omega=0) and wB97X-V (omega!=0). The omega=0 rows
    /// are unaffected by #314 and calibrate ferric-vs-PySCF agreement for this
    /// quantity; the wB97X-V row is the one #314 changed.
    #[test]
    #[ignore = "scratch diagnostic"]
    fn scratch_o2_lambda_min_control() {
        use crate::stability::{uhf_internal_stability, StabilityConfig};
        use crate::uhf::solve_uhf;
        // testdata/molecules/validation/o2.xyz
        let path = workspace_root().join("testdata/molecules/validation/o2.xyz");
        let mol = Molecule::load_xyz_with_charge(path.to_str().unwrap(), 0, 3).unwrap();
        let bs = basis::bundled("def2-svp").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let ctx = ParallelContext::default();
        let budget = ferric_core::memory::resolve_budget_bytes(None);
        let dfbs = basis::bundled("def2-universal-jkfit").unwrap();
        let dfbs_prep = PreparedBasis::new(&mol, &dfbs).unwrap();

        for (name, lam_ref) in [
            ("PBE", 0.2287647933779071),
            ("B3LYP", 0.1847715331776547),
            ("wB97X-V", 0.18059386581648604),
        ] {
            let cfg = crate::rhf::RhfConfig {
                xc: Some(name.into()),
                df_j_aux: Some(String::new()),
                df_k_aux: Some("def2-universal-jkfit".into()),
                energy_conv: 1e-10,
                density_conv: 1e-7,
                max_iter: 500,
                ..Default::default()
            };
            let res = solve_uhf(&ctx, &mol, &prep, &bounds, &cfg).unwrap();
            let c_a = res.mos_alpha.clone();
            let c_b = res.mos_beta.clone().unwrap();
            let f_a_mo = c_a.t().dot(&res.fock_alpha).dot(&c_a);
            let f_b_mo = c_b.t().dot(res.fock_beta.as_ref().unwrap()).dot(&c_b);
            let nelec = mol.nelec() as usize;
            let two_s = mol.multiplicity - 1;
            let nocc_a = (nelec + two_s) / 2;
            let nocc_b = (nelec - two_s) / 2;
            let d_a = {
                let o = c_a.slice(ndarray::s![.., ..nocc_a]);
                o.dot(&o.t())
            };
            let d_b = {
                let o = c_b.slice(ndarray::s![.., ..nocc_b]);
                o.dot(&o.t())
            };
            let grid = cfg.dft_grid.clone().unwrap_or_default();
            let store = crate::rohf::FxcKernelStore::build(
                &mol,
                &prep,
                &grid,
                name,
                cfg.xc_omega,
                &d_a,
                &d_b,
            )
            .unwrap();
            let fxc = store.response();
            let def = ferric_dft::libxc::xc_def_from_name_nspin(name, 2).unwrap();
            let km = ferric_dft::libxc::k_mix_from_xc_def(&def);
            let (sr, lr);
            let rsh_opt = if km.omega > 0.0 {
                sr = RefCell::new(
                    DfK::new(Operator::erfc(km.omega), &prep, &dfbs_prep, budget).unwrap(),
                );
                lr = RefCell::new(
                    DfK::new(Operator::erf(km.omega), &prep, &dfbs_prep, budget).unwrap(),
                );
                Some(RshResponse::new(&sr, &lr, km.sr, km.lr, km.omega))
            } else {
                None
            };
            let inputs = UhfNewtonInputs {
                prep: &prep,
                bounds: &bounds,
                c_a: &c_a,
                c_b: &c_b,
                f_a_mo: &f_a_mo,
                f_b_mo: &f_b_mo,
                nocc_a,
                nocc_b,
                k_mix_sr: km.sr,
                rsh: rsh_opt.as_ref(),
                fxc: Some(&*fxc),
                thresh: cfg.integral_thresh,
                ooc_budget: budget,
            };
            let st = uhf_internal_stability(&ctx, &inputs, &StabilityConfig::default()).unwrap();
            eprintln!(
                "CTRL O2 {name:<8} omega {:<5} lambda ferric {:+.8e}  PySCF {lam_ref:+.8e}                   ratio {:.4}  E {:.9} (resid {:.1e} conv {})",
                km.omega, st.lowest_eigenvalue, st.lowest_eigenvalue / lam_ref,
                res.energy, st.residual, st.converged
            );
        }
    }
}
